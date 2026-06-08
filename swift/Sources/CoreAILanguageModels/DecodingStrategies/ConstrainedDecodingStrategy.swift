// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Tokenizers

/// Grammar-constrained decoding strategy using xgrammar.
///
/// Conforms to `DecodingStrategy` so it can be used anywhere a decoding strategy is expected.
/// Streams `GenerationResult` tokens as they are generated, enabling incremental text display.
///
/// Uses xgrammar bitmask enforcement to ensure generated text conforms to a JSON schema.
/// Each step: (1) run one inference step to get logits, (2) apply the grammar bitmask
/// to zero out tokens that would violate the JSON schema, (3) sample from the masked
/// logits, (4) accept the token in the grammar matcher to advance the grammar state.
public struct ConstrainedDecodingStrategy: DecodingStrategy {
    /// The JSON schema that constrains generation output.
    private let jsonSchema: String

    /// Vocabulary size override. If nil, derived from tokenizer at generation time.
    private let vocabSizeOverride: Int?

    /// Initialize with a JSON schema string.
    ///
    /// - Parameters:
    ///   - jsonSchema: A valid JSON schema string that constrains output structure
    ///   - vocabSize: Optional vocabulary size override. If nil, derived from tokenizer.
    public init(jsonSchema: String, vocabSize: Int? = nil) {
        self.jsonSchema = jsonSchema
        self.vocabSizeOverride = vocabSize
    }

    // MARK: - DecodingStrategy conformance

    public func decode(
        from input: Input,
        tokenizer: any Tokenizer,
        inferenceEngine: any InferenceEngine,
        samplingConfiguration: SamplingConfiguration,
        options: InferenceOptions,
        stopSequences: StopSequences
    ) -> AsyncThrowingStream<GenerationResult, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    try await runConstrainedGeneration(
                        input: input,
                        tokenizer: tokenizer,
                        inferenceEngine: inferenceEngine,
                        samplingConfiguration: samplingConfiguration,
                        options: options,
                        stopSequences: stopSequences,
                        with: continuation
                    )
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Core constrained generation loop

    private func runConstrainedGeneration(
        input: Input,
        tokenizer: any Tokenizer,
        inferenceEngine: any InferenceEngine,
        samplingConfiguration: SamplingConfiguration,
        options: InferenceOptions,
        stopSequences: StopSequences,
        with continuation: AsyncThrowingStream<GenerationResult, any Error>.Continuation
    ) async throws {
        CLILogger.log("Starting constrained decoding strategy with schema", component: "ConstrainedDecodingStrategy")

        // Setup: session, tokens, options
        var session = try createSession(tokenizer: tokenizer, stopSequences: stopSequences)
        var inputTokens = try PromptUtils.maybeApplyTokenizerChatTemplate(input, tokenizer: tokenizer)
            .map(Int32.init)
        let constrainedOptions = InferenceOptions(maxTokens: 1, includeLogits: true)
        let maxTokens = options.maxTokens ?? 512

        try await inferenceEngine.reset()

        var generatedTokens: [Int32] = []
        var previousDecodedText: String = ""
        var tokenStep: Int = 0

        // Token-by-token generation loop
        for _ in 0..<maxTokens {
            if session.isTerminated { break }

            // Step 1: Get logits → mask → sample → accept
            let (bestToken, logits) = try await generateOneToken(
                inputTokens: inputTokens,
                session: &session,
                inferenceEngine: inferenceEngine,
                samplingConfiguration: samplingConfiguration,
                constrainedOptions: constrainedOptions
            )
            guard let bestToken, let logits else { break }

            if stopSequences.matches(recentTokens: [bestToken]) { break }

            inputTokens.append(bestToken)
            generatedTokens.append(bestToken)
            tokenStep += 1

            // Step 2: Decode to text and yield delta
            let delta = computeTextDelta(
                generatedTokens: generatedTokens,
                previousDecodedText: &previousDecodedText,
                tokenizer: tokenizer,
                tokenStep: tokenStep
            )
            continuation.yield(GenerationResult(text: delta, tokenId: bestToken, rawLogits: logits))

            if session.isTerminated { break }
        }

        continuation.finish()
    }

    // MARK: - Private helpers

    /// Create a constrained generation session with stop token extraction.
    private func createSession(
        tokenizer: any Tokenizer,
        stopSequences: StopSequences
    ) throws -> ConstrainedGenerationSession {
        guard let vocabSize = vocabSizeOverride ?? Self.deriveVocabSize(from: tokenizer) else {
            throw InferenceRuntimeError.invalidArgument(
                "Cannot determine vocabulary size from tokenizer. "
                    + "Pass vocabSize explicitly via CoreAIRunner or LLMAsset metadata."
            )
        }

        let singleTokenStops = stopSequences.sequences.filter { $0.count == 1 }.map { $0[0] }
        if stopSequences.sequences.contains(where: { $0.count > 1 }) {
            CLILogger.log(
                "Warning: Multi-token stop sequences not supported by xgrammar, using single-token stops only",
                component: "ConstrainedDecodingStrategy")
        }
        let stopTokenIds: [Int32]? = singleTokenStops.isEmpty ? nil : singleTokenStops

        let session = try ConstrainedGenerationSession(
            jsonSchema: jsonSchema,
            tokenizer: tokenizer,
            vocabSize: vocabSize,
            stopTokenIds: stopTokenIds
        )
        CLILogger.log(
            "Constrained session created (vocabSize=\(vocabSize), stopTokenIds=\(stopTokenIds ?? []))",
            component: "ConstrainedDecodingStrategy")
        return session
    }

    /// Run one inference step: get logits, apply mask, sample, accept.
    /// Returns `(nil, nil)` if generation should stop.
    private func generateOneToken(
        inputTokens: [Int32],
        session: inout ConstrainedGenerationSession,
        inferenceEngine: any InferenceEngine,
        samplingConfiguration: SamplingConfiguration,
        constrainedOptions: InferenceOptions
    ) async throws -> (Int32?, [LogitsScalarType]?) {
        var rawLogits: [LogitsScalarType]? = nil
        for try await output in try inferenceEngine.generate(
            with: inputTokens,
            samplingConfiguration: samplingConfiguration,
            inferenceOptions: constrainedOptions
        ) {
            rawLogits = output.logits
            break
        }
        guard let logits = rawLogits else {
            throw ConstrainedGenerationError.generationFailed("No logits returned from engine")
        }

        var maskedLogits = logits
        _ = session.applyMask(to: &maskedLogits)

        let bestToken = CompositeSampler.sample(from: &maskedLogits, config: samplingConfiguration)

        if !session.acceptToken(bestToken) {
            return (nil, nil)
        }
        return (bestToken, logits)
    }

    private func computeTextDelta(
        generatedTokens: [Int32],
        previousDecodedText: inout String,
        tokenizer: any Tokenizer,
        tokenStep: Int
    ) -> String {
        let decodeSpan = InstrumentsProfiler.beginDecode(step: tokenStep)
        let fullDecodedText = tokenizer.decode(tokens: generatedTokens.map { Int($0) })
        decodeSpan.end()

        let common = fullDecodedText.commonPrefix(with: previousDecodedText)
        let delta = String(fullDecodedText.dropFirst(common.count))

        if delta.unicodeScalars.contains(where: { $0 == "\u{FFFD}" }) {
            return ""
        }

        previousDecodedText = fullDecodedText
        return delta
    }

    // MARK: - Vocabulary size derivation

    /// Derive vocabulary size from a tokenizer using binary search.
    static func deriveVocabSize(from tokenizer: any Tokenizer) -> Int? {
        var low = 0
        var high = 524_288

        // Binary search for the last valid token ID
        while low < high {
            let mid = (low + high) / 2
            if tokenizer.convertIdToToken(mid) != nil {
                low = mid + 1
            } else {
                high = mid
            }
        }

        if low == 0 {
            CLILogger.log(
                "Warning: Could not determine vocab size from tokenizer — grammar mask may be wrong",
                component: "ConstrainedDecodingStrategy")
            return nil
        }
        return low
    }
}
