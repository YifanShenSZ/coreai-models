# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Root test configuration."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from coreai.runtime import NDArray, SpecializationOptions

_current_test_id: str = ""
_dump_optests: bool = False

_COMPUTE_UNIT_KIND_CHOICES = ("interpreter", "cpu", "gpu", "neural_engine")
_COMPUTE_UNIT_KIND_DEFAULT = "interpreter"
_compute_unit_kind: str = _COMPUTE_UNIT_KIND_DEFAULT


@pytest.fixture(autouse=True)
def update_current_test_id(request: pytest.FixtureRequest) -> None:
    """Track the running test id so the runtime layer can name dump dirs."""
    global _current_test_id
    _current_test_id = request.node.nodeid


def get_current_test_id() -> str:
    return _current_test_id


def dump_optests_enabled() -> bool:
    return _dump_optests


def get_test_specialization_options() -> SpecializationOptions | None:
    """Translate ``--compute-unit-kind`` into ``SpecializationOptions`` (or None).

    On non-macOS platforms only ``interpreter`` is supported — the runtime
    does not expose ``SpecializationOptions`` outside Darwin.
    """
    if _compute_unit_kind == "interpreter":
        return None
    if platform.system() != "Darwin":
        msg = (
            f"--compute-unit-kind={_compute_unit_kind} is only supported on macOS; "
            "use --compute-unit-kind=interpreter on this platform."
        )
        raise RuntimeError(msg)
    from coreai.runtime import (  # type: ignore[attr-defined]
        ComputeUnitKind,
        SpecializationOptions,
    )

    if _compute_unit_kind == "cpu":
        return SpecializationOptions.cpu_only()
    if _compute_unit_kind == "gpu":
        return SpecializationOptions.from_preferred_compute_unit_kind(
            compute_unit_kind=ComputeUnitKind.gpu(),
        )
    if _compute_unit_kind == "neural_engine":
        return SpecializationOptions.from_preferred_compute_unit_kind(
            compute_unit_kind=ComputeUnitKind.neural_engine(),
        )
    msg = f"Unknown compute unit kind: {_compute_unit_kind!r}"
    raise ValueError(msg)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI options."""
    parser.addoption(
        "--compute-unit-kind",
        choices=list(_COMPUTE_UNIT_KIND_CHOICES),
        default=_COMPUTE_UNIT_KIND_DEFAULT,
        help=(
            "Compute unit used by validate_numerical_output:\n"
            "  interpreter (default) - bundled runtime (USE_LOCAL_COREAI=1)\n"
            "  cpu                   - SpecializationOptions.cpu_only() (BNNS)\n"
            "  gpu                   - preferred ComputeUnitKind.gpu() (MPSGraph)\n"
            "  neural_engine         - preferred ComputeUnitKind.neural_engine()\n"
            "Anything other than 'interpreter' unsets USE_LOCAL_COREAI so the OS\n"
            "runtime is used."
        ),
    )
    parser.addoption(
        "--dump-optests",
        action="store_true",
        default=False,
        help="Trigger optest dumping",
    )


def pytest_configure(config: pytest.Config) -> None:
    global _dump_optests, _compute_unit_kind
    _dump_optests = config.getoption("--dump-optests")
    _compute_unit_kind = config.getoption("--compute-unit-kind")
    if _compute_unit_kind == "interpreter":
        os.environ.setdefault("USE_LOCAL_COREAI", "1")
    else:
        os.environ.pop("USE_LOCAL_COREAI", None)


def optest_dump_path(test_id: str) -> Path:
    """Map a pytest nodeid to ``op_tests/<test-file-stem-path>/<sanitized-test>``.

    Example: ``python/tests/test_model_units/test_primitives/test_macos/test_rope.py::TestRoPE::test_rope[Llama3RoPE-f32]``
    -> ``op_tests/test_model_units/test_primitives/test_macos/test_rope/TestRoPE_test_rope-params-Llama3RoPE-f32``
    """
    raw = test_id
    for prefix in ("python/tests/", "tests/"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
            break
    dir_part, test = raw.split(".py", maxsplit=1)
    test = test.removeprefix("::").replace("::", "_").replace("[", "-params-").replace("]", "")
    return Path(f"op_tests/{dir_part}") / test


def _add_npz_entry(io_numpy: dict[str, np.ndarray], key: str, arr: np.ndarray) -> None:
    """Add an array to the npz, emitting a ``_dtype_<key>='bf16'`` companion if void16."""
    io_numpy[key] = arr
    if arr.dtype.str == "|V2":
        io_numpy[f"_dtype_{key}"] = np.array("bf16")


def _dump_optest_artifacts(
    coreai_program: Any,
    inputs: dict[str, NDArray],
    rt_outputs: dict[str, NDArray],
    dump_path: Path,
) -> None:
    """Write a `<testname>.aimodel` + `<testname>_test_data.npz` pair.

    Format: aimodel prefix == npz prefix == dump_path
    leaf name; npz holds an ``op_name`` scalar plus ``input_<n>`` /
    ``output_<n>`` keys.
    """
    testname = dump_path.name
    coreai_program.save_asset(dump_path / f"{testname}.aimodel")

    io_numpy: dict[str, np.ndarray] = {"op_name": np.array("main")}
    for name, arr in inputs.items():
        _add_npz_entry(io_numpy, f"input_{name}", arr.numpy())
    for name, arr in rt_outputs.items():
        _add_npz_entry(io_numpy, f"output_{name}", arr.numpy())
    np.savez(dump_path / f"{testname}_test_data.npz", **io_numpy)
