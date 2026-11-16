"""Suite-wide isolation guard: no test may leave a loaded `verity` module swapped for another copy.

A re-executed `verity.circuits.defs` is a second module with its own empty REGISTRY; every registry module first imported against it registers
into that orphan for the rest of the process, so the failure surfaces in whichever later file shares the worker, never in the test that
caused it.  The guard fails the causing test by name and puts the original modules back.  It looks once the test's `monkeypatch`
has undone what was set with it, so a module swapped with `monkeypatch` is not one the test left: the `monkeypatch` a test or a
fixture of this suite is handed is this file's, which is pytest's set up after the guard.

Every process of the suite runs under the engine profile's fixed Inductor switches (`engine.env.FIXED`), as an engine's process
does: torch reads them once, when `torch._inductor.config` is imported, so this file pins them and imports that module itself,
before any test.  No test is then the first to import it, whatever that test has cleared from the environment, and a test that
builds engine arguments (`engine.env.require_inductor`) doesn't depend on which test ran before it.  A second guard fails, by
name, a test that leaves the loaded module reading a fixed switch otherwise, and puts the pinned value back; it looks once the
test's teardown has finalized every fixture, so a change the test's `monkeypatch` undoes is not one it left."""
from __future__ import annotations

import importlib.util
import sys

import pytest

from tests.lint._imports import core as _core
from verity_vllm.engine import env as _engine_env

_engine_env.pin({k: _engine_env.ENGINE[k] for k in _engine_env.FIXED})
if importlib.util.find_spec("torch") is not None:  # torch is an extra of this package: where it is absent, nothing reads the switches
    import torch._inductor.config  # noqa: F401

import verity.circuits.defs  # noqa: E402,F401  (loaded before the first test, so the guard's snapshot always holds it)

_MISSING = object()


def _verity_modules() -> dict[str, tuple[object, object, object]]:
    """{name: (module, parent package, the parent's attribute of that name)} -- the attribute is snapshotted, not assumed to be the
    module: a package may re-export a function under its submodule's name (`verity.circuits.query`)."""
    snap = {}
    for n, m in list(sys.modules.items()):
        if m is None or not _core(n):
            continue
        parent, _, child = n.rpartition(".")
        pkg = sys.modules.get(parent) if parent else None
        snap[n] = (m, pkg, _MISSING if pkg is None else getattr(pkg, child, _MISSING))
    return snap


@pytest.fixture(autouse=True)
def _verity_modules_not_swapped():
    before = _verity_modules()
    yield
    swapped = []
    for n, (m, pkg, attr) in before.items():
        child = n.rpartition(".")[2]
        if sys.modules.get(n) is not m or (pkg is not None and getattr(pkg, child, _MISSING) is not attr):
            swapped.append(n)
            sys.modules[n] = m
            if pkg is not None and attr is not _MISSING:
                setattr(pkg, child, attr)
    if swapped:
        pytest.fail(f"test left loaded verity module(s) swapped for another copy (sys.modules or package attribute): {sorted(swapped)}; "
                    f"a re-executed verity.circuits.defs carries its own REGISTRY and orphans every registry module imported against it",
                    pytrace=False)


@pytest.fixture
def monkeypatch(_verity_modules_not_swapped, monkeypatch):
    """pytest's `monkeypatch`, set up after the swap guard whoever asks for it first, and so torn down before the guard looks.
    Under pytest's own, a plugin whose autouse fixture takes `monkeypatch` (the one `suites.py` loads into every suite) has it set
    up before the guard, and the guard fails a test for a swap its `monkeypatch` was about to undo."""
    return monkeypatch


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item):
    """The Inductor guard, after the test's every fixture is finalized (the hook's own work), not as a fixture among them: an
    autouse fixture is torn down before `monkeypatch` undoes the test's change wherever `monkeypatch` was set up first, as under a
    plugin whose own autouse fixture takes it, and would read what the test was about to put back."""
    try:
        result = yield
    finally:
        cfg, pinned = sys.modules.get("torch._inductor.config"), dict(_engine_env.FIXED.values())
        left = {attr: getattr(cfg, attr) for attr, want in pinned.items() if cfg is not None and getattr(cfg, attr, want) != want}
        for attr in left:
            setattr(cfg, attr, pinned[attr])
    if left:
        pytest.fail(f"test left torch._inductor.config reading {left}, where every process of the suite reads {pinned} "
                    f"(engine.env.FIXED): it set the attribute, or unloaded the module and imported it again without the pin in its "
                    f"environment, and every later engine build in this process would be refused (engine.env.require_inductor); undo "
                    f"what the test changed (monkeypatch.setattr, monkeypatch.delitem)", pytrace=False)
    return result


# The conftest of the suite's tests/ before its tables moved there.
"""Known failures on `main`, marked expected to fail so that `check` gates every other test of this suite.

Each one failed on `main` with torch 2.14 and 2.4.1 on CPU, before this suite joined
`check`; the reason says what it found. They are not strict: the check machines' torch and numpy differ from the lock's,
and a strict mark that passed there would fail `check`. Fix the test or the code, then delete its entry.

Tests that take over about 5 s carry `@pytest.mark.slow` in their own module. The random Boolean-vs-word sweeps carry
`boolean_sweep` and skip unless `VERITY_VLLM_BOOLEAN_SWEEPS=1`, which no job of `check` sets: each file's special-word and edge
cases stay in the default run, and circuit-check's `word_view` checks each Boolean Definition against its word Definition off the
merge gate (`off_gate` in `tools/check/policy.toml`: a candidate's check runs it, and its red holds nothing).
"""

import importlib  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

import pytest  # noqa: E402
import verity_torch.pouw  # noqa: E402,F401 - hands PoUW's serving code its tensors and H1's device, as the adapter's import does
from verity.protocols.resource_accounting.compute.pouw.prover.serving import (  # noqa: E402
    use_int8_arm,
)

#: the opt-in random Boolean-vs-word sweeps: about 19 minutes of test time (5 minutes on 12 workers), on top of the suite's own
SWEEPS = os.environ.get("VERITY_VLLM_BOOLEAN_SWEEPS") == "1"

KNOWN_FAILURES = {
    "program/test_analytic.py::test_check_cos_sin_against_the_captured_b0_table":
        "torch CPU cos/sin against the captured b0 table: max ulp 29,154, the bound 2",
    "engine/test_gen_dense2.py::test_inv_freq_models_cuda_reciprocal_multiply_and_is_identity_for_power_of_two_D":
        "15 inv_freq elements differ at D=96; the model expects 14",
    "program/test_ref_prims.py::test_gelu_ref_vs_torch[none-bf16]":
        "exact GELU against torch CPU: max ulp 128, normalised error inf",
    "program/test_ref_prims.py::test_gelu_ref_vs_torch[none-f32]":
        "exact GELU against torch CPU: max ulp 875,646,710, normalised error nan",
}

#: PoUW's H1 checks the int8 arm's records with the arm's own GEMM and dequantization, as the option hands them in at install
use_int8_arm(lambda: importlib.import_module("verity_vllm.engine.int8_exact"))


def pytest_configure(config) -> None:
    """Make this process's first MKL VML call on one thread, before any test.  torch's CPU `exp`, `log`, `tanh`, `erf` and the
    other VML functions run in the MKL it links statically, whose first-call initialisation races when several threads call at
    once: one intra-op thread then computes its whole chunk at about 2e-4 relative error (the `exp` reference against torch,
    max ULP 1771, in some 6% of fresh processes).  Any VML call initialises all of them.

    And keep oneDNN, which runs torch's CPU bf16 matmuls, off AMX, set before its first primitive reads it.  On a KVM guest
    with AMX (the cloud agent VMs) a preempted bf16 matmul now and then returns one batch entry's rows wrong, single-threaded
    too: 49 of 16,000 `(4, 1, 130) @ (4, 130, 64)` matmuls under 16 busy processes, and none with AMX off
    (`test_eager_attention_head_ref_vs_torch[div-bf16]`, normalised error 779)."""
    os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "AVX512_CORE_BF16")
    try:
        import torch
    except ImportError:
        return
    torch.exp(torch.zeros(1))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items) -> None:
    sweep = pytest.mark.skip(reason="VERITY_VLLM_BOOLEAN_SWEEPS=1: a random Boolean-vs-word sweep (all of them: about 19 minutes "
                                    "of test time)")
    for item in items:
        if reason := KNOWN_FAILURES.get(item.nodeid):
            item.add_marker(pytest.mark.xfail(reason=f"known failure on main: {reason}", strict=False))
        if not SWEEPS and "boolean_sweep" in item.keywords:
            item.add_marker(sweep)


@pytest.fixture(autouse=True)
def _sweep_dir_in_tmp(tmp_path_factory, monkeypatch):
    """`verity-vllm row`'s sweep directory default (`cli.MACHINE`'s /workspace/cp/sweep) exists on a GPU node, so a test that leaves
    SWEEP_DIR unset would write its workload dirs there; every test gets its own instead, which a test can still override."""
    monkeypatch.setenv("SWEEP_DIR", str(tmp_path_factory.mktemp("sweep")))


@pytest.fixture
def code_versions_stood_in(monkeypatch):
    """The code versions the library stamps on what it writes or keys it by (`score.code_version`, `experiment.code_version`,
    `source_identity.code_identity`) fixed, for a test that writes a stamp or a key but never checks it: each runs git, which the
    suite's read guard can't trace, and reads the whole tree."""
    from verity_vllm.pipeline import experiment, source_identity
    from verity_vllm.program.predict import score
    monkeypatch.setattr(score, "code_version", lambda: "0" * 16)
    monkeypatch.setattr(experiment, "code_version", lambda repo: {"source": "git", "commit": "0" * 40, "dirty": False})
    monkeypatch.setattr(source_identity, "code_identity", lambda root: {"sha256": "0" * 64, "files": 1})


@pytest.fixture
def gnu_timeout() -> None:
    """Skips where the machine has no GNU timeout, for a test that starts one. A fixture, not a check at import, so a test file
    that doesn't ask for it starts no program the suite's read guard can't trace."""
    t = shutil.which("timeout")
    if t is None or b"GNU" not in subprocess.run([t, "--version"], capture_output=True).stdout:
        pytest.skip("GNU timeout (coreutils) required -- the pods have it, macOS does not")
