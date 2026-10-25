"""Known failures on `main`, marked expected to fail so that `check` gates every other test of this suite.

Each one failed on `main` a8e72c81 (with #331's dead-module fix) with torch 2.14 and 2.4.1 on CPU, before this suite joined
`check`; the reason says what it found. They are not strict: the check machines' torch and numpy differ from the lock's,
and a strict mark that passed there would fail `check`. Fix the test or the code, then delete its entry.

Tests that take over about 5 s carry `@pytest.mark.slow` in their own module. The random Boolean-vs-word sweeps carry
`boolean_sweep` and skip unless `VERITY_VLLM_BOOLEAN_SWEEPS=1`: circuit-check's `word_view` checks each Boolean Definition against
its word Definition on every merge, and each file's special-word and edge cases stay in the default run.
"""
from __future__ import annotations

import os

import pytest

#: the opt-in random Boolean-vs-word sweeps: about 19 minutes of test time (5 minutes on 12 workers), on top of the suite's own
SWEEPS = os.environ.get("VERITY_VLLM_BOOLEAN_SWEEPS") == "1"

KNOWN_FAILURES = {
    "tests/acquire/test_compiled_source.py::test_renumber_assigns_invocations_per_call_site":
        "CPU torch: `Tried to instantiate dummy base class _cuda_isCurrentStreamCapturing` (owner vllm-epoch-prep)",
    "tests/program/test_analytic.py::test_check_cos_sin_against_the_captured_b0_table":
        "torch CPU cos/sin against the captured b0 table: max ulp 29,154, the bound 2",
    "tests/engine/test_gen_dense2.py::test_inv_freq_models_cuda_reciprocal_multiply_and_is_identity_for_power_of_two_D":
        "15 inv_freq elements differ at D=96; the model expects 14",
    "tests/program/test_ref_prims.py::test_gelu_ref_vs_torch[none-bf16]":
        "exact GELU against torch CPU: max ulp 128, normalised error inf",
    "tests/program/test_ref_prims.py::test_gelu_ref_vs_torch[none-f32]":
        "exact GELU against torch CPU: max ulp 875,646,710, normalised error nan",
}


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


_PROBE = os.environ.get("VLLM_MEM_PROBE")
_probe_f = []


def _probe_write(rec):
    import json
    if not _probe_f:
        w = os.environ.get("PYTEST_XDIST_WORKER", "main")
        _probe_f.append(open(os.path.join(_PROBE, f"{w}-{os.getpid()}.jsonl"), "a", buffering=1))
    _probe_f[0].write(json.dumps(rec) + "\n")


def _probe_rss():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


if _PROBE:
    import resource
    import time

    def pytest_collection_finish(session):
        _probe_write({"ev": "collected", "t": time.time(), "n": len(session.items), "rss": _probe_rss(),
                      "maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(item, nextitem):
        t0, r0 = time.time(), _probe_rss()
        m0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        c0 = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        yield
        _probe_write({"ev": "test", "id": item.nodeid, "t0": t0, "t1": time.time(), "rss0": r0, "rss1": _probe_rss(), "max0": m0,
                      "max1": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      "cmax0": c0, "cmax1": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024})


@pytest.fixture(autouse=True)
def _sweep_dir_in_tmp(tmp_path_factory, monkeypatch):
    """The row driver's sweep directory default (`cli.MACHINE`'s /workspace/cp/sweep) exists on a GPU node, so a test that leaves
    SWEEP_DIR unset would write rows there (VLLM-21); every test gets its own instead, which a test can still override."""
    monkeypatch.setenv("SWEEP_DIR", str(tmp_path_factory.mktemp("sweep")))

