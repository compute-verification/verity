"""The fused v2 / v3 hint kernel (``hints_fused.py``) against the torch generators it replaces.

* coverage (torch-free): the kernel's emission order names every hint row of every v2 / v3 system (chain and unit mode) once;
* byte identity (``torch.equal``) with ``_hints_eager`` and with the CUDA-graph path, per relation:
  - real sub-batches: the relation's instance set marshalled by the chain runner (what the prover hands the generator);
  - in-domain random units in unit mode (``pubsel.relation_test._cases``: zeros, subnormals, chained states, zero sums);
  - adversarial columns: operand words uniform over the whole word range (NaN, inf, off-domain words included), the
    accumulator uniform over all 2^32 words plus every FP32 class (+-0, subnormals, the extremes, inf, NaN), all-zero and
    all-maximal operands, exact cancellations, exponent-extreme words (``HINTS_FUSED_N`` columns, default 16384, for each of
    ``HINTS_FUSED_SEEDS`` seeds, default 1);
* the Python-integer reference (``reference_hints``) on an in-domain subset;
* per-stream safety: two sub-batches' hints generated at once on two streams equal the sequential ones.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from .relations import RELATIONS

V2 = ["bf16-hopper-v2", "bf16-ampere-v2", "fp8-hopper-v2", "fp8-ada-v2",
      "bf16-hopper-v2x4", "bf16-ampere-v2x4", "fp8-hopper-v2x4", "fp8-ada-v2x4"]
V3 = ["bf16-hopper-v3", "bf16-ampere-v3", "fp8-hopper-v3", "fp8-ada-v3",
      "bf16-hopper-v3x4", "bf16-ampere-v3x4", "fp8-hopper-v3x4", "fp8-ada-v3x4"]
N_ADV = int(os.environ.get("HINTS_FUSED_N", "16384"))
SEEDS_ADV = int(os.environ.get("HINTS_FUSED_SEEDS", "1"))
CACHE = os.environ.get("LIGERO_INSTANCES_CACHE")

FP32_SPECIALS = [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000,
                 0x3F800000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000,
                 0x7FFFFFFF, 0xFFFFFFFF, 0x4B000000, 0x33800000, 0x0C000000, 0x72000000]


def _family(name: str) -> str:
    return "v3" if "-v3" in name else "v2"


def _drop_graphs():
    """The torch generators cache one CUDA graph per ``(id(sys), l, k, stream)`` without holding ``sys``: a test process that
    compiles many systems can see a freed system's id reused and replay a stale graph (a matrix of the wrong height), and
    the graphs' private pools add up to the whole card.  Tests start from empty caches."""
    try:
        import torch
    except ImportError:
        return
    from .privsel import hints as h3
    from .pubsel import hints as h2
    h2._GRAPHS.clear()
    h3._GRAPHS.clear()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def _fresh_graphs():
    _drop_graphs()
    yield
    _drop_graphs()


@pytest.mark.parametrize("chain", [True, False], ids=["chain", "unit"])
@pytest.mark.parametrize("name", V2 + V3)
def test_emission_order_covers_every_hint_row(name, chain):
    from .hints_fused import _order
    from .pubsel.relation import plan
    rel = RELATIONS[name]
    p = rel.params
    pl = plan(p)
    sp = None
    if _family(name) == "v3":
        from .privsel.relation import shift_plan
        sp = shift_plan(p, pl)
    sys = rel.compile(p, chain)
    order = _order(_family(name), p, pl, sp, chain)
    names = [n for _, n in sys.hints]
    assert len(set(order)) == len(order)
    assert set(names) <= set(order), sorted(set(names) - set(order))[:8]
    if _family(name) == "v3":
        assert set(order) == set(names)          # v3 commits every value its generator computes


def _adversarial(rel, n: int, seed: int):
    """(c, a, b) int64 of ``n`` columns, see the module docstring."""
    from .pubsel.relation import plan
    pl = plan(rel.params)
    wb, k = pl.op_word_bits, rel.k
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2**wb, (n, k), dtype=np.int64)
    b = rng.integers(0, 2**wb, (n, k), dtype=np.int64)
    c = rng.integers(0, 2**32, n, dtype=np.int64)
    sign = 1 << (wb - 1)
    emax = (2**pl.op_exp - 1) << pl.op_frac
    blocks = 8
    per = n // blocks
    i0 = 0
    c[i0:i0 + per] = np.resize(np.array(FP32_SPECIALS, dtype=np.int64), per)          # every accumulator class
    i0 += per
    a[i0:i0 + per], b[i0:i0 + per] = 0, 0                                               # all-zero operands
    i0 += per
    a[i0:i0 + per] = rng.integers(0, 2, (per, k)) * sign | (emax - (1 << pl.op_frac)) | (2**pl.op_frac - 1)
    b[i0:i0 + per] = rng.integers(0, 2, (per, k)) * sign | (emax - (1 << pl.op_frac)) | (2**pl.op_frac - 1)
    i0 += per                                                                           # the largest finite words
    w = rng.integers(0, 2**wb, (per, k), dtype=np.int64)
    a[i0:i0 + per, 0::2], a[i0:i0 + per, 1::2] = w[:, 0::2], w[:, 0::2]
    b[i0:i0 + per, 0::2], b[i0:i0 + per, 1::2] = w[:, 1::2], w[:, 1::2] ^ sign        # exact cancellations
    c[i0:i0 + per:2] = 0
    i0 += per
    ex = rng.choice(np.array([0, 1, 2**pl.op_exp - 2, 2**pl.op_exp - 1], dtype=np.int64), (2, per, k))
    fr = rng.integers(0, 2**pl.op_frac, (2, per, k), dtype=np.int64)
    sg = rng.integers(0, 2, (2, per, k), dtype=np.int64)
    a[i0:i0 + per] = sg[0] * sign | ex[0] << pl.op_frac | fr[0]                      # exponent-extreme words
    b[i0:i0 + per] = sg[1] * sign | ex[1] << pl.op_frac | fr[1]
    i0 += per
    ce = rng.choice(np.array([0, 1, 2, 126, 127, 128, 253, 254, 255], dtype=np.int64), per)
    c[i0:i0 + per] = rng.integers(0, 2, per) << 31 | ce << 23 | rng.integers(0, 2**23, per)   # accumulator exponent extremes
    return c, a, b


def _paths(name: str, sys, device="cuda"):
    """(fused, eager, graph) generators for ``sys``."""
    import torch
    rel = RELATIONS[name]
    p = rel.params
    if _family(name) == "v3":
        from .privsel import hints as mod
        entry = mod.hints_v3
    else:
        from .pubsel import hints as mod
        entry = mod.hints_v2
    from .hints_fused import FusedHints
    fh = FusedHints(_family(name), p, sys, device)

    def fused(c, a, b):
        return fh.run(c, a, b)

    def eager(c, a, b):
        return mod._hints_eager(p, sys, c, a, b, None, device)

    def graph(c, a, b):
        old = os.environ.get("LIGERO_FUSED_HINTS")
        os.environ["LIGERO_FUSED_HINTS"] = "0"
        try:
            return entry(p, sys, c, a, b, None, device)
        finally:
            if old is None:
                del os.environ["LIGERO_FUSED_HINTS"]
            else:
                os.environ["LIGERO_FUSED_HINTS"] = old

    return fused, eager, graph, torch


def _cuda_ok() -> bool:
    try:
        import cupy  # noqa: F401
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _diff(h1, h2, sys) -> str:
    bad = (h1 != h2).any(dim=1).nonzero().flatten().tolist()
    names = [n for _, n in sys.hints]
    return f"{len(bad)} hint rows differ, e.g. {[names[i] for i in bad[:6]]}"


@pytest.mark.skipif(not _cuda_ok(), reason="the fused kernel is CUDA (cupy / NVRTC)")
@pytest.mark.parametrize("name", V2 + V3)
def test_fused_equals_torch_adversarial(name):
    rel = RELATIONS[name]
    for chain in (True, False):
        _drop_graphs()
        sys = rel.compile(rel.params, chain)
        fused, eager, graph, torch = _paths(name, sys)
        for s in range(SEEDS_ADV):
            seed = 7 + chain + 2 * s
            c, a, b = (torch.as_tensor(x).cuda() for x in _adversarial(rel, N_ADV, seed=seed))
            hf, he = fused(c, a, b), eager(c, a, b)
            assert torch.equal(hf, he), f"{name} chain={chain} seed={seed} adversarial: fused != eager: {_diff(hf, he, sys)}"
            hg = graph(c, a, b)
            assert torch.equal(hf, hg), f"{name} chain={chain} seed={seed} adversarial: fused != graph: {_diff(hf, hg, sys)}"
            del hf, he, hg


@pytest.mark.skipif(not _cuda_ok(), reason="the fused kernel is CUDA (cupy / NVRTC)")
@pytest.mark.parametrize("name", V2 + V3)
def test_fused_equals_torch_and_reference_in_domain(name):
    from .pubsel.relation_test import _cases
    rel = RELATIONS[name]
    sys = rel.compile(rel.params, False)
    fused, eager, graph, torch = _paths(name, sys)
    c, a, b, _ = _cases(rel, 4096, seed=11)
    ct, at, bt = (torch.as_tensor(np.asarray(x, dtype=np.int64)).cuda() for x in (c, a, b))
    hf = fused(ct, at, bt)
    he = eager(ct, at, bt)
    assert torch.equal(hf, he), f"{name} in-domain: fused != eager: {_diff(hf, he, sys)}"
    n_ref = 96
    units = [(int(c[i]), a[i].tolist(), b[i].tolist(), None) for i in range(n_ref)]
    ref = rel.reference_hints(rel.params, sys, units, "cuda")
    assert torch.equal(hf[:, :n_ref], ref), f"{name} in-domain: fused != Python reference: {_diff(hf[:, :n_ref], ref, sys)}"


@pytest.mark.skipif(not _cuda_ok(), reason="the fused kernel is CUDA (cupy / NVRTC)")
@pytest.mark.parametrize("name", V2 + V3)
def test_fused_equals_torch_real_subbatch(name):
    """The instance set marshalled by the chain runner (chain mode, pad units included), as the prover calls it."""
    from dataclasses import replace
    from . import protocol as protocol_mod
    from .relchain import RelationChainRunner, instances
    rel = RELATIONS[name]
    R = RelationChainRunner(rel, "cuda", -128.0, 2, zk=False, mode=protocol_mod.MODE_INTERACTIVE)
    per = 4096 // rel.steps
    try:
        data = instances(rel, per, procs=4, cache=CACHE)
    except FileNotFoundError:   # the bf16-ampere family reads the frozen vu-k1536 set, built only with BENCH_INSTANCES=1
        data = instances(replace(rel, frozen_tier=None), per, procs=4, cache=CACHE)
    bd = R.marshal(data)
    fused, eager, graph, torch = _paths(name, R.sys)
    args = (bd["c_d"], bd["a_d"], bd["b_d"])
    hf, he, hg = fused(*args), eager(*args), graph(*args)
    assert torch.equal(hf, he), f"{name} real sub-batch: fused != eager: {_diff(hf, he, R.sys)}"
    assert torch.equal(hf, hg), f"{name} real sub-batch: fused != graph: {_diff(hf, hg, R.sys)}"
    ref = rel.reference_hints(R.p, R.sys, bd["units"][:64], "cuda")
    assert torch.equal(hf[:, :64], ref), f"{name} real sub-batch: fused != Python reference"


@pytest.mark.skipif(not _cuda_ok(), reason="the fused kernel is CUDA (cupy / NVRTC)")
@pytest.mark.parametrize("name", ["fp8-ada-v3x4", "fp8-ada-v2x4"])
def test_fused_hints_not_shared_across_streams(name):
    from .pipeline_race_test import hints_concurrent_trials
    assert hints_concurrent_trials(name, 5) == 0
