"""attn_bf16 on the CPU: the SASS gate's verdicts, the kernels attn.py launches, and fixture.py's words against the
Definition's reference evaluator on kinds_d64's generators."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import gemm as G

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import attn
import fixture as F
import sass_gate


def _listing(arch: str = "sm_90a", skip: str = "") -> tuple[str, str]:
    """A clean ptxas report and SASS for every kernel the gate wants but ``skip``."""
    ptxas, sass = [], []
    for name, want in sass_gate.WANT.items():
        if name == skip:
            continue
        ptxas.append(f"ptxas info    : Compiling entry function '{name}' for '{arch}'\n"
                     f"ptxas info    : Function properties for {name}\n"
                     "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
                     "ptxas info    : Used 40 registers\n")
        sass.append(f"        Function : {name}\n        /*0090*/                   HFMA2.MMA R4, -RZ, RZ, 0, 0 ;\n"
                    "        /*00a0*/                   FSEL R5, R5, RZ, !P0 ;            /* 0x000000ff05057208 */\n"
                    + "        /*0100*/                   HMMA.16816.F32.BF16 R4, R8, R12, R4 ;\n" * (want or 1)
                    + "        /*0200*/                   F2FP.BF16.F32.PACK_AB R3, R5, R4 ;\n")
    return "".join(ptxas), "".join(sass)


def test_the_gate_wants_every_kernel_attn_loads():
    assert set(sass_gate.WANT) == set(attn.KERNELS)


@pytest.mark.parametrize("arch", ["sm_90a", "sm_120a"])
def test_gate_passes_a_clean_listing(arch):
    funcs, problems = sass_gate.gate(arch, *_listing(arch))
    assert problems == []
    assert funcs["attn_qk_d64_ref"]["HMMA"] == 1 and funcs["attn_pv_d128_ref"]["F2FP"] == 1
    assert funcs["attn_qk_d128_s16"]["HMMA"] == 32


PTXAS, SASS = _listing()
REF_PV = "        Function : attn_pv_d64_ref\n"


@pytest.mark.parametrize("ptxas, sass, why", [
    (PTXAS, SASS.replace(REF_PV, REF_PV + "        /*0300*/  FFMA R3, R4, R5, R6 ;\n"), "floating-point instructions other than"),
    (PTXAS, SASS.replace(REF_PV, REF_PV + "        /*0300*/  FSEL R3, -R4, R5, P0 ;\n"), "floating-point instructions other than"),
    (PTXAS, SASS.replace(REF_PV, REF_PV + "        /*0300*/  FSEL R3, |R4|, RZ, P0 ;\n"), "floating-point instructions other than"),
    (PTXAS, SASS + "        /*0300*/  FADD R1, R2, R3 ;\n", "floating-point instructions other than"),
    (PTXAS.replace("0 bytes spill stores", "8 bytes spill stores", 1), SASS, "local memory"),
    (PTXAS, SASS.replace("HMMA.16816.F32.BF16", "HMMA.1688.F32.TF32", 1), "MMA forms other than"),
    (PTXAS, SASS.replace(REF_PV, REF_PV + "        /*0300*/  IGMMA.64x8x32.S8.S8 R4, R8 ;\n"), "MMA forms other than"),
    (PTXAS.replace("'sm_90a'", "'sm_120a'", 1), SASS, "compiled for sm_120a"),
    (PTXAS + "ptxas warning : Registers are spilled\n", SASS, "Registers are spilled"),
    (*_listing(skip="attn_pv_d64_ref"), "kernels missing from the cubin: ['attn_pv_d64_ref']"),
    (PTXAS.replace("Used 40 registers", "", 1), SASS, "no ptxas resource line"),
    (PTXAS, SASS.replace("HMMA.16816.F32.BF16 R4, R8, R12, R4 ;\n", "", 1), "HMMA.16816.F32.BF16, want 4"),
])
def test_gate_refuses(ptxas, sass, why):
    _, problems = sass_gate.gate("sm_90a", ptxas, sass)
    assert any(why in p for p in problems), problems


def test_ref_grid_covers_every_column_of_every_kv_head():
    assert attn.grid("ref", 7, 256, 1, cols=33, Hkv=1) == (5, 1, 7)
    assert attn.grid("ref", 3, 16, 9, cols=128, Hkv=2) == (16, 2, 3)
    assert all(attn.serves("ref", g, t, k) for g in (1, 2, 256) for t in (1, 33, 4096) for k in (0, 32, 48))


@pytest.fixture(scope="module")
def kinds():
    inp, D = F.build("kinds_d64")
    return inp, D


def _sequences(inp: dict, idx: list[int]) -> dict:
    return {k: (v[idx] if getattr(v, "ndim", 0) else v) for k, v in inp.items()}


def _pad_definition(x: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    """`GemmCoordinatePad_v1{T=n, K}` by the reference evaluator on the first n words of rows ``x`` and ``w`` (N, K)."""
    spec = bind(G.GemmCoordinatePad, T=n, K=x.shape[1], **G.GEMM_HOPPER)
    return np.asarray(E.evaluate_batch(spec, np.ascontiguousarray(x[:, :n]), np.ascontiguousarray(w[:, :n]),
                                       kernel="reference"), np.uint16)


@pytest.mark.parametrize("kind", ["specials", "tiny", "subacc"])
def test_kind_words_are_the_definitions(kinds, kind):
    """A kind's sequence at KV lengths on and off the 16-grid: the reference sample `fixture.py` holds S and O to, and O as
    GemmCoordinatePad{T=n, K=32} on the first n keys (the H200 circuit's PV)."""
    inp, D = kinds
    i = F.KINDS.index(kind)
    ns = (1, 15, 17, 31)
    sub = _sequences(inp, [32 * i + n - 1 for n in ns])
    sample: list = []
    S, O = F.expected(sub, D, sample, np.random.default_rng(0))
    ref = F.reference_sample(sample)
    assert ref["qk"]["words"] and ref["pv"]["words"] and not ref["qk"]["mismatches"] and not ref["pv"]["mismatches"], ref
    rng = np.random.default_rng(1)
    for j, n in enumerate(ns):
        assert (S[j, :, 0, n:] == F.POISON).all() and (S[j, :, 0, :n] != F.POISON).any()
        h, d = rng.integers(0, 256, 8), rng.integers(0, D, 8)
        x, w = np.zeros((8, 32), np.uint16), np.zeros((8, 32), np.uint16)
        x[:, :n], w[:, :n] = sub["P"][j, h, 0, :n], sub["V"][j, 0, :n, d]  # both (8, n): the indices' axis first
        assert (_pad_definition(x, w, n) == O[j, h, 0, d]).all()


def test_subacc_reaches_the_padding_step_with_subnormal_accumulators():
    """The constructed case: at n <= 16 most of PV's accumulators after the one live step are nonzero FP32 subnormals, the
    words GemmCoordinatePad{K=32}'s all-+0 second step reads; and its words there equal one step's, K = pad16(n)."""
    seq = F.kind_sequence(F.KINDS.index("subacc"))
    P, V = seq["P"], seq["V"]
    for n in (1, 7, 16):
        x = np.zeros((P.shape[0], 32), np.uint16)
        w = np.zeros((V.shape[1], 32), np.uint16)
        x[:, :n], w[:, :n] = P[:, :n], V[:n].T
        x, w = np.repeat(x, w.shape[0], axis=0), np.tile(w, (P.shape[0], 1))
        acc = np.asarray(E.evaluate_batch(bind(G.DotBf16, K=16, **G.GEMM_HOPPER), x[:, :16], w[:, :16]), np.uint32)
        subnormal = (((acc >> 23) & 0xFF) == 0) & ((acc & 0x7FFFFF) != 0)
        assert subnormal.mean() > 0.5, (n, subnormal.mean())
        assert (F.coordinate(x, w) == F.coordinate(x[:, :16], w[:, :16])).all()


@pytest.mark.slow
def test_every_kind_agrees_with_the_reference(kinds):
    """Every kind's first sequence at every KV length 1..32 against the reference sample (about 2 s a kind): what the
    quick kinds above don't reach (cancellations, overflow, uniform words)."""
    inp, D = kinds
    for i, kind in enumerate(F.KINDS):
        sample: list = []
        F.expected(_sequences(inp, list(range(32 * i, 32 * i + 32))), D, sample, np.random.default_rng(i))
        ref = F.reference_sample(sample)
        assert not ref["qk"]["mismatches"] and not ref["pv"]["mismatches"], (kind, ref)
