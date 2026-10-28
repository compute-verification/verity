"""The reference-mode kinds of `verity/ml/definitions/registry/ref_prims.py`.

Three layers, kept separate (semantics vs evidence):
  1. **definition consistency** — the numpy twins equal the gate-by-gate circuit evaluation of the registered bodies on
     small shapes (the body is the definition; the twin is a testing aid);
  2. **conformance vs torch CPU over a stated, seeded random domain** — agreement statistics (exact-word fraction, max
     ulp distance in the output dtype) are *measured and recorded* to `docs/data/ref-prims/<kind>-<DT>.json`, labelled
     "conformance over tested domain, not universal correctness".  Where torch's reduction order or Sleef transcendental
     differs from the reference definition the test asserts only the recorded bound (max ulp), never bit equality;
     where the operation has a single rounding sequence in both (element-wise adds/multiplies, gathers, argmax, the
     rotate_half RoPE) bit equality is asserted;
  3. **vocabulary pin** — manifest/digest stability and the codec round trip of a reference program.

Run on a pod:  `REF_PRIMS_RECORD_DIR=docs/data/ref-prims PYTHONPATH=. python -m pytest tests/program/test_ref_prims.py -q`
Records are written to `$REF_PRIMS_RECORD_DIR`, and without it to a temporary directory: a plain run leaves the tree alone.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verity.ml.circuits import codec
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, bind
from verity.ml.circuits.evaluation import evaluate_call
from verity.ml.circuits.program import Program
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry import ref_prims as R
from verity.ml.definitions.numerics import ref_prims_twins

torch = pytest.importorskip("torch")
F = torch.nn.functional

REPO = Path(__file__).resolve().parents[2]
RECORD_DIR = Path(os.environ.get("REF_PRIMS_RECORD_DIR") or tempfile.mkdtemp(prefix="ref-prims-"))
SEED = 20260909
LABEL = "conformance over tested domain, not universal correctness"

_ST = CompositeDefinition("Standalone", 1, ("FN",), lambda S: S.FN.signature(), lambda B, S, *a: B.call(S.FN, *a),
                          doc="root wrapper", register=False)


# ---------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------


def gates(spec, *arrays) -> tuple[np.ndarray, int]:
    """Evaluate one call of `spec` on encodings through the reference evaluator; returns (output words, gate count)."""
    prog = Program(bind(_ST, FN=spec))
    circ = prog.circuit
    call = [c for c in circ.root_calls() if not c.is_primitive][0]
    args = [int(v) for a in arrays for v in np.asarray(a).ravel()]
    return np.array(evaluate_call(circ, call, args), dtype=np.uint32), call.size


def t_words(t: torch.Tensor) -> np.ndarray:
    """torch tensor -> encoding words (u16 for bf16, u32 for f32)."""
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16).numpy().astype(np.uint16)
    assert t.dtype == torch.float32
    return t.contiguous().view(torch.int32).numpy().astype(np.uint32)


def words_t(w: np.ndarray, DT: str) -> torch.Tensor:
    if DT == "bf16":
        return torch.from_numpy(np.asarray(w, dtype=np.uint16).astype(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(np.asarray(w, dtype=np.uint32).astype(np.int32)).view(torch.float32)


def tdt(DT: str):
    return torch.bfloat16 if DT == "bf16" else torch.float32


def rand_words(rng, shape, DT: str, scale: float = 1.0) -> np.ndarray:
    x = (rng.standard_normal(shape) * scale).astype(np.float32)
    return ref_prims_twins.np_narrow(x, DT)


def _ordered(w: np.ndarray, DT: str) -> np.ndarray:
    bits = 16 if DT == "bf16" else 32
    w = np.asarray(w).astype(np.int64)
    sign = w >> (bits - 1)
    mag = w & ((1 << (bits - 1)) - 1)
    return np.where(sign == 1, -mag, mag)


def _is_nan(w: np.ndarray, DT: str) -> np.ndarray:
    w = np.asarray(w).astype(np.int64)
    if DT == "bf16":
        return ((w & 0x7F80) == 0x7F80) & ((w & 0x7F) != 0)
    return ((w & 0x7F800000) == 0x7F800000) & ((w & 0x7FFFFF) != 0)


U32 = 2.0 ** -24  # f32 unit roundoff


def ulp_dt(vals: np.ndarray, DT: str) -> np.ndarray:
    """One ulp of the output dtype at each |value| (f32 spacing, scaled by 2^16 for bf16)."""
    sp = np.spacing(np.abs(np.asarray(vals, dtype=np.float32))).astype(np.float64)
    return sp * (2.0 ** 16 if DT == "bf16" else 1.0)


def compare(ref: np.ndarray, got: np.ndarray, DT: str, *, tol: np.ndarray | None = None, tol_formula: str = "") -> dict:
    """Agreement statistics between reference words and oracle words in the output dtype.

    `max_ulp`/`exact_fraction` are always recorded. Ulps of the *output* are the wrong yardstick for a reduction whose
    result nearly cancels (a 1e-3 result of a K=768 dot carries the absolute error of the ~1-sized partial sums), so
    reduction kinds also pass a per-element tolerance `tol` (an a-priori reassociation bound + one output ulp) and record
    `max_normalised_err = max |ref - got| / tol`; the test asserts that <= 1."""
    ref, got = np.asarray(ref).ravel(), np.asarray(got).ravel()
    assert ref.shape == got.shape
    both_nan = _is_nan(ref, DT) & _is_nan(got, DT)
    ulp = np.abs(_ordered(ref, DT) - _ordered(got, DT))
    ulp[both_nan] = 0
    exact = int(np.count_nonzero((ref == got) | both_nan))
    hist = {str(k): int(v) for k, v in zip(*np.unique(np.minimum(ulp, 8), return_counts=True))}
    out = {"words": int(ref.size), "exact_words": exact, "exact_fraction": exact / ref.size, "max_ulp": int(ulp.max()),
           "ulp_histogram_capped_8": hist}
    rv, gv = ref_prims_twins.np_widen(ref, DT).astype(np.float64), ref_prims_twins.np_widen(got, DT).astype(np.float64)
    err = np.abs(rv - gv)
    err[both_nan] = 0.0
    out["max_abs_err"] = float(err.max())
    if tol is not None:
        tol = np.asarray(tol, dtype=np.float64).ravel()
        q = err / tol
        i = int(np.argmax(q))
        out["max_normalised_err"] = float(q[i])
        out["tolerance"] = tol_formula
        out["worst"] = {"index": i, "ref": float(rv[i]), "oracle": float(gv[i]), "tol": float(tol[i])}
    return out


def reassoc_tol(ref_words: np.ndarray, DT: str, K: int, scale: np.ndarray, c: float = 2.0) -> np.ndarray:
    """`c*(K+2)*u32*scale + ulp_DT(ref)`: two K-term f32 accumulations in different orders differ by at most
    2*gamma_K*scale (scale = sum of |terms|, or a row scale for composed reductions) before the final rounding."""
    rv = ref_prims_twins.np_widen(np.asarray(ref_words).ravel(), DT)
    return c * (K + 2) * U32 * np.asarray(scale, dtype=np.float64).ravel() + ulp_dt(rv, DT)


def row_scale(ref_words: np.ndarray, DT: str, shape: tuple) -> np.ndarray:
    """max |ref| over each row, broadcast back to the elements (a composed-reduction row's magnitude scale)."""
    rv = np.abs(ref_prims_twins.np_widen(np.asarray(ref_words).reshape(shape), DT).astype(np.float64))
    return np.broadcast_to(rv.max(axis=-1, keepdims=True), shape)


def write_record(kind: str, DT: str, *, statics: dict, oracle_op: str, domain: dict, result: dict, gate_check: dict,
                 notes: str = "") -> Path:
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"schema": "verity-ir/ref-prims-conformance/v0", "kind": kind, "DT": DT, "statics_tested": statics, "label": LABEL,
           "oracle": {"torch": torch.__version__, "device": "cpu", "op": oracle_op, "platform": platform.platform(),
                      "machine": platform.machine(), "python": platform.python_version(), "numpy": np.__version__},
           "domain": {"seed": SEED, **domain}, "result": result, "gate_check": gate_check,
           "definition": {"ref_vocab_version": R.REF_VOCAB_VERSION, "ref_vocab_digest": R.ref_vocab_digest()},
           "notes": notes}
    p = RECORD_DIR / f"{kind}-{DT}.json"
    p.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return p


# ---------------------------------------------------------------------------------------------------------
# torch oracles (the source patterns the kinds model)
# ---------------------------------------------------------------------------------------------------------


def hf_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dt = x.dtype
    h = x.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    return w * h.to(dt)


def hf_rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def hf_rope(x, cos, sin):
    return (x * cos) + (hf_rotate_half(x) * sin)


def hf_eager_attention_head(q, k, v, scale_op: str):
    """q (R, D), k/v (R, T, D): one query row per batch entry, all keys visible."""
    D = q.shape[-1]
    s = torch.matmul(q[:, None, :], k.transpose(1, 2))  # (R, 1, T)
    if scale_op == "mul":
        s = s * (D ** -0.5)
    else:
        s = s / torch.full([], D ** 0.5, dtype=s.dtype)
    p = F.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(p, v)[:, 0, :]


# ---------------------------------------------------------------------------------------------------------
# 1. definition consistency: twins == gates (small shapes, both dtypes)
# ---------------------------------------------------------------------------------------------------------

DTS = ("bf16", "f32")


@pytest.mark.parametrize("DT", DTS)
def test_twins_equal_gates(DT):
    rng = np.random.default_rng(SEED)
    K, N, D, T, NH, KVH, V, H = 16, 3, 8, 5, 2, 1, 7, 4
    x, w, b = rand_words(rng, K, DT), rand_words(rng, (N, K), DT), rand_words(rng, N, DT)
    out, _ = gates(bind(R.LinearRef, K=K, N=N, DT=DT, BIAS=True), x, w, b)
    assert (out == ref_prims_twins.np_linear_ref(x[None], w, b, DT)[0]).all()
    out, _ = gates(bind(R.LinearRef, K=K, N=N, DT=DT, BIAS=False), x, w)
    assert (out == ref_prims_twins.np_linear_ref(x[None], w, None, DT)[0]).all()
    wv, g = rand_words(rng, K, DT), rand_words(rng, K, DT)
    out, _ = gates(bind(R.RMSNormRef, N=K, EPS=1e-6, DT=DT), x, wv)
    assert (out == ref_prims_twins.np_rmsnorm_ref(x[None], wv, 1e-6, DT)[0]).all()
    out, _ = gates(bind(R.LayerNormRef, N=K, EPS=1e-5, DT=DT), x, g, wv)
    assert (out == ref_prims_twins.np_layernorm_ref(x[None], g, wv, 1e-5, DT)[0]).all()
    for ap in R.GELU_APPROX:
        out, _ = gates(bind(R.GeluRef, N=K, DT=DT, APPROX=ap), x)
        assert (out == ref_prims_twins.np_gelu_ref(x[None], DT, ap)[0]).all()
    out, _ = gates(bind(R.SiluRef, N=K, DT=DT), x)
    assert (out == ref_prims_twins.np_silu_ref(x[None], DT)[0]).all()
    out, _ = gates(bind(R.SoftmaxRef, N=K, DT=DT), x)
    assert (out == ref_prims_twins.np_softmax_ref(x[None], DT)[0]).all()
    xr, c, s = rand_words(rng, D, DT), rand_words(rng, D, DT), rand_words(rng, D, DT)
    out, _ = gates(bind(R.RoPERef, D=D, DT=DT), xr, c, s)
    assert (out == ref_prims_twins.np_rope_ref(xr[None], c, s, DT)[0]).all()
    q, kc, vc = rand_words(rng, (NH, D), DT), rand_words(rng, (T, KVH, D), DT), rand_words(rng, (T, KVH, D), DT)
    for op in R.SCALE_OPS:
        out, _ = gates(bind(R.EagerAttentionRef, T=T, NH=NH, KVH=KVH, D=D, DT=DT, SCALE_OP=op), q, kc, vc)
        ref = np.stack([ref_prims_twins.np_eager_attention_head_ref(q[h:h + 1], kc[None, :, h // (NH // KVH), :], vc[None, :, h // (NH // KVH), :], DT, op)[0]
                        for h in range(NH)])
        assert (out.reshape(NH, D) == ref).all()
    tab = rand_words(rng, (V, H), DT)
    out, _ = gates(bind(R.EmbeddingRef, V=V, H=H, DT=DT), np.array([3], dtype=np.uint32), tab)
    assert (out == tab[3]).all()
    lg = rand_words(rng, V, DT)
    out, _ = gates(bind(R.ArgmaxRef, V=V, DT=DT), lg)
    assert int(out[0]) == int(np.argmax(ref_prims_twins.np_widen(lg, DT)))


# ---------------------------------------------------------------------------------------------------------
# 2. conformance vs torch CPU (recorded)
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("DT", DTS)
def test_linear_ref_vs_torch(DT):
    rng = np.random.default_rng(SEED)
    shapes = [(16, 8), (64, 32), (256, 16), (768, 4)]
    rows = 8
    stats = []
    total_ref, total_got, total_tol = [], [], []
    formula = "2*(K+2)*u32*(sum_k |x_k w_k| + |b|) + ulp_DT(ref)"
    for K, N in shapes:
        for bias in (True, False):
            x = rand_words(rng, (rows, K), DT)
            w = rand_words(rng, (N, K), DT, scale=K ** -0.5)
            b = rand_words(rng, N, DT, scale=0.1) if bias else None
            ref = ref_prims_twins.np_linear_ref(x, w, b, DT)
            got = t_words(F.linear(words_t(x, DT), words_t(w, DT), None if b is None else words_t(b, DT)))
            xa, wa = np.abs(ref_prims_twins.np_widen(x, DT).astype(np.float64)), np.abs(ref_prims_twins.np_widen(w, DT).astype(np.float64))
            scale = xa @ wa.T + (0.0 if b is None else np.abs(ref_prims_twins.np_widen(b, DT).astype(np.float64)))
            tol = reassoc_tol(ref, DT, K, scale)
            total_ref.append(ref.ravel()), total_got.append(got.ravel()), total_tol.append(tol)
            stats.append({"K": K, "N": N, "BIAS": bias, **compare(ref, got, DT, tol=tol, tol_formula=formula)})
    # the body is the definition: check twin == gates on one small shape from the same domain
    K, N = 16, 8
    x, w, b = rand_words(rng, K, DT), rand_words(rng, (N, K), DT, scale=0.25), rand_words(rng, N, DT, scale=0.1)
    out, n_gates = gates(bind(R.LinearRef, K=K, N=N, DT=DT, BIAS=True), x, w, b)
    assert (out == ref_prims_twins.np_linear_ref(x[None], w, b, DT)[0]).all()
    res = compare(np.concatenate(total_ref), np.concatenate(total_got), DT, tol=np.concatenate(total_tol), tol_formula=formula)
    res["per_shape"] = stats
    # torch's GEMM is blocked (oneDNN/MKL/cpublas); the reference is the sequential fma chain: a-priori bound, not equality
    assert res["max_normalised_err"] <= 1.0, res
    write_record("LinearRef_v1", DT, statics={"K,N": shapes, "BIAS": [True, False]}, oracle_op="torch.nn.functional.linear",
                 domain={"rows": rows, "x": "N(0,1)", "w": "N(0,1)/sqrt(K)", "b": "N(0,0.1)", "dtype": DT}, result=res,
                 gate_check={"shape": {"K": K, "N": N, "BIAS": True}, "gates": n_gates, "twin_equals_gates": True},
                 notes="torch CPU accumulates in a blocked order (oneDNN/MKL or cpublas fallback); the reference is the "
                       "sequential IEEE fma chain from +0. Disagreements are reassociation differences within the a-priori "
                       "bound; output-ulp distance is large only where the dot product nearly cancels.")


@pytest.mark.parametrize("DT", DTS)
def test_rmsnorm_ref_vs_torch(DT):
    rng = np.random.default_rng(SEED)
    Ns, rows, scales, eps = [16, 64, 576, 1536], 8, [0.01, 1.0, 30.0], 1e-6
    refs, gots, tols, per = [], [], [], []
    formula = "4*(N+2)*u32*max_row|ref| + ulp_DT(ref)"
    for N in Ns:
        for sc in scales:
            x, w = rand_words(rng, (rows, N), DT, scale=sc), rand_words(rng, N, DT)
            ref = ref_prims_twins.np_rmsnorm_ref(x, w, eps, DT)
            got = t_words(hf_rmsnorm(words_t(x, DT), words_t(w, DT), eps))
            tol = reassoc_tol(ref, DT, N, row_scale(ref, DT, (rows, N)), c=4.0)
            refs.append(ref.ravel()), gots.append(got.ravel()), tols.append(tol)
            per.append({"N": N, "scale": sc, **compare(ref, got, DT, tol=tol, tol_formula=formula)})
    x, w = rand_words(rng, 16, DT), rand_words(rng, 16, DT)
    out, n_gates = gates(bind(R.RMSNormRef, N=16, EPS=eps, DT=DT), x, w)
    assert (out == ref_prims_twins.np_rmsnorm_ref(x[None], w, eps, DT)[0]).all()
    res = compare(np.concatenate(refs), np.concatenate(gots), DT, tol=np.concatenate(tols), tol_formula=formula)
    res["per_shape"] = per
    assert res["max_normalised_err"] <= 1.0, res
    write_record("RMSNormRef_v1", DT, statics={"N": Ns, "EPS": eps}, oracle_op="HF LlamaRMSNorm/Qwen2RMSNorm forward (torch ops)",
                 domain={"rows": rows, "x": "N(0,scale)", "scales": scales, "w": "N(0,1)", "dtype": DT}, result=res,
                 gate_check={"shape": {"N": 16}, "gates": n_gates, "twin_equals_gates": True},
                 notes="torch `mean` reduces in a vectorised order; the reference sums squares sequentially. `rsqrt` on x86 "
                       "torch CPU is 1/sqrt (two roundings) as in the reference.")


@pytest.mark.parametrize("DT", DTS)
def test_layernorm_ref_vs_torch(DT):
    rng = np.random.default_rng(SEED)
    Ns, rows, eps = [16, 64, 768], 8, 1e-5
    refs, gots, tols, per = [], [], [], []
    formula = "8*(N+2)*u32*max_row|ref| + ulp_DT(ref)"
    for N in Ns:
        for sc in (1.0, 10.0):
            x = rand_words(rng, (rows, N), DT, scale=sc)
            g, b = rand_words(rng, N, DT), rand_words(rng, N, DT, scale=0.1)
            ref = ref_prims_twins.np_layernorm_ref(x, g, b, eps, DT)
            got = t_words(F.layer_norm(words_t(x, DT), (N,), words_t(g, DT), words_t(b, DT), eps))
            tol = reassoc_tol(ref, DT, N, row_scale(ref, DT, (rows, N)), c=8.0)
            refs.append(ref.ravel()), gots.append(got.ravel()), tols.append(tol)
            per.append({"N": N, "scale": sc, **compare(ref, got, DT, tol=tol, tol_formula=formula)})
    x, g, b = rand_words(rng, 16, DT), rand_words(rng, 16, DT), rand_words(rng, 16, DT)
    out, n_gates = gates(bind(R.LayerNormRef, N=16, EPS=eps, DT=DT), x, g, b)
    assert (out == ref_prims_twins.np_layernorm_ref(x[None], g, b, eps, DT)[0]).all()
    res = compare(np.concatenate(refs), np.concatenate(gots), DT, tol=np.concatenate(tols), tol_formula=formula)
    res["per_shape"] = per
    assert res["max_normalised_err"] <= 1.0, res
    write_record("LayerNormRef_v1", DT, statics={"N": Ns, "EPS": eps}, oracle_op="torch.nn.functional.layer_norm",
                 domain={"rows": rows, "x": "N(0,scale)", "scales": [1.0, 10.0], "g": "N(0,1)", "b": "N(0,0.1)", "dtype": DT}, result=res,
                 gate_check={"shape": {"N": 16}, "gates": n_gates, "twin_equals_gates": True},
                 notes="torch CPU uses RowwiseMoments (chunked Welford) and `x*(rstd*g) + (b - mean*rstd*g)`; the reference is "
                       "two-pass sequential with `((x-mean)*rstd)*g + b`.")


def _elementwise_domain(rng, DT: str) -> np.ndarray:
    """Every finite bf16 word (exhaustive) for DT=bf16; 65,536 random f32 values ~ N(0, 4) plus edge words for f32."""
    if DT == "bf16":
        w = np.arange(1 << 16, dtype=np.uint16)
        finite = (w & 0x7F80) != 0x7F80
        return w[finite]
    x = (rng.standard_normal(1 << 16) * 4).astype(np.float32).view(np.uint32)
    edges = np.array([0, 0x80000000, 0x00000001, 0x80000001, 0x00800000, 0x3F800000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF,
                      0x42B17217, 0x42B17218, 0xC2CE0000, 0x41133333, 0xC1133333, 0x40866666, 0xC0866666], dtype=np.uint32)
    return np.concatenate([x, edges])


@pytest.mark.parametrize("DT", DTS)
@pytest.mark.parametrize("approx", R.GELU_APPROX)
@pytest.mark.slow
def test_gelu_ref_vs_torch(DT, approx):
    rng = np.random.default_rng(SEED)
    x = _elementwise_domain(rng, DT)
    ref = ref_prims_twins.np_gelu_ref(x, DT, approx)
    xt = words_t(x, DT)
    got_vec = t_words(F.gelu(xt, approximate=approx))  # contiguous: ATen's vectorised kernel (Vectorized erf/tanh)
    got_sca = t_words(F.gelu(torch.stack([xt, xt], 1)[:, 0], approximate=approx))  # strided: ATen's scalar std::erf/std::tanh path
    # gelu = 0.5*x*(1 + f(.)) with f = erf or tanh: a d-difference in f becomes 0.5*|x|*d absolutely, and for x < 0
    # 1 + f cancels (factor |f|/(1+f): 2 at x=-1, unbounded in the tail) so that becomes many *ulps*; the bound is absolute.
    xv = np.abs(ref_prims_twins.np_widen(x, DT).astype(np.float64))
    formula = "0.5*|x|*2^-21 + 4*ulp_DT(ref)   (oracle's erf/tanh within 4.8e-7 absolute of correctly rounded; measured worst 3.5e-7)"
    tol = 0.5 * xv * 2.0 ** -21 + 4 * ulp_dt(ref_prims_twins.np_widen(ref, DT), DT)
    res_vec = compare(ref, got_vec, DT, tol=tol, tol_formula=formula)
    res_sca = compare(ref, got_sca, DT, tol=tol, tol_formula=formula)
    xs = x[rng.integers(0, x.size, 16)]
    out, n_gates = gates(bind(R.GeluRef, N=16, DT=DT, APPROX=approx), xs)
    assert (out == ref_prims_twins.np_gelu_ref(xs, DT, approx)).all()
    assert res_vec["max_normalised_err"] <= 1.0, res_vec
    assert res_sca["max_normalised_err"] <= 1.0, res_sca
    if DT == "bf16":  # exhaustive domain: measured 3 (erf) / 4 (tanh) differing words of 65,280 on the vectorised path
        assert res_vec["exact_fraction"] >= 0.9999, res_vec
    res = {"torch_vectorised_path_contiguous": res_vec, "torch_scalar_path_strided": res_sca,
           "words": res_vec["words"], "exact_fraction": res_vec["exact_fraction"], "max_ulp": res_vec["max_ulp"],
           "max_normalised_err": max(res_vec["max_normalised_err"], res_sca["max_normalised_err"])}
    write_record(f"GeluRef_v1[APPROX={approx}]", DT, statics={"APPROX": approx, "N": "element-wise"},
                 oracle_op=f"torch.nn.functional.gelu(approximate={approx!r}) on a contiguous (vectorised) and a strided (scalar) tensor",
                 domain={"x": "every finite bf16 word (exhaustive)" if DT == "bf16" else "65,536 x N(0,4) f32 + 16 edge words"},
                 result=res, gate_check={"shape": {"N": 16}, "gates": n_gates, "twin_equals_gates": True},
                 notes="The reference is the erf/tanh form x*0.5*(1+f(.)) with a correctly rounded f (F32ErfRn/F32TanhRn); torch "
                       "uses the same formula on two code paths. (i) Strided tensors -> scalar libm std::erf/std::tanh: erf form "
                       "98.8% of f32 words exact (glibc erff is not correctly rounded everywhere), tanh form 93.6%. (ii) Contiguous "
                       "tensors (all bf16, f32) -> ATen Vectorized<float>::erf/tanh: the vectorised erf is a polynomial "
                       "approximation with absolute error up to ~3.5e-7 near x=0 (worst tested word x=-0.03167: 8 ulps of the "
                       "result) and is NOT the same erf as torch.erf on a contiguous tensor (MKL VML vsErf, <= 1 ulp: see the "
                       "F32ErfRn_v1 record; probe: F.gelu(x) != x*0.5*(1+torch.erf(x*M_SQRT1_2)) at x=-1.0); the vectorised tanh is "
                       "Sleef <= 1 ulp (tanh(-9.009) = -1.0 where correctly rounded is -0.99999994). For x < 0 the 1+f cancellation "
                       "amplifies f's error (2 ulps at x=-1; factor-2 or -0.0-vs-1.6e-7 results for x < -5), which is why the f32 "
                       "exact fraction is low while the absolute difference stays within 0.5*|x|*2^-21 + 4 ulp on every path. bf16 "
                       "outputs absorb all of it but 3 (erf) / 4 (tanh, the tail -0.0) words of the exhaustive 65,280.")


@pytest.mark.parametrize("DT", DTS)
def test_silu_ref_vs_torch(DT):
    rng = np.random.default_rng(SEED)
    x = _elementwise_domain(rng, DT)
    ref = ref_prims_twins.np_silu_ref(x, DT)
    xt = words_t(x, DT)
    res_vec = compare(ref, t_words(F.silu(xt)), DT)
    res_sca = compare(ref, t_words(F.silu(torch.stack([xt, xt], 1)[:, 0])), DT)
    xs = x[rng.integers(0, x.size, 16)]
    out, n_gates = gates(bind(R.SiluRef, N=16, DT=DT), xs)
    assert (out == ref_prims_twins.np_silu_ref(xs, DT)).all()
    assert res_vec["max_ulp"] <= (1 if DT == "bf16" else 4), res_vec
    assert res_sca["max_ulp"] <= (1 if DT == "bf16" else 4), res_sca
    res = {"torch_vectorised_path_contiguous": res_vec, "torch_scalar_path_strided": res_sca,
           "words": res_vec["words"], "exact_fraction": res_vec["exact_fraction"], "max_ulp": max(res_vec["max_ulp"], res_sca["max_ulp"])}
    write_record("SiluRef_v1", DT, statics={"N": "element-wise"}, oracle_op="torch.nn.functional.silu (contiguous and strided)",
                 domain={"x": "every finite bf16 word (exhaustive)" if DT == "bf16" else "65,536 x N(0,4) f32 + 16 edge words"},
                 result=res, gate_check={"shape": {"N": 16}, "gates": n_gates, "twin_equals_gates": True},
                 notes="torch silu = x / (1 + exp(-x)) (vectorised Sleef expf, scalar std::exp); the reference uses the correctly "
                       "rounded F32ExpRn with the same division form. No cancellation: differences stay at 1 ulp.")


@pytest.mark.parametrize("DT", DTS)
def test_softmax_ref_vs_torch(DT):
    rng = np.random.default_rng(SEED)
    Ns, rows, scales = [5, 64, 512], 8, [1.0, 10.0]
    refs, gots, tols, per = [], [], [], []
    formula = "4*(N+2)*u32*|ref| + ulp_DT(ref)  (positive terms: relative)"
    for N in Ns:
        for sc in scales:
            x = rand_words(rng, (rows, N), DT, scale=sc)
            ref = ref_prims_twins.np_softmax_ref(x, DT)
            got = t_words(F.softmax(words_t(x, DT), dim=-1, dtype=torch.float32).to(tdt(DT)))
            tol = reassoc_tol(ref, DT, N, np.abs(ref_prims_twins.np_widen(ref, DT).astype(np.float64)), c=4.0)
            refs.append(ref.ravel()), gots.append(got.ravel()), tols.append(tol)
            per.append({"N": N, "scale": sc, **compare(ref, got, DT, tol=tol, tol_formula=formula)})
    x = rand_words(rng, 16, DT)
    out, n_gates = gates(bind(R.SoftmaxRef, N=16, DT=DT), x)
    assert (out == ref_prims_twins.np_softmax_ref(x[None], DT)[0]).all()
    res = compare(np.concatenate(refs), np.concatenate(gots), DT, tol=np.concatenate(tols), tol_formula=formula)
    res["per_shape"] = per
    assert res["max_normalised_err"] <= 1.0, res
    write_record("SoftmaxRef_v1", DT, statics={"N": Ns}, oracle_op="torch.nn.functional.softmax(dim=-1, dtype=float32).to(dt)",
                 domain={"rows": rows, "x": "N(0,scale)", "scales": scales, "dtype": DT}, result=res,
                 gate_check={"shape": {"N": 16}, "gates": n_gates, "twin_equals_gates": True},
                 notes="torch: vectorised max, Sleef exp, vectorised sum, then multiply by 1/sum; reference: sequential max/sum, "
                       "correctly rounded exp, multiply by RN(1/sum).")


@pytest.mark.parametrize("DT", DTS)
def test_rope_ref_vs_torch(DT):
    rng = np.random.default_rng(SEED)
    Ds, rows = [8, 64, 128], 8
    refs, gots, per = [], [], []
    for D in Ds:
        x = rand_words(rng, (rows, D), DT, scale=4.0)
        pos = torch.arange(rows, dtype=torch.float32)[:, None]
        inv = 1.0 / (10000.0 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        freqs = pos * inv[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos().to(tdt(DT)), emb.sin().to(tdt(DT))  # the model's derived cos/sin rows (Inputs of the kind)
        ref = ref_prims_twins.np_rope_ref(x, t_words(cos), t_words(sin), DT)
        got = t_words(hf_rope(words_t(x, DT), cos, sin))
        refs.append(ref.ravel()), gots.append(got.ravel())
        per.append({"D": D, **compare(ref, got, DT)})
    D = 8
    x, c, s = rand_words(rng, D, DT), rand_words(rng, D, DT), rand_words(rng, D, DT)
    out, n_gates = gates(bind(R.RoPERef, D=D, DT=DT), x, c, s)
    assert (out == ref_prims_twins.np_rope_ref(x[None], c, s, DT)[0]).all()
    res = compare(np.concatenate(refs), np.concatenate(gots), DT)
    res["per_shape"] = per
    # three DT tensor ops, each a single IEEE rounding in torch as in the reference: bit equality expected
    assert res["exact_fraction"] == 1.0, res
    write_record("RoPERef_v1", DT, statics={"D": Ds}, oracle_op="HF apply_rotary_pos_emb (x*cos + rotate_half(x)*sin)",
                 domain={"rows": rows, "x": "N(0,4)", "cos/sin": "HF rotary table rows, positions 0..7, theta 1e4, cast to dt"},
                 result=res, gate_check={"shape": {"D": D}, "gates": n_gates, "twin_equals_gates": True},
                 notes="Element-wise products and one add, each a DT tensor op: no reduction, bit-exact on the domain.")


@pytest.mark.parametrize("DT", DTS)
@pytest.mark.parametrize("scale_op", R.SCALE_OPS)
def test_eager_attention_head_ref_vs_torch(DT, scale_op):
    rng = np.random.default_rng(SEED)
    cases, rows = [(1, 16), (5, 16), (64, 64), (130, 64)], 4
    refs, gots, tols, per = [], [], [], []
    formula = "16*(T+D+2)*u32*max_row|v| + ulp_DT(ref)"
    for T, D in cases:
        q = rand_words(rng, (rows, D), DT)
        k = rand_words(rng, (rows, T, D), DT)
        v = rand_words(rng, (rows, T, D), DT)
        ref = ref_prims_twins.np_eager_attention_head_ref(q, k, v, DT, scale_op)
        got = t_words(hf_eager_attention_head(words_t(q, DT), words_t(k, DT), words_t(v, DT), scale_op))
        vmax = np.abs(ref_prims_twins.np_widen(v, DT).astype(np.float64)).reshape(rows, -1).max(axis=1)  # |out| <= max|v| (convex combination)
        tol = reassoc_tol(ref, DT, T + D, np.broadcast_to(vmax[:, None], (rows, D)), c=16.0)
        refs.append(ref.ravel()), gots.append(got.ravel()), tols.append(tol)
        per.append({"T": T, "D": D, **compare(ref, got, DT, tol=tol, tol_formula=formula)})
    T, D = 5, 8
    q, k, v = rand_words(rng, D, DT), rand_words(rng, (T, D), DT), rand_words(rng, (T, D), DT)
    out, n_gates = gates(bind(R.EagerAttentionHeadRef, T=T, D=D, DT=DT, SCALE_OP=scale_op), q, k, v)
    assert (out == ref_prims_twins.np_eager_attention_head_ref(q[None], k[None], v[None], DT, scale_op)[0]).all()
    res = compare(np.concatenate(refs), np.concatenate(gots), DT, tol=np.concatenate(tols), tol_formula=formula)
    res["per_shape"] = per
    assert res["max_normalised_err"] <= 1.0, res
    write_record(f"EagerAttentionHeadRef_v1[SCALE_OP={scale_op}]", DT, statics={"T,D": cases, "SCALE_OP": scale_op},
                 oracle_op="HF eager_attention_forward on one query row (matmul, *scaling or /sqrt(D), softmax(f32).to(dt), matmul)",
                 domain={"rows": rows, "q,k,v": "N(0,1)", "dtype": DT}, result=res,
                 gate_check={"shape": {"T": T, "D": D}, "gates": n_gates, "twin_equals_gates": True},
                 notes="Two torch matmuls (blocked accumulation) and a softmax (Sleef exp, vectorised sum) vs sequential fma "
                       "chains and the correctly rounded exp.")


def test_the_torch_oracles_bf16_matmuls_run_off_amx():
    """The suite's conftest keeps oneDNN off AMX, where a preempted bf16 matmul now and then returns wrong rows: a child with
    this process's environment runs its bf16 matmul on another ISA."""
    code = "import torch; torch.matmul(torch.ones(4, 1, 130, dtype=torch.bfloat16), torch.ones(4, 130, 64, dtype=torch.bfloat16))"
    r = subprocess.run([sys.executable, "-c", code], env={**os.environ, "ONEDNN_VERBOSE": "1"}, capture_output=True, text=True,
                       check=True)
    ran = [line for line in r.stdout.splitlines() if ",exec,cpu,matmul," in line]
    if not ran:
        pytest.skip("torch ran this bf16 matmul outside oneDNN")
    assert not any("amx" in line for line in ran), ran


@pytest.mark.parametrize("DT", DTS)
def test_gqa_attention_ref_vs_torch_repeat_kv(DT):
    """`EagerAttentionRef` GQA (gates) == HF eager attention with `repeat_kv` on the same words."""
    rng = np.random.default_rng(SEED + 1)
    T, NH, KVH, D = 6, 4, 2, 8
    q, kc, vc = rand_words(rng, (NH, D), DT), rand_words(rng, (T, KVH, D), DT), rand_words(rng, (T, KVH, D), DT)
    out, n_gates = gates(bind(R.EagerAttentionRef, T=T, NH=NH, KVH=KVH, D=D, DT=DT, SCALE_OP="mul"), q, kc, vc)
    qt, kt, vt = words_t(q, DT), words_t(kc, DT), words_t(vc, DT)
    G = NH // KVH
    k_rep = kt.permute(1, 0, 2).repeat_interleave(G, dim=0)  # (NH, T, D)
    v_rep = vt.permute(1, 0, 2).repeat_interleave(G, dim=0)
    got = t_words(hf_eager_attention_head(qt, k_rep, v_rep, "mul"))
    vmax = float(np.abs(ref_prims_twins.np_widen(vc, DT).astype(np.float64)).max())
    tol = reassoc_tol(out.reshape(NH, D), DT, T + D, np.full((NH, D), vmax), c=16.0)
    res = compare(out.reshape(NH, D), got, DT, tol=tol, tol_formula="16*(T+D+2)*u32*max|v| + ulp_DT(ref)")
    assert res["max_normalised_err"] <= 1.0, res
    write_record("EagerAttentionRef_v1", DT, statics={"T": T, "NH": NH, "KVH": KVH, "D": D, "SCALE_OP": "mul"},
                 oracle_op="HF eager attention with repeat_kv (torch ops)", domain={"q,kc,vc": "N(0,1)", "dtype": DT}, result=res,
                 gate_check={"gates": n_gates, "evaluated": "gate-by-gate (this record IS the gate evaluation)"},
                 notes="GQA head->KV-head mapping h // (NH/KVH) checked against repeat_kv.")


@pytest.mark.parametrize("DT", DTS)
def test_embedding_add_mul_argmax_ref_vs_torch(DT):
    rng = np.random.default_rng(SEED)
    V, H, N = 50, 16, 64
    tab = rand_words(rng, (V, H), DT)
    toks = rng.integers(0, V, 8)
    ref = np.stack([gates(bind(R.EmbeddingRef, V=V, H=H, DT=DT), np.array([t], dtype=np.uint32), tab)[0] for t in toks])
    got = t_words(F.embedding(torch.tensor(toks), words_t(tab, DT)))
    assert (ref == got).all()
    a, b = rand_words(rng, N, DT), rand_words(rng, N, DT)
    add, _ = gates(bind(R.AddRef, N=N, DT=DT), a, b)
    mul, _ = gates(bind(R.MulRef, N=N, DT=DT), a, b)
    assert (add == t_words(words_t(a, DT) + words_t(b, DT))).all()
    assert (mul == t_words(words_t(a, DT) * words_t(b, DT))).all()
    exact_words = 2 * N + ref.size
    am_ok = 0
    for _ in range(16):
        lg = rand_words(rng, V, DT)
        lg[rng.integers(0, V)] = lg[rng.integers(0, V)]  # inject a tie
        out, _ = gates(bind(R.ArgmaxRef, V=V, DT=DT), lg)
        am_ok += int(int(out[0]) == int(torch.argmax(words_t(lg, DT))))
    assert am_ok == 16
    write_record("EmbeddingRef_v1+AddRef_v1+MulRef_v1+ArgmaxRef_v1", DT, statics={"V": V, "H": H, "N": N},
                 oracle_op="F.embedding, aten.add, aten.mul, torch.argmax",
                 domain={"tokens": 8, "add/mul words": N, "argmax rows": 16, "dtype": DT},
                 result={"words": exact_words, "exact_words": exact_words, "exact_fraction": 1.0, "max_ulp": 0, "argmax_rows_exact": am_ok},
                 gate_check={"evaluated": "gate-by-gate"}, notes="No reduction with rounding: bit-exact on NaN-free words.")


def gptj_rotate_every_two(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def gptj_apply_rotary(x, sin, cos):
    """HF `GPTJAttention._apply_rotary_pos_emb` with the D/2 tables (it repeat_interleaves them by 2)."""
    sin = torch.repeat_interleave(sin, 2, -1)
    cos = torch.repeat_interleave(cos, 2, -1)
    return (x * cos) + (gptj_rotate_every_two(x) * sin)


@pytest.mark.parametrize("DT", DTS)
def test_rope_interleaved_ref_vs_torch_gptj(DT):
    rng = np.random.default_rng(SEED)
    Ds, rows = [8, 64, 128], 8
    refs, gots, per = [], [], []
    for D in Ds:
        x = rand_words(rng, (rows, D), DT, scale=4.0)
        pos = torch.arange(rows, dtype=torch.float32)[:, None]
        inv = 1.0 / (10000.0 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        freqs = pos * inv[None, :]
        cos, sin = freqs.cos().to(tdt(DT)), freqs.sin().to(tdt(DT))  # (rows, D/2)
        ref = ref_prims_twins.np_rope_interleaved_ref(x, t_words(cos), t_words(sin), DT)
        got = t_words(gptj_apply_rotary(words_t(x, DT), sin, cos))
        refs.append(ref.ravel()), gots.append(got.ravel())
        per.append({"D": D, **compare(ref, got, DT)})
    D = 8
    x, c, s = rand_words(rng, D, DT), rand_words(rng, D // 2, DT), rand_words(rng, D // 2, DT)
    out, n_gates = gates(bind(R.RoPEInterleavedRef, D=D, DT=DT), x, c, s)
    assert (out == ref_prims_twins.np_rope_interleaved_ref(x[None], c, s, DT)[0]).all()
    res = compare(np.concatenate(refs), np.concatenate(gots), DT)
    res["per_shape"] = per
    assert res["exact_fraction"] == 1.0, res
    write_record("RoPEInterleavedRef_v1", DT, statics={"D": Ds}, oracle_op="HF GPT-J _apply_rotary_pos_emb (x*cos + rotate_every_two(x)*sin)",
                 domain={"rows": rows, "x": "N(0,4)", "cos/sin": "GPT-J table rows (D/2), positions 0..7, theta 1e4, cast to dt"},
                 result=res, gate_check={"shape": {"D": D}, "gates": n_gates, "twin_equals_gates": True},
                 notes="Element-wise products and one add per output, each a DT tensor op: no reduction, bit-exact on the domain.")


def test_rope_gptj_profile_head_is_a_pairing_permutation_of_rope_head():
    """`RoPEHeadGptJ_v1` (vLLM is_neox=False) == unpermute(RoPEHead_v1(permute(x))): same RopeOut/RopeOutAdd gates."""
    rng = np.random.default_rng(SEED)
    for D in (8, 64, 128):
        R_ = D // 2
        x, cs = rand_words(rng, D, "bf16", scale=3.0), rand_words(rng, D, "bf16", scale=0.7)
        gptj, n = gates(bind(R.RoPEHeadGptJ, D=D), x, cs)
        neox, n2 = gates(bind(b1.RoPEHead, D=D), np.concatenate([x[0::2], x[1::2]]), cs)
        unperm = np.empty(D, dtype=neox.dtype)
        unperm[0::2], unperm[1::2] = neox[:R_], neox[R_:]
        assert (gptj == unperm).all() and n == n2 == D
        heads, _ = gates(bind(R.RoPEGptJ, NHEADS=3, D=D), np.stack([x, x, x]), cs)
        assert (heads.reshape(3, D) == gptj).all()
    write_record("RoPEHeadGptJ_v1+RoPEGptJ_v1", "bf16", statics={"D": [8, 64, 128], "NHEADS": 3},
                 oracle_op="RoPEHead_v1 (exact-model-tested vLLM NeoX kernel) under the (2i,2i+1)<->(i,i+D/2) permutation; no CUDA run",
                 domain={"x": "N(0,3) bf16", "cs": "N(0,0.7) bf16"},
                 result={"words": 3 * (8 + 64 + 128), "exact_words": 3 * (8 + 64 + 128), "exact_fraction": 1.0, "max_ulp": 0},
                 gate_check={"evaluated": "gate-by-gate on both sides"},
                 notes="PROFILE-MODE kind (not in the reference digest): the vLLM is_neox=False kernel with one bf16 rounding. "
                       "Conformance status model-untested: the pairing is read from vLLM's kernel source; its arithmetic is the "
                       "exact-model-tested RopeOut_v1/RopeOutAdd_v1. Proof-supported (RRopeOut/RRopeOutAdd twins).")


@pytest.mark.parametrize("DT", DTS)
def test_relu_ref_vs_torch(DT):
    rng = np.random.default_rng(SEED)
    x = _elementwise_domain(rng, DT)
    ref = ref_prims_twins.np_relu_ref(x, DT)
    xt = words_t(x, DT)
    got_vec = t_words(F.relu(xt))
    got_sca = t_words(F.relu(torch.stack([xt, xt], 1)[:, 0]))
    neg_zero = x == (0x8000 if DT == "bf16" else 0x80000000)
    res_vec, res_sca = compare(ref, got_vec, DT), compare(ref, got_sca, DT)
    assert res_vec["exact_fraction"] == 1.0 and res_sca["exact_fraction"] == 1.0, (res_vec, res_sca)
    nz = {"reference": hex(int(ref[neg_zero][0])), "torch_vectorised": hex(int(got_vec[neg_zero][0])),
          "torch_scalar": hex(int(got_sca[neg_zero][0]))}
    xs = x[rng.integers(0, x.size, 16)]
    xs[0] = 0x8000 if DT == "bf16" else 0x80000000
    out, n_gates = gates(bind(R.ReluRef, N=16, DT=DT), xs)
    assert (out == ref_prims_twins.np_relu_ref(xs, DT)).all() and out[0] == xs[0]
    write_record("ReluRef_v1", DT, statics={"N": "element-wise"}, oracle_op="torch.nn.functional.relu (contiguous and strided)",
                 domain={"x": "every finite bf16 word (exhaustive, incl. -0.0)" if DT == "bf16" else "65,536 x N(0,4) f32 + 16 edge words (incl. ±0)"},
                 result={"vectorised": res_vec, "scalar": res_sca, "relu(-0.0)": nz,
                         "words": res_vec["words"], "exact_fraction": 1.0, "max_ulp": 0},
                 gate_check={"shape": {"N": 16}, "gates": n_gates, "twin_equals_gates": True},
                 notes="Bit-exact on every tested word on both torch paths, including relu(-0.0) = -0.0.")


def test_i64_to_i32_vs_torch():
    rng = np.random.default_rng(SEED)
    vals = np.concatenate([rng.integers(-(1 << 31), 1 << 31, 4096, dtype=np.int64),
                           np.array([0, 1, -1, (1 << 31) - 1, -(1 << 31)], dtype=np.int64)])
    got = torch.from_numpy(vals).to(torch.int32).numpy().astype(np.uint32)
    ref = np.array([R.I64ToI32.evaluate(int(v) & ((1 << 64) - 1)) for v in vals], dtype=np.uint32)
    assert (ref == got).all()
    outside = np.array([1 << 31, -(1 << 31) - 1, 1 << 40, -(1 << 40)], dtype=np.int64)
    ref_o = [R.I64ToI32.evaluate(int(v) & ((1 << 64) - 1)) for v in outside]
    got_o = torch.from_numpy(outside).to(torch.int32).numpy().astype(np.uint32).tolist()
    assert all(r == 0x80000000 for r in ref_o) and got_o != ref_o  # torch wraps; the reference sentinels
    write_record("I64ToI32_v1", "i32", statics={}, oracle_op="torch.Tensor.to(torch.int32)",
                 domain={"in_range": "4096 uniform int64 in [-2^31, 2^31) + 5 boundary values", "outside": outside.tolist()},
                 result={"words": int(vals.size), "exact_words": int(vals.size), "exact_fraction": 1.0, "max_ulp": 0,
                         "outside_range": {"reference": [hex(r) for r in ref_o], "torch_wraps_to": [hex(g) for g in got_o]}},
                 gate_check={"evaluated": "primitive evaluator"},
                 notes="Identical inside int32 range; outside, torch wraps modulo 2^32 and the reference returns the INT32_MIN sentinel.")


# ---------------------------------------------------------------------------------------------------------
# transcendental primitives vs torch (the oracle is <= 1 ulp; recorded)
# ---------------------------------------------------------------------------------------------------------




def test_transcendental_specials():
    assert R.f32_exp_rn_bits(0x7FC00000) == P.F32_NAN and R.f32_exp_rn_bits(0x7F800000) == 0x7F800000 and R.f32_exp_rn_bits(0xFF800000) == 0
    # last finite input / first overflow; torch.exp gives the same two words (measured on the pod, torch 2.4.1 CPU)
    assert R.f32_exp_rn_bits(0x42B17217) == 0x7F7FFF84 and R.f32_exp_rn_bits(0x42B17218) == 0x7F800000
    assert R.f32_exp_rn_bits(0) == 0x3F800000 and R.f32_exp_rn_bits(0x80000000) == 0x3F800000
    for f in (R.f32_erf_rn_bits, R.f32_tanh_rn_bits):
        assert f(0) == 0 and f(0x80000000) == 0x80000000 and f(0x7F800000) == 0x3F800000 and f(0xFF800000) == 0xBF800000
        assert f(0x7FC00000) == P.F32_NAN and f(0x41200000) == 0x3F800000 and f(0xC1200000) == 0xBF800000  # ±10 -> ±1
    assert R.F32Sqrt.evaluate(0xBF800000) == P.F32_NAN and R.F32Sqrt.evaluate(0x80000000) == 0x80000000
    assert R.F32MaxRef.evaluate(0x7FC00000, 0) == P.F32_NAN and R.F32MaxRef.evaluate(0x3F800000, 0x40000000) == 0x40000000
    assert R.F32Sub.evaluate(0x7F800000, 0x7F800000) == P.F32_NAN
    assert R.gather_f32(4).evaluate(7, 1, 2, 3, 4) == P.F32_NAN and R.gather_f32(4).evaluate(2, 1, 2, 3, 4) == 3


# ---------------------------------------------------------------------------------------------------------
# 3. vocabulary pin and codec
# ---------------------------------------------------------------------------------------------------------


def test_ref_vocab_manifest_and_digest():
    m = R.ref_vocab_manifest()
    assert m["version"] == R.REF_VOCAB_VERSION == "verity-ref-vocab/1"
    ids = set(m["kinds"])
    for d in R.REF_PRIMITIVES + R.REF_COMPOSITES:
        assert d.id in ids and d.id in REGISTRY
    assert "GatherBf16x{V}" in m["families"] and "GatherF32x{V}" in m["families"]
    assert all(len(v["evaluator_sha256"]) == 64 for v in m["shared"].values() if "evaluator_sha256" in v)
    d1, d2 = R.ref_vocab_digest(), R.ref_vocab_digest()
    assert d1 == d2 and len(d1) == 64
    assert all(d.conformance.startswith("reference-defined") for d in R.REF_PRIMITIVES + R.REF_COMPOSITES)
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    (RECORD_DIR / "ref-vocab-manifest.json").write_text(json.dumps({"digest": d1, **m}, indent=1, sort_keys=True) + "\n")


def test_codec_roundtrip_and_lazy_gather_f32():
    spec = bind(R.LinearRef, K=16, N=3, DT="bf16", BIAS=True)
    prog = Program(bind(_ST, FN=spec))
    desc = codec.encode_program(prog)
    dec = codec.decode_program(desc, REGISTRY)
    assert dec.gates == prog.circuit.root.gates
    # the on-demand f32 gather family resolves through the extended lazy table even for a vocab never built here
    p = codec._lookup_primitive(REGISTRY, "GatherF32x12345_v1")
    assert p.id == "GatherF32x12345_v1" and p.params[1][1] == __import__("verity.ml.circuits.types", fromlist=["Array"]).Array(12345, P.F32)
    prog2 = Program(bind(_ST, FN=bind(R.EmbeddingRef, V=9, H=3, DT="f32")))
    desc2 = codec.encode_program(prog2)
    assert codec.decode_program(desc2, REGISTRY).gates == prog2.circuit.root.gates


def test_existing_kinds_untouched():
    """Registering the reference vocabulary changes no existing definition object or evaluator."""
    assert P.F32Add.evaluate(0x3F800000, 0x3F800000) == 0x40000000
    assert P.SiluMulBf16.conformance.startswith("exact-model-tested")
    assert b1.Gemm.conformance.startswith("exact-model-tested") and b1.RMSNormTriton.statics == ("N", "EPS")
    assert REGISTRY["Bf16Add_v1"] is P.Bf16Add and REGISTRY["Embedding_v1"] is b1.Embedding
