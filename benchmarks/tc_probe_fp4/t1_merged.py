"""T1's merged k128 route on sm_120 (``probe.py --instruction sp_nvf4_t1``): how many words one 2:4-sparse
``mma.sp::ordered_metadata m16n8k128 kind::mxf4nvf4 .scale_vec::4X .ue4m3`` from C = H reproduces against two chained dense
``mma.sync m16n8k64`` NVF4 atoms from C = H (the first from H, the second from the first's D), on the operands
``fp4_emulation.py routes`` feeds its merged-k128 route.

Operands.  ``fp4_emulation_1ad1aaa2.py`` is bc-f5bf55c8's ``benchmarks/pouw/fp4_emulation.py`` at commit 1ad1aaa2, verbatim
(``EMULATION``; ``emulation_gate`` checks its bytes).  Each family draws what ``routes`` draws, in its order (``family`` and then
Pearl-C4's row-relative noise for A, then for B, from one generator seeded ``--seed``), at routes' size (``ROUTES``).  Two changes
make every aligned 128-span one the sparse instruction takes; both are this capture's shortcut, and the simulation's rates on
routes' own operands are reported beside it:

* A keeps, in each 4-group of k, the one of its two element pairs with the larger energy; the other is zeroed before
  quantization.  So every 4-group holds at most two nonzero codes (D-24's 2:4) and every 8-chunk two of its four pairs (the
  hardware's pair-wise 4:8, metadata ``(p1 << 2) | p0`` with p0 in {0, 1} and p1 in {2, 3});
* A and B take one UE4M3 scale per 32 k (the sparse 4X layout), which the dense atoms read as two equal per-16 scales.

H is ``hot_accumulator(p, h, "row")`` on these operands (one H per row, as T1 defines it).  Per family and accumulator (``h`` for
each ``--hot-bits``, and ``cold``: C = +0, the control):

* ``span``: every (word, aligned 128-span) from C, the sparse word against the dense two-atom chain's word, both hardware; and
  among them the exact spans (the chained word is exactly C plus the span's exact sum), where D-24 derives that they agree;
* ``chain``: the whole k from C, 64 sparse instructions against 128 dense atoms (the simulation's measure);
* ``emulation``, the same operands word for word: ``native`` (the pinned atom chain) against the dense hardware, a check that
  must hold on every word; ``merged_k128`` (one align-add over 8 groups of 16, the simulation's model of the k128 instruction:
  ``sparse.logical_g16``, which the 06:21Z capture refuted) and ``gather4`` (four groups of 32 logical k:
  ``sparse.gather_dense``, that capture's finding) against the sparse hardware; ``merged_k128`` against ``native``, the
  simulation's rate on these operands; and ``merged_k128`` against ``gather4``, where the two k128 models part (a hot H's
  window keeps every group term on the grid, and there they are the same sum);
* ``routes_operands``: the simulation's merged-against-native rates on routes' own operands (dense, per-16 scales).

Gates, before any family is captured: the emulation's bytes, the device UUID, the SASS gate (the sparse kernel's one
``OMMA.SF.SP.168128...UE4M3.4X``, the generic kernel's one ``OMMA.SF.16864...UE4M3.4X``, every kernel of the build one MMA),
the sparse layout check and the dense chain layout checks (2 and 128 atoms) on exact small integers, and the operands' structure.
Validation: the dense hardware words equal ``native`` on every word, and the pinned models (``BLACKWELL_SM120_NVF4``;
``sparse.gather_dense``) on a sample of ``--model-sample`` span words and a sixty-fourth as many whole chains per accumulator.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from emulation import E
import numpy as np
import probe as P
import sparse as S
from verity.ml.circuits.silicon import models as M

HERE = Path(__file__).resolve().parent
EMULATION = HERE / "fp4_emulation_1ad1aaa2.py"
EMULATION_SHA256 = "4b3e0e857bef64d947dd826ec4c509362d98c15249aa028af491040f05f84cae"
EMULATION_ORIGIN = {"path": "benchmarks/pouw/fp4_emulation.py", "commit": "1ad1aaa2479fd976e5da77b71cd14779b0228c0e",
                    "blob": "31259572226ead0860d4c9b778e646aabff8194f", "branch": "cursor/fp4-emulation-cf5b", "lane": "bc-f5bf55c8"}
#: routes' defaults at 1ad1aaa2 (its --seed is passed on the command line: probe.py's own default differs)
ROUTES = {"rows": 64, "cols": 64, "k": 8192, "delta": 0.25, "seed": 20260930,
          "families": ("gaussian", "t4", "massive", "spread", "spread-both")}
SPAN = 128
PTX = {"sparse": S.PTX["sp_nvf4"], "dense": P.PTX["nvf4"]}
INSTRUCTION_ID = "sm120.mma.sp.m16n8k128.e2m1.nvf4.t1-merged"
#: kernel (substring of its SASS function name) -> the one MMA opcode it must hold
SASS_GATE = {S.KERNEL["sp_nvf4"]: S.SASS_OPCODE["sp_nvf4"], f"row_tiles_kernelILi{P.ROW_KIND['nvf4']}EE": "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X"}
GATHER = S.candidates("sp_nvf4")["gather_dense"]
DENSE = M.BLACKWELL_SM120_NVF4


def emulation_gate() -> str | None:
    got = hashlib.sha256(EMULATION.read_bytes()).hexdigest()
    return None if got == EMULATION_SHA256 else f"{EMULATION.name}: sha256 {got} is not 1ad1aaa2's {EMULATION_SHA256}"


# --- operands --------------------------------------------------------------------------------------------------------------------


def quantize(x: np.ndarray, delta: float, rng: np.random.Generator, prune: bool, block: int = 32):
    """``E.quantize_pearl_c4`` with one scale per ``block`` k and, for A (``prune``), one element pair kept per 4-group of k:
    codes (n, k), scales (n, k / block), and the kept pair of each 4-group (n, k / 4; None for B)."""
    s = np.abs(x).max(axis=1, keepdims=True)
    rho = np.sqrt((x[:, ::8] ** 2).mean(axis=1, keepdims=True))
    sigma = delta * rho
    alpha0 = 2688.0 / np.maximum(s + 4 * sigma, 2.0 ** -32)
    v = alpha0 * (x + sigma * rng.standard_normal(x.shape))
    n, k = v.shape
    keep = None
    if prune:
        energy = (v.reshape(n, k // 4, 2, 2) ** 2).sum(axis=3)
        keep = (energy[..., 1] > energy[..., 0]).astype(np.int64)
        kept = np.stack([keep == 0, keep == 1], axis=-1)[..., None]
        v = np.where(np.broadcast_to(kept, (n, k // 4, 2, 2)).reshape(n, k), v, 0.0)
    amax = np.abs(v).reshape(n, k // block, block).max(axis=2)
    sc = np.maximum(E.ue4m3_rn(amax / 6.0), 1)
    ma, ea = E.ue4m3_dyadic(sc)
    codes = E.e2m1_rn(v / np.repeat(ma * np.exp2(ea), block, axis=1))
    return codes.astype(np.uint8), sc.astype(np.uint8), keep


@dataclass
class Operands:
    family: str
    ca: np.ndarray      # (rows, k) logical A codes
    sa: np.ndarray      # (rows, k / 32) UE4M3
    cb: np.ndarray      # (cols, k)
    sb: np.ndarray      # (cols, k / 32)
    meta: np.ndarray    # (rows, k / 8) chunk nibbles
    stored: np.ndarray  # (rows, k / 2) stored A codes
    problem: E.Problem

    @property
    def sa16(self) -> np.ndarray:
        return np.repeat(self.sa, 2, axis=1)

    @property
    def sb16(self) -> np.ndarray:
        return np.repeat(self.sb, 2, axis=1)


def operands(fam: str, seed: int, rows: int = ROUTES["rows"], cols: int = ROUTES["cols"], k: int = ROUTES["k"],
             delta: float = ROUTES["delta"]) -> Operands:
    rng = np.random.default_rng(seed)
    ca, sa, keep = quantize(E.family(fam, "A", rows, k, rng), delta, rng, prune=True)
    cb, sb, _ = quantize(E.family(fam, "B", cols, k, rng), delta, rng, prune=False)
    meta = (((2 + keep[:, 1::2]) << 2) | keep[:, 0::2]).astype(np.uint8)
    ns = k // SPAN
    lc = S.logical_columns(meta.reshape(rows, ns, SPAN // 8))
    stored = np.take_along_axis(ca.reshape(rows, ns, SPAN), lc, axis=-1).reshape(rows, k // 2)
    p = E.Problem.from_codes(ca.astype(np.int64), np.repeat(sa, 2, axis=1).astype(np.int64), cb.astype(np.int64),
                             np.repeat(sb, 2, axis=1).astype(np.int64))
    return Operands(fam, ca, sa, cb, sb, meta, stored, p)


def structure_failures(o: Operands) -> list[str]:
    """What must hold of the operands before they reach the device."""
    rows, k = o.ca.shape
    ns = k // SPAN
    bad = []
    if not np.isin(o.meta, S.META_VALID).all():
        bad.append("metadata nibble outside mma.sp's six")
    back = S.expand(o.stored.reshape(rows, ns, SPAN // 2), o.meta.reshape(rows, ns, SPAN // 8)).reshape(rows, k)
    if not np.array_equal(back, o.ca):
        bad.append("stored A and its metadata do not expand to the logical A")
    if (((o.ca.reshape(rows, k // 4, 4) & 7) != 0).sum(axis=2) > 2).any():
        bad.append("a 4-group of A holds more than two nonzero codes (not D-24's 2:4)")
    if not (np.isin(o.sa, P.UE4M3_VALID).all() and np.isin(o.sb, P.UE4M3_VALID).all()):
        bad.append("a scale is not a finite UE4M3 code")
    if not (o.problem.x[0::2] == o.problem.x[1::2]).all():
        bad.append("the per-16 exponents of a 32-k scale block differ")
    return [f"{o.family}: {b}" for b in bad]


# --- tiles -----------------------------------------------------------------------------------------------------------------------


def _tiles(x: np.ndarray, per: int, side: str, ns: int, rb: int, cb: int) -> np.ndarray:
    """(n_side, ns * w) -> (ns * rb * cb, per, w), tiles ordered (span, row block, column block)."""
    w = x.shape[1] // ns
    t = x.reshape(-1, per, ns, w).transpose(2, 0, 1, 3)
    t = t[:, :, None] if side == "A" else t[:, None]
    return np.ascontiguousarray(np.broadcast_to(t, (ns, rb, cb, per, w))).reshape(-1, per, w)


def _acc_tiles(C: np.ndarray, ns: int) -> np.ndarray:
    rows, cols = C.shape
    t = C.reshape(rows // 16, 16, cols // 8, 8).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(np.broadcast_to(t[None], (ns, *t.shape))).reshape(-1, 16, 8)


def _words(D: np.ndarray, ns: int, rows: int, cols: int) -> np.ndarray:
    return D.reshape(ns, rows // 16, cols // 8, 16, 8).transpose(0, 1, 3, 2, 4).reshape(ns, rows, cols)


def hardware(sparse_dev, dense_dev, o: Operands, C: np.ndarray) -> dict:
    """The device's words from C (rows, cols): per span (ns, rows, cols) and over the whole k (rows, cols), sparse and dense."""
    rows, k = o.ca.shape
    cols = o.cb.shape[0]
    ns, rb, cb = k // SPAN, rows // 16, cols // 8
    As = _tiles(o.stored, 16, "A", ns, rb, cb)
    META = _tiles(o.meta, 16, "A", ns, rb, cb)
    A = _tiles(o.ca, 16, "A", ns, rb, cb)
    B = _tiles(o.cb, 8, "B", ns, rb, cb)
    SFA, SFB = _tiles(o.sa, 16, "A", ns, rb, cb), _tiles(o.sb, 8, "B", ns, rb, cb)
    SFA16, SFB16 = _tiles(o.sa16, 16, "A", ns, rb, cb), _tiles(o.sb16, 8, "B", ns, rb, cb)
    Ct = _acc_tiles(C, ns)
    t0 = time.time()
    span_sparse = sparse_dev.tiles("sp_nvf4", As, B, META, SFA, SFB, Ct)
    span_dense = dense_dev.tiles("nvf4", A, B, Ct, SFA16, SFB16, steps=2)
    per = rb * cb
    c = Ct[:per]
    for s in range(ns):
        sl = slice(s * per, (s + 1) * per)
        c = sparse_dev.tiles("sp_nvf4", As[sl], B[sl], META[sl], SFA[sl], SFB[sl], c)
    chain_sparse = c
    chain_dense = dense_dev.tiles("nvf4", _tiles(o.ca, 16, "A", 1, rb, cb), _tiles(o.cb, 8, "B", 1, rb, cb), Ct[:per],
                                  _tiles(o.sa16, 16, "A", 1, rb, cb), _tiles(o.sb16, 8, "B", 1, rb, cb), steps=k // 64)
    return {"span_sparse": _words(span_sparse, ns, rows, cols), "span_dense": _words(span_dense, ns, rows, cols),
            "chain_sparse": _words(chain_sparse, 1, rows, cols)[0], "chain_dense": _words(chain_dense, 1, rows, cols)[0],
            "gpu_seconds": round(time.time() - t0, 2)}


# --- the emulation's models on the same operands -----------------------------------------------------------------------------


def _gather4(c, m, x, part):
    """sparse.gather_dense vectorised: four groups of 32 logical k, each the sum of two per-16 group dots under one scale pair."""
    return E.nvf4_step(c, [m[2 * j] + m[2 * j + 1] for j in range(4)], [x[2 * j] for j in range(4)],
                       [part[2 * j] | part[2 * j + 1] for j in range(4)])


def emulation(p: E.Problem, C: np.ndarray, gather: bool = True) -> dict:
    """``native``, ``merged_k128`` and (``gather``) ``gather4`` per span from C and over the whole k from C."""
    nb = p.mant.shape[0]
    ns = nb // 8
    m, x, part = (a.reshape(ns, 8, *a.shape[1:]).swapaxes(0, 1) for a in (p.mant, p.x, p.part))
    Cs = np.broadcast_to(C, (ns, *C.shape))
    out = {"span_native": E.nvf4_step(E.nvf4_step(Cs, list(m[:4]), list(x[:4]), list(part[:4])), list(m[4:]), list(x[4:]),
                                      list(part[4:])),
           "span_merged_k128": E.nvf4_step(Cs, list(m), list(x), list(part)),
           "chain_native": p.native(C), "chain_merged_k128": E.merged_k128(p, C)}
    if gather:
        out["span_gather4"] = _gather4(Cs, m, x, part)
        c = C.copy()
        for s in range(ns):
            c = _gather4(c, m[:, s], x[:, s], part[:, s])
        out["chain_gather4"] = c
    return out


def exact_spans(p: E.Problem, C: np.ndarray, words: np.ndarray) -> np.ndarray:
    """(ns, rows, cols): the span's word is exactly C plus the span's exact sum (every value a multiple of 2^-20)."""
    sums = p.step_sums()
    s = sums.reshape(sums.shape[0] // 2, 2, *sums.shape[1:]).sum(axis=1)
    cm, ce, _ = E.f32_dyadic(C)
    cint = np.where(cm == 0, 0, cm << np.clip(ce + 20, 0, 62))
    if ((cm != 0) & (ce + 20 < 0)).any():
        raise ValueError("an accumulator below 2^-20")
    want = cint[None] + s
    m, e, _ = E.f32_dyadic(words)
    sh = e + 20
    mag = np.abs(m)
    left_ok = (sh >= 0) & (sh < 39)
    right_ok = (sh < 0) & ((mag & ((np.int64(1) << np.clip(-sh, 0, 62)) - 1)) == 0)
    val = np.where(sh >= 0, mag << np.clip(sh, 0, 38), mag >> np.clip(-sh, 0, 62))
    val = np.where(m < 0, -val, val)
    return np.where(sh >= 0, left_ok, right_ok) & (val == want)


# --- the pinned models on a sample -------------------------------------------------------------------------------------------


def pinned_sample(o: Operands, C: np.ndarray, hw: dict, n_span: int, n_chain: int, rng: np.random.Generator) -> dict:
    """``BLACKWELL_SM120_NVF4`` against the dense hardware words and ``sparse.gather_dense`` against the sparse ones, on sampled
    (span, row, column) words and whole chains."""
    rows, k = o.ca.shape
    cols, ns = o.cb.shape[0], k // SPAN
    ca, cb, sa, sb = o.ca.tolist(), o.cb.tolist(), o.sa16.tolist(), o.sb16.tolist()
    dense_bad = sparse_bad = 0
    first: list = []
    for f in rng.choice(ns * rows * cols, size=min(n_span, ns * rows * cols), replace=False).tolist():
        s, i, j = f // (rows * cols), (f // cols) % rows, f % cols
        a, b = ca[i][SPAN * s:SPAN * (s + 1)], cb[j][SPAN * s:SPAN * (s + 1)]
        xa, xb = sa[i][8 * s:8 * (s + 1)], sb[j][8 * s:8 * (s + 1)]
        c0 = int(C[i, j])
        d = DENSE.step_scaled(DENSE.step_scaled(c0, a[:64], b[:64], xa[:4], xb[:4]), a[64:], b[64:], xa[4:], xb[4:])
        g = GATHER.step_scaled(c0, a, b, xa, xb)
        dense_bad += d != int(hw["span_dense"][s, i, j])
        sparse_bad += g != int(hw["span_sparse"][s, i, j])
        if (d != int(hw["span_dense"][s, i, j]) or g != int(hw["span_sparse"][s, i, j])) and len(first) < 8:
            first.append({"span": s, "row": i, "col": j, "acc": f"{c0:08x}", "dense_model": f"{d:08x}",
                          "dense_hw": f"{int(hw['span_dense'][s, i, j]):08x}", "gather_dense": f"{g:08x}",
                          "sparse_hw": f"{int(hw['span_sparse'][s, i, j]):08x}"})
    chain_dense_bad = chain_sparse_bad = 0
    for f in rng.choice(rows * cols, size=min(n_chain, rows * cols), replace=False).tolist():
        i, j = f // cols, f % cols
        c0 = int(C[i, j])
        d = DENSE.chain(c0, ca[i], cb[j], sa[i], sb[j])
        g = c0
        for s in range(ns):
            g = GATHER.step_scaled(g, ca[i][SPAN * s:SPAN * (s + 1)], cb[j][SPAN * s:SPAN * (s + 1)], sa[i][8 * s:8 * (s + 1)],
                                   sb[j][8 * s:8 * (s + 1)])
        chain_dense_bad += d != int(hw["chain_dense"][i, j])
        chain_sparse_bad += g != int(hw["chain_sparse"][i, j])
    return {"span_words": min(n_span, ns * rows * cols), "dense_mismatches": dense_bad, "gather_dense_mismatches": sparse_bad,
            "chain_words": min(n_chain, rows * cols), "chain_dense_mismatches": chain_dense_bad,
            "chain_gather_dense_mismatches": chain_sparse_bad, "first": first}


# --- devices for a dry run -----------------------------------------------------------------------------------------------------


def _terms(A, B, SA16, SB16):
    """Codes (n,16,K) / (n,8,K) and per-16 scales -> each block's (mant, x, part), each (K/16, n, 16, 8)."""
    ha, hb = E.halves(np.asarray(A, np.int64)), E.halves(np.asarray(B, np.int64))
    n, _, k = ha.shape
    nb = k // 16
    a4, b4 = ha.reshape(n, 16, nb, 16).astype(np.float64), hb.reshape(n, 8, nb, 16).astype(np.float64)
    gd = np.rint(np.einsum("nigk,njgk->gnij", a4, b4)).astype(np.int64)
    nz = np.einsum("nigk,njgk->gnij", np.abs(a4), np.abs(b4)) > 0
    maa, eaa = E.ue4m3_dyadic(np.asarray(SA16, np.int64))
    mab, eab = E.ue4m3_dyadic(np.asarray(SB16, np.int64))
    mm = np.einsum("nig,njg->gnij", maa, mab)
    x = eaa.transpose(2, 0, 1)[:, :, :, None] + eab.transpose(2, 0, 1)[:, :, None, :] - 2
    return gd * mm, x, nz & (mm != 0)


class FakeDenseT1:
    """CPU stand-in for the dense rows (the pinned atom, vectorised): ``tiles`` as ``probe.Probe.tiles`` for ``nvf4``."""

    tcgen05 = False

    def tiles(self, instr, A, Bt, C, SFA=None, SFB=None, steps: int = 1, **_kw):
        assert instr == "nvf4"
        m, x, part = _terms(A, Bt, SFA, SFB)
        c = np.asarray(C, np.uint32)
        for t in range(m.shape[0] // 4):
            b = slice(4 * t, 4 * t + 4)
            c = E.nvf4_step(c, list(m[b]), list(x[b]), list(part[b]))
        return c


class FakeSparseT1:
    """CPU stand-in for the sparse row: ``gather4`` (the 06:21Z finding) or ``merged_k128`` (the simulation's model)."""

    tcgen05 = False

    def __init__(self, model: str):
        self.model = model

    def identity(self) -> dict:
        return {"device": f"FAKE SILICON ({self.model})", "compute_capability": "fake", "driver": None, "gpu_uuid": "GPU-fake",
                "backend": "fake", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    def clocks(self, uuid):
        return None

    def tiles(self, instr, As, B, META, SFA, SFB, C):
        m, x, part = _terms(S.expand(np.asarray(As), META), B, S.model_scales(SFA), S.model_scales(SFB))
        c = np.asarray(C, np.uint32)
        if self.model == "gather4":
            return _gather4(c, m, x, part)
        return E.nvf4_step(c, list(m), list(x), list(part))


FAKES = ("gather4", "merged_k128")


# --- driver ----------------------------------------------------------------------------------------------------------------------


def _count(a, b) -> int:
    return int(np.count_nonzero(a == b))


def main(args) -> int:
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    arch = args.arch or "sm_120a"
    if args.chain != 1:
        raise SystemExit("--chain is not defined for sp_nvf4_t1 (its chains are the whole k)")
    if args.fake_silicon and os.environ.get("RESEARCH_RUN_ID"):
        raise SystemExit("--fake-silicon is a dry run and never publishes: not under a research run")
    if args.fake_silicon and args.fake_silicon not in FAKES:
        raise SystemExit(f"--fake-silicon for sp_nvf4_t1: one of {FAKES}")
    fams = args.families.split(",") if args.families else list(ROUTES["families"])
    hot_bits = [int(h) for h in args.hot_bits.split(",") if h]
    settings = ["cold"] + [f"h{h}" for h in hot_bits]
    if args.fake_silicon and not P.shutil.which(P.nvcc_bin()):
        so, nvcc = None, "not built (fake silicon)"
    else:
        so, nvcc = P.build(arch, out, "mma_fp4.cu")
    sparse_dev = FakeSparseT1(args.fake_silicon) if args.fake_silicon else (None if args.sass_only else S.SparseDevice(so))
    dense_dev = FakeDenseT1() if args.fake_silicon else (None if args.sass_only else P.Probe(so))
    identity = sparse_dev.identity() if sparse_dev is not None else {"device": None, "gpu_uuid": None, "driver": None,
                                                                     "compute_capability": None, "backend": "none (--sass-only)"}
    identity.update(nvcc=nvcc, arch=arch)
    results: dict = {"schema": "pouw-rtxpro/tc-probe-fp4-t1-merged/v1", "instruction": PTX, "instruction_id": INSTRUCTION_ID,
                     "identity": identity, "emulation": {**EMULATION_ORIGIN, "sha256": EMULATION_SHA256, "file": EMULATION.name},
                     "operands": {**{k: v for k, v in ROUTES.items() if k != "families"}, "seed": args.seed, "families": fams,
                                  "hot_bits": hot_bits, "hot_rule": "row",
                                  "shortcut": "A keeps one element pair per 4-group (the larger energy), zeroed before quantization; "
                                              "A and B take one UE4M3 scale per 32 k"},
                     "fake_silicon": args.fake_silicon, "corrupt_word": bool(args.corrupt_word)}
    failures: list[str] = []

    def finish() -> int:
        results["seconds"] = round(time.time() - t_all, 1)
        results["failures"] = failures
        (out / "probe_results.json").write_text(json.dumps(results, indent=1))
        return 1 if failures else 0

    bad = emulation_gate()
    if bad:
        failures.append(bad)
        print("FAILURES", failures, flush=True)
        return finish()
    if args.expect_uuid and sparse_dev is not None and not args.fake_silicon and not P.uuid_allowed(identity.get("gpu_uuid"), args.expect_uuid):
        failures.append(f"device UUID {identity.get('gpu_uuid')} != expected {args.expect_uuid}: stopped before any measurement")
        print("FAILURES", failures, flush=True)
        finish()
        return 2
    print("device", json.dumps(identity), flush=True)
    if so is not None and (args.sass_check or args.sass_only):
        sass = P.sass_check(so)
        results["sass"] = sass
        for kern, rec in sass.items():
            if rec["mma_count"] != 1:
                failures.append(f"sass: {kern} has {rec['mma_count']} MMA instructions")
        for kern, opcode in SASS_GATE.items():
            mine = [rec for name, rec in sass.items() if kern in name]
            if len(mine) != 1 or mine[0]["opcodes"] != [opcode]:
                failures.append(f"sass: {kern} must hold exactly one {opcode}: {mine}")
        print("sass", json.dumps({k: v for k, v in sass.items() if any(g in k for g in SASS_GATE)}), flush=True)
    if args.sass_only or failures:
        print("FAILURES" if failures else "SASS GATE PASSED", failures, flush=True)
        return finish()

    results["clocks_start"] = sparse_dev.clocks(identity.get("gpu_uuid"))
    try:
        results["layout_check"] = {"sparse": S.layout_check(sparse_dev, "sp_nvf4", out),
                                   "dense_chain_2": P.chain_layout_check(dense_dev, "nvf4", 2),
                                   "dense_chain_128": P.chain_layout_check(dense_dev, "nvf4", ROUTES["k"] // 64)}
    except RuntimeError as e:
        failures.append(str(e))
        print("FAILURES", failures, flush=True)
        return finish()
    print("layout_check", json.dumps(results["layout_check"]), flush=True)

    ops = {fam: operands(fam, args.seed) for fam in fams}
    for o in ops.values():
        failures += structure_failures(o)
    if failures:
        print("FAILURES", failures, flush=True)
        return finish()

    rng = np.random.default_rng(args.seed + 1)
    fam_results: dict = {}
    corrupted = False
    for fam, o in ops.items():
        rows, k = o.ca.shape
        cols = o.cb.shape[0]
        orig = E.Problem(fam, rows, cols, k, ROUTES["delta"], args.seed)
        accs = {"cold": np.zeros((rows, cols), np.uint32)}
        accs.update({f"h{h}": E.hot_accumulator(o.problem, h, "row") for h in hot_bits})
        saved: dict = {"stored_a": o.stored, "meta": o.meta, "b": o.cb, "sfa": o.sa, "sfb": o.sb}
        fr: dict = {}
        for name in settings:
            C = accs[name]
            hw = hardware(sparse_dev, dense_dev, o, C)
            if args.corrupt_word and not corrupted:
                hw["span_dense"] = hw["span_dense"].copy()
                hw["span_dense"][0, 0, 0] ^= 1
                corrupted = True
            t0 = time.time()
            em = emulation(o.problem, C)
            exact = exact_spans(o.problem, C, hw["span_dense"])
            same = hw["span_sparse"] == hw["span_dense"]
            r = {"span": {"words": int(same.size), "matched": int(same.sum()), "exact_spans": int(exact.sum()),
                          "matched_exact": int((same & exact).sum()), "matched_inexact": int((same & ~exact).sum())},
                 "chain": {"words": int(hw["chain_sparse"].size), "matched": _count(hw["chain_sparse"], hw["chain_dense"])},
                 "emulation": {
                     "native_vs_dense_hw": {"span": _count(em["span_native"], hw["span_dense"]),
                                            "chain": _count(em["chain_native"], hw["chain_dense"])},
                     "gather4_vs_sparse_hw": {"span": _count(em["span_gather4"], hw["span_sparse"]),
                                              "chain": _count(em["chain_gather4"], hw["chain_sparse"])},
                     "merged_k128_vs_sparse_hw": {"span": _count(em["span_merged_k128"], hw["span_sparse"]),
                                                  "chain": _count(em["chain_merged_k128"], hw["chain_sparse"])},
                     "merged_k128_vs_native": {"span": _count(em["span_merged_k128"], em["span_native"]),
                                               "chain": _count(em["chain_merged_k128"], em["chain_native"])},
                     # where the two k128 models agree, these operands cannot tell them apart
                     "merged_k128_vs_gather4": {"span": _count(em["span_merged_k128"], em["span_gather4"]),
                                                "chain": _count(em["chain_merged_k128"], em["chain_gather4"])}},
                 "gpu_seconds": hw["gpu_seconds"]}
            if name != "cold":
                eh = ((C.astype(np.int64) >> 23) & 0xFF) - 127
                r["h_exponent"] = {"min": int(eh.min()), "max": int(eh.max())}
            diff = np.argwhere(~same)[:6]
            r["first_differing"] = [{"span": int(s), "row": int(i), "col": int(j), "acc": f"{int(C[i, j]):08x}",
                                     "sparse_hw": f"{int(hw['span_sparse'][s, i, j]):08x}", "dense_hw": f"{int(hw['span_dense'][s, i, j]):08x}",
                                     "merged_k128": f"{int(em['span_merged_k128'][s, i, j]):08x}",
                                     "gather4": f"{int(em['span_gather4'][s, i, j]):08x}"} for s, i, j in diff]
            r["pinned_sample"] = pinned_sample(o, C, hw, args.model_sample, max(1, args.model_sample // 64), rng)
            C_orig = np.zeros((rows, cols), np.uint32) if name == "cold" else E.hot_accumulator(orig, int(name[1:]), "row")
            eo = emulation(orig, C_orig, gather=False)
            r["routes_operands"] = {"span_words": int(eo["span_native"].size), "chain_words": int(eo["chain_native"].size),
                                    "merged_k128_vs_native": {"span": _count(eo["span_merged_k128"], eo["span_native"]),
                                                              "chain": _count(eo["chain_merged_k128"], eo["chain_native"])}}
            r["cpu_seconds"] = round(time.time() - t0, 1)
            for key in ("span_sparse", "span_dense", "chain_sparse", "chain_dense"):
                saved[f"{name}_{key}"] = hw[key]
            saved[f"{name}_acc"] = C
            fr[name] = r
            span, chain = r["span"], r["chain"]
            print(fam, name, f"span {span['matched']}/{span['words']} (exact {span['matched_exact']}/{span['exact_spans']}),",
                  f"chain {chain['matched']}/{chain['words']};", "emulation", json.dumps(r["emulation"]),
                  "pinned", json.dumps({k: v for k, v in r["pinned_sample"].items() if k != "first"}), flush=True)
        np.savez_compressed(out / f"t1_{fam}.npz", **saved)
        fam_results[fam] = fr
    results["families"] = fam_results

    for fam, fr in fam_results.items():
        for name, r in fr.items():
            nat = r["emulation"]["native_vs_dense_hw"]
            if nat["span"] != r["span"]["words"] or nat["chain"] != r["chain"]["words"]:
                failures.append(f"{fam}/{name}: the dense hardware words differ from the pinned atom chain (native) on "
                                f"{r['span']['words'] - nat['span']} span and {r['chain']['words'] - nat['chain']} chain words")
            ps = r["pinned_sample"]
            if ps["dense_mismatches"] or ps["chain_dense_mismatches"]:
                failures.append(f"{fam}/{name}: BLACKWELL_SM120_NVF4 misses {ps['dense_mismatches']} sampled span and "
                                f"{ps['chain_dense_mismatches']} chain words of the dense hardware")
    results["finding"] = {
        name: {fam: {"span": [fr[name]["span"]["matched"], fr[name]["span"]["words"]],
                     "chain": [fr[name]["chain"]["matched"], fr[name]["chain"]["words"]]} for fam, fr in fam_results.items()}
        for name in settings}
    results["clocks_end"] = sparse_dev.clocks(identity.get("gpu_uuid"))
    rc = finish()
    run_id = os.environ.get("RESEARCH_RUN_ID")
    if run_id:
        result = {
            "schema": "research/result/v0.1", "kind": "tc-evidence/v1", "run_id": run_id, "instruction": INSTRUCTION_ID, "ptx": PTX,
            "arch": arch, "sku": identity["device"], "compute_capability": identity.get("compute_capability"),
            "driver": identity.get("driver"), "gpu_uuid": identity.get("gpu_uuid"), "nvcc": nvcc,
            "cuda_visible_devices": identity.get("cuda_visible_devices"),
            "sass_opcodes": {k: sorted(set(v["opcodes"])) for k, v in results.get("sass", {}).items()},
            "emulation": results["emulation"], "operands": results["operands"], "layout_check": results["layout_check"],
            "finding": results["finding"],
            "families": {fam: {name: {k: v for k, v in r.items() if k not in ("first_differing",)} for name, r in fr.items()}
                         for fam, fr in fam_results.items()},
            "clocks": {"start": results.get("clocks_start"), "end": results.get("clocks_end"),
                       "note": "recorded, not locked: a capture's words do not depend on the clock"},
            "workload_fingerprint": {"probe": "benchmarks/tc_probe_fp4/t1_merged.py", "instruction": INSTRUCTION_ID, "arch": arch,
                                     "device": identity["device"], "seed": args.seed, "families": fams, "hot_bits": hot_bits},
            "validation": {"status": "passed" if not failures else "failed",
                           "checker": "tc_probe_fp4 t1_merged.py: emulation bytes, UUID, SASS, exact-integer layout and operand "
                                      "structure gates; the dense words against the pinned atom chain",
                           "detail": "; ".join(failures) or "capture recorded; the dense hardware words are the pinned atom chain's",
                           "evidence": ["probe_results.json"]},
            "measurements": [{"name": "wall", "value": results["seconds"], "unit": "s"}],
            "measurement_files": ["probe_results.json"],
            "artifacts": sorted(p.name for p in out.glob("t1_*.npz")),
        }
        (out / "result.json.tmp").write_text(json.dumps(result, indent=1))
        os.replace(out / "result.json.tmp", out / "result.json")
    print("FAILURES" if failures else "CAPTURE RECORDED", failures, f"{results['seconds']}s", flush=True)
    return rc
