"""One NVRTC kernel per system for the hint matrices of the v2 (``pubsel``) and v3 (``privsel``) units.

The torch generators (``pubsel/hints.py``, ``privsel/hints.py``: ``_hints_eager``) launch several small kernels per hint row
per group -- a CUDA graph of ~10^4 nodes for a folded v3 unit, whose replay cost is per group and so does not fall with the
sub-batch count.  Here one thread per unit (column) runs the same integer program, int64 throughout with torch's shift
semantics (a left shift by < 0 or >= 64 is 0; a right shift by < 0 or >= 63 is the sign), and stores each hint through a row
map into ``sys.hints`` order.  One launch per sub-batch, no static buffers (the output is allocated per call on the current
stream), so sub-batches pipelined on different streams share nothing but the read-only row map.  The row map is built by
walking the generator's emission order in Python (``_order_v2`` / ``_order_v3``, the kernel's loops name by name) and must
cover every hint row exactly once.  ``hints_fused_test.py`` checks byte identity against the torch path on real sub-batches
and on adversarial words (every word of the operand range, every FP32 class of the accumulator)."""
from __future__ import annotations

import os

import numpy as np

from .compile import System
from .pubsel.relation import ACC_FRAC, LIMB, plan

__all__ = ["FusedHints", "enabled", "fused_hints_for"]

THREADS = 64

_SRC = r"""
typedef long long i64;
typedef unsigned long long u64;

__device__ __constant__ int GSIZE[NG] = { GSIZES };

__device__ __forceinline__ i64 shl(i64 a, i64 b) { return (b < 0 || b >= 64) ? 0 : (i64)((u64)a << b); }
__device__ __forceinline__ i64 shr(i64 a, i64 b) { return (b < 0 || b >= 63) ? (a >> 63) : (a >> b); }
__device__ __forceinline__ i64 fdiv(i64 a, i64 b) { i64 q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
__device__ __forceinline__ i64 fmodp(i64 a, i64 b) { i64 r = a % b; return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r; }
__device__ __forceinline__ i64 imax(i64 a, i64 b) { return a > b ? a : b; }
__device__ __forceinline__ i64 imin(i64 a, i64 b) { return a < b ? a : b; }
__device__ __forceinline__ i64 bitlen(i64 v) { i64 l = 0; for (int k = 0; k <= SB; ++k) l += (v >= (1LL << k)); return l; }

#define EMIT(val) do { const int r_ = __ldg(rowmap + s_); if (r_ >= 0) out[(i64)r_ * ncol + col] = (i64)(val); ++s_; } while (0)
#define LMASK ((1LL << LIMB) - 1)

struct Dec { i64 s, m, e, t, f, nz; };
__device__ __forceinline__ Dec decode(i64 w) {
  Dec d;
  d.t = shr(w, OP_FRAC) & ((1LL << OP_EXP) - 1);
  d.f = w & ((1LL << OP_FRAC) - 1);
  d.nz = d.t != 0;
  d.s = shr(w, OP_WORD_BITS - 1);
  d.m = d.f + d.nz * (1LL << OP_FRAC);
  d.e = d.t - d.nz - (OP_BIAS - 1);
  return d;
}

extern "C" __global__ void hints_fused(const i64* __restrict__ C, const i64* __restrict__ A, const i64* __restrict__ B,
                                       const int* __restrict__ rowmap, i64* __restrict__ out, const int ncol) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= ncol) return;
  int s_ = 0;
  const i64 c = C[col];
  const i64 t = shr(c, ACC_FRAC) & ((1LL << ACC_EXP) - 1);
  const i64 f = c & ((1LL << ACC_FRAC) - 1);
  const i64 nz = t != 0;
  const i64 z = (t == 0) && (f == 0);
  const i64 s_c = shr(c, 31);
#if CHAIN
  EMIT(s_c); EMIT(t); EMIT(f); EMIT(nz); EMIT(z);
#endif
  const i64 M24 = f + nz * (1LL << ACC_FRAC);
  const i64 e = t - nz - (ACC_BIAS - 1);
  i64 s_st = s_c, e_st = e + z * (FLOOR - e), Mt = shr(M24, DROP);
  const i64* a = A + (i64)col * K;
  const i64* b = B + (i64)col * K;
#if V3
  for (int w = 0; w < 2; ++w) {
    const i64* x = w ? b : a;
    for (int i = 0; i < K; ++i) {
      const Dec d = decode(x[i]);
      EMIT(d.s); EMIT(d.nz);
      const i64 xx = d.t - d.nz;
      for (int j = 0; j < OP_EXP; ++j) EMIT(shr(xx, j) & 1);
      for (int j = 0; j < OP_FRAC; ++j) EMIT(shr(d.f, j) & 1);
    }
  }
#endif
  i64 fo = 0;
  int start = 0;
  for (int g = 0; g < NG; ++g) {
    const int size = GSIZE[g];
    i64 u[MAXG], ep[MAXG], sg[MAXG];
    for (int j = 0; j < size; ++j) {
      const Dec da = decode(a[start + j]), db = decode(b[start + j]);
      u[j] = da.m * db.m;
      sg[j] = fmodp(da.s + db.s, 2);
#if V3
      ep[j] = da.e + db.e;
#else
      ep[j] = u[j] == 0 ? (i64)FLOOR : da.e + db.e;
#endif
    }
#if V3
    const i64 NEG = -(1LL << 20);
    i64 E = e_st;
    for (int j = 0; j < size; ++j) E = imax(E, u[j] != 0 ? ep[j] : NEG);
    EMIT(E);
    i64 E_eff = E;
#if HI
    EMIT(E >= HI_THR);
    E_eff = imin(E, (i64)HI_THR);
#endif
    int which = -1;
    if (e_st == E) which = 0;
    else for (int j = 0; j < size; ++j) if ((u[j] != 0 ? ep[j] : NEG) == E) { which = j + 1; break; }
    for (int j = 0; j <= size; ++j) EMIT(j == which);
    i64 Q = 0, SL = 0, Cv = (which == 0);
    for (int j = 0; j < size; ++j) {
      const i64 d = E_eff - ep[j];
      const bool zero = u[j] == 0;
      const bool dead = (d >= PB) && !zero;
      const bool live = !dead && !zero;
      const i64 Lq = PS - d;
      const i64 cfl = fdiv(Lq, RB);
      const i64 R = live ? Lq - RB * cfl : (dead ? RB - 1 - fmodp(d - PB, RB) : 0);
      const i64 Kq = dead ? fdiv(d - PB, RB) : 0;
      const i64 ci = live ? 1 - cfl : (dead ? 4 : 1);
      const i64 Y = shl(u[j], R);
      i64 X = (d <= PS) ? shl(u[j], imax(PS - d, 0)) : ((d < PB) ? shr(u[j], imax(d - PS, 0)) : 0);
      if (zero) X = 0;
      for (int jj = 0; jj < UB; ++jj) EMIT(shr(u[j], jj) & 1);
      for (int r = 0; r < RB; ++r) EMIT(R == r);
      for (int kk = 0; kk < ND; ++kk) EMIT(shr(Y, kk * RB) & ((1LL << RB) - 1));
      for (int cc = 0; cc < NCOARSE; ++cc) EMIT(ci == cc);
#if WIDE
      EMIT(X & LMASK); EMIT(shr(X, LIMB));
#else
      EMIT(X);
#endif
      for (int jj = 0; jj < KB; ++jj) EMIT(shr(Kq, jj) & 1);
      const i64 sgn1 = 1 - 2 * sg[j];
#if WIDE
      EMIT(sgn1 * (X & LMASK)); EMIT(sgn1 * shr(X, LIMB));
#else
      EMIT(sgn1 * X);
#endif
      if (which == j + 1) Cv += X;
      Q += sgn1 * X;
      SL += sgn1 * (X & LMASK);
    }
    EMIT(Cv);
#else
    i64 G = ep[0];
    for (int j = 1; j < size; ++j) G = imax(G, ep[j]);
    G = imax(G, (i64)FLOOR);
    const i64 E = imax(G, e_st);
    EMIT(E - G);
    const i64 kc = imin(imax(e_st - G, (i64)-W), (i64)PB);
    for (int i = 0; i < W + PB + 1; ++i) EMIT(kc == i - W);
    i64 Q = 0;
    for (int j = 0; j < size; ++j) {
      const i64 d = E - ep[j];
      const i64 al = (d <= PS) ? shl(u[j], imax(PS - d, 0)) : ((d < PB) ? shr(u[j], imax(d - PS, 0)) : 0);
      Q += (1 - 2 * sg[j]) * al;
    }
#endif
    const i64 dacc = E - e_st;
    const i64 q_acc = shr(shl(Mt, SS), imin(dacc, (i64)W));
    const i64 sign = 1 - 2 * s_st;
    const i64 S = sign * q_acc + Q;
    const i64 sgn = S < 0;
    const i64 v = S < 0 ? -S : S;
    EMIT(sgn);
#if WIDE
    const i64 vL = v & LMASK;
    EMIT(vL); EMIT(shr(v, LIMB));
    const i64 qL = q_acc & LMASK;
    EMIT(qL); EMIT(shr(q_acc, LIMB));
#if V3
    EMIT(shr(sign * qL + SL - (1 - 2 * sgn) * vL, LIMB));
#else
    const i64 QL = Q & LMASK;
    EMIT(QL); EMIT(shr(Q - QL, LIMB));
    EMIT(shr(sign * qL + QL - (1 - 2 * sgn) * vL, LIMB));
#endif
#else
    EMIT(v); EMIT(q_acc);
#if !V3
    EMIT(Q);
#endif
#endif
    const i64 l = bitlen(v);
    const i64 e1 = imax(E + l - WIDTH, (i64)ACC_E_MIN);
#if V3
    const i64 tt = e1 - E + WIDTH - NB;
#else
    const i64 tt = v != 0 ? e1 - E + WIDTH - NB : (i64)T_LO;
#endif
    const i64 Mn = tt >= 0 ? shr(v, imax(tt, 0)) : shl(v, imax(-tt, 0));
    for (int i = 0; i < N_T; ++i) EMIT(tt == T_LO + i);
    const i64 hid = shr(Mn, NB - 1);
    fo = Mn & ((1LL << (NB - 1)) - 1);
    EMIT(hid); EMIT(fo);
#if !V3
    EMIT(Mn);
#endif
    const i64 zo = Mn == 0;
    EMIT(zo);
#if V3
    EMIT(e1 > ACC_E_MAX);
#else
    EMIT((e1 > ACC_E_MAX) && (zo == 0));
#endif
    s_st = sgn * (1 - zo);
    e_st = e1 + zo * (FLOOR - e1);
    Mt = Mn;
    start += size;
  }
#if CHAIN && !FP8
  {
    const int Le = ACC_FRAC - FRAC;
    const i64 sticky = (fo & ((1LL << (Le - 1)) - 1)) + (shr(fo, Le) & 1);
#if V3
    EMIT(sticky == 0);
#else
    EMIT(shr(fo, Le)); EMIT(shr(fo, Le - 1) & 1); EMIT(sticky); EMIT(sticky == 0);
#endif
  }
#endif
  (void)fo;
}
"""


# ---------------------------------------------------------------------------------------------------------------------
# The kernel's emission order, name by name (the same loops as the source above)
# ---------------------------------------------------------------------------------------------------------------------

def _order(family: str, p, pl, sp, chain: bool) -> list[str]:
    v3 = family == "v3"
    o: list[str] = []
    if chain:
        o += ["acc.s", "acc.t", "acc.f", "acc.nz", "acc.z"]
    k = sum(pl.groups)
    if v3:
        for name in ("a_words", "b_words"):
            for i in range(k):
                o += [f"{name}[{i}].s", f"{name}[{i}].nz"]
                o += [f"{name}[{i}].x[{j}]" for j in range(pl.op_exp)]
                o += [f"{name}[{i}].f[{j}]" for j in range(pl.op_frac)]
    start = 0
    for g, size in enumerate(pl.groups):
        tag = f"g{g}"
        if v3:
            o.append(f"{tag}.E")
            if sp.hi_thr is not None:
                o.append(f"{tag}.hi")
            o += [f"{tag}.which[{j}]" for j in range(size + 1)]
            for i in range(start, start + size):
                s = f"{tag}.p[{i}]"
                o += [f"{s}.u[{jj}]" for jj in range(sp.UB)]
                o += [f"{s}.r[{r}]" for r in range(sp.RB)]
                o += [f"{s}.D{kk}" for kk in range(sp.ND)]
                o += [f"{s}.c[{cc}]" for cc in range(sp.ncoarse)]
                o += [f"{s}.XL", f"{s}.XH"] if pl.wide else [f"{s}.X"]
                o += [f"{s}.K[{jj}]" for jj in range(sp.kb)]
                o += [f"{s}.XsL", f"{s}.XsH"] if pl.wide else [f"{s}.Xs"]
            o.append(f"{tag}.C")
        else:
            o.append(f"{tag}.delta")
            o += [f"{tag}.sel[{i}]" for i in range(pl.W + pl.PB + 1)]
        o.append(f"{tag}.sgn")
        if pl.wide:
            o += [f"{tag}.vL", f"{tag}.vH", f"{tag}.qL", f"{tag}.qH"]
            if not v3:
                o += [f"{tag}.QL", f"{tag}.QH"]
            o.append(f"{tag}.carry")
        else:
            o += [f"{tag}.v", f"{tag}.q"] + ([] if v3 else [f"{tag}.Q"])
        o += [f"{tag}.nsel[{i}]" for i in range(pl.n_t)]
        o += [f"{tag}.hid", f"{tag}.fo"] + ([] if v3 else [f"{tag}.M"])
        o += [f"{tag}.z", f"{tag}.ovf"]
        start += size
    if chain and not pl.fp8:
        o += ["epi.zl"] if v3 else ["epi.fhi", "epi.half", "epi.sticky", "epi.zl"]
    return o


class FusedHints:
    """``run(c (l,), a (l, k), b (l, k)) -> (n_hints, l) int64`` in ``sys.hints`` order, equal to the family's torch
    generator on every input of those shapes."""

    def __init__(self, family: str, p, sys: System, device):
        import cupy as cp
        import torch
        if family not in ("v2", "v3"):
            raise ValueError(family)
        self.cp, self.family, self.device = cp, family, torch.device(device)
        pl = plan(p)
        sp = None
        if family == "v3":
            from .privsel.relation import shift_plan
            sp = shift_plan(p, pl)
        if pl.SB > 62:
            raise ValueError("|sum| wider than 62 bits: the bit-length loop's 1 << k would leave int64")
        chain = sys.chain is not None
        order = _order(family, p, pl, sp, chain)
        names = [n for _, n in sys.hints]
        pos = {n: i for i, n in enumerate(names)}
        if len(pos) != len(names):
            raise ValueError("duplicate hint names in the system")
        if len(set(order)) != len(order):
            raise ValueError("the kernel's emission order names a hint twice")
        missing = sorted(set(names) - set(order))
        if missing:
            raise ValueError(f"the fused {family} kernel does not produce {len(missing)} hint rows, e.g. {missing[:4]}")
        self.k = sum(pl.groups)
        self.n_hints = len(names)
        self.n_emit = len(order)
        D = dict(V3=int(family == "v3"), CHAIN=int(chain), FP8=int(pl.fp8), WIDE=int(pl.wide), K=self.k, NG=len(pl.groups),
                 MAXG=max(pl.groups), ACC_FRAC=ACC_FRAC, ACC_EXP=p.acc_exp, ACC_BIAS=p.acc_bias, FLOOR=p.floor,
                 WIDTH=p.width, ACC_E_MIN=p.acc_e_min, ACC_E_MAX=p.acc_e_max, DROP=pl.drop, W=pl.W, SS=pl.ss, PB=pl.PB,
                 PS=pl.PS, SB=pl.SB, NB=pl.nb, T_LO=pl.t_lo, N_T=pl.n_t, OP_EXP=pl.op_exp, OP_FRAC=pl.op_frac,
                 OP_BIAS=pl.op_bias, OP_WORD_BITS=pl.op_word_bits, LIMB=LIMB,
                 FRAC=0 if pl.fp8 else p.frac)
        if sp is not None:
            D.update(UB=sp.UB, RB=sp.RB, ND=sp.ND, KB=sp.kb, NCOARSE=sp.ncoarse, HI=int(sp.hi_thr is not None),
                     HI_THR=sp.hi_thr if sp.hi_thr is not None else 0)
        head = "".join(f"#define {k_} ({int(v)})\n" for k_, v in D.items())
        self.src = head + _SRC.replace("GSIZES", ", ".join(str(int(s)) for s in pl.groups))
        self.mod = cp.RawModule(code=self.src, options=("-std=c++17",))
        self.kernel = self.mod.get_function("hints_fused")
        with cp.cuda.Device(self.device.index or 0):
            self.rowmap = cp.asarray(np.array([pos.get(n, -1) for n in order], dtype=np.int32))

    def run(self, c, a, b):
        import warnings

        import torch
        cp = self.cp
        l = c.shape[0]
        assert a.shape == (l, self.k) and b.shape == (l, self.k), (a.shape, b.shape, l, self.k)
        c = c.to(self.device, torch.int64).contiguous()
        a = a.to(self.device, torch.int64).contiguous()
        b = b.to(self.device, torch.int64).contiguous()
        out = torch.empty((self.n_hints, l), dtype=torch.int64, device=self.device)
        ts = torch.cuda.current_stream(self.device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            stream = cp.cuda.ExternalStream(ts.cuda_stream)
        with stream:
            self.kernel((-(-l // THREADS),), (THREADS,),
                        (cp.asarray(c), cp.asarray(a), cp.asarray(b), self.rowmap, cp.asarray(out), np.int32(l)))
        return out


def enabled() -> bool:
    """``LIGERO_FUSED_HINTS=0`` selects the torch generators (the A/B switch); default on."""
    return os.environ.get("LIGERO_FUSED_HINTS", "1") != "0"


_FH: dict = {}


def fused_hints_for(family: str, p, sys: System, device) -> FusedHints | None:
    """The fused generator for this system on this CUDA device, or None (CPU, disabled, or no NVRTC -- the torch
    generator runs; ``LIGERO_GPU_STRICT=1`` makes a missing kernel an error)."""
    if not str(device).startswith("cuda") or not enabled():
        return None
    key = (family, id(sys), str(device))
    hit = _FH.get(key)
    if hit is not None and hit[0] is sys:
        return hit[1]
    try:
        f = FusedHints(family, p, sys, device)
    except Exception as e:  # noqa: BLE001
        if os.environ.get("LIGERO_GPU_STRICT"):
            raise
        import warnings
        warnings.warn(f"fused {family} hint kernel unavailable ({e!r}); the torch generator runs instead")
        f = None
    _FH[key] = (sys, f)
    return f
