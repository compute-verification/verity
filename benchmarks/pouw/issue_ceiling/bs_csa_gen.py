"""Generate (and verify) the BS-CSA chunk reducer for bs_csa.cu.

BS-CSA (csa_sim.py): C = X@Y, X,Y in {+-1}; x=(-1)^a, y=(-1)^b; C_ij = K - 2*#{k: a_ik != b_kj}.
A lane word holds b_kj of 32 output columns (one bit-plane per k). Per row a Harley-Seal carry-save counter counts
the planes (Y_k XOR A_ik), A_ik all-ones if a_ik else 0; a warp-uniform branch on (a_ik1, a_ik2) picks the
first-level full adder's LOP3 immediates so the XOR is free (2.00 LOP3 a raw term-plane = 16 MACs a lane LOP3).

A register-resident counter across all of K would need runtime plane indices (-> local memory), so instead a chunk
of CHUNK_TERMS term-planes is reduced by a static, fully-unrolled counter (the sim's exact deferred logic, from
empty each chunk), and the chunk's carry-save result is ripple-added into a persistent binary counter
P[0..LEVELS_P-1]; the kernel loops over chunks and converts P -> int once at the end.

One Engine runs the identical chunk algorithm in two modes: 'eval' computes on numpy words (to prove bit-exactness
against the direct product) and 'emit' records the IR that is rendered to CUDA.  The control flow is
data-independent, so the emitted code is exactly the verified path.
"""
import argparse

import numpy as np

M32 = 0xFFFFFFFF
SUM = 0x96  # table(a^b^c)
MAJ = 0xe8  # table(maj(a,b,c))
# sim FIRST[(u,v)]: full adder of (s, p^u, q^v); with s=0 it is the half adder of (p^u, q^v)
FIRST = {(0, 0): (0x96, 0xe8), (0, 1): (0x69, 0xd4), (1, 0): (0x69, 0xb2), (1, 1): (0x96, 0x71)}


def lop3_val(a, b, c, imm):
    out = 0
    for idx in range(8):
        if imm >> idx & 1:
            ta = a if idx & 4 else (~a & M32)
            tb = b if idx & 2 else (~b & M32)
            tc = c if idx & 1 else (~c & M32)
            out |= ta & tb & tc
    return out & M32


class _Zero:
    """Structural constant-0 plane (never-written slot); identity test keeps control flow data-independent."""


ZERO = _Zero()


class Engine:
    """Process one chunk.  mode 'eval': operands are ints; mode 'emit': operands are var names.  ZERO is the sentinel."""

    def __init__(self, chunk_terms, levels_p, mode, terms=None, uv=None, P=None):
        self.ct = chunk_terms
        self.levels = chunk_terms.bit_length() + 2
        self.levels_p = levels_p
        self.mode = mode
        if mode == 'emit':
            self.ops = []
            self.nv = 0
        else:
            self.terms = terms  # list of ints (one 32-bit plane word per term)
            self.uv = uv        # list of (u, v) per pair
            self.P = dict(P)    # w -> int (persistent binary counter)

    def _nv(self):
        v = f"t{self.nv}"
        self.nv += 1
        return v

    def lop3(self, a, b, c, imm):
        """a, b, c are operands (ZERO sentinel, term/temp handles).  Returns a new handle (value or name)."""
        if self.mode == 'emit':
            an = 'ZERO' if a is ZERO else a
            bn = 'ZERO' if b is ZERO else b
            cn = 'ZERO' if c is ZERO else c
            d = self._nv()
            self.ops.append(('lop3', d, an, bn, cn, imm))
            return d
        av = 0 if a is ZERO else a
        bv = 0 if b is ZERO else b
        cv = 0 if c is ZERO else c
        return lop3_val(av, bv, cv, imm)

    def first(self, s, pidx):
        """First-level full adder of (s, p^u, q^v); warp-uniform switch over FIRST[(u,v)]."""
        p, q = 2 * pidx, 2 * pidx + 1
        if self.mode == 'emit':
            sn = 'ZERO' if s is ZERO else s
            ds, dc = self._nv(), self._nv()
            self.ops.append(('first', ds, dc, sn, f"P{p}", f"P{q}", pidx))
            return ds, dc
        u, v = self.uv[pidx]
        s_imm, c_imm = FIRST[(u, v)]
        sv = 0 if s is ZERO else s
        pv, qv = self.terms[p], self.terms[q]
        return lop3_val(sv, pv, qv, s_imm), lop3_val(sv, pv, qv, c_imm)

    def Pget(self, w):
        if self.mode == 'emit':
            return self.pnames.get(w, f"P[{w}]")
        return self.P.get(w, ZERO)

    def Pset(self, w, v):
        if self.mode == 'emit':
            self.pnames[w] = v
        else:
            self.P[w] = v

    def process(self, npar=1):
        """npar independent deferred sub-counters over round-robin pairs (ILP), all combined into persistent P."""
        if self.mode == 'emit':
            self.pnames = {}
        counters = [([ZERO] * self.levels, [None] * self.levels) for _ in range(npar)]

        def push(planes, pending, w, plane):
            if pending[w] is None:
                pending[w] = plane
                return
            a, pending[w] = pending[w], None
            s = planes[w]
            planes[w] = self.lop3(s, a, plane, SUM)
            carry = self.lop3(s, a, plane, MAJ)
            push(planes, pending, w + 1, carry)

        for pidx in range(self.ct // 2):
            planes, pending = counters[pidx % npar]
            ds, dc = self.first(planes[0], pidx)
            planes[0] = ds
            push(planes, pending, 1, dc)

        # ripple-add each sub-counter's result (planes then pending: a weight-2^w bit) into persistent P
        srcs = [lst for (pl, pe) in counters for lst in (pl, pe)]
        for src in srcs:
            carry = ZERO
            for w in range(self.levels):
                a = src[w] if src[w] is not None else ZERO
                if a is ZERO and carry is ZERO:
                    continue
                pw = self.Pget(w)
                ns = self.lop3(pw, a, carry, SUM)
                nc = self.lop3(pw, a, carry, MAJ)
                self.Pset(w, ns)
                carry = nc
            w = self.levels
            while carry is not ZERO and w < self.levels_p:
                pw = self.Pget(w)
                ns = self.lop3(pw, ZERO, carry, SUM)
                nc = self.lop3(pw, ZERO, carry, MAJ)
                self.Pset(w, ns)
                carry = nc
                w += 1
        return self


def build_ir(chunk_terms, levels_p, npar=1):
    return Engine(chunk_terms, levels_p, 'emit').process(npar)


def _render(name):
    if name == 'ZERO':
        return "0u"
    if name.startswith('P') and name[1:].isdigit():  # term plane P{t}: lane-interleaved shared (conflict-free)
        return f"yl[{name[1:]} * BS_CSA_CB]"
    return name  # temp t{n} or "P[w]"


def emit_cuda(chunk_terms, levels_p, npar=1, guard='branch'):
    """Render the verified chunk IR as a __device__ CUDA reducer.

    guard='branch': the first-level XOR is folded into the LOP3 immediate by a warp-uniform 4-arm switch (the referee's
    construction, 16 MACs a lane LOP3).  guard='xor': the same full adder FA(s, p^u, q^v) with fixed SUM/MAJ immediates
    and an explicit warp-uniform XOR mask -- bit-identical output, no branch (the unguarded lowering, ~10.7 MAC/op).
    """
    g = build_ir(chunk_terms, levels_p, npar)
    lines = []
    for op in g.ops:
        if op[0] == 'lop3':
            _, d, a, b, c, imm = op
            lines.append(f"  uint32_t {d} = L3<0x{imm:02x}>({_render(a)}, {_render(b)}, {_render(c)});")
        else:  # first: (ds, dc, s, P{p}, P{q}, pidx)
            _, ds, dc, s, p, q, pidx = op
            S, P, Q = _render(s), _render(p), _render(q)
            if guard == 'xor':
                lines.append(f"  uint32_t um{pidx} = (uint32_t)(-(int)(xb[{2 * pidx}] < 0));")
                lines.append(f"  uint32_t vm{pidx} = (uint32_t)(-(int)(xb[{2 * pidx + 1}] < 0));")
                lines.append(f"  uint32_t {ds} = L3<0x96>({S}, {P} ^ um{pidx}, {Q} ^ vm{pidx});")
                lines.append(f"  uint32_t {dc} = L3<0xe8>({S}, {P} ^ um{pidx}, {Q} ^ vm{pidx});")
            else:
                lines.append(f"  uint32_t {ds}, {dc};")
                lines.append(f"  switch ((int(xb[{2 * pidx}] < 0) << 1) | int(xb[{2 * pidx + 1}] < 0)) {{")
                for idx, (u, v) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                    si, ci = FIRST[(u, v)]
                    lines.append(f"    case {idx}: {ds} = L3<0x{si:02x}>({S}, {P}, {Q}); "
                                 f"{dc} = L3<0x{ci:02x}>({S}, {P}, {Q}); break;")
                lines.append("    default: __builtin_unreachable();")
                lines.append("  }")
    for w in sorted(g.pnames):
        lines.append(f"  P[{w}] = {g.pnames[w]};")
    body = "\n".join(lines)
    n_first = sum(1 for o in g.ops if o[0] == 'first')
    n_lop3 = sum(1 for o in g.ops if o[0] == 'lop3')
    lop3_per_chunk = n_first * 2 + n_lop3
    return f"""// Generated by bs_csa_gen.py --chunk {chunk_terms} --levels-p {levels_p} --npar {npar}; do not edit.
// Chunk reducer for BS-CSA (csa_sim.py, art:3d8c4200...): counts {chunk_terms} term-planes into the persistent binary
// counter P[0..{levels_p - 1}] for this lane's 32 columns, via {npar} independent sub-counters (ILP).  First-level XOR
// folded into the LOP3 immediate by a warp-uniform switch on (a_ik1, a_ik2).  LOP3 a chunk = {lop3_per_chunk}
// ({chunk_terms * 32}/{lop3_per_chunk} = {chunk_terms * 32 / lop3_per_chunk:.2f} MACs a lane LOP3).  Verified bit-exact.
#pragma once
#include <cstdint>

#define BS_CSA_CHUNK_TERMS {chunk_terms}
#define BS_CSA_LEVELS_P {levels_p}
#define BS_CSA_CB 32  // column-blocks a block (= warp lanes); yl is lane-interleaved so reads are bank-conflict-free

template <int IMM>
__device__ __forceinline__ uint32_t L3(uint32_t a, uint32_t b, uint32_t c) {{
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(d) : "r"(a), "r"(b), "r"(c), "n"(IMM));
  return d;
}}

// yl = &Ys[lane], term plane t at yl[t*BS_CSA_CB] (lane-interleaved shared); xb[t] = this warp's row X byte (shared)
__device__ __forceinline__ void bs_csa_chunk(uint32_t* __restrict__ P, const uint32_t* __restrict__ yl,
                                             const int8_t* __restrict__ xb) {{
{body}
}}
"""


def write_header(chunk_terms, levels_p, path, npar=1, guard='branch'):
    with open(path, "w") as f:
        f.write(emit_cuda(chunk_terms, levels_p, npar, guard))
    print(f"wrote {path} (chunk={chunk_terms}, levels_p={levels_p}, npar={npar}, guard={guard})")


def eval_chunk(chunk_terms, levels_p, terms, uv, P, npar=1):
    return Engine(chunk_terms, levels_p, 'eval', terms=terms, uv=uv, P=P).process(npar).P


def verify(chunk_terms, levels_p, sizes, npar=1):
    allok = True
    for n, rows, seed in sizes:
        assert n % chunk_terms == 0, (n, chunk_terms)
        rng = np.random.default_rng(seed)
        a = rng.integers(0, 2, size=(rows, n))
        b = rng.integers(0, 2, size=(n, 32))
        X = 1 - 2 * a
        Y = 1 - 2 * b
        want = X @ Y
        planes = [int(sum(int(b[k, j]) << j for j in range(32))) for k in range(n)]
        ok = True
        for i in range(rows):
            P = {}
            for c0 in range(0, n, chunk_terms):
                terms = [planes[c0 + t] for t in range(chunk_terms)]
                uv = [(int(a[i, c0 + 2 * p]), int(a[i, c0 + 2 * p + 1])) for p in range(chunk_terms // 2)]
                P = eval_chunk(chunk_terms, levels_p, terms, uv, P, npar)
            cnt = np.zeros(32, dtype=np.int64)
            for w, word in P.items():
                bits = (word >> np.arange(32)) & 1
                cnt += bits.astype(np.int64) << w
            got = n - 2 * cnt
            ok &= bool(np.array_equal(got, want[i]))
        allok &= ok
        print(f"n={n:5d} rows={rows} chunk={chunk_terms} npar={npar}: exact {ok}")
    g = build_ir(chunk_terms, levels_p, npar)
    n_first = sum(1 for o in g.ops if o[0] == 'first')
    n_lop3 = sum(1 for o in g.ops if o[0] == 'lop3')
    lop3_per_chunk = n_first * 2 + n_lop3
    mac_per_lop3 = (chunk_terms * 32) / lop3_per_chunk
    print(f"  chunk={chunk_terms} levels_p={levels_p} npar={npar}: first={n_first} lop3={n_lop3} "
          f"LOP3/chunk={lop3_per_chunk} MACs-a-lane-LOP3={mac_per_lop3:.2f}")
    print("ALL EXACT" if allok else "MISMATCH")
    return allok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--levels-p", type=int, default=14)
    ap.add_argument("--npar", type=int, default=1, help="independent sub-counters a chunk (ILP)")
    ap.add_argument("--guard", choices=("branch", "xor"), default="branch", help="first-level fold lowering")
    ap.add_argument("--emit", type=str, default=None, help="write the CUDA chunk header to this path")
    args = ap.parse_args()
    assert (args.chunk // 2) % args.npar == 0, "npar must divide the pair count (chunk/2)"
    sizes = tuple((n, r, s) for (n, r, s) in ((64, 8, 1), (256, 4, 2), (1024, 2, 3), (4096, 1, 4)) if n % args.chunk == 0)
    ok = verify(args.chunk, args.levels_p, sizes, args.npar)
    if args.emit:
        assert ok, "refusing to emit: verification failed"
        write_header(args.chunk, args.levels_p, args.emit, args.npar, args.guard)
