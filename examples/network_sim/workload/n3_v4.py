"""N3Window_v4: `n3_unit`'s N3Window_v3 with its openings as runs. It accepts exactly what v3 accepts on every window
whose reads fall into R runs (below), and it hashes each internal node above a run once, not once a frame.

    N3Window_v4{FMT, N, G, SALTED, REC, R}
        (layout, layout_domain: Digest, at: Array<DL>, record, record_domain: Digest, direction: Array<1>,
         roots, domains: Array<U, Digest>, [salt: Array<72, Byte>],
         frames: Array<N, Array<B, Byte>>,
         descriptors: Array<n, Array<LW>>, descriptor_salts: Array<n, Salt>,
         descriptor_left, descriptor_right: Array<DL, Digest>,
         rows: Array<n, G, Row>, row_salts: Array<n, G, Salt>,
         runs: Array<T, Array<R>>, fresh: Array<T>,
         run_units: Array<R, Array<UB>>, run_starts: Array<R, Array<DB>>, run_fill: Array<R, Array<S, Digest>>,
         run_left, run_right: Array<R, Array<DB, Digest>>,
         head: Array<HL, Byte>, record_salt: Salt, record_path: Array<2, Digest>, [slot: Array<SB>]) -> Value<1>
        T = nG reads (read t = i G + g is descriptor i's segment g), S = min(T, 2^DB)

v3's statics, inputs and checks, except how descriptors and outbox rows open:

- **Descriptors, one run at public positions.** Descriptor i's hiding leaf is at position at + i (`N3HidingLeaf`, the
  leaf v3's read climbs from), and the n leaves open as one run (`N3Range`): every node above them is hashed once, with
  the run's two boundary siblings a level (`descriptor_left`, `descriptor_right`) as witness. at + n - 1 < 2^DL
  (`N3Fits`): a run doesn't wrap the layout, which v3 reads mod 2^DL (the service never builds such a window,
  `Setup.attach`).
- **Rows, R runs at private positions.** Read t's hiding leaf is at position q in unit u's tree, as v3's. Each kind-1
  read belongs to at least one run r (`runs[t][r]`), and within a run, in read order, each member is `fresh` or repeats
  the member before it (the same leaf, so the same row at the same position; `N3Carry`). A run's fresh leaves, in
  order, are its first slots (`N3CompactStep`, a compaction network over the leaves); the slots after them hold
  `run_fill`. The S slots open at run_starts[r], run_starts[r] + 1, ... in unit run_units[r]'s tree (`N3Range`,
  `N3RunRoot`), and every fresh leaf's slot is in that tree (`N3RunFits`). A leaf binds its position and its tree's
  domain id, so a fresh leaf at the wrong slot, a run that skips a position, or a member of another unit doesn't open.
  Kind-0 reads are zero rows, as v3's.

R is the size class's: the most runs the port's declared reads need (`runs_of`, a property of the layout as G is). On
the windows the tests build, R is 1, and 2 for the demo's two units.

    python -m workload.n3_v4 circuit-check --out FILE [--window]   circuit-check on v4's new parts [and the demo's window]
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ZERO, add, const_bits, sub
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array
from verity.ml.commitments.gates.frame_v3 import FrameV3Leaf, FrameV3Node, index_bytes, leaf_ref, node_ref
from verity.ml.commitments.gates.hm96 import Hm96Sha512Leaf
from verity.ml.commitments.gates.lms import LmsEqual

from . import n3_unit as U
from .n3_unit import BIT, DIGEST, LEVEL, SALT_T, SCHEMA

ZD = bytes(64)


# --- the run of consecutive leaves ------------------------------------------------------------------------------------

def slots(S: int, k: int) -> int:
    """The most level-k nodes a run of S consecutive leaves spans: f + 1 for f = (S - 1) >> k, and f + 2 when the run
    can straddle a level-k boundary (S - 1 not a multiple of 2^k)."""
    return ((S - 1) >> k) + 1 + ((S - 1) % (1 << k) != 0)


def width(S: int, k: int, D: int) -> int:
    """The level-k nodes `N3Range` hashes: the run's slots, at most the 2^(D - k) nodes of the level, which are all
    that level D's first node reads."""
    return min(slots(S, k), 1 << (D - k))


def _check_run(S, D) -> None:
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (S, D)) or not 1 <= D <= 64 or not 1 <= S <= 1 << D:
        raise ValueError(f"a run is 1 <= S <= 2^D leaves of a tree of depth 1 <= D <= 64, not S={S!r}, D={D!r}")


def sel_js(S: int, D: int, k: int) -> list[int]:
    """The j of the positions f + 1 + j (f = (S - 1) >> k) the right sibling can take at level k that are among the
    nodes `N3Range` hashes there."""
    return [j for j in range(3) if ((S - 1) >> k) + 1 + j < 2 * width(S, k + 1, D)]


def _sig_rangesel(S):
    _check_run(S.S, S.D)
    n = sum(len(sel_js(S.S, S.D, k)) for k in range(S.D))
    if not n:
        raise ValueError(f"a run of {S.S} leaves of a depth-{S.D} tree has no right sibling to place")
    return (("start", Array(S.D, BIT)),), Array(n, BIT)


def _build_rangesel(C, S, start):
    js = [sel_js(S.S, S.D, k) for k in range(S.D)]
    W = max(k for k in range(S.D) if js[k]) + 2
    total = add(C, list(start[:W]), const_bits(S.S - 1, W), width=W)
    out = []
    for k in range(S.D):
        if js[k]:
            x = [total[k], total[k + 1] ^ start[k + 1] if k + 1 < S.D else total[k + 1]]
            d = add(C, x, const_bits(-((S.S - 1) >> k) % 4, 2), width=2)
            out += [C.AND(d[0] if j & 1 else d[0].inv(), d[1] if j & 2 else d[1].inv()) for j in js[k]]
    return out


N3RangeSel = U._part(
    "N3RangeSel", ("S", "D"), _sig_rangesel, _build_rangesel,
    "per level k of a run of S leaves from start, and each position f + 1 + j (j < 3, f = (S - 1) >> k) among the "
    "nodes N3Range hashes at k (sel_js): whether the right sibling takes it, the one after the run's last node with "
    "the first's shift")


def _sig_range(S):
    _check_run(S.S, S.D)
    return ((("start", Array(S.D, BIT)), ("leaves", Array(S.S, DIGEST)), ("domain", DIGEST),
             ("left", Array(S.D, DIGEST)), ("right", Array(S.D, DIGEST))), Array(S.D + 1, DIGEST))


def _digests(colls: Sequence[Coll]) -> Coll:
    return Coll(Array(len(colls), DIGEST), concat([c.refs for c in colls]))


def _body_range(B, S, start, leaves, domain, left, right):
    K = U._Consts(B)
    zero = U._coll(DIGEST, K.bits(0, 512))
    js = [sel_js(S.S, S.D, k) for k in range(S.D)]
    sel = B.call(bind(N3RangeSel, S=S.S, D=S.D), start).refs if any(js) else None
    o = 0
    sb = U._refs(start)
    X = [leaves[j] for j in range(S.S)]
    firsts = [X[0]]
    for k in range(S.D):
        M1 = width(S.S, k + 1, S.D)
        f = (S.S - 1) >> k
        P = X + [zero] * max(0, 2 * M1 - len(X))
        a = U._coll(BIT, [sb[k]])
        base = B.batch(bind(U.N3Mux, W=512), _digests(P[:2 * M1]).reshape(Array(2 * M1, Array(512, BIT))),
                       _digests([left[k]] + P[:2 * M1 - 1]).reshape(Array(2 * M1, Array(512, BIT))), a,
                       axes=(0, 0, None))
        E = [base[i].reshape(DIGEST) for i in range(2 * M1)]
        for j in js[k]:
            p = f + 1 + j
            E[p] = B.call(bind(U.N3Mux, W=512), E[p].reshape(Array(512, BIT)),
                          right[k].reshape(Array(512, BIT)), U._coll(BIT, [sel[o]])).reshape(DIGEST)
            o += 1
        hi = S.D - k - 1
        nb = index_bytes(hi)
        if hi:
            idx = B.call(bind(U.N3Positions, DL=hi, N=M1), U._coll(Array(hi, BIT), sb[k + 1:]))
            ids = Coll(Array(M1, Array(8 * nb, BIT)),
                       concat([U._coll(Array(8 * nb, BIT), U._refs(idx[j]) + K.bits(0, 8 * nb - hi)).refs
                               for j in range(M1)]))
        else:
            ids = Coll(Array(M1, Array(8, BIT)), concat([U._coll(Array(8, BIT), K.bits(0, 8)).refs] * M1))
        X = list(B.batch(bind(FrameV3Node, B=nb), domain, U._coll(LEVEL, K.bits(k, 8)), ids,
                         U._coll(BIT, K.bits(0, 1)), _digests(E[0::2]), _digests(E[1::2]),
                         axes=(None, None, 0, None, 0, 0)))
        firsts.append(X[0])
    return _digests(firsts)


N3Range = CompositeDefinition(
    "N3Range", 1, ("S", "D"), _sig_range, _body_range, conformance="exact-model-tested", register=False,
    doc="the first node at each level 0 ... D of a run of S consecutive leaves from position start in a depth-D "
        "frame-v3-sha512 tree of domain id domain: each level's nodes over the run hashed once, the boundary siblings "
        "left[k] and right[k] beside them; slot 0 at level d is the root of a depth-d tree when start < 2^d")


def range_levels(domain: bytes, start: int, leaves: Sequence[bytes], left: Sequence[bytes], right: Sequence[bytes],
                 D: int) -> list[bytes]:
    """`N3Range`'s reference, garbage past the tree included."""
    S = len(leaves)
    X, out = list(leaves), [leaves[0]]
    total = start + S - 1
    for k in range(D):
        M1, f = width(S, k + 1, D), (S - 1) >> k
        P = X + [ZD] * max(0, 2 * M1 - len(X))
        E = [(left[k] if i == 0 else P[i - 1]) if start >> k & 1 else P[i] for i in range(2 * M1)]
        delta = ((total >> k) - ((start >> (k + 1)) << 1) - f) % 4
        for j in range(3):
            if f + 1 + j < 2 * M1 and delta == j:
                E[f + 1 + j] = right[k]
        hi = D - k - 1
        X = [node_ref(domain, k, ((start >> (k + 1)) + j) % (1 << hi) if hi else 0, E[2 * j], E[2 * j + 1])
             for j in range(M1)]
        out.append(X[0])
    return out


def siblings(levels: Sequence[Sequence[bytes]], start: int, S: int, D: int) -> tuple[list[bytes], list[bytes]]:
    """A run's boundary siblings in a tree's levels (zero past the tree): at level k, the node left of the run's first
    when that is a right child, and the node right of its last when that is a left child."""
    left, right = [], []
    for k in range(D):
        lo, hi = start >> k, (start + S - 1) >> k
        lv = levels[k] if k < len(levels) else ()
        left.append(lv[lo - 1] if lo & 1 and lo - 1 < len(lv) else ZD)
        right.append(lv[hi + 1] if not hi & 1 and hi + 1 < len(lv) else ZD)
    return left, right


# --- small parts ------------------------------------------------------------------------------------------------------

def _sig_hleaf(S):
    return (("row", Array(S.N_BITS, BIT)), ("salt", SALT_T), ("domain", DIGEST), ("pos", Array(S.D, BIT))), DIGEST


def _body_hleaf(B, S, row, salt, domain, pos):
    K = U._Consts(B)
    x = B.call(bind(U.N3RowDigest, N_BITS=S.N_BITS), row)
    value = B.call(Hm96Sha512Leaf, x, salt)[2]
    nb = index_bytes(S.D)
    return B.call(bind(FrameV3Leaf, B=nb, SCHEMA=SCHEMA), domain, value,
                  U._coll(Array(8 * nb, BIT), U._refs(pos) + K.bits(0, 8 * nb - S.D)))


N3HidingLeaf = CompositeDefinition(
    "N3HidingLeaf", 1, ("N_BITS", "D"), _sig_hleaf, _body_hleaf, conformance="exact-model-tested", register=False,
    doc="the frame-v3-sha512 leaf at pos (D bits) of an N_BITS-bit row with its salt under domain id domain: the "
        "hm96-sha512/row/v2 leaf of its sha512/row/v2 digest, the leaf N3HidingRead climbs from")


def _sig_fits(S):
    _check_run(S.N, S.D)
    return (("start", Array(S.D, BIT)),), BIT


N3Fits = U._part("N3Fits", ("D", "N"), _sig_fits,
                 lambda C, S, start: [add(C, list(start) + [ZERO], const_bits(S.N - 1, S.D + 1), width=S.D + 1)[S.D]
                                      .inv()],
                 "1 iff start + N - 1 < 2^D: a run of N from start doesn't wrap the tree")


N3Skip = U._part("N3Skip", ("W",), lambda S: ((("z", Array(S.W, BIT)), ("b", BIT)), Array(S.W, BIT)),
                 lambda C, S, z, b: add(C, list(z), [b[0].inv()] + [ZERO] * (S.W - 1), width=S.W),
                 "z + 1 - b mod 2^W: a count of the elements not selected")


def _build_carry(C, S, *e):
    last, leaf, member, fresh = [x for w in e[0:8] for x in w], [x for w in e[8:16] for x in w], e[16][0], e[17][0]
    repeat = C.AND(member, fresh.inv())
    return [C.AND(repeat, U._eq(C, leaf, last).inv()).inv(), C.AND(member, fresh)]


N3Carry = U._part(
    "N3Carry", (), lambda S: ((("last", DIGEST), ("leaf", DIGEST), ("member", BIT), ("fresh", BIT)), Array(2, BIT)),
    _build_carry,
    "a run's member that isn't fresh has last, the leaf of the run's member before it: the read's ok bit, and whether "
    "it is a fresh member")


def _sig_step(S):
    if not isinstance(S.W, int) or isinstance(S.W, bool) or S.W < 2:
        raise ValueError(f"a compaction element is a valid bit, a shift bit and more, not W={S.W!r}")
    return (("stay", Array(S.W, BIT)), ("inc", Array(S.W, BIT))), Array(S.W - 1, BIT)


def _build_step(C, S, stay, inc):
    move = C.AND(inc[0], inc[1])
    keep = C.AND(stay[0], stay[1].inv())
    return [move ^ keep ^ C.AND(move, keep)] + [x ^ C.AND(move, x ^ y) for x, y in zip(stay[2:], inc[2:])]


N3CompactStep = U._part(
    "N3CompactStep", ("W",), _sig_step, _build_step,
    "one stage of a compaction network over elements (valid, shift, data), the shift's bit for this stage first: inc "
    "(the element the stage's distance to the right) if it is valid and has that bit, else stay, valid if it stays "
    "(hasn't the bit); the element without that bit, which later stages don't read")


def compaction(T: int, Sr: int) -> list[tuple[int, int]]:
    """Each stage b of a run's compaction network over its T reads, as (positions, W): the positions a slot below Sr
    still reads, Sr + 2^n - 2^(b + 1) (n stages) and at most T, and its elements' width, a valid bit, the shift's
    bits b and up, and a leaf. An element's shift is below T, so n = bitlen(T - 1) bits."""
    n = (T - 1).bit_length()
    return [(min(T, Sr + (1 << n) - (2 << b)), 1 + n - b + 512) for b in range(n)]


def _sig_runs(S):
    return (("kind", BIT), ("zero", BIT), ("mem", Array(S.R, BIT))), Array(S.R + 1, BIT)


def _build_runs(C, S, kind, zero, mem):
    k = kind[0]
    covered = k.inv()
    for m in mem:
        covered = covered ^ C.AND(covered.inv(), C.AND(k, m))
    return [C.AND(k, m) for m in mem] + [C.AND(k ^ zero[0] ^ C.AND(k, zero[0]), covered)]


N3Runs = U._part(
    "N3Runs", ("R",), _sig_runs, _build_runs,
    "a read's membership of each run (kind 1 and named), then its ok bit: kind 1 in some run, or kind 0 and a zero row")


def _depths(S):
    U_, DB, UB, _ = U._idw(S.DEPTHS)
    return U_, DB, UB


def _zw(T: int) -> int:
    return max(1, T.bit_length())


def _sig_runfits(S):
    _, DB, UB = _depths(S)
    return (("u", Array(UB, BIT)), ("start", Array(DB, BIT)), ("z", Array(_zw(S.T), BIT))), BIT


def _build_runfits(C, S, u, start, z):
    U_, DB, _ = _depths(S)
    zw = _zw(S.T)
    W = max(DB, zw, (S.T + S.S).bit_length()) + 2
    count = sub(C, const_bits(S.T, W), list(z) + [ZERO] * (W - zw), W)[0]
    end = add(C, list(start) + [ZERO] * (W - DB), count, width=W)
    ge = U._thresholds(C, end)
    e = U._onehot(C, list(u), U_)
    fits = ZERO
    for k in range(U_):
        fits = fits ^ C.AND(e[k], ge((1 << S.DEPTHS[k]) + 1).inv())
    return [C.AND(fits, U._thresholds(C, count)(S.S + 1).inv())]


N3RunFits = U._part(
    "N3RunFits", ("DEPTHS", "T", "S"), _sig_runfits, _build_runfits,
    "1 iff a run's fresh leaves, T - z of them, fit its S slots and its unit u's tree from start: start + T - z <= "
    "2^DEPTHS[u] (0 for a u past the units)")


def _sig_runroot(S):
    U_, DB, UB = _depths(S)
    return (("u", Array(UB, BIT)), ("levels", Array(DB + 1, DIGEST)), ("roots", Array(U_, DIGEST))), BIT


def _build_runroot(C, S, u, *rest):
    U_, DB, _ = _depths(S)
    levels = [rest[8 * k:8 * k + 8] for k in range(DB + 1)]
    roots = [rest[8 * (DB + 1) + 8 * k:8 * (DB + 1) + 8 * k + 8] for k in range(U_)]
    e = U._onehot(C, list(u), U_)
    hit = ZERO
    for k in range(U_):
        hit = hit ^ C.AND(e[k], U._eq(C, [x for w in levels[S.DEPTHS[k]] for x in w], [x for w in roots[k] for x in w]))
    return [hit]


N3RunRoot = U._part("N3RunRoot", ("DEPTHS",), _sig_runroot, _build_runroot,
                    "1 iff the run's level DEPTHS[u] is unit u's root (0 for a u past the units)")


# --- the window -------------------------------------------------------------------------------------------------------

def _sizes(S) -> tuple:
    f = U._format(S.FMT)
    n = S.N - S.SALTED
    T = n * S.G
    Sr = min(T, 1 << f.DB)
    return f, n, T, Sr, _zw(T)


def _window_params(S) -> list[tuple[str, object]]:
    base = U._window_params(S)
    f, n, T, Sr, _ = _sizes(S)
    if not isinstance(S.R, int) or isinstance(S.R, bool) or S.R < 1:
        raise ValueError(f"N3Window_v4 takes R >= 1 runs, not {S.R!r}")
    if n > 1 << f.DL:
        raise ValueError(f"{n} descriptors don't fit a layout of 2^{f.DL} positions")
    params = []
    for k, t in base:
        if k == "descriptor_paths":
            params += [("descriptor_left", Array(f.DL, DIGEST)), ("descriptor_right", Array(f.DL, DIGEST))]
        elif k == "row_paths":
            params += [("runs", Array(T, Array(S.R, BIT))), ("fresh", Array(T, BIT)),
                       ("run_units", Array(S.R, Array(f.UB, BIT))), ("run_starts", Array(S.R, Array(f.DB, BIT))),
                       ("run_fill", Array(S.R, Array(Sr, DIGEST))), ("run_left", Array(S.R, Array(f.DB, DIGEST))),
                       ("run_right", Array(S.R, Array(f.DB, DIGEST)))]
        else:
            params.append((k, t))
    return params


def _sig_window(S):
    return tuple(_window_params(S)), BIT


def _body_window(B, S, *args):
    f, n, T, Sr, ZW = _sizes(S)
    N, G = S.N, S.G
    a = dict(zip([k for k, _ in _window_params(S)], args))
    K = U._Consts(B)
    oks: list[Coll] = []
    frames = a["frames"]
    data = frames
    if S.SALTED:
        flat = frames.reshape(Array(N, Array(8 * f.B, BIT)))
        slot = B.call(bind(U.N3Slot, N=N), a["slot"])
        picked = B.batch(bind(U.N3Gate, W=8 * f.B), flat, slot[N:2 * N], axes=(0, 0))
        chosen = B.call(bind(U.N3Xor, N=N, W=8 * f.B), picked) if N > 1 else picked[0]
        oks.append(B.call(bind(U.N3SaltEq, B=f.B), chosen, a["salt"]))
        if n:
            routed = B.batch(bind(U.N3Mux, W=8 * f.B), flat[0:n], flat[1:N], slot[0:n], axes=(0, 0, 0))
            data = routed.reshape(Array(n, Array(f.B, BYTE)))
    if n:
        d = a["descriptors"]
        pos = B.call(bind(U.N3Positions, DL=f.DL, N=n), a["at"])
        leaves = B.batch(bind(N3HidingLeaf, N_BITS=f.LW, D=f.DL), d, a["descriptor_salts"], a["layout_domain"], pos,
                         axes=(0, 0, None, 0), name="descriptors")
        lev = B.call(bind(N3Range, S=n, D=f.DL), a["at"], leaves, a["layout_domain"], a["descriptor_left"],
                     a["descriptor_right"], name="descriptors")
        oks.append(B.call(bind(LmsEqual, N=64), lev[f.DL], a["layout"], name="descriptors"))
        oks.append(B.call(bind(N3Fits, D=f.DL, N=n), a["at"], name="descriptors"))
        idw = [d.refs.slice(i * f.LW + f.HW + g * f.SW, i * f.LW + f.HW + g * f.SW + f.IDW)
               for i in range(n) for g in range(G)]
        kinds = Coll(Array(T, BIT), concat([r.slice(0, 1) for r in idw]))
        us = Coll(Array(T, Array(f.UB, BIT)), concat([r.slice(1, 1 + f.UB) for r in idw]))
        qs = Coll(Array(T, Array(f.DB, BIT)), concat([r.slice(1 + f.UB, f.IDW) for r in idw]))
        rows = a["rows"].reshape(Array(T, Array(16 * f.B, BIT)))
        salts = a["row_salts"].reshape(Array(T, SALT_T))
        doms = B.batch(bind(U.N3Pick, U=f.U), us, a["domains"], axes=(0, None), name="rows")
        rleaves = B.batch(bind(N3HidingLeaf, N_BITS=16 * f.B, D=f.DB), rows, salts, doms, qs, axes=(0, 0, 0, 0),
                          name="rows")
        zeros = B.batch(bind(U.N3IsZero, W=16 * f.B), rows, axes=(0,), name="rows")
        mem = B.batch(bind(N3Runs, R=S.R), kinds, zeros, a["runs"], axes=(0, 0, 0), name="rows")
        oks.append(Coll(Array(T, BIT), concat([mem.refs.slice(t * (S.R + 1) + S.R, (t + 1) * (S.R + 1))
                                                for t in range(T)])))
        fresh = a["fresh"]
        zero_digest = U._coll(DIGEST, K.bits(0, 512))
        stages = compaction(T, Sr)
        nz = len(stages)
        for r in range(S.R):
            last, z, elems, carried = zero_digest, U._coll(Array(ZW, BIT), K.bits(0, ZW)), [], []
            for t in range(T):
                m = U._coll(BIT, [mem.refs[t * (S.R + 1) + r]])
                c = B.call(N3Carry, last, rleaves[t], m, fresh[t], name="rows")
                carried.append(c.refs.slice(0, 1))
                sel = Coll(BIT, c.refs.slice(1, 2))
                elems.append(Coll(Array(1 + nz + 512, BIT),
                                  concat([sel.refs] + ([z.refs.slice(0, nz)] if nz else []) + [rleaves[t].refs])))
                if t < T - 1:
                    last = B.call(bind(U.N3Mux, W=512), last.reshape(Array(512, BIT)),
                                  rleaves[t].reshape(Array(512, BIT)), m, name="rows").reshape(DIGEST)
                z = B.call(bind(N3Skip, W=ZW), z, sel, name="rows")
            oks.append(Coll(Array(T, BIT), concat(carried)))
            for b, (npos, W) in enumerate(stages):
                s, invalid = 1 << b, U._coll(Array(W, BIT), K.bits(0, W))
                stay = Coll(Array(npos, Array(W, BIT)), concat([e.refs for e in elems[:npos]]))
                inc = Coll(Array(npos, Array(W, BIT)),
                           concat([(elems[p + s] if p + s < len(elems) else invalid).refs for p in range(npos)]))
                out = B.batch(bind(N3CompactStep, W=W), stay, inc, axes=(0, 0), name="rows")
                elems = [out[p] for p in range(npos)]
            valid = Coll(Array(Sr, BIT), concat([elems[j].refs.slice(0, 1) for j in range(Sr)]))
            packed = Coll(Array(Sr, Array(512, BIT)), concat([elems[j].refs.slice(1, 513) for j in range(Sr)]))
            fill = a["run_fill"][r].reshape(Array(Sr, Array(512, BIT)))
            slots_ = B.batch(bind(U.N3Mux, W=512), fill, packed, valid, axes=(0, 0, 0), name="rows")
            u_r, start = a["run_units"][r], a["run_starts"][r]
            domain = B.call(bind(U.N3Pick, U=f.U), u_r, a["domains"], name="rows")
            lev = B.call(bind(N3Range, S=Sr, D=f.DB), start, slots_.reshape(Array(Sr, DIGEST)), domain,
                         a["run_left"][r], a["run_right"][r], name="rows")
            oks.append(B.call(bind(N3RunRoot, DEPTHS=f.DEPTHS), u_r, lev, a["roots"], name="rows"))
            oks.append(B.call(bind(N3RunFits, DEPTHS=f.DEPTHS, T=T, S=Sr), u_r, start, z, name="rows"))
        oks.append(B.batch(bind(U.N3Frame, FMT=S.FMT, G=G), d, a["rows"], data, axes=(0, 0, 0)))
    rec = U._shape(S.REC)
    root = B.call(bind(U.N3FramesRoot, B=f.B, N=N, D=rec.D), frames)
    hb = U._refs(a["head"])
    oks.append(B.call(bind(U.N3IsZero, W=8), U._coll(Array(8, BIT), hb[8 * rec.ST:8 * rec.ST + 8])))
    nh = 8 * (rec.HL + 64)
    head = U._coll(Array(nh, BIT), hb + [x for byte in U._digest_bytes(root) for x in byte])
    pos2 = U._coll(Array(2, BIT), U._refs(a["direction"]) + K.bits(1, 1))
    oks.append(B.call(bind(U.N3HidingRead, N_BITS=nh, D=2), head, a["record_salt"], a["record"], a["record_domain"],
                      pos2, a["record_path"]))
    bits = concat([o.refs for o in oks])
    return B.call(BL.and_gate(bits.count), Coll(Array(bits.count, BIT), bits))


BYTE = U.BYTE

N3Window = CompositeDefinition(
    "N3Window", 4, ("FMT", "N", "G", "SALTED", "REC", "R"), _sig_window, _body_window,
    conformance="exact-model-tested", register=False,
    doc="N3Window_v3's ok bit with its openings as runs: the descriptors' leaves at at, at + 1, ... open as one run "
        "of the layout; each kind-1 read's leaf is a member of one of R runs, fresh or the run's last leaf again, each "
        "run's fresh leaves its first slots, opening at consecutive positions in its unit's outbox")


def unit(f: U.Format, N: int, G: int, salted: bool, rec: U.Shape, R: int):
    return bind(N3Window, FMT=tuple(f), N=N, G=G, SALTED=int(salted), REC=tuple(rec), R=R)


# --- runs, instances and the reference --------------------------------------------------------------------------------

def reads(f: U.Format, descriptors: Sequence[bytes], G: int) -> list[U.Segment]:
    """The window's reads in order: descriptor i's segment g is read i G + g."""
    return [U.fields(f, x).segs[g] for x in descriptors for g in range(G)]


def partition(rs: Sequence[U.Segment]) -> tuple[list[int | None], list[bool]]:
    """Each read's run (None for kind 0) and whether it is fresh, greedily: a kind-1 read joins the first run of its
    unit whose last member is its row or the one before, else starts a run."""
    last: list[tuple[int, int]] = []
    of, fresh = [], []
    for s in rs:
        if not s.kind:
            of.append(None)
            fresh.append(False)
            continue
        r = next((i for i, (u, q) in enumerate(last) if u == s.u and s.q - q in (0, 1)), None)
        if r is None:
            r = len(last)
            last.append((s.u, s.q))
            fresh.append(True)
        else:
            fresh.append(s.q != last[r][1])
            last[r] = (s.u, s.q)
        of.append(r)
    return of, fresh


def runs_of(layout: U.Layout, port: str) -> int:
    """R of the port's windows: the runs its declared reads need, at least one."""
    f, G = layout.fmt, layout.segments(port)
    ds = [U.encode(f, d) for d in layout.descriptors.get(port, [])]
    of, _ = partition(reads(f, ds, G))
    return max([1] + [r + 1 for r in of if r is not None])


def instance(setup: U.Setup, w: U.Window, R: int | None = None) -> U.Instance:
    """The window's v4 instance from its v3 one: the descriptors' run siblings, and the reads' runs, fresh bits, run
    units, starts, fill and siblings from the layout's and the outboxes' trees."""
    v3 = w.instance
    if v3 is None:
        raise ValueError(f"{w.port}[{w.window}] runs no unit: {w.why}")
    return convert(v3, setup.layout.tree, setup.units, runs_of(setup.layout, w.port) if R is None else R)


def convert(v3: U.Instance, layout: U.Tree, units: U.UnitTable, R: int, runs: Sequence[Sequence[int]] | None = None,
            fresh: Sequence[int] | None = None, starts: dict[int, int] | None = None,
            run_units: dict[int, int] | None = None) -> U.Instance:
    """The v4 instance of a v3 one: its runs `partition`'s unless given (each read's membership of each run, and
    whether it is fresh), each run's unit and start its first member's unless given, and the siblings and fill each
    run's unit's tree gives at its start, past its fresh members."""
    f, n, G = v3.fmt, v3.N - v3.salted, v3.G
    args = {k: v for k, v in v3.args.items() if k not in ("descriptor_paths", "row_paths")}
    inst = U.Instance(f, v3.N, G, v3.salted, v3.rec, args)
    inst.R = R  # type: ignore[attr-defined]
    if not n:
        return inst
    at = args["at"]
    args["descriptor_left"], args["descriptor_right"] = siblings(layout.levels, at, n, f.DL)
    rs = reads(f, args["descriptors"], G)
    if runs is None:
        of, new = partition(rs)
        runs, fresh = [[int(o == r) for r in range(R)] for o in of], [int(x) for x in new]
    T, Sr = len(rs), min(len(rs), 1 << f.DB)
    args["runs"], args["fresh"] = [list(x) for x in runs], list(fresh)  # type: ignore[arg-type]
    units_, starts_, fills, lefts, rights = [], [], [], [], []
    for r in range(R):
        members = [t for t in range(T) if rs[t].kind and runs[t][r]]
        u = (run_units or {}).get(r, rs[members[0]].u if members else 0)
        q0 = (starts or {}).get(r, rs[members[0]].q if members else 0)
        levels = units.trees[units.names[u]].levels if u < len(units.names) else ()
        count = sum(fresh[t] for t in members)  # type: ignore[index]
        leaves0 = levels[0] if levels else ()
        fills.append([leaves0[q0 + j] if count <= j and q0 + j < len(leaves0) else ZD for j in range(Sr)])
        left, right = siblings(levels, q0, Sr, f.DB)
        units_.append(u)
        starts_.append(q0)
        lefts.append(left)
        rights.append(right)
    args.update(run_units=units_, run_starts=starts_, run_fill=fills, run_left=lefts, run_right=rights)
    return inst


def _leaf(domain: bytes, row: bytes, n_bits: int, salt: bytes, pos: int) -> bytes:
    return leaf_ref(domain, SCHEMA, U.hiding_value(row, n_bits, salt), pos)


def reference(inst: U.Instance) -> bool:
    """`N3Window_v4`'s output on the instance, computed from its arguments as the circuit does."""
    f, a, N, G, R = inst.fmt, inst.args, inst.N, inst.G, inst.R  # type: ignore[attr-defined]
    n = N - inst.salted
    frames = list(a["frames"])
    if inst.salted:
        slot = a["slot"]
        head = U.HEADER.pack(U.SALT, U.SALT_STREAM, U.SALT_PAYLOAD)
        if slot >= N or frames[slot] != (head + a["salt"]).ljust(f.B, b"\0"):
            return False
        frames = frames[:slot] + frames[slot + 1:]
    descs = a.get("descriptors", [])
    if n:
        at = a["at"]
        if at + n - 1 >= 1 << f.DL:
            return False
        leaves = [_leaf(a["layout_domain"], descs[i], f.LW, a["descriptor_salts"][i], at + i) for i in range(n)]
        if range_levels(a["layout_domain"], at, leaves, a["descriptor_left"], a["descriptor_right"],
                        f.DL)[f.DL] != a["layout"]:
            return False
        rs = reads(f, descs, G)
        rows = [r for per in a["rows"] for r in per]
        salts = [s for per in a["row_salts"] for s in per]
        T, Sr = len(rs), min(len(rs), 1 << f.DB)
        doms = [a["domains"][s.u] if s.u < f.U else ZD for s in rs]
        rl = [_leaf(doms[t], rows[t], 16 * f.B, salts[t], rs[t].q) for t in range(T)]
        for t, s in enumerate(rs):
            if s.kind and not any(a["runs"][t]) or not s.kind and any(rows[t]):
                return False
        for r in range(R):
            last, fresh_leaves = ZD, []
            for t, s in enumerate(rs):
                if not (s.kind and a["runs"][t][r]):
                    continue
                if a["fresh"][t]:
                    fresh_leaves.append(rl[t])
                elif rl[t] != last:
                    return False
                last = rl[t]
            u, start = a["run_units"][r], a["run_starts"][r]
            if u >= f.U or len(fresh_leaves) > Sr or start + len(fresh_leaves) > 1 << f.DEPTHS[u]:
                return False
            slots_ = fresh_leaves + list(a["run_fill"][r][len(fresh_leaves):])
            lev = range_levels(a["domains"][u], start, slots_, a["run_left"][r], a["run_right"][r], f.DB)
            if lev[f.DEPTHS[u]] != a["roots"][u]:
                return False
    for i, x in enumerate(U.fields(f, d) for d in descs):
        if any(s.kind for s in x.segs[G:]) or x.mode not in (U.RAW, U.B64) or U.assemble(f, descs[i], a["rows"][i],
                                                                                          G) != frames[i]:
            return False
    rec = inst.rec
    head = a["head"] + U.CT.frames_root(a["frames"], U.GridParams(rec.T, rec.R, f.B))
    if head[rec.ST]:
        return False
    value = U.hiding_value(head, 8 * len(head), a["record_salt"])
    return U.root_ref(a["record_domain"], SCHEMA, value, U.N2.HEADS + a["direction"], a["record_path"]) == a["record"]


def arguments(inst: U.Instance) -> list[np.ndarray]:
    """The unit's arguments for one instance in its signature's order, each (1, leaves) of 0/1."""
    f, a = inst.fmt, inst.args
    digest = U._digest_bits
    flat: dict[str, list[int]] = {
        "layout": digest(a["layout"]), "layout_domain": digest(a["layout_domain"]), "at": U._int_bits(a["at"], f.DL),
        "record": digest(a["record"]), "record_domain": digest(a["record_domain"]), "direction": [a["direction"]],
        "roots": [b for r in a["roots"] for b in digest(r)], "domains": [b for r in a["domains"] for b in digest(r)],
        "frames": [b for fr in a["frames"] for b in U._bytes_bits(fr, 8 * f.B)],
        "head": U._bytes_bits(a["head"], 8 * inst.rec.HL), "record_salt": digest(a["record_salt"]),
        "record_path": [b for s in a["record_path"] for b in digest(s)],
    }
    if inst.salted:
        flat["salt"] = U._bytes_bits(a["salt"], 8 * U.SALT_PAYLOAD)
        flat["slot"] = U._int_bits(a["slot"], max(1, (inst.N - 1).bit_length()))
    if inst.N - inst.salted:
        flat["descriptors"] = [b for x in a["descriptors"] for b in U._bytes_bits(x, f.LW)]
        flat["descriptor_salts"] = [b for s in a["descriptor_salts"] for b in digest(s)]
        flat["descriptor_left"] = [b for s in a["descriptor_left"] for b in digest(s)]
        flat["descriptor_right"] = [b for s in a["descriptor_right"] for b in digest(s)]
        flat["rows"] = [b for per in a["rows"] for r in per for b in U._bytes_bits(r, 16 * f.B)]
        flat["row_salts"] = [b for per in a["row_salts"] for s in per for b in digest(s)]
        flat["runs"] = [b for per in a["runs"] for b in per]
        flat["fresh"] = list(a["fresh"])
        flat["run_units"] = [b for u in a["run_units"] for b in U._int_bits(u, f.UB)]
        flat["run_starts"] = [b for q in a["run_starts"] for b in U._int_bits(q, f.DB)]
        flat["run_fill"] = [b for per in a["run_fill"] for s in per for b in digest(s)]
        flat["run_left"] = [b for per in a["run_left"] for s in per for b in digest(s)]
        flat["run_right"] = [b for per in a["run_right"] for s in per for b in digest(s)]
    params, _ = unit_of(inst).signature()
    out = []
    for name, t in params:
        if len(flat[name]) != t.leaves:
            raise ValueError(f"argument {name} has {len(flat[name])} bits, the unit takes {t.leaves}")
        out.append(np.asarray(flat[name], dtype=np.uint8).reshape(1, -1))
    return out


def unit_of(inst: U.Instance):
    return unit(inst.fmt, inst.N, inst.G, inst.salted, inst.rec, inst.R)  # type: ignore[attr-defined]


def evaluate(*insts: U.Instance) -> list[int]:
    """The unit's output on each instance, all of one unit, in one evaluation."""
    if len({(i.key, i.R) for i in insts}) != 1:  # type: ignore[attr-defined]
        raise ValueError("the instances are of one unit")
    args = [np.concatenate(x) for x in zip(*(arguments(i) for i in insts))]
    return [int(b) for b in np.asarray(U.evaluate_bits(unit_of(insts[0]), *args)).reshape(-1)]


_GROUPS = {"descriptors": "descriptor_reads", "rows": "row_reads"}


def size(target, memo: dict | None = None) -> dict:
    """The window unit's ANDs by part, as `n3_unit.size` groups v3's: the descriptors' and the rows' runs are
    descriptor and row reads."""
    target = unit_of(target) if isinstance(target, U.Instance) else target
    memo = {} if memo is None else memo
    out = dict.fromkeys(("row_reads", "descriptor_reads", "record_opening", "segments", "frames", "frames_root",
                         "other"), 0)
    for nd in target.body.nodes:
        name = getattr(nd.fn, "name", "")
        group = _GROUPS.get(nd.name or "", U._GROUPS.get(name, "other"))
        if name == "N3HidingRead":
            group = "record_opening"
        ands = U._times(nd) * U._ands(nd.fn, memo)
        if name == "N3Frame":
            seg = U._times(nd) * sum(U._times(x) * U._ands(x.fn, memo) for x in nd.fn.body.nodes
                                     if getattr(x.fn, "name", "") == "N3Seg")
            out["segments"] += seg
            ands -= seg
        out[group] += ands
    out["ands"] = sum(out.values())
    return out


# --- circuit-check ----------------------------------------------------------------------------------------------------

def parts(inst: U.Instance) -> list[tuple[object, bool]]:
    """v4's new parts at the statics of the instance's window, each with whether circuit-check lowers it whole: the
    SHA-free ones whole, the openings by their parts (they call SHA-512's compression, as v3's window does)."""
    f, n, G, R = inst.fmt, inst.N - inst.salted, inst.G, inst.R  # type: ignore[attr-defined]
    T = n * G
    Sr, zw = min(T, 1 << f.DB), _zw(T)
    sels = [bind(N3RangeSel, S=s, D=d) for s, d in ((n, f.DL), (Sr, f.DB)) if any(sel_js(s, d, k) for k in range(d))]
    whole = [*sels, bind(N3Fits, D=f.DL, N=n), bind(N3Skip, W=zw), bind(N3Carry),
             *(bind(N3CompactStep, W=W) for _, W in compaction(T, Sr)),
             bind(N3Runs, R=R), bind(N3RunFits, DEPTHS=f.DEPTHS, T=T, S=Sr), bind(N3RunRoot, DEPTHS=f.DEPTHS)]
    hashed = [bind(N3Range, S=n, D=f.DL), bind(N3Range, S=Sr, D=f.DB), bind(N3HidingLeaf, N_BITS=16 * f.B, D=f.DB)]
    return [(d, True) for d in whole] + [(d, False) for d in hashed]


def circuit_check(window: bool) -> dict:
    """circuit-check's checks on v4's new parts at the demo window's statics and, with `window`, on its whole v4 window
    (lowered by its parts and not as a Call, as `n3_unit`'s command checks v3's)."""
    from circuit_check import checks as CK

    d = U.demo()
    inst = instance(d.setup, d.window)
    opts = CK.Options()
    entries = []
    for target, whole in parts(inst) + ([(unit_of(inst), False)] if window else []):
        t0 = time.time()
        try:
            r = CK.check_definition(target, opts, lower_whole=whole, as_call=False)
        except Exception as e:  # noqa: BLE001  (a target the checks can't run on is a failure by name, as the CLI's)
            r = {"definition": target.id, "failures": [{"name": f"error/raised {target.id}: {type(e).__name__}: {e}"}],
                 "warnings": []}
        entries.append({**r, "kind": "definition", "target": target.id, "seconds": round(time.time() - t0, 3)})
        print(target.id, len(r.get("failures", [])), "failures", round(time.time() - t0, 1), "s", flush=True)
    rep = CK.report(entries, opts)
    rep["size"] = size(inst)
    rep["format"] = inst.fmt._asdict()
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub_ = ap.add_subparsers(dest="cmd", required=True)
    c = sub_.add_parser("circuit-check", help="circuit-check's checks on v4's new parts and, with --window, its window")
    c.add_argument("--window", action="store_true", help="the demo's whole v4 window too")
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rep = circuit_check(args.window)
    args.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
