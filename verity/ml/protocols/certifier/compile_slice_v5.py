"""The compile slice with the pool chained between windows: `CompileSlice_v4` (`compile_slice_v4`) cut in two units, the
compile slice a window proves and the pool window a sampled proof draws, the pool passed from window to window as a
committed root.

    CompileSlice_v5{R, W, A, M, T, L, B, H, CH, WM, FULL, PC, NB, IW, AS, EW, Q, D, P, K, CTW, CU, DA, NS, NM}
        (adv, ing, req, srv, cout, res, calls) -> verdict[16]
    PoolWindow_v1{R, W, A, M, T, L, B, H, CH, WM, FULL, PC, NB, IW, AS, EW, Q, D, P, K, CTW, CU, DA, NS, NM}
        (req, calls, pin, pout, coin, prow, nrow, chk) -> verdict[16]

**Why.** CompileSlice_v4 runs each window from the empty pool, so a window never hits what an earlier one cached, and a
unit that carried the pool between windows would have to be proved every window. Here the pool between windows is one
row (`row_fields`), committed as its unsalted `Sha512RowSeg_v1` digest, the root: a window's pool unit opens the
previous root, runs CacheStep_v3's stream from the pool it opens, and commits the pool it leaves as the next root. Its
output stays in the public size class, so the boundary between windows is the root, never the pool's bits, and the
pool unit is sampled like any other unit. The compile slice carries the steps' calls to it as the registered ``calls``.

**The pool row**, field-major, each element least significant bit first: the free queue's ``head`` and ``tail`` (NW bits
each), then per block b = 1 … NB − 1 its links ``prev``, ``next``, ``cprev``, ``cnext`` (B's), ``mlast`` (the key's
last copy where b is its first copy, else 0) and ``key``, the 512-bit BlockKey_v1 key b is cached under (`Digest`
layout, 0 for none). It holds no count, owner or stamp: a pool between windows holds no block (every ref 0), and owners
and stamps are one window's. The empty pool's root is `genesis`, the first window's ``pin``; window w + 1's ``pin`` is
window w's ``pout``, one registered row both units read.

**The inputs**, each a field list (`fields`):
- ``adv`` (*advice*), ``ing`` (*ingress*), ``req``, ``srv`` and ``cout`` (*registered*): CompileSlice_v4's.
- ``res`` (*registered*): the steps' results, CompileSlice_v4's first four checked fields (backs, hits, evs, rets).
- ``calls`` (*registered*): the steps' `nrec` bus records, field-major as `CompileSlice_v4.CompileWindow` makes them.
- ``pin`` and ``pout`` (*registered*): the roots the window opens and leaves, 512 bits each.
- ``coin`` (the verifier's coins): α, γ and δ, CompileSlice_v4's, drawn live: δ after every registered row, ``prow``
  and ``nrow`` are committed, α and γ after ``chk`` is.
- ``prow`` and ``nrow`` (*checked*): the pool rows under ``pin`` and ``pout``.
- ``chk`` (*checked*): CompileSlice_v4's stream witness (its slots, its dense run's finals, its M run), the M run
  holding every key the stream touches and every key the opened pool caches.
A registered row is padded to whole words (`compile_slice.WORD`); its padding must be zero. `CLASSES` names each input's
class; the pool unit's only free input is none, the compile slice's its advice.

**The compile slice**: CompileSlice_v4's `CompileWindow` (the load and the steps, from ``res``), then `CallsVerdict`: its
flags, and bit 5 ``calls``, 1 iff its records are ``calls`` and the padding of ``res`` and ``calls`` is zero. It has no
coin. `COMPILE_ACCEPT` = 0x3F.

**The pool window**'s verdict, `POOL_ACCEPT` = 0x7:
- bit 0 ``pool``: the stream checks (`cache_step_v3.PoolEnd`), its bus product over ``calls`` is the stream's and its call
  count theirs (CompileSlice_v4's pool bit);
- bit 1 ``open``: ``prow``'s digest is ``pin``;
- bit 2 ``close``: ``nrow``'s digest is ``pout``, the stream's queue ends are ``nrow``'s and every block closes
  (`PoolClose`): its final links are ``nrow``'s, it has no holder, its final key print is the print of ``nrow``'s key, and
  ``mlast`` is 0 but at a first copy; and the first copies with their last copies are the M run's nonzero final cells
  (R5 = W5 over α and γ, records of tag 5).

**The opening** is the memory check's initial multiset, from ``prow`` alone: the stream starts from its queue ends; B's
initial values are its links, every count, owner and stamp 0; K's are its keys' prints under δ (`block_key.KeyPrint`, 0
for no key); RB's are 0 and RK's the arrivals' keys' prints, CompileSlice_v4's; M's are 0 in the touched list, and
`PoolOpen` multiplies, at each block b whose print is nonzero and whose cprev is 0, (M, print, 0, 0) into R and (M,
print, b | mlast << NW, 0) into W: the key's cell holds (first b, last mlast) before the stream, so the M run must list
it. No witness about the previous pool is free: a wrong opening is a ``prow`` other than the one under ``pin``.

**Soundness error** a window: CompileSlice_v4's, and for the close (N + N²)/2^128 over α and γ for R5 = W5 (N its
records) and 3·n(n − 1)/2^129 over δ for the n keys ``nrow`` and the window print; a root opened to two rows is a
SHA-512 collision. A faulty pool unit's charge in the sampled bound is its hits' bits, 512 for the root and SHA-512's
collision term.

**Cost** (`size`): the pool window is CompileSlice_v4's stream, bus and BlockKeys, two row digests (`Sha512RowSeg_v1`
over `data_bits` ``prow``: nine compressions a 8,192-bit segment), 2(NB − 1) KeyPrints, NB − 1 `PoolOpen` and
`PoolClose` steps and NM `PoolCopies`.

**The reference** is `reference`: CompileSlice_v4's run of the window on the stream `opened` makes from the row, and the
row it leaves (`closed`), None for ``next`` where a block is still held at the window's end. `arguments` gives both
units' rows, `run_compile` and `run_pool` their verdicts.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import memcheck as MC
from verity.ml.circuits.boolean.forms import OR, ZERO, and_reduce, const_bits
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array, Tuple
from verity.ml.commitments import rowleaf as RL
from verity.ml.commitments.gates import block_key as BK, rowleaf as RG
from verity.core.protocols.certifier import compile_slice as CS, replay_chain as RC
from verity.ml.protocols.certifier import cache_step_v3 as V3S, compile_slice_v4 as V4
from verity.core.protocols.certifier.head_check import BIT, _coll, _Consts, _part, _refs, ands

STATICS = V4.STATICS
COMPILE_FLAGS = V4.FLAGS[:-1] + ("calls",)
POOL_FLAGS = ("pool", "open", "close")
COMPILE_ACCEPT = (1 << len(COMPILE_FLAGS)) - 1
POOL_ACCEPT = (1 << len(POOL_FLAGS)) - 1
COMPILE_ROWS = ("adv", "ing", "req", "srv", "cout", "res", "calls")
POOL_ROWS = ("req", "calls", "pin", "pout", "coin", "prow", "nrow", "chk")
#: each input row's class (CompileSlice_v4's, and the roots registered, the rows under them checked)
CLASSES = {"adv": "advice", "ing": "ingress", "req": "registered", "srv": "registered", "cout": "registered",
           "res": "registered", "calls": "registered", "pin": "registered", "pout": "registered", "coin": "coin",
           "prow": "checked", "nrow": "checked", "chk": "checked"}
PADDED = tuple(r for r in CLASSES if CLASSES[r] == "registered")
LINKS = ("prev", "next", "cprev", "cnext")
#: the close's record tag, past the memories'
COPIES = 5
NOKEY = bytes(64)
T3 = V3S.TAGS
F = V4.F
COINS = V4.COINS


def row_fields(S) -> list[tuple[str, int, int]]:
    """The pool row's fields (name, count, bits)."""
    NW, n = V4.shape(S).Z3.NW, S.NB - 1
    return [("head", 1, NW), ("tail", 1, NW)] + [(f, n, NW) for f in LINKS + ("mlast",)] + [("key", n, 512)]


def fields(S) -> dict[str, list[tuple[str, int, int]]]:
    f4 = V4.fields(S)
    return {**{r: f4[r] for r in ("adv", "ing", "req", "srv", "cout")}, "res": f4["chk"][:4],
            "calls": [(name, V4.nrec(S), sum(BL.elements(t))) for name, t in V4._rec_types(S)],
            "pin": [("root", 1, 512)], "pout": [("root", 1, 512)], "coin": f4["coin"], "prow": row_fields(S),
            "nrow": row_fields(S), "chk": f4["chk"][4:]}


def data_bits(S) -> dict[str, int]:
    """Each input row's bits before its padding."""
    return {r: RC._bits_of(fs) for r, fs in fields(S).items()}


def row_bits(S) -> dict[str, int]:
    return {r: CS._words_of(b) if r in PADDED else b for r, b in data_bits(S).items()}


def free_width(S, widths: dict[str, int]) -> int:
    """The input bits of no class but advice among `widths` (a unit's rows and their bits)."""
    if not set(widths) <= set(CLASSES):
        raise ValueError(f"inputs {sorted(set(widths) - set(CLASSES))} aren't classified rows")
    return sum(b for r, b in widths.items() if CLASSES[r] == "advice")


# --- the circuit ------------------------------------------------------------------------------------------------------------


def _calls_forms(C, S, flags, recs, calls, res):
    nres = data_bits(S)["res"]
    ok = and_reduce(C, [RC._eq(C, list(recs), list(calls)), RC._zero(C, list(res[nres:]))])
    return list(flags) + [ok] + [ZERO] * (16 - len(COMPILE_FLAGS))


def _sig_calls(S):
    bits = row_bits(S)
    return ((("flags", Array(len(COMPILE_FLAGS) - 1, BIT)), ("recs", Array(data_bits(S)["calls"], BIT)),
             ("calls", Array(bits["calls"], BIT)), ("res", Array(bits["res"], BIT))), Array(16, BIT))


def _open_forms(C, S, carry, b, prt, cprev, mlast, alpha, gamma):
    R, W = list(carry[:128]), list(carry[128:])
    head = C.AND(V3S._nz(C, list(prt)), V3S._z(C, list(cprev)))
    tag, zt = const_bits(T3["M"], V3S.G), [ZERO] * S.TW
    R2 = MC._fold(C, alpha, gamma, R, head, [*tag, *prt, *[ZERO] * 128, *zt])
    W2 = MC._fold(C, alpha, gamma, W, head, [*tag, *prt, *V3S._pad([*b, *mlast], 128), *zt])
    return [R2 + W2, [head]]


def _sig_open(S):
    NW = V3S.widths(S).NW
    carry = Array(256, BIT)
    return ((("carry", carry), ("b", Array(NW, BIT)), ("print", F), ("cprev", Array(NW, BIT)), ("mlast", Array(NW, BIT)),
             ("alpha", F), ("gamma", F)), Tuple(carry, BIT))


def _close_forms(C, S, carry, b, prev, nxt, cprev, cnext, mlast, prt, bf, kf, alpha, gamma):
    Z = V3S.widths(S)
    R5, ok = list(carry[:128]), carry[128]
    ref = bf[Z.off["ref"]:Z.off["ref"] + S.REFW]
    head = C.AND(V3S._nz(C, list(kf)), V3S._z(C, list(cprev)))
    good = and_reduce(C, [V3S._eq(C, list(bf[:4 * Z.NW]), [*prev, *nxt, *cprev, *cnext]), V3S._z(C, list(ref)),
                          V3S._eq(C, list(kf), list(prt)), OR(C, head, V3S._z(C, list(mlast)))])
    R2 = MC._fold(C, alpha, gamma, R5, head,
                  [*const_bits(COPIES, V3S.G), *kf, *V3S._pad([*b, *mlast], 128), *[ZERO] * S.TW])
    return [R2 + [C.AND(ok, good)], [good]]


def _sig_close(S):
    NW = V3S.widths(S).NW
    carry, w = Array(129, BIT), Array(NW, BIT)
    return ((("carry", carry), ("b", w), ("prev", w), ("next", w), ("cprev", w), ("cnext", w), ("mlast", w),
             ("print", F), ("bf", F), ("kf", F), ("alpha", F), ("gamma", F)), Tuple(carry, BIT))


def _copies_forms(C, S, carry, en, a, v, alpha, gamma):
    NW = V3S.widths(S).NW
    on = C.AND(en[0], V3S._nz(C, list(v[:NW])))
    W2 = MC._fold(C, alpha, gamma, list(carry), on,
                  [*const_bits(COPIES, V3S.G), *a, *V3S._pad(list(v[:2 * NW]), 128), *[ZERO] * S.TW])
    return [W2, [on]]


def _sig_copies(S):
    V3S.widths(S)
    return (("carry", F), ("en", BIT), ("a", F), ("v", F), ("alpha", F), ("gamma", F)), Tuple(F, BIT)


def _end_width(S) -> int:
    return 1 + 2 * V3S.widths(S).NW + S.TW + 128


def _pool_verdict(C, S, end, bus, d_in, pin, d_out, pout, ends, cok, R5, W5):
    NW, TT = V3S.widths(S).NW, S.TW
    ok, qe, calls, P = end[0], end[1:1 + 2 * NW], end[1 + 2 * NW:1 + 2 * NW + TT], end[1 + 2 * NW + TT:]
    pool = and_reduce(C, [ok, RC._eq(C, calls, bus[:TT]), RC._eq(C, P, bus[TT:])])
    close = and_reduce(C, [RC._eq(C, d_out, pout), RC._eq(C, qe, ends), cok[0], RC._eq(C, R5, W5)])
    return [pool, RC._eq(C, d_in, pin), close] + [ZERO] * (16 - len(POOL_FLAGS))


def _sig_pool_verdict(S):
    NW, R = V3S.widths(S).NW, Array(512, BIT)
    return ((("end", Array(_end_width(S), BIT)), ("bus", Array(S.TW + 128, BIT)), ("d_in", R), ("pin", R),
             ("d_out", R), ("pout", R), ("ends", Array(2 * NW, BIT)), ("cok", BIT), ("R5", F), ("W5", F)),
            Array(16, BIT))


CallsVerdict = _part(
    "CallsVerdict", STATICS, _sig_calls, _calls_forms,
    "the compile slice's verdict: the steps' flags, and calls, 1 iff the steps' bus records are the registered calls row "
    "and the padding of the calls and results rows is zero")
PoolOpen = _part(
    "PoolOpen", V3S.SLOT_STATICS, _sig_open, _open_forms,
    "one block of the opened pool row: where its key's print is nonzero and it is the key's first copy, the key's M "
    "cell's initial record (first the block, last its mlast) folded into the memory check, its zero record out")
PoolClose = _part(
    "PoolClose", V3S.SLOT_STATICS, _sig_close, _close_forms,
    "one block of the pool row a window leaves: its final links the row's, no holder, its final key print the print of "
    "the row's key, mlast 0 but at a key's first copy; a first copy's (print, block, mlast) folded into R5")
PoolCopies = _part(
    "PoolCopies", V3S.SLOT_STATICS, _sig_copies, _copies_forms,
    "one entry of the M run: where its final first copy is nonzero, (key, first, last) folded into W5")
PoolVerdict = _part(
    "PoolVerdict", V3S.SLOT_STATICS, _sig_pool_verdict, _pool_verdict,
    "the pool window's verdict: pool, the stream's check with its bus product and call count the calls row's; open, the "
    "opened row's digest the input root; close, the left row's digest the output root, its queue ends the stream's, "
    "every block closed and R5 = W5")
PARTS = (CallsVerdict, PoolOpen, PoolClose, PoolCopies, PoolVerdict)


def _statics(S) -> dict:
    return {k: getattr(S, k) for k in STATICS}


def _compile_sig(S):
    bits = row_bits(S)
    return tuple((r, Array(bits[r], BIT)) for r in COMPILE_ROWS), Array(16, BIT)


def _compile_body(B, S, adv, ing, req, srv, cout, res, calls):
    st, nres, nf = _statics(S), data_bits(S)["res"], len(COMPILE_FLAGS) - 1
    out = _refs(B.call(bind(V4.CompileWindow, **st), adv, ing, req, srv, cout,
                       _coll(Array(nres, BIT), _refs(res)[:nres])))
    return B.call(bind(CallsVerdict, **st), _coll(Array(nf, BIT), out[:nf]), _coll(Array(len(out) - nf, BIT), out[nf:]),
                  calls, res)


def _pool_sig(S):
    bits = row_bits(S)
    return tuple((r, Array(bits[r], BIT)) for r in POOL_ROWS), Array(16, BIT)


def _pool_body(B, S, req, calls, pin, pout, coin, prow, nrow, chk):
    Kc = _Consts(B)
    sh = V4.shape(S)
    S3, TT, CW, RW, NW, G = sh.S3, sh.TT, sh.CW, sh.S3.RW, sh.Z3.NW, V3S.G
    st3 = {k: getattr(S3, k) for k in V3S.SLOT_STATICS}
    slot_p, entry_p, end_p = (bind(d, **st3) for d in V3S.PARTS)
    open_p, close_p, copies_p, ver_p = (bind(d, **st3) for d in PARTS[1:])
    F_, n, nb = fields(S), S.NB - 1, data_bits(S)["prow"]
    ck = RC._split(_refs(chk), F_["chk"])
    co = _refs(coin)
    alpha, gamma, delta = (_coll(F, co[128 * i:128 * (i + 1)]) for i in range(3))

    def col(t, elems):
        return _coll(Array(len(elems), t), [x for e in elems for x in e])

    cl = RC._split(_refs(calls), F_["calls"])
    recs = [col(t, cl[name]) for name, t in V4._rec_types(S)]
    bus = _refs(B.scan(bind(V4.CompileBus, **_statics(S)), _coll(Array(TT + 128, BIT), Kc.bits(0, TT) + Kc.bits(1, 128)),
                       recs, [alpha]))[:TT + 128]
    prints = V4.block_keys(B, Kc, S, req, delta)

    pr, nr = RC._split(_refs(prow), F_["prow"]), RC._split(_refs(nrow), F_["nrow"])
    d_in, d_out = (_refs(RG.digest_calls(B, _refs(x), nb)) for x in (prow, nrow))
    kp = bind(BK.KeyPrint)
    p0, pf = ([_refs(B.call(kp, delta, _coll(BK.DIGEST, k))) for k in r["key"]] for r in (pr, nr))

    lay, elay = V3S._carry_layout(S3), V3S._entry_layout(S3)
    size, esize = V3S._size(lay), V3S._size(elay)
    carry_t, ecarry_t = Array(size, BIT), Array(esize, BIT)
    rest = size - (1 + 3 * 128 + 2 * TT + 2 * NW)
    init = (Kc.bits(1, 1) + Kc.bits(1, 128) * 3 + Kc.bits(1, TT) + Kc.bits(0, TT) + pr["head"][0] + pr["tail"][0]
            + Kc.bits(0, rest))
    xs = [col(t, ck[name]) for name, t in (("s_start", BIT), ("s_kind", Array(3, BIT)), ("s_r", Array(RW, BIT)),
                                            ("s_a", Array(CW, BIT)), ("s_b", Array(CW, BIT)), ("s_v", F),
                                            ("s_t", Array(TT, BIT)))]
    done = _refs(B.scan(slot_p, _coll(carry_t, init), xs, [alpha, gamma]))[:size]
    fin = V3S._unpack(lay, done)

    blocks, w = col(Array(NW, BIT), [Kc.bits(b, NW) for b in range(1, S.NB)]), Array(NW, BIT)
    op = _refs(B.scan(open_p, _coll(Array(256, BIT), fin.R + fin.W),
                      [blocks, col(F, p0), col(w, pr["cprev"]), col(w, pr["mlast"])], [alpha, gamma]))[:256]

    def entries(R, W, ok, en, tag, a, v0, v, t):
        init = R + W + ok + Kc.bits(1, 1) + Kc.bits(0, 1 + G + 128)
        xs = [col(BIT, en), col(Array(G, BIT), tag), col(F, a), col(F, v0), col(F, v), col(Array(TT, BIT), t)]
        return V3S._unpack(elay, _refs(B.scan(entry_p, _coll(ecarry_t, init), xs, [alpha, gamma]))[:esize])

    def v0(tag, a):
        if tag == T3["B"]:
            return [x for f in LINKS for x in pr[f][a - 1]] + Kc.bits(0, 128 - 4 * NW)
        if tag == T3["K"]:
            return p0[a - 1]
        return prints[(a & ((1 << RW) - 1), a >> RW)] if tag == T3["RK"] else Kc.bits(0, 128)

    dn = V4.dense(S)
    e1 = entries(op[:128], op[128:], fin.ok, [Kc.bits(1, 1)] * len(dn), [Kc.bits(tag, G) for tag, _a in dn],
                 [Kc.bits(a, 128) for _t, a in dn], [v0(tag, a) for tag, a in dn], ck["d_v"], ck["d_t"])
    zero = Kc.bits(0, 1)[0]
    e2 = entries(e1.R, e1.W, e1.ok, ck["m_en"], [[e[0] if T3["M"] >> j & 1 else zero for j in range(G)] for e in ck["m_en"]],
                 ck["m_a"], [Kc.bits(0, 128)] * S.NM, ck["m_v"], ck["m_t"])

    cz = _refs(B.scan(close_p, _coll(Array(129, BIT), Kc.bits(1, 128) + Kc.bits(1, 1)),
                      [blocks] + [col(w, nr[f]) for f in LINKS + ("mlast",)]
                      + [col(F, pf), col(F, ck["d_v"][:n]), col(F, ck["d_v"][n:2 * n])], [alpha, gamma]))[:129]
    W5 = _refs(B.scan(copies_p, _coll(F, Kc.bits(1, 128)), [col(BIT, ck["m_en"]), col(F, ck["m_a"]), col(F, ck["m_v"])],
                      [alpha, gamma]))[:128]
    end = _refs(B.call(end_p, _coll(carry_t, done), _coll(ecarry_t, [x for f, _w in elay for x in getattr(e2, f)])))
    R512 = Array(512, BIT)
    return B.call(ver_p, _coll(Array(len(end), BIT), end), _coll(Array(TT + 128, BIT), bus), _coll(R512, d_in), pin,
                  _coll(R512, d_out), pout, _coll(Array(2 * NW, BIT), nr["head"][0] + nr["tail"][0]),
                  _coll(BIT, cz[128:]), _coll(F, cz[:128]), _coll(F, W5))


CompileSlice = CompositeDefinition(
    "CompileSlice", 5, STATICS, _compile_sig, _compile_body, conformance="exact-model-tested",
    doc="the compile slice of one window, its pool carried out to the pool window: CompileSlice_v4's load and steps from "
        "the registered results, and its pool calls' bus records held to the registered calls row (verdict bits load, "
        "steps, cache, served, carry, calls)")
PoolWindow = CompositeDefinition(
    "PoolWindow", 1, STATICS, _pool_sig, _pool_body, conformance="exact-model-tested",
    doc="one window of the block pool chained between windows: the pool row under the input root opened as the memory "
        "check's initial values, CacheStep_v3's stream of NS slots run from it over the registered calls row's bus, and "
        "the pool row the stream leaves, every block unheld, committed as the output root (verdict bits pool, open, "
        "close)")


def compile_unit(**st):
    return bind(CompileSlice, **st)


def pool_unit(**st):
    return bind(PoolWindow, **st)


statics = V4.statics


def demo() -> tuple[dict, CS.Slice]:
    """`compile_slice_v4.demo`'s class and slice with NM = 16 M entries: a window from a carried pool lists the keys the
    pool caches (to NB − 1 = 5) beside the ones it touches, which evictions of carried keys raise (to 11 in the tests'
    random windows)."""
    st, sl = V4.demo()
    return {**st, "NM": 16}, sl


def demo_async() -> tuple[dict, CS.Slice]:
    """`compile_slice_v4.demo_async`'s class and slice with NM = 16, as `demo`."""
    st, sl = V4.demo_async()
    return {**st, "NM": 16}, sl


# --- the specification ---------------------------------------------------------------------------------------------------


class PoolRow(NamedTuple):
    """The pool between windows (`row_fields`), block b's fields at index b − 1."""
    head: int
    tail: int
    prev: tuple[int, ...]
    next: tuple[int, ...]
    cprev: tuple[int, ...]
    cnext: tuple[int, ...]
    mlast: tuple[int, ...]
    key: tuple[bytes, ...]


def empty_row(S) -> PoolRow:
    """The empty pool: every block free, in order, nothing cached (`cache_step_v3.empty_pool`'s)."""
    n = S.NB - 1
    return PoolRow(1, n, tuple(range(n)), tuple(b + 1 if b < n else 0 for b in range(1, S.NB)), (0,) * n, (0,) * n,
                   (0,) * n, (NOKEY,) * n)


def encode_row(S, row: PoolRow) -> list[int]:
    """The row's bits (`row_fields`)."""
    NW, bits = V4.shape(S).Z3.NW, CS._bits
    out = bits(row.head, NW) + bits(row.tail, NW)
    for f in LINKS + ("mlast",):
        out += [b for v in getattr(row, f) for b in bits(v, NW)]
    return out + [b for k in row.key for b in RG.digest_bits(k)]


def decode_row(S, bits: Sequence[int]) -> PoolRow:
    """The row whose bits (`encode_row`) are ``bits``."""
    f = RC._split(list(bits), row_fields(S))
    val = [CS._int(e) for e in f["head"] + f["tail"]]
    return PoolRow(*val, *(tuple(CS._int(e) for e in f[k]) for k in LINKS + ("mlast",)),
                   tuple(RG.digest_bytes(e) for e in f["key"]))


def root(S, row: PoolRow) -> bytes:
    """The row's root: its unsalted `Sha512RowSeg_v1` digest."""
    return RL.sha512_row_seg_digest(*RL.bit_row((encode_row(S, row), 1)))


def genesis(S) -> bytes:
    return root(S, empty_row(S))


def _heads(S, row: PoolRow, delta: int) -> dict[int, int]:
    """M's cells the row opens: each key's print, where nonzero at a first copy b, to (first b, last mlast)."""
    NW = V4.shape(S).Z3.NW
    return {p: b | row.mlast[b - 1] << NW for b, p in ((b, BK.key_print(row.key[b - 1], delta)) for b in range(1, S.NB))
            if p and not row.cprev[b - 1]}


def opened(S, row: PoolRow, sl: CS.Slice, delta: int) -> V3S.Stream:
    """CacheStep_v3's stream from the pool `row`, its keys printed under δ and RK's the window's (`PoolWindow_v1`'s
    opening). A link past the blocks reads as a block of value 0, outside the touched list, which `reference` refuses."""
    S3 = V4.shape(S).S3
    rk, m = V4.rk_keys(S, sl, delta), _heads(S, row, delta)
    prints = [BK.key_print(k, delta) for k in row.key]

    def initial(tag: int, addr: int) -> int:
        if tag in (T3["B"], T3["K"]) and not 1 <= addr < S.NB:
            return 0
        if tag == T3["B"]:
            return V3S.bput(S3, {f: getattr(row, f)[addr - 1] for f in LINKS})
        if tag == T3["K"]:
            return prints[addr - 1]
        if tag == T3["M"]:
            return m.get(addr, 0)
        return rk.get(addr, 0) if tag == T3["RK"] else 0

    return V3S.Stream(S3, initial, row.head, row.tail)


def _chains(S, sl: CS.Slice) -> list[bytes]:
    """Every arrival slot's CU block keys (`compile_slice_v4.rk_keys`' before printing)."""
    n, tok, out = S.CU * S.B, V4.key_tok(S), []
    for k in range(S.K):
        toks = tuple(sl.tokens[k])[:n] if k < len(sl.tokens) else ()
        out += BK.chain(toks + (0,) * (n - len(toks)), S.B, tok)
    return out


def closed(S, stream: V3S.Stream, row: PoolRow, sl: CS.Slice, delta: int) -> tuple[PoolRow, bool]:
    """The pool row `stream` leaves, run on `opened`'s stream of `row` and `sl`, and whether it holds no block."""
    sh = V4.shape(S)
    S3, NW = sh.S3, sh.Z3.NW
    keys = {BK.key_print(k, delta): k for k in list(row.key) + _chains(S, sl) if k != NOKEY}
    links, mlast, key, quiet = {f: [] for f in LINKS}, [], [], True
    for b in range(1, S.NB):
        d = V3S.bsplit(S3, stream.value(T3["B"], b))
        quiet &= d["ref"] == 0
        for f in LINKS:
            links[f].append(d[f])
        p = stream.value(T3["K"], b)
        key.append(keys[p] if p else NOKEY)
        mlast.append(stream.value(T3["M"], p) >> NW & ((1 << NW) - 1) if p and not d["cprev"] else 0)
    return PoolRow(stream.head, stream.tail, *(tuple(links[f]) for f in LINKS), tuple(mlast), tuple(key)), quiet


def _mrun(S, stream: V3S.Stream, carried: dict[int, int]) -> list[tuple[int, int, int]]:
    """The M run: (address, final value, final time) of every key the stream touches or the row opens, sorted."""
    cells = {a: (v, 0) for a, v in carried.items()}
    cells.update({a: stream.cells[(tag, a)] for tag, a in stream.first if tag == T3["M"]})
    return sorted((a, v, t) for a, (v, t) in cells.items())


def with_mrun(S, chk: Sequence[int], ms: Sequence[tuple[int, int, int]]) -> list[int]:
    """The checked row `chk` with its M run the entries `ms` (address, final value, final time), unused entries zero."""
    if len(ms) > S.NM:
        raise ValueError(f"an M run of {len(ms)} entries passes NM = {S.NM}")
    TT, bits, used = V4.shape(S).TT, CS._bits, len(ms)
    ms = list(ms) + [(0, 0, 0)] * (S.NM - used)
    out = list(chk[:len(chk) - S.NM * (1 + 256 + TT)]) + [1] * used + [0] * (S.NM - used)
    out += [b for a, _v, _t in ms for b in bits(a, 128)] + [b for _a, v, _t in ms for b in bits(v, 128)]
    return out + [b for _a, _v, t in ms for b in bits(t, TT)]


def reference(S, sl: CS.Slice, row: PoolRow | None = None, delta: int = COINS[2]) -> SimpleNamespace | None:
    """Both units' registered rows and the pool window's checked witness for `sl` from the pool `row` (default the empty
    pool) under the print coin δ (unpadded): ``srv``, ``cout``, ``res`` and ``chk``, the row ``left`` the stream leaves
    and ``next``, it where no block is held at the end (else None). None where CompileSlice_v4's reference refuses the
    window on `opened`'s stream, or the M run passes NM entries."""
    row = empty_row(S) if row is None else row
    carried = _heads(S, row, delta)
    r4 = V4.reference(S, sl, delta, opened(S, row, sl, delta))
    if r4 is None:
        return None
    stm = r4.scan.stream
    ms = _mrun(S, stm, carried)
    if len(ms) > S.NM:
        return None
    nres = data_bits(S)["res"]
    chk = with_mrun(S, r4.chk[nres:], ms)
    left, quiet = closed(S, stm, row, sl, delta)
    return SimpleNamespace(srv=r4.srv, cout=r4.cout, res=r4.chk[:nres], chk=chk, left=left, next=left if quiet else None,
                           mrun=ms, scan=r4.scan)


def _padded(S, r: str, bits: Sequence[int]) -> list[int]:
    n = data_bits(S)[r]
    if len(bits) != n:
        raise ValueError(f"row {r} holds {n} bits, not {len(bits)}")
    return list(bits) + [0] * (row_bits(S)[r] - n)


def records(st: dict, args: Sequence[dict[str, list[int]]]) -> list[list[int]]:
    """The steps' bus records (``calls``' bits, unpadded) on each of `args`' rows: `CompileSlice_v4.CompileWindow`'s."""
    S = SimpleNamespace(**st)
    nres, nf = data_bits(S)["res"], len(COMPILE_FLAGS) - 1
    out = evaluate_bits(bind(V4.CompileWindow, **st), *[[a[r] for a in args] for r in COMPILE_ROWS[:5]],
                        [a["res"][:nres] for a in args]).reshape(len(args), -1).tolist()
    return [[int(b) for b in o[nf:]] for o in out]


def arguments(S, sl: CS.Slice, row: PoolRow | None = None, coins: Sequence[int] = COINS, **given) -> dict[str, list[int]]:
    """Both units' inputs for window `sl` from the pool `row` (default the empty pool) under the coins (α, γ, δ): every row
    of `COMPILE_ROWS` and `POOL_ROWS` (the registered ones zero-padded), each of `given` as given and the rest
    `reference`'s, ``calls`` the records the reference's rows make (zero where it refuses the window, the pool rows
    ``row`` and its root both in and out)."""
    row = empty_row(S) if row is None else row
    ref, nb = reference(S, sl, row, coins[2]), data_bits(S)
    zero = {r: [0] * nb[r] for r in ("srv", "cout", "res", "chk")}
    v4 = V4.arguments(S, sl, coins, srv=ref.srv if ref else zero["srv"], cout=ref.cout if ref else zero["cout"],
                      chk=[0] * V4.data_bits(S)["chk"])
    left = ref.left if ref else row
    out = {**{r: v4[r] for r in ("adv", "ing", "req", "srv", "cout", "coin")},
           "res": _padded(S, "res", ref.res if ref else zero["res"]), "chk": ref.chk if ref else zero["chk"],
           "pin": RG.digest_bits(root(S, row)), "prow": encode_row(S, row),
           "pout": RG.digest_bits(root(S, left)), "nrow": encode_row(S, left)}
    if "calls" not in given:
        out["calls"] = _padded(S, "calls", records(_statics(S), [out])[0])
    return {**out, **{r: list(v) for r, v in given.items()}}


def _run(unit, rows, args) -> list[int]:
    out = evaluate_bits(unit, *[[a[r] for a in args] for r in rows]).reshape(len(args), -1).tolist()
    return [sum(int(b) << j for j, b in enumerate(o)) for o in out]


def run_compile(st: dict, args: Sequence[dict[str, list[int]]]) -> list[int]:
    """The compile slice's verdict word on each of `args` (`arguments`' form), in one evaluation."""
    return _run(compile_unit(**st), COMPILE_ROWS, args)


def run_pool(st: dict, args: Sequence[dict[str, list[int]]]) -> list[int]:
    """The pool window's verdict word on each of `args` (`arguments`' form), in one evaluation."""
    return _run(pool_unit(**st), POOL_ROWS, args)


def size(**st) -> dict[str, int]:
    """Each unit's ANDs (an n-input AND n − 1), every input row's width (padded) and each unit's free (advice) width at
    statics `st`."""
    S, memo = SimpleNamespace(**st), {}
    bits = row_bits(S)
    return {"compile_ands": ands(compile_unit(**st), memo), "pool_ands": ands(pool_unit(**st), memo),
            **{f"{r}_bits": b for r, b in bits.items()},
            "compile_free_bits": free_width(S, {r: bits[r] for r in COMPILE_ROWS}),
            "pool_free_bits": free_width(S, {r: bits[r] for r in POOL_ROWS})}
