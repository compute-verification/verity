"""The compile slice with the block pool carried between windows: `CompileSlice_v5` (`compile_slice_v5`) cut in two
units, both proved every window, the pool passed from window to window as a committed root.

    CompileSlice_v6{R, W, A, M, T, L, B, H, CH, WM, FULL, PC, NB, IW, AS, EW, Q, D, P, K, CTW, CU, DA, NS, NM, NRB, NRK}
        (adv, ing, req, srv, cout, res, calls) -> verdict[16]
    PoolWindow_v1{R, W, A, M, T, L, B, H, CH, WM, FULL, PC, NB, IW, AS, EW, Q, D, P, K, CTW, CU, DA, NS, NM, NRB, NRK}
        (req, calls, pin, pout, coin, prow, nrow, src, chk) -> verdict[16]
    PageCopy_v1{CB} (keyed, fresh, wrote: Array<CB>, carried: Array<CB>, out: Array<CB>) -> verdict[16]

**Why.** CompileSlice_v5 runs each window from the empty pool, so a window never hits what an earlier one cached. Here
the pool between windows is one row (`row_fields`), committed as its unsalted `Sha512RowSeg_v1` digest, the root: a
window's pool unit opens the previous root, runs CacheStep_v3's stream from the pool it opens, and commits the pool it
leaves as the next root. Both units are proved every window, none sampled: window w + 1's ``pin`` is window w's
``pout``, one registered row both windows read, registered before window w + 1's coins are drawn (`PoolChain` is two
windows so wired). The first window's ``pin`` is the empty pool's root, `genesis`. A window whose proof fails ends its
chain, and the next starts again from `genesis`, as the first does. The window pays for its activity as CompileSlice_v5
does: its keys are the ragged row's blocks, its RB run the addresses its calls touch.

**The edge.** A window ends at its edge, and a request still open there (running, waiting, or certified and not yet
taken) is cut: its registered ``a_cut`` bit is 1 and its ``a_O`` the outputs it emitted by the edge, so the load reads
its O as L − P and its cap as 0 (`CompileCut`) and it doesn't leave at ``a_O``. After the last step `CompileEdge`
preempts every running slot in vLLM's preemption order, the last slot first, each a free on the bus whose returns are the
registered ``erets``, and holds the open requests to the cut ones: each running or waiting request cut, having emitted
its ``a_O``, each certified arrival not taken cut with ``a_O`` 0, and as many open as cut, so every other request left.
The next window takes each cut request's continuation as an ordinary arrival, (id, P + e) with its tokens its prompt
then its e released outputs, in the order the edge leaves them (`opened_at_edge`: the running in slot order, the
waiting, the arrivals not taken), and its prefix hits find the blocks the window cached, carried in the pool. At AS = 1
a request running at the edge refuses: its in-flight output isn't modeled at the edge.

**The carried pages.** The pool row carries the blocks' keys, and the carried port the pages they key. At the close
each keyed block b's page has a source, ``src``: the window's own write where the window allocated b (``fresh``, the
stream's owner nonzero), else row b of the previous window's carried port. Copy blocks read that source and write row b
of the window's carried port, each held by `PageCopy` (`copied`), and are sampled like any other stratum. After a pool
reset the carried port is empty, as the pool is.

**The pool row** (`verity.commitments.pool_row`), field-major, each element least significant bit first: the free
queue's ``head`` and ``tail`` (NW bits each), then per block b = 1 … NB − 1 its links ``prev``, ``next``, ``cprev``,
``cnext`` (B's), ``mlast`` (the key's last copy where b is its first copy, else 0) and ``key``, the 512-bit BlockKey_v1
key b is cached under (`Digest` layout, 0 for none). It holds no count, owner or stamp: a pool between windows holds no
block (every ref 0), and owners and stamps are one window's.

**The inputs**, each a field list (`fields`):
- ``adv`` (*advice*), ``ing`` (*ingress*), ``srv`` and ``cout`` (*registered*): CompileSlice_v5's.
- ``req`` (*registered*): CompileSlice_v5's, with each arrival's ``a_cut`` after its head fields.
- ``res`` (*registered*): the steps' results, CompileSlice_v5's first four checked fields (backs, hits, evs, rets), then
  the edge's R returns ``erets``.
- ``calls`` (*registered*): the steps' and the edge's `nrec` bus records, field-major as `CompileWindow` makes them.
- ``pin`` and ``pout`` (*registered*): the roots the window opens and leaves, 512 bits each.
- ``coin`` (the verifier's coins): α, γ and δ, drawn live: δ once every registered row is, α and γ once ``prow``,
  ``nrow`` and ``chk`` are committed.
- ``prow`` (*opened*): the pool row under ``pin``. The unit opens it against its root, so it is ``pin``'s up to a
  SHA-512 collision, and no input is free of it.
- ``nrow`` (*checked*): the pool row the window leaves, under ``pout``; its every field is held to the stream's finals.
- ``src`` (*checked*): per block b = 1 … NB − 1 ``keyed`` (its final key nonzero), then per block ``fresh`` (keyed, and
  its final owner nonzero: the window allocated it), held to the stream's finals by the close.
- ``chk`` (*checked*): CompileSlice_v5's stream witness less the results (its slots, its dense run's finals, its M and
  RB runs, the ragged blocks' numbers and times), the M run holding every key the stream touches and every key the
  opened pool caches.
A registered row is padded to whole words (`compile_slice.WORD`); its padding must be zero. `CLASSES` names each input's
class. The checked rows are computed checks, as CompileSlice_v5's witness is, and count as free input at their widths
until a Lean lemma shows the witnesses the unit accepts are unique (AGENTS.md's nondeterminism rule): `free_width`
counts the advice and the checked rows of the units' inputs.

**The compile slice**: `CompileWindow` (`CompileCut`, CompileSlice_v5's load, the arrivals' ids distinct, its steps from
``res``, each running slot's request number carried in its slot row, and `CompileEdge`), `CallsEq` scanned over its
records beside ``calls``', then `CallsVerdict`: CompileSlice_v5's flags load, steps, cache, served and carry, bit 5
``calls``, 1 iff its records are ``calls`` and the padding of ``res`` and ``calls`` is zero, and bit 6 ``edge``, the
edge's check. It has no coin. `COMPILE_ACCEPT` = 0x7F.

**The pool window**'s verdict, `POOL_ACCEPT` = 0xF:
- bit 0 ``pool``: the stream checks (`cache_step_v3.PoolEnd`), its bus product over ``calls`` is the stream's and its call
  count theirs (CompileSlice_v5's pool bit);
- bit 1 ``open``: ``prow``'s digest is ``pin``;
- bit 2 ``close``: ``nrow``'s digest is ``pout``, the stream's queue ends are ``nrow``'s and every block closes
  (`PoolClose`): its final links are ``nrow``'s, it has no holder, its final key print is the print of ``nrow``'s key,
  ``mlast`` is 0 but at a first copy and its source is ``src``'s; and the first copies with their last copies are the M
  run's nonzero final cells (R5 = W5 over α and γ, records of tag 5);
- bit 3 ``keys``: CompileSlice_v5's, the ragged row's runs well formed, all placed, and the slots' counts
  (`PoolCounts`).

**The opening** is the memory check's initial multiset, from ``prow`` alone: the stream starts from its queue ends; B's
initial values are its links, every count, owner and stamp 0; K's are its keys' prints under δ (`block_key.KeyPrint`, 0
for no key); RB's are 0 and RK's the ragged row's keys' prints, CompileSlice_v5's; M's are 0 in the touched list, and
`PoolOpen` multiplies, at each block b whose print is nonzero and whose cprev is 0, (M, print, 0, 0) into R and (M,
print, b | mlast << NW, 0) into W: the key's cell holds (first b, last mlast) before the stream, so the M run must list
it. No witness about the previous pool is free: a wrong opening is a ``prow`` other than the one under ``pin``.

**Soundness error** a window: CompileSlice_v5's, and for the close (N + N²)/2^128 over α and γ for R5 = W5 (N its
records) and 3·n(n − 1)/2^129 over δ for the n keys of ``nrow`` and the window's prints; a root opened to two rows is a
SHA-512 collision.

**Cost** (`size`, `parts`): the compile slice is `CompileCut`, CompileSlice_v5's load and steps, `CompileEdge`, its ids'
check and the calls row's (a `CallsEq` a record);
the pool window is CompileSlice_v5's stream, bus (over the edge's R records too) and ragged keys, two row digests
(`Sha512RowSeg_v1` over `data_bits` ``prow``), 2(NB − 1) KeyPrints, NB − 1 `PoolOpen` and `PoolClose` steps and NM
`PoolCopies`; its M run lists the carried keys beside the touched ones. A copy block is one `PageCopy`.

**The reference** is `reference`: CompileSlice_v5's run of the window, its cut arrivals' O their length past their prompt
(`effective`), on the stream `opened` makes from the row, the edge's frees on that stream (`edge`), the blocks' sources
(`sources`) and the row it leaves (`closed`), None for ``next`` where the edge doesn't hold or a block is still held
after it. `cut_open` cuts a window at its edge. `arguments` gives both units' rows, `run_compile` and `run_pool` their
verdicts, `run_copy` `PageCopy`'s.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Sequence

from verity.definitions.circuits.boolean import memcheck as MC
from verity.circuits.boolean.forms import ONE, ZERO, and_reduce, const_bits, mux, or_reduce
from verity.circuits.boolean.gf128 import mul_forms
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.evaluation.bits import evaluate_bits
from verity.circuits.types import Array, Tuple
from verity.commitments import pool_row as PR
from verity.definitions.commitments.gates import block_key as BK
from verity.commitments.gates import rowleaf as RG
from verity.protocols.certifier.verifier import cache_step as CST, compile_slice as CS, replay_chain as RC
from verity.protocols.certifier.verifier import replay_step as RS
from verity.protocols.certifier.prover import cache_step_v2 as C2, cache_step_v3 as V3S, compile_slice_v4 as V4
from verity.protocols.certifier.prover import compile_slice_v5 as V5
from verity.protocols.certifier.verifier.head_check import BIT, _coll, _Consts, _part, _refs, ands

STATICS = V5.STATICS
COMPILE_FLAGS = ("load", "steps", "cache", "served", "carry", "calls", "edge")
POOL_FLAGS = ("pool", "open", "close", "keys")
COMPILE_ACCEPT = (1 << len(COMPILE_FLAGS)) - 1
POOL_ACCEPT = (1 << len(POOL_FLAGS)) - 1
COPY_ACCEPT = 0x1
COMPILE_ROWS = ("adv", "ing", "req", "srv", "cout", "res", "calls")
POOL_ROWS = ("req", "calls", "pin", "pout", "coin", "prow", "nrow", "src", "chk")
#: each input row's class: CompileSlice_v5's, the results, calls and roots registered, the row under the input root opened
#: against it, and the row the window leaves and its blocks' sources checked (the module's docstring)
CLASSES = {"adv": "advice", "ing": "ingress", "req": "registered", "srv": "registered", "cout": "registered",
           "res": "registered", "calls": "registered", "pin": "registered", "pout": "registered", "coin": "coin",
           "prow": "opened", "nrow": "checked", "src": "checked", "chk": "checked"}
PADDED = tuple(r for r in CLASSES if CLASSES[r] == "registered")
IDS_STATICS = ("K", "IW")
COUNTS_STATICS = ("K", "CU")
REC_STATICS = ("K", "CU")
CUT_STATICS = ("K", "L", "AS")
EDGE_STATICS = V5.STEP_STATICS
COPY_STATICS = ("CB",)
LINKS = PR.LINKS
#: the close's record tag, past the memories'
COPIES = 5
NOKEY = PR.NOKEY
T3 = V3S.TAGS
F = V5.F
COINS = V5.COINS
_NRES = 4


def _bitlen(v: int) -> int:
    return max(int(v).bit_length(), 1)


def row_fields(S) -> list[tuple[str, int, int]]:
    """The pool row's fields (name, count, bits)."""
    return PR.fields(S.NB)


def nrec(S) -> int:
    """The bus records a window makes: its steps' (`compile_slice_v5.nrec`) and the edge's R frees."""
    return V5.nrec(S) + S.R


def fields(S) -> dict[str, list[tuple[str, int, int]]]:
    f5, nh = V5.fields(S), len(V5._head_fields(S))
    return {**{r: f5[r] for r in ("adv", "ing", "srv", "cout")},
            "req": f5["req"][:nh] + [("a_cut", S.K, 1)] + f5["req"][nh:],
            "res": f5["chk"][:_NRES] + [("erets", S.R, V5.shape(S).CW)],
            "calls": [(name, nrec(S), w) for (name, _t), w in zip(V4._rec_types(S), V5._rec_widths(S))],
            "pin": [("root", 1, 512)], "pout": [("root", 1, 512)], "coin": f5["coin"], "prow": row_fields(S),
            "nrow": row_fields(S), "src": [("keyed", S.NB - 1, 1), ("fresh", S.NB - 1, 1)], "chk": f5["chk"][_NRES:]}


def data_bits(S) -> dict[str, int]:
    """Each input row's bits before its padding."""
    return {r: RC._bits_of(fs) for r, fs in fields(S).items()}


def row_bits(S) -> dict[str, int]:
    return {r: CS._words_of(b) if r in PADDED else b for r, b in data_bits(S).items()}


def free_width(S, widths: dict[str, int]) -> int:
    """The free input bits among `widths` (a unit's rows and their bits): the advice and the checked rows (`CLASSES`; the
    module's docstring)."""
    if not set(widths) <= set(CLASSES):
        raise ValueError(f"inputs {sorted(set(widths) - set(CLASSES))} aren't classified rows")
    return sum(b for r, b in widths.items() if CLASSES[r] in ("advice", "checked"))


# --- the circuit ------------------------------------------------------------------------------------------------------------


def _ids_layout(S) -> list[tuple[str, int]]:
    return [("ok", 1), ("mask", S.K), ("ptr", S.K)]


def _ids(C, S, carry, valid_i, id_i, valid, ids):
    """Arrival slot i: 0 in ok where it repeats a valid earlier slot's id (CompileSlice_v5's arrivals' first half)."""
    c = V3S._unpack(_ids_layout(S), carry)
    IW = S.IW
    dup = C.AND(valid_i[0], or_reduce(C, [and_reduce(C, [c.mask[j], valid[j], CST._eq(C, ids[IW * j:IW * (j + 1)], id_i)])
                                          for j in range(S.K)]))
    out = [C.AND(c.ok[0], dup.inv())] + [V5.OR(C, m, p) for m, p in zip(c.mask, c.ptr)] + [ZERO] + list(c.ptr[:-1])
    return [out, [dup]]


def _sig_ids(S):
    carry = Array(V3S._size(_ids_layout(S)), BIT)
    return ((("carry", carry), ("valid_i", BIT), ("id_i", Array(S.IW, BIT)), ("valid", Array(S.K, BIT)),
             ("ids", Array(S.K * S.IW, BIT))), Tuple(carry, BIT))


def _counts_layout(S) -> list[tuple[str, int]]:
    return [("i", _bitlen(S.K)), ("P", 128), ("total", V5._total_width(S))]


def _counts(C, S, carry, n_i, alpha):
    """Arrival slot i's keyed block count: its factor (α + (i ‖ a_nb)) where a_nb > 0, and the counts' sum
    (CompileSlice_v5's arrivals' second half)."""
    c = V3S._unpack(_counts_layout(S), carry)
    on = V5._nz(C, n_i)
    P = mul_forms(C, c.P, V5._factor(C, on, alpha, list(c.i) + list(n_i)))
    return [RS._add(C, c.i, [ONE], _bitlen(S.K)) + P + RS._add(C, c.total, n_i, V5._total_width(S)), [on]]


def _sig_counts(S):
    carry = Array(V3S._size(_counts_layout(S)), BIT)
    return (("carry", carry), ("n_i", Array(_bitlen(S.CU), BIT)), ("alpha", F)), Tuple(carry, BIT)


def _cut_head(C, S, a_P, a_O, a_cut, *a_cap):
    """The O and cap of each arrival the load reads: a cut arrival's L − P and 0, so it doesn't leave at its a_O, and
    another's a_O and a_cap."""
    LW = _bitlen(S.L)
    out = []
    for k in range(S.K):
        P, O = list(a_P[LW * k:LW * (k + 1)]), list(a_O[LW * k:LW * (k + 1)])
        out += mux(C, a_cut[k], RS._sub(C, const_bits(S.L, LW), P, LW), O)
    if S.AS:
        out += [C.AND(c, x.inv()) for c, x in zip(a_cap[0], a_cut)]
    return out


def _sig_cut(S):
    LW, K = _bitlen(S.L), S.K
    caps = (("a_cap", Array(K, BIT)),) if S.AS else ()
    return ((("a_P", Array(K * LW, BIT)), ("a_O", Array(K * LW, BIT)), ("a_cut", Array(K, BIT))) + caps,
            Array(K * LW + K * S.AS, BIT))


def _edge(C, S, state, slots, erets, valid, ids, a_O, a_cut):
    """The edge after the window's last step: at AS = 0 each running slot's free in vLLM's preemption order, from the
    last slot down, then 1 in ok iff the open requests are the cut ones, each having emitted its a_O (the module's
    docstring); at AS = 1 it frees nothing, and a request still running refuses."""
    Sc, sh = CS.chain(S), V5._step_shape(S)
    Z, R, W, K, CW, RW, IW = RC.widths(Sc), S.R, S.W, S.K, sh.CW, sh.RW, S.IW
    st, p = RC._split(list(state), RC.row_fields(Sc)), RC._split(list(slots), V5.pool_fields(S))
    run, wait = [e[0] for e in st["run_valid"]], [e[0] for e in st["wait_valid"]]
    outs = [list(a_O[Z.LW * k:Z.LW * (k + 1)]) for k in range(K)]
    rets = [list(erets[CW * t:CW * (t + 1)]) for t in range(R)]
    cuts = [[c] for c in a_cut]
    bad = [C.AND(a_cut[k], valid[k].inv()) for k in range(K)]

    def held_to(sel, on, out):
        """`on` refuses unless the arrival `sel` picks is cut and its a_O is `out`."""
        c, o = C2._pick(C, sel, cuts)[0], C2._pick(C, sel, outs)
        bad.append(C.AND(on, C.AND(c, RC._eq(C, list(out), o)).inv()))

    for j in range(R):
        if S.AS:
            bad.append(run[j])
        else:
            held_to([CST._eq(C, p["rq"][j], const_bits(k, RW)) for k in range(K)], run[j], st["run_out"][j])
    for w in range(W):
        held_to([C.AND(valid[k], CST._eq(C, ids[IW * k:IW * (k + 1)], st["wait_id"][w])) for k in range(K)], wait[w],
                st["wait_out"][w])
    QW = max(Z.QW, _bitlen(K))
    left = [C.AND(valid[k], RS._lt(C, const_bits(k, QW), st["qh"][0]).inv()) for k in range(K)]
    for k in range(K):
        bad.append(C.AND(left[k], C.AND(a_cut[k], RC._zero(C, outs[k])).inv()))
    NW = _bitlen(R + W + K)
    opened = RS._popcount(C, run + wait + left, NW)

    LB, recs = S.B.bit_length() - 1, []
    for t in range(R):
        j = R - 1 - t
        held = RS._cdiv(C, st["run_comp"][j], LB, CW)
        on = ZERO if S.AS else C.AND(run[j], V5._nz(C, held))
        bad.append(C.AND(on.inv(), V5._nz(C, rets[t])))
        recs.append(([on], const_bits(V3S.KINDS["free"], 3), RS._pad(p["rq"][j], RW), held, [ZERO] * CW, rets[t],
                     [ZERO] * CW))
    ok = C.AND(or_reduce(C, bad).inv(), RC._eq(C, opened, RS._popcount(C, list(a_cut), NW)))
    return [ok] + [x for f in range(7) for r in recs for x in r[f]]


def _sig_edge(S):
    Sc, sh = CS.chain(S), V5._step_shape(S)
    LW = RC.widths(Sc).LW
    return ((("state", Array(RC.width(Sc), BIT)), ("slots", Array(V5.pool_width(S), BIT)),
             ("erets", Array(S.R * sh.CW, BIT)), ("valid", Array(S.K, BIT)), ("ids", Array(S.K * S.IW, BIT)),
             ("a_O", Array(S.K * LW, BIT)), ("a_cut", Array(S.K, BIT))), Array(1 + S.R * rec_width(S), BIT))


def _copy_forms(C, S, keyed, fresh, wrote, carried, out):
    want = [C.AND(keyed[0], x) for x in mux(C, fresh[0], list(wrote), list(carried))]
    return [RC._eq(C, list(out), want)] + [ZERO] * 15


def _sig_copy(S):
    blk = Array(S.CB, BIT)
    return (("keyed", BIT), ("fresh", BIT), ("wrote", blk), ("carried", blk), ("out", blk)), Array(16, BIT)


def _window_width(S) -> int:
    """`CompileWindow`'s output: the load's, ids', the steps' three and the edge's flags, the last state, window and slot
    rows, the records."""
    Sc = CS.chain(S)
    return 6 + RC.width(Sc) + RC.window_width(Sc) + V5.pool_width(S) + data_bits(S)["calls"]


def _pad_width(S, r: str) -> int:
    """Row r's padding as the verdict reads it: at least one bit (a constant zero where the row fills its words)."""
    return max(row_bits(S)[r] - data_bits(S)[r], 1)


def rec_width(S) -> int:
    """A bus record's bits, from `REC_STATICS` alone: on, kind, r and a, b, x, y (`compile_slice_v5._rec_widths`)."""
    return 1 + 3 + _bitlen(S.K) + 4 * (_bitlen(S.CU) + 1)


def _calls_eq(C, S, ok, rec, call):
    """One bus record: 1 in ok while each record so far is the calls row's."""
    e = RC._eq(C, list(rec), list(call))
    return [[C.AND(ok[0], e)], [e]]


def _sig_calls_eq(S):
    rec = Array(rec_width(S), BIT)
    return (("carry", Array(1, BIT)), ("rec", rec), ("call", rec)), Tuple(Array(1, BIT), BIT)


def _calls_forms(C, S, head, rows, cout, eq, spad, rpad, xpad, cpad):
    load, ids, oks, edge = head[0], head[1], head[2:5], head[5]
    flags = [and_reduce(C, [load, ids, RC._zero(C, list(rpad))]), oks[0], oks[1],
             C.AND(oks[2], RC._zero(C, list(spad))), RC._eq(C, list(rows), list(cout)),
             and_reduce(C, [eq[0], RC._zero(C, list(xpad)), RC._zero(C, list(cpad))]), edge]
    return flags + [ZERO] * (16 - len(flags))


def _sig_calls(S):
    Sc, bits = CS.chain(S), row_bits(S)
    return ((("head", Array(6, BIT)), ("rows", Array(RC.width(Sc) + RC.window_width(Sc) + V5.pool_width(S), BIT)),
             ("cout", Array(bits["cout"], BIT)), ("eq", Array(1, BIT)), ("spad", Array(_pad_width(S, "srv"), BIT)),
             ("rpad", Array(_pad_width(S, "req"), BIT)), ("xpad", Array(_pad_width(S, "res"), BIT)),
             ("cpad", Array(_pad_width(S, "calls"), BIT))), Array(16, BIT))


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


def _close_forms(C, S, carry, b, prev, nxt, cprev, cnext, mlast, prt, bf, kf, keyed, fresh, alpha, gamma):
    Z = V3S.widths(S)
    R5, ok = list(carry[:128]), carry[128]
    ref, owner = bf[Z.off["ref"]:Z.off["ref"] + S.REFW], bf[Z.off["owner"]:Z.off["owner"] + S.RW]
    has = V3S._nz(C, list(kf))
    head = C.AND(has, V3S._z(C, list(cprev)))
    new = C.AND(has, V3S._nz(C, list(owner)))
    good = and_reduce(C, [V3S._eq(C, list(bf[:4 * Z.NW]), [*prev, *nxt, *cprev, *cnext]), V3S._z(C, list(ref)),
                          V3S._eq(C, list(kf), list(prt)), V5.OR(C, head, V3S._z(C, list(mlast))),
                          (keyed[0] ^ has).inv(), (fresh[0] ^ new).inv()])
    R2 = MC._fold(C, alpha, gamma, R5, head,
                  [*const_bits(COPIES, V3S.G), *kf, *V3S._pad([*b, *mlast], 128), *[ZERO] * S.TW])
    return [R2 + [C.AND(ok, good)], [good]]


def _sig_close(S):
    NW = V3S.widths(S).NW
    carry, w = Array(129, BIT), Array(NW, BIT)
    return ((("carry", carry), ("b", w), ("prev", w), ("next", w), ("cprev", w), ("cnext", w), ("mlast", w),
             ("print", F), ("bf", F), ("kf", F), ("keyed", BIT), ("fresh", BIT), ("alpha", F), ("gamma", F)),
            Tuple(carry, BIT))


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


def _pool_verdict(C, S, end, bus, d_in, pin, d_out, pout, ends, cok, R5, W5, kok, left, live, r, k, Pk, Pa, alpha):
    NW, TT = V3S.widths(S).NW, S.TW
    ok, qe, calls, P = end[0], end[1:1 + 2 * NW], end[1 + 2 * NW:1 + 2 * NW + TT], end[1 + 2 * NW + TT:]
    pool = and_reduce(C, [ok, RC._eq(C, calls, bus[:TT]), RC._eq(C, P, bus[TT:])])
    close = and_reduce(C, [RC._eq(C, d_out, pout), RC._eq(C, qe, ends), cok[0], RC._eq(C, R5, W5)])
    last = mul_forms(C, list(Pk), V5._factor(C, live[0], alpha, list(r) + RS._add(C, list(k), [ONE], S.KW)))
    keys = and_reduce(C, [kok[0], RC._zero(C, list(left)), RC._eq(C, last, list(Pa))])
    return [pool, RC._eq(C, d_in, pin), close, keys] + [ZERO] * (16 - len(POOL_FLAGS))


def _sig_pool_verdict(S):
    NW, R = V3S.widths(S).NW, Array(512, BIT)
    return ((("end", Array(_end_width(S), BIT)), ("bus", Array(S.TW + 128, BIT)), ("d_in", R), ("pin", R),
             ("d_out", R), ("pout", R), ("ends", Array(2 * NW, BIT)), ("cok", BIT), ("R5", F), ("W5", F),
             ("kok", BIT), ("left", Array(S.RW + S.KW, BIT)), ("live", BIT), ("r", Array(S.RW, BIT)),
             ("k", Array(S.KW, BIT)), ("Pk", F), ("Pa", F), ("alpha", F)), Array(16, BIT))


CompileIds = _part(
    "CompileIds", IDS_STATICS, _sig_ids, _ids,
    "one arrival slot of the compile slice: 0 in ok where its id repeats a valid earlier slot's")
PoolCounts = _part(
    "PoolCounts", COUNTS_STATICS, _sig_counts, _counts,
    "one arrival slot of the pool window: its keyed block count's factor (α + (slot ‖ count)) where the count is nonzero, "
    "and the counts' sum")
CallsEq = _part(
    "CallsEq", REC_STATICS, _sig_calls_eq, _calls_eq,
    "one bus record of the compile slice: 1 in ok while each record so far is the registered calls row's")
CallsVerdict = _part(
    "CallsVerdict", STATICS, _sig_calls, _calls_forms,
    "the compile slice's verdict: the load (and the arrivals' ids distinct, and the padding zero), the steps' flags, the "
    "last rows held to cout, and calls, 1 iff the steps' bus records are the registered calls row (CallsEq's scan) and "
    "the padding of the calls and results rows is zero")
PoolOpen = _part(
    "PoolOpen", V3S.SLOT_STATICS, _sig_open, _open_forms,
    "one block of the opened pool row: where its key's print is nonzero and it is the key's first copy, the key's M "
    "cell's initial record (first the block, last its mlast) folded into the memory check, its zero record out")
PoolClose = _part(
    "PoolClose", V3S.SLOT_STATICS, _sig_close, _close_forms,
    "one block of the pool row a window leaves: its final links the row's, no holder, its final key print the print of "
    "the row's key, mlast 0 but at a key's first copy, its source the stream's (keyed: a final key; fresh: keyed and "
    "allocated in the window); a first copy's (print, block, mlast) folded into R5")
CompileCut = _part(
    "CompileCut", CUT_STATICS, _sig_cut, _cut_head,
    "the arrivals' O and cap the compile slice's load reads: a cut arrival's L − P and 0, so it doesn't leave at its "
    "a_O, another's a_O and a_cap")
CompileEdge = _part(
    "CompileEdge", EDGE_STATICS, _sig_edge, _edge,
    "the compile slice's edge after its last step: every running slot's free in vLLM's preemption order (the last slot "
    "first) as bus records, its returns the registered results', and 1 iff the open requests (running, waiting, or "
    "certified and not taken) are the cut arrivals, each having emitted its a_O; at AS = 1 none may be running")
PageCopy = _part(
    "PageCopy", COPY_STATICS, _sig_copy, _copy_forms,
    "one copy block of a carried page: bit 0 is 1 iff the block written to the window's carried port is the window's "
    "own write where its source is fresh, the previous carried port's where it is keyed and not fresh, and zero where "
    "the block holds no key")
PoolCopies = _part(
    "PoolCopies", V3S.SLOT_STATICS, _sig_copies, _copies_forms,
    "one entry of the M run: where its final first copy is nonzero, (key, first, last) folded into W5")
PoolVerdict = _part(
    "PoolVerdict", V3S.SLOT_STATICS, _sig_pool_verdict, _pool_verdict,
    "the pool window's verdict: pool, the stream's check with its bus product and call count the calls row's; open, the "
    "opened row's digest the input root; close, the left row's digest the output root, its queue ends the stream's, "
    "every block closed and R5 = W5; keys, the ragged runs well formed, all placed, and the slots' counts")
PARTS = (CompileIds, PoolCounts, CallsEq, CallsVerdict, PoolOpen, PoolClose, PoolCopies, PoolVerdict, CompileCut,
         CompileEdge, PageCopy)


def _statics(S, keys=STATICS) -> dict:
    return {k: getattr(S, k) for k in keys}


def _flat(w, refs):
    return _coll(Array(len(refs), BIT) if w is None else Array(len(refs) // w, Array(w, BIT)), refs)


def _col(t, elems):
    return _coll(Array(len(elems), t), [x for e in elems for x in e])


def _cut(row, fs) -> dict[str, list]:
    """Each field's refs of `row`, flat."""
    refs, out, at = _refs(row), {}, 0
    for name, n, w in fs:
        out[name] = refs[at:at + n * w]
        at += n * w
    return out


def _window_sig(S):
    bits = row_bits(S)
    return tuple((r, Array(bits[r], BIT)) for r in ("adv", "ing", "req", "srv", "res")), Array(_window_width(S), BIT)


def _window_body(B, S, adv, ing, req, srv, res):
    Kc = _Consts(B)
    sh = V5.shape(S)
    CW, CA = sh.CW, sh.CA
    Sc, F_, P, nps = CS.chain(S), fields(S), S.P, V5._nps(S)
    Wd, Ww, PWD = RC.width(Sc), RC.window_width(Sc), V5.pool_width(S)
    load_p = bind(V5.CompileLoad, **_statics(S, V5.LOAD_STATICS))
    step_p = bind(V5.CompileStep, **_statics(S, V5.STEP_STATICS))
    ids_p = bind(CompileIds, **_statics(S, IDS_STATICS))
    cut_p, edge_p = bind(CompileCut, **_statics(S, CUT_STATICS)), bind(CompileEdge, **_statics(S, EDGE_STATICS))
    g, q, rs = _cut(ing, F_["ing"]), _cut(req, F_["req"]), _cut(res, F_["res"])
    valid, ids = _flat(None, g["a_valid"]), _flat(None, g["a_id"])
    a_O, a_cut = _flat(None, q["a_O"]), _flat(None, q["a_cut"])
    head = B.call(cut_p, _flat(None, g["a_P"]), a_O, a_cut, *([_flat(None, q["a_cap"])] if S.AS else []))
    lo = _refs(B.call(load_p, adv, ing, head))
    load_ok, win = lo[:1], lo[1 + Wd:1 + Wd + Ww]
    carry = V5._step_carry(S)
    srb = CS.step_row_bits(S)
    xs = [_flat(srb, _refs(srv)[:P * srb])] + [_flat(w * CW, rs[k]) for k, w in (("backs", S.R), ("hits", CA),
                                                                                   ("evs", CA), ("rets", S.R))]
    stepped = _refs(B.scan(step_p, _coll(Array(carry, BIT), lo[1:1 + Wd] + Kc.bits(0, PWD) + Kc.bits(7, 3)), xs,
                           [_flat(None, win), valid, ids]))
    fin, ys = stepped[:carry], stepped[carry:]
    edge = _refs(B.call(edge_p, _flat(None, fin[:Wd]), _flat(None, fin[Wd:Wd + PWD]), _flat(None, rs["erets"]), valid,
                        ids, a_O, a_cut))
    yw, recs, at, et = nps * sum(V5._rec_widths(S)), [], 0, 1
    for w in V5._rec_widths(S):
        recs += [ys[u * yw + at + e] for u in range(P) for e in range(nps * w)] + edge[et:et + S.R * w]
        at, et = at + nps * w, et + S.R * w
    isz = V3S._size(_ids_layout(S))
    ido = _refs(B.scan(ids_p, _coll(Array(isz, BIT), Kc.bits(1, 1) + Kc.bits(0, S.K) + Kc.bits(1, S.K)),
                       [valid, _flat(S.IW, g["a_id"])], [valid, ids]))[:1]
    out = load_ok + ido + fin[Wd + PWD:] + edge[:1] + fin[:Wd] + win + fin[Wd:Wd + PWD] + recs
    return _coll(Array(len(out), BIT), out)


CompileWindow = CompositeDefinition(
    "CompileWindow", 2, STATICS, _window_sig, _window_body, conformance="exact-model-tested",
    doc="the compile slice's window from the registered results: CompileSlice_v5's load from the empty state and window "
        "with each cut arrival's O its length past its prompt (CompileCut), the arrivals' ids distinct, its steps, each "
        "ReplayStep_v1 with its pool calls in vLLM's order, each running slot's request number carried in its slot row, "
        "and the edge (CompileEdge); out the load's, ids', the steps' three and the edge's flags, the last state, window "
        "and slot rows, and the steps' and the edge's bus records, field-major")


def _compile_sig(S):
    bits = row_bits(S)
    return tuple((r, Array(bits[r], BIT)) for r in COMPILE_ROWS), Array(16, BIT)


def _records(refs, S) -> list[list]:
    """Each bus record's bits, on through y, from the records field-major (`fields`' ``calls``)."""
    f = RC._split(list(refs), fields(S)["calls"])
    return [[x for name, _n, _w in fields(S)["calls"] for x in f[name][i]] for i in range(nrec(S))]


def _compile_body(B, S, adv, ing, req, srv, cout, res, calls):
    Kc = _Consts(B)
    Sc, nb, W = CS.chain(S), data_bits(S), rec_width(S)
    nrows = RC.width(Sc) + RC.window_width(Sc) + V5.pool_width(S)
    out = _refs(B.call(bind(CompileWindow, **_statics(S)), adv, ing, req, srv, res))
    recs, regs = (_records(r, S) for r in (out[6 + nrows:], _refs(calls)[:nb["calls"]]))
    eq = _refs(B.scan(bind(CallsEq, **_statics(S, REC_STATICS)), _coll(Array(1, BIT), Kc.bits(1, 1)),
                      [_col(Array(W, BIT), recs), _col(Array(W, BIT), regs)], []))[:1]

    def pad(row, r):
        return _flat(None, _refs(row)[nb[r]:] or Kc.bits(0, 1))

    return B.call(bind(CallsVerdict, **_statics(S)), _flat(None, out[:6]), _flat(None, out[6:6 + nrows]), cout,
                  _flat(None, eq), pad(srv, "srv"), pad(req, "req"), pad(res, "res"), pad(calls, "calls"))


def _pool_sig(S):
    bits = row_bits(S)
    return tuple((r, Array(bits[r], BIT)) for r in POOL_ROWS), Array(16, BIT)


def _pool_body(B, S, req, calls, pin, pout, coin, prow, nrow, src, chk):
    Kc = _Consts(B)
    sh = V5.shape(S)
    S3, TT, CW, RW, KW, NW, G = sh.S3, sh.TT, sh.CW, sh.RW, sh.KW, sh.Z3.NW, V3S.G
    st3 = {k: getattr(S3, k) for k in V3S.SLOT_STATICS}
    slot_p, entry_p, end_p = (bind(d, **st3) for d in V3S.PARTS)
    open_p, close_p, copies_p, ver_p = (bind(d, **st3) for d in (PoolOpen, PoolClose, PoolCopies, PoolVerdict))
    bus_p, cnt_p = bind(V4.CompileBus, **_statics(S, V4.STATICS)), bind(PoolCounts, **_statics(S, COUNTS_STATICS))
    key_p = bind(V5.RaggedKey, **_statics(S, V5.KEY_STATICS))
    F_, n, nb = fields(S), S.NB - 1, data_bits(S)
    ck, q = RC._split(_refs(chk), F_["chk"]), RC._split(_refs(req), F_["req"])
    co = _refs(coin)
    alpha, gamma, delta = (_coll(F, co[128 * i:128 * (i + 1)]) for i in range(3))

    cl = RC._split(_refs(calls), F_["calls"])
    recs = [_col(t, cl[name]) for name, t in V4._rec_types(S)]
    bus = _refs(B.scan(bus_p, _coll(Array(TT + 128, BIT), Kc.bits(0, TT) + Kc.bits(1, 128)), recs, [alpha]))[:TT + 128]
    clay = _counts_layout(S)
    cnt = V3S._unpack(clay, _refs(B.scan(cnt_p, _coll(Array(V3S._size(clay), BIT), Kc.bits(0, RW) + Kc.bits(1, 128)
                                                       + Kc.bits(0, V5._total_width(S))),
                                         [_col(Array(KW, BIT), q["a_nb"])], [alpha]))[:V3S._size(clay)])

    pr, nr, sr = RC._split(_refs(prow), F_["prow"]), RC._split(_refs(nrow), F_["nrow"]), RC._split(_refs(src), F_["src"])
    digest = RG.digest(nb["prow"])
    d_in, d_out = (_refs(B.call(digest, x)) for x in (prow, nrow))
    kp = bind(BK.KeyPrint)
    p0, pf = ([_refs(B.call(kp, delta, _coll(BK.DIGEST, k))) for k in r["key"]] for r in (pr, nr))

    lay, elay = V3S._carry_layout(S3), V3S._entry_layout(S3)
    size, esize = V3S._size(lay), V3S._size(elay)
    rest = size - (1 + 3 * 128 + 2 * TT + 2 * NW)
    init = (Kc.bits(1, 1) + Kc.bits(1, 128) * 3 + Kc.bits(1, TT) + Kc.bits(0, TT) + pr["head"][0] + pr["tail"][0]
            + Kc.bits(0, rest))
    sx = [_col(t, ck[name]) for name, t in (("s_start", BIT), ("s_kind", Array(3, BIT)), ("s_r", Array(RW, BIT)),
                                             ("s_a", Array(CW, BIT)), ("s_b", Array(CW, BIT)), ("s_v", F),
                                             ("s_t", Array(TT, BIT)))]
    done = _refs(B.scan(slot_p, _coll(Array(size, BIT), init), sx, [alpha, gamma]))[:size]
    fin = V3S._unpack(lay, done)

    w = Array(NW, BIT)
    blocks = _col(w, [Kc.bits(b, NW) for b in range(1, S.NB)])
    op = _refs(B.scan(open_p, _coll(Array(256, BIT), fin.R + fin.W),
                      [blocks, _col(F, p0), _col(w, pr["cprev"]), _col(w, pr["mlast"])], [alpha, gamma]))[:256]

    def entries(R, W, ok, en, tag, a, v0, v, t):
        start = R + W + ok + Kc.bits(1, 1) + Kc.bits(0, 1 + G + 128)
        ex = [_col(BIT, en), _col(Array(G, BIT), tag), _col(F, a), _col(F, v0), _col(F, v), _col(Array(TT, BIT), t)]
        return V3S._unpack(elay, _refs(B.scan(entry_p, _coll(Array(esize, BIT), start), ex, [alpha, gamma]))[:esize])

    def tagged(en, tag):
        zero = Kc.bits(0, 1)[0]
        return [[e[0] if tag >> j & 1 else zero for j in range(G)] for e in en]

    def v0(tag, a):
        if tag == T3["B"]:
            return [x for f in LINKS for x in pr[f][a - 1]] + Kc.bits(0, 128 - 4 * NW)
        return p0[a - 1]

    dn = V5.dense(S)
    e1 = entries(op[:128], op[128:], fin.ok, [Kc.bits(1, 1)] * len(dn), [Kc.bits(tag, G) for tag, _a in dn],
                 [Kc.bits(a, 128) for _t, a in dn], [v0(tag, a) for tag, a in dn], ck["d_v"], ck["d_t"])
    e2 = entries(e1.R, e1.W, e1.ok, ck["m_en"], tagged(ck["m_en"], T3["M"]), ck["m_a"], [Kc.bits(0, 128)] * S.NM,
                 ck["m_v"], ck["m_t"])
    e3 = entries(e2.R, e2.W, e2.ok, ck["b_en"], tagged(ck["b_en"], T3["RB"]),
                 [a + Kc.bits(0, 128 - RW - KW) for a in ck["b_a"]], [Kc.bits(0, 128)] * S.NRB,
                 [v + Kc.bits(0, 128 - NW) for v in ck["b_v"]], ck["b_t"])

    klay, ksz = V5._key_layout(S), V3S._size(V5._key_layout(S))
    kinit = e3.R + e3.W + e3.ok + Kc.bits(1, 1) + Kc.bits(0, 512 + RW + KW + 1) + cnt.total + Kc.bits(1, 128)
    tok = [[x for e in q["blk"][S.B * i:S.B * (i + 1)] for x in e] for i in range(S.NRK)]
    keys = V3S._unpack(klay, _refs(B.scan(key_p, _coll(Array(ksz, BIT), kinit),
                                          [_col(Array(RW, BIT), ck["k_r"]), _col(Array(TT, BIT), ck["k_t"]),
                                           _col(Array(S.B * S.CTW, BIT), tok)], [delta, alpha, gamma]))[:ksz])
    ecarry = keys.R + keys.W + keys.eok + Kc.bits(0, 1 + 1 + G + 128)
    end = _refs(B.call(end_p, _coll(Array(size, BIT), done), _coll(Array(esize, BIT), ecarry)))[:_end_width(S3)]

    cz = _refs(B.scan(close_p, _coll(Array(129, BIT), Kc.bits(1, 128) + Kc.bits(1, 1)),
                      [blocks] + [_col(w, nr[f]) for f in LINKS + ("mlast",)]
                      + [_col(F, pf), _col(F, ck["d_v"][:n]), _col(F, ck["d_v"][n:2 * n]), _col(BIT, sr["keyed"]),
                         _col(BIT, sr["fresh"])], [alpha, gamma]))[:129]
    W5 = _refs(B.scan(copies_p, _coll(F, Kc.bits(1, 128)), [_col(BIT, ck["m_en"]), _col(F, ck["m_a"]), _col(F, ck["m_v"])],
                      [alpha, gamma]))[:128]
    R512 = Array(512, BIT)
    return B.call(ver_p, _flat(None, end), _flat(None, bus), _coll(R512, d_in), pin, _coll(R512, d_out), pout,
                  _flat(None, nr["head"][0] + nr["tail"][0]), _coll(BIT, cz[128:]), _coll(F, cz[:128]), _coll(F, W5),
                  _coll(BIT, keys.kok), _flat(None, keys.left), _coll(BIT, keys.live), _flat(None, keys.r),
                  _flat(None, keys.k), _coll(F, keys.P), _coll(F, cnt.P), alpha)


CompileSlice = CompositeDefinition(
    "CompileSlice", 6, STATICS, _compile_sig, _compile_body, conformance="exact-model-tested",
    doc="the compile slice of one window, its pool carried out to the pool window: CompileSlice_v5's load (its arrivals' "
        "ids distinct, a cut arrival's O its length past its prompt) and steps from the registered results, each running "
        "slot's request number carried in its slot row, the edge that preempts every running slot and holds the open "
        "requests to the cut ones, and its pool calls' bus records held to the registered calls row (verdict bits load, "
        "steps, cache, served, carry, calls, edge)")
PoolWindow = CompositeDefinition(
    "PoolWindow", 1, STATICS, _pool_sig, _pool_body, conformance="exact-model-tested",
    doc="one window of the block pool carried between windows, proved every window: the pool row under the input root "
        "opened as the memory check's initial values, CacheStep_v3's stream of NS slots run from it over the registered "
        "calls row's bus, the ragged row's blocks keyed once each as RK's entries, and the pool row the stream leaves, "
        "every block unheld, committed as the output root, with each block's source (verdict bits pool, open, close, "
        "keys)")


def _chain_sig(S):
    bits = row_bits(S)
    win = tuple(r for r in COMPILE_ROWS + POOL_ROWS if r not in ("pin", "pout"))
    win = tuple(dict.fromkeys(win))
    rows = tuple((f"{r}{w}", Array(bits[r], BIT)) for w in (1, 2) for r in win)
    return rows + tuple((f"root{i}", Array(512, BIT)) for i in range(3)), Array(64, BIT)


def _chain_body(B, S, *rows):
    win = tuple(dict.fromkeys(r for r in COMPILE_ROWS + POOL_ROWS if r not in ("pin", "pout")))
    at = {f"{r}{w}": rows[i * len(win) + j] for i, w in enumerate((1, 2)) for j, r in enumerate(win)}
    roots = rows[2 * len(win):]
    cu, pu = bind(CompileSlice, **_statics(S)), bind(PoolWindow, **_statics(S))
    out = []
    for w in (1, 2):
        get = {r: at[f"{r}{w}"] for r in win} | {"pin": roots[w - 1], "pout": roots[w]}
        out += _refs(B.call(cu, *(get[r] for r in COMPILE_ROWS))) + _refs(B.call(pu, *(get[r] for r in POOL_ROWS)))
    return _coll(Array(64, BIT), out)


PoolChain = CompositeDefinition(
    "PoolChain", 1, STATICS, _chain_sig, _chain_body, conformance="exact-model-tested",
    doc="two consecutive windows of the carried pool, both units of each: window 1 opens root0 and leaves root1, which "
        "window 2 opens and leaves root2, one input both windows read (verdicts: window 1's compile slice and pool "
        "window, then window 2's)")


def compile_unit(**st):
    return bind(CompileSlice, **st)


def pool_unit(**st):
    return bind(PoolWindow, **st)


def chain_unit(**st):
    return bind(PoolChain, **st)


def copy_unit(CB: int):
    return bind(PageCopy, CB=CB)


statics = V5.statics


def demo() -> tuple[dict, CS.Slice]:
    """`compile_slice_v5.demo`'s class and slice with NM = 16 M entries: a window from a carried pool lists the keys the
    pool caches (to NB − 1) beside the ones it touches."""
    st, sl = V5.demo()
    return {**st, "NM": 16}, sl


def demo_async() -> tuple[dict, CS.Slice]:
    """`compile_slice_v5.demo_async`'s class and slice with NM = 16, as `demo`."""
    st, sl = V5.demo_async()
    return {**st, "NM": 16}, sl


# --- the specification ---------------------------------------------------------------------------------------------------


PoolRow = PR.PoolRow


def empty_row(S) -> PoolRow:
    """The empty pool: every block free, in order, nothing cached (`cache_step_v3.empty_pool`'s)."""
    return PR.empty(S.NB)


def encode_row(S, row: PoolRow) -> list[int]:
    """The row's bits (`row_fields`)."""
    return PR.encode(row)


def decode_row(S, bits: Sequence[int]) -> PoolRow:
    """The row whose bits (`encode_row`) are ``bits``."""
    return PR.decode(S.NB, bits)


def root(S, row: PoolRow) -> bytes:
    """The row's root: its unsalted `Sha512RowSeg_v1` digest."""
    return PR.root(row)


def genesis(S) -> bytes:
    """The empty pool's root: a chain's first window's input root, and the one after a failed window."""
    return PR.genesis(S.NB)


def heads(S, row: PoolRow, delta: int) -> dict[int, int]:
    """M's cells the row opens: each key's print, where nonzero at a first copy b, to (first b, last mlast)."""
    NW = V5.shape(S).Z3.NW
    return {p: b | row.mlast[b - 1] << NW for b, p in ((b, BK.key_print(row.key[b - 1], delta)) for b in range(1, S.NB))
            if p and not row.cprev[b - 1]}


def opened(S, row: PoolRow, sl: CS.Slice, delta: int) -> V3S.Stream:
    """CacheStep_v3's stream from the pool `row`, its keys printed under δ and RK's the window's (`PoolWindow_v1`'s
    opening). A link past the blocks reads as a block of value 0, outside the touched list, which `reference` refuses."""
    S3 = V4.shape(S).S3
    rk, m = V4.rk_keys(S, sl, delta), heads(S, row, delta)
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
    S3, NW = V4.shape(S).S3, V5.shape(S).Z3.NW
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


def mrun(S, stream: V3S.Stream, carried: dict[int, int]) -> list[tuple[int, int, int]]:
    """The M run: (address, final value, final time) of every key the stream touches or the row opens, sorted."""
    cells = {a: (v, 0) for a, v in carried.items()}
    cells.update({a: stream.cells[(tag, a)] for tag, a in stream.first if tag == T3["M"]})
    return sorted((a, v, t) for a, (v, t) in cells.items())


def _offset(S, name: str) -> int:
    at = 0
    for f, n, w in fields(S)["chk"]:
        if f == name:
            return at
        at += n * w
    raise KeyError(name)


def with_mrun(S, chk: Sequence[int], ms: Sequence[tuple[int, int, int]]) -> list[int]:
    """The checked row `chk` with its M run the entries `ms` (address, final value, final time), unused entries zero."""
    if len(ms) > S.NM:
        raise ValueError(f"an M run of {len(ms)} entries passes NM = {S.NM}")
    TT, bits, used = V5.shape(S).TT, CS._bits, len(ms)
    ms = list(ms) + [(0, 0, 0)] * (S.NM - used)
    run = [1] * used + [0] * (S.NM - used)
    run += [b for a, _v, _t in ms for b in bits(a, 128)] + [b for _a, v, _t in ms for b in bits(v, 128)]
    run += [b for _a, _v, t in ms for b in bits(t, TT)]
    at = _offset(S, "m_en")
    return list(chk[:at]) + run + list(chk[at + len(run):])


def cuts(S, sl: CS.Slice, cut: Sequence[int] | None = None) -> tuple[int, ...]:
    """The K arrival slots' cut bits: `cut`'s for `sl`'s arrivals (default none cut), 0 past them."""
    cut = tuple(int(c) for c in (cut or ()))
    if len(cut) > S.K or any(c not in (0, 1) for c in cut) or any(cut[len(sl.arrivals):]):
        raise ValueError(f"cut bits {cut} for {len(sl.arrivals)} arrivals")
    return cut + (0,) * (S.K - len(cut))


def effective(S, sl: CS.Slice, cut: Sequence[int] | None = None) -> CS.Slice:
    """The slice the steps replay (`CompileCut`): `sl`, each arrival's O its served a_O, with each cut arrival's O its
    length past its prompt, L − P, and its cap 0."""
    cut = cuts(S, sl, cut)
    arr = tuple(((a[0], a[1], S.L - a[1]) + ((0,) if S.AS else ())) if c else tuple(a)
                for a, c in zip(sl.arrivals, cut))
    return replace(sl, arrivals=arr)


def opened_at_edge(S, sc: SimpleNamespace) -> list[tuple[int, int]]:
    """The requests open after `compile_slice_v4.scan`'s run `sc`'s last step, each (id, outputs emitted), in the order
    the edge leaves them: the running in slot order, the waiting, then the certified arrivals not taken. A next window
    takes their continuations first, in this order."""
    last = sc.states[-1]
    return ([(x[0], x[3]) for x in last.running] + [(x[0], x[3]) for x in last.waiting]
            + [(a[0], 0) for a in sc.window.queue[last.qh:]])


def cut_open(S, sl: CS.Slice, row: PoolRow | None = None, delta: int = COINS[2]
             ) -> tuple[CS.Slice, tuple[int, ...]] | None:
    """`sl` served to its edge from the pool `row` (default the empty pool): each request open after its last step cut,
    its a_O the outputs it emitted by then, the others as given, and the cut bits (`cuts`'). None where the scan refuses
    the window or cutting changes which requests are open."""
    row, cut = empty_row(S) if row is None else row, ()
    for _ in range(2):
        run = effective(S, sl, cut)
        sc = V4.scan(S, run, delta, opened(S, row, run, delta))
        if sc.refused is not None:
            return None
        out = dict(opened_at_edge(S, sc))
        got = tuple(int(a[0] in out) for a in sl.arrivals)
        if got == cut:
            return sl, cuts(S, sl, cut)
        sl, cut = replace(sl, arrivals=tuple((a[0], a[1], out[a[0]], *a[3:]) if a[0] in out else tuple(a)
                                             for a in sl.arrivals)), got
    return None


def edge(S, sc: SimpleNamespace, sl: CS.Slice, cut: Sequence[int]) -> SimpleNamespace:
    """`CompileEdge` on `compile_slice_v4.scan`'s run `sc` of `effective`'s slice of `sl`: at AS = 0 each running slot's
    free from the last slot down, on the run's stream; ``rets``, each record's returns (record t frees slot R − 1 − t),
    and ``ok``, whether the open requests are the cut ones, each having emitted its served a_O, and at AS = 1 none
    running."""
    last, R, rnum = sc.states[-1], S.R, V4.request_numbers(sl)
    rets = [0] * R
    if not S.AS:
        for j, x in reversed(list(enumerate(last.running))):
            held = RS.cdiv(x[4], S.B)
            if held:
                rets[R - 1 - j] = sc.stream.call(V3S.Call("free", rnum[x[0]], held))[0]
    served = {a[0]: a[2] for a in sl.arrivals}
    cut_ids = {a[0] for a, c in zip(sl.arrivals, cut) if c}
    opened = opened_at_edge(S, sc)
    ok = (not (S.AS and last.running) and all(i in cut_ids and o == served[i] for i, o in opened)
          and sorted(i for i, _o in opened) == sorted(cut_ids))
    return SimpleNamespace(rets=tuple(rets), ok=ok)


def sources(S, stream: V3S.Stream) -> tuple[tuple[int, int], ...]:
    """Each block's source at the window's end (``src``): (keyed, fresh), keyed where it holds a key, fresh where it does
    and the window allocated it, so the window wrote its page."""
    S3, out = V4.shape(S).S3, []
    for b in range(1, S.NB):
        keyed = int(stream.value(T3["K"], b) != 0)
        out.append((keyed, int(keyed and V3S.bsplit(S3, stream.value(T3["B"], b))["owner"] != 0)))
    return tuple(out)


def reference(S, sl: CS.Slice, row: PoolRow | None = None, delta: int = COINS[2],
              counts: Sequence[int] | None = None, cut: Sequence[int] | None = None) -> SimpleNamespace | None:
    """Both units' registered rows and the pool window's checked witness for `sl` (each arrival's O its served a_O) with
    the arrivals `cut` cut (default none) from the pool `row` (default the empty pool) under the print coin δ
    (unpadded): ``srv``, ``cout``, ``res`` (the steps' results and the edge's returns), ``chk`` and ``counts``
    (CompileSlice_v5's on `effective`'s slice, the edge's frees on its stream), the blocks' sources ``src``, ``edge``
    (whether the edge holds), the row ``left`` the stream leaves and ``next``, it where the edge holds and no block is
    held after it (else None). None where CompileSlice_v5's reference refuses the window on `opened`'s stream, or the M
    run passes NM entries."""
    row, cut = empty_row(S) if row is None else row, cuts(S, sl, cut)
    run, carried, got = effective(S, sl, cut), heads(S, row, delta), {}
    r5 = V5.reference(S, run, delta, counts, opened(S, row, run, delta),
                      edge=lambda sc: got.update(edge=edge(S, sc, sl, cut)))
    if r5 is None:
        return None
    stm = r5.scan.stream
    ms = mrun(S, stm, carried)
    if len(ms) > S.NM:
        return None
    n5, CW, e = RC._bits_of(fields(S)["res"][:_NRES]), V5.shape(S).CW, got["edge"]
    chk = with_mrun(S, r5.chk[n5:], ms)
    left, quiet = closed(S, stm, row, sl, delta)
    return SimpleNamespace(srv=r5.srv, cout=r5.cout, res=r5.chk[:n5] + [b for v in e.rets for b in CS._bits(v, CW)],
                           chk=chk, counts=r5.counts, src=sources(S, stm), edge=e.ok, left=left,
                           next=left if quiet and e.ok else None, mrun=ms, scan=r5.scan)


def encode_sources(S, src: Sequence[tuple[int, int]]) -> list[int]:
    """``src``'s bits (`fields`): the blocks' keyed bits, then their fresh bits."""
    return [k for k, _f in src] + [f for _k, f in src]


def copied(src: Sequence[tuple[int, int]], wrote: Sequence[Any], carried: Sequence[Any], empty: Any = 0) -> list[Any]:
    """The window's carried port (`PageCopy`): each block's page, the window's own write `wrote` where its source is
    fresh, the previous carried port's `carried` where it is keyed and not, `empty` where it holds no key. After a
    pool reset the previous carried port is empty, as the pool is."""
    return [(w if f else c) if k else empty for (k, f), w, c in zip(src, wrote, carried, strict=True)]


def _padded(S, r: str, bits: Sequence[int]) -> list[int]:
    n = data_bits(S)[r]
    if len(bits) != n:
        raise ValueError(f"row {r} holds {n} bits, not {len(bits)}")
    return list(bits) + [0] * (row_bits(S)[r] - n)


def records(st: dict, args: Sequence[dict[str, list[int]]]) -> list[list[int]]:
    """The steps' bus records (``calls``' bits, unpadded) on each of `args`' rows: `CompileWindow`'s."""
    S = SimpleNamespace(**st)
    Sc = CS.chain(S)
    skip = 6 + RC.width(Sc) + RC.window_width(Sc) + V5.pool_width(S)
    out = evaluate_bits(bind(CompileWindow, **st), *[[a[r] for a in args] for r in ("adv", "ing", "req", "srv", "res")])
    return [[int(b) for b in o[skip:]] for o in out.reshape(len(args), -1).tolist()]


def _req(S, v5: Sequence[int], sl: CS.Slice, cut: Sequence[int]) -> list[int]:
    """``req``'s bits: CompileSlice_v5's (of `effective`'s slice) with each arrival's served a_O and a_cap, and its cut
    bit."""
    got = RC._split(list(v5[:V5.data_bits(S)["req"]]), V5.fields(S)["req"])
    Sc = CS.chain(S)
    arr = [dict(zip(RC.arrival_names(Sc), a)) for a in sl.arrivals] + [None] * (S.K - len(sl.arrivals))
    got["a_O"] = [CS._bits(a["O"] if a else 0, RC.widths(Sc).LW) for a in arr]
    if S.AS:
        got["a_cap"] = [[a["cap"] if a else 0] for a in arr]
    got["a_cut"] = [[c] for c in cut]
    return _padded(S, "req", [b for name, _n, _w in fields(S)["req"] for e in got[name] for b in e])


def arguments(S, sl: CS.Slice, row: PoolRow | None = None, coins: Sequence[int] = COINS, cut: Sequence[int] | None = None,
              **given) -> dict[str, list[int]]:
    """Both units' inputs for window `sl` (each arrival's O its served a_O) with the arrivals `cut` cut (default none)
    from the pool `row` (default the empty pool) under the coins (α, γ, δ): every row of `COMPILE_ROWS` and
    `POOL_ROWS` (the registered ones zero-padded), each of `given` as given and the rest `reference`'s (``counts`` given
    lays out the ragged row), ``calls`` the records the reference's rows make (zero where it refuses the window, the
    pool rows ``row`` and its root both in and out)."""
    row, cut = empty_row(S) if row is None else row, cuts(S, sl, cut)
    counts = given.pop("counts", None)
    ref, nb = reference(S, sl, row, coins[2], counts, cut), data_bits(S)
    zero = {r: [0] * nb[r] for r in ("srv", "cout", "res", "src", "chk")}
    v5 = V5.arguments(S, effective(S, sl, cut), coins, srv=ref.srv if ref else zero["srv"],
                      cout=ref.cout if ref else zero["cout"], chk=[0] * V5.data_bits(S)["chk"],
                      counts=ref.counts if ref else (list(counts) if counts is not None else [0] * S.K))
    left = ref.left if ref else row
    out = {**{r: v5[r] for r in ("adv", "ing", "srv", "cout", "coin")}, "req": _req(S, v5["req"], sl, cut),
           "res": _padded(S, "res", ref.res if ref else zero["res"]), "chk": ref.chk if ref else zero["chk"],
           "pin": RG.digest_bits(root(S, row)), "prow": encode_row(S, row),
           "pout": RG.digest_bits(root(S, left)), "nrow": encode_row(S, left),
           "src": encode_sources(S, ref.src) if ref else zero["src"]}
    out = {**out, **{r: list(v) for r, v in given.items() if r != "calls"}}
    out["calls"] = list(given["calls"]) if "calls" in given else _padded(S, "calls", records(_statics(S), [out])[0])
    return out


def _run(unit, rows, args) -> list[int]:
    out = evaluate_bits(unit, *[[a[r] for a in args] for r in rows]).reshape(len(args), -1).tolist()
    return [sum(int(b) << j for j, b in enumerate(o)) for o in out]


def run_compile(st: dict, args: Sequence[dict[str, list[int]]]) -> list[int]:
    """The compile slice's verdict word on each of `args` (`arguments`' form), in one evaluation."""
    return _run(compile_unit(**st), COMPILE_ROWS, args)


def run_pool(st: dict, args: Sequence[dict[str, list[int]]]) -> list[int]:
    """The pool window's verdict word on each of `args` (`arguments`' form), in one evaluation."""
    return _run(pool_unit(**st), POOL_ROWS, args)


def run_copy(CB: int, args: Sequence[dict[str, list[int]]]) -> list[int]:
    """`PageCopy`'s verdict word on each of `args` (``keyed``, ``fresh``: one bit each; ``wrote``, ``carried``, ``out``: CB
    bits each), in one evaluation."""
    return _run(copy_unit(CB), ("keyed", "fresh", "wrote", "carried", "out"), args)


def chain_arguments(first: dict[str, list[int]], second: dict[str, list[int]]) -> dict[str, list[int]]:
    """`PoolChain`'s inputs for two windows' `arguments`, the second opening the root the first leaves."""
    if first["pout"] != second["pin"]:
        raise ValueError("the second window doesn't open the root the first leaves")
    win = tuple(dict.fromkeys(r for r in COMPILE_ROWS + POOL_ROWS if r not in ("pin", "pout")))
    return {**{f"{r}{w}": a[r] for w, a in ((1, first), (2, second)) for r in win},
            "root0": first["pin"], "root1": first["pout"], "root2": second["pout"]}


def run_chain(st: dict, args: Sequence[dict[str, list[int]]]) -> list[tuple[int, int, int, int]]:
    """`PoolChain`'s four verdict words on each of `args` (`chain_arguments`' form), in one evaluation."""
    params, _ret = chain_unit(**st).signature()
    out = evaluate_bits(chain_unit(**st), *[[a[name] for a in args] for name, _t in params])
    words = [[sum(int(b) << j for j, b in enumerate(o[16 * i:16 * (i + 1)])) for i in range(4)]
             for o in out.reshape(len(args), -1).tolist()]
    return [tuple(w) for w in words]


def size(**st) -> dict[str, int]:
    """Each unit's ANDs (an n-input AND n − 1), every input row's width (padded) and each unit's free width at statics
    `st`."""
    S, memo = SimpleNamespace(**st), {}
    bits = row_bits(S)
    return {"compile_ands": ands(compile_unit(**st), memo), "pool_ands": ands(pool_unit(**st), memo),
            **{f"{r}_bits": b for r, b in bits.items()},
            "compile_free_bits": free_width(S, {r: bits[r] for r in COMPILE_ROWS}),
            "pool_free_bits": free_width(S, {r: bits[r] for r in POOL_ROWS})}


def parts(**st) -> dict[str, list[tuple[object, int]]]:
    """Each unit's parts at statics `st`, each bound at the statics it reads, with the times the unit calls it: a unit's
    ANDs are the sum of each part's times its calls (the bodies' other nodes are constants), so this prices a class
    whose inputs are too wide to build the bodies at."""
    S = SimpleNamespace(**st)
    S3, n = V5.shape(S).S3, S.NB - 1

    def bound(d, keys):
        return bind(d, **{k: st[k] for k in keys})

    slot, entry, end = (bind(d, **{k: getattr(S3, k) for k in V3S.SLOT_STATICS}) for d in V3S.PARTS)
    op, cl, cp, ver = (bind(d, **{k: getattr(S3, k) for k in V3S.SLOT_STATICS})
                       for d in (PoolOpen, PoolClose, PoolCopies, PoolVerdict))
    return {"compile": [(bound(CompileCut, CUT_STATICS), 1), (bound(V5.CompileLoad, V5.LOAD_STATICS), 1),
                        (bound(V5.CompileStep, V5.STEP_STATICS), S.P), (bound(CompileEdge, EDGE_STATICS), 1),
                        (bound(CompileIds, IDS_STATICS), S.K), (bound(CallsEq, REC_STATICS), nrec(S)),
                        (bound(CallsVerdict, STATICS), 1)],
            "pool": [(bound(V4.CompileBus, V4.STATICS), nrec(S)), (bound(PoolCounts, COUNTS_STATICS), S.K),
                     (RG.digest(data_bits(S)["prow"]), 2), (bind(BK.KeyPrint), 2 * n), (slot, S.NS), (op, n),
                     (entry, len(V5.dense(S)) + S.NM + S.NRB), (bound(V5.RaggedKey, V5.KEY_STATICS), S.NRK), (end, 1),
                     (cl, n), (cp, S.NM), (ver, 1)]}
