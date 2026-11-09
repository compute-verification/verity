"""The compile slice that pays for its window's activity rather than its positions: `CompileSlice_v4`
(`compile_slice_v4`) with the arrivals' tokens registered as one ragged row of the blocks the window keys, and each
running slot's request number carried in the slot row rather than looked up among the arrivals at every call.

    CompileSlice_v5{R, W, A, M, T, L, B, H, CH, WM, FULL, PC, NB, IW, AS, EW, Q, D, P, K, CTW, CU, DA, NS, NM, NRB, NRK}
        (adv, ing, req, srv, cout, coin, chk) -> verdict[16]

**Why.** CompileSlice_v4 keys every arrival slot's CU blocks (K·CU BlockKey_v1 calls), lists RB's and RK's K·CU
addresses whole in its touched list, and finds each call's request number by comparing its id with all K arrivals'
((3R + 2)·K comparisons a step), so it pays for every position of its class whatever the window does
(`benchmarks/mempool/cost.py` prices both). Here the keys are one per block the window keys, RB's touched list is the
addresses its calls touch, and a request's number is set once, when it is admitted, and carried with its slot.

**The inputs**, each a field list (elements least significant bit first):
- ``adv`` (*advice*), ``ing`` (*ingress*) and ``srv`` (*registered*): CompileSlice_v4's.
- ``req`` (*registered*): each arrival slot's served output length ``a_O`` (and at AS = 1 ``a_cap``), its keyed block
  count ``a_nb`` (KW bits), and ``blk``, the ragged token row: NRK blocks of B tokens, slot r's first ``a_nb[r]`` blocks
  of its tokens (its prompt, then its outputs, zero past them) back to back in slot order, then zero blocks. The
  registration that binds ``req`` to the request Programs' tokens binds ``blk``'s layout by the counts beside it.
- ``cout`` (*registered*): the state row S_{P+1} and the window row W_1 the window leaves, and its slot row: each
  running slot's cached blocks ``nc`` (and at AS = 1 ``fl``) and its request number ``rq``. A request number is state
  the window carries from step to step and leaves, so it is registered with the rest of that state, never witness: a
  wrong ``rq`` in ``cout`` is refused by ``carry``.
- ``coin`` (the verifier's coins): α, γ and δ, CompileSlice_v4's.
- ``chk`` (*checked*): each step's bus results and the stream's slots, as CompileSlice_v4's; its touched list's runs
  (B's and K's blocks whole, and the M and RB addresses the stream touches, each run sorted); and each ragged block's
  request number ``k_r`` and RK time ``k_t``.
A registered row is padded to whole words (`compile_slice.WORD`); its padding must be zero. `CLASSES` names each input's
class. The checked witness is a computed check, as CompileSlice_v4's is, and counts as free input at its width until a
Lean lemma shows the witnesses the unit accepts are unique (AGENTS.md's nondeterminism rule), so `free_width` counts the
advice and the checked witness. Against CompileSlice_v4's, the checked witness loses RB's and RK's K·CU dense entries
(128 + TW bits each) and gains the RB run's (1 + RW + KW + NW + TW bits an entry) and the ragged blocks' (RW + TW).

**The output**: ``verdict``, 16 bits: CompileSlice_v4's ``load`` (here with the arrivals' ids distinct), ``steps``,
``cache``, ``served``, ``carry`` and ``pool``, and bit 6 ``keys``: the ragged row's runs are well formed and are the
counts' (below), every block placed in the NRK the class has. `ACCEPT` = 0x7F.

**The ragged row.** Block i of ``blk`` is on iff i < Σ a_nb. An on block's request number r_i is ``k_r[i]``; it starts
a run iff it is block 0 or r_i differs from the block before it, and the numbers never decrease, so the runs' numbers
strictly increase. Within a run the block index k counts 0, 1, … and stays below CU. A run's blocks chain: block k's key
is BlockKey_v1 over its parent, `block_key.ROOT` at k = 0 and the block before it else, and its tokens (widened to TOK
bits), printed under δ. The arrivals' side multiplies (α + (r ‖ a_nb[r])) over the slots with a_nb[r] > 0, the runs'
side (α + (r ‖ length)) over the runs; equal products make the runs exactly the slots with blocks, in slot order, each
a_nb[r] long, so block i's tokens are slot r_i's k-th block as the registration laid them out. An off block's ``k_r``
and ``k_t`` are zero.

**The touched list** has four runs, each starting the list's order afresh (their tags differ): B's and K's blocks 1 … NB
− 1 (initial values the empty pool's), the M addresses the stream touches (initial 0), the RB addresses it touches
(initial 0), each of these two sorted by `MemEntry_v1`'s check, and the ragged run: block i is RK's entry (r_i, k_i),
its initial and final value its key's print (RK is read only) and its time ``k_t[i]``. The ragged run's addresses are
distinct because the runs' numbers strictly increase and k increases within a run, which is what the sorted order
gives the other runs. A key the stream reads past a slot's count has no entry, so the memory check refuses it; a
count past what the stream reads only costs. Soundness error a window: CompileSlice_v4's, and (K + NRK)/2^128 over α
for the counts' products.

**The request numbers.** Request r is the arrival slot its id occupies: the load refuses a window whose valid arrivals
repeat an id (every pair compared once, about K²·IW/2 ANDs). The window starts from the empty engine and its queue is
its arrivals, so the candidate the waiting loop takes from the queue's front at position j is request qh + j, qh the
step's queue head (shifted by one on a quiet step, whose first arrival the step moves into waiting slot 0). Arrivals
are admitted as a prefix of the candidates, so an arrival placed at slot p ≥ lo, lo the first arrival's slot, is
request qh + forced + p − lo. A waiting candidate (W of them a step) is looked up among the arrivals' ids, as
CompileSlice_v4 looks up every call's. The slot row carries each running slot's number through the step's compaction,
beside ``nc``. On every window it accepts, the numbers are CompileSlice_v4's.

**The parts.** `CompileLoad` is the load; `CompileStep`, scanned over the P steps, is one step, ReplayStep_v1 with its
pool calls (CompileSlice_v4's, with the carried numbers), carrying its state row, its slot row and the steps', cache
and served flags, and making its step's records; `compile_slice_v4.CompileBus` folds the records in; `CompileArrivals`,
scanned over the K arrival slots, checks the ids distinct and makes the counts' side; `cache_step_v3`'s `PoolSlot` is
scanned over the NS slots and `PoolEntry` over the dense, M and RB runs; `RaggedKey`, scanned over the NRK blocks, is a
block's key, its RK entry and the runs' side; `PoolEnd` and `CompileVerdict` end it. Each part binds only the statics it
reads, so a class's step is traced once, and `size` is the unit at any class. Each step costs the same, the first ones
too, since the scan doesn't fold the empty engine's constants into them.

**The window** starts from the empty engine and pool, as CompileSlice_v4's does.

**The reference** is `reference`: `compile_slice_v4.scan`'s run of the slice, with each slot's count the RK blocks the
stream reads (or the counts given), and the request numbers of the slots it leaves running.

**Statics**: CompileSlice_v4's, and NRB RB entries and NRK ragged blocks, each from the public ladder
(`cache_step_v3.size_class`). The stream's RW = bitlen(K), KW = bitlen(CU), REFW = bitlen(R + 1) and time width
bitlen(NS) follow (`stream_statics`).
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits.boolean import memcheck as MC
from verity.ml.circuits.boolean.forms import ONE, ZERO, and_reduce, const_bits, mux, or_reduce
from verity.ml.circuits.boolean.gf128 import mul_forms
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array, Tuple
from verity.ml.commitments.gates import block_key as BK
from verity.core.protocols.certifier import cache_step as CST, compile_slice as CS
from verity.ml.protocols.certifier import cache_step_v2 as C2, cache_step_v3 as V3S, compile_slice_v4 as V4
from verity.core.protocols.certifier import replay_chain as RC, replay_step as RS
from verity.core.protocols.certifier.head_check import BIT, _coll, _Consts, _part, _refs, ands

STATICS = V4.STATICS + ("NRB", "NRK")
#: the statics a step reads: ReplayStep_v1's, the window's shape and the slot row's widths
STEP_STATICS = RS.STATICS + ("Q", "D", "P", "K", "CU")
LOAD_STATICS = STEP_STATICS + ("DA",)
ARRIVAL_STATICS = ("K", "IW", "CU")
KEY_STATICS = ("B", "CTW", "K", "CU", "NS")
FLAGS = V4.FLAGS + ("keys",)
ACCEPT = (1 << len(FLAGS)) - 1
ROWS = V4.ROWS
CLASSES = V4.CLASSES
PADDED = V4.PADDED
T3 = V3S.TAGS
F = Array(128, BIT)
_ONE128 = [ONE] + [ZERO] * 127


def _bitlen(v: int) -> int:
    return max(int(v).bit_length(), 1)


def _total_width(S) -> int:
    """The width of Σ a_nb over the K slots."""
    return _bitlen(S.K) + _bitlen(S.CU)


def stream_statics(S) -> SimpleNamespace:
    """CacheStep_v3's statics for the class: request numbers below K, block indices to CU, holders to R, times to NS."""
    return SimpleNamespace(NS=S.NS, NE=len(dense(S)) + S.NM + S.NRB + S.NRK, NB=S.NB, RW=_bitlen(S.K), KW=_bitlen(S.CU),
                           REFW=_bitlen(S.R + 1), TW=_bitlen(S.NS))


def dense(S) -> list[tuple[int, int]]:
    """The touched list's first run, (tag, address) sorted: B's and K's blocks 1 … NB − 1."""
    return [(T3["B"], b) for b in range(1, S.NB)] + [(T3["K"], b) for b in range(1, S.NB)]


def shape(S) -> SimpleNamespace:
    if not 0 <= S.DA <= S.D:
        raise ValueError(f"statics {dict(S.__dict__)}: the advice's DA decisions are at most D")
    if not S.PC:
        raise ValueError("CompileSlice_v5 is prefix caching's pool (PC = 1)")
    if min(S.NM, S.NRB, S.NRK) < 1:
        raise ValueError(f"NM = {S.NM}, NRB = {S.NRB}, NRK = {S.NRK}: each run holds at least one entry")
    S3 = stream_statics(S)
    Z3 = V3S.widths(S3)
    return SimpleNamespace(Z=RC.widths(CS.chain(S)), ZR=RS.widths(CS.step_statics(S)), S3=S3, Z3=Z3, CW=Z3.CW,
                           TT=S3.TW, CA=S.W + S.A, TOK=V4.key_tok(S), RW=S3.RW, KW=S3.KW)


def _step_shape(S) -> SimpleNamespace:
    """What a step reads of the shape, from `STEP_STATICS` alone."""
    KW, RW = _bitlen(S.CU), _bitlen(S.K)
    return SimpleNamespace(Z=RC.widths(CS.chain(S)), ZR=RS.widths(CS.step_statics(S)), CW=KW + 1, RW=RW,
                           S3=SimpleNamespace(RW=RW), CA=S.W + S.A)


def pool_fields(S) -> list[tuple[str, int, int]]:
    return ([("nc", S.R, _bitlen(S.CU) + 1)] + ([("fl", S.R, 1)] if S.AS else [])
            + [("rq", S.R, _bitlen(S.K))])


def pool_width(S) -> int:
    return RC._bits_of(pool_fields(S))


def _adv_fields(S) -> list[tuple[str, int, int]]:
    return [("dec", S.DA, 1), ("steps", 1, RC.widths(CS.chain(S)).PW)]


def _ing_fields(S) -> list[tuple[str, int, int]]:
    LW = RC.widths(CS.chain(S)).LW
    return [("a_valid", S.K, 1), ("a_id", S.K, S.IW), ("a_P", S.K, LW)]


def _head_fields(S) -> list[tuple[str, int, int]]:
    """``req``'s fields the load reads."""
    return [("a_O", S.K, RC.widths(CS.chain(S)).LW)] + ([("a_cap", S.K, 1)] if S.AS else [])


def fields(S) -> dict[str, list[tuple[str, int, int]]]:
    sh = shape(S)
    Sc, CW, TT, CA, NS, NM, ND, RW, KW = CS.chain(S), sh.CW, sh.TT, sh.CA, S.NS, S.NM, len(dense(S)), sh.RW, sh.KW
    return {"adv": _adv_fields(S),
            "ing": _ing_fields(S),
            "req": _head_fields(S) + [("a_nb", S.K, KW), ("blk", S.NRK * S.B, S.CTW)],
            "srv": [("rows", S.P, CS.step_row_bits(S))],
            "cout": [("state", 1, RC.width(Sc)), ("window", 1, RC.window_width(Sc)), ("pool", 1, pool_width(S))],
            "coin": [("alpha", 1, 128), ("gamma", 1, 128), ("delta", 1, 128)],
            "chk": [("backs", S.P * S.R, CW), ("hits", S.P * CA, CW), ("evs", S.P * CA, CW), ("rets", S.P * S.R, CW),
                    ("s_start", NS, 1), ("s_kind", NS, 3), ("s_r", NS, RW), ("s_a", NS, CW), ("s_b", NS, CW),
                    ("s_v", NS, 128), ("s_t", NS, TT), ("d_v", ND, 128), ("d_t", ND, TT), ("m_en", NM, 1),
                    ("m_a", NM, 128), ("m_v", NM, 128), ("m_t", NM, TT), ("b_en", S.NRB, 1), ("b_a", S.NRB, RW + KW),
                    ("b_v", S.NRB, sh.Z3.NW), ("b_t", S.NRB, TT), ("k_r", S.NRK, RW), ("k_t", S.NRK, TT)]}


def data_bits(S) -> dict[str, int]:
    """Each input row's bits before its padding."""
    return {r: RC._bits_of(fs) for r, fs in fields(S).items()}


def row_bits(S) -> dict[str, int]:
    return {r: CS._words_of(b) if r in PADDED else b for r, b in data_bits(S).items()}


def free_width(S, widths: dict[str, int] | None = None) -> int:
    """The unit's free input bits: the advice and the checked witness (`CLASSES`; the module's docstring); `widths`, each
    input row's bits (default the signature's)."""
    widths = row_bits(S) if widths is None else widths
    if set(widths) != set(CLASSES):
        raise ValueError(f"inputs {sorted(widths)} aren't the classified rows {sorted(CLASSES)}")
    return sum(b for r, b in widths.items() if CLASSES[r] in ("advice", "checked"))


def nrec(S) -> int:
    """The bus records a window's steps make: `compile_slice_v4.nrec`'s."""
    return V4.nrec(S)


def _nps(S) -> int:
    """A step's records."""
    return 2 * S.R + 4 * (S.W + S.A) + S.R * (1 + S.AS)


# --- the circuit ------------------------------------------------------------------------------------------------------------


def _nz(C, x):
    return or_reduce(C, list(x))


def OR(C, a, b):
    return a ^ b ^ C.AND(a, b)


def _factor(C, on, alpha, e):
    """(α + e) where `on`, else 1."""
    return [o ^ C.AND(on, a ^ x ^ o) for a, x, o in zip(alpha, RS._pad(e, 128), _ONE128)]


class _Bus(V4._Bus):
    """`compile_slice_v4._Bus` with each slot's request number ``rq`` carried in the slot row: a running slot's is its
    row's, an arrival candidate's its queue position, a waiting candidate's looked up among the arrivals' ids (the
    module's docstring)."""

    def __init__(self, C, S, sh, prow, cert, w, act, qh):
        self.C, self.S, self.sh, self.ZR = C, S, sh, sh.ZR
        p = RC._split(prow, pool_fields(S))
        self.nc = [list(x) for x in p["nc"]]
        self.fl = [e[0] for e in p["fl"]] if S.AS else [ZERO] * S.R
        self.rq = [list(x) for x in p["rq"]]
        self.cert, self.w, self.act, self.qh = cert, w, act, qh
        self.bad: list = []
        self.zero: list = []
        self.recs: list = []
        self.pending = None
        self.base_ = None
        self.LB = S.B.bit_length() - 1
        self.NW = max(sh.ZR.LW, sh.ZR.TW) + 1

    def base(self, tap):
        """The queue position of the step's first arrival candidate: qh, past the one a quiet step moves into waiting."""
        if self.base_ is None:
            RW = self.sh.RW
            self.base_ = RS._add(self.C, RC._pad(self.qh, RW), [tap["forced"]], RW)
        return self.base_

    def running(self, tap):
        """The running loop's events at their positions, each slot's request number its row's."""
        C, S, ZR, LB, NW = self.C, self.S, self.ZR, self.LB, self.NW
        R, RCW = S.R, ZR.RC + 1
        run, kept, popped, avail, upto = tap["run"], tap["kept"], tap["popped"], tap["avail"], tap["upto"]
        backs = self.w["backs"]
        nums = [list(q) for q in self.rq]
        held = [RS._cdiv(C, run[j]["comp"], LB, ZR.HW + 1) for j in range(R)]
        news = [RS._sel(C, kept[x], tap["needs"][x]) for x in range(R)]
        tgts = []
        for x in range(R):
            r, n = run[x], tap["run_n"][x]
            reach = RS._add(C, r["comp"], n, NW)
            known = RS._monus(C, RS._add(C, r["P"], r["out"], NW), [C.AND(self.fl[x], kept[x])], NW)
            tgts.append(RS._sel(C, _nz(C, n), RS._min(C, reach, known, NW)[LB:]))
        for j in range(R):
            self.carried(C.AND(popped[j], _nz(C, held[j])), backs[j])
        before = [[RS._lt(C, avail[j], upto[i]) for i in range(R)] for j in range(R)]
        pos_c = [RS._add(C, const_bits(i, RCW), RS._popcount(C, [C.AND(popped[j], before[j][i]) for j in range(R)], RCW),
                         RCW) for i in range(R)]
        pos_f = [RS._add(C, RS._sub(C, tap["cnt0"], const_bits(j + 1, RCW), RCW),
                         RS._popcount(C, [C.AND(kept[k], before[j][k].inv()) for k in range(R)], RCW), RCW)
                 for j in range(R)]
        for t in range(R):
            comp = [C.AND(kept[x], CST._eq(C, pos_c[x], const_bits(t, RCW))) for x in range(R)]
            frees = [C.AND(popped[j], CST._eq(C, pos_f[j], const_bits(t, RCW))) for j in range(R)]
            sel = [a ^ b for a, b in zip(comp, frees)]
            ca, fa = or_reduce(C, comp), or_reduce(C, frees)
            q, h0 = C2._pick(C, sel, nums), C2._pick(C, sel, held)
            new, back = C2._pick(C, comp, news), C2._pick(C, frees, backs)
            self.rec(OR(C, C.AND(ca, _nz(C, new)), C.AND(fa, _nz(C, h0))), [ONE, fa.inv(), fa], q, h0,
                     RS._sel(C, ca, new), RS._sel(C, fa, back))
            self.cache(ca, q, C2._pick(C, comp, self.nc), C2._pick(C, comp, tgts))
        for x in range(R):
            self.nc[x] = RS._sel(C, popped[x].inv(), self.cache_value(kept[x], self.nc[x], tgts[x]))
            self.rq[x] = RS._sel(C, popped[x].inv(), self.rq[x])

    def settle(self, tap):
        """The last candidate looked up, once the waiting loop has decided it: its hit, and its take, alloc and cache
        where admitted; a waiting candidate's number placed in its slot where admitted (an arrival's are placed after
        the loop, `candidates`)."""
        if self.pending is None:
            return
        C, S, ZR, LB, NW, RW = self.C, self.S, self.ZR, self.LB, self.NW, self.sh.RW
        c, rec, ntok = self.pending
        self.pending = None
        x = tap["cands"][c]
        reach, admit = x["reach"], x["admit"]
        if c < S.W:
            found, q = self.rnum(rec["id"])
            self.bad.append(C.AND(reach, found.inv()))
        else:
            q = RS._add(C, self.base(tap), const_bits(c - S.W, RW), RW)
        h, ev = self.w["hits"][c], self.w["evs"][c]
        lim = RS._sub(C, ntok, const_bits(1, len(ntok)), len(ntok))[LB:]
        on = C.AND(reach, _nz(C, lim))
        self.rec(on, V3S.KINDS["hit"], q, lim, None, h, ev)
        self.carried(on, list(h) + list(ev))
        self.rec(C.AND(admit, _nz(C, h)), V3S.KINDS["take"], q, None, h)
        new = RS._sel(C, admit, x["new"])
        self.rec(C.AND(admit, _nz(C, new)), V3S.KINDS["alloc"], q, h, new)
        tg = RS._sel(C, C.AND(admit, _nz(C, x["n"])), RS._min(C, RS._add(C, x["hb"], x["n"], NW), x["ntok"], NW)[LB:])
        _on, val = self.cache(admit, q, h, tg)
        val = mux(C, admit, val, [ZERO] * self.sh.CW)
        sb = [CST._eq(C, x["at"], const_bits(s, ZR.RC + 1)) for s in range(S.R)]
        self.nc = [mux(C, C.AND(admit, sb[s]), val, self.nc[s]) for s in range(S.R)]
        if c < S.W:
            self.rq = [mux(C, C.AND(admit, sb[s]), RS._pad(q, RW), self.rq[s]) for s in range(S.R)]

    def candidates(self, tap):
        """The waiting loop's end: the arrivals it admitted, a prefix placed at slots lo, lo + 1, … below hi, numbered
        on from `base`."""
        self.settle(tap)
        C, S, RW, RCW = self.C, self.S, self.sh.RW, self.ZR.RC + 1
        if not S.A:
            return
        cands = tap["cands"]
        lo = cands[S.W]["at"]
        hi = RS._add(C, cands[-1]["at"], [cands[-1]["admit"]], RCW)
        off = RS._sub(C, self.base(tap), RC._pad(lo, RW), RW)
        for p in range(S.R):
            placed = C.AND(RS._lt(C, const_bits(p, RCW), lo).inv(), RS._lt(C, const_bits(p, RCW), hi))
            self.rq[p] = mux(C, placed, RS._add(C, off, const_bits(p, RW), RW), self.rq[p])

    def leave(self, p, tap):
        """Placed slot p's update: at AS = 1 a kept slot's late cache, then a leaving request's free; its returns."""
        C, S, ZR, LB = self.C, self.S, self.ZR, self.LB
        x, q = tap["update"][p], self.rq[p]
        if S.AS:
            late = C.AND(tap["kept"][p], self.fl[p])
            tg = RS._sel(C, late, RS._sub(C, x["comp"], [x["sampled"]], ZR.LW)[LB:])
            _on, self.nc[p] = self.cache(late, q, self.nc[p], tg)
        held = RS._sel(C, x["leave"], x["held"])
        ret = self.w["rets"][p]
        on = C.AND(x["leave"], _nz(C, held))
        self.rec(on, V3S.KINDS["free"], q, held, None, ret)
        self.carried(on, ret)
        self.nc[p] = RS._sel(C, x["leave"].inv(), self.nc[p])
        return list(ret)

    def finish(self, tap) -> tuple[object, object, list]:
        """(1 iff an active step's bounds hold, 1 iff every result no record carries is zero, the next slot row)."""
        C, S = self.C, self.S
        upd = tap["update"]
        keep = [u["keep"] for u in upd]
        rows = [RS._sel(C, keep[p], self.nc[p] + ([upd[p]["sampled"]] if S.AS else []) + list(self.rq[p]))
                for p in range(S.R)]
        moved = RS._compact(C, rows, keep, self.ZR.RC + 1)
        CW, RW = self.sh.CW, self.sh.RW
        nxt = ([b for q in range(S.R) for b in moved[q][:CW]] + ([moved[q][CW] for q in range(S.R)] if S.AS else [])
               + [b for q in range(S.R) for b in moved[q][CW + S.AS:CW + S.AS + RW]])
        return or_reduce(C, self.bad).inv(), or_reduce(C, self.zero).inv(), nxt


def _load(C, S, adv, ing, head):
    """The load from the empty state and window: its checks' AND, state row 1 and the window row."""
    Sc, Z = CS.chain(S), RC.widths(CS.chain(S))
    a, g, w = RC._split(adv, _adv_fields(S)), RC._split(ing, _ing_fields(S)), RC._split(head, _head_fields(S))
    arrivals = {**g, "a_O": w["a_O"], **({"a_cap": w["a_cap"]} if S.AS else {})}
    dec = [e[0] for e in a["dec"]] + [ZERO] * (S.D - S.DA)
    checks, cur, win = RC._load_rows(C, Sc, CS._consts(RC.encode(RC.empty(Sc), Sc)),
                                     CS._consts(RC.encode_window(RC.QueueRow(), Sc)),
                                     RC._flat(arrivals, RC.arrival_fields(Sc)), dec, const_bits(S.DA, Z.DW),
                                     a["steps"][0])
    return [[and_reduce(C, checks)], list(cur), list(win)]


def _sig_load(S):
    Sc = CS.chain(S)
    widths = [RC._bits_of(f(S)) for f in (_adv_fields, _ing_fields, _head_fields)]
    return (tuple((n, Array(w, BIT)) for n, w in zip(("adv", "ing", "head"), widths)),
            Tuple(BIT, Array(RC.width(Sc), BIT), Array(RC.window_width(Sc), BIT)))


def _step_carry(S) -> int:
    return RC.width(CS.chain(S)) + pool_width(S) + 3


def _rec_widths(S) -> list[int]:
    """A record's fields' widths: on, kind, r and a, b, x, y."""
    sh = _step_shape(S)
    return [1, 3, sh.RW] + [sh.CW] * 4


def _step(C, S, carry, srv_u, backs, hits, evs, rets, win, valid, ids):
    """One step (CompileSlice_v4's, its pool calls with the carried request numbers): the carry (state row, slot row, and
    the steps', cache and served flags so far) and the step's records, field by field."""
    sh = _step_shape(S)
    Sc, St = CS.chain(S), CS.step_statics(S)
    Z, Wd, PWD, CW, A, R, CA = sh.Z, RC.width(Sc), pool_width(S), sh.CW, S.A, S.R, sh.CA
    cur, prow, oks = list(carry[:Wd]), list(carry[Wd:Wd + PWD]), list(carry[Wd + PWD:])
    wf = RC._split(win, RC.window_fields(Sc))
    r = RC._split(cur, RC.row_fields(Sc))
    qh, dh, steps = r["qh"][0], r["dh"][0], r["steps"][0]
    act = or_reduce(C, steps)
    fr = {an: RC._front(C, wf[qn], qh, A) for (qn, _n, _w), (an, _a, _x) in zip(RC.queue_fields(Sc), RC.front_fields(Sc))}
    bits = [e[0] for e in RC._front(C, wf["dec"], dh, A)]
    elems = []
    for name, n, wd in RS._in_fields(St):
        if name in r:
            elems += r[name]
        elif name in fr:
            elems += fr[name]
        elif name == "bits":
            elems += [[b] for b in bits]
        else:
            elems += [[ZERO] * wd for _ in range(n)]
    cert = {"valid": list(valid), "id": [list(ids[S.IW * k:S.IW * (k + 1)]) for k in range(S.K)]}
    wu = {k: [list(x[CW * i:CW * (i + 1)]) for i in range(len(x) // CW)]
          for k, x in (("backs", backs), ("hits", hits), ("evs", evs), ("rets", rets))}
    tp: dict = {}
    dv = _Bus(C, S, sh, prow, cert, wu, act, qh)
    out = RS._split(RS._out_fields(St), RS._build(C, St, *elems, tap=tp, derive=dv))
    stepped = [b for name, _n, _w in RC.state_fields(Sc) for e in out[name] for b in e]
    taken, used = out["taken"][0], out["used"][0]
    fv = [e[0] for e in fr["arr_valid"]]
    took = [C.AND(RC._lt(C, const_bits(k, Z.AC), taken), fv[k].inv()).inv() for k in range(A)]
    m = max(Z.DW, Z.AC) + 1
    dh_new = RS._add(C, RC._pad(dh, m), RC._pad(used, m), m)
    step_ok = and_reduce(C, [out["ok"][0][0], RC._lt(C, const_bits(A, Z.AC + 1), taken).inv(), *took,
                             RC._lt(C, const_bits(A, Z.AC + 1), used).inv(), RC._lt(C, wf["left"][0], dh_new).inv()])
    stepped_row = RC._pad(stepped + RS._add(C, qh, RC._pad(taken, Z.QW), Z.QW) + dh_new[:Z.DW]
                          + RS._sub(C, steps, const_bits(1, Z.PW), Z.PW), Wd)
    srv = [C.AND(act, x) for name in CS.ROW_FIELDS for e in out[name] for x in e]
    c_ok, z_ok, c_next = dv.finish(tp)
    oks = [C.AND(oks[0], C.AND(act, step_ok.inv()).inv()),
           and_reduce(C, [oks[1], C.AND(act, c_ok.inv()).inv(), z_ok]),
           C.AND(oks[2], RC._eq(C, srv, list(srv_u)))]
    if len(dv.recs) != _nps(S):
        raise AssertionError(f"a step made {len(dv.recs)} records, not {_nps(S)}")
    cols = list(zip(*dv.recs))
    recs = list(cols[0]) + [b for col in cols[1:] for e in col for b in e]
    return [list(mux(C, act, stepped_row, cur)) + list(mux(C, act, c_next, prow)) + oks, recs]


def _sig_step(S):
    Sc, sh = CS.chain(S), _step_shape(S)
    CW, CA = sh.CW, sh.CA
    ins = (("carry", Array(_step_carry(S), BIT)), ("srv", Array(CS.step_row_bits(S), BIT)),
           ("backs", Array(S.R * CW, BIT)), ("hits", Array(CA * CW, BIT)), ("evs", Array(CA * CW, BIT)),
           ("rets", Array(S.R * CW, BIT)), ("win", Array(RC.window_width(Sc), BIT)), ("valid", Array(S.K, BIT)),
           ("ids", Array(S.K * S.IW, BIT)))
    return ins, Tuple(Array(_step_carry(S), BIT), Array(_nps(S) * sum(_rec_widths(S)), BIT))


def _arrival_layout(S) -> list[tuple[str, int]]:
    return [("ok", 1), ("mask", S.K), ("ptr", S.K), ("i", _bitlen(S.K)), ("P", 128), ("total", _total_width(S))]


def _arrival(C, S, carry, valid_i, id_i, n_i, valid, ids, alpha):
    """Arrival slot i: 0 in ok where it repeats a valid earlier slot's id; its count's factor (α + (i ‖ a_nb)) where a_nb
    > 0, and the counts' sum."""
    c = V3S._unpack(_arrival_layout(S), carry)
    IW, RW = S.IW, _bitlen(S.K)
    dup = C.AND(valid_i[0], or_reduce(C, [and_reduce(C, [c.mask[j], valid[j], CST._eq(C, ids[IW * j:IW * (j + 1)], id_i)])
                                          for j in range(S.K)]))
    P = mul_forms(C, c.P, _factor(C, _nz(C, n_i), alpha, list(c.i) + list(n_i)))
    out = ([C.AND(c.ok[0], dup.inv())] + [OR(C, m, p) for m, p in zip(c.mask, c.ptr)] + [ZERO] + list(c.ptr[:-1])
           + RS._add(C, c.i, [ONE], RW) + P + RS._add(C, c.total, n_i, _total_width(S)))
    return [out, [dup]]


def _sig_arrival(S):
    carry = Array(V3S._size(_arrival_layout(S)), BIT)
    return ((("carry", carry), ("valid_i", BIT), ("id_i", Array(S.IW, BIT)), ("n_i", Array(_bitlen(S.CU), BIT)),
             ("valid", Array(S.K, BIT)), ("ids", Array(S.K * S.IW, BIT)), ("alpha", F)), Tuple(carry, BIT))


def _key_layout(S) -> list[tuple[str, int]]:
    return [("R", 128), ("W", 128), ("eok", 1), ("kok", 1), ("key", 512), ("r", _bitlen(S.K)), ("k", _bitlen(S.CU)),
            ("live", 1), ("left", _total_width(S)), ("P", 128)]


_ROOT = [b for i in range(8) for b in const_bits(int.from_bytes(BK.ROOT[8 * i:8 * i + 8], "big"), 64)]


def _key(C, S, carry, r, t, tok, delta, alpha, gamma):
    """Ragged block i (the module's docstring): its run's bookkeeping, its key and print, its RK entry's records folded
    into the memory check's (R, W), and where the run before it ends, that run's factor (α + (r ‖ length))."""
    RW, KW, TT, TOK = _bitlen(S.K), _bitlen(S.CU), _bitlen(S.NS), V4.key_tok(S)
    c = V3S._unpack(_key_layout(S), carry)
    on, live = _nz(C, c.left), c.live[0]
    first = C.AND(on, OR(C, live.inv(), CST._eq(C, r, c.r).inv()))
    k1 = RS._add(C, c.k, [ONE], KW + 1)
    k = RS._sel(C, first.inv(), k1[:KW])
    bad = [C.AND(C.AND(on, live), RS._lt(C, r, c.r)),
           C.AND(C.AND(on, first.inv()), RS._lt(C, k1, const_bits(S.CU, KW + 1)).inv()),
           C.AND(on.inv(), _nz(C, r))]
    parent = mux(C, first, _ROOT, c.key)
    words = BK.key_forms(C, None, [parent[64 * i:64 * (i + 1)] for i in range(8)],
                         [list(tok[S.CTW * j:S.CTW * (j + 1)]) + [ZERO] * (TOK - S.CTW) for j in range(S.B)])
    pr = BK.print_forms(C, delta, words)
    tag, addr = RS._sel(C, on, const_bits(T3["RK"], V3S.G)), RS._pad(RS._sel(C, on, RS._pad(r, RW) + RS._pad(k, KW)), 128)
    v0 = RS._sel(C, on, pr)
    R2 = MC._fold(C, alpha, gamma, c.R, on, [*tag, *addr, *v0, *t])
    W2 = MC._fold(C, alpha, gamma, c.W, on, [*tag, *addr, *v0, *[ZERO] * TT])
    close = C.AND(live, OR(C, first, on.inv()))
    P = mul_forms(C, c.P, _factor(C, close, alpha, list(c.r) + RS._add(C, c.k, [ONE], KW)))
    out = (R2 + W2 + [C.AND(c.eok[0], C.AND(on.inv(), _nz(C, t)).inv()), C.AND(c.kok[0], or_reduce(C, bad).inv())]
           + [x for w in words for x in w] + list(mux(C, on, r, c.r)) + list(mux(C, on, k, c.k)) + [on]
           + RS._sub(C, c.left, [on], _total_width(S)) + P)
    return [out, [on]]


def _sig_key(S):
    carry = Array(V3S._size(_key_layout(S)), BIT)
    return ((("carry", carry), ("r", Array(_bitlen(S.K), BIT)), ("t", Array(_bitlen(S.NS), BIT)),
             ("tok", Array(S.B * S.CTW, BIT)), ("delta", F), ("alpha", F), ("gamma", F)), Tuple(carry, BIT))


def _verdict(C, S, load, ids, oks, cur, win, prow, cout, spad, rpad, pool, calls, Ps, seq, P, kok, left, live, r, k,
             Pk, Pa, alpha):
    RW, KW = _bitlen(S.K), _bitlen(S.CU)
    last = mul_forms(C, list(Pk), _factor(C, live[0], alpha, list(r) + RS._add(C, list(k), [ONE], KW)))
    flags = [and_reduce(C, [load[0], ids[0], RC._zero(C, list(rpad))]), oks[0], oks[1],
             C.AND(oks[2], RC._zero(C, list(spad))), RC._eq(C, list(cur) + list(win) + list(prow), list(cout)),
             and_reduce(C, [pool[0], RC._eq(C, Ps, P), RC._eq(C, calls, seq)]),
             and_reduce(C, [kok[0], RC._zero(C, list(left)), RC._eq(C, last, list(Pa))])]
    return flags + [ZERO] * (16 - len(FLAGS))


def _pad_width(S, r: str) -> int:
    """Row r's padding as the verdict reads it: at least one bit (a constant zero where the row fills its words)."""
    return max(row_bits(S)[r] - data_bits(S)[r], 1)


def _sig_verdict(S):
    Sc, bits = CS.chain(S), row_bits(S)
    TT = Array(_bitlen(S.NS), BIT)
    return ((("load", BIT), ("ids", BIT), ("oks", Array(3, BIT)), ("cur", Array(RC.width(Sc), BIT)),
             ("win", Array(RC.window_width(Sc), BIT)), ("prow", Array(pool_width(S), BIT)), ("cout", Array(bits["cout"], BIT)),
             ("spad", Array(_pad_width(S, "srv"), BIT)), ("rpad", Array(_pad_width(S, "req"), BIT)),
             ("pool", BIT), ("calls", TT), ("Ps", F), ("seq", TT), ("P", F), ("kok", BIT),
             ("left", Array(_total_width(S), BIT)), ("live", BIT), ("r", Array(_bitlen(S.K), BIT)),
             ("k", Array(_bitlen(S.CU), BIT)), ("Pk", F), ("Pa", F), ("alpha", F)), Array(16, BIT))


CompileLoad = _part(
    "CompileLoad", LOAD_STATICS, _sig_load, _load,
    "the compile slice's load from the empty state and window: its checks' AND, state row 1 and the window row")
CompileStep = _part(
    "CompileStep", STEP_STATICS, _sig_step, _step,
    "one step of the compile slice: ReplayStep_v1 with its pool calls in vLLM's order, each running slot's request "
    "number carried in the slot row; the state row, slot row and the steps', cache and served flags carried, and the "
    "step's bus records")
CompileArrivals = _part(
    "CompileArrivals", ARRIVAL_STATICS, _sig_arrival, _arrival,
    "one arrival slot of the compile slice: 0 in ok where its id repeats a valid earlier slot's, its keyed block "
    "count's factor (α + (slot ‖ count)) where the count is nonzero, and the counts' sum")
RaggedKey = _part(
    "RaggedKey", KEY_STATICS, _sig_key, _key,
    "one block of the compile slice's ragged token row: its run's number and index, its BlockKey_v1 key over its "
    "parent (ROOT at a run's first block) and its print under δ, its RK entry folded into the memory check, and the "
    "factor (α + (number ‖ length)) of the run that ends before it")
CompileVerdict = _part(
    "CompileVerdict", STATICS, _sig_verdict, _verdict,
    "the compile slice's verdict: the load (and the arrivals' ids distinct, and the padding zero), the steps' flags, the "
    "last rows held to cout, pool (the stream checked and its calls the steps'), and keys (the ragged runs well formed, "
    "all placed, and the counts')", version=2)
PARTS = (CompileLoad, CompileStep, V4.CompileBus, CompileArrivals, RaggedKey, CompileVerdict)


def _sig(S):
    bits = row_bits(S)
    return tuple((r, Array(bits[r], BIT)) for r in ROWS), Array(16, BIT)


def _body(B, S, adv, ing, req, srv, cout, coin, chk):
    Kc = _Consts(B)
    sh = shape(S)
    S3, TT, CW, RW, KW, NW, G = sh.S3, sh.TT, sh.CW, sh.RW, sh.KW, sh.Z3.NW, V3S.G
    Sc, F_, nb, P, n, nps = CS.chain(S), fields(S), data_bits(S), S.P, nrec(S), _nps(S)
    Wd, Ww, PWD, CA = RC.width(Sc), RC.window_width(Sc), pool_width(S), sh.CA

    def bound(d, keys):
        return bind(d, **{k: getattr(S, k) for k in keys})

    load_p, step_p, bus_p = bound(CompileLoad, LOAD_STATICS), bound(CompileStep, STEP_STATICS), bound(V4.CompileBus, V4.STATICS)
    arr_p, key_p, ver_p = bound(CompileArrivals, ARRIVAL_STATICS), bound(RaggedKey, KEY_STATICS), bound(CompileVerdict, STATICS)
    slot_p, entry_p, end_p = (bind(d, **{k: getattr(S3, k) for k in V3S.SLOT_STATICS}) for d in V3S.PARTS)

    def split(row, fs):
        refs, out, at = _refs(row), {}, 0
        for name, cnt, w in fs:
            out[name] = refs[at:at + cnt * w]
            at += cnt * w
        return out

    def flat(w, refs):
        return _coll(Array(len(refs), BIT) if w is None else Array(len(refs) // w, Array(w, BIT)), refs)

    g, q, ck = split(ing, F_["ing"]), split(req, F_["req"]), split(chk, F_["chk"])
    co = _refs(coin)
    alpha, gamma, delta = (_coll(F, co[128 * i:128 * (i + 1)]) for i in range(3))
    valid, ids = flat(None, g["a_valid"]), flat(None, g["a_id"])

    head = q["a_O"] + (q["a_cap"] if S.AS else [])
    lo = _refs(B.call(load_p, adv, ing, flat(None, head)))
    load_ok, win = lo[:1], lo[1 + Wd:1 + Wd + Ww]
    carry_t = Array(_step_carry(S), BIT)
    srb = CS.step_row_bits(S)
    xs = [flat(srb, _refs(srv)[:P * srb])] + [flat(w * CW, ck[k]) for k, w in (("backs", S.R), ("hits", CA), ("evs", CA),
                                                                             ("rets", S.R))]
    stepped = _refs(B.scan(step_p, _coll(carry_t, lo[1:1 + Wd] + Kc.bits(0, PWD) + Kc.bits(7, 3)), xs,
                           [flat(None, win), valid, ids]))
    fin, ys = stepped[:_step_carry(S)], stepped[_step_carry(S):]
    yw, recs, at = nps * sum(_rec_widths(S)), [], 0
    for (_name, t), w in zip(V4._rec_types(S), _rec_widths(S)):
        recs.append(_coll(Array(n, t), [ys[u * yw + at + e] for u in range(P) for e in range(nps * w)]))
        at += nps * w
    bus = _refs(B.scan(bus_p, _coll(Array(TT + 128, BIT), Kc.bits(0, TT) + Kc.bits(1, 128)), recs, [alpha]))[:TT + 128]

    alay = _arrival_layout(S)
    ainit = Kc.bits(1, 1) + Kc.bits(0, S.K) + Kc.bits(1, S.K) + Kc.bits(0, RW) + Kc.bits(1, 128) + Kc.bits(0, _total_width(S))
    arr = V3S._unpack(alay, _refs(B.scan(arr_p, _coll(Array(V3S._size(alay), BIT), ainit),
                                         [valid, flat(S.IW, g["a_id"]), flat(KW, q["a_nb"])],
                                         [valid, ids, alpha]))[:V3S._size(alay)])

    def col(t, elems):
        return _coll(Array(len(elems), t), [x for e in elems for x in e])

    def parts(name, w):
        return [ck[name][w * i:w * (i + 1)] for i in range(len(ck[name]) // w)]

    lay, elay = V3S._carry_layout(S3), V3S._entry_layout(S3)
    size, esize = V3S._size(lay), V3S._size(elay)
    rest = size - (1 + 3 * 128 + 2 * TT + 2 * NW)
    init = (Kc.bits(1, 1) + Kc.bits(1, 128) * 3 + Kc.bits(1, TT) + Kc.bits(0, TT) + Kc.bits(1, NW) + Kc.bits(S.NB - 1, NW)
            + Kc.bits(0, rest))
    sx = [col(t, parts(name, w)) for name, t, w in (("s_start", BIT, 1), ("s_kind", Array(3, BIT), 3),
                                                    ("s_r", Array(RW, BIT), RW), ("s_a", Array(CW, BIT), CW),
                                                    ("s_b", Array(CW, BIT), CW), ("s_v", F, 128), ("s_t", Array(TT, BIT), TT))]
    done = _refs(B.scan(slot_p, _coll(Array(size, BIT), init), sx, [alpha, gamma]))[:size]
    fin_s = V3S._unpack(lay, done)

    def entries(head, en, tag, a, v0, v, t):
        start = head.R + head.W + head.ok + Kc.bits(1, 1) + Kc.bits(0, 1 + G + 128)
        ex = [col(BIT, en), col(Array(G, BIT), tag), col(F, a), col(F, v0), col(F, v), col(Array(TT, BIT), t)]
        return V3S._unpack(elay, _refs(B.scan(entry_p, _coll(Array(esize, BIT), start), ex, [alpha, gamma]))[:esize])

    def tagged(en, tag):
        zero = Kc.bits(0, 1)[0]
        return [[e[0] if tag >> j & 1 else zero for j in range(G)] for e in en]

    dn, empty = dense(S), V3S.empty_pool(S3, {})
    e1 = entries(fin_s, [Kc.bits(1, 1)] * len(dn), [Kc.bits(tag, G) for tag, _a in dn], [Kc.bits(a, 128) for _t, a in dn],
                 [Kc.bits(empty(tag, a), 128) for tag, a in dn], parts("d_v", 128), parts("d_t", TT))
    m_en = parts("m_en", 1)
    e2 = entries(e1, m_en, tagged(m_en, T3["M"]), parts("m_a", 128), [Kc.bits(0, 128)] * S.NM, parts("m_v", 128),
                 parts("m_t", TT))
    b_en = parts("b_en", 1)
    e3 = entries(e2, b_en, tagged(b_en, T3["RB"]), [a + Kc.bits(0, 128 - RW - KW) for a in parts("b_a", RW + KW)],
                 [Kc.bits(0, 128)] * S.NRB, [v + Kc.bits(0, 128 - NW) for v in parts("b_v", NW)], parts("b_t", TT))

    klay = _key_layout(S)
    kinit = (e3.R + e3.W + e3.ok + Kc.bits(1, 1) + Kc.bits(0, 512 + RW + KW + 1) + arr.total + Kc.bits(1, 128))
    keys = V3S._unpack(klay, _refs(B.scan(key_p, _coll(Array(V3S._size(klay), BIT), kinit),
                                          [flat(RW, ck["k_r"]), flat(TT, ck["k_t"]), flat(S.B * S.CTW, q["blk"])],
                                          [delta, alpha, gamma]))[:V3S._size(klay)])
    ecarry = keys.R + keys.W + keys.eok + Kc.bits(0, 1 + 1 + G + 128)
    end = _refs(B.call(end_p, _coll(Array(size, BIT), done), _coll(Array(esize, BIT), ecarry)))
    calls, Ps = end[1 + 2 * NW:1 + 2 * NW + TT], end[1 + 2 * NW + TT:1 + 2 * NW + TT + 128]
    def pad(row, r):
        return flat(None, _refs(row)[nb[r]:] or Kc.bits(0, 1))

    def bit(refs):
        return _coll(BIT, refs)

    return B.call(ver_p, bit(load_ok), bit(arr.ok), flat(None, fin[Wd + PWD:]), flat(None, fin[:Wd]), flat(None, win),
                  flat(None, fin[Wd:Wd + PWD]), cout, pad(srv, "srv"), pad(req, "req"), bit(end[:1]), flat(None, calls),
                  _coll(F, Ps), flat(None, bus[:TT]), _coll(F, bus[TT:]), bit(keys.kok), flat(None, keys.left),
                  bit(keys.live), flat(None, keys.r), flat(None, keys.k), _coll(F, keys.P), _coll(F, arr.P), alpha)


_DOC = ("the compile slice over the memory-checked pool at its window's activity, one window whole from the empty engine "
        "and pool: from the advice (its DA decisions and step count), the arrivals' ingress fields, their registered output "
        "lengths, keyed block counts and ragged token row, the load, a scan of ReplayStep_v1 steps with their pool calls in "
        "vLLM's order as records on a bus to CacheStep_v3's stream of NS memory-checked slots, each running slot's request "
        "number carried in its slot row, which runs in the unit from the empty pool with RK's entries the ragged row's "
        "BlockKey_v1 chains printed under δ; each hit, evictable count and return checked witness the bus carries; the "
        "served rows and last rows held to the registered srv and cout (verdict bits load, steps, cache, served, carry, "
        "pool, keys)")
CompileSlice = CompositeDefinition("CompileSlice", 5, STATICS, _sig, _body, conformance="exact-model-tested", doc=_DOC)


def statics(R: int, W: int, A: int, engine: dict, IW: int, Q: int, D: int, P: int, K: int, CTW: int, CU: int, DA: int,
            NS: int, NM: int, NRB: int, NRK: int) -> dict:
    st = {**V4.statics(R, W, A, engine, IW, Q=Q, D=D, P=P, K=K, CTW=CTW, CU=CU, DA=DA, NS=NS, NM=NM), "NRB": NRB, "NRK": NRK}
    shape(SimpleNamespace(**st))
    return st


def unit(**st):
    return bind(CompileSlice, **st)


#: the demonstrations' coins, fixed so the tests are reproducible; a window's are the verifier's, drawn live
COINS = tuple(int.from_bytes(hashlib.sha512(b"compile slice v5 demo coin " + bytes([i])).digest()[:16], "little")
              for i in range(3))


def demo() -> tuple[dict, CS.Slice]:
    """`compile_slice_v4.demo`'s class and slice, with the ladder's NRB = 12 RB entries and NRK = 8 ragged blocks, which
    hold its window's 9 and 7 and every advice and random window of the tests (to 10 and 8)."""
    st, sl = V4.demo()
    return {**st, "NRB": 12, "NRK": 8}, sl


def demo_async() -> tuple[dict, CS.Slice]:
    """`compile_slice_v4.demo_async`'s class and slice at AS = 1, with NRB = 12 and NRK = 8, which hold its window's 8 and
    7 and every advice and random window of the tests (to 10 and 7)."""
    st, sl = V4.demo_async()
    return {**st, "NRB": 12, "NRK": 8}, sl


# --- the specification ---------------------------------------------------------------------------------------------------


def _distinct(sl: CS.Slice) -> bool:
    ids = [a[0] for a in sl.arrivals]
    return len(set(ids)) == len(ids)


def touched_counts(S, stream: V3S.Stream) -> list[int]:
    """Each arrival slot's count: one past the last RK block the stream reads of it, 0 where it reads none."""
    S3, out = stream_statics(S), [0] * S.K
    for tag, a, *_rest in stream.entries():
        if tag == T3["RK"]:
            r, k = a & ((1 << S3.RW) - 1), a >> S3.RW
            out[r] = max(out[r], k + 1)
    return out


def reference(S, sl: CS.Slice, delta: int = COINS[2], counts: Sequence[int] | None = None,
              stream: V3S.Stream | None = None) -> SimpleNamespace | None:
    """The registered rows, the counts and the checked witness the unit accepts `sl`'s advice with under the print coin δ,
    by `compile_slice_v4.scan`'s run on `stream` (default the empty pool's) (``srv``, ``cout`` and ``chk``'s bits,
    unpadded, and ``counts``: `touched_counts` unless given), or None where CompileSlice_v4's reference refuses the slice
    or the class doesn't hold it (the stream past NS slots, NM M or NRB RB entries, its RK reads past a slot's count, the
    counts past NRK blocks or a slot's past CU), or the arrivals repeat an id."""
    shape(S)
    if not _distinct(sl):
        return None
    sc = V4.scan(S, sl, delta, stream)
    if sc.refused is not None:
        return None
    for u, (cur, nxt) in enumerate(zip(sc.states, sc.states[1:])):
        if cur.steps:
            arr, _bits = sc.fronts[u]
            if nxt.qh - cur.qh > len(arr) or nxt.dh - cur.dh > S.A or nxt.dh > len(sc.window.decisions):
                return None
    stm = sc.stream
    got = touched_counts(S, stm)
    counts = got if counts is None else list(counts)
    by = {}
    for x in stm.entries():
        by.setdefault(x[0], []).append(x)
    where = set(dense(S))
    if (len(stm.slots) > S.NS or len(by.get(T3["M"], [])) > S.NM or len(by.get(T3["RB"], [])) > S.NRB
            or any((x[0], x[1]) not in where for t in (T3["B"], T3["K"]) for x in by.get(t, []))
            or len(counts) != S.K or any(c < g for c, g in zip(counts, got)) or sum(counts) > S.NRK
            or any(not 0 <= c <= S.CU for c in counts)):
        return None
    Sc = CS.chain(S)
    steps = list(sc.steps) + [None] * (S.P - len(sc.steps))
    srv = [b for rows in steps for b in CS.rows_bits(S, rows)]
    rnum = V4.request_numbers(sl)
    last = sc.states[-1]
    rq = tuple(rnum[x[0]] for x in last.running) + (0,) * (S.R - len(last.running))
    cout = list(RC.encode(last, Sc)) + list(RC.encode_window(sc.window, Sc)) + _encode_pool(S, *sc.pools[-1], rq)
    return SimpleNamespace(srv=srv, cout=cout, chk=checked(S, sc, counts), counts=counts, scan=sc)


def checked(S, sc, counts: Sequence[int]) -> list[int]:
    """The checked row (`fields`' ``chk`` layout) of `compile_slice_v4.scan`'s run `sc`, its ragged blocks laid out by
    `counts` (whether or not they hold the stream's RK reads)."""
    sh = shape(S)
    CW, TT, S3, NW = sh.CW, sh.TT, sh.S3, sh.Z3.NW
    bits = CS._bits
    wit = list(sc.wit) + [((0,) * S.R, (0,) * sh.CA, (0,) * sh.CA, (0,) * S.R)] * (S.P - len(sc.wit))
    out = []
    for i in range(4):
        out += [b for w in wit for v in w[i] for b in bits(v, CW)]
    stm = sc.stream
    slots = list(stm.slots) + [V3S.Slot(*[0] * 11, use=0)] * (S.NS - len(stm.slots))
    out += [x.start for x in slots]
    out += [b for x in slots for b in bits(x.kind, 3)] + [b for x in slots for b in bits(x.r, S3.RW)]
    out += [b for x in slots for b in bits(x.a, CW)] + [b for x in slots for b in bits(x.b, CW)]
    out += [b for x in slots for b in bits(x.v, 128)] + [b for x in slots for b in bits(x.t, TT)]
    cells = [stm.cells.get(ta, (stm.initial(*ta), 0)) for ta in dense(S)]
    out += [b for v, _t in cells for b in bits(v, 128)] + [b for _v, t in cells for b in bits(t, TT)]
    entries = stm.entries()
    ms = [x for x in entries if x[0] == T3["M"]]
    ms += [(0, 0, 0, 0, 0)] * (S.NM - len(ms))
    out += [int(x[0] == T3["M"]) for x in ms] + [b for x in ms for b in bits(x[1], 128)]
    out += [b for x in ms for b in bits(x[3], 128)] + [b for x in ms for b in bits(x[4], TT)]
    rbs = [x for x in entries if x[0] == T3["RB"]]
    rbs += [(0, 0, 0, 0, 0)] * (S.NRB - len(rbs))
    out += [int(x[0] == T3["RB"]) for x in rbs] + [b for x in rbs for b in bits(x[1], S3.RW + S3.KW)]
    out += [b for x in rbs for b in bits(x[3], NW)] + [b for x in rbs for b in bits(x[4], TT)]
    blocks = [(r, stm.cells.get((T3["RK"], V3S.rk(S3, r, k)), (0, 0))[1]) for r, c in enumerate(counts) for k in range(c)]
    blocks = (blocks + [(0, 0)] * S.NRK)[:S.NRK]
    out += [b for r, _t in blocks for b in bits(r, S3.RW)] + [b for _r, t in blocks for b in bits(t, TT)]
    return out


def _encode_pool(S, nc: Sequence[int], fl: Sequence[int], rq: Sequence[int]) -> list[int]:
    CW, RW = shape(S).CW, _bitlen(S.K)
    return ([b for v in nc for b in CS._bits(v, CW)] + ([int(x) for x in fl] if S.AS else [])
            + [b for v in rq for b in CS._bits(v, RW)])


def _padded(S, r: str, bits: Sequence[int]) -> list[int]:
    n = data_bits(S)[r]
    if len(bits) != n:
        raise ValueError(f"row {r} holds {n} bits, not {len(bits)}")
    return list(bits) + [0] * (row_bits(S)[r] - n)


def ragged(S, sl: CS.Slice, counts: Sequence[int]) -> list[tuple[int, ...]]:
    """``blk``'s NRK blocks: each slot's first counts[r] blocks of its tokens (zero past them) in slot order, then zero
    blocks (cut at NRK)."""
    out = []
    for r, c in enumerate(counts):
        toks = tuple(sl.tokens[r]) if r < len(sl.tokens) else ()
        for k in range(c):
            out.append((toks[k * S.B:(k + 1) * S.B] + (0,) * S.B)[:S.B])
    return (out + [(0,) * S.B] * S.NRK)[:S.NRK]


def arguments(S, sl: CS.Slice, coins: Sequence[int] = COINS, srv: Sequence[int] | None = None,
              cout: Sequence[int] | None = None, chk: Sequence[int] | None = None,
              counts: Sequence[int] | None = None) -> dict[str, list[int]]:
    """The unit's inputs for slice `sl` under the coins (α, γ, δ) (bits each, `fields`' layouts, the registered rows
    zero-padded): ``srv``, ``cout``, ``chk`` and the slots' keyed block counts as given, by default `reference`'s (zero
    where it refuses the slice)."""
    Sc = CS.chain(S)
    Z, KW = RC.widths(Sc), _bitlen(S.CU)
    if len(sl.decisions) != S.DA or len(sl.arrivals) > S.K:
        raise ValueError(f"the slice's {len(sl.decisions)} decisions aren't DA = {S.DA}, or its arrivals exceed K")
    names = RC.arrival_names(Sc)
    arr = [dict(zip(names, a)) for a in sl.arrivals] + [None] * (S.K - len(sl.arrivals))

    def col(name, wd):
        return [b for x in arr for b in CS._bits(x[name] if x else 0, wd)]

    if srv is None or cout is None or chk is None or counts is None:
        ref = reference(S, sl, coins[2], counts)
        nb = data_bits(S)
        counts = (ref.counts if ref else [0] * S.K) if counts is None else counts
        srv = (ref.srv if ref else [0] * nb["srv"]) if srv is None else srv
        cout = (ref.cout if ref else [0] * nb["cout"]) if cout is None else cout
        chk = (ref.chk if ref else [0] * nb["chk"]) if chk is None else chk
    req = (col("O", Z.LW) + (col("cap", 1) if S.AS else []) + [b for c in counts for b in CS._bits(c, KW)]
           + [b for blk in ragged(S, sl, counts) for v in blk for b in CS._bits(v, S.CTW)])
    return {"adv": [int(d) for d in sl.decisions] + CS._bits(sl.steps, Z.PW),
            "ing": [int(x is not None) for x in arr] + col("id", S.IW) + col("P", Z.LW),
            "req": _padded(S, "req", req), "srv": _padded(S, "srv", srv), "cout": _padded(S, "cout", cout),
            "coin": [b for c in coins for b in CS._bits(c, 128)], "chk": list(chk)}


def run(st: dict, args: Sequence[dict[str, list[int]]]) -> list[int]:
    """The unit's verdict word on each of `args` (`arguments`' form), in one evaluation."""
    out = evaluate_bits(unit(**st), *[[a[r] for a in args] for r in ROWS]).reshape(len(args), -1).tolist()
    return [sum(int(b) << j for j, b in enumerate(o)) for o in out]


def evaluate(st: dict, slices: Sequence[CS.Slice], coins: Sequence[int] = COINS) -> list[int]:
    """The unit's verdict word on each slice, its registered rows, counts and checked witness `reference`'s."""
    S = SimpleNamespace(**st)
    return run(st, [arguments(S, sl, coins) for sl in slices])


def size(**st) -> dict[str, int]:
    """The unit's ANDs (an n-input AND n − 1), its inputs' widths (padded) and its free width at statics `st`."""
    S = SimpleNamespace(**st)
    return {"ands": ands(unit(**st)), **{f"{r}_bits": b for r, b in row_bits(S).items()}, "free_bits": free_width(S)}


def parts(**st) -> list[tuple[object, int]]:
    """The unit's parts at statics `st`, each bound at the statics it reads, with the times the unit calls it: the unit's
    ANDs are the sum of each part's times its calls (the body's other nodes are constants), so this prices a class whose
    inputs are too wide to build the body at."""
    S = SimpleNamespace(**st)
    S3 = shape(S).S3

    def bound(d, keys):
        return bind(d, **{k: st[k] for k in keys})

    slot, entry, end = (bind(d, **{k: getattr(S3, k) for k in V3S.SLOT_STATICS}) for d in V3S.PARTS)
    return [(bound(CompileLoad, LOAD_STATICS), 1), (bound(CompileStep, STEP_STATICS), S.P),
            (bound(V4.CompileBus, V4.STATICS), nrec(S)), (bound(CompileArrivals, ARRIVAL_STATICS), S.K), (slot, S.NS),
            (entry, len(dense(S)) + S.NM + S.NRB), (bound(RaggedKey, KEY_STATICS), S.NRK), (end, 1),
            (bound(CompileVerdict, STATICS), 1)]
