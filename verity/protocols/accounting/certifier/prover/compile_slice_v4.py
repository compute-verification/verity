"""The compile slice over the memory-checked pool: one window from the empty engine and pool, its schedule computed from
the advice and held to the registered rows it reads, each pool call a record on a bus to `CacheStep_v3`'s stream
(`cache_step_v3`), which runs inside the unit and answers them.

    CompileSlice_v4{R, W, A, M, T, L, B, H, CH, WM, FULL, PC, NB, IW, AS, EW, Q, D, P, K, CTW, CU, DA, NS, NM}
        (adv, ing, req, srv, cout, coin, chk) -> verdict[16]

**Why.** A unit that derives each hit, evictable count and return by calls that read and write the whole pool table
pays for the pool's size at every call. Here a step's pool calls are records on a bus, and one stream of NS
memory-checked access slots (CacheStep_v3) runs them: a window costs its accesses, whatever the pool's size.

**The inputs**, each a field list (elements least significant bit first):
- ``adv`` (*advice*, unpadded, since a padding gate would be a free input too): the window's DA decision bits ``dec``
  and its executed step count ``steps`` (PW = bitlen(P) bits). DA, the window's advice length in decisions, is a
  static: the advice's length is registered and public, so each length is its own binding. ``steps`` is advice because
  how many engine steps a certifier window holds is the scheduler's timing, which no public input fixes.
- ``ing`` (*ingress*, unpadded): the window's K arrival slots' ``a_valid``, ``a_id`` and prompt length ``a_P``, which
  the layout maps to the ingress gates of each request's arrival.
- ``req`` (*registered*): each slot's served output length ``a_O`` (and at AS = 1 whether ``a_O`` is its max_tokens),
  the request Program's end, and ``tok``, each slot's first CU·B token ids (its prompt, then its outputs), the request
  Program's token at each position.
- ``srv`` (*registered*): the served rows, each step's R rows (valid, request, kind, start, n) in slot order, as
  `compile_slice.rows_bits` packs them, zero on an idle step: the row `served_rows` registers and its consumers read.
- ``cout`` (*registered*): the state row S_{P+1} and the window row W_1 the window leaves, and its slot row: each
  running slot's cached blocks ``nc`` (and at AS = 1 ``fl``: its last row sampled, so its output is still in flight),
  which a later step's cache call starts from.
- ``coin`` (the verifier's coins): α and γ, the memory check's and the bus's, and δ, the keys' print coin, each in
  GF(2^128), flipped live after the window's rows are committed.
- ``chk`` (*checked*): each step's bus results (each running slot's returns, each candidate's hit and evictable count,
  each placed slot's returns), and the stream's witness: its NS slots (start, kind, r, a, b, the read's v and t) and
  its touched list's final values and times.
A registered row is padded to whole words (`compile_slice.WORD`); its padding must be zero. `CLASSES` names each input's
class. The checked witness is a computed check: the valid witness is unique, its coins are flipped after it is committed,
and a wrong one passes only with the fingerprints' collision probability, which enters the soundness error. It still
counts as free input at its width until a Lean lemma shows the witnesses the unit accepts are unique (AGENTS.md's
nondeterminism rule), so `free_width` counts the advice and the checked witness.

**The output**: ``verdict``, 16 bits: bit 0 ``load`` (with ``req``'s padding), bit 1 ``steps``, bit 2 ``cache`` (the step
side's bounds on its calls), bit 3 ``served`` (``srv`` is the computed rows), bit 4 ``carry`` (``cout`` is the computed last
rows), and bit 5 ``pool``: the stream's verdict, its bus product equal to the steps' and its call count equal to theirs.
`ACCEPT` = 0x3F.

**The steps.** The load from the empty state and window (`replay_chain._load_rows`) computes S_1 and W_1. Each step u
computes its arrivals and decision bits as the window row's at S_u's heads (`replay_chain._front`) and runs
ReplayStep_v1 on them and S_u; the next state is the step's (heads past what it took and read), or S_u on an idle step.

**The calls.** Request r is the arrival slot its id first occupies in ``ing``. Each step makes its calls in vLLM's order,
each a record (seq, kind, r, a, b, x, y) that is on iff the call has an operation:
- the running loop's events at their positions (`positions`): a kept slot's compute, alloc (r, ⌈comp / B⌉,
  needs) then cache (r, nc, target − nc), or a victim's free (r, ⌈comp / B⌉), its x the slot's returns;
- each candidate the waiting loop reaches, in order: hit (r, (ntok − 1) // B), x and y its hit and evictable count,
  then if admitted take (r, 0, h), alloc (r, h, new) and cache (r, h, target − h);
- the update, per placed slot: at AS = 1 a kept slot's late cache, then a leaving request's free, its x the returns.
The step side multiplies (α + e) over the on records, e the record's bits as the stream packs them. A result the
step reads is checked witness that the record carries, and zero where no record carries it: a kept slot's returns are
never read as anything else (the kept slots are a prefix whose boundary reads only the victims' returns, which are
their frees' x), and a candidate the loop doesn't reach has no hit. A hit is a call even for an arrival whose decision
bit then admits nothing: the waiting loop reads it before the bit, and it writes nothing the pool's semantics read.

**The stream** runs from the empty pool (the queue's head 1, tail NB − 1), with its touched list in two sorted runs:
B, K, RB and RK listed whole (B's initial values the empty pool's, K's and RB's 0, RK's the keys' prints), then the M
addresses the stream touches (initial 0), unused entries last. RK's keys are each arrival slot's CU blocks of its
registered tokens chained by BlockKey_v1 from `block_key.ROOT` (tokens widened to TOK bits, B·TOK a whole number of
bytes), printed under δ. Soundness error a window: (N + N²)/2^128 over α and γ for the memory check (N its records),
calls/2^128 over α for the bus, and 3·n(n − 1)/2^129 over δ for n keys.

**The parts.** `CompileWindow` is the load and the steps with their calls (the flags but ``pool``, and `nrec` records);
`CompileBus` folds the records in, scanned over them; `block_key.BlockKey` is called K·CU times; `cache_step_v3`'s
`PoolSlot` is scanned over the NS slots, `PoolEntry` over each run of the touched list, then `PoolEnd`; `CompileVerdict`
ends it. Each GF(2^128) product's operands are a part's inputs, so no product reads another's uncommitted forms. At the
demo's class (`demo`, NS = 192) the unit is 4,117,521 ANDs, two thirds of them the stream's slots.

**The window** starts from the empty engine and pool, as CompileSlice_v1's slice does.

**The reference** is `reference`: `scan`'s run of the slice, the load (`replay_chain.load`) and ReplayStep_v1 a step
(`replay_step.reference`), each step's calls (`pool_step`) on `cache_step_v3.Stream`. A step's hits, evictable counts
and returns are both what ReplayStep_v1 reads and what its calls derive, so `scan` runs them to a fixed point: from
no hit and each running slot's returns were it freed (`_returns`), it takes the first call, in the calls' order, whose
result differs, and runs the step again. `benchmarks/network_e2e/encodings/compile_slice_pool.py` (`check_trace_v4`)
holds `pool_step`'s hits, evictable counts and returns to the replay's `_KV` over the encodings' traces.

**Statics**: ReplayStep_v1's (`replay_step.STATICS`) with PC = 1; the chain's Q, D, P and K (`replay_chain`); CTW, a
token id's width, and CU, the blocks of each arrival's tokens ``req`` holds; DA ≤ D, the advice's decisions; NS stream
slots and NM M entries, from the public ladder (`cache_step_v3.size_class`). The stream's RW = bitlen(K), KW =
bitlen(CU), REFW = bitlen(R + 1) and time width bitlen(NS) follow (`stream_statics`).
"""
from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace
from typing import Sequence

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import ONE, ZERO, and_reduce, const_bits, mux, or_reduce, sub
from verity.circuits.boolean.gf128 import mul_forms
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.evaluation.bits import evaluate_bits
from verity.circuits.types import Array, Tuple
from verity.commitments.gates import block_key as BK
from verity.protocols.accounting.certifier.verifier import cache_step as CST, compile_slice as CS
from verity.protocols.accounting.certifier.prover import cache_step_v3 as V3S
from verity.protocols.accounting.certifier.verifier import replay_chain as RC, replay_step as RS
from verity.protocols.accounting.certifier.verifier.head_check import BIT, _coll, _Consts, _part, _refs, ands

STATICS = RS.STATICS + ("Q", "D", "P", "K", "CTW", "CU", "DA", "NS", "NM")
FLAGS = ("load", "steps", "cache", "served", "carry", "pool")
ACCEPT = (1 << len(FLAGS)) - 1
ROWS = ("adv", "ing", "req", "srv", "cout", "coin", "chk")
#: each input row's class: the advice is free; ingress and registered gates aren't counted in the window's c, the coins
#: are the verifier's, and the checked witness is a computed check, free at its width until its lemma (the module's
#: docstring)
CLASSES = {"adv": "advice", "ing": "ingress", "req": "registered", "srv": "registered", "cout": "registered",
           "coin": "coin", "chk": "checked"}
#: the rows padded to whole words (`compile_slice.WORD`)
PADDED = tuple(r for r in ROWS if CLASSES[r] == "registered")
#: ReplayStep_v1's name, in `replay_step.Req` and in an arrival's row, for the bit that says a request's O is its max_tokens
O_IS_MAX_TOKENS = "cap"
T3 = V3S.TAGS
F = Array(128, BIT)


def _bitlen(v: int) -> int:
    return max(int(v).bit_length(), 1)


def stream_statics(S) -> SimpleNamespace:
    """CacheStep_v3's statics for the class: request numbers below K, block indices to CU, holders to R, times to NS."""
    return SimpleNamespace(NS=S.NS, NE=len(dense(S)) + S.NM, NB=S.NB, RW=_bitlen(S.K), KW=_bitlen(S.CU),
                           REFW=_bitlen(S.R + 1), TW=_bitlen(S.NS))


def dense(S) -> list[tuple[int, int]]:
    """The touched list's first run, (tag, address) sorted: B's and K's blocks 1 … NB − 1, then RB's and RK's addresses
    r | k << RW of every arrival slot r < K and block k < CU."""
    RW = _bitlen(S.K)
    rk = [r | k << RW for k in range(S.CU) for r in range(S.K)]
    return ([(T3["B"], b) for b in range(1, S.NB)] + [(T3["K"], b) for b in range(1, S.NB)]
            + [(T3["RB"], a) for a in rk] + [(T3["RK"], a) for a in rk])


def key_tok(S) -> int:
    """The key's token width: CTW, widened until a block's B tokens are whole bytes (`block_key.token_width`)."""
    return BK.token_width(S.CTW, S.B)


def shape(S) -> SimpleNamespace:
    if not 0 <= S.DA <= S.D:
        raise ValueError(f"statics {dict(S.__dict__)}: the advice's DA decisions are at most D")
    if not S.PC:
        raise ValueError("CompileSlice_v4 is prefix caching's pool (PC = 1)")
    if S.NM < 1:
        raise ValueError(f"NM = {S.NM}: the touched list holds at least one M entry")
    S3 = stream_statics(S)
    Z3 = V3S.widths(S3)
    return SimpleNamespace(Z=RC.widths(CS.chain(S)), ZR=RS.widths(CS.step_statics(S)), S3=S3, Z3=Z3, CW=Z3.CW,
                           TT=S3.TW, CA=S.W + S.A, TOK=key_tok(S))


def pool_fields(S) -> list[tuple[str, int, int]]:
    return [("nc", S.R, shape(S).CW)] + ([("fl", S.R, 1)] if S.AS else [])


def pool_width(S) -> int:
    return RC._bits_of(pool_fields(S))


def fields(S) -> dict[str, list[tuple[str, int, int]]]:
    sh = shape(S)
    Sc, Z, CW, TT, CA, NS, NM, ND = CS.chain(S), sh.Z, sh.CW, sh.TT, sh.CA, S.NS, S.NM, len(dense(S))
    return {"adv": [("dec", S.DA, 1), ("steps", 1, Z.PW)],
            "ing": [("a_valid", S.K, 1), ("a_id", S.K, S.IW), ("a_P", S.K, Z.LW)],
            "req": [("a_O", S.K, Z.LW)] + ([("a_cap", S.K, 1)] if S.AS else []) + [("tok", S.K * S.CU * S.B, S.CTW)],
            "srv": [("rows", S.P, CS.step_row_bits(S))],
            "cout": [("state", 1, RC.width(Sc)), ("window", 1, RC.window_width(Sc)), ("pool", 1, pool_width(S))],
            "coin": [("alpha", 1, 128), ("gamma", 1, 128), ("delta", 1, 128)],
            "chk": [("backs", S.P * S.R, CW), ("hits", S.P * CA, CW), ("evs", S.P * CA, CW), ("rets", S.P * S.R, CW),
                    ("s_start", NS, 1), ("s_kind", NS, 3), ("s_r", NS, sh.S3.RW), ("s_a", NS, CW), ("s_b", NS, CW),
                    ("s_v", NS, 128), ("s_t", NS, TT), ("d_v", ND, 128), ("d_t", ND, TT), ("m_en", NM, 1),
                    ("m_a", NM, 128), ("m_v", NM, 128), ("m_t", NM, TT)]}


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


# --- the circuit ------------------------------------------------------------------------------------------------------------


def _nz(C, x):
    return or_reduce(C, list(x))


def _and(C, *xs):
    return and_reduce(C, list(xs))


def _fits(C, v, w: int) -> tuple[list, object]:
    """v's low w bits, and 1 iff a bit past them is set."""
    return RS._pad(v, w), or_reduce(C, list(v[w:])) if len(v) > w else ZERO


def _pick(C, sel, rows) -> list:
    """The row a one-hot (or all-zero) `sel` selects (all-zero for none)."""
    out = [ZERO] * len(rows[0])
    for s, row in zip(sel, rows):
        out = [o ^ C.AND(s, x) for o, x in zip(out, row)]
    return out


def _encode(sel, w: int) -> list:
    """The index a one-hot `sel` selects, in w bits."""
    out = [ZERO] * w
    for e, s in enumerate(sel):
        out = [x ^ (s if e >> b & 1 else ZERO) for b, x in enumerate(out)]
    return out


class _Bus:
    """The step's pool calls as bus records, made where ReplayStep_v1 reads what they return (the module's docstring).
    ``bad`` collects what refuses an active step, ``zero`` what refuses any step: a result no record carries that isn't
    zero."""

    def __init__(self, C, S, sh, prow, cert, w, act):
        self.C, self.S, self.sh, self.ZR = C, S, sh, sh.ZR
        p = RC._split(prow, pool_fields(S))
        self.nc = [list(x) for x in p["nc"]]
        self.fl = [e[0] for e in p["fl"]] if S.AS else [ZERO] * S.R
        self.cert, self.w, self.act = cert, w, act
        self.bad: list = []
        self.zero: list = []
        self.recs: list = []
        self.pending = None
        self.LB = S.B.bit_length() - 1
        self.NW = max(sh.ZR.LW, sh.ZR.TW) + 1

    def fit(self, v, guard):
        """v in CW bits; a value past them refuses where `guard`."""
        bits, over = _fits(self.C, v, self.sh.CW)
        self.bad.append(self.C.AND(guard, over))
        return bits

    def rec(self, on, kind, r, a, b=None, x=None, y=None):
        C, z = self.C, [ZERO]
        on = C.AND(self.act, on)
        kind = const_bits(kind, 3) if isinstance(kind, int) else kind
        self.recs.append((on, kind, RS._pad(r, self.sh.S3.RW), *(self.fit(f if f is not None else z, on)
                                                                 for f in (a, b, x, y))))

    def carried(self, on, x):
        """A result `x` the record `on` carries: it must be zero where the record is off."""
        self.zero.append(self.C.AND(self.C.AND(self.act, on).inv(), _nz(self.C, x)))

    def rnum(self, rid):
        C, S = self.C, self.S
        m = [C.AND(self.cert["valid"][k], CST._eq(C, self.cert["id"][k], rid)) for k in range(S.K)]
        return or_reduce(C, m), _encode(CST._first(C, m), self.sh.S3.RW)

    def cache(self, guard, q, nc, tg):
        """cache (q, nc, tg − nc) where `guard` and tg > nc: (whether it is on, max(nc, tg))."""
        C, w = self.C, self.sh.CW + 1
        self.bad.append(C.AND(guard, RS._lt(C, const_bits(self.S.CU, len(tg)), tg)))
        d, under = sub(C, RS._pad(tg, w), RS._pad(nc, w), w)
        on = _and(C, guard, under.inv(), _nz(C, d))
        self.rec(on, V3S.KINDS["cache"], q, nc, d)
        return on, mux(C, on, self.fit(tg, on), nc)

    def returns(self, tap):
        return [list(x) for x in self.w["backs"]]

    def running(self, tap):
        """The running loop's events at their positions: a kept slot's alloc and cache, or a victim's free."""
        C, S, ZR, LB, NW = self.C, self.S, self.ZR, self.LB, self.NW
        R, RCW = S.R, ZR.RC + 1
        run, kept, popped, avail, upto = tap["run"], tap["kept"], tap["popped"], tap["avail"], tap["upto"]
        backs = self.w["backs"]
        nums = []
        for x in range(R):
            found, q = self.rnum(run[x]["id"])
            nums.append(q)
            self.bad.append(C.AND(run[x]["valid"], found.inv()))
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
            q, h0 = _pick(C, sel, nums), _pick(C, sel, held)
            new, back = _pick(C, comp, news), _pick(C, frees, backs)
            self.rec(OR(C, C.AND(ca, _nz(C, new)), C.AND(fa, _nz(C, h0))), [ONE, fa.inv(), fa], q, h0,
                     RS._sel(C, ca, new), RS._sel(C, fa, back))
            self.cache(ca, q, _pick(C, comp, self.nc), _pick(C, comp, tgts))
        for x in range(R):
            self.nc[x] = RS._sel(C, popped[x].inv(), self.cache_value(kept[x], self.nc[x], tgts[x]))

    def cache_value(self, guard, nc, tg):
        """max(nc, tg) where `guard`, else nc, without a record (the record is made at the event's position)."""
        C, w = self.C, self.sh.CW + 1
        d, under = sub(C, RS._pad(tg, w), RS._pad(nc, w), w)
        return mux(C, _and(C, guard, under.inv(), _nz(C, d)), RS._pad(tg, self.sh.CW), nc)

    def hit(self, c, rec, ntok, tap):
        C, HW = self.C, self.ZR.HW
        self.settle(tap)
        self.pending = (c, rec, ntok)
        h, o1 = _fits(C, self.w["hits"][c], HW)
        ev, o2 = _fits(C, self.w["evs"][c], HW)
        self.bad += [o1, o2]
        return h, ev

    def settle(self, tap):
        """The last candidate looked up, once the waiting loop has decided it: its hit, and its take, alloc and cache
        where admitted."""
        if self.pending is None:
            return
        C, S, ZR, LB, NW = self.C, self.S, self.ZR, self.LB, self.NW
        c, rec, ntok = self.pending
        self.pending = None
        x = tap["cands"][c]
        reach, admit = x["reach"], x["admit"]
        found, q = self.rnum(rec["id"])
        self.bad.append(C.AND(reach, found.inv()))
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

    def candidates(self, tap):
        self.settle(tap)

    def leave(self, p, tap):
        """Placed slot p's update: at AS = 1 a kept slot's late cache, then a leaving request's free; its returns."""
        C, S, ZR, LB = self.C, self.S, self.ZR, self.LB
        x, s = tap["update"][p], tap["slots"][p]
        found, q = self.rnum(s["id"])
        self.bad.append(C.AND(s["valid"][0], found.inv()))
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
        rows = [RS._sel(C, keep[p], self.nc[p] + ([upd[p]["sampled"]] if S.AS else [])) for p in range(S.R)]
        moved = RS._compact(C, rows, keep, self.ZR.RC + 1)
        CW = self.sh.CW
        nxt = [b for q in range(S.R) for b in moved[q][:CW]] + ([moved[q][CW] for q in range(S.R)] if S.AS else [])
        return or_reduce(C, self.bad).inv(), or_reduce(C, self.zero).inv(), nxt


def OR(C, a, b):
    return a ^ b ^ C.AND(a, b)


def nrec(S) -> int:
    """The bus records a window's steps make: a step's two at each of its R running positions, four for each of its
    W + A candidates (hit, take, alloc, cache) and at each placed slot a free, after a late cache at AS = 1."""
    return S.P * (2 * S.R + 4 * (S.W + S.A) + S.R * (1 + S.AS))


def _rec_types(S) -> list[tuple[str, object]]:
    sh = shape(S)
    return [("on", BIT), ("kind", Array(3, BIT)), ("r", Array(sh.S3.RW, BIT))] + [(k, Array(sh.CW, BIT)) for k in "abxy"]


def _steps_bits(S) -> int:
    """chk's first bits: the steps' witness (backs, hits, evs, rets)."""
    return RC._bits_of(fields(S)["chk"][:4])


def _window(C, S, adv, ing, req, srv_in, cout_in, chk):
    """The load and the steps with their pool calls: the flags load, steps, cache, served and carry, then the records."""
    sh = shape(S)
    Sc, St = CS.chain(S), CS.step_statics(S)
    Z = sh.Z
    F_, n_req = fields(S), data_bits(S)["req"]
    a, g, w = RC._split(adv, F_["adv"]), RC._split(ing, F_["ing"]), RC._split(req, F_["req"])
    ck = RC._split(chk, F_["chk"][:4])
    arrivals = {**g, "a_O": w["a_O"], **({"a_cap": w["a_cap"]} if S.AS else {})}
    dec = [e[0] for e in a["dec"]] + [ZERO] * (S.D - S.DA)
    checks, cur, win = RC._load_rows(C, Sc, CS._consts(RC.encode(RC.empty(Sc), Sc)),
                                     CS._consts(RC.encode_window(RC.QueueRow(), Sc)),
                                     RC._flat(arrivals, RC.arrival_fields(Sc)), dec, const_bits(S.DA, Z.DW),
                                     a["steps"][0])
    cert = {"valid": [e[0] for e in g["a_valid"]], "id": g["a_id"]}
    prow = [ZERO] * pool_width(S)
    wf = RC._split(win, RC.window_fields(Sc))
    Wd, A, R, CA = RC.width(Sc), S.A, S.R, sh.CA
    cache_ok, steps_ok, srv, recs = [], [], [], []
    for u in range(S.P):
        r = RC._split(cur, RC.row_fields(Sc))
        qh, dh, steps = r["qh"][0], r["dh"][0], r["steps"][0]
        act = or_reduce(C, steps)
        fr = {an: RC._front(C, wf[qn], qh, A) for (qn, _n, _w), (an, _a, _x) in zip(RC.queue_fields(Sc),
                                                                                   RC.front_fields(Sc))}
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
        tp: dict = {}
        wu = {k: ck[k][u * n:(u + 1) * n] for k, n in (("backs", R), ("hits", CA), ("evs", CA), ("rets", R))}
        dv = _Bus(C, S, sh, prow, cert, wu, act)
        out = RS._split(RS._out_fields(St), RS._build(C, St, *elems, tap=tp, derive=dv))
        stepped = [b for name, _n, _w in RC.state_fields(Sc) for e in out[name] for b in e]
        taken, used = out["taken"][0], out["used"][0]
        fv = [e[0] for e in fr["arr_valid"]]
        took = [C.AND(RC._lt(C, const_bits(k, Z.AC), taken), fv[k].inv()).inv() for k in range(A)]
        m = max(Z.DW, Z.AC) + 1
        dh_new = RS._add(C, RC._pad(dh, m), RC._pad(used, m), m)
        step_ok = and_reduce(C, [out["ok"][0][0], RC._lt(C, const_bits(A, Z.AC + 1), taken).inv(), *took,
                                 RC._lt(C, const_bits(A, Z.AC + 1), used).inv(), RC._lt(C, wf["left"][0], dh_new).inv()])
        steps_ok.append(C.AND(act, step_ok.inv()).inv())
        stepped_row = RC._pad(stepped + RS._add(C, qh, RC._pad(taken, Z.QW), Z.QW) + dh_new[:Z.DW]
                              + sub(C, steps, const_bits(1, Z.PW), Z.PW)[0], Wd)
        srv += [C.AND(act, x) for name in CS.ROW_FIELDS for e in out[name] for x in e]
        c_ok, z_ok, c_next = dv.finish(tp)
        cache_ok += [C.AND(act, c_ok.inv()).inv(), z_ok]
        recs += dv.recs
        cur, prow = mux(C, act, stepped_row, cur), mux(C, act, c_next, prow)
    if len(recs) != nrec(S):
        raise AssertionError(f"the steps made {len(recs)} records, not nrec = {nrec(S)}")
    flags = [and_reduce(C, checks + [RC._zero(C, req[n_req:])]), and_reduce(C, steps_ok), and_reduce(C, cache_ok),
             RC._eq(C, srv, srv_in), RC._eq(C, list(cur) + list(win) + list(prow), cout_in)]
    cols = list(zip(*recs))
    return [flags, list(cols[0])] + [list(e) for col in cols[1:] for e in col]


def _sig_window(S):
    bits, n = row_bits(S), nrec(S)
    ins = tuple((r, Array(bits[r], BIT)) for r in ROWS[:5]) + (("chk", Array(_steps_bits(S), BIT)),)
    return ins, Tuple(Array(len(FLAGS) - 1, BIT), *(Array(n, t) for _, t in _rec_types(S)))


def _bus_forms(C, S, carry, on, kind, r, a, b, x, y, alpha):
    """One bus record folded in: the count plus `on`, and the product times (α + e) where `on`, e (seq ‖ kind ‖ r ‖ a ‖ b ‖
    x ‖ y) as `cache_step_v3`'s slots pack their records."""
    TT = shape(S).TT
    seq = RS._add(C, list(carry[:TT]), list(on), TT)
    e = RS._pad(list(seq) + list(kind) + list(r) + list(a) + list(b) + list(x) + list(y), 128)
    f = [o ^ C.AND(on[0], al ^ ee ^ o) for al, ee, o in zip(alpha, e, const_bits(1, 128))]
    return [list(seq) + mul_forms(C, list(carry[TT:]), f), list(on)]


def _sig_bus(S):
    carry = Array(shape(S).TT + 128, BIT)
    return (("carry", carry),) + tuple(_rec_types(S)) + (("alpha", F),), Tuple(carry, BIT)


def _verdict(C, S, flags, ok, calls, Ps, seq, P):
    pool = and_reduce(C, [ok[0], RC._eq(C, Ps, P), RC._eq(C, calls, seq)])
    return list(flags) + [pool] + [ZERO] * (16 - len(FLAGS))


def _sig_verdict(S):
    TT = Array(shape(S).TT, BIT)
    return (("flags", Array(len(FLAGS) - 1, BIT)), ("ok", BIT), ("calls", TT), ("Ps", F), ("seq", TT), ("P", F)), Array(16, BIT)


CompileWindow = _part(
    "CompileWindow", STATICS, _sig_window, _window,
    "the compile slice's load and steps, each ReplayStep_v1 with its pool calls in vLLM's order: the flags load, steps, "
    "cache, served and carry, and the calls' bus records, nrec of them, off where a call returns at once")
CompileBus = _part(
    "CompileBus", STATICS, _sig_bus, _bus_forms,
    "one bus record of the compile slice's steps folded in: the call count, and the product of (α + e) over the records on")
CompileVerdict = _part(
    "CompileVerdict", STATICS, _sig_verdict, _verdict,
    "the compile slice's verdict: the steps' flags, and pool, 1 iff the stream checked and its calls are the steps' (the "
    "counts and the bus products equal)")
PARTS = (CompileWindow, CompileBus, CompileVerdict)


def _sig(S):
    bits = row_bits(S)
    return tuple((r, Array(bits[r], BIT)) for r in ROWS), Array(16, BIT)


def _body(B, S, adv, ing, req, srv, cout, coin, chk):
    Kc = _Consts(B)
    sh = shape(S)
    S3, TT, CW, RW, NW, G = sh.S3, sh.TT, sh.CW, sh.S3.RW, sh.Z3.NW, V3S.G
    st = {k: getattr(S, k) for k in STATICS}
    win_p, bus_p, ver_p = (bind(d, **st) for d in PARTS)
    slot_p, entry_p, end_p = (bind(d, **{k: getattr(S3, k) for k in V3S.SLOT_STATICS}) for d in V3S.PARTS)
    F_, n = fields(S), nrec(S)
    ck = RC._split(_refs(chk), F_["chk"])
    co = _refs(coin)
    alpha, gamma, delta = (_coll(F, co[128 * i:128 * (i + 1)]) for i in range(3))

    def col(t, elems):
        return _coll(Array(len(elems), t), [x for e in elems for x in e])

    nw = _steps_bits(S)
    out = _refs(B.call(win_p, adv, ing, req, srv, cout, _coll(Array(nw, BIT), _refs(chk)[:nw])))
    flags, at, recs = out[:len(FLAGS) - 1], len(FLAGS) - 1, []
    for _, t in _rec_types(S):
        wd = n * sum(BL.elements(t))
        recs.append(_coll(Array(n, t), out[at:at + wd]))
        at += wd
    bus = _refs(B.scan(bus_p, _coll(Array(TT + 128, BIT), Kc.bits(0, TT) + Kc.bits(1, 128)), recs, [alpha]))[:TT + 128]

    bk = bind(BK.BlockKey, B=S.B, TOK=sh.TOK)
    tok = RC._split(_refs(req), F_["req"])["tok"]
    root = [x for i in range(8) for x in Kc.bits(int.from_bytes(BK.ROOT[8 * i:8 * i + 8], "big"), 64)]
    prints = {}
    for k in range(S.K):
        parent = root
        for j in range(S.CU):
            at = (k * S.CU + j) * S.B
            tk = [list(t) + Kc.bits(0, sh.TOK - S.CTW) for t in tok[at:at + S.B]]
            o = _refs(B.call(bk, delta, _coll(BK.DIGEST, parent), col(Array(sh.TOK, BIT), tk)))
            parent, prints[(k, j)] = o[:512], o[512:640]

    lay, elay = V3S._carry_layout(S3), V3S._entry_layout(S3)
    size, esize = V3S._size(lay), V3S._size(elay)
    carry_t, ecarry_t = Array(size, BIT), Array(esize, BIT)
    rest = size - (1 + 3 * 128 + 2 * TT + 2 * NW)
    init = (Kc.bits(1, 1) + Kc.bits(1, 128) * 3 + Kc.bits(1, TT) + Kc.bits(0, TT) + Kc.bits(1, NW) + Kc.bits(S.NB - 1, NW)
            + Kc.bits(0, rest))
    xs = [col(t, ck[name]) for name, t in (("s_start", BIT), ("s_kind", Array(3, BIT)), ("s_r", Array(RW, BIT)),
                                            ("s_a", Array(CW, BIT)), ("s_b", Array(CW, BIT)), ("s_v", F),
                                            ("s_t", Array(TT, BIT)))]
    done = _refs(B.scan(slot_p, _coll(carry_t, init), xs, [alpha, gamma]))[:size]
    fin = V3S._unpack(lay, done)

    def entries(head, en, tag, a, v0, v, t):
        init = head.R + head.W + head.ok + Kc.bits(1, 1) + Kc.bits(0, 1 + G + 128)
        xs = [col(BIT, en), col(Array(G, BIT), tag), col(F, a), col(F, v0), col(F, v), col(Array(TT, BIT), t)]
        return V3S._unpack(elay, _refs(B.scan(entry_p, _coll(ecarry_t, init), xs, [alpha, gamma]))[:esize])

    dn, empty = dense(S), V3S.empty_pool(S3, {})
    e1 = entries(fin, [Kc.bits(1, 1)] * len(dn), [Kc.bits(tag, G) for tag, _a in dn], [Kc.bits(a, 128) for _t, a in dn],
                 [prints[(a & ((1 << RW) - 1), a >> RW)] if tag == T3["RK"] else Kc.bits(empty(tag, a), 128)
                  for tag, a in dn], ck["d_v"], ck["d_t"])
    zero = Kc.bits(0, 1)[0]
    e2 = entries(e1, ck["m_en"], [[e[0] if T3["M"] >> j & 1 else zero for j in range(G)] for e in ck["m_en"]],
                 ck["m_a"], [Kc.bits(0, 128)] * S.NM, ck["m_v"], ck["m_t"])
    end = _refs(B.call(end_p, _coll(carry_t, done), _coll(ecarry_t, [x for f, _w in elay for x in getattr(e2, f)])))
    calls, Ps = end[1 + 2 * NW:1 + 2 * NW + TT], end[1 + 2 * NW + TT:1 + 2 * NW + TT + 128]
    return B.call(ver_p, _coll(Array(len(FLAGS) - 1, BIT), flags), _coll(BIT, end[:1]), _coll(Array(TT, BIT), calls),
                  _coll(F, Ps), _coll(Array(TT, BIT), bus[:TT]), _coll(F, bus[TT:]))


_DOC = ("the compile slice over the memory-checked pool, one window whole from the empty engine and pool: from the advice "
        "(its DA decisions and step count), the arrivals' ingress fields and their registered output lengths and tokens, "
        "the load, ReplayStep_v1 a step and its pool calls in vLLM's order as records on a bus to CacheStep_v3's stream "
        "of NS memory-checked slots, which runs in the unit from the empty pool with RK's keys the arrivals' BlockKey_v1 "
        "chains printed under δ; each hit, evictable count and return checked witness the bus carries; the served rows "
        "and last rows held to the registered srv and cout (verdict bits load, steps, cache, served, carry, pool)")
CompileSlice = CompositeDefinition("CompileSlice", 4, STATICS, _sig, _body, conformance="exact-model-tested", doc=_DOC)


def statics(R: int, W: int, A: int, engine: dict, IW: int, Q: int, D: int, P: int, K: int, CTW: int, CU: int, DA: int,
            NS: int, NM: int) -> dict:
    st = CS.statics(R, W, A, engine, IW, Q=Q, D=D, P=P, K=K, CTW=CTW, CU=CU)
    st = {**{k: v for k, v in st.items() if k != "CN"}, "DA": DA, "NS": NS, "NM": NM}
    shape(SimpleNamespace(**st))
    return st


def unit(**st):
    return bind(CompileSlice, **st)


#: the demonstrations' coins, fixed so the tests are reproducible; a window's are the verifier's, flipped live
COINS = tuple(int.from_bytes(hashlib.sha512(b"compile slice v4 demo coin " + bytes([i])).digest()[:16], "little")
              for i in range(3))


#: `compile_slice.DEMO_ENGINE` at two sequences of 16 tokens and six blocks (five past the null block), so the pool evicts
DEMO_ENGINE = {**CS.DEMO_ENGINE, "max_num_seqs": 2, "max_model_len": 16, "num_gpu_blocks": 6}


def demo() -> tuple[dict, CS.Slice]:
    """The demonstration class's statics and slice: five steps at (R 2, W 1, A 2) on `DEMO_ENGINE`'s six 4-token blocks,
    three arrivals of 4-bit tokens, 4 units a request, its five decisions the advice (DA = 5), with the stream's class
    the ladder's NS = 192 slots, which holds its window's 19 calls (120 slots) and every random window of the tests (to
    130), and NM = 8 M entries. Request 1 shares request 0's first block: it is looked up at step 0, takes that hit at
    step 1 and leaves at once, freeing three blocks one of which slot 0 still holds (the shared-block free). Both
    computes of step 2 evict queued blocks; request 2 is preempted at step 3 and readmitted at step 4 with a hit on its
    own first block."""
    st = statics(2, 1, 2, DEMO_ENGINE, 3, Q=3, D=5, P=5, K=3, CTW=4, CU=4, DA=5, NS=192, NM=8)
    tokens = ((8, 14, 9, 14, 10, 4, 3, 13, 9, 8, 11, 10, 13, 3, 2), (8, 14, 9, 14, 7, 8, 11, 12, 10, 11, 3, 10, 1),
              (10, 1, 13, 8, 6, 8))
    return st, CS.Slice(((0, 11, 4), (1, 12, 1), (2, 4, 2)), (0, 1, 1, 0, 0), 5, tokens)


def demo_async() -> tuple[dict, CS.Slice]:
    """The asynchronous demonstration class's: `demo`'s at AS = 1, with NS = 128 slots, which holds its window's 14 calls
    (86 slots) and every advice and random window of the tests (to 115), and NM = 8. Request 2 shares request 0's first
    block: it is looked up at step 3 and admitted at step 4 with that hit, its compute evicting the oldest queued
    block, and request 0 then leaves holding the shared block; slot 0's late `cache_blocks` caches a block at step 4."""
    st = statics(2, 1, 2, {**DEMO_ENGINE, "async_scheduling": True}, 3, Q=3, D=5, P=5, K=3, CTW=4, CU=4, DA=5, NS=128,
                 NM=8)
    tokens = ((6, 10, 14, 1, 8, 5, 12, 14, 11, 9, 2, 13, 8), (13, 5, 6, 1, 11),
              (6, 10, 14, 1, 10, 9, 14, 4, 5, 10, 9, 3, 4, 9))
    return st, CS.Slice(((0, 9, 4, 1), (1, 4, 1, 0), (2, 12, 2, 0)), (0, 1, 0, 1, 1), 5, tokens)


# --- the specification ---------------------------------------------------------------------------------------------------


class Refused(Exception):
    """The slice is out of the class, or a call or a bound refuses it; in `pool_step` ``partial`` holds the hits,
    evictable counts and returns derived before the refusal (the given ones past it) and the order they were derived
    in."""

    partial: tuple | None = None


def _need(ok: bool, why: str) -> None:
    if not ok:
        raise Refused(why)


def tap(S, st: RS.State, inp: RS.Inputs) -> SimpleNamespace:
    """The wires of `ReplayStep_v1`'s circuit the calls bind (`replay_step._build`'s ``tap``), for a step it accepts:
    ``run`` (the running slots, None empty), ``needs``, ``kept``, ``popped``, ``avail`` and ``upto``, ``run_n``,
    ``cands`` (per candidate: id, ntok, reach, admit, new, c), ``post`` (the placed slots, each a dict with its
    candidate c or None) and ``update`` (per placed slot: on, comp, sampled, leave, held, keep)."""
    R, W, A, B, H, L, AS = S.R, S.W, S.A, S.B, S.H, S.L, S.AS

    def rec(r: RS.Req, comp: int, running: bool) -> dict:
        return {"id": r.id, "P": r.P, "O": r.O, "out": r.out, "comp": comp,
                "max_tokens": getattr(r, O_IS_MAX_TOKENS) if AS else 0, "fin": r.fin if AS and running else 0}

    run = [rec(r, r.comp, True) for r in st.running] + [None] * (R - len(st.running))
    wait = [rec(r, 0, False) for r in st.waiting]
    arr = [{**rec(a, 0, False), "out": 0} for a in inp.arrivals]
    if not st.running and not st.waiting and arr:
        wait, arr = arr[:1], arr[1:]
    wait, arr = (wait + [None] * W)[:W], (arr + [None] * A)[:A]
    budget, ns, needs = S.T, [], []
    for r in run:
        n = 0
        if r is not None and not (r["fin"] and r["max_tokens"]):
            n = r["P"] + r["out"] - r["comp"]
            n = H if 0 < H < n else n
            n = min(n, budget, L - 1 - r["comp"])
        ns.append(n)
        needs.append(RS.cdiv(r["comp"] + n, B) - RS.cdiv(r["comp"], B) if r is not None else 0)
        budget -= n
    backs = [inp.ret[j] if run[j] is not None else 0 for j in range(R)]
    avail = [st.free + sum(backs[i + 1:]) for i in range(R)]
    upto = [sum(needs[:i + 1]) for i in range(R)]
    kept = [run[i] is not None and avail[i] >= upto[i] for i in range(R)]
    popped = [run[i] is not None and not kept[i] for i in range(R)]
    run_n = [ns[i] if kept[i] else 0 for i in range(R)]
    free = st.free + sum(b for b, p in zip(backs, popped) if p) - sum(x for x, k in zip(needs, kept) if k)
    budget = S.T - sum(run_n)
    cnt0 = cnt = len(st.running)
    going, cands = not any(popped), []
    for c in range(W + A):
        r = wait[c] if c < W else arr[c - W]
        reach = going and r is not None and budget > 0 and (cnt < S.M if S.M <= R else True)
        x = {"c": c, "id": r["id"] if r else 0, "ntok": r["P"] + r["out"] if r else 0, "reach": reach, "admit": False,
             "new": 0, "rec": r}
        go = False
        if reach:
            h, ev = inp.hit[c], inp.evict[c]
            hb, ntok = h * B, x["ntok"]
            n = ntok - hb
            n = H if 0 < H < n else n
            fits = S.CH or budget >= n
            n = min(n, budget)
            extra = ev + (S.WM if cnt else 0)
            full = max(RS.cdiv(min(ntok, L), B) - h, 0)
            new = max(RS.cdiv(min(hb + n, L), B) - h, 0)
            go = fits and (not S.FULL or free >= full + extra) and free >= new + extra
            x.update(admit=go and bool(inp.bits[c - W]) if c >= W else go, new=new, hb=hb, n=n)
            if x["admit"]:
                cnt, free, budget = cnt + 1, free - new - ev, budget - n
        going = x["admit"] if c >= W else going and (r is None or go)
        cands.append(x)
    post: list[dict | None] = [{**run[p], "n": run_n[p], "cand": None} if kept[p] else None for p in range(R)]
    for k, x in enumerate(c for c in cands if c["admit"]):
        post[cnt0 + k] = {**x["rec"], "comp": x["hb"], "n": x["n"], "fin": 0, "cand": x}
    update = []
    for s in post:
        if s is None:
            update.append(None)
            continue
        on, comp = s["n"] > 0, s["comp"] + s["n"]
        sampled = on and comp >= s["P"] + s["out"]
        sample = sampled and not s["fin"]
        stop = sample and s["out"] + 1 == s["O"]
        leave = bool(s["fin"]) if AS else stop
        update.append({"on": on, "comp": comp, "sampled": sampled, "leave": leave, "held": RS.cdiv(comp, B),
                       "keep": not leave})
    return SimpleNamespace(run=run, needs=needs, kept=kept, popped=popped, avail=avail, upto=upto, run_n=run_n,
                           cnt0=cnt0, cands=cands, post=post, update=update)


def positions(S, t: SimpleNamespace) -> dict[int, tuple[str, int]]:
    """The running loop's events by position: kept slot i's compute at i plus the victims freed before it, victim j's
    free at (the running count − 1 − j) plus the computes before it. Victim j is freed before compute i iff
    avail[j] < upto[i]."""
    out = {}
    for i in range(S.R):
        if t.kept[i]:
            out[i + sum(t.popped[j] and t.avail[j] < t.upto[i] for j in range(S.R))] = ("compute", i)
        if t.popped[i]:
            out[t.cnt0 - 1 - i + sum(t.kept[k] and not t.avail[i] < t.upto[k] for k in range(S.R))] = ("free", i)
    return out


def pool_step(S, stream: V3S.Stream, st: RS.State, inp: RS.Inputs, nc: Sequence[int], fl: Sequence[int],
              rnum: dict[int, int]) -> SimpleNamespace:
    """The step's pool calls (the module's docstring) on `stream`, for ReplayStep_v1's state `st` and inputs `inp` (an
    accepted step), each running slot's cached blocks `nc` and ``fl`` `fl`, and the request numbers `rnum` (id -> r);
    raises `Refused`. Returns the hits, evictable counts and returns the calls derive (0 where no call derives one, so
    a kept slot's returns are 0), the order they were derived in, ReplayStep_v1's wires (`tap`) and the next slot
    row (nc, fl)."""
    B, R, HW = S.B, S.R, RS.widths(CS.step_statics(S)).HW
    t = tap(S, st, inp)
    hit, ev, ret = list(inp.hit), list(inp.evict), list(inp.ret)
    seen: list[tuple[str, int]] = []
    nc = list(nc) + [0] * (R - len(nc))

    def num(rid: int) -> int:
        _need(rid in rnum, f"request {rid}'s tokens aren't in req")
        return rnum[rid]

    def call(kind: str, q: int, a: int, b: int = 0) -> tuple[int, int]:
        if not (a if kind in ("hit", "free") else b):
            return 0, 0
        try:
            return stream.call(V3S.Call(kind, q, a, b))
        except (V3S.Refused, ValueError) as x:
            raise Refused(f"the stream refuses {kind} {(q, a, b)}: {x}") from None

    def cache_to(p: int, q: int, tg: int) -> None:
        _need(tg <= S.CU, f"slot {p} caches to block {tg}, past CU = {S.CU}")
        if tg > nc[p]:
            call("cache", q, nc[p], tg - nc[p])
            nc[p] = tg

    def run() -> None:
        events = positions(S, t)
        for pos in sorted(events):
            kind, i = events[pos]
            r = t.run[i]
            q, held = num(r["id"]), RS.cdiv(r["comp"], B)
            if kind == "free":
                ret[i] = call("free", q, held)[0]
                seen.append(("p", i))
                nc[i] = 0
            else:
                call("alloc", q, held, t.needs[i])
                if t.run_n[i]:
                    cache_to(i, q, min(r["comp"] + t.run_n[i], r["P"] + r["out"] - fl[i]) // B)
        placed = t.cnt0
        for c, x in enumerate(t.cands):
            if not x["reach"]:
                continue
            q = num(x["id"])
            hit[c], ev[c] = call("hit", q, (x["ntok"] - 1) // B)
            seen.append(("c", c))
            _need(hit[c] < 1 << HW and ev[c] < 1 << HW, f"candidate {c}'s hit {hit[c]} or count {ev[c]} past HW")
            if x["admit"]:
                h, p = inp.hit[c], placed
                placed += 1
                s = t.post[p]
                call("take", q, 0, h)
                call("alloc", q, h, x["new"])
                tg = min(s["comp"] + s["n"], s["P"] + s["out"]) // B if s["n"] else 0
                _need(tg <= S.CU, f"slot {p} caches to block {tg}, past CU = {S.CU}")
                if tg > h:
                    call("cache", q, h, tg - h)
                nc[p] = max(h, tg)
        for p in range(R):
            s, u = t.post[p], t.update[p]
            if s is None:
                continue
            q = num(s["id"])
            if S.AS and s["cand"] is None and fl[p]:
                cache_to(p, q, (u["comp"] - u["sampled"]) // B)
            if u["leave"]:
                ret[p] = call("free", q, u["held"])[0]
                seen.append(("p", p))
                nc[p] = 0

    try:
        run()
    except Refused as x:
        x.partial = (tuple(hit), tuple(ev), tuple(ret), tuple(seen))
        raise
    hit, ev = ([v if ("c", c) in seen else 0 for c, v in enumerate(xs)] for xs in (hit, ev))
    ret = [v if ("p", p) in seen else 0 for p, v in enumerate(ret)]
    keep = [p for p in range(R) if t.update[p] is not None and t.update[p]["keep"]]
    nnc = tuple(nc[p] for p in keep) + (0,) * (R - len(keep))
    nfl = (tuple(int(t.update[p]["sampled"]) for p in keep) + (0,) * (R - len(keep))) if S.AS else (0,) * R
    return SimpleNamespace(hit=tuple(hit), ev=tuple(ev), ret=tuple(ret), order=tuple(seen), tap=t, nc=nnc, fl=nfl)


def _returns(S, stream: V3S.Stream, st: RS.State, rnum: dict[int, int]) -> list[int]:
    """Each running slot's returns were it freed after the slots past it (frees from the last slot down on a copy): the
    fixed point's first guess, exact for the victims."""
    trial, out = copy.deepcopy(stream), [0] * S.R
    for j in reversed(range(len(st.running))):
        r = st.running[j]
        held = RS.cdiv(r.comp, S.B)
        if held and r.id in rnum:
            out[j] = trial.call(V3S.Call("free", rnum[r.id], held))[0]
    return out


def request_numbers(sl: CS.Slice) -> dict[int, int]:
    """Each request id's number: the arrival slot it first occupies."""
    out: dict[int, int] = {}
    for k, x in enumerate(sl.arrivals):
        out.setdefault(x[0], k)
    return out


def rk_keys(S, sl: CS.Slice, delta: int) -> dict[int, int]:
    """RK's initial values: each arrival slot's CU block keys over its tokens (zero past them) printed under δ."""
    S3, n, tok = stream_statics(S), S.CU * S.B, key_tok(S)
    out = {}
    for k in range(S.K):
        toks = tuple(sl.tokens[k])[:n] if k < len(sl.tokens) else ()
        for j, key in enumerate(BK.chain(toks + (0,) * (n - len(toks)), S.B, tok)):
            out[V3S.rk(S3, k, j)] = BK.key_print(key, delta)
    return out


def scan(S, sl: CS.Slice, delta: int = COINS[2], stream: V3S.Stream | None = None) -> SimpleNamespace:
    """The window as a prover runs it from `sl` (the module's docstring), its calls on `stream` (default a new one on the
    empty pool, RK's keys `rk_keys`'; a given stream is copied, never changed). ``refused`` says why the load, a step or
    a call refuses the slice, where one does, and the run ends there; it is None otherwise."""
    sh = shape(S)
    Sc, St = CS.chain(S), CS.step_statics(S)
    arrivals = tuple(map(tuple, sl.arrivals))
    loaded = RC.load(Sc, RC.empty(Sc), RC.QueueRow(), arrivals, sl.decisions, sl.steps)
    if loaded is None:
        return SimpleNamespace(refused="the load refuses the slice")
    cur, win = loaded
    rnum = request_numbers(sl)
    if stream is None:
        stream = V3S.Stream(sh.S3, V3S.empty_pool(sh.S3, rk_keys(S, sl, delta)), 1, S.NB - 1)
    CA, R, names = sh.CA, S.R, RC.arrival_names(Sc)
    nc, fl = (0,) * R, (0,) * R
    zero = ((0,) * R, (0,) * CA, (0,) * CA, (0,) * R)
    states, fronts, steps, pools, wit = [cur], [], [], [(nc, fl)], []
    for u in range(S.P):
        arr, bits = RC.front(Sc, win, cur)
        fronts.append((arr, bits))
        if not cur.steps:
            states.append(cur)
            pools.append((nc, fl))
            wit.append(zero)
            continue
        st = CS._state(Sc, cur)
        reqs = tuple(RS.Req(**dict(zip(names, x))) for x in arr)
        h, e, r = [0] * CA, [0] * CA, _returns(S, stream, st, rnum)
        for _round in range(4 * (CA + R) + 8):
            inp = RS.Inputs(reqs, tuple(bits), tuple(h), tuple(e), tuple(r))
            out = RS.reference(St, st, inp)
            if not out.ok:
                return SimpleNamespace(refused=f"step {u} refuses: the slice's class can't take it")
            trial, refused = copy.deepcopy(stream), None
            try:
                got = pool_step(S, trial, st, inp, nc, fl, rnum)
                dh, de, dr, order = got.hit, got.ev, got.ret, got.order
            except Refused as x:
                if x.partial is None:
                    raise
                refused, (dh, de, dr, order) = x, x.partial
            k = next(((kind, i) for kind, i in order
                      if ((dh[i], de[i]) != (h[i], e[i]) if kind == "c" else dr[i] != r[i])), None)
            if k is not None:
                if k[0] == "c":
                    h[k[1]], e[k[1]] = dh[k[1]], de[k[1]]
                else:
                    r[k[1]] = dr[k[1]]
                continue
            if refused is not None:
                return SimpleNamespace(refused=f"step {u}: {refused}")
            if (got.hit, got.ev, got.ret) == (tuple(h), tuple(e), tuple(r)):
                break
            h, e, r = list(got.hit), list(got.ev), list(got.ret)
        else:
            return SimpleNamespace(refused=f"step {u}'s hits and returns don't settle")
        stream, nc, fl, t = trial, got.nc, got.fl, got.tap
        backs = tuple(r[j] if t.popped[j] else 0 for j in range(R))
        rets = tuple(r[p] if (t.update[p] or {}).get("leave") else 0 for p in range(R))
        wit.append((backs, tuple(h), tuple(e), rets))
        pools.append((nc, fl))
        stepped = CS._row(Sc, out.state)
        nxt = RC.Row(stepped.running, stepped.waiting, stepped.free, cur.qh + out.taken, cur.dh + out.used, cur.steps - 1)
        steps.append(out.rows)
        states.append(nxt)
        cur = nxt
    return SimpleNamespace(states=states, window=win, fronts=fronts, steps=steps, pools=pools, wit=wit, stream=stream,
                           refused=None)


def _encode_pool(S, nc: Sequence[int], fl: Sequence[int]) -> list[int]:
    CW = shape(S).CW
    return [b for v in nc for b in CS._bits(v, CW)] + ([int(x) for x in fl] if S.AS else [])


def reference(S, sl: CS.Slice, delta: int = COINS[2]) -> SimpleNamespace | None:
    """The registered rows and the checked witness the unit accepts `sl`'s advice with under the print coin δ, by `scan`'s
    run (``srv``, ``cout`` and ``chk``'s bits, unpadded), or None where the load, a step, a step's heads
    (`replay_chain.advance`: it takes more arrivals than are queued, or reads more than A decisions or past the
    window's), a call or the class (the stream past NS slots or NM M entries, an access outside the touched list's
    runs) refuses the slice."""
    sh = shape(S)
    sc = scan(S, sl, delta)
    if sc.refused is not None:
        return None
    for u, (cur, nxt) in enumerate(zip(sc.states, sc.states[1:])):
        if cur.steps:
            arr, _bits = sc.fronts[u]
            if nxt.qh - cur.qh > len(arr) or nxt.dh - cur.dh > S.A or nxt.dh > len(sc.window.decisions):
                return None
    stm = sc.stream
    ms = [x for x in stm.entries() if x[0] == T3["M"]]
    where = set(dense(S))
    if len(stm.slots) > S.NS or len(ms) > S.NM or any(x[0] != T3["M"] and (x[0], x[1]) not in where
                                                      for x in stm.entries()):
        return None
    Sc = CS.chain(S)
    steps = list(sc.steps) + [None] * (S.P - len(sc.steps))
    srv = [b for rows in steps for b in CS.rows_bits(S, rows)]
    cout = list(RC.encode(sc.states[-1], Sc)) + list(RC.encode_window(sc.window, Sc)) + _encode_pool(S, *sc.pools[-1])
    return SimpleNamespace(srv=srv, cout=cout, chk=_chk(S, sh, sc), scan=sc)


def _chk(S, sh, sc) -> list[int]:
    """The checked row (`fields`' ``chk`` layout) of `scan`'s run."""
    CW, TT, S3 = sh.CW, sh.TT, sh.S3
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
    ms = [x for x in stm.entries() if x[0] == T3["M"]]
    ms += [(0, 0, 0, 0, 0)] * (S.NM - len(ms))
    out += [int(x[0] == T3["M"]) for x in ms] + [b for x in ms for b in bits(x[1], 128)]
    out += [b for x in ms for b in bits(x[3], 128)] + [b for x in ms for b in bits(x[4], TT)]
    return out


def _padded(S, r: str, bits: Sequence[int]) -> list[int]:
    n = data_bits(S)[r]
    if len(bits) != n:
        raise ValueError(f"row {r} holds {n} bits, not {len(bits)}")
    return list(bits) + [0] * (row_bits(S)[r] - n)


def arguments(S, sl: CS.Slice, coins: Sequence[int] = COINS, srv: Sequence[int] | None = None,
              cout: Sequence[int] | None = None, chk: Sequence[int] | None = None) -> dict[str, list[int]]:
    """The unit's inputs for slice `sl` under the coins (α, γ, δ) (bits each, `fields`' layouts, the registered rows
    zero-padded): ``srv``, ``cout`` and ``chk`` as given, by default `reference`'s (zero where it refuses the slice)."""
    Sc = CS.chain(S)
    Z = RC.widths(Sc)
    if len(sl.decisions) != S.DA or len(sl.arrivals) > S.K:
        raise ValueError(f"the slice's {len(sl.decisions)} decisions aren't DA = {S.DA}, or its arrivals exceed K")
    names = RC.arrival_names(Sc)
    arr = [dict(zip(names, a)) for a in sl.arrivals] + [None] * (S.K - len(sl.arrivals))

    def col(name, wd):
        return [b for x in arr for b in CS._bits(x[name] if x else 0, wd)]

    n = S.CU * S.B
    toks = [tuple(t)[:n] for t in sl.tokens] + [()] * (S.K - len(sl.tokens))
    if srv is None or cout is None or chk is None:
        ref = reference(S, sl, coins[2])
        nb = data_bits(S)
        srv = (ref.srv if ref else [0] * nb["srv"]) if srv is None else srv
        cout = (ref.cout if ref else [0] * nb["cout"]) if cout is None else cout
        chk = (ref.chk if ref else [0] * nb["chk"]) if chk is None else chk
    req = (col("O", Z.LW) + (col(O_IS_MAX_TOKENS, 1) if S.AS else [])
           + [b for t in toks for v in list(t) + [0] * (n - len(t)) for b in CS._bits(v, S.CTW)])
    return {"adv": [int(d) for d in sl.decisions] + CS._bits(sl.steps, Z.PW),
            "ing": [int(x is not None) for x in arr] + col("id", S.IW) + col("P", Z.LW),
            "req": _padded(S, "req", req), "srv": _padded(S, "srv", srv), "cout": _padded(S, "cout", cout),
            "coin": [b for c in coins for b in CS._bits(c, 128)], "chk": list(chk)}


def run(st: dict, args: Sequence[dict[str, list[int]]]) -> list[int]:
    """The unit's verdict word on each of `args` (`arguments`' form), in one evaluation."""
    out = evaluate_bits(unit(**st), *[[a[r] for a in args] for r in ROWS]).reshape(len(args), -1).tolist()
    return [sum(int(b) << j for j, b in enumerate(o)) for o in out]


def evaluate(st: dict, slices: Sequence[CS.Slice], coins: Sequence[int] = COINS) -> list[int]:
    """The unit's verdict word on each slice, its registered rows and checked witness `reference`'s."""
    S = SimpleNamespace(**st)
    return run(st, [arguments(S, sl, coins) for sl in slices])


def size(**st) -> dict[str, int]:
    """The unit's ANDs (an n-input AND n − 1), its inputs' widths (padded) and its free (advice) width at statics `st`."""
    S = SimpleNamespace(**st)
    return {"ands": ands(unit(**st)), **{f"{r}_bits": b for r, b in row_bits(S).items()}, "free_bits": free_width(S)}
