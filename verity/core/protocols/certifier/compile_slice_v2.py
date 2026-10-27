"""The compile slice with vLLM's block pool: `CompileSlice_v1` (`compile_slice`) with its prefix cache replaced by
`cache_step_v2.CacheStep_v2`'s pool, wired to `ReplayStep_v1` by equality, so the evictable and returned counts are
derived at every step and eviction and frees are in the slice.

    CompileSlice_v2{R, W, A, M, T, L, B, H, CH, WM, FULL, PC, NB, IW, AS, EW, Q, D, P, K, CN, CTW, CU, G, CD, SW}
        (pub, adv, cert, srv0, …, srv{P/G − 1}, cin, cout, wit) -> verdict[16]

The rows, the verdict (bits load, steps, code, charge, rows, cache; `ACCEPT` = 0x3F), the load, the steps, the charge and
the served rows are CompileSlice_v1's, at any G dividing P (``srv`` alone at G = P). What changes is ``wit``'s cache
part and what ``cache`` checks.

**The witness** ``wit``: the state rows, the window row, each step's ``arr`` and ``bits`` (as CompileSlice_v1), then per
step its candidates' prefix hits ``hit`` and evictable counts ``ev`` (W + A each), its slots' returned counts ``ret``
(R), its computes' threshold stamps ``tau`` (R, by the row's slot), the pool rows B_0 … B_P (`pool_fields`: the pool
of CacheStep_v2 at N = CN, U = B, TW = CTW, RW = IW, L = CU·B, R, D = CD, NB and SW, and at AS = 1 each slot's ``fl``,
its last row's output still in flight), and at AS = 1 ``map``. ReplayStep_v1 reads ``hit``, ``ev`` (its ``evict``) and
``ret`` as its inputs. A field no call reads is 0: the hits and counts of a candidate the waiting loop doesn't reach,
the returns of a slot that is neither a victim nor leaving, the stamp of a slot with no row.

**The calls** (`pool_step`, the step's CacheStep_v2 calls on B_u, in the order vLLM makes them; `cache_pool_vectors`
holds the same calls to the replay's `_KV`). A pool slot is the running slot its request holds at the step's start.
- *The schedule's R positions.* Position t is the t-th event of the running loop: the compute of a kept slot i, or the
  free of a victim j. vLLM frees victims from the last slot down, each just before the allocation that needs it, so
  victim j is freed before kept slot i's compute iff free + (the returns of the slots after j) < (the needs up to i),
  both ReplayStep_v1's. At a position t at or past the running count, where the step admits, the row placed at slot t
  first takes its hit (a lookup that takes it, from its candidate's ``hit`` and ``ev``), then computes.
- *The lookup* of the candidate the waiting loop reaches and doesn't admit (at most one: the loop stops there), from
  its ``hit`` and ``ev``. ReplayStep_v1 reads that hit whether or not the candidate's decision is 1: the replay asks an
  arrival's decision only where it would be admitted.
- *The update*, per placed slot in order: at AS = 1 the late `cache_blocks` of a slot whose output was in flight
  (vLLM caches a decode's block when its output is appended, one step late), then the free of a leaving request.
Every compute's ``new`` is ReplayStep_v1's: a kept slot's need, an admitted row's new blocks, 0 for a late cache. Every
free's ``held`` is ReplayStep_v1's ⌈comp / B⌉, a victim's before the step, a leaver's after its row, and its ``ret`` is
the ``ret`` ReplayStep_v1 reads. A compute caches the units from its walk entry's level up to ⌊t / B⌋ for t the
tokens `cache_blocks` caches: min(comp + n, P + out − fl) for a running row, min(start + n, P + out) for an admitted
one, comp − (the row samples) for a late cache. So none of new, held, nins or a call's slot is witness. A free that
overstates ``held`` by k adds k blocks that don't exist to the free count, so the next k allocations
skip their evictions; a leaving request that shares a block with another slot can claim one block more and pass
ReplayStep_v1's ``ret`` ≤ held and the free count. Here its ``ret`` is held to the free of ReplayStep_v1's own
⌈comp / B⌉, which refuses it. After the update each next running slot takes its placed slot's walk entry, mask bits
and ``fl`` (the row sampled), and a slot that leaves the list must hold nothing; an invalid entry's fields are zero,
so a pool row has one encoding.

**cache** (bit 5) is: B_0 the empty pool; each state row's free count equal to its pool row's hashless free count plus
its queued copies; and at every active step every call's ok, the class's bounds (a lookup's hit within KL = min(CN + 1,
CU) units, a compute's units within KC = min(⌊(B − 1 + T) / B⌋, CU) and a late cache's within one, every unit within
CU, each count within CacheStep_v2's widths, every request's tokens in ``cert``), the unread witness 0, and B_{u+1}
the calls' pool (B_u on an idle step).

**The reference** is `check`; `case` builds the rows of a prover's run (`scan`), with the choices: ``overstate``, the
free of a request that holds ``held`` + k; ``made_up``, a compute that evicts the oldest queued
copies with no eviction due; ``taus`` and ``rets``, witness stamps and returns other than the pool's.

**Statics**: CompileSlice_v1's with G always given (P for one ``srv`` row), PC = 1, and CacheStep_v2's D (CD) and SW. A
window whose pool needs more than CN entries or CD copies of an entry, or whose clock passes SW bits, is refused: the
slice's class. Without prefix caching CompileSlice_v1 has no pool to wire: its returns are ⌈comp / B⌉ and nothing is
evicted.

**Cost** (`size`): a step makes 3R + 1 calls (R takes, R running-loop events, a lookup, R frees) and at AS = 1 R late
caches more, each CacheStep_v2's eviction, free and end and a window of KL, KC, 1 or no units (`cache_step_v2.size`).
The demonstration units are 83,092 ANDs (`demo`, at G = 5 and G = 1) and 91,785 (`demo_async`). On corpus-smallkv's
first seven steps (CN 24, CU 25, 16-token blocks of 17-bit tokens, CD 1, SW 12) the unit is 7,984,682 ANDs, against
CompileSlice_v1's about 7.6M, and is its reference on the honest run and a stamp or a return off. At the crush traces'
class (CN 159 = NB − 1, R 16, CU 256) a call's eviction is 41,476 ANDs, a unit of its window 99,172 and its free 10,694,
so a step's calls are about 300M ANDs, 85% of them the R takes' KL = 160-unit windows: there
`benchmarks/network_e2e/encodings/compile_slice_pool.py` holds the calls to the replay's `_KV`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ONE, ZERO, Circuit, add, and_reduce, const_bits, mux, or_reduce
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array
from verity.core.protocols.certifier import arith_encode as AE, cache_step as CST, cache_step_v2 as C2
from verity.core.protocols.certifier import charge_chain as CC, compile_slice as CS, replay_chain as RC, replay_step as RS
from verity.core.protocols.certifier.head_check import BIT, _part

STATICS = CS.BLOCK_STATICS + ("CD", "SW")
FLAGS, ACCEPT, WORD = CS.FLAGS, CS.ACCEPT, CS.WORD
#: the units a late cache_blocks completes at most (the block holding the output it appends)
LATE = 1


def pool_statics(S) -> SimpleNamespace:
    """CacheStep_v2's statics in the slice: N = CN, U = B, TW = CTW, RW = IW, L = CU·B, R, D = CD, NB and SW."""
    return SimpleNamespace(**C2.statics(S.CN, S.B, S.CTW, S.IW, S.CU * S.B, S.R, S.CD, S.NB, S.SW))


def shape(S) -> SimpleNamespace:
    """The pool's statics Y and widths Z, ReplayStep_v1's widths ZR, the lookups' KL and the computes' KC units."""
    if not S.PC or S.EW:
        raise ValueError(f"statics {dict(S.__dict__)}: the pool slice is prefix caching's (PC = 1, EW = 0)")
    Y = pool_statics(S)
    return SimpleNamespace(Y=Y, Z=C2.widths(Y), ZR=RS.widths(CS.step_statics(S)), KL=min(S.CN + 1, S.CU),
                           KC=min((S.B - 1 + S.T) // S.B, S.CU))


# --- the pool row ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pool:
    """The pool between steps: CacheStep_v2's, its slots the running slots, and at AS = 1 each slot's ``fl``."""

    pool: C2.Pool
    fl: tuple[int, ...] = ()


def empty(S) -> Pool:
    return Pool(C2.empty(pool_statics(S)), (0,) * S.R if S.AS else ())


def pool_fields(S) -> list[tuple[str, int, int]]:
    Z = shape(S).Z
    N, D, R = S.CN, S.CD, S.R
    return ([("parent", N, Z.EW), ("tok", N, Z.UW), ("owner", N, S.IW), ("level", N, Z.HW), ("live", N * D, 1),
             ("mask", N * D, R), ("stamp", N * D, S.SW), ("spfx", R, Z.EW), ("hf", 1, Z.BW), ("clock", 1, S.SW)]
            + ([("fl", R, 1)] if S.AS else []))


def pool_width(S) -> int:
    return RC._bits_of(pool_fields(S))


def encode_pool(S, p: Pool) -> list[int]:
    """A pool row: CacheStep_v2's pool fields (an invalid entry and a dead copy zero), then ``fl``."""
    Y = pool_statics(S)
    bits = [b for f in C2.arguments(Y, p.pool, C2.Op())[:10] for b in f]
    return bits + ([int(x) for x in p.fl] if S.AS else [])


def decode_pool(S, bits: Sequence[int]) -> Pool:
    """A pool row as `Pool`; ValueError unless it is `encode_pool`'s of one."""
    F = RC._split(bits, pool_fields(S))
    N, D = S.CN, S.CD
    v = {k: [CS._int(e) for e in F[k]] for k, _n, _w in pool_fields(S)}
    table = []
    for x in range(N):
        n = next((d for d in range(D) if not v["live"][x * D + d]), D)
        dead = range(x * D + n, (x + 1) * D)
        if any(v["live"][k] or v["mask"][k] or v["stamp"][k] for k in dead) or (
                not n and any(v[k][x] for k in ("parent", "tok", "owner", "level"))):
            raise ValueError(f"entry {x}: live copies not a prefix, or a dead copy or invalid entry not zero")
        toks = tuple((v["tok"][x] >> (S.CTW * j)) & ((1 << S.CTW) - 1) for j in range(S.B))
        table.append(C2.Entry(v["parent"][x], toks, v["owner"][x], v["level"][x],
                              tuple(C2.Copy(v["mask"][x * D + d], v["stamp"][x * D + d]) for d in range(n)))
                     if n else None)
    pool = C2.Pool(tuple(table), tuple(v["spfx"]), v["hf"][0], v["clock"][0])
    p = Pool(pool, tuple(v["fl"]) if S.AS else ())
    if encode_pool(S, p) != [int(b) for b in bits]:
        raise ValueError("a pool row that isn't one pool's encoding")
    return p


def queued(pool: C2.Pool) -> int:
    return sum(1 for x in pool.table if x for c in x.copies if not c.mask)


def _level(Y, pool: C2.Pool, slot: int) -> int:
    e = pool.spfx[slot]
    return 0 if e == Y.N or pool.table[e] is None else pool.table[e].level


def returns(S, pool: Pool, st: RS.State) -> tuple[int, ...]:
    """Each running slot's returns were it freed on `pool` (the copies only it holds and its hashless blocks)."""
    out = [0] * S.R
    for j, r in enumerate(st.running):
        has = [c for x in pool.pool.table if x for c in x.copies if c.mask >> j & 1]
        out[j] = sum(c.mask == 1 << j for c in has) + RS.cdiv(r.comp, S.B) - len(has)
    return tuple(out)


# --- ReplayStep_v1's wires, in Python ------------------------------------------------------------------------------------------


def tap(S, st: RS.State, inp: RS.Inputs) -> SimpleNamespace:
    """The wires of `ReplayStep_v1`'s circuit the calls bind (`replay_step._build`'s ``tap``), for a step it accepts:
    ``run`` (the running slots, None empty), ``needs``, ``kept``, ``popped``, ``avail`` and ``upto``, ``run_n``,
    ``cands`` (per candidate: id, ntok, reach, admit, new, c), ``post`` (the placed slots, each a dict with its
    candidate c or None) and ``update`` (per placed slot: on, comp, sampled, leave, held, keep)."""
    R, W, A, B, H, L, AS = S.R, S.W, S.A, S.B, S.H, S.L, S.AS

    def rec(r: RS.Req, comp: int, running: bool) -> dict:
        return {"id": r.id, "P": r.P, "O": r.O, "out": r.out, "comp": comp, "cap": r.cap if AS else 0,
                "fin": r.fin if AS and running else 0}

    run = [rec(r, r.comp, True) for r in st.running] + [None] * (R - len(st.running))
    wait = [rec(r, 0, False) for r in st.waiting]
    arr = [{**rec(a, 0, False), "out": 0} for a in inp.arrivals]
    if not st.running and not st.waiting and arr:
        wait, arr = arr[:1], arr[1:]
    wait, arr = (wait + [None] * W)[:W], (arr + [None] * A)[:A]
    budget, ns, needs = S.T, [], []
    for r in run:
        n = 0
        if r is not None and not (r["fin"] and r["cap"]):
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


# --- the calls ----------------------------------------------------------------------------------------------------------------


class Refused(Exception):
    """A call or a bound of the class refuses the step; in `pool_step`'s derive mode ``partial`` holds the hits,
    evictable counts and returns derived before the refusal (the given ones past it) and the order they were derived
    in."""

    partial: tuple | None = None


def _need(ok: bool, why: str) -> None:
    if not ok:
        raise Refused(why)


def _longest(Y, pool: C2.Pool, units: Sequence[tuple[int, ...]], K: int, KL: int) -> tuple[int, int]:
    """A lookup's derived hit (the longest cached prefix it may take) and its evictable count."""
    index = {(x.parent, x.tokens): i for i, x in enumerate(pool.table) if x}
    pfx, j, ev = Y.N, 0, 0
    while j < KL and (j + 1) * Y.U < K and (pfx, tuple(units[j])) in index:
        pfx = index[(pfx, tuple(units[j]))]
        ev += not pool.table[pfx].copies[0].mask
        j += 1
    return j, ev


def _threshold(pool: C2.Pool, new: int) -> int:
    e = max(new - pool.hf, 0)
    q = sorted(c.stamp for x in pool.table if x for c in x.copies if not c.mask)
    _need(e <= len(q), "more blocks evicted than queued")
    return q[e - 1] if e else 0


def _evict_oldest(pool: C2.Pool, k: int) -> tuple[C2.Pool, int]:
    """`pool` with its k oldest queued copies taken as new blocks (each a free block, so hf gains them), and the k-th's
    stamp: a made-up eviction."""
    q = sorted(c.stamp for x in pool.table if x for c in x.copies if not c.mask)
    _need(0 < k <= len(q), f"no {k} queued copies to evict")
    tau = q[k - 1]
    table = []
    for x in pool.table:
        kept = tuple(c for c in x.copies if c.mask or c.stamp > tau) if x else ()
        table.append(replace(x, copies=kept) if kept else None)
    return replace(pool, table=tuple(table), hf=pool.hf + k), tau


def pool_step(S, pool: Pool, st: RS.State, inp: RS.Inputs, out: RS.Out, tokens: dict[int, Sequence[int]],
              tau: Sequence[int] = (), *, derive: bool = False, overstate: dict[int, int] | None = None,
              made_up: dict[int, int] | None = None) -> SimpleNamespace:
    """The step's CacheStep_v2 calls on `pool` (the module's docstring), for ReplayStep_v1's state `st`, inputs `inp`
    and output `out` (an accepted step), the requests' certified `tokens` and the computes' stamps `tau` (by slot);
    raises `Refused`. Returns the next pool, the calls made (kind, op), the candidates looked up, and the hits,
    evictable counts, returns and stamps the calls read. With `derive` each lookup's hit and evictable count, each
    free's returns and each compute's stamp are the pool's, the rest 0 (the prover's run); else `inp`'s and `tau`'s,
    the fields no call reads 0. A prover's choices: `overstate`, request -> the blocks its free claims beyond
    ⌈comp / B⌉ (and returns beyond its own); `made_up`, slot -> the queued copies its compute evicts with no eviction
    due."""
    sh = shape(S)
    Y, Z, KL, KC = sh.Y, sh.Z, sh.KL, sh.KC
    t = tap(S, st, inp)
    R, W, A, B, N = S.R, S.W, S.A, S.B, S.CN
    cur, fl = pool.pool, list(pool.fl) if S.AS else [0] * R
    seen: list[tuple[str, int]] = []
    if derive:
        hit, ev, ret, tau = list(inp.hit), list(inp.evict), list(inp.ret), [0] * R
    else:
        hit, ev, ret, tau = list(inp.hit), list(inp.evict), list(inp.ret), list(tau) + [0] * (R - len(tau))
        _need(not any(hit[c] or ev[c] for c in range(W + A) if not t.cands[c]["reach"]),
              "a hit or count of a candidate the waiting loop doesn't reach")
        _need(not any(ret[p] for p in range(R) if not (t.popped[p] or (t.update[p] or {}).get("leave"))),
              "returns of a slot that is neither a victim nor leaving")
        _need(not any(tau[p] for p in range(R) if t.post[p] is None), "a stamp of a slot with no row")
    overstate, made_up = overstate or {}, made_up or {}
    calls: list[tuple[str, C2.Op]] = []
    looked: list[int] = []
    lim = 1 << Z.HW

    def units(rid: int) -> tuple[tuple[int, ...], ...]:
        _need(rid in tokens, f"request {rid}'s tokens aren't in cert")
        return CST.units_of(S, tokens[rid])

    def call(kind: str, op: C2.Op) -> None:
        nonlocal cur
        got = C2.reference(Y, cur, op)
        _need(got.ok, f"CacheStep_v2 refuses the {kind} {op}")
        calls.append((kind, op))
        cur = got.pool

    def lookup(c: int, slot: int, take: bool) -> None:
        x = t.cands[c]
        us = units(x["id"])
        if derive:
            hit[c], ev[c] = _longest(Y, cur, us, x["ntok"], KL)
            seen.append(("c", c))
        _need(hit[c] < KL and ev[c] < lim, f"candidate {c}'s hit {hit[c]} past KL = {KL} units, or its count past HW")
        K = min(x["ntok"], (1 << Z.LW) - 1)
        looked.append(c)
        call("take" if take else "lookup", C2.Op(slot=slot, r=x["id"], units=us[:KL], admit=True, take=take, K=K,
                                                 wh=hit[c], ev=ev[c]))

    def compute(slot: int, rid: int, new: int, target: int, window: int, kind: str = "compute") -> None:
        nonlocal cur
        us = units(rid)
        lvl = _level(Y, cur, slot)
        nins = max(target - lvl, 0)
        _need(new < lim and target <= S.CU and nins <= window,
              f"slot {slot}'s compute: {new} new blocks, caching to unit {target} from {lvl} (window {window})")
        stamp = tau[slot] if kind == "compute" else 0
        if kind == "compute" and slot in made_up:
            cur, tau[slot] = _evict_oldest(cur, made_up[slot])
            stamp = 0
        elif kind == "compute" and derive:
            stamp = tau[slot] = _threshold(cur, new)
        window_units = (us + ((0,) * B,) * window)[lvl:lvl + window]
        call(kind, C2.Op(slot=slot, r=rid, units=window_units, nins=nins, new=new, tau=stamp))

    def free(slot: int, p: int, rid: int, held: int, kind: str = "free") -> None:
        held += overstate.get(rid, 0)
        if derive:
            bit = 1 << slot
            has = [c for x in cur.table if x for c in x.copies if c.mask & bit]
            ret[p] = sum(c.mask == bit for c in has) + held - len(has)
            seen.append(("p", p))
        _need(held < lim and ret[p] < lim, f"slot {slot}'s free: {held} held, {ret[p]} returned past HW")
        call(kind, C2.Op(slot=slot, free=True, held=held, ret=ret[p]))

    def run() -> None:
        events = positions(S, t)
        for p in range(R):
            s = t.post[p]
            if s is not None and s["cand"] is not None:
                lookup(s["cand"]["c"], p, True)
            if p in events:
                kind, i = events[p]
                r = t.run[i]
                if kind == "free":
                    free(i, i, r["id"], RS.cdiv(r["comp"], B), "preempt")
                else:
                    target = min(r["comp"] + t.run_n[i], r["P"] + r["out"] - fl[i]) // B if t.run_n[i] else 0
                    compute(i, r["id"], t.needs[i], target, KC)
            elif s is not None and s["cand"] is not None:
                compute(p, s["id"], s["cand"]["new"], min(s["comp"] + s["n"], s["P"] + s["out"]) // B, KC)
        left = [x["c"] for x in t.cands if x["reach"] and not x["admit"]]
        if left:
            lookup(left[0], 0, False)
        for p in range(R):
            s, u = t.post[p], t.update[p]
            if s is None:
                continue
            if S.AS and s["cand"] is None and fl[p]:
                compute(p, s["id"], 0, (u["comp"] - u["sampled"]) // B, LATE, "late")
            if u["leave"]:
                free(p, p, s["id"], u["held"])
        _need(not any(c.mask >> p & 1 for x in cur.table if x for c in x.copies for p in gone)
              and all(cur.spfx[p] == N for p in gone), "a slot that leaves the running list still holds blocks")

    keep = [p for p in range(R) if t.update[p] is not None and t.update[p]["keep"]]
    gone = [p for p in range(R) if p not in keep]
    try:
        run()
    except Refused as x:
        if derive:
            x.partial = (tuple(hit), tuple(ev), tuple(ret), tuple(seen))
        raise
    if derive:
        hit, ev = ([v if ("c", c) in seen else 0 for c, v in enumerate(xs)] for xs in (hit, ev))
        ret = [v if ("p", p) in seen else 0 for p, v in enumerate(ret)]

    def moved(m: int) -> int:
        return sum((m >> p & 1) << q for q, p in enumerate(keep))

    table = tuple(replace(x, copies=tuple(replace(c, mask=moved(c.mask)) for c in x.copies)) if x else None
                  for x in cur.table)
    spfx = tuple(cur.spfx[p] for p in keep) + (N,) * (R - len(keep))
    nfl = tuple(int(t.update[p]["sampled"]) for p in keep) + (0,) * (R - len(keep)) if S.AS else ()
    return SimpleNamespace(pool=Pool(replace(cur, table=table, spfx=spfx), nfl), calls=calls, looked=tuple(looked),
                           hit=tuple(hit), ev=tuple(ev), ret=tuple(ret), tau=tuple(tau), order=tuple(seen))


# --- the rows ----------------------------------------------------------------------------------------------------------------


def fields(S) -> dict[str, list[tuple[str, int, int]]]:
    """CompileSlice_v1's rows, with ``wit``'s cache part the pool's (the module's docstring)."""
    shape(S)
    F = CS.fields(S)
    Sc, HW, CA = CS.chain(S), RS.widths(CS.step_statics(S)).HW, S.W + S.A
    F["wit"] = ([("state", S.P + 1, RC.width(Sc)), ("window", 1, RC.window_width(Sc)),
                 ("arr", S.P, RC._bits_of(RC.front_fields(Sc))), ("bits", S.P, S.A), ("hit", S.P * CA, HW),
                 ("ev", S.P * CA, HW), ("ret", S.P * S.R, HW), ("tau", S.P * S.R, S.SW), ("pool", S.P + 1, pool_width(S))]
                + ([("map", S.P * S.R, S.R)] if S.AS else []))
    return F


def row_bits(S) -> dict[str, int]:
    return {r: CS._words_of(RC._bits_of(fs)) for r, fs in fields(S).items()}


def row_names(S) -> tuple[str, ...]:
    return CS.row_names(S)


# --- the circuit ------------------------------------------------------------------------------------------------------------


def _flat(xs) -> list:
    return [b for e in xs for b in e]


def _unpool(S, bits) -> dict:
    """A pool row's wires by field: lists of elements, ``live`` and ``fl`` single wires, ``hf`` and ``clock`` words."""
    F = RC._split(bits, pool_fields(S))
    return {"parent": F["parent"], "tok": F["tok"], "owner": F["owner"], "level": F["level"],
            "live": [e[0] for e in F["live"]], "mask": F["mask"], "stamp": F["stamp"], "spfx": F["spfx"],
            "hf": F["hf"][0], "clock": F["clock"][0], "fl": [e[0] for e in F["fl"]] if S.AS else [ZERO] * S.R}


def _call(C, Y, pool: dict, slot, r, flags, *, K=None, wh=None, ev=None, new=None, tau=None, held=None, ret=None,
          units=(), window: int = 0, target=None, derive: bool = False) -> tuple:
    """One CacheStep_v2 call on `pool` (`cache_step_v2._body`'s parts, its window `window` units): (ok, the next pool).
    With `target` the window starts at the slot's walk level and caches up to unit `target` (nins = target − level,
    within the window); else it is `units`' first `window` from ROOT with nins 0. With `derive` the free's returns are
    its own, not `ret`, and follow: (ok, the next pool, the returns in HW bits)."""
    Z = C2.widths(Y)
    H = [ZERO] * Z.HW
    K, wh, ev, new, held, ret = (x if x is not None else z for x, z in ((K, [ZERO] * Z.LW), (wh, H), (ev, H), (new, H),
                                                                        (held, H), (ret, H)))
    tau = tau if tau is not None else [ZERO] * Y.SW
    parent, tok, owner, level = (_flat(pool[k]) for k in ("parent", "tok", "owner", "level"))
    spfx = _flat(pool["spfx"])
    live2, mask2, stamp2, hf2, sb, pfx0, lvl0, ok0 = C2._build_evict(
        C, Y, parent, level, pool["live"], _flat(pool["mask"]), _flat(pool["stamp"]), spfx, pool["hf"], slot, flags, new,
        tau)
    if target is None:
        nins, over, us = H, ZERO, list(units[:window])
    else:
        d = RS._monus(C, target, lvl0, max(len(target), Z.HW) + 1)
        nins, over = d[:Z.HW], RS._lt(C, const_bits(window, len(d)), d)
        us = CST._front(C, units, lvl0, window) if window else []
    carry = parent + tok + owner + level + live2 + mask2 + [ONE] + pfx0 + lvl0 + [ZERO] * (2 * Z.HW + 1)
    for unit in us:
        carry, _nxt = C2._build_unit(C, Y, carry, unit, sb, r, flags[:2], K, wh, nins)
    fin = CST._unpack(C2._carry_layout(Y), carry)
    mask3, stamp3, spfx3, hf3, clock3, okf, *back = C2._build_free(
        C, Y, fin["level"], fin["mask"], stamp2, spfx, fin["pfx"], hf2, pool["clock"], sb, flags, held, ret,
        derive=derive)
    if derive:
        ret, r_over = _fits(C, back[0], Z.HW)
        okf = [C.AND(okf[0], r_over.inv())]
    ok = C2._build_end(C, Y, [ok0[0], fin["ok"][0], okf[0]], fin["ev"], ev, wh, nins, flags, new, held, ret)[0]
    cut = C2._cut
    nxt = {"parent": cut(fin["parent"], Z.EW), "tok": cut(fin["tok"], Z.UW), "owner": cut(fin["owner"], Y.RW),
           "level": cut(fin["level"], Z.HW), "live": fin["live"], "mask": cut(mask3, Y.R), "stamp": cut(stamp3, Y.SW),
           "spfx": cut(spfx3, Z.EW), "hf": hf3, "clock": clock3, "fl": pool["fl"]}
    return (C.AND(ok, over.inv()), nxt) + ((ret,) if derive else ())


def _fits(C, v, w: int) -> tuple[list, object]:
    """v's low w bits, and 1 iff a bit past them is set."""
    return RS._pad(v, w), or_reduce(C, list(v[w:])) if len(v) > w else ZERO


def _free_ok(C, S, free, prow) -> object:
    """1 iff a state row's free count is its pool row's hashless free count plus its queued copies."""
    pool = _unpool(S, prow)
    queued_ = [C.AND(pool["live"][k], or_reduce(C, pool["mask"][k]).inv()) for k in range(S.CN * S.CD)]
    n = C2._count(C, queued_)
    w = max(len(free), len(pool["hf"]), len(n)) + 1
    return CST._eq(C, RS._pad(free, w), add(C, RS._pad(pool["hf"], w), RS._pad(n, w), width=w))


def _build_pool_step(C, S, prow, tp: dict, hit, ev, ret, tau, cert) -> tuple:
    """`pool_step` as a circuit on ReplayStep_v1's tapped wires `tp`: (ok, the next pool row)."""
    sh = shape(S)
    Y, Z, ZR, KL, KC = sh.Y, sh.Z, sh.ZR, sh.KL, sh.KC
    R, W, A, B, N, CA = S.R, S.W, S.A, S.B, S.CN, S.W + S.A
    LB, RCW, NW = B.bit_length() - 1, ZR.RC + 1, max(ZR.LW, ZR.TW) + 1
    pool = _unpool(S, prow)
    fl = pool["fl"]
    run, slots, upd, cands = tp["run"], tp["slots"], tp["update"], tp["cands"]
    kept, popped, avail, upto = tp["kept"], tp["popped"], tp["avail"], tp["upto"]
    recs = [tp["wait"][c] if c < W else tp["arr"][c - W] for c in range(CA)]
    bad = []

    def tokens_of(rid):
        m = [C.AND(cert["valid"][k], CST._eq(C, cert["id"][k], rid)) for k in range(S.K)]
        sel = CST._first(C, m)
        return or_reduce(C, m), [C2._pick(C, sel, [t[u] for t in cert["tok"]]) for u in range(S.CU)]

    def hit_of(sel):
        wh, wbad = _fits(C, C2._pick(C, sel, hit), Z.HW)
        e, ebad = _fits(C, C2._pick(C, sel, ev), Z.HW)
        K = CST._clamp(C, C2._pick(C, sel, [x["ntok"] for x in cands]), Z.LW)
        return wh, e, K, or_reduce(C, [wbad, ebad, RS._lt(C, wh, const_bits(KL, Z.HW + 1)).inv()])

    # per placed slot: its request's units, its compute's new blocks and target unit
    at = [[C.AND(cands[c]["admit"], CST._eq(C, cands[c]["at"], const_bits(x, RCW))) for c in range(CA)] for x in range(R)]
    admitted = [or_reduce(C, at[x]) for x in range(R)]
    units, new_x, target_x = [], [], []
    for x in range(R):
        s = slots[x]
        found, us = tokens_of(s["id"])
        bad.append(C.AND(s["valid"][0], found.inv()))
        units.append(us)
        nw = [a ^ b for a, b in zip(RS._sel(C, kept[x], RS._pad(tp["needs"][x], ZR.HW + 1)),
                                    RS._pad(C2._pick(C, at[x], [c["new"] for c in cands]), ZR.HW + 1))]
        nw, nbad = _fits(C, nw, Z.HW)
        new_x.append(nw)
        reach = RS._add(C, s["comp"], s["n"], NW)
        known = RS._monus(C, RS._add(C, s["P"], s["out"], NW), [C.AND(fl[x], kept[x])], NW)
        tg = RS._sel(C, or_reduce(C, s["n"]), RS._min(C, reach, known, NW)[LB:])
        target_x.append(tg)
        bad += [nbad, RS._lt(C, const_bits(S.CU, len(tg)), tg)]

    # the schedule's R positions
    before = [[RS._lt(C, avail[j], upto[i]) for i in range(R)] for j in range(R)]
    pos_c = [RS._add(C, const_bits(i, RCW), RS._popcount(C, [C.AND(popped[j], before[j][i]) for j in range(R)], RCW), RCW)
             for i in range(R)]
    pos_f = [RS._add(C, RS._sub(C, tp["cnt0"], const_bits(j + 1, RCW), RCW),
                     RS._popcount(C, [C.AND(kept[k], before[j][k].inv()) for k in range(R)], RCW), RCW) for j in range(R)]
    held_pre = [RS._cdiv(C, run[j]["comp"], LB, ZR.HW + 1) for j in range(R)]
    for t in range(R):
        wh, e, K, hbad = hit_of(at[t])
        bad.append(C.AND(admitted[t], hbad))
        ok, pool = _call(C, Y, pool, const_bits(t, Z.SL), slots[t]["id"], [ONE, admitted[t], ZERO], K=K, wh=wh, ev=e,
                         units=units[t], window=KL)
        bad.append(C.AND(admitted[t], ok.inv()))
        comp = [C.AND(kept[x], CST._eq(C, pos_c[x], const_bits(t, RCW))) for x in range(R)]
        comp[t] = comp[t] ^ admitted[t]
        frees = [C.AND(popped[j], CST._eq(C, pos_f[j], const_bits(t, RCW))) for j in range(R)]
        held, h_over = _fits(C, C2._pick(C, frees, held_pre), Z.HW)
        back, r_over = _fits(C, C2._pick(C, frees, ret), Z.HW)
        ok, pool = _call(C, Y, pool, C2._encode([a ^ b for a, b in zip(comp, frees)], Z.SL),
                         C2._pick(C, comp, [s["id"] for s in slots]), [ZERO, ZERO, or_reduce(C, frees)],
                         new=C2._pick(C, comp, new_x), tau=C2._pick(C, comp, tau), held=held, ret=back,
                         units=[C2._pick(C, comp, [us[u] for us in units]) for u in range(S.CU)], window=KC,
                         target=C2._pick(C, comp, target_x))
        bad += [h_over, r_over, ok.inv()]

    # the lookup of the candidate the waiting loop reaches and doesn't admit
    star = [C.AND(c["reach"], c["admit"].inv()) for c in cands]
    on = or_reduce(C, star)
    rid = C2._pick(C, star, [x["id"] for x in recs])
    found, us = tokens_of(rid)
    wh, e, K, hbad = hit_of(star)
    ok, _same = _call(C, Y, pool, const_bits(0, Z.SL), rid, [ONE, ZERO, ZERO], K=K, wh=wh, ev=e, units=us, window=KL)
    bad.append(C.AND(on, or_reduce(C, [found.inv(), hbad, ok.inv()])))

    # the update: the late cache_blocks (AS = 1), then a leaving request's free
    for p in range(R):
        x = upd[p]
        if S.AS:
            late = C.AND(kept[p], fl[p])
            tg = RS._sel(C, late, RS._sub(C, x["comp"], [x["sampled"]], ZR.LW)[LB:])
            ok, pool = _call(C, Y, pool, const_bits(p, Z.SL), slots[p]["id"], [ZERO, ZERO, ZERO], units=units[p],
                             window=LATE, target=tg)
            bad += [RS._lt(C, const_bits(S.CU, len(tg)), tg), ok.inv()]
        held, h_over = _fits(C, RS._sel(C, x["leave"], x["held"]), Z.HW)
        back, r_over = _fits(C, RS._sel(C, x["leave"], ret[p]), Z.HW)
        ok, pool = _call(C, Y, pool, const_bits(p, Z.SL), [ZERO] * S.IW, [ZERO, ZERO, x["leave"]], held=held, ret=back)
        bad += [h_over, r_over, ok.inv()]

    # the witness no call reads is 0
    bad += [C.AND(c["reach"].inv(), or_reduce(C, list(h) + list(v))) for c, h, v in zip(cands, hit, ev)]
    bad += [C.AND(or_reduce(C, [popped[p], upd[p]["leave"]]).inv(), or_reduce(C, ret[p])) for p in range(R)]
    bad += [C.AND(slots[p]["valid"][0].inv(), or_reduce(C, tau[p])) for p in range(R)]

    nxt = _next_row(C, S, pool, upd, bad)
    return or_reduce(C, bad).inv(), nxt


def _next_row(C, S, pool: dict, upd, bad: list) -> list:
    """The pool row after the step's calls: a slot leaving the running list holds nothing (else `bad`), the kept move
    to the front with their walk entries and at AS = 1 their ``fl`` (the row sampled), an invalid entry's fields 0."""
    Z, N, R = shape(S).Z, S.CN, S.R
    ND, root, RCW = N * S.CD, const_bits(N, Z.EW), shape(S).ZR.RC + 1
    cols = [[pool["mask"][k][p] for k in range(ND)] + [a ^ b for a, b in zip(pool["spfx"][p], root)] for p in range(R)]
    keep = [x["keep"] for x in upd]
    bad += [C.AND(keep[p].inv(), or_reduce(C, cols[p])) for p in range(R)]
    moved = RS._compact(C, [cols[p] + [upd[p]["sampled"]] for p in range(R)], keep, RCW)
    valid = [pool["live"][x * S.CD] for x in range(N)]
    nxt = [b for k in ("parent", "tok", "owner", "level") for x in range(N) for b in RS._sel(C, valid[x], pool[k][x])]
    nxt += pool["live"] + [moved[q][k] for k in range(ND) for q in range(R)] + _flat(pool["stamp"])
    nxt += [a ^ b for q in range(R) for a, b in zip(moved[q][ND:ND + Z.EW], root)] + list(pool["hf"]) + list(pool["clock"])
    nxt += [moved[q][ND + Z.EW] for q in range(R)] if S.AS else []
    return nxt


def _build(C, S, *ins):
    Sc, St = CS.chain(S), CS.step_statics(S)
    F = fields(S)
    split = {r: RC._split(x, F[r]) for x, r in zip(ins, row_names(S))}
    p, a, c, ci, co, w = (split[r] for r in ("pub", "adv", "cert", "cin", "cout", "wit"))
    served = [e for r in CS.srv_names(S) for e in split[r]["rows"]]
    states, win = w["state"], w["window"][0]
    load_ok = RC._build_load(C, Sc, CS._consts(RC.encode(RC.empty(Sc), Sc)), CS._consts(RC.encode_window(RC.Window(), Sc)),
                             RC._flat(c, RC.arrival_fields(Sc)), [e[0] for e in a["dec"]], a["count"][0], a["steps"][0],
                             states[0], win)[0]
    CA, R = S.W + S.A, S.R
    tok = c["tok"]
    cert = {"valid": [e[0] for e in c["a_valid"]], "id": c["a_id"],
            "tok": [[[x for t in tok[(k * S.CU + j) * S.B:(k * S.CU + j + 1) * S.B] for x in t] for j in range(S.CU)]
                    for k in range(S.K)]}
    pools = w["pool"]
    cache_ok = [RC._eq(C, pools[0], CS._consts(encode_pool(S, empty(S))))]
    cache_ok += [_free_ok(C, S, RC._split(states[u], RC.row_fields(Sc))["free"][0], pools[u]) for u in range(S.P + 1)]

    steps_ok, rows, mism = [], [], []
    for u in range(S.P):
        cur, nxt, arr, bits = states[u], states[u + 1], w["arr"][u], w["bits"][u]
        r, fa = RC._split(cur, RC.row_fields(Sc)), RC._split(arr, RC.front_fields(Sc))
        given = {"hit": w["hit"][u * CA:(u + 1) * CA], "evict": w["ev"][u * CA:(u + 1) * CA], "ret": w["ret"][u * R:(u + 1) * R]}
        elems = []
        for name, n, wd in RS._in_fields(St):
            if name in r:
                elems += r[name]
            elif name in fa:
                elems += fa[name]
            elif name == "bits":
                elems += [[b] for b in bits]
            elif name in given:
                elems += given[name]
            else:
                elems += [[ZERO] * wd for _ in range(n)]
        tp: dict = {}
        out = RS._split(RS._out_fields(St), RS._build(C, St, *elems, tap=tp))
        stepped = [b for name, _n, _w in RC.state_fields(Sc) for e in out[name] for b in e]
        ok, active = RC._build_advance(C, Sc, out["ok"][0], cur, win, stepped, arr, bits, out["taken"][0],
                                       out["used"][0], nxt)
        act = active[0]
        steps_ok.append(ok[0])
        mine = [C.AND(act, x) for name in CS.ROW_FIELDS for e in out[name] for x in e]
        if S.AS:
            mism += CS._build_map(C, S, CS._by_slot(S, mine), CS._by_slot(S, served[u]), w["map"][u * R:(u + 1) * R])
        else:
            rows += mine
        c_ok, c_next = _build_pool_step(C, S, pools[u], tp, given["hit"], given["evict"], given["ret"],
                                        w["tau"][u * R:(u + 1) * R], cert)
        cache_ok += [C.AND(act, c_ok.inv()).inv(), RC._eq(C, mux(C, act, c_next, pools[u]), pools[u + 1])]

    count = a["count"][0]
    en = [RC._lt(C, const_bits(k, len(count)), count) for k in range(S.D)]
    book = [e for name, _n, _w in CC.book_fields(S.D) for e in p[name]]
    code_ok, charged = CC.build_link(C, S.D, book, ci["carry"][0], co["carry"][0], en, [e[0] for e in a["dec"]])
    charge_ok = RC._lt(C, p["c"][0], charged).inv()
    rows_ok = RC._zero(C, mism) if S.AS else RC._eq(C, rows, [x for e in served for x in e])

    flags = [load_ok, and_reduce(C, steps_ok), code_ok, charge_ok, rows_ok, and_reduce(C, cache_ok)]
    return [flags + [ZERO] * (16 - len(flags))]


def _sig(S):
    bits = row_bits(S)
    return tuple((r, Array(bits[r], BIT)) for r in row_names(S)), Array(16, BIT)


_DOC = ("a slice of the compile unit, one window whole, with vLLM's block pool: CompileSlice_v1's load, steps, charge "
        "and served rows, and per step CacheStep_v2's calls on the pool rows in vLLM's order, each call's new, held, "
        "nins and slot ReplayStep_v1's own and its hits, evictable counts and returns the ones ReplayStep_v1 reads, "
        "each state row's free count the pool's (verdict bits load, steps, code, charge, rows, cache)")
CompileSlice = _part("CompileSlice", STATICS, _sig, _build, _DOC, version=2)


def statics(R: int, W: int, A: int, engine: dict, IW: int, Q: int, D: int, P: int, K: int, CN: int, CTW: int, CU: int,
            CD: int, SW: int, G: int | None = None) -> dict:
    """The slice's statics; G, the steps a ``srv`` row holds, is P unless given."""
    st = CS.statics(R, W, A, engine, IW, Q=Q, D=D, P=P, K=K, CN=CN, CTW=CTW, CU=CU)
    st = {**st, "G": P if G is None else G, "CD": CD, "SW": SW}
    CS.block(SimpleNamespace(**st))
    shape(SimpleNamespace(**st))
    return st


def unit(**st):
    return bind(CompileSlice, **st)


#: `compile_slice.DEMO_ENGINE` at two sequences of 16 tokens and six blocks (five past the null block), so the pool evicts
DEMO_ENGINE = {**CS.DEMO_ENGINE, "max_num_seqs": 2, "max_model_len": 16, "num_gpu_blocks": 6}


def demo() -> tuple[dict, CS.Slice]:
    """The demonstration class's statics and slice: five steps at (R 2, W 1, A 2) on `DEMO_ENGINE`, three arrivals,
    five decisions, a 5-entry pool of 4-bit tokens with two copies an entry and 6-bit stamps, 4 units a request.
    Request 1 shares request 0's first block: it is looked up at step 0, takes that hit at step 1 and leaves at once,
    freeing three blocks one of which slot 0 still holds (the shared-block free). Both computes of step 2 evict
    queued copies; request 2 is preempted at step 3 and readmitted at step 4 with a hit on its own first block."""
    st = statics(2, 1, 2, DEMO_ENGINE, 3, Q=3, D=5, P=5, K=3, CN=5, CTW=4, CU=4, CD=2, SW=6)
    tokens = ((8, 14, 9, 14, 10, 4, 3, 13, 9, 8, 11, 10, 13, 3, 2), (8, 14, 9, 14, 7, 8, 11, 12, 10, 11, 3, 10, 1),
              (10, 1, 13, 8, 6, 8))
    return st, CS.Slice(((0, 11, 4), (1, 12, 1), (2, 4, 2)), (0, 1, 1, 0, 0), 5, tokens)


def demo_async() -> tuple[dict, CS.Slice]:
    """The asynchronous demonstration class's: `demo`'s at AS = 1 (so ``wit`` has a ``map`` and the pool rows ``fl``).
    Request 2 shares request 0's first block: it is looked up at step 3 and admitted at step 4 with that hit, its
    compute evicting the oldest queued copy, and request 0 then leaves holding the shared block; slot 0's late
    `cache_blocks` caches a block at step 4."""
    st = statics(2, 1, 2, {**DEMO_ENGINE, "async_scheduling": True}, 3, Q=3, D=5, P=5, K=3, CN=5, CTW=4, CU=4, CD=2,
                 SW=6)
    tokens = ((6, 10, 14, 1, 8, 5, 12, 14, 11, 9, 2, 13, 8), (13, 5, 6, 1, 11),
              (6, 10, 14, 1, 10, 9, 14, 4, 5, 10, 9, 3, 4, 9))
    return st, CS.Slice(((0, 9, 4, 1), (1, 4, 1, 0), (2, 12, 2, 0)), (0, 1, 0, 1, 1), 5, tokens)


# --- the specification -------------------------------------------------------------------------------------------------------


def _cert_tokens(S, bits: Sequence[int]) -> dict[int, tuple[int, ...]]:
    """The ``cert`` row's tokens by request, its first valid slot's for each id (the circuit's selection)."""
    f = RC._split(bits, fields(S)["cert"])
    n, out = S.CU * S.B, {}
    for k in range(S.K):
        if f["a_valid"][k][0]:
            out.setdefault(CS._int(f["a_id"][k]), tuple(CS._int(t) for t in f["tok"][k * n:(k + 1) * n]))
    return out


def check(S, rows: dict) -> dict[str, bool | None]:
    """The circuit's flags on a unit's rows (`fields`' layouts): CompileSlice_v1's `compile_slice.check` for ``load``,
    ``steps``, ``code``, ``charge`` and ``rows``, with each active step's ReplayStep_v1 reading ``wit``'s hits,
    evictable counts and returns; ``cache`` the module's docstring's, by `pool_step` on each active step's pool row with
    ``cert``'s tokens and ``wit``'s stamps. For well-formed rows (`replay_chain.decode`'s); a pool row that isn't one
    pool's encoding is refused by ``cache``. A flag is None where a step's ReplayStep refuses (``steps`` is then
    False), and ``charge`` is None where the charge refuses its carry-in or its count."""
    Sc, St = CS.chain(S), CS.step_statics(S)
    F = fields(S)
    p, a, w = (RC._split(rows[r], F[r]) for r in ("pub", "adv", "wit"))
    states = [RC.decode(e, Sc) for e in w["state"]]
    win = RC.decode_window(w["window"][0], Sc)
    cert = RC._items(RC._unflatten(rows["cert"], RC.arrival_fields(Sc))[0], "a", RC.arrival_names(Sc))
    toks = _cert_tokens(S, rows["cert"])
    CA, R = S.W + S.A, S.R
    hit, ev, ret, tau = ([CS._int(e) for e in w[k]] for k in ("hit", "ev", "ret", "tau"))
    prow = [[int(b) for b in e] for e in w["pool"]]
    cache_ok = prow[0] == encode_pool(S, empty(S))
    pools: dict[int, Pool] = {}

    def pool_at(u: int) -> Pool:
        if u not in pools:
            pools[u] = decode_pool(S, prow[u])
        return pools[u]

    count = CS._int(a["count"][0])
    dec = [e[0] for e in a["dec"]]
    if count > S.D or any(dec[count:]):
        raise RC.MalformedRow(f"the advice's count {count} with a decision past it")
    loaded = RC.load(Sc, RC.empty(Sc), RC.Window(), cert, dec[:count], CS._int(a["steps"][0]))
    flags: dict[str, bool | None] = {"load": loaded == (states[0], win)}
    ok, length, carried = AE.reference_charge(S.D, CS._book(S, p), CC.carry_of([int(b) for b in rows["cin"]]),
                                              [int(k < count) for k in range(S.D)], dec)
    flags.update(code=bool(ok) and tuple(int(b) for b in rows["cout"]) == CC.encode(carried),
                 charge=length <= CS._int(p["c"][0]) if ok else None)
    names = RC.arrival_names(Sc)
    steps_ok, out_rows = True, []
    for u in range(S.P):
        cur, nxt = states[u], states[u + 1]
        arr = RC._items(RC._unflatten(w["arr"][u], RC.front_fields(Sc))[0], "arr", names)
        bits = [int(b) for b in w["bits"][u]]
        if not cur.steps:
            steps_ok &= nxt == cur
            cache_ok = cache_ok and prow[u + 1] == prow[u]
            out_rows.append(None)
            continue
        inp = RS.Inputs(tuple(RS.Req(**dict(zip(names, x))) for x in arr), tuple(bits), tuple(hit[u * CA:(u + 1) * CA]),
                        tuple(ev[u * CA:(u + 1) * CA]), tuple(ret[u * R:(u + 1) * R]))
        out = RS.reference(St, CS._state(Sc, cur), inp)
        if not out.ok:
            flags.update(steps=False, rows=None, cache=None)
            return {f: flags[f] for f in FLAGS}
        want = RC.advance(Sc, win, cur, True, CS._row(Sc, out.state), arr, bits, out.taken, out.used)
        steps_ok &= want is not None and want == nxt
        if cache_ok:
            try:
                got = pool_step(S, pool_at(u), CS._state(Sc, cur), inp, out, toks, tau[u * R:(u + 1) * R])
                cache_ok = encode_pool(S, got.pool) == prow[u + 1]
            except (Refused, ValueError):
                cache_ok = False
        out_rows.append(out.rows)
    if cache_ok:
        cache_ok = all(st.free == pool_at(u).pool.hf + queued(pool_at(u).pool) for u, st in enumerate(states))
    flags.update(steps=steps_ok, rows=CS._rows_ok(S, out_rows, CS._served(S, rows), w.get("map")), cache=cache_ok)
    return {f: flags[f] for f in FLAGS}


def scan(S, sl: CS.Slice, *, overstate: dict[int, int] | None = None, made_up: dict[int, dict[int, int]] | None = None,
         taus: dict[int, dict[int, int]] | None = None, rets: dict[int, dict[int, int]] | None = None) -> SimpleNamespace:
    """The window as a prover runs it from `sl`: ReplayLoad's rows, then each step by `replay_step.reference` on the state
    and the window row's arrivals and bits at the heads, and the pool through the step by `pool_step` from the empty
    pool, each hit, evictable count and return the pool's (run to a fixed point: a candidate's hit can be a block an
    earlier row of the step caches, and a victim's returns decide the victims). The prover's choices: `overstate`,
    request -> the blocks its frees claim beyond ⌈comp / B⌉; `made_up`, step u -> {slot: k}, a compute that evicts the
    k oldest queued copies with no eviction due; `taus` and `rets`, step u -> {slot: value} it puts in ``tau`` and
    ``ret`` in place of the pool's (ReplayStep_v1 reads the returns)."""
    Sc, St = CS.chain(S), CS.step_statics(S)
    arrivals = tuple(map(tuple, sl.arrivals))
    toks = {x[0]: tuple(t) for x, t in zip(arrivals, sl.tokens)}
    loaded = RC.load(Sc, RC.empty(Sc), RC.Window(), arrivals, sl.decisions, sl.steps)
    if loaded is None:
        raise ValueError("the load refuses the slice")
    cur, win = loaded
    CA, R, names = S.W + S.A, S.R, RC.arrival_names(Sc)
    pool = empty(S)
    states, fronts, steps, asks, pools, wit = [cur], [], [], [], [pool], []
    for u in range(S.P):
        arr, bits = RC.front(Sc, win, cur)
        fronts.append((arr, bits))
        if not cur.steps:
            states.append(cur)
            pools.append(pool)
            wit.append(((0,) * CA, (0,) * CA, (0,) * R, (0,) * R))
            continue
        st = CS._state(Sc, cur)
        reqs = tuple(RS.Req(**dict(zip(names, x))) for x in arr)
        h, e, r = [0] * CA, [0] * CA, list(returns(S, pool, st))
        for _round in range(4 * (CA + R) + 8):
            inp = RS.Inputs(reqs, tuple(bits), tuple(h), tuple(e), tuple(r))
            out = RS.reference(St, st, inp)
            if not out.ok:
                raise ValueError(f"step {u} refuses: the slice's class can't take it")
            refused = None
            try:
                got = pool_step(S, pool, st, inp, out, toks, derive=True, overstate=overstate,
                                made_up=(made_up or {}).get(u))
                dh, de, dr, order = got.hit, got.ev, got.ret, got.order
            except Refused as x:
                if x.partial is None:
                    raise ValueError(f"step {u}: {x}") from None
                refused, (dh, de, dr, order) = x, x.partial
            # the first value in call order that the calls derive otherwise: the calls before it ran on settled values
            k = next(((kind, i) for kind, i in order
                      if ((dh[i], de[i]) != (h[i], e[i]) if kind == "c" else dr[i] != r[i])), None)
            if k is not None:
                if k[0] == "c":
                    h[k[1]], e[k[1]] = dh[k[1]], de[k[1]]
                else:
                    r[k[1]] = dr[k[1]]
                continue
            if refused is not None:
                raise ValueError(f"step {u}: {refused}") from None
            if (got.hit, got.ev, got.ret) == tuple(map(tuple, (h, e, r))):
                break
            h, e, r = list(got.hit), list(got.ev), list(got.ret)
        else:
            raise ValueError(f"step {u}'s hits and returns don't settle")
        h, e, r = tuple(h), tuple(e), tuple(r)
        t = list(got.tau)
        for slot, v in (taus or {}).get(u, {}).items():
            t[slot] = v
        if u in (rets or {}):
            r = tuple(rets[u].get(j, x) for j, x in enumerate(r))
            out = RS.reference(St, st, RS.Inputs(reqs, tuple(bits), h, e, r))
            if not out.ok:
                raise ValueError(f"step {u} refuses the returns {r}")
        wit.append((h, e, r, tuple(t)))
        pool = got.pool
        pools.append(pool)
        stepped = CS._row(Sc, out.state)
        nxt = RC.Row(stepped.running, stepped.waiting, stepped.free, cur.qh + out.taken, cur.dh + out.used, cur.steps - 1)
        asks += bits[:out.used]
        steps.append(out.rows)
        states.append(nxt)
        cur = nxt
    return SimpleNamespace(states=states, window=win, fronts=fronts, steps=steps, asks=asks, arrivals=arrivals,
                           tokens=toks, pools=pools, wit=wit)


def case(S, sl: CS.Slice, book: dict, *, carry: AE.Carry | None = None, cout: AE.Carry | None = None,
         advice: Sequence[int] | None = None, served: Sequence | None = None, charge: int | None = None,
         maps: dict[int, Sequence[Sequence[int]]] | None = None, **choices) -> CS.Case:
    """A unit's rows for the prover's run of `sl` (`scan`'s, with `choices`) under the registered `book` (its JSON), and
    `check`'s verdict on them; `carry`, `cout`, `advice`, `served`, `charge` and `maps` as `compile_slice.case`'s."""
    Sc = CS.chain(S)
    cb, first = AE.charge_book(book, S.D)
    carry = first if carry is None else carry
    dec = tuple(sl.decisions if advice is None else advice)
    if len(dec) > S.D:
        raise ValueError(f"{len(dec)} decisions over D = {S.D}")
    padded = list(dec) + [0] * (S.D - len(dec))
    ok, length, nxt = AE.reference_charge(S.D, cb, carry, [1] * len(dec) + [0] * (S.D - len(dec)), padded)
    if not ok:
        raise ValueError("the charge refuses the advice from the carry-in")
    cout = nxt if cout is None else cout
    sc = scan(S, sl, **choices)
    served = sc.steps if served is None else served
    width = row_bits(S)
    rows = {"pub": CS._pack(S, "pub", {"c": [length if charge is None else charge], **CC.book_values(cb)}),
            "adv": CS._pack(S, "adv", {"dec": padded, "count": [len(dec)], "steps": [sl.steps]}),
            **CS.srv_rows(S, served),
            "cin": list(CC.encode(carry)), "cout": list(CC.encode(cout))}
    cert_bits = RC.load_arguments(Sc, RC.empty(Sc), RC.Window(), sc.arrivals, sl.decisions, len(sl.decisions), sl.steps,
                                  RC.encode(sc.states[0], Sc), RC.encode_window(sc.window, Sc))[2]
    n = S.CU * S.B
    held = [tuple(sc.tokens[x[0]])[:n] for x in sc.arrivals] + [()] * (S.K - len(sc.arrivals))
    cert_bits += [b for t in held for v in list(t) + [0] * (n - len(t)) for b in CS._bits(v, S.CTW)]
    rows["cert"] = cert_bits + [0] * (width["cert"] - len(cert_bits))
    HW = RS.widths(CS.step_statics(S)).HW
    wit = [b for st in sc.states for b in RC.encode(st, Sc)] + list(RC.encode_window(sc.window, Sc))
    wit += [b for arr, _b in sc.fronts for b in RC.front_bits(Sc, arr)]
    wit += [b for _a, bits in sc.fronts for b in bits]
    for k, wd in ((0, HW), (1, HW), (2, HW), (3, S.SW)):
        wit += [b for x in sc.wit for v in x[k] for b in CS._bits(v, wd)]
    wit += [b for pl in sc.pools for b in encode_pool(S, pl)]
    if S.AS:
        mine, theirs = (list(x) + [None] * (S.P - len(x)) for x in (sc.steps, served))
        wit += [b for u in range(S.P) for r in (maps or {}).get(u, CS.slot_map(S, mine[u], theirs[u])) for b in r]
    rows["wit"] = wit + [0] * (width["wit"] - len(wit))
    assert set(rows) == set(width) and all(len(rows[r]) == width[r] for r in width)
    flags = check(S, rows)
    return CS.Case(rows, CS.verdict(flags), flags, tuple(sc.steps), tuple(sc.asks), length, cout,
                   tuple(x[0] for x in sc.wit))


def evaluate(st: dict, cases: Sequence[CS.Case]) -> list[int]:
    """The unit's verdict word on each case, in one evaluation."""
    args = [[c.rows[r] for c in cases] for r in row_names(SimpleNamespace(**st))]
    out = evaluate_bits(unit(**st), *args).reshape(len(cases), -1).tolist()
    return [sum(int(b) << j for j, b in enumerate(o)) for o in out]


@lru_cache(maxsize=None)
def _counted(key: tuple) -> dict[str, int]:
    S = SimpleNamespace(**dict(key))
    C = Circuit(cse=True)
    ins = [C.input(name, sum(BL.elements(t))) for name, t in _sig(S)[0]]
    C.output("verdict", _build(C, S, *ins)[0])
    return C.counts()


def size(**st) -> dict[str, int]:
    """The unit's ANDs and its rows' widths at statics `st`."""
    S = SimpleNamespace(**st)
    return {"ands": _counted(tuple((k, st[k]) for k in STATICS))["ands"], **{f"{r}_bits": b for r, b in row_bits(S).items()}}
