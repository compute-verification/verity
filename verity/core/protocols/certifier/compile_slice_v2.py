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
``ret`` as its inputs. Nothing else in ``wit`` is read by it.

**The calls** (`pool_step`, the step's CacheStep_v2 calls on B_u, in the order vLLM makes them; `cache_pool_vectors`
holds the same calls to the replay's `_KV`). A pool slot is the running slot its request holds at the step's start.
- *The schedule's R positions.* Position t is the t-th event of the running loop: the compute of a kept slot i, or the
  free of a victim j. vLLM frees victims from the last slot down, each just before the allocation that needs it, so
  victim j is freed before kept slot i's compute iff free + (the returns of the slots after j) < (the needs up to i),
  both ReplayStep_v1's. At a position t at or past the running count, where the step admits, the row placed at slot t
  first takes its hit (a lookup that takes it, from its candidate's ``hit`` and ``ev``), then computes.
- *The lookup* of the first candidate the waiting loop reaches and doesn't admit, from its ``hit`` and ``ev``.
- *The update*, per placed slot in order: at AS = 1 the late `cache_blocks` of a slot whose output was in flight
  (vLLM caches a decode's block when its output is appended, one step late), then the free of a leaving request.
Every compute's ``new`` is ReplayStep_v1's: a kept slot's need, an admitted row's new blocks, 0 for a late cache. Every
free's ``held`` is ReplayStep_v1's ⌈comp / B⌉, a victim's before the step, a leaver's after its row, and its ``ret`` is
the ``ret`` ReplayStep_v1 reads. A compute caches the units from its walk entry's level up to ⌊t / B⌋ for t the
tokens `cache_blocks` caches: min(comp + n, P + out − fl) for a running row, min(start + n, P + out) for an admitted
one, comp − (the row samples) for a late cache. So none of new, held, nins or a take's slot is witness: REDTEAM phase
18's free that overstates ``held`` by k, adding k blocks that don't exist to the free count so the next k allocations
skip their evictions (and the leaving request that shares a block, which claims one block more and passes
ReplayStep_v1's ``ret`` ≤ held), is refused by the free's own check of ``ret``. After the update each next running
slot takes its placed slot's walk entry, mask bits and ``fl`` (the row sampled), and a slot that leaves the list must
hold nothing; an invalid entry's fields are zero, so a pool row has one encoding.

**cache** (bit 5) is: B_0 the empty pool; and at every active step the state row's free count equal to the pool's
hashless free count plus its queued copies, every call's ok, the class's bounds (a lookup's hit within KL = min(CN + 1,
CU) units, a compute's units within KC = ⌊(B − 1 + T) / B⌋ and a late cache's within one, every unit within CU, each
count within CacheStep_v2's widths, every request's tokens in ``cert``) and B_{u+1} the calls' pool (B_u on an idle step).

**The reference** is `check`; `case` builds the rows of a prover's run (`scan`), with the choices: ``overstate``, the
free of a request that holds ``held`` + k (REDTEAM phase 18's); ``made_up``, a compute that evicts the oldest queued
copies with no eviction due; ``taus`` and ``rets``, witness stamps and returns other than the pool's.

**Statics**: CompileSlice_v1's with G always given (P for one ``srv`` row), PC = 1, and CacheStep_v2's D (CD) and SW. A
window whose pool needs more than CN entries or CD copies of an entry, or whose clock passes SW bits, is refused: the
slice's class. Without prefix caching CompileSlice_v1 has no pool to wire: its returns are ⌈comp / B⌉ and nothing is
evicted.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ONE, ZERO, Circuit, add, and_reduce, const_bits, mux, or_reduce, sub
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


def _bitlen(v: int) -> int:
    return max(int(v).bit_length(), 1)


def pool_statics(S) -> SimpleNamespace:
    """CacheStep_v2's statics in the slice: N = CN, U = B, TW = CTW, RW = IW, L = CU·B, R, D = CD, NB, SW."""
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
    Y, F = pool_statics(S), RC._split(bits, pool_fields(S))
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
    """A call or a bound of the class refuses the step."""


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


def _evict_oldest(Y, pool: C2.Pool, k: int) -> tuple[C2.Pool, int]:
    """`pool` with its k oldest queued copies dropped, and the k-th's stamp: a made-up eviction."""
    tau = sorted(c.stamp for x in pool.table if x for c in x.copies if not c.mask)[k - 1]
    table = []
    for x in pool.table:
        kept = tuple(c for c in x.copies if c.mask or c.stamp > tau) if x else ()
        table.append(replace(x, copies=kept) if kept else None)
    return replace(pool, table=tuple(table)), tau


def pool_step(S, pool: Pool, st: RS.State, inp: RS.Inputs, out: RS.Out, tokens: dict[int, Sequence[int]],
              tau: Sequence[int] = (), *, derive: bool = False, overstate: dict[int, int] | None = None,
              made_up: dict[int, int] | None = None) -> SimpleNamespace:
    """The step's CacheStep_v2 calls on `pool` (the module's docstring), for ReplayStep_v1's state `st`, inputs `inp`
    and output `out` (an accepted step), the requests' certified `tokens` and the computes' stamps `tau` (by slot);
    raises `Refused`. Returns the next pool, the calls made (kind, op) and the hits, evictable counts, returns and
    stamps the calls read. With `derive` each lookup's hit and evictable count, each free's returns and each compute's
    stamp are the pool's (the prover's run); else `inp`'s and `tau`'s. A prover's choices: `overstate`, request ->
    the blocks its free claims beyond ⌈comp / B⌉ (and returns beyond its own); `made_up`, slot -> the queued copies its
    compute evicts with no eviction due."""
    sh = shape(S)
    Y, Z, KL, KC = sh.Y, sh.Z, sh.KL, sh.KC
    t = tap(S, st, inp)
    R, W, A, B, N = S.R, S.W, S.A, S.B, S.CN
    cur, fl = pool.pool, list(pool.fl) if S.AS else [0] * R
    hit, ev, ret = list(inp.hit), list(inp.evict), list(inp.ret)
    tau = list(tau) + [0] * (R - len(tau))
    overstate, made_up = overstate or {}, made_up or {}
    calls: list[tuple[str, C2.Op]] = []
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
        _need(hit[c] < KL and ev[c] < lim, f"candidate {c}'s hit {hit[c]} past KL = {KL} units, or its count past HW")
        K = min(x["ntok"], (1 << Z.LW) - 1)
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
            cur, stamp = _evict_oldest(Y, cur, made_up[slot])
            tau[slot] = stamp
            stamp = 0
        elif kind == "compute" and derive:
            stamp = tau[slot] = _threshold(cur, new)
        window_units = (us + ((0,) * B,) * window)[lvl:lvl + window]
        call(kind, C2.Op(slot=slot, r=rid, units=window_units, nins=nins, new=new, tau=stamp))

    def free(slot: int, p: int, rid: int, held: int) -> None:
        held += overstate.get(rid, 0)
        if derive:
            bit = 1 << slot
            has = [c for x in cur.table if x for c in x.copies if c.mask & bit]
            ret[p] = sum(c.mask == bit for c in has) + held - len(has)
        _need(held < lim and ret[p] < lim, f"slot {slot}'s free: {held} held, {ret[p]} returned past HW")
        call("free", C2.Op(slot=slot, free=True, held=held, ret=ret[p]))

    events = positions(S, t)
    for p in range(R):
        s = t.post[p]
        if s is not None and s["cand"] is not None:
            lookup(s["cand"]["c"], p, True)
        if p in events:
            kind, i = events[p]
            r = t.run[i]
            if kind == "free":
                free(i, i, r["id"], RS.cdiv(r["comp"], B))
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
    keep = [p for p in range(R) if t.update[p] is not None and t.update[p]["keep"]]
    gone = [p for p in range(R) if p not in keep]
    _need(not any(c.mask >> p & 1 for x in cur.table if x for c in x.copies for p in gone)
          and all(cur.spfx[p] == N for p in gone), "a slot that leaves the running list still holds blocks")

    def moved(m: int) -> int:
        return sum((m >> p & 1) << q for q, p in enumerate(keep))

    table = tuple(replace(x, copies=tuple(replace(c, mask=moved(c.mask)) for c in x.copies)) if x else None
                  for x in cur.table)
    spfx = tuple(cur.spfx[p] for p in keep) + (N,) * (R - len(keep))
    nfl = tuple(int(t.update[p]["sampled"]) for p in keep) + (0,) * (R - len(keep)) if S.AS else ()
    return SimpleNamespace(pool=Pool(replace(cur, table=table, spfx=spfx), nfl), calls=calls, hit=tuple(hit),
                           ev=tuple(ev), ret=tuple(ret), tau=tuple(tau))
