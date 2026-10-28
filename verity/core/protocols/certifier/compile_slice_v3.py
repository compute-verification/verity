"""The compile slice that computes its schedule: `CompileSlice_v2` (`compile_slice_v2`) with every row it checked
computed instead, held to the registered rows it reads, so the advice is the unit's only free input.

    CompileSlice_v3{R, W, A, M, T, L, B, H, CH, WM, FULL, PC, NB, IW, AS, EW, Q, D, P, K, CN, CTW, CU, CD, SW, DA}
        (adv, ing, req, srv, cout) -> verdict[16]

**Why.** `NetTiming.Guarantees.SampledLeakAccepted` bounds what an accepted audit carries by c + …, c the window
circuit's input gates that are neither ingress nor registered, and `Accepted` constrains only the egress rows, so a
check's verdict pins nothing: every witness input of a check-form unit counts toward c. CompileSlice_v2 takes its state
rows, each step's arrivals, bits, hits, counts, returns and stamps, its pool rows and the map as witness, so its free
inputs are thousands of bits. Here each of them is a wire the unit computes, and the free inputs are the advice's
ℓ_w = DA + PW bits. A unit's public output is at most 16 bits, and here state crosses units by registered reads, so
the served rows and the state the window leaves are registered rows the unit reads and holds equal to the ones it
computes.

**The inputs**, each a field list (elements least significant bit first):
- ``adv`` (*advice*, the only free input, unpadded since a padding gate would be a free input too): the window's DA
  decision bits ``dec`` and its executed step count ``steps`` (PW = bitlen(P) bits). DA, the window's advice length in
  decisions, is a static: ℓ_w is registered and public, so each length is its own binding and the advice input is
  exactly ℓ_w gates wide. core refuses ℓ_w past the cap, so the count is the check. ``steps`` is advice because how
  many engine steps a certifier window holds is the scheduler's timing, which no public input fixes.
- ``ing`` (*ingress*, unpadded): the window's K arrival slots' ``a_valid``, ``a_id`` and prompt length ``a_P``, which
  the layout maps to the ingress gates of each request's arrival.
- ``req`` (*registered*): each slot's served output length ``a_O`` (and at AS = 1 ``a_cap``, O being its max_tokens),
  the request Program's end, and ``tok``, each slot's first CU·B token ids (its prompt, then its outputs), the request
  Program's token at each position (vLLM's V3).
- ``srv`` (*registered*): the served rows, each step's R rows (valid, request, kind, start, n) in slot order, as
  `compile_slice.rows_bits` packs them, zero on an idle step: CompileSlice_v2's ``srv`` row at G = P, the row
  `served_rows` registers and its consumers read. At AS = 1 they stay in slot order: CompileSlice_v2's witnessed
  ``map`` from batch positions to slots would be a free input.
- ``cout`` (*registered*): the state row S_{P+1}, the window row W_1 and the pool row B_{P+1} the window leaves.
A registered row is padded to whole words (`compile_slice.WORD`), as CompileSlice_v2's are; its padding must be zero.
`CLASSES` names each input's class, and `free_width` counts the inputs of no other class.

**The output**: ``verdict``, 16 bits, bit 0 ``load`` (with ``req``'s padding), bit 1 ``steps``, bit 2 ``cache``, bit 3
``served`` (``srv`` is the computed rows), bit 4 ``carry`` (``cout`` is the computed last rows); `ACCEPT` = 0x1F.

**The circuit.** The load from the empty state and window (`replay_chain._load_rows`) computes S_1 and W_1. Each step
u computes its arrivals and decision bits as the window row's at S_u's heads (`replay_chain._front`), runs
ReplayStep_v1 on them and S_u, and makes its CacheStep_v2 calls on B_u in vLLM's order, each where ReplayStep_v1
reads what it derives (`replay_step._build`'s ``derive``):
- *the returns* of each running slot, were it a victim: the copies whose lowest-numbered holder it is, plus its
  hashless blocks (⌈comp / B⌉ less the copies it holds). The victims are a suffix freed from the last slot down, so
  when slot j is freed every slot after it already is;
- *the running loop's events*, `compile_slice_v2`'s positions: each kept slot's compute and each victim's free;
- *each candidate*, in the waiting loop's order: a lookup from ROOT that reads, not writes, its hit (the longest
  cached prefix the lookup may take) and evictable count, then, masked by its admission, its take (its slot's bit on
  each hit unit's copy 0, its walk entry the hit's end) and its compute. A candidate's hit can be a block an earlier
  candidate's compute caches or evicts, so each lookup reads the pool after the candidates before it;
- *the update*, per placed slot: at AS = 1 the late `cache_blocks`, then a leaving request's free, whose returns are
  the free's own.
Each compute's threshold stamp is the e-th oldest queued stamp, found bit by bit from the top (`_tau`). The next
state is the step's (heads past what it took and read) or S_u on an idle step, and the next pool row the calls' (as
`compile_slice_v2._next_row`) or B_u.

**What became redundant.** Every equality against a witness row (the load's two rows, each step's arrivals, bits and
next state, the pool rows and the map), the ``hit``, ``ev``, ``ret`` and ``tau`` checks in the calls, and
ChargeLink_v1 with ``pub``, ``cin`` and ``cout``'s carry: the charge is ℓ_w. What stays in the verdict is what refuses
advice: the load's bounds, ReplayStep_v1's ok and each step's head bounds, the class's bounds (a request's tokens in
``req``, a lookup's hit within KL units, units within CU, counts within the widths, the calls' oks) and each state
row's free count equal to its pool row's hashless and queued blocks; and the two registered rows the unit computes,
``srv`` and ``cout``, about an AND a bit.

**Cost** (`size`). The demonstration units are 76,554 ANDs (`demo`) and 83,287 (`demo_async`), against
CompileSlice_v2's 83,092 and 91,785: dropping the witness checks saves more than the computed derivations add there,
and holding ``srv`` and ``cout`` to the computed rows is 543 and 572 of them. On corpus-smallkv's first seven steps
(CN 24, CU 25, 16-token blocks of 17-bit tokens, CD 1, SW 12, DA 7) the computing part is 12,477,467 ANDs against
CompileSlice_v2's 7,984,682, and is its reference on the honest advice (its served rows the fixture's), one step fewer
and each decision flipped (with ``srv`` and ``cout`` outputs). There a step makes W + A = 3
read-only lookups of KL = 25 units (168,368 ANDs each) and as many computes with a KC = 16-unit window (227,862 each,
masked by admission), where CompileSlice_v2's witness named the R admitted rows, and R + W + A threshold searches
(1,440 each).

**The window** starts from the empty engine and pool, as CompileSlice_v1's and v2's slices do; a 2a-class window, whose
requests start and end inside it, is one. A chained window would read the last window's ``cout`` as a registered
input, with the tokens of requests it carries.

**The reference** is `reference`: `compile_slice_v2.scan`'s run of the slice, its served rows and last rows, or None
where the load, a step or the pool refuses; the unit accepts exactly the advice it compiles with those rows registered.

**Statics**: CompileSlice_v2's with G = P (one ``srv`` row) and DA ≤ D.
"""
from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ONE, ZERO, Circuit, add, and_reduce, const_bits, mux, or_reduce, sub
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array
from verity.core.protocols.certifier import cache_step as CST, cache_step_v2 as C2, compile_slice as CS
from verity.core.protocols.certifier import compile_slice_v2 as V2, replay_chain as RC, replay_step as RS
from verity.core.protocols.certifier.head_check import BIT, _part

STATICS = CS.STATICS + ("CD", "SW", "DA")
FLAGS = ("load", "steps", "cache", "served", "carry")
ACCEPT = (1 << len(FLAGS)) - 1
ROWS = ("adv", "ing", "req", "srv", "cout")
#: each input row's class: the advice is free; ingress and registered gates aren't counted in the window's c
CLASSES = {"adv": "advice", "ing": "ingress", "req": "registered", "srv": "registered", "cout": "registered"}
#: the rows padded to whole words (`compile_slice.WORD`)
PADDED = tuple(r for r in ROWS if CLASSES[r] == "registered")


def shape(S) -> SimpleNamespace:
    """`compile_slice_v2.shape`, with DA within D."""
    if not 0 <= S.DA <= S.D:
        raise ValueError(f"statics {dict(S.__dict__)}: the advice's DA decisions are at most D")
    return V2.shape(S)


def fields(S) -> dict[str, list[tuple[str, int, int]]]:
    shape(S)
    Sc = CS.chain(S)
    Z = RC.widths(Sc)
    return {"adv": [("dec", S.DA, 1), ("steps", 1, Z.PW)],
            "ing": [("a_valid", S.K, 1), ("a_id", S.K, S.IW), ("a_P", S.K, Z.LW)],
            "req": [("a_O", S.K, Z.LW)] + ([("a_cap", S.K, 1)] if S.AS else []) + [("tok", S.K * S.CU * S.B, S.CTW)],
            "srv": [("rows", S.P, CS.step_row_bits(S))],
            "cout": [("state", 1, RC.width(Sc)), ("window", 1, RC.window_width(Sc)), ("pool", 1, V2.pool_width(S))]}


def data_bits(S) -> dict[str, int]:
    """Each input row's bits before its padding."""
    return {r: RC._bits_of(fs) for r, fs in fields(S).items()}


def row_bits(S) -> dict[str, int]:
    return {r: CS._words_of(b) if r in PADDED else b for r, b in data_bits(S).items()}


def free_width(S, widths: dict[str, int] | None = None) -> int:
    """The unit's input bits of no class but advice: those not ingress or registered (`CLASSES`); `widths`, each input
    row's bits (default the signature's)."""
    widths = row_bits(S) if widths is None else widths
    if set(widths) != set(CLASSES):
        raise ValueError(f"inputs {sorted(widths)} aren't the classified rows {sorted(CLASSES)}")
    return sum(b for r, b in widths.items() if CLASSES[r] not in ("ingress", "registered"))


# --- the circuit ------------------------------------------------------------------------------------------------------------


def _fits(C, v, w: int) -> tuple[list, object]:
    return V2._fits(C, v, w)


class _Pool:
    """The step's CacheStep_v2 calls on its pool row, made where ReplayStep_v1 reads what they derive (the module's
    docstring); `bad` collects what refuses the step."""

    def __init__(self, C, S, prow, cert):
        sh = shape(S)
        self.C, self.S, self.Y, self.Z, self.ZR, self.KL, self.KC = C, S, sh.Y, sh.Z, sh.ZR, sh.KL, sh.KC
        self.pool = V2._unpool(S, prow)
        self.fl = self.pool["fl"]
        self.cert = cert
        self.bad: list = []
        self.pending = None
        self.run_units: list = []
        self.under: list = []

    def tokens_of(self, rid):
        C, S, cert = self.C, self.S, self.cert
        m = [C.AND(cert["valid"][k], CST._eq(C, cert["id"][k], rid)) for k in range(S.K)]
        sel = CST._first(C, m)
        return or_reduce(C, m), [C2._pick(C, sel, [t[u] for t in cert["tok"]]) for u in range(S.CU)]

    def tau(self, new):
        """The compute's threshold stamp: the smallest t with at least e = new − hf queued stamps ≤ t (0 when e = 0,
        all ones when fewer than e are queued, which the eviction refuses)."""
        C, Y, Z, pool = self.C, self.Y, self.Z, self.pool
        ND = Y.N * Y.D
        W = max(Z.HW, Z.BW) + 1
        d, short = sub(C, RS._pad(new, W), RS._pad(pool["hf"], W), W)
        e = [C.AND(x, short.inv()) for x in d[:Z.HW]]
        queued = [C.AND(pool["live"][k], or_reduce(C, pool["mask"][k]).inv()) for k in range(ND)]
        lt, eq, t = [ZERO] * ND, [ONE] * ND, [ZERO] * Y.SW
        for b in reversed(range(Y.SW)):
            s = [pool["stamp"][k][b] for k in range(ND)]
            zb = [C.AND(eq[k], s[k].inv()) for k in range(ND)]
            le = [C.AND(queued[k], lt[k] ^ zb[k]) for k in range(ND)]
            t[b] = RS._lt(C, C2._count(C, le), e)
            lt = [lt[k] ^ C.AND(t[b], zb[k]) for k in range(ND)]
            eq = [C.AND(eq[k], (s[k] ^ t[b]).inv()) for k in range(ND)]
        return t

    def returns(self, tap):
        """Each running slot's returns were it freed after the slots past it: the copies it is the lowest holder of,
        plus ⌈comp / B⌉ less the copies it holds."""
        C, S, Y, ZR = self.C, self.S, self.Y, self.ZR
        R, ND, LB = S.R, Y.N * Y.D, S.B.bit_length() - 1
        low = []
        for m in self.pool["mask"]:
            seen, row = ZERO, []
            for s in range(R):
                row.append(C.AND(m[s], seen.inv()))
                seen = seen ^ row[s]
            low.append(row)
        out = []
        for j in range(R):
            lh = C2._count(C, [low[k][j] for k in range(ND)])
            nh = C2._count(C, [self.pool["mask"][k][j] for k in range(ND)])
            held = RS._cdiv(C, tap["run"][j]["comp"], LB, ZR.HW + 1)
            w = max(ZR.HW + 1, len(nh)) + 1
            hl, under = sub(C, RS._pad(held, w), RS._pad(nh, w), w)
            back, over = _fits(C, add(C, RS._pad(lh, w), hl, width=w + 1), ZR.HW)
            self.under.append(or_reduce(C, [under, over]))
            out.append(back)
        return out

    def running(self, tap):
        """The running loop's events at their positions: a kept slot's compute, or a victim's free."""
        C, S, Y, Z, ZR = self.C, self.S, self.Y, self.Z, self.ZR
        R, LB, RCW, NW = S.R, S.B.bit_length() - 1, ZR.RC + 1, max(ZR.LW, ZR.TW) + 1
        run, kept, popped, avail, upto = tap["run"], tap["kept"], tap["popped"], tap["avail"], tap["upto"]
        bad = self.bad
        bad += [C.AND(popped[j], self.under[j]) for j in range(R)]
        new_x, target_x = [], []
        for x in range(R):
            r = run[x]
            found, us = self.tokens_of(r["id"])
            bad.append(C.AND(r["valid"], found.inv()))
            self.run_units.append(us)
            nw, nbad = _fits(C, RS._sel(C, kept[x], RS._pad(tap["needs"][x], ZR.HW + 1)), Z.HW)
            new_x.append(nw)
            n = tap["run_n"][x]
            reach = RS._add(C, r["comp"], n, NW)
            known = RS._monus(C, RS._add(C, r["P"], r["out"], NW), [C.AND(self.fl[x], kept[x])], NW)
            tg = RS._sel(C, or_reduce(C, n), RS._min(C, reach, known, NW)[LB:])
            target_x.append(tg)
            bad += [nbad, RS._lt(C, const_bits(S.CU, len(tg)), tg)]
        before = [[RS._lt(C, avail[j], upto[i]) for i in range(R)] for j in range(R)]
        pos_c = [RS._add(C, const_bits(i, RCW), RS._popcount(C, [C.AND(popped[j], before[j][i]) for j in range(R)], RCW),
                         RCW) for i in range(R)]
        pos_f = [RS._add(C, RS._sub(C, tap["cnt0"], const_bits(j + 1, RCW), RCW),
                         RS._popcount(C, [C.AND(kept[k], before[j][k].inv()) for k in range(R)], RCW), RCW)
                 for j in range(R)]
        held_pre = [RS._cdiv(C, run[j]["comp"], LB, ZR.HW + 1) for j in range(R)]
        for t in range(R):
            comp = [C.AND(kept[x], CST._eq(C, pos_c[x], const_bits(t, RCW))) for x in range(R)]
            frees = [C.AND(popped[j], CST._eq(C, pos_f[j], const_bits(t, RCW))) for j in range(R)]
            held, h_over = _fits(C, C2._pick(C, frees, held_pre), Z.HW)
            new = C2._pick(C, comp, new_x)
            ok, self.pool, _back = V2._call(
                C, Y, self.pool, C2._encode([a ^ b for a, b in zip(comp, frees)], Z.SL),
                C2._pick(C, comp, [r["id"] for r in run]), [ZERO, ZERO, or_reduce(C, frees)], new=new, tau=self.tau(new),
                held=held, units=[C2._pick(C, comp, [us[u] for us in self.run_units]) for u in range(S.CU)],
                window=self.KC, target=C2._pick(C, comp, target_x), derive=True)
            bad += [h_over, ok.inv()]

    def lookup(self, units, K):
        """A lookup from ROOT that writes nothing: its hit (the longest cached prefix within the cap K), its evictable
        count, each unit's hit entry (one-hot, zero past the hit), the hit's end entry and whether it fills the window."""
        C, Y, Z = self.C, self.Y, self.Z
        N, D, EW = Y.N, Y.D, Z.EW
        pool = self.pool
        valid = [pool["live"][x * D] for x in range(N)]
        pfx, h, ev, hs, flags = const_bits(N, EW), ONE, [ZERO] * Z.HW, [], []
        for j in range(self.KL):
            hits = [C.AND(valid[x], and_reduce(C, [(a ^ b).inv() for a, b in zip(pool["parent"][x] + pool["tok"][x],
                                                                                   pfx + list(units[j]))]))
                    for x in range(N)]
            cap = (j + 1) * Y.U
            look = RS._lt(C, const_bits(cap, max(len(K), cap.bit_length()) + 1), K)
            h = and_reduce(C, [h, or_reduce(C, hits), look])
            unheld = C2._xor(C.AND(hits[x], or_reduce(C, pool["mask"][x * D]).inv()) for x in range(N))
            ev = add(C, ev, [C.AND(h, unheld)], width=Z.HW)
            hs.append([C.AND(h, x) for x in hits])
            pfx = mux(C, h, C2._encode(hits, EW), pfx)
            flags.append(h)
        wh, _over = _fits(C, C2._count(C, flags), Z.HW)
        return wh, ev, hs, pfx, flags[-1]

    def take(self, hs, sb, admit, end):
        """The admitted candidate's take: its slot's bit on each hit unit's copy 0 and its walk entry the hit's end; the
        slot must hold nothing."""
        C, Y, Z = self.C, self.Y, self.Z
        N, D, R = Y.N, Y.D, Y.R
        pool = self.pool
        root = const_bits(N, Z.EW)
        tx = [C.AND(admit, C2._xor(h[x] for h in hs)) for x in range(N)]
        mask = [list(m) for m in pool["mask"]]
        for x in range(N):
            mask[x * D] = [m ^ C.AND(tx[x], s) for m, s in zip(mask[x * D], sb)]
        self.bad.append(C.AND(admit, CST._eq(C, C2._pick(C, sb, pool["spfx"]), root).inv()))
        spfx = [mux(C, C.AND(admit, sb[s]), end, pool["spfx"][s]) for s in range(R)]
        self.pool = {**pool, "mask": mask, "spfx": spfx}

    def hit(self, c, rec, ntok, tap):
        C, Z, ZR = self.C, self.Z, self.ZR
        self.settle(tap)
        found, us = self.tokens_of(rec["id"])
        wh, ev, hs, end, full = self.lookup(us, CST._clamp(C, ntok, Z.LW))
        self.pending = (c, rec, found, us, hs, end, full)
        return _fits(C, wh, ZR.HW)[0], _fits(C, ev, ZR.HW)[0]

    def settle(self, tap):
        """The last candidate looked up, once the waiting loop has decided it: its take and compute where admitted."""
        if self.pending is None:
            return
        C, S, Y, Z, ZR = self.C, self.S, self.Y, self.Z, self.ZR
        LB, RCW, NW = S.B.bit_length() - 1, ZR.RC + 1, max(ZR.LW, ZR.TW) + 1
        c, rec, found, us, hs, end, full = self.pending
        self.pending = None
        x = tap["cands"][c]
        admit = x["admit"]
        self.bad.append(C.AND(x["reach"], OR(C, found.inv(), full)))
        sb = [CST._eq(C, x["at"], const_bits(s, RCW)) for s in range(S.R)]
        self.take(hs, sb, admit, end)
        tg = RS._sel(C, C.AND(admit, or_reduce(C, x["n"])), RS._min(C, RS._add(C, x["hb"], x["n"], NW), x["ntok"], NW)[LB:])
        new, nbad = _fits(C, RS._sel(C, admit, x["new"]), Z.HW)
        ok, self.pool = V2._call(C, Y, self.pool, C2._encode(sb, Z.SL), rec["id"], [ZERO, ZERO, ZERO], new=new,
                                 tau=self.tau(new), units=us, window=self.KC, target=tg)
        self.bad.append(C.AND(admit, OR(C, OR(C, nbad, ok.inv()), RS._lt(C, const_bits(S.CU, len(tg)), tg))))

    def candidates(self, tap):
        self.settle(tap)

    def leave(self, p, tap):
        """Placed slot p's update: at AS = 1 its late cache_blocks, then a leaving request's free; its returns."""
        C, S, Y, Z, ZR = self.C, self.S, self.Y, self.Z, self.ZR
        LB = S.B.bit_length() - 1
        x, s = tap["update"][p], tap["slots"][p]
        if S.AS:
            late = C.AND(tap["kept"][p], self.fl[p])
            tg = RS._sel(C, late, RS._sub(C, x["comp"], [x["sampled"]], ZR.LW)[LB:])
            ok, self.pool = V2._call(C, Y, self.pool, const_bits(p, Z.SL), s["id"], [ZERO, ZERO, ZERO],
                                     units=self.run_units[p], window=V2.LATE, target=tg)
            self.bad += [RS._lt(C, const_bits(S.CU, len(tg)), tg), ok.inv()]
        held, h_over = _fits(C, RS._sel(C, x["leave"], x["held"]), Z.HW)
        ok, self.pool, back = V2._call(C, Y, self.pool, const_bits(p, Z.SL), [ZERO] * S.IW, [ZERO, ZERO, x["leave"]],
                                       held=held, derive=True)
        back, r_over = _fits(C, back, ZR.HW)
        self.bad += [h_over, ok.inv(), r_over]
        return back

    def finish(self, tap) -> tuple[object, list]:
        """(1 iff the step's calls and bounds hold, the next pool row)."""
        nxt = V2._next_row(self.C, self.S, self.pool, tap["update"], self.bad)
        return or_reduce(self.C, self.bad).inv(), nxt


def OR(C, a, b):
    return a ^ b ^ C.AND(a, b)


def _build(C, S, adv, ing, req, srv_in, cout_in):
    Sc, St = CS.chain(S), CS.step_statics(S)
    Z = RC.widths(Sc)
    F, n_req = fields(S), data_bits(S)["req"]
    a, g, w = RC._split(adv, F["adv"]), RC._split(ing, F["ing"]), RC._split(req, F["req"])
    arrivals = {**g, "a_O": w["a_O"], **({"a_cap": w["a_cap"]} if S.AS else {})}
    dec = [e[0] for e in a["dec"]] + [ZERO] * (S.D - S.DA)
    checks, cur, win = RC._load_rows(C, Sc, CS._consts(RC.encode(RC.empty(Sc), Sc)),
                                     CS._consts(RC.encode_window(RC.Window(), Sc)),
                                     RC._flat(arrivals, RC.arrival_fields(Sc)), dec, const_bits(S.DA, Z.DW),
                                     a["steps"][0])
    tok = w["tok"]
    cert = {"valid": [e[0] for e in g["a_valid"]], "id": g["a_id"],
            "tok": [[[x for t in tok[(k * S.CU + j) * S.B:(k * S.CU + j + 1) * S.B] for x in t] for j in range(S.CU)]
                    for k in range(S.K)]}
    prow = CS._consts(V2.encode_pool(S, V2.empty(S)))
    wf = RC._split(win, RC.window_fields(Sc))
    Wd, A = RC.width(Sc), S.A
    cache_ok, steps_ok, srv = [], [], []
    for u in range(S.P):
        r = RC._split(cur, RC.row_fields(Sc))
        cache_ok.append(V2._free_ok(C, S, r["free"][0], prow))
        qh, dh, steps = r["qh"][0], r["dh"][0], r["steps"][0]
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
        dv = _Pool(C, S, prow, cert)
        out = RS._split(RS._out_fields(St), RS._build(C, St, *elems, tap=tp, derive=dv))
        stepped = [b for name, _n, _w in RC.state_fields(Sc) for e in out[name] for b in e]
        act = or_reduce(C, steps)
        taken, used = out["taken"][0], out["used"][0]
        fv = [e[0] for e in fr["arr_valid"]]
        took = [C.AND(RC._lt(C, const_bits(k, Z.AC), taken), fv[k].inv()).inv() for k in range(A)]
        m = max(Z.DW, Z.AC) + 1
        dh_new = add(C, RC._pad(dh, m), RC._pad(used, m), width=m)
        step_ok = and_reduce(C, [out["ok"][0][0], RC._lt(C, const_bits(A, Z.AC + 1), taken).inv(), *took,
                                 RC._lt(C, const_bits(A, Z.AC + 1), used).inv(), RC._lt(C, wf["left"][0], dh_new).inv()])
        steps_ok.append(C.AND(act, step_ok.inv()).inv())
        stepped_row = RC._pad(stepped + add(C, qh, RC._pad(taken, Z.QW), width=Z.QW) + dh_new[:Z.DW]
                              + sub(C, steps, const_bits(1, Z.PW), Z.PW)[0], Wd)
        srv += [C.AND(act, x) for name in CS.ROW_FIELDS for e in out[name] for x in e]
        c_ok, c_next = dv.finish(tp)
        cache_ok.append(C.AND(act, c_ok.inv()).inv())
        cur, prow = mux(C, act, stepped_row, cur), mux(C, act, c_next, prow)
    r = RC._split(cur, RC.row_fields(Sc))
    cache_ok.append(V2._free_ok(C, S, r["free"][0], prow))
    flags = [and_reduce(C, checks + [RC._zero(C, req[n_req:])]), and_reduce(C, steps_ok), and_reduce(C, cache_ok),
             RC._eq(C, srv, srv_in), RC._eq(C, list(cur) + list(win) + list(prow), cout_in)]
    return flags + [ZERO] * (16 - len(flags))


def _sig(S):
    bits = row_bits(S)
    return tuple((r, Array(bits[r], BIT)) for r in ROWS), Array(16, BIT)


_DOC = ("the compile slice computing its schedule, one window whole from the empty engine, with vLLM's block pool: from "
        "the advice (its DA decisions and step count), the arrivals' ingress fields and their registered output lengths "
        "and tokens, the load, ReplayStep_v1 a step and CacheStep_v2's calls in vLLM's order, each hit, evictable count, "
        "return and stamp computed where ReplayStep_v1 reads it, the served rows and last rows held to the registered "
        "srv and cout (verdict bits load, steps, cache, served, carry)")
CompileSlice = _part("CompileSlice", STATICS, _sig, _build, _DOC, version=3)


def statics(R: int, W: int, A: int, engine: dict, IW: int, Q: int, D: int, P: int, K: int, CN: int, CTW: int, CU: int,
            CD: int, SW: int, DA: int) -> dict:
    st = CS.statics(R, W, A, engine, IW, Q=Q, D=D, P=P, K=K, CN=CN, CTW=CTW, CU=CU)
    st = {**st, "CD": CD, "SW": SW, "DA": DA}
    shape(SimpleNamespace(**st))
    return st


def unit(**st):
    return bind(CompileSlice, **st)


def _v2(st: dict) -> dict:
    """compile_slice_v2's statics of the same class (G = P), for its `scan`."""
    return {**{k: v for k, v in st.items() if k != "DA"}, "G": st["P"]}


def demo() -> tuple[dict, CS.Slice]:
    """`compile_slice_v2.demo`'s class and slice, its five decisions the advice (DA = 5)."""
    st, sl = V2.demo()
    return {**{k: v for k, v in st.items() if k != "G"}, "DA": len(sl.decisions)}, sl


def demo_async() -> tuple[dict, CS.Slice]:
    """`compile_slice_v2.demo_async`'s class and slice, its five decisions the advice (DA = 5)."""
    st, sl = V2.demo_async()
    return {**{k: v for k, v in st.items() if k != "G"}, "DA": len(sl.decisions)}, sl


# --- the specification and arguments -----------------------------------------------------------------------------------------


def _padded(S, r: str, bits: Sequence[int]) -> list[int]:
    n = data_bits(S)[r]
    if len(bits) != n:
        raise ValueError(f"row {r} holds {n} bits, not {len(bits)}")
    return list(bits) + [0] * (row_bits(S)[r] - n)


def arguments(S, sl: CS.Slice, srv: Sequence[int] | None = None, cout: Sequence[int] | None = None) -> dict[str, list[int]]:
    """The unit's inputs for slice `sl` (bits each, `fields`' layouts, the registered rows zero-padded): ``srv`` and
    ``cout`` as given, by default `reference`'s rows (zero where it refuses the slice)."""
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
    if srv is None or cout is None:
        ref = reference(S, sl)
        nb = data_bits(S)
        srv = (ref.srv if ref else [0] * nb["srv"]) if srv is None else srv
        cout = (ref.cout if ref else [0] * nb["cout"]) if cout is None else cout
    req = (col("O", Z.LW) + (col("cap", 1) if S.AS else [])
           + [b for t in toks for v in list(t) + [0] * (n - len(t)) for b in CS._bits(v, S.CTW)])
    return {"adv": [int(d) for d in sl.decisions] + CS._bits(sl.steps, Z.PW),
            "ing": [int(x is not None) for x in arr] + col("id", S.IW) + col("P", Z.LW),
            "req": _padded(S, "req", req), "srv": _padded(S, "srv", srv), "cout": _padded(S, "cout", cout)}


def reference(S, sl: CS.Slice) -> SimpleNamespace | None:
    """The registered rows the unit accepts `sl`'s advice with, by `compile_slice_v2.scan`'s run (``srv`` and ``cout``'s
    bits, unpadded), or None where the load, a step, a step's heads (`replay_chain.advance`: it takes more arrivals than
    are queued, or reads more than A decisions or past the window's) or the pool refuses the slice."""
    Sc = CS.chain(S)
    V = SimpleNamespace(**_v2(S.__dict__))
    try:
        sc = V2.scan(V, sl)
    except (ValueError, V2.Refused):
        return None
    for u, (cur, nxt) in enumerate(zip(sc.states, sc.states[1:])):
        if cur.steps:
            arr, _bits = sc.fronts[u]
            if nxt.qh - cur.qh > len(arr) or nxt.dh - cur.dh > S.A or nxt.dh > len(sc.window.decisions):
                return None
    steps = list(sc.steps) + [None] * (S.P - len(sc.steps))
    srv = [b for rows in steps for b in CS.rows_bits(S, rows)]
    cout = list(RC.encode(sc.states[-1], Sc)) + list(RC.encode_window(sc.window, Sc)) + list(V2.encode_pool(S, sc.pools[-1]))
    return SimpleNamespace(srv=srv, cout=cout)


def run(st: dict, args: Sequence[dict[str, list[int]]]) -> list[int]:
    """The unit's verdict word on each of `args` (`arguments`' form), in one evaluation."""
    out = evaluate_bits(unit(**st), *[[a[r] for a in args] for r in ROWS]).reshape(len(args), -1).tolist()
    return [sum(int(b) << j for j, b in enumerate(o)) for o in out]


def evaluate(st: dict, slices: Sequence[CS.Slice]) -> list[int]:
    """The unit's verdict word on each slice, its registered rows `reference`'s."""
    S = SimpleNamespace(**st)
    return run(st, [arguments(S, sl) for sl in slices])


@lru_cache(maxsize=None)
def _counted(key: tuple) -> dict[str, int]:
    S = SimpleNamespace(**dict(key))
    C = Circuit(cse=True)
    ins = [C.input(name, sum(BL.elements(t))) for name, t in _sig(S)[0]]
    C.output("verdict", _build(C, S, *ins))
    return C.counts()


def size(**st) -> dict[str, int]:
    """The unit's ANDs, its inputs' widths (padded) and its free (advice) width at statics `st`."""
    S = SimpleNamespace(**st)
    return {"ands": _counted(tuple((k, st[k]) for k in STATICS))["ands"], **{f"{r}_bits": b for r, b in row_bits(S).items()},
            "free_bits": free_width(S)}
