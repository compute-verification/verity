"""dither-a in the statement's order (RT4-INT-12; CA's decision, thread 1792447454.296229): the arms
`dither-a-{hbk,hbr}-zy-{aom,aomn,ab,abn}[-pt]-so<D>[-ov|-ov1|-ovp]`, labelled by their unit as every dither-a arm
(`dither_a.form_of`, RT4-INT-39): form=beside, receipts beside the served GEMM, since neither unit commits z (hbr's write
none; hbk's write z and y into scratch, which nothing commits).

A step's A, scales and z/y leaves go under its round's root before its round's coin; its units read that coin and run
after it, beside a later step's forward; their receipts go under a later round.  Step s, in order:

    forward s        exactlin's quantizer (the call's A slot), the call's A table (`h1c_a_call`, form 3, on NA_MULTI A
                     streams above the GEMM's priority, joined at the last call and at every qkv call under piecewise
                     graphs), exactlin's GEMM with the z/y records (dither-a's -zy) and the served y's records
                     (`dithera_ycommit`).  No seeds and no units: nothing in the forward reads a coin.
    between steps    under -ab, every call's rows >= m cleared and the step's A tables in one h1c_build (h1agate.py's
                     `batch`: a per-step A commitment, a scheme change CA has not ruled in, so a separate arm); the step's
                     A slots, scales, A roots, live rows and step into ring slot s mod D; the admit tables (under -pt on
                     their own stream), the z/y leaf tables and the served y's tables (RT4-INT-11's run-time compare);
                     round s over a-step, admit-step, z-step, y-step and receipt-step, whose leaf is the receipt-table
                     roots of step s - D + 1 (each table's domain names its own step; zeros before the window's first
                     D - 1 steps); round s's root sent (`coins.RoundPipe`), after which alone the challenger opens coin s.
    units of step s  at the end of step s + D - 2's between-steps (D = 2: step s's own, just after its root is sent), once
                     coin s is back (the host waits for it there: coin_wait_ms): on the units stream, after the ring slot
                     is written, the seeds of every call in one h1r launch (version 0: `h1tile.live_salt` of coin s and
                     round s's root, root(A_j) from the ring, root(B_j)), each call's kh_one (hbk; hbr its receipts-only
                     build) on the ring slot, under -aomn and -abn each call's noise on NA_MULTI streams of its own
                     beside the units, then the commit (no-write and fault from the step's flags) and the receipt trees.
                     They run beside step s + D - 1's forward, and round s + D - 1 waits for them.

So D is the ring's depth: the A slots of D steps live at once (the forward's own, and D - 1 steps' whose units are pending
or running), and the units of a step wait D - 2 steps for their coin.  In process (``coins: in-process (not certified)``)
a coin is back once its root is sent; off the node a round trip of R ms needs D - 2 >= R / step.

`check` holds a capture to dither_a's checks with its own round s for the A, admit, z and y leaves (the receipt-step leaf
there an earlier step's, read as recorded) and round s + D - 1 for its receipts (dither_a.check's `r_round`), every
receipt against h1r v1 under the salt of coin s as the challenger opened it, and to the order itself: coin s is round s's
coin opened after round s's root (`live_reason`), the roots sent are the rounds captured, and the receipts' round was
sent after coin s came back and after the units were launched (the pipe's clock).  Beyond dither-a's gate, control,
fault, fault_y and fault_rr (here: round s + D - 1 recorded without step s's receipt roots), two faults of the order
itself, each refused: "fault_rearly" (step s's units under the salt of coin s - 1, a coin the prover held before round
s's root: refused on every receipt) and "fault_zlate" (step s's z/y leaves withheld from round s, 0xA5 in their place,
and committed under round s + 1, after coin s: refused on round s's z-step and y-step leaves).

-ov (`-so<D>-ov`, ring R = 2; `-ov1`, R = 1; `-ovp`, R = 2 with the engine serving on a stream above the trees' and the
units' priority) moves round s and step s's units off the steps' path, every value committed
under round s as without it.  Step s + 1's forward reads nothing of round s, so the engine's step s + 1 starts once forward
s (its served y among it) is done: on the serving stream after forward s, before forward s + 1 can write them, step s's
z/y and served-y records, admit rows, step and A roots are copied into round-ring slot s mod R (and its A slots into the
units' ring, as without -ov); a sender thread builds round s's tables and tree on the tree streams from that slot, beside
forward s + 1, and sends round s's root; a dispatcher thread dispatches step s's units as soon as coin s is back.  The
host holds the rings: step s's copies wait until round s - R is sent and step s - D's units are dispatched (and, on the
GPU, done).  `check` holds each capture to what forward s committed, cloned on the serving stream at its end, against
round s as built from the ring once forward s + 1 had started (a captured step's round waits for it; the GPU's events
give how long the build ran beside it), and "fault_ring" overwrites a bit of step s's ring copy of call 0's z on the
serving stream as forward s + 1 starts, before round s's tables: refused on round s's z-step leaf and root alone.

    import dither_so as SO
    SO.patch(t=T, h1c="libh1c.so", h1o="libh1o.so", unit="hbr", h1u="h1u_kr.so", ao="m", pt=True, depth=4)
    llm = LLM(model=int8_exact_checkpoint, ...)
    win = DO.hook(llm); win.captures = SO.CAPTURES
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import queue
import threading
import time
import traceback

import torch
import triton

import dither_a as DA
import dither_fused as DF
import dither_one as DO
from dither_fused import CO, Call, unwords, words
from hb_side import auto_j

LIB_ENV = DA.LIB_ENV
CG = None
ORDER = "statement (RT4-INT-12): A, scales, z/y under round s before coin s; units after coin s; receipts under round s + D - 1"
OV_ORDER = ("-ov: round s built beside forward s + 1 from ring copies taken after forward s, its root sent before coin s; "
            "step s's units dispatched once coin s is back")
# the faults of the order last: fault_zlate moves round s + 1's z/y leaves, so the step after it is no capture's
CAPTURES = ("gate", "control", "fault", "fault_y", "fault_rr", "fault_rearly", "fault_zlate")
CAPTURES_OV = ("gate", "control", "fault", "fault_y", "fault_rr", "fault_ring", "fault_rearly", "fault_zlate")
# the A tables: "m" per call in the forward on NA_MULTI streams; "b" once a step between steps (not ruled in); "n" after
# either: each call's noise on NA_MULTI streams beside the units
AO = {"m": "per call, in the forward", "mn": "per call, in the forward; noise on its own streams",
      "b": "once a step (h1agate's batch; not ruled in by CA)", "bn": "once a step (not ruled in by CA); noise on its own streams"}
NA = DA.NA_MULTI


class SoWindow(DO.Window):
    """dither_one's window with the rounds in the statement's order (the module docstring): its round coins through a
    `coins.RoundPipe`, its slots grouped a K so the ring copies them in a launch a K, the units of step s dispatched at
    the end of step s + D - 2's `after`."""
    group_slots = True
    depth = 2
    a_once = False
    overlap = 0     # -ov: the round ring's depth (`Ov`), 0 without

    def __init__(self, *args, **kwargs):
        if self.depth < 2:
            raise ValueError(f"dither-a-so's ring depth is at least 2, not {self.depth}")
        super().__init__(*args, **kwargs)
        self.pipe = CO.RoundPipe(self.coins, time.monotonic)
        # the gate's generation runs past its last capture until that step's receipts are in a round
        self.gate_extra = self.depth + 8 + (1 if self.overlap else 0)
        self.cap_of: dict[int, str] = {}
        self.m_of: dict[int, int] = {}
        self.stashed: set[int] = set()
        self.zlate: tuple | None = None
        self.coin_waits: list[float] = []
        self.rtt_from = 1
        # `stats` holds each opening received since its last call to `live_reason` (the pipe's reader does too, off the
        # node; in process nothing else does)
        self.order_from, self.order_why = 0, None
        self.ev_w = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        # the engine's own host time a step (from one `after`'s end to the next `before`), and `before`'s
        self.t_b0 = self.t_after_end = 0.0

    def host_ms(self) -> dict:
        t_end = time.perf_counter()
        x = {"before_ms": 1e3 * (self.t_before - self.t_b0)}
        if self.t_after_end:
            x["engine_ms"] = 1e3 * (self.t_b0 - self.t_after_end)
        self.t_after_end = t_end
        return x

    def _a_step(self) -> None:
        if not self.a_once:
            super()._a_step()

    def poisoned(self) -> list:
        # not the plan's receipts and seeds, nor the units' stage: an earlier step's units are writing them now (a
        # captured step's units poison theirs on the units stream, `SoSide.dispatch`)
        out = self.slot + self.scale + self.z + list(self.ydump or []) + [self.pre["mid"].level, self.pre["a"].level]
        if DA.SIDE:
            sd = DA.SIDE[0]
            out += sd.zy + sd.ysv + [sd.zyset.level, sd.ysset.level, sd.yuset.level]
        return out

    def before(self, m: int) -> None:
        self.t_b0 = time.perf_counter()
        super().before(m)
        self.m_of[self.s] = m
        if self.cap is not None:
            self.cap_of[self.s] = self.cap
        if self.overlap and DA.SIDE:
            DA.SIDE[0].ov.forward_starts(self.s)

    def cap_sync(self) -> None:
        # under -ov an earlier round's trees and units run now on streams of their own; the poison and the step's
        # inputs are on the serving stream
        if self.overlap:
            torch.cuda.current_stream().synchronize()
        else:
            super().cap_sync()

    def arm(self, m: int, names: tuple | None = None) -> None:
        self.drain()
        super().arm(m, names)

    def drain(self) -> None:
        """Under -ov: every round handed to the sender sent, every step's units dispatched, the device idle."""
        if self.overlap and DA.SIDE:
            DA.SIDE[0].ov.drain()

    def after(self) -> None:
        if self.overlap:
            return self._after_ov()
        sd: SoSide = DA.SIDE[0]
        d = self.depth - 1
        main = torch.cuda.current_stream()
        self.phase("forward_wait")
        main.synchronize()
        t_fwd = time.perf_counter()
        s, m, cap, pre, ev, kinds = self.s, self.m, self.cap, self.pre, self.ev, self.round_kinds
        if self.a_once:
            self.phase("a_tables")
            if not self.flags & DO.F_NW_TILES:
                for x in self.slot_groups.values():
                    x[:, m:].zero_()
                self.scale_all[:, m:].zero_()
            self.c.build(pre["a"], self.stepv[0:1])
        self.phase("stash")
        sd.stash(s)
        self.stashed.add(s)
        ads = self.tree_streams[1] if self.tree_streams is not None else None
        self.phase("admit_tables")
        ev[0].record()
        if ads is not None:
            ads.wait_event(ev[0])
            with torch.cuda.stream(ads):
                self.c.build(pre["admit"], self.stepv[0:1])
                ev[7].record(ads)
        else:
            self.c.build(pre["admit"], self.stepv[0:1])
        self.phase("z_tables")
        self.c.build(pre["z"], self.stepv[0:1])
        for f in self.before_mid:
            f()
        self.phase("receipt_leaf")
        o = pre["mid"].level_off[kinds.index("receipt-step")]
        lv = pre["mid"].level[o:o + 8 * self.L]
        j = s - d
        p = sd.dispatched.pop(j, None)
        waited = p is not None
        if p is None:
            lv.zero_()
        else:
            self.ev_w[0].record()
            main.wait_event(sd.udone[p])
            self.ev_w[1].record()
            if self.cap_of.get(j) == "fault_rr":
                lv.view(torch.uint8).fill_(DF.POISON)
            else:
                lv.copy_(sd.rroots[p])
        if ads is not None:
            main.wait_event(ev[7])
        self.c.build(pre["mid"], self.stepv[0:1])
        rl = pre["round"].level
        zi, yi = kinds.index("z-step"), kinds.index("y-step")
        if cap == "fault_zlate":
            self.zlate = (rl[8 * zi:8 * zi + 8].clone(), rl[8 * yi:8 * yi + 8].clone())
            for i in (zi, yi):
                rl[8 * i:8 * i + 8].view(torch.uint8).fill_(DF.POISON)
        elif self.zlate is not None:
            rl[8 * zi:8 * zi + 8].copy_(self.zlate[0])
            rl[8 * yi:8 * yi + 8].copy_(self.zlate[1])
            self.zlate = None
        self.c.build(pre["round"], self.stepv[0:1])
        self.phase("root_read")
        root = unwords(pre["round"].root(0))
        t_root = time.perf_counter()
        self.phase("record_host")
        i = self.pipe.send(root)
        if i != s + 1:
            raise RuntimeError(f"step {s}'s round went out as round {i}, not {s + 1}")
        t_sent = time.perf_counter()
        bad = sd.ybad.nonzero().flatten().tolist()
        if bad:
            sd.y_refused.append({"s": s, "capture": cap, "calls": len(bad)})
        prev = self.cap_of.get(s - 1)
        late = self.cap_of.pop(j, None) if j >= 0 else None
        if (cap in self.caps) or (late in self.caps) or (prev == "fault_zlate" and prev in self.caps):
            leaves = [unwords(pre["round"].leaf(0, k)) for k in range(len(kinds))]
            if cap in self.caps:
                self.caps[cap].update(
                    root=root, index=i, a=[x.clone() for x in self.slot], sa=[x.clone() for x in self.scale],
                    z=[x.clone() for x in self.z], y=[x[:m].clone() for x in self.ydump] if self.ydump else None,
                    a_roots=[unwords(pre["mid"].leaf(0, w)) for w in range(self.L)],
                    admit_roots=[unwords(pre["mid"].leaf(1, w)) for w in range(self.L)],
                    z_roots=[unwords(pre["mid"].leaf(2, w)) for w in range(self.L)], leaves=leaves,
                    b_roots=[unwords(self.bset.root(w)) for w in range(self.L)],
                    zy=[x.clone() for x in sd.zy], ysv=[x.clone() for x in sd.ysv], y_root_bad=bad,
                    y_roots=[unwords(pre["mid"].leaf(yi, w)) for w in range(self.L)],
                    yu_roots=[unwords(sd.yuset.root(w)) for w in range(self.L)])
            if late in self.caps:
                self.caps[late].update(late_s=s, late_index=i, late_leaves=leaves, late_root=root)
            if prev == "fault_zlate" and prev in self.caps:
                self.caps[prev].update(next_leaves=leaves)
        units_wait = self.ev_w[0].elapsed_time(self.ev_w[1]) if waited else None
        self.phase("coin_wait")
        ju = s + 1 - d
        t_wait = t_coin = t_disp = time.perf_counter()
        if ju in self.stashed:
            self.stashed.discard(ju)
            opening = self.pipe.opening(ju + 1)
            t_coin = time.perf_counter()
            self.coin_waits.append(1e3 * (t_coin - t_wait))
            name = self.cap_of.get(ju)
            if name == "fault_rearly":
                held = self.pipe.opening(ju)
                salt = self.HT.live_salt(self.coins.commitment, held.coin, held.index, self.pipe.roots[ju])
            else:
                salt = self.HT.live_salt(self.coins.commitment, opening.coin, opening.index, self.pipe.roots[ju + 1])
            g = self.caps.get(name) if name is not None else None
            if g is not None:
                g.update(opening=opening, salt_used=salt, coin_got_at=self.pipe.got_at.get(ju + 1),
                         dispatched_at=time.monotonic())
            self.phase("units_dispatch")
            sd.dispatch(ju, salt, self.m_of.pop(ju), g)
            t_disp = time.perf_counter()
        t_end = time.perf_counter()
        self.phase(None)
        ph = {"m": m, "forward_ms": 1e3 * (t_fwd - self.t_before), "trees_ms": 1e3 * (t_root - t_fwd),
              "record_host_ms": 1e3 * (t_sent - t_root), "captures_ms": 1e3 * (t_wait - t_sent),
              "coin_wait_ms": 1e3 * (t_coin - t_wait),
              "dispatch_ms": 1e3 * (t_disp - t_coin), "between_ms": 1e3 * (t_end - t_fwd)}
        if units_wait is not None:
            ph["units_wait_ms"] = units_wait
        self.phases.append({**ph, **self.host_ms()})
        self.round_log.append({"s": s, "m": m, "root": root.hex()[:32]})
        self.s += 1
        self.steps += 1
        if self.s + 1 >= DF.ROUNDS:
            raise RuntimeError(f"the window's {DF.ROUNDS} rounds are spent")

    def _after_ov(self) -> None:
        """-ov's step end: step s's ring copies on the serving stream (once the rings' last readers are past them), a
        capture's view of what forward s committed, and round s handed to the sender."""
        sd: SoSide = DA.SIDE[0]
        ov: Ov = sd.ov
        main = torch.cuda.current_stream()
        s, m, cap, D = self.s, self.m, self.cap, self.depth
        ov.ev["fwd1"][s % Ov.EV].record(main)
        self.phase("forward_wait")
        main.synchronize()
        t_fwd = time.perf_counter()
        if self.a_once:
            self.phase("a_tables")
            if not self.flags & DO.F_NW_TILES:
                for x in self.slot_groups.values():
                    x[:, m:].zero_()
                self.scale_all[:, m:].zero_()
            self.c.build(self.pre["a"], self.stepv[0:1])
        self.phase("ring_wait")
        # ring slot s mod D's last reader is step s - D's units, round slot s mod R's round s - R
        ov.wait(lambda: ov.dispatched_upto >= s - D and ov.sent_upto >= s - ov.R)
        t_ring = time.perf_counter()
        self.phase("stash")
        if s >= D:
            main.wait_event(sd.udone[s % D])
        if cap is not None:
            ov.poison(s % ov.R)
        sd.stash(s)
        ov.copy(s)
        if cap is not None:
            ov.watch.add(s)
            br = self.bset.roots.cpu()
            self.caps[cap].update(
                a=[x.clone() for x in self.slot], sa=[x.clone() for x in self.scale], z=[x.clone() for x in self.z],
                y=[x[:m].clone() for x in self.ydump] if self.ydump else None,
                zy=list(ov.zy.clone().split(ov.zl)), ysv=list(ov.ysv.clone().split(ov.yl)),
                b_roots=[unwords(br[8 * w:8 * w + 8]) for w in range(self.L)])
        ov.put(s)
        t_end = time.perf_counter()
        self.phase(None)
        self.phases.append({"s": s, "m": m, "forward_ms": 1e3 * (t_fwd - self.t_before),
                            "ring_wait_ms": 1e3 * (t_ring - t_fwd), "stash_ms": 1e3 * (t_end - t_ring),
                            "between_ms": 1e3 * (t_end - t_fwd), **self.host_ms()})
        self.s += 1
        self.steps += 1
        if self.s + 1 >= DF.ROUNDS:
            raise RuntimeError(f"the window's {DF.ROUNDS} rounds are spent")

    def stats(self) -> dict:
        ov = DA.SIDE[0].ov if self.overlap and DA.SIDE else None
        if ov is not None:
            ov.drain()
            for ph in self.phases:
                s = ph.pop("s", None)
                if s is not None:
                    ph.update(ov.ph_round.pop(s, {}))
                    ph.update(ov.ph_units.pop(s, {}))
        doc = super().stats()
        pipe = self.pipe
        n = len(pipe.openings)
        rtt = sorted(1e3 * (pipe.got_at[i] - pipe.sent_at[i]) for i in range(self.rtt_from, n)
                     if i in pipe.got_at and i in pipe.sent_at)
        self.rtt_from = n
        waits, self.coin_waits = sorted(self.coin_waits), []
        for i in range(self.order_from, n):
            if self.order_why is None:
                self.order_why = CO.Ch.live_reason(self.coins.commitment, pipe.openings[i], pipe.label, i, pipe.roots[i])
        self.order_from = n
        sent = list(pipe.roots)

        def q(xs: list, f: float):
            return round(xs[min(len(xs) - 1, int(f * len(xs)))], 3) if xs else None
        doc["salt"] = "round (the step's own coin)"
        doc["so"] = {"order": ORDER, "depth": self.depth, "units_lag_steps": self.depth - 2,
                     "receipts_round_lag": self.depth - 1, "a_tables": "once a step (not ruled in)" if self.a_once
                     else "per call, in the forward", "coins": self.coins.label, "coins_record": self.coins.record(),
                     "coin_rtt_ms": {"n": len(rtt), "p50": q(rtt, 0.5), "p90": q(rtt, 0.9), "max": q(rtt, 1.0)},
                     "coin_wait_ms": {"n": len(waits), "p50": q(waits, 0.5), "p90": q(waits, 0.9), "max": q(waits, 1.0)},
                     "order_reason": self.order_why, "coins_back": n, "rounds_sent": len(sent),
                     # the challenger's log holds these round roots in this order (its published session's record)
                     "roots_sha512": hashlib.sha512(b"".join(sent)).hexdigest()}
        if ov is not None:
            doc["so"]["overlap"] = {"order": OV_ORDER, "ring": ov.R, "rounds_built": ov.sent_upto + 1,
                                    "units_dispatched": ov.dispatched_upto + 1,
                                    "serving_stream_priority": torch.cuda.current_stream().priority}
        return doc


class SoSide(DA.Side):
    """dither-a's serving-stream GEMM state with the forward in the statement's order (no seeds, no units) and the
    units of a step dispatched after its coin (`dispatch`) on a ring of the window's depth (`stash`)."""

    def __init__(self, win: SoWindow, unit: str, h1u: str, ao: str, pt: bool):
        if unit not in DA.KONE or ao not in AO:
            raise ValueError(f"dither-a-so's unit: one of {DA.KONE}, not {unit!r}; its A tables: one of {tuple(AO)}, not {ao!r}")
        super().__init__(win, unit, h1u, "zy", "", pt)
        dev, L, D = win.dev, win.L, win.depth
        self.so_ao, self.a_once, self.noise = ao, ao.startswith("b"), ao.endswith("n")
        if not self.a_once:
            self.astreams = [torch.cuda.Stream(device=dev, priority=-1) for _ in range(NA)]
            self.a_join = [torch.cuda.Event() for _ in range(NA)]
            self.a_used: set[int] = set()
        if self.noise:
            self.nstreams = [torch.cuda.Stream(device=dev) for _ in range(NA)]
            self.ws_ring = [self.ws] + [torch.empty_like(self.ws) for _ in range(NA - 1)]
            self.n_done = [torch.cuda.Event() for _ in range(NA)]
            self.u_done = [torch.cuda.Event() for _ in range(NA)]
            self.seeded = torch.cuda.Event()
        self.ring_groups = [{k: torch.empty_like(x) for k, x in win.slot_groups.items()} for _ in range(D)]
        self.ring_slot = []
        for p in range(D):
            at = dict.fromkeys(self.ring_groups[p], 0)
            views = []
            for _, k in win.nk:
                views.append(self.ring_groups[p][k][at[k]])
                at[k] += 1
            self.ring_slot.append(views)
        self.ring_scale = [torch.empty_like(win.scale_all) for _ in range(D)]
        self.ring_live = [torch.empty_like(win.live) for _ in range(D)]
        self.ring_aroots = [torch.empty(8 * L, dtype=torch.int64, device=dev) for _ in range(D)]
        self.ring_S = [torch.empty_like(win.S) for _ in range(D)]
        self.ring_s1 = [torch.empty_like(win.S) for _ in range(D)]
        self.ring_stepv = [torch.empty_like(win.stepv) for _ in range(D)]
        self.ring_base = [torch.empty_like(win.step_base) for _ in range(D)]
        self.ring_salt = [torch.empty(8, dtype=torch.int64, device=dev) for _ in range(D)]
        self.salt_host = [torch.empty(8, dtype=torch.int64).pin_memory() for _ in range(D)]
        self.rroots = [torch.zeros(8 * L, dtype=torch.int64, device=dev) for _ in range(D)]
        self.ready = [torch.cuda.Event() for _ in range(D)]
        self.udone = [torch.cuda.Event() for _ in range(D)]
        self.tables = [win.r.plan([Call(self.ring_slot[p][w], win.wt[w], win.offs[w], self.ring_aroots[p].data_ptr() + 64 * w,
                                        self.ring_live[p], root_b=win.bset.roots.data_ptr() + 64 * w) for w in range(L)],
                                  seeds=win.plan.seeds, out=win.plan.out) for p in range(D)]
        self.a_off = win.pre["mid"].level_off[0]
        self.dispatched: dict[int, int] = {}
        self.unit_steps = 0
        self.ov = Ov(win, self, win.overlap) if win.overlap else None

    def call(self, w: int, x: torch.Tensor, wt: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor) -> None:
        win = self.win
        M, k = x.shape
        main = torch.cuda.current_stream()
        xq = torch.empty((M, k), dtype=torch.int8, device=x.device)
        s_a = torch.empty(M, dtype=torch.float32, device=x.device)
        DA.exactlin_quant_rows_dithera[(M,)](x, xq, s_a, win.slot[w], win.scale[w], win.S, k, x.stride(0), xq.stride(0),
                                             BLOCK_K=triton.next_power_of_2(k), num_warps=DA.EL.quant_warps(k),
                                             **DA.EL.OPTS)
        rc = 0
        if not self.a_once:
            self.fork.record(main)
            i = w % NA
            ast = self.astreams[i]
            ast.wait_event(self.fork)
            rc = win.a_call(w, win.slot[w], win.scale[w], ast.cuda_stream)
            self.a_used.add(i)
        self.gemm(w, xq, s_a, wt, s_w, y)
        self.ycommit(w, y)
        if rc:
            raise RuntimeError(f"dither-a-so call {w} (M={M}): its A table's launch returned {rc}")
        if not self.a_once and self.a_used and (w == win.L - 1 or (w % 4 == 0 and DA._piecewise())):
            for i in sorted(self.a_used):
                self.a_join[i].record(self.astreams[i])
                main.wait_event(self.a_join[i])
            self.a_used.clear()
            self.joins += 1
        self.calls += 1

    def stash(self, s: int) -> None:
        """On the serving stream after step s's forward: its A slots, scales, A roots, live rows, step (with and without
        its flags) and base into ring slot s mod D (`ready`)."""
        win = self.win
        p = s % win.depth
        for k, x in win.slot_groups.items():
            self.ring_groups[p][k].copy_(x)
        self.ring_scale[p].copy_(win.scale_all)
        self.ring_live[p].copy_(win.live)
        self.ring_aroots[p].copy_(win.pre["mid"].level[self.a_off:self.a_off + 8 * win.L])
        self.ring_S[p].copy_(win.S)
        self.ring_s1[p].copy_(win.S)
        self.ring_s1[p][3:].zero_()
        self.ring_stepv[p].copy_(win.stepv)
        self.ring_base[p].copy_(win.step_base)
        self.ready[p].record()

    def so_args(self, w: int, p: int, m: int) -> tuple:
        """kh_one's arguments for call w of the step in ring slot p but its stream (as `one_args`)."""
        win = self.win
        n, k = win.nk[w]
        live = -(-min(m, win.t) // 32)
        ws = self.ws_ring[w % NA] if self.noise else self.ws
        return (self.ring_slot[p][w].data_ptr(), win.wt[w].data_ptr(), win.t, n, k,
                win.plan.seeds.data_ptr() + 8 * 16 * self.seed0[w], self.ring_live[p].data_ptr(),
                self.stage.data_ptr() + 4 * win.plan.offs[w][0], ws.data_ptr(),
                auto_j(live * (n // 32), self.nb[w], self.sms), self.zs[0].data_ptr(), self.zs[1].data_ptr(),
                self.ys.data_ptr(), None, self.ring_scale[p][w].data_ptr(), win.sw[w].data_ptr(),
                self.ring_s1[p].data_ptr(), min(m, win.t))

    def dispatch(self, j: int, salt: bytes, m: int, g: dict | None) -> None:
        """Step j's units on the units stream once its ring slot is written: seeds under `salt`, each call's units,
        the commit and the receipt trees, whose roots go into `rroots` for round j + D - 1 (`udone`); a capture's
        receipts and roots into `g`."""
        win = self.win
        p, st, L = j % win.depth, self.side, win.L
        self.salt_host[p].copy_(torch.from_numpy(words(salt)))
        with torch.cuda.stream(st):
            st.wait_event(self.ready[p])
            self.ring_salt[p].copy_(self.salt_host[p], non_blocking=True)
            if g is not None:
                for x in (self.stage, win.plan.out, win.plan.seeds, win.rset.level):
                    x.view(torch.uint8).fill_(DF.POISON)
            tab = self.tables[p]
            rc = win.r.lib.h1r_launch(tab.table.data_ptr(), L, tab.max_units, tab.ks, self.ring_salt[p].data_ptr(),
                                      self.ring_base[p].data_ptr(), win.plan.seeds.data_ptr(), self.stage.data_ptr(), 0, 0,
                                      st.cuda_stream)
            if self.noise:
                self.seeded.record(st)
            for w in range(L):
                args = self.so_args(w, p, m)
                if self.noise:
                    i = w % NA
                    ns = self.nstreams[i]
                    ns.wait_event(self.seeded if w < NA else self.u_done[i])
                    rc = rc or self.h.kh_one_noise(*args, ns.cuda_stream)
                    self.n_done[i].record(ns)
                    st.wait_event(self.n_done[i])
                    rc = rc or self.h.kh_one_units(*args, st.cuda_stream)
                    self.u_done[i].record(st)
                else:
                    rc = rc or self.h.kh_one(*args, st.cuda_stream)
            if rc:
                raise RuntimeError(f"dither-a-so step {j}'s units: a launch returned {rc}")
            n = self.stage.numel()
            DA.dithera_commit[(triton.cdiv(n, 4096),)](self.stage, win.plan.out, self.ring_S[p], n, BLOCK=4096)
            DA.dithera_fault[(1,)](win.plan.out, self.ring_S[p], self.at, L, 1 << self.shift,
                                   BLOCK=triton.next_power_of_2(L))
            win.c.build(win.rset, self.ring_stepv[p][0:1])
            self.rroots[p].copy_(win.rset.roots)
            if g is not None:
                g["receipts"], g["r_roots"] = win.plan.out.clone(), win.rset.roots.clone()
            self.udone[p].record(st)
        self.dispatched[j] = p
        self.unit_steps += 1

    def info(self) -> dict:
        return {**super().info(), "order": ORDER, "depth": self.win.depth, "a_tables": AO[self.so_ao],
                "a_overlap": self.so_ao, "unit_steps": self.unit_steps,
                "overlap": None if self.ov is None else {"order": OV_ORDER, "ring": self.ov.R}}


def _remap(c, st, data: list | None = None, out: list | None = None):
    """A set of `st`'s tables over its own levels and roots, table w's rows at data[w] and its root at out[w] where
    given."""
    return c.set([dataclasses.replace(t, data=t.data if data is None else data[w], out=t.out if out is None else out[w])
                  for w, t in enumerate(st.tabs)])


class Ov:
    """-ov's rounds beside the next forward (the module docstring): the round ring (R slots of step s's records, admit
    rows, step and A roots, each with its own admit, z/y, served-y and units' y tables, step trees and round), the sender
    thread (round s's tables and tree on the tree streams, its root sent) and the dispatcher thread (step s's units once
    coin s is back).  The host's counters order the rings: `put_upto` (rounds handed over), `sent_upto` (rounds built
    and sent) and `dispatched_upto` (steps whose units are enqueued)."""

    EV = 8      # the GPU's events a step, in a ring of this many steps (the host's waits keep them that far apart)

    def __init__(self, win: SoWindow, sd: SoSide, ring: int):
        if not 1 <= ring < self.EV - 1:
            raise ValueError(f"dither-a-so -ov: the round ring's depth is 1 to {self.EV - 2}, not {ring}")
        self.win, self.sd, self.R = win, sd, ring
        dev, c, L = win.dev, win.c, win.L
        # the live records as one buffer each (no forward has written them yet), so a step's ring copy is a launch each
        yu_off = [t.data.data_ptr() - sd.zy[w].data_ptr() for w, t in enumerate(sd.yuset.tabs)]
        self.zl, self.yl = [x.numel() for x in sd.zy], [x.numel() for x in sd.ysv]
        self.zy = torch.empty(sum(self.zl), dtype=torch.uint8, device=dev)
        self.ysv = torch.empty(sum(self.yl), dtype=torch.uint8, device=dev)
        sd.zy, sd.ysv = list(self.zy.split(self.zl)), list(self.ysv.split(self.yl))
        sd.zyset = _remap(c, sd.zyset, data=sd.zy)
        sd.ysset = _remap(c, sd.ysset, data=sd.ysv)
        sd.yuset = _remap(c, sd.yuset, data=[sd.zy[w][o:] for w, o in enumerate(yu_off)])
        win.pre["z"] = sd.zyset
        kinds = win.round_kinds
        self.yi, self.zi, self.ri = kinds.index("y-step"), kinds.index("z-step"), kinds.index("receipt-step")
        self.zy_r = [torch.empty_like(self.zy) for _ in range(ring)]
        self.ysv_r = [torch.empty_like(self.ysv) for _ in range(ring)]
        self.admit_r = [torch.empty_like(win.admit) for _ in range(ring)]
        self.stepv_r = [torch.empty_like(win.stepv) for _ in range(ring)]
        self.a_off = win.pre["mid"].level_off[0]
        self.sets = []
        def leaves(st, at: int) -> list[int]:
            return [at + 64 * w for w in range(len(st.tabs))]
        for q in range(ring):
            zv, yv = self.zy_r[q].split(self.zl), self.ysv_r[q].split(self.yl)
            rnd = c.set(list(win.pre["round"].tabs))
            mid = _remap(c, win.pre["mid"], out=leaves(win.pre["mid"], rnd.leaves_addr(0)))
            if mid.level_off != win.pre["mid"].level_off:
                raise RuntimeError("dither-a-so -ov: the ring's step trees are not laid out as the window's")
            ad = win.pre["admit"]
            self.sets.append({
                "round": rnd, "mid": mid,
                "admit": _remap(c, ad, data=[self.admit_r[q]] * len(ad.tabs), out=leaves(ad, mid.leaves_addr(1))),
                "zy": _remap(c, sd.zyset, data=zv, out=leaves(sd.zyset, mid.leaves_addr(2))),
                "ys": _remap(c, sd.ysset, data=yv, out=leaves(sd.ysset, mid.leaves_addr(self.yi))),
                "yu": _remap(c, sd.yuset, data=[zv[w][o:] for w, o in enumerate(yu_off)])})
        if win.tree_streams is not None:
            self.ts, self.ads = win.tree_streams
        else:
            self.ts, self.ads = torch.cuda.Stream(device=dev), None
        self.fork, self.joined, self.done = torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event()
        self.ev = {k: [torch.cuda.Event(enable_timing=True) for _ in range(self.EV)]
                   for k in ("copied", "t0", "t1", "fwd0", "fwd1", "fault")}
        self.root_host = torch.empty(8, dtype=torch.int64).pin_memory()
        self.ybad_host = torch.empty(L, dtype=torch.bool).pin_memory()
        self.cv = threading.Condition()
        self.put_upto = self.sent_upto = self.dispatched_upto = self.fwd_upto = -1
        self.error: str | None = None
        self.draining = False
        self.zlate: tuple | None = None
        self.watch: set[int] = set()
        self.faulted: set[int] = set()
        self.t: dict[int, dict] = {}
        self.dev_index = torch.cuda.current_device()
        self.ph_round: dict[int, dict] = {}
        self.ph_units: dict[int, dict] = {}
        self.q_send: queue.SimpleQueue = queue.SimpleQueue()
        self.q_units: queue.SimpleQueue = queue.SimpleQueue()
        for name, qu, fn in (("dither-ov-rounds", self.q_send, self._round), ("dither-ov-units", self.q_units, self._units)):
            threading.Thread(target=self._loop, args=(qu, fn), name=name, daemon=True).start()

    # -- the host's order -------------------------------------------------------------------------------------------

    def wait(self, ready) -> None:
        with self.cv:
            while self.error is None and not ready():
                self.cv.wait()
            if self.error is not None:
                raise RuntimeError(f"dither-a-so -ov: {self.error}")

    def _set(self, **kw) -> None:
        with self.cv:
            for k, v in kw.items():
                setattr(self, k, v)
            self.cv.notify_all()

    def forward_starts(self, s: int) -> None:
        """Step s's forward is next on the serving stream (its `before` done).  After a fault_ring step, its fault:
        bit 0 of call 0's first z word in step s - 1's ring copy, on the serving stream as forward s starts, before
        round s - 1's tables (its round waits for it)."""
        self.ev["fwd0"][s % self.EV].record()
        if self.win.cap_of.get(s - 1) == "fault_ring":
            self.zy_r[(s - 1) % self.R][0:1].bitwise_xor_(1)
            self.ev["fault"][(s - 1) % self.EV].record()
            self.faulted.add(s - 1)
        self._set(fwd_upto=s)

    def put(self, s: int) -> None:
        self._set(put_upto=s)
        self.q_send.put(s)

    def drain(self) -> None:
        self._set(draining=True)
        try:
            self.wait(lambda: self.sent_upto >= self.put_upto and self.dispatched_upto >= self.put_upto)
        finally:
            self._set(draining=False)
        torch.cuda.synchronize()

    def _loop(self, qu: queue.SimpleQueue, fn) -> None:
        torch.cuda.set_device(self.dev_index)
        while True:
            s = qu.get()
            try:
                with torch.inference_mode():
                    fn(s)
            except BaseException as e:  # noqa: BLE001
                tb = traceback.format_exc(limit=6).strip().splitlines()[-4:]
                self._set(error=f"{threading.current_thread().name}, step {s}: {type(e).__name__}: {e} ({' | '.join(tb)})")
                return

    @contextlib.contextmanager
    def phase(self, name: str):
        """`Window.phase`'s NVTX range on this thread (stepprof joins a launch to its own thread's phase)."""
        if not self.win.nvtx:
            yield
            return
        torch.cuda.nvtx.range_push(DO.PHASE_RANGE + name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    # -- the serving stream, after forward s --------------------------------------------------------------------------

    def poison(self, q: int) -> None:
        for st in self.sets[q].values():
            st.level.view(torch.uint8).fill_(DF.POISON)
            st.roots.view(torch.uint8).fill_(DF.POISON)

    def copy(self, s: int) -> None:
        """Step s's records, admit rows, step and A roots (round s's inputs that forward s + 1 overwrites) into round
        slot s mod R."""
        win, q = self.win, s % self.R
        self.zy_r[q].copy_(self.zy)
        self.ysv_r[q].copy_(self.ysv)
        self.admit_r[q].copy_(win.admit)
        self.stepv_r[q].copy_(win.stepv)
        o, n = self.a_off, 8 * win.L
        self.sets[q]["mid"].level[o:o + n].copy_(win.pre["mid"].level[o:o + n])
        self.ev["copied"][s % self.EV].record()

    # -- the sender: round s ------------------------------------------------------------------------------------------

    def _round(self, s: int) -> None:
        win, sd, c, L, D = self.win, self.sd, self.win.c, self.win.L, self.win.depth
        q, e = s % self.R, s % self.EV
        st, stepv, ts, ads = self.sets[q], self.stepv_r[q][0:1], self.ts, self.ads
        cap, kinds = win.cap_of.get(s), win.round_kinds
        t_start = time.perf_counter()
        with self.phase("ov_leaf_tables"), torch.cuda.stream(ts):
            ts.wait_event(self.ev["copied"][e])
            if s in self.watch:
                # a captured step's round starts once forward s + 1 has (once it may write the live records), and
                # fault_ring's once its fault is in (`forward_starts`)
                self.wait(lambda: self.fwd_upto > s or self.draining)
                if self.fwd_upto > s:
                    ts.wait_event(self.ev["fault"][e] if cap == "fault_ring" else self.ev["fwd0"][(s + 1) % self.EV])
            self.ev["t0"][e].record(ts)
            if ads is not None:
                self.fork.record(ts)
                ads.wait_event(self.fork)
                with torch.cuda.stream(ads):
                    c.build(st["admit"], stepv)
                    self.joined.record(ads)
            else:
                c.build(st["admit"], stepv)
            for k in ("zy", "ys", "yu"):
                c.build(st[k], stepv)
            o = st["mid"].level_off[self.yi]
            self.ybad_host.copy_((st["mid"].level[o:o + 8 * L].view(L, 8) != st["yu"].roots.view(L, 8)).any(1),
                                 non_blocking=True)
        # its receipt-step leaf: step s - D + 1's receipt roots, once its units are enqueued
        j = s - (D - 1)
        t_leaves = time.perf_counter()
        with self.phase("ov_receipt_wait"):
            self.wait(lambda: self.dispatched_upto >= j)
        t_units = time.perf_counter()
        with self.phase("ov_round"), torch.cuda.stream(ts):
            o = st["mid"].level_off[self.ri]
            lv = st["mid"].level[o:o + 8 * L]
            if j < 0:
                lv.zero_()
            else:
                sd.dispatched.pop(j, None)
                ts.wait_event(sd.udone[j % D])
                if win.cap_of.get(j) == "fault_rr":
                    lv.view(torch.uint8).fill_(DF.POISON)
                else:
                    lv.copy_(sd.rroots[j % D])
            if ads is not None:
                ts.wait_event(self.joined)
            c.build(st["mid"], stepv)
            rl, zi, yi = st["round"].level, self.zi, self.yi
            if cap == "fault_zlate":
                self.zlate = (rl[8 * zi:8 * zi + 8].clone(), rl[8 * yi:8 * yi + 8].clone())
                for i in (zi, yi):
                    rl[8 * i:8 * i + 8].view(torch.uint8).fill_(DF.POISON)
            elif self.zlate is not None:
                rl[8 * zi:8 * zi + 8].copy_(self.zlate[0])
                rl[8 * yi:8 * yi + 8].copy_(self.zlate[1])
                self.zlate = None
            c.build(st["round"], stepv)
            self.root_host.copy_(st["round"].roots[0:8], non_blocking=True)
            self.ev["t1"][e].record(ts)
            self.done.record(ts)
        with self.phase("ov_root_wait"):
            self.done.synchronize()
        t_root = time.perf_counter()
        root = unwords(self.root_host)
        with self.phase("ov_send"):
            i = win.pipe.send(root)
        if i != s + 1:
            raise RuntimeError(f"step {s}'s round went out as round {i}, not {s + 1}")
        t_sent = time.perf_counter()
        bad = self.ybad_host.nonzero().flatten().tolist()
        if bad:
            sd.y_refused.append({"s": s, "capture": cap, "calls": len(bad)})
        caps = win.caps
        prev = win.cap_of.get(s - 1)
        late = win.cap_of.pop(j, None) if j >= 0 else None
        if (cap in caps) or (late in caps) or (prev == "fault_zlate" and prev in caps):
            with torch.cuda.stream(ts):
                rv, mv, yu = st["round"].level.cpu(), st["mid"].level.cpu(), st["yu"].roots.cpu()
            mo = st["mid"].level_off
            leaves = [unwords(rv[8 * k:8 * k + 8]) for k in range(len(kinds))]

            def mleaves(i: int) -> list[bytes]:
                return [unwords(mv[mo[i] + 8 * w:mo[i] + 8 * w + 8]) for w in range(L)]
            if cap in caps:
                caps[cap].update(root=root, index=i, leaves=leaves, a_roots=mleaves(0), admit_roots=mleaves(1),
                                 z_roots=mleaves(2), y_roots=mleaves(self.yi), y_root_bad=bad,
                                 yu_roots=[unwords(yu[8 * w:8 * w + 8]) for w in range(L)])
            if late in caps:
                caps[late].update(late_s=s, late_index=i, late_leaves=leaves, late_root=root)
            if prev == "fault_zlate" and prev in caps:
                caps[prev].update(next_leaves=leaves)
        win.round_log.append({"s": s, "m": win.m_of.get(s), "root": root.hex()[:32]})
        self.ph_round[s] = {"trees_ms": 1e3 * (t_root - t_start - (t_units - t_leaves)),
                            "receipts_wait_ms": 1e3 * (t_units - t_leaves), "record_host_ms": 1e3 * (t_sent - t_root),
                            "round_host_ms": 1e3 * (time.perf_counter() - t_start)}
        if s >= 1:
            self._beside(s - 1)
        self._set(sent_upto=s)
        self.q_units.put(s)

    def _beside(self, s: int) -> None:
        """Round s's tree build against forward s + 1 on the GPU's clock (both done: round s + 1 is handed over after
        forward s + 1), from step s's ring copy."""
        e, f = s % self.EV, (s + 1) % self.EV
        ref = self.ev["copied"][e]
        t0, t1 = ref.elapsed_time(self.ev["t0"][e]), ref.elapsed_time(self.ev["t1"][e])
        f0, f1 = ref.elapsed_time(self.ev["fwd0"][f]), ref.elapsed_time(self.ev["fwd1"][f])
        x = {"tree_gpu_ms": t1 - t0, "beside_forward_ms": max(0.0, min(t1, f1) - max(t0, f0))}
        if s in self.ph_round:
            self.ph_round[s].update(x)
        if s in self.watch:
            self.watch.discard(s)
            x.update(copied_to_tree_ms=t0, tree_end_ms=t1, forward_next_start_ms=f0, forward_next_end_ms=f1)
            if s in self.faulted:
                self.faulted.discard(s)
                x["fault_at_ms"] = ref.elapsed_time(self.ev["fault"][e])
            self.t[s] = x

    # -- the dispatcher: step s's units --------------------------------------------------------------------------------

    def _units(self, s: int) -> None:
        win, sd, pipe = self.win, self.sd, self.win.pipe
        t0 = time.perf_counter()
        with self.phase("ov_coin_wait"):
            opening = pipe.opening(s + 1)
        t_coin = time.perf_counter()
        win.coin_waits.append(1e3 * (t_coin - t0))
        name = win.cap_of.get(s)
        if name == "fault_rearly":
            held = pipe.opening(s)
            salt = win.HT.live_salt(win.coins.commitment, held.coin, held.index, pipe.roots[s])
        else:
            salt = win.HT.live_salt(win.coins.commitment, opening.coin, opening.index, pipe.roots[s + 1])
        g = win.caps.get(name) if name is not None else None
        if g is not None:
            g.update(opening=opening, salt_used=salt, coin_got_at=pipe.got_at.get(s + 1), dispatched_at=time.monotonic())
        with self.phase("ov_units_dispatch"):
            sd.dispatch(s, salt, win.m_of.pop(s), g)
        self.ph_units[s] = {"coin_wait_ms": 1e3 * (t_coin - t0), "dispatch_ms": 1e3 * (time.perf_counter() - t_coin)}
        self._set(dispatched_upto=s)


CONF: dict = {}


def side() -> SoSide:
    if not DA.SIDE:
        win = DO._window(torch.device("cuda"))
        DA.SIDE.append(SoSide(win, CONF["unit"], CONF["h1u"], CONF["ao"], CONF["pt"]))
    return DA.SIDE[0]


def patch(t: int, h1c: str, h1o: str, unit: str, h1u: str, ao: str, pt: bool, depth: int, key: bytes | None = None,
          ov: int = 0, serve_prio: bool = False) -> None:
    """dither_a's patch with z/y leaves (its op, `dithera::mm`, and the round's y-step and receipt-step leaves), then its
    window this module's (ring depth `depth`; `ov` > 0: -ov with a round ring that deep, and CAPTURES its) and its op's
    calls this module's Side (before the engine is built).  `serve_prio`: the engine serves on a stream at the A tables'
    priority, above the trees' and the units' (so the next forward's blocks take the SMs first as theirs retire)."""
    global CAPTURES
    if unit not in DA.KONE or ao not in AO or depth < 2 or ov < 0:
        raise ValueError(f"dither-a-so: unit one of {DA.KONE} (not {unit!r}), A tables one of {tuple(AO)} (not {ao!r}), "
                         f"depth at least 2 (not {depth}), round ring 0 (no -ov) or more (not {ov})")
    DA.patch(t=t, h1c=h1c, h1o=h1o, unit=unit, h1u=h1u, key=key, z="zy", ao="", pt=pt, ls=False)
    CONF.update(unit=unit, h1u=h1u, ao=ao, pt=pt, depth=depth, ov=ov, serve_prio=serve_prio)
    SoWindow.depth, SoWindow.a_once, SoWindow.overlap = depth, ao.startswith("b"), ov
    if ov:
        CAPTURES = CAPTURES_OV
    if serve_prio:
        # vLLM serves (captures and replays its graphs) on the stream its patched set_stream last recorded
        from vllm.utils import torch_utils  # noqa: F401
        torch.cuda.set_stream(torch.cuda.Stream(priority=-1))
    DO.Window = SoWindow
    DA.side = side


def stats(win: SoWindow):
    def doc() -> dict:
        return {**win.stats(), "dither_a": DA.SIDE[0].info() if DA.SIDE else None}
    return doc


# -- the gate ----------------------------------------------------------------------------------------------------------------

def check(win: SoWindow, lib_h1o: str, which: str = "gate", n_cpu_units: int = 3) -> dict:
    """`which`'s capture: dither_a's checks on round s and, for its receipts, round s + D - 1, every receipt under coin
    s's salt from the challenger's opening, and the order (the module docstring); under -ov also that round s, built
    beside forward s + 1 from the ring, covers what forward s committed ("ov")."""
    win.drain()
    torch.cuda.synchronize()
    g = win.caps.get(which, {})
    need = ("a", "receipts", "opening", "late_root")
    if any(k not in g for k in need):
        return {"ok": False, "complete": False, "why": f"{which} step incomplete: {sorted(g)}"}
    s, pipe, com = g["s"], win.pipe, win.coins.commitment
    o, i, li = g["opening"], g["index"], g["late_index"]
    why = CO.Ch.live_reason(com, o, pipe.label, s + 1, g["root"])
    salt = win.HT.live_salt(com, o.coin, s + 1, g["root"])
    order = {"round": i, "coin": s + 1, "receipts_round": li,
             "coin_after_its_round_root": why is None and i == s + 1, "coin_reason": why,
             "roots_sent_are_captured": len(pipe.roots) > li and pipe.roots[i] == g["root"] and pipe.roots[li] == g["late_root"],
             "receipts_round_after_coin": (li > i and g["coin_got_at"] is not None
                                           and g["coin_got_at"] <= g["dispatched_at"] <= pipe.sent_at.get(li, -1.0)),
             "units_salt_is_the_coins": g["salt_used"] == salt}
    order["ok"] = (order["coin_after_its_round_root"] and order["roots_sent_are_captured"]
                   and order["receipts_round_after_coin"])
    r_round = {"s": g["late_s"], "leaves": g["late_leaves"], "root": g["late_root"]}
    doc = DA.check(win, lib_h1o, which, n_cpu_units, g={**g, "salt": salt}, r_round=r_round)
    doc["order"], doc["depth"] = order, win.depth
    rd = doc["round"]
    rest_ok = not (doc["receipts_vs_h1r_bad"] or doc["z_vs_int_mm_bad"] or doc["y_vs_dequant_z_bad"]
                   or doc["y_leaf_vs_dequant_z_bad"] or doc["a_roots_bad"] or doc["admit_roots_bad"] or doc["z_roots_bad"]
                   or doc["y_served_roots_bad"] or doc.get("receipt_roots_bad"))
    ov = None
    if win.overlap:
        # what forward s committed (the capture: cloned on the serving stream at its end, before forward s + 1) against
        # round s as the sender built it from the ring, its tables started once forward s + 1 had (from then on that
        # forward writes the live records), and how long the build ran beside it (GPU events, ms from the ring copy)
        t = DA.SIDE[0].ov.t.get(s, {})
        ov = {"ring": win.overlap, **{k: round(v, 3) for k, v in t.items()},
              "covers_forward_s": (all(rd[k] for k in ("a_step", "admit_step", "z_step", "y_step")) and rd["root"]
                                   and not (doc["a_roots_bad"] or doc["admit_roots_bad"] or doc["z_roots_bad"]
                                            or doc["y_served_roots_bad"])),
              "after_forward_next_start": t.get("copied_to_tree_ms", -1.0) >= t.get("forward_next_start_ms", 0.0)}
        doc["ov"] = ov
    if which == "fault_ring" and ov is not None:
        # step s's ring copy of call 0's z overwritten during step s + 1, before round s's tables: refused on call 0's
        # z root, round s's z-step leaf and its root alone, forward s's own values and every other commitment kept
        fa = ov.get("fault_at_ms")
        ov["fault_during_next_forward"] = (fa is not None and ov["forward_next_start_ms"] <= fa
                                           <= min(ov["forward_next_end_ms"], ov["copied_to_tree_ms"]))
        others = all(v for k, v in rd.items() if k not in ("z_step", "root"))
        doc["conforming"] = (doc["z_roots_bad"] == [0] and not rd["z_step"] and not rd["root"] and others
                             and not (doc["receipts_vs_h1r_bad"] or doc["z_vs_int_mm_bad"] or doc["y_vs_dequant_z_bad"]
                                      or doc["y_leaf_vs_dequant_z_bad"] or doc["a_roots_bad"] or doc["admit_roots_bad"]
                                      or doc["y_served_roots_bad"] or doc.get("receipt_roots_bad"))
                             and order["ok"] and order["units_salt_is_the_coins"] and ov["fault_during_next_forward"])
        doc["ok"] = doc["ok"] and order["ok"]
        return doc
    if which == "fault_zlate":
        # withheld from round s (refused there on its z-step and y-step leaves and root alone) and found under round
        # s + 1, after coin s
        nl, sr, kinds = g.get("next_leaves"), doc["step_roots"], win.round_kinds
        zi, yi = kinds.index("z-step"), kinds.index("y-step")
        doc["committed_after_coin"] = bool(nl) and (nl[zi].hex(), nl[yi].hex()) == (sr["z-step"], sr["y-step"])
        others = all(v for k, v in rd.items() if k not in ("z_step", "y_step", "root"))
        doc["conforming"] = (not rd["z_step"] and not rd["y_step"] and not rd["root"] and others and rest_ok
                             and order["ok"] and order["units_salt_is_the_coins"] and doc["committed_after_coin"])
        doc["ok"] = doc["ok"] and order["ok"]
        return doc
    if which == "fault_rearly":
        # the units under coin s - 1's salt: refused on every call's receipts, every commitment and the order kept
        doc["conforming"] = (doc["receipts_vs_h1r_bad"] == list(range(win.L)) and all(rd.values())
                             and not (doc["z_vs_int_mm_bad"] or doc["y_vs_dequant_z_bad"] or doc["a_roots_bad"]
                                      or doc["z_roots_bad"] or doc["y_served_roots_bad"] or doc.get("receipt_roots_bad"))
                             and order["ok"] and not order["units_salt_is_the_coins"])
        doc["ok"] = doc["ok"] and order["ok"] and order["units_salt_is_the_coins"]
        return doc
    good = order["ok"] and order["units_salt_is_the_coins"]
    if ov is not None and which == "gate":
        good = good and ov["covers_forward_s"] and ov["after_forward_next_start"]
    if "conforming" in doc:
        doc["conforming"] = doc["conforming"] and good
    doc["ok"] = doc["ok"] and good
    return doc


def gate_verdict(gate: dict, control: dict, fault: dict, *more: dict) -> dict:
    """dither_a's verdict over the captures (each fault refused, CAPTURES' fault_rearly and fault_zlate among them),
    with the gate's order as its own line."""
    v = DA.gate_verdict(gate, control, fault, *more)
    v["order"] = ORDER
    v["order_ok"] = bool(gate.get("order", {}).get("ok"))
    v["ok"] = v["ok"] and v["order_ok"]
    if "ov" in gate:
        # -ov: round s built once forward s + 1 had started covers what forward s committed, and fault_ring (in `more`)
        # refused
        v["ov"] = OV_ORDER
        v["ov_ok"] = bool(gate["ov"].get("covers_forward_s") and gate["ov"].get("after_forward_next_start"))
        v["ok"] = v["ok"] and v["ov_ok"] and any(x.get("which") == "fault_ring" for x in more)
    return v
