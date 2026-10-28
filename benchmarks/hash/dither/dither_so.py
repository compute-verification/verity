"""dither-a in the statement's order: the arms
`dither-a-{hbk,hbr}-zy-{aom,aomn,ab,abn}[-pt]-so<D>[-ov|-ov1|-ovp][-we|-wea]`, labelled as every arm by
`forms.form_of` (`SoSide.label`): -aom/-aomn without -ov form=`forms.FORM_A`, since the served epilogue commits
the per-unit z/y records under round s's root before coin s; -ab/-abn and -ov (-ov1, -ovp) form=beside (unadopted
arm), neither ruled in (`SoSide.unadopted`).

A step's A, scales and z/y leaves go under its round's root before its round's coin; its units read that coin and run
after it, beside a later step's forward; their receipts go under a later round.  Step s, in order:

    forward s        exactlin's quantizer (the call's A slot), the call's A table (`h1c_a_call`, form 3, on NA_MULTI A
                     streams above the GEMM's priority, joined at the last call and at every qkv call under piecewise
                     graphs), exactlin's GEMM with the z/y records (dither-a's -zy) and the served y's records
                     (`dithera_ycommit`).  No seeds and no units: nothing in the forward reads a coin.
    between steps    under -ab, every call's rows >= m cleared and the step's A tables in one h1c_build (h1agate.py's
                     `batch`: a per-step A commitment, a scheme change not ruled in, so a separate arm); the step's
                     A slots, scales, A roots, live rows and step into ring slot s mod D; the admit tables (under -pt on
                     their own stream), the z/y leaf tables and the served y's tables (their run-time compare);
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
a coin is back once its root is sent; off the node round s + D - 1 waits for step s's units unless (D - 1) * step >= R +
their span (R the round trip, the span about 10 ms), so the served T64 row on the RTX PRO 6000 (R about 100 ms) is -so8,
at 54.6 MB of GPU memory a slot past -so6 (T * sum K bytes of A codes, and their scales).

`check` holds a capture to dither_a's checks with its own round s for the A, admit, z and y leaves (the receipt-step leaf
there an earlier step's, read as recorded) and round s + D - 1 for its receipts (dither_a.check's `r_round`), every
receipt against h1r v1 under the salt of coin s as the challenger opened it, and to the order itself: coin s is round s's
coin opened after round s's root (`live_reason`), the roots sent are the rounds captured, and the receipts' round was
sent after coin s came back and after the units were launched (the pipe's clock).  Beyond dither-a's gate, control,
fault, fault_y and fault_rr (here: round s + D - 1 recorded without step s's receipt roots), two faults of the order
itself, each refused: "fault_rearly" (step s's units under the salt of coin s - 1, a coin the prover held before round
s's root: refused on every receipt) and "fault_zlate" (step s's z/y leaves withheld from round s, 0xA5 in their place,
and committed under round s + 1, after coin s: refused on round s's z-step and y-step leaves).  $DITHER_SO_UNITS=j10r
(`UNITS_ENV`; the verifier's form, at $DITHER_SO_J10R_M=slot on a slot T with every row tile computed:
`forms.j10r_part`) runs the units under J10-R's call-scale noise, every check here and at the window's end on j10r_so's
references in h1r v1's place, and the gate step's units held to J10R's Definition.

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

A window's open and end (served_h1's window close, since dither-a has none).  Every window opens with the salt
coin after its root (off the node the root the gate's open names, for development `coins.stand_in_root` of the session;
in process the epoch root), then the epoch round (round 0) over the B tables' roots.  Without -we a window has no end: its
last D - 1 steps' receipts lie under no round, and its last D - 2 steps' units never run.  With -we
(`-so<D>[-ov|-ov1|-ovp]-we`) each window ends (`SoWindow.end`) after its last step S - 1: the remaining coins awaited and
the tail's units run (steps S - D + 2 to S - 1; under -ov the dispatcher's, drained), the receipt-tail roots of steps
S - D + 1 to S - 1 (each the tree over a step's receipt-table roots, its domain naming that step) under one final round
sent after the last step's coin, whose coin comes back; then the manifest root, the tree over every round root of the
window in order (epoch, steps, final; its domain naming the window, its root and its steps), which binds each step's
round leaves and every call's A, admit, z, y and receipt roots under them; then the draw coin opened after the manifest
root, and the session closed.  Until the draw the prover keeps what it must open a drawn unit from: each step's live A
rows and scales and its z/y records' live row tiles (-wea: A and the scales alone, z/y recomputed from them), copied
to pinned host memory on a stream of their own from the ring slot that holds them, a slot rewritten only once its copy is
out (`Spill`).  In process the window ends after each of sbench's generates (and a new one opens at the
next step on a challenger made beforehand, `prepare`, as the gate makes an off-node session's); off the node the
session's window ends at its layout's last step (R rounds: R - 2 steps), so sbench serves it full before it stops, and
every family releases all its coins.  `check_end` holds an ended window to: the salt coin after the window's root, every
round's coin after its root, the draw coin after the manifest root recomputed from the rounds the challenger received;
each step's round recomputed from its manifest row (the round's leaves and each call's roots), the final round from the
tail's receipt roots; every step's units dispatched after its coin came back, and its receipts' round (round s + D - 1,
or the final round) sent after that and before the draw coin, no round after it; and the last step's receipts, from its
ring slot, against h1r v1 under its coin's salt, up to 16 calls' A tables and receipt tables rehashed on the CPU
(`rehashed_calls`); and three steps opened from the spill (`open_spill`).  The draw coin's time is read under the
round pipe's lock (`coins.RoundPipe.draw`), so no round goes out between it and the draw.  Two end faults, each
in a window of its own (`end_steps`: 2D + 2 steps) and refused: "fault_tail" (the last step's units dropped: refused on
its order and every call's receipts) and "fault_late_receipt" (the last step's receipt-tail root withheld from the final
round and sent in a round after the draw coin: refused on that leaf and that round alone).  An off-node window's
manifest (`dither_manifest`: every round's root and leaves, the final round, the manifest root, the coins) goes to
$DITHER_MANIFEST_DIR, which `dither_manifest.py check` holds to the challenger's own run.

    import dither_so as SO
    SO.patch(t=T, h1c="libh1c.so", h1o="libh1o.so", unit="hbr", h1u="h1u_kr.so", ao="m", pt=True, depth=4)
    llm = LLM(model=int8_exact_checkpoint, ...)
    win = DO.hook(llm); win.captures = SO.CAPTURES
"""
from __future__ import annotations

import contextlib
import ctypes as C
import dataclasses
import hashlib
import queue
import threading
import time
import traceback

import numpy as np
import torch
import triton

import dither_a as DA
import dither_fused as DF
import dither_manifest as MF
import dither_one as DO
import forms as F
import h1c as H1CM
import j10r_ws as JWS
from dither_fused import CM, CO, H1R, Call, Dom, Tab, unwords, words
from hb_side import auto_j
from verity_config.settings import of

settings = of(__file__)
LIB_ENV = DA.LIB_ENV
CG = None
ORDER = "statement: A, scales, z/y under round s before coin s; units after coin s; receipts under round s + D - 1"
OV_ORDER = ("-ov: round s built beside forward s + 1 from ring copies taken after forward s, its root sent before coin s; "
            "step s's units dispatched once coin s is back")
# the faults of the order last: fault_zlate moves round s + 1's z/y leaves, so the step after it is no capture's
CAPTURES = ("gate", "control", "fault", "fault_y", "fault_rr", "fault_rearly", "fault_zlate")
CAPTURES_OV = ("gate", "control", "fault", "fault_y", "fault_rr", "fault_ring", "fault_rearly", "fault_zlate")
END_ORDER = ("-we: at the window's end the tail's units after their coins, their receipt-tail roots under one final round "
             "sent after the last coin, the manifest root over every round root, then the draw coin after it")
NO_END = "none: the last D - 1 steps' receipts lie under no round and no draw coin opens; rows pending"
END_FAULTS = ("fault_tail", "fault_late_receipt")
MANIFEST_ENV = "DITHER_MANIFEST_DIR"    # an off-node window's manifest.json goes there (its per-call roots to scratch)
GRAPH_ENV = "DITHER_SO_GRAPH"           # "0": a step's units launched call by call (`SoSide.dispatch`), else one graph
PRIO_ENV = "DITHER_SO_PRIO"             # "1": any -so arm serves as -ovp does, on a stream above the units' (`patch`)
# "1": under the serving priority, vLLM's graphs captured on a stream at that priority too (`capture_at`): vLLM's
# graph_capture captures on a fresh priority-0 stream, and a replayed graph's kernels keep their capture's priority
CAPTURE_PRIO_ENV = "DITHER_SO_CAPTURE_PRIO"
# "1": a step's units in one launch over every call (h1u.cu's kh_units_multi: each unit's noise hashed by its own warp,
# no workspace), in place of each call's noise and units launches; the same receipts
FUSED_ENV = "DITHER_SO_FUSED"
# "1": step s's units, when coin s isn't back at the end of step s + D - 2's `after`, dispatched while the host waits on
# forward s + D - 1, once the coin is back (at the latest just after that forward, before round s + D - 1 takes their
# receipts), so the host launches forward s + D - 1 without waiting on the coin; the same order (`check`: coin, units,
# receipts' round).  "2": as "1", but the latest point is just before round s + D - 1's receipt leaf, after that step's
# A, admit and z tables are launched (the host polls between them), so the GPU builds them while the coin is out
COIN_ASYNC_ENV = "DITHER_SO_COIN_ASYNC"
# "q" or "q/den" (den 8 without it): that launch on the SMs whose id % den < q alone (kh_units_multi_sm), each warp
# taking the next unit in turn, so the rest of the SMs stay whole to the next forward's kernels; "q/den/cap" with at
# most cap of its CTAs on each of those SMs (kh_units_multi_smc; "1/1/cap" every SM), so each keeps the rest of its slots
# to them; the same receipts.  den at most SMQ_DEN_MAX (h1u.cu's), past every %smid, so q/256 keeps the SMs below q
UNITS_SMQ_ENV = "DITHER_SO_UNITS_SMQ"
SMQ_DEN_MAX = 256
# "1": the receipt tables' rows in a dead row tile (the zero receipt) take their x from one hash of that row, against
# the step's live flags (`SoSide._units`), and hash no segments; the same leaves and roots
RDEAD_ENV = "DITHER_SO_RDEAD"
# "j10r": a step's units under J10-R's call-scale noise (j10r_so.LABEL; forms.j10r_part): in h1r_launch's
# version 0 place every call's seed_c at its first position and each unit's sigma_R at its own (h1u.cu's
# kh_seeds_j10r_h1r, on the same call table), then each call's row tiles hashed once and its units on them
# (kh_units_multi_j10r, under $DITHER_SO_UNITS_SMQ kh_units_multi_j10r_sm); needs $DITHER_SO_FUSED=1.  The gate holds it
# to j10r_so's references (h1r's J10-R version 1 on every capture and the window's end, J10R's Definition on the gate step)
UNITS_ENV = "DITHER_SO_UNITS"
UNITS_FORMS = {"today": "each unit's own seed_j stream", "j10r": "J10-R: each call's seed_c stream at call scale"}
# R (with $DITHER_SO_UNITS=j10r and $DITHER_SO_FUSED=1): a step whose units compute at least R rows (the layout's M, its
# live rows under $DITHER_UNITS_SKIP_DEAD=1) hashes each call's column tiles once too, beside its rows, and its units
# read both (h1u.cu's kh_units_multi_j10rc, an h1u build with H1U_JRC=1, each call's workspace kh_j10r_ws_bytes at T
# rows; under $DITHER_SO_UNITS_SMQ without a cap, kh_units_multi_j10rc_sm); the same receipts.  Unset, never: J10-R's
# units hash a column tile once in each row tile
# they compute, so it pays at prefill's 64 and costs at decode's one, where its own launch is the difference (unitsbench
# --j10r, H1U_JSWZ 32: T 2048 step 287.1 -> 233.1 ms, T 512 71.9 -> 59.8 ms; T 64 M 32 5.11 -> 7.16 ms unswizzled)
J10RC_MIN_T_ENV = "DITHER_SO_J10RC_MIN_T"
# J10-R's layout rows of a call (h1tile.j10r_blocks' m; j10r_so.M_MODES), open until J10R rules: "slot" (default) or "live"
J10R_M_ENV = "DITHER_SO_J10R_M"
# "1" (with $DITHER_SO_UNITS=j10r): a row tile wholly past the step's live rows is skipped, h1tile's canonical dead tile
# (PouwH1Tile_v4's: the zero receipt, no noise derived), as today's units always do; unset, every unit of the call's M
# layout rows is computed in full, as J10-R's PouwH1Tile_v5 / v6 define it (the skip stays off by default until the
# credit counts only live units).  Unset, $DITHER_SO_RDEAD's zero-receipt rows don't hold
SKIP_DEAD_ENV = "DITHER_UNITS_SKIP_DEAD"
# "1" (without -ov, D >= 3): a step that holds no capture leaves its round on the serving stream with its root and y
# check copied to pinned memory, and returns to the engine; the next step's `before` waits for that copy, then sends
# the root (`_lag_finish`), so the trees run while the engine's host work does.  Forward s + 1 stays behind them on the
# serving stream; the same roots, round s's sent before step s + 1's forward is launched.  "2": as "1", but `before`'s
# inputs go through pinned buffers without a host wait (dither_one's `_h2d`) and round s goes out there only if its copy
# is already out, else at the next `after`'s start (after forward s + 1), so the trees also run while `before` and the
# engine's input preparation do; round s then waits up to a forward longer for its coin (a deeper ring: -so10)
ROOT_LAG_ENV = "DITHER_SO_ROOT_LAG"
# "1" (with the tree streams, -pt under the serving priority, without -ov): a step's z tables built on the first tree
# stream from the forward's end, beside its A tables, stash and spill on the serving stream, which waits for them
# before `before_mid`; the same tables
ZSIDE_ENV = "DITHER_SO_ZSIDE"
# DO.A_STEP_ENV's tables' level schedule (`H1C.set`'s lvf: their levels from L up in one block a table, on an H1C_HB
# build).  A K a table makes a step's tables few and deep (Llama-8B at T = 64: 2 tables of 2^13 leaves), so the build's
# H1C_LVF = 2 would run 11 of their levels on two blocks of the GPU.  Unset: from depth - A_LVF_TOP (at least 0), the
# level of 128 nodes a table (n1, RTX PRO 6000, T = 64: L = 4 to 7 within 0.01 ms, 0.16 ms under L = 2); "L": from L;
# "": the build's H1C_LVF, as every other set.  The roots are the same at every L
A_LVF_ENV = "DITHER_A_LVF"
A_LVF_TOP = 8
ROUNDS_ENV = "DITHER_ROUNDS"            # the window's layout (its round coins), the session's when off the node
MF_CHUNK = 64                           # manifest rows a pinned host chunk
# -we's spill (`Spill`): what each step keeps past the ring until its window's draw, so the prover can open a drawn unit
SPILLS = {"azy": "A rows and scales, and the z/y records (live rows; the rest are zero)",
          "a": "A rows and scales (live rows); z/y recomputed from them at an opening (z = A B^T, y = dequant(z))"}
SPILL_CHUNK = 2 << 30                   # pinned host bytes a chunk of the spill's pool
SPILL_MAX_ENV = "DITHER_SPILL_MAX_GB"   # the pool's bound, past which a window's spill refuses (default SPILL_MAX_GB)
SPILL_MAX_GB = 128
# "1": a step's GEMMs and y commits write a row past m's z/y records only where an earlier step left something else
# (a live row, a captured step's poison, a control's unwritten records, a new window): the same bytes in every record,
# without the dead rows' ~350 MB of stores a step at m = 32 of T = 64 (`dead_from`, S[2], dither_a._rows_written); and
# `after` clears the A slots' and scales' rows past m only below that row (the quantizer writes rows below m alone)
DEAD_ONCE_ENV = "DITHER_DEAD_ONCE"
# "1": a step's inputs written from pinned host memory without a wait (`_put`): the serving stream orders them after
# step s - 1's trees, lm_head and sampler, so the host launches forward s while the sampler runs
PUT_ASYNC_ENV = "DITHER_SO_PUT_ASYNC"
# "1": the units step s + 1 - d's `after` dispatches, queued as their graph before round s's root read, behind a wait on
# a pinned host word (cuStreamWaitValue32) that `SoSide.dispatch` sets once round s is sent, the coin is back and the
# salt is written: the same order (round s, coin, units), without the host's dispatch between them and the units' start.
# Their salt (their coin, back steps ago, and its round's root) is taken and staged with them (`_salt_early`), so the
# flag is the dispatch's one write. Graph steps only, and never a step that holds a capture or a fault, nor a window's
# last step's units
PREQ_ENV = "DITHER_SO_PREQ"
# "1": a step's mid and round trees through h1c's `Tail`: their domains, pads and node Snaps (no leaf read) before the
# serving stream's wait on the units, every level of both in one launch after it, not two h1c_builds after it; the same
# bytes.  "2": as "1", and the round's root written by that launch into pinned host memory as well, read there once
# the launch's number is, and the step's ybad put there by a kernel before the wait: no copy engine (whose queue the
# spill's transfers hold) between the units and the root read.  A step that withholds or replaces its z/y round leaves
# (fault_zlate and the step after it) builds them as before
TAIL_ENV = "DITHER_SO_TAIL"
# "1" (a diagnostic): the host's progress as events on an idle stream, so a step's phases place it against the GPU's:
# h_tail_ms (the tail's launch after the units' end), h_read_ms (the root read after the tail), h_ret_ms (the root
# read's return after the tail), h_udone_ms (the units' end to the serving stream past its wait) and h_flag_ms (the
# units' flag set to their start, the step before's); on the host, h_spin_ms (TAIL=2's spin on the root's word past the
# units' event), h_wall_ms and h_cpu_ms (the root's return to the units' flag, wall and main's thread CPU), h_salt_ms
# (dispatch_units' salt) and h_side_ms (SoSide.dispatch)
HMARK_ENV = "DITHER_SO_HMARK"
# "1": a capture's copies go to host memory, not the device (a prefill capture at T = 2048 is ~30 GB); `DA.check` moves
# each call's tensors to the device as it reads them
CAP_HOST_ENV = "DITHER_SO_CAP_HOST"


def keep(x: torch.Tensor) -> torch.Tensor:
    """A capture's copy of x: on the host under CAP_HOST_ENV, else a device clone."""
    return x.to("cpu") if settings.get(CAP_HOST_ENV) else x.clone()


def end_steps(depth: int) -> int:
    """An end fault's window: a full tail (D - 1 steps) after at least D + 3 steps whose receipts went under rounds."""
    return 2 * depth + 2


def smq_of(v: str) -> tuple[int, int, int]:
    """$DITHER_SO_UNITS_SMQ's "q[/den[/cap]]" as (q, den, cap): den 8 and cap 0 (none) where unset, cap 0 where q is."""
    q, den, cap, *more = v.split("/") + ["", ""]
    if any(more) or not q.isdigit() or not (den or "8").isdigit() or not (cap or "0").isdigit():
        raise ValueError(f"${UNITS_SMQ_ENV}={v}: q, q/den or q/den/cap")
    return int(q), int(den or 8), int(cap or 0) if int(q) else 0


# the A tables: "m" per call in the forward on NA_MULTI streams; "b" once a step between steps (not ruled in); "n" after
# either: each call's noise on NA_MULTI streams beside the units
AO = {"m": "per call, in the forward", "mn": "per call, in the forward; noise on its own streams",
      "b": "once a step (h1agate's batch; not ruled in)", "bn": "once a step (not ruled in); noise on its own streams"}
NA = DA.NA_MULTI


class SoWindow(DO.DitherState):
    """dither_one's window with the rounds in the statement's order (the module docstring): its round coins through a
    `coins.RoundPipe`, its slots grouped a K so the ring copies them in a launch a K, the units of step s dispatched at
    the end of step s + D - 2's `after`."""
    group_slots = True
    depth = 2
    a_once = False
    overlap = 0     # -ov: the round ring's depth (`Ov`), 0 without
    window_end = False  # -we: each window ends (`end`)
    spill_mode: str | None = None   # -we: what each step keeps until the draw (SPILLS), held by `SoSide.spill`

    def __init__(self, *args, **kwargs):
        if self.depth < 2:
            raise ValueError(f"dither-a-so's ring depth is at least 2, not {self.depth}")
        # the window's own state (`_start` sets it at each open, the first from `_epoch` at the build)
        self.k, self.open, self.s0, self.prepared = -1, False, 0, None
        self.next_end: tuple | None = None
        self.end_name = self.end_fault = self.end_at = None
        self.end_each = False
        self.ends: list[dict] = []
        self.ends_from = 0
        self.end_recs: dict[str, dict] = {}
        self.mf_chunks: list[torch.Tensor] = []
        self.mf_pool: list[torch.Tensor] = []
        self.starts_ms: list[float] = []
        self.rtts: list[float] = []
        self.order_due: list[tuple] = []
        self.cap_of: dict[int, str] = {}
        self.m_of: dict[int, int] = {}
        self.stashed: set[int] = set()
        self.zlate: tuple | None = None
        self.coin_waits: list[float] = []
        self.order_why = None
        self.coin_async = settings.get(COIN_ASYNC_ENV)
        if self.coin_async not in (0, 1, 2):
            raise RuntimeError(f"{COIN_ASYNC_ENV}={self.coin_async}: 0, 1 or 2")
        self.deferred: int | None = None
        self.deferrals = {"deferred": 0, "in_forward": 0, "in_after": 0, "after_forward": 0}
        self.z_side = settings.get(ZSIDE_ENV)
        self.ev_z = (torch.cuda.Event(), torch.cuda.Event())
        self.late_waits: list[float] = []
        self.a_step = settings.get(DO.A_STEP_ENV)
        if self.a_step and (self.a_once or self.overlap):
            raise RuntimeError(f"{DO.A_STEP_ENV}=1 serves -aom/-aomn without -ov (its A tables after the forward, on "
                               f"the serving stream)")
        self.dead_once = settings.get(DEAD_ONCE_ENV)
        if self.dead_once and self.overlap:
            raise RuntimeError(f"{DEAD_ONCE_ENV}=1 serves an -so arm without -ov")
        self.clean_from: int | None = None
        self.dead_c: int | None = None
        self.put_async = settings.get(PUT_ASYNC_ENV)
        super().__init__(*args, **kwargs)
        if self.a_step:
            self._a_step_set()
        # the gate's generation runs past its last capture until that step's receipts are in a round
        self.gate_extra = self.depth + 8 + (1 if self.overlap else 0)
        # the serving stream's wait on the units (0 to 1), then its mid (to 2) and round (to 3) builds; under TAIL_ENV
        # both trees' levels are 1 to 2
        self.ev_w = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        self.ev_fwd = torch.cuda.Event()
        self.tail: H1CM.Tail | None = None
        tm = settings.get(TAIL_ENV)
        if tm not in ("0", "1", "2"):
            raise RuntimeError(f"${TAIL_ENV}={tm}: 0, 1 or 2")
        if tm != "0":
            if self.overlap:
                raise RuntimeError(f"${TAIL_ENV}={tm} serves an -so arm without -ov (whose rounds build on the sender's ring)")
            self.tail = H1CM.Tail(self.c, self.pre["mid"], self.pre["round"], host_root=tm == "2")
        self.hmark = settings.get(HMARK_ENV)
        if self.hmark:
            self.hm_stream = torch.cuda.Stream()
            # 0 the tail's launch, 1 the root read, 2 + s % 2 step s's units' flag, 4 the root read's return
            self.ev_h = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            self.h_flagged: set[int] = set()
            # perf_counter at dispatch_units' salt and at its SoSide.dispatch call
            self.h_t = [0.0, 0.0]
        self.root_lag = settings.get(ROOT_LAG_ENV)
        if self.root_lag not in (0, 1, 2):
            raise RuntimeError(f"${ROOT_LAG_ENV}={self.root_lag}: 0, 1 or 2")
        self.h2d_pinned = self.root_lag == 2
        if self.root_lag and (self.overlap or self.depth < 3):
            raise RuntimeError(f"${ROOT_LAG_ENV}=1 serves -so<D> with D >= 3 and without -ov (its units read coin s "
                               f"D - 2 steps on, and -ov sends its rounds from a thread of its own)")
        self.lag: dict | None = None
        self.lag_root = torch.empty(8, dtype=torch.int64).pin_memory() if self.root_lag else None
        self.lag_ybad: torch.Tensor | None = None
        self.ybad_pin: torch.Tensor | None = None
        self.ev_lag = torch.cuda.Event()
        self.lag_n, self.lag_waits = 0, []
        # the engine's own host time a step (from one `after`'s end to the next `before`), and `before`'s
        self.t_b0 = self.t_after_end = 0.0

    def _a_step_set(self) -> None:
        """DO.A_STEP_ENV's tables, pre["ak"]: a K's table over its slot group's rows, c = `DO.a_chunks` (K) leaves a
        row (the group read K / c bytes a leaf), and its scales' run of `scale_all` (a scale every c leaves);
        `a_leaves` the a-step table's leaves, call w's its K's root (`a_gidx`)."""
        ks, t, tabs, at = sorted(self.a_calls), self.t, [], 0
        for k in ks:
            n, c = len(self.a_calls[k]), DO.a_chunks(k)
            tabs.append(Tab(Dom("a-table", DO.a_step_fields(self, k, 0), c * n * t, {"step": (1, 0)}), c * n * t,
                            data=self.slot_groups[k].view(torch.uint8), stride=k // c, body=k // c, length=k // c + 4,
                            tail=self.scale_all[at:at + n].view(torch.uint8), tail_stride=4, chunks=c,
                            lab=(f"a/{k}@", "")))
            at += n
        v = settings.get(A_LVF_ENV)
        lvf = max(0, max(H1CM.merkle_depth(t.count) for t in tabs) - A_LVF_TOP) if v is None else int(v) if v else None
        self.pre["ak"] = self.c.set(tabs, lvf=lvf)
        self.a_gidx = torch.tensor([ks.index(k) for _, k in self.nk], dtype=torch.long, device=self.dev)
        o = self.pre["mid"].level_off[0]
        self.a_leaves = self.pre["mid"].level[o:o + 8 * self.L].view(self.L, 8)
        self.ev_a = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def _put(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        if self.put_async:
            dst.copy_(src.pin_memory(), non_blocking=True)
        else:
            super()._put(dst, src)

    def a_step_build(self) -> None:
        """On the current stream, after the forward: the step's K tables and each call's a-step leaf (between `ev_a`,
        the step's a_tables_ms)."""
        self.ev_a[0].record()
        self.c.build(self.pre["ak"], self.stepv[0:1])
        self.a_leaves.copy_(self.pre["ak"].roots.view(-1, 8).index_select(0, self.a_gidx))
        self.ev_a[1].record()

    def _epoch(self) -> None:
        self._start()

    def _start(self) -> None:
        """A window opens at step s (its first, s0): its coins (made beforehand by `prepare`, else now), the salt coin
        opened after the window's root, the epoch round (round 0) over the B tables' roots, and the round pipe; under -we
        its last step (`end_at`): the layout's (R rounds: R - 2 steps), or an `arm_end` window's."""
        from verity.core.protocols.pouw.circuit import h1tile as HT
        t0 = time.perf_counter()
        self.HT = HT
        self.clean_from = None
        at = CO.session()
        if at is not None and self.k >= 0:
            raise RuntimeError(f"dither-a-so: session {at.id} served this process's one off-node window, which ended at "
                               f"step {self.s - 1}; a window after it needs a session of its own (more rounds: "
                               f"${ROUNDS_ENV})")
        coins = self.prepared if self.prepared is not None else CO.open_coins(at, 1, rounds_of())
        self.prepared = None
        self.k += 1
        self.s0 = self.s
        self.coins, self.rounds = coins, coins.rounds
        br = self.bset.roots.cpu()
        self.b_roots = [unwords(br[8 * w:8 * w + 8]) for w in range(self.L)]
        self.epoch_root = CM.step_root(CM.domain({"window": self.window, "linears": self.L}, self.L, "b-step"),
                                       self.b_roots)
        self.open_root = CO.stand_in_root(at.id) if at is not None else self.epoch_root
        self.salt_open = coins.salt(self.open_root)
        opening = self.rounds.record(self.epoch_root)
        self.salt_bytes = HT.live_salt(coins.commitment, opening.coin, 0, self.epoch_root)
        self.salt.copy_(torch.from_numpy(words(self.salt_bytes)))
        self.pipe = CO.RoundPipe(coins, time.monotonic)
        self.layout_rounds = coins.commitment.family(CO.Wd.ROUND)
        self.end_each = self.window_end and at is None
        name, fault, steps = self.next_end or ("live" if at is not None else None, None, None)
        self.next_end = None
        self.end_name, self.end_fault = name, fault
        last = self.s0 + self.layout_rounds - 3
        self.end_at = (min(last, self.s0 + steps - 1) if steps else last) if self.window_end else None
        self.disp_at: dict[int, float] = {}
        self.salt_of: dict[int, bytes] = {}
        # step -> (its opening, its salt), taken at its units' prequeue (`_salt_early`)
        self.salt_early: dict[int, tuple] = {}
        self.cap_at: dict[int, str] = {}
        self.rtt_from, self.order_from = 1, 0
        self.zlate = None
        self.deferred = None
        if self.overlap and DA.SIDE:
            DA.SIDE[0].ov.zlate = None
        self.open = True
        self.starts_ms.append(1e3 * (time.perf_counter() - t0))

    def prepare(self) -> None:
        """In process, the next window's challenger made now, outside a timed generate (its commitment over 16384 rounds
        takes 1.7 s), as the challenger's gate makes an off-node session's before its window."""
        if not self.open and self.prepared is None and CO.session() is None:
            self.prepared = CO.open_coins(None, 1, rounds_of())

    def arm_end(self, name: str, fault: str | None = None, steps: int | None = None) -> None:
        """The next window opened is `name`'s: its end kept for `check_end`, under `fault` (one of END_FAULTS), and with
        `steps` it ends at its steps-th step."""
        if fault is not None and fault not in END_FAULTS:
            raise ValueError(f"dither-a-so: an end fault is one of {END_FAULTS}, not {fault!r}")
        self.next_end = (name, fault, steps)

    def ix(self, s: int) -> int:
        """Step s's round: its index in its window's round coins (round 0 the epoch round)."""
        return s - self.s0 + 1

    def steps_left(self) -> int:
        return self.end_at - self.s + 1 if self.open and self.end_at is not None else 0

    def put_row(self, s: int, mid, rnd) -> None:
        """Under -we, step s's manifest row on the current stream once its round is built: each round leaf's per-call
        roots (mid's leaves) and the round's leaves, to pinned host memory without a wait."""
        if not self.window_end:
            return
        c, o = divmod(s - self.s0, MF_CHUNK)
        L8, nr = 8 * self.L, len(self.round_kinds)
        while len(self.mf_chunks) <= c:
            self.mf_chunks.append(self.mf_pool.pop() if self.mf_pool else
                                  torch.empty((MF_CHUNK, (nr + 1) * L8), dtype=torch.int64).pin_memory())
        row = self.mf_chunks[c][o]
        for i, off in enumerate(mid.level_off):
            row[i * L8:(i + 1) * L8].copy_(mid.level[off:off + L8], non_blocking=True)
        ro = rnd.level_off[0]
        row[nr * L8:nr * L8 + 8 * nr].copy_(rnd.level[ro:ro + 8 * nr], non_blocking=True)

    def dispatch_units(self, j: int) -> float:
        """Step j's units once its coin is back (`SoSide.dispatch`, under its coin's salt; fault_rearly's under the coin
        before), unless fault_tail drops the window's last step's; when the coin came back (perf_counter)."""
        i = self.ix(j)
        t0 = time.perf_counter()
        opening = self.pipe.opening(i)
        t_coin = time.perf_counter()
        self.coin_waits.append(1e3 * (t_coin - t0))
        name = self.cap_of.get(j)
        early = self.salt_early.pop(j, None)
        if name == "fault_rearly":
            held = self.pipe.opening(i - 1)
            salt = self.HT.live_salt(self.coins.commitment, held.coin, held.index, self.pipe.roots[i - 1])
        elif early is not None and early[0] is opening:
            salt = early[1]
        else:
            salt = self.HT.live_salt(self.coins.commitment, opening.coin, opening.index, self.pipe.roots[i])
        if self.hmark:
            self.h_t[0] = time.perf_counter()
        g = self.caps.get(name) if name is not None else None
        if g is not None:
            g.update(opening=opening, salt_used=salt, coin_got_at=self.pipe.got_at.get(i), dispatched_at=time.monotonic())
        m = self.m_of.pop(j)
        if j == self.end_at and self.end_fault == "fault_tail":
            return t_coin
        self.salt_of[j] = salt
        self.disp_at[j] = time.monotonic()
        self.phase("units_dispatch")
        if self.hmark:
            self.h_t[1] = time.perf_counter()
        DA.SIDE[0].dispatch(j, salt, m, g)
        return t_coin

    def _salt_early(self, sd, j: int) -> None:
        """At step j's units' prequeue (before the root read, the host well ahead of the GPU), its salt from its coin if
        that is back, and the salt's words staged in its ring slot: its dispatch, after the round's root is sent, then
        sets the flag alone.  The salt is `live_salt` of a coin and a root both already recorded, as at dispatch."""
        i = self.ix(j)
        o = self.pipe.opening(i, wait=False)
        if o is None:
            return
        salt = self.HT.live_salt(self.coins.commitment, o.coin, o.index, self.pipe.roots[i])
        self.salt_early[j] = (o, salt)
        sd.stage_salt(j, salt)

    def _forward_wait_deferred(self, main: torch.cuda.Stream) -> None:
        """$DITHER_SO_COIN_ASYNC: while forward s runs, the deferred step's units dispatched once its coin is back."""
        self.ev_fwd.record(main)
        j = self.deferred
        i = self.ix(j)
        while not self.ev_fwd.query():
            if self.pipe.opening(i, wait=False) is not None:
                self.deferred = None
                self.deferrals["in_forward"] += 1
                self.stashed.discard(j)
                self.dispatch_units(j)
                self.phase("forward_wait")
                return
            time.sleep(1e-4)

    def _poll_deferred(self) -> None:
        """$DITHER_SO_COIN_ASYNC=2, between `after`'s table launches: the deferred step's units if its coin is back."""
        if self.deferred is not None and self.pipe.opening(self.ix(self.deferred), wait=False) is not None:
            j, self.deferred = self.deferred, None
            self.deferrals["in_after"] += 1
            self.stashed.discard(j)
            self.dispatch_units(j)

    def _late_dispatch(self) -> float:
        """The deferred step's units, the host waiting here for its coin (coin_wait_late); the wait in seconds."""
        self.phase("coin_wait_late")
        j, self.deferred = self.deferred, None
        self.deferrals["after_forward"] += 1
        self.stashed.discard(j)
        t0 = time.perf_counter()
        self.dispatch_units(j)
        t = time.perf_counter() - t0
        self.late_waits.append(1e3 * t)
        return t

    def _lag_finish(self) -> None:
        """$DITHER_SO_ROOT_LAG: the lagged step's round root, once its copy is out (the host waits here for what is left
        of the step's trees), sent as that step's round; its y check and units wait read."""
        lag, self.lag = self.lag, None
        if lag is None:
            return
        t0 = time.perf_counter()
        self.ev_lag.synchronize()
        wait = 1e3 * (time.perf_counter() - t0)
        root = unwords(self.lag_root)
        i = self.pipe.send(root)
        if i != lag["i"]:
            raise RuntimeError(f"step {lag['s']}'s round went out as round {i}, not {lag['i']}")
        bad = self.lag_ybad.nonzero().flatten().tolist()
        if bad:
            DA.SIDE[0].y_refused.append({"s": lag["s"], "capture": lag["cap"], "calls": len(bad)})
        row = lag["row"]
        row["root_wait_ms"] = wait
        if lag["waited"]:
            row["units_wait_ms"] = self.ev_w[0].elapsed_time(self.ev_w[1])
        self.lag_n += 1
        self.lag_waits.append(wait)
        self.round_log.append({"s": lag["s"], "m": lag["m"], "root": root.hex()[:32]})

    def end(self) -> dict | None:
        """The window's end (the module docstring): the tail's units after their coins, the receipt-tail roots under the
        final round, its coin back, the manifest root, the draw coin after it, the session closed."""
        self._lag_finish()
        if not self.open:
            return None
        sd: SoSide = DA.SIDE[0]
        W, L, D, s0, s1 = self.window, self.L, self.depth, self.s0, self.s
        if s1 == s0:
            return self.close()
        t0 = time.perf_counter()
        if self.overlap:
            pending = s1 - 1 - sd.ov.dispatched_upto
            sd.ov.drain()
        else:
            self.deferred = None
            pending = len(self.stashed)
            for j in sorted(self.stashed):
                self.stashed.discard(j)
                self.dispatch_units(j)
            self.phase(None)
        t_units = time.perf_counter()
        if sd.spill is not None:
            sd.spill.stream.synchronize()
        t_spill = time.perf_counter()
        torch.cuda.synchronize()
        t_sync = time.perf_counter()
        tail = list(range(max(s0, s1 - D + 1), s1))
        rr = {}
        for j in tail:
            sd.dispatched.pop(j, None)
            self.cap_of.pop(j, None)
            x = sd.rroots[j % D].cpu()
            rr[j] = [unwords(x[8 * w:8 * w + 8]) for w in range(L)]
        leaves = [MF.tail_leaf(W, j, rr[j]) for j in tail]
        withheld = None
        if self.end_fault == "fault_late_receipt":
            withheld, leaves[-1] = leaves[-1], bytes(64)
        froot = MF.final_root(W, tail, leaves)
        t_rr = time.perf_counter()
        i_f = self.pipe.send(froot)
        if i_f != self.ix(s1):
            raise RuntimeError(f"window {self.k}'s final round went out as round {i_f}, not {self.ix(s1)}")
        self.pipe.opening(i_f)
        t_final = time.perf_counter()
        mroot = MF.manifest_root(W, self.k, self.open_root, s0, s1, list(self.pipe.roots[:i_f + 1]))
        t_mf = time.perf_counter()
        draw, draw_at = self.pipe.draw(mroot)
        t_draw = time.perf_counter()
        late = None
        if withheld is not None:
            late = self.pipe.send(MF.late_root(W, tail[-1], withheld))
            self.pipe.opening(late)
        self._harvest(now=False)
        self.pipe.close()
        t_close = time.perf_counter()
        self.open = False
        n = s1 - s0
        live = self.coins.source == "off-node"
        rec = None
        if self.end_name is not None or live:
            rows = torch.cat(self.mf_chunks)[:n].numpy() if self.mf_chunks else None
            rec = {"k": self.k, "fault": self.end_fault, "s0": s0, "s1": s1, "tail": tail, "rr": rr,
                   "final_leaves": leaves, "withheld": withheld, "i_f": i_f, "late_index": late, "mroot": mroot,
                   "draw": draw, "draw_at": draw_at, "pipe": self.pipe, "coins": self.coins, "open_root": self.open_root,
                   "salt_open": self.salt_open, "rows": rows, "disp_at": self.disp_at, "salt_of": self.salt_of,
                   "cap_at": self.cap_at, "b_roots": self.b_roots}
        if self.end_name is not None:
            p = (s1 - 1) % D
            rec["last"] = {"a": [x.clone() for x in sd.ring_slot[p]],
                           "sa": [sd.ring_scale[p][self.spos[w]].clone() for w in range(L)],
                           "receipts": self.plan.out.clone(), "m": self.m_last}
            self.end_recs[self.end_name] = rec
        if sd.spill is not None:
            held = sd.spill.take_window()
            if self.end_name is not None:
                rec["spill"] = held
            else:
                sd.spill.release(held)
        self.mf_pool.extend(self.mf_chunks)
        self.mf_chunks = []
        t_end = time.perf_counter()
        e = {"k": self.k, "name": self.end_name, "fault": self.end_fault, "s0": s0, "steps": n, "tail": len(tail),
             "units_pending_at_end": pending, "final_index": i_f, "end_ms": 1e3 * (t_end - t0),
             "split_ms": {"tail_units": 1e3 * (t_units - t0), "spill_sync": 1e3 * (t_spill - t_units),
                          "tail_sync": 1e3 * (t_sync - t_spill),
                          "receipt_tail_roots": 1e3 * (t_rr - t_sync), "final_round": 1e3 * (t_final - t_rr),
                          "manifest_root": 1e3 * (t_mf - t_final), "draw_coin": 1e3 * (t_draw - t_mf),
                          "close": 1e3 * (t_close - t_draw), "record": 1e3 * (t_end - t_close)}}
        e["amortized_ms_per_step"] = e["end_ms"] / n
        self.ends.append(e)
        if live and settings.get(MANIFEST_ENV):
            e["manifest"] = write_manifest(self, rec, settings.get(MANIFEST_ENV))
        return e

    def close(self) -> dict | None:
        """At the arm's stop: an open window with steps ended under -we, else its rounds drained and its coins closed."""
        self._lag_finish()
        if not self.open:
            return None
        if self.window_end and self.s > self.s0:
            return self.end()
        self.deferred = None
        self.drain()
        self._harvest()
        self.pipe.close()
        self.open = False
        return None

    def _harvest(self, now: bool = True) -> None:
        """The round trips of each coin come back since the last harvest of this window's pipe, and each one's order
        (`live_reason`) now, or at the next `stats` (`order_due`: a timed end leaves this check of its own coins out)."""
        pipe = self.pipe
        n = len(pipe.openings)
        self.rtts += [1e3 * (pipe.got_at[i] - pipe.sent_at[i]) for i in range(self.rtt_from, n)
                      if i in pipe.got_at and i in pipe.sent_at]
        self.rtt_from = n
        self.order_due.append((pipe, self.order_from, n))
        self.order_from = n
        if now:
            self._order()

    def _order(self) -> None:
        for pipe, a, b in self.order_due:
            for i in range(a, b):
                if self.order_why is None:
                    self.order_why = CO.Ch.live_reason(pipe.commitment, pipe.openings[i], pipe.label, i, pipe.roots[i])
        self.order_due = []

    def host_ms(self) -> dict:
        t_end = time.perf_counter()
        x = {"before_ms": 1e3 * (self.t_before - self.t_b0)}
        if self.t_after_end:
            x["engine_ms"] = 1e3 * (self.t_b0 - self.t_after_end)
        self.t_after_end = t_end
        return x

    def _a_step(self) -> None:
        if not (self.a_once or self.a_step):
            super()._a_step()

    def poisoned(self) -> list:
        # not the plan's receipts and seeds, nor the units' stage: an earlier step's units are writing them now (a
        # captured step's units poison theirs on the units stream, `SoSide.dispatch`)
        out = self.slot + self.scale + self.z + list(self.ydump or []) + [self.pre["mid"].level, self.pre["a"].level]
        if self.a_step:
            out.append(self.pre["ak"].level)
        if DA.SIDE:
            sd = DA.SIDE[0]
            out += sd.zy + sd.ysv + [sd.zyset.level, sd.ysset.level, sd.yuset.level]
        return out

    def dead_from(self, m: int) -> int:
        """Under DEAD_ONCE_ENV the row from which every row already holds a row past m's z/y records (T after a capture,
        a control's no-write step or a window's start), and from the next step on m, or T after a step that writes
        none; else 0, every row past m written."""
        if not self.dead_once:
            return 0
        c = self.t if self.cap is not None or self.clean_from is None else self.clean_from
        self.clean_from = self.t if self.flags & DO.F_NW_TILES else m
        self.dead_c = c
        return c

    def before(self, m: int) -> None:
        self.t_b0 = time.perf_counter()
        if self.root_lag == 1 or (self.lag is not None and self.ev_lag.query()):
            self._lag_finish()
        if not self.open:
            self._start()
        if not self.overlap and DA.SIDE and DA.SIDE[0].spill is not None:
            DA.SIDE[0].spill.wait_z(0)
        super().before(m)
        self.m_of[self.s] = self.m_last = m
        if self.cap is not None:
            self.cap_of[self.s] = self.cap_at[self.s] = self.cap
            self.caps[self.cap].update(k=self.k, pipe=self.pipe, coins=self.coins, ix=self.ix(self.s))
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
        self._lag_finish()
        if self.overlap and DA.SIDE:
            DA.SIDE[0].ov.drain()

    def after(self) -> None:
        if self.overlap:
            return self._after_ov()
        self._lag_finish()
        sd: SoSide = DA.SIDE[0]
        d = self.depth - 1
        main = torch.cuda.current_stream()
        self.phase("forward_wait")
        if self.deferred is not None:
            self._forward_wait_deferred(main)
        main.synchronize()
        t_fwd = time.perf_counter()
        t_late = 0.0
        if self.deferred is not None and self.coin_async == 1:
            t_late = self._late_dispatch()
        s, m, cap, pre, ev, kinds = self.s, self.m, self.cap, self.pre, self.ev, self.round_kinds
        zs = None
        if self.z_side:
            if self.tree_streams is None:
                raise RuntimeError(f"{ZSIDE_ENV}=1 needs the tree streams (-pt under the serving priority, no -ov)")
            zs = self.tree_streams[0]
            self.phase("z_tables")
            self.ev_z[0].record()
            zs.wait_event(self.ev_z[0])
            with torch.cuda.stream(zs):
                self.c.build(pre["z"], self.stepv[0:1])
                self.ev_z[1].record(zs)
        if self.a_once or self.a_step:
            self.phase("a_tables")
            top = self.t if self.dead_c is None else self.dead_c
            if not self.flags & DO.F_NW_TILES and top > m:
                for x in self.slot_groups.values():
                    x[:, m:top].zero_()
                self.scale_all[:, m:top].zero_()
            if self.a_step:
                self.a_step_build()
            else:
                self.c.build(pre["a"], self.stepv[0:1])
            self._poll_deferred()
        self.phase("stash")
        if sd.spill is not None:
            sd.spill.wait_a(s % self.depth)
        sd.stash(s)
        self.stashed.add(s)
        if sd.spill is not None:
            self.phase("spill")
            sd.spill.put(s, m, sd.ready[s % self.depth])
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
        self._poll_deferred()
        self.phase("z_tables")
        if zs is not None:
            main.wait_event(self.ev_z[1])
        else:
            self.c.build(pre["z"], self.stepv[0:1])
        for f in self.before_mid:
            f()
        if self.deferred is not None:
            t_late = self._late_dispatch()
        tail = self.tail is not None and cap != "fault_zlate" and self.zlate is None
        zc = tail and self.tail.root_host is not None
        if tail:
            self.tail.pre(self.stepv[0:1])
        if zc:
            self._ybad_copy(sd, put=True)
        self.phase("receipt_leaf")
        o = pre["mid"].level_off[kinds.index("receipt-step")]
        lv = pre["mid"].level[o:o + 8 * self.L]
        j = s - d
        p = sd.dispatched.pop(j, None) if j >= self.s0 else None
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
        zi, yi = kinds.index("z-step"), kinds.index("y-step")
        if self.hmark:
            self.ev_h[0].record(self.hm_stream)
        if tail:
            self.tail.levels()
            self.ev_w[2].record()
        else:
            self.c.build(pre["mid"], self.stepv[0:1])
            self.ev_w[2].record()
            rl = pre["round"].level
            if cap == "fault_zlate":
                self.zlate = (rl[8 * zi:8 * zi + 8].clone(), rl[8 * yi:8 * yi + 8].clone())
                for i in (zi, yi):
                    rl[8 * i:8 * i + 8].view(torch.uint8).fill_(DF.POISON)
            elif self.zlate is not None:
                rl[8 * zi:8 * zi + 8].copy_(self.zlate[0])
                rl[8 * yi:8 * yi + 8].copy_(self.zlate[1])
                self.zlate = None
            self.c.build(pre["round"], self.stepv[0:1])
        self.ev_w[3].record()
        prev = self.cap_of.get(s - 1)
        late = self.cap_of.pop(j, None) if j >= self.s0 else None
        held = (cap in self.caps) or (late in self.caps) or (prev == "fault_zlate" and prev in self.caps)
        lag = h_sync = c_root = c_disp = None
        if self.root_lag and not held and s != self.end_at:
            self.phase("root_lag")
            if self.lag_ybad is None:
                self.lag_ybad = torch.empty(sd.ybad.shape, dtype=sd.ybad.dtype).pin_memory()
            self.lag_root.copy_(pre["round"].root(0), non_blocking=True)
            self.lag_ybad.copy_(sd.ybad, non_blocking=True)
            self.ev_lag.record()
            lag = {"s": s, "m": m, "i": self.ix(s), "cap": cap, "waited": waited}
            t_root = t_sent = time.perf_counter()
        else:
            self.phase("root_read")
            ju = s + 1 - d
            if (sd.preq and not held and ju in self.stashed and ju != self.end_at and self.cap_of.get(ju) is None
                    and ju in self.m_of) and sd.prequeue(ju, self.m_of[ju]):
                self._salt_early(sd, ju)
            # ybad rides the root read's sync rather than a nonzero's own
            if not zc:
                self._ybad_copy(sd)
            if self.hmark:
                self.ev_h[1].record(self.hm_stream)
            if zc:
                # the stream past its wait on the units, then a spin on the tail's own word: the wake-up's latency
                # rides the tail's launch
                (self.ev_w[1] if waited else self.ev_w[2]).synchronize()
                if self.hmark:
                    h_sync = time.perf_counter()
                root = self.tail.root(0)
            else:
                root = unwords(pre["round"].root(0))
            if self.hmark:
                self.ev_h[4].record(self.hm_stream)
            t_root = time.perf_counter()
            if self.hmark:
                c_root = time.thread_time()
            self.phase("record_host")
            i = self.pipe.send(root)
            if i != self.ix(s):
                raise RuntimeError(f"step {s}'s round went out as round {i}, not {self.ix(s)}")
            t_sent = time.perf_counter()
            # a held step's captures read the refused calls now; any other step reads them once its units are out
            bad = self._y_refused(s, cap) if held else None
        if held:
            leaves = [unwords(pre["round"].leaf(0, k)) for k in range(len(kinds))]
            if cap in self.caps:
                self.caps[cap].update(
                    root=root, index=i, a=[keep(x) for x in self.slot], sa=[keep(x) for x in self.scale],
                    z=[keep(x) for x in self.z], y=[keep(x[:m]) for x in self.ydump] if self.ydump else None,
                    a_roots=[unwords(pre["mid"].leaf(0, w)) for w in range(self.L)],
                    admit_roots=[unwords(pre["mid"].leaf(1, w)) for w in range(self.L)],
                    z_roots=[unwords(pre["mid"].leaf(2, w)) for w in range(self.L)], leaves=leaves,
                    b_roots=[unwords(self.bset.root(w)) for w in range(self.L)],
                    zy=[keep(x) for x in sd.zy], ysv=[keep(x) for x in sd.ysv], y_root_bad=bad,
                    y_roots=[unwords(pre["mid"].leaf(yi, w)) for w in range(self.L)],
                    yu_roots=[unwords(sd.yuset.root(w)) for w in range(self.L)])
            if late in self.caps:
                self.caps[late].update(late_s=s, late_index=i, late_leaves=leaves, late_root=root)
            if prev == "fault_zlate" and prev in self.caps:
                self.caps[prev].update(next_leaves=leaves)
        self.phase("coin_wait")
        ju = s + 1 - d
        t_wait = t_coin = t_disp = time.perf_counter()
        if ju in self.stashed:
            if self.coin_async and s != self.end_at and self.pipe.opening(self.ix(ju), wait=False) is None:
                self.deferred = ju
                self.deferrals["deferred"] += 1
            else:
                self.stashed.discard(ju)
                t_coin = self.dispatch_units(ju)
                t_disp = time.perf_counter()
                if self.hmark:
                    c_disp = time.thread_time()
                    self.ev_h[2 + s % 2].record(self.hm_stream)
                    self.h_flagged = {s - 1, s} & (self.h_flagged | {s})
        if lag is None and not held:
            self._y_refused(s, cap)
        units_wait = self.ev_w[0].elapsed_time(self.ev_w[1]) if waited and lag is None else None
        # the manifest row's copies queue behind the root read and the units' dispatch, off the step's critical path
        self.put_row(s, pre["mid"], pre["round"])
        t_end = time.perf_counter()
        self.phase(None)
        ph = {"m": m, "forward_ms": 1e3 * (t_fwd - self.t_before), "trees_ms": 1e3 * (t_root - t_fwd),
              "record_host_ms": 1e3 * (t_sent - t_root), "captures_ms": 1e3 * (t_wait - t_sent),
              "coin_wait_ms": 1e3 * (t_coin - t_wait),
              "dispatch_ms": 1e3 * (t_disp - t_coin), "between_ms": 1e3 * (t_end - t_fwd),
              "coin_late_ms": 1e3 * t_late}
        if self.a_step:
            ph["a_tables_ms"] = self.ev_a[0].elapsed_time(self.ev_a[1])
        if units_wait is not None:
            ph["units_wait_ms"] = units_wait
            # under TAIL=2 the root came from the tail's own word, which the host can read before the stream is past
            # ev_w[2] and ev_w[3]: a step whose stream isn't there yet has no tail times
            if not zc or self.ev_w[3].query():
                ph["tail_mid_ms"] = self.ev_w[1].elapsed_time(self.ev_w[2])
                ph["tail_round_ms"] = self.ev_w[2].elapsed_time(self.ev_w[3])
            if self.hmark:
                ph["h_tail_ms"] = self.ev_w[1].elapsed_time(self.ev_h[0])
                ph["h_read_ms"] = self.ev_w[3].elapsed_time(self.ev_h[1])
                ph["h_ret_ms"] = self.ev_w[3].elapsed_time(self.ev_h[4])
                ph["h_udone_ms"] = sd.ev_u[p][1].elapsed_time(self.ev_w[1])
                if s - 1 in self.h_flagged:
                    ph["h_flag_ms"] = self.ev_h[2 + (s - 1) % 2].elapsed_time(sd.ev_u[p][0])
                if h_sync is not None:
                    ph["h_spin_ms"] = 1e3 * (t_root - h_sync)
                if c_disp is not None and c_root is not None:
                    # main's CPU time against its wall time from the root's return to the units' flag: less CPU than
                    # wall is a wait (the GIL, the scheduler), not code
                    ph["h_cpu_ms"] = 1e3 * (c_disp - c_root)
                    ph["h_wall_ms"] = 1e3 * (t_disp - t_root)
                    ph["h_salt_ms"] = 1e3 * (self.h_t[0] - t_coin)
                    ph["h_side_ms"] = 1e3 * (t_disp - self.h_t[1])
            ph["units_gpu_ms"] = sd.ev_u[p][0].elapsed_time(sd.ev_u[p][1])
            q = sd.prev_of.get(p)
            if q is not None:
                ph["units_idle_ms"] = sd.ev_u[q][1].elapsed_time(sd.ev_u[p][0])
        row = {**ph, **self.host_ms()}
        self.phases.append(row)
        if lag is None:
            self.round_log.append({"s": s, "m": m, "root": root.hex()[:32]})
        else:
            self.lag = {**lag, "row": row}
        self.s += 1
        self.steps += 1
        self._step_end(s)

    def _ybad_copy(self, sd, put: bool = False) -> None:
        """sd.ybad into ybad_pin on the serving stream: a copy, or (put) h1c's kernel stores, which no other stream's
        transfers hold up."""
        if self.ybad_pin is None:
            self.ybad_pin = torch.empty(sd.ybad.shape, dtype=sd.ybad.dtype).pin_memory()
        if put:
            self.c.put(sd.ybad.contiguous(), self.ybad_pin)
        else:
            self.ybad_pin.copy_(sd.ybad, non_blocking=True)

    def _y_refused(self, s: int, cap: str | None) -> list[int]:
        """Step s's calls whose y root the commit refused, from the pinned copy that rode its root read."""
        bad = self.ybad_pin.nonzero().flatten().tolist()
        if bad:
            DA.SIDE[0].y_refused.append({"s": s, "capture": cap, "calls": len(bad)})
        return bad

    def _step_end(self, s: int) -> None:
        if s == self.end_at:
            self.end()
        elif self.ix(self.s) >= self.layout_rounds:
            raise RuntimeError(f"the window's {self.layout_rounds} rounds are spent")

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
        if sd.spill is not None:
            sd.spill.wait_a(s % D)
            sd.spill.wait_z(s % ov.R)
        if cap is not None:
            ov.poison(s % ov.R)
        sd.stash(s)
        ov.copy(s)
        if sd.spill is not None:
            sd.spill.put(s, m, ov.ev["copied"][s % Ov.EV])
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
        self._step_end(s)

    def stats(self) -> dict:
        self._lag_finish()
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
        if self.open:
            self._harvest()
        self._order()
        rtt, self.rtts = sorted(self.rtts), []
        waits, self.coin_waits = sorted(self.coin_waits), []
        late, self.late_waits = sorted(self.late_waits), []
        deferrals, self.deferrals = self.deferrals, dict.fromkeys(self.deferrals, 0)
        lagw, self.lag_waits = sorted(self.lag_waits), []
        lagn, self.lag_n = self.lag_n, 0
        sent = list(pipe.roots)
        doc["salt"] = "round (the step's own coin)"
        doc["so"] = {"order": ORDER, "depth": self.depth, "units_lag_steps": self.depth - 2,
                     "receipts_round_lag": self.depth - 1, "a_tables": "once a step (not ruled in)" if self.a_once
                     else ("a/<K>@<s> once a step after the forward (DITHER_A_STEP: ruled in, core's form pending)"
                           if self.a_step else "per call, in the forward"),
                     "coins": self.coins.label, "coins_record": self.coins.record(),
                     "coin_rtt_ms": {"n": len(rtt), "p50": _q(rtt, 0.5), "p90": _q(rtt, 0.9), "max": _q(rtt, 1.0)},
                     "coin_wait_ms": {"n": len(waits), "p50": _q(waits, 0.5), "p90": _q(waits, 0.9), "max": _q(waits, 1.0)},
                     "coin_async": {"on": self.coin_async, **deferrals,
                                    "late_wait_ms": {"n": len(late), "p50": _q(late, 0.5), "max": _q(late, 1.0)}},
                     "z_side": self.z_side,
                     "root_lag": {"on": self.root_lag, "steps": lagn,
                                  "wait_ms": {"p50": _q(lagw, 0.5), "p90": _q(lagw, 0.9), "max": _q(lagw, 1.0)}},
                     "order_reason": self.order_why, "windows": self.k + 1, "window_open": self.open,
                     "coins_back": len(pipe.openings), "rounds_sent": len(sent),
                     # the challenger's log holds these round roots in this order (its published session's record)
                     "roots_sha512": hashlib.sha512(b"".join(sent)).hexdigest(),
                     "window_end": self.end_stats(),
                     "dead_once": self.dead_once, "put_async": self.put_async,
                     "serving_stream_priority": torch.cuda.current_stream().priority,
                     "units_stream_priority": DA.SIDE[0].side.priority if DA.SIDE else None,
                     "capture_stream_priority": getattr(torch.cuda.graph, "so_priority", None)}
        if ov is not None:
            doc["so"]["overlap"] = {"order": OV_ORDER, "ring": ov.R, "rounds_built": ov.sent_upto + 1,
                                    "units_dispatched": ov.dispatched_upto + 1,
                                    "serving_stream_priority": torch.cuda.current_stream().priority}
        return doc

    def end_stats(self) -> dict:
        """The windows ended since the last call: each end's time (in the generate that served its last step), split,
        and over its steps."""
        if not self.window_end:
            return {"on": False, "status": NO_END}
        ends, self.ends_from = self.ends[self.ends_from:], len(self.ends)
        starts, self.starts_ms = sorted(self.starts_ms), []
        ms = sorted(e["end_ms"] for e in ends)
        split = {k: _q(sorted(e["split_ms"][k] for e in ends), 0.5) for k in (ends[0]["split_ms"] if ends else ())}
        return {"on": True, "order": END_ORDER, "windows_ended": len(ends), "steps": [e["steps"] for e in ends],
                "end_ms": {"n": len(ms), "p50": _q(ms, 0.5), "max": _q(ms, 1.0)}, "split_ms_p50": split,
                "amortized_ms_per_step": [round(e["amortized_ms_per_step"], 4) for e in ends],
                "start_ms": {"n": len(starts), "p50": _q(starts, 0.5), "max": _q(starts, 1.0)},
                "ends": [{k: e[k] for k in ("k", "name", "fault", "s0", "steps", "tail", "units_pending_at_end",
                                            "final_index")} for e in ends[-4:]],
                "spill": DA.SIDE[0].spill.stats() if DA.SIDE and DA.SIDE[0].spill is not None else None}


def _q(xs: list, f: float):
    return round(xs[min(len(xs) - 1, int(f * len(xs)))], 3) if xs else None


def rounds_of() -> int:
    """A window's round coins: the session's layout off the node (`$DITHER_ROUNDS`, which its open reserved), else
    DF.ROUNDS."""
    r = settings.get(ROUNDS_ENV)
    return DF.ROUNDS if r is None else r


class SoSide(DA.Side):
    """dither-a's serving-stream GEMM state with the forward in the statement's order (no seeds, no units) and the
    units of a step dispatched after its coin (`dispatch`) on a ring of the window's depth (`stash`)."""

    in_order = True

    def _j10r_part(self) -> str | None:
        return F.j10r_part(self.j10r_m, self.skip_dead, self.win.t) if getattr(self, "j10r", False) else None

    def unadopted(self) -> tuple[str, ...]:
        return tuple(x for x, on in (("ab", self.a_once), ("ov", self.ov is not None),
                                     ("j10r-off", self._j10r_part() == "j10r-off")) if on)

    def adopted(self) -> tuple[str, ...]:
        return ("j10r",) if self._j10r_part() == "j10r" else ()

    def __init__(self, win: SoWindow, unit: str, h1u: str, ao: str, pt: bool):
        if unit not in DA.KONE or ao not in AO:
            raise ValueError(f"dither-a-so's unit: one of {DA.KONE}, not {unit!r}; its A tables: one of {tuple(AO)}, not {ao!r}")
        super().__init__(win, unit, h1u, "zy", "", pt)
        dev, L, D = win.dev, win.L, win.depth
        self.so_ao, self.a_once, self.noise = ao, ao.startswith("b"), ao.endswith("n")
        self.a_percall = not (self.a_once or win.a_step)
        if self.a_percall:
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
        self.salt_np = [x.numpy() for x in self.salt_host]
        self.rroots = [torch.zeros(8 * L, dtype=torch.int64, device=dev) for _ in range(D)]
        self.ready = [torch.cuda.Event() for _ in range(D)]
        self.udone = [torch.cuda.Event() for _ in range(D)]
        # each slot's units on the units stream (start, end), and the slot dispatched before it: the step's
        # units_gpu_ms and the stream's units_idle_ms before them, read once round j + D - 1 has waited for them
        self.ev_u = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(D)]
        self.prev_of: dict[int, int | None] = {}
        self.last_p: int | None = None
        self.tables = [win.r.plan([Call(self.ring_slot[p][w], win.wt[w], win.offs[w], self.ring_aroots[p].data_ptr() + 64 * w,
                                        self.ring_live[p], root_b=win.bset.roots.data_ptr() + 64 * w) for w in range(L)],
                                  seeds=win.plan.seeds, out=win.plan.out) for p in range(D)]
        self.rdead = settings.get(RDEAD_ENV)
        if self.rdead:
            self.r_live, nrb = torch.empty_like(win.live), win.t // 32
            win.rset = win.c.set([dataclasses.replace(t, xd=H1CM.dead_x([bytes(t.length)] * (t.count // nrb), dev),
                                                      rlive=self.r_live, rbw=t.count // nrb) for t in win.rset.tabs])
        self.a_off = win.pre["mid"].level_off[0]
        self.dispatched: dict[int, int] = {}
        self.unit_steps = 0
        self.graph = settings.get(GRAPH_ENV)
        self.fused = settings.get(FUSED_ENV)
        self.units_smq, self.units_smd, self.units_cap = smq_of(settings.get(UNITS_SMQ_ENV) or "0")
        self.units_form = settings.get(UNITS_ENV)
        self.j10r = self.units_form == "j10r"
        self.j10r_m = settings.get(J10R_M_ENV)
        self.j10rc_min_t = settings.get(J10RC_MIN_T_ENV)
        if self.units_form not in UNITS_FORMS:
            raise ValueError(f"${UNITS_ENV}={self.units_form}: one of {tuple(UNITS_FORMS)}")
        if self.j10rc_min_t and not self.j10r:
            raise ValueError(f"${J10RC_MIN_T_ENV}: needs ${UNITS_ENV}=j10r")
        self._check_smq(self.units_smq, self.units_smd, f"${UNITS_SMQ_ENV}={settings.raw(UNITS_SMQ_ENV)}",
                        self.units_cap)
        self.skip_dead = not self.j10r or settings.get(SKIP_DEAD_ENV)
        if settings.is_set(SKIP_DEAD_ENV) and not self.j10r:
            raise ValueError(f"${SKIP_DEAD_ENV}: needs ${UNITS_ENV}=j10r (today's units are PouwH1Tile_v4's, whose dead "
                             "tile is always the zero receipt)")
        if not self.skip_dead and self.rdead:
            raise ValueError(f"${UNITS_ENV}=j10r without ${SKIP_DEAD_ENV}=1 computes a dead row block's units in full, "
                             f"not the zero receipt ${RDEAD_ENV}=1 hashes those rows as: set ${RDEAD_ENV}=0, or "
                             f"${SKIP_DEAD_ENV}=1")
        self.units_live = self.ring_live
        if not self.skip_dead:
            all_live = torch.ones_like(win.live)
            self.units_live = [all_live] * D
            self.tables = [win.r.plan([dataclasses.replace(c, live=all_live) for c in tab.calls], seeds=win.plan.seeds,
                                      out=win.plan.out) for tab in self.tables]
        if self.j10r:
            import j10r_so as JS
            JS.rows(self.j10r_m, win.t, win.t)
            fns = ("kh_seeds_j10r_h1r", "kh_units_multi_j10r", "kh_ucallr_bytes", "kh_j10r_rows_bytes", "kh_h1rcall_bytes",
                   *(("kh_units_multi_j10r_sm",) if self.units_smq else ()),
                   *(("kh_units_multi_j10rc", "kh_j10r_ws_bytes") if self.j10rc_min_t else ()))
            missing = [f for f in fns if not hasattr(self.h, f)]
            if not self.fused or missing or settings.get(DA.LANE_REFS_ENV) or not settings.get(JS.TREE_ENV):
                raise ValueError(f"${UNITS_ENV}=j10r: needs ${FUSED_ENV}=1, J10R's tree at ${JS.TREE_ENV} (sbench.sh's "
                                 f"J10R_SHA={JS.J10R_SHA}), no ${DA.LANE_REFS_ENV} (its lane records are today's form) "
                                 f"and an h1u build with {', '.join(fns)} (missing: {missing})")
            self.h.kh_h1rcall_bytes.restype = C.c_uint
            if int(self.h.kh_h1rcall_bytes()) != int(win.r.lib.h1r_call_bytes()):
                raise RuntimeError(f"h1u.cu's H1RCall is {int(self.h.kh_h1rcall_bytes())} bytes, h1r.cu's H1Call "
                                   f"{int(win.r.lib.h1r_call_bytes())}")
            vp, u32 = C.c_void_p, C.c_uint
            self.h.kh_seeds_j10r_h1r.argtypes = [vp, C.c_int, C.c_int, vp, vp, vp, vp]
            self.h.kh_units_multi_j10r.argtypes = [vp, u32, u32, u32, vp, C.c_int, vp]
            if self.j10rc_min_t:
                self.h.kh_units_multi_j10rc.argtypes = [vp, u32, u32, u32, vp, C.c_int, vp]
        if self.fused:
            self.mtab, self.m_units = self._multi_tables()
            vp, u32 = C.c_void_p, C.c_uint
            for f in ("kh_units_multi_sm", "kh_units_multi_j10r_sm", "kh_units_multi_j10rc_sm"):
                if hasattr(self.h, f):
                    getattr(self.h, f).argtypes = [vp, u32, u32, u32, vp, u32, u32, vp, vp]
            for f in ("kh_units_multi_smc", "kh_units_multi_j10r_smc"):
                if hasattr(self.h, f):
                    getattr(self.h, f).argtypes = [vp, u32, u32, u32, vp, u32, u32, u32, vp, vp]
            words = int(self.h.kh_smc_words()) if hasattr(self.h, "kh_smc_words") else 1
            self.unit_next = torch.zeros(words, dtype=torch.int32, device=dev)
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.warm: set[int] = set()
        self.preq = settings.get(PREQ_ENV)
        self.queued: dict[int, tuple[int, int]] = {}
        # step -> the salt `stage_salt` wrote into its ring slot's pinned words ahead of its dispatch
        self.staged: dict[int, bytes] = {}
        self.preq_gen = 0
        if self.preq:
            from cuda.bindings import driver as cu
            self.cu = cu
            hp = self._cu_ok(cu.cuMemHostAlloc(4 * D, cu.CU_MEMHOSTALLOC_DEVICEMAP | cu.CU_MEMHOSTALLOC_PORTABLE))
            self.flag_dev = int(self._cu_ok(cu.cuMemHostGetDevicePointer(hp, 0)))
            self.flag_host = (C.c_uint32 * D).from_address(int(hp))
            for p in range(D):
                self.flag_host[p] = 0
        self.capture_ms: list[float] = []
        if CONF.get("serve_prio") and pt and not win.overlap:
            win.tree_streams = tuple(torch.cuda.Stream(device=dev, priority=-1) for _ in range(2))
        self.ov = Ov(win, self, win.overlap) if win.overlap else None
        self.spill = Spill(win, self, win.spill_mode) if win.window_end and win.spill_mode else None

    def _check_smq(self, q: int, den: int, what: str, cap: int = 0) -> None:
        f = ("kh_units_multi_j10r_sm" if self.j10r else "kh_units_multi_sm") + ("c" if cap else "")
        if q and not (self.fused and 1 <= q <= den <= SMQ_DEN_MAX and 0 <= cap <= 64 and hasattr(self.h, f)):
            raise ValueError(f"{what}: needs ${FUSED_ENV}=1, 1 <= q <= den <= {SMQ_DEN_MAX}, cap <= 64 and an h1u build "
                             f"with {f}")
        if q and self.j10rc_min_t and (cap or not hasattr(self.h, "kh_units_multi_j10rc_sm")):
            raise ValueError(f"{what} with ${J10RC_MIN_T_ENV}: needs no cap (kh_units_multi_j10rc has no capped form) "
                             "and an h1u build with kh_units_multi_j10rc_sm")

    def set_units_smq(self, q: int, den: int = 8, cap: int = 0) -> None:
        """The units launch on the SMs whose id % den < q, at most cap CTAs on each (0: no cap), from the next dispatch
        on (q 0: kh_units_multi's every SM), each (ring slot, m) graph captured again at its next step: a screen of the
        SM share inside one engine (sbench's SBENCH_SMQ_SWEEP); a served row's share is $DITHER_SO_UNITS_SMQ's, fixed at
        the build."""
        self._check_smq(q, den, f"set_units_smq({q}, {den}, {cap})", cap)
        torch.cuda.synchronize()
        self.units_smq, self.units_smd, self.units_cap = q, den, cap if q else 0
        self.graphs.clear()

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
        if self.a_percall:
            self.fork.record(main)
            i = w % NA
            ast = self.astreams[i]
            ast.wait_event(self.fork)
            rc = win.a_call(w, win.slot[w], win.scale[w], ast.cuda_stream)
            self.a_used.add(i)
        self.gemm(w, xq, s_a, wt, s_w, y)
        self.ycommit(w, y, s_w)
        if rc:
            raise RuntimeError(f"dither-a-so call {w} (M={M}): its A table's launch returned {rc}")
        if self.a_percall and self.a_used and (w == win.L - 1 or (w % 4 == 0 and DA._piecewise())):
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

    def _multi_tables(self) -> tuple[list[torch.Tensor], int]:
        """$DITHER_SO_FUSED's kh_units_multi tables, one a ring slot: each call's UCall (its slot's A, its B, seeds and
        receipts as `so_args`; under J10-R a UCallR, with its row workspace), the K = 14336 calls (three digits, the
        longest units) first so they start first; and the units in all."""
        win, h = self.win, self.h
        if not hasattr(h, "kh_units_multi"):
            raise RuntimeError(f"${FUSED_ENV}=1: this h1u build has no kh_units_multi")
        vp, u32 = C.c_void_p, C.c_uint
        h.kh_units_multi.argtypes = [vp, u32, u32, u32, vp, vp]
        h.kh_digits.argtypes = [u32]
        rows = [("rws", "<u8")] if self.j10r else []
        rec = np.dtype([("A", "<u8"), ("B", "<u8"), ("seeds", "<u8"), ("out", "<u8"), *rows, ("N", "<u4"), ("K", "<u4"),
                        ("u0", "<u4"), ("md", "<u4")])
        size, what = ((h.kh_ucallr_bytes, "kh_units_multi_j10r's UCallR") if self.j10r
                      else (h.kh_ucall_bytes, "kh_units_multi's UCall"))
        size.restype = u32
        if int(size()) != rec.itemsize:
            raise RuntimeError(f"{what} is {int(size())} bytes, not {rec.itemsize}")
        md = [int(h.kh_digits(k)) for _, k in win.nk]
        order = sorted(range(win.L), key=lambda w: -md[w])
        rws = []
        if self.j10r:
            # each call's workspace at T rows (the most a layout takes) as the build sizes it (j10r_ws: an H1U_JW
            # build's column tiles and W' too, RC's under $DITHER_SO_J10RC_MIN_T), one workspace for every ring slot:
            # a step's units run on the units stream after the step before's
            nbytes = [JWS.workspace_bytes(h, win.t, n, k, rc=bool(self.j10rc_min_t)) for n, k in win.nk]
            self.j10r_ws = torch.empty(sum(nbytes), dtype=torch.uint8, device=win.dev)
            rws = [self.j10r_ws.data_ptr() + sum(nbytes[:w]) for w in range(win.L)]
        tabs = []
        for p in range(win.depth):
            t = np.zeros(win.L + 1, dtype=rec)
            u0 = 0
            for i, w in enumerate(order):
                n, k = win.nk[w]
                t[i] = (self.ring_slot[p][w].data_ptr(), win.wt[w].data_ptr(),
                        win.plan.seeds.data_ptr() + 8 * 16 * self.seed0[w], self.stage.data_ptr() + 4 * win.plan.offs[w][0],
                        *rws[w:w + 1], n, k, u0, md[w])
                u0 += (win.t // 32) * (n // 32)
            t[win.L]["u0"] = u0
            tabs.append(torch.from_numpy(t.view(np.uint8).copy()).to(win.dev))
        return tabs, u0

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
                self.ys.data_ptr(), None, self.ring_scale[p][win.spos[w]].data_ptr(), win.sw[w].data_ptr(),
                self.ring_s1[p].data_ptr(), min(m, win.t))

    def _units(self, p: int, m: int) -> int:
        """The units of the step in ring slot p at m live rows, on the units stream (current): seeds from
        `ring_salt[p]`, each call's units, the commit and the receipt trees, their roots into `rroots[p]`; the first
        launch's nonzero return code, else 0.  Every launch reads only ring slot p and buffers fixed at the build, so the same launches
        replay as one graph (`dispatch`)."""
        win, st, L = self.win, self.side, self.win.L
        tab = self.tables[p]
        if self.j10r:
            rc = self.h.kh_seeds_j10r_h1r(tab.table.data_ptr(), L, tab.max_units, self.ring_salt[p].data_ptr(),
                                          self.ring_base[p].data_ptr(), win.plan.seeds.data_ptr(), st.cuda_stream)
        else:
            rc = win.r.lib.h1r_launch(tab.table.data_ptr(), L, tab.max_units, tab.ks, self.ring_salt[p].data_ptr(),
                                      self.ring_base[p].data_ptr(), win.plan.seeds.data_ptr(), self.stage.data_ptr(), 0, 0,
                                      st.cuda_stream)
        if self.noise:
            self.seeded.record(st)
        if self.j10r:
            import j10r_so as JS
            M = JS.rows(self.j10r_m, win.t, m)
            j10rc = self.j10rc_min_t and (min(m, M) if self.skip_dead else M) >= self.j10rc_min_t
            if j10rc and self.units_smq:
                rc = rc or self.h.kh_units_multi_j10rc_sm(self.mtab[p].data_ptr(), L, self.m_units, M,
                                                          self.units_live[p].data_ptr(), self.units_smq, self.units_smd,
                                                          self.unit_next.data_ptr(), st.cuda_stream)
            elif j10rc:
                rc = rc or self.h.kh_units_multi_j10rc(self.mtab[p].data_ptr(), L, self.m_units, M,
                                                       self.units_live[p].data_ptr(), 0, st.cuda_stream)
            elif self.units_cap:
                rc = rc or self.h.kh_units_multi_j10r_smc(self.mtab[p].data_ptr(), L, self.m_units, M,
                                                          self.units_live[p].data_ptr(), self.units_smq, self.units_smd,
                                                          self.units_cap, self.unit_next.data_ptr(), st.cuda_stream)
            elif self.units_smq:
                rc = rc or self.h.kh_units_multi_j10r_sm(self.mtab[p].data_ptr(), L, self.m_units, M,
                                                         self.units_live[p].data_ptr(), self.units_smq, self.units_smd,
                                                         self.unit_next.data_ptr(), st.cuda_stream)
            else:
                rc = rc or self.h.kh_units_multi_j10r(self.mtab[p].data_ptr(), L, self.m_units, M,
                                                      self.units_live[p].data_ptr(), 0, st.cuda_stream)
        elif self.fused and self.units_cap:
            rc = rc or self.h.kh_units_multi_smc(self.mtab[p].data_ptr(), L, self.m_units, win.t,
                                                 self.ring_live[p].data_ptr(), self.units_smq, self.units_smd,
                                                 self.units_cap, self.unit_next.data_ptr(), st.cuda_stream)
        elif self.fused and self.units_smq:
            rc = rc or self.h.kh_units_multi_sm(self.mtab[p].data_ptr(), L, self.m_units, win.t,
                                                self.ring_live[p].data_ptr(), self.units_smq, self.units_smd,
                                                self.unit_next.data_ptr(), st.cuda_stream)
        elif self.fused:
            rc = rc or self.h.kh_units_multi(self.mtab[p].data_ptr(), L, self.m_units, win.t, self.ring_live[p].data_ptr(),
                                             st.cuda_stream)
        for w in range(0 if self.fused else L):
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
            return rc
        n = self.stage.numel()
        DA.dithera_commit[(triton.cdiv(n, 4096),)](self.stage, win.plan.out, self.ring_S[p], n, BLOCK=4096)
        DA.dithera_fault[(1,)](win.plan.out, self.ring_S[p], self.at, L, 1 << self.shift,
                               BLOCK=triton.next_power_of_2(L))
        if self.rdead:
            self.r_live.copy_(self.ring_live[p])
        win.c.build(win.rset, self.ring_stepv[p][0:1])
        self.rroots[p].copy_(win.rset.roots)
        return 0

    @staticmethod
    def _cu_ok(res):
        err, *out = res if isinstance(res, tuple) else (res,)
        if int(err) != 0:
            raise RuntimeError(f"${PREQ_ENV}=1: {err!r}")
        return out[0] if len(out) == 1 else out

    def prequeue(self, j: int, m: int) -> bool:
        """Under $DITHER_SO_PREQ=1, step j's units graph at m live rows queued on the units stream behind a wait on its
        slot's pinned host word, which `dispatch` sets once their salt is written; False (nothing queued) if that
        (ring slot, m) has no graph yet."""
        p = j % self.win.depth
        cg = self.graphs.get((p, m)) if self.preq and self.graph and m in self.warm else None
        if cg is None:
            return False
        self.preq_gen += 1
        st = self.side
        with torch.cuda.stream(st):
            st.wait_event(self.ready[p])
            self._cu_ok(self.cu.cuStreamWaitValue32(self.cu.CUstream(st.cuda_stream), self.cu.CUdeviceptr(self.flag_dev + 4 * p),
                                                    self.preq_gen, self.cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ))
            self.ev_u[p][0].record(st)
            self.ring_salt[p].copy_(self.salt_host[p], non_blocking=True)
            cg.replay()
            self.ev_u[p][1].record(st)
            self.udone[p].record(st)
        self.queued[j] = (p, self.preq_gen)
        return True

    def stage_salt(self, j: int, salt: bytes) -> None:
        """Step j's salt into its ring slot's pinned words before its dispatch: the slot's last reader (step j - depth's
        copy) is long done, and its next (step j's, queued behind the flag) reads it only once `dispatch` sets that."""
        self.salt_np[j % self.win.depth][:] = words(salt)
        self.staged[j] = salt

    def dispatch(self, j: int, salt: bytes, m: int, g: dict | None) -> None:
        """Step j's units on the units stream once its ring slot is written (`_units`) under `salt`, whose receipt
        roots go into `rroots` for round j + D - 1 (`udone`); a capture's receipts and roots into `g`.  Unless
        DITHER_SO_GRAPH=0, `_units` runs as a CUDA graph of its (ring slot, m), captured at that pair's first step and
        replayed after, so the host launches one graph a step in place of each call's launches; the first step at each
        m runs eagerly (Triton's compiles and the units' one-time attributes outside any capture).  A capture's poison
        and copies stay outside the graph, and `g["units_graph"]` says which way its units ran."""
        win = self.win
        p, st = j % win.depth, self.side
        if self.staged.pop(j, None) != salt:
            self.salt_np[p][:] = words(salt)
        q = self.queued.pop(j, None)
        if q is not None:
            self.flag_host[q[0]] = q[1]
            if g is not None or q[0] != p:
                raise RuntimeError(f"dither-a-so step {j}'s units were queued ({PREQ_ENV}) in slot {q[0]}, then "
                                   f"dispatched in slot {p} or under a capture")
            self.prev_of[p], self.last_p = self.last_p, p
            self.dispatched[j] = p
            self.unit_steps += 1
            return
        with torch.cuda.stream(st):
            st.wait_event(self.ready[p])
            self.ev_u[p][0].record(st)
            self.ring_salt[p].copy_(self.salt_host[p], non_blocking=True)
            if g is not None:
                for x in (self.stage, win.plan.out, win.plan.seeds, win.rset.level):
                    x.view(torch.uint8).fill_(DF.POISON)
            graph = self.graph and m in self.warm
            if graph:
                cg = self.graphs.get((p, m))
                if cg is None:
                    t0 = time.perf_counter()
                    cg = torch.cuda.CUDAGraph()
                    cg.capture_begin(capture_error_mode="thread_local")
                    try:
                        rc = self._units(p, m)
                    finally:
                        cg.capture_end()
                    if rc:
                        raise RuntimeError(f"dither-a-so step {j}'s units (graph capture): a launch returned {rc}")
                    self.graphs[(p, m)] = cg
                    self.capture_ms.append(round(1e3 * (time.perf_counter() - t0), 3))
                cg.replay()
            elif rc := self._units(p, m):
                raise RuntimeError(f"dither-a-so step {j}'s units: a launch returned {rc}")
            self.warm.add(m)
            if g is not None:
                g["receipts"], g["r_roots"] = keep(win.plan.out), keep(win.rset.roots)
                g["units_graph"] = graph
            self.ev_u[p][1].record(st)
            self.udone[p].record(st)
        self.prev_of[p], self.last_p = self.last_p, p
        self.dispatched[j] = p
        self.unit_steps += 1

    def _j10r_info(self) -> dict:
        import j10r_so as JS
        return {"label": JS.LABEL, "rows_mode": self.j10r_m, "rows": JS.M_MODES[self.j10r_m],
                "slot_rows": self.win.t, "rows_workspace_bytes": self.j10r_ws.numel(),
                "dead_tiles": JS.DEAD[self.skip_dead], "j10rc_min_t": self.j10rc_min_t or None}

    def info(self) -> dict:
        return {**super().info(), "order": ORDER, "depth": self.win.depth, "a_tables": AO[self.so_ao],
                "a_overlap": self.so_ao, "unit_steps": self.unit_steps,
                "units_dispatch": {"graph": self.graph, "graphs": sorted(self.graphs), "capture_ms": self.capture_ms},
                "units_fused": {"one_launch": self.fused, "units": self.m_units if self.fused else None,
                                "sm_share": f"{self.units_smq}/{self.units_smd}" if self.units_smq else None,
                                "sm_cap": self.units_cap or None},
                "units_form": {"form": self.units_form, "noise": UNITS_FORMS[self.units_form],
                               **(self._j10r_info() if self.j10r else {})},
                "receipt_dead_rows_x_once": self.rdead,
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
        """`DitherState.phase`'s NVTX range on this thread (stepprof joins a launch to its own thread's phase)."""
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
            if j < win.s0:
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
            win.put_row(s, st["mid"], st["round"])
            self.done.record(ts)
        with self.phase("ov_root_wait"):
            self.done.synchronize()
        t_root = time.perf_counter()
        root = unwords(self.root_host)
        with self.phase("ov_send"):
            i = win.pipe.send(root)
        if i != win.ix(s):
            raise RuntimeError(f"step {s}'s round went out as round {i}, not {win.ix(s)}")
        t_sent = time.perf_counter()
        bad = self.ybad_host.nonzero().flatten().tolist()
        if bad:
            sd.y_refused.append({"s": s, "capture": cap, "calls": len(bad)})
        caps = win.caps
        prev = win.cap_of.get(s - 1)
        late = win.cap_of.pop(j, None) if j >= win.s0 else None
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
        i = win.ix(s)
        t0 = time.perf_counter()
        with self.phase("ov_coin_wait"):
            opening = pipe.opening(i)
        t_coin = time.perf_counter()
        win.coin_waits.append(1e3 * (t_coin - t0))
        name = win.cap_of.get(s)
        if name == "fault_rearly":
            held = pipe.opening(i - 1)
            salt = win.HT.live_salt(win.coins.commitment, held.coin, held.index, pipe.roots[i - 1])
        else:
            salt = win.HT.live_salt(win.coins.commitment, opening.coin, opening.index, pipe.roots[i])
        g = win.caps.get(name) if name is not None else None
        if g is not None:
            g.update(opening=opening, salt_used=salt, coin_got_at=pipe.got_at.get(i), dispatched_at=time.monotonic())
        m = win.m_of.pop(s)
        if not (s == win.end_at and win.end_fault == "fault_tail"):
            win.salt_of[s] = salt
            win.disp_at[s] = time.monotonic()
            with self.phase("ov_units_dispatch"):
                sd.dispatch(s, salt, m, g)
        self.ph_units[s] = {"coin_wait_ms": 1e3 * (t_coin - t0), "dispatch_ms": 1e3 * (time.perf_counter() - t_coin)}
        self._set(dispatched_upto=s)


class Spill:
    """-we: each step's live A rows and scales and (mode "azy") its z/y records' live row tiles, the values the prover
    opens a drawn unit from after the draw, copied on a stream of their own into pinned host memory from the ring slot
    that holds them, and held until the window's end (`take_window`).  The serving stream rewrites a slot only once its
    copy is out (`wait_a`, `wait_z`): ring slot s mod D at stash s + D; the z/y records at forward s + 1 (under -ov round
    slot s mod R at its copy s + R).  The records' live prefixes go through one device buffer and one transfer.  The
    pool's chunks are never freed: a window's go back to it at its end, a named or off-node window's after `check_end`
    (`release`).  An off-node window's pool is pinned whole when the spill is made, at its open (`reserve`); an
    in-process arm's grows in sbench's warm-up."""

    def __init__(self, win: SoWindow, sd: SoSide, mode: str):
        if mode not in SPILLS:
            raise ValueError(f"dither-a-so -we: a spill is one of {tuple(SPILLS)}, not {mode!r}")
        self.win, self.sd, self.mode, self.zy = win, sd, mode, mode == "azy"
        L = win.L
        self.stream = torch.cuda.Stream(device=win.dev)
        self.a_done = [torch.cuda.Event() for _ in range(win.depth)]
        self.z_done = [torch.cuda.Event() for _ in range(win.overlap or 1)]
        self.a_used, self.z_used = [False] * win.depth, [False] * (win.overlap or 1)
        # call w's A rows: index i of the ring's group of its K
        self.slot_of, at = [], {}
        for _, k in win.nk:
            self.slot_of.append((k, at.get(k, 0)))
            at[k] = at.get(k, 0) + 1
        self.row_block = [(sd.unit_of[w][0], (win.nk[w][0] // sd.unit_of[w][1]) * DA.zy_bytes(sd.unit_of[w]))
                          for w in range(L)]
        self.stage = torch.empty(sum(x.numel() for x in sd.zy), dtype=torch.uint8, device=win.dev) if self.zy else None
        self.src: dict[tuple, list] = {}
        self.free: list[torch.Tensor] = []
        self.chunks: list[torch.Tensor] = []
        self.at, self.n_chunks, self.grown, self.alloc_ms = 0, 0, 0, 0.0
        gb = settings.get(SPILL_MAX_ENV)
        self.max_bytes = int((SPILL_MAX_GB if gb is None else gb) * (1 << 30))
        self.held: dict[int, dict] = {}
        self.log: list[tuple] = []
        self.unfinished = {"a": 0, "z": 0}
        self.reserved = None
        if win.open and not win.end_each and win.end_at is not None:
            self.reserve(win.end_at - win.s + 1)

    def step_bytes(self, m: int) -> list[int]:
        """A step's copies at m rows, in `put`'s order: each K's A rows, the scales, under "azy" the z/y records."""
        sd = self.sd
        out = [g.shape[0] * m * k for k, g in sd.ring_groups[0].items()]
        sa = sd.ring_scale[0]
        out.append(sa.shape[0] * m * sa.element_size())
        if self.zy:
            out.append(sum(self.prefix(w, m) for w in range(self.win.L)))
        return out

    def reserve(self, steps: int) -> None:
        """An off-node window's pool, pinned when the window opens, before any step it times: the chunks `take` fills
        with `steps` steps at m = T, the most a step copies.  Refused there when that passes the bound, rather than
        partway through the window."""
        at, n = SPILL_CHUNK, 0
        for b in self.step_bytes(self.win.t) * steps:
            if at + b > SPILL_CHUNK:
                n, at = n + 1, 0
            at += -(-b // 512) * 512
        if (self.n_chunks + n) * SPILL_CHUNK > self.max_bytes:
            raise RuntimeError(f"dither-a-so -we: an off-node window of {steps} steps holds up to {n * SPILL_CHUNK >> 30} "
                               f"GiB of spill at m = {self.win.t}, past {self.max_bytes >> 30} GiB: open a session of "
                               f"fewer rounds, spill A alone (-wea), or raise ${SPILL_MAX_ENV} with the lease's memory")
        t0 = time.perf_counter()
        self.free += [torch.empty(SPILL_CHUNK, dtype=torch.uint8, pin_memory=True) for _ in range(n)]
        self.n_chunks += n
        self.reserved = {"steps": steps, "m": self.win.t, "chunks": n, "gib": n * SPILL_CHUNK / (1 << 30),
                         "ms": round(1e3 * (time.perf_counter() - t0), 1)}

    def prefix(self, w: int, m: int) -> int:
        """Call w's records' live bytes at m rows: its row tiles below m."""
        ur, per = self.row_block[w]
        return -(-m // ur) * per

    def take(self, n: int) -> torch.Tensor:
        """n bytes of the window's pinned chunks, 512-aligned, a chunk from the pool (or a new one) when the last is full."""
        if not self.chunks or self.at + n > SPILL_CHUNK:
            if n > SPILL_CHUNK:
                raise RuntimeError(f"dither-a-so -we: a spill copy of {n} bytes outgrows a chunk ({SPILL_CHUNK})")
            if self.free:
                c = self.free.pop()
            else:
                if (self.n_chunks + 1) * SPILL_CHUNK > self.max_bytes:
                    raise RuntimeError(f"dither-a-so -we: the spill's pool would pass {self.max_bytes >> 30} GiB at step "
                                       f"{self.win.s}, {len(self.held)} steps into its window: serve a shorter window, "
                                       f"spill A alone (-wea), or raise ${SPILL_MAX_ENV} with the lease's memory")
                t0 = time.perf_counter()
                c = torch.empty(SPILL_CHUNK, dtype=torch.uint8, pin_memory=True)
                self.alloc_ms += 1e3 * (time.perf_counter() - t0)
                self.n_chunks += 1
                self.grown += 1
            self.chunks.append(c)
            self.at = 0
        v = self.chunks[-1][self.at:self.at + n]
        self.at += -(-n // 512) * 512
        return v

    def wait_a(self, p: int) -> None:
        """On the current stream, before ring slot p is rewritten: its last copy out."""
        if self.a_used[p]:
            self.unfinished["a"] += not self.a_done[p].query()
            torch.cuda.current_stream().wait_event(self.a_done[p])

    def wait_z(self, q: int) -> None:
        """On the current stream, before the z/y records (under -ov round slot q) are rewritten: their last gather."""
        if self.zy and self.z_used[q]:
            self.unfinished["z"] += not self.z_done[q].query()
            torch.cuda.current_stream().wait_event(self.z_done[q])

    def put(self, s: int, m: int, ready: torch.cuda.Event) -> None:
        """Step s's copies on the spill's stream once `ready` (ring slot s mod D written; under -ov round slot s mod R
        too)."""
        t0 = time.perf_counter()
        win, sd, st = self.win, self.sd, self.stream
        p, ov = s % win.depth, sd.ov
        q = s % ov.R if ov is not None else 0
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        x: dict = {"m": m, "a": {}, "zy": None}
        nz = 0
        with torch.cuda.stream(st):
            st.wait_event(ready)
            e0.record(st)
            for k, g in sd.ring_groups[p].items():
                h = self.take(g.shape[0] * m * k).view(torch.int8).view(g.shape[0], m, k)
                h.copy_(g[:, :m], non_blocking=True)
                x["a"][k] = h
            sa = sd.ring_scale[p]
            x["sa"] = self.take(sa.shape[0] * m * sa.element_size()).view(sa.dtype).view(sa.shape[0], m)
            x["sa"].copy_(sa[:, :m], non_blocking=True)
            self.a_done[p].record(st)
            self.a_used[p] = True
            if self.zy:
                key = (q, m)
                if key not in self.src:
                    recs = ov.zy_r[q].split(ov.zl) if ov is not None else sd.zy
                    self.src[key] = [r[:self.prefix(w, m)] for w, r in enumerate(recs)]
                nz = sum(self.prefix(w, m) for w in range(win.L))
                torch.cat(self.src[key], out=self.stage[:nz])
                self.z_done[q].record(st)
                self.z_used[q] = True
                x["zy"] = self.take(nz)
                x["zy"].copy_(self.stage[:nz], non_blocking=True)
            e1.record(st)
        na = sum(h.numel() for h in x["a"].values()) + x["sa"].numel() * x["sa"].element_size()
        self.held[s] = x
        self.log.append((na, nz, e0, e1, 1e3 * (time.perf_counter() - t0)))

    def take_window(self) -> dict:
        """The ended window's held steps and their chunks; the next window's from none."""
        w = {"mode": self.mode, "steps": self.held, "chunks": self.chunks}
        self.held, self.chunks, self.at = {}, [], 0
        return w

    def release(self, w: dict | None) -> None:
        if w:
            self.free.extend(w["chunks"])
            w["chunks"], w["steps"] = [], {}

    def stats(self) -> dict:
        """The steps spilled since the last call: bytes, the copies' time on the spill's stream, the serving thread's
        time enqueuing them, how often the serving stream's wait found a copy unfinished, and the pool's growth."""
        done = [x for x in self.log if x[3].query()]
        self.log = [x for x in self.log if not x[3].query()]
        gpu = sorted(x[2].elapsed_time(x[3]) for x in done)
        host = sorted(x[4] for x in done)
        na, nz = sorted(x[0] for x in done), sorted(x[1] for x in done)
        gbps = sorted((x[0] + x[1]) / (1e6 * x[2].elapsed_time(x[3])) for x in done if x[2].elapsed_time(x[3]) > 0)
        doc = {"mode": self.mode, "what": SPILLS[self.mode], "steps": len(done),
               "a_bytes_per_step": _q(na, 0.5), "zy_bytes_per_step": _q(nz, 0.5),
               "copy_gpu_ms": {"p50": _q(gpu, 0.5), "p90": _q(gpu, 0.9), "max": _q(gpu, 1.0)},
               "copy_gb_per_s_p50": _q(gbps, 0.5), "host_ms": {"p50": _q(host, 0.5), "max": _q(host, 1.0)},
               "serving_waits_on_unfinished_copy": dict(self.unfinished),
               "pool_gib": self.n_chunks * SPILL_CHUNK / (1 << 30), "chunks_grown": self.grown,
               "alloc_ms": round(self.alloc_ms, 1), "reserved_at_open": self.reserved}
        self.unfinished = {"a": 0, "z": 0}
        self.grown, self.alloc_ms = 0, 0.0
        return doc


CONF: dict = {}


def side() -> SoSide:
    if not DA.SIDE:
        win = DO._window(torch.device("cuda"))
        DA.SIDE.append(SoSide(win, CONF["unit"], CONF["h1u"], CONF["ao"], CONF["pt"]))
    return DA.SIDE[0]


def patch(t: int, h1c: str, h1o: str, unit: str, h1u: str, ao: str, pt: bool, depth: int, key: bytes | None = None,
          ov: int = 0, serve_prio: bool = False, window_end: bool = False, spill: str | None = "azy") -> None:
    """dither_a's patch with z/y leaves (its op, `dithera::mm`, and the round's y-step and receipt-step leaves), then its
    window this module's (ring depth `depth`; `ov` > 0: -ov with a round ring that deep, and CAPTURES its) and its op's
    calls this module's Side (before the engine is built).  `serve_prio` (or $DITHER_SO_PRIO=1): the engine serves on a
    stream at the A tables' priority, above the trees' and the units' (so the next forward's blocks take the SMs first as
    theirs retire); without -ov the admit tables too, since the serving stream waits on them between steps.
    `window_end`: -we, each window ends (`SoWindow.end`), each step's values held until it under `spill` (SPILLS;
    -we "azy", -wea "a")."""
    global CAPTURES
    if window_end and spill not in SPILLS:
        raise ValueError(f"dither-a-so -we: a spill is one of {tuple(SPILLS)}, not {spill!r}")
    if unit not in DA.KONE or ao not in AO or depth < 2 or ov < 0:
        raise ValueError(f"dither-a-so: unit one of {DA.KONE} (not {unit!r}), A tables one of {tuple(AO)} (not {ao!r}), "
                         f"depth at least 2 (not {depth}), round ring 0 (no -ov) or more (not {ov})")
    DA.patch(t=t, h1c=h1c, h1o=h1o, unit=unit, h1u=h1u, key=key, z="zy", ao="", pt=pt, ls=False)
    serve_prio = serve_prio or settings.get(PRIO_ENV)
    CONF.update(unit=unit, h1u=h1u, ao=ao, pt=pt, depth=depth, ov=ov, serve_prio=serve_prio, window_end=window_end)
    SoWindow.depth, SoWindow.a_once, SoWindow.overlap = depth, ao.startswith("b"), ov
    SoWindow.window_end = window_end
    SoWindow.spill_mode = spill if window_end else None
    if ov:
        CAPTURES = CAPTURES_OV
    if serve_prio:
        # vLLM serves (replays its graphs) on the stream its patched set_stream last recorded
        from vllm.utils import torch_utils  # noqa: F401
        torch.cuda.set_stream(torch.cuda.Stream(priority=-1))
        if settings.get(CAPTURE_PRIO_ENV):
            capture_at(-1)
    DO.DitherState = SoWindow
    DA.side = side


def capture_at(priority: int) -> None:
    """Every `torch.cuda.graph` capture asked of a stream below `priority` (vLLM's) taken on one at it instead; the
    units' graphs (`SoSide.dispatch`, capture_begin on the units stream) keep theirs."""
    base = torch.cuda.graph
    if getattr(base, "so_priority", None) is not None:
        return
    streams: dict[int, torch.cuda.Stream] = {}

    class graph(base):
        so_priority = priority

        def __init__(self, cuda_graph, pool=None, stream=None, *args, **kwargs):
            if stream is None or stream.priority > priority:
                dev = torch.cuda.current_device()
                if dev not in streams:
                    streams[dev] = torch.cuda.Stream(device=dev, priority=priority)
                stream = streams[dev]
            super().__init__(cuda_graph, pool, stream, *args, **kwargs)

    torch.cuda.graph = graph


def stats(win: SoWindow):
    def doc() -> dict:
        return {**win.stats(), **DA.arm_label(), "dither_a": DA.SIDE[0].info() if DA.SIDE else None}
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
    s, pipe, com, ix = g["s"], g["pipe"], g["coins"].commitment, g["ix"]
    o, i, li = g["opening"], g["index"], g["late_index"]
    why = CO.Ch.live_reason(com, o, pipe.label, ix, g["root"])
    salt = win.HT.live_salt(com, o.coin, ix, g["root"])
    order = {"window": g["k"], "round": i, "coin": ix, "receipts_round": li,
             "coin_after_its_round_root": why is None and i == ix, "coin_reason": why,
             "roots_sent_are_captured": len(pipe.roots) > li and pipe.roots[i] == g["root"] and pipe.roots[li] == g["late_root"],
             "receipts_round_after_coin": (li > i and g["coin_got_at"] is not None
                                           and g["coin_got_at"] <= g["dispatched_at"] <= pipe.sent_at.get(li, -1.0)),
             "units_salt_is_the_coins": g["salt_used"] == salt}
    order["ok"] = (order["coin_after_its_round_root"] and order["roots_sent_are_captured"]
                   and order["receipts_round_after_coin"])
    r_round = {"s": g["late_s"], "leaves": g["late_leaves"], "root": g["late_root"]}
    sd = DA.SIDE[0] if DA.SIDE else None
    if getattr(sd, "j10r", False):
        import j10r_so as JS
        doc = JS.check(win, lib_h1o, which, n_cpu_units, {**g, "salt": salt}, r_round, sd.j10r_m, sd.skip_dead)
    else:
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


# -- the window's end ----------------------------------------------------------------------------------------------------------

def check_end(win: SoWindow, name: str, lib_h1o: str) -> dict:
    """Ended window `name` (`SoWindow.end_recs`) against the module docstring's list: its coins; each step's round
    recomputed from its manifest row (fault_zlate's two steps' z-step and y-step leaves expected off); each step's units
    under its coin's salt (fault_rearly's expected not) and dispatched after its coin came back, its receipts' round sent
    after that and before the draw coin, no round after it; the final round from the tail's receipt roots; and the last
    step's receipts against h1r v1 under its coin's salt, `rehashed_calls`' A tables and receipt tables rehashed; under -we
    three steps opened from its spill (`open_spill`).  Under fault_tail and fault_late_receipt, ``conforming`` holds
    when it is refused exactly where its fault is."""
    from verity.core.protocols.pouw.circuit import h1tile as HT
    rec = win.end_recs.get(name)
    if rec is None or rec.get("rows") is None or "last" not in rec:
        return {"which": name, "ok": False, "complete": False, "why": f"no ended window {name!r} with its rows"}
    pipe, com, rows, kinds = rec["pipe"], rec["coins"].commitment, rec["rows"], win.round_kinds
    W, L, D, key, t = win.window, win.L, win.depth, win.key, win.t
    s0, s1, i_f, li, draw_at = rec["s0"], rec["s1"], rec["i_f"], rec["late_index"], rec["draw_at"]
    nr, L8, n, caps = len(kinds), 8 * L, s1 - s0, rec["cap_at"]
    doc: dict = {"which": name, "fault": rec["fault"], "complete": True, "window": rec["k"], "s0": s0, "steps": n,
                 "depth": D, "tail": rec["tail"], "final_index": i_f, "late_index": li}
    mroot = MF.manifest_root(W, rec["k"], rec["open_root"], s0, s1, list(pipe.roots[:i_f + 1]))
    salt_why = CO.Ch.live_reason(com, rec["salt_open"], CO.Wd.SALT, 0, rec["open_root"])
    draw_why = CO.Ch.live_reason(com, rec["draw"], CO.Wd.DRAW, 0, mroot)
    coins = {"source": rec["coins"].source, "salt_after_window_root": salt_why is None, "salt_reason": salt_why,
             "rounds_reason": CO.Wd.rounds_reason(com, pipe.roots, pipe.openings),
             "draw_after_manifest_root": draw_why is None and mroot == rec["mroot"], "draw_reason": draw_why,
             "rounds_cover_steps": i_f == n + 1 and len(rows) == n}
    coins["ok"] = bool(coins["salt_after_window_root"] and coins["rounds_reason"] is None
                       and coins["draw_after_manifest_root"] and coins["rounds_cover_steps"])
    doc["coins"] = coins
    bad = []
    for r in range(n):
        s, row = s0 + r, rows[r]
        leaves = [unwords(row[nr * L8 + 8 * i:nr * L8 + 8 * i + 8]) for i in range(nr)]
        for i, kd in enumerate(kinds):
            if MF.step_leaf(W, s, kd, [unwords(row[i * L8 + 8 * w:i * L8 + 8 * w + 8]) for w in range(L)]) != leaves[i]:
                bad.append((s, kd))
        if MF.round_root(W, s, kinds, leaves) != pipe.roots[r + 1]:
            bad.append((s, "round"))
    want = {(s + d, k) for s, c in caps.items() if c == "fault_zlate" for d in (0, 1) for k in ("z-step", "y-step")}
    doc["rounds_recomputed"] = {"steps": n, "bad": [list(b) for b in bad if b not in want][:8],
                                "expected_bad": sorted(map(list, want)), "expected_found": want <= set(bad)}
    order_bad = []
    for s in range(s0, s1):
        i = s - s0 + 1
        ri = i + D - 1 if s + D - 1 < s1 else i_f
        got, disp, sent = pipe.got_at.get(i), rec["disp_at"].get(s), pipe.sent_at.get(ri)
        if not (got is not None and disp is not None and sent is not None and got <= disp <= sent < draw_at):
            order_bad.append(s)
    after = sorted(i for i, x in pipe.sent_at.items() if x >= draw_at)
    salts_bad = []
    for s, salt in rec["salt_of"].items():
        o = pipe.opening(s - s0 + 1, wait=False)
        if o is None or salt != HT.live_salt(com, o.coin, s - s0 + 1, pipe.roots[s - s0 + 1]):
            salts_bad.append(s)
    rearly = sorted(s for s, c in caps.items() if c == "fault_rearly")
    tail, rr = rec["tail"], rec["rr"]
    want_leaves = [MF.tail_leaf(W, j, rr[j]) for j in tail]
    tail_bad = [j for j, a, b in zip(tail, want_leaves, rec["final_leaves"]) if a != b]
    final_ok = MF.final_root(W, tail, rec["final_leaves"]) == pipe.roots[i_f]
    late_steps = [j for j, a in zip(tail, want_leaves)
                  if li is not None and MF.late_root(W, j, a) == pipe.roots[li] and pipe.sent_at[li] >= draw_at]
    # the last step: its receipts (the units' last output) against h1r v1 on its ring slot's A under its coin's salt
    last, s, i = rec["last"], s1 - 1, n
    o, m = pipe.opening(i, wait=False), last["m"]
    salt = HT.live_salt(com, o.coin, i, pipe.roots[i])
    a_rows = [unwords(rows[n - 1][8 * w:8 * w + 8]) for w in range(L)]
    a_bad, rr_bad = [], []
    t_cpu = time.perf_counter()
    rehash = rehashed_calls(L, rec["draw"].coin)
    a_cpu = DO.a_roots_cpu(win, s, rehash, lambda v: last["a"][v].cpu().numpy(), lambda v: last["sa"][v].cpu().numpy())
    for w in rehash:
        if a_cpu[w] != a_rows[w]:
            a_bad.append(w)
        o_, units, nw = win.plan.offs[w]
        fr = {"window": W, "step": s, "linear": w, "module": win.names[w], "base": s * win.u_step + win.offs[w],
              "units": units, "words": nw, "schema": CM.SCHEMA}
        rd = CM.domain(fr, units, "receipts")
        rrows = CM.receipt_rows(last["receipts"][o_:o_ + units * nw].view(units, nw).cpu().numpy().view(np.uint32))
        if CM.levels(rd, [CM.leaf(rd, key, f"r/{s}/{w}", u, bytes(rrows[u]))[0] for u in range(units)])[-1][0] != rr[s][w]:
            rr_bad.append(w)
    doc["rehashed"] = {"calls": len(rehashed_calls(L, rec["draw"].coin)), "cpu_s": round(time.perf_counter() - t_cpu, 2)}
    sd = DA.SIDE[0] if DA.SIDE else None
    if getattr(sd, "j10r", False):
        import j10r_so as JS
        JS.RefH1R.M, JS.RefH1R.SKIP = JS.rows(sd.j10r_m, t, m), sd.skip_dead
        r = JS.RefH1R(lib_h1o)
        doc["j10r"] = {"label": JS.LABEL, "rows": JS.RefH1R.M, "receipts_reference": "h1r.cu's h1r_launch_j10r (version 1)",
                       "dead_tiles": JS.DEAD[sd.skip_dead]}
    else:
        r = H1R(lib_h1o)
    live = torch.from_numpy((np.arange(t // 32) * 32 < m).astype(np.uint8)).to(win.dev)
    roots = [torch.from_numpy(np.concatenate([np.frombuffer(a_rows[w], ">u8"), np.frombuffer(rec["b_roots"][w], ">u8")])
                              .astype("<u8").view("<i8").copy()).to(win.dev) for w in range(L)]
    plan = r.plan([Call(last["a"][w], win.wt[w], s * win.u_step + win.offs[w], roots[w], live) for w in range(L)])
    r.launch(plan, torch.from_numpy(words(salt)).to(win.dev), version=1)
    torch.cuda.synchronize()
    rc_bad = []
    for w in range(L):
        o_, units, nw = win.plan.offs[w]
        if (plan.receipts(w) != last["receipts"][o_:o_ + units * nw].view(units, nw)).any():
            rc_bad.append(w)
    doc.update(order_bad=order_bad[:8], order_bad_n=len(order_bad), rounds_after_draw=after, salts_bad=salts_bad[:8],
               salts_expected_bad=rearly, tail_leaves_bad=tail_bad, final_root_ok=final_ok,
               receipts_after_draw=late_steps, last_step=s, last_m=m, a_roots_bad=a_bad, receipt_roots_bad=rr_bad,
               receipts_vs_h1r_bad=rc_bad[:8], receipts_vs_h1r_bad_n=len(rc_bad))
    sp = open_spill(win, rec, kinds)
    if sp is not None:
        doc["spill"] = sp
    base = (coins["ok"] and not doc["rounds_recomputed"]["bad"] and doc["rounds_recomputed"]["expected_found"]
            and sorted(salts_bad) == rearly and final_ok and (sp is None or sp["ok"]))
    doc["ok"] = bool(base and not order_bad and not after and not tail_bad and not late_steps and not rc_bad
                     and not a_bad and not rr_bad)
    if rec["fault"] == "fault_tail":
        # the last step's units dropped: refused on its order and every call's receipts, the rest kept
        doc["conforming"] = bool(base and order_bad == [s] and not after and not tail_bad and not late_steps
                                 and len(rc_bad) == L and not a_bad)
    elif rec["fault"] == "fault_late_receipt":
        # the last step's receipt-tail root withheld from the final round and sent after the draw coin: refused on that
        # leaf and that round alone
        doc["conforming"] = bool(base and not order_bad and tail_bad == [s] and late_steps == [s] and after == [li]
                                 and not rc_bad and not a_bad and not rr_bad)
    return doc


def rehashed_calls(L: int, coin: bytes, drawn: int = 8) -> list[int]:
    """The last step's calls whose A table and receipt table `check_end` rehashes on the CPU: the first and last layers'
    four (every linear's shape) and `drawn` the draw coin picks: at most 16 of Llama 3.1 8B's 128, about half a second
    where all 128 take about 4 s (under DO.A_STEP_ENV the A side is every K's table, so all 128 calls' rows)."""
    h = hashlib.sha512(b"verity/kernels-int12/end-rehash/v1" + coin).digest()
    return sorted({*range(min(4, L)), *range(max(0, L - 4), L),
                   *(int.from_bytes(h[8 * j:8 * j + 8], "big") % L for j in range(drawn))})


def open_spill(win: SoWindow, rec: dict, kinds: tuple) -> dict | None:
    """After the draw, the prover's openings from an ended window's spill (`Spill`): three uncaptured steps (its first,
    its last, and one its draw coin picks), each at a call the draw coin picks.  The call's A table rehashed from the
    held rows and scales (zero past m) against its manifest row's A root; its z/y records recomputed from them (z = A B^T
    by int_mm, y = dequant(z) at the slot's s_a, so past m z 0 and y dq(0, +0, s_w)) against the row's z root; and under
    "azy" the held records against both."""
    from verity_vllm.engine import int8_exact as E
    sp = rec.get("spill")
    if sp is None:
        return None
    sd: SoSide = DA.SIDE[0]
    L, t, key, s0, s1, rows = win.L, win.t, win.key, rec["s0"], rec["s1"], rec["rows"]
    near = set(rec["cap_at"]) | {s + 1 for s in rec["cap_at"]}
    held = [s for s in range(s0, s1) if s not in near and s in sp["steps"]]
    doc: dict = {"mode": sp["mode"], "what": SPILLS[sp["mode"]], "held_steps": len(sp["steps"]), "steps": s1 - s0}
    if not held or len(sp["steps"]) != s1 - s0:
        return {**doc, "ok": False, "why": "the spill holds not every step of the window, or none uncaptured"}
    h = hashlib.sha512(b"verity/kernels-int12/spill-open/v1" + rec["draw"].coin).digest()
    steps = sorted({held[0], held[int.from_bytes(h[:8], "big") % len(held)], held[-1]})
    ai, zi, L8 = kinds.index("a-step"), kinds.index("z-step"), 8 * L
    out = []
    for j, s in enumerate(steps):
        w = int.from_bytes(h[8 + 8 * j:16 + 8 * j], "big") % L
        x, row = sp["steps"][s], rows[s - s0]
        m, n, u = x["m"], win.nk[w][0], sd.unit_of[w]

        def q_of(v: int, x=x, m=m) -> np.ndarray:
            kv, iv = sd.spill.slot_of[v]
            q = np.zeros((t, win.nk[v][1]), dtype=np.int8)
            q[:m] = x["a"][kv][iv].numpy()
            return q

        def sc_of(v: int, x=x, m=m) -> np.ndarray:
            sc = np.zeros(t, dtype=np.float32)
            sc[:m] = x["sa"][win.spos[v]].numpy()
            return sc
        a, sc = q_of(w), sc_of(w)
        a_ok = DO.a_roots_cpu(win, s, [w], q_of, sc_of)[w] == unwords(row[ai * L8 + 8 * w:ai * L8 + 8 * w + 8])
        z = torch.zeros((t, n), dtype=torch.int32, device=win.dev)
        z[:m] = E.int_mm(torch.from_numpy(a[:m]).to(win.dev), win.wt[w])
        y = E.dequantize(z, torch.from_numpy(sc).to(win.dev), win.sw[w])
        recs = DA.zy_join(z, y, u).cpu().numpy()
        cnt = recs.shape[0]
        zd = CM.domain(DA.zy_fields(win, w, s, u), cnt, DA.ZY_KIND)

        def root(r: np.ndarray, s=s, w=w, zd=zd, cnt=cnt) -> bytes:
            return CM.levels(zd, [CM.leaf(zd, key, f"zy/{s}/{w}", e, bytes(r[e]))[0] for e in range(cnt)])[-1][0]
        want = unwords(row[zi * L8 + 8 * w:zi * L8 + 8 * w + 8])
        o = {"s": s, "call": w, "module": win.names[w], "m": m, "a_root": a_ok, "zy_recomputed_root": root(recs) == want}
        if x["zy"] is not None:
            off = sum(sd.spill.prefix(v, m) for v in range(w))
            got = np.zeros(recs.size, dtype=np.uint8)
            got[:sd.spill.prefix(w, m)] = x["zy"][off:off + sd.spill.prefix(w, m)].numpy()
            eq = bool(np.array_equal(got, recs.reshape(-1)))
            o.update(zy_held_eq_recomputed=eq, zy_held_root=o["zy_recomputed_root"] if eq else root(got.reshape(cnt, -1)) == want)
        out.append(o)
    doc["opened"] = out
    doc["ok"] = all(o["a_root"] and o["zy_recomputed_root"] and o.get("zy_held_eq_recomputed", True)
                    and o.get("zy_held_root", True) for o in out)
    return doc


def end_verdict(docs: dict) -> dict:
    """The window end's verdict: the gate's ended window passes and each end fault's window is refused where its fault
    is; a live window's alone when it is the arm's one window."""
    v: dict = {"order": END_ORDER}
    if "end_gate" in docs:
        v["gate_ok"] = bool(docs["end_gate"].get("ok"))
        for f in END_FAULTS:
            x = docs.get(f, {})
            v[f"{f}_refused"] = bool(x.get("complete") and not x.get("ok") and x.get("conforming"))
        v["ok"] = v["gate_ok"] and all(v[f"{f}_refused"] for f in END_FAULTS)
    else:
        v["live_ok"] = bool(docs.get("live", {}).get("ok"))
        v["ok"] = v["live_ok"]
    return v


def end_gate(win: SoWindow, generate) -> None:
    """The end faults, each in a window of its own `end_steps` steps that ends on its own inside `generate(n)`."""
    n = end_steps(win.depth)
    for f in END_FAULTS:
        win.arm_end(f, f, n)
        generate(n + 2)


def end_checks(win: SoWindow, lib_h1o: str) -> dict:
    """Every named ended window's `check_end` and their verdict (`end_verdict`); their records released."""
    names = [x for x in ("end_gate", *END_FAULTS, "live") if x in win.end_recs]
    with torch.inference_mode():
        docs = {x: check_end(win, x, lib_h1o) for x in names}
    docs["verdict"] = end_verdict(docs)
    if DA.SIDE and DA.SIDE[0].spill is not None:
        for rec in win.end_recs.values():
            DA.SIDE[0].spill.release(rec.get("spill"))
    win.end_recs.clear()
    torch.cuda.empty_cache()
    return docs


def write_manifest(win: SoWindow, rec: dict, out_dir: str) -> dict:
    """An off-node window's manifest (`dither_manifest`): its window, coins in `coins.check`'s form, every round root,
    each step's round leaves and unit salt (a fault_rearly capture's apart, under its name), the final round, the
    manifest root; its per-call roots to the temporary directory."""
    import tempfile
    pipe, coins, rows, kinds = rec["pipe"], rec["coins"], rec["rows"], win.round_kinds
    s0, s1, i_f, nr, L8 = rec["s0"], rec["s1"], rec["i_f"], len(kinds), 8 * win.L
    caps = rec["cap_at"]
    salts = {str(s - s0 + 1): x.hex() for s, x in rec["salt_of"].items() if caps.get(s) != "fault_rearly"}
    rearly = {str(s - s0 + 1): x.hex() for s, x in rec["salt_of"].items() if caps.get(s) == "fault_rearly"}
    doc = {"schema": MF.SCHEMA, "window": win.window, "index": rec["k"], "depth": win.depth, "first": s0,
           "steps": s1 - s0, "linears": win.L, "round_kinds": list(kinds),
           "open_root": rec["open_root"].hex(),
           "open_root_is": CO.STAND_IN if coins.source == "off-node" else "the epoch root (in process)",
           "b_roots": [x.hex() for x in rec["b_roots"]],
           "rounds": [x.hex() for x in pipe.roots[:i_f + 1]],
           "round_leaves": [[unwords(rows[r][nr * L8 + 8 * i:nr * L8 + 8 * i + 8]).hex() for i in range(nr)]
                            for r in range(s1 - s0)],
           "final": {"index": i_f, "tail": rec["tail"], "leaves": [x.hex() for x in rec["final_leaves"]]},
           "manifest_root": rec["mroot"].hex(), "salts": salts, "salts_fault_rearly": rearly,
           "captures": {str(s): c for s, c in sorted(caps.items())},
           "coins": {**coins.record(), "window": {"root": rec["open_root"].hex()},
                     "commitment": coins.commitment.to_json(), "salt_coin": rec["salt_open"].to_json(),
                     "rounds": [{"round": i, "root": pipe.roots[i].hex(), "opening": pipe.opening(i, wait=False).to_json()}
                                for i in range(i_f + 1)],
                     "draw_coin": rec["draw"].to_json(), "draw_after": rec["mroot"].hex()}}
    return MF.write(out_dir, coins.session.id if coins.session else f"k{rec['k']}", doc, rows.tobytes(),
                    tempfile.gettempdir())
