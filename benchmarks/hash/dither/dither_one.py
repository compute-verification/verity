"""Dither's served linear in one stream in compiled vLLM (`h1o.cu`, H1V3): every int8-exact linear's y = dequant(z),
z = O' - sum_d 128^d D_d + 256 X (wb⁺ - wb⁻)^T from the noised mainloop whose words the call's H1 + fx1 receipts fold and
the wrap bits of W' = B + c + U, computed in the unit from B and the noise (H1V3's exact wrap correction).  The seeds are
the call's own, from the window's salt and root(A_j), root(B_j) (the seeded regime: epsilon is 'seeded'), so each
call commits its A table on the device before its units and nothing waits on a later round.

    import dither_one as DO
    DO.patch(t=T, h1c="libh1c.so", h1o="libh1o.so")   # before LLM(): Int8PlainLinearMethod.apply -> row_quant + dither1::one_mm
    llm = LLM(model=int8_exact_checkpoint, ...)       # compiled, CUDA graphs: the op is opaque to Inductor, its launches captured
    DO.hook(llm)

The window: its B tables at install, then one live round whose coin gives the epoch salt (`h1tile.live_salt` of the
round over the B roots).  Before step s's forward, the A tables' parts that read the step but no row (`h1c_a_step`:
domains, salts, prefixes, and the leaves and nodes over rows >= m alone).  A linear call (`dither1::one_mm`), on the
forward's stream, is two launches: its A slot and table (`h1c_a_call`, the root where its seeds read it), then its units
(each unit's seeds, receipt, z and y; DITHER_UNITS picks the form, `block` (`h1o_unitsb`, a unit a block, the default),
`split` (`h1o_unitss`, a unit a block of six warps) or `warp` (`h1o_units`, a unit a warp)).  After step s's forward: its
admit and z tables, the a-step,
admit-step and z-step trees, the round over them, and the receipt trees (with `DitherState.tree_streams` set, the receipt
trees and the admit tables on streams of their own beside the rest), then the round root recorded
(`sampled_proofs.challenger.LiveRecord`; its coin is not read: no seed waits on it).  With `DitherState.step_salt` set the
salt is each step's instead: after step s's round is recorded, step s + 1's seeds read
`h1tile.live_salt` of that round's opened coin and root, copied to the device between steps (in record_host_ms), so
each step's noise comes from the round before it.  The rounds' coins are `dither_fused`'s switch (`coins.open_coins`):
the challenger off the node with ``$POUW_CHALLENGER`` and ``$POUW_CHALLENGER_SESSION``, so the auditor host's round
trip is in record_host_ms, else in
this process, labelled ``in-process (not certified)`` in `stats`.

The served value is the dithered product z = X (B + c)^T, exact for every int8 B, with X the int8 A + b the MMA reads (it
wraps where A = 127 and b = 1), so y is not today's served y (`gate` measures how far) and the tokens are not the exact
arm's.  The gate records wb's density on the run's weights, and how many cells differ from (A + b)(B + c)^T and A B^T.
Peeled (DITHER_PEEL=1, H1V3's PouwPeelSeg; the block and split forms) z is A B^T, today's
product, taken from the same O: z = O + 256 X (wb⁺ - wb⁻)^T - X c^T - (b - 256 xa) B^T with xa = [A = 127 and b = 1],
the binary-by-int8 MMAs beside the wrap MMA, never a product of A itself: y is today's served y and the tokens are exact's
(both gated); the receipts are the same words, and a perturbed noised word moves z.

Labelled simplifications (none changes a step's cost): the admitted slots are the step's first m rows; the window is the
engine's life; the epoch round's root (a b-step tree over the B roots) and the z table (AROW's row-table format) are this
harness's rendering of the credit's forms.  Its stats, checks and verdict say form=beside (`forms.form_of`): not
`forms.FORM_A`.
"""
from __future__ import annotations

import ctypes as C
import os
import time

import numpy as np
import torch

import dither_fused as DF
import forms as F
from dither_fused import CM, CO, CR, SV, H1C, Dom, Tab, unwords, words, H1R, Call
import h1c as H1CM
from verity_config.settings import of

settings = of(__file__)
F_DUMP, F_NW_TILES, F_NW_UNITS, F_FAULT, F_FAULT_WB, F_FAULT_XA = 1, 2, 4, 8, 16, 32     # h1o.cu's H1OStep.flags
F_FAULT_Y = 64      # dither_a's ycommit only: bit 0 of the served y[0, 0] flipped, in place, every call
# the round's leaves past a-step, admit-step and z-step (set before the window is built): "y-step" the calls' served-y
# roots (an arm's Side fills them in `before_mid`), "receipt-step" the calls' receipt-table roots, both
# under the round recorded before the next coin
ROUND_EXTRA: tuple = ()
CAPTURES = ("gate", "control", "fault", "fault_wb")     # a peeled window's add "fault_xa" (`DitherState.captures`)
LIB_ENV = "DITHER_H1O"
PHASE_RANGE = "verity/dither/phase/"    # `DitherState.phase`'s NVTX ranges, under DITHER_NSYS=prof only
UNITS_ENV = "DITHER_UNITS"
UNITS_FORMS = {"split": "h1o_unitss_launch", "block": "h1o_unitsb_launch", "warp": "h1o_units_launch",   # h1o.cu's forms of the same units
               "split2": "h1o_unitss2_launch", "block2": "h1o_unitsb2_launch"}   # those two at the maximum carveout (`h1o_init`)
# a form per call: (wide, narrow), wide where the call's column tiles outnumber the SMs (a second block an SM pays only
# past one wave: qkv and gate_up at decode), narrow elsewhere
MIX_FORMS = {"mix": ("block2", "split")}
# "1": z = A B^T from O (the peel of b and c: `h1o_unitsbp` / `h1o_unitssp`), so y is today's served y; the warp form has none
PEEL_ENV = "DITHER_PEEL"
# the per-call challenger round, measured only: a file of waits (ns, one a line; "0": no wait) that h1o_percall replays, a call each
PERCALL_ENV = "DITHER_PERCALL"
# "1" (dither_so's -aom/-aomn without -ov; the credit's form, not yet in core): one A table a K a step,
# a/<K>@<s> (`a_step_fields`), its rows every call of that K's slots in call order (call c's slot r at row c T + r: the K
# slots' buffer and their scales' rows of `scale_all`), built once after the forward; the a-step leaf of call w is the
# root of its K's table.  Past K = 4096 a row is n = `weights.chunks` (K) leaves (SPLITK): leaf n i + c its codes
# [c K / n, (c + 1) K / n), then the row's scale
A_STEP_ENV = "DITHER_A_STEP"
# "0": A_STEP_ENV's rows whole at every K (the layout before SPLITK), only to time the chunk leaves against
A_CHUNKS_ENV = "DITHER_A_CHUNKS"
LIB = "dither1"
REG = DF.REG
CFG: dict = {}
WIN: "DitherState | None" = None
M32 = (1 << 32) - 1


class _O(C.Structure):
    _fields_ = [("slot", C.c_void_p * 2), ("scale", C.c_void_p * 2), ("z", C.c_void_p * 2), ("rc", C.c_void_p * 2),
                ("out", C.c_void_p * 2), ("seeds", C.c_void_p), ("ydump", C.c_void_p), ("t", C.c_int), ("n", C.c_int),
                ("k", C.c_int), ("units", C.c_int)]


class DitherState:
    """A window's device state: per linear one A, scale and z slot (a step's tables are built before the next step's
    forward reads its slots); the h1c sets (B tables once; the A, admit and z tables, the step trees and the round); one
    call table for the seeds and receipts; the `h1o` table and step struct; the epoch salt."""
    group_slots = False     # `slot_groups` and `scale_all` (dither_so's ring copies a step's slots from them)
    a_step = False          # A_STEP_ENV's tables (dither_so sets it, with group_slots)
    h2d_pinned = False      # `before`'s inputs through pinned buffers, without a host wait (`_h2d`)

    def __init__(self, t: int, key: bytes, h1c_path: str, h1o_path: str, gate: bool, dev):
        self.t, self.key, self.dev = t, key, dev
        self.pins: dict[int, torch.Tensor] = {}
        self.L = L = len(REG)
        if sorted(REG) != list(range(L)):
            raise RuntimeError(f"linears registered: {sorted(REG)}")
        self.names = [REG[w][0] for w in range(L)]
        self.wt = [REG[w][1].weight for w in range(L)]
        self.sw = [REG[w][1].weight_scale.reshape(-1) for w in range(L)]
        self.nk = [(int(x.shape[0]), int(x.shape[1])) for x in self.wt]
        if self.group_slots:
            # the calls' A slots as views of one buffer a K (`slot_groups`) and their scales of one (`scale_all`), so a
            # step's slots copy or clear in a launch a K
            ks = sorted({k for _, k in self.nk})
            self.slot_groups = {k: torch.zeros(sum(kk == k for _, kk in self.nk), t, k, dtype=torch.int8, device=dev)
                                for k in ks}
            at = dict.fromkeys(ks, 0)
            self.slot = []
            for _, k in self.nk:
                self.slot.append(self.slot_groups[k][at[k]])
                at[k] += 1
            self.scale_all = torch.zeros(L, t, dtype=torch.float32, device=dev)
            # under a_step the scales' rows a K in call order (`a_calls`), K by K, so a K's run is its table's tails
            self.a_calls = {k: [w for w, (_, kk) in enumerate(self.nk) if kk == k] for k in ks}
            order = [w for k in ks for w in self.a_calls[k]] if self.a_step else list(range(L))
            self.spos = [order.index(w) for w in range(L)]
            self.scale = [self.scale_all[self.spos[w]] for w in range(L)]
        else:
            self.slot = [torch.zeros(t, k, dtype=torch.int8, device=dev) for _, k in self.nk]
            self.scale = [torch.zeros(t, dtype=torch.float32, device=dev) for _ in self.nk]
            self.spos = list(range(L))
        self.z = [torch.zeros(t, n, dtype=torch.int32, device=dev) for n, _ in self.nk]
        self.live = torch.zeros(t // 32, dtype=torch.uint8, device=dev)
        self.admit = torch.zeros(t // 16 * 2, dtype=torch.uint8, device=dev)
        self.ydump = [torch.zeros(t, n, dtype=torch.bfloat16, device=dev) for n, _ in self.nk] if gate else None
        self.offs, self.u_step = SV.static_layout([n for n, _ in self.nk], t)
        self.window = f"dither1-{os.getpid()}-0"
        self.c = H1C(h1c_path, key, dev)
        self.step0 = torch.zeros(1, dtype=torch.int32, device=dev)
        self.bset = self.c.set([Tab(Dom("b-table", SV.b_fields(DF.MODEL, w, self.names[w], n, k), n), n,
                                    data=self.wt[w].view(torch.uint8), stride=k, body=k, length=k + 4,
                                    tail=self.sw[w].view(torch.uint8), tail_stride=4, lab=(f"b/{w}", ""), labs=False)
                                for w, (n, k) in enumerate(self.nk)])
        self.c.build(self.bset, self.step0)
        self.pre = self._pre_sets()
        a = self.pre["a"]
        self.awords = self.c.lib.h1c_a_words(t)
        self.apre = torch.zeros(L * self.awords, dtype=torch.int64, device=dev)
        rc = self.c.lib.h1c_a_init(C.addressof(a.host), a.dev.data_ptr(), L, self.apre.data_ptr(), self.awords,
                                   torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"h1c_a_init returned {rc}")
        self.r = H1R(h1o_path)
        self.plan = self.r.plan([Call(self.slot[w], self.wt[w], self.offs[w], self.pre["mid"].leaves_addr(0) + 64 * w,
                                      self.live, root_b=self.bset.roots.data_ptr() + 64 * w) for w in range(L)])
        rt = []
        for w, (n, k) in enumerate(self.nk):
            o, units, nw = self.plan.offs[w]
            fr = {"window": self.window, "step": 0, "linear": w, "module": self.names[w], "base": 0, "units": units,
                  "words": nw, "schema": CM.SCHEMA}
            rt.append(Tab(Dom("receipts", fr, units, {"step": (1, 0), "base": (self.u_step, self.offs[w])}), units,
                          data=self.plan.receipts(w).view(torch.uint8), stride=4 * nw, body=4 * nw, length=4 * nw,
                          lab=("r/", f"/{w}")))
        self.rset = self.c.set(rt)
        self.o = C.CDLL(h1o_path)
        self.units_form = settings.get(UNITS_ENV)
        if self.units_form not in UNITS_FORMS and self.units_form not in MIX_FORMS:
            raise RuntimeError(f"{UNITS_ENV}={self.units_form}: one of {sorted(UNITS_FORMS) + sorted(MIX_FORMS)}")
        self.peel = settings.get(PEEL_ENV)
        if self.peel and self.units_form == "warp":
            raise RuntimeError(f"{PEEL_ENV}=1: the warp form serves the dithered product only")
        self.captures = CAPTURES + (("fault_xa",) if self.peel else ())
        if self.units_form in MIX_FORMS:
            wide, narrow = MIX_FORMS[self.units_form]
            sms = torch.cuda.get_device_properties(dev).multi_processor_count
            self.call_forms = [wide if n // 32 > sms else narrow for n, _ in self.nk]
        else:
            self.call_forms = [self.units_form] * L

        def units_fn(form: str):
            fn = UNITS_FORMS[form]
            f = getattr(self.o, fn.replace("_launch", "p_launch") if self.peel else fn)
            f.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_int,
                          C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
            return f
        self.units_fns = [units_fn(f) for f in self.call_forms]
        if self.o.h1o_struct_bytes(0) != C.sizeof(_O) or self.o.h1o_struct_bytes(1) != 16:
            raise RuntimeError("h1o.cu's structs differ from dither_one's")
        if self.o.h1o_init():
            raise RuntimeError("h1o_init refused")
        self.percall = None
        pc = settings.get(PERCALL_ENV)
        if pc:
            ns = [0] if pc == "0" else [int(x) for x in open(pc).read().split()]
            if self.o.h1o_percall_init(L, (C.c_longlong * len(ns))(*ns), len(ns)):
                raise RuntimeError("h1o_percall_init refused")
            self.o.h1o_percall_launch.argtypes = [C.c_int, C.c_void_p, C.c_void_p, C.c_void_p]
            self.o.h1o_percall_done.restype = C.c_ulonglong
            self.coin = torch.zeros(L * 64, dtype=torch.uint8, device=dev)
            self.percall = {"waits": pc, "n": len(ns), "mean_us": round(sum(ns) / len(ns) / 1e3, 3),
                            "p50_us": sorted(ns)[len(ns) // 2] / 1e3}
        self.call_bytes = self.r.lib.h1r_call_bytes()
        self.tab_bytes = C.sizeof(H1CM._Tab)
        tab = (_O * L)()
        for w, (n, k) in enumerate(self.nk):
            e = _O()
            for p in range(2):
                e.slot[p], e.scale[p], e.z[p] = self.slot[w].data_ptr(), self.scale[w].data_ptr(), self.z[w].data_ptr()
                e.rc[p] = self.plan.table.data_ptr() + self.call_bytes * w
                e.out[p] = self.plan.out.data_ptr()
            e.seeds = self.plan.seeds.data_ptr()
            e.ydump = self.ydump[w].data_ptr() if gate else None
            e.t, e.n, e.k, e.units = t, n, k, self.plan.offs[w][1]
            tab[w] = e
        self.F = torch.frombuffer(bytearray(bytes(tab)), dtype=torch.uint8).to(dev)
        self.fsize = C.sizeof(_O)
        self.S = torch.zeros(4, dtype=torch.int32, device=dev)
        self.units = [self.plan.offs[w][1] for w in range(L)]
        self.salt = torch.zeros(8, dtype=torch.int64, device=dev)
        self.step_salt = False
        self.step_base = torch.zeros(1, dtype=torch.int64, device=dev)
        self.stepv = torch.zeros(2, dtype=torch.int32, device=dev)
        # the engine's own forwards (its profile run, its graph captures) come before any step's `before`, and
        # h1c_a_call3_k traps on node and leaf prefixes h1c_a_step hasn't written
        self._a_step()
        self.s, self.flags, self.m = 0, 0, 0
        self.calls = self.steps = 0
        self.arms: list[str] = []
        self.arm_m: int | None = None
        self.caps: dict[str, dict] = {}
        self.cap: str | None = None
        self.round_log: list[dict] = []
        self.phases: list[dict] = []
        self.t_before = 0.0
        self.ev = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
        # (receipt trees', admit tables') streams: between steps those two builds run beside the z tables and the step
        # trees (they read none of each other's outputs; the round waits on the admit tables, the record on all)
        self.tree_streams: tuple | None = None
        # called on the serving stream after the z tables and before the step trees (an arm's leaves of its own)
        self.before_mid: list = []
        self.nvtx = settings.DITHER_NSYS == "prof"
        self.phase_open = False
        self._epoch()

    def _epoch(self) -> None:
        from verity.core.protocols.pouw.circuit import h1tile as HT
        self.HT = HT
        self.coins = CO.open_coins(CO.session(), 1, DF.ROUNDS)
        self.rounds = self.coins.rounds
        broots = [unwords(self.bset.root(w)) for w in range(self.L)]
        self.epoch_root = CM.step_root(CM.domain({"window": self.window, "linears": self.L}, self.L, "b-step"), broots)
        opening = self.rounds.record(self.epoch_root)
        self.salt_bytes = HT.live_salt(self.coins.commitment, opening.coin, 0, self.epoch_root)
        self.salt.copy_(torch.from_numpy(words(self.salt_bytes)))

    def arm(self, m: int, names: tuple | None = None) -> None:
        """Capture `names` (`captures` by default) at the next steps of m tokens, one a step: each step's A, scale and z
        slots, y dump, receipts, seeds and step roots poisoned with 0xA5 before its forward; "control" with the no-write
        flags (its kernels write none of them, so `check` must refuse it); "fault" with F_FAULT (`check` wants that one
        word to reach z, y and the receipt, peeled too); "fault_wb" with F_FAULT_WB (`check` wants only column 0 of unit
        0's z and y moved, by 256 (1 - 2 wb⁺) X); "fault_xa", peeled, with F_FAULT_XA (only row 0 of unit 0's z and y,
        by 256 (1 - 2 xa) B[:, 0])."""
        names = self.captures if names is None else names
        self.arms, self.arm_m = list(names), m
        self.caps = {n: {} for n in names}

    def _pre_sets(self) -> dict:
        c, W, t, L = self.c, self.window, self.t, self.L
        f_step = {"window": W, "step": 0, "linears": L}
        self.round_kinds = ("a-step", "admit-step", "z-step") + tuple(ROUND_EXTRA)
        nr = len(self.round_kinds)
        rnd = c.set([Tab(Dom("round", {"window": W, "step": 0, "leaves": round_leaves(self.round_kinds)}, nr,
                             {"step": (1, 0)}), nr)])
        mid = c.set([Tab(Dom(kd, f_step, L, {"step": (1, 0)}), L, out=rnd.leaves_addr(0) + 64 * i)
                     for i, kd in enumerate(self.round_kinds)])
        at, st, zt = [], [], []
        for w, (n, k) in enumerate(self.nk):
            fa = {"window": W, "step": 0, "linear": w, "module": self.names[w], "slots": t, "k": k, "schema": CM.SCHEMA}
            at.append(Tab(Dom("a-table", fa, t, {"step": (1, 0)}), t, data=self.slot[w].view(torch.uint8), stride=k,
                          body=k, length=k + 4, tail=self.scale[w].view(torch.uint8), tail_stride=4,
                          lab=("a/", f"/{w}"), out=mid.leaves_addr(0) + 64 * w))
            fs = {"window": W, "step": 0, "linear": w, "module": self.names[w], "slots": t, "schema": CM.SCHEMA}
            st.append(Tab(Dom("admit", fs, t // 16, {"step": (1, 0)}), t // 16, data=self.admit, stride=2, body=2,
                          length=2, lab=("s/", f"/{w}"), out=mid.leaves_addr(1) + 64 * w))
            fz = {"window": W, "step": 0, "linear": w, "module": self.names[w], "slots": t, "n": n, "schema": CM.SCHEMA}
            zt.append(Tab(Dom("z-table", fz, t, {"step": (1, 0)}), t, data=self.z[w].view(torch.uint8), stride=4 * n,
                          body=4 * n, length=4 * n, lab=("z/", f"/{w}"), out=mid.leaves_addr(2) + 64 * w))
        return {"round": rnd, "mid": mid, "a": c.set(at), "admit": c.set(st), "z": c.set(zt)}

    def launch(self, w: int, xq: torch.Tensor, s_a: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor,
               z: torch.Tensor) -> None:
        st = torch.cuda.current_stream().cuda_stream
        rc = (self.a_call(w, xq, s_a, st) or (self.percall is not None and self.coin_call(w, st))
              or self.units_call(w, xq, s_a, s_w, y, z, st))
        if rc:
            raise RuntimeError(f"dither1 call {w} (M={xq.shape[0]}): a launch returned {rc}")
        self.calls += 1

    def a_call(self, w: int, xq: torch.Tensor, s_a: torch.Tensor, st: int) -> int:
        a, tb = self.pre["a"], self.tab_bytes * w
        return self.c.lib.h1c_a_call(C.addressof(a.host) + tb, a.dev.data_ptr() + tb,
                                     self.apre.data_ptr() + 8 * self.awords * w, self.S.data_ptr(), xq.data_ptr(),
                                     s_a.data_ptr(), self.slot[w].data_ptr(), self.scale[w].data_ptr(), self.nk[w][1], st)

    def coin_call(self, w: int, st: int) -> int:
        """root(A_w) to the host, one challenger round's wait, a coin back (`h1o_percall_launch`)."""
        return self.o.h1o_percall_launch(w, self.pre["mid"].leaves_addr(0) + 64 * w, self.coin.data_ptr() + 64 * w, st)

    def units_call(self, w: int, xq: torch.Tensor, s_a: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor,
                   z: torch.Tensor, st: int) -> int:
        n, k = self.nk[w]
        return self.units_fns[w](self.F.data_ptr() + self.fsize * w, self.S.data_ptr(), self.units[w], n, k, xq.data_ptr(),
                                 s_a.data_ptr(), int(xq.shape[0]), s_w.data_ptr(), y.data_ptr(), z.data_ptr(),
                                 self.salt.data_ptr(), self.step_base.data_ptr(), st)

    # -- the step hook ---------------------------------------------------------------------------------------------------

    def _poison(self, ts) -> None:
        for x in ts:
            x.view(torch.uint8).fill_(DF.POISON)

    def _h2d(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        """dst from the host tensor src on the current stream; under `h2d_pinned` (dither_so's $DITHER_SO_ROOT_LAG=2)
        through dst's own pinned buffer without a host wait, its last copy out (the engine waited on the forward after
        it)."""
        if not self.h2d_pinned:
            dst.copy_(src)
            return
        buf = self.pins.get(dst.data_ptr())
        if buf is None:
            buf = self.pins[dst.data_ptr()] = torch.empty(src.shape, dtype=src.dtype).pin_memory()
        buf.copy_(src)
        dst.copy_(buf, non_blocking=True)

    def phase(self, name: str | None) -> None:
        """Under DITHER_NSYS=prof, close the open phase's NVTX range and open PHASE_RANGE/NAME's (stepprof.py gives each
        kernel the phase its launch was in)."""
        if not self.nvtx:
            return
        if self.phase_open:
            torch.cuda.nvtx.range_pop()
        self.phase_open = name is not None
        if name is not None:
            torch.cuda.nvtx.range_push(PHASE_RANGE + name)

    def poisoned(self) -> list:
        """What a captured step's `before` fills with 0xA5."""
        return (self.slot + self.scale + self.z + list(self.ydump or [])
                + [self.plan.out, self.plan.seeds, self.pre["mid"].level, self.pre["a"].level])

    def dead_from(self, m: int) -> int:
        """The step's S[2] (after `before` sets its flags and capture): 0, every row past m's z/y records written."""
        return 0

    def _a_step(self) -> None:
        a = self.pre["a"]
        rc = self.c.lib.h1c_a_step(a.dev.data_ptr(), self.L, self.apre.data_ptr(), self.awords, self.stepv.data_ptr(),
                                   self.c.btab.data_ptr(), self.S.data_ptr(), torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"h1c_a_step returned {rc}")

    def before(self, m: int) -> None:
        if m > self.t:
            raise RuntimeError(f"a step of {m} tokens in a window of T = {self.t} slots")
        self.phase("before")
        self.m, self.flags, self.cap = m, 0, None
        if self.arms and m == self.arm_m:
            self.cap = name = self.arms.pop(0)
            self.flags = (F_DUMP | ((F_NW_TILES | F_NW_UNITS) if name == "control" else 0) | (F_FAULT if name == "fault" else 0)
                          | (F_FAULT_WB if name == "fault_wb" else 0) | (F_FAULT_XA if name == "fault_xa" else 0)
                          | (F_FAULT_Y if name == "fault_y" else 0))
            self._poison(self.poisoned())
            self.caps[name] = {"s": self.s, "m": m, "salt": self.salt_bytes}
        t, s = self.t, self.s
        bits = np.arange(t) < m
        self._put(self.live, torch.from_numpy((np.arange(t // 32) * 32 < m).astype(np.uint8)))
        self._put(self.admit, torch.frombuffer(bytearray(b"".join(CR.word_rows(CR.words(bits)))), dtype=torch.uint8))
        self._put(self.stepv, torch.tensor([s, s], dtype=torch.int32))
        self.step_base.fill_(s * self.u_step)
        self._put(self.S, torch.tensor([0, m, self.dead_from(m), self.flags], dtype=torch.int32))
        self._a_step()
        if self.cap:
            self.cap_sync()
        self.phase(None)
        self.t_before = time.perf_counter()

    def _put(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        """A step input from the host into `dst`, on the current stream (`_h2d`)."""
        self._h2d(dst, src)

    def cap_sync(self) -> None:
        """A captured step's wait, after its poison and inputs, before its forward."""
        torch.cuda.synchronize()

    def after(self) -> None:
        self.phase("forward_wait")
        torch.cuda.synchronize()
        t_fwd = time.perf_counter()
        s, m, pre, ev = self.s, self.m, self.pre, self.ev
        par = self.tree_streams is not None
        main = torch.cuda.current_stream()
        self.phase("admit_tables")
        ev[0].record()
        if par:
            rs, ads = self.tree_streams
            for x in (rs, ads):
                x.wait_event(ev[0])
            with torch.cuda.stream(rs):
                ev[5].record(rs)
                self.c.build(self.rset, self.stepv[0:1])
                ev[6].record(rs)
            with torch.cuda.stream(ads):
                self.c.build(pre["admit"], self.stepv[0:1])
                ev[7].record(ads)
        else:
            self.c.build(pre["admit"], self.stepv[0:1])
        ev[1].record()
        self.phase("z_tables")
        self.c.build(pre["z"], self.stepv[0:1])
        for f in self.before_mid:
            f()
        ev[2].record()
        self.phase("step_trees")
        rleaf = "receipt-step" in self.round_kinds
        if par:
            main.wait_event(ev[7])
            if rleaf:
                main.wait_event(ev[6])
        elif rleaf:
            self.c.build(self.rset, self.stepv[0:1])
        if rleaf:
            # the receipt roots under the round, so the round recorded before the next coin fixes them; "fault_rr"
            # records the round without them (`check` must refuse it)
            o = pre["mid"].level_off[self.round_kinds.index("receipt-step")]
            lv = pre["mid"].level[o:o + 8 * self.L]
            if self.cap == "fault_rr":
                lv.view(torch.uint8).fill_(DF.POISON)
            else:
                lv.copy_(self.rset.roots)
        self.c.build(pre["mid"], self.stepv[0:1])
        self.c.build(pre["round"], self.stepv[0:1])
        ev[3].record()
        self.phase("receipt_trees")
        if par:
            main.wait_event(ev[6])
        elif not rleaf:
            self.c.build(self.rset, self.stepv[0:1])
        ev[4].record()
        self.phase("root_read")
        root = unwords(pre["round"].root(0))
        t_root = time.perf_counter()
        self.phase("record_host")
        opening = self.rounds.record(root)
        if self.step_salt:
            self.salt_bytes = self.HT.live_salt(self.coins.commitment, opening.coin, opening.index, root)
            self.salt.copy_(torch.from_numpy(words(self.salt_bytes)))
        t_end = time.perf_counter()
        self.phase(None)
        self.phases.append({"m": m, "forward_ms": 1e3 * (t_fwd - self.t_before),
                            "admit_tables_ms": ev[0].elapsed_time(ev[7] if par else ev[1]),
                            "z_tables_ms": ev[1].elapsed_time(ev[2]), "step_trees_ms": ev[2].elapsed_time(ev[3]),
                            "receipt_trees_ms": ev[5].elapsed_time(ev[6]) if par else ev[3].elapsed_time(ev[4]),
                            "trees_ms": ev[0].elapsed_time(ev[4]), "record_host_ms": 1e3 * (t_end - t_root),
                            "between_ms": 1e3 * (t_end - t_fwd)})
        if self.cap is not None:
            self.caps[self.cap].update(
                root=root, a=[x.clone() for x in self.slot], sa=[x.clone() for x in self.scale], z=[x.clone() for x in self.z],
                y=[x[:m].clone() for x in self.ydump] if self.ydump else None, receipts=self.plan.out.clone(),
                r_roots=self.rset.roots.clone(), a_roots=[unwords(pre["mid"].leaf(0, w)) for w in range(self.L)],
                admit_roots=[unwords(pre["mid"].leaf(1, w)) for w in range(self.L)],
                z_roots=[unwords(pre["mid"].leaf(2, w)) for w in range(self.L)],
                leaves=[unwords(pre["round"].leaf(0, i)) for i in range(len(self.round_kinds))],
                b_roots=[unwords(self.bset.root(w)) for w in range(self.L)])
        self.round_log.append({"s": s, "m": m, "root": root.hex()[:32]})
        self.s += 1
        self.steps += 1
        if self.s + 1 >= DF.ROUNDS:
            raise RuntimeError(f"the window's {DF.ROUNDS} rounds are spent")

    def stats(self) -> dict:
        ph, self.phases = self.phases, []
        by: dict = {}
        for x in ph:
            by.setdefault(x["m"], []).append(x)
        top = sorted(by.items(), key=lambda kv: -len(kv[1]))[:4]

        def per(f):
            return {str(m): {"steps": len(xs), **{k: round(float(f([x[k] for x in xs if k in x])), 4)
                                                  for k in dict.fromkeys(k for x in xs for k in x) if k != "m"}}
                    for m, xs in top}
        med = per(np.median)
        pc = None if self.percall is None else {**self.percall, "rounds_served": int(self.o.h1o_percall_done())}
        return {**F.form_of("dither-1s"), "steps": self.steps, "one_calls_eager": self.calls, "t": self.t, "s": self.s,
                "epsilon": "seeded",
                "salt": "step" if self.step_salt else "epoch", "coins": self.coins.label, "units_form": self.units_form, "peel": self.peel,
                "percall": pc, "phases_by_m": med, "phases_mean_by_m": per(np.mean),
                "phases_p90_by_m": per(lambda v: np.percentile(v, 90)), "call_forms": {DF.SITES[w % 4]: self.call_forms[w] for w in range(min(4, self.L))}}


def round_leaves(kinds) -> str:
    """The round domain's "leaves" field: "a-step root, admit-step root, z-step root" and ROUND_EXTRA's."""
    return ", ".join(f"{k} root" for k in kinds)


# -- the op and the patch --------------------------------------------------------------------------------------------------

def _window(dev) -> DitherState:
    global WIN
    if WIN is None:
        WIN = DitherState(CFG["t"], CFG["key"], CFG["h1c"], CFG["h1o"], CFG["gate"], dev)
    return WIN


# A kernel op (`torch.library.Library`, a CUDA implementation), not a `custom_op`: the credit gate's trace counts an op without a
# Python definition as one GEMM, and its two fresh results, z and y, as written by it.
_LIB = torch.library.Library(LIB, "DEF")
_LIB.define("one_mm(Tensor xq, Tensor s_a, Tensor w, Tensor s_w, int idx) -> (Tensor, Tensor)")


def _one_mm_cuda(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor, idx: int):
    z = torch.empty((xq.shape[0], w.shape[0]), dtype=torch.int32, device=xq.device)
    y = torch.empty((xq.shape[0], w.shape[0]), dtype=torch.bfloat16, device=xq.device)
    if xq.shape[0]:
        _window(xq.device).launch(idx, xq.contiguous(), s_a.contiguous(), s_w.reshape(-1), y, z)
    return z, y


_LIB.impl("one_mm", _one_mm_cuda, "CUDA")


@torch.library.register_fake(f"{LIB}::one_mm")
def _one_mm_fake(xq, s_a, w, s_w, idx):
    return (xq.new_empty((xq.shape[0], w.shape[0]), dtype=torch.int32),
            xq.new_empty((xq.shape[0], w.shape[0]), dtype=torch.bfloat16))


CREDIT: dict = {}
CG = None


def credit_doc() -> dict:
    """The credit gate's verdicts over the traced calls: how many, how many credited, each refusal's reason and the kernels."""
    return {"calls": CREDIT.get("calls", 0), "credited": CREDIT.get("credited", 0),
            "reasons": dict(CREDIT.get("reasons", {})), "kernels": dict(CREDIT.get("kernels", {})),
            "sample": CREDIT.get("sample", [])}


def patch(t: int, h1c: str, h1o: str, key: bytes | None = None, gate: bool = False, credit: str | None = None) -> None:
    """Every int8-exact linear through ``dither1::one_mm`` (call before the engine is built, in its process).  ``credit``,
    the path of `credit_gate.py`: each `apply` runs inside its `Trace` and its verdict on (the y handed back, the
    op's z) is tallied in `CREDIT` (an eager engine: a compiled one never runs `apply` per call)."""
    from collections import Counter

    from verity_vllm.engine import int8_ops
    from verity_vllm.engine import int8_vllm as IV
    global CG
    CFG.update(t=t, h1c=h1c, h1o=h1o, key=key or os.urandom(64), gate=gate)
    IV.register()
    orig_pw = IV.Int8PlainLinearMethod.process_weights_after_loading

    def process_weights_after_loading(self, layer):
        orig_pw(self, layer)
        self.dither_idx = DF.index_of(self.prefix)
        REG[self.dither_idx] = (self.prefix, layer)

    def apply(self, layer, x, bias=None):
        if bias is not None or x.dtype != torch.bfloat16:
            raise ValueError(f"{self.prefix}: the one-stream arm takes bf16 activations and no bias")
        lead, k = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, k)
        n = layer.weight.shape[0]
        if x2.shape[0] == 0:
            return x.new_empty(*lead, n)
        xq, s_a = int8_ops.row_quant(x2)
        z, y = torch.ops.dither1.one_mm(xq, s_a, layer.weight, layer.weight_scale, self.dither_idx)
        if CG is not None:
            self.dither_z = z
        return y.reshape(*lead, n)

    if credit:
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("credit_gate_2684", credit)
        CG = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = CG     # its dataclasses look their module up there
        spec.loader.exec_module(CG)
        CREDIT.update(calls=0, credited=0, reasons=Counter(), kernels=Counter(), sample=[])
        served = apply

        def apply(self, layer, x, bias=None):
            self.dither_z = None
            with CG.Trace(self.prefix) as tr:
                y = served(self, layer, x, bias)
            v = CG.verdict(tr, y, self.dither_z)
            CREDIT["calls"] += 1
            CREDIT["credited"] += int(v["credit"])
            if v["credit"]:
                CREDIT["kernels"][v["kernel"]] += 1
            else:
                CREDIT["reasons"][v["reason"]] += 1
            if len(CREDIT["sample"]) < 3:
                CREDIT["sample"].append(v)
            return y

    IV.Int8PlainLinearMethod.process_weights_after_loading = process_weights_after_loading
    IV.Int8PlainLinearMethod.apply = apply


def hook(llm) -> DitherState:
    from verity_vllm.engine import handle
    runner = handle.model_runner(llm)
    win = WIN if WIN is not None else _window(torch.device("cuda"))
    orig = runner.execute_model

    def execute_model(so, *args, **kw):
        m = int(getattr(so, "total_num_scheduled_tokens", 0) or 0)
        if m:
            with torch.inference_mode():
                win.before(m)
        out = orig(so, *args, **kw)
        if m:
            with torch.inference_mode():
                win.after()
        return out

    runner.execute_model = execute_model
    return win


# -- the gate ----------------------------------------------------------------------------------------------------------------

def _sx(v: np.ndarray) -> np.ndarray:
    return ((v.astype(np.int64) & 0xFF) ^ 0x80) - 0x80


def _s32(v: np.ndarray) -> np.ndarray:
    v = v & M32
    return np.where(v >> 31, v - (1 << 32), v)


def cap_salt(win, g: dict) -> tuple[bytes, torch.Tensor]:
    """The salt capture g's step ran under (the epoch's, or with step_salt its round's) and its words on the device."""
    b = g.get("salt", win.salt_bytes)
    return b, torch.from_numpy(words(b)).to(win.dev)


def h1v3_unit(a32: np.ndarray, b32: np.ndarray, salt: bytes, position: int, roots: tuple[bytes, bytes]) -> dict:
    """H1V3's Definition on the CPU for one 32 x 32 unit (`h1tile.noise`, `segment.call_words` and the CP-W fold, as
    `h1tile.reference`): z = O' - sum_d 128^d D_d + 256 X (wb⁺ - wb⁻)^T per cell (s32), wb⁺ = [B + c + U > 127] and
    wb⁻ = [B + c + U < -128] on the integers, the receipt, and the products to hold z to: X (B + c)^T with X the int8
    A + b (z must equal it), (A + b)(B + c)^T and today's A B^T.  And the peel of b and c, as written,
    z - b B^T - A c^T - b c^T, and with the term the s8 X's wrap needs, + 256 xw (B + c)^T, xw = [A = 127 and b = 1]
    (X = -128 there): A B^T, as a peeled kernel's z must be; and the peeled kernel's own form of it (H1V3's PouwPeelSeg),
    z + 256 xw B^T - X c^T - b B^T."""
    from verity.core.protocols.pouw.circuit import h1tile as HT
    from verity.core.protocols.pouw.circuit import seeds as SD
    from verity.core.protocols.pouw.circuit import segment as SG
    a = [[int(v) & 0xFF for v in row] for row in a32]
    b = [[int(v) & 0xFF for v in row] for row in b32]
    lay = HT.Layout(32, 32, len(a[0]))
    rows, cols = HT.noise(SD.seed(salt, position, *roots), lay)
    keys = HT.fold_keys(SD.receipt_seed(salt, position, *roots), lay)
    ks, m = lay.ks, lay.m
    cp = [[0, 0] for _ in range(lay.sets)]
    z = np.zeros((32, 32), dtype=np.int64)
    for ln in range(lay.lanes):
        R, Cs = lay.rows(ln), lay.cols(ln)
        seg = [range(32 * s, 32 * s + 32) for s in range(ks)]
        xr = [[[a[i][t] for t in ts] for i in R] for ts in seg]
        bb = [[[rows[i][0][t] for t in ts] for i in R] for ts in seg]
        wr = [[[b[j][t] for t in ts] for j in Cs] for ts in seg]
        cc = [[[cols[j][0][t] for t in ts] for j in Cs] for ts in seg]
        sets = SG.call_words(xr, bb, wr, cc, [cols[j][1] for j in Cs], [[rows[i][1][d] for i in R] for d in range(m - 1)])
        st = list(keys[ln])
        for s, p in enumerate(sets):
            st = SG.absorb(HT.FOLD, st, p)
            for h in range(2):
                for v in st[8 * h:8 * h + 8]:
                    cp[s][h] ^= v
        for q in range(32):
            z[R[SG.row_slot(q)], Cs[SG.col_slot(q)]] = (sets[ks][q] - sum((128 ** d) * sets[ks + d][q]
                                                                          for d in range(1, m))) & M32
    bits_a = np.array([rows[i][0] for i in range(32)], dtype=np.int64)
    bits_b = np.array([cols[j][0] for j in range(32)], dtype=np.int64)
    k = a32.shape[1]
    w_int = b32.astype(np.int64) + bits_b + np.tile(_sx(np.array([cols[j][1] for j in range(32)])), (1, k // 32))
    wbp, wbn = w_int > 127, w_int < -128
    x = _sx(a32.astype(np.int64) + bits_a)
    z = _s32(z + 256 * (x @ (wbp.astype(np.int64) - wbn.astype(np.int64)).T))
    wc = b32.astype(np.int64) + bits_b
    a64 = a32.astype(np.int64)
    peel = bits_a @ b32.astype(np.int64).T + a64 @ bits_b.T + bits_a @ bits_b.T
    xwb = ((a64 == 127) & (bits_a == 1)).astype(np.int64)
    xw_term = 256 * (xwb @ wc.T)
    b64 = b32.astype(np.int64)
    return {"z": z, "receipt": [w & M32 for s in cp for w in s], "xw": x @ wc.T,
            "dithered": (a64 + bits_a) @ wc.T, "today": a64 @ b64.T,
            "wb_plus": int(wbp.sum()), "wb_minus": int(wbn.sum()), "entries": int(wbp.size),
            "peel": peel, "xw_term": xw_term, "x_wraps": int(xwb.sum()),
            "peel_ca": _s32(z - peel), "peel_xw": _s32(z - peel + xw_term),
            "peel_kernel": _s32(z + 256 * (xwb @ b64.T) - x @ bits_b.T - bits_a @ b64.T)}


def wb_fault_expected(a: np.ndarray, b: np.ndarray, salt: bytes, position: int, roots: tuple[bytes, bytes]) -> list[int]:
    """F_FAULT_WB's move of unit 0's z column 0 (rows of `a`, 32 x K; `b` the unit's 32 B rows), mod 2^32: its wb⁺ bit at
    entry 0 flipped changes wb⁺ - wb⁻ there by 1 - 2 wb⁺, so z[i, 0] moves by 256 (1 - 2 wb⁺) X[i, 0]."""
    from verity.core.protocols.pouw.circuit import h1tile as HT
    from verity.core.protocols.pouw.circuit import seeds as SD
    rows, cols = HT.noise(SD.seed(salt, position, *roots), HT.Layout(32, 32, a.shape[1]))
    s = int(b[0, 0]) + int(cols[0][0][0]) + int(_sx(np.array(cols[0][1][0])))
    d = 1 - 2 * int(s > 127)
    return [(256 * d * int(_sx(np.array(int(a[i, 0]) + int(rows[i][0][0]))))) & M32 for i in range(32)]


def xa_fault_expected(a: np.ndarray, b: np.ndarray, salt: bytes, position: int, roots: tuple[bytes, bytes]) -> list[int]:
    """F_FAULT_XA's move of unit 0's z row 0 (as `wb_fault_expected`), mod 2^32: its xa bit at entry 0 flipped changes
    xa there by 1 - 2 xa, so z[0, j] moves by 256 (1 - 2 xa) B[j, 0]."""
    from verity.core.protocols.pouw.circuit import h1tile as HT
    from verity.core.protocols.pouw.circuit import seeds as SD
    rows, _ = HT.noise(SD.seed(salt, position, *roots), HT.Layout(32, 32, a.shape[1]))
    d = 1 - 2 * int(int(a[0, 0]) == 127 and int(rows[0][0][0]) == 1)
    return [(256 * d * int(b[j, 0])) & M32 for j in range(32)]


def a_chunks(k: int) -> int:
    """A_STEP_ENV's leaves a row of K = k: `weights.chunks` (k), or 1 under A_CHUNKS_ENV=0."""
    from verity.core.protocols.pouw import weights as WT
    return 1 if settings.get(A_CHUNKS_ENV) == "0" else WT.chunks(k)


def a_step_fields(win: DitherState, k: int, s: int) -> dict:
    """A_STEP_ENV's table a/<k>@<s>: its domain's fields (kind "a-table", count n len(linears) T at n = `a_chunks` (k),
    which the fields name past 1)."""
    n = a_chunks(k)
    return {"window": win.window, "step": s, "k": k, "linears": win.a_calls[k], "slots": win.t, "schema": CM.SCHEMA,
            **({"chunks": n} if n > 1 else {})}


def a_step_rows(q: np.ndarray, sc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A_STEP_ENV's leaves over rows q ([R, K] int8) and their scales sc ([R]): R n leaves of K / n codes at n =
    `a_chunks` (K), leaf n i + c row i's chunk c, each ending in row i's scale."""
    n = a_chunks(q.shape[1])
    return q.reshape(-1, q.shape[1] // n), np.repeat(np.asarray(sc).reshape(-1), n)


def a_roots_cpu(win: DitherState, s: int, calls, q_of, sc_of) -> dict[int, bytes]:
    """commit.py's A root of each call in `calls` at step s from its slot rows q_of(w) ([T, k] int8) and scales sc_of(w)
    ([T] float32): its own table a/<s>/<w>, or under a_step its K's a/<K>@<s> (the K's calls' rows in call order, as
    `a_step_rows` leaves)."""
    calls, out = list(calls), {}
    for w in calls:
        k = win.nk[w][1]
        if not win.a_step:
            fa = {"window": win.window, "step": s, "linear": w, "module": win.names[w], "slots": win.t, "k": k,
                  "schema": CM.SCHEMA}
            out[w] = CM.s8_table(CM.domain(fa, win.t, "a-table"), win.key, f"a/{s}/{w}", q_of(w), sc_of(w)).root
            continue
        if w in out:
            continue
        cs = win.a_calls[k]
        q, sc = a_step_rows(np.concatenate([q_of(v) for v in cs]), np.concatenate([sc_of(v) for v in cs]))
        r = CM.s8_table(CM.domain(a_step_fields(win, k, s), len(q), "a-table"), win.key, f"a/{k}@{s}", q, sc).root
        out.update(dict.fromkeys(cs, r))
    return {w: out[w] for w in calls}


def a_bad_diag(g: dict, bad: list[int]) -> list[dict]:
    """For each call whose A root is off commit.py's, what of its saved slot rows and scales the A kernel left unwritten
    or wrote wrong: live rows still all 0xA5, live scales still 0xA5 or not positive, dead rows not zero."""
    m, out = g["m"], []
    for w in bad:
        q, sc = g["a"][w], g["sa"][w].reshape(-1).contiguous()
        rows_p = (q.view(torch.uint8) == DF.POISON).all(dim=1)
        sc_p = (sc.view(torch.uint8).view(-1, sc.element_size()) == DF.POISON).all(dim=1)
        out.append({"call": w, "live_rows_poison": rows_p[:m].nonzero().flatten().tolist()[:8],
                    "live_scales_poison": sc_p[:m].nonzero().flatten().tolist()[:8],
                    "live_scales_not_positive": (sc[:m].float() <= 0).nonzero().flatten().tolist()[:8],
                    "dead_rows_nonzero": (q[m:] != 0).any(dim=1).nonzero().flatten().add(m).tolist()[:8],
                    "dead_scales_nonzero": (sc[m:] != 0).nonzero().flatten().add(m).tolist()[:8]})
    return out


def h1v3_itself(units: list[dict]) -> dict:
    """`h1v3_ref.py` under H1V3's tree: z and receipt of each unit by its own `h1tile.reference(stream=True)`. The tree is
    this source's own (PouwH1Tile_v4, ungated: v3's z, receipts and wrap bits with the peel) unless DITHER_H1V3_TREE
    names another (a launcher's H1V3_SHA)."""
    import json
    import subprocess
    import sys
    import tempfile
    tree = settings.DITHER_H1V3_TREE or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    sha = settings.DITHER_H1V3_SHA or settings.RESEARCH_SOURCE_SHA or "source"
    with tempfile.TemporaryDirectory() as td:
        src, dst = os.path.join(td, "in.json"), os.path.join(td, "out.json")
        with open(src, "w") as f:
            json.dump(units, f)
        env = {**os.environ, "PYTHONPATH": tree}
        p = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "h1v3_ref.py"), src, dst],
                           env=env, capture_output=True, text=True)
        if p.returncode:
            return {"tree": tree, "sha": sha, "error": p.stderr[-2000:]}
        with open(dst) as f:
            doc = json.load(f)
    return {"tree": tree, "sha": sha, **doc}


def check(win: DitherState, lib_h1o: str, which: str = "gate", n_cpu_units: int = 3,
          cpu_calls: tuple = (0, 1, 2, 3, 126, 127)) -> dict:
    """`which`'s capture against references other than the served kernel: every call's A, admit and z roots, the receipt
    roots of CPU_CALLS and the step's round = commit.py's; every receipt = h1r v1's (another unit implementation, gated
    against receipts.py) on the saved slots under the window's salt and the CPU's roots; every z = h1o_ref's (noise from
    full compressions, T_last and Y by scalar dp4a, z = T_last - U Y + 256 X wbᵀ by a scalar scatter over the integer
    B + c + U, which also counts wb: its density); y (dumped) = dequant of the kernel's own z on the rows < m; a few units =
    H1V3's Definition on the CPU (`h1v3_unit`: z, receipt, z = X (B + c)^T) and by H1V3's own code (`h1v3_itself`).  Every
    buffer read was 0xA5 before the kernels wrote it.  "fault" wants z off h1o_ref's by 2^FAULT_SHIFT at unit 0's cell
    (0, 0) of every call and nowhere else, that unit's receipt off h1r's and no other, and y = dequant of that z (the
    perturbed MMA word reached what serves).  "fault_wb" wants every receipt unmoved (wb is not receipted), z moved only in
    column 0 of unit 0, by `wb_fault_expected` on CPU_CALLS and by multiples of 256 elsewhere, and y = dequant of that z.
    Peeled (`DitherState.peel`), z's reference is A B^T itself (a float64 product on the device, exact: every partial sum is
    an integer below 2^53), y must be today's served y (`int8_ops.scaled_mm`'s body) on every call, the CPU units' z the kernel's
    peel of their z, and each fault wants what it wants unpeeled against A B^T: so a z that reads no noised word (one
    computed beside the receipted MMAs) is refused at "fault".  "fault_xa" (peeled) wants every receipt unmoved, z moved
    only in row 0 of unit 0, by `xa_fault_expected` on CPU_CALLS and by ±256 B[:32, 0] elsewhere, and y = dequant of it."""
    from verity_vllm.engine import int8_exact as E
    g = win.caps.get(which, {})
    if "receipts" not in g or "a" not in g:
        return {"ok": False, "complete": False, "why": f"{which} step incomplete: {sorted(g)}"}
    m, s, t, L = g["m"], g["s"], win.t, win.L
    doc: dict = {"which": which, **F.form_of("dither-1s"), "complete": True, "s": s, "m": m, "t": t,
                 "poison": f"{DF.POISON:#04x}", "epsilon": "seeded"}
    key, W = win.key, win.window
    # commitments against commit.py
    bits = np.arange(t) < m
    wrows = CR.word_rows(CR.words(bits))
    a_cpu, s_cpu, z_cpu = {}, {}, {}
    for w in range(L):
        n, k = win.nk[w]
        q, sc = g["a"][w].cpu().numpy(), g["sa"][w].cpu().numpy()
        fa = {"window": W, "step": s, "linear": w, "module": win.names[w], "slots": t, "k": k, "schema": CM.SCHEMA}
        a_cpu[w] = CM.s8_table(CM.domain(fa, t, "a-table"), key, f"a/{s}/{w}", q, sc).root
        fs = {"window": W, "step": s, "linear": w, "module": win.names[w], "slots": t, "schema": CM.SCHEMA}
        sd = CM.domain(fs, len(wrows), "admit")
        s_cpu[w] = CM.levels(sd, [CM.leaf(sd, key, f"s/{s}/{w}", i, x)[0] for i, x in enumerate(wrows)])[-1][0]
        fz = {"window": W, "step": s, "linear": w, "module": win.names[w], "slots": t, "n": n, "schema": CM.SCHEMA}
        zd = CM.domain(fz, t, "z-table")
        zrows = g["z"][w].cpu().numpy().astype("<i4").view(np.uint8)
        z_cpu[w] = CM.levels(zd, [CM.leaf(zd, key, f"z/{s}/{w}", i, bytes(zrows[i]))[0] for i in range(t)])[-1][0]
    doc["a_roots_bad"] = [w for w in range(L) if a_cpu[w] != g["a_roots"][w]]
    doc["a_bad_diag"] = a_bad_diag(g, doc["a_roots_bad"])
    doc["admit_roots_bad"] = [w for w in range(L) if s_cpu[w] != g["admit_roots"][w]]
    doc["z_roots_bad"] = [w for w in range(L) if z_cpu[w] != g["z_roots"][w]]
    f_step = {"window": W, "step": s, "linears": L}
    a_step = CM.step_root(CM.domain(f_step, L, "a-step"), [a_cpu[w] for w in range(L)])
    admit_step = CM.step_root(CM.domain(f_step, L, "admit-step"), [s_cpu[w] for w in range(L)])
    z_step = CM.step_root(CM.domain(f_step, L, "z-step"), [z_cpu[w] for w in range(L)])
    rnd = CM.step_root(CM.domain({"window": W, "step": s, "leaves": "a-step root, admit-step root, z-step root"}, 3, "round"),
                       [a_step, admit_step, z_step])
    doc["round"] = {"a_step": a_step == g["leaves"][0], "admit_step": admit_step == g["leaves"][1],
                    "z_step": z_step == g["leaves"][2], "root": rnd == g["root"]}
    # receipts (h1r v1) and z (h1o_ref) on the saved slots, the window's salt and the CPU's roots
    r = H1R(lib_h1o)
    o = C.CDLL(lib_h1o)
    o.h1o_ref_launch.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p]
    shift = o.h1o_fault_shift()
    wbc = torch.zeros(L, 3, dtype=torch.int64, device=win.dev)
    live = torch.from_numpy((np.arange(t // 32) * 32 < m).astype(np.uint8)).to(win.dev)
    roots = [torch.from_numpy(np.concatenate([np.frombuffer(a_cpu[w], ">u8"), np.frombuffer(g["b_roots"][w], ">u8")])
                              .astype("<u8").view("<i8").copy()).to(win.dev) for w in range(L)]
    plan = r.plan([Call(g["a"][w], win.wt[w], s * win.u_step + win.offs[w], roots[w], live) for w in range(L)])
    r.launch(plan, cap_salt(win, g)[1], version=1)
    cb = r.lib.h1r_call_bytes()
    zref = [torch.empty(t, n, dtype=torch.int32, device=win.dev) for n, _ in win.nk]
    for w in range(L):
        rc = o.h1o_ref_launch(plan.table.data_ptr() + cb * w, win.units[w], plan.seeds.data_ptr(), zref[w].data_ptr(),
                              win.nk[w][0], wbc[w].data_ptr(), torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"h1o_ref_launch({w}) returned {rc}")
    zab = []
    if win.peel:
        for w in range(L):
            a = g["a"][w].to(torch.float64)
            a[m:] = 0
            zab.append(torch.cat([(a @ win.wt[w][i:i + 4096].to(torch.float64).T).to(torch.int32)
                                  for i in range(0, win.nk[w][0], 4096)], dim=1))
    torch.cuda.synchronize()
    doc["peel"] = win.peel
    got = g["receipts"]

    def got_of(w: int) -> torch.Tensor:
        o_, units, nw = win.plan.offs[w]
        return got[o_:o_ + units * nw].view(units, nw)

    delta = torch.zeros(1, dtype=torch.int64, device=win.dev) + (1 << shift)
    zbase = zab if win.peel else zref
    rc_bad, z_bad, y_bad, fault_bad, fault_moved = [], [], [], [], []
    for w in range(L):
        n, k = win.nk[w]
        rdiff = (plan.receipts(w) != got_of(w)).any(dim=1).nonzero().flatten().tolist()
        dz = (g["z"][w].to(torch.int64) - zbase[w].to(torch.int64)) & M32
        cells = dz.nonzero().tolist()
        if g["y"] is not None:
            yk = g["y"][w]
            if not torch.equal(E.dequantize(g["z"][w][:m], g["sa"][w][:m], win.sw[w]).view(torch.int16), yk.view(torch.int16)):
                y_bad.append(w)
        if which == "fault_xa":
            row = dz[0, :32].tolist()
            if w in cpu_calls:
                exp = xa_fault_expected(g["a"][w][:32].cpu().numpy(), win.wt[w][:32].cpu().numpy(),
                                        g["salt"] if "salt" in g else win.salt_bytes, s * win.u_step + win.offs[w],
                                        (a_cpu[w], g["b_roots"][w]))
                exact = row == exp
            else:
                b0 = win.wt[w][:32, 0].to(torch.int64).cpu()
                exact = row in (((256 * b0) & M32).tolist(), ((-256 * b0) & M32).tolist())
            if not (not rdiff and all(i == 0 and c < 32 for i, c in cells) and exact):
                fault_bad.append({"call": w, "receipt_units_off": rdiff[:4], "z_cells_off": cells[:4],
                                  "dz_row0": row[:4], "exact_checked": w in cpu_calls})
            fault_moved.append(len(cells))
        elif which == "fault":
            moved = g["y"] is not None and float(g["sa"][w][0]) != 0.0 and not torch.equal(
                E.dequantize(zbase[w][:1, :1], g["sa"][w][:1], win.sw[w][:1]).view(torch.int16), g["y"][w][:1, :1].view(torch.int16))
            if not (rdiff == [0] and cells == [[0, 0]] and int(dz[0, 0]) == int(delta) and moved):
                fault_bad.append({"call": w, "receipt_units_off": rdiff[:4], "z_cells_off": cells[:4],
                                  "dz00": int(dz[0, 0]), "y_moved": bool(moved)})
        elif which == "fault_wb":
            col = dz[:32, 0].tolist()
            if w in cpu_calls:
                exp = wb_fault_expected(g["a"][w][:32].cpu().numpy(), win.wt[w][:32].cpu().numpy(),
                                        g["salt"] if "salt" in g else win.salt_bytes, s * win.u_step + win.offs[w],
                                        (a_cpu[w], g["b_roots"][w]))
                exact = col == exp
            else:
                exact = all(v % 256 == 0 for v in col)
            if not (not rdiff and all(c == 0 and i < 32 for i, c in cells) and exact):
                fault_bad.append({"call": w, "receipt_units_off": rdiff[:4], "z_cells_off": cells[:4],
                                  "dz_col0": col[:4], "exact_checked": w in cpu_calls})
            fault_moved.append(len(cells))
        else:
            if rdiff:
                rc_bad.append(w)
            if cells:
                z_bad.append(w)
    doc.update(receipts_vs_h1r_bad=rc_bad, z_vs_ref_bad=z_bad, y_vs_dequant_z_bad=y_bad if g["y"] is not None else None)
    doc["live_units"] = int(sum((min(m, t) + 31) // 32 * (n // 32) for n, _ in win.nk))
    rb = (min(m, t) + 31) // 32
    cnt = wbc.cpu().numpy()
    ent = np.array([rb * n * k for n, k in win.nk], dtype=np.int64)
    doc["wb"] = {"plus": int(cnt[:, 0].sum()), "minus": int(cnt[:, 1].sum()), "entries": int(ent.sum()),
                 "density": float(cnt.sum() / ent.sum()),
                 "by_site": {site: float(cnt[i::4].sum() / ent[i::4].sum()) for i, site in enumerate(DF.SITES)}}
    # the activation wrap's entries (A = 127 and b = 1) each call's live units hold, by h1o_ref's own noise, and the
    # A = 127 entries of the call's live rows (each such entry meets every unit of its row tile)
    a127 = [int((g["a"][w][:m] == 127).sum()) for w in range(L)]
    doc["xa"] = {"entries": int(cnt[:, 2].sum()), "by_call": cnt[:, 2].tolist(), "calls_without": int((cnt[:, 2] == 0).sum()),
                 "min": int(cnt[:, 2].min()), "max": int(cnt[:, 2].max()),
                 "by_site": {site: int(cnt[i::4, 2].sum()) for i, site in enumerate(DF.SITES)},
                 "a_eq_127_rows": {"total": sum(a127), "min": min(a127), "max": max(a127)}}
    if which in ("fault", "fault_wb", "fault_xa"):
        doc["fault_shift"] = shift
        doc["fault_bad"] = fault_bad
        if which != "fault":
            doc["z_cells_moved"] = {"min": min(fault_moved), "max": max(fault_moved)}
        doc["conforming"] = (not fault_bad and not y_bad and not doc["a_roots_bad"] and all(doc["round"].values())
                             and not doc["z_roots_bad"])
        doc["ok"] = doc["conforming"]
        return doc
    # the served y against today's (gated peeled; informational unpeeled, as the dithered product is served)
    if g["y"] is not None and which == "gate":
        eq, rel = 0, []
        for w in range(L):
            # the op's own body, not the op: an -xl arm serves verity_int8::scaled_mm from exactlin
            yt = E.dequantize(E.int_mm(g["a"][w][:m], win.wt[w]), g["sa"][w][:m], REG[w][1].weight_scale).float()
            yk = g["y"][w].float()
            eq += int(torch.equal(yt, yk))
            rel.append(float((yk - yt).norm() / yt.norm().clamp_min(1e-30)))
        doc["y_vs_today"] = {"calls_equal": eq, "rel_l2_median": float(np.median(rel)), "rel_l2_max": float(np.max(rel))}
    # H1V3's Definition on the CPU, a few units of row tile 0
    units_doc, h1v3_in, kernel_out = [], [], []
    if which == "gate":
        rng = np.random.default_rng(s)
        salt = g["salt"] if "salt" in g else win.salt_bytes
        for w in [0, 3, 2][:n_cpu_units]:
            n, k = win.nk[w]
            u = int(rng.integers(n // 32))
            a32 = g["a"][w][0:32].cpu().numpy()
            b32 = win.wt[w][32 * u:32 * u + 32].cpu().numpy()
            pos = s * win.u_step + win.offs[w] + u
            ref = h1v3_unit(a32, b32, salt, pos, (a_cpu[w], g["b_roots"][w]))
            zk = g["z"][w][0:32, 32 * u:32 * u + 32].cpu().numpy().astype(np.int64)
            rk = (got_of(w)[u].cpu().numpy().astype(np.int64) & M32).tolist()
            units_doc.append({"call": w, "unit": u, "k": k,
                              "z_equal": bool((zk == (ref["today"] if win.peel else ref["z"])).all()),
                              "receipt_equal": rk == ref["receipt"],
                              "x_wraps": ref["x_wraps"],
                              "cells_peel_ca_ne_today": int((ref["peel_ca"] != ref["today"]).sum()),
                              "cells_peel_xw_ne_today": int((ref["peel_xw"] != ref["today"]).sum()),
                              "cells_peel_kernel_ne_today": int((ref["peel_kernel"] != ref["today"]).sum()),
                              "cells_z_ne_x_bc": int((ref["z"] != ref["xw"]).sum()),
                              "cells_z_ne_ab_bc": int((ref["z"] != ref["dithered"]).sum()),
                              "cells_z_ne_today": int((ref["z"] != ref["today"]).sum()),
                              "wb_plus": ref["wb_plus"], "wb_minus": ref["wb_minus"], "wb_entries": ref["entries"],
                              "a_eq_127": int((a32 == 127).sum()), "b_abs_gt_62": int((np.abs(b32.astype(np.int64)) > 62).sum())})
            h1v3_in.append({"a": (a32.astype(np.int64) & 0xFF).tolist(), "b": (b32.astype(np.int64) & 0xFF).tolist(),
                            "salt": salt.hex(), "position": int(pos), "roots": [a_cpu[w].hex(), g["b_roots"][w].hex()]})
            kernel_out.append((zk, rk, ref))
        hv = h1v3_itself(h1v3_in)
        if "units" in hv:
            for x, (zk, rk, ref), hu in zip(units_doc, kernel_out, hv["units"]):
                hz = np.array(hu["z"], dtype=np.int64)
                if hv.get("peeled"):    # H1V3's z is A B^T (its PouwPeelSeg): unpeeled, X (B + c)^T on the CPU from it
                    if not win.peel:
                        hz = _s32(hz - ref["today"] + ref["xw"])
                elif win.peel:          # an H1V3 without the peel: its z peeled on the CPU
                    x["h1v3_peel_ca_cells_ne_today"] = int((_s32(hz - ref["peel"]) != ref["today"]).sum())
                    hz = _s32(hz - ref["peel"] + ref["xw_term"])
                x["h1v3_z_equal"] = bool((zk == hz).all())
                x["h1v3_receipt_equal"] = rk == hu["receipt"]
        doc["h1v3"] = {k_: v for k_, v in hv.items() if k_ != "units"}
    doc["vs_h1v3_cpu"] = units_doc
    h1v3_ok = which != "gate" or ("units" in hv and all(x["h1v3_z_equal"] and x["h1v3_receipt_equal"] for x in units_doc))
    today_ok = not (win.peel and which == "gate") or doc.get("y_vs_today", {}).get("calls_equal") == L
    doc["ok"] = (not rc_bad and not z_bad and (g["y"] is None or not y_bad) and today_ok
                 and all(x["z_equal"] and x["receipt_equal"] and x["cells_z_ne_x_bc"] == 0
                         and (not win.peel or x["cells_peel_xw_ne_today"] == x["cells_peel_kernel_ne_today"] == 0)
                         for x in units_doc)
                 and h1v3_ok and not doc["a_roots_bad"] and not doc["admit_roots_bad"] and not doc["z_roots_bad"]
                 and all(doc["round"].values()))
    return doc


def gate_verdict(gate: dict, control: dict, fault: dict, fault_wb: dict, fault_xa: dict | None = None) -> dict:
    """The poisoned gate passes, its no-write control (complete) is refused on z and on the receipts, the fault step
    conforms (the one perturbed noised word moved that cell's z and y and that unit's receipt, and nothing else) and so
    does the flipped wb⁺ bit (only column 0 of unit 0's z and y moved, by 256 (1 - 2 wb⁺) X, no receipt); peeled, so does
    the flipped xa bit (only row 0 of unit 0's z and y, by 256 (1 - 2 xa) B[:, 0], no receipt), which a peeled gate needs."""
    refused = (control.get("complete") and not control.get("ok") and bool(control.get("z_vs_ref_bad"))
               and bool(control.get("receipts_vs_h1r_bad")))
    xa_ok = bool(fault_xa.get("conforming")) if fault_xa is not None else not gate.get("peel")
    ok = (bool(gate.get("ok")) and bool(refused) and bool(fault.get("conforming")) and bool(fault_wb.get("conforming"))
          and xa_ok)
    return {"ok": ok, **F.form_of("dither-1s"), "gate_ok": bool(gate.get("ok")), "control_refused": bool(refused),
            "fault_conforming": bool(fault.get("conforming")), "fault_wb_conforming": bool(fault_wb.get("conforming")),
            "fault_xa_conforming": None if fault_xa is None else bool(fault_xa.get("conforming")),
            "control_bad": {k: len(control.get(k) or []) for k in ("z_vs_ref_bad", "y_vs_dequant_z_bad", "receipts_vs_h1r_bad",
                                                                    "z_roots_bad")}}
