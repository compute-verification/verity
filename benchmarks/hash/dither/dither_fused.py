"""Dither's fused served linear in compiled vLLM: every int8-exact linear's y from `h1f.cu`'s one GEMM (z = xq W^T, the
H1 tile's exact product, with the int8-exact dequantization in its epilogue), which also computes the previous step's H1 +
fx1 receipts; z committed per step before the step's coin, like root(A).

    import dither_fused as DF
    DF.patch(t=T, h1c="libh1c.so", h1f="libh1f.so")   # before LLM(): Int8PlainLinearMethod.apply -> row_quant + dither::fused_mm
    llm = LLM(model=int8_exact_checkpoint, ...)        # compiled, CUDA graphs: the op is opaque to Inductor, its launch captured
    DF.hook(llm)                                       # the step hook below

Per engine step s (parity p = s mod 2), around the runner's `execute_model`:
    before   the step struct (p, m, receipts on, flags) written to the device, m the step's scheduled tokens; the forward's
             128 `h1f` launches read it: z_s, y_s, A_s's slots; step s - 1's receipts
    after    on the device: step s's A tables (`commit.s8_table`'s format, labels a/<s>/<w>), admit tables (the T / 16 words of
             the slots < m, s/<s>/<w>), z tables (each slot row's N int32 LE, z/<s>/<w>, domain 'z-table'), the a-step,
             admit-step and z-step trees and the round over the three ('round'); step s - 1's receipt trees (r/<s-1>/<w>); the
             round root read back, recorded through `sampled_proofs.challenger.LiveRecord` before its coin opens
             (the window's coins), the salt `h1tile.live_salt`, step s's seeds (`h1r_launch` version 0) into the buffer
             step s + 1's launches read

The window's coins are served_h1's (`coins.open_coins`, a live window of `ROUNDS` rounds at epoch 1): with
``$POUW_CHALLENGER`` (HOST:PORT), ``$POUW_CHALLENGER_SESSION`` and ``$POUW_CHALLENGER_SECRET_FILE`` (the session the
challenger's gate opened, and its secret: `coins.session`) the challenger off the node, one session a window, opened at
the window's build; else `window.live_challenger` in this process.  `stats` says which (``coins``: ``off-node session ID at
HOST:PORT (unchecked)`` or ``in-process (not certified)``).  Only the round's coin changes source: the salt, the seeds
and what they read are the same either way.

With ``rc="side"`` (`patch`) no `h1f` launch computes receipts: after step s's coin, its seeds, units (`h1r_launch` version
2) and receipt trees go on a second stream and run beside the engine's step s + 1, whose main stream first waits for step
s - 1's (they read the slots step s + 1 overwrites).  What is committed, before which coin, and every receipt are the same.

Labelled simplifications (none changes a step's cost): the admitted slots are the step's first m rows (SERVE-H1 admits
from its queue); the window's registration, queue commit and close are not run (once a window, not a step); a window is
the engine's life up to `ROUNDS` steps.  The z table and the third round leaf are this harness's rendering of the
credit's form (a z table in the row-table format, a root per step before the coin); the credit fixes the kind, fields
and labels.  Its stats, checks and verdict say form=beside (`forms.form_of`): not `forms.FORM_A`.
"""
from __future__ import annotations

import ctypes as C
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT / "benchmarks" / "pouw" / "served_h1", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import coins as CO  # noqa: E402
import commit as CM  # noqa: E402
import credit as CR  # noqa: E402
import forms as F  # noqa: E402
import serve as SV  # noqa: E402
from h1c import H1C, Dom, Tab, unwords, words  # noqa: E402
from h1r import H1R, Call  # noqa: E402

SITES = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
MODEL = {"repo": "verity-int8/Llama-3.1-8B-INT8"}
ROUNDS = 1 << 14
F_DUMP, F_NW_TILES, F_NW_UNITS = 1, 2, 4    # h1f.cu's H1FStep.flags
POISON = 0xA5
RC_MODES = ("fused", "side")
LIB = "dither"
REG: dict[int, tuple[str, torch.nn.Module]] = {}
CFG: dict = {}
WIN: "Window | None" = None


def index_of(prefix: str) -> int:
    parts = prefix.split(".")
    i = parts.index("layers")
    return 4 * int(parts[i + 1]) + SITES.index(".".join(parts[i + 2:]))


class _F(C.Structure):
    _fields_ = [("slot", C.c_void_p * 2), ("scale", C.c_void_p * 2), ("z", C.c_void_p * 2), ("rc", C.c_void_p * 2),
                ("ydump", C.c_void_p), ("t", C.c_int), ("n", C.c_int), ("k", C.c_int), ("pad", C.c_int)]


class Window:
    """A window's device state: per linear two parities of A, scale and z slots; the h1c sets (B tables once; per parity
    the A, admit and z tables, the step trees and the round); the receipts' call tables (one per parity, sharing one seeds
    and one receipts buffer) and the receipt trees; the `h1f` table and step struct; the live coins."""

    def __init__(self, t: int, key: bytes, h1c_path: str, h1f_path: str, gate: bool, dev, rc: str = "fused"):
        if rc not in RC_MODES:
            raise ValueError(f"receipts mode {rc!r}, not one of {RC_MODES}")
        self.t, self.key, self.dev, self.rc_mode = t, key, dev, rc
        self.L = L = len(REG)
        if sorted(REG) != list(range(L)):
            raise RuntimeError(f"linears registered: {sorted(REG)}")
        self.names = [REG[w][0] for w in range(L)]
        self.wt = [REG[w][1].weight for w in range(L)]
        self.sw = [REG[w][1].weight_scale.reshape(-1) for w in range(L)]
        self.nk = [(int(x.shape[0]), int(x.shape[1])) for x in self.wt]
        self.slot = [[torch.zeros(t, k, dtype=torch.int8, device=dev) for _ in range(2)] for _, k in self.nk]
        self.scale = [[torch.zeros(t, dtype=torch.float32, device=dev) for _ in range(2)] for _ in self.nk]
        self.z = [[torch.zeros(t, n, dtype=torch.int32, device=dev) for _ in range(2)] for n, _ in self.nk]
        self.live = [torch.zeros(t // 32, dtype=torch.uint8, device=dev) for _ in range(2)]
        self.admit = [torch.zeros(t // 16 * 2, dtype=torch.uint8, device=dev) for _ in range(2)]
        self.ydump = [torch.zeros(t, n, dtype=torch.bfloat16, device=dev) for n, _ in self.nk] if gate else None
        self.offs, self.u_step = SV.static_layout([n for n, _ in self.nk], t)
        self.window = f"dither-{os.getpid()}-0"
        self.c = H1C(h1c_path, key, dev)
        self.step0 = torch.zeros(1, dtype=torch.int32, device=dev)
        self.bset = self.c.set([Tab(Dom("b-table", SV.b_fields(MODEL, w, self.names[w], n, k), n), n,
                                    data=self.wt[w].view(torch.uint8), stride=k, body=k, length=k + 4,
                                    tail=self.sw[w].view(torch.uint8), tail_stride=4, lab=(f"b/{w}", ""), labs=False)
                                for w, (n, k) in enumerate(self.nk)])
        self.c.build(self.bset, self.step0)
        self.pre = [self._pre_sets(p) for p in range(2)]
        self.r = H1R(h1f_path)
        self.plans = [self.r.plan([Call(self.slot[w][p], self.wt[w], self.offs[w], self.pre[p]["mid"].leaves_addr(0) + 64 * w,
                                        self.live[p], root_b=self.bset.roots.data_ptr() + 64 * w) for w in range(L)])
                      for p in range(2)]
        self.plans[1].seeds, self.plans[1].out = self.plans[0].seeds, self.plans[0].out
        rt = []
        for w, (n, k) in enumerate(self.nk):
            o, units, nw = self.plans[0].offs[w]
            fr = {"window": self.window, "step": 0, "linear": w, "module": self.names[w], "base": 0, "units": units,
                  "words": nw, "schema": CM.SCHEMA}
            rt.append(Tab(Dom("receipts", fr, units, {"step": (1, 0), "base": (self.u_step, self.offs[w])}), units,
                          data=self.plans[0].receipts(w).view(torch.uint8), stride=4 * nw, body=4 * nw, length=4 * nw,
                          lab=("r/", f"/{w}")))
        self.rset = self.c.set(rt)
        self.f = C.CDLL(h1f_path)
        self.f.h1f_launch.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p,
                                      C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
        if self.f.h1f_struct_bytes(0) != C.sizeof(_F) or self.f.h1f_struct_bytes(1) != 16:
            raise RuntimeError("h1f.cu's structs differ from dither_fused's")
        call_bytes = self.r.lib.h1r_call_bytes()
        tab = (_F * L)()
        for w, (n, k) in enumerate(self.nk):
            e = _F()
            for p in range(2):
                e.slot[p], e.scale[p], e.z[p] = self.slot[w][p].data_ptr(), self.scale[w][p].data_ptr(), self.z[w][p].data_ptr()
                e.rc[p] = self.plans[p].table.data_ptr() + call_bytes * w
            e.ydump = self.ydump[w].data_ptr() if gate else None
            e.t, e.n, e.k = t, n, k
            tab[w] = e
        self.F = torch.frombuffer(bytearray(bytes(tab)), dtype=torch.uint8).to(dev)
        self.fsize = C.sizeof(_F)
        self.S = torch.zeros(4, dtype=torch.int32, device=dev)
        self.units = [self.plans[0].offs[w][1] for w in range(L)]
        self.salt = torch.zeros(8, dtype=torch.int64, device=dev)
        self.step_base = torch.zeros(1, dtype=torch.int64, device=dev)
        self.stepv = torch.zeros(2, dtype=torch.int32, device=dev)
        self.s, self.parity, self.rc_on, self.flags, self.m = 0, 0, 0, 0, 0
        self.calls = 0
        self.steps = 0
        self.arms: list[str] = []
        self.arm_m: int | None = None
        self.caps: dict[str, dict] = {}
        self.cap_step: str | None = None
        self.cap_rc: str | None = None
        self._next_rc: str | None = None
        self.price_y = False
        self.yset = None
        if gate:
            self.yset = self.c.set([Tab(Dom("y-table", {"window": self.window, "step": 0, "linear": w, "module": self.names[w],
                                                        "slots": t, "n": n, "schema": CM.SCHEMA}, t, {"step": (1, 0)}), t,
                                        data=self.ydump[w].view(torch.uint8), stride=2 * n, body=2 * n, length=2 * n,
                                        lab=("y/", f"/{w}")) for w, (n, k) in enumerate(self.nk)])
        self.round_log: list[dict] = []
        self.phases: list[dict] = []
        self.t_before = 0.0
        self.ev = [torch.cuda.Event(enable_timing=True) for _ in range(9)]
        if rc == "side":
            self.side = torch.cuda.Stream(device=dev)
            self.ready, self.rc_done = torch.cuda.Event(), torch.cuda.Event()
            self.rc_pending = False
            self.salt_p = [torch.zeros(8, dtype=torch.int64, device=dev) for _ in range(2)]
            self.base_p = [torch.zeros(1, dtype=torch.int64, device=dev) for _ in range(2)]
            self.rstep = [torch.zeros(1, dtype=torch.int32, device=dev) for _ in range(2)]
            self.sev = [[torch.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(2)]
            self.sev_set = [False, False]
        self._coins()

    def arm(self, m: int, names: tuple = ("gate", "control")) -> None:
        """Capture `names` in turn, each at the next step of m tokens once the one before has its receipts: its parity's
        A, scale and z slots and the y dump poisoned with 0xA5 before its forward, the receipts and seeds before they are
        written; "control" also with the no-write flags (`F_NW_TILES` on its step, `F_NW_UNITS` on the next), so the
        transcript buffers keep the poison and `check` must refuse it."""
        self.arms, self.arm_m = list(names), m
        self.caps = {n: {} for n in names}

    def _poison(self, ts) -> None:
        for x in ts:
            x.view(torch.uint8).fill_(POISON)

    def _coins(self) -> None:
        from verity.core.protocols.pouw.circuit import h1tile as HT
        self.HT = HT
        self.coins = CO.open_coins(CO.session(), 1, ROUNDS)
        self.rounds = self.coins.rounds

    def _pre_sets(self, p: int) -> dict:
        c, W, t, L = self.c, self.window, self.t, self.L
        f_step = {"window": W, "step": 0, "linears": L}
        rnd = c.set([Tab(Dom("round", {"window": W, "step": 0, "leaves": "a-step root, admit-step root, z-step root"}, 3,
                             {"step": (1, 0)}), 3)])
        mid = c.set([Tab(Dom("a-step", f_step, L, {"step": (1, 0)}), L, out=rnd.leaves_addr(0)),
                     Tab(Dom("admit-step", f_step, L, {"step": (1, 0)}), L, out=rnd.leaves_addr(0) + 64),
                     Tab(Dom("z-step", f_step, L, {"step": (1, 0)}), L, out=rnd.leaves_addr(0) + 128)])
        tabs = []
        for w, (n, k) in enumerate(self.nk):
            fa = {"window": W, "step": 0, "linear": w, "module": self.names[w], "slots": t, "k": k, "schema": CM.SCHEMA}
            tabs.append(Tab(Dom("a-table", fa, t, {"step": (1, 0)}), t, data=self.slot[w][p].view(torch.uint8), stride=k,
                            body=k, length=k + 4, tail=self.scale[w][p].view(torch.uint8), tail_stride=4,
                            lab=("a/", f"/{w}"), out=mid.leaves_addr(0) + 64 * w))
        for w, (n, k) in enumerate(self.nk):
            fs = {"window": W, "step": 0, "linear": w, "module": self.names[w], "slots": t, "schema": CM.SCHEMA}
            tabs.append(Tab(Dom("admit", fs, t // 16, {"step": (1, 0)}), t // 16, data=self.admit[p], stride=2, body=2,
                            length=2, lab=("s/", f"/{w}"), out=mid.leaves_addr(1) + 64 * w))
        zt = []
        for w, (n, k) in enumerate(self.nk):
            fz = {"window": W, "step": 0, "linear": w, "module": self.names[w], "slots": t, "n": n, "schema": CM.SCHEMA}
            zt.append(Tab(Dom("z-table", fz, t, {"step": (1, 0)}), t, data=self.z[w][p].view(torch.uint8), stride=4 * n,
                          body=4 * n, length=4 * n, lab=("z/", f"/{w}"), out=mid.leaves_addr(2) + 64 * w))
        return {"round": rnd, "mid": mid, "calls": c.set(tabs), "z": c.set(zt)}

    def launch(self, w: int, xq: torch.Tensor, s_a: torch.Tensor, wt: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor) -> None:
        n, k = self.nk[w]
        units = self.units[w] if self.rc_mode == "fused" else 0
        rc = self.f.h1f_launch(self.F.data_ptr() + self.fsize * w, self.S.data_ptr(), self.t, n, k, units,
                               xq.data_ptr(), s_a.data_ptr(), int(xq.shape[0]), wt.data_ptr(), s_w.data_ptr(), y.data_ptr(),
                               self.plans[0].seeds.data_ptr(), self.plans[0].out.data_ptr(),
                               torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"h1f_launch({w}, M={xq.shape[0]}) returned {rc}")
        self.calls += 1

    # -- the step hook ---------------------------------------------------------------------------------------------------

    def before(self, m: int) -> None:
        if m > self.t:
            raise RuntimeError(f"a step of {m} tokens in a window of T = {self.t} slots")
        self.m = m
        self.flags = 0
        self.cap_step = self.cap_rc = None
        if self._next_rc is not None:
            self.cap_rc, self._next_rc = self._next_rc, None
            if self.rc_mode == "fused":
                self._poison([self.plans[0].out])
                if self.cap_rc == "control":
                    self.flags |= F_NW_UNITS
        elif self.arms and m == self.arm_m:
            self.cap_step = name = self.arms.pop(0)
            p = self.parity
            self._poison([x[p] for x in self.slot] + [x[p] for x in self.scale] + [x[p] for x in self.z] + list(self.ydump or []))
            self.flags |= F_DUMP | (F_NW_TILES if name == "control" else 0)
            self.caps[name] = {"s": self.s, "m": m}
        rc_on = self.rc_on if self.rc_mode == "fused" else 0
        self.S.copy_(torch.tensor([self.parity, m, rc_on, self.flags], dtype=torch.int32))
        if self.cap_step or self.cap_rc:
            torch.cuda.synchronize()
        self.t_before = time.perf_counter()

    def after(self) -> None:
        side = self.rc_mode == "side"
        main = torch.cuda.current_stream()
        if side:
            main.synchronize()
        else:
            torch.cuda.synchronize()
        t_fwd = time.perf_counter()
        s, p, m, t = self.s, self.parity, self.m, self.t
        ev = self.ev
        side_ms = {}
        if side and self.sev_set[p]:   # step s - 2's receipts: forward s ran after the main stream waited on them
            sv = self.sev[p]
            side_ms = {"side_h1r_ms": sv[0].elapsed_time(sv[1]), "receipt_trees_ms": sv[1].elapsed_time(sv[2])}
        bits = np.arange(t) < m
        self.live[p].copy_(torch.from_numpy((np.arange(t // 32) * 32 < m).astype(np.uint8)))
        self.admit[p].copy_(torch.frombuffer(bytearray(b"".join(CR.word_rows(CR.words(bits)))), dtype=torch.uint8))
        self.stepv.copy_(torch.tensor([s, max(s - 1, 0)], dtype=torch.int32))
        pre = self.pre[p]
        ev[0].record()
        self.c.build(pre["calls"], self.stepv[0:1])
        ev[1].record()
        self.c.build(pre["z"], self.stepv[0:1])
        ev[2].record()
        self.c.build(pre["mid"], self.stepv[0:1])
        self.c.build(pre["round"], self.stepv[0:1])
        ev[3].record()
        if self.rc_on and not side:
            self.c.build(self.rset, self.stepv[1:2])
        ev[4].record()
        if self.cap_rc is not None and not side:
            self.caps[self.cap_rc]["receipts"] = self.plans[0].out.clone()
            self.caps[self.cap_rc]["r_roots"] = self.rset.roots.clone()
        root = unwords(pre["round"].root(0))
        t_root = time.perf_counter()
        opening = self.rounds.record(root)
        salt = self.HT.live_salt(self.coins.commitment, opening.coin, s, root)
        t_coin = time.perf_counter()
        if side:
            self._side_receipts(main, s, p, salt)
        else:
            self.salt.copy_(torch.from_numpy(words(salt)))
            self.step_base.fill_(s * self.u_step)
            if self.cap_step is not None:
                self._poison([self.plans[0].seeds])
            ev[5].record()
            self.r.launch(self.plans[p], self.salt, version=0, step_base=self.step_base)
            ev[6].record()
            ev[6].synchronize()
        t_end = time.perf_counter()
        ph = {"m": m, "forward_ms": 1e3 * (t_fwd - self.t_before), "a_admit_tables_ms": ev[0].elapsed_time(ev[1]),
              "z_tables_ms": ev[1].elapsed_time(ev[2]), "step_trees_ms": ev[2].elapsed_time(ev[3]),
              "coin_host_ms": 1e3 * (t_coin - t_root), "between_ms": 1e3 * (t_end - t_fwd), **side_ms}
        if not side:
            ph.update(receipt_trees_ms=ev[3].elapsed_time(ev[4]), seeds_ms=ev[5].elapsed_time(ev[6]))
        if self.price_y and self.yset is not None:
            ev[7].record()
            self.c.build(self.yset, self.stepv[0:1])
            ev[8].record()
            ev[8].synchronize()
            ph["y_tables_ms"] = ev[7].elapsed_time(ev[8])
        self.phases.append(ph)
        if self.cap_step is not None:
            nm, self._next_rc = self.cap_step, self.cap_step
            self.caps[nm].update(p=p, root=root, salt=salt, a=[x[p].clone() for x in self.slot],
                                 sa=[x[p].clone() for x in self.scale], z=[x[p].clone() for x in self.z],
                                 y=[x[:m].clone() for x in self.ydump] if self.ydump else None,
                                 a_roots=[unwords(pre["mid"].leaf(0, w)) for w in range(self.L)],
                                 z_roots=[unwords(pre["mid"].leaf(2, w)) for w in range(self.L)],
                                 admit_roots=[unwords(pre["mid"].leaf(1, w)) for w in range(self.L)],
                                 leaves=[unwords(pre["round"].leaf(0, i)) for i in range(3)],
                                 b_roots=[unwords(self.bset.root(w)) for w in range(self.L)])
        self.round_log.append({"s": s, "m": m, "root": root.hex()[:32]})
        self.rc_on = 1
        self.parity ^= 1
        self.s += 1
        self.steps += 1
        if self.s == ROUNDS:
            raise RuntimeError(f"the window's {ROUNDS} rounds are spent")

    def _side_receipts(self, main, s: int, p: int, salt: bytes) -> None:
        """Step s's seeds, units (`h1r_launch` version 2) and receipt trees on the side stream, after its coin, while the
        engine runs step s + 1.  The main stream first waits for step s - 1's: forward s + 1 overwrites the slots they
        read, and they share the seeds and receipts buffers; salt, base and step are per parity for the same reason."""
        self.salt_p[p].copy_(torch.from_numpy(words(salt)))
        self.base_p[p].fill_(s * self.u_step)
        self.rstep[p].fill_(s)
        if self.rc_pending:
            main.wait_event(self.rc_done)
        if self.cap_rc is not None:
            self.caps[self.cap_rc]["receipts"] = self.plans[0].out.clone()
            self.caps[self.cap_rc]["r_roots"] = self.rset.roots.clone()
        if self.cap_step is not None:
            self._poison([self.plans[0].seeds, self.plans[0].out])
        self.ready.record(main)
        sv = self.sev[p]
        with torch.cuda.stream(self.side):
            self.side.wait_event(self.ready)
            sv[0].record()
            self.r.launch(self.plans[p], self.salt_p[p], nowrite=self.cap_step == "control", version=2,
                          step_base=self.base_p[p])
            sv[1].record()
            self.c.build(self.rset, self.rstep[p])
            sv[2].record()
            self.rc_done.record()
        self.rc_pending = True
        self.sev_set[p] = True

    def stats(self) -> dict:
        """Steps so far and the per-step phase medians since the last call, by step size m (decode B32: m = 32)."""
        ph, self.phases = self.phases, []
        by: dict = {}
        for x in ph:
            by.setdefault(x["m"], []).append(x)
        med = {str(m): {"steps": len(xs), **{k: round(float(np.median([x[k] for x in xs if k in x])), 4)
                                             for k in dict.fromkeys(k for x in xs for k in x) if k != "m"}}
               for m, xs in sorted(by.items(), key=lambda kv: -len(kv[1]))[:4]}
        return {**F.form_of("dither-fused"), "steps": self.steps, "fused_calls_eager": self.calls, "t": self.t, "s": self.s,
                "coins": self.coins.label, "phases_by_m": med}


# -- the op and the patch --------------------------------------------------------------------------------------------------

def _window(dev) -> Window:
    global WIN
    if WIN is None:
        WIN = Window(CFG["t"], CFG["key"], CFG["h1c"], CFG["h1f"], CFG["gate"], dev, CFG["rc"])
    return WIN


@torch.library.custom_op(f"{LIB}::fused_mm", mutates_args=())
def fused_mm(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor, idx: int) -> torch.Tensor:
    y = torch.empty((xq.shape[0], w.shape[0]), dtype=torch.bfloat16, device=xq.device)
    if xq.shape[0]:
        _window(xq.device).launch(idx, xq.contiguous(), s_a.contiguous(), w, s_w.reshape(-1), y)
    return y


@fused_mm.register_fake
def _fused_mm_fake(xq, s_a, w, s_w, idx):
    return xq.new_empty((xq.shape[0], w.shape[0]), dtype=torch.bfloat16)


def patch(t: int, h1c: str, h1f: str, key: bytes | None = None, gate: bool = False, rc: str = "fused") -> None:
    """Every int8-exact linear through ``dither::fused_mm`` (call before the engine is built, in its process); `rc` where
    the receipts run: "fused" in the next step's `h1f` launches, "side" in one `h1r` launch on another stream."""
    from verity_vllm.engine import int8_ops
    from verity_vllm.engine import int8_vllm as IV
    CFG.update(t=t, h1c=h1c, h1f=h1f, key=key or os.urandom(64), gate=gate, rc=rc)
    IV.register()
    orig_pw = IV.Int8PlainLinearMethod.process_weights_after_loading

    def process_weights_after_loading(self, layer):
        orig_pw(self, layer)
        self.dither_idx = index_of(self.prefix)
        REG[self.dither_idx] = (self.prefix, layer)

    def apply(self, layer, x, bias=None):
        if bias is not None or x.dtype != torch.bfloat16:
            raise ValueError(f"{self.prefix}: the fused arm takes bf16 activations and no bias")
        lead, k = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, k)
        n = layer.weight.shape[0]
        if x2.shape[0] == 0:
            return x.new_empty(*lead, n)
        xq, s_a = int8_ops.row_quant(x2)
        return torch.ops.dither.fused_mm(xq, s_a, layer.weight, layer.weight_scale, self.dither_idx).reshape(*lead, n)

    IV.Int8PlainLinearMethod.process_weights_after_loading = process_weights_after_loading
    IV.Int8PlainLinearMethod.apply = apply


def hook(llm) -> Window:
    """The step hook around the engine's runner (`execute_model`), and the window (built at the engine's first forward)."""
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


def int8_checkpoint(pin: str, scratch: Path) -> Path:
    """The pin's files linked into SCRATCH under the int8-exact linears' quantization config."""
    import json
    d = scratch / "int8-exact-ckpt"
    d.mkdir(parents=True, exist_ok=True)
    for f in Path(pin).iterdir():
        if f.name != "config.json" and not (d / f.name).exists():
            (d / f.name).symlink_to(f.resolve())
    cfg = json.loads((Path(pin) / "config.json").read_text())
    cfg["quantization_config"] = {"quant_method": "verity_int8", "weight_max": 127}
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    return d


# -- the gate ----------------------------------------------------------------------------------------------------------------

def check(win: Window, lib_h1r: str, which: str = "gate", n_ref_units: int = 3,
          cpu_calls: tuple = (0, 1, 2, 3, 126, 127)) -> dict:
    """Capture `which`'s checks (`Window.arm`), after the step after it ran (its receipts): per call z = torch._int_mm
    (cuBLASLt) on the slots' rows < m and zero above, y (dumped) = int8_ops.scaled_mm's body on the same rows (today's served y),
    every receipt = h1r v1 (another unit implementation than the served v2's, gated against receipts.py) on the saved
    slots, a few units = h1tile.reference (z and receipt),
    every call's A, admit and z roots, the receipt roots of CPU_CALLS and the step's round = commit.py's.  Every buffer
    read was 0xA5 before the kernels wrote it; the control capture's kernels wrote none of them, so it must fail."""
    from verity_vllm.engine import int8_exact as E
    g = win.caps.get(which, {})
    if "receipts" not in g or "a" not in g:
        return {"ok": False, "complete": False, "why": f"{which} step incomplete: {sorted(g)}"}
    m, s, t, L = g["m"], g["s"], win.t, win.L
    doc: dict = {"which": which, **F.form_of("dither-fused"), "complete": True, "s": s, "m": m, "t": t,
                 "poison": f"{POISON:#04x}"}
    z_bad, y_bad, zero_bad = [], [], []
    for w in range(L):
        a, sa, z = g["a"][w], g["sa"][w], g["z"][w]
        zr = E.int_mm(a[:m], win.wt[w])
        if not torch.equal(zr, z[:m]):
            z_bad.append(w)
        if bool(z[m:].any()) or bool(a[m:].any()) or bool(sa[m:].any()):
            zero_bad.append(w)
        if g["y"] is not None:
            # the op's own body, not the op: an -xl arm serves verity_int8::scaled_mm from exactlin
            yr = E.dequantize(zr, sa[:m], REG[w][1].weight_scale)
            if not torch.equal(yr.view(torch.int16), g["y"][w].view(torch.int16)):
                y_bad.append(w)
    doc.update(z_vs_int_mm_bad=z_bad, y_vs_scaled_mm_bad=y_bad if g["y"] is not None else None, empty_rows_bad=zero_bad)
    # every receipt against h1r on the saved slots
    r = H1R(lib_h1r)
    live = torch.from_numpy((np.arange(t // 32) * 32 < m).astype(np.uint8)).to(win.dev)
    roots = [torch.from_numpy(np.concatenate([np.frombuffer(g["a_roots"][w], ">u8"), np.frombuffer(g["b_roots"][w], ">u8")])
                              .astype("<u8").view("<i8").copy()).to(win.dev) for w in range(L)]
    plan = r.plan([Call(g["a"][w], win.wt[w], s * win.u_step + win.offs[w], roots[w], live) for w in range(L)])
    salt = torch.from_numpy(words(g["salt"])).to(win.dev)
    r.launch(plan, salt, version=1)
    torch.cuda.synchronize()
    got = g["receipts"]

    def got_of(w: int) -> torch.Tensor:
        o, units, nw = win.plans[0].offs[w]
        return got[o:o + units * nw].view(units, nw)
    rc_bad = [w for w in range(L) if not torch.equal(plan.receipts(w), got_of(w))]
    doc["receipts_vs_h1r_bad"] = rc_bad
    doc["live_units"] = int(sum((min(m, t) + 31) // 32 * (n // 32) for n, _ in win.nk))
    # h1tile.reference on a few live units: z and receipt
    import receipts as RC
    refs = []
    rng = np.random.default_rng(s)
    for w in [0, 3, 2][:n_ref_units]:
        n, k = win.nk[w]
        u = int(rng.integers(n // 32))          # row tile 0, column tile u
        zz, rr = RC.reference(RC.Call(g["a"][w], win.wt[w]), g["salt"], s * win.u_step + win.offs[w], g["a_roots"][w],
                              g["b_roots"][w], u)
        rk = (got_of(w)[u].cpu().numpy().astype(np.int64) & 0xFFFFFFFF).tolist()
        zk = g["z"][w][0:32, 32 * u:32 * u + 32].cpu().numpy().astype(np.int64).reshape(-1)
        zref = np.array([(int(v) + 2 ** 31) % 2 ** 32 - 2 ** 31 for v in zz], dtype=np.int64)
        refs.append({"call": w, "unit": u, "z_equal": bool((zk == zref).all()),
                     "receipt_equal": rk == [int(x) & 0xFFFFFFFF for x in rr]})
    doc["vs_h1tile_reference"] = refs
    # commitments of a few calls and the round, against commit.py
    key, W = win.key, win.window
    bits = np.arange(t) < m
    wrows = CR.word_rows(CR.words(bits))
    cm = []
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
        if w in cpu_calls:
            o, units, nw = win.plans[0].offs[w]
            fr = {"window": W, "step": s, "linear": w, "module": win.names[w], "base": s * win.u_step + win.offs[w],
                  "units": units, "words": nw, "schema": CM.SCHEMA}
            rd = CM.domain(fr, units, "receipts")
            rrows = CM.receipt_rows(got_of(w).cpu().numpy().view(np.uint32))
            r_cpu = CM.levels(rd, [CM.leaf(rd, key, f"r/{s}/{w}", i, bytes(rrows[i]))[0] for i in range(units)])[-1][0]
            cm.append({"call": w, "a": a_cpu[w] == g["a_roots"][w], "admit": s_cpu[w] == g["admit_roots"][w],
                       "z": z_cpu[w] == g["z_roots"][w], "receipts": r_cpu == unwords(g["r_roots"][8 * w:8 * w + 8])})
    doc["commitments"] = cm
    doc["a_roots_bad"] = [w for w in range(L) if a_cpu[w] != g["a_roots"][w]]
    doc["admit_roots_bad"] = [w for w in range(L) if s_cpu[w] != g["admit_roots"][w]]
    doc["z_roots_bad"] = [w for w in range(L) if z_cpu[w] != g["z_roots"][w]]
    f_step = {"window": W, "step": s, "linears": L}
    a_step = CM.step_root(CM.domain(f_step, L, "a-step"), [a_cpu[w] for w in range(L)])
    admit_step = CM.step_root(CM.domain(f_step, L, "admit-step"), [s_cpu[w] for w in range(L)])
    z_step = CM.step_root(CM.domain(f_step, L, "z-step"), [z_cpu[w] for w in range(L)])
    r_fields = {"window": W, "step": s, "leaves": "a-step root, admit-step root, z-step root"}
    rnd = CM.step_root(CM.domain(r_fields, 3, "round"), [a_step, admit_step, z_step])
    doc["round"] = {"a_step": a_step == g["leaves"][0], "admit_step": admit_step == g["leaves"][1],
                    "z_step": z_step == g["leaves"][2], "root": rnd == g["root"]}
    doc["ok"] = (not z_bad and not zero_bad and (g["y"] is None or not y_bad) and not rc_bad
                 and all(x["z_equal"] and x["receipt_equal"] for x in refs)
                 and all(all(v for k_, v in x.items() if k_ != "call") for x in cm)
                 and not doc["a_roots_bad"] and not doc["admit_roots_bad"] and not doc["z_roots_bad"]
                 and all(doc["round"].values()))
    return doc


def gate_verdict(gate: dict, control: dict) -> dict:
    """The poisoned gate passes and its no-write control, complete, is refused on z and on the receipts."""
    refused = (control.get("complete") and not control.get("ok") and bool(control.get("z_vs_int_mm_bad"))
               and bool(control.get("receipts_vs_h1r_bad")))
    return {"ok": bool(gate.get("ok")) and bool(refused), **F.form_of("dither-fused"), "gate_ok": bool(gate.get("ok")),
            "control_refused": bool(refused),
            "control_bad": {k: len(control.get(k) or []) for k in ("z_vs_int_mm_bad", "y_vs_scaled_mm_bad", "empty_rows_bad",
                                                                    "receipts_vs_h1r_bad", "z_roots_bad")}}
