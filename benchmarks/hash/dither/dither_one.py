"""Dither's served linear in one stream in compiled vLLM (`h1o.cu`, CA's H1V3): every int8-exact linear's y = dequant(O),
O = O' - sum_d 128^d D_d from the noised mainloop whose words the call's H1 + fx1 receipts fold, and z = O.  The seeds are
the call's own, from the window's salt and root(A_j), root(B_j) (the 17 Oct seeded regime: epsilon is 'seeded'), so each
call commits its A table on the device before its units and nothing waits on a later round.

    import dither_one as DO
    DO.patch(t=T, h1c="libh1c.so", h1o="libh1o.so")   # before LLM(): Int8PlainLinearMethod.apply -> row_quant + dither1::one_mm
    llm = LLM(model=int8_exact_checkpoint, ...)       # compiled, CUDA graphs: the op is opaque to Inductor, its launches captured
    DO.hook(llm)

The window: its B tables at install, then one live round whose coin gives the epoch salt (`h1tile.live_salt` of the
round over the B roots).  A linear call (`dither1::one_mm`), on the forward's stream: its A slot (`h1o_slot`), its A table
(`h1c`, the root where its seeds read it), its seeds (`h1r_seeds`), its units (`h1o_units`: receipts, z, y).  After step s's
forward: its admit and z tables, the a-step, admit-step and z-step trees, the round over them, and the receipt trees,
then the round root recorded (`sampled_proofs.challenger.LiveRecord`; its coin is not read: no seed waits on it).

The served value is the dithered product, so y is not today's served y (`gate` measures how far) and the tokens are not
the exact arm's.  It is that product only where no s8 entry of X = A + b or W' = B + c + U wraps; today's A and B (symmetric
at amax / 127) do wrap, and the gate counts the cells where the peel and (A + b)(B + c)^T differ.

Labelled simplifications (none changes a step's cost): the admitted slots are the step's first m rows; the window is the
engine's life; the epoch round's root (a b-step tree over the B roots) and the z table (AROW's row-table format) are this
lane's rendering of CA's forms.
"""
from __future__ import annotations

import ctypes as C
import os
import time

import numpy as np
import torch

import dither_fused as DF
from dither_fused import CM, CR, SV, H1C, Dom, Tab, unwords, words, H1R, Call
import h1c as H1CM

F_DUMP, F_NW_TILES, F_NW_UNITS, F_FAULT = 1, 2, 4, 8     # h1o.cu's H1OStep.flags
CAPTURES = ("gate", "control", "fault")
LIB_ENV = "DITHER_H1O"
LIB = "dither1"
REG = DF.REG
CFG: dict = {}
WIN: "Window | None" = None
M32 = (1 << 32) - 1


class _O(C.Structure):
    _fields_ = [("slot", C.c_void_p * 2), ("scale", C.c_void_p * 2), ("z", C.c_void_p * 2), ("rc", C.c_void_p * 2),
                ("out", C.c_void_p * 2), ("seeds", C.c_void_p), ("ydump", C.c_void_p), ("t", C.c_int), ("n", C.c_int),
                ("k", C.c_int), ("units", C.c_int)]


class Window:
    """A window's device state: per linear one A, scale and z slot (a step's tables are built before the next step's
    forward reads its slots); the h1c sets (B tables once; the A, admit and z tables, the step trees and the round); one
    call table for the seeds and receipts; the `h1o` table and step struct; the epoch salt."""

    def __init__(self, t: int, key: bytes, h1c_path: str, h1o_path: str, gate: bool, dev):
        self.t, self.key, self.dev = t, key, dev
        self.L = L = len(REG)
        if sorted(REG) != list(range(L)):
            raise RuntimeError(f"linears registered: {sorted(REG)}")
        self.names = [REG[w][0] for w in range(L)]
        self.wt = [REG[w][1].weight for w in range(L)]
        self.sw = [REG[w][1].weight_scale.reshape(-1) for w in range(L)]
        self.nk = [(int(x.shape[0]), int(x.shape[1])) for x in self.wt]
        self.slot = [torch.zeros(t, k, dtype=torch.int8, device=dev) for _, k in self.nk]
        self.scale = [torch.zeros(t, dtype=torch.float32, device=dev) for _ in self.nk]
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
        self.o.h1o_slot_launch.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p]
        self.o.h1o_units_launch.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p,
                                            C.c_int, C.c_void_p, C.c_void_p, C.c_void_p]
        if self.o.h1o_struct_bytes(0) != C.sizeof(_O) or self.o.h1o_struct_bytes(1) != 16:
            raise RuntimeError("h1o.cu's structs differ from dither_one's")
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
        self.ks1 = [(C.c_int * 1)(k) for _, k in self.nk]
        self.salt = torch.zeros(8, dtype=torch.int64, device=dev)
        self.step_base = torch.zeros(1, dtype=torch.int64, device=dev)
        self.stepv = torch.zeros(2, dtype=torch.int32, device=dev)
        self.s, self.flags, self.m = 0, 0, 0
        self.calls = self.steps = 0
        self.arms: list[str] = []
        self.arm_m: int | None = None
        self.caps: dict[str, dict] = {}
        self.cap: str | None = None
        self.round_log: list[dict] = []
        self.phases: list[dict] = []
        self.t_before = 0.0
        self.ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        self._epoch()

    def _epoch(self) -> None:
        from verity.core.protocols.pouw import window as Wd
        from verity.core.protocols.pouw.circuit import h1tile as HT
        from verity.core.protocols.sampled_proofs import challenger as Ch
        self.HT = HT
        self.lch = Wd.live_challenger(1, DF.ROUNDS)
        self.rounds = Ch.LiveRecord(self.lch, Wd.ROUND)
        broots = [unwords(self.bset.root(w)) for w in range(self.L)]
        self.epoch_root = CM.step_root(CM.domain({"window": self.window, "linears": self.L}, self.L, "b-step"), broots)
        opening = self.rounds.record(self.epoch_root)
        self.salt_bytes = HT.live_salt(self.lch.commitment, opening.coin, 0, self.epoch_root)
        self.salt.copy_(torch.from_numpy(words(self.salt_bytes)))

    def arm(self, m: int, names: tuple = CAPTURES) -> None:
        """Capture `names` at the next steps of m tokens, one a step: each step's A, scale and z slots, y dump, receipts,
        seeds and step roots poisoned with 0xA5 before its forward; "control" with the no-write flags (its kernels write
        none of them, so `check` must refuse it); "fault" with F_FAULT (`check` wants that one word to reach z, y and the
        receipt)."""
        self.arms, self.arm_m = list(names), m
        self.caps = {n: {} for n in names}

    def _pre_sets(self) -> dict:
        c, W, t, L = self.c, self.window, self.t, self.L
        f_step = {"window": W, "step": 0, "linears": L}
        rnd = c.set([Tab(Dom("round", {"window": W, "step": 0, "leaves": "a-step root, admit-step root, z-step root"}, 3,
                             {"step": (1, 0)}), 3)])
        mid = c.set([Tab(Dom("a-step", f_step, L, {"step": (1, 0)}), L, out=rnd.leaves_addr(0)),
                     Tab(Dom("admit-step", f_step, L, {"step": (1, 0)}), L, out=rnd.leaves_addr(0) + 64),
                     Tab(Dom("z-step", f_step, L, {"step": (1, 0)}), L, out=rnd.leaves_addr(0) + 128)])
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

    def launch(self, w: int, xq: torch.Tensor, s_a: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor) -> None:
        n, k = self.nk[w]
        st = torch.cuda.current_stream().cuda_stream
        fw = self.F.data_ptr() + self.fsize * w
        a = self.pre["a"]
        rc = self.o.h1o_slot_launch(fw, self.S.data_ptr(), self.t, xq.data_ptr(), s_a.data_ptr(), st)
        rc = rc or self.c.lib.h1c_build(C.addressof(a.host) + self.tab_bytes * w, a.dev.data_ptr() + self.tab_bytes * w, 1,
                                        self.stepv.data_ptr(), self.c.btab.data_ptr(), 1, st)
        rc = rc or self.r.lib.h1r_launch(self.plan.table.data_ptr() + self.call_bytes * w, 1, self.units[w], self.ks1[w],
                                         self.salt.data_ptr(), self.step_base.data_ptr(), self.plan.seeds.data_ptr(),
                                         self.plan.out.data_ptr(), 0, 0, st)
        rc = rc or self.o.h1o_units_launch(fw, self.S.data_ptr(), self.units[w], n, k, xq.data_ptr(), s_a.data_ptr(),
                                           int(xq.shape[0]), s_w.data_ptr(), y.data_ptr(), st)
        if rc:
            raise RuntimeError(f"dither1 call {w} (M={xq.shape[0]}): a launch returned {rc}")
        self.calls += 1

    # -- the step hook ---------------------------------------------------------------------------------------------------

    def _poison(self, ts) -> None:
        for x in ts:
            x.view(torch.uint8).fill_(DF.POISON)

    def before(self, m: int) -> None:
        if m > self.t:
            raise RuntimeError(f"a step of {m} tokens in a window of T = {self.t} slots")
        self.m, self.flags, self.cap = m, 0, None
        if self.arms and m == self.arm_m:
            self.cap = name = self.arms.pop(0)
            self.flags = F_DUMP | ((F_NW_TILES | F_NW_UNITS) if name == "control" else 0) | (F_FAULT if name == "fault" else 0)
            self._poison(self.slot + self.scale + self.z + list(self.ydump or [])
                         + [self.plan.out, self.plan.seeds, self.pre["mid"].level])
            self.caps[name] = {"s": self.s, "m": m}
        t, s = self.t, self.s
        bits = np.arange(t) < m
        self.live.copy_(torch.from_numpy((np.arange(t // 32) * 32 < m).astype(np.uint8)))
        self.admit.copy_(torch.frombuffer(bytearray(b"".join(CR.word_rows(CR.words(bits)))), dtype=torch.uint8))
        self.stepv.copy_(torch.tensor([s, s], dtype=torch.int32))
        self.step_base.fill_(s * self.u_step)
        self.S.copy_(torch.tensor([0, m, 0, self.flags], dtype=torch.int32))
        if self.cap:
            torch.cuda.synchronize()
        self.t_before = time.perf_counter()

    def after(self) -> None:
        torch.cuda.synchronize()
        t_fwd = time.perf_counter()
        s, m, pre, ev = self.s, self.m, self.pre, self.ev
        ev[0].record()
        self.c.build(pre["admit"], self.stepv[0:1])
        ev[1].record()
        self.c.build(pre["z"], self.stepv[0:1])
        ev[2].record()
        self.c.build(pre["mid"], self.stepv[0:1])
        self.c.build(pre["round"], self.stepv[0:1])
        ev[3].record()
        self.c.build(self.rset, self.stepv[0:1])
        ev[4].record()
        root = unwords(pre["round"].root(0))
        t_root = time.perf_counter()
        self.rounds.record(root)
        t_end = time.perf_counter()
        self.phases.append({"m": m, "forward_ms": 1e3 * (t_fwd - self.t_before), "admit_tables_ms": ev[0].elapsed_time(ev[1]),
                            "z_tables_ms": ev[1].elapsed_time(ev[2]), "step_trees_ms": ev[2].elapsed_time(ev[3]),
                            "receipt_trees_ms": ev[3].elapsed_time(ev[4]), "record_host_ms": 1e3 * (t_end - t_root),
                            "between_ms": 1e3 * (t_end - t_fwd)})
        if self.cap is not None:
            self.caps[self.cap].update(
                root=root, a=[x.clone() for x in self.slot], sa=[x.clone() for x in self.scale], z=[x.clone() for x in self.z],
                y=[x[:m].clone() for x in self.ydump] if self.ydump else None, receipts=self.plan.out.clone(),
                r_roots=self.rset.roots.clone(), a_roots=[unwords(pre["mid"].leaf(0, w)) for w in range(self.L)],
                admit_roots=[unwords(pre["mid"].leaf(1, w)) for w in range(self.L)],
                z_roots=[unwords(pre["mid"].leaf(2, w)) for w in range(self.L)],
                leaves=[unwords(pre["round"].leaf(0, i)) for i in range(3)],
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
        med = {str(m): {"steps": len(xs), **{k: round(float(np.median([x[k] for x in xs if k in x])), 4)
                                             for k in dict.fromkeys(k for x in xs for k in x) if k != "m"}}
               for m, xs in sorted(by.items(), key=lambda kv: -len(kv[1]))[:4]}
        return {"steps": self.steps, "one_calls_eager": self.calls, "t": self.t, "s": self.s, "epsilon": "seeded",
                "phases_by_m": med}


# -- the op and the patch --------------------------------------------------------------------------------------------------

def _window(dev) -> Window:
    global WIN
    if WIN is None:
        WIN = Window(CFG["t"], CFG["key"], CFG["h1c"], CFG["h1o"], CFG["gate"], dev)
    return WIN


@torch.library.custom_op(f"{LIB}::one_mm", mutates_args=())
def one_mm(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor, idx: int) -> torch.Tensor:
    y = torch.empty((xq.shape[0], w.shape[0]), dtype=torch.bfloat16, device=xq.device)
    if xq.shape[0]:
        _window(xq.device).launch(idx, xq.contiguous(), s_a.contiguous(), s_w.reshape(-1), y)
    return y


@one_mm.register_fake
def _one_mm_fake(xq, s_a, w, s_w, idx):
    return xq.new_empty((xq.shape[0], w.shape[0]), dtype=torch.bfloat16)


def patch(t: int, h1c: str, h1o: str, key: bytes | None = None, gate: bool = False) -> None:
    """Every int8-exact linear through ``dither1::one_mm`` (call before the engine is built, in its process)."""
    from verity_vllm.engine import int8_ops
    from verity_vllm.engine import int8_vllm as IV
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
        return torch.ops.dither1.one_mm(xq, s_a, layer.weight, layer.weight_scale, self.dither_idx).reshape(*lead, n)

    IV.Int8PlainLinearMethod.process_weights_after_loading = process_weights_after_loading
    IV.Int8PlainLinearMethod.apply = apply


def hook(llm) -> Window:
    from verity_vllm.engine import engine_profile as prof
    runner = prof.get_model_runner(llm)
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

def h1v3_unit(a32: np.ndarray, b32: np.ndarray, salt: bytes, position: int, roots: tuple[bytes, bytes]) -> dict:
    """CA's definition on the CPU for one 32 x 32 unit (`h1tile.noise`, `segment.call_words` and the CP-W fold, as
    `h1tile.reference`): z = O' - sum_d 128^d D_d per cell (s32), the receipt, and the dithered product (A + b)(B + c)^T in
    integers, whose cells differ from z exactly where an s8 entry of X or W' wraps."""
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
            o = (sets[ks][q] - sum((128 ** d) * sets[ks + d][q] for d in range(1, m))) & M32
            z[R[SG.row_slot(q)], Cs[SG.col_slot(q)]] = o - (1 << 32) if o >> 31 else o
    bits_a = np.array([rows[i][0] for i in range(32)], dtype=np.int64)
    bits_b = np.array([cols[j][0] for j in range(32)], dtype=np.int64)
    zd = (a32.astype(np.int64) + bits_a) @ (b32.astype(np.int64) + bits_b).T
    return {"z": z, "receipt": [w & M32 for s in cp for w in s], "dithered": zd}


def check(win: Window, lib_h1o: str, which: str = "gate", n_cpu_units: int = 3,
          cpu_calls: tuple = (0, 1, 2, 3, 126, 127)) -> dict:
    """`which`'s capture against references other than the served kernel: every call's A, admit and z roots, the receipt
    roots of CPU_CALLS and the step's round = commit.py's; every receipt = h1r v1's (another unit implementation, gated
    against receipts.py) on the saved slots under the window's salt and the CPU's roots; every z = h1o_ref's (noise from
    full compressions, T_last and Y by scalar dp4a, z = T_last - U Y); y (dumped) = dequant of the kernel's own z on the rows
    < m; a few units = CA's definition on the CPU (`h1v3_unit`: z and receipt).  Every buffer read was 0xA5 before the
    kernels wrote it.  "fault" wants z off h1o_ref's by 2^FAULT_SHIFT at unit 0's cell (0, 0) of every call and nowhere
    else, that unit's receipt off h1r's and no other, and y = dequant of that z (the perturbed MMA word reached what serves)."""
    from verity_vllm.engine import int8_exact as E
    from verity_vllm.engine import int8_ops
    g = win.caps.get(which, {})
    if "receipts" not in g or "a" not in g:
        return {"ok": False, "complete": False, "why": f"{which} step incomplete: {sorted(g)}"}
    m, s, t, L = g["m"], g["s"], win.t, win.L
    doc: dict = {"which": which, "complete": True, "s": s, "m": m, "t": t, "poison": f"{DF.POISON:#04x}", "epsilon": "seeded"}
    key, W = win.key, win.window
    # commitments against commit.py
    bits = np.arange(t) < m
    wrows = CR.word_rows(CR.words(bits))
    a_cpu, s_cpu, z_cpu, cm = {}, {}, {}, []
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
    o.h1o_ref_launch.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p]
    shift = o.h1o_fault_shift()
    live = torch.from_numpy((np.arange(t // 32) * 32 < m).astype(np.uint8)).to(win.dev)
    roots = [torch.from_numpy(np.concatenate([np.frombuffer(a_cpu[w], ">u8"), np.frombuffer(g["b_roots"][w], ">u8")])
                              .astype("<u8").view("<i8").copy()).to(win.dev) for w in range(L)]
    plan = r.plan([Call(g["a"][w], win.wt[w], s * win.u_step + win.offs[w], roots[w], live) for w in range(L)])
    r.launch(plan, win.salt, version=1)
    cb = r.lib.h1r_call_bytes()
    zref = [torch.empty(t, n, dtype=torch.int32, device=win.dev) for n, _ in win.nk]
    for w in range(L):
        rc = o.h1o_ref_launch(plan.table.data_ptr() + cb * w, win.units[w], plan.seeds.data_ptr(), zref[w].data_ptr(),
                              win.nk[w][0], torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"h1o_ref_launch({w}) returned {rc}")
    torch.cuda.synchronize()
    got = g["receipts"]

    def got_of(w: int) -> torch.Tensor:
        o_, units, nw = win.plan.offs[w]
        return got[o_:o_ + units * nw].view(units, nw)

    delta = torch.zeros(1, dtype=torch.int64, device=win.dev) + (1 << shift)
    rc_bad, z_bad, y_bad, fault_bad = [], [], [], []
    for w in range(L):
        n, k = win.nk[w]
        rdiff = (plan.receipts(w) != got_of(w)).any(dim=1).nonzero().flatten().tolist()
        dz = (g["z"][w].to(torch.int64) - zref[w].to(torch.int64)) & M32
        cells = dz.nonzero().tolist()
        if g["y"] is not None:
            yk = g["y"][w]
            if not torch.equal(E.dequantize(g["z"][w][:m], g["sa"][w][:m], win.sw[w]).view(torch.int16), yk.view(torch.int16)):
                y_bad.append(w)
        if which == "fault":
            moved = g["y"] is not None and float(g["sa"][w][0]) != 0.0 and not torch.equal(
                E.dequantize(zref[w][:1, :1], g["sa"][w][:1], win.sw[w][:1]).view(torch.int16), g["y"][w][:1, :1].view(torch.int16))
            if not (rdiff == [0] and cells == [[0, 0]] and int(dz[0, 0]) == int(delta) and moved):
                fault_bad.append({"call": w, "receipt_units_off": rdiff[:4], "z_cells_off": cells[:4],
                                  "dz00": int(dz[0, 0]), "y_moved": bool(moved)})
        else:
            if rdiff:
                rc_bad.append(w)
            if cells:
                z_bad.append(w)
    doc.update(receipts_vs_h1r_bad=rc_bad, z_vs_ref_bad=z_bad, y_vs_dequant_z_bad=y_bad if g["y"] is not None else None)
    doc["live_units"] = int(sum((min(m, t) + 31) // 32 * (n // 32) for n, _ in win.nk))
    if which == "fault":
        doc["fault_shift"] = shift
        doc["fault_bad"] = fault_bad
        doc["conforming"] = (not fault_bad and not y_bad and not doc["a_roots_bad"] and all(doc["round"].values())
                             and not doc["z_roots_bad"])
        doc["ok"] = doc["conforming"]
        return doc
    # the served y against today's (informational: the dithered product is served)
    if g["y"] is not None and which == "gate":
        eq, rel = 0, []
        for w in range(L):
            yt = int8_ops.scaled_mm(g["a"][w][:m], g["sa"][w][:m], win.wt[w], REG[w][1].weight_scale).float()
            yk = g["y"][w].float()
            eq += int(torch.equal(yt, yk))
            rel.append(float((yk - yt).norm() / yt.norm().clamp_min(1e-30)))
        doc["y_vs_today"] = {"calls_equal": eq, "rel_l2_median": float(np.median(rel)), "rel_l2_max": float(np.max(rel))}
    # CA's definition on the CPU, a few units of row block 0
    units_doc = []
    if which == "gate":
        rng = np.random.default_rng(s)
        for w in [0, 3, 2][:n_cpu_units]:
            n, k = win.nk[w]
            u = int(rng.integers(n // 32))
            a32 = g["a"][w][0:32].cpu().numpy()
            b32 = win.wt[w][32 * u:32 * u + 32].cpu().numpy()
            ref = h1v3_unit(a32, b32, g["salt"] if "salt" in g else win.salt_bytes, s * win.u_step + win.offs[w] + u,
                            (a_cpu[w], g["b_roots"][w]))
            zk = g["z"][w][0:32, 32 * u:32 * u + 32].cpu().numpy().astype(np.int64)
            rk = (got_of(w)[u].cpu().numpy().astype(np.int64) & M32).tolist()
            units_doc.append({"call": w, "unit": u, "z_equal": bool((zk == ref["z"]).all()),
                              "receipt_equal": rk == ref["receipt"],
                              "cells_peel_ne_dithered": int((ref["z"] != ref["dithered"]).sum()),
                              "a_eq_127": int((a32 == 127).sum()), "b_abs_gt_62": int((np.abs(b32.astype(np.int64)) > 62).sum())})
    doc["vs_h1v3_cpu"] = units_doc
    doc["ok"] = (not rc_bad and not z_bad and (g["y"] is None or not y_bad)
                 and all(x["z_equal"] and x["receipt_equal"] for x in units_doc)
                 and not doc["a_roots_bad"] and not doc["admit_roots_bad"] and not doc["z_roots_bad"]
                 and all(doc["round"].values()))
    return doc


def gate_verdict(gate: dict, control: dict, fault: dict) -> dict:
    """The poisoned gate passes, its no-write control (complete) is refused on z and on the receipts, and the fault step
    conforms (the one perturbed noised word moved that cell's z and y and that unit's receipt, and nothing else)."""
    refused = (control.get("complete") and not control.get("ok") and bool(control.get("z_vs_ref_bad"))
               and bool(control.get("receipts_vs_h1r_bad")))
    return {"ok": bool(gate.get("ok")) and bool(refused) and bool(fault.get("conforming")), "gate_ok": bool(gate.get("ok")),
            "control_refused": bool(refused), "fault_conforming": bool(fault.get("conforming")),
            "control_bad": {k: len(control.get(k) or []) for k in ("z_vs_ref_bad", "y_vs_dequant_z_bad", "receipts_vs_h1r_bad",
                                                                    "z_roots_bad")}}
