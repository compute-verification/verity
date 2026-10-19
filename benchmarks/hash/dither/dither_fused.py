"""Dither's fused served linear in compiled vLLM: every int8-exact linear's y from `h1f.cu`'s one GEMM (z = xq W^T, the
H1 tile's exact product, with the int8-exact dequantization in its epilogue), which also computes the previous step's H1 +
fx1 receipts; z committed per step before the step's coin, like root(A).

    import dither_fused as DF
    DF.patch(t=T, h1c="libh1c.so", h1f="libh1f.so")   # before LLM(): Int8PlainLinearMethod.apply -> row_quant + dither::fused_mm
    llm = LLM(model=int8_exact_checkpoint, ...)        # compiled, CUDA graphs: the op is opaque to Inductor, its launch captured
    DF.hook(llm)                                       # the step hook below

Per engine step s (parity p = s mod 2), around the runner's `execute_model`:
    before   the step struct (p, m, receipts on, dump) written to the device, m the step's scheduled tokens; the forward's
             128 `h1f` launches read it: z_s, y_s, A_s's slots; step s - 1's receipts
    after    on the device: step s's A tables (`commit.s8_table`'s format, labels a/<s>/<w>), admit tables (the T / 16 words of
             the slots < m, s/<s>/<w>), z tables (each slot row's N int32 LE, z/<s>/<w>, domain 'z-table'), the a-step,
             admit-step and z-step trees and the round over the three ('round'); step s - 1's receipt trees (r/<s-1>/<w>); the
             round root read back, recorded through `sampled_proofs.challenger.LiveRecord` before its coin opens
             (`pouw.window.live_challenger`), the salt `h1tile.live_salt`, step s's seeds (`h1r_launch` version 0) into the
             buffer step s + 1's launches read

Labelled simplifications (none changes a step's cost): the admitted slots are the step's first m rows (SERVE-H1 admits
from its queue); the window's registration, queue commit and close are not run (once a window, not a step); a window is
the engine's life up to `ROUNDS` steps.  The z table and the third round leaf are this lane's rendering of CA's form (a z
table in AROW's row-table format, a root per step before the coin); the kind, fields and labels are CA's to fix.
"""
from __future__ import annotations

import ctypes as C
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT / "benchmarks" / "pouw" / "served_h1", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import commit as CM  # noqa: E402
import credit as CR  # noqa: E402
import serve as SV  # noqa: E402
from h1c import H1C, Dom, Tab, unwords, words  # noqa: E402
from h1r import H1R, Call  # noqa: E402

SITES = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
MODEL = {"repo": "verity-int8/Llama-3.1-8B-INT8"}
ROUNDS = 1 << 14
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

    def __init__(self, t: int, key: bytes, h1c_path: str, h1f_path: str, gate: bool, dev):
        self.t, self.key, self.dev = t, key, dev
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
        self.s, self.parity, self.rc_on, self.dump, self.m = 0, 0, 0, 0, 0
        self.calls = 0
        self.steps = 0
        self.gate_armed: int | None = None
        self.gate: dict = {}
        self.round_log: list[dict] = []
        self._coins()

    def _coins(self) -> None:
        from verity.core.protocols.pouw import window as Wd
        from verity.core.protocols.pouw.circuit import h1tile as HT
        from verity.core.protocols.sampled_proofs import challenger as Ch
        self.HT = HT
        self.lch = Wd.live_challenger(1, ROUNDS)
        self.rounds = Ch.LiveRecord(self.lch, Wd.ROUND)

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
        for w, (n, k) in enumerate(self.nk):
            fz = {"window": W, "step": 0, "linear": w, "module": self.names[w], "slots": t, "n": n, "schema": CM.SCHEMA}
            tabs.append(Tab(Dom("z-table", fz, t, {"step": (1, 0)}), t, data=self.z[w][p].view(torch.uint8), stride=4 * n,
                            body=4 * n, length=4 * n, lab=("z/", f"/{w}"), out=mid.leaves_addr(2) + 64 * w))
        return {"round": rnd, "mid": mid, "calls": c.set(tabs)}

    def launch(self, w: int, xq: torch.Tensor, s_a: torch.Tensor, wt: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor) -> None:
        n, k = self.nk[w]
        rc = self.f.h1f_launch(self.F.data_ptr() + self.fsize * w, self.S.data_ptr(), self.t, n, k, self.units[w],
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
        self.dump = 0
        if self.gate_armed is not None and m == self.gate_armed:
            self.dump, self.gate_armed = 1, None
            self.gate = {"s": self.s, "m": m}
        self.S.copy_(torch.tensor([self.parity, m, self.rc_on, self.dump], dtype=torch.int32))

    def after(self) -> None:
        torch.cuda.synchronize()
        s, p, m, t = self.s, self.parity, self.m, self.t
        bits = np.arange(t) < m
        self.live[p].copy_(torch.from_numpy((np.arange(t // 32) * 32 < m).astype(np.uint8)))
        self.admit[p].copy_(torch.frombuffer(bytearray(b"".join(CR.word_rows(CR.words(bits)))), dtype=torch.uint8))
        self.stepv.copy_(torch.tensor([s, max(s - 1, 0)], dtype=torch.int32))
        pre = self.pre[p]
        for k in ("calls", "mid", "round"):
            self.c.build(pre[k], self.stepv[0:1])
        if self.rc_on:
            self.c.build(self.rset, self.stepv[1:2])
        if self.gate.get("s") == s - 1 and "receipts" not in self.gate:
            self.gate["receipts"] = self.plans[0].out.clone()
            self.gate["r_roots"] = self.rset.roots.clone()
        root = unwords(pre["round"].root(0))
        opening = self.rounds.record(root)
        salt = self.HT.live_salt(self.lch.commitment, opening.coin, s, root)
        self.salt.copy_(torch.from_numpy(words(salt)))
        self.step_base.fill_(s * self.u_step)
        self.r.launch(self.plans[p], self.salt, version=0, step_base=self.step_base)
        if self.gate.get("s") == s and "a" not in self.gate:
            self.gate.update(p=p, root=root, salt=salt, a=[x[p].clone() for x in self.slot],
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

    def stats(self) -> dict:
        return {"steps": self.steps, "fused_calls_eager": self.calls, "t": self.t, "s": self.s}


# -- the op and the patch --------------------------------------------------------------------------------------------------

def _window(dev) -> Window:
    global WIN
    if WIN is None:
        WIN = Window(CFG["t"], CFG["key"], CFG["h1c"], CFG["h1f"], CFG["gate"], dev)
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


def patch(t: int, h1c: str, h1f: str, key: bytes | None = None, gate: bool = False) -> None:
    """Every int8-exact linear through ``dither::fused_mm`` (call before the engine is built, in its process)."""
    from verity_vllm.engine import int8_ops
    from verity_vllm.engine import int8_vllm as IV
    CFG.update(t=t, h1c=h1c, h1f=h1f, key=key or os.urandom(64), gate=gate)
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

def check(win: Window, lib_h1r: str, n_ref_units: int = 3, cpu_calls: tuple = (0, 1, 2, 3, 126, 127)) -> dict:
    """The gate step's checks, after the step after it ran (its receipts): per call z = torch._int_mm (cuBLASLt) on the
    slots' rows < m and zero above, y (dumped) = int8_ops.scaled_mm on the same rows (today's served y), every receipt
    = h1r (gated against receipts.py) on the saved slots, a few units = h1tile.reference (z and receipt), the A, admit, z
    and receipt roots of CPU_CALLS and the step's round = commit.py's."""
    from verity_vllm.engine import int8_exact as E
    from verity_vllm.engine import int8_ops
    g = win.gate
    if "receipts" not in g or "a" not in g:
        return {"ok": False, "why": f"gate step incomplete: {sorted(g)}"}
    m, s, t, L = g["m"], g["s"], win.t, win.L
    doc: dict = {"s": s, "m": m, "t": t}
    z_bad, y_bad, zero_bad = [], [], []
    for w in range(L):
        a, sa, z = g["a"][w], g["sa"][w], g["z"][w]
        zr = E.int_mm(a[:m], win.wt[w])
        if not torch.equal(zr, z[:m]):
            z_bad.append(w)
        if bool(z[m:].any()) or bool(a[m:].any()) or bool(sa[m:].any()):
            zero_bad.append(w)
        if g["y"] is not None:
            yr = int8_ops.scaled_mm(a[:m], sa[:m], win.wt[w], REG[w][1].weight_scale)
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
    r.launch(plan, salt, version=2)
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
        u = int(rng.integers(n // 32))          # row block 0, column block u
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
        if w in cpu_calls:
            fz = {"window": W, "step": s, "linear": w, "module": win.names[w], "slots": t, "n": n, "schema": CM.SCHEMA}
            zd = CM.domain(fz, t, "z-table")
            zrows = g["z"][w].cpu().numpy().astype("<i4").view(np.uint8)
            z_cpu[w] = CM.levels(zd, [CM.leaf(zd, key, f"z/{s}/{w}", i, bytes(zrows[i]))[0] for i in range(t)])[-1][0]
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
    f_step = {"window": W, "step": s, "linears": L}
    a_step = CM.step_root(CM.domain(f_step, L, "a-step"), [a_cpu[w] for w in range(L)])
    admit_step = CM.step_root(CM.domain(f_step, L, "admit-step"), [s_cpu[w] for w in range(L)])
    z_step = CM.step_root(CM.domain(f_step, L, "z-step"), g["z_roots"])
    r_fields = {"window": W, "step": s, "leaves": "a-step root, admit-step root, z-step root"}
    rnd = CM.step_root(CM.domain(r_fields, 3, "round"), [a_step, admit_step, z_step])
    doc["round"] = {"a_step": a_step == g["leaves"][0], "admit_step": admit_step == g["leaves"][1],
                    "z_step": z_step == g["leaves"][2], "root": rnd == g["root"]}
    doc["ok"] = (not z_bad and not zero_bad and (g["y"] is None or not y_bad) and not rc_bad
                 and all(x["z_equal"] and x["receipt_equal"] for x in refs)
                 and all(all(v for k_, v in x.items() if k_ != "call") for x in cm)
                 and not doc["a_roots_bad"] and not doc["admit_roots_bad"] and all(doc["round"].values()))
    return doc
