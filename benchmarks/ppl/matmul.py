"""The matmul a step sees (`MM`), the chain that runs an arm's steps on it, and the gated exact integer product the chain
ends in.

Three kinds of matmul, all written y = x w^T but for P V: `linear` (x the layer's input rows, w the weight, n x k),
`qk` (x = Q, the query rows of one KV group's heads, w = K) and `pv` (x = P, w = V, y = P V)."""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Callable

import torch

import gates as G
import methodology

FP32_EXACT = 2 ** 24
FP64_EXACT = 2 ** 53
#: rows of each product recomputed in float64 on the check window
CHECK_ROWS = {"linear": 32, "qk": 64, "pv": 64}


def int_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a b^T for integer-valued a (m x k, |a| <= 127) and int8 b (n x k): exact, as int64 (torch._int_mm, s8 x s8 ->
    s32; rows padded to 32)."""
    m = a.shape[0]
    pad = max(0, 32 - m)
    ai = a.to(torch.int8)
    if pad:
        ai = torch.nn.functional.pad(ai, (0, 0, 0, pad))
    return torch._int_mm(ai, b.T)[:m].long()


def exact_mm(a: torch.Tensor, b: torch.Tensor, term: int) -> torch.Tensor:
    """a @ b for integer-valued FP32 a, b with |a_ik b_kj| <= term: FP32 GEMMs over k-chunks whose every partial sum is
    <= 2^24 (so exact in any order), summed in float64. When the chunks would be shorter than 1024, b is split first
    into b = 16 hi + lo (lo in [-8, 7]) and each part's product is taken the same way."""
    c = max(1, FP32_EXACT // max(1, term))
    k = a.shape[-1]
    if k <= c:
        return (a @ b).double()
    am, bm = (float(t.abs().max()) for t in (a, b))
    if c < 1024 and am < 2048 and bm > 15:
        hi = torch.floor((b + 8) / 16)
        lo = b - 16 * hi
        return 16 * exact_mm(a, hi, int(am * float(hi.abs().max()))) + exact_mm(a, lo, int(am * 8))
    out = None
    for i in range(0, k, c):
        part = (a[..., i:i + c] @ b[..., i:i + c, :]).double()
        out = part if out is None else out + part
    return out


@dataclass
class Softmaxed:
    """What a qk chain returns when its quantizer computes the softmax itself (an integer softmax): `p`, the
    probabilities, takes the float softmax's place as P V's x, and `extra` joins P V's `mm.extra`."""
    p: torch.Tensor
    extra: dict = field(default_factory=dict)


class Run:
    """One arm at one seed through one window of one layer: the services its steps use (caches, coins, calibration,
    gates). `cache` is the engine's per-layer cache, shared by every arm, seed and window; `rotations` is run-wide."""

    def __init__(self, arm, seed: int, window: int, layer, gates: G.Gates, cache: dict, rotations: dict,
                 calib: dict | None, hessian: Callable | None = None):
        self.arm, self.seed, self.window, self.layer, self.gates = arm, seed, window, layer, gates
        self.cache, self.rotations, self.calib, self.hessian = cache, rotations, calib or {}, hessian
        self.checking = window == methodology.CHECK_WINDOW and hessian is None
        self._gen = self._ugen = None
        self.peel_u: dict = {}
        self.attn_err = [0.0, 0.0]

    @property
    def li(self) -> int:
        return self.layer.li

    def attention_noise_seed(self) -> int:
        return (self.seed + 1) * 1_000_003 + self.li + self.window * 7_919_003

    def gen(self, device) -> torch.Generator:
        """The attention's coins for this (seed, layer, window), taken in the order the steps ask (K's dither bits,
        then V's, per KV group)."""
        if self._gen is None:
            self._gen = torch.Generator(device=device).manual_seed(self.attention_noise_seed())
        return self._gen

    def ugen(self, device) -> torch.Generator:
        """The peels' own stream, so a peel never moves the served coins."""
        if self._ugen is None:
            self._ugen = torch.Generator(device=device).manual_seed(self.attention_noise_seed() + 104_729 * 10 ** 9)
        return self._ugen

    def steps_for(self, kind: str, name: str) -> list:
        steps = self.arm.linear if kind == "linear" else self.arm.attention
        return [s for s in steps if s.applies(self.li, kind, name)]


class MM:
    """One matmul inside an arm's chain. `x` is the left operand; the right one is `weight()`, the matmul's own w with
    every step's weight map applied (lazily: a step that caches what it derives from the weight never recomputes it).
    After the quantizer (`integer`), both operands are integer-valued and the chain returns the exact integer product."""

    def __init__(self, kind: str, name: str, run: Run, x: torch.Tensor, w: torch.Tensor, maps: tuple = (),
                 cache: dict | None = None, extra: dict | None = None, stage: str = "real", prefix: tuple = (),
                 draws: list | None = None):
        self.kind, self.name, self.run, self.x, self._w, self.maps = kind, name, run, x, w, maps
        self.store = run.cache if cache is None else cache
        self.extra, self.stage, self.prefix = extra or {}, stage, prefix
        self._draws = draws if draws is not None else [0]

    # ------------------------------------------------------------ what a step reads

    @property
    def layer(self) -> int:
        return self.run.li

    @property
    def window(self) -> int:
        return self.run.window

    @property
    def seed(self) -> int:
        return self.run.seed

    @property
    def device(self):
        return self.x.device

    @property
    def checking(self) -> bool:
        """This is the check window: steps run their costly identity checks here, on every layer."""
        return self.run.checking

    def weight(self) -> torch.Tensor:
        w = self._w
        for f in self.maps:
            w = f(w)
        return w

    # ------------------------------------------------------------ what a step does

    def replace(self, x: torch.Tensor | None = None, w: Callable | torch.Tensor | None = None, **extra) -> "MM":
        """A copy with a new x, and w either mapped (a callable, applied after the current maps) or replaced."""
        maps, base = self.maps, self._w
        if callable(w):
            maps = maps + (w,)
        elif w is not None:
            maps, base = (), w
        return MM(self.kind, self.name, self.run, self.x if x is None else x, base, maps, self.store,
                  {**self.extra, **extra}, self.stage, self.prefix, self._draws)

    def integer(self, x: torch.Tensor, w: torch.Tensor, **extra) -> "MM":
        """The integer stage: x and w on the grid; what the rest of the chain returns is their exact product."""
        m = self.replace(x=x, w=w, **extra)
        m.stage = "int"
        return m

    def cached(self, name: str, fn: Callable, coinless: bool = False):
        """fn(), once per matmul (linear name, or the attention KV group), chain prefix and, if anything before it
        is seeded, seed: the layer's linears share it across windows and arms; attention's across its row chunks.
        `coinless`: fn reads none of this step's coins, so only the steps before it decide whether it is per seed."""
        seeded = any(s.defn.seeded(s.kw) for s in (self.prefix[:-1] if coinless else self.prefix))
        key = (self.name, tuple(s.key for s in self.prefix), name, self.seed if seeded else None)
        if key not in self.store:
            self.store[key] = fn()
        return self.store[key]

    def shared(self, key, fn: Callable):
        """fn(), once per run (the rotations)."""
        if key not in self.run.rotations:
            self.run.rotations[key] = fn()
        return self.run.rotations[key]

    def coins(self, shape, levels: int = 2) -> torch.Tensor:
        """Fresh coins of the given shape, uniform on {0, ..., levels - 1} (bits by default; a b-bit dither takes
        levels = 2^b). Linears: one stream per (seed, layer, linear, window); attention: the (seed, layer, window)
        stream, in draw order."""
        if self.kind == "linear":
            n = self._draws[0]
            self._draws[0] += 1
            tag = f"{self.seed}/{self.layer}/{self.name}/{self.window}" + (f"/{n}" if n else "")
            g = torch.Generator(device=self.device)
            g.manual_seed(zlib.crc32(tag.encode()))
            return torch.randint(0, levels, tuple(shape), generator=g, device=self.device).float()
        return torch.randint(0, levels, tuple(shape), generator=self.run.gen(self.device), device=self.device).float()

    def gate(self, name: str, ok: bool, detail=None) -> bool:
        return self.run.gates.record(name, ok, self.where, detail)

    @property
    def where(self) -> str:
        g = f" group {self.extra['group']}" if "group" in self.extra else ""
        return f"{self.run.arm.name}/s{self.seed} layer {self.layer} {self.name}{g} window {self.window}"

    def int_product(self, x: torch.Tensor | None = None, w: torch.Tensor | None = None) -> torch.Tensor:
        """The exact integer product of x and w (default: this matmul's), gated: both within int8 (or, when the
        quantizer declared a wider grid with `integer(..., top=T)`, within +-T: `grid_range`) and the accumulator
        within s32 (a wider grid's FP32-chunk route: within 2^53) by k max|x| max|w| on every call, and on the check
        window against float64 on CHECK_ROWS rows."""
        x = self.x if x is None else x
        w = self.weight() if w is None else w
        top = self.extra.get("top", 127)
        if top > 127:
            self.gate("grid_range", G.within(top, x, w), top)
        else:
            self.gate("int8_range", G.int8_ok(x, w))
        k = x.shape[-1]
        bound = k * G.absmax(x) * G.absmax(w)
        self.gate("accumulator", bound <= (G.S32 if top <= 127 else FP64_EXACT), bound)
        if self.kind == "linear" and top > 127:
            acc = exact_mm(x.float(), w.float().T, int(G.absmax(x) * G.absmax(w)))
        elif self.kind == "linear":
            acc = int_mm(x, w)
        else:
            acc = exact_mm(x, w.transpose(-1, -2) if self.kind == "qk" else w, int(G.absmax(x) * G.absmax(w)))
        if self.checking:
            m = x.shape[-2]
            idx = torch.linspace(0, m - 1, min(CHECK_ROWS[self.kind], m), device=x.device).long()
            xs = x.index_select(-2, idx).double()
            wd = w.double()
            ref = xs @ (wd.transpose(-1, -2) if self.kind != "pv" else wd)
            self.gate("product_exact", bool(torch.equal(acc.index_select(-2, idx).double(), ref)))
        return acc

    def enter(self, step) -> "MM":
        m = self.replace()
        m.prefix = self.prefix + (step,)
        return m


def terminal(mm: MM) -> torch.Tensor:
    if mm.stage == "int":
        return mm.int_product()
    w = mm.weight()
    return mm.x @ (w if mm.kind == "pv" else w.T if w.dim() == 2 else w.transpose(-1, -2))


def run_chain(steps: list, mm: MM) -> torch.Tensor:
    """The steps (already filtered to this matmul) from the first: each gets (mm, nxt) and its parameters."""
    def call(i: int, m: MM):
        if i == len(steps):
            return terminal(m)
        st = steps[i]
        phase = st.defn.phase
        if phase == "quantize" and m.stage != "real":
            raise ValueError(f"{m.where}: a second quantizer ({st.defn.name})")
        if phase == "integer" and m.stage != "int":
            raise ValueError(f"{m.where}: {st.defn.name} acts on integers, before any quantizer")
        if phase == "real" and m.stage != "real":
            raise ValueError(f"{m.where}: {st.defn.name} acts on reals, after the quantizer")
        return st.defn.fn(m.enter(st), lambda m2: call(i + 1, m2), **st.kw)
    return call(0, mm)
