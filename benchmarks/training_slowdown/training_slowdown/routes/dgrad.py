"""Route `dgrad`: the step's gradient reduced over its blocks of rows through a tree of coded messages, and the
trained weights kept between steps in 8-bit codes. Every weight of the start is frozen but the matrices `targets`
names, each trained whole (`rank` 0) or through a low-rank adapter beside it (W acts as W + B·A on its input, as two
thin products, A·x then B·(A·x)), and with `gains` the norm gains of the blocks those matrices are in and above.

- `targets`: comma-separated patterns over the start's parameter names (fnmatch), naming matrices;
- `rank`: 0 trains each matrix whole; r ≥ 1, an adapter of rank r: A is r × in, N(0, 1/in) at the first step that
  holds it, and B out × r, zero, so the model is θ until B moves;
- `grow`: later phases of the adapters, `FRAC:PATTERNS:RANK` joined by `;`: from step ⌊FRAC·n⌋ + 1 on, each matrix
  PATTERNS names (comma-separated) holds an adapter of rank RANK, raised from what it held (its new rows of A
  random as at the start, of B zero, and the optimizer's state for them zero);
- `micro_batches`: the step's blocks of rows, each a forward and a backward, the leaves of a heap: block i (counted
  from the last, 0 the root) sums its own gradient with the messages of blocks 2i + 1 and 2i + 2 and passes the sum
  on, coded at `msg_bits` (32: fp32, uncoded; 1 to 8 a stochastic rounding of `group` entries to an f16 scale, in a
  basis rotated a group with `rotate`), or with `vote` as its sign (a majority vote: each block's own gradient enters
  as its sign too, a zero a random sign). The root codes its own gradient too, and its sum, divided by the blocks, is the step's gradient;
- `opt`: `sign` (lr · sign), `adamw` (b1, b2; its bias correction counts the step) or `muon` (Newton–Schulz of the
  gradient plus b1 times the momentum); `mom_bits` (sign, muon): the momentum m ← b1 m + g kept between steps in
  that many bits a stochastic rounding (`rotate` as the messages), 32 in fp32, 0 none; `clip`: the step's gradient
  (and the gains') scaled to that norm at most, 0 none; `lr`, `warmup`, `decay`: `subset`'s schedule;
- `store_bits`, `group`: the trained tensors, each adapter factor (A, and B transposed) or each matrix's change from
  θ, are kept in `store_bits`-bit codes of `_codes.grid`, an f16 scale a `group` of entries that stays until a value
  outgrows it (`_codes.sticky_add`); a step adds −its step to them and reads the weights back from them before its
  first forward;
- `gains`, `gain_lr`: AdamW (betas 0.9, 0.95, no decay) over the gains at `gain_lr` times the schedule, their
  gradient summed over the blocks in fp32 and divided by them;
- `e_bits`: an adapter's E, each rank component rounded to nearest at that many bits with an f16 scale.

The random values, all from torch's global generator, in this order: at `build`, the messages' rotation signs, then
the momentum's; in a step, each new adapter's rows of A and their store's rounding, then its B's rounding, each
matrix in the model's order, before the step's weights are read; then each block's coded messages' rounding (or a
vote's ties), in the order the blocks send them; then each trained tensor's coded momentum and its store's rounding,
in the model's order, an adapter's A before its B.

Each step rebuilds the weights it trains from the codes: an adapter's factors as new tensors, so the codes are all
one step hands the next of them, and a matrix trained whole written over its weight, which θ holds. The codes, θ's
copy of a matrix trained whole, the optimizer's state and the codes' grids and rotations are buffers no state dict
holds. The output hands each adapter's factors, dequantized, as its trained parts
(`codecs.LowRank` at `e_bits`), and each matrix trained whole as its codes and scales (`codecs.ScaledQuant`).
"""
from __future__ import annotations

import fnmatch
import math

import torch
import torch.nn.functional as F

from training_slowdown.codecs import Identity, LowRank, Plan, ScaledQuant
from training_slowdown.model import LM, Linear

from . import _codes as codes
from . import lora, subset
from ._optimizers import newton_schulz

VERSION = 1
DIALS = {"targets": "blocks.*.weight", "rank": 0, "grow": "", "opt": "sign", "lr": 3e-4, "warmup": 0.0,
         "decay": 1.0, "b1": 0.9, "b2": 0.95, "clip": 0.0, "mom_bits": 0, "msg_bits": 32, "vote": False,
         "rotate": False, "micro_batches": 8, "store_bits": 8, "group": 64, "gains": False, "gain_lr": 1e-3,
         "e_bits": 4}
OPTS = ("sign", "adamw", "muon")
GAINS = ("attn_norm", "mlp_norm")


def phases(dials: dict) -> list[tuple[float, list[str], int]]:
    """The adapters' phases: (FRAC, patterns, rank), the first `targets` at `rank` from the start."""
    out = [(0.0, lora.patterns(dials), dials["rank"])]
    for part in filter(None, (p.strip() for p in dials["grow"].split(";"))):
        frac, pats, rank = part.split(":")
        out.append((float(frac), [p.strip() for p in pats.split(",") if p.strip()], int(rank)))
    return out


def plan(dials: dict) -> Plan:
    if dials["opt"] not in OPTS:
        raise ValueError(f"dgrad: opt {dials['opt']!r} is none of {', '.join(OPTS)}")
    if not 0 < dials["decay"] <= 1 or dials["warmup"] < 0 or dials["micro_batches"] < 1:
        raise ValueError("dgrad: decay is a share of the steps in (0, 1], warmup is not negative and micro_batches is "
                         "at least 1")
    if not (1 <= dials["msg_bits"] <= 8 or dials["msg_bits"] == 32) or dials["vote"] and dials["msg_bits"] != 1:
        raise ValueError("dgrad: msg_bits is 1 to 8, or 32 uncoded; a vote is one bit")
    if not (0 <= dials["mom_bits"] <= 8 or dials["mom_bits"] == 32) or dials["mom_bits"] and dials["opt"] == "adamw":
        raise ValueError("dgrad: mom_bits is 0 to 8, or 32 in fp32, and adamw keeps its own moments")
    if not 2 <= dials["store_bits"] <= 8 or dials["group"] < 1 or dials["rank"] < 0:
        raise ValueError("dgrad: store_bits is 2 to 8, group at least 1 and rank not negative")
    steps = phases(dials)
    if dials["grow"] and (dials["rank"] < 1 or any(not 0 < f < 1 or r < 1 for f, _, r in steps[1:])
                          or any(a[0] >= b[0] for a, b in zip(steps, steps[1:]))):
        raise ValueError("dgrad: grow raises adapters (rank at least 1) at increasing FRACs in (0, 1), each to a rank "
                         "of at least 1")
    if dials["grow"] and 0 < dials["mom_bits"] < 32:
        raise ValueError("dgrad: grow keeps no coded momentum (a grown factor's codes would lie across its groups)")
    pats = sorted({p for _, ps, _ in steps for p in ps})
    gains = [("norm", Identity()), *((f"blocks.*.{g}", Identity()) for g in GAINS)] if dials["gains"] else []
    if dials["rank"]:
        return Plan([(p, LowRank(max(r for *_, r in steps), 1.0, dials["e_bits"])) for p in pats] + gains)
    return Plan([(p, ScaledQuant(dials["store_bits"], dials["group"])) for p in pats] + gains)


def trained(model: torch.nn.Module, dials: dict) -> dict[str, Linear]:
    """Each trained matrix's layer, by the matrix's name: every matrix a phase names."""
    pats = {p for _, ps, _ in phases(dials) for p in ps}
    return lora.adapted(model, {"targets": ",".join(sorted(pats))})


def ranks_at(names: list[str], dials: dict, t: int, n: int) -> dict[str, int]:
    """Each adapted matrix's rank at step `t` of `n` (0: no adapter yet)."""
    out = dict.fromkeys(names, 0)
    for frac, pats, rank in phases(dials):
        if t >= (math.floor(frac * n) + 1 if frac else 0):
            out |= {k: rank for k in names if any(fnmatch.fnmatchcase(k, p) for p in pats)}
    return out


def gains_of(model: torch.nn.Module, layers: dict[str, Linear]) -> list[torch.nn.Parameter]:
    """The final norm's gain and every block's from the lowest trained matrix's block up."""
    lo = min(int(k.split(".")[1]) for k in layers)
    return [model.norm] + [getattr(b, g) for i, b in enumerate(model.blocks) if i >= lo for g in GAINS]


def through_adapter(layer: Linear, inputs, out: torch.Tensor) -> torch.Tensor:
    return out + F.linear(F.linear(inputs[0], layer.a), layer.b) if layer.rank else out


def start_as_loaded(layer: Linear, state: dict, prefix: str, *_) -> None:
    if (w := state.get(prefix + "weight")) is not None and not layer.loaded:
        layer.start, layer.loaded = w.detach().clone(), True


def held(layer: Linear, name: str, shape, dials: dict) -> None:
    """The buffers of one trained tensor: its codes, and the optimizer's state for it."""
    n, g = math.prod(shape), dials["group"]
    layer.register_buffer(f"{name}_codes", torch.full((n // g, g), codes.zero(dials["store_bits"]), dtype=torch.uint8),
                          persistent=False)
    layer.register_buffer(f"{name}_scales", torch.zeros(n // g, dtype=torch.float16), persistent=False)
    state = {}
    if dials["opt"] == "adamw":
        state = {"m": torch.zeros(shape), "v": torch.zeros(shape)}
    elif dials["mom_bits"] == 32:
        state = {"m": torch.zeros(shape)}
    elif dials["mom_bits"]:
        bits = dials["mom_bits"]
        z = torch.full((n // g, g), codes.zero(bits), dtype=torch.int64)
        state = {"planes": torch.stack([((z >> b) & 1).bool() for b in range(bits)]),
                 "mscales": torch.zeros(n // g, dtype=torch.float16)}
    for k, v in state.items():
        layer.register_buffer(f"{name}_{k}", v, persistent=False)


def build(shape: dict, dials: dict):
    plan(dials)
    model = LM.of(shape).requires_grad_(False)
    layers = trained(model, dials)
    if not layers:
        raise SystemExit(f"dgrad: targets {dials['targets']!r} name no matrix of the start")
    g = dials["group"]
    held_ranks = dict.fromkeys(layers, 0)
    for _, pats, rank in phases(dials) if dials["rank"] else ():
        for k in [k for k in layers if any(fnmatch.fnmatchcase(k, p) for p in pats)]:
            if rank < held_ranks[k]:
                raise SystemExit(f"dgrad: a phase of grow lowers {k}'s adapter from rank {held_ranks[k]} to {rank}")
            held_ranks[k] = rank
    for name, layer in layers.items():
        out, into = layer.weight.shape
        if dials["rank"]:
            if out % g or into % g:
                raise SystemExit(f"dgrad: groups of {g} don't divide {name}'s sides (a factor's rows)")
            top = held_ranks[name]
            layer.a, layer.b, layer.rank = None, None, 0
            held(layer, "a", (top, into), dials)
            held(layer, "b", (out, top), dials)
            layer.register_forward_hook(through_adapter)
        else:
            if out * into % g:
                raise SystemExit(f"dgrad: groups of {g} don't divide {name}'s {out * into} entries")
            layer.weight.requires_grad_(True)
            layer.register_buffer("start", layer.weight.detach().clone(), persistent=False)
            layer.loaded = False
            held(layer, "w", (out, into), dials)
            layer.register_load_state_dict_pre_hook(start_as_loaded)
    model.register_buffer("store_levels", codes.grid(dials["store_bits"]), persistent=False)
    model.register_buffer("store_codebook", codes.codebook(dials["store_bits"]), persistent=False)
    if dials["vote"] or dials["msg_bits"] < 32:
        model.register_buffer("msg_levels", codes.grid(dials["msg_bits"]), persistent=False)
        if dials["rotate"] and not dials["vote"]:
            model.register_buffer("msg_rotation", codes.rotation(g), persistent=False)
    if 0 < dials["mom_bits"] < 32:
        model.register_buffer("mom_levels", codes.grid(dials["mom_bits"]), persistent=False)
        if dials["rotate"]:
            model.register_buffer("mom_rotation", codes.rotation(g), persistent=False)
    optimizers = []
    if dials["gains"]:
        gains = gains_of(model, layers)
        for p in gains:
            p.requires_grad_(True)
        optimizers.append(torch.optim.AdamW(gains, lr=dials["gain_lr"], betas=(0.9, 0.95), weight_decay=0.0,
                                            foreach=True))
    return model, optimizers


class Heap:
    """A step's messages between its blocks: `push` each block's gradients in turn, `root` the sum the last one
    reaches."""

    def __init__(self, model, dials: dict, blocks: int):
        self.n, self.i, self.pending, self.out = blocks, 0, {}, None
        self.vote, self.bits, self.group = dials["vote"], dials["msg_bits"], dials["group"]
        self.coded = self.vote or self.bits < 32
        self.levels = getattr(model, "msg_levels", None)
        self.rot = getattr(model, "msg_rotation", None)

    def _pack(self, g: dict):
        if not self.coded:
            return g
        self.layout = [(k, t.shape) for k, t in g.items()]
        flat = torch.cat([t.reshape(-1) for t in g.values()])
        return codes.vote_pack(flat) if self.vote else (codes.pack(flat, self.bits, self.group, self.levels, self.rot),
                                                         flat.shape)

    def _unpack(self, m) -> dict:
        if not self.coded:
            return m
        if self.vote:
            flat = codes.vote_unpack(m)
        else:
            (planes, s), shape = m
            flat = codes.unpack(planes, s, shape, self.levels, self.rot)
        sizes = [math.prod(sh) for _, sh in self.layout]
        return {k: t.reshape(sh) for (k, sh), t in zip(self.layout, flat.split(sizes))}

    def push(self, g: dict) -> None:
        j = self.n - 1 - self.i
        own = self._unpack(self._pack(g)) if self.vote or self.coded and not j else g
        kids = [self._unpack(self.pending.pop(k)) for k in (2 * j + 1, 2 * j + 2) if k in self.pending]
        total = {k: sum(x[k] for x in (own, *kids)) for k in own}
        if j:
            self.pending[j] = self._pack(total)
        else:
            self.out = total
        self.i += 1


def tensors(layers: dict[str, Linear], ranks: dict[str, int]) -> dict[str, tuple[Linear, str, int]]:
    """Each trained tensor this step, by its key: (its layer, its name there, its rank, 0 for a matrix whole)."""
    out = {}
    for name, layer in layers.items():
        if hasattr(layer, "a_codes"):
            if ranks[name]:
                out |= {f"{name}#A": (layer, "a", ranks[name]), f"{name}#B": (layer, "b", ranks[name])}
        else:
            out[name] = (layer, "w", 0)
    return out


def stored(layer: Linear, name: str, r: int) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """A trained tensor's codes and scales in use at rank `r`, and the shape they stand for (B's transposed)."""
    out, into = layer.weight.shape
    shape = {"a": (r, into), "b": (r, out), "w": (out, into)}[name]
    groups = math.prod(shape) // getattr(layer, f"{name}_codes").shape[1]
    return getattr(layer, f"{name}_codes")[:groups], getattr(layer, f"{name}_scales")[:groups], shape


def factor(layer: Linear, name: str, r: int, levels: torch.Tensor) -> torch.Tensor:
    c, s, shape = stored(layer, name, r)
    v = codes.values(c, s, levels, shape)
    # B laid out (out, r) in rows, as a parameter is: autograd arranges B's gradient product by its strides, which
    # `contiguous` leaves a column's view at
    return v.T.clone(memory_format=torch.contiguous_format) if name == "b" else v


def sliced(t: torch.Tensor, name: str, r: int) -> torch.Tensor:
    return t[:r] if name == "a" else t[:, :r] if name == "b" else t


def momentum(model, layer: Linear, name: str, r: int, x: torch.Tensor, keep: float, dials: dict) -> torch.Tensor:
    """m ← b1·m + keep·x, kept at `mom_bits`; the kept m, as the next step reads it."""
    b1 = dials["b1"]
    if dials["mom_bits"] == 32:
        m = sliced(getattr(layer, f"{name}_m"), name, r)
        m.copy_(b1 * m + keep * x)
        return m
    planes, s = getattr(layer, f"{name}_planes"), getattr(layer, f"{name}_mscales")
    groups = x.numel() // dials["group"]
    levels, rot = model.mom_levels, getattr(model, "mom_rotation", None)
    m = codes.unpack(planes[:, :groups], s[:groups], x.shape, levels, rot)
    new_planes, new_s = codes.pack(b1 * m + keep * x, dials["mom_bits"], dials["group"], levels, rot)
    planes[:, :groups].copy_(new_planes)
    s[:groups].copy_(new_s)
    return codes.unpack(planes[:, :groups], s[:groups], x.shape, levels, rot)


def update(model, layer: Linear, name: str, r: int, g: torch.Tensor, lr: float, t: int, dials: dict) -> torch.Tensor:
    """The step the optimizer takes on one trained tensor, given its gradient `g`."""
    b1, b2 = dials["b1"], dials["b2"]
    if dials["opt"] == "adamw":
        m, v = sliced(getattr(layer, f"{name}_m"), name, r), sliced(getattr(layer, f"{name}_v"), name, r)
        m.mul_(b1).add_(g, alpha=1 - b1)
        v.mul_(b2).addcmul_(g, g, value=1 - b2)
        mh, vh = m / (1 - b1 ** (t + 1)), v / (1 - b2 ** (t + 1))
        return lr * mh / (vh.sqrt() + 1e-8)
    if dials["opt"] == "muon":
        u = g + b1 * momentum(model, layer, name, r, g, 1.0, dials) if dials["mom_bits"] else g
        return lr * max(1.0, g.shape[0] / g.shape[1]) ** 0.5 * newton_schulz(u)
    return lr * (momentum(model, layer, name, r, g, 1 - b1, dials) if dials["mom_bits"] else g).sign()


def write(model, layer: Linear, name: str, r: int, step: torch.Tensor, dials: dict) -> None:
    c, s, shape = stored(layer, name, r)
    new_c, new_s = codes.sticky_add(c, s, step.reshape(shape), dials["group"], model.store_levels)
    c.copy_(new_c)
    s.copy_(new_s)


def step(model, rows, optimizers, dials: dict, t: int, n: int) -> None:
    layers = trained(model, dials)
    frac = subset.schedule(dials, t, n)
    for o in optimizers:
        for grp in o.param_groups:
            grp["lr"] = dials["gain_lr"] * frac
    ranks = ranks_at(list(layers), dials, t, n) if dials["rank"] else {}
    before = ranks_at(list(layers), dials, t - 1, n) if dials["rank"] and t else dict.fromkeys(ranks, 0)
    with torch.no_grad():
        for name, layer in layers.items():
            if dials["rank"] and ranks[name] > before[name]:
                r0, r = before[name], ranks[name]
                out, into = layer.weight.shape
                for part, rows_ in (("a", torch.randn((r - r0, into)) / math.sqrt(into)),
                                    ("b", torch.zeros((r - r0, out)))):
                    c, s, _ = stored(layer, part, r)
                    lo = r0 * c.shape[0] // r
                    new_c, new_s = codes.sticky_add(c[lo:], s[lo:], rows_, dials["group"], model.store_levels)
                    c[lo:].copy_(new_c)
                    s[lo:].copy_(new_s)
        for name, layer in layers.items():
            if dials["rank"]:
                layer.rank = r = ranks[name]
                layer.a, layer.b = ((factor(layer, "a", r, model.store_levels).requires_grad_(),
                                     factor(layer, "b", r, model.store_levels).requires_grad_()) if r else (None, None))
            else:
                layer.weight.copy_(ScaledQuant.apply(layer.start, model.store_codebook, layer.w_scales, layer.w_codes))
    keys = tensors(layers, ranks)
    blocks = rows.chunk(dials["micro_batches"])
    heap = Heap(model, dials, len(blocks))
    for x in blocks:
        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        loss.backward()
        heap.push({k: getattr(layer, "weight" if name == "w" else name).grad for k, (layer, name, _) in keys.items()})
        for layer, name, _ in keys.values():
            getattr(layer, "weight" if name == "w" else name).grad = None
    g = {k: v / len(blocks) for k, v in heap.out.items()}
    gains = [p for o in optimizers for grp in o.param_groups for p in grp["params"]]
    with torch.no_grad():
        for p in gains:
            p.grad.div_(len(blocks))
        if dials["clip"]:
            norm = torch.sqrt(sum(v.pow(2).sum() for v in g.values()) + sum(p.grad.pow(2).sum() for p in gains))
            f = torch.clamp(dials["clip"] / (norm + 1e-6), max=1.0)
            g = {k: v * f for k, v in g.items()}
            for p in gains:
                p.grad.mul_(f)
        lr = dials["lr"] * frac
        for k, (layer, name, r) in keys.items():
            s = update(model, layer, name, r, g[k], lr, t, dials)
            write(model, layer, name, r, -(s.T if name == "b" else s), dials)
    for o in optimizers:
        o.step()
        o.zero_grad(set_to_none=True)


def output(model, dials: dict) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
    layers = trained(model, dials)
    state = model.state_dict()
    if dials["rank"]:
        return state | {name: (factor(layer, "a", layer.rank, model.store_levels),
                               factor(layer, "b", layer.rank, model.store_levels))
                        for name, layer in layers.items() if layer.rank}
    return state | {name: (model.store_codebook, layer.w_scales, layer.w_codes) for name, layer in layers.items()}
