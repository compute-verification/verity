"""Each entry of the library held to a reference in plain torch: the entry's own dials, read from its config, run for
a few steps at a small shape through its route, and the same steps written here against torch's own optimizers and
its own average of the weights, or against the published rule where torch has none. The two end at the same output
model. An entry whose output is an average over a horizon is held to torch's average of the weights the entry beside
it writes, the one whose output is those weights, which has its own reference.

What is held to a reference is the update's rule, so both sides run their forward in float32 (`forward_in_float32`):
a reference sums a gradient in another order than its route (accumulation's micro-batches against the whole batch,
LoRA's two factors against the merged matrix), and under the routes' bf16 autocast the two orders differ in the
gradient's third digit, enough to turn the sign of an Adam step."""
import contextlib
import fnmatch
import math
import tomllib

import pytest
import torch
import torch.nn.functional as F
from synthetic import SHAPE as TINY
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from training_slowdown import config, library, starts
from training_slowdown.model import LM
from training_slowdown.routes import _codes as codes
from training_slowdown.routes import subset
from training_slowdown.routes._optimizers import Muon, newton_schulz

#: 16 wide; an entry's own test runs it at its start's depth (`at_its_depth`), so each block its patterns name is there
SHAPE = TINY | {"n_layer": 6}
STEPS, BATCH, CTX = 3, 4, 8
#: the steps of an entry whose average has a horizon: enough that a tenth of them is over a step, where the average
#: starts to keep some of itself
LONG = 24


@pytest.fixture(autouse=True)
def forward_in_float32(monkeypatch):
    """The routes' forward in float32: their bf16 autocast (`subset.step`'s and `qat.step`'s) is switched off for
    these tests."""
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())


def entry(name: str):
    """An entry's route and its dials, the route's defaults filled in, as a job reads them."""
    raw = tomllib.loads((library.ENTRIES / f"{name}.toml").read_text())
    route = config.route(raw["route"])
    return route, config.dials(route, raw.get("dials", {}), name)


def at_its_depth(name: str, monkeypatch) -> None:
    """`SHAPE` with the blocks of the entry's start."""
    raw = tomllib.loads((library.ENTRIES / f"{name}.toml").read_text())
    monkeypatch.setitem(SHAPE, "n_layer", starts.start(raw["start"]).shape["n_layer"])


def steps_of(dials: dict) -> int:
    return LONG if dials.get("ema_horizon") else STEPS


def batches(steps: int = STEPS) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    return [torch.randint(0, SHAPE["vocab"], (BATCH, CTX + 1), generator=g) for _ in range(steps)]


def routed(route, dials: dict) -> dict:
    """The output model after the route's own steps. A weight the route hands over as its trained parts is the one
    the route's plan rebuilds on the start from its packing of them, as a job's checkpoint takes it."""
    torch.manual_seed(0)
    model, optimizers = route.build(SHAPE, dials)
    start = {k: v.clone() for k, v in model.state_dict().items()}
    steps = steps_of(dials)
    for t, rows in enumerate(batches(steps)):
        route.step(model, rows, optimizers, dials, t, steps)
    out = dict(route.output(model, dials) if hasattr(route, "output") else model.state_dict())
    parts, plan = {k: v for k, v in out.items() if isinstance(v, tuple)}, route.plan(dials)
    return out | (plan.decode(plan.encode(parts, start), start) if parts else {})


class Rule(torch.optim.Optimizer):
    """A published update rule as an optimizer: `rule(p, g, state, group)` takes one tensor's step in place."""

    def __init__(self, params, rule, **group):
        super().__init__(params, group)
        self.rule = rule

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                self.rule(p, p.grad, self.state[p], group)


def plain(dials: dict, optimizers) -> dict:
    """The weights after the same steps as plain torch writes them: the parameters the dials' patterns name, one
    forward and backward over the whole batch, the gradient clipped, each optimizer of `optimizers(matrices, beside)`
    (the blocks' matrices; the gains and the embedding) stepped at its groups' rates times the schedule. With an
    `ema_decay`, torch's own average of them (`AveragedModel`)."""
    torch.manual_seed(0)
    model = LM.of(SHAPE)
    average = "ema_decay" in dials and AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(dials["ema_decay"]))
    patterns = [p.strip() for p in dials["trainable"].split(",")]
    for n, p in model.named_parameters():
        p.requires_grad_(any(fnmatch.fnmatchcase(n, pat) for pat in patterns))
    params = [p for p in model.parameters() if p.requires_grad]
    beside = [p for p in params if p.ndim != 2 or p is model.embed]
    opts = optimizers([p for p in params if p.ndim == 2 and p is not model.embed], beside)
    for t, rows in enumerate(batches()):
        logits = model(rows[:, :-1])
        F.cross_entropy(logits.flatten(0, 1), rows[:, 1:].flatten()).backward()
        torch.nn.utils.clip_grad_norm_(params, dials["clip"])
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g.setdefault("rate", g["lr"]) * subset.schedule(dials, t, STEPS)
            o.step()
            o.zero_grad()
        if average:
            average.update_parameters(model)
    return (average.module if average else model).state_dict()


def under(optimizers):
    return lambda dials: plain(dials, optimizers(dials))


def averaged(dials: dict) -> dict:
    """torch's own average (`AveragedModel`) of the weights `subset`'s steps write under the same dials, each update
    taking the share of the new weights that reaches back `ema_horizon` of the steps so far."""
    torch.manual_seed(0)
    model, optimizers = subset.build(SHAPE, dials)
    average = AveragedModel(model, avg_fn=lambda mean, new, seen: mean + (new - mean) / max(
        1.0, dials["ema_horizon"] * (seen + 1)))
    for t, rows in enumerate(batches(LONG)):
        subset.step(model, rows, optimizers, dials, t, LONG)
        average.update_parameters(model)
    return average.module.state_dict()


def rated(beside: list, d: dict) -> list[dict]:
    """The gains and the embedding as AdamW's groups, each tensor at its kind's own rate where the dials give one."""
    return [{"params": [p], "lr": d.get("embed_lr" if p.ndim == 2 else "gain_lr") or d["lr"]} for p in beside]


def lora(dials: dict) -> dict:
    """LoRA written the other way round: each adapted matrix is built whole, W + (alpha/rank) B A, before one forward
    of the start's own model on it, where the route keeps W frozen and sends its input through A then B. A starts as
    the route builds it and B at zero, both under torch's AdamW; the output is the matrices as built."""
    torch.manual_seed(0)
    built = config.route("lora").build(SHAPE, dials)[0].state_dict()
    theta = {k: v for k, v in built.items() if not k.endswith((".a", ".b"))}
    a = {f"{k[:-2]}.weight": built[k].clone().requires_grad_() for k in built if k.endswith(".a")}
    b = {k: torch.zeros(len(theta[k]), dials["rank"], requires_grad=True) for k in a}
    opt = torch.optim.AdamW([*a.values(), *b.values()], lr=dials["lr"], betas=(0.9, 0.95),
                            weight_decay=dials["weight_decay"])

    def merged() -> dict:
        return theta | {k: theta[k] + dials["alpha"] / dials["rank"] * b[k] @ a[k] for k in a}

    model = LM.of(SHAPE)
    for t, rows in enumerate(batches()):
        logits = torch.func.functional_call(model, merged(), (rows[:, :-1],))
        F.cross_entropy(logits.flatten(0, 1), rows[:, 1:].flatten()).backward()
        torch.nn.utils.clip_grad_norm_([*a.values(), *b.values()], dials["clip"])
        opt.param_groups[0]["lr"] = dials["lr"] * subset.schedule(dials, t, STEPS)
        opt.step()
        opt.zero_grad()
    return {k: v.detach() for k, v in merged().items()}


def qat(dials: dict) -> dict:
    """Quantization-aware training written out: every matrix the dials name is the start's plus the rounding of a
    latent, zero at first. The rounding is by hand: the latent's entries at least as large as its ceil(keep · n)-th
    largest magnitude, and not zero, keep their sign, at their mean magnitude as f16 holds it. Each step builds those matrices, runs one forward of the
    start's own model on them, takes each matrix's gradient as its latent's, clips over the latents and steps them
    at the scheduled rate; the output is the matrices as the last latents build them.

    The latents step under the route's own Muon, which the Muon entries hold to the published rule, and the scale is
    summed as the codec sums it: one rounding apart before Muon's bf16 Newton-Schulz moves a latent by percents of
    its step, and a latent that crosses the rounding's threshold is a weight a whole scale off."""
    torch.manual_seed(0)
    model = LM.of(SHAPE)
    theta = {k: v.detach().clone() for k, v in model.state_dict().items()}
    latents = {k: torch.zeros_like(v, requires_grad=True) for k, v in theta.items() if v.ndim == 2 and any(
        fnmatch.fnmatchcase(k, p.strip()) for p in dials["targets"].split(","))}
    opt = Muon(list(latents.values()), lr=dials["lr"])

    def built() -> dict:
        out = {}
        for k, v in latents.items():
            a = v.detach().abs().reshape(1, -1)
            kept = (a >= a.topk(math.ceil(dials["keep"] * a.numel())).values[0, -1]) & (a > 0)
            scale = ((a * kept).sum(1) / kept.sum(1).clamp(min=1)).half().float()
            out[k] = theta[k] + (v.detach().sign().reshape(1, -1) * kept * scale).view_as(v)
        return out

    for t, rows in enumerate(batches()):
        weights = {k: w.requires_grad_() for k, w in built().items()}
        logits = torch.func.functional_call(model, theta | weights, (rows[:, :-1],))
        F.cross_entropy(logits.flatten(0, 1), rows[:, 1:].flatten()).backward()
        for k, v in latents.items():
            v.grad = weights[k].grad
        torch.nn.utils.clip_grad_norm_(list(latents.values()), dials["clip"])
        left = min(1.0, (1 - t / STEPS) / dials["decay"])
        fall = {"linear": left, "sqrt": 1 - math.sqrt(1 - left)}[dials["cooldown"]]
        opt.param_groups[0]["lr"] = dials["lr"] * min(1.0, (t + 1) / dials["warmup"]) * fall
        opt.step()
        opt.zero_grad()
    return theta | built()


def dgrad(dials: dict) -> dict:
    """The coded reduction written out, a block of rows at a time. Each block of rows takes one forward of the start's
    own model on the matrices as the store gives them (θ + its codes' values for a matrix trained whole; for an
    adapter, θ's product plus x·Aᵀ·Bᵀ, the factors from their codes) and its gradient with respect to each trained
    tensor. An adapter is its own product, as the route computes it, not merged into θ: the two round apart in a
    last bit on some CPUs, which the store's stochastic rounding can turn into a whole level. Blocks are sent
    in turn: block j (the last block 0) adds its children's (2j + 1, 2j + 2) to its own, coded (a vote takes its
    own as a sign too, and the root codes its own), and sends the sum on, coded; the root's sum over the blocks is
    the step's gradient. The update rule is the program's, on that gradient: the sign, the sign of a momentum or
    Muon of the gradient plus the momentum (the momentum coded at its bits), or AdamW with its bias correction; its
    step is added to the store's codes; the gains step under torch's AdamW. Every random value comes from the
    global generator in the order the route's docstring states. The output is θ with the store's matrices, an adapter's
    factors rounded as E rounds them."""
    torch.manual_seed(0)
    model = LM.of(SHAPE)
    theta = {k: v.detach().clone() for k, v in model.state_dict().items()}
    g, bits, vote = dials["group"], dials["msg_bits"], dials["vote"]
    coded = vote or bits < 32
    msg_grid = codes.grid(bits) if coded else None
    msg_rot = codes.rotation(g) if coded and dials["rotate"] and not vote else None
    mom_coded = 0 < dials["mom_bits"] < 32
    mom_grid = codes.grid(dials["mom_bits"]) if mom_coded else None
    mom_rot = codes.rotation(g) if mom_coded and dials["rotate"] else None
    grid = codes.grid(dials["store_bits"])
    phases = [(0, [p.strip() for p in dials["targets"].split(",")], dials["rank"])] + [
        (math.floor(float(f) * STEPS) + 1, ps.split(","), int(r))
        for f, ps, r in (p.split(":") for p in dials["grow"].split(";") if p)]
    names = [k for k, v in theta.items() if v.ndim == 2 and any(fnmatch.fnmatchcase(k, p) for _, ps, _ in phases
                                                                 for p in ps)]

    def rank(k: str, t: int) -> int:
        return max([r for at, ps, r in phases if at <= t and any(fnmatch.fnmatchcase(k, p) for p in ps)], default=0)

    def zero(n: int):
        return (torch.full((n // g, g), codes.zero(dials["store_bits"]), dtype=torch.uint8),
                torch.zeros(n // g, dtype=torch.float16))

    def to_codes(t: torch.Tensor):
        return codes.pack(t, bits, g, msg_grid, msg_rot) if not vote else codes.vote_pack(t)

    def from_codes(m, n: int) -> torch.Tensor:
        return codes.vote_unpack(m) if vote else codes.unpack(*m, (n,), msg_grid, msg_rot)

    def coded_sum(t: torch.Tensor) -> torch.Tensor:
        return from_codes(to_codes(t), t.numel()) if coded else t

    store = {} if dials["rank"] else {k: zero(theta[k].numel()) for k in names}
    shapes = {}
    state: dict[str, dict] = {}
    gains = {}
    if dials["gains"]:
        lo = min(int(k.split(".")[1]) for k in names)
        gains = {k: theta[k].clone().requires_grad_() for k in theta
                 if k == "norm" or k.endswith(("attn_norm", "mlp_norm")) and int(k.split(".")[1]) >= lo}
    gain_opt = torch.optim.AdamW(list(gains.values()), lr=dials["gain_lr"], betas=(0.9, 0.95), weight_decay=0.0,
                                 foreach=True) if gains else None
    factors: dict[str, torch.Tensor] = {}
    for k in names if dials["rank"] else ():
        def through(layer, inputs, out, k=k):
            if f"{k}#A" not in factors:
                return out
            return out + F.linear(F.linear(inputs[0], factors[f"{k}#A"]), factors[f"{k}#B"])
        model.get_submodule(k.removesuffix(".weight")).register_forward_hook(through)
    for t, rows in enumerate(batches()):
        frac = subset.schedule(dials, t, STEPS)
        if dials["rank"]:
            for k in names:
                r0, r = rank(k, t - 1) if t else 0, rank(k, t)
                if r > r0:
                    out, into = theta[k].shape
                    new = {"A": torch.randn((r - r0, into)) / math.sqrt(into), "B": torch.zeros((r - r0, out))}
                    for part, x in new.items():
                        c, s = codes.sticky_add(*zero(x.numel()), x, g, grid)
                        kept = store.get(f"{k}#{part}", (c[:0], s[:0]))
                        store[f"{k}#{part}"] = (torch.cat([kept[0], c]), torch.cat([kept[1], s]))
                        shapes[f"{k}#{part}"] = (r, into if part == "A" else out)
        else:
            shapes = {k: theta[k].shape for k in names}
        store = {k: store[k] for n in names for k in (n, f"{n}#A", f"{n}#B") if k in store}
        held = {k: codes.values(*store[k], grid, shapes[k]) for k in store}
        tensors = {k: (v.T if k.endswith("#B") else v if dials["rank"] else theta[k] + v).clone().requires_grad_()
                   for k, v in held.items()}
        for k, v in tensors.items():
            for name in ("m", "v"):
                kept, state.setdefault(k, {})[name] = state.setdefault(k, {}).get(name), torch.zeros(v.shape)
                if kept is not None:
                    state[k][name][tuple(slice(0, n) for n in kept.shape)] = kept

        factors.clear()
        factors.update(tensors if dials["rank"] else {})

        def weights() -> dict:
            return theta | gains | ({} if dials["rank"] else tensors)

        blocks, sent, root = rows.chunk(dials["micro_batches"]), {}, None
        for i, x in enumerate(blocks):
            logits = torch.func.functional_call(model, weights(), (x[:, :-1],))
            own = torch.autograd.grad(F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten()),
                                      [*tensors.values(), *gains.values()])
            for p, d in zip(gains.values(), own[len(tensors):]):
                p.grad = d if p.grad is None else p.grad + d
            flat = torch.cat([d.reshape(-1) for d in own[:len(tensors)]])
            j = len(blocks) - 1 - i
            total = coded_sum(flat) if vote or coded and not j else flat
            for kid in (2 * j + 1, 2 * j + 2):
                if kid in sent:
                    total = total + sent.pop(kid)
            if j:
                sent[j] = coded_sum(total)
            else:
                root = total
        grad = dict(zip(tensors, (root / len(blocks)).split([v.numel() for v in tensors.values()])))
        grad = {k: d.view(tensors[k].shape) for k, d in grad.items()}
        with torch.no_grad():
            for p in gains.values():
                p.grad.div_(len(blocks))
            if dials["clip"]:
                norm = torch.sqrt(sum(d.pow(2).sum() for d in grad.values()) +
                                  sum(p.grad.pow(2).sum() for p in gains.values()))
                f = torch.clamp(dials["clip"] / (norm + 1e-6), max=1.0)
                grad = {k: d * f for k, d in grad.items()}
                for p in gains.values():
                    p.grad.mul_(f)
            lr, b1, b2 = dials["lr"] * frac, dials["b1"], dials["b2"]
            for k, d in grad.items():
                st = state[k]
                if dials["opt"] == "adamw":
                    st["m"] = b1 * st["m"] + (1 - b1) * d
                    st["v"] = b2 * st["v"] + (1 - b2) * d * d
                    step = lr * (st["m"] / (1 - b1 ** (t + 1))) / ((st["v"] / (1 - b2 ** (t + 1))).sqrt() + 1e-8)
                else:
                    keep = 1.0 if dials["opt"] == "muon" else 1 - b1
                    if dials["mom_bits"] == 32:
                        st["m"] = b1 * st["m"] + keep * d
                    elif mom_coded:
                        before = st.get("coded", (torch.full((dials["mom_bits"], d.numel() // g, g), False),
                                                  torch.zeros(d.numel() // g, dtype=torch.float16)))
                        m = codes.unpack(*before, d.shape, mom_grid, mom_rot)
                        st["coded"] = codes.pack(b1 * m + keep * d, dials["mom_bits"], g, mom_grid, mom_rot)
                        st["m"] = codes.unpack(*st["coded"], d.shape, mom_grid, mom_rot)
                    if dials["opt"] == "muon":
                        u = d + b1 * st["m"] if dials["mom_bits"] else d
                        step = lr * max(1.0, d.shape[0] / d.shape[1]) ** 0.5 * newton_schulz(u)
                    else:
                        step = lr * (st["m"] if dials["mom_bits"] else d).sign()
                step = -(step.T if k.endswith("#B") else step)
                store[k] = codes.sticky_add(*store[k], step.reshape(shapes[k]), g, grid)
            if gain_opt:
                gain_opt.param_groups[0]["lr"] = dials["gain_lr"] * frac
                gain_opt.step()
                gain_opt.zero_grad(set_to_none=True)
    held = {k: codes.values(*store[k], grid, shapes[k]) for k in store}
    out = theta | {k: p.detach() for k, p in gains.items()}
    if not dials["rank"]:
        return out | {k: theta[k] + held[k] for k in names}

    def rounded(rows: torch.Tensor) -> torch.Tensor:
        top = 2 ** (dials["e_bits"] - 1) - 1
        s = (rows.abs().amax(1) / top).half().float()[:, None]
        return torch.where(s > 0, torch.round(rows / torch.where(s > 0, s, 1.0)).clamp(-top, top) * s, 0.0)

    return out | {k: theta[k] + rounded(held[f"{k}#B"]).T @ rounded(held[f"{k}#A"]) for k in names
                  if f"{k}#A" in held}


def adamw(d: dict):
    return lambda matrices, beside: [torch.optim.AdamW(matrices + beside, lr=d["lr"], betas=(0.9, 0.95),
                                                      weight_decay=d["weight_decay"])]


def muon_rule(p, g, state, group):
    """Muon: Nesterov momentum 0.95, the update orthogonalized (the route's Newton-Schulz, which test_routes holds
    to the SVD's polar factor) and scaled by the matrix's shape, the decay decoupled."""
    buf = state["buf"] = 0.95 * state.get("buf", torch.zeros_like(p)) + g
    u = newton_schulz(g + 0.95 * buf)
    p.mul_(1 - group["lr"] * group["weight_decay"]).sub_(group["lr"] * max(1.0, p.shape[0] / p.shape[1]) ** 0.5 * u)


def muon(d: dict):
    return lambda matrices, beside: [Rule(matrices, muon_rule, lr=d["lr"], weight_decay=d["weight_decay"]),
                                    torch.optim.AdamW(rated(beside, d), betas=(0.9, 0.95), weight_decay=0.0)]


def sgd(d: dict):
    return lambda matrices, beside: [torch.optim.SGD(matrices + beside, lr=d["lr"], momentum=0.9, nesterov=True,
                                                    weight_decay=d["weight_decay"])]


def adafactor(d: dict):
    return lambda matrices, beside: [torch.optim.Adafactor(matrices + beside, lr=d["lr"],
                                                          weight_decay=d["weight_decay"])]


def lion_rule(p, g, state, group):
    """Lion: the step is the sign of 0.9 m + 0.1 g with the decay beside it, then m = 0.99 m + 0.01 g."""
    m = state.get("m", torch.zeros_like(p))
    p.sub_(group["lr"] * ((0.9 * m + 0.1 * g).sign() + group["weight_decay"] * p))
    state["m"] = 0.99 * m + 0.01 * g


def lion(d: dict):
    return lambda matrices, beside: [Rule(matrices + beside, lion_rule, lr=d["lr"], weight_decay=d["weight_decay"])]


def inverse_fourth_root(a: torch.Tensor) -> torch.Tensor:
    u, s, _ = torch.linalg.svd(a)
    return u @ torch.diag(s ** -0.25) @ u.T


def shampoo_rule(p, g, state, group):
    """Shampoo: L = 1e-4 I + the sum of G Gᵀ, R = 1e-4 I + the sum of Gᵀ G, the step L^(-1/4) G R^(-1/4), here with
    the factors in float64 and each root from an SVD; the decay decoupled."""
    g = g.double()
    left = state["left"] = state.get("left", 1e-4 * torch.eye(len(g), dtype=g.dtype)) + g @ g.T
    right = state["right"] = state.get("right", 1e-4 * torch.eye(g.shape[1], dtype=g.dtype)) + g.T @ g
    u = inverse_fourth_root(left) @ g @ inverse_fourth_root(right)
    p.mul_(1 - group["lr"] * group["weight_decay"]).sub_(group["lr"] * u.to(p.dtype))


def shampoo(d: dict):
    return lambda matrices, beside: [Rule(matrices, shampoo_rule, lr=d["lr"], weight_decay=d["weight_decay"]),
                                    torch.optim.AdamW(beside, lr=d["lr"], betas=(0.9, 0.95), weight_decay=0.0)]


#: the most a route's weight may differ from its reference's, as a share of the tensor's largest move from the start:
#: float rounding in another order of the same sums
CLOSE = 1e-3
#: the same where the update passes Newton-Schulz in bf16, which turns a rounding in its input into a few percent of
#: its output; a rule with another momentum, scale or decay is off by the whole move
CLOSE_BF16 = 0.1
#: beside it, a weight's own float32 rounding, a few times a step: all a sign step of a small rate differs by
EPS, ROUNDINGS = torch.finfo(torch.float32).eps, 4
#: each entry's reference: the output model plain torch ends at under the entry's dials, and how close the route's
#: comes to it
REFERENCES = {
    "l16-t20-1.1x-fullmuon-fe-wsd": (under(muon), CLOSE_BF16),
    "l16-t20-1.1x-fullmuon-fe-wsd-5383": (under(muon), CLOSE_BF16),
    "l16-t20-1.1x-fullmuon-fe-wsd-5383-ema": (averaged, CLOSE),
    "l16-t20-1.1x-fullmuon-fe-wsd-ema": (averaged, CLOSE),
    "l16-t20-1.1x-ref-optim-fullmuon-fe-wsd": (under(muon), CLOSE_BF16),
    "l16-t20-1.1x-ref-optim-fullmuon-fe-wsd-102": (under(muon), CLOSE_BF16),
    "l16-t20-1.1x-ref-optim-fullmuon-wsd": (under(muon), CLOSE_BF16),
    "l16-t20-1.1x-shallow-depthshare-4": (under(muon), CLOSE_BF16),
    "l16-t20-2x-fullmuon-fe-wsd-40326": (under(muon), CLOSE_BF16),
    "m-t20-1.1x-dgrad-grow": (dgrad, CLOSE),
    "m-t20-1.1x-fullmuon-wsd": (under(muon), CLOSE_BF16),
    "m-t20-1.1x-fullmuon-wsd-4611": (under(muon), CLOSE_BF16),
    "m-t20-1.1x-fullmuon-wsd-4611-ema": (averaged, CLOSE),
    "m-t20-1.1x-fullmuon-wsd-ema": (averaged, CLOSE),
    "m-t20-1.1x-ref-optim-1372": (under(muon), CLOSE_BF16),
    "m-t20-1.1x-ref-optim-fullmuon": (under(muon), CLOSE_BF16),
    "m-t20-1.1x-shallow-depthshare-4": (under(muon), CLOSE_BF16),
    "m-t20-1.1x-shallow-depthshare-8": (under(muon), CLOSE_BF16),
    "m-t20-1.1x-wildcard-topcode": (under(muon), CLOSE_BF16),
    "s-t20-1.1x-accumulate": (under(adamw), CLOSE),
    "s-t20-1.1x-adafactor": (under(adafactor), CLOSE),
    "s-t20-1.1x-adamw": (under(adamw), CLOSE),
    "s-t20-1.1x-dgrad-q2r": (dgrad, CLOSE),
    "s-t20-1.1x-dgrad-q4muon-full": (dgrad, CLOSE_BF16),
    "s-t20-1.1x-dgrad-q4r": (dgrad, CLOSE),
    "s-t20-1.1x-dgrad-q4r-mlp": (dgrad, CLOSE),
    "s-t20-1.1x-dgrad-vote": (dgrad, CLOSE),
    "s-t20-1.1x-ema": (under(adamw), CLOSE),
    "s-t20-1.1x-fullmuon": (under(muon), CLOSE_BF16),
    "s-t20-1.1x-fullmuon-ema": (averaged, CLOSE),
    "s-t20-1.1x-fullmuon-fe-wsd": (under(muon), CLOSE_BF16),
    "s-t20-1.1x-fullmuon-fe-wsd-ema": (averaged, CLOSE),
    "s-t20-1.1x-last-blocks": (under(adamw), CLOSE),
    "s-t20-1.1x-lion": (under(lion), CLOSE),
    "s-t20-1.1x-lora": (lora, CLOSE),
    "s-t20-1.1x-muon-whole": (under(muon), CLOSE_BF16),
    "s-t20-1.1x-qat-top2": (qat, CLOSE),
    "s-t20-1.1x-qat-top3": (qat, CLOSE),
    "s-t20-1.1x-qat-top4": (qat, CLOSE),
    "s-t20-1.1x-ref-optim": (under(muon), CLOSE_BF16),
    "s-t20-1.1x-ref-optim-1027": (under(muon), CLOSE_BF16),
    "s-t20-1.1x-ref-optim-747": (under(muon), CLOSE_BF16),
    "s-t20-1.1x-ref-optim-961": (under(muon), CLOSE_BF16),
    "s-t20-1.1x-ref-optim-fullmuon-758": (under(muon), CLOSE_BF16),
    "s-t20-1.1x-ref-optim-fullmuon-841": (under(muon), CLOSE_BF16),
    "s-t20-1.1x-sgd": (under(sgd), CLOSE),
    "s-t20-1.1x-shallow-depthshare-3": (under(muon), CLOSE_BF16),
    "s-t20-1.1x-shallow-depthshare-6": (under(muon), CLOSE_BF16),
    "s-t20-1.1x-shallow-strideshare": (under(muon), CLOSE_BF16),
    "s-t20-1.1x-shampoo": (under(shampoo), CLOSE),
    "s-t20-1.1x-shampoo-whole": (under(shampoo), CLOSE),
    "s-t20-1.1x-wildcard-topcode": (under(muon), CLOSE_BF16),
    "xl-t20-1.1x-ref-optim-fullmuon-fe-wsd": (under(muon), CLOSE_BF16),
    "xl-t20-1.1x-ref-optim-fullmuon-fe-wsd-3884": (under(muon), CLOSE_BF16),
    "xl-t20-1.1x-shallow-depthshare-4": (under(muon), CLOSE_BF16),
}


def test_every_entry_of_the_library_has_a_reference():
    assert sorted(REFERENCES) == [p.stem for p in library.entries()], (
        "an entry of attacks/ needs a reference in tests/test_entries.py's REFERENCES, and a reference an entry")


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_an_entrys_steps_end_at_its_references_weights(name, monkeypatch):
    at_its_depth(name, monkeypatch)
    route, dials = entry(name)
    if group := dials.get("group"):
        monkeypatch.setitem(SHAPE, "d_model", group)
        monkeypatch.setitem(SHAPE, "d_ff", 2 * group)
    reference, close = REFERENCES[name]
    got, want = routed(route, dials), reference(dials)
    torch.manual_seed(0)
    start = LM.of(SHAPE).state_dict()
    moved = [k for k in got if not torch.equal(got[k], start[k])]
    assert moved and set(moved) == {k for k in want if not torch.equal(want[k], start[k])}
    for k in moved:
        move, rounding = (want[k] - start[k]).abs().max(), ROUNDINGS * steps_of(dials) * EPS * want[k].abs().max()
        assert (got[k] - want[k]).abs().max() <= close * move + rounding, k


def test_the_weights_and_their_average_are_each_an_entry():
    """An entry whose output is the average over a horizon has an entry beside it, the same config with `subset`'s
    route and no horizon, whose output is the weights themselves: the two are two runs, a frontier takes the lower,
    and no entry chooses between them inside a run. The two outputs differ, so each reference above holds its own."""
    configs = {p.stem: tomllib.loads(p.read_text()) for p in library.entries()}
    horizons = {name: raw for name, raw in configs.items() if raw.get("dials", {}).get("ema_horizon")}
    assert horizons
    for name, raw in horizons.items():
        beside = name.removesuffix("-ema")
        assert name.endswith("-ema") and beside in configs, (
            f"attacks/{name}.toml sets ema_horizon: name it NAME-ema and write attacks/NAME.toml beside it, the same "
            f"config with route subset and no ema_horizon")
        dials = {k: v for k, v in raw["dials"].items() if k != "ema_horizon"}
        assert configs[beside] == raw | {"route": "subset", "dials": dials}, (name, beside)
        route, filled = entry(name)
        averages, weights = routed(route, filled), routed(subset, filled)
        assert any(not torch.equal(averages[k], weights[k]) for k in weights)
