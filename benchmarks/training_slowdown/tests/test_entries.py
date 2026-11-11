"""Each entry of the library held to a reference in plain torch: the entry's own dials, read from its config, run for
a few steps at a small shape through its route, and the same steps written here against torch's own optimizers and
its own average of the weights, or against the published rule where torch has none. The two end at the same output
model."""
import fnmatch
import tomllib

import pytest
import torch
import torch.nn.functional as F
from synthetic import SHAPE as TINY
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from training_slowdown import config, library
from training_slowdown.model import LM
from training_slowdown.routes import subset
from training_slowdown.routes._optimizers import newton_schulz

#: six blocks, as the entries' patterns name them, 16 wide
SHAPE = TINY | {"n_layer": 6}
STEPS, BATCH, CTX = 3, 4, 8


def entry(name: str):
    """An entry's route and its dials, the route's defaults filled in, as a job reads them."""
    raw = tomllib.loads((library.ENTRIES / f"{name}.toml").read_text())
    route = config.route(raw["route"])
    return route, config.dials(route, raw.get("dials", {}), name)


def batches() -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    return [torch.randint(0, SHAPE["vocab"], (BATCH, CTX + 1), generator=g) for _ in range(STEPS)]


def routed(route, dials: dict) -> dict:
    """The output model after the route's own steps."""
    torch.manual_seed(0)
    model, optimizers = route.build(SHAPE, dials)
    for t, rows in enumerate(batches()):
        route.step(model, rows, optimizers, dials, t, STEPS)
    return route.output(model, dials) if hasattr(route, "output") else model.state_dict()


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
    (the blocks' matrices; the gains and the embedding) stepped at the scheduled rate. With an `ema_decay`, torch's own
    average of them (`AveragedModel`)."""
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
                g["lr"] = dials["lr"] * subset.schedule(dials, t, STEPS)
            o.step()
            o.zero_grad()
        if average:
            average.update_parameters(model)
    return (average.module if average else model).state_dict()


def under(optimizers):
    return lambda dials: plain(dials, optimizers(dials))


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
                                    torch.optim.AdamW(beside, lr=d["lr"], betas=(0.9, 0.95), weight_decay=0.0)]


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
EPS, ROUNDINGS = torch.finfo(torch.float32).eps, 4 * STEPS
#: each entry's reference: the output model plain torch ends at under the entry's dials, and how close the route's
#: comes to it
REFERENCES = {
    "s-t20-1.1x-accumulate": (under(adamw), CLOSE),
    "s-t20-1.1x-adafactor": (under(adafactor), CLOSE),
    "s-t20-1.1x-adamw": (under(adamw), CLOSE),
    "s-t20-1.1x-ema": (under(adamw), CLOSE),
    "s-t20-1.1x-last-blocks": (under(adamw), CLOSE),
    "s-t20-1.1x-lion": (under(lion), CLOSE),
    "s-t20-1.1x-lora": (lora, CLOSE),
    "s-t20-1.1x-muon-whole": (under(muon), CLOSE_BF16),
    "s-t20-1.1x-ref-optim": (under(muon), CLOSE_BF16),
    "s-t20-1.1x-sgd": (under(sgd), CLOSE),
    "s-t20-1.1x-shallow-strideshare": (under(muon), CLOSE_BF16),
    "s-t20-1.1x-shampoo": (under(shampoo), CLOSE),
    "s-t20-1.1x-shampoo-whole": (under(shampoo), CLOSE),
}


def test_every_entry_of_the_library_has_a_reference():
    assert sorted(REFERENCES) == [p.stem for p in library.entries()], (
        "an entry of attacks/ needs a reference in tests/test_entries.py's REFERENCES, and a reference an entry")


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_an_entrys_steps_end_at_its_references_weights(name):
    route, dials = entry(name)
    reference, close = REFERENCES[name]
    got, want = routed(route, dials), reference(dials)
    torch.manual_seed(0)
    start = LM.of(SHAPE).state_dict()
    moved = [k for k in got if not torch.equal(got[k], start[k])]
    assert moved and set(moved) == {k for k in want if not torch.equal(want[k], start[k])}
    for k in moved:
        move, rounding = (want[k] - start[k]).abs().max(), ROUNDINGS * EPS * want[k].abs().max()
        assert (got[k] - want[k]).abs().max() <= close * move + rounding, k
