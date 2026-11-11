"""Each entry of the library held to a reference in plain torch: the entry's own dials, read from its config, run for
a few steps at a small shape through its route, and the same steps written here against torch's own optimizers, or
against the published rule where torch has none. The two end at the same weights."""
import fnmatch
import tomllib

import pytest
import torch
import torch.nn.functional as F
from synthetic import SHAPE as TINY

from training_slowdown import config, library
from training_slowdown.model import LM
from training_slowdown.routes import subset

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
    """The weights after the route's own steps."""
    torch.manual_seed(0)
    model, optimizers = route.build(SHAPE, dials)
    for t, rows in enumerate(batches()):
        route.step(model, rows, optimizers, dials, t, STEPS)
    return model.state_dict()


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
    forward and backward over the whole batch, the gradient clipped, each optimizer of `optimizers(matrices, gains)`
    stepped at the scheduled rate."""
    torch.manual_seed(0)
    model = LM.of(SHAPE)
    patterns = [p.strip() for p in dials["trainable"].split(",")]
    for n, p in model.named_parameters():
        p.requires_grad_(any(fnmatch.fnmatchcase(n, pat) for pat in patterns))
    params = [p for p in model.parameters() if p.requires_grad]
    opts = optimizers([p for p in params if p.ndim == 2], [p for p in params if p.ndim != 2])
    for t, rows in enumerate(batches()):
        logits = model(rows[:, :-1])
        F.cross_entropy(logits.flatten(0, 1), rows[:, 1:].flatten()).backward()
        torch.nn.utils.clip_grad_norm_(params, dials["clip"])
        for o in opts:
            for g in o.param_groups:
                g["lr"] = dials["lr"] * subset.schedule(dials, t, STEPS)
            o.step()
            o.zero_grad()
    return model.state_dict()


def adamw(d: dict):
    return lambda matrices, gains: [torch.optim.AdamW(matrices + gains, lr=d["lr"], betas=(0.9, 0.95),
                                                      weight_decay=d["weight_decay"])]


def muon_rule(p, g, state, group):
    """Muon: Nesterov momentum 0.95, the update orthogonalized (the route's Newton-Schulz, which test_routes holds
    to the SVD's polar factor) and scaled by the matrix's shape, the decay decoupled."""
    buf = state["buf"] = 0.95 * state.get("buf", torch.zeros_like(p)) + g
    u = subset.newton_schulz(g + 0.95 * buf)
    p.mul_(1 - group["lr"] * group["weight_decay"]).sub_(group["lr"] * max(1.0, p.shape[0] / p.shape[1]) ** 0.5 * u)


def muon(d: dict):
    return lambda matrices, gains: [Rule(matrices, muon_rule, lr=d["lr"], weight_decay=d["weight_decay"]),
                                    torch.optim.AdamW(gains, lr=d["lr"], betas=(0.9, 0.95), weight_decay=0.0)]


#: the most a route's weight may differ from its reference's, as a share of the tensor's largest move from the start:
#: float rounding in another order of the same sums
CLOSE = 1e-3
#: the same where the update passes Newton-Schulz in bf16, which turns a rounding in its input into a few percent of
#: its output; a rule with another momentum, scale or decay is off by the whole move
CLOSE_BF16 = 0.1
#: each entry's reference: the optimizers plain torch steps with at the entry's dials, and how close the route's
#: weights come to its own
REFERENCES = {
    "s-t20-1.1x-accumulate": (adamw, CLOSE),
    "s-t20-1.1x-adamw": (adamw, CLOSE),
    "s-t20-1.1x-ref-optim": (muon, CLOSE_BF16),
    "s-t20-1.1x-shallow-strideshare": (muon, CLOSE_BF16),
}


def test_every_entry_of_the_library_has_a_reference():
    assert sorted(REFERENCES) == [p.stem for p in library.entries()], (
        "an entry of attacks/ needs a reference in tests/test_entries.py's REFERENCES, and a reference an entry")


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_an_entrys_steps_end_at_its_references_weights(name):
    route, dials = entry(name)
    optimizers, close = REFERENCES[name]
    got, want = routed(route, dials), plain(dials, optimizers(dials))
    torch.manual_seed(0)
    start = LM.of(SHAPE).state_dict()
    moved = [k for k in got if not torch.equal(got[k], start[k])]
    assert moved and set(moved) == {k for k in want if not torch.equal(want[k], start[k])}
    for k in moved:
        move = (want[k] - start[k]).abs().max()
        assert (got[k] - want[k]).abs().max() <= close * move, k
