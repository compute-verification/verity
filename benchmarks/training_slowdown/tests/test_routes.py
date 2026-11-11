import pytest
import torch
from synthetic import SHAPE, run_job

from training_slowdown import config, run, starts
from training_slowdown.codecs import ScaledQuant
from training_slowdown.routes import subset
from training_slowdown.routes._optimizers import newton_schulz


def test_subsets_schedule_ramps_holds_then_falls_to_zero():
    linear = subset.DIALS
    assert [subset.schedule(linear, t, 4) for t in range(4)] == [1.0, 0.75, 0.5, 0.25]
    held = linear | {"warmup": 50, "decay": 0.4}
    assert [subset.schedule(held, t, 1400) for t in (0, 24, 49, 500, 840)] == [1 / 50, 0.5, 1.0, 1.0, 1.0]
    assert [subset.schedule(held, t, 1400) for t in (1120, 1399)] == pytest.approx([0.5, 1 / 1400 / 0.4])
    share = linear | {"warmup": 0.02, "decay": 0.4}
    assert subset.schedule(share, 0, 838) == 1 / (0.02 * 838) and subset.schedule(share, 16, 838) == 1.0


def test_every_module_of_routes_passes_the_screen_a_route_does():
    """A route's load screens the route's own source; what it imports from beside it is held to the same names."""
    for path in sorted(config.ROUTES.glob("*.py")):
        assert run.screen(path.read_text()) == [], path.name


def test_newton_schulz_brings_a_matrixs_singular_values_near_one():
    g = torch.randn(8, 16, generator=torch.Generator().manual_seed(0))
    u = newton_schulz(g)
    assert u.dtype == g.dtype and torch.equal(newton_schulz(g.T), u.T)
    s = torch.linalg.svdvals(u)
    assert 0.6 < s.min() and s.max() < 1.3
    a, _, b = torch.linalg.svd(g, full_matrices=False)
    assert (u - a @ b).norm() / (a @ b).norm() < 0.3


def test_muon_takes_the_matrices_and_adamw_the_gains():
    dials = subset.DIALS | {"trainable": "blocks.0.*,norm", "opt": "muon", "lr": 0.01, "weight_decay": 0.05}
    torch.manual_seed(0)
    model, (muon, adamw) = subset.build(SHAPE, dials)
    names = {p: n for n, p in model.named_parameters()}
    assert sorted(names[p] for p in muon.param_groups[0]["params"]) == sorted(
        f"blocks.0.{w}.weight" for w in ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"))
    assert sorted(names[p] for g in adamw.param_groups for p in g["params"]) == [
        "blocks.0.attn_norm", "blocks.0.mlp_norm", "norm"]
    assert isinstance(adamw, torch.optim.AdamW) and adamw.defaults["weight_decay"] == 0.0
    assert not model.embed.requires_grad
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    rows = torch.randint(0, SHAPE["vocab"], (2, 9), generator=torch.Generator().manual_seed(1))
    w = model.blocks[0].w_up.weight    # 32 x 16: taller than wide
    seen = {}
    w.register_hook(lambda g: seen.setdefault("grad", g.clone()))
    subset.step(model, rows, [muon, adamw], dials | {"clip": 0.0}, 0, 1)
    after = dict(model.named_parameters())
    assert all(torch.equal(before[n], p) != p.requires_grad for n, p in after.items())
    # the first step's buffer is the gradient, so the orthogonalized direction is of (1 + momentum) times it
    lr = 0.01 * subset.schedule(dials, 0, 1)
    want = before["blocks.0.w_up.weight"] * (1 - lr * 0.05) - lr * 2 ** 0.5 * newton_schulz(1.95 * seen["grad"])
    assert torch.allclose(w, want, rtol=0, atol=1e-7)
    with pytest.raises(SystemExit, match="opt muon has no rule for the embedding"):
        subset.build(SHAPE, dials | {"trainable": "*"})


@pytest.mark.slow
def test_a_muon_job_with_coded_deltas_is_counted_traced_and_scored_on_what_its_e_rebuilds(tmp_path):
    """A tiny start through the whole job at the dials of the tree's configs: two micro-batches a step, Muon and
    AdamW, a ramp and a held rate, E in 8-bit codes. Every op is counted and the step traces; each checkpoint's
    output is the model E rebuilds from θ, not the trained weights, and that model is what was scored. The unit
    tests above run no job: only this shows the counter prices Muon's products and the capture takes the step."""
    dials = {"trainable": "blocks.0.*,norm", "opt": "muon", "lr": 1e-2, "weight_decay": 0.05, "warmup": 0.5,
             "decay": 0.4, "micro_batches": 2, "store": "q8"}
    rec, cfg, reg, out = run_job(tmp_path, dials=dials, budget={"steps": 3, "batch": 4})
    assert rec["invalid"] == [] and rec["route"]["version"] == subset.VERSION == 2
    tr = rec["trace"]
    assert "refused" not in tr and tr["tokens"] == rec["steps"][0]["tokens"] == 4096 and tr["rows"]
    assert len(tr["updated"]) >= 7
    theta = starts.load(cfg.start, reg.parent / "starts")
    plan = config.route(cfg.route).plan(cfg.dials)
    assert all(isinstance(c, ScaledQuant) and c.bits == 8 for _, c in plan.rules)
    for cp in rec["checkpoints"]:
        assert (cp["format"], cp["changed"]) == ("codecs", 10) and cp["trained_sha256"] != cp["output_sha256"]
        assert cp["score"]["slice"]["weights_digest"] == cp["output_sha256"]
    last = rec["checkpoints"][-1]
    enc = (out / last["e_path"]).read_bytes()
    assert len(enc) == last["e_bytes"] and len(plan.keys(enc)) == 10
    rebuilt = theta | plan.decode(enc, {k: theta[k] for k in plan.keys(enc)})
    assert run.theta_sha256(rebuilt) == last["output_sha256"]
