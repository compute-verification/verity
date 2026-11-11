import math
from fnmatch import fnmatchcase

import pytest
import torch
import torch.nn.functional as F
from synthetic import SHAPE, run_job, tiny_registry, write_config

from training_slowdown import config, run, starts
from training_slowdown.codecs import ScaledQuant, SparseTernary
from training_slowdown.data import Data, Stream
from training_slowdown.model import LM
from training_slowdown.routes import qat, subset
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


def ternary(v: torch.Tensor, keep: float) -> torch.Tensor:
    """Sparse ternary rounding by hand: a latent's entries at least as large as its ceil(keep · n)-th largest
    magnitude, and not zero, each keep their sign, at their mean magnitude as f16 holds it; the rest are 0."""
    flat = v.detach().flatten().double()
    kept = (flat.abs() >= flat.abs().topk(math.ceil(keep * flat.numel())).values[-1]) & (flat != 0)
    scale = flat[kept].abs().mean().float().half().double() if kept.any() else 0.0
    return (flat.sign() * kept * scale).view_as(v).float()


def test_a_qat_step_reads_the_start_plus_its_codes_and_steps_the_latents_straight_through():
    """Route `qat` against its steps written by hand, at a first step and at one in the cooldown. The weights a
    step's forward reads are the start plus the rounding of the latents as the step found them (the start itself at
    the first step, whose latents are zero). Each latent then moves by Muon's rule on the gradient of the loss at
    those weights, taken as the latent's own, clipped over all the latents, at the scheduled rate: here with the
    gradient from a forward of the start's own model on the weights built by hand, the momentum and the scaled step
    written out, and Newton-Schulz the route's (held to the polar factor above). What the step leaves, and `output`
    hands over, is the rounding of the latents it wrote; every other weight is the start's, and the output has no
    latent. A latent is compared within Newton-Schulz's bf16 rounding, a rounding to within f16's last bit of the
    scale: a kept entry more or fewer is a whole scale off."""
    targets = ("blocks.0.wq.weight", "blocks.0.w_gate.weight", "blocks.0.w_down.weight")
    dials = qat.DIALS | {"targets": ",".join(targets), "keep": 0.3, "lr": 0.02, "warmup": 4.0, "decay": 0.5,
                         "cooldown": "sqrt", "clip": 0.01}
    torch.manual_seed(0)
    model, (opt,) = qat.build(SHAPE, dials)
    theta = {k: v.clone() for k, v in model.state_dict().items() if not k.endswith(".v")}
    layers = qat.coded(model, dials)
    assert sorted(layers) == sorted(targets) and all(not layer.v.any() for layer in layers.values())
    gen, n = torch.Generator().manual_seed(1), 8

    def rounded(v, k):
        """How far off a weight may be from the start plus `v`'s rounding by hand."""
        return ternary(v, dials["keep"]), 2.0 ** -10 * ternary(v, dials["keep"]).abs().max() + 1e-7 * theta[k].abs().max()

    for t, rate in ((0, 1 / 4), (6, 1 - math.sqrt(0.5))):
        rows = torch.randint(0, SHAPE["vocab"], (3, 9), generator=gen)
        before = {k: layer.v.detach().clone() for k, layer in layers.items()}
        momentum = {k: opt.state[layer.v]["buf"].clone() if opt.state[layer.v] else torch.zeros_like(layer.v)
                    for k, layer in layers.items()}
        qat.step(model, rows, [opt], dials, t, n)
        weights = {}
        for k, layer in layers.items():
            q, off = rounded(before[k], k)
            assert (layer.weight - (theta[k] + q)).abs().max() <= off, (t, k)
            assert t > 0 or torch.equal(layer.weight, theta[k])
            weights[k] = (theta[k] + q).requires_grad_()
        logits = torch.func.functional_call(LM.of(SHAPE), theta | weights, (rows[:, :-1],))
        F.cross_entropy(logits.flatten(0, 1), rows[:, 1:].flatten()).backward()
        norm = torch.sqrt(sum((w.grad ** 2).sum() for w in weights.values()))
        assert norm > dials["clip"]   # the clip binds
        for k, layer in layers.items():
            g = weights[k].grad * (dials["clip"] / norm)
            buf = 0.95 * momentum[k] + g
            out, into = g.shape
            want = before[k] - dials["lr"] * rate * max(1.0, out / into) ** 0.5 * newton_schulz(g + 0.95 * buf)
            move = (want - before[k]).abs().max()
            assert move > 0 and (layer.v - want).abs().max() <= 0.1 * move, (t, k)
            assert layer.v.grad is None and layer.weight.grad is None
    out = qat.output(model, dials)
    assert set(out) == set(theta)
    for k in theta:
        if k in layers:
            q, off = rounded(layers[k].v, k)
            assert q.count_nonzero() >= math.ceil(0.3 * q.numel())   # more where magnitudes tie at the least kept
            assert (SparseTernary.apply(theta[k], *out[k]) - (theta[k] + q)).abs().max() <= off
        else:
            assert torch.equal(out[k], theta[k])


def test_qats_rate_falls_linearly_or_as_one_less_a_square_root():
    linear = qat.DIALS | {"warmup": 50.0, "decay": 0.4}
    falls = [qat.schedule(linear, t, 1400) for t in (0, 24, 49, 840, 1120)]
    assert falls == [subset.schedule(linear, t, 1400) for t in (0, 24, 49, 840, 1120)]
    assert falls == pytest.approx([1 / 50, 0.5, 1, 1, 0.5])
    root = linear | {"cooldown": "sqrt"}
    assert [qat.schedule(root, t, 1400) for t in (0, 49, 840, 1120, 1399)] == pytest.approx([
        1 / 50, 1, 1, 1 - math.sqrt(0.5), 1 - math.sqrt(1 - 1 / 1400 / 0.4)])
    with pytest.raises(ValueError, match="cooldown 'cosine' is none of linear, sqrt"):
        qat.plan(root | {"cooldown": "cosine"})
    with pytest.raises(ValueError, match="keep is a share"):
        qat.plan(root | {"keep": 0.0})
    with pytest.raises(SystemExit, match="names no matrix of the start"):
        qat.build(SHAPE, root | {"targets": "norm"})
    with pytest.raises(SystemExit, match="groups of 7 don't divide blocks.0.wq.weight's 256 entries"):
        qat.build(SHAPE, root | {"group": 7})


def loop(tmp_path, **over):
    """A route's three steps through the harness's own loop, in this process: the record, each checkpoint's output
    model as the evaluator is handed it, the route's model after the run, θ and the step's trace."""
    reg = tiny_registry(tmp_path / "reg")
    registry = starts.registry(reg)
    cfg = config.load(write_config(tmp_path / "a.toml", budget={"steps": 3, "batch": 4}, **over), registry)
    theta = starts.load(cfg.start, reg.parent / "starts")
    outputs = []
    te = run.Harness(theta, tmp_path / "out", start=cfg.start.entry(), clock=lambda: 0.0, evaluator=lambda state, _: (
        outputs.append({k: v.clone() for k, v in state.items()}) or {"bpb": 0.0}))
    route = config.route(cfg.route)
    torch.manual_seed(0)
    with te.counter:
        model, optimizers = route.build(cfg.start.shape, cfg.dials)
        stream = Stream(Data.of(cfg.start.data, registry), 0, ctx=int(cfg.start.shape["ctx"]))
        run.train(te, cfg, route, stream, model, optimizers)
    return te.finish({}), outputs, model, theta, run.trace_step(cfg, route, model, optimizers, "cpu")


def test_an_ema_runs_checkpoints_are_the_average_of_the_weights_it_wrote(tmp_path):
    """Route `ema` through the harness: the average's ops are counted and priced and the step traces, and each
    checkpoint's output is the average, half of the one before it and half of the weights the step wrote, where the
    harness would hand out the trained weights. The reference test steps the route itself: only this shows a job
    scores and encodes the average."""
    rec, outputs, model, theta, trace = loop(tmp_path, route="ema", dials={
        "trainable": "blocks.0.w_*", "lr": 1e-2, "ema_decay": 0.5})
    assert rec["invalid"] == [] and "refused" not in trace
    assert [(cp["step"], cp["changed"]) for cp in rec["checkpoints"]] == [(0, 3), (1, 3), (2, 3)]
    trained = model.state_dict()
    assert set(outputs[-1]) == set(theta) == set(trained)
    for k, v in outputs[-1].items():
        if fnmatchcase(k, "blocks.0.w_*"):
            assert torch.allclose(v, 0.5 * outputs[1][k] + 0.5 * trained[k], rtol=0, atol=1e-7)
            assert not torch.equal(v, trained[k])
        else:
            assert torch.equal(v, theta[k])


def test_a_lora_runs_checkpoints_are_the_start_with_its_adapters_merged(tmp_path):
    """Route `lora` through the harness: θ loads into a model that holds adapters θ has none of, every op is counted
    and priced and the step traces, the start's own weights never move, and each checkpoint's output has θ's names,
    the two adapted matrices each with its B·A merged in. The merge is a product of the trace."""
    targets = ("blocks.0.wq.weight", "blocks.0.w_up.weight")
    rec, outputs, model, theta, trace = loop(tmp_path, route="lora", dials={
        "targets": ",".join(targets), "rank": 2, "alpha": 4.0, "lr": 1e-2})
    assert rec["invalid"] == [] and "refused" not in trace
    assert [(cp["step"], cp["changed"]) for cp in rec["checkpoints"]] == [(0, 2), (1, 2), (2, 2)]
    state = model.state_dict()
    assert set(outputs[-1]) == set(theta) == {k for k in state if not k.endswith((".a", ".b"))}
    assert all(torch.equal(state[k], theta[k]) for k in theta)
    for k, v in outputs[-1].items():
        if k in targets:
            a, b = state[k.replace(".weight", ".a")], state[k.replace(".weight", ".b")]
            assert torch.allclose(v, theta[k] + 2.0 * b @ a, rtol=0, atol=1e-7) and not torch.equal(v, theta[k])
        else:
            assert torch.equal(v, theta[k])
    merges = [r for r in trace["rows"] if r["phase"] == "forward" and not r["token_dims"]]
    assert sum(r["count"] for r in merges) == 2 and all(r["dims"]["k"] == 2 and r["affected"] == [True, True]
                                                        for r in merges)


def test_a_qat_runs_checkpoints_are_the_start_plus_the_codes_it_trained(tmp_path):
    """Route `qat` through the harness: θ loads into a model that holds latents and codes θ has none of, every op is
    counted and priced, and each checkpoint's output is what the library's codec rebuilds on θ from the codes the
    route handed over: the weights the route's next step would train on, bit for bit, and θ everywhere else. E is
    those codes, at most two bits an entry. The count takes the coded matrices' products as updated from the second
    step on. The step traces, and what one step hands the next forward is the codes:
    the update's cut is two bits an entry and a 16-bit scale a matrix, where the latents and Muon's momentum, 32
    bits an entry each, are carried state no token op reads."""
    targets = ("blocks.0.wq.weight", "blocks.0.w_up.weight")
    rec, outputs, model, theta, trace = loop(tmp_path, route="qat", dials={
        "targets": ",".join(targets), "keep": 0.5, "lr": 1e-2})
    assert rec["invalid"] == [] and "refused" not in trace
    entries = sum(theta[k].numel() for k in targets)
    for cp in rec["checkpoints"]:
        assert (cp["format"], cp["changed"], cp["parts"], cp["rebuild_macs"]) == ("codecs", 2, 2, 2 * entries)
        assert "trained_sha256" not in cp and 8 * cp["e_bytes"] <= 2 * entries + 8 * 600
    assert [cp["step"] for cp in rec["checkpoints"]] == [0, 1, 2]
    layers = qat.coded(model, config.dials(qat, {"targets": ",".join(targets)}, "test"))
    assert set(outputs[-1]) == set(theta) and sorted(layers) == sorted(targets)
    for k, v in outputs[-1].items():
        if k in targets:
            layer = layers[k]
            assert torch.equal(v, SparseTernary.apply(theta[k], layer.scales, layer.plus, layer.minus))
            assert torch.equal(layer.start, theta[k]) and not torch.equal(v, theta[k]) and v.dtype == theta[k].dtype
            assert not torch.equal(v, layer.weight)   # the route's own weight is the last step's, a step behind
            kept = (v != theta[k]).sum().item()
            assert kept == int(layer.plus.sum() + layer.minus.sum()) >= math.ceil(0.5 * v.numel())
        else:
            assert torch.equal(v, theta[k])
    first, second = (rec["steps"][t]["by"]["fwd/updated"] for t in (0, 1))
    assert second - first == trace["tokens"] * entries   # the first step's codes are zero: its forward reads θ
    c = trace["circuit"]
    assert c["update_cut_bits"] == c["u1_bits"] == 2 * entries + 16 * len(targets)
    assert c["state_bits"] == 2 * 32 * entries and c["grad_cut_bits"] == 32 * entries
    assert not trace["p_match"] and not trace["data_dependent"] and not trace["carried_scalars"]


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
