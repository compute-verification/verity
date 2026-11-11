import math
from fnmatch import fnmatchcase

import pytest
import torch
import torch.nn.functional as F
from synthetic import SHAPE, run_job, tiny_registry, write_config

from training_slowdown import config, run, starts
from training_slowdown.codecs import ScaledQuant
from training_slowdown.data import Data, Stream
from training_slowdown.model import LM
from training_slowdown.routes import ema, subset
from training_slowdown.routes._optimizers import newton_schulz


def test_subsets_schedule_ramps_holds_then_falls_to_zero():
    linear = subset.DIALS
    assert [subset.schedule(linear, t, 4) for t in range(4)] == [1.0, 0.75, 0.5, 0.25]
    held = linear | {"warmup": 50, "decay": 0.4}
    assert [subset.schedule(held, t, 1400) for t in (0, 24, 49, 500, 840)] == [1 / 50, 0.5, 1.0, 1.0, 1.0]
    assert [subset.schedule(held, t, 1400) for t in (1120, 1399)] == pytest.approx([0.5, 1 / 1400 / 0.4])
    share = linear | {"warmup": 0.02, "decay": 0.4}
    assert subset.schedule(share, 0, 838) == 1 / (0.02 * 838) and subset.schedule(share, 16, 838) == 1.0


def test_subsets_schedule_after_the_ramp_is_linear_half_a_cosine_or_constant():
    cosine = subset.DIALS | {"schedule": "cosine"}
    assert [subset.schedule(cosine, t, 4) for t in range(4)] == pytest.approx(
        [1.0, (1 + 0.5 ** 0.5) / 2, 0.5, (1 - 0.5 ** 0.5) / 2])
    assert subset.schedule(cosine | {"warmup": 2}, 0, 4) == 0.5
    constant = subset.DIALS | {"schedule": "constant", "warmup": 2, "decay": 0.5}
    assert [subset.schedule(constant, t, 4) for t in range(4)] == [0.5, 1.0, 1.0, 1.0]
    lora = config.route("lora").DIALS
    assert "schedule" not in lora and [subset.schedule(lora, t, 4) for t in range(4)] == [1.0, 0.75, 0.5, 0.25]


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


GAINS = ["norm", "blocks.0.attn_norm", "blocks.0.mlp_norm"]
BLOCK = [f"blocks.0.{w}.weight" for w in ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")]


def grouped(model, optimizer) -> list:
    """Each of the optimizer's groups: its rate and its tensors' names."""
    names = {p: n for n, p in model.named_parameters()}
    return [(g["lr"], [names[p] for p in g["params"]]) for g in optimizer.param_groups]


@pytest.mark.parametrize("opt", sorted(subset.MATRICES))
def test_the_embedding_trains_under_adamw_beside_the_matrices_optimizer_or_under_it(opt):
    """The whole model under Muon or Shampoo: the block's matrices under it at `lr`, the embedding and the gains under
    one undecayed AdamW at their own rates, or the embedding among the optimizer's matrices where the config says so.
    Every tensor moves, and a later step scales each group's own rate, not the last step's."""
    dials = subset.DIALS | {"trainable": "*", "opt": opt, "lr": 0.01, "weight_decay": 0.05, "gain_lr": 0.002,
                            "embed_lr": 0.003}
    torch.manual_seed(0)
    model, (matrices, adamw) = subset.build(SHAPE, dials)
    assert isinstance(matrices, subset.MATRICES[opt]) and grouped(model, matrices) == [(0.01, BLOCK)]
    assert isinstance(adamw, torch.optim.AdamW) and adamw.defaults["weight_decay"] == 0.0
    assert grouped(model, adamw) == [(0.003, ["embed"]), (0.002, GAINS)]
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    rows = torch.randint(0, SHAPE["vocab"], (2, 9), generator=torch.Generator().manual_seed(1))
    for t in (1, 2):
        subset.step(model, rows, [matrices, adamw], dials, t, 4)
        factor = subset.schedule(dials, t, 4)
        assert grouped(model, matrices) == [(0.01 * factor, BLOCK)]
        assert grouped(model, adamw) == [(0.003 * factor, ["embed"]), (0.002 * factor, GAINS)]
    assert not any(torch.equal(before[n], p) for n, p in model.named_parameters())
    model, (matrices, adamw) = subset.build(SHAPE, dials | {"embed": "opt"})
    assert grouped(model, matrices) == [(0.003, ["embed"]), (0.01, BLOCK)] and grouped(model, adamw) == [(0.002, GAINS)]
    model, (matrices, adamw) = subset.build(SHAPE, dials | {"gain_lr": 0.0, "embed_lr": 0.0})
    assert grouped(model, matrices) == [(0.01, BLOCK)] and grouped(model, adamw) == [(0.01, ["embed", *GAINS])]


def test_a_loaded_state_steps_at_its_groups_own_rates_and_one_from_before_them_at_lr():
    """A continued job loads its optimizers' states, and a loaded state's groups replace the built ones. A state this
    route wrote states each group's own rate, so the next step is at it. A state from before a group had a rate of
    its own states only the last step's scaled rate, when every group was at `lr`: the next step is at `lr` times
    the schedule, not at that scaled rate scaled again."""
    rows = torch.randint(0, SHAPE["vocab"], (2, 9), generator=torch.Generator().manual_seed(1))

    def continued(dials: dict, before_own_rates: bool) -> list[float]:
        torch.manual_seed(0)
        model, (opt,) = subset.build(SHAPE, dials)
        subset.step(model, rows, [opt], dials, 0, 4)
        state = opt.state_dict()
        assert [g["lr"] for g in state["param_groups"]] != [g["initial_lr"] for g in state["param_groups"]]
        if before_own_rates:
            for g in state["param_groups"]:
                del g["initial_lr"]
        model, (opt,) = subset.build(SHAPE, dials)
        opt.load_state_dict(state)
        subset.step(model, rows, [opt], dials, 1, 4)
        return [g["lr"] for g in opt.param_groups]

    dials = subset.DIALS | {"lr": 0.01, "warmup": 2}
    factor = subset.schedule(dials, 1, 4)
    assert factor not in (1.0, subset.schedule(dials, 0, 4))
    assert continued(dials, before_own_rates=True) == [0.01 * factor]
    assert continued(dials, before_own_rates=False) == [0.01 * factor]
    assert continued(dials | {"gain_lr": 0.002}, before_own_rates=False) == [0.01 * factor, 0.002 * factor]


def test_three_rates_under_one_optimizer_are_torchs_three_groups():
    """AdamW over the whole model with a rate for the gains and one for the embedding, on a cosine schedule, against
    torch's AdamW over three groups stepped at those rates: the same weights."""
    dials = subset.DIALS | {"lr": 0.01, "gain_lr": 0.002, "embed_lr": 0.003, "weight_decay": 0.05, "warmup": 2,
                            "schedule": "cosine"}
    torch.manual_seed(0)
    model, (opt,) = subset.build(SHAPE, dials)
    assert grouped(model, opt) == [(0.003, ["embed"]), (0.002, GAINS), (0.01, BLOCK)]
    torch.manual_seed(0)
    plain = LM.of(SHAPE)
    by_name = dict(plain.named_parameters())
    rates = {0.003: ["embed"], 0.002: GAINS, 0.01: BLOCK}
    theirs = torch.optim.AdamW([{"params": [by_name[n] for n in names], "lr": rate} for rate, names in rates.items()],
                               betas=(0.9, 0.95), weight_decay=0.05)
    g = torch.Generator().manual_seed(1)
    for t in range(3):
        rows = torch.randint(0, SHAPE["vocab"], (2, 9), generator=g)
        subset.step(model, rows, [opt], dials, t, 3)
        F.cross_entropy(plain(rows[:, :-1]).flatten(0, 1), rows[:, 1:].flatten()).backward()
        torch.nn.utils.clip_grad_norm_(list(plain.parameters()), dials["clip"])
        for group, rate in zip(theirs.param_groups, rates):
            group["lr"] = rate * min(1.0, (t + 1) / 2) * 0.5 * (1 + math.cos(math.pi * t / 3))
        theirs.step()
        theirs.zero_grad()
    torch.manual_seed(0)
    start = LM.of(SHAPE).state_dict()
    for n, p in model.named_parameters():
        assert not torch.equal(p, start[n]) and torch.allclose(p, by_name[n], rtol=0, atol=1e-6), n


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


def test_an_averages_horizon_is_a_share_of_the_steps_so_far():
    """`ema_horizon`: the average is the weights themselves until that share of the steps is a whole step, then keeps
    more of itself each step; over every step so far it is the mean of the weights the steps wrote."""
    tenth = ema.DIALS | {"ema_horizon": 0.1}
    assert [ema.keeps(tenth, t) for t in (0, 9, 10, 19, 99)] == pytest.approx([0, 0, 1 - 1 / 1.1, 0.5, 0.9])
    assert [ema.keeps(ema.DIALS | {"ema_decay": 0.9}, t) for t in (1, 99)] == [0.9, 0.9]
    g = torch.Generator().manual_seed(1)
    batches = [torch.randint(0, SHAPE["vocab"], (2, 9), generator=g) for _ in range(6)]
    for horizon, raw in ((1.0, 1), (0.25, 4)):
        dials = ema.DIALS | {"trainable": "blocks.0.w_*", "lr": 1e-2, "ema_horizon": horizon}
        torch.manual_seed(0)
        model, optimizers = ema.build(SHAPE, dials)
        names, wrote = subset.trained(model, dials), []
        for t, rows in enumerate(batches):
            ema.step(model, rows, optimizers, dials, t, len(batches))
            wrote.append({n: model.get_parameter(n).detach().clone() for n in names})
            out = ema.output(model, dials)
            assert all(torch.equal(out[n], wrote[t][n]) == (t < raw) for n in names), (horizon, t)
        if horizon == 1.0:
            for n in names:
                assert torch.allclose(out[n], sum(w[n] for w in wrote) / len(wrote), rtol=0, atol=1e-7)


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
