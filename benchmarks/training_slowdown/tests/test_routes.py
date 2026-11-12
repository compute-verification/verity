import contextlib
import math
from fnmatch import fnmatchcase

import pytest
import torch
import torch.nn.functional as F
from synthetic import SHAPE, run_job, tiny_registry, write_config
from torch.utils._python_dispatch import TorchDispatchMode

from training_slowdown import config, run, starts
from training_slowdown.codecs import ScaledQuant, SparseTernary
from training_slowdown.data import Data, Stream
from training_slowdown.model import LM
from training_slowdown.routes import ema, grow, qat, subset
from training_slowdown.routes._optimizers import newton_schulz
from training_slowdown.traced import trace_step


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


def test_a_routes_forward_runs_under_bf16_autocast_on_a_cpu():
    """An entry's program is its job's on every device: the step runs the model's forward under bf16 autocast on a
    CPU as on a GPU, so its products are bf16 there too, and the weights stay float32."""
    seen = []

    class Probe(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(8, 8, generator=torch.Generator().manual_seed(0)))

        def forward(self, x):
            out = self.w[x] @ self.w
            seen.append((torch.is_autocast_enabled("cpu"), out.dtype))
            return out

    model = Probe()
    rows = torch.randint(0, 8, (2, 5), generator=torch.Generator().manual_seed(1))
    dials = subset.DIALS | {"lr": 1e-2, "micro_batches": 2}
    subset.step(model, rows, [torch.optim.SGD(model.parameters(), lr=1e-2)], dials, 0, 4)
    assert seen == [(True, torch.bfloat16)] * 2 and model.w.dtype == torch.float32
    assert not torch.is_autocast_enabled("cpu")


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


def test_newton_schulz_of_a_stack_is_the_same_update_in_another_op():
    """`stack`: the matrix normalized by its norm over the last two dimensions kept, as a program that orthogonalizes
    a stack writes it; the update is the same to bf16's rounding."""
    g = torch.randn(8, 16, generator=torch.Generator().manual_seed(0))
    u, stacked = newton_schulz(g), newton_schulz(g, stack=True)
    assert torch.equal(newton_schulz(g.T, stack=True), stacked.T)
    assert (stacked - u).abs().max() <= 0.02 * u.abs().max()


def test_grows_blocks_copy_the_starts_in_order_with_each_repeats_output_projections_zero():
    """8 blocks grown to 12: block j copies the start's ⌊8j/12⌋, and blocks 1, 4, 7 and 10, each the second copy of
    one, start with `wo` and `w_down` zero; the rest of each repeat, and every tensor outside the blocks, is the
    start's. Without `zero` each repeat is a plain copy. A depth that grows nothing is refused, and `build` refuses
    it too."""
    dials = grow.DIALS | {"depth": 12}
    shape = SHAPE | {"n_layer": 8}
    assert grow.sources(shape, dials) == [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7]
    origin = grow.origin(shape, dials)
    assert sorted({int(k.split(".")[1]) for k, v in origin.items() if v is None}) == [1, 4, 7, 10]
    assert all(origin[f"blocks.{j}.{t}"] is None for j in (1, 4, 7, 10) for t in grow.ZEROED)
    assert origin["blocks.1.wq.weight"] == "blocks.0.wq.weight" and origin["blocks.11.w_down.weight"] == (
        "blocks.7.w_down.weight")
    assert "blocks.0.wq.weight" not in origin and all(k.startswith("blocks.") for k in origin)
    plain = grow.origin(shape, dials | {"zero": False})
    assert None not in plain.values() and plain["blocks.1.wo.weight"] == "blocks.0.wo.weight"
    for depth in (8, 4):
        with pytest.raises(ValueError, match=f"depth {depth} doesn't grow the start's 8 blocks"):
            grow.origin(shape, dials | {"depth": depth})
    with pytest.raises(ValueError, match="depth is the grown model's blocks"):
        grow.plan(grow.DIALS)
    with pytest.raises(ValueError, match="doesn't grow"):
        grow.build(SHAPE, dials | {"depth": SHAPE["n_layer"]})


def test_grow_trains_its_blocks_matrices_under_muon_and_its_gains_and_embedding_under_adamw():
    """The grown model's every matrix of the blocks under Muon at `lr`, decayed and normalized as a stack; the gains at
    `adam_lr` and the embedding at `embed_lr` under one undecayed fused AdamW, a group each, the embedding frozen
    where its rate is 0."""
    dials = grow.DIALS | {"depth": 3, "lr": 0.01, "adam_lr": 0.002, "embed_lr": 0.003, "weight_decay": 0.05}
    torch.manual_seed(0)
    model, (muon, adamw) = grow.build(SHAPE, dials)
    assert len(model.blocks) == 3 and muon.defaults["stack"] and muon.defaults["weight_decay"] == 0.05
    gains = ["norm"] + [f"blocks.{j}.{n}" for j in range(3) for n in ("attn_norm", "mlp_norm")]
    matrices = [f"blocks.{j}.{w}.weight" for j in range(3) for w in ("wq", "wk", "wv", "wo", "w_gate", "w_up",
                                                                     "w_down")]
    assert [(r, sorted(n)) for r, n in grouped(model, muon)] == [(0.01, sorted(matrices))]
    assert [(r, sorted(n)) for r, n in grouped(model, adamw)] == [(0.002, sorted(gains)), (0.003, ["embed"])]
    assert adamw.defaults["fused"] is True and adamw.defaults["weight_decay"] == 0.0
    model, (_, adamw) = grow.build(SHAPE, dials | {"embed_lr": 0.0})
    assert not model.embed.requires_grad and [r for r, _ in grouped(model, adamw)] == [0.002]


def test_the_grown_model_gives_the_starts_logits_bit_for_bit_at_its_first_step():
    """With `zero`, each repeated block adds zero to the residual stream: its attention's and its MLP's output
    projections are zero, so each adds an exact zero, and the stream reaching the next block is the start's. The grown
    model's start, built from θ by `origin` as the harness builds it, gives θ's logits bit for bit, in float32 and
    under bf16 autocast. Without `zero` it doesn't."""
    torch.manual_seed(0)
    theta = LM.of(SHAPE)
    rows = torch.randint(0, SHAPE["vocab"], (2, 9), generator=torch.Generator().manual_seed(1))
    for depth in (SHAPE["n_layer"] + 1, 2 * SHAPE["n_layer"] + 1):
        for zero in (True, False):
            dials = grow.DIALS | {"depth": depth, "zero": zero}
            model, _ = grow.build(SHAPE, dials)
            model.load_state_dict(run.start_state(theta.state_dict(), model.state_dict(), grow.origin(SHAPE, dials)))
            with torch.no_grad():
                same = torch.equal(model(rows), theta(rows))
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    same_bf16 = torch.equal(model(rows), theta(rows))
            assert same == same_bf16 == zero, (depth, zero)


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


class Kernel(TorchDispatchMode):
    """The fused AdamW kernel's calls, each as its tensors' shapes."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if "_fused_adamw_" in str(func):
            self.calls.append([tuple(p.shape) for p in args[0]])
        return func(*args, **(kwargs or {}))


def test_fused_steps_adamw_in_torchs_kernel_once_a_kind_to_the_same_weights():
    """`fused`: AdamW steps in torch's fused kernel, called once a group over the group's tensors, where the default
    never calls it; and its groups are one a kind of tensor, so two kinds at one rate are two calls, where without
    the dial they are one group. The weights agree to rounding. An optimizer that is no AdamW has no such kernel."""
    rows = torch.randint(0, SHAPE["vocab"], (2, 9), generator=torch.Generator().manual_seed(1))

    def stepped(**over):
        dials = subset.DIALS | {"trainable": "*", "opt": "muon", "lr": 0.01} | over
        torch.manual_seed(0)
        model, optimizers = subset.build(SHAPE, dials)
        with Kernel() as kernel:
            for t in range(2):
                subset.step(model, rows, optimizers, dials, t, 2)
        return kernel.calls, model.state_dict(), optimizers[-1]

    embed, gain = (SHAPE["vocab"], SHAPE["d_model"]), (SHAPE["d_model"],)
    def rated(optimizer) -> list:
        return [(g["lr"], g["initial_lr"], len(g["params"])) for g in optimizer.param_groups]

    calls, plain, adamw = stepped()
    assert calls == [] and adamw.defaults["fused"] is None, "a false dial states nothing, so torch's default stands"
    assert rated(adamw) == [(0.005, 0.01, 4)], "without the dial a group is a rate: the embedding with the gains"
    calls, fused, adamw = stepped(fused=True)
    assert adamw.defaults["fused"] is True and calls == [[embed], [gain, gain, gain]] * 2
    assert rated(adamw) == [(0.005, 0.01, 1), (0.005, 0.01, 3)]
    torch.manual_seed(0)
    start = LM.of(SHAPE).state_dict()
    for k, v in fused.items():
        assert not torch.equal(v, start[k]) and torch.allclose(v, plain[k], rtol=0, atol=1e-6), k
    assert stepped(fused=True, embed_lr=0.003)[0] == [[embed], [gain, gain, gain]] * 2
    assert stepped(fused=True, trainable="blocks.*,norm")[0] == [[gain, gain, gain]] * 2
    whole, _, adamw = stepped(fused=True, opt="adamw")
    assert adamw.defaults["fused"] is True and [len(call) for call in whole] == [1, len(GAINS), len(BLOCK)] * 2
    with pytest.raises(ValueError, match="fused is AdamW's kernel, and opt 'sgd' steps no AdamW"):
        subset.plan(subset.DIALS | {"opt": "sgd", "fused": True})


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


def test_three_rates_under_one_optimizer_are_torchs_three_groups(monkeypatch):
    """AdamW over the whole model with a rate for the gains and one for the embedding, on a cosine schedule, against
    torch's AdamW over three groups stepped at those rates: the same weights. What is held is the update's rule, so
    the route's forward runs in float32 as the reference's does: torch's autocast is a no-op while the test runs."""
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())
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


def ternary(v: torch.Tensor, keep: float) -> torch.Tensor:
    """Sparse ternary rounding by hand: a latent's entries at least as large as its ceil(keep · n)-th largest
    magnitude, and not zero, each keep their sign, at their mean magnitude as f16 holds it; the rest are 0."""
    flat = v.detach().flatten().double()
    kept = (flat.abs() >= flat.abs().topk(math.ceil(keep * flat.numel())).values[-1]) & (flat != 0)
    scale = flat[kept].abs().mean().float().half().double() if kept.any() else 0.0
    return (flat.sign() * kept * scale).view_as(v).float()


def test_qats_forward_runs_under_bf16_autocast_on_a_cpu():
    """Route `qat` has a step of its own, and it casts as `subset`'s does: the model's forward runs under bf16
    autocast on a CPU as on a GPU, and the latents and the start's copies stay float32."""
    dials = qat.DIALS | {"targets": "blocks.0.wq.weight", "lr": 1e-2, "micro_batches": 2}
    torch.manual_seed(0)
    model, optimizers = qat.build(SHAPE, dials)
    seen = []
    model.register_forward_hook(lambda module, args, out: seen.append((torch.is_autocast_enabled("cpu"), out.dtype)))
    rows = torch.randint(0, SHAPE["vocab"], (2, 9), generator=torch.Generator().manual_seed(1))
    qat.step(model, rows, optimizers, dials, 0, 4)
    assert seen == [(True, torch.bfloat16)] * 2 and not torch.is_autocast_enabled("cpu")
    assert all(layer.v.dtype == layer.start.dtype == torch.float32 for layer in qat.coded(model, dials).values())


def test_a_qat_step_reads_the_start_plus_its_codes_and_steps_the_latents_straight_through():
    """Route `qat` against its steps written by hand, at a first step and at one in the cooldown. The weights a
    step's forward reads are the start plus the rounding of the latents as the step found them (the start itself at
    the first step, whose latents are zero). Each latent then moves by Muon's rule on the gradient of the loss at
    those weights, taken as the latent's own, clipped over all the latents, at the scheduled rate: here with the
    gradient from a forward of the start's own model on the weights built by hand, the momentum and the scaled step
    written out, and Newton-Schulz the route's (held to the polar factor above). What the step leaves, and `output`
    hands over, is the rounding of the latents it wrote; every other weight is the start's, and the output has no
    latent. A latent is compared within Newton-Schulz's bf16 rounding, a rounding to within f16's last bit of the
    scale: a kept entry more or fewer is a whole scale off. A kept state's load, which holds the latents, leaves each
    layer's copy of the start the very tensor it held, which the counter names; θ's load makes the copy anew."""
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
    held = {k: layer.start for k, layer in layers.items()}
    model.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
    assert all(layer.start is held[k] and torch.equal(layer.start, theta[k]) for k, layer in layers.items())
    model.load_state_dict(theta)
    assert all(layer.start is not held[k] and torch.equal(layer.start, theta[k]) for k, layer in layers.items())


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
    return te.finish({}), outputs, model, theta, trace_step(cfg, route, model, optimizers, "cpu")


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


def test_a_qat_runs_checkpoints_are_the_start_plus_the_codes_it_trained(tmp_path):
    """Route `qat` through the harness: θ loads into a model that holds latents and codes θ has none of, every op is
    counted and priced, and each checkpoint's output is what the library's codec rebuilds on θ from the codes the
    route handed over: the weights the route's next step would train on, bit for bit, and θ everywhere else. E is
    those codes, at most two bits an entry, and the checkpoint states the digest of the model the route trains, its
    latents among it (`trained_sha256`). The count takes the coded matrices' products as updated from the second
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
        assert cp["trained_sha256"] != cp["output_sha256"] and 8 * cp["e_bytes"] <= 2 * entries + 8 * 600
    assert [cp["step"] for cp in rec["checkpoints"]] == [0, 1, 2]
    assert rec["checkpoints"][-1]["trained_sha256"] == run.theta_sha256(model.state_dict())
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
    first, second = (rec["steps"][t]["by"].get("fwd/updated", 0) for t in (0, 1))
    assert first == 0 and second == trace["tokens"] * entries   # the first step's codes are zero: its forward reads θ
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
