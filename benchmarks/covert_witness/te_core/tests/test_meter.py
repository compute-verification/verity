"""covert_witness's training-FLOP meter (`prepare.train_flops`) on a small tied-embedding start: an adapter on the tied
embedding is trained from the lookup, so its pass runs the whole backward, and the meter's lowest trained node is the one
`deep_share` finds for the same tensor. On an lf model, and at v3's starts' shapes, the meter is the harness's own
count (`te_core.flops`) of a training step's matmuls and attention, so a planner that sizes its steps by it and the
`flop_budget` check read one count."""
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils import flop_counter

from model import SIZES, Size
from te_core.flops import Counter

_spec = importlib.util.spec_from_file_location("cw_prepare_meter", Path(__file__).resolve().parents[2] / "prepare.py")
prepare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare)

L, D, V, I, CTX = 2, 8, 32, 16, 64
BLOCK = {"attn_norm": (D,), "wq": (D, D), "wk": (D, D), "wv": (D, D), "wo": (D, D), "mlp_norm": (D,),
         "w_gate": (I, D), "w_up": (I, D), "w_down": (D, I)}
REC = {"config": {"n_layer": L, "family": "lf", "d_model": D},
       "tensors": {"embed": ("float32", (V, D)), "norm": ("float32", (D,)),
                   **{f"blocks.{i}.{k}": ("float32", s) for i in range(L) for k, s in BLOCK.items()}}}
MATRICES = [n for n, (_, s) in REC["tensors"].items() if len(s) == 2]


def per_token(trainable: dict) -> dict:
    return prepare.train_flops(REC, trainable, [{"op": "train", "tokens": 1, "ctx": CTX}])


def test_a_tied_embedding_adapter_runs_the_whole_backward():
    r = 4
    blocks = L * sum(2 * s[0] * s[1] for s in BLOCK.values() if len(s) == 2)
    head = 2 * V * D
    entries = r * (D + V) + r * D                   # its factors at the head, and B[token] · A at the lookup
    # the harness's attention: the whole CTX x CTX square in two products forward, flash attention's five backward
    forward = blocks + head + 2 * entries + L * 2 * 2 * D * CTX
    backward = blocks + head + 4 * entries + L * 2 * 5 * D * CTX
    m = per_token({"adapters": {"embed": r}})
    assert m["lowest_trained"] == [0, 0]
    assert m["flops"] == forward + backward
    assert m["flops"] > per_token({"adapters": {"blocks.0.wq": r}})["flops"]


@pytest.mark.parametrize("name", MATRICES)
def test_an_adapters_lowest_trained_layer_is_deep_shares(name):
    assert per_token({"adapters": {name: 1}})["lowest_trained"][0] == prepare.deep_share(REC, {name})["lowest_updated_layer"]


@pytest.mark.parametrize("name", list(REC["tensors"]))
def test_a_full_tensors_lowest_trained_layer_is_deep_shares(name):
    assert per_token({"full": [name]})["lowest_trained"][0] == prepare.deep_share(REC, {name})["lowest_updated_layer"]


@pytest.mark.parametrize("name", ["norm", "blocks.1.mlp_norm"])
def test_an_adapter_on_a_gain_is_refused(name):
    with pytest.raises(ValueError, match="an adapter needs a matrix"):
        per_token({"adapters": {name: 2}})


@pytest.fixture
def gpu_flash(monkeypatch):
    """The CPU flash attention kernel priced as te_core.flops prices the GPUs' (torch's registry has no formula for the
    CPU one before 2.15, and from 2.15 it is the same one)."""
    aten = torch.ops.aten
    for cpu, gpu in ((aten._scaled_dot_product_flash_attention_for_cpu, aten._scaled_dot_product_flash_attention),
                     (aten._scaled_dot_product_flash_attention_for_cpu_backward,
                      aten._scaled_dot_product_flash_attention_backward)):
        monkeypatch.setitem(flop_counter.flop_registry, cpu, flop_counter.flop_registry[gpu])


def counted(config: dict, trained: list[str], ctx: int) -> tuple[int, int]:
    """One `ctx`-token training step of the lf model `config` with `trained` its trainable tensors: 2x the matmul MACs
    (weight matmuls and attention) te_core.flops counts on CPU under flash attention, the GPUs' kernel (`gpu_flash`),
    with no op it can't price, and `train_flops`' figure for the same step."""
    shapes = prepare.architecture(config)
    g = torch.Generator().manual_seed(1)
    theta = {k: torch.randn(s, generator=g) * 0.02 for k, s in shapes.items()}
    net = prepare.Net(config).load({k: v.clone() for k, v in theta.items()})
    state = net.inner.state_dict(keep_vars=True)
    for k, p in state.items():
        p.requires_grad_(k in trained)
    c = Counter(theta, digests=False)
    c.register(state)
    idx = torch.randint(0, config["vocab"], (1, ctx), generator=g)
    with c, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        c.begin_step()
        F.cross_entropy(net.logits(idx)[0, :-1].float(), idx[0, 1:]).backward()
        c.end_step()
    assert not c.unknown_ops and not c.foreign_ops
    rec = {"config": config, "tensors": {k: ["torch.float32", list(s)] for k, s in shapes.items()}}
    meter = prepare.train_flops(rec, {"full": trained}, [{"op": "train", "tokens": ctx, "ctx": ctx}])
    return 2 * c.steps[0].matmul_macs, meter["flops"]


TINY = SIZES["tiny"].config(vocab=256)
TINY_TENSORS = sorted(prepare.architecture(TINY))


@pytest.mark.parametrize("trained", [[n] for n in TINY_TENSORS]
                         + [TINY_TENSORS, ["blocks.1.wv.weight", "blocks.0.w_down.weight"]],
                         ids=lambda t: "all" if len(t) > 2 else "+".join(t))
def test_the_meter_is_the_harness_count_of_a_training_step(trained, gpu_flash):
    """Wherever the lowest trained tensor sits: no input gradient at its own position, and attention's backward in each
    block at or above it."""
    harness, meter = counted(TINY, trained, 64)
    assert harness == meter


def v3_shapes() -> dict[str, dict]:
    """v3's starts' lf configs, one per shape, in two blocks (a block's count doesn't depend on how many there are),
    with the MLP width their parameter count gives and 64-wide heads (the count doesn't depend on the split)."""
    out = {}
    for name, e in prepare.STORE.items():
        sh = e["shape"]
        if sh.get("tokenizer") != "bpe32k":
            continue
        d, n, v = sh["d_model"], sh["n_layer"], sh["vocab"]
        per_block = (sh["parameters"] - v * d - d) / n
        d_ff = (per_block - 4 * d * d - 2 * d) / (3 * d)
        assert d_ff == int(d_ff), f"{name}: no lf MLP width gives its {sh['parameters']} parameters"
        config = Size(name, d, 2, d // 64, int(d_ff)).config(vocab=v) | {"ctx": sh["ctx"]}
        if config | {"name": ""} not in [c | {"name": ""} for c in out.values()]:
            out[name] = config
    return out


@pytest.mark.slow
@pytest.mark.parametrize("name", list(v3_shapes()))
def test_the_meter_is_the_harness_count_on_v3s_starts(name, gpu_flash):
    """At each v3 start's width, MLP, vocabulary and its whole context, every tensor trained: the coefficients the tiny
    model's test can't check at a start's own ratio of context to width, where the harness metered 14 to 16% over the
    causal-half attention `train_flops` once priced."""
    config = v3_shapes()[name]
    harness, meter = counted(config, sorted(prepare.architecture(config)), config["ctx"])
    assert harness == meter
