"""covert_witness's training-FLOP meter (`prepare.train_flops`) on a small tied-embedding start: an adapter on the tied
embedding is trained from the lookup, so its pass runs the whole backward, and the meter's lowest trained node is the one
`deep_share` finds for the same tensor."""
import importlib.util
from pathlib import Path

import pytest

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
    head, attn = 2 * V * D, 2 * D * (CTX + 1)
    entries = r * (D + V) + r * D                   # its factors at the head, and B[token] · A at the lookup
    forward = blocks + head + 2 * entries + L * attn
    backward = blocks + head + 4 * entries + 2 * L * attn
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
