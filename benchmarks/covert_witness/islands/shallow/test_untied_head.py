"""The untied-head extension (coordinator 06:10Z ruling, for TRN-40's head-only share): at its initialisation an untied
lf model computes the tied start's function exactly, E as CWMODL1 carries only `head`, and positions place `head` at the
head (layer L + 1) and `embed` at the lookup alone, so a head-only update is shallow; the meter prices its step as the
final norm's plus the head's weight gradient. Run from benchmarks/covert_witness: `python -m pytest islands/shallow/test_untied_head.py`."""
import pytest
import torch

import prepare
from model import Size


def lf(tied: bool = True) -> dict:
    return {**Size("t", 32, 2, 2, 64).config(vocab=320), "tied": tied}


def theta_of(config: dict) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    return {k: (0.02 * torch.randn(s, generator=g)).float() for k, s in prepare.architecture(config).items()}


def test_an_untied_model_at_its_initialisation_computes_the_tied_starts_function_exactly():
    tied, untied = lf(), lf(tied=False)
    th = theta_of(tied)
    assert set(prepare.architecture(untied)) == set(th) | {"head"}
    blob = prepare.encode_model(untied, {"head": th["embed"].clone()})
    rec = {"config": tied, "tokenizer": prepare.data_v3.TOKENIZER}
    assert prepare.model_config(rec, blob) == untied
    out = prepare.decode(th, blob, device="cpu")
    with torch.device("cpu"):
        a, b = prepare.Net(tied).load(th), prepare.Net(untied).load(out)
    x = torch.randint(0, 320, (2, 16), generator=torch.Generator().manual_seed(2))
    assert torch.equal(a.logits(x), b.logits(x))
    with torch.no_grad():
        b.params()["head"].add_(1.0)
    assert not torch.equal(a.logits(x), b.logits(x))


def test_untying_is_the_only_change_of_tying_a_model_format_may_make():
    untied_start = {"config": lf(tied=False)}
    with pytest.raises(ValueError, match="θ's family, vocab, ctx, tied"):
        prepare.model_config(untied_start, prepare.encode_model(lf(), {}))
    assert prepare.model_config(untied_start, prepare.encode_model(lf(tied=False), {})) == lf(tied=False)


def test_an_untied_head_is_layer_l_plus_1_and_a_head_only_step_adds_its_weight_gradient_to_the_norms():
    untied = lf(tied=False)
    rec = prepare.model_record({"config": lf(), "tokenizer": prepare.data_v3.TOKENIZER}, untied, theta_of(untied))
    L = untied["n_layer"]
    by = {n["param"]: n for n in prepare.positions(rec)}
    assert by["embed"]["layer"] == 0 and by["embed"]["flops"] == 0
    assert by["head"]["layer"] == L + 1 and by["head"]["flops"] == 2 * 320 * 32
    ds = prepare.deep_share(rec, {"head"})
    assert ds["lowest_updated_layer"] == L + 1 and ds["phi"] == 0
    step = [{"op": "train", "tokens": 1}]
    fwd = prepare.train_flops(rec, {"full": []}, [{"op": "forward", "tokens": 1}])["flops"]
    head_only = prepare.train_flops(rec, {"full": ["head"]}, step)["flops"]
    norm_only = prepare.train_flops(rec, {"full": ["norm"]}, step)["flops"]
    assert head_only == norm_only + 2 * 320 * 32 == fwd + 2 * 2 * 320 * 32
