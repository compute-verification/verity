"""training_slowdown.evaluate: a checkpoint's score on the search pool (a slice from a live coin, paired with θ) and on
the final set (whole)."""
import json
import math

import pytest
import torch
from synthetic import SCORE_REL, tiny_data, tiny_start

from training_slowdown.data import sample_slice, held_out, windows
from training_slowdown.evaluate import Evaluator, window_nats
from training_slowdown.model import LM


@pytest.fixture
def setup(tmp_path):
    data = tiny_data(tmp_path / "data")
    s, theta = tiny_start(tmp_path / "starts")
    g = torch.Generator().manual_seed(1)
    other = {k: v + 0.05 * torch.randn(v.shape, generator=g) for k, v in theta.items()}
    return data, s, theta, other


def near(x):
    """`x` to within what two forward passes may differ by."""
    return pytest.approx(x, rel=SCORE_REL)


def bpb_of(state, shape, toks, starts, nbytes):
    m = LM.of(shape)
    m.load_state_dict(state)
    return float(window_nats(m, toks, starts, "cpu").sum()) / nbytes / math.log(2)


def test_theta_scores_its_own_start_bpb_and_a_coin_replays_its_slice(setup):
    data, s, theta, _ = setup
    ev = Evaluator(data, "search", theta, s.shape, "cpu")
    toks, man = held_out(data, "search")
    full, tail = windows(len(toks))
    assert ev.start_bpb == near(bpb_of(theta, s.shape, toks, full + tail, man["bytes"]))
    sc = ev(theta, "w")
    assert sc["split"] == "search" and len(sc["slice"]["coin"]) == 32
    assert sc["slice_bpb"] == near(sc["start_slice_bpb"])  # the whole pool is the slice here: the same windows, again
    assert sc["bpb"] == near(ev.start_bpb)
    passed = {k: near(sc[k]) for k in ("bpb", "slice_bpb")}  # what the state's pass gives; the other fields are exact
    assert ev(theta, "w", coin=sc["slice"]["coin"]) == sc | passed


def test_a_search_score_is_the_states_bpb_on_the_slice_paired_with_thetas(setup):
    data, s, theta, other = setup
    ev = Evaluator(data, "search", theta, s.shape, "cpu")
    ev.slice_windows = 3
    sc = ev(other, "w")
    toks, _ = held_out(data, "search")
    full, _ = windows(len(toks))
    picked = [full[i] for i in sample_slice(len(full), "w", sc["slice"]["coin"], 3)["windows"]]
    tb = data.token_bytes()
    nbytes = int(sum(tb[toks[w + 1: w + 1025]].sum() for w in picked))
    assert sc["slice"] | {"coin": None} == {"coin": None, "weights_digest": "w", "windows": 3, "bytes": nbytes,
                                            "targets": 3 * 1024}
    assert sc["slice_bpb"] == near(bpb_of(other, s.shape, toks, picked, nbytes))
    assert sc["start_slice_bpb"] == near(bpb_of(theta, s.shape, toks, picked, nbytes))
    assert sc["bpb"] == sc["start_bpb"] + sc["slice_bpb"] - sc["start_slice_bpb"]
    assert sc["bpb"] != near(ev.start_bpb)


def test_the_final_split_scores_the_whole_set(setup):
    data, s, theta, other = setup
    ev = Evaluator(data, "final", theta, s.shape, "cpu")
    toks, man = held_out(data, "final")
    full, tail = windows(len(toks))
    sc = ev(other, "w")
    assert sc["bpb"] == near(bpb_of(other, s.shape, toks, full + tail, man["bytes"]))
    assert (sc["split"], sc["targets"], sc["bytes"], sc["documents"]) == ("final", len(toks) - 1, man["bytes"],
                                                                          man["documents"])
    assert "slice" not in sc
    assert ev(theta, "w")["bpb"] == near(ev.start_bpb) and ev.start_bpb == sc["start_bpb"]


def test_a_state_is_scored_at_the_depth_its_names_give_and_another_architecture_is_refused_by_name(setup):
    """A state of more blocks than θ (a grown model) is scored in the start's architecture at its own depth, and the
    score states it; θ's base stays θ. A state whose blocks don't run 0..N-1, or with a tensor of another shape, one
    the architecture lacks or one missing, is refused by name."""
    data, s, theta, _ = setup
    n = s.shape["n_layer"]
    ev = Evaluator(data, "final", theta, s.shape, "cpu")
    torch.manual_seed(0)
    deeper = LM.of(s.shape | {"n_layer": n + 2}).state_dict()
    toks, man = held_out(data, "final")
    full, tail = windows(len(toks))
    sc = ev(deeper, "w")
    assert sc["blocks"] == n + 2 and sc["bpb"] == near(
        bpb_of(deeper, s.shape | {"n_layer": n + 2}, toks, full + tail, man["bytes"]))
    again = ev(theta, "w")
    assert again["blocks"] == n and again["bpb"] == near(ev.start_bpb) and sc["start_bpb"] == ev.start_bpb
    wq = tuple(theta["blocks.0.wq.weight"].shape)
    for state, why in (
            ({k: v for k, v in deeper.items() if not k.startswith("blocks.1.")},
             r"the state has no block 1, and a model's blocks run 0..N-1 \(it has blocks \[0, 2\]\)"),
            (deeper | {"blocks.0.wq.weight": torch.zeros(2, 2)},
             rf"the state's blocks.0.wq.weight is of shape \(2, 2\) where the start's architecture has \({wq[0]}, "
             rf"{wq[1]}\): a state is the start's shape at {n + 2} blocks"),
            (theta | {"blocks.0.extra": torch.zeros(1)},
             "the state's blocks.0.extra is a tensor the start's architecture lacks"),
            ({k: v for k, v in theta.items() if k != "norm"}, "the state's norm is missing")):
        with pytest.raises(ValueError, match=why):
            ev(state, "w")


def test_a_search_pool_whose_tokens_arent_its_bytes_is_refused(setup):
    data, s, theta, _ = setup
    p = data.sets["search"] / "manifest.json"
    p.write_text(json.dumps(json.loads(p.read_text()) | {"bytes": 1}))
    with pytest.raises(SystemExit, match="documents' bytes"):
        Evaluator(data, "search", theta, s.shape, "cpu")
