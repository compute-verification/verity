import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import catalog  # noqa: E402
import library as L  # noqa: E402


def test_library_has_no_problems():
    assert L.problems() == []


def test_every_query_is_consistent():
    """Lower bound at most the construction's cost, and reads within the cap, for every algorithm, model, policy and
    data mode."""
    for a in L.load_json("algorithms"):
        for m in L.load_json("models"):
            for c in catalog.load("calibrations"):
                for data in ("novel", "committed"):
                    try:
                        r = L.query(a["id"], m["id"], c["id"], data)
                    except ValueError:
                        continue
                    assert 1 <= r["lower"] <= r["upper"] * (1 + 1e-9), r
                    assert sum(r["reads"].values()) <= c["K"] * (1 + 1e-9), r


def test_training_data_does_not_matter_at_frontier_scale():
    """When activations must cross blocks, data is a sliver of the reads."""
    r = L.data_importance("backprop", "kimi-k2", "kimi-k2-1m-raw")
    assert r["factor"] == pytest.approx(1.0, abs=0.01)
    assert r["read_shares"]["data"] < 0.01


def test_training_data_is_everything_when_the_model_fits_in_a_block():
    """A model whose state fits trains for free on committed data and pays about N_inf / (3N) on novel data."""
    r = L.data_importance("backprop", "dense-4m", "kimi-k2-1m-raw")
    assert r["upper_committed"] == pytest.approx(1.0)
    assert r["factor"] > 1e3


def test_committed_data_silences_the_theorem():
    assert L.query("backprop", "kimi-k2", "kimi-k2-1m-raw", "committed")["lower"] == 1.0
    assert L.query("backprop", "kimi-k2", "kimi-k2-1m-raw", "novel")["lower"] > 1e4
