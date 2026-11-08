import figure1 as f1
import figure6 as f
import pytest

ADVICE = {"machine_windows": {f1.PRIMARY: 2.0e12, f1.SECONDARY: 6.0e8},
          "arms": [{"label": "Without the tick", "tick": False, "bits": 5000, "pending": "prefix caching on"},
                   {"label": "With the tick", "tick": True, "bits": 0}]}


def _point(load, rate, tick, ttft):
    return {"point": load, "load": load, "rate": rate, "tick": tick, "ttft_p50_s": ttft, "ttft_p99_s": 2 * ttft,
            "itl_p50_s": .03, "output_tok_per_s": 1000.0, "goodput_ttft_req_per_s": rate}


POINTS = [_point("p60", 1.0, None, .4), _point("p60", 1.0, {"sat_ms": 180, "free_ms": 220}, .7),
          _point("p60", 1.0, {"sat_ms": 90, "free_ms": 110}, .5), _point("p100", 1.67, None, .6)]


@pytest.fixture(scope="module")
def out():
    return f.run("GLM-5.3", ADVICE, POINTS)


def test_an_arms_term_is_its_bits_a_window_times_the_scopes_windows(out):
    no, tick = out["arms"]
    assert no["by_scope"][f1.PRIMARY] == 1e16 and no["by_scope"][f1.SECONDARY] == 3e12
    assert tick["by_scope"][f1.PRIMARY] == 0
    assert out["target"]["label"].startswith("GLM-5.3's weights at FP4")


def test_the_sweep_groups_by_load_with_its_no_tick_point_and_ticks_by_period(out):
    p100, p60 = out["sweep"]
    assert p100["load"] == "p100" and not p100["tick"]
    assert p60["none"]["ttft_p50_s"] == .4 and [r["free_ms"] for r in p60["tick"]] == [110, 220]


def test_with_the_tick_a_request_joins_the_first_tick_after_its_bucket_closes():
    sc = f.schematic()
    for r in sc["requests"]:
        assert r["continuous_ms"] >= r["arrival_ms"] and r["tick_ms"] >= r["bucket_closes_ms"] > r["arrival_ms"]
        assert r["tick_ms"] - r["bucket_closes_ms"] < sc["tick_ms"]


def test_the_figure_renders_with_and_without_the_sweep(out, tmp_path):
    text = f.caption(out)
    assert "0 at every scope" in text and "(pending, hatched: prefix caching on)" in text
    assert "The run with the real network certifier is pending" in text and "pool model" not in text
    assert [p.name for p in f.render(out, tmp_path)] == ["figure6.png", "figure6.svg"]
    empty = f.run("GLM-5.3", ADVICE, [])
    assert "(b) Pending" in f.caption(empty)
    assert [p.name for p in f.render(empty, tmp_path / "empty")] == ["figure6.png", "figure6.svg"]
