import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "probes" / "routed"))

import analysis as R  # noqa: E402

ALPHA, BETA = R.EXPONENTS["besiroglu"]
A_COEF, B_COEF, E = 0.2 * 1e9 ** ALPHA, 0.1 * 1e12 ** BETA, 0.45


def law_bpb(N: float, D: float = math.inf) -> float:
    return E + A_COEF * N ** -ALPHA + (B_COEF * D ** -BETA if math.isfinite(D) else 0.0)


def test_windows_cut_at_character_boundaries_and_drop_a_short_tail():
    text = "aé€😀" * 700
    ws = R.windows("d", "s", text, size=1000, prefix=100)
    data = text.encode()
    assert len(ws) == len(data) // 1000
    joined = "".join(w["text"] for w in ws)
    assert text.startswith(joined) and len(data) - len(joined.encode()) < 500
    for w in ws:
        assert 997 <= len(w["text"].encode()) <= 1000
        assert 97 <= len(w["text"][: w["prefix_chars"]].encode()) <= 100
    assert [w["id"] for w in ws[:2]] == ["d:0", "d:1"]


def _rec(w, prefix, span, nbytes=100):
    return {"window": w, "stratum": "s", "prefix_bits": prefix, "span_bits": span, "span_bytes": nbytes}


def test_the_oracle_pays_for_naming_its_member_and_the_prefix_router_pays_nothing():
    members = {"m1": {"w0": _rec("w0", 10, 50), "w1": _rec("w1", 30, 90)},
               "m2": {"w0": _rec("w0", 20, 70), "w1": _rec("w1", 20, 60)},
               "m3": {"w0": _rec("w0", 40, 80), "w1": _rec("w1", 50, 95)}}
    oracle = R.compose(members, "oracle")
    assert {w: r["choice"] for w, r in oracle.items()} == {"w0": "m1", "w1": "m2"}
    assert oracle["w1"]["span_bits"] == pytest.approx(60 + math.log2(3))
    prefix = R.compose(members, "prefix")
    assert {w: r["choice"] for w, r in prefix.items()} == {"w0": "m1", "w1": "m2"}
    assert prefix["w1"]["span_bits"] == 60
    members["m3"]["w1"] = _rec("w1", 50, 95, nbytes=101)
    with pytest.raises(ValueError, match="share a tokenizer"):
        R.compose(members, "oracle")


def test_the_ladder_fit_recovers_the_law_and_inverts_it():
    pts = [(N, D, law_bpb(N, D)) for N in (1e9, 3e9, 7e9, 2e10) for D in (1e12, 4e12)]
    law = R.fit_ladder(pts, "besiroglu")
    assert (law["E"], law["A"], law["B"]) == pytest.approx((E, A_COEF, B_COEF), rel=1e-6)
    assert law["data_term"] and law["anneal_offset"] == 0.0
    assert R.dense_size(law, law_bpb(5e10)) == pytest.approx(5e10, rel=1e-6)
    assert R.dense_size(law, law_bpb(5e10, 2e12), 2e12) == pytest.approx(5e10, rel=1e-6)
    assert R.dense_size(law, E - 0.01) == math.inf
    assert R.gains(law, law_bpb(8e9), 2e9, 1e12)["g"] == pytest.approx(4.0, rel=1e-6)


def test_free_exponents_are_found_on_the_grid_and_need_three_token_counts():
    a, b = 0.42, 0.27
    pts = [(N, D, 0.5 + 300 * N ** -a + 900 * D ** -b) for N in (1e9, 3e9, 7e9, 2e10) for D in (1e12, 2e12, 4e12)]
    law = R.fit_ladder(pts, "free")
    assert (law["alpha"], law["beta"]) == (a, b) and law["rms"] < 1e-9
    with pytest.raises(ValueError, match="three"):
        R.fit_ladder([p for p in pts if p[1] != 2e12], "free")


def test_unannealed_checkpoints_get_their_own_offset():
    pts = [(N, D, law_bpb(N, D)) for N, D in ((1e9, 4e12), (7e9, 4e12), (2e10, 5e12))]
    pts += [(N, D, law_bpb(N, D) + 0.03, True) for N, D in ((1e9, 1e12), (7e9, 1e12), (7e9, 2e12), (2e10, 2e12))]
    law = R.fit_ladder(pts, "besiroglu")
    assert law["anneal_offset"] == pytest.approx(0.03, rel=1e-6)
    assert (law["E"], law["A"], law["B"]) == pytest.approx((E, A_COEF, B_COEF), rel=1e-6)


def test_a_ladder_at_one_token_count_has_no_data_term():
    law = R.fit_ladder([(N, 36e12, law_bpb(N, 36e12)) for N in (1e9, 4e9, 1.5e10)], "besiroglu")
    assert not law["data_term"] and law["B"] == 0.0
    assert law["E"] == pytest.approx(E + B_COEF * 36e12 ** -BETA, rel=1e-6)
    assert not R.fit_ladder([(1e9, 1e12, 1.0), (2e9, 2e12, 0.9)], "besiroglu", data_term=False)["data_term"]


def test_d_at_gain_is_the_pages_table():
    assert R.d_at_gain(1, 6, 2.5)["D"] == pytest.approx(217, rel=2e-3)
    assert R.d_at_gain(2, 6, 2.5)["D"] == pytest.approx(112.8, rel=2e-3)
    assert R.d_at_gain(8, 6, 16)["D"] == pytest.approx(1208, rel=2e-3)
    assert R.d_at_gain(4, 6, 2.5)["void"] and R.d_at_gain(2, 13, 2.5)["void"]
    assert R.gain_at(100, 6, 2.5) == pytest.approx(2.15, abs=0.01)
    assert R.gain_at(100, 13, 2.5) == 0.0


@pytest.mark.parametrize("a, b1, call", [
    ((2.3, 3.0), (0.9, 1.1), "drop"),
    ((1.0, 2.5), (2.2, 2.6), "hedge"),
    ((0.8, 1.1), (0.9, 1.2), "keep-margin"),
    ((1.0, 1.5), (1.0, 1.4), "keep-run-trend"),
    (None, (0.9, 1.1), "incomplete"),
    (None, (2.2, 2.6), "hedge"),
])
def test_the_verdict_follows_the_pages_rule(a, b1, call):
    assert R.verdict(a, b1)["call"] == call


def _write(path: Path, bits_per_byte: list[float], prefix: list[float] | None = None):
    with open(path, "w") as f:
        for i, b in enumerate(bits_per_byte):
            f.write(json.dumps(_rec(f"w{i}", prefix[i] if prefix else 0.0, b * 1000, 1000)) + "\n")


def test_analyze_reads_a_known_gain_back(tmp_path):
    """A single configuration at 2B weights scoring what the law gives a dense 8B with unlimited data has g = 4, and an
    oracle over two members, each good on half the windows, picks each on its half."""
    n = 200
    noise = [0.02 * math.sin(7.0 * i) for i in range(n)]
    (tmp_path / "scores").mkdir()
    ladder = {}
    for N in (1e9, 2e9, 4e9, 1.6e10):
        for D in (1e12, 4e12):
            name = f"d{N:g}-{D:g}"
            ladder[name] = {"N": N, "D": D}
            _write(tmp_path / "scores" / f"{name}.jsonl", [law_bpb(N, D) + e for e in noise])
    _write(tmp_path / "scores" / "mix.jsonl", [law_bpb(8e9) + e for e in noise])
    half = [i < n // 2 for i in range(n)]
    _write(tmp_path / "scores" / "m1.jsonl", [law_bpb(8e9) + (0 if h else 0.1) + e for h, e in zip(half, noise)])
    _write(tmp_path / "scores" / "m2.jsonl", [law_bpb(8e9) + (0.1 if h else 0) + e for h, e in zip(half, noise)])
    manifest = {"studies": {"s": {
        "scores": [*ladder, "mix", "m1", "m2"], "ladder": ladder,
        "compositions": {"pair": {"members": ["m1", "m2"], "routing": "oracle", "P": 2e9}},
        "singles": {"mix": {"P": 2e9, "D_train": 1e12, "decides": "A"}}}}}
    out = R.analyze(manifest, tmp_path, resamples=50, seed=1)
    s = out["studies"]["s"]
    assert s["windows"] == n and s["ladder_error"] is None
    mix = s["configs"]["mix"]
    assert mix["gains"]["besiroglu"]["g"] == pytest.approx(4.0, rel=1e-6)
    lo, hi = mix["gains"]["besiroglu"]["g_interval"]
    assert lo <= 4.0 <= hi
    assert mix["gains"]["besiroglu"]["g_same"] > 4.0
    assert mix["D_at_g"]["6 GB, b=2.5"]["void"]
    assert s["configs"]["pair"]["choices"] == {"m1": n // 2, "m2": n // 2}
    assert out["verdict"]["A"] == mix["g_interval"]
    assert out["verdict"]["call"] == ("drop" if mix["g_interval"][0] >= out["verdict"]["claim_line"] else "incomplete")
