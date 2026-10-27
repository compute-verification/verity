"""The PoUS average-memory explorer's data (`benchmarks/pous/explorer.py`, `pous-explorer/v1`): `Π₂`'s call time, the
Poisson quantile and its table, the sampling term against its optimum, the honest curves, and the file's vectors
reproduced from the file alone."""

from __future__ import annotations

import importlib.util
import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("verity.core.protocols.pous")

ROOT = Path(__file__).resolve().parents[3]


def _explorer():
    spec = importlib.util.spec_from_file_location("pous_explorer", ROOT / "benchmarks" / "pous" / "explorer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _explorer()


@pytest.fixture(scope="module")
def doc():
    return json.loads(json.dumps(E.build(generated_at="2026-10-02T00:00:00Z", commit="test"), allow_nan=False))


def test_call_ns_is_pi2s_and_the_threshold_sits_under_the_cliff():
    assert [E.call_ns(b) for b in E.BLOCK_BYTES] == [241_500, 542_500, 1_446_000]
    assert all(E.call_ns(b) == 10 * ((b + 34) // 136 + 1) * 50 for b in E.BLOCK_BYTES)
    assert E.tau_us(1, 241_500, 1) == pytest.approx(482.999, abs=1e-9)
    assert E.tau_us(0, 1_446_000, 2) == pytest.approx(722.999, abs=1e-9)


def test_a_cliff_past_the_deepest_eps_proof_is_audited_at_that_depth(doc):
    base = {"model": "measured", "block_bytes": 65_536, "m": 1, "lambda_per_s": 100, "T_s": 600, "alpha": 0.01,
            "beta": 0.01}
    assert [E.rounds(doc, d) for d in (0, 5, 6, 12)] == [0, 5, 5, 5]
    assert E.evaluate(doc, {**base, "d": 12}) == E.evaluate(doc, {**base, "d": 5})
    assert E.evaluate(doc, {**base, "d": 12})["tau_us"] == pytest.approx(6 * 241.5 - 0.001, abs=1e-9)


def _cdf_tail(f: int, mu: float) -> float:
    return 1 - sum(math.exp(-mu) * mu**k / math.factorial(k) for k in range(f + 1))


def test_poisson_quantile_at_mean_zero_small_means_and_1e5():
    for a in E.ALPHAS:
        assert E.poisson_quantile(0.0, a) == 0
    assert E.poisson_quantile(0.01, 0.01) == 0 and E.poisson_quantile(0.0101, 0.01) == 1   # 1 − e^−μ against α
    for mu in (0.3, 2.0, 7.5, 40.0):
        for a in E.ALPHAS:
            f = E.poisson_quantile(mu, a)
            assert _cdf_tail(f, mu) <= a + 1e-15 and (f == 0 or _cdf_tail(f - 1, mu) > a)
            assert E.poisson_tail(f, mu) == pytest.approx(_cdf_tail(f, mu), rel=1e-9, abs=1e-15)
    mu = 1e5                                          # e^−μ underflows: the tail is summed from its largest term
    for a, z in zip(E.ALPHAS, (2.326, 3.090, 4.753)):
        f = E.poisson_quantile(mu, a)
        assert E.poisson_tail(f, mu) <= a < E.poisson_tail(f - 1, mu)
        assert abs(f - (mu + z * math.sqrt(mu))) < 0.02 * math.sqrt(mu) + 5
        assert f <= E.bernstein_f0(mu, a)


def test_the_f0_table_is_the_exact_quantile(doc):
    for key, a in zip(doc["f0_max_mean"], E.ALPHAS):
        assert key == str(a)
        t = doc["f0_max_mean"][key]
        assert len(t) == 1000 and all(x < y for x, y in pairwise(t))
        assert t[0] == pytest.approx(-math.log1p(-a), rel=1e-12)
        for f in (0, 1, 7, 100, 999):
            assert E.poisson_tail(f, t[f]) <= a < E.poisson_tail(f, t[f] * (1 + 1e-9))
        for mu in (1e-4, 0.05, 0.9, 3.3, 17.0, 250.0, t[-1]):
            assert E.f0_from_table(mu, a, t) == E.poisson_quantile(mu, a)
        for mu in (t[-1] * 1.0001, 5e3, 1e5):            # past the table: Bernstein's, never under the quantile
            assert E.f0_from_table(mu, a, t) == E.bernstein_f0(mu, a) >= E.poisson_quantile(mu, a)


def test_the_sampling_term_bounds_the_optimum_closely():
    for beta in E.BETAS:
        lg, n = math.log(1 / beta), 6e4
        assert E.g_opt(0, n, beta) == E.sampling(0, n, beta) == lg / n         # θ → ∞
        assert E.g_theta(60.0, 0, n, beta) == pytest.approx(lg / n, rel=1e-12)
        assert E.g_theta(5.0, 0, n, beta) > E.g_theta(10.0, 0, n, beta) > lg / n
        for f in (1, 2, 10, 110, 1000, 10**5):
            opt = E.g_opt(f, n, beta)
            assert opt <= min(E.g_theta(t, f, n, beta) for t in (0.01, 0.1, 0.5, 1, 2, 5))
            assert opt <= E.sampling(f, n, beta) <= opt * 1.0015


def test_honest_curves_fall_and_longer_blocks_are_later(doc):
    assert [m["id"] for m in doc["honest_models"]] == ["hardened", "measured", "rack", "datacenter", "heavy-tail"]
    for m in doc["honest_models"]:
        by_block = {c["block_bytes"]: c for c in m["curves"]}
        assert sorted(by_block) == list(E.BLOCK_BYTES)
        for c in m["curves"]:
            t, s = c["t_us"], c["survival_fraction"]
            assert len(t) == 161 and t[0] == 10.0 and t[-1] == pytest.approx(1e5) and t[40] == pytest.approx(100.0)
            assert all(1.0 >= x >= y >= E.S_FLOOR for x, y in pairwise(s)), (m["id"], c["block_bytes"])
        assert all(a <= b <= c for a, b, c in zip(*(by_block[b]["survival_fraction"] for b in E.BLOCK_BYTES)))
    measured = next(m for m in doc["honest_models"] if m["id"] == "measured")["curves"][0]
    at = lambda t: E.survival_at(measured["t_us"], measured["survival_fraction"], t)
    assert at(5.0) == 1.0 and at(150) == pytest.approx(62 / 99_900, rel=0.03)
    assert at(1000) == pytest.approx((1 / 99_900) * (1000 / 750) ** -1.8, rel=0.01)


def test_the_presets_reproduce_the_notes_first_cut(doc):
    """The explorer note's §2 survival table and §4 bounds (λ 100/s, T 600 s, α β 1%), to their printed precision."""
    s = {m["id"]: m["curves"][0] for m in doc["honest_models"]}
    table = {"hardened": (1.9e-6, 7.0e-7, 6.0e-8), "measured": (1.9e-4, 7.0e-5, 6.0e-6), "rack": (3.0e-4, 9.3e-5, 1.1e-5),
             "datacenter": (0.15, 6.1e-3, 5.3e-4), "heavy-tail": (1.4e-3, 5.0e-4, 2.0e-4)}
    for mid, want in table.items():
        got = [E.survival_at(s[mid]["t_us"], s[mid]["survival_fraction"], t) for t in (250, 500, 1000)]
        assert got == pytest.approx(list(want), rel=0.05), mid
    base = {"m": 1, "lambda_per_s": 100, "T_s": 600, "alpha": 0.01, "beta": 0.01}
    first_cut = {(0, 65_536): ((0.22, 1), (0.27, 20), (0.29, 29), (18.25, None), (0.45, 110)),
                 (1, 65_536): ((0.41, 1), (0.44, 10), (0.44, 12), (1.32, 484), (0.52, 46)),
                 (2, 65_536): ((0.61, 0), (0.62, 3), (0.63, 5), (0.82, 99), (0.68, 29)),
                 (0, 147_456): ((0.22, 1), (0.25, 10), (0.25, 11), (0.82, 311), (0.32, 41)),
                 (0, 393_216): ((0.22, 0), (0.23, 2), (0.23, 2), (0.28, 26), (0.26, 15))}
    for (d, b), cells in first_cut.items():
        for mid, (bound_pct, f0) in zip(table, cells):
            out = E.evaluate(doc, {"model": mid, "d": d, "block_bytes": b, **base})
            assert round(100 * out["bound_fraction"], 2) == pytest.approx(bound_pct, abs=0.06), (mid, d, b)
            if f0 is not None:
                assert out["f0_count"] == f0, (mid, d, b)


def test_the_vectors_reproduce_from_the_file_alone(doc):
    vs = doc["vectors"]
    assert 40 <= len(vs) <= 60
    for v in vs:
        got = E.evaluate(doc, v["inputs"])
        assert got["f0_count"] == v["expected"]["f0_count"]
        for k in ("tau_us", "p_fraction", "sampling_fraction", "bound_fraction"):
            assert got[k] == pytest.approx(v["expected"][k], rel=v["tolerance"]["relative"], abs=0), k
    inputs = [v["inputs"] for v in vs]
    assert {i["model"] for i in inputs} == set(E.LABELS)
    assert {(i["d"], i["block_bytes"]) for i in inputs} >= {(0, 65_536), (1, 65_536), (2, 65_536), (0, 147_456), (0, 393_216)}
    assert {i["alpha"] for i in inputs} == set(E.ALPHAS) and {i["beta"] for i in inputs} == set(E.BETAS)
    assert {i["d"] for i in inputs} == {*range(6), 12}
    assert all(v["expected"]["rounds_D"] == min(v["inputs"]["d"], 5) for v in vs)
    f0s = [v["expected"]["f0_count"] for v in vs]
    assert 0 in f0s and any(0 < f < 1000 for f in f0s) and any(f >= 1000 for f in f0s)


def test_the_file_is_the_contract(doc):
    assert doc["schema"] == "pous-explorer/v1"
    assert doc["block_sizes"] == [{"block_bytes": b, "call_ns": E.call_ns(b)} for b in E.BLOCK_BYTES]
    assert [e["D"] for e in doc["eps_proof"]] == list(range(6))
    assert {e["D"]: e["eps_proof_fraction"] for e in doc["eps_proof"] if e["D"] in (0, 1)} == {0: 0.0021, 1: 0.004}
    sets = [w["id"] for w in doc["weight_sets"]]
    assert len(set(sets)) == len(sets)
    for s in doc["schemes"]:
        assert s["kind"] in ("measured", "projected") and set(s["overhead_x_by_tokens"]) <= {"1", "8", "32", "2048"}
        assert s["weight_set"] in sets and s["block_bytes"] in E.BLOCK_BYTES
        assert "art" not in s or (s["art"].startswith("art:") and len(s["art"]) == 68)
    assert len(json.dumps(doc)) < 1_000_000


def _soak() -> dict:
    """A small `pous-soak/v1`: per label, two loaded 1000 Hz cells of 3,000 and 1,000 answers, a loaded 10 Hz cell of 1,000,
    an idle cell and a timestamping-off row that the presets must leave out."""
    grid = [10 ** (k / 40) for k in range(201)]
    params = {"link_bps": 4e11, "setting": "tuned", "nic": "mlx5_core", "mtu": 1500, "mss_bytes": 1448,
              "frame_overhead_bytes": 90, "header_bytes": 192, "b_pcie_bytes_per_s": 25e9, "rx_usecs": 0.0,
              "nic_rx_base_us": 2.0, "tx_fixed_us": 1.0, "request_frame_bytes": 106, "prop_ns_per_m": 5.0, "cable_m": 3.0,
              "t_nic_rx_us": 2.0}
    metrics = [{"id": "tor_400_tuned", "label": "400", "params": params},
               {"id": "tor_100_default", "label": "100", "params": {**params, "link_bps": 1e11, "setting": "default",
                                                                    "rx_usecs": 8.0, "t_nic_rx_us": 10.0}}]
    cells, survival = [], []

    def s_of(times: list[float]) -> list[float]:
        return [sum(x >= t for x in times) / len(times) for t in grid]

    for j, b in enumerate(E.BLOCK_BYTES):
        for k, (load, rate, n, slow) in enumerate((("decode", 1000, 3000, 900.0), ("mixed", 10, 1000, 3000.0),
                                                    ("idle", 100, 2000, 50_000.0), ("prefill", 1000, 1000, 100.0 + j))):
            cid = 4 * j + k
            cells.append({"cell": cid, "load": load, "rate_hz": rate, "block_bytes": b, "config": "tuned"})
            times = [100.0 + j] * (n - 1) + [slow]
            for metric in ("tor_400_tuned", "tor_100_default"):
                survival.append({"cell": cid, "metric": metric, "op": "challenge", "timestamping": True, "n": n,
                                 "max_us": slow, "s": s_of(times)})
        survival.append({"cell": 4 * j, "metric": "tor_400_tuned", "op": "challenge", "timestamping": False, "n": 10,
                         "max_us": 1e5, "s": s_of([1e5] * 10)})
    return {"run": {"host": "node", "gpu": "gpu"}, "t_us": grid, "metrics": metrics, "cells": cells, "survival": survival}


def test_the_tor_presets_weight_each_rate_equally():
    soak = _soak()
    built = E.build(generated_at="x", commit="test", soak=(soak, "art:soak"))
    models = {m["id"]: m for m in built["honest_models"]}
    assert list(models) == ["hardened", "measured", "rack", "rack-100g", "datacenter", "heavy-tail"]
    rack = models["rack"]["curves"][0]
    at = lambda m, t: E.survival_at(models[m]["curves"][0]["t_us"], models[m]["curves"][0]["survival_fraction"], t)
    # 1000 Hz: 1 slow answer in 4,000; 10 Hz: 1 in 1,000; idle and ts-off rows left out
    i900 = max(i for i, t in enumerate(rack["t_us"]) if t <= 900.0)
    assert rack["survival_fraction"][i900] == pytest.approx((1 / 4000 + 1 / 1000) / 2, rel=1e-5)
    i3000 = max(i for i, t in enumerate(rack["t_us"]) if t <= 3000.0)
    assert rack["survival_fraction"][i3000] == pytest.approx((0 + 1 / 1000) / 2, rel=1e-5)
    by_answers, _ = E.soak_pooled(soak, "tor_400_tuned", 65_536, by_answers=True)
    assert by_answers[i900] == pytest.approx(2 / 5000, rel=1e-5)
    t_last = rack["t_us"][i3000]
    assert at("rack", 2 * t_last) == pytest.approx((1 / 2000) / 2 ** E.SOAK_TAIL_ALPHA, rel=1e-4)   # the Pareto past the data
    assert "weighted equally" in models["rack"]["description"] and "by its answers instead" in models["rack"]["description"]
    assert all(a <= b for a, b in zip(models["rack-100g"]["curves"][0]["survival_fraction"],
                                      models["datacenter"]["curves"][0]["survival_fraction"]))
    assert "rx-usecs 0" in models["rack"]["description"] and "400 Gb/s" in models["rack"]["description"]
    assert "rx-usecs 8" in models["rack-100g"]["description"] and "median 120 µs" in models["datacenter"]["description"]
    assert all(m["sources"] == ["art:soak"] for k, m in models.items() if k in E.SOAK_PRESETS)
    assert models["measured"] == next(m for m in E.build(generated_at="x", commit="test")["honest_models"]
                                      if m["id"] == "measured")
    for v in built["vectors"]:
        got = E.evaluate(built, v["inputs"])
        assert got["f0_count"] == v["expected"]["f0_count"]
    assert {v["inputs"]["model"] for v in built["vectors"]} == set(models) and 40 <= len(built["vectors"]) <= 60


def test_the_tor_presets_leave_zerocopy_out_and_end_with_the_runs_note():
    soak, plain = _soak(), E.build(generated_at="x", commit="test", soak=(_soak(), "art:soak"))
    for j, b in enumerate(E.BLOCK_BYTES):
        cid = 100 + j
        soak["cells"].append({"cell": cid, "load": "decode", "rate_hz": 1000, "block_bytes": b, "config": "zerocopy"})
        for metric in ("tor_400_tuned", "tor_100_default"):
            soak["survival"].append({"cell": cid, "metric": metric, "op": "challenge", "timestamping": True, "n": 10,
                                     "max_us": 1e5, "s": [1.0] * len(soak["t_us"])})
    built = E.build(generated_at="x", commit="test", soak=(soak, "art:soak"), soak_note="The rerun, on node 1.")
    for a, b in zip(plain["honest_models"], built["honest_models"]):
        assert a["curves"] == b["curves"]
        assert b["description"] == (f"{a['description']}  The rerun, on node 1." if a["id"] in E.SOAK_PRESETS
                                    else a["description"])


def test_the_tor_presets_give_s_at_d0_from_the_file_before_its_exclusions():
    soak, raw = _soak(), _soak()
    for r in raw["survival"]:                                        # the raw file's 144 KiB answers all at 5 ms
        if raw["cells"][r["cell"]]["block_bytes"] == 147_456 and r["timestamping"]:
            r["s"] = [1.0 if t <= 5000 else 0.0 for t in raw["t_us"]]
            r["max_us"] = 5000.0
    built = E.build(generated_at="x", commit="test", soak=(soak, "art:soak"), soak_raw=(raw, "art:raw"))
    plain = E.build(generated_at="x", commit="test", soak=(soak, "art:soak"))
    for a, b in zip(plain["honest_models"], built["honest_models"]):
        assert a["curves"] == b["curves"]
        if a["id"] not in E.SOAK_PRESETS:
            assert a == b
            continue
        here = [E.survival_at(c["t_us"], c["survival_fraction"], E.tau_us(0, E.call_ns(c["block_bytes"]), 1)) for c in a["curves"]]
        tail = b["description"][len(a["description"]):]
        assert tail.startswith("  S at d = 0 (64, 144, 384 KiB) is " + ", ".join(f"{x:.3g}" for x in here) + " here")
        assert "with the runs this file leaves out (art:raw) it would be" in tail and tail.split("would be ")[1].split(", ")[1] == "1"


def test_measured_from_pooled_histograms():
    edges = [10 ** (k / 40) for k in range(201)]

    def run(at_us: list[float]) -> dict:
        counts = [0] * 200
        for x in at_us:
            counts[max(i for i in range(200) if edges[i] <= x)] += 1
        return {"histograms": {"edges_us": edges, "elapsed": {"counts": counts, "under": 0, "over": 0}}}
    docs = [run([57.0] * 9_990 + [200.0] * 9 + [1000.0]), run([57.0] * 9_990 + [300.0] * 10)]
    _, s_edges, total = E.pool_histograms(docs)
    assert total == 20_000 and s_edges[0] == 1.0 and s_edges[-1] == 0.0
    s, n = E.measured_histograms(docs)
    t = np.array([10.0, 100.0, 150.0, 250.0, 750.0, 2000.0, 1e5])
    got = s(t)
    assert n == 20_000 and got[0] == 1.0 and got[1] == pytest.approx(20 / 20_000, rel=0.02)
    assert got[3] == pytest.approx(11 / 20_000, rel=0.1) and all(a >= b for a, b in pairwise(got))
    assert got[4] == pytest.approx(1 / 20_000, rel=0.1) and got[5] == pytest.approx(got[4] * (2000 / 750) ** -1.8, rel=0.1)
    built = E.build([d | {"source": f"art:{i}"} for i, d in enumerate(docs)], generated_at="x", commit="test")
    measured = next(m for m in built["honest_models"] if m["id"] == "measured")
    assert measured["sources"] == ["art:0", "art:1"] and "pooled histograms of 2 series runs" in measured["description"]
