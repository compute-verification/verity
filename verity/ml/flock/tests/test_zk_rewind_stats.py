"""The Flock ZK rewinding statistics (``zk_rewind_stats.py``) on synthetic dumps: matching real and simulated views pass; a simulator whose
views end elsewhere, or whose online values do not depend on the prover's randomness, fails. The dumps from
``flock-circuit zkrewind`` run on a VM; this checks the analysis itself."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
sys.path.insert(0, os.path.dirname(__file__))
import zk_rewind_stats  # noqa: E402
from test_zk_stats import _row  # noqa: E402


def _hexes(rng, n):
    return [rng.integers(0, 256, 16, dtype=np.uint8).tobytes().hex() for _ in range(n)]


def _dump(n, sim_stops_early=False, sim_online_fixed=False):
    rng = np.random.default_rng(11)
    fixed = [bytes([i % 5] * 16).hex() for i in range(zk_rewind_stats_online())]
    lines = []
    for kind in ("real", "sim"):
        for run in range(n):
            stop = rng.random() < 0.5
            if kind == "sim" and sim_stops_early:
                stop = True
            k = int(rng.integers(0, 3)) if kind == "sim" and sim_stops_early else int(rng.integers(0, 100))
            end = "verifier" if stop else "complete"
            online = fixed if kind == "sim" and sim_online_fixed else _hexes(rng, zk_rewind_stats_online())
            v = {"kind": kind, "strategy": "abort", "run": run, "end": end, "last_round": ["t/rep0", k] if stop else ["t/rep1", 99],
                 "rounds": {}, "accepted": None if stop else True, "online": online}
            lines.append("ZKVIEW\t" + json.dumps(v))
            if not stop:
                for rep in range(2):
                    lines.append("ZKVIEWP\t" + json.dumps(dict(_row(kind, run, rep, rng), strategy="abort")))
    sims = [{"extraction_complete": True, "estimation_runs": 2, "rewinds": 1, "fail": None} for _ in range(n)]
    lines.append("ZKREWIND\t" + json.dumps({"strategy": "abort", "sims": sims}))
    return lines


def zk_rewind_stats_online():
    return 32


def test_matching_views_pass():
    res = zk_rewind_stats.analyse(*zk_rewind_stats.load(_dump(40)))
    assert res["pass"], json.dumps(res, indent=1)
    s = res["strategies"]["abort"]
    assert s["online"]["online_values"]["real_indistinguishable_from_sim"]
    assert isinstance(s["proofs"], dict)


def test_views_that_end_elsewhere_fail():
    res = zk_rewind_stats.analyse(*zk_rewind_stats.load(_dump(40, sim_stops_early=True)))
    assert not res["pass"]
    assert res["strategies"]["abort"]["end_p"] < zk_rewind_stats.ALPHA


def test_online_values_without_the_provers_randomness_fail():
    res = zk_rewind_stats.analyse(*zk_rewind_stats.load(_dump(40, sim_online_fixed=True)))
    assert not res["pass"]
    assert not res["strategies"]["abort"]["online"]["online_values"]["real_indistinguishable_from_sim"]


def _two_tables(lines, shared):
    """``lines`` with a second table's proof per proof: fresh values, or table 0's own (randomness shared across tables)."""
    rng = np.random.default_rng(7)
    out = []
    for line in lines:
        out.append(line)
        tag, _, body = line.partition("\t")
        if tag == "ZKVIEWP":
            v = json.loads(body)
            w = dict(v) if shared else dict(_row(v["kind"], v["run"], v["rep"], rng), strategy=v["strategy"])
            out.append("ZKVIEWP\t" + json.dumps(dict(w, table=1)))
    return out


def test_two_tables_are_tested_apart_and_shared_randomness_fails():
    res = zk_rewind_stats.analyse(*zk_rewind_stats.load(_two_tables(_dump(40), shared=False)))
    assert res["pass"], json.dumps(res, indent=1)
    s = res["strategies"]["abort"]
    assert not any(s["across_tables"].values()) and any(k.startswith("table1:") for k in s["proofs"])
    bad = zk_rewind_stats.analyse(*zk_rewind_stats.load(_two_tables(_dump(40), shared=True)))
    assert not bad["pass"] and all(bad["strategies"]["abort"]["across_tables"].values())


def _gk(real, fresh, reused=None):
    v = {"strategy": "cap-gated", "eps": 1 / 16, "attempts": 40, "real_complete": real, "extraction_tries": 3,
         "rewinds_complete_fresh": fresh, "rewinds_complete_reused": reused}
    return zk_rewind_stats.analyse(*zk_rewind_stats.load(["ZKGK\t" + json.dumps(v)]))


def test_independent_rewinds_pass_and_reused_ones_are_told_apart():
    res = _gk(21, 19, reused=2)
    assert res["pass"], json.dumps(res, indent=1)
    assert res["rewind_independence"]["cap-gated"]["p_reused"] < zk_rewind_stats.ALPHA


def test_rewinds_on_one_dummy_opening_fail():
    """red-team-m2-zk G1: rewinds that reuse the extraction run's dummy opening complete at the gate's epsilon."""
    assert not _gk(21, 2)["pass"]


def test_the_family_verdict_adjusts_for_the_number_of_tests():
    row = {k: 0.5 for k in zk_rewind_stats.FIELD_TESTS}
    out = {"strategies": {"s": {"end_p": 0.5, "proofs": {"a": dict(row, bits_real=1e-4), "b": row}}}}
    fam = zk_rewind_stats.family(out)
    assert fam["tests"] == 11 and abs(fam["bonferroni_p"] - 11e-4) < 1e-12 and fam["pass"]
    out["strategies"]["s"]["proofs"]["a"]["bits_real"] = 1e-6
    assert not zk_rewind_stats.family(out)["pass"]
