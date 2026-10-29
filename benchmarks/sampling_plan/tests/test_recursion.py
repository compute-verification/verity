"""`recursion.py`: a recursion_e2e result's measured session scaled to every machine-window of each deployment."""
import pytest
import recursion as R

RESULT = {"headline": {"ok": True, "pipeline": {"inner M0 e2e_s": 10.0, "V* staging_s (vstage + vcap steps)": 20.0,
                                                "outer prove (firewall) e2e_s": 570.0, "Lean step_s": 60.0}},
          "summary": {"outer": {"firewall": {"statements": 12}}},
          "rows": [{"stage": "baseline --zk", "e2e_s": {"median": 20.0}}], "staged": {"n": 3, "unit": "u", "m": 29}}


def test_a_session_scales_to_every_machine_window_of_each_deployment():
    deps = R.deployments()
    assert [d["id"] for d in deps] == [R.DEPLOYMENT, R.WORLD] and deps[0]["machines"] == 12500 and deps[1]["machines"] > 1e7
    s = R.session(RESULT)
    assert (s["prover_s"], s["auditor_s"], s["statements"], s["baseline_zk_e2e_s"]) == (600.0, 60.0, 12, 20.0)
    week = next(r for r in R.periods(s, deps) if r["deployment"] == R.DEPLOYMENT and r["period"] == "week")
    assert week["sessions"] == 12500 * 7 * 86400 / 600
    # a 600 s session every 600 s window keeps one prover node busy per machine
    assert week["prover_nodes"] == pytest.approx(12500) and week["auditor_nodes"] == pytest.approx(1250)
    assert "| week |" in R.render(s, deps, R.periods(s, deps), 1.0)


def test_a_result_without_a_whole_recursive_run_is_refused():
    with pytest.raises(ValueError, match="not a whole recursive run"):
        R.session({**RESULT, "headline": {"pipeline": {**RESULT["headline"]["pipeline"], "Lean step_s": None}}})
