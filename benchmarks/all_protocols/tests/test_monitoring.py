"""The result per monitoring period, over all compute:

* each period is ``machines × ⌊period / window⌋`` blocks of the census deployment, and a missing one names where to add it;
* the bound is onsite's own ``onsite bound`` on the windows onsite accepted, over the run's blocks and each period's,
  with each resource's pooled figure at both δ, and its figures pass through unchanged: this module computes none;
* with no executable, or no window onsite accepted, there is no bound and the report says why;
* a window that rejects rejects the monitoring period;
* the summary line leads with the run, the week, the month and the year, and what the bound leaves open.
"""
from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from verity_all_protocols import monitoring as M

DEP = {"id": "test-dc", "machines": 10, "window_s": 3600}
ANSWER = {"commitment": "ab", "profile": {}, "protocol": "verity/onsite/v0", "public": {"k": 1}, "reason": None,
          "verdict": "accept"}


def report(w: int, composed: str, answer: dict | None = None) -> dict:
    return {"window": w, "protocols": {"integrity": {"verdict": "accept", "owner": "compute-accounting"}},
            "composed": {"verdict": composed, **({"answer": answer} if answer else {})}}


def bound_exe(tmp_path: Path, rc: int = 0) -> Path:
    """A stand-in `onsite` whose ``bound`` records its arguments and answers in ``seen.json`` and prints, per horizon,
    an error of ``1/blocks``, and at each δ an integrity figure of ``blocks`` and a space figure with no inputs."""
    exe = tmp_path / "onsite"
    exe.write_text(f"#!{sys.executable}\n" + f'''import json, sys
assert sys.argv[1] == "bound"
lines = open(sys.argv[2]).read().splitlines()
rest = sys.argv[3:]
deltas = [rest[i + 1] for i, x in enumerate(rest) if x == "--delta"]
counts = [int(x) for i, x in enumerate(rest) if x != "--delta" and (i == 0 or rest[i - 1] != "--delta")]
json.dump({{"answers": [json.loads(x) for x in lines], "counts": counts, "deltas": deltas}},
          open({str(tmp_path / "seen.json")!r}, "w"))
if {rc}:
    print("refused", file=sys.stderr)
    sys.exit({rc})
pooled = lambda b: {{d: {{"integrity": {{"faulty_blocks_at_most": b}}, "space": {{"why": "no block's PoUS certified"}}}}
                     for d in deltas}}
print(json.dumps({{"statement": "S", "open": [{{"figure": "bandwidth", "missing": "M"}}], "blocks": [],
                  "run": {{"blocks": len(lines), "error": "1/1000", "vacuous": False}},
                  "horizons": [{{"blocks": b, "error": f"1/{{b}}", "vacuous": False, "pooled": pooled(b)}}
                               for b in counts]}}))
''')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


def test_the_deployment_is_the_census_one():
    d = M.deployment()
    assert d["id"] == M.REFERENCE and d["machines"] == 12500 and d["window_s"] == 600
    with pytest.raises(KeyError, match="add it there"):
        M.deployment("no-such-dc")


def test_each_period_is_the_deployments_machines_times_its_windows():
    ps = M.periods(DEP)
    assert {p: (v["blocks_per_machine"], v["blocks"]) for p, v in ps.items()} == {
        "week": (168, 1680), "month": (720, 7200), "year": (8760, 87600)}


def test_the_bound_is_onsites_own_evaluation_on_the_accepted_windows(tmp_path):
    exe = bound_exe(tmp_path)
    other = {**ANSWER, "commitment": "cd"}
    reports = [report(0, "accept", ANSWER), report(1, "accept", other), report(2, "undecided", {**ANSWER, "verdict": "reject"})]
    m = M.monitoring(reports, DEP, str(exe))
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert seen == {"answers": [ANSWER, other], "counts": [1680, 7200, 87600], "deltas": ["1/100", "1/1000000"]}
    assert m["accepted_windows"] == [0, 1]
    assert m["bound"]["run"] == {"blocks": 2, "error": "1/1000", "vacuous": False}
    assert {p: (v["error"], v["vacuous"]) for p, v in m["periods"].items()} == {
        "week": ("1/1680", False), "month": ("1/7200", False), "year": ("1/87600", False)}
    assert m["periods"]["week"]["pooled"]["1/100"]["integrity"] == {"faulty_blocks_at_most": 1680}
    lead = M.lead(m)
    assert set(lead) == {"run", "week", "month", "year", "open", "composed"}
    assert lead["year"] == {"blocks": 87600, "error": "1/87600", "vacuous": False,
                            **{d: {"integrity": 87600, "space": "no block's PoUS certified"} for d in M.DELTAS}}
    assert lead["open"] == ["bandwidth"]


def test_a_refused_evaluation_is_an_error_not_a_figure(tmp_path):
    with pytest.raises(RuntimeError, match="onsite bound exited 1: refused"):
        M.monitoring([report(0, "accept", ANSWER)], DEP, str(bound_exe(tmp_path, rc=1)))


def test_with_no_executable_or_no_accepted_window_there_is_no_bound(tmp_path):
    m = M.monitoring([report(0, "accept", ANSWER)], DEP)
    assert m["bound"] is None and m["why"] == "no onsite executable given"
    assert M.lead(m) == {"why": "no onsite executable given", "composed": "accept"}
    m = M.monitoring([report(0, "pending-ports")], DEP, str(bound_exe(tmp_path)))
    assert m["bound"] is None and "no block accepted" in m["why"] and not (tmp_path / "seen.json").exists()
    assert "error" not in m["periods"]["week"]


def test_a_rejected_window_rejects_the_monitoring_period():
    m = M.monitoring([report(0, "accept", ANSWER), report(1, "reject"), {"window": 2, "error": "the loop: x"}], DEP)
    assert m["composed"] == {"verdict": "reject", "windows": 2, "errors": 1}
    assert m["protocols"]["integrity"] == {"verdicts": ["accept"], "owner": "compute-accounting"}
    assert [w["window"] for w in m["windows"]] == [0, 1, 2]
