"""`research metrics show` against a fake route: each declared metric's line, one no run has recorded, a refused row and
why the site couldn't read something; filtering by id and owner; an undeclared id or a refusal as one line."""
from __future__ import annotations

import json

from research import metrics
from research.slack import RouteError

COST = {"id": "vllm/serving-cost-glm53", "title": "GLM-5.3's serving cost over stock vLLM", "owner": "vllm",
        "unit": "percent of stock's serving compute", "better": "lower", "command": "bash benchmarks/serve_tax/run.sh",
        "every": "7d", "floor": None, "pastFloor": None, "stale": False, "ageMs": 2 * 86_400_000,
        "change7d": -0.4, "change30d": None,
        "latest": {"value": 2.0, "ci95": [1.8, 2.2], "verdict": "meets", "target": 2.05, "run": "r20261109-000000-abcd",
                   "commit": "d" * 40}}
LAND = {"id": "ci/mark-to-main", "title": "Ready mark to main", "owner": "ci", "unit": "min", "better": "lower",
        "command": "research run -- python tools/ci/landing.py", "every": "12h", "floor": 90, "pastFloor": None,
        "stale": False, "ageMs": None, "change7d": None, "change30d": None, "latest": None}


class Fake:
    def __init__(self, report: dict | None = None, refuse: RouteError | None = None):
        self.report, self.refuse = report, refuse

    def call(self) -> dict:
        if self.refuse:
            raise self.refuse
        return json.loads(json.dumps(self.report))


REPORT = {"sha": "s1", "metrics": [COST, LAND], "refused": [{"id": "ci/a", "why": "better is neither lower nor higher"}],
          "why": None}


def test_show_prints_each_metric_its_standing_and_what_was_refused(capsys):
    assert metrics.main(["show"], Fake(REPORT)) == 0
    assert capsys.readouterr().out.splitlines() == [
        "vllm/serving-cost-glm53  @vllm  GLM-5.3's serving cost over stock vLLM",
        "  2 [1.8, 2.2] percent of stock's serving compute  7d -0.4  30d —  age 2 d (every 7d)  meets (target 2.05)",
        "  run r20261109-000000-abcd at dddddddddddd",
        "ci/mark-to-main  @ci  Ready mark to main",
        "  not recorded yet; measured by: research run -- python tools/ci/landing.py",
        "refused: ci/a: better is neither lower nor higher",
    ]


def test_show_filters_by_id_and_owner_and_prints_json(capsys):
    assert metrics.main(["show", "--owner", "ci", "--json"], Fake(REPORT)) == 0
    assert [m["id"] for m in json.loads(capsys.readouterr().out)["metrics"]] == ["ci/mark-to-main"]
    assert metrics.main(["show", "vllm/serving-cost-glm53"], Fake(REPORT)) == 0
    assert capsys.readouterr().out.startswith("vllm/serving-cost-glm53  @vllm")


def test_an_undeclared_id_a_refusal_and_the_sites_note_are_one_line_each(capsys):
    assert metrics.main(["show", "vllm/nope"], Fake(REPORT)) == 1
    assert "not declared in benchmarks/metrics.toml: vllm/nope" in capsys.readouterr().out
    assert metrics.main(["show"], Fake(refuse=RouteError(f"GET {metrics.METRICS}: 401 invalid_identity", 401))) == 1
    assert capsys.readouterr().out == f"research metrics: GET {metrics.METRICS}: 401 invalid_identity\n"
    assert metrics.main(["show"], Fake({"sha": None, "metrics": [], "refused": [], "why": "not on main yet"})) == 0
    assert capsys.readouterr().out.splitlines() == ["no metric is declared yet", "note: not on main yet"]


def test_a_site_key_is_sent_as_is_and_no_identity_is_minted(monkeypatch):
    sent = []
    monkeypatch.setattr(metrics, "http_json", lambda method, url, data, headers: sent.append((method, url, headers)) or REPORT)
    monkeypatch.setattr(metrics, "_Identity", lambda aud: (_ for _ in ()).throw(AssertionError("minted an identity")))
    assert metrics.Route(key="vsk_test").call()["sha"] == "s1"
    assert sent == [("GET", metrics.METRICS, {"Authorization": "Bearer vsk_test"})]
