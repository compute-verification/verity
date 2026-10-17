import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NCI = HERE.parent
sys.path.insert(0, str(NCI))

import catalog  # noqa: E402
import render  # noqa: E402
import slowdown  # noqa: E402

LOCK = NCI.parents[1] / "Security" / "lean-audit.json"


def test_catalog_has_no_problems():
    assert catalog.problems() == []


def test_problems_catches_a_missing_quote_and_a_dangling_citation(tmp_path):
    for name in ("parameters", "rows", "calibrations"):
        (tmp_path / f"{name}.json").write_text((NCI / f"{name}.json").read_text())
    (tmp_path / "sources").mkdir()
    for path in (NCI / "sources").glob("*.json"):
        (tmp_path / "sources" / path.name).write_text(path.read_text())
    sources = catalog.load("sources", tmp_path)
    sources[0]["findings"][0]["quote"] = ""
    first = tmp_path / "sources" / f"{sources[0]['topics'][0]}.json"
    first.write_text(json.dumps([s for s in sources if s["topics"][0] == sources[0]["topics"][0]]))
    parameters = catalog.load("parameters", tmp_path)
    parameters[0]["estimates"][0]["evidence"].append("no-such-source/1")
    (tmp_path / "parameters.json").write_text(json.dumps(parameters))
    found = catalog.problems(tmp_path)
    assert f"finding {sources[0]['findings'][0]['id']}: missing quote" in found
    assert f"estimate {parameters[0]['estimates'][0]['id']}: evidence no-such-source/1 is no finding" in found


def test_every_row_has_a_calibration_of_its_model():
    models = {c["model"] for c in catalog.load("calibrations")}
    assert {r["model"] for r in catalog.load("rows")} <= models


def test_cited_theorems_are_guarantees_in_the_lock():
    guarantees = json.loads(LOCK.read_text())["guarantees"]
    assert slowdown.THREAD_THEOREM in guarantees
    assert slowdown.CHAIN_THEOREM in guarantees


def test_index_renders_every_source_and_estimate():
    text = render.render()
    for s in catalog.load("sources"):
        assert f"**{s['id']}**" in text
    for e in catalog.estimates(catalog.load("parameters")):
        assert f"**{e}**" in text
