"""`records.build_summary`: the summary each writer lays out (`pipeline.row_records`' TP1 and TP Builds, and the retired `tp2_evidence.py`) loads
as the typed record and dumps back to the same JSON; a missing or unknown field is refused by name."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from verity_vllm.pipeline import row_records as R
from verity_vllm.records.build_summary import BuildSummary
from verity_vllm.records.fields import MissingField, RecordError, UnknownField

STEP, REQ, SHAPE = "e6" * 32, "fc" * 32, "ab" * 32


def _result(d: Path, digest: str, correspondence: str | None = None, **extra) -> None:
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": digest, "correspondence_digest": correspondence}, **extra}))


def test_a_tp1_build_loads_as_its_writers_wrote_it(tmp_path):
    _result(tmp_path / "build_step", "p1" * 32, STEP)
    _result(tmp_path / "build_request", "p2" * 32, REQ, census={"census": "c" * 64})
    _result(tmp_path / "build_request_LP9_T1", "p3" * 32, SHAPE)
    _result(tmp_path / "build_request_LP9_T1_C4", "p4" * 32)
    assert R.build_summary(str(tmp_path)) == "PASS"
    (tmp_path / "build_workload").mkdir()
    (tmp_path / "build_workload" / "workload_program.json").write_text(json.dumps({"workload_digest": "w" * 64, "stop_rule": {"eos": [2]}}))
    R.add_workload(str(tmp_path), "w" * 64, 0)
    doc = json.loads((tmp_path / "build_summary.json").read_text())
    s = BuildSummary.load(tmp_path / "build_summary.json")
    assert s.layout == "tp1" and s.world is None and s.ranks is None and s.dump() == doc
    assert s.correspondence_digests() == {"step": STEP, "request": REQ, "build_request_LP9_T1": SHAPE}
    assert s.wrappers["workload"].workload_digest == "w" * 64 and s.wrappers["workload"].program_digest is None
    assert s.wrappers["request"].entry["census_digest"] == "c" * 64
    assert BuildSummary.of(s, "elsewhere") is s


def test_a_tp_build_loads_with_its_ranks(tmp_path):
    b = tmp_path / "build"
    for r in range(2):
        _result(b / f"rank{r}" / "build_step", f"s{r}" * 32, aot=True)
        _result(b / f"rank{r}" / "build_request_LP9_T1", f"q{r}" * 32)
        (b / f"rank{r}" / "build_request_LP9_T1.log").write_text("")
    assert R.tp_build_summary(str(b), 2) == "FAIL"                # no workload compose: the workload entries say so
    doc = json.loads((tmp_path / "build_summary.json").read_text())
    s = BuildSummary.load(tmp_path / "build_summary.json")
    assert s.layout == "tp" and s.world == 2 and s.dump() == doc and s.correspondence_digests() == {}
    assert s.wrappers["step"].program_digest == "s0" * 32 and s.wrappers["step"].ok is None
    assert s.ranks["rank1"]["build_step"].program_digest == "s1" * 32 and s.ranks["rank1"]["build_step"].entry["construction"] == "aot"
    assert s.ranks["rank0"]["workload"].ok is False


def test_r19_row_pod_tp2_entries_for_the_log_beside_a_derive_load():
    doc = {"tp": {"world": 2}, "step": {"program_digest": "s0"},
           "ranks": {f"rank{r}": {"build_step": {"ok": True, "unsupported": None, "message": "", "program_digest": f"s{r}"},
                                  "build_request_LP9_T1.rc": {"ok": False, "unsupported": None, "message": "no result.json", "program_digest": None},
                                  "workload": {"ok": False, "workload_digest": None, "reason": None, "message": None, "mechanism": "GP-01"}}
                     for r in range(2)}}
    s = BuildSummary.of(doc, "r19")
    assert s.layout == "tp" and s.dump() == doc


def _r19_tp2() -> dict:
    """What `verity_capture/bench/tp2_evidence.py` wrote for a B=1 TP2 row (the regression harness's TP2 reference rows)."""
    ent = {"ok": True, "unsupported": None, "exception": None, "message": "", "program_digest": "s0", "identity": None, "target": {},
           "applicability": None, "tp": {"world": 2, "rank": 0}, "LP": None, "T": None, "M": 1, "wall_secs_total": 3.0, "max_rss_mb": 900,
           "sampler": None, "cmd": "derive_step --tp 2 --rank 0"}
    comps = {"ok": True, "unsupported": None, "exception": None, "message": "", "program_digest": "w0", "program_digest_kind": "workload",
             "component_digests": ["c0"], "components": [{"dir": "build_request_LP9_T1", "ok": True}], "tp": None, "n_components": 1,
             "declared_requests": 1, "gp01_compose": "ok", "cmd": "derive_step"}
    return {"tp": {"world": 2, "ranks": {"step_rank0": {"ok": True, "program_digest": "s0"}, "request_rank1": {"ok": False, "missing": "/b"}},
                   "all_ranks_ok": False},
            "cmd": "derive_step --model <repo>", "note": "TP2: `step`/`request` mirror rank 0",
            "step": ent, "step_rank0": ent, "step_rank1": dict(ent, program_digest="s1"), "request": comps, "request_rank0": comps}


def test_an_r19_tp2_build_loads_and_records_no_correspondence():
    doc = _r19_tp2()
    s = BuildSummary.of(doc, "r19")
    assert s.layout == "tp2-r19" and s.world == 2 and s.ranks is None and s.dump() == doc
    assert s.correspondence_digests() == {} and s.wrappers["step_rank1"].program_digest == "s1"
    bad = _r19_tp2()
    bad["step"]["census_digest"] = "c"
    with pytest.raises(UnknownField, match="'step.census_digest'"):
        BuildSummary.of(bad, "r19")


def _tp1() -> dict:
    return {"step": {"ok": True, "program_digest": "p1", "correspondence_digest": STEP}, "request": {"ok": True, "program_digest": "p2"}}


@pytest.mark.parametrize("edit, error, field", [
    (lambda d: d.pop("step"), MissingField, "step"),
    (lambda d: d.pop("request"), MissingField, "request"),
    (lambda d: d.update(build_request_LP9=_tp1()["request"]), RecordError, "build_request_LP9"),
    (lambda d: d["step"].update(digest="x"), UnknownField, "step.digest"),
    (lambda d: d["step"].update(ok="yes"), RecordError, "step.ok"),
    (lambda d: d["request"].update(correspondence_digest=7), RecordError, "request.correspondence_digest"),
    (lambda d: d.update(step=[1]), RecordError, "step"),
])
def test_a_tp1_summary_is_refused_by_field(edit, error, field):
    doc = copy.deepcopy(_tp1())
    edit(doc)
    with pytest.raises(error) as e:
        BuildSummary.of(doc, "build_summary.json")
    assert e.value.field == field and e.value.where == "build_summary.json"


@pytest.mark.parametrize("edit, error, field", [
    (lambda d: d["tp"].update(rank=0), UnknownField, "tp.rank"),
    (lambda d: d["tp"].pop("world"), MissingField, "tp.world"),
    (lambda d: d.update(request={"ok": True}), UnknownField, "request"),
    (lambda d: d["ranks"].update(r2={}), RecordError, "ranks.r2"),
    (lambda d: d["ranks"]["rank0"].update(build_request={"ok": True}), RecordError, "ranks.rank0.build_request"),
    (lambda d: d["ranks"]["rank0"]["build_step"].update(n_gates=3), UnknownField, "ranks.rank0.build_step.n_gates"),
])
def test_a_tp_summary_is_refused_by_field(edit, error, field):
    doc = {"tp": {"world": 1}, "step": {"program_digest": "s0"}, "ranks": {"rank0": {"build_step": {"ok": True, "program_digest": "s0"}}}}
    edit(doc)
    with pytest.raises(error) as e:
        BuildSummary.of(doc, "build_summary.json")
    assert e.value.field == field


def test_a_summary_that_is_not_an_object_is_refused():
    with pytest.raises(RecordError, match="not a JSON object"):
        BuildSummary.of([], "build_summary.json")
