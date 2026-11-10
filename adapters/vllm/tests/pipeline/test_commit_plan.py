"""The Commit's plan (`pipeline.commit_plan`): the producer facts derived on CPU before the GPU is held, taken by the Commit only under
its own key, and the row's `plan` stage that writes it.  CPU only: the Programs are the synthetic Gemma-like step of the oracle tests."""
from __future__ import annotations

import copy
import json
import types
from pathlib import Path

import pytest

from tests.check.test_oracle_compare_v2_producers import RES, gemma_like_program
from tests.pipeline.test_row import PD, _record_manifest, _row_with_build
from verity_vllm.check import oracle_compare as OC
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.pipeline import commit_plan as CP
from verity_vllm.pipeline import row_config as RC
from verity_vllm.pipeline import row_plan as RP
from verity_vllm.pipeline import row_records as R
from verity_vllm.query.required import request_manifest


def _commit_args(tmp_path: Path) -> types.SimpleNamespace:
    """The inputs of a B=1 Commit: its workload, the manifest the query path built from the synthetic Program, its Program dir."""
    P = gemma_like_program()
    man = request_manifest(P, Correspondence(P), RES, request_id="r0").manifest
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    (tmp_path / "workload.json").write_text(json.dumps({"requests": [{"request_id": "r0", "prompt_token_ids": [1, 2], "max_tokens": 2}]}))
    (tmp_path / "build_request").mkdir()
    return types.SimpleNamespace(workload=str(tmp_path / "workload.json"), required_manifest=str(tmp_path / "manifest.json"),
                                 program_digest=PD, program_dir=str(tmp_path / "build_request"), programs_root=None)


@pytest.fixture
def planned(tmp_path, monkeypatch):
    """A plan of the synthetic Program's facts, as the plan stage writes it, and the Commit's arguments it was derived for."""
    pytest.importorskip("torch")                      # the plan reads the workload as the Commit does (vllm_adapter)
    P = gemma_like_program()
    facts = OC.program_producer_facts(P, Correspondence(P))
    monkeypatch.setattr(CP, "oracle_producers_of_record", lambda args, man, reqs, sampling: (copy.deepcopy(facts), {"used": True, "members": len(facts)}))
    args = _commit_args(tmp_path)
    CP.write(tmp_path / "plan" / "commit_plan.json", CP.derive(args))
    return args, str(tmp_path / "plan" / "commit_plan.json"), facts


def test_the_commit_takes_the_planned_facts_as_they_were_derived(planned):
    """The facts are a dict keyed by (op_path, member) whose order the promotion reads: the plan gives back an equal dict in the same
    order, and the manifest the Commit then arms with (rows promoted to their first consumer's input) is the one derived facts give."""
    args, plan, facts = planned
    (got, rec), use = CP.use(plan, args)
    assert use["used"] is True and use["plan_recomputed"] is False and use["key"] == CP.key(args)
    assert got == facts and list(got) == list(facts) and rec["members"] == len(facts)
    man = json.loads(Path(args.required_manifest).read_text())
    a, b = copy.deepcopy(man), copy.deepcopy(man)
    assert OC.promote_handed_down(a, facts) == OC.promote_handed_down(b, got) and a == b
    assert OC.promote_handed_down(copy.deepcopy(man), got)["rows"] >= 1, "the synthetic step hands model/out down to the fused norm"


@pytest.mark.parametrize("perturb, field", [
    (lambda a: Path(a.required_manifest).write_text(Path(a.required_manifest).read_text() + " "), "required_manifest_sha256"),
    (lambda a: Path(a.workload).write_text(Path(a.workload).read_text().replace('"max_tokens": 2', '"max_tokens": 3')), "workload_sha256"),
    (lambda a: setattr(a, "program_digest", "q" * 64), "program_digest"),
    (lambda a: (setattr(a, "programs_root", a.program_dir), setattr(a, "program_dir", None)), "program_dir"),
])
def test_a_plan_keyed_otherwise_is_recomputed_by_the_commit_with_the_reason(planned, perturb, field):
    """A plan whose inputs are not the Commit's (another manifest, workload, Program, or the Programs read another way) is never used:
    the Commit derives the facts itself and records plan_recomputed naming the key field."""
    args, plan, _ = planned
    perturb(args)
    got, use = CP.use(plan, args)
    assert got is None and use["used"] is False and use["plan_recomputed"] is True
    assert use["why"].startswith("plan key differs from this Commit's: ") and field in use["why"], use["why"]


def test_another_code_identity_or_vllm_is_another_key(planned, monkeypatch):
    args, plan, _ = planned
    from verity_vllm.pipeline import source_identity as SI
    monkeypatch.setattr(SI, "code_identity", lambda root: {"sha256": "c" * 64, "files": 1})
    assert "code" in CP.use(plan, args)[1]["why"]
    monkeypatch.undo()
    monkeypatch.setattr(CP, "_version", lambda dist: "0.0.0")
    assert "vllm" in CP.use(plan, args)[1]["why"]


def test_an_unreadable_plan_or_another_schema_is_recomputed(planned, tmp_path):
    args, plan, _ = planned
    got, use = CP.use(str(tmp_path / "absent.json"), args)
    assert got is None and use["plan_recomputed"] is True and use["why"].startswith("plan unreadable: FileNotFoundError")
    doc = json.loads(Path(plan).read_text())
    Path(plan).write_text(json.dumps({**doc, "schema": "verity-vllm/commit-plan/v0"}))
    assert CP.use(plan, args)[1]["why"] == "plan schema 'verity-vllm/commit-plan/v0' != verity-vllm/commit-plan/v1"


def test_the_commit_s_producer_facts_are_the_plan_s_or_derived_with_the_reason(planned, monkeypatch, tmp_path):
    from verity_vllm.pipeline.timeline import Timeline
    TL = Timeline(tmp_path / "timeline.jsonl")
    args, plan, facts = planned
    man = json.loads(Path(args.required_manifest).read_text())
    (got, _), rec = CP.producer_facts(types.SimpleNamespace(**vars(args), plan=plan), man, [], {}, tl=TL)
    assert got == facts and rec["used"] is True
    stale = types.SimpleNamespace(**{**vars(args), "program_digest": "q" * 64}, plan=plan)
    monkeypatch.setattr(CP, "oracle_producers_of_record", lambda *a: (None, {"used": False, "why": "derived here"}))
    assert CP.producer_facts(stale, man, [], {}, tl=TL) == ((None, {"used": False, "why": "derived here"}), CP.use(plan, stale)[1])
    monkeypatch.setattr(CP, "oracle_producers_of_record", lambda *a: 1 / 0)
    (got, why), rec = CP.producer_facts(types.SimpleNamespace(**vars(args), plan=None), man, [], {}, tl=TL)
    assert got is None and rec is None and why == {"used": False, "why": "producer facts not derived: ZeroDivisionError('division by zero')"}
    ends = [json.loads(ln) for ln in (tmp_path / "timeline.jsonl").read_text().splitlines()]
    assert [(e["source"], e["members"]) for e in ends if e["ev"] == "end"] == [("plan", len(facts)), ("derived", 0), ("derived", 0)]


def test_the_commit_s_parser_takes_the_plan(tmp_path):
    pytest.importorskip("torch")
    from verity_vllm.pipeline import cli
    assert cli.parse("commit", ["--out", str(tmp_path), "--plan", "p.json"], {}).plan == "p.json"
    o = cli.parse("commit-plan", ["--workload", "w.json", "--required-manifest", "m.json", "--program-dir", "pd", "--out", "o.json"], {})
    assert (o.workload, o.required_manifest, o.program_dir, o.programs_root, o.out) == ("w.json", "m.json", "pd", None, "o.json")


# ---------------------------------------------------------------------------------------------------------------- the row's plan stage

def test_the_plan_stage_plans_the_inputs_the_commit_stage_passes(tmp_path, monkeypatch):
    """The plan stage derives the manifest and weights of record (on CPU) and runs `commit-plan` over the same workload, manifest,
    Program digest and Programs the Commit stage then names; the Commit stage passes the plan with `--plan`."""
    r, built = _row_with_build(tmp_path, monkeypatch)
    r.I, r.O, r.config_run = 256, 32, True
    _record_manifest(Path(r.d) / "manifest.json", R.QUERY_OF_RECORD)
    wor: list[str] = []
    monkeypatch.setattr(r, "weights_of_record", lambda stage="commit": wor.append(stage) or ["--weights-of-record", "w.json"])
    ran: dict[str, list[str]] = {}

    def run_tl(stage, to, log, cmd, *extra):
        ran[stage] = cmd
        if stage == "plan.commit_plan":
            Path(cmd[cmd.index("--out") + 1]).parent.mkdir(parents=True, exist_ok=True)
            Path(cmd[cmd.index("--out") + 1]).write_text("{}")
        return 0

    monkeypatch.setattr(r, "run_tl", run_tl)
    RP.plan(r)
    plan = ran["plan.commit_plan"]
    assert plan[plan.index("commit-plan") + 1:] == ["--workload", r.wl, "--required-manifest", f"{r.d}/manifest.json", "--program-digest", PD,
                                                     "--program-dir", f"{r.d}/build_request", "--out", RP.path(r)]
    assert built == [] and wor == ["plan"] and "plan PASS" in (Path(r.d) / "stages.txt").read_text()

    r.hidden_so = str(tmp_path / "verity_fa2_matReqG.so")
    Path(r.hidden_so).touch()
    monkeypatch.setattr(R, "applicable_taps", lambda man, ck: [])
    monkeypatch.setattr(r, "gpu_exclusive", lambda why: None)
    monkeypatch.setattr(RC, "commit_verdict", lambda row, rck, wall: True)
    r.commit()
    cmd = ran["commit.commit_delta"]
    for flag in ("--workload", "--required-manifest", "--program-digest", "--program-dir"):
        assert cmd[cmd.index(flag) + 1] == plan[plan.index(flag) + 1], flag
    assert cmd[cmd.index("--plan") + 1] == RP.path(r) and wor == ["plan", "commit"]


def test_a_failed_plan_stops_the_row_and_leaves_no_plan(tmp_path, monkeypatch):
    from verity_vllm.pipeline.row_driver import Stop
    r, _ = _row_with_build(tmp_path, monkeypatch)
    _record_manifest(Path(r.d) / "manifest.json", R.QUERY_OF_RECORD)
    Path(RP.path(r)).parent.mkdir(parents=True)
    Path(RP.path(r)).write_text("{}")                                     # an earlier attempt's plan is not left for this Commit
    monkeypatch.setattr(r, "weights_of_record", lambda stage="commit": [])
    monkeypatch.setattr(r, "run_tl", lambda stage, to, log, cmd, *extra: 1)
    with pytest.raises(Stop) as e:
        RP.plan(r)
    assert e.value.rc == 15 and not Path(RP.path(r)).exists() and "plan FAIL" in (Path(r.d) / "stages.txt").read_text()


def test_a_failed_plan_that_is_not_required_leaves_no_plan_and_does_not_stop(tmp_path, monkeypatch):
    r, _ = _row_with_build(tmp_path, monkeypatch)
    _record_manifest(Path(r.d) / "manifest.json", R.QUERY_OF_RECORD)
    monkeypatch.setattr(r, "weights_of_record", lambda stage="commit": [])

    def run_tl(stage, to, log, cmd, *extra):
        Path(cmd[cmd.index("--out") + 1]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[cmd.index("--out") + 1]).write_text("{")                # a partial plan of a process that failed
        return 1

    monkeypatch.setattr(r, "run_tl", run_tl)
    RP.plan(r, required=False)
    assert not Path(RP.path(r)).exists() and "plan FAIL" in (Path(r.d) / "stages.txt").read_text()


@pytest.mark.parametrize("stages, config_run, planned", [("build", True, [False]), ("build,plan", True, [True]), ("build,commit", True, []),
                                                      ("plan", True, [True]), ("build", False, [])])
def test_a_config_runs_build_without_the_commit_or_its_own_plan_stage_ends_with_the_plan(tmp_path, monkeypatch, stages, config_run, planned):
    """A config run's Build task ahead of its Commit task derives the plan on CPU, so the Commit never derives its producer facts on the
    GPU; the plan is not required there (`required=False`)."""
    r, _ = _row_with_build(tmp_path, monkeypatch)
    r.stages, r.config_run, ran = stages, config_run, []
    for name in ("prepare", "build", "commit", "research_result", "timeline_report"):
        monkeypatch.setattr(r, name, lambda *a, **k: None)
    monkeypatch.setattr(RC, "word_check", lambda row: None)
    monkeypatch.setattr(RC, "config_record", lambda row: None)
    monkeypatch.setattr(RP, "plan", lambda row, required=True: ran.append(required))
    assert r.run() == 0 and ran == planned
