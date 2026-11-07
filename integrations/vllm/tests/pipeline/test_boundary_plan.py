"""The Commit's call-boundary plan (`pipeline.boundary_plan`): the Build's call-boundary check over the Programs the Commit reads, which on
a PASS writes the call-boundary source's host plan keyed as the producer-facts plan is; the Commit takes it only under its own key, else
derives it and records `plan_recomputed` with the reason; and the row passes it from the Build to the Commit.  Torch-free, on the
call-boundary source's dense stand-in Program."""
from __future__ import annotations

import copy
import json
import os
import types
from pathlib import Path

import pytest

from tests.acquire.call_boundary_programs import LP, PF, build_dir, dense_program
from verity_vllm.acquire.sources import call_boundary_plan as CBP
from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.pipeline import boundary_plan as BP
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import commit_plan as CP
from verity_vllm.pipeline import row_plan as RP
from verity_vllm.pipeline.timeline import Timeline
from tests.pipeline.test_row import _GNU_TIMEOUT, _built, _row


def _commit_args(tmp_path: Path, **kw) -> types.SimpleNamespace:
    """A B=1 row after its Build (the dense stand-in's Build dir and manifest.json) and the inputs its Commit reads."""
    b, _at = dense_program(**kw)
    pdir, man = build_dir(tmp_path, b.doc(LP), [PF[len("model."):]])
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    (tmp_path / "workload.json").write_text(json.dumps({"requests": [{"request_id": "r0", "prompt_token_ids": [1, 2], "max_tokens": 2}]}))
    return types.SimpleNamespace(workload=str(tmp_path / "workload.json"), required_manifest=str(tmp_path / "manifest.json"),
                                 program_digest="d0", program_dir=pdir, programs_root=None, call_boundary_plan=str(tmp_path / "cbp.json"))


def _plan(args, capsys) -> tuple[int, str, dict]:
    rc = cli.main(["boundary-plan", "--workload", args.workload, "--required-manifest", args.required_manifest, "--program-digest",
                   args.program_digest, "--program-dir", args.program_dir, "--gate-out", str(Path(args.workload).parent / "cb.json"),
                   "--out", args.call_boundary_plan])
    return rc, capsys.readouterr().out.strip().splitlines()[-1], json.loads((Path(args.workload).parent / "cb.json").read_text())


def _man(args) -> dict:
    return json.loads(Path(args.required_manifest).read_text())


def _tl(tmp_path):
    return Timeline(tmp_path / "timeline.jsonl")


def _ends(tmp_path) -> list[tuple]:
    return [(e["source"], e["identities"]) for e in map(json.loads, (tmp_path / "timeline.jsonl").read_text().splitlines()) if e["ev"] == "end"]


def test_the_check_is_the_gates_and_the_plan_is_keyed_as_the_commit_reads_it(tmp_path, capsys):
    """Over the Programs the Commit reads, the record, the line (wall time aside) and the exit code are `gate call-boundaries`'s over
    the row dir; the plan written beside it is the derived plan under the Commit's key."""
    args = _commit_args(tmp_path)
    rc, line, rec = _plan(args, capsys)
    grc = cli.main(["gate", "call-boundaries", str(tmp_path), "--out", str(tmp_path / "gate.json")])
    gline, grec = capsys.readouterr().out.strip().splitlines()[-1], json.loads((tmp_path / "gate.json").read_text())
    assert rc == grc == 0 and rec["pass"] is True and rec["covered"] == len(CB.manifest_sites(_man(args))) > 0
    assert {**rec, "seconds": 0} == {**grec, "seconds": 0} and line.split(" wall=")[0] == gline.split(" wall=")[0]
    doc = json.loads(Path(args.call_boundary_plan).read_text())
    assert doc["schema"] == BP.SCHEMA and doc["key"] == CP.key(args)
    assert doc["plan"] == json.loads(json.dumps(CBP.to_json(CB.plan_of(_man(args), args.program_dir))))


def test_a_check_that_fails_writes_no_plan_and_removes_an_earlier_one(tmp_path, capsys):
    args = _commit_args(tmp_path, embed=PF)
    Path(args.call_boundary_plan).write_text("{}")
    rc, line, rec = _plan(args, capsys)
    assert rc == 1 and rec["pass"] is False and rec["uncovered"] > 0 and "reads Program input 'prompt'" in line
    assert not Path(args.call_boundary_plan).exists()


def test_the_commit_takes_the_plan_under_its_own_key(tmp_path, capsys):
    args = _commit_args(tmp_path)
    _plan(args, capsys)
    derived = CB.plan_of(_man(args), args.program_dir)
    got, rec = BP.use(args, _man(args), tl=_tl(tmp_path))
    assert rec["used"] is True and rec["plan_recomputed"] is False and rec["key"] == CP.key(args) and rec["identities"] == derived.identities
    assert CBP.to_json(got) == CBP.to_json(derived) and "from the plan" in capsys.readouterr().out
    assert BP.verdict_entry(args, rec) == {"call_boundary_plan": rec} and _ends(tmp_path) == [("plan", derived.identities)]


@pytest.mark.parametrize("perturb, why", [
    (lambda a: Path(a.workload).write_text(Path(a.workload).read_text().replace('"max_tokens": 2', '"max_tokens": 3')), "workload_sha256"),
    (lambda a: Path(a.required_manifest).write_text(Path(a.required_manifest).read_text() + " "), "required_manifest_sha256"),
    (lambda a: setattr(a, "program_digest", "q" * 64), "program_digest"),
    (lambda a: Path(a.call_boundary_plan).unlink(), "plan unreadable: FileNotFoundError"),
    (lambda a: Path(a.call_boundary_plan).write_text(json.dumps({**json.loads(Path(a.call_boundary_plan).read_text()), "schema": "x"})),
     "plan schema 'x' != verity-vllm/call-boundary-plan/v1"),
])
def test_a_plan_the_commit_cannot_take_is_derived_with_the_reason(tmp_path, capsys, perturb, why):
    """Another workload, manifest or Program, no plan, another schema: the Commit derives the plan itself (the same plan) and records
    plan_recomputed naming why."""
    args = _commit_args(tmp_path)
    _plan(args, capsys)
    perturb(args)
    got, rec = BP.use(args, _man(args), tl=_tl(tmp_path))
    assert rec["used"] is False and rec["plan_recomputed"] is True and why in rec["why"] and "derive_s" in rec
    assert CBP.to_json(got) == CBP.to_json(CB.plan_of(_man(args), args.program_dir)) and _ends(tmp_path) == [("derived", got.identities)]
    assert f"plan_recomputed: {rec['why']}" in capsys.readouterr().out


def test_a_plan_that_does_not_load_is_derived_with_the_reason(tmp_path, capsys):
    """A Definition of another arity than the plan records (the registry changed under one code identity: never in practice)."""
    args = _commit_args(tmp_path)
    _plan(args, capsys)
    doc = json.loads(Path(args.call_boundary_plan).read_text())
    bad = copy.deepcopy(doc)
    spec = next(iter(bad["plan"]["definitions"]))
    bad["plan"]["definitions"][spec] += 1
    Path(args.call_boundary_plan).write_text(json.dumps(bad))
    got, rec = BP.use(args, _man(args), tl=_tl(tmp_path))
    assert got is not None and rec["plan_recomputed"] is True and rec["why"].startswith("plan does not load: ValueError(") and spec in rec["why"]


def test_a_manifest_naming_no_call_boundaries_reads_no_plan(tmp_path):
    args = _commit_args(tmp_path)
    man = {"identities": [{"family": "instance_outputs"}]}
    assert BP.use(types.SimpleNamespace(**{**vars(args), "call_boundary_plan": None}), man, tl=_tl(tmp_path)) == (None, None)
    got, rec = BP.use(args, man, tl=_tl(tmp_path))
    assert got is None and rec["used"] is False and rec["plan_recomputed"] is False and "names no call-boundary identity" in rec["why"]
    assert BP.use(args, None, tl=_tl(tmp_path))[0] is None and not (tmp_path / "timeline.jsonl").exists()
    assert BP.verdict_entry(types.SimpleNamespace(call_boundary_plan=None), None) == {}
    assert BP.verdict_entry(args, None)["call_boundary_plan"]["why"] == "no call-boundary plan read (no manifest or no committer)"


def test_the_commit_s_parser_takes_the_call_boundary_plan(tmp_path):
    pytest.importorskip("torch")
    assert cli.parse("commit", ["--out", str(tmp_path), "--call-boundary-plan", "p.json"], {}).call_boundary_plan == "p.json"
    o = cli.parse("boundary-plan", ["--workload", "w.json", "--required-manifest", "m.json", "--programs-root", "r", "--out", "o.json"], {})
    assert (o.workload, o.required_manifest, o.program_dir, o.programs_root, o.gate_out, o.out) == ("w.json", "m.json", None, "r", None, "o.json")


# ------------------------------------------------------------------------------------------------------------------ the row's Build
@pytest.mark.skipif(not _GNU_TIMEOUT, reason="GNU timeout (coreutils) required -- the pods have it, macOS does not")
def test_the_builds_check_leaves_the_plan_the_commit_stage_passes_under_the_commits_key(tmp_path, monkeypatch):
    """After a PASS Build the row's check is `boundary-plan` over the Programs the Commit reads: `call_boundaries.json` and the verdict
    line as before, and the plan keyed by the workload, manifest, Program digest and Programs the Commit stage names, which passes it."""
    r = _row(tmp_path, "--stages", "build,match")
    man = _built(r, monkeypatch)
    monkeypatch.setattr(r, "match", lambda: None)
    Path(RP.boundary_path(r)).write_text("{}")                                     # an earlier attempt's plan
    cmd = RP.call_boundaries_cmd(r)
    assert cmd[cmd.index("boundary-plan") + 1:] == ["--workload", r.wl, "--required-manifest", f"{r.d}/manifest.json", "--program-dir",
                                                    f"{r.d}/build_request", "--gate-out", f"{r.d}/call_boundaries.json", "--out", RP.boundary_path(r)]
    assert not Path(RP.boundary_path(r)).exists()
    assert r.run() == 0
    n = sum(1 for x in man["identities"] if x["family"] == "call_boundaries")
    gate = (Path(r.d) / "stages.txt").read_text().splitlines()[1].split(" ", 3)
    assert gate[:2] == ["build-call-boundaries", "PASS"] and gate[3].startswith(f"identities={n} claimed=0 covered={n} uncovered=0 ")
    assert json.loads((Path(r.d) / "call_boundaries.json").read_text())["covered"] == n
    doc = json.loads(Path(RP.boundary_path(r)).read_text())
    args = types.SimpleNamespace(workload=os.path.join(r.tree, r.wl), required_manifest=f"{r.d}/manifest.json", program_digest=None,
                                 program_dir=f"{r.d}/build_request", programs_root=None)
    assert doc["schema"] == BP.SCHEMA and doc["key"] == CP.key(args) and doc["plan"]["identities"] == n
    monkeypatch.setattr(r, "weights_of_record", lambda stage="commit": [])
    flags = RP.commit_flags(r)
    assert flags[flags.index("--call-boundary-plan") + 1] == RP.boundary_path(r)


def test_a_row_without_the_builds_programs_checks_the_row_dir(tmp_path, monkeypatch):
    r = _row(tmp_path)
    r.B = 1
    Path(RP.boundary_path(r)).write_text("{}")
    assert RP.call_boundaries_cmd(r) is None and not Path(RP.boundary_path(r)).exists()
    ran = []

    def run_tl(stage, to, lf, cmd, *extra):
        ran.append(cmd)
        Path(lf).write_text("line\n")
        return 0

    monkeypatch.setattr(r, "run_tl", run_tl)
    r.call_boundaries(60)
    assert ran[0][ran[0].index("gate"):] == ["gate", "call-boundaries", r.d, "--out", f"{r.d}/call_boundaries.json"]
    monkeypatch.setattr(r, "weights_of_record", lambda stage="commit": [])
    assert "--call-boundary-plan" not in RP.commit_flags(r)

