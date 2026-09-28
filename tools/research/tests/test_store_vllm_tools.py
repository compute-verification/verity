"""Phase 5a: the vLLM stage Tools (integrations/vllm/verity_vllm/pipeline/research_tools.py), typed outputs (`outputs.json`) and the
durability gate (`research data preserved`, store/preserved.py), against a LocalStore with the FsRemote."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from research.pythonpath import package_dirs
from research.store import attempt as A
from research.store import derivation as D
from research.store import preserved as P
from research.store import tools_registry
from research.store.cli import main as data_main
from research.store.kinds import KINDS
from research.store.local import LocalStore, attempt_key, object_key

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN_FIXTURE = HERE / "fixtures" / "run_v0.2"
RUN_ID = "r20260922-120000-abcd"
ROW = "llama32-1b__bf16__l40s__tp1__b1__i4096__o512__mixed__greedy__bi-eager"
CMD = ["verity-vllm", "row", "stage"]
CONDS = {"hw.family": "l40s", "tool.vllm.build.torch": "2.9.0", "tool.vllm.build.vllm": "0.11.0",
         "tool.vllm.match.torch": "2.9.0", "tool.vllm.match.vllm": "0.11.0", "tool.vllm.commit.torch": "2.9.0", "tool.vllm.commit.vllm": "0.11.0"}


def _argv(stage: str, *flags: str, row: str = ROW) -> list[str]:
    return [*CMD, stage, row, "LLAMA32_1B", "meta-llama/Llama-3.2-1B", "4e20da8c", *flags]


def _drv(tool, argv: list[str], inputs: dict | None = None) -> str:
    _, key = tool.resolve_params(argv, REPO)
    return D.derivation_id(tool.identity(), inputs or {}, key, CONDS, tool.key_conditions)


# ---- Tool declarations -------------------------------------------------------------------------------------------------------------------------

def test_vllm_tools_are_registered_and_loadable_by_name():
    for stage in ("build", "plan", "match", "commit", "replay"):
        t = tools_registry.load(f"vllm.{stage}", REPO)
        assert t.name == f"vllm.{stage}" and t.spec == f"integrations.vllm.verity_vllm.pipeline.research_tools:VLLM_{stage.upper()}"
        assert t.command == [*CMD, stage]
        assert t.key_conditions == ["hw.family", f"tool.vllm.{stage}.torch", f"tool.vllm.{stage}.vllm"]
        assert t.closure == ["integrations/vllm/**", *(f"{d}/**" for d in package_dirs(REPO, "verity"))]
        for k in t.produces:
            assert k in KINDS, k
        assert "execution-evidence/v1" in t.produces and "run-files/v1" in t.produces
    assert tools_registry.load("vllm.build", REPO).produces[0] == "vllm-build/v1"
    assert tools_registry.load("vllm.plan", REPO).produces[0] == "vllm-commit-plan/v1"
    assert tools_registry.load("vllm.match", REPO).produces[0] == "vllm-match-record/v1"
    assert tools_registry.load("vllm.commit", REPO).produces[0] == "vllm-verdict/v1"
    assert tools_registry.load("vllm.replay", REPO).produces[0] == "vllm-verdict/v1"


def test_row_configuration_is_the_key_and_retain_is_not():
    t = tools_registry.load("vllm.build", REPO)
    params, key = t.resolve_params(_argv("build", "--retain", "host", "--build-jobs", "4"), REPO)
    assert key["row"] == ROW and key["model"] == "meta-llama/Llama-3.2-1B" and key["revision"] == "4e20da8c" and key["role"] == "LLAMA32_1B"
    assert (key["model_tag"], key["dtype"], key["hw"], key["tp"], key["batch"], key["input"], key["output"]) == ("llama32-1b", "bf16", "l40s", 1, 1, 4096, 512)
    assert (key["arrivals"], key["sampling"], key["exec"], key["query"]) == ("mixed", "greedy", "bi-eager", "Q_word_v1{X=16,W=32,R=no-recompute}")
    assert "retain" not in key and "build_jobs" not in key, "footprint / schedule knobs are not key params"
    assert params["retain"] == "host" and params["build_jobs"] == "4", "...but they are recorded"
    assert params["argv"][:3] == CMD


def test_derivation_stable_under_retain_changes_under_batch():
    t = tools_registry.load("vllm.build", REPO)
    base = _drv(t, _argv("build"))
    assert base == _drv(t, _argv("build", "--retain", "host")), "RETAIN mode does not change the derivation"
    assert base == _drv(t, _argv("build", "--retain", "gpu", "--build-jobs", "8", "--sweep-dir", "/tmp/x", "--gpu-util", "0.5"))
    assert base != _drv(t, _argv("build", row=ROW.replace("__b1__", "__b8__"))), "batch is part of the row configuration"
    assert base != _drv(t, _argv("build", row=ROW.replace("__i4096__", "__i2048__")))
    assert base != _drv(t, _argv("build", "--query", "Q_other"))
    assert base != _drv(t, [*CMD, "build", ROW, "LLAMA32_1B", "meta-llama/Llama-3.2-1B", "otherrev"]), "the revision pins the checkpoint"
    other = dict(CONDS, **{"hw.family": "h100"})
    _, key = t.resolve_params(_argv("build"), REPO)
    assert base != D.derivation_id(t.identity(), {}, key, other, t.key_conditions), "hw.family is a key condition"
    assert base != D.derivation_id(t.identity(), {}, key, dict(CONDS, **{"tool.vllm.build.vllm": "0.12.0"}), t.key_conditions), "the vllm pin is"
    assert base == D.derivation_id(t.identity(), {}, key, dict(CONDS, **{"hw.sku": "NVIDIA L40S", "exec.campaign": "c1"}), t.key_conditions)


def test_stage_flags_that_change_outputs_are_key_with_explicit_defaults():
    m = tools_registry.load("vllm.match", REPO)
    assert _drv(m, _argv("match")) == _drv(m, _argv("match", "--within-step", "dag", "--snapshot-steps", "row-default", "--match-impl", "fast"))
    assert _drv(m, _argv("match")) != _drv(m, _argv("match", "--snapshot-steps", "all"))
    assert _drv(m, _argv("match")) != _drv(m, _argv("match", "--within-step", "sequence"))
    c = tools_registry.load("vllm.commit", REPO)
    assert _drv(c, _argv("commit")) == _drv(c, _argv("commit", "--pairs", "3", "--retain", "host", "--window-mb", "2048"))
    assert _drv(c, _argv("commit")) != _drv(c, _argv("commit", "--pairs", "5"))
    assert _drv(c, _argv("commit")) != _drv(c, _argv("commit", "--extra-engine-args", "max_num_batched_tokens=512"))
    assert _drv(c, _argv("commit")) != _drv(c, _argv("commit", "--replay-deferred", "1")), "a deferred Commit's outputs differ"
    assert _drv(c, _argv("commit")) == _drv(c, _argv("commit", "--replay-deferred", "0"))
    assert "replay_deferred" not in c.resolve_params(_argv("commit"))[1], "off, it leaves the derivation of every earlier Commit as it was"
    build_art = "art:" + "1" * 64
    assert _drv(c, _argv("commit")) != _drv(c, _argv("commit"), inputs={"build": build_art}), "the consumed Build is part of the derivation"
    assert _drv(m, _argv("match")) != _drv(c, _argv("commit")), "different tools, different derivations"
    r = tools_registry.load("vllm.replay", REPO)
    commit_art = "art:" + "2" * 64
    assert _drv(r, _argv("replay"), inputs={"build": build_art, "commit": commit_art}) != \
        _drv(r, _argv("replay"), inputs={"build": build_art, "commit": "art:" + "3" * 64}), "the replayed Commit is part of the derivation"


def test_parse_accepts_the_module_form_row_run_and_the_earlier_runner_argvs(monkeypatch):
    # Only for this test: left on sys.path, the repo root makes a later test's `--tool check` import the repo's own
    # tools/verity/check/tool.py instead of its fixture's, outside the suite's inputs.
    monkeypatch.syspath_prepend(str(REPO))
    from integrations.vllm.verity_vllm.pipeline import research_tools as RT

    m = RT.parse(["/workspace/venv312/bin/python", "-m", "verity_vllm.pipeline.cli", "row", "stage", "commit", ROW, "R", "repo", "rev",
                  "--pairs", "4"], stage="commit")
    assert m["stage"] == "commit" and m["row"] == ROW and m["pairs"] == "4"
    r = RT.parse(["verity-vllm", "row", "run", ROW, "LLAMA32_1B", "repo", "rev", "--stages", "build,match"], stage="build")
    assert r["stage"] == "build,match" and r["row"] == ROW and r["model"] == "repo" and "stages" not in r
    # the earlier shell runners, as records in the store carry them
    p = RT.parse(["bash", "/workspace/src/integrations/vllm/verity_vllm/ops/row_pod.sh", ROW, "LLAMA32_1B", "repo", "rev", "build,match"], stage="build")
    assert p["stage"] == "build,match" and p["row"] == ROW and p["model"] == "repo" and p["batch"] == 1
    q = RT.parse(["bash", "integrations/vllm/verity_vllm/ops/run_row_v2.sh", "stage", "commit", ROW, "R", "repo", "rev", "--pairs=4",
                  "--required-classes", "module_out"], stage="commit")
    assert q["pairs"] == "4" and q["required_classes"] == "module_out" and q["replay_args"] is None
    # a value that is itself a flag list is a value (the R19 admit fix (a) commit flags), not a bare `--commit-args` switch
    c = RT.parse(_argv("commit", "--commit-args", "--bounded-staging --retain-exclude hidden_m1", "--pairs", "3"), stage="commit")
    assert c["commit_args"] == "--bounded-staging --retain-exclude hidden_m1" and c["pairs"] == "3"
    assert RT.parse(_argv("commit", "--commit-args", "--pairs", "3"), stage="commit")["commit_args"] is True   # a real switch stays one
    assert RT.row_config("not-a-row") == {**{k: None for k in RT._ROW_FIELDS}, "model_tag": "not-a-row"}


# ---- typed outputs + the durability gate -------------------------------------------------------------------------------------------------------

@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    monkeypatch.setenv("RESEARCH_REPO", str(tmp_path / "repo"))
    (tmp_path / "repo").mkdir()
    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    return tmp_path


def _run_with_outputs(tmp_path: Path, build_ref: str | None = None) -> Path:
    d = tmp_path / "runs" / RUN_ID
    shutil.copytree(RUN_FIXTURE, d)
    (d / "state.json").rename(d / "status.json")
    (d / "outputs" / "build").mkdir(parents=True)
    (d / "outputs" / "build" / "build_summary.json").write_text(json.dumps({"program_digest": "sha256:" + "9" * 64}))
    (d / "outputs" / "build" / "required_values.json").write_text('{"classes": ["module_out"]}')
    (d / "evidence").mkdir()
    (d / "evidence" / "timeline.jsonl").write_text('{"stage": "build", "state": "done"}\n')
    (d / "logs").mkdir()
    (d / "logs" / "build.log").write_text("ok\n")
    refs = {"build": build_ref} if build_ref else {}
    doc = {"schema": A.OUTPUTS_SCHEMA, "outputs": [
        {"name": "build", "kind": "vllm-build/v1", "meta": {"row": ROW, "program_digest": "sha256:" + "9" * 64}, "tree": "outputs/build", "refs": refs},
        {"name": "evidence", "kind": "execution-evidence/v1", "meta": {"row": ROW, "stage": "build"}, "file": "evidence/timeline.jsonl"},
        {"name": "logs", "kind": "run-files/v1", "files": ["logs", "missing.log"], "refs": {"build": "@build"}},
        {"name": "absent", "kind": "vllm-verdict/v1", "file": "nope.json", "optional": True},
    ]}
    (d / "outputs.json").write_text(json.dumps(doc))
    return d


def test_publish_typed_outputs_and_preserve_on_fs_remote(env):
    d = _run_with_outputs(env)
    st = LocalStore()
    out = A.publish(d, store=st)
    assert out["published"] is True and out.get("already") is None, out
    o = out["outputs"]
    assert set(o) == {"result", "run_files", "build", "evidence", "logs"} and all(o.values()), o
    b = st.get_manifest(o["build"])
    assert b.kind == "vllm-build/v1" and b.meta["program_digest"] == "sha256:" + "9" * 64 and b.meta["run_id"] == RUN_ID
    assert sorted(f.path for f in b.payload.files) == ["build_summary.json", "required_values.json"]
    e = st.get_manifest(o["evidence"])
    assert e.kind == "execution-evidence/v1" and e.payload.type == "file" and e.payload.name == "timeline.jsonl"
    lg = st.get_manifest(o["logs"])
    assert lg.kind == "run-files/v1" and [f.path for f in lg.payload.files] == ["logs/build.log"]
    assert lg.refs == {"build": o["build"]}, "@name refs resolve to the earlier output's id"
    assert lg.meta["files"] == 1 and lg.meta["missing"] == ["missing.log"]
    att = st.get_attempt(RUN_ID)
    assert att["outputs"] == o and "absent" not in att["outputs"]
    assert out["preserved"] is True and all(out["preserved_artifacts"].values()), out
    # the gate: every named artifact + the attempt manifest are verifiably on the remote
    rep = P.check(st, [RUN_ID])
    assert rep["preserved"] is True and rep["not_preserved"] == [] and rep["targets"][0]["attempt_manifest"] is True
    assert rep["targets"][0]["n_artifacts"] == 6, "5 outputs + events telemetry"
    assert all(a["preserved"] for a in rep["artifacts"].values())
    assert P.check(st, [o["build"]])["preserved"] is True
    assert P.check(st, [RUN_ID], mode="recorded")["preserved"] is True, "push recorded verified replicas"
    assert P.check(st, [RUN_ID], mode="readback")["preserved"] is True
    assert P.check(st, [])["preserved"] is False, "no target, nothing is vouched for"


def test_preserved_lists_what_is_missing_and_the_cli_exits_nonzero(env, capsys):
    d = _run_with_outputs(env)
    st = LocalStore()
    out = A.publish(d, store=st)
    o = out["outputs"]
    vault = env / "vault"
    b = st.get_manifest(o["build"])
    victim = b.blobs()[0][0]
    (vault / object_key(victim)).unlink()                        # one blob of the build tree disappears from the remote
    rep = P.check(st, [RUN_ID])
    assert rep["preserved"] is False
    bad = {x["id"]: x for x in rep["not_preserved"]}
    assert set(bad) == {o["build"]} and bad[o["build"]]["kind"] == "vllm-build/v1" and "absent" in bad[o["build"]]["errors"][0]
    assert rep["targets"][0]["missing_names"] == ["outputs.build"]
    assert P.check(st, [o["evidence"]])["preserved"] is True, "the other artifacts are still preserved"
    assert P.check(st, [RUN_ID], mode="recorded")["preserved"] is True, "the offline mode trusts the catalog: it cannot see the loss"
    # the attempt manifest itself missing from the remote is not preserved either
    (vault / object_key(victim)).write_bytes(st.object_path(victim).read_bytes())
    assert P.check(st, [RUN_ID])["preserved"] is True
    (vault / attempt_key(RUN_ID)).unlink()
    rep = P.check(st, [RUN_ID])
    assert rep["preserved"] is False and rep["targets"][0]["attempt_manifest"] is False and rep["not_preserved"][0]["id"] == RUN_ID
    st.push_attempt(RUN_ID)
    # a wrong-size remote object is caught, not just absence
    ev = st.get_manifest(o["evidence"]).blobs()[0][0]
    (vault / object_key(ev)).write_bytes(b"truncated")
    rep = P.check(st, [o["evidence"]])
    assert not rep["preserved"] and "remote size" in rep["not_preserved"][0]["errors"][0]
    # CLI: exit 1 with the list, exit 0 once repaired, exit 2 without targets
    assert data_main(["preserved", RUN_ID]) == 1
    text = capsys.readouterr().out
    assert "NOT PRESERVED: 1 item(s)" in text and "outputs.evidence" in text and "remote size" in text
    assert data_main(["push", o["evidence"]]) == 1, "immutable keys: a conflicting remote object is reported, never overwritten"
    capsys.readouterr()
    (vault / object_key(ev)).unlink()                            # the operator removes the bad object; push is then the repair
    assert data_main(["push", o["evidence"]]) == 0
    capsys.readouterr()
    assert data_main(["preserved", RUN_ID, o["build"]]) == 0
    text = capsys.readouterr().out
    assert text.strip().splitlines()[-1].startswith("PRESERVED on ")
    assert data_main(["preserved"]) == 2
    assert data_main(["preserved", "not-an-id"]) == 1
    capsys.readouterr()
    assert data_main(["preserved", RUN_ID, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["preserved"] is True
    assert data_main(["preserved", "r20260101-000000-zzzz"]) == 1, "an attempt the store does not know is not preserved"


def test_preserved_without_a_remote_says_so(env, monkeypatch):
    (env / "store.toml").unlink()
    d = _run_with_outputs(env)
    st = LocalStore()
    out = A.publish(d, store=st)
    assert out["published"] and out["preserved"] is False
    rep = P.check(st, [RUN_ID])
    assert rep["preserved"] is False and rep["remote"] is None and "no remote configured" in rep["reason"]
    assert all("no remote configured" in x["errors"][0] for x in rep["not_preserved"])
    lines = P.format_report(rep)
    assert lines[-1].startswith("NOT PRESERVED") and any("no remote configured" in ln for ln in lines)


def test_outputs_json_errors_are_publish_errors_not_crashes(env):
    d = _run_with_outputs(env)
    doc = json.loads((d / "outputs.json").read_text())
    doc["outputs"].append({"name": "verdict", "kind": "vllm-verdict/v1", "file": "nope.json"})   # required, absent
    (d / "outputs.json").write_text(json.dumps(doc))
    st = LocalStore()
    out = A.publish(d, store=st)
    assert out["published"] is False and "nope.json" in out["error"] and st.get_attempt(RUN_ID) is None
    assert (d / "store_publish_error.json").exists()
    for bad in ({"name": "x", "kind": "notakind", "file": "evidence/timeline.jsonl"},
                {"name": "x", "kind": "run-files/v1", "file": "../outside"},
                {"name": "x", "kind": "run-files/v1", "refs": {"b": "@nothing"}},
                {"name": "build", "kind": "run-files/v1"}):
        doc["outputs"] = doc["outputs"][:3] + [bad]
        (d / "outputs.json").write_text(json.dumps(doc))
        assert A.publish(d, store=st)["published"] is False, bad
    (d / "outputs.json").write_text(json.dumps({"schema": "other/v9", "outputs": []}))
    assert A.publish(d, store=st)["published"] is False


@pytest.fixture
def no_dirty_digest(monkeypatch):
    """This checkout's source identity without its dirty digest, which reads every untracked file of the working tree: the
    derivations these tests compare don't depend on it, and a suite may not read files outside its inputs."""
    from research import env as E
    monkeypatch.setattr(E, "_dirty_digest", lambda root, capture=None: None)


PINNED = {"hw.family": "l40s", "torch": "2.9.0", "vllm": "0.11.0"}


@pytest.fixture
def pinned_probes(monkeypatch):
    """The probes behind the key conditions answer as an L40S host with the workload interpreter's torch and vllm: `hw.family`
    (nvidia-smi) and the vLLM Tools' own `tool.vllm.<stage>.torch|vllm` (a `$PY` subprocess). Each probe gives None when it
    times out, so on a loaded machine one of two identical requests could see None and get another derivation (train T640's
    `check` on vy-nebius-1, 1 Oct 2026). Pinned to values, not None, so a caller that skips a probe gets another derivation."""
    from research import env as E
    monkeypatch.setattr(E, "_nvidia_smi", lambda: {"hw.sku": "NVIDIA L40S", "hw.count": 1, "hw.vram_mb": 46068, "rt.driver": None, "rt.cuda": None})
    for stage in ("build", "match", "commit", "replay"):
        monkeypatch.setattr(tools_registry.load(f"vllm.{stage}", REPO), "conditions", lambda env: {"torch": PINNED["torch"], "vllm": PINNED["vllm"]})


def _pinned(stage: str) -> dict:
    return {"hw.family": PINNED["hw.family"], f"tool.vllm.{stage}.torch": PINNED["torch"], f"tool.vllm.{stage}.vllm": PINNED["vllm"]}




@pytest.mark.usefixtures("no_dirty_digest", "pinned_probes")
def test_lookup_computes_the_derivation_the_runner_records(tmp_path, monkeypatch, capsys):
    """`research data lookup --tool vllm.build -- ARGV` fills the Tool's own key conditions (`tool.vllm.build.torch|vllm`, its probe) the
    way `research run --tool` records them, so the derivation it looks up is the one an attempt was published under."""
    from research import cli as RC
    import argparse

    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "absent.toml"))
    cmd = _argv("build", "--retain", "host")
    ns = argparse.Namespace(tool="vllm.build", campaign="c", project="verity", exclusive_lock=False, source_identity=None, scratch=[])
    sec = RC._prepare_store(ns, argparse.Namespace(cwd=str(REPO), input=[]), cmd, "r20260922-000000-0004", tmp_path / "run")["section"]
    assert "error" not in sec, sec
    assert set(sec["conditions"]) >= {"tool.vllm.build.torch", "tool.vllm.build.vllm", "hw.family"}
    rc = data_main(["lookup", "--tool", "vllm.build", "--source-root", str(REPO), "--any-source", "--json", "--", *cmd])
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["derivation"] == sec["derivation"], (doc["key_conditions"], {k: sec["conditions"].get(k) for k in doc["key_conditions"]})
    assert doc["key_conditions"] == _pinned("build"), "lookup fills hw.family and the Tool's own probe"
    assert doc["attempts"] == [] and doc["key_params"]["row"] == ROW


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
