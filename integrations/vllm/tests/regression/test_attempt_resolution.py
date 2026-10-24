"""Phase 5a: the harness resolves a v2 candidate from Attempt ids (`VERITY_REGRESSION_CANDIDATE=r2026…,…`) -- `store_io.attempt` /
`attempt_outputs` / `fetch_attempt`, `resolver.Row.candidate_root()` as the union of the attempts' typed output trees, and the
`attempt_provenance` check (the re-run cites the fixture row's configuration; Match cites the Build artifact, Commit cites both).

Self-contained: three fake stage Attempts are published into a temporary LocalStore (the `research` package of this checkout,
`tools/research/src`; skipped when absent), so it needs neither VERITY_REGRESSION=1, a GPU, nor R2.  Run from integrations/vllm:
    python -m pytest tests/regression/test_attempt_resolution.py -q
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from . import resolver, store_io
from .checks import attempt_provenance as AP

RESEARCH_SRC = store_io.REPO_ROOT / "tools" / "research" / "src"
RUN_FIXTURE = store_io.REPO_ROOT / "tools" / "research" / "tests" / "fixtures" / "run_v0.2"
ROW = "llama32-1b__bf16__l40s__tp1__b1__i4096__o512__mixed__greedy__bi-eager"
IDS = {"build": "r20260923-060930-553f", "match": "r20260923-071000-0a1b", "commit": "r20260923-072000-0c2d"}
PD = "9ce48951f04eb440501c4e829e3a492a58ca963997ca707d47fb129fe73cf590"
KEY_PARAMS = ["stage", "row", "role", "model", "revision", "model_tag", "dtype", "hw", "tp", "batch", "input", "output", "arrivals",
              "sampling", "exec", "query", "lp", "t"]
CONFIG = {"row": ROW, "role": "LLAMA32_1B", "model": "unsloth/Llama-3.2-1B", "revision": "9535bd9b1d1dea6acafbdc4813b728796aeb28da",
          "model_tag": "llama32-1b", "dtype": "bf16", "hw": "l40s", "tp": 1, "batch": 1, "input": 4096, "output": 512, "arrivals": "mixed",
          "sampling": "greedy", "exec": "bi-eager", "query": "Q_word_v1{X=16,W=32,R=no-recompute}", "lp": None, "t": None}

pytestmark = pytest.mark.skipif(not (RESEARCH_SRC / "research" / "store").is_dir() or not RUN_FIXTURE.is_dir(),
                                reason="needs tools/research/src (research.store) and its run_v0.2 fixture in this checkout")


def _research():
    if str(RESEARCH_SRC) not in sys.path:
        sys.path.insert(0, str(RESEARCH_SRC))
    from research.store import attempt as A
    from research.store.local import LocalStore
    return A, LocalStore


def _stage_run(tmp: Path, stage: str, *, params: dict, inputs: dict, files: dict[str, str], kind: str, meta: dict,
               evidence: str = '{"stage": "%s"}\n') -> Path:
    """A run dir for one stage: the v0.2 fixture re-identified as IDS[stage], with `outputs.json` declaring the stage's typed outputs."""
    A, _ = _research()
    rid = IDS[stage]
    d = tmp / "runs" / rid
    shutil.copytree(RUN_FIXTURE, d)
    (d / "state.json").rename(d / "status.json")
    job = json.loads((d / "job.json").read_text())
    job["run"]["id"] = rid
    job["ownership"]["job"] = rid
    job["store"] = {"tool": {"name": f"vllm.{stage}", "version": "1", "spec": f"integrations.vllm.verity_vllm.pipeline.research_tools:VLLM_{stage.upper()}"},
                    "key_params": KEY_PARAMS + (["pairs"] if stage == "commit" else []), "key_conditions": ["hw.family"],
                    "params": params, "key_params_resolved": {k: params.get(k) for k in KEY_PARAMS}, "inputs": inputs, "closure": {},
                    "campaign": "test"}
    (d / "job.json").write_text(json.dumps(job))
    out = d / "outputs" / stage
    for rel, text in files.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    (d / "evidence").mkdir()
    (d / "evidence" / "timeline.jsonl").write_text(evidence % stage)
    doc = {"schema": A.OUTPUTS_SCHEMA, "outputs": [
        {"name": {"build": "build", "match": "match", "commit": "verdict"}[stage], "kind": kind, "meta": meta, "tree": f"outputs/{stage}",
         "refs": {k: v for k, v in inputs.items()}},
        {"name": "evidence", "kind": "execution-evidence/v1", "meta": {"row": ROW, "stage": stage}, "file": "evidence/timeline.jsonl"},
    ]}
    (d / "outputs.json").write_text(json.dumps(doc))
    return d


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Three published stage Attempts (build -> match -> commit) of ROW in a temporary LocalStore; returns {stage: attempt manifest}."""
    A, LocalStore = _research()
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.delenv(resolver.ROWS_ROOT_ENV, raising=False)
    monkeypatch.setenv(resolver.CANDIDATE_CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(resolver, "_ATTEMPT_DOCS", {})
    st = LocalStore()
    p_build = {**CONFIG, "stage": "build", "retain": "host", "build_jobs": "auto"}
    b = A.publish(_stage_run(tmp_path, "build", params=p_build, inputs={}, kind="vllm-build/v1", meta={"row": ROW, "program_digest": PD},
                             files={"build_summary.json": json.dumps({"program_digest": PD}), "build_request/required_values.json": "{}"}), store=st)
    assert b["published"], b
    build_art = b["outputs"]["build"]
    p_match = {**CONFIG, "stage": "match", "retain": "none", "snapshot_steps": "0,1", "within_step": "dag"}
    m = A.publish(_stage_run(tmp_path, "match", params=p_match, inputs={"build": build_art}, kind="vllm-match-record/v1",
                             meta={"row": ROW, "program_digest": PD, "match_ok": True},
                             files={"global_match.json": json.dumps({"ok": True, "checks": {"tokens": "PASS"}}),
                                    "match/instances.jsonl": "{}\n", "match/versions.json": "{}"}), store=st)
    assert m["published"], m
    match_art = m["outputs"]["match"]
    p_commit = {**CONFIG, "stage": "commit", "pairs": "3", "retain": "none"}
    c = A.publish(_stage_run(tmp_path, "commit", params=p_commit, inputs={"build": build_art, "match": match_art}, kind="vllm-verdict/v1",
                             meta={"row": ROW, "verdict": "PASS"},
                             files={"commit/commit_verdict.json": json.dumps({"verdict": "PASS"}), "commit/commit_summary.json": "{}"}), store=st)
    assert c["published"], c
    return {"build": st.get_attempt(IDS["build"]), "match": st.get_attempt(IDS["match"]), "commit": st.get_attempt(IDS["commit"]),
            "arts": {"build": build_art, "match": match_art, "verdict": c["outputs"]["verdict"]}, "tmp": tmp_path}


def _row() -> resolver.Row:
    return next(r for r in resolver.rows() if r.key == ROW)


# ---- store_io ----------------------------------------------------------------------------------------------------------------------------------

def test_store_io_reads_attempts_and_their_outputs_by_kind(store):
    assert store_io.is_attempt_id(IDS["build"]) and not store_io.is_attempt_id("art:" + "0" * 64) and not store_io.is_attempt_id("")
    with pytest.raises(ValueError):
        store_io.attempt("not-an-id")
    doc = store_io.attempt(IDS["match"])
    assert doc["id"] == IDS["match"] and doc["tool"]["name"] == "vllm.match" and doc["inputs"] == {"build": store["arts"]["build"]}
    outs = store_io.attempt_outputs(IDS["match"])
    assert outs["match"]["id"] == store["arts"]["match"] and outs["match"]["kind"] == "vllm-match-record/v1"
    assert outs["match"]["payload"] == {"type": "tree", "name": None} and outs["match"]["meta"]["program_digest"] == PD
    assert outs["evidence"]["kind"] == "execution-evidence/v1" and outs["evidence"]["payload"] == {"type": "file", "name": "timeline.jsonl"}
    assert set(outs) >= {"match", "evidence"}
    only = store_io.attempt_outputs(IDS["match"], ("vllm-match-record/v1",))
    assert list(only) == ["match"]
    assert store_io.attempt_outputs(IDS["build"], store_io.ROW_KINDS).keys() == {"build", "evidence"}


def test_fetch_attempt_unions_the_stage_trees_like_a_row_dir(store, tmp_path):
    to = store_io.fetch_attempt([IDS["build"], IDS["match"], IDS["commit"]], tmp_path / "union")
    rels = sorted(str(p.relative_to(to)) for p in to.rglob("*") if p.is_file())
    assert rels == ["ATTEMPTS.json", "build_request/required_values.json", "build_summary.json", "commit/commit_summary.json",
                    "commit/commit_verdict.json", "evidence/build/timeline.jsonl", "evidence/commit/timeline.jsonl",
                    "evidence/match/timeline.jsonl", "global_match.json", "match/instances.jsonl", "match/versions.json"], rels
    assert (to / "evidence" / "match" / "timeline.jsonl").read_text() == '{"stage": "match"}\n', "each stage's evidence, by stage"
    rec = json.loads((to / "ATTEMPTS.json").read_text())
    assert set(rec["attempts"]) == set(IDS.values()) and rec["conflicts"] == []
    # an account file two attempts carry with different bytes is a conflict, named -- never a silent overwrite
    A, LocalStore = _research()
    st = LocalStore()
    d = _stage_run(store["tmp"] / "again", "match", params={**CONFIG, "stage": "match"}, inputs={}, kind="vllm-match-record/v1",
                   meta={"row": ROW}, files={"global_match.json": json.dumps({"ok": False}), "match/versions.json": "{}"})
    job = json.loads((d / "job.json").read_text())
    job["run"]["id"] = job["ownership"]["job"] = "r20260923-080000-ffff"
    (d / "job.json").write_text(json.dumps(job))
    assert A.publish(d, store=st)["published"]
    with pytest.raises(store_io.StoreUnavailable, match="differ between attempts"):
        store_io.fetch_attempt([IDS["match"], "r20260923-080000-ffff"], tmp_path / "conflict")
    rec = json.loads((tmp_path / "conflict" / "ATTEMPTS.json").read_text())
    assert [c["path"] for c in rec["conflicts"]] == ["global_match.json"], "identical bytes (match/versions.json, the evidence) are not"


# ---- resolver ----------------------------------------------------------------------------------------------------------------------------------

def test_candidate_env_names_attempt_ids_or_a_directory(monkeypatch, tmp_path):
    monkeypatch.setenv(resolver.CANDIDATE_ENV, f"{IDS['build']}, {IDS['match']}\n{IDS['commit']}")
    assert resolver.candidate_attempt_ids() == [IDS["build"], IDS["match"], IDS["commit"]]
    monkeypatch.setenv(resolver.CANDIDATE_ENV, str(tmp_path))
    assert resolver.candidate_attempt_ids() == []
    monkeypatch.setenv(resolver.CANDIDATE_ENV, f"{IDS['build']},{tmp_path}")
    assert resolver.candidate_attempt_ids() == [], "a mixed list is neither: treated as a directory (which will not resolve)"
    monkeypatch.delenv(resolver.CANDIDATE_ENV)
    assert resolver.candidate_attempt_ids() == []
    assert _row().candidate_root() is None and _row().candidate("global_match.json") is None


def test_naming_the_commit_attempt_resolves_the_chain_behind_it(store, monkeypatch):
    """`VERITY_REGRESSION_CANDIDATE=<commit attempt>`: its inputs cite the Match and Build artifacts, whose `meta.run_id` (stamped by
    `typed_outputs`) names the attempts that published them -- all three become the row's candidate attempts."""
    monkeypatch.setenv(resolver.CANDIDATE_ENV, IDS["commit"])
    assert [d["id"] for d in resolver.candidate_attempt_docs()] == [IDS["commit"], IDS["build"], IDS["match"]]
    row = _row()
    assert sorted(d["id"] for d in row.candidate_attempts()) == sorted(IDS.values())
    root = row.candidate_root()
    assert root is not None and row.candidate("global_match.json") is not None and row.candidate("build_summary.json") is not None
    assert row.candidate("commit/commit_verdict.json") is not None


@pytest.mark.slow
def test_row_candidate_is_the_union_of_its_attempts(store, monkeypatch):
    monkeypatch.setenv(resolver.CANDIDATE_ENV, ",".join(IDS.values()))
    row = _row()
    assert [d["id"] for d in row.candidate_attempts()] == list(IDS.values())
    root = row.candidate_root()
    assert root is not None and root == store["tmp"] / "cache" / ROW
    assert row.candidate("global_match.json") == root / "global_match.json"
    assert json.loads(row.candidate("build_summary.json").read_text())["program_digest"] == PD
    assert row.candidate("commit/commit_verdict.json").is_file() and row.candidate("nope.json") is None
    # cached: a second call does not refetch; a different attempt set does
    (root / "sentinel").write_text("x")
    assert row.candidate_root() == root and (root / "sentinel").exists()
    monkeypatch.setenv(resolver.CANDIDATE_ENV, IDS["build"])
    assert row.candidate_root() == root and not (root / "sentinel").exists() and not (root / "global_match.json").exists()
    # another row: none of these attempts name it
    other = next(r for r in resolver.rows() if r.key != ROW)
    assert other.candidate_attempts() == [] and other.candidate_root() is None


def test_unreadable_attempt_is_not_resolvable_here(monkeypatch, tmp_path):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "empty-store"))
    monkeypatch.setattr(resolver, "_ATTEMPT_DOCS", {})
    monkeypatch.setenv(resolver.CANDIDATE_ENV, "r20260101-000000-dead")
    with pytest.raises(resolver.NotResolvable):
        _row().candidate_attempts()


# ---- the attempt_provenance check ------------------------------------------------------------------------------------------------------------

def test_attempt_provenance_reproduces_when_the_rerun_cites_the_fixture_row(store, monkeypatch):
    monkeypatch.setenv(resolver.CANDIDATE_ENV, ",".join(IDS.values()))
    row = _row()
    assert AP.applies(row) is None
    exp = AP.expected(row)
    assert exp["config"]["model"] == "unsloth/Llama-3.2-1B" and exp["config"]["tp"] == 1 and exp["config"]["batch"] == 1
    assert exp["config"]["query"] == "Q_word_v1{X=16,W=32,R=no-recompute}" and exp["chain_consistent"] is True
    act = AP.actual(row)
    assert resolver.diff(exp, act) == [], resolver.diff(exp, act)
    assert act["chain_problems"] == [] and act["stages"]["match"]["inputs"] == {"build": store["arts"]["build"]}
    assert act["stages"]["commit"]["inputs"] == {"build": store["arts"]["build"], "match": store["arts"]["match"]}


def test_attempt_provenance_names_a_different_configuration_and_a_broken_chain(store, monkeypatch):
    monkeypatch.setenv(resolver.CANDIDATE_ENV, ",".join(IDS.values()))
    row = _row()
    exp = AP.expected(row)
    # a re-run at batch 8 with another revision is not this row's re-run
    docs = resolver.candidate_attempt_docs()
    docs[0]["params"] = {**docs[0]["params"], "batch": 8, "revision": "deadbeef"}
    act = AP.actual(row)
    problems = resolver.diff(exp, act)
    assert any(".config.batch" in p for p in problems) and any(".config.revision" in p for p in problems), problems
    assert any("batch=1" in p and "!=" in p for p in act["chain_problems"]), "the stages disagree among themselves too"
    # the Commit citing a Build artifact nobody published
    docs[2]["inputs"] = {**docs[2]["inputs"], "build": "art:" + "f" * 64}
    act = AP.actual(row)
    assert act["chain_consistent"] is False and any("commit cites build=art:ffff" in p for p in act["chain_problems"])
    assert any("chain_consistent" in p for p in resolver.diff(exp, act))


def test_attempt_provenance_does_not_apply_without_attempt_ids(monkeypatch, tmp_path):
    monkeypatch.delenv(resolver.CANDIDATE_ENV, raising=False)
    assert "no candidate Attempt ids" in AP.applies(_row())
    monkeypatch.setenv(resolver.CANDIDATE_ENV, str(tmp_path))
    assert "no candidate Attempt ids" in AP.applies(_row())


def test_row_config_grammar_matches_the_tool_declaration():
    """The harness's copy of the row-id grammar agrees with research_tools.row_config (which key params the Attempt carries)."""
    sys.path.insert(0, str(RESEARCH_SRC))
    sys.path.insert(0, str(store_io.REPO_ROOT))
    from integrations.vllm.verity_vllm.pipeline import research_tools as RT
    for rid in (ROW, "mistral-7b__bf16__l40s__tp1__b8__i1024__o128__mixed__greedy__bi-eager", "gemma2-2b__bf16__h100__tp1__b1__i512__o64__burst__stochastic__bi-graph", "bad"):
        assert AP.row_config(rid) == RT.row_config(rid), rid
    assert AP.CONFIG_KEYS == ("row", "model", "revision", "query", *AP.ROW_FIELDS)
    assert set(AP.CONFIG_KEYS) <= set(RT.ROW_KEY_PARAMS)
    assert AP.DEFAULT_QUERY == RT.DEFAULT_QUERY
