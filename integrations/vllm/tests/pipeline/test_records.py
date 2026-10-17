"""The circuit database (`verity_vllm.records`): the index, and a TP row published from its row dir and loaded back by row id."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from research.store.local import LocalStore
from verity.ml.circuits import encode_program, program_digest
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.evaluation.reference import evaluate
from verity_vllm import records
from verity_vllm.config import WORKLOADS, RowId, UsageError
from verity_vllm.pipeline import research_outputs as ro

from tests.program.test_running_example import make_program

ROW = "llama32-1b__bf16__l40s__tp2__b1__i256__o32__mixed__greedy__bi-eager"
PARTS = ("build_step", "build_request_LP9_T2", "build_workload")
SUMMARY = {"schema": "verity/commit-integ/delta-tp/v1", "pass": True, "pairs": 1, "commit_pass": [True], "tp_run_roots": ["ab" * 32],
           "per_rank_roots": [{"0": "c0" * 32, "1": "c1" * 32}], "openings": 64, "weights_pin": {"result": True},
           "value_check": {"verdict": "PASS"}, "match_oracle": {"0": {"verdict": "PASS"}, "1": {"verdict": "PASS"}},
           "required_manifest": {"manifest_digest": "mm", "identities": 7},
           "sampled_replay": {"result": True, "sample": {"picked": 4, "evaluated": 4, "equal": 4, "mismatch_n": 0},
                              "per_rank": {"0": {"result": True, "grade": "complete"}, "1": {"result": True, "grade": "complete"}}}}


def test_index_names_rows_of_record():
    for row, art in records.index().items():
        RowId.parse(row)
        assert (WORKLOADS / f"{row}.json").is_file(), row
        assert records.ART.fullmatch(art), (row, art)


def _w(p: Path, doc) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))


def _row_dir(tmp_path: Path, digest: str | None = None) -> Path:
    """A TP2 row dir as the stages leave it: per rank the Programs' descriptors, the Build summary, the manifest and the fold Match."""
    d = tmp_path / ROW
    cj = canonical_json(encode_program(make_program()))
    real = program_digest(json.loads(cj))
    ranks = {}
    for r in range(2):
        for part in PARTS:
            p = d / "build" / f"rank{r}" / part / "descriptor.json.gz"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(gzip.compress(cj))
            _w(p.parent / "result.json", {"ok": True})
        ranks[f"rank{r}"] = {"build_step": {"ok": True, "program_digest": digest or real},
                             "build_request_LP9_T2": {"ok": True, "program_digest": real},
                             "workload": {"ok": True, "workload_digest": real}}
        _w(d / "build" / f"rank{r}" / "build_workload" / "workload_program.json", {"workload_digest": real})
    _w(d / "build_summary.json", {"tp": {"world": 2}, "ranks": ranks, "step": {"program_digest": real}})
    _w(d / "manifest.json", {"manifest_digest": "mm", "query": {"query_id": "Q_word_v1{X=16,W=32,R=no-recompute}"}, "identities": []})
    _w(d / "match" / "fold_match" / "tp2_fold_match.json", {"schema": "verity/tp2/fold-match/v1", "pass": True, "world": 2,
                                                            "profile": "derived_LLAMA32_1B_tp2", "legs": {"cross_rank": "PASS"}})
    (d / "stages.txt").write_text("build PASS 2026-10-17T00:00:00Z rc=0\n")
    return d


@pytest.fixture
def store(tmp_path, monkeypatch):
    """The store the research CLI selects: a local root with a directory as its remote."""
    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    monkeypatch.setattr(records, "INDEX", tmp_path / "records.json")
    return LocalStore(tmp_path / "store", config_path=tmp_path / "store.toml")


def _build_art(st: LocalStore, d: Path, run: Path) -> str:
    doc = ro.build_outputs(d, ROW, "build", run, {})
    build = next(e for e in doc["outputs"] if e["name"] == "build")
    return st.put_artifact(build["kind"], {"row": ROW}, tree=run / build["tree"])


def _served(d: Path) -> dict:
    """SUMMARY with the per-rank Program digests its manifest was built for: the row dir's own."""
    ranks = json.loads((d / "build_summary.json").read_text())["ranks"]
    per = {r.removeprefix("rank"): v["workload"]["workload_digest"] for r, v in ranks.items()}
    return {**SUMMARY, "required_manifest": {**SUMMARY["required_manifest"], "program_digest_per_rank": per}}


def test_a_published_row_loads_back_by_id(tmp_path, store):
    d = _row_dir(tmp_path)
    build = _build_art(store, d, tmp_path / "run")
    ev = records.evidence_of(build, manifest="art:" + "1" * 64, match="art:" + "2" * 64, match_run="r20261016-083240-d76e",
                             commit="art:" + "3" * 64, commit_run="r20261016-213336-6c35",
                             conformance={"glm53_bf16_gemm_capture": "art:" + "4" * 64})
    art = records.publish(d, ROW, evidence=ev, tree="cursor/x@1b53f685a", by="circuits", summary=_served(d))
    assert records.index() == {ROW: art}
    m = store.get_manifest(art)
    assert m.kind == "vllm-config-record/v1" and m.meta == {"row": ROW, "outcome": "PASS", "world": 2, "tree": "cursor/x@1b53f685a"}
    assert m.refs == {"build": build, "manifest": ev["manifest"], "match": ev["match"], "commit": ev["commit"],
                      "glm53_bf16_gemm_capture": "art:" + "4" * 64}
    assert [lb.value for lb in store.labels(art, "row")] == [ROW]
    assert any(w.get("location") == "remote" and w.get("present") for w in store.where(art))

    rec = records.record(ROW)
    assert rec["outcome"] == "PASS" and rec["evidence"] == ev and rec["tree"] == "cursor/x@1b53f685a"
    assert rec["row"]["file"] == f"workloads/{ROW}.json" and rec["row"]["profile"] == "derived_LLAMA32_1B_tp2"
    assert rec["match"]["ok"] is True and rec["commit"]["run_root"] == "ab" * 32
    assert rec["commit"]["per_rank"] == {"0": {"run_roots": ["c0" * 32]}, "1": {"run_roots": ["c1" * 32]}}
    assert rec["commit"]["gates"] == {"weights_pin": True, "value_check": True, "match_oracle": True}
    assert rec["replay"]["ok"] is True and rec["timing"] == {} and rec["engine"] is None

    progs = records.programs(ROW, to=tmp_path / "fetched")
    P = make_program()
    assert sorted(progs) == [0, 1] and sorted(progs[1]) == ["build_request_LP9_T2", "build_step", "workload"]
    assert all(p.gates == P.gates for ps in progs.values() for p in ps.values())
    assert evaluate(progs[0]["workload"].defs["Mul16_v1"], 3, 5) == (15,)
    assert sorted(records.programs(ROW, [1])) == [1]
    with pytest.raises(UsageError, match="no rank"):
        records.programs(ROW, [2])


def test_a_descriptor_that_is_not_the_records_is_refused(tmp_path, store):
    d = _row_dir(tmp_path, digest="0" * 128)
    build = _build_art(store, d, tmp_path / "run")
    records.publish(d, ROW, evidence=records.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", summary=SUMMARY)
    with pytest.raises(ValueError, match="build_step: the descriptor's digest"):
        records.programs(ROW, [0])


def test_publish_refuses_a_failed_row_another_builds_commit_and_a_build_without_its_programs(tmp_path, store):
    d = _row_dir(tmp_path)
    build = _build_art(store, d, tmp_path / "run")
    with pytest.raises(UsageError, match="FAIL"):
        records.publish(d, ROW, evidence=records.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits",
                        summary={**SUMMARY, "pass": False})
    served = _served(d)
    for rm in ({**served["required_manifest"], "manifest_digest": "other"},
               {**served["required_manifest"], "program_digest_per_rank": {"0": "0" * 128}}):
        with pytest.raises(UsageError, match="not the one served"):
            records.publish(d, ROW, evidence=records.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits",
                            summary={**served, "required_manifest": rm})
    (d / "build" / "rank1" / "build_workload" / "descriptor.json.gz").unlink()
    short = _build_art(store, d, tmp_path / "run2")
    with pytest.raises(UsageError, match="lacks 1 Program descriptor"):
        records.publish(d, ROW, evidence=records.evidence_of(short), tree="cursor/x@1b53f685a", by="circuits", summary=SUMMARY)
    assert records.index() == {}


def test_evidence_ids_are_checked():
    with pytest.raises(UsageError, match="match_run"):
        records.evidence_of("art:" + "0" * 64, match_run="20261016")
    with pytest.raises(UsageError, match="build"):
        records.evidence_of("art:abc")
    with pytest.raises(UsageError, match="conformance"):
        records.evidence_of("art:" + "0" * 64, conformance={"p": "r20261016-083240-d76e"})
    with pytest.raises(UsageError, match="conformance build"):
        records.evidence_of("art:" + "0" * 64, conformance={"build": "art:" + "1" * 64})
