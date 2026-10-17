"""The circuit database (`verity_vllm.records`): the index, a TP row published from its row dir, pinned and loaded back by row id, its
conformance (captures of record, named gaps, a property left silent refused), the in-stage manifest a manifest of record
supersedes, and a job's bundle read with no store."""
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
from verity_vllm.commit import capture_map as CM
from verity_vllm.config import WORKLOADS, RowId, UsageError
from verity_vllm.pipeline import epoch
from verity_vllm.pipeline import research_outputs as ro
from verity_vllm.properties.record import REGISTRY, seal

from tests.program.test_running_example import make_program

ROW = "llama32-1b__bf16__l40s__tp2__b1__i256__o32__mixed__greedy__bi-eager"
REPO = "meta-llama/Llama-3.2-1B"
PARTS = ("build_step", "build_request_LP9_T2", "build_workload")
PROP = "glm53_bf16_gemm_capture"
SUMMARY = {"schema": "verity/commit-integ/delta-tp/v1", "pass": True, "pairs": 1, "commit_pass": [True], "tp_run_roots": ["ab" * 32],
           "per_rank_roots": [{"0": "c0" * 32, "1": "c1" * 32}], "openings": 64, "weights_pin": {"result": True},
           "value_check": {"verdict": "PASS"}, "match_oracle": {"0": {"verdict": "PASS"}, "1": {"verdict": "PASS"}},
           "required_manifest": {"manifest_digest": "mm", "identities": 7},
           "sampled_replay": {"result": True, "sample": {"picked": 4, "evaluated": 4, "equal": 4, "mismatch_n": 0},
                              "per_rank": {"0": {"result": True, "grade": "complete"}, "1": {"result": True, "grade": "complete"}}},
           "served": {"schema": "verity-vllm/served-switches/v1", "ok": True, "problems": []}}


def test_index_names_rows_of_record():
    for row, e in records.index().items():
        RowId.parse(row)
        assert (WORKLOADS / f"{row}.json").is_file(), row
        assert e["record"] is None or records.ART.fullmatch(e["record"]), (row, e["record"])


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
            if part != "build_workload":
                _w(p.parent / records.CONSTRUCTION, {"model": {"pin": {"model": REPO}}, "tp": {"world": 2}, "spec_histogram": {"Mul16_v1": 3}})
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
    """The store the research CLI selects: a local root with a directory as its remote; and no capture map but the test's."""
    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    monkeypatch.setattr(records, "INDEX", tmp_path / "records.json")
    (tmp_path / "maps").mkdir()
    monkeypatch.setattr(CM, "MAPS", tmp_path / "maps")
    return LocalStore(tmp_path / "store", config_path=tmp_path / "store.toml")


def _map(tmp_path: Path) -> None:
    """The checkpoint's capture map at TP2: the Program the Build uses (Mul16_v1) needs PROP; one it does not use needs another."""
    _w(tmp_path / "maps" / "m.json", {"schema": CM.SCHEMA, "id": "llama-test-tp2", "pins": {"checkpoint": {"repo": REPO}},
                                      "target": {"tensor_parallel_size": 2},
                                      "programs": {"mul": {"status": "exists", "definitions": ["Mul16_v1"], "captures": [PROP]},
                                                   "attn": {"status": "port", "definitions": ["Attention_v4"],
                                                            "captures": ["glm53_fa3_prefill_capture"]}}})


def _build_art(st: LocalStore, d: Path, run: Path) -> str:
    doc = ro.build_outputs(d, ROW, "build", run, {})
    build = next(e for e in doc["outputs"] if e["name"] == "build")
    return st.put_artifact(build["kind"], {"row": ROW}, tree=run / build["tree"])


_runs = iter(range(1 << 16))


def _capture_run(st: LocalStore, tmp_path: Path, prop: str = PROP, ok: bool = True, tamper: bool = False) -> str:
    """A capture run's attempt whose run record holds `prop`'s property record under out/properties/."""
    rid = f"r20261016-234133-{next(_runs):04x}"
    doc = seal({"schema": REGISTRY[prop].schema, "words": 7}, ok)
    _w(tmp_path / rid / "out" / "properties" / f"{prop}.json", {**doc, "words": 8} if tamper else doc)
    rr = st.put_artifact("run-record/v1", {"run_id": rid}, tree=tmp_path / rid)
    st.put_attempt({"schema": "research/attempt/v0.1", "id": rid, "tool": {"name": "bash", "version": None},
                    "derivation": "drv:" + "0" * 64, "reuse_key": None, "argv": [], "params": {}, "key_params": [], "inputs": {},
                    "conditions": {}, "key_conditions": [], "source": {"commit": "850fb2f6f" + "0" * 31, "dirty": False},
                    "closure": None, "execution": {}, "cost": None, "status": {"state": "done", "rc": 0},
                    "outputs": {"run_record": rr}, "telemetry": None})
    return rid


def _served(d: Path) -> dict:
    """SUMMARY with the per-rank Program digests its manifest was built for: the row dir's own."""
    ranks = json.loads((d / "build_summary.json").read_text())["ranks"]
    per = {r.removeprefix("rank"): v["workload"]["workload_digest"] for r, v in ranks.items()}
    return {**SUMMARY, "required_manifest": {**SUMMARY["required_manifest"], "program_digest_per_rank": per}}


def test_a_published_row_loads_back_by_id(tmp_path, store):
    d = _row_dir(tmp_path)
    build = _build_art(store, d, tmp_path / "run")
    cap = records.capture(_capture_run(store, tmp_path))
    ev = records.evidence_of(build, manifest="art:" + "1" * 64, match="art:" + "2" * 64, match_run="r20261016-083240-d76e",
                             commit="art:" + "3" * 64, commit_run="r20261016-213336-6c35", conformance=cap)
    with pytest.raises(UsageError, match=r"a row new to records.json needs --tier fast\|heavy"):
        records.publish(d, ROW, evidence=ev, tree="cursor/x@1b53f685a", by="circuits", summary=_served(d))
    art = records.publish(d, ROW, evidence=ev, tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=_served(d))
    assert records.index() == {ROW: {"record": art, "tier": "fast", "pinned_from": {"record": art, "build": build, "tree": "cursor/x@1b53f685a"},
                                     "manifest_digest": "mm", "programs": epoch.build_programs(d)}}
    assert sorted(records.index()[ROW]["programs"]) == ["0", "1"]
    m = store.get_manifest(art)
    assert m.kind == "vllm-config-record/v1" and m.meta == {"row": ROW, "outcome": "PASS", "world": 2, "tree": "cursor/x@1b53f685a"}
    assert m.refs == {"build": build, "manifest": ev["manifest"], "match": ev["match"], "commit": ev["commit"], PROP: cap[PROP]}
    assert [lb.value for lb in store.labels(art, "row")] == [ROW]
    assert any(w.get("location") == "remote" and w.get("present") for w in store.where(art))

    rec = records.record(ROW)
    assert rec["outcome"] == "PASS" and rec["evidence"] == ev and rec["tree"] == "cursor/x@1b53f685a"
    assert rec["conformance"] == {"map": None, "due": {}}, "no capture map describes the checkpoint: nothing is due"
    assert rec["stages"] == {"build": {"status": "PASS", "line": "2026-10-17T00:00:00Z rc=0"}}
    assert rec["build"]["stage"] == {"status": "PASS", "manifest_rc": None}
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
    assert {r: sorted(ps) for r, ps in records.programs(ROW, part=["workload"]).items()} == {0: ["workload"], 1: ["workload"]}
    with pytest.raises(UsageError, match="no rank"):
        records.programs(ROW, [2])
    with pytest.raises(UsageError, match="no part"):
        records.programs(ROW, part=["build_request"])
    assert records.publish(d, ROW, evidence=ev, tree="cursor/x@1b53f685a", by="circuits", summary=_served(d)) == art
    assert records.index()[ROW]["tier"] == "fast", "a re-publish keeps the row's tier"
    doc = json.loads(records.INDEX.read_text())
    doc["rows"]["pinned-row"] = {**doc["rows"][ROW], "record": None}
    records.INDEX.write_text(json.dumps(doc))
    with pytest.raises(UsageError, match=r"pinned-row: not in the circuit database \(records.json pins its Programs but names no record"):
        records.art_of("pinned-row")


def test_a_descriptor_that_is_not_the_records_is_refused(tmp_path, store):
    d = _row_dir(tmp_path, digest="0" * 128)
    build = _build_art(store, d, tmp_path / "run")
    records.publish(d, ROW, evidence=records.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=SUMMARY)
    with pytest.raises(ValueError, match="build_step: the descriptor's digest"):
        records.programs(ROW, [0])


def test_publish_refuses_a_failed_row_another_builds_commit_and_a_build_without_its_programs(tmp_path, store):
    d = _row_dir(tmp_path)
    build = _build_art(store, d, tmp_path / "run")
    with pytest.raises(UsageError, match="FAIL"):
        records.publish(d, ROW, evidence=records.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="fast",
                        summary={**SUMMARY, "pass": False})
    served = _served(d)
    with pytest.raises(UsageError, match=r"the record is FAIL \(served switches not run\)"):
        records.publish(d, ROW, evidence=records.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="fast",
                        summary={k: v for k, v in served.items() if k != "served"})
    for rm in ({**served["required_manifest"], "manifest_digest": "other"},
               {**served["required_manifest"], "program_digest_per_rank": {"0": "0" * 128}}):
        with pytest.raises(UsageError, match="not the one served"):
            records.publish(d, ROW, evidence=records.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="fast",
                            summary={**served, "required_manifest": rm})
    (d / "build" / "rank1" / "build_workload" / "descriptor.json.gz").unlink()
    short = _build_art(store, d, tmp_path / "run2")
    with pytest.raises(UsageError, match="lacks 1 Program descriptor"):
        records.publish(d, ROW, evidence=records.evidence_of(short), tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=SUMMARY)
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
    with pytest.raises(UsageError, match="gap p=' '"):
        records.evidence_of("art:" + "0" * 64, gaps={"p": " "})
    with pytest.raises(UsageError, match="cited or a gap, not both"):
        records.evidence_of("art:" + "0" * 64, conformance={"p": "art:" + "1" * 64}, gaps={"p": "no capture"})


def test_a_capture_run_s_records_are_preserved_and_one_that_does_not_hold_or_recompute_is_refused(tmp_path, store):
    run = _capture_run(store, tmp_path)
    cap = records.capture(run)
    m = store.get_manifest(cap[PROP])
    rr = store.get_attempt(run)["outputs"]["run_record"]
    assert m.kind == records.PROPERTY_KIND and m.refs == {"run": rr}
    assert m.meta == {"property": PROP, "schema": REGISTRY[PROP].schema, "digest": m.meta["digest"], "ok": True,
                      "path": f"out/properties/{PROP}.json", "run_id": run, "commit": "850fb2f6f" + "0" * 31}
    assert records.capture(run, [PROP]) == cap, "the same record preserved again is the same artifact"
    with pytest.raises(UsageError, match="does not hold"):
        records.capture(_capture_run(store, tmp_path, ok=False))
    with pytest.raises(UsageError, match="digest does not match"):
        records.capture(_capture_run(store, tmp_path, tamper=True))
    with pytest.raises(UsageError, match="for glm53_dsa_capture"):
        records.capture(run, ["glm53_dsa_capture"])


def test_publish_refuses_a_needed_property_left_silent_and_records_a_named_gap(tmp_path, store):
    """The map's Program the Build uses (Mul16_v1 in its construction manifests) needs PROP; the one it does not use needs nothing."""
    _map(tmp_path)
    d = _row_dir(tmp_path)
    build = _build_art(store, d, tmp_path / "run")
    with pytest.raises(UsageError, match=f"need {PROP} \\(llama-test-tp2\\), which the record neither cites nor names a gap"):
        records.publish(d, ROW, evidence=records.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=_served(d))
    other = records.capture(_capture_run(store, tmp_path, prop="glm53_dsa_capture"))["glm53_dsa_capture"]
    with pytest.raises(UsageError, match=f"not a holding property-record/v1 of {PROP}"):
        records.publish(d, ROW, evidence=records.evidence_of(build, conformance={PROP: other}), tree="cursor/x@1b53f685a",
                        by="circuits", tier="fast", summary=_served(d))
    assert records.index() == {}
    ev = records.evidence_of(build, gaps={PROP: "no capture of the tree's harness"})
    records.publish(d, ROW, evidence=ev, tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=_served(d))
    rec = records.record(ROW)
    assert rec["conformance"] == {"map": "llama-test-tp2", "due": {PROP: ["mul"]}}
    assert rec["evidence"]["gaps"] == {PROP: "no capture of the tree's harness"} and rec["evidence"]["conformance"] == {}


def test_a_build_without_construction_manifests_is_refused(tmp_path, store):
    d = _row_dir(tmp_path)
    for p in d.glob(f"build/rank*/*/{records.CONSTRUCTION}"):
        p.unlink()
    build = _build_art(store, d, tmp_path / "run")
    with pytest.raises(UsageError, match="the conformance they need, are unknown"):
        records.publish(d, ROW, evidence=records.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=_served(d))


def test_a_job_reads_the_programs_from_a_bundle_with_no_store(tmp_path, store, monkeypatch):
    d = _row_dir(tmp_path)
    build = _build_art(store, d, tmp_path / "run")
    records.publish(d, ROW, evidence=records.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=_served(d))
    b = records.bundle(ROW, tmp_path / "job.tar", [1], part=["workload"])
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "no-store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "no-store.toml"))
    monkeypatch.setattr(records, "INDEX", tmp_path / "no-index.json")
    progs = records.programs(ROW, [1], part=["workload"], bundle=b)
    assert {r: sorted(ps) for r, ps in progs.items()} == {1: ["workload"]} and progs[1]["workload"].gates == make_program().gates
    assert records.record(ROW, b)["evidence"]["build"] == build
    with pytest.raises(UsageError, match="lacks 1 descriptor"):
        records.programs(ROW, [0], part=["workload"], bundle=b)
    with pytest.raises(UsageError, match="the record of .*, not other-row"):
        records.programs("other-row", bundle=b)
    with pytest.raises(UsageError, match="not in the circuit database"):
        records.programs(ROW, [1])
    with pytest.raises(RuntimeError, match=r"no remote configured.*a research run job has no store.*records fetch ROW --to FILE.*"
                                           r"--send FILE.*--bundle"):
        records.fetch(build)
