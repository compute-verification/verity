"""The circuit database (`verity_vllm.pipeline.circuit_db`): the index, a TP row published from its row dir, pinned and loaded back
by row id, its conformance (captures of record, named gaps, a property left silent refused), the in-stage manifest a manifest of record
supersedes, a row file not done only at its public cap refused by publish, pull and switch before anything is written, a job's
bundle read with no store, and a checkpoint's staging event."""
from __future__ import annotations

import gzip
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from research.store.local import LocalStore
from verity.circuits import encode_program, program_digest
from verity.circuits.codec import canonical_json
from verity.circuits.evaluation.reference import evaluate

from tests.program.test_running_example import make_program
from verity_vllm import glue
from verity_vllm.commit import capture_map as CM
from verity_vllm.config import WORKLOADS, RowId, UsageError
from verity_vllm.pipeline import circuit_db, cli, epoch
from verity_vllm.pipeline import config_record as CR
from verity_vllm.pipeline import research_outputs as ro
from verity_vllm.pipeline import row_records as RR
from verity_vllm.properties.record import REGISTRY, seal

ROW = "llama32-1b__bf16__l40s__tp2__b1__i256__o32__mixed__greedy__bi-eager"
REPO = "meta-llama/Llama-3.2-1B"
PARTS = ("build_step", "build_request_LP9_T2", "build_workload")
PROP = "glm53_bf16_gemm_capture"
SUMMARY: dict[str, Any] = {
    "schema": "verity/commit-integ/delta-tp/v1", "pass": True, "pairs": 1, "commit_pass": [True], "tp_run_roots": ["ab" * 32],
    "per_rank_roots": [{"0": "c0" * 32, "1": "c1" * 32}], "openings": 64, "weights_pin": {"result": True},
    "value_check": {"verdict": "PASS"}, "match_oracle": {"0": {"verdict": "PASS"}, "1": {"verdict": "PASS"}},
    "required_manifest": {"manifest_digest": "mm", "identities": 7},
    "sampled_replay": {"result": True, "sample": {"picked": 4, "evaluated": 4, "equal": 4, "mismatch_n": 0},
                       "per_rank": {"0": {"result": True, "grade": "complete"}, "1": {"result": True, "grade": "complete"}}},
    "served": {"schema": "verity-vllm/served-switches/v1", "ok": True, "problems": []}}


def test_index_names_rows_of_record():
    for row, e in circuit_db.index().items():
        RowId.parse(row)
        assert (WORKLOADS / f"{row}.json").is_file(), row
        assert e["record"] is None or circuit_db.ART.fullmatch(e["record"]), (row, e["record"])
        assert "aot" not in e or (circuit_db.RUN.fullmatch(e["aot"]["comparison"]) and circuit_db.ART.fullmatch(e["aot"]["trace"])
                                  and RowId.parse(row).world > 1), (row, e["aot"])


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
                _w(p.parent / circuit_db.CONSTRUCTION, {"model": {"pin": {"model": REPO}}, "tp": {"world": 2}, "spec_histogram": {"Mul16_v1": 3}})
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


def _capped(doc: dict) -> dict:
    return {**doc, "sampling": {**doc["sampling"], "ignore_eos": True}}


@pytest.fixture
def store(tmp_path, monkeypatch):
    """The store the research CLI selects: a local root with a directory as its remote; no capture map but the test's; and ROW's
    file done only at its public cap, as the file of a row `publish` indexes must be."""
    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    monkeypatch.setattr(circuit_db, "INDEX", tmp_path / "records.json")
    (tmp_path / "maps").mkdir()
    monkeypatch.setattr(CM, "MAPS", tmp_path / "maps")
    _w(tmp_path / "workloads" / f"{ROW}.json", _capped(json.loads((WORKLOADS / f"{ROW}.json").read_text())))
    monkeypatch.setattr(CR, "WORKLOADS", tmp_path / "workloads")
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
    cap = circuit_db.capture(_capture_run(store, tmp_path))
    ev = circuit_db.evidence_of(build, manifest="art:" + "1" * 64, match="art:" + "2" * 64, match_run="r20261016-083240-d76e",
                             commit="art:" + "3" * 64, commit_run="r20261016-213336-6c35", conformance=cap)
    with pytest.raises(UsageError, match=r"a row new to records.json needs --tier fast\|heavy"):
        circuit_db.publish(d, ROW, evidence=ev, tree="cursor/x@1b53f685a", by="circuits", summary=_served(d))
    art = circuit_db.publish(d, ROW, evidence=ev, tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=_served(d))
    assert circuit_db.index() == {ROW: {"record": art, "tier": "fast", "pinned_from": {"record": art, "build": build, "tree": "cursor/x@1b53f685a"},
                                     "manifest_digest": "mm", "programs": epoch.build_programs(d)}}
    assert sorted(circuit_db.index()[ROW]["programs"]) == ["0", "1"]
    m = store.get_manifest(art)
    assert m.kind == "vllm-config-record/v1" and m.meta == {"row": ROW, "outcome": "PASS", "world": 2, "tree": "cursor/x@1b53f685a",
                                                            "tier": "fast"}
    assert m.refs == {"build": build, "manifest": ev["manifest"], "match": ev["match"], "commit": ev["commit"], PROP: cap[PROP]}
    assert [lb.value for lb in store.labels(art, "row")] == [ROW]
    assert any(w.get("location") == "remote" and w.get("present") for w in store.where(art))

    rec = circuit_db.record(ROW)
    assert rec["outcome"] == "PASS" and rec["evidence"] == ev and rec["tree"] == "cursor/x@1b53f685a"
    assert rec["conformance"] == {"map": None, "due": {}}, "no capture map describes the checkpoint: nothing is due"
    assert rec["stages"] == {"build": {"status": "PASS", "line": "2026-10-17T00:00:00Z rc=0"}}
    assert rec["build"]["stage"] == {"status": "PASS", "manifest_rc": None}
    assert rec["row"]["file"] == f"workloads/{ROW}.json" and rec["row"]["profile"] == "derived_LLAMA32_1B_tp2"
    assert rec["match"]["ok"] is True and rec["commit"]["run_root"] == "ab" * 32
    assert rec["commit"]["per_rank"] == {"0": {"run_roots": ["c0" * 32]}, "1": {"run_roots": ["c1" * 32]}}
    assert rec["commit"]["gates"] == {"weights_pin": True, "value_check": True, "match_oracle": True}
    assert rec["replay"]["ok"] is True and rec["timing"] == {} and rec["engine"] is None

    progs = circuit_db.programs(ROW, to=tmp_path / "fetched")
    P = make_program()
    assert sorted(progs) == [0, 1] and sorted(progs[1]) == ["build_request_LP9_T2", "build_step", "workload"]
    assert all(p.gates == P.gates for ps in progs.values() for p in ps.values())
    assert evaluate(progs[0]["workload"].defs["Mul16_v1"], 3, 5) == (15,)
    assert sorted(circuit_db.programs(ROW, [1])) == [1]
    assert {r: sorted(ps) for r, ps in circuit_db.programs(ROW, part=["workload"]).items()} == {0: ["workload"], 1: ["workload"]}
    with pytest.raises(UsageError, match="no rank"):
        circuit_db.programs(ROW, [2])
    with pytest.raises(UsageError, match="no part"):
        circuit_db.programs(ROW, part=["build_request"])
    assert circuit_db.publish(d, ROW, evidence=ev, tree="cursor/x@1b53f685a", by="circuits", summary=_served(d)) == art
    assert circuit_db.index()[ROW]["tier"] == "fast", "a re-publish keeps the row's tier"
    doc = json.loads(circuit_db.INDEX.read_text())
    doc["rows"]["pinned-row"] = {**doc["rows"][ROW], "record": None}
    circuit_db.INDEX.write_text(json.dumps(doc))
    with pytest.raises(UsageError, match=r"pinned-row: not in the circuit database \(records.json pins its Programs but names no record"):
        circuit_db.art_of("pinned-row")


def test_a_descriptor_that_is_not_the_records_is_refused(tmp_path, store):
    d = _row_dir(tmp_path, digest="0" * 128)
    build = _build_art(store, d, tmp_path / "run")
    circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=SUMMARY)
    with pytest.raises(ValueError, match="build_step: the descriptor's digest"):
        circuit_db.programs(ROW, [0])


def test_publish_refuses_a_failed_row_another_builds_commit_and_a_build_without_its_programs(tmp_path, store):
    d = _row_dir(tmp_path)
    build = _build_art(store, d, tmp_path / "run")
    with pytest.raises(UsageError, match="FAIL"):
        circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="fast",
                        summary={**SUMMARY, "pass": False})
    served = _served(d)
    with pytest.raises(UsageError, match=r"the record is FAIL \(served switches not run\)"):
        circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="fast",
                        summary={k: v for k, v in served.items() if k != "served"})
    for rm in ({**served["required_manifest"], "manifest_digest": "other"},
               {**served["required_manifest"], "program_digest_per_rank": {"0": "0" * 128}}):
        with pytest.raises(UsageError, match="not the one served"):
            circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="fast",
                            summary={**served, "required_manifest": rm})
    (d / "build" / "rank1" / "build_workload" / "descriptor.json.gz").unlink()
    short = _build_art(store, d, tmp_path / "run2")
    with pytest.raises(UsageError, match="lacks 1 Program descriptor"):
        circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(short), tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=SUMMARY)
    assert circuit_db.index() == {}


def test_publish_refuses_a_row_not_done_only_at_its_public_cap_before_it_writes_anything(tmp_path, store):
    d = _row_dir(tmp_path)
    build = _build_art(store, d, tmp_path / "run")
    wl, eos = CR.WORKLOADS / f"{ROW}.json", json.loads((WORKLOADS / f"{ROW}.json").read_text())
    publish = lambda: circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits",  # noqa: E731
                                      tier="fast", summary=_served(d))
    stored = store.artifact_ids()
    for doc, why in ((eos, r"its done source is 'eos-token' \(sampling.ignore_eos is None\)"),
                     ({**_capped(eos), "sampling": {**_capped(eos)["sampling"], "stop_token_ids": [128001]}},
                      r"sampling\.stop_token_ids \[128001\] stops a request before its cap")):
        _w(wl, doc)
        with pytest.raises(UsageError, match=rf"{ROW}: a row of record \(records\.json\) is done only at its public cap, and {why}.*"
                                             r"publishes nothing"):
            publish()
        assert store.artifact_ids() == stored and not circuit_db.INDEX.exists(), "a refused row writes no record and no index"
    _w(wl, _capped(eos))
    art = publish()
    stored, indexed = store.artifact_ids(), circuit_db.index()
    _w(wl, eos)
    with pytest.raises(UsageError, match=rf"{ROW}: a row of record .* its done source is 'eos-token'"):
        publish()
    assert store.artifact_ids() == stored and circuit_db.index() == indexed and indexed[ROW]["record"] == art, \
        "a row of record whose file no longer is at its cap is not re-published"


def test_evidence_ids_are_checked():
    with pytest.raises(UsageError, match="match_run"):
        circuit_db.evidence_of("art:" + "0" * 64, match_run="20261016")
    with pytest.raises(UsageError, match="build"):
        circuit_db.evidence_of("art:abc")
    with pytest.raises(UsageError, match="conformance"):
        circuit_db.evidence_of("art:" + "0" * 64, conformance={"p": "r20261016-083240-d76e"})
    with pytest.raises(UsageError, match="conformance build"):
        circuit_db.evidence_of("art:" + "0" * 64, conformance={"build": "art:" + "1" * 64})
    with pytest.raises(UsageError, match="gap p=' '"):
        circuit_db.evidence_of("art:" + "0" * 64, gaps={"p": " "})
    with pytest.raises(UsageError, match="cited or a gap, not both"):
        circuit_db.evidence_of("art:" + "0" * 64, conformance={"p": "art:" + "1" * 64}, gaps={"p": "no capture"})


def test_a_capture_run_s_records_are_preserved_and_one_that_does_not_hold_or_recompute_is_refused(tmp_path, store):
    run = _capture_run(store, tmp_path)
    cap = circuit_db.capture(run)
    m = store.get_manifest(cap[PROP])
    rr = store.get_attempt(run)["outputs"]["run_record"]
    assert m.kind == circuit_db.PROPERTY_KIND and m.refs == {"run": rr}
    assert m.meta == {"property": PROP, "schema": REGISTRY[PROP].schema, "digest": m.meta["digest"], "ok": True,
                      "path": f"out/properties/{PROP}.json", "run_id": run, "commit": "850fb2f6f" + "0" * 31}
    assert circuit_db.capture(run, [PROP]) == cap, "the same record preserved again is the same artifact"
    with pytest.raises(UsageError, match="does not hold"):
        circuit_db.capture(_capture_run(store, tmp_path, ok=False))
    with pytest.raises(UsageError, match="digest does not match"):
        circuit_db.capture(_capture_run(store, tmp_path, tamper=True))
    with pytest.raises(UsageError, match="for glm53_dsa_capture"):
        circuit_db.capture(run, ["glm53_dsa_capture"])


def test_publish_refuses_a_needed_property_left_silent_and_records_a_named_gap(tmp_path, store):
    """The map's Program the Build uses (Mul16_v1 in its construction manifests) needs PROP; the one it does not use needs nothing."""
    _map(tmp_path)
    d = _row_dir(tmp_path)
    build = _build_art(store, d, tmp_path / "run")
    with pytest.raises(UsageError, match=f"need {PROP} \\(llama-test-tp2\\), which the record neither cites nor names a gap"):
        circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=_served(d))
    other = circuit_db.capture(_capture_run(store, tmp_path, prop="glm53_dsa_capture"))["glm53_dsa_capture"]
    with pytest.raises(UsageError, match=f"not a holding property-record/v1 of {PROP}"):
        circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build, conformance={PROP: other}), tree="cursor/x@1b53f685a",
                        by="circuits", tier="fast", summary=_served(d))
    assert circuit_db.index() == {}
    ev = circuit_db.evidence_of(build, gaps={PROP: "no capture of the tree's harness"})
    circuit_db.publish(d, ROW, evidence=ev, tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=_served(d))
    rec = circuit_db.record(ROW)
    assert rec["conformance"] == {"map": "llama-test-tp2", "due": {PROP: ["mul"]}}
    assert rec["evidence"]["gaps"] == {PROP: "no capture of the tree's harness"} and rec["evidence"]["conformance"] == {}


def test_a_request_programs_census_digest_is_in_the_record_beside_its_program_digest(tmp_path):
    """The census digest a request Build cached (its result.json's `census`, `pipeline.memory_map.cache`) reaches the Build summary
    and the record's `census_digests`, keyed as `programs`: per rank on a TP row, and on a TP1 row; a Build with none has no entry."""
    census = lambda r: {"census": f"c{r}" * 64, "kv_rows": True}  # noqa: E731
    d = tmp_path / ROW
    for r in range(2):
        rd = d / "build" / f"rank{r}"
        _w(rd / "build_step" / "result.json", {"ok": True, "program": {"digest": f"s{r}"}})
        _w(rd / "build_request_LP9_T2" / "result.json", {"ok": True, "program": {"digest": f"q{r}"}, "census": census(r)})
        _w(rd / "build_request_LP9_T3" / "result.json", {"ok": True, "program": {"digest": f"p{r}"}, "census": {"error": "MemoryError()"}})
        _w(rd / "build_workload" / "workload_program.json", {"workload_digest": f"w{r}"})
    assert RR.tp_build_summary(str(d / "build"), 2) == "PASS"
    per = CR.record(d, ROW)["build"]["per_rank"]
    assert {r: (p["programs"], p["census_digests"]) for r, p in per.items()} == {
        str(r): ({"build_request_LP9_T2": f"q{r}", "build_request_LP9_T3": f"p{r}", "build_step": f"s{r}"},
                 {"build_request_LP9_T2": census(r)["census"]}) for r in range(2)}

    one = tmp_path / "tp1"
    _w(one / "build_step" / "result.json", {"ok": True, "program": {"digest": "s"}})
    _w(one / "build_request" / "result.json", {"ok": True, "program": {"digest": "q"}, "census": census(0)})
    assert RR.build_summary(str(one)) == "PASS"
    b = CR.build_part(one)
    assert (b["programs"], b["census_digests"]) == ({"request": "q", "step": "s"}, {"request": census(0)["census"]})


def test_publish_cites_each_needed_property_s_newest_holding_capture_when_asked(tmp_path, store):
    """`--newest-captures`, as the row chain publishes: the latest capture run's record of PROP at its registered schema, not an
    older one, not one at another schema, and never in place of what the caller cites."""
    _map(tmp_path)
    d = _row_dir(tmp_path)
    build = _build_art(store, d, tmp_path / "run")
    old, new = (circuit_db.capture(_capture_run(store, tmp_path))[PROP] for _ in range(2))
    stale = tmp_path / "stale.json"
    _w(stale, {"schema": "verity-vllm/stale/v0", "words": 9, "ok": True})
    meta = {"property": PROP, "schema": "verity-vllm/stale/v0", "ok": True, "run_id": "r20991231-000000-ffff"}
    circuit_db.data("put", "--kind", circuit_db.PROPERTY_KIND, "--meta", json.dumps(meta), "--file", str(stale))
    assert circuit_db.newest_capture(PROP) == new != old and circuit_db.newest_capture("glm53_dsa_capture") is None
    art = circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="fast",
                          summary=_served(d), newest=True)
    assert circuit_db.record(ROW)["evidence"]["conformance"] == {PROP: new} and store.get_manifest(art).refs[PROP] == new
    circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build, conformance={PROP: old}), tree="cursor/x@1b53f685a", by="circuits",
                    summary=_served(d), newest=True)
    assert circuit_db.record(ROW)["evidence"]["conformance"] == {PROP: old}


def test_newest_captures_with_no_store_index_refuse_a_property_the_catalog_lacks(tmp_path, store, monkeypatch):
    """A `research run` job's store has no index, so `research data select` there reads only its own catalog: a property missing from
    it is refused naming the missing index and --conformance, never read as no capture.  A gap leaves nothing to resolve, and with an
    index the miss is a property left silent."""
    from research.store import bucket_index as BI
    monkeypatch.delenv(BI.READER_ENV, raising=False)
    _map(tmp_path)
    d = _row_dir(tmp_path)
    build = _build_art(store, d, tmp_path / "run")
    kw = dict(tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=_served(d), newest=True)
    with pytest.raises(UsageError, match=f"no capture of {PROP} in this store's catalog, and there is no store index here.*"
                                         "--conformance PROPERTY=art:"):
        circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), **kw)
    assert circuit_db.index() == {}
    circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build, gaps={PROP: "no capture of the tree's harness"}), **kw)
    BI.SqliteIndex(BI.default_path(store.root)).close()
    assert circuit_db.selected_from("--kind", circuit_db.PROPERTY_KIND) == "the index"
    with pytest.raises(UsageError, match="which the record neither cites nor names a gap"):
        circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), **kw)


def test_a_record_published_from_another_tree_is_pulled_into_this_index(tmp_path, store, monkeypatch):
    d = _row_dir(tmp_path)
    build = _build_art(store, d, tmp_path / "run")
    art = circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="heavy",
                          summary=_served(d))
    pod = circuit_db.index()
    monkeypatch.setattr(circuit_db, "INDEX", tmp_path / "checkout.json")
    assert circuit_db.pull(art) == ROW and circuit_db.index() == pod
    assert pod[ROW]["programs"] == epoch.build_programs(d), "the entry rebuilt from the record is the one its row dir gave"
    with pytest.raises(UsageError, match="not a PASS vllm-config-record/v1"):
        circuit_db.pull(build)


def test_pull_refuses_a_row_not_done_only_at_its_public_cap_before_it_touches_the_index(tmp_path, store, monkeypatch):
    d = _row_dir(tmp_path)
    build = _build_art(store, d, tmp_path / "run")
    art = circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="heavy",
                          summary=_served(d))
    monkeypatch.setattr(circuit_db, "INDEX", tmp_path / "checkout.json")
    _w(CR.WORKLOADS / f"{ROW}.json", json.loads((WORKLOADS / f"{ROW}.json").read_text()))
    with pytest.raises(UsageError, match=rf"{ROW}: a row of record \(records\.json\) is done only at its public cap, and its done source "
                                         r"is 'eos-token'.*pull would index it as one, so it indexes nothing"):
        circuit_db.pull(art)
    assert not circuit_db.INDEX.exists(), "a checkout whose file is not at its cap does not index the pod's record"


def test_a_checkpoints_staging_is_labelled_and_glue_counts_its_hours_from_it(tmp_path, store):
    rev = "aca966e4e02791568aa6a4ced368624b3d897f42"
    circuit_db.staged(REPO, rev, at="2026-10-12T21:42:55Z", ref="vy-glm53-eu:/fs/x.verify", by="circuits")
    circuit_db.staged(REPO, rev, at="2026-10-13T00:00:00Z", ref="r20261013-000000-0000", by="circuits")
    h = glue.hours(circuit_db.checkpoint(REPO, rev), [ROW], glue.store_sql)
    assert h["staged"] == {"at": "2026-10-12T21:42:55Z", "ref": "vy-glm53-eu:/fs/x.verify"} and h["record"] is None and h["hours"] is None
    with pytest.raises(UsageError, match="a UTC time"):
        circuit_db.staged(REPO, rev, at="2026-10-12 21:42", ref="x", by="circuits")


def test_a_build_without_construction_manifests_is_refused(tmp_path, store):
    d = _row_dir(tmp_path)
    for p in d.glob(f"build/rank*/*/{circuit_db.CONSTRUCTION}"):
        p.unlink()
    build = _build_art(store, d, tmp_path / "run")
    with pytest.raises(UsageError, match="the conformance they need, are unknown"):
        circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=_served(d))


def _comparison(d: Path) -> dict:
    """`verity-vllm predict compare`'s comparison of the row dir's Build: every Program predicted as traced."""
    ranks = {r: {"step": p["step"], "workload": p["workload"], **p["requests"]} for r, p in epoch.build_programs(d).items()}
    return {"row": ROW, "equal": True, "ranks": {r: {**{n: {"traced": v, "predicted": v, "equal": True} for n, v in ps.items()},
                                                     "secs": {"step": 1.0}} for r, ps in ranks.items()}}


def test_a_row_is_switched_by_a_comparison_equal_on_every_rank_and_publishes_only_its_aot_build(tmp_path, store):
    d = _row_dir(tmp_path)
    run, trace, cmp = "r20261018-001500-aaaa", "art:" + "5" * 64, _comparison(d)
    with pytest.raises(UsageError, match="NOT equal"):
        circuit_db.switch(ROW, {**cmp, "equal": False}, run=run, trace=trace, tier="fast")
    with pytest.raises(UsageError, match="the comparison's run id"):
        circuit_db.switch(ROW, cmp, run="20261018", trace=trace, tier="fast")
    with pytest.raises(UsageError, match="needs --tier"):
        circuit_db.switch(ROW, cmp, run=run, trace=trace)
    with pytest.raises(UsageError, match="a single-GPU row's Build traces"):
        circuit_db.switch(ROW.replace("__tp2__", "__tp1__"), {**cmp, "row": ROW.replace("__tp2__", "__tp1__")}, run=run, trace=trace, tier="fast")
    e = circuit_db.switch(ROW, cmp, run=run, trace=trace, tier="fast")
    assert e == {"record": None, "tier": "fast", "pinned_from": {"build": trace, "run": run}, "manifest_digest": None,
                 "programs": epoch.build_programs(d), "aot": {"comparison": run, "trace": trace}}
    assert CR.aot_of(ROW, circuit_db.INDEX) == e["aot"]
    moved = {**cmp, "ranks": {**cmp["ranks"], "1": {**cmp["ranks"]["1"], "step": {"traced": "x", "predicted": "x", "equal": True}}}}
    with pytest.raises(UsageError, match="re-pin it first"):
        circuit_db.switch(ROW, moved, run=run, trace=trace)

    build = _build_art(store, d, tmp_path / "run")
    assert circuit_db.constructions(d) == {r: {"build_step": "trace", "build_request_LP9_T2": "trace"} for r in ("0", "1")}
    with pytest.raises(UsageError, match=r"this Build traced 4 of them \(rank0 build_request_LP9_T2.*--build-construction record"):
        circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", summary=_served(d))
    s = json.loads((d / "build_summary.json").read_text())
    _w(d / "build_summary.json", {**s, "ranks": {r: {p: {**v, "construction": "aot"} if p.startswith("build_") else v for p, v in rk.items()}
                                                 for r, rk in s["ranks"].items()}})
    build = _build_art(store, d, tmp_path / "run2")
    art = circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", summary=_served(d))
    assert circuit_db.index()[ROW] == {"record": art, "tier": "fast", "pinned_from": {"record": art, "build": build, "tree": "cursor/x@1b53f685a"},
                                    "manifest_digest": "mm", "programs": e["programs"], "aot": e["aot"]}


def test_switch_refuses_a_row_not_done_only_at_its_public_cap_before_it_touches_the_index(tmp_path, store):
    d = _row_dir(tmp_path)
    run, trace, cmp = "r20261018-001500-aaaa", "art:" + "5" * 64, _comparison(d)
    wl, eos = CR.WORKLOADS / f"{ROW}.json", json.loads((WORKLOADS / f"{ROW}.json").read_text())
    _w(wl, eos)
    with pytest.raises(UsageError, match=rf"{ROW}: a row of record \(records\.json\) is done only at its public cap, and its done source "
                                         r"is 'eos-token'.*switch would index it as one, so it switches nothing"):
        circuit_db.switch(ROW, cmp, run=run, trace=trace, tier="fast")
    assert not circuit_db.INDEX.exists()
    _w(wl, _capped(eos))
    circuit_db.switch(ROW, cmp, run=run, trace=trace, tier="fast")
    indexed = circuit_db.index()
    _w(wl, {**_capped(eos), "sampling": {**_capped(eos)["sampling"], "stop": ["</s>"]}})
    with pytest.raises(UsageError, match=r"sampling\.stop \['</s>'\] stops a request before its cap"):
        circuit_db.switch(ROW, cmp, run=run, trace=trace)
    assert circuit_db.index() == indexed, "a switched row whose file has lost its cap is not switched again"


def test_a_job_reads_the_programs_from_a_bundle_with_no_store(tmp_path, store, monkeypatch):
    d = _row_dir(tmp_path)
    build = _build_art(store, d, tmp_path / "run")
    circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=_served(d))
    b = circuit_db.bundle(ROW, tmp_path / "job.tar", [1], part=["workload"])
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "no-store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "no-store.toml"))
    monkeypatch.setattr(circuit_db, "INDEX", tmp_path / "no-index.json")
    progs = circuit_db.programs(ROW, [1], part=["workload"], bundle=b)
    assert {r: sorted(ps) for r, ps in progs.items()} == {1: ["workload"]} and progs[1]["workload"].gates == make_program().gates
    assert circuit_db.record(ROW, b)["evidence"]["build"] == build
    with pytest.raises(UsageError, match="lacks 1 descriptor"):
        circuit_db.programs(ROW, [0], part=["workload"], bundle=b)
    with pytest.raises(UsageError, match="the record of .*, not other-row"):
        circuit_db.programs("other-row", bundle=b)
    with pytest.raises(UsageError, match="not in the circuit database"):
        circuit_db.programs(ROW, [1])
    with pytest.raises(RuntimeError, match=r"no remote configured.*a research run job has no store.*records fetch ROW --to FILE.*"
                                           r"--send FILE.*--bundle"):
        circuit_db.fetch(build)


def test_a_bundle_whose_record_is_not_a_file_is_refused(tmp_path):
    b = tmp_path / "job.tar"
    with tarfile.open(b, "w") as tf:
        member = tarfile.TarInfo(CR.NAME)
        member.type = tarfile.DIRTYPE
        tf.addfile(member)
    with pytest.raises(UsageError, match=f"its {CR.NAME} is not a file"):
        circuit_db.record(ROW, b)


def test_each_records_subcommand_runs_its_own_function_with_its_options(tmp_path, monkeypatch, capsys):
    """`records SUBCOMMAND ...`, parsed as the command line parses it, reaches that subcommand's function and no other."""
    cmp = tmp_path / "comparison.json"
    _w(cmp, {"row": ROW})
    calls: list[tuple] = []

    def stand_in(name, result):
        def call(*args, **kw):
            calls.append((name, args, kw))
            return result
        monkeypatch.setattr(circuit_db, name, call)

    ev, tar = {"build": "art:b"}, tmp_path / "job.tar"
    for name, result in (("switch", {"aot": {"run": "r1"}}), ("evidence_of", ev), ("publish", "art:rec"), ("pull", ROW),
                         ("capture", {PROP: "art:p"}), ("record", {"config": {"row": ROW}}), ("index", {ROW: {"record": "art:rec"}}),
                         ("bundle", tar), ("staged", "staged\n"), ("parts", {1: {"workload": "ab" * 32}}),
                         ("programs", {1: {"workload": SimpleNamespace(gates=7, defs={"Mul16_v1": None})}})):
        stand_in(name, result)
    cases = [
        (["switch", ROW, "--comparison", str(cmp), "--run", "r1", "--trace", "art:t"],
         [("switch", (ROW, {"row": ROW}), {"run": "r1", "trace": "art:t", "tier": None})], '{"run": "r1"}'),
        (["publish", "DIR", ROW, "--build", "art:b", "--tree", "x@1", "--by", "circuits", "--gap", f"{PROP}=no capture yet"],
         [("evidence_of", ("art:b",), {"manifest": None, "match": None, "match_run": None, "commit": None, "commit_run": None,
                                       "conformance": {}, "gaps": {PROP: "no capture yet"}}),
          ("publish", ("DIR", ROW), {"evidence": ev, "tree": "x@1", "by": "circuits", "summary": None, "tier": None, "newest": False})],
         "art:rec"),
        (["pull", "art:1", "art:2", "--tier", "fast"], [("pull", ("art:1", "fast"), {}), ("pull", ("art:2", "fast"), {})],
         f"{ROW} art:1\n{ROW} art:2"),
        (["capture", "r1", "--property", PROP], [("capture", ("r1", [PROP]), {})], f"{PROP}=art:p"),
        (["show"], [("index", (), {})], json.dumps({ROW: "art:rec"}, indent=1, sort_keys=True)),
        (["show", ROW], [("record", (ROW,), {})], json.dumps({"config": {"row": ROW}}, indent=1, sort_keys=True)),
        (["fetch", ROW, "--to", str(tar), "--rank", "1"], [("bundle", (ROW, tar, [1]), {"part": None})], str(tar)),
        (["staged", REPO, "abc", "--at", "2026-01-01T00:00:00Z", "--ref", "r1", "--by", "circuits"],
         [("staged", (REPO, "abc"), {"at": "2026-01-01T00:00:00Z", "ref": "r1", "by": "circuits"})], "staged"),
        (["programs", ROW, "--part", "workload", "--bundle", str(tar)],
         [("record", (ROW, tar), {}), ("parts", ({"config": {"row": ROW}},), {}),
          ("programs", (ROW, [1]), {"part": ["workload"], "bundle": tar})],
         f"rank 1  {'workload':28s} {'ab' * 8}  gates 7  definitions 1"),
    ]
    assert {c[0][0] for c in cases} == set(circuit_db.COMMANDS)
    for argv, want, printed in cases:
        calls.clear()
        assert circuit_db.main(cli.parse("records", argv, {})) == 0
        assert calls == want, argv
        assert capsys.readouterr().out.rstrip("\n") == printed, argv


# What each reader makes of a document that leaves an optional field out, or holds it null or empty.

@pytest.mark.parametrize("build", [{}, {"programs": None}, {"programs": {}}, {"per_rank": None}, {"per_rank": {}, "programs": {}}])
def test_a_record_whose_build_names_no_program_has_rank_0_alone_with_no_part(build):
    rec = {"build": {**build, "workload_digest": "w"}, "evidence": {"build": "art:b"}, "tree": "x@1234567"}
    assert circuit_db.parts({"build": build}) == {0: {}} and circuit_db.parts(rec) == {0: {"workload": "w"}}
    assert circuit_db.entry(rec, "art:r", "fast")["programs"] == {"0": {"step": None, "workload": "w", "requests": {}}}


@pytest.mark.parametrize("rank", [{}, {"programs": None}, {"programs": {}}])
def test_a_rank_that_names_no_program_has_its_workload_program_alone(rank):
    assert circuit_db.parts({"build": {"per_rank": {"1": {**rank, "workload_digest": "w"}, "0": rank}}}) == {1: {"workload": "w"}, 0: {}}


@pytest.mark.parametrize("attempt", [{}, {"outputs": None}, {"outputs": {}}, {"outputs": {"run_record": None}}])
def test_a_run_whose_attempt_states_no_run_record_is_refused(attempt, monkeypatch):
    monkeypatch.setattr(circuit_db, "data", lambda *args: json.dumps(attempt))
    with pytest.raises(UsageError, match=r"r1: no run record \(outputs\.run_record"):
        circuit_db.capture("r1")


@pytest.mark.parametrize("source", ["absent", None, {}])
def test_a_run_whose_attempt_states_no_source_is_preserved_with_no_commit(source, tmp_path, store, monkeypatch):
    run, real = _capture_run(store, tmp_path), circuit_db.data

    def data(*args: str) -> str:
        out = real(*args)
        if args[:2] != ("show", run):
            return out
        attempt = {k: v for k, v in json.loads(out).items() if k != "source"}
        return json.dumps(attempt if source == "absent" else {**attempt, "source": source})

    monkeypatch.setattr(circuit_db, "data", data)
    assert store.get_manifest(circuit_db.capture(run)[PROP]).meta["commit"] is None


@pytest.mark.parametrize("unstated", [{}, {"meta": None}, {"meta": {}}])
def test_an_artifact_whose_manifest_states_no_meta_is_no_capture_and_no_record_to_pull(unstated, monkeypatch):
    held = {"id": "art:a", "manifest": {"kind": circuit_db.PROPERTY_KIND, **unstated}}
    monkeypatch.setattr(circuit_db, "data", lambda *args: json.dumps([held] if args[0] == "select" else held))
    with pytest.raises(UsageError, match=rf"--conformance {PROP}=art:a: a property-record/v1 of None at None \(ok None\), not a holding"):
        circuit_db.check_capture(PROP, "art:a")
    assert circuit_db.newest_capture(PROP) is None
    held = {"id": "art:a", "manifest": {"kind": CR.KIND, **unstated}}
    with pytest.raises(UsageError, match=r"art:a: a vllm-config-record/v1 \(None\), not a PASS vllm-config-record/v1"):
        circuit_db.pull("art:a")


def test_the_newest_capture_is_one_that_holds_though_it_names_no_run(monkeypatch):
    held = [{"id": "art:a", "manifest": {"meta": {"ok": True, "schema": "another/v0", "run_id": "r9"}}},
            {"id": "art:b", "manifest": {"meta": {"ok": True, "schema": REGISTRY[PROP].schema}}},
            {"id": "art:c", "manifest": {"meta": {"ok": False, "schema": REGISTRY[PROP].schema, "run_id": "r9"}}}]
    monkeypatch.setattr(circuit_db, "data", lambda *args: json.dumps(held))
    assert circuit_db.newest_capture(PROP) == "art:b"


def _construction(tmp_path: Path, monkeypatch, doc: dict) -> list[str]:
    """The listing of a Build whose one construction manifest is `doc`, which `due` fetches from under `tmp_path`."""
    p = f"build_step/{circuit_db.CONSTRUCTION}"
    _w(tmp_path / "fetched" / p, doc)
    monkeypatch.setattr(circuit_db, "fetch", lambda art, to, paths: tmp_path / "fetched")
    return [p]


@pytest.mark.parametrize("doc", [{}, {"model": None, "tp": None}, {"model": {}, "tp": {}}, {"model": {"pin": None}, "tp": {"world": None}},
                                 {"model": {"pin": {}}, "tp": {"world": 0}}, {"model": {"pin": {"model": None}}, "tp": {"world": 1}}])
def test_a_construction_manifest_that_states_no_checkpoint_has_no_map_and_nothing_due(doc, tmp_path, monkeypatch):
    assert circuit_db.due("art:b", _construction(tmp_path, monkeypatch, doc)) == {"map": None, "due": {}}


@pytest.mark.parametrize("histogram", [{}, {"spec_histogram": None}, {"spec_histogram": {}}])
def test_a_construction_manifest_that_states_no_definition_has_its_map_and_nothing_due(histogram, tmp_path, store, monkeypatch):
    _map(tmp_path)
    doc = {"model": {"pin": {"model": REPO}}, "tp": {"world": 2}, **histogram}
    assert circuit_db.due("art:b", _construction(tmp_path, monkeypatch, doc)) == {"map": "llama-test-tp2", "due": {}}
    used = {**doc, "spec_histogram": {"Mul16_v1": 3}}
    assert circuit_db.due("art:b", _construction(tmp_path, monkeypatch, used)) == {"map": "llama-test-tp2", "due": {PROP: ["mul"]}}


def test_a_row_the_index_holds_nothing_of_needs_its_tier(monkeypatch):
    monkeypatch.setattr(circuit_db, "index", lambda: {"null": None, "empty": {}, "tiered": {"tier": "heavy"}})
    for row in ("new", "null", "empty"):
        with pytest.raises(UsageError, match=rf"{row}: a row new to records\.json needs --tier"):
            circuit_db._tier(row, None)
        assert circuit_db._tier(row, "fast") == "fast"
    assert circuit_db._tier("tiered", None) == "heavy" and circuit_db._tier("tiered", "fast") == "fast"


@pytest.mark.parametrize("manifest", ["absent", None, {}])
def test_publish_holds_a_commit_that_states_no_manifest_to_no_build(manifest, tmp_path, store, monkeypatch):
    """The summary's manifest is another Build's, which `publish` refuses when the record's Commit states it."""
    d, real = _row_dir(tmp_path), CR.record

    def record(*args, **kw) -> dict:
        rec = real(*args, **kw)
        commit = {k: v for k, v in rec["commit"].items() if k != "manifest"}
        return {**rec, "commit": commit if manifest == "absent" else {**commit, "manifest": manifest}}

    build = _build_art(store, d, tmp_path / "run")
    served = {**_served(d), "required_manifest": {"manifest_digest": "other", "program_digest_per_rank": {"0": "0" * 128}}}
    with pytest.raises(UsageError, match="not the one served"):
        circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=served)
    monkeypatch.setattr(CR, "record", record)
    art = circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=served)
    assert circuit_db.index()[ROW]["record"] == art


def test_publish_reads_the_record_s_build_before_what_its_commit_served(tmp_path, store, monkeypatch):
    """A Build whose Programs are no mapping raises where its parts are read, also where the Commit's manifest is no mapping."""
    d, real = _row_dir(tmp_path), CR.record

    def record(*args, **kw) -> dict:
        rec = real(*args, **kw)
        return {**rec, "commit": {**rec["commit"], "manifest": "no mapping"}, "build": {"per_rank": {"0": {"programs": ["no mapping"]}}}}

    build = _build_art(store, d, tmp_path / "run")
    monkeypatch.setattr(CR, "record", record)
    with pytest.raises(AttributeError, match="'list' object has no attribute 'items'"):
        circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=_served(d))


@pytest.mark.parametrize("unstated", [{}, {"conformance": None, "gaps": None}, {"conformance": {}, "gaps": {}}])
def test_publish_reads_evidence_that_states_no_conformance_and_no_gap_as_citing_nothing(unstated, tmp_path, store):
    _map(tmp_path)
    d = _row_dir(tmp_path)
    ev = {"build": _build_art(store, d, tmp_path / "run"), **unstated}
    with pytest.raises(UsageError, match=f"need {PROP} .*which the record neither cites nor names a gap"):
        circuit_db.publish(d, ROW, evidence=ev, tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=_served(d))
    new = circuit_db.capture(_capture_run(store, tmp_path))[PROP]
    art = circuit_db.publish(d, ROW, evidence=ev, tree="cursor/x@1b53f685a", by="circuits", tier="fast", summary=_served(d), newest=True)
    assert circuit_db.record(ROW)["evidence"] == {**ev, "conformance": {PROP: new}} and store.get_manifest(art).refs == {"build": ev["build"], PROP: new}


@pytest.mark.parametrize("entry", [None, {}])
def test_pull_reads_an_index_entry_that_holds_nothing_as_a_row_on_the_trace(entry, tmp_path, store, monkeypatch):
    d = _row_dir(tmp_path)
    build = _build_art(store, d, tmp_path / "run")
    art = circuit_db.publish(d, ROW, evidence=circuit_db.evidence_of(build), tree="cursor/x@1b53f685a", by="circuits", tier="heavy",
                          summary=_served(d))
    pod = circuit_db.index()
    monkeypatch.setattr(circuit_db, "INDEX", tmp_path / "checkout.json")
    _w(circuit_db.INDEX, {"schema": CR.INDEX_SCHEMA, "rows": {ROW: entry}})
    assert circuit_db.pull(art) == ROW and circuit_db.index() == pod


def test_a_row_dir_whose_build_summary_states_no_rank_has_no_construction(tmp_path):
    assert circuit_db.constructions(tmp_path) == {}
    unstated: list[dict] = [{}, {"ranks": None}, {"ranks": {}}]
    for summary in unstated:
        _w(tmp_path / "build_summary.json", summary)
        assert circuit_db.constructions(tmp_path) == {}
