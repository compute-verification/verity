"""research_outputs on a tp_stage.sh evidence dir: rank Builds under build/rank<r>/, the `tp-ranks` digest of record, TP Match record."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from verity_vllm.pipeline import research_outputs as ro
from verity_vllm.pipeline import research_result, serve_commit

ROW = "llama32-1b__bf16__l40s__tp4__b8__i256__o32__mixed__greedy__bi-eager"
COMMIT_SH = Path(__file__).resolve().parents[4] / "demos" / "vllm" / "commit.sh"
WDS = ["d" * 64, "a" * 64, "c" * 64, "b" * 64]
#: an evidence dir's files that hold request values, where each rank ran or when each process ran: never published
PRIVATE_FILES = {"match/capture/tokens.json": {"prompt_token_ids": [[1, 2]], "generated_token_ids": [[3]], "timing": {}},
                 "match/control/tokens.json": {"prompt_token_ids": [[1, 2]]}, "commit/tokens_p0.json": {"generated_token_ids": [[3]]},
                 "match_stoch.json": {"rows": [{"served": 3, "recomputed": 3}]},
                 "commit/sampled_replay_p0.json": {"facts_of_record": {"served": [3]}},
                 "commit/sampled_replay.json": {"facts_of_record": {"served": [3]}},
                 "serve/requests.jsonl": '{"engine_id": "0", "prompt_ids": [1, 2], "output_ids": [3]}\n',
                 "serve/step_trace.jsonl": '{"step": 0, "rows": {"0": 2}}\n',
                 "serve/advice_trace.jsonl": '{"kind": "arrive", "request": "0", "prompt": 2, "params": {"seed": 1}}\n',
                 "load.jsonl": '{"user": "u", "prompt": 2, "response": {"ids": [3]}}\n', "rep-0/load-1.jsonl": "{}\n",
                 "match/capture/topology.json": {"ranks": [{"pid": 7, "cuda_visible_devices": "0"}]},
                 "match/control/topology.json": {"driver_pid": 8}, "build/rank0/build_step/topology.json": {"build_s": 1.0},
                 "timeline.jsonl": '{"utc": "2026-10-28T00:00:00Z", "pod": "computeinstance-x"}\n', "timeline_report.json": {}}


def _w(p: Path, doc) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc) if not isinstance(doc, str) else doc)


def _tp_dir(tmp_path: Path, world: int = 4) -> Path:
    d = tmp_path / ROW
    ranks = {}
    for r in range(world):
        rd = d / "build" / f"rank{r}"
        _w(rd / "build_step" / "result.json", {"ok": True})
        _w(rd / "build_request_LP9_T2" / "result.json", {"ok": True})
        _w(rd / "build_request_LP9_T2" / "instances.json.gz", "planes")
        _w(rd / "build_request_LP9_T2.log", "derive log")
        _w(rd / "build_workload" / "workload_program.json", {"workload_digest": WDS[r]})
        _w(rd / "build_workload" / "descriptor.json.gz", "workload descriptor")
        _w(rd / "build_workload" / "address_map.json", "{}")
        ranks[f"rank{r}"] = {"build_step": {"ok": True, "program_digest": f"{r}" * 64}, "workload": {"ok": True, "workload_digest": WDS[r]}}
    _w(d / "build_summary.json", {"tp": {"world": world}, "ranks": ranks, "step": {"program_digest": "0" * 64}})
    _w(d / "manifest.json", {"program_digest": "x"})
    _w(d / "match" / "tp2_match.json", {"pass": True})
    _w(d / "match_summary.json", {"pass": True})
    _w(d / "stages.txt", "build PASS 2026-09-23T00:00:00Z rc=0\nmatch PASS 2026-09-23T00:00:01Z rc=0/0/0/-1\n")
    return d


def test_tp_digest_of_record_is_the_rank_fold(tmp_path):
    d = _tp_dir(tmp_path)
    dig = ro.program_digest(d, ROW)
    assert dig["tp"] == 4
    assert dig["rank_workload_digests"] == WDS
    assert dig["step_digest"] == "0" * 64
    assert dig["program_digest"] == hashlib.sha256(("tp-ranks/" + ",".join(sorted(WDS))).encode()).hexdigest()


def test_tp_digest_absent_when_a_rank_has_no_workload_program(tmp_path):
    d = _tp_dir(tmp_path)
    bs = json.loads((d / "build_summary.json").read_text())
    bs["ranks"]["rank2"]["workload"] = {"ok": False, "workload_digest": None}
    _w(d / "build_summary.json", bs)
    assert ro.program_digest(d, ROW)["program_digest"] is None


def test_tp_build_and_match_outputs_take_the_rank_files(tmp_path):
    d = _tp_dir(tmp_path)
    run = tmp_path / "run"
    doc = ro.build_outputs(d, ROW, "build", run, {})
    build = next(e for e in doc["outputs"] if e["name"] == "build")
    staged = {p.relative_to(run / "outputs" / "build").as_posix() for p in (run / "outputs" / "build").rglob("*") if p.is_file()}
    for r in range(4):
        assert f"build/rank{r}/build_step/result.json" in staged
        assert f"build/rank{r}/build_request_LP9_T2/result.json" in staged
        assert f"build/rank{r}/build_workload/workload_program.json" in staged
        assert f"build/rank{r}/build_workload/descriptor.json.gz" in staged
        assert f"build/rank{r}/build_workload/address_map.json" not in staged
        assert f"build/rank{r}/build_request_LP9_T2/instances.json.gz" not in staged
    assert {"manifest.json", "build_summary.json"} <= staged
    assert build["meta"]["program_digest"] == ro.program_digest(d, ROW)["program_digest"]
    assert build["meta"]["required_manifest"] is True
    logs = next(e for e in doc["outputs"] if e["name"] == "logs")
    assert logs["meta"]["files"] == 4

    doc = ro.build_outputs(d, ROW, "match", tmp_path / "run_m", {"build": "art:" + "f" * 64})
    match = next(e for e in doc["outputs"] if e["name"] == "match")
    assert match["meta"]["tp_match_ok"] is True
    assert match["refs"] == {"build": "art:" + "f" * 64}
    assert (tmp_path / "run_m" / "outputs" / "match" / "match" / "tp2_match.json").is_file()


def _commit_line(d: Path, word: str) -> None:
    with open(d / "stages.txt", "a") as f:
        f.write(f"commit {word} 2026-09-23T00:00:02Z rc=0\n")


def test_tp_commit_reads_tp_commit_summary(tmp_path):
    d = _tp_dir(tmp_path)
    _w(d / "commit" / "summary.json", {"pass": True, "tokens_equal": True, "pairs": 3})
    _commit_line(d, "PASS")
    doc = ro.build_outputs(d, ROW, "commit", tmp_path / "run_c", {"build": "art:" + "f" * 64, "match": "art:" + "e" * 64})
    v = next(e for e in doc["outputs"] if e["name"] == "verdict")
    assert v["meta"]["decision_record"] == "commit/summary.json"
    assert (v["meta"]["tp_pass"], v["meta"]["tokens_equal"], v["meta"]["pairs"]) == (True, True, 3)
    assert "decision_record_missing" not in v["meta"]


def test_commit_pass_without_its_decision_document_is_refused(tmp_path):
    d = _tp_dir(tmp_path)
    _commit_line(d, "PASS")
    try:
        ro.build_outputs(d, ROW, "commit", tmp_path / "run_c", {})
    except ValueError as e:
        assert "commit/summary.json" in str(e)
    else:
        raise AssertionError("a PASS Commit without commit/summary.json published")
    t1 = tmp_path / "t1"
    _w(t1 / "stages.txt", "commit PASS 2026-09-23T00:00:02Z rc=0\n")
    _w(t1 / "commit" / "verdict.json", {"verdict": "PASS"})
    try:
        ro.build_outputs(t1, "smollm2-135m__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager", "commit", tmp_path / "run_t1", {})
    except ValueError as e:
        assert "verdict.json" in str(e)
    else:
        raise AssertionError("a PASS TP1 Commit without verdict.json published")


def test_failed_commit_names_the_missing_decision_document(tmp_path):
    d = _tp_dir(tmp_path)
    _commit_line(d, "FAIL")
    _w(d / "commit.log", "log")
    doc = ro.build_outputs(d, ROW, "commit", tmp_path / "run_c", {})
    assert all(e["name"] != "verdict" or e["meta"]["decision_record_missing"] == "commit/summary.json" for e in doc["outputs"])


def test_tp1_commit_reads_the_verdict_of_record(tmp_path):
    d = tmp_path / "row"
    _w(d / "stages.txt", "commit PASS 2026-09-23T00:00:02Z rc=0\n")
    _w(d / "verdict.json", {"outcome": "PASS", "runs": {"instrumented": 3, "control": 3, "failed": 0}})
    doc = ro.build_outputs(d, "smollm2-135m__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager", "commit", tmp_path / "run", {})
    v = next(e for e in doc["outputs"] if e["name"] == "verdict")
    assert (v["meta"]["decision_record"], v["meta"]["outcome"], v["meta"]["n_runs"], v["meta"]["runs_failed"]) == ("verdict.json", "PASS", 6, 0)


def _config_commit(d: Path, outcome: str = "PASS") -> None:
    _w(d / "commit" / "verdict.json", {"verdict": "PASS"})
    _w(d / "commit" / "summary.json", {"pass": True})
    _w(d / "config_record.json", {"schema": "verity-vllm/config-record/v1", "outcome": outcome, "why": None,
                                  "replay": {"equal": 460, "k": 460}})
    with open(d / "stages.txt", "a") as f:
        f.write(f"commit PASS 2026-09-23T00:00:02Z rc=0\nconfig {outcome} 2026-09-23T00:00:03Z replay 460/460 equal\n")


def test_config_run_commit_decides_in_its_config_record_at_tp1_and_tp(tmp_path):
    t1 = tmp_path / "t1"
    _w(t1 / "stages.txt", "build PASS 2026-09-23T00:00:00Z rc=0\n")
    for d, row in ((t1, "smollm2-135m__bf16__rtxpro6000__tp1__b1__i256__o32__mixed__greedy__bi-eager"), (_tp_dir(tmp_path, world=2), ROW)):
        _config_commit(d)
        run = tmp_path / f"run_{d.name}"
        doc = ro.build_outputs(d, row, "commit", run, {"build": "art:" + "f" * 64})
        v = next(e for e in doc["outputs"] if e["name"] == "verdict")
        assert v["meta"]["decision_record"] == "config_record.json" and "decision_record_missing" not in v["meta"]
        assert (v["meta"]["outcome"], v["meta"]["replay_equal"], v["meta"]["replay_k"]) == ("PASS", 460, 460)
        assert (run / "outputs" / "verdict" / "config_record.json").is_file()
        assert (run / "outputs" / "verdict" / "commit" / "verdict.json").is_file()


def test_config_run_commit_pass_without_its_config_record_is_refused(tmp_path):
    d = tmp_path / "row"
    _config_commit(d)
    (d / "config_record.json").unlink()
    try:
        ro.build_outputs(d, "smollm2-135m__bf16__rtxpro6000__tp1__b1__i256__o32__mixed__greedy__bi-eager", "commit", tmp_path / "run", {})
    except ValueError as e:
        assert "config_record.json" in str(e)
    else:
        raise AssertionError("a PASS config-run Commit without config_record.json published")


def test_tp1_layout_unchanged(tmp_path):
    d = tmp_path / "row"
    _w(d / "build_summary.json", {"request": {"program_digest": "r" * 64}, "step": {"program_digest": "s" * 64}})
    dig = ro.program_digest(d, "smollm2-135m__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager")
    assert dig == {"program_digest": "r" * 64, "request_digest": "r" * 64, "step_digest": "s" * 64, "workload_digest": None}


def test_a_replay_stage_publishes_the_config_record_it_decided(tmp_path):
    row = "smollm2-135m__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager"
    d = tmp_path / "row"
    _w(d / "stages.txt", "commit PENDING 2026-09-30T00:00:01Z rc=0\nreplay PASS 2026-09-30T00:00:02Z rc=0\n")
    _w(d / "config_record.json", {"outcome": "PASS"})
    _w(d / "commit" / "verdict.json", {"commit_pass": True})
    doc = ro.build_outputs(d, row, "replay", tmp_path / "run", {})
    v = next(e for e in doc["outputs"] if e["name"] == "verdict")
    assert (v["kind"], v["meta"]["decision_record"], v["meta"]["outcome"]) == ("vllm-verdict/v1", "config_record.json", "PASS")
    assert (tmp_path / "run" / "outputs" / "verdict" / "config_record.json").is_file()
    (d / "config_record.json").unlink()
    try:
        ro.build_outputs(d, row, "replay", tmp_path / "run2", {})
    except ValueError as e:
        assert "config_record.json" in str(e)
    else:
        raise AssertionError("a PASS replay without its config record published")


def test_a_tp_replay_stage_decides_in_the_folded_commit_summary_unless_a_config_run(tmp_path):
    d = _tp_dir(tmp_path, world=2)
    with open(d / "stages.txt", "a") as f:
        f.write("commit PENDING 2026-09-30T00:00:01Z rc=0\nreplay PASS 2026-09-30T00:00:02Z rc=0\n")
    _w(d / "commit" / "summary.json", {"pass": True, "tokens_equal": True, "pairs": 1})
    _w(d / "commit" / "replay_bundle_p0_rank1" / "MANIFEST.json", {})
    doc = ro.build_outputs(d, ROW, "replay", tmp_path / "run", {})
    v = next(e for e in doc["outputs"] if e["name"] == "verdict")
    assert (v["meta"]["decision_record"], v["meta"]["tp_pass"], v["meta"]["pairs"]) == ("commit/summary.json", True, 1)
    assert (tmp_path / "run" / "outputs" / "verdict" / "commit" / "summary.json").is_file()
    assert not [x for x in (tmp_path / "run").rglob("*") if "replay_bundle_p" in x.as_posix()]
    (d / "commit" / "summary.json").unlink()
    try:
        ro.build_outputs(d, ROW, "replay", tmp_path / "run2", {})
    except ValueError as e:
        assert "commit/summary.json" in str(e)
    else:
        raise AssertionError("a PASS TP replay without its folded summary published")
    _config_commit(d)
    v = next(e for e in ro.build_outputs(d, ROW, "replay", tmp_path / "run3", {})["outputs"] if e["name"] == "verdict")
    assert (v["meta"]["decision_record"], v["meta"]["outcome"]) == ("config_record.json", "PASS")


def test_a_replay_stage_publishes_neither_its_slim_keep_nor_the_bundle(tmp_path):
    """A keep holds clear values (committed values with their salts, the requests and tokens, opened weights): it stays on the node."""
    row = "smollm2-135m__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager"
    d = tmp_path / "row"
    _w(d / "stages.txt", "commit PENDING 2026-09-30T00:00:01Z rc=0\nreplay PASS 2026-09-30T00:00:02Z rc=0\n")
    _w(d / "config_record.json", {"outcome": "PASS"})
    _w(d / "commit" / "verdict.json", {"commit_pass": True})
    k = d / "commit" / "replay_slim_p1"
    _w(k / "slim.json", {"pair": 1, "run_root": "ab" * 32, "picked": 460})
    _w(k / "store" / "store.json", {"steps": []})
    (k / "store" / "salts").mkdir()
    (k / "store" / "salts" / "s0.bin").write_bytes(b"\x01" * 64)
    _w(d / "commit" / "replay_bundle_p1" / "MANIFEST.json", {})
    doc = ro.build_outputs(d, row, "replay", tmp_path / "run", {})
    assert [(e["name"], e["kind"]) for e in doc["outputs"]] == [("verdict", "vllm-verdict/v1"), ("evidence", "execution-evidence/v1")]
    assert not [x for x in (tmp_path / "run").rglob("*") if "replay_slim_p" in x.as_posix() or "replay_bundle_p" in x.as_posix()]


def test_no_stage_publishes_request_values_placement_or_timing(tmp_path):
    """Every stage at TP1 and TP, alone or as one run's stages, publishes none of PRIVATE_FILES wherever it sits (a listed record, under
    a glob dir, or the evidence), and the sweep's result.json copies none beside the run: the requests' prompt and served token ids
    are values, and where each part of the workload ran and when are private."""
    t1 = tmp_path / "t1"
    _w(t1 / "build_summary.json", {"request": {"program_digest": "r" * 64}, "step": {"program_digest": "s" * 64}})
    _w(t1 / "verdict.json", {"outcome": "PASS"})
    for d, row in ((t1, "smollm2-135m__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager"), (_tp_dir(tmp_path, world=2), ROW)):
        _w(d / "stages.txt", "".join(f"{s} PASS 2026-10-28T00:00:0{i}Z rc=0\n" for i, s in enumerate(ro.KINDS)))
        for rel, doc in {"commit/summary.json": {"pass": True}, "commit/verdict.json": {"commit_pass": True}, "config_record.json":
                         {"outcome": "PASS"}, "match/run.json": {}, "global_match.json": {"verdict": "PASS"}, **PRIVATE_FILES}.items():
            _w(d / rel, doc)
        for stages in [[s] for s in ro.KINDS] + [list(ro.KINDS)]:
            run = tmp_path / f"run_{d.name}_{'_'.join(stages)}"
            ro.build_stages(d, row, stages, run, {})
            staged = {p.relative_to(run / "outputs").as_posix().split("/", 1)[1] for p in (run / "outputs").rglob("*") if p.is_file()}
            assert "stages.txt" in staged and not staged & set(PRIVATE_FILES), (d.name, stages, staged & set(PRIVATE_FILES))
        doc = research_result.build_result(d, row, "v2", ["build", "match", "commit"], "r", tmp_path / f"result_{d.name}")
        assert "evidence/match/run.json" in doc["artifacts"]
        assert not {f.removeprefix("evidence/") for f in doc["artifacts"]} & set(PRIVATE_FILES), doc["artifacts"]
    assert [rel for rel in PRIVATE_FILES if not ro.private(rel)] == [] and [r for r in research_result.RECORDS if ro.private(r)] == []
    assert [n for n in serve_commit.MIRRORED if ro.private(n)] == [], "a served run's $OUT gets no request value"


def test_the_example_commit_publishes_its_stages_records_and_none_of_its_private_files(tmp_path):
    """demos/vllm/commit.sh publishes what `research-outputs write` declares for the stages it names, and copies nothing into its
    run record: the sealed bundle (the committed store with its salts, Kw in weights/tree.json, the opened weights, the requests'
    context, the manifest of the values the store doesn't retain), the slim keep and the Match's capture planes stay on the node."""
    lines = [ln for ln in COMMIT_SH.read_text().splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in lines if "data put" in ln or (re.search(r"\bcp\b", ln) and "$OUT" in ln)]
    writes = [m.group(1).split() for ln in lines if (m := re.search(r'research-outputs write "\$D" "\$ROW" ([a-z ]+)', ln))]
    assert len(writes) == 1, f"commit.sh publishes through one `research-outputs write \"$D\" \"$ROW\" STAGE...`: {writes}"
    (stages,) = writes
    row = "llama32-1b__bf16__rtxpro6000__tp1__b4__i256__o32__mixed__greedy__bi-eager"
    d = tmp_path / row
    ran = [("build", "PASS"), ("plan", "PASS"), ("config", "FAIL"), ("match", "PASS"), ("commit", "PENDING"), ("replay", "PASS")]
    _w(d / "stages.txt", "".join(f"{s} {v} 2026-10-24T06:00:0{i}Z rc=0\n" for i, (s, v) in enumerate(ran)))
    public = {"build_summary.json": {"request": {"program_digest": "r" * 64}}, "build_workload/workload_program.json": {},
              "plan/commit_plan.json": {"key": "k"}, "match/program.json": {}, "global_match.json": {"verdict": "PASS"},
              "config_record.json": {"outcome": "PASS"}, "commit/verdict.json": {"commit_pass": True}, "commit.log": "log"}
    private = {"commit/replay_bundle_p0/store/store.json": {"steps": []}, "commit/replay_bundle_p0/store/salts/s0.bin": "salt",
               "commit/replay_bundle_p0/weights/tree.json": {"key": "aa" * 64}, "commit/replay_bundle_p0/weights/extra/0.bin": "w",
               "commit/replay_bundle_p0/context.json": {"requests": [{"prompt_token_ids": [1, 2]}]},
               "commit/replay_bundle_p0/MANIFEST.json": {"committed_not_retained": {}}, "commit/replay_bundle_p0/binding_map.json": {},
               "commit/replay_slim_p0/slim.json": {"pair": 0}, "commit/replay_slim_p0/store/salts/s0.bin": "salt",
               "match/capture/log.jsonl.gz": "planes", **PRIVATE_FILES}
    for rel, doc in {**public, **private}.items():
        _w(d / rel, doc)
    run = tmp_path / "run"
    ro.main(ro.WriteOptions(evidence_dir=str(d), row=row, stages=stages, run_dir=str(run)))
    doc = json.loads((run / "outputs.json").read_text())
    assert doc["stage"] == "build plan match commit replay"
    assert sorted((e["name"], e["kind"], e["meta"]["stage"]) for e in doc["outputs"]) == [
        ("build", "vllm-build/v1", "build"), ("evidence", "execution-evidence/v1", "replay"), ("logs", "run-files/v1", "replay"),
        ("match", "vllm-match-record/v1", "match"), ("plan", "vllm-commit-plan/v1", "plan"), ("verdict", "vllm-verdict/v1", "replay")]
    staged = {p.relative_to(run / "outputs").as_posix().split("/", 1)[1] for p in (run / "outputs").rglob("*") if p.is_file()}
    assert staged == set(public) | {"stages.txt"}
