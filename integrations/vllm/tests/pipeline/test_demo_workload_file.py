"""The vLLM demo (`examples/vllm/`). Its workload (`workloads/`) is a fixture as the integration's are (`test_workload_file`): its
directory's SHA256SUMS lists it and fixtures/artifacts.json registers it. Its Commit (`commit.sh`) publishes its stages' records
through `research-outputs write` and nothing else."""
from __future__ import annotations

import json
import re

from tests.pipeline.test_research_outputs_tp import PRIVATE_FILES, _w
from tests.pipeline.test_workload_file import registered_and_listed
from verity_vllm import config as C
from verity_vllm.pipeline import research_outputs as ro

DEMO = C.REPO / "examples" / "vllm" / "workloads"
COMMIT_SH = C.REPO / "examples" / "vllm" / "commit.sh"


def test_the_demos_one_workload_is_a_fixture_its_directory_lists_and_the_registry_holds():
    registered, listed = registered_and_listed(DEMO)
    assert registered == listed, (f"in fixtures/artifacts.json, not in {DEMO / C.DIGESTS}: {sorted(registered - listed)}; "
                                  f"in it, not registered: {sorted(listed - registered)}")
    assert len(C.stored_workloads(DEMO)) == 1


def test_the_example_commit_publishes_its_stages_records_and_none_of_its_private_files(tmp_path):
    """examples/vllm/commit.sh publishes what `research-outputs write` declares for the stages it names, and copies nothing into its
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
