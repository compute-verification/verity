from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from verity_vllm import commit, relations, synthetic, verify
from verity_vllm.capture import CaptureReader
from verity_vllm.negatives import NEGATIVE_IDS

TOY_PLUGIN = "verity_vllm.synthetic:install"
PKG_DIR = Path(__file__).resolve().parents[1]


def write_production_record(run_dir: Path, path: Path, *, tamper: bool = False) -> Path:
    reader = CaptureReader(run_dir)
    with open(path, "w") as handle:
        for rid in reader.request_ids():
            b = reader.boundary(rid)
            outputs = list(b.output_token_ids)
            if tamper and outputs:
                outputs[-1] = (outputs[-1] + 1) % 32
            handle.write(
                json.dumps(
                    {
                        "request_id": rid,
                        "prompt_token_ids": b.prompt_token_ids,
                        "output_token_ids": outputs,
                        "finish_reason": b.finish_reason,
                        "sampling_params": b.sampling_params,
                    }
                )
                + "\n"
            )
    return path


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    root = tmp_path_factory.mktemp("verify")
    toy = synthetic.build_toy_run(root / "good", seed=3)
    production = write_production_record(toy.run_dir, root / "production.jsonl")
    wrong = root / "wrong"
    shutil.copytree(toy.run_dir, wrong)
    reader = CaptureReader(toy.run_dir)
    target = next(c for c in reader.ru("req-0", 0).ids() if "layers.0.residual" in c)
    old, new = synthetic.corrupt_committed_value(wrong, target)
    assert old != new and len(old) == len(new)
    return {"root": root, "toy": toy, "production": production, "wrong": wrong, "target": target}


def test_wrong_bundle_is_internally_consistent(bundles):
    """The corruption is the N02 shape: authentication passes, only the relation can catch it."""
    reader = CaptureReader(bundles["wrong"])
    assert reader.validate() == []
    commitments = commit.load_commitments(reader)
    assert commit.verify_commitments(reader, commitments) == []


def test_effective_seed_depends_on_roots(bundles):
    good = commit.load_commitments(CaptureReader(bundles["toy"].run_dir))
    wrong = commit.load_commitments(CaptureReader(bundles["wrong"]))
    s1, d1 = verify.effective_seed(7, good)
    s2, d2 = verify.effective_seed(7, wrong)
    s3, _ = verify.effective_seed(8, good)
    assert d1 != d2 and s1 != s2 and s1 != s3
    assert verify.effective_seed(7, good) == (s1, d1)


def test_verify_good_bundle_passes(bundles):
    out = bundles["root"] / "out-good"
    result = verify.verify(
        bundles["toy"].run_dir,
        seed=7,
        sample_size=10,
        out_dir=out,
        case="TOY",
        experiment_id="TOY-good",
        production_record=bundles["production"],
        plugins=[TOY_PLUGIN],
        torch_mode="never",
    )
    assert result.verdict == "PASS", result.table1["first_blocker"]
    assert result.exit_code == 0
    assert result.report.all_exact()
    assert result.report.openings_verified == result.report.openings_attempted > 0
    assert result.adjacent["ok"] and result.structural_problems == []
    assert result.table1["boundary_noninterference"]["identical"] is True
    assert result.table1["boundary_noninterference"]["compared"] > 0
    assert result.table1["token_selection"]["attempted"] > 0
    assert [r.negative_id for r in result.negatives] == list(NEGATIVE_IDS)
    assert all(r.rejected == r.attempted > 0 for r in result.negatives)
    for name in (
        "challenge.json",
        "report.json",
        "table1.jsonl",
        "table2.jsonl",
        "table3.jsonl",
        "summary.txt",
    ):
        assert (out / name).exists(), name
    challenge = json.loads((out / "challenge.json").read_text())
    assert challenge["seed"] == 7 and challenge["roots_digest"] == result.roots_digest
    assert challenge["challenge"]["digest"] == result.challenge.digest().hex()
    assert challenge["adjacent_pairs"]
    report = json.loads((out / "report.json").read_text())
    assert report["verdict"]["result"] == "PASS"
    assert report["checkers"]["plugins"][TOY_PLUGIN]["installed"] is True
    summary = (out / "summary.txt").read_text()
    assert "verdict PASS" in summary and "adjacent VUs: ok" in summary


def test_verify_is_deterministic_in_seed(bundles):
    kw = dict(sample_size=8, case="TOY", plugins=[TOY_PLUGIN], torch_mode="never", negatives=None)
    a = verify.verify(bundles["toy"].run_dir, seed=1, out_dir=bundles["root"] / "d1", **kw)
    b = verify.verify(bundles["toy"].run_dir, seed=1, out_dir=bundles["root"] / "d2", **kw)
    c = verify.verify(bundles["toy"].run_dir, seed=2, out_dir=bundles["root"] / "d3", **kw)
    assert a.sample == b.sample and a.challenge == b.challenge
    assert a.sample != c.sample


def test_verify_wrong_bundle_fails_on_the_relation(bundles):
    result = verify.verify(
        bundles["wrong"],
        seed=7,
        exhaustive=True,
        out_dir=bundles["root"] / "out-wrong",
        case="TOY",
        experiment_id="TOY-wrong",
        production_record=bundles["production"],
        plugins=[TOY_PLUGIN],
        torch_mode="never",
        negatives=None,
    )
    assert result.verdict == "FAIL" and result.exit_code == 1
    assert result.table1["first_blocker"].startswith("relation_checks.independent")
    # authentication is intact (consistent commitment); the relation is what rejects
    auth = result.table1["authentication"]
    assert auth["valid"] == auth["attempted"] > 0
    mismatches = [r for r in result.report.results if r.status == "mismatch"]
    assert bundles["target"] in {r.canonical_id for r in mismatches}
    # the corrupted value is also an input of its consumer; the wiring makes it visible there too
    assert any("layers.1.mlp.gemm" in r.canonical_id for r in mismatches)
    assert all(r.first_divergence is not None for r in mismatches)
    summary = (bundles["root"] / "out-wrong" / "summary.txt").read_text()
    assert "verdict FAIL" in summary and "first divergence" in summary


def test_verify_without_the_toy_relation_is_partial_not_pass(bundles):
    # in-process the toy relation is already registered (synthetic is imported), so hand the
    # verifier a registry without it, as a verifier process that never loaded the plugin has
    registry = {k: v for k, v in relations.REGISTRY.items() if k != "gemm.toy.f32acc.v1"}
    result = verify.verify(
        bundles["toy"].run_dir,
        seed=7,
        sample_size=10,
        out_dir=bundles["root"] / "out-noplugin",
        case="TOY",
        production_record=bundles["production"],
        plugins=[],
        torch_mode="never",
        negatives=None,
        registry=registry,
    )
    assert result.verdict == "PARTIAL"
    assert "unknown_relation" in result.report.statuses()
    t2 = {r["op_id"]: r for r in result.table2}
    gemm_rows = [r for r in result.table2 if r["primitive_semantics"] == "gemm.toy.f32acc.v1"]
    assert gemm_rows and all(
        r["checker"] == "none" and r["status"] == "unknown_relation" for r in gemm_rows
    )
    assert t2["embed_tokens"]["checker"] == "independent"


def test_tampered_production_record_fails_noninterference(bundles):
    tampered = write_production_record(
        bundles["toy"].run_dir, bundles["root"] / "production-tampered.jsonl", tamper=True
    )
    result = verify.verify(
        bundles["toy"].run_dir,
        seed=3,
        sample_size=6,
        out_dir=bundles["root"] / "out-tampered",
        case="TOY",
        production_record=tampered,
        plugins=[TOY_PLUGIN],
        torch_mode="never",
        negatives=None,
    )
    assert result.verdict == "FAIL"
    assert result.table1["first_blocker"].startswith("boundary_noninterference")
    assert result.table1["boundary_noninterference"]["mismatched_scalars"] == 2


def test_manifest_engine_flags_reach_the_relation_evaluator(tmp_path, monkeypatch):
    """W9 §8.1 / W1 B1-p1: the generic verifier must re-execute the relations *under the profile*:
    a manifest that declares ``engine_flags`` (``VLLM_BATCH_INVARIANT=1`` on the real bundles) has
    them set in the checker's process before the first relation runs -- as the adapter verifier
    does -- else a same-runtime kernel picks another variant (the 1-ulp rmsnorm row).  Toy bundle,
    numpy only: the flag is a test-only variable and the toy GEMM relation is wrapped to record what
    it sees in the environment when it is called."""
    flag = "VERITY_TEST_ENGINE_FLAG"
    monkeypatch.delenv(flag, raising=False)
    monkeypatch.setitem(synthetic.TOY_MANIFEST_BASE, "engine_flags", {flag: "on"})
    toy = synthetic.build_toy_run(tmp_path / "flagged", seed=3)
    assert json.loads((toy.run_dir / "manifest.json").read_text())["engine_flags"] == {flag: "on"}
    seen: list[str | None] = []
    toy_rel = relations.REGISTRY["gemm.toy.f32acc.v1"]

    def recording_checker(inst, inputs):
        seen.append(os.environ.get(flag))
        return toy_rel.checker(inst, inputs)

    registry = dict(relations.REGISTRY)
    registry["gemm.toy.f32acc.v1"] = relations.Relation(
        toy_rel.relation_id, toy_rel.klass, toy_rel.op_type, recording_checker,
        toy_rel.includes_full_op, toy_rel.version, toy_rel.doc, toy_rel.implemented, toy_rel.owner)
    production = write_production_record(toy.run_dir, tmp_path / "production.jsonl")
    result = verify.verify(
        toy.run_dir, seed=7, sample_size=10, out_dir=tmp_path / "out", case="TOY", plugins=[],
        torch_mode="never", negatives=None, registry=registry, production_record=production,
    )
    assert result.report.all_exact(), result.table1["first_blocker"]
    assert result.verdict in ("PASS", "PARTIAL"), result.table1["first_blocker"]
    # every relation evaluation saw the declared flag
    assert seen and all(v == "on" for v in seen), seen
    assert result.checkers["engine_flags"] == {"applied": {flag: "on"}, "conflicts": {}}
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["checkers"]["engine_flags"]["applied"] == {flag: "on"}
    # a conflicting environment: refused when runtime relations may run, recorded (not fatal)
    # for the numpy-only path
    monkeypatch.setenv(flag, "off")
    with pytest.raises(verify.BundleError, match="engine_flags"):
        verify.apply_engine_flags({"engine_flags": {flag: "on"}}, strict=True)
    assert verify.apply_engine_flags({"engine_flags": {flag: "on"}}, strict=False) == {
        "applied": {}, "conflicts": {flag: {"environment": "off", "declared": "on"}}}
    assert os.environ[flag] == "off"  # the environment is never overwritten on a conflict


def test_results_root_layout_renders(bundles, tmp_path):
    results_root = tmp_path / "results"
    result = verify.verify(
        bundles["toy"].run_dir,
        seed=5,
        sample_size=6,
        out_dir=tmp_path / "out",
        case="TOY",
        experiment_id="TOY-43524e8aa2cb-20260907T0700Z",
        production_record=bundles["production"],
        plugins=[TOY_PLUGIN],
        torch_mode="never",
        negatives=["N01", "N02"],
        results_root=results_root,
    )
    exp = results_root / result.experiment_id
    for name in (
        "table1.jsonl",
        "table2.jsonl",
        "table3.jsonl",
        "manifest.json",
        "commitments.json",
        "description.json",
        "log.txt",
    ):
        assert (exp / name).exists(), name
    from verity_vllm import render

    _, collected = render.render(results_root, tmp_path / "results.md")
    assert collected.issues == []
    assert len(collected.rows(3)) == 2


def test_load_bundle_rejects_non_bundles(bundles, tmp_path):
    with pytest.raises(verify.BundleError):
        verify.load_bundle(tmp_path / "nowhere")
    capture_only = tmp_path / "capture"
    shutil.copytree(bundles["toy"].run_dir, capture_only)
    (capture_only / "commitments.json").unlink()
    with pytest.raises(verify.BundleError, match="commitments.json"):
        verify.load_bundle(capture_only)


def test_adjacent_pair_helpers(bundles):
    toy = bundles["toy"]
    reader = toy.reader()
    all_ids = list(toy.description.expected_ids(reader.boundary("req-0"), reader.profile_id))
    embeds = [c for c in all_ids if "embed_tokens" in c]
    # a sample of producers only has no pair; the helper adds a consumer of one of them
    assert relations.adjacent_pairs(toy.description, reader, embeds) == []
    extended, pairs = relations.ensure_adjacent_pair(toy.description, reader, embeds[:2], seed=0)
    assert pairs and len(extended) == 3
    assert pairs[0].producer in embeds and pairs[0].consumer == extended[-1]
    assert relations.adjacent_pairs(toy.description, reader, extended) == pairs
    # a consumer alone: its committed producer is added
    gemm = [c for c in all_ids if "layers.0.mlp.gemm" in c][:1]
    extended, pairs = relations.ensure_adjacent_pair(toy.description, reader, gemm, seed=0)
    assert len(extended) == 2 and pairs[0].consumer == gemm[0] and pairs[0].arg == "x"
    # already has a pair: unchanged
    again, pairs2 = relations.ensure_adjacent_pair(toy.description, reader, extended, seed=9)
    assert again == extended and pairs2 == pairs
    report = relations.check(toy.description, reader, toy.commitments, extended)
    assert relations.adjacent_check(pairs, report)["ok"]
    # if the consumer does not authenticate/agree the invariant fails
    bad = relations.adjacent_check(
        pairs, relations.check(toy.description, reader, toy.commitments, gemm)
    )
    assert not bad["ok"] and bad["pairs"][0]["producer_status"] == "absent"


def test_cli(bundles, tmp_path):
    cmd = [
        sys.executable, "-m", "verity_vllm.verify", str(bundles["toy"].run_dir),
        "--seed", "7", "--sample", "8", "--case", "TOY", "--out", str(tmp_path / "cli"),
        "--plugin", TOY_PLUGIN, "--production-record", str(bundles["production"]),
        "--torch", "never", "--negatives", "N01,N02,N07",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=PKG_DIR)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "verdict PASS" in proc.stdout
    # a small sample may or may not include the corrupted VU; exhaustive must find it
    wrong = [x for x in cmd if x not in ("--sample", "8")] + ["--exhaustive"]
    wrong[3] = str(bundles["wrong"])
    wrong[wrong.index("--out") + 1] = str(tmp_path / "cli-wrong")
    proc = subprocess.run(wrong, capture_output=True, text=True, cwd=PKG_DIR)
    assert proc.returncode == 1, proc.stdout
    assert "verdict FAIL" in proc.stdout and "relation_checks" in proc.stdout
    proc = subprocess.run(
        [sys.executable, "-m", "verity_vllm.verify", str(tmp_path / "nothing"), "--seed", "1"],
        capture_output=True,
        text=True,
        cwd=PKG_DIR,
    )
    assert proc.returncode == 2 and "cannot verify" in proc.stderr
