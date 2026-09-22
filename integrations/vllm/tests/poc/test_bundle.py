"""verity_vllm.bundle: pack / unpack / verify-from-tarball on a synthetic bundle (contract 16.8)."""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import pytest

from verity_vllm import bundle, synthetic, verify
from verity_vllm.capture import CaptureReader

TOY_PLUGIN = "verity_vllm.synthetic:install"
KW = dict(plugins=[TOY_PLUGIN], torch_mode="never", negatives=None, case="TOY")


def _production_record(run_dir: Path, path: Path) -> Path:
    reader = CaptureReader(run_dir)
    with open(path, "w") as handle:
        for rid in reader.request_ids():
            b = reader.boundary(rid)
            handle.write(json.dumps({
                "request_id": rid, "prompt_token_ids": b.prompt_token_ids,
                "output_token_ids": b.output_token_ids, "finish_reason": b.finish_reason,
                "sampling_params": b.sampling_params,
            }) + "\n")
    return path


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    root = tmp_path_factory.mktemp("bundle")
    toy = synthetic.build_toy_run(root / "run", seed=9)
    production = _production_record(toy.run_dir, root / "production.jsonl")
    verify.verify(toy.run_dir, seed=5, sample_size=8, out_dir=root / "run" / "verify",
                  production_record=production, **KW)
    tar = bundle.pack(toy.run_dir, root / "TOY-toy.tar.gz", production_record=production,
                      verify_dir=root / "run" / "verify")
    return {"root": root, "toy": toy, "production": production, "tar": tar}


def test_pack_layout_and_manifest(packed):
    doc = bundle.read_manifest(packed["tar"])
    assert doc["schema"] == bundle.BUNDLE_SCHEMA and doc["name"] == "TOY-toy"
    assert doc["profile_id"] == packed["toy"].profile_id
    members = set(doc["members"])
    for core in ("bundle/manifest.json", "bundle/description.json", "bundle/required.json",
                 "bundle/commitments.json", "bundle/weights/index.json",
                 "production/production.jsonl", "verify/challenge.json", "verify/report.json",
                 "verify/sampling_plan.json", "verify/table1.jsonl"):
        assert core in members, core
    assert "bundle/weights/values.bin" not in members  # referenced by sha, not copied
    assert doc["weights"]["included"] is False and doc["weights"]["index"] is True
    assert any(m.startswith("bundle/ru/") and m.endswith("/values.bin") for m in members)
    assert doc["digests"]["production_record_sha256"]
    rec = doc["recorded_verification"]
    assert rec["seed"] == 5 and rec["effective_seed"] is not None and rec["verdict"] == "PARTIAL"
    assert doc["roots"]["ru"] > 0 and doc["roots"]["boundary"] == 2 and doc["roots"]["weights"]
    with tarfile.open(packed["tar"]) as tar:
        names = tar.getnames()
    assert names[0] == "TOY-toy/BUNDLE.json" and all(n.startswith("TOY-toy/") for n in names)
    assert "reverify" in doc and "python -m verity_vllm.bundle verify" in doc["reverify"]["command"]


def test_unpack_verifies_every_member(packed):
    up = bundle.unpack(packed["tar"], packed["root"] / "unpacked")
    assert up.bundle_dir.is_dir() and up.production_record is not None and up.verify_dir is not None
    assert (up.bundle_dir / "commitments.json").exists()
    # tamper with one member -> refused
    bad = packed["root"] / "bad.tar.gz"
    shutil.copyfile(packed["tar"], bad)
    with tarfile.open(bad) as tar:
        members = tar.getmembers()
        data = {m.name: (tar.extractfile(m).read() if m.isfile() else None) for m in members}
    victim = next(n for n in data if n.endswith("/production/production.jsonl"))
    data[victim] = data[victim].replace(b'"finish_reason"', b'"finish_reasoN"')
    with tarfile.open(bad, "w:gz") as tar:
        import io

        for m in members:
            if m.isfile():
                m.size = len(data[m.name])
                tar.addfile(m, io.BytesIO(data[m.name]))
            else:
                tar.addfile(m)
    with pytest.raises(bundle.BundleError, match="sha256"):
        bundle.unpack(bad, packed["root"] / "unpacked-bad")


def test_verify_from_tarball_reproduces_the_recorded_challenge(packed):
    doc = bundle.verify_tarball(packed["tar"], out_dir=packed["root"] / "reverify", sample_size=8, **KW)
    assert doc["challenge_reproduced"] is True, doc
    assert doc["fresh"]["verdict"] == "PARTIAL" and doc["verdict_agrees"] is True
    assert doc["fresh"]["challenge_digest"] == doc["recorded"]["challenge_digest"]
    assert (packed["root"] / "reverify" / "reverify.json").exists()
    # weight-input VUs are `missing` (weights referenced, not copied): PARTIAL, never PASS or FAIL
    report = json.loads((packed["root"] / "reverify" / "verify" / "report.json").read_text())
    statuses = report["check"]["statuses"]
    assert "missing" in statuses and "mismatch" not in statuses and "auth_failed" not in statuses
    assert report["adjacent_vus"].get("reason", "").startswith("no pair executable") or report["adjacent_vus"]["ok"]


def test_verify_from_tarball_with_weight_values_passes(packed):
    doc = bundle.verify_tarball(
        packed["tar"], out_dir=packed["root"] / "reverify-w", sample_size=8,
        weights_values=packed["toy"].run_dir / "weights" / "values.bin", **KW,
    )
    assert doc["fresh"]["verdict"] == "PARTIAL"  # negatives=None keeps it PARTIAL
    report = json.loads((packed["root"] / "reverify-w" / "verify" / "report.json").read_text())
    assert set(report["check"]["statuses"]) == {"exact"}
    assert report["adjacent_vus"]["ok"] is True


def test_verify_from_tarball_with_a_new_seed_is_not_comparable(packed):
    doc = bundle.verify_tarball(packed["tar"], seed=6, out_dir=packed["root"] / "reverify-6", sample_size=8, **KW)
    assert doc["challenge_reproduced"] is None and doc["challenge_comparison"].startswith("not comparable")
    assert doc["fresh"]["challenge_digest"] != doc["recorded"]["challenge_digest"]


def test_cli_inspect_and_pack_copy_weights(packed, capsys):
    assert bundle.main(["inspect", str(packed["tar"])]) == 0
    out = capsys.readouterr().out
    assert "TOY-toy" in out and "reverify" in out
    tar2 = packed["root"] / "with-weights.tar"
    bundle.pack(packed["toy"].run_dir, tar2, production_record=packed["production"], weights="copy")
    doc = bundle.read_manifest(tar2)
    assert "bundle/weights/values.bin" in doc["members"] and doc["weights"]["included"] is True
