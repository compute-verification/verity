"""`manifest.recorded_program_digest` reads a Build recorded before the SHA-512 re-baseline (ad58900df) as that commit's code did:
its `program.sha512`, never its SHA-256 `program.digest`, which `verity/partition/v1` refuses (the R19 reference regression's
rebuilds of the frozen v1 records)."""

from __future__ import annotations

import gzip
import json

from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.codec import program_digest
from verity_vllm.pipeline import manifest as M
from verity_vllm.query.required import partition_of

SHA256, SHA512 = "a" * 64, "b" * 128


def test_a_record_before_the_re_baseline_is_named_by_its_sha512(tmp_path):
    assert M.recorded_program_digest(str(tmp_path), {"program": {"digest": SHA256, "sha512": SHA512}}) == SHA512
    assert PO.validate(partition_of(SHA512)["object"]).ok


def test_a_sha256_digest_alone_falls_back_to_the_descriptor(tmp_path):
    assert M.recorded_program_digest(str(tmp_path), {"program": {"digest": SHA256}}) is None
    desc = {"schema": "verity-ir/descriptor/v1", "definitions": [], "root": None}
    with gzip.open(tmp_path / "descriptor.json.gz", "wt") as f:
        json.dump(desc, f)
    assert M.recorded_program_digest(str(tmp_path), {"program": {"digest": SHA256}}) == program_digest(desc)


def test_a_record_after_the_re_baseline_is_named_by_its_digest(tmp_path):
    assert M.recorded_program_digest(str(tmp_path), {"program": {"digest": SHA512}}) == SHA512
