"""The transfer test's audit probe on the vLLM adapter's S3 claim: `GreedyServe(B0, steps=15)` derives, the family rule
claims `Serve_v1`, the digest is the certificate of record's, and the KV-ownership oracle holds on the derived program.
The transfer cases themselves, a second model family through the frozen importer, are the PyTorch adapter's
`tests/frontend/test_derive_transfer.py`."""

from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")

IMPORTER_NAME = "verity_torch.frontend.derive.derive"

OUT_PATH = os.environ.get("XFER_OUT") or "/tmp/xfer/transfer-derive.jsonl"
def _git_head() -> str:
    if os.environ.get("XFER_HEAD"):
        return os.environ["XFER_HEAD"]
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


GIT_HEAD = _git_head()


def _record(rec: dict) -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


# ---------------------------------------------------------------------------------------------------------
# support-matrix audit probes (heavy: ~8 s / 0.5 GB; run with XFER_HEAVY=1, on the pod)
# ---------------------------------------------------------------------------------------------------------

CERTIFIED_B0_DIGEST = "d4f4b1f1b4d592d697eef5b6c9c803108bdf02d0ef24c945eace6e705832e1b8"  # INTPAD2-B0-20260909T013519Z/certificate.json


@pytest.mark.skipif(not os.environ.get("XFER_HEAVY"), reason="heavy audit probe; set XFER_HEAVY=1")
def test_audit_s3_derived_b0_serve_digest_and_kv_ownership():
    """Independent reproduction of the importer's S3 claim: `GreedyServe(B0, steps=15)` on a 32-token prompt derives, the
    family rule claims `Serve_v1{B0,LP=32,STEPS=15}`, the digest equals the certificate of record's, and this file's
    name-independent KV-ownership oracle holds on the derived program (30 layers x 47 positions)."""
    from verity_vllm.definitions.registry.b1 import B0_CONFIG

    from tests.program.test_derive_negative import (
        check_cache_ownership,
        check_cache_ownership_by_kind,
    )
    from verity_vllm.program.frontend import derive_examples as dex
    from verity_torch.frontend.derive import derive_report
    rep = derive_report(dex.GreedyServe(B0_CONFIG, 15), (torch.zeros(32, dtype=torch.int32),), mode="profile", profile="b1-eager",
                        config=B0_CONFIG)
    assert rep.ok, rep.unsupported
    fam = rep.family or {}
    rec = {"case": "audit_s3_b0_serve", "root": rep.root[:40], "digest": rep.digest, "certified": rep.digest == CERTIFIED_B0_DIGEST,
           "family_matched": bool(fam.get("matched")), "gates": rep.gates, "flat_digest": rep.flat_digest,
           "head": GIT_HEAD, "importer": IMPORTER_NAME}
    rec["ownership_by_name"] = check_cache_ownership(rep.program)[:5]
    rec["ownership_by_kind"] = check_cache_ownership_by_kind(rep.program, B0_CONFIG["NL"], 32, 15)[:5]
    _record(rec)
    assert fam.get("matched"), fam
    assert rep.digest == CERTIFIED_B0_DIGEST, rec
    assert rec["ownership_by_name"] == [] and rec["ownership_by_kind"] == [], rec
