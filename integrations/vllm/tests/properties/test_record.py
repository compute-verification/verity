"""The one property record: sealing, the digest a run cites, the registry of properties, and the records a run carries."""
from __future__ import annotations

import json

import pytest

from verity_vllm.properties import fa_tap_exactness, golden, holdout, quarantine_lint
from verity_vllm.properties.record import BY_SCHEMA, REGISTRY, PropertyRecord, digest_of, records_of, seal

NI = REGISTRY["noninterference"].schema


def test_seal_binds_ok_and_every_field_to_the_digest():
    doc = seal({"schema": NI, "world": 1, "tokens": {"equal": True}}, True)
    rec = PropertyRecord(doc)
    assert rec.name == "noninterference" and rec.ok and rec.digest == digest_of(doc) and rec.problems() == []
    for k, v in (("ok", False), ("world", 2)):
        assert PropertyRecord({**doc, k: v}).problems() == ["digest does not match the record"]
    assert seal(doc, False)["digest"] != doc["digest"]
    with pytest.raises(ValueError):
        seal({"schema": "not-a-property/v1"}, True)
    assert PropertyRecord({"schema": "not-a-property/v1", "ok": True, "digest": "x"}).name is None


def test_the_registry_names_each_harness_schema_once():
    from tests.properties import (
        dsv4_indexer_capture_gpu,
        dsv4_moe_capture_gpu,
        fa2_softcap_capture_gpu,
        fa2_target_capture_gpu,
        fp8_sm120_capture_gpu,
        glm53_bf16_gemm_capture_gpu,
        glm53_block_fp8_capture_gpu,
        glm53_dsa_capture_gpu,
        glm53_tp8_allreduce_capture_gpu,
        sigmoid_router_capture_gpu,
    )

    assert len(BY_SCHEMA) == len(REGISTRY) and all(p.claim for p in REGISTRY.values())
    assert REGISTRY["fa_tap_exactness"].schema == fa_tap_exactness.SCHEMA
    assert REGISTRY["fa2_target_capture"].schema == fa2_target_capture_gpu.SCHEMA
    assert REGISTRY["fa2_softcap_capture"].schema == fa2_softcap_capture_gpu.SCHEMA
    assert REGISTRY["glm53_block_fp8_capture"].schema == glm53_block_fp8_capture_gpu.SCHEMA
    assert REGISTRY["glm53_bf16_gemm_capture"].schema == glm53_bf16_gemm_capture_gpu.SCHEMA
    assert REGISTRY["glm53_dsa_capture"].schema == glm53_dsa_capture_gpu.SCHEMA
    assert REGISTRY["dsv4_indexer_capture"].schema == dsv4_indexer_capture_gpu.SCHEMA
    assert REGISTRY["glm53_tp8_allreduce_capture"].schema == glm53_tp8_allreduce_capture_gpu.SCHEMA
    assert REGISTRY["blackwell_fp8_capture"].schema == fp8_sm120_capture_gpu.SCHEMA
    assert REGISTRY["sigmoid_router_capture"].schema == sigmoid_router_capture_gpu.SCHEMA
    assert REGISTRY["dsv4_moe_capture"].schema == dsv4_moe_capture_gpu.SCHEMA


def test_harnesses_return_sealed_records(tmp_path):
    (tmp_path / "holdouts").mkdir()
    (tmp_path / "quarantine").mkdir()
    for doc in (golden.check(tmp_path / "corpus.json"), holdout.assemble(tmp_path / "holdouts"), quarantine_lint.lint(tmp_path / "quarantine")):
        rec = PropertyRecord(doc)
        assert rec.name in REGISTRY and rec.problems() == [], (doc["schema"], rec.problems())
    assert PropertyRecord(golden.check(tmp_path / "corpus.json")).ok is False            # nothing checked is not a pass


def test_records_of_a_run_and_their_citations(tmp_path):
    good = PropertyRecord(seal({"schema": NI, "world": 2}, True))
    assert good.write(tmp_path) == tmp_path / "properties" / "noninterference.json"
    bad = seal({"schema": REGISTRY["quarantine_lint"].schema, "n_files": 0}, False)
    (tmp_path / "properties" / "quarantine_lint.json").write_text(json.dumps({**bad, "n_files": 3}))
    got = records_of(tmp_path)
    assert [r.source for r in got] == ["properties/noninterference.json", "properties/quarantine_lint.json"]
    cites = [r.citation() for r in got]
    assert cites[0] == {"name": "noninterference", "schema": NI, "digest": good.digest, "ok": True,
                        "source": "properties/noninterference.json", "problems": []}
    assert cites[1]["problems"] == ["digest does not match the record"] and cites[1]["ok"] is False
    assert records_of(tmp_path / "nothing") == []
