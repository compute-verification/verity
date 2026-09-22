from __future__ import annotations

import json

import numpy as np
import pytest
from _helpers import TINY_MANIFEST, make_tiny_run

from verity_vllm import capture, hashing
from verity_vllm.capture import (
    Boundary,
    CaptureError,
    CaptureReader,
    CaptureWriter,
    RunWriter,
    StaleCaptureError,
    StepRecord,
)
from verity_vllm.identity import value_id


def test_round_trip(tiny_run):
    reader = CaptureReader(tiny_run)
    assert reader.validate() == []
    assert reader.profile_id == hashing.profile_id(TINY_MANIFEST)
    assert hashing.sha256_hex(reader.manifest_bytes())[:12] == reader.profile_id
    assert reader.request_ids() == ["req-0"]
    assert reader.ru_keys() == [("req-0", 0), ("req-0", 1)]
    ru = reader.ru("req-0", 0)
    assert len(ru.index.entries) == 7
    emb1 = str(value_id(reader.profile_id, "req-0", 1, "prefill", "embed", "embed_tokens", 0))
    assert ru.array(emb1).tolist() == [0x3F81] * 4
    assert reader.value(emb1).float32().tolist() == pytest.approx([1.0078125] * 4)
    assert reader.locate(emb1) == ("req-0", 0)
    assert reader.tree_of(emb1) == "ru/req-0/000000"
    assert reader.tree_of("w=lm_head.weight") == "weights"
    assert reader.weights.value("w=lm_head.weight").array()[0, 0] == 1
    # index order == file order, offsets cumulative
    entries = ru.index.entries
    assert [e.offset for e in entries] == [0, 8, 16, 24, 32, 40, 48]
    assert entries[-1].nbytes == 4
    # entries carry the leaf hash the writer computed
    for entry in entries:
        assert ru.value(entry.canonical_id).leaf().hex() == entry.leaf


def test_writer_validates_ids(tmp_path):
    run = RunWriter(tmp_path, TINY_MANIFEST)
    ru = run.ru("req-0", 0, phase="prefill", positions=[0, 1])
    pid = run.profile_id
    good = str(value_id(pid, "req-0", 0, "prefill", "embed", "embed_tokens", 0))
    ru.add(good, "bfloat16", [2], b"\x00\x00\x00\x00")
    with pytest.raises(CaptureError):  # wrong request
        ru.add(
            str(value_id(pid, "req-1", 0, "prefill", "embed", "e", 0)), "bfloat16", [2], bytes(4)
        )
    with pytest.raises(CaptureError):  # wrong phase
        ru.add(str(value_id(pid, "req-0", 0, "decode", "embed", "e", 0)), "bfloat16", [2], bytes(4))
    with pytest.raises(CaptureError):  # position outside the unit
        ru.add(
            str(value_id(pid, "req-0", 5, "prefill", "embed", "e", 0)), "bfloat16", [2], bytes(4)
        )
    with pytest.raises(CaptureError):  # wrong profile
        ru.add(
            str(value_id("0" * 12, "req-0", 0, "prefill", "embed", "e", 0)),
            "bfloat16",
            [2],
            bytes(4),
        )
    with pytest.raises(capture.dtypes.DtypeError):  # nbytes mismatch
        ru.add(
            str(value_id(pid, "req-0", 1, "prefill", "embed", "e", 0)), "bfloat16", [2], bytes(3)
        )
    with pytest.raises(CaptureError):  # weight id in a replay unit
        ru.add("w=x", "bfloat16", [2], bytes(4))
    with pytest.raises(CaptureError):
        ru.add(
            str(value_id(pid, "req-0", 1, "prefill", "embed", "e", 0)),
            "bfloat16",
            [2],
            bytes(4),
            origin="guessed",
        )
    ru.finish()
    with pytest.raises(CaptureError):
        ru.add(good, "bfloat16", [2], bytes(4))
    with pytest.raises(CaptureError):  # a finished unit cannot be reopened
        CaptureWriter(tmp_path, "req-0", 0, phase="prefill", positions=[0])


def test_duplicate_capture_is_refused(tmp_path):
    run = RunWriter(tmp_path, TINY_MANIFEST)
    pid = run.profile_id
    cid = str(value_id(pid, "req-0", 0, "prefill", "embed", "embed_tokens", 0))
    ru = run.ru("req-0", 0, phase="prefill", positions=[0])
    ru.add(cid, "bfloat16", [2], bytes(4))
    with pytest.raises(StaleCaptureError):
        ru.add(cid, "bfloat16", [2], b"\x01\x00\x00\x00")
    with pytest.raises(StaleCaptureError):  # identical bytes are refused too in strict mode
        ru.add(cid, "bfloat16", [2], bytes(4))
    ru.finish()


def test_duplicate_capture_flag_mode_taints_the_unit(tmp_path):
    run = RunWriter(tmp_path, TINY_MANIFEST)
    pid = run.profile_id
    cid = str(value_id(pid, "req-0", 0, "prefill", "embed", "embed_tokens", 0))
    ru = run.ru("req-0", 0, phase="prefill", positions=[0], on_duplicate="flag")
    first = ru.add(cid, "bfloat16", [2], bytes(4))
    again = ru.add(cid, "bfloat16", [2], b"\x01\x00\x00\x00")
    assert again == first  # the first snapshot is kept
    index = ru.finish()
    assert index.tainted and len(index.conflicts) == 1 and not index.conflicts[0]["identical"]
    reader = CaptureReader(tmp_path)
    assert reader.ru("req-0", 0).read(cid) == bytes(4)
    assert any("tainted" in p for p in reader.ru("req-0", 0).validate())
    # identical duplicates are flagged but do not taint
    ru = run.ru("req-0", 1, phase="decode", positions=[1], on_duplicate="flag")
    cid1 = str(value_id(pid, "req-0", 1, "decode", "embed", "embed_tokens", 0))
    ru.add(cid1, "bfloat16", [2], bytes(4))
    ru.add(cid1, "bfloat16", [2], bytes(4))
    index = ru.finish()
    assert not index.tainted and index.conflicts[0]["identical"]


def test_reader_detects_tampered_index_and_bytes(tiny_run):
    reader = CaptureReader(tiny_run)
    path = tiny_run / "ru" / "req-0" / "000000" / "index.json"
    doc = json.loads(path.read_text())
    doc["entries"][0]["shape"] = [2]  # nbytes no longer matches
    path.write_text(json.dumps(doc))
    problems = CaptureReader(tiny_run).ru("req-0", 0).validate()
    assert any("needs 4 bytes" in p for p in problems)
    doc["entries"][0]["shape"] = [4]
    doc["entries"][0]["dtype"] = "float16"  # same nbytes, different dtype -> leaf mismatch
    path.write_text(json.dumps(doc))
    problems = CaptureReader(tiny_run).ru("req-0", 0).validate()
    assert any("stored leaf" in p for p in problems)
    # truncated values.bin
    values = tiny_run / "ru" / "req-0" / "000001" / "values.bin"
    values.write_bytes(values.read_bytes()[:-1])
    problems = CaptureReader(tiny_run).ru("req-0", 1).validate()
    assert any("values.bin is" in p for p in problems)
    with pytest.raises(CaptureError):
        CaptureReader(tiny_run).ru("req-0", 1).read(
            str(value_id(reader.profile_id, "req-0", 3, "decode", "sampler", "sampler.select", 0))
        )


def test_bare_list_index_is_accepted(tiny_run):
    path = tiny_run / "ru" / "req-0" / "000001" / "index.json"
    doc = json.loads(path.read_text())
    path.write_text(json.dumps(doc["entries"]))
    ru = CaptureReader(tiny_run).ru("req-0", 1)
    assert ru.index.phase == "decode" and ru.index.positions == (3,)
    assert ru.validate() == []


def test_boundary_build_and_queries():
    b = Boundary.build(
        "req-0",
        [1, 2, 3, 4],
        [17, 5, 9],
        {"temperature": 0.0, "max_tokens": 4},
        seed=None,
        finish_reason="stop",
        steps=[
            StepRecord(0, "prefill", (0, 1, 2, 3)),
            StepRecord(1, "decode", (4,)),
            StepRecord(2, "decode", (5,)),
        ],
    )
    assert b.validate() == []
    assert [r.t for r in b.output_records] == [3, 4, 5, 6]
    assert [r.present for r in b.output_records] == [True, True, True, False]
    assert b.output_records[0].step == 0 and b.output_records[3].step is None
    assert (
        b.token_at(2) == 3 and b.token_at(4) == 17 and b.token_at(6) == 9 and b.token_at(7) is None
    )
    assert (
        b.sampled_token_at(3) == 17 and b.sampled_token_at(5) == 9 and b.sampled_token_at(6) is None
    )
    assert b.sampled_positions() == [3, 4, 5]
    assert Boundary.from_json(b.to_json()) == b
    leaves = b.leaves("0123456789ab")
    ids = [v.canonical_id for v in leaves]
    assert len(leaves) == 4 + 4 + 3 + 3
    assert "p=0123456789ab/r=req-0/b=output_presence/t=6" in ids
    assert "p=0123456789ab/r=req-0/b=output_token/t=6" not in ids
    assert next(v for v in leaves if v.canonical_id.endswith("b=seed")).json() is None
    assert next(v for v in leaves if v.canonical_id.endswith("b=finish_reason")).text() == "stop"


def test_boundary_validate_catches_broken_records():
    b = Boundary.build(
        "req-0",
        [1, 2],
        [5, 6],
        {"max_tokens": 3},
        seed=1,
        finish_reason=None,
        steps=[StepRecord(0, "prefill", (0, 1)), StepRecord(1, "decode", (2,))],
    )
    assert b.validate() == []
    dropped = Boundary.from_json(b.to_json())
    dropped.output_records.pop(1)
    assert dropped.validate()
    permuted = Boundary.from_json(b.to_json())
    permuted.output_records[0], permuted.output_records[1] = (
        permuted.output_records[1],
        permuted.output_records[0],
    )
    assert permuted.validate()
    length = Boundary.build(
        "req-0",
        [1, 2],
        [5],
        {"max_tokens": 3},
        seed=None,
        finish_reason="length",
        steps=[StepRecord(0, "prefill", (0, 1))],
    )
    assert any("length" in p for p in length.validate())
    bad_steps = Boundary.build(
        "req-0",
        [1, 2],
        [5, 6],
        {"max_tokens": 2},
        seed=None,
        finish_reason="length",
        steps=[StepRecord(0, "prefill", (0, 1)), StepRecord(1, "decode", (2, 3))],
    )
    assert any("decode step" in p for p in bad_steps.validate())


def test_run_writer_refuses_invalid_boundary(tmp_path):
    run = RunWriter(tmp_path, TINY_MANIFEST)
    b = Boundary.build(
        "req-0", [1], [2, 3], {"max_tokens": 1}, seed=None, finish_reason="stop", steps=[]
    )
    with pytest.raises(CaptureError):
        run.write_boundary(b)


def test_weights_without_stored_values(tmp_path):
    run = RunWriter(tmp_path, TINY_MANIFEST)
    w = run.weights(store_values=False, source_sha256={"model.safetensors": "ab" * 32})
    raw = np.arange(8, dtype=np.uint16).tobytes()
    entry = w.add(
        "model.embed_tokens.weight", "bfloat16", [4, 2], raw, source={"file": "model.safetensors"}
    )
    assert entry.offset is None and entry.sha256 == hashing.sha256_hex(raw)
    w.add("model.norm.weight", "bfloat16", [2], sha256="00" * 32, leaf="11" * 32)
    with pytest.raises(CaptureError):
        w.add("model.x", "bfloat16", [2])  # neither bytes nor sha/leaf
    w.finish()
    reader = CaptureReader(tmp_path)
    assert reader.weights.validate() == []
    with pytest.raises(CaptureError):
        reader.weights.read("w=model.embed_tokens.weight")
    reader.weights.resolver = lambda e: raw
    assert reader.weights.read("w=model.embed_tokens.weight") == raw
    reader.weights.resolver = lambda e: raw[::-1]
    with pytest.raises(CaptureError):
        reader.weights.read("w=model.embed_tokens.weight")


def test_weights_per_entry_store_flag(tmp_path):
    """W1c (§16.8): a storing writer may hash-only individual entries (checkpoint tensors referenced by
    source); the reader serves stored entries from values.bin and referenced ones via the resolver."""
    run = RunWriter(tmp_path, TINY_MANIFEST)
    w = run.weights(store_values=True, source_sha256={"model.safetensors": "ab" * 32})
    stored = np.arange(8, dtype=np.uint16).tobytes()
    referenced = np.arange(8, 16, dtype=np.uint16).tobytes()
    e1 = w.add("layers.0.self_attn.rotary_emb.cos_sin_cache", "bfloat16", [4, 2], stored)
    e2 = w.add("model.embed_tokens.weight", "bfloat16", [4, 2], referenced, store=False,
               source={"kind": "checkpoint", "file": "model.safetensors", "tensors": ["model.embed_tokens.weight"], "cat_dim": 0})
    assert e1.offset == 0 and e2.offset is None and e2.sha256 == hashing.sha256_hex(referenced)
    w.finish()
    reader = CaptureReader(tmp_path)
    assert reader.weights.validate() == []
    assert reader.weights.read("w=layers.0.self_attn.rotary_emb.cos_sin_cache") == stored
    with pytest.raises(CaptureError):
        reader.weights.read("w=model.embed_tokens.weight")
    reader.weights.resolver = lambda e: referenced if e.source["tensors"] == ["model.embed_tokens.weight"] else b""
    assert reader.weights.read("w=model.embed_tokens.weight") == referenced
    reader.weights.resolver = lambda e: stored
    with pytest.raises(CaptureError):
        reader.weights.read("w=model.embed_tokens.weight")
    hash_only = RunWriter(tmp_path / "h", TINY_MANIFEST).weights(store_values=False)
    with pytest.raises(CaptureError):
        hash_only.add("model.norm.weight", "bfloat16", [2], np.zeros(2, dtype=np.uint16).tobytes(), store=True)


def test_manifest_schema_required(tmp_path):
    with pytest.raises(CaptureError):
        RunWriter(tmp_path, {"model": {}})
    assert make_tiny_run(tmp_path / "again").profile_id == hashing.profile_id(TINY_MANIFEST)
