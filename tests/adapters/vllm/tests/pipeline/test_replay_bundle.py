"""The replay bundle a sealed window leaves (`pipeline.replay_bundle`): its sealing and refusal of any file that is not the
manifest's.  CPU only; the store dump itself is `tests/commit/test_store_dump.py`."""
from __future__ import annotations

import hashlib
import json

import pytest
from verity_vllm.pipeline import replay_bundle as RB
from verity_vllm.records import schema_ids


def _sealed(tmp_path):
    p = RB.partial(tmp_path, 1)
    (p / "store" / "raw").mkdir(parents=True)
    (p / "store" / "raw" / "s0.bin").write_bytes(bytes(range(256)))
    (p / "store" / "store.json").write_text("{}")
    (p / "store" / "keys.bin").write_bytes(bytes(64))
    (tmp_path / "binding_map_p1.json").write_text(json.dumps({"entries": []}))
    return RB.seal(tmp_path, 1, {"pair": 1, "run_root": "ab" * 32}, tmp_path / "binding_map_p1.json")


def test_a_sealed_bundle_opens_and_its_partial_is_gone(tmp_path):
    m = _sealed(tmp_path)
    assert not RB.partial(tmp_path, 1).exists() and m["path"] == str(RB.path(tmp_path, 1))
    b = RB.open_bundle(m["path"])
    assert b["manifest_sha256"] == m["sha256"] and b["context"]["run_root"] == "ab" * 32 and b["context"]["schema"] == RB.SCHEMA
    assert set(b["manifest"]["files"]) == {"store/raw/s0.bin", "store/store.json", "store/keys.bin", "binding_map.json", "context.json"}


def test_a_digest_taken_as_the_file_was_written_is_the_manifest_s_and_is_checked_on_opening(tmp_path):
    p = RB.partial(tmp_path, 1)
    (p / "store" / "raw").mkdir(parents=True)
    (p / "store" / "raw" / "s0.bin").write_bytes(bytes(range(256)))
    (p / "store" / "keys.bin").write_bytes(bytes(64))
    (tmp_path / "binding_map_p1.json").write_text("{}")
    good = hashlib.sha256(bytes(range(256))).hexdigest()
    m = RB.seal(tmp_path, 1, {"pair": 1}, tmp_path / "binding_map_p1.json", hashed={"store/raw/s0.bin": good})
    assert m["files"]["store/raw/s0.bin"]["sha256"] == good and RB.open_bundle(m["path"])
    (p / "store").mkdir(parents=True)
    (p / "s.bin").write_bytes(b"x")
    (p / "store" / "keys.bin").write_bytes(bytes(64))
    m = RB.seal(tmp_path, 1, {"pair": 1}, tmp_path / "binding_map_p1.json", hashed={"s.bin": "00" * 32})
    with pytest.raises(ValueError, match="sha256"):
        RB.open_bundle(m["path"])


@pytest.mark.parametrize("tamper,why", [
    (lambda b: (b / "store" / "raw" / "s0.bin").write_bytes(bytes(255) + b"\x01"), "sha256"),
    (lambda b: (b / "store" / "raw" / "s0.bin").unlink(), "missing"),
    (lambda b: (b / "store" / "raw" / "s1.bin").write_bytes(b"x"), "extra"),
])
def test_a_changed_missing_or_extra_file_is_refused(tmp_path, tamper, why):
    b = RB.path(tmp_path, 1)
    _sealed(tmp_path)
    tamper(b)
    with pytest.raises(ValueError, match=why):
        RB.open_bundle(b)


def test_a_discarded_bundle_leaves_nothing(tmp_path):
    (RB.partial(tmp_path, 1) / "store").mkdir(parents=True)
    RB.discard(tmp_path, 1)
    assert not RB.partial(tmp_path, 1).exists() and not RB.path(tmp_path, 1).exists()


def test_the_bundle_reads_its_schema_id_from_the_records_package() -> None:
    """The record's id is written once, in `records.schema_ids`: its writer here and a reader elsewhere cannot hold two."""
    assert RB.SCHEMA is schema_ids.REPLAY_BUNDLE_SCHEMA
