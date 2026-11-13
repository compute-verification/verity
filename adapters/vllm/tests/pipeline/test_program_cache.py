"""The derived-Program cache (`verity_vllm.pipeline.program_cache`): the text a derive's key is the SHA-256 of, and what cannot
be keyed; an entry found only whole; a derive published by one rename with its `ENTRY.json`, the first publisher's standing; and
an entry restored in the place of what was there."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from verity_vllm.pipeline import program_cache as PC

DEV = {"name": "NVIDIA L40S", "compute_cap": "8.9"}
ARGV = ["/venv/bin/python", "-m", "the.cli", "build", "--model", "m", "--workload", "?", "--out", "/x/out", "--T", "31"]
KEYED = ('{"argv":["build","--model","m","--workload",{"batch":2,"requests":3,"sampling":"named","sampling_params":{"top_p":0.5},'
         '"sampling_seeds":null},"--T","31"],"device":{"compute_cap":"8.9","name":"NVIDIA L40S"},"tree":"t1"}')
NO_WORKLOAD = '{"argv":["build","--model","m"],"device":{"at":"/dev/gpu0"},"tree":"t1"}'


def _workload(tmp_path, name="workload.json"):
    p = tmp_path / name
    p.write_text(json.dumps({"sweep": {"sampling": "named", "sampling_params": {"top_p": 0.5}, "batch": 2, "row_id": "r"}, "requests": [{}] * 3}))
    return str(p)


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _derived(d, ok=True, text=None):
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"ok": ok}) if text is None else text)
    (d / "instances.json.gz").write_bytes(b"program")
    return d


def test_a_derives_key_is_the_sha256_of_its_command_from_build_on_without_its_out_its_workload_as_the_derive_reads_it_its_tree_and_its_device(tmp_path):
    w = _workload(tmp_path)
    argv = [w if a == "?" else a for a in ARGV]
    assert PC.key(argv, workload=w, tree="t1", device=DEV) == _sha(KEYED)
    assert PC.key(argv, workload=Path(w), tree="t1", device=dict(reversed(DEV.items()))) == _sha(KEYED)
    assert PC.key(argv, workload="", tree="t1", device=DEV) == _sha(KEYED), "with no workload given, the command's own"
    other = _workload(tmp_path, "other.json")
    assert PC.key([*argv[:4], "--out", "/y", *argv[4:8], *argv[10:]], workload=other, tree="t1", device=DEV) == _sha(KEYED)
    assert PC.key([*argv[3:6]], workload="/no/such/file", tree="t1", device={"at": Path("/dev/gpu0")}) == _sha(NO_WORKLOAD)


@pytest.mark.parametrize("change", [{"tree": None}, {"tree": ""}, {"device": None}, {"device": {}}, {"argv": ["python", "-m", "the.cli", "manifest"]}])
def test_a_derive_with_no_tree_no_device_or_no_build_in_its_command_is_not_keyed(tmp_path, change):
    given = {"argv": ["build", "--model", "m"], "tree": "t1", "device": DEV, **change}
    assert PC.key(given["argv"], workload=_workload(tmp_path), tree=given["tree"], device=given["device"]) is None


def test_an_entry_is_found_only_with_its_entry_file_and_a_program_whose_result_is_ok(tmp_path):
    whole = tmp_path / "whole"
    _derived(whole / "program")
    assert PC.lookup(tmp_path, "whole") is None, "a program and no ENTRY.json"
    (whole / "ENTRY.json").write_text("{}")
    assert PC.lookup(str(tmp_path), "whole") == whole
    (whole / "ENTRY.json").unlink()
    (whole / "ENTRY.json").mkdir()
    assert PC.lookup(tmp_path, "whole") is None, "ENTRY.json is a directory"
    for i, text in enumerate(('{"ok": false}', '{"ok": 1}', '{"ok": "true"}', "[true]", "not json", "{}")):
        _derived(tmp_path / f"e{i}" / "program", text=text)
        (tmp_path / f"e{i}" / "ENTRY.json").write_text("{}")
        assert PC.lookup(tmp_path, f"e{i}") is None, text
    (tmp_path / "bare").mkdir()
    (tmp_path / "bare" / "ENTRY.json").write_text("{}")
    assert PC.lookup(tmp_path, "bare") is None and PC.lookup(tmp_path, "absent") is None


def test_a_derive_that_succeeded_is_published_whole_under_its_key_with_its_entry_file(tmp_path):
    derived, root = _derived(tmp_path / "row" / "build_request"), tmp_path / "a" / "cache"
    stale = root / f".k1.tmp-{os.getpid()}"
    (stale / "program").mkdir(parents=True)
    (stale / "program" / "left by a run that died").write_text("")
    assert PC.publish(str(root), "k1", str(derived), {"row": "r", "stage": Path("build_request"), "device": DEV}) == root / "k1"
    assert sorted(p.name for p in root.iterdir()) == ["k1"]
    assert sorted(p.name for p in (root / "k1").rglob("*")) == ["ENTRY.json", "instances.json.gz", "program", "result.json"]
    assert (root / "k1" / "ENTRY.json").read_text() == ('{\n "device": {\n  "compute_cap": "8.9",\n  "name": "NVIDIA L40S"\n },\n "key": "k1",\n "row": "r",\n'
                                                       ' "schema": "verity-vllm/program-cache-entry/v1",\n "stage": "build_request"\n}\n')
    assert (root / "k1" / "program" / "instances.json.gz").read_bytes() == b"program" and PC.lookup(root, "k1") == root / "k1"
    assert PC.publish(root, "k2", derived, {}) == root / "k2", "a second key in a root that is there"
    assert json.loads((root / "k2" / "ENTRY.json").read_text()) == {"schema": "verity-vllm/program-cache-entry/v1", "key": "k2"}


def test_a_derive_that_did_not_succeed_is_not_published_and_an_entry_that_is_there_is_kept(tmp_path):
    root = tmp_path / "cache"
    assert PC.publish(root, "k1", _derived(tmp_path / "failed", ok=False), {}) is None and not root.exists()
    assert PC.publish(root, "k1", tmp_path / "absent", {}) is None and not root.exists()
    first = PC.publish(root, "k1", _derived(tmp_path / "first"), {"row": "first"})
    second = _derived(tmp_path / "second")
    (second / "instances.json.gz").write_bytes(b"another program")
    assert PC.publish(root, "k1", second, {"row": "second"}) == first == root / "k1"
    assert json.loads((first / "ENTRY.json").read_text())["row"] == "first" and (first / "program" / "instances.json.gz").read_bytes() == b"program"
    assert sorted(p.name for p in root.iterdir()) == ["k1"]


def test_a_publisher_that_loses_the_rename_leaves_the_other_ones_entry_and_nothing_of_its_own(tmp_path, monkeypatch):
    root, rename = tmp_path / "cache", os.rename

    def after_another_job(src, dst):
        _derived(Path(dst) / "program")
        (Path(dst) / "ENTRY.json").write_text('{"row": "the other job"}')
        rename(src, dst)

    monkeypatch.setattr(PC.os, "rename", after_another_job)
    assert PC.publish(root, "k1", _derived(tmp_path / "mine"), {"row": "mine"}) == root / "k1"
    assert (root / "k1" / "ENTRY.json").read_text() == '{"row": "the other job"}' and sorted(p.name for p in root.iterdir()) == ["k1"]


def test_an_entry_is_restored_as_a_copy_in_the_place_of_what_was_there(tmp_path):
    entry = PC.publish(tmp_path / "cache", "k1", _derived(tmp_path / "derived"), {})
    assert entry is not None
    out = tmp_path / "row" / "build_request"
    (out / "sub").mkdir(parents=True)
    (out / "sub" / "of an older derive").write_text("")
    PC.restore(entry, str(out))
    assert sorted(p.name for p in out.rglob("*")) == ["instances.json.gz", "result.json"]
    (out / "instances.json.gz").write_bytes(b"changed in the row")
    assert (entry / "program" / "instances.json.gz").read_bytes() == b"program"
    PC.restore(entry, tmp_path / "new" / "row" / "build_request")
    assert (tmp_path / "new" / "row" / "build_request" / "instances.json.gz").read_bytes() == b"program"
