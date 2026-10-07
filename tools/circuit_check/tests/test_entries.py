"""circuit-check's shared lists, one file per entry (`circuit_check.entries`): the known failures (`known/`) and the pins no
module owns (``pins/``).  An entry's file is named by its key, and a file that is not its key's, or holds more than its one
entry, is refused by name."""
from __future__ import annotations

import json

import pytest

from circuit_check import entries as E
from circuit_check import known as KN
from circuit_check import sidecars as SC


def test_an_entry_file_is_named_by_its_key():
    assert E.name("GatherBf16x64_v2") == "GatherBf16x64_v2"
    assert E.name("partition/gate-recomputed ScaledMmFp8Block_v1") == "partition_gate-recomputed_ScaledMmFp8Block_v1-d6a67494"
    assert E.name("RMSNormTriton_v1{N=64,EPS=1e-05}") == "RMSNormTriton_v1_N_64_EPS_1e-05_-e7566f8c"
    assert E.name("a b") != E.name("a/b") and E.name("a b").startswith("a_b-")       # one slug, two keys: the hash tells them
    long = "X" * (E.WIDTH + 1)
    assert E.name(long) == f"{'X' * E.WIDTH}-{E.name(long)[-8:]}" and E.name("X" * E.WIDTH) == "X" * E.WIDTH


def test_every_file_of_the_trees_lists_is_an_entry_they_load():
    """A file the loaders would not read (another suffix, another directory) is no entry anyone sees: there is none."""
    known = sorted(p for p in KN.DIR.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert known == E.files(KN.DIR / "failures", ".toml") + E.files(KN.DIR / "pieces", ".toml")
    assert {p.stem for p in known} == {E.name(k) for k in (*KN.KNOWN, *KN.KNOWN_PIECES)}
    shared = sorted(p for p in SC.SHARED_PINS.rglob("*") if p.is_file())
    assert shared and set(shared) <= set(SC.pin_files())


def _known(tmp_path, name: str, text: str):
    (tmp_path / name).write_text(text)
    return KN.load(tmp_path)


def test_a_known_entry_is_its_keys_file_with_a_reason(tmp_path):
    key = "partition/gate-recomputed Syn_v1"
    assert _known(tmp_path, f"{E.name(key)}.toml", f'key = "{key}"\nreason = "why"\n') == {key: "why"}
    with pytest.raises(ValueError, match=f"whose file is {E.name('Other_v1')}.toml"):
        _known(tmp_path, "Syn_v1.toml", 'key = "Other_v1"\nreason = "why"\n')


def test_a_known_entry_with_another_field_is_refused(tmp_path):
    with pytest.raises(ValueError, match="a key and a reason"):
        _known(tmp_path, "Syn_v1.toml", 'key = "Syn_v1"\nreason = "why"\nalso = "x"\n')


@pytest.mark.parametrize("rel,pins,why", [
    ("pieces/Syn_v1.json", {"pieces": {"Other_v1": {"and": 1}}}, "whose file is Other_v1.json"),
    ("pieces/Syn_v1.json", {"pieces": {"Syn_v1": {"and": 1}, "Two_v1": {"and": 2}}}, "is one pin"),
    ("definitions/Syn_v1.json", {"pieces": {"Syn_v1": {"and": 1}}}, "is one pin"),
    ("Syn_v1.json", {"pieces": {"Syn_v1": {"and": 1}}}, "is one pin"),
], ids=["misnamed", "two-pins", "wrong-kind", "no-kind-directory"])
def test_a_shared_pins_file_that_is_not_its_one_pins_is_refused(tmp_path, monkeypatch, rel, pins, why):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(pins))
    monkeypatch.setattr(SC, "_SHARED", [tmp_path])
    with pytest.raises(ValueError, match=why):
        SC.pin_files()


def test_a_shared_pin_is_its_keys_file(tmp_path, monkeypatch):
    key = "GatherBf16x3_v1{N=3}"
    p = tmp_path / "definitions" / f"{E.name(key)}.json"
    p.parent.mkdir()
    p.write_text(json.dumps({"definitions": {key: {"and": 7}}}))
    monkeypatch.setattr(SC, "_SHARED", [tmp_path])
    assert SC.shared_pin("definitions", key) == p and SC.pin_files()[0] == p
