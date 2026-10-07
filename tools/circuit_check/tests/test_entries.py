"""circuit-check's shared lists, one file per entry (`circuit_check.entries`): the known failures (`known/`).  An entry's file
is named by its key, and a file that is not its key's, or holds more than its one entry, is refused by name."""
from __future__ import annotations

import json

import pytest

from circuit_check import entries as E
from circuit_check import known as KN


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
