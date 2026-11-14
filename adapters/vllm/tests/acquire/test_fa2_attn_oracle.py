"""The FA2 attention reference (`fa2_attn_oracle`) on the CPU: each case's words by their digest, a document against itself and against
one with moved words, and the command that writes a case and compares two documents."""
from __future__ import annotations

import copy
import hashlib
import json
import runpy
import sys

import pytest

from tests.acquire import fa2_attn_oracle as O

#: sha256 of each case's words, `json.dumps(words(name), sort_keys=True)`
WORDS = {
    "g2a": "7c89d16cd541f623e4f966bcbe3eb7e27542695e9d26414b9ac6c00c00fb033c",
    "g2b": "600b170de9b5350454d16d602212439150e53f1330ea7e849481d35a3bd19a97",
    "p0a": "f2f0b8a740a8b960f114bb82f436518779bc76c66c460b8266cd894f36811841",
    "p0b": "b10eb0c3b4f97e33210cf9b4c9d7e1d57972a4f68d6fff34de6971164194637e",
    "p0c": "91d713b884967dfa886b328642c13925ab65b642a964d1e7382df6b342ade7d0",
    "p0d": "216281c928181ff45757ad9e4ce77ec8a951fbc9cda999f04523b39976c7e1af",
}


def document(name: str) -> dict:
    return O.hexify(O.run_case(name))


def words(name: str) -> dict:
    """A case's document without each row's `lse_approx`, a float through numpy's `log` and the one value that is not a Definition's word."""
    doc = document(name)
    for row in doc["rows"]:
        del row["lse_approx"]
    return doc


def test_every_case_has_a_digest() -> None:
    assert sorted(WORDS) == sorted(O.CASES)


@pytest.mark.parametrize("name", sorted(WORDS))
def test_a_case_gives_its_words(name: str) -> None:
    assert hashlib.sha256(json.dumps(words(name), sort_keys=True).encode()).hexdigest() == WORDS[name]


def test_compare_counts_the_words_that_differ(capsys: pytest.CaptureFixture[str]) -> None:
    ref = O.unhex(document("p0c"))
    assert O.compare(ref, copy.deepcopy(ref)) == 0 and "BIT-EXACT" in capsys.readouterr().out
    moved = copy.deepcopy(ref)
    moved["rows"][0]["out"][0] ^= 1
    moved["rows"][1]["blocks"][0]["o"][3] ^= 1
    assert O.compare(ref, moved) == 2 and "2 mismatching words" in capsys.readouterr().out


def test_the_command_writes_a_case_and_compares_two_documents(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    out, other = tmp_path / "p0c.json", tmp_path / "moved.json"
    monkeypatch.setattr(sys, "argv", ["fa2_attn_oracle.py", "--case", "p0c", "--out", str(out)])
    runpy.run_path(O.__file__, run_name="__main__")
    doc = json.loads(out.read_text())
    assert doc == document("p0c") and capsys.readouterr().out.startswith("p0c: M=3 T=160 ")
    doc["rows"][0]["out"][0] = f"{int(doc['rows'][0]['out'][0], 16) ^ 1:04x}"
    other.write_text(json.dumps(doc))
    for dump, code in ((out, 0), (other, 1)):
        monkeypatch.setattr(sys, "argv", ["fa2_attn_oracle.py", "--compare", str(out), str(dump)])
        with pytest.raises(SystemExit) as stop:
            runpy.run_path(O.__file__, run_name="__main__")
        assert stop.value.code == code
