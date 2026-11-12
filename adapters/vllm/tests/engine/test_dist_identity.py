"""`verity_vllm.engine.engine_profile.dist_identity`: an installed distribution's version, the digest of its RECORD, the URL pip
installed it from and its wheel's sha256 (the one pip recorded, else the local wheel file's own); stand-in distributions, no
package installed."""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json

import pytest

from verity_vllm.engine import engine_profile as EP


class _Dist:
    version = "1.2.3"

    def __init__(self, **files: str):
        self.files = files

    def read_text(self, name: str) -> str | None:
        return self.files.get(name)


def _identity(monkeypatch, **files: str) -> dict:
    monkeypatch.setattr(md, "distribution", lambda name: _Dist(**files))
    return EP.dist_identity("stand-in")


def _direct(doc) -> dict[str, str]:
    return {"direct_url.json": json.dumps(doc)}


BARE = {"installed": True, "version": "1.2.3", "record_sha256": None, "direct_url": None, "wheel_sha256": None}


def test_a_distribution_that_is_not_installed_says_so():
    assert EP.dist_identity("verity-vllm-no-such-distribution") == {"installed": False}


def test_a_distribution_with_neither_file_has_its_version_alone(monkeypatch):
    assert _identity(monkeypatch) == BARE
    assert _identity(monkeypatch, **{"RECORD": "", "direct_url.json": ""}) == BARE


def test_the_record_is_named_by_its_sha256(monkeypatch):
    record = "pkg/__init__.py,sha256=abc,12\n"
    assert _identity(monkeypatch, RECORD=record) == {**BARE, "record_sha256": hashlib.sha256(record.encode()).hexdigest()}


def test_the_wheels_hash_is_the_one_pip_recorded(monkeypatch, tmp_path):
    wheel = tmp_path / "pkg.whl"
    wheel.write_bytes(b"not the bytes pip hashed")
    for url in ("https://example.invalid/pkg.whl", f"file://{wheel}"):
        got = _identity(monkeypatch, **_direct({"url": url, "archive_info": {"hash": "sha256=0123abcd"}}))
        assert got == {**BARE, "direct_url": url, "wheel_sha256": "0123abcd"}


@pytest.mark.parametrize("rest", [{}, {"archive_info": None}, {"archive_info": {}}, {"archive_info": {"hash": None}},
                                  {"archive_info": {"hash": "md5=0123"}}, {"archive_info": {"hashes": {"sha256": "0123"}}},
                                  {"dir_info": {"editable": True}}],
                         ids=["absent", "null", "empty", "hash null", "another digest", "hashes only", "a directory"])
def test_a_direct_url_without_a_recorded_sha256_has_no_wheel_hash_unless_its_file_is_here(monkeypatch, tmp_path, rest):
    remote = "https://example.invalid/pkg.whl"
    assert _identity(monkeypatch, **_direct({"url": remote, **rest})) == {**BARE, "direct_url": remote}
    gone = f"file://{tmp_path / 'gone.whl'}"
    assert _identity(monkeypatch, **_direct({"url": gone, **rest})) == {**BARE, "direct_url": gone}
    wheel = tmp_path / "pkg.whl"
    wheel.write_bytes(b"wheel bytes")
    here = f"file://{wheel}"
    assert _identity(monkeypatch, **_direct({"url": here, **rest})) == {
        **BARE, "direct_url": here, "wheel_sha256": hashlib.sha256(b"wheel bytes").hexdigest()}


def test_a_direct_url_that_is_no_json_or_names_no_url_gives_none(monkeypatch):
    assert _identity(monkeypatch, **{"direct_url.json": "{not json"}) == BARE
    assert _identity(monkeypatch, **_direct({})) == BARE
    assert _identity(monkeypatch, **_direct({"archive_info": {"hash": "sha256=0123"}})) == {**BARE, "wheel_sha256": "0123"}
