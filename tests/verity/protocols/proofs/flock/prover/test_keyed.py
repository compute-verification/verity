"""`keyed.key`'s code part leaves out where entries are kept and which run asks, never what a build reads."""
from __future__ import annotations

import os

import keyed

#: reads `RESEARCH_RUN_ID` and `RESEARCH_RUN_DIR` (`os.environ.get`), so both are in its closure's environment
MODULE = "verity_flock.class_statement"


def _key(monkeypatch, **env) -> str | None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    keyed._code.cache_clear()
    return keyed.key("part", code=(MODULE,))


def test_a_run_s_identity_is_not_in_the_key(monkeypatch):
    a = _key(monkeypatch, RESEARCH_RUN_ID="r1", RESEARCH_RUN_DIR="/runs/r1")
    b = _key(monkeypatch, RESEARCH_RUN_ID="r2", RESEARCH_RUN_DIR="/runs/r2")
    assert a is not None and a == b


def test_a_file_rewritten_to_the_same_length_in_one_tick_is_keyed_by_its_new_bytes(tmp_path):
    f = tmp_path / "inner.json"
    f.write_text('{"regions": 2}')
    st = f.stat()
    before = keyed.key("run", str(f))
    f.write_text('{"regions": 4}')
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert f.stat().st_ino == st.st_ino and f.stat().st_size == st.st_size
    assert keyed.key("run", str(f)) != before


def test_a_variable_the_closure_reads_is_still_in_the_key(monkeypatch):
    names = {v for v, _ in __import__("json").loads(keyed._code((MODULE,))[0])[-1]}
    assert names, f"{MODULE} reads no environment variable keyed.py sees"
    other = sorted(names)[0]
    assert _key(monkeypatch, **{other: "x"}) != _key(monkeypatch, **{other: "y"})
