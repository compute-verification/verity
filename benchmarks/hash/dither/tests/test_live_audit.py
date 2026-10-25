"""live_audit and dither_manifest: the pins log and main reach `coins.check`, and the default pins log is the newest head
infra published."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

DITHER = Path(__file__).resolve().parents[1]
PINS = ['{"commit": "aa"}', '{"commit": "bb"}']
MAIN = ["bb", "aa"]


def _module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


@pytest.fixture
def LA(monkeypatch):
    monkeypatch.syspath_prepend(str(DITHER))
    monkeypatch.delitem(sys.modules, "live_audit", raising=False)
    import live_audit
    return live_audit


def _store(monkeypatch, LA, labels: dict[str, list[dict]]) -> list[list[str]]:
    """`research data` answering from `labels` (art -> its challenger-pins labels); the argvs it was called with."""
    calls = []

    def run(argv, **kw):
        calls.append(argv[1:])
        if argv[1:3] == ["data", "select"]:
            return "".join(a + "\n" for a in labels).encode()
        if argv[1:3] == ["data", "labels"]:
            return json.dumps(labels[argv[3]]).encode()
        if argv[1:3] == ["data", "fetch"]:
            d = Path(argv[argv.index("--to") + 1])
            d.mkdir(parents=True)
            (d / "pins.jsonl").write_text(argv[3] + "\n")
            return str(d / "pins.jsonl").encode()
        raise AssertionError(argv)

    monkeypatch.setattr(LA, "_run", run)
    return calls


def head(ts: str, by: str = "infra") -> dict:
    return {"by": by, "key": "challenger-pins", "value": "head", "ts": ts}


def test_newest_pins_is_the_head_infra_labelled_last(LA, monkeypatch, tmp_path):
    calls = _store(monkeypatch, LA, {"art:4b61": [head("2026-10-23T23:43:34Z")],
                                     "art:1929": [head("2026-10-24T08:04:14Z")],
                                     "art:0000": [head("2026-10-24T09:00:00Z", by="kernels")],
                                     "art:75f0": [head("2026-10-23T12:00:00Z")]})
    path, art = LA.newest_pins(tmp_path)
    assert (art, path.read_text()) == ("art:1929", "art:1929\n")
    assert calls[0] == ["data", "select", "--kind", "challenger-pins/v1", "--label", "challenger-pins=head", "--ids"]
    assert calls[-1] == ["data", "fetch", "art:1929", "--to", str(tmp_path / "pins")]


def test_newest_pins_refuses_without_an_infra_head(LA, monkeypatch, tmp_path):
    _store(monkeypatch, LA, {"art:0000": [head("2026-10-24T09:00:00Z", by="kernels")]})
    with pytest.raises(SystemExit, match="labelled `challenger-pins head` by infra: pass --pins"):
        LA.newest_pins(tmp_path)


def test_manifest_check_hands_pins_and_main_to_dither_manifest(LA, monkeypatch, tmp_path):
    got = {}

    def check(*args):
        got["args"] = args
        return {"ok": True}

    main_lines = lambda p: [x for x in Path(p).read_text().split("\n") if x]  # noqa: E731
    CO = _module("coins", stand_in_root=lambda sid: b"r" * 64, key_record=lambda p: "key",
                 R=types.SimpleNamespace(main_lines=main_lines))
    MF = _module("dither_manifest", check=check,
                 expect=lambda m: {"rounds": [b"x" * 64], "draw_after": b"y" * 64})
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "coins", CO)
    monkeypatch.setitem(sys.modules, "dither_manifest", MF)
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "pins.jsonl").write_text("".join(x + "\n" for x in PINS))
    (tmp_path / "main.txt").write_text("".join(x + "\n" for x in MAIN))
    p = {"manifest": tmp_path / "manifest.json", "session": tmp_path / "session", "ledger": tmp_path / "ledger.jsonl",
         "key": tmp_path / "key.json", "pins": tmp_path / "pins.jsonl", "main": tmp_path / "main.txt"}
    a = types.SimpleNamespace(session="s1", period="p1")
    assert LA.manifest_check(a, p, tmp_path) == {"ok": True}
    assert got["args"] == (p["manifest"], p["session"], [str(p["ledger"])], PINS, MAIN, "key", "p1", b"r" * 64)


def test_dither_manifest_check_hands_pins_and_main_to_coins_check(monkeypatch, tmp_path):
    got = {}

    def check(*args):
        got["args"] = args
        return {"ok": True}

    monkeypatch.syspath_prepend(str(DITHER))
    monkeypatch.setitem(sys.modules, "dither_one", _module("dither_one"))
    monkeypatch.setitem(sys.modules, "dither_fused", _module("dither_fused", CM=None, CO=_module("coins", check=check)))
    monkeypatch.delitem(sys.modules, "dither_manifest", raising=False)
    import dither_manifest as MF
    e = {"rounds": [b"x" * 64], "salts": {0: "00"}, "draw_after": b"y" * 64, "differ": [], "manifest_root_listed": True}
    monkeypatch.setattr(MF, "expect", lambda m: e)
    (tmp_path / "manifest.json").write_text(json.dumps({"coins": {"session": "s1"}, "steps": 3}))
    out = MF.check(tmp_path / "manifest.json", tmp_path / "session", ["ledger"], PINS, MAIN, "key", "p1", b"r" * 64)
    assert out["ok"]
    assert got["args"] == (tmp_path / "session", {"session": "s1"}, ["ledger"], PINS, MAIN, e["rounds"], e["salts"],
                           e["draw_after"], "key", "p1", b"r" * 64)
