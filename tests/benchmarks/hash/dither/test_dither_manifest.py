"""dither_manifest: the pins log and main reach `coins.check`."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

DITHER = (Path(__file__).resolve().parents[4] / "benchmarks" / "hash" / "dither")
PINS = ['{"commit": "aa"}', '{"commit": "bb"}']
MAIN = ["bb", "aa"]


def _module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


def test_dither_manifest_check_hands_pins_and_main_to_coins_check(monkeypatch: pytest.MonkeyPatch,
                                                                  tmp_path: Path) -> None:
    got = {}

    def check(*args) -> dict:
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
