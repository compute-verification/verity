"""PoUW's leg on one period of the torch loop: PoUW's verifier of record refuses the loop's template at its parameters, so
the period certifies no work, and that refusal is the template's alone."""
from __future__ import annotations

import json
import stat

import pytest

import pouw
from verity.core.protocols.pouw import core_check as CC
from verity_torch_loop import serve

UNPINNED = "params: check.template is not a tile template the verifier pins"


@pytest.fixture(scope="module")
def period(tmp_path_factory):
    d = tmp_path_factory.mktemp("period") / "p0"
    assert serve.main(["--out", str(d), "--requests", "3,7", "--steps", "2", "--run-id", "allp-pouw-test"]) == 0
    return d


def test_the_loop_is_refused_at_its_template_and_certifies_no_work(period, tmp_path):
    a = pouw.leg(period, "local", tmp_path)
    assert a == {"protocol": CC.ID, "verdict": "reject", "reason": UNPINNED, "commitment": None, "public": None,
                 "profile": None}
    assert (tmp_path / "pouw.answer.json").read_text() == CC.line(a) + "\n"
    assert CC.case(json.loads((tmp_path / "pouw.case.json").read_text())) == a
    d = json.loads((tmp_path / "pouw.detail.json").read_text())
    assert d["certified_work"] is None and d["answered_by"] == ["core_check"]
    assert d["served"] == {"template": 'ScaledMmS8Coordinate_v1{K=128,DOT={"fn":"BlackwellS8ImmaDot32_v1"}}',
                           "units": 2 * 2 * 2 * 128, "k": 128, "macs": 2 * 2 * 2 * 128 * 128, "max_tokens": [2, 2]}


def test_only_the_template_is_refused(period):
    """The same parameters naming a template the verifier pins, with its work and query, pass PoUW's parameter rule."""
    p, _ = pouw.case(period)
    [(name, tile)] = [(n, t) for n, t in CC.TILES.items() if n.startswith("PouwH1Tile_v1{")]
    j = p["params"]
    pinned = {**j, "check": {**j["check"], "template": name, "tile_work": tile.tile_work}, "query": tile.query}
    assert CC.params(pinned)["check"]["template"] == name
    with pytest.raises(CC.Refused, match="not a tile template the verifier pins"):
        CC.params(j)


def _exe(tmp_path, line: str):
    exe = tmp_path / "pouw-check"
    exe.write_text(f"#!/bin/sh\nprintf '%s\\n' '{line}'\nexit 1\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


def test_the_lean_answer_must_agree(period, tmp_path, monkeypatch):
    want = CC.line(CC.case(pouw.case(period)[0]))
    monkeypatch.setenv(pouw.LEAN_ENV, str(_exe(tmp_path, want)))
    assert pouw.leg(period, "local", tmp_path / "agree")["reason"] == UNPINNED
    assert json.loads((tmp_path / "agree" / "pouw.detail.json").read_text())["answered_by"] == ["core_check", "pouw-check"]
    monkeypatch.setenv(pouw.LEAN_ENV, str(_exe(tmp_path, want.replace("reject", "accept"))))
    with pytest.raises(RuntimeError, match="disagree"):
        pouw.leg(period, "local", tmp_path / "disagree")
