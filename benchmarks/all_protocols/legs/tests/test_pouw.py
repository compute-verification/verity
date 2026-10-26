"""PoUW's leg on one period of the torch loop: PoUW's verifier of record refuses the loop's template at its parameters, so
the period certifies no work, and that refusal is the template's alone."""
from __future__ import annotations

import json
import shutil
import stat
from fractions import Fraction

import pytest

import pouw
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import window as Wd
from verity_toy import serve

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


def _h1_window(period, dst, n_real: int = 128, **edit):
    """A copy of the period with an H1 window registered at the verifier's parameters (`pouw.h1_params`) and the proof
    service's acceptance of it, in the form `pouw_check_cases.honest` builds: a stand-in for the loop's ``--pouw h1``."""
    shutil.copytree(period, dst)
    p = {**pouw.h1_params(period, pouw.served(period)), **edit}
    tw, bits = p["check"]["tile_work"], p["grid_bits"]
    w = Wd.bucket(n_real, bits)
    eps, delta = Fraction(p["eps"]), Fraction(p["delta"])
    k = Wd.k_prime(eps, delta, w.credited * tw, w.n, tw)
    items = {"check": dict(p["check"]), "root_W": "ab" * 64, "work": w.to_json(tw),
             "sampling": Wd.sampling(eps, delta, w, tw, bits), "length": Wd.length_item(p["length"]["max_tokens"])}
    rec = {"format": CC.FORMATS[0], "protocol": CC.ONE_STAGE, "program": "01" * 64, "query": p["query"],
           "partition": "02" * 64, "population": w.n, "law": {"law": "subset", "k": k}, "scheme": CC.SCHEME,
           "roots": [{"port": "entries", "root": "03" * 64, "leaves": p["check"]["entries"], "domain": "04" * 64},
                     {"port": "W", "root": "ab" * 64, "leaves": 16, "domain": "05" * 64}],
           "leaf_layer": None, "anchors": [], "previous": None,
           "window": {"kind": "served", "context": p["context"], "public": items}}
    digest = CC.digest(CC.commitment(CC.line(rec).encode())).hex()
    answer = {"commitment": digest, "protocol": CC.ONE_STAGE, "reason": None, "verdict": "accept",
              "profile": {"covered": [], "delta": p["delta"], "drawn": {CC.LEVEL: k}, "linked": True,
                          "levels": [{"draw": {"subset": k}, "name": CC.LEVEL, "query": rec["partition"], "size": w.n}],
                          "prescribed": [], "programs": [rec["program"]]},
              "public": {"draw": {"k": k, "law": "subset", "population": w.n, "units": list(range(0, 2 * k, 2))}}}
    (dst / pouw.H1_DIR).mkdir()
    (dst / pouw.H1_DIR / "registration.json").write_text(CC.line(rec) + "\n")
    (dst / pouw.H1_DIR / "answer.json").write_text(CC.line(answer) + "\n")
    return dst, w, k


def test_an_honest_h1_window_is_accepted_with_its_credited_work_and_eps_at_both_deltas(period, tmp_path):
    d, w, k = _h1_window(period, tmp_path / "h1")
    a = pouw.leg(d, "local", tmp_path / "out")
    assert a["verdict"] == "accept" and a["reason"] is None
    tw = CC.TILES[pouw.H1_TEMPLATE].tile_work
    assert a["public"]["work"] == w.to_json(tw) and a["public"]["drawn"] == list(range(0, 2 * k, 2))
    detail = json.loads((tmp_path / "out" / "pouw.detail.json").read_text())
    assert detail["window"] == "h1" and detail["W"] == w.credited * tw == 128 * 2048
    assert detail["evidence"] == "pouw/answer.json"
    assert [q["delta"] for q in detail["quotes"]] == list(pouw.QUOTE_DELTAS)
    for q in detail["quotes"]:
        assert q["eps"] == str(Wd.eps_at(k, Fraction(q["delta"]), w.credited * tw, w.n, tw))
    assert Fraction(detail["quotes"][1]["eps"]) <= Fraction(pouw.H1_EPS)
    assert Fraction(detail["quotes"][0]["eps"]) < Fraction(detail["quotes"][1]["eps"])


def test_an_h1_window_registered_at_another_eps_is_refused(period, tmp_path):
    """The parameters are the verifier's: a window drawn at the prover's own ε certifies nothing."""
    d, _, _ = _h1_window(period, tmp_path / "h1", eps="1")
    a = pouw.leg(d, "local", tmp_path / "out")
    assert a["verdict"] == "reject" and a["reason"].startswith("public-value: sampling")
    detail = json.loads((tmp_path / "out" / "pouw.detail.json").read_text())
    assert detail["W"] == 0 and all(q["eps"] is None for q in detail["quotes"])


def test_an_h1_window_without_the_services_answer_is_refused(period, tmp_path):
    d, _, _ = _h1_window(period, tmp_path / "h1")
    (d / pouw.H1_DIR / "answer.json").unlink()
    a = pouw.leg(d, "local", tmp_path / "out")
    assert a["verdict"] == "reject" and a["reason"].startswith("evidence:"), a["reason"]


def test_the_lean_answer_must_agree(period, tmp_path, monkeypatch):
    want = CC.line(CC.case(pouw.case(period)[0]))
    monkeypatch.setenv(pouw.LEAN_ENV, str(_exe(tmp_path, want)))
    assert pouw.leg(period, "local", tmp_path / "agree")["reason"] == UNPINNED
    assert json.loads((tmp_path / "agree" / "pouw.detail.json").read_text())["answered_by"] == ["core_check", "pouw-check"]
    monkeypatch.setenv(pouw.LEAN_ENV, str(_exe(tmp_path, want.replace("reject", "accept"))))
    with pytest.raises(RuntimeError, match="disagree"):
        pouw.leg(period, "local", tmp_path / "disagree")
