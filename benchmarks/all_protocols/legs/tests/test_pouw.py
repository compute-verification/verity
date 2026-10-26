"""PoUW's leg on one window of the torch loop: it answers the window's PoUW window (the runner's ``--pouw h1``), whatever
law the integrity plan draws under, and a window without one has no PoUW case, since PoUW's verifier of record refuses the
integrity window under any law."""
from __future__ import annotations

import json
import shutil
import stat
from fractions import Fraction

import pytest

import pouw
from verity.core.protocols.pouw import core_check as CC, operands as OP
from verity.core.protocols.pouw import window as Wd
from verity_torch_loop import serve

UNPINNED = "params: check.template is not a tile template the verifier pins"
NOT_POUWS = "commitment: window is not {context, kind, public} with strings and an object"
TEMPLATE = 'ScaledMmS8Coordinate_v1{K=128,DOT={"fn":"BlackwellS8ImmaDot32_v1"}}'


@pytest.fixture(scope="module")
def period(tmp_path_factory):
    d = tmp_path_factory.mktemp("period") / "p0"
    assert serve.main(["--out", str(d), "--requests", "3,7", "--steps", "2", "--run-id", "allp-pouw-test"]) == 0
    return d


def _work_law(period, dst):
    """A copy of the period whose integrity window is registered under the work law ``work:64``, in the form S1's
    ``--tie`` registers it (one stratum of the loop's template, work 1 and floor 1 a unit)."""
    shutil.copytree(period, dst)
    rec = json.loads((dst / "registration.json").read_text())
    rec["law"] = {"k": 64, "law": "work", "strata": [{"floor": 1, "k": 64, "template": TEMPLATE,
                                                      "units": [[0, rec["population"]]], "work": 1}]}
    (dst / "registration.json").write_text(CC.line(rec) + "\n")
    return dst


def test_a_window_without_a_pouw_window_has_no_case_under_either_law(period, tmp_path):
    """No PoUW window: the leg raises, which the runner records as work's error, so the composed verdict waits on work
    instead of rejecting by reject dominance."""
    for d in (period, _work_law(period, tmp_path / "work")):
        with pytest.raises(pouw.NoWindow, match="--pouw h1"):
            pouw.leg(d, "local", tmp_path / "out")
        assert not (tmp_path / "out" / "pouw.answer.json").exists()


def test_core_refuses_the_integrity_window_as_pouws_under_either_law(period, tmp_path):
    """Why the leg doesn't answer the integrity window: PoUW refuses it at the loop's template (no pinned tile), and at
    the H1 tile's parameters refuses its registration, which carries none of PoUW's items, whichever law it was drawn
    under."""
    work = pouw.served(period)
    assert work == {"template": TEMPLATE, "units": 2 * 2 * 2 * 128, "k": 128, "macs": 2 * 2 * 2 * 128 * 128,
                    "max_tokens": [2, 2]}
    h1 = pouw.h1_params(period, work)
    at_template = {**h1, "check": {"entries": 2, "scheme": CC.SCHEME, "template": TEMPLATE, "tile_work": work["k"]}}
    for d in (period, _work_law(period, tmp_path / "work")):
        rec = CC.line(json.loads((d / "registration.json").read_text())).encode()
        assert CC.answer(CC.line(at_template).encode(), rec, b"")["reason"] == UNPINNED
        assert CC.answer(CC.line(h1).encode(), rec, b"")["reason"] == NOT_POUWS


def _exe(tmp_path, line: str):
    exe = tmp_path / "pouw-check"
    exe.write_text(f"#!/bin/sh\nprintf '%s\\n' '{line}'\nexit 1\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


def _h1_window(period, dst, n_real: int = 128, operands=None, **edit):
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
           "window": {"kind": "served", "context": p["context"], "public": items,
                      **({OP.KEY: operands(p["context"])} if operands else {})}}
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
    assert detail["operands"] is None


def test_an_h1_windows_operands_are_reported_with_what_they_dont_establish(period, tmp_path):
    """The record names the operands' root X, each value's root, and the gaps: the A rows' tie to the served x rows is a
    gather (CIR-100), not proved."""
    block = {}

    def operands(context):
        block.update(OP.block("p", OP.commit("p", context, {"a": [[1, 2], [3, 4]], "b": [[5, 6, 7, 8]]})))
        return block

    d, _, _ = _h1_window(period, tmp_path / "h1", operands=operands)
    assert pouw.leg(d, "local", tmp_path / "out")["verdict"] == "accept"
    got = json.loads((tmp_path / "out" / "pouw.detail.json").read_text())["operands"]
    assert got["held"] and got["X"] == OP.root(block).hex()
    assert got["roots"] == {v: e["root"] for v, e in block["values"].items()}
    assert any("CIR-100" in g for g in got["gaps"])


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


def test_a_work_law_window_with_its_h1_window_is_accepted(period, tmp_path):
    """S1's window: the integrity plan draws under ``work:64`` (``--tie``), and PoUW answers the window's own H1 window,
    which it accepts, crediting the same W as under a subset law: the integrity law never reaches PoUW's case."""
    d, w, k = _h1_window(_work_law(period, tmp_path / "work"), tmp_path / "h1")
    a = pouw.leg(d, "local", tmp_path / "out")
    assert a["verdict"] == "accept" and a["reason"] is None, a["reason"]
    assert json.loads((tmp_path / "out" / "pouw.detail.json").read_text())["W"] == 128 * 2048
    same, _, _ = _h1_window(period, tmp_path / "subset")
    assert pouw.case(d)[0] == pouw.case(same)[0]


def test_the_lean_answer_must_agree(period, tmp_path, monkeypatch):
    d, _, _ = _h1_window(period, tmp_path / "h1")
    want = CC.line(CC.case(pouw.case(d)[0]))
    monkeypatch.setenv(pouw.LEAN_ENV, str(_exe(tmp_path, want)))
    assert pouw.leg(d, "local", tmp_path / "agree")["verdict"] == "accept"
    assert json.loads((tmp_path / "agree" / "pouw.detail.json").read_text())["answered_by"] == ["core_check", "pouw-check"]
    monkeypatch.setenv(pouw.LEAN_ENV, str(_exe(tmp_path, want.replace('"accept"', '"reject"'))))
    with pytest.raises(RuntimeError, match="disagree"):
        pouw.leg(d, "local", tmp_path / "disagree")


def test_a_window_with_the_loops_salt_has_live_coins_and_its_salt_sessions_inputs(period, tmp_path):
    """With ``salt.json`` beside the H1 window, its parameters' coins are live and hold the salt session's inputs (``live``),
    which core's parameter rule takes (and refuses live coins without); where core takes no live parameters, the file is
    not read."""
    work = pouw.served(period)
    seeded = pouw.h1_params(period, work)
    assert Wd.COINS not in seeded["check"] and "live" not in seeded
    d = tmp_path / "p0"
    shutil.copytree(period, d)
    (d / pouw.H1_DIR).mkdir()
    live = {k: f"{k}-of-the-session" for k in ("key", "ledger", "main", "pins", "session")}
    (d / pouw.H1_DIR / pouw.SALT).write_text(json.dumps({"answer": {"epoch": 0}, "live": live}))
    h1 = pouw.h1_params(d, work)
    if pouw.LIVE_PARAM is None:
        assert h1 == seeded
        return
    assert h1["check"][Wd.COINS] == "live" and h1[pouw.LIVE_PARAM] == live
    assert CC.params(h1)[pouw.LIVE_PARAM] == live
    with pytest.raises(CC.Refused, match="live is exactly when"):
        CC.params({k: v for k, v in h1.items() if k != pouw.LIVE_PARAM})
