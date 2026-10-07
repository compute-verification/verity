"""PoUW's check as a core protocol (`core_check`), the Python reference of the Lean port `pouw_check`, and the difftest's
Python half.

`verity/experimental/core/protocols/pouw/check/scripts/difftest_vectors.py` writes generated `serve` cases (honest
windows, and one mutation each aimed at one refusal) and this module's answer to each.  The Lean check runs the Lean half
(`runs` in that package's `lean-audit.json`: `PouwCheck.DifftestMain` answers every case with `pouw_check` and fails on
any line that differs), against the data's SHA-256.  Its pass is cached on the Lean package's files alone, so the test
here is what says the data is still what this module computes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import pytest

from verity.experimental.protocols.pouw import core_check as CC
from verity.experimental.protocols.pouw import window as Wd
from verity.experimental.protocols.sampled_proofs import service as S
from verity.experimental.protocols.sampled_proofs.one_stage import registration as R

ROOT = Path(__file__).resolve().parents[5]
PKG = ROOT / "verity/experimental/core/protocols/pouw/check"
_spec = importlib.util.spec_from_file_location("pouw_check_vectors", PKG / "scripts/difftest_vectors.py")
vectors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vectors)

#: the refusal each mutation aims at: the reason's start
EXPECT = {
    "params-keys": "params: keys are not", "params-check-keys": "params: check's keys", "params-entries": "params: check.entries",
    "params-scheme": "params: check.scheme", "params-template": "params: check.template",
    "params-tile-work": "params: check.tile_work", "params-context": "params: context",
    "params-delta-type": "params: delta is not a string", "params-delta-form": 'params: delta: "01" is not a natural',
    "params-delta-den": 'params: delta: "1/1" has denominator 1', "params-delta-zero-den": 'params: delta: "3/0" has denominator 0',
    "params-delta-lowest": 'params: delta: "2/4" is not in lowest terms', "params-delta-ratio": 'params: delta: "1/2/3" is not a ratio',
    "params-delta-range": "params: delta is not in (0, 1)", "params-delta-zero": "params: delta is not in (0, 1)",
    "params-eps-range": "params: eps is not in (0, 1]", "params-eps-zero": "params: eps is not in (0, 1]",
    "params-eps-empty": 'params: eps: "" is not a natural', "params-grid": "params: grid_bits",
    "params-query": "params: query", "params-range": "params: range", "params-range-zero": "params: range",
    "params-eps-other": "public-value: sampling", "params-delta-other": "public-value: sampling",
    "params-grid-other": ("public-value:", "level:"), "params-tile-other": "public-value: check, work",
    "params-template-other": "public-value: check", "params-query-other": "query:", "params-range-low": "level:",
    "params-range-high": "level:", "record-format": "commitment: format", "record-program": "commitment: program",
    "record-partition": "commitment: partition", "record-population-type": "commitment: population",
    "record-law-missing": "commitment: law is missing", "record-roots": "commitment: roots",
    "record-window-keys": "commitment: window", "record-window-public": "commitment: window", "record-kind": "window:",
    "record-context": "window:", "record-extra-item": "public-items:", "record-missing-item": "public-items:",
    "record-level-missing": "level: the registration names no work level",
    "record-level-type": "level: the registration names no work level", "record-level-high": "level:",
    "record-level-huge": "level:", "record-level-step": ("public-value:", "level:"), "record-work": "public-value: work",
    "record-check": "public-value: check", "record-sampling": "public-value: sampling", "record-root-w": "public-value: root_W",
    "record-no-w-root": "public-value: root_W", "record-population": "population:", "record-law": "law:",
    "record-law-kind": "law:", "record-query": "query:", "record-scheme": "scheme:", "answer-keys": "evidence: keys",
    "answer-protocol-type": "evidence: protocol", "answer-verdict": "evidence: verdict",
    "answer-commitment-type": "evidence: commitment", "answer-shape": "evidence: an accepted answer",
    "answer-profile": "evidence: profile", "answer-profile-delta": "evidence: profile",
    "answer-profile-draw": "evidence: profile", "answer-public": "evidence: public", "answer-public-law": "evidence: public",
    "answer-protocol": "outcome: protocol", "answer-other": "outcome: for registration",
    "answer-null": "outcome: for registration null", "answer-reject": "service: the audit's reject",
    "answer-undecided": "service: the audit's undecided", "answer-programs": "profile: programs",
    "answer-programs-two": "profile: programs", "answer-level-query": "profile: its levels",
    "answer-level-size": "profile: its levels", "answer-level-draw": "profile: its levels",
    "answer-level-bernoulli": "profile: its levels", "answer-level-stratified": "profile: its levels",
    "answer-level-name": "profile: its levels", "answer-levels-two": "profile: its levels", "answer-delta": "profile: delta",
    "answer-drawn": "profile: drawn", "answer-draw-k": "draw: subset:", "answer-draw-population": "draw: subset:",
    "answer-units-order": "draw: its", "answer-units-range": "draw: its", "answer-units-count": "draw: its",
    "answer-unsorted-lists": None, "noncanonical-record": "commitment: not canonical JSON",
    "noncanonical-answer": "evidence: not canonical JSON", "float-answer": "evidence: canonical JSON: a non-integer number",
    "utf8-record": "commitment: not UTF-8", "honest": None,
}


def _answer(inst: dict) -> dict:
    return json.loads(vectors.python_answer(inst))


def test_every_mutation_hits_its_refusal():
    """Each mutation, over enough cases to cover it several times, is answered with the refusal it aims at; an honest
    window and one whose lists only need sorting are accepted."""
    assert set(EXPECT) == set(vectors.NAMES) | {"honest"}
    seen: dict[str, int] = {}
    for seed in range(2 * 10 * len(vectors.NAMES) // 9 + 10):
        inst = vectors.instance(seed)
        a, want = _answer(inst), EXPECT[inst["name"]]
        seen[inst["name"]] = seen.get(inst["name"], 0) + 1
        if want is None:
            assert a["verdict"] == "accept" and a["reason"] is None, (seed, a["reason"])
        else:
            assert a["reason"].startswith(want), (seed, inst["name"], a["reason"])
            assert a["verdict"] == ("undecided" if inst["name"] == "answer-undecided" else "reject")
            assert a["public"] is None and a["profile"] is None
    assert set(seen) == set(EXPECT)


def test_an_accepted_window_publishes_exactly_pouw_public():
    for seed in range(0, 200, 10):
        inst = vectors.instance(seed)
        a = _answer(inst)
        case = inst["case"]
        rec = json.loads(bytes.fromhex(case["commitment"]))
        ans = json.loads(bytes.fromhex(case["evidence"]))
        assert a["verdict"] == "accept" and set(a["public"]) == {p[0] for p in Wd.PUBLIC}
        assert a["public"]["verdict"] is True and a["public"]["drawn"] == ans["public"]["draw"]["units"]
        assert {k: a["public"][k] for k in Wd.REGISTERED} == rec["window"]["public"]
        assert a["profile"] == ans["profile"] and a["commitment"] == ans["commitment"]
        k = rec["window"]["public"]["sampling"]["law"]["k"]
        w = Wd.Work.at(rec["window"]["public"]["work"]["level"], case["params"]["grid_bits"])
        tw = case["params"]["check"]["tile_work"]
        assert k == Wd.k_prime(Fraction(case["params"]["eps"]), Fraction(case["params"]["delta"]), w.credited * tw, w.n, tw)
        assert rec["population"] == w.n == a["profile"]["levels"][0]["size"]


def test_the_digest_is_one_stages():
    """The commitment's digest is `registration.digest` of the record, the one the service's receipt and outcome cite."""
    for seed in range(0, 50, 10):
        b = bytes.fromhex(vectors.instance(seed)["case"]["commitment"])
        c = CC.commitment(b)
        assert CC.digest(c).hex() == R.digest(json.loads(b))
    assert CC.ONE_STAGE == R.PROTOCOL and CC.SCHEME == S.SCHEME and CC.FORMATS == (R.FORMAT, R.FORMAT_V2)


def test_ratios_have_one_spelling():
    for s in ("0", "1", "1/2", "3/7", "12"):
        assert str(CC.ratio(s, "x")) == s
    for s, why in (("2/4", "lowest terms"), ("1/1", "denominator 1"), ("1/0", "denominator 0"), ("01", "canonical form"),
                   ("", "canonical form"), ("1/2/3", "not a ratio"), ("-1", "canonical form"), ("1.5", "canonical form")):
        with pytest.raises(CC.Refused, match=why):
            CC.ratio(s, "x")


def test_canonical_json_is_leans():
    assert CC.canonical(b'{"a":[1,null,true],"b":"x"}') == {"a": [1, None, True], "b": "x"}
    for b, why in ((b'{"b":1,"a":2}', "not canonical"), (b'{"a": 1}', "not canonical"), (b'{"a":1.5}', "non-integer"),
                   (b"\xff", "not UTF-8"), (b"{", "not JSON")):
        with pytest.raises(CC.Refused, match=why):
            CC.canonical(b)

