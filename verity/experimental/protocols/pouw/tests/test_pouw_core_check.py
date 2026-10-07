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
    "record-law-kind": "law:", "record-query": "query:", "record-scheme": "scheme:",
    "answer-keys": "evidence: answer: not in the form Answer.toJson writes",
    "answer-protocol-type": "evidence: answer.protocol: not a string",
    "answer-verdict": "evidence: answer.verdict: not accept, reject or undecided",
    "answer-commitment-type": "evidence: answer.commitment: not a string",
    "answer-shape": "evidence: answer: an accept has no reason", "answer-profile": "evidence: profile.levels: not a list",
    "answer-profile-delta": 'evidence: profile.delta: "2/4" is not in lowest terms',
    "answer-profile-draw": "evidence: profile.levels[0].draw: not a bernoulli, subset or stratified draw",
    "answer-public": "evidence: public: keys are not draw",
    "answer-public-law": "evidence: public.draw: keys are not law, p, population, units",
    "answer-protocol": "outcome: protocol", "answer-other": "outcome: for registration",
    "answer-null": "outcome: for registration null", "answer-reject": "service: the audit's reject",
    "answer-undecided": "service: the audit's undecided", "answer-programs": "profile: programs",
    "answer-programs-two": "profile: programs", "answer-level-query": "profile: its levels",
    "answer-level-size": "profile: its levels", "answer-level-draw": "profile: its levels",
    "answer-level-bernoulli": "profile: its levels", "answer-level-stratified": "profile: its levels",
    "answer-level-name": "evidence: profile: not in the form toJson writes", "answer-levels-two": "profile: its levels",
    "answer-delta": "profile: delta", "answer-drawn": "profile: drawn", "answer-draw-k": "draw: subset:",
    "answer-draw-population": "draw: subset:", "answer-units-order": "draw: its", "answer-units-range": "draw: its",
    "answer-units-count": "draw: its", "answer-unsorted-lists": "evidence: profile: not in the form toJson writes",
    "answer-commitment-odd": "evidence: odd-length hex", "answer-commitment-upper": "evidence: not lowercase hex",
    "answer-commitment-empty": "outcome: for registration null", "answer-verdict-type": "evidence: answer.verdict: not a string",
    "answer-verdict-missing": "evidence: answer: verdict missing", "answer-reason-type": "evidence: answer.reason: not a string",
    "answer-public-missing": "evidence: answer: public missing",
    "answer-public-null": "evidence: answer: an accept has no reason",
    "answer-reject-public": "evidence: answer: a reject has a reason, and neither",
    "answer-profile-type": "evidence: profile: levels missing", "answer-public-bernoulli": "draw: bernoulli:1/2 over",
    "answer-public-bernoulli-unreduced": "draw: bernoulli:2/4 over", "answer-public-stratified": "draw: stratified:",
    "answer-public-work": "draw: work:", "answer-public-work-bare": "draw: work:",
    "answer-public-law-other": "evidence: public.draw.law: not subset, bernoulli, stratified or work",
    "answer-public-law-type": "evidence: public.draw.law: not a string",
    "answer-public-subset-closure": "evidence: public.draw: keys are not k, law, population, units",
    "answer-public-stratum-keys": "evidence: public.draw.strata[0]: keys are not k, template, units",
    "answer-public-stratum-range": "evidence: public.draw.strata[0].units: not a range [lo, hi]",
    "answer-public-stratum-work": "evidence: public.draw.strata[0].work: not a natural number",
    "answer-public-closure-type": "evidence: public.draw.closure: not a list",
    "answer-public-p-type": "evidence: public.draw.p: not a string",
    "answer-public-units-type": "evidence: public.draw.units: not a list",
    "answer-public-unit-type": "evidence: public.draw.units: not a natural number",
    "answer-public-population-type": "evidence: public.draw.population: not a natural number",
    "answer-drawn-extra": "evidence: profile: not in the form toJson writes",
    "answer-drawn-type": "evidence: profile.drawn.unit: not a natural number",
    "answer-linked-type": "evidence: profile.linked: not a boolean",
    "answer-programs-type": "evidence: profile.programs: not a list",
    "answer-program-type": "evidence: profile.programs: not a string",
    "answer-level-keys": "evidence: profile.levels[0]: not in the form toJson writes",
    "answer-level-missing": "evidence: profile.levels[0]: size missing",
    "answer-level-null-query": "profile: its levels", "answer-level-null-draw": "profile: its levels",
    "answer-level-strata-keys": "evidence: profile.levels[0].draw: not in the form toJson writes",
    "answer-level-draw-two": "evidence: profile.levels[0].draw: not in the form toJson writes",
    "answer-level-bernoulli-lowest": 'evidence: profile.levels[0].draw.bernoulli: "2/4" is not in lowest terms',
    "answer-delta-type": "evidence: profile.delta: not a string", "answer-level-renamed": "profile: its levels",
    "noncanonical-record": "commitment: not canonical JSON",
    "noncanonical-answer": "evidence: not canonical JSON", "float-answer": "evidence: canonical JSON: a non-integer number",
    "utf8-record": "commitment: not UTF-8", "honest": None,
}


def _answer(inst: dict) -> dict:
    return json.loads(vectors.python_answer(inst))


def test_every_mutation_hits_its_refusal():
    """Each mutation, over enough cases to cover it several times, is answered with the refusal it aims at, and an honest
    window is accepted."""
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


def test_every_answer_reads_back():
    """An answer is what `Core.Answer.ofJson` reads back, so an accept's public outputs are never `null`, and an accept's
    profile is what `Core.Integrity.ofJson` reads back."""
    accepted = 0
    for seed in range(0, 600, 3):
        a = _answer(vectors.instance(seed))
        assert CC.answer_of(a)["verdict"] == a["verdict"]
        if a["verdict"] == "accept":
            accepted += 1
            assert a["public"] is not None and CC.integrity_json(CC.integrity(a["profile"])) == a["profile"]
    assert accepted


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





def test_the_data_is_what_the_lean_check_pins(tmp_path):
    """Every case and this module's answer to it, byte for byte as the Lean package's record pins them."""
    run = json.loads((PKG / "lean-audit.json").read_text())["runs"]["PouwCheck.DifftestMain"]
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == run["data"]
    assert run["args"] == ["{difftest/cases.jsonl}", "{difftest/python.jsonl}", str(vectors.COUNT)]
