"""PoUW's check as a core protocol (`core_check`), the Python reference of the Lean port `pouw_check`.

`verity/core/protocols/pouw/lean/check/tests/pouw_check_cases.py` generates `serve` cases (honest windows, and one
mutation each aimed at one refusal); that package's `tests/test_difftest.py` holds `pouw-check serve` to this module on
them, on the shared harness. The tests here hold this module to PoUW's rule: each mutation reaches the refusal it aims
at, and an accepted window publishes exactly `window.PUBLIC`."""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import registration as R

ROOT = Path(__file__).resolve().parents[5]
PKG = ROOT / "verity/core/protocols/pouw/lean/check"
_spec = importlib.util.spec_from_file_location("pouw_check_cases", PKG / "tests/pouw_check_cases.py")
vectors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vectors)
sys.modules["pouw_check_cases"] = vectors
_spec = importlib.util.spec_from_file_location("pouw_opening_cases", PKG / "tests/pouw_opening_cases.py")
openings = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(openings)

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
    "case-commitment-upper": "case: not lowercase hex", "case-evidence-odd": "case: odd-length hex",
    "case-params-missing": "case: the case: params missing", "case-evidence-missing": "case: the case: evidence missing",
    "case-commitment-type": "case: the case.commitment: not a string", "case-not-object": "case: the case: params missing",
    "noncanonical-record": "commitment: not canonical JSON",
    "noncanonical-answer": "evidence: not canonical JSON", "float-answer": "evidence: canonical JSON: a non-integer number",
    "utf8-record": "commitment: not UTF-8", "honest": None,
}


def _instance(seed: int, name: str = "honest") -> dict:
    return vectors.instance(random.Random(seed), name)


def _answer(case) -> dict:
    return json.loads(CC.line(CC.case(case)))


def test_every_mutation_hits_its_refusal():
    """Each mutation, over a few cases each, is answered with the refusal it aims at, and an honest window is accepted."""
    assert set(EXPECT) == set(vectors.NAMES) | {"honest"}
    for name, want in EXPECT.items():
        for seed in range(3):
            a = _answer(_instance(seed, name))
            if want is None:
                assert a["verdict"] == "accept" and a["reason"] is None, (seed, a["reason"])
            else:
                assert a["reason"].startswith(want), (seed, name, a["reason"])
                assert a["verdict"] == ("undecided" if name == "answer-undecided" else "reject")
                assert a["public"] is None and a["profile"] is None


def test_an_accepted_window_publishes_exactly_pouw_public():
    for seed in range(20):
        case = _instance(seed)
        a = _answer(case)
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
    for seed, name in enumerate(["honest", *vectors.NAMES] * 2):
        a = _answer(_instance(seed, name))
        assert CC.answer_of(a)["verdict"] == a["verdict"]
        if a["verdict"] == "accept":
            accepted += 1
            assert a["public"] is not None and CC.integrity_json(CC.integrity(a["profile"])) == a["profile"]
    assert accepted


def test_the_digest_is_one_stages():
    """The commitment's digest is `registration.digest` of the record, the one the service's receipt and outcome cite."""
    for seed in range(5):
        b = bytes.fromhex(_instance(seed)["commitment"])
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


_WINDOW_ENTRY = "is not the registered window's entry"
_HEADER, _COUNT = "the proof's public file has no JSON header", "the proof's header has no instance count"
_ROWS = "the proof's first rows are not an A row and a B row of the registered schemas"
_FIELDS = "the opening's salt is not 192 bytes, or a path"
#: the refusal each opening mutation aims at, by its reason's start or end ("…" before an end); None for an acceptance
OPENING_EXPECT = {
    "honest": None, "undrawn": "…is not drawn", "domain": "the registered window domain is not",
    "context": "the registered window domain is not", "root-hex": "the registered window root is not",
    "root-short": "the registered window root is not", **{n: "the opening names no entry" for n in (
        "entry-missing", "entry-kind", "entry-root", "entry-negative", "entry-huge", "entry-float", "entry-null",
        "opening-list", "index-string", "index-bool", "index-missing")},
    "entry-bool": {None, "…" + _WINDOW_ENTRY}, "salt-upper": None, "salt-spaced": None,
    **{n: _FIELDS for n in ("salt-short", "salt-type", "salt-bad", "window-path-type", "window-path-digest",
                            "side-path-type", "side-path-digest", "side-missing", "side-index-type", "side-index-bool")},
    **{n: "…" + _WINDOW_ENTRY for n in ("window-path-short", "entry-other", "entry-negative-index", "entry-past",
                                        "forged-entry")},
    "out-of-range": "…is not in the entry's range", "kind-none": "…is not in the entry's range",
    **{n: _HEADER for n in ("no-newline", "header-not-json", "header-nan", "header-list", "header-utf8")},
    **{n: _COUNT for n in ("instances-missing", "instances-bool", "instances-float", "instances-string")},
    "instances-two": "…instances, not one tile",
    **{n: _ROWS for n in ("schemas-missing", "frame-type", "schemas-type", "schemas-swapped", "schemas-one",
                          "schemas-prefixed")},
    "schemas-unsorted": None, "body-short": "the proof's body holds no A row and B row",
    "a-index": "side a: the opening is of row", "b-index": "side b: the opening is of row",
    **{n: "side a: the proof's committed row is not" for n in ("a-row", "a-path-short", "rows-swapped", "past-side")},
    **{n: "side b: the proof's committed row is not" for n in ("b-row", "b-path")},
    "params-bad": CC.Refused, "commitment-bad": CC.Refused,
}


def _opening(case: dict) -> str | None:
    return CC.opening(CC.params(case["params"]), CC.commitment(bytes.fromhex(case["commitment"])), case["units"],
                      case["unit"], bytes.fromhex(case["pub"]), case["opening"])


def _hits(reason: str | None, want) -> bool:
    if want is None or reason is None:
        return reason is want
    return reason.endswith(want[1:]) if want.startswith("…") else reason.startswith(want)


def test_every_opening_mutation_hits_its_refusal():
    """`core_check.opening` accepts an honest opening, a call's tile or the filler's, and refuses each mutation, over a
    few cases each, with the reason it aims at (or reads hex and the header as both sides must)."""
    assert set(OPENING_EXPECT) == set(openings.NAMES) | {"honest"}
    for name, want in OPENING_EXPECT.items():
        for seed in range(4):
            case = openings.instance(random.Random(f"{name}/{seed}"), name)
            if want is CC.Refused:
                with pytest.raises(CC.Refused):
                    _opening(case)
                continue
            got = _opening(case)
            assert any(_hits(got, w) for w in (want if isinstance(want, set) else {want})), (name, seed, got)


def test_the_opening_reads_what_served_zk_writes():
    """`core_check.opening`'s constants are the ones served_zk commits under: one-stage's served domain, and the fp8-p
    A and B rows' schemas."""
    from verity.core.protocols.pouw.circuit import leaves as L
    from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X
    assert CC.SERVED_DOMAIN == R.SERVED_DOMAIN
    assert CC.OPENED == (("a", L.ROW_SCHEMA["fp8-p", ROLE_X]), ("b", L.ROW_SCHEMA["fp8-p", ROLE_W]))
    case = openings.honest(random.Random(1))
    assert CC.window_domain(CC.params(case["params"]), case["record"]).domain_id.hex() == R.served_domain(
        case["record"]["program"], case["record"]["partition"], Wd.PORT, Wd.ENTRY_SCHEMA,
        case["params"]["check"]["entries"], case["params"]["context"])


def test_canonical_json_is_leans():
    assert CC.canonical(b'{"a":[1,null,true],"b":"x"}') == {"a": [1, None, True], "b": "x"}
    for b, why in ((b'{"b":1,"a":2}', "not canonical"), (b'{"a": 1}', "not canonical"), (b'{"a":1.5}', "non-integer"),
                   (b"\xff", "not UTF-8"), (b"{", "not JSON")):
        with pytest.raises(CC.Refused, match=why):
            CC.canonical(b)
