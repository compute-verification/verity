"""PoUW's check as a core protocol (`core_check`), the Python reference beside the Lean `pouw_check`.

`verity/core/protocols/pouw/lean/check/tests/pouw_check_cases.py` generates `serve` cases (honest windows, and one
mutation each aimed at one refusal); that package's `tests/test_difftest.py` holds this module to `pouw-check serve` on
them, on the shared harness. The tests here hold this module to PoUW's rule: each mutation reaches the refusal it aims
at, and an accepted window publishes exactly `window.PUBLIC`."""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from verity.core.protocols.pouw import coins as Co
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.ml.crypto import lms
from verity.ml.randomness import derive

ROOT = Path(__file__).resolve().parents[5]
PKG = ROOT / "verity/core/protocols/pouw/lean/check"
_spec = importlib.util.spec_from_file_location("pouw_rounds_cases", PKG / "tests/pouw_rounds_cases.py")
rounds_cases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rounds_cases)
sys.modules["pouw_rounds_cases"] = rounds_cases
_spec = importlib.util.spec_from_file_location("pouw_pass_cases", PKG / "tests/pouw_pass_cases.py")
passes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(passes)
sys.modules["pouw_pass_cases"] = passes
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
    "params-grid-other": ("public-value:", "level:"),
    "params-tile-other": "params: check.tile_work is not its pinned template's tile work",
    "params-template-other": "public-value: check",
    "params-template-unpinned": "params: check.template is not a tile template the verifier pins",
    "params-query-other": "params: query is not its pinned template's query digest", "params-range-low": "level:",
    "params-coins": "params: check.coins is not live or seeded", "record-coins": "public-value: check",
    "params-range-high": "level:", "record-format": "commitment: format", "record-program": "commitment: program",
    "record-partition": "commitment: partition", "record-population-type": "commitment: population",
    "record-law-missing": "commitment: law is missing", "record-roots": "commitment: roots",
    "record-window-keys": "commitment: window", "record-window-public": "commitment: window", "record-kind": "window:",
    "record-context": "window:", "record-extra-item": "public-items:", "record-missing-item": "public-items:",
    "params-length-stop": "params: length is not", "params-length-empty": "params: length is not",
    "params-length-zero": "params: length is not", "params-length-keys": "params: length is not",
    "params-length-other": "window-form: max-tokens:", "record-legacy": "window-form: no-fixed-length:",
    "record-eos-row": "window-form: eos-keyed-row:", "record-length-mismatch": "window-form: length-mismatch: slot",
    "record-length-count": "window-form: length-mismatch:", "record-max-tokens": "window-form: max-tokens:",
    "record-length-keys": "window-form: length is not", "record-length-stop-type": "window-form: length's stop",
    "record-length-lists": "window-form: length's max_tokens and served",
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
    **{n: f"params: {CC.LIVE_PARAMS}" for n in ("live-missing", "live-null", "live-keys", "live-seeded")},
    "live-other-key": "salt: the challenger's session is not trusted: ",
    "live-incomplete": f"salt: {CC.INCOMPLETE}", "live-unreserved": f"salt: {CC.NO_RESERVE}",
    "live-layout": "salt: the coin commitment's coins are not a Pearl-C salt session's",
    "salt-missing": f"salt: {CC.NO_WINDOW_SALT}", "salt-type": "evidence: public.salt: not an object",
    "salt-seeded": f"salt: {CC.SEEDED_SALT}", "salt-keys": f"salt: {CC.WINDOW_SALT_FORM}",
    "salt-no-w": f"salt: {CC.WINDOW_NO_W}", "salt-two-w": f"salt: {CC.WINDOW_NO_W}",
    "salt-no-x": f"salt: {CC.WINDOW_NO_X}", "salt-values-form": f"salt: {CC.WINDOW_NO_X}",
    "salt-x-after": "salt: the challenger's session is not trusted: the auditor's ledger opens no challenger session",
    "salt-epoch": f"salt: {CC.WINDOW_EPOCH}",
    "salt-other-epoch": "salt: the challenger's session is not trusted: the auditor's ledger opens no challenger session",
    "salt-coins-form": "salt: the salt's coins are malformed: ", "salt-coins": f"salt: {CC.WINDOW_NOT_SIGNED}",
    "salt-opening": "salt: the salt's coin does not open against", "salt-value": f"salt: {CC.WINDOW_SALT}",
    "salt-session": f"salt: {CC.WINDOW_SESSION}", "salt-secret": f"salt: {CC.WINDOW_SESSION}",
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
_ORDER = "the proof's header does not list its rows a0, b0 first, in its circuit's port order"
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
    **{n: "the proof's circuit is not the one the verifier pins" for n in ("circuit-other", "circuit-missing",
                                                                          "circuit-type")},
    **{n: _ROWS for n in ("schemas-missing", "frame-type", "schemas-type", "schemas-swapped", "schemas-one",
                          "schemas-shadowed")},
    "schemas-prefixed": _ORDER, "schemas-between": _ORDER,
    "schemas-unsorted": None, "body-short": "the proof's body holds no A row and B row",
    "a-index": "side a: the opening is of row", "b-index": "side b: the opening is of row",
    **{n: "side a: the proof's committed row is not" for n in ("a-row", "a-path-short", "rows-swapped", "past-side")},
    **{n: "side b: the proof's committed row is not" for n in ("b-row", "b-path")},
    "params-bad": CC.Refused, "commitment-bad": CC.Refused,
    "certified-unsalted": "the window is certified, and its template's unit reads no salt",
    "uncertified-salt": "the window is not certified, so it has no certified salt",
    "salted": None, **{n: "the window is certified, and the verifier holds no 64-byte certified salt" for n in (
        "salted-no-salt", "salted-salt-short")},
    **{n: "the proof's public row is not the window's certified salt" for n in (
        "salted-other-salt", "salted-position", "salted-roots", "salted-row-salt")},
    "salted-uncertified": "the window's template's unit reads a salt, which only a certified window has",
    **{n: "the proof's rows a0, a1, a2, a3, b0," for n in ("salted-schemas", "salted-schemas-missing")},
    "salted-header-order": "the proof's header does not list its rows a0, a1, a2, a3, b0,",
    "salted-body-short": "the proof's body holds no A rows, B rows and public row",
    **{n: _FIELDS for n in ("salted-sides-short", "salted-side-dict")},
    "salted-a-index": "side a: the opening is of row", "salted-b-row": "side b: the proof's committed row is not",
    "salted-pub-swapped": "side b: the proof's committed row is not",
    "salted-circuit": "the proof's circuit is not the one the verifier pins",
}


def _opening(case: dict) -> str | None:
    return CC.opening(CC.params(case["params"]), CC.commitment(bytes.fromhex(case["commitment"])), case["units"],
                      case["unit"], bytes.fromhex(case["pub"]), case["opening"],
                      None if case["salt"] is None else bytes.fromhex(case["salt"]))


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


def _certified_pass(p: int) -> tuple[bytes, str, bytes, bytes, bytes, bytes]:
    """A certified pass of PoUW window `p` (two certifier windows per PoUW window) under a stream and a prover: its
    manifest, the window's context, and the verifier's reveal, record, stream and prover."""
    import hashlib
    from verity.core.protocols.pouw import salt as PS
    from verity.core.protocols.pouw.audit import Epoch, Workload, weights_root
    from verity.core.protocols.pouw.protocol import Matmul, Shape
    roots, work = {2: hashlib.sha512(b"salt-bind weights").digest()}, Workload((Matmul(5, Shape(16, 128, 4), 2),))
    reveal, sid, prover = (hashlib.sha512(b"salt-bind " + x).digest() for x in (b"reveal", b"stream", b"prover"))
    man = {"run": "r-salt-bind", "matmuls": [{"index": 5, "m": 16, "k": 128, "n": 4, "weight": 2}],
           "weights": {"2": {"root": roots[2].hex()}}, "schema": CC.MANIFEST_CERTIFIED,
           "salt": Epoch.certified(work, roots, [2 * p, 2 * p + 1], 2, reveal, sid, prover).salt.hex()}
    raw = json.dumps(man, sort_keys=True).encode()
    ctx = PS.context("served-zk/salt-bind", p, hashlib.sha512(raw).hexdigest())
    return raw, ctx, reveal, PS.pouw_record(weights_root(roots), sid, prover), sid, prover


def test_a_salted_unit_opens_only_under_its_certified_windows_salt():
    """At the opening of record. A certified window's pass passes `pass_epoch` under the certifier's reveal,
    and `certified_salt` is the salt it checked, `salt.pouw_salt(reveal, p, stream, prover)`. A salted unit whose public
    row was formed under that salt opens under it (the honest twin: every rule of the opening met). The same unit with
    its public row formed under a salt the prover chose, which it can form before any certifier salt exists, is refused
    at its public row under the certified salt (and opens only under its own, which no verifier holds); with no salt, or
    in a window the context doesn't certify, the opening refuses before it reads a row."""
    import hashlib
    from verity.core.protocols.pouw import salt as PS
    raw, ctx, reveal, committed, sid, prover = _certified_pass(1)
    assert CC.pass_epoch(None, raw, ctx, reveal=reveal, committed=committed, stream=sid, prover=prover) == (
        "certifier", None)
    salt = CC.certified_salt(raw, ctx, reveal, committed, sid, prover)
    assert salt == PS.pouw_salt(reveal, 1, sid, prover) == bytes.fromhex(json.loads(raw)["salt"])
    own = hashlib.sha512(b"the prover's own salt, chosen before any certifier window").digest()
    for seed in range(3):
        cases = {s: openings.honest(random.Random(f"salt-bind/{seed}"), salted=True, context=ctx, salt=s)
                 for s in (salt, own)}
        assert cases[salt]["opening"] == cases[own]["opening"] and cases[salt]["pub"] != cases[own]["pub"]

        def opened(case, held):
            return CC.opening(CC.params(case["params"]), CC.commitment(vectors._canon(case["record"]).encode()),
                              case["units"], case["unit"], case["pub"], case["opening"], held)

        assert opened(cases[salt], salt) is None
        why = opened(cases[own], salt)
        assert why is not None and why.startswith("the proof's public row is not the window's certified salt")
        assert opened(cases[own], own) is None
        assert opened(cases[salt], None).startswith("the window is certified, and the verifier holds no 64-byte")
        other = {**cases[salt]["params"], "context": "served-zk/salt-bind/uncertified"}
        assert opened({**cases[salt], "params": other}, salt).startswith("the window is not certified")
        assert opened({**cases[salt], "params": other}, None).startswith("the window's template's unit reads a salt")


def test_the_public_row_is_the_compiled_programs_salt_position_and_roots():
    """`core_check.public_row`'s bits are the leaves `Compiled.inputs` fills for one unit: `pouw.salt`, the unit's
    position, then its group's two roots (`seeds.leaves`, `seeds.position_leaves`, Boolean), bit t of the row bit t % 8
    of byte t // 8."""
    import hashlib
    import numpy as np
    from verity.core.protocols.pouw.circuit import seeds as SD
    salt, roots = hashlib.sha512(b"s").digest(), (hashlib.sha512(b"a").digest(), hashlib.sha512(b"b").digest())
    for pos in (0, 1, 5, 1 << 39, (1 << 64) - 1):
        bits = np.unpackbits(np.frombuffer(CC.public_row(salt, pos, roots), np.uint8), bitorder="little").tolist()
        assert len(bits) == CC.PUBLIC_ROW_BITS
        assert bits == SD.leaves(salt, True) + SD.position_leaves(pos, True) + SD.leaves(roots[0] + roots[1], True), pos


def test_a_j10r_public_row_is_then_its_call_and_blocks():
    """A J10-R unit's public row (`core_check.public_row` with ``j10r``) is the leaves `PouwH1Tile_v5`'s public parameters
    take, in order: today's row, then its ``call`` and its two ``blocks``, each a position's leaves
    (`seeds.position_leaves`, `seeds.POS_BITS`)."""
    import hashlib
    import numpy as np
    from verity.core.protocols.pouw.circuit import seeds as SD
    salt, roots = hashlib.sha512(b"s").digest(), (hashlib.sha512(b"a").digest(), hashlib.sha512(b"b").digest())
    for pos, call, blocks in ((5, 0, (0, 64)), (1 << 39, 7, (3, (1 << 33) + 1)), ((1 << 64) - 1, 1 << 63, (1, 2))):
        row = CC.public_row(salt, pos, roots, (call, blocks))
        bits = np.unpackbits(np.frombuffer(row, np.uint8), bitorder="little").tolist()
        assert len(bits) == CC.PUBLIC_ROW_J10R_BITS == CC.PUBLIC_ROW_BITS + 3 * len(SD.position_leaves(0, True))
        assert bits == (SD.leaves(salt, True) + SD.position_leaves(pos, True) + SD.leaves(roots[0] + roots[1], True)
                        + SD.position_leaves(call, True) + SD.position_leaves(blocks[0], True)
                        + SD.position_leaves(blocks[1], True)), pos
        assert row[:CC.PUBLIC_ROW_BITS // 8] == CC.public_row(salt, pos, roots)
        assert len(CC.public_words(salt, pos, roots, (call, blocks))) == CC.PUBLIC_ROW_J10R_BITS // 16 == 112


def test_the_verifiers_j10r_blocks_are_the_tiles():
    """`core_check.j10r_blocks`, the verifier's arithmetic, is `h1tile.j10r_blocks` (`h1tile.Layout`'s row and column
    blocks at call scale) at every slot, K, tile and unit tried, and `core_check.n_digits` is `segment.n_digits`."""
    from verity.core.protocols.pouw.circuit import h1tile as H
    for k in (32, 64, 384, 416, 896, 4096, 14336, 1 << 20):
        assert CC.n_digits(k) == H.SG.n_digits(k // 32), k
        for m, (r, c) in ((4, (4, 8)), (8, (4, 8)), (32, (32, 32)), (64, (32, 32))):
            for i in range(m // r):
                for j in range(3):
                    assert CC.j10r_blocks(m, k, (r, c), i, j) == H.j10r_blocks(m, k, (r, c), i, j), (m, k, r, c, i, j)


def test_the_opening_reads_what_served_zk_writes():
    """`core_check.opening`'s constants are the ones served_zk commits under: one-stage's served domain, and the fp8-p
    A and B rows' schemas."""
    from verity.core.protocols.pouw.circuit import leaves as L
    from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X
    assert CC.SERVED_DOMAIN == R.WINDOW_DOMAIN
    assert CC.OPENED == (("a", L.ROW_SCHEMA["fp8-p", ROLE_X]), ("b", L.ROW_SCHEMA["fp8-p", ROLE_W]))
    case = openings.honest(random.Random(1))
    assert CC.window_domain(CC.params(case["params"]), case["record"]).domain_id.hex() == R.window_domain(
        case["record"]["program"], case["record"]["partition"], Wd.PORT, Wd.ENTRY_SCHEMA,
        case["params"]["check"]["entries"], case["params"]["context"])


def test_every_pinned_tiles_side_ports_sort_in_port_order():
    """C-Flock reads port k's b ‖ c at the body's k-th place, in its circuit's port order, so the opening reads each side
    row's schema by port name and refuses a header whose keys, in key order, don't list the opened side rows first and in
    port order, rather than read the rows in key order (`schemas-between`, `salted-header-order`). It refuses no honest
    header of a pinned template: each one's side ports (Pc8's a0 and b0, the H1 unit's a0..a3 and b0..b7) sort in port
    order, and its other ports (outputs, a salted unit's pub) after them. A salted layout of more than ten rows a side
    would not (a10 sorts before a2): the opening would refuse its honest proofs until it reads them otherwise."""
    for t in CC.TILES.values():
        ports = CC._ports(t.salted)
        assert sorted(ports) == ports and len(ports) == (2 if t.salted is None else t.salted.r + t.salted.c), t


def test_the_pinned_tiles_are_their_templates():
    """Each pinned template the code still builds has as its query one-stage's query digest of its Definition's instance
    query: the served window's (`pc8.side_tile`) at the served K and the test window's, with tile work K, and the hiding
    tile's. The leaf-H3 tile's Definition is retired, and its pin is its recorded windows'. The salted H1 tile unit at
    K = 64 (4 x 8 dot products of K) is pinned salted, its R and C its Definition's. The served circuit is tied to the build
    by benchmarks/pouw's staged-unit test, and the H1 unit's by its pin test (`test_h1_pin.py`)."""
    from verity.core.protocols.pouw.circuit import h1tile as H
    from verity.core.protocols.pouw.circuit import pc8
    from verity.core.protocols.sampled_proofs.one_stage import partition as PT
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    pairs = tuple((i, j) for i in range(4) for j in range(8))
    h1 = bind(H.PouwH1Tile, EXACT=bind(BG.GemmAccumulatorS8, K=64, **BG.GEMM_SM120_S8), R=4, C=8, PAIRS=pairs)
    built = [pc8.side_tile(4096), pc8.side_tile(128), pc8.templates(4096, (1, 1), hidden=True)["tile"], h1]
    for d in built:
        assert CC.TILES[d.id].query == R.query_digest(PT.template_instance_query(d)), d.id
    assert set(CC.TILES) - {d.id for d in built} == {"Pc8TileLeafH3_v1{K=4096,DEV='sm120'}"}
    for k in (4096, 128):
        assert CC.TILES[pc8.side_tile(k).id].tile_work == k
    assert [t for t, x in CC.TILES.items() if x.salted] == [h1.id]
    assert CC.TILES[h1.id].salted == CC.Salted(4, 8) and CC.TILES[h1.id].tile_work == 64 * len(pairs)


def test_canonical_json_is_leans():
    assert CC.canonical(b'{"a":[1,null,true],"b":"x"}') == {"a": [1, None, True], "b": "x"}
    for b, why in ((b'{"b":1,"a":2}', "not canonical"), (b'{"a": 1}', "not canonical"), (b'{"a":1.5}', "non-integer"),
                   (b"\xff", "not UTF-8"), (b"{", "not JSON")):
        with pytest.raises(CC.Refused, match=why):
            CC.canonical(b)


#: the refusal each pass mutation aims at: (code, the reason's start), or the honest pass's ``how`` (a reading both sides
#: share)
PASS_EXPECT = {
    "honest": {"coin", "recorded beacon", "certifier", "live coin"},
    "spaced-hex": {"coin", "recorded beacon", "certifier", "live coin"},
    "no-pass": ("coin", "the window names no pass"), "not-utf8": ("manifest", "the pass's manifest is not a JSON object"),
    "not-json": ("manifest", "the pass's manifest is not a JSON object"),
    "not-object": ("manifest", "the pass's manifest is not a JSON object"),
    "constant": ("manifest", "the pass's manifest is not a JSON object"),
    "recorded-unread": ("coin", f"the pass's manifest is {CC.MANIFEST_RECORDED}, recorded before epoch coins"),
    "schema-other": ("coin", "schema "), "schema-missing": ("manifest", "the pass's manifest names no schema"),
    "matmuls-form": ("manifest", "the pass's matmuls are not"), "matmuls-not-list": ("manifest", "the pass's matmuls are not"),
    "shape": ("coin", "a shape has m, k, n >= 1"), "ids-64-bit": ("coin", "a matmul's index and weight id are 64-bit"),
    "weights-form": ("manifest", "the pass's weights are not"), "weights-not-object": ("manifest", "the pass's weights are not"),
    "no-salt-coin": ("coin", "the pass has no salt coin"),
    "no-verifier-coins": ("coin", "the pass's salt coin opens against the epoch's published coin commitment"),
    "verifier-coins": ("coin", "the verifier's coins are malformed: an epoch is"),
    "opening-form": ("coin", "the pass's coins are malformed: a coin opening is"),
    "pass-coins-form": ("coin", "the pass's coins are malformed: a coin commitment's"),
    "recorded-scheme": ("coin", "the verifier's coins are malformed: a coin commitment's scheme is 'verity/pouw/epoch-coins/v1'"),
    "not-the-epochs": ("coin", "the pass's coin commitment is not the epoch's"),
    "indices": ("coin", "matmul indices are unique"), "dims": ("coin", "every matmul on weight"),
    "weights-set": ("coin", "every weight of the workload is committed, and only those"),
    "coin-epoch": ("coin", "the salt's coin is epoch"), "coin-label": ("coin", "the salt's coin is the epoch's"),
    "coin-size": ("coin", "the salt's coin opening is malformed"),
    "coin-foreign": ("coin", "the salt's coin does not open against the epoch's coin commitment"),
    "coin-tampered": ("coin", "the salt's coin does not open against the epoch's coin commitment"),
    "coin-path": ("coin", "the salt's coin does not open against the epoch's coin commitment"),
    "coin-early": ("coin", "the salt's coin was not opened after what the verifier recorded"),
    "resalted": ("coin", "the pass's salt is not "), "salt-missing": ("coin", "the pass's salt is not "),
    "beacon-empty": ("manifest", "the recorded pass's beacon is not nonempty hex"),
    "no-beacon": ("coin", "the recorded pass has no beacon"),
    "not-named": ("manifest", "the pass is not the one the window's parameters name"),
    "certified-unread": ("coin", f"the pass's manifest is {CC.MANIFEST_CERTIFIED}, salted by the network certifier"),
    "uncertified-pass": ("coin", "schema 'verity-vllm/pouw-pearl-c-run/v"),
    "reveal-size": ("coin", "the verifier's reveal is not a 64-byte salt"),
    "no-window": ("coin", "the window's context names no PoUW window"),
    "other-window": ("coin", "the pass's salt is not the certifier's for its window, stream and prover"),
    "other-reveal": ("coin", "the pass's salt is not the certifier's for its window, stream and prover"),
    "raw-reveal": ("coin", "the pass's salt is not the certifier's for its window, stream and prover"),
    "weights-unrecorded": ("coin", "the verifier holds no record the certifier made before it drew the window's salt"),
    "weights-after-salt": ("coin", "the pass's weights, stream and prover are not those the certifier recorded before it "
                                   "drew the window's salt"),
    "binding-form": ("coin", "the verifier holds no 64-byte stream digest and 64-byte prover id for the window"),
    "stream-other": ("coin", "the pass's weights, stream and prover are not those the certifier recorded"),
    "prover-other": ("coin", "the pass's weights, stream and prover are not those the certifier recorded"),
    "salted-apart": ("coin", "the pass's salt is not the certifier's for its window, stream and prover"),
    "live-unheld": ("coin", CC.LIVE_UNHELD), "live-form": ("coin", CC.LIVE_FORM),
    "live-in-process": ("coin", "the pass's salt coin opens against the epoch's published coin commitment"),
    "live-own-challenger": ("coin", CC.LIVE_NOT_SIGNED), "live-forged": ("coin", "the salt's coin "),
    "live-session": ("coin", CC.LIVE_SESSION),
    "live-other-root": ("coin", "the challenger's session is not trusted: the auditor's ledger opens no challenger "
                                "session for this window"),
    "live-h1": ("coin", "the coin commitment's coins are not a Pearl-C salt session's"),
    "live-incomplete": ("coin", CC.INCOMPLETE), "live-pass-index": ("manifest", CC.PASS_FORM),
    "live-other-epoch": ("coin", "the challenger's session is not trusted: the auditor's ledger opens no challenger "
                         "session for this window"),
}

DRAW_EXPECT = {
    "honest": None, "verifier-coins": ("coin", "the verifier's coins are malformed"),
    "recorded-scheme": ("coin", "the verifier's coins are malformed: a coin commitment's scheme is"),
    "registration": ("draw", "the registration is not a SHA-512 digest"), "stream": ("draw", "the registration is not a SHA-512"),
    "receipt": ("draw", "the receipt is not"), "law": ("draw", "the law is not subset:k over N"),
    "opening-form": ("coin", "the draw's a coin opening is"), "label": ("coin", "the draw's coin is the epoch's 'salt' coin"),
    "epoch": ("coin", "the draw's coin is epoch"),
    "foreign": ("coin", "the draw's coin does not open"), "tampered": ("coin", "the draw's coin does not open"),
    "early": ("coin", "the draw's coin was not opened after what the verifier recorded"),
    "units": ("draw", "the units are not the draw coin's subset:"), "other-key": ("draw", "the units are not the draw coin's"),
    "early-coins-live": ("coin", "the draw's coin does not open"),
    "live-coins": ("coin", "the challenger record is malformed: "),
    "live-not-epoch": ("coin", "the coin commitment's coins are not a live epoch's"),
    "live-bare": ("coin", CC.SIGNED_LIVE),
    "live-key": ("coin", "the verifier's challenger key is malformed: a challenger key is {H, W, public}"),
    "live-no-record": ("coin", "no challenger record: the verifier holds no coin commitment"),
    "live-unsigned": ("coin", "the challenger record's signature does not verify under the verifier's challenger key"),
    "live-own-key": (CC.COIN_KEY, CC.OTHER_KEY), "live-pinned-other": (CC.COIN_KEY, CC.OTHER_KEY),
    "live-no-pin": (CC.COIN_KEY, f"{CC.NO_PINNED_KEY}: a challenger key is {{H, W, public}}"),
    "seeded-pinned": (CC.COIN_KEY, CC.SEEDED_PINNED),
    "live-moved": ("coin", "the challenger record's signature does not verify under the verifier's challenger key"),
    "live-opening-form": ("coin", "the draw's a live coin opening is"),
    "live-label": ("coin", "the draw's coin is the epoch's pouw/salt coin 0, not its pouw/draw coin 0"),
    "live-epoch": ("coin", "the draw's coin is epoch "),
    "live-foreign": ("coin", "the draw's coin does not open against"),
    "live-tampered": ("coin", "the draw's coin does not open against"),
    "live-early": ("coin", "the draw's coin was not opened after what the verifier recorded"),
    "live-units": ("draw", "the units are not the draw coin's subset:"),
}


def test_every_pass_mutation_hits_its_refusal():
    """`core_check.pass_epoch` re-starts an honest pass's epoch (each coin version, and a recorded beacon's) and refuses
    each mutation, over a few cases each, with the refusal it aims at."""
    assert set(PASS_EXPECT) == set(passes.PASS_NAMES) | {"honest"}
    for name, want in PASS_EXPECT.items():
        for seed in range(6):
            how, refused = CC.pass_epoch(*passes.pass_args(passes.pass_instance(random.Random(f"{name}/{seed}"), name)))
            if isinstance(want, set):
                assert refused is None and how in want, (name, seed, refused)
            else:
                assert refused is not None and refused[0] == want[0] and refused[1].startswith(want[1]), (name, seed, refused)


def test_each_honest_pass_is_had_as_it_was_salted():
    """Each kind of honest pass is accepted, and says how its salt was had: by the epoch's coins (either version), a
    recorded replay's beacon, the network certifier's reveal over the weights, stream and prover it recorded before
    it, or a live Pearl-C pass's session of the auditor's challenger, opened on its salt epoch's root."""
    want = {"coin": "coin", "recorded-coin": "coin", "beacon": "recorded beacon", "certified": "certifier",
            "live": "live coin"}
    for kind, how in want.items():
        for seed in range(6):
            case = passes.pass_instance(random.Random(f"{kind}/{seed}"), kind=kind)
            assert {case[k] is not None for k in passes.CERTIFIED} == {kind == "certified"}
            assert CC.pass_epoch(*passes.pass_args(case)) == (how, None), (kind, seed)


def test_every_draw_mutation_hits_its_refusal():
    """`core_check.draw` accepts an honest draw under seeded coins or a live epoch's (a certified window's, held as its
    challenger's signed record with the key it is signed under, which must be the one the window's parameters pin) and
    refuses each mutation with the refusal it aims at: a bare live commitment, a malformed key, no record, a record
    signed under another key than the one its coins name, a tampered signature or a record whose commitment was
    swapped (``coin``); a record under a key of the prover's own, a pin of another key, no pin or a malformed one, and
    seeded coins under a pin (``coin-key``). A live draw's coin is ``pouw/draw`` 0, so a seeded opening under live
    coins is refused, and so is a live opening under seeded coins."""
    assert set(DRAW_EXPECT) == set(passes.DRAW_NAMES) | {"honest"}
    for name, want in DRAW_EXPECT.items():
        for seed in range(6):
            got = CC.draw(*passes.draw_args(passes.draw_instance(random.Random(f"{name}/{seed}"), name)))
            assert (got is None) if want is None else (got is not None and got[0] == want[0] and got[1].startswith(want[1])), \
                (name, seed, got)
    for seed in range(6):
        live, _ = passes.honest_live_draw(random.Random(f"live/{seed}"))
        seeded, _ = passes.honest_draw(random.Random(f"seeded/{seed}"), Co.SCHEME, early=False)
        assert CC.draw(*passes.draw_args(live)) is None and CC.draw(*passes.draw_args(seeded)) is None
        assert CC.draw(*passes.draw_args({**live, "opening": seeded["opening"]}))[1].startswith(
            "the draw's a live coin opening is")
        assert CC.draw(*passes.draw_args({**seeded, "opening": live["opening"]}))[0] == "coin"


def _live_draw(sk: lms.PrivateKey, coins: str, reg: str, stream: str, receipt: dict, n: int, k: int, pinned: dict) -> dict:
    """The draw's check of record's input (`pouw_pass_cases.honest_live_draw`'s layout) for a challenger run on coins
    `coins`, its record signed under `sk` and naming that key, in a window whose parameters pin challenger key `pinned`."""
    ch = Wd.live_challenger(0, 1, random.Random(coins).randbytes)
    message = Ch.record_message(lms.public_key(sk)[:16], "provers-node", ch.commitment)
    record = Ch.ChallengerRecord(ch.commitment, "provers-node", lms.sign_at(sk, 0, message).encode())
    o = ch.open(Wd.DRAW, 0, bytes.fromhex(reg))
    ctx, size = CC.draw_key_context(stream, reg, receipt), 2 * k * CC.width(n) + 64
    return {"coins": {"key": Ch.ChallengerKey(4, 2, lms.public_key(sk)).to_json(), "record": record.to_json()},
            "recorded": False, "opening": o.to_json(), "stream": stream, "registration": reg, "receipt": receipt, "k": k,
            "population": n, "units": CC.stream_subset(k, n, derive(o.coin, CC.DRAW_DOMAIN, ctx).stream("draw", size)),
            "key": pinned}


def test_a_draw_record_the_prover_signed_over_its_own_coins_is_refused():
    """A self-run challenger, refused. A prover that runs a challenger on coins of its own and signs its record under a key
    of its own sees each candidate registration's draw before it registers, and keeps one whose draw misses the unit it
    can't prove (unit 0). The draw's check of record now verifies the record under the challenger key the window's
    parameters pin, not the key its coins name: the prover's record naming its own key is refused (``coin-key``), and
    naming the pinned key it doesn't verify under it (``coin``); with no key pinned, no live record is read
    (``coin-key``). The verifier's own challenger's record, signed under the pinned key, passes on the same draw."""
    verifier = lms.keygen(lms.Params(64, 4, 2), hashlib.sha512(b"the verifier's challenger").digest())
    prover = lms.keygen(lms.Params(64, 4, 2), hashlib.sha512(b"the prover's challenger key").digest())
    pinned = Ch.ChallengerKey(4, 2, lms.public_key(verifier)).to_json()
    stream, n, k, bad = hashlib.sha512(b"stream").hexdigest(), 26, 3, 0
    for i in itertools.count():
        reg = hashlib.sha512(b"candidate %d" % i).hexdigest()
        receipt = {"format": "verity/registration-receipt/v1", "received": str(i), "registration": reg,
                   "verifier": "served-zk/challenger/window"}
        d = _live_draw(prover, "the prover's coins", reg, stream, receipt, n, k, pinned)
        if bad not in d["units"]:
            break
    assert CC.draw(*passes.draw_args(d)) == (CC.COIN_KEY, CC.OTHER_KEY)
    assert CC.draw(*passes.draw_args({**d, "coins": {**d["coins"], "key": pinned}})) == \
        ("coin", "the challenger record's signature does not verify under the verifier's challenger key")
    code, why = CC.draw(*passes.draw_args({**d, "key": None}))
    assert code == CC.COIN_KEY and why.startswith(CC.NO_PINNED_KEY)
    assert CC.draw(*passes.draw_args({**d, "key": d["coins"]["key"]})) is None
    honest = _live_draw(verifier, "the verifier's coins", reg, stream, receipt, n, k, pinned)
    assert CC.draw(*passes.draw_args(honest)) is None


#: the refusal each mutation of the session's pin aims at, `rounds`' and `salt`'s alike
PIN_EXPECT = {
    "unpinned": ("coin", f"ran commit {rounds_cases.UNLOGGED}, which the host's pins log doesn't record as a commit of "
                         "main's history"),
    "branch-pin": ("coin", f"ran commit {rounds_cases.BRANCH}, which the host's pins log records but which is not on "
                           "main's first-parent line in the verifier's copy"),
    "unsealed-pins": ("coin", "seals no pins log (pins: {lines, sha512})"),
    "no-pins": ("coin", f"not trusted: {Ch.NO_PINS}"), "no-main": ("coin", f"not trusted: {Ch.NO_MAIN}"),
    "main-form": ("coin", f"not trusted: the verifier's copy of main's history is malformed: {Ch.MAIN_FORM}"),
}

#: the refusal each rounds mutation aims at: its code and a part of its reason
ROUNDS_EXPECT = {
    "honest": None, "key-form": ("coin", "the verifier's challenger key is malformed: "),
    "seeded-coins": ("coin", "not trusted: the challenger record is malformed: a live coin commitment is"),
    "in-process": ("coin", "not trusted: the challenger record's signature does not verify"),
    "not-live": ("coin", "the coin commitment's "),
    "unsigned": ("coin", "not trusted: no challenger close record"),
    "close-signature": ("coin", "not trusted: the challenger's close record's signature does not verify"),
    "tampered-log": ("coin", "not trusted: the challenger's release log is not the one its close record signs"),
    "unclosed": ("coin", "not trusted: the challenger's session ended without its client closing it: "),
    "no-ledger": ("coin", "not trusted: no auditor's ledger"),
    "no-session": ("coin", "not trusted: the auditor's ledger opens no challenger session for this window"),
    "second-session": ("coin", ": a second open for root "),
    "v1-ledger": ("coin", "under verity/challenger-auditor/v1, which names no prover and seals no refused hellos"),
    "end-form": ("coin", "not trusted: the auditor's ledger's end of session s-1 is malformed: "),
    "record-leaf": ("coin", "not trusted: the challenger's record is signed at leaf 8 over commitment "),
    "hellos": ("coin", "not trusted: the challenger's refused hellos are not the 0 the auditor's ledger sealed"),
    "hellos-form": ("coin", "not trusted: the challenger's refused hellos are a list of their lines"),
    "reused-leaf": ("coin", ", not after the last entry's "),
    "other-session": ("coin", "not trusted: the challenger's session is s-2, not s-1, the one the auditor's ledger opens"),
    "other-key": ("coin", ", not the window's pinned key "),
    "forged-ledger": ("coin", "not trusted: the auditor's ledger is refused: entry 0: its signature doesn't verify under "
                              "the challenger key"),
    "truncated": ("coin", "not trusted: the auditor's ledger has no end for session s-1: "),
    "unsealed": ("coin", "not trusted: the challenger's release log is not the one the auditor's ledger sealed"),
    "rewritten": ("rounds", "the round roots are not the roots the challenger's log shows it received, in order"),
    "roots-form": ("rounds", "the round roots are not"),
    "openings-list": ("rounds", "the round coins are not a list"), "openings-form": ("coin", ": a live coin opening is"),
    "twice": ("coin", "is opened twice: a retry never redraws"), "early": ("coin", "was opened before its root was recorded"),
    "early-root": ("coin", "was opened before its root was recorded"), "never": ("coin", "was never opened"),
    "swapped": ("coin", ": coin is the epoch's pouw/round coin 1, not its pouw/round coin 0"),
    "after": ("coin", ": coin was not opened after what the verifier recorded"),
    "root": ("coin", ": coin was not opened after what the verifier recorded"),
    "foreign-epoch": ("coin", ": coin is epoch "), "foreign": ("coin", ": coin does not open against"),
    "tampered": ("coin", ": coin does not open against"), "malformed": ("coin", ": coin opening is malformed"),
    "too-many": ("coin", "roots recorded, more than the epoch's"),
    "label": ("coin", "pouw/round coin 0: coin is the epoch's pouw/salt coin 0, not its pouw/round coin 0"),
    **PIN_EXPECT,
}


def test_every_rounds_mutation_hits_its_refusal():
    """`core_check.rounds` accepts an honest live window's round coins, read through the challenger's signed session the
    auditor's signed ledger opens and ends for the window, and refuses each mutation (an in-process challenger's coins,
    an unsigned or tampered log, a ledger signed under another key, cut before the session's end or at a reused leaf, a
    second session for the window's root, a v1 ledger, an end naming another record leaf or no hellos, hellos
    it didn't seal, a session the host's pins log doesn't pin or pins off main's first-parent line, an open sealing no
    pins log, no pins log or copy of main, roots the challenger didn't receive, a coin opened twice,
    one opened before its round's root was recorded, a foreign epoch's, …), over a few cases each, with the refusal it
    aims at."""
    assert set(ROUNDS_EXPECT) == set(rounds_cases.ROUNDS_NAMES) | {"honest"}
    for name, want in ROUNDS_EXPECT.items():
        for seed in range(6):
            got = CC.rounds(*rounds_cases.rounds_args(rounds_cases.rounds_instance(random.Random(f"{name}/{seed}"), name)))
            assert (got is None) if want is None else (got is not None and got[0] == want[0] and want[1] in got[1]), \
                (name, seed, got)


#: the refusal each salt mutation aims at: its code and a part of its reason
SALT_EXPECT = {
    "honest": None, "key-form": ("coin", "the verifier's challenger key is malformed: "),
    "no-ledger": ("coin", "not trusted: no auditor's ledger"),
    "no-session": ("coin", "not trusted: the auditor's ledger opens no challenger session for this window"),
    "forged-ledger": ("coin", "not trusted: the auditor's ledger is refused: entry 0: its signature doesn't verify"),
    "in-process": ("coin", "not trusted: the challenger record's signature does not verify"),
    "unsealed": ("coin", "not trusted: the challenger's release log is not the one the auditor's ledger sealed"),
    "other-session": ("coin", ", the one the auditor's ledger opens for this window"),
    "not-live": ("coin", "the coin commitment's "), "no-reserve": ("salt", CC.NO_RESERVE),
    "form": ("salt", CC.SALT_FORM),
    "weights": ("salt", CC.OTHER_WINDOW), "requests": ("salt", CC.OTHER_WINDOW), "session": ("salt", CC.OTHER_WINDOW),
    "weights-root": ("salt", CC.OTHER_WINDOW), "received": ("salt", CC.NOT_RECEIVED),
    "opening-form": ("coin", "the salt's a live coin opening is"),
    "after": ("coin", "the salt's coin was not opened after what the verifier recorded"),
    "label": ("coin", "coin 0, not its pouw/salt coin 0"), "foreign": ("coin", "the salt's coin does not open against"),
    "foreign-epoch": ("coin", "the salt's coin is epoch "), "tampered": ("coin", "the salt's coin does not open against"),
    "malformed": ("coin", "the salt's coin opening is malformed"), **PIN_EXPECT,
}


def test_every_salt_mutation_hits_its_refusal():
    """`core_check.salt` accepts an honest live window's salt coin, read through the challenger's signed session the
    auditor's signed ledger reserves, opens and ends for the window, and refuses each mutation with the refusal it aims
    at: the session's (no ledger, a forged one, an in-process challenger's coins, a log the ledger didn't seal, another
    session's files, a session not pinned to a logged commit on main's first-parent line), a session opened without a
    reserve, the open root under another session's id than the reserve's
    or of another window's weights or requests (so one session salts one window), H1's salt before the open root (the
    session opened on the weights' root, the same in every window), a log showing another root, and a coin that isn't
    the epoch's salt coin opened after it."""
    assert set(SALT_EXPECT) == set(rounds_cases.SALT_NAMES) | {"honest"}
    for name, want in SALT_EXPECT.items():
        for seed in range(6):
            got = CC.salt(*rounds_cases.salt_args(rounds_cases.salt_instance(random.Random(f"{name}/{seed}"), name)))
            assert (got is None) if want is None else (got is not None and got[0] == want[0] and want[1] in got[1]), \
                (name, seed, got)


#: the refusal each live salt mutation aims at: (code, a part of the reason)
LIVE_SALT_EXPECT = {
    "honest": None, "not-object": ("manifest", "the pass's manifest is not a JSON object"),
    "schema": ("coin", f"not {CC.MANIFEST_SCHEMA}"), "no-schema": ("manifest", "the pass's manifest names no schema"),
    "matmuls-form": ("manifest", "the pass's matmuls are not"),
    "weights-set": ("coin", "every weight of the workload is committed, and only those"),
    "call-range": ("manifest", CC.CALL_RANGE), "pass-index": ("manifest", CC.PASS_FORM),
    "key-form": ("coin", "the verifier's challenger key is malformed: "),
    "no-ledger": ("coin", "not trusted: no auditor's ledger"),
    "forged-ledger": ("coin", "not trusted: the auditor's ledger is refused: entry 0: its signature doesn't verify"),
    "in-process": ("coin", "not trusted: the challenger record's signature does not verify"),
    "unsealed": ("coin", "not trusted: the challenger's release log is not the one the auditor's ledger sealed"),
    "other-session": ("coin", ", the one the auditor's ledger opens for this window"),
    "other-root": ("coin", "not trusted: the auditor's ledger opens no challenger session for this window"),
    "on-weights": ("coin", "not trusted: the auditor's ledger opens no challenger session for this window"),
    "other-epoch": ("coin", "not trusted: the auditor's ledger opens no challenger session for this window"),
    "named-epoch": ("coin", "not trusted: the auditor's ledger opens no challenger session for this window"),
    "h1-layout": ("salt", "the coin commitment's coins are not a Pearl-C salt session's: its salt coin alone"),
    "not-salt-only": ("salt", "the coin commitment's "), "no-reserve": ("salt", CC.NO_RESERVE),
    "reserve-layout": ("salt", CC.RESERVE_LAYOUT), "incomplete": ("salt", CC.INCOMPLETE),
    "received": ("salt", CC.LIVE_NOT_RECEIVED), "coins-form": ("coin", "the pass's coins are malformed: "),
    "not-signed": ("coin", CC.LIVE_NOT_SIGNED), "opening-form": ("coin", "the salt's a live coin opening is"),
    "after": ("coin", "the salt's coin was not opened after what the verifier recorded"),
    "index": ("coin", ", not its pouw/salt coin 0"), "label": ("coin", ", not its pouw/salt coin 0"),
    "foreign": ("coin", "the salt's coin does not open against"), "foreign-epoch": ("coin", "the salt's coin is epoch "),
    "tampered": ("coin", "the salt's coin does not open against"),
    "malformed": ("coin", "the salt's coin opening is malformed"), "resalted": ("coin", CC.LIVE_SALT),
    "session": ("coin", CC.LIVE_SESSION), **PIN_EXPECT,
}


def test_every_live_salt_mutation_hits_its_refusal():
    """`core_check.live_salt` (LIVESALT2) accepts a live Pearl-C pass whose salt coin came from a salt-only session the
    auditor's signed ledger reserved, opened on the pass's salt epoch's root R = `epoch_open_root(W, pass_index)` and
    ended complete, read through the session's own files, and refuses each mutation with the refusal it aims at: the
    manifest's form and workload, a call index of 2^32 or more, a `pass_index` that isn't an integer in [0, 2^64), the
    session's (as `salt`'s, a session opened on another root among them: W itself, another epoch's root, or the root of
    an epoch other than the one the manifest names), a layout other than the salt coin alone (an
    H1 layout, its salt alone released, among them), a session opened without a reserve or reserved for another layout,
    an end that isn't complete with the salt coin alone released, a log showing another root or none, coins other than
    the session's, a coin that isn't its salt coin opened after R, another salt, and a session the manifest misnames
    (its root W among them)."""
    assert set(LIVE_SALT_EXPECT) == set(passes.LIVE_SALT_NAMES) | {"honest"}
    for name, want in LIVE_SALT_EXPECT.items():
        for seed in range(6):
            salt, got = CC.live_salt(*passes.live_salt_args(passes.live_salt_instance(random.Random(f"{name}/{seed}"),
                                                                                      name)))
            assert (got is None and len(salt) == 32) if want is None else (
                salt is None and got is not None and got[0] == want[0] and want[1] in got[1]), (name, seed, got)
    d = passes.live_salt_instance(random.Random("a float pass_index"))
    for bad in (0.0, 1.0, 1.5, float(d["manifest"]["pass_index"])):
        m = {**d["manifest"], "pass_index": bad}
        assert CC.live_salt(*passes.live_salt_args({**d, "manifest": m})) == (None, ("manifest", CC.PASS_FORM)), bad


def _live_pass(seed: str, name: str = "honest") -> tuple[str | None, tuple[str, str] | None]:
    return CC.pass_epoch(*passes.pass_args(passes.pass_instance(random.Random(seed), name, kind="live")))


def test_a_live_pearl_c_pass_is_credited_only_from_its_complete_salt_only_session():
    """An honest live Pearl-C manifest is accepted: its salt coin came from the salt-only
    session the auditor's ledger reserved, opened on its salt epoch's root and ended complete, having released that coin
    alone, and the verifier's seeded coins play no part. Refused: a verifier that holds no ledger, the session opened on
    another root (a random one, or H1's window-open root), an H1 session (H1's layout, ended incomplete), an end that
    isn't complete, a forged opening, and an in-process coin (seeded coins the prover ran) presented in place of the
    session's."""
    for seed in range(4):
        assert _live_pass(f"live/{seed}") == ("live coin", None)
        h = passes.honest_pass(random.Random(f"live end/{seed}"), "live")
        sid = h["man"]["session"]["session"]
        named = {e["kind"]: e["body"] for e in map(json.loads, h["case"]["live"]["ledger"]) if e["session"] == sid}
        assert named["reserve"]["families"] == [["pouw/salt", 1]] and named["end"]["complete"] is True
        assert named["end"]["released"] == {"pouw/salt": 1}
        assert _live_pass(f"unheld/{seed}", "live-unheld") == (None, ("coin", CC.LIVE_UNHELD))
        assert _live_pass(f"other/{seed}", "live-other-root")[1][1].endswith(
            "the auditor's ledger opens no challenger session for this window")
        assert _live_pass(f"h1/{seed}", "live-h1")[1][1].startswith(
            "the coin commitment's coins are not a Pearl-C salt session's")
        assert _live_pass(f"incomplete/{seed}", "live-incomplete") == (None, ("coin", CC.INCOMPLETE))
        assert _live_pass(f"forged/{seed}", "live-forged")[1][1].startswith("the salt's coin ")
        assert _live_pass(f"in-process/{seed}", "live-in-process")[1][1].startswith(
            "the pass's salt coin opens against the epoch's published coin commitment")


def test_one_epoch_reads_one_session_and_one_salt():
    """`OnePair` on the Python side: two passes of one salt epoch k over one weights' root W, the second's call indices
    continuing the first's, read the one session the ledger opened on `epoch_open_root(W, k)`, so they get one salt (a
    second open on that root is the ledger's own refusal, `rounds`' ``second-session``). Whether they share a call index
    is the Pearl-C period's to refuse, not `live_salt`'s."""
    for seed in range(4):
        d = passes.live_salt_instance(random.Random(f"one epoch, one session/{seed}"))
        salt, refused = CC.live_salt(*passes.live_salt_args(d))
        assert refused is None and d["manifest"]["pass_index"] < 1 << 64
        assert all(u["index"] < CC.CALL_BOUND for u in d["manifest"]["matmuls"])
        other = json.loads(json.dumps(d["manifest"]))
        n, top = len(other["matmuls"]), max(u["index"] for u in other["matmuls"])
        other.update(run="r-the-next-pass", matmuls=[{**u, "index": i} for i, u in
                                                     enumerate(other["matmuls"], start=min(top + 1, CC.CALL_BOUND - n))])
        assert CC.live_salt(*passes.live_salt_args({**d, "manifest": other})) == (salt, None)


def test_two_epochs_over_one_weights_root_read_two_sessions_and_two_salts():
    """`OneRoot` and `EpochRootBinding` on the Python side: epochs 0 and 1 over one weights' root W are opened on two
    roots, so they read two sessions with two salts, and neither epoch's pass is accepted against the other's session."""
    rng = random.Random("two epochs")
    man = passes._live_manifest(rng)
    h0 = passes.honest_live(rng, json.loads(json.dumps(man)), k=0)
    h1 = passes.honest_live(rng, json.loads(json.dumps(man)), k=1)
    assert h0["w"] == h1["w"] and h0["r"] != h1["r"] and h0["s"].sid != h1["s"].sid
    s0, r0 = CC.live_salt(*passes.live_salt_args({**h0["live"], "manifest": h0["man"]}))
    s1, r1 = CC.live_salt(*passes.live_salt_args({**h1["live"], "manifest": h1["man"]}))
    assert (r0, r1) == (None, None) and s0 != s1
    for live, m in ((h0["live"], h1["man"]), (h1["live"], h0["man"])):
        _, refused = CC.live_salt(*passes.live_salt_args({**live, "manifest": m}))
        assert refused == ("coin", "the challenger's session is not trusted: the auditor's ledger opens no challenger "
                                   "session for this window"), refused


def test_one_session_salts_one_window():
    """Two windows of one model (one weights' root) with their own request tables: the salt opened in the first's
    session passes the first and is refused for the second, and a second session for the first window has another open
    root."""
    from verity.core.protocols.pouw import salt as PS

    d = rounds_cases.salt_instance(random.Random("one session, one window"))
    second = random.Random("the second window's requests").randbytes(64).hex()
    assert CC.salt(*rounds_cases.salt_args(d)) is None
    assert CC.salt(*rounds_cases.salt_args({**d, "requests": second})) == ("salt", CC.OTHER_WINDOW)
    assert PS.window_open_root(bytes.fromhex(d["weights"]), bytes.fromhex(d["requests"]),
                               "s20261020T120000Z-0002") != bytes.fromhex(d["window"]["root"])


def test_the_versioned_coin_rule_is_the_coins_on_the_current_version():
    """On `coins.SCHEME`, `core_check.coin_reason` and `weights_root` are `coins.coin_reason` and `audit.weights_root`;
    the recorded version differs from it in its binding and weights' root."""
    from verity.core.protocols.pouw import coins as Co
    from verity.core.protocols.pouw.audit import weights_root
    rng = random.Random(3)
    for seed in range(8):
        ch = Co.Challenger(seed % 3, rng.randbytes)
        roots = {rng.randrange(99): rng.randbytes(64) for _ in range(3)}
        assert CC.weights_root(Co.SCHEME, roots) == weights_root(roots)
        assert CC.weights_root(CC.COINS_RECORDED, roots) != weights_root(roots)[:32]
        o = ch.open("salt", weights_root(roots))
        c = CC.coins_of(ch.commitment.to_json(), False)
        for label, after, op in (("salt", o.after, o), ("salt", bytes(64), o), ("draw", o.after, o),
                                 ("salt", o.after, Co.CoinOpening(o.epoch + 1, o.label, o.after, o.coin, o.salt, o.path)),
                                 ("salt", o.after, Co.CoinOpening(o.epoch, o.label, o.after, o.coin[:-1] + b"x", o.salt, o.path))):
            assert CC.coin_reason(c, op, label, after) == Co.coin_reason(ch.commitment, op, label, after)
