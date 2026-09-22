"""Coverage of every selected equation (PROTOCOL row R9) -- on top of A2's expected-position module.

Division of labour (integrator note, 2026-09-10 18:32Z):

* ``veritor.interface.expected`` (A2, mainline ``b47e0d85``) derives, from the fixed circuit / partition /
  sample, **what one selected unit's obligation must bind** -- an ``ExpectedObligation`` of ordered
  ``ExpectedValue(tree, leaf, offset)`` incoming/outgoing values, the subcircuit digest and the prescribed
  Input words -- and ``check_expected_positions(obj, expectations, trees=, rank_of=)`` fails closed on one
  typed object's roots, ranks, layouts and unused parts.  Per-kind derivations there:
  ``gemm_coordinate_expectation``, ``input_equation_expectation``; the locator is the record description
  (``rec_p1_obligations.ProgramLocator`` + ``Description``), **never** ``provenance``.
* This module does the **per-VU accounting across all accepted proofs**: ``expected_statements`` turns
  ``{V in Q}`` into digests of A2-shaped expectations (one per VU, refusing anything it cannot derive);
  ``check_coverage`` matches every obligation of every accepted proof by digest -- covered exactly once /
  uncovered / duplicate / unexpected -- and, when the committed trees and the rank oracle are given, runs
  A2's position check per proof on the expectations that proof claims to meet (a proof failing it covers
  nothing: fail closed).

Rules made mechanical: the number of proofs proves nothing (coverage is by digest of the *bound* object's
positions, never by ``selected_vus`` / ``provenance.units``); a duplicate never hides an omission (each
expected statement is counted separately); batching cannot change the partition (an obligation over any
other boundary has another digest); an empty ``Q`` is legal and covered by zero proofs; a VU that cannot be
located is a refusal (:class:`CoverageError`), never a silently uncovered entry.

Precondition for ``check_coverage`` (the caller's -- A3's ``verifier_v3``): the SP1 proof verified under an
approved (ELF, vk) and ``check_binding`` bound its public values to the typed object.  Only then do the
object's obligations describe what was proved.

Digest: ``statement_digest`` = ``tagged_sha256("veritor/coverage_v1/statement", canonical_json({family,
subcircuit, incoming, outgoing, pins}))`` where a value is ``[tree, leaf, *offset]`` (A2's ``ExpectedValue.key``)
and ``pins`` is A2's per-outgoing prescribed word (hex) or null.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Position",
    "ExpectedStatement",
    "ObligationView",
    "AcceptedProof",
    "CoverageReport",
    "CoverageError",
    "Circuit",
    "Partition",
    "statement_digest",
    "expected_statements",
    "check_coverage",
    "obligations_of_typed_object",
    "StatementView",
    "is_statement_expectation",
    "statement_expectation_digest",
    "statement_proof",
]

STATEMENT_TAG = "veritor/coverage_v1/statement"

#: One committed word as A2 spells it: ``(tree, leaf canonical id, *offset)`` -- ``ExpectedValue.key`` flattened.
Position = tuple[str | int, ...]


class CoverageError(ValueError):
    """The verifier cannot derive the expected statement of a selected VU (a refusal, not a skip)."""


@runtime_checkable
class Partition(Protocol):
    """The fixed partition over the actual record (A2's locator: ``ProgramLocator`` + ``Description``).

    ``locate(vu_id)`` returns an A2 ``ExpectedObligation`` (or any object with ``incoming``/``outgoing``
    sequences of values having ``tree``/``leaf``/``offset``, an optional ``subcircuit`` digest and ``pins``)."""

    def locate(self, vu_id: str) -> Any:
        ...


@runtime_checkable
class Circuit(Protocol):
    """The fixed circuit: the gate-set family and kind program (subcircuit digest) of a located VU.

    ``kind_program(vu_id, expectation) -> (family_id, subcircuit_digest_hex)``.  When the expectation already
    carries a subcircuit digest the two must agree (refusal otherwise)."""

    def kind_program(self, vu_id: str, expectation: Any) -> tuple[str, str]:
        ...


def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _tagged_sha256(tag: str, payload: bytes) -> str:
    h = hashlib.sha256()
    h.update(len(tag).to_bytes(4, "big"))
    h.update(tag.encode("utf-8"))
    h.update(payload)
    return h.hexdigest()


def _pos(p: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for x in p:
        if isinstance(x, bool) or not isinstance(x, (str, int)):
            raise CoverageError(f"position component {x!r} is not a str or int")
        out.append(x)
    return out


def _value_key(v: Any) -> Position:
    """A2 ``ExpectedValue`` (``tree``/``leaf``/``offset``) or an already-flat tuple -> ``(tree, leaf, *offset)``."""

    if isinstance(v, tuple):
        return tuple(_pos(v))
    return (str(v.tree), str(v.leaf), *[int(i) for i in v.offset])


def statement_digest(
    family: str, subcircuit: str, incoming: Sequence[Any], outgoing: Sequence[Any], pins: Sequence[str | None] | None = None
) -> str:
    """The digest of one expected/proved statement: the relation (family + subcircuit) over its committed
    positions and prescribed words.  ``pins`` defaults to no pin per outgoing value."""

    outs = [_value_key(p) for p in outgoing]
    pin_list = [None] * len(outs) if pins is None else [None if p is None else str(p).lower() for p in pins]
    if len(pin_list) != len(outs):
        raise CoverageError(f"{len(pin_list)} pins for {len(outs)} outgoing values")
    body = {
        "family": str(family),
        "subcircuit": str(subcircuit).lower(),
        "incoming": [list(_value_key(p)) for p in incoming],
        "outgoing": [list(o) for o in outs],
        "pins": pin_list,
    }
    return _tagged_sha256(STATEMENT_TAG, _canon(body))


@dataclass(frozen=True)
class ExpectedStatement:
    vu_id: str
    family: str
    subcircuit: str
    incoming: tuple[Position, ...]
    outgoing: tuple[Position, ...]
    pins: tuple[str | None, ...]
    digest: str
    expectation: Any = field(compare=False, repr=False, default=None)  # A2's ExpectedObligation, for check_expected_positions
    label: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "vu_id": self.vu_id, "family": self.family, "subcircuit": self.subcircuit,
            "incoming": [list(p) for p in self.incoming], "outgoing": [list(p) for p in self.outgoing],
            "pins": list(self.pins), "digest": self.digest, "label": self.label,
        }


@dataclass(frozen=True)
class ObligationView:
    """One obligation of an accepted proof, as the bound typed object states it (never as the prover labels it)."""

    family: str
    subcircuit: str
    incoming: tuple[Position, ...]
    outgoing: tuple[Position, ...]
    pins: tuple[str | None, ...] = ()

    @property
    def digest(self) -> str:
        pins = self.pins if self.pins else None
        return statement_digest(self.family, self.subcircuit, self.incoming, self.outgoing, pins)


@dataclass(frozen=True)
class StatementView(ObligationView):
    """[E7, interface 6] The one obligation of a proof whose family binds a whole STATEMENT rather than typed positions (E5's
    ``veritor.select-argmax@1``: the guest's public values are ``sha256(statement) || verdict``).  Its digest is that sha256 --
    matched against :func:`statement_expectation_digest` of the VU's expectation, never against a prover label."""

    statement_sha256: str = ""

    @property
    def digest(self) -> str:  # type: ignore[override]
        return self.statement_sha256


@dataclass(frozen=True)
class AcceptedProof:
    """An accepted proof: its id, the obligations of its bound typed object, and (optionally) the object itself
    for A2's position check.  Build ``obligations`` with :func:`obligations_of_typed_object`; a statement-bound proof
    (:func:`statement_proof`) carries its own ``position_checker`` instead of a typed object."""

    proof_id: str
    obligations: tuple[ObligationView, ...]
    typed_object: Any = field(default=None, compare=False, repr=False)
    #: ``(expectations) -> problems``: the family's own check that the proven statement is the one expected for the matched VUs
    #: (E5's ``ck_select.check_expected``); when set, it replaces A2's ``check_expected_positions`` for this proof
    position_checker: Callable[[Sequence[Any]], Sequence[str]] | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class CoverageReport:
    """Per-VU verdicts.  ``ok`` iff every expected statement is covered exactly once by a proof whose positions
    check passed (when run) and nothing unexpected was proved."""

    covered: dict[str, str]                       # vu_id -> proof_id (exactly one)
    uncovered: tuple[str, ...]                    # vu_ids with no covering obligation
    duplicate: dict[str, tuple[str, ...]]         # vu_id -> proof_ids of every covering obligation (> 1)
    unexpected: tuple[tuple[str, str], ...]       # (proof_id, obligation digest) matching no expected statement
    position_problems: dict[str, tuple[str, ...]]  # proof_id -> A2 check_expected_positions problems (proof voided)
    expected_count: int
    proof_count: int
    obligation_count: int
    strict_unexpected: bool = True
    positions_checked: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)
    #: [interface 6, E7] sampled VUs that are prescribed inputs (not proof-covered, not uncovered): ``per_vu`` says ``input`` for them
    inputs: tuple[str, ...] = field(default_factory=tuple)
    #: [interface 6, E7] proof objects listed separately: proof_id -> the VU identities it covers (after the position check)
    proof_chunks: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.uncovered and not self.duplicate and not self.position_problems and (not self.unexpected or not self.strict_unexpected)

    def per_vu(self) -> dict[str, str]:
        """``vu_index -> covered | uncovered | duplicate | input`` (interface 6: coverage by VU identity; proof objects are NOT counted here)."""
        out = {v: "covered" for v in self.covered}
        out.update({v: "uncovered" for v in self.uncovered})
        out.update({v: "duplicate" for v in self.duplicate})
        out.update({v: "input" for v in self.inputs if v not in out})
        return out

    def counts(self) -> dict[str, int]:
        """The three numbers reported separately (never "|Q| = proofs + input equations"): sampled VUs, covered VUs, proof objects."""
        per = self.per_vu()
        return {"sampled_vus": len(per), "covered_vus": sum(1 for s in per.values() if s == "covered"), "input_vus": sum(1 for s in per.values() if s == "input"),
                "uncovered_vus": sum(1 for s in per.values() if s in ("uncovered", "duplicate")), "proof_objects": self.proof_count}

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "expected": self.expected_count, "proofs": self.proof_count, "obligations": self.obligation_count,
            "covered": len(self.covered), "uncovered": list(self.uncovered), "duplicate": {k: list(v) for k, v in self.duplicate.items()},
            "unexpected": [list(u) for u in self.unexpected], "position_problems": {k: list(v) for k, v in self.position_problems.items()},
            "positions_checked": self.positions_checked, "strict_unexpected": self.strict_unexpected, "notes": list(self.notes),
            "per_vu": self.per_vu(), "counts": self.counts(),
            "proof_chunks": {"chunks": [{"proof_id": p, "covers_vu_ids": list(v)} for p, v in self.proof_chunks.items()], "count": self.proof_count},
        }


def expected_statements(circuit: Circuit, partition: Partition, Q: Iterable[Hashable], *,
                        on_unlocated: Callable[[str, BaseException], None] | None = None) -> dict[str, ExpectedStatement]:
    """The expected statement of every ``V in Q``; keyed by ``vu_id``.

    ``partition.locate(vu_id)`` gives A2's ``ExpectedObligation`` for the VU (its committed positions, from the
    record description -- never from provenance); ``circuit.kind_program(vu_id, expectation)`` the family and
    subcircuit digest.  Refuses (``CoverageError``) a repeated VU, a VU the partition cannot locate, an
    expectation without an outgoing value, a subcircuit disagreement between partition and circuit, or two VUs
    with the same expected statement (the partition would not be one).

    ``on_unlocated`` [E9, D48.2 2026-09-12 -- E10's Qwen q2: ONE VU's ``LITERAL_UNPINNED`` voided all 707 expectations]: when given, a VU
    the partition cannot locate is NOT a whole-Q refusal: ``on_unlocated(vu_id, error)`` is called, the VU is left OUT of the map (the
    caller records it as ``unlocated`` BY NAME and refuses the record -- an unaccounted sampled VU is never accepted) and every other
    VU's expectation is still derived and accounted.  The structural refusals (a repeated VU, a digest collision: the partition is not one)
    stay whole-Q refusals.  Without it the behaviour is unchanged (the first locate failure raises)."""

    out: dict[str, ExpectedStatement] = {}
    seen_digests: dict[str, str] = {}
    for raw in Q:
        vu_id = str(raw)
        if vu_id in out:
            raise CoverageError(f"VU {vu_id} appears twice in Q")
        try:
            e = partition.locate(vu_id)
        except CoverageError as error:
            if on_unlocated is None:
                raise
            on_unlocated(vu_id, error)
            continue
        except Exception as error:  # noqa: BLE001 -- any locator failure is a refusal with the cause attached
            if on_unlocated is not None:
                on_unlocated(vu_id, error)
                continue
            raise CoverageError(f"VU {vu_id}: the partition cannot locate it: {error}") from error
        if is_statement_expectation(e):
            family, subcircuit = circuit.kind_program(vu_id, e)
            incoming, outgoing = _statement_positions(e, vu_id)
            digest = statement_expectation_digest(e)
            if digest in seen_digests:
                raise CoverageError(f"VU {vu_id} and VU {seen_digests[digest]} have the same expected statement {digest[:16]}: not a partition")
            seen_digests[digest] = vu_id
            out[vu_id] = ExpectedStatement(vu_id, str(family), str(subcircuit or "").lower(), incoming, outgoing, (None,) * len(outgoing), digest, e,
                                           str(getattr(e, "label", "") or ""))
            continue
        outgoing = tuple(_value_key(v) for v in e.outgoing)
        if not outgoing:
            raise CoverageError(f"VU {vu_id}: no outgoing value -- nothing identifies the relation")
        incoming = tuple(_value_key(v) for v in e.incoming)
        family, subcircuit = circuit.kind_program(vu_id, e)
        declared = getattr(e, "subcircuit", None)
        if declared is not None and str(declared).lower() != str(subcircuit).lower():
            raise CoverageError(f"VU {vu_id}: the partition expects subcircuit {str(declared)[:12]}, the circuit gives {str(subcircuit)[:12]}")
        pins_raw = tuple(getattr(e, "pins", ()) or ())
        pins: tuple[str | None, ...] = tuple(None if p is None else str(p).lower() for p in pins_raw) if pins_raw else (None,) * len(outgoing)
        if len(pins) != len(outgoing):
            raise CoverageError(f"VU {vu_id}: {len(pins)} pins for {len(outgoing)} outgoing values")
        digest = statement_digest(family, subcircuit, incoming, outgoing, pins)
        if digest in seen_digests:
            raise CoverageError(f"VU {vu_id} and VU {seen_digests[digest]} have the same expected statement {digest[:16]}: not a partition")
        seen_digests[digest] = vu_id
        out[vu_id] = ExpectedStatement(vu_id, str(family), str(subcircuit).lower(), incoming, outgoing, pins, digest, e, str(getattr(e, "label", "") or ""))
    return out


def check_coverage(
    expected: Mapping[str, ExpectedStatement],
    proofs: Iterable[AcceptedProof],
    *,
    strict_unexpected: bool = True,
    trees: Mapping[str, Any] | None = None,
    rank_of: Callable[[str, str], int | None] | None = None,
    position_check: Callable[..., Any] | None = None,
    inputs: Iterable[Hashable] = (),
) -> CoverageReport:
    """Match every obligation of every accepted proof against the expected statements by digest.

    Per VU: ``covered`` (exactly one matching obligation, in a proof whose position check passed), ``uncovered``
    (none), ``duplicate`` (more than one, within one proof or across proofs).  Obligations matching no expected
    statement are ``unexpected`` (they fail the check under ``strict_unexpected``, the default).

    With ``trees`` (A2 ``CommittedTree`` by tree id) and ``rank_of`` (A2 ``rank_oracle``), every proof carrying a
    ``typed_object`` is also run through ``veritor.interface.expected.check_expected_positions`` on the
    expectations its obligations matched (roots, ranks, layouts, unused parts, pins); any problem voids that
    proof's coverage (its VUs become uncovered) and is reported under ``position_problems``.  A proof without a
    ``typed_object`` is a problem in that mode (fail closed).  ``position_check`` overrides the checker (tests)."""

    by_digest: dict[str, str] = {e.digest: vu for vu, e in expected.items()}
    if len(by_digest) != len(expected):
        raise CoverageError("expected statements are not digest-unique")
    check_positions = trees is not None or rank_of is not None
    if check_positions and (trees is None or rank_of is None):
        raise CoverageError("the position check needs both `trees` and `rank_of`")
    if check_positions and position_check is None:
        from veritor.interface.expected import (
            check_expected_positions as position_check,  # type: ignore[no-redef]
        )

    hits: dict[str, list[str]] = {vu: [] for vu in expected}
    unexpected: list[tuple[str, str]] = []
    position_problems: dict[str, tuple[str, ...]] = {}
    proof_chunks: dict[str, tuple[str, ...]] = {}
    n_proofs = 0
    n_obl = 0
    seen_proofs: Counter[str] = Counter()
    input_ids = tuple(str(v) for v in inputs)
    if set(input_ids) & set(expected):
        raise CoverageError(f"VUs both expected (proof-covered) and input: {sorted(set(input_ids) & set(expected))[:5]}")
    for proof in proofs:
        n_proofs += 1
        seen_proofs[proof.proof_id] += 1
        matched: list[str] = []
        for ob in proof.obligations:
            n_obl += 1
            vu = by_digest.get(ob.digest)
            if vu is None:
                unexpected.append((proof.proof_id, ob.digest))
            else:
                matched.append(vu)
        if check_positions:
            if proof.typed_object is None and proof.position_checker is None:
                position_problems[proof.proof_id] = ("no typed object to check positions against",)
                continue
            expectations = [expected[vu].expectation for vu in dict.fromkeys(matched)]
            if any(x is None for x in expectations):
                position_problems[proof.proof_id] = ("an expected statement carries no A2 expectation object",)
                continue
            try:
                if proof.position_checker is not None:  # a statement-bound proof: the family's own check of statement vs expectation
                    problems = tuple(str(p) for p in proof.position_checker(expectations))
                else:
                    report = position_check(proof.typed_object, expectations, trees=trees, rank_of=rank_of)  # type: ignore[misc]
                    problems = tuple(str(p) for p in getattr(report, "problems", ()))
            except Exception as error:  # noqa: BLE001 -- an invalid object has no positions: fail closed
                problems = (f"position check raised: {error}",)
            if problems:
                position_problems[proof.proof_id] = problems
                proof_chunks[proof.proof_id] = ()
                continue  # the proof covers nothing
        for vu in matched:
            hits[vu].append(proof.proof_id)
        proof_chunks[proof.proof_id] = tuple(proof_chunks.get(proof.proof_id, ())) + tuple(dict.fromkeys(matched))
    notes: list[str] = []
    repeated = [p for p, n in seen_proofs.items() if n > 1]
    if repeated:
        notes.append(f"proof ids given more than once: {sorted(repeated)} (each occurrence counted; a re-submitted proof duplicates its VUs)")
    covered = {vu: ps[0] for vu, ps in hits.items() if len(ps) == 1}
    uncovered = tuple(vu for vu, ps in hits.items() if not ps)
    duplicate = {vu: tuple(ps) for vu, ps in hits.items() if len(ps) > 1}
    return CoverageReport(covered, uncovered, duplicate, tuple(unexpected), position_problems, len(expected), n_proofs, n_obl,
                          strict_unexpected, check_positions, tuple(notes), input_ids, proof_chunks)


# --- adapter: typed-obligation/v0 objects ---------------------------------------------------------------------------------


def obligations_of_typed_object(obj: Any) -> tuple[ObligationView, ...]:
    """The :class:`ObligationView` rows of a bound ``veritor.interface.TypedObligationSet``.

    A value is ``(leaf.root, leaf.id, *reference.offset)`` -- A2's ``ExpectedValue.key`` (``TypedObligationSet.to_json``,
    ``src/veritor/interface/typed_obligation.py:634-665``); the family is ``family.id``; the subcircuit digest is
    the obligation's; the pin per outgoing value is the prescribed word of an ``Input`` gate at that outgoing
    slot (``None`` otherwise), as A2's ``check_expected_positions`` reads it.  Accepts the dataclass or its
    ``to_json()`` dict.  Only what the object *declares* is read -- never provenance or any prover label."""

    doc = obj.to_json() if hasattr(obj, "to_json") else obj
    family = doc["family"]["id"] if isinstance(doc["family"], dict) else str(doc["family"])
    leaves = {lf["id"]: lf for lf in doc["leaves"]}
    refs = {r["ref"]: r for r in doc["references"]}
    subcircuits = {s["digest"]: s for s in doc["subcircuits"]}

    def position(ref: str) -> Position:
        r = refs.get(ref)
        if r is None:
            raise CoverageError(f"reference {ref!r} is not declared by the object")
        leaf = leaves.get(r["leaf"])
        if leaf is None:
            raise CoverageError(f"reference {ref!r} names undeclared leaf {r['leaf']!r}")
        return (str(leaf["root"]), str(leaf["id"]), *[int(k) for k in r["offset"]])

    views = []
    for ob in doc["obligations"]:
        boundary = ob["boundary"]
        s = subcircuits.get(ob["subcircuit"])
        if s is None:
            raise CoverageError(f"obligation names undeclared subcircuit {ob['subcircuit']!r}")
        pins: list[str | None] = []
        for m in range(len(boundary["out"])):
            gate = s["gates"][s["out"][m]] if m < len(s["out"]) else None
            value = gate.get("value") if isinstance(gate, dict) and gate.get("op") == "Input" else None  # typed_obligation.INPUT
            pins.append(None if value is None else str(value).lower())
        views.append(ObligationView(str(family), str(ob["subcircuit"]).lower(),
                                    tuple(position(x) for x in boundary["in"]),
                                    tuple(position(x) for x in boundary["out"]), tuple(pins)))
    return tuple(views)


# --- adapter: statement-bound families (E5's veritor.select-argmax@1) -----------------------------------------------------
# A builder whose ``expectation()`` is the guest's STATEMENT (``encode() -> bytes``), not an A2 ``ExpectedObligation``: the guest
# commits to ``sha256(statement) || verdict`` and the verifier checks the statement bytes against what it derived.  One VU per proof.


def is_statement_expectation(e: Any) -> bool:
    """An expectation that is a whole statement (``encode() -> bytes``: the family's statement object, or ``builders.StatementExpectation``
    which also spells A2 positions) rather than typed positions alone (an A2 ``ExpectedObligation`` has no ``encode``)."""

    return callable(getattr(e, "encode", None))


def statement_expectation_digest(e: Any) -> str:
    """``sha256(statement bytes)`` -- what the guest's public values carry for an accepted statement."""

    return hashlib.sha256(bytes(e.encode())).hexdigest()


def _statement_positions(e: Any, vu_id: str) -> tuple[tuple[Position, ...], tuple[Position, ...]]:
    """The committed words a statement binds, as ``(tree, leaf id, rank)`` positions.  A statement may spell them itself
    (``incoming`` / ``outgoing`` of ``(tree, leaf, *offset)`` tuples); E5's ``SelectStatement`` names one logits leaf and one select
    leaf of one RU tree."""

    if hasattr(e, "incoming") and hasattr(e, "outgoing"):
        return tuple(_value_key(v) for v in e.incoming), tuple(_value_key(v) for v in e.outgoing)
    if all(hasattr(e, a) for a in ("tree", "logits_id", "logits_rank", "select_id", "select_rank")):
        tree = str(e.tree)
        return ((tree, str(e.logits_id), int(e.logits_rank)),), ((tree, str(e.select_id), int(e.select_rank)),)
    raise CoverageError(f"VU {vu_id}: statement of type {type(e).__name__} names no committed positions")


def statement_proof(proof_id: str, family: str, statement_bytes: bytes, public_values: bytes, *, expectation_of_statement: Callable[[bytes], Any],
                    check_expected: Callable[[bytes, Any], Sequence[str]]) -> AcceptedProof:
    """An :class:`AcceptedProof` for a statement-bound family from the proof's verified public values.

    Refuses (``CoverageError``) public values that are not ``sha256(statement) || 0x01`` -- the verdict byte MUST be 1 (a proof of a
    rejecting statement binds nothing; gate G4) and the digest MUST be of these statement bytes (gates G2/G5).  ``expectation_of_statement``
    decodes the bytes to the family's statement object (its positions become the obligation view); ``check_expected(statement_bytes,
    expected)`` is the family's own verifier check (E5's ``ck_select.check_expected``), run as this proof's ``position_checker`` on the
    expectations its digest matched -- exactly one, since a statement is one VU."""

    pv = bytes(public_values)
    h = hashlib.sha256(bytes(statement_bytes)).hexdigest()
    if len(pv) != 33:
        raise CoverageError(f"{proof_id}: public values are {len(pv)} bytes, not sha256(statement) || verdict (33)")
    if pv[:32].hex() != h:
        raise CoverageError(f"{proof_id}: public digest {pv[:32].hex()[:16]} != sha256 of the presented statement {h[:16]}")
    if pv[32] != 1:
        raise CoverageError(f"{proof_id}: the guest's verdict byte is {pv[32]}, not 1 -- a proof of a rejecting statement covers nothing")
    st = expectation_of_statement(bytes(statement_bytes))
    incoming, outgoing = _statement_positions(st, proof_id)
    view = StatementView(str(family), "", incoming, outgoing, (), h)

    def checker(expectations: Sequence[Any]) -> list[str]:
        if len(expectations) != 1:
            return [f"a statement-bound proof covers one VU, its digest matched {len(expectations)}"]
        e = expectations[0]
        return [str(p) for p in check_expected(bytes(statement_bytes), getattr(e, "statement", e))]  # builders.StatementExpectation or the bare statement

    return AcceptedProof(proof_id, (view,), None, checker)
