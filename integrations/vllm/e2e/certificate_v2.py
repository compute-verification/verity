"""certificate_v2 — the Bound certificate under the governing contract (docs/PROTOCOL.md, 2026-09-10).

What it does
------------
Builds one certificate JSON (+ a text rendering) from *contract-semantics* inputs:

* the population of verification units (VUs) **without input gates** — A1's population when it lands, or a
  legacy kind table with the Input kinds subtracted (labelled ``legacy-minus-inputs``);
* the replay-unit (RU) grouping (banks: copies, VUs per copy, exit-cut width);
* the configured exemption budget ``B_ex`` **as an RU count**, scoped to a declared audit/epoch (R16);
* the law ``q, s, delta`` (Fractions);
* the actual output domain (declared cap ``W`` and, if the vocabulary is given, the exact-domain upper bound);
* the sampled ``|J|, |F|, |Q|`` (recorded, not used by the Bound);
* the cost classes of the contract's cost list (each MEASURED with provenance or ``UNAVAILABLE``);

and calls ``veritor.analysis.certified.certified_bound`` — the theory is not re-implemented here.  The output states
``U`` (bits and TB), ``W``, ``U == W`` explicitly when capped, the winning ``(t, lambda, z)``, every input with its
provenance, missing costs as ``UNAVAILABLE``, and the assumptions (S1–S3 assumed, S4 unestablished).

What it refuses (clear errors, never a silent reinterpretation)
--------------------------------------------------------------
* ``B_ex`` given as a VU count, or without a unit, or under a legacy key (``allowance_B_ex``, ``exempt_B_ex``,
  ``exempt_by_kind``, ``exempt (charged)``, ``B_ex_0_*``) — :class:`LegacyFieldError` / :class:`BudgetUnitError`;
* a per-request budget multiplied by the request count, unless the configuration declares separately analysed
  audits — :class:`EpochScopeError` (R16);
* input kinds presented as VUs without the ``legacy`` label — :class:`PopulationError`;
* a Summary whose families and banks disagree (``N_V != sum_a B_a v_a``) — :class:`PopulationError`.

Companions (not part of the bound)
----------------------------------
* :func:`completion_risk` — R18: ``1-(1-s)^n`` per retained RU and ``1-(1-s)^N`` aggregate, exact Fractions,
  labelled "probability of selecting an unsupported check — completion risk"; explicitly not a security error
  allowance, not an exemption, not the total rejection probability.
* :func:`multi_epoch_integrity` — R16: the union bound ``sum_e delta_e`` over a declared set of epochs, with the
  statistical and cryptographic contributions identified (the latter UNAVAILABLE unless supplied) and the
  unestablished assumptions listed.  It never sums per-epoch ``U``.

Schema: ``veritor/certificate/v2`` (see :data:`SCHEMA` and :func:`schema_description`).
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Mapping, Sequence

from veritor.analysis.certified import (
    Bank,
    CertifiedBound,
    Family,
    Summary,
    certified_bound,
)
from veritor.core.description import REPLAY, VERIFICATION
from veritor.core.policy import VerificationPolicy

SCHEMA = "veritor/certificate/v2"
COMPLETION_RISK_LABEL = "probability of selecting an unsupported check — completion risk"

# The contract's cost list (docs/PROTOCOL.md, "Costs"); every class is reported MEASURED (with provenance) or UNAVAILABLE.
COST_CLASSES: tuple[str, ...] = (
    "native_or_trusted_replay_baseline",
    "original_execution_and_collection",
    "initial_commitments_and_io_checks",
    "replay_including_attempted_work_for_later_exempted_RUs",
    "interior_commitments",
    "proof_generation",
    "verifier_derivation_of_expected_statements",
    "proof_verification",
)

# Legacy keys the builder refuses wherever they appear in a configuration (R8 / R14).
LEGACY_KEYS: frozenset[str] = frozenset({
    "allowance_B_ex", "exempt_B_ex", "exempt_by_kind", "exempt (charged)", "exempt_charged",
    "B_ex_0_bits", "B_ex_0_capped", "B_ex_0_precap_bits", "allowance", "B_ex_VU", "B_ex_vu",
})

INPUT_KIND_PREFIX = "Input"


class CertificateError(ValueError):
    """Base class: the configuration cannot be certified as given."""

    code = "E_CERTIFICATE"


class LegacyFieldError(CertificateError):
    code = "E_LEGACY_FIELD"


class BudgetUnitError(CertificateError):
    code = "E_BEX_VU_COUNT"


class EpochScopeError(CertificateError):
    code = "E_BEX_PER_REQUEST_SCALED"


class PopulationError(CertificateError):
    code = "E_POPULATION"


# --------------------------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Epoch:
    """The declared audit/epoch (R16): identifier, the exact requests/RUs covered, the scope of the error bound."""

    identifier: str
    requests: tuple[str, ...]
    replay_units: int
    scope_statement: str = "one audit: J drawn after every RU's step-1 values are fixed; Q after all step-3 values and the complete F"
    separately_analysed_audits: bool = False


@dataclass(frozen=True)
class Budget:
    """The exemption budget ``B_ex``: an RU count per audit/epoch, with provenance."""

    value: int
    unit: str  # must be "RU"
    provenance: str
    per_request: bool = False  # True means the caller scaled a per-request figure by the request count -> refused


@dataclass(frozen=True)
class Law:
    q: Fraction
    s: Fraction
    delta: Fraction
    provenance: str


@dataclass(frozen=True)
class OutputDomain:
    """The declared cap ``W`` (F-2) and what determines the exact domain, if known."""

    W_declared_bits: int
    provenance: str
    tokens: int | None = None
    vocabulary: int | None = None

    @property
    def W_exact_upper_bits(self) -> int | None:
        """ceil(tokens * log2(vocabulary)) computed exactly: the smallest k with vocabulary**tokens <= 2**k."""

        if self.tokens is None or self.vocabulary is None or self.vocabulary <= 0 or self.tokens < 0:
            return None
        domain = self.vocabulary ** self.tokens
        return max(0, (domain - 1).bit_length())


@dataclass(frozen=True)
class Sampled:
    """What was actually drawn (recorded; the Bound does not read these)."""

    J: int | None
    F: int | None
    Q: int | None
    provenance: str
    J_identities: tuple[str, ...] = ()
    F_identities: tuple[str, ...] = ()


@dataclass(frozen=True)
class PopulationPin:
    """The identity of the population *file* a Summary was built from (what the sampler pins at stage 1 and the verifier re-checks).

    ``vu_count`` / ``ru_count`` are the file's per-instance counts (``partition_sample.PopulationView`` replicates them
    ``instances`` times); ``inputs_removed_per_ru`` is what the certificate subtracted per RU copy (0 for a computed-only file).
    """

    file_sha256: str | None
    content_sha256: str | None
    partition_digest: str | None
    partition_digest_kind: str | None
    vu_count: int | None
    ru_count: int | None
    instances: int | None
    inputs_are_vus: bool | None
    construction: str | None
    inputs_removed_per_ru: tuple[int, ...] = ()


@dataclass(frozen=True)
class Population:
    summary: Summary
    source: str  # "a1" | "computed-only" | "legacy-minus-inputs" | "explicit"
    provenance: str
    input_kinds_excluded: tuple[tuple[str, int], ...] = ()  # (kind, total units removed)
    families_provenance: Mapping[str, str] = field(default_factory=dict)
    banks_provenance: Mapping[str, str] = field(default_factory=dict)
    pin: PopulationPin | None = None


@dataclass(frozen=True)
class SamplerPin:
    """What the *sampler* drew from and under which budget, as the verifier's record states it (``record_v3.json``).

    The certificate refuses unless the population it prices is the same object (digests, counts, per-RU candidate counts) and the
    budget it prices is the same RU count the sampler enforced ``|F| <= B_ex`` against.
    """

    population_file_sha256: str | None
    partition_digest: str | None
    partition_digest_kind: str | None
    replay_units: int
    inputs_are_vus: bool | None
    construction: str | None
    B_ex: int
    B_ex_unit: str
    J: tuple[int, ...] = ()  # RU indices
    F: tuple[int, ...] = ()
    Q_candidates: int | None = None  # sum of |VUs of r| over r in J \ F as the sampler counted them (stage-2 ``candidates``)
    provenance: str = "verify/record_v3.json"


class PopulationMismatchError(CertificateError):
    code = "E_POPULATION_MISMATCH"


def sampler_pin_from_record_v3(doc: Mapping[str, Any]) -> SamplerPin:
    """The sampler's pins from ``veritor/verifier-record/v3`` (``e2e.verifier_v3.reverify`` writes them)."""

    rec = doc.get("record") or {}
    pop = rec.get("population") or {}
    e = doc.get("epoch") or {}
    b = doc.get("B_ex") or {}
    s = doc.get("sampled") or {}
    if not isinstance(b, Mapping) or "value" not in b:
        raise BudgetUnitError(f"{BudgetUnitError.code}: record_v3 B_ex {b!r} lacks value/unit")
    return SamplerPin(
        population_file_sha256=rec.get("population_sha256"),
        partition_digest=rec.get("partition_digest"),
        partition_digest_kind=rec.get("partition_digest_kind") or pop.get("partition_digest_kind"),
        replay_units=int(e.get("replay_units", -1)),
        inputs_are_vus=pop.get("inputs_are_vus"),
        construction=pop.get("construction"),
        B_ex=int(b["value"]),
        B_ex_unit=str(b.get("unit")),
        J=tuple(int(x) for x in (s.get("J") or ())),
        F=tuple(int(x) for x in (s.get("F") or ())),
        Q_candidates=(int(s["candidates"]) if s.get("candidates") is not None else None),
        provenance=f"record_v3.json ({rec.get('path')}); status {doc.get('status')}",
    )


@dataclass(frozen=True)
class Cost:
    seconds: float
    provenance: str
    label: str = "MEASURED"


# --------------------------------------------------------------------------------------------------------------
# Population construction
# --------------------------------------------------------------------------------------------------------------


def _is_input_kind(kind: str, row: Any | None = None) -> bool:
    if kind.startswith(INPUT_KIND_PREFIX):
        return True
    if row is not None and getattr(row, "role", None) == VERIFICATION:
        # a source-only verification row (every gate an input/weight record) is an input kind under A1
        size = getattr(row, "size", None)
        src = (getattr(row, "source_inputs", 0) or 0) + (getattr(row, "source_weights", 0) or 0)
        if size is not None and size > 0 and src >= size and src > 0:
            return True
    return False


def population_from_legacy_kind_table(table: Any, *, provenance: str) -> Population:
    """The contract population from a legacy ``KindTable`` (A1's population not yet landed): Input kinds subtracted.

    Families are the VERIFICATION rows that are not input kinds (count = copies, width = the row's bottleneck cut);
    banks are the REPLAY rows with ``units`` = the per-copy multiplicity of the *non-input* verification kinds they
    reach.  Labelled ``legacy-minus-inputs``.
    """

    rows = {row.kind: row for row in table.rows}
    removed: dict[str, int] = {}
    counts: dict[str, int] = {}
    banks: list[Bank] = []
    for row in rows.values():
        if row.role != REPLAY:
            continue
        units = 0
        for inner, per_copy in row.verification_kinds:
            inner_row = rows.get(inner)
            if inner_row is None or inner_row.role != VERIFICATION:
                continue
            if _is_input_kind(inner, inner_row):
                removed[inner] = removed.get(inner, 0) + per_copy * row.copies
                continue
            units += per_copy
            counts[inner] = counts.get(inner, 0) + per_copy * row.copies
        banks.append(Bank(row.kind, row.copies, units, rows[row.kind].cut_bits))
    families = tuple(Family(kind, n, rows[kind].cut_bits) for kind, n in counts.items() if n > 0)
    root = rows[table.root]
    summary = Summary(families, tuple(banks), root.out_bits)
    _check_summary(summary)
    return Population(
        summary,
        source="legacy-minus-inputs",
        provenance=provenance,
        input_kinds_excluded=tuple(sorted(removed.items())),
        families_provenance={f.kind: "legacy kind table row (copies, bottleneck cut)" for f in families},
        banks_provenance={b.kind: "legacy kind table REPLAY row minus input kinds" for b in banks},
    )


def population_from_families(
    families: Sequence[Mapping[str, Any]],
    banks: Sequence[Mapping[str, Any]],
    out_width: int,
    *,
    source: str,
    provenance: str,
) -> Population:
    """An explicit population (A1's format or a hand-built one). Input kinds are refused unless ``source`` says legacy.

    ``families``: ``{"kind", "count", "width", "provenance"?, "role"?}``; ``banks``: ``{"kind", "copies",
    "units", "exit_width", "provenance"?}``.  A family whose ``role`` is ``"input"`` or whose kind starts with
    ``Input`` is a contract violation (Amendment A1: input gates are not VUs) — :class:`PopulationError` — unless
    ``source == "legacy"``, in which case it is subtracted and the population is labelled ``legacy-minus-inputs``.
    """

    removed: list[tuple[str, int]] = []
    fams: list[Family] = []
    fprov: dict[str, str] = {}
    for f in families:
        kind = str(f["kind"])
        is_input = f.get("role") == "input" or kind.startswith(INPUT_KIND_PREFIX)
        if is_input:
            if source != "legacy":
                raise PopulationError(
                    f"{PopulationError.code}: family {kind!r} is an input kind; input gates are not verification "
                    "units under the contract (Amendment A1). Supply the population without input kinds, or pass "
                    "source='legacy' to have them subtracted (labelled legacy-minus-inputs)."
                )
            removed.append((kind, int(f["count"])))
            continue
        fams.append(Family(kind, int(f["count"]), int(f["width"])))
        fprov[kind] = str(f.get("provenance", provenance))
    bks: list[Bank] = []
    bprov: dict[str, str] = {}
    for b in banks:
        bks.append(Bank(str(b["kind"]), int(b["copies"]), int(b["units"]), int(b["exit_width"])))
        bprov[str(b["kind"])] = str(b.get("provenance", provenance))
    if removed and source == "legacy":
        # the banks' units must be reduced by the removed input units per copy; refuse if not divisible
        total_removed = sum(n for _, n in removed)
        copies = sum(b.copies for b in bks)
        if copies == 0 or total_removed % copies:
            raise PopulationError(
                f"{PopulationError.code}: cannot subtract {total_removed} input units evenly over {copies} RU copies; "
                "supply banks whose units exclude input kinds."
            )
        per = total_removed // copies
        bks = [Bank(b.kind, b.copies, b.units - per, b.exit_width) for b in bks]
    summary = Summary(tuple(fams), tuple(bks), int(out_width))
    _check_summary(summary)
    return Population(
        summary,
        source="legacy-minus-inputs" if removed else source,
        provenance=provenance,
        input_kinds_excluded=tuple(removed),
        families_provenance=fprov,
        banks_provenance=bprov,
    )


POPULATION_SCHEMAS = ("veritor/population/v2", "veritor/a9/population-v5/v1")


def population_from_population_file(
    path: Any,
    *,
    instances: int,
    ru_exit_bits: int | Sequence[int],
    provenance: str | None = None,
) -> Population:
    """The certificate's population built from **the sampler's population file itself** (``population/v2`` compact rows or A9's
    ``population-v5``; A1's computed-only file when it lands in one of these shapes), so the two sides price and draw the same object.

    Families: one per kind, ``count`` = units of that kind in the file × ``instances``, ``width`` = the kind's declared
    ``out_bits`` capped at W.  Banks: one per RU index of the per-instance file, ``copies = instances``, ``units`` = the RU's
    non-input VUs, ``exit_width`` = ``ru_exit_bits`` (one int for all RUs or one per RU; the file does not carry RU interfaces).
    Rows whose kind is ``Input`` are subtracted and the population labelled ``legacy-minus-inputs``; a file without them is
    ``computed-only``.  The returned :class:`PopulationPin` carries the file's sha256, ``content_sha256``, ``partition_digest``
    and counts for :func:`check_sampler_pin`.
    """

    import hashlib
    from pathlib import Path

    p = Path(path)
    raw = p.read_bytes()
    doc = json.loads(raw)
    schema = doc.get("schema")
    if schema not in POPULATION_SCHEMAS:
        raise PopulationError(f"{PopulationError.code}: {p}: schema {schema!r} is not one of {POPULATION_SCHEMAS}")
    if instances < 1:
        raise PopulationError(f"{PopulationError.code}: instances must be >= 1")
    file_instances = int(doc.get("instances", 1))
    if file_instances != 1:
        raise PopulationError(f"{PopulationError.code}: {p} is a {file_instances}-instance file; the sampler replicates a per-instance file")
    if "rows" not in doc or "entries" not in (doc.get("kinds") or {}):
        raise PopulationError(f"{PopulationError.code}: {p} has no rows / kind entries (A9's population-v5 'rules' basis = D4's MODELED kind table "
                              "without unit widths or RU rows): not a sampled population the certificate can price; supply a rows-basis file")
    kinds = {int(k["id"]): k for k in doc["kinds"]["entries"]}
    ru_count = int(doc["ru_count"])
    if isinstance(ru_exit_bits, int):
        exits = [int(ru_exit_bits)] * ru_count
    else:
        exits = [int(x) for x in ru_exit_bits]
        if len(exits) != ru_count:
            raise PopulationError(f"{PopulationError.code}: {len(exits)} RU exit widths for {ru_count} RUs")
    W = instances * sum(exits)
    if doc.get("row_fields") != ["first_vu", "count", "ru", "kind", "lo", "stride", "unit_size"]:
        raise PopulationError(f"{PopulationError.code}: unexpected row_fields {doc.get('row_fields')!r}")
    per_kind: dict[int, int] = {}
    units = [0] * ru_count
    removed_ru = [0] * ru_count
    removed_kind: dict[str, int] = {}
    total = 0
    for first, count, ru, kid, _lo, _stride, _usz in doc["rows"]:
        total += count
        k = kinds[kid]
        if k["fn"] == INPUT_KIND_PREFIX:
            removed_ru[ru] += count
            removed_kind[f"{k['fn']}|{k['form']}"] = removed_kind.get(f"{k['fn']}|{k['form']}", 0) + count * instances
            continue
        per_kind[kid] = per_kind.get(kid, 0) + count
        units[ru] += count
    if total != int(doc["vu_count"]):
        raise PopulationError(f"{PopulationError.code}: rows cover {total} VUs, vu_count is {doc['vu_count']}")
    families = tuple(
        Family(f"{kinds[kid]['fn']}|{kinds[kid]['form']}", n * instances, min(int(kinds[kid]["out_bits"]), W))
        for kid, n in sorted(per_kind.items()) if n > 0
    )
    banks = tuple(Bank(f"RU{c}", instances, units[c], exits[c]) for c in range(ru_count))
    summary = Summary(families, banks, W)
    _check_summary(summary)
    removed_any = bool(removed_kind)
    construction = doc.get("construction") or ("legacy (population/v2 rows: Input kinds are VUs)" if removed_any else "computed-only rows")
    pin = PopulationPin(
        file_sha256=hashlib.sha256(raw).hexdigest(),
        content_sha256=doc.get("content_sha256"),
        partition_digest=doc.get("partition_digest"),
        partition_digest_kind=doc.get("partition_digest_kind"),
        vu_count=int(doc["vu_count"]),
        ru_count=ru_count,
        instances=instances,
        inputs_are_vus=removed_any,
        construction=construction,
        inputs_removed_per_ru=tuple(removed_ru),
    )
    prov = provenance or f"{p} (sha256 {pin.file_sha256[:16]}, partition {str(pin.partition_digest)[:16]}, {schema}) x {instances} instances"
    # the file's own construction label is carried (A9's population-v5 says "legacy-minus-inputs" although no Input row remains:
    # computed-only by subtraction, an interim population until A1's); a file that still has Input rows is labelled by the subtraction here
    source = ("legacy-minus-inputs (Input rows subtracted by the certificate)" if removed_any
              else f"computed-only ({doc.get('construction')})" if doc.get("construction") else "computed-only")
    return Population(
        summary,
        source=source,
        provenance=prov,
        input_kinds_excluded=tuple(sorted(removed_kind.items())),
        families_provenance={f.kind: "population file kind entry (count x instances; width = declared out_bits capped at W)" for f in families},
        banks_provenance={b.kind: "population file RU (non-input rows); exit width from the declared output domain" for b in banks},
        pin=pin,
    )


def certificate_from_record_v3(
    record_path: Any,
    population_path: Any,
    *,
    require_verified: bool = True,
    on_refused: str = "document",
    allow_legacy_sampler_population: bool = False,
    costs: Mapping[str, Cost] | None = None,
    resolution: int = 16,
    label: str = "DERIVED",
    force_exact_completion_risk: bool = False,
) -> dict[str, Any]:
    """The runner's one call: ``verify/record_v3.json`` (A3's verifier record) + **the population file the sampler ran on** →
    certificate (+ the R18 companion under ``completion_risk``).

    Everything the certificate prices comes from the record (``e2e.verifier_v3.certificate_inputs``) except the population, which is
    built from ``population_path`` and pinned against the record (:func:`check_sampler_pin`).  RU exit widths: the record's declared
    ``W_declared_bits`` split equally over the epoch's RUs — valid when every RU is one request (P1/cu=whole, the only validated
    plan); a population file with more than one RU per instance is refused until per-RU interfaces are supplied.

    A record whose status is not VERIFIED gets **no certificate**: with ``on_refused="document"`` (default) the return value is a
    refusal document (``veritor/certificate/v2/refused``: no ``U``, the verifier's reason, the same-population/same-budget check —
    which still runs, so a population mismatch on a refused run is reported — the epoch scope and the R18 companion); with
    ``on_refused="raise"`` the ``RecordRefused`` propagates.  ``require_verified=False`` builds a certificate anyway (labelled dry-run
    only; never for a run).
    """

    from e2e.verifier_v3 import certificate_inputs as ci  # local import: certificate_inputs imports this module

    doc = ci.load(record_path)
    refused = require_verified and doc.get("status") != "VERIFIED"
    if refused and on_refused == "raise":
        ci.certificate_inputs(doc, require_verified=True)  # raises RecordRefused
    inputs = ci.certificate_inputs(doc, require_verified=False)
    rec = doc["record"]
    N = int(rec["batch"])
    W = int(doc["output_domain"]["W_declared_bits"])
    ru = int(doc["epoch"]["replay_units"])
    if ru != N:
        raise PopulationError(f"{PopulationError.code}: {ru} RUs for {N} requests — per-RU exit widths are not in the population file; "
                              "supply them (A1) and call population_from_population_file directly")
    if W % ru:
        raise PopulationError(f"{PopulationError.code}: W_declared_bits={W} is not a multiple of the {ru} RUs")
    population = population_from_population_file(population_path, instances=N, ru_exit_bits=W // ru)
    pin = sampler_pin_from_record_v3(doc)
    record_ref = {"path": str(record_path), "status": doc.get("status"), "status_line": doc.get("status_line"), "utc": doc.get("utc"),
                  "checks": doc.get("checks"), "instance_digest": rec.get("instance_digest"),
                  # [E9, D40 rule (2)] the verifier's coverage line, quoted verbatim: proof-checked units (identities, k proof objects) and directly
                  # checked units (gather-equality/v1: authenticated openings + equality) SEPARATELY; the claim is "all sampled units verified"
                  "coverage_line": (doc.get("coverage") or {}).get("coverage_line"), "completion_claim": (doc.get("coverage") or {}).get("completion_claim")}
    cr = ci.completion_risk_inputs(doc)
    risk = (completion_risk(**cr, force_exact=force_exact_completion_risk) if cr is not None
            else "UNAVAILABLE (record carries no per-retained-RU unsupported counts)")
    if refused:
        return _refusal_document(doc, population, pin, inputs, record_ref, risk, allow_legacy_sampler_population)
    cert = build_certificate(population=population, sampler_pin=pin, costs=costs, resolution=resolution, label=label,
                             allow_legacy_sampler_population=allow_legacy_sampler_population, **inputs)
    cert["record_v3"] = record_ref
    cert["completion_risk"] = risk
    return cert


REFUSED_SCHEMA = SCHEMA + "/refused"


def _refusal_document(doc, population, pin, inputs, record_ref, risk, allow_legacy) -> dict[str, Any]:
    """No certificate for a refused run: what is stated instead."""

    epoch: Epoch = inputs["epoch"]
    budget: Budget = inputs["budget"]
    law: Law = inputs["law"]
    sampled: Sampled = inputs["sampled"]
    od: OutputDomain = inputs["output_domain"]
    summary = population.summary
    try:
        _check_summary(summary)
        check_budget(budget, epoch, summary)
        same: dict[str, Any] = check_sampler_pin(population, pin, budget, allow_legacy_sampler_population=allow_legacy)
        same_status = "same population and budget established (digests, candidates, B_ex agree)"
    except CertificateError as e:  # reported, not worked around: a mismatch on a refused run is a finding
        same = {"same_population": False, "refused": str(e), "label": "MISMATCH"}
        same_status = f"MISMATCH: {e}"
    retained = [r for r in pin.J if r not in set(pin.F)]
    return {
        "schema": REFUSED_SCHEMA,
        "status": "REFUSED",
        "certificate": None,
        "U": "NONE — no certificate is issued for a refused run; the verifier's refusal is the result (docs/PROTOCOL.md step 5: every selected equation covered)",
        "reason": record_ref.get("status_line"),
        "checks": record_ref.get("checks"),
        "record_v3": record_ref,
        "statement": (f"Epoch {epoch.identifier}: the verifier REFUSED the run at step 5; no bound on the adversary's output freedom is certified. "
                      f"The population and budget the sampler used are {'the' if same.get('same_population') else 'NOT the'} ones a certificate would price."),
        "epoch": {"identifier": epoch.identifier, "requests": list(epoch.requests), "request_count": len(epoch.requests), "replay_units": epoch.replay_units,
                  "J": list(pin.J), "F": list(pin.F), "retained": retained, "separately_analysed_audits": epoch.separately_analysed_audits,
                  "scope_of_error_bound": f"delta = {law.delta} is this epoch's (one audit: one B_ex, one delta; never per request); the union bound "
                                          "sum_e delta_e applies across epochs (R16)"},
        "B_ex": {"value": budget.value, "unit": "RU", "scope": "per audit/epoch", "provenance": budget.provenance, "F_used": len(pin.F)},
        "law": {"q": str(law.q), "s": str(law.s), "delta": str(law.delta), "provenance": law.provenance},
        "output_domain": {"W_declared_bits": od.W_declared_bits, "W_exact_upper_bits": od.W_exact_upper_bits, "tokens": od.tokens, "vocabulary": od.vocabulary,
                          "provenance": od.provenance, "note": "the domain a certificate would be capped at; no U is stated"},
        "population": {"source": population.source, "provenance": population.provenance, "N_V": summary.N_V, "N_R": summary.N_R,
                       "units_per_RU": [b.units for b in summary.banks], "families": len(summary.families),
                       "input_kinds_excluded": [{"kind": k, "units": n} for k, n in population.input_kinds_excluded],
                       "pin": {"file_sha256": population.pin.file_sha256, "content_sha256": population.pin.content_sha256, "partition_digest": population.pin.partition_digest,
                               "partition_digest_kind": population.pin.partition_digest_kind, "vu_count_per_instance": population.pin.vu_count, "construction": population.pin.construction}
                       if population.pin else None},
        "same_population_and_budget": same,
        "same_population_and_budget_status": same_status,
        "sampled": {"J": sampled.J, "F": sampled.F, "Q": sampled.Q, "provenance": sampled.provenance},
        "completion_risk": risk,
        "label": "REFUSED RUN — no certificate; MEASURED verifier outcome; the pin check and the R18 companion are DERIVED from the record and the population file",
    }


def render_refusal_text(doc: Mapping[str, Any]) -> str:
    e = doc["epoch"]
    lines = [
        f"NO CERTIFICATE ({doc['schema']}) — epoch {e['identifier']}: {e['request_count']} requests, {e['replay_units']} RUs; J={e['J']} F={e['F']} retained={e['retained']}",
        f"  status: REFUSED — {doc['reason']}",
        f"  U: {doc['U']}",
        f"  law: q={doc['law']['q']} s={doc['law']['s']} delta={doc['law']['delta']}; B_ex = {doc['B_ex']['value']} RU per epoch (|F| = {doc['B_ex']['F_used']}); {e['scope_of_error_bound']}",
        f"  output domain a certificate would be capped at: W = {doc['output_domain']['W_declared_bits']} bits (exact upper {doc['output_domain']['W_exact_upper_bits']}); no U is stated",
        f"  population ({doc['population']['source']}): N_V={doc['population']['N_V']:,} in N_R={doc['population']['N_R']} RUs ({doc['population']['units_per_RU'][0]:,} per RU); "
        f"file sha256 {str((doc['population']['pin'] or {}).get('file_sha256'))[:16]}, partition {str((doc['population']['pin'] or {}).get('partition_digest'))[:16]}",
        f"  same population and budget as the sampler: {doc['same_population_and_budget_status']}",
        f"  sampled: |J|={doc['sampled']['J']} |F|={doc['sampled']['F']} |Q|={doc['sampled']['Q']}",
    ]
    cr = doc.get("completion_risk")
    if isinstance(cr, dict):
        per = ", ".join(f"RU {k}: n={v['n']:,} p in [{v['probability_exact_decimal_interval'][0][:12]}, {v['probability_exact_decimal_interval'][1][:12]}] ({v['arithmetic']})"
                        for k, v in cr["per_retained_RU"].items())
        agg = cr["aggregate_over_J_minus_F"]
        agg_s = (f"N={agg['N']:,} p in [{agg['probability_exact_decimal_interval'][0][:12]}, {agg['probability_exact_decimal_interval'][1][:12]}]"
                 if isinstance(agg.get("N"), int) else str(agg.get("N")))
        lines.append(f"  completion risk (R18 companion, NOT part of any bound; {cr['label']}): {per}; aggregate over J \\ F: {agg_s}")
    else:
        lines.append(f"  completion risk: {cr}")
    lines.append(f"  {doc['label']}")
    return "\n".join(lines)


def check_sampler_pin(population: Population, sampler: SamplerPin, budget: Budget, *, allow_legacy_sampler_population: bool = False) -> dict[str, Any]:
    """Same population, same budget — established, not assumed.  Returns the comparison (goes into the certificate) or raises.

    Refuses (:class:`PopulationMismatchError`) unless: the certificate's population was built from a file (``pin`` present) whose
    sha256 / partition digest equal the sampler's; the RU count equals the epoch's; the sampler's stage-2 candidate count equals
    ``sum_{r in J \\ F}`` of the certificate's per-RU units (plus the subtracted input units when the sampler's file had them);
    the sampler's population has no Input rows unless ``allow_legacy_sampler_population`` (then the certificate is labelled);
    and the budget the sampler enforced ``|F| <= B_ex`` against equals the budget priced here (unit RU).
    """

    pin = population.pin
    if pin is None:
        raise PopulationMismatchError(
            f"{PopulationMismatchError.code}: the population was not built from a population file (no pin); use "
            "population_from_population_file on the file the sampler drew from (record.population_sha256)."
        )
    problems: list[str] = []
    if sampler.population_file_sha256 and pin.file_sha256 != sampler.population_file_sha256:
        problems.append(f"population file sha256 {pin.file_sha256[:16]} != sampler's {sampler.population_file_sha256[:16]}")
    if sampler.partition_digest and pin.partition_digest != sampler.partition_digest:
        problems.append(f"partition_digest {str(pin.partition_digest)[:16]} != sampler's {sampler.partition_digest[:16]}")
    if sampler.partition_digest_kind and pin.partition_digest_kind and pin.partition_digest_kind != sampler.partition_digest_kind:
        problems.append(f"partition_digest_kind {pin.partition_digest_kind!r} != sampler's {sampler.partition_digest_kind!r}")
    N_R = population.summary.N_R
    if sampler.replay_units != N_R:
        problems.append(f"epoch replay_units {sampler.replay_units} != certificate N_R {N_R}")
    if sampler.inputs_are_vus and not allow_legacy_sampler_population:
        problems.append("the sampler's population has Input rows as VUs (draws were spent on input positions); the contract wants the same "
                        "computed-only population on both sides — pass allow_legacy_sampler_population=True only for a labelled legacy dry-run")
    if sampler.inputs_are_vus is False and pin.inputs_are_vus:
        problems.append("the certificate subtracted Input rows from a file the sampler reports as computed-only")
    # per-RU candidate identity: sum over retained RUs of the sampler's |VUs of r| == certificate units (+ subtracted inputs)
    cand_check: dict[str, Any] = {"checked": False}
    if sampler.Q_candidates is not None and sampler.J:
        retained = [r for r in sampler.J if r not in set(sampler.F)]
        per_ru_units = {}
        ru_count = pin.ru_count or N_R
        for g in retained:
            inst, local = divmod(g, ru_count)  # global RU = instance * ru_count + local (partition_sample.ReplicatedView)
            bank = population.summary.banks[local] if local < len(population.summary.banks) else None
            if bank is None or inst >= (pin.instances or 1):
                problems.append(f"sampler RU index {g} outside the certificate's {N_R} RUs")
                break
            per_ru_units[g] = bank.units + (pin.inputs_removed_per_ru[local] if sampler.inputs_are_vus else 0)
        expected = sum(per_ru_units.values())
        cand_check = {"checked": True, "sampler_candidates": sampler.Q_candidates, "certificate_sum_over_retained": expected,
                      "includes_subtracted_inputs": bool(sampler.inputs_are_vus), "retained": retained}
        if expected != sampler.Q_candidates:
            problems.append(f"stage-2 candidates {sampler.Q_candidates} != sum over J\\F of the certificate's per-RU units {expected}")
    if sampler.B_ex_unit not in ("RU", "replay_units"):
        problems.append(f"sampler budget unit {sampler.B_ex_unit!r} is not an RU count")
    if sampler.B_ex != budget.value:
        problems.append(f"sampler enforced |F| <= B_ex = {sampler.B_ex} but the certificate prices B_ex = {budget.value}")
    if len(sampler.F) > budget.value:
        problems.append(f"|F| = {len(sampler.F)} exceeds B_ex = {budget.value}")
    if problems:
        raise PopulationMismatchError(f"{PopulationMismatchError.code}: " + "; ".join(problems))
    return {
        "same_population": True,
        "population_file_sha256": pin.file_sha256,
        "partition_digest": pin.partition_digest,
        "partition_digest_kind": pin.partition_digest_kind,
        "sampler_population_construction": sampler.construction,
        "sampler_population_includes_inputs": bool(sampler.inputs_are_vus),
        "inputs_subtracted_by_certificate": sum(pin.inputs_removed_per_ru) * (pin.instances or 1),
        "candidates": cand_check,
        "same_budget": {"B_ex_RU": budget.value, "sampler_B_ex": sampler.B_ex, "F": len(sampler.F)},
        "sampler_provenance": sampler.provenance,
        "label": ("LEGACY SAMPLER POPULATION (Input rows drawn as VUs; certificate prices the computed-only remainder — labelled dry-run)"
                  if sampler.inputs_are_vus else "same computed-only population and RU budget on both sides"),
    }


def _check_summary(summary: Summary) -> None:
    """Every VU belongs to exactly one RU: ``N_V == sum_a B_a v_a`` (``certified.Summary`` does not check this)."""

    in_banks = sum(b.copies * b.units for b in summary.banks)
    if in_banks != summary.N_V:
        raise PopulationError(
            f"{PopulationError.code}: E_SUMMARY_INCONSISTENT — families count N_V={summary.N_V} VUs but the banks hold "
            f"sum_a B_a v_a={in_banks}; every VU must belong to exactly one RU."
        )
    if any(b.units < 0 for b in summary.banks):
        raise PopulationError(f"{PopulationError.code}: a bank has a negative unit count after subtraction")


# --------------------------------------------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------------------------------------------


def refuse_legacy_fields(config: Mapping[str, Any], path: str = "") -> None:
    """Raise :class:`LegacyFieldError` for any legacy key anywhere in ``config`` (recursively)."""

    for key, value in config.items():
        here = f"{path}.{key}" if path else str(key)
        if key in LEGACY_KEYS:
            raise LegacyFieldError(
                f"{LegacyFieldError.code}: legacy field {here!r} is not accepted by certificate_v2. "
                "'exempt (charged)' / 'allowance_B_ex' / 'exempt_B_ex' name selected VUs, not exemptions; the "
                "contract's exemption is a selected RU in F under one global RU-count budget. Supply "
                "B_ex = {value, unit: 'RU', provenance}."
            )
        if key == "B_ex" and not isinstance(value, Mapping):
            raise BudgetUnitError(
                f"{BudgetUnitError.code}: {here!r} must be an object {{value, unit: 'RU', provenance}}; a bare number has "
                "no unit and is refused (legacy paths passed a VU count here)."
            )
        if isinstance(value, Mapping):
            refuse_legacy_fields(value, here)


def check_budget(budget: Budget, epoch: Epoch, summary: Summary) -> None:
    if budget.unit != "RU":
        raise BudgetUnitError(
            f"{BudgetUnitError.code}: B_ex is given in unit {budget.unit!r}; the contract's exemption budget counts "
            "replay units (RUs) exempted per audit/epoch. A count of selected VUs (legacy 'exempt (charged)') is "
            "not a B_ex and is refused, not reinterpreted."
        )
    if budget.value < 0:
        raise BudgetUnitError(f"{BudgetUnitError.code}: B_ex must be >= 0, got {budget.value}")
    if budget.per_request and not epoch.separately_analysed_audits:
        raise EpochScopeError(
            f"{EpochScopeError.code}: B_ex={budget.value} is declared as a per-request budget multiplied by the request "
            f"count ({len(epoch.requests)} requests in epoch {epoch.identifier!r}). B_ex is per audit/epoch, never per "
            "request (R16). Declare separately analysed audits (one certificate each) or give the epoch's RU budget."
        )
    if epoch.replay_units != summary.N_R:
        raise EpochScopeError(
            f"{EpochScopeError.code}: the epoch declares {epoch.replay_units} RUs but the population's banks hold "
            f"N_R={summary.N_R}; the certificate's RU set must be exactly the epoch's."
        )
    if budget.value > summary.N_R:
        raise BudgetUnitError(
            f"{BudgetUnitError.code}: B_ex={budget.value} exceeds the epoch's N_R={summary.N_R} RUs — a budget larger than the RU set "
            "is not an RU count (protocol_instance.validate_instance refuses the same); the historical 4,079 was a selected-VU count."
        )


# --------------------------------------------------------------------------------------------------------------
# The certificate
# --------------------------------------------------------------------------------------------------------------


def _frac(x: Fraction | int | None) -> str | None:
    return None if x is None else str(Fraction(x))


def _bits_to_TB(bits: float) -> float:
    return bits / 8 / 1e12


def _winner_doc(result: CertifiedBound) -> dict[str, Any]:
    st = result.status
    w = st.winner
    return {
        "case": w.case,
        "threshold_t": w.threshold,
        "lambda": _frac(w.lam),
        "multiplier_z": _frac(w.multiplier),
        "origin": w.origin,
        "value_bits_exact": _frac(w.value),
        "branch": st.branch,
        "thresholds_tried": st.thresholds_tried,
        "candidates_certified": st.candidates_certified,
        "fallback_won": st.fallback_won,
        "cap_won": st.cap_won,
    }


def build_certificate(
    *,
    epoch: Epoch,
    population: Population,
    budget: Budget,
    law: Law,
    output_domain: OutputDomain,
    sampled: Sampled,
    costs: Mapping[str, Cost] | None = None,
    assumptions_assumed: Sequence[str] = (),
    resolution: int = 16,
    extra_config: Mapping[str, Any] | None = None,
    label: str = "DERIVED",
    sampler_pin: SamplerPin | None = None,
    allow_legacy_sampler_population: bool = False,
) -> dict[str, Any]:
    """One certificate for one declared epoch. Raises a :class:`CertificateError` subclass on refused inputs.

    ``sampler_pin`` (from ``record_v3.json`` via :func:`sampler_pin_from_record_v3`) makes the builder **refuse** unless the population
    it prices and the budget it prices are the sampler's (:func:`check_sampler_pin`); without it the certificate says so
    (``same_population_and_budget: "NOT ESTABLISHED"``).
    """

    if extra_config:
        refuse_legacy_fields(extra_config)
    summary = population.summary
    _check_summary(summary)
    check_budget(budget, epoch, summary)
    if sampled.F is not None and sampled.F > budget.value:
        raise EpochScopeError(f"{EpochScopeError.code}: |F| = {sampled.F} exceeds B_ex = {budget.value} (the sampler must have refused at step 4)")
    same = (check_sampler_pin(population, sampler_pin, budget, allow_legacy_sampler_population=allow_legacy_sampler_population)
            if sampler_pin is not None else
            {"same_population": None, "label": "NOT ESTABLISHED: no sampler pin was supplied (pass sampler_pin=sampler_pin_from_record_v3(record))"})
    if output_domain.W_declared_bits != summary.out_width:
        raise PopulationError(
            f"{PopulationError.code}: the population's cap ({summary.out_width} bits) differs from the declared output "
            f"domain W={output_domain.W_declared_bits}; the cap is the declared domain (F-2) and must agree."
        )
    W_exact = output_domain.W_exact_upper_bits
    if W_exact is not None and W_exact > output_domain.W_declared_bits:
        raise PopulationError(
            f"{PopulationError.code}: the exact output domain ({output_domain.tokens} tokens x vocabulary "
            f"{output_domain.vocabulary} -> {W_exact} bits) exceeds the declared W={output_domain.W_declared_bits}; a "
            "declared cap below the actual domain is not a valid cap."
        )

    policy = VerificationPolicy(q=law.q, s=law.s)
    result = certified_bound(summary, policy, law.delta, allowance=budget.value, resolution=resolution)
    U = result.bits
    W = summary.out_width
    capped = bool(result.capped or U >= W)

    cost_doc: dict[str, Any] = {}
    for cls in COST_CLASSES:
        c = (costs or {}).get(cls)
        cost_doc[cls] = (
            {"seconds": c.seconds, "label": c.label, "provenance": c.provenance}
            if c is not None
            else "UNAVAILABLE"
        )
    unknown = sorted(set(costs or {}) - set(COST_CLASSES))
    if unknown:
        raise CertificateError(
            f"E_COST_CLASS: cost classes {unknown} are not in the contract's cost list {list(COST_CLASSES)}"
        )

    families_doc = [
        {"kind": f.kind, "count": f.count, "width_bits": f.width,
         "provenance": population.families_provenance.get(f.kind, population.provenance)}
        for f in summary.families
    ]
    banks_doc = [
        {"kind": b.kind, "copies": b.copies, "units_per_copy": b.units, "exit_width_bits": b.exit_width,
         "provenance": population.banks_provenance.get(b.kind, population.provenance)}
        for b in summary.banks
    ]

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "label": label,
        "epoch": {
            "identifier": epoch.identifier,
            "requests": list(epoch.requests),
            "request_count": len(epoch.requests),
            "replay_units": epoch.replay_units,
            "scope_of_error_bound": epoch.scope_statement,
            "separately_analysed_audits": epoch.separately_analysed_audits,
        },
        "scope": "single-epoch",
        "law": {"q": str(law.q), "s": str(law.s), "delta": str(law.delta), "provenance": law.provenance,
                "form": "adaptive early-exemption law: E_J[max_{F ⊆ J, |F| ≤ B_ex} (1-s)^{sum_{R in J\\F} j_R}]"},
        "B_ex": {"value": budget.value, "unit": "RU", "scope": "per audit/epoch", "provenance": budget.provenance},
        "population": {
            "source": population.source,
            "provenance": population.provenance,
            "N_V": summary.N_V,
            "N_R": summary.N_R,
            "v_max": summary.v_max,
            "families": families_doc,
            "banks": banks_doc,
            "input_kinds_excluded": [{"kind": k, "units": n} for k, n in population.input_kinds_excluded],
            "consistency": "N_V == sum_a B_a v_a (checked)",
        },
        "output_domain": {
            "W_declared_bits": output_domain.W_declared_bits,
            "W_exact_upper_bits": W_exact,
            "tokens": output_domain.tokens,
            "vocabulary": output_domain.vocabulary,
            "provenance": output_domain.provenance,
            "cap_used": "W_declared_bits (F-2: the declared interface is the cap)",
        },
        "same_population_and_budget": same,
        "sampled": {
            "J": sampled.J, "F": sampled.F, "Q": sampled.Q, "provenance": sampled.provenance,
            "J_identities": list(sampled.J_identities), "F_identities": list(sampled.F_identities),
            "note": "recorded; the Bound is for the stated law, not conditioned on the draw",
        },
        "bound": {
            "U_bits": U,
            "U_TB": _bits_to_TB(U),
            "W_bits": W,
            "U_equals_W": capped,
            "statement": (f"U = W = {W} bits (capped: no certified candidate below the declared output domain)" if capped
                          else f"U = {U} bits < W = {W} bits"),
            "precap_bits": _frac(result.precap_bits) if result.precap_bits is not None else None,
            "winner": _winner_doc(result),
            "calculator": "veritor.analysis.certified.certified_bound (Appendix A search; rational arithmetic)",
            "resolution": resolution,
        },
        "costs": cost_doc,
        "assumptions": {
            "assumed": list(assumptions_assumed) or [
                "S1–S3 as listed in docs/PROTOCOL-CONFORMANCE.md §Assumptions (assumed, not established here)",
            ],
            "unestablished": [
                "S4: transfer of the abstract guarantee to the implementation is conditional — the association of "
                "commitments with stage-fixed assignments and the agreement of later proved values with them is not "
                "established (docs/PROTOCOL.md 'The proof obligation'); this certificate bounds the abstract game only",
                "cryptographic contribution to delta (commitment binding, proof soundness): UNAVAILABLE",
            ],
        },
        "not_in_this_certificate": [
            "(B_ex + sigma)/q — a check from discussion, not an output",
            "completion risk (R18) — see completion_risk(); not part of the bound",
            "any sum of per-epoch U — combined-output freedom is one certificate over the union construction or per epoch",
        ],
    }
    return doc


def render_text(doc: Mapping[str, Any]) -> str:
    """A human-readable certificate."""

    b = doc["bound"]; e = doc["epoch"]; p = doc["population"]; law = doc["law"]; od = doc["output_domain"]
    w = b["winner"]
    lines = [
        f"CERTIFICATE ({doc['schema']}, {doc['label']}) — epoch {e['identifier']}: {e['request_count']} requests, "
        f"{e['replay_units']} RUs; scope: {e['scope_of_error_bound']}",
        f"  {b['statement']}  [U = {b['U_bits']} bits = {b['U_TB']:.6g} TB; W = {b['W_bits']} bits; U == W: {b['U_equals_W']}]",
        f"  law: q={law['q']} s={law['s']} delta={law['delta']} ({law['provenance']}); B_ex = {doc['B_ex']['value']} RU "
        f"per epoch ({doc['B_ex']['provenance']})",
        f"  winner: case={w['case']} t={w['threshold_t']} lambda={w['lambda']} z={w['multiplier_z']} origin={w['origin']} "
        f"branch={w['branch']} (thresholds tried {w['thresholds_tried']}, candidates {w['candidates_certified']})",
        f"  population ({p['source']}; {p['provenance']}): N_V={p['N_V']:,} VUs in N_R={p['N_R']} RUs; "
        f"input kinds excluded: {', '.join(f'{x['kind']} ({x['units']:,})' for x in p['input_kinds_excluded']) or 'none'}",
        f"  output domain: W_declared={od['W_declared_bits']} bits"
        + (f", exact upper {od['W_exact_upper_bits']} bits ({od['tokens']} tokens x vocabulary {od['vocabulary']})"
           if od["W_exact_upper_bits"] is not None else "")
        + f" ({od['provenance']})",
        f"  sampled: |J|={doc['sampled']['J']} |F|={doc['sampled']['F']} |Q|={doc['sampled']['Q']} ({doc['sampled']['provenance']})",
        *([f"  coverage (record_v3.json#coverage.coverage_line): {doc['record_v3']['coverage_line']}"]
          if isinstance(doc.get("record_v3"), dict) and doc["record_v3"].get("coverage_line") else []),
        f"  same population and budget as the sampler: {doc['same_population_and_budget'].get('label')}",
        "  costs:",
    ]
    for cls, c in doc["costs"].items():
        lines.append(f"    {cls}: " + (c if isinstance(c, str) else f"{c['seconds']:.1f} s {c['label']} ({c['provenance']})"))
    lines.append("  assumptions assumed: " + "; ".join(doc["assumptions"]["assumed"]))
    lines.append("  UNESTABLISHED: " + "; ".join(doc["assumptions"]["unestablished"]))
    cr = doc.get("completion_risk")
    if isinstance(cr, dict):
        per = ", ".join(f"RU {k}: n={v['n']} p in [{v['probability_exact_decimal_interval'][0][:10]}, {v['probability_exact_decimal_interval'][1][:10]}] ({v['arithmetic']})"
                        for k, v in cr["per_retained_RU"].items())
        agg = cr["aggregate_over_J_minus_F"]
        agg_s = (f"N={agg['N']} p in [{agg['probability_exact_decimal_interval'][0][:10]}, {agg['probability_exact_decimal_interval'][1][:10]}]"
                 if isinstance(agg.get("N"), int) else str(agg.get("N")))
        lines.append(f"  completion risk (R18 companion, NOT part of the bound; {cr['label']}): {per}; aggregate over J \\ F: {agg_s}")
    elif cr is not None:
        lines.append(f"  completion risk (R18 companion): {cr}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------
# R16: several epochs
# --------------------------------------------------------------------------------------------------------------


def multi_epoch_integrity(
    epochs: Sequence[Mapping[str, Any]],
    *,
    horizon: str,
    crypto_delta: Mapping[str, Fraction] | None = None,
) -> dict[str, Any]:
    """The union bound over a declared set of epochs: ``Pr[some epoch claim fails] <= sum_e delta_e``.

    ``epochs``: ``{"identifier", "delta": Fraction, "certificate"?: a v2 certificate doc}``.  ``crypto_delta``: an
    optional per-epoch cryptographic failure term (binding/soundness); UNAVAILABLE when absent.  Independence is not
    needed; the per-epoch guarantees must hold on the actual sequence (listed as an assumption).  Output-choice
    freedom is **not** combined: per-epoch ``U`` are listed side by side; ``combined_U`` names the union construction.
    """

    stat = Fraction(0)
    crypto: Fraction | None = Fraction(0) if crypto_delta else None
    per: list[dict[str, Any]] = []
    for e in epochs:
        d = Fraction(e["delta"])
        stat += d
        cd = (crypto_delta or {}).get(e["identifier"])
        if crypto is not None:
            if cd is None:
                crypto = None
            else:
                crypto += Fraction(cd)
        cert = e.get("certificate")
        per.append({
            "identifier": e["identifier"],
            "delta_statistical": str(d),
            "delta_cryptographic": str(Fraction(cd)) if cd is not None else "UNAVAILABLE",
            "U_bits": None if cert is None else cert["bound"]["U_bits"],
            "U_equals_W": None if cert is None else cert["bound"]["U_equals_W"],
        })
    total = None if crypto is None else stat + crypto
    return {
        "schema": SCHEMA + "/multi-epoch",
        "horizon": horizon,
        "epochs": per,
        "integrity_union_bound": {
            "statement": "Pr[some epoch claim fails] <= sum_e delta_e over the stated set (union bound; independence not needed)",
            "statistical_sum": str(stat),
            "statistical_sum_float": float(stat),
            "cryptographic_sum": "UNAVAILABLE" if crypto is None else str(crypto),
            "total": "UNAVAILABLE (cryptographic contribution missing)" if total is None else str(total),
        },
        "combined_U": "not a sum — the combined-output freedom is one certificate over the union construction "
                      "(veritor.analysis.union: one Bound with one J, one B_ex, one delta over all RUs) or is read per epoch above",
        "assumptions": {
            "unestablished": [
                "each per-epoch guarantee applies to the actual sequence (epoch barrier held: J drawn after all step-1 "
                "values of the epoch, Q after all step-3 values and the complete F) — to be recorded by A3",
                "S4 (transfer of the abstract guarantee) — conditional",
            ] + ([] if crypto is not None else ["cryptographic contribution to each delta_e: UNAVAILABLE"]),
        },
    }


# --------------------------------------------------------------------------------------------------------------
# R18: completion risk (not part of the bound)
# --------------------------------------------------------------------------------------------------------------


def completion_risk_exact(s: Fraction, n: int) -> Fraction:
    """``1 - (1-s)^n`` as an exact Fraction (its denominator is ``den(s)**n``: use :func:`_decimal_interval` to print).

    Exact only while ``n * bit_length(den(s))`` is small (say <= 2**22 bits); :func:`completion_risk_interval` is the general form.
    """

    return 1 - (1 - Fraction(s)) ** n


EXACT_RISK_BITS = 1 << 22  # exact Fraction power while den((1-s)^n) has at most this many bits (~4 Mbit: milliseconds)
RISK_INTERVAL_BITS = 256  # directed-rounding precision of the interval power beyond that


def _round_down(x: Fraction, bits: int) -> Fraction:
    return Fraction((x.numerator << bits) // x.denominator, 1 << bits)


def _round_up(x: Fraction, bits: int) -> Fraction:
    return Fraction(-((-x.numerator << bits) // x.denominator), 1 << bits)


def _pow_interval(base: Fraction, n: int, bits: int) -> tuple[Fraction, Fraction]:
    """``[lo, hi]`` with ``lo <= base**n <= hi`` for ``0 <= base <= 1`` by square-and-multiply with directed rounding to ``bits``
    binary digits after every product (each step keeps ``lo <= exact <= hi``: products of non-negative bounds are monotone)."""

    lo = hi = Fraction(1)
    b_lo, b_hi = Fraction(base), Fraction(base)
    e = int(n)
    while e:
        if e & 1:
            lo, hi = _round_down(lo * b_lo, bits), _round_up(hi * b_hi, bits)
        e >>= 1
        if e:
            b_lo, b_hi = _round_down(b_lo * b_lo, bits), _round_up(b_hi * b_hi, bits)
    return lo, min(hi, Fraction(1))


@functools.lru_cache(maxsize=64)
def completion_risk_interval(s: Fraction, n: int, *, force_exact: bool = False) -> tuple[Fraction, Fraction, bool]:
    """``(lo, hi, exact)`` with ``lo <= 1 - (1-s)^n <= hi``: exact (``lo == hi``) while the exact Fraction is small enough
    (:data:`EXACT_RISK_BITS`), otherwise a certified interval from :func:`_pow_interval` at :data:`RISK_INTERVAL_BITS` bits.
    Sound at any ``n`` (L1: ``n = 2.3e7`` per RU; 6.5K/1.6K: ``n ~ 1e11``): every rounding is in the direction of the bound."""

    s = Fraction(s)
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    if not (0 <= s <= 1):
        raise ValueError("s must be in [0, 1]")
    if n == 0 or s == 0:
        return Fraction(0), Fraction(0), True
    if s == 1:
        return Fraction(1), Fraction(1), True
    if force_exact or n * (1 - s).denominator.bit_length() <= EXACT_RISK_BITS:
        p = completion_risk_exact(s, n)  # force_exact: n = 5.6e6 at s = 2^-14 takes ~30 s and ~20 MB (78-Mbit numerator/denominator)
        return p, p, True
    q_lo, q_hi = _pow_interval(1 - s, n, RISK_INTERVAL_BITS)
    return 1 - q_hi, 1 - q_lo, False


def _one_minus_pow_float(s: Fraction, n: int) -> float:
    """``1 - (1-s)^n`` via ``-expm1(n * log1p(-s))`` (no cancellation for tiny ``s``)."""

    return -math.expm1(n * math.log1p(-float(s)))


def _decimal_interval(p: Fraction, digits: int = 30) -> tuple[str, str]:
    """Exact decimal lower/upper bounds on ``p`` in ``[0, 1]`` with ``digits`` decimals (floor and ceil of ``p * 10**digits``)."""

    scale = 10 ** digits
    lo = (p.numerator * scale) // p.denominator
    hi = -((-p.numerator * scale) // p.denominator)

    def fmt(k: int) -> str:
        s = str(k).rjust(digits + 1, "0")
        return f"{s[:-digits]}.{s[-digits:]}"

    return fmt(lo), fmt(hi)


def _risk_entry(s: Fraction, n: int, *, force_exact: bool = False) -> dict[str, Any]:
    p_lo, p_hi, exact = completion_risk_interval(s, n, force_exact=force_exact)
    lo, hi = _decimal_interval(p_lo)[0], _decimal_interval(p_hi)[1]
    q_hi = 1 - p_lo  # upper bound on (1-s)^n = the probability of selecting NO unsupported check
    return {
        "n": n,
        "probability_exact_decimal_interval": [lo, hi],  # certified rational bounds, 30 decimals
        "one_minus_p_lt_2_pow": (q_hi.numerator.bit_length() - q_hi.denominator.bit_length() + 1) if q_hi > 0 else None,  # (1-s)^n < 2^this
        "probability_float": _one_minus_pow_float(s, n),  # -expm1(n log1p(-s)), display only
        "exact_is": ("1 - (1 - s)**n as a Fraction (completion_risk_exact); denominator den(s)**n, not printed" if exact else
                     f"certified interval: (1-s)**n by square-and-multiply with directed rounding to {RISK_INTERVAL_BITS} bits "
                     "(completion_risk_interval); the exact Fraction is too large to form"),
        "arithmetic": "exact" if exact else "interval (sound bounds)",
    }


def completion_risk(
    s: Fraction,
    per_retained_RU: Mapping[str, int],
    *,
    aggregate_N: int | None = None,
    disjoint: bool = False,
    sampled_unsupported: Sequence[str] = (),
    provenance: str = "UNAVAILABLE (counts must come from the actual configuration's unsupported set)",
    force_exact: bool = False,
) -> dict[str, Any]:
    """R18: probability of selecting at least one unsupported check, per retained RU and in aggregate over J \\ F.

    ``force_exact``: form the exact Fraction even beyond :data:`EXACT_RISK_BITS` (minutes at n ~ 1e7; never at 1e11).

    ``per_retained_RU``: ``{RU identifier: n_R}`` distinct VUs in that RU whose check cannot be established.
    ``aggregate_N``: the number of distinct unsupported VUs over ``J \\ F`` (no double counting across overlapping
    type labels).  If omitted, it is taken as the sum of ``n_R`` only when ``disjoint=True`` (the caller asserts the
    per-RU sets are disjoint, which holds when every VU belongs to one RU); otherwise it is UNAVAILABLE.
    Exact ``Fraction`` arithmetic; floats via ``log1p`` for display only.
    """

    s = Fraction(s)
    if not (0 <= s <= 1):
        raise ValueError("s must be in [0, 1]")
    per: dict[str, Any] = {}
    for ru, n in per_retained_RU.items():
        n = int(n)
        if n < 0:
            raise ValueError(f"n_R must be >= 0 (RU {ru!r})")
        per[str(ru)] = _risk_entry(s, n, force_exact=force_exact)
    if aggregate_N is None and disjoint:
        aggregate_N = sum(int(n) for n in per_retained_RU.values())
    agg: dict[str, Any]
    if aggregate_N is None:
        agg = {"N": "UNAVAILABLE (aggregate over J \\ F without double counting not supplied)",
               "probability_float": "UNAVAILABLE"}
    else:
        if aggregate_N < 0:
            raise ValueError("N must be >= 0")
        agg = {"N": aggregate_N, **{k: v for k, v in _risk_entry(s, aggregate_N, force_exact=force_exact).items() if k != "n"}}
    return {
        "schema": SCHEMA + "/completion-risk",
        "label": COMPLETION_RISK_LABEL,
        "s": str(s),
        "per_retained_RU": per,
        "aggregate_over_J_minus_F": agg,
        "sampled_unsupported_identities": list(sampled_unsupported),
        "provenance": provenance,
        "not_security_error_allowance": True,
        "not_exemption": True,
        "not_total_rejection_probability": True,
        "note": "conditional on J, F; a fallback removes a VU from the unsupported set only when implemented and accounted for",
    }


# --------------------------------------------------------------------------------------------------------------
# Schema (shared interface for A9's runner)
# --------------------------------------------------------------------------------------------------------------


def schema_description() -> dict[str, Any]:
    """The v2 certificate schema as A9's runner consumes it (field -> type/meaning)."""

    return {
        "schema": SCHEMA,
        "label": "DERIVED | MEASURED | MODELED",
        "epoch": {"identifier": "str", "requests": "[str]", "request_count": "int", "replay_units": "int (== N_R)",
                  "scope_of_error_bound": "str", "separately_analysed_audits": "bool"},
        "scope": "'single-epoch' (one U) — several epochs go through multi_epoch_integrity (no total U)",
        "law": {"q": "Fraction str", "s": "Fraction str", "delta": "Fraction str", "provenance": "str", "form": "str"},
        "B_ex": {"value": "int (RU count)", "unit": "'RU' (only)", "scope": "'per audit/epoch'", "provenance": "str"},
        "population": {"source": "'a1' | 'legacy-minus-inputs' | 'explicit'", "N_V": "int", "N_R": "int", "v_max": "int",
                       "families": "[{kind, count, width_bits, provenance}]",
                       "banks": "[{kind, copies, units_per_copy, exit_width_bits, provenance}]",
                       "input_kinds_excluded": "[{kind, units}]", "consistency": "str"},
        "output_domain": {"W_declared_bits": "int (the cap)", "W_exact_upper_bits": "int | null", "tokens": "int | null",
                          "vocabulary": "int | null", "provenance": "str", "cap_used": "str"},
        "sampled": {"J": "int | null", "F": "int | null", "Q": "int | null", "provenance": "str",
                    "J_identities": "[str]", "F_identities": "[str]"},
        "bound": {"U_bits": "int", "U_TB": "float", "W_bits": "int", "U_equals_W": "bool", "statement": "str",
                  "precap_bits": "Fraction str | null",
                  "winner": {"case": "str", "threshold_t": "int | null", "lambda": "Fraction str | null",
                             "multiplier_z": "Fraction str | null", "origin": "str", "value_bits_exact": "Fraction str",
                             "branch": "str", "thresholds_tried": "int", "candidates_certified": "int",
                             "fallback_won": "bool", "cap_won": "bool"},
                  "calculator": "str", "resolution": "int"},
        "costs": {cls: "{seconds, label, provenance} | 'UNAVAILABLE'" for cls in COST_CLASSES},
        "assumptions": {"assumed": "[str]", "unestablished": "[str]"},
        "not_in_this_certificate": "[str]",
        "same_population_and_budget": "check_sampler_pin() result: same_population true | null (NOT ESTABLISHED); refused with E_POPULATION_MISMATCH otherwise",
        "refused": {
            "E_LEGACY_FIELD": sorted(LEGACY_KEYS),
            "E_BEX_VU_COUNT": "B_ex.unit != 'RU' or a bare number",
            "E_BEX_PER_REQUEST_SCALED": "B_ex.per_request without epoch.separately_analysed_audits; epoch RUs != N_R",
            "E_POPULATION": "input kinds as VUs (unless source='legacy'); N_V != sum B_a v_a; cap != W_declared; W_exact > W_declared",
            "E_POPULATION_MISMATCH": "sampler_pin given and: population not from a file; file sha256 / partition digest differ; replay_units != N_R; stage-2 candidates != sum over J\\F of per-RU units; sampler population has Input rows (unless labelled legacy dry-run); sampler B_ex != priced B_ex; |F| > B_ex",
        },
    }


def dumps(doc: Mapping[str, Any]) -> str:
    return json.dumps(doc, indent=2, sort_keys=False, ensure_ascii=False)


__all__ = [
    "SCHEMA", "COST_CLASSES", "LEGACY_KEYS", "COMPLETION_RISK_LABEL",
    "CertificateError", "LegacyFieldError", "BudgetUnitError", "EpochScopeError", "PopulationError", "PopulationMismatchError",
    "Epoch", "Budget", "Law", "OutputDomain", "Sampled", "Population", "PopulationPin", "SamplerPin", "Cost",
    "population_from_legacy_kind_table", "population_from_families", "population_from_population_file", "sampler_pin_from_record_v3", "certificate_from_record_v3", "render_refusal_text",
    "check_sampler_pin", "refuse_legacy_fields", "check_budget",
    "build_certificate", "render_text", "multi_epoch_integrity", "completion_risk", "schema_description", "dumps",
]
