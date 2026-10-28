"""The sampled replay's reconciliation against Q(P) (`population.query_checks`), which `check.c2_rules` grades the C2 population
scope from.

Its writers are the replay's coverage (`check.replay.coverage.query_population`) and the served check
(`check.replay.served_check.population`); both write every field typed here as required.  The block travels inside the
sampled-replay record (a Commit's `sampled_replay`, a rank's `sampled_replay.json`), so it is read from that record's parsed
JSON and has no file of its own.  The texts beside a finding (`why`, `note`) are not typed: nothing reads them.

    c = QueryChecks.of(sr["population"]["query_checks"], "the sampled replay's population.query_checks")
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from verity_vllm.records.fields import RecordError, objects, optional, present, required


@dataclass(frozen=True)
class Findings:
    """One check's findings: how many, and the first few as the writer wrote them."""
    n: int
    first: Sequence[Mapping[str, Any]]

    @classmethod
    def of(cls, doc: Mapping[str, Any], key: str, where: str, name: str | None = None) -> Self:
        name = name or key
        f = required(doc, key, where, Mapping, name)
        return cls(n=required(f, "n", where, int, f"{name}.n"), first=objects(f, "first", where, f"{name}.first"))

    @classmethod
    def if_written(cls, doc: Mapping[str, Any], key: str, where: str, name: str | None = None) -> Self | None:
        return None if doc.get(key) is None else cls.of(doc, key, where, name)


@dataclass(frozen=True)
class Unreachable(Findings):
    """The identities no `op_path_aliases` entry reaches, beside the alias table's size."""
    alias_entries: int

    @classmethod
    def of(cls, doc: Mapping[str, Any], key: str, where: str, name: str | None = None) -> Self:
        f = Findings.of(doc, key, where, name)
        return cls(n=f.n, first=f.first, alias_entries=required(doc[key], "alias_entries", where, int, f"{name or key}.alias_entries"))


@dataclass(frozen=True)
class Ranks:
    replay: Sequence[int]
    manifest: Sequence[int] | None                    # None: no manifest supplied to read them from
    equal: bool | None                                # None: no manifest supplied to compare with

    @classmethod
    def of(cls, doc: Mapping[str, Any], where: str) -> Ranks:
        r = required(doc, "ranks", where, Mapping)
        return cls(replay=required(r, "replay", where, (list, tuple), "ranks.replay"),
                   manifest=present(r, "manifest", where, (list, tuple), "ranks.manifest"),
                   equal=present(r, "equal", where, bool, "ranks.equal"))


@dataclass(frozen=True)
class ExecutedPrefixRule:
    applied: bool
    faults: Sequence[Mapping[str, Any]]               # {request_id, faults} for each request the rule faulted
    evaluated_past_stop: Findings
    manifest_cut_before_classification: Findings

    @classmethod
    def if_written(cls, doc: Mapping[str, Any], where: str) -> ExecutedPrefixRule | None:
        k = "executed_prefix_rule"
        if doc.get(k) is None:
            return None
        e = required(doc, k, where, Mapping)
        return cls(applied=required(e, "applied", where, bool, f"{k}.applied"), faults=objects(e, "faults", where, f"{k}.faults"),
                   evaluated_past_stop=Findings.of(e, "evaluated_past_stop", where, f"{k}.evaluated_past_stop"),
                   manifest_cut_before_classification=Findings.of(e, "manifest_cut_before_classification", where,
                                                                  f"{k}.manifest_cut_before_classification"))


@dataclass(frozen=True)
class QueryChecks:
    where: str                                        # what handed the block over: a refusal names it
    ok: bool | None                                   # None: no required manifest supplied, nothing reconciled
    vus_outside_query: Findings
    identities_without_rows: Findings
    rows_disagree: Findings
    ranks: Ranks
    accounted: bool | None                            # None: no manifest supplied to account against
    unstated_families: Sequence[str]
    # the replay's own checks: the served check has no alias table, row annotations, controller rows or executed-prefix
    # rule, and writes none of them
    identities_unreachable_by_alias: Unreachable | None
    rows_annotation_inconsistent: Findings | None
    controller_rows_disagree: Findings | None
    executed_prefix_rule: ExecutedPrefixRule | None
    # {field, why}: written only when the replay refused the manifest it was handed
    manifest_refused: Mapping[str, Any] | None
    # an older record's, which a record of record may still carry; no writer in this tree writes it
    missing_acquisition_families: Sequence[str] | None

    @classmethod
    def of(cls, doc: QueryChecks | Mapping[str, Any], where: str) -> QueryChecks:
        """The block as its writer wrote it (`doc`, the parsed JSON); a record passes through."""
        if isinstance(doc, cls):
            return doc
        if not isinstance(doc, Mapping):
            raise RecordError(where, "ok", f"cannot be read: the block is a {type(doc).__name__}, not a JSON object")
        return cls(where=where, ok=present(doc, "ok", where, bool),
                   vus_outside_query=Findings.of(doc, "vus_outside_query", where),
                   identities_without_rows=Findings.of(doc, "identities_without_rows", where),
                   rows_disagree=Findings.of(doc, "rows_disagree", where),
                   ranks=Ranks.of(doc, where),
                   accounted=present(doc, "accounted", where, bool),
                   unstated_families=required(doc, "unstated_families", where, (list, tuple)),
                   identities_unreachable_by_alias=Unreachable.if_written(doc, "identities_unreachable_by_alias", where),
                   rows_annotation_inconsistent=Findings.if_written(doc, "rows_annotation_inconsistent", where),
                   controller_rows_disagree=Findings.if_written(doc, "controller_rows_disagree", where),
                   executed_prefix_rule=ExecutedPrefixRule.if_written(doc, where),
                   manifest_refused=optional(doc, "manifest_refused", where, Mapping),
                   missing_acquisition_families=optional(doc, "missing_acquisition_families", where, (list, tuple)))
