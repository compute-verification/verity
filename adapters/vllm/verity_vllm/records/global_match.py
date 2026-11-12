"""The Match result (`cb-a/global-match/v1`): the global match's `global_match.json`, its verdict over the named checks G1..G8, the requests it
matched and the lag it read, as the Commit's account link (`check.executed_prefix.link_to_match_account`) reads it.

Its writer is `check.match.global_match` (`check`, and its CLI's `impl`; the per-check modules append to `checks` and set
`execution`, `achieved_concurrency`, `sampler_geometry`, `scheduled_live` and `lag`), also as a TP workload's per-rank Match
(`pipeline.tp.fold_match`).  A document refused before the checks ran (a Program of another schema) holds no `failed_checks`.
A check's evidence and a request's entry are each check's own (`check.match.match_run.NamedCheck.info`): each is kept as written
(`entry`, which `dump` writes back), with the fields a reader reads typed from it.

    gm = GlobalMatch.load(row_dir / "global_match.json")       # MissingField / UnknownField name the file and the field
    gm.check("G6").lengths_by_request
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.records.fields import RecordError, objects, only, optional, required

SCHEMA = "cb-a/global-match/v1"
#: what the account link doesn't interpret: the checker's provenance, the Program it read, the batch-decomposition evidence (`x09`) and the other criterion's
INFO = frozenset({"comparator", "rules_version", "utc", "oracle_dir", "log", "global_program", "chronology_unit", "request_id_convention",
                  "within_step_criterion", "within_step_note", "x09", "request_ids", "execution", "achieved_concurrency",
                  "sampler_geometry", "scheduled_live", "moe_construction", "eos_truncated_requests", "eos_verified", "seconds",
                  "alternate_verdict", "phases", "impl", "fold_form"})


def _strs(v: Sequence[Any] | None, where: str, name: str) -> tuple[str, ...] | None:
    if v is None:
        return None
    if not all(isinstance(x, str) for x in v):
        raise RecordError(where, name, "holds a value that is not a string")
    return tuple(v)


def _put(out: dict[str, Any], **kv: Any) -> dict[str, Any]:
    out.update({k: v for k, v in kv.items() if v is not None})
    return out


@dataclass(frozen=True)
class MatchCheck:
    id: str                                           # G1..G8
    status: str                                       # PASS | FAIL
    what: str
    problems: tuple[str, ...]                         # the first 20
    n_problems: int
    # G6's (chronology): recorder request id -> {served_output_len, executed_prefix_len, cap, cap_exhausted}; the other checks have none
    lengths_by_request: Mapping[str, Mapping[str, Any]] | None
    lag: Mapping[str, Any] | None                     # G6's: declared, record_attests, tokens_json_lag, exercised, variable_lag
    entry: Mapping[str, Any]                          # as written, the check's own evidence with it

    @classmethod
    def of(cls, doc: Mapping[str, Any], where: str, name: str) -> MatchCheck:
        return cls(id=required(doc, "id", where, str, f"{name}.id"), status=required(doc, "status", where, str, f"{name}.status"),
                   what=required(doc, "what", where, str, f"{name}.what"),
                   problems=_strs(required(doc, "problems", where, (list, tuple), f"{name}.problems"), where, f"{name}.problems"),
                   n_problems=required(doc, "n_problems", where, int, f"{name}.n_problems"),
                   lengths_by_request=optional(doc, "lengths_by_request", where, Mapping, f"{name}.lengths_by_request"),
                   lag=optional(doc, "lag", where, Mapping, f"{name}.lag"), entry=doc)

    def dump(self) -> dict[str, Any]:
        return dict(self.entry)


@dataclass(frozen=True)
class MatchRequest:
    request_id: str                                   # the Program's r<k> (the record's own id before `canonicalize_record`)
    workload_request_id: str | None                   # the workload's id; a request the Program doesn't declare has none
    index: int | None                                 # its manifest position, as above
    entry: Mapping[str, Any]                          # as written, the request's lengths and comparisons with it

    @classmethod
    def of(cls, doc: Mapping[str, Any], where: str, name: str) -> MatchRequest:
        return cls(request_id=required(doc, "request_id", where, str, f"{name}.request_id"),
                   workload_request_id=optional(doc, "workload_request_id", where, str, f"{name}.workload_request_id"),
                   index=optional(doc, "index", where, int, f"{name}.index"), entry=doc)

    def dump(self) -> dict[str, Any]:
        return dict(self.entry)


@dataclass(frozen=True)
class GlobalMatch:
    where: str                                        # the file it was read from, or what handed it over: a refusal names it
    verdict: str                                      # PASS | FAIL
    reasons: tuple[str, ...]
    checks: tuple[MatchCheck, ...]                    # empty when the document was refused before the checks ran
    requests: tuple[MatchRequest, ...]
    failed_checks: tuple[str, ...] | None             # a document refused before the checks ran has none
    lag: Mapping[str, Any] | None                     # G6's lag block (`check.match.chronology`); as above, and in older documents
    info: Mapping[str, Any]                           # the `INFO` fields it carries

    def check(self, check_id: str) -> MatchCheck | None:
        """The named check (`G6`), None when the document has none."""
        return next((c for c in self.checks if c.id == check_id), None)

    @classmethod
    def of(cls, doc: GlobalMatch | Mapping[str, Any], where: str) -> GlobalMatch:
        """The record of a Match result as its writer wrote it (`doc`, the parsed JSON); a record passes through."""
        if isinstance(doc, cls):
            return doc
        if not isinstance(doc, Mapping):
            raise RecordError(where, "schema", f"cannot be read: the Match result is a {type(doc).__name__}, not a JSON object")
        schema = required(doc, "schema", where, str)
        if schema != SCHEMA:
            raise RecordError(where, "schema", f"is {schema!r}, not {SCHEMA!r}")
        only(doc, {"schema", "verdict", "reasons", "checks", "requests", "failed_checks", "lag", *INFO}, where)
        return cls(where=where, verdict=required(doc, "verdict", where, str),
                   reasons=_strs(required(doc, "reasons", where, (list, tuple)), where, "reasons"),
                   checks=tuple(MatchCheck.of(c, where, f"checks[{i}]") for i, c in enumerate(objects(doc, "checks", where))),
                   requests=tuple(MatchRequest.of(r, where, f"requests[{i}]") for i, r in enumerate(objects(doc, "requests", where))),
                   failed_checks=_strs(optional(doc, "failed_checks", where, (list, tuple)), where, "failed_checks"),
                   lag=optional(doc, "lag", where, Mapping),
                   info={k: doc[k] for k in INFO if k in doc})

    @classmethod
    def load(cls, path: str | Path) -> GlobalMatch:
        with open(path) as f:
            return cls.of(json.load(f), str(path))

    def dump(self) -> dict[str, Any]:
        """The JSON its writer writes: every field it holds, an absent optional one left out."""
        out = {"schema": SCHEMA, **self.info, "checks": [c.dump() for c in self.checks], "requests": [r.dump() for r in self.requests],
               "verdict": self.verdict, "reasons": list(self.reasons)}
        return _put(out, failed_checks=None if self.failed_checks is None else list(self.failed_checks), lag=self.lag)
