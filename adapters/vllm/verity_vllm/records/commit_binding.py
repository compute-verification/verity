"""The Commit block's `binding`: one run's binding map checked against its population, as the Commit's rules read it
(`check.commit_rules`, `check.c2_rules`) and as the not-executed class's facts read it (`commit.executed_prefix.executed_prefix_facts`).

Its writers are `pipeline.commit.binding_record` (or, when that raised, only its error), a TP rank's `pipeline.tp.rank_worker` and
`pipeline.serve_seal` (a sealed window's map digests alone).  The sub-records are the ones those readers read: the identity coverage of
the population (`coverage`), on it the executed prefix of record (`commit.executed_prefix`), the prefix's link to the Match account
(`check.executed_prefix`) and to the Commit's token record (`check.commit_tokens`), the rows past the stop, and the population's
request count.  Their writers add fields often and the rules report many of them as written, so an unknown field is kept in `entry`,
never refused; a missing required field, or one of the wrong type, is refused by its dotted path.

    b = CommitBinding.of(commit["binding"], where)
    b.coverage.executed_prefix_of_record.requests["r0"].executed_prefix_len
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from verity_vllm.records.fields import RecordError, optional, required

NAME = "commit.binding"


def _object(doc: Any, where: str, name: str) -> Mapping[str, Any]:
    if not isinstance(doc, Mapping):
        raise RecordError(where, name, f"is a {type(doc).__name__}, not a JSON object")
    return doc


def _by_request(doc: Mapping[str, Any], key: str, where: str, kind: type, name: str, *, always: bool = True) -> Mapping[str, Any] | None:
    """`doc[key]`, a JSON object of request id -> a `kind` or null; required unless not `always`."""
    m = (required if always else optional)(doc, key, where, Mapping, name)
    for rid in m or ():
        optional(m, rid, where, kind, f"{name}.{rid}")
    return m


def _items(doc: Mapping[str, Any], key: str, where: str, kind: type, name: str) -> tuple | None:
    """`doc[key]`, a list of `kind` (an int is never a bool), when its writer wrote it."""
    v = optional(doc, key, where, (list, tuple), name)
    for i, x in enumerate(v or ()):
        if not isinstance(x, kind) or (isinstance(x, bool) and kind is not bool):
            raise RecordError(where, f"{name}[{i}]", f"is a {type(x).__name__}, not a {kind.__name__}")
    return None if v is None else tuple(v)


def _problems(doc: Mapping[str, Any], where: str, name: str) -> tuple[str, ...]:
    """`doc["problems"]`, which every writer writes: each problem by name, empty when it found none."""
    required(doc, "problems", where, (list, tuple), f"{name}.problems")
    return tuple(_items(doc, "problems", where, str, f"{name}.problems") or ())


@dataclass(frozen=True)
class ExecutedRequest:
    """One request's two lengths on the executed prefix of record."""
    cap: int
    executed_prefix_len: int
    served_output_len: int | None                     # the driver served the request no tokens
    last_served_token: int | None                     # likewise, or it served none
    eos_stop: bool | None                             # it served none, past an EOS or over its cap: a problem names it

    @classmethod
    def of(cls, doc: Any, where: str, name: str) -> ExecutedRequest:
        d = _object(doc, where, name)
        return cls(cap=required(d, "cap", where, int, f"{name}.cap"),
                   executed_prefix_len=required(d, "executed_prefix_len", where, int, f"{name}.executed_prefix_len"),
                   served_output_len=optional(d, "served_output_len", where, int, f"{name}.served_output_len"),
                   last_served_token=optional(d, "last_served_token", where, int, f"{name}.last_served_token"),
                   eos_stop=optional(d, "eos_stop", where, bool, f"{name}.eos_stop"))


@dataclass(frozen=True)
class ExecutedPrefix:
    """`commit.executed_prefix.executed_prefix_of_record`: every request's executed prefix under the Program's declared stop rule."""
    ok: bool
    problems: tuple[str, ...]
    requests: Mapping[str, ExecutedRequest]
    lag: int | None                                   # the Program declares none
    eos_of_record: tuple[int, ...] | None             # the Program declares no stop rule
    served_known: bool | None                         # the rules report these two, never decide on them: a record without them loads
    any_eos_stop: bool | None

    @classmethod
    def of(cls, doc: Any, where: str, name: str) -> ExecutedPrefix:
        d = _object(doc, where, name)
        reqs = required(d, "requests", where, Mapping, f"{name}.requests")
        return cls(ok=required(d, "ok", where, bool, f"{name}.ok"), problems=_problems(d, where, name),
                   requests={str(r): ExecutedRequest.of(v, where, f"{name}.requests.{r}") for r, v in reqs.items()},
                   lag=optional(d, "lag", where, int, f"{name}.lag"), eos_of_record=_items(d, "eos_of_record", where, int, f"{name}.eos_of_record"),
                   served_known=optional(d, "served_known", where, bool, f"{name}.served_known"),
                   any_eos_stop=optional(d, "any_eos_stop", where, bool, f"{name}.any_eos_stop"))


@dataclass(frozen=True)
class MatchTokensLeg:
    """The Match record's `capture/tokens.json`, as the Match-account link read it."""
    path: str
    sha256: str
    served_by_request: Mapping[str, int]              # each request's served length there
    tokens_equal: Mapping[str, bool | None]           # the Commit's served tokens equal the Match record's; null: the request isn't there

    @classmethod
    def of(cls, doc: Any, where: str, name: str) -> MatchTokensLeg:
        d = _object(doc, where, name)
        return cls(path=required(d, "path", where, str, f"{name}.path"), sha256=required(d, "sha256", where, str, f"{name}.sha256"),
                   served_by_request=_by_request(d, "served_by_request", where, int, f"{name}.served_by_request"),
                   tokens_equal=_by_request(d, "tokens_equal", where, bool, f"{name}.tokens_equal"))


@dataclass(frozen=True)
class GlobalMatchLeg:
    """The global match's `global_match.json`, as the Match-account link read it: its verdict and G6's account of each request."""
    path: str
    sha256: str
    verdict: str
    g6_status: str | None                             # the record has no G6 check
    requests_linked: tuple[str, ...] | None           # likewise
    served_by_request: Mapping[str, int | None] | None  # G6's `lengths_by_request`, each request's `served_output_len`; likewise

    @classmethod
    def of(cls, doc: Any, where: str, name: str) -> GlobalMatchLeg:
        d = _object(doc, where, name)
        lens = optional(d, "lengths_by_request", where, Mapping, f"{name}.lengths_by_request")
        served = None if lens is None else {
            str(r): optional(_object(v, where, f"{name}.lengths_by_request.{r}"), "served_output_len", where, int,
                             f"{name}.lengths_by_request.{r}.served_output_len") for r, v in lens.items()}
        return cls(path=required(d, "path", where, str, f"{name}.path"), sha256=required(d, "sha256", where, str, f"{name}.sha256"),
                   verdict=required(d, "verdict", where, str, f"{name}.verdict"), g6_status=optional(d, "g6_status", where, str, f"{name}.g6_status"),
                   requests_linked=_items(d, "requests_linked", where, str, f"{name}.requests_linked"), served_by_request=served)


@dataclass(frozen=True)
class MatchAccountLink:
    """`check.executed_prefix.link_to_match_account`: the executed prefix linked to the Match account of the same attempt."""
    ok: bool
    required: bool                                    # some request's prefix was reduced from its cap: only then does the link decide
    lag_attested: bool
    problems: tuple[str, ...]
    match_dir: str | None                             # no Match record beside the Commit (a config run)
    tokens_json: MatchTokensLeg | None                # no Match `capture/tokens.json` beside the Commit
    global_match: GlobalMatchLeg | None               # likewise

    @classmethod
    def of(cls, doc: Any, where: str, name: str) -> MatchAccountLink:
        d = _object(doc, where, name)
        tj, gm = d.get("tokens_json"), d.get("global_match")
        return cls(ok=required(d, "ok", where, bool, f"{name}.ok"), required=required(d, "required", where, bool, f"{name}.required"),
                   lag_attested=required(d, "lag_attested", where, bool, f"{name}.lag_attested"), problems=_problems(d, where, name),
                   match_dir=optional(d, "match_dir", where, str, f"{name}.match_dir"),
                   tokens_json=None if tj is None else MatchTokensLeg.of(tj, where, f"{name}.tokens_json"),
                   global_match=None if gm is None else GlobalMatchLeg.of(gm, where, f"{name}.global_match"))


@dataclass(frozen=True)
class CommitAccountLink:
    """`check.commit_tokens.account_links`: the executed prefix linked to the Commit's own token record."""
    ok: bool | None                                   # not applied (the row committed no sampled id): the Match link decides
    required: bool
    lag_attested: bool | None                         # likewise
    problems: tuple[str, ...]
    not_applied: str | None                           # why the link was not applied, when it was not
    record_path: str | None                           # no token record was read: not applied, or not built
    record_sha256: str | None                         # likewise
    served_by_request: Mapping[str, int | None] | None  # likewise
    tokens_committed: Mapping[str, bool] | None       # likewise

    @classmethod
    def of(cls, doc: Any, where: str, name: str) -> CommitAccountLink:
        d = _object(doc, where, name)
        rec = optional(d, "record", where, Mapping, f"{name}.record")
        return cls(ok=optional(d, "ok", where, bool, f"{name}.ok"), required=required(d, "required", where, bool, f"{name}.required"),
                   lag_attested=optional(d, "lag_attested", where, bool, f"{name}.lag_attested"), problems=_problems(d, where, name),
                   not_applied=optional(d, "not_applied", where, str, f"{name}.not_applied"),
                   record_path=None if rec is None else optional(rec, "path", where, str, f"{name}.record.path"),
                   record_sha256=None if rec is None else optional(rec, "sha256", where, str, f"{name}.record.sha256"),
                   served_by_request=_by_request(d, "served_by_request", where, int, f"{name}.served_by_request", always=False),
                   tokens_committed=_by_request(d, "tokens_committed", where, bool, f"{name}.tokens_committed", always=False))


@dataclass(frozen=True)
class RowsPastTheStop:
    """`commit.executed_prefix.rows_past_the_stop`: committed rows of a request past its executed prefix."""
    ok: bool
    entries_n: int
    steps_past_by_request: Mapping[str, int]
    error: str | None                                 # none past the stop and no padding inside the prefix
    padded_record: bool | None                        # a record from before padded Commits has neither
    padding_entries_n: int | None

    @classmethod
    def of(cls, doc: Any, where: str, name: str) -> RowsPastTheStop:
        d = _object(doc, where, name)
        return cls(ok=required(d, "ok", where, bool, f"{name}.ok"), entries_n=required(d, "entries_n", where, int, f"{name}.entries_n"),
                   steps_past_by_request=_by_request(d, "steps_past_by_request", where, int, f"{name}.steps_past_by_request"),
                   error=optional(d, "error", where, str, f"{name}.error"),
                   padded_record=optional(d, "padded_record", where, bool, f"{name}.padded_record"),
                   padding_entries_n=optional(d, "padding_entries_n", where, int, f"{name}.padding_entries_n"))


@dataclass(frozen=True)
class BindingCoverage:
    """`binding.coverage`: every identity of the population bound once (`commit.binding.coverage_check`), and on a single-GPU Commit
    the execution extent beside it."""
    ok: bool
    expected_n: int
    bound_n: int
    missing_n: int
    error: str | None                                 # a refusal by name: an incomplete manifest, a prefix that can't be placed, ...
    ambiguous_n: int | None                           # a record from before the one-unambiguous-binding check has none
    missing: tuple[Mapping[str, Any], ...] | None     # the first missing identities, as written; likewise
    ambiguous: tuple[Mapping[str, Any], ...] | None   # the first ambiguous ones; likewise
    executed_prefix_of_record: ExecutedPrefix | None  # a TP rank's coverage has none of the execution extent
    match_account_link: MatchAccountLink | None       # likewise
    commit_account_link: CommitAccountLink | None     # likewise
    rows_past_the_stop: RowsPastTheStop | None        # likewise
    entry: Mapping[str, Any]                          # as written

    @classmethod
    def of(cls, doc: Any, where: str, name: str) -> BindingCoverage:
        d = _object(doc, where, name)
        epr, ml, cl, past = (d.get(k) for k in ("executed_prefix_of_record", "match_account_link", "commit_account_link", "rows_past_the_stop"))
        return cls(ok=required(d, "ok", where, bool, f"{name}.ok"),
                   expected_n=required(d, "expected_n", where, int, f"{name}.expected_n"), bound_n=required(d, "bound_n", where, int, f"{name}.bound_n"),
                   missing_n=required(d, "missing_n", where, int, f"{name}.missing_n"), error=optional(d, "error", where, str, f"{name}.error"),
                   ambiguous_n=optional(d, "ambiguous_n", where, int, f"{name}.ambiguous_n"),
                   missing=_items(d, "missing", where, Mapping, f"{name}.missing"), ambiguous=_items(d, "ambiguous", where, Mapping, f"{name}.ambiguous"),
                   executed_prefix_of_record=None if epr is None else ExecutedPrefix.of(epr, where, f"{name}.executed_prefix_of_record"),
                   match_account_link=None if ml is None else MatchAccountLink.of(ml, where, f"{name}.match_account_link"),
                   commit_account_link=None if cl is None else CommitAccountLink.of(cl, where, f"{name}.commit_account_link"),
                   rows_past_the_stop=None if past is None else RowsPastTheStop.of(past, where, f"{name}.rows_past_the_stop"), entry=d)


@dataclass(frozen=True)
class BindingPopulation:
    """`binding.population`: the population the map was checked against."""
    source: str | None
    requests: int | None                              # the single-GPU binding's request count; a TP rank's lists the ids instead
    request_ids: tuple[str, ...] | None               # a TP rank's requests
    steps: int | None
    entry: Mapping[str, Any]                          # as written

    @classmethod
    def of(cls, doc: Any, where: str, name: str) -> BindingPopulation:
        d = _object(doc, where, name)
        reqs = optional(d, "requests", where, (int, list, tuple), f"{name}.requests")
        return cls(source=optional(d, "source", where, str, f"{name}.source"), requests=reqs if isinstance(reqs, int) else None,
                   request_ids=None if isinstance(reqs, int) else _items(d, "requests", where, str, f"{name}.requests"),
                   steps=optional(d, "steps", where, int, f"{name}.steps"), entry=d)


@dataclass(frozen=True)
class CommitBinding:
    """`commit.binding`: the run's binding map, checked."""
    error: str | None                                 # a built map has none; a failed one records only this, its site and rules version
    context_ok: bool | None                           # the map's context check: a failed binding, a skipped TP rank and a sealed window record none
    context_why: str | None                           # likewise
    program_digest: str | None                        # a failed binding has none
    collector: Mapping[str, Any] | None               # the instrumented runs' collector record; other runs have none
    coverage: BindingCoverage | None                  # a failed binding, a skipped TP rank and a sealed window have none
    population: BindingPopulation | None              # likewise
    entry: Mapping[str, Any]                          # as written

    @classmethod
    def of(cls, doc: CommitBinding | Any, where: str, name: str = NAME) -> CommitBinding:
        """The block as its writer wrote it (`doc`, the run's `commit.binding`); a record passes through."""
        if isinstance(doc, cls):
            return doc
        d = _object(doc, where, name)
        ctx = optional(d, "context", where, Mapping, f"{name}.context")
        cov, pop = d.get("coverage"), d.get("population")
        return cls(error=optional(d, "error", where, str, f"{name}.error"),
                   context_ok=None if ctx is None else optional(ctx, "ok", where, bool, f"{name}.context.ok"),
                   context_why=None if ctx is None else optional(ctx, "why", where, str, f"{name}.context.why"),
                   program_digest=optional(d, "program_digest", where, str, f"{name}.program_digest"),
                   collector=optional(d, "collector", where, Mapping, f"{name}.collector"),
                   coverage=None if cov is None else BindingCoverage.of(cov, where, f"{name}.coverage"),
                   population=None if pop is None else BindingPopulation.of(pop, where, f"{name}.population"), entry=d)
