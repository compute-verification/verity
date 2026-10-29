"""The build summary (`build_summary.json`): each wrapper's derive (its outcome and Program digests) and the composed workload Program,
as the Commit's verdict reads the Build's `runtime-correspondence/v1` digest from it (`check.verdict.correspondence_of_build`).

Its writers lay it out three ways (`layout`):

* `tp1`: `pipeline.row_records.build_summary` and `add_workload` (the retired `row_pod.sh` before them), one entry per wrapper;
* `tp`: `pipeline.row_records.tp_build_summary` (the retired `row_pod_tp2.sh` before it), `tp.world`, the entries per rank under
  `ranks`, rank 0's step digest under `step`;
* `tp2-r19`: the retired `verity_capture/bench/tp2_evidence.py` (the regression harness's TP2 reference rows carry it), rank 0's
  entries as `step`/`request`, each rank's as `<wrapper>_rank<r>`, a digest per rank under `tp.ranks`, and its `cmd` and `note`.

The file carries no schema field: its writers predate one, so `SCHEMA` names the record in a refusal and `dump` writes none.  An
entry is kept as written (`entry`), with the fields a reader reads typed from it.

    s = BuildSummary.load(row_dir / "build_summary.json")      # MissingField / UnknownField name the file and the field
    s.wrappers["step"].correspondence_digest
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.records.fields import RecordError, only, optional, required

SCHEMA = "verity-vllm/build-summary/v1"
#: a request shape's derive: `build_request_` + `program.prefix_plan.shape_name`
SHAPE = r"build_request_LP\d+_T\d+(?:_C\d+)?"
#: a `tp1` Build's wrappers, and the composed `workload`
TP1_WRAPPER = re.compile(rf"^(?:step|request|workload|{SHAPE})$")
#: a `tp` Build's wrappers on one rank, named by their directories; the retired `row_pod_tp2.sh` globbed the `.log` and `.rc` beside
#: each request derive too, as entries saying `no result.json`
RANK_WRAPPER = re.compile(rf"^(?:build_step|workload|{SHAPE}(?:\.log|\.rc)?)$")
RANK = re.compile(r"^rank\d+$")
R19_WRAPPER = re.compile(r"^(?:step|request)(?:_rank\d+)?$")
ENTRY = frozenset({"ok", "unsupported", "exception", "message", "program_digest", "census_digest", "identity", "correspondence_digest",
                   "target", "applicability", "ir", "construction", "workload_digest", "mechanism", "artifact", "reason", "stop_rule"})
#: what `tp2_evidence.py` wrote per wrapper: the derive's result, and a B >= 2 rank's request Programs as its components
R19_ENTRY = frozenset({"ok", "unsupported", "exception", "message", "program_digest", "identity", "target", "applicability", "tp", "LP", "T", "M",
                       "wall_secs_total", "max_rss_mb", "sampler", "cmd", "program_digest_kind", "component_digests", "components",
                       "n_components", "declared_requests", "gp01_compose"})


@dataclass(frozen=True)
class WrapperBuild:
    ok: bool | None                                   # a `tp` Build's top-level `step` (rank 0's digest alone) has none
    program_digest: str | None                        # a failed derive, and the composed workload's entry (its digest is `workload_digest`), have none
    workload_digest: str | None                       # the composed workload's (`workload`); the derives have none
    # the Build's `runtime-correspondence/v1` digest: a Build made with the v1 frontend, a `tp` Build and the composed workload record none
    correspondence_digest: str | None
    entry: Mapping[str, Any]                          # as written

    @classmethod
    def of(cls, doc: Any, where: str, name: str, keys: frozenset[str] = ENTRY) -> WrapperBuild:
        if not isinstance(doc, Mapping):
            raise RecordError(where, name, f"is a {type(doc).__name__}, not a JSON object")
        only(doc, keys, where, f"{name}.")
        return cls(ok=optional(doc, "ok", where, bool, f"{name}.ok"),
                   program_digest=optional(doc, "program_digest", where, str, f"{name}.program_digest"),
                   workload_digest=optional(doc, "workload_digest", where, str, f"{name}.workload_digest"),
                   correspondence_digest=optional(doc, "correspondence_digest", where, str, f"{name}.correspondence_digest"), entry=doc)


def _wrappers(doc: Mapping[str, Any], where: str, prefix: str, names: re.Pattern[str], keys: frozenset[str] = ENTRY) -> dict[str, WrapperBuild]:
    for k in doc:
        if not names.match(str(k)):
            raise RecordError(where, prefix + str(k), f"is not a wrapper of the Build ({names.pattern})")
    return {k: WrapperBuild.of(v, where, prefix + k, keys) for k, v in doc.items()}


@dataclass(frozen=True)
class BuildSummary:
    where: str                                        # the file it was read from, or what handed it over: a refusal names it
    layout: str                                       # `tp1` | `tp` | `tp2-r19`
    wrappers: Mapping[str, WrapperBuild]              # the top-level entries: a `tp` Build's is `step` alone
    world: int | None                                 # the tensor-parallel world; a `tp1` Build has none
    ranks: Mapping[str, Mapping[str, WrapperBuild]] | None   # a `tp` Build's entries per rank (`rank<r>`); the other layouts have none
    r19: Mapping[str, Any] | None                     # a `tp2-r19` Build's `tp`, `cmd` and `note`, as written; the other layouts have none

    @classmethod
    def of(cls, doc: BuildSummary | Mapping[str, Any], where: str) -> BuildSummary:
        """The record of a build summary as its writer wrote it (`doc`, the parsed JSON); a record passes through."""
        if isinstance(doc, cls):
            return doc
        if not isinstance(doc, Mapping):
            raise RecordError(where, "step", f"cannot be read: the {SCHEMA} summary is a {type(doc).__name__}, not a JSON object")
        if "tp" not in doc:
            required(doc, "step", where, Mapping)
            required(doc, "request", where, Mapping)
            return cls(where=where, layout="tp1", wrappers=_wrappers(doc, where, "", TP1_WRAPPER), world=None, ranks=None, r19=None)
        tp = required(doc, "tp", where, Mapping)
        world = required(tp, "world", where, int, "tp.world")
        if "ranks" in tp:
            only(tp, {"world", "ranks", "all_ranks_ok"}, where, "tp.")
            required(tp, "ranks", where, Mapping, "tp.ranks")
            wrappers = {k: v for k, v in doc.items() if k not in ("tp", "cmd", "note")}
            required(doc, "step", where, Mapping)
            return cls(where=where, layout="tp2-r19", wrappers=_wrappers(wrappers, where, "", R19_WRAPPER, R19_ENTRY), world=world, ranks=None,
                       r19={k: doc[k] for k in ("tp", "cmd", "note") if k in doc})
        only(doc, {"tp", "ranks", "step"}, where)
        only(tp, {"world"}, where, "tp.")
        ranks = required(doc, "ranks", where, Mapping)
        for r in ranks:
            if not RANK.match(str(r)):
                raise RecordError(where, f"ranks.{r}", "is not a rank (rank<r>)")
        return cls(where=where, layout="tp", wrappers={"step": WrapperBuild.of(required(doc, "step", where, Mapping), where, "step")}, world=world,
                   ranks={r: _wrappers(required(ranks, r, where, Mapping, f"ranks.{r}"), where, f"ranks.{r}.", RANK_WRAPPER) for r in ranks},
                   r19=None)

    @classmethod
    def load(cls, path: str | Path) -> BuildSummary:
        with open(path) as f:
            return cls.of(json.load(f), str(path))

    def dump(self) -> dict[str, Any]:
        """The JSON its writer wrote (no schema field)."""
        out: dict[str, Any] = {k: dict(w.entry) for k, w in self.wrappers.items()}
        if self.layout == "tp":
            return {"tp": {"world": self.world}, "ranks": {r: {k: dict(w.entry) for k, w in ws.items()} for r, ws in (self.ranks or {}).items()}} | out
        return dict(self.r19 or {}) | out

    def correspondence_digests(self) -> dict[str, str]:
        """{wrapper: its runtime-correspondence/v1 digest}, for the top-level wrappers that record one."""
        return {k: w.correspondence_digest for k, w in self.wrappers.items() if w.correspondence_digest}
