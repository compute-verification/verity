"""The Commit record: one run of a Commit pair, a line of `commit/runs.jsonl`, as the verdict reads it (`check.verdict.build` and
`execution_of_record`).

Its writers are `pipeline.commit` (`verity/commit-integ/delta/v1`; the retired `verity_capture/bench/commit_delta.py` before it),
`pipeline.tp.commit` (`verity/commit-integ/delta-tp/v1`; the retired `tp2_commit.py` before it) and `pipeline.serve_seal` (a sealed
serving window's run, with no schema field).  A top-level field none of them writes is refused by name.  The run's `commit` block
is each rule's evidence, which `check.commit_rules` reads as written (`CommitBlock.entry`); the fields the verdict reads beside the
rules are typed from it, and so is its binding map (`records.commit_binding`), whose sub-records the rules read.

    runs = CommitRun.load_lines(row_dir / "commit" / "runs.jsonl")   # a refusal names the file, the line and the field
    [r for r in runs if r.arm != "control" and r.commit is not None]
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.records.commit_binding import CommitBinding
from verity_vllm.records.fields import RecordError, only, optional, required

SCHEMA = "verity/commit-integ/delta/v1"
TP_SCHEMA = "verity/commit-integ/delta-tp/v1"
SCHEMAS = (SCHEMA, TP_SCHEMA)
#: every top-level field a writer writes, across every version a reader still meets
FIELDS = frozenset({
    "schema", "label", "workload", "workload_sha256", "case", "model", "pair", "order_in_pair", "arm", "committer", "committer_name",
    "wall_s", "iterations", "engine_steps", "generated_tokens", "prompt_tokens", "generated_tokens_per_s", "total_tokens_per_s",
    "step_wall_s", "step_wall_prefill_s", "step_wall_decode_mean_s", "tokens_equal_ref", "tokens_digest", "gpu_mem", "host_mem_kb",
    "utc", "remote_session", "fault_retained_flip", "root_equal_expected", "commit",
    "tp", "commit_started", "step_tokens_out", "thread_cpu_s",                       # pipeline.tp.commit
    "served_window",                                                                 # pipeline.serve_seal
    "tokens_equal",                                                                  # older runs: tokens_equal_ref's old name
})


def _within(doc: Mapping[str, Any], key: str, where: str, name: str) -> Mapping[str, Any]:
    """`doc[key]`, a JSON object when written, else empty."""
    return optional(doc, key, where, Mapping, name) or {}


@dataclass(frozen=True)
class CommitBlock:
    commit_pass: bool | None                          # null while the run's verdict waits on its sampled replay, or when a negative wasn't applied
    commit_fail_reason: str | None                    # a passing run has none
    run_root: str | None                              # a TP run's root is `tp_run_root`; a TP run that failed closed records neither
    rules_version: str | None                         # written from the binding: a TP run, a sealed window and a run whose binding failed have none
    classes: Mapping[str, Any] | None                 # the run's coverage declaration read back; TP runs and sealed windows record none
    openings_after_release: Mapping[str, Any] | None  # only a run opened after its release (`--open-after-release`)
    binding: CommitBinding | None                     # a TP run (its ranks' blocks carry theirs) and older runs have none
    binding_program_digest: str | None                # `binding.program_digest`: a run without a binding, or whose binding failed, has none
    weights_pin_program_digest: str | None            # `weights_pin.program_digest`: a run without a weights pin has none
    coverage_manifest_digest: str | None              # `coverage.manifest_digest`: a run without a required manifest has none
    collector: Mapping[str, Any] | None               # `binding.collector`, the instrumented runs' collector record; older runs have none
    entry: Mapping[str, Any]                          # as written

    @classmethod
    def of(cls, doc: Any, where: str) -> CommitBlock:
        if not isinstance(doc, Mapping):
            raise RecordError(where, "commit", f"is a {type(doc).__name__}, not a JSON object")
        b = doc.get("binding")
        binding = None if b is None else CommitBinding.of(b, where)
        return cls(commit_pass=optional(doc, "commit_pass", where, bool, "commit.commit_pass"),
                   commit_fail_reason=optional(doc, "commit_fail_reason", where, str, "commit.commit_fail_reason"),
                   run_root=optional(doc, "run_root", where, str, "commit.run_root"),
                   rules_version=optional(doc, "rules_version", where, str, "commit.rules_version"),
                   classes=optional(doc, "classes", where, Mapping, "commit.classes"),
                   openings_after_release=optional(doc, "openings_after_release", where, Mapping, "commit.openings_after_release"),
                   binding=binding, binding_program_digest=None if binding is None else binding.program_digest,
                   weights_pin_program_digest=optional(_within(doc, "weights_pin", where, "commit.weights_pin"), "program_digest", where, str,
                                                       "commit.weights_pin.program_digest"),
                   coverage_manifest_digest=optional(_within(doc, "coverage", where, "commit.coverage"), "manifest_digest", where, str,
                                                     "commit.coverage.manifest_digest"),
                   collector=None if binding is None else binding.collector, entry=doc)


@dataclass(frozen=True)
class CommitRun:
    where: str                                        # the file and line it was read from, or what handed it over: a refusal names it
    schema: str | None                                # a sealed serving window's run (`pipeline.serve_seal`) has none
    pair: int
    arm: str                                          # `control`, or the instrumented arm (`instrumented`, or the committer's name)
    committer: str | None                             # the control arm has none; a sealed window names it in `commit.committer`
    tokens_equal: bool | None                         # `tokens_equal_ref` (`tokens_equal` in older runs); a sealed window has no reference run
    commit: CommitBlock | None                        # the control arm has none
    entry: Mapping[str, Any]                          # as written

    @classmethod
    def of(cls, doc: CommitRun | Mapping[str, Any], where: str) -> CommitRun:
        """The record of one run as its writer wrote it (`doc`, the parsed JSON line); a record passes through."""
        if isinstance(doc, cls):
            return doc
        if not isinstance(doc, Mapping):
            raise RecordError(where, "pair", f"cannot be read: the Commit run is a {type(doc).__name__}, not a JSON object")
        only(doc, FIELDS, where)
        schema = optional(doc, "schema", where, str)
        if schema is not None and schema not in SCHEMAS:
            raise RecordError(where, "schema", f"is {schema!r}, not one of {', '.join(SCHEMAS)}")
        teq = optional(doc, "tokens_equal_ref", where, bool)
        commit = doc.get("commit")
        return cls(where=where, schema=schema, pair=required(doc, "pair", where, int), arm=required(doc, "arm", where, str),
                   committer=optional(doc, "committer", where, str),
                   tokens_equal=teq if teq is not None else optional(doc, "tokens_equal", where, bool),
                   commit=None if commit is None else CommitBlock.of(commit, where), entry=doc)

    @classmethod
    def load_lines(cls, path: str | Path) -> list[CommitRun]:
        """Every run in a `runs.jsonl`, one per non-blank line."""
        with open(path) as f:
            return [cls.of(json.loads(line), f"{path}:{n}") for n, line in enumerate(f, 1) if line.strip()]

    def dump(self) -> dict[str, Any]:
        """The JSON line its writer wrote."""
        return dict(self.entry)
