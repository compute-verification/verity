"""What `verity bench prove layers` reads each layer into (`proof_layers`): its benchmark result (`Result`), the
error a reader raises where an output can't be found or read (`Unreadable`), and the names, outputs and words the
readers and the printer share."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

SCHEMA = "verity-bench/proof-layers/v2"
#: each layer's flag and its benchmark result's title, in the table's order
LAYERS = {"flock": "Flock", "zk": "Flock with zero knowledge", "sampling": "with sampling", "recursion": "with recursion"}
RECURSION, PROVE_PATH, A5, LAYERS_SH = "recursion_e2e", "prove_path", "a5", "layers.sh"
#: the outputs the benchmark result for a layer may be read from
ACCEPTS = {"flock": (PROVE_PATH, A5, LAYERS_SH), "zk": (PROVE_PATH, A5, LAYERS_SH), "sampling": (A5, PROVE_PATH, LAYERS_SH),
           "recursion": (RECURSION,)}
#: the directory of a run each layer's output is in, where the run's root holds none itself: a layers.sh run's
#: (`benchmarks/one_stage/layers.sh all`), or a prove_path served-arm run's (``flock/out``, ``zk/out``)
LAYERS_SH_DIRS = {"flock": "flock", "zk": "zk", "sampling": "sampled"}
#: the words a layers.sh output states its values are stand-ins in, which its benchmark result's workload opens
#: with
STAND_IN = "stand-in values, never served"
#: the proof a benchmark result for the layer must state it ran, whichever command wrote it: the sampling layer is
#: Flock with zero knowledge at the sampling plan
PROOF = {"flock": "plain", "zk": "zk", "sampling": "zk"}
#: the name the report writes its JSON under
REPORT = "proof-layers"
#: what a runner's metric id may name (`benchmarks/prove/run.py`'s ``LAYER_METRICS``), each with the unit it is written in
FIGURES = {"prover_s": "s a proof block", "lean_s": "s a proof block", "proof_bytes": "bytes a proof block"}
#: the layers each metric id holds, in the order they are written, each with its member: the sampling layer, the system
#: as served, is the id's own reading, written last with no member
MEMBERS = {"flock": "flock", "zk": "zero knowledge", "sampling": None}
#: a research run's id, which the store fetches
RUN_ID = re.compile(r"^r\d{8}-\d{6}-[0-9a-f]{4}$")
#: the files of a run's record the readers read, fetched from the store by these globs
FETCH = ("out/recursion_e2e.json", "out/*audit.log", "out/*summary.json", "out/summary.txt", "out/verdicts.txt",
         "out/machine.txt", "out/*a5.log", "*/out/layers.txt", "*/out/job.log", "*/out/a6/summary.json",
         "*/out/a6/session-*/lean-verify.log", "*/gpu-lease-usage.jsonl", "*/out/machine.txt", "*/out/ran-with.txt",
         "*/out/*audit.log", "*/out/*summary.json", "*/out/*-forged/commit.log")
NOT_STATED = "not stated"
#: what a benchmark result's workload opens with where its output states neither stand-in nor served values
VALUES_NOT_STATED = "values not stated"
#: what a run id that can't be fetched here takes instead
FIX = "give the run's out/ as a directory instead (where the store's read key is: `research fetch RUN`, then ~/.research/runs/RUN)"
#: the line a table prints unless every benchmark result ran on a deployed challenger's coins at deployed parameters, which this
#: command never takes as an audit: it checks no challenger's signature
NOT_AN_AUDIT = "a measurement, not an audit"
#: what a ``ran with:`` line may say a benchmark result's coins and parameters were, each only where its output says so outright
COINS = ("deployed challenger", "local challenger", "in-process")
PARAMETERS = ("deployed", "development")
#: a comma that ends a clause of recursion_e2e's ``inputs``, not one inside a Definition's braces
CLAUSE = re.compile(r",(?![^{]*\})")


class Unreadable(Exception):
    """A benchmark result's output can't be found or read."""


@dataclass
class Result:
    """One layer's benchmark result: what was given, what its output states, and why it has no number where it has none."""

    layer: str
    status: str = "not run"
    given: str | None = None
    format: str | None = None
    workload: str = "not run"
    lean: str = "not run"
    accepted: bool = False
    prover_s: float | None = None
    prover_unit: str = "s"
    per: str = ""
    device: str = NOT_STATED
    eps_delta: str = NOT_STATED
    prover_what: str = ""
    lean_s: float | None = None
    lean_what: str = ""
    proof_bytes: float | None = None
    bytes_what: str = ""
    per_block: bool = False
    ran_with: str = "ran with: not stated"
    detail: str | None = None
    proof: str | None = None
    why: str | None = None
    notes: list[str] = field(default_factory=list)
    #: the commit its output states it ran at, and which of its files states it
    commit: str | None = None
    commit_from: str | None = None
    #: a layers.sh benchmark result's figures, one a statement (its name, proof blocks, Flock's and Lean's seconds, its proof's bytes)
    statements: list[dict[str, Any]] = field(default_factory=list)

    def withhold(self, why: str) -> None:
        self.prover_s = self.lean_s = self.proof_bytes = None
        self.statements = []
        self.why = self.why or why

    def said(self, coins: str | None, parameters: str | None, detail: str | None) -> None:
        """The benchmark result's ``ran with:`` line from what its output states (`COINS`, `PARAMETERS`, None where it doesn't say),
        and `detail`, the output's own words, beside it."""

        assert coins in (None, *COINS) and parameters in (None, *PARAMETERS)
        self.detail = detail or None
        self.ran_with = ("ran with: not stated" if coins is None and parameters is None else
                         f"ran with: coins {coins or NOT_STATED}; parameters {parameters or NOT_STATED}"
                         + (f" ({detail})" if detail else ""))
