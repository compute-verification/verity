"""What every part of `verity bench exfiltration` (`verity_bench.exfiltration`) reads by: the records, rows, inputs
and marks it names, the case and share it reads the figure at, and the modules of other benchmarks it loads."""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA = "exfiltration/v3"
#: the network certifier's document the command reads: its `value` the term of record, its rows' `term` by jitter
#: bound and their ε (`eps`), its inputs, and the label of what rests on a set-time card, at its top and on each row
#: that carries it
TERM_SCHEMA = "certifier-term/v2"
#: the network certifier's scopes the command prints
SCOPES = ("datacenter", "world")
#: the error a monitoring period's figure is to carry at most. Given no δ for the network certifier's scans, the
#: command takes the one at which the row it prints that sums the most execution phases carries this
PERIOD_ERROR = Fraction(1, 2 ** 20)
#: what a figure is printed as where its error, summed over everything it covers, is 1 or more: it carries no amount
NO_BOUND = "no bound at this period"
#: a network certifier's row's amounts, which a row with no bound doesn't carry
AMOUNTS = ("schedules", "cells", "jitter", "term", "entries")
#: a sum's two figures, the network certifier's term and the two parts together, each with the key of its error in
#: its scope's monitoring period
ERRORS = {"certifier_GB": "eps", "GB": "error"}
MODEL = "GLM-5.3"
#: the record whose runs hold the planner's rows, and the rows' files in a run's files: the driver's default command's
#: and its rows over the world's year alone
RECORD = "exfiltration-glm53-compute-audit"
ROWS_FILES = ("out/rows.jsonl", "out/three/rows.jsonl")
#: a row's plan in a run's files: the sampling plan the row is priced over at one audit budget (`beta`), with each
#: cell's block and its share of the plan's proving time (`cells`, and `left_out` for the cells it lists apart)
PLAN_FILE = "out/mix.json"
#: a plan's bound and its rows' at the plan's budget agree to this, or it is another row's plan
PLAN_TOLERANCE = 1e-3
#: the field in which the planner's driver names a row's variant, where it writes several of one row
VARIANT = "h8"
#: what makes rows one case of the plan over one scope, which one file of a run holds
CASE_KEYS = ("world", "row", "step", "scope", VARIANT)
#: how a block family is priced where the rows don't say
UNSTATED = "in no stated way"
#: the ways of the rows' `split.priced` whose proving time is measured: the planner's measured price, and a block
#: type's own steady-state row in `verity bench prove`'s table (`steady`); a family's row times another type and is
#: modeled, as is every other way
MEASURED_WAYS = ("measured", "steady")
#: what prices a plan's proving, by the planner's driver's names (`split.sources`): a block type's own steady-state
#: row, its Definition's at the same static K, and the planner's price
SOURCES = ("own row", "family row", "planner's price")
#: the source of each `split.basis` key, by its text before any colon; every other key is the planner's price, but
#: its copy stratum's model (`COPY_BASIS`), which is a source of its own
BASIS_SOURCES = {"steady": "own row", "family": "family row"}
COPY_BASIS = "headline.copy_stratum"
#: the planner's assumed default price, a key of `split.basis`
DEFAULT_BASIS = "reprice.DEFAULT"
#: printed where a row's plan records no source
NOT_RECORDED = "not recorded"
#: the share of total compute the figure is read at
SHARE = 0.10
#: the network check's own proving as a share of total compute, which the planner's rows don't count: no run measures it
NETWORK_CHECK_SHARE = 0.003
#: the case of the plan the figure is: the planner's world (the slowdown calculator's world fleet), the plan of record's row, with every
#: modeled lever at its measured value, and the step of that row that prices the verifier as measured (a prefix of the
#: step's name)
WORLD, ROW, STEP = "ours", "(ii) honest", "+ the verifier as measured"
#: the row the planner prices mostly at its own prices, and the label of its figures where its rows record no table's row
ASSUMED_ROW = "(i) honest"
ASSUMED_LABEL = "at the planner's assumed prices"
#: the most ids a response holds in the network certifier's class the figure is read at: its class that holds the
#: longest output the plan of record serves (assumed, as every class is: no run has certified one)
RESPONSE_IDS = 1024
#: a row's parts of total compute; `wait` is a part of `serve`
PARTS = ("serve", "wait", "prove", "replay", "pool", "edge", "compile")
#: the network certifier's rows the two parts are summed under, and what each is called beside the compute audit's
TOGETHER = {"scan": "as the scan's check charges it today",
            "all three": "with the lengths held by a stop block, the timing taken once and the entries charged through "
                         "the compute audit",
            "design, no stop block": "by the design no run has behind it, before a served audit has sampled a stop block",
            "design": "by the design no run has behind it, once a served audit has sampled a stop block"}
#: what a row of `TOGETHER` that the calculator reads from verdict records (its document's `stop`'s `rows`) is called
#: where the stop verdicts' reader's answer charges its stop class its whole schedule on some execution phase: there
#: nothing holds its lengths by a stop block, and no served audit's sample of one is read
WHOLE = {"all three": "with the timing taken once and the entries charged through the compute audit, its stop class "
                      "charged its whole schedule",
         "design": "by the design no run has behind it, its stop class charged its whole schedule"}
#: the network certifier's prover's module that reads the stop class's verdict records of the runs of `VERDICTS`:
#: its `of_runs` answers how the calculator's rows read from verdict records are charged (`_reader`)
READER = "stop_verdicts"
READER_MODULE = f"verity.protocols.resource_accounting.network.prover.{READER}"
#: the network certifier's row whose schedules are what a response's output count is charged while the server chooses
#: it: the design before a served audit has sampled a stop block
OUTPUT_COUNT = "design, no stop block"
GB = 8e9
#: an input whose value a run's verdicts give, each checked by a verifier no Lean proof covers yet, and how it prints
RUN_NOT_PROVED = "run, not proved"
PRINTED = {RUN_NOT_PROVED: "run (Python verifier, not proved)"}
#: a served run's proving read from a session's seconds, waiting included, and every share that sums it: an upper bound
AT_MOST = ", at most"
#: an input's marks: `missing` is a served run's field its record lacks, which the share it would measure is not claimed
#: without; `RUN_NOT_PROVED` is never `measured`, which a verdict is only where its record names the Lean guarantee
#: that proves its verifier (`proved_by`)
MARKS = ("measured", "modeled", "simulated", "assumed", "missing", RUN_NOT_PROVED, "measured" + AT_MOST,
         "modeled" + AT_MOST)
#: how the count of inputs names the upper bounds' marks
BOUNDS = {"measured" + AT_MOST: "measured upper bounds", "modeled" + AT_MOST: "modeled upper bounds"}
#: the record whose runs hold the verdicts on the stop class, each verdict's record one JSON object a line of
#: `VERDICTS_FILE` in a run's files
VERDICTS = "exfiltration-glm53-stop-verdicts"
VERDICTS_FILE = "out/verdicts.jsonl"
#: the four verdicts on a stop class (a verdict's `piece`), each accepted and complete, before the output count's 0 in
#: the sums that take it at 0 is marked a run's: one missing or not accepted, and the 0 stays assumed
PIECES = {"stop_record": "the stop record", "response_tie": "the response tie", "request_tie": "the request tie",
          "second_stage": "the compute audit's second stage"}
#: the negative controls (a verdict's `control`; null on the honest run): a run on the same path with one thing changed,
#: each refused by the verifier that accepted the honest run's verdict on the same piece
CONTROLS = {"id": "a changed id", "max_tokens": "a changed `max_tokens`"}
#: a verdict: accepted and complete, accepted and not complete (which moves no figure), refused, or withheld: the
#: verifier held back its last accept itself because the form it would accept is not yet proved (every SPOT proof
#: accepted, every check before the sample held, every tampered input refused), which is neither an accept nor a refusal
VERDICT_READS = ("accept", "not complete", "refuse", "withheld")
#: a verdict's record: its `piece` and `control`, the `run` that produced it, the `verifier` that checked it (its
#: `module` and `commit`), the `verdict`, `proved_by` (the Lean guarantee whose proof covers that verifier, null until
#: one does) and `open` (while `proved_by` is null, what stands between the verdict and proved, stated in words)
VERDICT_FIELDS = ("piece", "control", "run", "verifier", "verdict", "proved_by", "open")
VERIFIER_FIELDS = ("module", "commit")
#: what an honest verdict's record names of the class it is on: `registration`, the class's boundary record's digest;
#: `stream_registration`, its own stream's registration's digest; `stream_names_boundary`, the verifier's recorded
#: check that that registration names the boundary; `serve`, the served run's store id; `class`, ``w<window>/p<P>-o<O>``
#: (``-nostop`` for a class with no stop block); and `rate`, ``NUM/DEN`` in lowest terms, from the registration record.
#: A record lacking one keeps the class short of complete and is named; a control's are not read, since a control
#: changes a committed value and so its registration
CLASS_FIELDS = ("registration", "stream_registration", "stream_names_boundary", "serve", "class", "rate")
#: what the four pieces' honest verdicts agree on for their class to be complete
AGREED = ("registration", "serve", "class", "rate")
CLASS_NAME = re.compile(r"^w[0-9]+/p[0-9]+-o[0-9]+(-nostop)?$")
RATE_TEXT = re.compile(r"^([1-9][0-9]*)/([1-9][0-9]*)$")
#: the rate the row assumes: every stop and tie block proved
RATE = "1/1"
#: an issue's id, which the print never carries: `open` states what is open, and the issue is on the board (SHA-512
#: is a hash's name, not an issue)
ISSUE_ID = re.compile(r"\b(?!SHA-)[A-Z]+-[0-9]+\b")
#: the audit's GPU time on a served two-stage run, by part, which the share of compute it measures sums: what each part
#: is, and why a run's record lacks it where it does
SERVED_PARTS = {
    "prove": ("the sessions' proving", ("the two-stage audit's record keeps each session's proving seconds "
              "(`audit.timing`) and not the GPUs it held")),
    "staging": ("staging the sampled proof blocks for their sessions", ("the one-stage audit times its staging "
                "(`staged_s`) and its summary leaves the seconds out")),
    "replay": ("the replay of the sampled spans", "the two-stage audit reads no clock over its replay"),
    "interior": ("the interior commitment of the sampled spans' proof blocks",
                 "the two-stage audit reads no clock over its interior commitment"),
    "serving_charge": ("serving under Verity over stock vLLM serving the same load", ("no stock serve of the window's "
                       "load (`stock`) and no serving cost of the served arm over stock (`serving_cost`, "
                       "`verity bench serve_tax`)")),
}
#: the record's field that says which seconds a session's proving is (`prove`): ``busy``, the proof server's busy
#: seconds on a card, measured; or ``session``, a session's seconds, its wait behind the others of its prover included,
#: measured, at most
PROVE_SOURCE = "prove_seconds_source"
PROVE_SOURCES = {"busy": ("measured", "the proof server's busy seconds a card"),
                 "session": ("measured" + AT_MOST, "a session's seconds, its wait included: an upper bound")}
PROVE_SOURCE_GAP = ("the record doesn't say whether a session's proving seconds are the proof server's busy seconds or "
                    "the whole session's, its wait included (`audit.timing`'s `prove_seconds_source`)")
#: why a run's record lacks the window's serving, which the share is of
SERVING_GAP = ("the window's serve is not tied to the audit's record: its load's seconds and GPUs (serve_tax's "
               "`meta.json`: `wall_s`, `gpu`) are in the serve's own run")
#: what the command prints for the share a served run measures where no record is given
NO_RUN = "not measured: no served run's record"
#: what the command prints beside a served run's part whose entries are all on 0 GPUs, so its 0 doesn't read as GPU
#: time given free
ON_CPUS = "on CPUs: no GPU-seconds"
#: the inputs both parts read: the fleet and what a request of the plan is, which give the requests, and the weights
SHARED = ("world_machines", "datacenter_machines", "machine_gpus", "mfu_serve", "peak_flops", "request_tokens",
          "token_macs", "world_requests", "request_count", "weights")
#: the part of the inputs a row's binding bound assumes, and what the bound is called
SECURITY = "security level"
#: where the Lean instances a binding bound names are declared
LEAN_NUMBERS = "Claims.Flock.Numbers"
#: a Lean instance's numerals, as a binding bound names them (`lean.past`), in the bound's own symbols
NUMERALS = {"T": "calls a run", "e": "reruns a session (e_s)", "Ns": "sessions (N_s)", "Ks": "classes of gates (K_s)",
            "Km": "classes of gates, session by session (K_m)", "c": "calls of the run's own opening checks (c)"}
#: the Lean statement whose right-hand side a binding bound counted in calls is
LEAN_BOUND = "CompiledTwo.eventTwoX_le_cls_at_budget"
#: printed where no Lean theorem gives a number at a design's sizes
NOT_PROVED = "not proved at these sizes"
#: the figures a runner records under a research run (`record`), each over every monitoring period at the share: the
#: one the command calls the headline, the compute audit's part and the network certifier's
FIGURES = ("headline", "compute_audit", "network_certifier")
#: the document a run's result keeps beside its measurements, relative to the run's directory
DOCUMENT = "out/exfiltration.json"


def _load(root: Path, directory: str, module: str) -> Any:
    """A module of another benchmark's directory, which is no package: loaded by its file's path, once a process."""

    path = root / "benchmarks" / directory / f"{module}.py"
    if (held := sys.modules.get(module)) is not None and Path(getattr(held, "__file__", "")) == path:
        return held
    spec = importlib.util.spec_from_file_location(module, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"no module {module} at {path}")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[module] = loaded
    try:
        spec.loader.exec_module(loaded)
    except BaseException:
        del sys.modules[module]
        raise
    return loaded


def _reader() -> Any:
    """The stop verdicts' reader (`READER_MODULE`)."""

    return importlib.import_module(READER_MODULE)


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9)
