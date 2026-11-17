"""`verity bench memory`: the share of a node's memory read rate that one accepted PoUB session leaves uncertified, at
the data sheet's read bandwidth: the session's γ in its verifier's ledger at the specification's rate, as text and as
one line of JSON (`value`, `unit`, `commit`, `run`, `error`), exiting 0. A session that the Lean verifier didn't accept
(its record's `verdict` is not `lean`) prints no number and exits 1; a command line or a record that can't be read exits
2; and where no session ran to a verdict (`CouldNotRun`: no NVIDIA driver, no data sheet for the card, no VLLM_PY,
a checkpoint that can't be fetched, a session's processes that failed) it prints no number and exits 3.
The number is printed with its error, the ledger's δ for that γ summed over the session's windows, and the text says
how it reads (`ERROR_FORM`): that δ, and the term no δ holds. Where the error is 1 or more the session bounds nothing:
the command prints "no bound at this period" (`verity.protocol.NO_BOUND`) with the error and no number, and its
document's `value` is null with the reason under `no_bound`.

    verity bench memory PERIOD [--dry-run]        the node run: a gate run under the challenger's period PERIOD
    verity bench memory --from RUN [--session K]  a recorded run
    verity bench memory --loopback                both processes on this machine: a test of this command
    verity bench memory --local                   the node's session on one card of this machine, audited beside it

The node run is `benchmarks/poub_session/gate_run.sh PERIOD` with one honest session: the prover on the whole GPU
node serving ROLE (default Llama 3.2 1B as upstream holds it, the role core's pin names `LLAMA32_1B_BF16`), each
engine's weights arena the store, the verifier on the auditor's verifier host, the coins from the challenger host.
The verifier holds no checkpoint, so the store is the developer's choice, which the text and the JSON (`bound`) say.
PACED is the session (default seven credited stretches of a second, each pass
under a live coin of its own); UNPACED asks for a session of one coin a window in its place, as
WINDOWS:WINDOW_MS:SAMPLES (`7:60000:33273` is seven windows of a minute with 33,273 openings each). `gate_run.sh`'s
own variables (COMMIT, ARENA_GIB, UTIL, HOST_GIB, LEAF_CTAS, LEASE_BY, FETCH, ...) pass through: with HOST_GIB the
store holds that many GiB of the node's host memory beside the arenas, and with LEAF_CTAS the node's passes are the
leaf pass on that many reading CTAs. It waits for the fire run to reach the evidence store and
prints from it as `--from` does; `--dry-run` sends nothing to either gate and prints no number.

`--from RUN` reads the verifier's record of a run in the store: a gate run's fire run (its host run's `verifier.json`)
or a run of `poub_session/run.sh` that held the verifier. The number is the verifier's alone: nothing the prover
recorded is read, so the session is named by its index (`--session K`, where the run holds several) and not by what
the prover says it did. A withdrawn run is refused before any of it is fetched. The text says whether the verifier ran
on the auditor's verifier host, what the store was bound to, that a share of 1 or more certifies nothing, and, of a
run this command's `--loopback` left, that it was one.

`--loopback` runs `poub_session/run.sh` with ROLE=both and the prover the driver's CPU reference over a small test
store. No node is audited, so it certifies nothing, and its output says so.

`--local` is the node's session on one card of a machine of your own: `poub_session/run.sh` with ROLE=both,
SERVE=node and GPUS=1, a vLLM engine (VLLM_PY's) serving ROLE's pinned checkpoint with its weights in one arena of
most of the card's memory (`arena_of`), which is the store; the node prover reads the whole arena in paced, timed
passes beside the serving, and the Lean verifier and its coins run in processes beside them, in one session form
(`PACE`, refused otherwise). The checkpoint is fetched into LOCAL_WORK; with no `gpu-lease` installed, the lease is
the tree's own program, its locks in LOCAL_WORK unless the machine has a lease directory of its own (`LEASE_DIR`).
The run's records stay in LOCAL_WORK/runs/ (the research run's own directory under one). A session whose store is below `STORE_MIN` of the card's memory prints no number. The verifier
and the coins ran on the machine being measured, so the number is a measurement to improve on and not an audit: the
text says so first, and the JSON says it under `local`. It reads no evidence store, so the two references beside it
are left out with that reason.

Beside the number it prints two references, each from its run in the store with the run's date and card and the error
its record states: PoUS's state bound over a card's memory, and the share a wipe left foreign at most.
`references.json` names their runs; a reference that can't be read prints its reason in place of a number, and so
does one whose error is 1 or more. Under `research run` the printed document is
also `$RESEARCH_RUN_DIR/out/memory.json`, and outside one `$OUT/memory.json` where `OUT` is set (`kit.report`); a
run that ran `--loopback` keeps a loopback's document there whatever it prints afterwards, since that document is how
`--from` knows the run's session for a loopback.

There an accepted session of a node run or of `--local` also writes the run's benchmark result,
`$RESEARCH_RUN_DIR/result.json` (where `research` reads it) or `$OUT/result.json`, through the harness kit's writer
(`verity_bench.kit`), in the fields every resource's benchmark reports (`amounts`), each from the verifier's record
alone and named in its `inputs` by the record's key: `metric` (the printed number in its unit), the same number under
`METRIC`, the id this directory's `metrics.toml` holds it to on the console's Metrics tab, with its `TARGET` (and, unbound,
`"bound": false` and the premise in its verdict, which also names an error above 2^`ERROR_BOUND_LOG2`), `verified`
(the bytes the verifier certified read over the session's span), `capacity at READING` and `free at READING` for each
reading the
ledger holds (`READINGS`: the node's read rate times the span, and γ of it), and `error` (the probability the verified
amount doesn't hold). A session whose error is 1 or more writes `metric`, `verified` and each `free` with no value and
the reason as their `why`; its capacities and its error are written. A reading's rate is the verifier's own
figure for the node (`timing.py`), not the registered hardware's, and the capacity's unit says so. The references are
other runs' numbers and are not in the result; a loopback writes none, since it certifies nothing, nor does `--from`
of another run, whose numbers are that run's (`kit.report`'s `from_run`), and a session that
wasn't accepted writes none, as it prints no number.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

from verity.device import load_card, load_node
from verity.protocol import no_bound
from verity.protocols.resource_accounting.memory.poub.verifier import ledger as poub_ledger
from verity.protocols.resource_accounting.memory.poub.verifier import receipts, timing
from verity_bench import kit
from verity_config.settings import of

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RS = ROOT / "benchmarks" / "poub_session"

#: this directory's registry: PACED, UNPACED, and the variables a research run sets
SETTINGS = of(__file__)
#: UNPACED's word: the windows, a window's milliseconds (to three places, as the verifier host takes them) and the
#: openings a window
UNPACED = re.compile(r"([1-9]\d*):(\d+(?:\.\d{1,3})?):([1-9]\d*)")
UNIT = "share of the node's memory read rate left uncertified, at the data sheet's read bandwidth"
#: the number's metric on the console's Metrics tab (this directory's `metrics.toml`), and the goal's target for it
METRIC, TARGET = "memory-accounting/node-read-rate-uncertified", 0.14
#: a `--local` session's metric: a share of one card's read rate against its own sheet, another quantity from a node's
LOCAL_METRIC = "memory-accounting/one-machine-read-rate-uncertified"
LOCAL_UNIT = "share of one card's memory read rate left uncertified, at its own data sheet's read bandwidth"
PASS_METRIC = "memory-accounting/one-card-pass-time"
PASS_UNIT = "ms a pass: the calibration's median pass over the store, as the prover reports it"
READ_METRIC = "memory-accounting/one-card-bare-read-rate"
READ_UNIT = "TB a second: the calibration's bare read of the store on the card's memory, as the prover reports it"
#: the model ROLE serves, as the first word of a --local point's `member`
MEMBER_MODEL = "llama-3.2-1b"
#: how the number's error reads, in PoUB's ledger's words (its `DELTA_FORM`): δ is the session's, in its verifier's
#: ledger, and the windows' PRF advantages are in no δ
ERROR_FORM = "δ + Σ_t adv_t (SHA-512 stream as PRF: not priced, not assumed)"
ROLE = "LLAMA32_1B_BF16"
REFERENCES = HERE / "references.json"
#: a block of PoUB's store, in bytes
BLOCK_BYTES = 4096
#: the files of a run this command reads
#: the verifier's one-card node a --local run writes into its records before the session starts (its `--node`): the mark
#: of a --local run, read where the run died before its out/memory.json
NODE_FILE = "out/local-node.json"
RECORDS = ["out/verifier.json", "out/host-run.txt", "out/host/*/out/verifier.json", "out/memory.json", NODE_FILE]
LOOPBACK = ("loopback: both processes ran on this machine, the prover the CPU reference over a test store; it "
            "certifies nothing")
LOCAL = "local: the verifier and its coins ran on the machine being measured"
#: the node sheet (`hardware/nodes/`) of a --local session's card, by a word of the name nvidia-smi gives it, taken at
#: that one card: the share is of the one card's read rate (LOCAL_NODE names another sheet)
LOCAL_SHEETS = {"H200": "h200-sxm-141gb-x1", "RTX PRO 6000": "rtx-pro-6000-server-x8"}
#: what a --local result's digests are of: the session's verifier, and the data sheets its node is read from
VERIFIER = ROOT / "verity" / "protocols" / "resource_accounting" / "memory" / "poub" / "verifier"
SHEETS = ROOT / "hardware"
#: the one session form `--local` runs, the node run's paced session (PACED's default) as `run.sh` passes it to its
#: verifier: a coin a pass group, seven credited stretches of a second at 2773 openings each, the groups counted at
#: the pass the session's calibration measures on the card (`--pace-pass-ms 0`)
PACE = {"PACED": "1:1000:2773:7", "PACE_PASS_MS": "0"}
#: the machine's own lease directory (`gpu_lease.sh`'s default), which every lane's lease on the machine counts in
LEASE_DIR = Path("/run/gpu-lease")
#: `run.sh`'s windowed form, which the paced session replaces: --local refuses each of them set
WINDOWED = ("UNPACED", "SESSIONS", "WINDOWS", "WINDOW_MS", "SAMPLES")
#: the node run's engine and prover, with one engine (`gate_run.sh`'s ENGINE_ARGS and NODE_ARGS), and its verifier's
#: lateness and quantile (`gate.sh`'s)
ENGINE_ARGS, NODE_ARGS = "--step-hz 2 --align-steps", "--decode-s 60 --warm-openings 2773 --wait-s 1800"
LATENESS = {"JITTER_MS": "50", "QUANTILE": "max"}
#: vLLM's share of the card, and the GiB of it the arena leaves vLLM for its KV cache and activations: the node run's
#: 112 GiB of an H200's 141 and 72 of an RTX PRO 6000's 96 (`arena_of`); vLLM keeps VLLM_ROOM_GIB of its share for
#: the served model's cache and activations, and the prover's context takes what is outside the share
UTIL, VLLM_ROOM_GIB = 0.88, 12
#: the least share of the card's memory a --local store may be: a session that read less read a request's bytes and
#: not the card's, and its share is not this benchmark's
STORE_MIN = 0.5
#: the pin of the checkpoint ROLE serves: its repo, revision and files, fetched into LOCAL_WORK/hf
PIN = ROOT / "verity" / "protocols" / "resource_accounting" / "compute" / "pouw" / "verifier" / "lean" / "check" / "models.json"
FETCH = ("import sys; from huggingface_hub import snapshot_download as d; "
         "print(d(sys.argv[1], revision=sys.argv[2], allow_patterns=sys.argv[3:], max_workers=8))")
VERSIONS = "import importlib.metadata as m; print(m.version('vllm'), m.version('torch'))"
#: who flipped a session's coins, by its verifier's `--challenger`, in the kit's words (`kit.COINS`): its own challenger
#: or serve.py's on the verifier's machine, or the auditor device's through its gate
COINS = {"local": "local challenger", "serve": "local challenger", "gate": "deployed challenger"}
#: why `--local` prints no reference beside its number
NO_STORE = "a --local run reads no evidence store, and the references are other runs' numbers in ours"
#: `numbers_decided_by`: which of the printed numbers the Lean verifier decided and which are the Python ledger's, each
#: by its key in the session's ledger.  Where the Lean decided the share, its error is the sum of the three: the
#: windows' δ and AuditRO's slack (the reading's own error, held to the Lean's), and the sketch's term and the sample's
#: rejection and binding terms (`draw`), which no Lean computes yet
NUMBERS_LEAN = {"lean": ["spec.gamma", "spec.error.windows"],
                "python_ledger": ["declared.gamma.spec.error.sketch", "declared.gamma.spec.error.draw"]}
NOTHING = "a share of 1 or more certifies nothing of the node's read rate"
#: what an unbound session's number rests on, in its metric's verdict
HELD_NONE = "the verifier held no checkpoint"
#: a node's store that its verifier bound to no model, in the store's line
UNBOUND = f"{HELD_NONE}: the store's bytes are the developer's choice"
#: the goal's bound on the number's error, as log2: a metric whose error the record holds above it, or holds
#: none of, names that premise in its verdict
ERROR_BOUND_LOG2 = -100
ABOVE_BOUND = f"its error is above 2^{ERROR_BOUND_LOG2}"
#: the readings of the verifier's ledger, each the key it holds one under and the read rate it takes for the node: the
#: number is printed at the first
READINGS = {"spec": "the data sheet's read bandwidth", "measured": "the measured read rate"}
#: the unit of the result's amounts
BYTES = "bytes over the session's span"
#: where a capacity's rate comes from, which its unit says
OWN_RATE = "the verifier's own figure for the node (its timing.py), not the registered hardware's"
#: the result's error: which of the record's two it is, how the whole error reads (`ERROR_FORM`), and so what this
#: number leaves out
ERROR = ("probability that the verified amount doesn't hold, read window by window as the share is: the ledger's error "
         "with the sketch's term and the sample's rejection and binding terms added. The whole error reads "
         f"{ERROR_FORM}, this number its δ: the windows' PRF advantages, which the record names with no value, are "
         "not in it")
#: the wipe run's label states its bound in words
WIPE = re.compile(r"foreign <= ([\d,]+) bits\b.*\bM = ([\d,]+) B\b")
#: and the `eps` it holds except with, a power of two
WIPE_EPS = re.compile(r"\bat eps 2\^-(\d+)\b")
#: minutes the node run waits for a launched run's record: the fire run's own bound and its publish
WAIT_MIN = 9 * 60


class NoNumber(Exception):
    """The session has no number to print: it wasn't accepted, or it was and its record holds none (exit 1)."""


class CouldNotRun(NoNumber):
    """No session ran to a verdict: this machine lacks what it needs, or the session's processes failed (exit 3)."""


class Unreadable(Exception):
    """A run or a record can't be read, or the command line doesn't name one session."""


class EvidenceStore:
    """The evidence store as this command reads it: a run's attempt, some of its files, and a label on it."""

    def __init__(self) -> None:
        try:
            from research.store import labels_sync
            from research.store.local import LocalStore, StoreError
        except ImportError as e:
            raise Unreadable(f"the evidence store's client isn't installed here ({e})") from e
        self.store, self.sync, self.error = LocalStore(), labels_sync, StoreError

    def attempt(self, run: str) -> dict | None:
        try:
            return self.store.get_attempt(run)
        except self.error as e:
            raise Unreadable(f"the evidence store can't be read: {e}") from e

    def _labels(self, targets: list[str]) -> None:
        self.sync.read_through(self.store, targets, lambda m: print(f"memory: {m}", file=sys.stderr))

    def files(self, run: str, globs: list[str]) -> Path | None:
        """A directory holding the files of `run`'s record that match `globs`, at their paths in the run; None when
        none does."""
        a = self.attempt(run)
        if a is None:
            raise Unreadable(f"{run} is not in the evidence store this machine reads")
        art = (a.get("outputs") or {}).get("run_record") or (a.get("outputs") or {}).get("run_files")
        if not art:
            raise Unreadable(f"{run} kept no record")
        self._labels([run, art])
        if why := self.store.withdrawn(art) or self.store.withdrawn(run):
            raise Unreadable(f"{run} was withdrawn ({why}): nothing of it is fetched")
        try:
            if not self.store.listing(art, globs):
                return None
            return Path(self.store.fetch(art, paths=globs))
        except self.error as e:
            raise Unreadable(f"{run}'s record can't be fetched: {e}") from e

    def label(self, run: str, key: str) -> str | None:
        self._labels([run])
        labels = self.store.labels(run, key=key)
        return str(labels[-1].value) if labels else None


def commit_of(attempt: dict) -> str | None:
    source = attempt.get("source") or {}
    return f"{source['commit']}-dirty" if source.get("commit") and source.get("dirty") else source.get("commit")


def _at(ledger: dict, *keys: str) -> float | None:
    """The number the session's ledger holds under `keys`, None where it holds none there."""
    x = ledger
    for k in keys:
        x = x.get(k) if isinstance(x, dict) else None
    return None if isinstance(x, bool) or not isinstance(x, (int, float)) else x


def error_of(ledger: dict) -> float | None:
    """The session's error as its verifier's ledger declares it at the reading the number is printed at, read window
    by window as γ is: the windows' δ summed, with the sketch's term and the sample's rejection and binding terms.
    None where the record holds none."""
    return _at(ledger, "declared", "gamma", next(iter(READINGS)), "error", "windows")


def session(record: Path, k: int | None) -> dict:
    """Session `k` (the only one when `k` is None) of the verifier's record at `record`, as the printer reads it:
    `NoNumber` unless the Lean verifier accepted it (`accepted`, with `verdict` `lean`) and its ledger holds γ at the
    specification's rate, which is
    `poub_session/summary.py`'s `gamma_spec` of the session. Where the Lean's answer for γ's terms isn't written,
    the reading is undecided and there is no number, nor where the reading doesn't say the Lean decided it
    (`gamma_decided_by`).  `numbers_decided_by` names which of the printed numbers the
    Lean decided and which are the Python ledger's (`NUMBERS_LEAN`)."""
    try:
        v = json.loads(record.read_text())
        sessions = {s["k"]: s for s in v["sessions"]}
        blocks = v["registration"]["blocks"]
    except (OSError, KeyError, TypeError, ValueError) as e:
        raise NoNumber(f"the verifier's record doesn't read as one ({type(e).__name__}: {e})") from e
    if not sessions:
        raise NoNumber("the verifier recorded no session")
    held = ", ".join(f"{i} ({'accepted' if s.get('accepted') is True else 'not accepted'})"
                     for i, s in sorted(sessions.items()))
    if k is None and len(sessions) > 1:
        raise Unreadable(f"the run holds sessions {held}: name one with --session K")
    if k is not None and k not in sessions:
        raise Unreadable(f"the run holds sessions {held}, and no session {k}")
    vs = sessions[next(iter(sessions)) if k is None else k]
    if vs.get("accepted") is not True:
        why = (vs.get("failure") or {}).get("failure") or "no reason recorded"
        raise NoNumber(f"session {vs['k']} was not accepted ({why})")
    if vs.get("verdict") != "lean":
        raise NoNumber(f"session {vs['k']} is recorded accepted, but its record names no Lean verdict "
                       f"({vs.get('verdict')!r}): only the Lean verifier's accept has a number")
    ledger = vs.get("ledger") or {}
    spec = ledger.get("spec") or {}
    if (undecided := spec.get("undecided")) is not None:
        raise NoNumber(f"session {vs['k']} was accepted and its γ at the specification's rate is {undecided}")
    if (why := spec.get("refused")) is not None:
        raise NoNumber(f"session {vs['k']} was accepted and its γ at the specification's rate is refused: {why}")
    if spec.get("gamma_decided_by") != "lean":
        raise NoNumber(f"session {vs['k']} was accepted, and its record doesn't say the Lean verifier decided its γ (no "
                       "`gamma_decided_by`, which no record made before `Poub.Gamma` holds): the share it holds is the "
                       "Python ledger's, with no Lean answer for it recorded")
    gamma = spec.get("gamma")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise NoNumber(f"session {vs['k']} was accepted and its ledger holds no γ at the specification's rate")
    binding = v.get("binding") or {}
    return {"value": gamma, "gamma_decided_by": "lean", "numbers_decided_by": NUMBERS_LEAN,
            "session": vs["k"], "verdict": vs.get("verdict"),
            "error": error_of(ledger),
            "rate_bytes_per_s": (ledger.get("inputs") or {}).get("spec_bytes_per_s"),
            "store_bytes": blocks * BLOCK_BYTES,
            "node_gpu_bytes": (ledger.get("memory") or {}).get("hbm_bytes"),
            "model": binding.get("name") if binding.get("bound") is True else None,
            "bound": binding.get("bound") is True, "parameters": parameters(v, vs), "ledger": ledger,
            "calibration": v.get("calibration") or {}}


def parameters(v: dict, vs: dict) -> dict:
    """What session `vs` of the verifier's record `v` ran with, as its record states it: where its coins came from
    (the verifier's `--challenger`), the store's rows, the openings a window, the sample's term a window (`delta`),
    and the windows, their length and the lateness allowed, and the prover its error is priced against:
    `budget_log2`, the log2 of the SHA-512 compressions it may make on any hardware (`audit.QUERIES_LOG2`), and
    `passes` the passes the ledger counted in all. A paced session has no windows: its record's `windows` are the pass
    groups it ran (`pass_groups`) and its `window_ns` a placeholder, so it states neither as windows; its `pace` is
    its verifier's (`pace_of`: the passes a coin, a stretch's milliseconds and openings, the openings a group, the
    credited stretches, the groups a stretch and the pass they are counted at). None where the record states none."""
    def ns(x):
        return x / 1e6 if isinstance(x, (int, float)) and not isinstance(x, bool) else None

    passes = ((vs.get("ledger") or {}).get("inputs") or {}).get("passes")
    pace = vs.get("pace") if isinstance(vs.get("pace"), dict) else None
    form = ({"pace": pace, "pass_groups": vs.get("windows")} if pace else
            {"windows": vs.get("windows"), "window_ms": ns(vs.get("window_ns"))})
    return {"coins": (vs.get("challenger") or {}).get("challenger"), "rows": (v.get("registration") or {}).get("rows"),
            "samples": vs.get("samples"), "delta": vs.get("delta"), **form, "allowance_ms": ns(vs.get("allowance_ns")),
            "budget_log2": ((vs.get("ledger") or {}).get("sketch_failure") or {}).get("queries_log2"),
            "passes": passes if type(passes) is int else
            sum(passes) if isinstance(passes, list) and passes and all(type(x) is int for x in passes) else None}


def document(s: dict, commit: str | None, run: str | None, *, loopback: bool, host_run: str | None = None,
             local: bool = False, machine: dict | None = None) -> dict:
    """The printed document of session `s` (`session`), before its references: the number, its unit, the commit the
    session ran from and its run, then what it was measured on, its error among it. Where the error is 1 or more the
    number is None and `no_bound` says why (`protocol.no_bound`); else `no_bound` is None. `host_run` is the auditor's
    verifier host's run that held the verifier, None for a verifier that ran anywhere else. `local` is `LOCAL`'s words
    for a `--local` session, whose verifier ran on the machine it measured, else None, and `machine` that session's
    card, the node sheet its share is of, and the SHA-256 of the tree's data sheets and of its verifier (`digest`),
    which a result is held to `main`'s by. It carries the session's
    ledger as its verifier recorded it under `ledger`, which the command takes out before it prints (`amounts` reads
    it)."""
    why = None if s.get("error") is None else no_bound(s["error"])
    return {"value": None if why else s["value"], "unit": UNIT, "commit": commit, "run": run, "loopback": loopback,
            "local": LOCAL if local else None, "machine": machine, "gamma_decided_by": s["gamma_decided_by"],
            "numbers_decided_by": s["numbers_decided_by"], "host_run": host_run,
            "no_bound": why, **{k: v for k, v in s.items() if k not in ("value", "gamma_decided_by",
                                                                       "numbers_decided_by")}}


def left(tree: Path) -> dict:
    """The document this command left in the run whose records are under `tree`; empty when it left none."""
    path = tree / "out" / "memory.json"
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise Unreadable(f"the run's out/memory.json doesn't read as this command's document ({type(e).__name__}: {e})") from e
    if not isinstance(doc, dict):
        raise Unreadable("the run's out/memory.json doesn't read as this command's document (not an object)")
    return doc


def kind(doc: dict) -> str | None:
    """What document `doc` is of: "loopback", "local" (`--local`'s), or None, a node session's."""
    return "loopback" if doc.get("loopback") is True else "local" if doc.get("local") else None


def kind_of(tree: Path, run: str) -> str | None:
    """What `run`, whose records are under `tree`, was (`kind`), as the document this command left in it says. A
    document about another run, or one that doesn't say, says nothing of this run's session, and a `--local` run that
    died before its document (its `NODE_FILE`, which only `local()` writes) isn't printed as a node's either."""
    doc = left(tree)
    if not doc:
        if (tree / NODE_FILE).is_file():
            raise Unreadable(f"{run} holds --local's {NODE_FILE} but no out/memory.json saying what it was, so it "
                             "isn't printed as a node session")
        return None
    if doc.get("run") != run:
        raise Unreadable(f"{run}'s out/memory.json is the document of {doc.get('run')}, so what {run}'s own session "
                         "was isn't recorded")
    looped, local = doc.get("loopback"), doc.get("local")
    if not isinstance(looped, bool) or not (local is None or (isinstance(local, str) and local)) or (looped and local):
        raise Unreadable(f"{run}'s out/memory.json doesn't say what its session was (loopback {looped!r}, local "
                         f"{local!r}), so it isn't printed as a node session")
    return kind(doc)


def from_run(store, run: str, k: int | None) -> dict:
    """The printed document of recorded run `run`, before its references."""
    attempt = store.attempt(run)
    if attempt is None:
        raise Unreadable(f"{run} is not in the evidence store this machine reads")
    tree = store.files(run, RECORDS)
    verifier, host = (None if tree is None else tree / "out" / "verifier.json"), None
    if tree is not None and not verifier.is_file() and (named := tree / "out" / "host-run.txt").is_file():
        host = named.read_text().strip()
        verifier = tree / "out" / "host" / host / "out" / "verifier.json"
    if verifier is None or not verifier.is_file():
        raise CouldNotRun(f"{run} left no verifier's record (it ended rc {(attempt.get('status') or {}).get('rc')})")
    was = kind_of(tree, run)
    return document(session(verifier, k), commit_of(attempt), run, loopback=was == "loopback", host_run=host,
                    local=was == "local", machine=left(tree).get("machine"))


def from_dir(tree: Path, k: int | None) -> dict:
    """The printed document of the run whose record is the directory `tree` (`kit.record_dir`), read with nothing of
    ours: its `out/verifier.json` (or its verifier host's, under `out/host/`), the run and commit `job.json` names, or
    else the document this command left there (`out/memory.json`). `tree` may be the run's directory or its `out/`."""
    if tree.name == "out" and not (tree / "out").is_dir():
        tree = tree.parent
    try:
        job = json.loads((tree / "job.json").read_text()) if (tree / "job.json").is_file() else {}
    except (OSError, ValueError) as e:
        raise Unreadable(f"{tree}/job.json doesn't read as a run's record ({type(e).__name__}: {e})") from e
    mine = left(tree)
    run = (job.get("run") or {}).get("id") or mine.get("run")
    commit = (job.get("computation") or {}).get("tree_sha") or mine.get("commit")
    verifier, host = tree / "out" / "verifier.json", None
    if not verifier.is_file() and (named := tree / "out" / "host-run.txt").is_file():
        host = named.read_text().strip()
        verifier = tree / "out" / "host" / host / "out" / "verifier.json"
    if not verifier.is_file():
        raise CouldNotRun(f"{tree} holds no verifier's record (out/verifier.json)")
    if mine and run and mine.get("run") not in (None, run):
        raise Unreadable(f"{tree}'s out/memory.json is the document of {mine.get('run')}, not of {run}")
    was = kind_of(tree, mine.get("run") if mine else run or str(tree))
    return document(session(verifier, k), commit, run, loopback=was == "loopback", host_run=host,
                    local=was == "local", machine=mine.get("machine"))


def cores() -> tuple[str, str]:
    """The cores this process may use, split between the verifier and the prover (`run.sh`'s VERIFIER_CPUS and
    PROVER_CPUS)."""
    have = sorted(os.sched_getaffinity(0))
    half = max(1, len(have) // 2)
    return ",".join(map(str, have[:half])), ",".join(map(str, have[half:] or have))


def source_commit(run: Callable = subprocess.run) -> str | None:
    """The commit this tree is at: a `research run`'s source, else git's HEAD, marked when the tree has changes."""
    if sha := SETTINGS.RESEARCH_SOURCE_SHA:
        return sha
    head = run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    if head.returncode:
        return None
    dirty = run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return head.stdout.strip() + ("-dirty" if dirty else "")


def loopback(run: Callable = subprocess.run) -> dict:
    """One honest session through `run.sh` with ROLE=both and the CPU reference as the prover, read from its records.
    The variables `run.sh` reads pass through; the defaults here are a store of 64 blocks and three windows of a
    second, with seconds of lateness allowed each way, far past a CPU pass on a busy machine, so the session's checks
    decide and the machine's load doesn't."""
    held = SETTINGS.RESEARCH_RUN_DIR
    with tempfile.TemporaryDirectory() as tmp:
        verifier_cpus, prover_cpus = cores()
        env = {"CPU": "64", "WINDOWS": "3", "WINDOW_MS": "1000", "SAMPLES": "16", "JITTER_MS": "5000",
               "VERIFIER_CPUS": verifier_cpus, "PROVER_CPUS": prover_cpus, "PY": sys.executable,
               **SETTINGS.environ(), "RESEARCH_RUN_DIR": held or tmp, "ROLE": "both",
               "PLAN": SETTINGS.PLAN or "honest"}
        env.pop("SERVE", None)
        done = run(["bash", str(RS / "run.sh")], cwd=ROOT, env=env, stdout=sys.stderr)
        if done.returncode:
            if (why := refusal(Path(env["RESEARCH_RUN_DIR"]) / "out" / "verifier.json")) is not None:
                raise NoNumber(f"{why} (run.sh exited {done.returncode})")
            raise CouldNotRun(f"the loopback's processes failed (run.sh exited {done.returncode}); their logs are above")
        s = session(Path(env["RESEARCH_RUN_DIR"]) / "out" / "verifier.json", None)
    return document(s, source_commit(run), SETTINGS.RESEARCH_RUN_ID, loopback=True)


def gpu_of(run: Callable = subprocess.run) -> tuple[str, str]:
    """The name and compute capability nvidia-smi gives for GPU (the first card where it is unset)."""
    try:
        asked = run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader",
                     *(["-i", SETTINGS.GPU] if SETTINGS.GPU else [])], capture_output=True, text=True)
    except FileNotFoundError as e:
        raise CouldNotRun("--local serves on an NVIDIA GPU, and there is no nvidia-smi here, so no NVIDIA driver") from e
    if asked.returncode or not asked.stdout.strip():
        raise CouldNotRun(f"--local serves on an NVIDIA GPU, and nvidia-smi names none here (rc {asked.returncode}: "
                       f"{(asked.stderr or asked.stdout).strip()[-200:]})")
    name, _, cc = asked.stdout.strip().splitlines()[0].rpartition(",")
    return name.strip(), cc.strip()


def sheet_of(name: str) -> str:
    """The node sheet of card `name`: LOCAL_NODE where it is set, else `LOCAL_SHEETS`'."""
    if kind := SETTINGS.LOCAL_NODE:
        return kind
    if (kind := next((k for word, k in LOCAL_SHEETS.items() if word in name), None)) is None:
        raise CouldNotRun(f"the tree has no data sheet for {name} (hardware/nodes/ has "
                          f"{', '.join(LOCAL_SHEETS.values())} for its cards), and the share is of the card's own sheet: "
                          "add its card and a node of it there (hardware/README.md) and name the node with LOCAL_NODE; "
                          "a result against a sheet not on main is not comparable, and its digest says so")
    return kind


def one_card(kind: str) -> tuple[dict, int]:
    """The verifier's node for a session on one card of node sheet `kind`: its capacities (`ledger.Node.to_json`,
    which the verifier takes as `--node`) and its HBM (`--hbm-bytes`), from the tree's data sheets
    (`timing.Figures.of_sheets`) at one card, so the share is of that card's read rate and its host's. A host card
    that states no measured disk read is taken at its ceiling (`disk.read_max`); the figures no sheet states are
    `timing`'s."""
    node = load_node(kind)
    node = {**node, "device": {**node["device"], "count": {**node["device"]["count"], "value": 1}}}
    host = load_card(node["host"]["sku"])
    if "read" not in host.get("disk", {}):
        host = {**host, "disk": {**host["disk"], "read": host["disk"]["read_max"]}}
    figures = timing.Figures.of_sheets(node, load_card(node["device"]["sku"]), host)
    return poub_ledger.Node.stated(figures).to_json(), figures.hbm_bytes


def digest(path: Path) -> str:
    """SHA-256 over the files under `path` (a file or a directory), each by its path in the tree and its bytes'
    SHA-256, in path order; Markdown, and a directory whose name starts with `.` or is `__pycache__`, are left out."""
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix != ".md"
        and not any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(path).parts))
    h = hashlib.sha256()
    for p in files:
        h.update(f"{p.relative_to(ROOT)}\0{hashlib.sha256(p.read_bytes()).hexdigest()}\n".encode())
    return h.hexdigest()


def thirds() -> tuple[str, str, str]:
    """The cores this process may use, split among `run.sh`'s ENGINE_CPUS (the first quarter), PROVER_CPUS (the
    middle) and VERIFIER_CPUS (the last quarter); with fewer than four, the prover and the verifier share."""
    have = sorted(os.sched_getaffinity(0))
    q = len(have) // 4
    engine, verifier = have[:max(1, q)], have[len(have) - q:] if q else have[-1:]
    prover = have[q:len(have) - q] or have
    return tuple(",".join(map(str, c)) for c in (engine, prover, verifier))


def arena_of(hbm_bytes: int) -> tuple[int, float]:
    """The engine's weights arena in whole GiB and vLLM's share of the card: ARENA_GIB and UTIL where set, else
    floor(`UTIL` × the card's GiB) − `VLLM_ROOM_GIB`: 112 GiB on an H200."""
    try:
        util = float(SETTINGS.UTIL or UTIL)
        arena = int(SETTINGS.ARENA_GIB or math.floor(util * hbm_bytes / 2 ** 30) - VLLM_ROOM_GIB)
    except ValueError as e:
        raise Unreadable(f"ARENA_GIB is whole GiB and UTIL a share of the card ({e})") from e
    if not 0 < util <= 1 or arena < 1:
        raise Unreadable(f"the arena is {arena} GiB at UTIL {util}: ARENA_GIB is at least 1 and UTIL in (0, 1]")
    return arena, util


def checkpoint(work: Path, vllm_py: str, run: Callable) -> Path:
    """ROLE's pinned checkpoint in LOCAL_WORK/hf, its snapshot directory: the files the pin names that aren't there
    are fetched from the Hub with VLLM_PY's `huggingface_hub`."""
    pin = json.loads(PIN.read_text())[ROLE]
    hf = work / "hf"
    snap = hf / "hub" / f"models--{pin['repo'].replace('/', '--')}" / "snapshots" / pin["revision"]
    if missing := sorted(f for f in pin["files"] if not (snap / f).is_file()):
        done = run([vllm_py, "-c", FETCH, pin["repo"], pin["revision"], *missing],
                   env={**SETTINGS.environ(), "HF_HOME": str(hf), "HF_HUB_OFFLINE": "0"}, stdout=sys.stderr)
        if done.returncode:
            raise CouldNotRun(f"{pin['repo']}@{pin['revision'][:12]} ({', '.join(missing)}) couldn't be fetched into "
                              f"{hf} (rc {done.returncode}): Hugging Face can't be reached from here, or VLLM_PY has no "
                              "huggingface_hub")
    return snap


def local(run: Callable = subprocess.run) -> tuple[dict, Path]:
    """The node's session on one card of this machine: `run.sh` with ROLE=both, SERVE=node and GPUS=1, a vLLM engine
    serving ROLE's pinned checkpoint with its weights in one arena (`engine.py`), the store, which the node prover
    (`node.py`) reads whole in paced, timed passes, and the verifier beside them on cores of its own, in the one
    session form (`PACE`), read from its records. VLLM_PY is the vLLM interpreter; the checkpoint and the caches are
    kept in LOCAL_WORK, and on a machine with no lease of its own (no `gpu-lease` and no `LEASE_DIR`) the tree's
    lease's locks too, while a machine's own lease directory stays the one every lane there counts in. The run's records are kept, in
    the research run's directory under one and else in a directory of LOCAL_WORK/runs/, which is returned with the
    document. A session whose store is below `STORE_MIN` of the card's memory prints no number."""
    given = SETTINGS.environ()
    other = [f"{k}={given[k]}" for k, v in PACE.items() if given.get(k, v) not in (v, "")]
    if other := other + [f"{k}={given[k]}" for k in WINDOWED if given.get(k)]:
        raise Unreadable(f"--local runs one session form, PACED={PACE['PACED']} at PACE_PASS_MS={PACE['PACE_PASS_MS']} "
                         f"(a coin a pass group, seven credited stretches of a second at 2773 openings each, at the "
                         "pass the calibration measures), so its "
                         f"numbers compare: {', '.join(other)} is another")
    if not SETTINGS.VLLM_PY:
        raise CouldNotRun("--local serves with vLLM, and VLLM_PY names no vLLM interpreter: make one (`uv venv "
                          "VENV && uv pip install --python VENV/bin/python vllm`) and set VLLM_PY=VENV/bin/python")
    name, _ = gpu_of(run)
    kind, work = sheet_of(name), Path(SETTINGS.LOCAL_WORK).expanduser()
    node, hbm_bytes = one_card(kind)
    arena, util = arena_of(hbm_bytes)
    work.mkdir(parents=True, exist_ok=True)
    snap = checkpoint(work, SETTINGS.VLLM_PY, run)
    engine_cpus, prover_cpus, verifier_cpus = thirds()
    held = SETTINGS.RESEARCH_RUN_DIR
    if not held:
        (work / "runs").mkdir(parents=True, exist_ok=True)
        held = tempfile.mkdtemp(prefix="run-", dir=work / "runs")
    (Path(held) / NODE_FILE).parent.mkdir(parents=True, exist_ok=True)
    (Path(held) / NODE_FILE).write_text(json.dumps(node, sort_keys=True))
    env = {"ENGINE_CPUS": engine_cpus, "PROVER_CPUS": prover_cpus, "VERIFIER_CPUS": verifier_cpus, **LATENESS,
           **given, **PACE, "ROLE": "both", "SERVE": "node", "GPUS": "1", "PLAN": SETTINGS.PLAN or "honest",
           "MODEL": str(snap),
           "NODE_WORKSPACE": str(work), "HF_HOME": str(work / "hf"), "UV_CACHE_DIR": str(work / "cache" / "uv"),
           "PIP_CACHE_DIR": str(work / "cache" / "pip"), "XDG_CACHE_HOME": str(work / "cache"),
           "RESEARCH_RUN_DIR": held,
           "ENGINE_ARGS": " ".join(filter(None, [ENGINE_ARGS, f"--arena-gib {arena} --util {util:g}",
                                                 given.get("ENGINE_ARGS")])),
           "NODE_ARGS": " ".join(filter(None, [NODE_ARGS, f"--model-dir {shlex.quote(str(snap))}",
                                               f"--node local/{kind}/serving", given.get("NODE_ARGS")])),
           "VERIFIER_ARGS": " ".join(filter(None, [given.get("VERIFIER_ARGS"), "--node",
                                                   shlex.quote(f"{held}/{NODE_FILE}"), "--hbm-bytes", str(hbm_bytes)]))}
    if shutil.which("gpu-lease") is None and not LEASE_DIR.is_dir():
        env.setdefault("GPU_LEASE_DIR", str(work / "gpu-lease"))
    print(f"memory: --local serves {ROLE} on {name} with a {arena} GiB arena of its {hbm_bytes / 2 ** 30:.0f} GiB, the "
          f"verifier's node one card of {kind}; its records go to the directory on the next line, which `verity bench "
          f"memory --from` reads back\n  {held}/out", file=sys.stderr)
    done = run(["bash", str(RS / "run.sh")], cwd=ROOT, env=env, stdout=sys.stderr)
    if done.returncode:
        if (why := refusal(Path(held) / "out" / "verifier.json")) is not None:
            raise NoNumber(f"{why} (run.sh exited {done.returncode}); its logs are above and in {held}/out")
        raise CouldNotRun(f"the node's session failed (run.sh exited {done.returncode}); its logs are above and in "
                          f"{held}/out")
    s = session(Path(held) / "out" / "verifier.json", None)
    if s["store_bytes"] < STORE_MIN * hbm_bytes:
        raise NoNumber(f"the store is {_gb(s['store_bytes'])}, {s['store_bytes'] / hbm_bytes:.3f} of the card's "
                       f"{_gb(hbm_bytes)}, below the {STORE_MIN} this benchmark reads: the session read a request's "
                       "bytes and not the card's memory, so its share is not this benchmark's")
    asked = run([SETTINGS.VLLM_PY, "-c", VERSIONS], capture_output=True, text=True)
    versions = asked.stdout.split() if asked.returncode == 0 else []
    machine = {"card": name, "sheet": kind, **engine_args(Path(held)), "sheet_sha256": digest(SHEETS),
               "verifier_sha256": digest(VERIFIER),
               **dict(zip(("vllm", "torch"), versions if len(versions) == 2 else (None, None)))}
    return (document(s, source_commit(run), SETTINGS.RESEARCH_RUN_ID, loopback=False, local=True, machine=machine),
            Path(held))


def refusal(record: Path) -> str | None:
    """The verdict of a session the verifier refused, from its record at `record`: which session, where (its pass
    group, or its window where the session is not paced), and the check that failed at which sample (pass,
    position). None where the verifier recorded no refusal, which is no verdict: `run.sh` failed some other way."""
    if not record.is_file():
        return None
    try:
        sessions = json.loads(record.read_text())["sessions"]
        vs = next((s for s in sessions if s.get("accepted") is False), None)
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as e:
        raise CouldNotRun(f"the verifier's record doesn't read as one ({type(e).__name__}: {e})") from e
    if vs is None:
        return None
    failure = vs.get("failure") if isinstance(vs.get("failure"), dict) else {}
    where = "" if failure.get("t") is None else f" at {'pass group' if vs.get('pace') else 'window'} {failure['t']}"
    at = failure.get("at")
    sample = f" at pass {at[0]}, position {at[1]}" if isinstance(at, list) and len(at) == 2 else ""
    return (f"the verifier refused session {vs.get('k')}{where}: its check `{failure.get('failure') or 'unrecorded'}` "
            f"failed{sample}")


def engine_args(held: Path) -> dict:
    """The arena's GiB, vLLM's share of the card and the engine's steps a second at most as the session's engine
    recorded them (`out/engine.0.json`'s `args`), and `busy`, the share of the session's phases its steps took
    (`out/engine.0.steps.jsonl`); None where it states none."""
    path, steps = held / "out" / "engine.0.json", held / "out" / "engine.0.steps.jsonl"
    if not path.is_file():
        return {"arena_gib": None, "util": None, "step_hz": None, "busy": None}
    try:
        record = json.loads(path.read_text())
        xs = [x for x in map(json.loads, steps.read_text().splitlines()) if x["phase"].startswith("session-")] \
            if steps.is_file() else []
    except (OSError, ValueError, KeyError, AttributeError) as e:
        raise Unreadable(f"the session's engine record {path} doesn't read ({type(e).__name__}: {e})") from e
    args = record.get("args") if isinstance(record, dict) and isinstance(record.get("args"), dict) else {}
    span = xs[-1]["t"] - xs[0]["t"] + xs[-1]["ms"] / 1e3 if xs else 0
    return {"arena_gib": args.get("arena_gib"), "util": args.get("util"), "step_hz": args.get("step_hz"),
            "busy": sum(x["ms"] for x in xs) / 1e3 / span if span > 0 else None}


def session_form() -> dict[str, str]:
    """The node run's one session as `gate_run.sh`'s variables: UNPACED's session of one coin a window where it is
    set, else PACED's paced one."""
    unpaced = SETTINGS.UNPACED
    if unpaced is None:
        paced = SETTINGS.PACED
        if len(paced.split()) != 1:
            raise Unreadable(f"PACED is {paced!r}: the bench runs one session, one word")
        return {"PACED": paced}
    if SETTINGS.is_set("PACED"):
        raise Unreadable("UNPACED and PACED are both set: the bench runs one session, in one form, so unset one")
    if not (m := UNPACED.fullmatch(unpaced)):
        raise Unreadable(f"UNPACED is {unpaced!r}: the bench runs one session of one coin a window, one word "
                         "WINDOWS:WINDOW_MS:SAMPLES (7:60000:33273 is seven windows of a minute with 33,273 openings "
                         "each)")
    return dict(zip(("SESSIONS", "WINDOWS", "WINDOW_MS", "SAMPLES"), ("1", *m.groups())))


def node(store, period: str, dry: bool, k: int | None, run: Callable = subprocess.run,
         wait: Callable[[float], None] = time.sleep) -> dict | int:
    """The gate run of one honest session under `period`, then its printed document from the store; `gate_run.sh`'s
    exit status when it launched nothing or ran dry."""
    env = {"ROLE": ROLE, **SETTINGS.environ(), "PLAN": "honest", **session_form()}
    rc = run(["bash", str(RS / "gate_run.sh"), *(["--dry-run"] if dry else []), period], cwd=ROOT, env=env,
             stdout=sys.stderr).returncode
    if dry:
        print("memory: a dry run: nothing reached either gate, and there is no number", file=sys.stderr)
        return rc
    if rc:
        raise CouldNotRun(f"the gate run did not launch (gate_run.sh exited {rc})")
    state = Path.home() / ".research" / "gate-runs" / f"{period}.env"
    ids = dict(line.split("=", 1) for line in state.read_text().splitlines() if "=" in line)
    fire, prover = ids["fire"], ids["prover"]
    print(f"memory: gate run {period}: fire {fire}, prover {prover}. Waiting for the fire run to reach the store; if "
          f"this is interrupted, `verity bench memory --from {fire}` prints once it has", file=sys.stderr)
    for _ in range(WAIT_MIN):
        if store.attempt(fire) is not None:
            return from_run(store, fire, k)
        wait(60)
    raise CouldNotRun(f"{fire} left no record in the store within {WAIT_MIN} minutes")


def _gb(n: float) -> str:
    return f"{n / 1e9:,.2f} GB"


def pous(store, run: str) -> dict:
    """A PoUS session's state bound over the card's memory, from its run's summary. `s_bits` is S₀ = ⌊ρ·|C|⌋, a
    lower bound on the device's memory, its on-chip state and a round of side-channel traffic together: it is not
    the device memory credited. PoUS's statement of the memory held over a monitoring period
    (`security/Properties/Pous/`) credits S over a window of `len` ticks where S·len + S_chip + c ≤ S₀, c the side
    channels' capacity over a round, and no credited figure is printed, since no record holds the window and no
    card file a side channel's capacity."""
    a = store.attempt(run) or {}
    tree = store.files(run, ["out/summary.json"])
    if tree is None:
        raise Unreadable(f"{run} holds no out/summary.json")
    d = json.loads((tree / "out" / "summary.json").read_text())
    if d["certificate"].get("certified") is not True:
        raise Unreadable(f"{run}'s record certifies nothing")
    held, total = d["certificate"]["s_bits"] / 8, d["held"]["total_bytes"]
    eps = d["certificate"].get("eps")
    return {"value": held / total, "unit": "PoUS's state bound over the card's memory: s_bits is S0 = floor(rho*|C|), "
            "a lower bound on the device's memory, its on-chip state and a round of side-channel traffic together, "
            "not the device memory credited",
            "detail": f"{_gb(held)} of {_gb(total)}", "error": None if eps is None else float(Fraction(eps)),
            "card": d["held"].get("device") or (a.get("conditions") or {}).get("hw.sku")}


def wipe(store, run: str) -> dict:
    """The share of the card's memory a wipe left foreign at most, from the label that states the run's bound, with
    the `eps` the label states it at."""
    a = store.attempt(run) or {}
    text = store.label(run, "label")
    m = WIPE.search(text or "")
    if m is None:
        raise Unreadable(f"{run}'s label doesn't state a residue: {text!r}")
    bits, memory = (int(x.replace(",", "")) for x in m.groups())
    eps = WIPE_EPS.search(text)
    return {"value": bits / 8 / memory, "unit": "share of the card's memory a wipe leaves foreign, at most",
            "detail": f"{_gb(bits / 8)} of {_gb(memory)}", "error": None if eps is None else 2.0 ** -int(eps.group(1)),
            "card": (a.get("conditions") or {}).get("hw.sku")}


READERS = {"PoUS state bound": pous, "wipe residue": wipe}
#: what the text says beside a reference's certified bits: PoUS's reach the card's memory through the device model its
#: statement reads, not through the answer, until an answer names its tier and leaf
THROUGH = {"PoUS state bound": "through the device model, not the answer"}


def beside(store_of: Callable[[], object]) -> list[dict]:
    """Each reference of `references.json` read from its run with the error its record states, or why it can't be.
    A reference is one run's number, so its error is that run's; at 1 or more it states no number
    (`protocol.no_bound`)."""
    out, store = [], None
    for name, run in json.loads(REFERENCES.read_text()).items():
        row = {"name": name, "run": run}
        try:
            store = store if store is not None else store_of()
            row.update(READERS[name](store, run))
            row["date"] = (((store.attempt(run) or {}).get("execution") or {}).get("start_utc") or "")[:10] or None
            if row["error"] is not None and (why := no_bound(row["error"])) is not None:
                row.update(value=None, no_bound=why)
        except Unreadable as e:
            row.update(value=None, why=str(e))
        except Exception as e:  # noqa: BLE001  (whatever a reference fails with, the session's number still prints)
            row.update(value=None, why=f"{type(e).__name__}: {e}")
        out.append(row)
    return out


def amounts(doc: dict, ledger: dict, metric: str = METRIC) -> list[dict]:
    """The result a printed document `doc` and its session's `ledger` give in the fields every resource's benchmark
    reports, as the kit writer's measurements, each from the verifier's record alone; its `inputs` are the recorded
    run and the keys of the session's entry in that record it was read from.

    - `"metric"`: the printed number in its unit.
    - `metric`: the same number, the Metrics tab's metric (this directory's `metrics.toml`; `METRIC`, or `LOCAL_METRIC`
      for a `--local` session), a node's with `TARGET` and where the number and its error stand against it, a
      `--local` session's with no target and its error in its unit (`LOCAL_UNIT`), and its `extra` `audit` false
      where the session's coins or parameters aren't deployed (`ran`). A session whose verifier held no checkpoint still writes the metric, with `"bound": false` and
      its verdict naming that premise (`HELD_NONE`); a bound session's metric holds neither. Where the number's error is
      above 2^`ERROR_BOUND_LOG2` the verdict names that too (`ABOVE_BOUND`): the record's declared error at the first
      reading where it holds one, else the ledger's own, and above it where the record holds neither.
    - `verified`: the bytes the verifier certified read over the session's span, its own `certified_bytes` at the
      reading the number is printed at; none at a share of 1 or more, which certifies nothing.
    - `capacity at READING`, for each reading the ledger holds (`READINGS`): the reading's read rate times the
      session's span, the whole the verifier takes γ's terms as fractions of. The rate is the verifier's own figure
      for the node, not the registered hardware's, and the unit says so.
    - `free at READING`: γ of that capacity, so free over capacity is the reading's γ, and at the first reading the
      printed number; all of it at a share of 1 or more. Verified and free are the capacity but for the float's last
      place.
    - `error`: the probability the verified amount doesn't hold, read window by window as γ is: the record's declared
      error at the first reading, which adds the sketch's term and the sample's rejection and binding terms to the
      ledger's own and holds no PRF advantage, so its unit says how the whole error reads (`ERROR`, with
      `ERROR_FORM`'s words). The rate isn't in it, so every reading's is the same.

    Where that error is 1 or more (the document's `no_bound`) the session bounds nothing: `metric`, `verified` and
    each `free` are written with no value and that reason as their `why`, and the capacities and the error as ever.
    A measurement whose keys the record doesn't hold as numbers is left out."""
    run = [doc["run"]] if doc["run"] else []
    none = {"value": None, "why": doc["no_bound"]} if doc.get("no_bound") else {}
    first = next(iter(READINGS))
    error = _at(ledger, "declared", "gamma", first, "error", "windows")
    error = _at(ledger, first, "error", "windows") if error is None else error
    premises = [*([] if doc.get("bound") is True else [HELD_NONE]),
                *([ABOVE_BOUND] if error is None or error > 2.0 ** ERROR_BOUND_LOG2 else [])]
    at_error = [] if error is None else [f"error {power(error)}"]
    row = {"name": metric, "value": doc["value"], "unit": doc["unit"], "inputs": [*run, f"ledger.{first}.gamma"]} | none
    if metric == LOCAL_METRIC:
        row["unit"] = "; ".join([LOCAL_UNIT, *at_error])
    else:
        row |= {"target": TARGET, "verdict": "; ".join([doc["no_bound"] if none else "within target"
                                                        if doc["value"] <= TARGET else "past target", *at_error, *premises])}
    said = ran(doc)
    audit = None if said is None else (said[1]["coins"], said[1]["parameters"]) == ("deployed challenger", "deployed")
    extra = {**({} if doc.get("bound") is True else {"bound": False}),
             **({"audit": False} if audit is False else {})}
    if extra:
        row["extra"] = extra
    out = [{"name": "metric", "value": doc["value"], "unit": doc["unit"],
            "inputs": [*run, f"ledger.{first}.gamma"]} | none, row]
    if (verified := _at(ledger, first, "certified_bytes")) is not None:
        out.append({"name": "verified", "value": max(verified, 0.0), "unit": BYTES,
                    "inputs": [*run, f"ledger.{first}.certified_bytes"]} | none)
    span = _at(ledger, "inputs", "span_ns")
    for reading, rate_is in READINGS.items():
        rate, gamma = _at(ledger, reading, "capacity_bytes_per_s"), _at(ledger, reading, "gamma")
        if span is None or rate is None or gamma is None:
            continue
        capacity = rate * span / 1e9
        keys = [f"ledger.{reading}.capacity_bytes_per_s", "ledger.inputs.span_ns"]
        out += [{"name": f"capacity at {reading}", "value": capacity, "unit": f"{BYTES}, at {rate_is}: {OWN_RATE}",
                 "inputs": [*run, *keys]},
                {"name": f"free at {reading}", "value": min(gamma, 1.0) * capacity, "unit": BYTES,
                 "inputs": [*run, f"ledger.{reading}.gamma", *keys]} | none]
    if (error := error_of(ledger)) is not None:
        out.append({"name": "error", "value": error, "unit": ERROR,
                    "inputs": [*run, f"ledger.declared.gamma.{first}.error.windows"]})
    return out


def member(machine: dict) -> str:
    card = next((word for word in LOCAL_SHEETS if word in machine["card"]), machine["sheet"])
    return "-".join([MEMBER_MODEL, *card.lower().split()])


def pass_points(doc: dict, calibration: dict) -> list[dict]:
    """A `--local` session's pass time and bare read from its record's `calibration`, each left out where it lacks
    it."""
    run = [doc["run"]] if doc["run"] else []
    pass_ns = _at(calibration, "pass_ns", "p50")
    read_bytes, read_ms = (_at(calibration, "read_tiers", "hbm", k) for k in ("bytes", "ms"))
    out = []
    if pass_ns is not None:
        out.append({"name": PASS_METRIC, "value": pass_ns / 1e6, "unit": PASS_UNIT,
                    "inputs": [*run, "calibration.pass_ns.p50"]})
    if read_bytes and read_ms:
        out.append({"name": READ_METRIC, "value": read_bytes / read_ms / 1e9, "unit": READ_UNIT,
                    "inputs": [*run, "calibration.read_tiers.hbm.bytes", "calibration.read_tiers.hbm.ms"]})
    return [m | {"member": member(doc["machine"]),
                 "extra": {"card": doc["machine"]["card"], "store_bytes": doc["store_bytes"]}} for m in out]


def text(doc: dict) -> str:
    """The printed lines of a document: the number and how its error reads, what it was measured on, and the
    references."""
    rate, error = doc.get("rate_bytes_per_s"), doc.get("error")
    at = "at the data sheet's read bandwidth" + (f" ({rate / 1e12:,.4f} TB/s)" if rate else "")
    reads = f"its error reads {ERROR_FORM}, " + (f"δ = {power(error)} in the verifier's ledger" if error is not None
                                                 else "and the verifier's record holds no δ for it")
    if doc.get("no_bound"):
        lines = [f"memory: {doc['no_bound']}: this session certifies no share of the node's memory read rate, {at}; "
                 f"{reads}"]
    else:
        lines = [f"memory: {doc['value']:.3f} of the node's memory read rate is left uncertified by this session, {at}"
                 f"; {reads}"]
    if doc["loopback"]:
        lines.append(f"  {LOOPBACK}")
    elif doc.get("local"):
        lines.append(f"  {doc['local']}")
    if m := doc.get("machine"):
        lines.append(f"  card: {m['card']}, the verifier's node one card of hardware/nodes/{m['sheet']}.toml; "
                     f"data sheets sha256 {m['sheet_sha256'][:16]}, verifier sha256 {m['verifier_sha256'][:16]}")
    if not doc["loopback"] and doc["value"] is not None and doc["value"] >= 1:
        lines.append(f"  {NOTHING}")
    where = ("" if doc["loopback"] or doc.get("local") else f", on the auditor's verifier host (its run {doc['host_run']})"
             if doc.get("host_run") else ", its verifier not on the auditor's verifier host")
    lines.append(f"  session {doc['session']} of {doc['run'] or 'no recorded run'}, commit {doc['commit'] or 'unknown'}: "
                 f"accepted by the Lean verifier{where}")
    whole = None if doc["loopback"] else doc.get("node_gpu_bytes")
    lines.append("  store: " + (f"{doc['store_bytes']:,} bytes, a test store" if doc["loopback"] else _gb(doc["store_bytes"]))
                 + (f", {doc['store_bytes'] / whole:.3f} of the node's {_gb(whole)} of GPU memory" if whole else "")
                 + (f", bound to {doc['model']}'s pinned checkpoint" if doc.get("model")
                    else ", bound to no checkpoint" if doc["loopback"] else f"; {UNBOUND}"))
    for r in doc["beside"]:
        where = f"{r['run']}, {r.get('date') or 'no date'}, {r.get('card') or 'no card'}"
        error = "no error in its record" if r.get("error") is None else f"error {r['error']:.3g}"
        through = f", {THROUGH[r['name']]}" if r["name"] in THROUGH else ""
        lines.append(f"  beside it, {r['name']}: " + (f"{r['value']:.3f} ({r['detail']}{through}; {error}; {where})"
                                                     if r["value"] is not None
                                                     else f"{r['no_bound']} ({where})" if r.get("no_bound")
                                                     else f"not read ({r['why']})"))
    return "\n".join(lines)


def ran(doc: dict) -> tuple[str, dict] | None:
    """The session's `ran with` line and object (`kit.ran_with`), from its record: its coins (`COINS`), and its
    parameters, `deployed` only for the auditor device's coins over the deployed registration's
    `receipts.WIDE_ROWS` sketch rows (`SketchBoundDeployed`'s), else `development`; None for a record that states no
    challenger."""
    p = doc.get("parameters") or {}
    if (coins := COINS.get(p.get("coins"))) is None:
        return None
    deployed = coins == "deployed challenger" and p.get("rows") == receipts.WIDE_ROWS
    return kit.ran_with(coins, "deployed" if deployed else "development",
                        detail(p, doc.get("machine"), doc.get("error"), doc.get("store_bytes"), doc.get("node_gpu_bytes")))


def power(x: float) -> str:
    """`x` and its power of two, as 2.8e-05 (2^-15.1)."""
    return f"{x:.3g}" + (f" (2^{math.log2(x):.1f})" if x > 0 else "")


def detail(p: dict, machine: dict | None = None, error: float | None = None, store: int | None = None,
           whole: int | None = None) -> str:
    """The `ran with` line's detail (`parameters`): under `--local` the arena against the card's memory first; the
    session's form (a paced one's stretches, groups and pass, or the windows), the passes, the rows, the openings,
    the sample's term a pass group or a window (the record's `delta`, which is not the error), the lateness, the error
    with its power of two and the prover it is priced against, and under `--local` the sheet the share is of. Every
    figure is the record's: the arena and vLLM's share as the engine's record states them (`engine_args`)."""
    def ms(x):
        return "not stated" if x is None else f"{x:,.0f} ms"
    delta, rows, pace = p.get("delta"), p.get("rows"), p.get("pace")
    held = f", the store {_gb(store)} of its {_gb(whole)}" if store and whole else ""
    load = "" if not machine or machine.get("step_hz") is None else (
        f"; vLLM at most {machine['step_hz']:g} steps a second"
        + ("" if machine.get("busy") is None else f", busy {machine['busy']:.3f} of the session"))
    arena = ("" if not machine or "arena_gib" not in machine else
             (f"an arena of {machine['arena_gib']} GiB at vLLM's {machine['util']:g} of the card"
              if machine.get("arena_gib") is not None and machine.get("util") is not None
              else "an arena its engine's record doesn't state") + held + load + "; ")
    form = (f"paced: {pace['windows']} credited stretches of {ms(pace['t_ms'])}, {pace['groups']} groups a stretch at "
            f"{pace['pass_ms']:g} ms a pass, a coin every {pace['k']:g} pass{'' if pace['k'] == 1 else 'es'}, "
            f"{pace['draws_t']} openings a stretch, {pace['draws']} a group"
            if pace else f"{p.get('samples') or 'no'} openings a window; {p.get('windows') or 'no'} windows of "
            f"{ms(p.get('window_ms'))}")
    groups = f" in {p['pass_groups']:,} groups" if pace and p.get("pass_groups") is not None else ""
    return (arena + form + (f"; {p['passes']:,} passes{groups}" if p.get("passes") is not None else "")
            + f"; {rows or 'no'} sketch rows" + ("" if rows in (None, receipts.WIDE_ROWS) else
                                                f", not the deployed registration's {receipts.WIDE_ROWS}")
            + f"; the sample's term a {'pass group' if pace else 'window'} "
            + (f"2^{math.log2(delta):.1f}" if isinstance(delta, (int, float)) and delta > 0 else "not stated")
            + f"; lateness allowed {ms(p.get('allowance_ms'))}; "
            + (f"error {power(error)}" if error is not None else "the error")
            + " against a prover of "
            + ("SHA-512 compressions not stated" if p.get("budget_log2") is None
               else f"2^{p['budget_log2']} SHA-512 compressions on any hardware")
            + (f"; the share against hardware/nodes/{machine['sheet']}.toml at one card" if machine else ""))


def refused(why: str, *, local: bool, run: Callable = subprocess.run) -> int:
    """A `--local` or `--loopback` session with no number, through the kit (`kit.report`'s `accepted` false): `memory: no
    number: WHY` first, its coins the verifier's own and its parameters development, as `run.sh` runs them, and under
    `--local` its metric written with no value and that reason."""
    doc = {"value": None, "unit": UNIT, "commit": source_commit(run), "run": SETTINGS.RESEARCH_RUN_ID or None,
           "loopback": not local, "local": LOCAL if local else None}
    measured = [{"name": LOCAL_METRIC, "unit": LOCAL_UNIT, "inputs": [doc["run"]] if doc["run"] else []}]
    return kit.report("memory", "", doc, measured if local else [], ran_with=kit.ran_with("local challenger", "development"),
                      accepted=False, why=why)


def _no_store():
    raise Unreadable(NO_STORE)


def main(argv: list[str] | None = None, *, store_of: Callable[[], object] = EvidenceStore, run: Callable = subprocess.run,
         wait: Callable[[float], None] = time.sleep) -> int:
    """The command: 0 with the number printed, 1 with none (the session wasn't accepted, or its run left no record),
    2 when the command line or a run can't be read. `store_of` opens the evidence store, `run` starts a process and
    `wait` sleeps a number of seconds."""
    ap = argparse.ArgumentParser(prog="verity bench memory", description=__doc__.split("\n\n")[0])
    ap.add_argument("period", nargs="?", help="the node run: the challenger's period, a name its ledger hasn't used")
    ap.add_argument("--dry-run", action="store_true", help="the node run with nothing sent to either gate; no number")
    ap.add_argument("--from", dest="run", metavar="DIR|RUN", help="print from a recorded run: the directory of its record "
                    "(its job.json and out/, or that out/), read with nothing of ours, or a run id, read through the "
                    "evidence store")
    ap.add_argument("--session", type=int, metavar="K", help="the session, in a run that holds several")
    ap.add_argument("--loopback", action="store_true", help="both processes on this machine; certifies nothing")
    ap.add_argument("--local", action="store_true",
                    help="the node's session on one card of this machine, vLLM serving Llama 3.2 1B from the store "
                         "and the verifier beside it: a measurement, not an audit")
    a = ap.parse_args(argv)
    if sum((a.period is not None, a.run is not None, a.loopback, a.local)) != 1:
        ap.error("one of PERIOD (the node run), --from RUN, --loopback or --local")
    records = None
    try:
        if a.loopback:
            doc = loopback(run)
        elif a.local:
            doc, records = local(run)
            store_of = _no_store
        elif a.run and Path(a.run).is_dir():
            doc = from_dir(kit.record_dir(a.run), a.session)
            store_of = _no_store
        elif a.run:
            doc = from_run(store_of(), a.run, a.session)
        else:
            doc = node(store_of(), a.period, a.dry_run, a.session, run, wait)
            if isinstance(doc, int):
                return doc
        held = SETTINGS.RESEARCH_RUN_DIR
        # a run that ran the loopback or --local keeps that document, the mark `--from` that run reads
        kept = bool(held) and (was := kind(left(Path(held)))) is not None and was != kind(doc)
        ledger, calibration = doc.pop("ledger"), doc.pop("calibration", {})
    except NoNumber as e:
        code = 3 if isinstance(e, CouldNotRun) else 1
        if a.local or a.loopback:
            refused(str(e), local=a.local, run=run)
        else:
            print(f"memory: no number: {e}", file=sys.stderr)
        return code
    except Unreadable as e:
        print(f"memory: {e}", file=sys.stderr)
        return 2
    doc["beside"] = beside(store_of)
    # a loopback certifies nothing, so it writes no result, and a --local session writes its own metric; a result holds
    # one measurement a name, so a run that prints a second session keeps its first one's result
    printed = text(doc)
    measured = ([] if doc["loopback"] else amounts(doc, ledger, LOCAL_METRIC if doc["local"] else METRIC)
                + (pass_points(doc, calibration) if doc["local"] and doc.get("machine") else []))
    try:
        kit.report("memory", printed, doc, measured, document=not kept, ran_with=ran(doc), accepted=True,
                   from_run=a.run and (doc["run"] or a.run))
    except ValueError as e:
        print(f"memory: the result ({', '.join(m['name'] for m in measured)}) is not written: {e}", file=sys.stderr)
    if records is not None and not SETTINGS.RESEARCH_RUN_DIR:
        # the document says the records are a --local session's, which `--from` that directory reads them as
        (records / "out" / "memory.json").write_text(json.dumps(doc))
    if kept:
        which = "the loopback" if was == "loopback" else "--local"
        print(f"memory: this run ran {which}, and its out/memory.json stays {which}'s document", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
