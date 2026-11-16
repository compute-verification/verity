"""`verity bench memory`: the share of a node's memory read rate that one accepted PoUB session leaves uncertified, at
the data sheet's read bandwidth: the session's γ in its verifier's ledger at the specification's rate, as text and as
one line of JSON (`value`, `unit`, `commit`, `run`, `error`), exiting 0. A session that the Lean verifier didn't accept
(its record's `verdict` is not `lean`) prints no number and exits 1; a command line or a record that can't be read exits
2; and where no session ran to a verdict (`CouldNotRun`: no NVIDIA driver, no row or data sheet for the card, a
bootstrap that failed or that `--local` won't run, a session's processes that failed) it prints no number and exits 3.
The number is printed with its error, the ledger's δ for that γ summed over the session's windows, and the text says
how it reads (`ERROR_FORM`): that δ, and the term no δ holds. Where the error is 1 or more the session bounds nothing:
the command prints "no bound at this period" (`verity.protocol.NO_BOUND`) with the error and no number, and its
document's `value` is null with the reason under `no_bound`.

    verity bench memory PERIOD [--dry-run]        the node run: a gate run under the challenger's period PERIOD
    verity bench memory --from RUN [--session K]  a recorded run
    verity bench memory --loopback                both processes on this machine: a test of this command
    verity bench memory --local                   vLLM serving Llama 3.2 1B on this machine's GPU, audited beside it

The node run is `benchmarks/remote_session/gate_run.sh PERIOD` with one honest session: the prover on the whole GPU
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
or a run of `remote_session/run.sh` that held the verifier. The number is the verifier's alone: nothing the prover
recorded is read, so the session is named by its index (`--session K`, where the run holds several) and not by what
the prover says it did. A withdrawn run is refused before any of it is fetched. The text says whether the verifier ran
on the auditor's verifier host, what the store was bound to, that a share of 1 or more certifies nothing, and, of a
run this command's `--loopback` left, that it was one.

`--loopback` runs `remote_session/run.sh` with ROLE=both and the prover the driver's CPU reference over a small test
store. No node is audited, so it certifies nothing, and its output says so.

`--local` is the served session on one machine of your own: `remote_session/run.sh` with ROLE=both and SERVE=row, vLLM
serving the Llama 3.2 1B row for this machine's card (`LOCAL_ROWS`, or LOCAL_ROW) on one GPU under the audit, the Lean
verifier and its coins in processes beside it. Unless VLLM_PY names a vLLM interpreter, the tree's bootstrap
(`pod_bootstrap.sh`) first makes one in LOCAL_WORK, with the row's checkpoint made from its public pin on Hugging Face.
With no `gpu-lease` installed, the lease is the tree's own program, its locks in LOCAL_WORK. The run's records stay in
LOCAL_WORK/runs/ (the research run's own directory under one). The verifier and the coins ran on the machine being
measured, so the number is a measurement to improve on and not an audit: the text says so first, and the JSON says it
under `local`. It reads no evidence store, so the two references beside it are left out with that reason.

Beside the number it prints two references, each from its run in the store with the run's date and card and the error
its record states: PoUS's state bound over a card's memory, and the share a wipe left foreign at most.
`references.json` names their runs; a reference that can't be read prints its reason in place of a number, and so
does one whose error is 1 or more. Under `research run` the printed document is
also `$RESEARCH_RUN_DIR/out/memory.json`, and outside one `$OUT/memory.json` where `OUT` is set (`kit.report`); a
run that ran `--loopback` keeps a loopback's document there whatever it prints afterwards, since that document is how
`--from` knows the run's session for a loopback.

There an accepted session of a node run or of `--from` also writes the run's result,
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
other runs' numbers and are not in the result; a loopback writes none, since it certifies nothing, nor does a
`--local` session, which is no audit, and a session that
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
from verity.protocols.accounting.poub.verifier import ledger as poub_ledger
from verity.protocols.accounting.poub.verifier import receipts, timing
from verity_bench import kit
from verity_config.settings import of

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RS = ROOT / "benchmarks" / "remote_session"

#: this directory's registry: PACED, UNPACED, and the variables a research run sets
SETTINGS = of(__file__)
#: UNPACED's word: the windows, a window's milliseconds (to three places, as the verifier host takes them) and the
#: openings a window
UNPACED = re.compile(r"([1-9]\d*):(\d+(?:\.\d{1,3})?):([1-9]\d*)")
UNIT = "share of the node's memory read rate left uncertified, at the data sheet's read bandwidth"
#: the number's row on the console's Metrics tab (this directory's `metrics.toml`), and the goal's target for it
METRIC, TARGET = "memory-accounting/node-read-rate-uncertified", 0.14
#: a `--local` session's row: a share of one card's read rate against its own sheet, another quantity from a node's
LOCAL_METRIC = "memory-accounting/one-machine-read-rate-uncertified"
#: how the number's error reads, in PoUB's ledger's words (its `DELTA_FORM`): δ is the session's, in its verifier's
#: ledger, and the windows' PRF advantages are in no δ
ERROR_FORM = "δ + Σ_t adv_t (SHA-512 stream as PRF: not priced, not assumed)"
ROLE = "LLAMA32_1B_BF16"
REFERENCES = HERE / "references.json"
#: a block of PoUB's store, in bytes
BLOCK_BYTES = 4096
#: the files of a run this command reads
RECORDS = ["out/verifier.json", "out/host-run.txt", "out/host/*/out/verifier.json", "out/memory.json", "node.json"]
LOOPBACK = ("loopback: both processes ran on this machine, the prover the CPU reference over a test store; it "
            "certifies nothing")
LOCAL = "local: the verifier and its coins ran on the machine being measured"
#: the served row `--local` runs for a card's compute capability (nvidia-smi's `compute_cap`): Llama 3.2 1B in int8 on
#: one GPU, one request at a time; a card of another capability needs a row of its own (LOCAL_ROW)
LOCAL_ROWS = {"9.0": "llama32-1b__int8__h200__tp1__b1__i256__o32__mixed__greedy__bi-eager",
              "12.0": "llama32-1b__int8__rtxpro6000__tp1__b1__i256__o32__mixed__greedy__bi-eager"}
#: the node sheet (`hardware/nodes/`) of a --local session's card, by a word of the name nvidia-smi gives it, taken at
#: that one card: the share is of the one card's read rate (LOCAL_NODE names another sheet)
LOCAL_SHEETS = {"H200": "h200-sxm-141gb-x1", "RTX PRO 6000": "rtx-pro-6000-server-x8"}
#: what a --local result's digests are of: the session's verifier, and the data sheets its node is read from
VERIFIER = ROOT / "verity" / "protocols" / "accounting" / "poub" / "verifier"
SHEETS = ROOT / "hardware"
#: the one session form `--local` runs, as `run.sh`'s variables: one session of seven windows of a second at 2773
#: openings a window, the form the board's numbers are compared at
FORM = {"SESSIONS": "1", "WINDOWS": "7", "WINDOW_MS": "1000", "SAMPLES": "2773"}
#: who drew a session's coins, by its verifier's `--challenger`, in the kit's words (`kit.COINS`): its own challenger
#: or serve.py's on the verifier's machine, or the auditor device's through its gate
COINS = {"local": "local challenger", "serve": "local challenger", "gate": "deployed challenger"}
#: the checkpoint those rows serve, which the bootstrap makes from its ungated BF16 pin (`unsloth/Llama-3.2-1B`)
LOCAL_CASE = "LLAMA32_1B_INT8"
BOOTSTRAP = ROOT / "adapters" / "vllm" / "verity_vllm" / "ops" / "pod_bootstrap.sh"
#: what a bootstrap that touches files outside LOCAL_WORK holds, which `--local` won't run: removing the machine's package
#: caches, and reading tap builds at a fixed path
OUTSIDE_WORK = {"rm -rf /root/.cache": "removes /root/.cache/uv and /root/.cache/pip, other programs' package caches",
                "/workspace/cp/fa2/build": "reads its tap builds from /workspace/cp/fa2, not from LOCAL_WORK"}
#: where the bootstrap builds each tap under LOCAL_WORK (its scripts' variables), and the served row's option that
#: names each build there
TAPS = {"FA2": "cp/fa2", "NT": "cp/norm_tap", "RT": "cp/router_tap"}
TAP_SOS = {"--hidden-so-fa2": "cp/fa2/build/matReq/verity_fa2_matReq.so",
           "--hidden-so-fa3": "cp/fa2/build/fa3_matReq/verity_fa3_matReq.so",
           "--hidden-so-fa2-guarded": "cp/fa2/build/matReqG/verity_fa2_matReqG.so",
           "--hidden-so-fa3-guarded": "cp/fa2/build/fa3_matReqG/verity_fa3_matReqG.so",
           "--norm-tap-so": "cp/norm_tap/build/verity_norm_tap.so",
           "--router-tap-so": "cp/router_tap/build/verity_router_tap.so",
           "--native-collect-default": "cp/nc_build"}
#: why `--local` prints no reference beside its number
NO_STORE = "a --local run reads no evidence store, and the references are other runs' numbers in ours"
#: `gamma_decided_by` for a share read from the Python ledger where the Lean's answer for γ's terms isn't written, and
#: the line the text prints for it
PYTHON_LEDGER = "python-ledger"
PYTHON_LEDGER_WORDS = ("python ledger: the share and its error are the Python ledger's reading of the accepted session; "
                       "the Lean verifier has not decided them, so no audit reads them")
#: and the premise a Metrics row of such a share names in its verdict
PYTHON_LEDGER_PREMISE = "the share is the Python ledger's: the Lean verifier has not decided it"
NOTHING = "a share of 1 or more certifies nothing of the node's read rate"
#: what an unbound session's number rests on, in its Metrics row's verdict
HELD_NONE = "the verifier held no checkpoint"
#: a node's store that its verifier bound to no model, in the store's line
UNBOUND = f"{HELD_NONE}: the store's bytes are the developer's choice"
#: the goal's bound on the number's error, as log2: a Metrics row whose error the record holds above it, or holds
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


def error_of(ledger: dict, *keys: str) -> float | None:
    """The session's error as its verifier's ledger declares it at the reading the number is printed at, read window
    by window as γ is: the windows' δ summed, with the sketch's term and the sample's rejection and binding terms.
    None where the record holds none. `keys` reads it inside the declaration (`python_ledger`)."""
    return _at(ledger, "declared", *keys, "gamma", next(iter(READINGS)), "error", "windows")


def session(record: Path, k: int | None, *, python_ledger: bool = False) -> dict:
    """Session `k` (the only one when `k` is None) of the verifier's record at `record`, as the printer reads it:
    `NoNumber` unless the Lean verifier accepted it (`accepted`, with `verdict` `lean`) and its ledger holds γ at the
    specification's rate, which is
    `remote_session/summary.py`'s `gamma_spec` of the session. Where the Lean's answer for γ's terms isn't written,
    the reading is undecided and there is no number, unless `python_ledger`, which only `--loopback` and `--local`
    pass: then γ and its error are the Python ledger's reading the record carries beside it (`python_ledger`), and
    `gamma_decided_by` says so (`PYTHON_LEDGER`); else it is `lean`."""
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
    spec, decided_by, under = ledger.get("spec") or {}, "lean", ()
    if (undecided := spec.get("undecided")) is not None:
        if not python_ledger or not isinstance(spec.get("python_ledger"), dict):
            raise NoNumber(f"session {vs['k']} was accepted and its γ at the specification's rate is {undecided}")
        spec, decided_by, under = spec["python_ledger"], PYTHON_LEDGER, ("python_ledger",)
    gamma = spec.get("gamma")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise NoNumber(f"session {vs['k']} was accepted and its ledger holds no γ at the specification's rate")
    binding = v.get("binding") or {}
    return {"value": gamma, "gamma_decided_by": decided_by, "session": vs["k"], "verdict": vs.get("verdict"),
            "error": error_of(ledger, *under),
            "rate_bytes_per_s": (ledger.get("inputs") or {}).get("spec_bytes_per_s"),
            "store_bytes": blocks * BLOCK_BYTES,
            "node_gpu_bytes": (ledger.get("memory") or {}).get("hbm_bytes"),
            "model": binding.get("name") if binding.get("bound") is True else None,
            "bound": binding.get("bound") is True, "parameters": parameters(v, vs), "ledger": ledger}


def parameters(v: dict, vs: dict) -> dict:
    """What session `vs` of the verifier's record `v` ran with, as its record states it: where its coins came from
    (the verifier's `--challenger`), the store's rows, the openings a window, δ a window, and the windows, their
    length and the lateness allowed, and the prover its error is priced against: `budget_log2`, the log2 of the SHA-512
    compressions it may make on any hardware (`audit.QUERIES_LOG2`). None where the record states none."""
    def ns(x):
        return x / 1e6 if isinstance(x, (int, float)) and not isinstance(x, bool) else None

    return {"coins": (vs.get("challenger") or {}).get("challenger"), "rows": (v.get("registration") or {}).get("rows"),
            "samples": vs.get("samples"), "delta": vs.get("delta"), "windows": vs.get("windows"),
            "window_ms": ns(vs.get("window_ns")), "allowance_ms": ns(vs.get("allowance_ns")),
            "budget_log2": ((vs.get("ledger") or {}).get("sketch_failure") or {}).get("queries_log2")}


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
            "host_run": host_run,
            "no_bound": why, **{k: v for k, v in s.items() if k not in ("value", "gamma_decided_by")}}


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
    document about another run, or one that doesn't say, says nothing of this run's session, and a `--local` run
    (its node.json) that left no document isn't printed as a node's either."""
    doc = left(tree)
    if not doc:
        if (tree / "node.json").is_file():
            raise Unreadable(f"{run} holds --local's node.json but no out/memory.json saying what it was, so it isn't "
                             "printed as a node session")
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
    else the document this command left there (`out/memory.json`)."""
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
            raise CouldNotRun(f"the loopback's processes failed (run.sh exited {done.returncode}); their logs are above")
        s = session(Path(env["RESEARCH_RUN_DIR"]) / "out" / "verifier.json", None, python_ledger=True)
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


def bootstrap_failed(out: Path, rc: int) -> str:
    """Why the bootstrap failed, from the step it names last in `out/bootstrap.fail` (`BOOTSTRAP_FAIL_STEP: why`)."""
    fail = out / "bootstrap.fail"
    lines = fail.read_text(errors="replace").strip().splitlines() if fail.is_file() else []
    step, _, why = (lines[-1] if lines else "").partition(": ")
    if not step.startswith("BOOTSTRAP_FAIL_"):
        return f"the bootstrap failed (pod_bootstrap.sh exited {rc}, naming no step); its logs are in {out}"
    step = step.removeprefix("BOOTSTRAP_FAIL_")
    hint = ": a checkpoint that won't download means Hugging Face can't be reached from here" if step == "CHECKPOINT" else ""
    return f"the bootstrap failed at its {step} step ({why.strip()[-300:]}); its logs are in {out}{hint}"


def card_row(name: str, cc: str) -> str:
    """The served row for card `name` of compute capability `cc`: LOCAL_ROW where it is set, else `LOCAL_ROWS`'."""
    if row := SETTINGS.LOCAL_ROW:
        return row
    if (row := LOCAL_ROWS.get(cc)) is None:
        raise CouldNotRun(f"{name} is compute capability {cc}, and the tree has a Llama 3.2 1B row only "
                       f"for {' and '.join(LOCAL_ROWS)} ({', '.join(LOCAL_ROWS.values())}): make one for this card "
                       "(`verity-vllm workload --row-id ROW ...`, its target with \"async_scheduling\": false) and "
                       "name it with LOCAL_ROW")
    return row


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


def local(run: Callable = subprocess.run) -> dict:
    """One served session on this machine: `run.sh` with ROLE=both and SERVE=row, the Llama 3.2 1B row for this
    machine's card under the audit on one GPU and the verifier beside it, read from its records. Unless VLLM_PY is
    set, the tree's bootstrap makes the vLLM environment and the row's checkpoint in LOCAL_WORK first (it skips what
    is already there); with no `gpu-lease` installed the tree's own lease keeps its locks there too. The run's records
    are kept, in the research run's directory under one and else in a directory of LOCAL_WORK/runs/."""
    given = SETTINGS.environ()
    if other := [f"{k}={given[k]}" for k, v in FORM.items() if given.get(k, v) not in (v, "")]:
        raise Unreadable(f"--local runs one session form, {FORM['WINDOWS']} windows of {FORM['WINDOW_MS']} ms at "
                         f"{FORM['SAMPLES']} openings a window, so its numbers compare: {', '.join(other)} is another")
    name, cc = gpu_of(run)
    row, kind, work = card_row(name, cc), sheet_of(name), Path(SETTINGS.LOCAL_WORK).expanduser()
    node, hbm_bytes = one_card(kind)
    verifier_cpus, prover_cpus = cores()
    env = {"VERIFIER_CPUS": verifier_cpus, "PROVER_CPUS": prover_cpus, **SETTINGS.environ(), **FORM, "ROLE": "both",
           "SERVE": "row", "ROW": row, "PLAN": SETTINGS.PLAN or "serve"}
    if not SETTINGS.VLLM_PY:
        script = BOOTSTRAP.read_text()
        if outside := [why for held, why in OUTSIDE_WORK.items() if held in script]:
            raise CouldNotRun(f"this tree's bootstrap ({BOOTSTRAP.name}) {' and '.join(outside)}, so --local doesn't run "
                           "it: set VLLM_PY to a vLLM environment of your own, or run from a tree whose bootstrap "
                           "keeps to LOCAL_WORK")
        work.mkdir(parents=True, exist_ok=True)
        boot = {**SETTINGS.environ(), "WORK": str(work), "VENV": str(work / "venv312"), "HF_HOME": str(work / "hf"),
                "UV_CACHE_DIR": str(work / "cache" / "uv"), "PIP_CACHE_DIR": str(work / "cache" / "pip"),
                "XDG_CACHE_HOME": str(work / "cache"), **{k: str(work / v) for k, v in TAPS.items()}}
        done = run(["bash", str(BOOTSTRAP), "--cases", LOCAL_CASE, "--out", str(work / "bootstrap")],
                   cwd=BOOTSTRAP.parents[2], env=boot, stdout=sys.stderr)
        if done.returncode:
            raise CouldNotRun(bootstrap_failed(work / "bootstrap", done.returncode))
        env.update(VLLM_PY=str(work / "venv312" / "bin" / "python"), NODE_WORKSPACE=str(work), HF_HOME=str(work / "hf"),
                   UV_CACHE_DIR=boot["UV_CACHE_DIR"], PIP_CACHE_DIR=boot["PIP_CACHE_DIR"],
                   XDG_CACHE_HOME=boot["XDG_CACHE_HOME"],
                   ROW_ARGS=" ".join([*(f"{flag} {shlex.quote(str(work / at))}" for flag, at in TAP_SOS.items()),
                                      *filter(None, [env.get("ROW_ARGS")])]))
    if shutil.which("gpu-lease") is None:
        env.setdefault("GPU_LEASE_DIR", str(work / "gpu-lease"))
    held = SETTINGS.RESEARCH_RUN_DIR
    if not held:
        (work / "runs").mkdir(parents=True, exist_ok=True)
        held = tempfile.mkdtemp(prefix="run-", dir=work / "runs")
    env["RESEARCH_RUN_DIR"] = held
    Path(held).mkdir(parents=True, exist_ok=True)
    (Path(held) / "node.json").write_text(json.dumps(node, sort_keys=True))
    env["VERIFIER_ARGS"] = " ".join(filter(None, [env.get("VERIFIER_ARGS"), "--node", shlex.quote(f"{held}/node.json"),
                                                  "--hbm-bytes", str(hbm_bytes)]))
    print(f"memory: --local serves {row} on {name}, the verifier's node one card of {kind}; its records go to "
          f"{held}/out", file=sys.stderr)
    done = run(["bash", str(RS / "run.sh")], cwd=ROOT, env=env, stdout=sys.stderr)
    if done.returncode:
        raise CouldNotRun(f"the served session failed (run.sh exited {done.returncode}); its logs are above and in "
                       f"{held}/out")
    s = session(Path(held) / "out" / "verifier.json", None, python_ledger=True)
    machine = {"card": name, "sheet": kind, "sheet_sha256": digest(SHEETS), "verifier_sha256": digest(VERIFIER)}
    return document(s, source_commit(run), SETTINGS.RESEARCH_RUN_ID, loopback=False, local=True, machine=machine)


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
    - `metric`: the same number, the Metrics tab's row (this directory's `metrics.toml`; `METRIC`, or `LOCAL_METRIC`
      for a `--local` session), with `TARGET` and where the number stands against it. A share the Python ledger
      computed names that in its verdict (`PYTHON_LEDGER_PREMISE`) and its `extra`, with `audit` false where the
      session's coins or parameters aren't deployed (`ran`). A session whose verifier held no checkpoint still writes the row, with `"bound": false` and
      its verdict naming that premise (`HELD_NONE`); a bound session's row holds neither. Where the number's error is
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
    python = doc.get("gamma_decided_by") == PYTHON_LEDGER
    # a share the Python ledger computed is read where the verifier records it, beside each undecided reading
    under = {r: (f"{r}.python_ledger" if python else r) for r in (*READINGS, "declared")}
    if python:
        ledger = {**ledger, **{r: (ledger.get(r) or {}).get("python_ledger") or {} for r in under}}
    error = _at(ledger, "declared", "gamma", first, "error", "windows")
    error = _at(ledger, first, "error", "windows") if error is None else error
    premises = [*([] if doc.get("bound") is True else [HELD_NONE]),
                *([ABOVE_BOUND] if error is None or error > 2.0 ** ERROR_BOUND_LOG2 else []),
                *([PYTHON_LEDGER_PREMISE] if python else [])]
    row = {"name": metric, "value": doc["value"], "unit": doc["unit"], "inputs": [*run, f"ledger.{under[first]}.gamma"],
           "target": TARGET,
           "verdict": "; ".join([doc["no_bound"] if none else "within target" if doc["value"] <= TARGET
                                  else "past target", *premises])} | none
    said = ran(doc)
    audit = None if said is None else (said[1]["coins"], said[1]["parameters"]) == ("deployed challenger", "deployed")
    extra = {**({} if doc.get("bound") is True else {"bound": False}),
             **({"audit": False} if audit is False else {}), **({"gamma_decided_by": PYTHON_LEDGER} if python else {})}
    if extra:
        row["extra"] = extra
    out = [{"name": "metric", "value": doc["value"], "unit": doc["unit"],
            "inputs": [*run, f"ledger.{under[first]}.gamma"]} | none, row]
    if (verified := _at(ledger, first, "certified_bytes")) is not None:
        out.append({"name": "verified", "value": max(verified, 0.0), "unit": BYTES,
                    "inputs": [*run, f"ledger.{under[first]}.certified_bytes"]} | none)
    span = _at(ledger, "inputs", "span_ns")
    for reading, rate_is in READINGS.items():
        rate, gamma = _at(ledger, reading, "capacity_bytes_per_s"), _at(ledger, reading, "gamma")
        if span is None or rate is None or gamma is None:
            continue
        capacity = rate * span / 1e9
        keys = [f"ledger.{under[reading]}.capacity_bytes_per_s", "ledger.inputs.span_ns"]
        out += [{"name": f"capacity at {reading}", "value": capacity, "unit": f"{BYTES}, at {rate_is}: {OWN_RATE}",
                 "inputs": [*run, *keys]},
                {"name": f"free at {reading}", "value": min(gamma, 1.0) * capacity, "unit": BYTES,
                 "inputs": [*run, f"ledger.{under[reading]}.gamma", *keys]} | none]
    if (error := error_of(ledger)) is not None:
        out.append({"name": "error", "value": error, "unit": ERROR,
                    "inputs": [*run, f"ledger.{under['declared']}.gamma.{first}.error.windows"]})
    return out


def text(doc: dict) -> str:
    """The printed lines of a document: the number and how its error reads, what it was measured on, and the
    references."""
    rate, error = doc.get("rate_bytes_per_s"), doc.get("error")
    at = "at the data sheet's read bandwidth" + (f" ({rate / 1e12:,.4f} TB/s)" if rate else "")
    reads = f"its error reads {ERROR_FORM}, " + (f"δ = {error:.3g} in the verifier's ledger" if error is not None
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
    if doc.get("gamma_decided_by") == PYTHON_LEDGER:
        lines.append(f"  {PYTHON_LEDGER_WORDS}")
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
    return kit.ran_with(coins, "deployed" if deployed else "development", detail(p, doc.get("machine")))


def detail(p: dict, machine: dict | None = None) -> str:
    """The `ran with` line's detail (`parameters`): the rows, the openings and δ a window, the windows, the lateness,
    the prover the error is priced against, and under `--local` the sheet the share is of."""
    def ms(x):
        return "not stated" if x is None else f"{x:,.0f} ms"
    delta, rows = p.get("delta"), p.get("rows")
    return (f"{rows or 'no'} sketch rows" + ("" if rows in (None, receipts.WIDE_ROWS) else
                                             f", not the deployed registration's {receipts.WIDE_ROWS}")
            + f"; {p.get('samples') or 'no'} openings a window; δ = "
            + (f"2^{math.log2(delta):.1f}" if isinstance(delta, (int, float)) and delta > 0 else "not stated")
            + f" a window; {p.get('windows') or 'no'} windows of {ms(p.get('window_ms'))}; lateness allowed "
            f"{ms(p.get('allowance_ms'))}; the error against a prover of "
            + ("SHA-512 compressions not stated" if p.get("budget_log2") is None
               else f"2^{p['budget_log2']} SHA-512 compressions on any hardware")
            + (f"; the share against hardware/nodes/{machine['sheet']}.toml at one card" if machine else ""))


def refused(why: str, *, local: bool, run: Callable = subprocess.run) -> int:
    """A `--local` or `--loopback` session with no number, through the kit (`kit.report`'s `accepted` false): `memory: no
    number: WHY` first, its coins the verifier's own and its parameters development, as `run.sh` runs them, and under
    `--local` its row written with no value and that reason."""
    doc = {"value": None, "unit": UNIT, "commit": source_commit(run), "run": SETTINGS.RESEARCH_RUN_ID or None,
           "loopback": not local, "local": LOCAL if local else None}
    measured = [{"name": LOCAL_METRIC, "unit": UNIT, "target": TARGET, "inputs": [doc["run"]] if doc["run"] else []}]
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
                    "(its job.json and out/), read with nothing of ours, or a run id, read through the evidence store")
    ap.add_argument("--session", type=int, metavar="K", help="the session, in a run that holds several")
    ap.add_argument("--loopback", action="store_true", help="both processes on this machine; certifies nothing")
    ap.add_argument("--local", action="store_true",
                    help="vLLM serving Llama 3.2 1B on this machine's GPU with the verifier beside it: a measurement, "
                         "not an audit")
    a = ap.parse_args(argv)
    if sum((a.period is not None, a.run is not None, a.loopback, a.local)) != 1:
        ap.error("one of PERIOD (the node run), --from RUN, --loopback or --local")
    try:
        if a.loopback:
            doc = loopback(run)
        elif a.local:
            doc = local(run)
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
        ledger = doc.pop("ledger")
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
    # a loopback certifies nothing, so it writes no result, and a --local session writes its own row; a result holds
    # one measurement a name, so a run that prints a second session keeps its first one's result
    printed = text(doc)
    measured = [] if doc["loopback"] else amounts(doc, ledger, LOCAL_METRIC if doc["local"] else METRIC)
    try:
        kit.report("memory", printed, doc, measured, document=not kept, ran_with=ran(doc), accepted=True)
    except ValueError as e:
        print(f"memory: the result ({', '.join(m['name'] for m in measured)}) is not written: {e}", file=sys.stderr)
    if kept:
        which = "the loopback" if was == "loopback" else "--local"
        print(f"memory: this run ran {which}, and its out/memory.json stays {which}'s document", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
