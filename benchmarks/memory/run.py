"""`verity bench memory`: the share of a node's memory read rate that one accepted PoUB session leaves uncertified, at
the data sheet's read bandwidth: the session's γ in its verifier's ledger at the specification's rate, as text and as
one line of JSON (`value`, `unit`, `commit`, `run`). A session that wasn't accepted prints no number and exits 1.

    verity bench memory PERIOD [--dry-run]        the node run: a gate run under the challenger's period PERIOD
    verity bench memory --from RUN [--session K]  a recorded run
    verity bench memory --loopback                both processes on this machine: a test of this command

The node run is `benchmarks/remote_session/gate_run.sh PERIOD` with one honest session: the prover on the whole GPU
node serving ROLE (default Llama 3.2 1B as upstream holds it, the role core's pin names `LLAMA32_1B_BF16`), each
engine's weights arena the store, the verifier on the auditor's verifier host bound to that role's pinned checkpoint,
the coins from the challenger host. PACED is the session (default seven credited stretches of a second, each pass
under a live coin of its own); UNPACED asks for a session of one coin a window in its place, as
WINDOWS:WINDOW_MS:SAMPLES (`7:60000:33273` is seven windows of a minute with 33,273 openings each). `gate_run.sh`'s
own variables (COMMIT, ARENA_GIB, UTIL, HOST_GIB, LEASE_BY, FETCH, ...) pass through: with HOST_GIB the store holds
that many GiB of the node's host memory beside the arenas. It waits for the fire run to reach the evidence store and
prints from it as `--from` does; `--dry-run` sends nothing to either gate and prints no number.

`--from RUN` reads the verifier's record of a run in the store: a gate run's fire run (its host run's `verifier.json`)
or a run of `remote_session/run.sh` that held the verifier. The number is the verifier's alone: nothing the prover
recorded is read, so the session is named by its index (`--session K`, where the run holds several) and not by what
the prover says it did. A withdrawn run is refused before any of it is fetched. The text says whether the verifier ran
on the auditor's verifier host, what the store was bound to, that a share of 1 or more certifies nothing, and, of a
run this command's `--loopback` left, that it was one.

`--loopback` runs `remote_session/run.sh` with ROLE=both and the prover the driver's CPU reference over a small test
store. No node is audited, so it certifies nothing, and its output says so.

Beside the number it prints two references, each from its run in the store with the run's date and card: PoUS's state
bound over a card's memory, and the share a wipe left foreign at most. `references.json` names their runs;
a reference that can't be read prints its reason in place of a number. Under `research run` the printed document is
also `$RESEARCH_RUN_DIR/out/memory.json`; a run that ran `--loopback` keeps a loopback's document there whatever it
prints afterwards, since that document is how `--from` knows the run's session for a loopback.

Under `research run` an accepted session of a node run or of `--from` also writes the run's result,
`$RESEARCH_RUN_DIR/result.json`, the file `research` reads a run's result from, through the harness kit's writer
(`verity_bench.kit`), in the fields every resource's benchmark reports (`amounts`), each from the verifier's record
alone and named in its `inputs` by the record's key: `metric` (the printed number in its unit), `verified` (the bytes
the verifier certified read over the session's span), `capacity at READING` and `free at READING` for each reading the
ledger holds (`READINGS`: the node's read rate times the span, and γ of it), and `error` (the probability the verified
amount doesn't hold). A reading's rate is the verifier's own figure for the node (`timing.py`), not the registered
hardware's, and the capacity's unit says so. The references are other runs' numbers and are not in the result; a
loopback writes none, since it certifies nothing, and a session that wasn't accepted writes none, as it prints no
number.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

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
ROLE = "LLAMA32_1B_BF16"
REFERENCES = HERE / "references.json"
#: a block of PoUB's store, in bytes
BLOCK_BYTES = 4096
#: the files of a run this command reads
RECORDS = ["out/verifier.json", "out/host-run.txt", "out/host/*/out/verifier.json", "out/memory.json"]
LOOPBACK = ("loopback: both processes ran on this machine, the prover the CPU reference over a test store; it "
            "certifies nothing")
NOTHING = "a share of 1 or more certifies nothing of the node's read rate"
#: the readings of the verifier's ledger, each the key it holds one under and the read rate it takes for the node: the
#: number is printed at the first
READINGS = {"spec": "the data sheet's read bandwidth", "measured": "the measured read rate"}
#: the unit of the result's amounts
BYTES = "bytes over the session's span"
#: where a capacity's rate comes from, which its unit says
OWN_RATE = "the verifier's own figure for the node (its timing.py), not the registered hardware's"
#: the result's error: which of the record's two it is, and what it leaves out
ERROR = ("probability that the verified amount doesn't hold, read window by window as the share is: the ledger's error "
         "with the sketch's term and the sample's rejection and binding terms added; the windows' PRF advantages, "
         "which the record names with no value, are not in it")
#: the wipe run's label states its bound in words
WIPE = re.compile(r"foreign <= ([\d,]+) bits\b.*\bM = ([\d,]+) B\b")
#: minutes the node run waits for a launched run's record: the fire run's own bound and its publish
WAIT_MIN = 9 * 60


class NoNumber(Exception):
    """The session has no number to print: it wasn't accepted, or its run left no record of one."""


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


def session(record: Path, k: int | None) -> dict:
    """Session `k` (the only one when `k` is None) of the verifier's record at `record`, as the printer reads it:
    `NoNumber` unless the verifier accepted it and its ledger holds γ at the specification's rate, which is
    `remote_session/summary.py`'s `gamma_spec` of the session."""
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
    ledger = vs.get("ledger") or {}
    gamma = (ledger.get("spec") or {}).get("gamma")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise NoNumber(f"session {vs['k']} was accepted and its ledger holds no γ at the specification's rate")
    binding = v.get("binding") or {}
    return {"value": gamma, "session": vs["k"], "verdict": vs.get("verdict"),
            "rate_bytes_per_s": (ledger.get("inputs") or {}).get("spec_bytes_per_s"),
            "store_bytes": blocks * BLOCK_BYTES,
            "node_gpu_bytes": (ledger.get("memory") or {}).get("hbm_bytes"),
            "model": binding.get("name") if binding.get("bound") is True else None,
            "ledger": ledger}


def document(s: dict, commit: str | None, run: str | None, *, loopback: bool, host_run: str | None = None) -> dict:
    """The printed document of session `s` (`session`), before its references: the number, its unit, the commit the
    session ran from and its run, then what it was measured on. `host_run` is the auditor's verifier host's run that
    held the verifier, None for a verifier that ran anywhere else. It carries the session's ledger as its verifier
    recorded it under `ledger`, which the command takes out before it prints (`amounts` reads it)."""
    return {"value": s["value"], "unit": UNIT, "commit": commit, "run": run, "loopback": loopback, "host_run": host_run,
            **{k: v for k, v in s.items() if k != "value"}}


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


def looped(tree: Path, run: str) -> bool:
    """Whether `run`, whose records are under `tree`, is this command's own loopback, as the document it left says. A
    document about another run says nothing of this run's session, and a session that may be a loopback's isn't
    printed as a node's."""
    doc = left(tree)
    if doc and doc.get("run") != run:
        raise Unreadable(f"{run}'s out/memory.json is the document of {doc.get('run')}, so whether {run}'s own session "
                         "was a loopback isn't recorded")
    return doc.get("loopback") is True


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
        raise NoNumber(f"{run} left no verifier's record (it ended rc {(attempt.get('status') or {}).get('rc')})")
    return document(session(verifier, k), commit_of(attempt), run, loopback=looped(tree, run), host_run=host)


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
               **SETTINGS.environ(), "RESEARCH_RUN_DIR": held or tmp, "ROLE": "both", "PLAN": "honest"}
        env.pop("SERVE", None)
        done = run(["bash", str(RS / "run.sh")], cwd=ROOT, env=env, stdout=sys.stderr)
        if done.returncode:
            raise NoNumber(f"the loopback's processes failed (run.sh exited {done.returncode}); their logs are above")
        s = session(Path(env["RESEARCH_RUN_DIR"]) / "out" / "verifier.json", None)
    return document(s, source_commit(run), SETTINGS.RESEARCH_RUN_ID, loopback=True)


def form() -> dict[str, str]:
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
    env = {"ROLE": ROLE, **SETTINGS.environ(), "PLAN": "honest", **form()}
    rc = run(["bash", str(RS / "gate_run.sh"), *(["--dry-run"] if dry else []), period], cwd=ROOT, env=env,
             stdout=sys.stderr).returncode
    if dry:
        print("memory: a dry run: nothing reached either gate, and there is no number", file=sys.stderr)
        return rc
    if rc:
        raise NoNumber(f"the gate run did not launch (gate_run.sh exited {rc})")
    state = Path.home() / ".research" / "gate-runs" / f"{period}.env"
    ids = dict(line.split("=", 1) for line in state.read_text().splitlines() if "=" in line)
    fire, prover = ids["fire"], ids["prover"]
    print(f"memory: gate run {period}: fire {fire}, prover {prover}. Waiting for the fire run to reach the store; if "
          f"this is interrupted, `verity bench memory --from {fire}` prints once it has", file=sys.stderr)
    for _ in range(WAIT_MIN):
        if store.attempt(fire) is not None:
            return from_run(store, fire, k)
        wait(60)
    raise NoNumber(f"{fire} left no record in the store within {WAIT_MIN} minutes")


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
    return {"value": held / total, "unit": "PoUS's state bound over the card's memory: s_bits is S0 = floor(rho*|C|), "
            "a lower bound on the device's memory, its on-chip state and a round of side-channel traffic together, "
            "not the device memory credited",
            "detail": f"{_gb(held)} of {_gb(total)}",
            "card": d["held"].get("device") or (a.get("conditions") or {}).get("hw.sku")}


def wipe(store, run: str) -> dict:
    """The share of the card's memory a wipe left foreign at most, from the label that states the run's bound."""
    a = store.attempt(run) or {}
    text = store.label(run, "label")
    m = WIPE.search(text or "")
    if m is None:
        raise Unreadable(f"{run}'s label doesn't state a residue: {text!r}")
    bits, memory = (int(x.replace(",", "")) for x in m.groups())
    return {"value": bits / 8 / memory, "unit": "share of the card's memory a wipe leaves foreign, at most",
            "detail": f"{_gb(bits / 8)} of {_gb(memory)}", "card": (a.get("conditions") or {}).get("hw.sku")}


READERS = {"PoUS state bound": pous, "wipe residue": wipe}


def beside(store_of: Callable[[], object]) -> list[dict]:
    """Each reference of `references.json` read from its run, or why it can't be."""
    out, store = [], None
    for name, run in json.loads(REFERENCES.read_text()).items():
        row = {"name": name, "run": run}
        try:
            store = store if store is not None else store_of()
            row.update(READERS[name](store, run))
            row["date"] = (((store.attempt(run) or {}).get("execution") or {}).get("start_utc") or "")[:10] or None
        except Unreadable as e:
            row.update(value=None, why=str(e))
        except Exception as e:  # noqa: BLE001  (whatever a reference fails with, the session's number still prints)
            row.update(value=None, why=f"{type(e).__name__}: {e}")
        out.append(row)
    return out


def _at(ledger: dict, *keys: str) -> float | None:
    """The number the session's ledger holds under `keys`, None where it holds none there."""
    x = ledger
    for k in keys:
        x = x.get(k) if isinstance(x, dict) else None
    return None if isinstance(x, bool) or not isinstance(x, (int, float)) else x


def amounts(doc: dict, ledger: dict) -> list[dict]:
    """The result a printed document `doc` and its session's `ledger` give in the fields every resource's benchmark
    reports, as the kit writer's measurements, each from the verifier's record alone; its `inputs` are the recorded
    run and the keys of the session's entry in that record it was read from.

    - `metric`: the printed number in its unit.
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
      ledger's own and holds no PRF advantage (`ERROR`). The rate isn't in it, so every reading's is the same.

    A measurement whose keys the record doesn't hold as numbers is left out."""
    run = [doc["run"]] if doc["run"] else []
    first = next(iter(READINGS))
    out = [{"name": "metric", "value": doc["value"], "unit": doc["unit"], "inputs": [*run, f"ledger.{first}.gamma"]}]
    if (verified := _at(ledger, first, "certified_bytes")) is not None:
        out.append({"name": "verified", "value": max(verified, 0.0), "unit": BYTES,
                    "inputs": [*run, f"ledger.{first}.certified_bytes"]})
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
                 "inputs": [*run, f"ledger.{reading}.gamma", *keys]}]
    if (error := _at(ledger, "declared", "gamma", first, "error", "windows")) is not None:
        out.append({"name": "error", "value": error, "unit": ERROR,
                    "inputs": [*run, f"ledger.declared.gamma.{first}.error.windows"]})
    return out


def result(held: Path, measured: list[dict]) -> None:
    """Appends `measured` to the result of the run whose directory is `held`, `held/result.json`, through the harness
    kit's writer, which stamps each with this run, its commit and this host. The writer is given the run's directory
    itself, not its `out/`: a run has one result, and that file is where `research` reads it. A result holds one
    measurement a name, so a run that prints a second session keeps its first one's result, and says so."""
    names = ", ".join(m["name"] for m in measured)
    try:
        writer = kit.writer(held)
        for m in measured:
            writer.measurement(**m)
    except ValueError as e:
        print(f"memory: the result ({names}) is not written: {e}", file=sys.stderr)


def text(doc: dict) -> str:
    """The printed lines of a document: the number, what it was measured on, and the references."""
    rate = doc.get("rate_bytes_per_s")
    lines = [f"memory: {doc['value']:.3f} of the node's memory read rate is left uncertified by this session, at the "
             f"data sheet's read bandwidth" + (f" ({rate / 1e12:,.4f} TB/s)" if rate else "")]
    if doc["loopback"]:
        lines.append(f"  {LOOPBACK}")
    elif doc["value"] >= 1:
        lines.append(f"  {NOTHING}")
    verdict = ("accepted by the Lean verifier" if doc.get("verdict") == "lean"
               else "accepted (its record names no Lean verdict)")
    where = ("" if doc["loopback"] else f", on the auditor's verifier host (its run {doc['host_run']})" if doc.get("host_run")
             else ", its verifier not on the auditor's verifier host")
    lines.append(f"  session {doc['session']} of {doc['run'] or 'no recorded run'}, commit {doc['commit'] or 'unknown'}: "
                 f"{verdict}{where}")
    whole = None if doc["loopback"] else doc.get("node_gpu_bytes")
    lines.append("  store: " + (f"{doc['store_bytes']:,} bytes, a test store" if doc["loopback"] else _gb(doc["store_bytes"]))
                 + (f", {doc['store_bytes'] / whole:.3f} of the node's {_gb(whole)} of GPU memory" if whole else "")
                 + (f", bound to {doc['model']}'s pinned checkpoint" if doc.get("model") else ", bound to no checkpoint"))
    for r in doc["beside"]:
        where = f"{r['run']}, {r.get('date') or 'no date'}, {r.get('card') or 'no card'}"
        lines.append(f"  beside it, {r['name']}: "
                     + (f"{r['value']:.3f} ({r['detail']}; {where})" if r["value"] is not None else f"not read ({r['why']})"))
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, store_of: Callable[[], object] = EvidenceStore, run: Callable = subprocess.run,
         wait: Callable[[float], None] = time.sleep) -> int:
    """The command: 0 with the number printed, 1 with none (the session wasn't accepted, or its run left no record),
    2 when the command line or a run can't be read. `store_of` opens the evidence store, `run` starts a process and
    `wait` sleeps a number of seconds."""
    ap = argparse.ArgumentParser(prog="verity bench memory", description=__doc__.split("\n\n")[0])
    ap.add_argument("period", nargs="?", help="the node run: the challenger's period, a name its ledger hasn't used")
    ap.add_argument("--dry-run", action="store_true", help="the node run with nothing sent to either gate; no number")
    ap.add_argument("--from", dest="run", metavar="RUN", help="print from a recorded run")
    ap.add_argument("--session", type=int, metavar="K", help="the session, in a run that holds several")
    ap.add_argument("--loopback", action="store_true", help="both processes on this machine; certifies nothing")
    a = ap.parse_args(argv)
    if sum((a.period is not None, a.run is not None, a.loopback)) != 1:
        ap.error("one of PERIOD (the node run), --from RUN or --loopback")
    try:
        if a.loopback:
            doc = loopback(run)
        elif a.run:
            doc = from_run(store_of(), a.run, a.session)
        else:
            doc = node(store_of(), a.period, a.dry_run, a.session, run, wait)
            if isinstance(doc, int):
                return doc
        held = SETTINGS.RESEARCH_RUN_DIR
        # a run that ran the loopback keeps a loopback's document, the mark `--from` that run reads
        kept = bool(held) and not doc["loopback"] and left(Path(held)).get("loopback") is True
        ledger = doc.pop("ledger")
    except NoNumber as e:
        print(f"memory: no number: {e}", file=sys.stderr)
        return 1
    except Unreadable as e:
        print(f"memory: {e}", file=sys.stderr)
        return 2
    doc["beside"] = beside(store_of)
    print(text(doc))
    line = json.dumps(doc)
    print(line)
    if kept:
        print("memory: this run ran the loopback, and its out/memory.json stays the loopback's document", file=sys.stderr)
    elif held:
        (Path(held) / "out").mkdir(parents=True, exist_ok=True)
        (Path(held) / "out" / "memory.json").write_text(line + "\n")
    if held and not doc["loopback"]:
        result(Path(held), amounts(doc, ledger))
    return 0


if __name__ == "__main__":
    sys.exit(main())
