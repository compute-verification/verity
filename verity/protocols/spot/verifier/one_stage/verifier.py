"""The one-stage audit's verifier: its Lean port (`sampled-proofs-verifier`, the Lake package
`verity/protocols/spot/verifier/lean`, built with `lake` on first use) and this package's Python, the reference, run side by
side on every check.

`RECORD_VERDICT` names the one whose verdict is the record's: `"lean"`, or `"python"` to switch back. Either way both run on
every call, both outcomes are recorded beside the one of record (a `Verdict`'s `verifiers`, an audit record's
`verifiers`), and where they disagree the check is refused naming both (`DISAGREE`): a `Verdict` failing with that code
alone, or an audit record that does not accept, has no profiles and fails its `verifiers` check. Lean that can't be built
or run, or that refuses an input (a form its decoders don't read), disagrees with any Python outcome: it fails closed.

* `check`: R1-R8 (`registration.check`), compared on `ok`, the codes and the fan-out and tied values a template query's
  verdict publishes (`Verdict.fanout` and `Verdict.rests_on_period_check`, the Lean's `fanout` and
  `rests_on_period_check`); the details are Python's (the Lean reports codes).
  History tables both refuse are the registration's refusal, `R.HISTORY`, with both reasons; one refusing alone disagrees;
* `matches_public`, `check_draw`, `check_proved`, and the proof system's public-items rule (`check_public`) and hold on
  an outcome's verdicts (`check_verdicts`), whose Python verdict the service passes in: compared whole, details included;
* `select`: the challenger's sample (`service.select_draw`, which the service passes in with the bytes of its coins that
  sample read), compared whole with the number of bytes read; the sample of record is the Lean's;
* `coins`: the entropy a challenger flips its coins from (`challenger.LiveChallenger`, on the challenger's host), which
  only the Lean reads from the operating system (`IO.getRandomBytes`, A3); there is no Python twin to compare, and a
  Lean that can't answer leaves no coins. The verifier reads none of it: its samples read the challenger's coins from its
  log (`run`'s `coins`);
* `run`: the stream's run of one registration from its public check through its verdict (`Run.run`), which `Stream.outcome`
  passes in with everything it read on the way: compared whole;
* `audit_record`: the whole record, equal under `canonical_json`. The registered roots the record's `registered` check
  reads are the verdicts' own on both sides, for the registration's own values (`audit.session_registered`), never an
  argument. Python's `delta` is a float, given to the Lean exactly (`str(Fraction(delta))`); the Lean writes it, and any
  float in the inputs, as Python's `repr` does. A float the Lean can't read back (`-0.0`, NaN, an infinity) makes the two
  disagree.

Building takes the lock `verity_difftest.build` takes (its `BUILD_LOCK`, which `test_verifier.py` holds this one to): Lake
builds the path requires in place, and two builds at once would rewrite one module's C under each other. This package
imports only the standard library and `verity`, so the convention is restated here rather than imported.
"""
from __future__ import annotations

import atexit
import fcntl
import functools
import json
import math
import secrets
import subprocess
import tempfile
import threading
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from verity.circuits import codec
from verity.circuits.codec import canonical_json
from verity.circuits.program import Program

from . import audit, draw as D, hello as H, registration as R
from .verdict import Verdict

RECORD_VERDICT = "lean"
"""Whose verdict is the record's: `"lean"` (`sampled-proofs-verifier`) or `"python"` (this package's, the reference)."""

DISAGREE = "disagree"
ROOT = Path(__file__).resolve().parents[5]
PACKAGE = ROOT / "verity/protocols/spot/verifier/lean"
EXE = "sampled-proofs-verifier"
BUILD_LOCK = ROOT / "verity/difftest/lean/.lake/harness-build.lock"
#: The δ the Lean's registration parameters carry, which R1-R8 never read.
REGISTRATION_DELTA = "1/2"


class LeanError(RuntimeError):
    """The Lean verifier can't be built or run, or answered outside its protocol."""


@functools.cache
def executable(exe: str = EXE) -> Path:
    """`lake build sampled-proofs-verifier` (or `exe`), once per process, under `BUILD_LOCK`; the executable's path."""
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            p = subprocess.run(["lake", "build", exe], cwd=PACKAGE, capture_output=True, text=True)
        except FileNotFoundError:
            raise LeanError(f"lake is not on the PATH, so {exe} in {PACKAGE} cannot be built") from None
    if p.returncode != 0:
        raise LeanError(f"lake build {exe} in {PACKAGE} failed ({p.returncode}):\n{p.stdout[-2000:]}{p.stderr[-2000:]}")
    return PACKAGE / ".lake" / "build" / "bin" / exe


def _dumps(value: Any) -> str:
    """`value` as one JSON line the Lean reads back as Python wrote it: a float of 10^16 or more, which Python writes in
    exponent form and `Flock.JsonParse.parse` would read as an integer, is written with its digits and `.0`."""
    nonce, big = secrets.token_hex(8), {}

    def walk(x: Any) -> Any:
        if isinstance(x, float) and math.isfinite(x) and abs(x) >= 1e16:
            token = f"\0{nonce}:{len(big)}\0"
            big[json.dumps(token)] = format(x, ".1f")
            return token
        if isinstance(x, Mapping):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [walk(v) for v in x]
        return x

    text = json.dumps(walk(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    for token, literal in big.items():
        text = text.replace(token, literal)
    return text


class _Server:
    """One serving process of `exe` per Python process (`Difftest.serve`: a request line, an answer line), started at
    the first call and again after it dies."""

    def __init__(self, exe: str = EXE) -> None:
        self.exe = exe
        self._proc: subprocess.Popen | None = None
        self._stderr: Any = None
        self._lock = threading.Lock()

    def call(self, fn: str, value: Any) -> tuple[str, Any]:
        """`("output", JSON)` or `("reject", reason)`; anything else raises `LeanError`."""
        line = _dumps({"fn": fn, "input": value})
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._stderr = tempfile.TemporaryFile()
                self._proc = subprocess.Popen([str(executable(self.exe))], stdin=subprocess.PIPE,
                                              stdout=subprocess.PIPE, stderr=self._stderr, text=True, encoding="utf-8")
            try:
                self._proc.stdin.write(line + "\n")
                self._proc.stdin.flush()
                reply = self._proc.stdout.readline()
            except BrokenPipeError:
                reply = ""
            if not reply.endswith("\n"):
                self._proc.kill()
                self._proc.wait()
                self._stderr.seek(0)
                err = self._stderr.read().decode("utf-8", "replace")[-2000:]
                self._proc = None
                raise LeanError(f"{self.exe} ended on {fn}: {err}")
        try:
            msg = json.loads(reply)
        except ValueError:
            raise LeanError(f"{self.exe} on {fn} answered a line that is not JSON: {reply[:500]}") from None
        if isinstance(msg, dict) and len(msg) == 1:
            (kind, body), = msg.items()
            if kind == "output" or (kind == "reject" and isinstance(body, str)):
                return kind, body
        raise LeanError(f"{self.exe} on {fn}: {reply[:500]}")

    def close(self) -> None:
        """End the server: `Difftest.serve` stops at the end of its input."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.stdin.close()
                self._proc.wait()
            self._proc = None


_server = _Server()
atexit.register(_server.close)


def lean(fn: str, value: Any) -> dict[str, Any]:
    """The Lean's answer: its output, or `{"refused": reason}` where it refuses the input or can't run."""
    try:
        kind, body = _server.call(fn, value)
    except LeanError as e:
        return {"refused": str(e)}
    return body if kind == "output" else {"refused": body}


#: The most bytes one `coins` request takes (the Lean's `COINS_MAX`).
COINS_MAX = 1 << 24


def coins(n: int) -> bytes:
    """`n` bytes of the operating system's randomness, read by the Lean verifier (`IO.getRandomBytes`, A3's
    `uniform/io-getrandombytes`): a challenger's entropy (`challenger.LiveChallenger`'s, on its own host), which the
    Python never reads from the operating system itself. No stream or verifier samples from it: they read the challenger's
    coins from its log. Raises `LeanError` where the Lean can't run, refuses or answers other than `n` bytes: no
    coins."""
    if not (type(n) is int and 0 <= n <= COINS_MAX):
        raise ValueError(f"coins are 0 to {COINS_MAX} bytes a request, not {n!r}")
    kind, body = _server.call("coins", n)
    if kind != "output":
        raise LeanError(f"{EXE} refused {n} coin bytes: {body}")
    if not (isinstance(body, str) and len(body) == 2 * n and all(c in "0123456789abcdef" for c in body)):
        raise LeanError(f"{EXE} answered other than {n} coin bytes in hex")
    return bytes.fromhex(body)


#: the claim instance naming `coins`'s bytes, which a served challenger reads (`challenger.coin_claim`)
CLAIM = "uniform/io-getrandombytes"
coins.claim = CLAIM  # type: ignore[attr-defined]


def _canon(x: Any) -> str | None:
    """`x` under `canonical_json`, or None where it isn't JSON (which then equals nothing)."""
    try:
        return canonical_json(x)
    except (TypeError, ValueError):
        return None


def _same(a: Any, b: Any) -> bool:
    ca = _canon(a)
    return ca is not None and ca == _canon(b)


def _summary(v: Mapping[str, Any]) -> dict[str, Any]:
    return {"refused": v["refused"]} if "refused" in v else {"ok": v["ok"], "codes": list(v["codes"])}


def _decide(got: Mapping[str, Any], py: Verdict, whole: bool) -> Verdict:
    """The verdict of record from the Lean's answer `got` and Python's `py`: compared on `ok`, the codes and the fan-out
    and tied values a registration verdict publishes, and with `whole` on the details too."""
    mine = py.to_json()
    both = {"lean": _summary(got), "python": _summary(mine)}
    verifiers = {"of_record": RECORD_VERDICT, **both}
    same = "refused" not in got and (_same(got, mine) if whole else both["lean"] == both["python"]
                                     and all(got.get(k) == mine.get(k) for k in ("fanout", "rests_on_period_check")))
    if not same:
        return Verdict(False, (DISAGREE,), {DISAGREE: {"lean": dict(got), "python": mine}}, verifiers, py.fanout,
                       py.rests_on_period_check)
    if RECORD_VERDICT == "lean":
        return Verdict(got["ok"], tuple(got["codes"]), dict(got.get("detail", py.detail)), verifiers, py.fanout,
                       py.rests_on_period_check)
    return Verdict(py.ok, py.codes, py.detail, verifiers, py.fanout, py.rests_on_period_check)


def _params(program: Program, query: Mapping[str, Any], law: Mapping[str, Any], scheme: str,
            ports: Mapping[str, Mapping[str, Any]], leaf_layer: str | None, anchors: Sequence[Any],
            log: Iterable[Mapping[str, Any]], partition: Mapping[str, Any] | None, work_table: Mapping[str, Any] | None,
            registered: Mapping[str, int] | None, supplied: Mapping[str, Any] | None = None,
            pinned: Mapping[str, str] | None = None, binding: Mapping[str, str] | None = None,
            schemas: Mapping[str, str] | None = None, window_root: str | None = None,
            held: Sequence[str] | None = None, outputs: Mapping[str, Mapping[str, str]] | None = None,
            day: int | None = None, declarations: Sequence[Mapping[str, Any]] | None = None,
            seed_salt: str | None = None, pool: Mapping[str, Any] | None = None,
            tied: Sequence[str] | None = None, history: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """`registration.check`'s copies as the Lean's parameters (`OneStage.paramsOfJson`), `pool` as the verifier holds it:
    the Lean adds its `pin` to the supplied or output values itself. The program function's parameters go with
    `registered` or `tied`, which name them."""
    p = {"anchors": list(anchors), "delta": REGISTRATION_DELTA, "law": dict(law), "leaf_layer": leaf_layer,
         "log": [dict(r) for r in log], "partition": None if partition is None else dict(partition),
         "ports": {k: dict(v) for k, v in ports.items()}, "program": codec.encode_program(program), "query": dict(query),
         "registered": None if registered is None else dict(registered), "scheme": scheme,
         "work_table": None if work_table is None else dict(work_table)}
    if registered is not None or tied is not None:
        p["parameters"] = [[name, t.to_json()] for name, t in program.fn.signature()[0]]
    if tied is not None:
        p["tied"] = list(tied)
    if supplied:
        p["supplied"] = {v: [[dict(s) for s in row] for row in rows] for v, rows in supplied.items()}
    if pinned:
        p["pinned"] = dict(pinned)
    if binding is not None:
        p["binding"] = dict(binding)
    if schemas is not None:
        p["row_schemas"] = dict(schemas)
    if window_root is not None:
        p["window_root"] = window_root
    if held is not None:
        p["held"] = list(held)
    if outputs is not None:
        p["outputs"] = {v: dict(o) for v, o in outputs.items()}
    if day is not None or declarations is not None:
        p["day"] = day
        p["declarations"] = None if declarations is None else [dict(d) for d in declarations]
    if seed_salt is not None:
        p["seed_salt"] = seed_salt
    if pool is not None:
        p["pool"] = dict(pool)
    if history is not None:
        p["history"] = dict(history)
    return p


def check(rec: Mapping[str, Any], program: Program, query: Mapping[str, Any], law: Mapping[str, Any], scheme: str,
          ports: Mapping[str, Mapping[str, Any]], leaf_layer: str | None = None, anchors: Sequence[Any] = (),
          log: Iterable[Mapping[str, Any]] = (), partition: Mapping[str, Any] | None = None,
          work_table: Mapping[str, Any] | None = None, registered: Mapping[str, int] | None = None,
          supplied: Mapping[str, Any] | None = None, pinned: Mapping[str, str] | None = None,
          binding: Mapping[str, str] | None = None, schemas: Mapping[str, str] | None = None,
          window_root: str | None = None, held: Sequence[str] | None = None,
          outputs: Mapping[str, Mapping[str, str]] | None = None, day: int | None = None,
          declarations: Sequence[Mapping[str, Any]] | None = None, seed_salt: str | None = None,
          pool: Mapping[str, Any] | None = None, tied: Sequence[str] | None = None,
          history: Mapping[str, Any] | None = None) -> Verdict:
    """`registration.check` (R1-R8) by the verifier, with R2's declared day when it passes `day` and `declarations`,
    R6's seed salt when it passes `seed_salt`, R6's pool reset, week and spent mark, a reset at most once a week,
    and R7's pool chain when it passes `pool`, R8's tied values when it passes `tied`, and the roots of the history
    tables it passes as `history`, which the Lean computes from the members' public bodies and puts in place of the
    supplied values' `history` sources. History tables both verifiers refuse fail it with `R.HISTORY` alone."""
    log = list(log)
    try:
        py: Verdict | None = R.check(rec, program, query, law, scheme, ports, leaf_layer, anchors, log, partition,
                                     work_table, registered, supplied, pinned, binding, outputs=outputs, schemas=schemas,
                                     window_root=window_root, held=held, day=day, declarations=declarations,
                                     seed_salt=seed_salt, pool=pool, tied=tied, history=history)
        why = None
    except R.HistoryRefused as e:
        py, why = None, str(e)
    got = lean("registration", {"params": _params(program, query, law, scheme, ports, leaf_layer, anchors, log, partition,
                                                  work_table, registered, supplied, pinned, binding, schemas,
                                                  window_root, held, outputs, day, declarations, seed_salt, pool, tied,
                                                  history),
                       "record": dict(rec)})
    if "codes" in got:
        got = {"ok": not got["codes"], "codes": got["codes"],
               **{k: got[k] for k in ("fanout", "rests_on_period_check") if k in got}}
    if py is None:
        verifiers = {"of_record": RECORD_VERDICT, "lean": _summary(got), "python": {"refused": why}}
        if "refused" in got:
            return Verdict(False, (R.HISTORY,), {R.HISTORY: {"lean": got["refused"], "python": why}}, verifiers)
        return Verdict(False, (DISAGREE,), {DISAGREE: {"lean": dict(got), "python": {"refused": why}}}, verifiers)
    return _decide(got, py, whole=False)


def matches_public(rec: Mapping[str, Any], scheme: str, roots: Mapping[str, str], domains: Mapping[str, str],
                   population: int) -> Verdict:
    """`registration.matches_public` by the verifier."""
    py = R.matches_public(rec, scheme, roots, domains, population)
    got = lean("matches_public", {"record": dict(rec), "scheme": scheme, "roots": dict(roots), "domains": dict(domains),
                                  "population": population})
    return _decide(got, py, whole=True)


def check_public(python: Verdict, record: Any, scheme: str, query: Mapping[str, Any], values: Iterable[tuple[str, bool]],
                 declared: Iterable[str]) -> Verdict:
    """The proof system's public-items rule by the verifier. Its Python is the service's (`service.check_public`,
    which imports this module), so `python` is that verdict, on `record` with `scheme`, the check's `query`, the values as
    `(name, registered)` and the Spec's declared items."""
    got = lean("check_public", {"record": record, "scheme": scheme, "query": dict(query),
                                "values": [{"name": n, "registered": r} for n, r in values], "public": list(declared)})
    return _decide(got, python, whole=True)


def check_verdicts(python: Verdict, verdicts: Mapping[str, Any], sessions: Sequence[str], outputs: Sequence[str]) -> Verdict:
    """The proof system's hold on an outcome's verdicts (`service.check_verdicts`, whose verdict is `python`) by the
    verifier, on the verdicts, the delivered sessions and the check's public outputs."""
    got = lean("check_verdicts", {"verdicts": verdicts, "sessions": list(sessions), "outputs": list(outputs)})
    return _decide(got, python, whole=True)


def hello(own: Mapping[str, Any], received: str, coin_tree: Any, seen: Sequence[str], selection: Sequence[str], slots: int,
          session: Sequence[str]) -> Verdict:
    """The run's Hello on its record (`hello.check_run`) by the verifier: R7 with the prover's nonce, the run's
    streams (the selection streams, then each of `slots` session slots' `session` streams), the record's `coin_tree` and
    its key unused by the runs in `seen`."""
    py = H.check_run(own, received, coin_tree, list(seen), list(selection), slots, list(session))
    got = lean("hello", {"own": dict(own), "hello": received, "coin_tree": coin_tree, "seen": list(seen),
                         "selection": list(selection), "slots": slots, "session": list(session)})
    return _decide(got, py, whole=False)


def select(python: Mapping[str, Any], read: bytes, rule: Mapping[str, Any], program: Program, query: Mapping[str, Any],
           work_table: Mapping[str, Any] | None = None,
           closure: Mapping[int, Sequence[int]] | None = None) -> tuple[dict[str, Any], Verdict]:
    """The challenger's sample by the verifier: the Lean's `Select.select` on `read`, the bytes of the challenger's
    coins that the Python sample `python` (`service.select_draw`'s) read in order, under the Spec's selection rule (its
    `to_json`) over the population `program` and the check's `query`, with the work table and the closure map. The two
    agree when the Lean samples the same object and reads exactly `read`; the sample of record is then `RECORD_VERDICT`'s. Where
    they disagree, or the Lean refuses, the verdict fails with `DISAGREE` alone (the sample returned is Python's, and the
    caller refuses it)."""
    got = lean("select", {"rule": dict(rule), "program": codec.encode_program(program), "query": dict(query),
                          "work_table": None if work_table is None else dict(work_table),
                          "closure": None if closure is None else [[u, list(cs)] for u, cs in closure.items()],
                          "source": read.hex()})
    mine = {"draw": dict(python), "read": len(read)}
    verifiers = {"of_record": RECORD_VERDICT, "lean": {"refused": got["refused"]} if "refused" in got else
                 {"read": got.get("read")}, "python": {"read": len(read)}}
    if "refused" in got or got.get("read") != len(read) or not _same(got.get("draw"), python):
        return dict(python), Verdict(False, (DISAGREE,), {DISAGREE: {"lean": dict(got), "python": mine}}, verifiers)
    return dict(got["draw"] if RECORD_VERDICT == "lean" else python), Verdict(True, (), {}, verifiers)


def select_second(python: Mapping[str, Any], read: bytes, sample: Sequence[int], ns: Sequence[int],
                  ks: Sequence[int]) -> tuple[dict[str, Any], Verdict]:
    """A two-stage design's second stage by the verifier: the Lean's `Select.selectSecond` on `read`, the bytes of the
    challenger's coins that the Python sample `python` (`service.select_second`'s) read in order, over the replay sample
    `sample`, each template's `ns[t]` proof blocks in each replay block, its cell, and `ks[t]` of them sampled in each;
    compared as `select` compares."""
    got = lean("select_second", {"sample": list(sample), "ns": list(ns), "ks": list(ks), "source": read.hex()})
    mine = {"draw": dict(python), "read": len(read)}
    verifiers = {"of_record": RECORD_VERDICT, "lean": {"refused": got["refused"]} if "refused" in got else
                 {"read": got.get("read")}, "python": {"read": len(read)}}
    if "refused" in got or got.get("read") != len(read) or not _same(got.get("draw"), python):
        return dict(python), Verdict(False, (DISAGREE,), {DISAGREE: {"lean": dict(got), "python": mine}}, verifiers)
    return dict(got["draw"] if RECORD_VERDICT == "lean" else python), Verdict(True, (), {}, verifiers)


def run(python: Mapping[str, Any], rec: Mapping[str, Any], program: Program, query: Mapping[str, Any],
        law: Mapping[str, Any], scheme: str, ports: Mapping[str, Mapping[str, Any]], leaf_layer: str | None,
        anchors: Sequence[Any], log: Iterable[Mapping[str, Any]], work_table: Mapping[str, Any] | None,
        registered: Mapping[str, int] | None, rule: Mapping[str, Any], values: Iterable[tuple[str, bool]],
        declared: Iterable[str], challenger: str, received: str, read: bytes,
        closure: Mapping[int, Sequence[int]] | None, sessions: Sequence[str], verdicts: Mapping[str, Mapping[str, Any]],
        outputs: Sequence[str], statement: Mapping[str, Any], delta: float, proved_units: Sequence[int] | None = None,
        instances: int | None = None, served: Sequence[int] | None = None,
        session_roots: Mapping[str, str] | None = None, *, supplied: Mapping[str, Any] | None = None,
        pinned: Mapping[str, str] | None = None, binding: Mapping[str, str] | None = None,
        schemas: Mapping[str, str] | None = None, window_root: str | None = None,
        held: Sequence[str] | None = None, coins: Mapping[str, Any] | None = None,
        value_outputs: Mapping[str, Mapping[str, str]] | None = None,
        interior: Mapping[str, Any] | None = None, seed_salt: str | None = None,
        pool: Mapping[str, Any] | None = None, tied: Sequence[str] | None = None,
        history: Mapping[str, Any] | None = None, family: str | None = None) -> Verdict:
    """The stream's run of one registration, from its public check through its verdict, by the Lean (`Run.run`), held whole to
    `python`, the stream's own (`{"receipt", "draw", "read", "symbol", "public", "audit"}`, the audit record without its
    `verifiers`): on `check`'s copies (``value_outputs`` its `outputs`, the registered values an earlier record of the
    log registered first, ``seed_salt`` its window's seed salt, ``pool`` its pool, ``tied`` the tied values its check
    let through, ``history`` its history tables), `check_public`'s values and declared items, the receipt's challenger and
    place in
    the log, `select`'s rule, the bytes of the coins the sample read and the closure map, and `audit_record`'s arguments
    with the sessions' roots whole. The verifier has no coins of its own: for a sample that read coins the Lean also checks
    `coins`, their evidence (`coins.coins_reason`'s), that the bytes the sample read are the coins of the challenger's
    family `family` (the one the stream read them from, never the evidence's) that the session the ledger
    opens for `window_root` released after the registration's digest, and refuses the select step where they aren't or
    `coins` or `family` is missing (code `coins`), and where the family's log shows a root other than the registration's
    digest (code `family`); for one that read none, `coins` must be None. Where the two differ,
    or the Lean refuses a step, the verdict fails with `DISAGREE` alone.

    Under a two-stage design `interior` is the interior registration and the second stage (`Run.Interior`): its
    `program`, `query`, `law` and `copies` (as `check` takes them, the binding `binding`, its window's `seed_salt`
    when it has one, and its `history` tables when it has any), one replay block's Program
    (`replay`), its `values`, `record` and `received`, the bytes the second stage read (`read`), their evidence
    (`coins`) and their family (`family`, the second stage's own: a family's coins follow one registration), its
    `proved_units`, `instances`, `served` and `session_roots`, the second-stage sessions' `statements`,
    and `proofs`, the auditor's params naming each template's circuit file and reference header
    (`service.template_circuits`'s form, under the interior's params; None names none); the Lean derives the design's
    links from the replay block's Definition.
    `python` then holds its `{"receipt", "draw", "read", "audit"}` under `interior`, its audit record with round 2's
    `statements` check."""
    exact = str(Fraction(delta)) if isinstance(delta, (int, float)) and math.isfinite(delta) else delta
    args = {"params": _params(program, query, law, scheme, ports, leaf_layer, anchors, log, None, work_table,
                              registered, supplied, pinned, binding, schemas, window_root, held, value_outputs,
                              seed_salt=seed_salt, pool=pool, tied=tied, history=history),
            "record": dict(rec), "rule": dict(rule), "values": [{"name": n, "registered": r} for n, r in values],
            "public": list(declared), "challenger": challenger, "received": received, "source": read.hex(),
            "closure": None if closure is None else [[u, list(cs)] for u, cs in closure.items()],
            "sessions": list(sessions), "verdicts": {k: dict(verdicts[k]) for k in sorted(verdicts)},
            "outputs": list(outputs), "statement": dict(statement), "delta": exact,
            "proved_units": None if proved_units is None else list(proved_units), "instances": instances,
            "served": None if served is None else list(served),
            "session_roots": None if session_roots is None else dict(session_roots),
            "coins": None if coins is None else dict(coins), "family": family}
    if interior is not None:
        it = interior
        i_ports, i_leaf, i_anchors, i_log, i_registered, i_supplied, i_pinned, i_schemas, i_window, i_held, i_outputs = \
            it["copies"]
        i_params = _params(it["program"], it["query"], it["law"], scheme, i_ports, i_leaf, i_anchors, i_log, None, None,
                           i_registered, i_supplied, i_pinned, binding, i_schemas, i_window, i_held, i_outputs,
                           seed_salt=it.get("seed_salt"), history=it.get("history"))
        if it.get("proofs") is not None:
            i_params["proofs"] = it["proofs"]
        args["interior"] = {
            "params": i_params,
            "replay": codec.encode_program(it["replay"]),
            "values": [{"name": n, "registered": r} for n, r in it["values"]], "record": dict(it["record"]),
            "received": it["received"], "source": it["read"].hex(),
            "coins": None if it["coins"] is None else dict(it["coins"]), "family": it.get("family"),
            "proved_units": None if it["proved_units"] is None else list(it["proved_units"]),
            "instances": it["instances"], "served": None if it["served"] is None else list(it["served"]),
            "session_roots": None if it["session_roots"] is None else dict(it["session_roots"]),
            "statements": None if it.get("statements") is None else list(it["statements"])}
    got = lean("run", args)
    for path in (("audit",), ("interior", "audit")):
        at = got
        for key in path[:-1]:
            at = at.get(key) if isinstance(at, dict) else None
        if isinstance(at, dict) and isinstance(at.get(path[-1]), str):
            try:
                at[path[-1]] = json.loads(at[path[-1]])
            except ValueError:
                got = {"refused": f"{EXE} answered a run whose record is not JSON: {at[path[-1]][:500]}"}
                break
    side = (lambda r: {"refused": r["refused"]} if "refused" in r else
            {"step": r["step"], "codes": r["codes"]} if "step" in r else {"symbol": r.get("symbol")})
    verifiers = {"of_record": RECORD_VERDICT, "lean": side(got), "python": side(python)}
    if "refused" in got or "step" in got or not _same(got, python):
        fields = sorted(k for k in set(python) | set(got) if not _same(got.get(k), python.get(k)))
        return Verdict(False, (DISAGREE,), {DISAGREE: {"fields": fields, "lean": {k: got.get(k) for k in fields},
                                                       "python": {k: python.get(k) for k in fields}}}, verifiers)
    return Verdict(True, (), {}, verifiers)


def check_draw(draw: Any, registration: Mapping[str, Any], law: Mapping[str, Any]) -> Verdict:
    """`draw.check_draw` by the verifier."""
    py = D.check_draw(draw, registration, law)
    got = lean("check_draw", {"draw": draw, "population": registration.get("population"), "law": dict(law)})
    return _decide(got, py, whole=True)


def check_proved(draw: Mapping[str, Any], proved_units: Sequence[int], instances: int) -> Verdict:
    """`draw.check_proved` by the verifier."""
    py = D.check_proved(draw, proved_units, instances)
    got = lean("check_proved", {"draw": dict(draw), "proved_units": list(proved_units), "instances": instances})
    return _decide(got, py, whole=True)


def stream(params: Mapping[str, Any], state: Mapping[str, Any], op: Mapping[str, Any], decision: Mapping[str, Any],
           after: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Verdict]:
    """A proof system stream's rules on `op` (the Lean's `Stream.run` on `state`) against Python's `decision` and the
    state `after` it (`service._step`'s), compared without Python's `detail`, which the decision of record keeps: the
    decision and the state of record, and the verdict, `disagree` where the two differ."""
    return _ruled("stream", {"params": dict(params), "state": dict(state), "op": dict(op)}, decision, after)


def fleet(state: Mapping[str, Any], op: Mapping[str, Any], decision: Mapping[str, Any],
          after: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Verdict]:
    """A proof system fleet's rules on `op` (the Lean's `Fleet.run` on `state`) against Python's `decision` and the state
    `after` it (`service._fleet_step`'s), as `stream` compares a stream's."""
    return _ruled("fleet", {"state": dict(state), "op": dict(op)}, decision, after)


def _ruled(piece: str, args: dict[str, Any], decision: Mapping[str, Any],
           after: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Verdict]:
    got = lean(piece, args)
    try:
        theirs = {"decision": got["decision"], "state": json.loads(got["state"])}
    except (KeyError, TypeError, ValueError):
        theirs = {"error": got.get("refused")}
    mine = {"decision": {k: v for k, v in decision.items() if k != "detail"}, "state": dict(after)}
    verifiers = {"of_record": RECORD_VERDICT, "lean": theirs, "python": mine}
    if not _same(theirs, mine):
        return dict(decision), dict(after), Verdict(False, (DISAGREE,), {DISAGREE: {"lean": theirs, "python": mine}},
                                                    verifiers)
    rec = theirs if RECORD_VERDICT == "lean" else mine
    d = {**rec["decision"], **({"detail": decision["detail"]} if "detail" in decision else {})}
    return d, rec["state"], Verdict(True, (), {}, verifiers)


def _record_summary(r: Mapping[str, Any]) -> dict[str, Any]:
    if "refused" in r:
        return {"refused": r["refused"]}
    return {"accepted": r["accepted"], "failed": [k for k, c in r["checks"].items() if not c["ok"]]}


def audit_record(registration: Mapping[str, Any], receipt: Mapping[str, Any], draw: Mapping[str, Any],
                 law: Mapping[str, Any], proved_units: Sequence[int], instances: int, statement: Mapping[str, Any],
                 verdicts: Mapping[str, Mapping[str, Any]], delta: float, sessions: Sequence[str] = (),
                 served: Sequence[int] | None = None, session_roots: Mapping[str, str] | None = None, *,
                 own_reads: bool = False) -> dict[str, Any]:
    """`audit.audit_record` by the verifier, with `session_registered` the verdicts' own for the registration's
    values (`audit.session_registered(verdicts, registration)`), with `own_reads` (a two-stage run's, whose round 2
    holds those reads' rows) less the reads of the registration's own ports at their own roots (`Record.lessOwnReads`):
    the record of record, with `verifiers` (`{"of_record", "lean", "python", "agree"}`, each side's acceptance and failed
    checks)."""
    verdicts = {k: dict(verdicts[k]) for k in sorted(verdicts)}
    opened = audit.session_registered(verdicts, registration)
    if own_reads and opened is not None:
        roots = {r["port"]: r["root"] for r in registration["roots"]}
        opened = {v: r for v, r in opened.items() if not (v in roots and roots[v] == r)}
    py = audit.audit_record(registration, receipt, draw, law, list(proved_units), instances, statement, verdicts, delta,
                            sessions, served, session_roots, opened)
    exact = str(Fraction(delta)) if isinstance(delta, (int, float)) and math.isfinite(delta) else delta
    got = lean("record", {"registration": dict(registration), "receipt": dict(receipt), "draw": dict(draw),
                          "law": dict(law), "proved_units": list(proved_units), "instances": instances,
                          "statement": dict(statement), "verdicts": verdicts, "delta": exact,
                          "sessions": list(sessions), "served": None if served is None else list(served),
                          "session_roots": None if session_roots is None else dict(session_roots),
                          **({"own_reads": True} if own_reads else {})})
    if isinstance(got, str):
        try:
            got = json.loads(got)
        except ValueError:
            got = {"refused": f"{EXE} answered a record that is not JSON: {got[:500]}"}
    if not isinstance(got, dict) or not ("refused" in got or isinstance(got.get("checks"), dict)):
        got = {"refused": f"{EXE} answered a record that is not one: {str(got)[:500]}"}
    agree = "refused" not in got and _same(got, py)
    verifiers = {"of_record": RECORD_VERDICT, "lean": _record_summary(got), "python": _record_summary(py), "agree": agree}
    rec = dict(got if RECORD_VERDICT == "lean" and "refused" not in got else py)
    if not agree:
        fields = sorted(k for k in set(py) | set(got) if "refused" in got or not _same(got.get(k), py.get(k)))
        rec.update(accepted=False, profile=None, integrity_profile=None, wrong_units=None)
        rec["checks"] = {**rec["checks"], "verifiers": {"ok": False, "codes": [DISAGREE], "detail": {
            "fields": fields, "lean": got if "refused" in got else {k: got.get(k) for k in fields},
            "python": {k: py.get(k) for k in fields}}}}
    rec["verifiers"] = verifiers
    return rec


__all__ = ["BUILD_LOCK", "COINS_MAX", "DISAGREE", "EXE", "LeanError", "PACKAGE", "RECORD_VERDICT", "audit_record",
           "check", "check_draw", "check_proved", "check_public", "check_verdicts", "coins", "executable", "fleet", "lean",
           "matches_public", "run", "select", "select_second", "stream"]
