"""The one-stage audit's verifier of record: its Lean port (`sampled-proofs-verifier`, the Lake package
`verity/core/service/sampled_proofs`, built with `lake` on first use) and this package's Python, the reference, run side by
side on every check.

`RECORD_VERDICT` names the one whose verdict is the record's: `"lean"`, or `"python"` to switch back. Either way both run on
every call, both outcomes are recorded beside the one of record (a `Verdict`'s `verifiers`, an audit record's
`verifiers`), and where they disagree the check is refused naming both (`DISAGREE`): a `Verdict` failing with that code
alone, or an audit record that does not accept, has no profiles and fails its `verifiers` check. Lean that can't be built
or run, or that refuses an input (a form its decoders don't read), disagrees with any Python outcome: it fails closed.

* `check`: R1-R7 (`registration.check`), compared on `ok` and the codes; the details are Python's (the Lean reports codes);
* `matches_public`, `check_draw`, `check_proved`, and the proof service's public-items rule (`check_public`) and hold on
  an outcome's verdicts (`check_verdicts`), whose Python verdict the service passes in: compared whole, details included;
* `audit_record`: the whole record, equal under `canonical_json`. The registered roots the record's `registered` check
  reads are the verdicts' own on both sides, for the registration's own values (`audit.session_registered`), never an
  argument. One field is Python's on both: `wrong_units.bound` under a Bernoulli law, a float relaxation
  (`IntegrityProfile.worst_case`, through `math.log`) the Lean does not compute and leaves null. Python's `delta` is a
  float, given to the Lean exactly (`str(Fraction(delta))`); the Lean writes it, and any float in the inputs, as Python's
  `repr` does. A float the Lean can't read back (`-0.0`, NaN, an infinity) makes the two disagree.

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

from verity.ml.circuits import codec
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.program import Program

from . import audit, draw as D, registration as R
from .verdict import Verdict

RECORD_VERDICT = "lean"
"""Whose verdict is the record's: `"lean"` (`sampled-proofs-verifier`) or `"python"` (this package's, the reference)."""

DISAGREE = "disagree"
VERIFIERS = ("lean", "python")
CORE = Path(__file__).resolve().parents[3]
PACKAGE = CORE / "service" / "sampled_proofs"
EXE = "sampled-proofs-verifier"
BUILD_LOCK = CORE / "difftest" / "lean" / ".lake" / "harness-build.lock"
#: The δ the Lean's registration parameters carry, which R1-R7 never read.
REGISTRATION_DELTA = "1/2"


class LeanError(RuntimeError):
    """The Lean verifier can't be built or run, or answered outside its protocol."""


@functools.cache
def executable() -> Path:
    """`lake build sampled-proofs-verifier`, once per process, under `BUILD_LOCK`; the executable's path."""
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            p = subprocess.run(["lake", "build", EXE], cwd=PACKAGE, capture_output=True, text=True)
        except FileNotFoundError:
            raise LeanError(f"lake is not on the PATH, so {EXE} in {PACKAGE} cannot be built") from None
    if p.returncode != 0:
        raise LeanError(f"lake build {EXE} in {PACKAGE} failed ({p.returncode}):\n{p.stdout[-2000:]}{p.stderr[-2000:]}")
    return PACKAGE / ".lake" / "build" / "bin" / EXE


def _dumps(value: Any) -> str:
    """`value` as one JSON line the Lean reads back as Python wrote it: a float of 10^16 or more, which Python writes in
    exponent form and `Json.parse` would read as an integer, is written with its digits and `.0`."""
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
    """One `sampled-proofs-verifier serve` process per Python process (`Difftest.serve`: a request line, an answer line),
    started at the first call and again after it dies."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._stderr: Any = None
        self._lock = threading.Lock()

    def call(self, fn: str, value: Any) -> tuple[str, Any]:
        """`("output", JSON)` or `("reject", reason)`; anything else raises `LeanError`."""
        line = _dumps({"fn": fn, "input": value})
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._stderr = tempfile.TemporaryFile()
                self._proc = subprocess.Popen([str(executable())], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                              stderr=self._stderr, text=True, encoding="utf-8")
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
                raise LeanError(f"{EXE} ended on {fn}: {err}")
        try:
            msg = json.loads(reply)
        except ValueError:
            raise LeanError(f"{EXE} on {fn} answered a line that is not JSON: {reply[:500]}") from None
        if isinstance(msg, dict) and len(msg) == 1:
            (kind, body), = msg.items()
            if kind == "output" or (kind == "reject" and isinstance(body, str)):
                return kind, body
        raise LeanError(f"{EXE} on {fn}: {reply[:500]}")

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
    """The verdict of record from the Lean's answer `got` and Python's `py`: compared on `ok` and the codes, and with
    `whole` on the details too."""
    mine = py.to_json()
    both = {"lean": _summary(got), "python": _summary(mine)}
    verifiers = {"of_record": RECORD_VERDICT, **both}
    same = "refused" not in got and (_same(got, mine) if whole else both["lean"] == both["python"])
    if not same:
        return Verdict(False, (DISAGREE,), {DISAGREE: {"lean": dict(got), "python": mine}}, verifiers)
    if RECORD_VERDICT == "lean":
        return Verdict(got["ok"], tuple(got["codes"]), dict(got.get("detail", py.detail)), verifiers)
    return Verdict(py.ok, py.codes, py.detail, verifiers)


def _params(program: Program, query: Mapping[str, Any], law: Mapping[str, Any], scheme: str,
            ports: Mapping[str, Mapping[str, Any]], leaf_layer: str | None, anchors: Sequence[Any],
            log: Iterable[Mapping[str, Any]], partition: Mapping[str, Any] | None, work_table: Mapping[str, Any] | None,
            registered: Mapping[str, int] | None) -> dict[str, Any]:
    """`registration.check`'s copies as the Lean's parameters (`OneStage.paramsOfJson`)."""
    p = {"anchors": list(anchors), "delta": REGISTRATION_DELTA, "law": dict(law), "leaf_layer": leaf_layer,
         "log": [dict(r) for r in log], "partition": None if partition is None else dict(partition),
         "ports": {k: dict(v) for k, v in ports.items()}, "program": codec.encode_program(program), "query": dict(query),
         "registered": None if registered is None else dict(registered), "scheme": scheme,
         "work_table": None if work_table is None else dict(work_table)}
    if registered is not None:
        p["parameters"] = [[name, t.to_json()] for name, t in program.fn.signature()[0]]
    return p


def check(rec: Mapping[str, Any], program: Program, query: Mapping[str, Any], law: Mapping[str, Any], scheme: str,
          ports: Mapping[str, Mapping[str, Any]], leaf_layer: str | None = None, anchors: Sequence[Any] = (),
          log: Iterable[Mapping[str, Any]] = (), partition: Mapping[str, Any] | None = None,
          work_table: Mapping[str, Any] | None = None, registered: Mapping[str, int] | None = None) -> Verdict:
    """`registration.check` (R1-R7) by the verifier of record."""
    log = list(log)
    py = R.check(rec, program, query, law, scheme, ports, leaf_layer, anchors, log, partition, work_table, registered)
    got = lean("registration", {"params": _params(program, query, law, scheme, ports, leaf_layer, anchors, log, partition,
                                                  work_table, registered), "record": dict(rec)})
    if "codes" in got:
        got = {"ok": not got["codes"], "codes": got["codes"]}
    return _decide(got, py, whole=False)


def matches_public(rec: Mapping[str, Any], scheme: str, roots: Mapping[str, str], domains: Mapping[str, str],
                   population: int) -> Verdict:
    """`registration.matches_public` by the verifier of record."""
    py = R.matches_public(rec, scheme, roots, domains, population)
    got = lean("matches_public", {"record": dict(rec), "scheme": scheme, "roots": dict(roots), "domains": dict(domains),
                                  "population": population})
    return _decide(got, py, whole=True)


def check_public(python: Verdict, record: Any, scheme: str, query: Mapping[str, Any], values: Iterable[tuple[str, bool]],
                 declared: Iterable[str]) -> Verdict:
    """The proof service's public-items rule by the verifier of record. Its Python is the service's (`service.check_public`,
    which imports this module), so `python` is that verdict, on `record` with `scheme`, the check's `query`, the values as
    `(name, registered)` and the Spec's declared items."""
    got = lean("check_public", {"record": record, "scheme": scheme, "query": dict(query),
                                "values": [{"name": n, "registered": r} for n, r in values], "public": list(declared)})
    return _decide(got, python, whole=True)


def check_verdicts(python: Verdict, verdicts: Mapping[str, Any], sessions: Sequence[str], outputs: Sequence[str]) -> Verdict:
    """The proof service's hold on an outcome's verdicts (`service.check_verdicts`, whose verdict is `python`) by the
    verifier of record, on the verdicts, the delivered sessions and the check's public outputs."""
    got = lean("check_verdicts", {"verdicts": verdicts, "sessions": list(sessions), "outputs": list(outputs)})
    return _decide(got, python, whole=True)


def check_draw(draw: Any, registration: Mapping[str, Any], law: Mapping[str, Any]) -> Verdict:
    """`draw.check_draw` by the verifier of record."""
    py = D.check_draw(draw, registration, law)
    got = lean("check_draw", {"draw": draw, "population": registration.get("population"), "law": dict(law)})
    return _decide(got, py, whole=True)


def check_proved(draw: Mapping[str, Any], proved_units: Sequence[int], instances: int) -> Verdict:
    """`draw.check_proved` by the verifier of record."""
    py = D.check_proved(draw, proved_units, instances)
    got = lean("check_proved", {"draw": dict(draw), "proved_units": list(proved_units), "instances": instances})
    return _decide(got, py, whole=True)


def _record_summary(r: Mapping[str, Any]) -> dict[str, Any]:
    if "refused" in r:
        return {"refused": r["refused"]}
    return {"accepted": r["accepted"], "failed": [k for k, c in r["checks"].items() if not c["ok"]]}


def audit_record(registration: Mapping[str, Any], receipt: Mapping[str, Any], draw: Mapping[str, Any],
                 law: Mapping[str, Any], proved_units: Sequence[int], instances: int, statement: Mapping[str, Any],
                 verdicts: Mapping[str, Mapping[str, Any]], delta: float, sessions: Sequence[str] = (),
                 served: Sequence[int] | None = None, session_roots: Mapping[str, str] | None = None) -> dict[str, Any]:
    """`audit.audit_record` by the verifier of record, with `session_registered` the verdicts' own for the registration's
    values (`audit.session_registered(verdicts, registration)`): the record of record, with `verifiers` (`{"of_record",
    "lean", "python", "agree"}`, each side's acceptance and failed checks)."""
    verdicts = {k: dict(verdicts[k]) for k in sorted(verdicts)}
    py = audit.audit_record(registration, receipt, draw, law, list(proved_units), instances, statement, verdicts, delta,
                            sessions, served, session_roots,
                            audit.session_registered(verdicts, registration))
    exact = str(Fraction(delta)) if isinstance(delta, (int, float)) and math.isfinite(delta) else delta
    got = lean("record", {"registration": dict(registration), "receipt": dict(receipt), "draw": dict(draw),
                          "law": dict(law), "proved_units": list(proved_units), "instances": instances,
                          "statement": dict(statement), "verdicts": verdicts, "delta": exact,
                          "sessions": list(sessions), "served": None if served is None else list(served),
                          "session_roots": None if session_roots is None else dict(session_roots)})
    if isinstance(got, str):
        try:
            got = json.loads(got)
        except ValueError:
            got = {"refused": f"{EXE} answered a record that is not JSON: {got[:500]}"}
        wrong = got.get("wrong_units") if isinstance(got, dict) else None
        if draw.get("law") == "bernoulli" and isinstance(wrong, dict) and wrong.get("bound") is None and py["wrong_units"]:
            got["wrong_units"] = {**wrong, "bound": py["wrong_units"]["bound"]}
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


__all__ = ["BUILD_LOCK", "DISAGREE", "EXE", "LeanError", "PACKAGE", "RECORD_VERDICT", "audit_record", "check", "check_draw",
           "check_proved", "check_public", "check_verdicts", "executable", "lean", "matches_public"]
