"""Sampled proofs' v2 case and onsite's registration case for a served window, from its records audit, each beside its
own Lean answer, for `composite.py` (`served_composite.sh` runs this after the record):

    served_cases.py (--audit DIR | --serve-dir D) --private DIR [--alpha N/D --gamma G] [--log FILE]
                    [--v2-exe PATH] [--registration-exe PATH]

``--audit`` is the window's `records_audit.py audit --circuit --registered` out directory (`served_verdicts.sh`'s
``$VW/audit``); ``--serve-dir`` finds it as the one ``D/v3c-w*/audit`` that holds a case. Both cases commit to the same
bytes, the window's registration record (``window/case.json``'s commitment, the record the audit drew against), and
each Lean answer must name that record's digest, SHA-512(format ‖ 0 ‖ canon(record)) (`registration.digest`), or nothing
is written: that digest is what remote's and onsite's record tie compare.

* **`verity/sampled-proofs/one-stage/v2`** (`v2_case`): a5's ``window/case-v2.json`` (`a5.case_v2`: v1's case with the
  verdicts replaced by the drawn member's proofs' files, and ``blobs``), refused when it commits to another record than
  v1's case or names a blob that isn't there, and when a5 wrote none (its ``case-v2-error.txt`` says why: v2 verifies
  one member's `--zk` sessions). Its answer is `sampled-proofs-one-stage-zk check` on the case and its blobs.
* **The registration case** (`registration_case`, `REGISTRATION`: `verity/inference-only/v0`; `verity/nci/v1` reads
  the same form, so a rename is `REGISTRATION` and `REGISTRATION_LEAN`):
  - params: ``alpha`` and ``gamma`` (the policy's α and Γ, ``--alpha`` and ``--gamma``; without both no case is
    written), ``log`` (the deployment's records before this one, ``--log``; empty when the record's ``previous`` is
    null), ``program`` (v1's params' Program as canonical text, the one the record's ``program`` names) and
    ``parameters`` (its program function's, which the descriptor doesn't carry: the population program rebuilt over
    its root's templates and held to the record's ``program``, `parameters`);
  - commitment: the registration record;
  - evidence: ``owner`` per root node of the Program, ``theta`` and ``provenance``. The window's Program is its
    population program (`interface.Circuit.program`), one batch node per template holding that template's units for
    every slot, so no division into one block a slot is a division of its root nodes: every non-input node is block 0,
    the window's m blocks of c gates as one of m·c, which complies only when Γ ≥ m·c. The Program registers no root
    input (its inputs are `Input<w>`), so ``theta`` and ``provenance`` are empty and every input bit counts as fresh
    against α.

Each case goes to its own directory under ``--private`` (``sampled-proofs-v2/``, ``inference-only/``; 0700, files
0600): ``case.json`` (``{"params", "commitment", "evidence"}``, hex commitment and evidence, and v2's ``blobs``) and
``answer.json``, its Lean answer line, the directory `composite.py --cases` reads. The Lean runs on files in a 0700
scratch directory under ``--private``. This prints a ``CASE {"protocol", "verdict", "dir"}`` line per case written and
a ``NOT_WRITTEN {"protocol", "reason"}`` line per case refused, and nothing of a case's params or evidence; the lines go
to a private log (a reason may quote a5's ``case-v2-error.txt``), of which only a line's ``protocol`` and ``verdict``
are public. Exit 0 when every case is written, 1 when one is refused, 64 on a usage error.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity.core.protocols.sampled_proofs.one_stage import proofs as PF
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.ml.circuits import codec

ROOT = Path(__file__).resolve().parents[2]
V2 = PF.V2
#: Onsite's registration check: the inference-only policy's v0 (`verity/nci/v1` takes the same case).
REGISTRATION = "verity/inference-only/v0"
REGISTRATION_LEAN = (ROOT / "verity/core/protocols/network/nci/lean", "inference-only")
NCI_REFERENCE = REGISTRATION_LEAN[0] / "scripts" / "nci_reference.py"
DIRS = {V2: "sampled-proofs-v2", REGISTRATION: "inference-only"}
CASE_KEYS = {"commitment", "evidence", "params"}
USAGE = 64


class Refused(Exception):
    """The audit's outputs don't make the case; the message says why."""


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise Refused(reason)


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"{path}: {type(e).__name__}: {e}") from None


def record_of(case: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """The registration record ``case`` commits to, held to be canonical JSON, and its digest (`registration.digest`)."""
    try:
        b = bytes.fromhex(case["commitment"])
        rec = json.loads(b)
    except (KeyError, TypeError, ValueError) as e:
        raise Refused(f"the case's commitment is not a registration record: {type(e).__name__}: {e}") from None
    _need(isinstance(rec, dict) and codec.canonical_json(rec) == b,
          "the case's commitment is not a registration record as canonical JSON")
    return rec, R.digest(rec)


def window_case(audit: Path) -> dict[str, Any]:
    """v1's case of the audit (a5's ``window/case.json``), whose commitment is the window's registration record."""
    case = _json(audit / "window" / "case.json")
    _need(isinstance(case, dict) and CASE_KEYS <= set(case), f"{audit}/window/case.json is not a case")
    return case


def v2_case(audit: Path) -> dict[str, Any]:
    """Sampled proofs' v2 case (module docstring): a5's ``case-v2.json`` held to v1's record and to its blobs."""
    w = audit / "window"
    v1 = window_case(audit)
    if not (w / "case-v2.json").is_file():
        why = (w / "case-v2-error.txt").read_text().strip() if (w / "case-v2-error.txt").is_file() else \
            "the audit's a5 writes no case-v2.json (a5.case_v2)"
        raise Refused(f"no v2 case: {why}")
    case = _json(w / "case-v2.json")
    _need(isinstance(case, dict) and set(case) == CASE_KEYS | {"blobs"},
          f"v2's case holds exactly {sorted(CASE_KEYS | {'blobs'})}")
    _need(case["commitment"] == v1["commitment"], "v2's case commits to another record than the audit's")
    record_of(case)
    _need(isinstance(case["blobs"], list) and all(isinstance(b, str) for b in case["blobs"]), "v2's blobs are paths")
    gone = [b for b in case["blobs"] if not Path(b).is_file()]
    _need(not gone, f"v2's case names {len(gone)} blob(s) that aren't files, the first {gone[:1]}")
    return case


def _reference():
    """`nci_reference`, the Python the policy's Lean check is held to, for the Program's root nodes."""
    spec = importlib.util.spec_from_file_location("nci_reference", NCI_REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ratio(x: Fraction | int | str) -> list[int]:
    f = Fraction(x)
    _need(f >= 0, f"α is a nonnegative ratio, not {x}")
    return [f.numerator, f.denominator]


def _chain(log: Sequence[Mapping[str, Any]], rec: Mapping[str, Any]) -> None:
    """``log`` then ``rec``, each record's ``previous`` the digest of the one before (null for the first)."""
    prev = None
    for r in [*log, rec]:
        _need(isinstance(r, Mapping) and r.get("previous") == prev,
              "the log and the record don't chain: each record's previous is the digest of the one before"
              + ("; the record names a previous record, so pass the deployment's log (--log)" if not log else ""))
        prev = R.digest(r)


def parameters(desc: Mapping[str, Any], root: Any) -> list[list[Any]]:
    """The program function's ``[[name, type], ...]``, which the descriptor doesn't carry (it inlines the function into
    its root): the population program (`population.mixed_population_program`, or `population_program` for one template)
    rebuilt over the root's template batches in order, held to the descriptor's digest."""
    from verity.ml.circuits.population import POPULATION_VERSIONS, mixed_population_program, population_program
    from verity.ml.circuits.program import is_input

    members = [(n.fn, getattr(n, "n", 1)) for n in root.body.nodes if not is_input(n.fn)]
    builds = [lambda v: mixed_population_program(members, version=v)]
    if len(members) == 1:
        builds.append(lambda v: population_program(*members[0], version=v))
    want = codec.program_digest(dict(desc))
    for build in builds:
        for v in POPULATION_VERSIONS:
            try:
                p = build(v)
            except (ValueError, TypeError) as e:
                raise Refused(f"v1's Program isn't a population program: {type(e).__name__}: {e}") from None
            if codec.program_digest(p) == want:
                return [[n, t.to_json()] for n, t in p.fn.signature()[0]]
    raise Refused("v1's Program isn't the population program over its root's templates, so its program function's "
                  "parameters aren't known")


def registration_case(v1: Mapping[str, Any], alpha: Fraction | int | str, gamma: int,
                      log: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Onsite's registration case (module docstring) from v1's case of the window's audit."""
    rec, _ = record_of(v1)
    desc = v1["params"].get("program") if isinstance(v1.get("params"), Mapping) else None
    _need(isinstance(desc, dict), "v1's params hold no Program descriptor")
    _need(codec.program_digest(desc) == rec.get("program"), "the record names another Program than v1's params")
    _need(isinstance(gamma, int) and not isinstance(gamma, bool) and gamma >= 0, f"Γ is a natural, not {gamma!r}")
    _chain(log, rec)
    text = codec.canonical_json(desc).decode()
    NR = _reference()
    try:
        inputs, _registered, _spans = NR.root_nodes(text)
        root = codec.decode_program(desc, NR.declared(desc)).root
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise Refused(f"v1's Program doesn't decode: {type(e).__name__}: {e}") from None
    params = {"alpha": _ratio(alpha), "gamma": gamma, "log": [dict(r) for r in log],
              "parameters": parameters(desc, root), "program": text}
    evidence = {"owner": ["input" if i else 0 for i in inputs], "provenance": [], "theta": []}
    return {"commitment": v1["commitment"], "evidence": codec.canonical_json(evidence).hex(), "params": params}


def lean_answer(exe: str | os.PathLike, case: Mapping[str, Any], scratch: Path) -> tuple[dict[str, Any], str]:
    """``exe check`` (`CoreProtocol.Cli`) on ``case`` and its blobs, its files in a 0700 directory under ``scratch``:
    the answer and its line."""
    scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix="case-", dir=scratch) as d:
        os.chmod(d, 0o700)
        files = [Path(d, n) for n in ("params.json", "commitment.bin", "evidence.json")]
        for f, b in zip(files, (codec.canonical_json(case["params"]), bytes.fromhex(case["commitment"]),
                                bytes.fromhex(case["evidence"]))):
            f.write_bytes(b)
            os.chmod(f, 0o600)
        p = subprocess.run([str(exe), "check", *map(str, files), *case.get("blobs", ())], capture_output=True, text=True)
    lines = p.stdout.strip().splitlines()
    try:
        answer = json.loads(lines[-1])
    except (IndexError, ValueError):
        raise Refused(f"{exe} answered no line (exit {p.returncode})") from None
    _need(isinstance(answer, dict), f"{exe}'s answer is not an object")
    return answer, lines[-1]


def tie(answer: Mapping[str, Any], id: str, digest: str) -> None:
    """The answer is ``id``'s and names the record's digest as its commitment."""
    _need(answer.get("protocol") == id, f"the answer is {answer.get('protocol')!r}'s, not {id}'s")
    _need(answer.get("commitment") == digest, f"{id}'s answer commits to {answer.get('commitment')!r}, not the "
          "window's registration record's digest")


def write(d: Path, case: Mapping[str, Any], line: str) -> None:
    """``case.json`` and ``answer.json`` in ``d``, 0700 with files 0600."""
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(d, 0o700)
    for name, body in (("case.json", json.dumps(dict(case), sort_keys=True)), ("answer.json", line)):
        fd = os.open(d / name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(body + "\n")
        os.chmod(d / name, 0o600)


def build(package: Path, name: str) -> Path:
    """``lake build name`` in ``package``; the executable's path."""
    p = subprocess.run(["lake", "build", name], cwd=package, capture_output=True, text=True)
    _need(p.returncode == 0, f"lake build {name} in {package} failed ({p.returncode})")
    return package / ".lake" / "build" / "bin" / name


def audit_of(a: argparse.Namespace) -> Path:
    if a.audit is not None:
        return a.audit
    found = sorted(d for d in a.serve_dir.glob("v3c-w*/audit") if (d / "window" / "case.json").is_file())
    _need(len(found) == 1, f"{a.serve_dir} holds {len(found)} audited windows with a case; pass --audit")
    return found[0]


def produce(a: argparse.Namespace) -> int:
    private, bad = a.private.resolve(), 0
    try:
        audit = audit_of(a)
        v1 = window_case(audit)
        _, digest = record_of(v1)
    except Refused as e:
        for id in DIRS:
            print("NOT_WRITTEN " + json.dumps({"protocol": id, "reason": str(e)}, sort_keys=True), flush=True)
        return 1

    def one(id: str, make, exe) -> None:
        nonlocal bad
        try:
            case = make()
            answer, line = lean_answer(exe() if callable(exe) else exe, case, private / "scratch")
            tie(answer, id, digest)
        except (Refused, PF.LeanError) as e:
            bad = 1
            print("NOT_WRITTEN " + json.dumps({"protocol": id, "reason": str(e)}, sort_keys=True), flush=True)
            return
        write(private / DIRS[id], case, line)
        print("CASE " + json.dumps({"dir": str(private / DIRS[id]), "protocol": id, "verdict": answer.get("verdict")},
                                   sort_keys=True), flush=True)

    one(V2, lambda: v2_case(audit), a.v2_exe or PF.v2_executable)

    def registration() -> dict[str, Any]:
        _need(a.alpha is not None and a.gamma is not None, "α and Γ are the policy's, not the run's: pass --alpha and "
              "--gamma")
        log = _json(a.log) if a.log is not None else []
        _need(isinstance(log, list), "--log is a JSON list of registration records")
        return registration_case(v1, a.alpha, a.gamma, log)

    one(REGISTRATION, registration, a.registration_exe or (lambda: build(*REGISTRATION_LEAN)))
    return bad


class Parser(argparse.ArgumentParser):
    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(USAGE, f"{self.prog}: error: {message}\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = Parser(description=__doc__.split("\n\n")[0])
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--audit", type=Path, help="the window's records_audit.py out directory")
    where.add_argument("--serve-dir", type=Path, help="the serve dir, whose one v3c-w*/audit is the audit")
    ap.add_argument("--private", type=Path, required=True, help="where each case's directory goes (0700)")
    ap.add_argument("--alpha", type=Fraction, help="the policy's α, N/D: fresh-input bits a gate")
    ap.add_argument("--gamma", type=int, help="the policy's Γ: gates an inference block")
    ap.add_argument("--log", type=Path, help="the deployment's registration records before the window's, a JSON list")
    ap.add_argument("--v2-exe", help="sampled-proofs-one-stage-zk (default: built here)")
    ap.add_argument("--registration-exe", help=f"{REGISTRATION_LEAN[1]} (default: built here)")
    return produce(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
