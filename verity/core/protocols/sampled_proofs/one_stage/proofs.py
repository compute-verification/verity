"""A statement's proof files, verified by C-Flock's Lean verifier of record (`flock-verify verify --zk`, the Lake package
`verity/core/service/flock`, built with `lake` on first use): the proof service's own run of the verifier, so a protocol
reads each session's verdict from that run, never from a verdict record the prover hands in.

`verify` takes a statement's files as the prover delivered them (`Proof`: the circuit file, the public file and the
sessions' directories, each its `session.json` and its rep files) and the verifier's own inputs: the SHA-512 its auditor
committed for the circuit file, the program's descriptor, the partition object and, for a statement that reads
registered values, the `--registered` file. It refuses
a circuit file that doesn't hash to the committed digest before anything runs, so each verdict is over the circuit the
auditor's own lowering built. The circuit file's tie to the program's Definitions rests on that lowering: the verifier
holds the file to the program only as far as its META names it, until it derives the circuit from the descriptor.
Then one `flock-verify verify --zk` over every session (`--registered` when the statement reads registered
values), and one `Finding` a session, in order:
whether it was accepted and why not, the program its verdict names (`program_sha512`, the digest of the canonical
descriptor without annotations, which an accepted verdict must give as the partition object's `program`),
its registered finding (each port's value and opened root) and the proof blocks it proved, the public file's global proof block
indices of the instances its session drew (`unit_draw`), read only off a session the verifier accepted.

`ks_form_why` mirrors `verify --zk`'s refusal of a statement form the sampled audit's knowledge soundness
(`SampledKSLiveV4`) does not cover, on the files' heads (`Flock.KsScope.filesWhy`, which `flock-verify ks-form` runs alone).
A session Lean accepts whose files it refuses is a ``form`` refusal, so the mirror can only refuse more.

The verifier reads the whole circuit file once a call, in time and memory linear in it (about six times the file at
peak), then O(proof) a session: so a protocol verifies one statement a stratum, every drawn proof block of it an instance.

The one-stage audit's v2 (`V2`, `SampledProofs.OneStageZk`) runs this verifier inside the audit's own Lean, on the files
as content-addressed blobs: `check_v2` hands it v1's inputs with the verdicts replaced by the files' digests
(`evidence_v2`) and the files themselves, and `verdicts_v2` is that run's verdicts alone, which the difftest holds to
`verify`'s.

Building takes the lock `verifier.BUILD_LOCK` takes: Lake builds the flock package in place for `sampled-proofs-verifier`
too, and two builds at once would rewrite one module's C under each other.
"""
from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .verifier import BUILD_LOCK, CORE
from .verifier import PACKAGE as V2_PACKAGE

PACKAGE = CORE / "service" / "flock"
EXE = "flock-verify"
#: the statement every drawn proof block's session proves
STATEMENT = "verity/flock-circuit"
_DIGEST = re.compile(r"[0-9a-f]{128}")
#: an input bound at registration (`program.REGISTERED_ID`, Lean's `Program.inputId?`), a descriptor's `definitions` key
_REGISTERED_INPUT = re.compile(r"RegisteredInput[1-9][0-9]*_v1")
_CHUNK = 1 << 24
#: `Flock.KsScope.SCOPE_WHY` and `TABLES_WHY`, `verify --zk`'s last refusals (`ks_form_why`)
SCOPE_WHY = ("a circuit outside the unit's form under --zk (scopeOkH: the unit reads its one input group and returns the "
             "outputs)")
TABLES_WHY = "a shared-row public file (header shared_rows) under --zk"


class LeanError(RuntimeError):
    """The Lean verifier can't be built."""


@dataclass(frozen=True)
class Proof:
    """A statement's proof files as the prover delivered them: its circuit file, its public file, and each session's
    directory (`session.json` and its rep files)."""

    circuit: Path
    public: Path
    sessions: tuple[Path, ...]


@dataclass(frozen=True)
class Finding:
    """One session's verdict from the verifier's own run: accepted or not and why, the program its verdict names, its
    registered finding (port -> {value, root}) and the global proof blocks it proved (none unless accepted)."""

    session: str
    accepted: bool
    why: str
    program: str | None
    registered: dict
    units: tuple[int, ...]


@functools.cache
def executable() -> Path:
    """`lake build flock-verify`, once per process, under `BUILD_LOCK`; the executable's path."""
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


def file_sha512(path: Path) -> str:
    """The SHA-512 of the file at `path`, read in chunks (a circuit file can be tens of GB)."""
    h = hashlib.sha512()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _header(path: Path) -> Any:
    with open(path, "rb") as f:
        return json.loads(f.readline().decode())


def _unit_head(circuit: Path) -> tuple[dict, list[str]] | None:
    """The circuit file's META and its ``unit`` circuit's header line (``flock-ir-unit/v2 name rows const n_in_groups
    ..``), read from the file's head; None for a file not in that form (a templated one among them). A META that is no
    JSON raises: `verify` reads it only off a session Lean accepted, whose parser read it."""
    with open(circuit, "rb") as f:
        f.readline()
        meta = f.readline().decode()
        if not meta.startswith("META "):
            return None
        m = json.loads(meta[len("META "):])
        while head := f.readline().decode().split():
            if len(head) != 3 or head[0] != "CIRCUIT" or not head[2].isdigit():
                return None
            if head[1] == "unit":
                return (m, f.readline().decode().split()) if isinstance(m, dict) else None
            for _ in range(int(head[2])):
                f.readline()
    return None


def ks_form_why(circuit: Path, public: Path) -> str | None:
    """`Flock.KsScope.filesWhy` on a `verity/flock-circuit` statement's files, with `ProvedScope.refusal`'s text: a circuit
    outside the unit's form (`ProvedScope.scopeOkH`'s two conditions past `ProvedScope.check`'s: the unit reads one
    input group, and META's ``out_net``, the net returning the outputs, is the unit), then a shared-row public file
    (header ``shared_rows``); None when neither applies or the circuit file's head is not in that form. `verify` refuses a
    session Lean accepts that it refuses, so it can only refuse more; `flock-verify ks-form` is the Lean it mirrors."""
    head = _unit_head(circuit)
    if head is not None:
        meta, unit = head
        groups = unit[4] if len(unit) > 4 else None
        if meta.get("out_net", "unit") != "unit" or groups != "1":
            return _refusal(SCOPE_WHY)
    if "shared_rows" in _header(public):
        return _refusal(TABLES_WHY)
    return None


def _refusal(why: str) -> str:
    return f"refused: {why}: a statement form the soundness proof does not cover"


def reads_registered(program: bytes) -> bool:
    """Whether the program descriptor `program` has an input bound at registration (a `RegisteredInput<w>_v1` among its
    `definitions`), or has no `definitions` object to tell: True unless it names none."""
    try:
        defs = json.loads(program)["definitions"]
    except (ValueError, KeyError, TypeError):
        return True
    return not isinstance(defs, dict) or any(_REGISTERED_INPUT.fullmatch(k) for k in defs)


def proved_units(public: Path, session: Path) -> tuple[int, ...]:
    """The global proof blocks a session proved: the public file's proof block indices (``units.indices``) of the instances its record
    drew (``unit_draw.units``), ascending. Read only off a session the verifier accepted."""
    indices = _header(public)["units"]["indices"]
    drawn = json.loads((session / "session.json").read_text())["unit_draw"]["units"]
    return tuple(sorted(indices[i] for i in drawn))


def verify(proof: Proof, *, circuit_sha512: str, program: bytes, partition: Mapping[str, Any],
           registered: Mapping[str, Any] | None = None) -> tuple[list[Finding] | None, tuple[str, str] | None]:
    """Each of `proof`'s sessions verified by the verifier of record against the verifier's own `program` (its
    descriptor's bytes), `partition` and `registered` file (None only for a program that reads no registered value, such
    as N2's, whose proof block opens the committed value inside its circuit), on the circuit file the auditor committed as
    `circuit_sha512`; the findings in session order and None, or None and the refusal (``registered``: `registered` is
    None and the program reads registered values, `reads_registered`; ``circuit``: the circuit file is not the
    committed one; ``proof``: a file is missing or the verifier's answer is not one verdict a session naming the
    partition object's program; ``form``: the verifier accepted a session whose files `ks_form_why` refuses). A session
    the verifier rejects is a finding, not a refusal.

    The ``registered`` refusal is the verifier's own: without ``--registered``, Lean refuses a statement's registered
    reads only where its public file, which the prover writes, names them (`Registered.check`), and doesn't read a
    `RegisteredInput` as demanding them, so a prover that commits such a port's rows afresh and names none would be
    checked against no registered root. A run input that only the caller's registered file reads registered (the
    served RoPE statement's ``cs``) is the caller's to state: without the file, Lean holds it a run input, and refuses a
    session whose public file names its reads or whose tables share its registered rows."""
    if registered is None and reads_registered(program):
        return None, ("registered", "the program reads registered values (a RegisteredInput among its definitions) and "
                                    "the verifier holds no registered reads: verify with its own registered file")
    if not (isinstance(circuit_sha512, str) and _DIGEST.fullmatch(circuit_sha512)):
        return None, ("circuit", "the committed circuit file's digest is not a SHA-512 digest (128 lowercase hex)")
    if not (proof.circuit.is_file() and proof.public.is_file() and proof.sessions
            and all((s / "session.json").is_file() for s in proof.sessions)):
        return None, ("proof", "the statement's proof files are not a circuit file, a public file and at least one session "
                               "directory holding its session.json")
    if file_sha512(proof.circuit) != circuit_sha512:
        return None, ("circuit", "the statement's circuit file is not the one the auditor committed for its stratum (its "
                                 "SHA-512 is not the period-open's): prove on the circuit file the auditor built")
    with tempfile.TemporaryDirectory(prefix="verity-proofs-") as d:
        files = {"program": Path(d, "program.json"), "partition": Path(d, "partition.json"),
                 "registered": Path(d, "registered.json")}
        files["program"].write_bytes(program)
        files["partition"].write_text(json.dumps(partition))
        reads = []
        if registered is not None:
            files["registered"].write_text(json.dumps(registered))
            reads = ["--registered", str(files["registered"])]
        p = subprocess.run([str(executable()), "verify", "--zk", "--statement", STATEMENT, "--circuit", str(proof.circuit),
                            "--public", str(proof.public), "--partition", str(files["partition"]),
                            "--program", str(files["program"]), *reads,
                            "--session", *map(str, proof.sessions)], capture_output=True, text=True)
    lines = [json.loads(x[len("VERDICT "):]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
    if [v.get("session") for v in lines] != [str(s) for s in proof.sessions]:
        return None, ("proof", f"the verifier answered no verdict for each session in order (exit {p.returncode}): "
                               f"{(p.stderr or p.stdout)[-400:]}")
    want = partition.get("program")
    accepted = any(v.get("accepted") is True for v in lines)
    form = ks_form_why(proof.circuit, proof.public) if accepted else None
    out = []
    for s, v in zip(proof.sessions, lines):
        ok = v.get("accepted") is True
        if ok and form is not None:
            return None, ("form", f"the verifier of record accepted session {s}, whose files its mirror refuses "
                                  f"(`ks_form_why`): {form}")
        if ok and v.get("program_sha512") != want:
            return None, ("proof", f"session {s}'s verdict names program {v.get('program_sha512')}, not the partition "
                                   f"object's {want}")
        out.append(Finding(str(s), ok, str(v.get("why", "")), v.get("program_sha512"), v.get("registered") or {},
                           proved_units(proof.public, s) if ok else ()))
    return out, None


#: the one-stage audit on the proofs themselves (`SampledProofs.OneStageZk`), which runs `verify`'s verifier on blobs
V2 = "verity/sampled-proofs/one-stage/v2"
V2_EXE = "sampled-proofs-one-stage-zk"


@functools.cache
def v2_executable() -> Path:
    """`lake build sampled-proofs-one-stage-zk`, once per process, under `BUILD_LOCK`; the executable's path."""
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            p = subprocess.run(["lake", "build", V2_EXE], cwd=V2_PACKAGE, capture_output=True, text=True)
        except FileNotFoundError:
            raise LeanError(f"lake is not on the PATH, so {V2_EXE} in {V2_PACKAGE} cannot be built") from None
    if p.returncode != 0:
        raise LeanError(f"lake build {V2_EXE} in {V2_PACKAGE} failed ({p.returncode}):\n{p.stdout[-2000:]}{p.stderr[-2000:]}")
    return V2_PACKAGE / ".lake" / "build" / "bin" / V2_EXE


def _canonical(x: Any) -> bytes:
    """`x` in the canonical JSON `Core.canonicalJson` reads, for the integers, strings, lists and objects of ASCII keys
    these inputs hold."""
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def evidence_v2(proof: Proof, names: Sequence[str] | None = None) -> tuple[dict, list[Path]]:
    """The v2 evidence's ``proofs`` for `proof`'s files, and the blobs that hold them: the public file's SHA-512 and, per
    session (named `names`, the v1 evidence's ``sessions``; by default its directory's name), its ``session.json``'s and
    each rep file's (``*.bin``). The blobs are the circuit file, which the parameters name by the auditor's digest, the
    public file and each session's files."""
    names = [s.name for s in proof.sessions] if names is None else list(names)
    if len(names) != len(proof.sessions):
        raise ValueError(f"{len(names)} names for {len(proof.sessions)} sessions")
    sessions, blobs = [], [proof.circuit, proof.public]
    for name, s in zip(names, proof.sessions):
        reps = sorted(s.glob("*.bin"))
        sessions.append({"name": name, "record": file_sha512(s / "session.json"),
                         "reps": {r.name: file_sha512(r) for r in reps}})
        blobs += [s / "session.json", *reps]
    return {"public": file_sha512(proof.public), "sessions": sessions}, blobs


def verdicts_v2(proof: Proof, *, circuit_sha512: str, program: Mapping[str, Any], partition: Mapping[str, Any],
                registered: Mapping[str, Any] | None = None) -> tuple[list[Finding] | None, str | None]:
    """`verify`, by the v2 audit's own run of the verifier on `proof`'s files as blobs
    (`sampled-proofs-one-stage-zk verdicts`): the findings in session order (``session`` the directory's name) and None,
    or None and Lean's refusal. `program` is the descriptor as JSON."""
    pr, blobs = evidence_v2(proof)
    own = {"circuit": circuit_sha512, "partition": dict(partition), "program": dict(program),
           "registered": None if registered is None else dict(registered)}
    with tempfile.TemporaryDirectory(prefix="verity-proofs-v2-") as d:
        paths = (Path(d, "own.json"), Path(d, "proofs.json"))
        paths[0].write_bytes(_canonical(own))
        paths[1].write_bytes(_canonical(pr))
        p = subprocess.run([str(v2_executable()), "verdicts", *map(str, paths), *map(str, blobs)], capture_output=True,
                           text=True)
    try:
        got = json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise LeanError(f"{V2_EXE} verdicts answered no line (exit {p.returncode}): {p.stderr[-400:]}") from None
    if "refused" in got:
        return None, got["refused"]
    return [Finding(v["session"], v["accepted"] is True, v["why"], v.get("program_sha512"), v.get("registered") or {},
                    tuple(v["units"])) for v in got["verdicts"]], None


def check_v2(params: Mapping[str, Any], record: Mapping[str, Any], evidence: Mapping[str, Any], proof: Proof, *,
             circuit_sha512: str, registered: Mapping[str, Any] | None) -> dict:
    """The one-stage audit on `proof` itself (`sampled-proofs-one-stage-zk check`): v1's `params` with ``proofs``, the
    auditor's circuit digest and the verifier's registered file, the registration `record`, and v1's `evidence` with
    its ``verdicts`` replaced by `proof`'s ``proofs`` (`evidence_v2`, under the evidence's ``sessions``); the outcome
    line."""
    pr, blobs = evidence_v2(proof, evidence["sessions"])
    p2 = {**params, "proofs": {"circuit": circuit_sha512, "registered": registered}}
    e2 = {k: v for k, v in evidence.items() if k != "verdicts"} | {"proofs": pr}
    with tempfile.TemporaryDirectory(prefix="verity-proofs-v2-") as d:
        paths = [Path(d, n) for n in ("params.json", "record.json", "evidence.json")]
        for path, x in zip(paths, (p2, record, e2)):
            path.write_bytes(_canonical(x))
        p = subprocess.run([str(v2_executable()), "check", *map(str, paths), *map(str, blobs)], capture_output=True,
                           text=True)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise LeanError(f"{V2_EXE} check answered no line (exit {p.returncode}): {p.stderr[-400:]}") from None


__all__ = ["EXE", "PACKAGE", "SCOPE_WHY", "STATEMENT", "TABLES_WHY", "V2", "V2_EXE", "Finding", "LeanError", "Proof",
           "check_v2", "evidence_v2", "executable", "file_sha512", "ks_form_why", "proved_units", "reads_registered",
           "v2_executable", "verdicts_v2", "verify"]
