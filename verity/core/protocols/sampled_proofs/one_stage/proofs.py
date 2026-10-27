"""A statement's proof files, verified by C-Flock's Lean verifier of record (`flock-verify verify --zk`, the Lake package
`verity/core/service/flock`, built with `lake` on first use): the proof service's own run of the verifier, so a protocol
reads each session's verdict from that run, never from a verdict record the prover hands in.

`verify` takes a statement's files as the prover delivered them (`Proof`: the circuit file, the public file and the
sessions' directories, each its `session.json` and its rep files) and the verifier's own inputs: the SHA-512 its auditor
committed for the circuit file, the program's descriptor, the partition object and, for a statement that reads
registered values, the `--registered` file. It refuses
a circuit file that doesn't hash to the committed digest before anything runs, so each verdict is over the circuit the
auditor's own lowering built. The circuit file's tie to the program's Definitions rests on that lowering: the verifier
holds the file to the program only as far as its META names it, until it derives the circuit from the descriptor
(PRF-133). Then one `flock-verify verify --zk` over every session (`--registered` when the statement reads registered
values), and one `Finding` a session, in order:
whether it was accepted and why not, the program its verdict names (`program_sha512`, the digest of the canonical
descriptor without annotations, which an accepted verdict must give as the partition object's `program`),
its registered finding (each port's value and opened root) and the units it proved, the public file's global unit
indices of the instances its session drew (`unit_draw`), read only off a session the verifier accepted.

The verifier reads the whole circuit file once a call, in time and memory linear in it (about six times the file at
peak), then O(proof) a session: so a protocol verifies one statement a stratum, every drawn unit of it an instance.

The one-stage audit's v2 (`V2`, `SampledProofs.OneStageZk`) runs this verifier inside the audit's own Lean, on the files
as content-addressed blobs: `check_v2` hands it v1's inputs with the verdicts replaced by the files' digests
(`evidence_v2`) and the files themselves, and `verdicts_v2` is that run's verdicts alone, which the difftest holds to
`verify`'s under ``challenger`` (session ``i``'s coins the window's withheld challenger's at slot ``i``).

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
#: the statement every drawn unit's session proves
STATEMENT = "verity/flock-circuit"
_DIGEST = re.compile(r"[0-9a-f]{128}")
#: an input bound at registration (`program.REGISTERED_ID`, Lean's `Program.inputId?`), a descriptor's `definitions` key
_REGISTERED_INPUT = re.compile(r"RegisteredInput[1-9][0-9]*_v1")
_CHUNK = 1 << 24


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
    registered finding (port -> {value, root}) and the global units it proved (none unless accepted)."""

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


def reads_registered(program: bytes) -> bool:
    """Whether the program descriptor `program` has an input bound at registration (a `RegisteredInput<w>_v1` among its
    `definitions`), or has no `definitions` object to tell: True unless it names none."""
    try:
        defs = json.loads(program)["definitions"]
    except (ValueError, KeyError, TypeError):
        return True
    return not isinstance(defs, dict) or any(_REGISTERED_INPUT.fullmatch(k) for k in defs)


def proved_units(public: Path, session: Path) -> tuple[int, ...]:
    """The global units a session proved: the public file's unit indices (``units.indices``) of the instances its record
    drew (``unit_draw.units``), ascending. Read only off a session the verifier accepted."""
    indices = _header(public)["units"]["indices"]
    drawn = json.loads((session / "session.json").read_text())["unit_draw"]["units"]
    return tuple(sorted(indices[i] for i in drawn))


def verify(proof: Proof, *, circuit_sha512: str, program: bytes, partition: Mapping[str, Any],
           registered: Mapping[str, Any] | None = None,
           challenger: bool = False) -> tuple[list[Finding] | None, tuple[str, str] | None]:
    """Each of `proof`'s sessions verified by the verifier of record against the verifier's own `program` (its
    descriptor's bytes), `partition` and `registered` file (None only for a program that reads no registered value, such
    as N2's, whose unit opens the committed value inside its circuit), on the circuit file the auditor committed as
    `circuit_sha512`; the findings in session order and None, or None and the refusal (``registered``: `registered` is
    None and the program reads registered values, `reads_registered`; ``circuit``: the circuit file is not the
    committed one; ``proof``: a file is missing or the verifier's answer is not one verdict a session naming the
    partition object's program). A session the verifier rejects is a finding, not a refusal.

    The ``registered`` refusal is the verifier's own: without ``--registered``, Lean refuses a statement's registered
    reads only where its public file, which the prover writes, names them (`Registered.check`), and doesn't read a
    `RegisteredInput` as demanding them, so a prover that commits such a port's rows afresh and names none would be
    checked against no registered root. A run input that only the caller's registered file reads registered (the
    served RoPE statement's ``cs``) is the caller's to state: without the file, Lean holds it a run input, and refuses a
    session whose public file names its reads or whose tables share its registered rows.

    Under `challenger` session ``i``'s coins are the window's withheld challenger's at slot ``i`` (v2's), one run of
    the verifier a session (``--challenger-slot i``)."""
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
        base = [str(executable()), "verify", "--zk", "--statement", STATEMENT, "--circuit", str(proof.circuit),
                "--public", str(proof.public), "--partition", str(files["partition"]), "--program", str(files["program"]),
                *reads]
        runs = ([[*base, "--challenger-slot", str(i), "--session", str(s)] for i, s in enumerate(proof.sessions)]
                if challenger else [[*base, "--session", *map(str, proof.sessions)]])
        lines, tail = [], ""
        for cmd in runs:
            p = subprocess.run(cmd, capture_output=True, text=True)
            lines += [json.loads(x[len("VERDICT "):]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
            tail = f"(exit {p.returncode}): {(p.stderr or p.stdout)[-400:]}"
    if [v.get("session") for v in lines] != [str(s) for s in proof.sessions]:
        return None, ("proof", f"the verifier answered no verdict for each session in order {tail}")
    want = partition.get("program")
    out = []
    for s, v in zip(proof.sessions, lines):
        ok = v.get("accepted") is True
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


def evidence_v2(proof: Proof, names: Sequence[str] | None = None,
                coins: Sequence[Sequence[Mapping[str, Any]]] | None = None) -> tuple[dict, list[Path]]:
    """The v2 evidence's ``proofs`` for `proof`'s files, and the blobs that hold them: the public file's SHA-512 and, per
    session (named `names`, the v1 evidence's ``sessions``; by default its directory's name), its ``session.json``'s and
    each rep file's (``*.bin``), and with `coins` its ``coins``: the challenger's segments of its uses, in
    `session_uses` order, one list a session. The blobs are the circuit file, which the parameters name by the auditor's
    digest, the public file and each session's files."""
    names = [s.name for s in proof.sessions] if names is None else list(names)
    if len(names) != len(proof.sessions):
        raise ValueError(f"{len(names)} names for {len(proof.sessions)} sessions")
    if coins is not None and len(coins) != len(proof.sessions):
        raise ValueError(f"{len(coins)} sessions' coins for {len(proof.sessions)} sessions")
    sessions, blobs = [], [proof.circuit, proof.public]
    for i, (name, s) in enumerate(zip(names, proof.sessions)):
        reps = sorted(s.glob("*.bin"))
        sessions.append({"name": name, "record": file_sha512(s / "session.json"),
                         "reps": {r.name: file_sha512(r) for r in reps},
                         **({} if coins is None else {"coins": [dict(c) for c in coins[i]]})})
        blobs += [s / "session.json", *reps]
    return {"public": file_sha512(proof.public), "sessions": sessions}, blobs


def verdicts_v2(proof: Proof, *, circuit_sha512: str, program: Mapping[str, Any], partition: Mapping[str, Any],
                registered: Mapping[str, Any] | None = None) -> tuple[list[Finding] | None, str | None]:
    """`verify` under ``challenger`` (session ``i`` at slot ``i``), by the v2 audit's own run of the verifier on `proof`'s
    files as blobs
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


def _hexes(x: Any, what: str) -> list[str]:
    if type(x) is not list or any(type(s) is not str for s in x):
        raise ValueError(f"{what} is not a list of hex strings")
    return list(x)


def _unhex(h: str) -> bytes:
    """Lean `Flock.unhex`, on an even-length string."""
    if re.fullmatch(r"(?:[0-9a-f]{2})*", h) is None:
        raise ValueError("not lowercase hex")
    return bytes.fromhex(h)


def session_uses(slot: int, record: str) -> list[tuple[str, bytes, int, list[str]]]:
    """Lean `OneStageZk.sessionUses`: session `slot`'s uses of the challenger's coins as `(label, after, coins, want)`,
    `want` the record's F128 hex its coins' bytes must begin with: its link points, of family `slot/<slot>/session`
    after the SHA-512 of its tables' `link.root_b_sha512` digests in key order, then each stream's rounds in the
    record's order, of family `slot/<slot>/<stream>` after the stream's chain at that round, a_k = SHA-512(a_{k-1} ‖
    msg_sha512_k) from a_{-1} empty (two rounds of equal bytes are two uses); ceil(n/4) coins of 64 bytes for n F128s.
    A prover asks the challenger for each use's coins after its `after`, once (`coins.round_coins`). Raises
    ValueError with Lean's reason."""
    j = json.loads(record)
    link = j.get("link") if isinstance(j, dict) else None
    pts_j = link.get("points") if isinstance(link, dict) else None
    if type(pts_j) is not list:
        raise ValueError("record: link.points is not a list")
    pts = [h for p in pts_j for h in _hexes(p, "a link point")]
    rbs = link.get("root_b_sha512")
    if not isinstance(rbs, dict):
        raise ValueError("record: link.root_b_sha512 is not an object")
    ds = b""
    for t in sorted(rbs):
        d = rbs[t]
        if type(d) is not str or len(d) != 128:
            raise ValueError(f"record: link.root_b_sha512.{t} is not 64 bytes of hex")
        ds += _unhex(d)
    out = [(f"slot/{slot}/session", hashlib.sha512(ds).digest(), (len(pts) + 3) // 4, pts)]
    streams = j.get("streams")
    if type(streams) is not list:
        raise ValueError("record: streams is not a list")
    for s in streams:
        name = s.get("stream") if isinstance(s, dict) else None
        if type(name) is not str:
            raise ValueError("record: a stream's name is not a string")
        if name == "session":
            raise ValueError("record: a stream is named session, the link points' family")
        rounds = s.get("rounds")
        if type(rounds) is not list:
            raise ValueError(f"record: {name}'s rounds are not a list")
        chain = b""
        for r in rounds:
            sha = r.get("msg_sha512") if isinstance(r, dict) else None
            if type(sha) is not str:
                raise ValueError(f"record: a round of {name} has no msg_sha512")
            if len(sha) != 128:
                raise ValueError(f"record: a round of {name}'s msg_sha512 is not 64 bytes of hex")
            cs = _hexes(r.get("coins"), f"the coins of a round of {name}")
            chain = hashlib.sha512(chain + _unhex(sha)).digest()
            out.append((f"slot/{slot}/{name}", chain, (len(cs) + 3) // 4, cs))
    return out


def uses_v2(slot: int, record: Path) -> dict:
    """`session_uses` by the v2 audit's Lean (`sampled-proofs-one-stage-zk uses`): its line, `{"uses": [...]}` or
    `{"refused": why}`."""
    p = subprocess.run([str(v2_executable()), "uses", str(slot), str(record)], capture_output=True, text=True)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise LeanError(f"{V2_EXE} uses answered no line (exit {p.returncode}): {p.stderr[-400:]}") from None


def check_v2(params: Mapping[str, Any], record: Mapping[str, Any], evidence: Mapping[str, Any], proof: Proof, *,
             circuit_sha512: str, registered: Mapping[str, Any] | None,
             coins: Sequence[Sequence[Mapping[str, Any]]] | None = None) -> dict:
    """The one-stage audit on `proof` itself (`sampled-proofs-one-stage-zk check`): v1's `params` with ``proofs``, the
    auditor's circuit digest and the verifier's registered file, the registration `record`, and v1's `evidence` with
    its ``verdicts`` replaced by `proof`'s ``proofs`` (`evidence_v2`, under the evidence's ``sessions``, each with its
    `coins`); the outcome line."""
    pr, blobs = evidence_v2(proof, evidence["sessions"], coins)
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


__all__ = ["EXE", "PACKAGE", "STATEMENT", "V2", "V2_EXE", "Finding", "LeanError", "Proof", "check_v2", "evidence_v2",
           "executable", "file_sha512", "proved_units", "reads_registered", "session_uses", "uses_v2", "v2_executable",
           "verdicts_v2", "verify"]
