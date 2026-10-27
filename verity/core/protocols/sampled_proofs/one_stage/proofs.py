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

The verifier reads the whole circuit file once a call, in time and memory linear in it (about six times the file at
peak), then O(proof) a session: so a protocol verifies one statement a stratum, every drawn proof block of it an instance.

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
#: the statement every drawn proof block's session proves
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
           registered: Mapping[str, Any] | None = None,
           challenger: bool = False) -> tuple[list[Finding] | None, tuple[str, str] | None]:
    """Each of `proof`'s sessions verified by the verifier of record against the verifier's own `program` (its
    descriptor's bytes), `partition` and `registered` file (None only for a program that reads no registered value, such
    as N2's, whose proof block opens the committed value inside its circuit), on the circuit file the auditor committed as
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


def round_request(stream: str, chain: bytes, n: int) -> bytes:
    """Lean `OneStageZk.roundRequest`: round k's request to the window's challenger, all of it public: its stream's name
    (two bytes big-endian of its UTF-8 length, then the name), the stream's chain at the round (64 bytes), then the
    round's coin count `n`, its F128s as the record's `coins` lists them (eight bytes little-endian). Its coins are
    released after its SHA-512."""
    s = stream.encode()
    if len(s) >= 1 << 16:
        raise ValueError("record: a stream's name is past 65535 bytes")
    return len(s).to_bytes(2, "big") + s + chain + n.to_bytes(8, "little")


def parse_round_request(request: bytes) -> tuple[str, bytes, int] | None:
    """`round_request`'s `(stream, chain, n)` read back from its bytes; None when they are not one."""
    if len(request) < 2:
        return None
    k = int.from_bytes(request[:2], "big")
    if len(request) != 2 + k + 64 + 8:
        return None
    stream = request[2:2 + k].decode(errors="replace")
    if stream.encode() != request[2:2 + k]:
        return None
    return stream, request[2 + k:2 + k + 64], int.from_bytes(request[-8:], "little")


def session_uses(slot: int, record: str) -> list[tuple[str, bytes, int, list[str]]]:
    """Lean `OneStageZk.sessionUses`: session `slot`'s uses of the challenger's coins as `(label, after, coins, want)`,
    `want` the record's F128 hex its coins' bytes must begin with: its link points, of family `slot/<slot>/session`
    after the SHA-512 of its tables' `link.root_b_sha512` digests in key order, then each stream's rounds in the
    record's order, of family `slot/<slot>/<stream>` after the SHA-512 of the round's request (`round_request`) at the
    stream's chain a_k = SHA-512(a_{k-1} ‖ msg_sha512_k) from a_{-1} empty and the round's own coin count, the record's
    F128s n (two rounds of equal bytes are two uses, and a release after a request of another count is no use);
    ceil(n/4) coins of 64 bytes for n F128s.
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
        if len(name.encode()) >= 1 << 16:
            raise ValueError("record: a stream's name is past 65535 bytes")
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
            out.append((f"slot/{slot}/{name}", hashlib.sha512(round_request(name, chain, len(cs))).digest(),
                        (len(cs) + 3) // 4, cs))
    return out


def uses_v2(slot: int, record: Path) -> dict:
    """`session_uses` by the v2 audit's Lean (`sampled-proofs-one-stage-zk uses`): its line, `{"uses": [...]}` or
    `{"refused": why}`."""
    p = subprocess.run([str(v2_executable()), "uses", str(slot), str(record)], capture_output=True, text=True)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise LeanError(f"{V2_EXE} uses answered no line (exit {p.returncode}): {p.stderr[-400:]}") from None


#: the key of the registration's ``window`` that names the public file the window's statement reads (Lean
#: `OneStageZk.PUBLIC_KEY`)
PUBLIC_KEY = "public_file"


def registered_public(record: Mapping[str, Any]) -> str | None:
    """Lean `OneStageZk.registeredPublic`: the public file the registration `record` names, ``window.public_file``, when
    it names one as a SHA-512 digest (lowercase hex of 64 bytes)."""
    w = record.get("window")
    d = w.get(PUBLIC_KEY) if isinstance(w, Mapping) else None
    return d if type(d) is str and re.fullmatch(r"[0-9a-f]{128}", d) else None


def public_why(record: Mapping[str, Any], pub: str) -> str | None:
    """Lean `OneStageZk.publicWhy`: why the evidence's public file of SHA-512 `pub` is not the one the registration
    `record` names; None when it is. v2 refuses the evidence then."""
    d = registered_public(record)
    if d is None:
        return f"the registration names no public file (window.{PUBLIC_KEY}, a SHA-512 digest)"
    return None if pub == d else f"the evidence's public file is not the one the registration names, {d}"


def with_public(record: Mapping[str, Any], public: Path) -> dict:
    """`record` with its window naming the public file `public` (``window.public_file``, its SHA-512), as a prover
    registers its window's statement for v2 before any coin."""
    return {**record, "window": {**record["window"], PUBLIC_KEY: file_sha512(public)}}


def public_v2(record: Path, pub: str) -> dict:
    """`public_why` by the v2 audit's Lean (`sampled-proofs-one-stage-zk public`): its line, ``{"why": why | null}`` or
    ``{"refused": why}``."""
    p = subprocess.run([str(v2_executable()), "public", str(record), pub], capture_output=True, text=True)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise LeanError(f"{V2_EXE} public answered no line (exit {p.returncode}): {p.stderr[-400:]}") from None


#: the key of the registration's ``window`` that names how many 64-byte coins its draw opens (Lean
#: `OneStageZk.DRAW_KEY`)
DRAW_KEY = "draw_coins"
#: the most rounds a session's record may hold, over both its rep streams (Lean `OneStageZk.MAX_ROUNDS`)
MAX_ROUNDS = 4096


def registered_draw(record: Mapping[str, Any]) -> int | None:
    """Lean `OneStageZk.registeredDraw`: how many coins the registration `record` says its draw opens,
    ``window.draw_coins``, when it names a positive count."""
    w = record.get("window")
    m = w.get(DRAW_KEY) if isinstance(w, Mapping) else None
    return m if type(m) is int and m > 0 else None


def draw_why(record: Mapping[str, Any], n: int) -> str | None:
    """Lean `OneStageZk.drawCountWhy`: why a draw that opens `n` coins is not the one the registration `record` names;
    None when it is. v2 refuses the evidence then."""
    m = registered_draw(record)
    if m is None:
        return f"the registration names no draw size (window.{DRAW_KEY}, a positive count of 64-byte coins)"
    return None if n == m else f"the draw opens {n} coins, and the registration names {m}"


def with_draw(record: Mapping[str, Any], coins: int) -> dict:
    """`record` with its window naming the `coins` its draw opens (``window.draw_coins``), as a prover registers its
    window's draw for v2 before any coin: the coins a draw of the registered law reads when it rejects no word."""
    return {**record, "window": {**record["window"], DRAW_KEY: coins}}


def rounds_why(uses: Sequence[tuple[str, bytes, int, list[str]]]) -> str | None:
    """Lean `OneStageZk.roundsCapWhy`: why a session's uses (`session_uses`) are past what v2 reads, more than
    `MAX_ROUNDS` rounds after its link points or a use of 2^64 F128s or more; None when they are not."""
    if MAX_ROUNDS + 1 < len(uses):
        return f"{len(uses) - 1} rounds, past the {MAX_ROUNDS} a session may hold"
    big = next(((label, want) for label, _, _, want in uses if len(want) >= 1 << 64), None)
    return None if big is None else f"{big[0]} reads {len(big[1])} F128s, past 2^64 - 1"


def sessions_why(count: int) -> str | None:
    """Lean `OneStageZk.sessionsWhy` (one-session): why `count` sessions are not the one v2 reads; None when there is
    exactly one."""
    return None if count == 1 else f"the evidence names {count} sessions, and v2 reads exactly one"


def _nat(x: Any) -> int | None:
    return x if type(x) is int and x >= 0 else None


def verifier_subset(law: Any, record: Any) -> tuple[int, int] | None:
    """Lean `OneStageZk.subsetAt`: the registration `record`'s ``population`` `n` and the `k` of the subset law the
    verifier's `law` names over it; None when the law over `n` is no subset law."""
    n = _nat(record.get("population")) if isinstance(record, Mapping) else None
    if n is None or not isinstance(law, Mapping) or law.get("law") != "subset" or sorted(law) != ["k", "law"]:
        return None
    k = _nat(law["k"])
    return None if k is None or n < k else (n, k)


def drawn_units(draw: Any) -> list[int]:
    """Lean `OneStageZk.drawnUnits`: the units a draw object names, as v1 reads them."""
    us = draw.get("units") if isinstance(draw, Mapping) else None
    return [u for u in us if _nat(u) is not None] if isinstance(us, list) else []


def draw_file(k: int, n: int, units: Sequence[int]) -> dict:
    """Lean `OneStageZk.drawFile`: the verifier's own draw file, the subset draw of `k` over ``[0, n)`` of `units`."""
    return {"law": "subset", "population": n, "k": k, "units": list(units)}


def record_draw(record: str) -> Any:
    """Lean `OneStageZk.rawDraw`: the session record's ``unit_draw``, when it is a draw (not absent, not null)."""
    try:
        j = json.loads(record)
    except ValueError:
        return None
    return j.get("unit_draw") if isinstance(j, dict) else None


def _ascending_below(us: Sequence[Any], n: int) -> bool:
    return all(_nat(u) is not None and u < n for u in us) and all(a < b for a, b in zip(us, us[1:]))


def _subset_draw(d: Any) -> tuple[int, int, bool]:
    """`Flock.Draw.UnitDraw.ofJson` on a subset draw `d`: its `k`, population and whether it has a closure; a
    `ValueError` for a draw it refuses, and a law other than ``subset`` taken as a draw of another law (its `k` -1)."""
    if not isinstance(d, dict):
        raise ValueError("U2: unit_draw is not an object")
    n, us = _nat(d.get("population")), d.get("units")
    if n is None or not isinstance(us, list) or any(_nat(u) is None for u in us):
        raise ValueError("U2: unit_draw's population and units")
    if d.get("law") != "subset":
        return -1, n, "closure" in d
    if len(d) != 4 or _nat(d.get("k")) is None:
        raise ValueError("U2: unit_draw's keys are not law, population, k, units")
    if not _ascending_below(us, n):
        raise ValueError("U2: the drawn units are not ascending, distinct units of the population")
    if d["k"] > n or len(us) != d["k"]:
        raise ValueError("U2: a subset draw of the wrong size")
    return d["k"], n, False


def draw_law_why(n: int, k: int, units: Sequence[int], record: str) -> str | None:
    """Lean `OneStageZk.drawLawAt` (draw-law): why the session record's draw is not a subset draw of the verifier's own
    `k` over the registered population `n`, with the evidence's draw's `units` ascending and below `n`, `k` of them;
    None when it is."""
    if not (_ascending_below(list(units), n) and len(units) == k):
        return f"the draw's units are not {k} ascending units below the registered population {n}"
    d = record_draw(record)
    if d is None:
        return "the record names no draw"
    try:
        got = _subset_draw(d)
    except ValueError as why:
        return f"the record's draw does not decode: {why}"
    return None if got == (k, n, False) else (
        f"the record's draw is not a subset draw of {k} over the registered population {n}")


def draw_file_why(n: int, k: int, units: Sequence[int], record: str) -> str | None:
    """Lean `OneStageZk.drawFileAt` (draw-file): why the session record's draw file is not the verifier's own,
    `draw_file(k, n, units)`, compared byte for byte in canonical JSON; None when it is."""
    d = record_draw(record)
    if d is None:
        return "the record names no draw"
    try:
        same = _canonical(d) == _canonical(draw_file(k, n, units))
    except ValueError:
        same = False
    return None if same else "the record's draw file is not the canonical encoding of the verifier's own draw"


def proved_why(proved: Sequence[int], record: str) -> str | None:
    """Lean `OneStageZk.provedAt` (proved-units): why the evidence's `proved` units are not the units the session
    record's draw names; None when they are."""
    d = record_draw(record)
    if d is None:
        return "the record names no draw"
    return None if drawn_units(d) == list(proved) else (
        "the evidence's proved_units are not the units the record's draw names")


def record_why(n: int, k: int, units: Sequence[int], proved: Sequence[int], record: str) -> dict:
    """The draw-law, draw-file and proved-units reasons on one session record, as `sampled-proofs-one-stage-zk record`
    writes them."""
    return {"file": draw_file_why(n, k, units, record), "law": draw_law_why(n, k, units, record),
            "proved": proved_why(proved, record)}


def units_identity_why(public: bytes) -> str | None:
    """Lean `OneStageZk.unitsIdentityWhy` (units-identity): why the public file's header does not name the identity as
    its ``units.indices``; None when it does."""
    try:
        line = public.split(b"\n", 1)[0].decode()
    except UnicodeDecodeError:
        return "the public file's header is not UTF-8"
    try:
        h = json.loads(line)
    except ValueError:
        return "the public file's header is not JSON"
    units = h.get("units") if isinstance(h, dict) else None
    indices = units.get("indices") if isinstance(units, dict) else None
    if not isinstance(indices, list):
        return "the public file's header names no units.indices"
    return None if all(_nat(x) == i for i, x in enumerate(indices)) else "the public file's units.indices is not the identity"


def _v2_line(command: str, *args: str) -> dict:
    p = subprocess.run([str(v2_executable()), command, *args], capture_output=True, text=True)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise LeanError(f"{V2_EXE} {command} answered no line (exit {p.returncode}): {p.stderr[-400:]}") from None


def draw_v2(record: Path, n: int) -> dict:
    """`draw_why` by the v2 audit's Lean (`sampled-proofs-one-stage-zk draw`): its line, ``{"why": why | null}`` or
    ``{"refused": why}``."""
    return _v2_line("draw", str(record), str(n))


def rounds_v2(slot: int, record: Path) -> dict:
    """`rounds_why` of `session_uses` by the v2 audit's Lean (`sampled-proofs-one-stage-zk rounds`): its line,
    ``{"why": why | null}`` or ``{"refused": why}``."""
    return _v2_line("rounds", str(slot), str(record))


def sessions_v2(count: int) -> dict:
    """`sessions_why` by the v2 audit's Lean (`sampled-proofs-one-stage-zk sessions`): its line, ``{"why": why | null}``."""
    return _v2_line("sessions", str(count))


def subset_v2(law: Path, record: Path) -> dict:
    """`verifier_subset` by the v2 audit's Lean (`sampled-proofs-one-stage-zk subset`) on the law object and the
    registration record files: its line, ``{"subset": [n, k] | null}`` or ``{"refused": why}``."""
    return _v2_line("subset", str(law), str(record))


def record_v2(n: int, k: int, units: Sequence[int], proved: Sequence[int], record: Path) -> dict:
    """`record_why` by the v2 audit's Lean (`sampled-proofs-one-stage-zk record`) on the session record file: its line,
    ``{"file": why, "law": why, "proved": why}`` or ``{"refused": why}``."""
    return _v2_line("record", str(n), str(k), json.dumps(list(units)), json.dumps(list(proved)), str(record))


def identity_v2(public: Path) -> dict:
    """`units_identity_why` by the v2 audit's Lean (`sampled-proofs-one-stage-zk identity`) on the public file: its
    line, ``{"why": why | null}``."""
    return _v2_line("identity", str(public))


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


__all__ = ["DRAW_KEY", "EXE", "MAX_ROUNDS", "PACKAGE", "PUBLIC_KEY", "STATEMENT", "V2", "V2_EXE", "Finding", "LeanError",
           "Proof", "check_v2", "draw_file", "draw_file_why", "draw_law_why", "draw_v2", "draw_why", "drawn_units",
           "evidence_v2", "executable", "file_sha512", "identity_v2", "parse_round_request", "proved_units", "proved_why",
           "public_v2", "public_why", "reads_registered", "record_draw", "record_v2", "record_why", "registered_draw",
           "registered_public", "round_request", "rounds_v2", "rounds_why", "session_uses", "sessions_v2", "sessions_why",
           "subset_v2", "units_identity_why", "uses_v2", "v2_executable", "verdicts_v2", "verifier_subset", "verify",
           "with_draw", "with_public"]
