"""One-stage v2's per-member form outside Lean: the check's run on a case, and the Python of the sessions' run.

A window of several members is audited in v2's per-member form of ``proofs`` (the parameters' ``members``, the
evidence's ``statements``; `proofs.members_v2` writes it). `Spot.OneStageZk` reads a case in either form and decides
a per-member one by `checkMembersAt`: it accepts where the form's run returns with every session accepted and v1's
check at the coins accepts on the sessions' verdicts. `check` runs the executable's ``check`` on such a case and
returns its verdict and reason.

`members_run` is the Python of `OneStageZk.membersRun`, the part of that check that reads the per-member form: the
statements' structure, the sessions' coins held to the trees the window's challenger session logged, each closed after
its session's evidence before the next is opened, the members laid over the docket's population, each registered
value one tree, the docket's roots each one member's, each statement's member one the sample holds a unit of
(`_sampled`, Lean's `sampledHeld`), then each statement's session verified on its member's own files. It refuses with
Lean's reason where Lean refuses, and is no verdict itself: its `Run` is the sessions' findings, which the audit above
it (v1's check at the coins) reads. Each session's verdict is the Flock verifier's own (`flock_verdict`: its
executable's ``verify --zk``, as `proofs.verify` runs it).

Before any session it holds every member's files, sampled or not, in the members' order (`OneStageZk.membersFiles`):
the reads Lean's `Member.statement` makes of them, on the member's own lookup tables (the blobs' pinned tables its
circuit looks up, `proofs.own_tables`, so the members of a window may look up different tables), then what a session's
verdict checks of the files alone (`files_why`, Lean's `StepAt.filesHeld`), a refusal naming the member. A member the
sample missed is refused there as one it holds is. The form the sampled audit's knowledge soundness covers is the Flock
verifier's own there too, and so is what a session's setup takes of the public file's header and rows
(`flock_form`: its executable's ``ks-form``).

Where it is not Lean's step for step, each on a case neither side accepts:

* **A link's ``root_B`` length** (`challenger.LINK_ROOT_LEN`). Lean holds each session's link to the length its
  statement fixes for its table's ``root_B`` (`OneStageZk.ownRootBTables`, the session set up on the member's files)
  before it verifies any session. This sets up no session, so it passes no length (`TABLES`): a session Lean refuses
  there for that length alone is here a session its own verdict rejects, since the verifier holds ``root_B`` to that
  length.
* **A circuit's ``vus_per_block``** (`OneStageZk.Member.blockHeld`). Lean refuses a circuit file that parses with
  another ``vus_per_block`` than the parameters', and one that does not parse for that, where it holds the member's
  files. This reads the file's META alone, so it refuses both for the ``vus_per_block``.
* **The scope's rules on a circuit that parses, and the ports' roots** (`Flock.ProvedScope.circuitRules`,
  `Flock.PortRoots.checkOf`, both in `StepAt.filesHeld`). Lean holds every member's files to them before any session.
  The Flock verifier's executable runs neither without a session, so `files_why` answers the rest of that check in
  Lean's order and passes these two: files Lean refuses for one of them are refused here by a later check of the files
  where one applies, else left to the member's session, whose verdict rejects, and unanswered for a member with no
  statement.
* **A session's coin-tree key, fresh among the earlier sessions'** (`Flock.CoinTree.checkFresh`): implied here by the
  trees' check, which holds each session to a logged tree of its own, no key twice in the log.
* **Lengths past a statement's 8-byte word** (`OneStageZk.fitsWhy`, `EVIDENCE_FITS`), which no file
  reaches.
* **Inputs the per-member parsers refuse** raise `Outside`: this reads parameters and evidence as
  `membersParamsOfJson` and `membersEvidenceOf` parsed them.
* **C's wiring** (`OneStageZk.derivedWiring`): a wiring Lean refuses raises `Outside` here, whose refusal codes are
  `edges.edges`'.
* **A registered file's own form** (`Flock.Registered.ofJson`, which `classHeld` and each statement read it through).
  This reads a member's registered file by the fields `classHeld` compares, `Outside` where one is not a registered
  port's; the file's other refusals are the Flock verifier's, at the member's session, and are not answered here for a
  member the sample missed.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.circuits import codec, edges as EG, partition_object as PO
from verity.circuits.program import RUN, Program, input_id
from verity.protocols.proof.spot.verifier.challenger import challenger as Ch
from verity.protocols.proof.spot.verifier.challenger.coins import DRAW_FORM

from . import closing as CL, docket as DK, one_circuit as OC, partition as P, proofs as PF, registered as G
from verity.commitments import registered_value as RV
from verity.commitments.domains import REGISTERED_READS_V2
from .proofs import V2_EXE, LeanError, v2_executable
from . import lean_child as LC

#: the exit of the v2 executable's ``check`` by its verdict (`Core.BlobProtocol.main`)
EXITS = {"accept": 0, "reject": 1, "undecided": 2}


def check(params: Mapping[str, Any], commitment: bytes, evidence: bytes, blobs: Sequence[Path] = ()) -> tuple[str, str | None]:
    """`check_out`'s verdict and reason."""
    return check_out(params, commitment, evidence, blobs)[:2]


def check_out(params: Mapping[str, Any], commitment: bytes, evidence: bytes,
              blobs: Sequence[Path] = ()) -> tuple[str, str | None, bytes]:
    """v2's check of the case of parameters `params`, docket `commitment` and evidence `evidence` (each
    the bytes the audit reads) on the files `blobs` (`V2_EXE`'s ``check``, `OneStageZk.protocol`): the verdict
    (``accept``, ``reject`` or ``undecided``) and the reason, None on an accept. A case in either form is checked in
    its own. Raises `LeanError` on an answer that is not an outcome's line, or whose exit is not its verdict's."""
    with tempfile.TemporaryDirectory(prefix="verity-members-check-") as d:
        paths = [Path(d, n) for n in ("params.json", "commitment.bin", "evidence.bin")]
        canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
        for path, b in zip(paths, (canonical, commitment, evidence)):
            path.write_bytes(b)
        p = LC.run([str(v2_executable()), "check", *map(str, paths), *map(str, blobs)], ok_statuses=LC.CHECK_EXITS)
    try:
        line = p.stdout.strip().splitlines()[-1]
        got = json.loads(line)
    except (IndexError, ValueError):
        raise LeanError(f"{V2_EXE} check answered no line (exit {p.returncode}): {p.stderr[-400:]}") from None
    if (not isinstance(got, dict) or got.get("verdict") not in EXITS or p.returncode != EXITS[got["verdict"]]
            or not (got.get("reason") is None or type(got["reason"]) is str)
            or (got["verdict"] == "accept") != (got.get("reason") is None)):
        raise LeanError(f"{V2_EXE} check answered a line that is not an outcome's (exit {p.returncode}): {got!r}"[:600])
    return got["verdict"], got["reason"], line.encode()


def case_check(case: Mapping[str, Any]) -> tuple[str, str | None]:
    """`check` on a case as a driver writes one: ``{"blobs": [path, ...], "commitment": hex, "evidence": hex,
    "params": {...}}``."""
    return case_check_out(case)[:2]


def case_check_out(case: Mapping[str, Any]) -> tuple[str, str | None, bytes]:
    """`check_out` on a case as `case_check` reads one: the verdict, the reason and the outcome's line as printed."""
    return check_out(case["params"], bytes.fromhex(case["commitment"]), bytes.fromhex(case["evidence"]),
                     [Path(b) for b in case["blobs"]])


#: `OneStageZk.sessionTables`, the tables of a member's session, each with no ``root_B`` length (module docstring)
TABLES: dict[str, int | None] = {"circuit": None}
#: `OneStageZk.NO_CHALLENGER`
NO_CHALLENGER = ("the parameters name no challenger: the draw's coins are the auditor's challenger's signed release "
                 "(challenger, with window_root), never the prover's record")
PARTITION_QUERY = DK.PARTITION_QUERY
#: `OneStageZk.POPULATION_ORDER`
POPULATION_ORDER = ("population-order: the program's proof blocks under the query are not the registration's "
                    "population units in order")
#: `OneStageZk.MEMBERS_LAID`
MEMBERS_LAID = ("members: the members' units are not the registration's population, each member's one range of it: a "
                "member is the units [base, base + n), and the members' counts sum to the population")
#: `OneStageZk.TABLE_PORT`: the port a docket names a window's row tables' root at
TABLE_PORT = "row-table"
#: `OneStageZk.ROW_TABLES`
ROW_TABLES = ("roots: the registration names a row tables' root (row-table), the window's and no member's: the "
              "per-member form is refused on a window with row tables until its guarantee covers a root every member "
              "reads")
#: `OneStageZk.MEMBER_STRATA`
MEMBER_STRATA = ("partition: the registration's strata, read stratum by stratum, are not its units in order, or a unit "
                 "of the member's is past them: a member's strata are the registration's, each restricted to the "
                 "member's units")
#: `OneStageZk.MEMBER_COVER`
MEMBER_COVER = ("partition: the registration's strata's ranges are not a disjoint cover of its units 0, 1, …, N - 1, or a "
                "unit of the member's is past them")


def strata_design(kind: str) -> str:
    """`FilePart.STRATA_DESIGN`: the refusal of strata that cover the units out of order under a design of kind `kind`."""
    return ("partition: the registration's strata are a disjoint cover of its units, not its units in order stratum by "
            f"stratum, and the sampling design is {kind}: only a subset design's draw reads no stratum")


def subset_design(law: Any) -> bool:
    """`OneStageZk.subsetDesign`'s verifier half: the resolved sampling design `law` (`resolved`) is a subset object, All
    among them."""
    return isinstance(law, dict) and law.get("law") == "subset"


def resolved(params: Mapping[str, Any], program: Program) -> Any:
    """The parameters' `law` as Lean's `OneStage.design` resolves it before any check reads it
    (`service.params_sampling_design`): a rate to its sampling design over the program's population, an object as given.
    Where it does not resolve, why, as text: no design, which `subset_design` and `_kind` read as no subset and unnamed,
    and which the parameters' check refuses before any member is read."""
    from verity.protocols.proof.spot.verifier.service import params_sampling_design
    try:
        return params_sampling_design(params.get("law"), program, params["query"], params.get("work_table"))[0]
    except ValueError as e:
        return f"the parameters' law does not resolve to a sampling design: {e}"


def subset_designs(law: Any, record: Any) -> bool:
    """`OneStageZk.subsetDesign`: the parameters' `law` is of the subset kind (`subset_design`) and so is the
    registration `record`'s."""
    rl = record.get("law") if isinstance(record, dict) else None
    return subset_design(law) and isinstance(rl, dict) and rl.get("law") == "subset"


def _kind(law: Any) -> str:
    """`OneStageZk.designKind` of a resolved sampling design `law`."""
    return law["law"] if isinstance(law, dict) and isinstance(law.get("law"), str) else "unnamed"


#: `OneStageZk.MEMBER_UNITS`
MEMBER_UNITS = ("units: the public file's units.indices are not exactly the member's units, in order: slot i of its "
                "table is the member's i-th unit")
def unwired_why(port: str) -> str:
    """`OneStageZk.UNWIRED_WHY`: the refusal of a member's public file naming ``port`` at a root its slice holds none
    of."""
    return (f"public file: port {port} has no tree under the member, and the record names no tree of its writer's "
            "output field: a wired port reads its writer's output field's tree")


#: `OneStageZk.UNITS_HANDED`
UNITS_HANDED = "units: a member's units are its block of the partition object's blocks, never the parameters' own"
PROVED_UNITS = "the units the sessions proved are not the evidence's proved_units"
#: the heads of the two refusals `Flock.KsScope.checkOf` and the scope (`Flock.ProvedScope.checkOf`) word alike, of
#: files neither reads: a circuit that does not parse, a public file that does not load
UNREAD = ("refused: a circuit that does not parse (", "refused: a public file that does not load (")
#: a statement's form held on its circuit and public files and the lookup tables, by name, with no session: None, or
#: the refusal (`flock_form`)
Form = Callable[[Path, Path, Sequence[tuple[str, Path]]], "str | None"]
_HEX = frozenset("0123456789abcdef")
#: `Flock.Registered.PORT_KEYS`, in its order, as its refusal prints them.
PORT_KEYS = ("value", "root", "leaves", "words", "word_bits", "domain", "row_schema", "known", "param")
#: the value a registered port reads a committed tree of the registration as, ``committed/<port>`` (`rows.own_value`)
COMMITTED = "committed/"
#: `OneStageZk.INPUT_MEMBER`
INPUT_MEMBER = "roots: a member is named input: a root input/n<w> is the window's tree of run input node w, no member's"


def own_root_why(member: str, port: str) -> str:
    """`OneStageZk.OWN_ROOT_WHY`: the refusal of a member's root of its own at a port C wires into it."""
    return (f"member {member}: port {port} is a row of in(S) (C wires it from another block's outputs), and the "
            "registration holds a root of its own for it: a crossing wire is committed once, in its writer's output "
            "field's tree")


def wired_tree_why(name: str, port: str) -> str:
    """`OneStageZk.WIRED_TREE_WHY`: the refusal of member ``name``'s registered port whose value is a field tree's
    name that is not its own output at its template's tree."""
    return f"registered: member {name}'s port {port}: a wired port's tree is not derived from C yet"


def _field_tree_name(v: str) -> bool:
    """`OneStageZk.fieldTreeName`: ``v`` is ``<template>/o<j>`` (`Flock.Registered.fieldTree`)."""
    parts = v.split("/")
    j = parts[-1][1:]
    return len(parts) >= 2 and parts[-1][:1] == "o" and j.isascii() and j.isdigit()


class Outside(ValueError):
    """An input `members_run` does not decide: one the per-member parsers refuse, a registered file or a verifier's
    answer it cannot read. Never a pass."""


class _Refused(Exception):
    """`membersRun`'s refusal, with Lean's reason."""


@dataclass(frozen=True)
class Statement:
    """What one member's session is verified on, each file a blob's: the member's circuit and public files, the
    verifier's program (its descriptor's bytes), partition object and registered file for the member, the member's own
    lookup tables, the pinned ones its circuit looks up, by name, and the session's record and proof files, by name."""

    circuit: Path
    public: Path
    program: bytes
    partition: Mapping[str, Any]
    registered: Mapping[str, Any] | None
    tables: tuple[tuple[str, Path], ...]
    record: Path
    reps: tuple[tuple[str, Path], ...]


@dataclass(frozen=True)
class Run:
    """`OneStageZk.Run` of the per-member form: each session's finding in the statements' order, the units the accepted
    ones proved, and whether every session accepted. No verdict of the audit."""

    findings: tuple[PF.Finding, ...]
    proved: tuple[int, ...]
    all: bool


def blobs_of(paths: Sequence[Path]) -> dict[str, Path]:
    """The files `paths` by SHA-512, the first of each digest, as `Core.Blobs.get` finds a blob."""
    out: dict[str, Path] = {}
    for f in paths:
        out.setdefault(PF.file_sha512(Path(f)), Path(f))
    return out


def _blob(blobs: Mapping[str, Path], digest: str, what: str) -> Path:
    """`Core.Blobs.get`: the blob whose SHA-512 is `digest`, or its refusal naming `what`."""
    if len(digest) % 2:
        raise _Refused("odd-length hex")
    if not set(digest) <= _HEX:
        raise _Refused("not lowercase hex")
    if len(digest) != 128:
        raise _Refused(f"{what}: {digest} is not a SHA-512 digest (128 lowercase hex)")
    if digest not in blobs:
        raise _Refused(f"{what}: no blob has the SHA-512 {digest}")
    return blobs[digest]


def _nat(x: Any) -> bool:
    return type(x) is int and x >= 0


def _same(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _first(m: Mapping[str, Any]) -> int:
    return min(lo for lo, _ in m["units"])


def _count(m: Mapping[str, Any]) -> int:
    return sum(hi - lo for lo, hi in m["units"])


def _block(name: str, m: Mapping[str, Any], descriptor: bytes, partition: Mapping[str, Any],
           decoded: OC.Decoded | None = None) -> list[tuple[int, int]]:
    """Member ``name``'s units, its block of the partition object's ``blocks`` (`one_circuit.block`). Lean's parse
    refuses the parameters of a member handed units of its own or naming a block the object lacks
    (`OneStageZk.membersOf`), so those raise `Outside` with Lean's ``params:`` reason."""
    if "units" in m:
        raise Outside(f"params: proofs: member {name}: {UNITS_HANDED}")
    try:
        return OC.block(descriptor, partition, m["block"], decoded)
    except OC.Refused as e:
        raise Outside(f"params: proofs: member {name}: partition: {e}") from None


def _apart(ms: Mapping[str, Any]) -> None:
    """`OneStageZk.membersOf`'s last check: no unit in two members' units, the first overlap of the ranges sorted by
    start refused as Lean's parse refuses the parameters."""
    rs = sorted((u for m in ms.values() for u in m["units"]), key=lambda u: u[0])
    for prev, r in zip(rs, rs[1:]):
        if r[0] < prev[1]:
            raise Outside(f"params: proofs: unit {r[0]} is in two members' units: a unit is one member's")


def _parsed(params: Mapping[str, Any], evidence: Mapping[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    """The parameters' members and the evidence's statements, as the per-member parsers leave them; `Outside` where
    they are not in that form."""
    form = Outside("the parameters and the evidence are not in the per-member form as its parsers read it "
                   "(`OneStageZk.membersParamsOfJson`, `membersEvidenceOf`)")
    try:
        ms, sts = params["proofs"]["members"], evidence["proofs"]["statements"]
        ok = (set(params["proofs"]) == {"members"} and set(evidence["proofs"]) == {"statements"} and isinstance(ms, dict)
              and bool(ms) and type(sts) is list and type(evidence["sessions"]) is list
              and type(evidence["proved_units"]) is list and all(_nat(u) for u in evidence["proved_units"])
              and isinstance(evidence["coins"], dict)
              and isinstance(params["challenger"], dict) and isinstance(params["partition"], dict)
              and isinstance(params["program"], dict))
        for m in ms.values() if ok else ():
            ok = ok and (set(m) - {"units"} == {"public", "registered", "block", "vus_per_block"}
                         and type(m["public"]) is str and (m["registered"] is None or isinstance(m["registered"], dict))
                         and _nat(m["block"]) and _nat(m["vus_per_block"]) and m["vus_per_block"] > 0)
        for st in sts if ok else ():
            ok = ok and set(st) == {"member", "sessions"} and type(st["sessions"]) is list
            for s in st["sessions"] if ok else ():
                ok = ok and (set(s) == {"name", "record", "reps"} and type(s["name"]) is str and type(s["record"]) is str
                             and isinstance(s["reps"], dict) and all(type(d) is str for d in s["reps"].values()))
    except (KeyError, TypeError, AttributeError):
        raise form from None
    if not ok:
        raise form
    return {name: ms[name] for name in sorted(ms)}, sts


def _held(ms: Mapping[str, Any], sts: Sequence[Any]) -> list[str]:
    """`OneStageZk.membersHeld`: the statements' members, statement by statement."""
    held: list[str] = []
    for st in sts:
        name = st["member"]
        if type(name) is not str:
            raise _Refused("a statement of the proofs names no member, and the parameters' proofs are per member")
        if name not in ms:
            raise _Refused(f"member {name}: the parameters name no such member")
        if name in held:
            raise _Refused(f"member {name}: a second statement: a member's units are one statement's")
        if held and _first(ms[name]) < _first(ms[held[-1]]):
            raise _Refused(f"member {name}: its statement follows a member's whose units start after its own: the "
                           "statements are in the members' order (by first unit, the registration's)")
        if len(st["sessions"]) != 1:
            raise _Refused(f"member {name}: its statement has {len(st['sessions'])} sessions: a statement is exactly "
                           "one session on live coins, never retried, and an undrawn member has no statement")
        held.append(name)
    return held


def _trees(params: Mapping[str, Any], evidence: Mapping[str, Any], verify: Ch.Verify) -> tuple[dict, list[str]]:
    """`OneStageZk.treesOf`: the coin trees the window's challenger session shows it sampled and opened, with its
    release log's lines."""
    window, coins = params.get("window_root"), evidence["coins"]
    try:
        key = Ch.ChallengerKey.from_json(params["challenger"]["key"])
    except (Ch.CoinError, KeyError, TypeError):
        raise Outside("the parameters' challenger names no key, which their parser refuses") from None
    if type(window) is not str:
        raise _Refused(f"coins: {NO_CHALLENGER}")
    try:
        if sorted(coins) != ["ledger", "main", "pins", "segment", "session"]:
            raise Ch.CoinError(DRAW_FORM)
        Ch.trusted_session_of(coins["ledger"], {"root": window}, key, coins["pins"], coins["main"], coins["session"], verify)
        log = coins["session"].get("log") if isinstance(coins["session"], dict) else None
        if type(log) is not list or not all(type(line) is str for line in log):
            raise Ch.CoinError(Ch.LOG_FORM)
        return Ch.coin_trees_of(log), log
    except Ch.CoinError as e:
        raise _Refused(f"coins: {e}") from None


def _record(path: Path, name: str) -> tuple[str, Any]:
    """A session's record: its text, and the JSON it parses to (None where it parses to none)."""
    try:
        text = path.read_bytes().decode()
    except UnicodeDecodeError:
        raise _Refused(f"session {name}'s record is not UTF-8") from None

    def refuse(_: str) -> Any:
        raise ValueError

    try:
        return text, json.loads(text, parse_constant=refuse)
    except (ValueError, RecursionError):
        return text, None


def _trees_held(trees: dict, log: Sequence[str], sts: Sequence[Any], blobs: Mapping[str, Path]) -> None:
    """`OneStageZk.membersTreesHeld`: each session's record and evidence on the blobs, then
    `challenger.members_tree_reason` at `TABLES`."""
    named = []
    for st in sts:
        for s in st["sessions"]:
            name = s["name"]
            record = _blob(blobs, s["record"], f"session {name}'s record")
            _, parsed = _record(record, name)
            reps = [(f, _blob(blobs, s["reps"][f], f"session {name}'s files: {f}")) for f in sorted(s["reps"])]
            try:
                evidence = CL.files_digest(record, reps).hex()
            except ValueError:
                raise _Refused(f"session {name}: {CL.EVIDENCE_FITS}") from None
            named.append((name, TABLES, parsed, evidence))
    why = Ch.members_tree_reason(trees, log, named)
    if why is not None:
        raise _Refused(why)


def _laid(ms: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    """`OneStageZk.laidHeld`: each member one range within the population, the counts summing to it."""
    n = record.get("population")
    if not _nat(n):
        raise _Refused("the record names no population")
    if not (all(len(m["units"]) == 1 and m["units"][0][1] <= n for m in ms.values())
            and sum(_count(m) for m in ms.values()) == n):
        raise _Refused(MEMBERS_LAID)


def _sampled(ms: Mapping[str, Any], held: Sequence[str], want: Sequence[int]) -> None:
    """`OneStageZk.sampledHeld`: each statement's member, in the statements' order, has a unit among the evidence's
    ``proved_units`` `want`; the first that has none refuses (`OneStageZk.unsampled`)."""
    for name in held:
        if not any(lo <= u < hi for u in want for lo, hi in ms[name]["units"]):
            raise _Refused(f"member {name}: the sample holds none of its units, and a member the sample missed has no "
                           "statement")


def _ports(name: str, registered: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A registered file's ports by the fields `classHeld` compares, in the file's order; `Outside` where one is not
    read. A file of another format, or a port holding a key `Flock.Registered.PORT_KEYS` doesn't name, is refused by name,
    as `Flock.Registered.portsOf` refuses it."""
    form = registered.get("format")
    if type(form) is str and form != REGISTERED_READS_V2:
        raise _Refused(f"member {name}: registered reads: not a {REGISTERED_READS_V2} file")
    ports = registered.get("ports")
    if not (isinstance(ports, dict) and ports):
        raise Outside(f"member {name}: its registered file names no ports")
    for port, e in ports.items():
        if isinstance(e, dict) and not set(e) <= set(PORT_KEYS):
            raise _Refused(f"member {name}: registered reads: port {port}: a {REGISTERED_READS_V2} port holds keys of "
                           f"[{', '.join(PORT_KEYS)}] and no other (no position: C and the partition give them)")
    out = []
    for port in sorted(ports):
        e = ports[port]
        q = {"value": e.get("value"), "root": e.get("root"), "leaves": e.get("leaves"), "words": e.get("words"),
             "word_bits": e.get("word_bits", RV.WORD_BITS), "domain": e.get("domain"),
             "row_schema": e.get(RV.SCHEMA_KEY, RV.SCHEMA)} if isinstance(e, dict) else None
        if not (q is not None and type(q["value"]) is str
                and all(type(q[k]) is str and len(q[k]) == 128 and set(q[k]) <= _HEX for k in ("root", "domain"))
                and _nat(q["leaves"]) and q["leaves"] > 0 and (q["words"] is None or (_nat(q["words"]) and q["words"] > 0))
                and q["word_bits"] in (16, 32) and q["row_schema"] in RV.SCHEMAS):
            raise Outside(f"member {name}: its registered file's port {port} is not read as a registered port")
        out.append(q)
    return out


def _class(ms: Mapping[str, Any]) -> None:
    """`OneStageZk.classHeld`: every port of one registered value names one root, leaf count, row width, domain and leaf
    schema, over all the members."""
    seen: dict[str, tuple[str, dict[str, Any]]] = {}
    for name, m in ms.items():
        for q in ([] if m["registered"] is None else _ports(name, m["registered"])):
            if q["value"] not in seen:
                seen[q["value"]] = (name, q)
                continue
            first, q0 = seen[q["value"]]
            k = next((k for k in ("root", "leaves", "words", "word_bits", "domain", "row_schema") if q0[k] != q[k]), None)
            if k is not None:
                a, b = ("none" if x[k] is None else x[k] for x in (q0, q))
                raise _Refused(f"registered: members {first} and {name} read the registered value {q['value']} as two "
                               f"({k} {a}, against {b}): a value the window registers is one tree, read by every "
                               "member at one root")


def pinned_read_why(member: str, port: str, value: str, what: str) -> str:
    """`OneStageZk.PINNED_READ_WHY`: the refusal of member ``member``'s registered port ``port`` reading the registered
    input ``value`` at another ``what`` (root, leaf count or domain) than the pin the docket's R8 accepted."""
    return (f"registered: member {member}'s port {port} reads the registered input {value} at another {what} than its "
            "pin of record: a registered input the weights' registration of record pins is read at the pin's root, leaf "
            "count and domain")


def unpinned_read_why(member: str, port: str, value: str) -> str:
    """`OneTree.UNPINNED_READ_WHY`."""
    return (f"registered: member {member}'s port {port} reads the registered input {value}, which the program marks, R8 "
            "accepted no pin of and the verifier holds no root of: a marked input is read only at a pin R8 accepted or a "
            "root the verifier holds")


def two_reads_why(member: str, param: str, w0: int, w1: int) -> str:
    """`OneStageZk.TWO_READS_WHY`: member ``member``'s units read ``param`` at C's input nodes ``w0`` and ``w1``."""
    return (f"registered reads: member {member}'s units read parameter {param} at C's input nodes {w0} and {w1}: a "
            "parameter that reads an input reads one node, each port one tree")


def read_form_why(k: int, param: str, what: str) -> str:
    """`Flock.OneCircuit.READ_FORM_WHY`: root node ``k``'s members read ``param`` in a form with no closed form here
    (``what``) where its argument reads an input node."""
    return (f"registered reads: parameter {param} of root node {k}: {what}, and its argument reads an input node: a "
            "parameter that reads an input reads one node, each row in one lane's run")


ReadPins = tuple[list[str], list[tuple[str, range]], list[tuple[int, str, int, int]], dict[str, dict[str, Any]],
                 Mapping[str, Mapping[str, Any]]]


def read_pins(params: Mapping[str, Any], program: Program) -> ReadPins:
    """`Docket.readPins` on the parameters' own verifier (`docket.read_pins`): refused by name (`docket.read_pins_why`)
    where the program marks an input the verifier does not expect and the partition is no template query applying to
    it, or the parameters carry no weights' registration of record R8 accepts."""
    w = params.get("weights_of_record")
    registered = params.get("registered") or {}
    roots = DK.held_entries(program, registered, params.get("row_schemas"), params.get("window_root"),
                            params.get("held"), params.get("supplied") or {}, params.get("pinned") or {},
                            params.get("outputs") or {}, params.get("log") or [])
    got = DK.read_pins(program, params["query"], registered,
                       None if w is None else (bytes.fromhex(w["document"]), w["digest"]), roots)
    if got is None:
        raise _Refused(DK.read_pins_why(program, params["query"]))
    return got


def _param_row(param: str) -> tuple[str, int | None]:
    """`Flock.Registered.paramRow`: ``p[i]`` as ``(p, i)``, any other name as itself and None."""
    head, sep, tail = param.partition("[")
    if sep and tail.endswith("]") and "[" not in tail and "]" not in tail[:-1] and tail[:-1].isdigit():
        return head, int(tail[:-1])
    return param, None


def _input(R: EG._Root, w: int, run: bool = False) -> bool:
    """Whether root node ``w`` is an input node of C, registered, supplied or run (`Flock.Registered.regNode`,
    `runNode`); with ``run``, a run input node alone."""
    if not (0 <= w < len(R.nodes) and R.nodes[w]["input"]):
        return False
    return not run or input_id(R.nodes[w]["key"])[0] == RUN


def _part_nodes(q: tuple) -> list[int]:
    """`Flock.Registered.partNodes`: the root nodes a reference part reads."""
    return [i for p, i, *_ in (q[1] if q[0] == "e" else q[2]) if not p]


def reg_rows(R: EG._Root, k: int, m0: int, n: int, param: str) -> tuple[int, int] | None:
    """`Flock.OneCircuit.regRows?`: the input node every member of ``[m0, m0 + n)`` of root node ``k`` reads ``param``
    at, with a row's leaves; None where they read only root nodes that are no input; `_Refused` (`read_form_why`) where
    the argument reads an input node and the members' rows have no closed form here or read two nodes."""
    node = R.nodes[k]
    p, row = _param_row(param)
    i = next((i for i, (q, _t) in enumerate(node["params"]) if q == p), None)
    if i is None:
        raise _Refused(f"registered reads: parameter {param} of root node {k}: {node['key']} has no parameter {p}")
    ty = node["params"][i][1]
    if row is None:
        t0, ln = 0, EG._leaves(ty)
    elif ty[0] == "Array" and row < ty[1]:
        t0, ln = row * EG._leaves(ty[2]), EG._leaves(ty[2])
    elif ty[0] == "Array":
        raise _Refused(f"registered reads: parameter {param} of root node {k}: row {row} of {ty[1]}")
    else:
        raise _Refused(f"registered reads: parameter {param} of root node {k}: a row of a parameter that is no array")
    a = node["args"][i]
    parts = a["parts"]
    inputs = any(_input(R, w) for q in parts for w in _part_nodes(q))

    def edge(what: str) -> None:
        if inputs:
            raise _Refused(read_form_why(k, param, what))
        return None

    dims = a["dims"]
    L = 0 if dims is None else dims[1]
    if dims is not None and (dims[1] == 0 or t0 % dims[1] + ln > dims[1]):
        return edge("a row across members")
    starts = [0]
    for q in parts:
        starts.append(starts[-1] + EG._count(q))

    def x_of(m: int) -> int:
        return t0 if dims is None else (t0 // dims[1] * dims[0]) * dims[1] + t0 % dims[1] + m * dims[1]

    def wire(x: int, c: int, L: int) -> int | None:
        j = next((j for j, q in enumerate(parts) if starts[j] <= x < starts[j] + EG._count(q)), None)
        if j is None or x + (c - 1) * L + ln > starts[j] + EG._count(parts[j]) or parts[j][0] != "g":
            return None
        try:
            return EG._lane_wire(parts[j], x - starts[j], L, ln, c)[0]
        except EG.Refusal:
            return None

    if (w := wire(x_of(m0), max(n, 1), L)) is not None:
        return (w, ln) if _input(R, w) else None
    got: list[int] = []
    for m in range(max(n, 1)):
        w = wire(x_of(m0 + m), 1, 0)
        if w is None:
            return edge("a row of no lane's run (an explicit part, or across parts or lanes)")
        if w not in got:
            got.append(w)
    if len(got) == 1:
        return (got[0], ln) if _input(R, got[0]) else None
    if any(_input(R, w) for w in got):
        raise _Refused(read_form_why(k, param, f"its members read root nodes [{', '.join(map(str, got))}]"))
    return None


def _param_input(R: EG._Root, k: int, m0: int, n: int, p: str, ty: Any) -> tuple[int, int] | None:
    """`OneStageZk.paramInput`: `reg_rows` of the whole parameter, else, for an array, of each row, which must be one;
    the whole parameter's refusal otherwise."""
    try:
        return reg_rows(R, k, m0, n, p)
    except _Refused:
        if ty[0] != "Array" or ty[1] == 0:
            raise
        rows = []
        for i in range(ty[1]):
            try:
                rows.append(reg_rows(R, k, m0, n, f"{p}[{i}]"))
            except _Refused:
                rows.append(_Refused)
        if rows[0] is not _Refused and all(r == rows[0] for r in rows):
            return rows[0]
        raise


def range_node_why(member: str, k: int) -> str:
    """`OneTree.RANGE_NODE_WHY`."""
    return (f"member {member}: its units meet a range the partition gives at C's root node {k}, which C does not have: a "
            "range is of C's root nodes")


def member_inputs(R: EG._Root, ranges: Sequence[Sequence[Any]], name: str,
                  units: Iterable[Sequence[int]]) -> list[tuple[str, int, int]]:
    """`OneStageZk.memberInputs`: each ``(parameter, node, row leaves)`` member ``name``'s units read at an input node
    of C, once, over every range they meet; two nodes at one parameter refused (`two_reads_why`), and a range at a root
    node C does not have (`range_node_why`)."""
    out: list[tuple[str, int, int]] = []
    for lo, hi in units:
        for k, _t, first, count in ranges:
            a, b = max(lo, first), min(hi, first + count)
            if b <= a:
                continue
            if not 0 <= k < len(R.nodes):
                raise _Refused(range_node_why(name, k))
            for p, ty in R.nodes[k]["params"]:
                try:
                    got = _param_input(R, k, a - first, b - a, p, ty)
                except _Refused as e:
                    raise _Refused(f"member {name}: {e}") from None
                if got is None:
                    continue
                w0 = next((w for q, w, _ in out if q == p), None)
                if w0 is None:
                    out.append((p, *got))
                elif w0 != got[0]:
                    raise _Refused(two_reads_why(name, p, w0, got[0]))
    return out


def _port_param(port: str, e: Mapping[str, Any]) -> str:
    """`OneStageZk.portParam`: the template parameter a registered port reads, its ``param`` (else its name) without a
    row index ``[i]``."""
    name = e.get(G.PARAM_KEY) or port
    head, sep, tail = name.partition("[")
    return head if sep and tail.endswith("]") and tail[:-1].isdigit() and "[" not in tail else name


def _pin_of(rp: ReadPins, v: str) -> tuple[str, int, str] | None:
    """`OneTree.pinOf`: R8's accepted pin of ``v``, else the root the verifier itself holds of it (`docket.held_entries`),
    as ``(root, leaves, domain)``; never a root the record names."""
    pin = rp[3].get(v) or rp[4].get(v)
    return None if pin is None else (pin["root"], pin["leaves"], pin["domain"])


def pins_held(ms: Mapping[str, Any], R: EG._Root, ranges: Sequence[Sequence[Any]], program: Program,
              marks: Sequence[tuple[str, range]], rp: ReadPins) -> None:
    """`OneStageZk.pinsHeld`: each member's registered port at a parameter its units read at a registered input node
    whose gates are a marked value's (`member_inputs`, `partition.held_value` over ``marks``), a tree's or not, and each
    port naming a marked input the verifier does not expect, is at that value's entry of record (`_pin_of`): its root,
    leaf count and domain; refused by name (`pinned_read_why`, `unpinned_read_why`)."""
    for name, m in ms.items():
        reads = member_inputs(R, ranges, name, m["units"])
        ports = (m["registered"] or {}).get("ports") or {}
        for port in sorted(ports):
            e = ports[port]
            if not isinstance(e, dict):
                continue
            p = _port_param(port, e)
            w = next((w for q, w, _ in reads if q == p), None)
            v = None if w is None else P.held_value(program, marks, w)
            if v is None and e.get("value") in rp[0]:
                v = e.get("value")
            if v is None:
                continue
            pin = _pin_of(rp, v)
            if pin is None:
                raise _Refused(unpinned_read_why(name, port, v))
            for got, want, what in ((e.get("root"), pin[0], "root"), (e.get("leaves"), pin[1], "leaf count"),
                                    (e.get("domain"), pin[2], "domain")):
                if got != want:
                    raise _Refused(pinned_read_why(name, port, v, what))


def _template_of(name: str) -> tuple[str, str] | None:
    """`OneStageZk.templateOf`: a name read at its last ``/``; None for a name with none."""
    i = name.rfind("/")
    return None if i < 0 else (name[:i], name[i + 1:])


def _templates(names: Any) -> set[str]:
    """`OneStageZk.templatesOf`: the templates of the members named `names`."""
    return {t[0] for n in names if (t := _template_of(n)) is not None}


def _field_trees(ts: set[str], record: Mapping[str, Any]) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """`OneStageZk.fieldTreesOf`: each root of `record` whose port is ``<template>/<field>`` of a template of `ts`, by
    template, as ``(field, root)`` in the record's order, the root under the bare field."""
    out: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    roots = record.get("roots")
    for r in roots if isinstance(roots, list) else ():
        port = r.get("port") if isinstance(r, dict) else None
        t = _template_of(port) if type(port) is str else None
        if t is not None and t[0] in ts:
            out.setdefault(t[0], []).append((t[1], {"domain": r.get("domain"), "leaves": r.get("leaves"), "port": t[1],
                                                    "root": r.get("root"), "tree": port}))
    return out


def _own_fields(ft: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]], name: str,
                roots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """`OneStageZk.ownFieldsOf`: the member `name`'s roots with its template's fields' trees at the fields it has no
    root of."""
    got = [dict(r) for r in roots]
    t = _template_of(name)
    for f, e in ft.get(t[0], ()) if t is not None else ():
        if not any(r.get("port") == f for r in got):
            got.append(dict(e))
    return got


def _tree_ports_held(ms: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    """`OneStageZk.treePortsHeld`: every port of a member's registered file that opens a committed tree, a value
    ``committed/<port>`` or a run input node's ``input/n<w>``, is the record's root of it, its root, domain and leaf
    count (`OneStageZk.portTreeHeld`); a port whose value is a field tree's name is opened only as the member's own
    output, ``<template>/<port>``, and any other refuses first (`wired_tree_why`)."""
    roots = record.get("roots")
    by_port = {r["port"]: r for r in (roots if isinstance(roots, list) else ())
               if isinstance(r, dict) and type(r.get("port")) is str}
    for name, m in ms.items():
        reg, tmpl = m["registered"], _template_of(name)
        for port in sorted(reg["ports"]) if reg is not None else ():
            e = reg["ports"][port]
            value = e["value"]
            field = not value.startswith(COMMITTED) and _field_tree_name(value)
            if field and (tmpl is None or f"{tmpl[0]}/{port}" != value):
                raise _Refused(wired_tree_why(name, port))
            t = value[len(COMMITTED):] if value.startswith(COMMITTED) else value if G.is_fresh_value(value) or field \
                else None
            if t is None:
                continue
            r = by_port.get(t)
            if r is None:
                raise _Refused(f"registered: member {name}'s port {port} opens the committed tree {t}, and the "
                               f"registration holds no root {t}: a committed tree a statement opens is one its window "
                               "registered before any coin")
            what = next((w for w, ok in (("root", r.get("root") == e["root"]), ("domain", r.get("domain") == e["domain"]),
                                         ("leaf count", _nat(r.get("leaves")) and r["leaves"] == e["leaves"]))
                         if not ok), None)
            if what is not None:
                raise _Refused(f"registered: member {name}'s port {port} opens the committed tree {t} at another {what} "
                               "than the registration's: a statement opens its window's registered tree, never another "
                               "window's or another member's")


def _roots(ms: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """`OneStageZk.rootsOf`: the docket's roots by the member each is of, its port without the member's name; a root
    ``<template>/<field>`` of a member's template, not named as a member, is the window's tree of the field, no
    member's own (`_field_trees`), and a member's own root at such a field refuses (`OneStageZk.ownFieldsHeld`)."""
    roots = record.get("roots")
    if type(roots) is not list:
        raise _Refused("roots: the registration's roots are not a list")
    ts = _templates(ms)
    if (n := next((n for n in ms if n in ts), None)) is not None:
        raise _Refused(DK.name_template(n))
    if "input" in ms:
        raise _Refused(INPUT_MEMBER)
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in ms}
    for r in roots:
        port = r.get("port") if isinstance(r, dict) else None
        if type(port) is not str:
            raise _Refused("roots: a root of the registration names no port")
        if port == TABLE_PORT:
            raise _Refused(ROW_TABLES)
        if G.is_fresh_value(port):
            continue
        mine = [(port[:i], port[i + 1:]) for i, c in enumerate(port) if c == "/" and port[:i] in out]
        if not mine:
            if port not in out and (t := _template_of(port)) is not None and t[0] in ts:
                continue
            raise _Refused(f"roots: the registration's root {port} is no member's: a member's roots are named "
                           "<member>/<port>, and a window's tree of a template's field <template>/<field>")
        if len(mine) > 1:
            raise _Refused(f"roots: the registration's root {port} is two members': a member's roots are named "
                           "<member>/<port>, and a root is one member's")
        name, bare = mine[0]
        out[name].append({"domain": r.get("domain"), "leaves": r.get("leaves"), "port": bare, "root": r.get("root")})
    ft = _field_trees(ts, record)
    for name, rs in out.items():
        t = _template_of(name)
        if (f := next((f for f, _ in ft.get(t[0], ()) if any(r["port"] == f for r in rs)), None) if t else None) is not None:
            raise _Refused(DK.own_field(name, f))
    return out


def field_trees_why(what: str) -> str:
    """`OneStageZk.FIELD_TREES_WHY`."""
    return ("roots: the registration's trees of its templates' output fields are not the ones C and the partition give "
            f"({what}): a window registers exactly one tree <template>/o<j> for each output field of each member's "
            "template, its leaves the template's units in C")


def _derived_field_trees(program: Program, query: Mapping[str, Any], ms: Mapping[str, Any]) -> list[tuple[str, int]]:
    """`OneStageZk.derivedFieldTrees`: for each member's template, in the members' name order (Lean's `membersOf` reads
    them from a JSON object, whose keys it holds sorted), ``<template>/o<j>`` for each return field
    (`return_fields`) of the one Definition the partition's ranges over C give its members' units, its leaves that
    Definition's units."""
    try:
        ranges = P._template(program, query, "the members' partition").ranges
    except P.QueryRefused as e:
        raise _Refused(str(e)) from None
    fns = {PO.descriptor_id(fn): fn for _k, fn in PO.calls(program)}
    tids: dict[str, str] = {}
    for name in sorted(ms):
        t = _template_of(name)
        if t is None:
            raise _Refused(field_trees_why(f"the member {name} names no template"))
        met = list(dict.fromkeys(tid for lo, hi in ms[name]["units"] for _k, tid, first, n in ranges
                                 if first < hi and lo < first + n))
        if len(met) != 1:
            raise _Refused(field_trees_why(f"the member {name}'s units are not one Definition's in C"))
        if t[0] in tids:
            if tids[t[0]] != met[0]:
                raise _Refused(field_trees_why(f"the template {t[0]}'s members' units are two Definitions'"))
        elif met[0] in tids.values():
            raise _Refused(field_trees_why(f"the Definition {met[0]}'s units are two templates'"))
        else:
            tids[t[0]] = met[0]
    out: list[tuple[str, int]] = []
    for t, tid in tids.items():
        n = sum(c for _k, r, _a, c in ranges if r == tid)
        out += [(f"{t}/o{j}", n) for j in range(return_fields(fns[tid].signature()[1]))]
    return out


def return_fields(ret: Any) -> int:
    """`Circuits.Descriptor.fieldsOf`'s length: a tuple's fields, none for the empty tuple; any other type one."""
    fields = getattr(ret, "fields", None)
    return 1 if fields is None else len(fields)


def _field_trees_held(want: Sequence[tuple[str, int]], ts: set[str], record: Mapping[str, Any]) -> None:
    """`OneStageZk.fieldTreesHeld`: each derived tree a root of `record`, once, at its derived leaf count, and no other
    root ``<template>/<field>`` of a template of `ts`."""
    roots = record.get("roots")
    got = [(r["port"], r.get("leaves")) for r in (roots if isinstance(roots, list) else ())
           if isinstance(r, dict) and type(r.get("port")) is str and (t := _template_of(r["port"])) is not None
           and t[0] in ts]
    for tree, n in want:
        held = [leaves for port, leaves in got if port == tree]
        if not held:
            raise _Refused(field_trees_why(f"it omits {tree}"))
        if len(held) > 1:
            raise _Refused(field_trees_why(f"it holds {tree} more than once"))
        if not (_nat(held[0]) and held[0] == n):
            raise _Refused(field_trees_why(f"it holds {tree} at {json.dumps(held[0])} leaves, not {n}"))
    if (port := next((p for p, _ in got if p not in dict(want)), None)) is not None:
        raise _Refused(field_trees_why(f"it holds {port}, no output field's tree"))


def _slice(m: Mapping[str, Any], roots: Sequence[Mapping[str, Any]], record: Mapping[str, Any]) -> dict[str, Any]:
    """`OneStageZk.Member.slice`: the member's count of units, its roots of the docket and its scheme."""
    return {"population": _count(m), "roots": list(roots), "scheme": record.get("scheme")}


def slice_no_port_why(member: str) -> str:
    """`OneStageZk.SLICE_NO_PORT_WHY`."""
    return f"member {member}: a root of its slice names no port"


def port_twice_why(member: str, port: str) -> str:
    """`OneStageZk.PORT_TWICE_WHY`: the refusal of a member's slice whose roots name ``port`` twice."""
    return (f"member {member}: its slice names port {port} twice: a member's slice holds one root of each port, the one "
            "its public file's root there is held to")


def _ports_once(ms: Mapping[str, Any], roots: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    """`OneStageZk.portsOnceAll`: each member's slice, in the members' name order (Lean's `membersOf`), names each port
    once, ports compared as strings: a root that names no port as a string refuses (`slice_no_port_why`), then the
    first port a later root names again (`OneStageZk.portTwice`, `port_twice_why`)."""
    for name in sorted(ms):
        ports = [r.get("port") if isinstance(r, Mapping) else None for r in roots.get(name, ())]
        if any(type(p) is not str for p in ports):
            raise _Refused(slice_no_port_why(name))
        if (p := next((p for i, p in enumerate(ports) if p in ports[i + 1:]), None)) is not None:
            raise _Refused(port_twice_why(name, p))


def _members_roots(ms: Mapping[str, Any], record: Mapping[str, Any]) -> tuple[dict[str, list[dict[str, Any]]],
                                                                             dict[str, list[tuple[str, dict[str, Any]]]]]:
    """`OneStageZk.membersRootsOf` before its wiring: `_roots`', each member's own output ports at its template's
    fields' trees (`_own_fields`), and the window's trees (`_field_trees`) the wiring reads."""
    roots = _roots(ms, record)
    ft = _field_trees(_templates(ms), record)
    return {name: _own_fields(ft, name, rs) for name, rs in roots.items()}, ft


def _local_bases(ranges: Sequence[Sequence[Any]]) -> list[int]:
    """`OneStageZk.localBases`: each range's base among its template's own units."""
    lbase, seen = [], {}
    for _k, t, _a, c in ranges:
        lbase.append(seen.get(t, 0))
        seen[t] = seen.get(t, 0) + c
    return lbase


def _wire_reaches(ranges: Sequence[Sequence[Any]], lbase: Sequence[int], first: int, n: int,
                  w: Mapping[str, Any]) -> bool:
    """`OneStageZk.wireReaches`: whether the wire `w` enters the member whose units start at `first`, `n` of them."""
    units = max(w["cut"][1], 1)
    return any(t == w["reader"] and a <= first and first + n <= a + c
               and (lo := lbase[j] + first - a) < w["at"] + math.prod(w["radix"]) // units and w["at"] < lo + n
               for j, (_k, t, a, c) in enumerate(ranges))


def _reaching(ranges: Sequence[Sequence[Any]], lbase: Sequence[int],
              wires: Sequence[Mapping[str, Any]]) -> Callable[[int, int], list[Mapping[str, Any]]]:
    """``(first, n) ->`` the wires of ``wires``, in their order, that `_wire_reaches` the member of units ``first`` to
    ``first + n``, for ``n >= 1``: the partition's ranges are disjoint and ascending (`TemplatePartition`), so at most one
    nonempty range holds the member, found by bisection, and only wires its template reads can reach it."""
    held = [(a, j) for j, (_k, _t, a, c) in enumerate(ranges) if c]
    starts = [a for a, _ in held]
    by_reader: dict[str, list[Mapping[str, Any]]] = {}
    for w in wires:
        by_reader.setdefault(w["reader"], []).append(w)

    def reach(first: int, n: int) -> list[Mapping[str, Any]]:
        i = bisect.bisect_right(starts, first) - 1
        if i < 0:
            return []
        j = held[i][1]
        _k, t, a, c = ranges[j]
        if not first + n <= a + c:
            return []
        return [w for w in by_reader.get(t, ()) if _wire_reaches([ranges[j]], [lbase[j]], first, n, w)]
    return reach


def _wired_roots(desc: Mapping[str, Any], ranges: Sequence[Sequence[Any]], ms: Mapping[str, Any],
                 ft: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]],
                 roots: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """`OneStageZk.fieldRoots` at C's wiring under the partition's `ranges` (`derivedWiring`), where C's root has an
    edge (`hasEdge`): each member's roots with, for each wire whose reader units meet its own, its writer's output
    field's tree under the reader's port, the first wire's of each port. A member's own root at a wired port refuses
    (`own_root_why`); `Outside` where C's wiring is refused."""
    if not EG.has_edge(desc):
        return roots
    try:
        got = EG.edges(desc, ranges, {"templates": {t: {} for _k, t, _a, _c in ranges}, "cuts": EG.io_cuts(desc, ranges)})
    except ValueError as e:
        raise Outside(f"C's wiring: {e}") from None
    if "refusal" in got:
        raise Outside(f"C's wiring: {got['refusal']}")
    lbase = _local_bases(ranges)
    reach = _reaching(ranges, lbase, got["wires"])
    out = {}
    for name, m in ms.items():
        first, n = m["units"][0][0], _count(m)
        own = roots.get(name, [])
        wires = reach(first, n) if n >= 1 else [w for w in got["wires"] if _wire_reaches(ranges, lbase, first, n, w)]
        if (w := next((w for w in wires if any(r.get("port") == w["port"] for r in own)), None)):
            raise _Refused(own_root_why(name, w["port"]))
        mine = list(own)
        for w in wires:
            if not any(r.get("port") == w["port"] for r in mine):
                e = next((e for f, e in ft.get(w["writer"], ()) if f == w["out"]), None)
                if e is not None:
                    mine.append({"domain": e.get("domain"), "leaves": e.get("leaves"), "port": w["port"],
                                 "root": e.get("root")})
        out[name] = mine
    return out


def slices(members: Mapping[str, Sequence[Sequence[int]]], record: Mapping[str, Any], *, program: Program,
           query: Mapping[str, Any], window_root: str | None = None, tied: Sequence[range] = ()
           ) -> dict[str, dict[str, Any]]:
    """Each member's slice of the docket `record` (`OneStageZk.Member.slice`), for the members `members` by name,
    each its units as ranges ``[lo, hi)``: ``{name: {population, roots, scheme}}``, the member's count of units, the
    docket's roots that are its own, its name taken off each port (`OneStageZk.rootsOf`), its template's fields' trees at
    the output ports it has no root of (`OneStageZk.ownFieldsOf`), the docket's scheme, each port C reads from a run
    input node at the docket's ``input/n<w>`` under the window's open root ``window_root`` outside the tied gates
    ``tied`` (`_run_input_roots`), and each wired port at its writer's tree under the verifier's `program` and `query` (`_wired`, Lean's `fieldRoots`). `population_why` holds a member's
    public file to its slice. `ValueError` with Lean's reason where the members are not laid over the docket's
    population (`OneStageZk.laidHeld`), a root of the docket is not one member's, or a slice names a port twice
    (`_ports_once`): an audit outside the per-member
    form's run reads a member's slice here, never by a rule of its own."""
    ms = {name: {"units": [list(r) for r in units]} for name, units in members.items()}
    try:
        _laid(ms, record)
        roots, ft = _members_roots(ms, record)
        roots = _wired(program, query, json.loads(json.dumps(codec.encode_program(program))), ms, record, roots, ft,
                       window_root, tied)
        _ports_once(ms, roots)
    except _Refused as e:
        raise ValueError(str(e)) from None
    return {name: _slice(m, roots[name], record) for name, m in ms.items()}


def _wired(program: Program, query: Mapping[str, Any], desc: Mapping[str, Any], ms: Mapping[str, Any],
           record: Mapping[str, Any], roots: dict[str, list[dict[str, Any]]],
           ft: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]], window: str | None,
           tied: Sequence[range]) -> dict[str, list[dict[str, Any]]]:
    """`OneStageZk.membersRootsOf` past `_members_roots`: the window's field trees held to C's (`_field_trees_held`),
    each member's ports at C's run input nodes' trees (`_run_input_roots`), then each member's wired ports at their
    writers' trees (`_wired_roots`)."""
    _field_trees_held(_derived_field_trees(program, query, ms), _templates(ms), record)
    ranges = P._template(program, query, "the members' partition").ranges
    roots = _run_input_roots(program, desc, ranges, window, tied, ms, record, roots)
    return _wired_roots(desc, ranges, ms, ft, roots)


def run_own_root_why(member: str, port: str, w: int) -> str:
    """`OneStageZk.RUN_OWN_ROOT_WHY`: the refusal of a member's root of its own at a port C reads from run input node
    ``w``."""
    return (f"member {member}: port {port} reads C's run input node {w}, and the registration holds a root of its own "
            f"for it: a run input node's rows are committed once, in input/n{w}")


def run_tree_missing_why(member: str, port: str, w: int) -> str:
    """`OneStageZk.RUN_TREE_MISSING_WHY`."""
    return (f"member {member}: port {port} reads C's run input node {w}, and the record holds no root of input/n{w}: a "
            "run input node's rows are committed once, in its one tree, before any coin")


def run_tree_why(w: int, what: str) -> str:
    """`OneStageZk.RUN_TREE_WHY`."""
    return (f"registered: the record's input/n{w} has another {what} than C's run input node {w} and the window's open "
            "root give it: a run input node's one tree binds its window")


def run_tree_unbound_why(w: int) -> str:
    """`OneStageZk.RUN_TREE_UNBOUND_WHY`."""
    return (f"registered: C's run input node {w} is read from its one tree input/n{w}, and the verifier names no window "
            "open root (window_root) for its domain to bind")


def run_tree_held(R: EG._Root, prog: str, window: str | None, w: int, ln: int, r: Mapping[str, Any]) -> None:
    """`OneStageZk.runTreeHeld`: the record's entry ``r`` of ``input/n<w>`` has C's node's rows (out leaves times
    members over ``ln``) and the domain under the program digest ``prog``, a row's 16-bit words (``ln`` leaves of the
    node's leaf width, as the writer packs a port's row), and the window's open root ``window``. The domain's schema is a constant of the domain, not the rows' format."""
    if window is None:
        raise _Refused(run_tree_unbound_why(w))
    node = R.nodes[w]
    rows = node["out_leaves"] * node["members"] // ln if ln else 0
    leaves = r.get("leaves")
    if not (_nat(leaves) and leaves == rows):
        raise _Refused(run_tree_why(w, "leaf count"))
    try:
        words = ln * (EG._bits(node["ret"]) // node["out_leaves"]) // 16 if node["out_leaves"] else 0
        d = RV.domain(prog, f"input/n{w}", words, rows, 16, window=window).domain_id.hex()
    except ValueError as e:
        raise _Refused(str(e)) from None
    if r.get("domain") != d:
        raise _Refused(run_tree_why(w, "domain"))


def run_input_roots_at(R: EG._Root, program: Program, ranges: Sequence[Sequence[Any]], prog: str, window: str | None,
                       tied: Sequence[range], name: str, m: Mapping[str, Any], own: Sequence[Mapping[str, Any]],
                       by_port: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """`OneStageZk.runInputRootsAt`: for each parameter member ``name``'s units read at a run input node ``w`` of C
    (`member_inputs`: every proof block it holds) outside the tied gates ``tied``, the record's root of
    ``input/n<w>`` under the port; refused by name where the record holds none (`run_tree_missing_why`), the member
    holds a root of its own there (`run_own_root_why`) or the tree is not C's and the window's (`run_tree_held`)."""
    got = list(own)
    for p, w, ln in member_inputs(R, ranges, name, m["units"]):
        if not _input(R, w, run=True) or any(P.within(program, g, w) for g in tied):
            continue
        t = f"input/n{w}"
        if (r := by_port.get(t)) is None:
            raise _Refused(run_tree_missing_why(name, p, w))
        if any(e.get("port") == p for e in own):
            raise _Refused(run_own_root_why(name, p, w))
        run_tree_held(R, prog, window, w, ln, r)
        got.append({"domain": r.get("domain"), "leaves": r.get("leaves"), "port": p, "root": r.get("root"), "tree": t})
    return got


def tied_roots_why(w: int) -> str:
    """`OneStageZk.TIED_ROOTS_WHY`: the refusal of two roots for one tied run input node ``w``."""
    return f"members: C's tied run input node {w} is read under two roots: every reader of a tied node opens one tree"


def tied_roots_at(R: EG._Root, program: Program, ranges: Sequence[Sequence[Any]], tied: Sequence[range], name: str,
                  m: Mapping[str, Any], own: Sequence[Mapping[str, Any]], seen: dict[int, Any]) -> dict[int, Any]:
    """`OneStageZk.tiedRootsAt`: ``seen``, the root each tied run input node is read under so far, with each node inside
    the tied gates ``tied`` that member ``name`` reads (`member_inputs`) at the root ``own`` holds for its port; refused
    (`tied_roots_why`) where ``seen`` holds another root for the node. That the root is the right one is the later
    period check's."""
    for p, w, _ln in member_inputs(R, ranges, name, m["units"]):
        if not _input(R, w, run=True) or not any(P.within(program, g, w) for g in tied):
            continue
        if (e := next((e for e in own if e.get("port") == p), None)) is None:
            continue
        if seen.setdefault(w, e.get("root")) != e.get("root"):
            raise _Refused(tied_roots_why(w))
    return seen


def _run_input_roots(program: Program, desc: Mapping[str, Any], ranges: Sequence[Sequence[Any]], window: str | None,
                     tied: Sequence[range], ms: Mapping[str, Any], record: Mapping[str, Any],
                     roots: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """`OneStageZk.runInputRoots`: `run_input_roots_at` for each member, under the program's digest, the window's open
    root ``window`` and the tied parameters' gates ``tied``, each tied node read under one root across the members
    (`tied_roots_at`)."""
    R = EG._Root(desc)
    by_port = {r.get("port"): r for r in record.get("roots") or () if isinstance(r, Mapping)}
    prog = P.program_digest(program)
    out = dict(roots)
    seen: dict[int, Any] = {}
    for name, m in ms.items():
        tied_roots_at(R, program, ranges, tied, name, m, roots.get(name, []), seen)
        out[name] = run_input_roots_at(R, program, ranges, prog, window, tied, name, m, roots.get(name, []), by_port)
    return out


def tied_gates(program: Program, params: Mapping[str, Any]) -> list[range]:
    """`Docket.tiedGates` on the parameters: the gates of the tied names R8 lets through."""
    return [g for g in DK._tied(program, params.get("tied") or (), list(params.get("registered") or {}),
                                params.get("held") or ()) if g is not None]


def _meta(circuit: Path) -> Any:
    """`Flock.HmRow.metaOf`: the circuit file's META, read off its second line alone."""
    with open(circuit, "rb") as f:
        head = f.readline()
        line = f.readline()
    if not head.endswith(b"\n"):
        raise _Refused("circuit: no META line")
    try:
        text = line.decode().rstrip(" \t\n\r\x0b\x0c")
    except UnicodeDecodeError:
        raise _Refused("circuit: not UTF-8") from None
    if not text.startswith("META "):
        raise _Refused("circuit: no META line")
    try:
        return json.loads(text[5:])
    except ValueError:
        raise Outside("the circuit file's META is not JSON, whose refusal is the Lean parser's own") from None


def _header(public: Path) -> Any:
    """`FilePart.header`: the public file's header line as JSON; `ValueError` naming why it is none."""
    with open(public, "rb") as f:
        line = f.readline()
    try:
        return json.loads(line.removesuffix(b"\n").decode())
    except UnicodeDecodeError:
        raise ValueError("the public file's header is not UTF-8") from None
    except ValueError as e:
        raise ValueError(f"the public file's header is not JSON: {e}") from None


#: `OneStageZk.POPULATION_WHY`: a header's ``instances`` not its slice's population, refused before any port is dropped
POPULATION_WHY = "public file: not the registration's: population"


def shared_unread(port: str, tree: str) -> str:
    """`OneStageZk.SHARED_UNREAD`: the refusal of a member's header port ``port`` held in its template's shared field tree
    ``tree``, that its registered reads open no row of there and whose header root is not the registration's."""
    return (f"public file: port {port} is held in the shared tree {tree}, and the member's registered reads open no row "
            "of it there: a member's rows of a tree other members' rows share are held to the registration's root by "
            "opening it at the leaves C gives, and the header's own root there is no commitment to them")


def registered_at(held: Mapping[str, Any], p: str, v: str) -> bool:
    """`OneStageZk.registeredAt`: the verifier's registered file's ports ``held`` (its ``ports``, by name) hold ``p`` at
    ``v``, which names no committed tree, run input's tree or field tree."""
    e = held.get(p)
    return (isinstance(e, dict) and e.get("value") == v and not v.startswith(COMMITTED) and not G.is_fresh_value(v)
            and not _field_tree_name(v))


def unregistered_why(p: str, v: str) -> str:
    """`OneStageZk.UNREGISTERED_WHY`: the refusal of a header port ``p`` the slice holds no root of, read at ``v``, which
    the verifier's registered file holds at ``p`` as no registered value."""
    return (f"public file: port {p} reads {v}, and the verifier's registered file holds no registered value {v} at {p}: "
            "a port the registration holds no root of is held only at a registered value's root")


def _held_at(reads: Collection[tuple[str, str]], held: Mapping[str, Any], slice_: Mapping[str, Any],
             header: Any) -> tuple[Any, Any]:
    """`OneStageZk.headerHeldAt` before `headerHeld`: the slice and the header without each port the slice holds at a
    template's field tree (its entry's ``tree``) that ``reads`` opens there, and each it holds no root of, that the
    header names and ``reads`` names at a value the verifier's registered file's ports ``held`` hold there as a
    registered value (`registered_at`); `_Refused` (`shared_unread`) for one ``reads`` doesn't open whose header root
    or domain is not the registration's, and (`unregistered_why`) for one read at any other value."""
    h = header if isinstance(header, dict) else {}
    if not _same(h.get("instances"), slice_.get("population")):
        raise _Refused(POPULATION_WHY)
    fv = h["frame_v3"] if isinstance(h.get("frame_v3"), dict) else {}
    roots = fv["roots"] if isinstance(fv.get("roots"), dict) else {}
    domains = fv["domain_ids"] if isinstance(fv.get("domain_ids"), dict) else {}
    drop = set()
    for r in slice_["roots"]:
        p, t = r.get("port"), r.get("tree")
        if type(p) is not str or type(t) is not str:
            continue
        if (p, t) in reads:
            drop.add(p)
        elif roots.get(p) is not None and (roots[p] != r.get("root") or domains.get(p) != r.get("domain")):
            raise _Refused(shared_unread(p, t))
    ported = {r.get("port") for r in slice_["roots"]}
    for p, v in sorted(reads):
        if p not in ported and roots.get(p) is not None:
            if not registered_at(held, p, v):
                raise _Refused(unregistered_why(p, v))
            drop.add(p)
    if not drop:
        return slice_, header
    fv2 = {**fv, "roots": {k: v for k, v in roots.items() if k not in drop},
           "domain_ids": {k: v for k, v in domains.items() if k not in drop}}
    return {**slice_, "roots": [r for r in slice_["roots"] if r.get("port") not in drop]}, {**h, "frame_v3": fv2}


def registered_reads(registered: Mapping[str, Any] | None) -> tuple[set[tuple[str, str]], Mapping[str, Any]]:
    """`OneStageZk.Member.reads` and `Member.held`: the member's registered file's ``(port, value)`` reads and its ports
    by name, none without one."""
    held = registered["ports"] if registered is not None else {}
    return {(p, e["value"]) for p, e in held.items()}, held


def _walk(strata: Sequence[tuple[str, Sequence[Sequence[int]]]]) -> int | None:
    """`FilePart.sizes`' walk: N where the strata read stratum by stratum are the units ``0, ..., N - 1`` in order."""
    at = 0
    for _t, rs in strata:
        for lo, hi in rs:
            if lo != at or hi < lo:
                return None
            at = hi
    return at


def population_why(units: Sequence[Sequence[int]], strata: Sequence[tuple[str, Sequence[Sequence[int]]]], header: Any,
                   slice_: Mapping[str, Any], reads: Collection[tuple[str, str]] = (),
                   held: Mapping[str, Any] | None = None, subset: bool = False) -> str | None:
    """`OneStageZk.Member.population` past its files' reads: why a member of the units `units` (ranges ``[lo, hi)``) is
    not what its public file's header `header` holds at the docket's `strata` and the member's slice `slice_` of
    the record (``{population, roots, scheme}``); None when it is. The docket's strata read stratum by stratum
    are its units in order and hold the member's (`MEMBER_STRATA`), or under a subset design (`subset`,
    `Member.subset`) are a disjoint cover of them that holds the member's (`MEMBER_COVER`), the header's ``units.indices`` are exactly the
    member's units in order (`MEMBER_UNITS`), a port of its ``frame_v3`` roots the slice holds no root of is refused
    by name, the first in key order (`unwired_why`), and its scheme, ``frame_v3`` roots, their ``domain_ids`` and its
    ``instances`` are the slice's (`Record.matchesPublic`'s codes)."""
    walk = _walk(strata)
    if subset and walk is None:
        n = P.cover(strata)
        if n is None or not all(hi <= n for _lo, hi in units):
            return MEMBER_COVER
    elif walk is None or not all(hi <= walk for _lo, hi in units):
        return MEMBER_STRATA
    h = header if isinstance(header, dict) else {}
    indices = h["units"].get("indices") if isinstance(h.get("units"), dict) else None
    mine = [u for lo, hi in units for u in range(lo, hi)]
    if not (type(indices) is list and len(indices) == len(mine) and all(_nat(i) and i == u for i, u in zip(indices, mine))):
        return MEMBER_UNITS
    try:
        slice_, h = _held_at(set(reads), held or {}, slice_, h)
    except _Refused as e:
        return str(e)
    fv = h["frame_v3"] if isinstance(h.get("frame_v3"), dict) else {}
    roots, domains = fv.get("roots"), fv.get("domain_ids")
    reg = {r["port"]: r for r in slice_["roots"]}
    for p in sorted(roots) if isinstance(roots, dict) else ():
        if p not in reg:
            return unwired_why(p)
    codes = []
    if not _same(h.get("scheme"), slice_.get("scheme")):
        codes.append("scheme")
    if not _same(roots if isinstance(roots, dict) else {}, {p: r["root"] for p, r in reg.items()}):
        codes.append("roots")
    if not all(_same(domains.get(p) if isinstance(domains, dict) else None, r["domain"]) for p, r in reg.items()):
        codes.append("domains")
    if not _same(h.get("instances"), slice_["population"]):
        codes.append("population")
    return f"public file: not the registration's: {', '.join(codes)}" if codes else None


def flock_form(circuit: Path, public: Path, tables: Sequence[tuple[str, Path]]) -> str | None:
    """The Flock verifier's check of a statement's form on its files, with no session: its executable's ``ks-form`` on
    the circuit and public files with the lookup tables `tables` (``--tables``), which is `Flock.KsScope.checkOf`
    without a sample and so, at the one table of a member's session, `KsScope.filesOf`, then
    `Flock.PortRoots.headerOf`: a header naming ``public_inputs`` where the circuit's META names none, and a public
    file whose rows are not its header's (``frame_v3``'s frame, row key, schemas and roots, and the pinned dummy
    rows). None where they pass, else the refusal. `Outside` where the verifier answers neither."""
    with tempfile.TemporaryDirectory(prefix="verity-members-form-") as d:
        for name, f in tables:
            os.symlink(f, Path(d, f"{name}.u32"))
        p = LC.run([str(PF.executable()), "ks-form", "--statement", PF.TMPL_STATEMENT, "--circuit", str(circuit),
                    "--public", str(public), "--tables", d], env=PF.verifier_env(), ok_statuses=LC.KS_FORM_EXITS)
    out = p.stdout.removesuffix("\n")
    if not out.startswith("KS-FORM ") or p.returncode != (0 if out == "KS-FORM ok" else 1):
        raise Outside(f"the verifier answered no form check for the files (exit {p.returncode}): {(p.stderr or p.stdout)[-400:]}")
    return None if out == "KS-FORM ok" else out[len("KS-FORM "):]


def files_why(circuit: Path, public: Path, tables: Sequence[tuple[str, Path]], units: Sequence[Sequence[int]],
              strata: Any, slice_: Mapping[str, Any], form: Form, reads: Collection[tuple[str, str]] = (),
              held: Mapping[str, Any] | None = None, subset: bool = False) -> str | None:
    """`OneStageZk.StepAt.filesHeld` as far as this answers it (module docstring): why the files of a member of the
    units `units` are refused with no session, the first of Lean's that applies, None where none does. A circuit that
    does not parse or a public file that does not load (`form`'s refusal, which is the scope's word for word,
    `UNREAD`), the template's refusals (`proofs.template_why`), the rest of the form without a sample and the public
    file's header and rows (`form`, Lean's `ksForm` then `header`), then
    the member's population check at the docket's `strata` and the member's slice `slice_` (`population_why`).
    `Outside` where the strata or the public file's header are not read."""
    formed = form(circuit, public, tables)
    if formed is not None and formed.startswith(UNREAD):
        return formed
    if (template := PF.template_why(circuit)) is not None:
        return template
    if formed is not None:
        return formed
    if isinstance(strata, Exception):
        raise Outside(f"the registration's strata are not read ({strata}), whose refusal is the Lean parser's own")
    try:
        header = _header(public)
    except ValueError as e:
        raise Outside(f"{e}, whose refusal is the Lean parser's own") from None
    return population_why(units, strata, header, slice_, reads, held, subset)


def _files(name: str, m: Mapping[str, Any], blobs: Mapping[str, Path], descriptor: bytes, partition: Mapping[str, Any],
           strata: Any, slice_: Mapping[str, Any], form: Form,
           derived: Path, decoded: OC.Decoded | None = None,
           subset: bool = False) -> tuple[Path, Path, tuple[tuple[str, Path], ...]]:
    """`OneStageZk.Member.files`: the member `name`'s circuit file, C_S as `OneStageZk.restrict` derives it
    (`one_circuit.restrict`, then `one_circuit.lower`), written under `derived`, and public files and its own lookup
    tables, the blobs' pinned tables its circuit looks up (`proofs.own_tables`, `OneStageZk.ownTables`), read as
    `Member.statement` reads them and held with no session (`files_why`); a refusal names the member. A table its
    circuit looks up that the blobs do not hold refuses (`proofs.tables_match_why`); one only another member's circuit
    looks up is none of its."""
    try:
        try:
            decoded = decoded or OC.Decoded(descriptor)
            c = OC.lower(OC.restrict(descriptor, partition, m["units"], decoded), partition, PF.lower_executable,
                         cut=OC.cuts(descriptor, partition, decoded),
                         formed=OC.formed_in(decoded, partition, m["units"]))
        except OC.Refused as e:
            if str(e).startswith(f"{e.code}: {OC.NOT_ADMITTED}"):
                raise _Refused(str(e).removeprefix(f"{e.code}: ")) from None
            raise _Refused(f"restrict: {e}") from None
        except OC.NotDerived as e:
            raise _Refused(f"restrict: {e}") from None
        circuit = derived / f"{hashlib.sha512(c).hexdigest()}.txt"
        circuit.write_bytes(c)
        meta = _meta(circuit)
        try:
            lookups = PF.lookups_of(meta)
        except ValueError as e:
            raise _Refused(str(e)) from None
        own = PF.own_tables(lookups, blobs)
        if (why := PF.tables_match_why(own, lookups)) is not None:
            raise _Refused(why)
        tables = tuple((n, blobs[PF.MUFU_SHA512[n]]) for n in own)
        if m["registered"] is None and PF.reads_registered(descriptor):
            raise _Refused("registered: the program reads registered values (a RegisteredInput among its "
                           "definitions) and the verifier holds no registered reads: verify with its own "
                           "registered file")
        public = _blob(blobs, m["public"], "public file")
        g = meta.get("vus_per_block") if isinstance(meta, dict) else None
        if _nat(g) and g != m["vus_per_block"]:
            raise _Refused(f"vus_per_block: the member's circuit has {g} units a proof block, not the "
                           f"{m['vus_per_block']} the parameters name")
        reads, held = registered_reads(m["registered"])
        if (why := files_why(circuit, public, tables, m["units"], strata, slice_, form, reads, held, subset)) is not None:
            raise _Refused(why)
    except _Refused as e:
        raise _Refused(f"member {name}: {e}") from None
    return circuit, public, tables


def flock_verdict(st: Statement) -> dict[str, Any]:
    """The Flock verifier's verdict on the statement `st`'s one session: its executable's ``verify --zk`` on its files, as
    `proofs.verify` runs it, with the lookup tables `st` holds (``--tables``); the verdict's line. `Outside` where the
    verifier answers none."""
    with tempfile.TemporaryDirectory(prefix="verity-members-") as d:
        at = Path(d)
        (at / "s").mkdir()
        (at / "tables").mkdir()
        os.symlink(st.record, at / "s" / "session.json")
        for name, f in st.reps:
            if name not in ("", ".", "..", "session.json") and "/" not in name and "\0" not in name:
                os.symlink(f, at / "s" / name)
        for name, f in st.tables:
            os.symlink(f, at / "tables" / f"{name}.u32")
        (at / "program.json").write_bytes(st.program)
        (at / "partition.json").write_text(json.dumps(st.partition))
        reads = []
        if st.registered is not None:
            (at / "registered.json").write_text(json.dumps(st.registered))
            reads = ["--registered", str(at / "registered.json")]
        p = LC.run([str(PF.executable()), "verify", "--zk", "--statement", PF.TMPL_STATEMENT, "--circuit", str(st.circuit),
                    "--public", str(st.public), "--tables", str(at / "tables"),
                    "--partition", str(at / "partition.json"), "--program", str(at / "program.json"), *reads,
                    "--session", str(at / "s")], env=PF.verifier_env(), ok_statuses=LC.VERIFY_EXITS)
    lines = [json.loads(x[len("VERDICT "):]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
    if len(lines) != 1 or not isinstance(lines[0], dict):
        raise Outside(f"the verifier answered no verdict for the session (exit {p.returncode}): {(p.stderr or p.stdout)[-400:]}")
    if (why := PF.not_zk_why(lines[0])) is not None:
        raise Outside(why)
    return lines[0]


def _session(name: str, member: str, st: Statement, record: Any,
             verdict: Callable[[Statement], Mapping[str, Any]]) -> PF.Finding:
    """One member's session's finding (`OneStageZk.step`): the verifier's verdict, refused where it accepts naming
    another program than the partition object's. The template's and the population's refusals, which Lean's verdict
    runs again, passed on the member's files before any session (`files_why`)."""
    v = verdict(st)
    ok, why = v.get("accepted") is True, str(v.get("why", ""))
    sha, want = v.get("program_sha512"), st.partition.get("program")
    if ok and (type(sha) is not str or sha != want):
        raise _Refused(f"member {member}: session {name}'s verdict names program {sha if type(sha) is str else 'none'}, "
                       f"not the partition object's {PF.printed(want)}")
    units: tuple[int, ...] = ()
    if ok:
        try:
            indices = _header(st.public)["units"]["indices"]
            units = tuple(sorted(indices[i] for i in record["unit_draw"]["units"]))
        except (ValueError, KeyError, TypeError, IndexError):
            raise Outside(f"session {name} is accepted and its record's sample names no units of its public file") from None
    return PF.Finding(name, ok, why, sha if type(sha) is str else None, v.get("registered") or {}, units,
                      v.get("openings"))


def members_run(params: Mapping[str, Any], record: Mapping[str, Any], evidence: Mapping[str, Any],
                blobs: Mapping[str, Path], *, program: Program, verify: Ch.Verify,
                verdict: Callable[[Statement], Mapping[str, Any]] = flock_verdict,
                form: Form = flock_form) -> tuple[Run | None, str | None]:
    """`OneStageZk.membersRun` (module docstring): the per-member form's sessions' run on the parameters `params`, the
    docket `record`, the evidence `evidence` (each as its parser read it) and the files `blobs`, by SHA-512
    (`blobs_of`), under the verifier's own `program`, the one `params` names, and its challenger's signature check
    `verify`. The run and None, or None and Lean's refusal, the reason `check` answers after ``proofs:``. It is no
    verdict: a run whose sessions all accepted is what the audit above it reads, v1's check at the coins, which
    accepts or refuses the case. `verdict` is one session's verdict of the Flock verifier (`flock_verdict`), and
    `form` its check of a member's files' form with no session (`flock_form`). `Outside` on what it does not decide."""
    ms, sts = _parsed(params, evidence)
    if P.program_sha512(program) != params["partition"].get("program"):
        raise Outside("the program is not the one the parameters' partition object names")
    if DK.query_digest(params["partition"].get("query") or {}) != DK.query_digest(params["query"]):
        raise Outside(PARTITION_QUERY)
    descriptor = json.dumps(params["program"], sort_keys=True, separators=(",", ":")).encode()
    decoded = OC.Decoded(descriptor)
    ms = {name: {**m, "units": _block(name, m, descriptor, params["partition"], decoded)} for name, m in ms.items()}
    _apart(ms)
    with tempfile.TemporaryDirectory(prefix="verity-members-cs-") as derived:
        return _members_run(params, record, evidence, blobs, ms, sts, descriptor, program, verify, verdict, form,
                            Path(derived), decoded)


def _members_run(params: Mapping[str, Any], record: Mapping[str, Any], evidence: Mapping[str, Any],
                 blobs: Mapping[str, Path], ms: dict[str, Any], sts: list[Any], descriptor: bytes, program: Program,
                 verify: Ch.Verify, verdict: Callable[[Statement], Mapping[str, Any]], form: Form,
                 derived: Path, decoded: OC.Decoded | None = None) -> tuple[Run | None, str | None]:
    try:
        held = _held(ms, sts)
        if [s["name"] for st in sts for s in st["sessions"]] != evidence["sessions"]:
            raise _Refused("the statements' sessions are not the evidence's sessions, in order")
        trees, log = _trees(params, evidence, verify)
        order = PF.order_why(params, record, evidence)
        if order is not None:
            raise _Refused(order)
        _trees_held(trees, log, sts, blobs)
        if not _nat(record.get("population")):
            raise _Refused("the record names no population")
        if not P.in_order(program, params["query"], record["population"]):
            if not P.covers(program, params["query"], record["population"]):
                raise _Refused(POPULATION_ORDER)
            mine = resolved(params, program)
            if not subset_designs(mine, record):
                raise _Refused(strata_design(_kind(record.get("law") if subset_design(mine) else mine)))
        _laid(ms, record)
        _class(ms)
        _tree_ports_held(ms, record)
        rp = read_pins(params, program)
        pins_held(ms, EG._Root(params["program"]), P._template(program, params["query"], "the members' partition").ranges,
                  program, DK.marked_gates(program), rp)
        roots, ft = _members_roots(ms, record)
        roots = _wired(program, params["query"], params["program"], ms, record, roots, ft, params.get("window_root"),
                       tied_gates(program, params))
        _ports_once(ms, roots)
        _sampled(ms, held, evidence["proved_units"])
        try:
            strata: Any = P.strata(program, params["query"])
        except (P.QueryRefused, ValueError) as e:
            strata = e
        files = {name: _files(name, m, blobs, descriptor, params["partition"], strata,
                              {"population": _count(m), "roots": roots[name], "scheme": record.get("scheme")}, form,
                              derived, decoded, subset_design(resolved(params, program)))
                 for name, m in ms.items()}
        findings: list[PF.Finding] = []
        for st, name in zip(sts, held):
            (s,) = st["sessions"]
            try:
                path = _blob(blobs, s["record"], f"session {s['name']}'s record")
                _, parsed = _record(path, s["name"])
            except _Refused as e:
                raise _Refused(f"member {name}: {e}") from None
            circuit, public, tables = files[name]
            statement = Statement(circuit, public, descriptor, params["partition"], ms[name]["registered"], tables,
                                  path, tuple((f, blobs[s["reps"][f]]) for f in sorted(s["reps"])))
            findings.append(_session(s["name"], name, statement, parsed, verdict))
        proved = tuple(u for f in findings for u in f.units)
        every = all(f.accepted for f in findings)
        if every and sorted(proved) != evidence["proved_units"]:
            raise _Refused(PROVED_UNITS)
        return Run(tuple(findings), proved, every), None
    except _Refused as e:
        return None, str(e)


def run_reason(run: Run | None, refused: str | None) -> str | None:
    """The reason `check` rejects with on a case whose per-member run is `members_run`'s `(run, refused)`, where v1's
    check at the coins passes on everything else: the run's refusal under ``proofs:``, and None where the run
    returned. There `check` accepts where every session accepted (`Run.all`), and where one did not, v1's check
    answers by its verdict's name."""
    return None if run is not None else f"proofs: {refused}"


__all__ = ["EXITS", "Form", "MEMBERS_LAID", "MEMBER_COVER", "MEMBER_STRATA", "MEMBER_UNITS", "NO_CHALLENGER",
           "PARTITION_QUERY", "Outside", "POPULATION_ORDER", "PROVED_UNITS", "ROW_TABLES", "Run", "Statement", "TABLES",
           "TABLE_PORT", "UNREAD", "blobs_of", "case_check", "check", "files_why", "flock_form", "flock_verdict",
           "field_trees_why", "members_run", "population_why", "POPULATION_WHY", "registered_at", "registered_reads",
           "return_fields", "run_reason", "shared_unread", "unregistered_why", "unwired_why", "own_root_why",
           "pinned_read_why", "pins_held", "port_twice_why", "read_pins", "unpinned_read_why", "run_own_root_why", "run_tree_missing_why", "run_tree_why", "run_tree_unbound_why", "run_input_roots_at", "run_tree_held", "tied_roots_at", "tied_roots_why", "member_inputs", "reg_rows", "read_form_why", "two_reads_why", "slice_no_port_why", "strata_design", "resolved", "subset_design", "subset_designs", "wired_tree_why"]
