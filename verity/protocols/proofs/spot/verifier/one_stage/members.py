"""One-stage v2's per-member form outside Lean: the refusal's run on a case, and the Python of the sessions' run.

A window of several members is audited in v2's per-member form of ``proofs`` (the parameters' ``members``, the
evidence's ``statements``; `proofs.members_v2` writes it). `Spot.OneStageZk` parses the form and accepts nothing of it
(`checkMembersAt_refuses`). The executable's ``refusal PARAMS COMMITMENT EVIDENCE [BLOB ...]``
(`Spot/MembersRefusal.lean`) runs the audit's check on such a case and prints why it is refused: the reason of the
first check that fails, and `MULTI_REFUSED` under ``proofs:`` exactly where every check before the refusal passed
(`ALL_PASSED`). The line is a refusal's reason and no verdict (`REFUSAL`): the command never exits 0, and nothing here
reads its answer as an audit's.

`members_run` is the Python of `OneStageZk.membersRun`, the part of that check that reads the per-member form: the
statements' structure, the sessions' coins held to the trees the window's challenger session logged, each closed after
its session's evidence before the next is opened, the members laid over the registration's population, each registered
value one tree, the registration's roots each one member's, then each statement's session verified on its member's own
files. It refuses with Lean's reason where Lean refuses, and accepts nothing either: its `Run` is the sessions'
findings, which the audit above it (v1's check at the coins, then the refusal) reads. Each session's verdict is the
Flock verifier's own (`flock_verdict`: its executable's ``verify --zk``, as `proofs.verify` runs it).

Before any session it holds every member's files, sampled or not, in the members' order (`OneStageZk.membersFiles`):
the reads Lean's `Member.statement` makes of them, then what a session's verdict checks of the files alone
(`files_why`, Lean's `StepAt.filesHeld`), a refusal naming the member. A member the sample missed is refused there as
one it holds is. The form the sampled audit's knowledge soundness covers is the Flock verifier's own there too
(`flock_form`: its executable's ``ks-form``).

Where it is not Lean's step for step, each on a case neither side accepts:

* **A link's ``root_B`` length** (`challenger.LINK_ROOT_LEN`). Lean holds each session's link to the length its
  statement fixes for its table's ``root_B`` (`OneStageZk.rootBTables`, the session set up on the member's files) before
  it verifies any session. This sets up no session, so it passes no length (`TABLES`): a session Lean refuses there for
  that length alone is here a session its own verdict rejects, since the verifier holds ``root_B`` to that length.
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
* **A registered file's own form** (`Flock.Registered.ofJson`, which `classHeld` and each statement read it through).
  This reads a member's registered file by the fields `classHeld` compares, `Outside` where one is not a registered
  port's; the file's other refusals are the Flock verifier's, at the member's session, and are not answered here for a
  member the sample missed.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.circuits.program import Program
from verity.protocols.proofs.spot.device import challenger as Ch
from verity.protocols.proofs.spot.device.coins import DRAW_FORM

from . import closing as CL, partition as P, proofs as PF, registered as REG
from .proofs import V2_EXE, LeanError, v2_executable

#: what the v2 executable's ``refusal`` says of what it prints (`OneStageZk.REFUSAL`)
REFUSAL = "the per-member form's refusal, no verdict"
#: `OneStageZk.MULTI_REFUSED`: the refusal of the per-member form's acceptance
MULTI_REFUSED = ("the per-member form (proofs.members) is refused until the guarantee for a sample over several members' "
                 "statements under one draw is in the lock: verify one statement (proofs {circuit, public, registered})")
#: `refusal`'s reason exactly where the sessions' verdicts and v1's whole check at the coins passed
ALL_PASSED = f"proofs: {MULTI_REFUSED}"


def refusal(params: Mapping[str, Any], commitment: bytes, evidence: bytes, blobs: Sequence[Path] = ()) -> tuple[str, bool]:
    """Why v2 refuses the case of parameters `params`, registration record `commitment` and evidence `evidence` (each
    the bytes the audit reads) on the files `blobs` (`V2_EXE`'s ``refusal``, `OneStageZk.membersOutcome`): the reason,
    and whether the check decided (False where it is undecided). The reason is `ALL_PASSED` exactly where every check
    before the per-member form's refusal passed. A case in the one statement's form is refused unchecked. Raises
    `LeanError` on an answer that is not a refusal's line."""
    with tempfile.TemporaryDirectory(prefix="verity-members-refusal-") as d:
        paths = [Path(d, n) for n in ("params.json", "commitment.bin", "evidence.bin")]
        canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
        for path, b in zip(paths, (canonical, commitment, evidence)):
            path.write_bytes(b)
        p = subprocess.run([str(v2_executable()), "refusal", *map(str, paths), *map(str, blobs)], capture_output=True,
                           text=True)
    try:
        got = json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise LeanError(f"{V2_EXE} refusal answered no line (exit {p.returncode}): {p.stderr[-400:]}") from None
    if (not isinstance(got, dict) or sorted(got) != ["decided", "refusal", "run"] or got["run"] != REFUSAL
            or type(got["refusal"]) is not str or type(got["decided"]) is not bool
            or p.returncode != (1 if got["decided"] else 2)):
        raise LeanError(f"{V2_EXE} refusal answered a line that is not a refusal's (exit {p.returncode}): {got!r}"[:600])
    return got["refusal"], got["decided"]


def case_refusal(case: Mapping[str, Any]) -> tuple[str, bool]:
    """`refusal` on a case as a driver writes one: ``{"blobs": [path, ...], "commitment": hex, "evidence": hex,
    "params": {...}}``."""
    return refusal(case["params"], bytes.fromhex(case["commitment"]), bytes.fromhex(case["evidence"]),
                   [Path(b) for b in case["blobs"]])


#: `OneStageZk.sessionTables`, the tables of a member's session, each with no ``root_B`` length (module docstring)
TABLES: dict[str, int | None] = {"circuit": None}
#: `OneStageZk.NO_CHALLENGER`
NO_CHALLENGER = ("the parameters name no challenger: the draw's coins are the auditor's challenger's signed release "
                 "(challenger, with window_root), never the prover's record")
#: `OneStageZk.POPULATION_ORDER`
POPULATION_ORDER = ("population-order: the program's proof blocks under the query are not the registration's "
                    "population units in order")
#: `OneStageZk.MEMBERS_LAID`
MEMBERS_LAID = ("members: the members' units are not the registration's population, each member's one range of it: a "
                "member is the units [base, base + n), and the members' counts sum to the population")
#: `OneStageZk.TABLE_PORT`: the port a registration names a window's row tables' root at
TABLE_PORT = "row-table"
#: `OneStageZk.ROW_TABLES`
ROW_TABLES = ("roots: the registration names a row tables' root (row-table), the window's and no member's: the "
              "per-member form is refused on a window with row tables until its guarantee covers a root every member "
              "reads")
#: `OneStageZk.MEMBER_STRATA`
MEMBER_STRATA = ("partition: the registration's strata, read stratum by stratum, are not its units in order, or a unit "
                 "of the member's is past them: a member's strata are the registration's, each restricted to the "
                 "member's units")
#: `OneStageZk.MEMBER_UNITS`
MEMBER_UNITS = ("units: the public file's units.indices are not exactly the member's units, in order: slot i of its "
                "table is the member's i-th unit")
#: `OneStageZk.provedHeld`'s refusal
PROVED_UNITS = "the units the sessions proved are not the evidence's proved_units"
#: the heads of the two refusals `Flock.KsScope.checkOf` and the scope (`Flock.ProvedScope.checkOf`) word alike, of
#: files neither reads: a circuit that does not parse, a public file that does not load
UNREAD = ("refused: a circuit that does not parse (", "refused: a public file that does not load (")
#: a statement's form held on its circuit and public files and the lookup tables, by name, with no session: None, or
#: the refusal (`flock_form`)
Form = Callable[[Path, Path, Sequence[tuple[str, Path]]], "str | None"]
_HEX = frozenset("0123456789abcdef")


class Outside(ValueError):
    """An input `members_run` does not decide: one the per-member parsers refuse, a registered file or a verifier's
    answer it cannot read. Never a pass."""


class _Refused(Exception):
    """`membersRun`'s refusal, with Lean's reason."""


@dataclass(frozen=True)
class Statement:
    """What one member's session is verified on, each file a blob's: the member's circuit and public files, the
    verifier's program (its descriptor's bytes), partition object and registered file for the member, the pinned lookup
    tables the blobs hold, by name, and the session's record and proof files, by name."""

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


def _parsed(params: Mapping[str, Any], evidence: Mapping[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    """The parameters' members and the evidence's statements, as the per-member parsers leave them; `Outside` where
    they are not in that form."""
    form = Outside("the parameters and the evidence are not in the per-member form as its parsers read it "
                   "(`OneStageZk.membersParamsOfJson`, `membersEvidenceOf`)")
    try:
        ms, sts = params["proofs"]["members"], evidence["proofs"]["statements"]
        ok = (set(params["proofs"]) == {"members"} and set(evidence["proofs"]) == {"statements"} and isinstance(ms, dict)
              and bool(ms) and type(sts) is list and type(evidence["sessions"]) is list
              and type(evidence["proved_units"]) is list and isinstance(evidence["coins"], dict)
              and isinstance(params["challenger"], dict) and isinstance(params["partition"], dict)
              and isinstance(params["program"], dict))
        for m in ms.values() if ok else ():
            ok = ok and (set(m) == {"circuit", "public", "registered", "units", "vus_per_block"}
                         and type(m["circuit"]) is str and type(m["public"]) is str
                         and (m["registered"] is None or isinstance(m["registered"], dict))
                         and type(m["units"]) is list and bool(m["units"])
                         and all(type(u) is list and len(u) == 2 and _nat(u[0]) and _nat(u[1]) and u[0] < u[1]
                                 for u in m["units"])
                         and _nat(m["vus_per_block"]) and m["vus_per_block"] > 0)
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


def _ports(name: str, registered: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A registered file's ports by the fields `classHeld` compares, in the file's order; `Outside` where one is not
    read."""
    ports = registered.get("ports")
    if not (isinstance(ports, dict) and ports):
        raise Outside(f"member {name}: its registered file names no ports")
    out = []
    for port in sorted(ports):
        e = ports[port]
        q = {"value": e.get("value"), "root": e.get("root"), "leaves": e.get("leaves"), "words": e.get("words"),
             "word_bits": e.get("word_bits", REG.WORD_BITS), "domain": e.get("domain"),
             "row_schema": e.get(REG.SCHEMA_KEY, REG.SCHEMA)} if isinstance(e, dict) else None
        if not (q is not None and type(q["value"]) is str
                and all(type(q[k]) is str and len(q[k]) == 128 and set(q[k]) <= _HEX for k in ("root", "domain"))
                and _nat(q["leaves"]) and q["leaves"] > 0 and (q["words"] is None or (_nat(q["words"]) and q["words"] > 0))
                and q["word_bits"] in (16, 32) and q["row_schema"] in REG.SCHEMAS):
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


def _roots(ms: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """`OneStageZk.rootsOf`: the registration's roots by the member each is of, its port without the member's name."""
    roots = record.get("roots")
    if type(roots) is not list:
        raise _Refused("roots: the registration's roots are not a list")
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in ms}
    for r in roots:
        port = r.get("port") if isinstance(r, dict) else None
        if type(port) is not str:
            raise _Refused("roots: a root of the registration names no port")
        if port == TABLE_PORT:
            raise _Refused(ROW_TABLES)
        mine = [(port[:i], port[i + 1:]) for i, c in enumerate(port) if c == "/" and port[:i] in out]
        if not mine:
            raise _Refused(f"roots: the registration's root {port} is no member's: a member's roots are named "
                           "<member>/<port>")
        if len(mine) > 1:
            raise _Refused(f"roots: the registration's root {port} is two members': a member's roots are named "
                           "<member>/<port>, and a root is one member's")
        name, bare = mine[0]
        out[name].append({"domain": r.get("domain"), "leaves": r.get("leaves"), "port": bare, "root": r.get("root")})
    return out


def _slice(m: Mapping[str, Any], roots: Sequence[Mapping[str, Any]], record: Mapping[str, Any]) -> dict[str, Any]:
    """`OneStageZk.Member.slice`: the member's count of units, its roots of the registration and its scheme."""
    return {"population": _count(m), "roots": list(roots), "scheme": record.get("scheme")}


def slices(members: Mapping[str, Sequence[Sequence[int]]], record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Each member's slice of the registration `record` (`OneStageZk.Member.slice`), for the members `members` by name,
    each its units as ranges ``[lo, hi)``: ``{name: {population, roots, scheme}}``, the member's count of units, the
    registration's roots that are its own, its name taken off each port (`OneStageZk.rootsOf`), and the registration's
    scheme. `population_why` holds a member's public file to its slice. `ValueError` with Lean's reason where the
    members are not laid over the registration's population (`OneStageZk.laidHeld`) or a root of the registration is
    not one member's: an audit outside the per-member form's run reads a member's slice here, never by a rule of its
    own."""
    ms = {name: {"units": [list(r) for r in units]} for name, units in members.items()}
    try:
        _laid(ms, record)
        roots = _roots(ms, record)
    except _Refused as e:
        raise ValueError(str(e)) from None
    return {name: _slice(m, roots[name], record) for name, m in ms.items()}


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


def population_why(units: Sequence[Sequence[int]], strata: Sequence[tuple[str, Sequence[Sequence[int]]]], header: Any,
                   slice_: Mapping[str, Any]) -> str | None:
    """`OneStageZk.Member.population` past its files' reads: why a member of the units `units` (ranges ``[lo, hi)``) is
    not what its public file's header `header` holds at the registration's `strata` and the member's slice `slice_` of
    the record (``{population, roots, scheme}``); None when it is. The registration's strata read stratum by stratum
    are its units in order and hold the member's (`MEMBER_STRATA`), the header's ``units.indices`` are exactly the
    member's units in order (`MEMBER_UNITS`), and its scheme, ``frame_v3`` roots, their ``domain_ids`` and its
    ``instances`` are the slice's (`Record.matchesPublic`'s codes)."""
    at = 0
    for _t, rs in strata:
        for lo, hi in rs:
            if lo != at or hi < lo:
                return MEMBER_STRATA
            at = hi
    if not all(hi <= at for _lo, hi in units):
        return MEMBER_STRATA
    h = header if isinstance(header, dict) else {}
    indices = h["units"].get("indices") if isinstance(h.get("units"), dict) else None
    mine = [u for lo, hi in units for u in range(lo, hi)]
    if not (type(indices) is list and len(indices) == len(mine) and all(_nat(i) and i == u for i, u in zip(indices, mine))):
        return MEMBER_UNITS
    fv = h["frame_v3"] if isinstance(h.get("frame_v3"), dict) else {}
    roots, domains = fv.get("roots"), fv.get("domain_ids")
    reg = {r["port"]: r for r in slice_["roots"]}
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
    without a sample and so, at the one table of a member's session, `KsScope.filesOf`. None where they pass, else the
    refusal. `Outside` where the verifier answers neither."""
    with tempfile.TemporaryDirectory(prefix="verity-members-form-") as d:
        for name, f in tables:
            os.symlink(f, Path(d, f"{name}.u32"))
        p = subprocess.run([str(PF.executable()), "ks-form", "--statement", PF.STATEMENT, "--circuit", str(circuit),
                            "--public", str(public), "--tables", d], capture_output=True, text=True,
                           env=PF.verifier_env())
    out = p.stdout.removesuffix("\n")
    if not out.startswith("KS-FORM ") or p.returncode != (0 if out == "KS-FORM ok" else 1):
        raise Outside(f"the verifier answered no form check for the files (exit {p.returncode}): {(p.stderr or p.stdout)[-400:]}")
    return None if out == "KS-FORM ok" else out[len("KS-FORM "):]


def files_why(circuit: Path, public: Path, tables: Sequence[tuple[str, Path]], units: Sequence[Sequence[int]],
              strata: Any, slice_: Mapping[str, Any], form: Form) -> str | None:
    """`OneStageZk.StepAt.filesHeld` as far as this answers it (module docstring): why the files of a member of the
    units `units` are refused with no session, the first of Lean's that applies, None where none does. A circuit that
    does not parse or a public file that does not load (`form`'s refusal, which is the scope's word for word,
    `UNREAD`), the template's refusals (`proofs.template_why`), the rest of the form without a sample (`form`), then
    the member's population check at the registration's `strata` and the member's slice `slice_` (`population_why`).
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
    return population_why(units, strata, header, slice_)


def _files(name: str, m: Mapping[str, Any], blobs: Mapping[str, Path], descriptor: bytes,
           tables: Sequence[tuple[str, Path]], strata: Any, slice_: Mapping[str, Any], form: Form) -> tuple[Path, Path]:
    """`OneStageZk.Member.files`: the member `name`'s circuit and public files, read as `Member.statement` reads them
    and held with no session (`files_why`); a refusal names the member."""
    try:
        circuit = _blob(blobs, m["circuit"], "circuit")
        meta = _meta(circuit)
        try:
            lookups = PF.lookups_of(meta)
        except ValueError as e:
            raise _Refused(str(e)) from None
        if (why := PF.tables_why(lookups, blobs)) is not None:
            raise _Refused(why)
        if m["registered"] is None and PF.reads_registered(descriptor):
            raise _Refused("registered: the program reads registered values (a RegisteredInput among its "
                           "definitions) and the verifier holds no registered reads: verify with its own "
                           "registered file")
        public = _blob(blobs, m["public"], "public file")
        g = meta.get("vus_per_block") if isinstance(meta, dict) else None
        if _nat(g) and g != m["vus_per_block"]:
            raise _Refused(f"vus_per_block: the member's circuit has {g} units a proof block, not the "
                           f"{m['vus_per_block']} the parameters name")
        if (why := files_why(circuit, public, tables, m["units"], strata, slice_, form)) is not None:
            raise _Refused(why)
    except _Refused as e:
        raise _Refused(f"member {name}: {e}") from None
    return circuit, public


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
        p = subprocess.run([str(PF.executable()), "verify", "--zk", "--statement", PF.STATEMENT, "--circuit", str(st.circuit),
                            "--public", str(st.public), "--tables", str(at / "tables"),
                            "--partition", str(at / "partition.json"), "--program", str(at / "program.json"), *reads,
                            "--session", str(at / "s")], capture_output=True, text=True, env=PF.verifier_env())
    lines = [json.loads(x[len("VERDICT "):]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
    if len(lines) != 1 or not isinstance(lines[0], dict):
        raise Outside(f"the verifier answered no verdict for the session (exit {p.returncode}): {(p.stderr or p.stdout)[-400:]}")
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
    registration record `record`, the evidence `evidence` (each as its parser read it) and the files `blobs`, by SHA-512
    (`blobs_of`), under the verifier's own `program`, the one `params` names, and its challenger's signature check
    `verify`. The run and None, or None and Lean's refusal, the reason `membersOutcome` prints after ``proofs:``. It
    accepts nothing: a run whose sessions all accepted is what the audit above it reads, v1's check at the coins and then
    the form's refusal (`ALL_PASSED`). `verdict` is one session's verdict of the Flock verifier (`flock_verdict`), and
    `form` its check of a member's files' form with no session (`flock_form`). `Outside` on what it does not decide."""
    ms, sts = _parsed(params, evidence)
    if P.program_sha512(program) != params["partition"].get("program"):
        raise Outside("the program is not the one the parameters' partition object names")
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
            raise _Refused(POPULATION_ORDER)
        _laid(ms, record)
        _class(ms)
        roots = _roots(ms, record)
        try:
            strata: Any = P.strata(program, params["query"])
        except (P.QueryRefused, ValueError) as e:
            strata = e
        descriptor = json.dumps(params["program"], sort_keys=True, separators=(",", ":")).encode()
        tables = tuple((n, blobs[d]) for n, d in PF.MUFU_SHA512.items() if d in blobs)
        files = {name: _files(name, m, blobs, descriptor, tables, strata, _slice(m, roots[name], record), form)
                 for name, m in ms.items()}
        findings: list[PF.Finding] = []
        for st, name in zip(sts, held):
            (s,) = st["sessions"]
            try:
                path = _blob(blobs, s["record"], f"session {s['name']}'s record")
                _, parsed = _record(path, s["name"])
            except _Refused as e:
                raise _Refused(f"member {name}: {e}") from None
            statement = Statement(*files[name], descriptor, params["partition"], ms[name]["registered"], tables, path,
                                  tuple((f, blobs[s["reps"][f]]) for f in sorted(s["reps"])))
            findings.append(_session(s["name"], name, statement, parsed, verdict))
        proved = tuple(u for f in findings for u in f.units)
        every = all(f.accepted for f in findings)
        if every and sorted(proved) != evidence["proved_units"]:
            raise _Refused(PROVED_UNITS)
        return Run(tuple(findings), proved, every), None
    except _Refused as e:
        return None, str(e)


def run_reason(run: Run | None, refused: str | None) -> str | None:
    """What `refusal` answers on a case whose per-member run is `members_run`'s `(run, refused)`, where v1's check at
    the coins passes on everything else: the run's refusal under ``proofs:``; `ALL_PASSED` where every session
    accepted; and None where a session did not, which v1's check answers by its verdict's name."""
    if run is None:
        return f"proofs: {refused}"
    return ALL_PASSED if run.all else None


__all__ = ["ALL_PASSED", "Form", "MEMBERS_LAID", "MEMBER_STRATA", "MEMBER_UNITS", "MULTI_REFUSED", "NO_CHALLENGER",
           "Outside", "POPULATION_ORDER", "PROVED_UNITS", "REFUSAL", "ROW_TABLES", "Run", "Statement", "TABLES",
           "TABLE_PORT", "UNREAD", "blobs_of", "case_refusal", "files_why", "flock_form", "flock_verdict", "members_run",
           "population_why", "refusal", "run_reason"]
