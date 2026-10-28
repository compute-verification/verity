"""The network certifier's period check (`NetworkCheck.Period`, `verity/certifier/v0`) in Python, for every difftest
that runs it: its mirror, `answer` (as `reference`), and its cases, `build` and `case`.

`answer(case)` is the outcome of one `serve` line in the core interface's answer format
(`verity_difftest.protocol.answer`): it decodes the protocol's formats (`NetworkCheck/Period.lean`'s module docstring)
and checks a period with the Python the period's pieces already have: one-stage's answers and integrity profiles as PoUW
reads them (`pouw.core_check.answer_of` and `integrity`), the network certifier's certificates and clock certificates
under its LMS key (`certifier.certificate`), Σ_sync (`certifier.schedule`), `registration.digest`, and in the rows form
the schedule (`warden.WardenParams`) and the verifier's read of each row registration
(`warden_rows.check_registration`), and the scan's terms read off its Program's descriptor (`_scan`), as each compute
audit's units and their committed bits are read off its own (`_audit`). Its verdict,
public, profile and digest are Lean's exactly; its reason is its own. Public is `K` and its terms over
`PeriodLeakAccepted`'s windows (the clock form's audits, the rows form's scans, all at one δ), each with its `delta`,
and `eps`, the union of those deltas as an exact ratio string.

`build(m, seed, small)` makes a period with one change `m` (`EXPECT`, each aimed at one path of the check, or malformed
parameters, commitments, evidence and cases). In the clock form: the link-windows of one link from the session's window
`first`, each registered under the link's Program with its provenance, certified at both ends by a small LMS key in a
drawn network certifier session, its clock certificate signing a sync of Σ_sync salted (`clock_commitment`, the sync and
the salt in no case), and answered by one-stage with its one unit proved. In the rows form (a change named `rows-…`):
each link-window's ingress and egress rows committed and registered under a small drawn schedule
(`warden_rows.Cells.commit` and `registration`, signed by the same key), its three records certified (`past-slots`:
none of them and neither registration, as the warden leaves a window in which it refused a connection), and the scan's
registration holding both rows' roots and domains and its `prog` (`PROG`) at the root and domain of an output port's
entry in an audit's registration (`_prog_tied`), answered by one-stage with a stratified or subset draw at the
period's δ; the scan's Program is its generator form, stand-ins of its three templates at small widths in the Boolean
profile's types, as the scan's Definitions have them (`scan_descriptor`), and a Program in the checker form is a bad
scan (`bad_scans`), as is each check's negative of what `NetTiming.Layout.Fits` asks (`scan_negatives`). Beside them,
the compute audits (in the rows form at least one, whose registrations' `roots` hold the scans' `prog`), each
registered over its Program, a population of one or two stand-in templates whose units commit from 8 to 320 bits
(`audit_descriptor`), and answered with a subset or stratified draw (a work law's) and its bound on wrong units, every
unit drawn for an audit a scan's `prog` is tied to (`tied-undrawn`: one such audit drawn `subset:k`, k < N).
`case(kind, seed, form)` is a small period of the form aimed at a verdict, or at a decoding failure (`KINDS`), and
`cases(seed, form)` one of each with its answer, as a composition's difftest takes a real component's cases.

This is test-side Python (it imports `verity_difftest` and PoUW's `core_check`, which the certifier's package may not),
so it lives beside the period's difftest (`test_network_period_protocol.py`), and other suites load it by path."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import re
from fractions import Fraction
from functools import lru_cache

import verity_difftest as D
from verity_difftest import protocol as P
from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.certificate import (SALT_BYTES, Certificate, Direction, LmsKey,
                                                            certificate_digest, clock_digest)
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.certifier.warden import WardenParams
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.sampled_proofs.one_stage.draw import stratum_k
from verity.ml.circuits import boolean as BL, codec, metadata as AN, population
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.commitments.identity import tagged_sha512
from verity.ml.crypto import lms

ID = "verity/certifier/v0"
COMMITMENT_TAG = "verity-network-certifier/period/v0"
ONE_STAGE = "verity/sampled-proofs/one-stage/v1"
FORMATS = ("verity/registration/v1", "verity/registration/v2")
#: a proof unit's public outputs are at most 16 bits (the values it commits can be wider)
OUT_BITS = 16
UNIT = "unit"
EGRESS, INGRESS = Direction.EGRESS, Direction.INGRESS
ACCEPT, REJECT, UNDECIDED = "accept", "reject", "undecided"
PARAMS_KEYS = ["audits", "cells", "first", "jitter", "key", "link", "lms", "program", "session", "sync_set", "windows"]
ROWS_PARAMS_KEYS = ["audits", "cells", "first", "jitter", "key", "link", "lms", "program", "scan", "schedule",
                    "session", "windows"]
SCHEDULE_KEYS = sorted(f.name for f in dataclasses.fields(WardenParams))
CERTIFICATE_KEYS = ["digest", "direction", "key_id", "link", "signature", "window"]
CLOCK_WINDOW_KEYS = ["answer", "certificates", "clock"]
ROWS_WINDOW_KEYS = ["answer", "certificates", "registrations"]
ROWS_KEYS = ["cell_bytes", "certificate", "direction", "domain", "format", "layout", "link", "parameters", "root",
             "row_schema", "rows", "value", "window", "wiring", "word_bits", "words"]
LAYOUT_KEYS = ["buckets", "connection_slots", "first_bucket", "slots"]
DIGEST = re.compile("[0-9a-f]{128}")
U32 = 1 << 32
FORMS = ("clock", "rows")
SCHEMA = "verity-ir/descriptor/v1"
INPUT_ID = re.compile(r"^(?:Registered)?Input([1-9][0-9]*)_v1$")
#: the scan's templates' parameters in the generator form: bucket 0's (`ingress_scan.IngressScanFirst`), each later
#: ingress bucket's (`ingress_scan.IngressScanBucket`) and the egress bucket's (`egress.EgressBucket`)
FIRST_PARAMS = ("row", "prog")
INGRESS_PARAMS = ("row", "sin", "prog")
EGRESS_PARAMS = ("state",)
#: a byte and a u16 in the Boolean profile, as the scan's Definitions type a row's bytes and the egress bucket's word
BOOL_BYTE, BOOL_U16 = ["Array", 8, ["Value", 1]], ["Array", OUT_BITS, ["Value", 1]]
#: the checker form's, which took the rows its units compared as inputs
CHECKER_PARAMS = (("row", "sin", "prog", "sout", "view"), ("state", "row"))
#: the closure digests (`slot_entry_digest`) of `io_shell.SlotEntry_v1` at its circuit-check rows,
#: `SlotEntry_v1{KO=2,IDB=17,NP=4,STOP=[]}` and `SlotEntry_v1{KO=64,IDB=17,NP=1024,STOP=[128001]}`, in that order, which
#: `test_slot_entry_pins.py` recomputes from `io_shell`
SLOT_ENTRY_PINS = ("0763102b5fd38583279471486ff8ce51b9caae1bc5dece3d7712ddcec456f1bb"
                   "4e543343ed1399c0665043402d7100173a270cb6a8cc541b60bcdd29b37d7d0a",
                   "662360bfd2b31949f09447a32dec775c3e2e26e2d9132af8898609475e75a753"
                   "4cb522ce4f794bc5fcce71a7f09ec5a2170e0b2406339cc4d9268ea8487e3361")

# --- the mirror --------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Audit:
    """`Audit`: a compute audit's layout terms (its free inputs and monitoring period) and what its Program gives: its
    SHA-512, its units (its root's template instances) and the most bits one of them commits."""
    free: int
    steps: int
    program: str
    units: int
    bits: int


@dataclasses.dataclass(frozen=True)
class Params:
    audits: tuple[Audit, ...]
    cells: tuple[int, ...]
    first: int
    jitter: int
    link: str
    key: LmsKey
    session: bytes
    program: str
    #: the clock form's Σ_sync; None in the rows form
    sync_set: SyncSet | None
    windows: int
    #: the rows form's schedule and the scan's terms, read off its Program; None in the clock form
    schedule: WardenParams | None = None
    scan: Scan | None = None


@dataclasses.dataclass(frozen=True)
class Scan:
    """`Scan`: the scan's monitoring period (T), its own free inputs (none: its `prog` is a compute audit's output,
    `_prog_tied`, whose free inputs a link-window's term counts, `_tied_audits`), and the bits one ingress and one
    egress instance commit."""
    steps: int
    free: int
    ingress: int
    egress: int


@dataclasses.dataclass(frozen=True)
class Registration:
    record: dict
    digest: bytes


@dataclasses.dataclass(frozen=True)
class Commitment:
    audits: list[Registration]
    windows: list[Registration]
    bytes: bytes


@dataclasses.dataclass(frozen=True)
class Window:
    answer: dict
    certificates: list[Certificate]
    #: the clock form's clock certificate, or the rows form's registrations
    clock: Certificate | None
    registrations: list[dict] | None = None


@dataclasses.dataclass(frozen=True)
class Facts:
    units: int
    drawn: int
    delta: Fraction
    wrong: int


def _exactly(j, keys: list[str], ctx: str) -> None:
    if not isinstance(j, dict):
        raise D.Reject(f"{ctx}: not an object")
    if sorted(j) != keys:
        raise D.Reject(f"{ctx}: the keys must be {keys}, not {sorted(j)}")


def _nat(v, ctx: str) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise D.Reject(f"{ctx}: not a natural number")
    return v


def _of(v, kind, ctx: str):
    if not isinstance(v, kind) or isinstance(v, bool) != (kind is bool):
        raise D.Reject(f"{ctx}: not a {kind.__name__}")
    return v


def _rho(j, ctx: str) -> Fraction:
    if not isinstance(j, list) or len(j) != 2:
        raise D.Reject(f"{ctx}: not [numerator, denominator]")
    n, d = (_nat(x, ctx) for x in j)
    if n < 1 or d < 1:
        raise D.Reject(f"{ctx}: a numerator and a denominator of at least 1")
    return Fraction(n, d)


def _sync_set(j) -> SyncSet:
    _exactly(j, ["deltas", "rhos"], "sync_set")
    deltas = [_nat(x, "sync_set.deltas") for x in _of(j["deltas"], list, "sync_set.deltas")]
    rhos = [_rho(x, "sync_set.rhos") for x in _of(j["rhos"], list, "sync_set.rhos")]
    if not deltas or not rhos or len(set(deltas)) < len(deltas) or len(set(rhos)) < len(rhos):
        raise D.Reject("sync_set: non-empty, each delta and each rho listed once")
    return SyncSet(tuple(deltas), tuple(rhos))


def _schedule(j) -> WardenParams:
    """`scheduleOf`: `WardenParams`' fields, each a positive integer, B past a cell's header, and B, S and T u32s."""
    _exactly(j, SCHEDULE_KEYS, "schedule")
    values = {k: _nat(j[k], "schedule") for k in SCHEDULE_KEYS}
    try:
        s = WardenParams(**values)
    except ValueError as e:
        raise D.Reject(f"schedule: {e}") from None
    if not (s.cell_bytes < U32 and s.lanes * s.slots < U32 and s.window < U32):
        raise D.Reject("schedule: B, S = lanes · slots and T = window are u32s")
    return s


def bits_of(t) -> int:
    """`bitsOf`: a descriptor type's leaves' widths summed."""
    kind = t[0]
    if kind == "Value":
        return t[1]
    if kind == "Array":
        return t[1] * bits_of(t[2])
    return sum(bits_of(f if kind == "Tuple" else f[1]) for f in t[1])


def program_sha512(desc: dict) -> str:
    """`partition_object.program_sha512`: SHA-512 of the canonical descriptor without `annotations`."""
    return hashlib.sha512(D.canonical({k: v for k, v in desc.items() if k != "annotations"}).encode()).hexdigest()


def _members(x: dict) -> int:
    """`Node.members`: a batch's members or a scan's iterations, else one."""
    return x["n"] if x["form"] in ("batch", "scan") else 1


def leaves_of(t) -> int:
    """`Ty.leaves`: a descriptor type's leaves."""
    kind = t[0]
    if kind == "Value":
        return 1
    if kind == "Array":
        return t[1] * leaves_of(t[2])
    return sum(leaves_of(f if kind == "Tuple" else f[1]) for f in t[1])


def _parts(refs: dict, nodes: list) -> list[dict]:
    """`Flock.Program.parseParts`'s parts of a sequence: a concatenation's spliced in, an alias the parts of the earlier
    node's argument it names."""
    if "concat" in refs:
        return [q for r in refs["concat"] for q in _parts(r, nodes)]
    if "alias" in refs:
        k, x = refs["alias"]
        return _parts(nodes[k]["args"][x]["refs"], nodes)
    return [refs]


def _affine(q: dict) -> tuple | None:
    """An affine part as the decoder reads it, `(space, index, base, count)`, a part of no references at index 0 and in
    the nodes' space unless it names the parameters'."""
    if "affine" not in q:
        return None
    s, i, b, c = q["affine"]
    return (s, i, b, c) if c else ("p" if s == "p" else "n", 0, 0, 0)


def _part_names(q: dict) -> list[tuple[str, int]]:
    """`partNames`: the collections a part names, its space and index, or each item's of an explicit part."""
    if "explicit" in q:
        return [(s, i) for s, i, _ in q["explicit"]]
    a = q.get("affine") or q["strided"]
    return [("p" if a[0] == "p" else "n", a[1] if a[-1] else 0)]


def _own_returns(j: dict):
    """`ownReturns`: whether every leaf a composite returns is a gate it computes, never a parameter's leaf passed
    through: each part of its return names nodes of its body, none a scan, each calling a primitive or a composite of
    which the same holds."""
    defs, callees, memo = j["definitions"], j["callees"], {}

    def computed(nodes: list, s: str, k: int) -> bool:
        if s != "n" or k >= len(nodes) or nodes[k]["form"] == "scan":
            return False
        f = callees[nodes[k]["fn"]]
        return defs[f]["form"] == "primitive" or own(f)

    def own(fid: str) -> bool:
        if fid not in memo:
            body = defs[fid]["body"]
            memo[fid] = all(computed(body["nodes"], s, k)
                            for q in _parts(body["ret"], body["nodes"]) for s, k in _part_names(q))
        return memo[fid]
    return own


def _input_arg(arg: dict, nodes: list, j: dict) -> int | None:
    """`inputArg`: the root node an argument reads when it is that input node's leaves in order from its first."""
    ps = _parts(arg["refs"], nodes)
    a = _affine(ps[0]) if len(ps) == 1 else None
    if a is None or a[0] != "n" or a[2] != 0 or a[1] >= len(nodes):
        return None
    fid = j["callees"][nodes[a[1]]["fn"]]
    return a[1] if j["definitions"][fid]["form"] == "primitive" and INPUT_ID.match(fid) else None


#: `Flock.Steps.FUEL`: the most steps `Prog.gate`'s walk to an operand takes
FUEL = 2 ** 62


def _walkable(j: dict) -> bool:
    """`NciCheck.Desc.walkable` as C-Flock's decoder gives a Program (its layout, callees first and the root without
    parameters): `(A + 1) · (H + 2)` at most half of `FUEL`, for the root's activations `A` (its own and each composite
    node member's, multiplied through batches and scans) and the depth `H` they nest to, and no scan of no iterations
    with output leaves. `rootLaidOut` and `inputsAtRoot` the decoder gives too."""
    defs, callees, memo = j["definitions"], j["callees"], {}

    def of(fid: str) -> tuple[int, int]:
        if defs[fid]["form"] == "primitive":
            return 0, 0
        if fid not in memo:
            subs = [(_members(n), *of(callees[n["fn"]])) for n in defs[fid]["body"]["nodes"]]
            memo[fid] = (1 + sum(m * a for m, a, _ in subs), 1 + max((h for _, _, h in subs), default=0))
        return memo[fid]
    acts, depth = of(j["root"])
    empty = any(n["form"] == "scan" and n["n"] == 0 and leaves_of(j["types"][n["out"]]) > 0
                for d in defs.values() if d["form"] != "primitive" for n in d["body"]["nodes"])
    return (acts + 1) * (depth + 2) <= FUEL // 2 and not empty


def _topological(j: dict) -> bool:
    """`Prog.orderViolation` is none: each argument of a body the root reaches names only the body's parameters and
    its earlier nodes (an affine or strided part of no leaves names nothing; the returns aren't read)."""
    defs, callees = j["definitions"], j["callees"]
    seen, todo = set(), [j["root"]]
    while todo:
        fid = todo.pop()
        if fid in seen or defs[fid]["form"] == "primitive":
            continue
        seen.add(fid)
        body = defs[fid]["body"]
        for k, x in enumerate(body["nodes"]):
            for a in x.get("args", ()):
                for q in _parts(a["refs"], body["nodes"]):
                    r = q.get("affine") or q.get("strided")
                    names = [(s, i) for s, i, _ in q["explicit"]] if "explicit" in q else [r[:2]] if r[-1] else []
                    if any(not i < (len(body["params"]) if s == "p" else k) for s, i in names):
                        return False
            todo.append(callees[x["fn"]])
    return True


def slot_entry_digest(j: dict, key: str) -> str:
    """The SHA-512, in hex, of the canonical JSON object that maps each Definition `key` reaches (itself, its nodes'
    callees, theirs, …) to its `annotations.definition_digest`: the template's whole body."""
    defs, callees, seen, todo = j["definitions"], j["callees"], {}, [key]
    while todo:
        k = todo.pop()
        if k not in seen:
            seen[k] = AN.definition_digest(j, k)
            if defs[k]["form"] == "composite":
                todo += [callees[x["fn"]] for x in defs[k]["body"]["nodes"]]
    return hashlib.sha512(codec.canonical_json(seen)).hexdigest()


def _audit(a, ctx: str) -> Audit:
    """`auditOf`: the layout terms and the Program's descriptor, whose root's template instances are the audit's units,
    each committing its template's return; a root with no template refused. It reads the descriptor's root nodes and
    signatures; C-Flock's decoder, which Lean runs on it, refuses no descriptor this module writes."""
    _exactly(a, ["free", "program", "steps"], ctx)
    free, steps, j = _nat(a["free"], ctx), _nat(a["steps"], ctx), a["program"]
    if not isinstance(j, dict) or j.get("schema") != SCHEMA:
        raise D.Reject(f"{ctx}: program: not a Program's descriptor")
    defs, callees = j["definitions"], j["callees"]
    units = [(_members(x), bits_of(d["body"]["ret_type"])) for x in defs[j["root"]]["body"]["nodes"]
             if (d := defs[callees[x["fn"]]])["form"] != "primitive"]
    if not units:
        raise D.Reject(f"{ctx}: program: no template")
    return Audit(free, steps, program_sha512(j), sum(n for n, _ in units), max(b for _, b in units))


def _scan(j, program: str, s: WardenParams) -> Scan:
    """`scanOf`: the scan's terms from its Program's descriptor, which must be the registered Program and the generator
    form of the schedule's link-window in the Boolean profile's types (`BOOL_BYTE`, `BOOL_U16`), an egress template
    with a row among its parameters (the checker form) refused first. Then what `NetTiming.Layout.Fits` asks: the
    walk within `FUEL` (`_walkable`), the root returning every egress leaf, the egress template returning only gates it
    computes (`_own_returns`), each ingress row and every template argument one input node's leaves (`_input_arg`),
    no egress argument an ingress row's, and the Program topological (`_topological`). It reads the descriptor's
    nodes, signatures and returns; C-Flock's
    decoder, which Lean runs on it, refuses no descriptor this module writes."""
    if not isinstance(j, dict) or j.get("schema") != SCHEMA:
        raise D.Reject("scan: not a Program's descriptor")
    if program_sha512(j) != program:
        raise D.Reject("scan: its SHA-512 is not the registered Program's")
    defs, callees = j["definitions"], j["callees"]
    nodes = defs[j["root"]]["body"]["nodes"]
    templates = [(x, defs[callees[x["fn"]]]["body"]) for x in nodes if defs[callees[x["fn"]]]["form"] != "primitive"]
    T, row = s.window, ["Array", s.lanes * s.slots * s.cell_bytes, BOOL_BYTE]
    if not templates:
        raise D.Reject("scan: no template")
    (xf, bf), later, (xe, be) = templates[0], templates[1:-1], templates[-1]
    if any(k == "row" or t == row for k, t in be["params"]):
        raise D.Reject("scan: the egress template takes its row as an input (the checker form)")
    if len(templates) not in (2, 3):
        raise D.Reject("scan: not the generator form's templates")
    names = lambda b: tuple(k for k, _ in b["params"])  # noqa: E731
    if (names(bf) != FIRST_PARAMS or any(names(b) != INGRESS_PARAMS for _, b in later)
            or names(be) != EGRESS_PARAMS):
        raise D.Reject("scan: the templates' parameters are not the generator form's")
    counts = [(xf, 1), *((x, T - 1) for x, _ in later), (xe, T)]
    if len(later) != (T > 1) or not all(x["form"] == "batch" and x["n"] == n for x, n in counts):
        raise D.Reject(f"scan: not bucket 0, {T - 1} later ingress buckets and {T} egress buckets")
    if bf["params"][0][1] != row or any(b["params"][0][1] != row for _, b in later):
        raise D.Reject("scan: an ingress template's row is not the schedule's")
    if any(b["params"][2][1] != bf["params"][1][1] or b["ret_type"] != bf["ret_type"] for _, b in later):
        raise D.Reject("scan: the ingress templates' prog or return differ")
    if be["ret_type"] != ["Tuple", [BOOL_U16, row]]:
        raise D.Reject("scan: the egress template doesn't return a u16 and the schedule's row")
    inputs = sum(int(w[1]) * _members(x) for x in nodes if (w := INPUT_ID.match(callees[x["fn"]])))
    bits = lambda b: sum(bits_of(t) for _, t in b["params"])  # noqa: E731
    if inputs != bits(bf) + (T - 1) * sum(bits(b) for _, b in later) + T * bits(be):
        raise D.Reject("scan: an input gate that is no template's parameter")
    if not _walkable(j):
        raise D.Reject("scan: not laid out as C-Flock's decoder lays a Program out, or past what its resolution walks "
                       "(NciCheck.Desc.rootLaidOut, inputsAtRoot and walkable)")
    ke = max(k for k, x in enumerate(nodes) if defs[callees[x["fn"]]]["form"] != "primitive")
    if not any(a[:3] == ("n", ke, 0) and T * leaves_of(be["ret_type"]) <= a[3]
               for q in _parts(defs[j["root"]]["body"]["ret"], nodes) if (a := _affine(q))):
        raise D.Reject("scan: the root doesn't return every egress leaf, so an egress gate would be one the Program "
                       "doesn't return (NetTiming.Layout.Fits)")
    if not _own_returns(j)(callees[xe["fn"]]):
        raise D.Reject("scan: the egress template returns a leaf it doesn't compute (a parameter's, passed through), "
                       "so an egress gate would be an input (NetTiming.Layout.Fits)")
    ingress = [xf, *(x for x, _ in later)]
    rows = [_input_arg(x["args"][0], nodes, j) if x["args"] else None for x in ingress]
    if None in rows:
        raise D.Reject("scan: an ingress template's row is not one input node's leaves, so an ingress gate would be no "
                       "input (NetTiming.Layout.Fits)")
    if any(_input_arg(a, nodes, j) is None for x in [*ingress, xe] for a in x["args"]):
        raise D.Reject("scan: a template's argument is not one input node's leaves")
    if any(_input_arg(a, nodes, j) in rows for a in xe["args"]):
        raise D.Reject("scan: an egress template's argument is an ingress row's input node, so an egress gate would "
                       "depend on the ingress (NetTiming.Layout.Fits)")
    if not _topological(j):
        raise D.Reject("scan: a body names a later node (Flock.Program.Prog.orderViolation): the Program is not "
                       "topological, so a gate's operands could be gates after it")
    return Scan(T, 0, bits_of(bf["ret_type"]), bits_of(be["ret_type"]))


def params(j) -> Params:
    if not isinstance(j, dict):
        raise D.Reject("the object: not an object")
    if sorted(j) not in (PARAMS_KEYS, ROWS_PARAMS_KEYS):
        raise D.Reject(f"the object: the keys must be {PARAMS_KEYS} or {ROWS_PARAMS_KEYS}, not {sorted(j)}")
    audits = [_audit(a, "an audit") for a in _of(j["audits"], list, "audits")]
    cells = tuple(_nat(x, "cells") for x in _of(j["cells"], list, "cells"))
    first = _nat(j["first"], "first")
    jitter = _nat(j["jitter"], "jitter")
    if jitter < 1:
        raise D.Reject("jitter: the observations' jitter bound is at least 1")
    link = _of(j["link"], str, "link")
    if len(link.encode()) >= 1 << 16:
        raise D.Reject("link: a name is at most 65,535 UTF-8 bytes")
    lj = j["lms"]
    _exactly(lj, ["H", "N", "W", "toy"], "lms")
    try:
        lp = lms.Params(*(_nat(lj[k], "lms") for k in ("N", "W", "H")), toy=_of(lj["toy"], bool, "lms"))
    except ValueError as x:
        raise D.Reject(str(x)) from None
    key = P.unhex(_of(j["key"], str, "key"))
    if len(key) != 16 + lp.N:
        raise D.Reject(f"an LMS public key is {16 + lp.N} bytes")
    session = P.unhex(_of(j["session"], str, "session"))
    if len(session) != lms.SESSION_BYTES:
        raise D.Reject(f"a certifier session id is {lms.SESSION_BYTES} bytes")
    program = _of(j["program"], str, "program")
    if not DIGEST.fullmatch(program):
        raise D.Reject("program: not a SHA-512 digest")
    if "sync_set" in j:
        sync_set, schedule, scan = _sync_set(j["sync_set"]), None, None
    else:
        schedule = _schedule(j["schedule"])
        sync_set, scan = None, _scan(j["scan"], program, schedule)
    windows = _nat(j["windows"], "windows")
    if windows < 1:
        raise D.Reject("windows: a period has at least one link-window")
    return Params(tuple(audits), cells, first, jitter, link, LmsKey(lp, key), session, program, sync_set, windows,
                  schedule, scan)


def registration_digest(record: dict) -> bytes:
    """`registration.digest`: SHA-512(format ‖ 0 ‖ the canonical record)."""
    return hashlib.sha512(record["format"].encode() + b"\0" + D.canonical(record).encode()).digest()


def _registration(j, ctx: str) -> Registration:
    if not isinstance(j, dict) or not isinstance(j.get("format"), str):
        raise D.Reject(f"{ctx}: no format")
    if j["format"] not in FORMATS:
        raise D.Reject(f"{ctx}: format is not {' or '.join(FORMATS)}")
    return Registration(j, registration_digest(j))


def commitment(b: bytes) -> Commitment:
    j = P.canonical_json(b)
    _exactly(j, ["audits", "windows"], "the object")
    return Commitment([_registration(x, "an audit's registration") for x in _of(j["audits"], list, "audits")],
                      [_registration(x, "a link-window's registration") for x in _of(j["windows"], list, "windows")], b)


def _answer(j, ctx: str) -> dict:
    try:
        return CC.answer_of(j, ctx)
    except CC.Refused as e:
        raise D.Reject(str(e)) from None


def _certificate(j, ctx: str) -> Certificate:
    _exactly(j, CERTIFICATE_KEYS, ctx)
    try:
        direction = Direction(_of(j["direction"], str, ctx))
    except ValueError:
        raise D.Reject(f"{ctx}: not a direction") from None
    return Certificate(P.unhex(_of(j["key_id"], str, ctx)), _of(j["link"], str, ctx), direction,
                       _nat(j["window"], ctx), P.unhex(_of(j["digest"], str, ctx)),
                       P.unhex(_of(j["signature"], str, ctx)))


def _rows(j, ctx: str) -> dict:
    """`rowsOf`: a row registration with exactly its keys (and its layout's and certificate's), its numbers naturals,
    its strings strings and its byte fields lowercase hex."""
    _exactly(j, ROWS_KEYS, ctx)
    _exactly(j["layout"], LAYOUT_KEYS, f"{ctx}.layout")
    _exactly(j["certificate"], ["digest", "key", "signature"], f"{ctx}.certificate")
    for k in ("format", "value", "link", "direction", "parameters", "row_schema", "domain"):
        _of(j[k], str, f"{ctx}.{k}")
    for k in ("window", "cell_bytes", "rows", "words", "word_bits"):
        _nat(j[k], f"{ctx}.{k}")
    for k in LAYOUT_KEYS:
        _nat(j["layout"][k], f"{ctx}.layout")
    for o, k in ((j, "wiring"), (j, "root"), *((j["certificate"], k) for k in ("digest", "key", "signature"))):
        P.unhex(_of(o[k], str, f"{ctx}.{k}"))
    return j


def _window(j, ctx: str) -> Window:
    clock = isinstance(j, dict) and sorted(j) == CLOCK_WINDOW_KEYS
    if not clock:
        _exactly(j, ROWS_WINDOW_KEYS, ctx)
    answer = _answer(j["answer"], f"{ctx}.answer")
    certificates = [_certificate(c, f"{ctx}.certificates") for c in _of(j["certificates"], list, ctx)]
    if clock:
        return Window(answer, certificates, _certificate(j["clock"], f"{ctx}.clock"))
    return Window(answer, certificates, None,
                  [_rows(x, f"{ctx}.registrations") for x in _of(j["registrations"], list, ctx)])


def evidence(b: bytes) -> tuple[list[dict], list[Window]]:
    j = P.canonical_json(b)
    _exactly(j, ["audits", "windows"], "the object")
    return ([_answer(a, "an audit's answer") for a in _of(j["audits"], list, "audits")],
            [_window(x, "a link-window") for x in _of(j["windows"], list, "windows")])


class Finding(Exception):
    """A finding against the period: a rejection, or what leaves it undecided."""

    def __init__(self, verdict: str, reason: str):
        super().__init__(reason)
        self.verdict, self.reason = verdict, reason


def _certifies_on(p: Params, link: str, d: Direction, w: int, c: Certificate) -> bool:
    return c.direction is d and c.link == link and c.window == w and c.verify(p.key, p.session)


def _certifies(p: Params, d: Direction, w: int, c: Certificate) -> bool:
    return _certifies_on(p, p.link, d, w, c)


def _clock_digest(s: ClockSync) -> bytes | None:
    try:
        return clock_digest(s)
    except ValueError:
        return None


def clear_clocks(ss: SyncSet) -> set[bytes]:
    """Each sync of Σ_sync's `clock_digest` itself: the unsalted form, whose digest shows the sync to anyone holding
    Σ_sync."""
    return {d for d in (_clock_digest(ClockSync(x, rho)) for x in ss.deltas for rho in ss.rhos) if d is not None}


def _answered(what: str, reg: Registration, a: dict) -> tuple[dict, object]:
    if a["protocol"] != ONE_STAGE:
        raise Finding(REJECT, f"{what}: protocol {a['protocol']}")
    if a["commitment"] != reg.digest.hex():
        raise Finding(REJECT, f"{what}: the answer is for another registration")
    if a["verdict"] == ACCEPT:
        try:
            return CC.integrity(a["profile"]), a["public"]
        except CC.Refused as e:
            raise Finding(REJECT, f"{what}: {e}") from None
    raise Finding(a["verdict"], f"{what}: service: {a['reason']}")


def served(w: int) -> dict:
    return {"kind": "served", "window": w}


def _window_check(p: Params, clear: set[bytes], earlier: list[bytes], w: int, reg: Registration, x: Window) -> None:
    what = f"window {w}"
    if x.clock is None:
        raise Finding(REJECT, f"{what}: registrations, not a clock certificate")
    if not (reg.record.get("program") == p.program
            and D.canonical(reg.record.get("window")) == D.canonical(served(w))):
        raise Finding(REJECT, f"{what}: registration")
    if not (len(x.certificates) == 2
            and all(_certifies(p, d, w, c) for c, d in zip(x.certificates, (EGRESS, INGRESS)))):
        raise Finding(REJECT, f"{what}: certificates")
    if not _certifies(p, EGRESS, w, x.clock):
        raise Finding(REJECT, f"{what}: clock")
    if x.clock.digest in clear:
        raise Finding(REJECT, f"{what}: clock: unsalted")
    if x.clock.digest in earlier:
        raise Finding(REJECT, f"{what}: clock: reused")
    prof, _ = _answered(what, reg, x.answer)
    if ([(lv["name"], lv["size"], lv["draw"]) for lv in prof["levels"]] != [(UNIT, 1, {"subset": 1})]
            or prof["drawn"] != [(UNIT, 1)]):
        raise Finding(REJECT, f"{what}: profile")


def _rows_check(p: Params, w: int, d: Direction, doc: dict) -> None:
    """`rowsCheck`: the verifier's own read of the registration (`warden_rows.check_registration`)."""
    try:
        WR.check_registration(doc, p.key, p.session, p.schedule, p.link, d, w)
    except ValueError as e:
        raise Finding(REJECT, f"window {w}: the {d.value} rows: {e}") from None


def _holds(reg: Registration, doc: dict) -> bool:
    """The scan's registration holds the rows: an entry of its `roots` or `registered` names their root and domain."""
    entries = [e for k in ("roots", "registered") if isinstance(reg.record.get(k), list) for e in reg.record[k]]
    return any(isinstance(e, dict) and e.get("root") == doc["root"] and e.get("domain") == doc["domain"]
               for e in entries)


#: the scan's Programs' row in its registration: its `roots` entry's port, or its `registered` entry's value
PROG = "scan/prog"


def output_port(port) -> bool:
    """`outputPort`: a unit's committed output port, ``<unit>/o<j>`` with j ≥ 1 (ASCII digits, no leading zero)."""
    if not isinstance(port, str) or len(parts := port.split("/")) < 2:
        return False
    j = parts[-1][1:]
    return parts[-1][:1] == "o" and j != "" and all("0" <= c <= "9" for c in j) and j[0] != "0"


def _tied_audits(audits: list[Registration], reg: Registration) -> list[int]:
    """`tiedAudits`: the audits, by index, whose registration's `roots` has an output port's entry (`output_port`) at
    the root and domain of the scan's registration's `scan/prog` entry (`PROG`)."""
    def entries(j: dict, k: str) -> list[dict]:
        v = j.get(k)
        return [e for e in v if isinstance(e, dict)] if isinstance(v, list) else []

    prog = [e for e in entries(reg.record, "roots") if e.get("port") == PROG] + \
        [e for e in entries(reg.record, "registered") if e.get("value") == PROG]
    return [i for i, a in enumerate(audits)
            if any(isinstance(e.get("root"), str) and isinstance(e.get("domain"), str)
                   and any(output_port(o.get("port")) and o.get("root") == e["root"] and o.get("domain") == e["domain"]
                           for o in entries(a.record, "roots"))
                   for e in prog)]


def _prog_tied(audits: list[Registration], reg: Registration) -> bool:
    """`progTied`: some audit is tied to the scan's `prog` (`_tied_audits`)."""
    return bool(_tied_audits(audits, reg))


def _tied_in(audits: list[Registration], windows: list[Registration], i: int) -> bool:
    """`tiedIn`: some link-window's `prog` is audit `i`'s output (`_tied_audits`)."""
    return any(i in _tied_audits(audits, reg) for reg in windows)


def records(link: str) -> tuple[tuple[str, Direction], ...]:
    """The rows form's certified records of a link-window (`warden_serve.CERTIFIED`): requests, responses, fails."""
    return (link, INGRESS), (link, EGRESS), (f"{link}-fails", EGRESS)


def _rows_window_check(p: Params, audits: list[Registration], w: int, reg: Registration,
                       x: Window) -> tuple[Facts, int]:
    """`rowsWindowCheck`: on acceptance the scan's facts and its tied audits' free inputs, the parameters' `free` of
    each audit its `prog` is an output of (`_tied_audits`)."""
    what = f"window {w}"
    if x.registrations is None:
        raise Finding(REJECT, f"{what}: a clock certificate, not registrations")
    if reg.record.get("program") != p.program:
        raise Finding(REJECT, f"{what}: registration")
    if not (len(x.certificates) == 3
            and all(_certifies_on(p, link, d, w, c) for c, (link, d) in zip(x.certificates, records(p.link)))):
        raise Finding(REJECT, f"{what}: certificates (none of a window in which the network certifier refused a "
                              "connection)")
    if len(x.registrations) != 2:
        raise Finding(REJECT, f"{what}: registrations")
    for doc, d in zip(x.registrations, (INGRESS, EGRESS)):
        _rows_check(p, w, d, doc)
    if not all(_holds(reg, doc) for doc in x.registrations):
        raise Finding(REJECT, f"{what}: another window's answer")
    if not _prog_tied(audits, reg):
        raise Finding(REJECT, f"{what}: the scan's prog is no compute audit's output")
    prof, pub = _answered(what, reg, x.answer)
    f = _facts(what, prof, pub)
    if f.units != 2 * p.schedule.window:
        raise Finding(REJECT, f"{what}: profile: {f.units} units, not the scan's 2T")
    return f, sum(p.audits[i].free for i in _tied_audits(audits, reg))


def _facts(what: str, prof: dict, pub) -> Facts:
    """`factsOf`: one level drawn `subset:k` or stratified, and the public's bound."""
    if len(prof["levels"]) != 1:
        raise Finding(REJECT, f"{what}: profile: not one level")
    (lv,) = prof["levels"]
    draw = lv["draw"]
    if isinstance(draw, dict) and "subset" in draw:
        drawn = draw["subset"]
    elif isinstance(draw, dict) and draw.get("law") == "stratified":
        drawn = sum(s["k"] for s in draw["strata"])
    else:
        raise Finding(REJECT, f"{what}: profile: not a subset or stratified draw")
    bound = pub.get("bound") if isinstance(pub, dict) else None
    if not isinstance(bound, int) or isinstance(bound, bool) or bound < 0:
        raise Finding(REJECT, f"{what}: no bound")
    return Facts(lv["size"], drawn, prof["delta"], bound)


def _audit_check(i: int, audit: Audit, reg: Registration, a: dict, tied: bool = False) -> Facts:
    """`auditCheck`: the registration names the audit's Program, the answer's units are the Program's, and one tied to
    a link-window's scan (`_tied_in`, in the rows form) has every unit drawn."""
    what = f"audit {i}"
    if reg.record.get("program") != audit.program:
        raise Finding(REJECT, f"{what}: registration: not its Program")
    prof, pub = _answered(what, reg, a)
    f = _facts(what, prof, pub)
    if f.units != audit.units:
        raise Finding(REJECT, f"{what}: profile: {f.units} units, not its Program's {audit.units}")
    if tied and f.drawn != f.units:
        raise Finding(REJECT, f"{what}: tied to a link-window's scan with {f.drawn} of its {f.units} units drawn: K "
                              "charges a tied audit no faulty term (its wrong units' outputs are the egress's), so "
                              f"draw all {f.units}")
    return f


def clog2(x: int) -> int:
    """`⌈log2 x⌉`, 0 for x ≤ 1."""
    return 0 if x <= 1 else (x - 1).bit_length()


def at_most(n: int, b: int) -> int:
    """`Σ_{j ≤ b} C(n, j)`: the sets of at most `b` of `n` units."""
    return sum(math.comb(n, j) for j in range(min(b, n) + 1))


def _facts_json(f: Facts) -> dict:
    return {"delta": str(f.delta), "drawn": f.drawn, "units": f.units, "wrong_units": f.wrong}


def _terms(free: int, steps: int, per_unit: int, f: Facts, jitter: int) -> dict:
    """`AuditTerms.of` (a compute audit's, `per_unit` its Program's widest unit's committed bits) and `Scan.terms` (a
    scan's, `per_unit` the wider instance's committed bits)."""
    return _facts_json(f) | {"faulty_bits": f.wrong * per_unit, "faulty_sets": clog2(at_most(f.units, f.wrong)),
                             "free": free, "jitter": clog2(jitter ** steps)}


def public(p: Params, facts: list[Facts], scans: list[tuple[Facts, int]] | None = None) -> dict:
    """K and its terms over `PeriodLeakAccepted`'s windows: the clock form's compute audits, or the rows form's
    link-windows' scans (`scans`, each its facts and its tied audits' free inputs; None in the clock form), where a
    compute audit's own terms are in none and a tied audit's free inputs are in its link-window's `free`."""
    cells = sum(clog2(cap + 1) for cap in p.cells)
    if scans is None:
        units = [_terms(a.free, a.steps, a.bits, f, p.jitter) for a, f in zip(p.audits, facts)]
        schedules = clog2((len(p.sync_set.deltas) * len(p.sync_set.rhos)) ** p.windows)
        out = {"audits": units}
    else:
        s = p.scan
        units = [_terms(s.free + tied, s.steps, max(s.ingress, s.egress), f, p.jitter) for f, tied in scans]
        schedules = cells
        out = {"audits": [_facts_json(f) for f in facts], "scans": units}
    K = schedules + cells + sum(t["free"] + t["faulty_sets"] + t["faulty_bits"] + t["jitter"] for t in units)
    eps = sum((Fraction(t["delta"]) for t in units), Fraction(0))
    return out | {"K": K, "cells": cells, "eps": str(eps), "schedules": schedules, "windows": p.windows}


def check(p: Params, c: Commitment, e) -> tuple:
    audits, windows = e
    if len(c.windows) != p.windows:
        return REJECT, "the commitment registers another number of link-windows"
    if len(c.audits) != len(p.audits):
        return REJECT, "the commitment registers another number of audits"
    if len(windows) != len(c.windows):
        return REJECT, "evidence: another number of link-windows"
    if len(audits) != len(c.audits):
        return REJECT, "evidence: another number of audits' answers"
    findings, facts, scans = [], [], []
    clear = clear_clocks(p.sync_set) if p.schedule is None else set()
    for i, (reg, x) in enumerate(zip(c.windows, windows)):
        try:
            if p.schedule is None:
                _window_check(p, clear, [y.clock.digest for y in windows[:i] if y.clock is not None], p.first + i,
                              reg, x)
            else:
                scans.append(_rows_window_check(p, c.audits, p.first + i, reg, x))
        except Finding as f:
            findings.append(f)
    if any(f.delta != scans[0][0].delta for f, _ in scans):
        findings.append(Finding(REJECT, "a scan's δ is not the period's"))
    for i, (audit, reg, a) in enumerate(zip(p.audits, c.audits, audits)):
        try:
            facts.append(_audit_check(i, audit, reg, a, p.schedule is not None and _tied_in(c.audits, c.windows, i)))
        except Finding as f:
            findings.append(f)
    if rejected := next((f for f in findings if f.verdict == REJECT), None):
        return REJECT, rejected.reason
    if findings:
        return UNDECIDED, findings[0].reason
    return ACCEPT, public(p, facts, None if p.schedule is None else scans), {"link": p.link, "windows": p.windows}


def reference(case) -> dict:
    return P.answer(ID, case, params=params, commitment=commitment, evidence=evidence, check=check,
                    digest=lambda c: tagged_sha512(COMMITMENT_TAG, c.bytes))


# --- the generator --------------------------------------------------------------------------------------------------------

#: Small LMS parameter sets (N, W, H, toy): the toy N = 8, and the deployed N = 64 at height 1.
KEYS = [(8, 8, 1, True), (8, 2, 2, True), (64, 4, 1, False)]
#: The rows form's δ, one a period.
DELTAS = ("1/100", "1/1000000", "1/7")


@lru_cache(maxsize=None)
def _key(i: int, seed: int = 0, same_id: bool = False) -> lms.PrivateKey:
    """KEYS[i]'s key from `seed`; `same_id`: the identifier of seed 0's key with another seed's one-time keys."""
    N, W, H, toy = KEYS[i]
    p = lms.Params(N, W, H, toy=toy)
    s = hashlib.sha512(b"network-period %d %d" % (i, seed)).digest()[:N]
    return lms.key(p, _key(i).I, s) if same_id else lms.keygen(p, s)


EXPECT = {
    "honest": ACCEPT,
    "program": REJECT, "provenance": REJECT,
    "cert-swap": REJECT, "cert-count": REJECT, "cert-forged": REJECT, "cert-window": REJECT, "cert-link": REJECT,
    "cert-session": REJECT, "cert-key": REJECT,
    "clock-ingress": REJECT, "clock-forged": REJECT, "clock-window": REJECT, "clock-session": REJECT,
    "clock-reused": REJECT, "clock-unsalted": REJECT,
    "answer-protocol": REJECT, "answer-other": REJECT, "answer-reject": REJECT, "answer-undecided": UNDECIDED,
    "answer-profile": REJECT,
    "audit-other": REJECT, "audit-reject": REJECT, "audit-undecided": UNDECIDED, "audit-profile": REJECT,
    "audit-bound": REJECT, "audit-program": REJECT, "audit-units": REJECT,
    "undecided-and-reject": REJECT, "form": REJECT,
    "first": REJECT, "count-windows": REJECT, "count-audits": REJECT, "evidence-count": REJECT,
    "params": REJECT, "commitment": REJECT, "evidence": REJECT, "case": REJECT,
    "rows-honest": ACCEPT,
    "rows-program": REJECT,
    "rows-cert-missing": REJECT, "rows-cert-swap": REJECT, "rows-cert-fails-link": REJECT,
    "rows-cert-direction": REJECT, "rows-cert-forged": REJECT, "rows-cert-window": REJECT, "rows-cert-session": REJECT,
    "rows-cert-key": REJECT, "rows-past-slots": REJECT,
    "rows-reg-count": REJECT, "rows-reg-swap": REJECT, "rows-reg-forged": REJECT, "rows-reg-window": REJECT,
    "rows-reg-link": REJECT, "rows-reg-session": REJECT, "rows-reg-key": REJECT, "rows-reg-field": REJECT,
    "rows-schedule": REJECT,
    "rows-answer-protocol": REJECT, "rows-answer-other": REJECT, "rows-answer-window": REJECT,
    "rows-answer-reject": REJECT, "rows-answer-undecided": UNDECIDED, "rows-answer-profile": REJECT,
    "rows-answer-bound": REJECT, "rows-delta": REJECT, "rows-units": REJECT, "rows-prog-untied": REJECT,
    "rows-tied-undrawn": REJECT,
    "rows-audit-other": REJECT, "rows-audit-reject": REJECT, "rows-audit-undecided": UNDECIDED,
    "rows-audit-profile": REJECT, "rows-audit-bound": REJECT, "rows-audit-program": REJECT, "rows-audit-units": REJECT,
    "rows-undecided-and-reject": REJECT, "rows-form": REJECT,
    "rows-first": REJECT, "rows-count-windows": REJECT, "rows-count-audits": REJECT, "rows-evidence-count": REJECT,
    "rows-params": REJECT, "rows-commitment": REJECT, "rows-evidence": REJECT,
}
MUTATIONS = list(EXPECT)


def form_of(m: str) -> str:
    """The form a change's period is in: the rows form's changes are named `rows-…`."""
    return "rows" if m.startswith("rows-") else "clock"


def _flip(b: bytes, i: int) -> bytes:
    i %= len(b)
    return b[:i] + bytes([b[i] ^ 1]) + b[i + 1:]


def _cert_json(c: Certificate) -> dict:
    return {"digest": c.digest.hex(), "direction": c.direction.value, "key_id": c.key_id.hex(), "link": c.link,
            "signature": c.signature.hex(), "window": c.window}


def clock_commitment(salt: bytes, sync: ClockSync) -> bytes:
    """The digest a salted clock certificate signs (`certificate.CertifiedClock`): the sync's clock digest under the
    certifier's salt, SHA-512("verity/net-cert/v1" ‖ salt_c ‖ clock digest)."""
    return certificate_digest(salt, clock_digest(sync))


def _accepted(digest: bytes, program: str, query: str, size: int, k: int, delta: str, bound: int,
              strata: list[dict] | None = None) -> dict:
    """One-stage's answer accepting `k` of `size` units drawn `subset:k`, or drawn stratified over `strata` (each
    `{k, name, size}`, k and size their sums), as `OneStage.check`'s public and profile have it."""
    units = list(range(k))
    draw = {"subset": k} if strata is None else {"law": "stratified", "strata": strata}
    return {"commitment": digest.hex(),
            "profile": {"covered": [], "delta": delta, "drawn": {UNIT: k},
                        "levels": [{"draw": draw, "name": UNIT, "query": query, "size": size}],
                        "linked": False, "prescribed": [], "programs": [program]},
            "protocol": ONE_STAGE, "public": {"bound": bound, "drawn": units, "per_template": None, "proved": units},
            "reason": None, "verdict": ACCEPT}


def _refused(answer: dict, verdict: str) -> dict:
    return answer | {"profile": None, "public": None, "reason": f"the audit is {verdict}", "verdict": verdict}


#: The schedule's fields that leave its rows' shape (B, S, T) as it is: a change to one changes only its digest.
SAME_SHAPE = ("bucket_ms", "delay", "egress_cells", "ingress_buckets", "ingress_cells", "queue")
#: The orders of a link-window's three records' certificates other than their own.
SWAPS = ((0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


def schedule_json(s: WardenParams) -> dict:
    return {k: getattr(s, k) for k in SCHEDULE_KEYS}


def _small_schedule(r: random.Random, small: bool) -> WardenParams:
    """A schedule of small rows: B just past a cell's header, S = lanes · slots at most 4, T at most 3 (2 if
    `small`), and a timestep `every` that divides T."""
    T = r.randint(1, 2 if small else 3)
    return WardenParams(cell_bytes=r.choice([11, 12, 16]), slots=r.randint(1, 2), bucket_ms=r.randint(1, 200),
                        delay=r.randint(1, 4), lanes=r.randint(1, 2), queue=r.randint(1, 8), window=T,
                        ingress_cells=r.randint(1, 8), ingress_buckets=r.randint(1, 4), egress_cells=r.randint(1, 16),
                        every=r.choice([d for d in range(1, T + 1) if T % d == 0]))


BIT, BYTE, U16 = Value(1), Value(8), Value(16)
#: the compute audits' stand-in unit's gate: a byte from a u16
LOW = PrimitiveDefinition("NetworkPeriodScanLow", 1, (("s", U16),), BYTE, lambda s: s & 0xFF, register=False)
#: a byte and a u16 in the Boolean profile, as the scan's stand-ins type them (`BOOL_BYTE`, `BOOL_U16`)
BYTE_BITS, U16_BITS = Array(8, BIT), Array(OUT_BITS, BIT)

#: the scan's stand-ins' gate, a bit from two, and `N` bits from two (a byte's or a u16's)
MIX = PrimitiveDefinition("NetworkPeriodScanMix", 2, (("a", BIT), ("b", BIT)), BIT, lambda a, b: a ^ b,
                          register=False)
SPREAD = CompositeDefinition("NetworkPeriodScanSpread", 0, ("N",),
                             lambda S: ((("a", BIT), ("b", BIT)), Array(S.N, BIT)),
                             lambda B, S, a, b: B.batch(MIX, a, b, axes=(None, None), n=S.N), register=False)


def _ingress_ret(S):
    return Tuple(U16_BITS, Array(S.STATE, BIT), Array(S.VIEW, BIT))


def _first_sig(S):
    return tuple(zip(S.NAMES, (Array(S.ROW, BYTE_BITS), Array(S.PROG, BIT)))), _ingress_ret(S)


def _ingress_sig(S):
    return tuple(zip(S.NAMES, (Array(S.ROW, BYTE_BITS), Array(S.STATE, BIT), Array(S.PROG, BIT)))), _ingress_ret(S)


def _egress_sig(S):
    return tuple(zip(S.NAMES, (Array(S.WORDS, U16_BITS),))), Tuple(U16_BITS, Array(S.ROW, BYTE_BITS))


def _first(B, S, row, prog):
    b, p = row[0][0], prog[0]
    return tuple_of(B.call(bind(SPREAD, N=OUT_BITS), b, p), B.batch(MIX, p, b, axes=(None, None), n=S.STATE),
                    B.batch(MIX, p, b, axes=(None, None), n=S.VIEW))


def _ingress(B, S, row, sin, prog):
    b, p = row[0][0], prog[0]
    return tuple_of(B.call(bind(SPREAD, N=OUT_BITS), b, p), B.batch(MIX, sin, b, axes=(0, None)),
                    B.batch(MIX, p, b, axes=(None, None), n=S.VIEW))


def _egress(B, S, state):
    s = state[0][0]
    return tuple_of(B.call(bind(SPREAD, N=OUT_BITS), s, s),
                    B.batch(bind(SPREAD, N=8), s, s, axes=(None, None), n=S.ROW))


#: Stand-ins of the scan's templates in the generator form, with their parameters and returns as `IngressScanFirst`,
#: `IngressScanBucket` and `EgressBucket` have them at small widths, in the Boolean profile's types: each unit computes
#: and returns its rows.
_STATICS = ("NAMES", "ROW", "STATE", "PROG", "VIEW")
FIRST_BUCKET = CompositeDefinition("NetworkPeriodScanFirst", 0, _STATICS, _first_sig, _first, register=False)
INGRESS_BUCKET = CompositeDefinition("NetworkPeriodScanIngress", 1, _STATICS, _ingress_sig, _ingress, register=False)
EGRESS_BUCKET = CompositeDefinition("NetworkPeriodScanEgress", 1, ("NAMES", "WORDS", "ROW"), _egress_sig, _egress,
                                    register=False)


def _checker_ingress_sig(S):
    return tuple(zip(S.NAMES, (Array(S.ROW, BYTE_BITS), Array(S.STATE, BIT), Array(S.PROG, BIT), Array(S.STATE, BIT),
                               Array(S.VIEW, BIT)))), BIT


def _checker_egress_sig(S):
    return tuple(zip(S.NAMES, (Array(S.WORDS, U16_BITS), Array(S.ROW, BYTE_BITS)))), U16_BITS


#: Stand-ins of the checker form's templates, which take the rows they compare as inputs (its egress row an input gate,
#: which `NetTiming.Layout.Fits` refuses).
CHECKER_INGRESS = CompositeDefinition(
    "NetworkPeriodScanIngress", 0, _STATICS, _checker_ingress_sig,
    lambda B, S, row, sin, prog, sout, view: B.call(BL.XOR, BL.operands(sin[0], sout[0])), register=False)
CHECKER_EGRESS = CompositeDefinition(
    "NetworkPeriodScanEgress", 0, ("NAMES", "WORDS", "ROW"), _checker_egress_sig,
    lambda B, S, state, row: B.call(bind(SPREAD, N=OUT_BITS), state[0][0], row[0][0]), register=False)


@lru_cache(maxsize=None)
def scan_descriptor(T: int, row: int, widths: tuple[int, int, int, int], counts: tuple[int, ...] | None = None,
                    names: tuple[tuple[str, ...], ...] = (FIRST_PARAMS, INGRESS_PARAMS, EGRESS_PARAMS),
                    checker: bool = False) -> str:
    """The canonical descriptor of a scan's Program in the generator form: bucket 0's stand-in instance, `T − 1` later
    ingress instances (no template when there are none) and `T` egress instances (or `counts`), each ingress template's
    row `row` bytes, the ingress buckets' state, prog and view bits and the egress bucket's state words `widths`;
    `checker`: in the checker form instead, `T` ingress and `T` egress instances (or `counts`) that take their rows."""
    state, prog, view, words = widths
    ingress = {"ROW": row, "STATE": state, "PROG": prog, "VIEW": view}
    if checker:
        templates = (bind(CHECKER_INGRESS, NAMES=CHECKER_PARAMS[0], **ingress),
                     bind(CHECKER_EGRESS, NAMES=CHECKER_PARAMS[1], WORDS=words, ROW=row))
    else:
        templates = (bind(FIRST_BUCKET, NAMES=names[0], **ingress), bind(INGRESS_BUCKET, NAMES=names[1], **ingress),
                     bind(EGRESS_BUCKET, NAMES=names[2], WORDS=words, ROW=row))
    counts = counts or ((T, T) if checker else (1, T - 1, T))
    program = population.mixed_population_program([(t, n) for t, n in zip(templates, counts) if n])
    return D.canonical(codec.encode_program(program))


def _unit_sig(S):
    return (("x", Array(1, U16)),), Array(S.OUT, BYTE)


#: A compute audit's stand-in unit: it commits `OUT` bytes, a proof unit's 16-bit public output (`OUT` = 2) or more or
#: fewer, each one gate.
AUDIT_UNIT = CompositeDefinition("NetworkPeriodAuditUnit", 0, ("OUT",), _unit_sig,
                                 lambda B, S, x: B.batch(LOW, x[0], axes=(None,), n=S.OUT), register=False)
#: The stand-in unit's widths in bytes: one under, at and over the public output's 16 bits.
UNIT_BYTES = (1, 2, 5, 40)


@lru_cache(maxsize=None)
def audit_descriptor(units: tuple[tuple[int, int], ...]) -> str:
    """The canonical descriptor of a compute audit's Program over the stand-in unit: `n` instances committing `out`
    bytes each, for each `(n, out)` of `units` (each `out` its own template), one template as the window's population
    program (`population_program`, `TemplateInstances_v1`), several as a mixed one."""
    templates = [(bind(AUDIT_UNIT, OUT=out), n) for n, out in units]
    program = population.population_program(*templates[0]) if len(templates) == 1 else \
        population.mixed_population_program(templates)
    return D.canonical(codec.encode_program(program))


#: How `rows-prog-untied` breaks its link-window's tie of the scan's `prog` to an audit's output: the scan's entry at
#: another root or domain, the audit's entry at a port that is no unit's output, no `scan/prog` entry or one under
#: another name, the output's entry in the scan's own registration and in no audit's, or neither entry's root a string
UNTIES = ("root", "domain", "port", "missing", "renamed", "scan", "untyped")


def _present(e: dict) -> dict:
    return {k: v for k, v in e.items() if v is not None}


def _prog_entries(rp: random.Random, tie: dict, where: str, untie: str | None) -> list[dict]:
    """The scan's registration's entries for its `prog` (`PROG`, named by port in `roots` and by value in
    `registered`) at its tie's root and domain, broken as `untie` says."""
    key = "port" if where == "roots" else "value"
    e = {"domain": tie["domain"], key: PROG, "root": tie["root"]}
    if untie == "missing":
        return []
    if untie in ("root", "domain"):
        e[untie] = rp.randbytes(64).hex()
    elif untie == "renamed":
        other = "value" if key == "port" else "port"
        e = rp.choice([{other if k == key else k: v for k, v in e.items()},
                       e | {key: rp.choice(["scan/prog2", "scan/state", "Scan/prog", "prog", "scan/prog/"])}])
    scan = [{"domain": tie["domain"], key: tie["port"], "root": tie["root"]}] if untie == "scan" else []
    return [_present(x) for x in [e, *scan]]


def _output_entries(rp: random.Random, i: int, ties: dict, T: int, aimed: int, untie: str | None) -> list[dict]:
    """Audit `i`'s registration's `roots`: for each link-window's `prog` tied to it, the output port its tie names at
    the tie's root and domain (`T` leaves, a bucket's row each), and an input port's entry at another root; at
    link-window `aimed`, broken as `untie` says."""
    out = []
    for w, x in ties.items():
        if x["audit"] != i or (w == aimed and untie == "scan"):
            continue
        port = rp.choice([f"block-{w}/o0", f"block-{w}/prog", f"block-{w}/o01", "o1", f"block-{w}/O1", f"block-{w}/o",
                          f"block-{w}/o1x", f"block-{w}/o\u0663", f"block-{w}/o\u00b9", 1]) \
            if w == aimed and untie == "port" else x["port"]
        out.append(_present({"domain": x["domain"], "leaves": T, "port": port, "root": x["root"]}))
    out.append({"domain": rp.randbytes(64).hex(), "leaves": T, "port": f"block-{rp.randrange(4)}/prog",
                "root": rp.randbytes(64).hex()})
    rp.shuffle(out)
    return out


def build(m: str, seed: int, small: bool = False) -> dict:
    """A period's case with the change `m`, in the form `m` names (`form_of`), every choice drawn from `seed`; `small`:
    the toy key, as few link-windows and audits as `m` needs."""
    r = random.Random(f"network-period {m} {seed}")
    rp = random.Random(f"network-period prog {m} {seed}")
    rows = form_of(m) == "rows"
    m = m.removeprefix("rows-")
    ki = 0 if small else r.randrange(len(KEYS))
    key = _key(ki)
    N, W, H, toy = KEYS[ki]
    session = r.randbytes(lms.SESSION_BYTES)
    # the rows form binds the link in its domain's canonical JSON, whose strings are ASCII without DEL
    link = r.choice(["eth0", "a", "d-e" if rows else "dé"])
    program = hashlib.sha512(b"link program %d" % r.randrange(3)).hexdigest()
    least_windows = 2 if m in ("undecided-and-reject", "clock-reused", "delta") else 1
    least_audits = 1 if m.startswith("audit") or m in ("undecided-and-reject", "count-audits") else 0
    n_windows = r.randint(least_windows, least_windows if small else 3)
    n_audits = r.randint(least_audits, max(least_audits, 1) if small else 2)
    n_audits = max(n_audits, 1) if rows else n_audits
    first = r.choice([0, 0, r.randrange(1, 6)])
    if rows:
        sched = bound = _small_schedule(r, small)
        widths = (r.randint(1, 3), r.randint(1, 4), r.randint(1, 3), r.randint(1, 2))
        scan = json.loads(scan_descriptor(sched.window, WR.row_bytes(sched), widths))
        program = program_sha512(scan)
        delta = r.choice(DELTAS)
    else:
        deltas = r.sample(range(4), r.randint(1, 3))
        rhos = r.sample([Fraction(1), Fraction(2), Fraction(1, 2), Fraction(3, 2)], r.randint(1, 3))
    t = r.randrange(1, n_windows) if m == "clock-reused" else r.randrange(n_windows)
    a = r.randrange(max(n_audits, 1))
    ties = {w: {"audit": rp.randrange(n_audits), "port": f"block-{w}/o{rp.choice([1, 1, 2, 10])}",
                "root": rp.randbytes(64).hex(), "domain": rp.randbytes(64).hex()}
            for w in range(first, first + n_windows)} if rows else {}
    untie = rp.choice(UNTIES) if m == "prog-untied" else None
    if untie == "untyped":
        ties[first + t]["root"] = rp.choice([None, 7, False])
    if m == "tied-undrawn":
        a = ties[first + t]["audit"]
    tied = {x["audit"] for x in ties.values()}
    signed = iter(range(1 << 30))

    def sign(link: str, d: Direction, w: int, digest: bytes, k: lms.PrivateKey = key, s: bytes = session) -> Certificate:
        c = Certificate(k.I, link, d, w, digest, b"")
        return dataclasses.replace(c, signature=lms.sign_at(k, next(signed) % 2 ** k.params.H, c.message(s)).encode())

    def registrations(w: int, s: WardenParams | None = None, on: str = link, ses: bytes = session) -> list[dict]:
        """Link `on`'s ingress and egress rows at window `w`, committed under schedule `s` (the period's) in session
        `ses` and registered, each head signed there."""
        s = s or sched
        out = []
        for d in (INGRESS, EGRESS):
            cells = WR.Cells.commit(on, d, w, [r.randbytes(WR.row_bytes(s)) for _ in range(s.window)],
                                    WR.Wiring((), r.randbytes(WR.WIRING_SALT_BYTES)), s, ses,
                                    [r.randbytes(WR.SALT_BYTES) for _ in range(s.window)])
            out.append(WR.registration(cells.head, sign(on, d, w, cells.head.digest(), s=ses), s))
        return out

    def resigned(doc: dict, **kw) -> dict:
        c = sign(doc["link"], Direction(doc["direction"]), doc["window"], bytes.fromhex(doc["certificate"]["digest"]),
                 **kw)
        return doc | {"certificate": {"digest": c.digest.hex(), "key": c.key_id.hex(), "signature": c.signature.hex()}}

    window_regs, windows, clocks = [], [], []
    for w in range(first, first + n_windows) if rows else ():
        aimed = w == first + t
        i, j = r.randrange(2), r.randrange(3)
        regs = registrations(w)
        named = regs
        if aimed and m == "answer-window":
            there = registrations(w + r.choice([1, -1] if w else [1]))
            named = r.choice([there, [regs[0], there[1]], [there[0], regs[1]],
                              [x | {"domain": y["domain"]} for x, y in zip(regs, there)]])
        elif aimed and m == "schedule":
            other = dataclasses.replace(sched, **{f: getattr(sched, f) + 1 for f in [r.choice(SAME_SHAPE)]})
            if r.random() < 0.5:
                bound = other
            else:
                under = registrations(w, other)
                regs = r.choice([under, [under[0], regs[1]], [regs[0], under[1]]])
        elif aimed and m == "reg-count":
            regs = r.choice([regs[:1], regs[1:], [], regs + regs[1:]])
        elif aimed and m == "reg-swap":
            regs = regs[::-1]
        elif aimed and m in ("reg-forged", "reg-window", "reg-link", "reg-session", "reg-key", "reg-field"):
            doc, c = regs[i], regs[i]["certificate"]
            sig, dig = bytes.fromhex(c["signature"]), bytes.fromhex(c["digest"])
            if m == "reg-forged":
                doc = r.choice([doc | {"certificate": c | {"signature": s.hex()}}
                                for s in (_flip(sig, seed), sig[:-1], b"")]
                               + [doc | {"certificate": c | {"digest": _flip(dig, seed).hex()}}]
                               + [doc | {k: r.randbytes(n).hex()} for k in ("root", "wiring") for n in (64, 63)])
            elif m == "reg-window":
                doc = registrations(w + r.choice([1, -1] if w else [1]))[i]
            elif m == "reg-link":
                doc = registrations(w, on=link + "x")[i]
            elif m == "reg-session":
                ses = _flip(session, seed)
                doc = resigned(doc, s=ses) if r.random() < 0.5 else registrations(w, ses=ses)[i]
            elif m == "reg-key":
                doc = resigned(doc, k=_key(ki, 1, r.random() < 0.5))
            else:
                lay = doc["layout"]
                f = r.choice(LAYOUT_KEYS)
                doc = doc | r.choice([
                    {"cell_bytes": doc["cell_bytes"] + 1}, {"rows": doc["rows"] + 1}, {"words": doc["words"] + 1},
                    {"word_bits": 8}, {"row_schema": "hm96-sha512/row/v1"},
                    {"format": "verity-network-warden/registration/v1"}, {"value": doc["value"] + "x"},
                    {"domain": r.randbytes(64).hex()}, {"parameters": r.randbytes(64).hex()},
                    {"direction": (EGRESS if i == 0 else INGRESS).value}, {"layout": lay | {f: lay[f] + 1}}])
            regs = regs[:i] + [doc] + regs[i + 1:]
        recs = [sign(on, d, w, r.randbytes(64)) for on, d in records(link)]
        if aimed and m == "past-slots":
            recs, regs = [], []
        if aimed and m.startswith("cert-"):
            on, d = records(link)[j]
            if m == "cert-missing":
                recs = r.choice([recs[:j] + recs[j + 1:], []])
            elif m == "cert-swap":
                recs = [recs[x] for x in r.choice(SWAPS)]
            elif m == "cert-fails-link":
                recs[2] = sign(r.choice([link, f"{link}-fail", f"{link}x-fails"]), EGRESS, w, recs[2].digest)
            elif m == "cert-direction":
                recs[j] = sign(on, INGRESS if d is EGRESS else EGRESS, w, recs[j].digest)
            elif m == "cert-forged":
                sig = recs[j].signature
                recs[j] = dataclasses.replace(recs[j], signature=r.choice([_flip(sig, seed), sig[:-1], b""]))
            elif m == "cert-window":
                recs[j] = sign(on, d, w + 1, recs[j].digest)
            elif m == "cert-session":
                recs[j] = sign(on, d, w, recs[j].digest, s=_flip(session, seed))
            elif m == "cert-key":
                recs[j] = sign(on, d, w, recs[j].digest, k=_key(ki, 1, r.random() < 0.5))
        entries = [{"domain": x["domain"], "root": x["root"], "value": x["value"]} for x in named]
        entries.insert(r.randrange(len(entries) + 1),
                       {"domain": r.randbytes(64).hex(), "root": r.randbytes(64).hex(), "value": "decoy"})
        T = sched.window
        units = 2 * T + (1 if aimed and m == "units" else 0)
        strata = [{"k": r.randint(1, T), "name": d.value, "size": T} for d in (INGRESS, EGRESS)] \
            if r.random() < 0.5 and units == 2 * T else None
        k = sum(s["k"] for s in strata) if strata else r.randint(1, units)
        law = {"k": k, "law": "subset"} if strata is None else {"k": k, "law": "stratified", "strata": [
            {"k": s["k"], "template": s["name"], "units": [[j * T, (j + 1) * T]]} for j, s in enumerate(strata)]}
        fmt, where = r.choice(FORMATS), r.choice(["roots", "registered"])
        entries += _prog_entries(rp, ties[w], where, untie if aimed else None)
        rp.shuffle(entries)
        rec = {"format": fmt, "law": law, "partition": f"{link}/scan/{w}", "population": units, "program": program,
               "protocol": ONE_STAGE, where: entries, "window": {"kind": "stand-in", "window": w}}
        if aimed and m == "program":
            rec["program"] = r.choice([hashlib.sha512(b"another program").hexdigest(), program.upper(), None])
        digest = registration_digest(rec)
        answer = _accepted(digest, program, rec["partition"], units, k,
                           r.choice([d for d in DELTAS if d != delta]) if aimed and m == "delta" else delta,
                           r.randint(0, units), strata)
        if aimed:
            if m == "answer-protocol":
                answer["protocol"] = "verity/sampled-proofs/two-stage/v1"
            elif m == "answer-other":
                answer["commitment"] = r.choice([r.randbytes(64), registration_digest(rec | {"population": 1})]).hex()
            elif m == "answer-reject":
                answer = _refused(answer, REJECT)
            elif m in ("answer-undecided", "undecided-and-reject"):
                answer = _refused(answer, UNDECIDED)
            elif m == "answer-profile":
                lv = answer["profile"]["levels"][0]
                answer["profile"]["levels"] = r.choice([
                    [lv, lv | {"name": "token", "size": 3}], [lv | {"draw": {"bernoulli": "1/2"}}],
                    [lv | {"draw": None}], []])
            elif m == "answer-bound":
                answer["public"] = r.choice([{"drawn": []}, answer["public"] | {"bound": -1},
                                             answer["public"] | {"bound": "3"}, []])
        window_regs.append(rec)
        shown = {"clock": _cert_json(recs[1])} if aimed and m == "form" else {"registrations": regs}
        windows.append({"answer": answer, "certificates": [_cert_json(c) for c in recs]} | shown)
    for w in range(first, first + n_windows) if not rows else ():
        rec = {"format": r.choice(FORMATS), "law": {"k": 1, "law": "subset"}, "partition": f"{link}/{w}",
               "population": 1, "program": program, "protocol": ONE_STAGE, "window": served(w)}
        sync = ClockSync(r.choice(deltas), r.choice(rhos))
        aimed = w == first + t
        if aimed and m == "program":
            rec["program"] = r.choice([hashlib.sha512(b"another program").hexdigest(), program.upper(), None])
        elif aimed and m == "provenance":
            rec["window"] = r.choice([served(w + 1), {"kind": "batch", "window": w}, served(w) | {"run": 0}, None])
        digest = registration_digest(rec)
        egress = sign(link, EGRESS, w, r.randbytes(64))
        ingress = sign(link, INGRESS, w, r.randbytes(64))
        clock = sign(link, EGRESS, w, clock_commitment(r.randbytes(SALT_BYTES), sync))
        answer = _accepted(digest, program, rec["partition"], 1, 1, r.choice(["1/100", "1/1000000"]), 0)
        if aimed:
            if m == "cert-forged":
                i = r.randrange(2)
                c = (egress, ingress)[i]
                sig = r.choice([_flip(c.signature, seed), c.signature[:-1], b""])
                forged = dataclasses.replace(c, signature=sig)
                egress, ingress = (forged, ingress) if i == 0 else (egress, forged)
            elif m == "cert-window":
                egress = sign(link, EGRESS, w + 1, egress.digest)
            elif m == "cert-link":
                ingress = sign(link + "x", INGRESS, w, ingress.digest)
            elif m == "cert-session":
                egress = sign(link, EGRESS, w, egress.digest, s=_flip(session, seed))
            elif m == "cert-key":
                ingress = sign(link, INGRESS, w, ingress.digest, k=_key(ki, 1, r.random() < 0.5))
            elif m == "clock-ingress":
                clock = sign(link, INGRESS, w, clock.digest)
            elif m == "clock-forged":
                sig, dig = clock.signature, clock.digest
                clock = r.choice([dataclasses.replace(clock, signature=s) for s in (_flip(sig, seed), sig[:-1], b"")]
                                 + [dataclasses.replace(clock, digest=d) for d in (_flip(dig, seed), dig[:-1])])
            elif m == "clock-window":
                clock = sign(link, EGRESS, w + 1, clock.digest)
            elif m == "clock-session":
                clock = sign(link, EGRESS, w, clock.digest, s=_flip(session, seed))
            elif m == "clock-reused":
                clock = sign(link, EGRESS, w, r.choice(clocks).digest)
            elif m == "clock-unsalted":
                clock = sign(link, EGRESS, w, clock_digest(sync))
            elif m == "answer-protocol":
                answer["protocol"] = "verity/sampled-proofs/two-stage/v1"
            elif m == "answer-other":
                answer["commitment"] = r.choice([r.randbytes(64), registration_digest(rec | {"population": 2})]).hex()
            elif m == "answer-reject":
                answer = _refused(answer, REJECT)
            elif m in ("answer-undecided", "undecided-and-reject"):
                answer = _refused(answer, UNDECIDED)
            elif m == "answer-profile":
                answer = _accepted(digest, program, rec["partition"], *r.choice([(2, 1), (2, 2), (1, 0)]), "1/100", 0)
                if r.random() < 0.3:
                    answer["profile"]["delta"] = "2/200"
        certificates = [egress, ingress]
        if aimed and m == "cert-swap":
            certificates = [ingress, egress]
        elif aimed and m == "cert-count":
            certificates = r.choice([[egress], [egress, ingress, ingress], []])
        window_regs.append(rec)
        clocks.append(clock)
        shown = {"registrations": []} if aimed and m == "form" else {"clock": _cert_json(clock)}
        windows.append({"answer": answer, "certificates": [_cert_json(c) for c in certificates]} | shown)

    audit_params, audit_regs, audits = [], [], []
    for i in range(n_audits):
        undrawn = i == a and m == "tied-undrawn"
        units = r.randint(2 if undrawn else 1, 40)
        n_prog = units + (r.choice([1, -1]) if units > 1 else 1) if i == a and m == "audit-units" else units
        split = r.randint(1, n_prog - 1) if n_prog > 1 and r.random() < 0.3 else None
        outs = r.sample(UNIT_BYTES, 1 if split is None else 2)
        desc = json.loads(audit_descriptor(((n_prog, outs[0]),) if split is None else
                                           ((split, outs[0]), (n_prog - split, outs[1]))))
        prog = program_sha512(desc)
        k, strata = r.randint(1, units - 1) if undrawn else units if i in tied else r.randint(1, units), None
        law = {"k": k, "law": "subset"}
        if r.random() < 0.5 and not undrawn:
            cuts = [0, *([r.randint(1, units - 1)] if units > 1 and r.random() < 0.5 else []), units]
            law = {"k": k, "law": "stratified", "strata": [
                {"k": stratum_k(k, hi - lo, units), "template": f"s{j}", "units": [[lo, hi]]}
                for j, (lo, hi) in enumerate(zip(cuts, cuts[1:]))]}
            strata = [{"k": s["k"], "name": s["template"], "size": s["units"][0][1] - s["units"][0][0]}
                      for s in law["strata"]]
            k = sum(s["k"] for s in strata)
        rec = {"format": r.choice(FORMATS), "law": law, "partition": f"loop/{i}", "population": units,
               "program": prog, "protocol": ONE_STAGE, "window": {"kind": "served", "window": i}}
        if rows:
            rec["roots"] = _output_entries(rp, i, ties, sched.window, first + t, untie)
        if i == a and m == "audit-program":
            if r.random() < 0.5:
                rec["program"] = r.choice([hashlib.sha512(b"another program").hexdigest(), prog.upper(), None])
            else:
                desc = json.loads(audit_descriptor(((n_prog, r.choice([o for o in UNIT_BYTES if o != outs[0]])),)))
        digest = registration_digest(rec)
        answer = _accepted(digest, prog, rec["partition"], units, k, r.choice(["1/100", "1/1000000", "1/10"]),
                           r.randint(0, units), strata)
        if i == a:
            if m == "audit-other":
                answer["commitment"] = r.randbytes(64).hex()
            elif m in ("audit-reject", "undecided-and-reject"):
                answer = _refused(answer, REJECT)
            elif m == "audit-undecided":
                answer = _refused(answer, UNDECIDED)
            elif m == "audit-profile":
                lv = answer["profile"]["levels"][0]
                answer["profile"]["levels"] = r.choice([
                    [lv, lv | {"name": "token", "size": 3}], [lv | {"draw": {"bernoulli": "1/2"}}], [lv | {"draw": None}],
                    []])
            elif m == "audit-bound":
                answer["public"] = r.choice([{"drawn": []}, answer["public"] | {"bound": -1},
                                             answer["public"] | {"bound": "3"}, []])
        audit_params.append({"free": r.randint(0, 5), "program": desc, "steps": r.randint(0, 4)})
        audit_regs.append(rec)
        audits.append(answer)

    case_params = {"audits": audit_params, "cells": [r.randint(0, 100) for _ in range(r.randint(0, 3))], "first": first,
                   "jitter": r.randint(1, 5), "key": lms.public_key(key).hex(), "link": link,
                   "lms": {"H": H, "N": N, "W": W, "toy": toy}, "program": program, "session": session.hex(),
                   "windows": n_windows}
    if rows:
        case_params |= {"scan": scan, "schedule": schedule_json(bound)}
    else:
        case_params["sync_set"] = {"deltas": deltas, "rhos": [[x.numerator, x.denominator] for x in rhos]}
    c_json = {"audits": audit_regs, "windows": window_regs}
    e_json = {"audits": audits, "windows": windows}
    if m == "first":
        case_params["first"] = first + r.choice([1, -1] if first else [1])
    elif m == "count-windows":
        case_params["windows"] = n_windows + r.choice([1, -1] if n_windows > 1 else [1])
    elif m == "count-audits":
        case_params["audits"] = r.choice([audit_params[1:], [*audit_params, {
            "free": 0, "program": json.loads(audit_descriptor(((1, 2),))), "steps": 0}]])
    elif m == "evidence-count":
        e_json = r.choice([{"audits": audits, "windows": windows[1:]}, {"audits": audits, "windows": windows + windows[:1]},
                           {"audits": [*audits, audits[0] if audits else windows[0]["answer"]], "windows": windows}])
    c_bytes, e_bytes = D.canonical(c_json).encode(), D.canonical(e_json).encode()
    if m == "params":
        case_params = r.choice(_bad_params(case_params, seed))
    elif m == "commitment":
        c_bytes = r.choice(_bad_commitments(c_json, c_bytes))
    elif m == "evidence":
        e_bytes = r.choice(_bad_evidence(e_json, e_bytes))
    case = {"commitment": c_bytes.hex(), "evidence": e_bytes.hex(), "params": case_params}
    if m == "case":
        case = r.choice([case | {"commitment": case["commitment"].upper()}, case | {"evidence": case["evidence"] + "0"},
                         {k: v for k, v in case.items() if k != "params"}, [case]])
    return case


def _bad_params(p: dict, n: int) -> list:
    """Parameters `params` refuses, each one change to `p`."""
    out = [{k: v for k, v in p.items() if k != sorted(p)[n % len(p)]}, p | {"x": 0}, [p], p | {"windows": 0},
           p | {"jitter": 0}, p | {"jitter": True}, p | {"link": "x" * (1 << 16)}, p | {"link": 1},
           p | {"key": p["key"][:-2]}, p | {"key": p["key"] + "00"}, p | {"key": p["key"].upper()},
           p | {"session": p["session"][:-2]}, p | {"session": ""}, p | {"program": p["program"][:-2]},
           p | {"program": p["program"].upper()}, p | {"program": 0},
           p | {"lms": p["lms"] | {"W": 3}}, p | {"lms": p["lms"] | {"H": 0}}, p | {"lms": p["lms"] | {"toy": 0}},
           p | {"lms": p["lms"] | {"x": 0}}, p | {"lms": {k: v for k, v in p["lms"].items() if k != "toy"}},
           p | {"cells": [-1]}, p | {"cells": [True]}, p | {"first": -1}, p | {"first": True}, p | {"first": "0"},
           *bad_audits(p)]
    if "sync_set" in p:
        ss = p["sync_set"]
        return out + [p | {"sync_set": ss | {"deltas": []}}, p | {"sync_set": ss | {"rhos": [[0, 1]]}},
                      p | {"sync_set": ss | {"rhos": [[1, 0]]}}, p | {"sync_set": ss | {"deltas": [1, 1]}},
                      p | {"sync_set": ss | {"rhos": [[1, 2], [2, 4]]}}, p | {"sync_set": ss | {"x": 0}}]
    s = p["schedule"]
    return out + [p | {"schedule": {k: v for k, v in s.items() if k != SCHEDULE_KEYS[n % len(s)]}},
                  p | {"schedule": s | {"x": 1}}, p | {"schedule": [s]}, p | {"schedule": s | {"delay": 0}},
                  p | {"schedule": s | {"lanes": True}}, p | {"schedule": s | {"window": -1}},
                  p | {"schedule": s | {"queue": "1"}}, p | {"schedule": s | {"cell_bytes": 10}},
                  p | {"schedule": s | {"cell_bytes": U32}}, p | {"schedule": s | {"lanes": U32}},
                  p | {"schedule": s | {"window": U32}}, p | {"schedule": s | {"every": s["window"] + 1}},
                  p | {"schedule": s | {"every": 0}}, p | {"sync_set": {"deltas": [0], "rhos": [[1, 1]]}},
                  *bad_scans(p)]


def bad_audits(p: dict) -> list[dict]:
    """Parameters whose audits `auditOf` refuses, each one change to `p`: an audit without its monitoring period, with a
    negative free count, not a list, and an audit whose Program isn't a descriptor, is of another schema, or has no
    template in its root (the stand-in unit alone)."""
    good = json.loads(audit_descriptor(((1, 2),)))
    bare = json.loads(D.canonical(codec.encode_program(Program(bind(AUDIT_UNIT, OUT=2)))))
    one = lambda **kw: p | {"audits": [{"free": 0, "program": good, "steps": 0} | kw]}  # noqa: E731
    return [p | {"audits": [{"free": 0, "program": good}]}, one(free=-1), p | {"audits": "x"}, one(program="x"),
            one(program=good | {"schema": "verity-ir/v0"}), one(program=bare)]


def _scan_widths(desc: dict) -> tuple[int, int, int, int]:
    """`scan_descriptor`'s widths of a generator-form descriptor it wrote: the ingress templates' state, prog and view
    bits, and the egress template's state words."""
    sig = {tuple(k for k, _ in d["body"]["params"]): d["body"] for d in desc["definitions"].values()
           if d["form"] == "composite"}
    first, egress = sig[FIRST_PARAMS], sig[EGRESS_PARAMS]
    _, state, view = first["ret_type"][1]
    return state[1], first["params"][1][1][1], view[1], egress["params"][0][1][1]


def bad_scans(p: dict) -> list[dict]:
    """Rows-form parameters whose scan `scanOf` refuses, each one change to `p`: not a descriptor, another Program's
    (its SHA-512 not `program`), and, each under its own SHA-512, the checker form (its egress row an input), templates
    with other parameters, another number of instances, rows not the schedule's, an input gate no template's, and each
    negative of a check of what `NetTiming.Layout.Fits` asks (`scan_negatives`)."""
    s, desc = p["schedule"], p["scan"]
    T, row, widths = s["window"], s["lanes"] * s["slots"] * s["cell_bytes"], _scan_widths(desc)

    def own(text: str) -> dict:
        d = json.loads(text)
        return p | {"scan": d, "program": program_sha512(d)}

    extra = json.loads(D.canonical(desc))
    extra["types"].append(["Array", 1, ["Value", 1]])
    extra["definitions"][extra["root"]]["body"]["nodes"].append(
        {"args": [], "axes": [], "fn": extra["callees"].index("Input1_v1"), "form": "batch", "n": 1,
         "out": len(extra["types"]) - 1})
    return [p | {"scan": "x"}, p | {"scan": {"free": 0, "steps": 0}}, p | {"scan": desc | {"schema": "verity-ir/v0"}},
            p | {"scan": json.loads(scan_descriptor(T, row, (widths[0] + 1, *widths[1:])))},
            own(scan_descriptor(T, row, widths, checker=True)),
            own(scan_descriptor(T, row, widths, names=(("row", "seed"), INGRESS_PARAMS, EGRESS_PARAMS))),
            *([own(scan_descriptor(T, row, widths, names=(FIRST_PARAMS, ("row", "sin", "seed"), EGRESS_PARAMS)))]
              if T > 1 else []),
            own(scan_descriptor(T, row, widths, names=(FIRST_PARAMS, INGRESS_PARAMS, ("words",)))),
            own(scan_descriptor(T, row, widths, counts=(1, T - 1, T + 1))),
            own(scan_descriptor(T, row, widths, counts=(2, T - 1, T))),
            own(scan_descriptor(T, row, widths, counts=(1, T, T))),
            own(scan_descriptor(T, row + 1, widths)),
            p | {"scan": extra, "program": program_sha512(extra)},
            *(q for q, _ in scan_negatives(p).values())]


#: A primitive that is no input, a constant gate the `row` negative binds bucket 0's row to.
ZERO = "NetworkPeriodScanZero_v1"
#: Members of each of the `walk` negative's nested batches: three deep, past `Flock.Steps.FUEL`'s walk.
DEEP = 2 ** 30


def scan_negatives(p: dict) -> dict[str, tuple[dict, str]]:
    """Rows-form parameters each refused by one of `scanOf`'s checks of what `NetTiming.Layout.Fits` asks and by none
    before it, each one change to `p`'s generator-form scan under its own SHA-512, with a fragment of both refusals:
    `walk`, the egress template calling a composite of `2^90` gates in three nested batches, unread, so the walk to an
    operand is past `FUEL`; `returns`, the root returning an input leaf for the last egress leaf; `computes`, the egress
    template returning its `state`'s first 16 leaves as its u16; `row`, bucket 0's row bound to a constant primitive's
    gates (an input node like it appended, keeping the input count); `argument`, egress member 0's first state leaf the
    last ingress template's first output leaf (a later step's ingress when `T > 1`); `ingress`, the egress state the
    last ingress template's row input node (resized, the state's node resized to keep the count); and `topological`,
    the last ingress template's row input node moved after the templates, so the template names a later node."""
    desc = p["scan"]

    def parts() -> tuple[dict, list, list[int]]:
        d = json.loads(D.canonical(desc))
        nodes = d["definitions"][d["root"]]["body"]["nodes"]
        ts = [k for k, x in enumerate(nodes) if d["definitions"][d["callees"][x["fn"]]]["form"] != "primitive"]
        return d, nodes, ts

    def own(d: dict) -> dict:
        return p | {"scan": d, "program": program_sha512(d)}

    def typed(d: dict, t: list) -> int:
        d["types"].append(t)
        return len(d["types"]) - 1

    out = {}
    d, nodes, ts = parts()
    v, mix = typed(d, ["Value", 1]), d["callees"].index(MIX.id)
    arr = typed(d, ["Array", DEEP, ["Value", 1]])
    one = {"refs": {"affine": ["p", 0, 0, 1]}, "type": v}
    for i in range(1, 4):
        fid = f"NetworkPeriodScanDeep{i}_v0"
        d["definitions"][fid] = {"bindings": {}, "form": "composite", "id": fid, "body": {
            "nodes": [{"args": [one, one][:3 - min(i, 2)], "axes": [None, None][:3 - min(i, 2)], "fn": mix,
                       "form": "batch", "n": DEEP, "out": arr}],
            "params": [["a", ["Value", 1]]], "ret": {"affine": ["n", 0, 0, 1]}, "ret_type": ["Value", 1]}}
        d["callees"].append(fid)
        mix = len(d["callees"]) - 1
    egress = d["definitions"][d["callees"][nodes[ts[-1]]["fn"]]]["body"]
    egress["nodes"].append({"args": [one], "fn": mix, "form": "call", "out": v})
    out["walk"] = own(d), "past what its resolution walks"

    d, nodes, ts = parts()
    r = d["definitions"][d["root"]]["body"]
    ret = _parts(r["ret"], nodes)
    i = next(i for i, q in enumerate(ret) if (_affine(q) or ())[:2] == ("n", ts[-1]))
    s, k, b, c = ret[i]["affine"]
    r["ret"] = {"concat": [*ret[:i], {"affine": [s, k, b, c - 1]}, {"affine": ["n", 0, 0, 1]}, *ret[i + 1:]]}
    out["returns"] = own(d), "the root doesn't return every egress leaf"

    d, nodes, ts = parts()
    egress = d["definitions"][d["callees"][nodes[ts[-1]]["fn"]]]["body"]
    egress["ret"] = {"concat": [{"affine": ["p", 0, 0, OUT_BITS]}, *_parts(egress["ret"], egress["nodes"])[1:]]}
    out["computes"] = own(d), "returns a leaf it doesn't compute"

    d, nodes, ts = parts()
    k0 = nodes[ts[0]]["args"][0]["refs"]["affine"][1]
    d["definitions"][ZERO] = {"form": "primitive", "id": ZERO, "params": [], "ret": ["Value", 1]}
    d["callees"].append(ZERO)
    nodes.append(dict(nodes[k0]))
    nodes[k0] = nodes[k0] | {"fn": len(d["callees"]) - 1}
    out["row"] = own(d), "an ingress template's row is not one input node's leaves"

    d, nodes, ts = parts()
    a = nodes[ts[-1]]["args"][0]
    s, k, b, c = a["refs"]["affine"]
    a["refs"] = {"concat": [{"affine": ["n", ts[-2], 0, 1]}, {"affine": [s, k, b + 1, c - 1]}]}
    out["argument"] = own(d), "a template's argument is not one input node's leaves"

    d, nodes, ts = parts()
    kr, ks = (nodes[k]["args"][0]["refs"]["affine"][1] for k in ts[-2:])
    R, S = nodes[kr]["n"], nodes[ks]["n"]
    nodes[kr] = nodes[kr] | {"n": max(R, S), "out": typed(d, ["Array", max(R, S), ["Value", 1]])}
    nodes[ks] = nodes[ks] | {"n": min(R, S), "out": typed(d, ["Array", min(R, S), ["Value", 1]])}
    nodes[ts[-1]]["args"][0]["refs"] = {"affine": ["n", kr, 0, S]}
    out["ingress"] = own(d), "an egress template's argument is an ingress row's input node"

    d, nodes, ts = parts()
    kr = nodes[ts[-2]]["args"][0]["refs"]["affine"][1]
    at = lambda i: len(nodes) - 1 if i == kr else i - (i > kr)  # noqa: E731

    def moved(r: dict) -> dict:
        if "concat" in r:
            return {"concat": [moved(q) for q in r["concat"]]}
        if "alias" in r:
            return {"alias": [at(r["alias"][0]), r["alias"][1]]}
        if "explicit" in r:
            return {"explicit": [[s, i if s == "p" else at(i), x] for s, i, x in r["explicit"]]}
        key = "affine" if "affine" in r else "strided"
        s, i, *rest = r[key]
        return {key: [s, i if s == "p" else at(i), *rest]}
    r = d["definitions"][d["root"]]["body"]
    for x in nodes:
        for a in x.get("args", ()):
            a["refs"] = moved(a["refs"])
    r["ret"] = moved(r["ret"])
    nodes.append(nodes.pop(kr))
    out["topological"] = own(d), "a body names a later node"
    return out


def _bad_commitments(c: dict, b: bytes) -> list[bytes]:
    """Commitments that don't decode: bytes that aren't canonical JSON, or a registration without a known format."""
    enc = lambda j: D.canonical(j).encode()  # noqa: E731
    return [b + b" ", b[:-1], b"", json.dumps(c).encode(), enc(c | {"x": 1}), enc({"audits": c["audits"]}),
            enc(c | {"windows": [1]}), enc(c | {"windows": [{"format": "verity/registration/v3"}]}),
            enc(c | {"audits": [{"format": 1}]}), enc(c | {"audits": "x"}), enc([c])]


def _bad_evidence(e: dict, b: bytes) -> list[bytes]:
    """Evidence that doesn't decode, each one change to `e`."""
    enc = lambda j: D.canonical(j).encode()  # noqa: E731
    x, rest = e["windows"][0], e["windows"][1:]
    cert = x["certificates"][0]
    bad_windows = [x | {"certificates": [cert | {"direction": "up"}]}, x | {"certificates": [cert | {"key_id": "abc"}]},
                   x | {"certificates": [cert | {"window": -1}]}, x | {"certificates": [cert | {"x": 0}]},
                   x | {"certificates": "x"}, x | {"answer": {k: v for k, v in x["answer"].items() if k != "reason"}},
                   x | {"answer": x["answer"] | {"verdict": "maybe"}}, x | {"answer": x["answer"] | {"reason": "r"}}]
    if "clock" in x:
        clock = x["clock"]
        bad_windows += [{k: v for k, v in x.items() if k != "clock"},
                        x | {"clock": {"certificate": clock, "sync": [0, 1, 1]}}, x | {"clock": clock | {"x": 0}},
                        x | {"clock": {k: v for k, v in clock.items() if k != "signature"}},
                        x | {"clock": clock | {"direction": "up"}}, x | {"clock": [clock]}]
    else:
        doc = x["registrations"][0]
        lay, c = doc["layout"], doc["certificate"]
        bad_rows = [{k: v for k, v in doc.items() if k != "root"}, doc | {"x": 0}, doc | {"layout": lay | {"x": 0}},
                    doc | {"layout": {k: v for k, v in lay.items() if k != "slots"}}, doc | {"layout": [lay]},
                    doc | {"layout": lay | {"buckets": -1}},
                    doc | {"certificate": {k: v for k, v in c.items() if k != "key"}},
                    doc | {"certificate": c | {"signature": c["signature"] + "0"}}, doc | {"certificate": "x"},
                    doc | {"root": doc["root"][:-1]}, doc | {"root": "g" + doc["root"][1:]}, doc | {"wiring": 0},
                    doc | {"window": "0"}, doc | {"window": -1}, doc | {"cell_bytes": True}, doc | {"format": 1},
                    doc | {"domain": 0}, "x"]
        bad_windows += [{k: v for k, v in x.items() if k != "registrations"}, x | {"clock": cert},
                        x | {"registrations": "x"}, x | {"registrations": [1]}] + \
            [x | {"registrations": [d, *x["registrations"][1:]]} for d in bad_rows]
    return [b[:-1], b"\xff" + b, json.dumps(e).encode(), enc(e | {"x": 1}), enc({"windows": e["windows"]}),
            enc(e | {"audits": [1]})] + [enc(e | {"windows": [w, *rest]}) for w in bad_windows]


answer = reference
"""The period check's outcome on one case: `reference` under the name a composition's components answer by."""

#: What a case can be aimed at: each verdict, and a case that fails to decode (its params, commitment or evidence).
KINDS = (ACCEPT, REJECT, UNDECIDED, "broken-params", "broken-commitment", "broken-evidence")
#: Each form's changes (`build`'s `m`) by the kind they aim at: a rejection by the check itself, not by decoding, for
#: `reject`; `case` breaks the line `serve` reads rather than the period, so no kind uses it.
AIMS: dict[str, dict[str, tuple[str, ...]]] = {
    form: {
        ACCEPT: tuple(m for m in ms if m.removeprefix("rows-") == "honest"),
        REJECT: tuple(m for m in ms if EXPECT[m] == REJECT
                      and m.removeprefix("rows-") not in ("params", "commitment", "evidence", "case")),
        UNDECIDED: tuple(m for m in ms if EXPECT[m] == UNDECIDED),
        **{f"broken-{x}": tuple(m for m in ms if m.removeprefix("rows-") == x)
           for x in ("params", "commitment", "evidence")},
    } for form in FORMS for ms in [[m for m in EXPECT if form_of(m) == form]]
}


def verdict_of(kind: str) -> str:
    """The verdict a case aimed at `kind` gets: a case that fails to decode is rejected."""
    return kind if kind in (ACCEPT, REJECT, UNDECIDED) else REJECT


def case(kind: str, seed: int = 0, form: str = "clock") -> dict:
    """A small period of the form (`build(..., small=True)`: the toy key, as few link-windows and audits as its change
    needs) aimed at `kind`, its change and every choice drawn from `seed`."""
    ms = AIMS[form][kind]
    return build(ms[random.Random(f"network-period {kind} {seed}").randrange(len(ms))], seed, small=True)


def cases(seed: int = 0, form: str = "clock") -> dict[str, tuple[dict, dict]]:
    """One small period of the form of each kind with its answer, `{kind: (case, answer)}`."""
    return {k: (c, answer(c)) for k in KINDS for c in [case(k, seed, form)]}
