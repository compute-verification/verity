"""The network certifier's period check (`NetworkCheck.Period`, `verity/certifier/v0`) in Python, for every difftest
that runs it: its mirror, `answer` (as `reference`), and its cases, `build` and `case`.

`answer(case)` is the outcome of one `serve` line in the core interface's answer format
(`verity_difftest.protocol.answer`): it decodes the protocol's formats (`NetworkCheck/Period.lean`'s module docstring)
and checks a period with the Python the period's pieces already have: one-stage's answers and integrity profiles as PoUW
reads them (`pouw.core_check.answer_of` and `integrity`), the network certifier's certificates and clock certificates
under its LMS key (`certifier.certificate`), Σ_sync (`certifier.schedule`), one-stage's `registration.digest` (a
registration record is ASCII-only, where its canonical JSON is Lean's), and in the rows form
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
(`warden_rows.Cells.commit` and `registration`, signed by the same key), whose caps the parameters' `cells` are
(`WardenParams.caps`; `bad_caps` are refused), its three records certified (`past-slots`:
none of them and neither registration, as the warden leaves a window in which it refused a connection), and the scan's
registration holding both rows at their ports (`rows_name`, of T leaves) in `roots` or at their values in
`registered`, their domains each naming one tree (`_holds`; `port`: the egress rows' root at `scan/view` and their
port at another, a second tree under one end's domain, or the like, `UNPORTS`), and each lane j's `scan/prog<j>`
(`prog_name`) at the root and domain of chunk j's entries port of that link-window's slot entries' template (`slot_entry`, `io_shell.SlotEntry_v1` at its
smallest circuit-check row, or `select_entry`, `SelectEntry_v1` at its smallest binding beside a stand-in scan slots'
`ScanSlots_v2`, each a pinned Definition with one member a slot) in its own audit's registration and Program
(`_prog_tied`, `_wired`), and its
state (`STATE`) at the root and domain of that audit's `SCAN_STATE` (`_state_tied`), a state tree of T rows (`_state_tree`)
whose units' reads its `window` states as the chain (`state_reads`, `_state_read`) and whose leaf 0 it opens to the
empty state (`state_empty`, `_state_empty`), answered by one-stage with a
stratified or subset draw at the period's δ; the scan's Program is its generator form, stand-ins of its three templates
at small widths in the Boolean profile's types, as the scan's Definitions have them (`scan_descriptor`), and a Program in
the checker form is a bad scan (`bad_scans`), as is each check's negative of what `NetTiming.BoundaryGates.Fits` asks
(`scan_negatives`). Beside them, the compute audits (in the rows form at least one, whose registrations' `roots` hold the
scans' `prog`), each registered over its Program, a population of one or two stand-in templates whose units commit from
8 to 320 bits, with its slot entries' in the rows form (`audit_descriptor`), and answered with a subset or stratified
draw (a work law's) and its bound on incorrect proof blocks, every unit drawn for an audit a scan's `prog` is tied to,
or, each accepted and charged, its slot entries' and scan slots' templates whole and the rest in part (`tied-partial`, a
stratum a template, `_by_template`), one or two of its proof blocks sampled (`tied-sampled`, `subset:k`, the demo's
sample), or its slot entries' or scan slots' stratum in part too (`tied-entries-sampled`, `tied-slots-sampled`), and a
link-window whose scan's state is not its
wired audit's `SCAN_STATE` refused (`state-untied`: a `SCAN_STATE` that skips or alters one of the scan's transitions,
at another root or domain, or none, `UNSTATES`), as is one whose units read their state off the chain
(`state-misindexed`, `MISREADS`) and one whose row 0 isn't opened to the empty state (`state-unpinned`, `UNPINS`).
The tied audits run one class circuit, a Program drawn once, whose SHA-512 is the parameters' `circuit` (`_held`), and
a phase whose tied audit runs another, or under another `circuit`, is refused (`circuit-swapped`, `UNHOLDS`). The
tied audits may be in the record form, each slot reading the scan's state at its anchor's row in the class circuit's
record (`state-anchored`, `anchor_audit`), and a phase whose tied audit reads it elsewhere is refused
(`state-misanchored`, `MISANCHORS`). Each is charged its free inputs with those reads held where its `SCAN_STATE` is
the scan's state in every phase tied to it (`_charge`); a second audit tied to a phase, whose `SCAN_STATE` is another
link-window's or none's, is charged them free (`state-unheld`). In the
clock form an audit may be in the record form (`audit-record`): its window's circuit record, wire order and the
verifier's pins beside its Program (`record_audit`), its registration's window binding the record and recording each
pin's root, a record whose wire reads a Boolean-profile writer's bits as 16-bit leaves accepted too
(`audit-record-bits`); refused where the window binds another record (`audit-record-circuit`) or records no pin's root
(`audit-record-pin`), or a pin is of `SCAN_STATE` (`audit-record-state`). `case(kind, seed, form)` is a small period of
the form aimed at a
verdict, or at a decoding failure (`KINDS`, those the form reaches: `kinds`), and `cases(seed, form)` one of each with
its answer, as a composition's difftest takes a real component's cases.

This is test-side Python (it imports `verity_difftest` and PoUW's `core_check`, which the certifier's package may not),
so it lives beside the period's difftest (`test_network_period_protocol.py`), and other suites load it by path."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import re
from collections.abc import Callable
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import verity_difftest as D
from verity_difftest import protocol as P
from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.certificate import (SALT_BYTES, Certificate, Direction, LmsKey,
                                                            certificate_digest, clock_digest)
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.certifier.warden import WardenParams
from verity.core.protocols import profile as PR
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.sampled_proofs.one_stage import registered as G, registration as R
from verity.core.protocols.sampled_proofs.one_stage.draw import stratum_k
from verity.ml.circuits import boolean as BL, codec, metadata as AN, population
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.defs import REGISTRY
from verity.ml.circuits.ops import gemm as GM
from verity.ml.circuits.ops.boolean import elementwise as BE
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import array_of, coll_concat, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.commitments import advice as ADV
from verity.ml.commitments.frame_v3 import BitRowLeaf
from verity.ml.commitments.identity import tagged_sha512
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.crypto import lms
from verity.ml.definitions.registry import io_shell
from verity.ml.definitions.registry.sampling import bit_at
from verity.ml.windows import interface as IF

ID = "verity/certifier/v0"
COMMITMENT_TAG = "verity-network-certifier/period/v0"
ONE_STAGE = R.PROTOCOL
FORMATS = (R.FORMAT, R.FORMAT_V2)
#: a proof block's public outputs are at most 16 bits (the values it commits can be wider)
OUT_BITS = 16
UNIT = "unit"
EGRESS, INGRESS = Direction.EGRESS, Direction.INGRESS
ACCEPT, REJECT, UNDECIDED = "accept", "reject", "undecided"
PARAMS_KEYS = ["audits", "cells", "first", "jitter", "key", "link", "lms", "program", "session", "sync_set", "windows"]
ROWS_PARAMS_KEYS = ["audits", "cells", "circuit", "first", "jitter", "key", "link", "lms", "program", "scan",
                    "schedule", "session", "windows"]
#: `rowsListedKeys`: the rows form's, with the scan's listed circuit files
ROWS_LISTED_KEYS = sorted([*ROWS_PARAMS_KEYS, "scan_circuits"])
#: `Flock.HmRow.PARTITION_V1`: the partition rule a circuit record's digest and a listed file's META partition are under
PARTITION_V1 = "verity/partition/v1"
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
#: the egress bucket's parameters in the generator form (`egress.EgressBucket`)
EGRESS_PARAMS = ("state",)
#: a byte and a u16 in the Boolean profile, as the scan's Definitions type a row's bytes and the egress bucket's word
BOOL_BYTE, BOOL_U16 = ["Array", 8, ["Value", 1]], ["Array", OUT_BITS, ["Value", 1]]
#: the checker form's, which took the rows its units compared as inputs
CHECKER_PARAMS = (("row", "sin", "prog", "sout", "view"), ("state", "row"))
#: `slotEntryId`: the slot entry's Definition, whose committed `o0` is the entry the scan reads at a slot
SLOT_ENTRY = "SlotEntry_v1"
#: `slotEntryPins`: the closure digests (`slot_entry_digest`) of `io_shell.SlotEntry_v1` at its circuit-check rows,
#: `SlotEntry_v1{KO=2,IDB=17,NP=4,STOP=[]}` and `SlotEntry_v1{KO=64,IDB=17,NP=1024,STOP=[128001]}`, in that order, which
#: `test_slot_entry_pins.py` recomputes from `io_shell` (its `SlotEntry_v1:4` row is no pin: at KO 4 a slot's entry is a
#: `SelectEntry_v1` unit's)
SLOT_ENTRY_PINS = ("8c8a033047ec49110114023204a9c265720ae9d1b63963c51a98e14f0314ebd3"
                   "7b741b90d6a374aeb681a1a5fe364580b68ff60cf5514fdf57fdaff88b36de30",
                   "2559f1c8e260aac6e8755c317cde07f990fcd3619dd394010d367a3b78156c2f"
                   "24b44567891a9ea2efb2f8a6830c9f9c957c14dc65041f8dd77bfd8d50c5b7ff")

#: `selectEntryId`: the slot's last position, whose select and collector end in `SlotEntry_v1`, so its `o0` is the entry
SELECT_ENTRY = "SelectEntry_v1"
#: `entryIds`: the Definitions whose `o0` is a slot's entry
ENTRY_IDS = (SLOT_ENTRY, SELECT_ENTRY)
#: The slots' occupancy's Definition (`boolean_io_shell`'s `ScanSlots_v2`), the `occ` the `SelectEntry_v1` proof blocks
#: read: a stratum the cases sample whole or in part (`_by_template`)
SCAN_SLOTS = "ScanSlots_v2"
#: `selectEntryPins`: the closure digests of `SelectEntry_v1` at
#: `SelectEntry_v1{V=256,C=16,KO=2,FIRST=false,IDB=17,NP=4,STOP=[]}`,
#: `SelectEntry_v1{V=128256,C=16,KO=4,FIRST=false,IDB=17,NP=32,STOP=[128001]}` and
#: `SelectEntry_v1{V=128256,C=16,KO=64,FIRST=false,IDB=17,NP=1024,STOP=[128001]}`, in that order, each calling the
#: `SlotEntry_v1` that writes the scan's read bit: `io_shell.SelectEntry_v1` at its circuit-check rows, which
#: `test_slot_entry_pins.py` recomputes from `io_shell`; the first is `fixtures/select_entry.json`'s
SELECT_ENTRY_PINS = ("f817250e0ac1d5028ea992bbe0c1ab12925bc1b709f020fdc54a57a0ebc0c9dc"
                     "bd6513d2bf72aa243c499833d1173fb14236e85bb27b4d03de88df93344007d2",
                     "236d569741e3fe2026aaa65be2077b4da82d495a5b35322428051e4e07ca556b"
                     "b44d1c5f59f2aef22d7c4c5b563c3819569add04ed3d2d4d57563ad471cf71c4",
                     "5460098589051bb756c5d7781e1727495952c296826b29315ec269b492ca3cea"
                     "bc394ee273a2dae92ffb8eef31d8d22e83208a9e280dd80e39be9073dfc7a68a")

# --- the mirror --------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Audit:
    """`Audit`: a compute audit's layout terms (its free inputs, its Program's input gates, and its monitoring period)
    and what its Program gives: its SHA-512, its units (its root's template instances), the most bits one of them
    commits, and its slot entries' templates (`entries`: each root template node calling `SlotEntry_v1` or
    `SelectEntry_v1`, by descriptor id, members and closure digest, `slot_entry_digest`); with its window's circuit
    record (`record`), the record's digest and the pins its count held, as (param, root) pairs, the record as
    `anchored` reads it (`circuit`), and its free inputs with its reads of the scan's state held too (`state_free`,
    `stateFree`; None for `free`, without a record)."""
    free: int
    steps: int
    program: str
    units: int
    bits: int
    entries: tuple[tuple[str, int, str], ...] = ()
    record: tuple[str, tuple[tuple[str, str], ...]] | None = None
    circuit: Stated | None = None
    state_free: int | None = None


@dataclasses.dataclass(frozen=True)
class Stated:
    """`NetworkCheck.Circuit.Stated` as `anchored` reads it: each template's input rows' words, by descriptor id and
    port, its cuts and its registered reads."""
    rows: dict[str, dict[str, int]]
    cuts: dict[str, dict[str, int]]
    registered: IF.Registered


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
    #: the rows form's class circuit, the SHA-512 of its window circuit's Program, which every tied audit runs
    #: (`_held`); None in the clock form
    circuit: str | None = None


@dataclasses.dataclass(frozen=True)
class Scan:
    """`Scan`: the scan's monitoring period (T), its own free inputs (none: its `prog` is a compute audit's slot
    entries, `_wired`, whose free inputs a link-window's term counts, `_tied_audits`), the bits one ingress and one
    egress instance commit (its parameters and its return), and its slot entries' bits (`E_w`), all a tied audit's incorrect proof blocks can set."""
    steps: int
    free: int
    ingress: int
    egress: int
    entries: int = 0


@dataclasses.dataclass(frozen=True)
class Commitment:
    """One-stage's registration records of the audits and the link-windows, as they decode."""
    audits: list[dict]
    windows: list[dict]
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
    #: `draw`: the sample's strata, each (size, k), a `subset:k` draw's one of every unit, off which `at` reads the bound
    #: at another δ; None where no draw is read (`subset 0`), at which every unit may be faulty
    strata: tuple[tuple[int, int], ...] | None = None

    def at(self, delta: Fraction) -> Facts:
        """`Facts.readAt`: the same draw's bound at `delta`, one-stage's (`boundOf`, `Stratified.count_bound`), no
        second draw. `countBound` reads no stratum's name, reads a `k` past its stratum's size as the whole stratum, and
        over no strata is 0; `Stratum` refuses the first two and `Stratified` the last."""
        strata = ((self.units, 0),) if self.strata is None else self.strata
        law = [PR.Stratum(f"s{j}", n, min(k, n)) for j, (n, k) in enumerate(strata)]
        return dataclasses.replace(self, delta=delta, wrong=PR.Stratified(law).count_bound(delta) if law else 0)


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


def rows_wf(s: WardenParams, windows: int) -> bool:
    """`rowsWf`: a cell's B bytes are under 2^32, so its payload's length is a u32, and the period's
    `lanes · windows · window` connections, the k-th at ordinal k + 1, have distinct u32 ordinals."""
    return s.cell_bytes < U32 and s.lanes * windows * s.window < U32


def rows_wf_refusal(s: WardenParams, windows: int) -> str:
    return (f"rows format: a frame's ordinal and payload length are u32s, so the lanes · windows · window = "
            f"{s.lanes * windows * s.window} connections and a cell's {s.cell_bytes} bytes must each be under 2^32, or "
            "two frames could encode alike (rowsWf)")


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


def words_of(t) -> int:
    """`SampledProofs.Bits.wordsOf`: a descriptor type's bits packed on whole 16-bit words."""
    return -(-bits_of(t) // 16)


def ret_words(t) -> int:
    """`SampledProofs.Bits.retWords`: a return's fields (a tuple's or a record's, else the return), each packed on whole
    16-bit words of its own."""
    fields = t[1] if t[0] == "Tuple" else [f[1] for f in t[1]] if t[0] == "Record" else [t]
    return sum(words_of(f) for f in fields)


def def_sig(d: dict) -> tuple[list, list]:
    """A Definition's parameters and return as `Flock.Program.decode` reads them: a primitive's own, a composite's
    body's."""
    return (d["params"], d["ret"]) if d["form"] == "primitive" else (d["body"]["params"], d["body"]["ret_type"])


def vu_bits(upv: int, words: int) -> int:
    """`vuBits`: what a wrong VU of `upv` template instances of `words` 16-bit words each is charged, and its unit's
    gates 0 and 1."""
    return upv * 16 * words + 2


def program_sha512(desc: dict) -> str:
    """`partition_object.program_sha512`: SHA-512 of the canonical descriptor without `annotations`."""
    return hashlib.sha512(D.canonical({k: v for k, v in desc.items() if k != "annotations"}).encode()).hexdigest()


def _members(x: dict) -> int:
    """`Node.members`: a batch's members or a scan's iterations, else one."""
    return x["n"] if x["form"] in ("batch", "scan") else 1


def prog_params(lanes: int) -> tuple[str, ...]:
    """`progParams`: an ingress bucket's entry ports, `prog<j>` lane j's slot entry."""
    return tuple(f"prog{j}" for j in range(lanes))


def first_params(lanes: int) -> tuple[str, ...]:
    """`firstParams`: bucket 0's parameters in the generator form (`ingress_scan.IngressScanFirst`)."""
    return ("row", *prog_params(lanes))


def ingress_params(lanes: int) -> tuple[str, ...]:
    """`ingressParams`: each later ingress bucket's parameters (`ingress_scan.IngressScanBucket`)."""
    return ("row", "sin", *prog_params(lanes))


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
    return a[1] if _is_input(nodes[a[1]], j) else None


def _is_input(x: dict, j: dict) -> bool:
    """`NciCheck.Desc.isInputNode`: a node calling an input primitive."""
    fid = j["callees"][x["fn"]]
    return j["definitions"][fid]["form"] == "primitive" and bool(INPUT_ID.match(fid))


def _vu_strata(j: dict) -> dict[str, list[tuple[int, int]]]:
    """`vuStrata`: the template query (`TemplatePartition`) over every template the root instantiates, its non-input
    root nodes' Definitions (each call one unit, each batch member one, in node then member order), its ranges a
    template at a time with adjacent ones merged (`Flock.Draw.strataOf`)."""
    callees, nodes = j["callees"], j["definitions"][j["root"]]["body"]["nodes"]
    tids = [t for x in nodes if not INPUT_ID.match(t := callees[x["fn"]])]
    strata: dict[str, list[tuple[int, int]]] = {}
    units, at = set(), 0
    for k, x in enumerate(nodes):
        if INPUT_ID.match(tid := callees[x["fn"]]):
            continue
        if x["form"] not in ("call", "batch"):
            raise D.Reject(f"scan: query-refused: root node {k} is a {x['form']} of '{tid}': an instance is a call or "
                           f"batch of one of {tids!r}")
        n = x["n"] if x["form"] == "batch" else 1
        rs = strata.setdefault(tid, [])
        if rs and rs[-1][1] == at:
            rs[-1] = (rs[-1][0], at + n)
        else:
            rs.append((at, at + n))
        if n:
            units.add(tid)
        at += n
    if missing := [t for t in tids if t not in units]:
        raise D.Reject(f"scan: query-refused: the program has no instance of {missing!r}")
    return strata


def _apart(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    """`apart`: ranges `a` and `b`, each `[lo, hi)`, share no unit."""
    return all(min(hi, hj) <= max(lo, lj) for lo, hi in a for lj, hj in b)


def _refs(arg: dict, nodes: list) -> list[tuple[str, int, int]]:
    """`Seq.toArray`: an argument's references in order, each `(space, collection, leaf)`, the space `p` or `n`."""
    out = []
    for q in _parts(arg["refs"], nodes):
        if "affine" in q:
            s, i, b, c = q["affine"]
            out += [("p" if s == "p" else "n", i, b + k) for k in range(c)]
        elif "strided" in q:
            s, i, b, inner, os, step, c = q["strided"]
            out += [("p" if s == "p" else "n", i, max(0, b + k // inner * os + k % inner * step)) for k in range(c)]
        else:
            out += [("p" if s == "p" else "n", i, k) for s, i, k in q["explicit"]]
    return out


def _axis_dims(t, ax: int) -> tuple[int, int] | None:
    """`Ty.axisDims`: a batch argument's `(d, L)` mapped over axis `ax`."""
    if t[0] != "Array":
        return None
    return (t[1], leaves_of(t[2])) if ax == 0 else _axis_dims(t[2], ax - 1)


def _member_refs(x: dict, o: int, j: dict) -> Callable[[int, int], tuple[str, int, int] | None]:
    """`Arg.memberRef` of node `x`'s argument `o`: member `m`'s leaf `k`, with the dims the decoder gives a batch
    argument from its axis; None past the argument's references. They are expanded once (`_refs`), as the decoded
    argument holds its `Seq`, and every member and leaf read through the function indexes them."""
    a, nodes = x["args"][o], j["definitions"][j["root"]]["body"]["nodes"]
    ax = (x.get("axes") or [None] * len(x["args"]))[o]
    t = j["types"][a["type"]] if isinstance(a["type"], int) else a["type"]
    dims = _axis_dims(t, ax) if ax is not None else None
    refs = _refs(a, nodes)

    def ref(m: int, k: int) -> tuple[str, int, int] | None:
        i = k if dims is None else (k // dims[1] * dims[0] + m) * dims[1] + k % dims[1]
        return refs[i] if i < len(refs) else None
    return ref


def _reads_entries_walk(x: dict, o: int, t0: int, every: int, E: int, ks: list[int], j: dict) -> bool:
    """`readsEntries` member by member and leaf by leaf, as it read before it read the arguments' parts: for each lane
    `i` and member `m` (bucket `t0 + m`), argument `o + i` is row `⌊(t0 + m) / every⌋` of input node `ks[i]`, `E`
    leaves a row. The reference `test_reads_entries_is_the_walk` holds `_reads_entries` and Lean's to."""
    def reads(i: int, k: int) -> bool:
        ref = _member_refs(x, o + i, j)
        return all(ref(m, ell) == ("n", k, (t0 + m) // every * E + ell) for m in range(x["n"]) for ell in range(E))
    return all(o + i < len(x["args"]) and reads(i, k) for i, k in enumerate(ks))


def _entry_leaf(t0: int, every: int, E: int, q: int) -> int:
    """`entryLeaf`: reference `q` of a `prog<j>` argument mapped over its leading axis, leaf `q mod E` of row
    `⌊(t0 + ⌊q / E⌋) / every⌋`."""
    return (t0 + q // E) // every * E + q % E


def _rows_step(t0: int, every: int, a: int, rho: int, a1: int, a2: int) -> bool:
    """`rowsStep`: `⌊(t0 + m + a) / every⌋ − ⌊(t0 + m) / every⌋ = rho` for every member `m` of `[a1, a2)`, from the
    first member's residue mod `every` and their count."""
    if a2 <= a1:
        return True
    r, res = a % every, (t0 + a1) % every
    last = res + a2 - a1 - 1
    if rho == a // every:
        return r == 0 or last < every - r
    return rho == a // every + 1 and r != 0 and every - r <= res and last < every


def _steps_by(t0: int, every: int, E: int, I: int, sigma: int, lo: int, hi: int) -> bool:
    """`stepsBy`: `_entry_leaf(q + I) − _entry_leaf(q) = sigma` for every `q` of `[lo, hi)`, from the members the range
    touches on each side of leaf `E − I mod E`."""
    if hi <= lo:
        return True
    beta, m1, m2 = I % E, lo // E, (hi - 1) // E

    def side(delta: int, a1: int, a2: int) -> bool:
        need = sigma - beta + delta * E
        return a2 <= a1 or (need >= 0 and need % E == 0 and _rows_step(t0, every, I // E + delta, need // E, a1, a2))
    return side(0, m1 if lo % E < E - beta else m1 + 1, m2 + 1) and \
        (beta == 0 or side(1, m1, m2 + 1 if E - beta <= (hi - 1) % E else m2))


def _part_reads(k: int, t0: int, every: int, E: int, s: int, q: dict) -> bool:
    """`partReads`: part `q`, its references from `s` on, names leaf `_entry_leaf(q)` of node `k` at each: an affine
    or strided part by its first reference and its strides, an explicit one reference by reference."""
    if "affine" in q:
        sp, i, b, c = q["affine"]
        return c == 0 or (sp == "n" and i == k and b == _entry_leaf(t0, every, E, s)
                          and _steps_by(t0, every, E, 1, 1, s, s + c - 1))
    if "strided" in q:
        sp, i, b, inner, os, step, c = q["strided"]
        I = inner or c
        return c == 0 or (sp == "n" and i == k and b == _entry_leaf(t0, every, E, s)
                          and _steps_by(t0, every, E, 1, step, s, s + min(I, c) - 1)
                          and _steps_by(t0, every, E, I, os, s, s + c - I))
    return all((sp, i, ell) == ("n", k, _entry_leaf(t0, every, E, s + u))
               for u, (sp, i, ell) in enumerate(q["explicit"]))


def _seq_reads(parts: list[dict], N: int, k: int, t0: int, every: int, E: int) -> bool:
    """`seqReads`: the parts are `N` references, each part's from its start."""
    at = 0
    for q in parts:
        if not _part_reads(k, t0, every, E, at, q):
            return False
        at += len(q["explicit"]) if "explicit" in q else (q.get("affine") or q["strided"])[-1]
    return at == N


def _reads_entries(x: dict, o: int, t0: int, every: int, E: int, ks: list[int], j: dict) -> bool:
    """`readsEntries`: for each lane `i` and member `m` (bucket `t0 + m`), argument `o + i` is row
    `⌊(t0 + m) / every⌋` of input node `ks[i]`, lane `i`'s slot entries, `E` leaves a row, read off the argument's
    parts (`_seq_reads`): mapped over its leading axis, its `n·E` references are `_entry_leaf`'s; unmapped, its `E`
    are one row all `n` members share; mapped over an inner axis, refused."""
    nodes = j["definitions"][j["root"]]["body"]["nodes"]
    axes = x.get("axes") or [None] * len(x["args"])

    def lane(i: int, k: int) -> bool:
        if o + i >= len(x["args"]):
            return False
        n, a = x["n"], x["args"][o + i]
        if n == 0 or E == 0:
            return True
        t = j["types"][a["type"]] if isinstance(a["type"], int) else a["type"]
        dims = _axis_dims(t, axes[o + i]) if axes[o + i] is not None else None
        parts = _parts(a["refs"], nodes)
        if dims is None:
            return t0 // every == (t0 + n - 1) // every and _seq_reads(parts, E, k, t0, every, E)
        return dims[1] == E and _seq_reads(parts, n * E, k, t0, every, E)
    return all(lane(i, k) for i, k in enumerate(ks))


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
    """`slotEntryDigest`: the SHA-512, in hex, of the canonical JSON object that maps each Definition `key` reaches
    (itself, its nodes' callees, theirs, …) to its `annotations.definition_digest`: the template's whole body."""
    defs, callees, seen, todo = j["definitions"], j["callees"], {}, [key]
    while todo:
        k = todo.pop()
        if k not in seen:
            seen[k] = AN.definition_digest(j, k)
            if defs[k]["form"] == "composite":
                todo += [callees[x["fn"]] for x in defs[k]["body"]["nodes"]]
    return hashlib.sha512(codec.canonical_json(seen)).hexdigest()


def input_gates(j: dict) -> int:
    """`inputGates`: each of the root's input nodes, of either binding time, its members times its width."""
    defs, callees = j["definitions"], j["callees"]
    return sum(_members(x) * int(m[1]) for x in defs[j["root"]]["body"]["nodes"]
               if defs[k := callees[x["fn"]]]["form"] == "primitive" and (m := INPUT_ID.fullmatch(k)))


AUDIT_FORMS = (["program", "steps"], ["circuit", "order", "pins", "program", "steps"],
               ["circuit", "circuits", "order", "pins", "program", "steps"])


@dataclasses.dataclass(frozen=True)
class Listed:
    """`Listed`: what an audit's or the scan's listed circuit files are held to: its Program's SHA-512 and Definitions
    (`def_sig` by key), its root templates' charges (`charges_of`: descriptor id, members and words), each one's file's
    SHA-512, in order, and for an audit its circuit record's partition digest (`record_partition`)."""
    program: str
    defs: dict[str, tuple[list, list]]
    templates: tuple[tuple[str, int, int], ...]
    files: tuple[str, ...]
    partition: str | None


@dataclasses.dataclass(frozen=True)
class Listing:
    """`Listing`: each audit's listed files (None for one that lists none) and the scan's (`scan_circuits`)."""
    audits: tuple[Listed | None, ...] = ()
    scan: Listed | None = None

    @property
    def listed(self) -> bool:
        return any(x is not None for x in self.audits) or self.scan is not None


def charges_of(j: dict) -> list[tuple[str, int, int]]:
    """`chargesOf`: each root template's descriptor id, members and the 16-bit words one instance commits, its
    parameters' and its return's (`words_of`, `ret_words`)."""
    defs, callees = j["definitions"], j["callees"]
    return [(k, _members(x), sum(words_of(t) for _, t in d["body"]["params"]) + ret_words(d["body"]["ret_type"]))
            for x in defs[j["root"]]["body"]["nodes"] if (d := defs[k := callees[x["fn"]]])["form"] != "primitive"]


def _listed(j: dict, fs, ctx: str, partition: str | None) -> Listed:
    """`listedOf`: a listing `[{"sha512", "template"}, …]` of a file each of the Program's root templates, in order."""
    ts = charges_of(j)
    got = []
    for f in _of(fs, list, ctx):
        _exactly(f, ["sha512", "template"], f"{ctx}: a file")
        if not DIGEST.fullmatch(_of(f["sha512"], str, ctx)):
            raise D.Reject(f"{ctx}: a file's sha512 is not a SHA-512 digest")
        got.append((_of(f["template"], str, ctx), f["sha512"]))
    if [t for t, _ in got] != [t for t, _, _ in ts]:
        raise D.Reject(f"{ctx}: the files' templates are not the Program's root templates in order, a file each")
    return Listed(program_sha512(j), {k: def_sig(d) for k, d in j["definitions"].items()}, tuple(ts),
                  tuple(d for _, d in got), partition)


def partition_digest(obj: dict) -> str:
    """`Flock.HmRow.partitionDigest`, hex: SHA-512 of `verity/partition/v1`, a zero byte and the canonical object."""
    return hashlib.sha512(PARTITION_V1.encode() + b"\0" + D.canonical(obj).encode()).hexdigest()


def record_partition(c, ctx: str) -> str:
    """`recordPartition`: a circuit record's `verity/partition/v1` digest of its program and query, which must be its
    own `partition`."""
    d = partition_digest({"format": PARTITION_V1, "program": c.get("program"), "query": c.get("query")})
    if c.get("partition") != d:
        raise D.Reject(f"{ctx}: its partition is not the {PARTITION_V1} digest of its program and query")
    return d


def _audit(a, ctx: str) -> Audit:
    """`auditOf`: `_audit_listed`'s audit, without its listed files."""
    return _audit_listed(a, ctx)[0]


def _audit_listed(a, ctx: str) -> tuple[Audit, Listed | None]:
    """`auditWith`: the layout terms and the Program's descriptor, whose root's template instances are the audit's
    units, each charged `vu_bits` at one instance a VU of its template's parameters' and return's words; a root with no
    template refused; its free inputs the Program's input gates (`input_gates`), never a parameter, or, with its
    window's circuit record, wire order and the verifier's pins, the record's count (`_record_count`); its slot entries'
    templates (`ENTRY_IDS`) with their closure digests; and with its record, its listed circuit files (`_listed`), the
    record's partition its program's and query's (`record_partition`). It reads the descriptor's root nodes, signatures
    and the slot entries' Definitions; C-Flock's decoder, which Lean runs on it, refuses no descriptor this module
    writes."""
    if not isinstance(a, dict) or sorted(a) not in AUDIT_FORMS:
        raise D.Reject(f"{ctx}: an object whose keys are {AUDIT_FORMS[0]}, {AUDIT_FORMS[1]} or {AUDIT_FORMS[2]}")
    steps, j = _nat(a["steps"], ctx), a["program"]
    if not isinstance(j, dict) or j.get("schema") != SCHEMA:
        raise D.Reject(f"{ctx}: program: not a Program's descriptor")
    defs, callees = j["definitions"], j["callees"]
    nodes = defs[j["root"]]["body"]["nodes"]
    units = charges_of(j)
    if not units:
        raise D.Reject(f"{ctx}: program: no template")
    entries = tuple((k, _members(x), slot_entry_digest(j, k)) for x in nodes
                    if (d := defs[k := callees[x["fn"]]])["form"] != "primitive" and d.get("id") in ENTRY_IDS)
    free, state_free, record, circuit = (input_gates(j), None, None, None) if "circuit" not in a else \
        _record_count(a, f"{ctx}: circuit")
    listed = _listed(j, a["circuits"], f"{ctx}: circuits", record_partition(a["circuit"], f"{ctx}: circuit")) \
        if "circuits" in a else None
    return Audit(free, steps, program_sha512(j), sum(n for _, n, _ in units), max(vu_bits(1, w) for _, _, w in units),
                 entries, record, circuit, state_free), listed


def _record_count(a: dict, ctx: str) -> tuple[int, int, tuple[str, tuple[tuple[str, str], ...]], Stated]:
    """`auditOf`'s record form: a pin of the scan's state refused (no value the verifier fixes before the run), then the
    free inputs `NetworkCheck.Circuit.check` counts on the record, every pin's parameter held: `interface.input_count`'s,
    which is its count on a record it accepts, and its count with `SCAN_STATE` held too (`stateFree`). Its checks of
    the record against the Program and of the order are Lean's alone: this module writes no record either refuses.
    With the record's digest (`interface.circuit_bytes`) and the pins, and the record (`recordOf`), its templates' rows
    the catalog's Definitions' (`interface.layout`), which are the Program's on a record Lean accepts."""
    pins = a["pins"]
    if not isinstance(pins, dict) or not all(isinstance(r, str) for r in pins.values()):
        raise D.Reject(f"{ctx}: pins are {{param: root}}")
    if SCAN_STATE in pins:
        raise D.Reject(f"{ctx}: a pin of {SCAN_STATE}: the scan's state is no value the verifier fixes before the run, "
                       "so its reads count free")
    doc = a["circuit"]
    count = IF.input_count(doc, pins=pins, roots=pins)
    held = pins | {SCAN_STATE: "00"}
    state = IF.input_count(doc, pins=held, roots=held)
    digest = hashlib.sha512(IF.circuit_bytes(doc)).hexdigest()
    _wires, cuts, reg, _ingress = IF.declarations(doc)
    rows = {t: dict(IF.layout(IF.resolve(t)).rows) for t, _n in doc["templates"]}
    return count["free"], state["free"], (digest, tuple(sorted(pins.items()))), \
        Stated(rows, cuts, reg or IF.Registered({}))


def _scan(j, program: str, s: WardenParams) -> Scan:
    """`scanOf`: the scan's terms from its Program's descriptor, which must be the registered Program, with no
    template keyed as an input (`INPUT_ID`; two definitions of one key can't be written, `definitions` being an object),
    and the generator form of the schedule's link-window in the Boolean profile's types (`BOOL_BYTE`, `BOOL_U16`), its
    egress template's VUs apart from every other template's (`_vu_strata`, `_apart`) first, then an egress template
    with a row among its parameters (the checker form) refused. Then what `NetTiming.BoundaryGates.Fits` asks: the
    walk within `FUEL` (`_walkable`), the root returning every egress leaf, the egress template returning only gates it
    computes (`_own_returns`), each ingress row and every template's row, state and egress argument one input node's
    leaves (`_input_arg`), no egress argument an ingress row's, bucket t's `prog<j>` row `⌊t / every⌋` of lane j's
    own input node of its slot entries (`_reads_entries`), and the Program topological (`_topological`). It reads
    the descriptor's nodes, signatures and returns; C-Flock's decoder, which Lean runs on it, refuses no descriptor
    this module writes."""
    if not isinstance(j, dict) or j.get("schema") != SCHEMA:
        raise D.Reject("scan: not a Program's descriptor")
    if program_sha512(j) != program:
        raise D.Reject("scan: its SHA-512 is not the registered Program's")
    defs, callees = j["definitions"], j["callees"]
    nodes = defs[j["root"]]["body"]["nodes"]
    templates = [(x, defs[callees[x["fn"]]]["body"]) for x in nodes if defs[callees[x["fn"]]]["form"] != "primitive"]
    if any(INPUT_ID.fullmatch(callees[x["fn"]]) for x, _ in templates):
        raise D.Reject("scan: a template is keyed as an input")
    T, row = s.window, ["Array", s.lanes * s.slots * s.cell_bytes, BOOL_BYTE]
    if not templates:
        raise D.Reject("scan: no template")
    (xf, bf), later, (xe, be) = templates[0], templates[1:-1], templates[-1]
    vus = _vu_strata(j)
    if not all(_apart(vus.get(callees[x["fn"]], []), vus.get(callees[xe["fn"]], [])) for x, _ in templates[:-1]):
        raise D.Reject("scan: the egress template's VUs are not apart from the ingress templates' (vuStrata): a "
                       "stratum of the draw would hold both egress and ingress units")
    if any(k == "row" or t == row for k, t in be["params"]):
        raise D.Reject("scan: the egress template takes its row as an input (the checker form)")
    if len(templates) not in (2, 3):
        raise D.Reject("scan: not the generator form's templates")
    names = lambda b: tuple(k for k, _ in b["params"])  # noqa: E731
    L = s.lanes
    if (names(bf) != first_params(L) or any(names(b) != ingress_params(L) for _, b in later)
            or names(be) != EGRESS_PARAMS):
        raise D.Reject("scan: the templates' parameters are not the generator form's")
    counts = [(xf, 1), *((x, T - 1) for x, _ in later), (xe, T)]
    if len(later) != (T > 1) or not all(x["form"] == "batch" and x["n"] == n for x, n in counts):
        raise D.Reject(f"scan: not bucket 0, {T - 1} later ingress buckets and {T} egress buckets")
    if bf["params"][0][1] != row or any(b["params"][0][1] != row for _, b in later):
        raise D.Reject("scan: an ingress template's row is not the schedule's")
    ety = bf["params"][1][1]
    if any(t != ety for _, t in bf["params"][1:]) or any(
            any(t != ety for _, t in b["params"][2:]) or b["ret_type"] != bf["ret_type"] for _, b in later):
        raise D.Reject("scan: the ingress templates' entry ports' types or returns differ")
    if be["ret_type"] != ["Tuple", [BOOL_U16, row]]:
        raise D.Reject("scan: the egress template doesn't return a u16 and the schedule's row")
    inputs = sum(int(w[1]) * _members(x) for x in nodes if (w := INPUT_ID.match(callees[x["fn"]])))
    bits = lambda ps: sum(bits_of(t) for _, t in ps)  # noqa: E731
    nb = T // s.every
    if inputs != (bits(bf["params"][:1]) + (T - 1) * sum(bits(b["params"][:2]) for _, b in later)
                  + T * bits(be["params"]) + L * nb * bits_of(ety)):
        raise D.Reject("scan: an input gate that is no template's row, state or egress parameter, nor one of the slot "
                       "entries")
    if not _walkable(j):
        raise D.Reject("scan: not laid out as C-Flock's decoder lays a Program out, or past what its resolution walks "
                       "(NciCheck.Desc.rootLaidOut, inputsAtRoot and walkable)")
    ke = max(k for k, x in enumerate(nodes) if defs[callees[x["fn"]]]["form"] != "primitive")
    if not any(a[:3] == ("n", ke, 0) and T * leaves_of(be["ret_type"]) <= a[3]
               for q in _parts(defs[j["root"]]["body"]["ret"], nodes) if (a := _affine(q))):
        raise D.Reject("scan: the root doesn't return every egress leaf, so an egress gate would be one the Program "
                       "doesn't return (NetTiming.BoundaryGates.Fits)")
    if not _own_returns(j)(callees[xe["fn"]]):
        raise D.Reject("scan: the egress template returns a leaf it doesn't compute (a parameter's, passed through), "
                       "so an egress gate would be an input (NetTiming.BoundaryGates.Fits)")
    ingress = [xf, *(x for x, _ in later)]
    rows = [_input_arg(x["args"][0], nodes, j) if x["args"] else None for x in ingress]
    if None in rows:
        raise D.Reject("scan: an ingress template's row is not one input node's leaves, so an ingress gate would be no "
                       "input (NetTiming.BoundaryGates.Fits)")
    own_args = [*xf["args"][:1], *(a for x, _ in later for a in x["args"][:2]), *xe["args"]]
    if (len(xf["args"]) != 1 + L or any(len(x["args"]) != 2 + L for x, _ in later)
            or any(_input_arg(a, nodes, j) is None for a in own_args)):
        raise D.Reject("scan: a template's row, state or egress argument is not one input node's leaves")
    if any(_input_arg(a, nodes, j) in rows for a in xe["args"]):
        raise D.Reject("scan: an egress template's argument is an ingress row's input node, so an egress gate would "
                       "depend on the ingress (NetTiming.BoundaryGates.Fits)")
    r0 = [_member_refs(xf, 1 + i, j)(0, 0) for i in range(L)]
    pks = [r[1] for r in r0 if r is not None]
    if (len(pks) != L or any(r[0] == "p" for r in r0) or len(set(pks)) != L
            or any(k in rows or not (k < len(nodes) and _is_input(nodes[k], j)) for k in pks)):
        raise D.Reject("scan: bucket 0's prog<j> reads no input node of lane j's own, its slot entries', distinct from "
                       "every other lane's")
    E = leaves_of(ety)
    if not (_reads_entries(xf, 1, 0, s.every, E, pks, j)
            and all(_reads_entries(x, 2, 1, s.every, E, pks, j) for x, _ in later)):
        raise D.Reject(f"scan: bucket t's prog<j> is not row ⌊t / {s.every}⌋ of lane j's slot entries' input node, "
                       "its slot of the boundary bucket t reads (NetworkCheck.Period.readsEntries)")
    if not _topological(j):
        raise D.Reject("scan: a body names a later node (Flock.Program.Prog.orderViolation): the Program is not "
                       "topological, so a gate's operands could be gates after it")
    words = lambda ps: sum(words_of(t) for _, t in ps)  # noqa: E731
    return Scan(T, 0, vu_bits(1, max([words(bf["params"]), *(words(b["params"]) for _, b in later)])
                               + ret_words(bf["ret_type"])),
                vu_bits(1, words(be["params"]) + ret_words(be["ret_type"])), L * nb * bits_of(ety))


def params(j) -> Params:
    """`paramsOf`: `params_listed`'s parameters, a listing of circuit files refused (only `check_with` reads one)."""
    p, listing = params_listed(j)
    if listing.listed:
        raise D.Reject("the period lists circuit files, which only network-period check PARAMS COMMITMENT EVIDENCE "
                       "CIRCUIT… reads")
    return p


def params_listed(j) -> tuple[Params, Listing]:
    """`paramsWith`: the parameters, and the circuit files each audit lists and, in the rows form, the scan's
    (`scan_circuits`)."""
    if not isinstance(j, dict):
        raise D.Reject("the object: not an object")
    if sorted(j) not in (PARAMS_KEYS, ROWS_PARAMS_KEYS, ROWS_LISTED_KEYS):
        raise D.Reject(f"the object: the keys must be {PARAMS_KEYS}, {ROWS_PARAMS_KEYS} or {ROWS_LISTED_KEYS}, not "
                       f"{sorted(j)}")
    read = [_audit_listed(a, "an audit") for a in _of(j["audits"], list, "audits")]
    audits = [a for a, _ in read]
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
    circuit = None
    if "circuit" in j and not DIGEST.fullmatch(circuit := _of(j["circuit"], str, "circuit")):
        raise D.Reject("circuit: not a SHA-512 digest")
    if "sync_set" in j:
        sync_set, schedule, scan = _sync_set(j["sync_set"]), None, None
    else:
        schedule = _schedule(j["schedule"])
        sync_set, scan = None, _scan(j["scan"], program, schedule)
    windows = _nat(j["windows"], "windows")
    if windows < 1:
        raise D.Reject("windows: a period has at least one link-window")
    if schedule is not None and not rows_wf(schedule, windows):
        raise D.Reject(rows_wf_refusal(schedule, windows))
    if schedule is not None and list(cells) != schedule.caps(windows):
        raise D.Reject(f"cells: the rows form's caps are egress_cells = {schedule.egress_cells} for each of the "
                       f"lanes · windows · window = {schedule.lanes * windows * schedule.window} connections the "
                       f"schedule can carry, not these {len(cells)}")
    listed = _listed(j["scan"], j["scan_circuits"], "scan_circuits", None) if "scan_circuits" in j else None
    return (Params(tuple(audits), cells, first, jitter, link, LmsKey(lp, key), session, program, sync_set, windows,
                   schedule, scan, circuit), Listing(tuple(x for _, x in read), listed))


def _registration(j, ctx: str) -> dict:
    if not isinstance(j, dict) or not isinstance(j.get("format"), str):
        raise D.Reject(f"{ctx}: no format")
    if j["format"] not in FORMATS:
        raise D.Reject(f"{ctx}: format is not {' or '.join(FORMATS)}")
    return j


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


def _answered(what: str, reg: dict, a: dict) -> tuple[dict, object]:
    if a["protocol"] != ONE_STAGE:
        raise Finding(REJECT, f"{what}: protocol {a['protocol']}")
    if a["commitment"] != R.digest(reg):
        raise Finding(REJECT, f"{what}: the answer is for another registration")
    if a["verdict"] == ACCEPT:
        try:
            return CC.integrity(a["profile"]), a["public"]
        except CC.Refused as e:
            raise Finding(REJECT, f"{what}: {e}") from None
    raise Finding(a["verdict"], f"{what}: service: {a['reason']}")


def served(w: int) -> dict:
    return {"kind": "served", "window": w}


def _window_check(p: Params, clear: set[bytes], earlier: list[bytes], w: int, reg: dict, x: Window) -> None:
    what = f"window {w}"
    if x.clock is None:
        raise Finding(REJECT, f"{what}: registrations, not a clock certificate")
    if not (reg.get("program") == p.program
            and D.canonical(reg.get("window")) == D.canonical(served(w))):
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


def rows_name(d: Direction, w: int) -> str:
    """`rowsName`: the registration's port of the network certifier's rows of end `d` at window `w`."""
    return f"rows/{d.value}/{w}"


def _holds(reg: dict, d: Direction, w: int, doc: dict) -> bool:
    """`holds`: the scan's registration has an entry of its `roots` at port `rows_name(d, w)` or of its `registered` at
    the rows' value, and every such entry names their root and domain, a `roots` one with their leaves; and every entry
    of either under their domain is at their root, so their domain names one tree."""
    ports = [e for e in _entries(reg, "roots") if e.get("port") == rows_name(d, w)]
    values = [e for e in _entries(reg, "registered") if e.get("value") == doc["value"]]
    under = [e for k in ("roots", "registered") for e in _entries(reg, k) if e.get("domain") == doc["domain"]]

    def at(e: dict) -> bool:
        return e.get("root") == doc["root"] and e.get("domain") == doc["domain"]

    return (bool(ports + values) and all(at(e) and _leaves(e) == doc["rows"] for e in ports) and all(map(at, values))
            and all(e.get("root") == doc["root"] for e in under))


#: the scan's Programs' rows in its registration, one tree a lane, `PROG<j>` lane j's (`progName`): its `roots` entry's
#: port, or its `registered` entry's value
PROG = "scan/prog"
#: the scan's state in a registration: its `roots` entry's port (`stateName`), or its `registered` entry's value, the
#: value a window circuit registers the scan's state rows as (`stateValue`, `ingress_scan`'s `SCAN_STATE`)
STATE = "scan/state"
SCAN_STATE = "verity/certifier/scan-state/v1"


def output_port(port) -> bool:
    """`outputPort`: a unit's committed output port, ``<unit>/o<j>`` with j ≥ 1 (ASCII digits, no leading zero)."""
    if not isinstance(port, str) or len(parts := port.split("/")) < 2:
        return False
    j = parts[-1][1:]
    return parts[-1][:1] == "o" and j != "" and all("0" <= c <= "9" for c in j) and j[0] != "0"


def prog_name(j: int) -> str:
    """`progName`: lane `j`'s Programs' rows, ``scan/prog<j>``."""
    return f"{PROG}{j}"


def _index(s: str) -> bool:
    """`indexChars`: a lane's or a chunk's index, ASCII digits with no leading zero."""
    return s != "" and all("0" <= c <= "9" for c in s) and (s[0] != "0" or len(s) == 1)


def prog_port(name) -> bool:
    """`progPort`: some lane's Programs' rows (`prog_name`)."""
    return isinstance(name, str) and name.startswith(PROG) and _index(name[len(PROG):])


def entries_port(key: str, j: int = 0) -> str:
    """`entriesPort`: a `SlotEntry_v1` or `SelectEntry_v1` template's entry rows of chunk `j`, its `j`-th member's
    public word, ``<descriptor id>/c<j>/o0``."""
    return f"{key}/c{j}/o0"


def entries_port_name(port) -> bool:
    """`entriesPortName`: some `SlotEntry_v1` or `SelectEntry_v1` template's entries port of some chunk
    (`entries_port`)."""
    if not (isinstance(port, str) and any(port.startswith(i + "{") for i in ENTRY_IDS) and port.endswith("/o0")):
        return False
    head = port[:-3]
    stem = head.rstrip("0123456789")
    return stem.endswith("}/c") and _index(head[len(stem):])


def _entries(j: dict, k: str) -> list[dict]:
    v = j.get(k)
    return [e for e in v if isinstance(e, dict)] if isinstance(v, list) else []


def _prog(reg: dict, j: int | None = None) -> list[dict]:
    """The scan's registration's Programs' rows entries of lane `j` (`prog_name`), or of any lane (`prog_port`), by
    port or value, their root and domain strings."""
    lane = prog_port if j is None else lambda q: q == prog_name(j)
    return [e for e in [*(e for e in _entries(reg, "roots") if lane(e.get("port"))),
                        *(e for e in _entries(reg, "registered") if lane(e.get("value")))]
            if isinstance(e.get("root"), str) and isinstance(e.get("domain"), str)]


def _tied_audits(audits: list[dict], reg: dict) -> list[int]:
    """`tiedAudits`: the audits, by index, whose registration's `roots` has an output port's or an entries port's entry
    (`output_port`, `entries_port_name`) at the root and domain of a lane's `scan/prog<j>` entry of the scan's
    registration (`prog_port`)."""
    return [i for i, a in enumerate(audits)
            if any(any((output_port(q := o.get("port")) or entries_port_name(q)) and o.get("root") == e["root"]
                       and o.get("domain") == e["domain"] for o in _entries(a, "roots"))
                   for e in _prog(reg))]


def _leaves(o: dict):
    """A roots entry's leaf count, `getNat?`'s: an int, not a bool, at least 0; else None."""
    n = o.get("leaves")
    return n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else None


def _wired(p: Params, audits: list[dict], reg: dict) -> bool:
    """`wired`: for each lane j, the scan's `scan/prog<j>` entry is at the root and domain of chunk j's entries port
    (`entries_port`), of T / every leaves, of a `SlotEntry_v1` or `SelectEntry_v1` template of one audit's own Program
    (`Audit.entries`), a pinned Definition (its closure digest one of `SLOT_ENTRY_PINS` and `SELECT_ENTRY_PINS`,
    `entryPins`), with one member a slot of the window."""
    s = p.schedule
    nb = s.window // s.every
    return any(n == nb * s.lanes and d in SLOT_ENTRY_PINS + SELECT_ENTRY_PINS and all(
        any(o.get("port") == entries_port(k, j) and _leaves(o) == nb and o.get("root") == e["root"]
            and o.get("domain") == e["domain"] for e in _prog(reg, j)
            for o in _entries(r, "roots"))
        for j in range(s.lanes))
        for a, r in zip(p.audits, audits) for k, n, d in a.entries)


def _states(reg: dict) -> list[dict]:
    """`stateEntries`: a registration's `roots` entries whose port is `STATE` and `registered` entries whose value is
    `SCAN_STATE`."""
    return [*(e for e in _entries(reg, "roots") if e.get("port") == STATE),
            *(e for e in _entries(reg, "registered") if e.get("value") == SCAN_STATE)]


def _get(j, k: str):
    """`getObjValD`: `j`'s value at `k`, None where `j` is no object or has none."""
    return j.get(k) if isinstance(j, dict) else None


def _natural(v) -> int | None:
    """`getNat?`: an int, not a bool, at least 0; else None."""
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None


def _unhex(v) -> bytes | None:
    """`Flock.unhex`, as an option: a string of lowercase hex digit pairs; else None."""
    ok = isinstance(v, str) and len(v) % 2 == 0 and all(c in "0123456789abcdef" for c in v)
    return bytes.fromhex(v) if ok else None


def _run(j, lo: int, hi: int, leaf: int) -> bool:
    """`isRun`: `j` is ``{"leaf": leaf, "units": [lo, hi]}``, each a natural (`_natural`)."""
    units = _get(j, "units")
    return _natural(_get(j, "leaf")) == leaf and isinstance(units, list) and len(units) == 2 \
        and _natural(units[0]) == lo and _natural(units[1]) == hi


def state_reads(T: int) -> dict:
    """The scan's state reads its registration's `window` states (`stateRead`): `sin` units 1 … T − 1 from leaf 1,
    `o1` units 0 … T − 2 from leaf 1, unit t's `sin` at leaf t and its `o1` at leaf t + 1."""
    return {STATE: {"o1": {"leaf": 1, "units": [0, T - 1]}, "sin": {"leaf": 1, "units": [1, T]}}}


def _state_read(p: Params, reg: dict) -> bool:
    """`stateRead`: the registration's `window`'s `reads` of `STATE` are `state_reads` at the schedule's T."""
    r = _get(_get(_get(reg, "window"), "reads"), STATE)
    T = p.schedule.window
    return _run(_get(r, "sin"), 1, T, 1) and _run(_get(r, "o1"), 0, T - 1, 1)


class _Domain:
    """A frame-v3-sha512 domain known by its id alone, as Lean's `climb` takes one: a leaf's hash and a node's read the
    id and nothing else of the domain."""
    hash = "sha512"
    leaf, node = CommitmentDomain.leaf, CommitmentDomain.node

    def __init__(self, domain_id: bytes, count: int) -> None:
        self.domain_id, self.count = domain_id, count


def _state_empty(p: Params, reg: dict) -> bool:
    """`stateEmpty`: the registration's `window`'s ``empty`` of `STATE` opens leaf 0 of its first state entry's root
    and domain to the empty state, ``words`` (at least 1) zero u16 words under a 192-byte ``salt``, through ``path``,
    one sibling a level of T leaves."""
    o, es, T = _get(_get(_get(reg, "window"), "empty"), STATE), _states(reg), p.schedule.window
    w = _natural(_get(o, "words"))
    if not es or not w:
        return False
    salt, path, root, dom = (_unhex(_get(o, "salt")), _unhex(_get(o, "path")), _unhex(es[0].get("root")),
                             _unhex(es[0].get("domain")))
    if salt is None or path is None or root is None or dom is None or len(salt) != ADV.INST.salt_bytes \
            or len(path) != 64 * clog2(T):
        return False
    bc = ADV.INST.commit_string(ADV.INST.default_key, BitRowLeaf(16 * w).layout().digest(bytes(2 * w)), salt)
    return G.verify_read(_Domain(dom, T), root, 0, bc, path, G.SCHEMA_V2)


def state_empty(tree: ADV.Committed) -> dict:
    """The opening of a state tree's leaf 0 its registration's `window` gives as ``empty`` (`stateEmpty`): its words,
    salt and path."""
    o = tree.open(0)
    return {STATE: {"path": b"".join(o.path).hex(), "salt": o.salt.hex(), "words": tree.n_bits // 16}}


def _state_tied(p: Params, audits: list[dict], reg: dict) -> bool:
    """`stateTied`: the scan's registration states its units' state reads as the chain (`_state_read`), has a state
    entry (`_states`), every one at one root and domain, both strings, and an audit wired to its `prog` (`_wired`) has
    one, every one at that root and domain."""
    es = _states(reg)
    if not _state_read(p, reg) or not es or not isinstance(root := es[0].get("root"), str) \
            or not isinstance(dom := es[0].get("domain"), str):
        return False

    def at(o: dict) -> bool:
        return o.get("root") == root and o.get("domain") == dom

    return all(map(at, es)) and any(
        _wired(dataclasses.replace(p, audits=[a]), [r], reg) and bool(ss := _states(r)) and all(map(at, ss))
        for a, r in zip(p.audits, audits))


def _anchor_digits(s: WardenParams, W: int) -> list[tuple[int, int]]:
    """`anchorDigits`: the digits, slowest first, of the read a tied audit's slots make of the scan's state at their
    anchors, `W` the reader port's words: unit l · nb + b from leaf b · every · W, a digit of radix 1 left out."""
    return [d for d in ((s.lanes, 0), (s.window // s.every, s.every * W)) if d[0] != 1]


def _anchor_read(s: WardenParams, c: Stated, x: IF.Read) -> bool:
    """`anchorRead`: registered read `x` of the scan's state is whole rows of its reader's uncut input port of `W`
    words, units [0, lanes · nb) from base 0, its digits less those of radix 1 `_anchor_digits`', of a parameter of
    T · W leaves."""
    W = c.rows.get(x.reader, {}).get(x.port)
    return W is not None and c.cuts.get(x.reader, {}).get(x.port, 1) == 1 and x.at == 0 and x.cut == (0, 1) \
        and x.base == 0 and [d for d in zip(x.radix, x.step) if d[0] != 1] == _anchor_digits(s, W) \
        and c.registered.params.get(SCAN_STATE) == s.window * W


def _anchored(s: WardenParams, a: Audit) -> bool:
    """`anchored`: with its circuit record, each of the record's registered reads of `SCAN_STATE` is the anchors'
    (`_anchor_read`) and no table is over it; with none, it states no read."""
    c = a.circuit
    return c is None or (all(_anchor_read(s, c, x) for x in c.registered.reads if x.param == SCAN_STATE)
                         and not any(t.param == SCAN_STATE for t in c.registered.tables))


def _unanchored(p: Params, audits: list[dict], reg: dict) -> list[int]:
    """The audits tied to a link-window's scan (`_tied_audits`) that don't read its state at their slots' anchors
    (`anchoredAt`'s exceptions)."""
    return [i for i in _tied_audits(audits, reg) if i < len(p.audits) and not _anchored(p.schedule, p.audits[i])]


def _held(p: Params, audits: list[dict], reg: dict) -> bool:
    """`held`: each audit the scan's `prog` is an output of (`_tied_audits`) is registered under the parameters' class
    circuit (`runsCircuit`), none with no class circuit."""
    return all(p.circuit is not None and audits[i].get("program") == p.circuit for i in _tied_audits(audits, reg))


def _prog_tied(audits: list[dict], reg: dict) -> bool:
    """`progTied`: some audit is tied to the scan's `prog` (`_tied_audits`)."""
    return bool(_tied_audits(audits, reg))


def _tied_in(audits: list[dict], windows: list[dict], i: int) -> bool:
    """`tiedIn`: some link-window's `prog` is audit `i`'s output (`_tied_audits`)."""
    return any(i in _tied_audits(audits, reg) for reg in windows)


def _state_held(audits: list[dict], windows: list[dict], i: int) -> bool:
    """`stateHeldIn`: audit `i`'s registration has a state entry (`_states`), and in each link-window whose `prog` is its
    output (`_tied_audits`) the scan's first state entry has a root and a domain, both strings, at which every one of the
    audit's is."""
    if i >= len(audits) or not (ss := _states(audits[i])):
        return False

    def at(reg: dict) -> bool:
        es = _states(reg)
        root, dom = (es[0].get("root"), es[0].get("domain")) if es else (None, None)
        return isinstance(root, str) and isinstance(dom, str) and all(
            o.get("root") == root and o.get("domain") == dom for o in ss)

    return all(i not in _tied_audits(audits, reg) or at(reg) for reg in windows)


def _charge(p: Params, audits: list[dict], windows: list[dict], i: int) -> int:
    """`chargeOf`: tied audit `i`'s free inputs with its reads of the scan's state held (`state_free`) where its
    `SCAN_STATE` is the scan's state in every phase tied to it (`_state_held`), else its free inputs; 0 for no audit."""
    if i >= len(p.audits):
        return 0
    a = p.audits[i]
    return a.state_free if a.state_free is not None and _state_held(audits, windows, i) else a.free


def _tied_free(p: Params, audits: list[dict], windows: list[dict], reg: dict) -> int:
    """`tiedFree`: what the form charges each audit a link-window's `prog` is an output of (`_tied_audits`, `_charge`),
    summed."""
    return sum(_charge(p, audits, windows, i) for i in _tied_audits(audits, reg))


def records(link: str) -> tuple[tuple[str, Direction], ...]:
    """The rows form's certified records of a link-window (`warden_serve.CERTIFIED`): requests, responses, fails."""
    return (link, INGRESS), (link, EGRESS), (f"{link}-fails", EGRESS)


def _rows_window_check(p: Params, audits: list[dict], w: int, reg: dict, x: Window) -> Facts:
    """`rowsWindowCheck`: on acceptance the scan's facts, which its term charges beside its tied audits' free inputs
    (`_tied_free`, `check`)."""
    what = f"window {w}"
    if x.registrations is None:
        raise Finding(REJECT, f"{what}: a clock certificate, not registrations")
    if reg.get("program") != p.program:
        raise Finding(REJECT, f"{what}: registration")
    if not (len(x.certificates) == 3
            and all(_certifies_on(p, link, d, w, c) for c, (link, d) in zip(x.certificates, records(p.link)))):
        raise Finding(REJECT, f"{what}: certificates (none of a window in which the network certifier refused a "
                              "connection)")
    if len(x.registrations) != 2:
        raise Finding(REJECT, f"{what}: registrations")
    for doc, d in zip(x.registrations, (INGRESS, EGRESS)):
        _rows_check(p, w, d, doc)
    for doc, d in zip(x.registrations, (INGRESS, EGRESS)):
        if not _holds(reg, d, w, doc):
            raise Finding(REJECT, f"{what}: {ROWS_UNHELD} {rows_name(d, w)}")
    if not _prog_tied(audits, reg):
        raise Finding(REJECT, f"{what}: the scan's prog is no compute audit's output")
    prof, pub = _answered(what, reg, x.answer)
    f = _facts(what, prof, pub)
    if f.units != 2 * p.schedule.window:
        raise Finding(REJECT, f"{what}: profile: {f.units} units, not the scan's 2T")
    return f


def _facts(what: str, prof: dict, pub) -> Facts:
    """`factsOf`: one level drawn `subset:k` or stratified, and the public's bound."""
    if len(prof["levels"]) != 1:
        raise Finding(REJECT, f"{what}: profile: not one level")
    (lv,) = prof["levels"]
    draw = lv["draw"]
    if isinstance(draw, dict) and "subset" in draw:
        strata = ((lv["size"], draw["subset"]),)
    elif isinstance(draw, dict) and draw.get("law") == "stratified":
        strata = tuple((s["size"], s["k"]) for s in draw["strata"])
    else:
        raise Finding(REJECT, f"{what}: profile: not a subset or stratified draw")
    bound = pub.get("bound") if isinstance(pub, dict) else None
    if not isinstance(bound, int) or isinstance(bound, bool) or bound < 0:
        raise Finding(REJECT, f"{what}: no bound")
    return Facts(lv["size"], sum(k for _, k in strata), prof["delta"], bound, strata)


def _audit_check(i: int, audit: Audit, reg: dict, a: dict) -> Facts:
    """`auditCheck`: the registration names the audit's Program, with a circuit record its window binds that record and
    records each pin as its parameter's registered root (`pinHeld`, the hex without case), and the answer's proof blocks
    are the Program's, at any sample."""
    what = f"audit {i}"
    if reg.get("program") != audit.program:
        raise Finding(REJECT, f"{what}: registration: not its Program")
    if audit.record is not None:
        digest, pins = audit.record
        window = reg.get("window") if isinstance(reg.get("window"), dict) else {}
        if window.get("circuit") != digest:
            raise Finding(REJECT, f"{what}: registration: its window binds another circuit record than the one its "
                                  "free inputs are counted on")
        roots = window.get("registered_roots") if isinstance(window.get("registered_roots"), dict) else {}
        for p, r in pins:
            if not isinstance(roots.get(p), str) or roots[p].lower() != r.lower():
                raise Finding(REJECT, f"{what}: registration: the pin of {p} is not its window's registered root for "
                                      f"{p}, so the count would hold the reads of a value the prover chose")
    prof, pub = _answered(what, reg, a)
    f = _facts(what, prof, pub)
    if f.units != audit.units:
        raise Finding(REJECT, f"{what}: profile: {f.units} units, not its Program's {audit.units}")
    return f


#: `holds`' refusal, after its window's number and before the rows' port (`rows_name`).
ROWS_UNHELD = "the scan's registration doesn't hold the rows at"
#: `wiredCheck`'s refusal, after its window's number.
UNWIRED = "the scan's prog is no compute audit's slot entries"
#: `stateCheck`'s refusals, after its window's number: the state untied, its reads off the chain, its row 0 unopened.
UNSTATED = "the scan's state is no wired audit's SCAN_STATE"
MISREAD = "the scan's state reads are not its transitions' chain"
UNPINNED = "the scan's state's row 0 is not the empty state"
#: and its refusal of a tied audit reading the state off its slots' anchors, after the audit's number
MISANCHORED = "reads the scan's state off its slots' anchors"


def _state_refusal(p: Params, audits: list[dict], reg: dict) -> str | None:
    """`stateCheck`'s refusal of a link-window's scan: none when its state is tied (`_state_tied`), its row 0
    opened to the empty state (`_state_empty`) and its tied audits read it at their slots' anchors
    (`_unanchored`)."""
    if not _state_tied(p, audits, reg):
        return UNSTATED if _state_read(p, reg) else MISREAD
    if not _state_empty(p, reg):
        return UNPINNED
    return f"tied audit {off[0]} {MISANCHORED}" if (off := _unanchored(p, audits, reg)) else None
#: `heldCheck`'s refusal, after its window's number.
UNHELD = "the tied audit's window circuit is not the registered class circuit"
#: `heldNames`: the checks of a tied audit the rows form holds, sorted, which its public names.
HELD = ("circuit", "free")
#: `tiedDelta`: δ_i, the δ at which the rows form reads a tied audit's draw (`chargedOf`), fixed before any coin.
TIED_DELTA = Fraction(1, 2**64)


def clog2(x: int) -> int:
    """`⌈log2 x⌉`, 0 for x ≤ 1."""
    return 0 if x <= 1 else (x - 1).bit_length()


def at_most(n: int, b: int) -> int:
    """`Σ_{j ≤ b} C(n, j)`: the sets of at most `b` of `n` units."""
    return sum(math.comb(n, j) for j in range(min(b, n) + 1))


def _facts_json(f: Facts) -> dict:
    return {"delta": str(f.delta), "drawn": f.drawn, "units": f.units, "wrong_units": f.wrong}


def _terms(free: int, steps: int, per_unit: int, f: Facts, jitter: int, entries: int | None = None,
           tied: list[int] | None = None) -> dict:
    """`AuditTerms.of` (a compute audit's, `per_unit` its Program's widest unit's committed bits) and `Scan.terms` (a
    scan's, `per_unit` the wider instance's committed bits, `entries` the slot entry bits its tied audits' incorrect
    proof blocks can set, `faulty_entries`, and `tied` those audits by index, `termsJson`'s)."""
    return _facts_json(f) | {"faulty_bits": f.wrong * per_unit, "faulty_sets": clog2(at_most(f.units, f.wrong)),
                             "free": free, "jitter": clog2(jitter ** steps)} | (
        {} if entries is None else {"faulty_entries": entries}) | ({} if tied is None else {"tied": tied})


def public(p: Params, facts: list[Facts], scans: list[tuple[Facts, int, int, list[int]]] | None = None,
           charged: list[Facts] = ()) -> dict:
    """K and its terms over `PeriodLeakAccepted`'s windows: the clock form's compute audits, or the rows form's
    link-windows' scans (`scans`, each its facts, its tied audits' free inputs, its `β_w` and its tied audits by index;
    None in the clock form), where a compute audit's own terms are in none, a tied audit's free inputs are in its
    phase's `free` and its incorrect proof blocks in its `faulty_entries`, and a tied audit drawn in part, read at
    `TIED_DELTA` (`charged`), is in the rows public's `charged` and its δ_i in `eps`."""
    cells = sum(clog2(cap + 1) for cap in p.cells)
    if scans is None:
        units = [_terms(a.free, a.steps, a.bits, f, p.jitter) for a, f in zip(p.audits, facts)]
        schedules = clog2((len(p.sync_set.deltas) * len(p.sync_set.rhos)) ** p.windows)
        out = {"audits": units}
    else:
        s = p.scan
        units = [_terms(s.free + t, s.steps, max(s.ingress, s.egress), f, p.jitter, b, tied)
                 for f, t, b, tied in scans]
        schedules = cells
        out = {"audits": [_facts_json(f) for f in facts], "charged": [_facts_json(f) for f in charged],
               "held": list(HELD), "scans": units}
    K = schedules + cells + sum(t["free"] + t["faulty_sets"] + t["faulty_bits"] + t["jitter"]
                                + t.get("faulty_entries", 0) for t in units)
    eps = sum((Fraction(t["delta"]) for t in units), Fraction(0)) + sum((f.delta for f in charged), Fraction(0))
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
                scans.append((_rows_window_check(p, c.audits, p.first + i, reg, x),
                              _tied_free(p, c.audits, c.windows, reg)))
        except Finding as f:
            findings.append(f)
    if any(f.delta != scans[0][0].delta for f, _ in scans):
        findings.append(Finding(REJECT, "a scan's δ is not the period's"))
    if p.schedule is not None and not rows_wf(p.schedule, p.windows):
        findings.append(Finding(REJECT, rows_wf_refusal(p.schedule, p.windows)))
    charged = {}
    for i, (audit, reg, a) in enumerate(zip(p.audits, c.audits, audits)):
        tied = p.schedule is not None and _tied_in(c.audits, c.windows, i)
        try:
            facts.append(f := _audit_check(i, audit, reg, a))
        except Finding as x:
            findings.append(x)
        else:
            if tied and f.drawn < f.units:
                charged[i] = f.at(TIED_DELTA)
    if p.schedule is not None:
        findings += [Finding(REJECT, f"window {p.first + i}: {UNWIRED}") for i, reg in enumerate(c.windows)
                     if not _wired(p, c.audits, reg)]
        findings += [Finding(REJECT, f"window {p.first + i}: {why}") for i, reg in enumerate(c.windows)
                     if (why := _state_refusal(p, c.audits, reg))]
        findings += [Finding(REJECT, f"window {p.first + i}: {UNHELD}") for i, reg in enumerate(c.windows)
                     if not _held(p, c.audits, reg)]
    if rejected := next((f for f in findings if f.verdict == REJECT), None):
        return REJECT, rejected.reason
    if findings:
        return UNDECIDED, findings[0].reason
    if p.schedule is None:
        return ACCEPT, public(p, facts), {"link": p.link, "windows": p.windows}
    beta = [p.scan.entries if any(i in charged and charged[i].wrong >= 1 for i in _tied_audits(c.audits, reg)) else 0
            for reg in c.windows]
    tied = [_tied_audits(c.audits, reg) for reg in c.windows]
    return ACCEPT, public(p, facts, [(f, t, b, a) for (f, t), b, a in zip(scans, beta, tied)], list(charged.values())), {
        "link": p.link, "windows": p.windows}


def reference(case) -> dict:
    return P.answer(ID, case, params=params, commitment=commitment, evidence=evidence, check=check,
                    digest=lambda c: tagged_sha512(COMMITMENT_TAG, c.bytes))


# --- the listed circuit files --------------------------------------------------------------------------------------------

#: a flat circuit file's head and META `format`, its statement and its blocks' keyword (`Flock.Tags.circuit`)
CIRCUIT_FORMAT, CIRCUIT_STATEMENT, CIRCUIT_BLOCK = "flock-circuit", "verity/flock-circuit", "CIRCUIT"
WORD = 128


@dataclasses.dataclass(frozen=True)
class CircuitHeader:
    """What `network-period check` reads of a circuit file: its SHA-512, META, units per VU and leaf maps, and its unit
    net's header (`headerNet`): its constant's column, input words and port groups, each `(col, words, bits)`; whether
    META's `out_net` is the unit; its row ports (`rowPorts`), each `(name, words, bits)`, and its output rows' count
    (`outRows`); for a file read whole (the scan's, `NetworkCheck.CircuitFile.whole`), its unit's read input columns
    (`Bits.unitReads`), else None (an audit's: its header and its unit's input rows, `NetworkCheck.Period.trim`); and whether
    each of its unit's input columns' rows is a self row, not a forced zero (`CircuitFile.withInputs`), None for a
    header whose input rows weren't read."""
    sha512: str
    meta: dict
    upv: int
    leaves_in: list
    leaves_out: list
    const: int
    in_words: int
    in_groups: list[tuple[int, int, list[int]]]
    out_groups: list[tuple[int, int, list[int]]]
    out_unit: bool
    ports: list[tuple[str, int, int]]
    out_rows: int
    unit_reads: list[bool] | None = None
    in_self: list[bool] | None = None


def _header_net(line: str) -> tuple[int, int, list, list]:
    """`headerNet`: a `flock-ir-unit/v2` net's header line's constant, input words and port groups."""
    h = line.split()
    if not h or h[0] != "flock-ir-unit/v2" or not all(x.isdigit() for x in h[2:]):
        raise D.Reject("not a flock-ir-unit/v2 circuit")

    def groups(at: int) -> tuple[list, int]:
        out = []
        for _ in range(int(h[at])):
            col, words, n = (int(x) for x in h[at + 1:at + 4])
            if words & (words - 1) or col % words:
                raise D.Reject("flock-ir-unit/v2: a port group is not an aligned power of two of words")
            out.append((col, words, [int(x) for x in h[at + 4:at + 4 + n]]))
            at += 3 + n
        return out, at + 1
    ins, at = groups(4)
    outs, end = groups(at)
    useful, const = int(h[2]), int(h[3])
    in_words = ins[-1][0] + ins[-1][1] if ins else 0
    if end != len(h) or not outs or const + 1 != useful:
        raise D.Reject("flock-ir-unit/v2: not a header headerNet reads")
    if any(sum(bs) > w * WORD or c + w > in_words for c, w, bs in ins) or any(sum(bs) > w * WORD for _, w, bs in outs):
        raise D.Reject("flock-ir-unit/v2: a port group's ports overrun its words, or an input group the input rows")
    if in_words * WORD > const:
        raise D.Reject("flock-ir-unit/v2: not a header headerNet reads")
    return const, in_words, ins, outs


def _unit_reads(const: int, in_words: int, rows: list[str]) -> list[bool]:
    """`Bits.unitReads`: of the unit net's input columns, those a computed row (`Rows.ofNet`, its rows from the input
    columns up to the constant) reads in its A or its B, each row `na a… nb b…`."""
    n_in = min(in_words * WORD, const)
    reads = [False] * n_in
    for line in rows[n_in:const]:
        t = [int(x) for x in line.split()]
        for x in t[1:1 + t[0]] + t[2 + t[0]:2 + t[0] + t[1 + t[0]]]:
            if x < n_in:
                reads[x] = True
    return reads


def _input_rows(in_words: int, rows: list[str]) -> list[bool]:
    """`CircuitFile.withInputs`: of the unit net's input columns, whether each one's row is a self row `[c]·[c]`, not a
    forced zero (empty in A and in B); any other input row is refused, as `Flock.Net.ofRows` refuses it."""
    out = []
    for c in range(in_words * WORD):
        t = (rows[c] if c < len(rows) else "").split()
        if not all(x.isdigit() for x in t):
            raise D.Reject(f"unit: flock-ir-unit/v2: row {c} is not numbers")
        v = [int(x) for x in t]
        na = v[0] if v else 0
        nb = v[1 + na] if 1 + na < len(v) else 0
        if not v or len(v) != 2 + na + nb:
            raise D.Reject(f"unit: flock-ir-unit/v2: row {c} is malformed")
        a, b = v[1:1 + na], v[2 + na:]
        if not ((a == [c] and b == [c]) or (not a and not b)):
            raise D.Reject(f"unit: flock-ir-unit/v2: row {c} is not an input row")
        out.append(bool(a))
    return out


def read_circuit(text: str, whole: bool = False) -> CircuitHeader:
    """What the period holds of a flat circuit file: its SHA-512, its META, each block's net header and its first unit's
    input rows (`_input_rows`), no computed row (`NetworkCheck.Period.trim`; `NetworkCheck.CircuitFile.read`, the
    test-only `network-period-chunked`'s); `whole`, the scan's file, its unit's rows too (`unit_reads`). The mirror
    doesn't hold a file to the session verifier's parse (`NetworkCheck.CircuitFile.whole`), which `network-period` reads
    every file with, so a toy file the mirror reads Lean may refuse. A file it can't read is refused (`D.Reject`)."""
    lines = text.split("\n")
    if not lines[0].startswith(CIRCUIT_FORMAT + " ") or len(lines) < 2 or not lines[1].startswith("META "):
        raise D.Reject(f"not a {CIRCUIT_FORMAT} file with a META line")
    meta = json.loads(lines[1][5:])
    if meta.get("format") != CIRCUIT_FORMAT or meta.get("statement") != CIRCUIT_STATEMENT:
        raise D.Reject(f"META: not {CIRCUIT_STATEMENT}")
    nets, reads, in_self, i = {}, None, None, 2
    while i < len(lines) and lines[i]:
        kw, name, n = lines[i].split()
        if kw != CIRCUIT_BLOCK or not 0 < int(n) or i + int(n) >= len(lines):
            raise D.Reject("a block is truncated")
        nets[name] = _header_net(lines[i + 1])
        if name == "unit" and in_self is None:
            in_self = _input_rows(nets[name][1], lines[i + 2:i + 1 + int(n)])
        if whole and name == "unit":
            reads = _unit_reads(nets[name][0], nets[name][1], lines[i + 2:i + 1 + int(n)])
        i += 1 + int(n)
    if "unit" not in nets or meta.get("out_net", "unit") not in nets:
        raise D.Reject("no unit circuit, or META's out_net is no circuit")
    for w in meta.get("wires", []):
        if len(w) != 6:
            raise D.Reject("META: a wire is not six fields")
        if (n := next((n for n in (w[0], w[3]) if n not in nets), None)) is not None:
            raise D.Reject(f"META: unknown circuit {n}")
    for x in meta.get("leaf_cuts", []):
        if len(x) != 4:
            raise D.Reject("META: a leaf cut is not [circuit, unit, port, leaf]")
        if x[0] != "unit":
            raise D.Reject("META: a leaf cut into anything but a unit")
    out_ports = meta.get("out_ports")
    if not isinstance(out_ports, list) or any(not isinstance(p, list) or len(p) != 2 for p in out_ports):
        raise D.Reject("META: an out port is not [name, words]")
    return CircuitHeader(hashlib.sha512(text.encode()).hexdigest(), meta, _nat(meta.get("units_per_vu"), "META"),
                         meta["leaves_in"], meta["leaves_out"], *nets["unit"], meta.get("out_net", "unit") == "unit",
                         _row_ports(meta), len(meta["out_row_ports"]) if "out_row_ports" in meta
                         else len(out_ports) - len(meta.get("pinned_outputs", [])), reads, in_self)


def _row_ports(meta: dict) -> list[tuple[str, int, int]]:
    """`CircuitFile.rowPorts`: META's row ports, each `(name, words, bits)`, its bits a bit row's length
    (`row_schemas`) or 16 per word."""
    ports = meta.get("ports")
    if not isinstance(ports, list) or any(not isinstance(p, list) or len(p) < 2 for p in ports):
        raise D.Reject("circuit META: a port is not [name, words, …]")
    schemas = meta.get("row_schemas")
    if schemas is not None and len(schemas) != len(ports):
        raise D.Reject("META: row_schemas is not one [schema, bits] per port")
    return [(p[0], p[1], schemas[i][1] if schemas is not None else 16 * p[1]) for i, p in enumerate(ports)]


def _n_in(c: CircuitHeader) -> int:
    """`HmOut.nIn`: the input row ports, those before the output rows."""
    return len(c.ports) - c.out_rows


def _starts(c: CircuitHeader) -> list[int]:
    """`KnownRead.starts`: each row port's first word among the rows'."""
    return [sum(w for _, w, _ in c.ports[:q]) for q in range(len(c.ports))]


def _port_cols(groups) -> list[tuple[int, int]]:
    """`Flock.portCols`: each port's first column and width, a group's ports from its column on."""
    out = []
    for col, _, bits in groups:
        at = col * WORD
        for w in bits:
            out.append((at, w))
            at += w
    return out


def param_of(name: str) -> str:
    """`Bits.paramOf`: a row port's parameter, its name up to its first `#`."""
    return name.split("#")[0]


def runs_of(c: CircuitHeader) -> list[tuple[str, int, int]]:
    """`Bits.runsOf`: the input row ports in order, consecutive ones of one parameter merged, each its parameter, its
    words and its first word among the rows'."""
    st, runs = _starts(c), []
    for q in range(_n_in(c)):
        nm, w, _ = c.ports[q]
        if runs and runs[-1][0] == param_of(nm):
            runs[-1] = (runs[-1][0], runs[-1][1] + w, runs[-1][2])
        else:
            runs.append((param_of(nm), w, st[q]))
    return runs


def field_bit(upv: int, words: list[tuple[str, int]], f: str, beta: int) -> tuple[int, int, int] | None:
    """`Bits.fieldBit`: bit `beta` of field `f`'s rows on the unit, `(k, P + j % w, beta % 16)` with `j = beta / 16`,
    `k = j / w` and `P` the words of the fields before `f`; None for no such field or a bit past its rows."""
    i = next((i for i, (nm, _) in enumerate(words) if nm == f), None)
    if i is None or (w := words[i][1]) == 0 or upv * w <= beta // 16:
        return None
    return beta // 16 // w, sum(x for _, x in words[:i]) + beta // 16 % w, beta % 16


def _msg_at(c: CircuitHeader, starts: list[int], x: int, t: int, wide: bool) -> bool:
    """`Bits.msgAt` is a message bit: leaf word `x`'s bit `t` within the bits of the row port holding the word."""
    q = max(sum(1 for s in starts if s <= x) - 1, 0)
    s = starts[q] if q < len(starts) else 0
    return (wide or t < 16) and 16 * max(x - s, 0) + t < (c.ports[q][2] if q < len(c.ports) else 0)


def _src_msg(c: CircuitHeader, starts: list[int], cuts: dict, k: int, p: int, t: int) -> bool:
    """`Bits.srcAt` is a message bit: a leaf port's leaf word's, else its first leaf cut's."""
    if p < (len(c.in_groups[0][2]) if c.in_groups else 0):
        row = c.leaves_in[k] if k < len(c.leaves_in) else []
        x = row[p] if p < len(row) else 0
        return x >= 0 and _msg_at(c, starts, x, t, True)
    return (k, p) in cuts and _msg_at(c, starts, cuts[(k, p)], t, False)


def params_why(c: CircuitHeader, params: list) -> str | None:
    """`Bits.paramsWhy`: at each bit of parameter `π`'s rows, its port (`field_bit`) reads the row word it is, and each
    of `π`'s bits is a message bit of its row."""
    starts, runs = _starts(c), runs_of(c)
    cut = len(c.in_groups[0][2]) if c.in_groups else 0
    cuts: dict[tuple[int, int], int] = {}
    for _, k, p, w in c.meta.get("leaf_cuts", []):
        cuts.setdefault((k, p), w)
    ws = [(nm, words_of(t)) for nm, t in params]
    for j, (nm, t) in enumerate(params):
        w, s = words_of(t), runs[j][2] if j < len(runs) else 0
        for beta in range(16 * c.upv * w):
            if (fb := field_bit(c.upv, ws, nm, beta)) is None:
                return f"bit {beta} of parameter {nm}'s rows is on no input port"
            k, p, b = fb
            if b == 0:
                row = c.leaves_in[k] if k < len(c.leaves_in) else []
                x = (row[p] if p < len(row) else -1) if p < cut else cuts.get((k, p), -1)
                if x != s + beta // 16:
                    return (f"template instance {k}'s input port {p} reads row word {x}, not word {beta // 16} of its "
                            f"parameter {nm}'s rows")
            at = 16 * (beta // 16 % w) + b
            if at < bits_of(t) and not _src_msg(c, starts, cuts, k, p, b):
                return f"bit {at} of parameter {nm} of template instance {k} is no message bit of its rows"
    return None


def bits_why(c: CircuitHeader, defs: dict[str, tuple[list, list]], template: str) -> str | None:
    """`SampledProofs.Bits.check` on a file's header: why Bits doesn't lay it out as `template` of a Program whose
    Definitions are `defs` (`def_sig` by key), or None."""
    n_in = min(c.in_words * WORD, c.const)
    gi = c.in_groups[0][2] if c.in_groups else []
    go = c.out_groups[0][2] if c.out_groups else []
    outs = [x for u in c.leaves_out for x in u]
    out_col = (c.out_groups[0][0] if c.out_groups else 0) * WORD
    in_rows = sum(w for _, w, _ in c.ports[:_n_in(c)])
    if template not in defs:
        return "the circuit file's template is no Definition of the Program"
    if c.upv == 0:
        return "the circuit file's unit holds no template instance"
    if any(b != 16 for b in gi + go):
        return "a port of the circuit file's unit is not one 16-bit word"
    if len(c.leaves_in) != c.upv or len(c.leaves_out) != c.upv or any(
            len(c.leaves_in[k]) != len(gi) or len(c.leaves_out[k]) != len(go) for k in range(c.upv)):
        return "the circuit file's leaf maps are not one per template instance over its unit's ports"
    if any(x >= in_rows for u in c.leaves_in for x in u):
        return "a leaf map names a word past the circuit file's input rows"
    if any(w[0] == "unit" for w in c.meta.get("wires", [])):
        return "a wire into the circuit file's unit"
    pc = [b for g in c.in_groups for b in g[2]]
    if any(k >= c.upv or p < len(gi) or p >= len(pc) or pc[p] < 16 or w >= in_rows
           for _, k, p, w in c.meta.get("leaf_cuts", [])):
        return "a leaf cut outside the unit's cut inputs or the input rows"
    cols = _port_cols(c.in_groups)
    if any(cols[p + 1][0] < cols[p][0] + cols[p][1] for p in range(len(cols) - 1)):
        return "the circuit file's unit's input ports overlap or are out of column order"
    wired: dict[tuple[int, int], int] = {}
    for kp in [(x[1], x[2]) for x in c.meta.get("leaf_cuts", [])]:
        wired[kp] = wired.get(kp, 0) + 1
    for k in range(c.upv):
        for p in range(len(gi), len(pc)):
            if wired.get((k, p), 0) != 1:
                return (f"input port {p} of the circuit file's unit, template instance {k}, is not wired exactly once by "
                        "a leaf cut")
    if len(set(outs)) != len(outs):
        return "the circuit file's unit returns an output word twice"
    if n_in != c.in_words * WORD:
        return "the circuit file's unit net has its constant among its input words"
    if not c.out_unit:
        return "the circuit file returns its outputs from a net other than its unit"
    if not all(n_in <= out_col + 16 * j + t < c.const for u in c.leaves_out for j in range(len(u)) for t in range(16)):
        return "a returned output bit of the circuit file's unit sits on no computed row"
    params, ret = defs[template]
    rw = ret_words(ret)
    if len({p[0] for p in c.ports}) != len(c.ports):
        return "two row ports of the circuit file have one name"
    if [(nm, w) for nm, w, _ in runs_of(c)] != [(nm, c.upv * words_of(t)) for nm, t in params]:
        return (f"the circuit file's input rows are not its template's parameters in order, each {c.upv} times its "
                "packed words")
    if sum(w for _, w, _ in c.ports[_n_in(c):]) != c.upv * rw:
        return f"the circuit file's output rows are not {c.upv} times its template's return's packed words"
    if any(len(u) != rw for u in c.leaves_out):
        return "a template instance's output leaves are not its template's return's packed words"
    if len(pc) != sum(words_of(t) for _, t in params):
        return "the circuit file's unit's input ports are not its template's parameters' packed words"
    if (why := params_why(c, params)) is not None:
        return why
    if c.upv != 1:
        return f"the circuit file has {c.upv} template instances a VU, not one"
    return None


def _port_held(c: CircuitHeader) -> list[bool]:
    """Of the unit net's input columns, whether a port holds each (`Flock.portCols`)."""
    held = [False] * min(c.in_words * WORD, c.const)
    for a0, w in _port_cols(c.in_groups):
        for t in range(w):
            if a0 + t < len(held):
                held[a0 + t] = True
    return held


def pad_cols(c: CircuitHeader) -> int:
    """`padCols`: the unit net's input columns no port holds whose row is not a forced zero (every one when its input
    rows weren't read), each a free input of each VU."""
    held = _port_held(c)
    return sum(1 for x, h in enumerate(held) if not h and (c.in_self is None or c.in_self[x]))


def reads_why(c: CircuitHeader, params: list, reads: list[bool], names: list[str]) -> str | None:
    """`SampledProofs.Bits.readsWhy`: the unit's read input columns `reads` are not one per input column, a computed row
    reads an input column no port holds (input padding), or no row reads a bit of a parameter in `names`."""
    n_in = min(c.in_words * WORD, c.const)
    pc = _port_cols(c.in_groups)
    if len(reads) != n_in:
        return "the circuit file's unit's read columns are not one per input column"
    held = _port_held(c)
    if (x := next((x for x in range(n_in) if reads[x] and not held[x]), None)) is not None:
        return f"a row of the circuit file's unit reads its input column {x}, which no port holds"
    at = 0
    for nm, t in params:
        if nm in names:
            for beta in range(bits_of(t)):
                col = (pc[at + beta // 16][0] if at + beta // 16 < len(pc) else 0) + beta % 16
                if not (col < len(reads) and reads[col]):
                    return f"no row of the circuit file's unit reads bit {beta} of its parameter {nm}"
        at += words_of(t)
    return None


def _listing_files(listing: Listing) -> list[tuple[str, bool, Listed, int]]:
    """`Listing.files`: every listed file in the order given, each audit's by index, then the scan's."""
    owners = [(f"audit {i}", False, x) for i, x in enumerate(listing.audits) if x is not None]
    owners += [("the scan", True, listing.scan)] if listing.scan is not None else []
    return [(w, s, x, k) for w, s, x in owners for k in range(len(x.files))]


def circuits_why(listing: Listing, files: list[CircuitHeader]) -> str | None:
    """`circuitsWhy`: why the files are not the period's listed circuit files, or None (`fileWhy`, the first in order:
    its SHA-512, META's `program_sha512`, an audit's file's META partition, `bits_why`, whole VUs, the scan's at one
    instance a VU, read whole and its unit's rows reading no padding and every bit of a `row` parameter, `reads_why`)."""
    want = _listing_files(listing)
    if len(files) != len(want):
        return f"the period lists {len(want)} circuit files, not the {len(files)} given"
    for (what, scan, x, k), c in zip(want, files):
        t, members, _ = x.templates[k]
        at, part = f"{what}'s circuit file for {t}", c.meta.get("partition")
        if c.sha512 != x.files[k]:
            return f"{at}: its SHA-512 is not its listing's"
        if c.meta.get("program_sha512") != x.program:
            return f"{at}: META program_sha512 is not the {what}'s Program's SHA-512"
        if x.partition is not None and not (isinstance(part, dict) and part.get("rule") == PARTITION_V1
                                            and part.get("digest") == x.partition):
            return f"{at}: META partition is not {PARTITION_V1} at its circuit record's digest"
        if (why := bits_why(c, x.defs, t)) is not None:
            return f"{at}: {why}"
        if members % c.upv:
            return f"{at}: its {members} template instances are not whole VUs of {c.upv}"
        if scan and c.upv != 1:
            return f"{at}: {c.upv} template instances a VU, not 1"
        if scan and c.unit_reads is None:
            return f"{at}: its unit net's rows were not read, so what they read can't be checked"
        if scan and (why := reads_why(c, x.defs[t][0], c.unit_reads, ["row"])) is not None:
            return f"{at}: {why}"
    return None


def charged(p: Params, listing: Listing, files: list[CircuitHeader]) -> Params:
    """`charged`: each listed audit's units its VUs, its templates' members over their files' `units_per_vu`, its
    bits the most a wrong VU is charged (`vu_bits`), and each VU's input padding (`pad_cols`) more free inputs, with the
    scan's state held or not."""
    at, audits = 0, []
    for a, x in zip(p.audits, listing.audits):
        if x is None:
            audits.append(a)
            continue
        heads = [(c.upv, pad_cols(c)) for c in files[at:at + len(x.files)]]
        at += len(x.files)
        pad = sum(n * q for (_, n, _), (_, q) in zip(x.templates, heads))
        audits.append(dataclasses.replace(a, units=sum(n // u for (_, n, _), (u, _) in zip(x.templates, heads)),
                                          bits=max(vu_bits(u, w) for (_, _, w), (u, _) in zip(x.templates, heads)),
                                          free=a.free + pad,
                                          state_free=None if a.state_free is None else a.state_free + pad))
    return dataclasses.replace(p, audits=tuple(audits))


def unlisted_tied(p: Params, listing: Listing, c: Commitment) -> int | None:
    """`unlistedTied`: the first audit some link-window's scan ties (`tiedOf`) that lists no circuit files, or None."""
    return next((i for i, x in enumerate(listing.audits)
                 if x is None and p.schedule is not None and _tied_in(c.audits, c.windows, i)), None)


def check_with(files: list[CircuitHeader], pl: tuple[Params, Listing], c: Commitment, e) -> tuple:
    """`checkWith`: an audit tied to the scan that lists no files refused (`unlisted_tied`), the files held to the
    listing (`circuits_why`), then `check` at what they charge (`charged`)."""
    p, listing = pl
    if (i := unlisted_tied(p, listing, c)) is not None:
        return REJECT, (f"circuit files: audit {i} is tied to the scan but lists no circuit files (its circuits), so no "
                        "Bits.check holds its templates' files: list a file for each of its Program's root templates")
    if (why := circuits_why(listing, files)) is not None:
        return REJECT, f"circuit files: {why}"
    return check(charged(p, listing, files), c, e)


def scan_count(params) -> int:
    """`scanCount`: how many files the parameters' `scan_circuits` lists, read before the parameters are: 0 for
    parameters that list none or aren't an object."""
    sc = params.get("scan_circuits") if isinstance(params, dict) else None
    return len(sc) if isinstance(sc, list) else 0


def reference_with(case: dict, texts: list[str]) -> dict:
    """`network-period check PARAMS COMMITMENT EVIDENCE CIRCUIT…`'s answer with the circuit files `texts`, in order
    (`protocolWith`), the last as many as the parameters' `scan_circuits` lists (`scan_count`) read whole; a file it
    can't read (`read_circuit`) refuses the period (`refusing`). With no files it is the check that reads none
    (`reference`)."""
    if not texts:
        return reference(case)
    files, why, whole = [], None, len(texts) - scan_count(case["params"])
    for i, t in enumerate(texts):
        try:
            files.append(read_circuit(t, i >= whole))
        except D.Reject as x:
            why = f"circuit files: circuit file {i}: {x}"
            break
    if why is None:
        chk = lambda pl, c, e: check_with(files, pl, c, e)  # noqa: E731
    else:
        chk = lambda *_: (REJECT, why)  # noqa: E731
    return P.answer(ID, case, params=params_listed, commitment=commitment, evidence=evidence, check=chk,
                    digest=lambda c: tagged_sha512(COMMITMENT_TAG, c.bytes))


def _pow2(x: int) -> int:
    """The least power of two at least `x` (1 for `x ≤ 1`)."""
    return 1 << max(0, (x - 1).bit_length())


def circuit_file(desc: dict, template: str, part: str | None, upv: int = 1, cut: bool = False,
                 cut_group: tuple[int, int] | None = None, in_rows: dict[int, str] | None = None, **meta) -> str:
    """A flat circuit file laying out `template` of the Program `desc` as `SampledProofs.Bits.check` places it, at `upv`
    template instances a VU: an input row a parameter, `upv` times its packed words (`words_of`), then an output row `o`
    of `upv` times its return's (`ret_words`); a unit net of a 16-bit input port a parameter word, each instance's read
    off its row word by its leaf map (with `cut`, the last port a cut input in an input group of its own, at
    `cut_group`'s word and width or else one word past the first group's, read through a leaf cut; the last group ends
    the input rows), and a 16-bit output port a return word, its sixteen computed rows reading the input bits congruent
    to theirs, so every port bit is read and no padding, each padding row a forced zero; META binding the Program's
    SHA-512 and, given one, the `verity/partition/v1` digest `part`. `in_rows` overrides input rows by column, `meta`
    META's keys."""
    params, ret = def_sig(desc["definitions"][template])
    pw, rw = [(nm, words_of(t)) for nm, t in params], ret_words(ret)
    n = sum(w for _, w in pw)
    starts = [upv * sum(w for _, w in pw[:i]) for i in range(len(pw))]
    words = [[starts[i] + k * w + j for i, (_, w) in enumerate(pw) for j in range(w)] for k in range(upv)]
    leaf = n - 1 if cut else n
    g0 = _pow2(-(-16 * leaf // WORD))
    ins = [(0, g0, [16] * leaf)] + [(*(cut_group or (g0, 1)), [16])] * cut
    in_words, ow = ins[-1][0] + ins[-1][1], _pow2(-(-16 * rw // WORD))
    oc = -(-in_words // ow) * ow
    const = oc * WORD + 16 * rw
    held = list(range(16 * leaf)) + [ins[-1][0] * WORD + t for t in range(16 if cut else 0)]
    groups = lambda gs: [len(gs), *(x for col, w, bs in gs for x in (col, w, len(bs), *bs))]  # noqa: E731
    head = " ".join(map(str, ["flock-ir-unit/v2", "toy", const + 1, const, *groups(ins), *groups([(oc, ow, [16] * rw)])]))
    reader = lambda xs: f"{len(xs)} {' '.join(map(str, xs))} 1 {const}"  # noqa: E731
    hs = set(held)
    net = [head] + [(in_rows or {}).get(i, f"1 {i} 1 {i}" if i in hs else "0 0") for i in range(in_words * WORD)] + [
        reader(held[:1] or [const]) for _ in range(in_words * WORD, oc * WORD)] + [
        reader(held[r::16 * rw] or [const]) for r in range(16 * rw)] + [f"1 {const} 1 {const}"]
    m = {"format": CIRCUIT_FORMAT, "statement": CIRCUIT_STATEMENT, "units_per_vu": upv,
         "ports": [[nm, upv * w, 0] for nm, w in pw] + [["o", upv * rw, 0]], "out_ports": [["o", upv * rw]],
         "leaves_in": [u[:leaf] for u in words], "leaves_out": [[k * rw + j for j in range(rw)] for k in range(upv)],
         **({"leaf_cuts": [["unit", k, n - 1, words[k][n - 1]] for k in range(upv)]} if cut else {}),
         "program_sha512": program_sha512(desc),
         **({"partition": {"rule": PARTITION_V1, "digest": part}} if part else {})}
    return "\n".join([f"{CIRCUIT_FORMAT} toy", "META " + json.dumps(m | meta, separators=(",", ":")),
                      f"{CIRCUIT_BLOCK} unit {len(net)}", *net]) + "\n"


def listed_files(case: dict, upv: int = 1, cut: bool = False, cut_group: tuple[int, int] | None = None,
                 in_rows: dict[int, str] | None = None, **meta) -> list[str]:
    """The circuit files `circuit_file` writes for each file a case's audits list, in `Listing.files`' order: each its
    template of the audit's Program, bound to its record's partition digest."""
    return [circuit_file(a["program"], f["template"], record_partition(a["circuit"], ""), upv, cut, cut_group, in_rows,
                         **meta) for a in case["params"]["audits"] if "circuits" in a for f in a["circuits"]]


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
    "audit-record": ACCEPT, "audit-record-bits": ACCEPT,
    "audit-record-circuit": REJECT, "audit-record-pin": REJECT, "audit-record-state": REJECT,
    "undecided-and-reject": REJECT, "form": REJECT,
    "first": REJECT, "count-windows": REJECT, "count-audits": REJECT, "evidence-count": REJECT,
    "params": REJECT, "commitment": REJECT, "evidence": REJECT, "case": REJECT,
    "rows-honest": ACCEPT, "rows-tied-partial": ACCEPT, "rows-tied-sampled": ACCEPT,
    "rows-tied-entries-sampled": ACCEPT, "rows-tied-slots-sampled": ACCEPT,
    "rows-program": REJECT,
    "rows-cert-missing": REJECT, "rows-cert-swap": REJECT, "rows-cert-fails-link": REJECT,
    "rows-cert-direction": REJECT, "rows-cert-forged": REJECT, "rows-cert-window": REJECT, "rows-cert-session": REJECT,
    "rows-cert-key": REJECT, "rows-past-slots": REJECT,
    "rows-reg-count": REJECT, "rows-reg-swap": REJECT, "rows-reg-forged": REJECT, "rows-reg-window": REJECT,
    "rows-reg-link": REJECT, "rows-reg-session": REJECT, "rows-reg-key": REJECT, "rows-reg-field": REJECT,
    "rows-schedule": REJECT,
    "rows-answer-protocol": REJECT, "rows-answer-other": REJECT, "rows-answer-window": REJECT, "rows-port": REJECT,
    "rows-answer-reject": REJECT, "rows-answer-undecided": UNDECIDED, "rows-answer-profile": REJECT,
    "rows-answer-bound": REJECT, "rows-delta": REJECT, "rows-units": REJECT, "rows-prog-untied": REJECT,
    "rows-prog-unwired": REJECT, "rows-state-untied": REJECT, "rows-state-misindexed": REJECT,
    "rows-state-unpinned": REJECT, "rows-state-anchored": ACCEPT, "rows-state-misanchored": REJECT,
    "rows-state-unheld": ACCEPT,
    "rows-circuit-swapped": REJECT,
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
    progs = (Array(S.PROG, BIT),) * (len(S.NAMES) - 1)
    return tuple(zip(S.NAMES, (Array(S.ROW, BYTE_BITS), *progs))), _ingress_ret(S)


def _ingress_sig(S):
    progs = (Array(S.PROG, BIT),) * (len(S.NAMES) - 2)
    return tuple(zip(S.NAMES, (Array(S.ROW, BYTE_BITS), Array(S.STATE, BIT), *progs))), _ingress_ret(S)


def _egress_sig(S):
    return tuple(zip(S.NAMES, (Array(S.WORDS, U16_BITS),))), Tuple(U16_BITS, Array(S.ROW, BYTE_BITS))


def _first(B, S, row, *progs):
    b, p = row[0][0], progs[0][0]
    return tuple_of(B.call(bind(SPREAD, N=OUT_BITS), b, p), B.batch(MIX, p, b, axes=(None, None), n=S.STATE),
                    B.batch(MIX, p, b, axes=(None, None), n=S.VIEW))


def _ingress(B, S, row, sin, *progs):
    b, p = row[0][0], progs[0][0]
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
#: which `NetTiming.BoundaryGates.Fits` refuses).
CHECKER_INGRESS = CompositeDefinition(
    "NetworkPeriodScanIngress", 0, _STATICS, _checker_ingress_sig,
    lambda B, S, row, sin, prog, sout, view: B.call(BL.XOR, BL.operands(sin[0], sout[0])), register=False)
CHECKER_EGRESS = CompositeDefinition(
    "NetworkPeriodScanEgress", 0, ("NAMES", "WORDS", "ROW"), _checker_egress_sig,
    lambda B, S, state, row: B.call(bind(SPREAD, N=OUT_BITS), state[0][0], row[0][0]), register=False)


def _scan_root(members: list[tuple], lanes: int, nb: int, every: int, prog: int, skew: int = 0,
               shared: bool = False) -> Program:
    """The scan's root in the generator form: each `(template, n, fixed)` of `members` one batch over its buckets, in
    order, its first `fixed` parameters inputs of its own and each later one, lane j's `prog<j>`, at bucket t row
    `⌊t / every⌋` (the last row past the rows), or `skew` rows on, cyclically, of lane j's input `prog<j>` of the
    slot entries, `nb` rows of `prog` bits (lane 0's for every lane, `shared`)."""
    sigs = [t.signature() for t, _, _ in members]

    def signature(S):
        params = tuple((f"{i}.{name}", Array(n, ty)) for i, ((ps, _), (_, n, fixed)) in enumerate(zip(sigs, members))
                       for name, ty in ps[:fixed])
        return (*params, *((f"prog{j}", Array(nb, Array(prog, BIT))) for j in range(lanes))), \
            Tuple(*(Array(n, ret) for (_, ret), (_, n, _) in zip(sigs, members)))

    def body(B, S, *args):
        entries, outs, at, start = args[len(args) - lanes:], [], 0, 0
        for (t, n, fixed), (ps, _) in zip(members, sigs):
            reads = [array_of([entries[0 if shared else j][(min((start + m) // every, nb - 1) + skew) % nb]
                               for m in range(n)]) for j in range(len(ps) - fixed)]
            outs.append(B.batch(t, *args[at:at + fixed], *reads, axes=(0,) * len(ps)))
            at, start = at + fixed, start + n
        return coll_concat(signature(S)[1], outs)

    defn = CompositeDefinition("NetworkPeriodScan", 0, ("templates", "counts", "lanes", "nb", "every", "skew", "shared"),
                               signature, body, register=False)
    return Program(bind(defn, templates=tuple(t.id for t, _, _ in members), counts=tuple(n for _, n, _ in members),
                        lanes=lanes, nb=nb, every=every, skew=skew, shared=shared))


@lru_cache(maxsize=None)
def scan_descriptor(T: int, row: int, widths: tuple[int, int, int, int], counts: tuple[int, ...] | None = None,
                    names: tuple[tuple[str, ...], ...] | None = None, checker: bool = False, lanes: int = 1,
                    every: int = 1, skew: int = 0, shared: bool = False, egress: int = 2) -> str:
    """The canonical descriptor of a scan's Program in the generator form: bucket 0's stand-in instance, `T − 1` later
    ingress instances (no template when there are none) and `T` egress instances (or `counts`), each ingress template's
    row `row` bytes, the ingress buckets' state, entry and view bits and the egress bucket's state words `widths`, and
    `lanes` entry ports a bucket, bucket t's `prog<j>` row `⌊t / every⌋` of lane j's input `prog<j>` of the slot
    entries (`_scan_root`), or `skew` rows on, cyclically, or lane 0's input (`shared`); `egress`: the template the
    egress instances are, bucket 0's (0), the later ingress buckets' (1) or the egress bucket's (2); `checker`: in the
    checker form instead, `T` ingress and `T` egress instances (or `counts`) that take their rows."""
    state, prog, view, words = widths
    ingress = {"ROW": row, "STATE": state, "PROG": prog, "VIEW": view}
    names = names or (first_params(lanes), ingress_params(lanes), EGRESS_PARAMS)
    if checker:
        templates = (bind(CHECKER_INGRESS, NAMES=CHECKER_PARAMS[0], **ingress),
                     bind(CHECKER_EGRESS, NAMES=CHECKER_PARAMS[1], WORDS=words, ROW=row))
        program = population.mixed_population_program([(t, n) for t, n in zip(templates, counts or (T, T)) if n])
        return D.canonical(codec.encode_program(program))
    templates = (bind(FIRST_BUCKET, NAMES=names[0], **ingress), bind(INGRESS_BUCKET, NAMES=names[1], **ingress),
                 bind(EGRESS_BUCKET, NAMES=names[2], WORDS=words, ROW=row))
    fixed = (1, 2, len(names[2]))
    kept = [(t, n, f) for t, n, f in zip((*templates[:2], templates[egress]), counts or (1, T - 1, T),
                                         (*fixed[:2], fixed[egress])) if n]
    return D.canonical(codec.encode_program(_scan_root(kept, lanes, T // every, every, prog, skew, shared)))


def _unit_sig(S):
    return (("x", Array(1, U16)),), Array(S.OUT, BYTE)


#: A compute audit's stand-in unit: it returns `OUT` bytes, a proof block's 16-bit public output (`OUT` = 2) or more
#: or fewer, each one gate, and commits its 16-bit parameter besides.
AUDIT_UNIT = CompositeDefinition("NetworkPeriodAuditUnit", 0, ("OUT",), _unit_sig,
                                 lambda B, S, x: B.batch(LOW, x[0], axes=(None,), n=S.OUT), register=False)
#: The stand-in unit's widths in bytes: one under, at and over the public output's 16 bits.
UNIT_BYTES = (1, 2, 5, 40)
#: The slot entries' template of the cases' audits: `io_shell.SlotEntry_v1` at its smallest circuit-check row, a pinned
#: Definition (`SLOT_ENTRY_PINS`' first), whose population of one member's descriptor is `fixtures/slot_entry.json`.
SLOT_KEY = "SlotEntry_v1{KO=2,IDB=17,NP=4,STOP=[]}"
SLOT_BINDING = {"KO": 2, "IDB": 17, "NP": 4, "STOP": ()}
#: `SLOT_KEY` as Verity's catalog binds it (`interface.resolve`), a circuit record's template
SLOT_TEMPLATE = bind(io_shell.SlotEntry, **SLOT_BINDING)
#: The select entries' template of the cases' audits that select a slot's token: `SelectEntry_v1` at its smallest
#: binding, a pinned Definition (`SELECT_ENTRY_PINS`' first) ending in the `SlotEntry_v1` that writes the read bit,
#: whose population of one member's descriptor is `fixtures/select_entry.json`.
SELECT_KEY = "SelectEntry_v1{V=256,C=16,KO=2,FIRST=false,IDB=17,NP=4,STOP=[]}"
SELECT_BINDING = {"V": 256, "C": 16, "KO": 2, "FIRST": False, "IDB": 17, "NP": 4, "STOP": ()}


def _stand_in(name: str, version: int, statics: tuple[str, ...], out=lambda S: 2) -> CompositeDefinition:
    """A template under `name`'s Definition id with the stand-in unit's body, committing `out(S)` bytes."""
    return CompositeDefinition(name, version, statics, lambda S: ((("x", Array(1, U16)),), Array(out(S), BYTE)),
                               lambda B, S, x: B.batch(LOW, x[0], axes=(None,), n=out(S)), register=False)


#: Templates under the slot entry's and the select entry's names and descriptor ids (`SLOT_KEY`, `SELECT_KEY`) with
#: another body, the stand-in unit's.
RENAMED_SLOT_ENTRY = _stand_in("SlotEntry", 1, tuple(SLOT_BINDING))
RENAMED_SELECT_ENTRY = _stand_in("SelectEntry", 1, tuple(SELECT_BINDING))
#: A stand-in of the scan slots' template under `ScanSlots_v2`'s Definition id, committing `NS` bytes.
SCAN_SLOTS_UNIT = _stand_in("ScanSlots", 2, ("NS",), lambda S: S.NS)


@lru_cache(maxsize=None)
def _fixture(name: str, key: str):
    """Template `key`'s Definition, decoded from its fixture `name` (`BitAtx32_v1` is a primitive the registry makes on
    demand)."""
    bit_at(32)
    desc = json.loads((Path(__file__).resolve().parent / "fixtures" / name).read_text())
    return codec.decode_program(desc, REGISTRY).defs[key]


def slot_entry():
    """`SLOT_KEY`'s Definition, decoded from `fixtures/slot_entry.json`."""
    return _fixture("slot_entry.json", SLOT_KEY)


def select_entry():
    """`SELECT_KEY`'s Definition, decoded from `fixtures/select_entry.json`."""
    return _fixture("select_entry.json", SELECT_KEY)


@lru_cache(maxsize=None)
def audit_descriptor(units: tuple[tuple[int, int], ...], entries: tuple[int, ...] = (), renamed: bool = False,
                     select: bool = False, scan_slots: tuple[int, ...] = ()) -> str:
    """The canonical descriptor of a compute audit's Program over the stand-in unit: `n` instances committing `out`
    bytes each, for each `(n, out)` of `units` (each `out` its own template), a scan slots' template of `n` members for
    each `n` of `scan_slots` (`SCAN_SLOTS_UNIT`), and a slot entries' template of `n` members for each `n` of `entries`
    (`slot_entry`, with `select` `select_entry`, or with `renamed` `RENAMED_SLOT_ENTRY` or `RENAMED_SELECT_ENTRY`),
    one template as the window's population program (`population_program`, `TemplateInstances_v1`), several as a mixed
    one."""
    if renamed:
        entry = bind(RENAMED_SELECT_ENTRY, **SELECT_BINDING) if select else bind(RENAMED_SLOT_ENTRY, **SLOT_BINDING)
    else:
        entry = select_entry() if select else slot_entry()
    templates = [(bind(AUDIT_UNIT, OUT=out), n) for n, out in units] + \
        [(bind(SCAN_SLOTS_UNIT, NS=1 + j), n) for j, n in enumerate(scan_slots)] + [(entry, n) for n in entries]
    program = population.population_program(*templates[0]) if len(templates) == 1 else \
        population.mixed_population_program(templates)
    return D.canonical(codec.encode_program(program))


RECORD_GEMM = bind(GM.Gemm, K=16, N=16, **GM.GEMM_AMPERE)
RECORD_BITS = bind(BE.Bf16Add)
#: the record-form audit's units: `record_audit`'s, and `record_audit(bits=True)`'s
RECORD_UNITS = {False: 8, True: 17}


@lru_cache(maxsize=None)
def record_audit(bits: bool = False) -> tuple[str, str, str]:
    """A record-form audit's Program, circuit record and wire order, each canonical JSON: two requests' two steps through
    two layers of one `Gemm_v2{K=16, N=16}` template (`RECORD_GEMM`), unit (layer, step, request) at 4·layer + 2·step +
    request, layer 1 reading layer 0's output and layer 0 at step 1 layer 1's at step 0, layer 0's input at step 0
    external; the record `interface.descriptor_record`'s and the Program its population program, and the order one
    segment ranking each unit at 2·layer + 4·step + request, so every wire's writer ranks below its reader. With
    ``bits``, one Gemm unit whose ``x``, cut 16 ways, reads 16 `Bf16Add_v2` units' outputs (`RECORD_BITS`), a
    Boolean-profile writer's 16 one-bit leaves a row for the reader's one 16-bit leaf, ranked below it."""
    gid = codec.descriptor_id(RECORD_GEMM)
    if bits:
        bid = codec.descriptor_id(RECORD_BITS)
        c = IF.Circuit("period-record-bits", ((RECORD_GEMM, 1), (RECORD_BITS, 16)), external=(f"{bid}/a", f"{bid}/b"),
                       cuts={gid: {"x": 16}}, wires=(
            IF.Wire(gid, "x", bid, "o0", at=0, radix=(16,), base=0, step=(1,), cut=(0, 16)),))
        order = {"format": "verity/wire-order/v1", "segments": [
            {"template": bid, "at": 0, "radix": [16], "c": 0, "r": [0]},
            {"template": gid, "at": 0, "radix": [1], "c": 1, "r": [0]}], "wires": [[0, 1]]}
    else:
        c = IF.Circuit("period-record", ((RECORD_GEMM, RECORD_UNITS[False]),), external=(f"{gid}/x",), wires=(
            IF.Wire(gid, "x", gid, "o0", at=4, radix=(2, 2), base=0, step=(2, 1)),
            IF.Wire(gid, "x", gid, "o0", at=2, radix=(1, 2), base=4, step=(2, 1))))
        order = {"format": "verity/wire-order/v1", "segments": [
            {"template": gid, "at": 0, "radix": [2, 2, 2], "c": 0, "r": [2, 4, 1]}], "wires": [[0, 0], [0, 0]]}
    assert sum(n for _, n in c.templates) == RECORD_UNITS[bits]
    return (D.canonical(codec.encode_program(c.program()[0])), D.canonical(IF.descriptor_record(c, 2)),
            D.canonical(order))


#: The slot entries' port a rows-form tied audit's record reads the scan's state into.
ENTRY_STATE = "state"


@lru_cache(maxsize=None)
def anchor_audit(n: int, lanes: int, every: int, T: int, idle: bool = False) -> tuple[str, str, str]:
    """A rows-form tied audit's Program, circuit record and wire order, each canonical JSON: `n` units of `RECORD_GEMM`
    and the slot entries' template of a slot each (`SLOT_TEMPLATE`, lanes · nb members, nb = T / every), the record
    registering the scan's state (`SCAN_STATE`, T rows of its `ENTRY_STATE` port's W words) and reading it at the slots'
    anchors, unit l · nb + b from row b · every (radix [lanes, nb], step [0, every · W]; `idle`: a digit of radix 1
    between), the Program its population program and the order no segment and no wire."""
    nb, W = T // every, dict(IF.layout(SLOT_TEMPLATE).rows)[ENTRY_STATE]
    radix, step = ((lanes, 1, nb), (0, 1, every * W)) if idle else ((lanes, nb), (0, every * W))
    reg = IF.Registered({SCAN_STATE: T * W}, reads=(IF.Read(SLOT_KEY, ENTRY_STATE, SCAN_STATE, 0, radix, 0, step),))
    c = IF.Circuit("period-anchored", ((RECORD_GEMM, n), (SLOT_TEMPLATE, lanes * nb)), registered=reg)
    order = {"format": "verity/wire-order/v1", "segments": [], "wires": []}
    return (D.canonical(codec.encode_program(c.program()[0])), D.canonical(IF.descriptor_record(c, 1)),
            D.canonical(order))


def _misanchor(how: str, registered: dict, s: WardenParams) -> dict:
    """`registered`, an `anchor_audit` record's, with its read of the scan's state put off the anchors as `how`
    (`MISANCHORS`) says, `s` a schedule it needs (`misanchor_schedule`)."""
    (x,), W = registered["reads"], registered["params"][SCAN_STATE] // s.window
    if how == "late":
        x = x | {"base": W}
    elif how == "lane":
        x = x | {"step": [W, *x["step"][1:]]}
    elif how == "swapped":
        x = x | {"radix": x["radix"][::-1], "step": x["step"][::-1]}
    elif how == "output":
        x = x | {"port": "o0"}
    elif how == "params":
        return registered | {"params": {SCAN_STATE: (s.window + 1) * W}}
    else:
        return registered | {"tables": [{"name": "state", "param": SCAN_STATE, "offset": 0, "words": W,
                                         "rows": s.window}]}
    return registered | {"reads": [x]}


def misanchor_schedule(how: str, s: WardenParams) -> WardenParams:
    """`s`, or a schedule beside it where `how` needs one to move its read off the anchors within the state: every 2
    or more for a slot reading the row after its anchor's (`late`), two lanes too for lane 1's slots reading it
    (`lane`), and two lanes and two boundaries to swap (`swapped`)."""
    if how == "late" and s.every < 2:
        return dataclasses.replace(s, window=2, every=2)
    if how == "lane" and (s.lanes != 2 or s.every < 2):
        return dataclasses.replace(s, lanes=2, window=2, every=2)
    if how == "swapped" and (s.lanes < 2 or s.window // s.every < 2):
        return dataclasses.replace(s, lanes=2, window=2, every=1)
    return s


def templates_of(desc: dict) -> list[tuple[str, str, int]]:
    """A Program's root's templates, in order, each its descriptor id, its Definition's id and its members: one-stage's
    strata, one a template, named by descriptor id."""
    defs, callees = desc["definitions"], desc["callees"]
    return [(k, d["id"], _members(x)) for x in defs[desc["root"]]["body"]["nodes"]
            if (d := defs[k := callees[x["fn"]]])["form"] != "primitive"]


#: How `rows-prog-untied` breaks its link-window's tie of the scan's `prog` to an audit's output, at every lane: the
#: scan's `scan/prog<j>` entries at another root or domain, the audit's chunks' entries at ports that are no unit's
#: output nor a chunk's entries port, no `scan/prog<j>` entry or each under another name (`scan/prog` among them),
#: the chunks' entries in the scan's own registration and in no audit's, or no entry's root a string
UNTIES = ("root", "domain", "port", "missing", "renamed", "scan", "untyped")
#: How `rows-prog-unwired` breaks its link-window's wiring at one lane j while its tie holds: the audit's chunk j
#: entry at an output port (another unit's `o1`, or the chunk's own), at another `SlotEntry_v1` or `SelectEntry_v1`
#: template's entries port than any of its Program's, at another chunk's (`chunk`), of other than T / every leaves
#: (`leaves`), the scan's lane j entry named another lane's (`lane`), or, in the class circuit every tied audit runs,
#: its slot entries' template one member off a slot each, or that template under its name and descriptor id with
#: another body (`RENAMED_SLOT_ENTRY`, `RENAMED_SELECT_ENTRY`)
UNWIRES = ("output", "key", "chunk", "leaves", "lane", "members", "body")
#: How `rows-state-untied` breaks its link-window's tie of the scan's state to its wired audit's `SCAN_STATE` while its
#: `prog` stays wired: the scan's state entry at another root or domain (a state tree that skips or alters one of the
#: scan's transitions), no state entry or one under another name, at another audit's `SCAN_STATE`, its root no string,
#: or a second state entry at another root; the wired audit's `SCAN_STATE` absent, at another root, or beside a second
UNSTATES = ("root", "domain", "missing", "renamed", "other", "untyped", "extra", "audit", "audit-root", "audit-extra")
#: How `rows-circuit-swapped` breaks the class circuit's hold while every tie, wiring and state holds: the aimed
#: phase's tied audit registered under a Program of one more unit, its answer and population that Program's
#: (`audit`), or the parameters' `circuit` another digest: of no Program, the scan's, or an untied audit's (`params`)
UNHOLDS = ("audit", "params")
#: How `rows-port` breaks the scan's registration's hold of the aimed link-window's rows while every other check
#: holds: the egress rows' root and domain at `scan/view` and their port at another root (`view`), the two ends' names
#: swapped (`swapped`), one end's at another window's (`window`), of other than T leaves (`leaves`, in `roots`), beside
#: a second entry of one end's name at another root (`second`), or beside a second tree under one end's domain, at
#: `scan/view` (`shadow`)
UNPORTS = ("view", "swapped", "window", "leaves", "second", "shadow")
#: How `rows-state-misindexed` puts its link-window's scan's state reads off the chain while its state stays tied by
#: root and domain: unit t's `sin` at leaf t + 1 (a leaf no unit registered) or its `o1` at leaf t (over the state
#: before it), a run over other units, the two ports' runs swapped, the reads or a port's run missing or under
#: another name, or a field no natural
MISREADS = ("sin-leaf", "o1-leaf", "units", "swapped", "missing", "untyped")
#: How `rows-state-unpinned` leaves its link-window's scan's row 0 unopened while its reads and its tie hold: a state
#: tree whose row 0 is a state the prover chose, opened as committed; the opening missing or under another name, its
#: salt or its path another, its words another count, leaf 1's opening in its place, or a field of another type
UNPINS = ("row", "missing", "salt", "path", "words", "leaf", "untyped")
#: How `rows-state-misanchored` puts its aimed link-window's tied audit's read of the scan's state off its slots'
#: anchors while every tie, wiring, state and hold stays (`_misanchor`): each slot reading the row after its anchor's
#: (`late`), lane 1's slots reading it (`lane`), the lanes' and the boundaries' digits swapped (`swapped`), a read of
#: the slot entries' output (`output`), the state a parameter of T + 1 rows (`params`), or a table over it (`table`)
MISANCHORS = ("late", "lane", "swapped", "output", "params", "table")


def _state_tree(rp: random.Random, T: int, empty: bool = True) -> ADV.Committed:
    """A scan's state tree of T rows of one to three u16 words under drawn salts and binding: row 0 the empty state, or
    a state the prover chose (``empty`` False, its first byte nonzero), the others any."""
    w = rp.randint(1, 3)
    first = bytes(2 * w) if empty else bytes([rp.randrange(1, 256)]) + rp.randbytes(2 * w - 1)
    rows = [first, *(rp.randbytes(2 * w) for _ in range(T - 1))]
    return ADV.commit(rows, 16 * w, rp.randbytes(64), salts=[rp.randbytes(ADV.INST.salt_bytes) for _ in rows])


def _flip_hex(rp: random.Random, h: str) -> str:
    """`h` with one of its bytes changed."""
    b = bytearray(bytes.fromhex(h))
    b[rp.randrange(len(b))] ^= 1 << rp.randrange(8)
    return b.hex()


def _misread(rp: random.Random, T: int) -> dict | None:
    """The scan's `window`'s ``reads`` off the chain, as one of `MISREADS` says; None for none."""
    reads = state_reads(T)
    s, how = reads[STATE], rp.choice(MISREADS)
    port = rp.choice(["sin", "o1"])
    if how == "sin-leaf":
        s["sin"]["leaf"] = 2
    elif how == "o1-leaf":
        s["o1"]["leaf"] = 0
    elif how == "units":
        lo, hi = s[port]["units"]
        s[port]["units"] = rp.choice([[lo, hi + 1], [lo + 1, hi + 1], [hi, lo] if lo != hi else [lo + 1, hi]])
    elif how == "swapped":
        s["sin"], s["o1"] = s["o1"], s["sin"]
    elif how == "missing":
        return rp.choice([None, {}, {"scan/state2": s}, {STATE: {k: v for k, v in s.items() if k != port}},
                          {STATE: {k if k != port else port.upper(): v for k, v in s.items()}}])
    else:
        s[port] = rp.choice([s[port] | {"leaf": rp.choice(["1", True, -1, None])},
                             s[port] | {"units": rp.choice([s[port]["units"][:1], s[port]["units"] + [0],
                                                            {"lo": s[port]["units"][0]}, str(s[port]["units"])])}])
    return reads


def _unpinned(rp: random.Random, tree: ADV.Committed, how: str) -> dict | None:
    """The scan's `window`'s ``empty`` of `tree` unopened, as `how`, one of `UNPINS`, says; None for none. A tree whose
    row 0 the prover chose (``row``) is opened as committed."""
    e = state_empty(tree)
    o, T = e[STATE], len(tree.rows)
    if how == "missing":
        return rp.choice([None, {}, {"scan/state2": o}])
    if how == "salt":
        o["salt"] = _flip_hex(rp, o["salt"])
    elif how == "path":
        o["path"] = rp.choice([_flip_hex(rp, o["path"]), o["path"][:-128], o["path"] + rp.randbytes(64).hex()]) \
            if T > 1 else rp.randbytes(64).hex()
    elif how == "words":
        o["words"] = rp.choice([o["words"] + 1, 0] + ([o["words"] - 1] if o["words"] > 1 else []))
    elif how == "leaf" and T > 1:
        x = tree.open(1)
        o |= {"path": b"".join(x.path).hex(), "salt": x.salt.hex()}
    elif how in ("leaf", "untyped"):
        o |= rp.choice([{"salt": o["salt"].upper()} if any(c in "abcdef" for c in o["salt"]) else {"salt": 7},
                        {"words": rp.choice([str(o["words"]), True, -1, None])}, {"path": None}])
    return e


def _present(e: dict) -> dict:
    return {k: v for k, v in e.items() if v is not None}


def _rows_entries(rp: random.Random, named: list[dict], where: str, unport: str | None) -> list[dict]:
    """The scan's registration's entries for the row registrations `named`, each at its own end's and window's port
    (`rows_name`, of its rows' leaves) in `roots` and at its value in `registered`, broken as `unport` says (the
    aimed link-window's `named` are its ingress and egress rows)."""
    def name(x: dict, d: str, w: int) -> dict:
        if where == "roots":
            return {"port": rows_name(Direction(d), w), "leaves": x["rows"]}
        return {"value": f"{x['link']}/{d}/{w}"}

    out = [{"domain": x["domain"], "root": x["root"], **name(x, x["direction"], x["window"])} for x in named]
    if unport is None:
        return out
    j = rp.randrange(2)
    x = named[j]
    if unport == "view":
        x = named[1]
        view = {"port": "scan/view", "leaves": x["rows"]} if where == "roots" else \
            {"value": "verity/certifier/scan-view/v1"}
        out[1] = out[1] | {"root": rp.randbytes(64).hex()}
        out.append({"domain": x["domain"], "root": x["root"], **view})
    elif unport == "swapped":
        out[0], out[1] = (out[0] | name(named[0], named[1]["direction"], named[0]["window"]),
                          out[1] | name(named[1], named[0]["direction"], named[1]["window"]))
    elif unport == "window":
        out[j] = out[j] | name(x, x["direction"], x["window"] + rp.choice([1, -1] if x["window"] else [1]))
    elif unport == "leaves":
        out[j] = out[j] | {"leaves": rp.choice([x["rows"] + 1, x["rows"] - 1, None, str(x["rows"]), True])}
    elif unport == "second":
        out.insert(rp.randrange(len(out) + 1), out[j] | {"root": rp.randbytes(64).hex()})
    else:
        view = {"port": "scan/view", "leaves": x["rows"]} if where == "roots" else \
            {"value": "verity/certifier/scan-view/v1"}
        out.insert(rp.randrange(len(out) + 1), {"domain": x["domain"], "root": rp.randbytes(64).hex(), **view})
    return [_present(e) for e in out]


def _prog_entries(rp: random.Random, tie: dict, where: str, untie: str | None, unwire: bool = False,
                  hit: int = 0) -> list[dict]:
    """The scan's registration's entries for its `prog`, lane j's `scan/prog<j>` (`prog_name`, named by port in
    `roots` and by value in `registered`) at its tie's chunk j's root and domain, each broken as `untie` says, or
    lane `hit`'s named another lane's (`unwire`)."""
    key = "port" if where == "roots" else "value"
    if untie == "missing":
        return []
    out = []
    for j, c in enumerate(tie["chunks"]):
        e = {"domain": c["domain"], key: prog_name(j), "root": c["root"]}
        if untie in ("root", "domain"):
            e[untie] = rp.randbytes(64).hex()
        elif untie == "renamed":
            other = "value" if key == "port" else "port"
            e = rp.choice([{other if k == key else k: v for k, v in e.items()},
                           e | {key: rp.choice([PROG, f"{PROG}0{j}", f"{PROG}-{j}", f"{PROG}\u0663", f"Scan/prog{j}",
                                                f"prog{j}", f"{PROG}{j}/", STATE])}])
        elif unwire and j == hit:
            e[key] = prog_name(rp.choice([x for x in range(len(tie["chunks"]) + 1) if x != j]))
        out.append(e)
        if untie == "scan":
            out.append({"domain": c["domain"], key: c["port"], "root": c["root"]})
    return [_present(x) for x in out]


def _state_entries(rp: random.Random, states: dict, i: int, where: str, unstate: str | None) -> list[dict]:
    """The scan's registration's entries for its state (`STATE` by port in `roots`, `SCAN_STATE` by value in
    `registered`) at the `SCAN_STATE` of audit `i`, the one its `prog` is tied to, broken as `unstate` says."""
    key, name = ("port", STATE) if where == "roots" else ("value", SCAN_STATE)
    e = {"domain": states[i]["domain"], key: name, "root": states[i]["root"]}
    if unstate == "missing":
        return []
    if unstate in ("root", "domain"):
        e[unstate] = rp.randbytes(64).hex()
    elif unstate == "untyped":
        e["root"] = rp.choice([None, 7, False])
    elif unstate == "renamed":
        other = "value" if key == "port" else "port"
        e = rp.choice([{other if k == key else k: v for k, v in e.items()},
                       e | {key: rp.choice(["scan/state2", "Scan/state", "state", "scan/view", "scan/state/",
                                            "verity/certifier/scan-state/v2"])}])
    elif unstate == "other":
        e |= states[rp.choice([j for j in states if j != i])]
    extra = [e | {"root": rp.randbytes(64).hex()}] if unstate == "extra" else []
    return [_present(x) for x in [e, *extra]]


def _output_entries(rp: random.Random, i: int, ties: dict, nb: int, aimed: int, untie: str | None,
                    unwire: str | None = None, hit: int = 0) -> list[dict]:
    """Audit `i`'s registration's `roots`: for each link-window's `prog` tied to it, each lane's chunk's entries port
    its tie names at the chunk's root and domain (`nb` leaves, a timestep's row each), and an input port's entry at
    another root; at link-window `aimed`, every chunk's broken as `untie` says, or lane `hit`'s as `unwire` says."""
    out = []
    for w, x in ties.items():
        if x["audit"] != i or (w == aimed and untie == "scan"):
            continue
        for j, c in enumerate(x["chunks"]):
            port, leaves, broken = c["port"], nb, w == aimed and j == hit
            if w == aimed and untie == "port":
                port = rp.choice([f"block-{w}/o0", f"block-{w}/prog", f"block-{w}/o01", "o1", f"block-{w}/O1",
                                  f"block-{w}/o", f"block-{w}/o1x", f"block-{w}/o\u0663", f"block-{w}/o\u00b9", 1,
                                  port.replace(f"/c{j}/", f"/c0{j}/"), port.replace(f"/c{j}/", "/c/"),
                                  port.replace(f"/c{j}/", "/"), port[port.index("{"):]])
            elif broken and unwire == "output":
                port = rp.choice([f"block-{w}/o{rp.choice([1, 2, 10])}", port.removesuffix("o0") + "o1"])
            elif broken and unwire == "key":
                port = port.replace("NP=4", f"NP={rp.choice([5, 1024])}")
            elif broken and unwire == "chunk":
                port = entries_port(x["key"], rp.choice([y for y in range(len(x["chunks"]) + 1) if y != j]))
            elif broken and unwire == "leaves":
                leaves = rp.choice([nb - 1, nb + 1, 2 * nb, -1, None, "1", True])
            out.append(_present({"domain": c["domain"], "leaves": leaves, "port": port, "root": c["root"]}))
    out.append({"domain": rp.randbytes(64).hex(), "leaves": nb, "port": f"block-{rp.randrange(4)}/prog",
                "root": rp.randbytes(64).hex()})
    rp.shuffle(out)
    return out


def _by_template(r: random.Random, desc: dict, m: str) -> tuple[dict, list[dict]]:
    """A tied audit's stratified law and its profile's strata, a stratum a template of its Program (`templates_of`):
    its slot entries' and scan slots' templates drawn whole and its first other template in part (`tied-partial`), or
    with its slot entries' (`tied-entries-sampled`) or its scan slots' (`tied-slots-sampled`) template in part too."""
    ts = templates_of(desc)
    whole, broken = (*ENTRY_IDS, SCAN_SLOTS), {"tied-entries-sampled": ENTRY_IDS, "tied-slots-sampled": (SCAN_SLOTS,)}
    fewer = lambda n: r.randint(min(1, n - 1), n - 1)  # noqa: E731
    ks = [fewer(n) if d in broken.get(m, ()) else n if d in whole else r.randint(1, n) for _, d, n in ts]
    j = next(j for j, (_, d, _) in enumerate(ts) if d not in whole)
    ks[j] = fewer(ts[j][2])
    lo, law, strata = 0, [], []
    for (key, _, n), k in zip(ts, ks):
        law.append({"k": k, "template": key, "units": [[lo, lo + n]]})
        strata.append({"k": k, "name": key, "size": n})
        lo += n
    return {"k": sum(ks), "law": "stratified", "strata": law}, strata


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
    # every form binds the link in its window record's canonical JSON, whose strings are ASCII without DEL
    link = r.choice(["eth0", "a", "d-e"])
    program = hashlib.sha512(b"link program %d" % r.randrange(3)).hexdigest()
    least_windows = 2 if m in ("undecided-and-reject", "clock-reused", "delta") else 1
    least_audits = 2 if m == "state-untied" else \
        1 if m.startswith("audit") or m in ("undecided-and-reject", "count-audits") else 0
    n_windows = r.randint(least_windows, least_windows if small else 3)
    n_audits = r.randint(least_audits, max(least_audits, 1) if small else 2)
    # a rows link-window's own audit, and at times one no link-window ties, which may leave units undrawn
    n_audits = max(n_audits, n_windows + (0 if small else rp.randint(0, 1))) if rows else n_audits
    # `state-unheld`: one audit beside the link-windows' own, which its second tie may take
    n_audits = max(n_audits, n_windows + 1) if m == "state-unheld" else n_audits
    first = r.choice([0, 0, r.randrange(1, 6)])
    anchors = m in ("state-anchored", "state-misanchored", "state-unheld")
    misanchor = rp.choice(MISANCHORS) if m == "state-misanchored" else None
    if rows:
        sched = bound = misanchor_schedule(misanchor, _small_schedule(r, small))
        widths = (r.randint(1, 3), r.randint(1, 4), r.randint(1, 3), r.randint(1, 2))
        scan = json.loads(scan_descriptor(sched.window, WR.row_bytes(sched), widths, lanes=sched.lanes,
                                          every=sched.every))
        program = program_sha512(scan)
        delta = r.choice(DELTAS)
    else:
        deltas = r.sample(range(4), r.randint(1, 3))
        rhos = r.sample([Fraction(1), Fraction(2), Fraction(1, 2), Fraction(3, 2)], r.randint(1, 3))
    t = r.randrange(1, n_windows) if m == "clock-reused" else r.randrange(n_windows)
    a = r.randrange(max(n_audits, 1))
    # one slot entries' template per audit: one-stage names a template's ports by its descriptor id; an audit that
    # selects has select entries and scan slots, as a window circuit's last position reads its slots' occupancy
    owners = rp.sample(range(n_audits), n_windows) if rows else []
    # the tied audits run one class circuit, so they all select, or none do; in the record form none do
    select = rows and (rp.random() < 0.5 or m == "tied-slots-sampled") and not anchors
    selects = set(owners) if select else set()
    lanes = sched.lanes if rows else 0
    ties = {w: {"audit": i, "key": (k := SELECT_KEY if i in selects else SLOT_KEY), "chunks": [
                {"port": entries_port(k, j), "root": rp.randbytes(64).hex(), "domain": rp.randbytes(64).hex()}
                for j in range(lanes)]}
            for w, i in zip(range(first, first + n_windows), owners)}
    untie = rp.choice(UNTIES) if m == "prog-untied" else None
    unwire = rp.choice(UNWIRES) if m == "prog-unwired" else None
    unstate = rp.choice(UNSTATES) if m == "state-untied" else None
    unhold = rp.choice(UNHOLDS) if m == "circuit-swapped" else None
    unport = rp.choice(UNPORTS) if m == "port" else None
    unpin = rp.choice(UNPINS) if m == "state-unpinned" else None
    # each audit's `SCAN_STATE`, the state tree of the link-window tied to it, whose leaf 0 the scan opens
    trees = {i: _state_tree(rp, sched.window, not (unpin == "row" and i == ties[first + t]["audit"]))
             for i in range(n_audits)} if rows else {}
    states = {i: {"root": x.root.hex(), "domain": ADV.domain(x.binding, len(x.rows)).domain_id.hex()}
              for i, x in trees.items()}
    if untie == "untyped":
        for c in ties[first + t]["chunks"]:
            c["root"] = rp.choice([None, 7, False])
    # the lane whose wiring the aimed link-window's `unwire` breaks
    hit = rp.randrange(lanes) if lanes else 0
    # the changes that draw the aimed link-window's audit stratified by its templates, its slot entries' and scan slots'
    # whole (`tied-partial`) or one of them in part
    by_template = m in ("tied-partial", "tied-entries-sampled", "tied-slots-sampled")
    if m == "tied-sampled" or by_template:
        a = ties[first + t]["audit"]
    # `state-unheld`: a second audit tied to the aimed link-window, which runs the class circuit and reads the scan's
    # state at its slots' anchors from its own `SCAN_STATE`, another link-window's scan's state or no link-window's
    extra = rp.choice([i for i in range(n_audits) if i != ties[first + t]["audit"]]) if m == "state-unheld" else None
    tied = {x["audit"] for x in ties.values()} | ({extra} if extra is not None else set())
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
        decoy_at = r.randrange(len(named) + 1)
        decoy = {"domain": r.randbytes(64).hex(), "root": r.randbytes(64).hex(), "value": "decoy"}
        T = sched.window
        units = 2 * T + (1 if aimed and m == "units" else 0)
        strata = [{"k": r.randint(1, T), "name": d.value, "size": T} for d in (INGRESS, EGRESS)] \
            if r.random() < 0.5 and units == 2 * T else None
        k = sum(s["k"] for s in strata) if strata else r.randint(1, units)
        law = {"k": k, "law": "subset"} if strata is None else {"k": k, "law": "stratified", "strata": [
            {"k": s["k"], "template": s["name"], "units": [[j * T, (j + 1) * T]]} for j, s in enumerate(strata)]}
        fmt, where = r.choice(FORMATS), r.choice(["roots", "registered"])
        if aimed and unport == "leaves":
            where = "roots"
        entries = _rows_entries(rp, named, where, unport if aimed else None)
        entries.insert(decoy_at, decoy)
        entries += _prog_entries(rp, ties[w], where, untie if aimed else None, unwire == "lane" and aimed, hit)
        entries += _state_entries(rp, states, ties[w]["audit"], where, unstate if aimed else None)
        rp.shuffle(entries)
        tree = trees[ties[w]["audit"]]
        reads = _misread(rp, T) if aimed and m == "state-misindexed" else state_reads(T)
        empty = _unpinned(rp, tree, unpin) if aimed and unpin else state_empty(tree)
        window = {"kind": "stand-in", "window": w} | ({} if reads is None else {"reads": reads}) | (
            {} if empty is None else {"empty": empty})
        rec = {"format": fmt, "law": law, "partition": f"{link}/scan/{w}", "population": units, "program": program,
               "protocol": ONE_STAGE, where: entries, "window": window}
        if aimed and m == "program":
            rec["program"] = r.choice([hashlib.sha512(b"another program").hexdigest(), program.upper(), None])
        digest = bytes.fromhex(R.digest(rec))
        answer = _accepted(digest, program, rec["partition"], units, k,
                           r.choice([d for d in DELTAS if d != delta]) if aimed and m == "delta" else delta,
                           r.randint(0, units), strata)
        if aimed:
            if m == "answer-protocol":
                answer["protocol"] = "verity/sampled-proofs/two-stage/v1"
            elif m == "answer-other":
                answer["commitment"] = r.choice([r.randbytes(64).hex(), R.digest(rec | {"population": 1})])
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
        digest = bytes.fromhex(R.digest(rec))
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
                answer["commitment"] = r.choice([r.randbytes(64).hex(), R.digest(rec | {"population": 2})])
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

    audit_params, audit_regs, audits, untied = [], [], [], []
    slots = sched.window // sched.every * sched.lanes if rows else 0
    # the class circuit, the one Program every tied audit runs: its units, outputs, slot entries and scan slots, drawn
    # once; `members` and `body` change it, so every tied audit's wiring breaks alike
    off = (rp.choice([1, -1]) if slots > 1 else 1) if unwire == "members" else 0
    n_class = r.randint(2 if m == "tied-sampled" or by_template else 1, 40) if rows else 0
    class_split = r.randint(1, n_class - 1) if n_class > 1 and r.random() < 0.3 else None
    class_outs = r.sample(UNIT_BYTES, 1 if class_split is None else 2) if rows else []
    class_slots = (rp.randint(1, 2),) if select else ()

    def shape_of(n: int, split: int | None, outs: list[int]) -> tuple[tuple[int, int], ...]:
        return ((n, outs[0]),) if split is None or split >= n else ((split, outs[0]), (n - split, outs[1]))

    # `state-anchored`, `state-misanchored` and `state-unheld`: the class circuit's record, which every tied audit carries
    anchor = anchor_audit(n_class, lanes, sched.every, sched.window, rp.random() < 0.5) if anchors else None
    circuit = program_sha512(json.loads(anchor[0] if anchors else audit_descriptor(
        shape_of(n_class, class_split, class_outs), (slots + off,), unwire == "body", select, class_slots))) \
        if rows else None
    for i in range(n_audits):
        sampled = i == a and m == "tied-sampled"
        partial = i == a and by_template
        in_class = i in tied
        entries = tuple(slots + off for x in ties.values() if x["audit"] == i) or ((slots + off,) if i == extra else ())
        renamed = rows and unwire == "body" and in_class
        scan_slots = class_slots if in_class else ()
        swapped = unhold == "audit" and i == ties[first + t]["audit"]
        units = (n_class + 1 if swapped else n_class) if in_class else r.randint(2 if sampled or partial else 1, 40)
        n_prog = units + (r.choice([1, -1]) if units > 1 else 1) if i == a and m == "audit-units" else units
        units += sum(entries) + sum(scan_slots)
        if in_class:
            split, outs = class_split, class_outs
        else:
            split = r.randint(1, n_prog - 1) if n_prog > 1 and r.random() < 0.3 else None
            outs = r.sample(UNIT_BYTES, 1 if split is None else 2)
        desc = json.loads(audit_descriptor(shape_of(n_prog, split, outs), entries, renamed, i in selects, scan_slots))
        listed = i == a and m == "audit-listed"
        recorded, bits = i == a and m.startswith("audit-record") or listed, m == "audit-record-bits"
        if recorded:
            desc, units = json.loads(record_audit(bits)[0]), RECORD_UNITS[bits]
        elif anchors and in_class:
            desc = json.loads(anchor[0])
        prog = program_sha512(desc)
        if not in_class:
            untied.append(prog)
        # `tied-sampled`: one or two of the tied audit's proof blocks, the demo's sample
        k, strata = r.randint(1, min(2, units - 1)) if sampled else units if i in tied else r.randint(1, units), None
        law = {"k": k, "law": "subset"}
        wrong = (0 if r.random() < 0.25 else r.randint(1, units)) if sampled else r.randint(0, units)
        if partial:
            law, strata = _by_template(r, desc, m)
            k, wrong = sum(s["k"] for s in strata), 0 if r.random() < 0.25 else r.randint(1, units)
        elif r.random() < 0.5 and not sampled:
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
            rec["roots"] = _output_entries(rp, i, ties, sched.window // sched.every, first + t, untie, unwire, hit)
            if i == extra:
                rec["roots"] += [{"domain": c["domain"], "leaves": sched.window // sched.every, "port": c["port"],
                                  "root": c["root"]} for c in ties[first + t]["chunks"]]
            stated = i == ties[first + t]["audit"]
            if not (stated and unstate == "audit"):
                st = states[i] | ({"root": rp.randbytes(64).hex()} if stated and unstate == "audit-root" else {})
                there = rp.choice(["roots", "registered"])
                got = [st | ({"port": STATE} if there == "roots" else {"value": SCAN_STATE}) | {"leaves": sched.window}]
                if stated and unstate == "audit-extra":
                    got.append(got[0] | {"root": rp.randbytes(64).hex()})
                rec[there] = rec.get(there, []) + got
                rp.shuffle(rec[there])
        record = {}
        if recorded:
            _, doc, order = record_audit(bits)
            pins = {"w": rp.randbytes(64).hex()} if r.random() < 0.5 or m == "audit-record-pin" else {}
            if m == "audit-record-state":
                pins[SCAN_STATE] = rp.randbytes(64).hex()
            roots = {p: v.upper() if r.random() < 0.5 else v for p, v in pins.items()}
            if m == "audit-record-pin":
                roots = r.choice([{}, {"w": rp.randbytes(64).hex()}, {"v": pins["w"]}])
            held = hashlib.sha512(IF.circuit_bytes(json.loads(doc))).hexdigest()
            if m == "audit-record-circuit":
                held = r.choice([hashlib.sha512(b"another circuit record").hexdigest(), held.upper(), None])
            rec["window"] |= {"circuit": held} | ({"registered_roots": roots} if roots else {})
            record = {"circuit": json.loads(doc), "order": json.loads(order), "pins": pins}
            if listed:
                part = record_partition(record["circuit"], "")
                record["circuits"] = [
                    {"sha512": hashlib.sha512(circuit_file(desc, t, part).encode()).hexdigest(), "template": t}
                    for t, _, _ in charges_of(desc)]
        elif anchors and in_class:
            _, doc, order = (json.loads(x) for x in anchor)
            if misanchor and i == ties[first + t]["audit"]:
                doc["registered"] = _misanchor(misanchor, doc["registered"], sched)
            rec["window"] |= {"circuit": hashlib.sha512(IF.circuit_bytes(doc)).hexdigest()}
            record = {"circuit": doc, "order": order, "pins": {}}
        if i == a and m == "audit-program":
            if r.random() < 0.5:
                rec["program"] = r.choice([hashlib.sha512(b"another program").hexdigest(), prog.upper(), None])
            else:
                desc = json.loads(audit_descriptor(((n_prog, r.choice([o for o in UNIT_BYTES if o != outs[0]])),),
                                                   entries, renamed, i in selects, scan_slots))
        digest = bytes.fromhex(R.digest(rec))
        answer = _accepted(digest, prog, rec["partition"], units, k, r.choice(["1/100", "1/1000000", "1/10"]),
                           wrong, strata)
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
        audit_params.append({"program": desc, "steps": r.randint(0, 4)} | record)
        audit_regs.append(rec)
        audits.append(answer)

    case_params = {"audits": audit_params, "cells": [r.randint(0, 100) for _ in range(r.randint(0, 3))], "first": first,
                   "jitter": r.randint(1, 5), "key": lms.public_key(key).hex(), "link": link,
                   "lms": {"H": H, "N": N, "W": W, "toy": toy}, "program": program, "session": session.hex(),
                   "windows": n_windows}
    if rows:
        case_params |= {"cells": bound.caps(n_windows), "circuit": circuit, "scan": scan, "schedule": schedule_json(bound)}
        if unhold == "params":
            case_params["circuit"] = r.choice([hashlib.sha512(b"another class circuit").hexdigest(), program, *untied])
    else:
        case_params["sync_set"] = {"deltas": deltas, "rhos": [[x.numerator, x.denominator] for x in rhos]}
    c_json = {"audits": audit_regs, "windows": window_regs}
    e_json = {"audits": audits, "windows": windows}
    if m == "first":
        case_params["first"] = first + r.choice([1, -1] if first else [1])
    elif m == "count-windows":
        case_params["windows"] = n_windows + r.choice([1, -1] if n_windows > 1 else [1])
        if rows:
            case_params["cells"] = bound.caps(case_params["windows"])
    elif m == "count-audits":
        case_params["audits"] = r.choice([audit_params[1:], [*audit_params, {
            "program": json.loads(audit_descriptor(((1, 2),))), "steps": 0}]])
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
    return out + [*bad_caps(p), p | {"schedule": {k: v for k, v in s.items() if k != SCHEDULE_KEYS[n % len(s)]}},
                  p | {"schedule": s | {"x": 1}}, p | {"schedule": [s]}, p | {"schedule": s | {"delay": 0}},
                  p | {"schedule": s | {"lanes": True}}, p | {"schedule": s | {"window": -1}},
                  p | {"schedule": s | {"queue": "1"}}, p | {"schedule": s | {"cell_bytes": 10}},
                  p | {"schedule": s | {"cell_bytes": U32}}, p | {"schedule": s | {"lanes": U32}},
                  p | {"schedule": s | {"window": U32}}, p | {"schedule": s | {"every": s["window"] + 1}},
                  p | {"schedule": s | {"every": 0}}, p | {"sync_set": {"deltas": [0], "rhos": [[1, 1]]}},
                  p | {"circuit": p["circuit"][:-2]}, p | {"circuit": p["circuit"].upper()}, p | {"circuit": 0},
                  p | {"circuit": None}, *bad_scans(p)]


def bad_caps(p: dict) -> list[dict]:
    """Rows-form parameters whose caps aren't the schedule's (`Schedule.caps`), each one change to `p`: none, one
    connection short, one more, and one cap below `egress_cells`."""
    caps = p["cells"]
    return [p | {"cells": []}, p | {"cells": caps[1:]}, p | {"cells": [*caps, caps[0]]},
            p | {"cells": [caps[0] - 1, *caps[1:]]}]


def bad_audits(p: dict) -> list[dict]:
    """Parameters whose audits `auditOf` refuses, each one change to `p`: an audit without its monitoring period, one
    stating its free inputs (the check counts them off its Program), a negative monitoring period, not a list, an
    audit whose Program isn't a descriptor, is of another schema, or has no template in its root (the stand-in unit
    alone), and in the record form (`record_audit`) one without its order and pins, or whose pins aren't
    `{param: root}`."""
    good = json.loads(audit_descriptor(((1, 2),)))
    bare = json.loads(D.canonical(codec.encode_program(Program(bind(AUDIT_UNIT, OUT=2)))))
    one = lambda **kw: p | {"audits": [{"program": good, "steps": 0} | kw]}  # noqa: E731
    prog, doc, order = (json.loads(x) for x in record_audit())
    held = {"circuit": doc, "order": order, "program": prog}
    return [p | {"audits": [{"program": good}]}, one(free=0), one(steps=-1), p | {"audits": "x"}, one(program="x"),
            one(program=good | {"schema": "verity-ir/v0"}), one(program=bare), one(circuit=doc, program=prog),
            one(**held, pins=[]), one(**held, pins={"w": 1})]


def _scan_widths(desc: dict) -> tuple[int, int, int, int]:
    """`scan_descriptor`'s widths of a generator-form descriptor it wrote: the ingress templates' state, prog and view
    bits, and the egress template's state words."""
    sig = {tuple(k for k, _ in d["body"]["params"]): d["body"] for d in desc["definitions"].values()
           if d["form"] == "composite"}
    first = next(b for k, b in sig.items() if k[:2] == ("row", "prog0"))
    egress = sig[EGRESS_PARAMS]
    _, state, view = first["ret_type"][1]
    return state[1], first["params"][1][1][1], view[1], egress["params"][0][1][1]


def bad_scans(p: dict) -> list[dict]:
    """Rows-form parameters whose scan `scanOf` refuses, each one change to `p`: not a descriptor, another Program's
    (its SHA-512 not `program`), and, each under its own SHA-512, the checker form (its egress row an input), templates
    with other parameters, another number of instances, rows not the schedule's, a lane's `prog<j>` another row of its
    input than its bucket's or another lane's input, an input gate no template's, the egress template keyed as an
    input, and each negative of a check of what
    `NetTiming.BoundaryGates.Fits` asks (`scan_negatives`)."""
    s, desc = p["schedule"], p["scan"]
    T, row, widths = s["window"], s["lanes"] * s["slots"] * s["cell_bytes"], _scan_widths(desc)
    L, every = s["lanes"], s["every"]
    first, ingress = first_params(L), ingress_params(L)

    def own(text: str) -> dict:
        d = json.loads(text)
        return p | {"scan": d, "program": program_sha512(d)}

    def scan(*a, **k) -> str:
        return scan_descriptor(*a, **({"lanes": L, "every": every} | k))

    keyed = json.loads(D.canonical(desc))
    fn = [x["fn"] for x in keyed["definitions"][keyed["root"]]["body"]["nodes"]
          if keyed["definitions"][keyed["callees"][x["fn"]]]["form"] != "primitive"][-1]
    old = keyed["callees"][fn]
    keyed["callees"][fn] = "Input3_v1"
    keyed["definitions"]["Input3_v1"] = keyed["definitions"].pop(old) | {"id": "Input3_v1", "bindings": {}}
    extra = json.loads(D.canonical(desc))
    extra["types"].append(["Array", 1, ["Value", 1]])
    extra["definitions"][extra["root"]]["body"]["nodes"].append(
        {"args": [], "axes": [], "fn": extra["callees"].index("Input1_v1"), "form": "batch", "n": 1,
         "out": len(extra["types"]) - 1})
    return [p | {"scan": "x"}, p | {"scan": {"free": 0, "steps": 0}}, p | {"scan": desc | {"schema": "verity-ir/v0"}},
            p | {"scan": json.loads(scan(T, row, (widths[0] + 1, *widths[1:])))},
            own(scan(T, row, widths, checker=True)),
            own(scan(T, row, widths, names=(("row", "seed"), ingress, EGRESS_PARAMS))),
            *([own(scan(T, row, widths, names=(first, ("row", "sin", "seed"), EGRESS_PARAMS)))] if T > 1 else []),
            own(scan(T, row, widths, names=(first, ingress, ("words",)))),
            own(scan(T, row, widths, lanes=L + 1)),
            own(scan(T, row, widths, counts=(1, T - 1, T + 1))),
            own(scan(T, row, widths, counts=(2, T - 1, T))),
            own(scan(T, row, widths, counts=(1, T, T))),
            own(scan(T, row + 1, widths)),
            *([own(scan(T, row, widths, skew=1))] if T // every > 1 else []),
            *([own(scan(T, row, widths, shared=True))] if L > 1 else []),
            p | {"scan": extra, "program": program_sha512(extra)},
            p | {"scan": keyed, "program": program_sha512(keyed)},
            *(q for q, _ in scan_negatives(p).values())]


#: A primitive that is no input, a constant gate the `row` negative binds bucket 0's row to.
ZERO = "NetworkPeriodScanZero_v1"
#: Members of each of the `walk` negative's nested batches: three deep, past `Flock.Steps.FUEL`'s walk.
DEEP = 2 ** 30


def scan_negatives(p: dict) -> dict[str, tuple[dict, str]]:
    """Rows-form parameters each refused by one of `scanOf`'s checks of what `NetTiming.BoundaryGates.Fits` asks and by none
    before it, each one change to `p`'s generator-form scan under its own SHA-512, with a fragment of both refusals:
    `walk`, the egress template calling a composite of `2^90` gates in three nested batches, unread, so the walk to an
    operand is past `FUEL`; `returns`, the root returning an input leaf for the last egress leaf; `computes`, the egress
    template returning its `state`'s first 16 leaves as its u16; `row`, bucket 0's row bound to a constant primitive's
    gates (an input node like it appended, keeping the input count); `argument`, egress member 0's first state leaf the
    last ingress template's first output leaf (a later step's ingress when `T > 1`); `ingress`, the egress state the
    last ingress template's row input node (resized, the state's node resized to keep the count); `topological`, the
    last ingress template's row input node moved after the templates, so the template names a later node; and `apart`,
    the egress instances bucket 0's template's, so one stratum holds bucket 0's VU and the egress buckets' (`vuStrata`),
    refused before any other check."""
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
    out["argument"] = own(d), "a template's row, state or egress argument is not one input node's leaves"

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

    s = p["schedule"]
    out["apart"] = own(json.loads(scan_descriptor(
        s["window"], s["lanes"] * s["slots"] * s["cell_bytes"], _scan_widths(desc), lanes=s["lanes"], every=s["every"],
        egress=0))), "the egress template's VUs are not apart from the ingress templates'"
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
        ACCEPT: tuple(m for m in ms if EXPECT[m] == ACCEPT),
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


def kinds(form: str = "clock") -> tuple[str, ...]:
    """The kinds a case of the form can be aimed at (`AIMS`)."""
    return tuple(k for k in KINDS if AIMS[form][k])


def cases(seed: int = 0, form: str = "clock") -> dict[str, tuple[dict, dict]]:
    """One small period of the form of each kind it can be aimed at (`kinds`) with its answer,
    `{kind: (case, answer)}`."""
    return {k: (c, answer(c)) for k in kinds(form) for c in [case(k, seed, form)]}


def audits_of(case: dict) -> list[tuple[dict, dict, dict]]:
    """A period's compute audits, each its params' entry, its registration record and one-stage's answer for it."""
    c, e = (json.loads(bytes.fromhex(case[k])) for k in ("commitment", "evidence"))
    return list(zip(case["params"]["audits"], c["audits"], e["audits"], strict=True))


def with_audits(case: dict, audits: list[tuple[dict, dict, dict]]) -> dict:
    """`case` with the compute audits `audits` (as `audits_of` gives them) in its params, commitment and evidence. The
    check holds no audit to a record outside the period: beside a rows-form period's tied audit, an audit no
    execution phase's scan ties (a stray, another period's) is accepted, and a composite that ties every compute audit
    to its policy's record is what refuses it."""
    c, e = (json.loads(bytes.fromhex(case[k])) for k in ("commitment", "evidence"))
    return {"commitment": D.canonical({**c, "audits": [a[1] for a in audits]}).encode().hex(),
            "evidence": D.canonical({**e, "audits": [a[2] for a in audits]}).encode().hex(),
            "params": {**case["params"], "audits": [a[0] for a in audits]}}
