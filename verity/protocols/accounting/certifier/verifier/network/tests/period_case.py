"""The network certifier's period check (`NetworkCheck.Period`, `verity/certifier/v0`) in Python, for every difftest
that runs it: its mirror, `answer` (as `reference`), and its cases, `build` and `case`.

`answer(case)` is the outcome of one `serve` line in the core interface's answer format
(`verity_difftest.protocol.answer`): it decodes the protocol's formats (`NetworkCheck/Period.lean`'s module docstring)
and checks a period with the Python the period's pieces already have: one-stage's answers and integrity profiles as PoUW
reads them (`pouw.core_check.answer_of` and `integrity`), the network certifier's certificates, each under the listed
entry its execution phase picks (`certifier.certificate`, `_entry_at`: the entry's LMS key, its session, and that
session's own number for the execution phase, the period's less the entry's start; a signer that is another listed key,
outside its stretch, or that the parameters don't list is refused by name, `OUTSIDE`, `UNLISTED`), each listed entry's
session's signed start (`certifier.certificate`'s `CertifiedStart.refusal`, under the entry's key in its session;
`_start_check`: an entry whose start is not where its session's signed base puts it after its key's first listed entry
is refused by name, `NOT_SIGNED`, and one whose shown start its session did not certify, `UNCERTIFIED`), one-stage's
`registration.digest` (a registration record is ASCII-only, where its canonical JSON is Lean's), the schedule
(`network_certifier.NetworkCertifierParams`), the verifier's read of each row registration
(`network_certifier_rows.check_registration`) and the scan's ok bits' constants, computed once a period and compared an execution
phase (`scan_ok.constants` and `held`), and the scan's terms read off its Program's descriptor (`_scan`), as each compute
audit's units and their committed bits are read off its own (`_audit`). Its verdict,
public, profile and digest are Lean's exactly; its reason is its own. The period is in the rows form, the one form the
check reads: parameters naming a `sync_set` (the clock form's) are refused before anything else is read. Public is `K`
and its terms over `MonitoringPeriodLeakAccepted`'s windows (the execution phases' scans, all at one δ), each with its
`delta`, and `eps`, the union of those deltas as an exact ratio string, and the schedule's response cap
(`max_tokens_cap`) when it has one.

`build(m, seed, small)` makes a period with one change `m` (`EXPECT`, each aimed at one path of the check, or malformed
parameters, commitments, evidence and cases). Its parameters list the signer's entry (its key, its session and the
start of its stretch, at or before `first`, so the signer's numbers are the period's less that start), alone in a
small period and otherwise among up to two more whose stretches hold none of the period's execution phases; `second-key`
lists the signer second of two, and `two-stretches` two entries, the period's one execution phase the second's
stretch's first, signed by that entry at its number 0. Refused: two listed entries that sign in turn,
each in its own session at its own number, in a period one entry's stretch holds (`two-signers`); a certificate or a
rows registration under the other listed entry, outside its stretch (`cert-outside-stretch`, `reg-outside-stretch`),
under a key that isn't listed (`cert-unlisted`, `reg-unlisted`), under the picked key in the other entry's session
(`cert-other-session`, `reg-other-session`), or at the period's number where the session's belongs
(`cert-period-number`, `reg-period-number`); and parameters whose `"keys"` don't read (`bad_entries`): no entry, entries
out of order or with one start, one session twice, two keys under one key identifier, a period that starts before the
first entry, or the earlier forms `"key"`, `"session"` and `"sessions"`. A period's evidence shows each listed entry's
session's signed start, its base on its key's life sampled for a key's first listed entry and, for a later one, as many
of the schedule's execution phases on as its start is, and its ordinal among its key's sessions, sampled for a key's
first listed entry and the next for each later one. `two-sessions` lists the signer's key in two sessions, the period's
one execution phase in the second's stretch, and is accepted. Refused: that list with the second session's base
another (`start-not-signed`), or its ordinal not the first's next (`start-skipped`); a period of two execution phases,
one in each of the two sessions' stretches (`restart`), as a period is one execution phase (`startCarriedRefusal`);
a shown start that is not its session's
certified start, by each code of `CertifiedStart.refusal` (`start-uncertified`, `UNCERTIFIED_BY`); and evidence that
shows a start too few or too many, or none (`start-count`). The execution phases of one link from the execution phase
`first`, each execution phase's ingress and egress rows committed and registered under a small sampled schedule
(`network_certifier_rows.Cells.commit` and `registration`, signed by a small LMS key in a sampled network certifier session;
`capped`: a schedule with a response cap, and
`cap-schedule` the parameters or a registration under it with another cap or none), its three records certified
(`past-slots`: none of them and neither registration, as the network certifier leaves a window in which it refused a
connection), and
the scan's registration holding both rows' roots and domains and each lane j's `scan/prog<j>` (`prog_name`) at the root
and domain of chunk j's entries port of that execution phase's slot entries' template (`slot_entry`,
`io_shell.SlotEntry_v1` at its smallest circuit-check row, or `select_entry`, `SelectEntry_v1` at its smallest binding
beside a stand-in scan slots' `ScanSlots_v2`, each a pinned Definition with one member a slot) in its own audit's
registration and Program (`_prog_tied`, `_wired`), and its
state (`STATE`) at the root and domain of that audit's `SCAN_STATE` (`_state_tied`), a state tree of T rows
(`_state_tree`) whose units' reads its `window` states as the chain (`state_reads`, `_state_read`), its start (`START`,
the start unit's input) a registered row its `window` opens to the empty state (`start_empty`, `_start_empty`), and
each member's ok bits
(`<member>/o0`) at the schedule's constant (`_ok_entries`; `ok-unheld`: one execution phase's broken, `UNOKS`; `ok-equal`:
a schedule of two ticks, where two members' are at one domain and root; `one-tick`: a schedule of one tick, which
has no constants), answered by one-stage with a
stratified or subset sample at the period's δ, its public outputs naming its lanes' `prog<j>` tied (`TIED_KEY`,
`prog_params`) where several of its units read one slot entry (`every` ≥ 2), as one-stage's registration check
publishes them; a rows period is one execution phase, and one of more is refused (`start-carried`, `CARRIES`), while
one whose execution phase is not the run's first is accepted at any start, charged the state's full width
(`start-charged`, `CHARGES`); the scan's Program is its generator form, stand-ins of its three templates
at small widths in the Boolean profile's types, as the scan's Definitions have them (`scan_descriptor`), and a Program in
the checker form is a bad scan (`bad_scans`), as is each negative of a check of the rows the scan's ties equate, of
what `NetTiming.BoundaryGates.Fits` asks or of the input nodes its arguments read (`scan_negatives`). Beside them, the
compute audits (at least one, whose registrations' `roots` hold the
scans' `prog`), each registered over its Program, a population of one or two stand-in templates whose units commit from
8 to 320 bits, with its slot entries' (`audit_descriptor`), and answered with a subset or stratified
sample (a work sampling design's) and its bound on incorrect proof blocks, every unit sampled for an audit a scan's
`prog` is tied to, or, each accepted and charged, its slot entries' and scan slots' templates whole and the rest in part
(`tied-partial`, a stratum a template, `_by_template`), one or two of its proof blocks sampled (`tied-sampled`,
`subset:k`, the demo's sample), or its slot entries' or scan slots' stratum in part too (`tied-entries-sampled`,
`tied-slots-sampled`), and an execution phase whose scan's state is not its
wired audit's `SCAN_STATE` refused (`state-untied`: a `SCAN_STATE` that skips or alters one of the scan's transitions,
at another root or domain, or none, `UNSTATES`), as is one whose units read their state off the chain
(`state-misindexed`, `MISREADS`) and one whose start isn't opened to the empty state (`start-unpinned`, `UNPINS`).
The tied audits run one class circuit, one Program a case, whose SHA-512 is the parameters' `circuit` (`_held`), and
an execution phase whose tied audit runs another, or under another `circuit`, is refused (`circuit-swapped`,
`UNHOLDS`).
A scan's answer that names only lanes' `prog<j>` tied is accepted in any shape (`answer-tied-held`: every lane's,
some, in another order, one twice, none, `TIED_HELD`), and one that names anything else, a tie the check does not
wire, is refused (`answer-tied-unheld`: a name that is no lane's, or no list of names, `TIED_UNHELD`). An answer whose
profile samples more than its level holds is refused
(`OVERSAMPLED`): a compute audit's (`audit-oversampled`, tied or not; `tied-oversampled`, the tied audit of the
aimed scan) and a scan's (`answer-oversampled`), each under a registration that samples within its population.
So is one with a stratum that samples more than the stratum holds in a level whose total still holds
(`STRATUM_OVERSAMPLED`): the aimed scan's tied audit, sampled whole in two strata, with some of one stratum's sample
moved to the other (`tied-stratum-oversampled`). So is one whose strata's sizes do not sum to its level's
(`STRATA_SIZES`): that audit with a stratum holding more than it samples, each holding twice as many, or one holding
and sampling one fewer (`tied-strata-sizes`).
Every accepted answer's profile is the sample its registration states, its `population` under its sampling design:
an answer is refused whose registration states no sample (`NO_SAMPLE`: `answer-unsampled`, `_unsample`) or whose
profile is another sample of its level than the registration's (`NOT_REGISTERED`: `answer-unregistered`,
`_resample`). An answer's count of incorrect proof blocks is the larger of its `bound` and that sample's bound at its
δ (`_sample_bound`), and none is refused for its `bound`: the answers' bounds are picked below their samples' as well
as above.
`case(kind, seed)` is a small period aimed at a verdict, or at a decoding failure (`KINDS`), and `cases(seed)` one of
each with its answer, as a composition's difftest takes a real component's cases.

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
from pathlib import Path

import verity_difftest as D
from verity_difftest import protocol as P
from verity.protocols.accounting.certifier.verifier import content_rows, io_request, scan_ok
from verity.protocols.accounting.certifier.verifier import network_certifier_rows as WR
from verity.protocols.accounting.certifier.verifier.certificate import (Certificate, CertifiedStart, Direction, LmsKey,
                                                                        start_digest, start_link)
from verity.protocols.accounting.certifier.verifier.link_frame import HEADER
from verity.protocols.accounting.certifier.verifier.id_form import ID_DIGITS, ID_FIELD_BYTES, MAX_TOKENS_AT
from verity.protocols.accounting.certifier.verifier.network_certifier import INTEGER_FIELDS, OPTIONAL_FIELDS, NetworkCertifierParams
from verity.protocols.accounting.pouw.verifier import core_check as CC
from verity.protocols.proofs.spot.verifier.one_stage import registered as G, docket as R
from verity.protocols.proofs.spot.verifier.one_stage.draw import stratum_k
from verity.protocols.proofs.spot.verifier.profile import Stratified
from verity.circuits import boolean as BL, codec, population
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.defs import REGISTRY
from verity.circuits.program import Program
from verity.circuits.refs import array_of, coll_concat, tuple_of
from verity.circuits.types import Array, Tuple, Value
from verity.commitments import advice as ADV
from verity.commitments.frame_v3 import BitRowLeaf
from verity.commitments.identity import tagged_sha512
from verity.commitments.merkle import CommitmentDomain
from verity.commitments.crypto import lms
from verity_vllm.definitions.registry.sampling import bit_at

ID = "verity/certifier/v0"
COMMITMENT_TAG = "verity-network-certifier/period/v0"
ONE_STAGE = R.PROTOCOL
FORMATS = (R.FORMAT, R.FORMAT_V2)
#: a proof block's public outputs are at most 16 bits (the values it commits can be wider)
OUT_BITS = 16
UNIT = "unit"
EGRESS, INGRESS = Direction.EGRESS, Direction.INGRESS
ACCEPT, REJECT, UNDECIDED = "accept", "reject", "undecided"
ROWS_PARAMS_KEYS = ["audits", "cells", "circuit", "first", "jitter", "keys", "link", "lms", "program", "scan",
                    "schedule", "windows"]
ENTRY_KEYS = ["key", "session", "start"]
#: What `params` says, as `paramsOf` does, of the entries the parameters list (`"keys"`): none; an entry that starts
#: before the one listed before it; two entries with one start; two entries with one key identifier (a key's first 16
#: bytes) and one session; two keys with one key identifier; a period that starts before the first entry's stretch; an
#: entry that is no object; and the earlier forms, `"key"`, `"session"` and `"sessions"`
NO_KEY = "keys: no key"
OUT_OF_ORDER = "keys: out of order"
OVERLAP = "their stretches overlap"
TWICE = "keys: one session listed twice"
TWO_KEYS = "keys: two keys with one key identifier"
#: What the check says of the same two of any parameters, before the starts (`entriesOrderRefusal`,
#: `entriesKeyRefusal`): entries whose starts do not strictly increase, and two keys with one key identifier
ENTRIES_OUT_OF_ORDER = "entries: out of order"
ENTRIES_ONE_ID = "entries: two keys with one key identifier"
BEFORE = "is before the first listed entry's stretch"
NO_ENTRY = "keys: an entry is not an object"
KEY_NOT_KEYS = 'key: write "keys"'
SESSION_IN_ENTRY = "session: write each key's session in its entry"
SESSIONS_IN_ENTRY = "sessions: write each key's session in its entry"
#: What a refusal says, as Lean's does (`unlisted`), of a certificate or a rows registration whose signer the
#: parameters don't list, of one whose signer is a listed key outside its stretch, and of an execution phase no listed
#: entry's stretch holds
UNLISTED = "a signer that is not one of the listed keys"
OUTSIDE = "a listed key outside its stretch"
NO_STRETCH = "no listed entry's stretch holds execution phase"
#: What a refusal says, as Lean's does (`startCheck`, `startChecks`), of a listed entry's start: a start shown for it
#: that is not its session's certified start, then why, by `CertifiedStart.refusal`'s code; an entry that does not
#: start where its key's first listed entry and their sessions' signed bases put it; an entry whose session's ordinal
#: is not the next after that of its key's entry listed before it (`follows`); and evidence that does not show a start
#: for each listed entry
UNCERTIFIED = "a start its session did not certify"
NOT_SIGNED = "a start that is not the one its session signed"
SKIPPED = "a session that is not the one after its key's entry listed before it"
STARTS_COUNT = "listed entries: show each listed entry's session's signed start"
START_KEYS = ["base_ms", "certificate", "ordinal", "schedule"]
SCHEDULE_KEYS = sorted(INTEGER_FIELDS)
#: the keys a schedule has only where it sets them (`optionalScheduleKeys`)
OPTIONAL_SCHEDULE_KEYS = sorted(OPTIONAL_FIELDS)
#: what the check says of a schedule with connection rows (`connectionRowsRefusal`)
CONNECTION_ROWS = ("schedule: connection rows: the check reads no connection rows' registration, so it refuses a "
                   "schedule with lengths rows, content rows or request words (connectionRowsCheck)")
CERTIFICATE_KEYS = ["digest", "direction", "key_id", "link", "signature", "window"]
ROWS_WINDOW_KEYS = ["answer", "certificates", "registrations"]
#: `paramsOf`'s refusal of parameters naming a `sync_set`, the clock form's, which the check doesn't read
CLOCK_FORM = "sync_set: the clock form is not one this check reads"
ROWS_KEYS = ["cell_bytes", "certificate", "direction", "domain", "format", "layout", "link", "parameters", "root",
             "row_schema", "rows", "value", "window", "wiring", "word_bits", "words"]
LAYOUT_KEYS = ["buckets", "connection_slots", "first_bucket", "slots"]
DIGEST = re.compile("[0-9a-f]{128}")
U32 = 1 << 32
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
    """`Audit`: a compute audit's layout terms (its free inputs and monitoring period) and what its Program gives: its
    SHA-512, its units (its root's template instances) and its slot entries'
    templates (`entries`: each root template node calling `SlotEntry_v1` or `SelectEntry_v1`, by descriptor id, members
    and closure digest, `slot_entry_digest`)."""
    free: int
    steps: int
    program: str
    units: int
    entries: tuple[tuple[str, int, str], ...] = ()


@dataclasses.dataclass(frozen=True)
class Entry:
    """One entry of the parameters' `"keys"`, a certifier session's: its LMS key, its session, and the start of its
    stretch of execution phases, in the numbers `first` is in."""
    key: LmsKey
    session: bytes
    start: int


@dataclasses.dataclass(frozen=True)
class Params:
    audits: tuple[Audit, ...]
    cells: tuple[int, ...]
    first: int
    jitter: int
    link: str
    #: by their stretches' starts, which strictly increase
    entries: tuple[Entry, ...]
    program: str
    windows: int
    #: the schedule and the scan's terms, read off its Program
    schedule: NetworkCertifierParams
    scan: Scan
    #: the class circuit, its window circuit's Program's SHA-512
    circuit: str | None = None


@dataclasses.dataclass(frozen=True)
class Scan:
    """`Scan`: the scan's monitoring period (T), its own free inputs (none: its `prog` is a compute audit's slot
    entries, `_wired`, whose free inputs an execution phase's term counts, `_tied_audits`), the bits one ingress and one
    egress instance commit, its slot entries' bits (`E_w`), all a tied audit's incorrect proof blocks can set, and its
    state's full width (`start`: the start unit's ``sin``, what a period's first start that is not the run's first is
    charged, `_start_refusal`)."""
    steps: int
    free: int
    ingress: int
    egress: int
    entries: int = 0
    start: int = 0


@dataclasses.dataclass(frozen=True)
class Commitment:
    """One-stage's registration records of the audits and the execution phases, as they decode."""
    audits: list[dict]
    windows: list[dict]
    bytes: bytes


@dataclasses.dataclass(frozen=True)
class Window:
    answer: dict
    certificates: list[Certificate]
    registrations: list[dict]


@dataclasses.dataclass(frozen=True)
class Facts:
    units: int
    sampled: int
    delta: Fraction
    wrong: int
    #: `design`: the sample's design as a profile's level writes it, the registered one (`_facts`), off which `at`
    #: reads the bound at another δ; None where none is read (`subset 0`), at which every unit may be incorrect
    design: dict | None = None

    def at(self, delta: Fraction) -> Facts:
        """`Facts.readAt`: the same sample read at `delta`, its count the larger of the facts' own and the sample's
        bound there (`_sample_bound`), no second sample."""
        bound = _sample_bound(self.units, {"subset": 0} if self.design is None else self.design, delta)
        return dataclasses.replace(self, delta=delta, wrong=max(self.wrong, bound))


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


def rows_wf(s: NetworkCertifierParams, windows: int) -> bool:
    """`rowsWf`: a cell's B bytes are under 2^32, so its payload's length is a u32, and the period's
    `lanes · windows · window` connections, the k-th at ordinal k + 1, have distinct u32 ordinals."""
    return s.cell_bytes < U32 and s.lanes * windows * s.window < U32


def rows_wf_refusal(s: NetworkCertifierParams, windows: int) -> str:
    return (f"rows format: a frame's ordinal and payload length are u32s, so the lanes · windows · window = "
            f"{s.lanes * windows * s.window} connections and a cell's {s.cell_bytes} bytes must each be under 2^32, or "
            "two frames could encode alike (rowsWf)")


def connection_rows(s: NetworkCertifierParams) -> bool:
    """`Schedule.connectionRows`: whether the schedule registers connection rows, its lengths rows, which a schedule
    with content rows has too, as one with request words has content rows. Each is asked, as the Lean asks it."""
    return s.lengths is not None or s.content is not None or s.request_words is not None


def _classes(j) -> tuple[tuple[int, ...], ...]:
    """A schedule's content rows' classes as its JSON holds them: a list of lists of naturals."""
    return tuple(tuple(_nat(x, "schedule.content") for x in _of(c, list, "schedule.content"))
                 for c in _of(j, list, "schedule.content"))


def _schedule(j) -> NetworkCertifierParams:
    """`scheduleOf`: `NetworkCertifierParams`' integer fields, each a positive integer, B past a cell's header, and B, S and T
    u32s, in the id egress form, and each optional field the schedule sets (`OPTIONAL_SCHEDULE_KEYS`), as
    `NetworkCertifierParams` holds it. `request_words` is the JSON `true` or it is refused: `NetworkCertifierParams` reads `None` as
    unset, and `scheduleOf` refuses a key that is there and not `true`."""
    keys = sorted(SCHEDULE_KEYS + [k for k in OPTIONAL_SCHEDULE_KEYS if isinstance(j, dict) and k in j])
    _exactly(j, keys, "schedule")
    if "request_words" in keys and j["request_words"] is not True:
        raise D.Reject("schedule: request_words is true, for a schedule with request words, or unset")
    values = {k: _classes(j[k]) if k == "content" else j[k] if k == "request_words" else _nat(j[k], "schedule")
              for k in keys}
    try:
        s = NetworkCertifierParams(**values)
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


def prog_params(lanes: int) -> tuple[str, ...]:
    """`progParams`: an ingress bucket's entry ports, `prog<j>` lane j's slot entry."""
    return tuple(f"prog{j}" for j in range(lanes))


#: `startParams`: the start unit's parameters in the generator form (`ingress_scan.IngressScanStart`)
START_PARAMS = ("sin",)


def ingress_params(lanes: int) -> tuple[str, ...]:
    """`ingressParams`: each ingress bucket's parameters (`ingress_scan.IngressScanBucket`)."""
    return ("row", "sin", *prog_params(lanes))


def leaves_of(t) -> int:
    """`Ty.leaves`: a descriptor type's leaves."""
    kind = t[0]
    if kind == "Value":
        return 1
    if kind == "Array":
        return t[1] * leaves_of(t[2])
    return sum(leaves_of(f if kind == "Tuple" else f[1]) for f in t[1])


def _widths(t) -> set[int]:
    """The widths of a descriptor type's leaves, each once (`Flock.Program.Widths.homo`'s, where there is one)."""
    kind = t[0]
    if kind == "Value":
        return {t[1]}
    if kind == "Array":
        return _widths(t[2]) if t[1] else set()
    return set().union(*(_widths(f if kind == "Tuple" else f[1]) for f in t[1]))


def _same_leaves(a, b) -> bool:
    """`sameLeaves`: two types of as many leaves, each of one width, the same in both."""
    return leaves_of(a) == leaves_of(b) and len(_widths(a)) == 1 and _widths(a) == _widths(b)


def _ingress_return(ret, sin, state) -> bool:
    """`ingressReturn`: a tuple of exactly a u16, a state of the leaves of `sin`, the ingress template's own, and a
    view of the leaves of `state`, the egress template's."""
    if ret[0] != "Tuple" or len(ret[1]) != 3:
        return False
    out, passed, view = ret[1]
    return out == BOOL_U16 and _same_leaves(passed, sin) and _same_leaves(view, state)


def _start_return(ret, own, sin) -> bool:
    """`startReturn`: the start template's return a tuple of exactly one row, and that row and its own parameter
    `own` each of the leaves of `sin`, an ingress bucket's."""
    return ret[0] == "Tuple" and len(ret[1]) == 1 and _same_leaves(ret[1][0], sin) and _same_leaves(own, sin)


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
    """`InferenceOnlyCheck.Desc.isInputNode`: a node calling an input primitive."""
    fid = j["callees"][x["fn"]]
    return j["definitions"][fid]["form"] == "primitive" and bool(INPUT_ID.match(fid))


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


def _member_ref(x: dict, o: int, m: int, k: int, j: dict) -> tuple[str, int, int] | None:
    """`Arg.memberRef`: member `m`'s leaf `k` of node `x`'s argument `o`, with the dims the decoder gives a batch
    argument from its axis; None past the argument's references."""
    a, nodes = x["args"][o], j["definitions"][j["root"]]["body"]["nodes"]
    ax = (x.get("axes") or [None] * len(x["args"]))[o]
    t = j["types"][a["type"]] if isinstance(a["type"], int) else a["type"]
    dims = _axis_dims(t, ax) if ax is not None else None
    refs = _refs(a, nodes)
    i = k if dims is None else (k // dims[1] * dims[0] + m) * dims[1] + k % dims[1]
    return refs[i] if i < len(refs) else None


def _reads_entries(x: dict, o: int, t0: int, every: int, E: int, ks: list[int], j: dict) -> bool:
    """`readsEntries`: for each lane `i` and member `m` (bucket `t0 + m`), argument `o + i` is row
    `⌊(t0 + m) / every⌋` of input node `ks[i]`, lane `i`'s slot entries, `E` leaves a row, leaf by leaf."""
    return all(o + i < len(x["args"]) and all(
        _member_ref(x, o + i, m, ell, j) == ("n", k, (t0 + m) // every * E + ell)
        for m in range(x["n"]) for ell in range(E)) for i, k in enumerate(ks))


#: `Flock.Steps.FUEL`: the most steps `Prog.gate`'s walk to an operand takes
FUEL = 2 ** 62


def _walkable(j: dict) -> bool:
    """`InferenceOnlyCheck.Desc.walkable` as Flock's decoder gives a Program (its layout, callees first and the root without
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
    """`InferenceOnlyCheck.DefinitionDigest.closureDigest`: the SHA-512, in hex, of the canonical JSON object that maps each
    Definition `key` reaches (itself, its nodes' callees, theirs, …) to its `codec.definition_digest`: the template's
    whole body."""
    defs, callees, seen, todo = j["definitions"], j["callees"], {}, [key]
    while todo:
        k = todo.pop()
        if k not in seen:
            seen[k] = codec.definition_digest(defs[k], j)
            if defs[k]["form"] == "composite":
                todo += [callees[x["fn"]] for x in defs[k]["body"]["nodes"]]
    return hashlib.sha512(codec.canonical_json(seen)).hexdigest()


def _audit(a, ctx: str) -> Audit:
    """`auditOf`: the layout terms and the Program's descriptor, whose root's template instances are the audit's units,
    each committing its template's return; a root with no template refused; its slot entries' templates
    (`ENTRY_IDS`) with their closure digests. It reads the descriptor's root nodes, signatures and the slot entries'
    Definitions; Flock's decoder, which Lean runs on it, refuses no descriptor this module writes."""
    _exactly(a, ["free", "program", "steps"], ctx)
    free, steps, j = _nat(a["free"], ctx), _nat(a["steps"], ctx), a["program"]
    if not isinstance(j, dict) or j.get("schema") != SCHEMA:
        raise D.Reject(f"{ctx}: program: not a Program's descriptor")
    defs, callees = j["definitions"], j["callees"]
    nodes = defs[j["root"]]["body"]["nodes"]
    units = [_members(x) for x in nodes if defs[callees[x["fn"]]]["form"] != "primitive"]
    if not units:
        raise D.Reject(f"{ctx}: program: no template")
    entries = tuple((k, _members(x), slot_entry_digest(j, k)) for x in nodes
                    if (d := defs[k := callees[x["fn"]]])["form"] != "primitive" and d.get("id") in ENTRY_IDS)
    return Audit(free, steps, program_sha512(j), sum(units), entries)


def _scan(j, program: str, s: NetworkCertifierParams) -> Scan:
    """`scanOf`: the scan's terms from its Program's descriptor, which must be the registered Program and the generator
    form of the schedule's execution phase in the Boolean profile's types (`BOOL_BYTE`, `BOOL_U16`): the start unit, T
    ingress buckets and T egress buckets. An egress template with a row among its parameters (the checker form) is
    refused first, and an ingress return that is not a u16, a state of `sin`'s leaves and a view of the egress
    `state`'s (`_ingress_return`), or a start template that does not take and return a state of those leaves
    (`_start_return`): the rows the scan's ties equate. Then what `NetTiming.BoundaryGates.Fits` asks: the
    walk within `FUEL` (`_walkable`), the root returning every egress leaf, the egress template returning only gates it
    computes (`_own_returns`), the ingress row and every template's row, state and egress argument one input node's
    leaves (`_input_arg`), no egress argument the ingress rows', bucket t's `prog<j>` row `⌊t / every⌋` of lane j's
    own input node of its slot entries (`_reads_entries`), the row, state and egress arguments' input nodes
    pairwise distinct and none a lane's slot entries' (`argsApart`), each of those arguments all its node's leaves,
    each lane's slot entries' node its entries' leaves and no other input node (`inputsRead`), and the Program
    topological (`_topological`). It reads
    the descriptor's nodes, signatures and returns; Flock's decoder, which Lean runs on it, refuses no descriptor
    this module writes."""
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
    xe, be = templates[-1]
    if any(k == "row" or t == row for k, t in be["params"]):
        raise D.Reject("scan: the egress template takes its row as an input (the checker form)")
    if len(templates) != 3:
        raise D.Reject("scan: not the generator form's templates")
    (xs, bs), (xi, bi) = templates[:2]
    names = lambda b: tuple(k for k, _ in b["params"])  # noqa: E731
    L = s.lanes
    if names(bs) != START_PARAMS or names(bi) != ingress_params(L) or names(be) != EGRESS_PARAMS:
        raise D.Reject("scan: the templates' parameters are not the generator form's")
    if not all(x["form"] == "batch" and x["n"] == n for x, n in ((xs, 1), (xi, T), (xe, T))):
        raise D.Reject(f"scan: not the start unit, {T} ingress buckets and {T} egress buckets")
    if bi["params"][0][1] != row:
        raise D.Reject("scan: the ingress template's row is not the schedule's")
    ety = bi["params"][2][1]
    if any(t != ety for _, t in bi["params"][2:]):
        raise D.Reject("scan: the ingress template's entry ports' types differ")
    if be["ret_type"] != ["Tuple", [BOOL_U16, row]]:
        raise D.Reject("scan: the egress template doesn't return a u16 and the schedule's row")
    sin = bi["params"][1][1]
    if not (_ingress_return(bi["ret_type"], sin, be["params"][0][1])
            and _start_return(bs["ret_type"], bs["params"][0][1], sin)):
        raise D.Reject("scan: the ingress template doesn't return a u16, a state of its sin's leaves and a view of an "
                       "egress bucket's state's leaves, or the start template doesn't take and return a state of "
                       "those leaves, so a tie of the scan's state or view would equate rows of two lengths "
                       "(NetworkCheck.Period.ingressReturn, startReturn)")
    inputs = sum(int(w[1]) * _members(x) for x in nodes if (w := INPUT_ID.match(callees[x["fn"]])))
    bits = lambda ps: sum(bits_of(t) for _, t in ps)  # noqa: E731
    nb = T // s.every
    if inputs != bits(bs["params"]) + T * bits(bi["params"][:2]) + T * bits(be["params"]) + L * nb * bits_of(ety):
        raise D.Reject("scan: an input gate that is no template's row, state or egress parameter, nor one of the slot "
                       "entries")
    if not _walkable(j):
        raise D.Reject("scan: not laid out as Flock's decoder lays a Program out, or past what its resolution walks "
                       "(NciCheck.Desc.rootLaidOut, inputsAtRoot and walkable)")
    ke = max(k for k, x in enumerate(nodes) if defs[callees[x["fn"]]]["form"] != "primitive")
    if not any(a[:3] == ("n", ke, 0) and T * leaves_of(be["ret_type"]) <= a[3]
               for q in _parts(defs[j["root"]]["body"]["ret"], nodes) if (a := _affine(q))):
        raise D.Reject("scan: the root doesn't return every egress leaf, so an egress gate would be one the Program "
                       "doesn't return (NetTiming.BoundaryGates.Fits)")
    if not _own_returns(j)(callees[xe["fn"]]):
        raise D.Reject("scan: the egress template returns a leaf it doesn't compute (a parameter's, passed through), "
                       "so an egress gate would be an input (NetTiming.BoundaryGates.Fits)")
    rows = _input_arg(xi["args"][0], nodes, j) if xi["args"] else None
    if rows is None:
        raise D.Reject("scan: the ingress template's row is not one input node's leaves, so an ingress gate would be "
                       "no input (NetTiming.BoundaryGates.Fits)")
    own_args = [*xs["args"][:1], *xi["args"][:2], *xe["args"]]
    if (len(xs["args"]) != 1 or len(xi["args"]) != 2 + L
            or any(_input_arg(a, nodes, j) is None for a in own_args)):
        raise D.Reject("scan: a template's row, state or egress argument is not one input node's leaves")
    if any(_input_arg(a, nodes, j) == rows for a in xe["args"]):
        raise D.Reject("scan: an egress template's argument is the ingress rows' input node, so an egress gate would "
                       "depend on the ingress (NetTiming.BoundaryGates.Fits)")
    r0 = [_member_ref(xi, 2 + i, 0, 0, j) for i in range(L)]
    pks = [r[1] for r in r0 if r is not None]
    if (len(pks) != L or any(r[0] == "p" for r in r0) or len(set(pks)) != L
            or any(k == rows or not (k < len(nodes) and _is_input(nodes[k], j)) for k in pks)):
        raise D.Reject("scan: bucket 0's prog<j> reads no input node of lane j's own, its slot entries', distinct from "
                       "every other lane's")
    E = leaves_of(ety)
    if not _reads_entries(xi, 2, 0, s.every, E, pks, j):
        raise D.Reject(f"scan: bucket t's prog<j> is not row ⌊t / {s.every}⌋ of lane j's slot entries' input node, "
                       "its slot of the boundary bucket t reads (NetworkCheck.Period.readsEntries)")
    own = [_input_arg(a, nodes, j) for a in own_args]
    if len(set(own)) != len(own) or any(k in own for k in pks):
        raise D.Reject("scan: two of the templates' row, state and egress arguments read one input node, or one reads "
                       "a lane's slot entries': an input node would be left that nothing reads, or a slot entry would "
                       "be a state gate (NetworkCheck.Period.argsApart)")
    whole = lambda a: (q := _affine(_parts(a["refs"], nodes)[0]))[3] == _members(nodes[q[1]])  # noqa: E731
    if not (all(whole(a) for a in own_args) and all(_members(nodes[k]) == nb * E for k in pks)
            and E <= bits_of(ety) and all(k in pks or k in own for k, x in enumerate(nodes) if _is_input(x, j))):
        raise D.Reject("scan: an input node with a leaf nothing reads: a row, state or egress argument that is not all "
                       f"its input node's leaves, a lane's slot entries' node of other leaves than its {nb} entries' "
                       "(each of at least a bit), or an input node that is neither's "
                       "(NetworkCheck.Period.inputsRead)")
    if not _topological(j):
        raise D.Reject("scan: a body names a later node (Flock.Program.Prog.orderViolation): the Program is not "
                       "topological, so a gate's operands could be gates after it")
    return Scan(T, 0, bits_of(bi["ret_type"]), bits_of(be["ret_type"]), L * nb * bits_of(ety),
                bits_of(bs["params"][0][1]))


def _entry(j, lp: lms.Params) -> Entry:
    if not isinstance(j, dict):
        raise D.Reject(f"{NO_ENTRY}: write each as a key, its session and its start")
    _exactly(j, ENTRY_KEYS, "keys")
    key = LmsKey(lp, P.unhex(_of(j["key"], str, "keys")))
    if len(key.public) != 16 + lp.N:
        raise D.Reject(f"an LMS public key is {16 + lp.N} bytes")
    session = P.unhex(_of(j["session"], str, "keys"))
    if len(session) != lms.SESSION_BYTES:
        raise D.Reject(f"a certifier session id is {lms.SESSION_BYTES} bytes")
    return Entry(key, session, _nat(j["start"], "keys"))


def _listed(j, lp: lms.Params) -> tuple[Entry, ...]:
    """`certifierEntriesOf`: the listed entries, at least one, their starts strictly increasing, no two with one key
    identifier and one session. The period reads each session's signed start, so a key may be listed in several
    sessions, and an identifier names one key."""
    entries = tuple(_entry(e, lp) for e in _of(j, list, "keys"))
    if not entries:
        raise D.Reject(f"{NO_KEY}: list the certifier's entries, at least one")
    for x, y in zip(entries, entries[1:]):
        if y.start < x.start:
            raise D.Reject(f"{OUT_OF_ORDER}: an entry starts at {y.start}, before the one listed before it, at {x.start}")
        if y.start == x.start:
            raise D.Reject(f"keys: two entries start at {x.start}: {OVERLAP}")
    if len({(e.key.key_id, e.session) for e in entries}) != len(entries):
        raise D.Reject(f"{TWICE}: two entries hold one key identifier and one session")
    ids = {e.key.key_id for e in entries}
    if len({e.key.public for e in entries}) != len(ids):
        raise D.Reject(f"{TWO_KEYS} (a key's first 16 bytes)")
    return entries


def params(j) -> Params:
    if not isinstance(j, dict):
        raise D.Reject("the object: not an object")
    if "sync_set" in j:
        raise D.Reject(CLOCK_FORM)
    if "key" in j:
        raise D.Reject(f"{KEY_NOT_KEYS}, the list of the certifier's entries, each a key, its session and its start")
    if "session" in j:
        raise D.Reject(f'{SESSION_IN_ENTRY} of "keys"')
    if "sessions" in j:
        raise D.Reject(f'{SESSIONS_IN_ENTRY} of "keys"')
    _exactly(j, ROWS_PARAMS_KEYS, "the object")
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
    entries = _listed(j["keys"], lp)
    if first < entries[0].start:
        raise D.Reject(f"first: execution phase {first} {BEFORE}, which starts at {entries[0].start}")
    program = _of(j["program"], str, "program")
    if not DIGEST.fullmatch(program):
        raise D.Reject("program: not a SHA-512 digest")
    schedule = _schedule(j["schedule"])
    if connection_rows(schedule):
        raise D.Reject(CONNECTION_ROWS)
    scan = _scan(j["scan"], program, schedule)
    windows = _nat(j["windows"], "windows")
    if windows < 1:
        raise D.Reject("windows: a period has at least one execution phase")
    if not rows_wf(schedule, windows):
        raise D.Reject(rows_wf_refusal(schedule, windows))
    if list(cells) != schedule.rows_cells(windows):
        raise D.Reject(f"cells: the rows form's cells are egress_cells = {schedule.egress_cells} for each of the "
                       f"lanes · windows · window = {schedule.lanes * windows * schedule.window} connections the "
                       f"schedule can carry (Schedule.rowsCells), not these {len(cells)}")
    circuit = _of(j["circuit"], str, "circuit")
    if not DIGEST.fullmatch(circuit):
        raise D.Reject("circuit: not a SHA-512 digest")
    return Params(tuple(audits), cells, first, jitter, link, entries, program, windows, schedule, scan,
                  circuit)


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
                      [_registration(x, "an execution phase's registration") for x in _of(j["windows"], list, "windows")], b)


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
    _exactly(j, ROWS_WINDOW_KEYS, ctx)
    answer = _answer(j["answer"], f"{ctx}.answer")
    certificates = [_certificate(c, f"{ctx}.certificates") for c in _of(j["certificates"], list, ctx)]
    return Window(answer, certificates, [_rows(x, f"{ctx}.registrations") for x in _of(j["registrations"], list, ctx)])


def _start(j, ctx: str) -> CertifiedStart:
    """`startOf`: a session's signed start, as the run's `deployment.json` holds it (`"start"`)."""
    _exactly(j, START_KEYS, ctx)
    return CertifiedStart(_nat(j["base_ms"], ctx), _nat(j["ordinal"], ctx), P.unhex(_of(j["schedule"], str, ctx)),
                          _certificate(j["certificate"], f"{ctx}.certificate"))


def evidence(b: bytes) -> tuple[list[dict], list[Window], list[CertifiedStart]]:
    """`evidenceOf`: the audits' answers, the execution phases', and the listed entries' sessions' signed starts where
    the evidence shows any (`"starts"`); evidence that shows none has no `"starts"`."""
    j = P.canonical_json(b)
    shown = isinstance(j, dict) and sorted(j) == ["audits", "starts", "windows"]
    if not shown:
        _exactly(j, ["audits", "windows"], "the object")
    answers = [_answer(a, "an audit's answer") for a in _of(j["audits"], list, "audits")]
    windows = [_window(x, "an execution phase") for x in _of(j["windows"], list, "windows")]
    starts = [_start(x, "a start") for x in _of(j["starts"], list, "starts")] if shown else []
    if shown and not starts:
        raise D.Reject('starts: none: evidence that shows no session\'s start has no "starts"')
    return answers, windows, starts


class Finding(Exception):
    """A finding against the period: a rejection, or what leaves it undecided."""

    def __init__(self, verdict: str, reason: str):
        super().__init__(reason)
        self.verdict, self.reason = verdict, reason


def _entry_at(p: Params, w: int) -> Entry | None:
    """`entryAt`: the entry execution phase `w` picks, the last listed one whose start is at most `w`, whose stretch
    holds `w`; none before the first entry's start. The execution phase alone picks it, and nothing a certificate names
    does."""
    held = [e for e in p.entries if e.start <= w]
    return held[-1] if held else None


def _unlisted(p: Params, w: int, key_ids: list[bytes]) -> str:
    """`unlisted`: what a refusal at execution phase `w` adds about its signers `key_ids`: no listed entry's stretch
    holds `w`, a signer no entry lists, or a listed key outside its stretch."""
    e = _entry_at(p, w)
    if e is None:
        return f": {NO_STRETCH} {w}"
    if all(i == e.key.key_id for i in key_ids):
        return ""
    return f": {OUTSIDE}" if all(any(x.key.key_id == i for x in p.entries) for i in key_ids) else f": {UNLISTED}"


def _certifies_on(p: Params, link: str, d: Direction, w: int, c: Certificate) -> bool:
    """`certifiesOn`: a certificate of `link`'s end `d` for the period's execution phase `w`, under the entry `w` picks:
    its key, its session, and that session's number for the execution phase, `w` less the entry's start."""
    e = _entry_at(p, w)
    return (e is not None and c.direction is d and c.link == link and c.window == w - e.start
            and c.verify(e.key, e.session))


def _held_to(L: int, a: Entry, xa: CertifiedStart, e: Entry, x: CertifiedStart) -> bool:
    """`heldTo`: the listed entry `e`, whose session's signed start is `x`, starts where its key's entry `a`, whose
    session's is `xa`, puts it: as many execution phases of `L` milliseconds after `a`'s start as its session's base
    is after `a`'s session's on their key's life."""
    return a.start * L + x.base_ms == e.start * L + xa.base_ms


def _follows(xb: CertifiedStart, x: CertifiedStart) -> bool:
    """`follows`: the session whose signed start is `x` is the one right after the session whose signed start is `xb`
    among their key's sessions, its ordinal the next."""
    return xb.ordinal + 1 == x.ordinal


def _start_check(p: Params, starts: list[CertifiedStart], i: int, e: Entry, x: CertifiedStart) -> Finding | None:
    """`startCheck`: listed entry `i`'s start's finding. The start shown for it is its session's
    certified start, read as the network certifier's own verifier reads it (`CertifiedStart.refusal`: under the
    entry's key, in its session, for the link, under the period's schedule's parameters digest); the entry starts
    where its key's first listed entry puts it (`_held_to`), a key's first listed entry being held to itself; and its
    session is the one right after that of its key's entry listed before it (`_follows`)."""
    why = x.refusal(e.key, e.session, p.link, p.schedule.digest)
    if why is not None:
        return Finding(REJECT, f"entry {i}: {UNCERTIFIED} ({why})")
    a, xa = next((y, z) for y, z in zip(p.entries, starts) if y.key.key_id == e.key.key_id)
    if not _held_to(p.schedule.window * p.schedule.bucket_ms, a, xa, e, x):
        return Finding(REJECT, f"entry {i}: {NOT_SIGNED}: it starts at execution phase {e.start} and its session at "
                               f"{x.base_ms} ms of its key's life, and its key's entry that starts at {a.start} at "
                               f"{xa.base_ms} ms")
    before = [z for y, z in zip(p.entries[:i], starts[:i]) if y.key.key_id == e.key.key_id]
    if before and not _follows(before[-1], x):
        return Finding(REJECT, f"entry {i}: {SKIPPED}: its session is its key's number {x.ordinal} and that entry's "
                               f"{before[-1].ordinal}; its key's device state numbers its sessions one after another, "
                               f"so a gap is a session no entry lists")
    return None


def _entries_refusal(p: Params) -> str | None:
    """`entriesOrderRefusal`, then `entriesKeyRefusal`: the first entry whose start is not after the one listed before
    it, else the first two listed entries with one key identifier and different keys. The decoder refuses both
    (`_listed`); the check refuses them of any parameters."""
    for i, (x, y) in enumerate(zip(p.entries, p.entries[1:])):
        if y.start <= x.start:
            return (f"{ENTRIES_OUT_OF_ORDER}: entry {i + 1} starts at {y.start}, not after entry {i}'s {x.start}: list "
                    f"the entries by their stretches' starts, no two at one")
    for i, a in enumerate(p.entries):
        for j, b in enumerate(p.entries[i + 1:], start=i + 1):
            if a.key.key_id == b.key.key_id and a.key.public != b.key.public:
                return (f"{ENTRIES_ONE_ID} (a key's first 16 bytes), entries {i} and {j}: an identifier names one "
                        f"key")
    return None


def _start_checks(p: Params, starts: list[CertifiedStart]) -> list[Finding]:
    """`startChecks`: the listed entries' starts' findings: the one finding that the entries are out of order or that
    two have one key identifier and different keys (`_entries_refusal`), else the one finding that the evidence does
    not show a start for each, else each entry's (`_start_check`)."""
    if (why := _entries_refusal(p)) is not None:
        return [Finding(REJECT, why)]
    if len(starts) != len(p.entries):
        return [Finding(REJECT, f"starts: {len(starts)} shown for {len(p.entries)} {STARTS_COUNT}")]
    return [f for i, (e, x) in enumerate(zip(p.entries, starts)) if (f := _start_check(p, starts, i, e, x))]


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


def _rows_check(p: Params, w: int, d: Direction, doc: dict) -> None:
    """`rowsCheck`: the verifier's own read of the registration (`network_certifier_rows.check_registration`) under the entry the
    period's execution phase `w` picks: its key, its session, which the registration's domain binds, and that session's
    number for the execution phase, `w` less the entry's start, which its value, window, layout, domain and head name. A
    signer that is another listed key, outside its stretch, or that the parameters don't list is refused by name."""
    e = _entry_at(p, w)
    if e is None:
        raise Finding(REJECT, f"window {w}: the {d.value} rows{_unlisted(p, w, [])}")
    signer = bytes.fromhex(doc["certificate"]["key"])
    if signer != e.key.key_id:
        raise Finding(REJECT, f"window {w}: the {d.value} rows: the registration's certificate doesn't verify under a "
                              f"listed key in its session{_unlisted(p, w, [signer])}")
    try:
        WR.check_registration(doc, e.key, e.session, p.schedule, p.link, d, w - e.start)
    except ValueError as e:
        raise Finding(REJECT, f"window {w}: the {d.value} rows: {e}") from None


def _holds(reg: dict, doc: dict) -> bool:
    """The scan's registration holds the rows: an entry of its `roots` or `registered` names their root and domain."""
    entries = [e for k in ("roots", "registered") if isinstance(reg.get(k), list) for e in reg[k]]
    return any(isinstance(e, dict) and e.get("root") == doc["root"] and e.get("domain") == doc["domain"]
               for e in entries)


#: the scan's Programs' rows in its registration, one tree a lane, `PROG<j>` lane j's (`progName`): its `roots` entry's
#: port, or its `registered` entry's value
PROG = "scan/prog"
#: the scan's state in a registration: its `roots` entry's port (`stateName`), or its `registered` entry's value, the
#: value a window circuit registers the scan's state rows as (`stateValue`, the certifier's `blocks.SCAN_STATE`)
STATE = "scan/state"
SCAN_STATE = "verity/certifier/scan-state/v1"
#: `tiedKey`: the key of one-stage's public outputs that holds the tied Inputs its registration check let through
TIED_KEY = "rests_on_period_check"


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
    """The scan's state reads its registration's `window` states (`stateRead`): `o0` unit 0, the start unit, at leaf
    0; `sin` units 1 … T from leaf 0 and `o1` units 1 … T − 1 from leaf 1, unit t + 1, bucket t, its `sin` at leaf t
    and its `o1` at leaf t + 1."""
    return {STATE: {"o0": {"leaf": 0, "units": [0, 1]}, "o1": {"leaf": 1, "units": [1, T]},
                    "sin": {"leaf": 0, "units": [1, T + 1]}}}


def _state_read(p: Params, reg: dict) -> bool:
    """`stateRead`: the registration's `window`'s `reads` of `STATE` are `state_reads` at the schedule's T."""
    r = _get(_get(_get(reg, "window"), "reads"), STATE)
    T = p.schedule.window
    return _run(_get(r, "o0"), 0, 1, 0) and _run(_get(r, "sin"), 1, T + 1, 0) and _run(_get(r, "o1"), 1, T, 1)


class _Domain:
    """A frame-v3-sha512 domain known by its id alone, as Lean's `climb` takes one: a leaf's hash and a node's read the
    id and nothing else of the domain."""
    hash = "sha512"
    leaf, node = CommitmentDomain.leaf, CommitmentDomain.node

    def __init__(self, domain_id: bytes, count: int) -> None:
        self.domain_id, self.count = domain_id, count


#: `startName`: the registered value an execution phase's scan starts from, the start unit's `sin`
START = "0.sin"


def _starts(reg: dict) -> list[dict]:
    """`startEntries`: a registration's `registered` entries whose value is `START`."""
    return [e for e in _entries(reg, "registered") if e.get("value") == START]


def _row_bc(row: bytes, salt: bytes) -> bytes:
    """The hm96 b‖c of a row of u16 words, `row` their bytes, under `salt` (`Registered.knownPayload`)."""
    return ADV.INST.commit_string(ADV.INST.default_key, BitRowLeaf(8 * len(row)).layout().digest(row), salt)


def _start_empty(reg: dict) -> bool:
    """`startEmpty`: the registration has a start entry (`_starts`), and its `window`'s ``empty`` of `START` opens
    every one to the empty state: ``words`` (at least 1) zero u16 words under a 192-byte ``salt``, whose leaf under
    the entry's domain is the entry's root, a tree of the one leaf."""
    o, es = _get(_get(_get(reg, "window"), "empty"), START), _starts(reg)
    w, salt = _natural(_get(o, "words")), _unhex(_get(o, "salt"))
    if not es or not w or salt is None or len(salt) != ADV.INST.salt_bytes:
        return False
    bc = _row_bc(bytes(2 * w), salt)
    return all((root := _unhex(e.get("root"))) is not None and (dom := _unhex(e.get("domain"))) is not None
               and G.verify_read(_Domain(dom, 1), root, 0, bc, b"", G.SCHEMA_V2) for e in es)


def start_empty(words: int, salt: bytes, domain: bytes, row: bytes | None = None) -> tuple[dict, dict]:
    """The empty state as a scan's registration starts from it (`startEmpty`): its `registered` entry of `START`, a
    tree of the one row of `words` zero u16 words (or of `row`, a row the prover chose) under `salt` at the domain
    whose id is `domain`, and the opening its `window` gives as ``empty``."""
    root = G.leaf(_Domain(domain, 1), 0, _row_bc(bytes(2 * words) if row is None else row, salt), G.SCHEMA_V2)
    return ({"domain": domain.hex(), "leaves": 1, "root": root.hex(), "value": START, "words": words},
            {START: {"salt": salt.hex(), "words": words}})


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


def lane_tied(s: NetworkCertifierParams, t) -> bool:
    """`laneTied`: an entry under `TIED_KEY` that names a lane's Programs' rows (`prog_params`)."""
    return isinstance(t, str) and t in prog_params(s.lanes)


def tied_input_wired(s: NetworkCertifierParams, a: dict) -> bool:
    """`tiedInputWired`: the answer's public outputs have no `TIED_KEY`, or under it a list whose every entry is a
    lane's name (`lane_tied`), in any order, a lane's more than once or not at all."""
    pub = a["public"]
    if not (isinstance(pub, dict) and TIED_KEY in pub):
        return True
    return isinstance(pub[TIED_KEY], list) and all(lane_tied(s, t) for t in pub[TIED_KEY])


def tied_unheld(s: NetworkCertifierParams, a: dict) -> str:
    """`tiedUnheld`: what an answer `tied_input_wired` refuses holds under `TIED_KEY`, as Lean's refusal words it: the
    first entry that is no lane's name, or the key's value where it is no list, beside the lanes' names."""
    pub = a["public"]
    tied, lanes = pub.get(TIED_KEY) if isinstance(pub, dict) else None, ", ".join(prog_params(s.lanes))

    def text(j) -> str:
        return json.dumps(j, separators=(",", ":"), sort_keys=True)

    if not isinstance(tied, list):
        return f"is {text(tied)}, no list of names of the scan's lanes' {lanes}"
    return f"names {text(next((t for t in tied if not lane_tied(s, t)), None))}, none of the scan's lanes' {lanes}"


def _held(p: Params, audits: list[dict], reg: dict) -> bool:
    """`held`: each audit the scan's `prog` is an output of (`_tied_audits`) is registered under the parameters' class
    circuit (`runsCircuit`), none with no class circuit."""
    return all(p.circuit is not None and audits[i].get("program") == p.circuit for i in _tied_audits(audits, reg))


def _prog_tied(audits: list[dict], reg: dict) -> bool:
    """`progTied`: some audit is tied to the scan's `prog` (`_tied_audits`)."""
    return bool(_tied_audits(audits, reg))


def _tied_in(audits: list[dict], windows: list[dict], i: int) -> bool:
    """`tiedIn`: some execution phase's `prog` is audit `i`'s output (`_tied_audits`)."""
    return any(i in _tied_audits(audits, reg) for reg in windows)


def records(link: str) -> tuple[tuple[str, Direction], ...]:
    """The rows form's certified records of an execution phase (`relay_check.CERTIFIED`): requests, responses,
    fails."""
    return (link, INGRESS), (link, EGRESS), (f"{link}-fails", EGRESS)


def _rows_window_check(p: Params, audits: list[dict], w: int, reg: dict,
                       x: Window) -> tuple[Facts, int]:
    """`rowsExecutionPhaseCheck`: on acceptance the scan's facts and its tied audits' free inputs, the parameters' `free` of
    each audit its `prog` is an output of (`_tied_audits`)."""
    what = f"window {w}"
    if reg.get("program") != p.program:
        raise Finding(REJECT, f"{what}: registration")
    if not (len(x.certificates) == 3
            and all(_certifies_on(p, link, d, w, c) for c, (link, d) in zip(x.certificates, records(p.link)))):
        raise Finding(REJECT, f"{what}: certificates (none of a window in which the network certifier refused a "
                              f"connection){_unlisted(p, w, [c.key_id for c in x.certificates])}")
    if len(x.registrations) != 2:
        raise Finding(REJECT, f"{what}: registrations")
    for doc, d in zip(x.registrations, (INGRESS, EGRESS)):
        _rows_check(p, w, d, doc)
    if not all(_holds(reg, doc) for doc in x.registrations):
        raise Finding(REJECT, f"{what}: another window's answer")
    if not _prog_tied(audits, reg):
        raise Finding(REJECT, f"{what}: the scan's prog is no compute audit's output")
    prof, pub = _answered(what, reg, x.answer)
    f = _facts(what, reg, prof, pub)
    if f.units != 2 * p.schedule.window + 1:
        raise Finding(REJECT, f"{what}: profile: {f.units} units, not the scan's 2T + 1")
    return f, sum(p.audits[i].free for i in _tied_audits(audits, reg))


def _sample_of(reg) -> tuple[int, dict] | None:
    """`sampleOf`: the sample a registration states, its `population` under its sampling design as a profile's
    level writes it (a stratum named by its template, its size its ranges' total); None where it names no
    `subset:k` or stratified sample."""
    def get(j, k):
        return j.get(k) if isinstance(j, dict) else None

    def integer(v) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    def nat(v):
        return v if integer(v) and v >= 0 else None

    n, stated = nat(get(reg, "population")), get(reg, "law")
    kind = get(stated, "law")
    if n is None or not isinstance(kind, str):
        return None
    if kind == "subset":
        k = nat(get(stated, "k"))
        return None if k is None else (n, {"subset": k})
    if kind not in ("stratified", "work") or not isinstance(strata := get(stated, "strata"), list):
        return None
    out = []
    for s in strata:
        t, rs, k = get(s, "template"), get(s, "units"), nat(get(s, "k"))
        if not (isinstance(t, str) and isinstance(rs, list) and rs and k is not None and all(
                isinstance(x, list) and len(x) == 2 and integer(x[0]) and integer(x[1]) and x[0] < x[1] for x in rs)):
            return None
        out.append({"name": t, "size": sum(hi - lo for lo, hi in rs), "k": k})
    return n, {"law": "stratified", "strata": out}


def _subset_bound(n: int, k: int, delta: Fraction) -> int:
    """`Spot.Bound.boundOf n (.subset k) δ` for `k ≤ n`: one-stage's own bound of a uniform k-subset of n units
    (`Stratified.count_bound` of the one stratum) where 0 < δ < 1, and outside it what the Lean's fill returns: every
    unit the sample can miss at δ = 0, all n at δ = 1 when nothing is sampled, and none otherwise."""
    if 0 < delta < 1:
        return Stratified.uniform(n, k).count_bound(delta)
    if delta == 0:
        return n - k
    return n if delta == 1 and k == 0 else 0


def _sample_bound(n: int, design: dict, delta: Fraction) -> int:
    """`sampleBound`: a `subset:k`'s bound at δ, and a stratified sample's the sum of its strata's own, each stratum
    one `subset:k_s` of its `n_s` units at δ split equally over the strata."""
    if "subset" in design:
        return _subset_bound(n, design["subset"], delta)
    strata = design["strata"]
    return sum(_subset_bound(s["size"], s["k"], delta / len(strata)) for s in strata)


def _facts(what: str, reg, prof: dict, pub) -> Facts:
    """`factsOf`: one level sampled `subset:k` or stratified, sampling no more than the level holds and no more of a
    stratum than the stratum holds, its strata's sizes summing to the level's; the level its registration's sample
    (`_sample_of`); and its count the larger of the public's bound and that sample's (`_sample_bound`)."""
    if len(prof["levels"]) != 1:
        raise Finding(REJECT, f"{what}: profile: not one level")
    (lv,) = prof["levels"]
    design = lv["draw"]
    if isinstance(design, dict) and "subset" in design:
        sampled = design["subset"]
    elif isinstance(design, dict) and design.get("law") == "stratified":
        sampled = sum(s["k"] for s in design["strata"])
    else:
        raise Finding(REJECT, f"{what}: profile: not a subset or stratified sample")
    if sampled > lv["size"]:
        raise Finding(REJECT, f"{what}: profile: {sampled} sampled of {lv['size']} units: {OVERSAMPLED}")
    for s in design.get("strata", ()):
        if s["k"] > s["size"]:
            raise Finding(REJECT, f"{what}: profile: stratum '{s['name']}': {s['k']} sampled of {s['size']} units: "
                                  f"{STRATUM_OVERSAMPLED}")
    if "strata" in design and (held := sum(s["size"] for s in design["strata"])) != lv["size"]:
        raise Finding(REJECT, f"{what}: profile: strata of {held} units in a level of {lv['size']}: {STRATA_SIZES}")
    if (sample := _sample_of(reg)) is None:
        raise Finding(REJECT, f"{what}: {NO_SAMPLE}: it names no subset or stratified sample of a population")
    if sample != (lv["size"], design):
        raise Finding(REJECT, f"{what}: profile: {NOT_REGISTERED}: its level of {lv['size']} units and its sampling "
                              f"design are not the registration's {sample[0]} units and theirs")
    bound = pub.get("bound") if isinstance(pub, dict) else None
    if not isinstance(bound, int) or isinstance(bound, bool) or bound < 0:
        raise Finding(REJECT, f"{what}: no bound")
    count = max(bound, _sample_bound(lv["size"], design, prof["delta"]))
    return Facts(lv["size"], sampled, prof["delta"], count, design)


def _audit_check(i: int, audit: Audit, reg: dict, a: dict) -> Facts:
    """`auditCheck`: the registration names the audit's Program and the answer's proof blocks are the Program's, at
    any sample."""
    what = f"audit {i}"
    if reg.get("program") != audit.program:
        raise Finding(REJECT, f"{what}: registration: not its Program")
    prof, pub = _answered(what, reg, a)
    f = _facts(what, reg, prof, pub)
    if f.units != audit.units:
        raise Finding(REJECT, f"{what}: profile: {f.units} units, not its Program's {audit.units}")
    return f


#: `factsOf`'s refusal of a profile whose level samples more than it holds, after the two counts.
OVERSAMPLED = "more than the level holds"
#: `factsOf`'s refusal of a profile with a stratum that samples more than it holds, after the stratum's two counts.
STRATUM_OVERSAMPLED = "more than the stratum holds"
#: `factsOf`'s refusal of a profile whose strata's sizes do not sum to its level's, after the two sizes.
STRATA_SIZES = "their sizes do not sum to the level's"
#: `factsOf`'s refusal of an answer whose registration states no subset or stratified sample.
NO_SAMPLE = "registration: no sample"
#: `factsOf`'s refusal of a profile whose level is not the sample its registration states.
NOT_REGISTERED = "not the registration's sample"
#: `wiredCheck`'s refusal, after its window's number.
UNWIRED = "the scan's prog is no compute audit's slot entries"
#: `stateCheck`'s refusals, after its window's number: the state untied, its reads off the chain.
UNSTATED = "the scan's state is no wired audit's SCAN_STATE"
#: `tiedDelta`: δ_i, the δ at which the rows form reads a tied audit's sample (`chargedOf`), fixed before any coin.
TIED_DELTA = Fraction(1, 2**64)
#: `tiedInputCheck`'s refusal, after its window's number and before what the answer holds instead (`tied_unheld`).
TIE_UNHELD = "the scan's answer rests on a tie this check does not hold"


MISREAD = "the scan's state reads are not its transitions' chain"
#: `scanStartCheck`'s refusals, after its window's number: an execution phase after its period's first, and the run's first
#: one's start that is not the empty state.
CARRIED = "the scan's start is the state the execution phase before it ended in"
UNPINNED = "the scan's start is not the empty state"


def _state_refusal(p: Params, audits: list[dict], reg: dict) -> str | None:
    """`stateCheck`'s refusal of the scan of an execution phase's transfer blocks: none when its state is tied
    (`_state_tied`)."""
    if not _state_tied(p, audits, reg):
        return UNSTATED if _state_read(p, reg) else MISREAD
    return None


def _start_refusal(w: int, i: int, reg: dict) -> str | None:
    """`scanStartCheck`'s refusal of execution phase `w`'s scan, the `i`-th of its period: none when it is its period's
    first and, where it is the run's first, execution phase 0, its scan starts from the empty state (`_start_empty`).
    A period's first that is not the run's first starts from a state the check holds to nothing, and is charged
    (`public`'s ``start``). A period is one execution phase: a later one is refused (`CARRIED`), so a period spanning a
    restart is too, though its device restarted honestly."""
    if i != 0:
        return CARRIED
    return None if w != 0 or _start_empty(reg) else UNPINNED


#: `heldCheck`'s refusal, after its execution phase's number.
UNHELD = "the tied audit's window circuit is not the registered class circuit"
#: `okCheck`'s refusal, after its execution phase's number.
UNOK = "the scan's ok bits are not the schedule's constants"
#: `okCheck`'s refusal where the schedule has no constants, after its execution phase's number and before why.
UNCOMPUTED = "the scan's ok bits' constants do not compute"


def clog2(x: int) -> int:
    """`⌈log2 x⌉`, 0 for x ≤ 1."""
    return 0 if x <= 1 else (x - 1).bit_length()


def at_most(n: int, b: int) -> int:
    """`Σ_{j ≤ b} C(n, j)`: the sets of at most `b` of `n` units."""
    return sum(math.comb(n, j) for j in range(min(b, n) + 1))


def _facts_json(f: Facts) -> dict:
    return {"delta": str(f.delta), "drawn": f.sampled, "units": f.units, "wrong_units": f.wrong}


def _terms(free: int, steps: int, per_unit: int, f: Facts, jitter: int, entries: int | None = None) -> dict:
    """`AuditTerms.of` (a compute audit's, `per_unit` its Program's widest unit's committed bits) and `Scan.terms` (a
    scan's, `per_unit` the wider instance's committed bits, `entries` the slot entry bits its tied audits' incorrect
    proof blocks can set, `faulty_entries`)."""
    return _facts_json(f) | {"faulty_bits": f.wrong * per_unit, "faulty_sets": clog2(at_most(f.units, f.wrong)),
                             "free": free, "jitter": clog2(jitter ** steps)} | (
        {} if entries is None else {"faulty_entries": entries})


def public(p: Params, facts: list[Facts], scans: list[tuple[Facts, int, int]], charged: list[Facts] = (),
           start: int = 0) -> dict:
    """K and its terms over `MonitoringPeriodLeakAccepted`'s windows: the execution phases' scans (`scans`, each its facts,
    its tied audits' free inputs and its `β_w`), where a compute audit's own terms are in none, a tied audit's free
    inputs are in its execution phase's `free` and its incorrect proof blocks in its `faulty_entries`, a tied audit
    sampled in part, read at `TIED_DELTA` (`charged`), is in the public's `charged` and its δ_i in `eps`, and the
    period's first start, where it is not the run's first, is charged ``start``, the state's full width, once."""
    cells = sum(clog2(cap + 1) for cap in p.cells)
    s = p.scan
    units = [_terms(s.free + tied, s.steps, max(s.ingress, s.egress), f, p.jitter, b) for f, tied, b in scans]
    schedules = cells
    out = {"audits": [_facts_json(f) for f in facts], "charged": [_facts_json(f) for f in charged], "scans": units,
           "start": start}
    if p.schedule.max_tokens_cap is not None:
        out["max_tokens_cap"] = p.schedule.max_tokens_cap
    K = schedules + cells + start + sum(t["free"] + t["faulty_sets"] + t["faulty_bits"] + t["jitter"]
                                + t.get("faulty_entries", 0) for t in units)
    eps = sum((Fraction(t["delta"]) for t in units), Fraction(0)) + sum((f.delta for f in charged), Fraction(0))
    return out | {"K": K, "cells": cells, "eps": str(eps), "schedules": schedules, "windows": p.windows}


def check(p: Params, c: Commitment, e) -> tuple:
    audits, windows, starts = e
    if len(c.windows) != p.windows:
        return REJECT, "the commitment registers another number of execution phases"
    if len(c.audits) != len(p.audits):
        return REJECT, "the commitment registers another number of audits"
    if len(windows) != len(c.windows):
        return REJECT, "evidence: another number of execution phases"
    if len(audits) != len(c.audits):
        return REJECT, "evidence: another number of audits' answers"
    findings, facts, scans = _start_checks(p, starts), [], []
    for i, (reg, x) in enumerate(zip(c.windows, windows)):
        try:
            scans.append(_rows_window_check(p, c.audits, p.first + i, reg, x))
        except Finding as f:
            findings.append(f)
    if any(f.delta != scans[0][0].delta for f, _ in scans):
        findings.append(Finding(REJECT, "a scan's δ is not the period's"))
    if not rows_wf(p.schedule, p.windows):
        findings.append(Finding(REJECT, rows_wf_refusal(p.schedule, p.windows)))
    if p.schedule is not None and connection_rows(p.schedule):
        findings.append(Finding(REJECT, CONNECTION_ROWS))
    charged = {}
    for i, (audit, reg, a) in enumerate(zip(p.audits, c.audits, audits)):
        tied = _tied_in(c.audits, c.windows, i)
        try:
            facts.append(f := _audit_check(i, audit, reg, a))
        except Finding as x:
            findings.append(x)
        else:
            if tied and f.sampled < f.units:
                charged[i] = f.at(TIED_DELTA)
    findings += [Finding(REJECT, f"window {p.first + i}: {UNWIRED}") for i, reg in enumerate(c.windows)
                 if not _wired(p, c.audits, reg)]
    findings += [Finding(REJECT, f"window {p.first + i}: {why}") for i, reg in enumerate(c.windows)
                 if (why := _state_refusal(p, c.audits, reg))]
    findings += [Finding(REJECT, f"window {p.first + i}: {TIE_UNHELD}: its public outputs' {TIED_KEY} "
                                 f"{tied_unheld(p.schedule, x.answer)}")
                 for i, x in enumerate(windows) if not tied_input_wired(p.schedule, x.answer)]
    findings += [Finding(REJECT, f"window {p.first + i}: {why}") for i, reg in enumerate(c.windows)
                 if (why := _start_refusal(p.first + i, i, reg))]
    # `okChecks`: the schedule's constants, hashed once a period, and each registration compared to them; a
    # schedule with none refuses every registration
    try:
        consts = scan_ok.constants(p.schedule.window)
    except scan_ok.FewTicks as few:
        findings += [Finding(REJECT, f"execution phase {p.first + i}: {UNCOMPUTED}: {few}")
                     for i in range(len(c.windows))]
    else:
        findings += [Finding(REJECT, f"execution phase {p.first + i}: {UNOK}") for i, reg in enumerate(c.windows)
                     if not scan_ok.held(consts, reg)]
    findings += [Finding(REJECT, f"execution phase {p.first + i}: {UNHELD}") for i, reg in enumerate(c.windows)
                 if not _held(p, c.audits, reg)]
    if rejected := next((f for f in findings if f.verdict == REJECT), None):
        return REJECT, rejected.reason
    if findings:
        return UNDECIDED, findings[0].reason
    beta = [p.scan.entries if any(i in charged and charged[i].wrong >= 1 for i in _tied_audits(c.audits, reg)) else 0
            for reg in c.windows]
    return ACCEPT, public(p, facts, [(f, t, b) for (f, t), b in zip(scans, beta)], list(charged.values()),
                          0 if p.first == 0 else p.scan.start), {"link": p.link, "windows": p.windows}


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
    "rows-honest": ACCEPT, "rows-second-key": ACCEPT, "rows-two-stretches": ACCEPT, "rows-two-signers": REJECT,
    "rows-two-sessions": ACCEPT, "rows-restart": REJECT, "rows-start-not-signed": REJECT,
    "rows-start-uncertified": REJECT, "rows-start-count": REJECT, "rows-start-skipped": REJECT,
    "rows-tied-partial": ACCEPT, "rows-tied-sampled": ACCEPT,
    "rows-tied-entries-sampled": ACCEPT, "rows-tied-slots-sampled": ACCEPT, "rows-answer-tied-held": ACCEPT,
    "rows-program": REJECT,
    "rows-cert-missing": REJECT, "rows-cert-swap": REJECT, "rows-cert-fails-link": REJECT,
    "rows-cert-direction": REJECT, "rows-cert-forged": REJECT, "rows-cert-window": REJECT, "rows-cert-session": REJECT,
    "rows-cert-key": REJECT, "rows-cert-unlisted": REJECT, "rows-cert-other-session": REJECT,
    "rows-cert-outside-stretch": REJECT, "rows-cert-period-number": REJECT,
    "rows-past-slots": REJECT,
    "rows-reg-count": REJECT, "rows-reg-swap": REJECT, "rows-reg-forged": REJECT, "rows-reg-window": REJECT,
    "rows-reg-link": REJECT, "rows-reg-session": REJECT, "rows-reg-key": REJECT, "rows-reg-unlisted": REJECT,
    "rows-reg-other-session": REJECT, "rows-reg-outside-stretch": REJECT, "rows-reg-period-number": REJECT,
    "rows-reg-field": REJECT,
    "rows-schedule": REJECT, "rows-capped": ACCEPT, "rows-cap-schedule": REJECT,
    "rows-answer-protocol": REJECT, "rows-answer-other": REJECT, "rows-answer-window": REJECT,
    "rows-answer-reject": REJECT, "rows-answer-undecided": UNDECIDED, "rows-answer-profile": REJECT,
    "rows-answer-bound": REJECT, "rows-answer-oversampled": REJECT, "rows-answer-tied-unheld": REJECT,
    "rows-answer-unsampled": REJECT, "rows-answer-unregistered": REJECT, "rows-delta": REJECT, "rows-units": REJECT,
    "rows-prog-untied": REJECT, "rows-prog-unwired": REJECT, "rows-state-untied": REJECT, "rows-state-misindexed": REJECT,
    "rows-start-unpinned": REJECT, "rows-start-carried": REJECT, "rows-start-charged": ACCEPT, "rows-circuit-swapped": REJECT,
    "rows-ok-unheld": REJECT, "rows-ok-equal": ACCEPT, "rows-one-tick": REJECT,
    "rows-audit-other": REJECT, "rows-audit-reject": REJECT, "rows-audit-undecided": UNDECIDED,
    "rows-audit-profile": REJECT, "rows-audit-bound": REJECT, "rows-audit-program": REJECT, "rows-audit-units": REJECT,
    "rows-audit-oversampled": REJECT, "rows-audit-unsampled": REJECT, "rows-audit-unregistered": REJECT,
    "rows-tied-oversampled": REJECT, "rows-tied-stratum-oversampled": REJECT, "rows-tied-strata-sizes": REJECT,
    "rows-undecided-and-reject": REJECT, "rows-form": REJECT,
    "rows-first": REJECT, "rows-count-windows": REJECT, "rows-count-audits": REJECT, "rows-evidence-count": REJECT,
    "rows-params": REJECT, "rows-commitment": REJECT, "rows-evidence": REJECT, "case": REJECT,
}
MUTATIONS = list(EXPECT)
#: The changes to which entries the parameters list and which of them sign (`rows-…`). Accepted: the
#: signer's entry the second of two, and two entries whose stretches each hold some of the period's execution phases,
#: each execution phase signed by its own entry at its own number. Refused: two listed entries that sign in turn, each
#: in its own session at its own number, in a period the second's stretch holds; a certificate or a rows registration
#: under a key that isn't listed, under the picked key in the other entry's session, under the other listed entry,
#: outside its stretch, and at the period's number where the session's belongs. The check reads each session's
#: signed start: accepted, one key listed in two sessions, the period's one execution phase in the second's stretch, the
#: second starting where the two sessions' signed bases put it (`two-sessions`); refused, that list with the second
#: session's base another (`start-not-signed`), or its ordinal not the first's next (`start-skipped`), and a period of
#: two execution phases, one in each session's stretch (`restart`): a period is one execution phase, so one spanning a
#: restart is refused (`startCarriedRefusal`). `case` aims at none (`AIMS`), so its period lists the one entry that
#: signs it.
LISTS = ("second-key", "two-stretches", "two-signers", "cert-unlisted", "reg-unlisted", "cert-other-session",
         "reg-other-session", "cert-outside-stretch", "reg-outside-stretch", "cert-period-number", "reg-period-number",
         "two-sessions", "restart", "start-not-signed", "start-skipped")
#: The changes whose parameters list the signer's key in two sessions, the period's one execution phase in the second's
#: stretch, the signer's; `restart` lists it so too, its period two execution phases, one in each stretch
RESTARTS = ("two-sessions", "start-not-signed", "start-skipped")
#: How a shown start fails to be its session's certified start (`start-uncertified`), by `CertifiedStart.refusal`'s
#: code: signed under the link's own name, as an ingress certificate or for window 1; by a key that isn't listed;
#: under another schedule's digest; shown with another base than the one signed, or naming the period's schedule over
#: a digest signed for another; and with a signature changed, or signed in another session
UNCERTIFIED_BY = {"start": ("link", "direction", "window"), "key": ("key",), "schedule": ("schedule",),
                  "opening": ("base", "ordinal", "digest"), "signature": ("forged", "session")}
#: The changes whose parameters list two entries, the period's execution phases all in the second's stretch, the
#: signer's: its start is at least 1, so the session's numbers are not the period's
SECOND = ("second-key", "two-signers", "cert-other-session", "reg-other-session", "cert-outside-stretch",
          "reg-outside-stretch", "cert-period-number", "reg-period-number")


@dataclasses.dataclass(frozen=True)
class Stretch:
    """An entry the generator lists: a certifier session's private key, its session, and its stretch's start."""
    key: lms.PrivateKey
    session: bytes
    start: int

    def json(self) -> dict:
        return {"key": lms.public_key(self.key).hex(), "session": self.session.hex(), "start": self.start}


def _flip(b: bytes, i: int) -> bytes:
    i %= len(b)
    return b[:i] + bytes([b[i] ^ 1]) + b[i + 1:]


def _cert_json(c: Certificate) -> dict:
    return {"digest": c.digest.hex(), "direction": c.direction.value, "key_id": c.key_id.hex(), "link": c.link,
            "signature": c.signature.hex(), "window": c.window}


def _accepted(digest: bytes, program: str, query: str, size: int, k: int, delta: str, bound: int,
              strata: list[dict] | None = None) -> dict:
    """One-stage's answer accepting `k` of `size` units sampled `subset:k`, or sampled stratified over `strata` (each
    `{k, name, size}`, k and size their sums), as `OneStage.check`'s public and profile have it."""
    units = list(range(k))
    sample = {"subset": k} if strata is None else {"law": "stratified", "strata": strata}
    return {"commitment": digest.hex(),
            "profile": {"covered": [], "delta": delta, "drawn": {UNIT: k},
                        "levels": [{"draw": sample, "name": UNIT, "query": query, "size": size}],
                        "linked": False, "prescribed": [], "programs": [program]},
            "protocol": ONE_STAGE, "public": {"bound": bound, "drawn": units, "per_template": None, "proved": units},
            "reason": None, "verdict": ACCEPT}


def _refused(answer: dict, verdict: str) -> dict:
    return answer | {"profile": None, "public": None, "reason": f"the audit is {verdict}", "verdict": verdict}


#: The schedule's fields that leave its rows' shape (B, S, T) as it is: a change to one changes only its digest.
SAME_SHAPE = ("bucket_ms", "delay", "egress_cells", "ingress_buckets", "ingress_cells", "queue")
#: The orders of an execution phase's three records' certificates other than their own.
SWAPS = ((0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


def schedule_json(s: NetworkCertifierParams) -> dict:
    """The schedule as the parameters hold it: its integer fields, and each optional field it sets."""
    optional = {k: getattr(s, k) for k in OPTIONAL_SCHEDULE_KEYS if getattr(s, k) is not None}
    if s.content is not None:
        optional["content"] = [list(c) for c in s.content]
    return {k: getattr(s, k) for k in SCHEDULE_KEYS} | optional


#: The least cell of a schedule with a response cap: its payload holds an id request's `max_tokens`.
CAPPED_CELL = HEADER.size + MAX_TOKENS_AT + ID_DIGITS
#: The least cell of a schedule with lengths rows: its payload holds an id request's head and first prompt id
#: (`firstPromptEnd`).
LENGTHS_CELL = HEADER.size + io_request.layout(1 << 12).start + ID_FIELD_BYTES


def _small_schedule(r: random.Random, small: bool, capped: bool = False, T: int | None = None) -> NetworkCertifierParams:
    """A schedule of small rows: B just past a cell's header, S = lanes · slots at most 4, T 2 or 3 (2 if
    `small`: the least a scan has, `scan_ok.constants`) or the `T` asked for, and a timestep `every` that divides T;
    `capped`: with a response cap, B just past the least cell that holds an id request's `max_tokens`
    (`CAPPED_CELL`)."""
    T = r.randint(2, 2 if small else 3) if T is None else T
    s = NetworkCertifierParams(cell_bytes=r.choice([11, 12, 16]), slots=r.randint(1, 2), bucket_ms=r.randint(1, 200),
                     delay=r.randint(1, 4), lanes=r.randint(1, 2), queue=r.randint(1, 8), window=T,
                     ingress_cells=r.randint(1, 8), ingress_buckets=r.randint(1, 4), egress_cells=r.randint(1, 16),
                     every=r.choice([d for d in range(1, T + 1) if T % d == 0]))
    if not capped:
        return s
    egress = r.randint(2, 16)
    return dataclasses.replace(s, cell_bytes=r.choice([CAPPED_CELL, CAPPED_CELL + 1, CAPPED_CELL + 5]),
                               egress_cells=egress, max_tokens_cap=r.randint(1, egress - 1))


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


def _start_sig(S):
    return tuple(zip(S.NAMES, (Array(S.STATE, BIT),))), Tuple(Array(S.OUT, BIT))


def _ingress_sig(S):
    progs = (Array(S.PROG, BIT),) * (len(S.NAMES) - 2)
    return tuple(zip(S.NAMES, (Array(S.ROW, BYTE_BITS), Array(S.STATE, BIT), *progs))), _ingress_ret(S)


def _egress_sig(S):
    return tuple(zip(S.NAMES, (Array(S.WORDS, U16_BITS),))), Tuple(U16_BITS, Array(S.ROW, BYTE_BITS))


def _start_unit(B, S, sin):
    return tuple_of(B.batch(MIX, sin[0], sin[0], axes=(None, None), n=S.OUT))


def _ingress(B, S, row, sin, *progs):
    b, p = row[0][0], progs[0][0]
    return tuple_of(B.call(bind(SPREAD, N=OUT_BITS), b, p), B.batch(MIX, sin, b, axes=(0, None)),
                    B.batch(MIX, p, b, axes=(None, None), n=S.VIEW))


def _egress(B, S, state):
    s = state[0][0]
    return tuple_of(B.call(bind(SPREAD, N=OUT_BITS), s, s),
                    B.batch(bind(SPREAD, N=8), s, s, axes=(None, None), n=S.ROW))


#: Stand-ins of the scan's templates in the generator form, with their parameters and returns as `IngressScanStart`,
#: `IngressScanBucket` and `EgressBucket` have them at small widths, in the Boolean profile's types: each unit computes
#: and returns its rows. The start unit takes a state of `STATE` bits and returns one of `OUT`.
_STATICS = ("NAMES", "ROW", "STATE", "PROG", "VIEW")
START_UNIT = CompositeDefinition("NetworkPeriodScanStart", 0, ("NAMES", "STATE", "OUT"), _start_sig, _start_unit,
                                 register=False)
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
    """The scan's root in the generator form: each `(template, n, fixed)` of `members` one batch over its units, in
    order, its first `fixed` parameters inputs of its own and each later one, lane j's `prog<j>`, at bucket t (the
    first unit that reads slot entries bucket 0, each next such unit the next bucket) row `⌊t / every⌋` (the last
    row past the rows), or `skew` rows on, cyclically, of lane j's input `prog<j>` of the slot entries, `nb` rows of
    `prog` bits (lane 0's for every lane, `shared`). A template that reads none (the start unit's, the egress
    buckets') counts no bucket."""
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
            at, start = at + fixed, start + (n if reads else 0)
        return coll_concat(signature(S)[1], outs)

    defn = CompositeDefinition("NetworkPeriodScan", 0, ("templates", "counts", "lanes", "nb", "every", "skew", "shared"),
                               signature, body, register=False)
    return Program(bind(defn, templates=tuple(t.id for t, _, _ in members), counts=tuple(n for _, n, _ in members),
                        lanes=lanes, nb=nb, every=every, skew=skew, shared=shared))


@lru_cache(maxsize=None)
def scan_descriptor(T: int, row: int, widths: tuple[int, int, int, int], counts: tuple[int, ...] | None = None,
                    names: tuple[tuple[str, ...], ...] | None = None, checker: bool = False, lanes: int = 1,
                    every: int = 1, skew: int = 0, shared: bool = False, start: tuple[int, int] | None = None) -> str:
    """The canonical descriptor of a scan's Program in the generator form: the start unit's stand-in instance, `T`
    ingress instances and `T` egress instances (or `counts`, a template of no instance left out), the ingress
    template's row `row` bytes, the ingress buckets' state, entry and view bits and the egress bucket's state words
    `widths` (in the generator form the view is 16 bits a state word, the egress bucket reading it as u16 words), the
    start unit taking and returning a state of those bits (or of `start`'s two counts), and `lanes` entry ports a
    bucket, bucket t's `prog<j>` row `⌊t / every⌋` of lane j's input `prog<j>` of the slot entries (`_scan_root`), or
    `skew` rows on, cyclically, or lane 0's input (`shared`); `checker`: in the checker form instead, `T` ingress and
    `T` egress instances (or `counts`) that take their rows."""
    state, prog, view, words = widths
    ingress = {"ROW": row, "STATE": state, "PROG": prog, "VIEW": view}
    names = names or (START_PARAMS, ingress_params(lanes), EGRESS_PARAMS)
    if checker:
        templates = (bind(CHECKER_INGRESS, NAMES=CHECKER_PARAMS[0], **ingress),
                     bind(CHECKER_EGRESS, NAMES=CHECKER_PARAMS[1], WORDS=words, ROW=row))
        program = population.mixed_population_program([(t, n) for t, n in zip(templates, counts or (T, T)) if n])
        return D.canonical(codec.encode_program(program))
    took, gave = start or (state, state)
    templates = (bind(START_UNIT, NAMES=names[0], STATE=took, OUT=gave), bind(INGRESS_BUCKET, NAMES=names[1], **ingress),
                 bind(EGRESS_BUCKET, NAMES=names[2], WORDS=words, ROW=row))
    kept = [(t, n, f) for t, n, f in zip(templates, counts or (1, T, T), (1, 2, len(names[2]))) if n]
    return D.canonical(codec.encode_program(_scan_root(kept, lanes, T // every, every, prog, skew, shared)))


def _unit_sig(S):
    return (("x", Array(1, U16)),), Array(S.OUT, BYTE)


#: A compute audit's stand-in unit: it commits `OUT` bytes, a proof block's 16-bit public output (`OUT` = 2) or more or
#: fewer, each one gate.
AUDIT_UNIT = CompositeDefinition("NetworkPeriodAuditUnit", 0, ("OUT",), _unit_sig,
                                 lambda B, S, x: B.batch(LOW, x[0], axes=(None,), n=S.OUT), register=False)
#: The stand-in unit's widths in bytes: one under, at and over the public output's 16 bits.
UNIT_BYTES = (1, 2, 5, 40)
#: The slot entries' template of the cases' audits: `io_shell.SlotEntry_v1` at its smallest circuit-check row, a pinned
#: Definition (`SLOT_ENTRY_PINS`' first), whose population of one member's descriptor is `fixtures/slot_entry.json`.
SLOT_KEY = "SlotEntry_v1{KO=2,IDB=17,NP=4,STOP=[]}"
SLOT_BINDING = {"KO": 2, "IDB": 17, "NP": 4, "STOP": ()}
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


def templates_of(desc: dict) -> list[tuple[str, str, int]]:
    """A Program's root's templates, in order, each its descriptor id, its Definition's id and its members: one-stage's
    strata, one a template, named by descriptor id."""
    defs, callees = desc["definitions"], desc["callees"]
    return [(k, d["id"], _members(x)) for x in defs[desc["root"]]["body"]["nodes"]
            if (d := defs[k := callees[x["fn"]]])["form"] != "primitive"]


#: How `rows-prog-untied` breaks its execution phase's tie of the scan's `prog` to an audit's output, at every lane: the
#: scan's `scan/prog<j>` entries at another root or domain, the audit's chunks' entries at ports that are no unit's
#: output nor a chunk's entries port, no `scan/prog<j>` entry or each under another name (`scan/prog` among them), the
#: chunks' entries in the scan's own registration and in no audit's, or no entry's root a string
UNTIES = ("root", "domain", "port", "missing", "renamed", "scan", "untyped")
#: How `rows-prog-unwired` breaks its execution phase's wiring at one lane j while its tie holds: the audit's chunk j
#: entry at an output port (another unit's `o1`, or the chunk's own), at another `SlotEntry_v1` or `SelectEntry_v1`
#: template's entries port than any of its Program's, at another chunk's (`chunk`), of other than T / every leaves
#: (`leaves`), the scan's lane j entry named another lane's (`lane`), its Program's slot entries' template one member
#: off a slot each, or that template under its name and descriptor id with another body (`RENAMED_SLOT_ENTRY`,
#: `RENAMED_SELECT_ENTRY`)
UNWIRES = ("output", "key", "chunk", "leaves", "lane", "members", "body")
#: How `rows-state-untied` breaks its execution phase's tie of the scan's state to its wired audit's `SCAN_STATE` while
#: its
#: `prog` stays wired: the scan's state entry at another root or domain (a state tree that skips or alters one of the
#: scan's transitions), no state entry or one under another name, at another audit's `SCAN_STATE`, its root no string,
#: or a second state entry at another root; the wired audit's `SCAN_STATE` absent, at another root, beside a second, or
#: held in its window's `activations.values` alone (`audit-activations`), as a window circuit's registration commits
#: its per-window values (`{"program", "values": {value: entry}}`), under its own name and at the scan's state's root
#: and domain: no entry of its `roots` or its `registered`, the two places `stateEntries` reads
UNSTATES = ("root", "domain", "missing", "renamed", "other", "untyped", "extra", "audit", "audit-root", "audit-extra",
            "audit-activations")
#: the program a tied audit's window's `activations` names
WINDOW_PROGRAM = "stand-in/window"
#: How `rows-circuit-swapped` breaks the class circuit's hold while every tie, wiring and state holds: the aimed
#: execution phase's tied audit registered under a Program of one more unit, its answer and population that Program's
#: (`audit`), or the parameters' `circuit` another digest: of no Program, the scan's, or an untied audit's (`params`)
UNHOLDS = ("audit", "params")
#: How `rows-state-misindexed` puts the state reads of its transfer blocks' scan off the chain while its state stays
#: tied by root and domain: bucket t's `sin` at leaf t + 1 (the leaf its own `o1` registers), its `o1` at leaf t (over
#: the state before it) or t + 2, the start unit's `o0` at leaf 1, a run over other units, the two buckets' ports' runs
#: swapped, the reads or a port's run missing or under another name, a field no natural, or the reads of a scan with no
#: start unit, bucket 0 unit 0 and nothing registered at leaf 0 (`older`)
MISREADS = ("sin-leaf", "o1-leaf", "o0-leaf", "units", "swapped", "missing", "untyped", "older")
#: How `rows-start-unpinned` leaves the start of its transfer blocks' scan unheld while its state's reads and tie hold:
#: the start entry's root a row the prover chose, opened as committed; the opening missing or under another name, its
#: salt another, its words another count; no start entry, one under another name or key, one in `roots` and none in
#: `registered`, a second at another root; or a field of another type
UNPINS = ("row", "missing", "salt", "words", "entry", "renamed", "roots", "extra", "untyped")
#: How `rows-start-carried` makes its period more than one execution phase: more after the run's first, or more after
#: a later one
CARRIES = ("later", "more")
#: How `rows-start-charged` states the start of a period whose one execution phase is not the run's first, which the
#: check holds to nothing and charges: the empty state's opening (None), or one of `UNPINS` that leaves the scan's state
#: entries as they are
CHARGES = (None, "row", "missing", "salt", "words", "entry", "extra")
#: What `rows-answer-tied-held` puts under `TIED_KEY` in its execution phase's scan's answer, each a list of lanes'
#: names: every lane's in the lanes' order, all but the last lane's (a proper subset), the lanes' in reverse, the first
#: lane's twice after them, and none
TIED_HELD = ("all", "subset", "order", "twice", "empty")
#: What `rows-answer-tied-unheld` puts there, each naming a tie the check does not wire or no list of names: after the
#: lanes' names the name past the last lane's (``prog<lanes>``), another Input's (``state``) or the first lane's
#: registration entry in place of its parameter (``scan/prog0``); ``state`` before them, or alone; a lane's name
#: without its lane, alone (``prog``); a lane's name as a string, in a nested list, or beside a number; and null
TIED_UNHELD = ("next", "state", "entry", "first", "alone", "bare", "string", "nested", "number", "null")


def tied_list(kind: str, lanes: int):
    """What a change of `TIED_HELD` or `TIED_UNHELD` puts under `TIED_KEY`, for a scan of `lanes` lanes."""
    names = list(prog_params(lanes))
    return {"all": names, "subset": names[:-1], "order": names[::-1], "twice": [*names, names[0]], "empty": [],
            "next": [*names, f"prog{lanes}"], "state": [*names, "state"], "entry": [*names, prog_name(0)],
            "first": ["state", *names], "alone": ["state"], "bare": ["prog"], "string": names[0],
            "nested": [names], "number": [*names, 0], "null": None}[kind]


#: How `rows-ok-unheld` breaks its execution phase's scan's ok bits while every other check holds: one member's
#: `<member>/o0` entry at another root (a tree with a row that is not the ok row, or under other salts), domain or
#: leaf count; no entry for it, or for any member; its entry under another port; a second entry for it at another
#: root, or an entry at a port ending `/o0` that is no member's; every member's at another schedule's constants; its
#: root or domain no string; the entries by value under `registered`, which the period reads no ok bit from; or the
#: members of a group of two execution phases' scan (`PAIR`), each at the root of its ok rows
UNOKS = ("root", "domain", "leaves", "missing", "renamed", "extra", "other", "untyped", "registered", "group")
#: The T `rows-ok-equal` runs at: where two members, `ingress-0` and `last`, are one proof block of one template
#: each, so their ok bits are at one domain and one root
EQUAL_T = 2
#: The T `rows-one-tick` runs at: `ingress-0` would be a member of no proof block, so it has no constants
ONE_TICK = 1


def PAIR(T: int) -> tuple[tuple[str, str, int], ...]:
    """The members with an ok bit (name, template, proof blocks) of one scan over a group of two execution phases of
    `T` ticks: `ingress-0`, buckets 0 … T − 1, and `ingress-1`, buckets T … 2T − 2; `last`, bucket 2T − 1; and
    `egress-0` and `egress-1`, T buckets each. The rows form reads one execution phase a registration, so such a
    registration is refused: `ingress-0` is over T buckets, not T − 1, and `ingress-1` and `egress-1` are no member
    of one execution phase's scan."""
    return (("ingress-0", "ingress", T), ("ingress-1", "ingress", T - 1), ("last", "ingress", 1),
            ("egress-0", "egress", T), ("egress-1", "egress", T))


def _present(e: dict) -> dict:
    return {k: v for k, v in e.items() if v is not None}


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


def _state_tree(rp: random.Random, T: int) -> ADV.Committed:
    """A scan's state tree of T rows of one to three u16 words, any, under random salts and binding."""
    w = rp.randint(1, 3)
    rows = [rp.randbytes(2 * w) for _ in range(T)]
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
    port = rp.choice(["o0", "sin", "o1"])
    if how == "sin-leaf":
        s["sin"]["leaf"] = 1
    elif how == "o1-leaf":
        s["o1"]["leaf"] = rp.choice([0, 2])
    elif how == "o0-leaf":
        s["o0"]["leaf"] = 1
    elif how == "units":
        lo, hi = s[port]["units"]
        s[port]["units"] = rp.choice([[lo, hi + 1], [lo + 1, hi + 1], [hi, lo]])
    elif how == "swapped":
        s["sin"], s["o1"] = s["o1"], s["sin"]
    elif how == "missing":
        return rp.choice([None, {}, {"scan/state2": s}, {STATE: {k: v for k, v in s.items() if k != port}},
                          {STATE: {k if k != port else port.upper(): v for k, v in s.items()}}])
    elif how == "older":
        return {STATE: {"o1": {"leaf": 1, "units": [0, T - 1]}, "sin": {"leaf": 1, "units": [1, T]}}}
    else:
        s[port] = rp.choice([s[port] | {"leaf": rp.choice([str(s[port]["leaf"]), True, -1, None])},
                             s[port] | {"units": rp.choice([s[port]["units"][:1], s[port]["units"] + [0],
                                                            {"lo": s[port]["units"][0]}, str(s[port]["units"])])}])
    return reads


def _unpinned(rp: random.Random, words: int, salt: bytes, domain: bytes,
              how: str) -> tuple[list[dict], list[dict], dict | None]:
    """A scan's registration's start unheld, as `how`, one of `UNPINS`, says: its `registered` entries, its `roots`
    entries and its `window`'s ``empty`` (None for none). A start entry whose row the prover chose (``row``) is
    opened as committed."""
    row = bytes([rp.randrange(1, 256)]) + rp.randbytes(2 * words - 1) if how == "row" else None
    entry, e = start_empty(words, salt, domain, row)
    o = e[START]
    if how == "missing":
        return [entry], [], rp.choice([None, {}, {"0.sin2": o}, {STATE: o}])
    if how == "salt":
        o["salt"] = _flip_hex(rp, o["salt"])
    elif how == "words":
        o["words"] = rp.choice([words + 1, 0] + ([words - 1] if words > 1 else []))
    elif how == "entry":
        return [], [], e
    elif how == "renamed":
        entry = rp.choice([{("port" if k == "value" else k): v for k, v in entry.items()},
                           entry | {"value": rp.choice(["1.sin", "0.sin/", "0.Sin", "sin", "0.sin0", STATE])}])
    elif how == "roots":
        return [], [{("port" if k == "value" else k): v for k, v in entry.items()}], e
    elif how == "extra":
        return [entry, entry | {"root": rp.randbytes(64).hex()}], [], e
    elif how == "untyped":
        if rp.random() < 0.5:
            o |= rp.choice([{"salt": o["salt"].upper()} if any(c in "abcdef" for c in o["salt"]) else {"salt": 7},
                            {"words": rp.choice([str(words), True, -1, None])}, {"salt": None}])
        else:
            entry = entry | rp.choice([{"root": rp.choice([None, 7, entry["root"][:-1]])},
                                       {"domain": rp.choice([None, False, "g" + entry["domain"][1:]])}])
    return [entry], [], e


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


def _ok_entries(ro: random.Random, T: int, unok: str | None) -> tuple[str, list[dict]]:
    """The scan's registration's entries for its members' ok bits, and the list they go in: each member's
    `<member>/o0` in `roots` at the schedule's constant (`scan_ok.constants`), broken as `unok` says, and at times
    the start unit's `start/o0` (`scan_ok.START`), a state and no ok bit, at any root. A schedule of one tick has no
    constants: its registration names those of two ticks."""
    es = [{"domain": c.domain, "leaves": c.leaves, "port": c.port, "root": c.root}
          for c in scan_ok.constants(max(T, 2))]
    hit = ro.randrange(len(es))
    e, member = es[hit], es[hit]["port"].removesuffix("/o0")
    if unok == "root":
        e["root"] = ro.choice([ro.randbytes(64).hex(), e["root"].upper(), e["root"][:-2], e["domain"]])
    elif unok == "domain":
        e["domain"] = ro.choice([ro.randbytes(64).hex(), e["domain"].upper(), e["root"]])
    elif unok == "leaves":
        e["leaves"] = ro.choice([e["leaves"] + 1, e["leaves"] - 1, -1, None, str(e["leaves"]),
                                 *([True] if e["leaves"] == 1 else [])])
    elif unok == "missing":
        es = ro.choice([es[:hit] + es[hit + 1:], []])
    elif unok == "renamed":
        e["port"] = ro.choice([f"{member}/o1", f"{member}/O0", f"{member}/o0/", f"{member}x/o0", f"{member}/o00",
                               "o0", member, 7, None])
    elif unok == "extra":
        es.append(ro.choice([e | {"root": ro.randbytes(64).hex()}, e | {"port": "other/o0"},
                             e | {"port": entries_port(SLOT_KEY)}]))
    elif unok == "other":
        es = [{"domain": c.domain, "leaves": c.leaves, "port": c.port, "root": c.root}
              for c in scan_ok.constants(T + ro.choice([1, 2]))]
    elif unok == "untyped":
        e[ro.choice(["root", "domain"])] = ro.choice([None, 7, False])
    elif unok == "registered":
        return "registered", [{("value" if k == "port" else k): v for k, v in x.items()} for x in es]
    elif unok == "group":
        es = [{"domain": c.domain, "leaves": c.leaves, "port": c.port, "root": c.root}
              for c in (scan_ok.constant(*m) for m in PAIR(T))]
    if ro.random() < 0.5:
        es.append({"domain": ro.randbytes(64).hex(), "leaves": 1, "port": scan_ok.START, "root": ro.randbytes(64).hex()})
    ro.shuffle(es)
    return "roots", [_present(x) for x in es]


def _output_entries(rp: random.Random, i: int, ties: dict, nb: int, aimed: int, untie: str | None,
                    unwire: str | None = None, hit: int = 0) -> list[dict]:
    """Audit `i`'s registration's `roots`: for each execution phase's `prog` tied to it, each lane's chunk's entries
    port its tie names at the chunk's root and domain (`nb` leaves, a timestep's row each), and an input port's entry at
    another root; at execution phase `aimed`, every chunk's broken as `untie` says, or lane `hit`'s as `unwire` says."""
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
    """A tied audit's stratified sampling design and its profile's strata, a stratum a template of its Program (`templates_of`):
    its slot entries' and scan slots' templates sampled whole and its first other template in part (`tied-partial`), or
    with its slot entries' (`tied-entries-sampled`) or its scan slots' (`tied-slots-sampled`) template in part too."""
    ts = templates_of(desc)
    whole, broken = (*ENTRY_IDS, SCAN_SLOTS), {"tied-entries-sampled": ENTRY_IDS, "tied-slots-sampled": (SCAN_SLOTS,)}
    fewer = lambda n: r.randint(min(1, n - 1), n - 1)  # noqa: E731
    ks = [fewer(n) if d in broken.get(m, ()) else n if d in whole else r.randint(1, n) for _, d, n in ts]
    j = next(j for j, (_, d, _) in enumerate(ts) if d not in whole)
    ks[j] = fewer(ts[j][2])
    lo, sampling_design, strata = 0, [], []
    for (key, _, n), k in zip(ts, ks):
        sampling_design.append({"k": k, "template": key, "units": [[lo, lo + n]]})
        strata.append({"k": k, "name": key, "size": n})
        lo += n
    return {"k": sum(ks), "law": "stratified", "strata": sampling_design}, strata


def _oversample(r: random.Random, size: int, k: int, strata: list[dict] | None) -> int:
    """A profile's `k` sampled of a level of `size` raised past the level, to at most three more than it holds, one of
    its `strata` taking the excess: what no sample of the level gives (`answer-oversampled`, `audit-oversampled`,
    `tied-oversampled`)."""
    over = size + r.randint(1, 3) - k
    if strata:
        r.choice(strata)["k"] += over
    return k + over


def _resample(r: random.Random, size: int, k: int, strata: list[dict] | None) -> tuple[int, list[dict] | None]:
    """A profile's sample of a level of `size` units, `k` of them sampled `subset:k` or over `strata`, changed to
    another sample of the same level, as `(k, strata)`: one that samples no more than the level or a stratum holds,
    with strata the level's size, so that the period check refuses it only for not being its registration's
    (`answer-unregistered`, `audit-unregistered`). A subset sample at another `k`, or as one stratum; a stratified
    one with a stratum renamed, a stratum sampled otherwise, a unit moved from one stratum to another, or as the
    subset of its total."""
    if strata is None:
        if r.random() < 0.3:
            return k, [{"k": k, "name": UNIT, "size": size}]
        return r.choice([x for x in range(size + 1) if x != k]), None
    strata = [dict(s) for s in strata]
    ways = ["rename", "subset"]
    ways += ["sample"] if any(s["size"] for s in strata) else []
    ways += ["move"] if len(strata) > 1 and any(s["k"] < s["size"] for s in strata) else []
    how = r.choice(ways)
    if how == "subset":
        return k, None
    if how == "rename":
        r.choice(strata)["name"] += "x"
    elif how == "sample":
        s = r.choice([s for s in strata if s["size"]])
        s["k"] = r.choice([x for x in range(s["size"] + 1) if x != s["k"]])
    else:
        src = r.choice([s for s in strata if s["k"] < s["size"]])
        src["size"] -= 1
        r.choice([s for s in strata if s is not src])["size"] += 1
    return sum(s["k"] for s in strata), strata


def _unsample(r: random.Random, rec: dict) -> dict:
    """The registration record `rec` stating no sample (`answer-unsampled`, `audit-unsampled`): without its sampling
    design or its population, with a Bernoulli design, or with a design or a population one-stage's profile reads no
    sample of (a count that is no natural number, strata that are no list, a stratum without its template, its units
    or its count, or with an empty range)."""
    stated = rec["law"]
    kind, strata = stated["law"], stated.get("strata")
    others = [None, {"law": "bernoulli", "p": "1/2"}, {"law": "two-stage", "k": stated["k"]}, stated | {"law": None},
              {k: v for k, v in stated.items() if k != "law"}]
    if kind == "subset":
        others += [{"law": "subset"}, stated | {"k": -1}, stated | {"k": str(stated["k"])}, stated | {"k": True}]
    else:
        j = r.randrange(len(strata))
        one = strata[j]
        at = lambda s: stated | {"strata": strata[:j] + [s] + strata[j + 1:]}  # noqa: E731
        lo, hi = one["units"][0]
        others += [stated | {"strata": None}, stated | {"strata": {"0": one}},
                   at({k: v for k, v in one.items() if k != "template"}), at(one | {"template": 0}),
                   at(one | {"units": []}), at(one | {"units": [[hi, lo]]}), at(one | {"units": [[lo, lo]]}),
                   at(one | {"units": [lo, hi]}), at(one | {"units": [[lo, hi, hi]]}), at(one | {"k": -1}),
                   at({k: v for k, v in one.items() if k != "k"})]
    n = rec["population"]
    return r.choice([rec | {"law": x} for x in others]
                    + [{k: v for k, v in rec.items() if k != gone} for gone in ("law", "population")]
                    + [rec | {"population": x} for x in (None, -1, str(n), True, [n])])


def _stratum_oversample(r: random.Random, strata: list[dict]) -> None:
    """One of a profile's `strata` raised past its own size, by one to three, and the others lowered by as much, so the
    level's sample is what it was: what no sample of that stratum gives, in a level whose total still holds
    (`tied-stratum-oversampled`). It needs the others to sample more than that stratum is short of its size."""
    j = r.randrange(len(strata))
    rest = [s for i, s in enumerate(strata) if i != j]
    short = strata[j]["size"] - strata[j]["k"]
    over = short + r.randint(1, min(3, sum(s["k"] for s in rest) - short))
    strata[j]["k"] += over
    for s in rest:
        cut = min(over, s["k"])
        s["k"], over = s["k"] - cut, over - cut


def _strata_resize(r: random.Random, strata: list[dict], seed: int) -> int:
    """A profile's `strata`, each sampled whole, no longer the size of their level: one of them holding one to three
    more units than it samples, each holding twice as many, or one holding and sampling one fewer, the seeds taking
    the three in turn, so any three seeds in a row hold strata above their level's size and below it. No stratum
    samples more than it holds and the level's sample is no more than the level holds (`tied-strata-sizes`). The
    strata's sample."""
    how, one = ("more", "twice", "fewer")[seed % 3], r.choice(strata)
    if how == "more":
        one["size"] += r.randint(1, 3)
    elif how == "twice":
        for s in strata:
            s["size"] *= 2
    else:
        one["size"], one["k"] = one["size"] - 1, one["k"] - 1
    return sum(s["k"] for s in strata)


def build(m: str, seed: int, small: bool = False) -> dict:
    """A period's case with the change `m`, every choice sampled from `seed`; `small`: the toy key, as few execution phases
    and audits as `m` needs."""
    r = random.Random(f"network-period {m} {seed}")
    rp = random.Random(f"network-period prog {m} {seed}")
    rt = random.Random(f"network-period tied {m} {seed}")
    ro = random.Random(f"network-period ok {m} {seed}")
    m = m.removeprefix("rows-")
    ki = 0 if small else r.randrange(len(KEYS))
    key = _key(ki)
    N, W, H, toy = KEYS[ki]
    # the choices of the entries the parameters list (`listed`, below, once the period's execution phases are known),
    # and the signer's session
    rk = random.Random(f"network-period keys {m} {seed}")
    session = r.randbytes(lms.SESSION_BYTES)
    # every form binds the link in its window record's canonical JSON, whose strings are ASCII without DEL
    link = r.choice(["eth0", "a", "d-e"])
    carry = rp.choice(CARRIES) if m == "start-carried" else None
    least_windows = 2 if m in ("undecided-and-reject", "delta", "restart") or carry else 1
    least_audits = 2 if m == "state-untied" else \
        1 if m.startswith("audit") or m in ("undecided-and-reject", "count-audits") else 0
    # a rows period is one execution phase (`scanStartCheck`): more only where the change asks, refused, and a later one
    # than the run's first only where the change asks, charged its start
    n_windows = r.randint(least_windows, least_windows if small or not carry else 3)
    n_audits = r.randint(least_audits, max(least_audits, 1) if small else 2)
    # an execution phase's own audit, and at times one no execution phase ties, which may leave units unsampled
    n_audits = max(n_audits, n_windows + (0 if small else rp.randint(0, 1)))
    first = r.choice([0, 0, r.randrange(1, 6)])
    if (m in (*SECOND, *RESTARTS, "two-stretches", "start-charged") or carry == "later") and first == 0:
        first = rk.randrange(1, 6)
    elif m == "start-unpinned" or carry == "more":
        first = 0
    # the entries the parameters list, by their stretches' starts, the signer's (`key`, `session`) among them, its
    # numbers the period's less its start; seed 1's key is never listed. Two, the period's one execution phase the
    # second's stretch's first, the signer's (`two-stretches`); two of the signer's key, each stretch holding one of the
    # period's two execution phases (`restart`); two, the period in the second's stretch, the signer's (`SECOND`, and
    # `RESTARTS` under the signer's key); else the signer's, alone in a small period, or among up to two more whose
    # stretches hold none of the period's execution phases, before the signer's or after the period
    other = lambda k, start: Stretch(k, rk.randbytes(lms.SESSION_BYTES), start)  # noqa: E731
    if m == "two-stretches":
        listed = [other(_key(ki, 2), rk.randrange(first)), Stretch(key, session, first)]
    elif m == "restart":
        listed = [other(key, rk.randint(0, first)), Stretch(key, session, first + 1)]
    elif m in SECOND or m in RESTARTS:
        start = rk.randint(1, first)
        listed = [other(key if m in RESTARTS else _key(ki, 2), rk.randrange(start)), Stretch(key, session, start)]
    else:
        start = rk.randint(0, first)
        more = [] if small else [_key(ki, 2), _key(ki, 3)][:rk.randrange(3)]
        before = rk.randint(0, min(len(more), start))
        starts = rk.sample(range(start), before) + [first + n_windows + x for x in rk.sample(range(4), len(more) - before)]
        listed = sorted([Stretch(key, session, start), *map(other, more, starts)], key=lambda e: e.start)

    def stretch(w: int) -> Stretch:
        """The listed entry whose stretch holds the period's execution phase `w`."""
        return [e for e in listed if e.start <= w][-1]

    def number(w: int) -> int:
        """The number of the period's execution phase `w` in its entry's session: `w` less the entry's start."""
        return w - stretch(w).start
    sched = bound = _small_schedule(r, small, capped=m in ("capped", "cap-schedule"),
                                    T={"ok-equal": EQUAL_T, "one-tick": ONE_TICK}.get(m))
    # an egress bucket's state is its ingress bucket's view read as u16 words, so the view is 16 bits a word
    widths = (r.randint(1, 3), r.randint(1, 4), OUT_BITS * (words := r.randint(1, 2)), words)
    scan = json.loads(scan_descriptor(sched.window, WR.row_bytes(sched), widths, lanes=sched.lanes, every=sched.every))
    program = program_sha512(scan)
    delta = r.choice(DELTAS)
    t = r.randrange(n_windows)
    a = r.randrange(max(n_audits, 1))
    # one slot entries' template per audit: one-stage names a template's ports by its descriptor id; an audit that
    # selects has select entries and scan slots, as a window circuit's last position reads its slots' occupancy
    owners = rp.sample(range(n_audits), n_windows)
    select = rp.random() < 0.5 or m == "tied-slots-sampled"
    selects = set(owners) if select else set()
    lanes = sched.lanes
    ties = {w: {"audit": i, "key": (k := SELECT_KEY if i in selects else SLOT_KEY), "chunks": [
                {"port": entries_port(k, j), "root": rp.randbytes(64).hex(), "domain": rp.randbytes(64).hex()}
                for j in range(lanes)]}
            for w, i in zip(range(first, first + n_windows), owners)}
    untie = rp.choice(UNTIES) if m == "prog-untied" else None
    unwire = rp.choice(UNWIRES) if m == "prog-unwired" else None
    unstate = rp.choice(UNSTATES) if m == "state-untied" else None
    unpin = rp.choice(UNPINS) if m == "start-unpinned" else rp.choice(CHARGES) if m == "start-charged" else None
    unhold = rp.choice(UNHOLDS) if m == "circuit-swapped" else None
    # each audit's `SCAN_STATE`, the state tree of the transfer blocks tied to it
    trees = {i: _state_tree(rp, sched.window) for i in range(n_audits)}
    states = {i: {"root": x.root.hex(), "domain": ADV.domain(x.binding, len(x.rows)).domain_id.hex()}
              for i, x in trees.items()}
    unok = ro.choice(UNOKS) if m == "ok-unheld" else None
    if untie == "untyped":
        for c in ties[first + t]["chunks"]:
            c["root"] = rp.choice([None, 7, False])
    # the lane whose wiring the aimed execution phase's `unwire` breaks
    hit = rp.randrange(lanes) if lanes else 0
    # the changes that sample the aimed execution phase's audit stratified by its templates, its slot entries' and scan
    # slots' whole (`tied-partial`) or one of them in part
    by_template = m in ("tied-partial", "tied-entries-sampled", "tied-slots-sampled")
    if m in ("tied-sampled", "tied-oversampled", "tied-stratum-oversampled", "tied-strata-sizes") or by_template:
        a = ties[first + t]["audit"]
    tied = {x["audit"] for x in ties.values()}
    signed = iter(range(1 << 30))
    # the entry that signs the `n`-th signature for the period's execution phase `w`: the one whose stretch holds `w`,
    # or the two listed entries in turn (`two-signers`), the first outside its stretch
    turn = lambda n, w: listed[n % len(listed)] if m == "two-signers" else stretch(w)  # noqa: E731

    def sign(link: str, d: Direction, w: int, digest: bytes, k: lms.PrivateKey | None = None,
             s: bytes | None = None, n: int | None = None, e: Stretch | None = None,
             num: int | None = None) -> Certificate:
        """A certificate for the period's execution phase `w` under the entry `e`, or the one whose turn it is: under
        its key, or `k`, in its session, or `s`, naming that session's number for the execution phase, `w` less the
        entry's start, or `num`."""
        n = next(signed) if n is None else n
        e = turn(n, w) if e is None else e
        k = e.key if k is None else k
        c = Certificate(k.I, link, d, w - e.start if num is None else num, digest, b"")
        s = e.session if s is None else s
        return dataclasses.replace(c, signature=lms.sign_at(k, n % 2 ** k.params.H, c.message(s)).encode())

    def registrations(w: int, s: NetworkCertifierParams | None = None, on: str = link, ses: bytes | None = None,
                      e: Stretch | None = None, num: int | None = None) -> list[dict]:
        """Link `on`'s ingress and egress rows for the period's execution phase `w`, each committed under schedule `s`
        (the period's) by the entry `e`, or the one whose turn it is: in its session, or `ses`, at that session's number
        for the execution phase, or `num`, and registered, its head signed there."""
        s = s or sched
        out = []
        for d in (INGRESS, EGRESS):
            n = next(signed)
            by = turn(n, w) if e is None else e
            in_ses, at = by.session if ses is None else ses, w - by.start if num is None else num
            cells = WR.Cells.commit(on, d, at, [r.randbytes(WR.row_bytes(s)) for _ in range(s.window)],
                                    WR.Wiring((), r.randbytes(WR.WIRING_SALT_BYTES)), s, in_ses,
                                    [r.randbytes(WR.SALT_BYTES) for _ in range(s.window)])
            out.append(WR.registration(cells.head, sign(on, d, w, cells.head.digest(), s=in_ses, n=n, e=by, num=at), s))
        return out

    def resigned(doc: dict, w: int, **kw) -> dict:
        """The registration `doc` of the period's execution phase `w` with its head signed again, at the number it
        names."""
        c = sign(doc["link"], Direction(doc["direction"]), w, bytes.fromhex(doc["certificate"]["digest"]),
                 num=doc["window"], **kw)
        return doc | {"certificate": {"digest": c.digest.hex(), "key": c.key_id.hex(), "signature": c.signature.hex()}}

    window_regs, windows = [], []
    for w in range(first, first + n_windows):
        aimed = w == first + t
        i, j = r.randrange(2), r.randrange(3)
        regs = registrations(w)
        named = regs
        if aimed and m == "answer-window":
            there = registrations(w, num=number(w) + r.choice([1, -1] if number(w) else [1]))
            named = r.choice([there, [regs[0], there[1]], [there[0], regs[1]],
                              [x | {"domain": y["domain"]} for x, y in zip(regs, there)]])
        elif aimed and m == "schedule":
            other = dataclasses.replace(sched, **{f: getattr(sched, f) + 1 for f in [r.choice(SAME_SHAPE)]})
            if r.random() < 0.5:
                bound = other
            else:
                under = registrations(w, other)
                regs = r.choice([under, [under[0], regs[1]], [regs[0], under[1]]])
        elif aimed and m == "cap-schedule":
            cap = sched.max_tokens_cap
            other = dataclasses.replace(sched, max_tokens_cap=r.choice(
                [None, *(c for c in (cap - 1, cap + 1) if 0 < c < sched.egress_cells)]))
            if r.random() < 0.5:
                bound = other
            else:
                under = registrations(w, other)
                regs = r.choice([under, [under[0], regs[1]], [regs[0], under[1]]])
        elif aimed and m == "reg-count":
            regs = r.choice([regs[:1], regs[1:], [], regs + regs[1:]])
        elif aimed and m == "reg-swap":
            regs = regs[::-1]
        elif aimed and m in ("reg-forged", "reg-window", "reg-link", "reg-session", "reg-key", "reg-unlisted",
                             "reg-other-session", "reg-outside-stretch", "reg-period-number", "reg-field"):
            doc, c = regs[i], regs[i]["certificate"]
            sig, dig = bytes.fromhex(c["signature"]), bytes.fromhex(c["digest"])
            if m == "reg-forged":
                doc = r.choice([doc | {"certificate": c | {"signature": s.hex()}}
                                for s in (_flip(sig, seed), sig[:-1], b"")]
                               + [doc | {"certificate": c | {"digest": _flip(dig, seed).hex()}}]
                               + [doc | {k: r.randbytes(n).hex()} for k in ("root", "wiring") for n in (64, 63)])
            elif m == "reg-window":
                doc = registrations(w, num=number(w) + r.choice([1, -1] if number(w) else [1]))[i]
            elif m == "reg-link":
                doc = registrations(w, on=link + "x")[i]
            elif m == "reg-session":
                ses = _flip(session, seed)
                doc = resigned(doc, w, s=ses) if r.random() < 0.5 else registrations(w, ses=ses)[i]
            elif m == "reg-key":
                doc = resigned(doc, w, k=_key(ki, 1, r.random() < 0.5))
            elif m == "reg-unlisted":
                doc = resigned(doc, w, k=_key(ki, 1))
            elif m == "reg-other-session":
                # the picked entry's key in the other entry's session: its head alone signed there, or its rows
                # committed there
                ses = listed[0].session
                doc = resigned(doc, w, s=ses) if r.random() < 0.5 else registrations(w, ses=ses)[i]
            elif m == "reg-outside-stretch":
                # the other listed entry's, outside its stretch: the head alone signed by it, or the rows committed by
                # it in its session at its own number for the execution phase
                doc = resigned(doc, w, e=listed[0]) if r.random() < 0.5 else registrations(w, e=listed[0])[i]
            elif m == "reg-period-number":
                # the picked entry's, at the period's number for the execution phase, not its session's
                doc = registrations(w, num=w)[i]
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
                recs[j] = sign(on, d, w, recs[j].digest, num=number(w) + 1)
            elif m == "cert-session":
                recs[j] = sign(on, d, w, recs[j].digest, s=_flip(session, seed))
            elif m == "cert-key":
                recs[j] = sign(on, d, w, recs[j].digest, k=_key(ki, 1, r.random() < 0.5))
            elif m == "cert-unlisted":
                recs[j] = sign(on, d, w, recs[j].digest, k=_key(ki, 1))
            elif m == "cert-other-session":
                recs[j] = sign(on, d, w, recs[j].digest, s=listed[0].session)
            elif m == "cert-outside-stretch":
                recs[j] = sign(on, d, w, recs[j].digest, e=listed[0])
            elif m == "cert-period-number":
                recs[j] = sign(on, d, w, recs[j].digest, num=w)
        entries = [{"domain": x["domain"], "root": x["root"], "value": x["value"]} for x in named]
        entries.insert(r.randrange(len(entries) + 1),
                       {"domain": r.randbytes(64).hex(), "root": r.randbytes(64).hex(), "value": "decoy"})
        T = sched.window
        units = 2 * T + 1 + (r.choice([1, -1]) if aimed and m == "units" else 0)
        strata = [{"k": r.randint(1, n), "name": name, "size": n}
                  for name, n in (("start", 1), (INGRESS.value, T), (EGRESS.value, T))] \
            if r.random() < 0.5 and units == 2 * T + 1 else None
        k = sum(s["k"] for s in strata) if strata else r.randint(1, units)
        cuts = [0, 1, T + 1, 2 * T + 1]
        sampling_design = {"k": k, "law": "subset"} if strata is None else {"k": k, "law": "stratified", "strata": [
            {"k": s["k"], "template": s["name"], "units": [[lo, hi]]} for s, lo, hi in zip(strata, cuts, cuts[1:])]}
        if aimed and m == "answer-oversampled":
            k = _oversample(r, units, k, strata)
        elif aimed and m == "answer-unregistered":
            k, strata = _resample(r, units, k, strata)
        fmt, where = r.choice(FORMATS), r.choice(["roots", "registered"])
        entries += _prog_entries(rp, ties[w], where, untie if aimed else None, unwire == "lane" and aimed, hit)
        entries += _state_entries(rp, states, ties[w]["audit"], where, unstate if aimed else None)
        rp.shuffle(entries)
        reads = _misread(rp, T) if aimed and m == "state-misindexed" else state_reads(T)
        # the scan's start, the empty state at a domain of its own, which every execution phase here states but where
        # the change states another: a period's later one is refused, and one not the run's first charged, whatever it
        # states
        start = (rp.randint(1, 3), rp.randbytes(ADV.INST.salt_bytes), rp.randbytes(64))
        (by_value, by_port, empty) = _unpinned(rp, *start, unpin) if aimed and unpin else \
            ((got := start_empty(*start))[:1], [], got[1])
        window = {"kind": "stand-in", "window": w} | ({} if reads is None else {"reads": reads}) | (
            {} if empty is None else {"empty": empty})
        rec = {"format": fmt, "law": sampling_design, "partition": f"{link}/scan/{w}", "population": units, "program": program,
               "protocol": ONE_STAGE, where: entries, "window": window}
        there, oks = _ok_entries(ro, T, unok if aimed else None)
        for into, more in ((there, oks), ("registered", list(by_value)), ("roots", by_port)):
            if more:
                rec[into] = rec.get(into, []) + more
                ro.shuffle(rec[into])
        if aimed and m == "program":
            rec["program"] = r.choice([hashlib.sha512(b"another program").hexdigest(), program.upper(), None])
        elif aimed and m == "answer-unsampled":
            rec = _unsample(r, rec)
        digest = bytes.fromhex(R.digest(rec))
        answer = _accepted(digest, program, rec["partition"], units, k,
                           r.choice([d for d in DELTAS if d != delta]) if aimed and m == "delta" else delta,
                           r.randint(0, units), strata)
        if sched.every > 1:
            # every `every` units read one slot entry of each lane, which one-stage's registration check lets through
            # only tied, and its verdict then names each lane's
            answer["public"][TIED_KEY] = list(prog_params(sched.lanes))
        if aimed:
            if m == "answer-tied-held":
                answer["public"][TIED_KEY] = tied_list(rt.choice(TIED_HELD), sched.lanes)
            elif m == "answer-tied-unheld":
                answer["public"][TIED_KEY] = tied_list(rt.choice(TIED_UNHELD), sched.lanes)
            elif m == "answer-protocol":
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

    audit_params, audit_regs, audits, untied = [], [], [], []
    slots = sched.window // sched.every * sched.lanes
    # the class circuit, the one window circuit every tied audit runs: its units, its slot entries' template (whose
    # members `prog-unwired`'s `members` puts one off a slot each, and whose body its `body` renames) and scan slots
    off = (rp.choice([1, -1]) if slots > 1 else 1) if unwire == "members" else 0
    twos = m in ("tied-sampled", "tied-stratum-oversampled", "tied-strata-sizes") or by_template
    n_class = r.randint(2 if twos else 1, 40)
    class_split = r.randint(1, n_class - 1) if n_class > 1 and r.random() < 0.3 else None
    class_outs = r.sample(UNIT_BYTES, 1 if class_split is None else 2)
    class_slots = (rp.randint(1, 2),) if select else ()

    def shape_of(n: int, split: int | None, outs: list[int]) -> tuple[tuple[int, int], ...]:
        return ((n, outs[0]),) if split is None or split >= n else ((split, outs[0]), (n - split, outs[1]))

    circuit = program_sha512(json.loads(audit_descriptor(
        shape_of(n_class, class_split, class_outs), (slots + off,), unwire == "body", select, class_slots)))
    for i in range(n_audits):
        sampled = i == a and m == "tied-sampled"
        partial = i == a and by_template
        # the tied audit in two strata, each whole, then one past its size or their sizes no longer the level's
        halves = i == a and m in ("tied-stratum-oversampled", "tied-strata-sizes")
        in_class = i in tied
        entries = (slots + off,) if in_class else ()
        renamed = unwire == "body" and in_class
        scan_slots = class_slots if in_class else ()
        swapped = unhold == "audit" and i == ties[first + t]["audit"]
        units = (n_class + 1 if swapped else n_class) if in_class else \
            r.randint(2 if sampled or partial or halves else 1, 40)
        n_prog = units + (r.choice([1, -1]) if units > 1 else 1) if i == a and m == "audit-units" else units
        units += sum(entries) + sum(scan_slots)
        if in_class:
            split, outs = class_split, class_outs
        else:
            split = r.randint(1, n_prog - 1) if n_prog > 1 and r.random() < 0.3 else None
            outs = r.sample(UNIT_BYTES, 1 if split is None else 2)
        desc = json.loads(audit_descriptor(shape_of(n_prog, split, outs), entries, renamed, i in selects, scan_slots))
        prog = program_sha512(desc)
        if not in_class:
            untied.append(prog)
        # `tied-sampled`: one or two of the tied audit's proof blocks, the demo's sample
        k, strata = r.randint(1, min(2, units - 1)) if sampled else units if i in tied else r.randint(1, units), None
        sampling_design = {"k": k, "law": "subset"}
        wrong = (0 if r.random() < 0.25 else r.randint(1, units)) if sampled else r.randint(0, units)
        if partial:
            sampling_design, strata = _by_template(r, desc, m)
            k, wrong = sum(s["k"] for s in strata), 0 if r.random() < 0.25 else r.randint(1, units)
        elif halves or (r.random() < 0.5 and not sampled):
            cuts = [0, *([r.randint(1, units - 1)] if units > 1 and (halves or r.random() < 0.5) else []), units]
            sampling_design = {"k": k, "law": "stratified", "strata": [
                {"k": stratum_k(k, hi - lo, units), "template": f"s{j}", "units": [[lo, hi]]}
                for j, (lo, hi) in enumerate(zip(cuts, cuts[1:]))]}
            strata = [{"k": s["k"], "name": s["template"], "size": s["units"][0][1] - s["units"][0][0]}
                      for s in sampling_design["strata"]]
            k = sum(s["k"] for s in strata)
        if i == a and m in ("audit-oversampled", "tied-oversampled"):
            k = _oversample(r, units, k, strata)
        if halves and m == "tied-stratum-oversampled":
            _stratum_oversample(r, strata)
        elif halves:
            k = _strata_resize(r, strata, seed)
        elif i == a and m == "audit-unregistered":
            k, strata = _resample(r, units, k, strata)
        rec = {"format": r.choice(FORMATS), "law": sampling_design, "partition": f"loop/{i}", "population": units,
               "program": prog, "protocol": ONE_STAGE, "window": {"kind": "served", "window": i}}
        rec["roots"] = _output_entries(rp, i, ties, sched.window // sched.every, first + t, untie, unwire, hit)
        stated = i == ties[first + t]["audit"]
        if not (stated and unstate == "audit"):
            st = states[i] | ({"root": rp.randbytes(64).hex()} if stated and unstate == "audit-root" else {})
            there = rp.choice(["roots", "registered"])
            got = [st | ({"port": STATE} if there == "roots" else {"value": SCAN_STATE}) | {"leaves": sched.window}]
            if stated and unstate == "audit-extra":
                got.append(got[0] | {"root": rp.randbytes(64).hex()})
            if stated and unstate == "audit-activations":
                held = st | {"value": SCAN_STATE, "leaves": sched.window, "words": 1}
                rec["window"]["activations"] = {"program": WINDOW_PROGRAM, "values": {SCAN_STATE: held}}
            else:
                rec[there] = rec.get(there, []) + got
                rp.shuffle(rec[there])
        if i == a and m == "audit-program":
            if r.random() < 0.5:
                rec["program"] = r.choice([hashlib.sha512(b"another program").hexdigest(), prog.upper(), None])
            else:
                desc = json.loads(audit_descriptor(((n_prog, r.choice([o for o in UNIT_BYTES if o != outs[0]])),),
                                                   entries, renamed, i in selects, scan_slots))
        elif i == a and m == "audit-unsampled":
            rec = _unsample(r, rec)
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
        audit_params.append({"free": r.randint(0, 5), "program": desc, "steps": r.randint(0, 4)})
        audit_regs.append(rec)
        audits.append(answer)

    case_params = {"audits": audit_params, "cells": [r.randint(0, 100) for _ in range(r.randint(0, 3))], "first": first,
                   "jitter": r.randint(1, 5), "keys": [e.json() for e in listed], "link": link,
                   "lms": {"H": H, "N": N, "W": W, "toy": toy}, "program": program, "windows": n_windows}
    case_params |= {"cells": bound.rows_cells(n_windows), "circuit": circuit, "scan": scan, "schedule": schedule_json(bound)}
    if unhold == "params":
        case_params["circuit"] = r.choice([hashlib.sha512(b"another class circuit").hexdigest(), program, *untied])
    c_json = {"audits": audit_regs, "windows": window_regs}
    # each listed entry's session's signed start, which the evidence shows: its base on its key's life, a key's first
    # listed entry's sampled and each later one's as many of the period's schedule's execution phases on as its start
    # is (`_held_to`), and its ordinal among its key's sessions, a key's first listed entry's sampled and each later
    # one's the next, signed under the entry's key in its session
    rs = random.Random(f"network-period starts {m} {seed}")
    L = bound.window * bound.bucket_ms
    anchors: dict[bytes, tuple[int, int]] = {}
    bases, ordinals, last = [], [], {}
    for entry in listed:
        sampled = rs.choice([0, rs.randrange(1 << 20), rs.randrange(1 << 62)])
        anchor, base = anchors.setdefault(entry.key.I, (entry.start, sampled))
        bases.append(base + (entry.start - anchor) * L)
        last[entry.key.I] = last[entry.key.I] + 1 if entry.key.I in last else rs.choice([0, rs.randrange(1 << 62)])
        ordinals.append(last[entry.key.I])
    if m == "start-not-signed":
        # the second session's base is not the one its start is read off: an execution phase on or back, a
        # millisecond on, or the first session's own
        bases[1] = rs.choice([bases[1] + L, bases[1] - L, bases[1] + 1, bases[0]])
    if m == "start-skipped":
        # the second session is not the one after the first: one or many of the key's sessions between them, which the
        # list leaves out, or the first's own ordinal or one before it
        ordinals[1] = rs.choice([ordinals[1] + 1, ordinals[1] + rs.randrange(2, 1 << 40), ordinals[0],
                                 ordinals[0] - 1 if ordinals[0] else ordinals[1] + 2])

    def signed_start(i: int, how: str | None = None) -> dict:
        """Listed entry `i`'s session's signed start, as a run's `deployment.json` holds it: its base, its ordinal and
        the period's schedule's digest, signed for the link's start name, egress, window 0; or one that is not its
        session's certified start, as `how` says (`UNCERTIFIED_BY`)."""
        e = listed[i]
        named = bound.digest
        named = _flip(named, seed) if how == "schedule" else named
        k = _key(ki, 1) if how == "key" else e.key
        c = Certificate(k.I, link if how == "link" else start_link(link), INGRESS if how == "direction" else EGRESS,
                        1 if how == "window" else 0,
                        start_digest(bases[i], ordinals[i], _flip(named, seed + 1) if how == "digest" else named),
                        b"")
        ses = _flip(e.session, seed) if how == "session" else e.session
        sig = lms.sign_at(k, i % 2 ** k.params.H, c.message(ses)).encode()
        if how == "forged":
            sig = rs.choice([_flip(sig, seed), sig[:-1], b""])
        return {"base_ms": bases[i] + (how == "base"), "certificate": _cert_json(dataclasses.replace(c, signature=sig)),
                "ordinal": ordinals[i] + (how == "ordinal"), "schedule": named.hex()}

    hows: list[str | None] = [None] * len(listed)
    if m == "start-uncertified":
        hows[rs.randrange(len(listed))] = rs.choice([h for hs in UNCERTIFIED_BY.values() for h in hs])
    shown_starts = [signed_start(i, how) for i, how in enumerate(hows)]
    if m == "start-count":
        # one too few (none, of one: evidence with no "starts"), one too many, or none
        shown_starts = rs.choice([shown_starts[1:], [*shown_starts, shown_starts[-1]], []])
    showing = {"starts": shown_starts} if shown_starts else {}
    e_json = {"audits": audits, "windows": windows} | showing
    if m == "first":
        case_params["first"] = first + r.choice([1, -1] if first else [1])
    elif m == "count-windows":
        case_params["windows"] = n_windows + r.choice([1, -1] if n_windows > 1 else [1])
        case_params["cells"] = bound.rows_cells(case_params["windows"])
    elif m == "count-audits":
        case_params["audits"] = r.choice([audit_params[1:], [*audit_params, {
            "free": 0, "program": json.loads(audit_descriptor(((1, 2),))), "steps": 0}]])
    elif m == "evidence-count":
        e_json = showing | r.choice([
            {"audits": audits, "windows": windows[1:]}, {"audits": audits, "windows": windows + windows[:1]},
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
           *bad_entries(p).values(), p | {"program": p["program"][:-2]},
           p | {"program": p["program"].upper()}, p | {"program": 0},
           p | {"lms": p["lms"] | {"W": 3}}, p | {"lms": p["lms"] | {"H": 0}}, p | {"lms": p["lms"] | {"toy": 0}},
           p | {"lms": p["lms"] | {"x": 0}}, p | {"lms": {k: v for k, v in p["lms"].items() if k != "toy"}},
           p | {"cells": [-1]}, p | {"cells": [True]}, p | {"first": -1}, p | {"first": True}, p | {"first": "0"},
           *bad_audits(p)]
    s = p["schedule"]
    return out + [*bad_cells(p), p | {"schedule": {k: v for k, v in s.items() if k != SCHEDULE_KEYS[n % len(s)]}},
                  p | {"schedule": s | {"x": 1}}, p | {"schedule": s | {"egress": "bytes"}}, p | {"schedule": [s]},
                  p | {"schedule": s | {"delay": 0}},
                  p | {"schedule": s | {"lanes": True}}, p | {"schedule": s | {"window": -1}},
                  p | {"schedule": s | {"queue": "1"}}, p | {"schedule": s | {"cell_bytes": 10}},
                  p | {"schedule": s | {"cell_bytes": U32}}, p | {"schedule": s | {"lanes": U32}},
                  p | {"schedule": s | {"window": U32}}, p | {"schedule": s | {"every": s["window"] + 1}},
                  p | {"schedule": s | {"every": 0}}, p | {"sync_set": {"deltas": [0], "rhos": [[1, 1]]}},
                  p | {"circuit": p["circuit"][:-2]}, p | {"circuit": p["circuit"].upper()}, p | {"circuit": 0},
                  p | {"circuit": None}, *_bad_caps(s, p), *bad_lengths(p), *connection_rows_params(p), *bad_scans(p)]


def bad_cells(p: dict) -> list[dict]:
    """Rows-form parameters whose `cells` aren't the schedule's (`Schedule.rowsCells`), each one change to `p`: none,
    one connection short, one more, and one below `egress_cells`."""
    cells = p["cells"]
    return [p | {"cells": []}, p | {"cells": cells[1:]}, p | {"cells": [*cells, cells[0]]},
            p | {"cells": [cells[0] - 1, *cells[1:]]}]


def bad_entries(p: dict) -> dict[str, dict]:
    """Parameters whose `"keys"` `params` refuses, each one change to `p`, by what it is. No entry (`empty`: `NO_KEY`);
    an entry that starts before the one listed before it (`out-of-order`: `OUT_OF_ORDER`); two entries with one start
    (`overlap`: `OVERLAP`); the first entry again, its key identifier and its session, further on (`twice`: `TWICE`);
    another key under the first's identifier (`one-id`: `TWO_KEYS`); a first entry that starts after the period does (`before`: `BEFORE`); an entry that is the key alone (`no-entry`:
    `NO_ENTRY`); the earlier forms, `"key"` in place of `"keys"` or beside it, `"session"` and `"sessions"` (`key`,
    `key-too`: `KEY_NOT_KEYS`; `session`: `SESSION_IN_ENTRY`; `sessions`: `SESSIONS_IN_ENTRY`); and a list or an entry
    that doesn't read: entries that are no list, an entry without its start or with a field more, a key or a session
    that is no string, a byte short, a byte long or in uppercase hex, and a start that is no natural number."""
    es = p["keys"]
    e, last = es[0], es[-1]["start"]
    k, ses = e["key"], e["session"]
    flip = lambda x, i: x[:i] + ("1" if x[i] == "0" else "0") + x[i + 1:]  # noqa: E731
    end = len(k) - 1
    # another key and another session, in no listed entry; and another key under the first's identifier
    fresh = {"key": flip(k, 0), "session": flip(ses, 0)}
    head = lambda x: p | {"keys": [e | x, *es[1:]]}  # noqa: E731
    return {
        "empty": p | {"keys": []},
        "out-of-order": p | {"keys": [fresh | {"start": e["start"] + 1}, *es]},
        "overlap": p | {"keys": [fresh | {"start": e["start"]}, *es]},
        "twice": p | {"keys": [*es, e | {"start": last + 1}]},
        "one-id": p | {"keys": [*es, fresh | {"key": flip(k, end), "start": last + 1}]},
        "before": p | {"keys": [e | {"start": p["first"] + 1}]},
        "no-entry": p | {"keys": [k]},
        "key": {x: v for x, v in p.items() if x != "keys"} | {"key": k}, "key-too": p | {"key": k},
        "session": p | {"session": ses}, "sessions": p | {"sessions": [x["session"] for x in es]},
        "no-list": p | {"keys": e}, "no-start": p | {"keys": [{"key": k, "session": ses}, *es[1:]]},
        "extra": head({"end": last + 1}),
        "key-no-string": head({"key": 1}), "key-short": head({"key": k[:-2]}), "key-long": head({"key": k + "00"}),
        "key-upper": head({"key": k.upper()}),
        "session-no-string": head({"session": 1}), "session-short": head({"session": ses[:-2]}),
        "session-long": head({"session": ses + "00"}), "session-upper": head({"session": ses.upper()}),
        "start-negative": head({"start": -1}), "start-bool": head({"start": True}),
        "start-string": head({"start": str(e["start"])})}


def _bad_caps(s: dict, p: dict) -> list[dict]:
    """Parameters whose response cap `scheduleOf` refuses, each one change to `p`'s schedule `s`: a cap that is no
    positive u32 below `egress_cells`, and a cap on a cell that doesn't hold an id request's `max_tokens`."""
    wide = s | {"cell_bytes": max(s["cell_bytes"], CAPPED_CELL), "egress_cells": max(s["egress_cells"], 2)}
    return [p | {"schedule": wide | {"max_tokens_cap": cap}}
            for cap in (0, -1, True, "1", wide["egress_cells"], wide["egress_cells"] + 1, U32)] + [
        p | {"schedule": wide | {"cell_bytes": CAPPED_CELL - 1, "max_tokens_cap": 1}}]


def _lengths_schedule(s: dict) -> dict:
    """An uncapped schedule `s` with the fields a schedule with lengths rows holds: a cell of at least `LENGTHS_CELL`,
    ingress that ends by the delay, and a connection that ends within a window of its admission."""
    return s | {"cell_bytes": max(s["cell_bytes"], LENGTHS_CELL), "delay": 1, "ingress_buckets": 1,
                "egress_cells": min(s["egress_cells"], s["slots"] * s["window"])}


def _words_schedule(s: dict) -> dict:
    """A schedule with lengths rows `s` (`_lengths_schedule`) with request words: content rows under the lengths
    rows' form 2, `max_tokens_cap` 1 below at least two egress cells, which a connection of a window of two buckets
    or more still ends within, and a cell that holds an id request's `max_tokens`."""
    return s | {"cell_bytes": max(s["cell_bytes"], CAPPED_CELL), "egress_cells": max(s["egress_cells"], 2),
                "lengths": 2, "content": [[2, 2]], "max_tokens_cap": 1, "request_words": True}


def connection_rows_params(p: dict) -> list[dict]:
    """Rows-form parameters whose schedule has connection rows, each one change to `p`'s uncapped schedule
    (`_lengths_schedule`): lengths rows of each form, content rows of the form's classes with them, and request
    words with form 2's (`_words_schedule`). Each is a schedule `scheduleOf` reads, as `NetworkCertifierParams` builds it, and
    one the check refuses by name (`CONNECTION_ROWS`)."""
    s = _lengths_schedule(p["schedule"])
    return [p | {"schedule": s | rows} for rows in (
        {"lengths": 1}, {"lengths": 2}, {"lengths": 1, "content": [[2, 2, 1]]}, {"lengths": 2, "content": [[2, 2]]},
        {"lengths": 2, "content": [[2, 4], [4, 2]]})] + [p | {"schedule": _words_schedule(s)}]


def bad_lengths(p: dict) -> list[dict]:
    """Rows-form parameters whose schedule's lengths rows or content rows `scheduleOf` refuses, each one change to
    `p`'s uncapped schedule with lengths rows (`_lengths_schedule`). Lengths rows: a form that is not 1 or 2,
    ingress that ends after the delay, a connection that outlasts the window, egress cells past a u16's count, a
    cell that doesn't hold an id request's first prompt id, and a slot stride of more releases than a lengths row
    has slots. Content rows: without lengths rows, of no class, of the other form's numbers, of a size that is no
    power of two, of more ids than a response row has, of more prompt ids than a request has bytes, of no row or of
    more rows than the lengths table has slots, and classes that are no list of lists of naturals. Request words
    (`_words_schedule`): not `true`, under form 1, without content rows, and without `max_tokens_cap`."""
    s = _lengths_schedule(p["schedule"])
    words = _words_schedule(s)
    one, two = s | {"lengths": 1}, s | {"lengths": 2}
    slots = s["window"] // s["every"] * NetworkCertifierParams(**one).stride_releases
    past = 1 << (s["ingress_cells"] * (s["cell_bytes"] - HEADER.size)).bit_length()
    schedules = [s | {"lengths": form} for form in (0, 3, True, "1", [1])] + [
        one | {"ingress_buckets": 2}, one | {"egress_cells": s["slots"] * s["window"] + 1},
        two | {"egress_cells": (1 << 16) + 1}, one | {"cell_bytes": LENGTHS_CELL - 1}, two | {"lanes": 2731}] + [
        with_rows | {"content": classes} for with_rows, classes in (
            (s, [[2, 2]]), (two, []), (two, [[2, 2, 1]]), (one, [[2, 2]]), (two, [[3, 2]]), (two, [[2, 1]]),
            (two, [[2, 2 * content_rows.most_ids()]]), (two, [[past, 2]]), (one, [[2, 2, 0]]),
            (one, [[2, 2, slots + 1]]), (two, "x"), (two, [2, 2]), (two, [[True, 2]]), (two, [[2, -1]]),
            (two, [[2]]), (one, [[2, 2, 1, 1]]))] + [
        words | {"request_words": v} for v in (False, 1, "true", None)] + [
        words | {"lengths": 1, "content": [[2, 2, 1]]}, {k: v for k, v in words.items() if k != "content"},
        {k: v for k, v in words.items() if k != "max_tokens_cap"}]
    return [p | {"schedule": schedule} for schedule in schedules]


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
    ingress = next(b for k, b in sig.items() if k[:2] == ("row", "sin"))
    egress = sig[EGRESS_PARAMS]
    _, state, view = ingress["ret_type"][1]
    return state[1], ingress["params"][2][1][1], view[1], egress["params"][0][1][1]


def bad_scans(p: dict) -> list[dict]:
    """Rows-form parameters whose scan `scanOf` refuses, each one change to `p`: not a descriptor, another Program's
    (its SHA-512 not `program`), and, each under its own SHA-512, the checker form (its egress row an input), templates
    with other parameters, another number of instances, no start unit's template, rows not the schedule's, a lane's `prog<j>` another row of its
    input than its bucket's or another lane's input, an input gate no template's, and each negative of a check of the
    rows the scan's ties equate, of what `NetTiming.BoundaryGates.Fits` asks or of the input nodes its arguments read
    (`scan_negatives`)."""
    s, desc = p["schedule"], p["scan"]
    T, row, widths = s["window"], s["lanes"] * s["slots"] * s["cell_bytes"], _scan_widths(desc)
    L, every = s["lanes"], s["every"]
    ingress = ingress_params(L)

    def own(text: str) -> dict:
        d = json.loads(text)
        return p | {"scan": d, "program": program_sha512(d)}

    def scan(*a, **k) -> str:
        return scan_descriptor(*a, **({"lanes": L, "every": every} | k))

    extra = json.loads(D.canonical(desc))
    extra["types"].append(["Array", 1, ["Value", 1]])
    extra["definitions"][extra["root"]]["body"]["nodes"].append(
        {"args": [], "axes": [], "fn": extra["callees"].index("Input1_v1"), "form": "batch", "n": 1,
         "out": len(extra["types"]) - 1})
    return [p | {"scan": "x"}, p | {"scan": {"free": 0, "steps": 0}}, p | {"scan": desc | {"schema": "verity-ir/v0"}},
            p | {"scan": json.loads(scan(T, row, (widths[0] + 1, *widths[1:])))},
            own(scan(T, row, widths, checker=True)),
            own(scan(T, row, widths, names=(("seed",), ingress, EGRESS_PARAMS))),
            own(scan(T, row, widths, names=(START_PARAMS, ("row", "sin", "seed"), EGRESS_PARAMS))),
            own(scan(T, row, widths, names=(START_PARAMS, ingress, ("words",)))),
            own(scan(T, row, widths, lanes=L + 1)),
            own(scan(T, row, widths, counts=(1, T, T + 1))),
            own(scan(T, row, widths, counts=(2, T, T))),
            own(scan(T, row, widths, counts=(1, T + 1, T))),
            own(scan(T, row, widths, counts=(1, T - 1, T))),
            own(scan(T, row, widths, counts=(0, T, T))),
            own(scan(T, row + 1, widths)),
            *([own(scan(T, row, widths, skew=1))] if T // every > 1 else []),
            *([own(scan(T, row, widths, shared=True))] if L > 1 else []),
            p | {"scan": extra, "program": program_sha512(extra)},
            *(q for q, _ in scan_negatives(p).values())]


#: A primitive that is no input, a constant gate the `row` negative binds bucket 0's row to.
ZERO = "NetworkPeriodScanZero_v1"
#: Members of each of the `walk` negative's nested batches: three deep, past `Flock.Steps.FUEL`'s walk.
DEEP = 2 ** 30


def scan_negatives(p: dict) -> dict[str, tuple[dict, str]]:
    """Rows-form parameters each refused by one of `scanOf`'s checks of the rows its ties equate, of what
    `NetTiming.BoundaryGates.Fits` asks or of the input nodes its arguments read, and by none before it, each one change
    to `p`'s generator-form scan under its own SHA-512, with a fragment of both refusals: `view`, the ingress
    template's view one bit more than the egress template's `state`; `state`, the ingress template's `sin` one bit
    more than the state it returns (its input node and the template's argument grown with it); `start`, the start
    unit's `sin` one bit more than the ingress template's, and `start-out`, the state it returns one bit more;
    `walk`, the egress template calling a composite of `2^90` gates in three nested batches, unread, so the walk to an
    operand is past `FUEL`; `returns`, the root returning an input leaf for the last egress leaf; `computes`, the egress
    template returning its `state`'s first 16 leaves as its u16; `row`, the ingress template's row bound to a constant primitive's
    gates (an input node like it appended, keeping the input count); `argument`, egress member 0's first state leaf the
    ingress template's first output leaf; `ingress`, the egress state the
    ingress template's row input node (resized, the state's node resized to keep the count); `shared`, the ingress
    template's `sin` its row's input node from its first leaf, the `sin` input node left
    unread; `carried`, the start unit's `sin` the ingress template's `sin`'s input node from its first leaf, its own
    input node left unread; `entries`, the egress state lane 0's slot entries' input node (resized, the state's node resized to keep
    the count); `unread`, an input node of no members appended, which no argument reads; and `topological`, the
    ingress template's row input node moved after the templates, so the template names a later node."""
    desc, sched = p["scan"], p["schedule"]

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

    out, two_lengths = {}, "would equate rows of two lengths"
    state, prog, view, words = _scan_widths(desc)
    shape = (sched["window"], sched["lanes"] * sched["slots"] * sched["cell_bytes"])
    out["view"] = own(json.loads(scan_descriptor(
        *shape, (state, prog, view + 1, words), lanes=sched["lanes"], every=sched["every"]))), two_lengths
    for name, start in (("start", (state + 1, state)), ("start-out", (state, state + 1))):
        out[name] = own(json.loads(scan_descriptor(
            *shape, (state, prog, view, words), lanes=sched["lanes"], every=sched["every"], start=start))), two_lengths

    d, nodes, ts = parts()
    x = nodes[ts[1]]
    sin = d["definitions"][d["callees"][x["fn"]]]["body"]["params"][1]
    sin[1] = ["Array", state + 1, ["Value", 1]]
    ks, n = x["args"][1]["refs"]["affine"][1], x["n"] * (state + 1)
    nodes[ks] = nodes[ks] | {"n": n, "out": typed(d, ["Array", n, ["Value", 1]])}
    x["args"][1] = {"refs": {"affine": ["n", ks, 0, n]}, "type": typed(d, ["Array", x["n"], sin[1]])}
    out["state"] = own(d), two_lengths

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
    k0 = nodes[ts[1]]["args"][0]["refs"]["affine"][1]
    d["definitions"][ZERO] = {"form": "primitive", "id": ZERO, "params": [], "ret": ["Value", 1]}
    d["callees"].append(ZERO)
    nodes.append(dict(nodes[k0]))
    nodes[k0] = nodes[k0] | {"fn": len(d["callees"]) - 1}
    out["row"] = own(d), "the ingress template's row is not one input node's leaves"

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
    out["ingress"] = own(d), "an egress template's argument is the ingress rows' input node"

    d, nodes, ts = parts()
    row, sin = nodes[ts[1]]["args"][:2]
    sin["refs"] = {"affine": ["n", row["refs"]["affine"][1], 0, sin["refs"]["affine"][3]]}
    out["shared"] = own(d), "(NetworkCheck.Period.argsApart)"

    d, nodes, ts = parts()
    own_sin, sin = nodes[ts[0]]["args"][0], nodes[ts[1]]["args"][1]
    own_sin["refs"] = {"affine": ["n", sin["refs"]["affine"][1], 0, own_sin["refs"]["affine"][3]]}
    out["carried"] = own(d), "(NetworkCheck.Period.argsApart)"

    d, nodes, ts = parts()
    ks, kp = nodes[ts[-1]]["args"][0]["refs"]["affine"][1], _member_ref(nodes[ts[1]], 2, 0, 0, d)[1]
    S, E = nodes[ks]["n"], nodes[kp]["n"]
    nodes[kp] = nodes[kp] | {"n": max(S, E), "out": typed(d, ["Array", max(S, E), ["Value", 1]])}
    nodes[ks] = nodes[ks] | {"n": min(S, E), "out": typed(d, ["Array", min(S, E), ["Value", 1]])}
    nodes[ts[-1]]["args"][0]["refs"] = {"affine": ["n", kp, 0, S]}
    out["entries"] = own(d), "(NetworkCheck.Period.argsApart)"

    d, nodes, ts = parts()
    nodes.append({"args": [], "axes": [], "fn": d["callees"].index("Input1_v1"), "form": "batch", "n": 0,
                  "out": typed(d, ["Array", 0, ["Value", 1]])})
    out["unread"] = own(d), "(NetworkCheck.Period.inputsRead)"

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
    bad_starts = []
    if "starts" in e:
        s, later = e["starts"][0], e["starts"][1:]
        bad_starts = [[], "x", [1], *([y, *later] for y in (
            {k: v for k, v in s.items() if k != "schedule"}, s | {"x": 0}, s | {"base_ms": -1}, s | {"base_ms": True},
            s | {"base_ms": str(s["base_ms"])}, {k: v for k, v in s.items() if k != "ordinal"}, s | {"ordinal": -1},
            s | {"ordinal": True}, s | {"schedule": s["schedule"][:-1]}, s | {"schedule": 0},
            s | {"certificate": s["certificate"] | {"x": 0}}, s | {"certificate": "x"},
            s | {"certificate": {k: v for k, v in s["certificate"].items() if k != "signature"}}))]
    return [b[:-1], b"\xff" + b, json.dumps(e).encode(), enc(e | {"x": 1}), enc({"windows": e["windows"]}),
            enc(e | {"audits": [1]})] + [enc(e | {"windows": [w, *rest]}) for w in bad_windows] + \
        [enc(e | {"starts": x}) for x in bad_starts]


answer = reference
"""The period check's outcome on one case: `reference` under the name a composition's components answer by."""

#: What a case can be aimed at: each verdict, and a case that fails to decode (its params, commitment or evidence).
KINDS = (ACCEPT, REJECT, UNDECIDED, "broken-params", "broken-commitment", "broken-evidence")
#: The changes (`build`'s `m`) by the kind they aim at, those to the listed keys aside (`LISTS`): a rejection by the
#: check itself, not by decoding, for `reject`; `case` breaks the line `serve` reads rather than the period, so no kind
#: uses it.
AIMED = [m for m in EXPECT if m.removeprefix("rows-") not in LISTS]
AIMS: dict[str, tuple[str, ...]] = {
    ACCEPT: tuple(m for m in AIMED if EXPECT[m] == ACCEPT),
    REJECT: tuple(m for m in AIMED if EXPECT[m] == REJECT
                  and m.removeprefix("rows-") not in ("params", "commitment", "evidence", "case")),
    UNDECIDED: tuple(m for m in AIMED if EXPECT[m] == UNDECIDED),
    **{f"broken-{x}": tuple(m for m in AIMED if m.removeprefix("rows-") == x) for x in ("params", "commitment", "evidence")},
}


def verdict_of(kind: str) -> str:
    """The verdict a case aimed at `kind` gets: a case that fails to decode is rejected."""
    return kind if kind in (ACCEPT, REJECT, UNDECIDED) else REJECT


def case(kind: str, seed: int = 0) -> dict:
    """A small period (`build(..., small=True)`: the toy key, as few execution phases and audits as its change needs) aimed
    at `kind`, its change and every choice sampled from `seed`."""
    ms = AIMS[kind]
    return build(ms[random.Random(f"network-period {kind} {seed}").randrange(len(ms))], seed, small=True)


def cases(seed: int = 0) -> dict[str, tuple[dict, dict]]:
    """One small period of each kind (`KINDS`) with its answer, `{kind: (case, answer)}`."""
    return {k: (c, answer(c)) for k in KINDS for c in [case(k, seed)]}
