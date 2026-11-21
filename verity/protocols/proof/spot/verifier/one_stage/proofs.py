"""A statement's proof files, verified by Flock's Lean verifier (`verity-verify-proof verify --zk`, the Lake package
`verity/protocols/proof/flock/verifier`, built with `lake` on first use): the proof system's own run of the verifier, so a protocol
reads each session's verdict from that run, never from a verdict record the prover hands in.

`verify` takes a statement's files as the prover delivered them (`Proof`: the circuit file, the public file and the
sessions' directories, each its `session.json` and its rep files) and the verifier's own inputs: the SHA-512 its auditor
committed for the circuit file, the program's descriptor, the partition object and, for a statement that reads
registered values, the `--registered` file. It refuses
a circuit file that doesn't hash to the committed digest before anything runs, so each verdict is over the circuit the
auditor's own lowering built. The circuit file's tie to the program's Definitions rests on that lowering: the verifier
holds the file to the program only as far as its META names it, until it derives the circuit from the descriptor.
Then one `verity-verify-proof verify --zk` over every session (`--registered` when the statement reads registered
values), and one `Finding` a session, in order:
whether it was accepted and why not, the program its verdict names (`program_sha512`, the digest of the canonical
descriptor without annotations, which an accepted verdict must give as the partition object's `program`),
its registered finding (each port's value and opened root) and the proof blocks it proved, the public file's global proof block
indices of the instances its session sampled (`unit_draw`), read only off a session the verifier accepted.

`ks_form_why` mirrors `verify --zk`'s refusal of a circuit outside the unit's form the sampled audit's knowledge
soundness (`SampledKSLiveV3`, and `SampledKSLiveV5` at a shared-row public file) covers, on the circuit file's head
(`Flock.KsScope.filesWhy`, which `verity-verify-proof ks-form` runs first). A session Lean accepts whose files it refuses is a
``form`` refusal, so the mirror can only refuse more.

`fits_why` and `draw_fits_why` mirror v2's refusal of a length a session's statement can't hold, since it writes each in
an 8-byte word (`OneStageZk.fitsWhy`, `sampleFitsWhy`, at `WORD`): a file, a lookup table, the program, or the partition
object, the registered reads or a session's sample as `printed` prints them, at ``2^64`` bytes or more.

`population_why` mirrors the v2 audit's check that the public file's units are the docket's population in order
(`Spot.FilePart.why`), which `verify` applies to the one-stage audit's statement when handed its strata.

The verifier reads the whole circuit file once a call, in time and memory linear in it (about six times the file at
peak), then O(proof) a session: so a protocol verifies one statement a stratum, every sampled proof block of it an instance.
It runs in `verifier_env`: its caller's environment over `VERIFIER_ENV`, the verifier's allocator setting.

The one-stage audit's v2 (`V2`, `Spot.OneStageZk`) runs this verifier inside the audit's own Lean, on the files
as content-addressed blobs: v1's inputs with the verdicts replaced by the files' digests (`evidence_v2`) and the
files themselves, and `verdicts_v2` is that run's verdicts alone, which the difftest holds to
`verify`'s. `order_why` mirrors its refusal of a challenger release log out of the order the sample, then the link,
then the other rounds (`OneStageZk.orderHeld`). The registered circuit names the lookup tables its session reads (its
META's ``lookups``, `lookups_of`), and v2 refuses blobs whose pinned MUFU tables are other than those (`tables_why`,
`OneStageZk.tablesHeld`); in the per-member form a member's statement holds the blobs' pinned tables its own circuit
looks up and no other (`own_tables`, `tables_match_why`).
The evidence's per-member form is one statement a sampled member (`OneStageZk.membersEvidenceOf`).

A window of several members has v2's per-member form of ``proofs`` (`members_v2`: every `Member` in the parameters, one
statement for each member with a session in the evidence), which the audit's check reads beside the one statement's
and decides by `OneStageZk.checkMembersAt` (`members.check`). `members_parsed` is what its parsers read of a case's
parameters and evidence, with no check and no verdict.

Building takes the lock `verifier.BUILD_LOCK` takes: Lake builds the flock package in place for `spot-verifier`
too, and two builds at once would rewrite one module's C under each other.
"""
from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from verity.circuits.library import TABLES

from verity.protocols.proof.spot.verifier.challenger.challenger import SAMPLE_LINK_FIRST, _nat, sample_link_first
from verity.commitments.docket_digest import digest as registration_digest
from . import one_circuit as OC
from .verifier import BUILD_LOCK, ROOT
from .verifier import PACKAGE as V2_PACKAGE
from .settings import TMPL_SWITCH, door_tmpl, lean_env
from . import lean_child as LC
from verity.commitments.domains import ONE_STAGE_PROTOCOL_V2

PACKAGE = ROOT / "verity/protocols/proof/flock/verifier"
EXE = "verity-verify-proof"
#: what the Lean verifier's `verify` runs under where its caller's environment sets no value of its own. The Lean runtime's
#: allocator (mimalloc) returns freed pages to the kernel after a delay and faults them in again when it reuses them; -1
#: keeps them, trading peak memory for the faults. No byte the verifier reads or prints depends on it.
VERIFIER_ENV = {"MIMALLOC_PURGE_DELAY": "-1"}
#: the statement every sampled proof block's session proves
STATEMENT = "verity/flock-circuit"
#: the door's templated statement (`Tags.byName`), the one statement the door sets up under `TMPL_SWITCH`
TMPL_STATEMENT = "verity/flock-circuit/tmpl"
TMPL_FORMAT = "flock-circuit/tmpl"
TMPL_FLAT_WHY = "a circuit file that is not templated (its head is not flock-circuit/tmpl) under the templated statement"
TMPL_DEFINITION_WHY = "a templated circuit whose META names a definition (a proof-block class has no templated statement)"
#: an input bound at docket (`program.REGISTERED_ID`, Lean's `Program.inputId?`), a descriptor's `definitions` key
_REGISTERED_INPUT = re.compile(r"RegisteredInput[1-9][0-9]*_v1")
#: `Flock.KsScope.SCOPE_WHY` (`ks_form_why`) and `TABLES_WHY`, the head of `verify --zk`'s refusals of a shared-row file
SCOPE_WHY = ("a circuit outside the unit's form under --zk (scopeOkH: the unit reads its one input group and returns the "
             "outputs)")
TABLES_WHY = "a shared-row public file (header shared_rows) under --zk"
#: `OneStageZk.NO_ROW_WHY` and `WIDE_WHY` (`template_why`), v2's refusals of the unit template
NO_ROW_WHY = "the circuit's units_per_vu is 0: a template instance of no unit, a proof block of no gate"
WIDE_WHY = ("a template instance returns more than 262144 bits (16 per output leaf of its units_per_vu units): past "
            "the widest size class")
#: `OneStageZk.SIZE_CLASS_MAX`: the widest public size class of a proof block's outputs, in bits
SIZE_CLASS_MAX = 262144
#: `Flock.MUFU_TABLES`' SHA-512s, in its order: the digests the v2 audit finds a pinned lookup table under among its blobs
#: (`OneStageZk.tablesOf`)
MUFU_SHA512 = {t.name: t.sha512 for t in TABLES if t.name in ("rsq", "sqrt", "rcp", "ex2")}
#: `OneStageZk.NO_TABLES`, the head of the v2 audit's refusal of blobs whose pinned MUFU tables are not the registered
#: circuit's lookups (`tables_why`)
NO_TABLES = "tables: the blobs' pinned MUFU tables are not the ones the registered circuit looks up (its META's lookups)"


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
    registered finding (port -> {value, root}), the global proof blocks it proved (none unless accepted) and the rows it
    opened (its verdict's ``openings``, which `produced.shared` reads; None when it reports none), and the drawn block
    its statement's units are exactly (`verify`'s match; None where no door matched one), by which a caller holds every
    drawn block to exactly one proof."""

    session: str
    accepted: bool
    why: str
    program: str | None
    registered: dict
    units: tuple[int, ...]
    openings: Any = None
    block: int | None = None


def verifier_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment `verify` starts the Lean verifier in: ``environ`` (this process's when None) over `VERIFIER_ENV`, so
    a variable the caller sets keeps the caller's value, under `lean_env`. `verifier_env.sh` is the same for a shell job."""
    return lean_env({**VERIFIER_ENV, **(os.environ if environ is None else environ)})


@functools.cache
def built(package: Path, target: str) -> Path:
    """`lake build <target>` in ``package``, once per process for each, under `BUILD_LOCK`; the executable's path."""
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            p = subprocess.run(["lake", "build", target], cwd=package, capture_output=True, text=True)
        except FileNotFoundError:
            raise LeanError(f"lake is not on the PATH, so {target} in {package} cannot be built") from None
    if p.returncode != 0:
        raise LeanError(f"lake build {target} in {package} failed ({p.returncode}):\n{p.stdout[-2000:]}{p.stderr[-2000:]}")
    return package / ".lake" / "build" / "bin" / target


def executable() -> Path:
    """`verity-verify-proof`, built (`built`)."""
    return built(PACKAGE, EXE)


def lower_executable() -> Path:
    """`flock-lower-check`, built (`built`): the door's lowering of C_S (`one_circuit.lower`)."""
    return built(PACKAGE, OC.LOWER_EXE)


def file_sha512(path: Path) -> str:
    """The SHA-512 of the file at `path`."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha512").hexdigest()


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


def statement() -> str:
    """The statement the door sets up: `TMPL_STATEMENT` where `TMPL_SWITCH` is ``1``, else `STATEMENT`."""
    return TMPL_STATEMENT if door_tmpl() else STATEMENT


def tmpl_why(circuit: Path) -> str | None:
    """The templated reader's refusals (`HmOut.parseTmpl`) on the circuit file's head: a file not in the templated form
    (`TMPL_FLAT_WHY`: its format line and META's ``format`` and ``statement`` not the templated ones), and a META naming
    a ``definition`` (`TMPL_DEFINITION_WHY`); None when neither applies."""
    with open(circuit, "rb") as f:
        fmt = f.readline().decode(errors="replace").split(" ")[0]
        line = f.readline().decode(errors="replace")
    try:
        meta = json.loads(line[len("META "):]) if line.startswith("META ") else None
    except ValueError:
        return TMPL_FLAT_WHY
    if fmt != TMPL_FORMAT or not isinstance(meta, dict) or meta.get("format") != TMPL_FORMAT \
            or meta.get("statement") != TMPL_STATEMENT:
        return TMPL_FLAT_WHY
    if "definition" in meta:
        return TMPL_DEFINITION_WHY
    return None


def ks_form_why(circuit: Path, public: Path) -> str | None:
    """`Flock.KsScope.filesWhy`'s scope refusal on a `verity/flock-circuit` statement's files, with
    `ProvedScope.refusal`'s text: a circuit outside the unit's form (`ProvedScope.scopeOkH`'s two conditions past
    `ProvedScope.check`'s: the unit reads one input group, and META's ``out_net``, the net returning the outputs, is the
    unit); None when it applies or the circuit file's head is not in that form. A shared-row public file (header
    ``shared_rows``) is Lean's alone to refuse (`KsScope.sharedWhy`, outside the form `SampledKSLiveV5` covers), since
    its check loads the whole file and computes every port's root. `verify` refuses a session Lean accepts that it
    refuses, so it can only refuse more; `verity-verify-proof ks-form`'s form check is the Lean it mirrors, and that
    command's check of the public file's header and rows after it (`Flock.PortRoots.headerOf`) has no mirror here."""
    del public
    head = _unit_head(circuit)
    if head is not None:
        meta, unit = head
        groups = unit[4] if len(unit) > 4 else None
        if meta.get("out_net", "unit") != "unit" or groups != "1":
            return _refusal(SCOPE_WHY)
    return None


def template_why(circuit: Path) -> str | None:
    """`OneStageZk.templateOk`'s refusals on the circuit file's head, with its reasons: a template instance of no unit
    (META's ``units_per_vu`` is 0), else one returning more than `SIZE_CLASS_MAX` bits (16 per entry of META's
    ``leaves_out`` over its ``units_per_vu`` units); None when neither applies or the head is not in that form. v2
    refuses each session of such a circuit with that reason, so `verify` turns a session Lean accepts into one rejected
    with it."""
    head = _unit_head(circuit)
    if head is None:
        return None
    meta, _ = head
    leaves, upv = meta.get("leaves_out", []), meta.get("units_per_vu")
    if not (isinstance(leaves, list) and isinstance(upv, int)):
        return None
    if upv == 0:
        return NO_ROW_WHY
    if 16 * sum(len(leaves[u]) if u < len(leaves) and isinstance(leaves[u], list) else 0 for u in range(upv)) \
            > SIZE_CLASS_MAX:
        return WIDE_WHY
    return None


def _refusal(why: str) -> str:
    return f"refused: {why}: a statement form the soundness proof does not cover"


def reads_registered(program: bytes) -> bool:
    """Whether the program descriptor `program` has an input bound at docket (a `RegisteredInput<w>_v1` among its
    `definitions`), or has no `definitions` object to tell: True unless it names none."""
    return not _reads_none_registered(program)


def _reads_none_registered(program: bytes) -> bool:
    """Whether `program` is JSON whose `definitions` object names no `RegisteredInput<w>_v1`; False for any other text."""
    try:
        defs = json.loads(program)["definitions"]
    except (ValueError, KeyError, TypeError):
        return False
    return isinstance(defs, dict) and not any(_REGISTERED_INPUT.fullmatch(k) for k in defs)


#: `OneStageZk.WORD`: the bound on each length a session's statement holds, which writes every length in an 8-byte word
WORD = 1 << 64
#: `OneStageZk.FITS`, v2's refusal of a length at `WORD` or more
FITS = "2^64 bytes or more, past the 8-byte word a session's statement holds each length in"


def _printed_number(text: str) -> str:
    """``JsonNumber.toString`` of the number Lean's parser reads from the JSON number ``text``."""
    sign = -1 if text.startswith("-") else 1
    body, _, exp = text.lstrip("-").lower().partition("e")
    whole, _, frac = body.partition(".")
    m, e = sign * int(whole + frac), len(frac)
    if exp:
        k = int(exp)
        if k >= 0:
            m, e = m * 10 ** max(0, k - e), max(0, e - k)
        else:
            e -= k
    if e == 0:
        return str(m)
    sg, m = ("" if m >= 0 else "-"), abs(m)
    x = 9 + len(str(m)) - e
    x = x if x < 0 else 0
    ep = 10 ** (e + x)
    left = str(m // ep)
    if m % ep == 0 and x == 0:
        return sg + left
    right = str(ep + m % ep)[1:].rstrip("0")
    return f"{sg}{left}.{right}" + (f"e{x}" if x else "")


def _printed_string(s: str) -> str:
    escapes = {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r"}
    return '"' + "".join(escapes.get(c) or (f"\\u{ord(c):04x}" if ord(c) < 0x20 else c) for c in s) + '"'


def printed(v: Any) -> str:
    """`Flock.RecordText.printT` of ``v`` (what ``json.loads`` returns), the text ``Json.compress`` prints: keys in
    code-point order and no whitespace; ``"``, ``\\``, newline and carriage return escaped and every other character
    below U+0020 as ``\\u00xx``; a number as ``JsonNumber.toString`` prints it. It is ``verity_flock.record_text``'s
    ``session_text``, which core doesn't import."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"printed: {v} is not a JSON number")
        return _printed_number(repr(v))
    if isinstance(v, str):
        return _printed_string(v)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(printed(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(f"{_printed_string(k)}:{printed(v[k])}" for k in sorted(v)) + "}"
    raise ValueError(f"printed: a {type(v).__name__} is not JSON")


def printed_size(v: Any) -> int:
    """`OneStageZk.printedSize`: a JSON value's length as a session's statement holds it, `printed`'s in UTF-8."""
    return len(printed(v).encode())


def fits_why(limit: int, circuit: bytes, public: bytes, tables: Sequence[tuple[str, bytes]], program: bytes | None,
             partition: Any, registered: Any) -> str | None:
    """`OneStageZk.fitsWhy`: what a session's statement would hold at ``limit`` bytes or more, the first of the circuit
    and public files, the number of lookup tables, a lookup table's name (in UTF-8) or bytes, the program, the partition
    object and the registered reads (`printed_size`, each None for none); None when every length is under ``limit``."""
    if limit <= len(circuit):
        return "the circuit file"
    if limit <= len(public):
        return "the public file"
    if limit <= len(tables):
        return "the number of lookup tables"
    if any(limit <= len(name.encode()) or limit <= len(b) for name, b in tables):
        return "a lookup table"
    if program is not None and limit <= len(program):
        return "the program"
    if partition is not None and limit <= printed_size(partition):
        return "the partition object"
    if registered is not None and limit <= printed_size(registered):
        return "the registered reads"
    return None


def draw_fits_why(limit: int, draw: Any) -> str | None:
    """`OneStageZk.sampleFitsWhy`: ``"the draw"`` when a session's statement would hold its record's sample ``draw`` (None
    for none) at ``limit`` bytes or more."""
    return "the draw" if draw is not None and limit <= printed_size(draw) else None


def population_why(indices: Sequence[int], strata: Sequence[tuple[str, Sequence[Sequence[int]]]]) -> str | None:
    """`Spot.FilePart.why`: why the docket's population is not the public file's units in order, None
    when it is. `strata` is the docket's over the verifier's program and query (`partition.strata`), read stratum
    by stratum, and must be its proof blocks ``0, 1, …`` in order; `indices` (the public file's ``units.indices``) must be
    exactly those, so the file's unit ``u`` is the docket's proof block ``u``."""
    at = 0
    for _t, rs in strata:
        for lo, hi in rs:
            if lo != at or hi < lo:
                return "partition: the registration's strata, read stratum by stratum, are not its units in order"
            at = hi
    if list(indices) != list(range(at)):
        return f"partition: the public file's {len(indices)} units are not the registration's {at}, in order"
    return None


def lookups_of(meta: Any) -> list[str]:
    """`OneStageZk.lookupsOf` on a circuit file whose META is `meta`: the pinned MUFU tables named under its ``lookups``,
    in `MUFU_SHA512`'s order, none where it has none; ValueError where ``lookups`` is neither an object nor null."""
    v = meta.get("lookups") if isinstance(meta, dict) else None
    if v is None:
        return []
    if not isinstance(v, dict):
        raise ValueError("META: lookups is not an object")
    return [n for n in MUFU_SHA512 if n in v]


def circuit_lookups(circuit: Path) -> list[str]:
    """`lookups_of` on the circuit file `circuit`'s META, read off its second line alone (`Flock.HmOut.metaOf`)."""
    with open(circuit, "rb") as f:
        f.readline()
        line = f.readline().decode().rstrip()
    if not line.startswith("META "):
        raise ValueError("circuit: no META line")
    return lookups_of(json.loads(line[5:]))


def tables_match_why(tables: Sequence[str], lookups: Sequence[str]) -> str | None:
    """`OneStageZk.tablesMatch`: the refusal of the lookup tables a statement is handed, by name `tables`, when they are
    not `lookups`, its circuit's (`lookups_of`), naming both; None when they are."""
    if list(tables) == list(lookups):
        return None
    names = ", ".join
    return (f"{NO_TABLES}: the circuit looks up {names(lookups) or 'none'}, the blobs hold {names(tables) or 'none'}")


def tables_why(lookups: Sequence[str], digests: Iterable[str]) -> str | None:
    """`OneStageZk.tablesHeld`: the v2 audit's refusal of blobs whose SHA-512s (lowercase hex) are `digests` when the
    pinned MUFU tables among them (`MUFU_SHA512`, in its order) are not `lookups`, the registered circuit's
    (`lookups_of`), naming both; None when they are."""
    held = set(digests)
    return tables_match_why([n for n, d in MUFU_SHA512.items() if d in held], lookups)


def own_tables(lookups: Sequence[str], digests: Iterable[str]) -> list[str]:
    """`OneStageZk.ownTables`, by name: a member's own lookup tables among blobs whose SHA-512s (lowercase hex) are
    `digests`, the pinned MUFU tables among them (`MUFU_SHA512`, in its order) that `lookups` name, its circuit's
    (`lookups_of`). A pinned table the blobs hold for another member's circuit is none of them; one of `lookups` the
    blobs do not hold is missing from them, which `tables_match_why` refuses."""
    held = set(digests)
    return [n for n, d in MUFU_SHA512.items() if d in held and n in lookups]


def proved_units(public: Path, session: Path) -> tuple[int, ...]:
    """The global proof blocks a session proved: the public file's proof block indices (``units.indices``) of the instances its record
    sampled (``unit_draw.units``), ascending. Read only off a session the verifier accepted."""
    indices = _header(public)["units"]["indices"]
    drawn = json.loads((session / "session.json").read_text())["unit_draw"]["units"]
    return tuple(sorted(indices[i] for i in drawn))


def units_exactly(units: Sequence[Sequence[int]], indices: Sequence[int]) -> bool:
    """Whether a public file's ``units.indices`` are exactly the units of the ``[lo, hi)`` ranges `units`, ascending, so
    slot ``i`` of its table is the ``i``-th of them (`PublicUnits.exactly`): the per-member form holds each statement's
    file to its member's units, and ``[[0, n]]`` is a docket's population in order."""
    return list(indices) == [u for lo, hi in sorted(map(tuple, units)) for u in range(lo, hi)]


def not_zk_why(line: Mapping[str, Any]) -> str | None:
    """Why the verifier's verdict line is no zero-knowledge session's, or None: a line that does not say ``zk`` true
    (Flock's verifier prints it on every line) is refused, accepted or not."""
    if line.get("zk") is True:
        return None
    return f"session {line.get('session')}'s verdict line says zk {json.dumps(line.get('zk'))}, not true: no --zk session's"


def drawn_block(program: bytes, partition: Mapping[str, Any], draw: Mapping[str, Any], public: Path,
                decoded: OC.Decoded | None = None) -> tuple[int, list[tuple[int, int]]] | tuple[None, tuple[str, str]]:
    """The proof block of the partition object a statement's proof is of, and its units, taken from the challenger's
    `draw` (`service.select_draw`, `coins.draw_first`: its ``units`` and ``closure`` over the partition's blocks): the drawn
    block whose units the proof's public file names exactly (`units_exactly`). None and the refusal otherwise
    (``circuit``: a drawn block the partition object has not, `one_circuit.block`; ``proof``: a public file naming no
    unit, or one that is no natural number, a float or a bool included; ``undrawn``: a draw that sampled no block
    holding its first unit, or a drawn block holding it whose units are not exactly the file's)."""
    ks = [*draw.get("units", ()), *draw.get("closure", ())] if isinstance(draw, Mapping) else []
    try:
        decoded = decoded or OC.Decoded(program)
        blocks = {k: OC.block(program, partition, k, decoded) for k in ks}
    except OC.Refused as e:
        return None, ("circuit", str(e))
    try:
        indices = list(_header(public)["units"]["indices"])
        u = indices[0]
    except (OSError, ValueError, LookupError, TypeError):
        return None, ("proof", "the statement's public file names no unit")
    if not all(map(_nat, indices)):
        bad = next(x for x in indices if not _nat(x))
        return None, ("proof", f"the statement's public file names unit {json.dumps(bad)}, which is no natural number")
    k = next((k for k, units in sorted(blocks.items()) if any(lo <= u < hi for lo, hi in units)), None)
    if k is None:
        return None, ("undrawn", f"the proof is of the proof block holding unit {u}, which the draw did not sample "
                                 f"(its blocks {sorted(blocks)})")
    if not units_exactly(blocks[k], indices):
        return None, ("undrawn", f"the statement's public file names units {indices}, not exactly drawn block {k}'s "
                                 f"{[list(r) for r in blocks[k]]}")
    return k, blocks[k]


def verify(proof: Proof, *, program: bytes, partition: Mapping[str, Any], draw: Mapping[str, Any],
           registered: Mapping[str, Any] | None = None,
           strata: Sequence[tuple[str, Sequence[Sequence[int]]]] | None = None,
           inputs: Callable[[int], Any] | None = None,
           ) -> tuple[list[Finding] | None, tuple[str, str] | None]:
    """Each of `proof`'s sessions verified by Flock's Lean verifier against the verifier's own `program` (C, its
    descriptor's bytes), `partition` and `registered` file (theta's registration; None only for a program that reads no
    registered value, such as N2's, whose proof block opens the committed value inside its circuit), on the circuit file
    of the drawn proof block, the block of the partition object `draw` sampled that the proof is of (`drawn_block`),
    derived from C and the partition (`one_circuit.restrict`), never `proof`'s own circuit file; the findings in session
    order, each naming that block (``block``), and None, or None and the refusal (``registered``: `registered` is None
    and the program reads registered values, `reads_registered`; ``undrawn``: the proof is of a block `draw` did not
    sample, or its public file's units are not exactly the drawn block's; ``inputs``: `inputs`, a scheme's reading of
    the drawn block's public inputs from its committed records, called with the drawn block's
    index, raises a ValueError; ``circuit``: C and the partition give no drawn block or no C_S of it, or Lean's
    lowering of C_S refuses, `one_circuit.lower`;
    ``proof``: a file is missing or the
    verifier's answer is not one verdict a session naming the
    partition object's program; ``form``: the verifier accepted a session whose files `ks_form_why` refuses). A session
    the verifier rejects is a finding, not a refusal, and so is one it accepts on a circuit `template_why` refuses,
    rejected with that reason as v2 rejects it.

    With `strata`, the docket's over `program` and the partition's query (`partition.strata`), the statement is
    the one-stage audit's, whose public file holds the whole population: a session the verifier accepts on a circuit
    `template_why` passes is rejected with `population_why`'s reason where the public file's units are not that
    population in order, as the v2 audit's own run rejects it, last (`Spot.FilePart.check`). Without it (a
    statement of some of a population's proof blocks, as the network attester's) the public file's units are the verifier's
    own.

    The ``registered`` refusal is the verifier's own: without ``--registered``, Lean refuses a statement's registered
    reads only where its public file, which the prover writes, names them (`Registered.check`), and doesn't read a
    `RegisteredInput` as demanding them, so a prover that commits such a port's rows afresh and names none would be
    checked against no registered root. A run input that only the caller's registered file reads registered (the
    served RoPE statement's ``cs``) is the caller's to state: without the file, Lean holds it a run input, and refuses a
    session whose public file names its reads or whose tables share its registered rows."""
    if registered is None and reads_registered(program):
        return None, ("registered", "the program reads registered values (a RegisteredInput among its definitions) and "
                                    "the verifier holds no registered reads: verify with its own registered file")
    decoded = OC.Decoded(program)
    k, units = drawn_block(program, partition, draw, proof.public, decoded)
    if k is None:
        return None, units
    if inputs is not None:
        try:
            inputs(k)
        except ValueError as e:
            return None, ("inputs", f"drawn block {k}: {e}")
    try:
        derived = OC.restrict(program, partition, units, decoded)
    except OC.Refused as e:
        return None, ("circuit", str(e))
    if not (proof.public.is_file() and proof.sessions and all((s / "session.json").is_file() for s in proof.sessions)):
        return None, ("proof", "the statement's proof files are not a public file and at least one session directory "
                               "holding its session.json")
    try:
        base = OC.lower(derived, partition, lower_executable, cut=OC.cuts(program, partition, decoded),
                        formed=OC.formed_in(decoded, partition, units))
    except (OC.NotDerived, OC.Refused) as e:
        return None, ("circuit", str(e))
    name = statement()
    with tempfile.TemporaryDirectory(prefix="verity-proofs-") as d:
        files = {"circuit": Path(d, "circuit.txt"), "program": Path(d, "program.json"),
                 "partition": Path(d, "partition.json"), "registered": Path(d, "registered.json")}
        files["circuit"].write_bytes(base)
        if name == TMPL_STATEMENT and (why := tmpl_why(files["circuit"])) is not None:
            return None, ("circuit", why)
        files["program"].write_bytes(program)
        files["partition"].write_text(json.dumps(partition))
        reads = []
        if registered is not None:
            files["registered"].write_text(json.dumps(registered))
            reads = ["--registered", str(files["registered"])]
        p = LC.run([str(executable()), "verify", "--zk", "--statement", name, "--circuit", str(files["circuit"]),
                    "--public", str(proof.public), "--partition", str(files["partition"]),
                    "--program", str(files["program"]), *reads,
                    "--session", *map(str, proof.sessions)], env=verifier_env(), ok_statuses=LC.VERIFY_EXITS)
        lines = [json.loads(x[len("VERDICT "):]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
        accepted = any(v.get("accepted") is True for v in lines)
        form = ks_form_why(files["circuit"], proof.public) if accepted else None
        template = template_why(files["circuit"]) if accepted else None
    if [v.get("session") for v in lines] != [str(s) for s in proof.sessions]:
        return None, ("proof", f"the verifier answered no verdict for each session in order (exit {p.returncode}): "
                               f"{(p.stderr or p.stdout)[-400:]}")
    if (why := next(filter(None, map(not_zk_why, lines)), None)) is not None:
        return None, ("proof", why)
    want = partition.get("program")
    population = (population_why(_header(proof.public)["units"]["indices"], strata)
                  if accepted and strata is not None else None)
    out = []
    for s, v in zip(proof.sessions, lines):
        ok, why = v.get("accepted") is True, str(v.get("why", ""))
        if ok and form is not None:
            return None, ("form", f"the verifier accepted session {s}, whose files its mirror refuses "
                                  f"(`ks_form_why`): {form}")
        if ok and template is not None:
            ok, why = False, template
        if ok and population is not None:
            ok, why = False, population
        if ok and v.get("program_sha512") != want:
            return None, ("proof", f"session {s}'s verdict names program {v.get('program_sha512')}, not the partition "
                                   f"object's {want}")
        out.append(Finding(str(s), ok, why, v.get("program_sha512"), v.get("registered") or {},
                           proved_units(proof.public, s) if ok else (), v.get("openings"), k))
    return out, None


#: the one-stage audit on the proofs themselves (`Spot.OneStageZk`), which runs `verify`'s verifier on blobs
V2 = ONE_STAGE_PROTOCOL_V2
V2_EXE = "spot-one-stage-zk"


def v2_executable() -> Path:
    """`spot-one-stage-zk`, built (`built`)."""
    return built(V2_PACKAGE, V2_EXE)


def order_why(params: Mapping[str, Any], record: Mapping[str, Any], evidence: Mapping[str, Any]) -> str | None:
    """`OneStageZk.orderHeld`'s refusal, with its reason: the challenger session's release log in v1's `evidence`
    (``coins``' ``session``'s ``log``), after its head, not the sample, then the link, then the other rounds
    (`challenger.sample_link_first`) at the parameters' challenger family (`params`' ``challenger``'s ``label``) and the
    docket `record`'s digest; None when it is. v2 answers it under ``proofs:`` before any session's record
    is read, once the log's coin trees parse (`challenger.coin_trees_of`), which it refuses first."""
    log = evidence["coins"]["session"]["log"]
    if sample_link_first(params["challenger"]["label"], bytes.fromhex(registration_digest(record)), log[1:]):
        return None
    return f"coins: {SAMPLE_LINK_FIRST}"


def _canonical(x: Any) -> bytes:
    """`x` in the canonical JSON `Core.canonicalJson` reads, for the integers, strings, lists and objects of ASCII keys
    these inputs hold."""
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def evidence_v2(proof: Proof, names: Sequence[str] | None = None) -> tuple[dict, list[Path]]:
    """The v2 evidence's ``proofs`` for `proof`'s files, and the blobs that hold them: per session (named `names`, the v1
    evidence's ``sessions``; by default its directory's name), its ``session.json``'s SHA-512 and each rep file's
    (``*.bin``). The blobs are the circuit and public files, which the parameters name by the auditor's digests, and each
    session's files."""
    names = [s.name for s in proof.sessions] if names is None else list(names)
    if len(names) != len(proof.sessions):
        raise ValueError(f"{len(names)} names for {len(proof.sessions)} sessions")
    sessions, blobs = [], [proof.circuit, proof.public]
    for name, s in zip(names, proof.sessions):
        reps = sorted(s.glob("*.bin"))
        sessions.append({"name": name, "record": file_sha512(s / "session.json"),
                         "reps": {r.name: file_sha512(r) for r in reps}})
        blobs += [s / "session.json", *reps]
    return {"sessions": sessions}, blobs


#: what the v2 executable's ``members`` says of what it prints (`OneStageZk.PARSED`): no check of the audit ran
PARSED = "parsed, not checked"


@dataclass(frozen=True)
class Member:
    """One member of a window in v2's per-member form (`OneStageZk.membersOf`): its name, its statement's files (`Proof`,
    with no session for a member the sample missed), the SHA-512 the auditor committed for its public file before any
    coin, theta's registered file for it, its units' ranges ``[lo, hi)`` of the docket's population, ascending, and its
    circuit's ``vus_per_block``. Its circuit file is C_S, which the verifier derives (`one_circuit.restrict`)."""
    name: str
    proof: Proof
    public_sha512: str
    registered: Mapping[str, Any] | None
    units: tuple[tuple[int, int], ...]
    vus_per_block: int


def members_v2(members: Sequence[Member],
               names: Mapping[str, Sequence[str]] | None = None) -> tuple[dict, dict, list[Path]]:
    """v2's per-member form of ``proofs`` for a window's `members`, every one of them: the parameters'
    (``{"members": {name: {public, registered, block, vus_per_block}}}``, ``block`` its rank by first unit, its block of
    the partition object's ``blocks``), the evidence's
    (``{"statements": [{member, sessions}]}``: one statement for each member with a session, in the members' order by
    first unit, its session as `evidence_v2` names it, under `names`' for that member), and the blobs: the
    statements' files, then the circuit and public files of each member with no session, since the form's check reads
    every member's files, sampled or not (`OneStageZk.membersFiles`). The case's blobs hold besides them each pinned
    lookup table a member's circuit looks up, any member's (`own_tables`). Two members of one name, a member of no unit
    or one with several sessions raise: a statement is exactly one session on live coins. `Spot.OneStageZk` checks a
    case of this form by `checkMembersAt` (`members.check`); `members_parsed` holds these keys to its parsers."""
    if len({m.name for m in members}) != len(members):
        raise ValueError("two members have one name: a member's name is its own")
    for m in members:
        if not m.units:
            raise ValueError(f"member {m.name} has no unit")
        if len(m.proof.sessions) > 1:
            raise ValueError(f"member {m.name} has {len(m.proof.sessions)} sessions: a statement is exactly one session "
                             f"on live coins, never retried")
    rank = {m.name: k for k, m in enumerate(sorted(members, key=lambda m: min(lo for lo, _ in m.units)))}
    params = {"members": {m.name: {"public": m.public_sha512,
                                   "registered": None if m.registered is None else dict(m.registered),
                                   "block": rank[m.name], "vus_per_block": m.vus_per_block}
                          for m in members}}
    statements, blobs = [], []
    for m in sorted((m for m in members if m.proof.sessions), key=lambda m: min(lo for lo, _ in m.units)):
        pr, files = evidence_v2(m.proof, None if names is None else names[m.name])
        statements.append({"member": m.name, "sessions": pr["sessions"]})
        blobs += files
    blobs += [f for m in members if not m.proof.sessions for f in (m.proof.circuit, m.proof.public)]
    return params, {"statements": statements}, blobs


def members_parsed(params: Mapping[str, Any], evidence: Mapping[str, Any]) -> tuple[dict | None, str | None]:
    """What v2's per-member parsers read of `params` and `evidence` (`V2_EXE`'s ``members``,
    `OneStageZk.membersParsed`): its object, ``{"members", "parsed", "statements"}`` with ``parsed`` the words `PARSED`,
    and None; or None and the refusal. The command reads the two objects alone and runs no check of the audit, so its
    answer is no verdict."""
    with tempfile.TemporaryDirectory(prefix="verity-proofs-v2-") as d:
        paths = (Path(d, "params.json"), Path(d, "evidence.json"))
        for path, x in zip(paths, (params, evidence)):
            path.write_bytes(_canonical(x))
        p = LC.run([str(v2_executable()), "members", *map(str, paths)], ok_statuses=LC.V2_EXITS)
    try:
        got = json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise LeanError(f"{V2_EXE} members answered no line (exit {p.returncode}): {p.stderr[-400:]}") from None
    if "refused" in got:
        return None, got["refused"]
    if got.get("parsed") != PARSED:
        raise LeanError(f"{V2_EXE} members answered a line that does not say {PARSED!r}: {sorted(got)}")
    return got, None


def verdicts_v2(proof: Proof, *, program: Mapping[str, Any], partition: Mapping[str, Any],
                registered: Mapping[str, Any] | None = None, tables: Sequence[Path] = ()) -> tuple[list[Finding] | None, str | None]:
    """`verify`, by the v2 audit's own run of the verifier on `proof`'s files as blobs
    (`spot-one-stage-zk verdicts`): the findings in session order (``session`` the directory's name) and None,
    or None and Lean's refusal. `program` is the descriptor as JSON; `tables`, the MUFU tables a statement with lookups
    reads, blobs beside the files (`OneStageZk.tablesOf` finds them by SHA-512)."""
    pr, blobs = evidence_v2(proof)
    blobs += list(tables)
    own = {"partition": dict(partition), "program": dict(program),
           "public": file_sha512(proof.public), "registered": None if registered is None else dict(registered)}
    with tempfile.TemporaryDirectory(prefix="verity-proofs-v2-") as d:
        paths = (Path(d, "own.json"), Path(d, "proofs.json"))
        paths[0].write_bytes(_canonical(own))
        paths[1].write_bytes(_canonical(pr))
        p = LC.run([str(v2_executable()), "verdicts", *map(str, paths), *map(str, blobs)], ok_statuses=LC.V2_EXITS)
    try:
        got = json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise LeanError(f"{V2_EXE} verdicts answered no line (exit {p.returncode}): {p.stderr[-400:]}") from None
    if "refused" in got:
        return None, got["refused"]
    return [Finding(v["session"], v["accepted"] is True, v["why"], v.get("program_sha512"), v.get("registered") or {},
                    tuple(v["units"])) for v in got["verdicts"]], None


__all__ = ["EXE", "FITS", "MUFU_SHA512", "NO_ROW_WHY", "NO_TABLES", "PACKAGE", "SCOPE_WHY", "STATEMENT", "TABLES_WHY",
           "SIZE_CLASS_MAX", "TMPL_SWITCH", "V2", "V2_EXE", "WIDE_WHY", "WORD", "Finding", "LeanError", "Proof", "circuit_lookups",
           "draw_fits_why", "drawn_block", "evidence_v2", "executable", "file_sha512", "fits_why", "ks_form_why",
           "lookups_of", "not_zk_why", "order_why", "own_tables", "population_why", "printed", "printed_size",
           "proved_units", "reads_registered", "tables_match_why", "tables_why", "template_why", "units_exactly",
           "v2_executable", "verdicts_v2", "verify"]
