"""One vocabulary (AGENTS.md "Lean"), over the whole repository's Python and Lean: each concept in ``CONCEPTS`` has one
canonical object, and every other declaration of it is listed, with its reason and owner.

What counts is a type-forming declaration whose whole name, or last component, is the concept: a Python `class`, or a Lean
`structure`, `inductive`, `class`, `abbrev`, `opaque`, or `def` whose type ends in `Type` or `Sort` (or `Prop`, for
``PROPOSITIONS``). `python tests/test_lean_vocabulary.py` lists every hit and what the guard makes of it.

Each hit is one of:

- **canonical**: ``CANONICAL`` names, per concept, at most three declarations: one under `Security/Definitions/Core/` or
  `Security/Definitions/Hash/`, one in the code-Lean package that owns it, and one Python class. A concept with no
  canonical yet says so, with an empty entry.
- **a mirror**: a Python class that ``MIRRORS`` holds to a Lean declaration through the `verity_difftest` port (a
  `[[lean_exe]]` of a tracked lakefile) that tests them against each other. The pair counts once, as its Lean.
- **reading the canonical**: an `abbrev` whose body is a canonical declaration, or a list, array or option of it, or a
  pair of an id and it.
- **a protocol's slot**: a type a `Core.Protocol` value names in its `Params`, `Commitment`, `Evidence` or `Public` field
  (read from its `Field := X` lines), one per protocol by design.
- **known**: everything else, listed in ``KNOWN`` as ``"path::Qualified.Name": "reason (owner)"``, where the reason is
  "duplicate of X: becomes …" or "homonym, what it is: rename …", and the owner is the registry's owner of the path
  (`.agents/skills/using-slack/registry.json`).

An unlisted hit fails, and so does a canonical that is no longer a hit. A ``KNOWN`` line or ``MIRRORS`` entry that no
longer matches a hit (its declaration went, or became a canonical, a mirror or exempt) is stale, and doesn't fail, so a
change that removes a duplicate needn't touch this file. `python tests/test_lean_vocabulary.py` lists the stale entries
under "stale (delete with --prune)", and `python tests/test_lean_vocabulary.py --prune` rewrites ``KNOWN`` without its
stale lines and sets ``KNOWN_MAX`` to the new count, so no later line takes a freed slot unreviewed (a stale ``MIRRORS``
entry is deleted by hand). Only lean prunes, so two prunes don't race on ``KNOWN_MAX``. ``KNOWN`` only shrinks, by count:
it holds at most ``KNOWN_MAX`` lines. A new line needs lean's review and raises ``KNOWN_MAX`` in the same change; lowering
it is welcome in any change.

The scan reads every tracked `.py` and `.lean` file (`git ls-files`, so `.lake`, `.venv` and build trees are never read)
except:

- `archive/`: kept, not developed (`archive/AGENTS.md`), so nothing there is renamed or merged;
- tests (a `tests/` directory, `test_*.py`, `conftest.py`) and `scripts/` directories: fixtures and one-off drivers state
  their own small types, which no guarantee or protocol reads;
- difftest drivers (`*Difftest*.lean`, `*TestMain.lean`): each wraps its port's inputs for the JSON server and is held to
  the Python by the difftest itself."""
from __future__ import annotations

import functools
import json
import re
import subprocess
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ".agents/skills/using-slack/registry.json"
SPEC = ("Security/Definitions/Core/", "Security/Definitions/Hash/")
EXCLUDED_PREFIXES = ("archive/",)
EXCLUDED_DIRS = ("tests", "scripts")
EXCLUDED_FILES = re.compile(r"(^test_.*\.py|^conftest\.py|Difftest.*\.lean|TestMain\.lean)$")
FIX = "see AGENTS.md 'One vocabulary' and this module's docstring"

#: The concepts held to one canonical object each.
CONCEPTS = ("Circuit", "Prog", "Program", "Gate", "Op", "Netlist", "Partition", "Coins", "CoinTree", "Draw", "Law", "Stmt",
            "Statement", "Witness", "Digest", "Commitment", "Merkle", "Collision", "Device", "InteractiveDevice", "Dag",
            "Transcript", "Hash", "Signature",
            "Site", "Layout", "Hardware", "Machine", "Node", "Card", "Link", "Registration", "Registry", "Period",
            "Window", "Session", "Block", "Verdict", "Outcome", "Answer", "Record", "Ledger", "Opening", "Challenger",
            "Certificate", "Store", "Placement")
#: The concepts that are propositions, so a `def … : Prop` of their name defines them too.
PROPOSITIONS = ("Collision",)
SLOT_FIELDS = ("Params", "Commitment", "Evidence", "Public")

D, CORE, FLOCK = "Security/Definitions/", "Security/Definitions/Core/", "verity/core/service/flock/Flock/"
PROTOCOL = "verity/core/protocols/interface/CoreProtocol/Protocol.lean"
REQ, REQ_PY = "verity/core/primitives/requirement/CircuitRequirement.lean", "verity/ml/circuits/requirement.py"
#: Each concept's canonical declarations: at most one in Core or Hash, one in code Lean and one in Python, and when a concept
#: has both code Lean and Python, ``MIRRORS`` holds the two together. An empty entry: no canonical yet, so every hit is known.
CANONICAL: dict[str, tuple[str, ...]] = {
    "Answer": (f"{PROTOCOL}::Core.Answer",),
    "Block": (f"{CORE}Blocks.lean::Verity.Block",),
    "Card": (f"{CORE}Accel/Card.lean::Verity.Accel.Card",),
    "Certificate": ("verity/core/protocols/network/lean/NetworkCheck/Check.lean::NetworkCheck.Check.Certificate",
                    "verity/core/protocols/certifier/certificate.py::Certificate"),
    "Challenger": ("verity/core/protocols/sampled_proofs/parties.py::Challenger",),
    "Circuit": (f"{CORE}Circuit.lean::Verity.Circuit", f"{FLOCK}Circuit.lean::Flock.Circuit"),
    "CoinTree": (),
    "Coins": (f"{CORE}Draw.lean::Verity.Coins", f"{FLOCK}Statement.lean::Flock.Coins"),
    "Collision": ("Security/Definitions/Hash/Basic.lean::Hash.Collision",),
    "Commitment": ("verity/ml/commitments/merkle.py::Commitment",),
    "Dag": (),
    "Device": (),
    "Digest": (f"{CORE}Digest.lean::Verity.Digest", f"{FLOCK}Hash.lean::Flock.Digest"),
    "Draw": (f"{CORE}Draw.lean::Verity.Draw", "verity/core/protocols/sampled_proofs/service.py::Draw"),
    "Gate": (),
    "Hardware": (),
    "Hash": (),
    "InteractiveDevice": (),
    "Law": (f"{CORE}Draw.lean::Verity.Law", f"{FLOCK}Draw.lean::Flock.Draw.Law",
            "verity/core/protocols/sampled_proofs/service.py::Law"),
    "Layout": (),
    "Ledger": (),
    "Link": (),
    "Machine": (f"{CORE}Accel/Machine.lean::Verity.Accel.Machine",),
    "Merkle": (),  # nothing in scope declares it
    "Netlist": ("verity/ml/private_circuits/universal.py::Netlist",),
    "Node": (),
    "Op": (f"{CORE}Boolean.lean::Verity.Boolean.Op",),
    "Opening": ("Security/Definitions/Hash/RowLeaf.lean::Hash.RowLeaf.Opening", "verity/ml/commitments/merkle.py::Opening"),
    "Outcome": (f"{PROTOCOL}::Core.Outcome",),
    "Partition": (f"{CORE}Partition.lean::Verity.Partition",),
    "Period": (),
    "Placement": (f"{REQ}::CircuitRequirement.Placement", f"{REQ_PY}::Placement"),
    "Prog": (f"{CORE}Accel/Machine.lean::Verity.Accel.Machine.Prog", f"{FLOCK}Program.lean::Flock.Program.Prog"),
    "Program": ("verity/ml/circuits/program.py::Program",),
    "Record": (),
    "Registration": (),
    "Registry": (),
    "Session": (),
    "Signature": ("verity/core/primitives/lms/lean/Lms/Verify.lean::Lms.Signature", "verity/ml/crypto/lms.py::Signature"),
    "Site": (),
    "Statement": (),
    "Stmt": (f"{FLOCK}Statement.lean::Flock.Stmt",),
    "Store": (f"{REQ}::CircuitRequirement.Store", f"{REQ_PY}::Store"),
    "Transcript": (f"{CORE}Circuit.lean::Verity.Circuit.Transcript", f"{FLOCK}Transcript.lean::Flock.Transcript"),
    "Verdict": (f"{PROTOCOL}::Core.Verdict",),
    "Window": (),
    "Witness": (f"{CORE}TC/Relation.lean::Verity.TC.Relation.Witness",),
}

RA, RA_PY = "verity/core/protocols/remote_bandwidth/lean/RemoteAudit/", "verity/core/protocols/remote_bandwidth/audit.py"
MC, MC_PY = "verity/core/devices/memory_challenger/lean/MemoryChallenger/", "verity/core/protocols/pous/"
#: A Python class held to a code-Lean declaration by a `verity_difftest` port, ``python key: (Lean key, port, reason)``: the
#: pair counts once, as its Lean. The port is a `[[lean_exe]]` of a tracked lakefile.
MIRRORS: dict[str, tuple[str, str, str]] = {
    "verity/core/protocols/certifier/certificate.py::Certificate":
        ("verity/core/protocols/network/lean/NetworkCheck/Check.lean::NetworkCheck.Check.Certificate", "network-check",
         "the network check's protocol port decodes the certificates the Python issues"),
    "verity/core/protocols/certifier/grid.py::Record":
        ("verity/core/devices/certifier/lean/CertifierDevice/Certifier.lean::CertifierDevice.Record",
         "certifier-device-difftest", "the certifier device's port replays the Python grid's records"),
    "verity/core/protocols/sampled_proofs/service.py::Law": (f"{FLOCK}Draw.lean::Flock.Draw.Law", "sampled-proofs-difftest",
                                                             "the service's port parses the Python's laws"),
    "verity/ml/crypto/lms.py::Signature": ("verity/core/primitives/lms/lean/Lms/Verify.lean::Lms.Signature", "lms-difftest",
                                           "the LMS port verifies the Python's signatures"),
    "verity/core/protocols/answer.py::Answer": (f"{PROTOCOL}::Core.Answer", "core-protocol-example",
                                               "the core interface's example port writes the Python's answers byte for byte"),
    "verity/core/protocols/answer.py::Verdict": (f"{PROTOCOL}::Core.Verdict", "core-protocol-example",
                                                "the core interface's example port, on each verdict"),
    f"{REQ_PY}::Store": (f"{REQ}::CircuitRequirement.Store", "requirement-difftest", "the requirement's port"),
    f"{REQ_PY}::Placement": (f"{REQ}::CircuitRequirement.Placement", "requirement-difftest", "the requirement's port"),
    f"{RA_PY}::Opening": (f"{RA}Window.lean::RemoteAudit.Opening", "remote-audit-difftest", "the remote audit's port"),
    f"{RA_PY}::Registration": (f"{RA}Window.lean::RemoteAudit.Registration", "remote-audit-difftest", "the remote audit's port"),
    f"{RA_PY}::Registry": (f"{RA}Registry.lean::RemoteAudit.Registry", "remote-audit-difftest", "the remote audit's port"),
    f"{RA_PY}::Verdict": (f"{RA}Window.lean::RemoteAudit.Verdict", "remote-audit-difftest", "the remote audit's port"),
    f"{MC_PY}memory_challenger.py::CoinTree": (f"{MC}Records.lean::MemoryChallenger.CoinTree", "memory-challenger-difftest",
                                               "the memory challenger's port"),
    f"{MC_PY}memory_challenger.py::Verdict": (f"{MC}Verdict.lean::MemoryChallenger.Verdict", "memory-challenger-difftest",
                                              "the memory challenger's port"),
    f"{MC_PY}streamed_wipe.py::Verdict": (f"{MC}StreamedWipe.lean::MemoryChallenger.StreamedWipe.Verdict",
                                          "memory-challenger-difftest", "the memory challenger's port, for the streamed wipe"),
    f"{MC_PY}run.py::Session": ("verity/core/protocols/pous/check/PousCheck/Run.lean::PousCheck.Run.Session", "pous-run",
                                "the PoUS run's protocol port reads the sessions the Python records"),
}


@dataclass(frozen=True)
class Hit:
    key: str            # "path::Qualified.Name"
    line: int
    keyword: str        # the Lean keyword, or "class" for Python
    body: str = ""      # an abbrev's body, whitespace collapsed
    scopes: tuple = ()  # the Lean namespaces it sits in, and the file's `open`s, to resolve its body's names
    opens: tuple = ()


_ALT = "|".join(sorted(CONCEPTS, key=len, reverse=True))
_NAME = rf"((?:[^\s(\[{{:]*\.)?(?:{_ALT}))(?![\w'.!?])"
_KEYWORD = r"(structure|class\s+inductive|class|inductive|abbrev|opaque|def)"
_MODS = r"(?:(?:private|protected|public|noncomputable|partial|unsafe|nonrec|meta)\s+)*"
# One pass over every file: a leading `\b` or `^` here costs Python's re its literal-prefix scan (2.5x slower).
_CANDIDATE = re.compile(rf"{_KEYWORD}\s+{_NAME}")
_PY_CANDIDATE = re.compile(rf"class\s+(?:{_ALT})\b")
_PY_HEADER = re.compile(r"^([ \t]*)(?:async[ \t]+)?(class|def)[ \t]+(\w+)", re.M)
_DECL = re.compile(rf"^[ \t]*(?:@\[[^\]\n]*\][ \t]*)*{_MODS}{_KEYWORD}\s+{_NAME}", re.M)
_SCOPE = re.compile(rf"^[ \t]*(?:{_MODS}(namespace|section)\b[ \t]*(\S*)|(mutual)\b|end\b)", re.M)
_OPEN = re.compile(r"^[ \t]*open\s+((?:[\w.]+[ \t]+)*[\w.]+)", re.M)
_PROTOCOL = re.compile(r"^[ \t]*(?:@\[[^\]\n]*\][ \t]*)*" + _MODS + r"(?:def|abbrev)\s+[^\n]*:\s*(?:Core\.)?Protocol\s+where\b",
                       re.M)
_FIELD = re.compile(rf"^[ \t]+({'|'.join(SLOT_FIELDS)})\s*:=\s*([\w.']+)[ \t]*$", re.M)
_TOP = re.compile(r"^\S", re.M)
_HEADER_TOKEN = re.compile(r":=|\bwhere\b|\||:|[(\[{⦃⟨)\]}⦄⟩]")
_SORT = re.compile(r"(?:^|→|->)\s*(?:Type|Sort)\b(?![\s\S]*(?:→|->))")
_SORT_OR_PROP = re.compile(r"(?:^|→|->)\s*(?:Type|Sort|Prop)\b(?![\s\S]*(?:→|->))")
_CODE_TOKEN = re.compile(r'/-|--[^\n]*|(?<!\')"(?:[^"\\]|\\.)*"', re.S)
_COMMENT_TOKEN = re.compile(r"/-|-/")


def strip_comments(text: str) -> str:
    """`text` with comments (nested `/- -/`, `--`) and string literals blanked, newlines kept, so offsets and lines stay."""
    keep, pos = [], 0
    while m := _CODE_TOKEN.search(text, pos):
        keep.append(text[pos:m.start()])
        end, depth = m.end(), m.group() == "/-"
        while depth and (c := _COMMENT_TOKEN.search(text, end)):
            depth, end = depth + (1 if c.group() == "/-" else -1), c.end()
        end = end if not depth else len(text)
        keep.append(re.sub(r"[^\n]", " ", text[m.start():end]))
        pos = end
    keep.append(text[pos:])
    return "".join(keep)


def _extent(text: str, start: int) -> int:
    """Where the declaration at `start` ends: the next line that starts in column 0."""
    nxt = _TOP.search(text, start)
    return nxt.start() if nxt else len(text)


def _header(text: str, start: int) -> tuple[int | None, int, str | None]:
    """From just past a declaration's name: where its type starts (past the top-level `:`), where the header ends, and
    the token that ends it (`:=`, `where`, `|`), or ``None`` when the declaration ends first (an `opaque`)."""
    depth, colon, stop = 0, None, _extent(text, start)
    for m in _HEADER_TOKEN.finditer(text, start, stop):
        t = m.group()
        if t in "([{⦃⟨":
            depth += 1
        elif t in ")]}⦄⟩":
            depth -= 1
        elif depth == 0 and t != ":":
            return colon, m.start(), t
        elif depth == 0:
            colon = m.end()
    return colon, stop, None


def returns_type(text: str, start: int, sort: re.Pattern = _SORT) -> bool:
    """Whether the declaration header from `start` (just past its name) ends its type in `Type …` or `Sort …` (`sort`)."""
    colon, end, _ = _header(text, start)
    return colon is not None and sort.search(text[colon:end]) is not None


def _body(text: str, start: int) -> str:
    """An abbrev's body: from its header's `:=` to the end of the declaration, whitespace collapsed."""
    _, end, token = _header(text, start)
    return " ".join(text[end + 2:_extent(text, end)].split()) if token == ":=" else ""


def lean_definitions(text: str) -> tuple[list[tuple[str, Hit]], list[tuple[str, str, tuple, tuple]]]:
    """The Lean source `text`'s ``(concept, Hit)`` pairs, keyed by qualified name alone, and its `Core.Protocol` fields:
    ``(field, name, scopes, opens)`` for each `Field := name` line of a protocol value."""
    if not (_CANDIDATE.search(text) or "Protocol where" in text):
        return [], []
    text = strip_comments(text)
    opens = tuple(n for m in _OPEN.finditer(text) for n in m.group(1).split())
    events = sorted([(m.start(), m) for m in _DECL.finditer(text)] + [(m.start(), m) for m in _SCOPE.finditer(text)]
                    + [(m.start(), m) for m in _PROTOCOL.finditer(text)], key=lambda e: e[0])
    scopes, out, fields = [], [], []
    for pos, m in events:
        if m.re is _SCOPE:
            if m.group(3) or m.group(1):
                scopes.append(m.group(2) if m.group(1) == "namespace" else "")
            elif scopes:
                scopes.pop()
            continue
        here = tuple(s for s in scopes if s)
        if m.re is _PROTOCOL:
            nxt = _TOP.search(text, m.end())
            for f in _FIELD.finditer(text, m.end(), nxt.start() if nxt else len(text)):
                fields.append((f.group(1), f.group(2), here, opens))
            continue
        keyword, name = m.group(1), m.group(2)
        concept = name.rpartition(".")[2]
        sort = _SORT_OR_PROP if concept in PROPOSITIONS else _SORT
        if keyword in ("def", "opaque") and not returns_type(text, m.end(), sort):
            continue
        qualified = name.removeprefix("_root_.") if name.startswith("_root_.") else ".".join(here + (name,))
        body = _body(text, m.end()) if keyword == "abbrev" else ""
        out.append((concept, Hit(qualified, text.count("\n", 0, pos) + 1, keyword.split()[0], body, here, opens)))
    return out, fields


def python_definitions(text: str) -> list[tuple[str, Hit]]:
    """The Python source `text`'s ``(concept, Hit)`` pairs: classes at module or class level (not local to a function),
    keyed by their qualified name (`Outer.Inner`)."""
    if not _PY_CANDIDATE.search(text):
        return []
    out, stack = [], []  # stack: (indent, keyword, name) of the enclosing `class` and `def` headers
    for m in _PY_HEADER.finditer(text):
        indent, keyword, name = len(m.group(1).expandtabs()), m.group(2), m.group(3)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if keyword == "class" and name in CONCEPTS and all(k == "class" for _, k, _ in stack):
            out.append((name, Hit(".".join([n for _, _, n in stack] + [name]), text.count("\n", 0, m.start()) + 1, "class")))
        stack.append((indent, keyword, name))
    return out


def tracked() -> list[str]:
    """The `.py` and `.lean` files the guard reads: tracked, and not excluded (this module's docstring)."""
    out = subprocess.run(["git", "ls-files", "-z", "--", "*.py", "*.lean"], cwd=ROOT, check=True,
                         capture_output=True).stdout
    paths = [p.decode() for p in out.split(b"\0") if p]
    return sorted(p for p in paths if not p.startswith(EXCLUDED_PREFIXES)
                  and not set(p.split("/")[:-1]) & set(EXCLUDED_DIRS) and not EXCLUDED_FILES.search(p.rpartition("/")[2]))


@dataclass
class Scan:
    hits: dict[str, list[Hit]]                        # concept: its hits, keyed "path::Qualified.Name"
    fields: list[tuple[str, str, str, tuple, tuple]]  # each protocol field: (path, field, name, scopes, opens)

    def __post_init__(self):
        self.index = defaultdict(lambda: defaultdict(list))
        for defs in self.hits.values():
            for h in defs:
                path, _, q = h.key.partition("::")
                self.index[_world(path)][q].append(h.key)

    def names(self, world: str) -> dict[str, list[str]]:
        """Each qualified name a Lean file of `world` can see: the keys of the hits that declare it."""
        return self.index[world]


def scan(texts: dict[str, str] | None = None) -> Scan:
    """The hits and protocol fields of `texts` (path: source), or of every file the guard reads."""
    hits, fields = defaultdict(list), []
    if texts is None:
        texts = {path: (ROOT / path).read_text(encoding="utf-8") for path in tracked()}
    for path, text in texts.items():
        if path.endswith(".py"):
            defs = python_definitions(text)
        else:
            defs, found = lean_definitions(text)
            fields += [(path, *f) for f in found]
        for concept, h in defs:
            hits[concept].append(Hit(f"{path}::{h.key}", h.line, h.keyword, h.body, h.scopes, h.opens))
    return Scan(dict(hits), fields)


def _world(path: str) -> str:
    """The Lean a name in `path` can see: `Security/` never requires the code's packages, and they never require it."""
    return "py" if path.endswith(".py") else "spec" if path.startswith("Security/") else "code"


def resolve(name: str, path: str, scopes: tuple, opens: tuple, s: Scan, skip: str = "") -> str | None:
    """The key of the hit a Lean name in `path` means inside the namespaces `scopes` with `opens` open: the innermost
    namespace's first, as Lean resolves it, then the opened ones, then the root; within that, the nearest package's.
    `skip` is the declaration whose body holds the name, which a non-recursive body can't mean."""
    name, parts = name.removeprefix("_root_."), ".".join(scopes).split(".") if scopes else []
    prefixes = [".".join(parts[:i]) for i in range(len(parts), 0, -1)]
    names = s.names(_world(path))
    for q in [f"{p}.{name}" for p in prefixes] + [f"{o}.{name}" for o in opens] + [name]:
        if keys := [k for k in names.get(q, ()) if k != skip]:
            return max(keys, key=lambda k: len(_common(k, path)))
    return None


def _common(key: str, path: str) -> str:
    a, b = key.partition("::")[0].split("/"), path.split("/")
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return "/".join(a[:n])


_CONTAINER = re.compile(r"^\(?\s*(?:(?:List|Array|Option)\s+|[\w.']+\s*×\s*)?\(?([\w.']+)(?:\s[^×→()]*(?:\([^×→()]*\))?)*\)?\s*\)?$")


def reads_canonical(h: Hit, canonical: set[str], s: Scan) -> bool:
    """Whether `h` is an `abbrev` whose body is a canonical declaration (keys in `canonical`), or a list, array or option
    of one, or a pair of an id and one."""
    m = h.keyword == "abbrev" and _CONTAINER.match(h.body)
    return bool(m) and resolve(m.group(1), h.key.partition("::")[0], h.scopes, h.opens, s, skip=h.key) in canonical


def slots(s: Scan) -> set[str]:
    """The keys of the hits that a `Core.Protocol` value names as its `Params`, `Commitment`, `Evidence` or `Public`."""
    return {k for path, _field, name, scopes, opens in s.fields if (k := resolve(name, path, scopes, opens, s))}


#: The most lines ``KNOWN`` may hold. Raising it, for a new line, needs lean's review; `--prune` lowers it to the count.
KNOWN_MAX = 387

#: Every hit that is not a canonical, a mirror, a reading of the canonical or a protocol's slot, with its reason and owner.
#: For new code, read the canonical or rename; `--prune` deletes the lines whose declarations went.
KNOWN: dict[str, str] = {
    # Security/Definitions/
    "Security/Definitions/Accounting/Bandwidth/Game.lean::Bandwidth.Device":
        "duplicate of the device model (`Accounting.Device`, not yet in Core): becomes it (deleted) (memory-accounting)",
    "Security/Definitions/Accounting/Bandwidth/Game.lean::Bandwidth.Store":
        "duplicate of `CircuitRequirement.Store`: becomes a reading of it (memory-accounting)",
    "Security/Definitions/Accounting/Bandwidth/Interactive.lean::Bandwidth.InteractiveDevice":
        "duplicate of the device model (`Accounting.Device`, not yet in Core): becomes its interactive case (deleted) (memory-accounting)",
    "Security/Definitions/Accounting/BoundReceipt.lean::Accounting.BoundReceipt.Coins":
        "duplicate of the live coins (`Verity.Coins`; `SampledProofs.Challenger.LiveCommitment` in code): becomes a reading of it (memory-accounting)",
    "Security/Definitions/Accounting/BoundReceipt.lean::Accounting.BoundReceipt.Node":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes a reading of it (memory-accounting)",
    "Security/Definitions/Accounting/CoinFlow.lean::Accounting.CoinFlow.Program":
        "homonym, a Program's dataflow over coins and values: rename (compute-accounting)",
    "Security/Definitions/Accounting/Compose.lean::Accounting.Compose.Certificate":
        "homonym, a credit certificate: rename (lean)",
    "Security/Definitions/Accounting/ComposePeriod.lean::Accounting.Compose.Period":
        "duplicate of the period (`Verity.Period`, not yet in Core): becomes a reading of it (lean)",
    "Security/Definitions/Accounting/Device.lean::Accounting.Device":
        "duplicate of the device model (`Accounting.Device`, not yet in Core): becomes `Verity.Device`, moved into Security/Definitions/Core/ (lean)",
    "Security/Definitions/Accounting/Device.lean::Accounting.Op":
        "homonym, the op classes a step charges: rename (`OpClass`) (lean)",
    "Security/Definitions/Accounting/Device.lean::Accounting.Window":
        "duplicate of the period (`Verity.Period`, not yet in Core): becomes `Verity.Period`, moved into Security/Definitions/Core/ (lean)",
    "Security/Definitions/Accounting/Gates/Codec.lean::Accounting.Gates.Draw":
        "homonym, a draw of a k-round key: rename (memory-accounting)",
    "Security/Definitions/Accounting/Gates/Model.lean::Accounting.Gates.Dag":
        "duplicate of `Verity.Circuit`: becomes a reading of it (memory-accounting)",
    "Security/Definitions/Accounting/Gates/Model.lean::Accounting.Gates.Gate":
        "duplicate of `Verity.Circuit`'s gate: becomes it, annotated (memory-accounting)",
    "Security/Definitions/Accounting/Gates/Model.lean::Accounting.Gates.Op":
        "duplicate of `Verity.Boolean.Op`: becomes it, over input slots (memory-accounting)",
    "Security/Definitions/Accounting/Gates/Model.lean::Accounting.Gates.Site":
        "homonym, a gate's place in its Definition (renamed Gates.Place by an open PR): rename (memory-accounting)",
    "Security/Definitions/Accounting/Physical.lean::Accounting.Hardware":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes a reading of it: the Site plus the device model (lean)",
    "Security/Definitions/Accounting/PouwWork.lean::Accounting.PouwWork.Verdict":
        "duplicate of `Core.Verdict`: becomes a spec verdict over it, once the spec has one (compute-accounting)",
    "Security/Definitions/Accounting/RateAudit.lean::Accounting.Rate.Coins":
        "duplicate of the live coins (`Verity.Coins`; `SampledProofs.Challenger.LiveCommitment` in code): becomes a reading of it (memory-accounting)",
    "Security/Definitions/Accounting/RateAudit.lean::Accounting.Rate.Device":
        "duplicate of the device model (`Accounting.Device`, not yet in Core): becomes it (deleted) (memory-accounting)",
    "Security/Definitions/Accounting/RateAudit.lean::Accounting.Rate.Store":
        "duplicate of `CircuitRequirement.Store`: becomes a reading of it (memory-accounting)",
    "Security/Definitions/Accounting/RateAuditHorizon.lean::Accounting.Rate.Session":
        "duplicate of the period (`Verity.Period`, not yet in Core): becomes a reading of it: a run over sampling periods (memory-accounting)",
    "Security/Definitions/Accounting/RateAuditInteractive.lean::Accounting.Rate.InteractiveDevice":
        "duplicate of the device model (`Accounting.Device`, not yet in Core): becomes its interactive case (deleted) (memory-accounting)",
    "Security/Definitions/Accounting/RateAuditSketch.lean::Accounting.RateSketch.Coins":
        "duplicate of the live coins (`Verity.Coins`; `SampledProofs.Challenger.LiveCommitment` in code): becomes a reading of it (memory-accounting)",
    "Security/Definitions/Accounting/RateAuditSketch.lean::Accounting.RateSketch.Device":
        "duplicate of the device model (`Accounting.Device`, not yet in Core): becomes it (deleted) (memory-accounting)",
    "Security/Definitions/Accounting/RateAuditSketchInteractive.lean::Accounting.RateSketch.InteractiveDevice":
        "duplicate of the device model (`Accounting.Device`, not yet in Core): becomes its interactive case (deleted) (memory-accounting)",
    "Security/Definitions/Accounting/Remote/Draw.lean::Accounting.Remote.Draw.Coins":
        "duplicate of the live coins (`Verity.Coins`; `SampledProofs.Challenger.LiveCommitment` in code): becomes a reading of it (memory-accounting)",
    "Security/Definitions/Accounting/Remote/Draw.lean::Accounting.Remote.Draw.Session":
        "duplicate of the period (`Verity.Period`, not yet in Core): becomes a reading of it (memory-accounting)",
    "Security/Definitions/Accounting/Remote/Node.lean::Accounting.Remote.Node":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes a reading of it: a node's projection (memory-accounting)",
    "Security/Definitions/Accounting/Remote/Sketch.lean::Accounting.Remote.Sketch.Coins":
        "duplicate of the live coins (`Verity.Coins`; `SampledProofs.Challenger.LiveCommitment` in code): becomes a reading of it (memory-accounting)",
    "Security/Definitions/Accounting/Remote/Sketch.lean::Accounting.Remote.Sketch.Outcome":
        "homonym, a session's random outcome (coins): rename (memory-accounting)",
    "Security/Definitions/Accounting/Remote/Sketch.lean::Accounting.Remote.Sketch.Session":
        "duplicate of the period (`Verity.Period`, not yet in Core): becomes a reading of it (memory-accounting)",
    "Security/Definitions/Accounting/Requirement.lean::Accounting.Requirement.Layout":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes a reading of it: values placed on the Site's memory leaves (memory-accounting)",
    "Security/Definitions/Accounting/Requirement.lean::Accounting.Requirement.Placement":
        "duplicate of `CircuitRequirement.Placement`, stated in Accounting: becomes Core's, moved into Security/Definitions/Core/ (memory-accounting)",
    "Security/Definitions/Accounting/Requirement.lean::Accounting.Requirement.Store":
        "duplicate of `CircuitRequirement.Store`, stated in Accounting: becomes Core's, moved into Security/Definitions/Core/ (memory-accounting)",
    "Security/Definitions/Accounting/Served.lean::Accounting.Served.Layout":
        "duplicate of the execution block (no canonical yet: LEAN-93): becomes a reading of it: an execution block's public shape (lean)",
    "Security/Definitions/Accounting/TS/J10R.lean::Accounting.TS.J10R.Layout":
        "homonym, a tile shape: rename (compute-accounting)",
    "Security/Definitions/Accounting/TS/Machine.lean::Accounting.TS.Machine":
        "duplicate of the device model (`Accounting.Device`, not yet in Core): becomes a reading of it (compute-accounting)",
    "Security/Definitions/Accounting/Tag/PolyHash.lean::Accounting.Tag.Block":
        "homonym, a chunk of the store: rename (lean)",
    "Security/Definitions/Claims.lean::Claims.Circuit":
        "duplicate of `Verity.Circuit`: becomes it (the claims layer's AND/XOR wire circuit) (lean)",
    "Security/Definitions/Claims.lean::Claims.Flock.Stmt": "duplicate of `Flock.Stmt`: becomes its spec, in Core (lean)",
    "Security/Definitions/Claims.lean::Claims.Gate":
        "duplicate of `Verity.Circuit`'s gate: becomes it (the AND/XOR gate) (lean)",
    "Security/Definitions/Claims/Recursion.lean::Claims.Recursion.Draw":
        "duplicate of `Verity.Draw`: becomes it, as a V* session's record holds it (lean)",
    "Security/Definitions/Claims/Recursion.lean::Claims.Recursion.Stmt":
        "homonym, the recursion's statement: rename (lean)",
    "Security/Definitions/Claims/Sampling.lean::Claims.Sampling.Commitment":
        "duplicate of `merkle.Commitment`, the claims layer's named primitive: becomes `Hash.RowLeaf`'s root (lean)",
    "Security/Definitions/Claims/Sampling.lean::Claims.Sampling.Netlist":
        "duplicate of `universal.Netlist`, stated in Claims: becomes Core's, moved into Security/Definitions/Core/ (lean)",
    "Security/Definitions/Claims/Sampling.lean::Claims.Sampling.Opening":
        "duplicate of `Hash.RowLeaf.Opening` (`merkle.Opening` in Python): becomes a reading of it (lean)",
    "Security/Definitions/Claims/Sampling.lean::Claims.Sampling.Partition":
        "duplicate of `Verity.Partition`: becomes it (classes, strata and units) (lean)",
    "Security/Definitions/Claims/Sampling.lean::Claims.Sampling.Prog":
        "duplicate of `Flock.Program.Prog`: becomes its spec, flattened to gates (lean)",
    "Security/Definitions/Claims/Sampling.lean::Claims.Sampling.Stmt": "homonym, the sampling statement: rename (lean)",
    "Security/Definitions/Core/Accel/Card.lean::Verity.Accel.Store": "homonym, a GPU's memory kinds: rename (proofs)",
    "Security/Definitions/Core/Accel/Isa.lean::Verity.Accel.Op": "homonym, an opcode over stores: rename (proofs)",
    "Security/Definitions/Core/Advice.lean::Verity.Advice.Registration":
        "homonym, a hiding commitment to a partition's rows: rename (proofs)",
    "Security/Definitions/Core/Boolean.lean::Verity.Boolean.Circuit":
        "duplicate of `Verity.Circuit`: becomes its Boolean case (proofs)",
    "Security/Definitions/Flock/Accounting/Bound.lean::FlockSoundness.Accounting.Stmt":
        "duplicate of `Flock.Stmt`: becomes a reading of its shape (lean)",
    "Security/Definitions/Flock/Circuit.lean::FlockSoundness.Audit.Circuit":
        "duplicate of `Verity.Circuit`: becomes it (a copy of Core/Boolean's) (lean)",
    "Security/Definitions/Flock/Circuit.lean::FlockSoundness.Audit.Op":
        "duplicate of `Verity.Boolean.Op`: becomes it (a copy of Core/Boolean's) (lean)",
    "Security/Definitions/Flock/Circuit.lean::FlockSoundness.Audit.Partition":
        "duplicate of `Verity.Partition`: becomes it (a copy of Core's) (lean)",
    "Security/Definitions/Flock/Lowering.lean::FlockSoundness.Placement":
        "homonym, a unit's rows in a statement: rename (lean)",
    "Security/Definitions/Flock/Lowering.lean::FlockSoundness.Prog":
        "duplicate of `Flock.Program.Prog`: becomes its spec, unit by unit (lean)",
    "Security/Definitions/Flock/Merkle.lean::FlockSoundness.Merkle.Collision":
        "duplicate of `Hash.Collision`: becomes it (`Hash.HasCollision`) (lean)",
    "Security/Definitions/Flock/Statement.lean::FlockSoundness.Model.Statement":
        "duplicate of `Flock.Stmt`: becomes its spec (one block R1CS statement) (lean)",
    "Security/Definitions/Flock/Statement.lean::FlockSoundness.Model.Witness":
        "duplicate of `Verity.TC.Relation.Witness`: becomes one Core witness (lean)",
    "Security/Definitions/Nci/Cover.lean::Nci.Cover.Circuit":
        "duplicate of `Verity.Circuit`: becomes it, with registered parameters (compute-accounting)",
    "Security/Definitions/Nci/Ldl.lean::Nci.Ldl.Registration":
        "duplicate of the registration log (`Verity.Registration`, not yet in Core: LEAN-93): becomes `Verity.Registration` (deleted) (compute-accounting)",
    "Security/Definitions/NetworkCertifier/Run.lean::NetTiming.Program":
        "homonym, a committed Program's egress prediction: rename (network-accounting)",
    "Security/Definitions/NetworkCertifier/SampledAudit.lean::NetTiming.Layout":
        "homonym, the boundary's gates in a program: rename (network-accounting)",
    "Security/Definitions/NetworkCertifier/WindowCharge.lean::NetTiming.WindowCharge.Window":
        "homonym, one window's timing advice: rename (network-accounting)",
    "Security/Definitions/Population/DrawPressure.lean::Population.Exfil.Draw":
        "duplicate of `Verity.Draw`: becomes it (k of n units, without replacement) (lean)",
    "Security/Definitions/Pous/Game/Oracle.lean::Pous.Prog":
        "homonym, PoUS's parallel oracle program: rename (memory-accounting)",
    "Security/Definitions/Pous/Model/Audit.lean::Pous.Outcome":
        "homonym, an audit's randomness: rename (memory-accounting)",
    "Security/Definitions/Pous/Model/StreamedWipeServer.lean::Pous.StreamedWipe.Verdict":
        "homonym, the wipe's acceptance rule over a prover's answers: rename (memory-accounting)",
    "Security/Definitions/Pouw/Assumptions/Dimension/Assumptions.lean::Pouw.Dimension.Assumptions.Machine":
        "homonym, a cost model of checked words: rename (compute-accounting)",
    "Security/Definitions/Pouw/Assumptions/Dimension/Embed.lean::Pouw.Dimension.Embed.Op":
        "homonym, an M4090 instruction: rename (compute-accounting)",
    "Security/Definitions/Pouw/Dither.lean::Pouw.Dither.Period":
        "duplicate of the period (`Verity.Period`, not yet in Core): becomes a reading of it: a monitoring period (compute-accounting)",
    "Security/Definitions/Pouw/Dither.lean::Pouw.Dither.Verdict":
        "duplicate of `Core.Verdict`: becomes a spec verdict over it, once the spec has one (compute-accounting)",
    "Security/Definitions/Pouw/DitherInstance.lean::Pouw.Dither.Layout":
        "homonym, where a round's words sit in memory: rename (compute-accounting)",
    "Security/Definitions/Pouw/Eps1/M3.lean::Pouw.Eps1.Prog":
        "homonym, an M3 program for Dither's words: rename (compute-accounting)",
    "Security/Definitions/Pouw/Game/Defs.lean::Pouw.Layout":
        "homonym, a declared tensor layout: rename (compute-accounting)",
    "Security/Definitions/Pouw/Game/Defs.lean::Pouw.Transcript":
        "homonym, PoUW's committed transcript of unit commitments: rename (compute-accounting)",
    "Security/Definitions/Pouw/PearlC/Game.lean::Pouw.PearlC.Layout":
        "duplicate of `Pouw.Layout`: becomes a reading of it (compute-accounting)",
    "Security/Definitions/Pouw/PearlC/Game.lean::Pouw.PearlC.Transcript":
        "homonym, Pearl-C's committed transcript, `Pouw.Transcript` again: rename (compute-accounting)",
    "Security/Definitions/Pouw/PearlC/TileCircuit.lean::Pouw.PearlC.TileCircuit.Gate":
        "homonym, one gate of Pearl-C's word circuit: rename (compute-accounting)",
    "Security/Definitions/Pouw/PearlC/W1.lean::Pouw.PearlC.W1.Prog":
        "homonym, Pearl-C's W1 constant bank and code tree: rename (compute-accounting)",
    "Security/Definitions/Pouw/PearlC/WholeCircuit.lean::Pouw.PearlC.WholeCircuit.Block":
        "homonym, a proof block kind of a table: rename (compute-accounting)",
    "Security/Definitions/Pouw/RelativeSecurity.lean::Pouw.RelativeSecurity.Machine":
        "homonym, a model of computation: rename (compute-accounting)",
    "Security/Definitions/Pouw/TileBound/Defs.lean::Pouw.TileBound.Gate":
        "homonym, one firing of the m16n8k64 tile: rename (compute-accounting)",
    # Security/Proofs/
    "Security/Proofs/Flock/Claims/Bridge/Committed.lean::Claims.Bridge.CoinTree":
        "duplicate of the live coins (`Verity.Coins`; `SampledProofs.Challenger.LiveCommitment` in code): becomes it (a coin tree over k rounds) (proofs)",
    "Security/Proofs/Flock/Claims/Complete/Replay/Lincheck.lean::FlockComplete.Replay.Lc.Coins":
        "homonym, the coins C-Flock's lincheck reads in one proof: rename (proofs)",
    "Security/Proofs/Flock/Claims/Complete/Replay/RingSwitch.lean::FlockComplete.Replay.Rs.Coins":
        "homonym, the coins C-Flock's ring switch reads in one proof: rename (proofs)",
    "Security/Proofs/Flock/Claims/Complete/Replay/Zerocheck.lean::FlockComplete.Replay.Zc.Coins":
        "homonym, the coins C-Flock's zerocheck reads in one proof: rename (proofs)",
    "Security/Proofs/Flock/Claims/Complete/Zk/InnerGo.lean::FlockComplete.Replay.InZ.Coins":
        "homonym, the ZK inner proof's coins, each a reply to its send: rename (proofs)",
    "Security/Proofs/Flock/Claims/GS/RR.lean::Claims.GS.Node": "homonym, a search node: rename (proofs)",
    "Security/Proofs/Flock/Claims/Recursion/Inner.lean::Claims.Recursion.Opening":
        "duplicate of `Hash.RowLeaf.Opening` (`merkle.Opening` in Python): becomes a reading of it (proofs)",
    "Security/Proofs/Flock/Claims/Recursion/VStar.lean::Claims.Recursion.VStar.Layout":
        "homonym, proof-internal layout: rename (proofs)",
    "Security/Proofs/Flock/Claims/ZK/Layout.lean::Claims.ZKMal.Layout":
        "homonym, proof-internal layout: rename (proofs)",
    "Security/Proofs/Flock/Soundness/Binding/Layout.lean::FlockSoundness.Binding.Layout":
        "homonym, proof-internal layout: rename (proofs)",
    "Security/Proofs/Flock/Soundness/Binding/Layout.lean::FlockSoundness.Binding.Layout.Opening":
        "duplicate of `Hash.RowLeaf.Opening` (`merkle.Opening` in Python): becomes a reading of it (proofs)",
    "Security/Proofs/Flock/Soundness/Discharge/Exec/Event.lean::FlockSoundness.Discharge.Exec.Transcript":
        "homonym, a session directory's files: rename (proofs)",
    "Security/Proofs/Flock/Soundness/ExecPlacement.lean::FlockSoundness.Layout":
        "homonym, proof-internal layout: rename (proofs)",
    "Security/Proofs/Flock/Soundness/GateRows.lean::FlockSoundness.GateRows.Gate":
        "duplicate of `Verity.Circuit`'s gate: becomes it (proofs)",
    "Security/Proofs/Flock/Soundness/Types/Flatten.lean::FlockSoundness.Types.Flatten.Gate":
        "duplicate of `Verity.Circuit`'s gate: becomes it (proofs)",
    "Security/Proofs/Flock/Soundness/ZK/GK/Extract.lean::FlockSoundness.ZK.GK.CoinTree":
        "duplicate of the live coins (`Verity.Coins`; `SampledProofs.Challenger.LiveCommitment` in code): becomes it (the extractor's tree) (proofs)",
    "Security/Proofs/Flock/VBridge/Family/AccRows.lean::FlockVBridge.Family.Site":
        "homonym, the registration a registered port reads: rename (proofs)",
    "Security/Proofs/Flock/Verifier/Hash.lean::Flock.Sha256.Collision":
        "duplicate of `Hash.Collision`: becomes it at SHA-256 (proofs)",
    "Security/Proofs/Flock/Verifier/Hash.lean::Flock.Sha512.Collision":
        "duplicate of `Hash.Collision`: becomes it at SHA-512 (proofs)",
    "Security/Proofs/Flock/Verifier/Merkle.lean::Flock.MerkleScheme.Collision":
        "duplicate of `Hash.Collision`: becomes it at the scheme's hash (proofs)",
    "Security/Proofs/Pouw/RelativeSecurity.lean::Pouw.RelativeSecurity.ModelB.Gate":
        "homonym, relative security's proof-local gate: rename (compute-accounting)",
    "Security/Proofs/Pouw/RelativeSecurity.lean::Pouw.RelativeSecurity.ModelB.Prog":
        "homonym, relative security's proof-local program: rename (compute-accounting)",
    # Security/Properties/
    "Security/Properties/Pous/MemoryChallenger.lean::MemoryChallenger.Properties.Device":
        "homonym, the memory challengers a property ranges over: rename (lean)",
    "Security/Properties/Pous/WipeSigned.lean::MemoryChallenger.Properties.Challenger":
        "homonym, the keys' signing sessions in the wipe-signing property: rename (lean)",
    "Security/Properties/RemoteAudit/Line/Record.lean::Properties.RemoteAudit.Line.Node":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes what `Accounting.Remote.Node` becomes (lean)",
    "Security/Properties/RemoteAudit/Line/Record.lean::Properties.RemoteAudit.Line.Record":
        "duplicate of the auditor's record chain (`SampledProofs.Challenger.LedgerEntry`): becomes what RemoteAudit becomes (lean)",
    "Security/Properties/RemoteAudit/Live.lean::Properties.RemoteAudit.Live.Answer":
        "homonym, a live answer: rename (lean)",
    # benchmarks/
    "benchmarks/bandwidth/gpu.py::Commitment":
        "duplicate of `merkle.Commitment`: becomes a client of it (one pass's root) (memory-accounting)",
    "benchmarks/circuit_properties/circuit_properties/ops.py::Site":
        "homonym, an op's site in a Definition: rename (training)",
    "benchmarks/datacenter/datacenter/account.py::Ledger":
        "duplicate of the auditor's record chain (`SampledProofs.Challenger.LedgerEntry`): becomes a reading of it (network-accounting)",
    "benchmarks/datacenter/datacenter/binding.py::Placement":
        "duplicate of `CircuitRequirement.Placement`: becomes a client of it (network-accounting)",
    "benchmarks/datacenter/datacenter/logical.py::Registration":
        "duplicate of the registration log (`Verity.Registration`, not yet in Core: LEAN-93): becomes a reading of it (network-accounting)",
    "benchmarks/datacenter/datacenter/memory.py::Store":
        "duplicate of `CircuitRequirement.Store`: becomes a client of it (network-accounting)",
    "benchmarks/datacenter/datacenter/observe.py::Answer":
        "homonym, the challenger's record of one request: rename (network-accounting)",
    "benchmarks/datacenter/datacenter/observe.py::Record":
        "duplicate of `NetworkCheck.Check.Certificate`: becomes its certified record (network-accounting)",
    "benchmarks/datacenter/datacenter/physical.py::Block":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes a zone of it (the old word for one) (network-accounting)",
    "benchmarks/datacenter/datacenter/physical.py::Device":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes its device entry (network-accounting)",
    "benchmarks/datacenter/datacenter/physical.py::Link":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes its link (network-accounting)",
    "benchmarks/harness/verity_bench/settings.py::Registry": "homonym, one distribution's settings: rename (benchmarks)",
    "benchmarks/hash/dither/dither_fused.py::Window": "homonym, a Dither kernel's device state: rename (kernels)",
    "benchmarks/hash/dither/dither_one.py::Window": "homonym, a Dither kernel's device state: rename (kernels)",
    "benchmarks/hash/lib/vclib.py::Opening": "homonym, one vc_open_launch record: rename (kernels)",
    "benchmarks/network_e2e/encodings/code_v1.py::Window":
        "homonym, one certifier window's charge: rename (network-accounting)",
    "benchmarks/network_e2e/encodings/v1.py::Record":
        "duplicate of `NetworkCheck.Check.Certificate`: becomes its certified record (network-accounting)",
    "benchmarks/network_sim/workload/block.py::Block":
        "homonym, the simulator's model of one served block: rename (network-accounting)",
    "benchmarks/network_sim/workload/hardware.py::Hardware":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes a reading of it (network-accounting)",
    "benchmarks/network_sim/workload/hardware.py::Link":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes its link (network-accounting)",
    "benchmarks/network_sim/workload/lab.py::Session":
        "homonym, the old word for a run, here a simulated read: rename (network-accounting)",
    "benchmarks/network_sim/workload/placement.py::Placement":
        "duplicate of `CircuitRequirement.Placement`: becomes a client of it (network-accounting)",
    "benchmarks/network_traces/n2_advised_zk.py::Challenger":
        "duplicate of `parties.Challenger`: becomes a client of it (network-accounting)",
    "benchmarks/network_traces/transfer_zk.py::Block":
        "homonym, a server block's one template: rename (network-accounting)",
    "benchmarks/numerical/python/verity_numerical/bench/lowerings.py::Registry":
        "homonym, a table of benchmark lowerings: rename (compute-accounting)",
    "benchmarks/numerical/python/verity_numerical/bench/templates.py::Draw":
        "homonym, one synthetic benchmark draw: rename (compute-accounting)",
    "benchmarks/numerical/python/verity_numerical/bench/views.py::Hash":
        "homonym, one hash a table row relies on: rename (compute-accounting)",
    "benchmarks/numerical/python/verity_numerical/reference/leaf_v2.py::Session":
        "homonym, the old word for a run: rename (compute-accounting)",
    "benchmarks/numerical/python/verity_numerical/reference/leaf_v2.py::Verdict":
        "duplicate of `Core.Verdict`: becomes a client of it (compute-accounting)",
    "benchmarks/one_stage/a5_coins.py::Gate": "homonym, infra's challenger gate (an SSH host): rename (proofs)",
    "benchmarks/pous/p2_v1/store.py::Store":
        "duplicate of `CircuitRequirement.Store`: becomes a client of it (memory-accounting)",
    "benchmarks/pouw/accel_v31/v31.py::Coins":
        "duplicate of the live coins (`Verity.Coins`; `SampledProofs.Challenger.LiveCommitment` in code): becomes a client of it (compute-accounting)",
    "benchmarks/pouw/clear/clear_audit.py::Transcript":
        "homonym, what a clear audit's prover publishes: rename (compute-accounting)",
    "benchmarks/pouw/clear/clear_audit.py::Verdict":
        "duplicate of `Core.Verdict`: becomes a client of it (compute-accounting)",
    "benchmarks/pouw/dc/census.py::Device": "homonym, a census family's merged runs: rename (compute-accounting)",
    "benchmarks/pouw/e2e_120/window.py::Coins":
        "duplicate of the live coins (`Verity.Coins`; `SampledProofs.Challenger.LiveCommitment` in code): becomes a client of it (compute-accounting)",
    "benchmarks/pouw/e2e_120/window.py::Draw":
        "duplicate of `service.Draw`: becomes a client of it (compute-accounting)",
    "benchmarks/pouw/e2e_120/window.py::Verdict":
        "duplicate of `Core.Verdict`: becomes a client of it (compute-accounting)",
    "benchmarks/pouw/e2e_120/window.py::Window":
        "duplicate of the execution block (no canonical yet: LEAN-93): becomes a client of it (compute-accounting)",
    "benchmarks/pouw/filler/arms.py::Placement": "homonym, a filler arm's placement: rename (compute-accounting)",
    "benchmarks/pouw/grind_price.py::Site": "homonym, one seed group: rename (compute-accounting)",
    "benchmarks/pouw/harness/arm.py::Gate": "homonym, a kernel gate's verdict at one shape: rename (compute-accounting)",
    "benchmarks/pouw/harness/chain.py::Link":
        "homonym, one operand set's call in a kernel chain: rename (compute-accounting)",
    "benchmarks/pouw/harness/native.py::Device": "homonym, one CUDA device handle: rename (compute-accounting)",
    "benchmarks/pouw/harness/sass_gate.py::Gate": "homonym, a run's SASS gate: rename (compute-accounting)",
    "benchmarks/pouw/highutil/period.py::Coins":
        "duplicate of the live coins (`Verity.Coins`; `SampledProofs.Challenger.LiveCommitment` in code): becomes a client of it (compute-accounting)",
    "benchmarks/pouw/highutil/period.py::Period":
        "duplicate of the period (`Verity.Period`, not yet in Core): becomes a client of it (compute-accounting)",
    "benchmarks/pouw/machine_model/verity_machine_model/capacity.py::Window":
        "homonym, a capacity model's window and node cell: rename (compute-accounting)",
    "benchmarks/pouw/machine_model/verity_machine_model/eps.py::Machine":
        "homonym, a machine model's Π: rename (compute-accounting)",
    "benchmarks/pouw/pearl_c_vllm/e2e.py::Coins":
        "duplicate of the live coins (`Verity.Coins`; `SampledProofs.Challenger.LiveCommitment` in code): becomes a client of it (compute-accounting)",
    "benchmarks/pouw/rt2_hopper/li_window_sat.py::Program":
        "homonym, a SAT model's instruction list: rename (compute-accounting)",
    "benchmarks/pouw/served_h1/coins.py::Coins":
        "duplicate of the live coins (`Verity.Coins`; `SampledProofs.Challenger.LiveCommitment` in code): becomes a client of it (compute-accounting)",
    "benchmarks/pouw/served_h1/coins.py::Session":
        "homonym, the old word for a run, here a gate session: rename (compute-accounting)",
    "benchmarks/pouw/served_h1/partials.py::Window":
        "duplicate of the execution block (no canonical yet: LEAN-93): becomes a client of it (compute-accounting)",
    "benchmarks/pouw/served_h1/serve.py::Record":
        "homonym, a served request's benchmark row: rename (compute-accounting)",
    "benchmarks/pouw/served_zk/of_record.py::Record":
        "homonym, a check's two verdicts side by side: rename (compute-accounting)",
    "benchmarks/pouw/served_zk/replay_window.py::Window":
        "homonym, one replayed window's checks: rename (compute-accounting)",
    "benchmarks/pouw/served_zk/served_commit.py::Challenger":
        "duplicate of `parties.Challenger`: becomes a client of it (compute-accounting)",
    "benchmarks/pouw/served_zk/statement.py::Statement":
        "homonym, a benchmark's statement bench: rename (compute-accounting)",
    "benchmarks/pouw/theory/checked_strip.py::Block":
        "homonym, a block of rows and columns of one design: rename (compute-accounting)",
    "benchmarks/pouw/theory/checked_strip.py::Ledger":
        "homonym, a strategy's priced products: rename (compute-accounting)",
    "benchmarks/pouw/theory/ksw_k.py::Layout":
        "homonym, the bits of a theory model's coin groups: rename (compute-accounting)",
    "benchmarks/property_redteam/demo.py::Circuit":
        "homonym, a red-team demo's multi-precision arithmetic builder: rename (memory-accounting)",
    "benchmarks/property_redteam/renorm.py::Gate":
        "homonym, a red-team model's gate over its own ops: rename (memory-accounting)",
    "benchmarks/property_redteam/renorm.py::Op": "homonym, a red-team model's op: rename (memory-accounting)",
    "benchmarks/property_redteam/squaring.py::Gate":
        "homonym, a red-team model's gate over its own ops: rename (memory-accounting)",
    "benchmarks/property_redteam/squaring.py::Op": "homonym, a red-team model's op: rename (memory-accounting)",
    "benchmarks/prove_path/window_cost.py::Registration":
        "homonym, a size model of one window's registration: rename (network-accounting)",
    "benchmarks/remote_session/challengers.py::Gate":
        "homonym, infra's challenger gate (an SSH host): rename (memory-accounting)",
    "benchmarks/remote_session/device.py::Device":
        "homonym, the prover's backend in a remote session: rename (memory-accounting)",
    "benchmarks/remote_session/gate_whatif.py::Record":
        "homonym, a session's replayed times: rename (memory-accounting)",
    "benchmarks/remote_session/lean_verifier.py::Transcript":
        "homonym, a remote session's difftest case: rename (memory-accounting)",
    "benchmarks/resource_accounting/joint.py::Challenger":
        "duplicate of `parties.Challenger`: becomes a client of the memory challenger's (memory-accounting)",
    "benchmarks/resource_accounting/joint.py::Device":
        "homonym, the store's bandwidth prover: rename (memory-accounting)",
    "benchmarks/rho/rho/circuits.py::Circuit":
        "homonym, a benchmark row naming a Definition at its size: rename (training)",
    "benchmarks/rho/rho/params.py::Registry": "homonym, a table of benchmark parameters: rename (training)",
    "benchmarks/rho/rho/theorem.py::Partition":
        "homonym, a benchmark's serving blocks and dark leaves: rename (training)",
    "benchmarks/sampling_plan/h200.py::Window": "homonym, one subcircuit's sampling block: rename (proofs)",
    "benchmarks/sampling_plan/transformer.py::Dag": "homonym, a block's Definition DAG for the plan: rename (proofs)",
    "benchmarks/stacked_share/units.py::Window": "homonym, an attention window of S keys: rename (memory-accounting)",
    "benchmarks/tc_probe_fp4/f1_rates.py::Op": "homonym, one CUDA-core multiplier a probe times: rename (circuits)",
    "benchmarks/tc_probe_fp4/sp_e4m3.py::Device": "homonym, a probe's loaded CUDA module: rename (circuits)",
    "benchmarks/torch_gpu/commit_slots/slots.py::Layout":
        "homonym, a step's public commitment structure: rename (kernels)",
    "benchmarks/torch_gpu/p2_stage/stage.py::Store":
        "duplicate of `CircuitRequirement.Store`: becomes a client of it (kernels)",
    "benchmarks/torch_gpu/s8_dec_linear/run.py::Device": "homonym, slices of a device byte buffer: rename (kernels)",
    "benchmarks/vllm_encoded/paged.py::Layout": "homonym, a layer cache's shape in words: rename (memory-accounting)",
    "benchmarks/vllm_encoded/serve.py::Window": "homonym, a profiler window: rename (memory-accounting)",
    # examples/
    "examples/h200/e2e_h200/store.py::Store": "homonym, where a demo's weights live: rename (compute-accounting)",
    "examples/h200/e2e_h200/window.py::Window":
        "duplicate of the execution block (no canonical yet: LEAN-93): becomes a client of it (compute-accounting)",
    "examples/h200/interfaces/verity_e2e_demo/replay_bundle.py::Commitment":
        "duplicate of `merkle.Commitment`: becomes a client of it (the run's step commitments) (compute-accounting)",
    "examples/lifecycle/lifecycle.py::Block":
        "duplicate of the execution block (no canonical yet: LEAN-93): becomes a client of it (top)",
    "examples/lifecycle/lifecycle.py::Card": "duplicate of `Verity.Accel.Card`: becomes a client of it (top)",
    "examples/lifecycle/lifecycle.py::Layout":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes a client of it (top)",
    "examples/lifecycle/lifecycle.py::Link":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes a client of its link (top)",
    "examples/lifecycle/lifecycle.py::Placement":
        "duplicate of `CircuitRequirement.Placement`: becomes a client of it (top)",
    "examples/lifecycle/lifecycle.py::Registration":
        "duplicate of the registration log (`Verity.Registration`, not yet in Core: LEAN-93): becomes a client of it (top)",
    "examples/lifecycle/lifecycle.py::Statement": "homonym, a drawn unit's public statement in the demo: rename (top)",
    "examples/lifecycle/onsite.py::Challenger": "duplicate of `parties.Challenger`: becomes a client of the memory challenger's (top)",
    # infra/
    "infra/cluster/src/cluster/ledger.py::Ledger": "homonym, the shared machines' spend ledger: rename (infra)",
    "infra/cluster/src/cluster/model.py::Node": "homonym, a shared machine in the cluster's model: rename (infra)",
    "infra/cluster/src/cluster/route.py::Placement": "homonym, a routed job's node: rename (infra)",
    "infra/ops/verity_ops/deploy.py::Node": "homonym, a deploy target: rename (infra)",
    # integrations/
    "integrations/vllm/tools/migrate/superseded_quarantine.py::Site":
        "homonym, a migration step's source site: rename (vllm)",
    "integrations/vllm/verity_vllm/acquire/committer_api.py::Opening":
        "duplicate of `Hash.RowLeaf.Opening` (`merkle.Opening` in Python): becomes a client of it (circuits)",
    "integrations/vllm/verity_vllm/acquire/plan.py::Site": "homonym, a quantisation site: rename (circuits)",
    "integrations/vllm/verity_vllm/acquire/served_device.py::Device":
        "homonym, a served build's device: rename (circuits)",
    "integrations/vllm/verity_vllm/check/batch.py::Verdict": "homonym, a template's check results: rename (vllm)",
    "integrations/vllm/verity_vllm/check/match/instance_sequence.py::Prog":
        "homonym, a decoded root body as an instance sequence: rename (vllm)",
    "integrations/vllm/verity_vllm/check/match/resolve.py::Program":
        "homonym, the Program side of a match's resolution: rename (vllm)",
    "integrations/vllm/verity_vllm/check/verdict.py::Verdict":
        "homonym, a check run's results and outcome: rename (vllm)",
    "integrations/vllm/verity_vllm/commit/memory_map.py::Layout": "homonym, the engine's KV pool map: rename (circuits)",
    "integrations/vllm/verity_vllm/commit/served_record.py::Record":
        "duplicate of the execution block (no canonical yet: LEAN-93): becomes an execution block's committed values (circuits)",
    "integrations/vllm/verity_vllm/observe/fold/resolver.py::Witness":
        "homonym, a request's witness leaves a row consumes: rename (vllm)",
    "integrations/vllm/verity_vllm/pipeline/c2_replay.py::Site": "homonym, what one C2 replay reads: rename (vllm)",
    "integrations/vllm/verity_vllm/pipeline/commit_coins.py::Gate":
        "homonym, infra's challenger gate (an SSH host): rename (vllm)",
    "integrations/vllm/verity_vllm/pipeline/commit_coins.py::Session":
        "homonym, the old word for a run, here the commit's challenge phase: rename (vllm)",
    "integrations/vllm/verity_vllm/pipeline/report.py::Card": "homonym, a report's summary card: rename (vllm)",
    "integrations/vllm/verity_vllm/pipeline/serve_commit.py::Ledger": "homonym, one request as served: rename (vllm)",
    "integrations/vllm/verity_vllm/program/frontend/registration.py::Registration":
        "homonym, the builder's allocation of registered inputs: rename (circuits)",
    "integrations/vllm/verity_vllm/protocols/pouw/h1/challenger.py::Ledger":
        "duplicate of the auditor's record chain (`SampledProofs.Challenger.LedgerEntry`): becomes it (deleted) (vllm)",
    "integrations/vllm/verity_vllm/protocols/pouw/h1/served_files.py::Registration":
        "duplicate of the registration log (`Verity.Registration`, not yet in Core: LEAN-93): becomes its builder entry (vllm)",
    "integrations/vllm/verity_vllm/protocols/pouw/h1/window.py::Window":
        "duplicate of the execution block (no canonical yet: LEAN-93): becomes a client of it (vllm)",
    "integrations/vllm/verity_vllm/protocols/remote_bandwidth/interface.py::Layout":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes its memory leaves by tier (vllm)",
    "integrations/vllm/verity_vllm/protocols/remote_bandwidth/interface.py::Opening":
        "duplicate of `Hash.RowLeaf.Opening` (`merkle.Opening` in Python): becomes a client of it (vllm)",
    "integrations/vllm/verity_vllm/protocols/remote_bandwidth/interface.py::Registration":
        "duplicate of the registration log (`Verity.Registration`, not yet in Core: LEAN-93): becomes a view of its head (vllm)",
    "integrations/vllm/verity_vllm/protocols/remote_bandwidth/registry.py::Store":
        "duplicate of `CircuitRequirement.Store`: becomes a reading of it (vllm)",
    "integrations/vllm/verity_vllm/query/units.py::Program": "homonym, what a query reads: rename (circuits)",
    # tools/
    "tools/config/verity_config/registry.py::Machine": "homonym, a pod's machine type: rename (ci)",
    "tools/config/verity_config/settings.py::Registry":
        "homonym, one distribution's settings from the environment: rename (ci)",
    "tools/move/layout.py::Layout": "homonym, a layout move's directory map: rename (ci)",
    "tools/research/src/research/compare.py::Verdict": "homonym, a run comparison's result: rename (infra)",
    "tools/research/src/research/drive.py::Verdict": "homonym, a queue train's land verdict: rename (ci)",
    "tools/verity/cli/verity_cli/run.py::Registry": "homonym, what `verity run` reads off step 0's table: rename (ci)",
    "tools/verity/lean/run_daily.py::Node": "homonym, node 1's paths for the daily Lean run: rename (lean)",
    # verity/core/
    "verity/core/deploy/manifest.py::Link":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes a reading of it (proofs)",
    "verity/core/deploy/manifest.py::Machine":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes a reading of it (proofs)",
    "verity/core/deploy/manifest.py::Placement": "homonym, the manifest check's result: rename (proofs)",
    "verity/core/deploy/manifest.py::Site":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes a reading of it (proofs)",
    "verity/core/deploy/party.py::Record": "homonym, a party's log entry: rename (proofs)",
    "verity/core/deploy/transcript.py::Transcript": "homonym, a party's message log: rename (proofs)",
    "verity/core/devices/certifier/lean/CertifierDevice/Certifier.lean::CertifierDevice.Record":
        "duplicate of `NetworkCheck.Check.Certificate`: becomes `grid.Record`'s port beside `RecordHead` (moved) (network-accounting)",
    "verity/core/devices/certifier/lean/CertifierDevice/Issuer.lean::CertifierDevice.Certificate":
        "duplicate of `NetworkCheck.Check.Certificate`: becomes it (deleted; it already requires `network_check`) (network-accounting)",
    "verity/core/devices/certifier/lean/CertifierDevice/Online.lean::CertifierDevice.Link":
        "homonym, a link's online state: rename (network-accounting)",
    "verity/core/devices/memory_challenger/lean/MemoryChallenger/Judge.lean::MemoryChallenger.Judge.Record":
        "homonym, a judge's decoded inputs: rename (memory-accounting)",
    "verity/core/devices/memory_challenger/lean/MemoryChallenger/Records.lean::MemoryChallenger.CoinTree":
        "duplicate of the live coins (`Verity.Coins`; `SampledProofs.Challenger.LiveCommitment` in code): becomes a reading of it (memory-accounting)",
    "verity/core/devices/memory_challenger/lean/MemoryChallenger/StreamedWipe.lean::MemoryChallenger.StreamedWipe.Verdict":
        "duplicate of `Core.Verdict`: becomes a reading of it (memory-accounting)",
    "verity/core/devices/memory_challenger/lean/MemoryChallenger/Verdict.lean::MemoryChallenger.Verdict":
        "duplicate of `Core.Verdict`: becomes a reading of it (memory-accounting)",
    "verity/core/devices/memory_challenger/lean/MemoryChallenger/WipeJudge.lean::MemoryChallenger.StreamedWipe.Record":
        "homonym, a judge's decoded inputs: rename (memory-accounting)",
    "verity/core/protocols/bandwidth/audit.py::Opening":
        "duplicate of `remote_bandwidth`'s own type (MEM-138): becomes it (the bandwidth package is deleted) (memory-accounting)",
    "verity/core/protocols/bandwidth/audit.py::Store":
        "duplicate of `remote_bandwidth`'s own type (MEM-138): becomes it (the bandwidth package is deleted) (memory-accounting)",
    "verity/core/protocols/bandwidth/audit.py::Verdict":
        "duplicate of `remote_bandwidth`'s own type (MEM-138): becomes it (the bandwidth package is deleted) (memory-accounting)",
    "verity/core/protocols/bandwidth/audit.py::Window":
        "duplicate of `remote_bandwidth`'s own type (MEM-138): becomes it (the bandwidth package is deleted) (memory-accounting)",
    "verity/core/protocols/certifier/audit.py::Verdict":
        "duplicate of `Core.Verdict`: becomes a reading of it (network-accounting)",
    "verity/core/protocols/certifier/cache_step_v2.py::Op":
        "homonym, a cache call's inputs: rename (network-accounting)",
    "verity/core/protocols/certifier/egress.py::Record": "homonym, a slot's record: rename (network-accounting)",
    "verity/core/protocols/certifier/io_request.py::Layout":
        "homonym, a request's cell layout: rename (network-accounting)",
    "verity/core/protocols/certifier/io_request.py::Link":
        "homonym, a link's served ingress: rename (network-accounting)",
    "verity/core/protocols/certifier/lean/NetworkCertifier/Run.lean::NetworkCertifier.Program":
        "homonym, a Program's egress prediction: rename (network-accounting)",
    "verity/core/protocols/certifier/lean/NetworkCertifier/WindowCharge.lean::NetworkCertifier.WindowCharge.Window":
        "homonym, one window's timing advice: rename (network-accounting)",
    "verity/core/protocols/certifier/replay_chain.py::Window": "homonym, a replay row: rename (network-accounting)",
    "verity/core/protocols/certifier/service.py::Opening":
        "homonym, a link-window's witness: rename (network-accounting)",
    "verity/core/protocols/interface/CoreProtocol/Profile.lean::Core.Draw":
        "homonym, how a profile level is drawn: rename (proofs)",
    "verity/core/protocols/network/lean/NetworkCheck/Check.lean::NetworkCheck.Check.Link":
        "homonym, one egress link's opening: rename (proofs)",
    "verity/core/protocols/network/lean/NetworkCheck/LinkWindow.lean::NetworkCheck.LinkWindow.Verdict":
        "duplicate of `Core.Verdict`: becomes a reading of it (proofs)",
    "verity/core/protocols/network/lean/NetworkCheck/Period.lean::NetworkCheck.Period.Layout":
        "homonym, the certifier's bucket and slot schedule: rename (proofs)",
    "verity/core/protocols/network/lean/NetworkCheck/Period.lean::NetworkCheck.Period.Registration":
        "duplicate of the registration log (`Verity.Registration`, not yet in Core: LEAN-93): becomes the log's projection (deleted) (proofs)",
    "verity/core/protocols/network/lean/NetworkCheck/Period.lean::NetworkCheck.Period.Window":
        "duplicate of the period (`Verity.Period`, not yet in Core): becomes a reading of it: a sampling period (proofs)",
    "verity/core/protocols/network/nci/lean/NciCheck/Check.lean::NciCheck.Circuit":
        "duplicate of `Flock.Program`: becomes a reading of it: one gate-level Boolean type beside `Flock.Program.Prog`, which both read (proofs)",
    "verity/core/protocols/pous/audit.py::Answer": "homonym, a challenge's answer: rename (memory-accounting)",
    "verity/core/protocols/pous/check/PousCheck/Certificate.lean::PousCheck.Certificate.Certificate":
        "homonym, what a PoUS audit establishes: rename (memory-accounting)",
    "verity/core/protocols/pous/check/PousCheck/Certificate.lean::PousCheck.Certificate.Store":
        "duplicate of `CircuitRequirement.Store`: becomes `PousCheck.P2.Params` (deleted) (memory-accounting)",
    "verity/core/protocols/pous/check/PousCheck/Run.lean::PousCheck.Run.Session":
        "homonym, the old word for a run, kept as one protocol's run type: rename (memory-accounting)",
    "verity/core/protocols/pous/check/PousCheck/Verdict.lean::PousCheck.Verdict.Gate":
        "duplicate of `Flock.Program`: becomes `Flock.Program` (deleted) (memory-accounting)",
    "verity/core/protocols/pous/check/PousCheck/Verdict.lean::PousCheck.Verdict.Opening":
        "duplicate of `Hash.RowLeaf.Opening` (`merkle.Opening` in Python): becomes a reading of it (memory-accounting)",
    "verity/core/protocols/pous/check/PousCheck/Verdict.lean::PousCheck.Verdict.Prog":
        "duplicate of `Flock.Program`: becomes `Flock.Program` (deleted) (memory-accounting)",
    "verity/core/protocols/pous/check/timed_audit.py::Store":
        "duplicate of `CircuitRequirement.Store`: becomes a reading of it (memory-accounting)",
    "verity/core/protocols/pous/layout.py::Layout":
        "duplicate of `CircuitRequirement.Store`: becomes a reading of it (memory-accounting)",
    "verity/core/protocols/pous/layout.py::Placement":
        "duplicate of `CircuitRequirement.Store`: becomes a reading of it (memory-accounting)",
    "verity/core/protocols/pous/period.py::Period":
        "duplicate of the period (`Verity.Period`, not yet in Core): becomes one `Core` period case, not one per protocol (memory-accounting)",
    "verity/core/protocols/pous/period.py::Store":
        "duplicate of `CircuitRequirement.Store`: becomes a reading of it: PoUS's view of the store entry (memory-accounting)",
    "verity/core/protocols/pous/protocol.py::Certificate":
        "homonym, what an accepted PoUS audit establishes: rename (memory-accounting)",
    "verity/core/protocols/pous/streamed_wipe.py::Challenger":
        "duplicate of `parties.Challenger`: becomes a reading of it (memory-accounting)",
    "verity/core/protocols/pous/wipe_parties.py::Challenger":
        "duplicate of `parties.Challenger`: becomes a reading of it (memory-accounting)",
    "verity/core/protocols/pouw/circuit/anchors.py::Ledger":
        "homonym, a window's admitted calls: rename (compute-accounting)",
    "verity/core/protocols/pouw/circuit/compile.py::Site":
        "homonym, a node of the served function and a kind of dot: rename (compute-accounting)",
    "verity/core/protocols/pouw/circuit/h1device.py::Device":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes a reading of it: the card by digest (compute-accounting)",
    "verity/core/protocols/pouw/circuit/h1tile.py::Layout": "homonym, a tile shape: rename (compute-accounting)",
    "verity/core/protocols/pouw/circuit/plan.py::Layout": "homonym, the units a plan reads: rename (compute-accounting)",
    "verity/core/protocols/pouw/coins.py::Challenger":
        "duplicate of `parties.Challenger`: becomes it (deleted) (compute-accounting)",
    "verity/core/protocols/pouw/lean/check/PouwCheck/Inputs.lean::PouwCheck.Law":
        "duplicate of `Flock.Draw.Law`: becomes it (deleted) (compute-accounting)",
    "verity/core/protocols/pouw/lean/check/PouwCheck/Inputs.lean::PouwCheck.Window":
        "duplicate of the execution block (no canonical yet: LEAN-93): becomes a reading of it: the execution block's record (compute-accounting)",
    "verity/core/protocols/pouw/lean/check/PouwCheck/Pass.lean::PouwCheck.Coins":
        "duplicate of the live coins (`Verity.Coins`; `SampledProofs.Challenger.LiveCommitment` in code): becomes a reading of it (compute-accounting)",
    "verity/core/protocols/pouw/lean/commit/PouwCommit/Frame.lean::PouwCommit.Hash":
        "homonym, a frame's hash kind: rename (compute-accounting)",
    "verity/core/protocols/pouw/protocol.py::Certificate":
        "homonym, what a scheme's guarantee rests on: rename (compute-accounting)",
    "verity/core/protocols/pouw/schemes/pearl_c_device.py::Device":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes a reading of it: the card by digest; prices stay the scheme's (compute-accounting)",
    "verity/core/protocols/pouw/schemes/pearl_kw.py::Device":
        "homonym, a code's parameters: rename (compute-accounting)",
    "verity/core/protocols/remote_bandwidth/audit.py::Ledger": "homonym, the γ cost terms: rename (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/audit.py::Record":
        "duplicate of the auditor's record chain (`SampledProofs.Challenger.LedgerEntry`): becomes a reading of it (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/audit.py::Session":
        "homonym, the old word for a run, kept as one protocol's run type: rename (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/audit.py::Store":
        "duplicate of `CircuitRequirement.Store`: becomes a reading of it: PoUB's view of PoUS's store (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/coins.py::Coins":
        "duplicate of the live coins (`Verity.Coins`; `SampledProofs.Challenger.LiveCommitment` in code): becomes a reading of it (the window's live challenge seeds its passes) (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/lean/RemoteAudit/P2.lean::RemoteAudit.P2.Opening":
        "duplicate of `Hash.RowLeaf.Opening` (`merkle.Opening` in Python): becomes a reading of it (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/lean/RemoteAudit/P2.lean::RemoteAudit.P2.Registration":
        "duplicate of the registration log (`Verity.Registration`, not yet in Core: LEAN-93): becomes the one store entry (merged) (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/lean/RemoteAudit/Registry.lean::RemoteAudit.Registry":
        "duplicate of the registration log (`Verity.Registration`, not yet in Core: LEAN-93): becomes the log's fold (deleted) (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/lean/RemoteAudit/Schedule.lean::RemoteAudit.Schedule.Block":
        "homonym, a chunk of the store: rename (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/lean/RemoteAudit/Window.lean::RemoteAudit.Opening":
        "duplicate of `Hash.RowLeaf.Opening` (`merkle.Opening` in Python): becomes a reading of it (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/lean/RemoteAudit/Window.lean::RemoteAudit.Verdict":
        "duplicate of `Core.Verdict`: becomes a reading of it (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/ledger.py::Draw": "homonym, a draw's cost terms: rename (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/ledger.py::Node":
        "duplicate of the Site (`Verity.Site`, not yet in Core: LEAN-93): becomes a reading of it: PoUB's projection (read and link rates) (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/opening.py::Draw":
        "homonym, a PoUB position draw: rename (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/opening.py::Session":
        "duplicate of `remote_bandwidth`'s own type (MEM-138): becomes `audit.Session`, `audit.Store` (folded) (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/opening.py::Store":
        "duplicate of `remote_bandwidth`'s own type (MEM-138): becomes `audit.Session`, `audit.Store` (folded) (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/p2.py::Opening":
        "duplicate of `Hash.RowLeaf.Opening` (`merkle.Opening` in Python): becomes it (one opening type, parameterised by codec) (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/p2.py::Registration":
        "duplicate of the registration log (`Verity.Registration`, not yet in Core: LEAN-93): becomes the one store entry (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/p2.py::Session":
        "duplicate of `remote_bandwidth`'s own type (MEM-138): becomes one type per concept, parameterised by codec (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/p2.py::Store":
        "duplicate of `CircuitRequirement.Store`: becomes a reading of it (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/period.py::Period":
        "duplicate of the period (`Verity.Period`, not yet in Core): becomes one `Core` period case (memory-accounting)",
    "verity/core/protocols/remote_bandwidth/store.py::Store":
        "duplicate of `CircuitRequirement.Store`: becomes a reading of it (memory-accounting)",
    "verity/core/protocols/sampled_proofs/one_stage/verdict.py::Verdict":
        "duplicate of `Core.Verdict`: becomes a reading of it (proofs)",
    "verity/core/protocols/sampled_proofs/optimize.py::Certificate":
        "homonym, what a sampling plan certifies: rename (proofs)",
    "verity/core/protocols/sampled_proofs/optimize.py::Commitment": "homonym, a tensor's commit cost: rename (proofs)",
    "verity/core/protocols/sampled_proofs/remote.py::Session": "homonym, the challenger's host session: rename (proofs)",
    "verity/core/protocols/sampled_proofs/service.py::Outcome":
        "duplicate of `Core.Verdict`: becomes a reading of it (proofs)",
    "verity/core/protocols/sampled_proofs/service.py::Registration":
        "duplicate of the registration log (`Verity.Registration`, not yet in Core: LEAN-93): becomes a reading of it: the service's admitted log entry (proofs)",
    "verity/core/protocols/sampled_proofs/service.py::Session":
        "homonym, the old word for a run, kept as one protocol's run type: rename (proofs)",
    "verity/core/service/flock/Flock/Firewall/Coins.lean::Flock.Firewall.Coins.Session":
        "homonym, a firewall release: rename (proofs)",
    "verity/core/service/flock/Flock/Firewall/Contract.lean::Flock.Firewall.Contract.Transcript":
        "duplicate of `Flock.Transcript`: becomes a reading of it (proofs)",
    "verity/core/service/flock/Flock/Firewall/Contract.lean::Flock.Firewall.Contract.Verdict":
        "homonym, the firewall's release decision: rename (proofs)",
    "verity/core/service/flock/Flock/Firewall/Frames.lean::Flock.Firewall.Frames.Registry":
        "homonym, the firewall loop's frames: rename (proofs)",
    "verity/core/service/flock/Flock/Firewall/Frames.lean::Flock.Firewall.Frames.Session":
        "homonym, a firewall release: rename (proofs)",
    "verity/core/service/flock/Flock/Firewall/Level0.lean::Flock.Firewall.Level0.Registry":
        "homonym, the firewall loop's tables: rename (proofs)",
    "verity/core/service/flock/Flock/Layout.lean::Flock.Layout": "homonym, C-Flock's block-type layout: rename (proofs)",
    "verity/core/service/flock/Flock/Program.lean::Flock.Program.Node":
        "homonym, a circuit program's node: rename (proofs)",
    "verity/core/service/flock/Flock/RecAlgebra.lean::Flock.RecAlgebra.Coins":
        "homonym, the recursion's coin sources: rename (proofs)",
    "verity/core/service/flock/Flock/RecCoins.lean::Flock.RecCoins.Opening":
        "homonym, a polynomial-commitment opening: rename (proofs)",
    "verity/core/service/flock/Flock/RecRecord.lean::Flock.RecRecord.Record":
        "duplicate of the auditor's record chain (`SampledProofs.Challenger.LedgerEntry`): becomes the challenger's live-coin entries (proofs)",
    "verity/core/service/flock/Flock/Record.lean::Flock.Session": "homonym, a proof's session record: rename (proofs)",
    "verity/core/service/flock/VStar/In.lean::VStar.In.Opening": "homonym, a Ligerito opening: rename (proofs)",
    "verity/core/service/flock/VStar/Stage.lean::VStar.Stmt": "homonym, one of V*'s statements: rename (proofs)",
    "verity/core/service/sampled_proofs/SampledProofs/Program.lean::SampledProofs.Program":
        "duplicate of `program.Program`, read from its descriptor's bytes: becomes its code-Lean side once a difftest holds the two (proofs)",
    "verity/core/service/sampled_proofs/SampledProofs/Slots.lean::SampledProofs.Slots.Layout":
        "homonym, the claims' partition: rename (proofs)",
    "verity/core/service/sampled_proofs/SampledProofs/Wires.lean::SampledProofs.Wires.Circuit":
        "homonym, an execution block's wiring: rename (proofs)",
    "verity/core/wire/__init__.py::Registry": "homonym, message types by version: rename (proofs)",
    # verity/ml/
    "verity/ml/circuits/annotations.py::Verdict": "homonym, an annotation check's result: rename (proofs)",
    "verity/ml/circuits/boolean/forms.py::Circuit":
        "duplicate of `Flock.Circuit`: becomes its Python mirror, held by a difftest (proofs)",
    "verity/ml/circuits/defs.py::Node": "homonym, one body node of a Definition: rename (proofs)",
    "verity/ml/circuits/defs.py::Registry": "homonym, the registered Definitions by id: rename (proofs)",
    "verity/ml/circuits/layout.py::Circuit":
        "duplicate of `Flock.Circuit`: becomes its Python mirror, held by a difftest (proofs)",
    "verity/ml/circuits/layout.py::Gate":
        "duplicate of `Flock.Program`'s gate: becomes its Python mirror, held by a difftest (proofs)",
    "verity/ml/circuits/partition_object.py::Partition":
        "duplicate of `Verity.Partition`: becomes its Python mirror, held by a difftest (proofs)",
    "verity/ml/circuits/requirement.py::Layout":
        "homonym, each value's place in the store: rename (`Accounting.Requirement.Layout`'s mirror) (proofs)",
    "verity/ml/circuits/stacking.py::Verdict": "homonym, whether a stacking property holds: rename (proofs)",
    "verity/ml/circuits/stacking.py::Window": "homonym, a stacking property's window: rename (proofs)",
    "verity/ml/circuits/types.py::Record": "homonym, a record type of the circuit type system: rename (proofs)",
    "verity/ml/circuits/units.py::Placement": "homonym, a unit's index in the partition: rename (proofs)",
    "verity/ml/commitments/advice.py::Opening":
        "duplicate of `Hash.RowLeaf.Opening` (`merkle.Opening` in Python): becomes a reading of it (proofs)",
    "verity/ml/commitments/advice.py::Registration":
        "homonym, a hiding commitment to a private partition's rows: rename (`Advice.Committed`) (proofs)",
    "verity/ml/commitments/layout.py::Layout": "homonym, a step's rows per slot: rename (proofs)",
    "verity/ml/flock/pod/gf2.py::Circuit": "homonym, a pod prover's GF(2) gate recorder: rename (proofs)",
    "verity/ml/flock/python/verity_flock/boolean_export.py::Registry":
        "homonym, the export's circuit types by digest: rename (proofs)",
    "verity/ml/flock/python/verity_flock/firewall_transcript.py::Session":
        "homonym, one inner proof through the firewall: rename (proofs)",
    "verity/ml/flock/python/verity_flock/ir_lower.py::Layout":
        "homonym, a unit circuit laid out as one Flock block type: rename (proofs)",
    "verity/ml/flock/python/verity_flock/layouts.py::Layout":
        "homonym, C-Flock's block-type layout (`Flock.Layout`'s pair): rename (proofs)",
    "verity/ml/flock/python/verity_flock/rec_algebra.py::Coins":
        "homonym, the recursion's coin sources in draw order: rename (proofs)",
    "verity/ml/flock/python/verity_flock/rec_algebra.py::Layout":
        "homonym, a GF residual gadget's ports: rename (proofs)",
    "verity/ml/flock/python/verity_flock/rec_sparse.py::Coins":
        "homonym, a rep's public values the sparse phase reads: rename (proofs)",
    "verity/ml/windows/interface.py::Circuit":
        "duplicate of `Flock.Circuit`: becomes a reading of it in proof blocks (proofs)",
    "verity/ml/windows/interface.py::Layout": "homonym, a template's bit rows: rename (proofs)",
    "verity/ml/windows/interface.py::Opening": "homonym, a template's open read of a table row: rename (proofs)",
    "verity/ml/windows/interface.py::Window":
        "duplicate of the execution block (no canonical yet: LEAN-93): becomes the builder's output for it (proofs)",
    "verity/ml/windows/rows.py::Window":
        "duplicate of the execution block (no canonical yet: LEAN-93): becomes `interface.Window` (merged) (proofs)",
}

_REASON = re.compile(r"^(?:duplicate of .+?: becomes .+|homonym, .+?: rename.*) \(([\w-]+)\)$")


@functools.cache
def _scanned() -> Scan:
    return scan()


def classify(s: Scan) -> dict[str, tuple[str, Hit, str]]:
    """Each hit's key: ``(concept, hit, status)``, the status what the guard makes of it (this module's docstring), or
    ``unlisted``."""
    canonical, protocol, out = {k for keys in CANONICAL.values() for k in keys}, slots(s), {}
    keys = {h.key for defs in s.hits.values() for h in defs}
    for concept, defs in s.hits.items():
        for h in defs:
            out[h.key] = (concept, h, "canonical" if h.key in canonical
                          else "mirror" if MIRRORS.get(h.key, ("",))[0] in keys
                          else "reads the canonical" if reads_canonical(h, set(CANONICAL.get(concept, ())), s)
                          else "a protocol's slot" if h.key in protocol else "known" if h.key in KNOWN else "unlisted")
    return out


def stale(found: dict[str, tuple[str, Hit, str]]) -> tuple[list[str], list[str]]:
    """The ``KNOWN`` lines and the ``MIRRORS`` entries that no longer match a hit of `found` (``classify``'s)."""
    return ([k for k in KNOWN if k not in found or found[k][2] != "known"],
            [py for py, (lean, _, _) in MIRRORS.items() if py not in found or lean not in found])


_ENTRY = re.compile(r'^    "([^"\n]+)":(?: "[^\n]*",| *\n        "[^\n]*",)\n', re.M)
_EMPTY_HEADING = re.compile(r"^    # [^\n]*\n(?=    # |\Z)", re.M)
_KNOWN_OPEN, _KNOWN_CLOSE = "\nKNOWN: dict[str, str] = {\n", "\n}\n"
_KNOWN_MAX = re.compile(r"^KNOWN_MAX = \d+$", re.M)


def pruned(source: str, drop: set[str]) -> str:
    """`source`, this module's text, with the ``KNOWN`` lines of the keys in `drop` deleted, the headings they leave empty,
    and ``KNOWN_MAX`` set to the lines left."""
    head, _, rest = source.partition(_KNOWN_OPEN)
    block, _, tail = rest.partition(_KNOWN_CLOSE)
    block = _EMPTY_HEADING.sub("", _ENTRY.sub(lambda m: "" if m.group(1) in drop else m.group(0), block + "\n"))
    head = _KNOWN_MAX.sub(f"KNOWN_MAX = {len(_ENTRY.findall(block))}", head)
    return head + _KNOWN_OPEN + block + _KNOWN_CLOSE.lstrip("\n") + tail


def _read(concept: str) -> str:
    keys = CANONICAL.get(concept, ())
    return (" or ".join(f"`{k}`" for k in keys) if keys
            else "the canonical (it has none yet: name one in CANONICAL, under Security/Definitions/Core/)")


def test_every_hit_is_canonical_or_listed():
    found = classify(_scanned())
    assert len(found) >= 300, f"the scan found only {len(found)} declarations of the concepts: has the tree moved?"
    unlisted = [f"{k} (line {h.line}) declares {concept} again: read {_read(concept)}, or rename it if it is another thing"
                for k, (concept, h, status) in sorted(found.items()) if status == "unlisted"]
    assert not unlisted, (f"declarations of a concept that are not its canonical object ({FIX}):\n  "
                          + "\n  ".join(unlisted))


def test_known_shrinks_by_count():
    assert len(KNOWN) <= KNOWN_MAX, (f"KNOWN has {len(KNOWN)} lines, above KNOWN_MAX = {KNOWN_MAX}: rename or read the "
                                     "canonical; a new KNOWN line needs lean's review, and raises KNOWN_MAX in the same PR")


def test_known_reasons_name_their_owner():
    """Every line's reason has the form; a line that still matches a hit names its path's owner."""
    from research.slack import owner
    registry, found = json.loads((ROOT / REGISTRY).read_text()), classify(_scanned())
    bad = []
    for k, why in KNOWN.items():
        m, own = _REASON.match(why), owner(registry, k.partition("::")[0])
        if not m:
            bad.append(f'{k}: the reason is "duplicate of X: becomes …" or "homonym, what it is: rename …", then "({own})"')
        elif k in found and found[k][2] == "known" and m.group(1) != own:
            bad.append(f"{k}: owned by {own} ({REGISTRY}), not {m.group(1)}")
    assert not bad, "\n  ".join(["KNOWN reasons:"] + bad)


def test_canonical_entries():
    concept_of = {h.key: c for c, defs in _scanned().hits.items() for h in defs}
    assert set(CANONICAL) == set(CONCEPTS), "CANONICAL has an entry for each concept, and only those"
    bad = []
    for concept, keys in CANONICAL.items():
        worlds = defaultdict(list)
        for k in keys:
            worlds[_world(k.partition("::")[0])].append(k)
            if concept_of.get(k) != concept:
                bad.append(f"{k}: no longer declares {concept}; name where it went")
        bad += [f"{concept}: {len(ks)} canonicals in {w}, one at most: list the others in KNOWN"
                for w, ks in worlds.items() if len(ks) > 1]
        bad += [f"{k}: a spec canonical sits under {' or '.join(SPEC)}" for k in worlds["spec"] if not k.startswith(SPEC)]
        if worlds["code"] and worlds["py"] and MIRRORS.get(worlds["py"][0], ("",))[0] != worlds["code"][0]:
            bad.append(f"{concept}: its code-Lean and Python canonicals are one pair in MIRRORS, with their difftest port")
    assert not bad, "\n  ".join(["CANONICAL:"] + bad)


def _lean_exes() -> set[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--", "*lakefile.toml"], cwd=ROOT, check=True,
                         capture_output=True).stdout
    return {exe["name"] for p in out.split(b"\0") if p
            for exe in tomllib.loads((ROOT / p.decode()).read_text()).get("lean_exe", [])}


def test_mirrors_name_their_port():
    concept_of, exes = {h.key: c for c, defs in _scanned().hits.items() for h in defs}, _lean_exes()
    bad = []
    for py, (lean, port, why) in MIRRORS.items():
        if _world(py.partition("::")[0]) != "py" or _world(lean.partition("::")[0]) != "code":
            bad.append(f"{py}: a mirror is a Python class held to a code-Lean declaration")
        elif py in concept_of and lean in concept_of and concept_of[lean] != concept_of[py]:
            bad.append(f"{py}: it and {lean} declare different concepts; fix the entry's keys")
        if port not in exes:
            bad.append(f"{py}: {port!r} is no `[[lean_exe]]` of a tracked lakefile.toml; name the port that tests the pair")
        if not why:
            bad.append(f"{py}: say what the port checks")
    assert not bad, "\n  ".join(["MIRRORS:"] + bad)


def test_the_scan_reads_lean_declarations():
    src = """namespace Pouw
/-- A `structure Coins` in a doc comment is not a declaration. -/
-- def Circuit : Type := Unit
structure Coins where
  bits : Nat
@[reducible] def Prog (n : Nat) : Type := Fin n
def Gate (n : Nat) : Nat := n
noncomputable abbrev Foo.Digest := ByteArray
def «term» := 1
section
inductive Op | a | b
end
mutual
inductive Dag | leaf
end
opaque Hash : Type
end Pouw
class inductive Law (α : Type) : Type
def Transcript {α : Type} (x : α) : Sort 1 := PUnit
def Witness : Nat → Type := fun _ => Unit
structure CoinsFor where
def Stmt : Type → Nat := fun _ => 0
def Coins : Prop := True
def Sha512.Collision (H : Nat → Nat) : Prop := ∃ u v, u ≠ v ∧ H u = H v
def Gate.ok (g : Nat) : Type := Unit
opaque Witness.Node : Nat
"""
    assert [(c, h.key, h.line) for c, h in lean_definitions(src)[0]] == [
        ("Coins", "Pouw.Coins", 4), ("Prog", "Pouw.Prog", 6), ("Digest", "Pouw.Foo.Digest", 8), ("Op", "Pouw.Op", 11),
        ("Dag", "Pouw.Dag", 14), ("Hash", "Pouw.Hash", 16), ("Law", "Law", 18), ("Transcript", "Transcript", 19),
        ("Witness", "Witness", 20), ("Collision", "Sha512.Collision", 24)]


def test_the_scan_reads_python_classes():
    src = """class Coins:
    class Draw(Base):
        pass

    def method(self):
        class Verdict:
            pass

def helper():
    class Node:
        pass

class Window: ...
class CoinsFor: ...
"""
    assert [(c, h.key, h.line) for c, h in python_definitions(src)] == [
        ("Coins", "Coins", 1), ("Draw", "Coins.Draw", 2), ("Window", "Window", 13)]


def test_readings_and_slots_are_exempt():
    """An abbrev of the canonical, or a list of it, reads it; a protocol's `Commitment := X` names its slot."""
    core, pouw = f"{CORE}Draw.lean", f"{D}Pouw/Protocol.lean"
    s = scan({core: "namespace Verity\nstructure Coins where\n  bits : Nat\nend Verity\n",
              pouw: "namespace Pouw\nopen Verity\nabbrev Coins := List Coins\nabbrev Draw := Nat\n"
                    "structure Registration where\n  x : Nat\n"
                    "def protocol : Core.Protocol where\n  Params := Nat\n  Commitment := Registration\nend Pouw\n"})
    reads = {h.key for h in s.hits["Coins"] if reads_canonical(h, {f"{core}::Verity.Coins"}, s)}
    assert reads == {f"{pouw}::Pouw.Coins"}
    assert not reads_canonical(s.hits["Draw"][0], {f"{core}::Verity.Coins"}, s)
    assert slots(s) == {f"{pouw}::Pouw.Registration"}


def test_prune_deletes_stale_lines():
    src = ('KNOWN_MAX = 4\nKNOWN: dict[str, str] = {\n    # a/\n    "a::A": "homonym, a: rename (o)",\n    # b/\n'
           '    "b::B":\n        "homonym, b: rename (o)",\n    "b::C": "homonym, c: rename (o)",\n}\n\ny = 2\n')
    assert pruned(src, set()) == src.replace("KNOWN_MAX = 4", "KNOWN_MAX = 3")
    assert pruned(src, {"a::A", "b::B"}) == ('KNOWN_MAX = 1\nKNOWN: dict[str, str] = {\n    # b/\n'
                                             '    "b::C": "homonym, c: rename (o)",\n}\n\ny = 2\n')


def test_prune_reads_every_known_line():
    """`--prune` finds each ``KNOWN`` line in this file as written, so it can delete any of them."""
    block = Path(__file__).read_text(encoding="utf-8").partition(_KNOWN_OPEN)[2].partition(_KNOWN_CLOSE)[0] + "\n"
    assert [m.group(1) for m in _ENTRY.finditer(block)] == list(KNOWN), (
        'each KNOWN line is `    "key": "reason",`, or the key alone with the reason on the next line, indented 8')


if __name__ == "__main__":
    found = classify(scan())
    known, mirrors = stale(found)
    if sys.argv[1:] == ["--prune"]:
        path = Path(__file__).resolve()
        path.write_text(pruned(path.read_text(encoding="utf-8"), set(known)), encoding="utf-8")
        print(f"deleted {len(known)} stale KNOWN lines: KNOWN has {len(KNOWN) - len(known)}, and so does KNOWN_MAX"
              + "".join(f"\nstale MIRRORS entry, delete by hand: {py}" for py in mirrors))
        sys.exit()
    if sys.argv[1:]:
        sys.exit("usage: python tests/test_lean_vocabulary.py [--prune]")
    for key, (concept, h, status) in sorted(found.items(), key=lambda e: (e[1][0], e[0])):
        print(f"{concept:<18} {status:<20} {key}:{h.line}" + (f"  {KNOWN[key]}" if key in KNOWN else ""))
    print(f"\nKNOWN: {len(KNOWN)} lines, KNOWN_MAX {KNOWN_MAX}\nstale (delete with --prune):")
    for k in known:
        print(f"  {k}: " + ("not declared" if k not in found else f"now {found[k][2]}"))
    for py in mirrors:
        print(f"  MIRRORS {py}: no longer a pair of hits (delete by hand)")
