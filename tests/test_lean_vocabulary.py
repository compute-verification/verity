"""One vocabulary (AGENTS.md "Lean", Daniel, 25 Oct): a concept that more than one guarantee reads has one definition in the
spec, in `Security/Definitions/Core` or `Security/Definitions/Hash`, and every area and protocol reads that one.

A type-forming declaration (`structure`, `inductive`, `class`, `abbrev`, `opaque`, or a `def` whose type is `Type` or `Sort`,
or `Prop` for ``PROPOSITIONS``) named for a concept in ``CONCEPTS`` (the whole name or its last component, `Foo.Coins`;
`python tests/test_lean_vocabulary.py` lists every definition the scan finds) may exist once in the Lean under
`Security/` (definitions, properties and proofs). When a concept has several, the one under ``CANONICAL`` (if exactly one is
there and ``KNOWN`` doesn't list it) is its definition, and every other is a duplicate that ``KNOWN`` must list. ``KNOWN`` only
shrinks: an entry that is no longer a duplicate fails too.

The Lean outside `Security/` is code (C-Flock's verifier, the protocol checkers, the devices' and primitives' ports): it never
requires `Security/` (`tests/test_lean_packages.py`), so it parses into types of its own, and isn't scanned."""
from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "Security"
CANONICAL = ("Security/Definitions/Core/", "Security/Definitions/Hash/")
FIX = "define it once in Security/Definitions/Core (or Hash) and read that one; see AGENTS.md 'One vocabulary'"

#: The concepts more than one guarantee reads.
CONCEPTS = ("Circuit", "Prog", "Program", "Gate", "Op", "Netlist", "Partition", "Coins", "CoinTree", "Draw", "Law", "Stmt",
            "Statement", "Witness", "Digest", "Commitment", "Merkle", "Collision", "Device", "InteractiveDevice", "Dag",
            "Transcript", "Hash", "Signature")
#: The concepts that are propositions, so a `def … : Prop` of their name defines them too.
PROPOSITIONS = ("Collision",)

#: Today's duplicates, ``"path::Qualified.Name": "reason; wave N"``, where wave N of the consolidation removes the entry:
#: 2 the circuit and what is stated over it, 3 coins and draws, 4 hashes and commitments, 5 devices.
D, P = "Security/Definitions/", "Security/Proofs/"
KNOWN: dict[str, str] = {
    # Circuit: Core/Circuit's `Verity.Circuit`.
    f"{D}Core/Boolean.lean::Verity.Boolean.Circuit": "the Boolean profile's circuit beside Core/Circuit's, in Core; wave 2",
    f"{D}Claims.lean::Claims.Circuit": "the claims layer's AND/XOR wire circuit; wave 2",
    f"{D}Flock/Circuit.lean::FlockSoundness.Audit.Circuit": "C-Flock's audit circuit, a copy of Core/Boolean's; wave 2",
    f"{D}Nci/Cover.lean::Nci.Cover.Circuit": "NCI's circuit with registered parameters; wave 2",
    # Gate
    f"{D}Claims.lean::Claims.Gate": "the claims layer's AND/XOR gate; wave 2",
    f"{D}Accounting/Gates/Model.lean::Accounting.Gates.Gate": "the gate accounting's annotated gate; wave 2",
    f"{D}Pouw/PearlC/TileCircuit.lean::Pouw.PearlC.TileCircuit.Gate": "Pearl-C's word-circuit gate; wave 2",
    f"{D}Pouw/TileBound/Defs.lean::Pouw.TileBound.Gate": "one m16n8k64 tile firing, a name clash: rename; wave 2",
    f"{P}Flock/Soundness/GateRows.lean::FlockSoundness.GateRows.Gate": "C-Flock's gate rows' AND/XOR/NOT gate; wave 2",
    f"{P}Flock/Soundness/Types/Flatten.lean::FlockSoundness.Types.Flatten.Gate": "C-Flock's flattened gate; wave 2",
    f"{P}Pouw/RelativeSecurity.lean::Pouw.RelativeSecurity.ModelB.Gate": "RS-MODEL's proof-local gate; wave 2",
    # Op: Core/Boolean's `Verity.Boolean.Op`.
    f"{D}Accounting/Device.lean::Accounting.Op": "the op classes a step charges, a name clash: rename; wave 2",
    f"{D}Accounting/Gates/Model.lean::Accounting.Gates.Op": "the gate accounting's operation over input slots; wave 2",
    f"{D}Flock/Circuit.lean::FlockSoundness.Audit.Op": "C-Flock's audit gate op, a copy of Core/Boolean's; wave 2",
    f"{D}Pouw/Assumptions/Dimension/Embed.lean::Pouw.Dimension.Embed.Op": "an M4090 instruction, a name clash: rename; wave 2",
    # Partition: Core/Partition's `Verity.Partition`.
    f"{D}Claims/Sampling.lean::Claims.Sampling.Partition": "the sampling claim's classes, strata and units; wave 2",
    f"{D}Flock/Circuit.lean::FlockSoundness.Audit.Partition": "C-Flock's audit partition, a copy of Core's; wave 2",
    # Prog
    f"{D}Claims/Sampling.lean::Claims.Sampling.Prog": "the sampling claim's flattened program (#4330's `Lowers`); wave 2",
    f"{D}Pous/Game/Oracle.lean::Pous.Prog": "PoUS's parallel oracle program, a name clash: rename; wave 2",
    f"{D}Pouw/Eps1/M3.lean::Pouw.Eps1.Prog": "an M3 program for Dither's words; wave 2",
    f"{D}Pouw/PearlC/W1.lean::Pouw.PearlC.W1.Prog": "Pearl-C's W1 program, a constant bank and a code tree; wave 2",
    f"{P}Flock/Soundness/Lowering.lean::FlockSoundness.Prog": "C-Flock's program of template units; wave 2",
    f"{P}Pouw/RelativeSecurity.lean::Pouw.RelativeSecurity.ModelB.Prog": "RS-MODEL's proof-local program; wave 2",
    # Program
    f"{D}Accounting/CoinFlow.lean::Accounting.CoinFlow.Program": "a Program's dataflow over coins and values; wave 2",
    f"{D}NetworkCertifier/Run.lean::NetTiming.Program": "a committed Program's egress prediction; wave 2",
    # Stmt
    f"{D}Claims.lean::Claims.Flock.Stmt": "Flock's statement in the claims layer; wave 2",
    f"{D}Claims/Recursion.lean::Claims.Recursion.Stmt": "the recursion's statement; wave 2",
    f"{D}Claims/Sampling.lean::Claims.Sampling.Stmt": "the sampling statement; wave 2",
    f"{D}Flock/Accounting/Bound.lean::FlockSoundness.Accounting.Stmt": "the statement's shape for C-Flock's cost bound; wave 2",
    # Transcript: Core/Circuit's `Verity.Circuit.Transcript`.
    f"{D}Pouw/Game/Defs.lean::Pouw.Transcript": "PoUW's committed transcript of unit commitments, a name clash; wave 2",
    f"{D}Pouw/PearlC/Game.lean::Pouw.PearlC.Transcript": "Pearl-C's copy of PoUW's committed transcript; wave 2",
    f"{P}Flock/Soundness/Discharge/Exec/Event.lean::FlockSoundness.Discharge.Exec.Transcript":
        "a session directory's files, a name clash: rename; wave 2",
    # Witness: Core/TC/Relation's `Verity.TC.Relation.Witness`, a tensor-core step's witness until wave 2 settles one.
    f"{D}Flock/Statement.lean::FlockSoundness.Model.Witness": "C-Flock's statement's witness bits; wave 2",
    # Coins
    f"{D}Accounting/BoundReceipt.lean::Accounting.BoundReceipt.Coins": "a bound receipt window's coins; wave 3",
    f"{D}Accounting/RateAudit.lean::Accounting.Rate.Coins": "a rate audit window's coins; wave 3",
    f"{D}Accounting/RateAuditSketch.lean::Accounting.RateSketch.Coins": "the rate sketch window's coins; wave 3",
    f"{D}Accounting/Remote/Draw.lean::Accounting.Remote.Draw.Coins": "a remote audit session's coins; wave 3",
    f"{D}Accounting/Remote/Sketch.lean::Accounting.Remote.Sketch.Coins": "the remote sketch model's coins; wave 3",
    f"{P}Flock/Claims/Complete/Replay/Lincheck.lean::FlockComplete.Replay.Lc.Coins": "the coins lincheck reads; wave 3",
    f"{P}Flock/Claims/Complete/Replay/RingSwitch.lean::FlockComplete.Replay.Rs.Coins": "the coins ring switch reads; wave 3",
    f"{P}Flock/Claims/Complete/Replay/Zerocheck.lean::FlockComplete.Replay.Zc.Coins": "the coins zerocheck reads; wave 3",
    f"{P}Flock/Claims/Complete/Zk/InnerGo.lean::FlockComplete.Replay.InZ.Coins": "the ZK inner proof's coins; wave 3",
    # CoinTree
    f"{P}Flock/Claims/Bridge/Committed.lean::Claims.Bridge.CoinTree": "the claims bridge's coin tree over k rounds; wave 3",
    f"{P}Flock/Soundness/ZK/GK/Extract.lean::FlockSoundness.ZK.GK.CoinTree": "the ZK extractor's coin tree; wave 3",
    # Draw
    f"{D}Accounting/Gates/Codec.lean::Accounting.Gates.Draw": "a k-round key's draw, a name clash: rename; wave 3",
    f"{D}Claims/Recursion.lean::Claims.Recursion.Draw": "a V* session's draw as its record holds it; wave 3",
    f"{D}Population/DrawPressure.lean::Population.Exfil.Draw": "k of n units without replacement; wave 3",
    # Law
    f"{D}Pouw/Window.lean::Pouw.Window.Law": "PoUW's draw law over units; wave 3",
    f"{P}Flock/Soundness/Audit/Law.lean::FlockSoundness.Audit.Law": "C-Flock's audit's copy of the draw law; wave 3",
    # Digest: Hash/Basic's `Hash.Digest`.
    f"{D}Core/Advice.lean::Verity.Advice.Digest": "the advice's 64-byte digest beside Hash's, in Core; wave 4",
    # Collision
    f"{D}Claims.lean::Claims.Collision": "a collision of a hash on messages; wave 4",
    f"{D}Flock/Merkle.lean::FlockSoundness.Merkle.Collision": "a collision of a hash on byte strings; wave 4",
    f"{P}Flock/Verifier/Hash.lean::Flock.Sha256.Collision": "a SHA-256 collision; wave 4",
    f"{P}Flock/Verifier/Hash.lean::Flock.Sha512.Collision": "a SHA-512 collision; wave 4",
    f"{P}Flock/Verifier/Merkle.lean::Flock.MerkleScheme.Collision": "a collision of a Merkle scheme's hash; wave 4",
    # Device
    f"{D}Accounting/Device.lean::Accounting.Device": "the resource audits' device by its limits, not in Core; wave 5",
    f"{D}Accounting/RateAudit.lean::Accounting.Rate.Device": "the rate audit's device; wave 5",
    f"{D}Accounting/RateAuditSketch.lean::Accounting.RateSketch.Device": "the rate sketch's device; wave 5",
    f"{D}Accounting/Bandwidth/Game.lean::Bandwidth.Device": "the bandwidth game's device; wave 5",
    "Security/Properties/Pous/MemoryChallenger.lean::MemoryChallenger.Properties.Device": "the memory challengers; wave 5",
    # InteractiveDevice
    f"{D}Accounting/RateAuditInteractive.lean::Accounting.Rate.InteractiveDevice": "the rate audit's; wave 5",
    f"{D}Accounting/RateAuditSketchInteractive.lean::Accounting.RateSketch.InteractiveDevice": "the rate sketch's; wave 5",
    f"{D}Accounting/Bandwidth/Interactive.lean::Bandwidth.InteractiveDevice": "the bandwidth game's; wave 5",
}

_NAME = rf"((?:[^\s(\[{{:]*\.)?(?:{'|'.join(sorted(CONCEPTS, key=len, reverse=True))}))(?![\w'.!?])"
_KEYWORD = r"(structure|class\s+inductive|class|inductive|abbrev|opaque|def)"
_MODS = r"(?:(?:private|protected|public|noncomputable|partial|unsafe|nonrec|meta)\s+)*"
# One pass over every file: a leading `\b` or `^` here costs Python's re its literal-prefix scan (2.5x slower).
_CANDIDATE = re.compile(rf"{_KEYWORD}\s+{_NAME}")
_DECL = re.compile(rf"^[ \t]*(?:@\[[^\]\n]*\][ \t]*)*{_MODS}{_KEYWORD}\s+{_NAME}", re.M)
_SCOPE = re.compile(rf"^[ \t]*(?:{_MODS}(namespace|section)\b[ \t]*(\S*)|(mutual)\b|end\b)", re.M)
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


def returns_type(text: str, start: int, sort: re.Pattern = _SORT) -> bool:
    """Whether the declaration header from `start` (just past its name) ends its type in `Type …` or `Sort …` (`sort`)."""
    depth, colon = 0, None
    for m in _HEADER_TOKEN.finditer(text, start, min(len(text), start + 4000)):
        t = m.group()
        if t in "([{⦃⟨":
            depth += 1
        elif t in ")]}⦄⟩":
            depth -= 1
        elif depth == 0 and t != ":":
            return colon is not None and sort.search(text[colon:m.start()]) is not None
        elif depth == 0:
            colon = m.end()
    return False


def definitions(text: str) -> list[tuple[str, str, int]]:
    """``(concept, qualified name, line)`` for each type-forming declaration of a concept in the Lean source `text`."""
    if not _CANDIDATE.search(text):
        return []
    text = strip_comments(text)
    events = sorted([(m.start(), m) for m in _DECL.finditer(text)] + [(m.start(), m) for m in _SCOPE.finditer(text)],
                    key=lambda e: e[0])
    scopes, out = [], []
    for pos, m in events:
        if m.re is _SCOPE:
            if m.group(3) or m.group(1):
                scopes.append(m.group(2) if m.group(1) == "namespace" else "")
            elif scopes:
                scopes.pop()
            continue
        keyword, name = m.group(1), m.group(2)
        concept = name.rpartition(".")[2]
        if keyword == "def" and not returns_type(text, m.end(), _SORT_OR_PROP if concept in PROPOSITIONS else _SORT):
            continue
        qualified = ".".join([s for s in scopes if s] + [name.removeprefix("_root_.")])
        out.append((concept, qualified, text.count("\n", 0, pos) + 1))
    return out


def spec_definitions() -> dict[str, list[tuple[str, int]]]:
    """For each concept, its definitions under `Security/`: ``("path::Qualified.Name", line)``."""
    found = defaultdict(list)
    for d, dirs, files in os.walk(SPEC):
        dirs[:] = sorted(x for x in dirs if not x.startswith("."))
        rel = Path(d).relative_to(ROOT).as_posix()
        for f in sorted(files):
            if f.endswith(".lean"):
                for concept, qualified, line in definitions((Path(d) / f).read_text(encoding="utf-8")):
                    found[concept].append((f"{rel}/{f}::{qualified}", line))
    return found


def duplicates(found, known) -> dict[str, list[tuple[str, int]]]:
    """For each concept defined more than once, every definition but its canonical one."""
    out = {}
    for concept, defs in found.items():
        if len(defs) > 1:
            canon = [d for d in defs if d[0].startswith(CANONICAL) and d[0] not in known]
            out[concept] = [d for d in defs if len(canon) != 1 or d != canon[0]]
    return out


def test_concepts_are_defined_once():
    found = spec_definitions()
    assert len(found) >= 10, f"the scan found definitions of only {sorted(found)}: has Security/ moved?"
    dups = duplicates(found, KNOWN)
    new = [f"{key}:{line} ({concept})" for concept, defs in sorted(dups.items()) for key, line in defs if key not in KNOWN]
    assert not new, f"a second definition of a shared concept: {FIX}.\n  " + "\n  ".join(new)


def test_known_only_shrinks():
    live = {key for defs in duplicates(spec_definitions(), KNOWN).values() for key, _ in defs}
    stale = sorted(set(KNOWN) - live)
    assert not stale, (f"KNOWN entries that are no longer duplicates (gone, renamed, or now the only one): remove them from "
                       f"KNOWN in {Path(__file__).relative_to(ROOT)}.\n  " + "\n  ".join(stale))


def test_every_known_entry_has_a_reason_and_a_wave():
    bad = [k for k, why in KNOWN.items() if not re.search(r"\S.*; wave \d+$", why)]
    assert not bad, f'each KNOWN reason is one line ending in "; wave N", the wave that removes it: {bad}'


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
"""
    assert definitions(src) == [
        ("Coins", "Pouw.Coins", 4), ("Prog", "Pouw.Prog", 6), ("Digest", "Pouw.Foo.Digest", 8), ("Op", "Pouw.Op", 11),
        ("Dag", "Pouw.Dag", 14), ("Hash", "Pouw.Hash", 16), ("Law", "Law", 18), ("Transcript", "Transcript", 19),
        ("Witness", "Witness", 20), ("Collision", "Sha512.Collision", 24)]


def test_canonical_and_known():
    """One definition under Core or Hash is the concept's; two there are both duplicates until ``KNOWN`` lists one."""
    core, pouw, hash_ = (f"{D}Core/C.lean::Verity.Coins", 1), (f"{D}Pouw/A.lean::Pouw.Coins", 2), (f"{D}Hash/B.lean::H.Coins", 3)
    found = {"Coins": [core, pouw], "Draw": [(f"{D}Pouw/A.lean::Pouw.Draw", 4)]}
    assert duplicates(found, {}) == {"Coins": [pouw]}
    assert duplicates({"Coins": [core, pouw, hash_]}, {}) == {"Coins": [core, pouw, hash_]}
    assert duplicates({"Coins": [core, pouw, hash_]}, {hash_[0]: ""}) == {"Coins": [pouw, hash_]}


if __name__ == "__main__":
    for concept, defs in sorted(spec_definitions().items()):
        for key, line in defs:
            print(f"{concept:18} {key}:{line}")
