"""A verifier costs O(circuit description), never O(unrolled circuit) (AGENTS.md "Design rules").

A scan over the Lean under `verity/core/` (the verifiers and the code they run) and over `Security/`'s verifier code
models (each `Security/Proofs/**/Code.lean`, a verifier a guarantee is stated at, with the `Security` modules it imports
and the `Properties` modules those import) refuses the forms an unrolled walk takes there: locating gates one by one
through the canonical layout (`P.gate g`, `Flock.Program.Prog.gate`), a range or loop bounded by a Definition's gate count
(`List.range (P.defs[P.root]!).gates`, `[0:defn.gates]`, `List.range C.gates.length`), an array or loop bounded by a name
bound to one (`let n := defn.gates` … `Array.mkEmpty n`, `for j in [0:n]`; `let gs := C.gates.toArray` … `List.range
gs.size`), a range or loop over a key program's gates, inputs or units (`List.range kp.size`, `List.range kp.units`: a
`Spot.Bits.KeyProg` lays a member's units out side by side), a range or
loop over a node's members (`for m in [0:node.members]`, `List.range node.members`: every batch member or scan iteration),
and outside the tests a templated unit unrolled (`Tmpl.unroll`, `Tmpl.unrollAt`, `Tmpl.unitFull`, `Tmpl.posMap`, `Tmpl.foldRef`):
a templated unit is read from its description (`Tmpl.unrollOk`, `Tmpl.unitNet`, `Tmpl.foldUnit`), or a member's key
program built (`Spot.Bits.keyProg c n`, every one of its `n` units laid out). Outside the tests it
also refuses a template query's units' gates asked one by one under a map over the units (`(Array.range pop).mapM fun u
=> … TemplateQuery.gates d rs u`), each of which walks every earlier root node: they are read a range at a time
(`TemplateQuery.unitGates`). Outside the tests it
also refuses the recursive inner's two: the comb (`.folded`) in a file that sets the inner statement up from its flat
file (`Stmt.setupTables innerTags …`), which `partialRangesT` folds row by row since a flat file carries no template, and
V*'s circuit text emitted outside V*'s namespace (`VStar.emitCircuitOf`), the whole unit laid out again. The fix is in the
scan's failure message: summarize once per specialized Definition and multiply through batch members and scan
iterations, as `NciCheck.Desc` does (its docstring), so a Definition's summary serves every activation of it.

What may stay is listed in ``ALLOWED``, one line of reason each: the gate-by-gate references a description-sized
verifier is difftested against, and the walks still open. An entry that no longer matches anything fails too.
"""

from __future__ import annotations

import re

from test_repository import ROOT, _tracked

ALLOWED: dict[str, str] = {
    "verity/definitions/evaluate/Evaluate/Walk.lean":
        "Evaluate.walk, the evaluator: a Program's values gate by gate, every member of every node, by definition; no "
        "verifier imports it",
    "verity/protocols/inference_only/verifier/nci/lean/NciCheck/Desc.lean":
        "NciCheck.Desc's members of a node reading part of a mapped argument, one by one: bounded by walkAtMost, "
        "which refuses past its limit",
    "verity/protocols/inference_only/verifier/nci/lean/NciCheck/DescFuzz.lean":
        "NciCheck.DescFuzz, the desc fuzzer the Lean audit runs (NciCheck.DescFuzzMain): it visits every member of every "
        "node of a generated Program, under caps.work, to decide DescAgrees' open statements there; the verifier never runs it",
    "verity/protocols/inference_only/verifier/nci/lean/NciCheck/Check.lean":
        "NciCheck.check, the gate-by-gate reference NciCheck.Desc is difftested against (runs NciCheck.DescDifftestMain); "
        "`nci-check` runs it only with --unrolled",
    "verity/protocols/flock/verifier/Flock/AnnotExpand.lean":
        "Annot.sets, expand and reference, an annotation laid out gate by gate: the reference Annot.check is held to "
        "(Flock.Test.Annot, annot-difftest's annot.reference, Properties.Annot.CheckAgrees); no verifier runs it",
    "verity/protocols/flock/verifier/Flock/Extract.lean":
        "open (PRF-72): Q_word v1 owns every computing gate, so its CallGraph lays out one activation of each distinct "
        "Call's Definition, batch members and scan iterations unrolled; no served window names Q_word (they name "
        "template queries, which read the root body only)",
    "verity/protocols/flock/verifier/Flock/Qcall.lean":
        "open (PRF-72): Q_call v1 owns every computing gate, so its Gates lays out one activation of each distinct "
        "Call's Definition, batch members and scan iterations unrolled (its reads deduplicated in linear time, "
        "dedupReads, proved equal to the quadratic loop it replaced: gates_ref); no served window names Q_call",
    "verity/protocols/spot/verifier/lean/Spot/PartitionRef.lean":
        "committedBitsRef, the walk of one activation Partition.committedBits replaced by reading Flock.Qcall.gates' "
        "layout (Gates.readTo): the reference the committed_bits difftest piece holds it to; the verifier never runs it",
    "verity/protocols/flock/verifier/Flock/TemplateQuery.lean":
        "TemplateQuery.unitGatesRef, every unit's gates unit by unit, the walk unitGates replaced: the reference the "
        "unit_gates difftest piece holds it to, and unitGates' answer on ranges evaluate never gives (not contiguous "
        "from unit 0, or a root that isn't a definition)",
    "verity/protocols/flock/verifier/Flock/RecCoins.lean":
        "open (PRF-283 Decision 1): Inner.load's comb, CircuitFold.folded on the inner statement set up from its flat "
        "file, folds every row of the unrolled inner unit, until PR 8 folds a templated inner once per Definition "
        "(InnerSuccinct)",
    "verity/protocols/flock/verifier/Flock/RecVStar.lean":
        "open (PRF-283 Decision 1): circuitFile regenerates V*'s unrolled circuit text (VStar.emitCircuitOf) for "
        "checkFile's byte comparison, until PR 8 folds a templated inner once per Definition (InnerSuccinct)",
    "verity/protocols/flock/verifier/Flock/NetAt.lean":
        "Tmpl.unrollAt, a templated unit's rows at their committed positions: the reference Tmpl.unitNet is proved "
        "equal to (FlockSoundness.TmplFold.unitNet_eq) and Flock.Test.NetAt compares it with; Tmpl.unitFull, its rows "
        "behind Tmpl.unitNet's thunks, built only where a check reads them: open on the templated statement's path, "
        "where verify --zk's KsScope.checkOf reads them (ProvedScope.scopeOkH's portReadsB) before refusing the "
        "statement as a version it doesn't serve (ProvedScope.servedForm; benchmarks/verifier_cost/flock_served)",
    "verity/protocols/flock/verifier/Flock/HmRow.lean":
        "HmOut.unrolled (and HmOut.parseStmt), a templated file's statement circuit, the flat circuit its template "
        "unrolls to (ASK-112 option A), which the soundness statements read; no check runs it: verify sets up on and "
        "checks the committed circuit (HmOut.parseCommit), its template by its description (ProvedScope.scopeOkT), "
        "but KsScope.checkOf's scope check reads its unit's rows (NetAt.lean's entry)",
    "verity/protocols/flock/verifier/Flock/TmplLayout.lean":
        "Tmpl.posMap, every column's committed position: the reference Tmpl.posOf is proved to read (posMap_tree)",
    "verity/protocols/flock/verifier/Flock/TmplFold.lean":
        "Tmpl.foldRef, the fold over the unrolled rows: the reference Tmpl.foldUnit is difftested against "
        "(Flock.Test.TmplFold) and proved equal to (foldUnit_eq_project)",
    "verity/protocols/flock/verifier/Flock/Gate.lean":
        "Prog.gateAt, one gate's location as Flock.Program.Gate: verity-evaluate's `gate` serves it so the difftest holds "
        "layout.Gate to it; no verifier calls it",
    "Security/Proofs/Flock/SpotFlock/Code.lean":
        "open: the two-stage verifier's stmtOk decides the registration (parse, partition, two) over the members' "
        "side-by-side program, every unit of every member laid out, and topoOk and partitionsOk range over its gates; "
        "until two decides the members' descriptors (each member's units, B and R, and filesOk) and stmtOk is compiled "
        "(@[csimp]) as a check that never builds bitsProgs, membersPart or filesReplay",
    "Security/Proofs/Flock/SpotFlock/ScopeFast.lean":
        "open: topoFast, coverFast and partitionsFast, what SpotFlock.code compiles topoOk, partitionsOk and scopeOk as "
        "(@[csimp]), read every gate of the unrolled program once (linear where the checks as stated are quadratic or "
        "cubic), and their equalities restate those checks; replaced as SpotFlock/Code.lean's entry says",
    "Security/Properties/Spot/Members.lean":
        "open: bitsProgs, membersPart and filesReplay build each member's key program (Bits.keyProg, every unit laid "
        "out) and lower it, and its blocks, unit by unit: what SpotFlock.code's parse and partition evaluate; replaced as "
        "SpotFlock/Code.lean's entry says",
    "verity/protocols/spot/verifier/lean/Spot/Bits.lean":
        "open: keyProg lays a member's every unit out (2 + n·B inputs and n·R rows), and its layout (prog, off, offs, "
        "unitGates, owner, owners, insOf, blockOf) walks that program gate by gate, which Properties.Spot.Members reads "
        "for SpotFlock.code; replaced as Security/Proofs/Flock/SpotFlock/Code.lean's entry says",
}

FIX = ("a verifier of record does work proportional to the description, never to the unrolled circuit (AGENTS.md "
       "\"Design rules\"): summarize once per specialized Definition and multiply through batch members and scan "
       "iterations, as NciCheck.Desc does; a gate-by-gate reference a difftest holds the verifier to goes in ALLOWED "
       "with its reason")

#: `/- … -/` and `-- …`, blanked so that line numbers stay
COMMENT = re.compile(r"/-.*?-/|--[^\n]*", re.S)
#: `P.gate g`: a value's gate locator (`Certificate.gate` and other namespaces' `gate` are not it)
LOCATOR = re.compile(r"(?<![\w.'])(?:[a-z]\w*|P)\.gate\b(?![s'])")
_ARG = r"(?:\((?:[^()\n]|\([^()\n]*\))*\.gates\)|(?:\([^()\n]*\))?[\w.\[\]!]*\.gates\b)"
#: a range or loop bounded by a gate count
RANGE = re.compile(rf"\b(?:List|Array)\.range\s*{_ARG}|\[\s*\w*\s*:\s*[^\]\n]*\.gates\s*\]")
#: `let n := … .gates` at the end of a line: a name bound to a gate count, or (`… .gates.toArray`) to the gates
BOUND = re.compile(r"\blet\s+(?:mut\s+)?(\w+)\s*(?::\s*\w+\s*)?:=[^\n]*\.gates(\.toArray)?\s*$", re.M)
#: a range or loop over a node's members: every batch member or scan iteration
MEMBERS = re.compile(r"\b(?:List|Array)\.range\s*\(?[\w.!\[\]()]*\.members\b|\[\s*\w*\s*:\s*[\w.!\[\]]*\.members\s*\]")
#: a name bound to a key program (`Spot.Bits.KeyProg`, a member's units side by side), by a binder or a `let`
KEYPROG = re.compile(r"[({]\s*((?:[\w']+\s+)*[\w']+)\s*:\s*(?:[\w.]*\.)?KeyProg\b|\blet\s+(\w+)\s*:=\s*(?:[\w.]*\.)?keyProg\b")
#: a range over a key program built in place (`List.range (keyProg c n).inputs`)
KEYPROG_RANGE = re.compile(r"\b(?:List|Array)\.range\s*\(\s*(?:[\w.]*\.)?keyProg\b[^\n]*?\)\.(?:size|units|inputs)\b")
#: a member's key program built, every unit laid out (`Spot.Bits.keyProg c n`)
KEYPROG_BUILT = re.compile(r"(?<![\w'.])(?:(?:Spot\.)?Bits\.)?keyProg\s+(?![:=|])\S")
#: a module a Lean file imports from `Security/`'s proofs or properties
IMPORT = re.compile(r"^(?:\w+\s+)*import\s+(?:all\s+)?((?:Proofs|Properties)(?:\.[\w']+)+)\s*$", re.M)
#: a templated unit unrolled: its rows, every column's position, or the fold over its rows
UNROLLED = re.compile(r"(?<![\w'])(?:unrollAt|unitFull|posMap|foldRef)\b(?!')|(?<![\w'])Tmpl\.unroll\b(?![\w'])")
#: the recursive inner statement set up from its flat file, which carries no template
INNER = re.compile(r"\bStmt\.setup\w*\s+(?:\w+\.)*innerTags\b")
#: the lincheck's comb (`CircuitFold.folded`, `Stmt.foldedPartial`)
COMB = re.compile(r"\.(?:folded|foldedPartial)\b(?![\w'])")
#: V*'s circuit text emitted from outside V*'s namespace: the unit laid out and written again
EMIT = re.compile(r"(?<![\w'.])VStar\.emitCircuit(?:Of|Holo)?\b(?![\w'])")
#: a range mapped unit by unit (`(Array.range pop).mapM fun u =>`), and the body it maps: the rest of its line and two more
UNIT_MAP = re.compile(r"\b(?:List|Array)\.range\b[^\n]*?\bfun\s+(\w+)\s*=>([^\n]*(?:\n[^\n]*){0,2})")


def _unit_gates(unit: str) -> re.Pattern:
    """A template query's gates of one unit, `TemplateQuery.gates d rs unit`."""
    return re.compile(rf"(?<![\w'.])(?:(?:Flock\.)?TemplateQuery\.)?gates\s+[\w.]+\s+\w+\s+{re.escape(unit)}\b(?![\w'.])")


def offenses(source: str, test: bool = False) -> list[tuple[int, str]]:
    """(line, what) for every unrolled walk in Lean `source`; a test (`test`) may unroll a templated unit."""
    src = COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), source)
    hits = [(m.start(), "the gate locator") for m in LOCATOR.finditer(src)]
    if not test:
        hits += [(m.start(), "a templated unit unrolled") for m in UNROLLED.finditer(src)]
        if INNER.search(src):
            hits += [(m.start(), "the comb over the flat inner circuit") for m in COMB.finditer(src)]
        hits += [(m.start(), "V*'s unrolled circuit text regenerated") for m in EMIT.finditer(src)]
        hits += [(m.start(), "a template query's units' gates asked one by one (TemplateQuery.unitGates reads them a "
                  "range at a time)") for m in UNIT_MAP.finditer(src) if _unit_gates(m.group(1)).search(m.group(2))]
        hits += [(m.start(), "a member's key program built, every unit laid out") for m in KEYPROG_BUILT.finditer(src)]
    hits += [(m.start(), "a range over a gate count") for m in RANGE.finditer(src)]
    hits += [(m.start(), "a range over a node's members") for m in MEMBERS.finditer(src)]
    for b in BOUND.finditer(src):
        name = b.group(1) + (".size" if b.group(2) else "")
        n = re.escape(name)
        use = re.compile(rf"\b(?:List|Array)\.range\s*\(?\s*{n}\b(?!\.)|\[\s*\w*\s*:\s*{n}\s*\]|"
                         rf"\b(?:mkEmpty|replicate)\s+{n}\b(?!\.)")
        hits += [(m.start(), f"an array or loop of `{name}`, a gate count") for m in use.finditer(src)]
    hits += [(m.start(), "a range over a key program's gates") for m in KEYPROG_RANGE.finditer(src)]
    for name in {x for k in KEYPROG.finditer(src) for x in (k.group(1) or k.group(2)).split()}:
        n = rf"{re.escape(name)}\.(?:size|units|inputs)"
        use = re.compile(rf"\b(?:List|Array)\.range\s*\(?\s*{n}\b|\[\s*\w*\s*:\s*{n}\s*\]|\b(?:mkEmpty|replicate)\s+{n}\b")
        hits += [(m.start(), f"a range over key program `{name}`'s gates or units") for m in use.finditer(src)]
    return sorted({(src.count("\n", 0, at) + 1, what) for at, what in hits})


def _security_code(tracked: list[str]) -> set[str]:
    """`Security/`'s verifier code models: each `Security/Proofs/**/Code.lean`, the `Security` modules it imports, and the
    `Properties` modules those import in turn."""
    have = set(tracked)
    code = {p for p in have if p.startswith("Security/Proofs/") and p.endswith("/Code.lean")}
    seen, todo = set(code), list(code)
    while todo:
        p = todo.pop()
        for mod in IMPORT.findall((ROOT / p).read_text()):
            q = f"Security/{mod.replace('.', '/')}.lean"
            if q in have and q not in seen and (p in code or mod.startswith("Properties.")):
                seen.add(q)
                todo.append(q)
    return seen


def _scanned() -> list[str]:
    tracked = _tracked()
    return sorted({p for p in tracked if p.startswith("verity/core/") and p.endswith(".lean")} | _security_code(tracked))


def test_no_verifier_of_record_walks_the_unrolled_circuit():
    found = {p: offenses((ROOT / p).read_text(), test="/Test/" in p) for p in _scanned()}
    bad = [f"{p}:{line}: {what}" for p, hits in found.items() if p not in ALLOWED for line, what in hits]
    assert not bad, f"{FIX}:\n" + "\n".join(bad)
    stale = [p for p in ALLOWED if not found.get(p)]
    assert not stale, f"ALLOWED entries that match nothing (remove them): {stale}"


def test_the_scan_sees_each_form():
    src = """
    -- P.gate g in a comment is not code
    def a (P : Prog) := (List.range (P.defs[P.root]!).gates).foldlM (fun c g => c.push <$> gateOf P g) init
    def b (P : Prog) (g : Nat) := P.gate g
    def c (defn : Defn) := do
      let n := defn.gates
      for j in [0:n] do pure ()
    def d (defn : Defn) := for j in [0:defn.gates] do pure ()
    def e := Certificate.gate s k delta rtt
    def f (G : Gates) := let G := c.gates; Array.replicate G.nc 0
    def g := (Array.range pop).map fun i => TemplateQuery.gates d rs i
    def h (t : Tmpl) (lays : Array Lay) := Tmpl.unrollAt t lays
    def i (t : Tmpl) := (Flock.Tmpl.posMap t lays).size + (Tmpl.unroll t).size + (Desc.unroll D).size
    def j (t : Tmpl) := Tmpl.unrollOk t lays && Tmpl.unitNet t lays == n && Tmpl.unrolled t
    def k (node : Node) := for m in [0:node.members] do pure ()
    def l (body : Body) := (List.range (body.nodes[k]!).members).map id ++ List.range node.membersOf
    def n (f : Files) := do
      let (_, sts) ← Stmt.setupTables innerTags f.circuit f.pub RecVStar.innerTree 1
      (Setup.ofCircuit sts[0]!).fold.folded a b lw rho ts
    def o (s : VStar.Stmt) := VStar.emitCircuitOf holo put u l s.sub b
    def p (t : Tmpl) := Tmpl.foldUnit t lays sl a lw rho rr
    def q (C : Prog) :=
      let gs := C.gates.toArray
      (List.range gs.size).all fun g => g < gs.size
    def r (kp : Spot.Bits.KeyProg) (u : Nat) := (List.range kp.size).filter (· < u) ++ List.range kp.units
    def s (c : Circuit) (n : Nat) := (List.range (keyProg c n).inputs).map id
    def t (ci : Circuit × Array Nat) := Bits.blockOf (Bits.keyProg ci.1 ci.2.size) κ 0
    """
    assert [line for line, _ in offenses(src)] == [3, 4, 7, 8, 11, 12, 13, 15, 16, 19, 20, 24, 25, 26, 26, 27]
    assert [line for line, _ in offenses(src, test=True)] == [3, 4, 7, 8, 15, 16, 24, 25, 26]
    plain = """
    def lincheck (c : CircuitFold) := c.folded alpha beta lw rho ts
    def stage := emitCircuitOf holo put u l sub b
    def one (u : Nat) := TemplateQuery.gates d rs u
    def laid := (Array.range pop).map fun i => match TemplateQuery.locate rs i, gatesOf i with
      | some l, some g => (l, g)
      | _, _ => none
    def other := (Array.range pop).mapM fun u => gates d rs v
    def sized (a : Array Nat) := List.range a.size
    def counted (kp : KeyProg) := kp.size + kp.units
    def placed (c : Circuit) := Placed.keyProg c n
    def held (C : Prog) := let gs := C.gates.toArray; gs.size
    """
    assert offenses(plain) == []
