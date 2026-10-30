"""A verifier of record costs O(circuit description), never O(unrolled circuit) (AGENTS.md "Design rules").

A scan over the Lean under `verity/core/` (the verifiers of record and the code they run) refuses the forms an unrolled
walk takes there: locating gates one by one through the canonical layout (`P.gate g`, `Flock.Program.Prog.gate`), a
range or loop bounded by a Definition's gate count (`List.range (P.defs[P.root]!).gates`, `[0:defn.gates]`), and an
array or loop bounded by a name bound to one (`let n := defn.gates` … `Array.mkEmpty n`, `for j in [0:n]`), and outside
the tests a templated unit unrolled (`Tmpl.unroll`, `Tmpl.unrollAt`, `Tmpl.unitFull`, `Tmpl.posMap`, `Tmpl.foldRef`):
a templated unit is read from its description (`Tmpl.unrollOk`, `Tmpl.unitNet`, `Tmpl.foldUnit`). The fix is in the
scan's failure message: summarize once per specialized Definition and multiply through batch members and scan
iterations, as `NciCheck.Desc` does (its docstring), so a Definition's summary serves every activation of it.

What may stay is listed in ``ALLOWED``, one line of reason each: the gate-by-gate references a description-sized
verifier is difftested against, and the walks still open. An entry that no longer matches anything fails too.
"""

from __future__ import annotations

import re

from test_repository import ROOT, _tracked

ALLOWED: dict[str, str] = {
    "verity/core/protocols/network/nci/lean/NciCheck/Check.lean":
        "NciCheck.check, the gate-by-gate reference NciCheck.Desc is difftested against (runs NciCheck.DescDifftestMain); "
        "`nci-check` runs it only with --unrolled",
    "verity/core/service/flock/Flock/Extract.lean":
        "open (PRF-72): Q_word v1 owns every computing gate, so its CallGraph lays out one activation of each distinct "
        "Call's Definition, batch members and scan iterations unrolled; no served window names Q_word (they name "
        "template queries, which read the root body only)",
    "verity/core/service/flock/Flock/Qcall.lean":
        "open (PRF-72): Q_call v1 owns every computing gate, so its Gates lays out one activation of each distinct "
        "Call's Definition, batch members and scan iterations unrolled (its reads deduplicated in linear time, "
        "dedupReads, proved equal to the quadratic loop it replaced: gates_ref); no served window names Q_call",
    "verity/core/service/sampled_proofs/SampledProofs/PartitionRef.lean":
        "committedBitsRef, the walk of one activation Partition.committedBits replaced by reading Flock.Qcall.gates' "
        "layout (Gates.readTo): the reference the committed_bits difftest piece holds it to; the verifier never runs it",
    "verity/core/service/flock/Flock/NetAt.lean":
        "Tmpl.unrollAt, a templated unit's rows at their committed positions: the reference Tmpl.unitNet is proved "
        "equal to (FlockSoundness.TmplFold.unitNet_eq) and Flock.Test.NetAt compares it with; Tmpl.unitFull, its rows "
        "behind Tmpl.unitNet's thunks, built only where a check reads them, which none on the templated statement's "
        "path does (benchmarks/verifier_cost/flock_tmpl stmt)",
    "verity/core/service/flock/Flock/HmRow.lean":
        "HmOut.unrolled (and HmOut.parseStmt), a templated file's statement circuit, the flat circuit its template "
        "unrolls to (ASK-112 option A), which the soundness statements read; no check runs it: verify sets up on and "
        "checks the committed circuit (HmOut.parseCommit), its template by its description (ProvedScope.scopeOkT)",
    "verity/core/service/flock/Flock/TmplLayout.lean":
        "Tmpl.posMap, every column's committed position: the reference Tmpl.posOf is proved to read (posMap_tree)",
    "verity/core/service/flock/Flock/TmplFold.lean":
        "Tmpl.foldRef, the fold over the unrolled rows: the reference Tmpl.foldUnit is difftested against "
        "(Flock.Test.TmplFold) and proved equal to (foldUnit_eq_project)",
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
#: `let n := … .gates` at the end of a line: a name bound to a gate count
BOUND = re.compile(r"\blet\s+(?:mut\s+)?(\w+)\s*(?::\s*\w+\s*)?:=[^\n]*\.gates\s*$", re.M)
#: a templated unit unrolled: its rows, every column's position, or the fold over its rows
UNROLLED = re.compile(r"(?<![\w'])(?:unrollAt|unitFull|posMap|foldRef)\b(?!')|(?<![\w'])Tmpl\.unroll\b(?![\w'])")


def offenses(source: str, test: bool = False) -> list[tuple[int, str]]:
    """(line, what) for every unrolled walk in Lean `source`; a test (`test`) may unroll a templated unit."""
    src = COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), source)
    hits = [(m.start(), "the gate locator") for m in LOCATOR.finditer(src)]
    if not test:
        hits += [(m.start(), "a templated unit unrolled") for m in UNROLLED.finditer(src)]
    hits += [(m.start(), "a range over a gate count") for m in RANGE.finditer(src)]
    for b in BOUND.finditer(src):
        n = re.escape(b.group(1))
        use = re.compile(rf"\b(?:List|Array)\.range\s*\(?\s*{n}\b(?!\.)|\[\s*\w*\s*:\s*{n}\s*\]|"
                         rf"\b(?:mkEmpty|replicate)\s+{n}\b(?!\.)")
        hits += [(m.start(), f"an array or loop of `{b.group(1)}`, a gate count") for m in use.finditer(src)]
    return sorted({(src.count("\n", 0, at) + 1, what) for at, what in hits})


def _scanned() -> list[str]:
    return sorted(p for p in _tracked() if p.startswith("verity/core/") and p.endswith(".lean"))


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
    """
    assert [line for line, _ in offenses(src)] == [3, 4, 7, 8, 12, 13]
    assert [line for line, _ in offenses(src, test=True)] == [3, 4, 7, 8]
