"""A verifier of record costs O(circuit description), never O(unrolled circuit) (AGENTS.md "Design rules", Daniel, 18 Oct).

A scan over the Lean under `verity/core/` (the verifiers of record and the code they run) refuses the forms an unrolled
walk takes there: locating gates one by one through the canonical layout (`P.gate g`, `Flock.Program.Prog.gate`), a
range or loop bounded by a Definition's gate count (`List.range (P.defs[P.root]!).gates`, `[0:defn.gates]`), and an
array or loop bounded by a name bound to one (`let n := defn.gates` … `Array.mkEmpty n`, `for j in [0:n]`). The fix is
in the scan's failure message: summarize once per specialized Definition and multiply through batch members and scan
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
        "open (PRF-72): Q_word v1's CallGraph lays out one activation of each distinct Call's Definition, its batch members "
        "and scan iterations unrolled",
    "verity/core/service/flock/Flock/Qcall.lean":
        "open (PRF-72): Q_call v1's Gates lays out one activation of each distinct Call's Definition, its batch members "
        "and scan iterations unrolled",
    "verity/core/service/sampled_proofs/SampledProofs/Partition.lean":
        "open (PRF-72): committedBits lays out one activation of a Q_call unit's Definition, as Flock.Qcall.gates does",
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


def offenses(source: str) -> list[tuple[int, str]]:
    """(line, what) for every unrolled walk in Lean `source`."""
    src = COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), source)
    hits = [(m.start(), "the gate locator") for m in LOCATOR.finditer(src)]
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
    found = {p: offenses((ROOT / p).read_text()) for p in _scanned()}
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
    """
    assert [line for line, _ in offenses(src)] == [3, 4, 7, 8]
