import Std.Data.HashMap
import PouwCheck.Protocol

/-!
# The difftest against `core_check.py`

Each line of the cases file is a `serve` case `scripts/difftest_vectors.py` generated; the Lean answer's canonical line
must be the Python's byte for byte. The tally counts answers by verdict and by the reason's rule (its text before the
first colon).
-/

namespace PouwCheck.Difftest

def lines (path : System.FilePath) : IO (Array String) := do
  pure ((← IO.FS.lines path).filter (· ≠ ""))

def rule (a : Core.Answer) : String :=
  match a.reason with
  | some r => s!"{a.verdict.name} {(r.splitOn ":").headD r}"
  | none => a.verdict.name

/-- `pouw-check-difftest CASES PYTHON N`: every one of the N cases gives Python's answer. -/
def main (args : List String) : IO UInt32 := do
  let [cp, pp, n] := args | IO.eprintln "usage: pouw-check-difftest CASES PYTHON N"; return 2
  let cases ← lines cp
  let want ← lines pp
  let some n := n.toNat? | IO.eprintln s!"not a count: {n}"; return 2
  unless cases.size = n ∧ want.size = n do
    IO.eprintln s!"expected {n} cases and answers, read {cases.size} and {want.size}"; return 1
  let mut bad := 0
  let mut tally : Std.HashMap String Nat := {}
  for i in [0:n] do
    let a := protocol.case cases[i]!
    let r := rule a
    tally := tally.insert r (tally.getD r 0 + 1)
    match a.canonical with
    | .ok got =>
      if got ≠ want[i]! then
        bad := bad + 1
        if bad ≤ 10 then IO.eprintln s!"case {i}: Lean {got}\n  Python {want[i]!}"
    | .error e => bad := bad + 1; if bad ≤ 10 then IO.eprintln s!"case {i}: the answer is not canonical JSON: {e}"
  let counts := ", ".intercalate ((tally.toList.toArray.qsort (·.1 < ·.1)).toList.map fun (k, v) => s!"{k} {v}")
  if bad = 0 then
    IO.println s!"pouw-check difftest: all {n} cases agree ({counts})"; return 0
  IO.eprintln s!"pouw-check difftest: {bad} of {n} cases differ"; return 1

end PouwCheck.Difftest
