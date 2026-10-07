import Definitions.Pouw.Fp8Atom.Fp32

/-!
# The bulk conformance test (compiled evaluation: a test, not a proof)

The Lean audit runs it (`runs` in `lean-audit.json`): `verity/protocols/accounting/work/pouw/lean/scripts/fp8atom_vectors.py` writes `conformance/steps.txt.gz`
and `conformance/adds.txt.gz`, the audit checks each decompressed file against the SHA-256 the record holds, and runs
`lake env lean --run Proofs/Pouw/Bulk.lean <steps.txt> <n> <adds.txt> <n>` in its sandbox.

On every line it evaluates the predicates of the kernel pins: `Vec.holds hopperStep` for the steps and `AddVec.holds`
for the adds. It fails unless every line parses and holds and the line counts are the expected ones. The kernel does
not check this evaluation. The kernel-pinned subsets are `Pouw.Fp8Atom.Proofs.conformance` and `addConformance`. The
file is not part of the `Pouw` library and is exempt in `lean-audit.json`.
-/

open Pouw.Fp8Atom

/-- A hexadecimal numeral, or `none`. -/
def hexNat? (s : String) : Option Nat :=
  if s.isEmpty then none else
    s.foldl (fun acc c => acc.bind fun n =>
      if '0' ≤ c ∧ c ≤ '9' then some (16 * n + (c.toNat - '0'.toNat))
      else if 'a' ≤ c ∧ c ≤ 'f' then some (16 * n + (c.toNat - 'a'.toNat + 10))
      else if 'A' ≤ c ∧ c ≤ 'F' then some (16 * n + (c.toNat - 'A'.toNat + 10))
      else none) (some 0)

/-- A step line, `acc a b d family`, holds for `hopperStep`. -/
def stepOk : List String → Bool
  | [acc, a, b, d, _] =>
    match hexNat? acc, hexNat? a, hexNat? b, hexNat? d with
    | some acc, some a, some b, some d => Vec.holds hopperStep ⟨acc, a, b, d⟩
    | _, _, _, _ => false
  | _ => false

/-- An add line, `a b c family`, holds for `rnAdd`. -/
def addOk : List String → Bool
  | [a, b, c, _] =>
    match hexNat? a, hexNat? b, hexNat? c with
    | some a, some b, some c => AddVec.holds ⟨a, b, c⟩
    | _, _, _ => false
  | _ => false

/-- The number of lines of `path`, and of those that fail `ok`; prints the first ten failures. -/
def checkLines (path : System.FilePath) (ok : List String → Bool) : IO (Nat × Nat) := do
  let lines ← IO.FS.lines path
  let mut bad := 0
  for l in lines do
    if !ok (l.splitOn " ") then
      bad := bad + 1
      if bad ≤ 10 then IO.eprintln s!"FAIL {path}: {l}"
  return (lines.size, bad)

def main (args : List String) : IO UInt32 := do
  match args with
  | [steps, nSteps, adds, nAdds] =>
    let (ns, bs) ← checkLines steps stepOk
    let (na, ba) ← checkLines adds addOk
    IO.println s!"bulk: {ns - bs} of {ns} steps hold for hopperStep, {na - ba} of {na} adds for rnAdd"
    if bs == 0 && ba == 0 && some ns == nSteps.toNat? && some na == nAdds.toNat? then
      return 0
    IO.eprintln s!"bulk: FAILED; expected {nSteps} steps and {nAdds} adds, all holding"
    return 1
  | _ =>
    IO.eprintln "usage: lake env lean --run PouwBulk.lean <steps.txt> <n steps> <adds.txt> <n adds>"
    return 2
