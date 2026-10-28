import NetworkCertifier.WindowCharge
import Lean.Data.Json

/-!
# The window charge's difftest: advice v1's whole window against `window_charge.reference_window`, on random instances

`scripts/window_vectors.py` generates instances and runs
`verity.ml.protocols.certifier.window_charge.reference_window` on them. `NetworkCertifier/WindowDifftestMain.lean`
reads both files and fails unless `evaluate`, which computes the same results from the program's definitions
(`NetworkCertifier.WindowCharge`), gives every one of them. The Lean audit runs it (`runs` in `lean-audit.json`):
compiled evaluation, a test and not a proof. `Proofs.NetworkCertifier.Code` (`CodeChargesWhole`) proves `charges` equal
to the reference's (`Security/Definitions/NetworkCertifier/WindowCharge.lean`), which `EncardChargedLeWhole` and the
whole window's leak bounds read.

An instance is a window's slot counts `N`, `H` and `O`, a book (each of `window_charge.SYMBOLS`' tables, its frequencies
at the bins a window reaches and its total, and the contexts' `priors`), a carry, and windows, each its `N` decision
slots (`en`, `bit`), `H` hit slots (`hen`, `short`) and `O` order slots (`oen`, `p`, `r`), as
`ChargeCount_v2{N, H, O}`'s witness holds them. A window's advice is each field's values at the slots that hold one, in
order (`advice`).

`evaluate` gives, from the definitions themselves:
* `steps`: from the instance's carry, each window's `window`, its charge and the carry it passes on (`[charge, counts,
  run, pos, rank]`), up to and including the first window it refuses (`null`): Python's `reference_window` window by
  window, each `ok` 0 a `null`;
* `charges`: `charges` of the windows' advice, from the book's first carry, or `null` when a window refuses.
-/

namespace NetworkCertifier.WindowCharge.WindowDifftest

open Lean

/-- A window's slots as read. -/
structure Slots where
  en : List Bool
  bit : List Bool
  hen : List Bool
  short : List Nat
  oen : List Bool
  p : List Nat
  r : List Nat

/-- An instance as read. -/
structure Raw where
  N : Nat
  H : Nat
  O : Nat
  tables : List Table
  book : Book
  carry : Carry
  windows : List Slots

def pairOfJson (j : Json) : Except String (Nat × Nat) := do
  let #[a, b] ← fromJson? (α := Array Nat) j | throw s!"not [a, b]: {j.compress}"
  pure (a, b)

def pairsOfJson (j : Json) : Except String (List (Nat × Nat)) := do
  (← fromJson? (α := List Json) j).mapM pairOfJson

/-- A slot's `0` or `1`. -/
def bitOfJson (j : Json) : Except String Bool := do
  match ← fromJson? (α := Nat) j with
  | 0 => pure false
  | 1 => pure true
  | n => throw s!"not 0 or 1: {n}"

def bitsOfJson (j : Json) (k : String) : Except String (List Bool) := do
  (← fromJson? (α := List Json) (← j.getObjVal? k)).mapM bitOfJson

/-- A table, `[freqs, total]`. -/
def tableOfJson (j : Json) : Except String Table := do
  let #[f, t] ← fromJson? (α := Array Json) j | throw s!"not [freqs, total]: {j.compress}"
  pure ⟨← fromJson? f, ← fromJson? t⟩

def rawOfJson (j : Json) : Except String Raw := do
  let bk ← j.getObjVal? "book"
  let cy ← j.getObjVal? "carry"
  let ts ← (← fromJson? (α := List Json) (← bk.getObjVal? "tables")).mapM tableOfJson
  let #[a, nh, h, rs, no, g, i, l] := ts.toArray | throw s!"not 8 tables: {ts.length}"
  let ws ← fromJson? (α := List Json) (← j.getObjVal? "windows")
  pure {
    N := ← j.getObjValAs? Nat "N"
    H := ← j.getObjValAs? Nat "H"
    O := ← j.getObjValAs? Nat "O"
    tables := ts
    book := ⟨a, nh, h, rs, no, g, i, l, ← pairsOfJson (← bk.getObjVal? "priors")⟩
    carry := ⟨⟨← pairsOfJson (← cy.getObjVal? "counts"), ← cy.getObjValAs? Nat "run"⟩, ← cy.getObjValAs? Nat "pos",
      ← cy.getObjValAs? Nat "rank"⟩
    windows := ← ws.mapM fun w => do
      pure ⟨← bitsOfJson w "en", ← bitsOfJson w "bit", ← bitsOfJson w "hen", ← w.getObjValAs? (List Nat) "short",
        ← bitsOfJson w "oen", ← w.getObjValAs? (List Nat) "p", ← w.getObjValAs? (List Nat) "r"⟩ }

/-- The values at the slots that hold one, in order. -/
def held {α : Type} (en : List Bool) (v : List α) : List α :=
  (en.zip v).filterMap fun (e, x) => if e then some x else none

/-- A window's advice: each field's values at the slots that hold one. -/
def advice (s : Slots) : Window := ⟨held s.en s.bit, held s.hen s.short, held s.oen (s.p.zip s.r)⟩

/-- The bins of each table a window of `N`, `H` and `O` slots reaches (`window_charge._bins`): bins `0 … bitlen v` of
the largest value `v` it codes, 0 for n_resumes and n_lag. -/
def bins (N H O : Nat) : List Nat :=
  [N, H, HIT_CAP, 0, O, GAP_CAP, ZIG_CAP, 0].map (AdviceV1.bitlen · + 1)

def pairsJson (l : List (Nat × Nat)) : Json := toJson (l.map fun p => [p.1, p.2])

/-- `window` from `cy` on each window in turn, up to and including the first it refuses. -/
def steps (bk : Book) (N H O : Nat) : Carry → List Window → List Json
  | _, [] => []
  | cy, x :: ws => match window bk N H O cy x with
    | none => [.null]
    | some (L, cy') =>
      .arr #[toJson L, pairsJson cy'.dec.counts, toJson cy'.dec.run, toJson cy'.pos, toJson cy'.rank] ::
        steps bk N H O cy' ws

/-- What the program's definitions give on an instance, in the shape of `reference_window`'s results. -/
def evaluate (r : Raw) : Except String Json := do
  unless r.windows.all fun s => s.en.length = r.N ∧ s.bit.length = r.N ∧ s.hen.length = r.H ∧ s.short.length = r.H ∧
      s.oen.length = r.O ∧ s.p.length = r.O ∧ s.r.length = r.O do
    throw "a window has N, H and O slots"
  unless r.tables.map (·.freqs.length) = bins r.N r.H r.O do
    throw "the book's tables are at the bins a window reaches"
  let ws := r.windows.map advice
  pure <| Json.mkObj [
    ("steps", .arr (steps r.book r.N r.H r.O r.carry ws).toArray),
    ("charges", match charges r.book r.N r.H r.O ws with
      | none => .null
      | some ls => toJson ls)]

/-- The results' fields, in the order they are compared. -/
def fields : List String := ["steps", "charges"]

/-- How the Lean side's results for instance line `i` differ from Python's, one line per field that differs. -/
def differences (i : Nat) (inst python : String) : List String :=
  match Json.parse inst >>= rawOfJson >>= evaluate, Json.parse python with
  | .error e, _ => [s!"instance {i}: the Lean side rejects it: {e}"]
  | _, .error e => [s!"instance {i}: Python's results don't parse: {e}"]
  | .ok lean, .ok py => fields.filterMap fun k =>
    let a := lean.getObjValD k
    let b := py.getObjValD k
    if a == b then none else some s!"instance {i}: {k}: Lean {a.compress}, Python {b.compress}"

/-- `lake env lean --run NetworkCertifier/WindowDifftestMain.lean INSTANCES PYTHON N`: every instance line of `INSTANCES`
evaluated, each compared with the same line of `PYTHON`. Fails unless both files have `N` lines and none differs. -/
def main (args : List String) : IO UInt32 := do
  let [insts, py, n] := args | IO.eprintln "usage: NetworkCertifier/WindowDifftestMain.lean INSTANCES PYTHON N"; return 2
  let lines (p : String) : IO (Array String) := do
    return ((← IO.FS.readFile p).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  let a ← lines insts
  let b ← lines py
  if n.toNat? != some a.size ∨ a.size != b.size then
    IO.eprintln s!"window difftest: {a.size} instances and {b.size} results, not {n}"
    return 1
  let tasks := (a.zip b).toList.zipIdx.map fun ((x, y), i) => Task.spawn fun _ => differences i x y
  let found := tasks.flatMap Task.get
  for d in found.take 50 do IO.eprintln d
  IO.println s!"window difftest: {a.size} instances compared, {found.length} differences"
  return if found.isEmpty then 0 else 1

end NetworkCertifier.WindowCharge.WindowDifftest
