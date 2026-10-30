import NetworkCertifier.AdviceV1
import Lean.Data.Json

/-!
# The charge's difftest: advice v1's charge against `arith_encode.reference_charge`, on random instances

`scripts/charge_vectors.py` generates instances and runs `verity.core.protocols.certifier.arith_encode.reference_charge`
on them. `NetworkCertifier/ChargeDifftestMain.lean` reads both files and fails unless `evaluate`, which computes the same
results from the program's definitions (`NetworkCertifier.AdviceV1`), gives every one of them. The Lean audit runs it
(`runs` in `lean-audit.json`): compiled evaluation, a test and not a proof. `Proofs.NetworkCertifier.Code`
(`CodeChargesV1`) proves `charges` equal to the reference's (`Security/Definitions/NetworkCertifier/AdviceV1.lean`).

An instance is a window size `N`, a book (`arrived`, n_arrived's bin frequencies at bins `0 … bitlen N`, its `total`,
each zero count's `(freq, total)`, and the contexts' `priors`), a first carry, and windows, each `N` slots that ask or
not (`en`) with their decisions (`bit`), as `ChargeCount_v1{N}`'s witness holds them. A window's decision stream is the
decisions of the slots that ask, in order.

`evaluate` gives, from the definitions themselves:
* `steps`: from the instance's carry, each window's `window`, its charge and the carry it passes on (`[charge, counts,
  run]`), up to and including the first window it refuses (`null`): Python's `reference_charge` window by window, each
  `ok` 0 a `null`;
* `charges`: `charges` of the decision streams, from the book's priors and run 0, or `null` when a window refuses.
-/

namespace NetworkCertifier.AdviceV1.ChargeDifftest

open Lean

/-- An instance as read. -/
structure Raw where
  N : Nat
  book : Book
  carry : Carry
  windows : List (List Bool × List Bool)

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

def rawOfJson (j : Json) : Except String Raw := do
  let bk ← j.getObjVal? "book"
  let cy ← j.getObjVal? "carry"
  let ws ← fromJson? (α := List Json) (← j.getObjVal? "windows")
  pure {
    N := ← j.getObjValAs? Nat "N"
    book := ⟨← bk.getObjValAs? (List Nat) "arrived", ← bk.getObjValAs? Nat "total", ← pairsOfJson (← bk.getObjVal? "zeros"),
      ← pairsOfJson (← bk.getObjVal? "priors")⟩
    carry := ⟨← pairsOfJson (← cy.getObjVal? "counts"), ← cy.getObjValAs? Nat "run"⟩
    windows := ← ws.mapM fun w => do
      pure ((← (← fromJson? (α := List Json) (← w.getObjVal? "en")).mapM bitOfJson),
        (← (← fromJson? (α := List Json) (← w.getObjVal? "bit")).mapM bitOfJson)) }

/-- A window's decision stream: the decisions of the slots that ask, in order. -/
def stream (w : List Bool × List Bool) : List Bool :=
  (w.1.zip w.2).filterMap fun (e, b) => if e then some b else none

def pairsJson (l : List (Nat × Nat)) : Json := toJson (l.map fun p => [p.1, p.2])

/-- `window` from `cy` on each window in turn, up to and including the first it refuses. -/
def steps (bk : Book) (N : Nat) : Carry → List (List Bool) → List Json
  | _, [] => []
  | cy, ds :: ws => match window bk N cy ds with
    | none => [.null]
    | some (L, cy') => .arr #[toJson L, pairsJson cy'.counts, toJson cy'.run] :: steps bk N cy' ws

/-- What the program's definitions give on an instance, in the shape of `reference_charge`'s results. -/
def evaluate (r : Raw) : Except String Json := do
  unless r.windows.all fun w => w.1.length = r.N ∧ w.2.length = r.N do
    throw "a window has N slots"
  unless r.book.arrived.length = bitlen r.N + 1 do
    throw "the book's arrived bins are 0 … bitlen N"
  let ds := r.windows.map stream
  pure <| Json.mkObj [
    ("steps", .arr (steps r.book r.N r.carry ds).toArray),
    ("charges", match charges r.book r.N ds with
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

/-- `lake env lean --run NetworkCertifier/ChargeDifftestMain.lean INSTANCES PYTHON N`: every instance line of `INSTANCES`
evaluated, each compared with the same line of `PYTHON`. Fails unless both files have `N` lines and none differs. -/
def main (args : List String) : IO UInt32 := do
  let [insts, py, n] := args | IO.eprintln "usage: NetworkCertifier/ChargeDifftestMain.lean INSTANCES PYTHON N"; return 2
  let lines (p : String) : IO (Array String) := do
    return ((← IO.FS.readFile p).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  let a ← lines insts
  let b ← lines py
  if n.toNat? != some a.size ∨ a.size != b.size then
    IO.eprintln s!"charge difftest: {a.size} instances and {b.size} results, not {n}"
    return 1
  let tasks := (a.zip b).toList.zipIdx.map fun ((x, y), i) => Task.spawn fun _ => differences i x y
  let found := tasks.flatMap Task.get
  for d in found.take 50 do IO.eprintln d
  IO.println s!"charge difftest: {a.size} instances compared, {found.length} differences"
  return if found.isEmpty then 0 else 1

end NetworkCertifier.AdviceV1.ChargeDifftest
