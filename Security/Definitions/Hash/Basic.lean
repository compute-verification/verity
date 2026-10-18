import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fintype.Prod
import Mathlib.Basic.Real.Basic

/-!
# Messages, digests, collisions and uniform probability (trusted)

The vocabulary the hash area's statements share. A message is a byte string, as the deployed hashes read it; a digest
is 64 bytes, SHA-512's whole output (`Digest`), so a node's input splits back into its children's digests. A collision
of `H` is two different inputs with one output; a collision *among* a list of inputs (`CollisionIn`) is one whose two
inputs both appear in the list, which is how a statement ties a break to the inputs a run itself hashed (ruling of
8 Oct: every `∃` in a break is bounded by something the run carries).

Probability is a uniform average over a finite coin space (`Pr`, `Ex`), as in `Definitions/Claims.lean`; the area
imports only Mathlib and itself, so it states its own.
-/

namespace Hash

/-- A message: whole bytes, as the deployed SHA-512 reads them. -/
abbrev Msg := List UInt8

/-- A digest: 64 bytes, SHA-512's whole output. -/
abbrev Digest := Fin 64 → UInt8

/-- A collision of `H`: two different inputs with one output. -/
def Collision {α β : Type} (H : α → β) (u v : α) : Prop := u ≠ v ∧ H u = H v

/-- A collision of `H` between two of the inputs `xs`: the break a statement ties to a run, `xs` being what the run
hashed. -/
def CollisionIn {β : Type} (H : Msg → β) (xs : List Msg) : Prop := ∃ u ∈ xs, ∃ v ∈ xs, Collision H u v

open Classical in
/-- `Pr[p]` for a uniform coin. -/
noncomputable def Pr {α : Type} [Fintype α] (p : α → Prop) : ℝ :=
  ((Finset.univ.filter p).card : ℝ) / Fintype.card α

/-- The expectation of `f` under a uniform coin. -/
noncomputable def Ex {α : Type} [Fintype α] (f : α → ℝ) : ℝ := (∑ a, f a) / Fintype.card α

/-- The first collision among the `(input, output)` pairs `l`: two different inputs with one output. -/
def findColl {β : Type} [DecidableEq β] : List (Msg × β) → Option (Msg × Msg)
  | [] => none
  | (u, a) :: l =>
    match l.find? (fun p => decide (p.1 ≠ u) && decide (p.2 = a)) with
    | some p => some (u, p.1)
    | none => findColl l

end Hash
