module

public import Definitions.Spot.Dataflow
public import Mathlib.Data.Fintype.Option

/-! Long values: what the verifier can learn of a slot is a value of the slot's width, or ⊥ when the slot has no opening of that width. -/

@[expose] public section

namespace Population

open Finset

def fill {S V : Type} (d : S → V) (σ : S → Option V) : S → V := fun s => (σ s).getD (d s)

namespace Dataflow

variable {U S : Type} (P : Dataflow U S)

/-- A unit that reads or writes ⊥ is incorrect. -/
def OptCorrect {V : Type} (f : P.Fns V) (σ : S → Option V) (u : U) : Prop :=
  ∃ x : P.reads u → V, (∀ r : P.reads u, σ r = some (x r)) ∧ ∀ s : P.writes u, σ s = some (f u x s)

open Classical in
noncomputable def optIncorrect [Fintype U] {V : Type} (f : P.Fns V) (σ : S → Option V) : Finset U :=
  univ.filter fun u => ¬ P.OptCorrect f σ u

def OptHolds {V : Type} (a : S → V) (X : Finset S) (σ : S → Option V) : Prop :=
  ∀ s ∈ P.input, s ∉ X → σ s = some (a s)

open Classical in
noncomputable def optEgress [Fintype U] [Fintype S] {V : Type} [Fintype V] (f : P.Fns V) (a : S → V)
    (X O : Finset S) (B : Finset U) : Finset (O → Option V) :=
  (univ.filter fun σ : S → Option V => P.OptHolds a X σ ∧ P.optIncorrect f σ ⊆ B).image fun σ o => σ o

open Classical in
noncomputable def optEgressAt [Fintype U] [Fintype S] {V : Type} [Fintype V] (f : P.Fns V) (a : S → V)
    (X O : Finset S) (m : ℕ) : Finset (O → Option V) :=
  (univ.filter fun σ : S → Option V => P.OptHolds a X σ ∧ (P.optIncorrect f σ).card ≤ m).image fun σ o => σ o

def OptSkipsLucky {V : Type} (f : P.Fns V) (honest : S → V) (σ : S → Option V) (K L : Finset U) : Prop :=
  ∀ u, u ∉ K → (∀ r ∈ P.reads u, σ r = some (honest r)) → P.OptCorrect f σ u → u ∈ L

end Dataflow

/-- A string of any length may climb, because the inner hash takes any length; `hashed` holds the prover's long values
among the run's hashed strings. -/
structure Openings (S V : Type) where
  width : S → ℕ
  read : S → List UInt8 → V
  climbs : S → Set (List UInt8)
  hashed : Set (List UInt8)

namespace Openings

variable {S V : Type} (O : Openings S V)

/-- The verifier tests the width before it reads `x`. -/
def Accepts (s : S) (x : List UInt8) : Prop := x.length = O.width s ∧ x ∈ O.climbs s

open Classical in
noncomputable def view (s : S) : Option V :=
  if h : ∃ x, O.Accepts s x then some (O.read s h.choose) else none

end Openings

def Collides {D : Type} (H : List UInt8 → D) (T : Set (List UInt8)) : Prop :=
  ∃ a ∈ T, ∃ b ∈ T, a ≠ b ∧ H a = H b

/-- A named assumption: hm96's binding with frame-v3's length-prefixed nodes on the path; the deployment's `H` is SHA-512
with its whole output. -/
def CollisionStep {S V D : Type} (H : List UInt8 → D) (O : Openings S V) : Prop :=
  ∀ s, ∀ x ∈ O.climbs s, ∀ y ∈ O.climbs s, x ≠ y → Collides H O.hashed

/-- In the clear or inside the zero-knowledge proof that opens it, the width test comes first. -/
def Dataflow.Opened {U S V : Type} (P : Dataflow U S) (O : Openings S V) (f : P.Fns V) (u : U) : Prop :=
  ∃ x : P.reads u → V, (∀ r : P.reads u, ∃ y, O.Accepts r y ∧ O.read r y = x r) ∧
    ∀ s : P.writes u, ∃ y, O.Accepts s y ∧ O.read s y = f u x s

end Population
