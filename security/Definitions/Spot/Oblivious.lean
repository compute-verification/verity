module

public import Definitions.Spot.Dataflow

/-! Obliviousness: the properties of a program trace's read sets that decide whether it is a population with public wiring, and the padding that makes it one. -/

@[expose] public section

namespace Population

open Finset

namespace DynPopulation

variable {U S V : Type} (D : DynPopulation U S V)

def Oblivious : Prop := ∀ u σ τ, D.reads u σ = D.reads u τ

/-- With `P` the public inputs, this is obliviousness relative to them. -/
def AddressedBy (P : Finset S) : Prop := ∀ u σ τ, (∀ s ∈ P, σ s = τ s) → D.reads u σ = D.reads u τ

/-- It says nothing about runs with an incorrect proof block. -/
def HonestlyOblivious : Prop :=
  ∀ u σ τ, (∀ v, D.Correct σ v) → (∀ v, D.Correct τ v) → D.reads u σ = D.reads u τ

/-- An address a unit uses is a slot it reads: a machine step that loads through a pointer reads the pointer. -/
def AddressClosed : Prop := ∀ u σ τ, (∀ s ∈ D.reads u σ, σ s = τ s) → D.reads u σ = D.reads u τ

open Classical in
/-- `padOver Set.univ` is a padded static circuit's static superset; `padOver {h}` freezes the read sets at `h`. -/
noncomputable def padOver [Fintype S] (C : Set (S → V)) : Dataflow U S where
  input := D.input
  reads u := univ.filter fun s => ∃ σ ∈ C, s ∈ D.reads u σ
  writes := D.writes

open Classical in
noncomputable def padFns [Fintype S] [Inhabited V] (C : Set (S → V)) : (D.padOver C).Fns V :=
  fun u r s => D.f u (fun t => if h : t ∈ (D.padOver C).reads u then r ⟨t, h⟩ else default) s.1

def downAt (σ : S → V) (B : Finset U) : Set U := {v | ∃ u ∈ B, Relation.ReflTransGen (D.Dep σ) u v}

def SkipsLucky (honest σ : S → V) (K L : Finset U) : Prop :=
  ∀ u, u ∉ K → (∀ r ∈ D.reads u honest, σ r = honest r) → D.Correct σ u → u ∈ L

open Classical in
/-- With `C` the honest runs it is a property of the program alone, and it can be smaller than the padding's cone, whose
paths may mix different runs' read sets. -/
noncomputable def downOver [Fintype U] (C : Set (S → V)) (B : Finset U) : Finset U :=
  univ.filter fun v => ∃ h ∈ C, v ∈ D.downAt h B

open Classical in
noncomputable def coneEgressOver [Fintype U] (C : Set (S → V)) (O : Finset S) (B : Finset U) : Finset S :=
  O ∩ D.writesOf (D.downOver C B)

end DynPopulation

end Population
