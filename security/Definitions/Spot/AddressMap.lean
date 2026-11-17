module

public import Definitions.Spot.Oblivious

/-! Address maps, a program's inputs, write sets and value-dependent read sets, and the static wirings that express them. -/

@[expose] public section

namespace Population

open Finset

/-- What unit `u` reads in `σ` depends only on `σ` on what it reads. -/
structure AddressMap (U S V : Type) where
  input : Finset S
  writes : U → Finset S
  reads : U → (S → V) → Finset S
  self : ∀ u σ τ, (∀ s ∈ reads u σ, σ s = τ s) → reads u σ = reads u τ

namespace AddressMap

variable {U S V : Type} (A : AddressMap U S V)

def Compatible (f : U → (S → V) → S → V) : Prop :=
  ∀ u σ τ, (∀ s ∈ A.reads u σ, σ s = τ s) → ∀ s ∈ A.writes u, f u σ s = f u τ s

def Correct (f : U → (S → V) → S → V) (σ : S → V) (u : U) : Prop := ∀ s ∈ A.writes u, σ s = f u σ s

/-- **The static wiring `R` expresses the address map at the public values `a`**: for every compatible choice of unit
functions, whether a unit is correct depends, among the assignments holding `a` on `Pub`, only on the slots `R` wires
into it and the slots it writes. -/
def Expresses (Pub : Finset S) (a : S → V) (R : U → Finset S) : Prop :=
  ∀ f, A.Compatible f → ∀ u (σ τ : S → V), (∀ t ∈ Pub, σ t = a t) → (∀ t ∈ Pub, τ t = a t) →
    (∀ s ∈ R u, σ s = τ s) → (∀ s ∈ A.writes u, σ s = τ s) → (A.Correct f σ u ↔ A.Correct f τ u)

open Classical in
/-- The closure's read sets: every slot unit `u` reads in some assignment holding `a` on `Pub`. -/
noncomputable def closureReads [Fintype S] (Pub : Finset S) (a : S → V) (u : U) : Finset S :=
  univ.filter fun s => ∃ σ : S → V, (∀ t ∈ Pub, σ t = a t) ∧ s ∈ A.reads u σ

end AddressMap

def DynPopulation.addressMap {U S V : Type} (D : DynPopulation U S V) (hac : D.AddressClosed) : AddressMap U S V where
  input := D.input
  writes := D.writes
  reads := D.reads
  self := hac

end Population
