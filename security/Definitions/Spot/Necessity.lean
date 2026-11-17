module

public import Definitions.Core.Coins
public import Definitions.Spot.Dataflow
public import Mathlib.Basic.Real.Basic
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs

/-! The IOP's sampling certificate, and what each policy needs of it. -/

@[expose] public section

namespace Population.Iop

open Finset

variable {U T : Type}

open Classical in
noncomputable def bad [Fintype U] (chk : U → T → Prop) (σ : T) : Finset U := univ.filter fun u => ¬ chk u σ

noncomputable abbrev mass (π : PMF U) (B : Finset U) : ℝ := Verity.Game.prPMFReal π (· ∈ B)

def Accepts (chk : U → T → Prop) (A : Finset U) {k : ℕ} (d : Fin k → U) (σ : T) : Prop :=
  (∀ u ∈ A, chk u σ) ∧ ∀ i, chk (d i) σ

noncomputable abbrev prob [Fintype U] (π : PMF U) (k : ℕ) (E : (Fin k → U) → Prop) : ℝ :=
  Verity.Game.prPMFReal (Verity.Game.iid π k) E

def Certified [Fintype U] (chk : U → T → Prop) (π : PMF U) (A : Finset U) (ε : ℝ) (σ : T) : Prop :=
  Disjoint (bad chk σ) A ∧ mass π (bad chk σ) ≤ ε

def Local {Q V : Type} (reads : U → Finset Q) (chk : U → (Q → V) → Prop) : Prop :=
  ∀ u σ τ, (∀ q ∈ reads u, σ q = τ q) → (chk u σ ↔ chk u τ)

/-- A policy is inferable from the certificate exactly when such a `K` is known (`Proofs.Spot.NecessitySkeleton`,
`inference_iff`). -/
def AmpLe [Fintype U] (chk : U → T → Prop) (A : Finset U) (c : U → ℝ) (dmg : T → ℝ) (K : ℝ) : Prop :=
  ∀ σ, Disjoint (bad chk σ) A → dmg σ ≤ K * ∑ u ∈ bad chk σ, c u

section Dataflow

variable {S : Type} (P : Dataflow U S)

def corr {V : Type} (f : P.Fns V) (u : U) (σ : S → V) : Prop := P.Correct f σ u

def DeterminateAt {V : Type} (f : P.Fns V) (a : S → V) : Prop :=
  ∃! σ : S → V, (∀ s ∈ P.input, σ s = a s) ∧ ∀ u, P.Correct f σ u

/-- `Dataflow.eq_of_writes` proves it from single assignment and acyclicity, for every `f`. -/
def RobustAt [Fintype U] {V : Type} (f : P.Fns V) : Prop :=
  ∀ (B : Finset U) (σ τ : S → V), (∀ s ∈ P.input, σ s = τ s) → P.incorrect f σ ⊆ B → P.incorrect f τ ⊆ B →
    (∀ u ∈ B, ∀ s ∈ P.writes u, σ s = τ s) → σ = τ

open Classical in
noncomputable def damaged {V : Type} (O : Finset S) (σ τ : S → V) : Finset S := O.filter fun o => σ o ≠ τ o

open Classical in
noncomputable def egressWidth [Fintype U] [Fintype S] {V : Type} [Fintype V] (f : P.Fns V) (a : S → V)
    (X O : Finset S) (M : ℕ) : Finset (O → V) :=
  (univ.filter fun σ : S → V => P.Holds a X σ ∧ ∑ u ∈ P.incorrect f σ, (P.writes u).card ≤ M).image fun σ o => σ o

end Dataflow

/-- When `R u` holds every slot `u` reads in every admitted run, addresses may be computed and corrupted but stay inside
`R u`. -/
def rangeFlow {V S : Type} (D : DynPopulation U S V) (R : U → Finset S) : Dataflow U S where
  input := D.input
  reads := R
  writes := D.writes

end Population.Iop
