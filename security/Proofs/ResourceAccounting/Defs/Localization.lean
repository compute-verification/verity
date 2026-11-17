module

public import Definitions.ResourceAccounting.Actual

/-! The actual-circuit model's reach on one actual circuit: `downstream`, `Reaches` and `SendsIndependent`. -/

@[expose] public section

namespace ResourceAccounting.Placed

open Finset

variable {L : Type*} (A : Placed L)

open Classical in
noncomputable def downstream (C : Finset (Fin A.N)) : Finset (Fin A.N) :=
  univ.filter fun g => ∃ c ∈ C, Relation.ReflTransGen (fun a g => a ∈ A.reads g) c g

def Reaches (C : Finset (Fin A.N)) (X Y : Finset L) (W : Set ℕ) : Prop :=
  ∀ g ∈ A.region Y W, ∀ a ∈ A.reads g, a ∉ C ∧ (A.x a ∉ X → A.x a ∉ Y → a ∉ A.downstream C)

def SendsIndependent (C : Finset (Fin A.N)) (X Y : Finset L) (W : Set ℕ) : Prop :=
  ∀ g ∈ A.region Y W, ∀ a ∈ A.reads g, A.x a ∈ X → a ∉ A.downstream C

end ResourceAccounting.Placed
