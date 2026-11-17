module

public import Definitions.Pouw.Guarantees.FillerLuck

/-! Steps of `Definitions.Pouw.Guarantees.FillerLuck`'s proofs, each a `def G : Prop` proved by the `SecurityProofs`
theorem of the same name and pinned by no record. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.FillerLuck

open ResourceAccounting.Filler ResourceAccounting.TS Pouw.RelativeSecurity

def ParityLuck : Prop :=
  ∀ {Y A : Type} [Fintype Y] [CommRing A] (φ : A →+* ZMod 2) {K : ℕ} (c : Y → Fin K → A) (R : ℕ)
    (P : (Fin K → ZMod 2) → Y → A),
    ParityBalanced φ c → Products R P →
      prob (fun xy : (Fin K → ZMod 2) × Y => P xy.1 xy.2 = cellWord c xy.1 xy.2) ≤ luck (K - R)

def FillerParityBalanced : Prop :=
  ∀ K : ℕ, ParityBalanced toF2 (bitsCol (K := K)) ∧ ParityBalanced (Int.castRingHom (ZMod 2)) (bitsCol (K := K)) ∧
    ParityBalanced toF2 (buCol (K := K)) ∧ ParityBalanced (Int.castRingHom (ZMod 2)) (buCol (K := K))

end ResourceAccounting.Guarantees.FillerLuck
