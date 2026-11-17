module

public import Definitions.Pouw.Guarantees.NCP

/-! Steps of NCP's guarantees' proofs, stated as Props that no record pins. -/

@[expose] public section

namespace Pouw.Guarantees.NCP

section

open Pouw.NCP Pouw.NCP.Pinned

def Cancel : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat),
    mm k (A + E1) (B + F1) + mm k (A + E2 k E1 R) (B + F2 k R F1) - mm k (A + E3 k E1 R) (B + F3 k R F1) = mm k A B

def Stacked : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat),
    mm (3 * k) (stackX k A E1 R) (stackY k B R F1)
        = mm k (A + E1) (B + F1) + mm k (A + E2 k E1 R) (B + F2 k R F1) - mm k (A + E3 k E1 R) (B + F3 k R F1) ∧
      mm (3 * k) (stackX k A E1 R) (stackY k B R F1) = mm k A B

def Range : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat), SignedPerm k R →
    (∀ i, ∀ l < k, -31 ≤ E1 i l ∧ E1 i l ≤ 31) → (∀ l < k, ∀ j, -31 ≤ F1 l j ∧ F1 l j ≤ 31) →
    (∀ i, ∀ l < k, InInt7 (A i l)) → (∀ l < k, ∀ j, InInt7 (B l j)) →
    (∀ i l, l < 3 * k → -126 ≤ stackX k A E1 R i l ∧ stackX k A E1 R i l ≤ 126) ∧
      (∀ l j, l < 3 * k → -126 ≤ stackY k B R F1 l j ∧ stackY k B R F1 l j ≤ 126)

end

section

open Pouw.NCP Pouw.NCP.GameProofs Finset QTree

def ShiftBy24Admissible : Prop := ∀ {k : ℕ}, 1024 ≤ k → 16 ∣ k → Admissible k 16 (shiftBy 24 k)

end

section

open Pouw.NCP Pouw.NCP.Relation Finset

def IndependentWitness : Prop := ∀ k, 1024 ≤ k → 16 ∣ k → ∃ R, Admissible k 16 R ∧ CheckpointIndependent k 16 R

end

section

open Pouw.NCP Pouw.NCP.RouteU Finset

def BiasU8Iff : Prop :=
  ∀ β : ℤ, (BiasU8 β Unsigned ↔ 127 ≤ β ∧ β ≤ 129) ∧ (BiasU8 β SignedPerm ↔ 127 ≤ β ∧ β ≤ 128)

def CheckedUEq : Prop :=
  ∀ (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat) (t i j : ℕ), 16 * t ≤ 3 * k →
    checkedU β k A B E1 F1 R t i j =
      mm (16 * t) (stackX k A E1 R) (stackY k B R F1) i j - β * ∑ l ∈ Ico (16 * t) (3 * k), stackY k B R F1 l j

def CheckedUFinal : Prop :=
  ∀ (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat) (i j : ℕ), 16 ∣ k → checkedU β k A B E1 F1 R (3 * k / 16) i j = mm k A B i j

def RangeU : Prop :=
  ∀ (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat), 127 ≤ β → β ≤ 129 → Unsigned k R →
    (∀ i, ∀ l < k, InNoise6 (E1 i l)) → (∀ l < k, ∀ j, InNoise6 (F1 l j)) →
    (∀ i, ∀ l < k, InInt7 (A i l)) → (∀ l < k, ∀ j, InInt7 (B l j)) →
    (∀ i l, l < 3 * k → 0 ≤ stackXU β k A E1 R i l ∧ stackXU β k A E1 R i l ≤ 255) ∧
      (∀ l j, l < 3 * k → InInt8 (stackY k B R F1 l j))

def WrapU : Prop :=
  ∀ (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat) (i j : ℕ), 16 ∣ k → k ≤ 2 ^ 19 - 1 →
    (∀ l < k, InInt7 (A i l)) → (∀ l < k, InInt7 (B l j)) →
    mmWU β k A B E1 F1 R (16 * (3 * k / 16)) i j = mm k A B i j ∧
      wrap32 (checkedU β k A B E1 F1 R (3 * k / 16) i j) = mm k A B i j

def MmWUExact : Prop :=
  ∀ (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat) (τ i j : ℕ), 127 ≤ β → β ≤ 129 →
    (SignedPerm k R ∧ k ≤ 2 ^ 15 ∨ Unsigned k R ∧ k ≤ 2 ^ 16) →
    (∀ i, ∀ l < k, InNoise6 (E1 i l)) → (∀ l < k, ∀ j, InNoise6 (F1 l j)) →
    (∀ i, ∀ l < k, InInt7 (A i l)) → (∀ l < k, ∀ j, InInt7 (B l j)) → τ ≤ 3 * k →
    mmWU β k A B E1 F1 R τ i j = runU β k A B E1 F1 R τ i j

def YUChecked : Prop :=
  ∀ {Q R S : Type} (P : Params Q R S) (β : ℤ) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) (t i j : ℕ),
    P.d = 16 → 16 ∣ (U.layout.shape u).k → t < 3 * (U.layout.shape u).k / 16 →
    P.YU β H s U u A t i j = wrap32 (checkedU β (U.layout.shape u).k A (U.fixed (U.layout.fixedId u))
      (P.e1 H s U u A) (P.wtNoise H s U u).1 (P.wtNoise H s U u).2 (t + 1) i j)

def TTNCPUIff : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Params Q R S) (β : ℤ) (D : Layout → Prop) (g₀ : ℝ) (δ : ℕ → ℕ → ℝ),
    Assumptions.TTNCP_U β CM P D g₀ δ ↔ Assumptions.TTNCP CM (P.toProtocolU β) P.d D g₀ δ

def TTNCPUWitness : Prop :=
  ∀ (Q R : Type) [Fintype Q] [DecidableEq Q] [Fintype R] (P : Params Q R Salt) (β : ℤ) (c : ℕ) (g₀ : ℝ),
    0 ≤ g₀ → g₀ < 1 → P.QueryBound c →
    (∀ U : Workload, U.InDomain leadDomain → ∀ u < U.layout.N, 3 * (U.layout.shape u).Wmm ≤ P.Wref U.layout u) →
    Assumptions.TTNCP_U β (outputPriced3 Q R Salt) P leadDomain g₀ (fun _ _ => 0) ∧
    AdmitsRef (outputPriced3 Q R Salt) (P.toProtocolU β) leadDomain (fun U => U.layout.N * c)

end

section

open Pouw.NCP Pouw.NCP.RouteU Finset Pouw.NCP.Relation

def NCPWordFloorIndU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary) (g₀ : ℝ), 16 ∣ k →
    CheckpointIndependent k 16 R → 0 ≤ g₀ → g₀ < 1 → Word.Valid (fun _ => positions k m n) adv →
    Word.NzNCP (fun _ => positions k m n) (checkedUPos β k m n N R A B) →
    pr (fun ω => (Word.cost (fun _ => positions k m n) N adv : ℝ) / (1 - g₀) <
        16 * ∑ _u ∈ Word.correctSet (fun _ => positions k m n) (checkedUPos β k m n N R A B) N adv ω,
          (positions k m n : ℝ)) ≤
      N * (2 : ℝ) ^ (-((m * n : ℕ) * (g₀ * (3 * k / 16 : ℕ) - 1)))

def Int7Int32UPos : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (ω : Noise N m n k) (u p : ℕ), 127 ≤ β → β ≤ 129 →
    (SignedPerm k R ∧ k ≤ 2 ^ 15 ∨ Unsigned k R ∧ k ≤ 2 ^ 16) →
    (∀ i, ∀ l < k, InInt7 (A u i l)) → (∀ l < k, ∀ j, InInt7 (B u l j)) → p < positions k m n →
    InInt32 (checkedUPos β k m n N R A B ω u p)

def WrapFinalOnlyIndU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary), 16 ∣ k →
    CheckpointIndependent k 16 R → Word.Valid (fun _ => positions k m n) adv →
    (∀ (ω : Noise N m n k) (u p : ℕ), u < N → p < positions k m n → InInt32 (checkedUPos β k m n N R A B ω u p)) →
    ∀ u < N, Word.identCount (fun _ => positions k m n) (wrappedU β k m n N R A B) adv u ≤ m * n

def WrapWordFloorIndU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary) (g₀ : ℝ), 16 ∣ k →
    CheckpointIndependent k 16 R → 0 ≤ g₀ → g₀ < 1 → Word.Valid (fun _ => positions k m n) adv →
    (∀ (ω : Noise N m n k) (u p : ℕ), u < N → p < positions k m n → InInt32 (checkedUPos β k m n N R A B ω u p)) →
    Word.NzNCP (fun _ => positions k m n) (wrappedU β k m n N R A B) →
    pr (fun ω => (Word.cost (fun _ => positions k m n) N adv : ℝ) / (1 - g₀) <
        16 * ∑ _u ∈ Word.correctSet (fun _ => positions k m n) (wrappedU β k m n N R A B) N adv ω,
          (positions k m n : ℝ)) ≤
      N * (2 : ℝ) ^ (-((m * n : ℕ) * (g₀ * (3 * k / 16 : ℕ) - 1)))

end
end Pouw.Guarantees.NCP
