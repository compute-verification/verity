module

public import Definitions.Pouw.Assumptions.NCP.Assumptions
public import Definitions.Pouw.Assumptions.NCP.GamePinned
public import Definitions.Pouw.Game.Params
public import Definitions.Pouw.NCP.Relation
public import Definitions.Pouw.NCP.Witness

/-! The chain end to end: the wrap link from `Params.Y` to the integer checked values, the int32-exact range, and the lead with independent checkpoints. -/

@[expose] public section

namespace Pouw.NCP.Chain

open Finset Pouw.NCP.Relation

def ParamsYLink : Prop :=
  ∀ {Q R S : Type} (P : Params Q R S) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) (t i j : ℕ),
    P.Y H s U u A t i j = wrap32 (ck (U.layout.shape u).k A (U.fixed (U.layout.fixedId u)) (P.e1 H s U u A)
      (P.wtNoise H s U u).1 (P.wtNoise H s U u).2 i j (D1.blockEnd P.d (3 * (U.layout.shape u).k) t))

def LeadDepth : Prop :=
  ∀ k m n p : ℕ, 16 ∣ k → p < positions k m n → D1.blockEnd 16 (3 * k) (p / (m * n)) = 16 * (p / (m * n) + 1)

/-- The protocol's checked values in the word model. -/
def wrapped (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (ω : Noise N m n k) (u p : ℕ) : ℤ :=
  wrap32 (checked k m n N R A B ω u p)

def WrapIdentity : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary) (u p : ℕ),
    Word.Valid (fun _ => positions k m n) adv → p < positions k m n →
    (∀ (ω : Noise N m n k) (q : ℕ), q ≤ p → InInt32 (checked k m n N R A B ω u q)) →
    (Word.Identity (wrapped k m n N R A B) adv u p ↔ Word.Identity (checked k m n N R A B) adv u p)

def WrapFinalOnlyInd : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary), 16 ∣ k → CheckpointIndependent k 16 R →
    Word.Valid (fun _ => positions k m n) adv →
    (∀ (ω : Noise N m n k) (u p : ℕ), u < N → p < positions k m n → InInt32 (checked k m n N R A B ω u p)) →
    ∀ u < N, Word.identCount (fun _ => positions k m n) (wrapped k m n N R A B) adv u ≤ m * n

def WrapWordFloorInd : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary) (g₀ : ℝ), 16 ∣ k → CheckpointIndependent k 16 R →
    0 ≤ g₀ → g₀ < 1 → Word.Valid (fun _ => positions k m n) adv →
    (∀ (ω : Noise N m n k) (u p : ℕ), u < N → p < positions k m n → InInt32 (checked k m n N R A B ω u p)) →
    Word.NzNCP (fun _ => positions k m n) (wrapped k m n N R A B) →
    pr (fun ω => (Word.cost (fun _ => positions k m n) N adv : ℝ) / (1 - g₀) <
        16 * ∑ _u ∈ Word.correctSet (fun _ => positions k m n) (wrapped k m n N R A B) N adv ω,
          (positions k m n : ℝ)) ≤
      N * (2 : ℝ) ^ (-((m * n : ℕ) * (g₀ * (3 * k / 16 : ℕ) - 1)))

def ZeroInputInt32 : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (ω : Noise N m n k) (u p : ℕ), SignedPerm k R → k ≤ 2 ^ 16 → p < positions k m n →
    InInt32 (checked k m n N R (fun _ => 0) (fun _ => 0) ω u p)

/-- **The int32 range at int7 inputs**, worst case, degenerate inputs included. -/
def Int7Int32 : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (ω : Noise N m n k) (u p : ℕ), SignedPerm k R → k ≤ 2 ^ 16 →
    (∀ i, ∀ l < k, InInt7 (A u i l)) → (∀ l < k, ∀ j, InInt7 (B u l j)) → p < positions k m n →
    InInt32 (checked k m n N R A B ω u p)

def RIndependent {Q R S : Type} (P : Params Q R S) : Prop :=
  ∀ (sh : Shape) (b : List R), Shape.InDefaultDomain sh → CheckpointIndependent sh.k P.d (P.expWt sh b).2

def IsLeadInd {Q R S : Type} (P : Params Q R S) : Prop := P.IsLead ∧ RIndependent P

/-- `GamePinned.LeadWitness` with `IsLeadInd` in place of `IsLead`. -/
def LeadIndWitness : Prop :=
  ∃ (P : Params Unit Unit Salt) (Lay : Layout), IsLeadInd P ∧ P.RSigned ∧ P.QueryBound 0 ∧
    (∀ (L : Layout) (u : ℕ), P.Wref L u = 3 * (L.shape u).Wmm) ∧
    P.toProtocol.Complete leadDomain ∧
    AdmitsRef (outputPriced3 Unit Unit Salt) P.toProtocol leadDomain (fun _ => 0) ∧
    (∀ g₀ : ℝ, 0 ≤ g₀ → g₀ < 1 →
      Assumptions.TTNCP (outputPriced3 Unit Unit Salt) P.toProtocol P.d leadDomain g₀ (fun _ _ => 0)) ∧
    (⟨Lay, fun _ => 0⟩ : Workload).InDomain leadDomain ∧ 0 < Lay.N ∧ ∀ u < Lay.N, P.d ∣ (Lay.shape u).k

end Pouw.NCP.Chain
