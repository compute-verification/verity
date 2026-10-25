import Definitions.Pouw.NCP.Relation
import Definitions.Pouw.Assumptions.NCP.GamePinned

/-!
# The chain end to end: the wrap link and the lead with independence (trusted)

`Params.Y`, the protocol's checked values, are the running sums of the stacked product mod `2^32` (`mmW`, the int32
accumulator). `ParamsYLink` states that the checked value of block `t`, entry `(i, j)` is `wrap32` of the integer
checked value `Relation.ck` at depth `blockEnd d (3k) t`, and `LeadDepth` that at `d = 16`, `16 ∣ k` this depth is
`Relation.checked`'s. `wrapped` is `Relation.checked` wrapped mod `2^32`: the protocol's checked values in the word
model. In the int32-exact range `wrap32` is the identity on the values that occur, so the wrapped word model has the
same identities (`WrapIdentity`), and `FinalOnly` and the floor under `CheckpointIndependent` hold for it
(`WrapFinalOnlyInd`, `WrapWordFloorInd`). The range holds at zero input for `k ≤ 2^16` (`ZeroInputInt32`: every
running sum is at most `3k·64²`) and for int7 inputs for `k ≤ 2^16` (`Int7Int32`: at most `20,480·k`).

`IsLeadInd` is the lead's parameters with every `R` of a default-domain shape having independent checkpoints;
`LeadIndWitness` is `LeadWitness` with `IsLeadInd` in place of `IsLead`.
-/

namespace Pouw.NCP.Chain

open Finset Pouw.NCP.Relation

/-- **The wrap link**: `Params.Y` at block `t`, entry `(i, j)` is `wrap32` of the integer checked value at depth
`blockEnd d (3k) t`. -/
def ParamsYLink : Prop :=
  ∀ {Q R S : Type} (P : Params Q R S) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) (t i j : ℕ),
    P.Y H s U u A t i j = wrap32 (ck (U.layout.shape u).k A (U.weight (U.layout.wid u)) (P.e1 H s U u A)
      (P.wtNoise H s U u).1 (P.wtNoise H s U u).2 i j (D1.blockEnd P.d (3 * (U.layout.shape u).k) t))

/-- At `d = 16` and `16 ∣ k`, block `p / (m n)` of position `p` ends at `Relation.checked`'s depth. -/
def LeadDepth : Prop :=
  ∀ k m n p : ℕ, 16 ∣ k → p < positions k m n → D1.blockEnd 16 (3 * k) (p / (m * n)) = 16 * (p / (m * n) + 1)

/-- The protocol's checked values in the word model: `Relation.checked` mod `2^32`. -/
def wrapped (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (ω : Noise N m n k) (u p : ℕ) : ℤ :=
  wrap32 (checked k m n N R A B ω u p)

/-- In the int32-exact range, a rule is an identity of the wrapped checked values iff it is one of the integer ones. -/
def WrapIdentity : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary) (u p : ℕ),
    Word.Valid (fun _ => positions k m n) adv → p < positions k m n →
    (∀ (ω : Noise N m n k) (q : ℕ), q ≤ p → InInt32 (checked k m n N R A B ω u q)) →
    (Word.Identity (wrapped k m n N R A B) adv u p ↔ Word.Identity (checked k m n N R A B) adv u p)

/-- **FinalOnly for the protocol's checked values**, under independence, in the int32-exact range. -/
def WrapFinalOnlyInd : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary), 16 ∣ k → CheckpointIndependent k 16 R →
    Word.Valid (fun _ => positions k m n) adv →
    (∀ (ω : Noise N m n k) (u p : ℕ), u < N → p < positions k m n → InInt32 (checked k m n N R A B ω u p)) →
    ∀ u < N, Word.identCount (fun _ => positions k m n) (wrapped k m n N R A B) adv u ≤ m * n

/-- **The floor for the protocol's checked values**, under independence, in the int32-exact range. -/
def WrapWordFloorInd : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary) (γ₀ : ℝ), 16 ∣ k → CheckpointIndependent k 16 R →
    0 ≤ γ₀ → γ₀ < 1 → Word.Valid (fun _ => positions k m n) adv →
    (∀ (ω : Noise N m n k) (u p : ℕ), u < N → p < positions k m n → InInt32 (checked k m n N R A B ω u p)) →
    Word.NzNCP (fun _ => positions k m n) (wrapped k m n N R A B) →
    pr (fun ω => (Word.cost (fun _ => positions k m n) N adv : ℝ) / (1 - γ₀) <
        16 * ∑ _u ∈ Word.correctSet (fun _ => positions k m n) (wrapped k m n N R A B) N adv ω,
          (positions k m n : ℝ)) ≤
      N * (2 : ℝ) ^ (-((m * n : ℕ) * (γ₀ * (3 * k / 16 : ℕ) - 1)))

/-- **The int32 range at zero input**: for a signed-permutation `R` and `k ≤ 2^16`, every checked value is at most
`3k·64² ≤ 805,306,368 < 2^31` in magnitude. -/
def ZeroInputInt32 : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (ω : Noise N m n k) (u p : ℕ), SignedPerm k R → k ≤ 2 ^ 16 → p < positions k m n →
    InInt32 (checked k m n N R (fun _ => 0) (fun _ => 0) ω u p)

/-- **The int32 range at int7 inputs** (worst case, degenerate inputs included): for a signed-permutation `R`, int7
`A u` and `B u` and `k ≤ 2^16`, every checked value is at most `20,480·k ≤ 1,342,177,280 < 2^31` in magnitude. Blocks 1
and 2 give at most `2k·96²`; in block 3 the running sum is `A·B` (at most `64²·k`) minus the rest of block 3 (at most
`128²·k`). -/
def Int7Int32 : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (ω : Noise N m n k) (u p : ℕ), SignedPerm k R → k ≤ 2 ^ 16 →
    (∀ i, ∀ l < k, InInt7 (A u i l)) → (∀ l < k, ∀ j, InInt7 (B u l j)) → p < positions k m n →
    InInt32 (checked k m n N R A B ω u p)

/-- Every `R` of a default-domain shape has independent checkpoints. -/
def RIndependent {Q R S : Type} (P : Params Q R S) : Prop :=
  ∀ (sh : Shape) (b : List R), Shape.InDefaultDomain sh → CheckpointIndependent sh.k P.d (P.expWt sh b).2

/-- The lead's parameters with independence. -/
def IsLeadInd {Q R S : Type} (P : Params Q R S) : Prop := P.IsLead ∧ RIndependent P

/-- **The lead's instance with independence, jointly**: `GamePinned.LeadWitness` with `IsLeadInd` in place of
`IsLead`. -/
def LeadIndWitness : Prop :=
  ∃ (P : Params Unit Unit Salt) (Lay : Layout), IsLeadInd P ∧ P.RSigned ∧ P.QueryBound 0 ∧
    (∀ (L : Layout) (u : ℕ), P.Wref L u = 3 * (L.shape u).Wmm) ∧
    P.toProtocol.Complete leadDomain ∧
    AdmitsRef (outputPriced3 Unit Unit Salt) P.toProtocol leadDomain (fun _ => 0) ∧
    (∀ γ₀ : ℝ, 0 ≤ γ₀ → γ₀ < 1 →
      Assumptions.TTNCP (outputPriced3 Unit Unit Salt) P.toProtocol P.d leadDomain γ₀ (fun _ _ => 0)) ∧
    (⟨Lay, fun _ => 0⟩ : Workload).InDomain leadDomain ∧ 0 < Lay.N ∧ ∀ u < Lay.N, P.d ∣ (Lay.shape u).k

end Pouw.NCP.Chain
