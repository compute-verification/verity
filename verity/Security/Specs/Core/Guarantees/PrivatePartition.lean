import Definitions.Core.PrivatePartition
import Specs.Core.Assumptions

/-! # What a private partition's public predicates establish -/

namespace Verity.Guarantees

/-- **The per-Call check suffices**: when every unit's outputs inside its Call meet the width rule, every unit's
outputs in the whole circuit do. -/
def PrivateFitsOfLocal : Prop :=
  ∀ (C : Circuit) (m n : ℕ) (K : Calls C m) (P : Partition C n) (X W : ℕ),
    P.FitsLocally K X W → P.Fits X W

/-- **The leakage weight**: under the width rule, any set of units, such as the wrong ones, has at most `max X W` output
bits per unit. -/
def PrivateOutBitsLe : Prop :=
  ∀ (C : Circuit) (n : ℕ) (P : Partition C n) (X W : ℕ), P.Fits X W →
    ∀ T : Finset (Fin n), ∑ u ∈ T, P.outBits u ≤ T.card * max X W

/-- **A registered partition fits the public count**: under the advice's binding, every partition the hidden root
commits to has at most `Q.units` units, each within the width rule, so any set of them has at most `max X W` output
bits per unit. -/
def RegisteredMeets : Prop :=
  ∀ {R : Type} (C : Circuit) (advice : (n : ℕ) → Partition C n → R), Assumptions.AdviceBinding C advice →
    ∀ (Q : Predicates) (root : R), Registered C advice Q root →
      ∀ n (P : Partition C n), advice n P = root → n ≤ Q.units ∧ P.Fits Q.X Q.W ∧
        ∀ T : Finset (Fin n), ∑ u ∈ T, P.outBits u ≤ T.card * max Q.X Q.W

end Verity.Guarantees
