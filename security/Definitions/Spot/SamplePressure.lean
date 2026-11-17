module

public import Definitions.Core.Draw
public import Mathlib.Data.Fintype.Powerset
public import Mathlib.Probability.Distributions.Uniform
import Mathlib.Algebra.Order.BigOperators.Expect
import Mathlib.Algebra.Order.Field.Power
import Mathlib.Analysis.Complex.Order
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Data.EReal.Inv

/-! The deployed sample: each window's coin a uniform `k`-subset of the `n` units, drawn without replacement. -/

@[expose] public section

namespace Population.Exfil

open Finset

noncomputable def uniformSample (n k : ℕ) (hk : k ≤ n) : PMF (Verity.Draw.Subset n k) :=
  haveI : Nonempty (Verity.Draw.Subset n k) :=
    (exists_subset_card_eq (s := (univ : Finset (Fin n))) (by simpa using hk)).elim fun t ht => ⟨⟨t, ht.2⟩⟩
  PMF.uniformOfFintype _

def missing (n k : ℕ) (B : Finset (Fin n)) : Finset (Verity.Draw.Subset n k) := univ.filter fun s => Disjoint s.1 B

end Population.Exfil
