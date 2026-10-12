import Models.Pouw.Assumptions.NCP.Chain
import Definitions.Pouw.NCP.RouteU

/-!
# The word layer on route U (trusted statements)

Route U's checked value at position `p` of unit `u` is `checkedU β` at checkpoint `t = p/(m n) + 1` and entry
`e = p % (m n)`, as `Relation.checked` is `ck` at depth `16·t` (`checkedUPos`); `wrappedU` is it mod `2^32`, the
protocol's checked value (`RouteU.YUChecked`).

The checkpoint matrices stay the unbiased ones: `M_τ` is `Relation.ckMat`, the coefficient of `E₁[i, a]·F₁[b, j]` in
`C_τ`, and route U's shift `−β·s_t` is a function of `F₁` alone (no `E₁`), so the coefficient of `E₁[i, a]·F₁[b, j]`
in `C^U_τ` is the same `M_τ[a, b]`. The four-point mixed difference cancels the shift, so the reduction's "only if"
transfers as it is (`ReductionOnlyIfU`). Its converse, `Relation.ReductionIf`, does not: it is stated at zero input,
where `C` is bilinear in the noise, and `C^U` is not (its shift is linear in `F₁`, and `c₀ ≠ 0`).

With the reduction, independence gives FinalOnly (`NCPFinalOnlyIndU`) and the floor (`NCPWordFloorIndU`) for route-U
words; in the int32-exact range (`Int7Int32UPos`: signed `R` and `k ≤ 2^15`, or a permutation matrix and
`k ≤ 2^16`) the same holds for the wrapped words (`WrapFinalOnlyIndU`, `WrapWordFloorIndU`). The floors take NzNCP
of the route-U values as their own hypothesis: NzNCP of `C` does not give NzNCP of `C^U`, since a rule's event for
`C^U` is not a rule's event for `C` (the shift is random through `F₁`). NzNCP itself is satisfiable
(`Word.WordNzWitness`), and the floors' other hypotheses hold at the lead (`RelationWitness.IndependentWitness`).
-/

namespace Pouw.NCP

open Finset Pouw.NCP.Relation

/-- Route U's checked value at position `p` of unit `u`: checkpoint `p / (m n) + 1`, entry `e = p % (m n)`, which is
`(i, j) = (e / n, e % n)`. -/
def checkedUPos (β : ℤ) (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (ω : Noise N m n k) (u p : ℕ) : ℤ :=
  checkedU β k (A u) (B u) (e1Of ω u) (f1Of ω) R (p / (m * n) + 1) (p % (m * n) / n) (p % (m * n) % n)

/-- Route U's protocol checked values in the word model: `checkedUPos` mod `2^32`. -/
def wrappedU (β : ℤ) (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (ω : Noise N m n k) (u p : ℕ) : ℤ :=
  wrap32 (checkedUPos β k m n N R A B ω u p)

namespace RouteU

/-! ## Pinned statements -/

/-- **The reduction on route U, grouped by entry** (`Relation.ReductionOnlyIf` for `C^U`, every `β` and input): if a
rule is an identity in the noise of route U's checked values at position `p`, then for every entry `e` the unbiased
coefficients agree: position `p`'s coefficient is the rule applied to the operands' coefficients, the constant
dropped. -/
def ReductionOnlyIfU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (u p : ℕ) (r : Word.Rule), u < N →
    (∀ ω : Noise N m n k, checkedUPos β k m n N R A B ω u p = r.target (checkedUPos β k m n N R A B ω u)) →
    ∀ e < m * n, ∀ a < k, ∀ b < k,
      coefAt k m n R p e a b = r.target (fun q => coefAt k m n R q e a b) - r.target (fun _ => 0)

/-- **FinalOnly for route U under independence** (`Relation.NCPFinalOnlyInd` for `C^U`): every unit has at most
`m·n` identity positions (the final ones). -/
def NCPFinalOnlyIndU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary), 16 ∣ k →
    CheckpointIndependent k 16 R → Word.Valid (fun _ => positions k m n) adv →
    ∀ u < N, Word.identCount (fun _ => positions k m n) (checkedUPos β k m n N R A B) adv u ≤ m * n

end RouteU

end Pouw.NCP
