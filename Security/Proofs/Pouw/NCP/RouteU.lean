import Definitions.Pouw.NCP.RouteU

/-! The proof-only part of `Definitions.Pouw.NCP.RouteU`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.NCP

open Finset

namespace RouteU

/-- **The int32 reading** (every `β`): route U's int32 accumulator at every depth is its running value mod `2^32`. -/
def MmWUEq : Prop :=
  ∀ (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat) (τ i j : ℕ), mmWU β k A B E1 F1 R τ i j = wrap32 (runU β k A B E1 F1 R τ i j)

/-- **The int32 range on route U, signed `R`** (`Chain.Int7Int32` on route U, at half its `k`): for
`β ∈ {127, 128, 129}`, a signed-permutation `R`, `E₁`, `F₁` in `[−31, 32]`, int7 `A`, `B` and `k ≤ 2^15`, route U's
running value at every depth `τ ≤ 3k` is at most `16,129·τ + 127·β·(3k − τ) ≤ 49,149·k ≤ 1,610,514,432 < 2^31` in
magnitude: `C_τ` minus `β` times the suffix sum. At `k = 2^16` the bound is `3.2·10^9`, and with signs `σ ≡ −1`
`c₀` alone can exceed `2^31`. -/
def Int7Int32U : Prop :=
  ∀ (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat) (τ i j : ℕ), 127 ≤ β → β ≤ 129 → SignedPerm k R →
    (∀ i, ∀ l < k, InNoise6 (E1 i l)) → (∀ l < k, ∀ j, InNoise6 (F1 l j)) →
    (∀ i, ∀ l < k, InInt7 (A i l)) → (∀ l < k, ∀ j, InInt7 (B l j)) → k ≤ 2 ^ 15 → τ ≤ 3 * k →
    InInt32 (runU β k A B E1 F1 R τ i j)

/-- **The int32 range on route U, permutation-matrix `R`** (the production `P`, `Chain.Int7Int32`'s `k`): for
`β ∈ {127, 128, 129}`, a permutation matrix `R`, `E₁`, `F₁` in `[−31, 32]`, int7 `A`, `B` and `k ≤ 2^16`, route U's
running value at every depth `τ ≤ 3k` is at most `29,856·k ≤ 1,956,642,816 < 2^31` in magnitude. There
`Σ_l F₂[l][j] = −Σ_l F₁[l][j]`, so `c₀ = −3β·Σ_l B[l][j]`; grouped by original index `a < k`, each of `a`'s three
stacked terms plus `c₀`'s share, `−3β·B[a][j]`, moved by a multiple of `β·(F₁ + F₂)[a][j]` (which sums to `0`), is
at most `96² + 2·64·129 + 32·129 = 29,856`. -/
def Int7Int32UUnsigned : Prop :=
  ∀ (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat) (τ i j : ℕ), 127 ≤ β → β ≤ 129 → Unsigned k R →
    (∀ i, ∀ l < k, InNoise6 (E1 i l)) → (∀ l < k, ∀ j, InNoise6 (F1 l j)) →
    (∀ i, ∀ l < k, InInt7 (A i l)) → (∀ l < k, ∀ j, InInt7 (B l j)) → k ≤ 2 ^ 16 → τ ≤ 3 * k →
    InInt32 (runU β k A B E1 F1 R τ i j)

/-- **Route U's honest reference is complete** on the lead's domain, for every NCP instance, bias, noise and `R`: its
checked values are `YU` by definition, and its useful output, the accumulator at `3k`, is `A·B`. -/
def LeadCompleteU : Prop :=
  ∀ (Q R S : Type) (P : Params Q R S) (β : ℤ), (P.toProtocolU β).Complete leadDomain

end RouteU

end Pouw.NCP
