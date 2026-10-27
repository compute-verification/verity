import Definitions.Pouw.NCP.Protocol
import Mathlib.Algebra.BigOperators.Intervals

/-!
# NCP's checked values on route U (trusted)

Route U (proposed) runs the stacked product as u8 × s8: the left operand is biased by a
parameter `β`, and the bias correction is the accumulator's initial value.
With `X = stackX k A E₁ R` (`m × 3k`), `Y = stackY k B R F₁` (`3k × n`) and `T = 3k/16`:

~~~text
X_b     = X + β·J                              stackXU: u8 for int7 A and noise in [−31, 32] (BiasU8)
c₀[j]   = −β · Σ_{l < 3k} Y[l][j]              initU: the same for every row
C^U_t   = c₀ + Σ_{l < 16t} X_b[:, l]·Y[l, :]   checkedU, over ℤ, t = 1 … T
~~~

`X + β` is u8 on the int7 domain exactly for `β ∈ {127, 128, 129}` when `R` is a permutation matrix (`Unsigned`, the
production `P`), and for `β ∈ {127, 128}` when `R` is a signed permutation (`BiasU8Iff`). ncp-v1 uses `β = 128`.

`runU` is the running value at any depth `τ` (`checkedU` reads it at `16t`), and `mmWU` is route U's int32
accumulator: it starts at `c₀` mod `2^32` and adds each exact product `X_b[i, l]·Y[l, j]`, wrapping. The MVP's
checkpoint `t` (from 0) is `mmWU β k A B E₁ F₁ R (min ((t + 1)·16) (3k))`, as its unbiased one is
`mmW (min ((t + 1)·16) (3k)) X Y`.

Over ℤ, `C^U_t = C_t − β·s_t` with `s_t[j] = Σ_{16t ≤ l < 3k} Y[l][j]` (`CheckedUEq`), so `C^U_T = A·B`. The
protocol whose checked values are route U's accumulator is `Params.toProtocolU β`: it is `Params.toProtocol` with
`Y`, `refChecked` and `refOut` read from `mmWU β`. Route U's honest reference with `f` forming units per position
costs `wrefU f = 3·m·k·n + f·m·k` (ncp-v1: `f = 24` at `β = 128`; C2: `f = 20` at `β = 129`).
-/

namespace Pouw.NCP

open Finset

/-- `R` is a permutation matrix on `{0, …, k − 1}`: a signed permutation whose signs are all `+1`. -/
def Unsigned (k : ℕ) (R : Mat) : Prop :=
  ∃ π : Equiv.Perm (Fin k), ∀ i j : Fin k, R i j = if i = π j then 1 else 0

/-- Route U's left operand `X_b = X + β·J`, the stacked activations biased entrywise. -/
def stackXU (β : ℤ) (k : ℕ) (A E1 R : Mat) : Mat := fun i l => stackX k A E1 R i l + β

/-- Route U's initial accumulator value `c₀[j] = −β · Σ_{l < 3k} Y[l][j]`, the same for every row. -/
def initU (β : ℤ) (k : ℕ) (B R F1 : Mat) (j : ℕ) : ℤ := -β * ∑ l ∈ range (3 * k), stackY k B R F1 l j

/-- Route U's running value at inner depth `τ`, over ℤ: `c₀[j] + Σ_{l < τ} X_b[i, l]·Y[l, j]`. -/
def runU (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat) (τ i j : ℕ) : ℤ :=
  initU β k B R F1 j + mm τ (stackXU β k A E1 R) (stackY k B R F1) i j

/-- **Route U's checked value** `C^U_t[i, j]`, over ℤ: the running value at depth `16t` (`t = 1 … 3k/16`). -/
def checkedU (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat) (t i j : ℕ) : ℤ := runU β k A B E1 F1 R (16 * t) i j

/-- An int32 accumulator started at `c`: `accWFrom c f 0 = wrap32 c`, `accWFrom c f (l + 1) = wrap32 (· + f l)`. -/
def accWFrom (c : ℤ) (f : ℕ → ℤ) : ℕ → ℤ
  | 0 => wrap32 c
  | l + 1 => wrap32 (accWFrom c f l + f l)

/-- Route U's int32 accumulator at depth `τ`: started at `c₀[j]`, adding `X_b[i, l]·Y[l, j]` for `l < τ`. -/
def mmWU (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat) (τ i j : ℕ) : ℤ :=
  accWFrom (initU β k B R F1 j) (fun l => stackXU β k A E1 R i l * stackY k B R F1 l j) τ

/-- Route U's honest reference per unit: the stacked product, `3·m·k·n`, and `f` forming units per position of the
left operand, `f·m·k` (the bias add and the u8 conversion). -/
def wrefU (f : ℕ) (L : Layout) (u : ℕ) : ℕ := 3 * (L.shape u).Wmm + f * (L.shape u).m * (L.shape u).k

/-- The bias `β` keeps `X + β` in u8 on the int7 domain for every `R` of the class `Rs`: every entry of `X_b` is in
`[0, 255]` for int7 `A` and `E₁` in `[−31, 32]`. -/
def BiasU8 (β : ℤ) (Rs : ℕ → Mat → Prop) : Prop :=
  ∀ (k : ℕ) (A E1 R : Mat), Rs k R → (∀ i, ∀ l < k, InNoise6 (E1 i l)) → (∀ i, ∀ l < k, InInt7 (A i l)) →
    ∀ i l, l < 3 * k → 0 ≤ stackXU β k A E1 R i l ∧ stackXU β k A E1 R i l ≤ 255

namespace Params

variable {Q R S : Type} (P : Params Q R S)

/-- Route U's checked values: block `t` is route U's int32 accumulator at `blockEnd d (3k) t`. -/
def YU (β : ℤ) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Checked :=
  fun t => mmWU β (U.layout.shape u).k A (U.weight (U.layout.wid u)) (P.e1 H s U u A) (P.wtNoise H s U u).1
    (P.wtNoise H s U u).2 (D1.blockEnd P.d (3 * (U.layout.shape u).k) t)

/-- Route U's useful output: its int32 accumulator at `3k`. -/
def refOutU (β : ℤ) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Mat :=
  mmWU β (U.layout.shape u).k A (U.weight (U.layout.wid u)) (P.e1 H s U u A) (P.wtNoise H s U u).1
    (P.wtNoise H s U u).2 (3 * (U.layout.shape u).k)

/-- NCP on route U at bias `β` as a protocol: `toProtocol` with route U's checked values and useful output. -/
def toProtocolU (β : ℤ) : Protocol Q R S where
  Y := P.YU β
  checkIdx := P.checkIdx
  refChecked := P.YU β
  refOut := P.refOutU β
  Wref := P.Wref

end Params

namespace RouteU

/-! ## Pinned statements -/

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
