import Models.Accounting.Assumptions.SharedNoise

/-!
# One noise seed a call: the bridge and the grid's moment (statements; a draft for the red team)

J10 (note:compute-accounting/20261019T0825Z-draft-j10-shared-noise). Statements only, none proved yet:

* `CondReceiptLuckMarginal`: luck given the rest implies the marginal luck, so `Assumptions.SharedNoise.KeyReceiptLuckAt`
  gives `ReceiptLuck` per tile over the call's coin, what `PerTile … ReceiptLuckAt` reads.
* `GridLuck`: `TileLuck` for a grid whose rows' noise is shared along its row and whose columns' along its column. Row
  `i` has its own coins in `Rw` (`TileLuck`'s rows' coins `Ω₀`, one row's part each) and column `j` its own in `Cl`.
  Unit `(i, j)` skips `w i j ≥ 0`; it is dead with probability at most `q` over row `i`'s coins alone, and otherwise
  lucky with probability at most `p` over its column's, whatever the rows. If every unit meets
  `q·e^{nθw} + (1 − q)·(p·e^{rθw} + (1 − p)·e^{−κ·rθw})^{n/r} ≤ 1`, the grid's gain has moment at most `1` at rate
  `θ`. Paper proof (the note's §4): given the rows, columns are independent, and Hölder over a column's `r` units gives
  `∏_{i,j} φ_{ij}^{1/r}`, `φ` the unit's moment at `rθ` (`e^{rθw}` dead, `LuckMoment`'s bound otherwise); `φ_{ij}`
  reads row `i`'s coins only, so rows are independent, and Hölder over a row's `n` units leaves each unit's
  `E[φ^{n/r}]`, which is the condition. `TileLuck` has `n` where this has `n/r`, because it lets dead events of different
  rows correlate; h1tile's dead events read one row's dither, so they don't.
* `GridTemplateCondition` and `GridTemplateMoment`: `GridLuck` at h1tile's template (`q = 2^{-w}`, `p = ½`, `κ = 4`,
  `w ≤ 32`) at `r·θ = 1/(4·max(64, n/r))`: `θ = 1/(4·max(64·r, n))` a product. At one tile (`r = n = 32`) it is
  `TemplateMoment`'s `1/(256·32)`; on a call grid of `m_c × n_c` tiles of `32 × 32` (`r = 32·m_c`, `n = 32·n_c`) it is
  that rate over `max(m_c, n_c/64)`, so a seed group's premium is `max(m_c, n_c/64)` times one seed a tile's: none at
  decode (`m_c = 1`) up to `n_c = 64`. The condition's largest `r·θ` at `n/r = 1` is about `0.0205`
  (`TileLuck`'s at 32 columns `0.0099`), and the stated point holds sampled at `n/r` up to `2^50`.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Pouw.RelativeSecurity

/-- **Luck given the rest is luck.** If a receipt game keeps its luck at `pr` at every value of the rest, over its own
part alone, it keeps it over the whole coin: each round's probabilities are the rest's averages of the own part's. -/
def CondReceiptLuckMarginal : Prop :=
  ∀ {Ω Z O : Type} [Fintype Ω] [Fintype Z] [Fintype O] (e : Ω ≃ Z × O) (pr : ℝ) {n : ℕ} (G : ReceiptGame Ω n),
    G.CondReceiptLuck e pr → G.ReceiptLuck pr

open Classical in
/-- **The grid's moment.** Rows `i` have their own coins in `Rw`, columns `j` theirs in `Cl`. Unit `(i, j)` skips
`w i j ≥ 0`; it is dead with probability at most `q i j` over row `i`'s coins alone (`dead i j` reads `ρ i`), and
otherwise lucky with probability at most `p i j ∈ [0, 1]` over column `j`'s coins, whatever every row's. If every unit
meets `q·e^{nθw} + (1 − q)·(p·e^{rθw} + (1 − p)·e^{−κ·rθw})^{n/r} ≤ 1`, the grid's gain at exchange rate `κ` has
moment at most `1` at rate `θ`. -/
def GridLuck : Prop :=
  ∀ {Rw Cl : Type} [Fintype Rw] [Fintype Cl] {r n : ℕ} (w : Fin r → Fin n → ℝ) (dead : Fin r → Fin n → Rw → Prop)
    (lucky : Fin r → Fin n → (Fin r → Rw) → Cl → Prop) (p q : Fin r → Fin n → ℝ) (θ κ : ℝ),
    0 < n → 0 < r → 0 ≤ θ → 0 ≤ κ → (∀ i j, 0 ≤ w i j) → (∀ i j, 0 ≤ p i j ∧ p i j ≤ 1) →
    (∀ i j ρ, ¬dead i j (ρ i) → prob (lucky i j ρ) ≤ p i j) → (∀ i j, prob (dead i j) ≤ q i j) →
    (∀ i j, q i j * Real.exp (n * θ * w i j) +
        (1 - q i j) * (p i j * Real.exp (r * θ * w i j) + (1 - p i j) * Real.exp (-(κ * (r * θ) * w i j))) ^
          ((n : ℝ) / r) ≤ 1) →
    ∑ ω : (Fin r → Rw) × (Fin n → Cl),
        Real.exp (θ * ∑ i, ∑ j, w i j * ((1 + κ) * (if lucky i j ω.1 (ω.2 j) then 1 else 0) - κ)) ≤
      Fintype.card ((Fin r → Rw) × (Fin n → Cl))

/-- **The grid template's condition.** h1tile's segment at `κ = 4`: a word of `w ≤ 32` skipped products, dead with
probability at most `2^{-w}` and otherwise lucky with probability at most `1/2`, meets `GridLuck`'s condition at
`r·θ = 1/(4·max(64, e))` for every ratio `e = n/r > 0` of a row's units to a column's. -/
def GridTemplateCondition : Prop :=
  ∀ e : ℝ, 0 < e → ∀ w : ℕ, w ≤ 32 →
    (1 / 2 : ℝ) ^ w * Real.exp (e * (1 / (4 * max 64 e)) * w) +
        (1 - (1 / 2 : ℝ) ^ w) *
          (1 / 2 * Real.exp (1 / (4 * max 64 e) * w) + (1 - 1 / 2) * Real.exp (-(4 * (1 / (4 * max 64 e)) * w))) ^ e ≤
      1

open Classical in
/-- **The grid template's moment (its θ).** `GridLuck` at h1tile's segment on a grid of `r` rows and `n` columns: each
unit's word skips `w i j ≤ 32` products, dead with probability at most `2^{-w}` over its row's coins and otherwise lucky
with probability at most `1/2` over its column's. At `κ = 4` the grid's gain has moment at most 1 at rate
`θ = 1/(4·max(64·r, n))` a product, so a seed group's premium is at most `4·max(64·r, n)·ln(T/δ)` products. -/
def GridTemplateMoment : Prop :=
  ∀ {Rw Cl : Type} [Fintype Rw] [Fintype Cl] {r n : ℕ} (w : Fin r → Fin n → ℕ) (dead : Fin r → Fin n → Rw → Prop)
    (lucky : Fin r → Fin n → (Fin r → Rw) → Cl → Prop), 0 < r → 0 < n → (∀ i j, w i j ≤ 32) →
    (∀ i j ρ, ¬dead i j (ρ i) → prob (lucky i j ρ) ≤ 1 / 2) → (∀ i j, prob (dead i j) ≤ (1 / 2) ^ w i j) →
    ∑ ω : (Fin r → Rw) × (Fin n → Cl),
        Real.exp (1 / (4 * max (64 * (r : ℝ)) n) *
          ∑ i, ∑ j, (w i j : ℝ) * ((1 + 4) * (if lucky i j ω.1 (ω.2 j) then 1 else 0) - 4)) ≤
      Fintype.card ((Fin r → Rw) × (Fin n → Cl))

end Accounting.Guarantees.TS
