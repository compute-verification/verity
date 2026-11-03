import Definitions.Accounting.TS.SharedNoise

/-!
# One noise seed a call: the bridge and the grid's moment

J10's guarantees: H1 + fx1's noise derived from one seed a call (`Definitions.Accounting.TS.SharedNoise`).

* `CondReceiptLuckMarginal`: luck given the rest implies the marginal luck. At `CallCoin.keySplit t` it turns the keys'
  argument for B′ into the marginal `ReceiptLuck` that `PerTile … ReceiptLuckAt` reads (`Definitions.Accounting.TS.
  SharedNoise`'s module doc).
* `GridLuck`: `TileLuck` for a grid whose rows' noise is shared along its row and whose columns' along its column. Row
  `i` has its own coins in `Rw` and column `j` its own in `Cl`. Unit `(i, j)` skips `w i j ≥ 0`; it is dead with
  probability at most `q` over row `i`'s coins alone, and otherwise lucky with probability at most `p` over column
  `j`'s coins, given every row's coins and the columns before `j`. A lucky event may read the columns before its own,
  so a skipped cell may guess from a computed neighbour's word. If every unit meets
  `q·e^{nθw} + (1 − q)·(p·e^{rθw} + (1 − p)·e^{−κ·rθw})^{n/r} ≤ 1`, the grid's gain has moment at most `1` at rate
  `θ`. Proof: integrate the columns out from the last. Given the rows and the columns before `j`, AM-GM over column
  `j`'s `r` units, each scaled by its moment `φ_{ij}` at `rθ` (`e^{rθw}` dead, `LuckMoment`'s bound otherwise), gives
  `∏_i φ_{ij}^{1/r}`. That reads row `i`'s coins only, so the rows are independent, and AM-GM over a row's `n` units
  leaves each unit's `E[φ^{n/r}]`, which is the condition. `TileLuck` has `n` where this has `n/r`, because it lets
  dead events of different rows correlate; h1tile's dead events read one row's dither, so they don't. Both hypotheses
  are load-bearing: with every row's dead event on one coin, or a row's lucky events made one event, h1tile's decode
  grid has log-moment 233.8 at the rate below.
* `GridTemplateCondition` and `GridTemplateMoment`: `GridLuck` at h1tile's template (`q = 2^{-w}`, `p = ½`, `κ = 4`,
  `w ≤ 32`) at `r·θ = 1/(4·max(64, n/r))`: `θ = 1/(4·max(64·r, n))` a product. The column streams carry the row
  block, so a call's grid splits into independent blocks of one row block by one band: `r = 32` rows and `n = 32·g`
  columns, `g ≤ 64` tiles a band. There `θ = 1/(256·32)`, `TemplateMoment`'s rate at one tile, for every `g ≤ 64` and
  every number of row blocks: banded, one seed a call costs the seeded premium nothing.
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
`w i j ≥ 0`; it is dead with probability at most `q i j` over row `i`'s coins alone (`dead i j` reads `ρ i`). Its lucky
event reads every row's coins and the columns up to its own (`lucky i j ρ γ` reads `γ k` for `k ≤ j` only), and off
its dead event it has probability at most `p i j ∈ [0, 1]` over column `j`'s coins, whatever the rest. If every unit
meets `q·e^{nθw} + (1 − q)·(p·e^{rθw} + (1 − p)·e^{−κ·rθw})^{n/r} ≤ 1`, the grid's gain at exchange rate `κ` has
moment at most `1` at rate `θ`. The order of the columns is any fixed one: relabel them. -/
def GridLuck : Prop :=
  ∀ {Rw Cl : Type} [Fintype Rw] [Fintype Cl] {r n : ℕ} (w : Fin r → Fin n → ℝ) (dead : Fin r → Fin n → Rw → Prop)
    (lucky : Fin r → Fin n → (Fin r → Rw) → (Fin n → Cl) → Prop) (p q : Fin r → Fin n → ℝ) (θ κ : ℝ),
    0 < n → 0 < r → 0 ≤ θ → 0 ≤ κ → (∀ i j, 0 ≤ w i j) → (∀ i j, 0 ≤ p i j ∧ p i j ≤ 1) →
    (∀ i j ρ γ γ', (∀ k, k ≤ j → γ k = γ' k) → (lucky i j ρ γ ↔ lucky i j ρ γ')) →
    (∀ i j ρ γ, ¬dead i j (ρ i) → prob (fun c => lucky i j ρ (Function.update γ j c)) ≤ p i j) →
    (∀ i j, prob (dead i j) ≤ q i j) →
    (∀ i j, q i j * Real.exp (n * θ * w i j) +
        (1 - q i j) * (p i j * Real.exp (r * θ * w i j) + (1 - p i j) * Real.exp (-(κ * (r * θ) * w i j))) ^
          ((n : ℝ) / r) ≤ 1) →
    ∑ ω : (Fin r → Rw) × (Fin n → Cl),
        Real.exp (θ * ∑ i, ∑ j, w i j * ((1 + κ) * (if lucky i j ω.1 ω.2 then 1 else 0) - κ)) ≤
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
with probability at most `1/2` over its column's, given the rows and the columns before it. At `κ = 4` the grid's gain
has moment at most 1 at rate `θ = 1/(4·max(64·r, n))` a product, so a seed group's premium is at most
`4·max(64·r, n)·ln(T/δ)` products. -/
def GridTemplateMoment : Prop :=
  ∀ {Rw Cl : Type} [Fintype Rw] [Fintype Cl] {r n : ℕ} (w : Fin r → Fin n → ℕ) (dead : Fin r → Fin n → Rw → Prop)
    (lucky : Fin r → Fin n → (Fin r → Rw) → (Fin n → Cl) → Prop), 0 < r → 0 < n → (∀ i j, w i j ≤ 32) →
    (∀ i j ρ γ γ', (∀ k, k ≤ j → γ k = γ' k) → (lucky i j ρ γ ↔ lucky i j ρ γ')) →
    (∀ i j ρ γ, ¬dead i j (ρ i) → prob (fun c => lucky i j ρ (Function.update γ j c)) ≤ 1 / 2) →
    (∀ i j, prob (dead i j) ≤ (1 / 2) ^ w i j) →
    ∑ ω : (Fin r → Rw) × (Fin n → Cl),
        Real.exp (1 / (4 * max (64 * (r : ℝ)) n) *
          ∑ i, ∑ j, (w i j : ℝ) * ((1 + 4) * (if lucky i j ω.1 ω.2 then 1 else 0) - 4)) ≤
      Fintype.card ((Fin r → Rw) × (Fin n → Cl))

end Accounting.Guarantees.TS
