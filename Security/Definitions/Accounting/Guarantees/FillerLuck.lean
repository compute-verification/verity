import Definitions.Accounting.Filler
import Definitions.Accounting.Assumptions.TS

/-!
# The filler's per-check luck (trusted)

Lemma P2 over `Accounting.Filler`, each `def G : Prop` proved as `Accounting.SecurityProofs.FillerLuck.G` in
`Proofs.Accounting.FillerLuck`:

* `ParityLuck`: Lemma P2 at a cell, in any word ring read mod 2 and any column of uniform parity: a value with `R`
  two-sided products equals the cell's word with probability at most `luck (K − R) = ½ + 2^(−(K−R)/2−1)`.
* `FillerCellLuck`: the filler's own cell check, a cell of a tile with every other row's and column's noise its own,
  at pfill-bits and pfill-bU, in the s32 word (`ZMod (2 ^ 32)`) and over `ℤ`.
* `FillerParityBalanced`: those four columns have uniform parity, so the statements below, stated at any balanced
  column, hold at each of them.
* `FillerLuckAt`: a full-checking filler window meets (I7) `LuckAt` at `luck 1 ≤ 0.854`.
* `FillerBudget`: a spot-checked filler window meets `BudgetAt (1/(1 − 0.854))`, Lemma C's budget at the filler's luck.
* `FillerBlocksDeficit`: Theorem S with each block at its own `p`, the budget `eps.py`'s cheater LP reads: beside any
  block whose one-draw check passes with probability at most `1 − (1 − p₀)·d₀`, outside probability `δ` an accepted
  window has `Σ_w ((1 − p₀)·d₀ + (1 − 0.854)·d_fill) < ln(1/δ)`.
* `PinnedFillLuckAt`, `PinnedFillBudget`, `PinnedFillBlocksDeficit`: the same three window statements at `p = 0` for a
  filler whose check recomputes a pinned receipt from committed data and live coins (lfill's tile receipt, a lane chain
  absorbed into the fold's receipt, keyed rounds in the fold, an FFMA chain recomputed bit for bit). Nothing in such a
  check is left to guess, so a wrong receipt fails whenever its tile is drawn; what a right receipt cost is FILL-OPT's
  (FOLD-OPT's for rounds in the fold).

Probability statements put acceptance in the event: the probability that the window is accepted and the bound fails is
at most `δ`.
-/

namespace Accounting.Guarantees.FillerLuck

open Accounting.Filler Accounting.TS Pouw.RelativeSecurity

/-- **Lemma P2 at a cell**. Let the row's bits `x ∈ {0,1}^K` be uniform and the column's noise `y` uniform on `Y`, with
its entries' parity uniform. A value with `R` two-sided products over `A`, its factors any functions of their side,
equals the cell's word `Σ_k x_k·c_k(y)` with probability at most `luck (K − R) = ½ + 2^(−(K−R)/2−1)`, at every `K`. -/
def ParityLuck : Prop :=
  ∀ {Y A : Type} [Fintype Y] [CommRing A] (φ : A →+* ZMod 2) {K : ℕ} (c : Y → Fin K → A) (R : ℕ)
    (P : (Fin K → ZMod 2) → Y → A),
    ParityBalanced φ c → Products R P →
      prob (fun xy : (Fin K → ZMod 2) × Y => P xy.1 xy.2 = cellWord c xy.1 xy.2) ≤ luck (K - R)

/-- **The filler's cell check**. On a tile of rows `ρ` and columns `ν` with the round's noise uniform, a cell `(r, s)`
whose run word takes `R` two-sided products in the cell projection is right with probability at most `luck (K − R)`: at
pfill-bits (`popcount(a & b)`, B on `{0,1}`) and at pfill-bU (`A·u`, B's entries `uOf` of uniform bytes), in the s32
word and over `ℤ`. -/
def FillerCellLuck : Prop :=
  ∀ (K : ℕ) {ρ ν : Type} [Fintype ρ] [Fintype ν] [DecidableEq ρ] [DecidableEq ν] (r : ρ) (s : ν) (R : ℕ),
    (∀ W : Noise K ρ ν (Fin K → ZMod 2) → ZMod (2 ^ 32), CellProducts W r s R →
      prob (fun ω => W ω = cellWord bitsCol (ω.1 r) (ω.2 s)) ≤ luck (K - R)) ∧
    (∀ W : Noise K ρ ν (Fin K → ZMod 2) → ℤ, CellProducts W r s R →
      prob (fun ω => W ω = cellWord bitsCol (ω.1 r) (ω.2 s)) ≤ luck (K - R)) ∧
    (∀ W : Noise K ρ ν (Fin K → Fin 256) → ZMod (2 ^ 32), CellProducts W r s R →
      prob (fun ω => W ω = cellWord buCol (ω.1 r) (ω.2 s)) ≤ luck (K - R)) ∧
    (∀ W : Noise K ρ ν (Fin K → Fin 256) → ℤ, CellProducts W r s R →
      prob (fun ω => W ω = cellWord buCol (ω.1 r) (ω.2 s)) ≤ luck (K - R))

/-- **The filler's columns have uniform parity**: pfill-bits' (its bits) and pfill-bU's (`uOf t ≡ t₀ ⊕ t₆`), in the s32
word and over `ℤ`, at every `K`. -/
def FillerParityBalanced : Prop :=
  ∀ K : ℕ, ParityBalanced toF2 (bitsCol (K := K)) ∧ ParityBalanced (Int.castRingHom (ZMod 2)) (bitsCol (K := K)) ∧
    ParityBalanced toF2 (buCol (K := K)) ∧ ParityBalanced (Int.castRingHom (ZMod 2)) (buCol (K := K))

/-- **A full-checking filler window meets (I7) at `luck 1`**: when every round checks every cell of the tile and the
column's parity is uniform, the window's game is one bet a round, a round with a short cell passing with probability
at most `luck 1 = ½ + 2^(−3/2) ≤ 0.854` given the past. -/
def FillerLuckAt : Prop :=
  ∀ {K : ℕ} {ρ ν Y A : Type} [Fintype ρ] [Fintype ν] [DecidableEq ρ] [DecidableEq ν] [Fintype Y] [CommRing A]
    (φ : A →+* ZMod 2) (c : Y → Fin K → A) {S : ℕ} (g : RoundGame (Noise K ρ ν Y) S),
    ParityBalanced φ c → FullFill c g → Accounting.Assumptions.TS.LuckAt (luck 1) g

/-- **A spot-checked filler window meets Lemma C's budget at the filler's luck**: when every round checks one uniform
cell and the column's parity is uniform, `Λ = 1/(1 − 0.854)` bounds the round's saving per unit of detection
(`RoundGame.BudgetAt`), so the window's short share is Theorem S's `d̄ = ln(1/δ)/((1 − 0.854)·S)`. -/
def FillerBudget : Prop :=
  ∀ {K : ℕ} {ρ ν Y A : Type} [Fintype ρ] [Fintype ν] [DecidableEq ρ] [DecidableEq ν] [Fintype Y] [CommRing A]
    (φ : A →+* ZMod 2) (c : Y → Fin K → A) {S : ℕ} (g : RoundGame (Noise K ρ ν Y × (ρ × ν)) S),
    ParityBalanced φ c → SpotFill c g → RoundGame.BudgetAt (1 / (1 - 427 / 500)) g

/-- **Theorem S with each block at its own `p`** (the budget row of `eps.py`'s cheater LP). Each round checks another
block on its own coins, its share `d₀` fixed before them and passing with probability at most `1 − (1 − p₀)·d₀`
(`p₀ ≤ 1`), and one uniform cell of the filler's tile, its column's parity uniform. Outside probability `δ`, an
accepted window has `Σ_w ((1 − p₀)·d₀_w + (1 − 0.854)·d_fill,w) < ln(1/δ)`. -/
def FillerBlocksDeficit : Prop :=
  ∀ {K : ℕ} {ρ ν Y A Ω₀ : Type} [Fintype ρ] [Fintype ν] [DecidableEq ρ] [DecidableEq ν] [Fintype Y] [CommRing A]
    [Fintype Ω₀] (φ : A →+* ZMod 2) (c : Y → Fin K → A) (p₀ δ : ℝ) {S : ℕ}
    (g : RoundGame (Ω₀ × (Noise K ρ ν Y × (ρ × ν))) S),
    ParityBalanced φ c → p₀ ≤ 1 → 0 < δ → FillBlocks p₀ c g →
      prob (fun x => g.accepts x ∧ Real.log (1 / δ) ≤ g.deficiency x) ≤ δ

/-- **A full-checking window of pinned receipts meets (I7) at `0`**: when every round recomputes every tile's receipt
from the round's committed data and live coins and compares it with the committed one, a round with a wrong receipt
never passes, whatever the receipt function. -/
def PinnedFillLuckAt : Prop :=
  ∀ {D C τ V : Type} [Fintype C] [Fintype τ] (F : D → C → τ → V) {S : ℕ} (g : RoundGame C S),
    FullPinnedFill F g → Accounting.Assumptions.TS.LuckAt 0 g

/-- **A spot-checked window of pinned receipts meets Lemma C's budget at `p = 0`**: when every round recomputes one
uniform tile's receipt, drawn after the receipts are committed, `Λ = 1` bounds the round's saving per unit of detection
(`RoundGame.BudgetAt`), so the window's wrong share is Theorem S's `d̄ = ln(1/δ)/S`. -/
def PinnedFillBudget : Prop :=
  ∀ {D C τ V : Type} [Fintype C] [Fintype τ] (F : D → C → τ → V) {S : ℕ} (g : RoundGame (C × τ) S),
    SpotPinnedFill F g → RoundGame.BudgetAt 1 g

/-- **Theorem S with another block beside pinned receipts** (the budget row of `eps.py`'s cheater LP at the filler's
`p = 0`). Each round checks another block on its own coins, its share `d₀` fixed before them and passing with
probability at most `1 − (1 − p₀)·d₀` (`p₀ ≤ 1`), and recomputes one uniform tile's receipt. Outside probability `δ`,
an accepted window has `Σ_w ((1 − p₀)·d₀_w + wrongShare_w) < ln(1/δ)`. -/
def PinnedFillBlocksDeficit : Prop :=
  ∀ {D C τ V Ω₀ : Type} [Fintype C] [Fintype τ] [Fintype Ω₀] (F : D → C → τ → V) (p₀ δ : ℝ) {S : ℕ}
    (g : RoundGame (Ω₀ × (C × τ)) S),
    p₀ ≤ 1 → 0 < δ → PinnedFillBlocks p₀ F g →
      prob (fun x => g.accepts x ∧ Real.log (1 / δ) ≤ g.deficiency x) ≤ δ

end Accounting.Guarantees.FillerLuck
