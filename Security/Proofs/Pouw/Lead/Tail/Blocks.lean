import Mathlib.Algebra.Order.Floor.Semiring
import Mathlib.Data.Fintype.Prod
import Definitions.Pouw.Lead.Gap

/-!
# The coin-branch tail's blocks

A **block** is a step, coin tile and column block (`Blk`): the at most `ρ = min 4 rowBlocks` tiles of one step and
column block whose rows share a coin tile (`rho`), ordered by step, coin tile and column block (`blkIdx`). The call's
repeated tiles split over two counts (`repeats_le_blocks`):

* `across`: the blocks with a tile equal to a tile of an earlier block. A block's own coin lanes are read by no
  earlier block (`Tail.across_tail`).
* `within`: the blocks with two equal tiles of their own. Rows of one parity in one coin tile carry the same coin
  lanes, so only the tag weights `G`, the step-0 bits `a₁` and the tag bits `b` separate them: `G` bounds the column
  blocks with any (`hitCols`, `Tail.columns_tail`), and `a₁` and `b` each column block's slots with one (`hitSlots`,
  `Tail.slots_tail`).

A block's tiles count at most `ρ` repeats when it repeats an earlier tile and at most `ρ − 1` when two of its own
agree, so `repeats ≤ ρ · across + (ρ − 1) · within`; and `within` is at most the column blocks with an agreement times
the most slots one of them has (`within_le`). `lamX`, `lamG` and `lamB` are the three counts' Poisson parameters, and
`cut` a count's bound with the counts it can't reach at 0.
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Pouw.Lead

/-- Row blocks a coin tile holds at most: 64 rows a coin tile, 16 a row block, and no more than the call has. -/
def rho (L : Pouw.Lead.Layout) : ℕ := min 4 L.rowBlocks

/-- Pairs of row blocks in one coin tile: `ρ (ρ − 1) / 2`, 6 at the three points and 0 at decode. -/
def pairs (L : Pouw.Lead.Layout) : ℕ := rho L * (rho L - 1) / 2

/-- A block: a step, coin tile and column block. -/
abbrev Blk (L : Pouw.Lead.Layout) : Type := Fin L.S × Fin L.tiles × Fin L.colBlocks

/-- The blocks a call: `S · tiles · colBlocks`. -/
def blkCount (L : Pouw.Lead.Layout) : ℕ := L.S * L.tiles * L.colBlocks

/-- A block's place in the order by step, then coin tile, then column block. -/
def blkIdx (L : Pouw.Lead.Layout) (t ct cb : ℕ) : ℕ := (t * L.tiles + ct) * L.colBlocks + cb

/-- **Block `β` repeats an earlier tile**: a tile of `β` equals a committed tile of a block before it. -/
def acrossAt (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) (β : Blk L) : Prop :=
  ∃ rb < L.rowBlocks, rb / 4 = (β.2.1 : ℕ) ∧ ∃ x ∈ tileIdx L,
    blkIdx L x.2.2 (x.1 / 4) x.2.1 < blkIdx L β.1 β.2.1 β.2.2 ∧ tileOf L κ ω rb β.2.2 β.1 = tileOf L κ ω x.1 x.2.1 x.2.2

/-- **Block `β` repeats a tile of its own**: two of its tiles, at different row blocks, are equal. -/
def withinAt (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) (β : Blk L) : Prop :=
  ∃ rb < L.rowBlocks, ∃ rb' < L.rowBlocks, rb / 4 = (β.2.1 : ℕ) ∧ rb' / 4 = (β.2.1 : ℕ) ∧ rb ≠ rb' ∧
    tileOf L κ ω rb β.2.2 β.1 = tileOf L κ ω rb' β.2.2 β.1

open Classical in
/-- The blocks that repeat an earlier tile. -/
noncomputable def across (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) : Finset (Blk L) :=
  Finset.univ.filter (acrossAt L κ ω)

open Classical in
/-- The blocks that repeat a tile of their own. -/
noncomputable def within (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) : Finset (Blk L) :=
  Finset.univ.filter (withinAt L κ ω)

open Classical in
/-- The column blocks with a block that repeats a tile of its own. -/
noncomputable def hitCols (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) :
    Finset (Fin L.colBlocks) :=
  Finset.univ.filter fun cb => ∃ t ct, withinAt L κ ω (t, ct, cb)

open Classical in
/-- Column block `cb`'s slots (step, coin tile) whose block repeats a tile of its own. -/
noncomputable def hitSlots (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L)
    (cb : Fin L.colBlocks) : Finset (Fin L.S × Fin L.tiles) :=
  Finset.univ.filter fun s => withinAt L κ ω (s.1, s.2, cb)

/-- `across`'s Poisson parameter: each of a block's `ρ` tiles against each earlier tile at `2^−56`, summed over the
blocks, at most `ρ² · blocks² / 2^57`. -/
noncomputable def lamX (L : Pouw.Lead.Layout) : ℝ := (rho L ^ 2 * blkCount L ^ 2 : ℕ) / 2 ^ 57

/-- `hitCols`' Poisson parameter: each column block's `pairs · S · tiles` pairs of tiles at `7^−8`. -/
noncomputable def lamG (L : Pouw.Lead.Layout) : ℝ := (pairs L * L.colBlocks * L.S * L.tiles : ℕ) / 7 ^ 8

/-- `hitSlots`' Poisson parameter: each of a column block's `S · tiles` slots, its `pairs` pairs at `2^−16`. -/
noncomputable def lamB (L : Pouw.Lead.Layout) : ℝ := (pairs L * L.S * L.tiles : ℕ) / 2 ^ 16

/-- **A count's Poisson bound, cut at its index set's size**: `λ^r / r!`, and 0 when `r` exceeds `size`, the most the
count can be. -/
noncomputable def cut (size : ℕ) (l : ℝ) (r : ℕ) : ℝ := if size < r then 0 else l ^ r / r.factorial

/-- **The least repeats in the coin-branch tail**: `tileWords · repeats` exceeds the mean and slack from here on. -/
noncomputable def tailCount (L : Pouw.Lead.Layout) : ℕ := ⌊(coinBranchMean L + coinSlack L) / tileWords⌋₊ + 1

theorem tailCount_le {L : Pouw.Lead.Layout} {r : ℕ} (h : coinBranchMean L + coinSlack L < tileWords * (r : ℚ)) :
    tailCount L ≤ r := by
  have h0 : 0 ≤ (coinBranchMean L + coinSlack L) / tileWords := by
    unfold coinBranchMean coinSlack stockRegs tileWords
    positivity
  have : (coinBranchMean L + coinSlack L) / tileWords < r := by
    rw [div_lt_iff₀ (by unfold tileWords; norm_num)]
    linarith
  exact Nat.floor_lt h0 |>.2 this

/-- **The within-block repeats sit in the hit column blocks**: if no column block has `x` slots that repeat, the
blocks that repeat a tile of their own are at most the hit column blocks times `x − 1`. -/
theorem within_le (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) {x : ℕ}
    (h : ∀ cb, (hitSlots L κ ω cb).card ≤ x - 1) : (within L κ ω).card ≤ (hitCols L κ ω).card * (x - 1) := by
  classical
  have hsum : (within L κ ω).card = ∑ cb, (hitSlots L κ ω cb).card := by
    rw [Finset.card_eq_sum_card_fiberwise (f := fun β : Blk L => β.2.2) (t := Finset.univ)
      fun _ _ => Finset.mem_univ _]
    refine Finset.sum_congr rfl fun cb _ => Finset.card_nbij' (fun β => (β.1, β.2.1)) (fun s => (s.1, s.2, cb))
      ?_ ?_ ?_ fun _ _ => rfl
    · intro β hβ
      simp only [Finset.coe_filter, within, hitSlots, Finset.mem_filter, Finset.mem_univ, true_and,
        Set.mem_ofPred_eq] at hβ ⊢
      obtain ⟨h1, rfl⟩ := hβ
      exact h1
    · intro s hs
      simp only [Finset.coe_filter, within, hitSlots, Finset.mem_filter, Finset.mem_univ, true_and,
        Set.mem_ofPred_eq] at hs ⊢
      exact ⟨hs, trivial⟩
    · intro β hβ
      simp only [Finset.coe_filter, within, Finset.mem_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq] at hβ
      obtain ⟨-, rfl⟩ := hβ
      rfl
  rw [hsum, ← Finset.sum_filter_add_sum_filter_not Finset.univ fun cb => ∃ t ct, withinAt L κ ω (t, ct, cb)]
  have hz : ∀ cb ∈ Finset.univ.filter (fun cb => ¬ ∃ t ct, withinAt L κ ω (t, ct, cb)),
      (hitSlots L κ ω cb).card = 0 := by
    intro cb hcb
    unfold hitSlots
    rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
    exact fun s _ hs => (Finset.mem_filter.1 hcb).2 ⟨s.1, s.2, hs⟩
  rw [Finset.sum_congr rfl hz, Finset.sum_const_zero, add_zero]
  exact (Finset.sum_le_card_nsmul _ _ _ fun cb _ => h cb).trans_eq (smul_eq_mul _ _)

/-- **The repeats split over the blocks**: a block counts at most `ρ` repeats when it repeats an earlier tile and at
most `ρ − 1` when only its own tiles agree. -/
theorem repeats_le_blocks (L : Pouw.Lead.Layout) (htile : L.tile = 64) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ)
    (ω : Coins L) : repeats L κ ω ≤ rho L * (across L κ ω).card + (rho L - 1) * (within L κ ω).card := by
  sorry

end Pouw.SecurityProofs.Lead.Tail
