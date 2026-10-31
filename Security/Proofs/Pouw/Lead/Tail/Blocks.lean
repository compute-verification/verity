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

theorem mul_add_lt {a b A B : ℕ} (ha : a < A) (hb : b < B) : a * B + b < A * B :=
  calc a * B + b < a * B + B := by omega
    _ = (a + 1) * B := by ring
    _ ≤ A * B := Nat.mul_le_mul_right _ ha

theorem mul_add_inj {C a₁ a₂ b₁ b₂ : ℕ} (hb₁ : b₁ < C) (hb₂ : b₂ < C) (h : a₁ * C + b₁ = a₂ * C + b₂) :
    a₁ = a₂ ∧ b₁ = b₂ := by
  have key : a₁ = a₂ := by
    by_contra hne
    rcases Nat.lt_or_gt_of_ne hne with hl | hl
    · have := Nat.mul_le_mul_right C hl
      rw [Nat.succ_mul] at this
      omega
    · have := Nat.mul_le_mul_right C hl
      rw [Nat.succ_mul] at this
      omega
  subst key
  omega

theorem blkIdx_inj (L : Pouw.Lead.Layout) {t₁ t₂ c₁ c₂ b₁ b₂ : ℕ} (hc₁ : c₁ < L.tiles) (hc₂ : c₂ < L.tiles)
    (hb₁ : b₁ < L.colBlocks) (hb₂ : b₂ < L.colBlocks) (h : blkIdx L t₁ c₁ b₁ = blkIdx L t₂ c₂ b₂) :
    t₁ = t₂ ∧ c₁ = c₂ ∧ b₁ = b₂ := by
  obtain ⟨h1, rfl⟩ := mul_add_inj hb₁ hb₂ h
  obtain ⟨rfl, rfl⟩ := mul_add_inj hc₁ hc₂ h1
  exact ⟨rfl, rfl, rfl⟩

open Classical in
/-- **Repeats count the elements equal to an earlier one**: under an order without ties, the elements less their
distinct values are at most the elements whose value an earlier element has. -/
theorem card_sub_card_image_le {α V : Type} [DecidableEq V] (s : Finset α) (f : α → V) (r : α → ℕ)
    (hr : Set.InjOn r s) :
    s.card - (s.image f).card ≤ (s.filter fun x => ∃ y ∈ s, r y < r x ∧ f y = f x).card := by
  have h1 := Finset.card_filter_add_card_filter_not (s := s) fun x => ∃ y ∈ s, r y < r x ∧ f y = f x
  have h2 : (s.filter fun x => ¬ ∃ y ∈ s, r y < r x ∧ f y = f x).card ≤ (s.image f).card := by
    refine Finset.card_le_card_of_injOn f (fun x hx => Finset.mem_image_of_mem f (Finset.mem_filter.1 hx).1) ?_
    intro x hx x' hx' hf
    rw [Finset.mem_coe, Finset.mem_filter] at hx hx'
    rcases lt_trichotomy (r x) (r x') with h | h | h
    · exact absurd ⟨x, hx.1, h, hf⟩ hx'.2
    · exact hr hx.1 hx'.1 h
    · exact absurd ⟨x', hx'.1, h, hf.symm⟩ hx.2
  omega

/-- **The repeats split over the blocks**: a block counts at most `ρ` repeats when it repeats an earlier tile and at
most `ρ − 1` when only its own tiles agree. -/
theorem repeats_le_blocks (L : Pouw.Lead.Layout) (htile : L.tile = 64) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ)
    (ω : Coins L) : repeats L κ ω ≤ rho L * (across L κ ω).card + (rho L - 1) * (within L κ ω).card := by
  classical
  have hmem : ∀ x : ℕ × ℕ × Fin L.S, x ∈ tileIdx L ↔ x.1 < L.rowBlocks ∧ x.2.1 < L.colBlocks := by
    intro x
    simp only [tileIdx, Finset.mem_product, Finset.mem_range, Finset.mem_univ, and_true]
  have hct : ∀ rb, rb < L.rowBlocks → rb / 4 < L.tiles := by
    intro rb h
    simp only [Layout.rowBlocks, Layout.tiles, htile] at h ⊢
    omega
  let r : ℕ × ℕ × Fin L.S → ℕ := fun x => blkIdx L x.2.2 (x.1 / 4) x.2.1 * 4 + x.1 % 4
  have hinj : Set.InjOn r (tileIdx L) := by
    intro x hx x' hx' h
    rw [Finset.mem_coe, hmem] at hx hx'
    obtain ⟨h1, h2⟩ := mul_add_inj (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) h
    obtain ⟨h3, h4, h5⟩ := blkIdx_inj L (hct _ hx.1) (hct _ hx'.1) hx.2 hx'.2 h1
    obtain ⟨a, b, c⟩ := x
    obtain ⟨a', b', c'⟩ := x'
    simp only at h2 h3 h4 h5
    obtain rfl : a = a' := by omega
    obtain rfl := Fin.ext h3
    rw [h5]
  refine (card_sub_card_image_le (tileIdx L) (fun x => tileOf L κ ω x.1 x.2.1 x.2.2) r hinj).trans ?_
  set D := (tileIdx L).filter fun x => ∃ y ∈ tileIdx L, r y < r x ∧
    tileOf L κ ω y.1 y.2.1 y.2.2 = tileOf L κ ω x.1 x.2.1 x.2.2 with hD
  let inB : Blk L → ℕ × ℕ × Fin L.S → Prop := fun β x => x.1 / 4 = β.2.1 ∧ x.2.1 = β.2.2 ∧ x.2.2 = β.1
  have hcover : D ⊆ Finset.univ.biUnion fun β : Blk L => D.filter (inB β) := by
    intro x hx
    have hxI := (hmem x).1 (Finset.mem_filter.1 hx).1
    exact Finset.mem_biUnion.2 ⟨(x.2.2, ⟨x.1 / 4, hct _ hxI.1⟩, ⟨x.2.1, hxI.2⟩), Finset.mem_univ _,
      Finset.mem_filter.2 ⟨hx, rfl, rfl, rfl⟩⟩
  have hblock : ∀ β : Blk L, (D.filter (inB β)).card ≤
      (if acrossAt L κ ω β then rho L else 0) + (if withinAt L κ ω β then rho L - 1 else 0) := by
    rintro ⟨t, ct, cb⟩
    have hsub : ∀ lo, (∀ x ∈ D.filter (inB (t, ct, cb)), 4 * (ct : ℕ) + lo ≤ x.1) →
        (D.filter (inB (t, ct, cb))).card ≤ min (4 * (ct : ℕ) + 4) L.rowBlocks - (4 * ct + lo) := by
      intro lo hlo
      calc (D.filter (inB (t, ct, cb))).card
          ≤ ((Finset.Ico (4 * (ct : ℕ) + lo) (min (4 * (ct : ℕ) + 4) L.rowBlocks)).image
              fun rb => (rb, (cb : ℕ), t)).card := by
            refine Finset.card_le_card fun x hx => ?_
            have hlx := hlo x hx
            obtain ⟨hxD, h1, h2, h3⟩ : x ∈ D ∧ x.1 / 4 = ct ∧ x.2.1 = cb ∧ x.2.2 = t := Finset.mem_filter.1 hx
            have hxI := (hmem x).1 (Finset.mem_filter.1 hxD).1
            refine Finset.mem_image.2 ⟨x.1, Finset.mem_Ico.2 ⟨hlx, lt_min (by omega) hxI.1⟩, ?_⟩
            rw [← h2, ← h3]
        _ ≤ (Finset.Ico (4 * (ct : ℕ) + lo) (min (4 * (ct : ℕ) + 4) L.rowBlocks)).card := Finset.card_image_le
        _ = _ := Nat.card_Ico _ _
    have hcase : ∀ x ∈ D.filter (inB (t, ct, cb)),
        acrossAt L κ ω (t, ct, cb) ∨ (withinAt L κ ω (t, ct, cb) ∧ 0 < x.1 % 4) := by
      intro x hx
      obtain ⟨hxD, h1, h2, h3⟩ : x ∈ D ∧ x.1 / 4 = ct ∧ x.2.1 = cb ∧ x.2.2 = t := Finset.mem_filter.1 hx
      obtain ⟨hxT, y, hyT, hry, hfy⟩ := Finset.mem_filter.1 hxD
      have hxI := (hmem x).1 hxT
      have hyI := (hmem y).1 hyT
      have hm4 : y.1 % 4 < 4 := Nat.mod_lt _ (by norm_num)
      have hm4' : x.1 % 4 < 4 := Nat.mod_lt _ (by norm_num)
      have hbx : blkIdx L x.2.2 (x.1 / 4) x.2.1 = blkIdx L t ct cb := by rw [h1, h2, h3]
      simp only [r, hbx] at hry
      rcases Nat.lt_or_ge (blkIdx L y.2.2 (y.1 / 4) y.2.1) (blkIdx L t ct cb) with hlt | hge
      · left
        refine ⟨x.1, hxI.1, h1, y, hyT, hlt, ?_⟩
        show tileOf L κ ω x.1 cb t = tileOf L κ ω y.1 y.2.1 y.2.2
        rw [hfy, h2, h3]
      · have heq : blkIdx L y.2.2 (y.1 / 4) y.2.1 = blkIdx L t ct cb := by omega
        obtain ⟨e1, e2, e3⟩ := blkIdx_inj L (hct _ hyI.1) ct.isLt hyI.2 cb.isLt heq
        have hlt : y.1 % 4 < x.1 % 4 := by omega
        refine Or.inr ⟨⟨x.1, hxI.1, y.1, hyI.1, h1, e2, fun h => by rw [h] at hlt; omega, ?_⟩, by omega⟩
        show tileOf L κ ω x.1 cb t = tileOf L κ ω y.1 cb t
        rw [show tileOf L κ ω y.1 cb t = tileOf L κ ω y.1 y.2.1 y.2.2 by rw [e3, Fin.ext e1], hfy, h2, h3]
    by_cases ha : acrossAt L κ ω (t, ct, cb)
    · have := hsub 0 fun x hx => by have : x.1 / 4 = ct := (Finset.mem_filter.1 hx).2.1; omega
      simp only [ha, ↓reduceIte, rho]
      omega
    by_cases hw : withinAt L κ ω (t, ct, cb)
    · have := hsub 1 fun x hx => by
        have : x.1 / 4 = ct := (Finset.mem_filter.1 hx).2.1
        have := ((hcase x hx).resolve_left ha).2
        omega
      simp only [ha, hw, ↓reduceIte, rho]
      omega
    · have : D.filter (inB (t, ct, cb)) = ∅ :=
        Finset.eq_empty_of_forall_notMem fun x hx => by
          rcases hcase x hx with h | h
          · exact ha h
          · exact hw h.1
      rw [this, Finset.card_empty]
      exact Nat.zero_le _
  calc D.card ≤ (Finset.univ.biUnion fun β : Blk L => D.filter (inB β)).card := Finset.card_le_card hcover
    _ ≤ ∑ β, (D.filter (inB β)).card := Finset.card_biUnion_le
    _ ≤ ∑ β : Blk L, ((if acrossAt L κ ω β then rho L else 0) + (if withinAt L κ ω β then rho L - 1 else 0)) :=
        Finset.sum_le_sum fun β _ => hblock β
    _ = rho L * (across L κ ω).card + (rho L - 1) * (within L κ ω).card := by
        rw [Finset.sum_add_distrib, ← Finset.sum_filter, ← Finset.sum_filter, Finset.sum_const, Finset.sum_const,
          smul_eq_mul, smul_eq_mul, mul_comm, mul_comm (Finset.card _)]
        rfl

end Pouw.SecurityProofs.Lead.Tail
