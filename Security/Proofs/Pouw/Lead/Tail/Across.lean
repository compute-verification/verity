import Proofs.Pouw.Lead.Tail.Words
import Proofs.Pouw.Lead.Tail.Fiber

/-!
# Blocks that repeat an earlier tile

`across_tail`: at a call of the lead, `a` or more blocks repeat a tile of an earlier block on at most a `λ_X^a / a!`
share of the coins, `λ_X = ρ² · blocks² / 2^57` (`lamX`).

A block's own coordinates are its step's three coin lanes of its coin tile at its column block's eight columns
(`setLanes`). The lead's columns fill whole column blocks, so every coin lane is one block's, and the coins are the
rest (`b`, `a₁`, `G`), drawn first, and a product over the blocks in block order (`blkIdx`). `count_tail_fiber` then
needs:

* **Reads**: block `β`'s tiles read the lanes of `β`'s coin tile and column block at steps up to `β`'s, and the
  earlier tiles it is compared with read their own blocks' lanes: all blocks at or before `β`.
* **The fiber** (`across_fiber`): with every other coin fixed, `β`'s tile at row block `rb` equals a fixed earlier
  tile only where, at its first row and each of its eight columns, `127 s₀ + s₁ + s₂` hits one value: for each
  `s₁, s₂` at most one `s₀`, a `2^−56` share (`Tags.tile_lane_card_le`'s count, on one fiber). `β` has at most `ρ`
  tiles, and at most `ρ · blkIdx β` tiles come before it.
* **The sum**: `Σ_β ρ² · blkIdx β / 2^56 = ρ² · B (B − 1) / 2^57 ≤ lamX`, `B` the blocks.
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Pouw.Lead

/-- `ω` with block `β`'s own coin lanes set to `v`: step `β.1`'s three lanes of coin tile `β.2.1` at column block
`β.2.2`'s columns, column `8 cb + c` taking `v ℓ c`. -/
def setLanes (L : Pouw.Lead.Layout) (ω : Coins L) (β : Blk L) (v : Fin 3 → Fin 8 → Fin (2 ^ L.e)) : Coins L :=
  (ω.1, ω.2.1, ω.2.2.1, fun t ct ℓ j =>
    if t = β.1 ∧ ct = β.2.1 ∧ (j : ℕ) / 8 = β.2.2 then v ℓ ⟨(j : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩
    else ω.2.2.2 t ct ℓ j)

open Finset

theorem blkIdx_le_of_le (L : Pouw.Lead.Layout) {t t' : ℕ} (ct cb : ℕ) (h : t ≤ t') :
    blkIdx L t ct cb ≤ blkIdx L t' ct cb := by
  unfold blkIdx
  exact Nat.add_le_add_right (Nat.mul_le_mul_right _ (Nat.add_le_add_right (Nat.mul_le_mul_right _ h) _)) _

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

/-- **A block's lanes move only its own coin tile and column block's coin lanes, at its step.** -/
theorem laneTerm_setLanes (L : Pouw.Lead.Layout) (htile : L.tile = 64) (ω : Coins L) (β : Blk L)
    (v w : Fin 3 → Fin 8 → Fin (2 ^ L.e)) (i : Fin L.m) (τ : Fin L.S) (j : Fin L.n) :
    laneTerm L (setLanes L ω β v) i τ j - laneTerm L (setLanes L ω β w) i τ j =
      if τ = β.1 ∧ (i : ℕ) / 64 = β.2.1 ∧ (j : ℕ) / 8 = β.2.2 then
        ∑ ℓ, rotWeight i ℓ * (2 * (((v ℓ ⟨(j : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩ : ℕ) : ℤ) -
          ((w ℓ ⟨(j : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩ : ℕ) : ℤ)))
      else 0 := by
  have hlt := Tags.tile_lt (by omega) i
  unfold laneTerm
  rw [← sum_sub_distrib]
  simp only [dite_eq_left hlt]
  split_ifs with hc
  · refine sum_congr rfl fun ℓ _ => ?_
    have hc' : τ = β.1 ∧ (⟨(i : ℕ) / L.tile, hlt⟩ : Fin L.tiles) = β.2.1 ∧ (j : ℕ) / 8 = β.2.2 :=
      ⟨hc.1, Fin.ext (by simp only [htile]; exact hc.2.1), hc.2.2⟩
    simp only [Coins.s, setLanes, ite_eq_left hc']
    ring
  · refine sum_eq_zero fun ℓ _ => ?_
    have hc' : ¬ (τ = β.1 ∧ (⟨(i : ℕ) / L.tile, hlt⟩ : Fin L.tiles) = β.2.1 ∧ (j : ℕ) / 8 = β.2.2) :=
      fun h => hc ⟨h.1, by rw [← h.2.1]; simp only [htile], h.2.2⟩
    simp only [Coins.s, setLanes, ite_eq_right hc']
    ring

/-- **A block's lanes move the words of its coin tile and column block from its step on.** -/
theorem word_setLanes (L : Pouw.Lead.Layout) (htile : L.tile = 64) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ)
    (ω : Coins L) (β : Blk L) (v w : Fin 3 → Fin 8 → Fin (2 ^ L.e)) (i : Fin L.m) (t : Fin L.S) (j : Fin L.n) :
    word L κ (setLanes L ω β v) i t j - word L κ (setLanes L ω β w) i t j =
      if β.1 ≤ t ∧ (i : ℕ) / 64 = β.2.1 ∧ (j : ℕ) / 8 = β.2.2 then
        ∑ ℓ, rotWeight i ℓ * (2 * (((v ℓ ⟨(j : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩ : ℕ) : ℤ) -
          ((w ℓ ⟨(j : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩ : ℕ) : ℤ)))
      else 0 := by
  rw [Tags.word_sub]
  simp only [Tags.coinTerm_sub_of_tags (setLanes L ω β v) (setLanes L ω β w) rfl rfl rfl,
    laneTerm_setLanes L htile]
  rw [sum_eq_single β.1]
  · by_cases h : β.1 ≤ t <;> by_cases hc : (i : ℕ) / 64 = β.2.1 ∧ (j : ℕ) / 8 = β.2.2 <;> simp [h, hc]
  · intro τ _ hτ
    simp [hτ]
  · simp

/-- **A tile of an earlier block doesn't read a block's lanes.** -/
theorem tileOf_setLanes_of_lt (L : Pouw.Lead.Layout) (htile : L.tile = 64) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ)
    (ω : Coins L) (β : Blk L) (v w : Fin 3 → Fin 8 → Fin (2 ^ L.e)) {rb cb : ℕ} {t : Fin L.S}
    (h : blkIdx L t (rb / 4) cb < blkIdx L β.1 β.2.1 β.2.2) :
    tileOf L κ (setLanes L ω β v) rb cb t = tileOf L κ (setLanes L ω β w) rb cb t := by
  funext p
  have hp : (p : ℕ) < 128 := p.isLt
  unfold tileOf
  split_ifs with hc
  · have d := word_setLanes L htile κ ω β v w ⟨16 * rb + (p : ℕ) / 8, hc.1⟩ t ⟨8 * cb + (p : ℕ) % 8, hc.2⟩
    rw [ite_eq_right ?_] at d
    · linarith
    rintro ⟨h1, h2, h3⟩
    simp only at h2 h3
    have e2 : (β.2.1 : ℕ) = rb / 4 := by omega
    have e3 : (β.2.2 : ℕ) = cb := by omega
    have := blkIdx_le_of_le L (β.2.1 : ℕ) (β.2.2 : ℕ) h1
    rw [e2, e3] at this
    rw [e2, e3] at h
    omega
  · rfl

open Classical in
/-- `across_fiber` on any layout of whole MMA tiles, 64-row coin tiles and 7-bit coin lanes. -/
theorem across_fiber_of (L : Pouw.Lead.Layout) (htile : L.tile = 64) (he : L.e = 7)
    (hLm : L.m = 16 * L.rowBlocks) (hLn : L.n = 8 * L.colBlocks) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ)
    (ω : Coins L) (β : Blk L) :
    ((Finset.univ.filter fun v => acrossAt L κ (setLanes L ω β v) β).card : ℝ) ≤
      (rho L ^ 2 * blkIdx L β.1 β.2.1 β.2.2 : ℕ) / 2 ^ 56 * Fintype.card (Fin 3 → Fin 8 → Fin (2 ^ L.e)) := by
  obtain ⟨t, ct, cb⟩ := β
  have htiles : L.tiles = (L.rowBlocks + 3) / 4 := by
    unfold Layout.tiles; rw [hLm, htile]; omega
  let rbs := (range L.rowBlocks).filter fun rb => rb / 4 = (ct : ℕ)
  let early := (tileIdx L).filter fun x => blkIdx L x.2.2 (x.1 / 4) x.2.1 < blkIdx L t ct cb
  have hcard := card_filter_le_cover (V := Fin 3 → Fin 8 → Fin (2 ^ L.e)) (W := Fin 2 → Fin 8 → Fin (2 ^ L.e))
    (fun v => acrossAt L κ (setLanes L ω (t, ct, cb) v) (t, ct, cb)) (rbs ×ˢ early)
    (fun i v => tileOf L κ (setLanes L ω (t, ct, cb) v) i.1 cb t =
      tileOf L κ (setLanes L ω (t, ct, cb) v) i.2.1 i.2.2.1 i.2.2.2)
    (fun v ⟨rb, hrb, hrb4, x, hx, hlt, heq⟩ =>
      ⟨(rb, x), mem_product.2 ⟨mem_filter.2 ⟨mem_range.2 hrb, hrb4⟩, mem_filter.2 ⟨hx, hlt⟩⟩, heq⟩)
    fun i hi => ⟨fun v k c => v k.succ c, fun v w hv hw hvw => funext fun ℓ => funext fun c => by
      obtain ⟨⟨hrb, hrb4⟩, hx, hlt⟩ := by simpa only [rbs, early, mem_product, mem_filter, mem_range] using hi
      refine Fin.cases ?_ (fun k => congrFun (congrFun hvw k) c) ℓ
      have hcb := cb.isLt
      have hc := c.isLt
      let j : Fin L.n := ⟨8 * cb + c, by omega⟩
      have hj8 : (j : ℕ) / 8 = cb := by simp only [j]; omega
      have hjc : (⟨(j : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩ : Fin 8) = c := Fin.ext (by simp only [j]; omega)
      have hr := Tags.row_lt hrb
      have ev := congrFun hv ⟨(j : ℕ) % 8, Tags.pos_lt j⟩
      have ew := congrFun hw ⟨(j : ℕ) % 8, Tags.pos_lt j⟩
      rw [Tags.tileOf_col κ _ hr t hj8] at ev ew
      have hx' := congrFun (tileOf_setLanes_of_lt L htile κ ω (t, ct, cb) v w hlt) ⟨(j : ℕ) % 8, Tags.pos_lt j⟩
      have d := word_setLanes L htile κ ω (t, ct, cb) v w ⟨16 * i.1, hr⟩ t j
      rw [ite_eq_left ⟨le_rfl, by simp only; omega, hj8⟩, ev, ew, hx', sub_self, hjc] at d
      have h1 := congrFun (congrFun hvw 0) c
      have h2 := congrFun (congrFun hvw 1) c
      simp only [Fin.succ_zero_eq_one, Fin.succ_one_eq_two] at h1 h2
      rw [Fin.sum_univ_three, h1, h2] at d
      have hw0 : rotWeight (16 * i.1) 0 = 127 := by unfold rotWeight; rw [ite_eq_left (by simp; omega)]
      simp only [hw0] at d
      exact Fin.ext (by exact_mod_cast (by linarith : ((v 0 c : ℕ) : ℤ) = ((w 0 c : ℕ) : ℤ)))⟩
  have hrbs : #rbs ≤ rho L := by
    refine le_min ?_ ?_
    · calc #rbs ≤ #((range 4).image fun a => 4 * (ct : ℕ) + a) := card_le_card fun rb hrb => by
            simp only [rbs, mem_filter, mem_range] at hrb
            exact mem_image.2 ⟨rb % 4, mem_range.2 (Nat.mod_lt _ (by norm_num)), by omega⟩
        _ ≤ 4 := card_image_le.trans (by simp)
    · exact (card_filter_le _ _).trans (by simp)
  have hearly : #early ≤ blkIdx L t ct cb * rho L := by
    calc #early ≤ #(range (blkIdx L t ct cb) ×ˢ range (rho L)) := by
          refine card_le_card_of_injOn (fun x => (blkIdx L x.2.2 (x.1 / 4) x.2.1, x.1 % 4)) ?_ ?_
          · intro x hx
            simp only [early, tileIdx, coe_filter, mem_product, mem_range, mem_univ, and_true,
              Set.mem_ofPred_eq] at hx
            simp only [coe_product, coe_range, Set.mem_prod, Set.mem_Iio]
            exact ⟨hx.2, by unfold rho; omega⟩
          · intro x hx y hy hxy
            simp only [early, tileIdx, coe_filter, mem_product, mem_range, mem_univ, and_true,
              Set.mem_ofPred_eq] at hx hy
            simp only [Prod.mk.injEq] at hxy
            obtain ⟨e1, e2, e3⟩ := blkIdx_inj L (by omega) (by omega) hx.1.2 hy.1.2 hxy.1
            exact Prod.ext (by omega) (Prod.ext e3 (Fin.ext e1))
      _ = blkIdx L t ct cb * rho L := by simp
  have hW : Fintype.card (Fin 2 → Fin 8 → Fin (2 ^ L.e)) = (2 ^ 56) ^ 2 := by
    simp only [Fintype.card_fun, Fintype.card_fin, he]; norm_num
  have hV : Fintype.card (Fin 3 → Fin 8 → Fin (2 ^ L.e)) = (2 ^ 56) ^ 3 := by
    simp only [Fintype.card_fun, Fintype.card_fin, he]; norm_num
  rw [hW, card_product] at hcard
  rw [hV]
  calc (#(univ.filter fun v => acrossAt L κ (setLanes L ω (t, ct, cb) v) (t, ct, cb)) : ℝ) ≤
        (rho L * (blkIdx L t ct cb * rho L) * (2 ^ 56) ^ 2 : ℕ) := by
        exact_mod_cast hcard.trans (Nat.mul_le_mul_right _ (Nat.mul_le_mul hrbs hearly))
    _ = (rho L ^ 2 * blkIdx L t ct cb : ℕ) / 2 ^ 56 * (((2 ^ 56) ^ 3 : ℕ) : ℝ) := by push_cast; ring

open Classical in
/-- **A block's own lanes rarely repeat an earlier tile**: with every other coin fixed, block `β` repeats an earlier
tile on at most a `ρ² · blkIdx β / 2^56` share of its own coin lanes' values. -/
theorem across_fiber (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (ω : Coins (lead T m n K)) (β : Blk (lead T m n K)) :
    ((Finset.univ.filter fun v => acrossAt _ κ (setLanes _ ω β v) β).card : ℝ) ≤
      (rho (lead T m n K) ^ 2 * blkIdx (lead T m n K) β.1 β.2.1 β.2.2 : ℕ) / 2 ^ 56 *
        Fintype.card (Fin 3 → Fin 8 → Fin (2 ^ (lead T m n K).e)) :=
  across_fiber_of _ rfl rfl (lead_m T m n K) (lead_n T m n K) κ ω β

/-- **Blocks that repeat an earlier tile are rarely many**: `a` or more of them on at most a `λ_X^a / a!` share of
the coins. -/
theorem across_tail (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (a : ℕ) :
    Pouw.pr (fun ω : Coins (lead T m n K) => a ≤ (across _ κ ω).card) ≤ lamX (lead T m n K) ^ a / a.factorial := by
  sorry

end Pouw.SecurityProofs.Lead.Tail
