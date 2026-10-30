import Proofs.Pouw.Lead.Tail.Words
import Proofs.Pouw.Lead.Tail.Fiber
import Mathlib.Algebra.BigOperators.Field

/-!
# Blocks that repeat an earlier tile

`across_tail`: at a call of the lead, `a` or more blocks repeat a tile of an earlier block on at most a `λ_X^a / a!`
share of the coins, `λ_X = ρ² · blocks² / 2^57` (`lamX`).

A block's own coordinates are its step's three coin lanes of its coin tile at its column block's eight columns
(`setLanes`). The lead's columns fill whole column blocks, so every coin lane is one block's, and the coins are the
rest (`b`, `a₁`, `G`), drawn first, and a product over the blocks in block order (`blkIdx`, `laneEquiv`).
`count_tail_equiv` then needs:

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

theorem blkIdx_lt (L : Pouw.Lead.Layout) (β : Blk L) : blkIdx L β.1 β.2.1 β.2.2 < blkCount L :=
  mul_add_lt (mul_add_lt β.1.isLt β.2.1.isLt) β.2.2.isLt

/-- The blocks in block order. -/
noncomputable def blkEquiv (L : Pouw.Lead.Layout) : Blk L ≃ Fin (blkCount L) :=
  Equiv.ofBijective (fun β => ⟨blkIdx L β.1 β.2.1 β.2.2, blkIdx_lt L β⟩)
    ((Fintype.bijective_iff_injective_and_card _).2 ⟨fun β β' h => by
      obtain ⟨e1, e2, e3⟩ := blkIdx_inj L β.2.1.isLt β'.2.1.isLt β.2.2.isLt β'.2.2.isLt (congrArg Fin.val h)
      exact Prod.ext (Fin.ext e1) (Prod.ext (Fin.ext e2) (Fin.ext e3)),
      by simp only [Fintype.card_prod, Fintype.card_fin, blkCount, mul_assoc]⟩)

theorem blkEquiv_val (L : Pouw.Lead.Layout) (β : Blk L) : (blkEquiv L β : ℕ) = blkIdx L β.1 β.2.1 β.2.2 := rfl

/-- **The coins are the rest and each block's own lanes**, the blocks in block order. -/
noncomputable def laneEquiv (L : Pouw.Lead.Layout) (hLn : L.n = 8 * L.colBlocks) :
    Coins L ≃ ((Fin L.m → Fin L.S → Fin 2) × (Fin L.m → Fin 4 → Fin 2) × (Fin 4 → Fin L.n → Fin 7)) ×
      (Fin (blkCount L) → Fin 3 → Fin 8 → Fin (2 ^ L.e)) where
  toFun ω := ((ω.1, ω.2.1, ω.2.2.1), fun i ℓ c =>
    ω.2.2.2 ((blkEquiv L).symm i).1 ((blkEquiv L).symm i).2.1 ℓ (colIdx L hLn ((blkEquiv L).symm i).2.2 c))
  invFun x := (x.1.1, x.1.2.1, x.1.2.2, fun t ct ℓ j =>
    x.2 (blkEquiv L (t, ct, colBlk L hLn j)) ℓ ⟨(j : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩)
  left_inv ω := by
    refine Prod.ext rfl (Prod.ext rfl (Prod.ext rfl (funext fun t => funext fun ct => funext fun ℓ => funext fun j => ?_)))
    simp only [Equiv.symm_apply_apply]
    exact congrArg (ω.2.2.2 t ct ℓ) (Fin.ext (by simp only [colIdx, colBlk]; omega))
  right_inv x := by
    refine Prod.ext rfl (funext fun i => funext fun ℓ => funext fun c => ?_)
    have h1 : colBlk L hLn (colIdx L hLn ((blkEquiv L).symm i).2.2 c) = ((blkEquiv L).symm i).2.2 :=
      Fin.ext (by simp only [colIdx, colBlk]; omega)
    have h2 : (⟨((colIdx L hLn ((blkEquiv L).symm i).2.2 c : Fin L.n) : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩ : Fin 8) =
        c := Fin.ext (by simp only [colIdx]; omega)
    simp only [h1, h2, Prod.mk.eta, Equiv.apply_symm_apply]

theorem laneEquiv_lane (L : Pouw.Lead.Layout) (hLn : L.n = 8 * L.colBlocks) (ω : Coins L) (t : Fin L.S)
    (ct : Fin L.tiles) (ℓ : Fin 3) (j : Fin L.n) :
    (laneEquiv L hLn ω).2 (blkEquiv L (t, ct, colBlk L hLn j)) ℓ ⟨(j : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩ =
      ω.2.2.2 t ct ℓ j := by
  simp only [laneEquiv, Equiv.coe_fn_mk, Equiv.symm_apply_apply]
  exact congrArg (ω.2.2.2 t ct ℓ) (Fin.ext (by simp only [colIdx, colBlk]; omega))

/-- Moving one block's coordinate is `setLanes`. -/
theorem laneEquiv_update (L : Pouw.Lead.Layout) (hLn : L.n = 8 * L.colBlocks) (ω : Coins L) (i : Fin (blkCount L))
    (v : Fin 3 → Fin 8 → Fin (2 ^ L.e)) :
    (laneEquiv L hLn).symm ((laneEquiv L hLn ω).1, Function.update (laneEquiv L hLn ω).2 i v) =
      setLanes L ω ((blkEquiv L).symm i) v := by
  refine Prod.ext rfl (Prod.ext rfl (Prod.ext rfl (funext fun t => funext fun ct => funext fun ℓ => funext fun j => ?_)))
  simp only [laneEquiv, Equiv.coe_fn_mk, Equiv.coe_fn_symm_mk, setLanes]
  by_cases hk : blkEquiv L (t, ct, colBlk L hLn j) = i
  · have hβ : (t, ct, colBlk L hLn j) = (blkEquiv L).symm i := by rw [← hk, Equiv.symm_apply_apply]
    rw [hk, Function.update_self, ite_eq_left ⟨by rw [← hβ], by rw [← hβ], by rw [← hβ]; rfl⟩]
  · rw [Function.update_of_ne hk]
    have hn : ¬ (t = ((blkEquiv L).symm i).1 ∧ ct = ((blkEquiv L).symm i).2.1 ∧
        (j : ℕ) / 8 = ((blkEquiv L).symm i).2.2) := by
      rintro ⟨h1, h2, h3⟩
      refine hk ((Equiv.eq_symm_apply _).1 ?_)
      exact Prod.ext h1 (Prod.ext h2 (Fin.ext h3))
    rw [ite_eq_right hn, Equiv.symm_apply_apply]
    exact congrArg (ω.2.2.2 t ct ℓ) (Fin.ext (by simp only [colIdx, colBlk]; omega))

/-- `across_tail` on any layout of whole MMA tiles, 64-row coin tiles and 7-bit coin lanes. -/
theorem across_tail_of (L : Pouw.Lead.Layout) (htile : L.tile = 64) (he : L.e = 7) (hLm : L.m = 16 * L.rowBlocks)
    (hLn : L.n = 8 * L.colBlocks) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (a : ℕ) :
    Pouw.pr (fun ω : Coins L => a ≤ (across L κ ω).card) ≤ lamX L ^ a / a.factorial := by
  classical
  let e := laneEquiv L hLn
  let p : Fin (blkCount L) → ℝ := fun i => (rho L ^ 2 * (i : ℕ) : ℕ) / 2 ^ 56
  have hcount : ∀ ω, (across L κ ω).card = (univ.filter fun i => acrossAt L κ ω ((blkEquiv L).symm i)).card :=
    fun ω => card_equiv (blkEquiv L) fun β => by simp only [across, mem_filter, mem_univ, true_and,
      Equiv.symm_apply_apply]
  have htail := count_tail_equiv e (fun i ω => acrossAt L κ ω ((blkEquiv L).symm i)) p (fun i => by positivity)
    (fun i ω ω' hr hl => ?_) (fun i ω => ?_) a
  · have hsum : ∑ i, p i ≤ lamX L := by
      have hg : (∑ i : Fin (blkCount L), (i : ℕ)) * 2 ≤ blkCount L ^ 2 := by
        rw [Fin.sum_univ_eq_sum_range (fun i => i) (blkCount L), Finset.sum_range_id_mul_two, sq]
        exact Nat.mul_le_mul_left _ (Nat.sub_le _ _)
      have hg' : ((rho L ^ 2 * ∑ i : Fin (blkCount L), (i : ℕ) : ℕ) : ℝ) * 2 ≤ (rho L ^ 2 * blkCount L ^ 2 : ℕ) := by
        exact_mod_cast (show (rho L ^ 2 * ∑ i : Fin (blkCount L), (i : ℕ)) * 2 ≤ rho L ^ 2 * blkCount L ^ 2 by
          rw [mul_assoc]; exact Nat.mul_le_mul_left _ hg)
      calc ∑ i, p i = ((rho L ^ 2 * ∑ i : Fin (blkCount L), (i : ℕ) : ℕ) : ℝ) / 2 ^ 56 := by
            simp only [p]
            rw [← Finset.sum_div, Nat.cast_mul, Nat.cast_sum, Finset.mul_sum]
            push_cast
            rfl
        _ ≤ lamX L := by
            unfold lamX
            rw [div_le_div_iff₀ (by positivity) (by positivity)]
            nlinarith
    have hp0 : 0 ≤ ∑ i, p i := Finset.sum_nonneg fun i _ => by positivity
    calc Pouw.pr (fun ω : Coins L => a ≤ (across L κ ω).card) =
          Pouw.pr (fun ω : Coins L => a ≤ (univ.filter fun i => acrossAt L κ ω ((blkEquiv L).symm i)).card) :=
          pr_congr fun ω => by rw [hcount]
      _ ≤ (∑ i, p i) ^ a / a.factorial := htail
      _ ≤ lamX L ^ a / a.factorial :=
          div_le_div_of_nonneg_right (pow_le_pow_left₀ hp0 hsum a) (by positivity)
  · have hr' : (ω.1, ω.2.1, ω.2.2.1) = (ω'.1, ω'.2.1, ω'.2.2.1) := hr
    simp only [Prod.mk.injEq] at hr'
    obtain ⟨h1, h2, h3⟩ := hr'
    have hlane : ∀ (τ : Fin L.S) (k : Fin L.tiles) (j : Fin L.n) ℓ, blkIdx L τ k ((j : ℕ) / 8) ≤ i →
        ω.2.2.2 τ k ℓ j = ω'.2.2.2 τ k ℓ j := fun τ k j ℓ hle => by
      rw [← laneEquiv_lane L hLn ω τ k ℓ j, ← laneEquiv_lane L hLn ω' τ k ℓ j]
      exact congrFun (congrFun (hl _ hle) ℓ) _
    have htl : ∀ rb cb (t : Fin L.S), blkIdx L t (rb / 4) cb ≤ i →
        tileOf L κ ω rb cb t = tileOf L κ ω' rb cb t := fun rb cb t hle =>
      tileOf_congr L κ ω ω' rb cb t fun i' τ j hi hj hτ =>
        coinTerm_congr L htile ω ω' i' τ j (fun _ => by rw [h1]) (fun _ q => by rw [h2]) (fun _ q => by rw [h3])
          fun k ℓ hk => hlane τ k j ℓ (by
            have e1 : (k : ℕ) = rb / 4 := by omega
            rw [e1, hj]
            exact le_trans (blkIdx_le_of_le L _ _ hτ) hle)
    have hβ : blkIdx L ((blkEquiv L).symm i).1 ((blkEquiv L).symm i).2.1 ((blkEquiv L).symm i).2.2 = (i : ℕ) := by
      rw [← blkEquiv_val, Equiv.apply_symm_apply]
    unfold acrossAt
    refine exists_congr fun rb => and_congr_right fun hrb => and_congr_right fun hrb4 => exists_congr fun x =>
      and_congr_right fun hx => and_congr_right fun hlt => ?_
    rw [htl rb _ _ (by rw [hrb4]; exact hβ.le), htl x.1 x.2.1 x.2.2 (by rw [← hβ]; exact hlt.le)]
  · simp only [e, laneEquiv_update]
    have := across_fiber_of L htile he hLm hLn κ ω ((blkEquiv L).symm i)
    rw [← blkEquiv_val, Equiv.apply_symm_apply] at this
    exact this

/-- **Blocks that repeat an earlier tile are rarely many**: `a` or more of them on at most a `λ_X^a / a!` share of
the coins. -/
theorem across_tail (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (a : ℕ) :
    Pouw.pr (fun ω : Coins (lead T m n K) => a ≤ (across _ κ ω).card) ≤ lamX (lead T m n K) ^ a / a.factorial :=
  across_tail_of _ rfl rfl (lead_m T m n K) (lead_n T m n K) κ a

end Pouw.SecurityProofs.Lead.Tail
