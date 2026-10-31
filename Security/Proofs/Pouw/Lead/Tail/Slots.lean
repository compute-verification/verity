import Proofs.Pouw.Lead.Tail.Words
import Proofs.Pouw.Lead.Tail.Fiber

/-!
# One column block's slots with a tile repeated within a block

`slots_tail`: at a call of the lead whose coin tiles are whole (or which has one row block), `x` or more of one column
block's slots (step, coin tile) hold two equal tiles on at most a `λ_b^x / x!` share of the coins,
`λ_b = pairs · S · tiles / 2^16` (`lamB`).

A slot's own coordinates are 64 fresh bits, one a row of its coin tile (`setBits`): digit 0 of the step-0 bits `a₁` at
step 0, the tag bits `b` of its step at a later step. With whole coin tiles the coins are the rest (`b` at step 0,
`a₁`'s digits 1 to 3, `G`, the coin lanes), drawn first, and a product over the slots in slot order (step, then coin
tile; `bitsEquiv`). `count_tail_equiv` then needs:

* **Reads**: a slot's tiles are prefix sums over steps up to its own, of its own coin tile's rows: the bits of slots
  at or before it.
* **The fiber** (`slots_fiber`): two tiles at one step, column block and coin tile differ, at each of their 16 row
  pairs, by a term the slot's bits don't move and the two rows' fresh bits' difference: `b` once, or `a₁`'s digit 0
  against a column code digit 0 that is nonzero at one of the block's first two columns (it is zero only at
  `j ≡ 12 (mod 13)`). With every other coin fixed, each pair of rows meets it on at most half of the first row's bit
  values: `2^−16` a pair of tiles, over `pairs` pairs.
* **The sum**: `S · tiles · pairs / 2^16 = lamB`.
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Pouw.Lead

/-- `ω` with slot `(t, ct)`'s fresh bits set to `v`, row `i` of coin tile `ct` taking `v (i % 64)`: digit 0 of the
step-0 bits `a₁` at step 0, and the tag bits `b` of step `t` at a later step. -/
def setBits (L : Pouw.Lead.Layout) (ω : Coins L) (t : Fin L.S) (ct : Fin L.tiles) (v : Fin 64 → Fin 2) : Coins L :=
  (fun i τ => if (t : ℕ) ≠ 0 ∧ τ = t ∧ (i : ℕ) / 64 = ct then v ⟨(i : ℕ) % 64, Nat.mod_lt _ (by norm_num)⟩
      else ω.1 i τ,
    fun i q => if (t : ℕ) = 0 ∧ (q : ℕ) = 0 ∧ (i : ℕ) / 64 = ct then v ⟨(i : ℕ) % 64, Nat.mod_lt _ (by norm_num)⟩
      else ω.2.1 i q,
    ω.2.2.1, ω.2.2.2)

open Finset in
/-- **A slot's bits move its coin tile's words at its step** by `b` once, or at step 0 by `a₁`'s digit 0 against the
column code's digit 0. -/
theorem word_setBits (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) (t : Fin L.S)
    (ct : Fin L.tiles) (v v' : Fin 64 → Fin 2) (i : Fin L.m) (hi : (i : ℕ) / 64 = ct) (j : Fin L.n) :
    word L κ (setBits L ω t ct v) i t j - word L κ (setBits L ω t ct v') i t j =
      (if (t : ℕ) = 0 then colCode j 0 else 1) *
        (((v ⟨(i : ℕ) % 64, Nat.mod_lt _ (by norm_num)⟩ : ℕ) : ℤ) -
          ((v' ⟨(i : ℕ) % 64, Nat.mod_lt _ (by norm_num)⟩ : ℕ) : ℤ)) := by
  rw [Tags.word_sub, sum_eq_single t]
  · rw [ite_eq_left le_rfl, Tags.coinTerm_sub_of_lanes (setBits L ω t ct v) (setBits L ω t ct v') rfl]
    by_cases ht : (t : ℕ) = 0
    · simp only [ht, ite_true, setBits, hi, Fin.sum_univ_four, true_and, and_true, Coins.G]
      simp
      ring
    · simp [ht, setBits, hi]
  · intro τ _ hτ
    split_ifs with h
    · rw [Tags.coinTerm_sub_of_lanes (setBits L ω t ct v) (setBits L ω t ct v') rfl]
      by_cases ht : (t : ℕ) = 0
      · have : (τ : ℕ) ≠ 0 := fun h0 => hτ (Fin.ext (h0.trans ht.symm))
        simp [setBits, ht, this]
      · simp [setBits, ht, hτ, Coins.G]
    · rfl
  · intro h
    exact absurd (mem_univ t) h

open Classical Finset in
/-- `slots_fiber` on any layout of whole MMA tiles. -/
theorem slots_fiber_of (L : Pouw.Lead.Layout) (hLm : L.m = 16 * L.rowBlocks) (hLn : L.n = 8 * L.colBlocks)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) (cb : Fin L.colBlocks) (t : Fin L.S) (ct : Fin L.tiles) :
    ((Finset.univ.filter fun v => withinAt L κ (setBits L ω t ct v) (t, ct, cb)).card : ℝ) ≤
      (pairs L : ℝ) / 2 ^ 16 * Fintype.card (Fin 64 → Fin 2) := by
  obtain ⟨c₀, hc₀, hcoef⟩ : ∃ c₀ < 8, (if (t : ℕ) = 0 then colCode (8 * cb + c₀) 0 else 1) ≠ 0 := by
    by_cases h12 : (8 * (cb : ℕ)) % 13 = 12
    · refine ⟨1, by norm_num, ?_⟩
      split_ifs
      · simp only [colCode, bal13, Fin.val_zero, pow_zero, Int.ediv_one]; push_cast; omega
      · norm_num
    · refine ⟨0, by norm_num, ?_⟩
      split_ifs
      · simp only [colCode, bal13, Fin.val_zero, pow_zero, Int.ediv_one]; push_cast; omega
      · norm_num
  have hcard := card_filter_le_cover (V := Fin 64 → Fin 2) (W := Fin 48 → Fin 2)
    (fun v => withinAt L κ (setBits L ω t ct v) (t, ct, cb)) (pairsAt L ct)
    (fun p v => tileOf L κ (setBits L ω t ct v) (4 * ct + p.1) cb t =
      tileOf L κ (setBits L ω t ct v) (4 * ct + p.2) cb t)
    (fun v hv => withinAt_pair hv) fun p hp => ⟨offBlock p.2, fun v v' hv hv' hvv => funext fun k => by
      simp only [pairsAt, mem_filter, mem_product, mem_range] at hp
      obtain ⟨⟨ha4, hb4⟩, hab, hbR⟩ := hp
      have hoff := offBlock_eq hb4 hvv
      by_cases hk : (k : ℕ) / 16 = p.2
      · have hr : (k : ℕ) % 16 < 16 := Nat.mod_lt _ (by norm_num)
        have hrbA : 4 * (ct : ℕ) + p.1 < L.rowBlocks := by omega
        have hcb := cb.isLt
        have eA := congrFun hv ⟨8 * ((k : ℕ) % 16) + c₀, by unfold tileWords; omega⟩
        have eA' := congrFun hv' ⟨8 * ((k : ℕ) % 16) + c₀, by unfold tileWords; omega⟩
        rw [tileOf_at hLm hLn _ _ hrbA hcb t hr hc₀, tileOf_at hLm hLn _ _ hbR hcb t hr hc₀] at eA eA'
        have dA := word_setBits L κ ω t ct v v' ⟨16 * (4 * ct + p.1) + (k : ℕ) % 16, by omega⟩ (by simp; omega)
          ⟨8 * cb + c₀, by omega⟩
        have dB := word_setBits L κ ω t ct v v' ⟨16 * (4 * ct + p.2) + (k : ℕ) % 16, by omega⟩ (by simp; omega)
          ⟨8 * cb + c₀, by omega⟩
        have hvA := hoff ⟨(16 * (4 * ct + p.1) + (k : ℕ) % 16) % 64, Nat.mod_lt _ (by norm_num)⟩ (by simp; omega)
        rw [hvA, sub_self, mul_zero] at dA
        have hk' : (⟨(16 * (4 * ct + p.2) + (k : ℕ) % 16) % 64, Nat.mod_lt _ (by norm_num)⟩ : Fin 64) = k :=
          Fin.ext (by simp only; omega)
        rw [hk'] at dB
        have h0 : (if (t : ℕ) = 0 then colCode (8 * cb + c₀) 0 else 1) * (((v k : ℕ) : ℤ) - ((v' k : ℕ) : ℤ)) = 0 := by
          rw [← dB]
          linarith
        rcases mul_eq_zero.1 h0 with h | h
        · exact absurd h hcoef
        · exact Fin.ext (by exact_mod_cast sub_eq_zero.1 h)
      · exact hoff k hk⟩
  rw [Fintype.card_fun, Fintype.card_fin, Fintype.card_fin] at hcard ⊢
  calc (#(univ.filter fun v => withinAt L κ (setBits L ω t ct v) (t, ct, cb)) : ℝ) ≤ (#(pairsAt L ct) * 2 ^ 48 : ℕ) := by
        exact_mod_cast hcard
    _ ≤ (pairs L * 2 ^ 48 : ℕ) := by exact_mod_cast Nat.mul_le_mul_right _ (pairsAt_card L ct)
    _ = (pairs L : ℝ) / 2 ^ 16 * ((2 ^ 64 : ℕ) : ℝ) := by push_cast; ring

open Classical in
/-- **A slot's fresh bits rarely make two tiles of a block agree**: with every other coin fixed, slot `(t, ct)` of
column block `cb` holds two equal tiles on at most a `pairs / 2^16` share of its fresh bits' values. -/
theorem slots_fiber (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (ω : Coins (lead T m n K)) (cb : Fin (lead T m n K).colBlocks) (t : Fin (lead T m n K).S)
    (ct : Fin (lead T m n K).tiles) :
    ((Finset.univ.filter fun v => withinAt _ κ (setBits _ ω t ct v) (t, ct, cb)).card : ℝ) ≤
      (pairs (lead T m n K) : ℝ) / 2 ^ 16 * Fintype.card (Fin 64 → Fin 2) :=
  slots_fiber_of _ (lead_m T m n K) (lead_n T m n K) κ ω cb t ct

/-- Row `64 ct + k`, with whole coin tiles. -/
def rowIdx (L : Pouw.Lead.Layout) (hmt : L.m = 64 * L.tiles) (ct : Fin L.tiles) (k : Fin 64) : Fin L.m :=
  ⟨64 * ct + k, by have := ct.isLt; have := k.isLt; omega⟩

/-- Row `i`'s coin tile, with whole coin tiles. -/
def rowTile (L : Pouw.Lead.Layout) (hmt : L.m = 64 * L.tiles) (i : Fin L.m) : Fin L.tiles :=
  ⟨(i : ℕ) / 64, by have := i.isLt; omega⟩

theorem rowIdx_rowTile (L : Pouw.Lead.Layout) (hmt : L.m = 64 * L.tiles) (i : Fin L.m) :
    rowIdx L hmt (rowTile L hmt i) ⟨(i : ℕ) % 64, Nat.mod_lt _ (by norm_num)⟩ = i :=
  Fin.ext (by simp only [rowIdx, rowTile]; omega)

/-- The slots in slot order: step, then coin tile. -/
noncomputable def slotEquiv (L : Pouw.Lead.Layout) : Fin L.S × Fin L.tiles ≃ Fin (L.S * L.tiles) :=
  Equiv.ofBijective (fun s => ⟨s.1 * L.tiles + s.2, mul_add_lt s.1.isLt s.2.isLt⟩)
    ((Fintype.bijective_iff_injective_and_card _).2 ⟨fun s s' h => by
      obtain ⟨e1, e2⟩ := mul_add_inj s.2.isLt s'.2.isLt (congrArg Fin.val h)
      exact Prod.ext (Fin.ext e1) (Fin.ext e2), by simp only [Fintype.card_prod, Fintype.card_fin]⟩)

/-- A slot's 64 fresh bits: `a₁`'s digit 0 at step 0, the tag bits of its step at a later step. -/
def slotBits (L : Pouw.Lead.Layout) (hmt : L.m = 64 * L.tiles) (ω : Coins L) (s : Fin L.S × Fin L.tiles)
    (k : Fin 64) : Fin 2 :=
  if (s.1 : ℕ) = 0 then ω.2.1 (rowIdx L hmt s.2 k) 0 else ω.1 (rowIdx L hmt s.2 k) s.1

theorem slotBits_row (L : Pouw.Lead.Layout) (hmt : L.m = 64 * L.tiles) (ω : Coins L) (τ : Fin L.S) (i : Fin L.m) :
    slotBits L hmt ω (τ, rowTile L hmt i) ⟨(i : ℕ) % 64, Nat.mod_lt _ (by norm_num)⟩ =
      if (τ : ℕ) = 0 then ω.2.1 i 0 else ω.1 i τ := by
  simp only [slotBits, rowIdx_rowTile]

/-- **With whole coin tiles, the coins are the rest and each slot's fresh bits**, the slots in slot order. -/
noncomputable def bitsEquiv (L : Pouw.Lead.Layout) (hS : 0 < L.S) (hmt : L.m = 64 * L.tiles) :
    Coins L ≃ ((Fin L.m → Fin 2) × (Fin L.m → Fin 3 → Fin 2) × (Fin 4 → Fin L.n → Fin 7) ×
      (Fin L.S → Fin L.tiles → Fin 3 → Fin L.n → Fin (2 ^ L.e))) × (Fin (L.S * L.tiles) → Fin 64 → Fin 2) where
  toFun ω := ((fun i => ω.1 i ⟨0, hS⟩, fun i k => ω.2.1 i k.succ, ω.2.2.1, ω.2.2.2),
    fun s => slotBits L hmt ω ((slotEquiv L).symm s))
  invFun x := (fun i τ => if (τ : ℕ) = 0 then x.1.1 i
      else x.2 (slotEquiv L (τ, rowTile L hmt i)) ⟨(i : ℕ) % 64, Nat.mod_lt _ (by norm_num)⟩,
    fun i q => if h : (q : ℕ) = 0 then
        x.2 (slotEquiv L (⟨0, hS⟩, rowTile L hmt i)) ⟨(i : ℕ) % 64, Nat.mod_lt _ (by norm_num)⟩
      else x.1.2.1 i ⟨(q : ℕ) - 1, by omega⟩,
    x.1.2.2.1, x.1.2.2.2)
  left_inv ω := by
    refine Prod.ext (funext fun i => funext fun τ => ?_) (Prod.ext (funext fun i => funext fun q => ?_) rfl)
    · simp only [Equiv.symm_apply_apply, slotBits_row]
      split_ifs with h0
      · exact congrArg (ω.1 i) (Fin.ext h0.symm)
      · rfl
    · simp only [Equiv.symm_apply_apply, slotBits_row]
      split_ifs with h0
      · exact congrArg (ω.2.1 i) (Fin.ext h0.symm)
      · exact congrArg (ω.2.1 i) (Fin.ext (by simp only [Fin.val_succ]; omega))
  right_inv x := by
    refine Prod.ext (Prod.ext (funext fun i => ?_) (Prod.ext (funext fun i => funext fun k => ?_) rfl))
      (funext fun s => funext fun k => ?_)
    · simp only [↓reduceIte]
    · simp only [Fin.val_succ, Nat.add_one_ne_zero, ↓reduceDIte, Nat.add_sub_cancel]
    · obtain ⟨⟨t, ct⟩, rfl⟩ := (slotEquiv L).surjective s
      have h1 : rowTile L hmt (rowIdx L hmt ct k) = ct := Fin.ext (by simp only [rowIdx, rowTile]; omega)
      have h2 : (⟨((rowIdx L hmt ct k : Fin L.m) : ℕ) % 64, Nat.mod_lt _ (by norm_num)⟩ : Fin 64) = k :=
        Fin.ext (by simp only [rowIdx]; omega)
      simp only [Equiv.symm_apply_apply, slotBits]
      by_cases h0 : (t : ℕ) = 0
      · have ht : (⟨0, hS⟩ : Fin L.S) = t := Fin.ext h0.symm
        simp only [h0, ↓reduceIte, Fin.val_zero, ↓reduceDIte]
        rw [h1, h2, ht]
      · simp only [h0, ↓reduceIte]
        rw [h1, h2]

/-- Moving one slot's coordinate is `setBits`. -/
theorem bitsEquiv_update (L : Pouw.Lead.Layout) (hS : 0 < L.S) (hmt : L.m = 64 * L.tiles) (ω : Coins L)
    (s : Fin (L.S * L.tiles)) (v : Fin 64 → Fin 2) :
    (bitsEquiv L hS hmt).symm ((bitsEquiv L hS hmt ω).1, Function.update (bitsEquiv L hS hmt ω).2 s v) =
      setBits L ω ((slotEquiv L).symm s).1 ((slotEquiv L).symm s).2 v := by
  obtain ⟨⟨t, ct⟩, rfl⟩ := (slotEquiv L).surjective s
  simp only [Equiv.symm_apply_apply]
  refine Prod.ext (funext fun i => funext fun τ => ?_) (Prod.ext (funext fun i => funext fun q => ?_) rfl)
  · simp only [bitsEquiv, Equiv.coe_fn_mk, Equiv.coe_fn_symm_mk, setBits]
    by_cases h0 : (τ : ℕ) = 0
    · rw [ite_eq_left h0, ite_eq_right (by rintro ⟨h1, rfl, -⟩; exact h1 h0)]
      exact congrArg (ω.1 i) (Fin.ext h0.symm)
    · rw [ite_eq_right h0]
      by_cases hk : slotEquiv L (τ, rowTile L hmt i) = slotEquiv L (t, ct)
      · have he := (slotEquiv L).injective hk
        simp only [Prod.mk.injEq] at he
        obtain ⟨ht, hct⟩ := he
        rw [hk, Function.update_self, ite_eq_left (show (t : ℕ) ≠ 0 ∧ τ = t ∧ (i : ℕ) / 64 = ct from
          ⟨by rw [← ht]; exact h0, ht, by rw [← hct]; rfl⟩)]
      · rw [Function.update_of_ne hk, ite_eq_right ?_]
        · simp only [Equiv.symm_apply_apply, slotBits_row, h0, ↓reduceIte]
        · rintro ⟨-, rfl, h3⟩
          exact hk (congrArg _ (Prod.ext rfl (Fin.ext h3)))
  · simp only [bitsEquiv, Equiv.coe_fn_mk, Equiv.coe_fn_symm_mk, setBits]
    by_cases h0 : (q : ℕ) = 0
    · simp only [h0, ↓reduceDIte]
      by_cases hk : slotEquiv L (⟨0, hS⟩, rowTile L hmt i) = slotEquiv L (t, ct)
      · have he := (slotEquiv L).injective hk
        simp only [Prod.mk.injEq] at he
        obtain ⟨ht, hct⟩ := he
        rw [hk, Function.update_self, ite_eq_left ⟨by rw [← ht], trivial, by rw [← hct]; rfl⟩]
      · rw [Function.update_of_ne hk, ite_eq_right ?_]
        · simp only [Equiv.symm_apply_apply, slotBits_row, ↓reduceIte]
          exact congrArg (ω.2.1 i) (Fin.ext h0.symm)
        · rintro ⟨h1, -, h3⟩
          exact hk (congrArg _ (Prod.ext (Fin.ext h1.symm) (Fin.ext h3)))
    · simp only [h0, ↓reduceDIte, false_and, and_false, ↓reduceIte]
      exact congrArg (ω.2.1 i) (Fin.ext (by simp only [Fin.val_succ]; omega))

open Classical Finset in
/-- `slots_tail` on any layout of whole MMA tiles, whole 64-row coin tiles and at least one step. -/
theorem slots_tail_of (L : Pouw.Lead.Layout) (hS : 0 < L.S) (htile : L.tile = 64) (hmt : L.m = 64 * L.tiles)
    (hLm : L.m = 16 * L.rowBlocks) (hLn : L.n = 8 * L.colBlocks) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ)
    (cb : Fin L.colBlocks) (x : ℕ) :
    Pouw.pr (fun ω : Coins L => x ≤ (hitSlots L κ ω cb).card) ≤ lamB L ^ x / x.factorial := by
  let e := bitsEquiv L hS hmt
  let p : Fin (L.S * L.tiles) → ℝ := fun _ => (pairs L : ℝ) / 2 ^ 16
  have hcount : ∀ ω, (hitSlots L κ ω cb).card =
      (univ.filter fun s => withinAt L κ ω (((slotEquiv L).symm s).1, ((slotEquiv L).symm s).2, cb)).card :=
    fun ω => Finset.card_equiv (slotEquiv L) fun s => by
      simp only [hitSlots, Finset.mem_filter, Finset.mem_univ, true_and, Equiv.symm_apply_apply]
  have htail := count_tail_equiv e
    (fun s ω => withinAt L κ ω (((slotEquiv L).symm s).1, ((slotEquiv L).symm s).2, cb)) p
    (fun i => by positivity) (fun s ω ω' hr hl => ?_) (fun s ω => ?_) x
  · have hsum : ∑ i, p i = lamB L := by
      simp only [p, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, lamB]
      push_cast
      ring
    rw [← hsum]
    exact le_of_eq_of_le (pr_congr fun ω => by rw [hcount]) htail
  · obtain ⟨⟨t, ct⟩, rfl⟩ := (slotEquiv L).surjective s
    simp only [Equiv.symm_apply_apply]
    have hr' : ((fun i => ω.1 i ⟨0, hS⟩), (fun i (k : Fin 3) => ω.2.1 i k.succ), ω.2.2.1, ω.2.2.2) =
        ((fun i => ω'.1 i ⟨0, hS⟩), (fun i (k : Fin 3) => ω'.2.1 i k.succ), ω'.2.2.1, ω'.2.2.2) := hr
    simp only [Prod.mk.injEq] at hr'
    obtain ⟨-, ha, hg, hs⟩ := hr'
    have hbit : ∀ (τ : Fin L.S) (i : Fin L.m), τ ≤ t → (i : ℕ) / 64 = ct →
        (if (τ : ℕ) = 0 then ω.2.1 i 0 else ω.1 i τ) = (if (τ : ℕ) = 0 then ω'.2.1 i 0 else ω'.1 i τ) :=
      fun τ i hτ hi => by
        have hle : slotEquiv L (τ, rowTile L hmt i) ≤ slotEquiv L (t, ct) := by
          change (τ : ℕ) * L.tiles + (i : ℕ) / 64 ≤ (t : ℕ) * L.tiles + ct
          rw [hi]
          exact Nat.add_le_add_right (Nat.mul_le_mul_right _ hτ) _
        have := congrFun (hl _ hle) ⟨(i : ℕ) % 64, Nat.mod_lt _ (by norm_num)⟩
        simp only [e, bitsEquiv, Equiv.coe_fn_mk, Equiv.symm_apply_apply, slotBits_row] at this
        exact this
    have htl : ∀ rb, rb / 4 = (ct : ℕ) → tileOf L κ ω rb cb t = tileOf L κ ω' rb cb t := fun rb hrb =>
      tileOf_congr L κ ω ω' rb cb t fun i' τ j hi _ hτ => by
        have hi64 : (i' : ℕ) / 64 = ct := by omega
        have hb := hbit τ i' hτ hi64
        refine coinTerm_congr L htile ω ω' i' τ j (fun h0 => ?_) (fun h0 q => ?_) (fun _ q => by rw [hg])
          fun k ℓ _ => by rw [hs]
        · simpa only [h0, ↓reduceIte] using hb
        · by_cases hq : (q : ℕ) = 0
          · have hq0 : q = 0 := Fin.ext hq
            subst hq0
            simpa only [h0, ↓reduceIte] using hb
          · have := congrFun (congrFun ha i') ⟨(q : ℕ) - 1, by omega⟩
            have hq' : (⟨(q : ℕ) - 1, by omega⟩ : Fin 3).succ = q := Fin.ext (by simp only [Fin.val_succ]; omega)
            rwa [hq'] at this
    unfold withinAt
    refine exists_congr fun rb => and_congr_right fun _ => exists_congr fun rb' => and_congr_right fun _ =>
      and_congr_right fun h1 => and_congr_right fun h2 => and_congr_right fun _ => ?_
    rw [htl rb h1, htl rb' h2]
  · simp only [e, bitsEquiv_update]
    exact slots_fiber_of L hLm hLn κ ω cb _ _

/-- **One column block's slots with a repeat inside a block are rarely many**: `x` or more of them on at most a
`λ_b^x / x!` share of the coins, when the coin tiles are whole or there is one row block. -/
theorem slots_tail (T m n K : ℕ) (hm : 64 ∣ m ∨ m ≤ 16)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (cb : Fin (lead T m n K).colBlocks) (x : ℕ) :
    Pouw.pr (fun ω : Coins (lead T m n K) => x ≤ (hitSlots _ κ ω cb).card) ≤
      lamB (lead T m n K) ^ x / x.factorial := by
  classical
  rcases hm with hm | hm
  · have hmt : (lead T m n K).m = 64 * (lead T m n K).tiles := by
      obtain ⟨k, rfl⟩ := hm
      simp only [lead, padRows, Layout.tiles]
      omega
    exact slots_tail_of _ (by simp only [lead, stepsFor]; omega) rfl hmt (lead_m T m n K) (lead_n T m n K) κ cb x
  · have hrb : (lead T m n K).rowBlocks ≤ 1 := by simp only [Layout.rowBlocks, lead, padRows]; omega
    have hempty : ∀ ω, hitSlots _ κ ω cb = ∅ := fun ω => by
      unfold hitSlots
      rw [Finset.filter_eq_empty_iff]
      rintro s - ⟨rb, hrb1, rb', hrb2, -, -, hne, -⟩
      omega
    have hp : pairs (lead T m n K) = 0 := by
      have : rho (lead T m n K) ≤ 1 := (min_le_right _ _).trans hrb
      unfold pairs
      rw [Nat.sub_eq_zero_of_le this, mul_zero, Nat.zero_div]
    rcases x with _ | x
    · simpa using Pouw.pr_le_one _
    · rw [Pouw.pr_eq_zero fun ω h => by rw [hempty] at h; simp at h]
      simp [lamB, hp]

end Pouw.SecurityProofs.Lead.Tail
