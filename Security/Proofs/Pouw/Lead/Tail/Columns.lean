import Proofs.Pouw.Lead.Tail.Words
import Proofs.Pouw.Lead.Tail.Fiber

/-!
# Column blocks with a tile repeated within a block

`columns_tail`: at a call of the lead, `k` or more column blocks hold a block with two equal tiles on at most a
`λ_G^k / k!` share of the coins, `λ_G = pairs · colBlocks · S · tiles / 7^8` (`lamG`).

A column block's own coordinates are its eight columns' tag weights (`setG`), and the coins are the rest (`b`, `a₁`,
the coin lanes), drawn first, and a product over the column blocks (`gEquiv`). `count_tail_equiv` then needs:

* **Reads**: a column block's tiles read the tag weights of its own columns only.
* **The fiber** (`columns_fiber`): two tiles at one step, column block and coin tile have rows of one parity in one
  coin tile, so their coin lanes cancel, and at their first rows and each column they differ by the row tags'
  difference against the column's four tag weights, plus terms the tag weights don't move. The tags differ
  (`Tags.rowTag_block_ne`), so with every other coin fixed, each column's weights meet the difference on at most `7³`
  of their `7⁴` values (`Tags.agreeG_card_le`'s count, on one fiber): `7^−8` a pair of tiles, over `pairs` pairs at
  each of the `S · tiles` slots.
* **The sum**: `colBlocks · pairs · S · tiles / 7^8 = lamG`.
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Pouw.Lead

/-- `ω` with column block `cb`'s tag weights set to `v`: column `8 cb + c` taking `v q c` as its digit `q`. -/
def setG (L : Pouw.Lead.Layout) (ω : Coins L) (cb : Fin L.colBlocks) (v : Fin 4 → Fin 8 → Fin 7) : Coins L :=
  (ω.1, ω.2.1,
    fun q j => if (j : ℕ) / 8 = cb then v q ⟨(j : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩ else ω.2.2.1 q j, ω.2.2.2)

open Finset in
theorem setG_eq (L : Pouw.Lead.Layout) (ω : Coins L) (cb : Fin L.colBlocks) (v : Fin 4 → Fin 8 → Fin 7) :
    setG L ω cb v =
      (Tags.blockG L (Tags.colsOf L cb)).upd ω fun j q => v q ⟨(j : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩ := by
  refine Prod.ext rfl (Prod.ext rfl (Prod.ext (funext fun q => funext fun j => ?_) rfl))
  simp [setG, Tags.blockG, Tags.colsOf]

open Finset in
/-- **A column block's tag weights move its words** by the row's tag against the change, at every step. -/
theorem word_setG (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) (cb : Fin L.colBlocks)
    (v : Fin 4 → Fin 8 → Fin 7) (i : Fin L.m) (t : Fin L.S) (j : Fin L.n) (hj : (j : ℕ) / 8 = cb) :
    word L κ (setG L ω cb v) i t j - word L κ ω i t j =
      Pouw.Dimension.Tag.tagTerm 3 (rowTag i) (fun q => v q ⟨(j : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩) -
        Pouw.Dimension.Tag.tagTerm 3 (rowTag i) (fun q => ω.2.2.1 q j) := by
  rw [setG_eq, Tags.word_updG, ite_eq_left (show j ∈ Tags.colsOf L cb from mem_filter.2 ⟨mem_univ _, hj⟩)]
  rfl

/-- A function on `Fin 4` off digit `q₀`. -/
def offDigit {Y : Type} (q₀ : Fin 4) (v : Fin 4 → Y) : Fin 3 → Y := fun k => v (q₀.succAbove k)

theorem offDigit_eq {Y : Type} {q₀ : Fin 4} {v w : Fin 4 → Y} (h : offDigit q₀ v = offDigit q₀ w) (q : Fin 4)
    (hq : q ≠ q₀) : v q = w q := by
  obtain ⟨k, rfl⟩ := Fin.exists_succAbove_eq hq
  exact congrFun h k

open Classical Finset in
/-- `columns_fiber` on any layout of whole MMA tiles. -/
theorem columns_fiber_of (L : Pouw.Lead.Layout) (hLn : L.n = 8 * L.colBlocks) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ)
    (ω : Coins L) (cb : Fin L.colBlocks) :
    ((Finset.univ.filter fun v => ∃ t ct, withinAt L κ (setG L ω cb v) (t, ct, cb)).card : ℝ) ≤
      (pairs L * L.S * L.tiles : ℕ) / 7 ^ 8 * Fintype.card (Fin 4 → Fin 8 → Fin 7) := by
  have hcard := card_filter_le_cover (V := Fin 4 → Fin 8 → Fin 7) (W := Fin 3 → Fin 8 → Fin 7)
    (fun v => ∃ t ct, withinAt L κ (setG L ω cb v) (t, ct, cb))
    ((univ : Finset (Fin L.S × Fin L.tiles)).sigma fun x => pairsAt L x.2)
    (fun i v => tileOf L κ (setG L ω cb v) (4 * i.1.2 + i.2.1) cb i.1.1 =
      tileOf L κ (setG L ω cb v) (4 * i.1.2 + i.2.2) cb i.1.1)
    (fun v ⟨t, ct, hw⟩ => by
      obtain ⟨p, hp, he⟩ := withinAt_pair hw
      exact ⟨⟨(t, ct), p⟩, mem_sigma.2 ⟨mem_univ _, hp⟩, he⟩)
    fun i hi => by
      obtain ⟨⟨t, ct⟩, p⟩ := i
      have hp := (mem_sigma.1 hi).2
      simp only [pairsAt, mem_filter, mem_product, mem_range] at hp
      obtain ⟨⟨ha4, hb4⟩, hab, hbR⟩ := hp
      have hrbA : 4 * (ct : ℕ) + p.1 < L.rowBlocks := by omega
      obtain ⟨q₀, hq₀⟩ := Tags.rowTag_block_ne (rb := 4 * ct + p.1) (rb' := 4 * ct + p.2) (by omega) (by omega)
      refine ⟨offDigit q₀, fun v v' hv hv' hvv => funext fun q => funext fun c => ?_⟩
      by_cases hq : q = q₀
      swap
      · rw [offDigit_eq hvv q hq]
      subst hq
      have hcb := cb.isLt
      have hc := c.isLt
      let j : Fin L.n := ⟨8 * cb + c, by omega⟩
      have hj8 : (j : ℕ) / 8 = cb := by simp only [j]; omega
      have hjc : (⟨(j : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩ : Fin 8) = c := Fin.ext (by simp only [j]; omega)
      have eA := congrFun hv ⟨(j : ℕ) % 8, Tags.pos_lt j⟩
      have eA' := congrFun hv' ⟨(j : ℕ) % 8, Tags.pos_lt j⟩
      simp only at eA eA'
      rw [Tags.tileOf_col κ _ (Tags.row_lt hrbA) t hj8, Tags.tileOf_col κ _ (Tags.row_lt hbR) t hj8] at eA eA'
      have d1 := word_setG L κ ω cb v ⟨16 * (4 * ct + p.1), Tags.row_lt hrbA⟩ t j hj8
      have d2 := word_setG L κ ω cb v ⟨16 * (4 * ct + p.2), Tags.row_lt hbR⟩ t j hj8
      have d1' := word_setG L κ ω cb v' ⟨16 * (4 * ct + p.1), Tags.row_lt hrbA⟩ t j hj8
      have d2' := word_setG L κ ω cb v' ⟨16 * (4 * ct + p.2), Tags.row_lt hbR⟩ t j hj8
      rw [hjc] at d1 d2 d1' d2'
      set r₁ := rowTag (16 * (4 * ct + p.1))
      set r₂ := rowTag (16 * (4 * ct + p.2))
      have hE : ∑ q, (r₁ - r₂) q * (((v q c : ℕ) : ℤ) - ((v' q c : ℕ) : ℤ)) = 0 := by
        have key : Pouw.Dimension.Tag.tagTerm 3 r₁ (fun q => v q c) -
            Pouw.Dimension.Tag.tagTerm 3 r₁ (fun q => v' q c) =
            Pouw.Dimension.Tag.tagTerm 3 r₂ (fun q => v q c) -
              Pouw.Dimension.Tag.tagTerm 3 r₂ (fun q => v' q c) := by
          linarith
        unfold Pouw.Dimension.Tag.tagTerm at key
        calc ∑ q, (r₁ - r₂) q * (((v q c : ℕ) : ℤ) - ((v' q c : ℕ) : ℤ)) =
              ((∑ q, r₁ q * (((v q c : ℕ) : ℤ) - 3)) - ∑ q, r₁ q * (((v' q c : ℕ) : ℤ) - 3)) -
                ((∑ q, r₂ q * (((v q c : ℕ) : ℤ) - 3)) - ∑ q, r₂ q * (((v' q c : ℕ) : ℤ) - 3)) := by
              simp only [← sum_sub_distrib]
              exact sum_congr rfl fun q _ => by simp only [Pi.sub_apply]; ring
          _ = 0 := by linarith
      rw [sum_eq_single q (fun q' _ hq' => by rw [offDigit_eq hvv q' hq']; ring) (by simp)] at hE
      rcases mul_eq_zero.1 hE with h | h
      · exact absurd h hq₀
      · exact Fin.ext (by exact_mod_cast sub_eq_zero.1 h)
  have hs : #((univ : Finset (Fin L.S × Fin L.tiles)).sigma fun x => pairsAt L x.2) ≤ L.S * L.tiles * pairs L := by
    rw [card_sigma]
    calc ∑ x : Fin L.S × Fin L.tiles, #(pairsAt L x.2) ≤ ∑ _x : Fin L.S × Fin L.tiles, pairs L :=
          sum_le_sum fun x _ => pairsAt_card L x.2
      _ = L.S * L.tiles * pairs L := by
          rw [sum_const, card_univ, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin, smul_eq_mul]
  rw [Fintype.card_fun, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin, Fintype.card_fin] at hcard ⊢
  calc (#(univ.filter fun v => ∃ t ct, withinAt L κ (setG L ω cb v) (t, ct, cb)) : ℝ) ≤
        (L.S * L.tiles * pairs L * (7 ^ 8) ^ 3 : ℕ) := by
        exact_mod_cast hcard.trans (Nat.mul_le_mul_right _ hs)
    _ = (pairs L * L.S * L.tiles : ℕ) / 7 ^ 8 * (((7 ^ 8) ^ 4 : ℕ) : ℝ) := by push_cast; ring

open Classical in
/-- **A column block's own tag weights rarely make two tiles of a block agree**: with every other coin fixed, column
block `cb` holds a block with two equal tiles on at most a `pairs · S · tiles / 7^8` share of its tag weights'
values. -/
theorem columns_fiber (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (ω : Coins (lead T m n K)) (cb : Fin (lead T m n K).colBlocks) :
    ((Finset.univ.filter fun v => ∃ t ct, withinAt _ κ (setG _ ω cb v) (t, ct, cb)).card : ℝ) ≤
      (pairs (lead T m n K) * (lead T m n K).S * (lead T m n K).tiles : ℕ) / 7 ^ 8 *
        Fintype.card (Fin 4 → Fin 8 → Fin 7) :=
  columns_fiber_of _ (lead_n T m n K) κ ω cb

/-- **The coins are the rest and each column block's tag weights.** -/
def gEquiv (L : Pouw.Lead.Layout) (hLn : L.n = 8 * L.colBlocks) :
    Coins L ≃ ((Fin L.m → Fin L.S → Fin 2) × (Fin L.m → Fin 4 → Fin 2) ×
      (Fin L.S → Fin L.tiles → Fin 3 → Fin L.n → Fin (2 ^ L.e))) × (Fin L.colBlocks → Fin 4 → Fin 8 → Fin 7) where
  toFun ω := ((ω.1, ω.2.1, ω.2.2.2), fun cb q c => ω.2.2.1 q (colIdx L hLn cb c))
  invFun x := (x.1.1, x.1.2.1, fun q j => x.2 (colBlk L hLn j) q ⟨(j : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩, x.1.2.2)
  left_inv ω := Prod.ext rfl (Prod.ext rfl (Prod.ext (funext fun q => funext fun j =>
    congrArg (ω.2.2.1 q) (colIdx_colBlk L hLn j)) rfl))
  right_inv x := by
    refine Prod.ext rfl (funext fun cb => funext fun q => funext fun c => ?_)
    have h1 : colBlk L hLn (colIdx L hLn cb c) = cb := Fin.ext (by simp only [colIdx, colBlk]; omega)
    have h2 : (⟨((colIdx L hLn cb c : Fin L.n) : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩ : Fin 8) = c :=
      Fin.ext (by simp only [colIdx]; omega)
    simp only [h1, h2]

theorem gEquiv_g (L : Pouw.Lead.Layout) (hLn : L.n = 8 * L.colBlocks) (ω : Coins L) (q : Fin 4) (j : Fin L.n) :
    (gEquiv L hLn ω).2 (colBlk L hLn j) q ⟨(j : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩ = ω.2.2.1 q j :=
  congrArg (ω.2.2.1 q) (colIdx_colBlk L hLn j)

/-- Moving one column block's coordinate is `setG`. -/
theorem gEquiv_update (L : Pouw.Lead.Layout) (hLn : L.n = 8 * L.colBlocks) (ω : Coins L) (cb : Fin L.colBlocks)
    (v : Fin 4 → Fin 8 → Fin 7) :
    (gEquiv L hLn).symm ((gEquiv L hLn ω).1, Function.update (gEquiv L hLn ω).2 cb v) = setG L ω cb v := by
  refine Prod.ext rfl (Prod.ext rfl (Prod.ext (funext fun q => funext fun j => ?_) rfl))
  simp only [gEquiv, Equiv.coe_fn_mk, Equiv.coe_fn_symm_mk, setG]
  by_cases hk : colBlk L hLn j = cb
  · rw [hk, Function.update_self, ite_eq_left (by rw [← hk]; rfl)]
  · rw [Function.update_of_ne hk, ite_eq_right (show ¬ (j : ℕ) / 8 = cb from fun h => hk (Fin.ext h))]
    exact congrArg (ω.2.2.1 q) (colIdx_colBlk L hLn j)

open Classical in
/-- `columns_tail` on any layout of whole MMA tiles and 64-row coin tiles. -/
theorem columns_tail_of (L : Pouw.Lead.Layout) (htile : L.tile = 64) (hLn : L.n = 8 * L.colBlocks)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (k : ℕ) :
    Pouw.pr (fun ω : Coins L => k ≤ (hitCols L κ ω).card) ≤ lamG L ^ k / k.factorial := by
  let e := gEquiv L hLn
  let p : Fin L.colBlocks → ℝ := fun _ => (pairs L * L.S * L.tiles : ℕ) / 7 ^ 8
  have htail := count_tail_equiv e (fun cb ω => ∃ t ct, withinAt L κ ω (t, ct, cb)) p (fun i => by positivity)
    (fun cb ω ω' hr hl => ?_) (fun cb ω => ?_) k
  · have hsum : ∑ i, p i = lamG L := by
      simp only [p, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, lamG]
      push_cast
      ring
    rw [← hsum]
    refine le_of_eq_of_le (pr_congr fun ω => ?_) htail
    unfold hitCols
    congr!
  · have hr' : (ω.1, ω.2.1, ω.2.2.2) = (ω'.1, ω'.2.1, ω'.2.2.2) := hr
    simp only [Prod.mk.injEq] at hr'
    obtain ⟨h1, h2, h3⟩ := hr'
    have hg : ∀ q (j : Fin L.n), (j : ℕ) / 8 = cb → ω.2.2.1 q j = ω'.2.2.1 q j := fun q j hj => by
      have hb : colBlk L hLn j = cb := Fin.ext hj
      rw [← gEquiv_g L hLn ω q j, ← gEquiv_g L hLn ω' q j, hb]
      exact congrFun (congrFun (hl cb le_rfl) q) _
    have htl : ∀ rb (t : Fin L.S), tileOf L κ ω rb cb t = tileOf L κ ω' rb cb t := fun rb t =>
      tileOf_congr L κ ω ω' rb cb t fun i' τ j _ hj _ =>
        coinTerm_congr L htile ω ω' i' τ j (fun _ => by rw [h1]) (fun _ q => by rw [h2])
          (fun _ q => hg q j hj) fun k ℓ _ => by rw [h3]
    simp only [withinAt, htl]
  · simp only [e, gEquiv_update]
    exact columns_fiber_of L hLn κ ω cb

/-- **Column blocks with a repeat inside a block are rarely many**: `k` or more of them on at most a `λ_G^k / k!`
share of the coins. -/
theorem columns_tail (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (k : ℕ) :
    Pouw.pr (fun ω : Coins (lead T m n K) => k ≤ (hitCols _ κ ω).card) ≤ lamG (lead T m n K) ^ k / k.factorial :=
  columns_tail_of _ rfl (lead_n T m n K) κ k

end Pouw.SecurityProofs.Lead.Tail
