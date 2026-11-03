import Mathlib.Algebra.BigOperators.Ring.Finset
import Definitions.Pouw.Lead.Model
import Proofs.Pouw.Lead.Tags.Shift
import Proofs.Pouw.Lead.Tags.TagProofs

/-!
# Two of the lead's MMA tiles agree on at most a `7^−8` share of the coins

The model's tile of committed words (`Pouw.Lead.tileOf`, 16 rows × 8 columns) is what a free copy moves. Two tiles at
different places, or a tile and one fixed before the coins, are equal on at most a `7^−|J|` share of the coins, with `J`
the first tile's columns (`colsOf`, 8 for a whole column block by `colsOf_card`): `tile_agree_card_le` and
`tile_fixed_card_le`, below 28,560 rows. Their first rows already tell them apart:

* **The columns differ, or the other tile is fixed** (`agreeG_card_le` against a target that reads no tag weight of
  `J`): the row tag is nonzero below 28,560 rows (`rowTag_ne_zero`), so each column's four tag weights meet the target
  on at most `7³` of their `7⁴` values.
* **The rows differ, the columns don't** (`tile_rows_card_le`): the two row tags differ (`rowTag_inj`), and their
  difference against the same weights does the same.
* **Only the step differs**: the step-0 tag term cancels, and the later step's coin lane `s` in each column moves one
  word under `-rot`'s weight and not the other (`agreeU_card_le`), a `2^−e` share a column.

Each is a count through a block of coins (`TileBlock.card_le`): rewriting the block leaves every other coin, so over each
value of the others the block's values meeting every column's condition are counted column by column.
-/

namespace Pouw.SecurityProofs.Lead.Tags

open Finset Pouw.Lead Pouw.Dimension.Tag

/-! ## Counting through a block of coins -/

/-- A block `J` of coins on a finite space `X`: `get x j` reads the block's coin `j`, and `upd x v` sets the block to
`v` and leaves every other coin. -/
structure TileBlock (X ι B : Type) (J : Finset ι) where
  get : X → ι → B
  upd : X → (ι → B) → X
  get_upd : ∀ x v, ∀ j ∈ J, get (upd x v) j = v j
  upd_upd : ∀ x v w, upd (upd x v) w = upd x w
  upd_get : ∀ x, upd x (get x) = x
  upd_congr : ∀ x v w, (∀ j ∈ J, v j = w j) → upd x v = upd x w

section TileBlock

variable {X ι B : Type} [Fintype X] [DecidableEq X] [Fintype B] [DecidableEq ι] {J : Finset ι}

/-- **A count through a block**: if, whatever the other coins, each coin `j` of the block meets its condition at at
most `M` of its `|B|` values, the points meeting every condition are at most an `(M/|B|)^|J|` share. -/
theorem TileBlock.card_le (β : TileBlock X ι B J) (b₀ : B) (P : ι → X → B → Prop) [∀ j r, DecidablePred (P j r)]
    [DecidablePred fun x => ∀ j ∈ J, P j (β.upd x fun _ => b₀) (β.get x j)] {M : ℕ} (hP : ∀ j ∈ J, ∀ r, (univ.filter (P j r)).card ≤ M) :
    (univ.filter fun x => ∀ j ∈ J, P j (β.upd x fun _ => b₀) (β.get x j)).card * Fintype.card B ^ J.card ≤
      Fintype.card X * M ^ J.card := by
  have hfib : ∀ x, β.upd (β.upd x fun _ => b₀) (β.get x) = x := fun x => by rw [β.upd_upd, β.upd_get]
  have himg : ∀ x ∈ (univ : Finset X), (β.upd x fun _ => b₀) ∈ (univ : Finset X).image fun x => β.upd x fun _ => b₀ :=
    fun x _ => mem_image_of_mem _ (mem_univ x)
  have hE : (univ.filter fun x => ∀ j ∈ J, P j (β.upd x fun _ => b₀) (β.get x j)).card ≤
      ((univ : Finset X).image fun x => β.upd x fun _ => b₀).card * M ^ J.card := by
    rw [card_eq_sum_card_fiberwise (f := fun x => β.upd x fun _ => b₀) fun x hx => himg x (mem_univ x)]
    refine (sum_le_card_nsmul _ _ _ fun r _ => ?_).trans (nsmul_eq_mul _ _).le
    calc _ ≤ (J.pi fun j => univ.filter (P j r)).card := by
          refine card_le_card_of_injOn (fun x j _ => β.get x j) (fun x hx => ?_) fun x hx x' hx' hxx' => ?_
          · obtain ⟨hxE, hxr⟩ := mem_filter.1 (mem_coe.1 hx)
            refine mem_coe.2 (mem_pi.2 fun j hj => mem_filter.2 ⟨mem_univ _, ?_⟩)
            rw [← hxr]
            exact (mem_filter.1 hxE).2 j hj
          · obtain ⟨-, hr⟩ := mem_filter.1 (mem_coe.1 hx)
            obtain ⟨-, hr'⟩ := mem_filter.1 (mem_coe.1 hx')
            have hv : ∀ j ∈ J, β.get x j = β.get x' j := fun j hj => congrFun (congrFun hxx' j) hj
            rw [← hfib x, ← hfib x', hr, hr']
            exact β.upd_congr r _ _ hv
      _ = ∏ j ∈ J, (univ.filter (P j r)).card := card_pi _ _
      _ ≤ M ^ J.card := prod_le_pow_card _ _ _ fun j hj => hP j hj r
  have hX : ((univ : Finset X).image fun x => β.upd x fun _ => b₀).card * Fintype.card B ^ J.card ≤
      Fintype.card X := by
    rw [← Finset.card_univ (α := X), card_eq_sum_card_fiberwise (f := fun x => β.upd x fun _ => b₀) fun x hx => himg x hx,
      ← smul_eq_mul]
    refine card_nsmul_le_sum _ _ _ fun r hr => ?_
    obtain ⟨y, -, rfl⟩ := mem_image.1 hr
    calc Fintype.card B ^ J.card = ∏ _j ∈ J, (univ : Finset B).card := by rw [prod_const, card_univ]
      _ = (J.pi fun _ => (univ : Finset B)).card := (card_pi _ _).symm
      _ ≤ _ := by
          refine card_le_card_of_injOn
            (fun v => β.upd (β.upd y fun _ => b₀) fun j => if hj : j ∈ J then v j hj else b₀)
            (fun v _ => ?_) fun v _ v' _ hvv' => ?_
          · refine mem_coe.2 (mem_filter.2 ⟨mem_univ _, ?_⟩)
            dsimp only
            rw [β.upd_upd, β.upd_upd]
          · funext j hj
            have := congrArg (fun x => β.get x j) hvv'
            simp only [β.get_upd _ _ j hj, dite_eq_left hj] at this
            exact this
  calc _ ≤ ((univ : Finset X).image fun x => β.upd x fun _ => b₀).card * M ^ J.card * Fintype.card B ^ J.card :=
        Nat.mul_le_mul_right _ hE
    _ = ((univ : Finset X).image fun x => β.upd x fun _ => b₀).card * Fintype.card B ^ J.card * M ^ J.card := by
        ring
    _ ≤ Fintype.card X * M ^ J.card := Nat.mul_le_mul_right _ hX

end TileBlock

/-! ## The lead's two blocks: a column block's tag weights and a coin lane -/

variable {L : Pouw.Lead.Layout}

/-- The tag weights of the columns `J`: column `j`'s four digits. -/
def blockG (L : Pouw.Lead.Layout) (J : Finset (Fin L.n)) : TileBlock (Coins L) (Fin L.n) (Fin 4 → Fin 7) J where
  get ω j q := ω.2.2.1 q j
  upd ω v := (ω.1, ω.2.1, fun q j => if j ∈ J then v j q else ω.2.2.1 q j, ω.2.2.2)
  get_upd ω v j hj := by funext q; simp [hj]
  upd_upd ω v w := by
    refine Prod.ext rfl (Prod.ext rfl (Prod.ext ?_ rfl))
    funext q j; dsimp only; split_ifs <;> rfl
  upd_get ω := by simp
  upd_congr ω v w h := by
    refine Prod.ext rfl (Prod.ext rfl (Prod.ext ?_ rfl))
    funext q j; dsimp only; by_cases hj : j ∈ J <;> simp [hj, h j]

/-- The coin lane `ℓ₀` of step `t₀` in coin tile `k₀`, at the columns `J`. -/
def blockU (L : Pouw.Lead.Layout) (t₀ : Fin L.S) (k₀ : Fin L.tiles) (ℓ₀ : Fin 3) (J : Finset (Fin L.n)) :
    TileBlock (Coins L) (Fin L.n) (Fin (2 ^ L.e)) J where
  get ω j := ω.2.2.2 t₀ k₀ ℓ₀ j
  upd ω v := (ω.1, ω.2.1, ω.2.2.1,
    fun t k ℓ j => if t = t₀ ∧ k = k₀ ∧ ℓ = ℓ₀ ∧ j ∈ J then v j else ω.2.2.2 t k ℓ j)
  get_upd ω v j hj := by simp [hj]
  upd_upd ω v w := by
    refine Prod.ext rfl (Prod.ext rfl (Prod.ext rfl ?_))
    funext t k ℓ j; dsimp only; split_ifs <;> rfl
  upd_get ω := by
    refine Prod.ext rfl (Prod.ext rfl (Prod.ext rfl ?_))
    funext t k ℓ j; dsimp only; split_ifs with h
    · rw [h.1, h.2.1, h.2.2.1]
    · rfl
  upd_congr ω v w h := by
    refine Prod.ext rfl (Prod.ext rfl (Prod.ext rfl ?_))
    funext t k ℓ j; dsimp only; split_ifs with h'
    · rw [h j h'.2.2.2]
    · rfl

theorem tagTerm_sub {ι : Type} [Fintype ι] {q : ℕ} (h : ℤ) (d d' : ι → ℤ) (g : ι → Fin q) :
    tagTerm h (d - d') g = tagTerm h d g - tagTerm h d' g := by
  unfold tagTerm
  rw [← sum_sub_distrib]
  exact sum_congr rfl fun e _ => by simp only [Pi.sub_apply]; ring

/-- Rewriting a column block's tag weights moves a word in it by its row tag against the change. -/
theorem word_updG (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (J : Finset (Fin L.n)) (ω : Coins L)
    (v : Fin L.n → Fin 4 → Fin 7) (i : Fin L.m) (t : Fin L.S) (j : Fin L.n) :
    word L κ ((blockG L J).upd ω v) i t j - word L κ ω i t j =
      if j ∈ J then tagTerm 3 (rowTag i) (v j) - tagTerm 3 (rowTag i) ((blockG L J).get ω j) else 0 := by
  have hS : 0 < L.S := Nat.lt_of_le_of_lt (Nat.zero_le _) t.isLt
  rw [word_sub, Finset.sum_eq_single ⟨0, hS⟩]
  · rw [ite_eq_left (Fin.le_def.2 (Nat.zero_le _)), coinTerm_sub_of_lanes ((blockG L J).upd ω v) ω rfl]
    by_cases hj : j ∈ J
    · simp [blockG, hj, tagTerm, Coins.G]
    · simp [blockG, hj, Coins.G]
  · intro τ _ hτ
    rw [coinTerm_sub_of_lanes ((blockG L J).upd ω v) ω rfl]
    have h : (τ : ℕ) ≠ 0 := fun h => hτ (Fin.ext h)
    simp [h, blockG]
  · simp

/-- Rewriting a step's coin lane moves a word of that coin tile from that step on by its `-rot` weight against twice
the change. -/
theorem word_updU (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (htile : 0 < L.tile) (t₀ : Fin L.S) (k₀ : Fin L.tiles) (ℓ₀ : Fin 3) (J : Finset (Fin L.n)) (ω : Coins L) (v : Fin L.n → Fin (2 ^ L.e))
    (i : Fin L.m) (t : Fin L.S) (j : Fin L.n) :
    word L κ ((blockU L t₀ k₀ ℓ₀ J).upd ω v) i t j - word L κ ω i t j =
      if j ∈ J ∧ rowTile htile i = k₀ ∧ t₀ ≤ t then
        rotWeight i ℓ₀ * (2 * ((v j : ℤ) - (ω.2.2.2 t₀ k₀ ℓ₀ j : ℤ))) else 0 := by
  rw [word_sub, Finset.sum_eq_single t₀]
  · rw [coinTerm_sub_of_tags ((blockU L t₀ k₀ ℓ₀ J).upd ω v) ω rfl rfl rfl]
    unfold laneTerm
    simp only [dite_eq_left (tile_lt htile i)]
    by_cases ht : t₀ ≤ t
    · rw [ite_eq_left ht]
      by_cases hc : j ∈ J ∧ rowTile htile i = k₀
      · obtain ⟨hj, hk⟩ := hc
        rw [ite_eq_left ⟨hj, hk, ht⟩]
        simp only [rowTile] at hk
        subst hk
        rw [← Finset.sum_sub_distrib, Finset.sum_eq_single ℓ₀]
        · simp [blockU, Coins.s, hj]
          ring
        · intro ℓ _ hℓ
          simp [blockU, Coins.s, hℓ]
        · simp
      · rw [ite_eq_right (fun h => hc ⟨h.1, h.2.1⟩)]
        have : ∀ ℓ : Fin 3, ¬ ((⟨(i : ℕ) / L.tile, tile_lt htile i⟩ : Fin L.tiles) = k₀ ∧ ℓ = ℓ₀ ∧ j ∈ J) :=
          fun ℓ h => hc ⟨h.2.2, h.1⟩
        simp [blockU, Coins.s, this]
    · rw [ite_eq_right ht, ite_eq_right (fun h => ht h.2.2)]
  · intro τ _ hτ
    rw [coinTerm_sub_of_tags ((blockU L t₀ k₀ ℓ₀ J).upd ω v) ω rfl rfl rfl]
    simp [laneTerm, blockU, Coins.s, hτ]
  · simp

/-! ## Agreement through each block -/

/-- **Through the tag weights**: a form `F` that moves with column `j`'s tag weights by a nonzero tag `d j` meets a
target `y` that reads no tag weight of `J` on every column of `J` for at most a `7^−|J|` share of the coins. -/
theorem agreeG_card_le (J : Finset (Fin L.n)) (F y : Coins L → Fin L.n → ℤ) (d : Fin L.n → Fin 4 → ℤ)
    (hd : ∀ j ∈ J, ∃ q, d j q ≠ 0)
    (hF : ∀ ω v, ∀ j ∈ J, F ((blockG L J).upd ω v) j - F ω j =
      tagTerm 3 (d j) (v j) - tagTerm 3 (d j) ((blockG L J).get ω j))
    (hy : ∀ ω v, ∀ j ∈ J, y ((blockG L J).upd ω v) j = y ω j) :
    (univ.filter fun ω => ∀ j ∈ J, F ω j = y ω j).card * 7 ^ J.card ≤ Fintype.card (Coins L) := by
  classical
  have key := (blockG L J).card_le (fun _ => 3) (fun j r g => F r j + tagTerm 3 (d j) g = y r j)
    (M := 7 ^ (Fintype.card (Fin 4) - 1)) fun j hj r => by
      obtain ⟨q, hq⟩ := hd j hj
      refine le_of_eq_of_le (congrArg card (filter_congr fun g _ => ?_)) (col_card_le 3 (d j) hq (y r j - F r j))
      constructor <;> intro h <;> linarith
  have h0 : ∀ j, tagTerm 3 (d j) (fun _ : Fin 4 => (3 : Fin 7)) = 0 := fun j => by simp [tagTerm]
  have hset : (univ.filter fun ω => ∀ j ∈ J, F ω j = y ω j) = univ.filter fun ω => ∀ j ∈ J,
      F ((blockG L J).upd ω fun _ => fun _ => 3) j + tagTerm 3 (d j) ((blockG L J).get ω j) =
        y ((blockG L J).upd ω fun _ => fun _ => 3) j := by
    refine filter_congr fun ω _ => forall₂_congr fun j hj => ?_
    have h1 := hF ω (fun _ => fun _ => 3) j hj
    rw [h0, hy ω _ j hj] at *
    constructor <;> intro h <;> linarith
  rw [hset]
  have hB : Fintype.card (Fin 4 → Fin 7) = 7 ^ 3 * 7 := by
    rw [Fintype.card_fun, Fintype.card_fin, Fintype.card_fin]; norm_num
  have hM : 7 ^ (Fintype.card (Fin 4) - 1) = 7 ^ 3 := by rw [Fintype.card_fin]
  rw [hB, hM] at key
  refine Nat.le_of_mul_le_mul_left ?_ (by positivity : 0 < (7 ^ 3) ^ J.card)
  calc (7 ^ 3) ^ J.card * (_ * 7 ^ J.card) = _ * (7 ^ 3 * 7) ^ J.card := by rw [mul_pow, mul_left_comm]
    _ ≤ _ := key
    _ = _ := by ring

/-- **Through a coin lane**: a form `F` that moves with column `j`'s coin lane value by a nonzero `c j` times the change
meets a target `y` that reads none of that lane's coins in `J` on every column of `J` for at most a `2^−e|J|` share. -/
theorem agreeU_card_le (t₀ : Fin L.S) (k₀ : Fin L.tiles) (ℓ₀ : Fin 3) (J : Finset (Fin L.n))
    (F y : Coins L → Fin L.n → ℤ) (c : Fin L.n → ℤ) (hc : ∀ j ∈ J, c j ≠ 0)
    (hF : ∀ ω v, ∀ j ∈ J, F ((blockU L t₀ k₀ ℓ₀ J).upd ω v) j - F ω j =
      c j * ((v j : ℤ) - ((blockU L t₀ k₀ ℓ₀ J).get ω j : ℤ)))
    (hy : ∀ ω v, ∀ j ∈ J, y ((blockU L t₀ k₀ ℓ₀ J).upd ω v) j = y ω j) :
    (univ.filter fun ω => ∀ j ∈ J, F ω j = y ω j).card * (2 ^ L.e) ^ J.card ≤ Fintype.card (Coins L) := by
  classical
  let u₀ : Fin (2 ^ L.e) := ⟨0, Nat.two_pow_pos _⟩
  have key := (blockU L t₀ k₀ ℓ₀ J).card_le u₀ (fun j r u => F r j + c j * (u : ℤ) = y r j) (M := 1)
    fun j hj r => card_le_one.2 fun a ha b hb => by
      have ha' := (mem_filter.1 ha).2
      have hb' := (mem_filter.1 hb).2
      have : c j * (a : ℤ) = c j * (b : ℤ) := by linarith
      exact Fin.ext (by exact_mod_cast mul_left_cancel₀ (hc j hj) this)
  have hset : (univ.filter fun ω => ∀ j ∈ J, F ω j = y ω j) = univ.filter fun ω => ∀ j ∈ J,
      F ((blockU L t₀ k₀ ℓ₀ J).upd ω fun _ => u₀) j + c j * ((blockU L t₀ k₀ ℓ₀ J).get ω j : ℤ) =
        y ((blockU L t₀ k₀ ℓ₀ J).upd ω fun _ => u₀) j := by
    refine filter_congr fun ω _ => forall₂_congr fun j hj => ?_
    have h1 := hF ω (fun _ => u₀) j hj
    rw [hy ω _ j hj]
    simp only [u₀, Nat.cast_zero, zero_sub, mul_neg] at h1
    constructor <;> intro h <;> linarith
  rw [hset]
  rw [Fintype.card_fin, one_pow, mul_one] at key
  exact key

/-! ## The lead's tiles -/

/-- The columns of column block `cb`. -/
def colsOf (L : Pouw.Lead.Layout) (cb : ℕ) : Finset (Fin L.n) := univ.filter fun j => (j : ℕ) / 8 = cb

/-- A whole column block has 8 columns. -/
theorem colsOf_card {cb : ℕ} (h : 8 * (cb + 1) ≤ L.n) : (colsOf L cb).card = 8 := by
  let f : Fin 8 ↪ Fin L.n :=
    ⟨fun c => ⟨8 * cb + (c : ℕ), by have := c.isLt; omega⟩, fun a b hab => by
      simp only [Fin.mk.injEq] at hab
      exact Fin.ext (by omega)⟩
  have hf : ∀ c, ((f c : Fin L.n) : ℕ) = 8 * cb + c := fun _ => rfl
  have e : colsOf L cb = (univ : Finset (Fin 8)).map f := by
    ext j
    simp only [colsOf, mem_filter, mem_univ, true_and, mem_map]
    constructor
    · intro hj
      exact ⟨⟨(j : ℕ) % 8, by omega⟩, Fin.ext (by rw [hf, Fin.val_mk]; omega)⟩
    · rintro ⟨c, rfl⟩
      rw [hf]
      omega
  rw [e, card_map, card_univ, Fintype.card_fin]

theorem row_lt {rb : ℕ} (hrb : rb < L.rowBlocks) : 16 * rb < L.m := by
  unfold Layout.rowBlocks at hrb
  have := (Nat.le_div_iff_mul_le (by norm_num : 0 < 16)).1 (Nat.succ_le_of_lt hrb)
  omega

theorem pos_lt (j : Fin L.n) : (j : ℕ) % 8 < tileWords := by unfold tileWords; omega

/-- A tile's first row holds its block's columns: position `j mod 8` is row `16 rb`'s word in column `j`. -/
theorem tileOf_col (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) {rb cb : ℕ} (hrb : 16 * rb < L.m)
    (t : Fin L.S) {j : Fin L.n} (hj : (j : ℕ) / 8 = cb) :
    tileOf L κ ω rb cb t ⟨(j : ℕ) % 8, pos_lt j⟩ = word L κ ω ⟨16 * rb, hrb⟩ t j := by
  have h1 : 16 * rb + (j : ℕ) % 8 / 8 = 16 * rb := by omega
  have h2 : 8 * cb + (j : ℕ) % 8 % 8 = j := by have := Nat.div_add_mod (j : ℕ) 8; rw [hj] at this; omega
  unfold tileOf
  rw [dite_eq_left ⟨by simp only [h1]; exact hrb, by simp only [h2]; exact j.isLt⟩]
  congr 1 <;> exact Fin.ext (by simp only [h1, h2])

/-- A tile's word at position `p` reads no tag weight outside its column block. -/
theorem tileOf_updG (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) {J : Finset (Fin L.n)} {rb cb : ℕ}
    (hJ : ∀ j ∈ J, (j : ℕ) / 8 ≠ cb) (ω : Coins L) (v : Fin L.n → Fin 4 → Fin 7) (t : Fin L.S)
    (p : Fin tileWords) : tileOf L κ ((blockG L J).upd ω v) rb cb t p = tileOf L κ ω rb cb t p := by
  unfold tileOf
  split_ifs with h
  · have hp : (⟨8 * cb + (p : ℕ) % 8, h.2⟩ : Fin L.n) ∉ J := fun hm => hJ _ hm (by simp; omega)
    have := word_updG κ J ω v ⟨16 * rb + (p : ℕ) / 8, h.1⟩ t ⟨8 * cb + (p : ℕ) % 8, h.2⟩
    rw [ite_eq_right hp] at this
    linarith
  · rfl

theorem rowTag_sub_ne {i i' : ℕ} (hi : i < 28560) (hi' : i' < 28560) (h : i ≠ i') :
    ∃ q, (rowTag i - rowTag i') q ≠ 0 := by
  by_contra hc
  push Not at hc
  exact h (rowTag_inj hi hi' (funext fun q => sub_eq_zero.1 (hc q)))

/-- **Two tiles at one column block** whose first rows' tags differ agree on at most a `7^−|J|` share of the coins, `J`
the block's columns: the difference of the two tags against each column's tag weights. -/
theorem tile_rows_card_le (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) {rb rb' cb : ℕ} {t t' : Fin L.S}
    (hrb : rb < L.rowBlocks) (hrb' : rb' < L.rowBlocks) (hd : ∃ q, (rowTag (16 * rb) - rowTag (16 * rb')) q ≠ 0) :
    (univ.filter fun ω => tileOf L κ ω rb cb t = tileOf L κ ω rb' cb t').card * 7 ^ (colsOf L cb).card ≤
      Fintype.card (Coins L) := by
  classical
  have hr := row_lt hrb
  have hr' := row_lt hrb'
  have hJ : ∀ j ∈ colsOf L cb, (j : ℕ) / 8 = cb := fun j hj => (mem_filter.1 hj).2
  refine (Nat.mul_le_mul_right _ (card_le_card fun ω hω => ?_)).trans
    (agreeG_card_le (colsOf L cb) (fun ω j => word L κ ω ⟨16 * rb, hr⟩ t j - word L κ ω ⟨16 * rb', hr'⟩ t' j)
      (fun _ _ => 0) (fun _ => rowTag (16 * rb) - rowTag (16 * rb')) (fun _ _ => hd)
      (fun ω v j hj => by
        have a := word_updG κ (colsOf L cb) ω v ⟨16 * rb, hr⟩ t j
        have b := word_updG κ (colsOf L cb) ω v ⟨16 * rb', hr'⟩ t' j
        rw [ite_eq_left hj] at a b
        rw [tagTerm_sub, tagTerm_sub]
        linarith)
      fun _ _ _ _ => rfl)
  refine mem_filter.2 ⟨mem_univ _, fun j hj => ?_⟩
  have := congrFun (mem_filter.1 hω).2 ⟨(j : ℕ) % 8, pos_lt j⟩
  rw [tileOf_col κ ω hr t (hJ j hj), tileOf_col κ ω hr' t' (hJ j hj)] at this
  simp only [this, sub_self]

/-- **A tile and one fixed before the coins** agree on at most a `7^−|J|` share of the coins, `J` the tile's columns,
below 28,560 rows. -/
theorem tile_fixed_card_le (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (hm : L.m ≤ 28560) {rb cb : ℕ}
    (hrb : rb < L.rowBlocks) (t : Fin L.S) (y : Fin tileWords → ℤ) :
    (univ.filter fun ω => tileOf L κ ω rb cb t = y).card * 7 ^ (colsOf L cb).card ≤
      Fintype.card (Coins L) := by
  classical
  have hr := row_lt hrb
  refine (Nat.mul_le_mul_right _ (card_le_card fun ω hω => ?_)).trans
    (agreeG_card_le (colsOf L cb) (fun ω j => word L κ ω ⟨16 * rb, hr⟩ t j) (fun _ j => y ⟨(j : ℕ) % 8, pos_lt j⟩)
      (fun _ => rowTag (16 * rb)) (fun _ _ => rowTag_ne_zero (by omega))
      (fun ω v j hj => by rw [word_updG, ite_eq_left hj]) fun _ _ _ _ => rfl)
  have hω := (mem_filter.1 hω).2
  refine mem_filter.2 ⟨mem_univ _, fun j hj => ?_⟩
  rw [← tileOf_col κ ω hr t (mem_filter.1 hj).2, hω]

/-- **Two tiles at different places** agree on at most a `7^−|J|` share of the coins, `J` the first tile's columns,
below 28,560 rows and with coin lanes of at least 3 bits: by the row tag when the rows or columns differ, and by the
later step's coin lane under `-rot`'s weights when only the step does. -/
theorem tile_agree_card_le (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (hm : L.m ≤ 28560) (he : 7 ≤ 2 ^ L.e)
    (htile : 0 < L.tile) {rb rb' cb cb' : ℕ} {t t' : Fin L.S} (hrb : rb < L.rowBlocks) (hrb' : rb' < L.rowBlocks)
    (hne : (rb, cb, t) ≠ (rb', cb', t')) :
    (univ.filter fun ω => tileOf L κ ω rb cb t = tileOf L κ ω rb' cb' t').card * 7 ^ (colsOf L cb).card ≤
      Fintype.card (Coins L) := by
  classical
  have hr := row_lt hrb
  have hr' := row_lt hrb'
  set J := colsOf L cb
  have hJ : ∀ j ∈ J, (j : ℕ) / 8 = cb := fun j hj => (mem_filter.1 hj).2
  by_cases hcb : cb = cb'
  · subst hcb
    by_cases hrbe : rb = rb'
    · subst hrbe
      have htt : t ≠ t' := fun h => hne (by rw [h])
      -- only the step differs: the later step's coin lane
      have hfirst : ∀ ω ∈ univ.filter (fun ω => tileOf L κ ω rb cb t = tileOf L κ ω rb cb t'),
          ∀ j ∈ J, word L κ ω ⟨16 * rb, hr⟩ t j - word L κ ω ⟨16 * rb, hr⟩ t' j = 0 := fun ω hω j hj => by
        have := congrFun (mem_filter.1 hω).2 ⟨(j : ℕ) % 8, pos_lt j⟩
        rw [tileOf_col κ ω hr t (hJ j hj), tileOf_col κ ω hr t' (hJ j hj)] at this
        rw [this, sub_self]
      have hpow : 7 ^ J.card ≤ (2 ^ L.e) ^ J.card := Nat.pow_le_pow_left he _
      refine (Nat.mul_le_mul_left _ hpow).trans ((Nat.mul_le_mul_right _ (card_le_card fun ω hω =>
        mem_filter.2 ⟨mem_univ _, hfirst ω hω⟩)).trans ?_)
      rcases lt_or_gt_of_ne htt with hlt | hlt
      · exact agreeU_card_le t' (rowTile htile ⟨16 * rb, hr⟩) 0 J _ (fun _ _ => 0)
          (fun _ => -(rotWeight (16 * rb) 0 * 2)) (fun _ _ => by have := rotWeight_ne_zero (16 * rb) 0; omega)
          (fun ω v j hj => by
            have a := word_updU κ htile t' (rowTile htile ⟨16 * rb, hr⟩) 0 J ω v ⟨16 * rb, hr⟩ t j
            have b := word_updU κ htile t' (rowTile htile ⟨16 * rb, hr⟩) 0 J ω v ⟨16 * rb, hr⟩ t' j
            rw [ite_eq_right fun h => absurd h.2.2 (not_le.2 hlt)] at a
            rw [ite_eq_left ⟨hj, rfl, le_rfl⟩] at b
            have hg : (((blockU L t' (rowTile htile ⟨16 * rb, hr⟩) 0 J).get ω j : ℕ) : ℤ) =
                ((ω.2.2.2 t' (rowTile htile ⟨16 * rb, hr⟩) 0 j : ℕ) : ℤ) := rfl
            rw [hg]
            linear_combination a - b)
          fun _ _ _ _ => rfl
      · exact agreeU_card_le t (rowTile htile ⟨16 * rb, hr⟩) 0 J _ (fun _ _ => 0)
          (fun _ => rotWeight (16 * rb) 0 * 2) (fun _ _ => by have := rotWeight_ne_zero (16 * rb) 0; omega)
          (fun ω v j hj => by
            have a := word_updU κ htile t (rowTile htile ⟨16 * rb, hr⟩) 0 J ω v ⟨16 * rb, hr⟩ t j
            have b := word_updU κ htile t (rowTile htile ⟨16 * rb, hr⟩) 0 J ω v ⟨16 * rb, hr⟩ t' j
            rw [ite_eq_left ⟨hj, rfl, le_rfl⟩] at a
            rw [ite_eq_right fun h => absurd h.2.2 (not_le.2 hlt)] at b
            have hg : (((blockU L t (rowTile htile ⟨16 * rb, hr⟩) 0 J).get ω j : ℕ) : ℤ) =
                ((ω.2.2.2 t (rowTile htile ⟨16 * rb, hr⟩) 0 j : ℕ) : ℤ) := rfl
            rw [hg]
            linear_combination a - b)
          fun _ _ _ _ => rfl
    · -- the rows differ, the columns don't: the two row tags' difference
      exact tile_rows_card_le κ hrb hrb' (rowTag_sub_ne (by omega) (by omega) (by omega))
  · -- the columns differ: the row tag against a target that reads no tag weight of the block
    refine (Nat.mul_le_mul_right _ (card_le_card fun ω hω => ?_)).trans
      (agreeG_card_le J (fun ω j => word L κ ω ⟨16 * rb, hr⟩ t j)
        (fun ω j => tileOf L κ ω rb' cb' t' ⟨(j : ℕ) % 8, pos_lt j⟩)
        (fun _ => rowTag (16 * rb)) (fun _ _ => rowTag_ne_zero (by omega))
        (fun ω v j hj => by rw [word_updG, ite_eq_left hj])
        fun ω v j _ => tileOf_updG κ (fun j' hj' h => hcb ((hJ j' hj').symm.trans h)) ω v t' _)
    refine mem_filter.2 ⟨mem_univ _, fun j hj => ?_⟩
    rw [← tileOf_col κ ω hr t (hJ j hj)]
    exact congrFun (mem_filter.1 hω).2 _

end Pouw.SecurityProofs.Lead.Tags
