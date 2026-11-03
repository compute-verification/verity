import Definitions.Pouw.Lead.Layout
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.FinCases
import Proofs.Pouw.Lead.PStar.Marginal
import Proofs.Pouw.Lead.PStar.Tile

/-!
# An MMA tile's coin lanes over a call's coins (proofs)

At every step `t`, `Pouw.Lead.word` adds the step's `coinTerm` to what the data and the earlier steps fixed, and the
term's coin lanes (`laneTerm`) on a 16 × 8 MMA tile inside the call read only the tile's lanes (`TileLanes`): in its
coin tile, its 8 columns' three lanes, `2^168` values. Overwriting them (`tileSet`) moves the tile's words by the lanes'
term alone (`stepTile_set`). A column's even and odd rows fix its three lanes (`tileTerm_col_eq`, by `rotPair_inj`), so
the term is injective, and by `part_fiber_le`:

* **A tile against a fixed value** (`call_tile_pstar`): whatever the data, a value fixed before the coins is the tile's
  words at step `t` on at most a `2^−168` share of the call's coin values.
-/

namespace Pouw.SecurityProofs.Lead.PStar

open Finset Pouw.Lead

/-- A tile's coin lanes at a step: in its coin tile, three for each of its 8 columns. -/
abbrev TileLanes (L : Layout) : Type := Fin 8 → Fin 3 → Fin (2 ^ L.e)

theorem card_tileLanes {L : Layout} (he : L.e = 7) : Fintype.card (TileLanes L) = 2 ^ 168 := by
  simp only [TileLanes, Fintype.card_fun, Fintype.card_fin, he]
  norm_num

/-- Row `r` of row block `rb`: the call's row `16·rb + r`. -/
def tileRow (L : Layout) {rb : ℕ} (hrb : 16 * rb + 16 ≤ L.m) (r : Fin 16) : Fin L.m := ⟨16 * rb + r, by omega⟩

/-- Column `q` of column block `cb`: the call's column `8·cb + q`. -/
def tileCol (L : Layout) {cb : ℕ} (hcb : 8 * cb + 8 ≤ L.n) (q : Fin 8) : Fin L.n := ⟨8 * cb + q, by omega⟩

/-- The 64-row coin tile holding row block `rb`. -/
def coinTileOf (L : Layout) {rb : ℕ} (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) : Fin L.tiles :=
  ⟨16 * rb / 64, by unfold Layout.tiles; rw [htile]; omega⟩

variable {L : Layout} {rb cb : ℕ}

/-- The tile's lanes at step `t`. -/
def tileGet (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S) (ω : Coins L) :
    TileLanes L :=
  fun q ℓ => ω.2.2.2 t (coinTileOf L htile hrb) ℓ (tileCol L hcb q)

/-- The coins with the tile's lanes at step `t` (in column block `cb`) overwritten by `d`. -/
def tileSet (cb : ℕ) (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (t : Fin L.S) (ω : Coins L) (d : TileLanes L) :
    Coins L :=
  (ω.1, ω.2.1, ω.2.2.1,
    fun τ k ℓ j => if h : τ = t ∧ k = coinTileOf L htile hrb ∧ 8 * cb ≤ (j : ℕ) ∧ (j : ℕ) < 8 * cb + 8 then
      d ⟨(j : ℕ) - 8 * cb, by omega⟩ ℓ else ω.2.2.2 τ k ℓ j)

theorem tileGet_set (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (ω : Coins L) (d : TileLanes L) : tileGet htile hrb hcb t (tileSet cb htile hrb t ω d) = d := by
  refine funext fun q => funext fun ℓ => ?_
  dsimp only [tileGet, tileSet]
  rw [dite_eq_left ⟨rfl, rfl, Nat.le_add_right _ _, Nat.add_lt_add_left q.isLt _⟩]
  exact congrFun (congrArg d (Fin.ext (Nat.add_sub_cancel_left ..))) ℓ

theorem tileSet_set (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (t : Fin L.S) (ω : Coins L)
    (d d' : TileLanes L) : tileSet cb htile hrb t (tileSet cb htile hrb t ω d) d' = tileSet cb htile hrb t ω d' := by
  refine Prod.ext rfl (Prod.ext rfl (Prod.ext rfl ?_))
  refine funext fun τ => funext fun k => funext fun ℓ => funext fun j => ?_
  simp only [tileSet]
  split_ifs <;> rfl

theorem tileSet_get (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (ω : Coins L) : tileSet cb htile hrb t ω (tileGet htile hrb hcb t ω) = ω := by
  refine Prod.ext rfl (Prod.ext rfl (Prod.ext rfl ?_))
  refine funext fun τ => funext fun k => funext fun ℓ => funext fun j => ?_
  simp only [tileSet, tileGet]
  split_ifs with h
  · obtain ⟨rfl, rfl, h1, h2⟩ := h
    exact congrArg (fun j' => ω.2.2.2 τ (coinTileOf L htile hrb) ℓ j') (Fin.ext (by simp only [tileCol]; omega))
  · rfl

/-- A coin lane's value `2u − (2^e − 1)`. -/
def laneValue (L : Layout) (u : Fin (2 ^ L.e)) : ℤ := 2 * ((u : ℕ) : ℤ) - (2 ^ L.e - 1)

/-- The tile's coin lanes at its lanes `d`: row `r`'s three lanes under `-rot`'s weights. -/
def tileTerm (L : Layout) (rb : ℕ) (d : TileLanes L) : Fin 16 → Fin 8 → ℤ := fun r q =>
  ∑ ℓ : Fin 3, rotWeight (16 * rb + r) ℓ * laneValue L (d q ℓ)

/-- The coin lanes on the tile read only the tile's lanes. -/
theorem laneTerm_tile (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (ω : Coins L) (r : Fin 16) (q : Fin 8) :
    laneTerm L ω (tileRow L hrb r) t (tileCol L hcb q) = tileTerm L rb (tileGet htile hrb hcb t ω) r q := by
  have hk : ((tileRow L hrb r : ℕ)) / L.tile < L.tiles := by
    simp only [tileRow, Layout.tiles, htile]; omega
  simp only [laneTerm, tileTerm, tileGet]
  refine sum_congr rfl fun ℓ _ => ?_
  rw [dite_eq_left hk]
  have : (⟨(tileRow L hrb r : ℕ) / L.tile, hk⟩ : Fin L.tiles) = coinTileOf L htile hrb :=
    Fin.ext (by simp only [tileRow, coinTileOf, htile]; omega)
  rw [this]
  rfl

/-- Away from step `t` the overwrite leaves the coin lanes. -/
theorem laneTerm_set_ne (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (t : Fin L.S) (ω : Coins L)
    (d : TileLanes L) {τ : Fin L.S} (hτ : τ ≠ t) (i : Fin L.m) (j : Fin L.n) :
    laneTerm L (tileSet cb htile hrb t ω d) i τ j = laneTerm L ω i τ j := by
  simp [laneTerm, Coins.s, tileSet, hτ]

/-- The overwrite moves the coin term by the coin lanes alone. -/
theorem coinTerm_set (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (t : Fin L.S) (ω : Coins L)
    (d : TileLanes L) (i : Fin L.m) (τ : Fin L.S) (j : Fin L.n) :
    coinTerm L (tileSet cb htile hrb t ω d) i τ j =
      coinTerm L ω i τ j - laneTerm L ω i τ j + laneTerm L (tileSet cb htile hrb t ω d) i τ j := by
  simp only [coinTerm, Coins.G]
  dsimp only [tileSet]
  ring

/-- The tile's words at step `t`. -/
def stepTile (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) : Fin 16 → Fin 8 → ℤ :=
  fun r q => word L κ ω (tileRow L hrb r) t (tileCol L hcb q)

/-- **The overwrite moves the tile's words by the lanes' term alone.** -/
theorem stepTile_set (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) (d : TileLanes L) :
    stepTile hrb hcb t κ (tileSet cb htile hrb t ω d) =
      stepTile hrb hcb t κ ω - tileTerm L rb (tileGet htile hrb hcb t ω) + tileTerm L rb d := by
  funext r q
  simp only [stepTile, word, Pi.add_apply, Pi.sub_apply]
  rw [← sum_erase_add _ _ (mem_univ t), ← sum_erase_add _ _ (mem_univ t)]
  rw [sum_congr rfl fun τ hτ => by
    rw [coinTerm_set htile hrb t ω d, laneTerm_set_ne htile hrb t ω d (ne_of_mem_erase hτ), sub_add_cancel]]
  rw [ite_eq_left le_rfl, ite_eq_left le_rfl, coinTerm_set htile hrb t ω d, laneTerm_tile htile hrb hcb t,
    laneTerm_tile htile hrb hcb t, tileGet_set]
  ring

/-- The tile's lanes in a column as `Lanes`' triple. -/
def encodeLanes (L : Layout) (d : TileLanes L) (q : Fin 8) : ℤ × ℤ × ℤ :=
  (laneValue L (d q 0), laneValue L (d q 1), laneValue L (d q 2))

theorem encodeLanes_mem (he : L.e = 7) (d : TileLanes L) (q : Fin 8) : encodeLanes L d q ∈ triples := by
  have h128 : 2 ^ L.e = 128 := by rw [he]; norm_num
  have hlane : ∀ u : Fin (2 ^ L.e), laneValue L u ∈ coins := fun u => by
    have hu : (u : ℕ) < 128 := lt_of_lt_of_eq u.isLt h128
    refine mem_coins.mpr ⟨((u : ℕ) : ℤ) - 64, by omega, by omega, ?_⟩
    rw [laneValue, show (2 : ℤ) ^ L.e = 128 by rw [he]; norm_num]
    ring
  exact mem_triples.mpr ⟨hlane _, hlane _, hlane _⟩

theorem encodeLanes_inj {d d' : TileLanes L} {q : Fin 8} (h : encodeLanes L d q = encodeLanes L d' q) : d q = d' q := by
  simp only [encodeLanes, Prod.mk.injEq, laneValue] at h
  refine funext fun ℓ => Fin.ext ?_
  fin_cases ℓ <;> simp only [Fin.zero_eta, Fin.mk_one, Fin.reduceFinMk] <;> omega

/-- **The term is `Lanes`' coin sum** of the row's parity on the column's triple. -/
theorem tileTerm_eq (d : TileLanes L) (r : Fin 16) (q : Fin 8) :
    tileTerm L rb d r q = rotSum (16 * rb + r) (encodeLanes L d q) := by
  simp only [tileTerm, rotSum, evenSum, oddSum, encodeLanes, Fin.sum_univ_three, rotWeight, Fin.val_zero,
    Fin.val_one, Fin.val_two]
  by_cases h : (16 * rb + (r : ℕ)) % 2 = 0
  · simp only [h, ite_true, show ¬ (1 : ℕ) = 0 by omega, show ¬ (2 : ℕ) = 0 by omega, ite_false]
    ring
  · have h' : (16 * rb + (r : ℕ)) % 2 = 1 := by omega
    simp only [h, ite_false]
    simp only [h', ite_true, ite_false, show ¬ (0 : ℕ) = 1 by omega, show ¬ (2 : ℕ) = 1 by omega]
    ring

/-- **An even and an odd row fix a column's lanes.** -/
theorem tileTerm_col_eq (he : L.e = 7) {d d' : TileLanes L} {q : Fin 8} {r r' : Fin 16} (hr : (r : ℕ) % 2 = 0)
    (hr' : (r' : ℕ) % 2 = 1) (h : tileTerm L rb d r q = tileTerm L rb d' r q)
    (h' : tileTerm L rb d r' q = tileTerm L rb d' r' q) : d q = d' q := by
  rw [tileTerm_eq, tileTerm_eq] at h h'
  have hr₀ : (16 * rb + (r : ℕ)) % 2 = 0 := by omega
  have hr₁ : (16 * rb + (r' : ℕ)) % 2 = 1 := by omega
  simp only [rotSum, hr₀, hr₁, ite_true, show ¬ (1 : ℕ) = 0 by omega, ite_false] at h h'
  exact encodeLanes_inj (rotPair_inj (encodeLanes_mem he d q) (encodeLanes_mem he d' q) h h')

theorem tileTerm_injective (he : L.e = 7) : Function.Injective (tileTerm L rb) := fun _ _ h =>
  funext fun q => tileTerm_col_eq (r := 0) (r' := 1) he rfl rfl (congrFun (congrFun h 0) q)
    (congrFun (congrFun h 1) q)

/-- **An MMA tile against a fixed value**: at any step `t`, for any data `κ` and any value `y` fixed before the coins,
the tile's words at step `t` are `y` on at most a `2^−168` share of the call's coin values. -/
theorem call_tile_pstar (he : L.e = 7) (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n)
    (t : Fin L.S) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (y : Fin 16 → Fin 8 → ℤ) :
    2 ^ 168 * (univ.filter fun ω : Coins L => stepTile hrb hcb t κ ω = y).card ≤ Fintype.card (Coins L) := by
  classical
  have h := part_fiber_le (tileGet htile hrb hcb t) (tileSet cb htile hrb t) (tileGet_set htile hrb hcb t)
    (tileSet_set htile hrb t) (tileSet_get htile hrb hcb t) (stepTile hrb hcb t κ) (tileTerm L rb)
    (stepTile_set htile hrb hcb t κ) (P := 1)
    (fun v => card_le_one.mpr fun d hd d' hd' =>
      tileTerm_injective he ((mem_filter.mp hd).2.trans (mem_filter.mp hd').2.symm)) y
  rwa [card_tileLanes he, one_mul] at h

end Pouw.SecurityProofs.Lead.PStar
