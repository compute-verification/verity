import Definitions.Pouw.Lead.Layout
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.FinCases
import Proofs.Pouw.Lead.PStar.Marginal
import Proofs.Pouw.Lead.PStar.Tile

/-!
# An MMA tile's `p*` over a call's coins (proofs)

At a step `t ≥ 1`, `Pouw.Lead.word` adds the step's `coinTerm` to what the data and the earlier steps fixed. On a
16 × 8 MMA tile inside the call, that term reads only the tile's part of the step's coins (`TilePart`): its 16 rows'
bits and, in its coin tile, its 8 columns' three lanes, `2^16 · 2^168 = 2^184` values. Overwriting the part
(`tileSet`) moves the tile's words by the term alone (`tileWords_set`). Under the encoding `s = 2u − 127` the term is
`Tile`'s step (`tileTerm_eq`), which is injective, so by `part_fiber_le`:

* **`p*` over the call** (`call_tile_pstar`): whatever the data, a value fixed before the coins is the tile's words at
  step `t` on at most a `2^−184` share of the call's coin values.
-/

namespace Pouw.SecurityProofs.Lead.PStar

open Finset Pouw.Lead

/-- A tile's part of a step's coins: a bit for each of its 16 rows and three lanes for each of its 8 columns. -/
abbrev TilePart (L : Layout) : Type := (Fin 16 → Fin 2) × (Fin 8 → Fin 3 → Fin (2 ^ L.e))

theorem card_tilePart {L : Layout} (he : L.e = 7) : Fintype.card (TilePart L) = 2 ^ 184 := by
  simp only [TilePart, Fintype.card_prod, Fintype.card_fun, Fintype.card_fin, he]
  norm_num

/-- Row `r` of row block `rb`: the call's row `16·rb + r`. -/
def tileRow (L : Layout) {rb : ℕ} (hrb : 16 * rb + 16 ≤ L.m) (r : Fin 16) : Fin L.m := ⟨16 * rb + r, by omega⟩

/-- Column `q` of column block `cb`: the call's column `8·cb + q`. -/
def tileCol (L : Layout) {cb : ℕ} (hcb : 8 * cb + 8 ≤ L.n) (q : Fin 8) : Fin L.n := ⟨8 * cb + q, by omega⟩

/-- The 64-row coin tile holding row block `rb`. -/
def coinTileOf (L : Layout) {rb : ℕ} (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) : Fin L.tiles :=
  ⟨16 * rb / 64, by unfold Layout.tiles; rw [htile]; omega⟩

variable {L : Layout} {rb cb : ℕ}

/-- The tile's part of step `t`'s coins. -/
def tileGet (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S) (ω : Coins L) :
    TilePart L :=
  (fun r => ω.1 (tileRow L hrb r) t, fun q ℓ => ω.2.2.2 t (coinTileOf L htile hrb) ℓ (tileCol L hcb q))

/-- The coins with the tile's part of step `t` (in column block `cb`) overwritten by `d`. -/
def tileSet (cb : ℕ) (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (t : Fin L.S) (ω : Coins L) (d : TilePart L) :
    Coins L :=
  (fun i τ => if h : τ = t ∧ 16 * rb ≤ (i : ℕ) ∧ (i : ℕ) < 16 * rb + 16 then d.1 ⟨(i : ℕ) - 16 * rb, by omega⟩
      else ω.1 i τ,
    ω.2.1, ω.2.2.1,
    fun τ k ℓ j => if h : τ = t ∧ k = coinTileOf L htile hrb ∧ 8 * cb ≤ (j : ℕ) ∧ (j : ℕ) < 8 * cb + 8 then
      d.2 ⟨(j : ℕ) - 8 * cb, by omega⟩ ℓ else ω.2.2.2 τ k ℓ j)

theorem tileGet_set (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (ω : Coins L) (d : TilePart L) : tileGet htile hrb hcb t (tileSet cb htile hrb t ω d) = d := by
  refine Prod.ext (funext fun r => ?_) (funext fun q => funext fun ℓ => ?_)
  · dsimp only [tileGet, tileSet]
    rw [dite_eq_left ⟨rfl, Nat.le_add_right _ _, Nat.add_lt_add_left r.isLt _⟩]
    exact congrArg d.1 (Fin.ext (Nat.add_sub_cancel_left ..))
  · dsimp only [tileGet, tileSet]
    rw [dite_eq_left ⟨rfl, rfl, Nat.le_add_right _ _, Nat.add_lt_add_left q.isLt _⟩]
    exact congrFun (congrArg d.2 (Fin.ext (Nat.add_sub_cancel_left ..))) ℓ

theorem tileSet_set (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (t : Fin L.S) (ω : Coins L)
    (d d' : TilePart L) : tileSet cb htile hrb t (tileSet cb htile hrb t ω d) d' = tileSet cb htile hrb t ω d' := by
  refine Prod.ext (funext fun i => funext fun τ => ?_) (Prod.ext rfl (Prod.ext rfl ?_))
  · simp only [tileSet]
    split_ifs <;> rfl
  · refine funext fun τ => funext fun k => funext fun ℓ => funext fun j => ?_
    simp only [tileSet]
    split_ifs <;> rfl

theorem tileSet_get (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (ω : Coins L) : tileSet cb htile hrb t ω (tileGet htile hrb hcb t ω) = ω := by
  refine Prod.ext (funext fun i => funext fun τ => ?_) (Prod.ext rfl (Prod.ext rfl ?_))
  · simp only [tileSet, tileGet]
    split_ifs with h
    · obtain ⟨rfl, h1, h2⟩ := h
      exact congrArg (fun i' => ω.1 i' τ) (Fin.ext (by simp only [tileRow]; omega))
    · rfl
  · refine funext fun τ => funext fun k => funext fun ℓ => funext fun j => ?_
    simp only [tileSet, tileGet]
    split_ifs with h
    · obtain ⟨rfl, rfl, h1, h2⟩ := h
      exact congrArg (fun j' => ω.2.2.2 τ (coinTileOf L htile hrb) ℓ j') (Fin.ext (by simp only [tileCol]; omega))
    · rfl

/-- The tile's step-`t` coin term at its part `d`: row `r`'s tag lane `1 + b_r` and its lanes under `-rot`'s weights. -/
def tileTerm (L : Layout) (rb : ℕ) (d : TilePart L) : Fin 16 → Fin 8 → ℤ := fun r q =>
  (1 + ((d.1 r : ℕ) : ℤ)) + ∑ ℓ : Fin 3, rotWeight (16 * rb + r) ℓ * (2 * ((d.2 q ℓ : ℕ) : ℤ) - (2 ^ L.e - 1))

/-- The coin term on the tile reads only the tile's part. -/
theorem coinTerm_tile (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (ht : (t : ℕ) ≠ 0) (ω : Coins L) (r : Fin 16) (q : Fin 8) :
    coinTerm L ω (tileRow L hrb r) t (tileCol L hcb q) = tileTerm L rb (tileGet htile hrb hcb t ω) r q := by
  have hk : ((tileRow L hrb r : ℕ)) / L.tile < L.tiles := by
    simp only [tileRow, Layout.tiles, htile]; omega
  simp only [coinTerm, ht, ite_false, tileTerm, tileGet]
  refine congrArg _ (sum_congr rfl fun ℓ _ => ?_)
  rw [dite_eq_left hk]
  have : (⟨(tileRow L hrb r : ℕ) / L.tile, hk⟩ : Fin L.tiles) = coinTileOf L htile hrb :=
    Fin.ext (by simp only [tileRow, coinTileOf, htile]; omega)
  rw [this]
  rfl

/-- Away from step `t` the overwrite leaves the coin term. -/
theorem coinTerm_set_ne (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (t : Fin L.S) (ω : Coins L)
    (d : TilePart L) {τ : Fin L.S} (hτ : τ ≠ t) (i : Fin L.m) (j : Fin L.n) :
    coinTerm L (tileSet cb htile hrb t ω d) i τ j = coinTerm L ω i τ j := by
  simp [coinTerm, Coins.s, Coins.G, tileSet, hτ]

/-- The tile's words at step `t`. -/
def tileWords (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) : Fin 16 → Fin 8 → ℤ :=
  fun r q => word L κ ω (tileRow L hrb r) t (tileCol L hcb q)

/-- **The overwrite moves the tile's words by the term alone.** -/
theorem tileWords_set (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (ht : (t : ℕ) ≠ 0) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) (d : TilePart L) :
    tileWords hrb hcb t κ (tileSet cb htile hrb t ω d) =
      tileWords hrb hcb t κ ω - tileTerm L rb (tileGet htile hrb hcb t ω) + tileTerm L rb d := by
  funext r q
  simp only [tileWords, word, Pi.add_apply, Pi.sub_apply]
  rw [← sum_erase_add _ _ (mem_univ t), ← sum_erase_add _ _ (mem_univ t)]
  rw [sum_congr rfl fun τ hτ => by rw [coinTerm_set_ne htile hrb t ω d (ne_of_mem_erase hτ)]]
  rw [ite_eq_left le_rfl, ite_eq_left le_rfl, coinTerm_tile htile hrb hcb t ht, coinTerm_tile htile hrb hcb t ht,
    tileGet_set]
  ring

/-- A coin lane's value `2u − (2^e − 1)`. -/
def laneValue (L : Layout) (u : Fin (2 ^ L.e)) : ℤ := 2 * ((u : ℕ) : ℤ) - (2 ^ L.e - 1)

/-- The tile's part as `Tile`'s draw: the bits as integers and the lanes as their values. -/
def encodePart (L : Layout) (d : TilePart L) : (Fin 16 → ℤ) × (Fin 8 → ℤ × ℤ × ℤ) :=
  (fun r => ((d.1 r : ℕ) : ℤ), fun q => (laneValue L (d.2 q 0), laneValue L (d.2 q 1), laneValue L (d.2 q 2)))

theorem encodePart_mem (he : L.e = 7) (d : TilePart L) : encodePart L d ∈ tileDraws (Fin 16) (Fin 8) := by
  have h128 : 2 ^ L.e = 128 := by rw [he]; norm_num
  have hlane : ∀ u : Fin (2 ^ L.e), laneValue L u ∈ coins := fun u => by
    have hu : (u : ℕ) < 128 := lt_of_lt_of_eq u.isLt h128
    refine mem_coins.mpr ⟨((u : ℕ) : ℤ) - 64, by omega, by omega, ?_⟩
    rw [laneValue, show (2 : ℤ) ^ L.e = 128 by rw [he]; norm_num]
    ring
  rw [tileDraws, mem_product, Fintype.mem_piFinset, Fintype.mem_piFinset]
  refine ⟨fun r => ?_, fun q => mem_triples.mpr ⟨hlane _, hlane _, hlane _⟩⟩
  have := (d.1 r).isLt
  simp only [encodePart, mem_insert, mem_singleton]
  omega

theorem encodePart_injective : Function.Injective (encodePart L) := by
  intro d d' h
  simp only [encodePart, Prod.mk.injEq, laneValue] at h
  refine Prod.ext (funext fun r => Fin.ext ?_) (funext fun q => funext fun ℓ => Fin.ext ?_)
  · have := congrFun h.1 r
    omega
  · have hq := congrFun h.2 q
    simp only [Prod.mk.injEq] at hq
    fin_cases ℓ <;> simp only [Fin.zero_eta, Fin.mk_one, Fin.reduceFinMk] <;> omega

/-- **The term is `Tile`'s step** under the encoding, with the tag lane's `1` as the constant. -/
theorem tileTerm_eq (d : TilePart L) :
    tileTerm L rb d = tileStep (mmaRow rb) (fun _ _ => 1) (encodePart L d) := by
  funext r q
  simp only [tileTerm, tileStep, rotLane, rotSum, evenSum, oddSum, mmaRow, encodePart, laneValue,
    Fin.sum_univ_three, rotWeight, Fin.val_zero, Fin.val_one, Fin.val_two]
  by_cases h : (16 * rb + (r : ℕ)) % 2 = 0
  · simp only [h, ite_true, show ¬ (1 : ℕ) = 0 by omega, show ¬ (2 : ℕ) = 0 by omega, ite_false]
    ring
  · have h' : (16 * rb + (r : ℕ)) % 2 = 1 := by omega
    simp only [h, ite_false]
    simp only [h', ite_true, ite_false, show ¬ (0 : ℕ) = 1 by omega, show ¬ (2 : ℕ) = 1 by omega]
    ring

theorem tileTerm_injective (he : L.e = 7) : Function.Injective (tileTerm L rb) := by
  intro d d' h
  rw [tileTerm_eq, tileTerm_eq] at h
  have h₀ : mmaRow rb 0 % 2 = 0 := by simp only [mmaRow, Fin.val_zero]; omega
  have h₁ : mmaRow rb 1 % 2 = 1 := by simp only [mmaRow, Fin.val_one]; omega
  exact encodePart_injective (tileStep_injOn (mmaRow rb) h₀ h₁ _ (encodePart_mem he d) (encodePart_mem he d') h)

/-- **An MMA tile's `p*` over the call's coins**: at a step `t ≥ 1`, for any data `κ` and any value `y` fixed before
the coins, the tile's words at step `t` are `y` on at most a `2^−184` share of the call's coin values. -/
theorem call_tile_pstar (he : L.e = 7) (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n)
    (t : Fin L.S) (ht : (t : ℕ) ≠ 0) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (y : Fin 16 → Fin 8 → ℤ) :
    2 ^ 184 * (univ.filter fun ω : Coins L => tileWords hrb hcb t κ ω = y).card ≤ Fintype.card (Coins L) := by
  classical
  have h := part_fiber_le (tileGet htile hrb hcb t) (tileSet cb htile hrb t) (tileGet_set htile hrb hcb t)
    (tileSet_set htile hrb t) (tileSet_get htile hrb hcb t) (tileWords hrb hcb t κ) (tileTerm L rb)
    (tileWords_set htile hrb hcb t ht κ) (P := 1)
    (fun v => card_le_one.mpr fun d hd d' hd' =>
      tileTerm_injective he ((mem_filter.mp hd).2.trans (mem_filter.mp hd').2.symm)) y
  rwa [card_tilePart he, one_mul] at h

end Pouw.SecurityProofs.Lead.PStar
