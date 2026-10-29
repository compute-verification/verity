import Definitions.Pouw.Lead.Model
import Proofs.Pouw.Lead.PStar.Call

/-!
# An MMA tile against the certifier's coin stream (proofs)

A streamed tile holds raw coins, so it can move with the tile's own coin lanes. Each coin is streamed once at most
(`CoinStream.once`), so at most 24 of a streamed tile's words hold one of the tile's 24 lanes (`card_lanePos_le`), and
at most 3 of the tile's columns have every even row's word, or every odd row's, holding one (`card_dirty_le`). In each
of the at least 5 other columns an even and an odd row compare the column's lanes with streamed words that don't move
with them, which fixes the lanes (`tileTerm_col_eq`). So the tile less the streamed tile moves with the lanes by a term
whose fibers have at most `(2^21)^3` values (`gapTerm_fiber_le`), and by `part_fiber_le`:

* **A tile against a streamed tile** (`call_tile_stream`): whatever the data and the stream's arrangement, at any step
  the tile's words are the stream's tile `c` on at most a `2^−105` share of the call's coin values.
-/

namespace Pouw.SecurityProofs.Lead.PStar

open Finset Pouw.Lead

/-- Row `r` and column `q` of a tile, the word at `8 r + q`. -/
def tilePos (r : Fin 16) (q : Fin 8) : Fin tileWords := ⟨8 * r + q, by unfold tileWords; omega⟩

theorem tilePos_inj {r r' : Fin 16} {q q' : Fin 8} (h : tilePos r q = tilePos r' q') : r = r' ∧ q = q' := by
  have h' := congrArg Fin.val h
  simp only [tilePos] at h'
  exact ⟨Fin.ext (by omega), Fin.ext (by omega)⟩

variable {L : Pouw.Lead.Layout} {rb cb : ℕ}

/-- The coin index of the tile's lane `ℓ` in its column `q` at step `t`. -/
def laneIdx (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S) (q : Fin 8)
    (ℓ : Fin 3) : CoinIdx L :=
  .inr (.inr (.inr (t, coinTileOf L htile hrb, ℓ, tileCol L hcb q)))

/-- A coin's value at the tile's lanes `d` if it is one of them, else zero. -/
def partVal (cb : ℕ) (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (t : Fin L.S) (d : TileLanes L) :
    CoinIdx L → ℤ
  | .inr (.inr (.inr (τ, k, ℓ, j))) =>
    if h : τ = t ∧ k = coinTileOf L htile hrb ∧ 8 * cb ≤ (j : ℕ) ∧ (j : ℕ) < 8 * cb + 8 then
      laneValue L (d ⟨(j : ℕ) - 8 * cb, by omega⟩ ℓ) else 0
  | _ => 0

/-- **The overwrite moves a coin by its lane's values alone.** -/
theorem coinAt_set (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (ω : Coins L) (d : TileLanes L) (x : CoinIdx L) :
    coinAt (tileSet cb htile hrb t ω d) x =
      coinAt ω x - partVal cb htile hrb t (tileGet htile hrb hcb t ω) x + partVal cb htile hrb t d x := by
  rcases x with ⟨i, τ⟩ | ⟨i, q⟩ | ⟨q, j⟩ | ⟨τ, k, ℓ, j⟩
  · simp [coinAt, partVal, tileSet]
  · simp [coinAt, partVal, tileSet]
  · simp [coinAt, partVal, tileSet, Coins.G]
  · simp only [coinAt, partVal, Coins.s, tileSet, laneValue, tileGet]
    split_ifs with h
    · obtain ⟨rfl, rfl, h1, h2⟩ := h
      have : tileCol L hcb ⟨(j : ℕ) - 8 * cb, by omega⟩ = j := Fin.ext (by simp only [tileCol]; omega)
      rw [this]
      ring
    · ring

/-- The stream's lanes on the tile: each streamed word's value at the lanes `d` if it holds one, else zero. -/
def streamPart (cb : ℕ) (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (t : Fin L.S) (stream : CoinStream L)
    (c : ℕ) (d : TileLanes L) : Fin 16 → Fin 8 → ℤ :=
  fun r q => (stream.ι c (tilePos r q)).elim 0 (partVal cb htile hrb t d)

/-- The stream's tile `c` at `ω`, row by row. -/
def streamTile (stream : CoinStream L) (c : ℕ) (ω : Coins L) : Fin 16 → Fin 8 → ℤ :=
  fun r q => stream.val ω c (tilePos r q)

theorem streamTile_set (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (stream : CoinStream L) (c : ℕ) (ω : Coins L) (d : TileLanes L) :
    streamTile stream c (tileSet cb htile hrb t ω d) = streamTile stream c ω -
      streamPart cb htile hrb t stream c (tileGet htile hrb hcb t ω) + streamPart cb htile hrb t stream c d := by
  funext r q
  simp only [streamTile, CoinStream.val, streamPart, Pi.add_apply, Pi.sub_apply]
  rcases stream.ι c (tilePos r q) with _ | x
  · simp
  · simp only [Option.elim]
    exact coinAt_set htile hrb hcb t ω d x

/-- The term the tile less the streamed tile moves by: the tile's lanes less the stream's. -/
def gapTerm (cb : ℕ) (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (t : Fin L.S) (stream : CoinStream L) (c : ℕ)
    (d : TileLanes L) : Fin 16 → Fin 8 → ℤ :=
  tileTerm L rb d - streamPart cb htile hrb t stream c d

/-- The streamed word at row `r` and column `q` holds one of the tile's lanes. -/
def LanePos (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (stream : CoinStream L) (c : ℕ) (x : Fin 16 × Fin 8) : Prop :=
  ∃ q' ℓ', stream.ι c (tilePos x.1 x.2) = some (laneIdx htile hrb hcb t q' ℓ')

/-- Where the streamed word holds none of the tile's lanes, the stream's part is zero. -/
theorem streamPart_eq_zero (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (stream : CoinStream L) (c : ℕ) {r : Fin 16} {q : Fin 8} (h : ¬ LanePos htile hrb hcb t stream c (r, q))
    (d : TileLanes L) : streamPart cb htile hrb t stream c d r q = 0 := by
  simp only [streamPart]
  rcases hx : stream.ι c (tilePos r q) with _ | x
  · rfl
  · simp only [Option.elim]
    rcases x with ⟨i, τ⟩ | ⟨i, q⟩ | ⟨q, j⟩ | ⟨τ, k, ℓ, j⟩
    · rfl
    · rfl
    · rfl
    · simp only [partVal]
      split_ifs with hc
      · obtain ⟨hτ, hk, h1, h2⟩ := hc
        refine absurd ⟨⟨(j : ℕ) - 8 * cb, by omega⟩, ℓ, ?_⟩ h
        have hj : tileCol L hcb ⟨(j : ℕ) - 8 * cb, by omega⟩ = j := Fin.ext (by simp only [tileCol]; omega)
        rw [hx, laneIdx, hτ, hk, hj]
      · rfl

open Classical in
/-- **At most 24 streamed words hold the tile's lanes**: each lane is streamed once at most. -/
theorem card_lanePos_le (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (stream : CoinStream L) (c : ℕ) : (univ.filter (LanePos htile hrb hcb t stream c)).card ≤ 24 := by
  have hsub : univ.filter (LanePos htile hrb hcb t stream c) ⊆ (univ : Finset (Fin 8 × Fin 3)).biUnion fun y =>
      univ.filter fun x : Fin 16 × Fin 8 => stream.ι c (tilePos x.1 x.2) = some (laneIdx htile hrb hcb t y.1 y.2) := by
    intro x hx
    obtain ⟨q', ℓ', h⟩ := (mem_filter.mp hx).2
    exact mem_biUnion.mpr ⟨(q', ℓ'), mem_univ _, mem_filter.mpr ⟨mem_univ _, h⟩⟩
  refine (card_le_card hsub).trans (card_biUnion_le.trans ?_)
  have hone : ∀ y : Fin 8 × Fin 3, (univ.filter fun x : Fin 16 × Fin 8 =>
      stream.ι c (tilePos x.1 x.2) = some (laneIdx htile hrb hcb t y.1 y.2)).card ≤ 1 := fun y =>
    card_le_one.mpr fun x hx x' hx' => by
      have := (stream.once c _ c _ _ (mem_filter.mp hx).2 (mem_filter.mp hx').2).2
      obtain ⟨h1, h2⟩ := tilePos_inj this
      exact Prod.ext h1 h2
  calc _ ≤ ∑ _y : Fin 8 × Fin 3, 1 := sum_le_sum fun y _ => hone y
    _ = 24 := by simp

/-- A column whose even rows, or odd rows, all hold the tile's lanes in the stream. -/
def Dirty (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (stream : CoinStream L) (c : ℕ) (q : Fin 8) : Prop :=
  (∀ r : Fin 16, (r : ℕ) % 2 = 0 → LanePos htile hrb hcb t stream c (r, q)) ∨
    ∀ r : Fin 16, (r : ℕ) % 2 = 1 → LanePos htile hrb hcb t stream c (r, q)

theorem card_parity (b : ℕ) (hb : b < 2) : ((univ : Finset (Fin 16)).filter fun r : Fin 16 => (r : ℕ) % 2 = b).card = 8 := by
  interval_cases b <;> decide

open Classical in
/-- **At most 3 dirty columns**: each takes 8 of the at most 24 streamed words holding lanes. -/
theorem card_dirty_le (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (stream : CoinStream L) (c : ℕ) : (univ.filter (Dirty htile hrb hcb t stream c)).card ≤ 3 := by
  set D := univ.filter (LanePos htile hrb hcb t stream c)
  have hcol : ∀ q ∈ univ.filter (Dirty htile hrb hcb t stream c), 8 ≤ (D.filter fun x => x.2 = q).card := by
    intro q hq
    have hq := (mem_filter.mp hq).2
    have key : ∀ b < 2, (∀ r : Fin 16, (r : ℕ) % 2 = b → LanePos htile hrb hcb t stream c (r, q)) →
        8 ≤ (D.filter fun x => x.2 = q).card := fun b hb hall => by
      have hinj : Function.Injective fun r : Fin 16 => (r, q) := fun r r' h => (Prod.mk.inj h).1
      refine (card_parity b hb).symm.le.trans ((card_image_of_injective _ hinj).symm.le.trans
        (card_le_card fun x hx => ?_))
      obtain ⟨r, hr, rfl⟩ := mem_image.mp hx
      exact mem_filter.mpr ⟨mem_filter.mpr ⟨mem_univ _, hall r (mem_filter.mp hr).2⟩, rfl⟩
    rcases hq with h | h
    · exact key 0 (by norm_num) h
    · exact key 1 (by norm_num) h
  have hsum : ∑ q ∈ univ.filter (Dirty htile hrb hcb t stream c), (D.filter fun x => x.2 = q).card ≤ D.card := by
    rw [card_eq_sum_card_fiberwise (f := Prod.snd) (t := univ) fun _ _ => mem_univ _]
    exact sum_le_sum_of_subset (subset_univ _)
  have h24 : D.card ≤ 24 := card_lanePos_le htile hrb hcb t stream c
  have h8 := (card_nsmul_le_sum _ _ 8 hcol).trans hsum
  rw [smul_eq_mul] at h8
  omega

open Classical in
/-- **The term's fibers have at most `(2^21)^3` values**: the clean columns' lanes are fixed by the value. -/
theorem gapTerm_fiber_le (he : L.e = 7) (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n)
    (t : Fin L.S) (stream : CoinStream L) (c : ℕ) (v : Fin 16 → Fin 8 → ℤ) :
    (univ.filter fun d : TileLanes L => gapTerm cb htile hrb t stream c d = v).card ≤ 2 ^ 63 := by
  set dirty := univ.filter (Dirty htile hrb hcb t stream c)
  set zero : Fin 3 → Fin (2 ^ L.e) := fun _ => ⟨0, Nat.two_pow_pos _⟩
  have hclean : ∀ {d d' : TileLanes L}, gapTerm cb htile hrb t stream c d = v →
      gapTerm cb htile hrb t stream c d' = v → ∀ q ∉ dirty, d q = d' q := by
    intro d d' hd hd' q hq
    have hq' : ¬ Dirty htile hrb hcb t stream c q := fun h => hq (mem_filter.mpr ⟨mem_univ _, h⟩)
    simp only [Dirty, not_or, not_forall] at hq'
    obtain ⟨⟨r, hr, hr0⟩, ⟨r', hr', hr1⟩⟩ := hq'
    have hrow : ∀ {r : Fin 16}, ¬ LanePos htile hrb hcb t stream c (r, q) →
        tileTerm L rb d r q = tileTerm L rb d' r q := fun {r} h => by
      have e := congrFun (congrFun (hd.trans hd'.symm) r) q
      simp only [gapTerm, Pi.sub_apply, streamPart_eq_zero htile hrb hcb t stream c h, sub_zero] at e
      exact e
    exact tileTerm_col_eq he hr hr' (hrow hr0) (hrow hr1)
  let restrict : TileLanes L → TileLanes L := fun d q => if q ∈ dirty then d q else zero
  have hinj : Set.InjOn restrict (univ.filter fun d => gapTerm cb htile hrb t stream c d = v) := by
    intro d hd d' hd' h
    refine funext fun q => ?_
    by_cases hq : q ∈ dirty
    · have := congrFun h q
      simp only [restrict, hq, ite_true] at this
      exact this
    · exact hclean (mem_filter.mp hd).2 (mem_filter.mp hd').2 q hq
  have himg : (univ.filter fun d => gapTerm cb htile hrb t stream c d = v).image restrict ⊆
      Fintype.piFinset fun q => if q ∈ dirty then univ else {zero} := by
    intro f hf
    obtain ⟨d, _, rfl⟩ := mem_image.mp hf
    refine Fintype.mem_piFinset.mpr fun q => ?_
    by_cases hq : q ∈ dirty
    · simp only [hq, ite_true, mem_univ]
    · simp only [restrict, hq, ite_false, mem_singleton]
  rw [← card_image_of_injOn hinj]
  refine (card_le_card himg).trans ?_
  rw [Fintype.card_piFinset]
  simp only [apply_ite card, card_univ, card_singleton]
  rw [prod_ite, prod_const_one, mul_one, prod_const, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin, he,
    filter_mem_eq_inter, univ_inter]
  calc ((2 ^ 7) ^ 3) ^ dirty.card ≤ ((2 ^ 7) ^ 3) ^ 3 :=
        Nat.pow_le_pow_right (by norm_num) (card_dirty_le htile hrb hcb t stream c)
    _ = 2 ^ 63 := by norm_num

/-- The tile less the stream's tile `c`. -/
def diffTile (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (stream : CoinStream L) (c : ℕ) (ω : Coins L) : Fin 16 → Fin 8 → ℤ :=
  stepTile hrb hcb t κ ω - streamTile stream c ω

/-- **The overwrite moves the difference by `gapTerm` alone.** -/
theorem diffTile_set (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (stream : CoinStream L) (c : ℕ) (ω : Coins L) (d : TileLanes L) :
    diffTile hrb hcb t κ stream c (tileSet cb htile hrb t ω d) = diffTile hrb hcb t κ stream c ω -
      gapTerm cb htile hrb t stream c (tileGet htile hrb hcb t ω) + gapTerm cb htile hrb t stream c d := by
  simp only [diffTile, gapTerm]
  rw [stepTile_set htile hrb hcb t κ ω d, streamTile_set htile hrb hcb t stream c ω d]
  abel

/-- **An MMA tile against a streamed tile**: at any step `t`, for any data `κ`, any arrangement of the coin stream and
any streamed tile `c`, the tile's words are the streamed tile on at most a `2^−105` share of the call's coin values. -/
theorem call_tile_stream (he : L.e = 7) (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n)
    (t : Fin L.S) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (stream : CoinStream L) (c : ℕ) :
    2 ^ 105 * (univ.filter fun ω : Coins L => stepTile hrb hcb t κ ω = streamTile stream c ω).card ≤
      Fintype.card (Coins L) := by
  classical
  have h := part_fiber_le (tileGet htile hrb hcb t) (tileSet cb htile hrb t) (tileGet_set htile hrb hcb t)
    (tileSet_set htile hrb t) (tileSet_get htile hrb hcb t) (diffTile hrb hcb t κ stream c)
    (gapTerm cb htile hrb t stream c) (diffTile_set htile hrb hcb t κ stream c) (P := 2 ^ 63)
    (fun v => by convert gapTerm_fiber_le he htile hrb hcb t stream c v) 0
  rw [card_tileLanes he] at h
  have hsub : (univ.filter fun ω : Coins L => stepTile hrb hcb t κ ω = streamTile stream c ω) =
      univ.filter fun ω => diffTile hrb hcb t κ stream c ω = 0 := by
    ext ω
    simp only [mem_filter, mem_univ, true_and, diffTile, sub_eq_zero]
  rw [hsub]
  refine Nat.le_of_mul_le_mul_left ?_ (show 0 < 2 ^ 63 by positivity)
  rw [← mul_assoc, ← pow_add]
  exact h

end Pouw.SecurityProofs.Lead.PStar
