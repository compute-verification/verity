import Definitions.Pouw.Lead.Layout
import Proofs.Pouw.Lead.Network.Rot
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.Ring

/-!
# The lead's cells carry `-rot`'s reads (`lead_cells_macs_le`)

A cell of the lead is `r` rows of one 64-row coin tile at one step `t`, by a set `J` of columns. Its MACs are
`r · |J| · dataLanes t`, and its words (`Pouw.Lead.word`) read the step's three lane coins on the tile at each column
of `J` through `-rot`'s weights.

* `word_sub`: two coin values that differ only in step `t`'s lane coins give step-`t` words that differ by their coin
  lanes (`laneTerm`), at every step, step 0 included.
* `lead_pair_bits`: an outsourcer that gets an even and an odd row's step-`t` words at the columns of `J` from a
  message of `b` bits, at every coin value, has `b ≥ 21 |J|` (`rot_bits_ge`). `lead_one_bits`: from one row's words,
  `b ≥ 13 |J|` (`even_bits_ge`).
* `cell_rot`: so every cell of the lead is a `-rot` cell (`Cell.Rot`): at most 28 data lanes, and either at most
  64 rows of both classes reading 21 bits a column or at most 32 of one class reading 13. `lead_cell_macs_le` and
  `lead_cells_macs_le`: it does at most `256 / 3` MACs a bit of its message, and `rot_cells_macs_le` sums that over
  any set of cells.
-/

namespace Pouw.SecurityProofs.Lead.Network

open Finset Pouw.Lead

/-! ## Words differ by their step's coin lanes -/

/-- Two coin values that agree on the tag bits, the step-1 bits and `G` give coin terms that differ by their coin
lanes. -/
theorem coinTerm_sub {L : Pouw.Lead.Layout} {ω ω' : Coins L} (h1 : ω.1 = ω'.1) (h2 : ω.2.1 = ω'.2.1)
    (h3 : ω.2.2.1 = ω'.2.2.1) (i : Fin L.m) (t : Fin L.S) (j : Fin L.n) :
    coinTerm L ω i t j - coinTerm L ω' i t j = laneTerm L ω i t j - laneTerm L ω' i t j := by
  unfold coinTerm Coins.G
  rw [h1, h2, h3]
  ring

theorem laneTerm_congr {L : Pouw.Lead.Layout} {ω ω' : Coins L} {t : Fin L.S} (h : ω.2.2.2 t = ω'.2.2.2 t) (i : Fin L.m)
    (j : Fin L.n) : laneTerm L ω i t j = laneTerm L ω' i t j := by
  simp only [laneTerm, Coins.s, h]

/-- **Words differ by their step's coin lanes**: two coin values that agree on everything but step `t`'s lane coins
give step-`t` words that differ by the step's coin lanes. -/
theorem word_sub {L : Pouw.Lead.Layout} (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) {ω ω' : Coins L} (h1 : ω.1 = ω'.1)
    (h2 : ω.2.1 = ω'.2.1) (h3 : ω.2.2.1 = ω'.2.2.1) {t : Fin L.S} (h4 : ∀ τ, τ ≠ t → ω.2.2.2 τ = ω'.2.2.2 τ)
    (i : Fin L.m) (j : Fin L.n) :
    Pouw.Lead.word L κ ω i t j - Pouw.Lead.word L κ ω' i t j = laneTerm L ω i t j - laneTerm L ω' i t j := by
  have hτ : ∀ τ : Fin L.S, ((if τ ≤ t then κ i τ j + coinTerm L ω i τ j else 0) -
      if τ ≤ t then κ i τ j + coinTerm L ω' i τ j else 0) =
      if τ = t then laneTerm L ω i t j - laneTerm L ω' i t j else 0 := by
    intro τ
    have hc := coinTerm_sub h1 h2 h3 i τ j
    by_cases hτt : τ = t
    · subst hτt
      rw [ite_eq_left le_rfl, ite_eq_left le_rfl, ite_eq_left rfl]
      linarith
    · have hl := laneTerm_congr (h4 τ hτt) i j
      rw [ite_eq_right hτt]
      split_ifs <;> linarith
  have e : Pouw.Lead.word L κ ω i t j - Pouw.Lead.word L κ ω' i t j = ∑ τ : Fin L.S,
      ((if τ ≤ t then κ i τ j + coinTerm L ω i τ j else 0) -
        if τ ≤ t then κ i τ j + coinTerm L ω' i τ j else 0) := by
    unfold Pouw.Lead.word
    rw [Finset.sum_sub_distrib]
    ring
  rw [e, Finset.sum_congr rfl fun τ _ => hτ τ, Finset.sum_ite_eq' Finset.univ t, ite_eq_left (Finset.mem_univ t)]

/-! ## The coin values a cell's step varies -/

/-- The coin values that vary only step `t`'s lane coins on tile `k` at the columns of `J`, by index (`y`): every other
coin is index 0. -/
def laneCoins (L : Pouw.Lead.Layout) (he : L.e = 7) (t : Fin L.S) (k : Fin L.tiles) (J : Finset (Fin L.n))
    (y : J → Fin 3 → Fin 128) : Coins L :=
  (fun _ _ => 0, fun _ _ => 0, fun _ _ => 0, fun t' k' ℓ j =>
    Fin.cast (by simp [he]) (if h : t' = t ∧ k' = k ∧ j ∈ J then y ⟨j, h.2.2⟩ ℓ else 0))

theorem laneCoins_ne {L : Pouw.Lead.Layout} (he : L.e = 7) {t : Fin L.S} {k : Fin L.tiles} (J : Finset (Fin L.n))
    (y y' : J → Fin 3 → Fin 128) {τ : Fin L.S} (hτ : τ ≠ t) :
    (laneCoins L he t k J y).2.2.2 τ = (laneCoins L he t k J y').2.2.2 τ := by
  funext k' ℓ j
  simp [laneCoins, hτ]

theorem laneCoins_s {L : Pouw.Lead.Layout} (he : L.e = 7) (t : Fin L.S) (k : Fin L.tiles) {J : Finset (Fin L.n)}
    (y : J → Fin 3 → Fin 128) (ℓ : Fin 3) {j : Fin L.n} (hj : j ∈ J) :
    (laneCoins L he t k J y).s t k ℓ j = coinVal (y ⟨j, hj⟩ ℓ) := by
  simp [Coins.s, laneCoins, hj, coinVal, he]

/-- A row's coin lanes read its own tile's lane coins. -/
theorem laneTerm_eq {L : Pouw.Lead.Layout} (ω : Coins L) {i : Fin L.m} {k : Fin L.tiles} (hik : (i : ℕ) / L.tile = k)
    (t : Fin L.S) (j : Fin L.n) : laneTerm L ω i t j = ∑ ℓ : Fin 3, rotWeight i ℓ * ω.s t k ℓ j := by
  have hk : (i : ℕ) / L.tile < L.tiles := by rw [hik]; exact k.isLt
  have hk' : (⟨(i : ℕ) / L.tile, hk⟩ : Fin L.tiles) = k := Fin.ext hik
  unfold laneTerm
  refine Finset.sum_congr rfl fun ℓ _ => ?_
  rw [dite_eq_left hk, hk']

/-- An even row's coin lanes are the even fold. -/
theorem rotSum_even {i : ℕ} (hi : i % 2 = 0) (v : Fin 3 → ℤ) :
    ∑ ℓ : Fin 3, rotWeight i ℓ * v ℓ = foldEven (v 0, v 1, v 2) := by
  simp [Fin.sum_univ_three, rotWeight, hi, foldEven]

/-- An odd row's coin lanes are the odd fold. -/
theorem rotSum_odd {i : ℕ} (hi : i % 2 = 1) (v : Fin 3 → ℤ) :
    ∑ ℓ : Fin 3, rotWeight i ℓ * v ℓ = foldOdd (v 0, v 1, v 2) := by
  simp [Fin.sum_univ_three, rotWeight, hi, foldOdd]

/-- A step-`t` word of a coin value of `laneCoins` is a constant of its row and column plus its coin lanes. -/
theorem word_laneCoins {L : Pouw.Lead.Layout} (he : L.e = 7) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (t : Fin L.S)
    (k : Fin L.tiles) (J : Finset (Fin L.n)) (y y₀ : J → Fin 3 → Fin 128) (i : Fin L.m) (j : Fin L.n) :
    Pouw.Lead.word L κ (laneCoins L he t k J y) i t j =
      (Pouw.Lead.word L κ (laneCoins L he t k J y₀) i t j - laneTerm L (laneCoins L he t k J y₀) i t j) +
        laneTerm L (laneCoins L he t k J y) i t j := by
  have := word_sub κ (ω := laneCoins L he t k J y) (ω' := laneCoins L he t k J y₀) rfl rfl rfl
    (fun τ hτ => laneCoins_ne he J y y₀ hτ) i j
  linarith

/-! ## Rows of both classes: 21 bits a column -/

/-- A triple's coin indices by lane. -/
def tripleAt (x : Triple) (ℓ : Fin 3) : Fin 128 :=
  if (ℓ : ℕ) = 0 then x.1 else if (ℓ : ℕ) = 1 then x.2.1 else x.2.2

theorem lanes_tripleAt {L : Pouw.Lead.Layout} (he : L.e = 7) (t : Fin L.S) (k : Fin L.tiles) {J : Finset (Fin L.n)}
    (x : J → Triple) {i : Fin L.m} (hik : (i : ℕ) / L.tile = k) (j : J) :
    laneTerm L (laneCoins L he t k J fun j => tripleAt (x j)) i t j =
      ∑ ℓ : Fin 3, rotWeight i ℓ * coinVal (tripleAt (x j) ℓ) := by
  rw [laneTerm_eq _ hik]
  refine Finset.sum_congr rfl fun ℓ _ => ?_
  rw [laneCoins_s he t k _ ℓ j.2]

/-- **Rows of both classes need 21 bits a column**: a message of `b` bits from which an outsourcer gets an even and an
odd row's step-`t` words at the columns of `J`, both rows of tile `k`, for every coin value, has `b ≥ 21 |J|`. -/
theorem lead_pair_bits {L : Pouw.Lead.Layout} (he : L.e = 7) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (t : Fin L.S)
    (k : Fin L.tiles) {i i' : Fin L.m} (hik : (i : ℕ) / L.tile = k) (hik' : (i' : ℕ) / L.tile = k)
    (hi : (i : ℕ) % 2 = 0) (hi' : (i' : ℕ) % 2 = 1) (J : Finset (Fin L.n)) (b : ℕ) (h : Coins L → Fin (2 ^ b))
    (g : Fin (2 ^ b) → Fin L.m → Fin L.n → ℤ)
    (hg : ∀ ω, ∀ j ∈ J, g (h ω) i j = Pouw.Lead.word L κ ω i t j ∧ g (h ω) i' j = Pouw.Lead.word L κ ω i' t j) :
    21 * J.card ≤ b := by
  classical
  have key := rot_bits_ge (ι := J) b (fun x => h (laneCoins L he t k J fun j => tripleAt (x j)))
    (fun c j =>
      (g c i j - (Pouw.Lead.word L κ (laneCoins L he t k J fun _ => tripleAt (0, 0, 0)) i t j -
          laneTerm L (laneCoins L he t k J fun _ => tripleAt (0, 0, 0)) i t j),
        g c i' j - (Pouw.Lead.word L κ (laneCoins L he t k J fun _ => tripleAt (0, 0, 0)) i' t j -
          laneTerm L (laneCoins L he t k J fun _ => tripleAt (0, 0, 0)) i' t j)))
    (fun x => by
      funext j
      obtain ⟨hgi, hgi'⟩ := hg (laneCoins L he t k J fun j => tripleAt (x j)) j j.2
      rw [hgi, hgi', word_laneCoins he κ t k J _ (fun _ => tripleAt (0, 0, 0)) i j,
        word_laneCoins he κ t k J _ (fun _ => tripleAt (0, 0, 0)) i' j, add_sub_cancel_left, add_sub_cancel_left,
        lanes_tripleAt he t k x hik j, lanes_tripleAt he t k x hik' j, rotSum_even hi, rotSum_odd hi']
      simp [rotReads, rotRead, foldPair, coins, tripleAt])
  rwa [Fintype.card_coe] at key

/-! ## Rows of one class: 13 bits a column -/

/-- **One fold needs 13 bits a column**: a message of `b` bits that gives `evenRead` of `N` index pairs for every
value of them has `b ≥ 13 N`. -/
theorem even_bits_ge {ι : Type*} [Fintype ι] [DecidableEq ι] (b : ℕ) (h : (ι → Fin 128 × Fin 127) → Fin (2 ^ b))
    (g : Fin (2 ^ b) → ι → ℤ) (hfac : ∀ x, g (h x) = fun i => evenRead (x i)) : 13 * Fintype.card ι ≤ b := by
  have hinj : Function.Injective h := fun x y hxy => funext fun i =>
    evenRead_injective (by rw [← congrFun (hfac x) i, ← congrFun (hfac y) i, hxy])
  have hcard : 2 ^ (13 * Fintype.card ι) ≤ Fintype.card (ι → Fin 128 × Fin 127) := by
    rw [Fintype.card_fun, pow_mul]
    exact Nat.pow_le_pow_left card_evenRead_dom _
  have := hcard.trans (Fintype.card_le_of_injective h hinj)
  rw [Fintype.card_fin] at this
  exact (Nat.pow_le_pow_iff_right (by norm_num)).1 this

/-- The indices a row of parity `p` reads one fold of: its own class's 127-weight lane takes `x.1`, the other class's
`x.2` and the third lane index 64 (the coin 1). -/
def halfAt (p : ℕ) (x : Fin 128 × Fin 127) (ℓ : Fin 3) : Fin 128 :=
  if (ℓ : ℕ) = p then x.1 else if (ℓ : ℕ) = 1 - p then Fin.castSucc x.2 else 64

theorem rotSum_halfAt {i : ℕ} (x : Fin 128 × Fin 127) :
    ∑ ℓ : Fin 3, rotWeight i ℓ * coinVal (halfAt (i % 2) x ℓ) = evenRead x := by
  rcases Nat.mod_two_eq_zero_or_one i with hi | hi
  · rw [rotSum_even hi]
    simp [halfAt, hi, evenRead, foldEven, coinVal]
  · rw [rotSum_odd hi]
    simp [halfAt, hi, evenRead, foldEven, foldOdd, coinVal]
    ring

/-- **Rows of one class need 13 bits a column**: a message of `b` bits from which an outsourcer gets one row's step-`t`
words at the columns of `J` for every coin value has `b ≥ 13 |J|`. -/
theorem lead_one_bits {L : Pouw.Lead.Layout} (he : L.e = 7) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (t : Fin L.S)
    (k : Fin L.tiles) {i : Fin L.m} (hik : (i : ℕ) / L.tile = k) (J : Finset (Fin L.n)) (b : ℕ)
    (h : Coins L → Fin (2 ^ b)) (g : Fin (2 ^ b) → Fin L.m → Fin L.n → ℤ)
    (hg : ∀ ω, ∀ j ∈ J, g (h ω) i j = Pouw.Lead.word L κ ω i t j) : 13 * J.card ≤ b := by
  classical
  have key := even_bits_ge (ι := J) b
    (fun x => h (laneCoins L he t k J fun j => halfAt ((i : ℕ) % 2) (x j)))
    (fun c j => g c i j - (Pouw.Lead.word L κ (laneCoins L he t k J fun _ => halfAt ((i : ℕ) % 2) (0, 0)) i t j -
      laneTerm L (laneCoins L he t k J fun _ => halfAt ((i : ℕ) % 2) (0, 0)) i t j))
    (fun x => by
      funext j
      rw [hg _ j j.2, word_laneCoins he κ t k J _ (fun _ => halfAt ((i : ℕ) % 2) (0, 0)) i j,
        add_sub_cancel_left, laneTerm_eq _ hik]
      rw [← rotSum_halfAt (i := (i : ℕ)) (x j)]
      refine Finset.sum_congr rfl fun ℓ _ => ?_
      rw [laneCoins_s he t k _ ℓ j.2])
  rwa [Fintype.card_coe] at key

/-! ## The lead's cells -/

theorem card_tile_le {L : Pouw.Lead.Layout} (hT : L.tile = 64) {k : ℕ} (R : Finset (Fin L.m))
    (hR : ∀ i ∈ R, (i : ℕ) / L.tile = k) : R.card ≤ 64 := by
  have hinj : Set.InjOn (fun i : Fin L.m => (i : ℕ) % 64) R := by
    intro a ha b hb hab
    have h1 := hR a ha
    have h2 := hR b hb
    rw [hT] at h1 h2
    have hab' : (a : ℕ) % 64 = (b : ℕ) % 64 := hab
    exact Fin.ext (by omega)
  calc R.card = (R.image fun i : Fin L.m => (i : ℕ) % 64).card := (card_image_of_injOn hinj).symm
    _ ≤ (range 64).card := card_le_card fun x hx => by
        obtain ⟨a, -, rfl⟩ := mem_image.1 hx
        exact mem_range.2 (Nat.mod_lt _ (by norm_num))
    _ = 64 := card_range 64

theorem card_class_le {L : Pouw.Lead.Layout} (hT : L.tile = 64) {k p : ℕ} (R : Finset (Fin L.m))
    (hR : ∀ i ∈ R, (i : ℕ) / L.tile = k) (hp : ∀ i ∈ R, (i : ℕ) % 2 = p) : R.card ≤ 32 := by
  have hinj : Set.InjOn (fun i : Fin L.m => (i : ℕ) % 64 / 2) R := by
    intro a ha b hb hab
    have h1 := hR a ha
    have h2 := hR b hb
    have h3 := hp a ha
    have h4 := hp b hb
    rw [hT] at h1 h2
    have hab' : (a : ℕ) % 64 / 2 = (b : ℕ) % 64 / 2 := hab
    exact Fin.ext (by omega)
  calc R.card = (R.image fun i : Fin L.m => (i : ℕ) % 64 / 2).card := (card_image_of_injOn hinj).symm
    _ ≤ (range 32).card := card_le_card fun x hx => by
        obtain ⟨a, -, rfl⟩ := mem_image.1 hx
        exact mem_range.2 (by omega)
    _ = 32 := card_range 32

/-- Data lanes of step `t`: 21 at step 0, whose other lanes are the three coin lanes, the row tag against `G` and `a₁`
against the column code, and 28 at each later step, whose other lanes are the three coin lanes and the tag lane. -/
def dataLanes (t : ℕ) : ℕ := if t = 0 then 21 else 28

theorem dataLanes_le (t : ℕ) : dataLanes t ≤ 28 := by
  unfold dataLanes
  split_ifs <;> norm_num

/-- **Every cell of the lead is a `-rot` cell**: `R`, rows of coin tile `k` (64 rows, 7-bit coins), at step `t` (step 0
included), by the columns of `J`, whose step-`t` words an outsourcer gets from a message of `b` bits at every coin
value, is a cell of `|R|` rows, `|J|` columns and `dataLanes t ≤ 28` data lanes reading `b` bits that meets
`Cell.Rot`: at most 64 rows with `b ≥ 21 |J|`, or, its rows of one class, at most 32 with `b ≥ 13 |J|`. -/
theorem cell_rot (L : Pouw.Lead.Layout) (hT : L.tile = 64) (he : L.e = 7) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ)
    (t : Fin L.S) (k : Fin L.tiles) (R : Finset (Fin L.m)) (hne : R.Nonempty)
    (hR : ∀ i ∈ R, (i : ℕ) / L.tile = k) (J : Finset (Fin L.n)) (b : ℕ) (h : Coins L → Fin (2 ^ b))
    (g : Fin (2 ^ b) → Fin L.m → Fin L.n → ℤ) (hg : ∀ ω, ∀ i ∈ R, ∀ j ∈ J, g (h ω) i j = Pouw.Lead.word L κ ω i t j) :
    (⟨R.card, J.card, dataLanes t, b⟩ : Cell).Rot := by
  show (0 : ℝ) ≤ R.card ∧ (0 : ℝ) ≤ J.card ∧ (0 : ℝ) ≤ (dataLanes t : ℕ) ∧ ((dataLanes t : ℕ) : ℝ) ≤ 28 ∧
    (((R.card : ℝ) ≤ 64 ∧ 21 * (J.card : ℝ) ≤ b) ∨ ((R.card : ℝ) ≤ 32 ∧ 13 * (J.card : ℝ) ≤ b))
  refine ⟨Nat.cast_nonneg _, Nat.cast_nonneg _, Nat.cast_nonneg _, by exact_mod_cast dataLanes_le t, ?_⟩
  by_cases hboth : (∃ i ∈ R, (i : ℕ) % 2 = 0) ∧ ∃ i ∈ R, (i : ℕ) % 2 = 1
  · obtain ⟨⟨i, hi, hi0⟩, ⟨i', hi', hi1⟩⟩ := hboth
    have hbits := lead_pair_bits he κ t k (hR i hi) (hR i' hi') hi0 hi1 J b h g
      fun ω j hj => ⟨hg ω i hi j hj, hg ω i' hi' j hj⟩
    exact Or.inl ⟨by exact_mod_cast card_tile_le hT R hR, by exact_mod_cast hbits⟩
  · obtain ⟨i₀, hi₀⟩ := hne
    have hp : ∀ i ∈ R, (i : ℕ) % 2 = (i₀ : ℕ) % 2 := fun i hi => by
      rcases Nat.mod_two_eq_zero_or_one (i : ℕ) with h0 | h0 <;>
        rcases Nat.mod_two_eq_zero_or_one (i₀ : ℕ) with h1 | h1
      · rw [h0, h1]
      · exact absurd ⟨⟨i, hi, h0⟩, ⟨i₀, hi₀, h1⟩⟩ hboth
      · exact absurd ⟨⟨i₀, hi₀, h1⟩, ⟨i, hi, h0⟩⟩ hboth
      · rw [h0, h1]
    have hbits := lead_one_bits he κ t k (hR i₀ hi₀) J b h g fun ω j hj => hg ω i₀ hi₀ j hj
    exact Or.inr ⟨by exact_mod_cast card_class_le hT R hR hp, by exact_mod_cast hbits⟩

/-- **A cell of the lead does at most `256 / 3` MACs a bit of its message.** -/
theorem lead_cell_macs_le (L : Pouw.Lead.Layout) (hT : L.tile = 64) (he : L.e = 7) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ)
    (t : Fin L.S) (k : Fin L.tiles) (R : Finset (Fin L.m)) (hne : R.Nonempty)
    (hR : ∀ i ∈ R, (i : ℕ) / L.tile = k) (J : Finset (Fin L.n)) (b : ℕ) (h : Coins L → Fin (2 ^ b))
    (g : Fin (2 ^ b) → Fin L.m → Fin L.n → ℤ) (hg : ∀ ω, ∀ i ∈ R, ∀ j ∈ J, g (h ω) i j = Pouw.Lead.word L κ ω i t j) :
    (R.card * J.card * dataLanes t : ℝ) ≤ 256 / 3 * b :=
  Cell.macs_le _ (cell_rot L hT he κ t k R hne hR J b h g hg)

/-- **`n_max ≤ 256 / 3` at every cell of the lead**: rows `R` of one 64-row coin tile at step `t` (step 0 included),
by the columns `J`, whose step-`t` words an outsourcer gets from a message of `b` bits at every coin value, do
`|R| · |J| · dataLanes t` MACs, at most `256 / 3` a bit of the message. -/
theorem lead_cells_macs_le (L : Pouw.Lead.Layout) (hT : L.tile = 64) (he : L.e = 7)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (t : Fin L.S) (k : Fin L.tiles) (R : Finset (Fin L.m))
    (hR : ∀ i ∈ R, (i : ℕ) / L.tile = k) (J : Finset (Fin L.n)) (b : ℕ) (h : Coins L → Fin (2 ^ b))
    (g : Fin (2 ^ b) → Fin L.m → Fin L.n → ℤ) (hg : ∀ ω, ∀ i ∈ R, ∀ j ∈ J, g (h ω) i j = Pouw.Lead.word L κ ω i t j) :
    3 * (R.card * J.card * dataLanes t) ≤ 256 * b := by
  rcases R.eq_empty_or_nonempty with rfl | hne
  · simp
  have hmacs := lead_cell_macs_le L hT he κ t k R hne hR J b h g hg
  have h3 : ((3 * (R.card * J.card * dataLanes t) : ℕ) : ℝ) ≤ ((256 * b : ℕ) : ℝ) := by
    push_cast
    linarith
  exact_mod_cast h3

end Pouw.SecurityProofs.Lead.Network
