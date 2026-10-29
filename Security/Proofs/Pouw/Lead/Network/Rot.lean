import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Pi
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

/-!
# What one exchanged bit carries under `-rot`'s weights

The lead's coin lanes on a 64-row tile are three odd coins in `±127` a (tile, step, column), which a k32 word reads
through the public weights `A' = (127, 1, 1)` on the tile's even rows and `(1, 127, 1)` on its odd rows (`foldEven`,
`foldOdd`). A step has at most 28 data lanes (21 on step 0), and at a tile of 64 rows every step carries the three
lanes.

* `rot_fold_injective`, `rotRead_injective`: the two folds determine the three coins, so a tile with rows of both
  classes reads all 21 of its coin bits a step and column (`card_triple`: `2 ^ 21` coin triples). One fold alone
  doesn't (`foldEven_not_injective`), but it still reads `127 s₁ + s₂` whole for `s₂ ≤ 125`
  (`evenRead_injective`, `2 ^ 7 · 127 ≥ 2 ^ 13` values).
* `card_le_of_factor`, `rot_card_le_on`, `rot_card_le`, `rot_bits_ge`: a computation through a message in `M` that
  agrees with an injective read of the coins on a set `T` of coin values has `|T| ≤ |M|`, so its chance on uniform
  coins is at most `|M|` over the coin values; right for every coin value of `N` triples, it needs `2 ^ (21 N)`
  message values, `21 N` bits.
* `sum_le_of_le_mul` (the mediant), `Cell.macs_le` and `rot_cells_macs_le`: a tile's rows at one step, `r ≤ 64` of
  both classes reading `21 n` bits or `r ≤ 32` of one class reading `13 n`, by `n` columns and at most 28 data lanes,
  do at most `256 / 3` MACs a coin bit, so every set of such cells does (every step interval, row set and column set
  of a call): `n_max ≤ 256 / 3` wherever every step carries the three lanes.
* `netShare`, `netShare_lead_cap32` and `netShare_lead_cap320`: the network term `n_max W_mac E` over the stock MACs a
  token, at `n_max ≤ 256 / 3` and Llama-3.2-1B's MACs a token, is at most 0.0361% at 32 bits a token and 0.361%
  at 320 while `W_mac ≤ 128.59`; `netShare_le_of_stock_le` bounds the same term over any `W_reg` of at least the stock.
-/

namespace Pouw.SecurityProofs.Lead.Network

open Finset

/-! ## The folds -/

/-- A lane coin at `e = 7`: odd, in `±127`. -/
def OddCoin (s : ℤ) : Prop := s % 2 = 1 ∧ -127 ≤ s ∧ s ≤ 127

/-- The even rows' fold of a (tile, step, column)'s three lane coins, `A' = (127, 1, 1)`. -/
def foldEven (s : ℤ × ℤ × ℤ) : ℤ := 127 * s.1 + s.2.1 + s.2.2

/-- The odd rows' fold, `A' = (1, 127, 1)`. -/
def foldOdd (s : ℤ × ℤ × ℤ) : ℤ := s.1 + 127 * s.2.1 + s.2.2

/-- What a tile with rows of both classes reads of its three lane coins a step and column: both folds. -/
def foldPair (s : ℤ × ℤ × ℤ) : ℤ × ℤ := (foldEven s, foldOdd s)

/-- The two folds determine the three coins: their difference is `126 (s₁ - s₂)`, and then `128 s₁ + s₃` with
`|s₃| ≤ 127` and `s₁` odd fixes `s₁` and `s₃`. -/
theorem rot_fold_injective {s t : ℤ × ℤ × ℤ} (hs : OddCoin s.1 ∧ OddCoin s.2.1 ∧ OddCoin s.2.2)
    (ht : OddCoin t.1 ∧ OddCoin t.2.1 ∧ OddCoin t.2.2) (he : foldEven s = foldEven t)
    (ho : foldOdd s = foldOdd t) : s = t := by
  obtain ⟨a, b, c⟩ := s
  obtain ⟨a', b', c'⟩ := t
  simp only [OddCoin, foldEven, foldOdd] at hs ht he ho
  have h : a = a' ∧ b = b' ∧ c = c' := by omega
  obtain ⟨rfl, rfl, rfl⟩ := h
  rfl

/-- One fold alone doesn't determine the coins: `(1, 1, -1)` and `(1, -1, 1)` agree under `A' = (127, 1, 1)`. -/
theorem foldEven_not_injective : ∃ s t : ℤ × ℤ × ℤ, (OddCoin s.1 ∧ OddCoin s.2.1 ∧ OddCoin s.2.2) ∧
    (OddCoin t.1 ∧ OddCoin t.2.1 ∧ OddCoin t.2.2) ∧ foldEven s = foldEven t ∧ s ≠ t :=
  ⟨(1, 1, -1), (1, -1, 1), by norm_num [OddCoin], by norm_num [OddCoin], by norm_num [foldEven], by norm_num⟩

/-! ## Coins by index -/

/-- The lane coin of index `k`: `2 k - 127`, the odd values in `±127` in order. -/
def coinVal (k : Fin 128) : ℤ := 2 * (k : ℤ) - 127

theorem coinVal_oddCoin (k : Fin 128) : OddCoin (coinVal k) := by
  have := k.isLt
  unfold OddCoin coinVal
  omega

theorem exists_coinVal {s : ℤ} (hs : OddCoin s) : ∃ k, coinVal k = s := by
  unfold OddCoin at hs
  refine ⟨⟨((s + 127) / 2).toNat, by omega⟩, ?_⟩
  simp only [coinVal]
  omega

/-- A (tile, step, column)'s three lane coins, by index. -/
abbrev Triple := Fin 128 × Fin 128 × Fin 128

theorem card_triple : Fintype.card Triple = 2 ^ 21 := by
  simp only [Fintype.card_prod, Fintype.card_fin]
  norm_num

/-- A triple's coins. -/
def coins (x : Triple) : ℤ × ℤ × ℤ := (coinVal x.1, coinVal x.2.1, coinVal x.2.2)

/-- What a tile with rows of both classes reads of a triple. -/
def rotRead (x : Triple) : ℤ × ℤ := foldPair (coins x)

theorem rotRead_injective : Function.Injective rotRead := by
  rintro ⟨a, b, c⟩ ⟨a', b', c'⟩ h
  have h1 := congrArg Prod.fst h
  have h2 := congrArg Prod.snd h
  simp only [rotRead, foldPair, coins, foldEven, foldOdd, coinVal] at h1 h2
  have := a.isLt; have := b.isLt; have := c.isLt; have := a'.isLt; have := b'.isLt; have := c'.isLt
  have ha : (a : ℕ) = a' := by omega
  have hb : (b : ℕ) = b' := by omega
  have hc : (c : ℕ) = c' := by omega
  rw [Fin.ext ha, Fin.ext hb, Fin.ext hc]

/-- What a tile with rows of only the even class reads, on the triples `(s₁, s₂, 1)` with `s₂ = 2 j - 127 ≤ 125`. -/
def evenRead (x : Fin 128 × Fin 127) : ℤ := foldEven (coinVal x.1, 2 * (x.2 : ℤ) - 127, 1)

theorem evenRead_injective : Function.Injective evenRead := by
  rintro ⟨a, b⟩ ⟨a', b'⟩ h
  simp only [evenRead, foldEven, coinVal] at h
  have := a.isLt; have := b.isLt; have := a'.isLt; have := b'.isLt
  have ha : (a : ℕ) = a' := by omega
  have hb : (b : ℕ) = b' := by omega
  rw [Fin.ext ha, Fin.ext hb]

theorem card_evenRead_dom : 2 ^ 13 ≤ Fintype.card (Fin 128 × Fin 127) := by
  rw [Fintype.card_prod, Fintype.card_fin, Fintype.card_fin]
  norm_num

/-! ## Counting -/

/-- **A message carries at most its values' worth of an injective read**: if `g ∘ h` agrees with `f` on `T`, where
`f` is injective on `T`, then `T` has at most as many elements as the message type `M`. With `T` the coin values on
which an outsourcer's words are right, its chance on uniform coins over `S ⊇ T` is `|T| / |S| ≤ |M| / |S|`. -/
theorem card_le_of_factor {α β M : Type*} [Fintype M] (f : α → β) (T : Finset α) (hf : Set.InjOn f T)
    (h : α → M) (g : M → β) (hfac : ∀ x ∈ T, g (h x) = f x) : T.card ≤ Fintype.card M := by
  classical
  have hinj : Set.InjOn h T := fun x hx y hy hxy => hf hx hy (by rw [← hfac x hx, ← hfac y hy, hxy])
  calc T.card = (T.image h).card := (card_image_of_injOn hinj).symm
    _ ≤ Fintype.card M := card_le_univ _

/-- The reads of `N` triples, pointwise. -/
def rotReads {ι : Type*} (x : ι → Triple) : ι → ℤ × ℤ := fun i => rotRead (x i)

theorem rotReads_injective {ι : Type*} : Function.Injective (rotReads (ι := ι)) := fun _ _ h =>
  funext fun i => rotRead_injective (congrFun h i)

theorem card_triples {ι : Type*} [Fintype ι] [DecidableEq ι] :
    Fintype.card (ι → Triple) = 2 ^ (21 * Fintype.card ι) := by
  rw [Fintype.card_fun, card_triple, pow_mul]

/-- **An outsourcer right on `T`**: a computation of `N` triples' reads through a message in `M`, right on a set `T` of
coin values, has `|T| ≤ |M|`, so its chance on uniform coins is at most `|M| / 2 ^ (21 N)`. -/
theorem rot_card_le_on {ι M : Type*} [Fintype M] (T : Finset (ι → Triple)) (h : (ι → Triple) → M)
    (g : M → ι → ℤ × ℤ) (hfac : ∀ x ∈ T, g (h x) = rotReads x) : T.card ≤ Fintype.card M :=
  card_le_of_factor rotReads T rotReads_injective.injOn h g hfac

/-- **The reads of `N` triples need `2 ^ (21 N)` message values**: a computation of every triple's reads from a message
in `M`, right for every coin value, has `2 ^ (21 N) ≤ |M|`. -/
theorem rot_card_le {ι M : Type*} [Fintype ι] [DecidableEq ι] [Fintype M] (h : (ι → Triple) → M)
    (g : M → ι → ℤ × ℤ) (hfac : ∀ x, g (h x) = rotReads x) : 2 ^ (21 * Fintype.card ι) ≤ Fintype.card M := by
  have hinj : Function.Injective h := fun x y hxy => rotReads_injective (by rw [← hfac x, ← hfac y, hxy])
  rw [← card_triples]
  exact Fintype.card_le_of_injective h hinj

/-- A message of `b` bits that gives `N` triples' reads for every coin value has `b ≥ 21 N`. -/
theorem rot_bits_ge {ι : Type*} [Fintype ι] [DecidableEq ι] (b : ℕ) (h : (ι → Triple) → Fin (2 ^ b))
    (g : Fin (2 ^ b) → ι → ℤ × ℤ) (hfac : ∀ x, g (h x) = rotReads x) : 21 * Fintype.card ι ≤ b := by
  have := rot_card_le h g hfac
  rw [Fintype.card_fin] at this
  exact (Nat.pow_le_pow_iff_right (by norm_num)).1 this

/-! ## MACs a coin bit -/

/-- **The mediant**: a sum of terms each at most `c` times its weight is at most `c` times the weights' sum. -/
theorem sum_le_of_le_mul {ι : Type*} (s : Finset ι) (a b : ι → ℝ) (c : ℝ) (h : ∀ i ∈ s, a i ≤ c * b i) :
    ∑ i ∈ s, a i ≤ c * ∑ i ∈ s, b i := by
  rw [mul_sum]
  exact sum_le_sum h

/-- A cell of a call's coin layout: `r` rows of one tile at one step, by `n` columns and `d` data lanes, whose words
read `bits` coin bits. -/
structure Cell where
  r : ℝ
  n : ℝ
  d : ℝ
  bits : ℝ

/-- A cell's MACs. -/
def Cell.macs (c : Cell) : ℝ := c.r * c.n * c.d

/-- A cell of a step that carries the three lanes under `-rot`: at most 28 data lanes, and either at most 64 rows of
both classes reading the 21 bits a column `rotRead_injective` gives, or at most 32 rows of one class reading the 13
`evenRead_injective` gives. -/
def Cell.Rot (c : Cell) : Prop :=
  0 ≤ c.r ∧ 0 ≤ c.n ∧ 0 ≤ c.d ∧ c.d ≤ 28 ∧ ((c.r ≤ 64 ∧ 21 * c.n ≤ c.bits) ∨ (c.r ≤ 32 ∧ 13 * c.n ≤ c.bits))

/-- **One cell**: at most `256 / 3` MACs a coin bit (`64 · 28 = 256 / 3 · 21`, and `32 · 28 < 256 / 3 · 13`). -/
theorem Cell.macs_le (c : Cell) (h : c.Rot) : c.macs ≤ 256 / 3 * c.bits := by
  obtain ⟨hr, hn, hd, hd28, hb⟩ := h
  have hnd : c.n * c.d ≤ c.n * 28 := mul_le_mul_of_nonneg_left hd28 hn
  have hnd0 : 0 ≤ c.n * c.d := mul_nonneg hn hd
  unfold Cell.macs
  rw [mul_assoc]
  rcases hb with ⟨hr64, hb⟩ | ⟨hr32, hb⟩
  · have : c.r * (c.n * c.d) ≤ 64 * (c.n * c.d) := mul_le_mul_of_nonneg_right hr64 hnd0
    linarith
  · have : c.r * (c.n * c.d) ≤ 32 * (c.n * c.d) := mul_le_mul_of_nonneg_right hr32 hnd0
    linarith

/-- **`n_max ≤ 256 / 3` wherever every step carries the three lanes**: any set of cells (a step interval, a set of
rows or of columns, of one call or several) does at most `256 / 3` MACs a coin bit its words read, whatever the
calls' `K` and `N`. -/
theorem rot_cells_macs_le {ι : Type*} (s : Finset ι) (c : ι → Cell) (h : ∀ i ∈ s, (c i).Rot) :
    ∑ i ∈ s, (c i).macs ≤ 256 / 3 * ∑ i ∈ s, (c i).bits :=
  sum_le_of_le_mul s _ _ _ fun i hi => (c i).macs_le (h i hi)

/-! ## The term -/

/-- **The network term as a share of stock**: `n_max W_mac E` of priced work over the stock MACs, `E` bits a token
exchanged and `macs` MACs a token. -/
noncomputable def netShare (nmax Wmac E macs : ℝ) : ℝ := nmax * Wmac * E / macs

/-- Llama-3.2-1B's linears' MACs a token: 16 layers of `2048 × 3072`, `2048 × 2048`, `2048 × 16384` and
`8192 × 2048`. -/
def leadMacs : ℕ := 16 * (2048 * 3072 + 2048 * 2048 + 2048 * 16384 + 8192 * 2048)

theorem leadMacs_eq : leadMacs = 973078528 := by norm_num [leadMacs]

theorem netShare_mono {nmax nmax' Wmac Wmac' E macs : ℝ} (hn : 0 ≤ nmax) (hn' : nmax ≤ nmax') (hW : 0 ≤ Wmac)
    (hW' : Wmac ≤ Wmac') (hE : 0 ≤ E) (hm : 0 < macs) : netShare nmax Wmac E macs ≤ netShare nmax' Wmac' E macs := by
  unfold netShare
  exact div_le_div_of_nonneg_right
    (mul_le_mul_of_nonneg_right (mul_le_mul hn' hW' hW (hn.trans hn')) hE) hm.le

/-- **The lead at 32 bits a token**: at most 0.0361% of stock while `W_mac ≤ 128.59`. -/
theorem netShare_lead_cap32 {nmax Wmac : ℝ} (hn : 0 ≤ nmax) (hn' : nmax ≤ 256 / 3) (hW : 0 ≤ Wmac)
    (hW' : Wmac ≤ 128.59) : netShare nmax Wmac 32 leadMacs ≤ 0.000361 := by
  refine (netShare_mono hn hn' hW hW' (by norm_num) (by rw [leadMacs_eq]; norm_num)).trans ?_
  rw [netShare, leadMacs_eq]
  norm_num

/-- **The lead at 320 bits a token**: at most 0.361% of stock while `W_mac ≤ 128.59`. -/
theorem netShare_lead_cap320 {nmax Wmac : ℝ} (hn : 0 ≤ nmax) (hn' : nmax ≤ 256 / 3) (hW : 0 ≤ Wmac)
    (hW' : Wmac ≤ 128.59) : netShare nmax Wmac 320 leadMacs ≤ 0.00361 := by
  refine (netShare_mono hn hn' hW hW' (by norm_num) (by rw [leadMacs_eq]; norm_num)).trans ?_
  rw [netShare, leadMacs_eq]
  norm_num

/-- Over a `W_reg` of at least the stock (every MAC priced at least 1), the term is at most its share of stock. -/
theorem netShare_le_of_stock_le {nmax Wmac E macs Wreg : ℝ} (hn : 0 ≤ nmax) (hW : 0 ≤ Wmac) (hE : 0 ≤ E)
    (hm : 0 < macs) (hreg : macs ≤ Wreg) : nmax * Wmac * E / Wreg ≤ netShare nmax Wmac E macs := by
  unfold netShare
  exact div_le_div_of_nonneg_left (mul_nonneg (mul_nonneg hn hW) hE) hm hreg

end Pouw.SecurityProofs.Lead.Network
