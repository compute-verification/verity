/-
Level 3, L3-A: the lookup slots' B side (`Lookup.foldB`, PROTOCOL.md §16.5). Product row `(j, h)` has a B-side 1 in the
column of low minterm `l` exactly when bit `j` of table word `h·2^lo + l` is set; `foldB` adds `Bᵀe` for these rows,
with a subset-sum table per byte of the word. `foldB_get`: entry `loTop + l` gains `Σ_h Σ_{j : bit j of word(h, l)} e(j, h)`,
and no other entry changes.
-/
import Proofs.Flock.Level3.Fold
import Flock.Lookup

open Flock Lookup

set_option maxRecDepth 20000

namespace FlockLevel3

/-! ### Subset sums of a byte -/

/-- `x` without its lowest set bit, and that bit (checked for every nonzero byte). -/
theorem lowbit_facts : ∀ x : Fin 256, x.val ≠ 0 →
    x.val ^^^ (x.val &&& (256 - x.val)) < x.val ∧ Nat.log2 (x.val &&& (256 - x.val)) < 8 ∧
    x.val.testBit (Nat.log2 (x.val &&& (256 - x.val))) = true ∧
    ∀ i : Fin 8, (x.val ^^^ (x.val &&& (256 - x.val))).testBit i =
      (x.val.testBit i && decide (i.val ≠ Nat.log2 (x.val &&& (256 - x.val)))) := by decide +kernel

/-- `Σ_{i < 8, bit i of y} f(i)`. -/
noncomputable def byteSum (f : ℕ → F128) (y : ℕ) : GF128 :=
  ∑ i ∈ Finset.range 8, if y.testBit i then GF128.of (f i) else 0

theorem byteSum_zero (f : ℕ → F128) : byteSum f 0 = 0 := by simp [byteSum]

theorem byteSum_step (f : ℕ → F128) (x : ℕ) (hx : x < 256) (h0 : x ≠ 0) :
    byteSum f x = byteSum f (x ^^^ (x &&& (256 - x))) + GF128.of (f (Nat.log2 (x &&& (256 - x)))) := by
  obtain ⟨_, hp, hxp, hbits⟩ := lowbit_facts ⟨x, hx⟩ h0
  set p := Nat.log2 (x &&& (256 - x))
  have hpm : p ∈ Finset.range 8 := Finset.mem_range.mpr hp
  unfold byteSum
  rw [← Finset.add_sum_erase _ _ hpm, ← Finset.add_sum_erase _ _ hpm]
  have e1 : (x ^^^ (x &&& (256 - x))).testBit p = false := by
    have := hbits ⟨p, hp⟩; simpa using this
  rw [if_pos hxp, if_neg (by simp [e1])]
  have e2 : ∑ i ∈ (Finset.range 8).erase p, (if x.testBit i then GF128.of (f i) else 0) =
      ∑ i ∈ (Finset.range 8).erase p, (if (x ^^^ (x &&& (256 - x))).testBit i then GF128.of (f i) else 0) := by
    refine Finset.sum_congr rfl fun i hi => ?_
    have hi8 : i < 8 := Finset.mem_range.mp (Finset.mem_of_mem_erase hi)
    have hip : i ≠ p := Finset.ne_of_mem_erase hi
    rw [hbits ⟨i, hi8⟩]; simp [hip]
  rw [e2]; ring

theorem subsetGo_get (f : ℕ → F128) : ∀ (fuel x : ℕ) (t : Array F128), x + fuel = 256 → 1 ≤ x → t.size = 256 →
    (∀ y < x, GF128.of t[y]! = byteSum f y) → ∀ y < 256, GF128.of (subsetGo f fuel x t)[y]! = byteSum f y := by
  intro fuel
  induction fuel with
  | zero => intro x t hx _ _ ht y hy; exact ht y (by omega)
  | succ fuel ih =>
    intro x t hx h1 hsz ht
    apply ih (x + 1) _ (by omega) (by omega) (by simp [subsetStep, hsz])
    intro y hy
    simp only [subsetStep, Array.set!]
    have hyt : y < t.size := by rw [hsz]; omega
    rw [getElem!_pos _ y (by simp [hsz]; omega), Array.getElem_setIfInBounds hyt]
    by_cases hyx : x = y
    · subst hyx
      rw [if_pos rfl, of_add]
      have hlt := (lowbit_facts ⟨x, by omega⟩ (show x ≠ 0 by omega)).1
      rw [ht _ hlt, byteSum_step f x (by omega) (by omega)]
    · rw [if_neg hyx, ← getElem!_pos t y (by rw [hsz]; omega)]; exact ht y (by omega)

theorem subsetTab_get (f : ℕ → F128) (y : ℕ) (hy : y < 256) : GF128.of (subsetTab f)[y]! = byteSum f y := by
  apply subsetGo_get f 255 1 _ (by norm_num) le_rfl (by simp) _ y hy
  intro y hy1
  have : y = 0 := by omega
  subst this
  rw [getElem!_pos _ 0 (by simp), Array.getElem_replicate, byteSum_zero]; exact GF128.zero_eq.symm

/-! ### The four bytes of a word -/

theorem sum_4x8 {M : Type} [AddCommMonoid M] (g : ℕ → M) :
    ∑ k ∈ Finset.range 4, ∑ i ∈ Finset.range 8, g (8 * k + i) = ∑ j ∈ Finset.range 32, g j := by
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add]
  abel

/-- `Σ_{j < 32, bit j of w} f(j)`. -/
noncomputable def wordSum (f : ℕ → F128) (w : ℕ) : GF128 :=
  ∑ j ∈ Finset.range 32, if w.testBit j then GF128.of (f j) else 0

theorem byte_bits (w k i : ℕ) (hi : i < 8) : ((w >>> (8 * k)) % 256).testBit i = w.testBit (8 * k + i) := by
  rw [show (256 : ℕ) = 2 ^ 8 from rfl, Nat.testBit_mod_two_pow, Nat.testBit_shiftRight]
  simp [hi, add_comm]

theorem word_split (f : ℕ → F128) (w : ℕ) :
    ∑ k ∈ Finset.range 4, byteSum (fun i => f (8 * k + i)) ((w >>> (8 * k)) % 256) = wordSum f w := by
  rw [wordSum, ← sum_4x8 (fun j => if w.testBit j then GF128.of (f j) else 0)]
  refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun i hi => ?_
  rw [byte_bits _ _ _ (Finset.mem_range.mp hi)]

/-! ### The loops -/

theorem wordSum_zero (f : ℕ → F128) : wordSum f 0 = 0 := by simp [wordSum]

/-- The subset-sum tables of high minterm `h`. -/
def tabsOf (lk : LookupTable) (nh : ℕ) (e : Array F128) (h : ℕ) : Array (Array F128) :=
  (Array.range 4).map fun k => subsetTab fun i => ej lk nh e h (8 * k + i)

theorem tabsOf_get (lk : LookupTable) (nh : ℕ) (e : Array F128) (h k : ℕ) (hk : k < 4) :
    (tabsOf lk nh e h)[k]! = subsetTab fun i => ej lk nh e h (8 * k + i) := by
  rw [tabsOf, getElem!_pos _ k (by simpa using hk)]; simp

/-- The value `v` the executable adds at low minterm `l`: the word's subset sum. -/
theorem v_eq (lk : LookupTable) (nh : ℕ) (e : Array F128) (h w : ℕ) :
    GF128.of (((tabsOf lk nh e h)[0]!)[w % 256]! + ((tabsOf lk nh e h)[1]!)[(w >>> 8) % 256]! +
      ((tabsOf lk nh e h)[2]!)[(w >>> 16) % 256]! + ((tabsOf lk nh e h)[3]!)[(w >>> 24) % 256]!) =
      wordSum (ej lk nh e h) w := by
  rw [← word_split]
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add, of_add]
  rw [tabsOf_get _ _ _ _ 0 (by norm_num), tabsOf_get _ _ _ _ 1 (by norm_num), tabsOf_get _ _ _ _ 2 (by norm_num),
    tabsOf_get _ _ _ _ 3 (by norm_num), subsetTab_get _ _ (Nat.mod_lt _ (by norm_num)),
    subsetTab_get _ _ (Nat.mod_lt _ (by norm_num)), subsetTab_get _ _ (Nat.mod_lt _ (by norm_num)),
    subsetTab_get _ _ (Nat.mod_lt _ (by norm_num))]
  simp

theorem foldBLow_size (lk : LookupTable) (nlo wmask : ℕ) (tabs : Array (Array F128)) (h : ℕ) :
    ∀ (fuel l : ℕ) (acc : Array F128), (foldBLow lk nlo wmask tabs h fuel l acc).size = acc.size := by
  intro fuel; induction fuel with
  | zero => intro l acc; rfl
  | succ fuel ih => intro l acc; simp only [foldBLow]; rw [ih]; split_ifs <;> simp

theorem foldBLow_get (lk : LookupTable) (nh nlo wmask : ℕ) (e : Array F128) (h : ℕ) :
    ∀ (fuel l : ℕ) (acc : Array F128) (c : ℕ), c < acc.size →
      GF128.of (foldBLow lk nlo wmask (tabsOf lk nh e h) h fuel l acc)[c]! = GF128.of acc[c]! +
        ∑ s ∈ Finset.range fuel, if lk.loTop + (l + s) = c then wordSum (ej lk nh e h) (word lk nlo wmask h (l + s)) else 0 := by
  intro fuel
  induction fuel with
  | zero => intro l acc c _; simp [foldBLow]
  | succ fuel ih =>
    intro l acc c hc
    simp only [foldBLow]
    have hsz : c < (if word lk nlo wmask h l == 0 then acc else
        acc.modify (lk.loTop + l) (· + (((tabsOf lk nh e h)[0]!)[word lk nlo wmask h l % 256]! +
          ((tabsOf lk nh e h)[1]!)[(word lk nlo wmask h l >>> 8) % 256]! +
          ((tabsOf lk nh e h)[2]!)[(word lk nlo wmask h l >>> 16) % 256]! +
          ((tabsOf lk nh e h)[3]!)[(word lk nlo wmask h l >>> 24) % 256]!))).size := by
      split_ifs <;> simp [hc]
    rw [ih (l + 1) _ c hsz, Finset.sum_range_succ']
    simp only [show ∀ s, l + 1 + s = l + (s + 1) from fun s => by omega, add_zero]
    split_ifs with hw hlc hlc
    · have h0 : word lk nlo wmask h l = 0 := by simpa using hw
      first | (rw [h0, wordSum_zero]; ring) | ring
    · have h0 : word lk nlo wmask h l = 0 := by simpa using hw
      first | (rw [h0, wordSum_zero]; ring) | ring
    · rw [modify_get! acc _ c _ hc, if_pos hlc, of_add, v_eq]; ring
    · rw [modify_get! acc _ c _ hc, if_neg hlc]; ring

theorem foldBHigh_get (lk : LookupTable) (nh nlo wmask : ℕ) (e : Array F128) :
    ∀ (fuel h : ℕ) (acc : Array F128) (c : ℕ), c < acc.size →
      GF128.of (foldBHigh lk nh nlo wmask e fuel h acc)[c]! = GF128.of acc[c]! +
        ∑ t ∈ Finset.range fuel, ∑ l ∈ Finset.range nlo,
          if lk.loTop + l = c then wordSum (ej lk nh e (h + t)) (word lk nlo wmask (h + t) l) else 0 := by
  intro fuel
  induction fuel with
  | zero => intro h acc c _; simp [foldBHigh]
  | succ fuel ih =>
    intro h acc c hc
    simp only [foldBHigh]
    rw [show (Array.range 4).map (fun k => subsetTab fun i => ej lk nh e h (8 * k + i)) = tabsOf lk nh e h from rfl,
      ih (h + 1) _ c (by rw [foldBLow_size]; exact hc), foldBLow_get lk nh nlo wmask e h nlo 0 acc c hc,
      Finset.sum_range_succ']
    simp only [show ∀ t, h + 1 + t = h + (t + 1) from fun t => by omega, add_zero, zero_add]
    ring

/-- **The lookup fold.** Entry `c` gains `Σ_{h < nh} Σ_{l < nlo, loTop + l = c} Σ_{j : bit j of word(h, l)} e(j, h)`. -/
theorem foldB_get (lk : LookupTable) (e acc0 : Array F128) (c : ℕ) (hc : c < acc0.size) :
    GF128.of (foldB lk e acc0)[c]! = GF128.of acc0[c]! +
      ∑ h ∈ Finset.range (2 ^ (lk.n - lk.lo)), ∑ l ∈ Finset.range (2 ^ lk.lo),
        if lk.loTop + l = c then
          wordSum (ej lk (2 ^ (lk.n - lk.lo)) e h) (word lk (2 ^ lk.lo) (2 ^ lk.bits - 1) h l) else 0 := by
  rw [foldB, foldBHigh_get _ _ _ _ _ _ 0 acc0 c hc]
  simp only [zero_add]

end FlockLevel3
