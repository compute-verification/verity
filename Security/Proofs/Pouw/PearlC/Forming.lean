import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Mathlib.Tactic.GCongr
import Definitions.Pouw.PearlC.Defs
import Proofs.Pouw.PearlC.Peel
import Definitions.Pouw.PearlC.SaltDead
import Proofs.Pouw.Fp8Atom.Fp32Proofs

/-!
# P3: Pearl-C's forming credit (staging, nothing pinned)
-/

namespace Pouw.PearlC.Forming

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.PearlC

/-! ## The E4M3 encoder's error -/

/-- E4M3's spacing at a magnitude `t`: `2^(max (binade t) (−6) − 3)` (the subnormals' `2^−9` below `2^−6`). -/
def e4m3Ulp (t : ℚ) : ℚ := 2 ^ (max (binade t) (-6) - 3)

private theorem abs_roundEven_sub (t : ℚ) : |(roundEven t : ℚ) - t| ≤ 1 / 2 := by
  have h1 := Int.floor_le t
  have h2 := Int.lt_floor_add_one t
  unfold roundEven; dsimp only
  split_ifs with ha hb hc <;> push_cast <;> rw [abs_le] <;> constructor <;> (try simp only [not_lt] at *) <;> linarith

/-- The code `e4m3` writes for a sign offset `s ∈ {0, 128}` and magnitude bits `g ≤ 126`, decoded. -/
private theorem decode_mk {s g : ℕ} (hs : s = 0 ∨ s = 128) (hg : g ≤ 126) :
    decode ⟨(s + g) % 256, Nat.mod_lt _ (by decide)⟩ =
      some ((if s = 128 then -1 else 1) * ((if g / 8 = 0 then g % 8 else 8 + g % 8 : ℕ) : ℚ) *
        (2 : ℚ) ^ (((max (g / 8) 1 : ℕ) : ℤ) - 10)) := by
  have hv : (s + g) % 256 = s + g := Nat.mod_eq_of_lt (by omega)
  have h1 : (s + g) % 128 = g := by omega
  have h2 : ((s + g) >>> 7 == 1) = decide (s = 128) := by
    rw [Nat.shiftRight_eq_div_pow]; rcases hs with rfl | rfl <;> simp <;> omega
  have h3 : ((s + g) >>> 3) % 16 = g / 8 := by
    rw [Nat.shiftRight_eq_div_pow]; rcases hs with rfl | rfl <;> norm_num <;> omega
  have h4 : (s + g) % 8 = g % 8 := by omega
  unfold decode Code.isNaN Code.sign Code.sig Code.lsbExp Code.expOr1 Code.expField Code.manField
  simp only [hv, h1, h2, h3, h4]
  have h5 : (g == 127) = false := by simp; omega
  simp only [h5, Bool.false_eq_true, ↓reduceIte, decide_eq_true_eq]

/-- The value `decode` gives magnitude bits `8e + R`, in the ranges `e4m3Mag` writes, is `R·2^(e − 9)`. -/
private theorem magVal {e R : ℕ} (h : (e = 0 ∧ R ≤ 8) ∨ (8 ≤ R ∧ R ≤ 16)) :
    (((if (8 * e + R) / 8 = 0 then (8 * e + R) % 8 else 8 + (8 * e + R) % 8 : ℕ) : ℚ) *
      (2 : ℚ) ^ (((max ((8 * e + R) / 8) 1 : ℕ) : ℤ) - 10)) = (R : ℚ) * 2 ^ ((e : ℤ) - 9) := by
  rcases h with ⟨rfl, hR⟩ | ⟨hR1, hR2⟩
  · rcases hR.lt_or_eq with hR | rfl
    · have ha : (8 * 0 + R) / 8 = 0 := by omega
      have hb : (8 * 0 + R) % 8 = R := by omega
      rw [ha, hb]; norm_num
    · norm_num
  · rcases hR2.lt_or_eq with hR2 | rfl
    · have ha : (8 * e + R) / 8 = e + 1 := by omega
      have hb : (8 * e + R) % 8 = R - 8 := by omega
      have hc : 8 + (R - 8) = R := by omega
      rw [ha, hb, ite_eq_right (by omega), hc, max_eq_left (by omega)]
      congr 2; push_cast; ring
    · have ha : (8 * e + 16) / 8 = e + 2 := by omega
      have hb : (8 * e + 16) % 8 = 0 := by omega
      rw [ha, hb, ite_eq_right (by omega), max_eq_left (by omega)]
      rw [show ((e + 2 : ℕ) : ℤ) - 10 = ((e : ℤ) - 9) + 1 by push_cast; ring, zpow_add_one₀ two_ne_zero]
      push_cast; ring

/-- **The encoder's value.** Below saturation, `e4m3 y`'s value is `y`'s sign times `|y|` rounded to a whole number of
E4M3 spacings, ties to even. -/
theorem codeVal_e4m3 {y : ℚ} (hy : |y| < 464) :
    codeVal (e4m3 y) = (if y < 0 then -1 else 1) * (roundEven (|y| / e4m3Ulp |y|) : ℚ) * e4m3Ulp |y| := by
  rcases eq_or_ne y 0 with rfl | hy0
  · have h : roundEven 0 = 0 := by simpa using roundEven_intCast 0
    have hd := decode_mk (s := 0) (g := 0) (Or.inl rfl) (by norm_num)
    simp only [e4m3, e4m3Mag, abs_zero, lt_irrefl, ↓reduceIte, codeVal, zero_div, h] at hd ⊢
    rw [hd]; norm_num
  have ht : 0 < |y| := abs_pos.2 hy0
  obtain ⟨hb1, hb2⟩ := binade_spec ht
  have hb9 : binade |y| < 9 := by
    have : (2 : ℚ) ^ binade |y| < 2 ^ (9 : ℤ) := lt_of_le_of_lt hb1 (by norm_num; linarith)
    exact (zpow_lt_zpow_iff_right₀ (one_lt_two (α := ℚ))).1 this
  set E := max (binade |y|) (-6) with hE
  have hu : (0 : ℚ) < 2 ^ (E - 3) := two_zpow_pos _
  set R := roundEven (|y| / 2 ^ (E - 3)) with hR
  have hRlo := floor_le_roundEven (|y| / 2 ^ (E - 3))
  have hRhi := roundEven_le (|y| / 2 ^ (E - 3))
  have hRnear := abs_roundEven_sub (|y| / 2 ^ (E - 3))
  rw [← hR] at hRlo hRhi hRnear
  have k8 : (2 : ℚ) ^ (E - 3) * 8 = 2 ^ E := by
    rw [show (8 : ℚ) = 2 ^ (3 : ℤ) by norm_num, ← zpow_add₀ two_ne_zero, sub_add_cancel]
  have k16 : (2 : ℚ) ^ (E - 3) * 16 = 2 ^ (E + 1) := by
    rw [show (16 : ℚ) = 2 ^ (4 : ℤ) by norm_num, ← zpow_add₀ two_ne_zero]; congr 1; ring
  -- the two ranges of `R`, and the top binade's
  have hrange : (E = -6 ∧ 0 ≤ R ∧ R ≤ 8) ∨ (8 ≤ R ∧ R ≤ 16) := by
    rcases le_or_gt (-6) (binade |y|) with h6 | h6
    · right
      have hEb : E = binade |y| := max_eq_left h6
      have h8 : (8 : ℚ) ≤ |y| / 2 ^ (E - 3) := by
        rw [le_div_iff₀ hu, mul_comm, k8, hEb]; exact hb1
      have h16 : |y| / 2 ^ (E - 3) < 16 := by
        rw [div_lt_iff₀ hu, mul_comm, k16, hEb]; exact hb2
      have f8 : (8 : ℤ) ≤ ⌊|y| / 2 ^ (E - 3)⌋ := Int.le_floor.2 (by exact_mod_cast h8)
      have f16 : ⌊|y| / 2 ^ (E - 3)⌋ < 16 := Int.floor_lt.2 (by exact_mod_cast h16)
      omega
    · left
      have hEb : E = -6 := max_eq_right h6.le
      have h8 : |y| / 2 ^ (E - 3) < 8 := by
        rw [div_lt_iff₀ hu, hEb]
        have : (2 : ℚ) ^ (binade |y| + 1) ≤ 2 ^ (-6 : ℤ) := zpow_le_zpow_right₀ one_le_two (by omega)
        norm_num at this ⊢; linarith
      have f0 : (0 : ℤ) ≤ ⌊|y| / 2 ^ (E - 3)⌋ := Int.floor_nonneg.2 (div_nonneg ht.le hu.le)
      have f8 : ⌊|y| / 2 ^ (E - 3)⌋ < 8 := Int.floor_lt.2 (by exact_mod_cast h8)
      exact ⟨hEb, by omega, by omega⟩
  have htop : E = 8 → R ≤ 14 := by
    intro h8
    have : |y| / 2 ^ (E - 3) < 29 / 2 := by
      rw [h8, div_lt_iff₀ (by norm_num)]; norm_num; linarith
    have h15 : (R : ℚ) < 15 := by rw [abs_le] at hRnear; linarith
    have : R < 15 := by exact_mod_cast h15
    omega
  have hE6 : -6 ≤ E := le_max_right _ _
  have hE8 : E ≤ 8 := max_le (by omega) (by norm_num)
  -- the magnitude bits as naturals
  obtain ⟨e, he⟩ : ∃ e : ℕ, (e : ℤ) = E + 6 := ⟨(E + 6).toNat, Int.toNat_of_nonneg (by omega)⟩
  obtain ⟨Rn, hRn⟩ : ∃ Rn : ℕ, (Rn : ℤ) = R := ⟨R.toNat, Int.toNat_of_nonneg (by omega)⟩
  have hmag : e4m3Mag |y| = 8 * e + Rn := by
    unfold e4m3Mag
    rw [ite_eq_right ht.ne', ite_eq_right (by linarith)]
    show (8 * (E + 6) + R).toNat = 8 * e + Rn
    omega
  have hg : 8 * e + Rn ≤ 126 := by
    rcases hrange with ⟨h1, -, h3⟩ | ⟨-, h3⟩
    · omega
    · rcases (show E ≤ 7 ∨ E = 8 by omega) with h | h
      · omega
      · have := htop h; omega
  have hs : (if y < 0 then 128 else 0) = 0 ∨ (if y < 0 then 128 else 0) = 128 := by split_ifs <;> simp
  have hd := decode_mk hs hg
  have hmv := magVal (e := e) (R := Rn) (by rcases hrange with ⟨h1, -, h3⟩ | ⟨h2, h3⟩ <;> omega)
  have hRq : ((Rn : ℕ) : ℚ) = ((R : ℤ) : ℚ) := by rw [← hRn]; norm_cast
  have hexp : ((e : ℕ) : ℤ) - 9 = E - 3 := by omega
  unfold codeVal
  simp only [e4m3, hmag]
  rw [hd, Option.getD_some, mul_assoc, hmv, hexp, hRq, e4m3Ulp, ← hE, ← hR]
  by_cases hneg : y < 0 <;> simp [hneg]

/-- **The encoder's error.** Below saturation, `e4m3` moves a value by at most `max (|y|/16) 2^−10`. -/
theorem e4m3_err {y : ℚ} (hy : |y| < 464) : |codeVal (e4m3 y) - y| ≤ max (|y| / 16) (1 / 1024) := by
  rcases eq_or_ne y 0 with rfl | hy0
  · have h : roundEven 0 = 0 := by simpa using roundEven_intCast 0
    rw [codeVal_e4m3 hy]; simp [h]
  have ht : 0 < |y| := abs_pos.2 hy0
  have hval := codeVal_e4m3 hy
  set u := e4m3Ulp |y| with hu_def
  have hu : 0 < u := two_zpow_pos _
  have hub : u / 2 ≤ max (|y| / 16) (1 / 1024) := by
    rcases le_or_gt (-6) (binade |y|) with h6 | h6
    · have hE : u = 2 ^ (binade |y| - 3) := by rw [hu_def, e4m3Ulp, max_eq_left h6]
      have k8 : (2 : ℚ) ^ (binade |y| - 3) * 8 = 2 ^ binade |y| := by
        rw [show (8 : ℚ) = 2 ^ (3 : ℤ) by norm_num, ← zpow_add₀ two_ne_zero, sub_add_cancel]
      have hb := (binade_spec ht).1
      apply le_max_of_le_left; rw [hE]; linarith
    · have hE : u = 2 ^ (-9 : ℤ) := by rw [hu_def, e4m3Ulp, max_eq_right h6.le]; norm_num
      apply le_max_of_le_right; rw [hE]; norm_num
  rw [hval]
  generalize hR : ((roundEven (|y| / u) : ℤ) : ℚ) = R
  have hRn := abs_roundEven_sub (|y| / u)
  rw [hR] at hRn
  have hmag : abs (R * u - |y|) ≤ u / 2 := by
    have : R * u - |y| = (R - |y| / u) * u := by rw [sub_mul, div_mul_cancel₀ _ hu.ne']
    rw [this, abs_mul, abs_of_pos hu]
    calc |R - |y| / u| * u ≤ 1 / 2 * u := mul_le_mul_of_nonneg_right hRn hu.le
      _ = u / 2 := by ring
  refine le_trans ?_ (le_trans hmag hub)
  rcases lt_or_ge y 0 with hn | hp
  · have : (-1) * R * u - y = -(R * u - |y|) := by rw [abs_of_neg hn]; ring
    rw [ite_eq_left hn, this, abs_neg]
  · rw [ite_eq_right (not_lt.2 hp), one_mul, abs_of_nonneg hp]

/-- **Separation.** Two values below saturation whose gap exceeds the sum of their encoding errors encode to different
values. -/
theorem codeVal_ne_of_gap {y z : ℚ} (hy : |y| < 464) (hz : |z| < 464)
    (hgap : max (|y| / 16) (1 / 1024) + max (|z| / 16) (1 / 1024) < |y - z|) :
    codeVal (e4m3 y) ≠ codeVal (e4m3 z) := by
  intro h
  have h1 := e4m3_err hy
  have h2 := e4m3_err hz
  rw [← h] at h2
  have h3 := abs_sub_le y (codeVal (e4m3 y)) z
  rw [abs_sub_comm y (codeVal (e4m3 y))] at h3
  linarith

/-! ## (a) Salt-live elements, over an abstract value function -/

/-- A forming value function: from `α`, `x`, `β` and the element's noise rows of codes `e = E[i]`, `f = F[l]` to the
number the cast reads: the signature of a forming chain's value (`FormChain.value`). -/
abbrev ValueFn (r : Nat) := ℚ → ℚ → ℚ → (Fin r → Code) → (Fin r → Code) → ℚ

/-- The noise's support for one element: the pairs `(E row, F row)` the salt can give it (Pearl's `L × L`). -/
abbrev Support (r : Nat) := Set ((Fin r → Code) × (Fin r → Code))

variable {r : Nat}

/-- An element is **salt-live** when its code, `e4m3 ∘ value`, is not constant over the noise's support. -/
def SaltLive (v : ValueFn r) (S : Support r) (α x β : ℚ) : Prop :=
  ∃ p ∈ S, ∃ q ∈ S, e4m3 (v α x β p.1 p.2) ≠ e4m3 (v α x β q.1 q.2)

/-- Salt-live on values: the code's value is not constant over the support. This excludes a salt that only flips the
sign of a zero (`0x00` against `0x80`), which changes the code but not the product. -/
def SaltLiveVal (v : ValueFn r) (S : Support r) (α x β : ℚ) : Prop :=
  ∃ p ∈ S, ∃ q ∈ S, codeVal (e4m3 (v α x β p.1 p.2)) ≠ codeVal (e4m3 (v α x β q.1 q.2))

theorem SaltLiveVal.saltLive {v : ValueFn r} {S : Support r} {α x β : ℚ} (h : SaltLiveVal v S α x β) :
    SaltLive v S α x β := by
  obtain ⟨p, hp, q, hq, h⟩ := h
  exact ⟨p, hp, q, hq, fun h' => h (by rw [h'])⟩

/-- **The core of (a).** A pair `p, q` in the support at which the value is within `η` of `a − N` and `a + N` makes the
element salt-live, if `|a| + 17η < 15N` (the gap beats E4M3's relative error), `N > 2^−10 + η` (it beats the
subnormal spacing) and `|a| + N + η < 464` (no saturation). -/
theorem saltLive_of_pair {v : ValueFn r} {S : Support r} {α x β a N η : ℚ} {p q : (Fin r → Code) × (Fin r → Code)}
    (hp : p ∈ S) (hq : q ∈ S) (hvp : |v α x β p.1 p.2 - (a - N)| ≤ η) (hvq : |v α x β q.1 q.2 - (a + N)| ≤ η)
    (hfloor : 1 / 1024 + η < N) (hthr : |a| + 17 * η < 15 * N) (hsat : |a| + N + η < 464) :
    SaltLiveVal v S α x β := by
  refine ⟨p, hp, q, hq, ?_⟩
  obtain ⟨hp1, hp2⟩ := abs_le.1 hvp
  obtain ⟨hq1, hq2⟩ := abs_le.1 hvq
  have ha1 := neg_abs_le a
  have ha2 := le_abs_self a
  have hy : |v α x β p.1 p.2| ≤ |a| + N + η := abs_le.2 ⟨by linarith, by linarith⟩
  have hz : |v α x β q.1 q.2| ≤ |a| + N + η := abs_le.2 ⟨by linarith, by linarith⟩
  have hgap : 2 * N - 2 * η ≤ |v α x β p.1 p.2 - v α x β q.1 q.2| := by
    rw [abs_sub_comm]; exact le_trans (by linarith) (le_abs_self _)
  have hm : ∀ t : ℚ, |t| ≤ |a| + N + η → max (|t| / 16) (1 / 1024) < N - η := fun t ht =>
    max_lt (by linarith) (by linarith)
  exact codeVal_ne_of_gap (by linarith) (by linarith) (by linarith [hm _ hy, hm _ hz])

/-- **(a) in σ units.** With §5's scale (`α·(|x| + 4σ) ≤ 448`, which `α = 448/(s + 4σ)` and `|x| ≤ s` give), a pair
whose noise is `N = c·ασ` with `c ≤ 4`, and a chain error `η < 16`, the element is salt-live when
`α|x| + 17η < 15·c·ασ` and `2^−10 + η < c·ασ`: in σ units, `|x| < (15c − 17η/(ασ))·σ`. -/
theorem saltLive_sigma {v : ValueFn r} {S : Support r} {α x β σ c η : ℚ} {p q : (Fin r → Code) × (Fin r → Code)}
    (hp : p ∈ S) (hq : q ∈ S) (hvp : |v α x β p.1 p.2 - (α * x - c * (α * σ))| ≤ η)
    (hvq : |v α x β q.1 q.2 - (α * x + c * (α * σ))| ≤ η) (hα : 0 < α) (hασ : 0 ≤ α * σ)
    (hsat : α * (|x| + 4 * σ) ≤ 448) (hc4 : c ≤ 4) (hη : η < 16) (hfloor : 1 / 1024 + η < c * (α * σ))
    (hthr : α * |x| + 17 * η < 15 * c * (α * σ)) : SaltLiveVal v S α x β := by
  have hax : |α * x| = α * |x| := by rw [abs_mul, abs_of_pos hα]
  have hN : c * (α * σ) ≤ 4 * (α * σ) := mul_le_mul_of_nonneg_right hc4 hασ
  exact saltLive_of_pair hp hq hvp hvq hfloor (by rw [hax]; linarith) (by rw [hax]; linarith)

/-- The chain's error on a flip pair, relative to its scale: at `p` and `q` the value is within `ε·(α|x| + N)` of
`α·x − N` and `α·x + N`. This is the named hypothesis the BF16 and atom-mix instances take until the chains' error
lemmas exist; the exact check measures it on Pearl's flat pairs under the atom-mix. -/
def PairErr (v : ValueFn r) (α x β N ε : ℚ) (p q : (Fin r → Code) × (Fin r → Code)) : Prop :=
  |v α x β p.1 p.2 - (α * x - N)| ≤ ε * (α * |x| + N) ∧ |v α x β q.1 q.2 - (α * x + N)| ≤ ε * (α * |x| + N)

/-- **(a) for a chain with relative error `ε ≤ 1/32`.** Under §5's scale, the noise floor `ασ ≥ 1` and a flip pair of
noise `c·ασ` with `8/15 ≤ c ≤ 4`, the element is salt-live when `(1 + 17ε)·|x| < (15 − 17ε)·c·σ`. -/
theorem saltLive_rel {v : ValueFn r} {S : Support r} {α x β σ c ε : ℚ} {p q : (Fin r → Code) × (Fin r → Code)}
    (hp : p ∈ S) (hq : q ∈ S) (herr : PairErr v α x β (c * (α * σ)) ε p q) (hα : 0 < α) (hfloor : 1 ≤ α * σ)
    (hsat : α * (|x| + 4 * σ) ≤ 448) (hc : 8 / 15 ≤ c) (hc4 : c ≤ 4) (hε0 : 0 ≤ ε) (hε : ε ≤ 1 / 32)
    (hx : (1 + 17 * ε) * |x| < (15 - 17 * ε) * c * σ) : SaltLiveVal v S α x β := by
  have hA0 : 0 ≤ α * |x| := mul_nonneg hα.le (abs_nonneg x)
  have hM : 8 / 15 ≤ c * (α * σ) := by
    calc (8 / 15 : ℚ) = 8 / 15 * 1 := by ring
      _ ≤ c * (α * σ) := mul_le_mul hc hfloor (by norm_num) (by linarith)
  have hM4 : c * (α * σ) ≤ 4 * (α * σ) := mul_le_mul_of_nonneg_right hc4 (by linarith)
  have hAx : α * ((1 + 17 * ε) * |x|) < α * ((15 - 17 * ε) * c * σ) := mul_lt_mul_of_pos_left hx hα
  have hεA : 0 ≤ ε * (α * |x|) := mul_nonneg hε0 hA0
  have hεM : 0 ≤ ε * (c * (α * σ)) := mul_nonneg hε0 (by linarith)
  have hA15 : α * |x| < 15 * (c * (α * σ)) := by nlinarith
  have h1 : ε * (α * |x|) ≤ ε * (15 * (c * (α * σ))) := mul_le_mul_of_nonneg_left hA15.le hε0
  have h2 : ε * (c * (α * σ)) ≤ 1 / 32 * (c * (α * σ)) := mul_le_mul_of_nonneg_right hε (by linarith)
  have h3 : ε * (α * |x| + c * (α * σ)) ≤ 1 / 32 * (α * |x| + c * (α * σ)) :=
    mul_le_mul_of_nonneg_right hε (by linarith)
  exact saltLive_sigma hp hq herr.1 herr.2 hα (by linarith) hsat hc4 (by nlinarith) (by nlinarith) (by nlinarith)

/-! ### The noise's support: flat lines -/

/-- A flat `E` line (norm 256): every entry the code of `±44`, negative where `s` is true. -/
def flat (s : Fin r → Bool) : Fin r → Code := fun t => e4m3 (if s t then -44 else 44)

/-- A flat `F` line (Π's norm 16): every entry the code of `±11/4`, negative where `s` is true. -/
def flatF (s : Fin r → Bool) : Fin r → Code := fun t => e4m3 (if s t then -(11 / 4) else 11 / 4)

/-- A line with every code's sign bit flipped (`Peel.negCode`). -/
def negLine (e : Fin r → Code) : Fin r → Code := fun t => Peel.negCode (e t)

theorem codeVal_negLine (e : Fin r → Code) (t : Fin r) : codeVal (negLine e t) = -codeVal (e t) :=
  Peel.codeVal_negCode _

theorem codeVal_flat (s : Fin r → Bool) (t : Fin r) : codeVal (flat s t) = if s t then -44 else 44 := by
  unfold flat; cases s t <;> decide +kernel

theorem codeVal_flatF (s : Fin r → Bool) (t : Fin r) : codeVal (flatF s t) = if s t then -(11 / 4) else 11 / 4 := by
  unfold flatF; cases s t <;> decide +kernel

/-- The support fact the instances use: every pair of a flat `E` line and a flat `F` line is in the support. Pearl's
line rule gives `±44` at norm 256 and `±11/4` at norm 16 at every entry when all `b_t mod 128` are equal, with the signs
free (the exact check, over all 128 magnitudes). -/
def FlatPairs (S : Support r) : Prop := ∀ s s' : Fin r → Bool, (flat s, flatF s') ∈ S

theorem neg_flat (s : Fin r → Bool) : negLine (flat s) = flat (fun t => !s t) := by
  funext t
  rcases Bool.eq_false_or_eq_true (s t) with h | h <;>
    simp only [negLine, flat, h, Bool.not_true, Bool.not_false, Bool.false_eq_true, ↓reduceIte] <;> decide +kernel

theorem neg_flatF (s : Fin r → Bool) : negLine (flatF s) = flatF (fun t => !s t) := by
  funext t
  rcases Bool.eq_false_or_eq_true (s t) with h | h <;>
    simp only [negLine, flatF, h, Bool.not_true, Bool.not_false, Bool.false_eq_true, ↓reduceIte] <;> decide +kernel

/-- Flat lines give flip pairs: `(−e, f)` and `(e, f)` are both in the support. -/
theorem flipPair_of_flat {S : Support r} (hS : FlatPairs S) (s s' : Fin r → Bool) :
    (negLine (flat s), flatF s') ∈ S ∧ (flat s, flatF s') ∈ S := by
  rw [neg_flat]; exact ⟨hS _ _, hS _ _⟩

/-- A flat `E` line and a flat `F` line with `j` sign disagreements have dot product `121·(r − 2j)`. -/
theorem dot_flat (s s' : Fin r → Bool) :
    ∑ t, codeVal (flat s t) * codeVal (flatF s' t) =
      121 * ((r : ℚ) - 2 * (Finset.univ.filter fun t => s t ≠ s' t).card) := by
  have h : ∀ t, codeVal (flat s t) * codeVal (flatF s' t) = 121 - 242 * (if s t ≠ s' t then 1 else 0) := by
    intro t; simp only [codeVal_flat, codeVal_flatF]; cases s t <;> cases s' t <;> norm_num
  simp only [h, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    ← Finset.mul_sum, Finset.sum_boole]
  ring

/-! ### The exact instance -/

/-- The exact value function, `α·x + β·(e·f)`: `FormChain.exact`'s. -/
def exactValue (r : Nat) : ValueFn r := fun α x β e f => α * x + β * ∑ t, codeVal (e t) * codeVal (f t)

/-- **(a), exact.** A flip pair `(−e, f)`, `(e, f)` in the support with noise `N = β·(e·f)` makes the element salt-live
when `α|x| < 15N`, `N > 2^−10` and `α|x| + N < 464`. -/
theorem exact_saltLive {S : Support r} {α x β : ℚ} {e f : Fin r → Code} (hp : (negLine e, f) ∈ S) (hq : (e, f) ∈ S)
    (hfloor : 1 / 1024 < β * ∑ t, codeVal (e t) * codeVal (f t))
    (hthr : |α * x| < 15 * (β * ∑ t, codeVal (e t) * codeVal (f t)))
    (hsat : |α * x| + β * ∑ t, codeVal (e t) * codeVal (f t) < 464) : SaltLiveVal (exactValue r) S α x β := by
  have hneg : ∑ t, codeVal (negLine e t) * codeVal (f t) = -∑ t, codeVal (e t) * codeVal (f t) := by
    simp [codeVal_negLine, Finset.sum_neg_distrib]
  refine saltLive_of_pair (a := α * x) (N := β * ∑ t, codeVal (e t) * codeVal (f t)) (η := 0) hp hq ?_ ?_
    (by linarith) (by linarith) (by linarith)
  · show |α * x + β * ∑ t, codeVal (negLine e t) * codeVal (f t) - (α * x - β * ∑ t, codeVal (e t) * codeVal (f t))| ≤ 0
    rw [hneg, show α * x + β * -∑ t, codeVal (e t) * codeVal (f t) -
      (α * x - β * ∑ t, codeVal (e t) * codeVal (f t)) = 0 by ring, abs_zero]
  · show |α * x + β * ∑ t, codeVal (e t) * codeVal (f t) - (α * x + β * ∑ t, codeVal (e t) * codeVal (f t))| ≤ 0
    rw [sub_self, abs_zero]

/-- **(a), exact, in σ units.** Under §5's scale and the noise floor `ασ ≥ 1`, a flip pair of noise
`β·(e·f) = c·ασ` with `2^−10 < c ≤ 4` makes the element salt-live when `|x| < 15·c·σ`. -/
theorem exact_saltLive_sigma {S : Support r} {α x β σ c : ℚ} {e f : Fin r → Code} (hp : (negLine e, f) ∈ S)
    (hq : (e, f) ∈ S) (hα : 0 < α) (hfloor : 1 ≤ α * σ) (hsat : α * (|x| + 4 * σ) ≤ 448) (hc0 : 1 / 1024 < c)
    (hc4 : c ≤ 4) (hN : β * ∑ t, codeVal (e t) * codeVal (f t) = c * (α * σ)) (hx : |x| < 15 * c * σ) :
    SaltLiveVal (exactValue r) S α x β := by
  have hneg : ∑ t, codeVal (negLine e t) * codeVal (f t) = -∑ t, codeVal (e t) * codeVal (f t) := by
    simp [codeVal_negLine, Finset.sum_neg_distrib]
  have hfl : 1 / 1024 < c * (α * σ) := by
    calc (1 / 1024 : ℚ) < c * 1 := by linarith
      _ ≤ c * (α * σ) := mul_le_mul_of_nonneg_left hfloor (by linarith)
  have hthr : α * |x| < α * (15 * c * σ) := mul_lt_mul_of_pos_left hx hα
  refine saltLive_sigma (η := 0) (c := c) (σ := σ) hp hq ?_ ?_ hα (by linarith) hsat hc4 (by norm_num)
    (by linarith) (by linarith)
  · show |α * x + β * ∑ t, codeVal (negLine e t) * codeVal (f t) - (α * x - c * (α * σ))| ≤ 0
    rw [hneg, mul_neg, hN, show α * x + -(c * (α * σ)) - (α * x - c * (α * σ)) = 0 by ring, abs_zero]
  · show |α * x + β * ∑ t, codeVal (e t) * codeVal (f t) - (α * x + c * (α * σ))| ≤ 0
    rw [hN, sub_self, abs_zero]

/-! ### The three chains -/

theorem exactValue_eq (r : Nat) : exactValue r = (FormChain.exact r).value := rfl

/-- **(a) for `FormChain.exact`**: `exact_saltLive_sigma` on the chain's value. -/
theorem exactChain_saltLive {S : Support r} {α x β σ c : ℚ} {e f : Fin r → Code} (hp : (negLine e, f) ∈ S)
    (hq : (e, f) ∈ S) (hα : 0 < α) (hfloor : 1 ≤ α * σ) (hsat : α * (|x| + 4 * σ) ≤ 448) (hc0 : 1 / 1024 < c)
    (hc4 : c ≤ 4) (hN : β * ∑ t, codeVal (e t) * codeVal (f t) = c * (α * σ)) (hx : |x| < 15 * c * σ) :
    SaltLiveVal (FormChain.exact r).value S α x β :=
  exact_saltLive_sigma hp hq hα hfloor hsat hc0 hc4 hN hx

private theorem bf16_chain_err {a n n0 η : ℚ} (hη : η ≤ |n0| / 256) (hn : |n - n0| ≤ η) :
    |bf16Rn (a + bf16Rn n) - (a + n0)| ≤ 1 / 64 * (|a| + |n0|) := by
  have e1 := Peel.bf16Rn_sub_le n
  have e2 := Peel.bf16Rn_sub_le (a + bf16Rn n)
  obtain ⟨h1, h2⟩ := abs_le.1 e1
  obtain ⟨h3, h4⟩ := abs_le.1 e2
  obtain ⟨h5, h6⟩ := abs_le.1 hn
  have ha1 := neg_abs_le a
  have ha2 := le_abs_self a
  have hm1 := neg_abs_le n0
  have hm2 := le_abs_self n0
  have hA := abs_nonneg a
  have hM := abs_nonneg n0
  have hnn : |n| ≤ |n0| + η := abs_le.2 ⟨by linarith, by linarith⟩
  have hy : |a + bf16Rn n| ≤ |a| + |n0| + η + (|n0| + η) / 256 :=
    abs_le.2 ⟨by norm_num at h1 h2 ⊢; linarith, by norm_num at h1 h2 ⊢; linarith⟩
  refine abs_le.2 ⟨?_, ?_⟩ <;> norm_num at h1 h2 h3 h4 ⊢ <;> linarith

/-- **Pearl's chain has relative error `1/64` on a pair**, if the noise wgmma (its accumulation order is a parameter) is
within `N/256` of `∓N` at the pair's two points. -/
theorem bf16_pairErr {noise : (Fin r → ℚ) → (Fin r → ℚ) → ℚ} {α x β N η : ℚ}
    {p q : (Fin r → Code) × (Fin r → Code)} (hα : 0 ≤ α) (hN : 0 ≤ N) (hη : η ≤ N / 256)
    (hp : |noise (fun t => β * codeVal (p.1 t)) (fun t => codeVal (p.2 t)) - -N| ≤ η)
    (hq : |noise (fun t => β * codeVal (q.1 t)) (fun t => codeVal (q.2 t)) - N| ≤ η) :
    PairErr (FormChain.bf16 r noise).value α x β N (1 / 64) p q := by
  have hax : |α * x| = α * |x| := by rw [abs_mul, abs_of_nonneg hα]
  constructor
  · have h := bf16_chain_err (a := α * x) (n0 := -N) (by rw [abs_neg, abs_of_nonneg hN]; exact hη) hp
    rw [hax, abs_neg, abs_of_nonneg hN, ← sub_eq_add_neg] at h
    exact h
  · have h := bf16_chain_err (a := α * x) (n0 := N) (by rw [abs_of_nonneg hN]; exact hη) hq
    rw [hax, abs_of_nonneg hN] at h
    exact h

/-- **(a) for `FormChain.bf16`** (Pearl's chain): with the wgmma within `N/256` of `∓N`, the element is salt-live when
`(81/64)·|x| < (943/64)·c·σ`, about `|x| < 11.6·c·σ`. -/
theorem bf16Chain_saltLive {noise : (Fin r → ℚ) → (Fin r → ℚ) → ℚ} {S : Support r} {α x β σ c η : ℚ}
    {p q : (Fin r → Code) × (Fin r → Code)} (hpS : p ∈ S) (hqS : q ∈ S)
    (hp : |noise (fun t => β * codeVal (p.1 t)) (fun t => codeVal (p.2 t)) - -(c * (α * σ))| ≤ η)
    (hq : |noise (fun t => β * codeVal (q.1 t)) (fun t => codeVal (q.2 t)) - c * (α * σ)| ≤ η)
    (hη : η ≤ c * (α * σ) / 256) (hα : 0 < α) (hfloor : 1 ≤ α * σ) (hsat : α * (|x| + 4 * σ) ≤ 448)
    (hc : 8 / 15 ≤ c) (hc4 : c ≤ 4) (hx : (1 + 17 * (1 / 64)) * |x| < (15 - 17 * (1 / 64)) * c * σ) :
    SaltLiveVal (FormChain.bf16 r noise).value S α x β := by
  have hN : 0 ≤ c * (α * σ) := mul_nonneg (by linarith) (by linarith)
  exact saltLive_rel hpS hqS (bf16_pairErr hα.le hN hη hp hq) hα hfloor hsat hc hc4 (by norm_num) (by norm_num) hx

/-- **(a) for `FormChain.atomMix`** (§5's chain): with the atom's relative error `ε ≤ 1/32` on the pair (`PairErr`, the
named hypothesis until `hopperStep` has an error lemma; the exact check measures `ε ≤ 0.00228` on Pearl's flat pairs),
the element is salt-live when `(1 + 17ε)·|x| < (15 − 17ε)·c·σ`. -/
theorem atomMixChain_saltLive {S : Support 32} {α x β σ c ε : ℚ} {p q : (Fin 32 → Code) × (Fin 32 → Code)} (hpS : p ∈ S)
    (hqS : q ∈ S) (herr : PairErr FormChain.atomMix.value α x β (c * (α * σ)) ε p q) (hα : 0 < α)
    (hfloor : 1 ≤ α * σ) (hsat : α * (|x| + 4 * σ) ≤ 448) (hc : 8 / 15 ≤ c) (hc4 : c ≤ 4) (hε0 : 0 ≤ ε)
    (hε : ε ≤ 1 / 32) (hx : (1 + 17 * ε) * |x| < (15 - 17 * ε) * c * σ) :
    SaltLiveVal FormChain.atomMix.value S α x β :=
  saltLive_rel hpS hqS herr hα hfloor hsat hc hc4 hε0 hε hx

/-! ## (b) The credit: every stage of a salt-live element reads the salt -/

/-- A function of the noise varies over the support: it takes two different values on it. -/
def Varies {γ : Type*} (S : Support r) (g : (Fin r → Code) × (Fin r → Code) → γ) : Prop :=
  ∃ p ∈ S, ∃ q ∈ S, g p ≠ g q

/-- A stage whose output varies has an input that varies: the chain's earlier stages all read the salt. -/
theorem Varies.of_comp {γ γ' : Type*} {S : Support r} {h : (Fin r → Code) × (Fin r → Code) → γ} (g : γ → γ')
    (hv : Varies S (g ∘ h)) : Varies S h := by
  obtain ⟨p, hp, q, hq, hne⟩ := hv
  exact ⟨p, hp, q, hq, fun h' => hne (by simp only [Function.comp, h'])⟩

/-- A salt-live element's cast output varies. -/
theorem SaltLive.varies {v : ValueFn r} {S : Support r} {α x β : ℚ} (h : SaltLive v S α x β) :
    Varies S (fun p => e4m3 (v α x β p.1 p.2)) := h

/-- At a salt-live element, the value the cast reads varies. -/
theorem SaltLive.value {v : ValueFn r} {S : Support r} {α x β : ℚ} (h : SaltLive v S α x β) :
    Varies S (fun p => v α x β p.1 p.2) :=
  Varies.of_comp e4m3 h.varies

/-- At a salt-live element, every stage the value factors through varies: if `v α x β e f = g (h e f)`, then `h`'s
output varies over the support (for the atom-mix, the atom's word; for Pearl's chain, the noise wgmma's output). -/
theorem SaltLive.stage {v : ValueFn r} {S : Support r} {α x β : ℚ} {γ : Type*}
    (h : (Fin r → Code) → (Fin r → Code) → γ) (g : γ → ℚ) (hv : ∀ e f, v α x β e f = g (h e f))
    (hl : SaltLive v S α x β) : Varies S (fun p => h p.1 p.2) := by
  have hvar := hl.varies
  have heq : (fun p : (Fin r → Code) × (Fin r → Code) => e4m3 (v α x β p.1 p.2)) = (e4m3 ∘ g) ∘ fun p => h p.1 p.2 := by
    funext p; simp only [Function.comp, hv]
  rw [heq] at hvar
  exact Varies.of_comp _ hvar

/-- The forming steps credited at a salt-live element (H100 units): Pearl's BF16 chain (the noise wgmma 32, f32→bf16 32,
the HFMA2 16, the cast 32). The row statistic reads `A` alone and is not credited. -/
def bf16Steps : List ℕ := [32, 32, 16, 32]

/-- §5's atom-mix: the noise atom 32 and the cast 32 (`Game.lean`'s `Costs.adopted.fs`). -/
def atomSteps : List ℕ := [32, 32]

theorem bf16Steps_sum : bf16Steps.sum = 112 := rfl

theorem atomSteps_sum : atomSteps.sum = 64 := rfl

/-! ## (c) The budget ρ_D -/

variable {m k : Nat}

/-- The loud entries at per-row thresholds `T`: `|X i l| ≥ T i`. -/
def loud (T : Fin m → ℚ) (X : Matrix (Fin m) (Fin k) ℚ) : Finset (Fin m × Fin k) :=
  Finset.univ.filter fun il => T il.1 ≤ |X il.1 il.2|

/-- Domain D's liveness rule in Pearl's aggregate form, for statistic `ρ` (the noise's `σ_i = δ·ρ_i`): at most
`m·k/64` entries have `|x| ≥ 8δ·ρ_i`. -/
def LiveOperand (δ : ℚ) (ρ : Fin m → ℚ) (X : Matrix (Fin m) (Fin k) ℚ) : Prop :=
  64 * (loud (fun i => 8 * δ * ρ i) X).card ≤ m * k

open Classical in
/-- The operand's salt-live elements. -/
noncomputable def live (v : ValueFn r) (S : Support r) (α β : Fin m → ℚ) (X : Matrix (Fin m) (Fin k) ℚ) :
    Finset (Fin m × Fin k) :=
  Finset.univ.filter fun il => SaltLiveVal v S (α il.1) (X il.1 il.2) (β il.1)

theorem card_loud (T : Fin m → ℚ) (X : Matrix (Fin m) (Fin k) ℚ) :
    (loud T X).card = ∑ i, (Finset.univ.filter fun l => T i ≤ |X i l|).card := by
  unfold loud
  rw [Finset.card_filter, Fintype.sum_prod_type]
  simp only [Finset.card_filter]

/-- Rows meeting `Defs.lean`'s `LiveRow` under a statistic `st` make the operand live, for any `ρ` with `ρ² ≥ st`. -/
theorem liveOperand_of_rows {δ : ℚ} {st : Stat} {ρ : Fin m → ℚ} {X : Matrix (Fin m) (Fin k) ℚ} (hδ : 0 ≤ δ)
    (hρ : ∀ i, 0 ≤ ρ i ∧ st k (X i) ≤ ρ i ^ 2) (h : ∀ i, LiveRow δ (X i) st) : LiveOperand δ ρ X := by
  unfold LiveOperand
  rw [card_loud, Finset.mul_sum]
  calc ∑ i, 64 * (Finset.univ.filter fun l => 8 * δ * ρ i ≤ |X i l|).card ≤ ∑ _i : Fin m, k := by
        refine Finset.sum_le_sum fun i _ => le_trans (Nat.mul_le_mul_left 64 (Finset.card_le_card ?_)) (h i)
        intro l hl
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hl ⊢
        have h0 : 0 ≤ 8 * δ * ρ i := by have := (hρ i).1; positivity
        have h1 := mul_self_le_mul_self h0 hl
        have h2 : 64 * δ ^ 2 * st k (X i) ≤ 64 * δ ^ 2 * ρ i ^ 2 :=
          mul_le_mul_of_nonneg_left (hρ i).2 (by positivity)
        nlinarith [abs_mul_abs_self (X i l)]
    _ = m * k := by simp

/-- At `δ ≥ 1` with the full row's RMS, `LiveRow` holds for every nonzero row (Markov): the rule binds only under a
statistic that reads part of the row, such as §5's every-8th RMS. -/
theorem liveRow_of_meanSq_pos {δ : ℚ} {x : Fin k → ℚ} (hδ : 1 ≤ δ) (hx : 0 < meanSq x) : LiveRow δ x := by
  unfold LiveRow
  set F := Finset.univ.filter fun l => 64 * δ ^ 2 * meanSq x ≤ x l ^ 2
  have hk : 0 < k := by
    rcases Nat.eq_zero_or_pos k with rfl | h
    · simp [meanSq] at hx
    · exact h
  have hsum : ∑ l, x l ^ 2 = k * meanSq x := by unfold meanSq; field_simp
  have hm : (F.card : ℚ) * (64 * δ ^ 2 * meanSq x) ≤ k * meanSq x := by
    rw [← hsum]
    calc (F.card : ℚ) * (64 * δ ^ 2 * meanSq x) = ∑ _l ∈ F, 64 * δ ^ 2 * meanSq x := by
          rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ ∑ l ∈ F, x l ^ 2 := Finset.sum_le_sum fun l hl => (Finset.mem_filter.1 hl).2
      _ ≤ ∑ l, x l ^ 2 :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) fun _ _ _ => sq_nonneg _
  have hδ2 : 1 ≤ δ ^ 2 := by nlinarith
  have hc : (64 : ℚ) * F.card ≤ k := by
    have h1 : (F.card : ℚ) * 64 * δ ^ 2 ≤ k := by
      rw [show (F.card : ℚ) * (64 * δ ^ 2 * meanSq x) = F.card * 64 * δ ^ 2 * meanSq x by ring] at hm
      exact le_of_mul_le_mul_right hm hx
    have h2 : (F.card : ℚ) * 64 ≤ F.card * 64 * δ ^ 2 := by
      have : (0 : ℚ) ≤ F.card * 64 := by positivity
      nlinarith
    linarith
  exact_mod_cast hc

/-- Elements below the thresholds `T` that are all salt-live leave at most the loud ones dead. -/
theorem card_live_ge {v : ValueFn r} {S : Support r} {α β T : Fin m → ℚ} {X : Matrix (Fin m) (Fin k) ℚ}
    (hcov : ∀ i l, |X i l| < T i → SaltLiveVal v S (α i) (X i l) (β i)) :
    m * k ≤ (live v S α β X).card + (loud T X).card := by
  classical
  have hsub : (Finset.univ : Finset (Fin m × Fin k)) ⊆ live v S α β X ∪ loud T X := by
    intro il _
    by_cases h : T il.1 ≤ |X il.1 il.2|
    · exact Finset.mem_union_right _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩)
    · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, hcov _ _ (not_le.1 h)⟩)
  calc m * k = (Finset.univ : Finset (Fin m × Fin k)).card := by simp
    _ ≤ (live v S α β X ∪ loud T X).card := Finset.card_le_card hsub
    _ ≤ _ := Finset.card_union_le _ _

/-- ρ_D, the fraction of an operand's credited forming the liveness rule leaves unbacked. -/
def rhoD : ℚ := 1 / 64

/-- **The forming budget.** If every element below `8δ·ρ_i` is salt-live and the operand meets the liveness rule, at
least `(1 − ρ_D)·m·k` elements are salt-live. -/
theorem formingBudget {v : ValueFn r} {S : Support r} {δ : ℚ} {α β ρ : Fin m → ℚ} {X : Matrix (Fin m) (Fin k) ℚ}
    (hcov : ∀ i l, |X i l| < 8 * δ * ρ i → SaltLiveVal v S (α i) (X i l) (β i)) (hlive : LiveOperand δ ρ X) :
    (1 - rhoD) * ((m : ℚ) * k) ≤ (live v S α β X).card := by
  have h1 : ((m * k : ℕ) : ℚ) ≤ ((live v S α β X).card + (loud (fun i => 8 * δ * ρ i) X).card : ℕ) := by
    exact_mod_cast card_live_ge hcov
  have h2 : ((64 * (loud (fun i => 8 * δ * ρ i) X).card : ℕ) : ℚ) ≤ ((m * k : ℕ) : ℚ) := by exact_mod_cast hlive
  push_cast at h1 h2
  unfold rhoD; linarith

/-- **The forming credit.** For `fs` credited units per salt-live element, `creditOf`'s forming term `fs·m·k`, less
`ρ_D·fs·m·k`, is backed by salt-live elements. -/
theorem formingCredit (fs : ℕ) {v : ValueFn r} {S : Support r} {δ : ℚ} {α β ρ : Fin m → ℚ}
    {X : Matrix (Fin m) (Fin k) ℚ} (hcov : ∀ i l, |X i l| < 8 * δ * ρ i → SaltLiveVal v S (α i) (X i l) (β i))
    (hlive : LiveOperand δ ρ X) :
    (fs : ℚ) * m * k - rhoD * fs * m * k ≤ fs * (live v S α β X).card := by
  have h := mul_le_mul_of_nonneg_left (formingBudget hcov hlive) (Nat.cast_nonneg (α := ℚ) fs)
  linarith

/-- **ρ_D at Π's parameters** (`Params.pi`: `δ = 1`, `r = 32`), for a `Shape`'s operand `A` (`m × k`) and any value
function: if every element below `8·δ·ρ_i` is salt-live and the liveness rule holds, the forming term
`fs·m·k` of `creditOf` exceeds the salt-live elements' credit by at most `ρ_D·fs·m·k`, with `ρ_D = 1/64`. -/
theorem rhoD_pi (fs : ℕ) (s : Shape) {v : ValueFn Params.pi.r} {S : Support Params.pi.r} {α β ρ : Fin s.m → ℚ}
    {X : Matrix (Fin s.m) (Fin s.k) ℚ}
    (hcov : ∀ i l, |X i l| < 8 * Params.pi.δ * ρ i → SaltLiveVal v S (α i) (X i l) (β i))
    (hlive : LiveOperand Params.pi.δ ρ X) :
    (fs : ℚ) * s.m * s.k - rhoD * fs * s.m * s.k ≤ fs * (live v S α β X).card ∧ rhoD = 1 / 64 :=
  ⟨formingCredit fs hcov hlive, rfl⟩

/-- **ρ_D for the exact chain at Π's parameters.** Under §5's scale (`α_i·(|x| + 4ρ_i) ≤ 448`), the noise floor
`α_i·ρ_i ≥ 1` and, per row, a flip pair of noise `c_i·α_iρ_i` with `8/15 ≤ c_i ≤ 4`, every element below `8ρ_i` is
salt-live, so the liveness rule gives the budget with `ρ_D = 1/64`. -/
theorem rhoD_exact_pi (fs : ℕ) (s : Shape) {S : Support Params.pi.r} {α β ρ : Fin s.m → ℚ}
    {X : Matrix (Fin s.m) (Fin s.k) ℚ} (hα : ∀ i, 0 < α i) (hfloor : ∀ i, 1 ≤ α i * ρ i)
    (hsat : ∀ i l, α i * (|X i l| + 4 * ρ i) ≤ 448)
    (hpair : ∀ i, ∃ (e f : Fin Params.pi.r → Code) (c : ℚ), (negLine e, f) ∈ S ∧ (e, f) ∈ S ∧ 8 / 15 ≤ c ∧ c ≤ 4 ∧
      β i * ∑ t, codeVal (e t) * codeVal (f t) = c * (α i * ρ i))
    (hlive : LiveOperand Params.pi.δ ρ X) :
    (fs : ℚ) * s.m * s.k - rhoD * fs * s.m * s.k ≤
      fs * (live (FormChain.exact Params.pi.r).value S α β X).card := by
  refine (rhoD_pi fs s ?_ hlive).1
  intro i l hx
  obtain ⟨e, f, c, hp, hq, hc, hc4, hN⟩ := hpair i
  have hδ : Params.pi.δ = 1 := rfl
  rw [hδ] at hx
  have hρ : 0 < ρ i := by
    by_contra h
    have := mul_nonpos_of_nonneg_of_nonpos (hα i).le (not_lt.1 h)
    linarith [hfloor i]
  exact exactChain_saltLive hp hq (hα i) (hfloor i) (hsat i l) (by linarith) hc4 hN (by nlinarith)

/-- **ρ_D for a chain of relative error `ε` at Π's parameters.** Per row, a flip pair of noise `c_i·α_iρ_i` on which
the chain meets `PairErr` with `ε ≤ 1/32`, and `8·(1 + 17ε) ≤ (15 − 17ε)·c_i ≤ 60` so that (a) covers `8ρ_i`; with §5's
scale and the noise floor, the liveness rule gives the budget with `ρ_D = 1/64`. -/
theorem rhoD_rel_pi (fs : ℕ) (s : Shape) {v : ValueFn Params.pi.r} {S : Support Params.pi.r} {α β ρ c : Fin s.m → ℚ}
    {ε : ℚ} {X : Matrix (Fin s.m) (Fin s.k) ℚ} {p q : Fin s.m → (Fin Params.pi.r → Code) × (Fin Params.pi.r → Code)}
    (hpS : ∀ i, p i ∈ S) (hqS : ∀ i, q i ∈ S)
    (herr : ∀ i l, PairErr v (α i) (X i l) (β i) (c i * (α i * ρ i)) ε (p i) (q i)) (hα : ∀ i, 0 < α i)
    (hfloor : ∀ i, 1 ≤ α i * ρ i) (hsat : ∀ i l, α i * (|X i l| + 4 * ρ i) ≤ 448)
    (hc : ∀ i, 8 * (1 + 17 * ε) ≤ (15 - 17 * ε) * c i) (hc4 : ∀ i, c i ≤ 4) (hε0 : 0 ≤ ε) (hε : ε ≤ 1 / 32)
    (hlive : LiveOperand Params.pi.δ ρ X) :
    (fs : ℚ) * s.m * s.k - rhoD * fs * s.m * s.k ≤ fs * (live v S α β X).card := by
  refine (rhoD_pi fs s ?_ hlive).1
  intro i l hx
  have hδ : Params.pi.δ = 1 := rfl
  rw [hδ] at hx
  have hρ : 0 < ρ i := by
    by_contra h
    have := mul_nonpos_of_nonneg_of_nonpos (hα i).le (not_lt.1 h)
    linarith [hfloor i]
  have hci := hc i
  have hc0 : 0 < c i := by nlinarith
  have hc815 : 8 / 15 ≤ c i := by nlinarith
  have h1 : (1 + 17 * ε) * |X i l| < (1 + 17 * ε) * (8 * ρ i) := mul_lt_mul_of_pos_left (by linarith) (by linarith)
  have h2 : 8 * (1 + 17 * ε) * ρ i ≤ (15 - 17 * ε) * c i * ρ i := mul_le_mul_of_nonneg_right hci hρ.le
  exact saltLive_rel (hpS i) (hqS i) (herr i l) (hα i) (hfloor i) (hsat i l) hc815 (hc4 i) hε0 hε (by linarith)

/-- **ρ_D for §5's atom-mix at Π's parameters**, with `PairErr` (the named hypothesis) at relative error `ε`. -/
theorem rhoD_atomMix_pi (fs : ℕ) (s : Shape) {S : Support Params.pi.r} {α β ρ c : Fin s.m → ℚ} {ε : ℚ}
    {X : Matrix (Fin s.m) (Fin s.k) ℚ} {p q : Fin s.m → (Fin Params.pi.r → Code) × (Fin Params.pi.r → Code)}
    (hpS : ∀ i, p i ∈ S) (hqS : ∀ i, q i ∈ S)
    (herr : ∀ i l, PairErr FormChain.atomMix.value (α i) (X i l) (β i) (c i * (α i * ρ i)) ε (p i) (q i))
    (hα : ∀ i, 0 < α i) (hfloor : ∀ i, 1 ≤ α i * ρ i) (hsat : ∀ i l, α i * (|X i l| + 4 * ρ i) ≤ 448)
    (hc : ∀ i, 8 * (1 + 17 * ε) ≤ (15 - 17 * ε) * c i) (hc4 : ∀ i, c i ≤ 4) (hε0 : 0 ≤ ε) (hε : ε ≤ 1 / 32)
    (hlive : LiveOperand Params.pi.δ ρ X) :
    (fs : ℚ) * s.m * s.k - rhoD * fs * s.m * s.k ≤ fs * (live FormChain.atomMix.value S α β X).card :=
  rhoD_rel_pi fs s hpS hqS herr hα hfloor hsat hc hc4 hε0 hε hlive

/-- **ρ_D for Pearl's BF16 chain at Π's parameters**: per row, the wgmma within `N_i/256` of `∓N_i` at a flip pair of
noise `N_i = c_i·α_iρ_i` with `648/943 ≤ c_i ≤ 4`. -/
theorem rhoD_bf16_pi (fs : ℕ) (s : Shape) {noise : (Fin Params.pi.r → ℚ) → (Fin Params.pi.r → ℚ) → ℚ}
    {S : Support Params.pi.r} {α β ρ c η : Fin s.m → ℚ} {X : Matrix (Fin s.m) (Fin s.k) ℚ}
    {p q : Fin s.m → (Fin Params.pi.r → Code) × (Fin Params.pi.r → Code)} (hpS : ∀ i, p i ∈ S) (hqS : ∀ i, q i ∈ S)
    (hp : ∀ i, |noise (fun t => β i * codeVal ((p i).1 t)) (fun t => codeVal ((p i).2 t)) - -(c i * (α i * ρ i))| ≤ η i)
    (hq : ∀ i, |noise (fun t => β i * codeVal ((q i).1 t)) (fun t => codeVal ((q i).2 t)) - c i * (α i * ρ i)| ≤ η i)
    (hη : ∀ i, η i ≤ c i * (α i * ρ i) / 256) (hα : ∀ i, 0 < α i) (hfloor : ∀ i, 1 ≤ α i * ρ i)
    (hsat : ∀ i l, α i * (|X i l| + 4 * ρ i) ≤ 448) (hc : ∀ i, 648 / 943 ≤ c i) (hc4 : ∀ i, c i ≤ 4)
    (hlive : LiveOperand Params.pi.δ ρ X) :
    (fs : ℚ) * s.m * s.k - rhoD * fs * s.m * s.k ≤ fs * (live (FormChain.bf16 Params.pi.r noise).value S α β X).card := by
  refine rhoD_rel_pi fs s (ε := 1 / 64) (c := c) hpS hqS (fun i l => ?_) hα hfloor hsat
    (fun i => by linarith [hc i]) hc4 (by norm_num) (by norm_num) hlive
  have hN : 0 ≤ c i * (α i * ρ i) := mul_nonneg (by linarith [hc i]) (by linarith [hfloor i])
  exact bf16_pairErr (hα i).le hN (hη i) (hp i) (hq i)

/-! ### Sharper: the loud entries at (a)'s threshold -/

/-- Markov on a row: `T²·#{|x_l| ≥ T} ≤ Σ x_l²`. -/
theorem markov_row (x : Fin k → ℚ) {T : ℚ} (hT : 0 ≤ T) :
    T ^ 2 * (Finset.univ.filter fun l => T ≤ |x l|).card ≤ ∑ l, x l ^ 2 := by
  calc T ^ 2 * (Finset.univ.filter fun l => T ≤ |x l|).card
        = ∑ _l ∈ Finset.univ.filter fun l => T ≤ |x l|, T ^ 2 := by rw [Finset.sum_const, nsmul_eq_mul, mul_comm]
    _ ≤ ∑ l ∈ Finset.univ.filter fun l => T ≤ |x l|, x l ^ 2 := Finset.sum_le_sum fun l hl => by
        have h1 := mul_self_le_mul_self hT (Finset.mem_filter.1 hl).2
        nlinarith [abs_mul_abs_self (x l)]
    _ ≤ ∑ l, x l ^ 2 := Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) fun _ _ _ => sq_nonneg _

/-- **Sharper budget, full-RMS statistic.** If (a) covers `|x| < 15·c·δρ_i` and `ρ_i` is at least the row's RMS, the
salt-dead elements number at most `m·k/(15cδ)²`: `ρ_D ≤ 1/(15cδ)²`. Under §5's every-8th statistic no deterministic
bound of this kind follows. -/
theorem formingBudget_markov {v : ValueFn r} {S : Support r} {δ c : ℚ} {α β ρ : Fin m → ℚ}
    {X : Matrix (Fin m) (Fin k) ℚ} (hδ : 0 < δ) (hc : 0 < c) (hρ : ∀ i, 0 < ρ i ∧ meanSq (X i) ≤ ρ i ^ 2)
    (hcov : ∀ i l, |X i l| < 15 * c * δ * ρ i → SaltLiveVal v S (α i) (X i l) (β i)) :
    (1 - 1 / (15 * c * δ) ^ 2) * ((m : ℚ) * k) ≤ (live v S α β X).card := by
  have hL : (15 * c * δ) ^ 2 * ((loud (fun i => 15 * c * δ * ρ i) X).card : ℚ) ≤ m * k := by
    rw [card_loud]; push_cast
    rw [Finset.mul_sum]
    calc ∑ i, (15 * c * δ) ^ 2 * ((Finset.univ.filter fun l => 15 * c * δ * ρ i ≤ |X i l|).card : ℚ)
          ≤ ∑ _i : Fin m, (k : ℚ) := Finset.sum_le_sum fun i _ => by
            obtain ⟨hρ0, hms⟩ := hρ i
            have hT : 0 ≤ 15 * c * δ * ρ i := by positivity
            have hmk := markov_row (X i) hT
            have hsum : ∑ l, X i l ^ 2 ≤ k * ρ i ^ 2 := by
              rcases Nat.eq_zero_or_pos k with hk | hk
              · subst hk; simp
              · have : ∑ l, X i l ^ 2 = k * meanSq (X i) := by unfold meanSq; field_simp
                rw [this]; exact mul_le_mul_of_nonneg_left hms (by positivity)
            have hρ2 : 0 < ρ i ^ 2 := by positivity
            have : ((15 * c * δ) ^ 2 * (Finset.univ.filter fun l => 15 * c * δ * ρ i ≤ |X i l|).card) * ρ i ^ 2
                ≤ k * ρ i ^ 2 := by nlinarith
            exact le_of_mul_le_mul_right this hρ2
      _ = m * k := by simp
  have h1 : ((m * k : ℕ) : ℚ) ≤ ((live v S α β X).card + (loud (fun i => 15 * c * δ * ρ i) X).card : ℕ) := by
    exact_mod_cast card_live_ge hcov
  push_cast at h1
  have hq : 0 < (15 * c * δ) ^ 2 := by positivity
  have h3 : ((loud (fun i => 15 * c * δ * ρ i) X).card : ℚ) ≤ 1 / (15 * c * δ) ^ 2 * (m * k) := by
    rw [div_mul_eq_mul_div, one_mul, le_div_iff₀ hq]; linarith
  linarith

/-! ## The shared debit (split §5): the forming credit over the chain's flags

`flag i j t` flags atom `t` (positions `32t, …, 32t + 31`) of word `(i, j)`, row `i` of `A′` against row `j` of `B̃`.
Element `(i, l)` of `A′` is read by atom `l / 32` of the `n` words `(i, j)` and by nothing else; it is *needed* when
some word leaves that atom unflagged. -/

/-- Per-(word, atom) flags of an `m × n` unit; atoms are indexed by `ℕ`, as by `wordFlags`. -/
abbrev Flags (m n : Nat) : Type := Fin m → Fin n → ℕ → Bool

variable {n : Nat}

/-- Element `(i, l)` of `A′` is needed: some word `(i, j)` leaves the atom reading it, `l / 32`, unflagged. -/
def Needed (flag : Flags m n) (i : Fin m) (l : Fin k) : Prop := ∃ j, flag i j ((l : ℕ) / 32) = false

instance instDecidableNeeded (flag : Flags m n) (i : Fin m) (l : Fin k) : Decidable (Needed flag i l) := by
  unfold Needed; infer_instance

/-- No atom flagged. -/
def noFlags : Flags m n := fun _ _ _ => false

/-- The elements of `A′` not needed. -/
def notNeeded (flag : Flags m n) : Finset (Fin m × Fin k) := Finset.univ.filter fun il => ¬Needed flag il.1 il.2

open Classical in
/-- **The forming credit under flags**: `fs` per element of `A′` that is salt-live and needed. -/
noncomputable def formCredit (fs : ℕ) (v : ValueFn r) (S : Support r) (α β : Fin m → ℚ) (X : Matrix (Fin m) (Fin k) ℚ)
    (flag : Flags m n) : ℚ :=
  fs * ((live v S α β X).filter fun il => Needed flag il.1 il.2).card

/-- **The forming debit**: `creditOf`'s forming term `fs·m·k` less the forming credit. -/
noncomputable def formDebit (fs : ℕ) (v : ValueFn r) (S : Support r) (α β : Fin m → ℚ) (X : Matrix (Fin m) (Fin k) ℚ)
    (flag : Flags m n) : ℚ :=
  fs * m * k - formCredit fs v S α β X flag

open Classical in
/-- The debit counts each element once, whether salt-dead or not needed. -/
theorem formDebit_eq_card (fs : ℕ) (v : ValueFn r) (S : Support r) (α β : Fin m → ℚ) (X : Matrix (Fin m) (Fin k) ℚ)
    (flag : Flags m n) :
    formDebit fs v S α β X flag =
      fs * (Finset.univ.filter fun il : Fin m × Fin k => ¬(il ∈ live v S α β X ∧ Needed flag il.1 il.2)).card := by
  have hf : ((live v S α β X).filter fun il => Needed flag il.1 il.2) =
      Finset.univ.filter fun il : Fin m × Fin k => il ∈ live v S α β X ∧ Needed flag il.1 il.2 := by
    ext il; simp
  have hc := Finset.card_filter_add_card_filter_not (s := (Finset.univ : Finset (Fin m × Fin k)))
    (fun il => il ∈ live v S α β X ∧ Needed flag il.1 il.2)
  rw [Finset.card_univ, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin] at hc
  have hq : (((Finset.univ.filter fun il : Fin m × Fin k => il ∈ live v S α β X ∧ Needed flag il.1 il.2).card +
      (Finset.univ.filter fun il : Fin m × Fin k => ¬(il ∈ live v S α β X ∧ Needed flag il.1 il.2)).card : ℕ) : ℚ) =
      ((m * k : ℕ) : ℚ) := by rw [hc]
  push_cast at hq
  rw [formDebit, formCredit, hf]
  linear_combination (-(fs : ℚ)) * hq

/-- The credit loses at most `fs` per element not needed. -/
theorem formCredit_ge (fs : ℕ) (v : ValueFn r) (S : Support r) (α β : Fin m → ℚ) (X : Matrix (Fin m) (Fin k) ℚ)
    (flag : Flags m n) :
    (fs : ℚ) * (live v S α β X).card - fs * (notNeeded flag : Finset (Fin m × Fin k)).card ≤
      formCredit fs v S α β X flag := by
  have hc := Finset.card_filter_add_card_filter_not (s := live v S α β X)
    (fun il => Needed flag il.1 il.2)
  have hsub : ((live v S α β X).filter fun il => ¬Needed flag il.1 il.2) ⊆ notNeeded flag := fun il hil =>
    Finset.mem_filter.2 ⟨Finset.mem_univ _, (Finset.mem_filter.1 hil).2⟩
  have h1 := Finset.card_le_card hsub
  have h2 : ((live v S α β X).card : ℚ) ≤
      ((live v S α β X).filter fun il => Needed flag il.1 il.2).card + (notNeeded flag : Finset (Fin m × Fin k)).card := by
    exact_mod_cast hc ▸ Nat.add_le_add_left h1 _
  have h3 := mul_le_mul_of_nonneg_left h2 (Nat.cast_nonneg (α := ℚ) fs)
  unfold formCredit
  linarith

/-- **The per-element salt-dead predicate** on a committed row, of `PearlCSem.saltDead`'s type: element `l` of row `x`
is salt-dead when its code's value is the same at every noise pair of `S`, under the chain `v` and the row's scales
`alpha k x`, `beta k x`. A salt that flips only a zero's sign counts as dead. -/
def saltDead (v : ValueFn r) (S : Support r) (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) :
    (k : ℕ) → (Fin k → ℚ) → Fin k → Prop :=
  fun k x l => ¬SaltLiveVal v S (alpha k x) (x l) (beta k x)

open Classical in
/-- **The debit in `unitDebit`'s form**: with the rows' own scales, `formDebit = fs·#{saltDead ∨ flagged in every
word}`. -/
theorem formDebit_eq_elems (fs : ℕ) (v : ValueFn r) (S : Support r) (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ)
    (X : Matrix (Fin m) (Fin k) ℚ) (flag : Flags m n) :
    formDebit fs v S (fun i => alpha k (X i)) (fun i => beta k (X i)) X flag =
      fs * (Finset.univ.filter fun e : Fin m × Fin k =>
        saltDead v S alpha beta k (X e.1) e.2 ∨ ∀ j : Fin n, flag e.1 j (e.2.val / 32) = true).card := by
  have h : (Finset.univ.filter fun il : Fin m × Fin k =>
      ¬(il ∈ live v S (fun i => alpha k (X i)) (fun i => beta k (X i)) X ∧ Needed flag il.1 il.2)) =
      Finset.univ.filter fun e : Fin m × Fin k =>
        saltDead v S alpha beta k (X e.1) e.2 ∨ ∀ j : Fin n, flag e.1 j (e.2.val / 32) = true := by
    refine Finset.filter_congr fun e _ => ?_
    simp only [live, saltDead, Needed, Finset.mem_filter, Finset.mem_univ, true_and, not_and_or, not_exists,
      Bool.not_eq_false]
  rw [formDebit_eq_card, h]

/-! ### Π's salt-dead predicate (`SaltDead.saltDead`) -/

theorem flat_eq_flatLine (s : Fin 32 → Bool) : (flat s : Fin 32 → Code) = SaltDead.flatLine s := rfl

theorem flatF_eq_flatLineF (s : Fin 32 → Bool) : (flatF s : Fin 32 → Code) = SaltDead.flatLineF s := rfl

/-- `SaltDead.support` has every flat pair (`E` at `±44`, `F` at `±11/4`), so `AtomErr.rhoD_atomMix_pi_flat` applies
at it. -/
theorem flatPairs_support : FlatPairs SaltDead.support := fun s s' => ⟨s, s', rfl⟩

/-- **The link:** `SaltDead.saltDead` is `saltDead` at Π, the atom-mix over `SaltDead.support` (not `SaltLiveVal`), so
`formDebit_eq_elems` reads it as `unitDebit` does. -/
theorem saltDead_pi (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) :
    SaltDead.saltDead alpha beta = saltDead FormChain.atomMix.value SaltDead.support alpha beta := by
  funext k x l
  simp only [SaltDead.saltDead, saltDead, SaltLiveVal, FormChain.elem, not_exists, not_and, not_not]

open Classical in
/-- **ρ_D for `SaltDead.saltDead`:** given the forming budget on `live` over `SaltDead.support` at the rows' own scales
(`AtomErr.rhoD_atomMix_pi_flat` with `flatPairs_support`, at `ρ = rhoD`), at most a `ρ` share of the elements are
salt-dead. -/
theorem saltDead_pi_card_le (fs : ℕ) (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) (X : Matrix (Fin m) (Fin k) ℚ) {ρ : ℚ}
    (h : (fs : ℚ) * m * k - ρ * fs * m * k ≤
      fs * (live FormChain.atomMix.value SaltDead.support (fun i => alpha k (X i)) (fun i => beta k (X i)) X).card) :
    (fs : ℚ) * (Finset.univ.filter fun e : Fin m × Fin k => SaltDead.saltDead alpha beta k (X e.1) e.2).card ≤
      ρ * fs * m * k := by
  have hc := Finset.card_filter_add_card_filter_not (s := (Finset.univ : Finset (Fin m × Fin k)))
    (fun e => SaltLiveVal FormChain.atomMix.value SaltDead.support (alpha k (X e.1)) (X e.1 e.2) (beta k (X e.1)))
  rw [Finset.card_univ, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin] at hc
  have hd : (Finset.univ.filter fun e : Fin m × Fin k => SaltDead.saltDead alpha beta k (X e.1) e.2) =
      Finset.univ.filter fun e : Fin m × Fin k =>
        ¬SaltLiveVal FormChain.atomMix.value SaltDead.support (alpha k (X e.1)) (X e.1 e.2) (beta k (X e.1)) := by
    refine Finset.filter_congr fun e _ => ?_
    rw [saltDead_pi]
    exact Iff.rfl
  have hq : ((live FormChain.atomMix.value SaltDead.support (fun i => alpha k (X i)) (fun i => beta k (X i)) X).card +
      (Finset.univ.filter fun e : Fin m × Fin k => SaltDead.saltDead alpha beta k (X e.1) e.2).card : ℚ) = m * k := by
    rw [hd]
    exact_mod_cast hc
  linear_combination (fs : ℚ) * hq + h

/-! ### (i) No flags: the ρ_D budget -/

theorem needed_noFlags (hn : 0 < n) (i : Fin m) (l : Fin k) : Needed (noFlags : Flags m n) i l := ⟨⟨0, hn⟩, rfl⟩

/-- With no flags (and at least one word per row) the credit is `fs·#live`. -/
theorem formCredit_noFlags (hn : 0 < n) (fs : ℕ) (v : ValueFn r) (S : Support r) (α β : Fin m → ℚ)
    (X : Matrix (Fin m) (Fin k) ℚ) : formCredit fs v S α β X (noFlags : Flags m n) = fs * (live v S α β X).card := by
  unfold formCredit
  rw [Finset.filter_true_of_mem fun il _ => needed_noFlags hn il.1 il.2]

/-- **(i) at Π's parameters**, any value function: with no flags, `formCredit ≥ (1 − ρ_D)·fs·m·k`, `ρ_D = 1/64`. -/
theorem formCredit_pi (fs : ℕ) (s : Shape) (hn : 0 < s.n) {v : ValueFn Params.pi.r} {S : Support Params.pi.r}
    {α β ρ : Fin s.m → ℚ} {X : Matrix (Fin s.m) (Fin s.k) ℚ}
    (hcov : ∀ i l, |X i l| < 8 * Params.pi.δ * ρ i → SaltLiveVal v S (α i) (X i l) (β i))
    (hlive : LiveOperand Params.pi.δ ρ X) :
    (1 - rhoD) * fs * s.m * s.k ≤ formCredit fs v S α β X (noFlags : Flags s.m s.n) := by
  rw [formCredit_noFlags hn]; linarith [(rhoD_pi fs s hcov hlive).1]

/-- **(i) for the exact chain**, under `rhoD_exact_pi`'s hypotheses. -/
theorem formCredit_exact_pi (fs : ℕ) (s : Shape) (hn : 0 < s.n) {S : Support Params.pi.r} {α β ρ : Fin s.m → ℚ}
    {X : Matrix (Fin s.m) (Fin s.k) ℚ} (hα : ∀ i, 0 < α i) (hfloor : ∀ i, 1 ≤ α i * ρ i)
    (hsat : ∀ i l, α i * (|X i l| + 4 * ρ i) ≤ 448)
    (hpair : ∀ i, ∃ (e f : Fin Params.pi.r → Code) (c : ℚ), (negLine e, f) ∈ S ∧ (e, f) ∈ S ∧ 8 / 15 ≤ c ∧ c ≤ 4 ∧
      β i * ∑ t, codeVal (e t) * codeVal (f t) = c * (α i * ρ i))
    (hlive : LiveOperand Params.pi.δ ρ X) :
    (1 - rhoD) * fs * s.m * s.k ≤
      formCredit fs (FormChain.exact Params.pi.r).value S α β X (noFlags : Flags s.m s.n) := by
  rw [formCredit_noFlags hn]; linarith [rhoD_exact_pi fs s hα hfloor hsat hpair hlive]

/-- **(i) for a chain of relative error `ε`**, under `rhoD_rel_pi`'s hypotheses. -/
theorem formCredit_rel_pi (fs : ℕ) (s : Shape) (hn : 0 < s.n) {v : ValueFn Params.pi.r} {S : Support Params.pi.r}
    {α β ρ c : Fin s.m → ℚ} {ε : ℚ} {X : Matrix (Fin s.m) (Fin s.k) ℚ}
    {p q : Fin s.m → (Fin Params.pi.r → Code) × (Fin Params.pi.r → Code)} (hpS : ∀ i, p i ∈ S) (hqS : ∀ i, q i ∈ S)
    (herr : ∀ i l, PairErr v (α i) (X i l) (β i) (c i * (α i * ρ i)) ε (p i) (q i)) (hα : ∀ i, 0 < α i)
    (hfloor : ∀ i, 1 ≤ α i * ρ i) (hsat : ∀ i l, α i * (|X i l| + 4 * ρ i) ≤ 448)
    (hc : ∀ i, 8 * (1 + 17 * ε) ≤ (15 - 17 * ε) * c i) (hc4 : ∀ i, c i ≤ 4) (hε0 : 0 ≤ ε) (hε : ε ≤ 1 / 32)
    (hlive : LiveOperand Params.pi.δ ρ X) :
    (1 - rhoD) * fs * s.m * s.k ≤ formCredit fs v S α β X (noFlags : Flags s.m s.n) := by
  rw [formCredit_noFlags hn]; linarith [rhoD_rel_pi fs s hpS hqS herr hα hfloor hsat hc hc4 hε0 hε hlive]

/-- **(i) for §5's atom-mix**, under `rhoD_atomMix_pi`'s hypotheses (`PairErr` named). -/
theorem formCredit_atomMix_pi (fs : ℕ) (s : Shape) (hn : 0 < s.n) {S : Support Params.pi.r} {α β ρ c : Fin s.m → ℚ}
    {ε : ℚ} {X : Matrix (Fin s.m) (Fin s.k) ℚ} {p q : Fin s.m → (Fin Params.pi.r → Code) × (Fin Params.pi.r → Code)}
    (hpS : ∀ i, p i ∈ S) (hqS : ∀ i, q i ∈ S)
    (herr : ∀ i l, PairErr FormChain.atomMix.value (α i) (X i l) (β i) (c i * (α i * ρ i)) ε (p i) (q i))
    (hα : ∀ i, 0 < α i) (hfloor : ∀ i, 1 ≤ α i * ρ i) (hsat : ∀ i l, α i * (|X i l| + 4 * ρ i) ≤ 448)
    (hc : ∀ i, 8 * (1 + 17 * ε) ≤ (15 - 17 * ε) * c i) (hc4 : ∀ i, c i ≤ 4) (hε0 : 0 ≤ ε) (hε : ε ≤ 1 / 32)
    (hlive : LiveOperand Params.pi.δ ρ X) :
    (1 - rhoD) * fs * s.m * s.k ≤ formCredit fs FormChain.atomMix.value S α β X (noFlags : Flags s.m s.n) :=
  formCredit_rel_pi fs s hn hpS hqS herr hα hfloor hsat hc hc4 hε0 hε hlive

/-- **(i) for Pearl's BF16 chain**, under `rhoD_bf16_pi`'s hypotheses. -/
theorem formCredit_bf16_pi (fs : ℕ) (s : Shape) (hn : 0 < s.n)
    {noise : (Fin Params.pi.r → ℚ) → (Fin Params.pi.r → ℚ) → ℚ} {S : Support Params.pi.r} {α β ρ c η : Fin s.m → ℚ}
    {X : Matrix (Fin s.m) (Fin s.k) ℚ} {p q : Fin s.m → (Fin Params.pi.r → Code) × (Fin Params.pi.r → Code)}
    (hpS : ∀ i, p i ∈ S) (hqS : ∀ i, q i ∈ S)
    (hp : ∀ i, |noise (fun t => β i * codeVal ((p i).1 t)) (fun t => codeVal ((p i).2 t)) - -(c i * (α i * ρ i))| ≤ η i)
    (hq : ∀ i, |noise (fun t => β i * codeVal ((q i).1 t)) (fun t => codeVal ((q i).2 t)) - c i * (α i * ρ i)| ≤ η i)
    (hη : ∀ i, η i ≤ c i * (α i * ρ i) / 256) (hα : ∀ i, 0 < α i) (hfloor : ∀ i, 1 ≤ α i * ρ i)
    (hsat : ∀ i l, α i * (|X i l| + 4 * ρ i) ≤ 448) (hc : ∀ i, 648 / 943 ≤ c i) (hc4 : ∀ i, c i ≤ 4)
    (hlive : LiveOperand Params.pi.δ ρ X) :
    (1 - rhoD) * fs * s.m * s.k ≤ formCredit fs (FormChain.bf16 Params.pi.r noise).value S α β X (noFlags : Flags s.m s.n) := by
  rw [formCredit_noFlags hn]; linarith [rhoD_bf16_pi fs s hpS hqS hp hq hη hα hfloor hsat hc hc4 hlive]

/-! ### (ii) The debit under the chain's flags -/

/-- The atoms `t < k/32` of row `i` flagged in every word `(i, j)`. -/
def allFlagged (k : ℕ) (flag : Flags m n) (i : Fin m) : Finset ℕ :=
  (Finset.range (k / 32)).filter fun t => ∀ j, flag i j t = true

/-- The flagged atoms `t < k/32` of word `(i, j)`. -/
def flaggedIn (k : ℕ) (flag : Flags m n) (i : Fin m) (j : Fin n) : Finset ℕ :=
  (Finset.range (k / 32)).filter fun t => flag i j t = true

/-- The chain's flagged atoms over all `m·n` words. -/
def flagCount (k : ℕ) (flag : Flags m n) : ℕ := ∑ i, ∑ j, (flaggedIn k flag i j).card

theorem card_range_div32 (Q : ℕ → Prop) [DecidablePred Q] (K : ℕ) :
    ((Finset.range (32 * K)).filter fun l => Q (l / 32)).card = 32 * ((Finset.range K).filter Q).card := by
  simp only [Finset.card_filter]
  induction K with
  | zero => simp
  | succ K ih =>
    have h32 : (∑ x ∈ Finset.range 32, if Q ((32 * K + x) / 32) then 1 else 0) = 32 * (if Q K then 1 else 0) := by
      rw [Finset.sum_congr rfl (g := fun _ => if Q K then 1 else 0) fun x hx => by
        rw [show (32 * K + x) / 32 = K by rw [Finset.mem_range] at hx; omega]]
      simp
    rw [show 32 * (K + 1) = 32 * K + 32 by ring, Finset.sum_range_add, ih, h32,
      Finset.sum_range_succ (fun i => if Q i then 1 else 0) K, mul_add]

theorem card_fin_div32 (Q : ℕ → Prop) [DecidablePred Q] (hk : 32 ∣ k) :
    (Finset.univ.filter fun l : Fin k => Q ((l : ℕ) / 32)).card = 32 * ((Finset.range (k / 32)).filter Q).card := by
  obtain ⟨K, rfl⟩ := hk
  rw [Nat.mul_div_cancel_left K (by norm_num), ← card_range_div32, Finset.card_filter, Finset.card_filter]
  exact Fin.sum_univ_eq_sum_range (fun l => if Q (l / 32) then 1 else 0) _

/-- `#{not needed} = 32·Σ_i #{t : ∀ j, flag i j t}`. -/
theorem card_notNeeded (flag : Flags m n) (hk : 32 ∣ k) :
    (notNeeded flag : Finset (Fin m × Fin k)).card = 32 * ∑ i, (allFlagged k flag i).card := by
  rw [notNeeded, Finset.card_filter, Fintype.sum_prod_type, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [← Finset.card_filter, allFlagged, ← card_fin_div32 _ hk]
  congr 1
  ext l; simp [Needed]

theorem card_allFlagged_le (flag : Flags m n) (i : Fin m) :
    n * (allFlagged k flag i).card ≤ ∑ j, (flaggedIn k flag i j).card := by
  calc n * (allFlagged k flag i).card = ∑ _j : Fin n, (allFlagged k flag i).card := by simp
    _ ≤ _ := Finset.sum_le_sum fun j _ => Finset.card_le_card fun t ht => by
        simp only [allFlagged, flaggedIn, Finset.mem_filter] at ht ⊢; exact ⟨ht.1, ht.2 j⟩

/-- `n·#{not needed} ≤ 32·(flagged atoms)`. -/
theorem card_notNeeded_le (flag : Flags m n) (hk : 32 ∣ k) :
    n * (notNeeded flag : Finset (Fin m × Fin k)).card ≤ 32 * flagCount k flag := by
  rw [card_notNeeded flag hk, mul_left_comm]
  refine Nat.mul_le_mul_left 32 ?_
  rw [Finset.mul_sum]
  exact Finset.sum_le_sum fun i _ => card_allFlagged_le flag i

/-- **(ii) The debit under the flags**: whatever salt-dead budget `D` the chain's (c) gives,
`formDebit ≤ D + (32·fs/n)·(flagged atoms)`. -/
theorem formDebit_le (fs : ℕ) {v : ValueFn r} {S : Support r} {α β : Fin m → ℚ} {X : Matrix (Fin m) (Fin k) ℚ}
    (flag : Flags m n) (hk : 32 ∣ k) (hn : 0 < n) {D : ℚ} (hD : (fs : ℚ) * m * k - D ≤ fs * (live v S α β X).card) :
    formDebit fs v S α β X flag ≤ D + 32 * fs * flagCount k flag / n := by
  have h1 := formCredit_ge fs v S α β X flag
  have hn' : (0 : ℚ) < n := by exact_mod_cast hn
  have h2 : (n : ℚ) * (notNeeded flag : Finset (Fin m × Fin k)).card ≤ 32 * flagCount k flag := by
    exact_mod_cast card_notNeeded_le flag hk
  have h3 : (fs : ℚ) * (notNeeded flag : Finset (Fin m × Fin k)).card ≤ 32 * fs * flagCount k flag / n := by
    rw [le_div_iff₀ hn']
    nlinarith [Nat.cast_nonneg (α := ℚ) fs]
  unfold formDebit
  linarith

/-- **(ii) at Π's parameters**: `formDebit ≤ ρ_D·fs·m·k + (32·fs/n)·(flagged atoms)`, `ρ_D = 1/64`. -/
theorem formDebit_pi (fs : ℕ) (s : Shape) (hk : 32 ∣ s.k) (hn : 0 < s.n) {v : ValueFn Params.pi.r}
    {S : Support Params.pi.r} {α β ρ : Fin s.m → ℚ} {X : Matrix (Fin s.m) (Fin s.k) ℚ} (flag : Flags s.m s.n)
    (hcov : ∀ i l, |X i l| < 8 * Params.pi.δ * ρ i → SaltLiveVal v S (α i) (X i l) (β i))
    (hlive : LiveOperand Params.pi.δ ρ X) :
    formDebit fs v S α β X flag ≤ rhoD * fs * s.m * s.k + 32 * fs * flagCount s.k flag / s.n :=
  formDebit_le fs flag hk hn (by linarith [(rhoD_pi fs s hcov hlive).1])

/-! ### (iii) Soundness: the hypotheses the flags must satisfy

The debit is sound when every atom a prover can skip is flagged: then an element whose atom is skippable in all `n`
words is not needed, so debited. `skippable i j t` is abstract here: word `(i, j)`'s output is unchanged with atom `t`
skipped in place, together with some set of its group's atoms. The literal reading (the words do not depend on an
element not needed once every flagged atom is skipped) needs, in addition, that the flagged atoms are jointly
skippable (`H` below). -/

section Soundness

variable {skippable : Fin m → Fin n → ℕ → Prop} {flag : Flags m n}

/-- **The flags are complete** for a skip relation: every skippable atom of every word is flagged. -/
def FlagsComplete (skippable : Fin m → Fin n → ℕ → Prop) (flag : Flags m n) : Prop :=
  ∀ i j t, skippable i j t → flag i j t = true

/-- Element `(i, l)` can be done without: the atom reading it is skippable in every word `(i, j)`. -/
def Avoidable (skippable : Fin m → Fin n → ℕ → Prop) (i : Fin m) (l : Fin k) : Prop :=
  ∀ j, skippable i j ((l : ℕ) / 32)

theorem not_needed_of_avoidable (hc : FlagsComplete skippable flag) {i : Fin m} {l : Fin k}
    (h : Avoidable skippable i l) : ¬Needed flag i l := by
  rintro ⟨j, hj⟩
  rw [hc i j _ (h j)] at hj
  exact Bool.noConfusion hj

open Classical in
/-- **(iii) The debit covers every avoidable element** when the flags are complete. -/
theorem formDebit_ge_avoidable (hc : FlagsComplete skippable flag) (fs : ℕ) (v : ValueFn r) (S : Support r)
    (α β : Fin m → ℚ) (X : Matrix (Fin m) (Fin k) ℚ) :
    (fs : ℚ) * (Finset.univ.filter fun il : Fin m × Fin k => Avoidable skippable il.1 il.2).card ≤
      formDebit fs v S α β X flag := by
  rw [formDebit_eq_card]
  refine mul_le_mul_of_nonneg_left ?_ (Nat.cast_nonneg fs)
  have hsub : (Finset.univ.filter fun il : Fin m × Fin k => Avoidable skippable il.1 il.2) ⊆
      Finset.univ.filter fun il : Fin m × Fin k => ¬(il ∈ live v S α β X ∧ Needed flag il.1 il.2) := by
    intro il hil
    rw [Finset.mem_filter] at hil ⊢
    exact ⟨hil.1, fun h => not_needed_of_avoidable hc hil.2 h.2⟩
  exact_mod_cast Finset.card_le_card hsub

/-- The words with every flagged atom skipped read only unflagged atoms: word `(i, j)` agrees on operands that agree
on every element an unflagged atom of the word reads. -/
def ReadsUnflagged {C W : Type*} (skipped : Matrix (Fin m) (Fin k) C → Fin m → Fin n → W) (flag : Flags m n) : Prop :=
  ∀ A A' i j, (∀ l : Fin k, flag i j ((l : ℕ) / 32) = false → A' i l = A i l) → skipped A' i j = skipped A i j

/-- Once every flagged atom is skipped, the words do not depend on the elements not needed. -/
theorem skipped_eq_of_needed {C W : Type*} {skipped : Matrix (Fin m) (Fin k) C → Fin m → Fin n → W}
    (h : ReadsUnflagged skipped flag) {A A' : Matrix (Fin m) (Fin k) C}
    (hA : ∀ i l, Needed flag i l → A' i l = A i l) : skipped A' = skipped A := by
  funext i j; exact h A A' i j fun l hl => hA i l ⟨j, hl⟩

/-- **(iii), literally**: if skipping every flagged atom of every word leaves every word unchanged (`H`), the words are
those of any operand that agrees on the needed elements, the rest skipped. -/
theorem words_eq_of_needed {C W : Type*} {words : Matrix (Fin m) (Fin k) C → Fin m → Fin n → W}
    {skipped : Matrix (Fin m) (Fin k) C → Fin m → Fin n → W} (h : ReadsUnflagged skipped flag)
    {A A' : Matrix (Fin m) (Fin k) C} (H : skipped A = words A) (hA : ∀ i l, Needed flag i l → A' i l = A i l) :
    words A = skipped A' := by
  rw [← H, skipped_eq_of_needed h hA]

end Soundness

end Pouw.PearlC.Forming
