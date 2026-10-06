import Proofs.Core.TC.Proofs.Norm

/-!
# The Ampere adder's arithmetic (`W = 25`, `rho = 1`, `A = 26`, `B = 30`, `K = 35`, floor `-132`)

`Verity.SecurityProofs.TC.Proofs.Fold` and `Norm` for the Ampere/Ada pipeline: the term correspondence and
`group_sum`'s folds with the rescale `1`, the 25-bit window of a 30-bit total, and the final shift against `K = 35`.
-/

namespace Verity.TC.Proofs.Ampere

open Verity.TC Verity.TC.Spec Verity.TC.Relation

/-- One semantic term against one relation row, for the Ampere adder (`rho = 1`, `A = 26`). -/
structure TermCorr (Z : Int) (t : Term) (x : TermIn) : Prop where
  scaled : ((t.magnitude * 2 : Nat) : Int) = x.scaled
  mag_lt : t.magnitude * 2 < 2 ^ 26
  exp : t.magnitude ≠ 0 → t.exponent = x.e
  sign : t.magnitude ≠ 0 → (t.negative ↔ x.neg = 1)
  zero : t.magnitude = 0 → x.e = Z
  neg01 : x.neg = 0 ∨ x.neg = 1

def Corr (Z : Int) : List Term → List TermIn → Prop
  | [], [] => True
  | t :: ts, x :: xs => TermCorr Z t x ∧ Corr Z ts xs
  | _, _ => False

theorem maxExp_eq (Z : Int) : ∀ (ts : List Term) (xs : List TermIn) (m : Int), Corr Z ts xs → Z ≤ m →
    ts.foldl maxStep m = maxE m xs
  | [], [], _, _, _ => rfl
  | t :: ts, x :: xs, m, ⟨h, hs⟩, hm => by
    simp only [List.foldl_cons, maxE] at *
    have hstep : maxStep m t = max m x.e := by
      unfold maxStep
      by_cases h0 : t.magnitude = 0
      · have := h.zero h0; simp only [h0, ne_eq, not_true_eq_false, false_and, ite_false]; omega
      · have := h.exp h0; simp only [ne_eq, h0, not_false_eq_true, true_and]; split <;> omega
    rw [hstep]
    exact maxExp_eq Z ts xs _ hs (by omega)
  | [], _ :: _, _, h, _ => h.elim
  | _ :: _, [], _, h, _ => h.elim

/-- A row's truncated, signed contribution to the group total, aligned to `E` (`A = 26`). -/
def contrib (E : Int) (x : TermIn) : Int := (1 - 2 * x.neg) * (x.scaled / 2 ^ (min (E - x.e).toNat 26))

/-- `group_sum`'s total step, with the Ampere rescale `rescale = 1`. -/
def totStep (mx : Int) (total : Int) (t : Term) : Int :=
  if t.magnitude = 0 then total
  else
    let aligned : Int := ((t.magnitude <<< 1 >>> (mx - t.exponent).toNat : Nat) : Int)
    if t.negative then total - aligned else total + aligned

theorem div_pow_min (n k : Nat) (hn : n < 2 ^ 26) : n / 2 ^ k = n / 2 ^ (min k 26) := by
  by_cases hk : k ≤ 26
  · rw [Nat.min_eq_left hk]
  · rw [Nat.min_eq_right (by omega)]
    have h1 : n < 2 ^ k := Nat.lt_of_lt_of_le hn (Nat.pow_le_pow_right (by decide) (by omega))
    rw [Nat.div_eq_of_lt h1, Nat.div_eq_of_lt hn]

theorem total_eq (Z E : Int) : ∀ (ts : List Term) (xs : List TermIn) (tot : Int), Corr Z ts xs →
    (∀ x ∈ xs, x.e ≤ E) → ts.foldl (totStep E) tot = tot + (xs.map (contrib E)).sum
  | [], [], _, _, _ => by simp
  | t :: ts, x :: xs, tot, ⟨h, hs⟩, hE => by
    simp only [List.foldl_cons, List.map_cons, List.sum_cons]
    have hstep : totStep E tot t = tot + contrib E x := by
      unfold totStep contrib
      rw [← h.scaled, Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow, natCast_div_pow]
      by_cases h0 : t.magnitude = 0
      · simp [h0]
      · have he := h.exp h0
        have hsg := h.sign h0
        simp only [h0, ite_false]
        rw [← natCast_div_pow, ← natCast_div_pow, ← div_pow_min _ _ h.mag_lt, he]
        simp only [Nat.pow_one] at *
        rcases h.neg01 with hn | hn
        · have : ¬ t.negative := by rw [hsg]; omega
          simp [this, hn]
        · have : t.negative := by rw [hsg]; omega
          simp [this, hn]; omega
    rw [hstep, total_eq Z E ts xs _ hs (fun y hy => hE y (List.mem_cons_of_mem _ hy))]
    omega
  | [], _ :: _, _, h, _ => h.elim
  | _ :: _, [], _, h, _ => h.elim

/-- The 25-bit window: `|T| * 2^(30 - L) / 2^5` is `group_sum`'s shift of `|T|` to 25 bits, with its top bit set. -/
theorem window (N : Nat) (hN0 : N ≠ 0) (hN : N < 2 ^ 30) :
    N * 2 ^ (30 - bitLength N) / 2 ^ 5 =
        (if bitLength N > 25 then N >>> (bitLength N - 25) else N <<< (25 - bitLength N)) ∧
      2 ^ 24 ≤ N * 2 ^ (30 - bitLength N) / 2 ^ 5 ∧ N * 2 ^ (30 - bitLength N) / 2 ^ 5 < 2 ^ 25 := by
  obtain ⟨hlo, hhi, h1⟩ := bitLength_spec N hN0
  have hle := bitLength_le N 30 hN
  generalize bitLength N = L at *
  have hlo' : 2 ^ 29 ≤ N * 2 ^ (30 - L) := by
    have : 2 ^ 29 = 2 ^ (L - 1) * 2 ^ (30 - L) := by rw [← Nat.pow_add]; congr 1; omega
    rw [this]; exact Nat.mul_le_mul_right _ hlo
  have hhi' : N * 2 ^ (30 - L) < 2 ^ 30 := by
    have : 2 ^ 30 = 2 ^ L * 2 ^ (30 - L) := by rw [← Nat.pow_add]; congr 1; omega
    rw [this]; exact Nat.mul_lt_mul_of_pos_right hhi (Nat.two_pow_pos _)
  refine ⟨?_, ?_, ?_⟩
  · split
    · rw [Nat.shiftRight_eq_div_pow]
      have : 2 ^ 5 = 2 ^ (30 - L) * 2 ^ (L - 25) := by rw [← Nat.pow_add]; congr 1; omega
      rw [this, Nat.mul_comm N, Nat.mul_div_mul_left _ _ (Nat.two_pow_pos _)]
    · rw [Nat.shiftLeft_eq]
      have : 2 ^ (30 - L) = 2 ^ (25 - L) * 2 ^ 5 := by rw [← Nat.pow_add]; congr 1; omega
      rw [this, ← Nat.mul_assoc, Nat.mul_div_cancel _ (Nat.two_pow_pos _)]
  · exact (Nat.le_div_iff_mul_le (Nat.two_pow_pos _)).mpr (by omega)
  · exact (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).mpr (by omega)

/-- The subnormal shift and the rescale: `(m >> k) >> 1`, or `m >> 1`, is `m / 2^min(e_fin - e_out + 1, 35)`. -/
theorem final_shift (m : Nat) (hm : m < 2 ^ 25) (e_out : Int) :
    (if e_out < -126 then m >>> (-126 - e_out).toNat else m) >>> 1 =
      m / 2 ^ (min (max e_out (-126) - e_out + 1) 35).toNat := by
  rw [Nat.shiftRight_eq_div_pow]
  split
  · rw [Nat.shiftRight_eq_div_pow, Nat.div_div_eq_div_mul, ← Nat.pow_add]
    by_cases hk : (-126 - e_out).toNat + 1 ≤ 35
    · congr 2; omega
    · have h1 : m < 2 ^ ((-126 - e_out).toNat + 1) :=
        Nat.lt_of_lt_of_le hm (Nat.pow_le_pow_right (by decide) (by omega))
      have h2 : m < 2 ^ (min (max e_out (-126) - e_out + 1) 35).toNat :=
        Nat.lt_of_lt_of_le hm (Nat.pow_le_pow_right (by decide) (by omega))
      rw [Nat.div_eq_of_lt h1, Nat.div_eq_of_lt h2]
  · congr 2; omega

end Verity.TC.Proofs.Ampere
