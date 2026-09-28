import Proofs.Core.TC.Proofs.Prod

/-!
# `group_sum`'s two folds, against the relation's terms

`Corr Z ts xs` pairs the semantics' finite terms of a group with the relation's `_terms` rows, one for one: equal scaled
magnitudes, and equal exponents and signs where the magnitude is nonzero; a zero term sits at the floor. Under it,
`group_sum`'s maximum exponent is the maximum of the rows' exponents and the floor (`maxExp_eq`), and its total is the
sum of the rows' truncated, signed terms (`total_eq`).
-/

namespace Verity.TC.Proofs

open Verity.TC Verity.TC.Spec Verity.TC.Relation

/-- One semantic term against one relation row, for the Hopper adder (`rho = 2`, `A = 27`). -/
structure TermCorr (Z : Int) (t : Term) (x : TermIn) : Prop where
  scaled : ((t.magnitude * 4 : Nat) : Int) = x.scaled
  mag_lt : t.magnitude * 4 < 2 ^ 27
  exp : t.magnitude ≠ 0 → t.exponent = x.e
  sign : t.magnitude ≠ 0 → (t.negative ↔ x.neg = 1)
  zero : t.magnitude = 0 → x.e = Z
  neg01 : x.neg = 0 ∨ x.neg = 1

def Corr (Z : Int) : List Term → List TermIn → Prop
  | [], [] => True
  | t :: ts, x :: xs => TermCorr Z t x ∧ Corr Z ts xs
  | _, _ => False

/-- `group_sum`'s maximum-exponent step. -/
def maxStep (mx : Int) (t : Term) : Int := if t.magnitude ≠ 0 ∧ t.exponent > mx then t.exponent else mx

/-- The rows' maximum exponent, from `m`. -/
def maxE (m : Int) (xs : List TermIn) : Int := xs.foldl (fun m x => max m x.e) m

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

theorem maxE_cons (m : Int) (x : TermIn) (xs : List TermIn) : maxE m (x :: xs) = maxE (max m x.e) xs := rfl

theorem le_maxE : ∀ (xs : List TermIn) (m : Int), m ≤ maxE m xs ∧ ∀ x ∈ xs, x.e ≤ maxE m xs
  | [], m => ⟨Int.le_refl m, by simp⟩
  | x :: xs, m => by
    rw [maxE_cons]
    have ih := le_maxE xs (max m x.e)
    refine ⟨by omega, ?_⟩
    intro y hy
    rcases List.mem_cons.mp hy with rfl | hy
    · omega
    · exact ih.2 y hy

theorem maxE_attained : ∀ (xs : List TermIn) (m : Int), maxE m xs = m ∨ ∃ x ∈ xs, maxE m xs = x.e
  | [], m => Or.inl rfl
  | x :: xs, m => by
    rw [maxE_cons]
    rcases maxE_attained xs (max m x.e) with h | ⟨y, hy, h⟩
    · rw [h]
      by_cases hmx : m ≤ x.e
      · exact Or.inr ⟨x, List.mem_cons_self .., by omega⟩
      · exact Or.inl (by omega)
    · exact Or.inr ⟨y, List.mem_cons_of_mem _ hy, h⟩

/-- A row's truncated, signed contribution to the group total, aligned to `E`. -/
def contrib (E : Int) (x : TermIn) : Int := (1 - 2 * x.neg) * (x.scaled / 2 ^ (min (E - x.e).toNat 27))

/-- `group_sum`'s total step, with the Hopper rescale `rescale = 2`. -/
def totStep (mx : Int) (total : Int) (t : Term) : Int :=
  if t.magnitude = 0 then total
  else
    let aligned : Int := ((t.magnitude <<< 2 >>> (mx - t.exponent).toNat : Nat) : Int)
    if t.negative then total - aligned else total + aligned

theorem natCast_div_pow (n k : Nat) : ((n / 2 ^ k : Nat) : Int) = (n : Int) / 2 ^ k := by
  rw [Int.natCast_ediv]; simp

theorem div_pow_min (n k : Nat) (hn : n < 2 ^ 27) : n / 2 ^ k = n / 2 ^ (min k 27) := by
  by_cases hk : k ≤ 27
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
        simp only [Nat.pow_two] at *
        rcases h.neg01 with hn | hn
        · have : ¬ t.negative := by rw [hsg]; omega
          simp [this, hn]
        · have : t.negative := by rw [hsg]; omega
          simp [this, hn]; omega
    rw [hstep, total_eq Z E ts xs _ hs (fun y hy => hE y (List.mem_cons_of_mem _ hy))]
    omega
  | [], _ :: _, _, h, _ => h.elim
  | _ :: _, [], _, h, _ => h.elim

end Verity.TC.Proofs
