import Mathlib.Algebra.Order.Chebyshev
import Mathlib.Combinatorics.Enumerative.DoubleCounting
import Mathlib.Data.Fintype.Prod
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Order.ConditionallyCompleteLattice.Finset

/-!
# The rewinding lemma: openings off the plurality table cost collisions

C-Flock's compiled theorem (its soundness package's `README.md` §1.2) covers every prover,
including one that never "knows" a table. Its committed table is the **plurality** table: at
each position, the column the prover opens in the most runs (`IsPlurality`). A run is `bad` when
it opens some position to a column off the plurality; a pair of runs `conflict`s when both open
one position, to different columns. By `Merkle.opening_binding`, each conflicting pair (both
paths valid against one cap) is a collision of the hash, found by running the prover twice with
independent coins.

`bad_sq_le`: if each run opens at most `Q` of the `N` positions, then

  `#bad ² ≤ N · Q · #conflict`   (counts over runs `Ω` and pairs `Ω × Ω`),

so `Pr[bad] ≤ √(N · Q · Pr[conflict])` (`prBad_le`). With `Pr[conflict] ≤ Adv_CR(B)` for the
two-run collision finder `B`, this is the per-level term `√(N_ℓ Q_ℓ · Adv_CR)` of that package's
`DESIGN.md` §1.2.

Proof: `bad` is covered by the per-position sets of off-plurality runs, of sizes `b_p`. A run
that opens `p` to `v` conflicts at `p` with every run opening `p` to anything but `v`, and there
are at least `b_p` of those because no value beats the plurality, so `#conflictAt p ≥ q_p b_p ≥
b_p²`. Each conflicting pair is counted at no more than `Q` positions, and Cauchy–Schwarz gives
`(Σ b_p)² ≤ N Σ b_p²`.
-/

open Finset

namespace Verity.Rewinding

variable {Ω P V : Type} [Fintype Ω] [DecidableEq Ω] [DecidableEq V]

-- `op ω p = some v`: the run with coins `ω` opens position `p` to `v` (with a path that
-- verifies); `none`: it does not open `p`.
variable (op : Ω → P → Option V)

/-- Runs that open `p` to `v`. -/
def openedTo (p : P) (v : V) : Finset Ω := univ.filter fun ω => op ω p = some v

/-- Runs that open `p`. -/
def opened (p : P) : Finset Ω := univ.filter fun ω => (op ω p).isSome

/-- `vstar` is a plurality table: at each position no value is opened in more runs. -/
def IsPlurality (vstar : P → V) : Prop := ∀ p v, #(openedTo op p v) ≤ #(openedTo op p (vstar p))

/-- The run opens `p` to a value other than `vstar p`. -/
abbrev Off (vstar : P → V) (ω : Ω) (p : P) : Prop := (op ω p).isSome ∧ op ω p ≠ some (vstar p)

/-- The two runs open `p` to different values. -/
abbrev ConflictAt (x : Ω × Ω) (p : P) : Prop :=
  (op x.1 p).isSome ∧ (op x.2 p).isSome ∧ op x.1 p ≠ op x.2 p

omit [DecidableEq Ω] in
/-- Every assignment of runs to openings has a plurality table. -/
theorem exists_plurality [Nonempty V] : ∃ vstar : P → V, IsPlurality op vstar := by
  have h : ∀ p, ∃ v : V, ∀ w, #(openedTo op p w) ≤ #(openedTo op p v) := by
    intro p
    let f : V → ℕ := fun w => #(openedTo op p w)
    have hbdd : BddAbove (Set.range f) :=
      ⟨Fintype.card Ω, by rintro _ ⟨w, rfl⟩; exact card_le_univ _⟩
    obtain ⟨v, hv⟩ := Nat.sSup_mem (Set.range_nonempty f) hbdd
    exact ⟨v, fun w => by
      have := le_csSup hbdd ⟨w, rfl⟩
      simp only [f] at hv this ⊢
      omega⟩
  exact ⟨fun p => (h p).choose, fun p => (h p).choose_spec⟩

section counting

variable {op}

omit [DecidableEq Ω] in
theorem openedTo_subset (p : P) (v : V) : openedTo op p v ⊆ opened op p := by
  intro ω
  simp only [openedTo, opened, mem_filter, mem_univ, true_and]
  intro h
  simp [h]

/-- The runs off the plurality at `p` are the runs that open `p`, minus those that open it to the
plurality value. -/
theorem off_eq (vstar : P → V) (p : P) :
    univ.filter (fun ω => Off op vstar ω p) = opened op p \ openedTo op p (vstar p) := by
  ext ω
  simp [Off, opened, openedTo]

/-- A run opening `p` to `v` conflicts at `p` with at least `#off_p` runs (plurality). -/
theorem off_le_row (vstar : P → V) (hv : IsPlurality op vstar) (p : P) (ω₁ : Ω) (v : V)
    (h₁ : op ω₁ p = some v) :
    #(univ.filter fun ω => Off op vstar ω p) ≤ #(univ.filter fun ω₂ => ConflictAt op (ω₁, ω₂) p) := by
  have hrow : (univ.filter fun ω₂ => ConflictAt op (ω₁, ω₂) p) = opened op p \ openedTo op p v := by
    ext ω₂
    simp only [ConflictAt, h₁, opened, openedTo, mem_filter, mem_univ, true_and, mem_sdiff,
      Option.isSome_some]
    constructor <;> rintro ⟨hs, hne⟩ <;> exact ⟨hs, fun h => hne h.symm⟩
  rw [off_eq, hrow, card_sdiff_of_subset (openedTo_subset _ _),
    card_sdiff_of_subset (openedTo_subset _ _)]
  have := hv p v
  omega

/-- `b_p² ≤ #conflictAt p`. -/
theorem off_sq_le (vstar : P → V) (hv : IsPlurality op vstar) (p : P) :
    #(univ.filter fun ω => Off op vstar ω p) ^ 2 ≤ #(univ.filter fun x => ConflictAt op x p) := by
  set b := #(univ.filter fun ω => Off op vstar ω p)
  -- count conflicting pairs by their first run
  have hsplit : #(univ.filter fun x : Ω × Ω => ConflictAt op x p) =
      ∑ ω₁, #(univ.filter fun ω₂ => ConflictAt op (ω₁, ω₂) p) := by
    rw [card_filter, ← univ_product_univ, sum_product]
    exact sum_congr rfl fun ω₁ _ => (card_filter _ _).symm
  have hrow : ∀ ω₁ ∈ opened op p, b ≤ #(univ.filter fun ω₂ => ConflictAt op (ω₁, ω₂) p) := by
    intro ω₁ hω₁
    simp only [opened, mem_filter, mem_univ, true_and, Option.isSome_iff_exists] at hω₁
    obtain ⟨v, hv₁⟩ := hω₁
    exact off_le_row vstar hv p ω₁ v hv₁
  have hbq : b ≤ #(opened op p) := by
    simp only [b]
    rw [off_eq]
    exact card_le_card sdiff_subset
  calc b ^ 2 = b * b := sq b
    _ ≤ #(opened op p) * b := Nat.mul_le_mul_right b hbq
    _ = ∑ _ω₁ ∈ opened op p, b := by rw [sum_const, smul_eq_mul]
    _ ≤ ∑ ω₁ ∈ opened op p, #(univ.filter fun ω₂ => ConflictAt op (ω₁, ω₂) p) := sum_le_sum hrow
    _ ≤ ∑ ω₁, #(univ.filter fun ω₂ => ConflictAt op (ω₁, ω₂) p) :=
        sum_le_sum_of_subset (subset_univ _)
    _ = _ := hsplit.symm

end counting

variable [Fintype P]

/-- Runs with an opening off the plurality table. -/
def bad (vstar : P → V) : Finset Ω := univ.filter fun ω => ∃ p, Off op vstar ω p

/-- Pairs of runs that open some position to different values. -/
def conflict : Finset (Ω × Ω) := univ.filter fun x => ∃ p, ConflictAt op x p

section rewinding

variable {op}

/-- Each conflicting pair is counted at no more than `Q` positions. -/
theorem sum_conflictAt_le (Q : ℕ) (hQ : ∀ ω, #(univ.filter fun p => (op ω p).isSome) ≤ Q) :
    ∑ p, #(univ.filter fun x => ConflictAt op x p) ≤ Q * #(conflict op) := by
  classical
  have hdc := sum_card_bipartiteAbove_eq_sum_card_bipartiteBelow
    (s := (univ : Finset P)) (t := (univ : Finset (Ω × Ω))) (r := fun p x => ConflictAt op x p)
  simp only [bipartiteAbove, bipartiteBelow] at hdc
  rw [hdc]
  calc ∑ x, #(univ.filter fun p => ConflictAt op x p)
      ≤ ∑ x, if x ∈ conflict op then Q else 0 := by
        refine sum_le_sum fun x _ => ?_
        split_ifs with hx
        · refine (card_le_card fun p => ?_).trans (hQ x.1)
          simp only [mem_filter, mem_univ, true_and]
          exact fun h => h.1
        · have : (univ.filter fun p => ConflictAt op x p) = ∅ := by
            ext p
            simp only [conflict, mem_filter, mem_univ, true_and, not_exists] at hx
            simp [hx p]
          simp [this]
    _ = Q * #(conflict op) := by
        rw [← sum_filter, sum_const, smul_eq_mul, mul_comm, filter_mem_eq_inter, univ_inter]

/-- **Rewinding lemma (counts).** `#bad ² ≤ N · Q · #conflict`. -/
theorem bad_sq_le (vstar : P → V) (hv : IsPlurality op vstar) (Q : ℕ)
    (hQ : ∀ ω, #(univ.filter fun p => (op ω p).isSome) ≤ Q) :
    #(bad op vstar) ^ 2 ≤ Fintype.card P * Q * #(conflict op) := by
  classical
  have hcover : #(bad op vstar) ≤ ∑ p, #(univ.filter fun ω => Off op vstar ω p) := by
    have : bad op vstar = univ.biUnion fun p => univ.filter fun ω => Off op vstar ω p := by
      ext ω
      simp [bad]
    rw [this]
    exact card_biUnion_le
  calc #(bad op vstar) ^ 2
      ≤ (∑ p, #(univ.filter fun ω => Off op vstar ω p)) ^ 2 := Nat.pow_le_pow_left hcover 2
    _ ≤ Fintype.card P * ∑ p, #(univ.filter fun ω => Off op vstar ω p) ^ 2 :=
        sq_sum_le_card_mul_sum_sq
    _ ≤ Fintype.card P * ∑ p, #(univ.filter fun x => ConflictAt op x p) :=
        Nat.mul_le_mul_left _ (sum_le_sum fun p _ => off_sq_le vstar hv p)
    _ ≤ Fintype.card P * (Q * #(conflict op)) :=
        Nat.mul_le_mul_left _ (sum_conflictAt_le Q hQ)
    _ = Fintype.card P * Q * #(conflict op) := by ring

/-- **Rewinding lemma (probabilities).** Over uniform coins, `Pr[bad] ≤ √(N · Q · Pr[conflict])`,
where `Pr[conflict]` is over two independent runs. -/
theorem prBad_le [Nonempty Ω] (vstar : P → V) (hv : IsPlurality op vstar) (Q : ℕ)
    (hQ : ∀ ω, #(univ.filter fun p => (op ω p).isSome) ≤ Q) :
    (#(bad op vstar) : ℝ) / Fintype.card Ω ≤
      Real.sqrt (Fintype.card P * Q * ((#(conflict op) : ℝ) / (Fintype.card Ω) ^ 2)) := by
  have hW : (0 : ℝ) < Fintype.card Ω := by exact_mod_cast Fintype.card_pos
  have hc : ((#(bad op vstar) : ℕ) : ℝ) ^ 2 ≤ Fintype.card P * Q * (#(conflict op) : ℝ) := by
    exact_mod_cast bad_sq_le vstar hv Q hQ
  apply Real.le_sqrt_of_sq_le
  rw [div_pow, mul_div_assoc']
  exact div_le_div_of_nonneg_right hc (by positivity)

end rewinding

end Verity.Rewinding
