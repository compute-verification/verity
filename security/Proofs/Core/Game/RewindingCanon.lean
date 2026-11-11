module

public import Proofs.Core.Game.Rewinding

/-!
# The rewinding lemma, counting each bad run at one position

`bad_sq_le` covers `bad` by the per-position sets of off-plurality runs and counts a conflicting pair at every position
where it conflicts, so it pays the most positions one run opens (`Q`). Count each bad run at one chosen position where it
is off the plurality instead (`canon`). A run off at `p` conflicts at `p` with at least as many runs as are off at `p`
(`off_le_row`), and each conflicting pair is then counted once, at its first run's chosen position, so

  `#bad ² ≤ N · #conflict`   (`bad_sq_le_canon`),

and `Pr[bad] ≤ √(N · Pr[conflict])` (`prBad_le_canon`). Positions no run opens don't count: at any `S` holding every
position some run opens, `N` reads `#S` (`bad_sq_le_canon_on`, `prBad_le_canon_on`).
-/

@[expose] public section

open Finset

namespace Verity.Rewinding

variable {Ω P V : Type} [Fintype Ω] [DecidableEq Ω] [DecidableEq V] [Fintype P]
variable {op : Ω → P → Option V}

open Classical in
/-- **The bad runs counted at `p`**: those whose chosen off-plurality position is `p`. -/
noncomputable def canon (op : Ω → P → Option V) (vstar : P → V) (p : P) : Finset Ω :=
  univ.filter fun ω => ∃ h : ∃ q, Off op vstar ω q, Classical.choose h = p

omit [DecidableEq Ω] in
theorem canon_off {vstar : P → V} {p : P} {ω : Ω} (h : ω ∈ canon op vstar p) : Off op vstar ω p := by
  simp only [canon, mem_filter, mem_univ, true_and] at h
  obtain ⟨h, rfl⟩ := h
  exact Classical.choose_spec h

omit [DecidableEq Ω] in
theorem canon_unique {vstar : P → V} {p p' : P} {ω : Ω} (h : ω ∈ canon op vstar p)
    (h' : ω ∈ canon op vstar p') : p = p' := by
  simp only [canon, mem_filter, mem_univ, true_and] at h h'
  obtain ⟨h, rfl⟩ := h
  obtain ⟨h', rfl⟩ := h'
  rfl

theorem bad_subset_canon (vstar : P → V) : bad op vstar ⊆ univ.biUnion (canon op vstar) := by
  classical
  intro ω hω
  simp only [bad, mem_filter, mem_univ, true_and] at hω
  refine mem_biUnion.mpr ⟨Classical.choose hω, mem_univ _, ?_⟩
  unfold canon
  exact mem_filter.mpr ⟨mem_univ _, hω, rfl⟩

/-- `#canon_p ² ≤ Σ_{ω₁ ∈ canon_p} #{ω₂ : ω₁, ω₂ conflict at p}`. -/
theorem canon_sq_le (vstar : P → V) (hv : IsPlurality op vstar) (p : P) :
    #(canon op vstar p) ^ 2 ≤
      ∑ ω₁ ∈ canon op vstar p, #(univ.filter fun ω₂ => ConflictAt op (ω₁, ω₂) p) := by
  have hsub : #(canon op vstar p) ≤ #(univ.filter fun ω => Off op vstar ω p) :=
    card_le_card fun ω hω => mem_filter.mpr ⟨mem_univ ω, canon_off hω⟩
  calc #(canon op vstar p) ^ 2 = ∑ _ω₁ ∈ canon op vstar p, #(canon op vstar p) := by
        rw [sum_const, smul_eq_mul, sq]
    _ ≤ ∑ ω₁ ∈ canon op vstar p, #(univ.filter fun ω₂ => ConflictAt op (ω₁, ω₂) p) := by
        refine sum_le_sum fun ω₁ hω₁ => hsub.trans ?_
        obtain ⟨hs, -⟩ := canon_off hω₁
        obtain ⟨v, hv₁⟩ := Option.isSome_iff_exists.mp hs
        exact off_le_row vstar hv p ω₁ v hv₁

/-- Each conflicting pair is counted at most once, at its first run's chosen position. -/
theorem sum_canon_le (vstar : P → V) :
    ∑ p, ∑ ω₁ ∈ canon op vstar p, #(univ.filter fun ω₂ => ConflictAt op (ω₁, ω₂) p) ≤ #(conflict op) := by
  have hrow : ∀ ω₁, (∑ p, if ω₁ ∈ canon op vstar p then
      #(univ.filter fun ω₂ => ConflictAt op (ω₁, ω₂) p) else 0) ≤
      #(univ.filter fun ω₂ => ∃ q, ConflictAt op (ω₁, ω₂) q) := by
    intro ω₁
    by_cases h : ∃ p, ω₁ ∈ canon op vstar p
    · obtain ⟨p, hp⟩ := h
      rw [sum_eq_single p (fun p' _ hne => if_neg fun h' => hne (canon_unique h' hp))
        (fun h => absurd (mem_univ p) h), if_pos hp]
      exact card_le_card fun ω₂ h => by
        simp only [mem_filter, mem_univ, true_and] at h ⊢
        exact ⟨p, h⟩
    · push_neg at h
      rw [sum_eq_zero fun p _ => if_neg (h p)]
      exact Nat.zero_le _
  have hsplit : #(conflict op) = ∑ ω₁, #(univ.filter fun ω₂ => ∃ q, ConflictAt op (ω₁, ω₂) q) := by
    rw [conflict, card_filter, ← univ_product_univ, sum_product]
    exact sum_congr rfl fun ω₁ _ => (card_filter _ _).symm
  calc ∑ p, ∑ ω₁ ∈ canon op vstar p, #(univ.filter fun ω₂ => ConflictAt op (ω₁, ω₂) p)
      = ∑ p, ∑ ω₁, if ω₁ ∈ canon op vstar p then
          #(univ.filter fun ω₂ => ConflictAt op (ω₁, ω₂) p) else 0 :=
        sum_congr rfl fun p _ => by rw [sum_ite_mem, univ_inter]
    _ = ∑ ω₁, ∑ p, if ω₁ ∈ canon op vstar p then
          #(univ.filter fun ω₂ => ConflictAt op (ω₁, ω₂) p) else 0 := sum_comm
    _ ≤ ∑ ω₁, #(univ.filter fun ω₂ => ∃ q, ConflictAt op (ω₁, ω₂) q) := sum_le_sum fun ω₁ _ => hrow ω₁
    _ = #(conflict op) := hsplit.symm

/-- **Rewinding lemma, one position a run (counts).** `#bad ² ≤ N · #conflict`. -/
theorem bad_sq_le_canon (vstar : P → V) (hv : IsPlurality op vstar) :
    #(bad op vstar) ^ 2 ≤ Fintype.card P * #(conflict op) := by
  calc #(bad op vstar) ^ 2
      ≤ (∑ p, #(canon op vstar p)) ^ 2 :=
        Nat.pow_le_pow_left ((card_le_card (bad_subset_canon vstar)).trans card_biUnion_le) 2
    _ ≤ Fintype.card P * ∑ p, #(canon op vstar p) ^ 2 := sq_sum_le_card_mul_sum_sq
    _ ≤ Fintype.card P * ∑ p, ∑ ω₁ ∈ canon op vstar p, #(univ.filter fun ω₂ => ConflictAt op (ω₁, ω₂) p) :=
        Nat.mul_le_mul_left _ (sum_le_sum fun p _ => canon_sq_le vstar hv p)
    _ ≤ Fintype.card P * #(conflict op) := Nat.mul_le_mul_left _ (sum_canon_le vstar)

/-- **Rewinding lemma, one position a run (probabilities).** Over uniform coins,
`Pr[bad] ≤ √(N · Pr[conflict])`, where `Pr[conflict]` is over two independent runs. -/
theorem prBad_le_canon [Nonempty Ω] (vstar : P → V) (hv : IsPlurality op vstar) :
    (#(bad op vstar) : ℝ) / Fintype.card Ω ≤
      Real.sqrt (Fintype.card P * ((#(conflict op) : ℝ) / (Fintype.card Ω) ^ 2)) := by
  have hc : ((#(bad op vstar) : ℕ) : ℝ) ^ 2 ≤ Fintype.card P * (#(conflict op) : ℝ) := by
    exact_mod_cast bad_sq_le_canon vstar hv
  apply Real.le_sqrt_of_sq_le
  rw [div_pow, mul_div_assoc']
  exact div_le_div_of_nonneg_right hc (by positivity)

/-- **Positions no run opens don't count.** At any `S` holding every position some run opens,
`#bad ² ≤ #S · #conflict`. -/
theorem bad_sq_le_canon_on (vstar : P → V) (hv : IsPlurality op vstar) (S : Finset P)
    (hS : ∀ ω p, (op ω p).isSome → p ∈ S) :
    #(bad op vstar) ^ 2 ≤ #S * #(conflict op) := by
  have hsub : bad op vstar ⊆ S.biUnion (canon op vstar) := fun ω hω => by
    obtain ⟨p, -, hp⟩ := mem_biUnion.mp (bad_subset_canon vstar hω)
    exact mem_biUnion.mpr ⟨p, hS ω p (canon_off hp).1, hp⟩
  calc #(bad op vstar) ^ 2
      ≤ (∑ p ∈ S, #(canon op vstar p)) ^ 2 :=
        Nat.pow_le_pow_left ((card_le_card hsub).trans card_biUnion_le) 2
    _ ≤ #S * ∑ p ∈ S, #(canon op vstar p) ^ 2 := sq_sum_le_card_mul_sum_sq
    _ ≤ #S * ∑ p ∈ S, ∑ ω₁ ∈ canon op vstar p, #(univ.filter fun ω₂ => ConflictAt op (ω₁, ω₂) p) :=
        Nat.mul_le_mul_left _ (sum_le_sum fun p _ => canon_sq_le vstar hv p)
    _ ≤ #S * ∑ p, ∑ ω₁ ∈ canon op vstar p, #(univ.filter fun ω₂ => ConflictAt op (ω₁, ω₂) p) :=
        Nat.mul_le_mul_left _ (sum_le_sum_of_subset (subset_univ S))
    _ ≤ #S * #(conflict op) := Nat.mul_le_mul_left _ (sum_canon_le vstar)

/-- `prBad_le_canon` at the positions some run opens: `Pr[bad] ≤ √(#S · Pr[conflict])`. -/
theorem prBad_le_canon_on [Nonempty Ω] (vstar : P → V) (hv : IsPlurality op vstar) (S : Finset P)
    (hS : ∀ ω p, (op ω p).isSome → p ∈ S) :
    (#(bad op vstar) : ℝ) / Fintype.card Ω ≤
      Real.sqrt ((#S : ℝ) * ((#(conflict op) : ℝ) / (Fintype.card Ω) ^ 2)) := by
  have hc : ((#(bad op vstar) : ℕ) : ℝ) ^ 2 ≤ (#S : ℝ) * (#(conflict op) : ℝ) := by
    exact_mod_cast bad_sq_le_canon_on vstar hv S hS
  apply Real.le_sqrt_of_sq_le
  rw [div_pow, mul_div_assoc']
  exact div_le_div_of_nonneg_right hc (by positivity)

end Verity.Rewinding
