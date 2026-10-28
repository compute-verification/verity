import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.Powerset
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Linarith
import Definitions.Pous
import Definitions.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers
import Proofs.Pous.Thm1.Theorem1iiSeq
import Proofs.Pous.Thm1.Theorem1iiiTimed

/-!
# Fibre counting: `PousProofs.fibre_count`, as drafted

Fix `W`. At `(pp, ω)` the codewords that decode to `W` form the fibre `F(W, pp, ω)`, and the true `C` lies in
it (`Correct`). The untimed expander sees `W`, `pp` and `ω`, so a fibre of at most `2^β` codewords has a `β`-bit
code (`exists_code`), and `INC(β, π)` gives `Pr[fibre ≤ 2^β] ≤ π`. So every `W` has a fibre above `2^β` with
probability at least `1 − π`. At a fixed `ω` the fibres over `W` partition the codewords for each `pp`, so fewer
than `2^(t + B·ℓ)/2^β` values of `W` have any fibre above `2^β`. Summing over `W` and the setup (double counting)
gives `(1 − π)·2^n·2^β ≤ 2^(t + B·ℓ)`.
-/

namespace PousThm1

open Pous
open scoped ENNReal

/-! ### Uniform probability -/

/-! ### Codes from counting -/

end PousThm1

namespace PousProofs

open Pous PousThm1
open scoped ENNReal

/-- Fibre counting (report Proposition 9(a), audit §5 item 1): untimed `INC(β, π)` for a correct scheme forces
`(1 − π)·2^(n + β) ≤ 2^(t + B·ℓ)`. Within the space bound this is `β ≲ 0.05·n`, far below `ρ|C|`, so the untimed
`INC` hypothesis of `Guarantees.Theorem1ii` cannot hold at the operating point in this model. -/
theorem fibre_count {M : IdealModel} (sch : Scheme M) (hc : sch.Correct) (β : ℕ) (π : ℝ≥0∞) (h : INC sch β π) :
    (1 - π) * 2 ^ (sch.n + β) ≤ 2 ^ (sch.t + sch.B * sch.ℓ) := by
  classical
  -- the codewords that decode to `W` under `(pp, ω)`
  let F : Bits sch.n → Bits sch.t → M.Ω → Finset sch.Code := fun W pp ω =>
    Finset.univ.filter fun C => sch.dec pp C (M.answer ω) = W
  let big : Bits sch.n → M.Ω × sch.Coins → Prop := fun W a =>
    2 ^ β < (F W (sch.enc W a.1 a.2).1 a.1).card
  -- a fibre of at most `2^β` codewords has a `β`-bit code, and the true codeword lies in it
  have hsmall : ∀ W, pr (fun a : M.Ω × sch.Coins => ¬ big W a) ≤ π := by
    intro W
    let T : Bits sch.n × Bits sch.t × M.Ω → Finset sch.Code := fun x =>
      if (F x.1 x.2.1 x.2.2).card ≤ 2 ^ β then F x.1 x.2.1 x.2.2 else ∅
    have hT : ∀ x, (T x).card ≤ 2 ^ β := by
      intro x
      simp only [T]
      split_ifs with hx
      · exact hx
      · simp
    choose c d hcd using fun x => exists_code (T x) β (hT x)
    refine le_trans (pr_mono fun a ha => ?_) (h W (fun W pp C ω => c (W, pp, ω) C) fun W pp w ω => d (W, pp, ω) w)
    show d _ (c _ _) = _
    refine hcd _ _ ?_
    simp only [big, not_lt] at ha
    simp only [T, ha, ite_true, F, Finset.mem_filter, Finset.mem_univ, true_and]
    exact hc W a.1 a.2
  -- so every `W` has a large fibre with probability at least `1 − π`
  have hbig : ∀ W, 1 - π ≤ pr (big W) := by
    intro W
    have h1 := pr_add_pr_not (big W)
    calc 1 - π ≤ 1 - pr (fun a => ¬ big W a) := tsub_le_tsub_left (hsmall W) _
      _ = pr (big W) := by
          rw [← h1, ENNReal.add_sub_cancel_right (ne_top_of_le_ne_top ENNReal.one_ne_top (pr_le_one _))]
  -- at each primitive, fewer than `2^(t + B·ℓ − β)` values of `W` have a large fibre
  have hcount : ∀ ω : M.Ω,
      (Finset.univ.filter fun W => ∃ pp, 2 ^ β < (F W pp ω).card).card * 2 ^ β ≤ 2 ^ (sch.t + sch.B * sch.ℓ) := by
    intro ω
    calc (Finset.univ.filter fun W => ∃ pp, 2 ^ β < (F W pp ω).card).card * 2 ^ β
        = ∑ _W ∈ Finset.univ.filter fun W => ∃ pp, 2 ^ β < (F W pp ω).card, 2 ^ β := by
          rw [Finset.sum_const, smul_eq_mul]
      _ ≤ ∑ W ∈ Finset.univ.filter fun W => ∃ pp, 2 ^ β < (F W pp ω).card, ∑ pp, (F W pp ω).card := by
          refine Finset.sum_le_sum fun W hW => ?_
          obtain ⟨pp, hpp⟩ := (Finset.mem_filter.1 hW).2
          exact hpp.le.trans (Finset.single_le_sum (f := fun pp => (F W pp ω).card) (fun _ _ => Nat.zero_le _)
            (Finset.mem_univ pp))
      _ ≤ ∑ W, ∑ pp, (F W pp ω).card := Finset.sum_le_sum_of_subset (Finset.subset_univ _)
      _ = ∑ pp : Bits sch.t, ∑ W, (F W pp ω).card := Finset.sum_comm
      _ = ∑ _pp : Bits sch.t, Fintype.card sch.Code := by
          refine Finset.sum_congr rfl fun pp _ => ?_
          rw [← Finset.card_univ, Finset.card_eq_sum_card_fiberwise (f := fun C => sch.dec pp C (M.answer ω))
            (t := Finset.univ) fun _ _ => Finset.mem_coe.2 (Finset.mem_univ _)]
      _ = 2 ^ (sch.t + sch.B * sch.ℓ) := by
          rw [Finset.sum_const, Finset.card_univ, smul_eq_mul, Fintype.card_fun, Fintype.card_fin,
            Fintype.card_fun, Fintype.card_fun, Fintype.card_bool, Fintype.card_fin, Fintype.card_fin, ← pow_mul,
            ← pow_add, mul_comm sch.ℓ]
  -- double counting over `(W, setup)`
  have hdouble : (∑ W, (Finset.univ.filter (big W)).card) * 2 ^ β ≤
      Fintype.card (M.Ω × sch.Coins) * 2 ^ (sch.t + sch.B * sch.ℓ) := by
    have hswap : ∑ W, (Finset.univ.filter (big W)).card =
        ∑ a : M.Ω × sch.Coins, (Finset.univ.filter fun W => big W a).card := by
      simp only [Finset.card_filter]
      exact Finset.sum_comm
    rw [hswap, Finset.sum_mul, ← Finset.card_univ, ← smul_eq_mul, ← Finset.sum_const]
    refine Finset.sum_le_sum fun a _ => ?_
    refine le_trans (Nat.mul_le_mul_right _ (Finset.card_le_card fun W hW => ?_)) (hcount a.1)
    rw [Finset.mem_filter] at hW ⊢
    exact ⟨Finset.mem_univ _, _, hW.2⟩
  have hN0 : (Fintype.card (M.Ω × sch.Coins) : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hNt : (Fintype.card (M.Ω × sch.Coins) : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  calc (1 - π) * 2 ^ (sch.n + β) = (∑ _W : Bits sch.n, (1 - π)) * 2 ^ β := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fun, Fintype.card_bool, Fintype.card_fin,
          nsmul_eq_mul, pow_add]
        push_cast
        ring
    _ ≤ (∑ W, pr (big W)) * 2 ^ β := by gcongr with W; exact hbig W
    _ = (∑ W, ((Finset.univ.filter (big W)).card : ℝ≥0∞)) * 2 ^ β / Fintype.card (M.Ω × sch.Coins) := by
        simp only [pr_def, div_eq_mul_inv, ← Finset.sum_mul]
        ring
    _ ≤ Fintype.card (M.Ω × sch.Coins) * 2 ^ (sch.t + sch.B * sch.ℓ) / Fintype.card (M.Ω × sch.Coins) := by
        exact ENNReal.div_le_div_right (by exact_mod_cast hdouble) _
    _ = 2 ^ (sch.t + sch.B * sch.ℓ) := by
        rw [mul_comm, ENNReal.mul_div_cancel_right hN0 hNt]

end PousProofs
