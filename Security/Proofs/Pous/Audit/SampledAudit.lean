import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Option
import Mathlib.Data.Fintype.Powerset
import Mathlib.Data.Fintype.Sum
import Mathlib.Data.List.Chain
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Linarith
import Definitions.Pous
import Definitions.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers
import Proofs.Pous.Thm1.Theorem1iTimed
import Proofs.Pous.Thm1.Theorem1iiSeq
import Proofs.Pous.Thm1.Theorem1iiiTimed
import Proofs.Pous.B5.StackedOneWay

/-!
# From "fewer than `k` answerable" to a sampled audit, and the stacked one-way graph's audit bound

The generic step, in the trusted layer's game (`PousAudit`):

* `sampled_audit_prod`: product-form responders (`TimedProdResponder`; simultaneous reveal, each answer from the
  state and its own index): `Pr[pass] ≤ p + ((k − 1)/B)^t`.
* `sampled_audit_seq`: sequential reveal (`SeqResponder`). The answerable set is taken after each history of fewer
  than `t` indices, and a union over the `|Hist| = Σ_{j<t} B^j` histories gives `((k − 1)/B)^t + |Hist|·p`.
* `sampled_audit_sim`: general simultaneous reveal (`SimResponder`), where an answer may depend on the whole tuple.
  As in `theorem1_iii_timed`, position `j`'s answerable set is `L_j = {i : Pr_I[answer_j(I[j := i]) = c_i] ≥ η}`,
  giving `p + ((k − 1)/B)^t + t·η`.

The instance, in B5's game on Fisch's stacked graph (`stacked_audit_seq`): a sequential responder answering `t`
uniform last-layer challenges within `(⌊βn⌋ − 1, Q)` passes with probability at most
`((n − ⌈εn/2⌉)/n)^t + |Hist| · N · 2^S · ((n(Q + 1) + N)²/2^w)^(⌊(1 − ε)n⌋ + 1)`. Each history's next answers are a
preprocessor (the state after that history) and per-challenge programs, so `b5_stacked` bounds them.
`stacked_audit_one_percent` certifies `t = 210` for `1% + 2^-128` at `n = 2^16`, `N = 2^17`, `w = 8192`, `Q = 2^20`,
`S = ⌊(18/19)·n·w⌋` and `ε = 1/23`.

It builds on `Pous.SecurityProofs.B5.StackedOneWay` (Claim 13, B5 and the bridge) and the Theorem 1 library (`PousThm1`,
`Pous.SecurityProofs.Thm1`).
-/

namespace PousThm1

open Pous
open scoped ENNReal

/-! ### Uniform probability -/

/-- `k` indices, all in `G`: `|G|^k` of them. -/
theorem card_pi_mem (k B : ℕ) (G : Finset (Fin B)) :
    (Finset.univ.filter fun I : Fin k → Fin B => ∀ j, I j ∈ G).card = G.card ^ k := by
  have : (Finset.univ.filter fun I : Fin k → Fin B => ∀ j, I j ∈ G) = Fintype.piFinset fun _ => G := by
    ext I; simp [Fintype.mem_piFinset]
  rw [this, Fintype.card_piFinset, Finset.prod_const, Finset.card_univ, Fintype.card_fin]

/-! ### The threshold `p₀` -/

open Classical in
/-- **Averaging.** `k` uniform challenges all land on blocks the responder gets right with probability at most
`Pr[the wrong blocks fit in β bits] + p₀^k`. -/
theorem pr_all_good_le {Ω : Type} [Fintype Ω] [Nonempty Ω] (B ℓ β S k : ℕ) (hB : 0 < B)
    (good : Ω → Fin B → Prop) :
    pr (fun x : Ω × (Fin k → Fin B) => ∀ j, good x.1 (x.2 j)) ≤
      pr (fun a => S + B + (Finset.univ.filter fun i => ¬ good a i).card * ℓ ≤ β) +
        ENNReal.ofReal (p₀ β S B ℓ) ^ k := by
  have : Nonempty (Fin B) := ⟨⟨0, hB⟩⟩
  refine pr_prod_le (α := Ω) (β := Fin k → Fin B) (fun a I => ∀ j, good a (I j))
    (fun a => S + B + (Finset.univ.filter fun i => ¬ good a i).card * ℓ ≤ β) (ENNReal.ofReal (p₀ β S B ℓ) ^ k)
    fun a ha => ?_
  have hgb := Finset.card_filter_add_card_filter_not (s := Finset.univ) (good a)
  rw [Finset.card_univ, Fintype.card_fin] at hgb
  have hr := good_le_p₀ B ℓ β S _ _ hB hgb (by omega)
  have hB0 : ((B : ℝ≥0∞) ^ k) ≠ 0 := pow_ne_zero _ (by exact_mod_cast hB.ne')
  have hBt : ((B : ℝ≥0∞) ^ k) ≠ ⊤ := ENNReal.pow_ne_top (ENNReal.natCast_ne_top _)
  have hcard : (Finset.univ.filter fun I : Fin k → Fin B => ∀ j, good a (I j)).card =
      (Finset.univ.filter (good a)).card ^ k := by
    rw [← card_pi_mem]; congr 1; ext I; simp
  rw [pr_def, hcard, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin, Nat.cast_pow, Nat.cast_pow,
    ENNReal.div_le_iff hB0 hBt, ← mul_pow]
  gcongr
  calc ((Finset.univ.filter (good a)).card : ℝ≥0∞)
      = ENNReal.ofReal ((Finset.univ.filter (good a)).card : ℝ) := (ENNReal.ofReal_natCast _).symm
    _ ≤ ENNReal.ofReal (p₀ β S B ℓ * B) := ENNReal.ofReal_le_ofReal hr
    _ = ENNReal.ofReal (p₀ β S B ℓ) * B := by
        rw [ENNReal.ofReal_mul' (Nat.cast_nonneg _), ENNReal.ofReal_natCast]

/-! ### Compression -/

/-! ### More uniform probability -/

/-! ### Programs -/

section progMap

variable {M : IdealModel} {α β : Type} (g : α → β) (f : M.Qry → M.Ans)

end progMap

/-! ### Compression against `TimedINC` -/

/-! ### Sequential responders -/

/-! ### Theorem 1(iii): probability tools -/

/-! ### Theorem 1(iii): the joint responder at a fixed setup -/

/-! ### Theorem 1(iii): missing a likely block -/

end PousThm1

/-! ## From "at most `k − 1` answerable" to a sampled audit -/

namespace PousAudit

open Pous PousThm1
open scoped ENNReal

variable {M : IdealModel}

/-- **Product form** (simultaneous reveal, each answer from the state and its own index: `TimedProdResponder`).
If, except with probability `p` over the setup, fewer than `k` blocks are answerable, a `t`-sample audit with
replacement passes with probability at most `p + ((k − 1)/B)^t`. -/
theorem sampled_audit_prod (sch : Scheme M) (S D Q t k : ℕ) (p : ℝ≥0∞) (W : Bits sch.n)
    (A₁ : Preprocessor sch S) (A₂ : TimedProdResponder sch S D Q)
    (h : pr (fun a : M.Ω × sch.Coins => k ≤ (Finset.univ.filter fun i =>
      (A₂.prog W (sch.enc W a.1 a.2).1 (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) i).run
        (M.answer a.1) = (sch.enc W a.1 a.2).2 i).card) ≤ p) :
    pr (timedProdAccepts sch t W A₁ A₂) ≤ p + (((k - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ t := by
  classical
  let good : M.Ω × sch.Coins → Fin sch.B → Prop := fun a i =>
    (A₂.prog W (sch.enc W a.1 a.2).1 (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) i).run
      (M.answer a.1) = (sch.enc W a.1 a.2).2 i
  have hre : pr (timedProdAccepts sch t W A₁ A₂) =
      pr (fun x : (M.Ω × sch.Coins) × (Fin t → Fin sch.B) => ∀ j, good x.1 (x.2 j)) :=
    pr_equiv (Equiv.prodAssoc _ _ _).symm (fun x : (M.Ω × sch.Coins) × (Fin t → Fin sch.B) => ∀ j, good x.1 (x.2 j))
  rw [hre]
  have : Nonempty (Fin sch.B) := ⟨⟨0, sch.B_pos⟩⟩
  have hB0 : (sch.B : ℝ≥0∞) ≠ 0 := by exact_mod_cast sch.B_pos.ne'
  refine (pr_prod_le (α := M.Ω × sch.Coins) (β := Fin t → Fin sch.B) (fun a I => ∀ j, good a (I j))
    (fun a => k ≤ (Finset.univ.filter fun i => good a i).card) ((((k - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ t)
    fun a ha => ?_).trans (add_le_add (le_of_eq_of_le (by rfl) h) le_rfl)
  have hlt : (Finset.univ.filter (good a)).card ≤ k - 1 := by
    have : ¬ k ≤ (Finset.univ.filter (good a)).card := by convert ha
    omega
  have hBt : ((sch.B : ℝ≥0∞) ^ t) ≠ ⊤ := ENNReal.pow_ne_top (ENNReal.natCast_ne_top _)
  have hcard : (Finset.univ.filter fun I : Fin t → Fin sch.B => ∀ j, good a (I j)).card =
      (Finset.univ.filter (good a)).card ^ t := by
    rw [← card_pi_mem]; congr 1; ext I; simp
  rw [pr_def, hcard, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin, Nat.cast_pow, Nat.cast_pow,
    ENNReal.div_le_iff (pow_ne_zero _ hB0) hBt, ← mul_pow, ENNReal.div_mul_cancel hB0 (ENNReal.natCast_ne_top _)]
  gcongr

/-- **Sequential reveal.** If after every history of fewer than `t` indices, except with probability `p` over the
setup, fewer than `k` blocks are answerable in the next round, a `t`-round audit passes with probability at most
`((k − 1)/B)^t + |Hist|·p`, where `|Hist| = Σ_{j<t} B^j`. -/
theorem sampled_audit_seq (sch : Scheme M) (S D Q t k : ℕ) (p : ℝ≥0∞) (W : Bits sch.n)
    (A₁ : Preprocessor sch S) (A₂ : SeqResponder sch S D Q)
    (h : ∀ hist : List (Fin sch.B), hist.length < t →
      pr (fun a : M.Ω × sch.Coins => k ≤ (Finset.univ.filter fun i =>
        ((A₂.step W (sch.enc W a.1 a.2).1 hist.length
          (seqState (M.answer a.1) (A₂.step W (sch.enc W a.1 a.2).1) 0 hist
            (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1)) i).run (M.answer a.1)).1 =
          (sch.enc W a.1 a.2).2 i).card) ≤ p) :
    pr (seqAccepts sch t W A₁ A₂) ≤
      (((k - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ t + (∑ j ∈ Finset.range t, (sch.B : ℝ≥0∞) ^ j) * p := by
  classical
  let good : List (Fin sch.B) → M.Ω × sch.Coins → Fin sch.B → Prop := fun hist a i =>
    ((A₂.step W (sch.enc W a.1 a.2).1 hist.length
      (seqState (M.answer a.1) (A₂.step W (sch.enc W a.1 a.2).1) 0 hist
        (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1)) i).run (M.answer a.1)).1 =
      (sch.enc W a.1 a.2).2 i
  let E : M.Ω × sch.Coins → Prop := fun a =>
    ∃ j ∈ Finset.range t, ∃ v ∈ (Finset.univ : Finset (Fin j → Fin sch.B)),
      k ≤ (Finset.univ.filter fun i => good (List.ofFn v) a i).card
  have hre : pr (seqAccepts sch t W A₁ A₂) = pr (fun x : (M.Ω × sch.Coins) × (Fin t → Fin sch.B) =>
      seqAnswers (M.answer x.1.1) (A₂.step W (sch.enc W x.1.1 x.1.2).1) 0 (List.ofFn x.2)
        (A₁ W (sch.enc W x.1.1 x.1.2).1 (sch.enc W x.1.1 x.1.2).2 x.1.1) =
      List.ofFn fun j => (sch.enc W x.1.1 x.1.2).2 (x.2 j)) :=
    pr_equiv (Equiv.prodAssoc _ _ _).symm (fun x : (M.Ω × sch.Coins) × (Fin t → Fin sch.B) =>
      seqAnswers (M.answer x.1.1) (A₂.step W (sch.enc W x.1.1 x.1.2).1) 0 (List.ofFn x.2)
        (A₁ W (sch.enc W x.1.1 x.1.2).1 (sch.enc W x.1.1 x.1.2).2 x.1.1) =
      List.ofFn fun j => (sch.enc W x.1.1 x.1.2).2 (x.2 j))
  rw [hre]
  have : Nonempty (Fin sch.B) := ⟨⟨0, sch.B_pos⟩⟩
  refine (pr_prod_le (α := M.Ω × sch.Coins) (β := Fin t → Fin sch.B)
    (fun a I => seqAnswers (M.answer a.1) (A₂.step W (sch.enc W a.1 a.2).1) 0 (List.ofFn I)
        (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) =
      List.ofFn fun j => (sch.enc W a.1 a.2).2 (I j))
    E ((((k - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ t) fun a ha => ?_).trans ?_
  · refine pr_seq_le _ _ _ _ t 0 _ fun hist hh => ?_
    have hlt : (Finset.univ.filter fun i => good hist a i).card ≤ k - 1 := by
      have : ¬ k ≤ (Finset.univ.filter fun i => good hist a i).card := by
        intro hk
        refine ha ⟨hist.length, Finset.mem_range.2 hh, hist.get, Finset.mem_univ _, ?_⟩
        rwa [List.ofFn_get]
      omega
    have hpr : pr (good hist a) ≤ ((k - 1 : ℕ) : ℝ≥0∞) / sch.B := by
      rw [pr_def, Fintype.card_fin]
      exact ENNReal.div_le_div_right (by exact_mod_cast hlt) _
    simpa only [good, Nat.zero_add] using hpr
  · have hE : pr E ≤ (∑ j ∈ Finset.range t, (sch.B : ℝ≥0∞) ^ j) * p := by
      calc pr E ≤ ∑ j ∈ Finset.range t, pr (fun a => ∃ v ∈ (Finset.univ : Finset (Fin j → Fin sch.B)),
            k ≤ (Finset.univ.filter fun i => good (List.ofFn v) a i).card) := pr_exists_le _ _
        _ ≤ ∑ j ∈ Finset.range t, ∑ _v : Fin j → Fin sch.B, p := by
            refine Finset.sum_le_sum fun j hj => (pr_exists_le _ _).trans (Finset.sum_le_sum fun v _ => ?_)
            have := h (List.ofFn v) (by rw [List.length_ofFn]; exact Finset.mem_range.1 hj)
            convert this using 3
        _ = (∑ j ∈ Finset.range t, (sch.B : ℝ≥0∞) ^ j) * p := by
            rw [Finset.sum_mul]
            refine Finset.sum_congr rfl fun j _ => ?_
            rw [Finset.sum_const, Finset.card_univ, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin,
              nsmul_eq_mul, Nat.cast_pow]
    calc pr E + (((k - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ t
        ≤ (∑ j ∈ Finset.range t, (sch.B : ℝ≥0∞) ^ j) * p + (((k - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ t :=
          add_le_add hE le_rfl
      _ = _ := add_comm _ _

/-- **General simultaneous reveal** (the answer at position `j` may depend on the whole challenge tuple). As in
`theorem1_iii_timed`, the answerable set at position `j` is `L_j = {i : Pr_I[answer_j(I[j := i]) = c_i] ≥ η}`. If,
except with probability `p` over the setup, every `|L_j| < k`, the audit passes with probability at most
`p + ((k − 1)/B)^t + t·η`. -/
theorem sampled_audit_sim (sch : Scheme M) (S D Q t k : ℕ) (p η : ℝ≥0∞) (W : Bits sch.n)
    (A₁ : Preprocessor sch S) (A₂ : SimResponder sch S D Q t)
    (h : pr (fun a : M.Ω × sch.Coins => ∃ j : Fin t, k ≤ (Finset.univ.filter fun i =>
      η ≤ pr (fun I : Fin t → Fin sch.B =>
        (A₂.prog W (sch.enc W a.1 a.2).1 (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1)
          (Function.update I j i)).run (M.answer a.1) j = (sch.enc W a.1 a.2).2 i)).card) ≤ p) :
    pr (simAccepts sch W A₁ A₂) ≤ p + ((((k - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ t + t * η) := by
  classical
  let ans : M.Ω × sch.Coins → (Fin t → Fin sch.B) → Fin t → Bits sch.ℓ := fun a I =>
    (A₂.prog W (sch.enc W a.1 a.2).1 (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) I).run (M.answer a.1)
  let L : M.Ω × sch.Coins → Fin t → Finset (Fin sch.B) := fun a j => Finset.univ.filter fun i =>
    η ≤ pr (fun I : Fin t → Fin sch.B => ans a (Function.update I j i) j = (sch.enc W a.1 a.2).2 i)
  have hre : pr (simAccepts sch W A₁ A₂) = pr (fun x : (M.Ω × sch.Coins) × (Fin t → Fin sch.B) =>
      ans x.1 x.2 = fun j => (sch.enc W x.1.1 x.1.2).2 (x.2 j)) :=
    pr_equiv (Equiv.prodAssoc _ _ _).symm (fun x : (M.Ω × sch.Coins) × (Fin t → Fin sch.B) =>
      ans x.1 x.2 = fun j => (sch.enc W x.1.1 x.1.2).2 (x.2 j))
  rw [hre]
  have : Nonempty (Fin sch.B) := ⟨⟨0, sch.B_pos⟩⟩
  have hB0 : (sch.B : ℝ≥0∞) ≠ 0 := by exact_mod_cast sch.B_pos.ne'
  refine (pr_prod_le (α := M.Ω × sch.Coins) (β := Fin t → Fin sch.B)
    (fun a I => ans a I = fun j => (sch.enc W a.1 a.2).2 (I j)) (fun a => ∃ j : Fin t, k ≤ (L a j).card)
    ((((k - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ t + t * η) fun a ha => ?_).trans
    (add_le_add (le_of_eq_of_le (by rfl) h) le_rfl)
  refine pr_joint_le (ans a) _ η (((k - 1 : ℕ) : ℝ≥0∞) / sch.B) (L a) (fun j i hi => ?_) fun j => ?_
  · simp only [L, Finset.mem_filter, Finset.mem_univ, true_and, not_le] at hi
    exact hi.le
  · have hlt : (L a j).card ≤ k - 1 := by
      have : ¬ k ≤ (L a j).card := fun hk => ha ⟨j, hk⟩
      omega
    rw [ENNReal.div_mul_cancel hB0 (ENNReal.natCast_ne_top _)]
    exact_mod_cast hlt

end PousAudit

/-! ## The sampled audit on the stacked one-way graph -/

namespace PousAudit

open Pous PousThm1 Pous.DRG Pous.Labelling
open scoped ENNReal

/-- **Sequential audit of the stacked one-way graph** (B5's game). The challenges are `t` uniform last-layer
indices, the responder carries an `S`-bit state and answers each within `(⌊βn⌋ − 1, Q)`, and it passes iff every
answer is the challenged label. Each history's next answers come from a preprocessor (the state after that
history) and a per-challenge program, so `b5_stacked` bounds them, and the chain rule plus the union over
histories finish. -/
theorem stacked_audit_seq {ℓ n N : ℕ} {V : Type} [Fintype V] [DecidableEq V] (α β ε : ℝ)
    (node : Fin ℓ → Fin n → V) (level : V → Fin ℓ) (drg : Fin ℓ → Fin n → Fin n → Prop) (green : V → V → Prop)
    (hnode : ∀ i, Function.Injective (node i)) (hlevel : ∀ i a, level (node i a) = i)
    (hdr : ∀ i, DepthRobust n (drg i) α β)
    (hsc : ∀ i j : Fin ℓ, (j : ℕ) = i + 1 → DisjointPaths node level green i j)
    (hn : 0 < n) (hε : 0 < ε) (hε1 : ε ≤ 1) (hℓ : 1 ≤ (1 - α) * ℓ) (hβ : 1 ≤ β * n)
    (e : V ≃ Fin N) (G : TopoDAG N) (hG : ∀ u v, e u ∈ G.par (e v) ↔ StackedEdge node drg green u v)
    (t : Fin ℓ) (ht : (t : ℕ) + 1 = ℓ)
    (w S Q tt : ℕ) (d : Fin N → Bits w)
    (A₁ : (LabelQry N w → Bits w) → Bits S)
    (step : ℕ → Bits S → Fin n → Prog (IdealModel.randomOracle (LabelQry N w) (Bits w)) (Bits w × Bits S))
    (hstep : ∀ j σ b, (step j σ b).Bounded (⌊β * n⌋₊ - 1) Q) :
    pr (fun x : (LabelQry N w → Bits w) × (Fin tt → Fin n) =>
        seqAnswers x.1 step 0 (List.ofFn x.2) (A₁ x.1) = List.ofFn fun j => label G d x.1 (e (node t (x.2 j)))) ≤
      (((n - ⌈ε / 2 * n⌉₊ : ℕ) : ℝ≥0∞) / n) ^ tt +
        (∑ j ∈ Finset.range tt, (n : ℝ≥0∞) ^ j) *
          (N * 2 ^ S * ((((n * (Q + 1) + N) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ w) ^ (⌊(1 - ε) * n⌋₊ + 1)) := by
  classical
  have : NeZero n := ⟨hn.ne'⟩
  have : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
  let lab : (LabelQry N w → Bits w) → Fin n → Bits w := fun H b => label G d H (e (node t b))
  let good : List (Fin n) → (LabelQry N w → Bits w) → Fin n → Prop := fun hist H b =>
    ((step hist.length (seqState H step 0 hist (A₁ H)) b).run H).1 = lab H b
  let E : (LabelQry N w → Bits w) → Prop := fun H =>
    ∃ j ∈ Finset.range tt, ∃ v ∈ (Finset.univ : Finset (Fin j → Fin n)),
      n - ⌈ε / 2 * n⌉₊ + 1 ≤ (Finset.univ.filter (good (List.ofFn v) H)).card
  have hinj : Function.Injective fun b : Fin n => e (node t b) := fun a b h => hnode t (e.injective h)
  let idx : Fin N → Fin n := Function.invFun fun b : Fin n => e (node t b)
  have hidx : ∀ b, idx (e (node t b)) = b := Function.leftInverse_invFun hinj
  refine (pr_prod_le (α := LabelQry N w → Bits w) (β := Fin tt → Fin n)
    (fun H I => seqAnswers H step 0 (List.ofFn I) (A₁ H) = List.ofFn fun j => lab H (I j))
    E ((((n - ⌈ε / 2 * n⌉₊ : ℕ) : ℝ≥0∞) / n) ^ tt) fun H hH => ?_).trans ?_
  · -- no history has `k` answerable challenges: every round is right w.p. `≤ (k − 1)/n`
    refine pr_seq_le _ _ _ _ tt 0 _ fun hist hh => ?_
    have hlt : (Finset.univ.filter (good hist H)).card ≤ n - ⌈ε / 2 * n⌉₊ := by
      have : ¬ n - ⌈ε / 2 * n⌉₊ + 1 ≤ (Finset.univ.filter (good hist H)).card := by
        intro hk
        refine hH ⟨hist.length, Finset.mem_range.2 hh, hist.get, Finset.mem_univ _, ?_⟩
        rwa [List.ofFn_get]
      omega
    have hpr : pr (good hist H) ≤ ((n - ⌈ε / 2 * n⌉₊ : ℕ) : ℝ≥0∞) / n := by
      rw [pr_def, Fintype.card_fin]
      exact ENNReal.div_le_div_right (by exact_mod_cast hlt) _
    simpa only [good, Nat.zero_add] using hpr
  · -- each history: `b5_stacked` for the state after it and the next round's per-challenge programs
    have hB5 : ∀ hist : List (Fin n), pr (fun H => n - ⌈ε / 2 * n⌉₊ + 1 ≤
        (Finset.univ.filter (good hist H)).card) ≤
        N * 2 ^ S * ((((n * (Q + 1) + N) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ w) ^ (⌊(1 - ε) * n⌋₊ + 1) := by
      intro hist
      refine le_trans (pr_mono fun H hH => ?_) (PousBridge.b5_stacked α β ε node level drg green hnode hlevel
        hdr hsc hn hε hε1 hℓ hβ e G hG t ht w S Q d (fun H => seqState H step 0 hist (A₁ H))
        (fun σ c => progMap Prod.fst (step hist.length σ (idx c)))
        fun σ c => bounded_progMap _ (hstep _ _ _))
      refine hH.trans (le_of_eq_of_le (Finset.card_image_of_injective _ hinj).symm (Finset.card_le_card ?_))
      intro c hc
      obtain ⟨b, hb, rfl⟩ := Finset.mem_image.1 hc
      rw [Finset.mem_filter] at hb ⊢
      refine ⟨Finset.mem_image.2 ⟨b, Finset.mem_univ _, rfl⟩, ?_⟩
      refine (run_progMap (M := IdealModel.randomOracle (LabelQry N w) (Bits w)) Prod.fst H _).trans ?_
      exact (congrArg (fun c => (Prog.run H (step hist.length (seqState H step 0 hist (A₁ H)) c)).1)
        (hidx b)).trans hb.2
    have hE : pr E ≤ (∑ j ∈ Finset.range tt, (n : ℝ≥0∞) ^ j) *
        (N * 2 ^ S * ((((n * (Q + 1) + N) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ w) ^ (⌊(1 - ε) * n⌋₊ + 1)) := by
      calc pr E ≤ ∑ j ∈ Finset.range tt, pr (fun H => ∃ v ∈ (Finset.univ : Finset (Fin j → Fin n)),
            n - ⌈ε / 2 * n⌉₊ + 1 ≤ (Finset.univ.filter (good (List.ofFn v) H)).card) := pr_exists_le _ _
        _ ≤ ∑ j ∈ Finset.range tt, ∑ _v : Fin j → Fin n,
            N * 2 ^ S * ((((n * (Q + 1) + N) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ w) ^ (⌊(1 - ε) * n⌋₊ + 1) :=
            Finset.sum_le_sum fun j _ => (pr_exists_le _ _).trans (Finset.sum_le_sum fun v _ => hB5 _)
        _ = _ := by
            rw [Finset.sum_mul]
            refine Finset.sum_congr rfl fun j _ => ?_
            rw [Finset.sum_const, Finset.card_univ, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin,
              nsmul_eq_mul, Nat.cast_pow]
    calc pr E + (((n - ⌈ε / 2 * n⌉₊ : ℕ) : ℝ≥0∞) / n) ^ tt
        ≤ _ + _ := add_le_add hE le_rfl
      _ = _ := add_comm _ _

/-! ### Numbers: the audit length for 1% -/

/-- `Σ_{j<t} n^j ≤ n^t` for `n ≥ 2`. -/
theorem sum_pow_le (n t : ℕ) (hn : 2 ≤ n) : ∑ j ∈ Finset.range t, (n : ℝ≥0∞) ^ j ≤ (n : ℝ≥0∞) ^ t := by
  have h : ∑ j ∈ Finset.range t, n ^ j ≤ n ^ t := by
    induction t with
    | zero => simp
    | succ t ih =>
      rw [Finset.sum_range_succ, pow_succ]
      nlinarith [pow_pos (by omega : 0 < n) t]
  exact_mod_cast h

/-- `2^a · 2^-b = 2^-(b − a)` for `a ≤ b`. -/
theorem two_pow_mul_inv_pow (a b : ℕ) (h : a ≤ b) :
    (2 : ℝ≥0∞) ^ a * (2⁻¹ : ℝ≥0∞) ^ b = (2⁻¹ : ℝ≥0∞) ^ (b - a) := by
  obtain ⟨c, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [pow_add, ← mul_assoc, ← mul_pow, ENNReal.mul_inv_cancel two_ne_zero ENNReal.ofNat_ne_top, one_pow, one_mul,
    Nat.add_sub_cancel_left]

/-- `2^a · (2^b · 2^c · 2^-d) = 2^-(d − (a + b + c))` for `a + b + c ≤ d`. -/
theorem two_pow_combine (a b c d : ℕ) (h : a + b + c ≤ d) :
    (2 : ℝ≥0∞) ^ a * ((2 : ℝ≥0∞) ^ b * 2 ^ c * (2⁻¹ : ℝ≥0∞) ^ d) = (2⁻¹ : ℝ≥0∞) ^ (d - (a + b + c)) := by
  rw [← two_pow_mul_inv_pow _ _ h, pow_add, pow_add]
  ring

/-- `x^2/2^w ≤ 2^-(w − c)` when `x^2 ≤ 2^c ≤ 2^w`. -/
theorem sq_div_two_pow_le (x c w : ℕ) (hx : x ^ 2 ≤ 2 ^ c) (hc : c ≤ w) :
    ((x ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ w ≤ (2⁻¹ : ℝ≥0∞) ^ (w - c) := by
  calc ((x ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ w ≤ (2 : ℝ≥0∞) ^ c / 2 ^ w := by
        gcongr
        exact_mod_cast hx
    _ = (2⁻¹ : ℝ≥0∞) ^ (w - c) := by
        rw [div_eq_mul_inv, ENNReal.inv_pow, two_pow_mul_inv_pow c w hc]

/-- **At the example parameters**: `n = 2^16` last-layer blocks of `w = 8192` bits (1 KB), `N = 2^17` nodes,
`Q = 2^20` queries per answer, an `S = ⌊(18/19)·n·w⌋ = 508614548`-bit state (`ρ` of the last layer) and `ε = 1/23`,
a `t = 210`-round sequential audit passes with probability at most `1% + 2^-128`. -/
theorem stacked_audit_one_percent {ℓ : ℕ} {V : Type} [Fintype V] [DecidableEq V] (α β : ℝ)
    (node : Fin ℓ → Fin (2 ^ 16) → V) (level : V → Fin ℓ) (drg : Fin ℓ → Fin (2 ^ 16) → Fin (2 ^ 16) → Prop)
    (green : V → V → Prop)
    (hnode : ∀ i, Function.Injective (node i)) (hlevel : ∀ i a, level (node i a) = i)
    (hdr : ∀ i, DepthRobust (2 ^ 16) (drg i) α β)
    (hsc : ∀ i j : Fin ℓ, (j : ℕ) = i + 1 → DisjointPaths node level green i j)
    (hℓ : 1 ≤ (1 - α) * ℓ) (hβ : 1 ≤ β * (2 ^ 16 : ℕ))
    (e : V ≃ Fin (2 ^ 17)) (G : TopoDAG (2 ^ 17)) (hG : ∀ u v, e u ∈ G.par (e v) ↔ StackedEdge node drg green u v)
    (t : Fin ℓ) (ht : (t : ℕ) + 1 = ℓ) (d : Fin (2 ^ 17) → Bits 8192)
    (A₁ : (LabelQry (2 ^ 17) 8192 → Bits 8192) → Bits 508614548)
    (step : ℕ → Bits 508614548 → Fin (2 ^ 16) →
      Prog (IdealModel.randomOracle (LabelQry (2 ^ 17) 8192) (Bits 8192)) (Bits 8192 × Bits 508614548))
    (hstep : ∀ j σ b, (step j σ b).Bounded (⌊β * (2 ^ 16 : ℕ)⌋₊ - 1) (2 ^ 20)) :
    pr (fun x : (LabelQry (2 ^ 17) 8192 → Bits 8192) × (Fin 210 → Fin (2 ^ 16)) =>
        seqAnswers x.1 step 0 (List.ofFn x.2) (A₁ x.1) = List.ofFn fun j => label G d x.1 (e (node t (x.2 j)))) ≤
      1 / 100 + (2⁻¹ : ℝ≥0∞) ^ 128 := by
  have hmain := stacked_audit_seq α β (1 / 23) node level drg green hnode hlevel hdr hsc (by norm_num)
    (by norm_num) (by norm_num) hℓ hβ e G hG t ht 8192 508614548 (2 ^ 20) 210 d A₁ step hstep
  have hceil : ⌈(1 / 23 : ℝ) / 2 * ((2 ^ 16 : ℕ) : ℝ)⌉₊ = 1425 := by
    rw [Nat.ceil_eq_iff (by norm_num)]
    norm_num
  have hfloor : ⌊(1 - 1 / 23 : ℝ) * ((2 ^ 16 : ℕ) : ℝ)⌋₊ = 62686 := by
    rw [Nat.floor_eq_iff (by norm_num)]
    norm_num
  rw [hceil, hfloor] at hmain
  refine hmain.trans (add_le_add ?_ ?_)
  · -- `(64111/65536)^210 ≤ 1/100`
    have hr : ((64111 / 65536 : ℝ)) ^ 210 ≤ 1 / 100 := by norm_num
    have h1 : (((2 ^ 16 - 1425 : ℕ) : ℝ≥0∞) / ((2 ^ 16 : ℕ) : ℝ≥0∞)) = ENNReal.ofReal (64111 / 65536) := by
      rw [ENNReal.ofReal_div_of_pos (by norm_num)]
      norm_num
    rw [h1, ← ENNReal.ofReal_pow (by norm_num)]
    calc ENNReal.ofReal ((64111 / 65536 : ℝ) ^ 210) ≤ ENNReal.ofReal (1 / 100) := ENNReal.ofReal_le_ofReal hr
      _ = 1 / 100 := by rw [ENNReal.ofReal_div_of_pos (by norm_num)]; simp
  · -- `|Hist| · N · 2^S · (Q_tot²/2^w)^(s+1) ≤ 2^-128`, in exponents of 2
    have hsq : (2 ^ 16 * (2 ^ 20 + 1) + 2 ^ 17) ^ 2 ≤ 2 ^ 74 := by norm_num
    have h16 : ((2 ^ 16 : ℕ) : ℝ≥0∞) = (2 : ℝ≥0∞) ^ 16 := by norm_num
    have h17 : ((2 ^ 17 : ℕ) : ℝ≥0∞) = (2 : ℝ≥0∞) ^ 17 := by norm_num
    calc _ ≤ ((2 ^ 16 : ℕ) : ℝ≥0∞) ^ 210 *
          (((2 ^ 17 : ℕ) : ℝ≥0∞) * 2 ^ 508614548 * ((2⁻¹ : ℝ≥0∞) ^ (8192 - 74)) ^ (62686 + 1)) := by
          gcongr
          · exact sum_pow_le _ _ (by norm_num)
          · exact sq_div_two_pow_le _ 74 8192 hsq (by norm_num)
      _ = (2 : ℝ≥0∞) ^ (16 * 210) *
          ((2 : ℝ≥0∞) ^ 17 * 2 ^ 508614548 * (2⁻¹ : ℝ≥0∞) ^ ((8192 - 74) * (62686 + 1))) := by
          rw [h16, h17, ← pow_mul, ← pow_mul]
      _ = (2⁻¹ : ℝ≥0∞) ^ ((8192 - 74) * (62686 + 1) - (16 * 210 + 17 + 508614548)) :=
          two_pow_combine _ _ _ _ (by norm_num)
      _ ≤ (2⁻¹ : ℝ≥0∞) ^ 128 :=
          pow_le_pow_right_of_le_one' (by norm_num) (by norm_num)

end PousAudit
