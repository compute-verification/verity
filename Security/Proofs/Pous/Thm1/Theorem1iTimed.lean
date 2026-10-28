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

/-!
# Report Theorem 1(i), timed: `PousProofs.theorem1_i_timed`, as drafted

A compressor to `S` bits with a per-block `(D, Q)`-bounded expander is a sequential cheater with an `S`-bit
state. `A₁` is the compressor, and each round runs the expander on the challenged index and keeps the state
(`progMap` pairs the block with the unchanged state, at the same cost). When the expander rebuilds every block,
every challenge sequence is answered correctly (`seqAnswers_keep`), so
`Pr_setup[rebuild] ≤ Pr[audit accepts] ≤ δ + ε` (`pr_le_outcome`).
-/

namespace PousThm1

open Pous
open scoped ENNReal

/-! ### Uniform probability -/

/-- `pr` does not depend on how the sample space is labelled. -/
theorem pr_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (P : β → Prop) :
    pr (fun a => P (e a)) = pr P := by
  classical
  rw [pr_def, pr_def, Fintype.card_congr e]
  congr 2
  rw [← Fintype.card_subtype, ← Fintype.card_subtype]
  exact Fintype.card_congr (e.subtypeEquiv fun _ => Iff.rfl)

/-- An event of the first coordinate. -/
theorem pr_fst {α β : Type} [Fintype α] [Fintype β] [Nonempty β] (P : α → Prop) :
    pr (fun x : α × β => P x.1) = pr P := by
  rw [← pr_snd (α := β) P]
  exact pr_equiv (Equiv.prodComm α β) (fun x : β × α => P x.2)

/-- An event of the setup, seen on the whole outcome `Ω × Coins × I`. -/
theorem pr_le_outcome {α β γ : Type} [Fintype α] [Fintype β] [Fintype γ] [Nonempty γ]
    (E : α × β → Prop) (F : α × β × γ → Prop) (h : ∀ x : α × β × γ, E (x.1, x.2.1) → F x) : pr E ≤ pr F := by
  calc pr E = pr (fun y : (α × β) × γ => E y.1) := (pr_fst E).symm
    _ = pr (fun x : α × β × γ => E (x.1, x.2.1)) :=
        (pr_equiv (Equiv.prodAssoc α β γ).symm (fun y : (α × β) × γ => E y.1)).symm
    _ ≤ pr F := pr_mono h

/-! ### Programs -/

/-- Post-process a program's output; the queries are unchanged. -/
def progMap {M : IdealModel} {α β : Type} (g : α → β) : OracleProg M α → OracleProg M β
  | .ret a => .ret (g a)
  | .round qs k => .round qs fun as => progMap g (k as)

section progMap

variable {M : IdealModel} {α β : Type} (g : α → β) (f : M.Qry → M.Ans)

theorem run_progMap (p : OracleProg M α) : (progMap g p).run f = g (p.run f) := by
  induction p with
  | ret a => rfl
  | round qs k ih => exact ih _

theorem depth_progMap (p : OracleProg M α) : (progMap g p).depth f = p.depth f := by
  induction p with
  | ret a => rfl
  | round qs k ih => exact congrArg (· + 1) (ih _)

theorem work_progMap (p : OracleProg M α) : (progMap g p).work f = p.work f := by
  induction p with
  | ret a => rfl
  | round qs k ih => exact congrArg (qs.length + ·) (ih _)

theorem bounded_progMap {D Q : ℕ} {p : OracleProg M α} (h : p.Bounded D Q) : (progMap g p).Bounded D Q :=
  fun ω => by rw [depth_progMap, work_progMap]; exact h ω

end progMap

/-! ### Sequential responders -/

/-- A responder that keeps its state answers `is` by running `p σ` on each index. -/
theorem seqAnswers_keep {M : IdealModel} {ℓ S B : ℕ} (f : M.Qry → M.Ans) (p : Bits S → Fin B → OracleProg M (Bits ℓ)) :
    ∀ (j : ℕ) (is : List (Fin B)) (σ : Bits S),
      seqAnswers f (fun _ σ i => progMap (fun b => (b, σ)) (p σ i)) j is σ = is.map fun i => (p σ i).run f
  | _, [], _ => rfl
  | j, i :: is, σ => by
      simp only [seqAnswers, run_progMap, List.map_cons, seqAnswers_keep f p (j + 1) is σ]

end PousThm1

namespace PousProofs

open Pous PousThm1
open scoped ENNReal

/-- Report Theorem 1(i), timed: a sequential audit bound gives `TimedINC` at the state size. -/
theorem theorem1_i_timed {M : IdealModel} (sch : Scheme M) (S D Q k : ℕ) (δ ε : ℝ≥0∞)
    (h : AuditSecure sch .sequential S D Q k δ ε) : TimedINC sch D Q S (δ + ε) := by
  intro W cmp exp hexp
  let A₂ : SeqResponder sch S D Q :=
    ⟨fun W pp _ σ i => progMap (fun b => (b, σ)) (exp W pp σ i),
      fun W pp _ σ i => bounded_progMap _ (hexp W pp σ i)⟩
  refine le_trans (pr_le_outcome _ (seqAccepts sch k W cmp A₂) fun x hx => ?_) (h W cmp A₂)
  show seqAnswers _ (fun j σ i => progMap (fun b => (b, σ)) (exp W _ σ i)) 0 _ _ = _
  rw [seqAnswers_keep, List.map_ofFn]
  exact congrArg List.ofFn (funext fun j => hx _)

end PousProofs
