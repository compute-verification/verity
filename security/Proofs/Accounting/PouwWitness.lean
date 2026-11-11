module

public import Definitions.Accounting.PouwWitness
public import Proofs.Pouw.Witness
import Proofs.Pouw.Prob

/-!
# PoUW's compute guarantees say something (witnesses)

`Accounting.Guarantees.ComputeOfPouw` and `Accounting.Guarantees.ComputeOfPouwPerRun`:

* honest (one lemma for both): the milestone's sixteen units are in `D1.m1Domain`, and the output-priced model, where
  TT holds and the honest reference runs at `≤ W_ref` (`Pouw.Proofs.ttWitness`), truncates (`Accounting.BudgetAbort`):
  the copy runs the program and commits its transcript when its committed units' GEMMs fit the budget, and blanks
  otherwise (`PouwWitness.truncated`);
* nontrivial: on a device with no capacity and an idle schedule, a run whose every audit accepts does no work, so its
  compute audit accepts short of the certificate `W = 1` with probability 1, above the error 0
  (`Accounting.PouwWitness.acceptAllRun`).
-/

@[expose] public section

namespace Accounting.SecurityProofs

open Accounting.PouwWitness

namespace PouwWitness

variable {Q R S : Type}

open Classical in
/-- The output-priced cost of the transcript `τ` on workload `U`: its committed units' GEMMs. -/
noncomputable def pricedCost (U : Pouw.Workload) (τ : Pouw.Transcript) : ℝ :=
  ∑ u ∈ (Finset.range U.layout.N).filter (fun u => (τ u).isSome), ((U.layout.shape u).Wmm : ℝ)

/-- The program `prog`, committing blanks when its transcript costs more than `b`. -/
noncomputable def truncated (prog : (Pouw.outputPriced Q R S).Prog) (b : ℝ) : (Pouw.outputPriced Q R S).Prog :=
  fun U act σ s => (prog U act σ s).bind fun τ =>
    Pouw.QTree.done (if pricedCost U τ ≤ b then τ else fun _ => none)

theorem outputPriced_cost (prog : (Pouw.outputPriced Q R S).Prog) (H : Q → R) (U : Pouw.Workload)
    (act : ℕ → Pouw.Mat) (σ : Pouw.State) (s : S) :
    (Pouw.outputPriced Q R S).cost prog H U act σ s = pricedCost U ((prog U act σ s).eval H) :=
  rfl

theorem eval_truncated (prog : (Pouw.outputPriced Q R S).Prog) (b : ℝ) (H : Q → R) (U : Pouw.Workload)
    (act : ℕ → Pouw.Mat) (σ : Pouw.State) (s : S) :
    (truncated prog b U act σ s).eval H =
      if pricedCost U ((prog U act σ s).eval H) ≤ b then (prog U act σ s).eval H else fun _ => none := by
  show ((prog U act σ s).bind fun τ =>
    Pouw.QTree.done (if pricedCost U τ ≤ b then τ else fun _ => none)).eval H = _
  rw [Pouw.QTree.eval_bind]
  rfl

theorem calls_truncated (prog : (Pouw.outputPriced Q R S).Prog) (b : ℝ) (H : Q → R) (U : Pouw.Workload)
    (act : ℕ → Pouw.Mat) (σ : Pouw.State) (s : S) :
    (truncated prog b U act σ s).calls H = (prog U act σ s).calls H := by
  show ((prog U act σ s).bind fun τ =>
    Pouw.QTree.done (if pricedCost U τ ≤ b then τ else fun _ => none)).calls H = _
  rw [Pouw.QTree.calls_bind]
  rfl

/-- **The output-priced model truncates.** -/
theorem outputPriced_budgetAbort : Accounting.BudgetAbort (Pouw.outputPriced Q R S) := by
  refine ⟨fun prog H U act σ s => ?_, fun prog U act b hb => ⟨truncated prog b, fun H σ s => ?_⟩⟩
  · rw [outputPriced_cost]
    exact Finset.sum_nonneg fun _ _ => Nat.cast_nonneg _
  · rw [outputPriced_cost, outputPriced_cost, eval_truncated]
    refine ⟨?_, (calls_truncated prog b H U act σ s).le, fun hc => ⟨?_, rfl⟩⟩
    · split_ifs with h
      · exact h
      · simp [pricedCost, hb]
    · show (truncated prog b U act σ s).eval H = (prog U act σ s).eval H
      rw [eval_truncated, if_pos hc]

theorem acceptAllRun_work (p : Price) (ω : Unit × Unit × Unit) : acceptAllRun.work p ω = 0 := by
  simp [Run.work, Schedule.work, Schedule.ops, acceptAllRun, idleStep]

end PouwWitness

open PouwWitness

/-- **`ComputeOfPouwPerRun`'s and `ComputeOfPouw`'s premises hold at the milestone, in the output-priced model**: the
model truncates, and the sixteen units are in the domain. -/
theorem ComputeOfPouwPerRun_honest :
    Accounting.BudgetAbort (Pouw.outputPriced Pouw.D1.M1Q Pouw.D1.M1R Pouw.Salt) ∧
      Pouw.Proofs.M1.sixteen.InDomain Pouw.D1.m1Domain :=
  ⟨outputPriced_budgetAbort, Pouw.Proofs.M1.sixteen_inDomain⟩

/-- **`ComputeOfPouw`'s conclusion fails where every audit accepts**: a run that does no work passes the compute audit
short of `W = 1` with probability 1. -/
theorem ComputeOfPouw_nontrivial : ¬ acceptAllRun.ComputeSoundUnless (fun _ => 1) 1 0 (fun _ _ => False) := by
  intro h
  have h1 := h ((), ()) 0 le_rfl fun _ => by simp [acceptAllRun_work]
  rw [Pouw.pr_eq_one] at h1
  · norm_num at h1
  · intro _
    exact ⟨trivial, one_pos, id⟩

/-- **`ComputeOfPouwPerRun`'s conclusion fails where every audit accepts**: a run that does no work passes the compute
audit short of `W = 1` with probability 1. -/
theorem ComputeOfPouwPerRun_nontrivial :
    ¬ acceptAllRun.ComputeSoundPerRunUnless (fun _ => 1) 1 0 (fun _ _ => False) := by
  intro h
  have h1 := h ((), ())
  rw [Pouw.pr_eq_one] at h1
  · norm_num at h1
  · intro x
    exact ⟨trivial, by simp [acceptAllRun_work], id⟩

end Accounting.SecurityProofs
