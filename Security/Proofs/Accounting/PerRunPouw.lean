import Definitions.Accounting.Guarantees.PerRun
import Proofs.Accounting.Device
import Proofs.Accounting.PerRunTruncation
import Proofs.Pouw.Game

/-!
# PoUW's guarantee per run, by truncation

PoUW's adversary class is closed under budget-abort when its cost model truncates (`Accounting.BudgetAbort`): the
adversary that keeps `A`'s preprocessing and activation input and runs the cost model's aborted program in place of
`A`'s online program is within `(b, q)` whenever `A` makes at most `q` oracle calls, and commits `A`'s transcript
wherever `A` costs at most `b` (`Accounting.Adversary.abort`). Truncation (`Accounting.pr_lt_of_abort`) then turns
`Pouw.SecurityProofs.EndToEnd` into its per-run form, and a run that realizes PoUW's game has the compute audit's
per-run guarantee, because the compute account's work covers the adversary's cost at every coin.
-/

namespace Accounting

open Pouw

variable {Q R S : Type}

/-- **PoUW's adversaries are closed under budget-abort.** When the cost model truncates, every adversary that makes at
most `q` oracle calls has, for every budget `b ≥ 0`, an adversary within `(b, q)` that commits its transcript wherever
its realized online cost is at most `b`. -/
theorem Adversary.abort {CM : CostModel Q R S} (hCM : BudgetAbort CM) (A : Adversary CM) (U : Workload) {q : ℕ}
    (hq : CallsWithin A U q) {b : ℝ} (hb : 0 ≤ b) :
    ∃ A' : Adversary CM, A'.Bounded U b q ∧
      ∀ H s, onlineCost A U H s ≤ b → A'.transcript U H s = A.transcript U H s := by
  obtain ⟨prog', h⟩ := hCM.2 A.online U A.act b hb
  refine ⟨⟨A.pre, prog', A.act⟩, fun H s => ⟨(h H _ s).1, ?_⟩, fun H s hc => (h H _ s).2.2 hc⟩
  exact (Nat.add_le_add_left (h H _ s).2.1 _).trans (hq H s)

/-- A cost model whose programs' costs and calls do not depend on the oracle, the preprocessed state or the salt, with
nonnegative costs and a program that costs nothing and calls nothing, truncates: the aborted program is the program
itself when its cost is within the budget, and the null program otherwise. So `BudgetAbort` holds of straight-line
cost models, and the per-run guarantees' extra hypothesis is satisfiable. -/
theorem budgetAbort_of_oblivious (CM : CostModel Q R S)
    (hnn : ∀ prog H U act σ s, 0 ≤ CM.cost prog H U act σ s)
    (hobl : ∀ prog U act, ∃ c : ℝ, ∃ n : ℕ, ∀ H σ s, CM.cost prog H U act σ s = c ∧ CM.calls prog H U act σ s = n)
    (null : CM.Prog) (hnull : ∀ H U act σ s, CM.cost null H U act σ s = 0 ∧ CM.calls null H U act σ s = 0) :
    BudgetAbort CM := by
  classical
  refine ⟨hnn, fun prog U act b hb => ?_⟩
  obtain ⟨c, n, hc⟩ := hobl prog U act
  by_cases hcb : c ≤ b
  · exact ⟨prog, fun H σ s => ⟨(hc H σ s).1 ▸ hcb, le_rfl, fun _ => rfl⟩⟩
  · refine ⟨null, fun H σ s => ⟨(hnull H U act σ s).1 ▸ hb, (hnull H U act σ s).2 ▸ Nat.zero_le _, fun h => ?_⟩⟩
    exact absurd ((hc H σ s).1 ▸ h) hcb

/-- The per-run form is at least as strong as `EndToEnd` at a cost model that truncates: an adversary within `(T, q)`
makes at most `q` calls, and its realized cost is at most `T` on every run, so `T` below the certificate puts every run
below it. -/
theorem endToEnd_of_perRun [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S}
    {P : Protocol Q R S} {U : Workload} {T c e : ℝ} {q t : ℕ} {A : Adversary CM} (hA : A.Bounded U T q)
    (h : CallsWithin A U q →
      pr (fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
          AuditAccepts (correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧ onlineCost A U x.1 x.2.1 < c)
        ≤ e) :
    pr (fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
        AuditAccepts (correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧ T < c) ≤ e :=
  (pr_mono fun x hx => ⟨hx.1, ((hA x.1 x.2.1).1.trans_lt hx.2)⟩).trans (h fun H s => (hA H s).2)

namespace SecurityProofs

theorem PouwEndToEndPerRun : Accounting.Guarantees.PouwEndToEndPerRun := by
  intro Q R S _ _ _ _ CM P D γ₀ Ωmax ε hγ hΩ hWref hTT hCM U hU q hq A hA t εs hεs
  refine pr_lt_of_abort
    (fun (A' : Adversary CM) (x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N)) =>
      AuditAccepts (correctSet P U x.1 x.2.1 (A'.transcript U x.1 x.2.1)) x.2.2)
    (fun A' T => A'.Bounded U T q) A (fun x => onlineCost A U x.1 x.2.1) _ _ (fun x => hCM.1 _ _ _ _ _ _)
    (fun A' T hT hA' =>
      Pouw.SecurityProofs.EndToEnd Q R S CM P D γ₀ Ωmax ε hγ hΩ hWref hTT U hU T hT q hq A' hA' t εs hεs)
    fun b hb => ?_
  obtain ⟨A', hA', htr⟩ := Adversary.abort hCM A U hA hb
  exact ⟨A', hA', fun x hx hacc => (htr x.1 x.2.1 hx).symm ▸ hacc⟩

theorem ComputeOfPouwPerRun : Accounting.Guarantees.ComputeOfPouwPerRun := by
  intro Q R S _ _ _ _ CM P D γ₀ Ωmax ε hγ hΩ hWref hTT hCM U hU t εs hεs hne ι α ΩQ ΩS _ _ _ d r p hq real o
  obtain ⟨⟨_, _, tix⟩⟩ := hne
  have hA : CallsWithin (real.adv o) U (d.rate .oracle * (r.win .compute).len) := fun H s =>
    (real.calls_le (H, s, tix) o).trans (Schedule.ops_le (r.fits _) _ _ _)
  refine le_trans (pr_mono fun x hx => ?_)
    (PouwEndToEndPerRun Q R S CM P D γ₀ Ωmax ε hγ hΩ hWref hTT hCM U hU _ hq (real.adv o) hA t εs hεs)
  exact ⟨(real.accepts_iff x o).1 hx.1, (real.cost_le x o).trans_lt hx.2⟩

end SecurityProofs

end Accounting
