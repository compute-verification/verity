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
most `q` oracle calls has, for every budget `b ≥ 0`, an adversary within `(b, q)` that commits its transcript and hashes
its inputs wherever its realized online cost is at most `b`. -/
theorem Adversary.abort {CM : CostModel Q R S} (hCM : BudgetAbort CM) (A : Adversary CM) (U : Workload) {q : ℕ}
    (hq : CallsWithin A U q) {b : ℝ} (hb : 0 ≤ b) :
    ∃ A' : Adversary CM, A'.Bounded U b q ∧
      ∀ H s, onlineCost A U H s ≤ b →
        A'.transcript U H s = A.transcript U H s ∧ A'.hashed U H s = A.hashed U H s := by
  obtain ⟨prog', h⟩ := hCM.2 A.online U A.act b hb
  refine ⟨⟨A.pre, prog', A.act⟩, fun H s => ⟨(h H _ s).1, ?_⟩, fun H s hc => (h H _ s).2.2 hc⟩
  exact (Nat.add_le_add_left (h H _ s).2.1 _).trans (hq H s)

namespace SecurityProofs

theorem PouwEndToEndPerRun : Accounting.Guarantees.PouwEndToEndPerRun := by
  intro Q R S _ _ _ _ CM P D γ₀ Ωmax ε hγ hΩ hWref hTT hCM U hU q hq A hA t εs hεs
  refine le_trans (pr_mono fun x hx => ⟨⟨hx.1, hx.2.2⟩, hx.2.1⟩) (pr_lt_of_abort
    (fun (A' : Adversary CM) (x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N)) =>
      AuditAccepts (correctSet P U x.1 x.2.1 (A'.transcript U x.1 x.2.1)) x.2.2 ∧ ¬ A'.Breaks U x.1 x.2.1)
    (fun A' T => A'.Bounded U T q) A (fun x => onlineCost A U x.1 x.2.1)
    ((1 - reductionGamma γ₀ Ωmax) * (1 - εs) * totalWref P U.layout) ((1 - εs) ^ t + ε q U.layout.N)
    (fun x => hCM.1 _ _ _ _ _ _)
    (fun A' T hT hA' => le_trans (pr_mono fun x hx => ⟨hx.1.1, hx.2, hx.1.2⟩)
      (Pouw.SecurityProofs.EndToEnd Q R S CM P D γ₀ Ωmax ε hγ hΩ hWref hTT U hU T hT q hq A' hA' t εs hεs))
    fun b hb => ?_)
  obtain ⟨A', hA', htr⟩ := Adversary.abort hCM A U hA hb
  refine ⟨A', hA', fun x hx hacc => ?_⟩
  obtain ⟨h1, h2⟩ := htr x.1 x.2.1 hx
  refine ⟨h1.symm ▸ hacc.1, ?_⟩
  show ¬ Hash.CollisionIn CM.sha (A'.hashed U x.1 x.2.1)
  rw [h2]
  exact hacc.2

theorem ComputeOfPouwPerRun : Accounting.Guarantees.ComputeOfPouwPerRun := by
  intro Q R S _ _ _ _ CM P D γ₀ Ωmax ε hγ hΩ hWref hTT hCM U hU t εs hεs hne ι α ΩQ ΩS _ _ _ d r p hq real o
  obtain ⟨⟨_, _, tix⟩⟩ := hne
  have hA : CallsWithin (real.adv o) U (d.rate .oracle * (r.win .compute).len) := fun H s =>
    (real.calls_le (H, s, tix) o).trans (Schedule.ops_le (r.fits _) _ _ _)
  refine le_trans (pr_mono fun x hx => ?_)
    (PouwEndToEndPerRun Q R S CM P D γ₀ Ωmax ε hγ hΩ hWref hTT hCM U hU _ hq (real.adv o) hA t εs hεs)
  exact ⟨(real.accepts_iff x o).1 hx.1, (real.cost_le x o).trans_lt hx.2.1, hx.2.2⟩

end SecurityProofs

end Accounting
