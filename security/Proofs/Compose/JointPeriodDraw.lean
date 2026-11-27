module

public import Definitions.Compose.JointPeriod
public import Proofs.Core.Game.Draw

/-! An audit's per-period guarantee is soundness (`Verity.Sound`) at the run of its own coin. -/

@[expose] public section

namespace ResourceAccounting.PeriodRun

open Verity

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}
  [DecidableEq I] [∀ i, Fintype (Ω i)] [∀ i, Nonempty (Ω i)] (r : PeriodRun d α C Ω)

/-- `PeriodRun.Sound` at an error of at least 0 is, at each value of the other audits' coins, soundness at the run
in which the audit's own coin is drawn. -/
theorem sound_iff_draw (c : Certificates) (p : Price) (a : Audit) {δ : ℝ} (hδ : 0 ≤ δ) :
    r.Sound c p a δ ↔ ∀ ξ : (i : I) → Ω i, Verity.Sound (Game.draw (Ω (r.own a)))
      (fun x => r.accepts a (Function.update ξ (r.own a) x))
      (fun x => r.Holds c p a (Function.update ξ (r.own a) x)) (ENNReal.ofReal δ) :=
  forall_congr' fun _ => (sound_draw_iff_pr hδ).symm

end ResourceAccounting.PeriodRun
