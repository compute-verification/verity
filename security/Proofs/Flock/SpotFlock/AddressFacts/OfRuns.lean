module

public import Proofs.Flock.SpotFlock.AddressFacts.PortsOnceOf

/-! The width and the copy at the address map hold of every accepted run, whose design evaluates, so the design is no
hypothesis of either (`widthFact_of_runs`, `copyFact_of_runs`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable {h : Held}

theorem CheckedRun.design_of_ok {r : CheckedRun h.stage} (hr : r.Ok h) : ∃ d, two h r.s = some d := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hrest⟩ := hr
  exact ⟨d, h2⟩

theorem widthFact_of_runs (hW : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h) : WidthFact h (addressOf h) :=
  fun r hr => by
    obtain ⟨d, h2⟩ := CheckedRun.design_of_ok hr
    exact widthFact_of_design h2 (hW r hr) r hr

theorem copyFact_of_runs {ω : ℕ → ℕ} {keep : ℕ → Bool} (hW : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h)
    (hrow : RowFact h ω keep (addressOf h)) : CopyFact h (addressOf h) :=
  fun r hr => by
    obtain ⟨d, h2⟩ := CheckedRun.design_of_ok hr
    exact copyFact_of_design h2 (hW r hr) hrow r hr

end SpotFlock
