module

public import Proofs.Flock.SpotFlock.AddressFacts.PortsOnceOf

/-!
# The width and the copy, of every accepted run

`WidthFact` and `CopyFact` speak of accepted runs only, and a run is accepted only where the second stage's design
evaluates (`CheckedRun.design_of_ok`). So the design is no hypothesis of either: the width at the address map holds
where every accepted run's held code has the wiring (`widthFact_of_runs`), and the copy where the row holds there too
(`copyFact_of_runs`). The wiring is asked at each accepted run, so a proof of it may read the run's design the same
way.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable {h : Held}

/-- **An accepted run's design evaluates**, at the run's own statement. -/
theorem CheckedRun.design_of_ok {r : CheckedRun h.stage} (hr : r.Ok h) : ∃ d, two h r.s = some d := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hrest⟩ := hr
  exact ⟨d, h2⟩

/-- **The width at the address map**, where every accepted run's held code has the wiring. -/
theorem widthFact_of_runs (hW : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h) : WidthFact h (addressOf h) :=
  fun r hr => by
    obtain ⟨d, h2⟩ := CheckedRun.design_of_ok hr
    exact widthFact_of_design h2 (hW r hr) r hr

/-- **The copy at the address map**, where every accepted run's held code has the wiring and each counted gate's
row is the address map's. -/
theorem copyFact_of_runs {ω : ℕ → ℕ} {keep : ℕ → Bool} (hW : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h)
    (hrow : RowFact h ω keep (addressOf h)) : CopyFact h (addressOf h) :=
  fun r hr => by
    obtain ⟨d, h2⟩ := CheckedRun.design_of_ok hr
    exact copyFact_of_design h2 (hW r hr) hrow r hr

end SpotFlock
