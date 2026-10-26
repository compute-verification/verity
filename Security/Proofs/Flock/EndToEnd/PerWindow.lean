import Proofs.Flock.EndToEnd.PerWindow.Statement

/-! `Flock.SecurityProofs.HiddenWindowPerWindowStated : Flock.Guarantees.HiddenWindowPerWindowStated`, by reflexivity:
the guarantee is there for its record, which holds `HiddenWindowPerWindow`'s text. `HiddenWindowPerWindow` is not
proved yet. -/

/-- **Sampled proofs, window by window, is stated** (`Flock.Guarantees.HiddenWindowPerWindowStated`). -/
theorem Flock.SecurityProofs.HiddenWindowPerWindowStated : Flock.Guarantees.HiddenWindowPerWindowStated := rfl
