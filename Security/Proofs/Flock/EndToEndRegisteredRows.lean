import Proofs.Flock.EndToEndRegisteredRows.Statement

/-! `Flock.SecurityProofs.EndToEndRegisteredRowsStated : Flock.Guarantees.EndToEndRegisteredRowsStated`, by
reflexivity: the guarantee is there for its record, which holds `Flock.Guarantees.EndToEndRegisteredRows`'s text until
its proof, `Flock.SecurityProofs.EndToEndRegisteredRows`, lands here. -/

/-- **The rows link is stated** (`Flock.Guarantees.EndToEndRegisteredRowsStated`). -/
theorem Flock.SecurityProofs.EndToEndRegisteredRowsStated : Flock.Guarantees.EndToEndRegisteredRowsStated := rfl
