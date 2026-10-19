import Models.Hash.Guarantees

/-! `Hash.SecurityProofs.Stated : Hash.Guarantees.Stated`, by reflexivity: the guarantee is there for its record, which
holds the hash area's statements' text. -/

/-- **The hash area's statements are stated** (`Hash.Guarantees.Stated`). -/
theorem Hash.SecurityProofs.Stated : Hash.Guarantees.Stated := rfl
