import Proofs.Hash.RowBinding

/-!
# `Hash.SecurityProofs.EmbeddingReadBinding : Hash.Guarantees.EmbeddingReadBinding`

For a token in the vocabulary the read's ok is the witness's acceptance at rank `tok` and its row is the witness's, so
`RowBinding` at rank `tok` against the registered opening is the statement.
-/

open Hash in
/-- **The embedding's row read binds its row** (`Hash.Guarantees.EmbeddingReadBinding`). -/
theorem Hash.SecurityProofs.EmbeddingReadBinding : Hash.Guarantees.EmbeddingReadBinding := by
  intro H G domain key root vocab words tok o hok o₀ hin h₀
  have ho : RowLeaf.accepts H G (EmbeddingRead.verifier domain key) root tok o :=
    Or.resolve_left (show vocab ≤ tok ∨ _ from hok) (Nat.not_le.mpr hin)
  rw [EmbeddingRead.h, if_pos hin]
  exact Hash.SecurityProofs.RowBinding H G _ root tok o o₀ ho h₀
