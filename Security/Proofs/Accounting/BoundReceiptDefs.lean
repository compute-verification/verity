import Definitions.Accounting.BoundReceipt

/-! The proof-only part of `Definitions.Accounting.BoundReceipt`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Accounting.BoundReceipt

open Pous (Bits)

open Accounting.Rate (Store Oracle Chip World Device)

variable {N a b h : ℕ}

namespace InteractiveNode

variable {S L q Lz qz : ℕ}

/-- A node whose locations answer each other as they do at every mix (`e = 0`). -/
def ofNode (A : Node N a b h S L q Lz qz) : InteractiveNode N a b h S L q Lz qz :=
  ⟨A, fun z _ => (A.gpu z).side, fun E _ => (A.rest E).side⟩

end InteractiveNode

end Accounting.BoundReceipt
