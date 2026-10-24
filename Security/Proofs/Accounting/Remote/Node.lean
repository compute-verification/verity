import Definitions.Accounting.Remote.Node

/-! The proof-only part of `Definitions.Accounting.Remote.Node`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Accounting.Remote

variable {τ : Type} [Fintype τ]

/-- A fleet audited at once, or an NVLink domain, as one node: the members' tiers, chips and rebuild work together, with
the boundary links of the whole (`linkIn`, `linkOut`); links between members are inside it. -/
def Node.fleet {ι : Type} {τ : ι → Type} [Fintype ι] (Ns : ∀ i, Node (τ i)) (linkIn linkOut : ℕ) :
    Node (Σ i, τ i) where
  ceil m := (Ns m.1).ceil m.2
  rate m := (Ns m.1).rate m.2
  chip := ∑ i, (Ns i).chip
  linkIn := linkIn
  linkOut := linkOut
  rebuild := ∑ i, (Ns i).rebuild
  rate_le m := (Ns m.1).rate_le m.2

end Accounting.Remote
