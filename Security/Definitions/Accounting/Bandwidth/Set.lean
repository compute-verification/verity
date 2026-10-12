import Mathlib.Logic.Equiv.Fin.Basic
import Definitions.Accounting.Bandwidth.Game

/-!
# The bandwidth audit v0 over a co-audited set

`d` devices run simultaneous windows over pairwise-distinct stores of `N` blocks each, and the verifier samples `s`
blocks of each store. Every device that could hold a copy of a store is in the set or isolated, so the set's side
sources are only its external links.

The set is one `Device` over its members' stores side by side, `d·N` blocks with member `i`'s block `j` at
`member i j`. Its on-chip state is the members' on-chip states together, its device memory theirs together, its side
sources the set's external links, and its chip program everything the members compute in the window, the traffic on
the links between members included. Those links are free. A member that computes another member's tags reads those
blocks from device memory, its own or the other member's, and the set's ingress counts the read: the set's ingress is
the members' ingress summed, and nothing bounds one member's ingress alone.

The stores are pairwise distinct: each store's salt is drawn from its GPU's UUID, and each window's key binds its
store. So no two members share a tag: member `i`'s block `j` is its own point `(member i j, E (member i j))` of the
oracle. Each member's sample is drawn after the set's output, independently of it and of the other members' samples.
-/

namespace Bandwidth

/-- Member `i`'s block `j` among a co-audited set's `d·N` blocks. -/
def member {d N : ℕ} (i : Fin d) (j : Fin N) : Fin (d * N) := finProdFinEquiv (i, j)

/-- The verifier accepts every member of a co-audited set: each member's sampled blocks carry the oracle's tags. -/
def SetAccepts {d N b h S c G q s : ℕ} (A : Device (d * N) b h S c G q) (E : Store (d * N) b)
    (O : Oracle (d * N) b h) (I : Fin d → Fin s → Fin N) : Prop :=
  ∀ i, Accepts A E O (fun t => member i (I i t))

end Bandwidth
