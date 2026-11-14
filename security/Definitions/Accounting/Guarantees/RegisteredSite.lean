module

public import Definitions.Accounting.Physical
public import Definitions.Core.SiteSubtree

/-!
# A subtree's egress limit, read off the registered links (trusted)

Stated in the actual-circuit model at the Site's own places and links; not proved here.

* `Placed.onto`: an actual circuit's image under a map of leaves, as `Annotation.map` is an annotation's.
* `EgressRegistered`: what leaves a subtree of the Site over any ticks is at most the capacity of the registered links
  with exactly one end in the subtree, a tick. The limit is the model's `β_X` (`linkCapacity`) at the Site's links
  (`Site.placeEnds`, `Site.linkβ`) and the subtree's places (`Site.under`), so the boundary has one definition.
-/

@[expose] public section

namespace Accounting

/-- **An actual circuit's image under a map of leaves**: the same circuit at the same ticks, each gate at the leaf
`f` sends its own to. Each gate takes a cell of its own: two leaves that `f` sends to one may each hold a gate in the
same cell at the same tick, and a cell takes one gate a tick. Wires, widths, ticks, and so `egress` and `Links`, read
no cell. -/
def Placed.onto {L L' : Type*} (A : Placed L) (f : L → L') : Placed L' :=
  { A.toCircuit with
    x := fun g => f (A.x g)
    t := A.t
    cell := fun g => g.val
    cell_inj := fun g h e => by
      simp only [Prod.mk.injEq] at e
      exact Fin.ext e.2.1 }

end Accounting

namespace Accounting.Guarantees

/-- **A subtree's egress limit, read off the registered links.** For a Site `S` and the subtree at `i`
(`Site.under`), with the Site's places as the model's leaves and its links as the model's table, and an actual
circuit `A` placed on the Site's places, a device or a memory leaf as much as a host:

* **The wires run on the registered links** (`Placed.Links` at the Site's links, the one physical premise). It is
  taken of `A`'s image under `Site.linkPlace`, which sends each device and each memory leaf to its host (the Site's
  own `host?`) and every other place to itself, since a registered link ends at a host, a certifier, a challenger or
  the outside: a wire between two places of one host needs no link, and every other wire between two places runs on
  a registered link between the places a link may end at for them, within the link's capacity.
* **The subtree holds each host whole** (`Site.HostsWhole`), so that what leaves it crosses a link. A subtree at a
  device or a memory leaf does not, and the statement says nothing of it.
* **The bound.** The subtree's egress during `[a, b)` (`Placed.egress`, of `A` as placed: the width of its gates that
  a gate outside it reads in those ticks) is at most `β (b − a)`, with `β` the summed capacity of the registered
  links with exactly one end in the subtree (`linkCapacity`). A link between two places of the subtree is not
  counted. On a Site with no registered link the bound is stated of the empty circuit only.

The limit is read off the Site: it has no parameter but the Site and the place. Where a network certifier stands in
front of a host without splitting its interface, the registration holds the host's own links to the outside and,
beside them, the host's link to the certifier and the certifier's to the outside, each at the interfaces' rates
summed. The limit of that host's subtree, and of its site's, then counts the interfaces' rates twice, and the bound
is loose there by up to a factor of two. -/
def EgressRegistered : Prop :=
  ∀ (S : Verity.Site) (i : String) (A : Placed S.Place) (link : Fin A.N → Fin A.N → Fin S.links.length),
    (A.onto S.linkPlace).Links S.placeEnds S.linkβ link → S.HostsWhole i → ∀ a b : ℕ,
      A.egress (S.placesIn (S.under i)) (Set.Ico a b) ≤
        linkCapacity S.placeEnds S.linkβ (S.placesIn (S.under i)) * (b - a)

end Accounting.Guarantees
