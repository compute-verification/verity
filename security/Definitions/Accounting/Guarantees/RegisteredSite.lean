module

public import Definitions.Accounting.Physical
public import Definitions.Core.SiteSubtree

/-!
# A subtree's egress limit, read off the registered links (trusted)

Stated in the actual-circuit model at the Site's own places and links; not proved here.

* `EgressRegistered`: what leaves a subtree of the Site over any ticks is at most the capacity of the registered links
  with exactly one end in the subtree, a tick; and that limit is at most the limit of the subtree's root alone plus
  its children's subtrees' limits. The limit is the model's `β_X` (`linkCapacity`) at the Site's links
  (`Site.placeEnds`, `Site.linkβ`) and the subtree's places (`Site.under`), so the boundary has one definition.
-/

@[expose] public section

namespace Accounting.Guarantees

/-- **A subtree's egress limit, read off the registered links.** For a Site `S` and the subtree at `i`
(`Site.under`), with the Site's places as the model's leaves and its links as the model's table:

* **The bound.** For an actual circuit placed on the Site's places whose every wire between two places runs on a
  registered link within its capacity (`Placed.Links` at the Site's links, the one physical premise), the subtree's
  egress during `[a, b)` (`Placed.egress`: the width of its gates that a gate outside it reads in those ticks) is at
  most `β (b − a)`, with `β` the summed capacity of the registered links with exactly one end in the subtree
  (`linkCapacity`). A link between two places of the subtree is not counted. On a Site with no registered link the
  premise holds only of a circuit with no wire between places.
* **Limits add over parts, at most.** For a place `i`, the subtree's limit is at most the limit of `i` alone plus the
  limits of the subtrees of the locations registered directly under `i` (`Site.Child`). It is an inequality: a link
  between two of those parts is counted on the right and not on the left.

The limit is read off the Site: it has no parameter but the Site and the place. -/
def EgressRegistered : Prop :=
  ∀ (S : Verity.Site) (i : String),
    (∀ (A : Placed S.Place) (link : Fin A.N → Fin A.N → Fin S.links.length),
      A.Links S.placeEnds S.linkβ link → ∀ a b : ℕ,
        A.egress (S.placesIn (S.under i)) (Set.Ico a b) ≤
          linkCapacity S.placeEnds S.linkβ (S.placesIn (S.under i)) * (b - a)) ∧
    (i ∈ S.places →
      linkCapacity S.placeEnds S.linkβ (S.placesIn (S.under i)) ≤
        linkCapacity S.placeEnds S.linkβ (S.placesIn {i}) +
          ∑ c ∈ S.places.filter (S.Child i), linkCapacity S.placeEnds S.linkβ (S.placesIn (S.under c)))

end Accounting.Guarantees
