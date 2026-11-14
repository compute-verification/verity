module

public import Definitions.Core.Site
public import Mathlib.Data.Finset.Dedup
public import Mathlib.Data.Fintype.Sets
public import Mathlib.Logic.Relation

/-!
# A subtree of the Site, its places and its links as a table

A **subtree** of the Site is a registered location with every location registered under it (`Site.under`). At a site
location it is that site's registered hardware, the auditor's devices at it among them. It is the definitions' twin
of the code's `Core.Site.within`, whose locations are those of `Core.Site.part`. A **place** is a registered location
or the outside (`Site.places`).

The hardware model (`Accounting.Placed`, `Accounting.linkCapacity`) takes its leaves as a type and its links as a
table of ends and capacities. Here the Site is read as that: its places are the model's leaves (`Site.Place`,
`Site.place`, `Site.placesIn`), so a gate of the actual circuit is placed at one, a device or a memory leaf as much as
a host; and its links are the table, each at its position in the Site (`Site.placeEnds`, `Site.linkβ`), so a bound of
the model at these is a bound read off the registration.

A registered link ends at a host, a certifier, a challenger or the outside, never at a device, a memory leaf or a site
(`Site.Kind.links`). So a wire between two places is held to the links through the place a link may end at for each
(`Site.linkEnd`): its host for a device and a memory leaf, and each other place for itself. A subtree that holds
each host with all its locations or not at all (`Site.HostsWhole`) is one whose egress the links bound.

* `Site.Child`: the tree's parent relation, which the Site holds as each location's `parent`.
* `Site.places`, `Site.under`: the places, and the subtree at a place.
* `Site.Place`, `Site.place`, `Site.placesIn`: the places as a type, the place an id names, and the places whose id
  is in a set.
* `Site.placeEnds`, `Site.linkβ`: the registered links as the model's.
* `Site.linkEnd`, `Site.linkPlace`: the place a link may end at for a place, read through the Site's own `host?`.
* `Site.HostsWhole`: a subtree holds each host whole.
-/

@[expose] public section

namespace Verity.Site

variable (S : Site)

/-- `j` is registered directly under `i`: a location of id `j` names `i` as its parent. -/
def Child (i j : String) : Prop := ∃ x ∈ S.locs, x.id = j ∧ x.parent = some i

instance instDecidableChild (i j : String) : Decidable (S.Child i j) := by unfold Child; infer_instance

/-- **The places** of the Site: every registered location and the outside, the ids the Site knows (`Known`) as a
finite set. A gate of the actual circuit is placed at one, whatever its kind. -/
def places : Finset String := insert outside (S.locs.map (·.id)).toFinset

open Classical in
/-- **The subtree at `i`**: the places at or under `i` in the tree of locations. At a site location it is that site's
registered hardware, the auditor's devices at it among them. The definitions' twin of the code's `Core.Site.within`,
which follows parents upward from a location, a bounded number of them; this is the same relation from `i` downward,
as a set. -/
noncomputable def under (i : String) : Finset String :=
  S.places.filter fun j => Relation.ReflTransGen S.Child i j

/-- A place of the Site as a leaf of the hardware model, whose leaves are a type. -/
abbrev Place : Type := {j : String // j ∈ S.places}

/-- The place an id names; the outside where it names none. -/
def place (i : String) : S.Place :=
  if h : i ∈ S.places then ⟨i, h⟩ else ⟨outside, Finset.mem_insert_self outside _⟩

/-- The places whose id is in `X`: a set of ids as a location of the hardware model. -/
def placesIn (X : Finset String) : Finset S.Place := Finset.univ.filter fun p => p.1 ∈ X

/-- The registered links as the model's, each at its position in the Site: link `e`'s two ends, as places. -/
def placeEnds (e : Fin S.links.length) : S.Place × S.Place :=
  (S.place (S.links.get e).ends.1, S.place (S.links.get e).ends.2)

/-- The registered links as the model's: link `e`'s capacity each way, in bits a tick. -/
def linkβ (e : Fin S.links.length) : ℕ := (S.links.get e).β

/-- **The place a link may end at** for the place `i`: the host the location `i` is in, as the Site reads it off the
location's parent (`host?`: the parent when that is a host, else the parent's parent), and `i` itself where that
names none or `i` names no location. On a valid registration's Site that is its host for a device and for a memory
leaf, and itself for a host, a site, a certifier, a challenger and the outside. -/
def linkEnd (i : String) : String := ((S.loc? i).bind S.host?).getD i

/-- The place a link may end at for a leaf of the model. -/
def linkPlace (p : S.Place) : S.Place := S.place (S.linkEnd p.1)

/-- **The subtree at `i` holds each host whole**: a place is in it exactly when the place a link may end at for it
is. A subtree at a device or at a memory leaf does not: its host is outside it, and no registered link says what
passes between the two. -/
def HostsWhole (i : String) : Prop := ∀ p : S.Place, p.1 ∈ S.under i ↔ (S.linkPlace p).1 ∈ S.under i

end Verity.Site
