module

public import Definitions.Core.Site
public import Mathlib.Data.Finset.Dedup
public import Mathlib.Data.Fintype.Sets
public import Mathlib.Logic.Relation

/-!
# A subtree of the Site, its places and its links as a table

A **subtree** of the Site is a registered location with every location registered under it (`Site.under`); at the
Site's root it is the datacenter. A **place** is a registered location or the outside (`Site.places`): a link ends at
a place, and a gate of the actual circuit is placed at one.

The hardware model (`Accounting.Placed`, `Accounting.linkCapacity`) takes its leaves as a type and its links as a
table of ends and capacities. Here the Site is read as that: its places are the leaves (`Site.Place`, `Site.place`,
`Site.placesIn`) and its links the table, each at its position in the Site (`Site.placeEnds`, `Site.linkβ`), so a
bound of the model at these is a bound read off the registration.

* `Site.Child`: the tree's parent relation, which the Site holds as each location's `parent`.
* `Site.places`, `Site.under`: the places, and the subtree at a place.
* `Site.Place`, `Site.place`, `Site.placesIn`: the places as a type, the place an id names, and the places whose id
  is in a set.
* `Site.placeEnds`, `Site.linkβ`: the registered links as the model's.
-/

@[expose] public section

namespace Verity.Site

variable (S : Site)

/-- `j` is registered directly under `i`: a location of id `j` names `i` as its parent. -/
def Child (i j : String) : Prop := ∃ x ∈ S.locs, x.id = j ∧ x.parent = some i

instance instDecidableChild (i j : String) : Decidable (S.Child i j) := by unfold Child; infer_instance

/-- **The places** of the Site: every registered location and the outside, the ids the Site knows (`Known`) as a
finite set. A link ends at a place, and a gate of the actual circuit is placed at one: a host is a place though
locations are registered under it. -/
def places : Finset String := insert outside (S.locs.map (·.id)).toFinset

open Classical in
/-- **The subtree at `i`**: the places at or under `i` in the tree of locations. At the Site's root it is the
datacenter. -/
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

end Verity.Site
