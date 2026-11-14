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
* `Site.proverUnder`, `Site.certifiersOf`, `Site.challengersOf`, `Site.auditorsOf`: a subtree's prover places, and the
  auditor's devices at it, read off the links' far ends (`Site.far`).
* `Site.Within`, `Site.sharedOf`: a place linked only into a subtree, and the memory challengers of a subtree that are
  not.

A far end of a place's links leaves the place itself out, on any Site. A valid log's Site has no link from a place to
itself, and there the far ends of a place are the other ends of the registered links that end at it.
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

open Classical in
/-- **The prover's places of the subtree at `i`**: the subtree less the auditor's devices (`Kind.auditor`). -/
noncomputable def proverUnder (i : String) : Finset String :=
  (S.under i).filter fun j => ∀ k ∈ S.kind? j, k.auditor = false

open Classical in
/-- **The certifiers of the subtree at `i`**: the registered certifiers at the far end (`far`) of a link of one of the
subtree's prover places. On any Site, valid or not, a far end of `x` is the other end of a registered link that ends
at `x`, `x` itself left out. Where no link joins a place to itself, as on a valid log's Site, that is every registered
certifier a registered link joins to one of those places. -/
noncomputable def certifiersOf (i : String) : Finset String :=
  S.places.filter fun y => S.kind? y = some .certifier ∧ ∃ x ∈ S.proverUnder i, y ∈ S.far x

open Classical in
/-- **The memory challengers of the subtree at `i`**, under the pins `pinned`: the registered memory challengers among
`pinned` at the far end (`far`) of a link of one of the subtree's prover places. `pinned` holds the ids of the
challengers whose key and session the auditor's pins hold, a parameter. A challenger that is not pinned, and a device
at a link's end that is not registered as a challenger, is not among them. On any Site, valid or not, a far end of `x`
is the other end of a registered link that ends at `x`, `x` itself left out. Where no link joins a place to itself, as
on a valid log's Site, that is every pinned registered challenger a registered link joins to one of those places. -/
noncomputable def challengersOf (pinned : Finset String) (i : String) : Finset String :=
  S.places.filter fun y => S.kind? y = some .challenger ∧ y ∈ pinned ∧ ∃ x ∈ S.proverUnder i, y ∈ S.far x

/-- **The auditor's devices at the subtree at `i`**: its certifiers and its memory challengers. -/
noncomputable def auditorsOf (pinned : Finset String) (i : String) : Finset String :=
  S.certifiersOf i ∪ S.challengersOf pinned i

/-- **`y` is linked only into the subtree at `i`**: the far end (`far`) of each of `y`'s links is in the subtree. On
any Site, valid or not, a link from `y` to `y` gives `y` no far end and so is not asked about. Where no link joins a
place to itself, as on a valid log's Site, it is every registered link that ends at `y` having its other end in the
subtree. -/
def Within (i y : String) : Prop := ∀ z ∈ S.far y, z ∈ S.under i

open Classical in
/-- **The shared memory challengers of the subtree at `i`**: its memory challengers with a link whose far end is
outside the subtree. On the Site's link kinds (`Kind.links`) that is a challenger linked to a host of the subtree and to
a host outside it. -/
noncomputable def sharedOf (pinned : Finset String) (i : String) : Finset String :=
  (S.challengersOf pinned i).filter fun y => ¬ S.Within i y

end Verity.Site
