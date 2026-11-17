module

public import Definitions.Core.Site
public import Mathlib.Data.Finset.Dedup
public import Mathlib.Data.Fintype.Sets
public import Mathlib.Logic.Relation

/-! A subtree of the Site, its places as the hardware model's leaves and its links as the model's table. -/

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

/-- The registered links as the model's: link `e`'s capacity each way, in bits a tick, and `none` where the Site
registers no rate for it. -/
def linkβ (e : Fin S.links.length) : Option ℕ := (S.links.get e).β

/-- **`β` is the registered rate of each link in `E`**: every link of `E` is registered with a rate, and it is `β e`.
A statement that sums the links' capacities takes it at the links it sums: a link with no rate is one the Site
registers no capacity for, so no limit is stated across it, and none is read as `0`. -/
def RatedOn (β : Fin S.links.length → ℕ) (E : Set (Fin S.links.length)) : Prop :=
  ∀ e ∈ E, S.linkβ e = some (β e)

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

open Classical in
noncomputable def proverUnder (i : String) : Finset String :=
  (S.under i).filter fun j => ∀ k ∈ S.kind? j, k.auditor = false

open Classical in
/-- The registered certifiers at the far end of a link of one of the subtree's prover places. On any Site, a far end
of `x` leaves `x` itself out. -/
noncomputable def certifiersOf (i : String) : Finset String :=
  S.places.filter fun y => S.kind? y = some .certifier ∧ ∃ x ∈ S.proverUnder i, y ∈ S.far x

open Classical in
/-- Only challengers among `pinned`, the ids whose key and session the auditor's pins hold, count. On any Site, a far
end of `x` leaves `x` itself out. -/
noncomputable def challengersOf (pinned : Finset String) (i : String) : Finset String :=
  S.places.filter fun y => S.kind? y = some .challenger ∧ y ∈ pinned ∧ ∃ x ∈ S.proverUnder i, y ∈ S.far x

noncomputable def auditorsOf (pinned : Finset String) (i : String) : Finset String :=
  S.certifiersOf i ∪ S.challengersOf pinned i

/-- `y` is linked only into the subtree at `i`. A link from `y` to `y` gives `y` no far end and so is not asked about. -/
def Within (i y : String) : Prop := ∀ z ∈ S.far y, z ∈ S.under i

open Classical in
noncomputable def sharedOf (pinned : Finset String) (i : String) : Finset String :=
  (S.challengersOf pinned i).filter fun y => ¬ S.Within i y

end Verity.Site
