import Mathlib.Data.List.Perm.Basic
import Mathlib.Data.List.Nodup

/-!
# The physical layout and its registration

The site the protocols run on is one `Layout`: the cards its devices name by digest, a tree of locations each with its
owner, and the links between them.

* **Locations** (`Layout.Loc`): the site at the root, its hosts, a host's devices, memory leaves by tier and bytes under
  a device or a host, and at the site the auditor's devices, its network certifiers and memory challengers
  (`Layout.Kind.parents`). A device names its card's digest, and the auditor's devices carry their keys.
* **Cards** (`Layout.Card`): an SKU's data-sheet limits, in the device model's terms.
* **Links** (`Layout.Link`): two ends, each a location or `outside`, the rest of the world, with a capacity `β` in bits a
  tick each way and a latency in ticks. A network certifier is the leaf that splits an uplink: one link from a host to
  it, one from it to the outside. The prover's boundary links are a host's links to the outside or to a certifier
  (`Layout.Boundary`), and a link is certified when a certifier is at one of its ends (`Layout.Certified`).

The layout is registered as an append-only log (`Registration`). Each entry adds cards, locations and links, and names
the head of the log it follows (`Registration.Entry.previous`): the digest of the log's last entry, which names its own
predecessor, so the head is a digest of the whole prefix. The layout of a log is its entries' additions in order
(`Layout.ofLog`), and a log is valid when each entry is valid against the layout and head of the prefix before it
(`Registration.Valid`). An entry is checked as a whole (`Registration.Entry.Valid`), and refused by the first check it
fails (`Registration.Entry.refusal`):

| Refusal | The check |
| --- | --- |
| `previous` | the entry follows the log's head |
| `duplicateId` | no id is registered twice, `outside`'s included, and no card twice |
| `unknownRef` | each new location's parent and each new link's ends are registered |
| `kind` | each new location sits under a parent its kind allows and has a tier exactly when it is memory; each new link joins two different ends of kinds a link may join, at a positive capacity; each certifier splits exactly one uplink |
| `cardDigest` | each new card is the auditor's data sheet for its SKU, and each new device names a registered card |
| `capacity` | each new memory leaf holds something, and each device's memory leaves fit in its card's memory |
| `owner` | each new certifier or challenger is one of the auditor's own devices with its key, and every other new location is the prover's |
| `uncertified` | onsite: every boundary link is certified |
| `challengerReach` | onsite: the host of every memory leaf is at an end of a memory challenger's link |

The digests are parameters (`Registration.Digests`), SHA-512 in the deployed check, and the auditor's data sheets and
keys are the check's own (`Registration.Auditor`). At every choice of them the check is decidable.
-/

namespace Verity

namespace Layout

/-- What a location is. `outside`, the kind of the rest of the world, is no registered location's. -/
inductive Kind
  | site
  | host
  | device
  | memory
  | certifier
  | challenger
  | outside
  deriving DecidableEq, Repr

/-- The kinds of parent a location of each kind may have, `none` being the root: the site at the root, a host under the
site, a device under a host, memory under a device or a host, and the auditor's devices at the site. -/
def Kind.parents : Kind → List (Option Kind)
  | .site => [none]
  | .host => [some .site]
  | .device => [some .host]
  | .memory => [some .device, some .host]
  | .certifier | .challenger => [some .site]
  | .outside => []

/-- The pairs of kinds a link may join, in either order: two hosts, a host and the outside, a certifier and a host or
the outside, and a memory challenger and a host. -/
def Kind.links : List (Kind × Kind) :=
  [(.host, .host), (.host, .outside), (.certifier, .host), (.certifier, .outside), (.challenger, .host)]

/-- The auditor's devices: the network certifiers and the memory challengers. -/
def Kind.auditor : Kind → Bool
  | .certifier | .challenger => true
  | _ => false

/-- A memory leaf's tier. -/
inductive Tier
  | hbm
  | dram
  | disk
  deriving DecidableEq, Repr

/-- Who owns a location. -/
inductive Owner
  | prover
  | auditor
  deriving DecidableEq, Repr

/-- An SKU's data-sheet card, in the device model's terms (`Accounting.Device`): tensor-core multiply-adds a tick
(`tc`), bits from device memory to the chip a tick (`bw`), bits of on-chip state (`chip`) and bytes of device memory
(`mem`). -/
structure Card where
  sku : String
  tc : ℕ
  bw : ℕ
  chip : ℕ
  mem : ℕ
  deriving DecidableEq, Repr

/-- A location, and who owns it. A device names its card's digest, a memory leaf its tier and its bytes, and the
auditor's devices their keys; a field the location's kind doesn't use is ignored. -/
structure Loc where
  id : String
  kind : Kind
  parent : Option String := none
  owner : Owner := .prover
  card : String := ""
  tier : Option Tier := none
  bytes : ℕ := 0
  key : String := ""
  deriving DecidableEq, Repr

/-- A link between two ends, with its capacity `β` in bits a tick each way and its latency in ticks. -/
structure Link where
  id : String
  ends : String × String
  β : ℕ
  latency : ℕ
  deriving DecidableEq, Repr

/-- The rest of the world: an end every layout has, whose id no location may take. -/
def outside : String := "outside"

end Layout

/-- **The physical layout**: its cards, its tree of locations and its links. -/
structure Layout where
  cards : List Layout.Card
  locs : List Layout.Loc
  links : List Layout.Link
  deriving DecidableEq, Repr

/-- **An entry of the registration log**: the cards, locations and links it adds, and the head of the log it follows
(`previous`). -/
structure Registration.Entry where
  previous : String
  cards : List Layout.Card := []
  locs : List Layout.Loc := []
  links : List Layout.Link := []
  deriving DecidableEq, Repr

/-- **The registration log**: the layout's entries, oldest first. It is append-only: an entry is checked against the
log before it, and nothing after it changes it. -/
abbrev Registration : Type := List Registration.Entry

namespace Layout

/-- The layout an entry extends `L` to, unchecked. -/
def add (L : Layout) (e : Registration.Entry) : Layout :=
  ⟨L.cards ++ e.cards, L.locs ++ e.locs, L.links ++ e.links⟩

/-- **The layout of a log**: its entries' cards, locations and links, in the log's order. The layout of a prefix of the
log is the site as registered when that prefix was the log. -/
def ofLog (log : Registration) : Layout :=
  ⟨log.flatMap (·.cards), log.flatMap (·.locs), log.flatMap (·.links)⟩

variable (L : Layout)

/-- The location an id names: the first registered under it. -/
def loc? (i : String) : Option Loc := L.locs.find? (·.id == i)

/-- The kind of what an id names: the outside's, or its location's; `none` when nothing is registered under it. -/
def kind? (i : String) : Option Kind := if i = outside then some .outside else (L.loc? i).map (·.kind)

/-- An id names something: the outside, or a registered location. -/
def Known (i : String) : Prop := i = outside ∨ i ∈ L.locs.map (·.id)

/-- Every registered id: the locations', the links' and the outside's. -/
def ids : List String := L.locs.map (·.id) ++ L.links.map (·.id) ++ [outside]

/-- A device's card: the registered card under the digest it names. -/
def card? (digest : Card → String) (x : Loc) : Option Card := L.cards.find? (digest · == x.card)

/-- The bytes of the memory leaves directly under `i`. -/
def held (i : String) : ℕ := ((L.locs.filter fun x => x.kind = .memory ∧ x.parent = some i).map (·.bytes)).sum

/-- Link `k` joins kinds `a` and `b`, in either order. -/
def Joins (k : Link) (a b : Kind) : Prop :=
  (L.kind? k.ends.1, L.kind? k.ends.2) = (some a, some b) ∨ (L.kind? k.ends.1, L.kind? k.ends.2) = (some b, some a)

/-- **A boundary link** of the prover's: a host's link to the outside, or to the certifier that splits its uplink. -/
def Boundary (k : Link) : Prop := L.Joins k .host .outside ∨ L.Joins k .certifier .host

/-- **A certified link**: a certifier is at one of its ends. -/
def Certified (k : Link) : Prop := L.kind? k.ends.1 = some .certifier ∨ L.kind? k.ends.2 = some .certifier

/-- The far ends of `i`'s links: for each link with an end at `i`, its other end. -/
def far (i : String) : List String :=
  (L.links.filter fun k => k.ends.1 = i ∨ k.ends.2 = i).flatMap fun k => [k.ends.1, k.ends.2].filter (· ≠ i)

/-- The host a memory leaf is in: its parent when that is a host, else its parent's parent (its device's host). -/
def host? (x : Loc) : Option String :=
  x.parent.bind fun p => if L.kind? p = some .host then some p else (L.loc? p).bind (·.parent)

/-- What the memory challengers reach: the ends of their links. -/
def reached : List String :=
  (L.links.filter fun k => L.kind? k.ends.1 = some .challenger ∨ L.kind? k.ends.2 = some .challenger).flatMap
    fun k => [k.ends.1, k.ends.2]

instance : DecidablePred L.Known := fun _ => by unfold Known; infer_instance

instance (k : Link) (a b : Kind) : Decidable (L.Joins k a b) := by unfold Joins; infer_instance

instance : DecidablePred L.Boundary := fun _ => by unfold Boundary; infer_instance

instance : DecidablePred L.Certified := fun _ => by unfold Certified; infer_instance

end Layout

namespace Registration

open Layout

/-- The composite a site's log is registered for: onsite, whose boundary links the auditor's certifiers split and whose
memory its challengers reach, or remote, where neither need hold. -/
inductive Composite
  | onsite
  | remote
  deriving DecidableEq, Repr

/-- The digests the log names things by, SHA-512 in the deployed check: a card's, an entry's, and the head of a
composite's empty log. -/
structure Digests where
  card : Card → String
  entry : Entry → String
  genesis : Composite → String

/-- What the auditor brings to the check: its data sheet for each SKU, and the key of each of its own devices. -/
structure Auditor where
  sheet : String → Option Card
  key : String → Option String

/-- **The log's head**: the digest of its last entry, or of its composite's genesis when it is empty. -/
def head (D : Digests) (c : Composite) (log : Registration) : String :=
  match log.getLast? with
  | none => D.genesis c
  | some e => D.entry e

/-- A refusal, named by the check that fails. -/
inductive Refusal
  | previous
  | duplicateId
  | unknownRef
  | kind
  | cardDigest
  | capacity
  | owner
  | uncertified
  | challengerReach
  deriving DecidableEq, Repr

namespace Entry

variable (D : Digests) (A : Auditor) (c : Composite) (L : Layout) (head : String) (e : Entry)

/-- `previous`: the entry follows the log's head. -/
def Follows : Prop := e.previous = head

/-- `duplicateId`: no id is registered twice, the outside's among them, and no card twice. -/
def Fresh : Prop := (L.add e).ids.Nodup ∧ ((L.add e).cards.map D.card).Nodup

/-- `unknownRef`: each new location's parent and each new link's ends name something registered. -/
def Refs : Prop :=
  (∀ x ∈ e.locs, ∀ p ∈ x.parent, (L.add e).Known p) ∧
    ∀ k ∈ e.links, (L.add e).Known k.ends.1 ∧ (L.add e).Known k.ends.2

/-- `kind`: each new location sits under a parent of a kind its own allows, and has a tier exactly when it is memory;
each new link joins two different ends, of kinds a link may join, at a positive capacity; and each certifier splits
exactly one uplink, the far ends of its links being one host and the outside. -/
def Kinds : Prop :=
  (∀ x ∈ e.locs, x.parent.bind (L.add e).kind? ∈ x.kind.parents ∧ (x.kind = .memory ↔ x.tier.isSome)) ∧
    (∀ k ∈ e.links, k.ends.1 ≠ k.ends.2 ∧ (∃ p ∈ Kind.links, (L.add e).Joins k p.1 p.2) ∧ 0 < k.β) ∧
    ∀ x ∈ (L.add e).locs, x.kind = .certifier →
      (((L.add e).far x.id).map (L.add e).kind?).Perm [some .host, some .outside]

/-- `cardDigest`: each new card is the auditor's data sheet for its SKU, and each new device names a registered card's
digest. -/
def Cards : Prop :=
  (∀ k ∈ e.cards, A.sheet k.sku = some k) ∧ ∀ x ∈ e.locs, x.kind = .device → x.card ∈ (L.add e).cards.map D.card

/-- `capacity`: each new memory leaf holds something, and each device's memory leaves fit in its card's memory. -/
def Capacity : Prop :=
  (∀ x ∈ e.locs, x.kind = .memory → 0 < x.bytes) ∧
    ∀ x ∈ (L.add e).locs, x.kind = .device → ∀ k ∈ (L.add e).card? D.card x, (L.add e).held x.id ≤ k.mem

/-- `owner`: each new certifier or challenger is one of the auditor's own devices, owned by it and with its key, and
every other new location is the prover's. -/
def Owners : Prop :=
  ∀ x ∈ e.locs, (x.kind.auditor = true → x.owner = .auditor ∧ A.key x.id = some x.key) ∧
    (x.kind.auditor = false → x.owner = .prover)

/-- `uncertified`, onsite: every boundary link is certified. -/
def Covered : Prop := c = .onsite → ∀ k ∈ (L.add e).links, (L.add e).Boundary k → (L.add e).Certified k

/-- `challengerReach`, onsite: the host of every memory leaf is at an end of a memory challenger's link. -/
def Reached : Prop :=
  c = .onsite → ∀ x ∈ (L.add e).locs, x.kind = .memory → ∀ h ∈ (L.add e).host? x, h ∈ (L.add e).reached

/-- **A valid entry** of a log for the composite `c`, against the layout `L` of the log before it and its head `head`:
it passes every check. -/
def Valid : Prop :=
  Follows head e ∧ Fresh D L e ∧ Refs L e ∧ Kinds L e ∧ Cards D A L e ∧ Capacity D L e ∧ Owners A e ∧
    Covered c L e ∧ Reached c L e

instance : Decidable (Follows head e) := by unfold Follows; infer_instance
instance : Decidable (Fresh D L e) := by unfold Fresh; infer_instance
instance : Decidable (Refs L e) := by unfold Refs; infer_instance
instance : Decidable (Kinds L e) := by unfold Kinds; infer_instance
instance : Decidable (Cards D A L e) := by unfold Cards; infer_instance
instance : Decidable (Capacity D L e) := by unfold Capacity; infer_instance
instance : Decidable (Owners A e) := by unfold Owners; infer_instance
instance : Decidable (Covered c L e) := by unfold Covered; infer_instance
instance : Decidable (Reached c L e) := by unfold Reached; infer_instance
instance : Decidable (Valid D A c L head e) := by unfold Valid; infer_instance

/-- **The refusal**: the first check the entry fails, in `Valid`'s order, or `none` when it passes them all. -/
def refusal : Option Refusal :=
  if ¬Follows head e then some .previous
  else if ¬Fresh D L e then some .duplicateId
  else if ¬Refs L e then some .unknownRef
  else if ¬Kinds L e then some .kind
  else if ¬Cards D A L e then some .cardDigest
  else if ¬Capacity D L e then some .capacity
  else if ¬Owners A e then some .owner
  else if ¬Covered c L e then some .uncertified
  else if ¬Reached c L e then some .challengerReach
  else none

end Entry

/-- **A valid log** for the composite `c`: each entry is valid against the layout and head of the prefix before it. -/
def Valid (D : Digests) (A : Auditor) (c : Composite) (log : Registration) : Prop :=
  ∀ i (h : i < log.length), log[i].Valid D A c (Layout.ofLog (log.take i)) (head D c (log.take i))

instance (D : Digests) (A : Auditor) (c : Composite) (log : Registration) : Decidable (Valid D A c log) := by
  unfold Valid; infer_instance

end Registration

end Verity
