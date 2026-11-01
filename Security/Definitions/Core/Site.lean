import Mathlib.Data.List.Perm.Basic
import Mathlib.Data.List.Nodup

/-!
# The Site and its registration

The hardware the protocols run on is one **Site** (`Verity.Site`): the auditor's data sheets its devices name by digest,
a tree of locations each with its owner, and the links between them. The Site is public. A workload's placement on it
(each block `j`'s location `loc_j`), its inference blocks and its declared advice are private, and only their padded
counts are public.

* **Locations** (`Site.Loc`): the site at the root, its hosts, a host's devices, memory leaves by tier and bytes under
  a device or a host, and at the site the auditor's devices, its network certifiers and memory challengers
  (`Site.Kind.parents`). A host is registered hardware with no compute rate: the Site's compute is its devices'. A
  device names its SKU's data sheet by digest, and the auditor's devices carry their keys.
* **Sheets** (`Site.Sheet`): the auditor's data sheet for an SKU, in the device model's terms.
* **Links** (`Site.Link`): two ends, each a location or `outside`, the rest of the world, with a capacity `β` in bits a
  tick each way and a latency in ticks. A network certifier is the leaf that splits an uplink: one link from a host to
  it, one from it to the outside. The prover's boundary links are a host's links to the outside or to a certifier
  (`Site.Boundary`), and a link is certified when a certifier is at one of its ends (`Site.Certified`).

The Site is folded from an append-only log, **the registration** (`Verity.Registration`), whose entries each have a
kind (`Registration.Entry`): today the site entry (`Registration.Entry.site`), which adds sheets, locations and links
(`Site.Entry`); θ and the builder `B` are the registration's other kinds of entry. Each entry names the head of the log
it follows (`Registration.Entry.previous`): the digest of the log's last entry, which names its own predecessor, so the
head is a digest of the whole prefix. The Site of a log is its entries' additions in order (`Site.ofLog`), and a log is
valid when each entry is valid against the Site and head of the prefix before it (`Registration.Valid`). An entry is
checked as a whole (`Registration.Entry.Valid`, a site entry's own checks being `Site.Entry.Valid`), and refused by the
first check it fails (`Registration.Entry.refusal`):

| Refusal | The check |
| --- | --- |
| `previous` | the entry follows the log's head |
| `duplicateId` | no id is registered twice, `outside`'s included, and no sheet twice |
| `unknownRef` | each new location's parent and each new link's ends are registered |
| `kind` | each new location sits under a parent its kind allows and has a tier exactly when it is memory; each new link joins two different ends of kinds a link may join, at a positive capacity; each certifier splits exactly one uplink |
| `sheetDigest` | each new sheet is the auditor's data sheet for its SKU, and each new device names a registered sheet |
| `capacity` | each new memory leaf holds something, and each device's memory leaves fit in its sheet's memory |
| `owner` | each new certifier or challenger is one of the auditor's own devices with its key, and every other new location is the prover's |
| `uncertified` | onsite: every boundary link is certified |
| `challengerReach` | onsite: the host of every memory leaf is at an end of a memory challenger's link |

The digests are parameters (`Registration.Digests`), SHA-512 in the deployed check, and the auditor's data sheets and
keys are the check's own (`Registration.Auditor`). At every choice of them the check is decidable.
-/

namespace Verity

namespace Site

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

/-- The auditor's data sheet for an SKU, in the device model's terms (`Accounting.Device`): tensor-core multiply-adds a
tick (`tc`), bits from device memory to the chip a tick (`bw`), bits of on-chip state (`chip`) and bytes of device
memory (`mem`). -/
structure Sheet where
  sku : String
  tc : ℕ
  bw : ℕ
  chip : ℕ
  mem : ℕ
  deriving DecidableEq, Repr

/-- A location, and who owns it. A device names its SKU's sheet by digest, a memory leaf its tier and its bytes, and the
auditor's devices their keys; a field the location's kind doesn't use is ignored. -/
structure Loc where
  id : String
  kind : Kind
  parent : Option String := none
  owner : Owner := .prover
  sheet : String := ""
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

/-- The rest of the world: an end every Site has, whose id no location may take. -/
def outside : String := "outside"

/-- **A site entry**: the sheets, locations and links it adds to the Site. -/
structure Entry where
  sheets : List Sheet := []
  locs : List Loc := []
  links : List Link := []
  deriving DecidableEq, Repr

end Site

/-- **The Site**, public: its sheets, its tree of locations and its links. -/
structure Site where
  sheets : List Site.Sheet
  locs : List Site.Loc
  links : List Site.Link
  deriving DecidableEq, Repr

/-- **An entry of the registration**, by its kind, naming the head of the log it follows: the site entry, which adds
hardware to the Site. -/
inductive Registration.Entry
  | site (previous : String) (entry : Site.Entry)
  deriving DecidableEq, Repr

/-- **The registration**: the log the Site is folded from, oldest entry first. It is append-only: an entry is checked
against the log before it, and nothing after it changes it. -/
abbrev Registration : Type := List Registration.Entry

namespace Registration.Entry

/-- The head of the log the entry follows. -/
def previous : Entry → String
  | .site p _ => p

/-- What the entry adds to the Site: a site entry's sheets, locations and links. -/
def added : Entry → Site.Entry
  | .site _ s => s

end Registration.Entry

namespace Site

/-- The Site an entry's additions extend `S` to, unchecked. -/
def add (S : Site) (e : Entry) : Site :=
  ⟨S.sheets ++ e.sheets, S.locs ++ e.locs, S.links ++ e.links⟩

/-- **The Site of a log**: its entries' sheets, locations and links, in the log's order. The Site of a prefix of the log
is the site as registered when that prefix was the log. -/
def ofLog (log : Registration) : Site :=
  ⟨log.flatMap (·.added.sheets), log.flatMap (·.added.locs), log.flatMap (·.added.links)⟩

variable (S : Site)

/-- The location an id names: the first registered under it. -/
def loc? (i : String) : Option Loc := S.locs.find? (·.id == i)

/-- The kind of what an id names: the outside's, or its location's; `none` when nothing is registered under it. -/
def kind? (i : String) : Option Kind := if i = outside then some .outside else (S.loc? i).map (·.kind)

/-- An id names something: the outside, or a registered location. -/
def Known (i : String) : Prop := i = outside ∨ i ∈ S.locs.map (·.id)

/-- Every registered id: the locations', the links' and the outside's. -/
def ids : List String := S.locs.map (·.id) ++ S.links.map (·.id) ++ [outside]

/-- A device's sheet: the registered sheet under the digest it names. -/
def sheet? (digest : Sheet → String) (x : Loc) : Option Sheet := S.sheets.find? (digest · == x.sheet)

/-- The bytes of the memory leaves directly under `i`. -/
def held (i : String) : ℕ := ((S.locs.filter fun x => x.kind = .memory ∧ x.parent = some i).map (·.bytes)).sum

/-- Link `k` joins kinds `a` and `b`, in either order. -/
def Joins (k : Link) (a b : Kind) : Prop :=
  (S.kind? k.ends.1, S.kind? k.ends.2) = (some a, some b) ∨ (S.kind? k.ends.1, S.kind? k.ends.2) = (some b, some a)

/-- **A boundary link** of the prover's: a host's link to the outside, or to the certifier that splits its uplink. -/
def Boundary (k : Link) : Prop := S.Joins k .host .outside ∨ S.Joins k .certifier .host

/-- **A certified link**: a certifier is at one of its ends. -/
def Certified (k : Link) : Prop := S.kind? k.ends.1 = some .certifier ∨ S.kind? k.ends.2 = some .certifier

/-- The far ends of `i`'s links: for each link with an end at `i`, its other end. -/
def far (i : String) : List String :=
  (S.links.filter fun k => k.ends.1 = i ∨ k.ends.2 = i).flatMap fun k => [k.ends.1, k.ends.2].filter (· ≠ i)

/-- The host a memory leaf is in: its parent when that is a host, else its parent's parent (its device's host). -/
def host? (x : Loc) : Option String :=
  x.parent.bind fun p => if S.kind? p = some .host then some p else (S.loc? p).bind (·.parent)

/-- What the memory challengers reach: the ends of their links. -/
def reached : List String :=
  (S.links.filter fun k => S.kind? k.ends.1 = some .challenger ∨ S.kind? k.ends.2 = some .challenger).flatMap
    fun k => [k.ends.1, k.ends.2]

instance instDecidablePredStringKnown : DecidablePred S.Known := fun _ => by unfold Known; infer_instance

instance instDecidableJoins (k : Link) (a b : Kind) : Decidable (S.Joins k a b) := by unfold Joins; infer_instance

instance instDecidablePredLinkBoundary : DecidablePred S.Boundary := fun _ => by unfold Boundary; infer_instance

instance instDecidablePredLinkCertified : DecidablePred S.Certified := fun _ => by unfold Certified; infer_instance

end Site

namespace Registration

open Site (Sheet)

/-- The composite a site's log is registered for: onsite, whose boundary links the auditor's certifiers split and whose
memory its challengers reach, or remote, where neither need hold. -/
inductive Composite
  | onsite
  | remote
  deriving DecidableEq, Repr

/-- The digests the log names things by, SHA-512 in the deployed check: a sheet's, an entry's, and the head of a
composite's empty log. -/
structure Digests where
  sheet : Sheet → String
  entry : Entry → String
  genesis : Composite → String

/-- What the auditor brings to the check: its data sheet for each SKU, and the key of each of its own devices. -/
structure Auditor where
  sheet : String → Option Sheet
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
  | sheetDigest
  | capacity
  | owner
  | uncertified
  | challengerReach
  deriving DecidableEq, Repr

end Registration

namespace Site.Entry

open Registration (Digests Auditor Composite Refusal)

variable (D : Digests) (A : Auditor) (c : Composite) (S : Site) (e : Entry)

/-- `duplicateId`: no id is registered twice, the outside's among them, and no sheet twice. -/
def Fresh : Prop := (S.add e).ids.Nodup ∧ ((S.add e).sheets.map D.sheet).Nodup

/-- `unknownRef`: each new location's parent and each new link's ends name something registered. -/
def Refs : Prop :=
  (∀ x ∈ e.locs, ∀ p ∈ x.parent, (S.add e).Known p) ∧
    ∀ k ∈ e.links, (S.add e).Known k.ends.1 ∧ (S.add e).Known k.ends.2

/-- `kind`: each new location sits under a parent of a kind its own allows, and has a tier exactly when it is memory;
each new link joins two different ends, of kinds a link may join, at a positive capacity; and each certifier splits
exactly one uplink, the far ends of its links being one host and the outside. -/
def Kinds : Prop :=
  (∀ x ∈ e.locs, x.parent.bind (S.add e).kind? ∈ x.kind.parents ∧ (x.kind = .memory ↔ x.tier.isSome)) ∧
    (∀ k ∈ e.links, k.ends.1 ≠ k.ends.2 ∧ (∃ p ∈ Kind.links, (S.add e).Joins k p.1 p.2) ∧ 0 < k.β) ∧
    ∀ x ∈ (S.add e).locs, x.kind = .certifier →
      (((S.add e).far x.id).map (S.add e).kind?).Perm [some .host, some .outside]

/-- `sheetDigest`: each new sheet is the auditor's data sheet for its SKU, and each new device names a registered
sheet's digest. -/
def Sheets : Prop :=
  (∀ k ∈ e.sheets, A.sheet k.sku = some k) ∧ ∀ x ∈ e.locs, x.kind = .device → x.sheet ∈ (S.add e).sheets.map D.sheet

/-- `capacity`: each new memory leaf holds something, and each device's memory leaves fit in its sheet's memory. -/
def Capacity : Prop :=
  (∀ x ∈ e.locs, x.kind = .memory → 0 < x.bytes) ∧
    ∀ x ∈ (S.add e).locs, x.kind = .device → ∀ k ∈ (S.add e).sheet? D.sheet x, (S.add e).held x.id ≤ k.mem

/-- `owner`: each new certifier or challenger is one of the auditor's own devices, owned by it and with its key, and
every other new location is the prover's. -/
def Owners : Prop :=
  ∀ x ∈ e.locs, (x.kind.auditor = true → x.owner = .auditor ∧ A.key x.id = some x.key) ∧
    (x.kind.auditor = false → x.owner = .prover)

/-- `uncertified`, onsite: every boundary link is certified. -/
def Covered : Prop := c = .onsite → ∀ k ∈ (S.add e).links, (S.add e).Boundary k → (S.add e).Certified k

/-- `challengerReach`, onsite: the host of every memory leaf is at an end of a memory challenger's link. -/
def Reached : Prop :=
  c = .onsite → ∀ x ∈ (S.add e).locs, x.kind = .memory → ∀ h ∈ (S.add e).host? x, h ∈ (S.add e).reached

/-- **A valid site entry** of a log for the composite `c`, against the Site `S` of the log before it: it passes every
check of a site entry. -/
def Valid : Prop :=
  Fresh D S e ∧ Refs S e ∧ Kinds S e ∧ Sheets D A S e ∧ Capacity D S e ∧ Owners A e ∧ Covered c S e ∧ Reached c S e

instance instDecidableFresh : Decidable (Fresh D S e) := by unfold Fresh; infer_instance
instance instDecidableRefs : Decidable (Refs S e) := by unfold Refs; infer_instance
instance instDecidableKinds : Decidable (Kinds S e) := by unfold Kinds; infer_instance
instance instDecidableSheets : Decidable (Sheets D A S e) := by unfold Sheets; infer_instance
instance instDecidableCapacity : Decidable (Capacity D S e) := by unfold Capacity; infer_instance
instance instDecidableOwners : Decidable (Owners A e) := by unfold Owners; infer_instance
instance instDecidableCovered : Decidable (Covered c S e) := by unfold Covered; infer_instance
instance instDecidableReached : Decidable (Reached c S e) := by unfold Reached; infer_instance
instance instDecidableValid : Decidable (Valid D A c S e) := by unfold Valid; infer_instance

/-- A site entry's refusal: the first of its checks it fails, in `Valid`'s order, or `none` when it passes them all. -/
def refusal : Option Refusal :=
  if ¬Fresh D S e then some .duplicateId
  else if ¬Refs S e then some .unknownRef
  else if ¬Kinds S e then some .kind
  else if ¬Sheets D A S e then some .sheetDigest
  else if ¬Capacity D S e then some .capacity
  else if ¬Owners A e then some .owner
  else if ¬Covered c S e then some .uncertified
  else if ¬Reached c S e then some .challengerReach
  else none

end Site.Entry

namespace Registration

namespace Entry

variable (D : Digests) (A : Auditor) (c : Composite) (S : Site) (head : String)

/-- `previous`: the entry follows the log's head. -/
def Follows (e : Entry) : Prop := e.previous = head

/-- The checks of the entry's kind, against the Site `S` of the log before it: a site entry's (`Site.Entry.Valid`). -/
def Checks : Entry → Prop
  | .site _ s => s.Valid D A c S

/-- **A valid entry** of a log for the composite `c`, against the Site `S` of the log before it and its head `head`: it
follows the head, and passes its kind's checks. -/
def Valid (e : Entry) : Prop := Follows head e ∧ Checks D A c S e

/-- The refusal of the entry's kind: a site entry's (`Site.Entry.refusal`). -/
def kindRefusal : Entry → Option Refusal
  | .site _ s => s.refusal D A c S

instance instDecidableFollows (e : Entry) : Decidable (Follows head e) := by unfold Follows; infer_instance

instance instDecidableChecks : ∀ e : Entry, Decidable (Checks D A c S e)
  | .site _ s => inferInstanceAs (Decidable (s.Valid D A c S))

instance instDecidableValid (e : Entry) : Decidable (Valid D A c S head e) := by unfold Valid; infer_instance

/-- **The refusal**: `previous` when the entry doesn't follow the log's head, else its kind's, or `none` when it passes
every check. -/
def refusal (e : Entry) : Option Refusal :=
  if ¬Follows head e then some .previous else kindRefusal D A c S e

end Entry

/-- **A valid log** for the composite `c`: each entry is valid against the Site and head of the prefix before it. -/
def Valid (D : Digests) (A : Auditor) (c : Composite) (log : Registration) : Prop :=
  ∀ i (h : i < log.length), log[i].Valid D A c (Site.ofLog (log.take i)) (head D c (log.take i))

instance instDecidableValid (D : Digests) (A : Auditor) (c : Composite) (log : Registration) :
    Decidable (Valid D A c log) := by
  unfold Valid; infer_instance

end Registration

end Verity
