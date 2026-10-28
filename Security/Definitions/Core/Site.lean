import Definitions.Accounting.Device
import Definitions.Core.Compiler
import Definitions.Core.DeviceFigures
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Data.List.Perm.Basic
import Mathlib.Data.List.Nodup
import Mathlib.Data.NNRat.Floor

/-!
# The Site and its registration

The hardware the protocols run on is one **Site** (`Verity.Site`): the sheets its devices name by digest, a tree of
locations each with its owner, and the links between them. The Site is public. A workload's placement on it
(each block `j`'s location `loc_j`), its inference blocks and its declared advice are private, and only their padded
counts are public.

* **Locations** (`Site.Loc`): the site at the root, its hosts, a host's devices, memory leaves by tier and bytes under
  a device or a host, and at the site the auditor's devices, its network certifiers and memory challengers
  (`Site.Kind.parents`). A host is registered hardware with no compute rate: the Site's compute is its devices'. A
  device names its sheet by digest, and the auditor's devices carry their keys.
* **Sheets** (`Site.Sheet`): a card, the figures the prover states for an SKU: its compute figures (`DeviceFigures`: the
  SKU, the tensor clock and the peak rate of each operand class) and its memory figures. A device's figures are its
  sheet's, and its device model (`Accounting.Device`) is read off them (`Site.device`).
* **Links** (`Site.Link`): two ends, each a location or `outside`, the rest of the world, with a capacity `β` in bits a
  tick each way and a latency in ticks. A network certifier is the leaf that splits an uplink: one link from a host to
  it, one from it to the outside. The prover's boundary links are a host's links to the outside or to a certifier
  (`Site.Boundary`), and a link is certified when a certifier is at one of its ends (`Site.Certified`).

The Site is folded from an append-only log, **the registration** (`Verity.Registration`), whose entries each have a
kind (`Registration.Entry`): the site entry (`Registration.Entry.site`), which adds sheets, locations and links
(`Site.Entry`), and the information entry (`Registration.Entry.information`), which registers items
(`Registration.Item`): θ's values, each by its committed root (`Registration.Theta`), and the builder `B`, the registered
compiler (`Registration.Builder`, holding `Verity.Compiler`). The log's θ and builder are what its items register
(`Registration.theta`, `Registration.builder?`), and the registered circuit reads both from it
(`Verity.RegisteredCircuit`). Each entry names the head of the log it follows (`Registration.Entry.previous`): the
digest of the log's last entry, which names its own predecessor, so the head is a digest of the whole prefix. The Site
of a log is its entries' additions in order (`Site.ofLog`), and a log is valid when each entry is valid against the
Site and head of the prefix before it and registers no item the prefix registers (`Registration.Valid`). An entry is
checked as a whole (`Registration.Entry.Valid`, a site entry's own checks being `Site.Entry.Valid`; an information
entry adds no hardware, so it passes them), and refused by the first check it fails (`Registration.Entry.refusal`):

| Refusal | The check |
| --- | --- |
| `previous` | the entry follows the log's head |
| `duplicateId` | no id is registered twice, `outside`'s included, and no sheet twice |
| `unknownRef` | each new location's parent and each new link's ends are registered |
| `kind` | each new location sits under a parent its kind allows and has a tier exactly when it is memory; each new link joins two different ends of kinds a link may join, at a positive capacity; each certifier splits exactly one uplink |
| `sheetDigest` | each new sheet's SKU is one the auditor has a data sheet for, and each new device names a registered sheet |
| `figures` | each new sheet's figures are within the auditor's tolerance of its estimate for the SKU (`DeviceFigures.Accepted`), and its memory figures are the auditor's |
| `capacity` | each new memory leaf holds something, and each device's memory leaves fit in its sheet's memory |
| `owner` | each new certifier or challenger is one of the auditor's own devices with its key, and every other new location is the prover's |
| `uncertified` | onsite: every boundary link is certified |
| `challengerReach` | onsite: the host of every memory leaf is at an end of a memory challenger's link |

What an entry registers is held to the log before it (`Registration.Entry.Once`, the code's `duplicate-information`):
no information item is registered twice.

The digests are parameters (`Registration.Digests`), SHA-512 in the deployed check. The auditor's data sheets, public
and looked up by SKU, its tolerance around their figures and its keys are the check's own (`Registration.Auditor`). At
every choice of them the check is decidable.
-/

namespace Verity

open scoped NNRat

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

/-- Two figures are equal when their SKUs, clocks and peak rates are. -/
instance _root_.Verity.DeviceFigures.instDecidableEq : DecidableEq DeviceFigures := fun a b =>
  decidable_of_iff (a.sku = b.sku ∧ a.clock = b.clock ∧ a.peakRate = b.peakRate) <| by
    cases a; cases b; simp

/-- A sheet, a card: the figures the prover states for an SKU. Its compute figures (`figures`): the SKU, the tensor clock
in Hz and the peak tensor-core MACs a second of each operand class. Its memory figures, in the device model's terms
(`Accounting.Device`): bits from device memory to the chip a tick (`bw`), bits of on-chip state (`chip`) and bytes of
device memory (`mem`). -/
structure Sheet where
  figures : DeviceFigures
  bw : ℕ
  chip : ℕ
  mem : ℕ
  deriving DecidableEq

/-- A sheet's SKU, its figures'. -/
def Sheet.sku (k : Sheet) : String := k.figures.sku

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
  deriving DecidableEq

end Site

/-- **The Site**, public: its sheets, its tree of locations and its links. -/
structure Site where
  sheets : List Site.Sheet
  locs : List Site.Loc
  links : List Site.Link
  deriving DecidableEq

/-- **The builder `B`**, as the information entry registers it: its digest, the format of the advice it compiles, and
the registered compiler it names (`Verity.Compiler`), whose advice cap `ℓa` is the builder's public cap. -/
structure Registration.Builder where
  digest : String
  advice : String
  B : Compiler

/-- **An item of an information entry**: a registered value of θ, or the builder. -/
inductive Registration.Item
  | theta (t : Registration.Theta)
  | builder (b : Registration.Builder)

/-- What an item registers, which the log registers once: each θ value by its name, and one builder. -/
def Registration.Item.key : Registration.Item → String
  | .theta t => "theta " ++ t.value
  | .builder _ => "builder"

/-- **An entry of the registration**, by its kind, naming the head of the log it follows: the site entry, which adds
hardware to the Site, and the information entry, which registers θ's values and the builder. -/
inductive Registration.Entry
  | site (previous : String) (entry : Site.Entry)
  | information (previous : String) (items : List Registration.Item)

/-- **The registration**: the log the Site is folded from, oldest entry first. It is append-only: an entry is checked
against the log before it, and nothing after it changes it. -/
abbrev Registration : Type := List Registration.Entry

namespace Registration.Entry

/-- The head of the log the entry follows. -/
def previous : Entry → String
  | .site p _ => p
  | .information p _ => p

/-- What the entry adds to the Site: a site entry's sheets, locations and links, and nothing for an information
entry. -/
def added : Entry → Site.Entry
  | .site _ s => s
  | .information _ _ => {}

/-- What the entry registers besides hardware: an information entry's items. -/
def items : Entry → List Item
  | .site _ _ => []
  | .information _ is => is

end Registration.Entry

namespace Registration

/-- The items the log registers, in the log's order. -/
def items (log : Registration) : List Item := log.flatMap Entry.items

/-- **θ**, the values the log registers. -/
def theta (log : Registration) : List Theta := log.items.filterMap fun | .theta t => some t | .builder _ => none

/-- **The builder** the log registers, which names the registered compiler. -/
def builder? (log : Registration) : Option Builder := log.items.findSome? fun | .builder b => some b | .theta _ => none

end Registration

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

/-- What the auditor brings to the check: its data sheet for each SKU, public, whose figures are its estimate of the SKU's
(`none` for an SKU it has no sheet for); the tolerance registration allows around an estimate; and the key of each of its
own devices. -/
structure Auditor where
  sheet : String → Option Sheet
  tolerance : FigureTolerance
  key : String → Option String

/-- **The auditor's estimate** its sheet `a` states: `a`'s figures, at the auditor's tolerance. -/
def Auditor.estimate (A : Auditor) (a : Sheet) : DeviceEstimate := ⟨a.figures, A.tolerance⟩

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
  | figures
  | capacity
  | owner
  | uncertified
  | challengerReach
  deriving DecidableEq, Repr

end Registration

namespace Site

open Registration (Auditor)

/-- **A device's model** (`Accounting.Device`), read off the Site. For a device `x` naming a registered sheet `k` whose
SKU the auditor has a sheet `a` for: its on-chip state and memory bandwidth are `k`'s, its device memory is `k`'s in
bits, and its tensor-core MACs a tick are the highest peak rate of the figures the bound reads
(`DeviceFigures.bounding`: class by class, the larger of `k`'s and `a`'s estimate's) over a tick of `tick` seconds,
rounded up. No sheet states a device's CUDA-core ops or oracle queries a tick, or its side channels, so they are
parameters, as in `Accel.Card.machine`. `none` for anything else. -/
def device (digest : Sheet → String) (A : Auditor) (S : Site) (x : Loc) (tick : ℚ≥0) {ι : Type} (cuda oracle : ℕ)
    (cap : ι → ℕ) : Option (Accounting.Device ι) :=
  if x.kind = .device then
    (S.sheet? digest x).bind fun k => (A.sheet k.sku).map fun a =>
      let f := k.figures.bounding (A.estimate a)
      { chip := k.chip
        mem := 8 * k.mem
        rate := fun
          | .tc => ⌈tick * Finset.univ.sup f.peakRate⌉₊
          | .cuda => cuda
          | .oracle => oracle
        bw := k.bw
        cap := cap }
  else none

end Site

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

/-- `sheetDigest`: each new sheet's SKU is one the auditor has a data sheet for, and each new device names a registered
sheet's digest. -/
def Sheets : Prop :=
  (∀ k ∈ e.sheets, (A.sheet k.sku).isSome) ∧ ∀ x ∈ e.locs, x.kind = .device → x.sheet ∈ (S.add e).sheets.map D.sheet

/-- `figures`: each new sheet is accepted against the auditor's sheet `a` for its SKU: its figures against `a`'s estimate
(`DeviceFigures.Accepted`: the SKU `a`'s, the clock and every peak rate within the tolerance of `a`'s both ways, every
peak rate positive), and its memory figures exactly `a`'s. -/
def Figures : Prop :=
  ∀ k ∈ e.sheets, ∀ a ∈ A.sheet k.sku, k.figures.Accepted (A.estimate a) ∧ k.bw = a.bw ∧ k.chip = a.chip ∧ k.mem = a.mem

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
  Fresh D S e ∧ Refs S e ∧ Kinds S e ∧ Sheets D A S e ∧ Figures A e ∧ Capacity D S e ∧ Owners A e ∧ Covered c S e ∧
    Reached c S e

instance instDecidableFresh : Decidable (Fresh D S e) := by unfold Fresh; infer_instance
instance instDecidableRefs : Decidable (Refs S e) := by unfold Refs; infer_instance
instance instDecidableKinds : Decidable (Kinds S e) := by unfold Kinds; infer_instance
instance instDecidableSheets : Decidable (Sheets D A S e) := by unfold Sheets; infer_instance
instance instDecidableFigures : Decidable (Figures A e) := by unfold Figures; infer_instance
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
  else if ¬Figures A e then some .figures
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

/-- The checks of the entry's kind, against the Site `S` of the log before it: a site entry's (`Site.Entry.Valid`). An
information entry adds no hardware, so the hardware checks hold of it whenever they held of the log before it. -/
def Checks : Entry → Prop
  | .site _ s => s.Valid D A c S
  | .information _ _ => True

/-- **A valid entry** of a log for the composite `c`, against the Site `S` of the log before it and its head `head`: it
follows the head, and passes its kind's checks. -/
def Valid (e : Entry) : Prop := Follows head e ∧ Checks D A c S e

/-- The refusal of the entry's kind: a site entry's (`Site.Entry.refusal`), and none for an information entry. -/
def kindRefusal : Entry → Option Refusal
  | .site _ s => s.refusal D A c S
  | .information _ _ => none

/-- `duplicateInformation`: no item the entry registers is one the log before it, `pre`, or the entry itself already
registers (`Item.key`). -/
def Once (pre : Registration) (e : Entry) : Prop := ((pre.items ++ e.items).map Item.key).Nodup

instance instDecidableFollows (e : Entry) : Decidable (Follows head e) := by unfold Follows; infer_instance

instance instDecidableChecks : ∀ e : Entry, Decidable (Checks D A c S e)
  | .site _ s => inferInstanceAs (Decidable (s.Valid D A c S))
  | .information _ _ => inferInstanceAs (Decidable True)

instance instDecidableOnce (pre : Registration) (e : Entry) : Decidable (Once pre e) := by
  unfold Once; infer_instance

instance instDecidableValid (e : Entry) : Decidable (Valid D A c S head e) := by unfold Valid; infer_instance

/-- **The refusal**: `previous` when the entry doesn't follow the log's head, else its kind's, or `none` when it passes
every check. -/
def refusal (e : Entry) : Option Refusal :=
  if ¬Follows head e then some .previous else kindRefusal D A c S e

end Entry

/-- **A valid log** for the composite `c`: each entry is valid against the Site and head of the prefix before it, and
registers no item the prefix registers. -/
def Valid (D : Digests) (A : Auditor) (c : Composite) (log : Registration) : Prop :=
  ∀ i (h : i < log.length),
    log[i].Valid D A c (Site.ofLog (log.take i)) (head D c (log.take i)) ∧ log[i].Once (log.take i)

instance instDecidableValid (D : Digests) (A : Auditor) (c : Composite) (log : Registration) :
    Decidable (Valid D A c log) := by
  unfold Valid; infer_instance

end Registration

end Verity
