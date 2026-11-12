module

public import Definitions.Accounting.Device
public import Definitions.Core.DeviceFigures
public import Definitions.Core.Period
public import Mathlib.Data.Finset.Lattice.Fold
public import Definitions.Core.Compiler
public import Mathlib.Data.List.Perm.Basic
public import Mathlib.Data.List.Nodup
public import Mathlib.Data.NNRat.Floor

/-!
# The Site and its registration

The hardware the protocols run on is one **Site** (`Verity.Site`): the sheets its devices name by digest, a tree of
locations each with its owner, and the links between them. The Site is public. A workload's placement on it
(each block `j`'s location `loc_j`), its inference blocks and its declared advice are private, and only their padded
counts are public.

* **Locations** (`Site.Loc`): the site at the root, which states the Site's unit of time (`Site.timeUnit`); its hosts,
  the prover's machines; a host's devices; memory leaves by tier and bytes under a device or a host; and the auditor's
  own: its hosts, machines of the auditor's at the site, and its devices, the network certifiers and memory challengers,
  each at the site or in one of the auditor's hosts (`Site.Kind.parents`). A host is registered hardware with no compute
  rate: the Site's compute is its devices'. A host states its on-chip state and, for a tier, the rate its leaves of
  that tier share; a memory leaf directly under a host states its own rate. A device names its sheet by digest. The
  auditor's devices and hosts carry their keys, each with the scheme the key is of (`Site.SCHEMES`), and an auditor's
  host states the time base its clock runs on and how far that clock may run from real time.
* **Sheets** (`Site.Sheet`): a card, the figures the prover states for an SKU: its compute figures (`DeviceFigures`: the
  SKU, the tensor clock and the peak rate of each operand class) and its memory figures. A device's figures are its
  sheet's, and its device model (`Accounting.Device`) is read off them (`Site.device`).
* **Links** (`Site.Link`): two ends, each a location or `outside`, the rest of the world, with a capacity `β` in bits a
  second each way and a latency in the Site's units of time. A network certifier is the leaf that splits an uplink: one
  link from a host to it, one from it to the outside. The prover's boundary links are a host's links to the outside or
  to a certifier (`Site.Boundary`), and a link is certified when a certifier is at one of its ends (`Site.Certified`).
  A boundary link no certifier splits says why none does (`Site.Link.unmediated`), and only such a link may state no
  capacity.

The Site is folded from an append-only log, **the registration** (`Verity.Registration`), whose entries each have a
kind (`Registration.Entry`): the site entry (`Registration.Entry.site`), which adds sheets, locations and links
(`Site.Entry`), and the information entry (`Registration.Entry.information`), which registers items
(`Registration.Item`): θ's values, each by its committed root (`Registration.Theta`), and the builder `B`, the registered
compiler (`Registration.Builder`, holding `Verity.Compiler`), and the declaration (`Registration.Entry.declaration`),
which declares a day's circuit (`Registration.Declared`: its program's digest and its block Definitions by descriptor
key). The log's θ and builder are what its items register (`Registration.theta`, `Registration.builder?`), and the
registered circuit reads both from it (`Verity.RegisteredCircuit`). Each entry names the head of the log it follows
(`Registration.Entry.previous`): the digest of the log's last entry, which names its own predecessor, so the head is a
digest of the whole prefix. The Site of a log is its entries' additions in order (`Site.ofLog`), and a log is valid when
each entry is valid against the Site and head of the prefix before it, registers no item the prefix registers and
declares no day the prefix declares (`Registration.Valid`). An entry is checked as a whole (`Registration.Entry.Valid`,
a site entry's own checks being `Site.Entry.Valid`; an information entry and a declaration add no hardware, so they pass
them), and refused by the first check it fails (`Registration.Entry.refusal`):

| Refusal | The check |
| --- | --- |
| `previous` | the entry follows the log's head |
| `duplicateId` | no id is registered twice, `outside`'s included, and no sheet twice |
| `unknownRef` | each new location's parent and each new link's ends are registered |
| `kind` | each new location sits under a parent its kind allows and has a tier exactly when it is memory; each new link joins two different ends of kinds a link may join, at a positive capacity, or at none where it says why no certifier splits it; each certifier splits exactly one uplink |
| `sheetDigest` | each new sheet's SKU is one the auditor has a data sheet for, and each new device names a registered sheet |
| `figures` | each new sheet's figures are within the auditor's tolerance of its estimate for the SKU (`DeviceFigures.Accepted`), and its memory figures are the auditor's; each new memory leaf directly under a host states the auditor's rate and bytes for it (`Rates`), each new host the auditor's on-chip state and shared rates for it (`Chips`), and each new location of kind `site` the Site's unit of time (`Timed`) |
| `capacity` | each new memory leaf holds something, and each device's memory leaves fit in its sheet's memory |
| `owner` | each new certifier, challenger or auditor's host is the auditor's own, with its key and its key's scheme, and every other new location is the prover's; each new auditor's host states the auditor's time base and clock figure for it |
| `uncertified` | onsite: every boundary link is certified, or says why no certifier splits it |
| `challengerReach` | onsite: the host of every memory leaf is at an end of a memory challenger's link |

A log is held to one more check at its end, and not entry by entry, so that a later entry may add hardware under a
registered host: its Site is complete against the auditor's figures (`Registration.Complete`, the refusal `inventory`):
under each host, every memory leaf, card and way out the auditor lists for it is registered.

What an entry registers is held to the log before it (`Registration.Entry.Once`): no information item is registered
twice (`duplicateInformation`), and no day is declared twice (`duplicateDeclaration`). The log refuses its next entry by
the entry's own refusal, else by these (`Registration.refusal`).

**Days.** A day is a part of a cut of the monitoring period's span (`MonitoringPeriod`, `TickSpan.Cut`), and a
declaration names its day by the part's index. The prover declares one circuit a day, before the day's first coin: the
verifier reads the day's declaration from the log it holds when it samples (`Registration.declared`), and since a second
declaration of the day is refused, that declaration is the day's for the rest of the log. A window runs its day's
declared circuit when its span lies within the day's and its program is the one the day declares
(`Registration.RunsDeclared`).

The digests are parameters (`Registration.Digests`), SHA-512 in the deployed check. The auditor's data sheets, public
and looked up by SKU, its tolerance around their figures, its keys with their schemes, its figures for each host and
each memory leaf directly under one, and its own hosts' time bases and clock figures are the check's own
(`Registration.Auditor`). That those figures are the hardware is a public input: no check holds it and no premise says
it, and a result is over the registered hardware. At every choice of them the check is decidable.
-/

@[expose] public section

namespace Verity

open scoped NNRat

namespace Site

/-- What a location is. `host` is a machine of the prover's; `auditorHost` is a machine of the auditor's, which its
devices may run in and which holds none of the prover's hardware. `outside`, the kind of the rest of the world, is no
registered location's. -/
inductive Kind
  | site
  | host
  | device
  | memory
  | certifier
  | challenger
  | auditorHost
  | outside
  deriving DecidableEq, Repr

/-- The kinds of parent a location of each kind may have, `none` being the root: the site at the root, a host under the
site, a device under a host, memory under a device or a host, an auditor's host at the site, and the auditor's devices
at the site or in one of its hosts. -/
def Kind.parents : Kind → List (Option Kind)
  | .site => [none]
  | .host => [some .site]
  | .device => [some .host]
  | .memory => [some .device, some .host]
  | .certifier | .challenger => [some .site, some .auditorHost]
  | .auditorHost => [some .site]
  | .outside => []

/-- The pairs of kinds a link may join, in either order: two hosts, a host and the outside, a certifier and a host or
the outside, and a memory challenger and a host. No link ends at an auditor's host. -/
def Kind.links : List (Kind × Kind) :=
  [(.host, .host), (.host, .outside), (.certifier, .host), (.certifier, .outside), (.challenger, .host)]

/-- The auditor's own: its devices, the network certifiers and the memory challengers, and its hosts. -/
def Kind.auditor : Kind → Bool
  | .certifier | .challenger | .auditorHost => true
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
(`Accounting.Device`): bits read from or written to device memory a second (`bw`), bits of on-chip state (`chip`) and
bytes of device memory (`mem`). -/
structure Sheet where
  figures : DeviceFigures
  bw : ℕ
  chip : ℕ
  mem : ℕ
  deriving DecidableEq

/-- A sheet's SKU, its figures'. -/
def Sheet.sku (k : Sheet) : String := k.figures.sku

/-- A location, and who owns it. A device names its SKU's sheet by digest, and a memory leaf its tier and its bytes.
`bw` is the bits a second read from or written to a memory leaf directly under a host, and none on every other
location: a memory leaf under a device reads at its device's sheet's `bw`. `chip` is a host's on-chip state in bits, as
a sheet's is, and none on every other location. `shared` is, for a tier, the bits a second read from or written to a
host's leaves of that tier together, where one ceiling holds them all; empty on every other location. `key` is the
public key one of the auditor's devices or hosts is registered under, and `scheme` the id of the signature scheme that
key is of (`SCHEMES`), which says how many levels of keys stand under it; both empty on every other location.
`timebase` is the SKU of the time base the clock of an auditor's host runs on, and `drift` how far that clock may run
from real time, as a share of it: the drift of every device of the auditor's in that host, which reads that clock. Both
are none on every other location, and a device at the site, or in a host that states no drift, times nothing.
`timeUnit` is the Site's unit of time in seconds, on a location of kind `site` and none on every other
(`Site.timeUnit`). -/
structure Loc where
  id : String
  kind : Kind
  parent : Option String := none
  owner : Owner := .prover
  sheet : String := ""
  tier : Option Tier := none
  bytes : ℕ := 0
  key : String := ""
  scheme : String := ""
  bw : Option ℕ := none
  chip : Option ℕ := none
  shared : List (Tier × ℕ) := []
  drift : Option ℚ≥0 := none
  timeUnit : Option ℚ≥0 := none
  timebase : Option String := none
  deriving DecidableEq

/-- The auditor's figures for one memory leaf directly under a host: the host it is under, the bits a second read from
or written to it, and the bytes it holds. -/
structure MemoryFigures where
  host : String
  bw : ℕ
  bytes : ℕ
  deriving DecidableEq, Repr

/-- The auditor's figures for one host: its on-chip state in bits, the cards it carries, each SKU with its count, for
a tier the bits a second its leaves of that tier read or write together, and the ways traffic leaves it, each by its
link's id with the bits a second it carries each way. -/
structure HostFigures where
  chip : ℕ
  cards : List (String × ℕ) := []
  shared : List (Tier × ℕ) := []
  ways : List (String × Option ℕ) := []
  deriving DecidableEq, Repr

/-- The signature schemes a registered key may be of: one LMS key that signs, and a root LMS key that certifies the
keys that sign. -/
def SCHEMES : List String := ["lms-sha512/n64-w1-h20/v0", "hss-lms-sha512/n64-w1-h20/n64-w1-h20/v0"]

/-- A link between two ends, with its capacity `β` in bits a second each way and its latency in the Site's units of
time. `unmediated` is, on a boundary link no certifier splits, the reason none does; none on every other link. A link
states no capacity (`β` none) only where it carries `unmediated`: an interface that reads no line rate has none, and
no amount is read from it. -/
structure Link where
  id : String
  ends : String × String
  β : Option ℕ
  latency : ℕ
  unmediated : Option String := none
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

/-- **A day's declared circuit**: the day, by its index among the parts of the monitoring period's span cut into days
(`Registration.RunsDeclared`), the digest of the circuit's program, and its block Definitions by descriptor key. -/
structure Registration.Declared where
  day : ℕ
  program : String
  blocks : List String
  deriving DecidableEq

/-- **An entry of the registration**, by its kind, naming the head of the log it follows: the site entry, which adds
hardware to the Site, the information entry, which registers θ's values and the builder, and the declaration, which
declares a day's circuit. -/
inductive Registration.Entry
  | site (previous : String) (entry : Site.Entry)
  | information (previous : String) (items : List Registration.Item)
  | declaration (previous : String) (declared : Registration.Declared)

/-- **The registration**: the log the Site is folded from, oldest entry first. It is append-only: an entry is checked
against the log before it, and nothing after it changes it. -/
abbrev Registration : Type := List Registration.Entry

namespace Registration.Entry

/-- The head of the log the entry follows. -/
def previous : Entry → String
  | .site p _ => p
  | .information p _ => p
  | .declaration p _ => p

/-- What the entry adds to the Site: a site entry's sheets, locations and links, and nothing for an information entry
or a declaration. -/
def added : Entry → Site.Entry
  | .site _ s => s
  | .information _ _ => {}
  | .declaration _ _ => {}

/-- What the entry registers besides hardware: an information entry's items. -/
def items : Entry → List Item
  | .site _ _ => []
  | .information _ is => is
  | .declaration _ _ => []

/-- The circuit the entry declares: a declaration's, and none for the other kinds. -/
def declared? : Entry → Option Declared
  | .declaration _ d => some d
  | .site _ _ => none
  | .information _ _ => none

end Registration.Entry

namespace Registration

/-- The items the log registers, in the log's order. -/
def items (log : Registration) : List Item := log.flatMap Entry.items

/-- **θ**, the values the log registers. -/
def theta (log : Registration) : List Theta := log.items.filterMap fun | .theta t => some t | .builder _ => none

/-- **The builder** the log registers, which names the registered compiler. -/
def builder? (log : Registration) : Option Builder := log.items.findSome? fun | .builder b => some b | .theta _ => none

/-- The days the log's declarations declare, in the log's order. -/
def days (log : Registration) : List ℕ := log.filterMap fun e => e.declared?.map (·.day)

/-- **Day `j`'s declared circuit** in the log: the declaration that declares day `j`, the first when a log the check
refuses declares it twice; `none` when the log declares no circuit for day `j`. -/
def declared (log : Registration) (j : ℕ) : Option Declared :=
  log.findSome? fun e => e.declared?.filter (·.day == j)

/-- **A window runs its day's declared circuit**: the days are the cut `days` of the monitoring period `m`'s span into
`n` consecutive spans, the window's span `w` lies within day `j`'s, and `program` is the program the log declares for
day `j`. -/
def RunsDeclared (log : Registration) (m : MonitoringPeriod) {n : ℕ} (cut : m.span.Cut n) (w : TickSpan)
    (program : String) : Prop :=
  ∃ j : Fin n, w.Within (cut.part j) ∧ ∃ d ∈ declared log j, d.program = program

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

/-- **The Site's unit of time** in seconds: the one its first location of kind `site` states. A device's model rounds a
rate at it, and the links' latency is counted in it. -/
def timeUnit : Option ℚ≥0 := (S.locs.find? (·.kind = .site)).bind (·.timeUnit)

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
(`none` for an SKU it has no sheet for); the tolerance registration allows around an estimate; the key of each of its
own devices and hosts, by its id, and the scheme that key is of; its figures for each memory leaf directly under a host,
by the leaf's id; its figures for each host of the prover's, by the host's id; and for each of its own hosts, by the
host's id, how far its clock may run from real time and its time base's SKU. That these figures are the hardware is a
public input: no check holds it and no premise says it, and a result is over the registered hardware. -/
structure Auditor where
  sheet : String → Option Sheet
  tolerance : FigureTolerance
  key : String → Option String
  scheme : String → Option String := fun _ => none
  memory : List (String × Site.MemoryFigures) := []
  hosts : List (String × Site.HostFigures) := []
  drift : List (String × ℚ≥0) := []
  timebases : List (String × String) := []

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
  | inventory
  | uncertified
  | challengerReach
  | duplicateInformation
  | duplicateDeclaration
  deriving DecidableEq, Repr

end Registration

namespace Site

open Registration (Auditor)

/-- **A device's model** (`Accounting.Device`), read off the Site at ticks of the Site's unit of time
(`Site.timeUnit`). For a device `x` naming a registered sheet `k` whose SKU the auditor has a sheet `a` for: its on-chip
state is `k`'s, its device memory is `k`'s in bits, and a tick's bits from device memory to the chip and tensor-core
MACs are each a rate a second over one tick, rounded up: `k`'s `bw`, and the highest peak rate of the figures the bound
reads (`DeviceFigures.bounding`: class by class, the larger of `k`'s and `a`'s estimate's). No sheet states a device's
CUDA-core ops or oracle queries a tick, or its side channels, so they are parameters, as in `Accel.Card.machine`.
`none` when the Site states no unit of time, and for anything else. -/
def device (digest : Sheet → String) (A : Auditor) (S : Site) (x : Loc) {ι : Type} (cuda oracle : ℕ)
    (cap : ι → ℕ) : Option (Accounting.Device ι) :=
  if x.kind = .device then
    S.timeUnit.bind fun tick => (S.sheet? digest x).bind fun k => (A.sheet k.sku).map fun a =>
      let f := k.figures.bounding (A.estimate a)
      { chip := k.chip
        mem := 8 * k.mem
        rate := fun
          | .tc => ⌈tick * Finset.univ.sup f.peakRate⌉₊
          | .cuda => cuda
          | .oracle => oracle
        bw := ⌈tick * k.bw⌉₊
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

/-- `kind`: each new location sits under a parent of a kind its own allows, so an auditor's host is at the site and
holds only the auditor's devices, and has a tier exactly when it is memory; each new link joins two different ends, of
kinds a link may join, at a positive capacity or, where it says why no certifier splits it, at none; and each certifier
splits exactly one uplink, the far ends of its links being one host and the outside. -/
def Kinds : Prop :=
  (∀ x ∈ e.locs, x.parent.bind (S.add e).kind? ∈ x.kind.parents ∧ (x.kind = .memory ↔ x.tier.isSome)) ∧
    (∀ k ∈ e.links, k.ends.1 ≠ k.ends.2 ∧ (∃ p ∈ Kind.links, (S.add e).Joins k p.1 p.2) ∧
      (∀ b ∈ k.β, 0 < b) ∧ (k.β = none → k.unmediated.isSome)) ∧
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

/-- `figures`, a host's memory: each new memory leaf directly under a host states the auditor's figures for it: the
host it is under, a rate that is not 0, and its bytes; every other new location states no rate. -/
def Rates : Prop :=
  ∀ x ∈ e.locs,
    if x.kind = .memory ∧ x.parent.bind (S.add e).kind? = some .host then
      ∃ f ∈ A.memory.lookup x.id, x.parent = some f.host ∧ x.bw = some f.bw ∧ 0 < f.bw ∧ x.bytes = f.bytes
    else x.bw = none

/-- `figures`, the unit of time: each new location of kind `site` states the Site's unit of time, which is not 0 and is
the one the Site states; every other new location states none. -/
def Timed : Prop :=
  ∀ x ∈ e.locs,
    if x.kind = .site then ∃ u ∈ x.timeUnit, 0 < u ∧ (S.add e).timeUnit = some u else x.timeUnit = none

/-- `figures`, a host's own: each new host states the auditor's figures for it: its on-chip state in bits, and each
tier's shared rate, no shared rate 0; every other new location states neither. -/
def Chips : Prop :=
  ∀ x ∈ e.locs,
    if x.kind = .host then
      ∃ h ∈ A.hosts.lookup x.id, x.chip = some h.chip ∧ x.shared = h.shared ∧ ∀ p ∈ h.shared, 0 < p.2
    else x.chip = none ∧ x.shared = []

/-- `capacity`: each new memory leaf holds something, and each device's memory leaves fit in its sheet's memory. -/
def Capacity : Prop :=
  (∀ x ∈ e.locs, x.kind = .memory → 0 < x.bytes) ∧
    ∀ x ∈ (S.add e).locs, x.kind = .device → ∀ k ∈ (S.add e).sheet? D.sheet x, (S.add e).held x.id ≤ k.mem

/-- `owner`: each new certifier, challenger or auditor's host is the auditor's own, owned by it, with its key and its
key's scheme, one the check knows (`SCHEMES`); every other new location is the prover's and states no scheme; and each
new auditor's host states its time base, the auditor's for it, held as its key is, and the auditor's figure for its
clock, below 1 where there is one, and no other new location states a time base or a clock. -/
def Owners : Prop :=
  ∀ x ∈ e.locs,
    (x.kind.auditor = true → x.owner = .auditor ∧ A.key x.id = some x.key ∧ A.scheme x.id = some x.scheme ∧
      x.scheme ∈ SCHEMES) ∧
    (x.kind.auditor = false → x.owner = .prover ∧ x.scheme = "") ∧
    if x.kind = .auditorHost then
      (∃ t ∈ A.timebases.lookup x.id, x.timebase = some t) ∧ x.drift = A.drift.lookup x.id ∧ ∀ q ∈ x.drift, q < 1
    else x.timebase = none ∧ x.drift = none

/-- `uncertified`, onsite: every boundary link is certified, or says why no certifier splits it. -/
def Covered : Prop :=
  c = .onsite → ∀ k ∈ (S.add e).links, (S.add e).Boundary k → (S.add e).Certified k ∨ k.unmediated.isSome

/-- `challengerReach`, onsite: the host of every memory leaf is at an end of a memory challenger's link. -/
def Reached : Prop :=
  c = .onsite → ∀ x ∈ (S.add e).locs, x.kind = .memory → ∀ h ∈ (S.add e).host? x, h ∈ (S.add e).reached

/-- **A valid site entry** of a log for the composite `c`, against the Site `S` of the log before it: it passes every
check of a site entry. -/
def Valid : Prop :=
  Fresh D S e ∧ Refs S e ∧ Kinds S e ∧ Sheets D A S e ∧ (Figures A e ∧ Rates A S e ∧ Timed S e ∧ Chips A e) ∧
    Capacity D S e ∧ Owners A e ∧ Covered c S e ∧ Reached c S e

instance instDecidableFresh : Decidable (Fresh D S e) := by unfold Fresh; infer_instance
instance instDecidableRefs : Decidable (Refs S e) := by unfold Refs; infer_instance
instance instDecidableKinds : Decidable (Kinds S e) := by unfold Kinds; infer_instance
instance instDecidableSheets : Decidable (Sheets D A S e) := by unfold Sheets; infer_instance
instance instDecidableFigures : Decidable (Figures A e) := by unfold Figures; infer_instance
instance instDecidableRates : Decidable (Rates A S e) := by unfold Rates; infer_instance
instance instDecidableTimed : Decidable (Timed S e) := by unfold Timed; infer_instance
instance instDecidableChips : Decidable (Chips A e) := by unfold Chips; infer_instance
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
  else if ¬(Figures A e ∧ Rates A S e ∧ Timed S e ∧ Chips A e) then some .figures
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
information entry or a declaration adds no hardware, so the hardware checks hold of it whenever they held of the log
before it. -/
def Checks : Entry → Prop
  | .site _ s => s.Valid D A c S
  | .information _ _ => True
  | .declaration _ _ => True

/-- **A valid entry** of a log for the composite `c`, against the Site `S` of the log before it and its head `head`: it
follows the head, and passes its kind's checks. -/
def Valid (e : Entry) : Prop := Follows head e ∧ Checks D A c S e

/-- The refusal of the entry's kind: a site entry's (`Site.Entry.refusal`), and none for an information entry or a
declaration. -/
def kindRefusal : Entry → Option Refusal
  | .site _ s => s.refusal D A c S
  | .information _ _ => none
  | .declaration _ _ => none

/-- `duplicateInformation`: no item the entry registers is one the log before it, `pre`, or the entry itself already
registers (`Item.key`). -/
def ItemsOnce (pre : Registration) (e : Entry) : Prop := ((pre.items ++ e.items).map Item.key).Nodup

/-- `duplicateDeclaration`: the entry declares no day the log before it, `pre`, declares. -/
def DayOnce (pre : Registration) (e : Entry) : Prop := ∀ d ∈ e.declared?, d.day ∉ days pre

/-- What the entry registers is held to the log before it, `pre`: no information item twice, and no day twice. -/
def Once (pre : Registration) (e : Entry) : Prop := ItemsOnce pre e ∧ DayOnce pre e

instance instDecidableFollows (e : Entry) : Decidable (Follows head e) := by unfold Follows; infer_instance

instance instDecidableChecks : ∀ e : Entry, Decidable (Checks D A c S e)
  | .site _ s => inferInstanceAs (Decidable (s.Valid D A c S))
  | .information _ _ => inferInstanceAs (Decidable True)
  | .declaration _ _ => inferInstanceAs (Decidable True)

instance instDecidableItemsOnce (pre : Registration) (e : Entry) : Decidable (ItemsOnce pre e) := by
  unfold ItemsOnce; infer_instance

instance instDecidableDayOnce (pre : Registration) (e : Entry) : Decidable (DayOnce pre e) := by
  unfold DayOnce; infer_instance

instance instDecidableOnce (pre : Registration) (e : Entry) : Decidable (Once pre e) := by
  unfold Once; infer_instance

instance instDecidableValid (e : Entry) : Decidable (Valid D A c S head e) := by unfold Valid; infer_instance

/-- **The refusal**: `previous` when the entry doesn't follow the log's head, else its kind's, or `none` when it passes
every check. -/
def refusal (e : Entry) : Option Refusal :=
  if ¬Follows head e then some .previous else kindRefusal D A c S e

end Entry

/-- **A valid log** for the composite `c`: each entry is valid against the Site and head of the prefix before it, and
registers no item and declares no day the prefix registers or declares, so the log declares each day at most once. -/
def Valid (D : Digests) (A : Auditor) (c : Composite) (log : Registration) : Prop :=
  ∀ i (h : i < log.length),
    log[i].Valid D A c (Site.ofLog (log.take i)) (head D c (log.take i)) ∧ log[i].Once (log.take i)

instance instDecidableValid (D : Digests) (A : Auditor) (c : Composite) (log : Registration) :
    Decidable (Valid D A c log) := by
  unfold Valid; infer_instance

/-- `inventory`, of the Site a log folds to: **the Site is complete** against the auditor's figures. Under each of its
hosts, every memory leaf the auditor has figures for is registered, and so is every card it lists for the host, at
least its count of devices whose sheet is of its SKU, and every way out it lists, as a link under that id with an end
at the host and the auditor's rate. A log is held to it at its end and not entry by entry, so a later entry may add
hardware under a registered host; a result at a head is over a Site complete against the auditor's figures at that
head. -/
def Complete (D : Digests) (A : Auditor) (S : Site) : Prop :=
  ∀ h ∈ S.locs, h.kind = .host →
    (∀ f ∈ A.memory, f.2.host = h.id → ∃ x ∈ S.locs, x.id = f.1 ∧ x.kind = .memory ∧ x.parent = some h.id) ∧
    ∀ g ∈ A.hosts.lookup h.id,
      (∀ k ∈ g.cards, k.2 ≤ (S.locs.filter fun d =>
        d.kind = .device ∧ d.parent = some h.id ∧ (S.sheet? D.sheet d).map (·.sku) = some k.1).length) ∧
      ∀ w ∈ g.ways, ∃ k ∈ S.links, k.id = w.1 ∧ (k.ends.1 = h.id ∨ k.ends.2 = h.id) ∧ k.β = w.2

instance instDecidableComplete (D : Digests) (A : Auditor) (S : Site) : Decidable (Complete D A S) := by
  unfold Complete; infer_instance

/-- **The log's refusal of its next entry** `e`: the entry's own (`Entry.refusal`) against the log's Site and head, else
`duplicateInformation` when it registers an item the log or the entry already registers, else `duplicateDeclaration`
when it declares a day the log declares, or `none` when the log with `e` is valid. -/
def refusal (D : Digests) (A : Auditor) (c : Composite) (log : Registration) (e : Entry) : Option Refusal :=
  (e.refusal D A c (Site.ofLog log) (head D c log)).or
    (if ¬e.ItemsOnce log then some .duplicateInformation
     else if ¬e.DayOnce log then some .duplicateDeclaration else none)

end Registration

end Verity
