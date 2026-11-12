module

public import Lean.Data.Json
public import Base.Canon
public import Base.Hash
public import CoreProtocol.Ratio

/-!
# The Site in code

The spec's `Verity.Site` (`security/Definitions/Core/Site.lean`) as code, on Lean core and Flock's canonical JSON and
SHA-512: the registered hardware, folded from the registration log (`Site.ofLog`), the check each entry passes
(`Registration.Entry.refusal`), and the views the protocols read. Its log holds site entries only; the spec's
declarations of a day's circuit are the registration log's (`Core.Registration`). Its checks are the spec's, clause for
clause and in the spec's order, over this file's own types, each decidable; `Properties.Core.Site` states that each is the spec's, and
`verity.hardware.site` is its Python mirror, held to it by the difftest (`site-difftest`). A sheet carries its device's
figures (`DeviceFigures`, the spec's `Definitions/Core/DeviceFigures.lean`), nonnegative rationals (`Figure`, the spec's
`ℚ≥0`).

The digests are parameters as in the spec (`Registration.Digests`); the deployed ones (`Registration.sha512`) are
`SHA-512(tag ‖ 0 ‖ canon(j))` over the JSON `verity.hardware.site` writes, under `verity/registration/site/v2`,
`verity/registration/sheet/v2` and `verity/registration/genesis/v1`,
a rational written as `Core.Ratio` writes one. The views (`Site.sheetOf`, `device`, `node`, `tiers`, `boundary`, `units`,
`registered`, `host`) are functions of the site alone, but `device`, which reads the auditor's sheet too.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Core

namespace Site

open Lean (Json)

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

/-- The kinds of parent a location of each kind may have, `none` being the root. -/
def Kind.parents : Kind → List (Option Kind)
  | .site => [none]
  | .host => [some .site]
  | .device => [some .host]
  | .memory => [some .device, some .host]
  | .certifier | .challenger => [some .site, some .auditorHost]
  | .auditorHost => [some .site]
  | .outside => []

/-- The pairs of kinds a link may join, in either order. No link ends at an auditor's host. -/
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

/-- A nonnegative rational, the spec's `ℚ≥0`: a clock in Hz, a peak rate in MACs a second, a tolerance, or a tick in
seconds. -/
abbrev Figure : Type := {q : Rat // 0 ≤ q}

/-- The larger of two figures. -/
def Figure.max (a b : Figure) : Figure := if a.1 ≤ b.1 then b else a

/-- The operand class of a tensor-core MAC (`Verity.OperandClass`): int8 operands sampled uniformly over [-128, 127],
[-2, 1], [0, 3] or [0, 15], or the one value 0. -/
inductive OperandClass
  | uniformS8
  | uniformS2
  | uniformU2
  | uniformU4
  | zero
  deriving DecidableEq, Repr

/-- Every operand class. -/
def OperandClass.all : List OperandClass := [.uniformS8, .uniformS2, .uniformU2, .uniformU4, .zero]

theorem OperandClass.mem_all (c : OperandClass) : c ∈ OperandClass.all := by cases c <;> simp [OperandClass.all]

/-- A device's compute figures (`Verity.DeviceFigures`): its SKU, its tensor clock in Hz, and the peak rate of
tensor-core MACs a second of each operand class. -/
structure DeviceFigures where
  sku : String
  clock : Figure
  peakRate : OperandClass → Figure

/-- Two figures are equal when their SKUs, clocks and peak rates are. -/
instance DeviceFigures.instDecidableEq : DecidableEq DeviceFigures := fun a b =>
  decidable_of_iff (a.sku = b.sku ∧ a.clock = b.clock ∧ ∀ c ∈ OperandClass.all, a.peakRate c = b.peakRate c) <| by
    cases a; cases b
    simp only [DeviceFigures.mk.injEq]
    constructor
    · rintro ⟨h1, h2, h3⟩; exact ⟨h1, h2, funext fun c => h3 c (OperandClass.mem_all c)⟩
    · rintro ⟨h1, h2, rfl⟩; exact ⟨h1, h2, fun _ _ => rfl⟩

/-- The auditor's estimate of one SKU's figures, with the relative tolerance registration allows around them
(`Verity.DeviceEstimate`). -/
structure DeviceEstimate where
  figures : DeviceFigures
  tolerance : Figure

/-- `x` is within relative tolerance `τ` of `e`, in both directions (`Verity.WithinTolerance`): `(1 - τ)·e ≤ x ≤
(1 + τ)·e`, `1 - τ` truncated at 0. -/
def WithinTolerance (τ e x : Figure) : Prop := max (1 - τ.1) 0 * e.1 ≤ x.1 ∧ x.1 ≤ (1 + τ.1) * e.1

instance instDecidableWithinTolerance (τ e x : Figure) : Decidable (WithinTolerance τ e x) := by
  unfold WithinTolerance; infer_instance

/-- Registration's check of a device's figures `d` against the auditor's estimate `est` (`DeviceFigures.Accepted`): the
SKU is the estimate's, the clock and every peak rate are within the tolerance of the estimate's in both directions, and
every peak rate is positive. -/
def DeviceFigures.Accepted (est : DeviceEstimate) (d : DeviceFigures) : Prop :=
  d.sku = est.figures.sku ∧ WithinTolerance est.tolerance est.figures.clock d.clock ∧
    (∀ c ∈ OperandClass.all, WithinTolerance est.tolerance (est.figures.peakRate c) (d.peakRate c)) ∧
    ∀ c ∈ OperandClass.all, 0 < (d.peakRate c).1

instance DeviceFigures.instDecidableAccepted (est : DeviceEstimate) (d : DeviceFigures) : Decidable (d.Accepted est) := by
  unfold DeviceFigures.Accepted; infer_instance

/-- The figures the bound reads (`DeviceFigures.bounding`): the larger clock and the larger peak rate of each class, of
`d`'s and the estimate's. The SKU is `d`'s. -/
def DeviceFigures.bounding (est : DeviceEstimate) (d : DeviceFigures) : DeviceFigures where
  sku := d.sku
  clock := Figure.max d.clock est.figures.clock
  peakRate c := Figure.max (d.peakRate c) (est.figures.peakRate c)

/-- A sheet, a card: the figures the prover states for an SKU. Its compute figures (`figures`), and its memory figures
in the device model's terms: bits read from or written to device memory a second (`bw`), bits of on-chip state (`chip`) and
bytes of device memory (`mem`). -/
structure Sheet where
  figures : DeviceFigures
  bw : Nat
  chip : Nat
  mem : Nat
  deriving DecidableEq

/-- A sheet's SKU, its figures'. -/
def Sheet.sku (k : Sheet) : String := k.figures.sku

/-- A location, and who owns it. `bw` is the bits a second read from or written to a memory leaf directly under a host,
and none on every other location: a memory leaf under a device reads at its device's sheet's `bw` (`Site.readRate`).
A disk leaf's `bw` bounds the bytes that come off the disk, not a read the host returns from memory, so a read rate
over a host's disk leaves counts the host's DRAM tier's rate beside them, once.
`chip` is a host's on-chip state in bits, as a sheet's is, and none on every other location. `shared` is, for a tier,
the bits a second read from or written to a host's leaves of that tier together, where one ceiling holds them all;
empty on every other location. `key` is the public key one of the auditor's devices or hosts is registered under, and
`scheme` the id of the signature scheme that key is of (`SCHEMES`), which says how many levels of keys stand under it;
both empty on every other location.
`drift` is how far the clock of an auditor's host may run from real time, as a share of it, and it is the drift of
every device of the auditor's in that host, which reads that clock (`Site.drift?`); none on every other location, and a
device at the site, or in a host that states none, times nothing. `timeUnit` is the Site's unit of time in seconds, on a
location of kind `site` and none on every other (`Site.timeUnit`). `timebase` is the SKU of the time base the clock of
an auditor's host runs on, the card the auditor reads the clock's drift from; none on every other location. -/
structure Loc where
  id : String
  kind : Kind
  parent : Option String := none
  owner : Owner := .prover
  sheet : String := ""
  tier : Option Tier := none
  bytes : Nat := 0
  key : String := ""
  scheme : String := ""
  bw : Option Nat := none
  chip : Option Nat := none
  shared : List (Tier × Nat) := []
  drift : Option Figure := none
  timeUnit : Option Figure := none
  timebase : Option String := none
  deriving DecidableEq

/-- The auditor's figures for one memory leaf directly under a host: the host it is under, the bits a second read from
or written to it, and the bytes it holds. -/
structure MemoryFigures where
  host : String
  bw : Nat
  bytes : Nat
  deriving DecidableEq, Repr

/-- The auditor's figures for one host: its on-chip state in bits, the cards it carries, each SKU with its count, for
a tier the bits a second its leaves of that tier read or write together, and the ways traffic leaves it, each by its
link's id with the bits a second it carries each way. -/
structure HostFigures where
  chip : Nat
  cards : List (String × Nat) := []
  shared : List (Tier × Nat) := []
  ways : List (String × Option Nat) := []
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
  β : Option Nat
  latency : Nat
  unmediated : Option String := none
  deriving DecidableEq, Repr

/-- The rest of the world: an end every Site has, whose id no location may take. -/
def outside : String := "outside"

/-- A site entry: the sheets, locations and links it adds to the Site. -/
structure Entry where
  sheets : List Sheet := []
  locs : List Loc := []
  links : List Link := []
  deriving DecidableEq

end Site

/-- **The Site**: its sheets, its tree of locations and its links. -/
structure Site where
  sheets : List Site.Sheet
  locs : List Site.Loc
  links : List Site.Link
  deriving DecidableEq

/-- An entry of the registration, by its kind, naming the head of the log it follows: the site entry. -/
inductive Site.Registration.Entry
  | site (previous : String) (entry : Site.Entry)
  deriving DecidableEq

/-- **The registration**: the log the Site is folded from, oldest entry first. -/
abbrev Site.Registration : Type := List Site.Registration.Entry

namespace Site

namespace Registration.Entry

/-- The head of the log the entry follows. -/
def previous : Entry → String
  | .site p _ => p

/-- What the entry adds to the Site. -/
def added : Entry → Site.Entry
  | .site _ s => s

end Registration.Entry

/-- The Site an entry's additions extend `S` to, unchecked. -/
def add (S : Site) (e : Entry) : Site :=
  ⟨S.sheets ++ e.sheets, S.locs ++ e.locs, S.links ++ e.links⟩

/-- **The Site of a log**: its entries' sheets, locations and links, in the log's order. -/
def ofLog (log : Registration) : Site :=
  ⟨log.flatMap (·.added.sheets), log.flatMap (·.added.locs), log.flatMap (·.added.links)⟩

variable (S : Site)

/-- The location an id names: the first registered under it. -/
def loc? (i : String) : Option Loc := S.locs.find? (·.id == i)

/-- The kind of what an id names: the outside's, or its location's. -/
def kind? (i : String) : Option Kind := if i = outside then some .outside else (S.loc? i).map (·.kind)

/-- An id names something: the outside, or a registered location. -/
def Known (i : String) : Prop := i = outside ∨ i ∈ S.locs.map (·.id)

/-- Every registered id: the locations', the links' and the outside's. -/
def ids : List String := S.locs.map (·.id) ++ S.links.map (·.id) ++ [outside]

/-- A device's sheet: the registered sheet under the digest it names. -/
def sheet? (digest : Sheet → String) (x : Loc) : Option Sheet := S.sheets.find? (digest · == x.sheet)

/-- The Site's unit of time in seconds: the one its first location of kind `site` states. Its views round a rate at it,
and its links' latency is counted in it. -/
def timeUnit : Option Figure := (S.locs.find? (·.kind = .site)).bind (·.timeUnit)

/-- No link of the Site says that no certifier splits it. -/
def mediated : Bool := S.links.all fun k => k.unmediated.isNone

/-- The bytes of the memory leaves directly under `i`. -/
def held (i : String) : Nat := ((S.locs.filter fun x => x.kind = .memory ∧ x.parent = some i).map (·.bytes)).sum

/-- Link `k` joins kinds `a` and `b`, in either order. -/
def Joins (k : Link) (a b : Kind) : Prop :=
  (S.kind? k.ends.1, S.kind? k.ends.2) = (some a, some b) ∨ (S.kind? k.ends.1, S.kind? k.ends.2) = (some b, some a)

/-- A boundary link of the prover's: a host's link to the outside, or to the certifier that splits its uplink. -/
def Boundary (k : Link) : Prop := S.Joins k .host .outside ∨ S.Joins k .certifier .host

/-- A certified link: a certifier is at one of its ends. -/
def Certified (k : Link) : Prop := S.kind? k.ends.1 = some .certifier ∨ S.kind? k.ends.2 = some .certifier

/-- The far ends of `i`'s links. -/
def far (i : String) : List String :=
  (S.links.filter fun k => k.ends.1 = i ∨ k.ends.2 = i).flatMap fun k => [k.ends.1, k.ends.2].filter (· ≠ i)

/-- The host a memory leaf is in: its parent when that is a host, else its parent's parent. -/
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

namespace Site.Registration

/-- The composite a site's log is registered for. -/
inductive Composite
  | onsite
  | remote
  deriving DecidableEq, Repr

/-- The digests the log names things by: a sheet's, an entry's, and the head of a composite's empty log. -/
structure Digests where
  sheet : Sheet → String
  entry : Entry → String
  genesis : Composite → String

/-- What the auditor brings to the check: its data sheet for each SKU, public, whose figures are its estimate of the
SKU's; the tolerance registration allows around an estimate; the key of each of its own devices and hosts, by its id;
its figures for each memory leaf directly under a host, by the leaf's id; its figures for each host of the prover's, by
the host's id; and for each of its own hosts, by the host's id, how far its clock may run from real time and its time
base's SKU. That these figures are the hardware is a public input: no check holds it and no premise says it, and a
result is over the registered hardware. -/
structure Auditor where
  sheet : String → Option Sheet
  tolerance : Figure
  key : String → Option String
  scheme : String → Option String := fun _ => none
  memory : List (String × MemoryFigures) := []
  hosts : List (String × HostFigures) := []
  drift : List (String × Figure) := []
  timebases : List (String × String) := []

/-- The auditor's estimate its sheet `a` states: `a`'s figures, at the auditor's tolerance. -/
def Auditor.estimate (A : Auditor) (a : Sheet) : DeviceEstimate := ⟨a.figures, A.tolerance⟩

/-- The log's head: the digest of its last entry, or of its composite's genesis when it is empty. -/
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
  deriving DecidableEq, Repr

end Site.Registration

namespace Site.Entry

open Site.Registration (Digests Auditor Composite Refusal)

variable (D : Digests) (A : Auditor) (c : Composite) (S : Site) (e : Entry)

/-- `duplicateId`: no id is registered twice, the outside's among them, and no sheet twice. -/
def Fresh : Prop := (S.add e).ids.Nodup ∧ ((S.add e).sheets.map D.sheet).Nodup

/-- `unknownRef`: each new location's parent and each new link's ends name something registered. -/
def Refs : Prop :=
  (∀ x ∈ e.locs, ∀ p ∈ x.parent, (S.add e).Known p) ∧
    ∀ k ∈ e.links, (S.add e).Known k.ends.1 ∧ (S.add e).Known k.ends.2

/-- `kind`: parents and tiers by kind, so an auditor's host is at the site and holds only the auditor's devices; links
between kinds a link may join, each at a positive capacity or, where it says no certifier splits it, at none, and each
certifier splitting exactly one uplink. -/
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

/-- `figures`: each new sheet is accepted against the auditor's sheet `a` for its SKU: its figures against `a`'s
estimate, and its memory figures exactly `a`'s. -/
def Figures : Prop :=
  ∀ k ∈ e.sheets, ∀ a ∈ A.sheet k.sku, k.figures.Accepted (A.estimate a) ∧ k.bw = a.bw ∧ k.chip = a.chip ∧ k.mem = a.mem

/-- `figures`, a host's memory: each new memory leaf directly under a host states the auditor's figures for it: the
host it is under, a rate that is not 0, and its bytes; every other new location states no rate. -/
def Rates : Prop :=
  ∀ x ∈ e.locs,
    if x.kind = .memory ∧ x.parent.bind (S.add e).kind? = some .host then
      ∃ f ∈ A.memory.lookup x.id, x.parent = some f.host ∧ x.bw = some f.bw ∧ 0 < f.bw ∧ x.bytes = f.bytes
    else x.bw = none

/-- `figures`, a host's own: each new host states the auditor's figures for it, its on-chip state in bits and each
tier's shared rate, none of them 0; every other new location states neither. -/
def Chips : Prop :=
  ∀ x ∈ e.locs,
    if x.kind = .host then
      ∃ h ∈ A.hosts.lookup x.id, x.chip = some h.chip ∧ x.shared = h.shared ∧
        (h.shared.all fun (p : Tier × Nat) => decide (0 < p.2)) = true
    else x.chip = none ∧ x.shared = []

/-- `figures`, the unit of time: each new location of kind `site` states the Site's unit of time, which is not 0 and is
the one the Site states; every other new location states none. -/
def Timed : Prop :=
  ∀ x ∈ e.locs,
    if x.kind = .site then ∃ u ∈ x.timeUnit, 0 < u.1 ∧ (S.add e).timeUnit = some u else x.timeUnit = none

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
      (∃ t ∈ A.timebases.lookup x.id, x.timebase = some t) ∧ x.drift = A.drift.lookup x.id ∧
        (x.drift.all fun (q : Figure) => decide (q.1 < 1)) = true
    else x.timebase = none ∧ x.drift = none

/-- `uncertified`, onsite: every boundary link is certified, or says why no certifier splits it. -/
def Covered : Prop :=
  c = .onsite → ∀ k ∈ (S.add e).links, (S.add e).Boundary k → (S.add e).Certified k ∨ k.unmediated.isSome

/-- `challengerReach`, onsite: the host of every memory leaf is at an end of a memory challenger's link. -/
def Reached : Prop :=
  c = .onsite → ∀ x ∈ (S.add e).locs, x.kind = .memory → ∀ h ∈ (S.add e).host? x, h ∈ (S.add e).reached

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

/-- A site entry's refusal: the first of its checks it fails, or `none` when it passes them all. -/
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

namespace Site.Registration

namespace Entry

variable (D : Digests) (A : Auditor) (c : Composite) (S : Site) (head : String)

/-- The refusal of the entry's kind: a site entry's. -/
def kindRefusal : Entry → Option Refusal
  | .site _ s => s.refusal D A c S

/-- **The refusal**: `previous` when the entry doesn't follow the log's head, else its kind's, or `none`. -/
def refusal (e : Entry) : Option Refusal :=
  if e.previous ≠ head then some .previous else kindRefusal D A c S e

end Entry

/-- `inventory`, of the Site a log folds to: under each of its hosts, every memory leaf the auditor has figures for is
registered, and so is every card it lists for the host, at least its count of devices whose sheet is of its SKU, and
every way out it lists, as a link under that id with an end at the host and the auditor's rate. A log is held to it at
its end and not entry by entry, so a later entry may add hardware under a registered host; a result at a head is over a
Site complete against the auditor's figures at that head. -/
def Complete (D : Digests) (A : Auditor) (S : Site) : Prop :=
  ∀ h ∈ S.locs, h.kind = .host →
    (∀ f ∈ A.memory, f.2.host = h.id →
      ∃ x ∈ S.locs, x.id = f.1 ∧ x.kind = .memory ∧ x.parent = some h.id) ∧
    ∀ g ∈ A.hosts.lookup h.id,
      (∀ k ∈ g.cards, k.2 ≤ (S.locs.filter fun d =>
        d.kind = .device ∧ d.parent = some h.id ∧ (S.sheet? D.sheet d).map (·.sku) = some k.1).length) ∧
      ∀ w ∈ g.ways, ∃ k ∈ S.links, k.id = w.1 ∧ (k.ends.1 = h.id ∨ k.ends.2 = h.id) ∧ k.β = w.2

instance instDecidableComplete (D : Digests) (A : Auditor) (S : Site) : Decidable (Complete D A S) := by
  unfold Complete; infer_instance

/-- **The check of a log**: the Site it folds to; or the first entry refused, with its index and refusal; or, where
every entry passes and the Site is not complete against the auditor's figures (`Complete`), `inventory` at the log's
length. -/
def check (D : Digests) (A : Auditor) (c : Composite) (log : Registration) : Except (Nat × Refusal) Site :=
  go 0 ⟨[], [], []⟩ (D.genesis c) log
where
  go : Nat → Site → String → Registration → Except (Nat × Refusal) Site
    | i, S, _, [] => if Complete D A S then .ok S else .error (i, .inventory)
    | i, S, h, e :: rest =>
      match e.refusal D A c S h with
      | some r => .error (i, r)
      | none => go (i + 1) (S.add e.added) (D.entry e) rest

end Site.Registration

/-! ## The views -/

namespace Site

variable (S : Site)

/-- The devices, in the log's order: the locations that compute. -/
def units : List String := (S.locs.filter (·.kind = .device)).map (·.id)

/-- A location or a link is registered under `i`. -/
def registered (i : String) : Bool := S.locs.any (·.id == i) || S.links.any (·.id == i)

/-- Device `d`'s data sheet, or `none` when `d` is no device or names no registered sheet. -/
def sheetOf (digest : Sheet → String) (d : String) : Option Sheet :=
  match S.loc? d with
  | some x => if x.kind = .device then S.sheet? digest x else none
  | none => none

/-- The classes of op a device model counts a tick (`Accounting.OpClass`): tensor-core MACs, CUDA-core ops and oracle
queries. -/
inductive OpClass
  | tc
  | cuda
  | oracle
  deriving DecidableEq, Repr

/-- A device's model (`Accounting.Device`): bits of on-chip state, bits of device memory, ops of each class a tick, bits
from device memory to the chip a tick, and bits in over each side channel `i` a tick. -/
structure DeviceModel (ι : Type) where
  chip : Nat
  mem : Nat
  rate : OpClass → Nat
  bw : Nat
  cap : ι → Nat

/-- **Device `d`'s model** (`Site.device`), at ticks of the Site's unit of time, read off its sheet `k` and the
auditor's sheet `a` for `k`'s SKU: `k`'s on-chip state, `k`'s device memory in bits, and a tick's bits from device
memory to the chip and tensor-core MACs, each a rate a second over one tick, rounded up: `k`'s `bw`, and the highest
peak rate of the figures the bound reads (`DeviceFigures.bounding`). No sheet states CUDA-core ops or oracle queries a
tick, or side channels, so they are parameters. `none` when the Site states no unit of time, `d` names no registered
sheet or the auditor has none for its SKU. -/
def device (digest : Sheet → String) (A : Registration.Auditor) (d : String) {ι : Type}
    (cuda oracle : Nat) (cap : ι → Nat) : Option (DeviceModel ι) :=
  S.timeUnit.bind fun (tick : Figure) => (S.sheetOf digest d).bind fun k => (A.sheet k.sku).map fun a =>
    let f := k.figures.bounding (A.estimate a)
    let peak := OperandClass.all.foldl (fun m c => max m (f.peakRate c).1) 0
    { chip := k.chip
      mem := 8 * k.mem
      rate := fun
        | .tc => (tick.1 * peak).ceil.toNat
        | .cuda => cuda
        | .oracle => oracle
      bw := (tick.1 * k.bw).ceil.toNat
      cap := cap }

/-- The host of the prover's a location is in: itself when it is one. None for a location in none, as the auditor's
devices and hosts are. -/
def host (i : String) : Option String :=
  (S.loc? i).bind fun x =>
    if x.kind = .host then some x.id else (S.host? x).filter fun h => S.kind? h = some .host

/-- The auditor's host whose clock device `d` of the auditor's reads: the one it is in. None for a device at the
site. -/
def clock? (d : String) : Option String :=
  ((S.loc? d).bind fun x => x.parent).filter fun p => S.kind? p = some .auditorHost

/-- How far the clock device `d` of the auditor's reads may run from real time, as a share of it: what the auditor's
host it is in states. None for a device at the site or in a host that states none. -/
def drift? (d : String) : Option Figure := (S.clock? d).bind fun h => (S.loc? h).bind fun x => x.drift

/-- `x` is `i` or lies under it, following at most `n` parents. -/
def within (i : String) : Nat → Loc → Bool
  | 0, _ => false
  | n + 1, x => x.id == i || match x.parent.bind S.loc? with
    | some y => within i n y
    | none => false

/-- The part of the site at location `i`: `i` and every location under it, the sheets its devices name, and the links
with an end in it. -/
def part (digest : Sheet → String) (i : String) : Site :=
  let locs := S.locs.filter (S.within i (S.locs.length + 1))
  let ids := locs.map (·.id)
  let digests := (locs.filter (·.kind = .device)).map (·.sheet)
  ⟨S.sheets.filter (digests.contains <| digest ·), locs,
    S.links.filter fun k => ids.contains k.ends.1 || ids.contains k.ends.2⟩

/-- Host `h` as registered, or `none` when `h` is no host. -/
def node (digest : Sheet → String) (h : String) : Option Site :=
  if S.kind? h = some .host then some (S.part digest h) else none

/-- The bytes of memory at location `h`, by tier. -/
def tiers (digest : Sheet → String) (h : String) : List (Tier × Nat) :=
  let p := S.part digest h
  [.hbm, .dram, .disk].map fun t => (t, ((p.locs.filter fun x => x.kind = .memory ∧ x.tier = some t).map (·.bytes)).sum)

/-- The bits a second readable from the memory leaves `xs`: a host's leaves of one tier at the sum of their own
registered rates, or at the host's shared rate for the tier where that is less; and each device with a leaf among them
at its sheet's `bw`, once. A disk leaf's rate bounds the bytes that come off the disk, and a host can return a disk
leaf's bytes from memory, so a host with a disk leaf among `xs` counts its whole DRAM tier's rate, whichever of its
DRAM leaves are among them. `none` when a leaf states no rate and is under no device with a registered sheet. -/
def readRate (digest : Sheet → String) (xs : List String) : Option Nat := do
  let ys ← xs.mapM S.loc?
  let devices ← (ys.filter fun y => y.bw.isNone).mapM fun y => y.parent
  let sheets ← devices.eraseDups.mapM fun d => (S.sheetOf digest d).map fun k => k.bw
  let rated : List Loc := ys.filter fun y => y.bw.isSome
  let tier (zs : List Loc) (g : Option String × Option Tier) : Nat :=
    let sum : Nat := ((zs.filter fun y => (y.parent, y.tier) = g).filterMap fun y => y.bw).sum
    match (g.1.bind S.loc?).bind fun h => g.2.bind fun t => h.shared.lookup t with
    | some c => min sum c
    | none => sum
  let disks : List (Option String) := ((rated.filter fun y => y.tier = some .disk).map fun y => y.parent).eraseDups
  let groups : List (Option String × Option Tier) :=
    ((rated.map fun y => (y.parent, y.tier)) ++ disks.map fun h => (h, some Tier.dram)).eraseDups
  let hosts : List Nat := groups.map fun g =>
    if g.2 = some Tier.dram ∧ disks.contains g.1 then tier S.locs g else tier rated g
  pure (hosts.sum + sheets.sum)

/-- What a rate of `rate` a second carries over `ticks` of the Site's units of time, rounded up once. `none` when the
Site states no unit of time. -/
def over (rate ticks : Nat) : Option Nat := S.timeUnit.map fun (u : Figure) => (u.1 * (rate * ticks : Nat)).ceil.toNat

/-- The boundary links with an end at host `h`, or every boundary link. -/
def boundary (h : Option String) : List Link :=
  S.links.filter fun k => S.Boundary k ∧ (h.all fun h => h = k.ends.1 ∨ h = k.ends.2)

end Site

/-! ## The JSON `verity.core.site` writes, and the deployed digests -/

namespace Site

open Lean (Json)
open Flock (V)

def SITE : String := "verity/registration/site/v2"
def SHEET : String := "verity/registration/sheet/v2"
def GENESIS : String := "verity/registration/genesis/v1"

/-- `SHA-512(tag ‖ 0 ‖ canon(j))` as hex. Every JSON these types write holds integers and strings only, which
`Flock.canon` always writes, so the fallback is never taken. -/
def tagged (tag : String) (j : Json) : String :=
  Flock.hexOf (Flock.Sha512.hash (tag.toUTF8.push 0 ++ ((Flock.canon j).toOption.getD "").toUTF8))

def Kind.name : Kind → String
  | .site => "site" | .host => "host" | .device => "device" | .memory => "memory" | .certifier => "certifier"
  | .challenger => "challenger" | .auditorHost => "auditor-host" | .outside => "outside"

def Tier.name : Tier → String
  | .hbm => "hbm" | .dram => "dram" | .disk => "disk"

def Owner.name : Owner → String
  | .prover => "prover" | .auditor => "auditor"

def Registration.Composite.name : Registration.Composite → String
  | .onsite => "onsite" | .remote => "remote"

/-- A refusal's name in `verity.core.site` (`REFUSALS`): the constructor's, in kebab case. -/
def Registration.Refusal.name : Registration.Refusal → String
  | .previous => "previous" | .duplicateId => "duplicate-id" | .unknownRef => "unknown-ref" | .kind => "kind"
  | .sheetDigest => "sheet-digest" | .figures => "figures" | .capacity => "capacity" | .owner => "owner"
  | .inventory => "inventory" | .uncertified => "uncertified" | .challengerReach => "challenger-reach"

/-- An operand class's name in `verity.core.site` (`OPERAND_CLASSES`): the constructor's, in kebab case. -/
def OperandClass.name : OperandClass → String
  | .uniformS8 => "uniform-s8" | .uniformS2 => "uniform-s2" | .uniformU2 => "uniform-u2" | .uniformU4 => "uniform-u4"
  | .zero => "zero"

private def optJson {α : Type} (f : α → Json) : Option α → Json
  | none => .null
  | some x => f x

/-- A figure as `Core.Ratio` writes a ratio: `n`, or `n/d` in lowest terms (`Rat`'s `toString`). -/
def Figure.toJson (q : Figure) : Json := .str (toString q.1)

def DeviceFigures.toJson (d : DeviceFigures) : Json :=
  Json.mkObj [("clock", d.clock.toJson),
    ("peak_rate", Json.mkObj (OperandClass.all.map fun c => (c.name, (d.peakRate c).toJson))), ("sku", .str d.sku)]

def Sheet.toJson (k : Sheet) : Json :=
  Json.mkObj [("bw", Lean.toJson k.bw), ("chip", Lean.toJson k.chip), ("figures", k.figures.toJson),
    ("mem", Lean.toJson k.mem)]

private def sharedJson (p : Tier × Nat) : Json := Json.mkObj [("bw", Lean.toJson p.2), ("tier", .str p.1.name)]

def Loc.toJson (x : Loc) : Json :=
  Json.mkObj [("bw", optJson Lean.toJson x.bw), ("bytes", Lean.toJson x.bytes), ("chip", optJson Lean.toJson x.chip),
    ("drift", optJson Figure.toJson x.drift), ("id", .str x.id), ("key", .str x.key), ("kind", .str x.kind.name),
    ("owner", .str x.owner.name), ("parent", optJson .str x.parent), ("scheme", .str x.scheme),
    ("shared", .arr (x.shared.map sharedJson).toArray), ("sheet", .str x.sheet),
    ("tier", optJson (fun t : Tier => .str t.name) x.tier), ("time_unit", optJson Figure.toJson x.timeUnit),
    ("timebase", optJson .str x.timebase)]

def Link.toJson (k : Link) : Json :=
  Json.mkObj [("beta", optJson Lean.toJson k.β), ("ends", .arr #[.str k.ends.1, .str k.ends.2]), ("id", .str k.id),
    ("latency", Lean.toJson k.latency), ("unmediated", optJson .str k.unmediated)]

def Registration.Entry.toJson : Registration.Entry → Json
  | .site p s => Json.mkObj [("format", .str SITE), ("links", .arr (s.links.map Link.toJson).toArray),
      ("locs", .arr (s.locs.map Loc.toJson).toArray), ("previous", .str p),
      ("sheets", .arr (s.sheets.map Sheet.toJson).toArray)]

def toJson (S : Site) : Json :=
  Json.mkObj [("links", .arr (S.links.map Link.toJson).toArray), ("locs", .arr (S.locs.map Loc.toJson).toArray),
    ("sheets", .arr (S.sheets.map Sheet.toJson).toArray)]

/-- The deployed digests: SHA-512 over the JSON `verity.core.site` writes, under the registration log's tags. -/
def Registration.sha512 : Registration.Digests where
  sheet k := tagged SHEET k.toJson
  entry e := tagged SITE e.toJson
  genesis c := tagged GENESIS (Json.mkObj [("composite", .str c.name)])

private def keysOf : Json → Option (List String)
  | .obj kvs => some (kvs.toList.map (·.1))
  | _ => none

/-- `j` is an object with exactly the keys `ks`, which are sorted. -/
def exactly (j : Json) (ks : List String) (what : String) : V Unit :=
  if keysOf j == some ks then pure () else throw s!"{what}: the keys must be [{", ".intercalate ks}]"

private def byName {α : Type} (all : List α) (name : α → String) (j : Json) (what : String) : V α := do
  let s ← Flock.J.asStr j what
  match all.find? (name · == s) with
  | some x => pure x
  | none => throw s!"{what}: {s} is not one of [{", ".intercalate (all.map name)}]"

private def optOf {α : Type} (f : Json → V α) : Json → V (Option α)
  | .null => pure none
  | j => some <$> f j

/-- A figure as `Figure.toJson` writes it, and only so (`Core.Ratio.ofString`). -/
def Figure.ofJson (j : Json) (what : String) : V Figure := do
  let r ← Ratio.ofString (← Flock.J.asStr j what) what
  let q := mkRat r.num r.den
  if h : 0 ≤ q then pure ⟨q, h⟩ else throw s!"{what}: negative"

def DeviceFigures.ofJson (j : Json) : V DeviceFigures := do
  exactly j ["clock", "peak_rate", "sku"] "a sheet's figures"
  let rates ← Flock.J.get j "peak_rate" "a sheet's figures"
  exactly rates ["uniform-s2", "uniform-s8", "uniform-u2", "uniform-u4", "zero"] "the peak rates"
  let rate : OperandClass → V Figure := fun c => do
    Figure.ofJson (← Flock.J.get rates c.name "the peak rates") s!"{c.name}'s peak rate"
  let s8 ← rate .uniformS8
  let s2 ← rate .uniformS2
  let u2 ← rate .uniformU2
  let u4 ← rate .uniformU4
  let z ← rate .zero
  pure { sku := ← Flock.J.str j "sku" "a sheet's figures",
         clock := ← Figure.ofJson (← Flock.J.get j "clock" "a sheet's figures") "clock",
         peakRate := fun | .uniformS8 => s8 | .uniformS2 => s2 | .uniformU2 => u2 | .uniformU4 => u4 | .zero => z }

def Sheet.ofJson (j : Json) : V Sheet := do
  exactly j ["bw", "chip", "figures", "mem"] "a sheet"
  pure { figures := ← DeviceFigures.ofJson (← Flock.J.get j "figures" "a sheet"), bw := ← Flock.J.nat j "bw" "a sheet",
         chip := ← Flock.J.nat j "chip" "a sheet", mem := ← Flock.J.nat j "mem" "a sheet" }

private def sharedOf (j : Json) : V (Tier × Nat) := do
  exactly j ["bw", "tier"] "a shared rate"
  pure (← byName [.hbm, .dram, .disk] Tier.name (← Flock.J.get j "tier" "a shared rate") "a shared rate's tier",
        ← Flock.J.nat j "bw" "a shared rate")

def Loc.ofJson (j : Json) : V Loc := do
  exactly j ["bw", "bytes", "chip", "drift", "id", "key", "kind", "owner", "parent", "scheme", "shared", "sheet",
    "tier", "time_unit", "timebase"] "a location"
  let get := fun k => Flock.J.get j k "a location"
  pure { id := ← Flock.J.str j "id" "a location",
         kind := ← byName [.site, .host, .device, .memory, .certifier, .challenger, .auditorHost, .outside] Kind.name
           (← get "kind") "a location's kind",
         parent := ← optOf (Flock.J.asStr · "a location's parent") (← get "parent"),
         owner := ← byName [.prover, .auditor] Owner.name (← get "owner") "a location's owner",
         sheet := ← Flock.J.str j "sheet" "a location",
         tier := ← optOf (byName [.hbm, .dram, .disk] Tier.name · "a location's tier") (← get "tier"),
         bytes := ← Flock.J.nat j "bytes" "a location", key := ← Flock.J.str j "key" "a location",
         scheme := ← Flock.J.str j "scheme" "a location",
         bw := ← optOf (Flock.J.asNat · "a location's bw") (← get "bw"),
         chip := ← optOf (Flock.J.asNat · "a location's chip") (← get "chip"),
         shared := ← (← Flock.J.arr j "shared" "a location").toList.mapM sharedOf,
         drift := ← optOf (Figure.ofJson · "a location's drift") (← get "drift"),
         timeUnit := ← optOf (Figure.ofJson · "a location's time unit") (← get "time_unit"),
         timebase := ← optOf (Flock.J.asStr · "a location's time base") (← get "timebase") }

def Link.ofJson (j : Json) : V Link := do
  exactly j ["beta", "ends", "id", "latency", "unmediated"] "a link"
  let ends ← Flock.J.arr j "ends" "a link"
  let #[a, b] := ends | throw "a link: ends: not two ends"
  pure { id := ← Flock.J.str j "id" "a link", ends := (← Flock.J.asStr a "a link's end", ← Flock.J.asStr b "a link's end"),
         β := ← optOf (Flock.J.asNat · "a link's beta") (← Flock.J.get j "beta" "a link"),
         latency := ← Flock.J.nat j "latency" "a link",
         unmediated := ← optOf (Flock.J.asStr · "a link's unmediated") (← Flock.J.get j "unmediated" "a link") }

def Registration.Entry.ofJson (j : Json) : V Registration.Entry := do
  exactly j ["format", "links", "locs", "previous", "sheets"] "an entry"
  if (← Flock.J.str j "format" "an entry") != SITE then throw s!"an entry: format is not {SITE}"
  let sheets ← (← Flock.J.arr j "sheets" "an entry").toList.mapM Sheet.ofJson
  let locs ← (← Flock.J.arr j "locs" "an entry").toList.mapM Loc.ofJson
  let links ← (← Flock.J.arr j "links" "an entry").toList.mapM Link.ofJson
  pure (.site (← Flock.J.str j "previous" "an entry") ⟨sheets, locs, links⟩)

/-- A Site as `toJson` writes it, and only so: what a check that takes the registered hardware as a parameter reads. -/
def ofJson (j : Json) : V Site := do
  exactly j ["links", "locs", "sheets"] "a site"
  let sheets ← (← Flock.J.arr j "sheets" "a site").toList.mapM Sheet.ofJson
  let locs ← (← Flock.J.arr j "locs" "a site").toList.mapM Loc.ofJson
  let links ← (← Flock.J.arr j "links" "a site").toList.mapM Link.ofJson
  pure ⟨sheets, locs, links⟩

def Registration.Composite.ofJson (j : Json) : V Registration.Composite :=
  byName [.onsite, .remote] Registration.Composite.name j "the log's composite"

/-- A log as `Registration.to_json` writes it: its composite and its entries. -/
def Registration.ofJson (j : Json) : V (Registration.Composite × Registration) := do
  exactly j ["composite", "entries"] "the log"
  pure (← Registration.Composite.ofJson (← Flock.J.get j "composite" "the log"),
    ← (← Flock.J.arr j "entries" "the log").toList.mapM Registration.Entry.ofJson)

/-- The auditor as `Auditor.to_json` writes it: its devices' and hosts' keys, its data sheets, the first under an SKU
or a device being the one the check reads, its hosts' time bases and its tolerance. -/
def Registration.Auditor.ofJson (j : Json) : V Registration.Auditor := do
  exactly j ["drift", "hosts", "keys", "memory", "sheets", "timebases", "tolerance"] "the auditor"
  let timebases ← (← Flock.J.arr j "timebases" "the auditor").toList.mapM fun t => do
    exactly t ["id", "sku"] "a host's time base"
    pure (← Flock.J.str t "id" "a host's time base", ← Flock.J.str t "sku" "a host's time base")
  let drift ← (← Flock.J.arr j "drift" "the auditor").toList.mapM fun d => do
    exactly d ["drift", "id"] "a host's drift"
    pure (← Flock.J.str d "id" "a host's drift",
          ← Figure.ofJson (← Flock.J.get d "drift" "a host's drift") "a host's drift")
  let keys ← (← Flock.J.arr j "keys" "the auditor").toList.mapM fun k => do
    exactly k ["id", "key", "scheme"] "a key"
    pure (← Flock.J.str k "id" "a key", ← Flock.J.str k "key" "a key", ← Flock.J.str k "scheme" "a key")
  let memory ← (← Flock.J.arr j "memory" "the auditor").toList.mapM fun m => do
    exactly m ["bw", "bytes", "host", "id"] "a memory leaf's figures"
    let f : MemoryFigures :=
      { host := ← Flock.J.str m "host" "a memory leaf's figures", bw := ← Flock.J.nat m "bw" "a memory leaf's figures",
        bytes := ← Flock.J.nat m "bytes" "a memory leaf's figures" }
    pure (← Flock.J.str m "id" "a memory leaf's figures", f)
  let hosts ← (← Flock.J.arr j "hosts" "the auditor").toList.mapM fun h => do
    exactly h ["cards", "chip", "id", "shared", "ways"] "a host's figures"
    let cards ← (← Flock.J.arr h "cards" "a host's figures").toList.mapM fun k => do
      exactly k ["count", "sku"] "a host's card"
      pure (← Flock.J.str k "sku" "a host's card", ← Flock.J.nat k "count" "a host's card")
    let ways ← (← Flock.J.arr h "ways" "a host's figures").toList.mapM fun w => do
      exactly w ["beta", "id"] "a host's way out"
      pure (← Flock.J.str w "id" "a host's way out",
        ← optOf (Flock.J.asNat · "a host's way out's beta") (← Flock.J.get w "beta" "a host's way out"))
    let f : HostFigures :=
      { chip := ← Flock.J.nat h "chip" "a host's figures", cards, ways,
        shared := ← (← Flock.J.arr h "shared" "a host's figures").toList.mapM sharedOf }
    pure (← Flock.J.str h "id" "a host's figures", f)
  let sheets ← (← Flock.J.arr j "sheets" "the auditor").toList.mapM Sheet.ofJson
  pure { sheet := fun sku => sheets.find? (·.sku == sku), memory, hosts, drift, timebases,
         tolerance := ← Figure.ofJson (← Flock.J.get j "tolerance" "the auditor") "the auditor's tolerance",
         key := fun i => (keys.find? (·.1 == i)).map (·.2.1),
         scheme := fun i => (keys.find? (·.1 == i)).map (·.2.2) }

end Site

end Core
