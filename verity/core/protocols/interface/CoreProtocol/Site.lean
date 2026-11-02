import Lean.Data.Json
import Flock.Canon
import Flock.Hash
import CoreProtocol.Ratio

/-!
# The Site in code

The spec's `Verity.Site` (`Security/Definitions/Core/Site.lean`) as code, on Lean core and C-Flock's canonical JSON and
SHA-512: the registered hardware, folded from the registration log (`Site.ofLog`), the check each entry passes
(`Registration.Entry.refusal`), and the views the protocols read. Its checks are the spec's, clause for clause and in
the spec's order, over this file's own types, each decidable; `Properties.Core.Site` states that each is the spec's, and
`verity.core.site` is its Python mirror, held to it by the difftest (`site-difftest`). A sheet carries its device's
figures (`DeviceFigures`, the spec's `Definitions/Core/DeviceFigures.lean`), nonnegative rationals (`Figure`, the spec's
`ℚ≥0`).

The digests are parameters as in the spec (`Registration.Digests`); the deployed ones (`Registration.sha512`) are
`SHA-512(tag ‖ 0 ‖ canon(j))` over the JSON `verity.core.site` writes, under `verity/registration/{site,sheet,genesis}/v1`,
a rational written as `Core.Ratio` writes one. The views (`Site.sheetOf`, `device`, `node`, `tiers`, `boundary`, `units`,
`registered`, `host`) are functions of the site alone, but `device`, which reads the auditor's sheet too.
-/

namespace Core

namespace Site

open Lean (Json)

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

/-- The kinds of parent a location of each kind may have, `none` being the root. -/
def Kind.parents : Kind → List (Option Kind)
  | .site => [none]
  | .host => [some .site]
  | .device => [some .host]
  | .memory => [some .device, some .host]
  | .certifier | .challenger => [some .site]
  | .outside => []

/-- The pairs of kinds a link may join, in either order. -/
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

/-- A nonnegative rational, the spec's `ℚ≥0`: a clock in Hz, a peak rate in MACs a second, a tolerance, or a tick in
seconds. -/
abbrev Figure : Type := {q : Rat // 0 ≤ q}

/-- The larger of two figures. -/
def Figure.max (a b : Figure) : Figure := if a.1 ≤ b.1 then b else a

/-- The operand class of a tensor-core MAC (`Verity.OperandClass`): int8 operands drawn uniformly over [-128, 127],
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
in the device model's terms: bits from device memory to the chip a tick (`bw`), bits of on-chip state (`chip`) and bytes
of device memory (`mem`). -/
structure Sheet where
  figures : DeviceFigures
  bw : Nat
  chip : Nat
  mem : Nat
  deriving DecidableEq

/-- A sheet's SKU, its figures'. -/
def Sheet.sku (k : Sheet) : String := k.figures.sku

/-- A location, and who owns it. -/
structure Loc where
  id : String
  kind : Kind
  parent : Option String := none
  owner : Owner := .prover
  sheet : String := ""
  tier : Option Tier := none
  bytes : Nat := 0
  key : String := ""
  deriving DecidableEq, Repr

/-- A link between two ends, with its capacity `β` in bits a tick each way and its latency in ticks. -/
structure Link where
  id : String
  ends : String × String
  β : Nat
  latency : Nat
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
SKU's; the tolerance registration allows around an estimate; and the key of each of its own devices. -/
structure Auditor where
  sheet : String → Option Sheet
  tolerance : Figure
  key : String → Option String

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

/-- `kind`: parents and tiers by kind, links between kinds a link may join at a positive capacity, and each certifier
splitting exactly one uplink. -/
def Kinds : Prop :=
  (∀ x ∈ e.locs, x.parent.bind (S.add e).kind? ∈ x.kind.parents ∧ (x.kind = .memory ↔ x.tier.isSome)) ∧
    (∀ k ∈ e.links, k.ends.1 ≠ k.ends.2 ∧ (∃ p ∈ Kind.links, (S.add e).Joins k p.1 p.2) ∧ 0 < k.β) ∧
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

instance instDecidableFresh : Decidable (Fresh D S e) := by unfold Fresh; infer_instance
instance instDecidableRefs : Decidable (Refs S e) := by unfold Refs; infer_instance
instance instDecidableKinds : Decidable (Kinds S e) := by unfold Kinds; infer_instance
instance instDecidableSheets : Decidable (Sheets D A S e) := by unfold Sheets; infer_instance
instance instDecidableFigures : Decidable (Figures A e) := by unfold Figures; infer_instance
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
  else if ¬Figures A e then some .figures
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

/-- **The check of a log**: the Site it folds to, or the first entry refused, with its index and refusal. -/
def check (D : Digests) (A : Auditor) (c : Composite) (log : Registration) : Except (Nat × Refusal) Site :=
  go 0 ⟨[], [], []⟩ (D.genesis c) log
where
  go : Nat → Site → String → Registration → Except (Nat × Refusal) Site
    | _, S, _, [] => .ok S
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

/-- **Device `d`'s model** (`Site.device`), read off its sheet `k` and the auditor's sheet `a` for `k`'s SKU: `k`'s
on-chip state and bandwidth, `k`'s device memory in bits, and as its tensor-core MACs a tick the highest peak rate of
the figures the bound reads (`DeviceFigures.bounding`) over a tick of `tick` seconds, rounded up. No sheet states CUDA-core
ops or oracle queries a tick, or side channels, so they are parameters. `none` when `d` names no registered sheet or the
auditor has none for its SKU. -/
def device (digest : Sheet → String) (A : Registration.Auditor) (d : String) (tick : Figure) {ι : Type}
    (cuda oracle : Nat) (cap : ι → Nat) : Option (DeviceModel ι) :=
  (S.sheetOf digest d).bind fun k => (A.sheet k.sku).map fun a =>
    let f := k.figures.bounding (A.estimate a)
    let peak := OperandClass.all.foldl (fun m c => max m (f.peakRate c).1) 0
    { chip := k.chip
      mem := 8 * k.mem
      rate := fun
        | .tc => (tick.1 * peak).ceil.toNat
        | .cuda => cuda
        | .oracle => oracle
      bw := k.bw
      cap := cap }

/-- The host a location is in: itself when it is a host. -/
def host (i : String) : Option String :=
  (S.loc? i).bind fun x => if x.kind = .host then some x.id else S.host? x

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

/-- The boundary links with an end at host `h`, or every boundary link. -/
def boundary (h : Option String) : List Link :=
  S.links.filter fun k => S.Boundary k ∧ (h.all fun h => h = k.ends.1 ∨ h = k.ends.2)

end Site

/-! ## The JSON `verity.core.site` writes, and the deployed digests -/

namespace Site

open Lean (Json)
open Flock (V)

def SITE : String := "verity/registration/site/v1"
def SHEET : String := "verity/registration/sheet/v1"
def GENESIS : String := "verity/registration/genesis/v1"

/-- `SHA-512(tag ‖ 0 ‖ canon(j))` as hex. Every JSON these types write holds integers and strings only, which
`Flock.canon` always writes, so the fallback is never taken. -/
def tagged (tag : String) (j : Json) : String :=
  Flock.hexOf (Flock.Sha512.hash (tag.toUTF8.push 0 ++ ((Flock.canon j).toOption.getD "").toUTF8))

def Kind.name : Kind → String
  | .site => "site" | .host => "host" | .device => "device" | .memory => "memory" | .certifier => "certifier"
  | .challenger => "challenger" | .outside => "outside"

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
  | .uncertified => "uncertified" | .challengerReach => "challenger-reach"

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

def Loc.toJson (x : Loc) : Json :=
  Json.mkObj [("bytes", Lean.toJson x.bytes), ("id", .str x.id), ("key", .str x.key), ("kind", .str x.kind.name),
    ("owner", .str x.owner.name), ("parent", optJson .str x.parent), ("sheet", .str x.sheet),
    ("tier", optJson (fun t : Tier => .str t.name) x.tier)]

def Link.toJson (k : Link) : Json :=
  Json.mkObj [("beta", Lean.toJson k.β), ("ends", .arr #[.str k.ends.1, .str k.ends.2]), ("id", .str k.id),
    ("latency", Lean.toJson k.latency)]

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

def Loc.ofJson (j : Json) : V Loc := do
  exactly j ["bytes", "id", "key", "kind", "owner", "parent", "sheet", "tier"] "a location"
  let get := fun k => Flock.J.get j k "a location"
  pure { id := ← Flock.J.str j "id" "a location",
         kind := ← byName [.site, .host, .device, .memory, .certifier, .challenger, .outside] Kind.name (← get "kind")
           "a location's kind",
         parent := ← optOf (Flock.J.asStr · "a location's parent") (← get "parent"),
         owner := ← byName [.prover, .auditor] Owner.name (← get "owner") "a location's owner",
         sheet := ← Flock.J.str j "sheet" "a location",
         tier := ← optOf (byName [.hbm, .dram, .disk] Tier.name · "a location's tier") (← get "tier"),
         bytes := ← Flock.J.nat j "bytes" "a location", key := ← Flock.J.str j "key" "a location" }

def Link.ofJson (j : Json) : V Link := do
  exactly j ["beta", "ends", "id", "latency"] "a link"
  let ends ← Flock.J.arr j "ends" "a link"
  let #[a, b] := ends | throw "a link: ends: not two ends"
  pure { id := ← Flock.J.str j "id" "a link", ends := (← Flock.J.asStr a "a link's end", ← Flock.J.asStr b "a link's end"),
         β := ← Flock.J.nat j "beta" "a link", latency := ← Flock.J.nat j "latency" "a link" }

def Registration.Entry.ofJson (j : Json) : V Registration.Entry := do
  exactly j ["format", "links", "locs", "previous", "sheets"] "an entry"
  if (← Flock.J.str j "format" "an entry") != SITE then throw s!"an entry: format is not {SITE}"
  let sheets ← (← Flock.J.arr j "sheets" "an entry").toList.mapM Sheet.ofJson
  let locs ← (← Flock.J.arr j "locs" "an entry").toList.mapM Loc.ofJson
  let links ← (← Flock.J.arr j "links" "an entry").toList.mapM Link.ofJson
  pure (.site (← Flock.J.str j "previous" "an entry") ⟨sheets, locs, links⟩)

def Registration.Composite.ofJson (j : Json) : V Registration.Composite :=
  byName [.onsite, .remote] Registration.Composite.name j "the log's composite"

/-- A log as `Registration.to_json` writes it: its composite and its entries. -/
def Registration.ofJson (j : Json) : V (Registration.Composite × Registration) := do
  exactly j ["composite", "entries"] "the log"
  pure (← Registration.Composite.ofJson (← Flock.J.get j "composite" "the log"),
    ← (← Flock.J.arr j "entries" "the log").toList.mapM Registration.Entry.ofJson)

/-- The auditor as `Auditor.to_json` writes it: its devices' keys, its data sheets, the first under an SKU or a device
being the one the check reads, and its tolerance. -/
def Registration.Auditor.ofJson (j : Json) : V Registration.Auditor := do
  exactly j ["keys", "sheets", "tolerance"] "the auditor"
  let keys ← (← Flock.J.arr j "keys" "the auditor").toList.mapM fun k => do
    exactly k ["id", "key"] "a key"
    pure (← Flock.J.str k "id" "a key", ← Flock.J.str k "key" "a key")
  let sheets ← (← Flock.J.arr j "sheets" "the auditor").toList.mapM Sheet.ofJson
  pure { sheet := fun sku => sheets.find? (·.sku == sku),
         tolerance := ← Figure.ofJson (← Flock.J.get j "tolerance" "the auditor") "the auditor's tolerance",
         key := fun i => (keys.find? (·.1 == i)).map (·.2) }

end Site

end Core
