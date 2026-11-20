module

public import Definitions.ResourceAccounting.Actual
public import Definitions.Core.Blocks
public import Definitions.Core.CircuitLevels
public import Definitions.Core.Period
public import Definitions.Pous.Model.StreamedWipeResidual

/-!
# Blocks and their properties: the interface from accounting to policy

A block is Core's `Verity.Block`: a window of ticks and the gates of a circuit it computes there (here the actual
circuit, `Placed`). Accounting establishes that the actual circuit divides into blocks nested by scale, and some of four
properties for each. The scales are `Verity.Partition`s of the actual circuit's computed gates (`security/Definitions/
Core/Partition.lean`: every computed gate in exactly one unit, input gates in none), nested by `Verity.Partition.Refines`:

* the **server-day blocks** and **hidden hardware's work**, a partition `outer` of the whole circuit (`kind u` says
  which, by location: a server-day block is at a location the hardware description lists, hidden hardware's work at
  leaves no listed location holds);
* the **inference blocks** and **free blocks** refining them, a partition `inner` with `inner.Refines outer`, a class
  per unit (`cls u`: inference, free, or `hidden`, under hidden hardware), the prover's label (`label u`) except that a
  unit labelled inference whose window starts before its epoch's challenge tick (`salt u`) is a free block;
* the **proof blocks** refine the inference blocks; they are the proof layer's own `Verity.Partition` and carry no block
  property here.

Coverage, disjointness and the tree shape are the partitions' own (`Partition.input_iff`, `Partition.gates` a function,
`Refines.sub`), so no field and no clause states them. Each block's location and window are fields of the run
(`locIn`/`winIn`, `locOut`/`winOut`), and each block's gates lie at its location during its window (`insideOut`,
`insideIn`, main's `Placed.region`): a server-day block is a `region`, so its size is bounded by `ComputeRate` and its
input and output by `Placed.ingress`/`egress`/`state` through main's lemmas, not assumed.

The four properties, each built on the actual circuit's definitions:
* **size**: its work at a pricing (`Placed.work`), in gates;
* **function**: a circuit that its gates, cut as a circuit (`Placed.cut`), implement (`Verity.Implements`). It is
  extensional: the cut may hold any further gates and input gates, so it bounds neither size nor input;
* **input** from named sources (`Source`), in bits (width);
* **output** to the outside world, in bits (width).

A monitoring period's runs (`Runs`) carry, per run, the actual circuit, its pricing and registered input gates, the
two partitions the prover committed before any coin, the registered numbers of each listed location (compute rate, the
wipe's label count and width, the storage the wipe does not cover, its attested boundary), and for each server-day block
the numbers accounting reads: the wipe's egress and challenge count, the network attester's ingress count for its
attested boundary and day (each tied to no attester's signed output yet), and the covert figure of its boundary
(assumed). `Realizes` is the feasible-set membership: every inference, free or hidden unit has every property claimed of
it, and every server-day block and its attested boundary's day block hold the three bounds (`BlockBounds`). Every accounting statement is `ExceptWith pr accepts shown ε`: the predicate `shown`
fails on an accepted run with probability at most `ε`. `UseAtLeast` states a lower bound on a used capacity, for every
tolerance set `M`, and the one conversion `FreeBlockAtMost` turns capacity minus use into a bound on the sum of the free
blocks directly inside each server-day block (for size, with `M`'s inference blocks); it alone carries the capacity
figure (`CapacityAtMost`) and the premise that declared work does not double as other work (`DeclaredWorkExclusive`).
-/

@[expose] public section

namespace ResourceAccounting

open Verity Finset
open scoped ENNReal

/-- **The source of a block's input**: registered information (the registered input gates `θ`); another block (placed
in the block's location during its window, outside it); the block's own past (placed in its location before its
window); the outside world (placed outside its location). The last three leave registered information out. -/
inductive Source where
  | registered
  | block
  | past
  | world
  deriving DecidableEq

/-- **A property of a block**, its bound in work at a pricing (size, gates) or in bits (input and output). An input or
output bound counts the block's declared ports too: the division fixes it before any coin. -/
inductive Property where
  | sizeAtMost (k : ℕ)
  | function (f : Circuit)
  | inputAtMost (sources : List Source) (k : ℕ)
  | outputAtMost (k : ℕ)

/-- **A block's committed size claim**, in gates: the bound of the first `sizeAtMost` in its claims list, `0` when the
list has none (the check refuses an inference block with none). -/
def claimedSize : List Property → ℕ
  | [] => 0
  | .sizeAtMost k :: _ => k
  | _ :: t => claimedSize t

/-- **The class of a unit of `inner`**: an inference block or a free block (inside a server-day block), or hidden (a unit
inside hidden hardware's work, which has no established property). -/
inductive Class where
  | inference
  | free
  | hidden
  deriving DecidableEq

/-- **A capacity of the hardware graph** (`security/Definitions/Registration/Physical.lean`: locations with a compute
rate `ComputeRate` and a memory size `Memory`, links with a capacity `Links`, under `LatencyAtMost`). Each is used at a
location or a link and counted in the actual circuit, with the same two lemma forms, a `UseAtLeast` and an
`inputAtMost`:
* `computeRate` at a location, counted as gates at the location in a span: use at least from PoUW; no input form.
* `memorySize` at a location, counted as wires crossing a tick at the location: use at least from PoUS; input at most
  from the memory wipe.
* `linkCapacity` of a link, counted as wires crossing the link in a span: use at least from PoUB, on the link from a
  device's memory to its chip; input at most from network attestation, on the network links. -/
inductive Capacity where
  | computeRate
  | memorySize
  | linkCapacity
  deriving DecidableEq

namespace Placed

variable {L : Type*} (A : Placed L)

open Classical in
/-- The gates of a source, for a block with gates `S`, location `X` and window `win`. -/
noncomputable def sourceGates (θ S : Finset (Fin A.N)) (X : Finset L) (win : Window) : Source → Finset (Fin A.N)
  | .registered => θ
  | .block => univ.filter fun g => g ∉ θ ∧ g ∉ S ∧ A.x g ∈ X ∧ A.t g ∈ win.ticks
  | .past => univ.filter fun g => g ∉ θ ∧ A.x g ∈ X ∧ A.t g < win.start
  | .world => univ.filter fun g => g ∉ θ ∧ A.x g ∉ X

open Classical in
/-- **Input** of block `b` (located at `X`) from `sources`, in bits: the width of the gates of those sources it reads. -/
noncomputable def blockInput (θ : Finset (Fin A.N)) (X : Finset L) (b : Verity.Block A.toCircuit)
    (sources : List Source) : ℕ :=
  A.width (A.into b.gates ∩ sources.toFinset.biUnion (A.sourceGates θ b.gates X b.win))

open Classical in
/-- **Output** of block `b` (located at `X`) to the outside world, in bits: the width of its gates that a gate placed
outside `X` reads. It counts gates, so it leaves out what the time a gate is read at can carry. -/
noncomputable def blockOutput (X : Finset L) (b : Verity.Block A.toCircuit) : ℕ :=
  A.width (b.gates.filter fun g => ∃ h, A.x h ∉ X ∧ g ∈ A.reads h)

/-- **Block `b` (located at `X`) has the property `p`** at pricing `π` and registered input gates `θ`. -/
noncomputable def Holds (π : Fin A.N → ℕ) (θ : Finset (Fin A.N)) (X : Finset L)
    (b : Verity.Block A.toCircuit) : Property → Prop
  | .sizeAtMost k => A.work π b.gates ≤ k
  | .function f => Verity.Implements f (A.cut b.gates)
  | .inputAtMost src k => A.blockInput θ X b src ≤ k
  | .outputAtMost k => A.blockOutput X b ≤ k

end Placed

/-- **`P` fails on an accepted run with probability at most `ε`**, `pr` the audit game's probability on runs, as
`Scheme.GetsBack` reads it. -/
def ExceptWith {Ω : Type} (pr : (Ω → Prop) → ℝ≥0∞) (accepts : Ω → Prop) (P : Ω → Prop) (ε : ℝ≥0∞) : Prop :=
  pr (fun r => accepts r ∧ ¬ P r) ≤ ε

/-- **A monitoring period's runs**, as accounting reads them: the audit game's probability `pr` (monotone, and
subadditive over finite unions), the verifier's acceptance, and each run's actual circuit, pricing, registered input
gates, and the two partitions the prover committed before any coin. `outer` partitions the whole circuit into server-day
blocks and hidden hardware (`kind`); `inner` refines it into units the prover labels inference, free or hidden
(`label`), a unit labelled hidden exactly when its server-day-or-hidden parent is hidden hardware (`labelHiddenIff`).
`kind` is determined by location: a top block is a server-day block exactly when its location is in the hardware
description's list (`kindIff`), which is registered information, the auditor's; a hidden block's location shares no leaf
with a listed one (`hiddenUnlisted`). A server-day block's window is one of the period's registered days (`winDay`), and
no two server-day blocks of a run share a location and a day (`oneBlock`), so a run has at most the listed locations
times the days of them (`serverDay_count_le`). A top block's location (`locOut`) and its day (`dayOf`) are fields of the
run, not the auditor's: `insideOut` ties them to the block's gates, and `kindIff` and `hiddenUnlisted` to the list. The
class every definition reads is `cls`, derived from the label and the epoch's challenge tick (`salt`). No server-day
block carries a claim: the run carries, for each, the numbers its three bounds read (`BlockBounds`). -/
structure Runs (L : Type*) where
  Run : Type
  pr : (Run → Prop) → ℝ≥0∞
  mono : ∀ p q : Run → Prop, (∀ r, p r → q r) → pr p ≤ pr q
  union : ∀ {κ : Type} (s : Finset κ) (p : κ → Run → Prop), pr (fun r => ∃ x ∈ s, p x r) ≤ ∑ x ∈ s, pr (p x)
  accepts : Run → Prop
  actual : Run → Placed L
  π : (r : Run) → Fin (actual r).N → ℕ
  θ : (r : Run) → Finset (Fin (actual r).N)
  registered : ℕ
  nOut : Run → ℕ
  outer : (r : Run) → Partition (actual r).toCircuit (nOut r)
  /-- **Whether a top block is a server-day block** (`true`) or hidden hardware's work (`false`): determined by whether
  its location is in the hardware description's list (`kindIff`). -/
  kind : (r : Run) → Fin (nOut r) → Bool
  /-- **Each top block's location**, a field of the run, not the auditor's: `insideOut` puts the block's gates there, and
  `kindIff` and `hiddenUnlisted` tie it to the list. -/
  locOut : (r : Run) → Fin (nOut r) → Finset L
  /-- **The locations the hardware description lists** (`Hardware.units` in `Registration/Physical.lean`), registered
  information for the period, the auditor's. -/
  listed : Set (Finset L)
  /-- A top block is a server-day block exactly when its location is listed. -/
  kindIff : ∀ r i, kind r i = true ↔ locOut r i ∈ listed
  /-- A hidden block's location shares no leaf with any listed location. -/
  hiddenUnlisted : ∀ r i, kind r i = false → ∀ X ∈ listed, Disjoint (locOut r i) X
  /-- **Each top block's window**, a field of the run; a server-day block's is one of the period's registered days
  (`winDay`). -/
  winOut : (r : Run) → Fin (nOut r) → Window
  /-- **The monitoring period's span** (`MonitoringPeriod.span`), registered information. -/
  period : TickSpan
  /-- The number of the period's days. -/
  nDays : ℕ
  /-- **The period's days**: a cut of its span into `nDays` consecutive spans, from its start to its stop, so they tile
  it; registered information, the auditor's. -/
  days : period.Cut nDays
  /-- **The day of each top block**, a field of the run, not the auditor's: `winDay` makes a server-day block's window
  that registered day. -/
  dayOf : (r : Run) → Fin (nOut r) → Fin nDays
  /-- A server-day block's window is one of the period's registered days, its `dayOf`. -/
  winDay : ∀ r i, kind r i = true →
    (winOut r i).start = (days.part (dayOf r i)).start ∧ (winOut r i).len = (days.part (dayOf r i)).len
  /-- No two server-day blocks of a run share a location and a day. -/
  oneBlock : ∀ r i j, kind r i = true → kind r j = true → locOut r i = locOut r j → dayOf r i = dayOf r j → i = j
  /-- **The hardware description's compute rate** of each location (`Hardware.compute`, `Φ`, in
  `Registration/Physical.lean`): work a tick, in gates at the run's pricing, as `ComputeRate` reads it; registered
  information for the period. -/
  rate : Finset L → ℕ
  /-- **PoUW's credited use** of each server-day block of a run, in gates at the run's pricing: the verifier's output for
  that run, a function of the registered claim, the proven share and the scheme's assumed factor from multiply-accumulates
  to gates; not a number the prover supplies. -/
  credited : (r : Run) → Fin (nOut r) → ℕ
  /-- **The tolerance**, in gates: the sampling design's, one number for the period. -/
  tol : ℕ
  /-- **The wipe's label count at each location** (`N`), in labels: the labels the served wipe fills there, from the
  hardware description; registered information for the period. -/
  wipeLabels : Finset L → ℕ
  /-- **The wipe's label width at each location** (`b`), in bits a label; registered information for the period. -/
  labelBits : Finset L → ℕ
  /-- **The storage at each location that the wipe does not cover**, in bits: the memory at that location outside the
  `wipeLabels · labelBits` bits the wipe fills; registered information for the period. At the served node it is all of
  the DRAM and the disks. -/
  uncovered : Finset L → ℕ
  /-- **The wipe's soundness exponent** (`s`), in bits: the wipe's bound holds at error `2^-s` plus the PRF's advantage;
  one stated number for the period. -/
  wipeError : ℕ
  /-- **The wipe's egress for each server-day block**, in bits: the bits that left the attested boundary during the wipe
  that opens the block, as `WipeInputAtMost` reads it. No attester's signed output is tied to it yet. -/
  wipeEgress : (r : Run) → Fin (nOut r) → ℕ
  /-- **The wipe's challenge count for each server-day block** (`k`), in challenges: the sequential challenges of the
  wipe that opens the block, which the wipe's row (`WipeInputAtMost`) reads its guarantee at. No attester's signed
  output is tied to it yet. -/
  wipeChallenges : (r : Run) → Fin (nOut r) → ℕ
  /-- **The attested boundary of each location**, a number naming it: the listed locations behind one network attester
  share it; registered information for the period. -/
  boundary : Finset L → ℕ
  /-- **The network attester's ingress count for each attested boundary and day**, in bits: what entered that boundary
  on that day, as `NetworkInputAtMost` reads it. No attester's signed output is tied to it yet. -/
  ingressCount : (r : Run) → ℕ → Fin nDays → ℕ
  /-- **The covert figure of each attested boundary**, in bits for the period: the bits the channels the network
  attester does not see carry into that boundary; one number per boundary, not the run's; assumed: no protocol stands
  behind it. `BlockBounds` and `check` add it whole to each day's bound, which is sound for each day and, summed over the
  period's days, counts it `nDays` times. -/
  covert : ℕ → ℕ
  nIn : (r : Run) → ℕ
  inner : (r : Run) → Partition (actual r).toCircuit (nIn r)
  ref : (r : Run) → (inner r).Refines (outer r)
  /-- **The prover's committed label of each unit**: inference, free or hidden. No definition reads it but `cls`. -/
  label : (r : Run) → Fin (nIn r) → Class
  locIn : (r : Run) → Fin (nIn r) → Finset L
  /-- **Each unit's window**, the committed partition's, so the prover's: its start tick is what the prover committed.
  `insideIn`, a field an instance owes, ties the unit's gates to it. -/
  winIn : (r : Run) → Fin (nIn r) → Window
  claimsIn : (r : Run) → Fin (nIn r) → List Property
  /-- A unit is labelled hidden exactly when its parent is hidden hardware. -/
  labelHiddenIff : ∀ r u, label r u = Class.hidden ↔ kind r ((ref r).parent u) = false
  /-- **The challenge tick of each unit's epoch**: the tick of the verifier's live coin that opens the epoch (PoUW's
  salt), public. -/
  salt : (r : Run) → Fin (nIn r) → ℕ
  /-- **A block is its place and time**: every server-day-or-hidden block's gates are computed at its location during
  its window (main's `Placed.region`), so a server-day block is a `region` and its size is bounded by `ComputeRate`, not
  assumed. -/
  insideOut : ∀ r i, (outer r).gates i ⊆ (actual r).region (locOut r i) ↑((winOut r i).ticks)
  /-- Every finer block too lies in its own location during its own window, so its input by source counts every gate at
  its location after its window (main's `region`). It lies inside its parent's region by `Refines` and `insideOut`. -/
  insideIn : ∀ r u, (inner r).gates u ⊆ (actual r).region (locIn r u) ↑((winIn r u).ticks)

namespace Runs

variable {L : Type*} (P : Runs L)

/-- **The class of unit `u` of `inner` in run `r`**, the one every definition, theorem and the check read: its label,
except that a unit labelled inference whose window starts before its epoch's challenge tick is a free block. It is a
function of what the prover committed (the label and the window) and of the verifier's coin's tick; no auditor-side
source for a unit's start tick is named here, so it bounds only what the prover committed. -/
def cls (r : P.Run) (u : Fin (P.nIn r)) : Class :=
  if P.label r u = Class.inference ∧ (P.winIn r u).start < P.salt r u then Class.free else P.label r u

/-- Server-day-or-hidden block `i` of run `r` as a `Verity.Block` (its window and its gates). -/
def outerBlock (r : P.Run) (i : Fin (P.nOut r)) : Verity.Block (P.actual r).toCircuit :=
  ⟨P.winOut r i, (P.outer r).gates i⟩

/-- Inference-or-free-or-hidden block `u` of run `r` as a `Verity.Block`. -/
def innerBlock (r : P.Run) (u : Fin (P.nIn r)) : Verity.Block (P.actual r).toCircuit :=
  ⟨P.winIn r u, (P.inner r).gates u⟩

/-- Block `u` of `inner` in run `r` has the property `p`. -/
noncomputable def HoldsIn (r : P.Run) (u : Fin (P.nIn r)) (p : Property) : Prop :=
  (P.actual r).Holds (P.π r) (P.θ r) (P.locIn r u) (P.innerBlock r u) p

/-- Block `i` of `outer` in run `r` has the property `p`. -/
noncomputable def HoldsOut (r : P.Run) (i : Fin (P.nOut r)) (p : Property) : Prop :=
  (P.actual r).Holds (P.π r) (P.θ r) (P.locOut r i) (P.outerBlock r i) p

open Classical in
/-- **The gates at unlisted locations** in run `r`: the computed gates placed at a leaf no listed location holds. Hidden
hardware's work is at most theirs (`hidden_work_le`). -/
noncomputable def hiddenGates (r : P.Run) : Finset (Fin (P.actual r).N) :=
  (P.actual r).computed.filter fun g => ∀ X ∈ P.listed, (P.actual r).x g ∉ X

/-- Block `u`'s size in run `r`, its work at the run's pricing, in gates. -/
def sizeIn (r : P.Run) (u : Fin (P.nIn r)) : ℕ := (P.actual r).work (P.π r) ((P.inner r).gates u)

/-- **Block `i`'s compute capacity** in run `r`, in gates at the run's pricing: its location's rate times its window's
length, for a server-day block its day's (`winDay`). -/
def capacity (r : P.Run) (i : Fin (P.nOut r)) : ℕ := P.rate (P.locOut r i) * (P.winOut r i).len

/-- Block `i`'s size in run `r`, its work at the run's pricing, in gates. -/
def sizeOut (r : P.Run) (i : Fin (P.nOut r)) : ℕ := (P.actual r).work (P.π r) ((P.outer r).gates i)

/-- The units of `inner` directly inside `outer` block `i`. -/
def children (r : P.Run) (i : Fin (P.nOut r)) : Finset (Fin (P.nIn r)) :=
  univ.filter fun u => (P.ref r).parent u = i

/-- The inference blocks directly inside `outer` block `i`. -/
def infChildren (r : P.Run) (i : Fin (P.nOut r)) : Finset (Fin (P.nIn r)) :=
  univ.filter fun u => (P.ref r).parent u = i ∧ P.cls r u = Class.inference

/-- The free blocks directly inside `outer` block `i`. -/
def freeChildren (r : P.Run) (i : Fin (P.nOut r)) : Finset (Fin (P.nIn r)) :=
  univ.filter fun u => (P.ref r).parent u = i ∧ P.cls r u = Class.free

/-- **The wipe's bound on block `i`'s input from its own past**, in bits, at egress `E` (bits) and challenge count `k`:
`E r i`, plus the wipe residual (`Pous.StreamedWipe.wipeResidual`) at its location's label count and label width, `k r i`
challenges and the soundness exponent, plus its location's storage the wipe does not cover. -/
def pastBound (E k : (r : P.Run) → Fin (P.nOut r) → ℕ) (r : P.Run) (i : Fin (P.nOut r)) : ℕ :=
  E r i + Pous.StreamedWipe.wipeResidual (P.wipeLabels (P.locOut r i)) (k r i) (P.labelBits (P.locOut r i)) P.wipeError +
    P.uncovered (P.locOut r i)

/-- **The wipe's guarantee applies at block `i`'s numbers**: its location's label count `N` is positive, its challenge
count `k r i` is positive, and the wipe residual is below the `N · b` bits the wipe fills; `Pous.Guarantees.
StreamedWipeResidual` says nothing about a wipe that fails one of the three. `check` compares it for every server-day
block, so `check_wipeApplies` gives it for every server-day block of a run that passes. -/
abbrev WipeApplies (k : (r : P.Run) → Fin (P.nOut r) → ℕ) (r : P.Run) (i : Fin (P.nOut r)) : Prop :=
  0 < P.wipeLabels (P.locOut r i) ∧ 0 < k r i ∧
    Pous.StreamedWipe.wipeResidual (P.wipeLabels (P.locOut r i)) (k r i) (P.labelBits (P.locOut r i)) P.wipeError <
      P.wipeLabels (P.locOut r i) * P.labelBits (P.locOut r i)

/-- The attested boundary of block `i`'s location. -/
def bnd (r : P.Run) (i : Fin (P.nOut r)) : ℕ := P.boundary (P.locOut r i)

/-- **The server-day blocks of run `r` behind attested boundary `b` on day `d`.** -/
def boundaryBlocks (r : P.Run) (b : ℕ) (d : Fin P.nDays) : Finset (Fin (P.nOut r)) :=
  univ.filter fun i => P.kind r i = true ∧ P.bnd r i = b ∧ P.dayOf r i = d

open Classical in
/-- **Attested boundary `b`'s day block on day `d`**: the union of its server-day blocks that day, over the day's span. -/
noncomputable def boundaryBlock (r : P.Run) (b : ℕ) (d : Fin P.nDays) : Verity.Block (P.actual r).toCircuit :=
  ⟨⟨(P.days.part d).start, (P.days.part d).len⟩, (P.boundaryBlocks r b d).biUnion (P.outer r).gates⟩

open Classical in
/-- The location of attested boundary `b`'s day block on day `d`: the union of its server-day blocks' locations. It holds
only the locations with a server-day block that day, so a listed location behind the boundary with no server-day block
that day counts as outside world. -/
noncomputable def boundaryLoc (r : P.Run) (b : ℕ) (d : Fin P.nDays) : Finset L :=
  (P.boundaryBlocks r b d).biUnion (P.locOut r)

/-- Attested boundary `b`'s day block on day `d` in run `r` has the property `p`. -/
noncomputable def HoldsBoundary (r : P.Run) (b : ℕ) (d : Fin P.nDays) (p : Property) : Prop :=
  (P.actual r).Holds (P.π r) (P.θ r) (P.boundaryLoc r b d) (P.boundaryBlock r b d) p

/-- **The wipe's bounds summed over attested boundary `b`'s server-day blocks on day `d`**, in bits. -/
def boundaryPastBound (r : P.Run) (b : ℕ) (d : Fin P.nDays) : ℕ :=
  ∑ j ∈ P.boundaryBlocks r b d, P.pastBound P.wipeEgress P.wipeChallenges r j

/-- **Every server-day block of run `r` and its attested boundary's day block hold the three bounds accounting
establishes** (old name `ServerDayBounds`, whose third clause bounded the server-day block itself): the block's size at
most its capacity (gates at the run's pricing), its input from its own past at most the wipe's bound (`pastBound`,
bits), and its attested boundary's day block's input from the outside world at most the boundary's ingress count that
day plus the boundary's covert figure (bits). A server behind a shared attested boundary has its input from the outside
world bounded only through the union: a policy can no longer conclude anything about that one server's input from the
outside world, which counts what the other servers behind the boundary send it. What these bounds rest on:
`wipeEgress`, `wipeChallenges` and `ingressCount` are tied to no attester's signed output yet; the covert figure is
assumed; the check's limit on hidden hardware's work (`lim.hidden`) is assumed and no protocol holds it; and no instance
of `Runs` exists, so nothing about a real run is proved by this file. -/
noncomputable def BlockBounds (r : P.Run) : Prop :=
  ∀ i, P.kind r i = true →
    P.HoldsOut r i (.sizeAtMost (P.capacity r i)) ∧
      P.HoldsOut r i (.inputAtMost [.past] (P.pastBound P.wipeEgress P.wipeChallenges r i)) ∧
      P.HoldsBoundary r (P.bnd r i) (P.dayOf r i)
        (.inputAtMost [.world] (P.ingressCount r (P.bnd r i) (P.dayOf r i) + P.covert (P.bnd r i)))

/-- **The run realizes its claims** (the feasible-set membership for `r`): `θ` is input gates of at most the registered
width, the pricing charges every computed gate, every unit of `inner` has each property claimed of it, and every
server-day block and its attested boundary's day block hold the three bounds (`BlockBounds`). -/
noncomputable def Realizes (r : P.Run) : Prop :=
  (∀ a ∈ P.θ r, (P.actual r).isInput a = true) ∧ (P.actual r).width (P.θ r) ≤ P.registered ∧
    (∀ g ∈ (P.actual r).computed, 1 ≤ P.π r g) ∧
    (∀ u, ∀ p ∈ P.claimsIn r u, P.HoldsIn r u p) ∧ P.BlockBounds r

/-- **The run realizes its claims outside a tolerance set** `M` of inference blocks (the feasible-set membership with a
tolerance): `θ` is registered, the pricing charges every computed gate, every claim of a unit that is not an inference
block holds, every inference block outside `M` holds its claims, every server-day block and its attested boundary's day
block hold the three bounds (`BlockBounds`), and `M` holds only inference blocks. The sampled proofs and
`DeclaredBlocksRun` give the inference claims only outside such an `M`, so the chain reads this, not `Realizes`. **The
tolerance is a term, not an unbounded quantifier**: what caps `M`'s actual work is not here but in `FreeWorkAtMost` (the
free blocks' sizes and `M`'s inference blocks' sizes at most capacity − credited use + tolerance) whose right-hand side
the check caps at `lim.freeSize`; taking `M = all inference blocks` then forces their total work under `lim.freeSize`, not
nothing (`excluded_run`). `FreeWorkAtMost` is deliberately not folded in here: it comes from PoUW's row through the
conversion (`freeWorkAtMost_of_conversion`), not from membership in the feasible set. `Realizes` implies `RealizesExcept
∅` (`realizes_realizesExcept`). -/
noncomputable def RealizesExcept (r : P.Run) (M : Finset (Fin (P.nIn r))) : Prop :=
  (∀ a ∈ P.θ r, (P.actual r).isInput a = true) ∧ (P.actual r).width (P.θ r) ≤ P.registered ∧
    (∀ g ∈ (P.actual r).computed, 1 ≤ P.π r g) ∧
    (∀ u, P.cls r u = Class.inference → u ∉ M → ∀ p ∈ P.claimsIn r u, P.HoldsIn r u p) ∧
    (∀ u, P.cls r u ≠ Class.inference → ∀ p ∈ P.claimsIn r u, P.HoldsIn r u p) ∧
    P.BlockBounds r ∧
    (∀ u ∈ M, P.cls r u = Class.inference)

/-- **The free and tolerance work is at most capacity − credited use + tolerance**: inside every server-day block `i`,
the free blocks' sizes and the tolerance set `M`'s inference blocks' sizes together are at most `i`'s capacity less
PoUW's credited use of `i` plus the tolerance (gates at the run's pricing). `check` caps that right-hand side at
`lim.freeSize`. It follows, for the `M` the declared row gives, from PoUW's row, the compute capacity and `M`'s declared
size at most the tolerance (`freeWorkAtMost_of_conversion`), never from `RealizesExcept`. -/
def FreeWorkAtMost (r : P.Run) (M : Finset (Fin (P.nIn r))) : Prop :=
  ∀ i, P.kind r i = true →
    (∑ u ∈ P.freeChildren r i, P.sizeIn r u) + (∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u)
      ≤ P.capacity r i - P.credited r i + P.tol

/-- **The run's pooled free bound**, in gates at the run's pricing: its server-day blocks' capacities summed less their
credited uses summed, plus the tolerance once. -/
def pooledFreeBound (r : P.Run) : ℕ :=
  (∑ i ∈ univ.filter (fun i => P.kind r i = true), P.capacity r i) -
    (∑ i ∈ univ.filter (fun i => P.kind r i = true), P.credited r i) + P.tol

/-- **The free and tolerance work, pooled over the run, is at most the pooled free bound**: summed over the run's
server-day blocks, the free blocks' sizes and the tolerance set `M`'s inference blocks' sizes together are at most
`pooledFreeBound r` (gates). `check` does not read it: which limit the policy states, per server-day block or pooled over
the period, is training's (`InferenceOnly.InferenceOnlyTheorem`, open). The pooled form matches the design: the sampling
design's tolerance is one number for the period, and `FreeWorkAtMost` summed over the server-day blocks counts it once for
each of them. -/
def FreeWorkAtMostPooled (r : P.Run) (M : Finset (Fin (P.nIn r))) : Prop :=
  ∑ i ∈ univ.filter (fun i => P.kind r i = true),
      ((∑ u ∈ P.freeChildren r i, P.sizeIn r u) + ∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u)
    ≤ P.pooledFreeBound r

/-- **Every accepted run realizes its claims** (open): the bridge from the statements to the feasible set. -/
def Fits : Prop := ∀ r, P.accepts r → P.Realizes r

/-- **The protocol's guarantee, read on the period's runs** (open, one per instance): the guarantee `G`, at this
deployment, bounds by `ε` the probability that an accepted run's own conclusion `shown` fails. -/
def RunReads (G : Prop) (shown : P.Run → Prop) (ε : ℝ≥0∞) : Prop := G → ExceptWith P.pr P.accepts shown ε

/-- **Use at least a capacity** (`what`, one of the three): for every set `M` of `inner`'s units and inside every
server-day block `i`, at least `b r M i` of `what`'s capacity at `i`'s location or link was used, in the quantity
`use r M i`, except with probability at most `ε`. `M` is the tolerance set: PoUW's use is the work outside `M` and its
bound is lowered by `M`'s declared size (`PouwUseAtLeast`); PoUS's and PoUB's use and bound do not read `M`. PoUW, PoUS
and PoUB are its three instances, over `computeRate`, `memorySize` and `linkCapacity`; the one conversion
`FreeBlockAtMost` turns each into a bound on a free-block quantity (its size with `M`'s inference blocks, its input from
its own past, its input from memory). -/
def UseAtLeast (what : Capacity) (use b : (r : P.Run) → Finset (Fin (P.nIn r)) → Fin (P.nOut r) → ℕ) (ε : ℝ≥0∞) :
    Prop :=
  ExceptWith P.pr P.accepts (fun r => ∀ M i, P.kind r i = true → b r M i ≤ use r M i) ε

/-- **The hardware description** for `what`: the number the hardware graph gives for a server-day block's capacity of
`what` at its location or link (`ComputeRate`'s `Φ`, `Memory`'s `M`, `Links`' `β` in `Registration/Physical.lean`),
`enclosing r i ≤ cap r i`. For `computeRate` the figure is not a premise of its own: `capacityAtMost_compute` derives it
for every server-day block from `ComputeRate` over the listed locations, a server-day block's location being listed. It
stays here for `memorySize` and `linkCapacity`, what the hardware graph gives, an assumption nothing in Verity tests
(owner none: the hardware description). -/
def CapacityAtMost (what : Capacity) (enclosing cap : (r : P.Run) → Fin (P.nOut r) → ℕ) : Prop :=
  ∀ r, ∀ i, P.kind r i = true → enclosing r i ≤ cap r i

/-- **Declared work does not double as other work**: in an accepted run, for every set `M` of `inner`'s units and inside
every server-day block `i`, the free quantity `free r M i` and the use `use r M i` add up to at most `i`'s own quantity
`enclosing r i`. For size, with the free quantity the free blocks' and `M`'s inference blocks' work and the use the work
of the inference blocks outside `M`, it follows from the partitions
(`ResourceAccounting.SecurityProofs.BlockProperties.sizeExclusive`); for other quantities it is open. -/
def DeclaredWorkExclusive (free use : (r : P.Run) → Finset (Fin (P.nIn r)) → Fin (P.nOut r) → ℕ)
    (enclosing : (r : P.Run) → Fin (P.nOut r) → ℕ) : Prop :=
  ∀ r, P.accepts r → ∀ M i, P.kind r i = true → free r M i + use r M i ≤ enclosing r i

/-- **The one conversion: capacity minus use bounds the free blocks.** For any capacity `what` (compute rate, memory
size or link), for every set `M` and inside every server-day block `i`: a use lower bound, the capacity figure and
exclusivity give the free quantity `free r M i` at most `cap r i − b r M i`, except with the use's probability. Its three
instances give the free blocks' size with `M`'s inference blocks' (from PoUW over `computeRate`), the free block's
unregistered input from its own past (from PoUS over `memorySize`) and from memory (from PoUB over `linkCapacity`).
Proved by `ResourceAccounting.SecurityProofs.BlockProperties.freeBlockAtMost`. -/
def FreeBlockAtMost : Prop :=
  ∀ (what : Capacity) (free use b : (r : P.Run) → Finset (Fin (P.nIn r)) → Fin (P.nOut r) → ℕ)
    (enclosing cap : (r : P.Run) → Fin (P.nOut r) → ℕ) (ε : ℝ≥0∞),
    P.UseAtLeast what use b ε → P.CapacityAtMost what enclosing cap → P.DeclaredWorkExclusive free use enclosing →
      ExceptWith P.pr P.accepts (fun r => ∀ M i, P.kind r i = true → free r M i ≤ cap r i - b r M i) ε

end Runs

end ResourceAccounting
