module

public import Definitions.Compose.Blocks
public import Definitions.Pouw.PearlC.TileSample
public import Definitions.ResourceAccounting.Network.AttesterScheme
public import Definitions.Registration.Guarantees
public import Definitions.InferenceOnly

/-!
# The protocols' block statements, the inference-only policy, and the chain

Each protocol's instance of the form in `Definitions/ResourceAccounting/BlockProperties.lean` takes its guarantee as a
hypothesis, named where this layer may import it, and the open premises that tie it to the actual circuit.

* **PoUW** (`PouwUseAtLeast`): use at least, in size, for every tolerance set, from `Pouw.PearlC.GGapSampledDraw`,
  through `SpendIsWork`.
* **The declared blocks run** (`DeclaredBlocksRun`): every inference block outside a set whose committed size claims sum
  to at most the tolerance implements its declared block, at a size and input at most stated. It is owed an argument
  from PoUW's coverage of the declared work, with the uncovered share and the incorrect proof blocks the sample missed
  counted in the free block.
* **The free work bound** (`FreeWorkAtMost`): the two rows above, with the compute capacity, give it: the free and
  tolerance work in each server-day block is at most capacity − credited use + tolerance, the number `check` compares
  with `lim.freeSize`.
* **PoUB** (`PoubUseAtLeast`) and **PoUS** (`PousUseAtLeast`): open and not placed; no term of the inference-only bound
  reads memory reads or memory held.
* **The memory wipe** (`WipeInputAtMost`): input at most from its own past, for every server-day block, from
  `Pous.Guarantees.StreamedWipeResidual` at the location's registered label count and width and the wipe's challenge
  count, at one wipe's error times the period's wipes.
* **Network attestation** (`NetworkInputAtMost`, `NetworkOutputAtMost`): input at most for each attested boundary's day
  block from `AttesterOnEveryLink` and the attester's ingress count for that boundary and day; output at most for each
  attested boundary's day block from `NetTiming.Attester.PhaseGetsBack` and the record's unexplained bits.
* **Registration** (`RegistrationBinds`): every registered input gate opens a root of the log, from
  `Verity.Guarantees.RegistrationSound`.
* **The proof layer** is `Verity.Scheme.GetsBack` itself and gives no block property here.

The policy is a condition on declarations (`InferenceOnly.Policy`), read of an actual circuit with its registered input
gates (`InferenceOnly.Compliant`); `PolicyTheorem` is that every run whose blocks pass the check and realize their
claims is compliant. `Chain` is the composition, proved by
`ResourceAccounting.SecurityProofs.BlockProperties.chain`.
-/

@[expose] public section

open Verity Finset
open scoped ENNReal

namespace ResourceAccounting.Runs

variable {L : Type*} (P : Runs L)

/-! ## PoUW -/

/-- **The spend is work outside `M` plus `M`'s declared size** (open, compute-accounting; no verifier checks it today). In
an accepted run whose spend clears the claim, for every set `M` of `inner`'s units and inside every server-day block `i`,
PoUW's credited use `credited r i` (gates at `π` at fetch = 0, not proved otherwise) is at most the work of the
inference blocks directly inside `i` outside `M`, plus the committed size claims (`claimedSize`) of the inference blocks
directly inside `i` in `M`, for a device that receives no salt-dependent word during the window. At `M` = every inference
block it says that a block's credit is at most its committed size claim, for a device that receives no salt-dependent
word during the window. The statement carries no hypothesis on the device, so for any other device it is a premise not
expected to hold. By the class's definition (`cls`), an inference block's window starts at or after its epoch's
challenge tick. The credited use is fixed before the sampling coins, by the committed claim (the epoch's salt is a live
coin that comes before the work). Stated with `+` on the right rather than `−` on the left, so no truncated subtraction
enters it. -/
def SpendIsWork (shown : P.Run → Prop) : Prop :=
  ∀ r, P.accepts r → shown r → ∀ (M : Finset (Fin (P.nIn r))) i, P.kind r i = true →
    P.credited r i ≤ (∑ u ∈ (P.infChildren r i).filter (· ∉ M), P.sizeIn r u) +
      ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u)

/-- **PoUW (Pearl+): use at least, in size.** From `Pouw.PearlC.GGapSampledDraw`: the accepted adversary's spend is at
least a share of the claimed work. Open: `RunReads` and `SpendIsWork`. For every set `M`, its use is the work of the
inference children outside `M`, and its bound is the run's credited use `credited r i` less the committed size claims of
`M`'s inference children; the partitions keep that use apart from the free blocks and `M`'s blocks (`sizeExclusive`), so
it feeds the conversion. A shape: its content is its premises. -/
def PouwUseAtLeast {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : Pouw.PearlC.CostModel Q R S) (Pr : Pouw.PearlC.Protocol Q R S) (TR : Pouw.PearlC.TileRules Q R S)
    (D : Pouw.PearlC.Layout → Prop) (Wd : Pouw.PearlC.Layout → ℕ → Pouw.PearlC.Decl → ℚ) (g : ℝ) (δ : ℕ → ℕ → ℝ)
    (shown : P.Run → Prop) (ε : ℝ≥0∞) : Prop :=
  Pouw.PearlC.GGapSampledDraw CM Pr TR D Wd g δ →
    P.RunReads (Pouw.PearlC.GGapSampledDraw CM Pr TR D Wd g δ) shown ε → P.SpendIsWork shown →
      P.UseAtLeast Capacity.computeRate
        (fun r M i => ∑ u ∈ (P.infChildren r i).filter (· ∉ M), P.sizeIn r u)
        (fun r M i => P.credited r i - ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u)) ε

/-! ## The declared blocks run -/

/-- **The declared blocks run** (open, compute-accounting with soundness): every inference block `u` (its window starts
at or after its epoch's challenge tick, by the class's definition `cls`), outside a set `M` of inference blocks whose
committed size claims summed (`∑ u ∈ M, claimedSize (claimsIn u)`, gates) are at most the run's tolerance `P.tol`
(gates), implements its declared block `f u`, at size at most `c u` gates and input at most `k u` bits from its own past,
another block and the outside world, except with probability at most `ε`. The declared size of a block is its committed
size claim, read from its claims list, which the check caps at `lim.c`; it is no parameter of the row. Nothing in Verity
proves that the prover ran the declared circuit: the proofs show committed values consistent with it. A unit labelled
inference whose window starts before its epoch's challenge tick is a free block, so it is in no row here and its work
is under the free bound. The set `M` is the tolerance set the chain reads through `RealizesExcept`; the unregistered
information the incorrect proof blocks in `M` pass into correct ones (the injected information) is the `.block` source
of the inference blocks' input here. The tolerance binds the actual work through `FreeWorkAtMost`: `M`'s inference blocks
sit inside the free block, counted at their actual size, so `feasible_work_le` reads them. The row gives `.function`,
`.sizeAtMost` and `.inputAtMost`; that every claim on an inference block is one of these three is
`InferenceClaimsDeclared`. -/
def DeclaredBlocksRun (f : (r : P.Run) → Fin (P.nIn r) → Circuit)
    (c k : (r : P.Run) → Fin (P.nIn r) → ℕ) (ε : ℝ≥0∞) : Prop :=
  ExceptWith P.pr P.accepts
    (fun r => ∃ M ⊆ (univ.filter fun u => P.cls r u = Class.inference),
      (∑ u ∈ M, claimedSize (P.claimsIn r u)) ≤ P.tol ∧
      ∀ u ∈ (univ.filter fun u => P.cls r u = Class.inference) \ M,
        P.HoldsIn r u (.function (f r u)) ∧ P.HoldsIn r u (.sizeAtMost (c r u)) ∧
          P.HoldsIn r u (.inputAtMost [.past, .block, .world] (k r u))) ε

/-! ## PoUS and PoUB

PoUW, PoUS and PoUB are the three instances of `UseAtLeast`, over the three capacities of the hardware graph (compute
rate, memory size, link capacity). Each feeds the one conversion `FreeBlockAtMost`, which gives the free block's size,
its unregistered input from its own past, and its unregistered input from memory. What is still open is not how they are
placed but three pieces of the model, each a named open `Prop` below owned by memory-accounting or lean. The three
questions memory-accounting left open now have answers, with their basis: the on-chip state is `2^30` bits in the honest
device (data sheet: at least `992,061,184`), a floor on every bound at the chip's scale; PoUS holds summed over the ticks
of the window, for one audit a period (a cut at a single tick stays open); and neither PoUS nor PoUB counts information.
PoUS's `held` (`Schedule.held` in `Registration/Device.lean`, read at the capacity window by `Run.held` in
`Compose/Joint.lean`) counts the bits of device memory the schedule charges to the account at the end of each tick,
summed over the window's ticks: bit-ticks of memory occupied. PoUB's `reads` (`Run.reads` in `Poub/Remote/Run.lean`)
sums, over the node's memory tiers, the bits each tier's port carried to the node's compute over the audited interval.
`Placed.ingress` counts each distinct gate once, so neither is the placed quantity until a switch ties them. -/

/-- **PoUS: use at least, over the memory-size capacity.** In each server-day block `i`, at least `b r i` of the memory's
registered content was held, in the quantity `held r i` (wires crossing the ticks of the window at the device's memory
location, `Memory` in `Registration/Physical.lean`), summed over the ticks of the window, not at one moment; a cut at one
tick is open (above). Through `FreeBlockAtMost` it bounds the free block's unregistered input from its own past: what a
server's memory can hold that is not registered information, which bounds the input, from its own past, of a time slice of
a server-day block. The guarantee that ties it is `Pous.Period.PeriodHeldDeployed` (`Properties.Pous.PeriodDeployed`),
held summed over ticks for one audit a period; that statement lives in the properties tree, which the definitions tree
may not import, so this instance is **untied** here (`held` and `b` are free functions until the switch names it). On-chip
state is `2^30` bits in the honest device (data sheet: at least `992,061,184`), a floor on every bound at the chip's
scale. PoUS's own `held` (`Schedule.held` in `Registration/Device.lean`) counts the bits of device memory the schedule
charges to the account at the end of each tick, summed over the window's ticks, in bit-ticks: memory occupied, neither
information nor distinct wires, so it is not this row's `held` until the switch ties them. Open on
`DeviceMemoryIsLocation`. -/
def PousUseAtLeast (held b : (r : P.Run) → Fin (P.nOut r) → ℕ) (ε : ℝ≥0∞) : Prop :=
  P.UseAtLeast Capacity.memorySize (fun r _ i => held r i) (fun r _ i => b r i) ε

/-- **PoUB: use at least, over the link capacity on the memory-to-chip link.** In each server-day block `i`, at least `b
r i` of what the processors read from memory was registered information, in the quantity `reads r i` (wires crossing the
link from the device's memory to its chip in the block's span, `Links` in `Registration/Physical.lean`). Through
`FreeBlockAtMost` it bounds the free block's unregistered input from memory, an input only when memory is a location
outside the block. The guarantee that ties it is `Poub.Remote.RemoteRunConservationDeployed` (the link capacity, bits
read); this instance is left **untied** (`reads` and `b` free) until the switch names it. PoUB has **two** proved results,
not one: bits read (`RemoteRunConservationDeployed`, the link capacity, this row) and registered blocks held over time
(`Poub.Guarantees.RemoteHeld.HeldOverTimeLinks`, an instance of the memory-size capacity beside PoUS), the second stated
and listed, not shown to say anything at the served node, its clock unread. Open on `DeviceMemoryIsLocation` and
`PoubOnActualCircuit` (today PoUB is proved in a separate model). -/
def PoubUseAtLeast (reads b : (r : P.Run) → Fin (P.nOut r) → ℕ) (ε : ℝ≥0∞) : Prop :=
  P.UseAtLeast Capacity.linkCapacity (fun r _ i => reads r i) (fun r _ i => b r i) ε

/-! ## What the model still owes (named open premises) -/

/-- **A device's memory is a location of its own with a link to its chip** (open, owner lean and memory-accounting): the
hardware graph places a device's memory at a location with a link to the chip's location, so that memory size and the
memory-to-chip link are the `memorySize` and `linkCapacity` capacities PoUS and PoUB use. For that location to be sound
it needs three conditions, each named: every gate placed at the memory location copies one value (`memCopiesOneValue`);
the link from memory to chip has a latency ceiling (`memChipLatencyBounded`, main's `LatencyAtMost`); and the chip has
its own memory size, its on-chip state (`chipHasMemorySize`). It carries those three as a conjunction. -/
def DeviceMemoryIsLocation (memCopiesOneValue memChipLatencyBounded chipHasMemorySize : Prop) : Prop :=
  memCopiesOneValue ∧ memChipLatencyBounded ∧ chipHasMemorySize

/-- **PoUB's guarantee restated on the actual circuit** (open, owner memory-accounting): today PoUB is proved in a
separate model; this is the premise that its bound holds of the actual circuit, so that `PoubUseAtLeast` reads it. -/
def PoubOnActualCircuit (G : Prop) : Prop := G

/-- **The registered input gates are at most the registered width** (open, owner proofs with registration): in an
accepted run, the gates the run counts as registered information (`θ`) have width at most `registered` bits.
`RegisteredReadsOpenLog` gives that they are input gates, not their width. -/
def RegisteredWidthAtMost : Prop := ∀ r, P.accepts r → (P.actual r).width (P.θ r) ≤ P.registered

/-- **The pricing charges every computed gate at least 1** (open, owner compute-accounting): in every run, the pricing
`π`, at which PoUW's credited use and `ComputeRate` are stated, prices every computed gate at least 1. -/
def PricingChargesEveryGate : Prop := ∀ r, ∀ g ∈ (P.actual r).computed, 1 ≤ P.π r g

/-- **An inference block's claims are the declared row's** (open, owner compute-accounting with soundness): in an
accepted run, every claim on an inference block `u` is the declared row's function claim `f u`, a size claim of at least
`c u` gates, or an input claim from its own past, another block and the outside world of at least `k u` bits. -/
def InferenceClaimsDeclared (f : (r : P.Run) → Fin (P.nIn r) → Circuit) (c k : (r : P.Run) → Fin (P.nIn r) → ℕ) :
    Prop :=
  ∀ r, P.accepts r → ∀ u, P.cls r u = Class.inference → ∀ p ∈ P.claimsIn r u,
    p = .function (f r u) ∨ (∃ s, p = .sizeAtMost s ∧ c r u ≤ s) ∨
      (∃ s, p = .inputAtMost [.past, .block, .world] s ∧ k r u ≤ s)

/-! ## The memory wipe -/

/-- **The wipe's information bound gives a width bound** (open, memory-accounting): in an accepted run where the prior
content surviving the wipe is under its bound in information, every server-day block `i` reads at most `b i` bits from
its own past that is neither registered information nor what the auditor sends. A block that reads the wipe's own labels
reads far more, and the labels are not registered information; the `Source` list has no place yet for what the auditor
sends (challenges, wipe labels), so this row is open until that source is added as the architecture page names it: its
`.past` source excludes registered information (`g ∉ θ`) but not the auditor's, which it must. A block can also read many
bits that carry little information, so this needs a premise on how prior content lies in the gates it reads; whether a
block's input is to be stated in information is open. -/
def WipeInformationGivesWidth (survives : P.Run → Prop) (b : (r : P.Run) → Fin (P.nOut r) → ℕ) : Prop :=
  ∀ r, P.accepts r → survives r → ∀ i, P.kind r i = true → P.HoldsOut r i (.inputAtMost [.past] (b r i))

/-- **The memory wipe: input at most from its own past, for every server-day block**, at `pastBound E k` bits: `E i`, the
bits that leave the attested boundary during the wipe that opens block `i`, plus the wipe residual at its location's
registered label count `N` and label width `b` (bits), the wipe's challenge count `k i` and the soundness exponent `s`,
plus the location's storage the wipe does not cover. Until the wipe's check refuses a fill below the registered memory,
the guarantee is at the `N` the wipe ran at, and nothing ties that `N` to the hardware description; the comparison an
instance needs is `memCap ≤ wipeLabels * labelBits + uncovered` with all three read from the registration. From
the wipe's guarantee `G` (`Pous.Guarantees.StreamedWipeResidual`, which this layer may not import; `rows_chain`
supplies it), read on the runs once for each wipe (`RunReads`), each wipe a location of `Ls` (a finite family holding every listed location) and a day of
the period, at error `2^-s` plus the PRF distinguisher's advantage `adv` a wipe; the row's error is that times the
wipes, `|Ls| · nDays`, which bounds the run's server-day blocks (`serverDay_count_le`). `survives X d r` is that the
prior content surviving the wipe at `X` on day `d` of run `r` is under its bound. That guarantee is for a fill that is
the whole memory, `N · b` bits, so the row's hypotheses are: the location's memory in the hardware description
(`memCap`, bits, `CapacityAtMost Capacity.memorySize`) is at most `N · b` plus the storage the wipe does not cover, so
a hardware description listing more memory than the wipe fills makes the row's hypothesis false unless that term holds
the rest; and in every accepted run the guarantee applies at every server-day block's numbers (`WipeApplies`), which
`check` compares (`check_wipeApplies`). `k` is tied to what the wipe's check output only through `RunReads`; no
attester's signed output is tied to it yet. Open: `RunReads` and `WipeInformationGivesWidth`.

Limit, to be mended or kept by Mon 23 Nov noon PST: the wipe's row holds for a server that is alone behind its attested
boundary, or for a boundary whose every location is filled and challenged in ONE wipe (then `N` is every location's
labels together and the egress is the boundary's over that wipe). Where servers behind one boundary are wiped at
different times, a server's content can sit on its neighbour through its wipe and return, and nothing crosses the
boundary: the row's `wipeEgress` is then no bound on what left the SERVER, and the row does not apply.

`wipeInputAtMost` proves the summing over the period's wipes from `RunReads` and `WipeInformationGivesWidth` alone;
`StreamedWipeResidual`, the memory cover and `WipeApplies` are not used by that proof: they are what an instance must
hold to discharge `RunReads`, which is open. That the wipe bounds a block's input is not proved here. -/
def WipeInputAtMost (G : Prop) (Ls : Finset (Finset L)) (survives : Finset L → Fin P.nDays → P.Run → Prop)
    (E k memEnclosing memCap : (r : P.Run) → Fin (P.nOut r) → ℕ) (adv : ℝ≥0∞) : Prop :=
  G →
    (∀ X ∈ Ls, ∀ d, P.RunReads G (survives X d) (2⁻¹ ^ P.wipeError + adv)) →
    P.CapacityAtMost Capacity.memorySize memEnclosing memCap →
    (∀ r i, P.kind r i = true →
      memCap r i ≤ P.wipeLabels (P.locOut r i) * P.labelBits (P.locOut r i) + P.uncovered (P.locOut r i)) →
    (∀ r, P.accepts r → ∀ i, P.kind r i = true → P.WipeApplies k r i) →
    (∀ X ∈ P.listed, X ∈ Ls) →
    P.WipeInformationGivesWidth (fun r => ∀ X ∈ Ls, ∀ d, survives X d r) (P.pastBound E k) →
      ExceptWith P.pr P.accepts
        (fun r => ∀ i, P.kind r i = true → P.HoldsOut r i (.inputAtMost [.past] (P.pastBound E k r i)))
        ((Ls.card * P.nDays : ℕ) * (2⁻¹ ^ P.wipeError + adv))

/-! ## Network attestation -/

/-- **Every link of the attested boundary passes through an attester** (assumed, about the site). In an accepted run,
for each server-day block `i`, its attested boundary's day block (the union of the boundary's server-day blocks that
day, `HoldsBoundary`) reads from the outside world at most the attester's ingress count for that boundary and day plus
the boundary's covert figure (`P.covert`, bits, assumed: the channels the attester does not see), and that day block's
output to the outside world is at most the attester's egress count for that boundary and day (`egress`, bits). The
links between servers behind one boundary are inside the union and not in the covert figure. A server behind a shared
boundary has its input from and its output to the outside world bounded only through the union: a policy can no longer
bound one server's output behind a shared boundary, which includes what it sends the other servers behind it. -/
def AttesterOnEveryLink (ingress egress : (r : P.Run) → ℕ → Fin P.nDays → ℕ) : Prop :=
  ∀ r, P.accepts r → ∀ i, P.kind r i = true →
    P.HoldsBoundary r (P.bnd r i) (P.dayOf r i)
        (.inputAtMost [.world] (ingress r (P.bnd r i) (P.dayOf r i) + P.covert (P.bnd r i))) ∧
      P.HoldsBoundary r (P.bnd r i) (P.dayOf r i) (.outputAtMost (egress r (P.bnd r i) (P.dayOf r i)))

/-- **The record's unexplained bits are the boundary's undeclared output** (open, network-accounting): in an accepted run
whose egress record the declared computation explains up to the missable cost and the charges, the egress count of
every server-day block's attested boundary on its day is at most `bOut` for that boundary and day (bits). -/
def RecordIsOutput (shown : P.Run → Prop) (egress bOut : (r : P.Run) → ℕ → Fin P.nDays → ℕ) : Prop :=
  ∀ r, P.accepts r → shown r → ∀ i, P.kind r i = true →
    egress r (P.bnd r i) (P.dayOf r i) ≤ bOut r (P.bnd r i) (P.dayOf r i)

/-- **Network attestation: input at most from the outside world, for each attested boundary's day block** (old subject:
the server-day block), from `AttesterOnEveryLink` and the attester's ingress count at most `bIn` for that boundary and
day. The boundary's covert figure (`P.covert`, bits, assumed) lives **inside** the assumption, not added on here. A
server behind a shared boundary has its input from the outside world bounded only through the union. A shape. -/
def NetworkInputAtMost (ingress bIn egress : (r : P.Run) → ℕ → Fin P.nDays → ℕ) : Prop :=
  P.AttesterOnEveryLink ingress egress →
    (∀ r, P.accepts r → ∀ i, P.kind r i = true → ingress r (P.bnd r i) (P.dayOf r i) ≤ bIn r (P.bnd r i) (P.dayOf r i)) →
    ExceptWith P.pr P.accepts
      (fun r => ∀ i, P.kind r i = true → P.HoldsBoundary r (P.bnd r i) (P.dayOf r i)
        (.inputAtMost [.world] (bIn r (P.bnd r i) (P.dayOf r i) + P.covert (P.bnd r i)))) 0

/-- **Network attestation: output at most to the outside world, for each attested boundary's day block** (the server-day
block its one-server case, `boundary_one_server_output`): a width bound that leaves the timing channel out. A policy can
no longer bound one server's output behind a shared boundary, which includes what it sends the other servers behind it. From `NetTiming.Attester.PhaseGetsBack`, which bounds
how much of an execution phase's egress record the declared computation does not explain. The fold from one execution
phase to the monitoring period — that the state each phase reads is the fold of the earlier ones (`PhasesFoldToPeriod`) —
is carried by `RunReads` here, which reads the phase guarantee on the period's runs; it is not a separate named def on this
head, and restoring it as its own premise is ours with lean. Its channel term, the log of how many timing and failure
patterns the operator can choose, is no width of any gate; whether the output is stated in information for that part is
open for training. Open: `RunReads` and `RecordIsOutput`; assumed: `AttesterOnEveryLink`. A shape.

Limit, Wed 25 Nov: the two counts are tied by no premise yet; the row bounds undeclared output, and reads the attester's
count only through an equality an instance owes.

Limit, Wed 25 Nov: `ε` is the period's: the phase's error times the period's execution phases, as the wipe's row sums
it. -/
def NetworkOutputAtMost (shown : P.Run → Prop) (egress bOut egress' ingress : (r : P.Run) → ℕ → Fin P.nDays → ℕ)
    (ε : ℝ≥0∞) : Prop :=
  NetTiming.Attester.PhaseGetsBack →
    P.RunReads NetTiming.Attester.PhaseGetsBack shown ε → P.RecordIsOutput shown egress bOut →
    P.AttesterOnEveryLink ingress egress' →
      ExceptWith P.pr P.accepts (fun r => ∀ i, P.kind r i = true →
        P.HoldsBoundary r (P.bnd r i) (P.dayOf r i) (.outputAtMost (bOut r (P.bnd r i) (P.dayOf r i)))) ε

/-! ## Registration -/

/-- **The registered reads open the log** (open, proofs): in an accepted run, every gate of `θ` is an input gate whose
value opens, at the position the verifier derives from the declared circuit, a root of the log `log`. The log is one for
the period, fixed before the first epoch's first challenge. Open: `opensAt r g root` has no definition on the actual
circuit; what defines it is the open question. The first instance (the registered reads the verifier checks, tied to
`Placed`) must carry with it the excluded run — an accepted-looking run whose `θ` holds a gate opening no logged root,
shown not to satisfy this — else an instance that makes `opensAt` trivially true brings the vacuity back. -/
def RegisteredReadsOpenLog {Root : Type} (log : Finset Root)
    (opensAt : (r : P.Run) → Fin (P.actual r).N → Root → Prop) : Prop :=
  ∀ r, P.accepts r → ∀ g ∈ P.θ r, (P.actual r).isInput g = true ∧ ∃ root ∈ log, opensAt r g root

/-- **Registration binds `θ`**: from `Verity.Guarantees.RegistrationSound` and `RegisteredReadsOpenLog`, every gate an
accepted run counts as registered opens a root of the log. A logged value read as a fresh input is not registered, so the
converse is no guarantee. A shape. -/
def RegistrationBinds {Root : Type} (log : Finset Root)
    (opensAt : (r : P.Run) → Fin (P.actual r).N → Root → Prop) : Prop :=
  Verity.Guarantees.RegistrationSound → P.RegisteredReadsOpenLog log opensAt →
    ∀ r, P.accepts r → ∀ g ∈ P.θ r, ∃ root ∈ log, opensAt r g root

end ResourceAccounting.Runs

namespace InferenceOnly

open ResourceAccounting

/-- **An actual circuit complies with the policy `pol`** at the registered input gates `θ`: they are input gates, and `A`
with them as its registered information satisfies `pol`. -/
def Compliant {L : Type*} (pol : Policy) (A : Placed L) (θ : Finset (Fin A.N)) : Prop :=
  ∃ h : ∀ a ∈ θ, A.isInput a = true, pol (.ofCircuit A.toCircuit θ h)

/-- **The policy's theorem** for the check `check`: every run whose committed blocks pass the check, realize their
claims outside a tolerance set `M` of inference blocks (`RealizesExcept`) and whose free and tolerance work for that `M`
is at most capacity − credited use + tolerance in every server-day block (`FreeWorkAtMost`) has a compliant actual
circuit. `FreeWorkAtMost` is a hypothesis of the statement, beside `RealizesExcept`: without it `M` could be every
inference block and the run would be bound by nothing. With it `M`'s actual work sits under capacity − credited use +
tolerance, which `check` caps at `lim.freeSize`, so a run whose tolerance set's work exceeds the limit satisfies no
instance of the hypotheses (`excluded_run`). `rows_chain` derives `FreeWorkAtMost` from PoUW's row, the compute capacity
and the declared row's tolerance (`freeWorkAtMost_of_conversion`).
`M`'s declared use is counted in the free block and the injected information it passes into correct blocks is a term of
their input.
The policy needs no new theorem for PoUS and PoUB: its bound already takes any family of blocks that have a size and an
unregistered input, so placing the free block's input from its own past (PoUS) and from memory (PoUB) adds terms to the
same bound and states no new theorem here. -/
def PolicyTheorem {L : Type*} (check : (P : Runs L) → P.Run → Bool) (pol : Policy) : Prop :=
  ∀ (P : Runs L) (r : P.Run) (M : Finset (Fin (P.nIn r))),
    check P r = true → P.RealizesExcept r M → P.FreeWorkAtMost r M → Compliant pol (P.actual r) (P.θ r)

/-- **The inference-only policy** at the limits `lim`, with training efficiency `efficiency`: a declaration complies when
its training efficiency is at most `share + allowance`. Training efficiency is an undefined parameter here, a function of
a declaration supplied by the policy's author; this layer fixes no definition for it and no utilization target, which are
likewise parameters (`lim.share`, `lim.allowance`). Stating the target is the policy author's, not this layer's. -/
def efficiencyAtMost (lim : Limits) (efficiency : RegisteredCircuit → ℝ) : Policy := fun d =>
  efficiency d ≤ lim.share + lim.allowance

/-- **The inference-only theorem** (open, training): every run whose committed blocks pass the inference-only check and
realize their claims has training efficiency at most the free blocks' share plus the allowance. -/
def InferenceOnlyTheorem {L : Type*} (lim : Limits) (efficiency : RegisteredCircuit → ℝ) : Prop :=
  PolicyTheorem (check (L := L) lim) (efficiencyAtMost lim efficiency)

end InferenceOnly

namespace ResourceAccounting

/-- **The chain**: over a monitoring period, with the policy's theorem, the check passing on every accepted run, and the
protocols' rows `shown x` (PoUW through the conversion for the free block's size, `DeclaredBlocksRun` for the inference
blocks' claims outside a tolerance set, the wipe and network attestation for input and output, registration for `θ`) that
between them give that an accepted run realizes its claims outside some tolerance set of inference blocks whose free and
tolerance work is at most capacity − credited use + tolerance (`FreeWorkAtMost`; each row established except with
probability at most `ε x`), an accepted run's actual circuit complies with `pol`, except with probability at most
`∑ x ∈ s, ε x`. A unit labelled inference that starts before its epoch's challenge tick is a free block by the class's
definition (`cls`), and the tolerance set is counted beside the free blocks, both under that bound. -/
def Chain {L : Type*} (P : Runs L) (check : (P : Runs L) → P.Run → Bool) (pol : InferenceOnly.Policy)
    {κ : Type} (s : Finset κ) (shown : κ → P.Run → Prop) (ε : κ → ℝ≥0∞) : Prop :=
  InferenceOnly.PolicyTheorem check pol → (∀ r, P.accepts r → check P r = true) →
    (∀ r, P.accepts r → (∀ x ∈ s, shown x r) → ∃ M, P.FreeWorkAtMost r M ∧ P.RealizesExcept r M) →
    (∀ x ∈ s, ExceptWith P.pr P.accepts (shown x) (ε x)) →
    ExceptWith P.pr P.accepts (fun r => InferenceOnly.Compliant pol (P.actual r) (P.θ r)) (∑ x ∈ s, ε x)

end ResourceAccounting
