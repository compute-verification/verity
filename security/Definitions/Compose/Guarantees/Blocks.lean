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

* **PoUW** (`PouwUseAtLeast`): use at least, in size, from `Pouw.PearlC.GGapSampledDraw`, through `SpendIsWork`.
* **The declared blocks run** (`DeclaredBlocksRun`): every inference block outside a set within the tolerance implements
  its declared block, at a size and input at most stated. It is owed an argument from PoUW's coverage of the declared
  work, with the uncovered share and the incorrect proof blocks the sample missed counted in the free block.
* **PoUB** (`PoubUseAtLeast`) and **PoUS** (`PousUseAtLeast`): open and not placed; no term of the inference-only bound
  reads memory reads or memory held.
* **The memory wipe** (`WipeInputAtMost`): input at most from its own past, for every server-day block; its guarantee's
  module is a layer this one may not import, so the hypothesis `G`.
* **Network attestation** (`NetworkInputAtMost`, `NetworkOutputAtMost`): input at most from `AttesterOnEveryLink` and the
  attester's ingress count; output at most from `NetTiming.Attester.PhaseGetsBack` and the record's unexplained bits.
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

/-- **The spend is the inference blocks' work after the salt** (open, compute-accounting): in an accepted run whose spend
clears the claim, inside every server-day block `i` the inference blocks directly inside it whose windows start at or
after their epoch's challenge tick `salt u` have work at least `b r i` at `π`. Those blocks are siblings, which the
partition keeps disjoint, so the sum counts no gate twice. `b` is work at `π` and the spend is in multiply-accumulates:
the conversion between them is this premise's content. -/
def SpendIsWork (salt : (r : P.Run) → Fin (P.nIn r) → ℕ) (shown : P.Run → Prop)
    (b : (r : P.Run) → Fin (P.nOut r) → ℕ) : Prop :=
  ∀ r, P.accepts r → shown r → ∀ i, P.kind r i = true →
    b r i ≤ ∑ u ∈ (P.infChildren r i).filter (fun u => salt r u ≤ (P.winIn r u).start), P.sizeIn r u

/-- **PoUW (Pearl+): use at least, in size.** From `Pouw.PearlC.GGapSampledDraw`: the accepted adversary's spend is at
least a share of the claimed work. Open: `RunReads` and `SpendIsWork`. Its use is the inference children's work after the
salt, which `sizeExclusive` bounds, so it feeds the conversion. A shape: its content is its premises. -/
def PouwUseAtLeast {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : Pouw.PearlC.CostModel Q R S) (Pr : Pouw.PearlC.Protocol Q R S) (TR : Pouw.PearlC.TileRules Q R S)
    (D : Pouw.PearlC.Layout → Prop) (Wd : Pouw.PearlC.Layout → ℕ → Pouw.PearlC.Decl → ℚ) (g : ℝ) (δ : ℕ → ℕ → ℝ)
    (salt : (r : P.Run) → Fin (P.nIn r) → ℕ) (shown : P.Run → Prop)
    (b : (r : P.Run) → Fin (P.nOut r) → ℕ) (ε : ℝ≥0∞) : Prop :=
  Pouw.PearlC.GGapSampledDraw CM Pr TR D Wd g δ →
    P.RunReads (Pouw.PearlC.GGapSampledDraw CM Pr TR D Wd g δ) shown ε → P.SpendIsWork salt shown b →
      P.UseAtLeast Capacity.computeRate
        (fun r i => ∑ u ∈ (P.infChildren r i).filter (fun u => salt r u ≤ (P.winIn r u).start), P.sizeIn r u) b ε

/-! ## The declared blocks run -/

/-- **The declared blocks run** (open, compute-accounting with soundness): every inference block `u` whose window starts
at or after its epoch's challenge tick `salt u`, outside a set `M` whose declared size summed over its blocks
(`∑ u ∈ M, dsize u`) is at most the tolerance `tol`, implements its declared block `f u`, at size at most `c u` gates (the
declared size times a stated overhead) and input at most `k u` bits from its own past, another block and the outside
world, except with probability at most `ε`. `dsize u` is the declared size the run carries for block `u` (the same
quantity the free block is charged), per block, not a free function of the set: compute-accounting asked for the sum in
the statement, so a set's tolerance is `∑ u ∈ M, dsize u` and no `decl ≡ 0` makes every `M` within any tolerance. Nothing
in Verity proves that the prover ran the declared circuit: the proofs show committed values consistent with it. An
inference block before its salt, with no row here, is free work the partition declares a free block. The set `M` is the
tolerance set the chain reads through `RealizesExcept`; the unregistered information the incorrect proof blocks in `M` pass
into correct ones (the injected information) is the `.block` source of the inference blocks' input here. The tolerance
binds the actual work through the committed free-capacity claim (`claimedFree`, the clause `FreeClaimHolds`): `M`'s
inference blocks sit inside the free block, counted at their actual size, so `feasible_work_le` reads them. The row gives `.function`,
`.sizeAtMost` and `.inputAtMost`; that these are an inference block's full claim list (`∀ p ∈ claimsIn`) is owed where the
rows become the chain's hypotheses by name (`rows_chain`). -/
def DeclaredBlocksRun (salt : (r : P.Run) → Fin (P.nIn r) → ℕ) (f : (r : P.Run) → Fin (P.nIn r) → Circuit)
    (c k dsize : (r : P.Run) → Fin (P.nIn r) → ℕ) (tol : ℕ) (ε : ℝ≥0∞) : Prop :=
  ExceptWith P.pr P.accepts
    (fun r => ∃ M ⊆ (univ.filter fun u => P.cls r u = Class.inference ∧ salt r u ≤ (P.winIn r u).start),
      (∑ u ∈ M, dsize r u) ≤ tol ∧
      ∀ u ∈ (univ.filter fun u => P.cls r u = Class.inference ∧ salt r u ≤ (P.winIn r u).start) \ M,
        P.HoldsIn r u (.function (f r u)) ∧ P.HoldsIn r u (.sizeAtMost (c r u)) ∧
          P.HoldsIn r u (.inputAtMost [.past, .block, .world] (k r u))) ε

/-! ## PoUS and PoUB

PoUW, PoUS and PoUB are the three instances of `UseAtLeast`, over the three capacities of the hardware graph (compute
rate, memory size, link capacity). Each feeds the one conversion `FreeBlockAtMost`, which gives the free block's size,
its unregistered input from its own past, and its unregistered input from memory. What is still open is not how they are
placed but three pieces of the model, each a named open `Prop` below owned by memory-accounting or lean. The three
questions memory-accounting left open now have answers, with their basis: the on-chip state is `2^30` bits in the honest
device (data sheet: at least `992,061,184`), a floor on every bound at the chip's scale; PoUS holds summed over the ticks
of the window, for one audit a period (a cut at a single tick stays open); and PoUS and PoUB count information from below,
which bounds wires from below. -/

/-- **PoUS: use at least, over the memory-size capacity.** In each server-day block `i`, at least `b r i` of the memory's
registered content was held, in the quantity `held r i` (wires crossing the ticks of the window at the device's memory
location, `Memory` in `Registration/Physical.lean`), summed over the ticks of the window, not at one moment; a cut at one
tick is open (above). Through `FreeBlockAtMost` it bounds the free block's unregistered input from its own past: what a
server's memory can hold that is not registered information, which bounds the input, from its own past, of a time slice of
a server-day block. The guarantee that ties it is `Pous.Period.PeriodHeldDeployed` (`Properties.Pous.PeriodDeployed`),
held summed over ticks for one audit a period; that statement lives in the properties tree, which the definitions tree
may not import, so this instance is **untied** here (`held` and `b` are free functions until the switch names it). On-chip
state is `2^30` bits in the honest device (data sheet: at least `992,061,184`), a floor on every bound at the chip's
scale; PoUS counts information from below, which bounds wires from below. Open on `DeviceMemoryIsLocation`. -/
def PousUseAtLeast (held b : (r : P.Run) → Fin (P.nOut r) → ℕ) (ε : ℝ≥0∞) : Prop :=
  P.UseAtLeast Capacity.memorySize held b ε

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
  P.UseAtLeast Capacity.linkCapacity reads b ε

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

/-- **The memory wipe: input at most from its own past, for the attested boundary** (the server-day block its one-server
case), at `E i + 8,600,000,000` bits, `E i` the bits that leave the boundary during the wipe that opens block `i`. From
the streamed-label wipe's guarantee (a layer this one may not import, so the hypothesis `G`), over a uniform prior, at
`2⁻⁴⁰` plus the PRF distinguisher's advantage `adv`. The guarantee is at "the fill is the whole memory" (`M = N·512`), so
the memory size from the hardware description is a hypothesis of this row (`CapacityAtMost Capacity.memorySize`, keeping a
device at each server). Open: `RunReads` and `WipeInformationGivesWidth`. A shape: its content is its premises. -/
def WipeInputAtMost (G : Prop) (survives : P.Run → Prop) (E : (r : P.Run) → Fin (P.nOut r) → ℕ)
    (memEnclosing memCap : (r : P.Run) → Fin (P.nOut r) → ℕ) (adv : ℝ≥0∞) : Prop :=
  G → P.RunReads G survives (2⁻¹ ^ 40 + adv) → P.CapacityAtMost Capacity.memorySize memEnclosing memCap →
    P.WipeInformationGivesWidth survives (fun r i => E r i + 8600000000) →
      ExceptWith P.pr P.accepts
        (fun r => ∀ i, P.kind r i = true → P.HoldsOut r i (.inputAtMost [.past] (E r i + 8600000000)))
        (2⁻¹ ^ 40 + adv)

/-! ## Network attestation -/

/-- **Every link of the attested boundary passes through an attester** (assumed, about the site): the block is everything
behind one attester, the server-day block its one-server case — not a per-server figure where servers behind one attester
are joined by links it does not see. In an accepted run, that block reads from the outside world at most the attester's
ingress count `ingress r i` plus `covert r i`, an assumed number of bits for the covert channels the attester does not
see, and its output to the outside world is at most the attester's egress count `egress r i`. It is that no link across
the attested boundary bypasses one. -/
def AttesterOnEveryLink (ingress egress covert : (r : P.Run) → Fin (P.nOut r) → ℕ) : Prop :=
  ∀ r, P.accepts r → ∀ i, P.kind r i = true →
    P.HoldsOut r i (.inputAtMost [.world] (ingress r i + covert r i)) ∧ P.HoldsOut r i (.outputAtMost (egress r i))

/-- **The record's unexplained bits are the block's undeclared output** (open, network-accounting): in an accepted run
whose egress record the declared computation explains up to the missable cost and the charges, every server-day block's
egress count is at most `bOut i`. -/
def RecordIsOutput (shown : P.Run → Prop) (egress bOut : (r : P.Run) → Fin (P.nOut r) → ℕ) : Prop :=
  ∀ r, P.accepts r → shown r → ∀ i, P.kind r i = true → egress r i ≤ bOut r i

/-- **Network attestation: input at most from the outside world, for the attested boundary** (the server-day block its
one-server case), from `AttesterOnEveryLink` and the attester's ingress count at most `bIn i`. The `covert i` term, an
assumed number of bits for the covert channels the attester does not see, lives **inside** the assumption
(`AttesterOnEveryLink` bounds input by the ingress count plus `covert`), not added on here: covert channels are an assumed
number of the adversary resource model's list, not zero, and the assumption is exactly the one that assumes them. A
shape. -/
def NetworkInputAtMost (ingress egress bIn covert : (r : P.Run) → Fin (P.nOut r) → ℕ) : Prop :=
  P.AttesterOnEveryLink ingress egress covert →
    (∀ r, P.accepts r → ∀ i, P.kind r i = true → ingress r i ≤ bIn r i) →
    ExceptWith P.pr P.accepts
      (fun r => ∀ i, P.kind r i = true → P.HoldsOut r i (.inputAtMost [.world] (bIn r i + covert r i))) 0

/-- **Network attestation: output at most to the outside world, for the attested boundary** (the server-day block its
one-server case): a width bound that leaves the timing channel out. From `NetTiming.Attester.PhaseGetsBack`, which bounds
how much of an execution phase's egress record the declared computation does not explain. The fold from one execution
phase to the monitoring period — that the state each phase reads is the fold of the earlier ones (`PhasesFoldToPeriod`) —
is carried by `RunReads` here, which reads the phase guarantee on the period's runs; it is not a separate named def on this
head, and restoring it as its own premise is ours with lean. Its channel term, the log of how many timing and failure
patterns the operator can choose, is no width of any gate; whether the output is stated in information for that part is
open for training. Open: `RunReads` and `RecordIsOutput`; assumed: `AttesterOnEveryLink`. A shape. -/
def NetworkOutputAtMost (shown : P.Run → Prop) (egress bOut ingress egress' covert' : (r : P.Run) → Fin (P.nOut r) → ℕ)
    (ε : ℝ≥0∞) : Prop :=
  NetTiming.Attester.PhaseGetsBack →
    P.RunReads NetTiming.Attester.PhaseGetsBack shown ε → P.RecordIsOutput shown egress bOut →
    P.AttesterOnEveryLink ingress egress' covert' →
      ExceptWith P.pr P.accepts (fun r => ∀ i, P.kind r i = true → P.HoldsOut r i (.outputAtMost (bOut r i))) ε

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
claims outside a tolerance set `M` of inference blocks (`RealizesExcept`) and whose committed free-capacity claim holds
for that `M` (`FreeClaimHolds`) has a compliant actual circuit. `FreeClaimHolds` is a hypothesis of the statement, beside
`RealizesExcept`: without it `M` could be every inference block and the run would be bound by nothing. With it `M`'s
actual work sits inside each server-day block's committed claim (`claimedFree`), which `check` caps at `lim.freeSize`,
so a run whose tolerance set's work exceeds the limit satisfies no instance of the hypotheses (`excluded_run`). Who
establishes `FreeClaimHolds` is the conversion (capacity minus the use credited outside `M`), open in `rows_chain`.
`M`'s declared use is counted in the free block and the injected information it passes into correct blocks is a term of
their input.
The policy needs no new theorem for PoUS and PoUB: its bound already takes any family of blocks that have a size and an
unregistered input, so placing the free block's input from its own past (PoUS) and from memory (PoUB) adds terms to the
same bound and states no new theorem here. -/
def PolicyTheorem {L : Type*} (check : (P : Runs L) → P.Run → Bool) (pol : Policy) : Prop :=
  ∀ (P : Runs L) (r : P.Run) (M : Finset (Fin (P.nIn r))),
    check P r = true → P.RealizesExcept r M → P.FreeClaimHolds r M → Compliant pol (P.actual r) (P.θ r)

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
between them give that an accepted run realizes its claims outside some tolerance set of inference blocks for which its
committed free-capacity claim holds (each row established except with probability at most `ε x`), an accepted run's
actual circuit complies with `pol`, except with probability at most `∑ x ∈ s, ε x`. An inference block no row covers
(before its epoch's salt), and the tolerance set the sample misses, are free work, counted under the committed claim. -/
def Chain {L : Type*} (P : Runs L) (check : (P : Runs L) → P.Run → Bool) (pol : InferenceOnly.Policy)
    {κ : Type} (s : Finset κ) (shown : κ → P.Run → Prop) (ε : κ → ℝ≥0∞) : Prop :=
  InferenceOnly.PolicyTheorem check pol → (∀ r, P.accepts r → check P r = true) →
    (∀ r, P.accepts r → (∀ x ∈ s, shown x r) → ∃ M, P.FreeClaimHolds r M ∧ P.RealizesExcept r M) →
    (∀ x ∈ s, ExceptWith P.pr P.accepts (shown x) (ε x)) →
    ExceptWith P.pr P.accepts (fun r => InferenceOnly.Compliant pol (P.actual r) (P.θ r)) (∑ x ∈ s, ε x)

end ResourceAccounting
