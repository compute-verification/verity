module

public import Definitions.ResourceAccounting.BlockProperties
public import Definitions.Pouw.PearlC.TileSample
public import Definitions.Poub.Guarantees.RateAuditDeployed
public import Definitions.ResourceAccounting.Network.AttesterScheme
public import Definitions.Registration.Guarantees
public import Definitions.InferenceOnly
public import Definitions.Slowdown.Guarantees.ServingBlocks

/-!
# The protocols' block statements, the inference-only policy over a division, and the chain

Each protocol's instance of the form in `Definitions/ResourceAccounting/BlockProperties.lean` takes its guarantee as a
hypothesis, named where this layer may import it, and the open premises that tie it to the actual circuit. An instance
whose conclusion is a one-line consequence of its hypotheses is a shape: its content is its premises.

* **PoUW** (`PouwUseAtLeast`): use at least, in size, from `Pouw.PearlC.GGapSampledDraw`, through `SpendIsWork`.
* **The declared blocks run** (`DeclaredBlocksRun`): every inference block outside a set within the tolerance
  implements its declared block, at a size and input at most stated: one premise, replacing a function claim read
  off the sampled proofs.
* **PoUB** (`PoubUseAtLeast`): use at least, in registered reads counted with multiplicity, from
  `ResourceAccounting.Guarantees.RemoteRunConservationDeployed`.
* **PoUS** (`PousUseAtLeast`): use at least, in the device model's held bit-ticks; its tie to the actual circuit is
  `HeldExclusive`. Its guarantee is in the properties tree, which this tree does not read, so the hypothesis `G`.
* **The memory wipe** (`WipeInputAtMost`): input at most from its own past. Its guarantee's module is a guarantee
  layer this one may not import, so the hypothesis `G`.
* **Network attestation** (`NetworkInputAtMost`, `NetworkOutputAtMost`): input at most from `AttesterOnEveryLink`
  alone; output at most from `NetTiming.Attester.PhaseGetsBack`, `PhasesFoldToPeriod`, `RecordIsOutput` and
  `AttesterOnEveryLink`.
* **Registration** (`RegistrationBinds`): the run's registered input gates are the log's, from
  `Verity.Guarantees.RegistrationSound` and `RegisteredIsLog`.
* **The proof layer** is `Verity.Scheme.GetsBack` itself and gives no block property here.

The policy is a condition on declarations (`InferenceOnly.Policy`), read of an actual circuit with its registered input
gates (`InferenceOnly.Compliant`); its theorem over a division's check is `InferenceOnly.PolicyTheorem`. `Chain` is the
composition, proved by `ResourceAccounting.SecurityProofs.BlockProperties.chain`.
-/

@[expose] public section

open Verity Finset
open scoped ENNReal

namespace ResourceAccounting.Division.Runs

variable {L : Type*} {Dv : Division L} (P : Dv.Runs)

/-- Block `i`'s size in run `r` at the run's pricing, 0 where the run has no blocks for the division. -/
def size (r : P.Run) (i : Dv.ι) : ℕ := match P.blocks r with
  | some B => B.size (P.π r) i
  | none => 0

/-- Block `i`'s registered reads in run `r`, 0 where the run has no blocks for the division. -/
def registeredReads (r : P.Run) (i : Dv.ι) : ℕ := match P.blocks r with
  | some B => B.registeredReads (P.θ r) i
  | none => 0

/-! ## PoUW -/

/-- **The spend is the inference blocks' work** (open, compute-accounting): in an accepted run whose spend clears the
claim, the work spent after the salt does not double as other work, and inside every block `i` of `K` the inference
blocks below it have work at least `b i` at `π`. -/
def SpendIsWork (K : Finset Dv.ι) (shown : P.Run → Prop) (b : Dv.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → shown r → ∀ i ∈ K, b i ≤ ∑ j ∈ Dv.within Dv.inference i, P.size r j

/-- **PoUW (Pearl+): use at least, in size.** From `Pouw.PearlC.GGapSampledDraw`: the accepted adversary's spend is at
least a share of the claimed work. Open: `RunReads` and `SpendIsWork`. Its evidence is pooled over the period's
devices, so the class it supports today is the whole circuit. A shape: its content is its premises. -/
def PouwUseAtLeast {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : Pouw.PearlC.CostModel Q R S) (Pr : Pouw.PearlC.Protocol Q R S) (TR : Pouw.PearlC.TileRules Q R S)
    (D : Pouw.PearlC.Layout → Prop) (Wd : Pouw.PearlC.Layout → ℕ → Pouw.PearlC.Decl → ℚ) (g : ℝ) (δ : ℕ → ℕ → ℝ)
    (K : Finset Dv.ι) (shown : P.Run → Prop) (b : Dv.ι → ℕ) (ε : ℝ≥0∞) : Prop :=
  Pouw.PearlC.GGapSampledDraw CM Pr TR D Wd g δ → P.RunReads (Pouw.PearlC.GGapSampledDraw CM Pr TR D Wd g δ) shown ε →
    P.SpendIsWork K shown b → P.UseAtLeast K (fun r i => ∑ j ∈ Dv.within Dv.inference i, P.size r j) b ε

/-! ## The declared blocks run -/

/-- **The declared blocks run** (open, compute-accounting with soundness): every inference block `j` whose window starts
at or after the salt's tick `salt`, outside a set whose declared use `decl` is at most the tolerance `tol`, implements
its declared block `f j`, at size at most `c j` (the declared size times a stated overhead at the pricing) and input at
most `k j` bits from its own past, another block and the outside world, except with probability at most `ε`.
Nothing in Verity proves that the prover ran the declared circuit: the proofs show committed values consistent with it,
and useful work shows an amount of work. Whether this is (a) a named assumption of the system or (b) owed an argument
from PoUW's coverage of the declared work, the uncovered share counted in the free block, is a claim for Daniel. -/
def DeclaredBlocksRun (salt : ℕ) (f : Dv.ι → Circuit) (c k : Dv.ι → ℕ) (decl : Finset Dv.ι → ℕ) (tol : ℕ)
    (ε : ℝ≥0∞) : Prop :=
  P.EstablishedOutside (Dv.inference.filter fun j => salt ≤ (Dv.win j).start)
    (fun j => [.function (f j), .sizeAtMost (c j), .inputAtMost [.past, .block, .world] (k j)]) decl tol ε

/-! ## PoUB -/

/-- **The tier reads are the enclosing blocks' registered reads** (open, memory-accounting): in an accepted run whose
reads clear the claim, every block `i` of `K` reads registered information at least `b i` bits, counted with
multiplicity. The session's passes are the audit's own work over the store, so they are the enclosing block's reads. -/
def TierReadsAreReads (K : Finset Dv.ι) (shown : P.Run → Prop) (b : Dv.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → shown r → ∀ i ∈ K, b i ≤ P.registeredReads r i

/-- **PoUB: use at least, in registered reads counted with multiplicity.** From
`ResourceAccounting.Guarantees.RemoteRunConservationDeployed`: every opening passes only while the session's tier reads
are at least a share of n·P·N·b. Open: `RunReads`, which carries `WorkHardness`, `SquaringCeiling` and the side-channel
independence on sessions of at least seven windows, and `TierReadsAreReads`. Its session is one node's, so the class is
the server-day blocks; the capacity is the node's read rate times the window. Reads are not one of the four
properties: which property the free block's input from memory takes (a rate form of `inputAtMost`, or reads) is open
for training. A shape: its content is its premises. -/
def PoubUseAtLeast (K : Finset Dv.ι) (shown : P.Run → Prop) (b : Dv.ι → ℕ) (ε : ℝ≥0∞) : Prop :=
  ResourceAccounting.Guarantees.RemoteRunConservationDeployed →
    P.RunReads ResourceAccounting.Guarantees.RemoteRunConservationDeployed shown ε → P.TierReadsAreReads K shown b →
      P.UseAtLeast K P.registeredReads b ε

/-! ## PoUS -/

/-- **The held bit-ticks clear the claim** (open, memory-accounting): in an accepted run whose audit passed, the device
model's held bit-ticks `held r i` over every block `i` of `K` are at least `b i`. -/
def HeldClearsClaim (K : Finset Dv.ι) (shown : P.Run → Prop) (held : P.Run → Dv.ι → ℕ) (b : Dv.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → shown r → ∀ i ∈ K, b i ≤ held r i

/-- **Held memory does not double as the free block's state** (open, memory-accounting with lean): at every tick `t`
of the enclosing block `encl j`, what the free block `j` carries across `t` plus what is held at `t` (`heldAt r i t`)
is at most the location's memory `mem (encl j)`. `Placed` has no memory occupied but not read; `heldAt` stands for it. -/
def HeldExclusive (F : Finset Dv.ι) (encl : Dv.ι → Dv.ι) (heldAt : P.Run → Dv.ι → ℕ → ℕ) (mem : Dv.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → ∀ j ∈ F, ∀ t ∈ Dv.ticks (encl j),
    (P.actual r).state (Dv.X j) t + heldAt r (encl j) t ≤ mem (encl j)

/-- **PoUS: use at least, in the device model's held bit-ticks.** From `Pous.Period.PeriodHeldDeployed` (in the
properties tree, so the hypothesis `G`; untied): the device held at least a share of the claimed bit-ticks. Its `ε` is
δ + ε_max for one audit, a break of the ring audit excluded from the event, so a monitoring period's bound needs the
pooled figure, which does not exist. Open: `RunReads`, which carries `P2RingAuditSecure`, `DeviceModel` and
`Instantiates`, and `HeldClearsClaim`. The class is the windows between challenges on one device. A shape: its content
is its premises. -/
def PousUseAtLeast (G : Prop) (K : Finset Dv.ι) (shown : P.Run → Prop) (held : P.Run → Dv.ι → ℕ) (b : Dv.ι → ℕ)
    (ε : ℝ≥0∞) : Prop :=
  G → P.RunReads G shown ε → P.HeldClearsClaim K shown held b → P.UseAtLeast K held b ε

/-! ## The memory wipe -/

/-- **The wipe's information bound gives a width bound** (open, memory-accounting): in an accepted run where the prior
content surviving the wipe is under its bound in information, every server-day block `i` reads at most `b i` bits from
its own past. A block can read many bits that carry little information about prior content, so this needs a premise on
how prior content lies in the gates it reads; whether a block's input is to be stated in information is open. -/
def WipeInformationGivesWidth (survives : P.Run → Prop) (b : Dv.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → survives r → ∀ i ∈ Dv.serverDay, P.Holds r i (.inputAtMost [.past] (b i))

/-- **The memory wipe: input at most from its own past, for every server-day block**, at `E i + 8,600,000,000` bits,
`E i` the bits that leave the boundary during the wipe that opens block `i`, as the network attester counts them. From
`Pous.Guarantees.StreamedWipeMeets2600GB` (a guarantee layer this one may not import, so the hypothesis `G`; untied),
over a uniform prior, at `2⁻⁴⁰` plus the PRF distinguisher's advantage `adv`, with the fill the whole memory: so the
capacity figure (`CapacityAtMost`, the memory `memOf r i` at most `cap i`) is a hypothesis. Open: `RunReads` and
`WipeInformationGivesWidth`. A shape: its content is its premises. -/
def WipeInputAtMost (G : Prop) (survives : P.Run → Prop) (E : Dv.ι → ℕ) (memOf : P.Run → Dv.ι → ℕ) (cap : Dv.ι → ℕ)
    (adv : ℝ≥0∞) : Prop :=
  G → P.RunReads G survives (2⁻¹ ^ 40 + adv) → P.CapacityAtMost Dv.serverDay memOf cap →
    P.WipeInformationGivesWidth survives (fun i => E i + 8600000000) →
      P.Established Dv.serverDay (fun i => .inputAtMost [.past] (E i + 8600000000)) (2⁻¹ ^ 40 + adv)

/-! ## Network attestation -/

/-- **Every link passes through an attester** (assumed, about the site): in an accepted run, every server-day block
`i` reads at most `bIn i` bits from the outside world, the attester's ingress rows being its declared inputs, and its
output to the outside world is at most the attester's egress count `egress r i`. It is not `trusted-device` (that the
auditor's device and its host are honest): it is that no link of the audited server bypasses one. Soundness names it
a named assumption of the site, in `Core` beside the devices' vocabulary and listed in the lock by name; it is stated
here until it moves there. -/
def AttesterOnEveryLink (bIn : Dv.ι → ℕ) (egress : P.Run → Dv.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → ∀ i ∈ Dv.serverDay,
    P.Holds r i (.inputAtMost [.world] (bIn i)) ∧ ∃ B, P.blocks r = some B ∧ B.output i ≤ egress r i

/-- **The phases' results are the period's** (open, network-accounting with lean): a server-day is many execution
phases, and the registered gates of a phase include the state carried in from earlier ones; in an accepted run whose
`phases` phases each give their result `phaseShown e`, the state each reads is the fold of the earlier ones, so the
period's result `shown` holds. -/
def PhasesFoldToPeriod (phases : ℕ) (phaseShown : ℕ → P.Run → Prop) (shown : P.Run → Prop) : Prop :=
  ∀ r, P.accepts r → (∀ e < phases, phaseShown e r) → shown r

/-- **The record's unexplained bits are the block's undeclared output** (open, network-accounting): in an accepted run
whose egress record the declared computation explains up to the missable cost and the charges, every server-day
block's egress count is at most `bOut i`. -/
def RecordIsOutput (shown : P.Run → Prop) (egress : P.Run → Dv.ι → ℕ) (bOut : Dv.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → shown r → ∀ i ∈ Dv.serverDay, egress r i ≤ bOut i

/-- **Network attestation: input at most from the outside world, for every server-day block**, from
`AttesterOnEveryLink` alone, with no sampled proof behind it. A shape: its content is its premises. -/
def NetworkInputAtMost (bIn : Dv.ι → ℕ) (egress : P.Run → Dv.ι → ℕ) : Prop :=
  P.AttesterOnEveryLink bIn egress → P.Established Dv.serverDay (fun i => .inputAtMost [.world] (bIn i)) 0

/-- **Network attestation: output at most to the outside world, for every server-day block**: a width bound that
leaves the timing channel out. From `NetTiming.Attester.PhaseGetsBack`, which bounds how much of one execution phase's
egress record the declared computation does not explain, and says nothing about what enters. Its channel term, the log
of how many timing and failure patterns the operator can choose, is no width of any gate: whether the output property
is stated in information for that part, or a gate's time is given a width, is open for training.
`NetTiming.Guarantees.MonitoringPeriodLeakAccepted` is the bound in information, stated and not deployed. Open:
`RunReads`, `PhasesFoldToPeriod` and `RecordIsOutput`; assumed: `AttesterOnEveryLink`. A shape: its content is its
premises. -/
def NetworkOutputAtMost (phases : ℕ) (phaseShown : ℕ → P.Run → Prop) (shown : P.Run → Prop) (bIn bOut : Dv.ι → ℕ)
    (egress : P.Run → Dv.ι → ℕ) (ε : ℝ≥0∞) : Prop :=
  NetTiming.Attester.PhaseGetsBack →
    P.RunReads NetTiming.Attester.PhaseGetsBack (fun r => ∀ e < phases, phaseShown e r) ε →
    P.PhasesFoldToPeriod phases phaseShown shown → P.RecordIsOutput shown egress bOut →
    P.AttesterOnEveryLink bIn egress → P.Established Dv.serverDay (fun i => .outputAtMost (bOut i)) ε

/-! ## Registration -/

/-- **The registered input gates are the log's** (open, proofs): in an accepted run, `θ` is exactly the gates
`logged r` whose values open the roots the log held before the first epoch's first challenge, and a read counted as
registered is held by a registered read the verifier checked. -/
def RegisteredIsLog (logged : (r : P.Run) → Finset (Fin (P.actual r).N)) : Prop :=
  ∀ r, P.accepts r → P.θ r = logged r

/-- **Registration binds `θ`**: from `Verity.Guarantees.RegistrationSound` and `RegisteredIsLog`, an accepted run's
registered input gates are the log's. A shape: its content is its premises. -/
def RegistrationBinds (logged : (r : P.Run) → Finset (Fin (P.actual r).N)) : Prop :=
  Verity.Guarantees.RegistrationSound → P.RegisteredIsLog logged → ∀ r, P.accepts r → P.θ r = logged r

end ResourceAccounting.Division.Runs

namespace InferenceOnly

open ResourceAccounting

/-- **An actual circuit complies with the policy `pol`** at the registered input gates `θ`: they are input gates, and
`A` with them as its registered information satisfies `pol`. -/
def Compliant {L : Type*} (pol : Policy) (A : Placed L) (θ : Finset (Fin A.N)) : Prop :=
  ∃ h : ∀ a ∈ θ, A.isInput a = true, pol (.ofCircuit A.toCircuit θ h)

/-- **The policy's theorem** for the check `check` on a division: every actual circuit in the feasible set of a
division that passes the check complies with `pol`. -/
def PolicyTheorem {L : Type*} (check : Division L → Bool) (pol : Policy) : Prop :=
  ∀ (Dv : Division L) (A : Placed L) (π : Fin A.N → ℕ) (θ : Finset (Fin A.N)),
    check Dv = true → Dv.FeasibleSet A π θ → Compliant pol A θ

/-- The claim is of size at most `c`. -/
def sizeAtMost (c : ℕ) : Property → Bool
  | .sizeAtMost s => decide (s ≤ c)
  | _ => false

/-- The claim is of input at most `k` bits from the block's own past, another block and the outside world. -/
def freshAtMost (k : ℕ) : Property → Bool
  | .inputAtMost [.past, .block, .world] s => decide (s ≤ k)
  | _ => false

/-- **The inference-only policy's limits**, public parameters: the inference blocks' count `n`, size `c` and input
`k`; the free blocks' size `freeSize`; and in the theorem's bound, the free blocks' share `share` and the allowance
inside inference blocks `allowance`. -/
structure Limits where
  n : ℕ
  c : ℕ
  k : ℕ
  freeSize : ℕ
  share : ℝ
  allowance : ℝ

/-- **The inference-only check on a division**: at most `n` inference blocks, each claimed of size at most `c` and
input at most `k` bits; each free block claimed of size at most `freeSize`. -/
def check {L : Type*} (lim : Limits) (Dv : Division L) : Bool :=
  decide (Dv.inference.card ≤ lim.n) &&
    decide (∀ i ∈ Dv.inference, (Dv.claims i).any (sizeAtMost lim.c) = true ∧
      (Dv.claims i).any (freshAtMost lim.k) = true) &&
    decide (∀ i ∈ Dv.free, (Dv.claims i).any (sizeAtMost lim.freeSize) = true)

/-- **The inference-only policy** at the limits `lim`, with training efficiency `efficiency` (open on a declaration):
a declaration complies when its training efficiency is at most `share + allowance`. -/
def efficiencyAtMost (lim : Limits) (efficiency : RegisteredCircuit → ℝ) : Policy := fun d =>
  efficiency d ≤ lim.share + lim.allowance

/-- **The inference-only theorem** (open, training): every actual circuit in the feasible set of a division that
passes the inference-only check has training efficiency at most the free blocks' share plus the allowance. -/
def DivisionTheorem {L : Type*} (lim : Limits) (efficiency : RegisteredCircuit → ℝ) : Prop :=
  PolicyTheorem (check (L := L) lim) (efficiencyAtMost lim efficiency)

open Classical in
/-- **The check's inference clauses give `CompliesAtMost`** (open, lean): in the feasible set of a division that passes
the check, the inference blocks' gates, cut as a circuit with the registered input gates, satisfy
`CompliesAtMost c k n`, as `Slowdown.Guarantees.SlowdownRemote` takes it. -/
def InferenceBlocksComply (lim : Limits) : Prop :=
  ∀ {L : Type*} (Dv : Division L) (A : Placed L) (π : Fin A.N → ℕ) (θ : Finset (Fin A.N))
    (hθ : ∀ a ∈ θ, A.isInput a = true) (B : Dv.Gates A), check lim Dv = true →
    (∀ i, ∀ p ∈ Dv.claims i, B.Holds π θ i p) →
    CompliesAtMost lim.c lim.k lim.n
      (.ofCircuit (A.cut (Dv.inference.biUnion fun i => (B.block i).gates)) θ fun a ha => by
        simp [Placed.cut, hθ a ha])

/-- **The free blocks' size gives the residual** (open, training): in the feasible set, the residual's counted
multiplications `Mr A` are at most `κ/2` per unit of the free blocks' work, the form
`Slowdown.Assumptions.ResidualAtMost` bounds them in. -/
def FreeBlocksResidual {L : Type*} (Mr : Placed L → ℕ) (κ : ℝ) : Prop :=
  ∀ (Dv : Division L) (A : Placed L) (π : Fin A.N → ℕ) (θ : Finset (Fin A.N)) (B : Dv.Gates A),
    (∀ i, ∀ p ∈ Dv.claims i, B.Holds π θ i p) → (Mr A : ℝ) ≤ κ * (∑ i ∈ Dv.free, B.size π i : ℕ) / 2

/-- **The server-day blocks are serving blocks** (open, training): in the feasible set, some partition of a declaration
has serving blocks whose sizes and inputs (`ServingSizes`) are the server-day blocks' sizes and their inputs from their
own past and the outside world. -/
def ServerDayBlocksServing : Prop :=
  ∀ {L : Type*} (Dv : Division L) (A : Placed L) (π : Fin A.N → ℕ) (θ : Finset (Fin A.N)) (B : Dv.Gates A),
    (∀ i, ∀ p ∈ Dv.claims i, B.Holds π θ i p) →
    ∃ (d : RegisteredCircuit) (nB : ℕ) (Pd : Partition d.D nB) (S : Finset (Fin nB)) (cS kS : Fin nB → ℕ),
      ServingBlocks.ServingSizes d Pd S cS kS ∧
      ∀ u ∈ S, ∃ i ∈ Dv.serverDay, cS u = B.size π i ∧ kS u = B.input θ i [.past, .world]

end InferenceOnly

namespace ResourceAccounting

/-- **The chain**: over a monitoring period whose division `Dv` passes the check, with runs that fit the division,
the accounting statements `s` (statement `x` establishing `p x i` for every block `i`
of the class `K x`, except with probability at most `ε x`) that between them establish every claim of `Dv`, and the
policy's theorem, give that an accepted run's actual circuit complies with `pol`, except with probability at most
`∑ x ∈ s, ε x`. It reads `Established`, every block of a class: a statement in the tolerance form
(`EstablishedOutside`) reaches it only once the blocks outside its set are counted in the free block. -/
def Chain {L : Type*} {Dv : Division L} (P : Dv.Runs) (check : Division L → Bool) (pol : InferenceOnly.Policy)
    {κ : Type} (s : Finset κ) (K : κ → Finset Dv.ι) (p : κ → Dv.ι → Property) (ε : κ → ℝ≥0∞) : Prop :=
  check Dv = true → InferenceOnly.PolicyTheorem check pol → P.FitsDivision →
    (∀ i, ∀ q ∈ Dv.claims i, ∃ x ∈ s, i ∈ K x ∧ p x i = q) → (∀ x ∈ s, P.Established (K x) (p x) (ε x)) →
    ExceptWith P.pr P.accepts (fun r => InferenceOnly.Compliant pol (P.actual r) (P.θ r)) (∑ x ∈ s, ε x)

end ResourceAccounting
