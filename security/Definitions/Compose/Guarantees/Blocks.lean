module

public import Definitions.Compose.Blocks
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
* **PoUB** (`PoubUseAtLeast`) and **PoUS** (`PousUseAtLeast`): open and not placed; no term of the inference-only
  bound reads memory reads or memory held.
* **The memory wipe** (`WipeInputAtMost`): input at most from its own past. Its guarantee's module is a guarantee
  layer this one may not import, so the hypothesis `G`.
* **Network attestation** (`NetworkInputAtMost`, `NetworkOutputAtMost`): input at most from `AttesterOnEveryLink`
  and the attester's ingress count; output at most from `NetTiming.Attester.PhaseGetsBack`, `PhasesFoldToPeriod`,
  `RecordIsOutput` and `AttesterOnEveryLink`.
* **Registration** (`RegistrationBinds`): every registered input gate of a run opens a root of the log, from
  `Verity.Guarantees.RegistrationSound` and `RegisteredReadsOpenLog`.
* **The proof layer** is `Verity.Scheme.GetsBack` itself and gives no block property here.

The policy is a condition on declarations (`InferenceOnly.Policy`), read of an actual circuit with its registered input
gates (`InferenceOnly.Compliant`); its theorem over a division's check (`Definitions/Compose/Blocks.lean`) is
`InferenceOnly.PolicyTheorem`. `Chain` is the
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

/-- **The spend is the inference blocks' work after the salt** (open, compute-accounting): in an accepted run whose
spend clears the claim, inside every block `i` of `K` the inference blocks directly inside it whose windows start at
or after their epoch's challenge tick `salt j` have work at least `b i` at `π`. Those blocks are siblings, which the
division keeps disjoint, so the sum counts no gate twice (`ResourceAccounting.SecurityProofs.BlockProperties.sizeAdds`).
`b` is work at `π` and the spend is in multiply-accumulates: the conversion between them is this premise's content. -/
def SpendIsWork (K : Finset Dv.ι) (salt : Dv.ι → ℕ) (shown : P.Run → Prop) (b : Dv.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → shown r → ∀ i ∈ K,
    b i ≤ ∑ j ∈ Dv.within (Dv.inference.filter fun j => salt j ≤ (Dv.win j).start) i, P.size r j

/-- **PoUW (Pearl+): use at least, in size.** From `Pouw.PearlC.GGapSampledDraw`: the accepted adversary's spend is at
least a share of the claimed work. Open: `RunReads` and `SpendIsWork`. Its use is the sum `sizeExclusive` bounds, so it
feeds the conversion. Its evidence is pooled over the period's devices, so the class it supports today is the whole
circuit. A shape: its content is its premises. -/
def PouwUseAtLeast {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : Pouw.PearlC.CostModel Q R S) (Pr : Pouw.PearlC.Protocol Q R S) (TR : Pouw.PearlC.TileRules Q R S)
    (D : Pouw.PearlC.Layout → Prop) (Wd : Pouw.PearlC.Layout → ℕ → Pouw.PearlC.Decl → ℚ) (g : ℝ) (δ : ℕ → ℕ → ℝ)
    (K : Finset Dv.ι) (salt : Dv.ι → ℕ) (shown : P.Run → Prop) (b : Dv.ι → ℕ) (ε : ℝ≥0∞) : Prop :=
  Pouw.PearlC.GGapSampledDraw CM Pr TR D Wd g δ → P.RunReads (Pouw.PearlC.GGapSampledDraw CM Pr TR D Wd g δ) shown ε →
    P.SpendIsWork K salt shown b →
      P.UseAtLeast K (fun r i => ∑ j ∈ Dv.within (Dv.inference.filter fun j => salt j ≤ (Dv.win j).start) i,
        P.size r j) b ε

/-! ## The declared blocks run -/

/-- **The declared blocks run** (open, compute-accounting with soundness): every inference block `j` of the class `K`
(the credited blocks) whose window starts at or after its epoch's challenge tick `salt j`, outside a set whose declared use `decl` is at most the tolerance `tol`, implements
its declared block `f j`, at size at most `c j` (the declared size times a stated overhead at the pricing) and input at
most `k j` bits from its own past, another block and the outside world, except with probability at most `ε`.
Nothing in Verity proves that the prover ran the declared circuit: the proofs show committed values consistent with it,
and useful work shows an amount of work. It is owed an argument from PoUW's coverage of the declared work, with the
uncovered share and the incorrect proof blocks the sample missed counted in the free block (owners compute-accounting
and soundness). An inference block outside `K` or before its salt has no row here, so a division the chain reads
declares it free. -/
def DeclaredBlocksRun (K : Finset Dv.ι) (salt : Dv.ι → ℕ) (f : Dv.ι → Circuit) (c k : Dv.ι → ℕ) (decl : Finset Dv.ι → ℕ) (tol : ℕ)
    (ε : ℝ≥0∞) : Prop :=
  P.EstablishedOutside (K.filter fun j => j ∈ Dv.inference ∧ salt j ≤ (Dv.win j).start)
    (fun j => [.function (f j), .sizeAtMost (c j), .inputAtMost [.past, .block, .world] (k j)]) decl tol ε

/-! ## PoUB -/

/-- **PoUB: use at least, in registered reads counted with multiplicity** (open, memory-accounting with training), from
`ResourceAccounting.Guarantees.RemoteRunConservationDeployed`. Not placed: which block's unregistered input this
bounds and which term of the bound it makes smaller is not decided. -/
def PoubUseAtLeast (K : Finset Dv.ι) (b : Dv.ι → ℕ) (ε : ℝ≥0∞) : Prop := P.UseAtLeast K P.registeredReads b ε

/-! ## PoUS -/

/-- **PoUS: use at least, in the device model's held bit-ticks `held`** (open, memory-accounting with training), from
`Pous.Period.PeriodHeldDeployed`. Not placed: which block's unregistered input this bounds and which term of the bound
it makes smaller is not decided. -/
def PousUseAtLeast (K : Finset Dv.ι) (held : P.Run → Dv.ι → ℕ) (b : Dv.ι → ℕ) (ε : ℝ≥0∞) : Prop :=
  P.UseAtLeast K held b ε

/-! ## The memory wipe -/

/-- **The wipe's information bound gives a width bound** (open, memory-accounting): in an accepted run where the prior
content surviving the wipe is under its bound in information, every server-day block `i` reads at most `b i` bits from
its own past. A block can read many bits that carry little information about prior content, so this needs a premise on
how prior content lies in the gates it reads; whether a block's input is to be stated in information is open. -/
def WipeInformationGivesWidth (survives : P.Run → Prop) (b : Dv.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → survives r → ∀ i ∈ Dv.serverDay, P.Holds r i (.inputAtMost [.past] (b i))

/-- **The memory wipe: input at most from its own past, for every server-day block**, at `E i + 8,600,000,000` bits,
`E i` the bits that leave the boundary during the wipe that opens block `i`: untied, nothing here reads the network
attester's count over the wipe's interval. From
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
`i` reads from the outside world at most the attester's ingress count `ingress r i` bits, and its output to the
outside world is at most the attester's egress count `egress r i`. It is not `trusted-host` (that the auditor's device
and its host are honest): it is that no link of the audited server bypasses one. Soundness names it a named assumption
of the site, in `Core` beside the devices' vocabulary and listed in the lock by name; it is stated here until it moves
there. -/
def AttesterOnEveryLink (ingress egress : P.Run → Dv.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → ∀ i ∈ Dv.serverDay, ∃ B, P.blocks r = some B ∧
    B.input (P.θ r) i [.world] ≤ ingress r i ∧ B.output i ≤ egress r i

/-- **The phases' results are the period's** (open, network-accounting with lean): a server-day is many execution
phases, and the registered gates of a phase include the state carried in from earlier ones; in an accepted run whose
`phases` phases each give their result `phaseShown e`, the state each reads is the fold of the earlier ones, so the
period's result `shown` holds. Its body is the implication alone: the fold of the state is in this docstring, until the
chain's statement, with the state and the verifier's log, replaces it. -/
def PhasesFoldToPeriod (phases : ℕ) (phaseShown : ℕ → P.Run → Prop) (shown : P.Run → Prop) : Prop :=
  ∀ r, P.accepts r → (∀ e < phases, phaseShown e r) → shown r

/-- **The record's unexplained bits are the block's undeclared output** (open, network-accounting): in an accepted run
whose egress record the declared computation explains up to the missable cost and the charges, every server-day
block's egress count is at most `bOut i`. -/
def RecordIsOutput (shown : P.Run → Prop) (egress : P.Run → Dv.ι → ℕ) (bOut : Dv.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → shown r → ∀ i ∈ Dv.serverDay, egress r i ≤ bOut i

/-- **Network attestation: input at most from the outside world, for every server-day block**, from
`AttesterOnEveryLink` and the attester's ingress count at most `bIn i`, with no sampled proof behind it. A shape: its
content is its premises. -/
def NetworkInputAtMost (ingress egress : P.Run → Dv.ι → ℕ) (bIn : Dv.ι → ℕ) : Prop :=
  P.AttesterOnEveryLink ingress egress → (∀ r, P.accepts r → ∀ i ∈ Dv.serverDay, ingress r i ≤ bIn i) →
    P.Established Dv.serverDay (fun i => .inputAtMost [.world] (bIn i)) 0

/-- **Network attestation: output at most to the outside world, for every server-day block**: a width bound that
leaves the timing channel out. From `NetTiming.Attester.PhaseGetsBack`, which bounds how much of one execution phase's
egress record the declared computation does not explain, and says nothing about what enters. Its channel term, the log
of how many timing and failure patterns the operator can choose, is no width of any gate: whether the output property
is stated in information for that part, or a gate's time is given a width, is open for training.
`NetTiming.Guarantees.MonitoringPeriodLeakAccepted` is the bound in information, stated and not deployed. Open:
`RunReads`, `PhasesFoldToPeriod` and `RecordIsOutput`; assumed: `AttesterOnEveryLink`. A shape: its content is its
premises. -/
def NetworkOutputAtMost (phases : ℕ) (phaseShown : ℕ → P.Run → Prop) (shown : P.Run → Prop) (bOut : Dv.ι → ℕ)
    (ingress egress : P.Run → Dv.ι → ℕ) (ε : ℝ≥0∞) : Prop :=
  NetTiming.Attester.PhaseGetsBack →
    P.RunReads NetTiming.Attester.PhaseGetsBack (fun r => ∀ e < phases, phaseShown e r) ε →
    P.PhasesFoldToPeriod phases phaseShown shown → P.RecordIsOutput shown egress bOut →
    P.AttesterOnEveryLink ingress egress → P.Established Dv.serverDay (fun i => .outputAtMost (bOut i)) ε

/-! ## Registration -/

/-- **The registered reads open the log** (open, proofs): in an accepted run, every gate of `θ` is an input gate whose
value opens, at the position the verifier derives from the declared circuit, a root of the log `log`. The log is one
for the period, fixed before the first epoch's first challenge, so it does not depend on the run. Open: `opensAt r g
root` has no definition on the actual circuit (in code it is `Flock.Registered.check` holding a port's rows to a root
of the verifier's file); what defines it is the open question. -/
def RegisteredReadsOpenLog {Root : Type} (log : Finset Root)
    (opensAt : (r : P.Run) → Fin (P.actual r).N → Root → Prop) : Prop :=
  ∀ r, P.accepts r → ∀ g ∈ P.θ r, (P.actual r).isInput g = true ∧ ∃ root ∈ log, opensAt r g root

/-- **Registration binds `θ`**: from `Verity.Guarantees.RegistrationSound` (the log is append-only and its heads are
what the verifier reads) and `RegisteredReadsOpenLog`, every gate an accepted run counts as registered opens a root
of the log. A logged value read as a fresh input is not registered, so the converse is no guarantee. A shape: its
content is its premises. -/
def RegistrationBinds {Root : Type} (log : Finset Root)
    (opensAt : (r : P.Run) → Fin (P.actual r).N → Root → Prop) : Prop :=
  Verity.Guarantees.RegistrationSound → P.RegisteredReadsOpenLog log opensAt →
    ∀ r, P.accepts r → ∀ g ∈ P.θ r, ∃ root ∈ log, opensAt r g root

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
(`EstablishedOutside`) reaches it only once the blocks outside its set are counted in the free block. An inference
block no row covers (before its epoch's salt, or outside the credited class) is free work: the division declares it a
free block, whose size the conversion bounds. -/
def Chain {L : Type*} {Dv : Division L} (P : Dv.Runs) (check : Division L → Bool) (pol : InferenceOnly.Policy)
    {κ : Type} (s : Finset κ) (K : κ → Finset Dv.ι) (p : κ → Dv.ι → Property) (ε : κ → ℝ≥0∞) : Prop :=
  check Dv = true → InferenceOnly.PolicyTheorem check pol → P.FitsDivision →
    (∀ i, ∀ q ∈ Dv.claims i, ∃ x ∈ s, i ∈ K x ∧ p x i = q) → (∀ x ∈ s, P.Established (K x) (p x) (ε x)) →
    ExceptWith P.pr P.accepts (fun r => InferenceOnly.Compliant pol (P.actual r) (P.θ r)) (∑ x ∈ s, ε x)

end ResourceAccounting
