module

public import Definitions.ResourceAccounting.Actual
public import Definitions.Core.Scheme
public import Definitions.Pouw.PearlC.TileSample
public import Definitions.Poub.Guarantees.RateAuditDeployed
public import Definitions.Pous.Guarantees.StreamedWipe
public import Definitions.ResourceAccounting.Network.AttesterScheme

/-!
# Blocks and their properties: the interface from accounting to policy

A **block** (`Block C`) is a set of a circuit's gates: of the actual circuit (`Placed`) or of the declared one
(`RegisteredCircuit`). Its suffix names its scale, not another kind of thing, and each existing use of the word is a
block at a scale: a proof block is a unit of a `Verity.Partition` (`Partition.gates`) and a `Verity.Scheme.Block`
placed by `Scheme.Placed`; an inference block is a unit of a registered circuit's `𝓘`; a server-day block is a
server-time block of `Definitions/Core/Blocks.lean` (`Verity.Block`, in the device model), a window of
`ResourceAccounting.Guarantees.Actual.ServerBlocksAt` and a serving block of `ServingBlocks.ServingSizes`; the whole
circuit is a block too. The block a location `X` and an interval `I` pick out is `Placed.region X I`; every block of a
statement lies inside one.

Accounting establishes that the actual circuit divides into blocks with some of four properties, each built on the
actual circuit's existing definitions:
* **size**: its work at a pricing (`Placed.work`), gates at unit pricing;
* **function**: a block of the declared circuit that the block's gates, cut as a circuit (`Placed.cut`), implement
  (`Verity.Implements`);
* **input** from a named source (`Source`), in bits (width): over `region X I`, from the outside world it is
  `Placed.ingress X I`, and from its own past and registered information it is the part of `Placed.state` it reads;
* **output** to the outside world, in bits (width): over `region X I`, `Placed.egress X I`.

The **statement** (`Statement`) is the public part, fixed before any coin: the blocks, their classes (server-day,
inference and free blocks), how they nest, and the properties claimed of each. A run's blocks (`Statement.Gates`) are
private. `FeasibleSet` is the feasible set's membership: an actual circuit with blocks having every claimed property.

Every accounting statement has one form: for every block of a class, a property holds, except with probability at
most `ε` (`Period.Established`, in the form `Scheme.GetsBack` uses). `UseAtLeast` states that at least `b` of a
capacity was used by declared work inside every block of a class (PoUW, PoUB, PoUS). `InputAtMost` and `OutputAtMost`
are two of the four properties themselves (the memory wipe, network attestation). One conversion,
`Period.FreeBlockAtMost`, turns capacity minus use into the free block's size or input at most; it alone carries the
capacity figure (`CapacityAtMost`) and the premise that declared work does not double as other work
(`DeclaredWorkExclusive`). Each instance takes the protocol's existing guarantee as its hypothesis, with the open
premises that tie it to the actual circuit: `RunReads` and one premise per instance.

**Against `Scheme`.** `Verity.Scheme` is what a protocol hands the proof layer and gets back (`Scheme.GetsBack`,
`Scheme.GetsBackPeriodRate`); this file is the function from that result to a block property. A scheme's result reaches
a block property through `RunReads` (its result read on the period's runs) and the instance's premise from its two
sides to the block's property; `b` is the declared amount less the tolerance (the `missable` cost, once for the period
under `GetsBackPeriodRate`) less the charges.

**What this is to replace, for lean to fold onto `Block`.** The server-time blocks and their carried state
(`Verity.Blocks`, `Blocks.carried`: a server-day block's input from its own past, in the device model); the serving
blocks' sizes and inputs (`ServingBlocks.ServingSizes`); the server blocks' windows and bounds (`ServerBlocksAt`,
`ServerBlocks`); the joint run's per-device work, reads and held bits (`ResourceAccounting.Run.work`, `read`, `held` in
`Definitions/Compose/Joint.lean`: `UseAtLeast`'s quantities in the device model); the policy as a condition on
declarations (`InferenceOnly.Policy`, against `Policy` in `BlockProperties/Policy.lean`); and the accounting envelopes
`Definitions/Core/Formalism.lean` names. `Placed.cut` and `Verity.Circuit.restrict` cut a block alike but for their
outputs.

Four points are not settled, and each is visible here:
1. *Which block the use is in*: `UseAtLeast` and `FreeBlockAtMost` range over a class of enclosing blocks, the whole
   circuit or the server-day blocks; each instance says which its evidence supports today.
2. *Work shown gives function*: `WorkGivesFunction`, its own open premise, read only by `PouwFunction`.
3. *How the check comes to run inside the outer proof*: not stated here.
4. *The wipe's unit*: its guarantee bounds information about prior content and a block's input is width;
   `WipeInformationGivesWidth` is the open premise between them, in `WipeInputAtMost`'s type.
-/

@[expose] public section

namespace ResourceAccounting

open Verity Finset
open scoped ENNReal

/-- **A block**: a set of the circuit `C`'s gates. -/
abbrev Block (C : Circuit) : Type := Finset (Fin C.N)

/-- **The source of a block's input**: registered information (the registered input gates `θ`); another block (placed
in the block's location during its interval, outside it); the block's own past (placed in its location before its
interval); the outside world (placed outside its location). The last three leave registered information out. -/
inductive Source where
  | registered
  | block
  | past
  | world
  deriving DecidableEq

/-- **A property of a block**, its bound in gates at a pricing (size) or in bits (input and output). -/
inductive Property where
  | sizeAtMost (k : ℕ)
  | function (f : Circuit)
  | inputAtMost (sources : List Source) (k : ℕ)
  | outputAtMost (k : ℕ)

/-- **The statement**, public and fixed before any coin: the blocks `ι`, the server-day, inference and free blocks
among them (a block in none is the whole circuit or a proof block), the block each lies inside, and the properties
claimed of each. -/
structure Statement where
  ι : Type
  [fintype : Fintype ι]
  [decEq : DecidableEq ι]
  serverDay : Finset ι
  inference : Finset ι
  free : Finset ι
  parent : ι → Option ι
  claims : ι → List Property

attribute [instance] Statement.fintype Statement.decEq

/-- The blocks of the class `K` inside block `i`. -/
def Statement.within (St : Statement) (K : Finset St.ι) (i : St.ι) : Finset St.ι :=
  K.filter fun j => St.parent j = some i

/-- **A statement's blocks in an actual circuit**: each block, inside the block `region (X i) [start i, stop i)` and
inside its parent, with its declared input and output gates, which its input and output bounds leave out. -/
structure Statement.Gates {L : Type*} (St : Statement) (A : Placed L) where
  gates : St.ι → Block A.toCircuit
  X : St.ι → Finset L
  start : St.ι → ℕ
  stop : St.ι → ℕ
  inside : ∀ i, gates i ⊆ A.region (X i) (Set.Ico (start i) (stop i))
  nest : ∀ i j, St.parent i = some j → gates i ⊆ gates j
  declared : St.ι → Finset (Fin A.N)

namespace Statement.Gates

variable {L : Type*} {St : Statement} {A : Placed L} (B : St.Gates A) (π : Fin A.N → ℕ) (θ : Finset (Fin A.N))

open Classical in
/-- The gates of a source, for block `i`. -/
noncomputable def source (i : St.ι) : Source → Finset (Fin A.N)
  | .registered => θ
  | .block => univ.filter fun g => g ∉ θ ∧ g ∉ B.gates i ∧ A.x g ∈ B.X i ∧ A.t g ∈ Set.Ico (B.start i) (B.stop i)
  | .past => univ.filter fun g => g ∉ θ ∧ A.x g ∈ B.X i ∧ A.t g < B.start i
  | .world => univ.filter fun g => g ∉ θ ∧ A.x g ∉ B.X i

/-- **Size** of block `i`: its work at the pricing `π`. -/
def size (i : St.ι) : ℕ := A.work π (B.gates i)

/-- **Input** of block `i` from `sources`, in bits: the width of the gates of those sources it reads, its declared
inputs left out. -/
noncomputable def input (i : St.ι) (sources : List Source) : ℕ :=
  A.width ((A.into (B.gates i) ∩ sources.toFinset.biUnion (B.source θ i)) \ B.declared i)

open Classical in
/-- **Output** of block `i` to the outside world, in bits: the width of its gates that a gate placed outside its
location reads, its declared outputs left out. -/
noncomputable def output (i : St.ι) : ℕ :=
  A.width ((B.gates i).filter fun g => g ∉ B.declared i ∧ ∃ h, A.x h ∉ B.X i ∧ g ∈ A.reads h)

/-- **Block `i` has the property `p`.** -/
def Holds (i : St.ι) : Property → Prop
  | .sizeAtMost k => B.size π i ≤ k
  | .function f => Verity.Implements f (A.cut (B.gates i))
  | .inputAtMost sources k => B.input θ i sources ≤ k
  | .outputAtMost k => B.output i ≤ k

end Statement.Gates

/-- **The feasible set**: `A` has blocks for the statement `St`, each with every property claimed of it, at the
pricing `π` and the registered input gates `θ`. -/
def FeasibleSet (St : Statement) {L : Type*} (A : Placed L) (π : Fin A.N → ℕ) (θ : Finset (Fin A.N)) : Prop :=
  ∃ B : St.Gates A, ∀ i, ∀ p ∈ St.claims i, B.Holds π θ i p

/-- **`P` fails on an accepted run with probability at most `ε`**, `pr` the audit game's probability on runs, as
`Scheme.GetsBack` reads it. -/
def ExceptWith {Ω : Type} (pr : (Ω → Prop) → ℝ≥0∞) (accepts : Ω → Prop) (P : Ω → Prop) (ε : ℝ≥0∞) : Prop :=
  pr (fun r => accepts r ∧ ¬ P r) ≤ ε

/-- **A monitoring period's runs**, as accounting reads them: the audit game's probability `pr` (monotone, and
subadditive over finite unions, as `Scheme.AccountingLayer` takes it), the verifier's acceptance, each run's actual
circuit with its pricing and registered input gates, and its blocks for the statement `St`. -/
structure Period (L : Type*) (St : Statement) where
  Run : Type
  pr : (Run → Prop) → ℝ≥0∞
  mono : ∀ p q : Run → Prop, (∀ r, p r → q r) → pr p ≤ pr q
  union : ∀ {κ : Type} (s : Finset κ) (p : κ → Run → Prop), pr (fun r => ∃ x ∈ s, p x r) ≤ ∑ x ∈ s, pr (p x)
  accepts : Run → Prop
  actual : Run → Placed L
  π : (r : Run) → Fin (actual r).N → ℕ
  θ : (r : Run) → Finset (Fin (actual r).N)
  blocks : (r : Run) → St.Gates (actual r)

namespace Period

variable {L : Type*} {St : Statement} (P : Period L St)

/-- Block `i` of run `r` has the property `p`. -/
def Holds (r : P.Run) (i : St.ι) (p : Property) : Prop := (P.blocks r).Holds (P.π r) (P.θ r) i p

/-- **Accounting establishes `p i` for every block `i` of the class `K`**, except with probability at most `ε`. -/
def Established (K : Finset St.ι) (p : St.ι → Property) (ε : ℝ≥0∞) : Prop :=
  ExceptWith P.pr P.accepts (fun r => ∀ i ∈ K, P.Holds r i (p i)) ε

/-- **The protocol's guarantee, read on the period's runs** (open, one per instance): the existing guarantee `G`, at
this deployment, bounds by `ε` the probability that an accepted run's own conclusion `shown` fails. -/
def RunReads (G : Prop) (shown : P.Run → Prop) (ε : ℝ≥0∞) : Prop := G → ExceptWith P.pr P.accepts shown ε

/-- **Use at least**: inside every block `i` of the class `K`, declared work used at least `b i` of a capacity, in the
quantity `use r i`, except with probability at most `ε`. -/
def UseAtLeast (K : Finset St.ι) (use : P.Run → St.ι → ℕ) (b : St.ι → ℕ) (ε : ℝ≥0∞) : Prop :=
  ExceptWith P.pr P.accepts (fun r => ∀ i ∈ K, b i ≤ use r i) ε

/-- **The capacity figure** (open, from the hardware description): every block `i` of the class `K` has at most
`cap i` of the resource, in the quantity `enclosing r i`. -/
def CapacityAtMost (K : Finset St.ι) (enclosing : P.Run → St.ι → ℕ) (cap : St.ι → ℕ) : Prop :=
  ∀ r, ∀ i ∈ K, enclosing r i ≤ cap i

/-- **Declared work does not double as other work** (open): in an accepted run, for every free block `j` of the class
`F`, its quantity `free r j` and the declared work's use inside its enclosing block `encl j` add up to at most the
enclosing block's own quantity. -/
def DeclaredWorkExclusive (F : Finset St.ι) (encl : St.ι → St.ι) (free use enclosing : P.Run → St.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → ∀ j ∈ F, free r j + use r (encl j) ≤ enclosing r (encl j)

/-- **The one conversion: capacity minus use bounds the free block.** For every free block `j` of the class `F`, whose
enclosing block `encl j` is of the class `K`: use at least `b` inside every block of `K`, the capacity figure and
exclusivity give `j`'s quantity (its size, or its input from a source) at most `cap (encl j) − b (encl j)`, except
with the use's probability. At `free r j` the free block's size or input, the conclusion is `Established F` of
`sizeAtMost` or `inputAtMost` at that bound. Proved by `ResourceAccounting.SecurityProofs.BlockProperties.freeBlockAtMost`. -/
def FreeBlockAtMost : Prop :=
  ∀ (K F : Finset St.ι) (encl : St.ι → St.ι) (free use enclosing : P.Run → St.ι → ℕ) (cap b : St.ι → ℕ)
    (ε : ℝ≥0∞), (∀ j ∈ F, encl j ∈ K) →
    P.UseAtLeast K use b ε → P.CapacityAtMost K enclosing cap → P.DeclaredWorkExclusive F encl free use enclosing →
      ExceptWith P.pr P.accepts (fun r => ∀ j ∈ F, free r j ≤ cap (encl j) - b (encl j)) ε

/-! ## The instances -/

/-- **The spend is the inference blocks' work** (open, compute-accounting): in an accepted run whose spend clears
the claim, inside every block `i` of `K` the inference blocks have work at least `b i` at `π`. -/
def SpendIsWork (K : Finset St.ι) (shown : P.Run → Prop) (b : St.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → shown r → ∀ i ∈ K, b i ≤ ∑ j ∈ St.within St.inference i, (P.blocks r).size (P.π r) j

/-- **PoUW (Pearl+): use at least, in size (work at `π`).** From `Pouw.PearlC.GGapSampledDraw` (proved): the
accepted adversary's spend is at least a share of the claimed work. Open: `RunReads` and `SpendIsWork`. Its evidence
is pooled over the period's devices, so the class it supports today is the whole circuit, not the server-day blocks. -/
def PouwUseAtLeast {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : Pouw.PearlC.CostModel Q R S) (Pr : Pouw.PearlC.Protocol Q R S) (TR : Pouw.PearlC.TileRules Q R S)
    (D : Pouw.PearlC.Layout → Prop) (Wd : Pouw.PearlC.Layout → ℕ → Pouw.PearlC.Decl → ℚ) (g : ℝ) (δ : ℕ → ℕ → ℝ)
    (K : Finset St.ι) (shown : P.Run → Prop) (b : St.ι → ℕ) (ε : ℝ≥0∞) : Prop :=
  Pouw.PearlC.GGapSampledDraw CM Pr TR D Wd g δ → P.RunReads (Pouw.PearlC.GGapSampledDraw CM Pr TR D Wd g δ) shown ε →
    P.SpendIsWork K shown b →
    P.UseAtLeast K (fun r i => ∑ j ∈ St.within St.inference i, (P.blocks r).size (P.π r) j) b ε

/-- **Work shown gives function** (open, compute-accounting; unsettled point 2): in an accepted run whose spend
clears the claim, every inference block `j` implements its block `f j` of the declared circuit. It is where the
premise that the work shown does not double as other work enters for function, apart from the conversion. -/
def WorkGivesFunction (shown : P.Run → Prop) (f : St.ι → Circuit) : Prop :=
  ∀ r, P.accepts r → shown r → ∀ j ∈ St.inference, P.Holds r j (.function (f j))

/-- **PoUW's other half: every inference block's function**, from its spend reading and `WorkGivesFunction`. -/
def PouwFunction (G : Prop) (shown : P.Run → Prop) (f : St.ι → Circuit) (ε : ℝ≥0∞) : Prop :=
  G → P.RunReads G shown ε → P.WorkGivesFunction shown f → P.Established St.inference (fun j => .function (f j)) ε

/-- **The tier reads are the blocks' input of registered information** (open, memory-accounting): in an accepted run
whose reads clear the claim, inside every block `i` of `K` the inference blocks read at least `b i` bits of registered
information. -/
def ReadsAreInput (K : Finset St.ι) (shown : P.Run → Prop) (b : St.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → shown r → ∀ i ∈ K,
    b i ≤ ∑ j ∈ St.within St.inference i, (P.blocks r).input (P.θ r) j [.registered]

/-- **PoUB: use at least, in bits read of registered information.** From
`ResourceAccounting.Guarantees.RemoteRunConservationDeployed` (proved): every opening passes only while the session's
tier reads are at least a share of its bits. Open: `RunReads` and `ReadsAreInput`. Its session is one node's, so the
class it supports is the server-day blocks once the session's windows are cut at the node's wipes. -/
def PoubUseAtLeast (K : Finset St.ι) (shown : P.Run → Prop) (b : St.ι → ℕ) (ε : ℝ≥0∞) : Prop :=
  Guarantees.RemoteRunConservationDeployed → P.RunReads Guarantees.RemoteRunConservationDeployed shown ε →
    P.ReadsAreInput K shown b →
    P.UseAtLeast K (fun r i => ∑ j ∈ St.within St.inference i, (P.blocks r).input (P.θ r) j [.registered]) b ε

/-- **The device model's held bits are the blocks' state** (open, memory-accounting): in an accepted run whose held
bit-ticks clear the claim, every block `i` of `K` (a window between two challenges) carries at least `b i` bits of
registered information in its location's state at its start (`Placed.stateIn` at `θ`). -/
def HeldIsState (K : Finset St.ι) (shown : P.Run → Prop) (b : St.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → shown r → ∀ i ∈ K, b i ≤ (P.actual r).stateIn ((P.blocks r).X i) ((P.blocks r).start i) (P.θ r)

/-- **PoUS: use at least, in bits held of registered information.** From `Pous.Period.PeriodHeldDeployed` (proved, in
the properties tree, which this tree does not read, so the hypothesis `held`): the device held at least a share of the
claimed bit-ticks at the challenge times. Open: `RunReads` and `HeldIsState`. The class it supports is the windows
between challenges on one device. -/
def PousUseAtLeast (held : Prop) (K : Finset St.ι) (shown : P.Run → Prop) (b : St.ι → ℕ) (ε : ℝ≥0∞) : Prop :=
  held → P.RunReads held shown ε → P.HeldIsState K shown b →
    P.UseAtLeast K (fun r i => (P.actual r).stateIn ((P.blocks r).X i) ((P.blocks r).start i) (P.θ r)) b ε

/-- **The wipe's information bound gives a width bound** (open, memory-accounting; unsettled point 4): in an
accepted run where the prior content surviving the wipe is under its bound in information, every server-day block `i`
reads at most `b i` bits from its own past. It is not a consequence: a block can read many bits that carry little
information about prior content, so a width bound needs a premise on how prior content lies in the gates it reads. -/
def WipeInformationGivesWidth (survives : P.Run → Prop) (b : St.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → survives r → ∀ i ∈ St.serverDay, P.Holds r i (.inputAtMost [.past] (b i))

/-- **The memory wipe: input at most from its own past, in bits, for every server-day block.** From
`Pous.Guarantees.StreamedWipeMeets2600GB` (proved), a bound in information on the prior content an accepted wipe
keeps. Open: `RunReads` and `WipeInformationGivesWidth`. -/
def WipeInputAtMost (survives : P.Run → Prop) (b : St.ι → ℕ) (ε : ℝ≥0∞) : Prop :=
  Pous.Guarantees.StreamedWipeMeets2600GB → P.RunReads Pous.Guarantees.StreamedWipeMeets2600GB survives ε →
    P.WipeInformationGivesWidth survives b → P.Established St.serverDay (fun i => .inputAtMost [.past] (b i)) ε

/-- **The attester's record is the blocks' boundary** (open, network-accounting): in an accepted run whose egress
used exceeds the egress credited by at most the missable cost and the charges, every server-day block `i` reads at
most `bIn i` bits from, and writes at most `bOut i` bits to, the outside world beyond its declared inputs and outputs. -/
def RecordIsBoundary (shown : P.Run → Prop) (bIn bOut : St.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → shown r → ∀ i ∈ St.serverDay,
    P.Holds r i (.inputAtMost [.world] (bIn i)) ∧ P.Holds r i (.outputAtMost (bOut i))

/-- **Network attestation: input at most from the outside world, in bits, for every server-day block.** From
`NetTiming.Attester.PhaseGetsBack` (proved). Open: `RunReads` and `RecordIsBoundary`. -/
def NetworkInputAtMost (shown : P.Run → Prop) (bIn bOut : St.ι → ℕ) (ε : ℝ≥0∞) : Prop :=
  NetTiming.Attester.PhaseGetsBack → P.RunReads NetTiming.Attester.PhaseGetsBack shown ε →
    P.RecordIsBoundary shown bIn bOut → P.Established St.serverDay (fun i => .inputAtMost [.world] (bIn i)) ε

/-- **Network attestation: output at most to the outside world, in bits, for every server-day block.** As
`NetworkInputAtMost`. `NetTiming.Guarantees.MonitoringPeriodLeakAccepted` is the attester's bound in information. -/
def NetworkOutputAtMost (shown : P.Run → Prop) (bIn bOut : St.ι → ℕ) (ε : ℝ≥0∞) : Prop :=
  NetTiming.Attester.PhaseGetsBack → P.RunReads NetTiming.Attester.PhaseGetsBack shown ε →
    P.RecordIsBoundary shown bIn bOut → P.Established St.serverDay (fun i => .outputAtMost (bOut i)) ε

/-- **The committed values are the actual circuit's** (open, soundness): in an accepted run whose committed values
are correct outside the missable blocks, every inference block `j` has size at most `c j` and input at most `k j`
bits from its own past, another block and the outside world, as its block of the declared circuit has. -/
def CommittedIsActual (shown : P.Run → Prop) (c k : St.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → shown r → ∀ j ∈ St.inference,
    P.Holds r j (.sizeAtMost (c j)) ∧ P.Holds r j (.inputAtMost [.past, .block, .world] (k j))

/-- **The proof layer: size at most, for every inference block.** From the sampled proofs' guarantee
(`Flock.SecurityProofs.SampledKSLiveV5`, in the proofs tree, so the hypothesis `G`): an accepted run's committed
values are correct outside a set the sample would likely have hit. Open: `RunReads` and `CommittedIsActual`. None of
the five accounting instances gives this property or the next. -/
def InferenceSizeAtMost (G : Prop) (shown : P.Run → Prop) (c k : St.ι → ℕ) (ε : ℝ≥0∞) : Prop :=
  G → P.RunReads G shown ε → P.CommittedIsActual shown c k → P.Established St.inference (fun j => .sizeAtMost (c j)) ε

/-- **The proof layer: input at most from its own past, another block and the outside world, in bits, for every
inference block.** As `InferenceSizeAtMost`. -/
def InferenceInputAtMost (G : Prop) (shown : P.Run → Prop) (c k : St.ι → ℕ) (ε : ℝ≥0∞) : Prop :=
  G → P.RunReads G shown ε → P.CommittedIsActual shown c k →
    P.Established St.inference (fun j => .inputAtMost [.past, .block, .world] (k j)) ε

end Period

end ResourceAccounting
