module

public import Definitions.ResourceAccounting.Actual
public import Definitions.Core.Blocks

/-!
# Blocks and their properties: the interface from accounting to policy

TO BE REPLACED: `Division` is a chain of `Verity.Partition`s under `Refines` with claims per unit. Each scale (the
server-day blocks; the inference and free blocks refining them, a class per unit; the proof blocks refining the
inference blocks) is a partition of the actual circuit, so `parent`, `depth`, `Gates.covers`, `children`, `nest` and
`disjoint` and the check's leaf and forest clauses go, and the claims, `Established`, `UseAtLeast` and the conversion
read units.

A block is Core's (`Verity.Block`): a window of ticks and the gates of a circuit it computes there. Here the circuit is
the actual circuit (`Placed`). Its suffix names its scale: a proof block is a unit of a `Verity.Partition` and a
`Verity.Scheme.Block`; an inference block is a unit of a registered circuit's `𝓘`; a server-day block is a server-time
block of `Definitions/Core/Blocks.lean`; the whole circuit is a block too.

Accounting establishes that the actual circuit divides into blocks with some of four properties, each built on the
actual circuit's definitions:
* **size**: its work at a pricing (`Placed.work`);
* **function**: a circuit that its gates, cut as a circuit (`Placed.cut`), implement (`Verity.Implements`). It is
  extensional: the cut may hold any further gates and input gates, so it bounds neither size nor input;
* **input** from named sources (`Source`), in bits (width);
* **output** to the outside world, in bits (width).

**The division** (`Division`) is committed by the prover before any coin and is private under the outer proof: the
blocks, their classes (server-day, inference and free blocks), how they nest, each block's location and window, the
registered width, and the properties claimed of each. A run's blocks (`Division.Gates`) are gates of its actual circuit
that the division's structure fixes: every computed gate is in a top block, every gate of a block with children is in
one of its children, siblings are disjoint, and each block lies in its location during its window. `FeasibleSet` is the
feasible set's membership: blocks with every claimed property, at a pricing that charges every computed gate and with
registered input gates of at most the registered width.

Every accounting statement has one form: for every block of a class, a property holds, except with probability at
most `ε` (`Division.Runs.Established`); a sampled statement holds outside a set of blocks whose declared use is at
most the tolerance (`Division.Runs.EstablishedOutside`). `UseAtLeast` states that at least `b` of a capacity was used
inside every block of a class. One conversion, `FreeBlockAtMost`, turns capacity minus use into a bound on the sum of
the free blocks' quantities directly inside each block; it alone carries the capacity figure (`CapacityAtMost`) and the premise that declared work does not
double as other work (`DeclaredWorkExclusive`), which for size follows from the division
(`ResourceAccounting.SecurityProofs.BlockProperties.sizeExclusive`). The policy's check is in `Definitions/Compose/Blocks.lean`; the
protocols' instances, the policy's theorem and the chain are in `Definitions/Compose/Guarantees/Blocks.lean`.

**Against `Scheme`.** `Verity.Scheme` is what a protocol hands the proof layer and gets back (`Scheme.GetsBack`); this
file is the form a block property takes once a scheme's result is read on the actual circuit.

`Placed` is in `ResourceAccounting`, which `Core` may not import, so the form sits beside it.
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

/-- **A property of a block**, its bound in work at a pricing (size) or in bits (input and output). An input or output
bound counts the block's declared ports too: the division fixes it before any coin. -/
inductive Property where
  | sizeAtMost (k : ℕ)
  | function (f : Circuit)
  | inputAtMost (sources : List Source) (k : ℕ)
  | outputAtMost (k : ℕ)

/-- **The division**, committed by the prover before any coin and private under the outer proof (the policy's check
reads it inside that proof): the blocks `ι`, the server-day, inference and free blocks among them, the block each lies
inside (`none` for a top block), a depth that grows from a block to each block inside it (so `parent` is a forest),
each block's location `X` and window `win`, the registered width, and the properties claimed of each. -/
structure Division (L : Type*) where
  ι : Type
  [fintype : Fintype ι]
  [decEq : DecidableEq ι]
  serverDay : Finset ι
  inference : Finset ι
  free : Finset ι
  parent : ι → Option ι
  depth : ι → ℕ
  depth_lt : ∀ i j, parent i = some j → depth j < depth i
  X : ι → Finset L
  win : ι → Window
  registered : ℕ
  claims : ι → List Property

attribute [instance] Division.fintype Division.decEq

namespace Division

variable {L : Type*} (Dv : Division L)

/-- The ticks of block `i`'s window. -/
def ticks (i : Dv.ι) : Set ℕ := Set.Ico (Dv.win i).start ((Dv.win i).start + (Dv.win i).len)

/-- The blocks of the class `K` directly inside block `i`. -/
def within (K : Finset Dv.ι) (i : Dv.ι) : Finset Dv.ι := K.filter fun j => Dv.parent j = some i

/-- **A division's blocks in an actual circuit**: block `i` is a `Verity.Block` of `A` in its window, lying in its
location during it. Every computed gate is in a top block, every gate of a block with children is in one of its
children, a block lies inside its parent, and siblings are disjoint. -/
structure Gates (A : Placed L) where
  block : Dv.ι → Verity.Block A.toCircuit
  win : ∀ i, (block i).win = Dv.win i
  inside : ∀ i, (block i).gates ⊆ A.region (Dv.X i) (Dv.ticks i)
  covers : ∀ g ∈ A.computed, ∃ i, Dv.parent i = none ∧ g ∈ (block i).gates
  nest : ∀ i j, Dv.parent i = some j → (block i).gates ⊆ (block j).gates
  children : ∀ i j, Dv.parent i = some j → ∀ g ∈ (block j).gates, ∃ i', Dv.parent i' = some j ∧ g ∈ (block i').gates
  disjoint : ∀ i i', i ≠ i' → Dv.parent i = Dv.parent i' → Disjoint (block i).gates (block i').gates

namespace Gates

variable {Dv} {A : Placed L} (B : Dv.Gates A) (π : Fin A.N → ℕ) (θ : Finset (Fin A.N))

open Classical in
/-- The gates of a source, for block `i`. -/
noncomputable def source (i : Dv.ι) : Source → Finset (Fin A.N)
  | .registered => θ
  | .block => univ.filter fun g => g ∉ θ ∧ g ∉ (B.block i).gates ∧ A.x g ∈ Dv.X i ∧ A.t g ∈ Dv.ticks i
  | .past => univ.filter fun g => g ∉ θ ∧ A.x g ∈ Dv.X i ∧ A.t g < (Dv.win i).start
  | .world => univ.filter fun g => g ∉ θ ∧ A.x g ∉ Dv.X i

/-- **Size** of block `i`: its work at the pricing `π`. -/
def size (i : Dv.ι) : ℕ := A.work π (B.block i).gates

/-- **Input** of block `i` from `sources`, in bits: the width of the gates of those sources it reads. -/
noncomputable def input (i : Dv.ι) (sources : List Source) : ℕ :=
  A.width (A.into (B.block i).gates ∩ sources.toFinset.biUnion (B.source θ i))

open Classical in
/-- **Output** of block `i` to the outside world, in bits: the width of its gates that a gate placed outside its
location reads. It counts gates, so it leaves out what the time a gate is read at can carry. -/
noncomputable def output (i : Dv.ι) : ℕ :=
  A.width ((B.block i).gates.filter fun g => ∃ h, A.x h ∉ Dv.X i ∧ g ∈ A.reads h)

/-- **Reads of registered information** by block `i`, in bits counted with multiplicity: for each of its gates, the
width of the registered gates it reads. It is a quantity, not one of the four properties. -/
def registeredReads (i : Dv.ι) : ℕ := ∑ g ∈ (B.block i).gates, A.width (A.reads g ∩ θ)

/-- **Block `i` has the property `p`.** -/
def Holds (i : Dv.ι) : Property → Prop
  | .sizeAtMost k => B.size π i ≤ k
  | .function f => Verity.Implements f (A.cut (B.block i).gates)
  | .inputAtMost sources k => B.input θ i sources ≤ k
  | .outputAtMost k => B.output i ≤ k

/-- **Sizes add over siblings**: the children of block `j` in `S` have work at most `j`'s, at any pricing. Proved by
`ResourceAccounting.SecurityProofs.BlockProperties.sizeAdds`. -/
def SizeAdds : Prop :=
  ∀ (j : Dv.ι) (S : Finset Dv.ι), (∀ i ∈ S, Dv.parent i = some j) → ∑ i ∈ S, B.size π i ≤ B.size π j

end Gates

/-- **The feasible set**: `θ` is input gates of `A` of at most the registered width, the pricing `π` charges every
computed gate, and `A` has blocks for the division `Dv`, each with every property claimed of it. -/
def FeasibleSet (A : Placed L) (π : Fin A.N → ℕ) (θ : Finset (Fin A.N)) : Prop :=
  (∀ a ∈ θ, A.isInput a = true) ∧ A.width θ ≤ Dv.registered ∧ (∀ g ∈ A.computed, 1 ≤ π g) ∧
    ∃ B : Dv.Gates A, ∀ i, ∀ p ∈ Dv.claims i, B.Holds π θ i p

end Division

/-- **`P` fails on an accepted run with probability at most `ε`**, `pr` the audit game's probability on runs, as
`Scheme.GetsBack` reads it. -/
def ExceptWith {Ω : Type} (pr : (Ω → Prop) → ℝ≥0∞) (accepts : Ω → Prop) (P : Ω → Prop) (ε : ℝ≥0∞) : Prop :=
  pr (fun r => accepts r ∧ ¬ P r) ≤ ε

/-- **A monitoring period's runs**, as accounting reads them: the audit game's probability `pr` (monotone, and
subadditive over finite unions, as `Scheme.AccountingLayer` takes it), the verifier's acceptance, and each run's actual
circuit, pricing, registered input gates and, where it has them, its blocks for the division `Dv`. -/
structure Division.Runs {L : Type*} (Dv : Division L) where
  Run : Type
  pr : (Run → Prop) → ℝ≥0∞
  mono : ∀ p q : Run → Prop, (∀ r, p r → q r) → pr p ≤ pr q
  union : ∀ {κ : Type} (s : Finset κ) (p : κ → Run → Prop), pr (fun r => ∃ x ∈ s, p x r) ≤ ∑ x ∈ s, pr (p x)
  accepts : Run → Prop
  actual : Run → Placed L
  π : (r : Run) → Fin (actual r).N → ℕ
  θ : (r : Run) → Finset (Fin (actual r).N)
  blocks : (r : Run) → Option (Dv.Gates (actual r))

namespace Division.Runs

variable {L : Type*} {Dv : Division L} (P : Dv.Runs)

/-- Block `i` of run `r` has the property `p`: the run has blocks for the division, and its block `i` has `p`. -/
def Holds (r : P.Run) (i : Dv.ι) (p : Property) : Prop :=
  ∃ B, P.blocks r = some B ∧ B.Holds (P.π r) (P.θ r) i p

/-- **Accounting establishes `p i` for every block `i` of the class `K`**, except with probability at most `ε`. -/
def Established (K : Finset Dv.ι) (p : Dv.ι → Property) (ε : ℝ≥0∞) : Prop :=
  ExceptWith P.pr P.accepts (fun r => ∀ i ∈ K, P.Holds r i (p i)) ε

/-- **Accounting establishes every claim of `ps i` for every block `i` of `K` outside a set `M` whose declared use
`use M` is at most the tolerance `tol`**, except with probability at most `ε`: the form a sampled statement gives. -/
def EstablishedOutside (K : Finset Dv.ι) (ps : Dv.ι → List Property) (use : Finset Dv.ι → ℕ) (tol : ℕ)
    (ε : ℝ≥0∞) : Prop :=
  ExceptWith P.pr P.accepts (fun r => ∃ M ⊆ K, use M ≤ tol ∧ ∀ i ∈ K \ M, ∀ q ∈ ps i, P.Holds r i q) ε

/-- **The runs fit the division** (open): on every accepted run, `θ` is input gates of at most the registered width
(registration's row, `RegistrationBinds`, with the log's width), the pricing charges every computed gate, and the run
has blocks for the division: its computed gates lie in its top blocks' locations and windows. -/
def FitsDivision : Prop :=
  ∀ r, P.accepts r → (∀ a ∈ P.θ r, (P.actual r).isInput a = true) ∧ (P.actual r).width (P.θ r) ≤ Dv.registered ∧
    (∀ g ∈ (P.actual r).computed, 1 ≤ P.π r g) ∧ (P.blocks r).isSome

/-- **The protocol's guarantee, read on the period's runs** (open, one per instance): the guarantee `G`, at this
deployment, bounds by `ε` the probability that an accepted run's own conclusion `shown` fails. -/
def RunReads (G : Prop) (shown : P.Run → Prop) (ε : ℝ≥0∞) : Prop := G → ExceptWith P.pr P.accepts shown ε

/-- **Use at least**: inside every block `i` of the class `K`, at least `b i` of a capacity was used, in the quantity
`use r i`, except with probability at most `ε`. -/
def UseAtLeast (K : Finset Dv.ι) (use : P.Run → Dv.ι → ℕ) (b : Dv.ι → ℕ) (ε : ℝ≥0∞) : Prop :=
  ExceptWith P.pr P.accepts (fun r => ∀ i ∈ K, b i ≤ use r i) ε

/-- **The capacity figure**: every block `i` of the class `K` has at most `cap i` of the resource, in the quantity
`enclosing r i`. It is the hardware description, an assumption nothing in Verity tests (owner none: the hardware
description). -/
def CapacityAtMost (K : Finset Dv.ι) (enclosing : P.Run → Dv.ι → ℕ) (cap : Dv.ι → ℕ) : Prop :=
  ∀ r, ∀ i ∈ K, enclosing r i ≤ cap i

/-- **Declared work does not double as other work**: in an accepted run, inside every block `i` of the class `K`, the
quantities `free r j` of the free blocks `j` of `F` directly inside `i`, summed, and the use inside `i` add up to at
most `i`'s own quantity. For size, with `use` the sum over inference blocks directly inside `i` and `F` disjoint from
them, it follows from the division (`ResourceAccounting.SecurityProofs.BlockProperties.sizeExclusive`); for other
quantities it is open. -/
def DeclaredWorkExclusive (K F : Finset Dv.ι) (free use enclosing : P.Run → Dv.ι → ℕ) : Prop :=
  ∀ r, P.accepts r → ∀ i ∈ K, ∑ j ∈ Dv.within F i, free r j + use r i ≤ enclosing r i

/-- **The one conversion: capacity minus use bounds the free blocks.** Inside every block `i` of the class `K`: use at
least `b` inside every block of `K`, the capacity figure and exclusivity give the free blocks of `F` directly inside
`i`, summed, at most `cap i − b i`, except with the use's probability. Proved by
`ResourceAccounting.SecurityProofs.BlockProperties.freeBlockAtMost`. -/
def FreeBlockAtMost : Prop :=
  ∀ (K F : Finset Dv.ι) (free use enclosing : P.Run → Dv.ι → ℕ) (cap b : Dv.ι → ℕ) (ε : ℝ≥0∞),
    P.UseAtLeast K use b ε → P.CapacityAtMost K enclosing cap → P.DeclaredWorkExclusive K F free use enclosing →
      ExceptWith P.pr P.accepts (fun r => ∀ i ∈ K, ∑ j ∈ Dv.within F i, free r j ≤ cap i - b i) ε

end Division.Runs

end ResourceAccounting
