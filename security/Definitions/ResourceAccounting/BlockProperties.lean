module

public import Definitions.ResourceAccounting.Actual
public import Definitions.Core.Blocks
public import Definitions.Core.CircuitLevels

/-!
# Blocks and their properties: the interface from accounting to policy

A block is Core's `Verity.Block`: a window of ticks and the gates of a circuit it computes there (here the actual
circuit, `Placed`). Accounting establishes that the actual circuit divides into blocks nested by scale, and some of four
properties for each. The scales are `Verity.Partition`s of the actual circuit's computed gates (`security/Definitions/
Core/Partition.lean`: every computed gate in exactly one unit, input gates in none), nested by `Verity.Partition.Refines`:

* the **server-day blocks** and **hidden hardware's work**, a partition `outer` of the whole circuit (`kind u` says
  which: a server-day block, or hidden hardware);
* the **inference blocks** and **free blocks** refining them, a partition `inner` with `inner.Refines outer`, a class
  per unit (`cls u`: inference, free, or hidden under a hidden parent);
* the **proof blocks** refine the inference blocks; they are the proof layer's own `Verity.Partition` and carry no block
  property here.

Coverage, disjointness and the tree shape are the partitions' own (`Partition.input_iff`, `Partition.gates` a function,
`Refines.sub`), so no field and no clause states them. Each block's location and window are fixed data (`locIn`/`winIn`,
`locOut`/`winOut`), not the run's choice.

The four properties, each built on the actual circuit's definitions:
* **size**: its work at a pricing (`Placed.work`), in gates;
* **function**: a circuit that its gates, cut as a circuit (`Placed.cut`), implement (`Verity.Implements`). It is
  extensional: the cut may hold any further gates and input gates, so it bounds neither size nor input;
* **input** from named sources (`Source`), in bits (width);
* **output** to the outside world, in bits (width).

A monitoring period's runs (`Runs`) carry, per run, the actual circuit, its pricing and registered input gates, and the
two partitions the prover committed before any coin. `Realizes` is the feasible-set membership: the run's blocks have
every property claimed of them. Every accounting statement is `ExceptWith pr accepts shown ε`: the predicate `shown`
fails on an accepted run with probability at most `ε`. `UseAtLeast` states a lower bound on a used capacity, and the one
conversion `FreeBlockAtMost` turns capacity minus use into a bound on the sum of the free blocks directly inside each
server-day block; it alone carries the capacity figure (`CapacityAtMost`, the hardware description, assumed) and the
premise that declared work does not double as other work (`DeclaredWorkExclusive`).
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

/-- **The class of a unit of `inner`**: an inference block or a free block (inside a server-day block), or hidden (a unit
inside hidden hardware's work, which has no established property). -/
inductive Class where
  | inference
  | free
  | hidden
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
blocks and hidden hardware (`kind`); `inner` refines it into inference blocks, free blocks and hidden units (`cls`), a
unit being hidden exactly when its server-day-or-hidden parent is hidden hardware (`hiddenIff`). Each block's location
and window are fixed data. -/
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
  kind : (r : Run) → Fin (nOut r) → Bool
  locOut : (r : Run) → Fin (nOut r) → Finset L
  winOut : (r : Run) → Fin (nOut r) → Window
  claimsOut : (r : Run) → Fin (nOut r) → List Property
  nIn : (r : Run) → ℕ
  inner : (r : Run) → Partition (actual r).toCircuit (nIn r)
  ref : (r : Run) → (inner r).Refines (outer r)
  cls : (r : Run) → Fin (nIn r) → Class
  locIn : (r : Run) → Fin (nIn r) → Finset L
  winIn : (r : Run) → Fin (nIn r) → Window
  claimsIn : (r : Run) → Fin (nIn r) → List Property
  hiddenIff : ∀ r u, cls r u = Class.hidden ↔ kind r ((ref r).parent u) = false

namespace Runs

variable {L : Type*} (P : Runs L)

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

/-- Block `u`'s size in run `r`, its work at the run's pricing, in gates. -/
def sizeIn (r : P.Run) (u : Fin (P.nIn r)) : ℕ := (P.actual r).work (P.π r) ((P.inner r).gates u)

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

/-- **The run realizes its claims** (the feasible-set membership for `r`): `θ` is input gates of at most the registered
width, the pricing charges every computed gate, and every block has each property claimed of it. -/
noncomputable def Realizes (r : P.Run) : Prop :=
  (∀ a ∈ P.θ r, (P.actual r).isInput a = true) ∧ (P.actual r).width (P.θ r) ≤ P.registered ∧
    (∀ g ∈ (P.actual r).computed, 1 ≤ P.π r g) ∧
    (∀ u, ∀ p ∈ P.claimsIn r u, P.HoldsIn r u p) ∧ (∀ i, ∀ p ∈ P.claimsOut r i, P.HoldsOut r i p)

/-- **Every accepted run realizes its claims** (open): the bridge from the statements to the feasible set. -/
def Fits : Prop := ∀ r, P.accepts r → P.Realizes r

/-- **The protocol's guarantee, read on the period's runs** (open, one per instance): the guarantee `G`, at this
deployment, bounds by `ε` the probability that an accepted run's own conclusion `shown` fails. -/
def RunReads (G : Prop) (shown : P.Run → Prop) (ε : ℝ≥0∞) : Prop := G → ExceptWith P.pr P.accepts shown ε

/-- **Use at least**: inside every server-day block `i`, at least `b r i` of a capacity was used, in the quantity
`use r i`, except with probability at most `ε`. -/
def UseAtLeast (use b : (r : P.Run) → Fin (P.nOut r) → ℕ) (ε : ℝ≥0∞) : Prop :=
  ExceptWith P.pr P.accepts (fun r => ∀ i, P.kind r i = true → b r i ≤ use r i) ε

/-- **The capacity figure**: every server-day block `i` has at most `cap r i` of the resource, in the quantity
`enclosing r i`. It is the hardware description, an assumption nothing in Verity tests (owner none: the hardware
description). -/
def CapacityAtMost (enclosing cap : (r : P.Run) → Fin (P.nOut r) → ℕ) : Prop :=
  ∀ r, ∀ i, P.kind r i = true → enclosing r i ≤ cap r i

/-- **Declared work does not double as other work**: in an accepted run, inside every server-day block `i`, the free
blocks' quantity `free r i` and the use `use r i` add up to at most `i`'s own quantity `enclosing r i`. For size it
follows from the partitions (`ResourceAccounting.SecurityProofs.BlockProperties.sizeExclusive`); for other quantities it
is open. -/
def DeclaredWorkExclusive (free use enclosing : (r : P.Run) → Fin (P.nOut r) → ℕ) : Prop :=
  ∀ r, P.accepts r → ∀ i, P.kind r i = true → free r i + use r i ≤ enclosing r i

/-- **The one conversion: capacity minus use bounds the free blocks.** Inside every server-day block `i`: a use lower
bound, the capacity figure and exclusivity give the free blocks' quantity `free r i` at most `cap r i − b r i`, except
with the use's probability. Proved by `ResourceAccounting.SecurityProofs.BlockProperties.freeBlockAtMost`. -/
def FreeBlockAtMost : Prop :=
  ∀ (free use enclosing cap b : (r : P.Run) → Fin (P.nOut r) → ℕ) (ε : ℝ≥0∞),
    P.UseAtLeast use b ε → P.CapacityAtMost enclosing cap → P.DeclaredWorkExclusive free use enclosing →
      ExceptWith P.pr P.accepts (fun r => ∀ i, P.kind r i = true → free r i ≤ cap r i - b r i) ε

end Runs

end ResourceAccounting
