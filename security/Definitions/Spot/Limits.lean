module

public import Definitions.Core.Draw
public import Definitions.Core.Structure
public import Mathlib.Data.Nat.Choose.Basic

/-!
# The sampled proofs' limits, computed from the circuit's description

The exfiltration, infiltration and inference-only limits, each stated as a consequence of the sampling layer's
soundness and computed from the circuit's description alone: its Definitions and its nodes, never the unrolled circuit
and no side text. A verifier summarizes each Definition once and multiplies through a node's count.

* `Description`: each Definition's output and fresh-input port widths, and each node's Definition and count (a batch's
  members or a scan's iterations). Its units are the nodes' members in node order.
* `Description.blockWidth`, `Description.blockFresh`: a unit's output bits and fresh input bits, its Definition's.
* `Description.Describes`, `Description.DescribesFresh`: the unrolled circuit's partition is the description's.
* `WidthsMeet`: the computed widths meet the width hypothesis of the audit's influence bound.
* `SampledSound`: the shape of the sampling layer's soundness; `OneValuePerWire`: the wires premise.
* `ExfiltrationLimit`, `InfiltrationLimit`, `InferenceOnlyLimit`: the three limits.

Nothing here is proved. Each statement says which theorem it follows from.
-/

@[expose] public section

namespace Spot.Limits

open Finset Verity Verity.Game
open scoped ENNReal

/-- One Definition as a description lists it: the widths in bits of its output ports, and of its input ports that read
fresh input (an input gate outside θ). A port that reads another block's output is a wire, an edge of the circuit, and
reads nothing fresh. -/
structure Defn where
  outs : List ℕ
  fresh : List ℕ

/-- **A circuit's description**: its Definitions, and its nodes, each a Definition's index and the number of units
that run it (a batch's members or a scan's iterations). Its units are the nodes' members, in node order. -/
structure Description where
  defs : List Defn
  nodes : List (ℕ × ℕ)

namespace Description

variable (D : Description)

/-- Definition `i`; an index past the list has no ports. -/
def defn (i : ℕ) : Defn := D.defs.getD i ⟨[], []⟩

/-- Definition `i`'s output bits, summed once over its ports. -/
def outWidth (i : ℕ) : ℕ := (D.defn i).outs.sum

/-- Definition `i`'s fresh input bits, summed once over its ports. -/
def freshWidth (i : ℕ) : ℕ := (D.defn i).fresh.sum

/-- The number of units: the nodes' counts, summed. -/
def units : ℕ := (D.nodes.map Prod.snd).sum

/-- The Definition that unit `j` runs: that of the node holding member `j`, in node order. -/
def defAt : List (ℕ × ℕ) → ℕ → ℕ
  | [], _ => 0
  | (i, c) :: ns, j => if j < c then i else defAt ns (j - c)

/-- **A proof block's output bits**, computed from the description: its Definition's output bits. This is the width
`hb` that `FlockSoundness.Audit.Partition.audit_exfiltration` takes as an argument today. -/
def blockWidth (j : ℕ) : ℕ := D.outWidth (defAt D.nodes j)

/-- An inference block's fresh input bits, computed from the description: its Definition's fresh input bits. -/
def blockFresh (j : ℕ) : ℕ := D.freshWidth (defAt D.nodes j)

/-- The most fresh input bits any unit reads, one term per node. -/
def maxFresh : ℕ := (D.nodes.map fun x => D.freshWidth x.1).foldr max 0

/-- **The partition is the description's**: it has the description's units, and each unit's exits (`Circuit.exits`)
hold at most its Definition's output bits.

It holds by construction once Lean defines the unrolled circuit from its description, as the readers of a wired
circuit will (the circuit with its wires as edges). Core's `Circuit` is still flat, so until then it is a hypothesis. -/
def Describes {C : Circuit} {n : ℕ} (P : Partition C n) : Prop :=
  D.units = n ∧ ∀ u : Fin n, C.width (C.exits (P.gates u)) ≤ D.blockWidth u

/-- **The inference partition is the description's**: it has the description's units, and each unit's external reads
(`Structure.externalIn`: input gates outside θ) hold at most its Definition's fresh input bits. A wire from another
block is a computed gate, so it is no external read. A hypothesis for the same reason as `Describes`. -/
def DescribesFresh (d : RegisteredCircuit) {m : ℕ} (I : Partition d.D m) : Prop :=
  D.units = m ∧ ∀ u : Fin m, d.D.width (Structure.externalIn d (I.gates u)) ≤ D.blockFresh u

end Description

/-- **The computed widths meet the influence bound's width hypothesis**: for a partition its description describes,
the exits of any set `B` of units hold at most the sum of their computed widths. This is the hypothesis `hw` of
`FlockSoundness.Audit.Partition.audit_exfiltration` and `card_influenceSet_le_harm` with `hb := blockWidth` and `K B`
the exits of `B`'s gates, which `separates_exits` shows separate `B`.

It follows from `Describes`, since the exits of a union lie in the union of the exits; it is not proved yet. -/
def WidthsMeet : Prop :=
  ∀ (D : Description) (C : Circuit) (n : ℕ) (P : Partition C n), D.Describes P →
    ∀ B : Finset (Fin n), C.width (C.exits (B.biUnion P.gates)) ≤ ∑ u ∈ B, D.blockWidth u

section Limits

variable {C : Circuit} {n : ℕ} (P : Partition C n) (L : SamplingDesign (Fin n)) {α : Type} (g : Game α)
  (accept : α → Prop) (X : α → C.Trace)

/-- **The sampling layer's soundness, in the shape the limits read**: whatever the prover does, the verifier accepts
while the committed trace's incorrect proof blocks form a set the sample misses with probability at least `δ`, with
probability at most `ε`.

A stand-in: the limits take `Spot.Sampling.Sound` once it is stated, and this definition goes. Today's theorem of this
shape is `FlockSoundness.Audit.audit_profile`, from knowledge soundness and binding. -/
def SampledSound (δ ε : ℝ≥0∞) : Prop :=
  Verity.Sound g accept (fun a => L.missProb (P.incorrect (X a)) < δ) ε

/-- **One value on each wire** (the wires premise): the verifier accepts while some block's opening `rd a h w` of a
wire `w` it reads from another block differs from the writer's committed value `X a w`, with probability at most `εw`.

Not yet true by construction: the deployed path checks each wire against a link and takes this as an open premise. Once
a wire has one commitment, the writer's row that the reader refers to, it holds with `εw = 0`, and the hypothesis
goes. -/
def OneValuePerWire (rd : α → (h w : Fin C.N) → BitVec (C.w w)) (εw : ℝ≥0∞) : Prop :=
  Verity.Sound g accept (fun a => ∀ h w, w ∈ C.reads h → P.unit h ≠ P.unit w → rd a h w = X a w) εw

/-- What both limits on the delivered wires `Dl` take: the description describes `P`; the sampling layer is sound at
`δ` with error `ε`; one value on each wire, up to `εw`; the committed inputs are the anchors `a₀`, up to `δin`; `Dl`
is committed; `H` bounds the computed widths of every set that escapes with probability at least `δ` (the profile's
harm bound); and `k + 1` units are caught except with probability below `δ`. -/
def Takes (D : Description) (rd : α → (h w : Fin C.N) → BitVec (C.w w)) (a₀ : C.Trace) (Dl : Finset (Fin C.N))
    (δ ε εw δin : ℝ≥0∞) (H k : ℕ) : Prop :=
  D.Describes P ∧ SampledSound P L g accept X δ ε ∧ OneValuePerWire P g accept X rd εw ∧
    Verity.Sound g accept (fun a => C.InputsAgree (X a) a₀) δin ∧ Dl ⊆ P.committed ∧
    (∀ B : Finset (Fin n), δ ≤ L.missProb B → ∑ u ∈ B, D.blockWidth u ≤ H) ∧ L.miss (k + 1) < δ

/-- **Exfiltration**: a prover that picks its strategy `σ m` from a uniform message `m`, and a receiver `rcv` that sees
only the delivered values, succeed together (accepted, and `rcv` returns `m`) with probability at most
`2^H · ∑_{j ≤ k} C(n, j) / |M|` plus the errors. With `|M| = 2^ℓ`, at most `H + log₂ ∑_{j ≤ k} C(n, j)` bits leave.
`H` and the computed widths come from the description; `consumers.exfiltration_bound` computes this figure.

It would follow from `FlockSoundness.Audit.Partition.audit_exfiltration` with `hb := blockWidth`, `hw` by
`WidthsMeet` and `hsep` by `separates_exits`, restated over core's circuit. Not yet true: the wires premise
(`OneValuePerWire`) stays a hypothesis until a wire has one commitment. -/
def ExfiltrationLimit (D : Description) (rd : α → (h w : Fin C.N) → BitVec (C.w w)) (a₀ : C.Trace)
    (Dl : Finset (Fin C.N)) (δ ε εw δin : ℝ≥0∞) (H k : ℕ) : Prop :=
  Takes P L g accept X D rd a₀ Dl δ ε εw δin H k →
    ∀ (M : Type) [Fintype M] [Nonempty M] (σ : M → Strategy g) (rcv : ((t : Dl) → BitVec (C.w t.1)) → M),
      avg (fun m => prob (fun a => accept a ∧ rcv (fun t => X a t.1) = m) g (σ m)) ≤
        ((2 ^ H * ∑ j ∈ range (k + 1), n.choose j : ℕ) : ℝ≥0∞) / (Fintype.card M : ℝ≥0∞) + (δ + ε + εw + δin)

/-- **Infiltration** (influence): a set `Y` of delivered values, fixed before any prover by the circuit, the partition,
the anchors, the design and `δ`, with at most `2^H · ∑_{j ≤ k} C(n, j)` members, such that the verifier accepts while
the delivered values lie outside `Y` with probability at most the errors. The incorrect proof blocks steer the delivered
values by at most `H + log₂ ∑_{j ≤ k} C(n, j)` bits; `consumers.infiltration_bound` computes `H`.

It would follow from `FlockSoundness.Audit.Partition.audit_influence` with `card_influenceSet_le_harm` and
`card_admissible_le` at `hb := blockWidth`, restated over core's circuit. Not yet true: the wires premise stays a
hypothesis until a wire has one commitment. -/
def InfiltrationLimit (D : Description) (rd : α → (h w : Fin C.N) → BitVec (C.w w)) (a₀ : C.Trace)
    (Dl : Finset (Fin C.N)) (δ ε εw δin : ℝ≥0∞) (H k : ℕ) : Prop :=
  Takes P L g accept X D rd a₀ Dl δ ε εw δin H k →
    ∃ Y : Finset ((t : Dl) → BitVec (C.w t.1)), Y.card ≤ 2 ^ H * ∑ j ∈ range (k + 1), n.choose j ∧
      ∀ σ : Strategy g, prob (fun a => accept a ∧ (fun t : Dl => X a t.1) ∉ Y) g σ ≤ δ + ε + εw + δin

end Limits

/-- **Inference-only**, at the wired circuit: when the inference partition `I` of the registered circuit is its
description's, and no Definition reads more than `k` fresh bits, every inference block reads at most `k` fresh bits
(a wire's bits are not fresh). And the verifier accepts while the committed trace leaves the registered circuit beyond
an escaped set of incorrect proof blocks, or a wire carries two values, with probability at most `ε + εw`.

`k` and the fresh bits are computed from the description, one term per node, and nothing reads a record's side text.
The first part would follow from `DescribesFresh`, the second from `SampledSound` and `OneValuePerWire`. Not yet true:
the wires premise stays a hypothesis until a wire has one commitment, and lean's reader of the wired circuit is what
will compute `DescribesFresh`'s description. -/
def InferenceOnlyLimit (d : RegisteredCircuit) {n m : ℕ} (P : Partition d.D n) (I : Partition d.D m)
    (L : SamplingDesign (Fin n)) {α : Type} (g : Game α) (accept : α → Prop) (X : α → d.D.Trace)
    (rd : α → (h w : Fin d.D.N) → BitVec (d.D.w w)) (DI : Description) (k : ℕ) (δ ε εw : ℝ≥0∞) : Prop :=
  DI.DescribesFresh d I → SampledSound P L g accept X δ ε → OneValuePerWire P g accept X rd εw → DI.maxFresh ≤ k →
    (∀ u : Fin m, d.D.width (Structure.externalIn d (I.gates u)) ≤ k) ∧
      Verity.Sound g accept (fun a => L.missProb (P.incorrect (X a)) < δ ∧
        ∀ h w, w ∈ d.D.reads h → P.unit h ≠ P.unit w → rd a h w = X a w) (ε + εw)

end Spot.Limits
