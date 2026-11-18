module

public import Definitions.Spot.Sampling.Analysis
public import Definitions.Spot.Sampling.Layer
public import Definitions.Core.Structure
public import Mathlib.Data.Nat.Choose.Basic

/-!
# The sampled proofs' limits, computed from the circuit's description

The exfiltration, infiltration and inference-only limits, each stated as a consequence of the sampling layer's
theorem (`Spot.Sampling.Sound`) and computed from the circuit's description alone: its Definitions and its nodes, never the unrolled circuit
and no side text. A verifier summarizes each Definition once and multiplies through a node's count.

* `Description`: each Definition's output and fresh-input port widths, and each node's Definition and count (a batch's
  members or a scan's iterations). Its units are the nodes' members in node order.
* `Description.blockWidth`, `Description.blockFresh`: a unit's output bits and fresh input bits, its Definition's.
* `Description.Describes`, `Description.DescribesFresh`: the unrolled circuit's partition is the description's.
* `WidthsMeet`: the computed widths meet the width hypothesis of the audit's influence bound.
* `OneValuePerWire`: the wires premise, over `Spot.Sampling`'s audit and analysis.
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
`hb` that `Spot.Sampling.Partition.audit_exfiltration` takes as an argument today. -/
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
`Spot.Sampling.Partition.audit_exfiltration` and `card_influenceSet_le_harm` with `hb := blockWidth` and `K B`
the exits of `B`'s gates, which `separates_exits` shows separate `B`.

It follows from `Describes`, since the exits of a union lie in the union of the exits; it is not proved yet. -/
def WidthsMeet : Prop :=
  ∀ (D : Description) (C : Circuit) (n : ℕ) (P : Partition C n), D.Describes P →
    ∀ B : Finset (Fin n), C.width (C.exits (B.biUnion P.gates)) ≤ ∑ u ∈ B, D.blockWidth u

section Limits

open Spot.Sampling

variable {C : Circuit} {n : ℕ} {U : Type} [DecidableEq U] {L : SamplingDesign U} {Reg : Type}
  {session : Finset U → Reg → Game Bool}

/-- **One value on each wire** (the wires premise): whatever the prover does, the verifier accepts while some block's
opening `rd σ h w` of a wire `w` it reads from another block differs from the writer's committed value
`tr (A.committedOf σ) w`, with probability at most `εw`.

Not yet true by construction: the deployed path checks each wire against a link and takes this as an open premise. Once
a wire has one commitment, the writer's row that the reader refers to, it holds with `εw = 0`, and the hypothesis
goes. -/
def OneValuePerWire (P : Partition C n) (A : ExtractionAnalysis L Reg session) (tr : A.Tr → C.Trace)
    (rd : Strategy (audit L Reg session) → (h w : Fin C.N) → BitVec (C.w w)) (εw : ℝ≥0∞) : Prop :=
  ∀ σ : Strategy (audit L Reg session), prob (fun o => o.1 = true ∧ ∃ h w, w ∈ C.reads h ∧ P.unit h ≠ P.unit w ∧
    rd σ h w ≠ tr (A.committedOf σ) w) (audit L Reg session) σ ≤ εw

/-- What both limits on the delivered wires `Dl` take, for a design `L` over units `U` whose sampled units are `P`'s
proof blocks under `blk` (`U = Fin n` and `blk` the identity for one level; the last level's blocks for a
`Verity.Levels` design) and an analysis `A` whose transcripts read as traces of `C` under `tr`:
* the sampling layer's theorem (`Spot.Sampling.Sound`), and `A`'s knowledge and binding errors `εks` and `εbind`;
* `A`'s incorrect units are `P`'s incorrect proof blocks under `blk`;
* one value on each wire, up to `εw`, and the committed inputs are the anchors `a₀`, up to `δin`;
* the description describes `P`, and `Dl` is committed;
* `H` bounds the computed widths of every set of proof blocks that escapes with probability at least `δ` (the
  profile's harm bound), and `k` bounds the size of every such set (the count at `δ` over `P`'s proof blocks, which
  at a `Verity.Levels` design with `blk` onto `Levels.blocks` is `Levels.IsCountBound`; the miss `L.miss` over every set
  of units would say nothing there). -/
def Takes (P : Partition C n) (L : SamplingDesign U) (blk : Fin n ↪ U) (A : ExtractionAnalysis L Reg session)
    (tr : A.Tr → C.Trace) (rd : Strategy (audit L Reg session) → (h w : Fin C.N) → BitVec (C.w w))
    (D : Description) (a₀ : C.Trace) (Dl : Finset (Fin C.N)) (δ εks εbind εw δin : ℝ≥0∞) (H k : ℕ) : Prop :=
  Spot.Sampling.Sound ∧ A.KnowledgeSound (fun _ _ => εks) ∧ A.BindSound (fun _ _ => εbind) ∧
    (∀ X, A.incorrect X = (P.incorrect (tr X)).map blk) ∧ OneValuePerWire P A tr rd εw ∧
    (∀ σ : Strategy (audit L Reg session), prob (fun o => o.1 = true ∧
      ¬ C.InputsAgree (tr (A.committedOf σ)) a₀) (audit L Reg session) σ ≤ δin) ∧
    D.Describes P ∧ Dl ⊆ P.committed ∧
    (∀ B : Finset (Fin n), δ ≤ L.missProb (B.map blk) → ∑ u ∈ B, D.blockWidth u ≤ H) ∧
    (∀ B : Finset (Fin n), δ ≤ L.missProb (B.map blk) → B.card ≤ k)

/-- **Exfiltration**: a prover that picks its audit strategy `σ m` from a uniform message `m`, and a receiver `rcv`
that sees only the delivered values, succeed together (accepted, and `rcv` returns `m`) with probability at most
`2^H · ∑_{j ≤ k} C(n, j) / |M|` plus the errors. With `|M| = 2^ℓ`, at most `H + log₂ ∑_{j ≤ k} C(n, j)` bits leave.
`H` and the computed widths come from the description; `consumers.exfiltration_bound` computes this figure.

It would follow from `Spot.Sampling.Sound` at the event "the incorrect set is missed with probability below `δ`", as
`Spot.Sampling.Partition.audit_exfiltration` follows from `audit_profile`, with `hb := blockWidth`, `hw` by
`WidthsMeet` and `hsep` by `separates_exits`, restated over core's circuit. Not yet true: the wires premise
(`OneValuePerWire`) stays a hypothesis until a wire has one commitment. -/
def ExfiltrationLimit (P : Partition C n) (L : SamplingDesign U) (blk : Fin n ↪ U)
    (A : ExtractionAnalysis L Reg session) (tr : A.Tr → C.Trace)
    (rd : Strategy (audit L Reg session) → (h w : Fin C.N) → BitVec (C.w w)) (D : Description) (a₀ : C.Trace)
    (Dl : Finset (Fin C.N)) (δ εks εbind εw δin : ℝ≥0∞) (H k : ℕ) : Prop :=
  Takes P L blk A tr rd D a₀ Dl δ εks εbind εw δin H k →
    ∀ (M : Type) [Fintype M] [Nonempty M] (σ : M → Strategy (audit L Reg session))
      (rcv : ((t : Dl) → BitVec (C.w t.1)) → M),
      avg (fun m => prob (fun o => o.1 = true ∧ rcv (fun t => tr (A.committedOf (σ m)) t.1) = m)
          (audit L Reg session) (σ m)) ≤
        ((2 ^ H * ∑ j ∈ range (k + 1), n.choose j : ℕ) : ℝ≥0∞) / (Fintype.card M : ℝ≥0∞) +
          (δ + εks + εbind + εw + δin)

/-- **Infiltration** (influence): a set `Y` of delivered values, fixed before any prover by the circuit, the partition,
the anchors, the design and `δ`, with at most `2^H · ∑_{j ≤ k} C(n, j)` members, such that the verifier accepts while
the delivered values lie outside `Y` with probability at most the errors. The incorrect proof blocks steer the delivered
values by at most `H + log₂ ∑_{j ≤ k} C(n, j)` bits; `consumers.infiltration_bound` computes `H`.

It would follow from `Spot.Sampling.Sound` as `Spot.Sampling.Partition.audit_influence` follows from
`audit_profile`, with `card_influenceSet_le_harm` and `card_admissible_le` at `hb := blockWidth`, restated over core's
circuit. Not yet true: the wires premise stays a hypothesis until a wire has one commitment. -/
def InfiltrationLimit (P : Partition C n) (L : SamplingDesign U) (blk : Fin n ↪ U)
    (A : ExtractionAnalysis L Reg session) (tr : A.Tr → C.Trace)
    (rd : Strategy (audit L Reg session) → (h w : Fin C.N) → BitVec (C.w w)) (D : Description) (a₀ : C.Trace)
    (Dl : Finset (Fin C.N)) (δ εks εbind εw δin : ℝ≥0∞) (H k : ℕ) : Prop :=
  Takes P L blk A tr rd D a₀ Dl δ εks εbind εw δin H k →
    ∃ Y : Finset ((t : Dl) → BitVec (C.w t.1)), Y.card ≤ 2 ^ H * ∑ j ∈ range (k + 1), n.choose j ∧
      ∀ σ : Strategy (audit L Reg session), prob (fun o => o.1 = true ∧
        (fun t : Dl => tr (A.committedOf σ) t.1) ∉ Y) (audit L Reg session) σ ≤ δ + εks + εbind + εw + δin

/-- **Inference-only**, at the wired circuit: when the inference partition `I` of the registered circuit is its
description's, and no Definition reads more than `k` fresh bits, every inference block reads at most `k` fresh bits
(a wire's bits are not fresh). And the verifier accepts while the committed trace's incorrect proof blocks form a set
the sample misses with probability below `δ`, or a wire carries two values, with probability at most
`δ + εks + εbind + εw`: an accepted run is the registered circuit's, outside a set of proof blocks that escapes.

`k` and the fresh bits are computed from the description, one term per node, and nothing reads a record's side text.
The first part would follow from `DescribesFresh`, the second from `Spot.Sampling.Sound` and from `OneValuePerWire`.
Not yet true: the wires premise stays a hypothesis until a wire has one commitment, and lean's reader of the wired
circuit is what will compute `DescribesFresh`'s description. -/
def InferenceOnlyLimit (d : RegisteredCircuit) {n m : ℕ} (P : Partition d.D n) (I : Partition d.D m)
    (L : SamplingDesign U) (blk : Fin n ↪ U) (A : ExtractionAnalysis L Reg session) (tr : A.Tr → d.D.Trace)
    (rd : Strategy (audit L Reg session) → (h w : Fin d.D.N) → BitVec (d.D.w w)) (DI : Description) (k : ℕ)
    (δ εks εbind εw : ℝ≥0∞) : Prop :=
  Spot.Sampling.Sound → A.KnowledgeSound (fun _ _ => εks) → A.BindSound (fun _ _ => εbind) →
    (∀ X, A.incorrect X = (P.incorrect (tr X)).map blk) → OneValuePerWire P A tr rd εw →
    DI.DescribesFresh d I → DI.maxFresh ≤ k →
    (∀ u : Fin m, d.D.width (Structure.externalIn d (I.gates u)) ≤ k) ∧
      ∀ σ : Strategy (audit L Reg session), prob (fun o => o.1 = true ∧
        (L.missProb ((P.incorrect (tr (A.committedOf σ))).map blk) < δ ∨
          ∃ h w, w ∈ d.D.reads h ∧ P.unit h ≠ P.unit w ∧ rd σ h w ≠ tr (A.committedOf σ) w))
        (audit L Reg session) σ ≤ δ + εks + εbind + εw

end Limits

end Spot.Limits
