module

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
* `audit`, `committedOf`, `Analyzed`: the audit `Spot.Sampling.Sound` speaks of, and its hypotheses for this circuit.
* `OneValuePerWire`: the wires premise.
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

variable {C : Circuit} {n : ℕ} (P : Partition C n) (L : SamplingDesign (Fin n)) {Reg : Type}
  (session : Finset (Fin n) → Reg → Game Bool)

/-- **The audit** that `Spot.Sampling.Sound` speaks of: the registration, the sample's coin and the session on the
sample, with the sample beside the session's verdict. -/
abbrev audit : Game (Bool × Finset (Fin n)) :=
  Game.send Reg fun R => Game.coin L.Ω fun ω => (session (L.draw ω) R).map fun ok => (ok, L.draw ω)

/-- The committed trace of an audit strategy `σ`: `committed` at its registration and its sessions' strategies, as
`Spot.Sampling.Sound` reads it. -/
def committedOf (committed : (R : Reg) → ((ω : L.Ω) → Strategy (session (L.draw ω) R)) → C.Trace)
    (σ : Strategy (audit L session)) : C.Trace :=
  committed σ.1 fun ω => Strategy.ofMap (fun ok => (ok, L.draw ω)) (session (L.draw ω) σ.1) (σ.2 ω)

/-- **The session's analysis for this circuit**: the hypotheses `Spot.Sampling.Sound` takes, at the transcripts
`C.Trace` and the incorrect proof blocks `P.incorrect`, with the knowledge error `εks` and the binding error `εbind`
constant. `ks` and `link` cover acceptance at every sampled incorrect proof block, and on average over the sample they
are at most `εks` and `εbind` at any unit picked before the session's first coin. -/
def Analyzed (committed : (R : Reg) → ((ω : L.Ω) → Strategy (session (L.draw ω) R)) → C.Trace)
    (ks : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → ℝ≥0∞)
    (link : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → C.Trace → ℝ≥0∞)
    (εks εbind : ℝ≥0∞) : Prop :=
  (∀ S R (τ : Strategy (session S R)) u X, u ∈ S → u ∈ P.incorrect X →
      prob (· = true) (session S R) τ ≤ ks S R τ u + link S R τ u X) ∧
    (∀ R (τ : (ω : L.Ω) → Strategy (session (L.draw ω) R)) (tgt : L.Ω → Option (Fin n)),
      avg (fun ω => (tgt ω).elim 0 fun u => if u ∈ L.draw ω then ks (L.draw ω) R (τ ω) u else 0) ≤ εks) ∧
    (∀ R (τ : (ω : L.Ω) → Strategy (session (L.draw ω) R)) (tgt : L.Ω → Option (Fin n)),
      avg (fun ω => (tgt ω).elim 0 fun u =>
        if u ∈ L.draw ω then link (L.draw ω) R (τ ω) u (committed R τ) else 0) ≤ εbind)

/-- **One value on each wire** (the wires premise): whatever the prover does, the verifier accepts while some block's
opening `rd σ h w` of a wire `w` it reads from another block differs from the writer's committed value, with
probability at most `εw`.

Not yet true by construction: the deployed path checks each wire against a link and takes this as an open premise. Once
a wire has one commitment, the writer's row that the reader refers to, it holds with `εw = 0`, and the hypothesis
goes. -/
def OneValuePerWire (committed : (R : Reg) → ((ω : L.Ω) → Strategy (session (L.draw ω) R)) → C.Trace)
    (rd : Strategy (audit L session) → (h w : Fin C.N) → BitVec (C.w w)) (εw : ℝ≥0∞) : Prop :=
  ∀ σ : Strategy (audit L session), prob (fun o => o.1 = true ∧ ∃ h w, w ∈ C.reads h ∧ P.unit h ≠ P.unit w ∧
    rd σ h w ≠ committedOf L session committed σ w) (audit L session) σ ≤ εw

/-- What both limits on the delivered wires `Dl` take: the sampling layer's theorem (`Spot.Sampling.Sound`) and this
circuit's analysis for it; one value on each wire, up to `εw`; the committed inputs are the anchors `a₀`, up to `δin`;
the description describes `P`; `Dl` is committed; `H` bounds the computed widths of every set that escapes with
probability at least `δ` (the profile's harm bound); and `k + 1` units are caught except with probability below `δ`. -/
def Takes (committed : (R : Reg) → ((ω : L.Ω) → Strategy (session (L.draw ω) R)) → C.Trace)
    (ks : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → ℝ≥0∞)
    (link : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → C.Trace → ℝ≥0∞)
    (rd : Strategy (audit L session) → (h w : Fin C.N) → BitVec (C.w w)) (D : Description) (a₀ : C.Trace)
    (Dl : Finset (Fin C.N)) (δ εks εbind εw δin : ℝ≥0∞) (H k : ℕ) : Prop :=
  Spot.Sampling.Sound ∧ Analyzed P L session committed ks link εks εbind ∧
    OneValuePerWire P L session committed rd εw ∧
    (∀ σ : Strategy (audit L session), prob (fun o => o.1 = true ∧
      ¬ C.InputsAgree (committedOf L session committed σ) a₀) (audit L session) σ ≤ δin) ∧
    D.Describes P ∧ Dl ⊆ P.committed ∧
    (∀ B : Finset (Fin n), δ ≤ L.missProb B → ∑ u ∈ B, D.blockWidth u ≤ H) ∧ L.miss (k + 1) < δ

/-- **Exfiltration**: a prover that picks its audit strategy `σ m` from a uniform message `m`, and a receiver `rcv`
that sees only the delivered values, succeed together (accepted, and `rcv` returns `m`) with probability at most
`2^H · ∑_{j ≤ k} C(n, j) / |M|` plus the errors. With `|M| = 2^ℓ`, at most `H + log₂ ∑_{j ≤ k} C(n, j)` bits leave.
`H` and the computed widths come from the description; `consumers.exfiltration_bound` computes this figure.

It would follow from `Spot.Sampling.Sound` at the event "the incorrect set is missed with probability below `δ`", as
`Spot.Sampling.Partition.audit_exfiltration` follows from `audit_profile`, with `hb := blockWidth`, `hw` by
`WidthsMeet` and `hsep` by `separates_exits`, restated over core's circuit. Not yet true: the wires premise
(`OneValuePerWire`) stays a hypothesis until a wire has one commitment. -/
def ExfiltrationLimit (committed : (R : Reg) → ((ω : L.Ω) → Strategy (session (L.draw ω) R)) → C.Trace)
    (ks : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → ℝ≥0∞)
    (link : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → C.Trace → ℝ≥0∞)
    (rd : Strategy (audit L session) → (h w : Fin C.N) → BitVec (C.w w)) (D : Description) (a₀ : C.Trace)
    (Dl : Finset (Fin C.N)) (δ εks εbind εw δin : ℝ≥0∞) (H k : ℕ) : Prop :=
  Takes P L session committed ks link rd D a₀ Dl δ εks εbind εw δin H k →
    ∀ (M : Type) [Fintype M] [Nonempty M] (σ : M → Strategy (audit L session))
      (rcv : ((t : Dl) → BitVec (C.w t.1)) → M),
      avg (fun m => prob (fun o => o.1 = true ∧
          rcv (fun t => committedOf L session committed (σ m) t.1) = m) (audit L session) (σ m)) ≤
        ((2 ^ H * ∑ j ∈ range (k + 1), n.choose j : ℕ) : ℝ≥0∞) / (Fintype.card M : ℝ≥0∞) +
          (δ + εks + εbind + εw + δin)

/-- **Infiltration** (influence): a set `Y` of delivered values, fixed before any prover by the circuit, the partition,
the anchors, the design and `δ`, with at most `2^H · ∑_{j ≤ k} C(n, j)` members, such that the verifier accepts while
the delivered values lie outside `Y` with probability at most the errors. The incorrect proof blocks steer the delivered
values by at most `H + log₂ ∑_{j ≤ k} C(n, j)` bits; `consumers.infiltration_bound` computes `H`.

It would follow from `Spot.Sampling.Sound` as `Spot.Sampling.Partition.audit_influence` follows from
`audit_profile`, with `card_influenceSet_le_harm` and `card_admissible_le` at `hb := blockWidth`, restated over core's
circuit. Not yet true: the wires premise stays a hypothesis until a wire has one commitment. -/
def InfiltrationLimit (committed : (R : Reg) → ((ω : L.Ω) → Strategy (session (L.draw ω) R)) → C.Trace)
    (ks : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → ℝ≥0∞)
    (link : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → C.Trace → ℝ≥0∞)
    (rd : Strategy (audit L session) → (h w : Fin C.N) → BitVec (C.w w)) (D : Description) (a₀ : C.Trace)
    (Dl : Finset (Fin C.N)) (δ εks εbind εw δin : ℝ≥0∞) (H k : ℕ) : Prop :=
  Takes P L session committed ks link rd D a₀ Dl δ εks εbind εw δin H k →
    ∃ Y : Finset ((t : Dl) → BitVec (C.w t.1)), Y.card ≤ 2 ^ H * ∑ j ∈ range (k + 1), n.choose j ∧
      ∀ σ : Strategy (audit L session), prob (fun o => o.1 = true ∧
        (fun t : Dl => committedOf L session committed σ t.1) ∉ Y) (audit L session) σ ≤
          δ + εks + εbind + εw + δin

end Limits

/-- **Inference-only**, at the wired circuit: when the inference partition `I` of the registered circuit is its
description's, and no Definition reads more than `k` fresh bits, every inference block reads at most `k` fresh bits
(a wire's bits are not fresh). And the verifier accepts while the committed trace's incorrect proof blocks form a set
the sample misses with probability below `δ`, or a wire carries two values, with probability at most
`δ + εks + εbind + εw`: an accepted run is the registered circuit's, outside a set of proof blocks that escapes.

`k` and the fresh bits are computed from the description, one term per node, and nothing reads a record's side text.
The first part would follow from `DescribesFresh`, the second from `Spot.Sampling.Sound` with `Analyzed` and from
`OneValuePerWire`. Not yet true: the wires premise stays a hypothesis until a wire has one commitment, and lean's reader
of the wired circuit is what will compute `DescribesFresh`'s description. -/
def InferenceOnlyLimit (d : RegisteredCircuit) {n m : ℕ} (P : Partition d.D n) (I : Partition d.D m)
    (L : SamplingDesign (Fin n)) {Reg : Type} (session : Finset (Fin n) → Reg → Game Bool)
    (committed : (R : Reg) → ((ω : L.Ω) → Strategy (session (L.draw ω) R)) → d.D.Trace)
    (ks : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → ℝ≥0∞)
    (link : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → d.D.Trace → ℝ≥0∞)
    (rd : Strategy (audit L session) → (h w : Fin d.D.N) → BitVec (d.D.w w)) (DI : Description) (k : ℕ)
    (δ εks εbind εw : ℝ≥0∞) : Prop :=
  Spot.Sampling.Sound → Analyzed P L session committed ks link εks εbind →
    OneValuePerWire P L session committed rd εw → DI.DescribesFresh d I → DI.maxFresh ≤ k →
    (∀ u : Fin m, d.D.width (Structure.externalIn d (I.gates u)) ≤ k) ∧
      ∀ σ : Strategy (audit L session), prob (fun o => o.1 = true ∧
        (L.missProb (P.incorrect (committedOf L session committed σ)) < δ ∨
          ∃ h w, w ∈ d.D.reads h ∧ P.unit h ≠ P.unit w ∧ rd σ h w ≠ committedOf L session committed σ w))
        (audit L session) σ ≤ δ + εks + εbind + εw

end Spot.Limits
