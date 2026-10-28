import Definitions.Accounting.PouwWork
import Definitions.Accounting.Requirement.Placed

/-!
# PoUW's credits and help when a node runs a declared circuit as placed (trusted)

PoUW's work statement (`Definitions/Accounting/Guarantees/PouwWork.lean`) reads a run's credits and help
(`Accounting.PouwWork.credits`, `help`). On the actual run of a declared circuit `W`'s placed execution
(`Placement.actual`, `Definitions/Accounting/Requirement/Placed.lean`), the run of a node that runs `W` as placed, they
are `W`'s: the credits are the gates `W`'s execution runs on the node in the window, the run's inputs aside, priced at
the credit pricing (a gate's credit, the target multiply-adds its class stands for), and the help is at most the bits
those gates and the verifier's use of values placed on neither. These are completeness: `Placement.actual` takes the
placement's leaf and ticks as given, its operations are constants, it has no outputs and the coins aren't its gates, so
they say what a node that runs `W` as placed is credited and helped, not what a node that passes PoUW ran. Their proofs
are in `Security/Proofs/Accounting/RequirementPouw.lean`.

* `RequirementCredits`: the node's credits at a coin are the work of `W`'s region `W[X, [ts, td))` outside the marking.
* `RequirementHelp`: the help the node takes in at a coin is `W`'s ingress `in_{X ∪ V}([ts, td))`, for a topological
  wiring the execution follows.
* `RequirementHelpBound`: the help is at most that ingress, with no hypothesis.
* `RequirementStepsInner`, `RequirementStepsOuter`: when every device of `X` paces its engine steps, `W[X, [ts, td))`
  holds `W`'s gates on `X` at the steps of `w` if each device begins `w` and its end inside `[ts, td)`, and is among them
  if each begins `w` at or before `ts` and its end at or after `td`.
* `RequirementCreditsSteps`: the credits are at least the work of the whole steps inside the window.

Completeness: statements about the run of a node that runs `W` as placed, about no function of the code.
-/

namespace Accounting.Guarantees

open Accounting.Requirement

/-- **PoUW's credits are `W`'s gates, when the node runs `W` as placed.** On the actual runs of `W`'s placed executions
`p r`, the node `X`'s credits at coin `r` are the price at `π r` of the gates `W`'s execution runs on `X` at the window's
ticks, the run's inputs and the marking `m r` left out:

  `credits X π m r = π r (W[X, [ts, td)) \ m r)`.

**The reading.** When the node runs `W` as placed, a credit pricing of `W`'s gates (each gate's target multiply-adds,
from its class) prices the actual run's gates the same: PoUW credits the multiply-adds of `W`'s own gates that the node
ran in the window, and nothing for a run input placed on it. It says what an honest node is credited, not what a node
that passes PoUW ran.

Proof: `Accounting.SecurityProofs.RequirementCredits`: the actual run's computed gates are the values the execution
produces but the registered ones and the run's inputs, each at its device and its step's tick.

Completeness: an identity over a wiring, its placed execution and PoUW's vocabulary, about no function of the code. -/
def RequirementCredits : Prop :=
  ∀ (Ω L : Type) (n : ℕ) (W : Wiring (Fin n)) (p : Ω → Placement L (Fin n)) (P : PouwWork.Verifier Ω L)
    (X : Finset L) (π : Ω → Fin n → ℕ) (m : Ω → Finset (Fin n)) (r : Ω),
    PouwWork.credits (fun r => (p r).actual W) P X π m r = work (π r) (region W (p r) X (Set.Ico P.ts P.td) \ m r)

open Classical in
/-- **PoUW's help is `W`'s ingress, when the node runs `W` as placed.** For a topological wiring that the execution
`p r` follows, the help the node `X` takes in at coin `r` is the bits of the values placed outside `X` and the
verifier's leaves `V` that `W`'s gates on `X` or `V` use in the window:

  `help X r = in_{X ∪ V}([ts, td))` of `W`'s placed execution.

**The reading.** When the node runs `W` as placed, a wire of the actual run is an operand a gate uses (a gather's index
and the word it selects, not its column), from the device that produced or stores it: PoUW's help (A5) is the bits
`W`'s schedule moves onto the node and the verifier from other devices in the window. Without the hypotheses the help
is at most this ingress (`RequirementHelpBound`).

Proof: `Accounting.SecurityProofs.RequirementHelp`: in a topological wiring every operand a gate uses comes before it,
so the actual run's wires into a region are the operands its gates use.

Completeness: an identity over a wiring, its placed execution and PoUW's vocabulary, about no function of the code. -/
def RequirementHelp : Prop :=
  ∀ (Ω L : Type) (n : ℕ) (W : Wiring (Fin n)) (p : Ω → Placement L (Fin n)) (P : PouwWork.Verifier Ω L)
    (X : Finset L) (r : Ω), W.Topological → (p r).Follows W →
    PouwWork.help (fun r => (p r).actual W) P X r = ingress W (p r) (X ∪ P.V) (Set.Ico P.ts P.td)

open Classical in
/-- **PoUW's help is at most `W`'s ingress, when the node runs `W` as placed.** With no hypothesis on the wiring or the
execution, the help the node `X` takes in at coin `r` is at most `W`'s ingress over `X ∪ V` in the window:

  `help X r ≤ in_{X ∪ V}([ts, td))`.

**The reading.** When the node runs `W` as placed, every wire of the actual run into its region is an operand a gate of
the region uses (the actual run reads only the earlier ones), so what crosses is at most `W`'s ingress, for any wiring.

Proof: `Accounting.SecurityProofs.RequirementHelpBound`: the actual run's wires into a region are among the operands
its gates use.

Completeness: a bound over a wiring, its placed execution and PoUW's vocabulary, about no function of the code. -/
def RequirementHelpBound : Prop :=
  ∀ (Ω L : Type) (n : ℕ) (W : Wiring (Fin n)) (p : Ω → Placement L (Fin n)) (P : PouwWork.Verifier Ω L)
    (X : Finset L) (r : Ω),
    PouwWork.help (fun r => (p r).actual W) P X r ≤ ingress W (p r) (X ∪ P.V) (Set.Ico P.ts P.td)

/-- **Whole steps inside a window.** If each device of `X` begins its engine steps at strictly increasing ticks, step
`w.start` at or after `ts` and step `w.start + w.len` at or before `td`, `W`'s gates on `X` at the steps of `w` are in its
region over the ticks `[ts, td)`:

  `W[X, steps w] ⊆ W[X, [ts, td))`.

**The reading.** When the node runs `W` as placed, a window holds every whole step that begins and ends inside it on
each of the node's devices, whatever the other devices' ticks: with `RequirementCredits`, its credits are at least
those steps' priced gates (`RequirementCreditsSteps`), what the census counts per step.

Proof: `Accounting.SecurityProofs.RequirementStepsInner`: a strictly increasing map orders the steps as their ticks.

Completeness: an inclusion over a placed execution, about no function of the code. -/
def RequirementStepsInner : Prop :=
  ∀ {G : Type} [DecidableEq G] {L : Type} (W : Wiring G) (p : Placement L G) (X : Finset L) (w : Window) (ts td : ℕ),
    p.Paced X → (∀ x ∈ X, ts ≤ p.ticks x w.start ∧ p.ticks x (w.start + w.len) ≤ td) →
    stepRegion W p X w ⊆ region W p X (Set.Ico ts td)

/-- **A window within whole steps.** If each device of `X` begins its engine steps at strictly increasing ticks, step
`w.start` at or before `ts` and step `w.start + w.len` at or after `td`, `W`'s region over the ticks `[ts, td)` is among
its gates on `X` at the steps of `w`:

  `W[X, [ts, td)) ⊆ W[X, steps w]`.

**The reading.** When the node runs `W` as placed, a window's credits are at most the priced gates of the whole steps
that cover it on each of the node's devices. With `RequirementStepsInner`, a window whose ends begin steps on each of
them is those steps exactly.

Proof: `Accounting.SecurityProofs.RequirementStepsOuter`, as `RequirementStepsInner`.

Completeness: an inclusion over a placed execution, about no function of the code. -/
def RequirementStepsOuter : Prop :=
  ∀ {G : Type} [DecidableEq G] {L : Type} (W : Wiring G) (p : Placement L G) (X : Finset L) (w : Window) (ts td : ℕ),
    p.Paced X → (∀ x ∈ X, p.ticks x w.start ≤ ts ∧ td ≤ p.ticks x (w.start + w.len)) →
    region W p X (Set.Ico ts td) ⊆ stepRegion W p X w

/-- **Credits cover the whole steps inside the window, when the node runs `W` as placed.** If each device of `X` paces
its engine steps and begins steps `w.start` and `w.start + w.len` inside the window's ticks at coin `r`, the node's
credits are at least the work of `W`'s gates on `X` at the steps of `w`, the marking left out:

  `π r (W[X, steps w] \ m r) ≤ credits X π m r`.

**The reading.** PoUW's credits, on the run of a node that runs `W` as placed, are at least the priced gates of every
whole step of `W`'s schedule inside the window.

Proof: `Accounting.SecurityProofs.RequirementCreditsSteps`, from `RequirementCredits` and `RequirementStepsInner`.

Completeness: a bound over a wiring, its placed execution and PoUW's vocabulary, about no function of the code. -/
def RequirementCreditsSteps : Prop :=
  ∀ (Ω L : Type) (n : ℕ) (W : Wiring (Fin n)) (p : Ω → Placement L (Fin n)) (P : PouwWork.Verifier Ω L)
    (X : Finset L) (π : Ω → Fin n → ℕ) (m : Ω → Finset (Fin n)) (r : Ω) (w : Window),
    (p r).Paced X → (∀ x ∈ X, P.ts ≤ (p r).ticks x w.start ∧ (p r).ticks x (w.start + w.len) ≤ P.td) →
    work (π r) (stepRegion W (p r) X w \ m r) ≤ PouwWork.credits (fun r => (p r).actual W) P X π m r

end Accounting.Guarantees
