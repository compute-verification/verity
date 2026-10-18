import Definitions.Accounting.PouwWork
import Definitions.Accounting.Requirement.Placed

/-!
# PoUW's credits and help over a declared circuit's placed execution (trusted)

PoUW's work statement (`Models/Accounting/Guarantees/PouwWork.lean`) reads a run's credits and help
(`Accounting.PouwWork.credits`, `help`). On the actual run of a declared circuit `W`'s placed execution
(`Placement.actual`, `Definitions/Accounting/Requirement/Placed.lean`) they are `W`'s: the credits are the gates `W`'s
execution runs on the node in the window, priced at the credit pricing (a gate's credit, the target multiply-adds its
class stands for), and the help is the bits those gates and the verifier's use of values placed on neither. Their proofs
are in `Security/Proofs/Accounting/RequirementPouw.lean`.

* `RequirementCredits`: the node's credits at a coin are the work of `W`'s region `W[X, [ts, td))` outside the marking.
* `RequirementHelp`: the help the node takes in at a coin is `W`'s ingress `in_{X ∪ V}([ts, td))`, for a topological
  wiring the execution follows.
* `RequirementSteps`: when every device of `X` paces its engine steps and begins steps `a` and `a + len` at the window's
  ends, `W[X, [ts, td))` is `W`'s gates on `X` at the engine steps `[a, a + len)`, so the credits are the work of whole
  steps of `W`'s schedule.

Frozen (no-function): identities over a wiring, its placed execution and PoUW's vocabulary, about no function of the
code.
-/

namespace Accounting.Guarantees

open Accounting.Requirement

/-- **PoUW's credits are `W`'s gates.** On the actual runs of `W`'s placed executions `p r`, the node `X`'s credits at
coin `r` are the price at `π r` of the gates `W`'s execution runs on `X` at the window's ticks, the marking `m r` left
out:

  `credits X π m r = π r (W[X, [ts, td)) \ m r)`.

**The reading.** A credit pricing of `W`'s gates (each gate's target multiply-adds, from its class) prices the actual
run's gates the same: PoUW's work statement, read over this model, is about the multiply-adds of `W`'s own gates that
the node ran in the window.

Proof: `Accounting.SecurityProofs.RequirementCredits`: the actual run's computed gates are the values the execution
produces but the registered ones, each at its device and its step's tick.

Frozen (no-function): an identity over a wiring, its placed execution and PoUW's vocabulary, about no function of the
code. -/
def RequirementCredits : Prop :=
  ∀ (Ω L : Type) (n : ℕ) (W : Wiring (Fin n)) (p : Ω → Placement L (Fin n)) (P : PouwWork.Verdict Ω L)
    (X : Finset L) (π : Ω → Fin n → ℕ) (m : Ω → Finset (Fin n)) (r : Ω),
    PouwWork.credits (fun r => (p r).actual W) P X π m r = work (π r) (region W (p r) X (Set.Ico P.ts P.td) \ m r)

open Classical in
/-- **PoUW's help is `W`'s ingress.** For a topological wiring that the execution `p r` follows, the help the node `X`
takes in at coin `r` is the bits of the values placed outside `X` and the verifier's leaves `V` that `W`'s gates on `X`
or `V` use in the window:

  `help X r = in_{X ∪ V}([ts, td))` of `W`'s placed execution.

**The reading.** A wire of the actual run is an operand a gate uses (a gather's index and the word it selects, not its
column), from the device that produced or stores it: PoUW's help bound (A5) on this model is a bound on the bits `W`'s
schedule moves onto the node and the verifier from other devices in the window.

Proof: `Accounting.SecurityProofs.RequirementHelp`: in a topological wiring every operand a gate uses comes before it,
so the actual run's wires into a region are the operands its gates use.

Frozen (no-function): an identity over a wiring, its placed execution and PoUW's vocabulary, about no function of the
code. -/
def RequirementHelp : Prop :=
  ∀ (Ω L : Type) (n : ℕ) (W : Wiring (Fin n)) (p : Ω → Placement L (Fin n)) (P : PouwWork.Verdict Ω L)
    (X : Finset L) (r : Ω), W.Topological → (p r).Follows W →
    PouwWork.help (fun r => (p r).actual W) P X r = ingress W (p r) (X ∪ P.V) (Set.Ico P.ts P.td)

/-- **A window of whole steps.** If every device of `X` begins its engine steps at strictly increasing ticks, and begins
steps `w.start` and `w.start + w.len` at ticks `ts` and `td`, then `W`'s region over the ticks `[ts, td)` is its gates
on `X` at the engine steps of `w`:

  `W[X, [ts, td)) = W[X, steps w]`.

**The reading.** With `RequirementCredits`, the credits of a window whose ends begin steps on the node's devices are
the priced gates of whole steps of `W`'s schedule: what the census counts per step.

Proof: `Accounting.SecurityProofs.RequirementSteps`: a strictly increasing map orders the steps as their ticks.

Frozen (no-function): an identity over a placed execution, about no function of the code. -/
def RequirementSteps : Prop :=
  ∀ {G : Type} [DecidableEq G] {L : Type} (W : Wiring G) (p : Placement L G) (X : Finset L) (w : Window) (ts td : ℕ),
    p.Paced → (∀ x ∈ X, p.ticks x w.start = ts ∧ p.ticks x (w.start + w.len) = td) →
    region W p X (Set.Ico ts td) = stepRegion W p X w

end Accounting.Guarantees
