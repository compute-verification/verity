import Definitions.Accounting.ServingRoute

/-!
# The serving route's premises: the bridge and coverage

Two named premises of `Accounting.Guarantees.ServingRoute.ServingEndToEnd`
(note:training-20261027T0134Z-draft-e2e-overhead-proof §9), each a bound over one experiment's game at every strategy,
on the play `play s o` (`Accounting.ServingRoute.Play`) read off the strategy `s` and the run `o`. Their errors may
depend on the strategy.

* `ServingBridge`: the subprotocols' claims put the play in the serving class (`Params.Class`), but with probability
  `δ_b`.
* `ServingCoverage`: the task model in materialization form. Succeeding at the task in the class takes at least `N`
  counted operations of the reference, but with probability `η`.
-/

namespace Accounting.Assumptions

open Verity Accounting.ServingRoute
open scoped ENNReal

/-- NEW PREMISE. **The serving bridge** (spine §9, premises 3 and 5): at every strategy `s`, the probability that every
claim holds (`Claims s o`) while the play is outside the serving class (`Params.Class`) is at most `δ_b s`. Its parts
and where each would come from:

* causality and the compute rate: the device model (Layer 1; LEAN-70's `FixedMachine` once it lands);
* the links as declared and each server's boundary capacity at most `F`: the network certifier's claim with the
  declared links, `Covered` and the trusted units (`trusted-nebius-host`); the spine's F12 keeps it a physics premise;
* windows of at most `ω` ticks with the state at each window's start at most `R` bits: the wipe. Its form here is
  structural (`Placed.state`, the width a server carries across its wipe tick); the wipe's guarantee bounds information
  (`Pous.Guarantees.StreamedWipeMeetsInfo`), and the structural width does not follow from it for every strategy (a
  server that passes a wipe and reads values it held through it carries them as state), so this part restricts the
  strategies until the informational route lands (MEM-98);
* the counted operations on the servers or at the dark leaves, and the dark inventory: the deployment's inventory
  (`ε_dark`, Daniel 26 Oct 6:55 PM PDT);
* the per-block cap `q` at `(c_S, k_S)`: `Accounting.Guarantees.ServingRoute.ServingPerBlockCap` gives it from the
  activation premise at serving scale, which is contested (a server-day block can replay training steps from data;
  note:training-20261024T2336Z-report-tvi-findings, E1's P0);
* the counting's structure (`OperationBodies`, `Overlap`, widths, weights): how the training class's counted operations
  are read off the actual circuit.

It fails when a claim leaves room the class doesn't: a link the certifier doesn't see, state a wipe leaves beyond `R`
bits, hardware outside the servers and the dark inventory, or blocks of the serving budget that compute more than `q`. -/
def ServingBridge {O L σ E : Type} [Fintype E] (G : Game O) (P : Params L σ E)
    (play : Game.Strategy G → O → Play L σ) (Claims : Game.Strategy G → O → Prop)
    (δb : Game.Strategy G → ℝ≥0∞) : Prop :=
  ∀ s, Game.prob (fun o => Claims s o ∧ ¬ P.Class (play s o)) G s ≤ δb s

/-- NEW PREMISE. **Serving coverage** (the task model, spine §9 premise 1, materialization form): at every strategy
`s`, the probability that the play is in the serving class and succeeds at the task (`Success s o`) with counted
weight below `N` (`(play s o).J.total < N`: fewer than `N` counted operations of the reference materialized in the
actual circuit) is at most `η s`. It fails when the task can be reached with less of the reference's computation than
`N`: a shortcut the counting doesn't see, or an `N` above what the task needs. The containment of the reference circuit
is folded in here (spine §8.1: a task premise, not a bridge). -/
def ServingCoverage {O L σ E : Type} [Fintype E] (G : Game O) (P : Params L σ E)
    (play : Game.Strategy G → O → Play L σ) (Success : Game.Strategy G → O → Prop) (N : ℝ)
    (η : Game.Strategy G → ℝ≥0∞) : Prop :=
  ∀ s, Game.prob (fun o => P.Class (play s o) ∧ Success s o ∧ (play s o).J.total < N) G s ≤ η s

end Accounting.Assumptions
