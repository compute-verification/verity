module

public import Proofs.Flock.SpotFlock.WiredCode

/-!
# Round 2's check of the wires and the wiring the proof covers: statements

`wiredOk h` (`WiredCode`) is the hypothesis each statement about the connected code takes: five conditions on the
code's own wiring, decided on what the verifier holds. Round 2's check of the wires (`Spot.SecondStage.wiresOk`, the
check `second-wires` fails on) is what the verifier decides of a run. This file states what would tie the two.
Statements only: definitions, no proofs.

* `wiresCheckedAt`: round 2's check of the wires at what the verifier holds of a run (`ownOf`).
* `SecondWiresOrder`: what the check accepts has each wire's writer's part before its reader's and no wire's
  writer's port named by a link (`writersFirst`, `noReturnedWired`): the check's own last two clauses.
* `SecondWiresMove`: at a registration whose two-stage design evaluates, what the check accepts gives every wired
  source bit a gate to go to (`wiresMove`).
* `SecondWiresCovered`: the tie. At a registration whose two-stage design evaluates and whose members fit the
  two-stage scope, what the check accepts is a wiring the proof covers (`wiredOk`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy Properties.Spot Finset

/-- **Round 2's check of the wires at what the verifier holds of a run**: `SecondStage.wiresOk` at the parts of the
run's replay sample and at `ownOf`, as `SecondStage.check` asks it before it reads a statement. It reads the parts
and the registration's links, wires and rows. -/
def wiresCheckedAt (h : Held) (st : Roots) (rec draw : Lean.Json) (c₀ : FlockDeploy.Coin) : Bool :=
  Spot.SecondStage.wiresOk (partsAt h (h.reg.sample (src c₀)).size) (ownOf h st rec draw c₀)

/-- **What the check of the wires accepts has its writers first and reads no returned output**: each wire's writer's
part is before its reader's (`writersFirst`) and no wire's writer's port is one a link names (`noReturnedWired`). -/
def SecondWiresOrder : Prop :=
  ∀ (h : Held) (st : Roots) (rec draw : Lean.Json) (c₀ : FlockDeploy.Coin),
    wiresCheckedAt h st rec draw c₀ = true → writersFirst h = true ∧ noReturnedWired h = true

/-- **What the check of the wires accepts gives every wired source bit a gate to go to** (`wiresMove`), at a
registration whose two-stage design evaluates. -/
def SecondWiresMove : Prop :=
  ∀ (h : Held) (s : AuditStatement) (st : Roots) (rec draw : Lean.Json) (c₀ : FlockDeploy.Coin),
    (two h s).isSome → wiresCheckedAt h st rec draw c₀ = true → wiresMove h = true

/-- **What the check of the wires accepts is a wiring the proof covers** (`wiredOk`), at a registration whose
two-stage design evaluates and whose members fit the two-stage scope (`memberOk`). -/
def SecondWiresCovered : Prop :=
  ∀ (h : Held) (s : AuditStatement) (st : Roots) (rec draw : Lean.Json) (c₀ : FlockDeploy.Coin),
    (two h s).isSome → h.ms.all memberOk = true → wiresCheckedAt h st rec draw c₀ = true → wiredOk h = true

end SpotFlock
