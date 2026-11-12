module

public import Proofs.Flock.SpotFlock.WiredCode

/-!
# Round 2's check of the wires and the wiring the proof covers: statements

`wiredOk h` (`WiredCode`) is the hypothesis each statement about the connected code takes: five conditions on the
code's own wiring, decided on what the verifier holds. Round 2's check of the wires (`Spot.SecondStage.wiresOk`, the
check `second-wires` fails on) is what the verifier decides of a run. This file states what ties the two.
Statements only: definitions, no proofs.

* `wiresCheckedAt`: round 2's check of the wires at what the verifier holds of a run (`ownOf`).
* `SecondWiresOrder`: what the check accepts has each wire's writer's part before its reader's and no wire's
  writer's port named by a link (`writersFirst`, `noReturnedWired`): the check's own last two clauses, read at the
  auditor's copies.
* `SecondWiresCovered`: the tie. At a registration whose two-stage design evaluates, whose members fit the two-stage
  scope and whose every wired source bit has a gate to go to, what the check accepts is a wiring the proof covers
  (`wiredOk`).

The check of the wires does not decide `wiresMove h`, which stays a hypothesis of the tie. It places each reader's
units in its rows (`SecondStage.fits`) and holds the rows a replay block to its writer's units a replay block; it
does not hold a unit's row to be one of those rows (round 2 does that statement by statement, `SecondStage.wiredRow`),
it does not compare the bits the reader's port reads with the bits the writer's output row holds, and it finds a
template by its name, which nothing holds to be the name of one template.
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

/-- **What the check of the wires accepts is a wiring the proof covers** (`wiredOk`), at a registration whose
two-stage design evaluates, whose members fit the two-stage scope (`memberOk`) and whose every wired source bit has a
gate to go to (`wiresMove`). -/
def SecondWiresCovered : Prop :=
  ∀ (h : Held) (s : AuditStatement) (st : Roots) (rec draw : Lean.Json) (c₀ : FlockDeploy.Coin),
    (two h s).isSome → h.ms.all memberOk = true → wiresMove h = true → wiresCheckedAt h st rec draw c₀ = true →
      wiredOk h = true

end SpotFlock
