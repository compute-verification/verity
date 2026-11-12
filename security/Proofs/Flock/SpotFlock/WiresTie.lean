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
* `rowsUnder`, `namesOnce`, `wiresWide`: three conditions the check of the wires does not decide today, each on
  what the verifier holds and none on a unit of a batch.
* `SecondWiresMove`: at a registration whose two-stage design evaluates (`two`), what the check accepts under the
  three conditions has a gate for every wired source bit to go to (`wiresMove`). Proved in `WiresMove`.

The check of the wires does not decide `wiresMove h`, which stays a hypothesis of the tie until the check decides the
three conditions. It places each reader's units in its rows (`SecondStage.fits`) and holds the rows a replay block to
its writer's units a replay block. It does not hold a unit's row to be one of those rows: round 2 does that statement
by statement (`SecondStage.wiredRow`), and `rowsUnder` asks it of the rows themselves. It finds a template by its
name, which nothing in what the model holds makes the name of one template: `namesOnce`. It does not compare the bits
the reader's port reads with the bits the writer's output row holds: `wiresWide`, which reads the two templates'
circuit files, and `SecondStage.Own` holds none.
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

/-! ## Three conditions the check of the wires does not decide -/

/-- **Each run's last row is one of its replay block's rows**: at the rows of each port a wire names, each run's last
unit reads a row under `per` (`row + (count - 1)·step < per`), so every unit the run holds does. Reads each wire's
`template` and `port` and, of its rows, `per` and each run's `row`, `count` and `step`: one comparison a run, and a
run is one entry however many units it holds. -/
def rowsUnder (own : Spot.SecondStage.Own) : Bool :=
  (own.wires.getD #[]).all fun w => (Spot.SecondStage.rowsOf own w.template w.port).all fun r =>
    r.runs.all fun x => decide (x.row + (x.count - 1) * x.step < r.per)

/-- **Each name is one template's**: no two parts have one template name. Reads the parts' `template`: one
comparison a pair of templates. -/
def namesOnce (ps : Array Spot.SecondStage.Part) : Bool := decide (ps.toList.map (·.template)).Nodup

/-- **Circuit file `c` writes bit `β` of its output row named `port`**: some instance's returned output leaf has that
bit on a computed row of the unit (`Bits.outGate`), at that commitment position (`Bits.outRowBit`); the same in every
unit, so it is read at unit 0 of one. Reads the file's output rows and one unit's returned leaves. -/
def writesBit (c : _root_.Flock.Circuit) (port : String) (β : ℕ) : Bool :=
  (List.range c.upv).any fun k => (List.range (c.leavesOut.getD k #[]).size).any fun j =>
    (Spot.Bits.outGate c 1 0 k j (β % 16)).isSome &&
      decide (Spot.Bits.outRowBit c 0 k j (β % 16) = some (Spot.Bits.RowPos.mk port 0 β))

/-- **A reader's port is no wider than what its writer writes**: for each wire, every bit under the `bits` of each
port of the wire's name in the reader's circuit file is a bit the writer's circuit file writes on the wire's output
row (`writesBit`). `cs` names each template's circuit file. Reads, a wire, the reader's ports' `name` and `bits` and
the writer's output rows and one unit's returned leaves; no unit of a batch. -/
def wiresWide (cs : Array (String × _root_.Flock.Circuit)) (own : Spot.SecondStage.Own) : Bool :=
  (own.wires.getD #[]).all fun w =>
    match cs.find? (·.1 == w.template), cs.find? (·.1 == w.writer) with
    | some r, some x => r.2.ports.all fun p => p.name != w.port ||
        (List.range p.bits).all fun β => writesBit x.2 w.wport β
    | _, _ => false

/-- **Each template's circuit file by its name**, as the verifier holds them (`Held.names`, `Held.ms`). -/
def circuitsOf (h : Held) : Array (String × _root_.Flock.Circuit) :=
  ((List.range h.ms.length).map fun t => (h.names.getD t "", (h.ms.getD t default).1)).toArray

/-- **What the check of the wires accepts, under the three conditions, moves every wired source bit**: at a
registration whose two-stage design evaluates (`two`: every proof block is in a replay block, and the replay layout
gives each replay block's units room in the members' files), where round 2's check of the wires passes, each run's
last row is one of its replay block's rows (`rowsUnder`), each name is one template's (`namesOnce`) and each
reader's port is no wider than what its writer writes (`wiresWide`), every source bit at a port a wire names and no
link names has a gate to go to (`wiresMove`). -/
def SecondWiresMove : Prop :=
  ∀ (h : Held) (s : AuditStatement) (st : Roots) (rec draw : Lean.Json) (c₀ : FlockDeploy.Coin),
    (two h s).isSome → wiresCheckedAt h st rec draw c₀ = true →
    rowsUnder (ownOf h st rec draw c₀) = true → namesOnce (partsAt h (h.reg.sample (src c₀)).size) = true →
    wiresWide (circuitsOf h) (ownOf h st rec draw c₀) = true →
      wiresMove h = true

end SpotFlock
