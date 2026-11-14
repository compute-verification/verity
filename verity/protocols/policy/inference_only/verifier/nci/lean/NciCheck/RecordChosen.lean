module

public import NciCheck.Record

/-!
# The bits the prover chose, over an execution phase's circuit record

`Record.summarize` counts a block: of each kind of input bits, the most one slot holds. This module counts the
**whole execution phase**: of each kind, the bits of every unit of every slot (`executionPhaseBits`), and of them
the bits the prover chose (`chosen`). It has a difftest (`RecordDifftest`), and the rows protocol (`RowsProtocol`)
reports `chosen` in its profile, which no verdict reads.

## What it counts

The execution phase's input bits are every input row of every unit of the record, whole, 16 bits a word, whether or
not a gate reads it (`executionPhaseInputs`): `Record.summarize`'s `inputs`, over every slot. Each row has one kind
(`Record.Case.rowKind`), and a kind's bits are the bits of its rows. A declaration fills its rows with one thing, so
all of them have one kind (`Record.Case.kind`), and a kind's bits are a sum over the declarations: each one's reader
units, times the rows it covers of each, times a row's bits. That is `Record.tally`'s term with every reader unit
counted, whatever its slot. The kind `free` is the rest, the rows no declaration fills.

## The bits the prover chose

`chosen` is the bits of three kinds:

* `free`: no declaration fills the row;
* `unregistered`: a registered read, or a table read, fills it from a parameter that is not θ;
* `openings`: it is a port of an opening that is not held.

Nothing the record declares ties such a row to a value fixed before the phase, computed in it or received in it. The
other four kinds are not the prover's alone: `inside` and `crossWires` are outputs of the phase's own units, `theta`
is registered at the start of the deployment, and `ingress` is the phase's ingress as the record declares it.

The rule for an opening is the rate rule's own, `Record.openingHeld`: an opening's ports are off this count only
where the opening is held, and every port of any other opening is on it. So a reader that is not pinned has every
opened port on the count, whatever table the record declares it to open.

This is the number the bound on what leaves a monitoring period reads of an execution phase as its free inputs
that are not registered (`NetTiming.freeInputs`, less the registered ones: a step's ingress is no free input there).
The check of a monitoring period's network records takes that number from its parameters, an audit's `free`, and
not from this module.

It is a count of what the record declares, as `Record.summarize` is: it holds no row to a value, and what an
opening that is held rests on is `Record`'s to say ("An opening, held").

## Refusals

`Record`'s own, in its order, and none of its own. The count reads no block, but `Record.read` decides which
parameter is θ and which opening is held after it has read the record's inference entries, so a record that states
none is refused (`blocks`) and not counted.

## The cost

One step a template and one a declaration: no unit, slot, row or gate is visited, whatever the units number.
-/

@[expose] public section

namespace NciCheck.RecordChosen

open Lean (Json toJson)
open Record (Kind Case)

/-- **The execution phase's input bits**: every input row of every unit of the record, whole, 16 bits a word. -/
def executionPhaseInputs (c : Case) : Nat :=
  (c.f.ts.map fun t => t.units * (t.rows.map fun (_, words, _) => 16 * words).sum).sum

/-- Per kind (`Kind.index`), the bits the declarations fill over the whole execution phase: each one's reader units,
times the rows it covers of each, times a row's bits, under the kind of what fills them (`Case.kind`). No declaration
has the kind `free`, whose entry is 0. -/
def executionPhaseDeclared (c : Case) : Array Nat :=
  c.x.decls.foldl (fun acc d =>
    acc.modify (c.kind (some d.fill)).index (· + d.w.readerUnits * d.w.cuts * d.bits))
    (Array.replicate 7 0)

/-- **The execution phase's input bits by kind** (`Kind.index`): of each declared kind the bits its declarations
fill (`executionPhaseDeclared`), and of `free` the rest of the phase's input bits, which no declaration fills. -/
def executionPhaseBits (c : Case) : Array Nat :=
  let declared := executionPhaseDeclared c
  declared.modify Kind.free.index fun _ => executionPhaseInputs c - declared.foldl (· + ·) 0

/-- **The bits the prover chose**, of the bits by kind `bits`: the kinds `free`, `unregistered` and `openings`. -/
def chosen (bits : Array Nat) : Nat :=
  bits[Kind.free.index]! + bits[Kind.unregistered.index]! + bits[Kind.openings.index]!

/-- The kinds, in `Kind.index`'s order. -/
def KINDS : List Kind := [.inside, .theta, .crossWires, .free, .ingress, .unregistered, .openings]

/-- The count on one instance: `{inputs, inside, theta, cross_wires, free, ingress, unregistered, openings, chosen}`,
the execution phase's input bits, those of each kind and the ones the prover chose; or the first refusal's code,
`Record`'s in its order. -/
def evaluate (j tr : Json) : Except String Json := do
  let c ← Record.read j tr
  let bits := executionPhaseBits c
  return Json.mkObj (("inputs", toJson (executionPhaseInputs c)) ::
    KINDS.map (fun k => (k.name, toJson bits[k.index]!)) ++ [("chosen", toJson (chosen bits))])

/-- `evaluate`, with a refusal as `refusal` writes it: the whole count, for the private record only. -/
def checkDetail (j tr : Json) : Json :=
  match evaluate j tr with
  | .ok r => r
  | .error code => refusal code

end NciCheck.RecordChosen
