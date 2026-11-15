module

public import NciCheck.Count
public import NciCheck.Record

/-!
# The count of `(c, k)`-inference blocks on an execution phase's circuit record

`Count` counts `(c, k)`-inference blocks on a Program whose blocks are activations of block Definitions. On
an execution phase's circuit record the blocks are the slots of the record's class, and `Record.summarize`
takes one summary of them. `evaluate` is that decision on the summary: a block's Boolean gates at most `c`, its fresh
bits at most `k`, the blocks at most `n`. `c` is in Boolean gates, the gate table's unit: the summary's `gates`, which
the rate rule reads, are the Program's gates at their widths, another figure.

A block's Boolean gates (`boolGates`) are, over the record's templates, a template's units in a block times its
Definition's Boolean gates per activation by the gate table (`Count.boolGates`: per primitive, the Boolean
gates its lowering computes). They are the same in every block, as the summary's gates are. The fresh bits and the
blocks are the summary's (`Record.Summary.fresh`, `blocks`).

The gate table is read here and nowhere in `Record`: the rate rule (`Record.rate`) reads the
Program's own gate counts, and nothing it depends on imports `Count`.

## The limits

One JSON object beside `Record`'s instance and trusted input, `{"table", "c", "k", "n"}`: the gate table
`{key: g}` and the policy, as `Count`'s instance states them.

## Refusals

In the order checked: `instance` (the limits are no object, a field is missing, or `table` is no object), `params`
(`c`, `k`, `n` or a value of `table` is not a natural), then `Record`'s own in its order, then `gate-table`
(the table lacks a primitive a template computes).

## What is published

As `Count`: `check` gives `{complies}`, or the refusal. A block's Boolean gates and the rule that failed stay
in the private record (`checkDetail`).
-/

@[expose] public section

namespace NciCheck.RecordCount

open Lean (Json toJson)

/-- The limits: the gate table and the policy `(c, k, n)`. -/
structure Limits where
  /-- Per primitive's key, the Boolean gates its lowering computes. -/
  table : Array (String × Nat)
  /-- The most Boolean gates a block computes: the gate table's unit, not the Program's gates at their widths
  (`Record.Summary.gates`, which the rate rule's `Γ` bounds). -/
  c : Nat
  /-- The most fresh bits a block reads. -/
  k : Nat
  /-- The most blocks. -/
  n : Nat

/-- The gate table's entry for `key`. -/
def Limits.gatesOf (l : Limits) (key : String) : Option Nat := (l.table.find? (·.1 == key)).map (·.2)

/-- The limits, refused (`instance`) where a field is missing or `table` is no object, and (`params`) where `c`, `k`,
`n` or a value of `table` is not a natural. -/
def parse (j : Json) : Except String Limits := do
  let .obj table ← field j "table" | throw "instance"
  let cJ ← field j "c"
  let kJ ← field j "k"
  let nJ ← field j "n"
  let some c := natOf? cJ | throw "params"
  let some k := natOf? kJ | throw "params"
  let some n := natOf? nJ | throw "params"
  let table ← table.toArray.mapM fun (key, v) => match natOf? v with
    | some g => pure (key, g)
    | none => throw "params"
  return { table, c, k, n }

/-- A block's Boolean gates: over the summary's templates, the template's units in a block times its Definition's
Boolean gates per activation; none where the table lacks a primitive one of them computes. -/
def boolGates (P : Flock.Program.Prog) (l : Limits) (s : Record.Summary) : Option Nat :=
  let g := Count.boolGates P l.gatesOf
  s.units.foldl (fun acc (d, n) => acc.bind fun a => (g[d]?.join).map fun b => a + n * b) (some 0)

/-- The first rule a summary fails: `gates` where a block's Boolean gates exceed `c`, else `fresh` where its fresh bits
exceed `k`, else `count` where the blocks exceed `n`. -/
def failure (l : Limits) (bool : Nat) (s : Record.Summary) : Option String :=
  if bool > l.c then some "gates" else if s.fresh > l.k then some "fresh" else if s.blocks > l.n then some "count"
  else none

/-- The count on one instance: `{bool_gates, complies, failure}`, or the first refusal's code, in the docstring's
order. -/
def evaluate (j tr limits : Json) : Except String Json := do
  let l ← parse limits
  let c ← Record.read j tr
  let s := Record.summarize c
  let some bool := boolGates c.f.P l s | throw "gate-table"
  let f := failure l bool s
  return Json.mkObj [("bool_gates", toJson bool), ("complies", Json.bool f.isNone), ("failure", Record.failureJson f)]

/-- `evaluate`, with a refusal as `NciCheck.refusal` writes it: the whole verdict, for the private record only. -/
def checkDetail (j tr limits : Json) : Json :=
  match evaluate j tr limits with
  | .ok r => r
  | .error code => NciCheck.refusal code

/-- What the count would publish: the judgment against `n`, `{complies}` from `evaluate`'s verdict, or the refusal. -/
def check (j tr limits : Json) : Json :=
  match evaluate j tr limits with
  | .ok r => Json.mkObj [("complies", r.getObjValD "complies")]
  | .error code => NciCheck.refusal code

end NciCheck.RecordCount
