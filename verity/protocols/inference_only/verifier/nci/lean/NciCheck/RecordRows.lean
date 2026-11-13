module

public import NciCheck.Record

/-!
# A block's rows on an execution phase's circuit record

`Record.summarize` counts a block in the Program's gates at their widths, and `RecordCount` in Boolean gates by the
gate table. This module counts the same block in **rows**: the computed rows its units have in their templates'
circuit files. A template's circuit file lays one unit of the template out as `m` computed rows, the same `m` for
every unit of it (SPOT's `Spot.FilePart.FilePart.m`: the unit net's computed rows, stacked `units_per_vu` times), so
a block's rows are, over the record's templates, the template's units in a block times its `m` (`blockRows`). They
are the same in every block, as the summary's gates are. A row here is a computed row of a circuit file, never one
of a unit's input rows, which `Record` classes and counts in bits.

A row counts once here. The sampling claims count the same file at three gates a row: there a row is lowered to three
gates (`Bridge.progOf`), and a unit's proof block holds the `3m` gates of its rows
(`Spot.FilePart.FilePart.block`). That count is the claims' own, and nothing here reads it.

Each template's `m` comes in a table, by the template's id. This module reads no circuit file: whoever calls it
says where the table is from. No entry of the table is 0. A circuit file lays a unit out as at least one computed
row for each of its `units_per_vu`, which a file the parser takes has at least one of (`Flock.HmRow.check`), so a
table that gives a template no rows is not of circuit files, and a block would look smaller to the rule than it
is.

`blockRows` is the one definition of what a block counts in rows. It takes each template with its units in a block,
and each template's `m`, and reads nothing else, so a check that has no summary calls it on its own templates,
however it names them. `ofSummary` is it on a record's summary, whose templates are Definitions of the Program.

## The rule

The rate rule in rows (`rate`): `Γ` is the most rows a block holds and `α = num/den` the most fresh bits a row. It
is `failure`, which `Record.rate` decides the rule in gates by, on the one pair of a block's rows and its fresh bits
(`Record.Summary.fresh`): `size` where a block has more than `Γ` rows, else `nci` where its fresh bits exceed `α`
times its rows.

## The limits

One JSON object beside `Record`'s instance and trusted input, `{"table", "gamma", "alpha"}`: the rows table
`{template id: m}`, each `m` at least 1, and `Γ` and `α = [num, den]` in rows. The instance's own `gamma` and
`alpha` are not read here: they are `Record.rate`'s, in the Program's gates.

## Refusals

In the order checked: `instance` (the limits are no object, a field is missing, or `table` is no object), `params`
(`gamma`, `alpha` or a value of `table` is not as stated), then `Record`'s own in its order, then `rows-table` (the
table lacks a template of the record). An entry of the table for no template of the record is held to the form of
the others, and the count does not read it.

## The cost

The fold takes one step a template, a lookup in the table: it visits no unit and no row, so its work is the
record's templates times the table's entries at most, whatever the units number.
-/

@[expose] public section

namespace NciCheck.RecordRows

open Lean (Json toJson)

/-- **A block's rows**: over `units`, each a template with its units in a block, those units times the rows of one
unit of the template (`m`); none where `m` has no entry for one of them. A template is named as the caller names
it (`κ`): by its id, or by its Definition. -/
def blockRows {κ : Type} (m : κ → Option Nat) (units : List (κ × Nat)) : Option Nat :=
  units.foldl (fun acc (t, n) => acc.bind fun a => (m t).map fun b => a + n * b) (some 0)

/-- The rows of one unit of the Definition `d` of `P`: the table's entry (`m`) at the template's id, which is the
Definition's key. The template query names a root node's template by its callee's key
(`Flock.TemplateQuery.evaluate`), and the template's Definition is that callee (`RecordText.tmplOf`). None where
`P` has no Definition `d`. -/
def rowsOfDefn (P : Flock.Program.Prog) (m : String → Option Nat) (d : Nat) : Option Nat :=
  P.defs[d]?.bind fun x => m x.key

/-- A block's rows on a record's summary, the Program `P` the record's: `blockRows` on the summary's templates, each
its Definition with its units in a block, and `m` the table by template id. -/
def ofSummary (P : Flock.Program.Prog) (m : String → Option Nat) (s : Record.Summary) : Option Nat :=
  blockRows (rowsOfDefn P m) s.units

/-- **The rate rule in rows**, `failure` on the one pair: `size` where a block has more than `Γ` rows, else `nci`
where its fresh bits exceed `α = num/den` times its rows. -/
def rate (gamma num den rows fresh : Nat) : Option String :=
  (failure gamma num den #[(rows, fresh)]).map (·.2)

/-- The limits: the rows table, `Γ` and `α = num/den`. -/
structure Limits where
  /-- Per template's id, the rows of one unit of it, not 0. -/
  table : Array (String × Nat)
  /-- The most rows a block holds. -/
  gamma : Nat
  /-- `α`'s numerator: `α = num/den` is the most fresh bits a row. -/
  num : Nat
  /-- `α`'s denominator, not 0. -/
  den : Nat

/-- The rows table's entry for the template `tid`. -/
def Limits.rowsOf (l : Limits) (tid : String) : Option Nat := (l.table.find? (·.1 == tid)).map (·.2)

/-- The limits, refused (`instance`) where a field is missing or `table` is no object, and (`params`) where `gamma`
is not a natural, a value of `table` is not a natural or is 0, or `alpha` is not two naturals, the second not 0. -/
def parse (j : Json) : Except String Limits := do
  let .obj table ← field j "table" | throw "instance"
  let gammaJ ← field j "gamma"
  let alphaJ ← field j "alpha"
  let some gamma := natOf? gammaJ | throw "params"
  let some (a, b) := pairOf? alphaJ | throw "params"
  let some num := natOf? a | throw "params"
  let some den := natOf? b | throw "params"
  if den = 0 then throw "params"
  let table ← table.toArray.mapM fun (tid, v) => match natOf? v with
    | some m => if m = 0 then throw "params" else pure (tid, m)
    | none => throw "params"
  return { table, gamma, num, den }

/-- The rule on one instance: `{rows, fresh, complies, failure}`, a block's rows and fresh bits with the rule's
verdict, or the first refusal's code, in the docstring's order. -/
def evaluate (j tr limits : Json) : Except String Json := do
  let l ← parse limits
  let c ← Record.read j tr
  let s := Record.summarize c
  let some rows := ofSummary c.f.P l.rowsOf s | throw "rows-table"
  let f := rate l.gamma l.num l.den rows s.fresh
  return Json.mkObj [("rows", toJson rows), ("fresh", toJson s.fresh), ("complies", Json.bool f.isNone),
    ("failure", Record.failureJson f)]

/-- `evaluate`, with a refusal as `refusal` writes it: the whole verdict, for the private record only. -/
def checkDetail (j tr limits : Json) : Json :=
  match evaluate j tr limits with
  | .ok r => r
  | .error code => refusal code

end NciCheck.RecordRows
