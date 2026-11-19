module

public import Flock.Qcall
public import Flock.TemplateQuery

/-! A `verity/partition/v1` object's form (`Flock.HmRow`'s names), which `HmRow` checks before it reads the object. -/

@[expose] public section

namespace Flock.HmRow

open Lean (Json)

def PARTITION_V1 : String := "verity/partition/v1"

def isSha512Hex (s : String) : Bool := s.length == 128 && s.all (fun c => c.isDigit || ('a' ≤ c && c ≤ 'f'))

/-- The queries the verifier can evaluate itself (`Flock.Qword`), given the program's Call graphs. -/
def EVALUABLE : List (String × Nat) := [("Q_word", 1), ("Q_word", 2)]

/-- `Q_call` v1, which the verifier evaluates itself (`Flock.Qcall`). -/
def Q_CALL : String × Nat := ("Q_call", 1)

/-- A `blocks` pair `[lo, hi]` of natural numbers with `lo < hi`, and none for any other value. -/
def blockPair (p : Json) : Option (Nat × Nat) :=
  match p with
  | .arr a =>
    if h : a.size = 2 then
      match (a[0]'(by omega)).getNat?, (a[1]'(by omega)).getNat? with
      | .ok lo, .ok hi => if lo < hi then some (lo, hi) else none
      | _, _ => none
    else none
  | _ => none

/-- `blocks-form`'s refusal. -/
def BLOCKS_FORM : String := "partition: blocks-form: blocks is not a non-empty list of pairs [lo, hi], 0 <= lo < hi"

/-- A partition object's `blocks` (`b`, the value at the key) as `[lo, hi)` unit ranges, none where it has none; their
form only, so a reader takes them from `blocksOf`. -/
def blocksIn (b : Option Json) : V (Option (Array (Nat × Nat))) :=
  match b with
  | none => pure none
  | some (.arr a) =>
    match a.mapM blockPair with
    | some ps => if ps.isEmpty then throw BLOCKS_FORM else pure (some ps)
    | none => throw BLOCKS_FORM
  | some _ => throw BLOCKS_FORM

/-- `blocks` on a template query only, the first from unit 0 and each from where the one before ends, in `validate`'s order. -/
def checkBlocks (template : Bool) (ps : Array (Nat × Nat)) : V Unit := do
  if !template then throw "partition: blocks-query: blocks on a query that is not a template query"
  -- `ps` is non-empty: `blocksIn` refuses an empty list
  if ps[0]!.1 != 0 then throw s!"partition: blocks-gap: units 0..{ps[0]!.1} are in no block"
  for i in [1:ps.size] do
    let (prev, p) := (ps[i - 1]!, ps[i]!)
    if p.1 < prev.1 then throw s!"partition: blocks-unsorted: [{p.1}, {p.2}] after [{prev.1}, {prev.2}]"
    else if p.1 < prev.2 then throw s!"partition: blocks-overlap: [{p.1}, {p.2}] inside [{prev.1}, {prev.2}]"
    else if p.1 > prev.2 then throw s!"partition: blocks-gap: units {prev.2}..{p.1} are in no block"

/-- `blocks` against the population `pop`, as `partition_object.verify`: the last block ends at it. -/
def checkBlocksCover (obj : Json) (pop : Nat) : V Unit := do
  let some ps ← blocksIn (obj.getObjVal? "blocks").toOption | pure ()
  let e := ps.back!.2
  if e > pop then throw s!"partition: blocks-range: the last block ends at {e}, past the population's {pop} units"
  if e < pop then throw s!"partition: blocks-gap: units {e}..{pop} are in no block"

/-- A `verity/partition/v1` object's form (`verity.ir.cut`'s canonical form, open query names): exactly `format`, `program`, `query`, and `blocks`
optionally (`blocksIn`, `checkBlocks`); a SHA-512 program; a query of exactly `name` (a non-empty string), `version` (a natural number) and `params` (an object). For a
query the verifier evaluates, its parameters too (`Q_word` v1 and v2: exactly `X`, `W`, positive integers; `Q_call` v1: exactly
`X`, an integer in 1..2^16). The query's name and version. -/
def checkPartitionObject (obj : Json) : V (String × Nat) := do
  let keys := fun (j : Json) => match j with
    | .obj kvs => (kvs.toArray.map (·.1)).qsort (· < ·)
    | _ => #[]
  if keys obj != #["format", "program", "query"] && keys obj != #["blocks", "format", "program", "query"] then
    throw "partition: the object's keys are not format, program, query and optionally blocks"
  if (obj.getObjValAs? String "format").toOption != some PARTITION_V1 then throw s!"partition: not a {PARTITION_V1} object"
  if !isSha512Hex ((obj.getObjValAs? String "program").toOption.getD "") then throw "partition: program is not a SHA-512 digest"
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  if keys q != #["name", "params", "version"] then throw "partition: the query's keys are not name, version, params"
  let name := (q.getObjValAs? String "name").toOption.getD ""
  if name.isEmpty then throw "partition: the query's name is not a non-empty string"
  let version ← match (q.getObjVal? "version").toOption with
    | some (.num n) => if n.exponent == 0 && n.mantissa ≥ 0 then pure n.mantissa.toNat else throw "partition: the query's version is not a natural number"
    | _ => throw "partition: the query's version is not a natural number"
  let ps := (q.getObjVal? "params").toOption.getD Json.null
  match ps with
  | .obj _ => pure ()
  | _ => throw "partition: the query's params are not an object"
  if EVALUABLE.contains (name, version) then
    if keys ps != #["W", "X"] then throw s!"partition: Q_word v{version}'s parameters are not X, W"
    for k in ["X", "W"] do
      match (ps.getObjVal? k).toOption with
      | some (.num n) => if !(n.exponent == 0 && n.mantissa > 0) then throw s!"partition: {k} is not a positive integer"
      | _ => throw s!"partition: {k} is not a positive integer"
  if (name, version) == Q_CALL then
    if keys ps != #["X"] then throw "partition: Q_call v1's parameters are not X"
    match (ps.getObjVal? "X").toOption with
    | some (.num n) =>
      if !(n.exponent == 0 && n.mantissa > 0 && n.mantissa ≤ Qcall.PARAM_MAX) then
        throw s!"partition: X is not an integer in 1..{Qcall.PARAM_MAX}"
    | _ => throw s!"partition: X is not an integer in 1..{Qcall.PARAM_MAX}"
  if let some ts := TemplateQuery.templatesOf name version ps then discard ts
  if let some bs ← blocksIn (obj.getObjVal? "blocks").toOption then
    checkBlocks (TemplateQuery.templatesOf name version ps).isSome bs
  return (name, version)

end Flock.HmRow
