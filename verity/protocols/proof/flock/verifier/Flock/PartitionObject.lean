module

public import Flock.Draw
public import Flock.Qcall
public import Flock.TemplateQuery

/-! A `verity/partition/v1` object's form (`Flock.HmRow`'s names), which `HmRow` checks before it reads the object, and a
template query's ranges and strata over the verifier's own program, where its `blocks` are held to cover the units. -/

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
def BLOCKS_FORM : String :=
  "partition: blocks-form: blocks is not a non-empty list of levels, each a non-empty list of pairs [lo, hi], 0 <= lo < hi"

/-- A level of `blocks`: a non-empty list of pairs, none for any other value. -/
def blockLevel (l : Json) : Option (Array (Nat × Nat)) :=
  match l with
  | .arr a => match a.mapM blockPair with
    | some ps => if ps.isEmpty then none else some ps
    | none => none
  | _ => none

/-- A partition object's `blocks` (`b`, the value at the key) as its levels, coarsest first, each `[lo, hi)` unit ranges,
none where it has none; their form only, so a reader takes them from `levelsOf` or `blocksOf`. Today's flat spelling, a
list of pairs, is no list of levels. -/
def blocksIn (b : Option Json) : V (Option (Array (Array (Nat × Nat)))) :=
  match b with
  | none => pure none
  | some (.arr a) =>
    match a.mapM blockLevel with
    | some ls => if ls.isEmpty then throw BLOCKS_FORM else pure (some ls)
    | none => throw BLOCKS_FORM
  | some _ => throw BLOCKS_FORM

/-- One level `ps` (index `k`): the first block from unit 0 and each from where the one before ends. -/
def checkLevel (k : Nat) (ps : Array (Nat × Nat)) : V Unit := do
  -- `ps` is non-empty: `blockLevel` refuses an empty level
  if ps[0]!.1 != 0 then throw s!"partition: blocks-gap: level {k}: units 0..{ps[0]!.1} are in no block"
  for i in [1:ps.size] do
    let (prev, p) := (ps[i - 1]!, ps[i]!)
    if p.1 < prev.1 then throw s!"partition: blocks-unsorted: level {k}: [{p.1}, {p.2}] after [{prev.1}, {prev.2}]"
    else if p.1 < prev.2 then throw s!"partition: blocks-overlap: level {k}: [{p.1}, {p.2}] inside [{prev.1}, {prev.2}]"
    else if p.1 > prev.2 then throw s!"partition: blocks-gap: level {k}: units {prev.2}..{p.1} are in no block"

/-- Level `k + 1` (`fine`) refines level `k` (`coarse`): every end of a block of `coarse` is the end of a block of
`fine`, in one pass over the two lists, which `checkLevel` holds in order. -/
def checkRefines (k : Nat) (coarse fine : Array (Nat × Nat)) : V Unit := do
  let mut j := 0
  for (_, e) in coarse do
    while j < fine.size && fine[j]!.2 < e do j := j + 1
    if h : j < fine.size then
      if fine[j].2 != e then
        throw s!"partition: blocks-refine: level {k + 1}: [{fine[j].1}, {fine[j].2}] does not lie inside one block of level {k}"
    else throw s!"partition: blocks-refine: level {k + 1}: no block ends at {e}, where a block of level {k} ends"

/-- `blocks` on a template query only, each level as `checkLevel`, then each level refining the one before
(`checkRefines`), in `validate`'s order. -/
def checkBlocks (template : Bool) (ls : Array (Array (Nat × Nat))) : V Unit := do
  if !template then throw "partition: blocks-query: blocks on a query that is not a template query"
  for k in [0:ls.size] do checkLevel k ls[k]!
  for k in [1:ls.size] do checkRefines (k - 1) ls[k - 1]! ls[k]!

/-- `blocks` against the population `pop`, as `partition_object.verify`: each level's last block ends at it. -/
def checkBlocksCover (obj : Json) (pop : Nat) : V Unit := do
  let some ls ← blocksIn (obj.getObjVal? "blocks").toOption | pure ()
  for k in [0:ls.size] do
    let e := ls[k]!.back!.2
    if e > pop then
      throw s!"partition: blocks-range: level {k}: the last block ends at {e}, past the population's {pop} units"
    if e < pop then throw s!"partition: blocks-gap: level {k}: units {e}..{pop} are in no block"

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

/-- Template query `name` v`version`'s templates and ranges over the verifier's own program, read by `readDesc`: the program
is the object's, and the object's `blocks`, where it has them, cover the population (`checkBlocksCover`). `refusal` refuses
a query that is not a template query. -/
def templateRangesBy {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]
    (readDesc : ByteArray → M TemplateQuery.Desc) (refusal name : String) (version : Nat) (obj : Json)
    (program : ByteArray) : M (Array String × Array TemplateQuery.TRange) := do
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  let some ts := TemplateQuery.templatesOf name version ((q.getObjVal? "params").toOption.getD Json.null)
    | throw refusal
  let ts ← ts
  let d ← readDesc program
  if (obj.getObjValAs? String "program").toOption != some (hexOf d.sha512) then
    throw "partition: the verifier's program is not the one the partition object names"
  let rs ← match TemplateQuery.evaluate d ts with
    | .ok rs => pure rs
    | .error e => throw s!"partition: {e}"
  checkBlocksCover obj (TemplateQuery.population rs)
  return (ts, rs)

/-- The strata of the verifier's own program, read by `readDesc`, and partition object, and the population: the query applied
to the program, its units grouped by template descriptor id (`Draw.strataOf`). Only `Q_template_instance` v0 and
`Q_template_instances` v0 define strata; `samplingDesign` names the sampling design that asks, in the refusal. -/
def templateStrataBy {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]
    (readDesc : ByteArray → M TemplateQuery.Desc) (samplingDesign : String) (obj : Json) (program : ByteArray) :
    M (Array (String × Array (Nat × Nat)) × Nat) := do
  let (name, version) ← checkPartitionObject obj
  let (_, rs) ← templateRangesBy readDesc
    s!"{samplingDesign}: {name} v{version} defines no strata (Q_template_instance v0 and Q_template_instances v0 do)"
    name version obj program
  return (Draw.strataOf (rs.map fun r => (r.template, r.first, r.count)), TemplateQuery.population rs)

/-- The partition object's levels of `blocks`, coarsest first, only once each covers the population the query derives
from the verifier's own `program` (`templateStrataBy`); none where it has none. A reader of `blocks` takes them here or
from `blocksOf`, never from the object. -/
def levelsOf (obj : Json) (program : ByteArray) : V (Option (Array (Array (Nat × Nat)))) := do
  let some b := (obj.getObjVal? "blocks").toOption | pure none
  discard <| templateStrataBy TemplateQuery.Desc.parse "blocks" obj program
  blocksIn (some b)

/-- `blocks-levels`' refusal, at an object of `n` levels. -/
def blocksLevels (n : Nat) : String :=
  s!"partition: blocks-levels: the object has {n} levels of blocks; this reader takes one"

/-- The partition object's proof blocks, its one level (`levelsOf`); an object of more than one level is refused
(`blocksLevels`) until its reader takes levels. -/
def blocksOf (obj : Json) (program : ByteArray) : V (Option (Array (Nat × Nat))) := do
  let some ls ← levelsOf obj program | pure none
  if ls.size != 1 then throw (blocksLevels ls.size)
  pure (some ls[0]!)

end Flock.HmRow
