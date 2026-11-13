module

public import Base.Bytes
public import Base.Canon
public import Base.Hash
public import Circuits.Steps

/-!
# A Definition's digests, read off its Program's descriptor

Two digests of a Definition of a Program (`verity-ir/descriptor/v1`), each computed from the descriptor's JSON, with no
decode of the Program:

* `definitionDigest`, the circuit format's own (`verity/circuits/PROTOCOL.md` §4.8, `codec.definition_digest`): the
  SHA-512 of the Definition's canonical entry (`canonicalEntry`). A callee enters it by descriptor id, so it covers the
  Definition's own body and none of its callees'.
* `closureDigest`: the SHA-512 of the canonical JSON object that maps each Definition one reaches, by descriptor id, to
  its `definitionDigest`. It covers the Definition's body and every body under it, so a check that takes a Definition
  by this digest takes no other body under its name: the digest a pin is computed with.

Code: Lean core, the base package's bytes, canonical JSON and SHA-512 (`Base.Bytes`, `Base.Canon`, `Base.Hash`) and the
circuit format's step bound (`Circuits.Steps`); nothing of the policy's check, whose package it shares so that every
check requiring this package reads one definition. The period check takes its slot entries' Definitions by
`closureDigest`.
-/

@[expose] public section

namespace NciCheck.DefinitionDigest

open Lean (Json)
open Flock (V)

/-- The parts reference sequence `j` stands for, flat (PROTOCOL.md §4.8): an alias's are the parts of argument `a` of node
`k` (`rows`, the nodes before the one `j` is in), a concatenation's are its parts' spliced in, and any other is its one
part. Concatenations nest at most `fuel` deep. -/
def flatParts (rows : Array (Array (Array Json))) : Nat → Json → V (Array Json)
  | 0, _ => throw "program: references nest deeper than this verifier follows"
  | fuel + 1, j => do
    if let .ok a := j.getObjVal? "alias" then
      let ps := match a with
        | .arr #[k, x] => match k.getNat?, x.getNat? with
          | .ok k, .ok x => rows[k]?.bind (·[x]?)
          | _, _ => none
        | _ => none
      let some ps := ps | throw s!"program: alias {a.compress} names no argument of an earlier node"
      return ps
    if let .ok (.arr ps) := j.getObjVal? "concat" then
      return (← ps.mapM (flatParts rows fuel)).foldl (· ++ ·) #[]
    return #[j]

/-- A flat sequence as the canonical entry writes it: its one part, or `{"concat": parts}`. -/
def written (ps : Array Json) : Json := match ps with
  | #[p] => p
  | ps => Json.mkObj [("concat", .arr ps)]

/-- A Definition's canonical entry (`codec.canonical_entry`, PROTOCOL.md §4.8), from its descriptor entry `e` and the
descriptor's interning (`callees`, `types`): each node's callee id (`fn`), output type (`out`) and argument types inline,
every alias replaced by the parts it stands for and every reference sequence flat, so one Definition has one entry in any
Program, whichever aliases its writer chose. -/
def canonicalEntry (callees types : Array Json) (e : Json) : V Json := do
  let set := fun (j : Json) (k : String) (v : Json) => match j with
    | .obj kvs => Json.obj (kvs.insert k v)
    | j => j
  let at_ := fun (xs : Array Json) (i : Json) => match i.getNat? with
    | .ok i => xs.getD i .null
    | .error _ => .null
  unless e.getObjValD "form" == .str "composite" do return e
  let body := e.getObjValD "body"
  let .arr nodes := body.getObjValD "nodes" | return e
  let mut rows : Array (Array (Array Json)) := #[]
  let mut out : Array Json := #[]
  for n in nodes do
    let .arr args := n.getObjValD "args" | throw "program: a node's arguments are not a list"
    let row ← args.mapM fun a => flatParts rows Flock.Steps.FUEL (a.getObjValD "refs")
    rows := rows.push row
    let args := (args.zip row).map fun (a, ps) => set (set a "type" (at_ types (a.getObjValD "type"))) "refs" (written ps)
    out := out.push (set (set (set n "fn" (at_ callees (n.getObjValD "fn"))) "out" (at_ types (n.getObjValD "out"))) "args"
      (.arr args))
  let ret ← flatParts rows Flock.Steps.FUEL (body.getObjValD "ret")
  return set e "body" (set (set body "nodes" (.arr out)) "ret" (written ret))

/-- `codec.definition_digest`: the SHA-512, in hex, of a Definition's canonical entry (`canonicalEntry`). -/
def definitionDigest (callees types : Array Json) (e : Json) : V String := do
  pure (Flock.hexOf (Flock.Sha512.hash (← Flock.canon (← canonicalEntry callees types e)).toUTF8))

/-- The closure digest of Definition `key` of the Program whose descriptor is `j`: the SHA-512, in hex, of the
canonical JSON object that maps each Definition `key` reaches (itself, its nodes' callees, theirs, …), by descriptor
id, to its `definitionDigest`. It digests the Definition's whole body, so another Definition under its id has another
digest. -/
def closureDigest (j : Json) (key : String) : V String := do
  let defs := j.getObjValD "definitions"
  let tab := fun (k : String) => match j.getObjValD k with | .arr xs => xs | _ => #[]
  let (callees, types) := (tab "callees", tab "types")
  let nodesOf := fun (e : Json) => match (e.getObjValD "body").getObjValD "nodes" with | .arr xs => xs | _ => #[]
  let fuel := match defs with
    | .obj kvs => kvs.foldl (fun n _ e => n + (nodesOf e).size) 1
    | _ => 1
  let mut seen : Array (String × Json) := #[]
  let mut todo : List String := [key]
  for _ in [0:fuel] do
    match todo with
    | [] => break
    | k :: rest =>
      todo := rest
      if seen.any (·.1 == k) then continue
      let e := defs.getObjValD k
      if let .null := e then throw s!"program: {key} reaches {k}, no Definition of its descriptor"
      seen := seen.push (k, .str (← definitionDigest callees types e))
      if e.getObjValD "form" == .str "composite" then
        for n in nodesOf e do
          match (n.getObjValD "fn").getNat?.toOption.bind (callees[·]?) with
          | some (.str c) => todo := c :: todo
          | _ => throw s!"program: {k}: a node's callee is outside callees"
  unless todo.isEmpty do throw s!"program: {key}'s callees outrun its descriptor's nodes"
  pure (Flock.hexOf (Flock.Sha512.hash (← Flock.canon (Json.mkObj seen.toList)).toUTF8))

end NciCheck.DefinitionDigest
