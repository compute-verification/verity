module

public import Spot.BlockFront

/-!
# `restrict`: the block's Program, at the word level

`restrict C S` is the Program of block S of the registered circuit C (`verity.circuits.restrict`), a value of
`Flock.Program.Prog`, never bytes: the lowering takes it and lifts it to bits (`Spot.BlockFront.program`). On the decoded C
and the block's units (`Spot.BlockFront.members`), with in(S) and out(S) as `Spot.BlockFront.front` computes them:

* the root `Root[Block_v1]_v1`, whose Program function is `Block_v1`, has one input batch node for each maximal run of
  in(S) of one width and one binding, in increasing C index: `RegisteredInput<w>_v1` for a run of C's registered inputs,
  `Input<w>_v1` for any other (a run input, or a gate a unit outside S computes);
* then each root node of C that S holds: a call, or a batch S holds whole, as C has it; a batch S holds in part, one node a
  member, its arguments cut on their mapped axis. Each argument leaf is the port or the earlier node that carries its gate
  of C, the first to carry it;
* its return is out(S), a tuple of `Array<n, Value<w>>`, one for each maximal run of one width;
* every Definition the root reaches is C's, and the Boolean map is C's on the word primitives it reaches.

It refuses a scan, a member's argument mapped over an axis other than its first, and a leaf no port or earlier node carries.
-/

@[expose] public section

namespace Spot.Restrict

open Flock Flock.Program Spot.BlockFront
open Lean (Json toJson)

def ROOT : String := "Root[Block_v1]_v1"
def FN : String := "Block_v1"

/-- Maximal runs of consecutive ports with one key. -/
def runsBy {α : Type} [BEq α] (ps : Array BlockFront.Port) (key : BlockFront.Port → α) : Array (Array BlockFront.Port) :=
  ps.foldl (fun acc p => match acc.back? with
    | some r => if r.back?.map key == some (key p) then acc.pop.push (r.push p) else acc.push #[p]
    | none => #[#[p]]) #[]

/-- The root node of C whose gates hold gate `g`. -/
def nodeOf (root : Defn) (g : Nat) : Option Nat := Id.run do
  let o := root.body.offsets
  if o.size < 2 || g ≥ o.back! then return none
  let mut lo := 0
  let mut hi := o.size - 1
  for _ in [0:o.size] do
    if lo + 1 ≥ hi then break
    let mid := (lo + hi) / 2
    if o[mid]! ≤ g then lo := mid else hi := mid
  return some lo

/-- Gate `g`'s binding: a registered or run input's, `RUN` for a gate a node computes. -/
def bindingOf (P : Prog) (g : Nat) : String :=
  let root := P.defs[P.root]!
  match nodeOf root g with
  | some k => ((P.defs[(root.body.nodes[k]!).fn]!).binding).getD RUN
  | none => RUN

def inputKey (b : String) (w : Nat) : String := (if b == REGISTRATION then "RegisteredInput" else "Input") ++ s!"{w}_v1"

/-- The root nodes of C that the units are, each `(k, first member, end member)`, in increasing C index. -/
def kept (ms : Array (Nat × Nat × Scope)) : Array (Nat × Nat × Nat) :=
  ms.foldl (fun acc (k, m, _) => match acc.back? with
    | some (k', m0, m1) => if k' == k && m1 == m then acc.pop.push (k, m0, m + 1) else acc.push (k, m, m + 1)
    | none => #[(k, m, m + 1)]) #[]

/-- A root-scoped reference of C as the port or earlier node of the block that carries its gate. -/
def leaf (P : Prog) (ref : Std.HashMap Nat Ref) (r : Ref) : V Ref := do
  let g ← resolve P P.rootScope r
  let some q := ref[g]? | throw s!"restrict: an argument leaf of the block is gate {g} of C, which no port or node carries"
  pure q

def spell (P : Prog) (ref : Std.HashMap Nat Ref) (rs : Array Ref) : V Seq := do
  pure (runs (← rs.mapM (leaf P ref)))

/-- **`restrict C S`**: block `units`' Program (the module docstring). -/
def restrict (P : Prog) (units : Array Nat) : V Prog := do
  let units := units.qsort (· < ·)
  let f ← front P units
  let (ms, _) ← members P units
  let root := P.defs[P.root]!
  let mut defs := P.defs
  let mut index := P.index
  let mut nodes : Array Node := #[]
  let mut ref : Std.HashMap Nat Ref := {}
  for r in runsBy f.ins fun p => (p.width, bindingOf P (Port.gate p)) do
    let w := r[0]!.width
    let b := bindingOf P (Port.gate r[0]!)
    let key := inputKey b w
    let fi ← match index[key]? with
      | some i => pure i
      | none => do
        defs := defs.push { key, id := key, prim := true, params := #[], ret := .value w, binding := some b, gates := 1,
                            paramLeaves := 0, outLeaves := 1 }
        index := index.insert key (defs.size - 1)
        pure (defs.size - 1)
    for j in [0:r.size] do
      ref := ref.insert (Port.gate r[j]!) { p := false, i := nodes.size, j }
    nodes := nodes.push { form := .batch, fn := fi, args := #[], out := .array r.size (.value w), n := r.size }
  for (k, m0, m1) in kept ms do
    let x := root.body.nodes[k]!
    if x.form != .call && x.form != .batch then throw s!"restrict: root node {k} is a {x.form.name}: a block holds calls and batches"
    if x.form == .call || (m0 == 0 && m1 == x.n) then
      let mut args := #[]
      for a in x.args do args := args.push { a with seq := ← spell P ref a.seq.toArray }
      let t := nodes.size
      nodes := nodes.push { x with args }
      for j in [0:x.out.leaves] do
        let g ← resolve P P.rootScope { p := false, i := k, j }
        if !ref.contains g then ref := ref.insert g { p := false, i := t, j }
    else
      let c := P.defs[x.fn]!
      let per := x.out.leaves / x.n
      for m in [m0:m1] do
        let mut args := #[]
        for i in [0:x.args.size] do
          let a := x.args[i]!
          match (x.axes[i]?).join with
          | none => args := args.push { ty := a.ty, seq := ← spell P ref a.seq.toArray }
          | some 0 =>
            let .array _ e := a.ty | throw s!"restrict: root node {k}'s argument {i} is mapped over axis 0 of a non-array"
            let L := e.leaves
            args := args.push { ty := e, seq := ← spell P ref (a.seq.toArray.extract (m * L) ((m + 1) * L)) }
          | some ax => throw s!"restrict: root node {k}'s argument {i} is mapped over axis {ax}, which a member's cut doesn't take"
        let t := nodes.size
        nodes := nodes.push { form := if c.prim then .prim else .call, fn := x.fn, args, out := c.ret }
        for j in [0:per] do
          let g ← resolve P P.rootScope { p := false, i := k, j := m * per + j }
          if !ref.contains g then ref := ref.insert g { p := false, i := t, j }
  let outRuns := runsBy f.outs fun (p : BlockFront.Port) => p.width
  let mut ret : Array Ref := #[]
  for r in outRuns do
    for p in r do
      let some q := ref[Port.gate p]? | throw s!"restrict: output gate {Port.gate p} of C, which no node carries"
      ret := ret.push q
  let retType := Ty.tuple (outRuns.map fun r => .array r.size (.value r[0]!.width))
  let ds := defs
  let offsets := nodes.foldl (fun o x => o.push (o.back! + x.members * (if x.form == .prim then 1 else (ds[x.fn]!).gates))) #[0]
  if index.contains ROOT then throw s!"restrict: '{ROOT}' is already a Definition of C"
  let blk : Defn := { key := ROOT, id := ROOT, prim := false, params := #[], ret := retType, gates := offsets.back!,
                      paramLeaves := 0, outLeaves := retType.leaves,
                      body := { params := #[], retType, nodes, ret := runs ret, offsets } }
  return { P with defs := defs.push blk, index := index.insert ROOT defs.size, root := defs.size, programFn := FN }

/-- A Program as `restrict`'s difference test compares it: its root's key, its Program function, every Definition the root
reaches by key (`defJson`, with whether it is a primitive and its binding), and the Boolean map on the word primitives it
reaches, each reference one by one. -/
def dump (P : Prog) : V Json := do
  let bound := P.defs.foldl (fun n d => n + d.body.nodes.size + 1) 1
  let mut seen : Std.HashSet Nat := {}
  let mut todo := #[P.root]
  let mut at_ := 0
  let mut bool : Array (String × Json) := #[]
  for _ in [0:bound] do
    if h : at_ < todo.size then
      let d := todo[at_]
      at_ := at_ + 1
      if seen.contains d then continue
      seen := seen.insert d
      let defn := P.defs[d]!
      if defn.prim then
        if let some bk := P.boolean[defn.key]? then
          let some bi := P.index[bk]? | throw s!"'{defn.key}''s Boolean Definition '{bk}' is not in the Program"
          bool := bool.push (defn.key, .str bk)
          todo := todo.push bi
      else todo := todo ++ defn.body.nodes.map (·.fn)
    else break
  if at_ < todo.size then throw "dump: a walk of more steps than this test follows"
  let defs := (seen.toArray.map fun d => P.defs[d]!).qsort (·.key < ·.key)
  return Json.mkObj [("root", .str (P.defs[P.root]!).key), ("program_fn", .str P.programFn),
    ("definitions", Json.mkObj (defs.toList.map fun d => (d.key, (defJson P d).setObjVal! "prim" (toJson d.prim)
      |>.setObjVal! "binding" (match d.binding with | some b => .str b | none => .null)))),
    ("boolean", Json.mkObj ((bool.qsort (·.1 < ·.1)).toList))]

def keys (j : Json) : Array String := match j with
  | .obj kvs => kvs.foldl (init := #[]) fun acc k _ => acc.push k
  | _ => #[]

/-- The first place two dumps differ, as a path. -/
partial def firstDiff (a b : Json) (at_ : String) : Option String :=
  if a.compress == b.compress then none else
  match a, b with
  | .obj _, .obj _ =>
    let ks := (keys a ++ keys b).qsort (· < ·)
    (ks.toList.findSome? fun k => firstDiff ((a.getObjVal? k).toOption.getD .null) ((b.getObjVal? k).toOption.getD .null)
      s!"{at_}/{k}").orElse fun _ => some at_
  | .arr x, .arr y =>
    if x.size != y.size then some s!"{at_} (lengths {x.size} and {y.size})" else
    ((List.range x.size).findSome? fun i => firstDiff x[i]! y[i]! s!"{at_}/{i}").orElse fun _ => some at_
  | _, _ => some at_

/-- The `restrict` piece: `{descriptor, units, python}` (C's text, the block's units, Python's `restrict` bytes as text) to
`{agree, at, root_nodes, definitions}`: whether Lean's `restrict` equals `Flock.Program.decode` of Python's bytes as
`dump` shows them, the first path where they differ, and the size of Lean's. -/
def piece (j : Json) : Except String Json := do
  let some d := (j.getObjValAs? String "descriptor").toOption | throw "descriptor is not a string"
  let some us := (j.getObjValAs? (Array Nat) "units").toOption | throw "units is not a list of naturals"
  let some py := (j.getObjValAs? String "python").toOption | throw "python is not a string"
  let Q ← restrict (← decode d.toUTF8) us
  let a ← dump Q
  let b ← dump (← decode py.toUTF8)
  let at_ := firstDiff a b ""
  let nDefs := (keys ((a.getObjVal? "definitions").toOption.getD .null)).size
  return Json.mkObj [("agree", toJson at_.isNone), ("at", match at_ with | some p => .str p | none => .null),
    ("root_nodes", toJson (Q.defs[Q.root]!).body.nodes.size), ("definitions", toJson nDefs)]

#guard inputKey "registration" 16 == "RegisteredInput16_v1" && inputKey "run" 1 == "Input1_v1"

end Spot.Restrict
