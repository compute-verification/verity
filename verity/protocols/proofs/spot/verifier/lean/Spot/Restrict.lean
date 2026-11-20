module

public import Spot.BlockFront
public import Flock.Lower
meta import Flock

/-!
# `restrict`: the block's Program, at the word level

`restrict C S` is the Program of block S of the registered circuit C (`verity.circuits.restrict`), a value of
`Flock.Program.Prog`, never bytes; `ported` makes its input nodes the root's parameters, the form `Flock.Lower.lower`
lifts to bits (`Flock.Lift.lift`) and lowers. On the decoded C
and the block's units (`Spot.BlockFront.members`), with in(S) and out(S) as `Spot.BlockFront.front` computes them:

* the root `Root[Block_v1]_v1`, whose Program function is `Block_v1`, has one input batch node for each maximal run of
  in(S) of one width and one binding, in increasing C index: `RegisteredInput<w>_v1` for a run of C's registered inputs,
  `SuppliedInput<w>_v1` for a run of its supplied inputs, `Input<w>_v1` for any other (a run input, or a gate a unit
  outside S computes); a supplied input keeps its kind and not its mark, which C's description holds by the input's
  name;
* then each root node of C that S holds: a call, or a batch S holds whole, as C has it; a batch S holds in part, one node a
  member, its arguments cut on their mapped axis. Each argument leaf is the port or the earlier node that carries its gate
  of C, the first to carry it;
* its return is out(S), a tuple of `Array<n, Value<w>>`, one for each maximal run of one width;
* every Definition the root reaches is C's, and the Boolean map is C's on the word primitives it reaches;
* its `sha512` is empty, so `Prog.digest` refuses it: nothing names the block's Program by a digest this function did not
  compute.

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

/-- The root node of C whose gates hold gate `g`: `offsets` ascending from 0, one more than the nodes, as `decode` makes
them. -/
def nodeOf (root : Defn) (g : Nat) : Option Nat := Id.run do
  let o := root.body.offsets
  if o.size < 2 || g ≥ o.back?.getD 0 then return none
  let mut lo := 0
  let mut hi := o.size - 1
  for _ in [0:o.size] do
    if lo + 1 ≥ hi then break
    let mid := (lo + hi) / 2
    if o[mid]?.getD 0 ≤ g then lo := mid else hi := mid
  return some lo

/-- Gate `g`'s binding: a registered or run input's, `RUN` for a gate a node computes. -/
def bindingOf (P : Prog) (root : Defn) (g : Nat) : String :=
  match nodeOf root g with
  | some k => match root.body.nodes[k]? with
    | some x => ((P.defs[x.fn]?).bind (·.binding)).getD RUN
    | none => RUN
  | none => RUN

def inputKey (b : String) (w : Nat) : String :=
  (if b == REGISTRATION then "RegisteredInput" else if b == SUPPLIED then "SuppliedInput" else "Input") ++ s!"{w}_v1"

/-- The input primitive of width `w` and binding `b`, as `decode` makes it. -/
def inputDefn (b : String) (w : Nat) : Defn :=
  let key := inputKey b w
  { key, id := key, prim := true, params := #[], ret := .value w, binding := some b, gates := 1, paramLeaves := 0,
    outLeaves := 1 }

def sameInput (d e : Defn) : Bool :=
  let w := fun (t : Ty) => match t with | .value w => some w | _ => none
  d.key == e.key && d.id == e.id && d.prim == e.prim && d.params.isEmpty && e.params.isEmpty && w d.ret == w e.ret &&
    d.binding == e.binding && d.gates == e.gates && d.paramLeaves == e.paramLeaves && d.outLeaves == e.outLeaves

/-- A root-scoped reference of C as the port or earlier node of the block that carries its gate. -/
def leaf (P : Prog) (ref : Std.HashMap Nat Ref) (r : Ref) : V Ref := do
  let g ← resolve P P.rootScope r
  let some q := ref[g]? | throw s!"restrict: an argument leaf of the block is gate {g} of C, which no port or node carries"
  pure q

def spell (P : Prog) (ref : Std.HashMap Nat Ref) (rs : Array Ref) : V Seq := do
  pure (runs (← rs.mapM (leaf P ref)))

/-- **`restrict C S`**: block `units`' Program (the module docstring). The units are each named once (`members` refuses
a repeat). An input Definition C already holds is taken only if it is `inputDefn`'s. Every index it reads is refused by
name when out of range, and a batch's return leaves are its members' (`n` times its callee's), or it refuses. -/
def restrict (P : Prog) (units : Array Nat) : V Prog := do
  let units := units.qsort (· < ·)
  let f ← front P units
  let (ms, _) ← members P units
  let some root := P.defs[P.root]? | throw "restrict: C's root is not one of its Definitions"
  let mut defs := P.defs
  let mut index := P.index
  let mut nodes : Array Node := #[]
  let mut ref : Std.HashMap Nat Ref := {}
  for r in runsBy f.ins fun p => (p.width, bindingOf P root (Port.gate p)) do
    let some p0 := r[0]? | throw "restrict: an empty run of in(S)"
    let w := p0.width
    let b := bindingOf P root (Port.gate p0)
    let want := inputDefn b w
    let fi ← match index[want.key]? with
      | some i =>
        if (defs[i]?.map (sameInput want)).getD false then pure i
        else throw s!"restrict: C's Definition '{want.key}' is not the input primitive of width {w} and binding {b}"
      | none => do
        defs := defs.push want
        index := index.insert want.key (defs.size - 1)
        pure (defs.size - 1)
    for j in [0:r.size] do
      if let some p := r[j]? then ref := ref.insert (Port.gate p) { p := false, i := nodes.size, j }
    nodes := nodes.push { form := .batch, fn := fi, args := #[], out := .array r.size (.value w), n := r.size }
  for (k, m0, m1) in kept ms do
    let some x := root.body.nodes[k]? | throw s!"restrict: root node {k} is past C's root's nodes"
    let some c := P.defs[x.fn]? | throw s!"restrict: root node {k} calls Definition {x.fn}, past C's Definitions"
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
      let per := c.ret.leaves
      if x.out.leaves != x.n * per then
        throw s!"restrict: root node {k} returns {x.out.leaves} leaves for {x.n} members of {per}"
      for m in [m0:m1] do
        let mut args := #[]
        for i in [0:x.args.size] do
          let some a := x.args[i]? | throw s!"restrict: root node {k}'s argument {i} is past its arguments"
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
  let mut fields : Array Ty := #[]
  for r in outRuns do
    let some p0 := r[0]? | throw "restrict: an empty run of out(S)"
    fields := fields.push (.array r.size (.value p0.width))
    for p in r do
      let some q := ref[Port.gate p]? | throw s!"restrict: output gate {Port.gate p} of C, which no node carries"
      ret := ret.push q
  let retType := Ty.tuple fields
  let mut offsets := #[0]
  let mut at_ := 0
  for x in nodes do
    let some d := defs[x.fn]? | throw s!"restrict: a node of the block calls Definition {x.fn}, past its Definitions"
    at_ := at_ + x.members * (if x.form == .prim then 1 else d.gates)
    offsets := offsets.push at_
  if index.contains ROOT then throw s!"restrict: '{ROOT}' is already a Definition of C"
  let blk : Defn := { key := ROOT, id := ROOT, prim := false, params := #[], ret := retType, gates := at_,
                      paramLeaves := 0, outLeaves := retType.leaves,
                      body := { params := #[], retType, nodes, ret := runs ret, offsets } }
  let Q : Prog := { P with defs := defs.push blk, index := index.insert ROOT defs.size, root := defs.size, programFn := FN }
  return { Q with sha512 := ByteArray.empty }

/-- `Q`'s root with its leading input nodes as its parameters, in order, each the node's type: a reference to input node
`i` reads parameter `i`, one to a later node its index less the input nodes'. This is the block's Program as
`Flock.Lower.lower` and `Prog.evalBool` read it, its parameters the input ports' bits in increasing C index. Refuses a
root with parameters of its own, and a registered input node: a parameter does not say it is registered, so the verifier
would commit its rows afresh. -/
def ported (Q : Prog) : V Prog := do
  let some root := Q.defs[Q.root]? | throw "ported: the root is not one of the Program's Definitions"
  if !root.params.isEmpty then throw "ported: the root has parameters already"
  let isIn := fun (x : Node) => ((Q.defs[x.fn]?).map fun d => d.prim && isInputId d.id).getD false
  let k := (root.body.nodes.toList.takeWhile isIn).length
  let ins := root.body.nodes.extract 0 k
  let registered := fun (x : Node) => ((Q.defs[x.fn]?).map fun d => (inputId? d.id).map (·.1) == some REGISTRATION).getD false
  if let some x := ins.find? registered then
    throw s!"ported: the block reads a registered value ({(Q.defs[x.fn]!).id}), and a parameter does not carry that \
      it is registered: its rows would be committed afresh"
  if root.body.nodes.toList.drop k |>.any isIn then throw "ported: an input node after a node that is not one"
  let re := fun (r : Ref) => if r.p then r else if r.i < k then { p := true, i := r.i, j := r.j } else { r with i := r.i - k }
  let move := fun (s : Seq) => BlockFront.runs (s.toArray.map re)
  let params := (Array.range k).zipWith (fun i (x : Node) => (s!"in{i}", x.out)) ins
  let nodes := (root.body.nodes.extract k root.body.nodes.size).map fun x =>
    { x with args := x.args.map fun a => { a with seq := move a.seq } }
  let some o0 := root.body.offsets[k]? | throw "ported: the root's offsets are fewer than its input nodes"
  let offsets := (root.body.offsets.extract k root.body.offsets.size).map (· - o0)
  let blk : Defn := { root with params, gates := root.gates - o0, paramLeaves := params.foldl (· + ·.2.leaves) 0,
                                body := { root.body with params, nodes, ret := move root.body.ret, offsets } }
  return { Q with defs := Q.defs.set! Q.root blk }

/-- Block `units`' Program in the lowering's form: `ported` of `restrict`. -/
def portedOf (P : Prog) (units : Array Nat) : V Prog := restrict P units >>= ported

/-- What `front`, `members` and `restrict` read of C for block `units`, by their loops' bounds, beside the size of C's
description and C's gate count: the units; the root nodes held; the references `restrict` reads and resolves (each
argument leaf of a node held, each return leaf); the leaves `front` resolves (each unit's parameter and return leaves);
the Definitions the lift reaches. -/
def cost (P : Prog) (units : Array Nat) : V Json := do
  let some root := P.defs[P.root]? | throw "cost: C's root is not one of its Definitions"
  let (ms, _) ← members P units
  let ks := kept ms
  let mut refs := 0
  let mut whole := 0
  for (k, m0, m1) in ks do
    let some x := root.body.nodes[k]? | throw s!"cost: root node {k}"
    let some c := P.defs[x.fn]? | throw s!"cost: root node {k}'s Definition"
    if x.form == .call || (m0 == 0 && m1 == x.n) then
      whole := whole + 1
      refs := refs + x.args.foldl (· + ·.seq.count) 0 + x.out.leaves
    else
      let la := (List.range x.args.size).foldl (fun n i => match x.args[i]?, (x.axes[i]?).join with
        | some a, some 0 => n + (match a.ty with | .array _ e => e.leaves | _ => 0)
        | some a, _ => n + a.seq.count
        | none, _ => n) 0
      refs := refs + (m1 - m0) * (la + c.ret.leaves)
  let mut leaves := 0
  for (k, _, _) in ms do
    let some x := root.body.nodes[k]? | throw s!"cost: root node {k}"
    let some c := P.defs[x.fn]? | throw s!"cost: root node {k}'s Definition"
    leaves := leaves + c.paramLeaves + (if c.prim then 1 else c.outLeaves)
  let f ← front P units
  let descNodes := P.defs.foldl (· + ·.body.nodes.size) 0
  let descParts := P.defs.foldl (fun n d => n + d.body.ret.parts.size +
    d.body.nodes.foldl (fun n x => n + x.args.foldl (· + ·.seq.parts.size) 0) 0) 0
  return Json.mkObj [("units", toJson units.size), ("root_nodes_held", toJson ks.size), ("held_whole", toJson whole),
    ("restrict_refs", toJson refs), ("front_leaves", toJson leaves), ("lift_words", toJson f.lift.size),
    ("ins", toJson f.ins.size), ("outs", toJson f.outs.size),
    ("description", Json.mkObj [("definitions", toJson P.defs.size), ("nodes", toJson descNodes),
      ("reference_parts", toJson descParts)]),
    ("c_gates", toJson root.gates)]

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

/-- The first place two dumps differ, as a path, descending at most `fuel` levels: a pair still different at that depth differs
at its own path. -/
def firstDiffTo (fuel : Nat) (a b : Json) (at_ : String) : Option String :=
  if a.compress == b.compress then none else
  match fuel with
  | 0 => some at_
  | n + 1 =>
    match a, b with
    | .obj _, .obj _ =>
      let ks := (keys a ++ keys b).qsort (· < ·)
      (ks.toList.findSome? fun k => firstDiffTo n ((a.getObjVal? k).toOption.getD .null) ((b.getObjVal? k).toOption.getD .null)
        s!"{at_}/{k}").orElse fun _ => some at_
    | .arr x, .arr y =>
      if x.size != y.size then some s!"{at_} (lengths {x.size} and {y.size})" else
      ((x.zip y).toList.zipIdx.findSome? fun ((u, v), i) => firstDiffTo n u v s!"{at_}/{i}").orElse fun _ => some at_
    | _, _ => some at_

/-- The first place two dumps differ, as a path. A value nests no deeper than its text is long, so the fuel is never spent
before a leaf. -/
def firstDiff (a b : Json) (at_ : String) : Option String := firstDiffTo (a.compress.length + b.compress.length + 1) a b at_

/-- The `restrict` piece: `{descriptor, units, python}` (C's text, the block's units, Python's `restrict` bytes as text) to
`{agree, at, root_nodes, definitions, digest_bytes, cost, lowers}`: whether Lean's `restrict` equals `Flock.Program.decode`
of Python's bytes as `dump` shows them, the first path where they differ, the size of Lean's, its digest's bytes, `cost`,
and `Flock.Lower.lower`'s refusal of `ported` of it at the front's port widths, `null` where it lowers. -/
def piece (j : Json) : Except String Json := do
  let some d := (j.getObjValAs? String "descriptor").toOption | throw "descriptor is not a string"
  let some us := (j.getObjValAs? (Array Nat) "units").toOption | throw "units is not a list of naturals"
  let some py := (j.getObjValAs? String "python").toOption | throw "python is not a string"
  let P ← decode d.toUTF8
  let Q ← restrict P us
  let a ← dump Q
  let b ← dump (← decode py.toUTF8)
  let at_ := firstDiff a b ""
  let nDefs := (keys ((a.getObjVal? "definitions").toOption.getD .null)).size
  let f ← front P us
  let lowers := match ported Q >>= fun R => Flock.Lower.lower R R.root (f.ins.map (·.width)) (f.outs.map (·.width)) with
    | .ok _ => Json.null
    | .error e => .str e
  return Json.mkObj [("agree", toJson at_.isNone), ("at", match at_ with | some p => .str p | none => .null),
    ("root_nodes", toJson (((Q.defs[Q.root]?).map (·.body.nodes.size)).getD 0)), ("definitions", toJson nDefs),
    ("digest_bytes", toJson Q.sha512.size), ("cost", ← cost P us),
    ("lowers", lowers)]

#guard sameInput (inputDefn "run" 16) (inputDefn "run" 16)
#guard !sameInput (inputDefn "run" 16) { inputDefn "run" 16 with prim := false }
#guard !sameInput (inputDefn "run" 16) { inputDefn "run" 16 with binding := some "registration" }
#guard !sameInput (inputDefn "run" 16) { inputDefn "run" 16 with params := #[("x", .value 16)] }
#guard inputKey "registration" 16 == "RegisteredInput16_v1" && inputKey "run" 1 == "Input1_v1"
#guard inputKey "supplied" 32 == "SuppliedInput32_v1" && inputKey "supplied" 1 == "SuppliedInput1_v1"
#guard !sameInput (inputDefn "supplied" 16) (inputDefn "run" 16)
#guard !sameInput (inputDefn "supplied" 16) (inputDefn "registration" 16)
#guard (inputId? (inputDefn "supplied" 16).id).map (·.1) == some SUPPLIED
#guard firstDiff (Json.arr #[toJson 1, toJson 2]) (Json.arr #[toJson 1, toJson 2, toJson 3]) "" == some " (lengths 2 and 3)"
#guard firstDiff (Json.arr #[toJson 1, toJson 2, toJson 3]) (Json.arr #[toJson 1, toJson 2]) "" == some " (lengths 3 and 2)"
#guard firstDiff (Json.arr #[toJson 1, toJson 2]) (Json.arr #[toJson 1, toJson 4]) "" == some "/1"
#guard firstDiff (Json.arr #[toJson 1, toJson 2]) (Json.arr #[toJson 1, toJson 2]) "" == none

end Spot.Restrict
