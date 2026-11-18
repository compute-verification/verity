module

public import Flock
public import Std.Data.HashMap
public import Std.Data.HashSet

/-!
# The front of a block's circuit: its gates, in(S), out(S) and the lift to bits

The part of `restrict : C -> Partition -> Block -> Circuit` that the door computes before any lowering, on the decoded
registered circuit `P` (`Flock.Program.decode`) and the block's units:

* **S**, as gate ranges `[lo, hi)` of C's canonical layout, one a unit: unit `u` is member `m` of root node `k`, the root's
  Calls (`Prog.calls`) counted in node order, a call one unit and a batch one a member, as `Flock.TemplateQuery.evaluate`
  numbers them;
* **the input ports**: the gates the units' parameter leaves name (`Flock.Program.operands` at the root) that are outside
  S, in increasing C index, each once, with its width;
* **the output ports**: the gates the units' return leaves name (`Flock.Program.resolve` in the member's activation) that
  are inside S, in increasing C index, each once, with its width;
* **the lift**: every primitive the units' Definitions reach, each Definition once; a gate, a constant bit or an input
  stays, and every other one is a word primitive (`isWordPrim`) and is replaced by the Boolean Definition C's Boolean map
  (`Prog.boolean`, PROTOCOL.md 4.9) names for it.

The ports are the leaves the block's Definitions are bound to, so they contain Core's `Circuit.into S` and
`Circuit.exits S`; they equal them when every parameter leaf is read inside S and every return leaf is read outside S or is
an output.

Its cost is the block's parameter and return leaves, each resolved down the call hierarchy, and the Definitions it reaches,
each once: the descriptor and the block's boundary, never a gate of the block's interior.

`program` is the block's Program on bits, the value `Flock.Lower.lower` takes: each word composite the units reach lifted
as circuits' `versions.lift` lifts it (`Ty.bitType`, `bitAt`, `liftSeq`), each word primitive's node a node of its Boolean
Definition, and the block Definition, whose parameters are the input ports and whose return is the output ports, each a
port's bits. Its references are equal to the lift's leaf by leaf, not yet part by part; the encoder that writes them as the
codec does follows circuits' written rule for the block's root, whose key `BLOCK` stands in for until then.
-/

@[expose] public section

namespace Spot.BlockFront

open Flock Flock.Program
open Lean (Json toJson)

/-- A gate of C on the block's boundary and its width in bits. -/
structure Port where
  gate : Nat
  width : Nat
  deriving BEq, Inhabited, Repr

/-- The front of a block's circuit (the module docstring). `lift` pairs a word primitive's descriptor key with its Boolean
Definition's, sorted by the first. -/
structure Front where
  ranges : Array (Nat × Nat)
  ins : Array Port
  outs : Array Port
  lift : Array (String × String)
  deriving Inhabited

/-- Unit `u` of C: the root node it is a member of and the member. -/
def unitAt (P : Prog) (u : Nat) : V (Nat × Nat) := do
  let root := P.defs[P.root]!
  let mut at_ := 0
  for k in P.calls do
    let x := root.body.nodes[k]!
    let n ← match x.form with
      | .call => pure 1
      | .batch => pure x.n
      | f => throw s!"root node {k} is a {f.name}: a unit is a member of a call or a batch"
    if u < at_ + n then return (k, u - at_)
    at_ := at_ + n
  throw s!"unit {u} is past C's {at_} units"

/-- Gate `g`'s width in bits: its primitive's return. -/
def width (P : Prog) (g : Nat) : V Nat := do
  let (prim, _, _) ← P.gate g
  return (P.defs[prim]!).ret.widths.foldl (· + ·) 0

def inside (S : Array (Nat × Nat)) (g : Nat) : Bool := S.any fun (lo, hi) => lo ≤ g && g < hi

/-- `xs` in increasing order, each once. -/
def sortDedup (xs : Array Nat) : Array Nat :=
  (xs.qsort (· < ·)).foldl (fun acc x => if acc.back? == some x then acc else acc.push x) #[]

/-- The lift (the module docstring): from the Definitions `start`, every Definition they reach once, each word primitive
paired with its Boolean Definition's key in C's Boolean map. -/
def lift (P : Prog) (start : Array Nat) : V (Array (String × String)) := do
  let bound := start.size + P.defs.foldl (fun n d => n + d.body.nodes.size) 0
  let mut seen : Std.HashSet Nat := {}
  let mut todo := start
  let mut at_ := 0
  let mut out : Array (String × String) := #[]
  for _ in [0:bound + 1] do
    if h : at_ < todo.size then
      let d := todo[at_]
      at_ := at_ + 1
      if seen.contains d then continue
      seen := seen.insert d
      let defn := P.defs[d]!
      if defn.prim then
        if !isWordPrim defn.prim defn.key then continue
        match P.boolean[defn.key]? with
        | some b => out := out.push (defn.key, b)
        | none => throw s!"the lift to bits: word primitive '{defn.key}' has no Boolean Definition in C's Boolean map"
      else
        todo := todo ++ defn.body.nodes.map (·.fn)
    else break
  return out.qsort (·.1 < ·.1)

/-- The units' root nodes, members and activations, in the order given, and S. -/
def members (P : Prog) (units : Array Nat) : V (Array (Nat × Nat × Scope) × Array (Nat × Nat)) := do
  if units.isEmpty then throw "a block has at least one unit"
  if (sortDedup units).size != units.size then throw "a block names each of its units once"
  let mut S := #[]
  let mut ms := #[]
  for u in units do
    let (k, m) ← unitAt P u
    let sc := P.child P.rootScope k m
    S := S.push (sc.off, sc.off + (P.defs[sc.d]!).gates)
    ms := ms.push (k, m, sc)
  return (ms, S)

/-- **The front of block `units`'s circuit** (the module docstring), the units in any order and each once. -/
def front (P : Prog) (units : Array Nat) : V Front := do
  let root := P.defs[P.root]!
  let (members, S) ← members P units
  let mut ins := #[]
  let mut outs := #[]
  for (k, m, sc) in members do
    let x := root.body.nodes[k]!
    for g in ← operands P P.rootScope x m do
      if !inside S g then ins := ins.push g
    let callee := P.defs[x.fn]!
    if callee.prim then
      outs := outs.push sc.off
    else
      for j in [0:callee.outLeaves] do
        let g ← resolve P sc (callee.body.ret.get j)
        if inside S g then outs := outs.push g
  let port := fun g => do return ({ gate := g, width := ← width P g } : Port)
  return { ranges := S.qsort (·.1 < ·.1), ins := ← (sortDedup ins).mapM port, outs := ← (sortDedup outs).mapM port,
           lift := ← lift P (members.map fun (k, _, _) => (root.body.nodes[k]!).fn) }

/-! ### The block's Program on bits -/

/-- `bit_offsets`: word leaf `j` of `t` as its first bit in `Ty.bitType t` and its width, each leaf `w` consecutive bits in leaf
order. -/
def bitAt (t : Ty) : Nat → Nat × Nat :=
  match Widths.homo t with
  | some w => fun j => (j * w, w)
  | none =>
    let ws := t.widths
    let pre := ws.foldl (fun a w => a.push (a.back! + w)) #[0]
    fun j => (pre[j]!, ws[j]!)

/-- A body's references on bits: each word reference as the bits it occupies, an affine run over a collection of one width
an affine run of bits, every other part its bits named one by one. `at_ p i` is collection `(p, i)`'s `bitAt`. -/
def liftSeq (at_ : Bool → Nat → Nat → Nat × Nat) (homo : Bool → Nat → Option Nat) (s : Seq) : Seq :=
  let expand := fun (q : Part) => (Array.range q.count).foldl (fun a k =>
    let r := q.get k
    let (o, w) := at_ r.p r.i r.j
    a ++ (Array.range w).map fun t => ({ p := r.p, i := r.i, j := o + t } : Ref)) #[]
  Seq.ofParts <| s.parts.map fun q => match q with
    | .affine p i b c => match homo p i with
      | some w => .affine p i (b * w) (c * w)
      | none => .explicit (expand q)
    | q => .explicit (expand q)

/-- Bit references as a part each run of consecutive leaves of one collection. -/
def runs (rs : Array Ref) : Seq := Id.run do
  let mut ps : Array Part := #[]
  let mut cur : Option (Bool × Nat × Nat × Nat) := none
  for r in rs do
    match cur with
    | some (p, i, b, c) =>
      if p == r.p && i == r.i && b + c == r.j then cur := some (p, i, b, c + 1)
      else
        ps := ps.push (.affine p i b c)
        cur := some (r.p, r.i, r.j, 1)
    | none => cur := some (r.p, r.i, r.j, 1)
  if let some (p, i, b, c) := cur then ps := ps.push (.affine p i b c)
  return Seq.ofParts ps

/-- A lifted Definition's descriptor key: its word's, with the static `IR="boolean"` after its own (`versions.lift`). -/
def liftedKey (key : String) : String :=
  if key.endsWith "}" then String.mk key.toList.dropLast ++ ",IR=\"boolean\"}" else key ++ "{IR=\"boolean\"}"

/-- Word composite `d` on bits (`versions.lift_body`): every type `Ty.bitType`, every reference `liftSeq`, each node calling
`fnOf` of its Definition, a word primitive's node a Call of its Boolean Definition, and the offsets and gate count from
the callees' `gatesOf`. -/
def liftDef (d : Defn) (fnOf : Nat → Nat) (primOf : Nat → Bool) (gatesOf : Nat → Nat) : Defn :=
  let b := d.body
  let tyOf := fun (p : Bool) (i : Nat) =>
    if p then (d.params[i]?.map (·.2)).getD (.value 1) else (b.nodes[i]?.map (·.out)).getD (.value 1)
  let pAt := d.params.map fun (_, t) => bitAt t
  let nAt := b.nodes.map fun x => bitAt x.out
  let at_ := fun (p : Bool) (i : Nat) => ((if p then pAt[i]? else nAt[i]?).getD fun _ => (0, 0))
  let homo := fun p i => Widths.homo (tyOf p i)
  let lseq := liftSeq at_ homo
  let nodes := b.nodes.map fun x =>
    let f := fnOf x.fn
    let args := x.args.mapIdx fun i a =>
      let ty := Ty.bitType a.ty
      let dims := match x.form with
        | .batch => match x.axes[i]? with
          | some (some ax) => ty.axisDims ax
          | _ => none
        | .scan => if 1 ≤ i && i ≤ x.nxs then (match ty with | .array ln e => some (ln, e.leaves) | _ => none) else none
        | _ => none
      ({ ty, seq := lseq a.seq, dims } : Arg)
    { x with fn := f, args, out := Ty.bitType x.out, form := if x.form == .prim && !primOf f then .call else x.form }
  let offsets := nodes.foldl (fun o x => o.push (o.back! + (if x.form == .prim then 1 else x.members * gatesOf x.fn))) #[0]
  let params := d.params.map fun (n, t) => (n, Ty.bitType t)
  let ret := Ty.bitType d.ret
  { d with key := liftedKey d.key, params, ret, gates := offsets.back!,
           paramLeaves := params.foldl (· + ·.2.leaves) 0, outLeaves := ret.leaves,
           body := { params, retType := ret, nodes, ret := lseq b.ret, offsets } }

/-- The block Definition's descriptor key, until circuits' rule for the block's root is written. -/
def BLOCK : String := "Block_v1"

/-- **The block's Program** (`restrict`'s Program value, which `Flock.Lower.lower` takes): `P` with every word composite
the units reach lifted to bits (`liftDef`) and the block Definition last, whose index is the result's `root`. Its
parameters are the input ports and its return the output ports, each port its bits, in increasing C index; its nodes are
the units, in increasing C index, each a Call of its lifted Definition, reading a port's bits or an earlier unit's return
bits. Each word primitive is replaced by its Boolean Definition in C's Boolean map (`Prog.boolean`), which is a gate or a
composite with one-bit parameters and return that reaches no word primitive, or is refused. -/
def program (P : Prog) (units : Array Nat) : V Prog := do
  let units := units.qsort (· < ·)
  let (ms, S) ← members P units
  let root := P.defs[P.root]!
  let word := fun (d : Defn) => isWordPrim d.prim d.key
  let bound := P.defs.foldl (fun n d => n + d.body.nodes.size + 1) 1
  let mut seen : Std.HashSet Nat := {}
  let mut todo := ms.map fun (k, _, _) => (root.body.nodes[k]!).fn
  let mut at_ := 0
  let mut toB : Std.HashMap Nat Nat := {}
  for _ in [0:bound] do
    if h : at_ < todo.size then
      let d := todo[at_]
      at_ := at_ + 1
      if seen.contains d then continue
      seen := seen.insert d
      let defn := P.defs[d]!
      if word defn then
        let some bk := P.boolean[defn.key]?
          | throw s!"the lift to bits: word primitive '{defn.key}' has no Boolean Definition in C's Boolean map"
        let some bi := P.index[bk]? | throw s!"the lift to bits: '{defn.key}''s Boolean Definition '{bk}' is not in C"
        let bd := P.defs[bi]!
        if !(bd.ret.bits && bd.params.all fun (_, t) => t.bits) || (bd.prim && !isBooleanPrim bd.key) then
          throw s!"the lift to bits: '{defn.key}''s Boolean Definition '{bk}' is neither a gate nor a composite on bits"
        toB := toB.insert d bi
        todo := todo.push bi
      else if !defn.prim then
        todo := todo ++ defn.body.nodes.map (·.fn)
    else break
  if at_ < todo.size then throw "the lift to bits: a walk of more steps than this verifier follows"
  let order := seen.toArray.qsort (· < ·)
  let mut wordy : Std.HashMap Nat Bool := {}
  for d in order do
    let defn := P.defs[d]!
    if defn.prim then wordy := wordy.insert d (word defn)
    else
      let mut w := false
      for x in defn.body.nodes do
        match wordy[x.fn]? with
        | some b => w := w || b
        | none => throw s!"the lift to bits: '{defn.key}' calls a Definition decoded after it"
      wordy := wordy.insert d w
  for (_, bi) in toB do
    if wordy[bi]?.getD true then throw s!"the lift to bits: Boolean Definition '{(P.defs[bi]!).key}' reaches a word primitive"
  let fnOf := fun (d : Nat) => toB[d]?.getD d
  let mut defs := P.defs
  let mut index := P.index
  for d in order do
    if wordy[d]?.getD false && !(P.defs[d]!).prim then
      let ds := defs
      let l := liftDef (P.defs[d]!) fnOf (fun f => (ds[f]!).prim) (fun f => (ds[f]!).gates)
      if let some j := index[l.key]? then
        if j != d then throw s!"the lift to bits: '{l.key}' is already a Definition of C"
      defs := defs.set! d l
      index := index.insert l.key d
  let mut read := #[]
  for (k, m, _) in ms do
    for g in ← operands P P.rootScope root.body.nodes[k]! m do
      if !inside S g then read := read.push g
  let ins ← (sortDedup read).mapM fun g => do return (g, ← width P g)
  let mut inIdx : Std.HashMap Nat Nat := {}
  for (g, _) in ins do inIdx := inIdx.insert g inIdx.size
  let mut made : Std.HashMap Nat (Nat × Nat) := {}
  let mut nodes : Array Node := #[]
  let mut outs : Array Nat := #[]
  for t in [0:ms.size] do
    let (k, m, sc) := ms[t]!
    let x := root.body.nodes[k]!
    let c := P.defs[x.fn]!
    let ops ← operands P P.rootScope x m
    let mut args : Array Arg := #[]
    let mut pos := 0
    for (_, pt) in c.params do
      let mut rs : Array Ref := #[]
      for w in pt.widths do
        let g := ops[pos]!
        pos := pos + 1
        match inIdx[g]? with
        | some i => rs := rs ++ (Array.range w).map fun b => ({ p := true, i, j := b } : Ref)
        | none => match made[g]? with
          | some (u, o) => rs := rs ++ (Array.range w).map fun b => ({ p := false, i := u, j := o + b } : Ref)
          | none => throw s!"unit {units[t]!} reads gate {g} of the block that no earlier unit returns"
      args := args.push { ty := Ty.bitType pt, seq := runs rs }
    let f := fnOf x.fn
    let lc := defs[f]!
    nodes := nodes.push { form := if lc.prim then .prim else .call, fn := f, args, out := Ty.bitType c.ret }
    if c.prim then
      if !made.contains sc.off then made := made.insert sc.off (t, 0)
      outs := outs.push sc.off
    else
      let ra := bitAt c.ret
      for j in [0:c.outLeaves] do
        let g ← resolve P sc (c.body.ret.get j)
        if inside S g then
          outs := outs.push g
          if !made.contains g then made := made.insert g (t, (ra j).1)
  let outPorts ← (sortDedup outs).mapM fun g => do return (g, ← width P g)
  let mut ret : Array Ref := #[]
  for (g, w) in outPorts do
    let (u, o) := made[g]!
    ret := ret ++ (Array.range w).map fun b => ({ p := false, i := u, j := o + b } : Ref)
  let params := ins.map fun (g, w) => (s!"g{g}", Ty.bitType (.value w))
  let retType := Ty.tuple (outPorts.map fun (_, w) => Ty.bitType (.value w))
  let ds := defs
  let offsets := nodes.foldl (fun o x => o.push (o.back! + (if x.form == .prim then 1 else (ds[x.fn]!).gates))) #[0]
  let blk : Defn := { key := BLOCK, id := BLOCK, prim := false, params, ret := retType, gates := offsets.back!,
                      paramLeaves := params.foldl (· + ·.2.leaves) 0, outLeaves := retType.leaves,
                      body := { params, retType, nodes, ret := runs ret, offsets } }
  if index.contains BLOCK then throw s!"'{BLOCK}' is already a Definition of C"
  return { P with defs := defs.push blk, index := index.insert BLOCK defs.size, root := defs.size }

/-- A Definition as the `block_program` and `definition` pieces show it: types, gates, and each node's and the return's
references one by one. -/
def defJson (P : Prog) (d : Defn) : Json :=
  let refs := fun (s : Seq) => Json.arr (s.toArray.map Ref.toJson)
  Json.mkObj [("key", .str d.key), ("params", .arr (d.params.map (·.2.toJson))), ("ret", d.ret.toJson),
    ("gates", toJson d.gates), ("refs", refs d.body.ret),
    ("nodes", .arr (d.body.nodes.map fun x => Json.mkObj [("form", .str x.form.name),
      ("fn", .str (P.defs[x.fn]!).key), ("out", x.out.toJson), ("n", toJson x.members),
      ("args", .arr (x.args.map fun a => Json.mkObj [("ty", a.ty.toJson), ("refs", refs a.seq),
        ("dims", match a.dims with | some (d, l) => Json.arr #[toJson d, toJson l] | none => .null)]))]))]

/-- The `block_program` piece: `{descriptor, units}` to `{block, lifted}`: the block Definition and each lifted Definition
by key (`defJson`). -/
def programPiece (j : Json) : Except String Json := do
  let some d := (j.getObjValAs? String "descriptor").toOption | throw "descriptor is not a string"
  let some us := (j.getObjValAs? (Array Nat) "units").toOption | throw "units is not a list of naturals"
  let P ← decode d.toUTF8
  let Q ← program P us
  let lifted := (Q.defs.extract 0 P.defs.size).toList.filter (·.key.endsWith "IR=\"boolean\"}")
  return Json.mkObj [("block", defJson Q Q.defs[Q.root]!),
    ("lifted", Json.mkObj (lifted.map fun x => (x.key, defJson Q x)))]

/-- The `definition` piece: `{descriptor, key}` to `defJson` of that Definition of the decoded descriptor. -/
def definitionPiece (j : Json) : Except String Json := do
  let some d := (j.getObjValAs? String "descriptor").toOption | throw "descriptor is not a string"
  let some k := (j.getObjValAs? String "key").toOption | throw "key is not a string"
  let P ← decode d.toUTF8
  let some i := P.index[k]? | throw s!"no Definition '{k}'"
  return defJson P P.defs[i]!

/-- The `block_front` piece: `{descriptor, units}` (the descriptor's text and the block's units) to
`{ranges, ins, outs, lift}`, each port `[gate, width]`. -/
def piece (j : Json) : Except String Json := do
  let some d := (j.getObjValAs? String "descriptor").toOption | throw "descriptor is not a string"
  let some us := (j.getObjValAs? (Array Nat) "units").toOption | throw "units is not a list of naturals"
  let P ← decode d.toUTF8
  let f ← front P us
  let ports := fun (ps : Array Port) => Json.arr (ps.map fun p => Json.arr #[toJson p.gate, toJson p.width])
  return Json.mkObj [("ranges", .arr (f.ranges.map fun (lo, hi) => Json.arr #[toJson lo, toJson hi])),
    ("ins", ports f.ins), ("outs", ports f.outs),
    ("lift", .arr (f.lift.map fun (k, b) => Json.arr #[.str k, .str b]))]

end Spot.BlockFront
