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
* **the output ports**: Core's `Circuit.exits S`, the gates of S that C returns or that the root reads outside S, in
  increasing C index, each once, with its width, found from the reads' descriptions;
* **the lift**: every primitive the units' Definitions reach, each Definition once; a gate, a constant bit or an input
  stays, and every other one is a word primitive (`isWordPrim`) and is replaced by the Boolean Definition C's Boolean map
  (`Prog.boolean`, PROTOCOL.md 4.9) names for it.

The input ports are the leaves the block's Definitions are bound to, so they contain Core's `Circuit.into S`, and equal it
when every parameter leaf is read inside S.

Its cost is the block's parameter leaves and its exits, each resolved down the call hierarchy, the root's reference parts,
and the Definitions it reaches, each once: the descriptor and the block's boundary, never a gate of the block's interior.
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

def inside (S : Array (Nat × Nat)) (g : Nat) : Bool := S.any fun (lo, hi) => lo ≤ g && g < hi

/-- `xs` in increasing order, each once. -/
def sortDedup (xs : Array Nat) : Array Nat :=
  (xs.qsort (· < ·)).foldl (fun acc x => if acc.back? == some x then acc else acc.push x) #[]

/-- Gates with their widths as ports, in increasing index, each once; refuses a gate met at two widths. -/
def portsOf (xs : Array (Nat × Nat)) : V (Array Port) :=
  (xs.qsort (·.1 < ·.1)).foldlM (fun acc (g, w) =>
    match acc.back? with
    | some p => if Port.gate p != g then pure (acc.push { gate := g, width := w })
                else if p.width == w then pure acc
                else throw s!"port-width: gate {g} is read or written at widths {p.width} and {w}"
    | none => pure (acc.push { gate := g, width := w })) #[]

/-- A unit's operands, each with its width: its callee's parameter leaves in order. -/
def operandsW (P : Prog) (x : Node) (m : Nat) : V (Array (Nat × Nat)) := do
  let ops ← operands P P.rootScope x m
  let ws := (P.defs[x.fn]!).params.foldl (fun a (_, t) => a ++ t.widths) #[]
  if ops.size != ws.size then throw s!"root node's member {m}: {ops.size} operands for {ws.size} parameter leaves"
  return ops.zip ws

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

/-- The root nodes of C that the units are, each `(k, first member, end member)`, in increasing C index. -/
def kept (ms : Array (Nat × Nat × Scope)) : Array (Nat × Nat × Nat) :=
  ms.foldl (fun acc (k, m, _) => match acc.back? with
    | some (k', m0, m1) => if k' == k && m1 == m then acc.pop.push (k, m0, m + 1) else acc.push (k, m, m + 1)
    | none => #[(k, m, m + 1)]) #[]

/-- Whether part `q` names one of the root's nodes `held` holds, read off its description. -/
def touches (held : Nat → Bool) : Part → Bool
  | .affine p i _ _ => !p && held i
  | .strided p i _ _ _ _ _ => !p && held i
  | .explicit it => it.any fun r => !r.p && held r.i
  | .grid _ lanes => lanes.any fun l => !l.p && held l.i

/-- The references of `s` at positions `keep` admits that name a leaf in `leaves` (each root node's held leaf ranges):
a part naming no held node is passed over by its description, an affine part's leaves are cut to the ranges by
arithmetic, and only another part naming a held node is read reference by reference. -/
def heldReads (leaves : Std.HashMap Nat (Array (Nat × Nat))) (keep : Nat → Bool) (s : Seq) : Array Ref := Id.run do
  let inS := fun (r : Ref) => !r.p && ((leaves[r.i]?).map (·.any fun (lo, hi) => lo ≤ r.j && r.j < hi)).getD false
  let mut out := #[]
  for t in [0:s.parts.size] do
    let some q := s.parts[t]? | continue
    let st := (s.starts[t]?).getD 0
    if !touches leaves.contains q then continue
    match q with
    | .affine false i b c =>
      for (lo, hi) in (leaves[i]?).getD #[] do
        for j in [max b lo:min (b + c) hi] do
          if keep (st + (j - b)) then out := out.push { p := false, i, j }
    | q =>
      for u in [0:q.count] do
        let r := q.get u
        if inS r && keep (st + u) then out := out.push r
  return out

/-- **The front of block `units`'s circuit** (the module docstring), the units in any order and each once. out(S) is
Core's `Circuit.exits S`: the gates of S that C returns or that the root reads outside S (a node S does not hold, or a
member outside S of a batch S holds in part), each found from the reads' descriptions (`heldReads`). -/
def front (P : Prog) (units : Array Nat) : V Front := do
  let some root := P.defs[P.root]? | throw "front: C's root is not one of its Definitions"
  let (members, S) ← members P units
  let mut ins := #[]
  for (k, m, _) in members do
    let some x := root.body.nodes[k]? | throw s!"front: root node {k} is past C's root's nodes"
    for (g, w) in ← operandsW P x m do
      if !inside S g then ins := ins.push (g, w)
  let ks := kept members
  let mut leaves : Std.HashMap Nat (Array (Nat × Nat)) := {}
  let mut heldOf : Std.HashMap Nat (Array (Nat × Nat)) := {}
  for (k, m0, m1) in ks do
    let some x := root.body.nodes[k]? | throw s!"front: root node {k} is past C's root's nodes"
    let some c := P.defs[x.fn]? | throw s!"front: root node {k} calls Definition {x.fn}, past C's Definitions"
    let per := c.ret.leaves
    if x.out.leaves != x.members * per then
      throw s!"front: root node {k} returns {x.out.leaves} leaves for {x.members} members of {per}"
    leaves := leaves.insert k (((leaves[k]?).getD #[]).push (m0 * per, m1 * per))
    heldOf := heldOf.insert k (((heldOf[k]?).getD #[]).push (m0, m1))
  let mut reads := heldReads leaves (fun _ => true) root.body.ret
  for k in [0:root.body.nodes.size] do
    let some x := root.body.nodes[k]? | continue
    match heldOf[k]? with
    | none => for a in x.args do reads := reads ++ heldReads leaves (fun _ => true) a.seq
    | some hs =>
      if x.form != .batch then continue
      let isHeld := fun (m : Nat) => hs.any fun (m0, m1) => m0 ≤ m && m < m1
      let nHeld := hs.foldl (fun n (m0, m1) => n + (m1 - m0)) 0
      for i in [0:x.args.size] do
        let some a := x.args[i]? | continue
        match (x.axes[i]?).join with
        | none => if nHeld < x.n then reads := reads ++ heldReads leaves (fun _ => true) a.seq
        | some 0 =>
          let L := match a.ty with | .array _ e => e.leaves | _ => 0
          if L == 0 then throw s!"front: root node {k}'s argument {i} is mapped over axis 0 of no array"
          reads := reads ++ heldReads leaves (fun t => !isHeld (t / L)) a.seq
        | some ax => throw s!"front: root node {k}'s argument {i} is mapped over axis {ax}, which a member's cut doesn't take"
  let mut outs := #[]
  for r in reads do
    let some x := root.body.nodes[r.i]? | throw s!"front: a read of root node {r.i}, past C's root's nodes"
    let some c := P.defs[x.fn]? | throw s!"front: root node {r.i} calls Definition {x.fn}, past C's Definitions"
    let rw := c.ret.widths
    let some w := rw[r.j % rw.size]? | throw s!"front: root node {r.i}'s callee returns no leaves"
    outs := outs.push (← resolve P P.rootScope r, w)
  return { ranges := S.qsort (·.1 < ·.1), ins := ← portsOf ins, outs := ← portsOf outs,
           lift := ← lift P (members.map fun (k, _, _) => ((root.body.nodes[k]?).map (·.fn)).getD 0) }

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

/-- A Definition as the `restrict` piece shows it: types, gates, and each node's and the return's
references one by one. -/
def defJson (P : Prog) (d : Defn) : Json :=
  let refs := fun (s : Seq) => Json.arr (s.toArray.map Ref.toJson)
  Json.mkObj [("key", .str d.key), ("params", .arr (d.params.map (·.2.toJson))), ("ret", d.ret.toJson),
    ("gates", toJson d.gates), ("refs", refs d.body.ret),
    ("nodes", .arr (d.body.nodes.map fun x => Json.mkObj [("form", .str x.form.name),
      ("fn", .str (P.defs[x.fn]!).key), ("out", x.out.toJson), ("n", toJson x.members),
      ("args", .arr (x.args.map fun a => Json.mkObj [("ty", a.ty.toJson), ("refs", refs a.seq),
        ("dims", match a.dims with | some (d, l) => Json.arr #[toJson d, toJson l] | none => .null)]))]))]

/-- The `block_front` piece: `{descriptor, units}` (the descriptor's text and the block's units) to
`{ranges, ins, outs, lift}`, each port `[gate, width]`. -/
def piece (j : Json) : Except String Json := do
  let some d := (j.getObjValAs? String "descriptor").toOption | throw "descriptor is not a string"
  let some us := (j.getObjValAs? (Array Nat) "units").toOption | throw "units is not a list of naturals"
  let P ← decode d.toUTF8
  let f ← front P us
  let ports := fun (ps : Array Port) => Json.arr (ps.map fun p => Json.arr #[toJson (Port.gate p), toJson p.width])
  return Json.mkObj [("ranges", .arr (f.ranges.map fun (lo, hi) => Json.arr #[toJson lo, toJson hi])),
    ("ins", ports f.ins), ("outs", ports f.outs),
    ("lift", .arr (f.lift.map fun (k, b) => Json.arr #[.str k, .str b]))]

#guard (portsOf #[(5, 16), (2, 8), (5, 16)]).toOption == some #[{ gate := 2, width := 8 }, { gate := 5, width := 16 }]
#guard match portsOf #[(5, 16), (2, 8), (5, 8)] with
  | .error e => e.startsWith "port-width: gate 5 is read or written at widths"
  | .ok _ => false

end Spot.BlockFront
