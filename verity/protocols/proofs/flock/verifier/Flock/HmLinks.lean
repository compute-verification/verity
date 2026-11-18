module

public import Flock.Statement
public import Flock.TemplateQuery
public import Flock.Extract
public import Flock.Qcall
public import Flock.RecSparse
public import Flock.RecOpening
public import Flock.SplitNL
public import Flock.Norm
public import Flock.LinkUnits
public import Flock.FV3
public import Flock.HmOutCols
public import Flock.TmplSection

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Flock

open Lean (Json)

/-! `HmOut` Linking: under a template query, with the verifier's own program, every cut wire between two units of
the statement is a check on the program. A reader whose parameter leaves are another unit's outputs reads them in a link
row (META `links`): an input row encoded as the one output row is, which the registered reads read by path in its writer's
output field's tree (`Registered.checkLinked`, a wired port's positions `Registered.Port.readAt`'s closed form), so the
reader's row is the writer's committed row as one commitment, not a second commitment compared with it. The wiring is
read from the verifier's own program, so in this model the topology is public, a known leak that later work removes. -/
namespace HmLinks

/-- META `links`: each `{port, leaves: [lo, hi]}`, an input row by name that holds template parameter leaves `[lo, hi)` as
the one output row holds outputs, a `hm96-sha512/row/v1` row of that row's words (neither a bit row nor segmented); a row
once, the ranges disjoint, at least one. None without the key. -/
def parse (c : Circuit) : V (Array (Nat × Nat × Nat)) := do
  let some lj := (c.metaJ.getObjVal? "links").toOption | return #[]
  let xs ← J.asArr lj "META links"
  let k := HmOut.nIn c
  let out := c.ports.getD k default
  if c.outRowPorts.size != 1 || out.v2 || out.segs.isSome then
    throw "META links: a statement with links has one output row, a hm96-sha512/row/v1 row"
  if xs.isEmpty then throw "META links: an empty list (a statement without links has no key)"
  let mut r := #[]
  for x in xs do
    let keys := match x with
      | .obj kvs => (kvs.toArray.map (·.1)).qsort (· < ·)
      | _ => #[]
    if keys != #["leaves", "port"] then throw "META links: a link is exactly {port, leaves}"
    let name ← J.str x "port" "META links"
    let lv ← (← J.arr x "leaves" "META links").mapM (J.asNat · "a link leaf")
    let some p := (List.range k).find? (fun q => (c.ports.getD q default).name == name)
      | throw s!"META links: {name} is not an input row"
    let pt := c.ports.getD p default
    if lv.size != 2 || lv[0]! ≥ lv[1]! then throw s!"META links: {name}'s leaves are not [lo, hi) with lo < hi"
    if pt.v2 || pt.segs.isSome || pt.words != out.words || lv[1]! - lv[0]! > out.words then
      throw s!"META links: {name} is not a hm96-sha512/row/v1 row of the output row's {out.words} words"
    if r.any (fun (q, a, b) => q == p || (a < lv[1]! && lv[0]! < b)) then
      throw s!"META links: {name} is linked twice, or its leaves overlap another link's"
    r := r.push (p, lv[0]!, lv[1]!)
  return r

/-- The template query's population over the program (`TemplateQuery.evaluate`'s order), by node: per root node that is not
an input node, its index, members, first gate and gates per member. Unit `u` is counted across them in order, member `m` of
node `k` holding gates `[offset + m · gates, offset + (m + 1) · gates)`; `unitAt` and `ownerAt` read a unit from this list,
never from its members unrolled (`LinkUnits`). -/
def nodes (P : Program.Prog) : List NodeUnits :=
  let body := (P.defs[P.root]!).body
  (List.range body.nodes.size).filterMap fun k =>
    let node := body.nodes[k]!
    let f := P.defs[node.fn]!
    if f.prim && Program.isInputId f.key then none else some (k, node.members, body.offsets[k]!, f.gates)

/-- Member `m` of root node `k`: its output gates, in order. -/
def outGates (P : Program.Prog) (k m : Nat) : V (Array Nat) := do
  let node := (P.defs[P.root]!).body.nodes[k]!
  let L := (P.defs[node.fn]!).outLeaves
  (Array.range L).mapM fun j => Program.resolve P P.rootScope ⟨false, k, if node.form == .batch then m * L + j else j⟩

/-- The unit whose output reader unit `u`'s parameter leaf `t` reads (`ownerOf`), none where the leaf is a circuit input. -/
def writerOf (P : Program.Prog) (ns : List NodeUnits) (u t : Nat) : V (Option Nat) := do
  let some (k, m) := unitAt ns u | throw s!"links: unit {u} is outside the program's {count ns} units"
  let ops ← Program.operands P P.rootScope ((P.defs[P.root]!).body.nodes[k]!) m
  pure ((ops[t]?).bind (ownerAt ns))

/-- Unit `v`'s template (its node's callee) and its leaf in each of that template's output field trees: its rank among
the template's units in the root body's order, the members of the earlier nodes of that callee and then its own. -/
def treeLeaf (P : Program.Prog) (ns : List NodeUnits) (v : Nat) : Option (String × Nat) := do
  let (k, m) ← unitAt ns v
  let body := (P.defs[P.root]!).body
  let fn := body.nodes[k]!.fn
  let before := ns.foldl (fun n (k', mm, _, _) => if k' < k && body.nodes[k']!.fn == fn then n + mm else n) 0
  pure ((P.defs[fn]!).key, before + m)

/-- Linking for the population's reader `i` (its place in the public file), with `unitOf` a unit's node and member and
`ownerOf` the unit whose gates hold a gate (of `n` units): every parameter leaf of the unit that reads another unit's output
lies in a link of `links`; each link's leaves are either all circuit inputs (the row is an ordinary input row) or exactly
one unit's outputs in order, which its link row reads by path in the writer's output field's tree (`Registered.checkLinked`),
whether or not the writer is a unit of the file. The cut wires checked. -/
def checkUnit (c : Circuit) (P : Program.Prog) (links : Array (Nat × Nat × Nat)) (popIdx : Array Nat)
    (unitOf : Nat → Option (Nat × Nat)) (ownerOf : Nat → Option Nat) (n i : Nat) : V Nat := do
  let some u := popIdx[i]? | throw s!"links: unit {i} of the draw is outside the public file's {popIdx.size} units"
  let some (k, m) := unitOf u | throw s!"links: unit {u} is outside the program's {n} units"
  let ops ← Program.operands P P.rootScope ((P.defs[P.root]!).body.nodes[k]!) m
  for t in [0:ops.size] do
    if let some v := ownerOf ops[t]! then
      if !links.any (fun (_, lo, hi) => lo ≤ t && t < hi) then
        throw s!"links: cut wire: unit {u}'s parameter leaf {t} reads unit {v}'s output (gate {ops[t]!}), and no link row (META links) carries it"
  let mut wires := 0
  for (p, lo, hi) in links do
    let name := (c.ports.getD p default).name
    if hi > ops.size then throw s!"links: {name}'s leaves [{lo}, {hi}) are past unit {u}'s {ops.size} parameter leaves"
    let rd := ops.extract lo hi
    match ownerOf rd[0]! with
    | none =>
      if rd.any (ownerOf · |>.isSome) then
        throw s!"links: unit {u}'s leaves [{lo}, {hi}) (row {name}) read both circuit inputs and a unit's outputs"
    | some v =>
      let (kv, mv) := (unitOf v).getD (0, 0)
      let og ← outGates P kv mv
      if og != rd then
        throw s!"links: unit {u}'s leaves [{lo}, {hi}) (row {name}) are not exactly unit {v}'s {og.size} outputs in order"
      wires := wires + 1
  return wires

/-- `checkUnit` for each reader of `readers` (places in the public file; every unit of it when none), summed. -/
def checkAt (c : Circuit) (pop : Public) (P : Program.Prog) (links : Array (Nat × Nat × Nat))
    (unitOf : Nat → Option (Nat × Nat)) (ownerOf : Nat → Option Nat) (n : Nat) (readers : Option (Array Nat)) : V Nat := do
  let popIdx ← (← J.arr (← J.get pop.header "units" "public file") "indices" "public file").mapM (J.asNat · "a unit index")
  let mut wires := 0
  for i in readers.getD (Array.range popIdx.size) do
    wires := wires + (← checkUnit c P links popIdx unitOf ownerOf n i)
  return wires

/-- Linking on the registered population `pop`, for each unit the sample proves (`readers`, places in the public file: the
sample's units and their closure; every unit without a sample): `checkUnit`, with each unit and each gate's unit read from the
root body's nodes (`unitAt`, `ownerAt`), so its work is the program's description and the proved units', never the
population's. A reader's link row is its writer's committed row by path whether or not the writer is sampled; an unsampled
reader's wires are its own proof's, which the sample samples like any other part of it. The cut wires checked. -/
def check (c : Circuit) (pop : Public) (readers : Option (Array Nat)) (program : ByteArray) : V Nat := do
  let links ← parse c
  let P ← Program.decode program
  let ns := nodes P
  checkAt c pop P links (unitAt ns) (ownerAt ns) (count ns) readers

/-- `check` over every unit of the population, each unit and each gate's unit looked up in the units unrolled (`unitsRef`):
the reference `check` is proved equal to on the whole population (`check_ref`); the verifier never runs it. -/
def checkRef (c : Circuit) (pop : Public) (program : ByteArray) : V Nat := do
  let links ← parse c
  let P ← Program.decode program
  let ns := nodes P
  checkAt c pop P links (unitRef ns) (ownerRef ns) (unitsRef ns).length none

theorem check_ref (c : Circuit) (pop : Public) (program : ByteArray) : check c pop none program = checkRef c pop program := by
  simp only [check, checkRef, unitAt_eq, ownerAt_eq, count_eq]

/-- The units a sample proves (places in the public file), none without a sample. -/
def readersOf (draw : Option Json) : Option (Array Nat) := draw.bind fun d => (Draw.UnitSample.ofJson d).toOption.map (·.proved)

end HmLinks

end Flock
