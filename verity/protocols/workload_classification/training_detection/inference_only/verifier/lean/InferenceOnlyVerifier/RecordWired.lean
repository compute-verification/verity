module

public import InferenceOnlyVerifier.Record

/-!
# The record check's wired reading: a root whose units read earlier units' outputs

`Record.read` refuses a root that hands a unit a row it cannot show to be the unit's own input gates (`own-rows`), so
a Program whose wires are edges of its root, a unit's argument reading an earlier unit's outputs, is refused there.
`readWired` is the same check reading such a root: every stage of `Record` in its order, with `own-rows` taking an
argument that is an edge (`edgesOf`), each edge counted as a wire of the record is (`finishWired`). On a root with no
edge it is `Record.read`. No verifier reads it yet: only its difftest (`RecordWiredDifftest`) does, which holds the
Python reference's wired reading to it.

## An edge

Root node `k`, no input node, is a call or a batch of `n_k` members of its template, and its argument `a` fills the
template's row `a`, the parameter of `L` leaves. An argument is an edge where a part of it reads the output of a root
node that is no input node (`readsUnit`). It is taken (`edgeOf`) where all of these hold, and refused (`edge-rows`)
otherwise:

* the argument is that one part, and node `k` a call or a batch;
* the part reads node `i`, earlier (`i < k`), a call or a batch of `n_i` members of its template, each returning `L_i`
  leaves; it starts at leaf `b = m0 · L_i + off` of node `i`'s output, and `off` is the first leaf of a field `j` of
  `L` leaves of the writer's return (`o<j>`);
* that field's leaves and the row's are of one width, the same (`Flock.Program.Widths.homo`): a conversion is a gate
  of a template, never an edge;
* a batch that maps the argument (its `dims` `(n_k, L)`) reads `n_k · L` leaves, member `m` reading writer member
  `m0 + m`: the part is `affine n i b (n_k · L)` or `strided n i b L L_i 1 (n_k · L)`, its members `L_i` leaves apart
  where `n_k > 1`; a call, or a batch that does not map it, reads `L` leaves, every member writer member `m0`'s field;
* the writer has those members: `m0 + n_k ≤ n_i` where mapped, `m0 < n_i` otherwise.

A reference to a later node is refused before this (`not-topological`), so no edge goes forward and no edges make a
cycle. Each edge is a wire in SPOT's own shape (`Spot.Wires.Wire`): reader the row's template and port, writer the
writer's template and `o<j>`, from the reader's unit 0 over `radix [n_k]`, from writer unit `m0` at `step [1]`, or `[0]`
where every member reads one row. A template is one root node (`RecordText.templatesOf`: one range a template), so a
member is its template's unit.

Every other argument is held to `own-rows`'s rule (`ownRowsBut`, `Record.ownRows` on the arguments that are no edge).

## What it counts

An edge is not one of the text's wires: SPOT's check and the record's wiring (`Case.W`, `Case.wired`) are the text's
alone. It is a declaration beside the text's (`finishWired`), wire `n + e` for edge `e` after the text's `n` wires,
whose rows are counted as a wire's are: `inside` where `Record.sameSlot` shows every reader row's writer in the
reader's slot, `crossWires` otherwise. A block's gates are unchanged, and so are its input bits: an edge adds no gate
and no row. After `Record`'s last stage, an edge into a port the text cuts is refused (`edge-rows`), and an edge onto a
row a declaration of the text fills (a wire, a registered read, a table read, an opening's port or ingress) is refused
with the declarations (`inputs`).

## Cost

The root's argument parts once, as `ownRows`; per edge one lookup of its writer's return fields; per edge and slot
tried one closed-form tally, and one `sameSlot` within `Record.FUEL`; per edge and declaration one interval test. No
unit, row or leaf is visited.
-/

@[expose] public section

namespace InferenceOnlyVerifier.RecordWired

open Lean (Json)
open Flock.Program
open Spot.RecordText
open InferenceOnlyVerifier.Record

/-- Whether node `i` of the root's `nodes` is a unit's: a root node that is no input node. -/
def isUnit (P : Prog) (nodes : Array Node) (i : Nat) : Bool :=
  (nodes[i]?.map fun n => !Desc.isInputNode P n).getD false

/-- Whether part `q` of a root node's argument reads a unit's output (`isUnit`): an edge's part. -/
def readsUnit (P : Prog) (nodes : Array Node) : Part → Bool
  | .affine false i _ c | .strided false i _ _ _ _ c => c != 0 && isUnit P nodes i
  | .explicit items => items.any fun r => !r.p && isUnit P nodes r.i
  | _ => false

/-- Whether argument `a` is an edge's: a part of it reads a unit's output. -/
def isEdgeArg (P : Prog) (nodes : Array Node) (a : Arg) : Bool := a.seq.parts.any (readsUnit P nodes)

/-- Whether some argument of a root node that is no input node reads a unit's output. -/
def hasEdge (P : Prog) : Bool :=
  let nodes := (P.defs[P.root]!).body.nodes
  nodes.any fun n => !Desc.isInputNode P n && n.args.any (isEdgeArg P nodes)

/-- The root nodes that are no input node, each with its template: the `k`-th of them is the `k`-th of `ts`. -/
def tmplAt (P : Prog) (ts : List Tmpl) : List (Nat × Tmpl) :=
  let nodes := (P.defs[P.root]!).body.nodes
  ((List.range nodes.size).filter fun k => !Desc.isInputNode P nodes[k]!).zip ts

/-- **An edge** (the module docstring): argument `a` of root node `k`, `n`, as the wire it is, or none where it is no
edge the rule takes. `tm` is each unit's root node with its template (`tmplAt`). -/
def edgeOf (P : Prog) (nodes : Array Node) (tm : List (Nat × Tmpl)) (k : Nat) (n : Node) (a : Nat) (arg : Arg) :
    Option Spot.Wires.Wire := do
  guard (n.form == .call || n.form == .batch)
  let #[q] := arg.seq.parts | none
  let rd ← tm.lookup k
  let (port, pty) ← (P.defs[n.fn]!).params[a]?
  let L := pty.leaves
  let nk := n.members
  let mapped := n.form == .batch && arg.dims.isSome
  guard (!mapped || arg.dims == some (nk, L))
  let c := if mapped then nk * L else L
  let (i, b, stride) ← match q with
    | .affine false i b cnt => if cnt == c then some (i, (b : Int), (L : Int)) else none
    | .strided false i b inner os is cnt => if cnt == c && inner == L && is == 1 then some (i, b, os) else none
    | _ => none
  guard (i < k)
  let w ← nodes[i]?
  guard (w.form == .call || w.form == .batch)
  let wt ← tm.lookup i
  let ret := (P.defs[w.fn]!).ret
  let Li := ret.leaves
  guard (0 ≤ b && Li != 0)
  let m0 := b.toNat / Li
  let off := b.toNat % Li
  guard (!(mapped && nk > 1) || stride == (Li : Int))
  guard (m0 + (if mapped then nk else 1) ≤ w.members)
  let fs := fieldsOf ret
  let j ← (List.range fs.length).find? fun j => ((fs.take j).map (·.leaves)).sum == off && fs[j]!.leaves == L
  guard ((Widths.homo pty).isSome && Widths.homo fs[j]! == Widths.homo pty)
  some { reader := rd.tid, port, writer := wt.tid, out := s!"o{j}", start := 0, radix := [nk], base := m0,
         step := [if mapped then 1 else 0], cut0 := 0, cuts := 1 }

/-- **The root's edges**, as wires (`edgeOf`), each unit's arguments in order; refused (`edge-rows`) where an argument
that reads a unit's output (`isEdgeArg`) is no edge the rule takes. It costs the root's argument parts. -/
def edgesOf (P : Prog) (ts : List Tmpl) : Except String (List Spot.Wires.Wire) :=
  let nodes := (P.defs[P.root]!).body.nodes
  let tm := tmplAt P ts
  tm.foldlM (init := []) fun acc (k, _) =>
    let n := nodes[k]!
    n.args.toList.zipIdx.foldlM (init := acc) fun acc (arg, a) =>
      if isEdgeArg P nodes arg then
        match edgeOf P nodes tm k n a arg with
        | some w => .ok (acc ++ [w])
        | none => .error "edge-rows"
      else .ok acc

/-- `Record.ownRows` on the arguments that are no edge's (`isEdgeArg`): each is runs of leaves of the root's input
nodes, a batch maps it over its members, and no two runs share a leaf. -/
def ownRowsBut (P : Prog) : Bool :=
  let nodes := (P.defs[P.root]!).body.nodes
  let units := nodes.toList.filter fun n => !Desc.isInputNode P n
  let own := fun (n : Node) => n.args.toList.filter fun a => !isEdgeArg P nodes a
  let parts := units.flatMap fun n => (own n).flatMap fun a => a.seq.parts.toList
  let runs := parts.filterMap fun
    | .affine false i base count =>
      if (nodes[i]?.map fun n => Desc.isInputNode P n && (n.form == .prim || n.form == .batch)).getD false then
        some (i, base, count)
      else none
    | _ => none
  let sorted := (runs.filter fun r => r.2.2 > 0).mergeSort fun a b => a.1 < b.1 || (a.1 == b.1 && a.2.1 ≤ b.2.1)
  units.all (fun n => n.form != .batch || (own n).all (·.dims.isSome)) &&
    runs.length == parts.length &&
    (sorted.zip sorted.tail).all fun (a, b) => a.1 != b.1 || a.2.1 + a.2.2 ≤ b.2.1

/-- `Record.front` with the root's edges: `front`, and no edge, where `front` does not refuse `own-rows`; where it
does, `own-rows` again unless the root has an edge, and else the stages as far as `own-rows` reread, the root's
edges (`edgesOf`, else `edge-rows`) and `own-rows` on its other arguments (`ownRowsBut`). -/
def frontWired (j tr : Json) : Except String (Front × List Spot.Wires.Wire) :=
  match front j tr with
  | .error "own-rows" => do
    let some P := (Record.parse j >>= fun i => decodeProgram i.program).toOption | throw "own-rows"
    if !hasEdge P then throw "own-rows"
    let i ← Record.parse j
    let doc ← match Flock.JsonParse.parse i.record with
      | .ok (.obj kvs) => pure (Json.obj kvs)
      | _ => throw "record"
    let T ← Theta.trustedOf tr
    let ts ← templatesOf P doc
    let es ← edgesOf P ts
    if !ownRowsBut P then throw "own-rows"
    return ({ i, P, T, doc, ts, leaves := ts.map (leavesOf P (passedOn P)) }, es)
  | r => r.map (·, [])

/-- `Record.finish`, then the edges `es` as declarations beside the text's (the module docstring's "What it counts"):
refused (`edge-rows`) where one fills a port the text cuts, and (`inputs`) where one fills a row a declaration of the
text fills. With no edge it is `finish`. -/
def finishWired (f : Front) (es : List Spot.Wires.Wire) (pin : String → Option String) : Except String Case := do
  let c ← finish f pin
  if es.any fun w => c.W.cutsOf w.reader w.port != 1 then throw "edge-rows"
  let n := c.wired.wires.length
  let ds := es.zipIdx.map fun (w, e) => ({ w, bits := 16 * rowWords f.ts c.W w, fill := .wire (n + e) } : Decl)
  if !(Spot.Wires.pairwiseDisjoint (ds.map (·.w)) &&
      ds.all fun d => c.x.decls.all fun d' => Spot.Wires.disjoint d.w d'.w) then throw "inputs"
  return { c with x := { c.x with decls := c.x.decls ++ ds },
                  inside := c.inside ++
                    (es.map fun w => sameSlot c.S (c.gridsAt w.reader) (c.gridsAt w.writer) w).toArray }

/-- **The wired reading**: `Record.read` with the root's edges (`frontWired`, `finishWired`), or the first refusal's
code. -/
def readWired (j tr : Json) : Except String Case := do
  let (f, es) ← frontWired j tr
  let desc := match Flock.JsonParse.parse f.i.program with
    | .ok d => d
    | .error _ => Json.null
  finishWired f es (pinOf desc f.i.pins)

/-- The edges `es` as a wiring's summary: its `writerOf` gives each reader row its edge's writer, its unit and its
output row. -/
def edges (es : List Spot.Wires.Wire) : Spot.Wires.Summary := { wires := es, rowWords := [] }

/-- `Record.evaluate` on the wired reading: the summary with the rate rule's verdict, or the first refusal's code. -/
def evaluate (j tr : Json) : Except String Json := do
  let c ← readWired j tr
  let s := summarize c
  let failure := rate c.f.i.gamma c.f.i.num c.f.i.den s
  return Json.mkObj (s.fields ++ [("complies", Json.bool failure.isNone), ("failure", failureJson failure)])

/-- `evaluate`, with a refusal as its code alone. -/
def checkDetail (j tr : Json) : Json :=
  match evaluate j tr with
  | .ok r => r
  | .error code => InferenceOnlyVerifier.refusal code

/-- `Record.kinds` on the wired reading. -/
def kinds (j tr rows : Json) : Json :=
  match readWired j tr, rows with
  | .error code, _ => InferenceOnlyVerifier.refusal code
  | .ok c, .arr rs => Json.arr (rs.map fun r => Json.str (match rowOf r with
    | some r => (c.rowKind r).name
    | none => Kind.free.name))
  | .ok _, _ => Json.arr #[]

end InferenceOnlyVerifier.RecordWired
