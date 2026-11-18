module

public import Flock.TemplateQuery
public import Spot.Bits
public import Spot.Wires
meta import Flock.Program
meta import Flock.TemplateQuery
meta import Spot.Bits
meta import Spot.Wires
meta import Std.Data.HashMap

/-!
# The wires a Program's root states

`edgesOf` reads the wiring off the Program itself: each argument part of a root node that reads an earlier non-input
root node's outputs is one `Spot.Wires.Wire` per row of the reader's port, in closed form. Units are numbered as
`UnitAt` numbers them: a root node's members after its template's units in the earlier ranges. A part all of whose
lanes read input nodes (a shared weight, however its members step over it) states no wire.

The rule is by row. A reader row (the port whole, or one of the `k` rows the wiring cuts it into: leaves
`[j·n/k, (j+1)·n/k)` of a member's `n`) lies in one part of its argument, adjacent affine parts that continue one
another read as one (`merged`), and reads one whole return field `f` of one member's output, the writer unit's, at its
bits. The wire reads its writer's one output row `out` (the unit's outputs in order, each its bits in u16 words) from
the field's first word (`fieldWord`). A part is a grid (`partLanes`): a row lies in one lane's run, and across the members
a part covers its place moves by the argument's member stride, inside that run or by whole periods of the lanes; the
writer leaves the members read are then a mixed radix of the lane's levels (`laneWire`), each step whole writer units
and a step of the field's place inside its unit, whose first word then steps by a word step a digit (`ostep`).

The refusals, by name: `edge-template` (a non-input root node whose template the ranges or the wiring don't list),
`edge-form` (a scan at the root, as reader or writer), `not-topological` (a part reads a later node, or the node
itself), `edge-rows` (a reader row across a part's edge or not in closed form: a row across runs or not contiguous,
members whose rows are no mixed radix of the lane's levels, a row of an explicit part mixing input-node leaves with
another node's or not that node's consecutive leaves, a reference to a parameter), `edge-units` (a reader row whose
leaves lie in two writer units: across a unit's end, `unitsCross`), `edge-run` (a row inside one unit that is not one
whole return field: it starts at no field's first leaf, or ends at no field's last, or runs over several, `runAt`)
and `edge-width` (a reader row's bits are not the field's, or a field before it is not whole u16 words). Each names
where it arose: the node, and for a part's refusal its argument and part.

The work is the root's nodes and arguments, then each part times its port's rows, and an explicit part's references:
nothing per unit or per block beyond the description.
-/

@[expose] public section

namespace Spot.Edges

open Flock.Program (Prog Node Arg Part Ref Ty Defn fieldsOf isInputId)
open Flock.TemplateQuery (TRange)
open Spot.Bits (bitsOf)

/-- Each ranged root node's template and the template's units in the earlier ranges, one pass over the ranges. -/
def firstUnits (rs : Array TRange) : Std.HashMap Nat (String × Nat) :=
  (rs.foldl (fun (acc : Std.HashMap Nat (String × Nat) × Std.HashMap String Nat) r =>
    let b := acc.2.getD r.template 0
    (acc.1.insert r.node (r.template, b), acc.2.insert r.template (b + r.count))) ({}, {})).1

/-- The parts of `s` with their first references and indices, adjacent affine parts over one space whose second
continues the first read as one (at the first's index): the same references, in one pass over the parts. -/
def merged (s : Flock.Program.Seq) : List (Part × Nat × Nat) :=
  ((List.range s.parts.size).foldl (fun (acc : List (Part × Nat × Nat)) qi =>
    let q := s.parts[qi]!
    match acc, q with
    | (.affine p i b c, s0, q0) :: rest, .affine p' i' b' c' =>
      if p == p' && i == i' && 0 < c && 0 < c' && b' == b + c then (.affine p i b (c + c'), s0, q0) :: rest
      else (q, s.starts[qi]!, qi) :: acc
    | _, _ => (q, s.starts[qi]!, qi) :: acc) []).reverse

/-- The first leaf of each return field of `t`, and the leaves past the last. -/
def fieldStarts (t : Ty) : List Nat :=
  ((fieldsOf t).foldl (fun (acc : List Nat × Nat) ty => (acc.2 :: acc.1, acc.2 + ty.leaves)) ([], 0)).1.reverse ++
    [t.leaves]

/-- The return field `f` of `t` whose leaves are exactly `[o, o + len)`; none unless the run is one whole field. -/
def runAt (t : Ty) (o len : Nat) : Option Nat :=
  let st := fieldStarts t
  (List.range (fieldsOf t).length).find? fun f => 0 < len && st[f]! == o && st[f + 1]! == o + len

/-- The first leaf of return field `f` of `t`. -/
def fieldOffset (t : Ty) (f : Nat) : Nat := (((fieldsOf t).take f).map Ty.leaves).sum

/-- The first u16 word of return field `f` of `t` in its unit's output row `out`, the fields before it whole words: a
row is its leaves in order, each its bits in u16 words low first, padded only at its end (`Registered.rowWords`). -/
def fieldWord (t : Ty) (f : Nat) : Nat := (((fieldsOf t).take f).map fun ty => bitsOf ty / 16).sum

/-- Return field `f` of `t` at leaf `p` of its unit, of `len` leaves and `bits` bits, the fields before it whole u16
words; none otherwise. -/
def fieldAt (t : Ty) (len bits : Nat) (p : Int) : Option Nat :=
  if p < 0 then none else
  (runAt t p.toNat len).filter fun f =>
    bitsOf (fieldsOf t)[f]! == bits && ((fieldsOf t).take f).all (bitsOf · % 16 == 0)

/-- The word step of each level of `lv` (its radix and its leaf step inside the unit), the members starting at a field
at leaf `p0`: every leaf they reach is a field `fieldAt` admits, and its first word is `p0`'s plus a word step a level
times its digit; none otherwise. The leaves reached are at most the fields, so the work is the levels times the fields
squared, the Definition's description. -/
def wordSteps (t : Ty) (len bits p0 : Nat) (lv : List (Nat × Int)) : Option (List Int) := do
  let f0 ← fieldAt t len bits p0
  let n := (fieldsOf t).length
  let mut ps : List (Int × Int) := [(p0, fieldWord t f0)]
  let mut ws : List Int := []
  for (r, o) in lv do
    if r ≤ 1 || o == 0 then
      ws := ws ++ [0]
      continue
    if r > n then none
    let f1 ← fieldAt t len bits ((p0 : Int) + o)
    let ow : Int := (fieldWord t f1 : Int) - fieldWord t f0
    let mut next := ps
    for (p, w) in ps do
      for d in [1:r] do
        let f ← fieldAt t len bits (p + (d : Int) * o)
        if (fieldWord t f : Int) != w + (d : Int) * ow then none
        if !next.any (·.1 == p + (d : Int) * o) then next := next ++ [(p + (d : Int) * o, w + (d : Int) * ow)]
    if next.length > n then none
    ps := next
    ws := ws ++ [ow]
  return ws

/-- The members `[lo, lo + n)` of `M` whose row, at argument index `x0 + m·L` and `len` long, lies in the part at
`[s0, s0 + c)`; `edge-rows` where some member's row meets the part without lying in it. -/
def membersIn (x0 L len M s0 c : Nat) : Except String (Nat × Nat) :=
  if L == 0 then
    let over := x0 < s0 + c && s0 < x0 + len
    let inside := s0 ≤ x0 && x0 + len ≤ s0 + c
    if over && !inside then .error "edge-rows" else .ok (0, if inside then M else 0)
  else
    let fl := fun (a : Int) => a / (L : Int)
    let x : Int := x0
    let hi : Int := (M : Int) - 1
    let clo := max (-(fl (x - s0))) 0
    let chi := min (fl ((s0 : Int) + c - len - x)) hi
    let olo := max (fl ((s0 : Int) - len - x) + 1) 0
    let ohi := min (fl ((s0 : Int) + c - 1 - x)) hi
    let cn := if clo ≤ chi then chi - clo + 1 else 0
    let on := if olo ≤ ohi then ohi - olo + 1 else 0
    if cn != on then .error "edge-rows" else .ok (clo.toNat, cn.toNat)

/-- A lane of a part (`circuits/PROTOCOL.md` §3): its collection `(p, i)`, base `b`, inner stride `e` and levels
`(radix, step)`, innermost first. -/
structure Lane where
  p : Bool
  i : Nat
  b : Int
  e : Int
  lv : List (Nat × Int)
  deriving Inhabited

/-- A part's run and lanes, the grid it spells; an explicit part none. -/
def partLanes : Part → Nat × List Lane
  | .affine p i b c => (max c 1, [⟨p, i, b, 1, []⟩])
  | .strided p i b inner os is c =>
    let T := max 1 ((c + inner - 1) / inner)
    (inner, [⟨p, i, b, is, if T > 1 then [(T, os)] else []⟩])
  | .explicit _ => (0, [])

/-- The digits of period `t` in levels `lv`, innermost first, the last unbounded. -/
def digitsOf (t : Nat) : List (Nat × Int) → List Nat
  | [] => []
  | [_] => [t]
  | (r, _) :: rest => t % r :: digitsOf (t / r) rest

/-- The levels from `a` on whose radices the members' count `n` and their first period `tt` (over those levels) are
multiples of, as far as they go: the first level not taken and the product of those taken. -/
def absorb (lv : List (Nat × Int)) (a tt n : Nat) : Nat × Nat := Id.run do
  let mut h := a
  let mut P := 1
  for _ in [0:lv.length] do
    if h < lv.length && tt % (P * lv[h]!.1) == 0 && n % (P * lv[h]!.1) == 0 then
      P := P * lv[h]!.1
      h := h + 1
  return (h, P)

/-- The members' radix and steps over levels `lv` (innermost first, in leaves), the members `n` stepping by level `a`
from first index `tt` (in level `a`'s units) at digits `ds`: the levels from `a` they fill, and at most one partial
level above, outermost first; `edge-rows` where they are no such mixed radix. -/
def radixOf (lv : List (Nat × Int)) (ds : List Nat) (a tt n : Nat) : Except String (List Nat × List Int) := do
  let (h, P) := absorb lv a tt n
  let c := n / P
  let inner := ((List.range (h - a)).map fun x => lv[a + x]!).reverse
  let levels ← if c == 1 then pure inner
    else if h < lv.length && ds[h]! + c ≤ lv[h]!.1 then pure ((c, lv[h]!.2) :: inner)
    else throw "edge-rows"
  if levels.isEmpty then return ([1], [0])
  return (levels.map (·.1), levels.map (·.2))

/-- The rows `k0 + m·L` (`m < n`) of the grid `(run, lanes)`, each `len` references inside one lane's run: the lane's
collection, its first leaf, and the members' radix and steps in leaves, outermost first; `edge-rows` where a row
leaves a run or the members' rows are no mixed radix of the lane's levels. Members stepping inside a one-lane run
whose length `L` divides read the run as two levels, `(L, e)` and `(run / L, L·e)`, under the lane's own. -/
def laneWire (run : Nat) (lanes : List Lane) (k0 L len n : Nat) : Except String (Nat × Nat × List Nat × List Int) := do
  let g := lanes.length
  if run == 0 || g == 0 then throw "edge-rows"
  let t0 := k0 / (g * run)
  let j := k0 % (g * run) / run
  let o0 := k0 % (g * run) % run
  let ln := lanes[j]!
  if ln.p || o0 + len > run || !(len == 1 || ln.e == 1) then throw "edge-rows"
  let ds := digitsOf t0 ln.lv
  let leaf : Int := ln.b + (o0 : Int) * ln.e + ((ds.zip ln.lv).map fun (d, _, st) => (d : Int) * st).sum
  if leaf < 0 then throw "edge-rows"
  if n ≤ 1 || L == 0 then return (ln.i, leaf.toNat, [max n 1], [0])
  if L % (g * run) != 0 then
    if g == 1 && o0 + (n - 1) * L + len ≤ run then return (ln.i, leaf.toNat, [n], [(L : Int) * ln.e])
    if g != 1 || run % L != 0 || o0 % L + len > L then throw "edge-rows"
    let lv := (if L > 1 then [(L, ln.e)] else []) ++ (run / L, (L : Int) * ln.e) :: ln.lv
    let (rs, ss) ← radixOf lv (digitsOf k0 lv) (if L > 1 then 1 else 0) (k0 / L) n
    return (ln.i, leaf.toNat, rs, ss)
  let dt := L / (g * run)
  let m := ln.lv.length
  let pre := ln.lv.foldl (fun (acc : Array Nat) (r, _) => acc.push (acc.back! * r)) #[1]
  if m > 0 && dt % pre[m - 1]! == 0 then
    return (ln.i, leaf.toNat, [n], [((dt / pre[m - 1]! : Nat) : Int) * ln.lv[m - 1]!.2])
  let some a := (List.range m).find? (pre[·]! == dt) | throw "edge-rows"
  let (rs, ss) ← radixOf ln.lv ds a (t0 / dt) n
  return (ln.i, leaf.toNat, rs, ss)

/-- Reference `k` of the grid `(run, lanes)`: its lane's collection and its leaf. -/
def leafAt (run : Nat) (lanes : List Lane) (k : Nat) : Option (Bool × Nat × Int) :=
  let g := lanes.length
  if run == 0 || g == 0 then none else
  let ln := lanes[k % (g * run) / run]!
  let ds := digitsOf (k / (g * run)) ln.lv
  some (ln.p, ln.i, ln.b + ((k % (g * run) % run : Nat) : Int) * ln.e +
    ((ds.zip ln.lv).map fun (d, _, st) => (d : Int) * st).sum)

/-- Whether the row of `len` references at `k0` of the grid `(run, lanes)` (`c` references) reads two units of one
writer node, at its first and its last reference. -/
def unitsCross (P : Prog) (run : Nat) (lanes : List Lane) (c k0 len : Nat) : Bool :=
  let nodes := (P.defs[P.root]!).body.nodes
  match leafAt run lanes k0, leafAt run lanes (k0 + len - 1) with
  | some (false, w, l0), some (false, w', l) =>
    if l0 < 0 || l < 0 || w ≥ nodes.size || w' != w || k0 + len - 1 ≥ c then false else
    let LW := (P.defs[nodes[w]!.fn]!).outLeaves
    LW != 0 && l0.toNat / LW != l.toNat / LW
  | _, _ => false

/-- Whether root node `k` calls an input. -/
def isInputNode (P : Prog) (k : Nat) : Bool :=
  let f := P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!
  f.prim && isInputId f.key

/-- The wire of row `j` (of `cuts`) of template `rt`'s port `port` (of type `ty`) at reader units from `start`, reading
root node `w`'s leaves from `l0`, the members' radix and steps in leaves; none where `w` is an input node. Each step is
whole writer units and a leaf step inside the unit (rounded toward zero), the field's word stepping with its digit
(`wordSteps`, the wire's `ostep`). -/
def wireOf (P : Prog) (fu : Std.HashMap Nat (String × Nat)) (W : Spot.Wires.Wiring) (k : Nat) (rt port : String)
    (ty : Ty) (cuts j start w l0 : Nat) (radix : List Nat) (steps : List Int) :
    Except String (List Spot.Wires.Wire) := do
  if radix.length != steps.length || radix.any (· == 0) then throw "edge-rows"
  if w ≥ k then throw "not-topological"
  if isInputNode P w then return []
  let wnode := (P.defs[P.root]!).body.nodes[w]!
  if wnode.form == .scan then throw "edge-form"
  let some (wt, wb) := fu[w]? | throw "edge-template"
  if (W.templates.lookup rt).isNone || (W.templates.lookup wt).isNone then throw "edge-template"
  let wd := P.defs[wnode.fn]!
  let LW := wd.outLeaves
  let len := ty.leaves / cuts
  if LW == 0 then throw "edge-rows"
  if l0 % LW + len > LW then throw "edge-units"
  let some f := runAt wd.ret (l0 % LW) len | throw "edge-run"
  if bitsOf (fieldsOf wd.ret)[f]! != bitsOf ty / cuts || ((fieldsOf wd.ret).take f).any (bitsOf · % 16 != 0) then
    throw "edge-width"
  let some ws := wordSteps wd.ret len (bitsOf ty / cuts) (l0 % LW)
    (radix.zip (steps.map fun s => s - s.tdiv LW * LW)) | throw "edge-run"
  let wire : Spot.Wires.Wire :=
    { reader := rt, port := port, writer := wt, out := "out", start := start, radix := radix,
      base := ((wb + l0 / LW : Nat) : Int), step := steps.map fun s => s.tdiv LW, cut0 := j, cuts := 1,
      offset := fieldWord wd.ret f, ostep := if ws.all (· == 0) then [] else ws }
  return [wire]

/-- The wires of an explicit part's member rows `rows` (row `j` of template `rt`'s port, at reader units from `start`),
one by one, its references being the description: a row of input-node leaves is no wire, one of an earlier node's
consecutive leaves a wire of one unit; `edge-rows` for a row mixing them or out of order. -/
def explicitWires (P : Prog) (fu : Std.HashMap Nat (String × Nat)) (W : Spot.Wires.Wiring) (k : Nat)
    (rt port : String) (ty : Ty) (cuts j start : Nat) (rows : List (List Ref)) :
    Except String (List Spot.Wires.Wire) := do
  let mut out : List Spot.Wires.Wire := []
  for m in [0:rows.length] do
    let row := rows[m]!
    if row.any fun r => !r.p && r.i ≥ k then throw "not-topological"
    if row.all fun r => !r.p && isInputNode P r.i then continue
    let some r0 := row.head? | continue
    if r0.p || row.zipIdx.any (fun (r, t) => r.p || r.i != r0.i || r.j != r0.j + t) then throw "edge-rows"
    out := out ++ (← wireOf P fu W k rt port ty cuts j (start + m) r0.i r0.j [1] [0])
  return out

/-- The wires of the reads of argument `i`'s part `q` (at `[s0, s0 + c)` of the argument) by root node `k`'s members,
one per row of the port: the module docstring's rule, or its refusal. An explicit part's rows are `explicitWires`. -/
def partWires (P : Prog) (fu : Std.HashMap Nat (String × Nat)) (W : Spot.Wires.Wiring) (k i : Nat) (q : Part)
    (s0 : Nat) : Except String (List Spot.Wires.Wire) := do
  let nodes := (P.defs[P.root]!).body.nodes
  let node := nodes[k]!
  let some (rt, rb) := fu[k]? | throw "edge-template"
  let lanes := (partLanes q).2
  if !lanes.isEmpty && lanes.all (fun ln => !ln.p && isInputNode P ln.i) then return []
  let (port, ty) := (P.defs[node.fn]!).params[i]!
  let a := node.args[i]!
  let cuts := W.cutsOf rt port
  let n := ty.leaves
  if cuts == 0 || n % cuts != 0 || bitsOf ty % cuts != 0 then throw "edge-rows"
  let len := n / cuts
  let M := node.members
  let mut out : List Spot.Wires.Wire := []
  for j in [0:cuts] do
    let t0 := j * len
    let (x0, L) ← match a.dims with
      | none => pure (t0, 0)
      | some (d, Lb) => do
        if Lb == 0 || t0 % Lb + len > Lb then throw "edge-rows"
        pure ((t0 / Lb * d) * Lb + t0 % Lb, Lb)
    let (lo, cnt) ← membersIn x0 L len M s0 q.count
    if cnt == 0 then continue
    if let .explicit it := q then
      let rows := (List.range cnt).map fun m =>
        (it.extract (x0 + (lo + m) * L - s0) (x0 + (lo + m) * L - s0 + len)).toList
      out := out ++ (← explicitWires P fu W k rt port ty cuts j (rb + lo) rows)
      continue
    let (run, lanes) := partLanes q
    let k0 := x0 + lo * L - s0
    let (w, l0, radix, steps) ← match laneWire run lanes k0 L len cnt with
      | .ok v => pure v
      | .error e => throw (if e == "edge-rows" && unitsCross P run lanes q.count k0 len then "edge-units" else e)
    out := out ++ (← wireOf P fu W k rt port ty cuts j (rb + lo) w l0 radix steps)
  return out

/-- **The wires the Program's root states**, under the ranges `rs` and the record's wiring `W` (its templates and
cuts): the module docstring's rule, in the root's node, argument and part order, or the first refusal, at the node
(`edge-template`, `edge-form` of the reader) or at its argument and part (`<code> at node k argument i part q`). -/
def edgesOf (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) : Except String (List Spot.Wires.Wire) := do
  let fu := firstUnits rs
  let nodes := (P.defs[P.root]!).body.nodes
  let mut out : List Spot.Wires.Wire := []
  for k in [0:nodes.size] do
    if isInputNode P k then continue
    let node := nodes[k]!
    let some (rt, _) := fu[k]? | throw s!"edge-template at node {k}"
    if (W.templates.lookup rt).isNone then throw s!"edge-template at node {k}"
    if node.form == .scan then throw s!"edge-form at node {k}"
    for i in [0:node.args.size] do
      for (q, s0, qi) in merged node.args[i]!.seq do
        match partWires P fu W k i q s0 with
        | .ok ws => out := out ++ ws
        | .error e => throw s!"{e} at node {k} argument {i} part {qi}"
  return out

/-! ## Examples: the A, B, A layout -/

private def v16 : Ty := .value 16
private def row2 : Ty := .array 2 v16

private def prim (key : String) (ret : Ty) : Defn :=
  { key := key, id := key, prim := true, params := #[("x", row2)], ret := ret, gates := 1, paramLeaves := 2,
    outLeaves := ret.leaves }

private def batch (fn n : Nat) (parts : Array Part) (out : Ty) : Node :=
  { form := .batch, fn := fn, out := .array n out, n := n,
    args := #[{ ty := .array n row2, seq := Flock.Program.Seq.ofParts parts, dims := some (n, 2) }] }

/-- Inputs, then A (2 members) reading them, B (2) reading A's outputs by `bParts`, and A (2) reading B's. -/
private def toy (aRet : Ty) (bParts : Array Part) : Prog :=
  let input : Defn :=
    { key := "Input16_v1", id := "Input16_v1", prim := true, params := #[], ret := v16, gates := 1, paramLeaves := 0,
      outLeaves := 1 }
  let nodes := #[{ form := .batch, fn := 0, args := #[], out := .array 4 v16, n := 4 },
    batch 1 2 #[.affine false 0 0 4] aRet, batch 2 2 bParts row2, batch 1 2 #[.affine false 2 0 4] aRet]
  let root : Defn :=
    { key := "R", id := "R", prim := false, params := #[], ret := row2, gates := 0, paramLeaves := 0, outLeaves := 2,
      body := { params := #[], retType := row2, nodes := nodes, ret := default, offsets := #[] } }
  { defs := #[input, prim "A" aRet, prim "B" row2, root], index := {}, root := 3, programFn := "", sha512 := .empty }

private def toyRanges : Array TRange :=
  #[⟨1, "A", 0, 2⟩, ⟨2, "B", 2, 2⟩, ⟨3, "A", 4, 2⟩]

private def toyWiring (cuts : List (String × List (String × Nat))) : Spot.Wires.Wiring :=
  let t (n : Nat) : Spot.Wires.Template := { units := n, rows := [("x", ⟨2, 2⟩)], outs := [("out", 2)] }
  { templates := [("A", t 4), ("B", t 2)], cuts := cuts, wires := [] }

private def wire (r w : String) (offset start : Nat) (radix : List Nat) (base : Int) (step : List Int) (cut0 : Nat) :
    Spot.Wires.Wire :=
  { reader := r, port := "x", writer := w, out := "out", start := start, radix := radix, base := base, step := step,
    cut0 := cut0, cuts := 1, offset := offset }

private def gives (P : Prog) (W : Spot.Wires.Wiring) (ws : List Spot.Wires.Wire) : Bool :=
  match edgesOf P toyRanges W with
  | .ok ws' => ws' == ws
  | .error _ => false

private def refuses (P : Prog) (W : Spot.Wires.Wiring) (code : String) : Bool :=
  match edgesOf P toyRanges W with
  | .ok _ => false
  | .error e => e.startsWith (code ++ " at ")

#guard gives (toy row2 #[.affine false 1 0 4]) (toyWiring [])
  [wire "B" "A" 0 0 [2] 0 [1] 0, wire "A" "B" 0 2 [2] 0 [1] 0]
#guard refuses (toy row2 #[.affine false 1 1 4]) (toyWiring []) "edge-units"
#guard refuses (toy (.tuple #[row2, row2]) #[.affine false 1 1 4]) (toyWiring []) "edge-run"
#guard gives (toy (.tuple #[v16, v16]) #[.affine false 1 0 4]) (toyWiring [("B", [("x", 2)])])
  [wire "B" "A" 0 0 [2] 0 [1] 0, wire "B" "A" 1 0 [2] 0 [1] 1, wire "A" "B" 0 2 [2] 0 [1] 0]
#guard refuses (toy (.tuple #[v16, v16]) #[.affine false 1 0 4]) (toyWiring []) "edge-run"
#guard gives (toy row2 #[.affine false 1 2 2, .affine false 1 0 2]) (toyWiring [])
  [wire "B" "A" 0 0 [1] 1 [0] 0, wire "B" "A" 0 1 [1] 0 [0] 0, wire "A" "B" 0 2 [2] 0 [1] 0]
#guard gives (toy row2 #[.affine false 1 0 2, .affine false 1 2 1, .affine false 1 3 1]) (toyWiring [])
  [wire "B" "A" 0 0 [2] 0 [1] 0, wire "A" "B" 0 2 [2] 0 [1] 0]
#guard refuses (toy row2 #[.affine false 1 0 1, .affine false 1 2 1, .affine false 1 1 1, .affine false 1 3 1])
  (toyWiring []) "edge-rows"
#guard refuses (toy row2 #[.affine false 3 0 4]) (toyWiring []) "not-topological"
#guard gives (toy row2 #[.explicit #[⟨false, 1, 0⟩, ⟨false, 1, 1⟩, ⟨false, 1, 2⟩, ⟨false, 1, 3⟩]]) (toyWiring [])
  [wire "B" "A" 0 0 [1] 0 [0] 0, wire "B" "A" 0 1 [1] 1 [0] 0, wire "A" "B" 0 2 [2] 0 [1] 0]
#guard refuses (toy row2 #[.explicit #[⟨false, 1, 1⟩, ⟨false, 1, 0⟩, ⟨false, 1, 2⟩, ⟨false, 1, 3⟩]]) (toyWiring [])
  "edge-rows"
#guard refuses (toy (.tuple #[.value 8, .value 24]) #[.affine false 1 0 4]) (toyWiring [("B", [("x", 2)])])
  "edge-width"

end Spot.Edges
