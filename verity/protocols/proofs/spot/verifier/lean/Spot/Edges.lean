module

public import Flock.TemplateQuery
public import Spot.Bits
public import Spot.Wires
public import Std.Data.HashSet
meta import Flock.Program
meta import Flock.TemplateQuery
meta import Spot.Bits
meta import Spot.Wires
meta import Std.Data.HashMap
meta import Std.Data.HashSet

/-!
# The wires a Program's root states

`edgesOf` reads the wiring off the Program itself: each argument part of a root node that reads an earlier non-input
root node's outputs is one `Spot.Wires.Wire` per row of the reader's port, in closed form. Units are numbered as
`UnitAt` numbers them: a root node's members after its template's units in the earlier ranges.

The rule is by row. A reader row (the port whole, or one of the `k` rows the wiring cuts it into: leaves
`[j·n/k, (j+1)·n/k)` of a member's `n`) lies in one part of its argument, and reads one contiguous run of one member's
output, the writer unit's: whole return fields `f … f'` of it, at their bits. The wire reads its writer's one output
row `out` (the fields' rows end to end, each whole u16 words) from the run's first word (`fieldWord`). A part is a
grid (`partLanes`): a row lies in one lane's run, and across the members a part covers its place moves by the
argument's member stride, inside that run or by whole periods of the lanes; the writer leaves the members read are then
a mixed radix of the lane's levels (`laneWire`), and a wire needs the run's place in its unit to stay the same, so each
step is a whole number of writer members.

The refusals, by name: `edge-parts` (an argument of a root node has more parts than `argPartsOk` allows),
`edge-template` (a non-input root node whose template the ranges or the wiring don't list), `edge-form` (a scan at the
root, as reader or writer), `not-topological` (a part reads a later node, or the node itself), `edge-rows` (a reader row
across a part's edge, not one whole writer field, or not in closed form: a row across runs or not contiguous, members
whose rows are no mixed radix of the lane's levels, an explicit part naming a non-input node, a reference to a
parameter), `edge-units` (a reader row whose leaves lie in two writer units: across a unit's end, `unitsCross`),
`edge-run` (a run inside one unit that does not start at a field's first leaf and end at a field's last, `runAt`) and
`edge-width` (a reader row's bits are not the run's, or a field of the run before its last is not whole u16 words).

The work is the root's nodes and arguments, then each part times its port's rows: nothing per unit or per block. The
parts are read only after `argPartsOk` has bounded them by the root's nodes.
-/

@[expose] public section

namespace Spot.Edges

open Flock.Program (Prog Node Arg Part Ty Defn fieldsOf isInputId)
open Flock.TemplateQuery (TRange)
open Spot.Bits (bitsOf)

/-- Each ranged root node's template and the template's units in the earlier ranges, one pass over the ranges. -/
def firstUnits (rs : Array TRange) : Std.HashMap Nat (String × Nat) :=
  (rs.foldl (fun (acc : Std.HashMap Nat (String × Nat) × Std.HashMap String Nat) r =>
    let b := acc.2.getD r.template 0
    (acc.1.insert r.node (r.template, b), acc.2.insert r.template (b + r.count))) ({}, {})).1

/-- The space a part reads: its node or parameter, or an explicit part's first reference's; none for an empty one. -/
def partKey : Part → Option (Bool × Nat)
  | .affine p i _ _ => some (p, i)
  | .strided p i _ _ _ _ _ => some (p, i)
  | .explicit it => it[0]?.map fun r => (r.p, r.i)

/-- Whether an argument of root node `k`, in a root of `np` parameters, has at most one part more than the spaces
(`partKey`) its parts read. A builder that folds an argument's reads of one writer node into one part, and its reads
of the inputs into a part an input node, meets it; one that emits a part per block has about as many parts in an
argument as members, and is refused. The bound is in the Program's own nodes alone: it reads neither the ranges nor
the record's wires. The parts are read only below `k + np + 2`, which no accepted argument exceeds, as distinct
spaces number at most the earlier nodes, the parameters and the empty part; an argument reading a later node may be
refused here rather than as `not-topological`. -/
def argPartsOk (k np : Nat) (s : Flock.Program.Seq) : Bool :=
  s.parts.size ≤ k + np + 2 &&
    s.parts.size ≤ (s.parts.foldl (fun (h : Std.HashSet (Option (Bool × Nat))) q => h.insert (partKey q)) {}).size
      + 1

/-- Whether every argument of every root node meets `argPartsOk`. -/
def partsOk (P : Prog) : Bool :=
  let b := (P.defs[P.root]!).body
  (List.range b.nodes.size).all fun k => (b.nodes[k]!).args.all fun a => argPartsOk k b.params.size a.seq

/-- The first leaf of each return field of `t`, and the leaves past the last. -/
def fieldStarts (t : Ty) : List Nat :=
  ((fieldsOf t).foldl (fun (acc : List Nat × Nat) ty => (acc.2 :: acc.1, acc.2 + ty.leaves)) ([], 0)).1.reverse ++
    [t.leaves]

/-- The return fields `f … f'` of `t` whose leaves are exactly `[o, o + len)`; none unless the run starts at a field's
first leaf and ends at a field's last. -/
def runAt (t : Ty) (o len : Nat) : Option (Nat × Nat) :=
  let st := fieldStarts t
  let n := (fieldsOf t).length
  match (List.range n).find? (st[·]! == o), (List.range n).find? (st[· + 1]! == o + len) with
  | some f, some f' => if 0 < len && f ≤ f' then some (f, f') else none
  | _, _ => none

/-- The first leaf of return field `f` of `t`. -/
def fieldOffset (t : Ty) (f : Nat) : Nat := (((fieldsOf t).take f).map Ty.leaves).sum

/-- The first word of return field `f` of `t` in its unit's output row `out`: the fields before it, each zero-padded to
whole u16 words. -/
def fieldWord (t : Ty) (f : Nat) : Nat := (((fieldsOf t).take f).map fun ty => (bitsOf ty + 15) / 16).sum

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

/-- The rows `k0 + m·L` (`m < n`) of the grid `(run, lanes)`, each `len` references inside one lane's run: the lane's
collection, its first leaf, and the members' radix and steps in leaves, outermost first; `edge-rows` where a row
leaves a run or the members' rows are no mixed radix of the lane's levels. -/
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
    throw "edge-rows"
  let dt := L / (g * run)
  let m := ln.lv.length
  let pre := ln.lv.foldl (fun (acc : Array Nat) (r, _) => acc.push (acc.back! * r)) #[1]
  if m > 0 && dt % pre[m - 1]! == 0 then
    return (ln.i, leaf.toNat, [n], [((dt / pre[m - 1]! : Nat) : Int) * ln.lv[m - 1]!.2])
  let some a := (List.range m).find? (pre[·]! == dt) | throw "edge-rows"
  let (h, P) := absorb ln.lv a (t0 / dt) n
  let c := n / P
  let inner := ((List.range (h - a)).map fun x => ln.lv[a + x]!).reverse
  let levels ← if c == 1 then pure inner
    else if h < m && ds[h]! + c ≤ ln.lv[h]!.1 then pure ((c, ln.lv[h]!.2) :: inner)
    else throw "edge-rows"
  if levels.isEmpty then return (ln.i, leaf.toNat, [1], [0])
  return (ln.i, leaf.toNat, levels.map (·.1), levels.map (·.2))

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

/-- The wires of the reads of argument `i`'s part `q` (at `[s0, s0 + c)` of the argument) by root node `k`'s members,
one per row of the port: the module docstring's rule, or its refusal. -/
def partWires (P : Prog) (fu : Std.HashMap Nat (String × Nat)) (W : Spot.Wires.Wiring) (k i : Nat) (q : Part)
    (s0 : Nat) : Except String (List Spot.Wires.Wire) := do
  let nodes := (P.defs[P.root]!).body.nodes
  let node := nodes[k]!
  let some (rt, rb) := fu[k]? | throw "edge-template"
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
      if it.any fun r => r.p || r.i ≥ k || !isInputNode P r.i then
        if it.any fun r => !r.p && r.i ≥ k then throw "not-topological"
        throw "edge-rows"
      continue
    let (run, lanes) := partLanes q
    let k0 := x0 + lo * L - s0
    let (w, l0, radix, steps) ← match laneWire run lanes k0 L len cnt with
      | .ok v => pure v
      | .error e => throw (if e == "edge-rows" && unitsCross P run lanes q.count k0 len then "edge-units" else e)
    if radix.length != steps.length || radix.any (· == 0) then throw "edge-rows"
    if w ≥ k then throw "not-topological"
    if isInputNode P w then continue
    let wnode := nodes[w]!
    if wnode.form == .scan then throw "edge-form"
    let some (wt, wb) := fu[w]? | throw "edge-template"
    if (W.templates.lookup rt).isNone || (W.templates.lookup wt).isNone then throw "edge-template"
    let wd := P.defs[wnode.fn]!
    let LW := wd.outLeaves
    if LW == 0 then throw "edge-rows"
    if l0 % LW + len > LW then throw "edge-units"
    let some (f, f') := runAt wd.ret (l0 % LW) len | throw "edge-run"
    let fs := ((fieldsOf wd.ret).drop f).take (f' + 1 - f)
    if (fs.map bitsOf).sum != bitsOf ty / cuts || fs.dropLast.any (bitsOf · % 16 != 0) then throw "edge-width"
    if (radix.zip steps).any (fun (r, d) => r > 1 && d % (LW : Int) != 0) then throw "edge-rows"
    let wire : Spot.Wires.Wire :=
      { reader := rt, port := port, writer := wt, out := "out", start := rb + lo, radix := radix,
        base := ((wb + l0 / LW : Nat) : Int), step := steps.map (· / (LW : Int)), cut0 := j, cuts := 1,
        offset := fieldWord wd.ret f }
    out := out ++ [wire]
  return out

/-- **The wires the Program's root states**, under the ranges `rs` and the record's wiring `W` (its templates and
cuts): the module docstring's rule, in the root's node, argument and part order, or the first refusal. -/
def edgesOf (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) : Except String (List Spot.Wires.Wire) := do
  if !partsOk P then throw "edge-parts"
  let fu := firstUnits rs
  let nodes := (P.defs[P.root]!).body.nodes
  let mut out : List Spot.Wires.Wire := []
  for k in [0:nodes.size] do
    if isInputNode P k then continue
    let node := nodes[k]!
    let some (rt, _) := fu[k]? | throw "edge-template"
    if (W.templates.lookup rt).isNone then throw "edge-template"
    if node.form == .scan then throw "edge-form"
    for i in [0:node.args.size] do
      let s := node.args[i]!.seq
      for qi in [0:s.parts.size] do
        out := out ++ (← partWires P fu W k i s.parts[qi]! s.starts[qi]!)
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
  | .error e => e == code

#guard gives (toy row2 #[.affine false 1 0 4]) (toyWiring [])
  [wire "B" "A" 0 0 [2] 0 [1] 0, wire "A" "B" 0 2 [2] 0 [1] 0]
#guard refuses (toy row2 #[.affine false 1 1 4]) (toyWiring []) "edge-units"
#guard refuses (toy (.tuple #[row2, row2]) #[.affine false 1 1 4]) (toyWiring []) "edge-run"
#guard gives (toy (.tuple #[v16, v16]) #[.affine false 1 0 4]) (toyWiring [("B", [("x", 2)])])
  [wire "B" "A" 0 0 [2] 0 [1] 0, wire "B" "A" 1 0 [2] 0 [1] 1, wire "A" "B" 0 2 [2] 0 [1] 0]
#guard gives (toy (.tuple #[v16, v16]) #[.affine false 1 0 4]) (toyWiring [])
  [wire "B" "A" 0 0 [2] 0 [1] 0, wire "A" "B" 0 2 [2] 0 [1] 0]
#guard gives (toy row2 #[.affine false 1 0 2, .affine false 1 2 2]) (toyWiring [])
  [wire "B" "A" 0 0 [1] 0 [0] 0, wire "B" "A" 0 1 [1] 1 [0] 0, wire "A" "B" 0 2 [2] 0 [1] 0]
#guard refuses (toy row2 #[.affine false 1 0 2, .affine false 1 2 1, .affine false 1 3 1]) (toyWiring [])
  "edge-parts"
#guard refuses (toy row2 #[.affine false 3 0 4]) (toyWiring []) "not-topological"
#guard refuses (toy row2 #[.explicit #[⟨false, 1, 0⟩, ⟨false, 1, 1⟩, ⟨false, 1, 2⟩, ⟨false, 1, 3⟩]]) (toyWiring [])
  "edge-rows"
#guard refuses (toy (.tuple #[.value 8, .value 24]) #[.affine false 1 0 4]) (toyWiring [("B", [("x", 2)])])
  "edge-width"

end Spot.Edges
