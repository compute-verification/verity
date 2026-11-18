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
`[j·n/k, (j+1)·n/k)` of a member's `n`) lies in one part of its argument, and reads exactly one whole return field of
one member of the writer node, at the field's bits. A part is a grid (`partLanes`): a row lies in one lane's run, and
across the members a part covers its place moves by the argument's member stride, inside that run or by whole periods
of the lanes; the writer leaves the members read are then a mixed radix of the lane's levels (`laneWire`), and a wire
needs the writer field to stay the same, so each step is a whole number of writer members.

The refusals, by name: `edge-parts` (an argument of a root node has more parts than `argPartsOk` allows, or a
grid's lanes count different numbers of periods, `gridOk`),
`edge-template` (a non-input root node whose template the ranges or the wiring don't list), `edge-form` (a scan at the
root, as reader or writer), `not-topological` (a part reads a later node, or the node itself), `edge-rows` (a reader row
across a part's edge, not one whole writer field, or not in closed form: a row across runs or not contiguous, members
whose rows are no mixed radix of the lane's levels, an explicit part naming a non-input node, a reference to a
parameter, a grid's lane outside its writer node), `edge-fields` (one of those, where the part reads more than one
output field of one writer: one field a part is the rule, `crossesFields`) and `edge-width` (a reader row's bits are
not the field's).

The work is the root's nodes and arguments, then each part times its port's rows: nothing per unit or per block. The
parts are read only after `argPartsOk` has bounded them by the root's nodes.
-/

@[expose] public section

namespace Spot.Edges

open Flock.Program (Prog Node Arg Part Ty Defn Lane fieldsOf isInputId)
open Flock.TemplateQuery (TRange)
open Spot.Bits (bitsOf)

/-- Each ranged root node's template and the template's units in the earlier ranges, one pass over the ranges. -/
def firstUnits (rs : Array TRange) : Std.HashMap Nat (String × Nat) :=
  (rs.foldl (fun (acc : Std.HashMap Nat (String × Nat) × Std.HashMap String Nat) r =>
    let b := acc.2.getD r.template 0
    (acc.1.insert r.node (r.template, b), acc.2.insert r.template (b + r.count))) ({}, {})).1

/-- The space a part reads: its node or parameter, or an explicit part's first reference's, or a grid's first lane's;
none for an empty one. -/
def partKey : Part → Option (Bool × Nat)
  | .affine p i _ _ => some (p, i)
  | .strided p i _ _ _ _ _ => some (p, i)
  | .explicit it => it[0]?.map fun r => (r.p, r.i)
  | .grid _ lanes => lanes[0]?.map fun l => (l.p, l.i)

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

/-- The return field of `t` that starts at leaf `o`: its index, leaves and bits. -/
def fieldAt (t : Ty) (o : Nat) : Option (Nat × Nat × Nat) :=
  ((fieldsOf t).foldl (fun (acc : Option (Nat × Nat × Nat) × Nat × Nat) ty =>
    (if acc.2.1 == o && acc.1.isNone then some (acc.2.2, ty.leaves, bitsOf ty) else acc.1,
      acc.2.1 + ty.leaves, acc.2.2 + 1)) (none, 0, 0)).1

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

/-- A part's run and lanes, the grid it spells; an explicit part none. -/
def partLanes : Part → Nat × List Lane
  | .affine p i b c => (max c 1, [⟨p, i, b, 1, []⟩])
  | .strided p i b inner os is c =>
    let T := max 1 ((c + inner - 1) / inner)
    (inner, [⟨p, i, b, is, if T > 1 then [(T, os)] else []⟩])
  | .explicit _ => (0, [])
  | .grid run lanes => (run, lanes.toList)

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

/-- The index of the field of `t` holding leaf `o`. -/
def fieldIx (t : Ty) (o : Nat) : Option Nat :=
  ((fieldsOf t).foldl (fun (acc : Option Nat × Nat × Nat) ty =>
    (if acc.2.1 ≤ o && o < acc.2.1 + ty.leaves && acc.1.isNone then some acc.2.2 else acc.1,
      acc.2.1 + ty.leaves, acc.2.2 + 1)) (none, 0, 0)).1

/-- Reference `k` of the grid `(run, lanes)`: its lane's collection and its leaf. -/
def leafAt (run : Nat) (lanes : List Lane) (k : Nat) : Option (Bool × Nat × Int) :=
  let g := lanes.length
  if run == 0 || g == 0 then none else
  let ln := lanes[k % (g * run) / run]!
  let ds := digitsOf (k / (g * run)) ln.lv
  some (ln.p, ln.i, ln.b + ((k % (g * run) % run : Nat) : Int) * ln.e +
    ((ds.zip ln.lv).map fun (d, _, st) => (d : Int) * st).sum)

/-- Whether the rows `k0 + m·L` (`m < cnt`), `len` references each, of the grid `(run, lanes)` (`c` references) read
more than one output field of one writer node, at a row's last reference or the next member's row. -/
def crossesFields (P : Prog) (run : Nat) (lanes : List Lane) (c k0 L len cnt : Nat) : Bool :=
  let nodes := (P.defs[P.root]!).body.nodes
  match leafAt run lanes k0 with
  | some (false, w, l0) =>
    if l0 < 0 || w ≥ nodes.size then false else
    let wd := P.defs[nodes[w]!.fn]!
    let LW := wd.outLeaves
    if LW == 0 then false else
    let f0 := fieldIx wd.ret (l0.toNat % LW)
    let ks := [k0 + len - 1] ++ (if cnt > 1 then [k0 + L, k0 + L + len - 1] else [])
    ks.any fun k => k < c && match leafAt run lanes k with
      | some (false, w', l) => w' == w && l ≥ 0 && fieldIx wd.ret (l.toNat % LW) != f0
      | _ => false
  | _ => false

/-- Part `q` of root node `k`, if a grid: its lanes count one number of periods (else `edge-parts`), and each lane
over an earlier root node lies inside that node's outputs at both its extremes (else `edge-rows`), in O(levels) a
lane. The decoder refuses both first (`grid-shape`, `grid-overrun`); these hold them for a Program built otherwise. -/
def gridOk (P : Prog) (k : Nat) : Part → Except String Unit
  | .grid run lanes => do
    if lanes.any (·.periods != (lanes[0]?.map Lane.periods).getD 0) then throw "edge-parts"
    let nodes := (P.defs[P.root]!).body.nodes
    for l in lanes do
      if !l.p && l.i < k then
        let n := nodes[l.i]!
        let (lo, hi) := l.bounds run
        if lo < 0 || hi ≥ (((P.defs[n.fn]!).outLeaves * n.members : Nat) : Int) then throw "edge-rows"
  | _ => pure ()

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
    gridOk P k q
    let (run, lanes) := partLanes q
    let k0 := x0 + lo * L - s0
    let rows := if crossesFields P run lanes q.count k0 L len cnt then "edge-fields" else "edge-rows"
    let (w, l0, radix, steps) ← match laneWire run lanes k0 L len cnt with
      | .ok v => pure v
      | .error e => throw (if e == "edge-rows" then rows else e)
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
    let some (f, fl, fb) := fieldAt wd.ret (l0 % LW) | throw rows
    if fl != len || (radix.zip steps).any (fun (r, d) => r > 1 && d % (LW : Int) != 0) then throw rows
    if fb != bitsOf ty / cuts then throw "edge-width"
    let wire : Spot.Wires.Wire :=
      { reader := rt, port := port, writer := wt, out := "o" ++ toString f, start := rb + lo, radix := radix,
        base := ((wb + l0 / LW : Nat) : Int), step := steps.map (· / (LW : Int)), cut0 := j, cuts := 1 }
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
private def toy (aRet : Ty) (bParts : Array Part) (bRet : Ty := row2) : Prog :=
  let input : Defn :=
    { key := "Input16_v1", id := "Input16_v1", prim := true, params := #[], ret := v16, gates := 1, paramLeaves := 0,
      outLeaves := 1 }
  let nodes := #[{ form := .batch, fn := 0, args := #[], out := .array 4 v16, n := 4 },
    batch 1 2 #[.affine false 0 0 4] aRet, batch 2 2 bParts bRet, batch 1 2 #[.affine false 2 0 4] aRet]
  let root : Defn :=
    { key := "R", id := "R", prim := false, params := #[], ret := row2, gates := 0, paramLeaves := 0, outLeaves := 2,
      body := { params := #[], retType := row2, nodes := nodes, ret := default, offsets := #[] } }
  { defs := #[input, prim "A" aRet, prim "B" bRet, root], index := {}, root := 3, programFn := "", sha512 := .empty }

private def toyRanges : Array TRange :=
  #[⟨1, "A", 0, 2⟩, ⟨2, "B", 2, 2⟩, ⟨3, "A", 4, 2⟩]

private def toyWiring (cuts : List (String × List (String × Nat))) : Spot.Wires.Wiring :=
  let t (n : Nat) : Spot.Wires.Template := { units := n, rows := [("x", ⟨2, 2⟩)], outs := [("o0", 2)] }
  { templates := [("A", t 4), ("B", t 2)], cuts := cuts, wires := [] }

private def wire (r w o : String) (start : Nat) (radix : List Nat) (base : Int) (step : List Int) (cut0 : Nat) :
    Spot.Wires.Wire :=
  { reader := r, port := "x", writer := w, out := o, start := start, radix := radix, base := base, step := step,
    cut0 := cut0, cuts := 1 }

private def gives (P : Prog) (W : Spot.Wires.Wiring) (ws : List Spot.Wires.Wire) : Bool :=
  match edgesOf P toyRanges W with
  | .ok ws' => ws' == ws
  | .error _ => false

private def refuses (P : Prog) (W : Spot.Wires.Wiring) (code : String) : Bool :=
  match edgesOf P toyRanges W with
  | .ok _ => false
  | .error e => e == code

#guard gives (toy row2 #[.affine false 1 0 4]) (toyWiring [])
  [wire "B" "A" "o0" 0 [2] 0 [1] 0, wire "A" "B" "o0" 2 [2] 0 [1] 0]
#guard refuses (toy row2 #[.affine false 1 1 4]) (toyWiring []) "edge-rows"
#guard gives (toy (.tuple #[v16, v16]) #[.affine false 1 0 4]) (toyWiring [("B", [("x", 2)])])
  [wire "B" "A" "o0" 0 [2] 0 [1] 0, wire "B" "A" "o1" 0 [2] 0 [1] 1, wire "A" "B" "o0" 2 [2] 0 [1] 0]
#guard gives (toy row2 #[.affine false 1 0 2, .affine false 1 2 2]) (toyWiring [])
  [wire "B" "A" "o0" 0 [1] 0 [0] 0, wire "B" "A" "o0" 1 [1] 1 [0] 0, wire "A" "B" "o0" 2 [2] 0 [1] 0]
#guard refuses (toy row2 #[.affine false 1 0 2, .affine false 1 2 1, .affine false 1 3 1]) (toyWiring [])
  "edge-parts"
#guard refuses (toy row2 #[.affine false 3 0 4]) (toyWiring []) "not-topological"
#guard refuses (toy row2 #[.explicit #[⟨false, 1, 0⟩, ⟨false, 1, 1⟩, ⟨false, 1, 2⟩, ⟨false, 1, 3⟩]]) (toyWiring [])
  "edge-rows"
#guard refuses (toy (.tuple #[.value 8, .value 24]) #[.affine false 1 0 4]) (toyWiring [("B", [("x", 2)])])
  "edge-width"

/-! ## Examples: grid parts (the twin's grid cases) -/

private def pair : Ty := .tuple #[v16, v16]
private def cut2 : Spot.Wires.Wiring := toyWiring [("A", [("x", 2)]), ("B", [("x", 2)])]
private def lanes2 (i : Nat) (d : Int := 2) : Array Lane := #[⟨false, i, 0, 0, [(2, d)]⟩, ⟨false, i, 3, 0, [(2, -2)]⟩]

#guard gives (toy pair #[.grid 1 (lanes2 1)] pair) cut2
  [wire "B" "A" "o0" 0 [2] 0 [1] 0, wire "B" "A" "o1" 0 [2] 1 [-1] 1, wire "A" "B" "o0" 2 [2] 0 [1] 0,
    wire "A" "B" "o1" 2 [2] 0 [1] 1]
#guard refuses (toy pair #[.grid 1 (lanes2 1)] pair) (toyWiring []) "edge-fields"
#guard gives (toy pair #[.grid 1 (lanes2 0)] pair) cut2 [wire "A" "B" "o0" 2 [2] 0 [1] 0, wire "A" "B" "o1" 2 [2] 0 [1] 1]
#guard refuses (toy pair #[.grid 1 #[⟨false, 1, 0, 0, [(2, 2)]⟩, ⟨false, 1, 0, 0, [(4, 1)]⟩]] pair) cut2 "edge-parts"
#guard refuses (toy pair #[.grid 1 (lanes2 1 4)] pair) cut2 "edge-rows"
#guard refuses (toy row2 #[.strided false 1 0 3 3 1 4]) (toyWiring []) "edge-rows"

private def laneGives (run : Nat) (lanes : List Lane) (k0 L len n : Nat) (want : Nat × Nat × List Nat × List Int) :
    Bool :=
  match laneWire run lanes k0 L len n with
  | .ok v => v == want
  | .error _ => false

#guard laneGives 64 [⟨false, 7, 0, 1, [(4, 0), (32, 64)]⟩] 0 64 64 128 (7, 0, [32, 4], [64, 0])
#guard laneGives 64 [⟨false, 7, 0, 1, [(4, 0), (32, 64)]⟩] (4 * 64) 64 64 8 (7, 64, [2, 4], [64, 0])

end Spot.Edges
