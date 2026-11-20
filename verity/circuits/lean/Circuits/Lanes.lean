/-
A reference part's leaves in closed form: the readers of C that SPOT's wiring (`Spot.Edges`) and Flock's registered
reads (`Flock.Registered`) read a root node's argument by. A part is a grid (`partLanes`): reference `k` is a lane's
leaf at a period and an offset in its run (`partLeafAt`), and the rows `k0 + m·L` of `n` members read a mixed radix of
the lane's levels (`laneWire`). Their work is a part's lanes and levels, nothing per member.
-/
module

public import Circuits.Program

@[expose] public section

namespace Flock.Program

/-- A part's run and lanes, the grid it spells; an explicit part none. -/
def partLanes : Part → Nat × List Lane
  | .affine p i b c => (max c 1, [⟨p, i, b, 1, []⟩])
  | .strided p i b inner os is c =>
    let T := max 1 ((c + inner - 1) / inner)
    (inner, [⟨p, i, b, is, if T > 1 then [(T, os)] else []⟩])
  | .explicit _ => (0, [])
  | .grid run lanes => (run, lanes.toList)

/-- The digits of period `t` in levels `lv`, innermost first, the last unbounded: `Lane.leafAt`'s, which `laneWire` reads one
of. -/
def laneDigits (t : Nat) : List (Nat × Int) → List Nat
  | [] => []
  | [_] => [t]
  | (r, _) :: rest => t % r :: laneDigits (t / r) rest

/-- The levels from `a` on whose radices the members' count `n` and their first period `tt` (over those levels) are
multiples of, as far as they go: the first level not taken and the product of those taken. -/
def laneAbsorb (lv : List (Nat × Int)) (a tt n : Nat) : Nat × Nat := Id.run do
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
  let ds := laneDigits t0 ln.lv
  let leaf := ln.leafAt t0 o0
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
  let (h, P) := laneAbsorb ln.lv a (t0 / dt) n
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

/-- The return field of `t` that starts at leaf `o`: its index and leaves. -/
def fieldSpan (t : Ty) (o : Nat) : Option (Nat × Nat) :=
  ((fieldsOf t).foldl (fun (acc : Option (Nat × Nat) × Nat × Nat) ty =>
    (if acc.2.1 == o && acc.1.isNone then some (acc.2.2, ty.leaves) else acc.1,
      acc.2.1 + ty.leaves, acc.2.2 + 1)) (none, 0, 0)).1

/-- Reference `k` of the grid `(run, lanes)`: its lane's collection and its leaf (`Lane.leafAt`, as `Part.get`'s grid). -/
def partLeafAt (run : Nat) (lanes : List Lane) (k : Nat) : Option (Bool × Nat × Int) :=
  let g := lanes.length
  if run == 0 || g == 0 then none else
  let ln := lanes[k % (g * run) / run]!
  some (ln.p, ln.i, ln.leafAt (k / (g * run)) (k % run))

end Flock.Program
