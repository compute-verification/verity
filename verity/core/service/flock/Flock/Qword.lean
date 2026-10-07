/-
PROTOCOL.md §16.7: `Q_word` v1, the partition query (`verity.ir.cut`, PR #111), over each Call's computing graph
(`Partition.CallGraph`): the flat cut (`cutWord`: output units packed from the returned gates, then values several units read
promoted to committed units, dead gates last), a Definition's cut (a primitive is one unit or none, a separable body its
members' cuts, any other body the flat cut), and the Program's units (by Call, then within the Call's cut) with `locate`.

Decoding the Program's descriptor and extracting each Call's graph (and so deciding separability) are not here yet: `Def`
is the evaluator's input as extraction will produce it.
-/
import Flock.Partition

namespace Flock.Qword

open Lean (Json)
open Partition

/-- Step (i): the returned computing gates, first occurrence in return order, packed into runs of at most `X` declared bits
(a gate wider than `X` is a run alone). -/
def targets (g : CallGraph) (X : Nat) : Array (Array Nat) := Id.run do
  let mut seen : Std.HashSet Nat := {}
  let mut out := #[]
  let mut cur := #[]
  let mut bits := 0
  for o in g.outputs do
    if seen.contains o then continue
    seen := seen.insert o
    let w := g.width[o]!
    if !cur.isEmpty && bits + w > X then
      out := out.push cur
      cur := #[]
      bits := 0
    cur := cur.push o
    bits := bits + w
  if !cur.isEmpty then out := out.push cur
  return out

/-- Step (ii) (core's `owners`): the owner of every computing gate and the heads, extended by the committed values; `kinds`
grows by the committed and dead units. Each round promotes at least one gate that heads nothing to a head, so `n + 1` rounds
suffice. -/
def owners (g : CallGraph) (head0 : Array Int) (kinds0 : Array String) : Array Int × Array Int × Array String := Id.run do
  let n := g.n
  let mut consumers : Array (Array Nat) := Array.replicate n #[]
  let mut producers : Array (Array Nat) := Array.replicate n #[]
  for (p, q) in g.reads do
    consumers := consumers.modify p (·.push q)
    producers := producers.modify q (·.push p)
  let mut head := head0
  let mut kinds := kinds0
  for _ in [0:n + 1] do
    let mut own := head
    for i in [0:n] do
      let x := n - 1 - i
      if own[x]! ≥ 0 then continue
      let mut o : Int := -1
      for c in consumers[x]! do
        let oc := own[c]!
        if oc == -1 || oc == o then continue
        if o == -1 && oc ≥ 0 then o := oc
        else
          o := -2
          break
      own := own.set! x o
    let mut promote := ((g.reads.filter fun (p, q) => own[p]! == -2 && own[q]! ≥ 0).map (·.1)).qsort (· < ·)
    promote := promote.foldl (fun a p => if a.back? == some p then a else a.push p) #[]
    if promote.isEmpty then
      let live := kinds.size
      for x in [0:n] do
        if own[x]! != -1 then continue
        let us := (producers[x]!.filter fun p => head[p]! < 0).foldl (fun (s : Array Int) p =>
          if s.contains own[p]! then s else s.push own[p]!) #[]
        let alive := us.filter fun (u : Int) => 0 ≤ u && u < (live : Int)
        own := own.set! x (if alive.size == 1 && us.size == 1 then alive[0]! else -1)
      for x in [0:n] do
        if own[x]! == -1 then
          own := own.set! x kinds.size
          kinds := kinds.push "dead"
      promote := ((g.reads.filter fun (p, q) => own[p]! != own[q]! && head[p]! < 0).map (·.1)).qsort (· < ·)
      promote := promote.foldl (fun a p => if a.back? == some p then a else a.push p) #[]
      if promote.isEmpty then return (own, head, kinds)
      kinds := kinds.extract 0 live
    for i in [0:promote.size] do head := head.set! promote[i]! (kinds.size + i)
    kinds := kinds ++ Array.replicate promote.size "committed"
  return (head, head, kinds)

/-- `Q_word`'s cut of one flat Call: the owner of every computing gate, the kind of every unit (`output`, `committed`,
`dead`), and each gate's head (its unit when it heads it, else −1). -/
def cutWord (g : CallGraph) (X : Nat) : Array Int × Array String × Array Int := Id.run do
  let mut head : Array Int := Array.replicate g.n (-1)
  let mut kinds : Array String := #[]
  for t in targets g X do
    for x in t do head := head.set! x kinds.size
    kinds := kinds.push "output"
  let (own, heads, units) := owners g head kinds
  return (own, units, heads)

/-- A Definition's cut: `flat` (owner per computing gate, kind per unit) for a body cut as a whole, or `parts`, per body node its
member cut (none for a structure primitive) and its members, whose units are the members' in node, then member, order. -/
inductive DCut where
  | flat (owner : Array Nat) (kinds : Array String)
  | parts (ps : Array (Option DCut × Nat))
  deriving Inhabited

namespace DCut

partial def units : DCut → Nat
  | .flat _ k => k.size
  | .parts ps => ps.foldl (fun acc (c, m) => acc + (c.map units |>.getD 0) * m) 0

partial def gates : DCut → Nat
  | .flat o _ => o.size
  | .parts ps => ps.foldl (fun acc (c, m) => acc + (c.map gates |>.getD 0) * m) 0

/-- The owner of every computing gate of the activation. -/
partial def owner : DCut → Array Nat
  | .flat o _ => o
  | .parts ps => Id.run do
    let mut out := #[]
    let mut base := 0
    for (c, m) in ps do
      if let some c := c then
        let sub := owner c
        for _ in [0:m] do
          out := out ++ sub.map (· + base)
          base := base + units c
    return out

/-- Unit `u` as a path of `(body node, member, unit within the member's cut)` down to a flat cut; none outside the cut. -/
partial def locate : DCut → Nat → Option (List (Nat × Nat × Nat))
  | .flat _ k, u => if u < k.size then some [] else none
  | .parts ps, u0 => Id.run do
    let mut u := u0
    for i in [0:ps.size] do
      let (c, m) := ps[i]!
      if let some c := c then
        let cu := units c
        if u < cu * m then
          return (locate c (u % cu)).map ((i, u / cu, u % cu) :: ·)
        u := u - cu * m
    return none

end DCut

/-- A Definition as the evaluator reads it (graph extraction produces it): a primitive (with or without parameters, wiring or
not), a separable body (per node its callee and members), or any other body with its Call graph. -/
inductive Def where
  | prim (params : Bool) (wiring : Bool)
  | separable (nodes : Array (Def × Nat))
  | flat (g : CallGraph)
  deriving Inhabited

/-- `Q_word{X}` on one activation (core's `evaluate_definition`): a primitive is one unit, none when it is structure (no
parameters, or wiring); a separable body is its members' cuts; any other body is `cutWord` of its graph. -/
partial def evaluate (X : Nat) : Def → DCut
  | .prim params wiring => if !params || wiring then .flat #[] #[] else .flat #[0] #["output"]
  | .separable nodes => .parts (nodes.map fun (d, m) => let c := evaluate X d; (if c.gates > 0 then some c else none, m))
  | .flat g => let (own, kinds, _) := cutWord g X; .flat (own.map Int.toNat) kinds

/-- The Program's units: by Call (its root-body index and cut), then within the Call's cut. -/
structure Units where
  calls : Array (Nat × DCut)
  deriving Inhabited

def Units.starts (P : Units) : Array Nat :=
  (P.calls.foldl (fun (acc : Array Nat × Nat) (_, c) => (acc.1.push acc.2, acc.2 + c.units)) (#[], 0)).1

def Units.population (P : Units) : Nat := P.calls.foldl (fun acc (_, c) => acc + c.units) 0

/-- Unit `i` → (position among the Calls, the Call's root-body index, unit within its cut, path within a separable cut): the
last Call whose units start at or before `i`. -/
def Units.locate (P : Units) (i : Nat) : Option (Nat × Nat × Nat × List (Nat × Nat × Nat)) := do
  if i ≥ P.population then none
  let st := P.starts
  let p := ((List.range st.size).reverse.find? (fun q => st[q]! ≤ i)).getD 0
  let (k, c) := P.calls[p]!
  let u := i - st[p]!
  let path ← c.locate u
  return (p, k, u, path)

/-! ### JSON (the `evaluate` command) -/

private def graphOf (j : Json) : V CallGraph := do
  let (g, _, _, _) ← checkCutOfJson (Json.mkObj [("graph", j), ("owner", Json.arr #[])])
  return g

partial def defOfJson (j : Json) : V Def := do
  match j.getObjVal? "prim", j.getObjVal? "separable", j.getObjVal? "flat" with
  | .ok p, _, _ =>
    pure (.prim ((p.getObjValAs? Bool "params").toOption.getD false) ((p.getObjValAs? Bool "wiring").toOption.getD false))
  | _, .ok (.arr ns), _ => do
    pure (.separable (← ns.mapM fun n => do
      let .ok a := n.getArr? | throw "evaluate: a separable node is not [def, members]"
      if a.size != 2 then throw "evaluate: a separable node is not [def, members]"
      let .ok m := a[1]!.getNat? | throw "evaluate: members is not a natural number"
      pure (← defOfJson a[0]!, m)))
  | _, _, .ok g => pure (.flat (← graphOf g))
  | _, _, _ => throw "evaluate: a definition is not prim, separable or flat"

end Flock.Qword
