import VStar.Algebra
import VStar.Compose

/-!
# V*'s algebra cut into statements (`verity_flock.rec_residuals.parts`)

A rep's `InnerRepCheck` S, of any number of residuals, is cut into parts of at most `MAX_REGIONS` public regions,
`MAX_SLOTS` sha512x3 slots and 16 residuals each, each residual in the first part it fits, in S's order; a residual that fits
no part, not even alone, is a part of its own if it reads at most `MAX_LONE_SLOTS` slots. A part reads the row ports its residuals touch (in
S's order) and the v elements they touch (in v's order), w and v renumbered. `parts` reads S's terms through their ports
without checking them, so its domain is a checked S (`checkS`): past it, Python's slices and lookups fail.
-/

namespace VStar

def MAX_REGIONS : Nat := 64
def MAX_SLOTS : Nat := 160
/-- `rec_residuals.MAX_LONE_SLOTS`: a lone residual's part beside hm96's 2^23 and a 2^23 unit, within a 2^27 block. -/
def MAX_LONE_SLOTS : Nat := 223

/-- `rec_residuals.regions`: a digest region per row port, `v`'s and `out`'s. -/
def regions (ports : Nat) : Nat := ports + 2

/-- `rec_residuals.slots`: the sha512x3 slots of the row ports', `v`'s and `out`'s compressions, three a slot. -/
def slots (ports : List (String × Nat)) (nv : Nat) : Nat :=
  let rows : List Row := ports.map (fun p => ⟨p.1, portWordsOf p.2⟩) ++ [⟨"v", portWordsOf nv⟩, ⟨"out", RES_OUT_WORDS⟩]
  (rows.foldl (fun a r => a + r.comps) 0 + PER_SLOT - 1) / PER_SLOT

/-- A part: its S, and the residuals and v elements of the whole S it holds. -/
structure Part where
  S : RepS
  res : List Nat
  v : List Nat
  deriving Inhabited

def sortDedup (a : Array Nat) : Array Nat :=
  (a.qsort (· < ·)).foldl (fun acc x => if acc.back? == some x then acc else acc.push x) #[]

/-- A residual's row ports and v elements (`parts`' `touched`), each ascending. -/
def touched (S : RepS) (starts : Array Nat) (r : Res) : Array Nat × Array Nat := Id.run do
  let ops := S.gf.ops.toArray
  let portOf (i : Nat) : Nat := (starts.findIdx? (· > i)).getD starts.size - 1
  let mut P : Array Nat := #[]
  let mut V : Array Nat := #[]
  let terms := r.add ++ r.mul.flatMap fun (a, b) => ops.getD a [] ++ ops.getD b []
  for t in terms do
    match t with
    | .w i => P := P.push (portOf i)
    | .wt i _ => for j in [0:128] do P := P.push (portOf (i + j))
    | .v i => V := V.push i
  return (sortDedup P, sortDedup V)

/-- `rec_residuals.parts(S, maxRegions, maxSlots, maxLoneSlots)`. -/
def parts (S : RepS) (maxRegions := MAX_REGIONS) (maxSlots := MAX_SLOTS) (maxLoneSlots := MAX_LONE_SLOTS) :
    Except String (Array Part) := do
  checkRepS S (whole := true)
  checkS S.gf
  let ports := S.ports.toArray
  let mut starts : Array Nat := #[0]
  for p in ports do starts := starts.push (starts.back! + p.2)
  let lone := max maxSlots maxLoneSlots
  let fitN (n : Nat) (g : Array Nat × Array Nat × Array Nat) (P V : Array Nat) : Bool :=
    let ps := Algebra.union g.2.1 P
    regions ps.size ≤ maxRegions && slots (ps.toList.map (ports[·]!)) (Algebra.union g.2.2 V).size ≤ n
      && 8 * (g.1.size + 1) ≤ RES_OUT_WORDS
  let fit := fitN maxSlots
  let mut groups : Array (Array Nat × Array Nat × Array Nat) := #[]
  let res := S.gf.res.toArray
  for i in [0:res.size] do
    let (P, V) := touched S starts res[i]!
    match groups.findIdx? (fit · P V) with
    | some gi =>
      let (R, P0, V0) := groups[gi]!
      groups := groups.set! gi (R.push i, Algebra.union P0 P, Algebra.union V0 V)
    | none =>
      unless fitN lone (#[], #[], #[]) P V do
        throw s!"residual {i} alone reads more than {maxRegions} regions or {lone} sha512x3 slots"
      groups := groups.push (#[i], P, V)
  let ops := S.gf.ops.toArray
  let mut out : Array Part := #[]
  for (R, P, V) in groups do
    let mut wmap : Std.HashMap Nat Nat := {}
    for p in P do
      for i in [starts[p]!:starts[p + 1]!] do wmap := wmap.insert i wmap.size
    let mut vmap : Std.HashMap Nat Nat := {}
    for i in V do vmap := vmap.insert i vmap.size
    let term (t : Term) : Except String Term := match t with
      | .w i => pure (.w (wmap.getD i 0))
      | .v i => pure (.v (vmap.getD i 0))
      | .wt i u =>
        let i' := wmap.getD i 0
        if (List.range 128).all fun j => wmap[i + j]? == some (i' + j) then pure (.wt i' u)
        else throw s!"[\"wt\", {i}, {u}]: a transposed block over row ports the part does not read in a run"
    let mut used : Array Nat := #[]
    for r in R do
      for (a, b) in (res[r]!).mul do used := Algebra.union used (Algebra.union #[a] #[b])
    let mut op : Std.HashMap Nat Nat := {}
    for k in used do op := op.insert k op.size
    let gops ← used.toList.mapM fun k => (ops[k]!).mapM term
    let gres ← R.toList.mapM fun r => do
      let rr := res[r]!
      pure ({ mul := rr.mul.map fun (a, b) => (op.getD a 0, op.getD b 0), add := ← rr.add.mapM term } : Res)
    let part : RepS := { gf := { nw := wmap.size, nv := vmap.size, ops := gops, res := gres }, ports := P.toList.map (ports[·]!) }
    out := out.push { S := part, res := R.toList, v := V.toList }
  pure out

end VStar
