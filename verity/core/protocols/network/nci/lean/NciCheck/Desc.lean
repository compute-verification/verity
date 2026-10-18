import NciCheck.Check

/-!
# The NCI check at the size of the description

`NciCheck.Desc.evaluate` decides what `NciCheck.evaluate` decides, for a division whose IUs are unions of the root's
nodes (IUs aligned to Calls), without laying out the circuit: its cost grows with the descriptor, not with the unrolled
circuit (the rule of 18 Oct, AGENTS.md "Design rules"). `NciCheck.evaluate` stays the reference it is held to: the
difftest (`NciCheck.DescDifftest`) checks both on the same declarations, the old one on the division expanded gate by gate.

## The instance

The fields are `NciCheck.Check`'s (`program`, `theta`, `owner`, `gamma`, `alpha`), read by `NciCheck.parse`, with two read
per root node instead of per gate:
* `theta`: θ as root nodes, each an input node (§5.2 of `verity/ml/circuits/PROTOCOL.md`), none twice. θ is every gate of
  those nodes.
* `owner`: one entry per root node: `"input"` for an input node, and `u` for a Call (any other root node), every gate of
  which is in IU `u`. A root node without gates (a batch of none, or a Call of a Definition without gates) holds no gate of
  any IU, so its entry is only read for its form.

The expansion, θ as the gates of its nodes and each gate's owner its root node's, is an instance of `NciCheck.evaluate`, and
the verdict is the one `NciCheck.evaluate` gives on it: `gates`, per IU its size and novel reads, the verdict and the first
failing IU with its rule.

## What is computed, once per Definition

Every gate of a root node reads its operands inside the node, or through the node's arguments (the root has no
parameters, and a node names only earlier nodes, `Prog.orderViolation`). So the gates outside an IU that it reads are what
its nodes' arguments resolve to (§7), restricted to the parameter leaves the callee reads. Two summaries per Definition,
computed in the decoder's order (callees first) and shared by every activation:
* `ret`: its return leaves as segments (`Seg`) of pieces (`Piece`). A piece is `rows` runs `base + r * rowStride +
  t * stride` of the activation's own gates (of one width) or of one parameter's leaves (a pass-through); a strided part
  is one piece of rows. A segment is one piece, or `reps` copies of one batch member's return pieces, each `shift` gates
  after the one before (a batch's whole members, read as a block);
* `used`: per parameter, the leaves some gate of it reads, as intervals (`ivsOf`: a piece of rows as periodic intervals,
  those of one period and count merged, so the rows of a transposed read become runs).

A body's references are resolved through its nodes' outputs with the callees' `ret`: a batch whose members' returns
make one run maps any piece by one affine map, whole members of any other batch become one segment of copies, and a
pass-through goes back through the node's arguments. Per IU, its reads are those segments with the IU's own nodes and θ's
taken out, and its novel reads the width of their union (`novelWidth`): copies of one frame that nothing else meets are
counted as copies times one copy's width.

What is still walked member by member or element by element, so not bounded by the description: a batch member whose
return has a pass-through, a read with stride other than 1 into a composite's batch output that isn't one run, a strided
read of a call's or a scan's output, and segments of copies that overlap other reads of the same IU (expanded copy by
copy). Serve_v1 meets none of them, at word level or on bits.

## Refusals

In the order checked: `instance` and `params` (`NciCheck.parse`), `program` (the decoder refuses it), `not-topological`
(a body names a later node, `Prog.orderViolation`), `program` (a reference the layout can't resolve: the output of a scan
of no iterations), then θ entry by entry (`theta-unknown`: not a root node, `theta-not-input`, `theta-duplicate`), then the
owners (`owner-length`: not one per root node; node by node `owner-entry`, `input-in-iu`, `gate-in-no-iu`), then
`empty-iu`. `NciCheck.evaluate` checks the order gate by gate, so it gives a verdict on a program in which a node reads
an earlier member of itself, which this check refuses.
-/

namespace NciCheck.Desc

open Lean (Json toJson)
open Flock.Program

/-! ## Interval sets and pieces -/

/-- Half-open intervals `[lo, hi)`, sorted, disjoint and not adjacent. -/
abbrev Ivs := Array (Nat × Nat)

def Ivs.norm (xs : Array (Nat × Nat)) : Ivs :=
  let ys := (xs.filter fun p => p.1 < p.2).qsort (fun x y => x.1 < y.1)
  ys.foldl (init := #[]) fun acc (a, b) =>
    match acc.back? with
    | some (c, e) => if a ≤ e then acc.pop.push (c, max b e) else acc.push (a, b)
    | none => acc.push (a, b)

/-- `s` is all of `[0, n)`. -/
def Ivs.covers (s : Ivs) (n : Nat) : Bool := n == 0 || (s.size == 1 && s[0]!.1 == 0 && s[0]!.2 ≥ n)

/-- What a piece's elements are: leaves of a body-scoped collection (a parameter when `p`, else a node's output), or
gates of the activation the pieces are relative to, each of width `w`. -/
inductive Tgt where
  | ref (p : Bool) (i : Nat)
  | gate (w : Nat)
  deriving Inhabited, BEq

/-- `rows · len` elements in order, element `r · len + t` being `base + r * rowStride + t * stride` (a strided part's rows,
or a batch's members): one row (`rows = 1`) unless the rows are what makes the piece. -/
structure Piece where
  tgt : Tgt
  base : Int
  stride : Int
  len : Nat
  rows : Nat := 1
  rowStride : Int := 0
  deriving Inhabited

def Piece.at (q : Piece) (t : Nat) : Int := q.base + t * q.stride

def Piece.count (q : Piece) : Nat := q.rows * q.len

/-- Row `r` of `q`. -/
def Piece.row (q : Piece) (r : Nat) : Piece := { q with base := q.base + r * q.rowStride, rows := 1, rowStride := 0 }

def Piece.rowsOf (q : Piece) : Array Piece := (Array.range q.rows).map q.row

/-- Elements `[a, b)` of `q`, in order: a part of a row, the whole rows, a part of a row. -/
def Piece.slice (q : Piece) (a b : Nat) : Array Piece :=
  if b ≤ a then #[]
  else if q.rows ≤ 1 then #[{ q.row 0 with base := q.at a, len := b - a }]
  else
    let part (r s e : Nat) : Array Piece := if s < e then #[{ q.row r with base := (q.row r).at s, len := e - s }] else #[]
    let (r0, t0, r1, t1) := (a / q.len, a % q.len, b / q.len, b % q.len)
    if r0 == r1 then part r0 t0 t1
    else
      let first := if t0 == 0 then #[] else part r0 t0 q.len
      let lo := if t0 == 0 then r0 else r0 + 1
      let mid : Array Piece := if lo < r1 then #[{ q with base := q.base + lo * q.rowStride, rows := r1 - lo }] else #[]
      first ++ mid ++ part r1 0 t1

/-- A set of naturals `⋃_{c < cnt} [lo + c·period, lo + c·period + len)`. -/
structure PI where
  lo : Nat
  len : Nat
  period : Nat
  cnt : Nat

/-- `q`'s elements as periodic intervals: each row a run or each column one when its rows are consecutive, else its
elements. -/
def Piece.pis (q : Piece) : Array PI :=
  let n := q.len
  let one (q : Piece) : Array PI :=
    if q.len == 0 then #[]
    else if q.stride == 0 || q.len == 1 then #[⟨q.base.toNat, 1, 1, 1⟩]
    else
      let lo := (if q.stride > 0 then q.base else q.at (q.len - 1)).toNat
      if q.stride.natAbs == 1 then #[⟨lo, q.len, 1, 1⟩] else #[⟨lo, 1, q.stride.natAbs, q.len⟩]
  if q.rows ≤ 1 || n == 0 then one (q.row 0)
  else
    let rlo (s : Int) (k : Nat) (b : Int) : Nat := (if s ≥ 0 then b else b + (k - 1 : Nat) * s).toNat
    if q.stride.natAbs == 1 || n == 1 then
      let run := rlo q.stride n q.base
      #[⟨rlo q.rowStride q.rows run, n, q.rowStride.natAbs, q.rows⟩]
    else if q.rowStride.natAbs == 1 && q.stride.natAbs ≥ q.rows then
      let col := rlo q.rowStride q.rows q.base
      #[⟨rlo q.stride n col, q.rows, q.stride.natAbs, n⟩]
    else q.rowsOf.flatMap one

/-- The pieces' elements as intervals of naturals (an order-free view), not necessarily disjoint. Periodic intervals of one
period and count are sorted and those that abut or overlap merged: the rows of a transposed or of a row-strided part,
read part by part, become one run per period, one interval when the runs fill the period. -/
def ivsOf (qs : Array Piece) : Array (Nat × Nat) := Id.run do
  let mut out := #[]
  let mut periodic : Array PI := #[]
  for q in qs do
    for x in q.pis do
      if x.cnt == 0 || x.len == 0 then continue
      if x.cnt == 1 || x.len ≥ x.period then out := out.push (x.lo, x.lo + (x.cnt - 1) * x.period + x.len)
      else periodic := periodic.push x
  let s := periodic.qsort fun x y =>
    x.period < y.period || (x.period == y.period && (x.cnt < y.cnt || (x.cnt == y.cnt && x.lo < y.lo)))
  let emit (out : Array (Nat × Nat)) (x : PI) : Array (Nat × Nat) :=
    if x.len ≥ x.period then out.push (x.lo, x.lo + (x.cnt - 1) * x.period + x.len)
    else (Array.range x.cnt).foldl (fun o c => o.push (x.lo + c * x.period, x.lo + c * x.period + x.len)) out
  let mut cur : Option PI := none
  for x in s do
    match cur with
    | some c =>
      if c.period == x.period && c.cnt == x.cnt && x.lo ≤ c.lo + c.len then
        cur := some { c with len := max c.len (x.lo + x.len - c.lo) }
      else
        out := emit out c
        cur := some x
    | none => cur := some x
  if let some c := cur then out := emit out c
  return out

/-- `reps` copies of one member's pieces `ps` (`len` positions), copy `m`'s gates `m * shift` after copy 0's: the output
of a batch whose members' return isn't one run, read by whole members. Only gates repeat: `ps` names no leaf when
`reps > 1`. -/
structure Seg where
  ps : Array Piece
  len : Nat
  reps : Nat := 1
  shift : Int := 0
  deriving Inhabited

def Seg.one (q : Piece) : Seg := { ps := #[q], len := q.count }

def Seg.count (s : Seg) : Nat := s.reps * s.len

def shiftGates (δ : Int) (ps : Array Piece) : Array Piece :=
  ps.map fun q => match q.tgt with
    | .gate _ => { q with base := q.base + δ }
    | _ => q

/-- Copy `m`'s pieces. -/
def Seg.copy (s : Seg) (m : Nat) : Array Piece := shiftGates (m * s.shift) s.ps

/-- Positions `[a, b)` of the pieces `ps` laid end to end. -/
def slicePieces (ps : Array Piece) (a b : Nat) : Array Piece := Id.run do
  let mut out := #[]
  let mut st := 0
  for q in ps do
    if st ≥ b then break
    let en := st + q.count
    if en > a then out := out ++ q.slice (max a st - st) (min b en - st)
    st := en
  return out

/-- Positions `[a, b)` of `s`: part of a copy, the whole copies, part of a copy. -/
def Seg.slice (s : Seg) (a b : Nat) : Array Seg :=
  if b ≤ a then #[]
  else if s.reps ≤ 1 then (slicePieces s.ps a b).map Seg.one
  else
    let part (m x y : Nat) : Array Seg := (slicePieces (s.copy m) x y).map Seg.one
    let (m0, t0, m1, t1) := (a / s.len, a % s.len, b / s.len, b % s.len)
    if m0 == m1 then part m0 t0 t1
    else
      let first := if t0 == 0 then #[] else part m0 t0 s.len
      let lo := if t0 == 0 then m0 else m0 + 1
      let mid : Array Seg := if lo < m1 then #[{ s with ps := s.copy lo, reps := m1 - lo }] else #[]
      first ++ mid ++ part m1 0 t1

/-- A sequence of segments with each one's first position. -/
structure PS where
  segs : Array Seg
  starts : Array Nat
  count : Nat
  deriving Inhabited

def PS.of (segs : Array Seg) : PS :=
  let segs := segs.filter (·.count > 0)
  let (starts, count) := segs.foldl (fun (acc : Array Nat × Nat) s => (acc.1.push acc.2, acc.2 + s.count)) (#[], 0)
  { segs, starts, count }

/-- Positions `[lo, hi)`: the first segment by bisection over the starts, then the segments it spans. -/
def PS.slice (s : PS) (lo hi : Nat) : Array Seg := Id.run do
  if lo ≥ hi || s.segs.isEmpty then return #[]
  let mut a := 0
  let mut b := s.segs.size
  for _ in [0:s.segs.size] do
    if b - a ≤ 1 then break
    let mid := (a + b) / 2
    if s.starts[mid]! ≤ lo then a := mid else b := mid
  let mut out := #[]
  for i in [a:s.segs.size] do
    let st := s.starts[i]!
    if st ≥ hi then break
    let g := s.segs[i]!
    let en := st + g.count
    if en > lo then out := out ++ g.slice (max lo st - st) (min hi en - st)
  return out

/-- Positions `[lo, hi)` as pieces, copy by copy. -/
def PS.pieces (s : PS) (lo hi : Nat) : Array Piece :=
  (s.slice lo hi).flatMap fun g => (Array.range g.reps).flatMap g.copy

/-- A reference part as pieces: an affine part is one, a strided part its whole rows as one and a part of a row. -/
def ofPart : Part → Array Piece
  | .affine p i b c => #[{ tgt := .ref p i, base := b, stride := 1, len := c }]
  | .strided p i b inner os is c =>
    if c == 0 then #[]
    else if inner == 1 then #[{ tgt := .ref p i, base := b, stride := os, len := c }]
    else if os == (inner : Int) * is then #[{ tgt := .ref p i, base := b, stride := is, len := c }]
    else
      let whole : Piece := { tgt := .ref p i, base := b, stride := is, len := inner, rows := c / inner, rowStride := os }
      let rest : Piece := { tgt := .ref p i, base := b + ((c / inner : Nat) : Int) * os, stride := is, len := c % inner }
      #[whole, rest].filter (·.count > 0)
  | .explicit it => it.foldl (init := #[]) fun acc r =>
    let one : Piece := { tgt := .ref r.p r.i, base := r.j, stride := 1, len := 1 }
    match acc.back? with
    | some q => if q.tgt == .ref r.p r.i && q.stride == 1 && q.at q.len == r.j then acc.pop.push { q with len := q.len + 1 }
      else acc.push one
    | none => acc.push one

def ofSeq (s : Seq) : PS := PS.of ((s.parts.foldl (fun acc q => acc ++ ofPart q) #[]).map Seg.one)

/-! ## The summaries -/

/-- A Definition's summary, valid in every activation of it: its return leaves as pieces of its own gates or parameter
leaves, and per parameter the leaves some gate of it reads. -/
structure Summary where
  ret : PS
  used : Array Ivs
  deriving Inhabited

/-- One body as the analysis reads it: the Program, the callees' summaries, the Definition, and its nodes' arguments as
pieces. -/
structure Ctx where
  P : Prog
  memo : Array Summary
  d : Nat
  args : Array (Array PS)

def Ctx.body (c : Ctx) : Body := (c.P.defs[c.d]!).body

def Ctx.node (c : Ctx) (k : Nat) : Except String Node :=
  match c.body.nodes[k]? with
  | some n => pure n
  | none => throw "program"

def Ctx.arg (c : Ctx) (k i : Nat) : PS := (c.args[k]?.bind (·[i]?)).getD default

/-- Positions of member `m`'s argument `i` of node `k` for its parameter leaves `[a, b)`: the whole argument's when it is
shared, else the member's slice of the mapped axis (`Arg.memberRef`), as runs. -/
def memberSpans (a : Arg) (m lo hi : Nat) : Array (Nat × Nat) :=
  match a.dims with
  | none => #[(lo, hi)]
  | some (d, L) =>
    if L == 0 then #[] else
    (Array.range ((hi + L - 1) / L - lo / L)).filterMap fun q' =>
      let q := lo / L + q'
      let s := max lo (q * L)
      let e := min hi ((q + 1) * L)
      if s < e then some ((q * d + m) * L + (s - q * L), (q * d + m) * L + (e - q * L)) else none

mutual

/-- The pieces of body references `q`, resolved (§7) down through the nodes they name until a gate of the activation or a
parameter leaf. -/
def down (c : Ctx) : Nat → Piece → Except String (Array Seg)
  | 0, _ => throw "references nest deeper than this verifier follows"
  | fuel + 1, q => do
    match q.tgt with
    | .gate _ | .ref true _ => return #[.one q]
    | .ref false k =>
      if q.count == 0 then return #[]
      let node ← c.node k
      let o : Int := c.body.offsets[k]!
      let callee := c.P.defs[node.fn]!
      let byRow (_ : Unit) : Except String (Array Seg) := do
        let mut out := #[]
        for x in q.rowsOf do out := out ++ (← down c fuel x)
        return out
      match node.form with
      | .prim =>
        let .value w := callee.ret | throw "program"
        return #[.one { tgt := .gate w, base := o, stride := 0, len := q.count }]
      | .batch =>
        if callee.prim then
          let .value w := callee.ret | throw "program"
          return #[.one { q with tgt := .gate w, base := o + q.base }]
        let L := callee.outLeaves
        let G : Int := callee.gates
        let r := (c.memo[node.fn]!).ret
        if L == 0 then throw "program"
        -- every member returns one run of its gates, and the members' runs make one: leaf `e` is gate `A + e * F`
        if r.segs.size == 1 && r.segs[0]!.reps ≤ 1 && r.segs[0]!.ps.size == 1 then
          let g := r.segs[0]!.ps[0]!
          if let .gate w := g.tgt then
            if g.rows ≤ 1 && g.len == L && (L == 1 || G == L * g.stride) then
              let F : Int := if L == 1 then G else g.stride
              let A : Int := o + g.base
              return #[.one { q with tgt := .gate w, base := A + q.base * F, stride := q.stride * F,
                                     rowStride := q.rowStride * F }]
        if q.rows > 1 then byRow ()
        else if q.stride == 1 then
          let a := q.base.toNat
          let b := a + q.len
          -- the whole members between `m0` and `m1` as copies of one member's return, when it names gates only
          let (m0, m1) := ((a + L - 1) / L, b / L)
          let gatesOnly := r.segs.all fun s => s.reps ≤ 1 && s.ps.all fun g => match g.tgt with
            | .gate _ => true
            | _ => false
          if gatesOnly && m0 + 1 < m1 then
            let mut out := #[]
            if a < m0 * L then out := out ++ (← fromCallee c fuel k (a / L) (a % L) L)
            out := out.push { ps := shiftGates (o + m0 * G) (r.segs.flatMap (·.ps)), len := L, reps := m1 - m0, shift := G }
            if m1 * L < b then out := out ++ (← fromCallee c fuel k m1 0 (b - m1 * L))
            return out
          let mut out := #[]
          for m in [a / L:(b - 1) / L + 1] do
            out := out ++ (← fromCallee c fuel k m (max a (m * L) - m * L) (min b ((m + 1) * L) - m * L))
          return out
        else
          let mut out := #[]
          for t in [0:q.len] do
            out := out ++ (← down c fuel { q with base := q.at t, stride := 1, len := 1 })
          return out
      | .call =>
        if q.rows > 1 then byRow ()
        else if q.stride == 1 then fromCallee c fuel k 0 q.base.toNat (q.base.toNat + q.len)
        else
          let mut out := #[]
          for t in [0:q.len] do
            out := out ++ (← fromCallee c fuel k 0 (q.at t).toNat ((q.at t).toNat + 1))
          return out
      | .scan =>
        if node.n == 0 then throw "a scan with no iterations has no output to read"
        if q.rows > 1 then return ← byRow ()
        let .tuple #[carry, y] := callee.ret | throw "a scan step returns (carry, y)"
        let cl := carry.leaves
        let yl := y.leaves
        let mut out := #[]
        for t in [0:q.len] do
          let j := (q.at t).toNat
          let (m, jj) := if j < cl then (node.n - 1, j) else ((j - cl) / yl, cl + (j - cl) % yl)
          out := out ++ (← fromCallee c fuel k m jj (jj + 1))
        return out
termination_by structural fuel _ => fuel

/-- Return leaves `[a, b)` of member (or iteration) `m` of node `k`, a composite's: its gates offset to the member's, and
its pass-throughs taken back through the node's arguments and resolved in the body. -/
def fromCallee (c : Ctx) : Nat → Nat → Nat → Nat → Nat → Except String (Array Seg)
  | 0, _, _, _, _ => throw "references nest deeper than this verifier follows"
  | fuel + 1, k, m, a, b => do
    let node ← c.node k
    let callee := c.P.defs[node.fn]!
    let o : Int := c.body.offsets[k]! + m * callee.gates
    let mut out := #[]
    for s in (c.memo[node.fn]!).ret.slice a b do
      if s.reps > 1 then out := out.push { s with ps := shiftGates o s.ps }
      else
        for g in s.ps do
          match g.tgt with
          | .gate _ => out := out.push (.one { g with base := o + g.base })
          | .ref false _ => throw "program"
          | .ref true i => out := out ++ (← throughArg c fuel k m i g)
    return out
termination_by structural fuel _ _ _ _ => fuel

/-- Parameter leaves `g` of member (or iteration) `m` of node `k`'s callee, as what node `k`'s arguments name in the body
(§7, up through a parameter), resolved there. -/
def throughArg (c : Ctx) : Nat → Nat → Nat → Nat → Piece → Except String (Array Seg)
  | 0, _, _, _, _ => throw "references nest deeper than this verifier follows"
  | fuel + 1, k, m, i, g => do
    let node ← c.node k
    let some a := node.args[i]? | throw "program"
    let ap := c.arg k i
    let one (lo hi : Nat) : Except String (Array Seg) := do
      let mut out := #[]
      for q in ap.pieces lo hi do out := out ++ (← down c fuel q)
      return out
    let elems : Array (Nat × Nat) := g.rowsOf.flatMap fun g =>
      if g.stride == 1 then #[(g.base.toNat, g.base.toNat + g.len)]
      else (Array.range g.len).map fun t => ((g.at t).toNat, (g.at t).toNat + 1)
    let mut out := #[]
    for (lo, hi) in elems do
      match node.form with
      | .scan =>
        if i == 0 then
          if m == 0 then out := out ++ (← one lo hi)
          else out := out ++ (← fromCallee c fuel k (m - 1) lo hi)
        else if i ≤ node.nxs then
          let L := (a.dims.map (·.2)).getD 0
          out := out ++ (← one (m * L + lo) (m * L + hi))
        else out := out ++ (← one lo hi)
      | .batch =>
        for (s, e) in memberSpans a m lo hi do out := out ++ (← one s e)
      | _ => out := out ++ (← one lo hi)
    return out
termination_by structural fuel _ _ _ _ => fuel

end

/-- The parameter leaves of member `m` of node `k`'s callee that it reads, as positions of argument `i` (`memberSpans` for
a batch, the iteration's slice for a scan's `xs`). -/
def argSpans (node : Node) (i m : Nat) (used : Ivs) : Array (Nat × Nat) :=
  let a := node.args[i]!
  match node.form with
  | .batch => used.foldl (fun acc (lo, hi) => acc ++ memberSpans a m lo hi) #[]
  | .scan =>
    if 1 ≤ i && i ≤ node.nxs then
      let L := (a.dims.map (·.2)).getD 0
      used.map fun (lo, hi) => (m * L + lo, m * L + hi)
    else used
  | _ => used

/-- What node `k`'s gates read outside the node, resolved in the body: every argument of a primitive (or of a batch of one
with members); for a composite callee, the positions its `used` names, member by member unless every member reads its
whole argument; and for a scan, an iteration's carry read from the one before it, through that one's pass-throughs. -/
def nodeReads (c : Ctx) (k : Nat) : Except String (Array Seg) := do
  let node ← c.node k
  let callee := c.P.defs[node.fn]!
  let fuel := Flock.Steps.FUEL
  let resolve (i : Nat) (spans : Array (Nat × Nat)) : Except String (Array Seg) := do
    let ap := c.arg k i
    let mut out := #[]
    for (lo, hi) in spans do
      for q in ap.pieces lo hi do out := out ++ (← down c fuel q)
    return out
  let whole (i : Nat) : Array (Nat × Nat) := #[(0, (c.arg k i).count)]
  let mut out := #[]
  if callee.prim then
    if node.form == .batch && node.n == 0 then return #[]
    for i in [0:node.args.size] do out := out ++ (← resolve i (whole i))
    return out
  if node.members == 0 then return #[]
  let used := (c.memo[node.fn]!).used
  for i in [0:node.args.size] do
    let u := used[i]?.getD #[]
    let lv := (callee.params[i]?.map (·.2.leaves)).getD 0
    let perMember := match node.form with
      | .batch => node.args[i]!.dims.isSome
      | .scan => 1 ≤ i && i ≤ node.nxs
      | _ => false
    if node.form == .scan && i == 0 then
      out := out ++ (← resolve 0 u)
    else if !perMember then out := out ++ (← resolve i u)
    else if u.covers lv then out := out ++ (← resolve i (whole i))
    else
      for m in [0:node.members] do out := out ++ (← resolve i (argSpans node i m u))
  if node.form == .scan then
    for m in [1:node.n] do
      for (lo, hi) in used[0]?.getD #[] do out := out ++ (← fromCallee c fuel k (m - 1) lo hi)
  let lo : Int := c.body.offsets[k]!
  let hi : Int := c.body.offsets[k + 1]!
  -- a segment's gates are one node's
  return out.filter fun s => match s.ps[0]? with
    | some q => match q.tgt with
      | .gate _ => !(lo ≤ q.base && q.base < hi)
      | _ => true
    | none => false

/-- Definition `d`'s summary, its callees' in `memo`. -/
def summarize (P : Prog) (memo : Array Summary) (d : Nat) : Except String Summary := do
  let defn := P.defs[d]!
  if defn.prim then return { ret := default, used := #[] }
  let body := defn.body
  let c : Ctx := { P, memo, d, args := body.nodes.map fun n => n.args.map (ofSeq ·.seq) }
  let mut ret := #[]
  for s in (ofSeq body.ret).segs do
    for q in s.ps do ret := ret ++ (← down c Flock.Steps.FUEL q)
  let mut acc : Array (Array Piece) := Array.replicate body.params.size #[]
  for k in [0:body.nodes.size] do
    for s in ← nodeReads c k do
      if s.reps > 1 then continue
      for q in s.ps do
        if let .ref true i := q.tgt then
          acc := acc.modify i (·.push q)
  return { ret := PS.of ret, used := acc.map fun qs => Ivs.norm (ivsOf qs) }

/-- The Definitions the root reaches, each once. -/
def reached (P : Prog) : Array Bool := Id.run do
  let mut seen := Array.replicate P.defs.size false
  seen := seen.set! P.root true
  for i in [0:P.defs.size] do
    let d := P.defs.size - 1 - i
    if !seen[d]! then continue
    let defn := P.defs[d]!
    if defn.prim then continue
    for n in defn.body.nodes do seen := seen.set! n.fn true
  return seen

/-- Every reached Definition's summary, callees first, and what each root node reads outside itself, as gates. -/
def analyze (P : Prog) : Except String (Array (Array Seg)) := do
  let seen := reached P
  let mut memo : Array Summary := #[]
  for d in [0:P.defs.size] do
    memo := memo.push (← if seen[d]! && d != P.root then summarize P memo d else pure default)
  let body := (P.defs[P.root]!).body
  let c : Ctx := { P, memo, d := P.root, args := body.nodes.map fun n => n.args.map (ofSeq ·.seq) }
  let mut reads := #[]
  for k in [0:body.nodes.size] do
    let rs ← nodeReads c k
    if rs.any (·.ps.any fun q => match q.tgt with | .gate _ => false | _ => true) then throw "program"
    reads := reads.push rs
  return reads

/-! ## θ, the division and the verdict -/

/-- The root node holding gate `g`, by bisection over the root body's offsets. -/
def nodeOf (offsets : Array Nat) (g : Nat) : Nat := Id.run do
  let mut a := 0
  let mut b := offsets.size - 1
  for _ in [0:offsets.size] do
    if b - a ≤ 1 then break
    let mid := (a + b) / 2
    if offsets[mid]! ≤ g then a := mid else b := mid
  return a

/-- The total width of the distinct gates of `ivs` (each `(lo, hi, w)`: gates `[lo, hi)` of width `w`). -/
def distinctWidth (ivs : Array (Nat × Nat × Nat)) : Nat :=
  let s := ivs.qsort fun x y => x.1 < y.1
  (s.foldl (init := (0, 0)) fun (tot, cover) (lo, hi, w) =>
    if hi ≤ cover then (tot, cover) else (tot + (hi - max lo cover) * w, hi)).1

/-- The gates of `qs` as intervals with their widths. -/
def gateIvs (qs : Array Piece) : Array (Nat × Nat × Nat) :=
  let ws := (qs.filterMap fun q => match q.tgt with | .gate w => some w | _ => none).qsort (· < ·) |>.toList.eraseDups
  ws.toArray.flatMap fun w => (ivsOf (qs.filter (·.tgt == .gate w))).map fun (lo, hi) => (lo, hi, w)

/-- Gates `cnt` copies, `period` apart, of `res` (each `(lo, hi, w)` from `lo`, none past `span ≤ period`). -/
structure Cl where
  lo : Nat
  period : Nat
  cnt : Nat
  span : Nat
  res : Array (Nat × Nat × Nat)
  deriving Inhabited

/-- The total width of the distinct gates of `segs`. A segment of copies is copies of one frame; copies of one frame are
counted as copies times one copy's width, alone in their span, and with what else meets their span as gates. -/
def novelWidth (segs : Array Seg) : Nat := Id.run do
  let mut ivs := gateIvs (segs.filter (·.reps ≤ 1) |>.flatMap (·.ps))
  let mut cls : Array Cl := #[]
  for s in segs do
    if s.reps ≤ 1 then continue
    let r := gateIvs s.ps
    if r.isEmpty then continue
    let lo := r.foldl (fun m (a, _, _) => min m a) r[0]!.1
    let hi := r.foldl (fun m (_, b, _) => max m b) 0
    if s.shift ≤ 0 || hi - lo > s.shift.toNat then
      ivs := ivs ++ (Array.range s.reps).flatMap fun m => gateIvs (s.copy m)
    else
      cls := cls.push { lo, period := s.shift.toNat, cnt := s.reps, span := hi - lo,
                        res := r.map fun (a, b, w) => (a - lo, b - lo, w) }
  let cs := cls.qsort fun x y =>
    x.lo < y.lo || (x.lo == y.lo && (x.period < y.period || (x.period == y.period && x.cnt < y.cnt)))
  let mut frames : Array Cl := #[]
  for x in cs do
    match frames.back? with
    | some y =>
      if y.lo == x.lo && y.period == x.period && y.cnt == x.cnt then
        frames := frames.pop.push { y with span := max y.span x.span, res := y.res ++ x.res }
      else frames := frames.push x
    | none => frames := frames.push x
  let n := ivs.size
  let items := (ivs.mapIdx fun i (lo, hi, _) => (lo, hi, i)) ++
    (frames.mapIdx fun i x => (x.lo, x.lo + (x.cnt - 1) * x.period + x.span, n + i))
  let expand (j : Nat) : Array (Nat × Nat × Nat) :=
    if j < n then #[ivs[j]!]
    else
      let x := frames[j - n]!
      (Array.range x.cnt).flatMap fun m => x.res.map fun (a, b, w) => (x.lo + m * x.period + a, x.lo + m * x.period + b, w)
  let width (comp : Array Nat) : Nat :=
    if comp.size == 1 && comp[0]! ≥ n then
      let x := frames[comp[0]! - n]!
      x.cnt * distinctWidth x.res
    else distinctWidth (comp.flatMap expand)
  let mut total := 0
  let mut comp : Array Nat := #[]
  let mut reach := 0
  for (st, en, j) in items.qsort fun x y => x.1 < y.1 do
    if !comp.isEmpty && st ≥ reach then
      total := total + width comp
      comp := #[]
      reach := 0
    comp := comp.push j
    reach := max reach en
  if !comp.isEmpty then total := total + width comp
  return total

def isInputNode (P : Prog) (n : Node) : Bool := isInputId (P.defs[n.fn]!).key

def evaluate (j : Json) : Except String Json := do
  let i ← parse j
  let P ← match decode i.program.toUTF8 with
    | .ok P => pure P
    | .error _ => throw "program"
  if P.orderViolation.isSome then throw "not-topological"
  let reads ← match analyze P with
    | .ok r => pure r
    | .error _ => throw "program"
  let body := (P.defs[P.root]!).body
  let K := body.nodes.size
  let offs := body.offsets
  let gatesOf (k : Nat) : Nat := offs[k + 1]! - offs[k]!
  let inp := body.nodes.map (isInputNode P)
  let mut theta := Array.replicate K false
  for t in i.theta do
    let some k := natOf? t | throw "theta-unknown"
    if k ≥ K then throw "theta-unknown"
    if !inp[k]! then throw "theta-not-input"
    if theta[k]! then throw "theta-duplicate"
    theta := theta.set! k true
  if i.owner.size != K then throw "owner-length"
  let mut own : Array (Option Nat) := #[]
  for k in [0:K] do
    let e := i.owner[k]!
    let o ← if isInputTag e then pure none else match natOf? e with
      | some u => pure (some u)
      | none => throw "owner-entry"
    if gatesOf k > 0 then
      if o.isSome && inp[k]! then throw "input-in-iu"
      if o.isNone && !inp[k]! then throw "gate-in-no-iu"
    own := own.push (if gatesOf k > 0 then o else none)
  let n := iuCount own
  if n > K then throw "empty-iu"
  let used := usedOf own
  if !(List.range n).all used.contains then throw "empty-iu"
  let mut size := Array.replicate n 0
  let mut into : Array (Array Seg) := Array.replicate n #[]
  for k in [0:K] do
    let some u := own[k]! | continue
    size := size.modify u (· + gatesOf k)
    for s in reads[k]! do
      let some q := s.ps[0]? | continue
      if s.count == 0 then continue
      let k' := nodeOf offs q.base.toNat
      if own[k']! == some u || theta[k']! then continue
      into := into.modify u (·.push s)
  let r := (Array.range n).map fun u => (size[u]!, novelWidth into[u]!)
  return verdict (P.defs[P.root]!).gates r (failure i.gamma i.num i.den r)

/-- `evaluate`, with a refusal as `{"error": code}`. -/
def check (j : Json) : Json :=
  match evaluate j with
  | .ok r => r
  | .error code => Json.mkObj [("error", Json.str code)]

end NciCheck.Desc
