import NciCheck.Check

/-!
# The NCI check at the size of the description

`NciCheck.Desc.evaluate` decides what `NciCheck.evaluate` decides, for a division whose IUs are unions of the root's
nodes (IUs aligned to Calls), without laying out the circuit: its cost grows with the descriptor, not with the unrolled
circuit (AGENTS.md "Design rules"). `NciCheck.evaluate` stays the reference it is held to: the
difftest (`NciCheck.DescDifftest`) checks both on the same declarations, the old one on the division expanded gate by gate.

## The instance

The fields are `NciCheck.Check`'s (`program`, `theta`, `owner`, `gamma`, `alpha`), read by `NciCheck.parse`, with two read
per root node instead of per gate:
* `theta`: θ as root nodes, each an input node (§5.2 of `verity/ml/circuits/PROTOCOL.md`) bound at registration
  (`RegisteredInput<w>_v1`), none twice. θ is every gate of those nodes.
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
  after the one before (a batch's whole members, read as a block). Gate pieces that continue one progression, then runs
  of one length at one step, are joined (`coalesce`): a body unrolled element by element returns one piece, not one per
  node, and every read of it is one segment;
* `used`: per parameter, the leaves some gate of it reads, as intervals (`ivsOf`: a piece of rows as periodic intervals,
  those of one period and count merged, so the rows of a transposed read become runs).

A body's references are resolved through its nodes' outputs with the callees' `ret`: a batch whose members' returns
make one run maps any piece by one affine map, whole members of any other batch become one segment of copies, and a
pass-through goes back through the node's arguments. A root node's reads are those segments, copies all alike as one and
one copy piece by piece, so that each names one node's gates: whole members returning leaves of several shared arguments
are one copy of pieces of as many nodes (`Seg.split`). Per IU, its reads are those with the IU's own nodes and θ's taken
out, and its novel reads the width of their union (`novelWidth`): copies of one frame that nothing else meets are
counted as copies times one copy's width.

Pass-throughs are taken in closed form in the number of members and iterations: whole members of a batch returning what
its members share are one member's return, repeated (and a stride of whole members one leaf's); those returning a run
of a mapped argument's row are one piece of rows of the argument; a part of an argument every member or iteration reads
is one piece of rows (`memberRows`); a scan's final carry, and the carry each iteration reads from the one before, are
followed back through the step's return (`carry`, `carriesRead`: a run returned in place is the initial carry's, any
other leaf a chain of positions to where it leaves the carry or to its cycle, `Chain`), never iteration by iteration.
What has no closed form here is walked one by one up to `WALK` and refused past it (`unwalked`, with what it would have
walked): the members of a batch whose returns are none of those, strided reads of a call's, scan's or batch's output
that aren't, a scan's ys read iteration by iteration, and chains longer than `WALK`. Still walked unbounded: the leaves
of a strided piece taken through a parameter (`throughArg`), and segments of copies that overlap other reads of the
same IU (expanded copy by copy, `novelWidth`). Serve_v1 meets none of the walks, at word level or on bits.

## Refusals

In the order checked: `instance` and `params` (`NciCheck.parse`), `program` (the decoder refuses it, or a reference
reads a leaf at another width than it was committed with, `E_REINTERPRET`: `NciCheck.decodeProgram`), `not-topological`
(a body names a later node, `Prog.orderViolation`), `program` (the root's offsets are not its layout, a body calls the
root or holds an input node outside the root's: `rootLaidOut`, `inputsAtRoot`, which the decoder never gives), `program`
(a `walkBound` past half of `Flock.Steps.FUEL`, past which `Prog.gate`'s walk can run out of steps; a scan of no
iterations with output leaves, read or not; or a body calling a Definition decoded after it, a Definition not as the
decoder gives it, or a root with parameters, which the decoder never gives: `walkable`), `program` (a reference the
layout can't resolve), `unwalked` (what `analyze` would walk one by one past `WALK`; its `message` says what and how
many), then θ entry by entry (`theta-unknown`: not a root node, `theta-not-input`,
`theta-not-registered`: an input node bound per run, `theta-duplicate`), then the owners (`owner-length`: not one per root
node; node by node `owner-entry`, `input-in-iu`, `gate-in-no-iu`), then
`empty-iu`. `NciCheck.evaluate` checks the order gate by gate, so it gives a verdict on a program in which a node reads
an earlier member of itself, which this check refuses.
-/

namespace NciCheck.Desc

open Lean (Json toJson)
open Flock.Program

/-! ## Interval sets and pieces -/

/-- Half-open intervals `[lo, hi)`, sorted, disjoint and not adjacent. -/
abbrev Ivs := Array (Nat × Nat)

/-- The intervals of `xs` as `Ivs`, sorted by `List.mergeSort`: `Array.qsort` takes time quadratic in the elements of one
key, and a body's nodes read the same leaves many times over. -/
def Ivs.norm (xs : Array (Nat × Nat)) : Ivs :=
  let ys := (xs.filter fun p => p.1 < p.2).toList.mergeSort fun x y => x.1 ≤ y.1
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

/-- `q`'s elements `e` as gates `A + e * F` of width `w`: a batch's output read where its members' returns make one run. -/
def Piece.affine (q : Piece) (w : Nat) (A F : Int) : Piece :=
  { q with tgt := .gate w, base := A + q.base * F, stride := q.stride * F, rowStride := q.rowStride * F }

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

/-- `x`'s set as intervals: one when its runs meet, else one per run. -/
def PI.emit (out : Array (Nat × Nat)) (x : PI) : Array (Nat × Nat) :=
  if x.len ≥ x.period then out.push (x.lo, x.lo + (x.cnt - 1) * x.period + x.len)
  else (Array.range x.cnt).foldl (fun o c => o.push (x.lo + c * x.period, x.lo + c * x.period + x.len)) out

/-- By period, then count, then first run (`List.mergeSort`'s order, which Lean core proves sorts). -/
def PI.le (x y : PI) : Bool :=
  x.period < y.period || (x.period == y.period && (x.cnt < y.cnt || (x.cnt == y.cnt && x.lo ≤ y.lo)))

/-- One periodic interval of the sorted sweep: merged into the current one when of its period and count and starting
within its first run, else the current one emitted and this one current. -/
def PI.mergeStep (st : Array (Nat × Nat) × Option PI) (x : PI) : Array (Nat × Nat) × Option PI :=
  match st.2 with
  | some c =>
    if c.period == x.period && c.cnt == x.cnt && x.lo ≤ c.lo + c.len then
      (st.1, some { c with len := max c.len (x.lo + x.len - c.lo) })
    else (PI.emit st.1 c, some x)
  | none => (st.1, some x)

/-- The pieces' elements as intervals of naturals (an order-free view), not necessarily disjoint. Periodic intervals of one
period and count are sorted and those that abut or overlap merged: the rows of a transposed or of a row-strided part,
read part by part, become one run per period, one interval when the runs fill the period. -/
def ivsOf (qs : Array Piece) : Array (Nat × Nat) :=
  let xs := (qs.flatMap Piece.pis).filter fun x => !(x.cnt == 0 || x.len == 0)
  let out := (xs.filter fun x => x.cnt == 1 || x.len ≥ x.period).map fun x =>
    (x.lo, x.lo + (x.cnt - 1) * x.period + x.len)
  let s := (xs.filter fun x => !(x.cnt == 1 || x.len ≥ x.period)).toList.mergeSort PI.le
  match s.foldl PI.mergeStep (out, none) with
  | (out, some c) => PI.emit out c
  | (out, none) => out

/-- `reps` copies of one member's pieces `ps` (`len` positions), copy `m`'s gates `m * shift` after copy 0's: the output
of a batch whose members' return isn't one run, read by whole members. Leaves never shift: `ps` names a leaf with
`reps > 1` only when `shift = 0`, every copy alike (members returning what they share, `down`). -/
structure Seg where
  ps : Array Piece
  len : Nat
  reps : Nat := 1
  shift : Int := 0
  deriving Inhabited

def Seg.one (q : Piece) : Seg := { ps := #[q], len := q.count }

def Seg.count (s : Seg) : Nat := s.reps * s.len

/-- `s` as a set of gates and leaves: copies all alike (`shift = 0`) as one. -/
def Seg.alike (s : Seg) : Seg := if s.reps > 1 && s.shift == 0 then { s with reps := 1 } else s

/-- `s` as reads, each of one node's gates: copies all alike as one (`alike`), and one copy piece by piece. A copy's
pieces can name different nodes' outputs (a batch member returning leaves of several arguments its members share), and a
read is kept, and its IU found, by its first piece. Copies that shift stay one read, since they lie in one node: they
name only gates of one batch's members (`down` makes them from whole members returning their own gates, and `fromCallee`
offsets a callee's into one member), and copies naming leaves never shift. -/
def Seg.split (s : Seg) : Array Seg :=
  let s := s.alike
  if s.reps == 1 then s.ps.map .one else #[s]

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

/-- The most members, iterations or rows `analyze` walks one by one. A description of a few hundred bytes names `2^52` of
them, so a walk it can't summarize past this is refused (`unwalked`). -/
def WALK : Nat := 2 ^ 20

/-- How a refusal of a walk past `WALK` starts: `evaluate` gives it as `unwalked`, with the rest as its reason. -/
def UNWALKED : String := "unwalked: "

/-- A refusal of walking `n` of `what` one by one, past `WALK`. -/
def walkAtMost (what : String) (n : Nat) : Except String Unit :=
  if n > WALK then throw s!"{UNWALKED}{what}: {n} of them, past the {WALK} this verifier walks one by one" else pure ()

/-- `q` as one row when its rows are consecutive (`rowStride` one row's span): the same elements in the same order. -/
def Piece.flat (q : Piece) : Piece :=
  if q.rows ≤ 1 then q
  else if q.len == 1 then { q with stride := q.rowStride, len := q.rows, rows := 1, rowStride := 0 }
  else if q.rowStride == q.len * q.stride then { q with len := q.rows * q.len, rows := 1, rowStride := 0 }
  else q

/-- Position `j` of a return, as a piece of one element. -/
def retAt (r : PS) (j : Nat) : Option Piece := ((r.slice j (j + 1))[0]?).bind (·.ps[0]?)

/-- Scan step return position `j` followed back through the iterations: `hist[0] = j`, each next position the carry leaf
(parameter 0) the one before returns, which the iteration before returns; then `exit`, the piece `hist`'s last position
returns when it is no carry leaf, else none and the next position is `hist[cyc]`, the positions a cycle. -/
structure Chain where
  hist : Array Nat
  exit : Option Piece
  cyc : Nat := 0

/-- Position `j` of step return `r`, followed (`Chain`) over at most `WALK` positions. -/
def chainOf (r : PS) (j : Nat) : Except String Chain := do
  let mut hist := #[j]
  let mut seen : Std.HashMap Nat Nat := ({} : Std.HashMap Nat Nat).insert j 0
  for _ in [0:WALK] do
    let some q := retAt r hist.back! | throw "program"
    match q.tgt with
    | .ref true 0 =>
      let nx := q.base.toNat
      match seen[nx]? with
      | some s => return { hist, exit := none, cyc := s }
      | none =>
        seen := seen.insert nx hist.size
        hist := hist.push nx
    | _ => return { hist, exit := some q }
  throw s!"{UNWALKED}a scan's carry leaves passed from iteration to iteration: more than {WALK} of them"

/-- Position `t` of the chain: `hist[t]`, or past it around the cycle. -/
def Chain.at (ch : Chain) (t : Nat) : Nat :=
  if t < ch.hist.size then ch.hist[t]! else ch.hist[ch.cyc + (t - ch.cyc) % (ch.hist.size - ch.cyc)]!

/-- Return positions `[a, b)` of `r` as runs, each `(lo, hi, id)`: `id` when the step returns the run in place as its
carry (parameter 0's leaves `[lo, hi)`), which every iteration then passes through from the scan's initial carry. -/
def carryRuns (r : PS) (a b : Nat) : Array (Nat × Nat × Bool) := Id.run do
  let mut out := #[]
  let mut p := a
  for s in r.slice a b do
    if s.reps > 1 then
      out := out.push (p, p + s.count, false)
      p := p + s.count
    else
      for q in s.ps do
        let id := q.tgt == .ref true 0 && q.rows ≤ 1 && q.base == (p : Int) && (q.stride == 1 || q.len ≤ 1)
        out := out.push (p, p + q.count, id)
        p := p + q.count
  return out

/-- Positions of an argument given as a piece (`base`, `stride`, `rows`, `rowStride` over the argument's positions, its
target unread), as one piece of the body when the argument is one piece of one row holding them all. -/
def throughPS (ap : PS) (q : Piece) : Option Piece :=
  match ap.segs with
  | #[s] => match s.ps with
    | #[p] =>
      if s.reps > 1 || p.rows > 1 || q.count == 0 then none else
      let first : Int := q.base + (if q.stride < 0 then ((q.len - 1 : Nat) : Int) * q.stride else 0) +
        (if q.rowStride < 0 then ((q.rows - 1 : Nat) : Int) * q.rowStride else 0)
      let hi : Int := q.base + (if q.stride > 0 then ((q.len - 1 : Nat) : Int) * q.stride else 0) +
        (if q.rowStride > 0 then ((q.rows - 1 : Nat) : Int) * q.rowStride else 0)
      if first < 0 || hi ≥ (p.len : Int) then none
      else some { tgt := p.tgt, base := p.base + q.base * p.stride, stride := q.stride * p.stride, len := q.len,
                  rows := q.rows, rowStride := q.rowStride * p.stride }
    | _ => none
  | _ => none

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
      let q := q.flat
      let node ← c.node k
      let o : Int := c.body.offsets[k]!
      let callee := c.P.defs[node.fn]!
      let byRow (_ : Unit) : Except String (Array Seg) := do
        walkAtMost "rows of a read resolved row by row" q.rows
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
              return #[.one (q.affine w (o + g.base) (if L == 1 then G else g.stride))]
        if q.rows > 1 then byRow ()
        else if q.stride == 1 then
          let a := q.base.toNat
          let b := a + q.len
          -- the whole members between `m0` and `m1` at once, the cut ones at either end each through its return
          let (m0, m1) := ((a + L - 1) / L, b / L)
          let ends (mid : Array Seg) : Except String (Array Seg) := do
            let mut out := #[]
            if a < m0 * L then out := out ++ (← fromCallee c fuel k (a / L) (a % L) L)
            out := out ++ mid
            if m1 * L < b then out := out ++ (← fromCallee c fuel k m1 0 (b - m1 * L))
            return out
          -- a member returns gates only: copies of one member's return
          let gatesOnly := r.segs.all fun s => s.reps ≤ 1 && s.ps.all fun g => match g.tgt with
            | .gate _ => true
            | _ => false
          if gatesOnly && m0 + 1 < m1 then
            return ← ends #[{ ps := shiftGates (o + m0 * G) (r.segs.flatMap (·.ps)), len := L, reps := m1 - m0, shift := G }]
          -- a member returns leaves of arguments the members share: every member returns the same, resolved once
          let shared := r.segs.all fun s => s.reps ≤ 1 && s.ps.all fun g => match g.tgt with
            | .ref true i => (node.args[i]?.map (·.dims.isNone)).getD false
            | _ => false
          if shared && m0 + 1 < m1 then
            let one ← fromCallee c fuel k m0 0 L
            if one.all (·.reps ≤ 1) then
              return ← ends #[{ ps := one.flatMap (·.ps), len := L, reps := m1 - m0, shift := 0 }]
          -- a member returns one run of a mapped argument's leaves within one row of its axis: the whole members' are
          -- that run in each member's row, one piece of rows
          if m0 + 1 < m1 && r.segs.size == 1 && r.segs[0]!.reps ≤ 1 && r.segs[0]!.ps.size == 1 then
            let g := r.segs[0]!.ps[0]!
            if let .ref true i := g.tgt then
              if let some (d, Li) := (node.args[i]?.bind (·.dims)) then
                let lo := g.base.toNat
                if Li > 0 && g.rows ≤ 1 && (g.stride == 1 || g.len ≤ 1) && g.len == L && lo / Li == (lo + L - 1) / Li then
                  let qb := lo / Li
                  let p : Piece := { tgt := .gate 0, base := (((qb * d + m0) * Li + (lo - qb * Li) : Nat) : Int),
                                     stride := 1, len := L, rows := m1 - m0, rowStride := (Li : Int) }
                  if let some bp := throughPS (c.arg k i) p then
                    return ← ends (← down c fuel bp)
          walkAtMost "members of a batch read one by one" ((b - 1) / L + 1 - a / L)
          let mut out := #[]
          for m in [a / L:(b - 1) / L + 1] do
            out := out ++ (← fromCallee c fuel k m (max a (m * L) - m * L) (min b ((m + 1) * L) - m * L))
          return out
        else
          -- a stride of whole members reads leaf `j` of every `stride / L`-th member: copies of one member's leaf
          if q.len > 1 && q.stride % (L : Int) == 0 && q.base ≥ 0 then
            let j := q.base.toNat % L
            let m0 := q.base.toNat / L
            let leaf := r.slice j (j + 1)
            if leaf.all (·.reps ≤ 1) then
              let ps := leaf.flatMap (·.ps)
              if ps.all fun g => match g.tgt with | .gate _ => true | _ => false then
                return #[{ ps := shiftGates (o + m0 * G) ps, len := 1, reps := q.len, shift := q.stride / L * G }]
              if ps.all fun g => match g.tgt with
                  | .ref true i => (node.args[i]?.map (·.dims.isNone)).getD false
                  | _ => false then
                let one ← fromCallee c fuel k m0 j (j + 1)
                if one.all (·.reps ≤ 1) then return #[{ ps := one.flatMap (·.ps), len := 1, reps := q.len, shift := 0 }]
          walkAtMost "leaves of a batch's output read one by one" q.len
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
        let .tuple #[ct, y] := callee.ret | throw "a scan step returns (carry, y)"
        let cl := ct.leaves
        let yl := y.leaves
        let mut out := #[]
        if q.stride == 1 then
          -- the final carry's leaves together, then the ys iteration by iteration
          let (a, b) := (q.base.toNat, q.base.toNat + q.len)
          if a < cl then out := out ++ (← carry c fuel k (node.n - 1) a (min b cl))
          if cl < b && yl > 0 then
            let (lo, hi) := (max a cl - cl, b - cl)
            walkAtMost "iterations of a scan whose ys are read one by one" ((hi - 1) / yl + 1 - lo / yl)
            for m in [lo / yl:(hi - 1) / yl + 1] do
              out := out ++ (← fromCallee c fuel k m (cl + max lo (m * yl) - m * yl) (cl + min hi ((m + 1) * yl) - m * yl))
          return out
        walkAtMost "leaves of a scan's output read one by one" q.len
        for t in [0:q.len] do
          let j := (q.at t).toNat
          if j < cl then out := out ++ (← carry c fuel k (node.n - 1) j (j + 1))
          else
            let (m, jj) := ((j - cl) / yl, cl + (j - cl) % yl)
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
      if s.reps > 1 then
        if s.ps.all fun g => match g.tgt with | .gate _ => true | _ => false then
          out := out.push { s with ps := shiftGates o s.ps }
        else
          -- copies naming leaves are all alike (`Seg`): one resolved, repeated
          if s.shift != 0 then throw "program"
          let mut one := #[]
          for g in s.ps do
            match g.tgt with
            | .gate _ => one := one.push (.one { g with base := o + g.base })
            | .ref false _ => throw "program"
            | .ref true i => one := one ++ (← throughArg c fuel k m i g)
          if one.all (·.reps ≤ 1) then out := out.push { ps := one.flatMap (·.ps), len := s.len, reps := s.reps }
          else
            walkAtMost "copies of a member's return read one by one" s.reps
            for _ in [0:s.reps] do out := out ++ one
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
          else out := out ++ (← carry c fuel k (m - 1) lo hi)
        else if i ≤ node.nxs then
          let L := (a.dims.map (·.2)).getD 0
          out := out ++ (← one (m * L + lo) (m * L + hi))
        else out := out ++ (← one lo hi)
      | .batch =>
        for (s, e) in memberSpans a m lo hi do out := out ++ (← one s e)
      | _ => out := out ++ (← one lo hi)
    return out
termination_by structural fuel _ _ _ _ => fuel

/-- Carry leaves `[a, b)` that iteration `m` of scan node `k` returns, followed back through the iterations before it
(`chainOf`) to where they leave the carry and resolved there: a gate, an `xs` or shared parameter of an iteration, or the
scan's initial carry. A run the step returns in place (`carryRuns`) is the initial carry's in every iteration. -/
def carry (c : Ctx) : Nat → Nat → Nat → Nat → Nat → Except String (Array Seg)
  | 0, _, _, _, _ => throw "references nest deeper than this verifier follows"
  | fuel + 1, k, m, a, b => do
    let node ← c.node k
    let r := (c.memo[node.fn]!).ret
    let init (lo hi : Nat) : Except String (Array Seg) :=
      throughArg c fuel k 0 0 { tgt := .ref true 0, base := lo, stride := 1, len := hi - lo }
    let mut out := #[]
    for (lo, hi, id) in carryRuns r a b do
      if id then out := out ++ (← init lo hi)
      else
        walkAtMost "carry leaves of a scan followed one by one" (hi - lo)
        for j in [lo:hi] do
          let ch ← chainOf r j
          let e := ch.hist.size - 1
          if ch.exit.isSome && e ≤ m then out := out ++ (← fromCallee c fuel k (m - e) ch.hist[e]! (ch.hist[e]! + 1))
          else out := out ++ (← init (ch.at (m + 1)) (ch.at (m + 1) + 1))
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

/-- The positions of argument `i` of node `k` its members read, each `used` of its parameter (`argSpans`), all members at
once: per used run (and row of a batch's mapped axis) one piece of rows, a member a row, as a piece of the body
(`throughPS`); none when the argument isn't one piece. -/
def memberRows (c : Ctx) (node : Node) (k i : Nat) (used : Ivs) : Except String (Option (Array Piece)) := do
  let a := node.args[i]!
  let mut out := #[]
  for (lo, hi) in used do
    let rows : Array Piece ← match node.form, a.dims with
      | .scan, _ =>
        let L := (a.dims.map (·.2)).getD 0
        pure #[{ tgt := .gate 0, base := (lo : Int), stride := 1, len := hi - lo, rows := node.members,
                 rowStride := (L : Int) }]
      | .batch, some (d, L) =>
        if L == 0 then pure #[] else do
        let (q0, q1) := (lo / L, (hi + L - 1) / L)
        walkAtMost "rows of a mapped axis a batch member reads" (q1 - q0)
        pure <| (Array.range (q1 - q0)).filterMap fun q' =>
          let q := q0 + q'
          let (s, e) := (max lo (q * L), min hi ((q + 1) * L))
          if s < e then some { tgt := .gate 0, base := ((q * d * L + (s - q * L) : Nat) : Int), stride := 1,
                               len := e - s, rows := node.members, rowStride := (L : Int) } else none
      | _, _ => pure #[]
    for p in rows do
      match throughPS (c.arg k i) p with
      | some bp => out := out.push bp
      | none => return none
  return some out

/-- What scan node `k`'s iterations after the first read of their carry, `used` of its step's parameter 0: return
positions `used` of every iteration but the last, all at once. A run returned in place is the initial carry's. Any other
position is followed back (`Chain`) for every iteration: the initial carry's leaves the chain reaches within the
iterations before it leaves (or around its cycle), and where it leaves: a gate of the node (which `nodeReads` drops), a
shared parameter (one read for every iteration) or an `xs` one (a row per iteration, one piece of rows when the argument
is one piece). -/
def carriesRead (c : Ctx) (k : Nat) (used : Ivs) : Except String (Array Seg) := do
  let node ← c.node k
  let fuel := Flock.Steps.FUEL
  let r := (c.memo[node.fn]!).ret
  let T := node.n - 1
  let init (lo hi : Nat) : Except String (Array Seg) :=
    throughArg c fuel k 0 0 { tgt := .ref true 0, base := lo, stride := 1, len := hi - lo }
  let mut out := #[]
  for (a, b) in used do
    for (lo, hi, id) in carryRuns r a b do
      if id then out := out ++ (← init lo hi)
      else
        walkAtMost "carry leaves of a scan followed one by one" (hi - lo)
        for j in [lo:hi] do
          let ch ← chainOf r j
          let e := ch.hist.size - 1
          -- iteration `t - 1 < T` reads the initial carry's leaf `ch.at t` while `t ≤ e` (always, around a cycle)
          let upto := if ch.exit.isSome then min T e else T
          let mut seen : Std.HashSet Nat := {}
          for t in [1:min upto e + 1] do seen := seen.insert ch.hist[t]!
          if upto > e then
            if upto - e ≥ ch.hist.size - ch.cyc then
              for t in [ch.cyc:ch.hist.size] do seen := seen.insert ch.hist[t]!
            else
              for t in [e + 1:upto + 1] do seen := seen.insert (ch.at t)
          for l in seen.toArray.qsort (· < ·) do out := out ++ (← init l (l + 1))
          -- where it leaves the carry, in iterations `0 .. T - e`
          if let some q := ch.exit then
            if e < T then
              match q.tgt with
              | .gate _ => pure ()
              | .ref false _ => throw "program"
              | .ref true i =>
                if 1 ≤ i && i ≤ node.nxs then
                  let L := ((node.args[i]?.bind (·.dims)).map (·.2)).getD 0
                  let p : Piece := { tgt := .gate 0, base := q.base, stride := 1, len := 1, rows := T - e,
                                     rowStride := (L : Int) }
                  match throughPS (c.arg k i) p with
                  | some bp => out := out ++ (← down c fuel bp)
                  | none =>
                    walkAtMost "iterations of a scan reading an xs row through its carry, one by one" (T - e)
                    for it in [0:T - e] do out := out ++ (← fromCallee c fuel k it ch.hist[e]! (ch.hist[e]! + 1))
                else out := out ++ (← fromCallee c fuel k 0 ch.hist[e]! (ch.hist[e]! + 1))
  return out

/-- What node `k`'s gates read outside the node, resolved in the body: every argument of a primitive (or of a batch of one
with members); for a composite callee, the positions its `used` names, all members at once (`memberRows`, else member by
member) unless every member reads its whole argument; and for a scan, an iteration's carry read from the one before it,
through that one's pass-throughs (`carriesRead`). Copies of one member's return that are all alike are one read, and one
copy's pieces each a read (`Seg.split`). -/
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
    return out.flatMap Seg.split
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
      match ← memberRows c node k i u with
      | some ps => for p in ps do out := out ++ (← down c fuel p)
      | none =>
        walkAtMost "members of a node reading part of a mapped argument, one by one" node.members
        for m in [0:node.members] do out := out ++ (← resolve i (argSpans node i m u))
  if node.form == .scan && node.n ≥ 2 then out := out ++ (← carriesRead c k (used[0]?.getD #[]))
  let lo : Int := c.body.offsets[k]!
  let hi : Int := c.body.offsets[k + 1]!
  -- a read's gates are one node's
  return (out.flatMap Seg.split).filter fun s => match s.ps[0]? with
    | some q => match q.tgt with
      | .gate _ => !(lo ≤ q.base && q.base < hi)
      | _ => true
    | none => false

/-- `p` then `q`, gates of one width in one row each, as one row: `q` the next elements of `p`'s progression. At most
`WALK` elements, so that no run `carryRuns` takes from a joined piece is longer than one it took from the two. -/
def Piece.join (p q : Piece) : Option Piece :=
  match p.tgt with
  | .ref _ _ => none
  | .gate _ =>
    let st := if p.len == 1 then q.base - p.base else p.stride
    if p.tgt == q.tgt && p.rows == 1 && q.rows == 1 && 0 < p.len && 0 < q.len && p.len + q.len ≤ WALK &&
        (q.len == 1 || q.stride == st) && q.base == p.base + p.len * st then
      some { p with stride := st, len := p.len + q.len }
    else none

/-- `p` then `q`, gates of one width, as rows: `q` one row of `p`'s length and stride, one row step after `p`'s last
row. At most `WALK` elements, as `Piece.join`. -/
def Piece.stack (p q : Piece) : Option Piece :=
  match p.tgt with
  | .ref _ _ => none
  | .gate _ =>
    let rs := if p.rows == 1 then q.base - p.base else p.rowStride
    if p.tgt == q.tgt && 0 < p.rows && q.rows == 1 && 0 < p.len && q.len == p.len &&
        (p.len == 1 || q.stride == p.stride) && (p.rows + 1) * p.len ≤ WALK && q.base == p.base + p.rows * rs then
      some { p with rows := p.rows + 1, rowStride := rs }
    else none

/-- The piece of a segment of one copy of one piece. -/
def Seg.single (s : Seg) : Option Piece := if s.reps == 1 && s.ps.size == 1 then s.ps[0]? else none

/-- `out` then `s`, `s` joined by `f` to `out`'s last segment where both are one copy of one piece and `f` joins them. -/
def Seg.joinWith (f : Piece → Piece → Option Piece) (out : Array Seg) (s : Seg) : Array Seg :=
  match out.back?.bind Seg.single, s.single with
  | some p, some q => match f p q with
    | some r => out.pop.push (.one r)
    | none => out.push s
  | _, _ => out.push s

/-- A return's segments with its gates coalesced, the same positions in the same order: runs of a progression joined
(`Piece.join`), then runs of one length and stride at one step stacked as rows (`Piece.stack`). A body unrolled element
by element, one primitive node an element, returns its elements every few gates: one piece, where `down` gives one per
node, and as many segments in every read of them. -/
def coalesce (segs : Array Seg) : Array Seg :=
  (segs.foldl (Seg.joinWith Piece.join) #[]).foldl (Seg.joinWith Piece.stack) #[]

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
  return { ret := PS.of (coalesce ret), used := acc.map fun qs => Ivs.norm (ivsOf qs) }

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

/-- One interval `(lo, hi, w)` of the sweep sorted by `lo`, after `(total, cover)`: its gates past `cover` added. -/
def coverStep (st : Nat × Nat) (x : Nat × Nat × Nat) : Nat × Nat :=
  if x.2.1 ≤ st.2 then st else (st.1 + (x.2.1 - max x.1 st.2) * x.2.2, x.2.1)

/-- The total width of the distinct gates of `ivs` (each `(lo, hi, w)`: gates `[lo, hi)` of width `w`). -/
def distinctWidth (ivs : Array (Nat × Nat × Nat)) : Nat :=
  ((ivs.toList.mergeSort fun x y => x.1 ≤ y.1).foldl coverStep (0, 0)).1

/-- The gates of `qs` as intervals with their widths. -/
def gateIvs (qs : Array Piece) : Array (Nat × Nat × Nat) :=
  let ws := (qs.filterMap fun q => match q.tgt with | .gate w => some w | _ => none).toList.mergeSort (· ≤ ·) |>.eraseDups
  ws.toArray.flatMap fun w => (ivsOf (qs.filter (·.tgt == .gate w))).map fun iv => (iv.1, iv.2, w)

/-- Gates `cnt` copies, `period` apart, of `res` (each `(lo, hi, w)` from `lo`, none past `span ≤ period`). -/
structure Cl where
  lo : Nat
  period : Nat
  cnt : Nat
  span : Nat
  res : Array (Nat × Nat × Nat)
  deriving Inhabited

/-- One segment of `novelWidth`'s: none when it is no segment of copies or names no gate, its copies' intervals when
copies may overlap or go backwards, else one frame. -/
def copiesStep (acc : Array (Nat × Nat × Nat) × Array Cl) (s : Seg) : Array (Nat × Nat × Nat) × Array Cl :=
  if s.reps ≤ 1 then acc else
  let r := gateIvs s.ps
  if r.isEmpty then acc else
  let lo := r.foldl (fun m x => min m x.1) r[0]!.1
  let hi := r.foldl (fun m x => max m x.2.1) 0
  if s.shift ≤ 0 || hi - lo > s.shift.toNat then (acc.1 ++ (Array.range s.reps).flatMap fun m => gateIvs (s.copy m), acc.2)
  else (acc.1, acc.2.push { lo, period := s.shift.toNat, cnt := s.reps, span := hi - lo,
                             res := r.map fun x => (x.1 - lo, x.2.1 - lo, x.2.2) })

/-- Frames of one start, period and count, next to each other once sorted: one frame. -/
def frameStep (frames : Array Cl) (x : Cl) : Array Cl :=
  match frames.back? with
  | some y =>
    if y.lo == x.lo && y.period == x.period && y.cnt == x.cnt then
      frames.pop.push { y with span := max y.span x.span, res := y.res ++ x.res }
    else frames.push x
  | none => frames.push x

/-- Item `j` of `novelWidth`'s sweep as gates: interval `j` of `ivs`, or frame `j - ivs.size`'s copies. -/
def expandIn (ivs : Array (Nat × Nat × Nat)) (frames : Array Cl) (j : Nat) : Array (Nat × Nat × Nat) :=
  if j < ivs.size then #[ivs[j]!]
  else
    let x := frames[j - ivs.size]!
    (Array.range x.cnt).flatMap fun m => x.res.map fun y => (x.lo + m * x.period + y.1, x.lo + m * x.period + y.2.1, y.2.2)

/-- The width of one component of the sweep: copies times one copy's for a frame alone, else of its items' gates. -/
def widthIn (ivs : Array (Nat × Nat × Nat)) (frames : Array Cl) (comp : Array Nat) : Nat :=
  if comp.size == 1 && comp[0]! ≥ ivs.size then
    let x := frames[comp[0]! - ivs.size]!
    x.cnt * distinctWidth x.res
  else distinctWidth (comp.flatMap (expandIn ivs frames))

/-- One item `(st, en, j)` of the sweep sorted by start: a new component once it starts past what the current one reaches. -/
def sweepStep (width : Array Nat → Nat) (acc : Nat × Array Nat × Nat) (it : Nat × Nat × Nat) : Nat × Array Nat × Nat :=
  if !acc.2.1.isEmpty && it.1 ≥ acc.2.2 then (acc.1 + width acc.2.1, #[it.2.2], it.2.1)
  else (acc.1, acc.2.1.push it.2.2, max acc.2.2 it.2.1)

/-- The total width of the distinct gates of `segs`. A segment of copies is copies of one frame; copies of one frame are
counted as copies times one copy's width, alone in their span, and with what else meets their span as gates. -/
def novelWidth (segs : Array Seg) : Nat :=
  let acc := segs.foldl copiesStep (gateIvs (segs.filter (·.reps ≤ 1) |>.flatMap (·.ps)), #[])
  let ivs := acc.1
  let frames := (acc.2.toList.mergeSort fun x y =>
    x.lo < y.lo || (x.lo == y.lo && (x.period < y.period || (x.period == y.period && x.cnt ≤ y.cnt)))).foldl frameStep #[]
  let n := ivs.size
  let items := (ivs.mapIdx fun i x => (x.1, x.2.1, i)) ++
    (frames.mapIdx fun i x => (x.lo, x.lo + (x.cnt - 1) * x.period + x.span, n + i))
  let fin := (items.toList.mergeSort fun x y => x.1 ≤ y.1).foldl (sweepStep (widthIn ivs frames)) (0, #[], 0)
  if fin.2.1.isEmpty then fin.1 else fin.1 + widthIn ivs frames fin.2.1

/-- An input node: its callee is an input primitive (§5.2). A composite's key may read as an input id (`Input16_v1` is a
well-formed static-free composite key), and its gates are computed. -/
def isInputNode (P : Prog) (n : Node) : Bool := let f := P.defs[n.fn]!; f.prim && isInputId f.key

/-- The root body's layout as `NciCheck.evaluate` reads it through `Prog.gate`: one offset per node and one more, from 0,
not decreasing, to the root's gates. The decoder lays every body out so; it is checked here, at the cost of the root's
nodes, because the agreement with `NciCheck.evaluate` rests on it. -/
def rootLaidOut (P : Prog) : Bool :=
  let r := P.defs[P.root]!
  let offs := r.body.offsets
  let K := r.body.nodes.size
  offs.size == K + 1 && offs[0]! == 0 && offs[K]! == r.gates && (List.range K).all fun k => offs[k]! ≤ offs[k + 1]!

/-- No body calls the root, and only the root's holds input nodes (the decoder's refusal of §5.2), so every gate of a
root node that is no input node is a computed gate. -/
def inputsAtRoot (P : Prog) : Bool :=
  (List.range P.defs.size).all fun d =>
    (P.defs[d]!).body.nodes.all fun n => n.fn != P.root && (d == P.root || !isInputNode P n)

/-- Every node calls a Definition decoded before its own (the decoder decodes callees first), so `activations` reads each
callee's count. -/
def calleesFirst (P : Prog) : Bool :=
  (List.range P.defs.size).all fun d => (P.defs[d]!).body.nodes.all fun n => n.fn < d

/-- Every Definition as the decoder gives it (`Flock.Program.getDef`), checked at the cost of the nodes because the
agreement with `NciCheck.evaluate` rests on it, through `Prog.gate`'s walk: the root a composite, a primitive returning one
`Value<w>`, a composite's offsets from 0 to its gates, each node's gates its members' callee's after the ones before, and
an input node called with no arguments. -/
def bodiesLaidOut (P : Prog) : Bool :=
  !(P.defs[P.root]!).prim && (List.range P.defs.size).all fun d =>
    let defn := P.defs[d]!
    if defn.prim then (match defn.ret with | .value _ => true | _ => false)
    else
      let b := defn.body
      b.offsets[0]! == 0 && b.offsets[b.nodes.size]! == defn.gates && (List.range b.nodes.size).all fun k =>
        let n := b.nodes[k]!
        b.offsets[k + 1]! == b.offsets[k]! + n.members * (P.defs[n.fn]!).gates && (!isInputNode P n || n.args.isEmpty)

/-- Per Definition, the activations of one activation of it: itself and, for each composite node, its members' (a
primitive's are none), once per Definition and multiplied through batches and scans. -/
def activations (P : Prog) : Array Nat :=
  (List.range P.defs.size).foldl (fun a d =>
    let defn := P.defs[d]!
    a.push (if defn.prim then 0 else 1 + defn.body.nodes.foldl (fun s n => s + n.members * a[n.fn]!) 0)) #[]

/-- Per Definition, how deep its activations nest: none for a primitive, else one more than its composite callees'. -/
def depths (P : Prog) : Array Nat :=
  (List.range P.defs.size).foldl (fun a d =>
    let defn := P.defs[d]!
    a.push (if defn.prim then 0 else 1 + defn.body.nodes.foldl (fun s n => max s a[n.fn]!) 0)) #[]

/-- A bound on the steps of `Prog.gate`'s walk to an operand, `(A + 1) · (H + 2)` for the root's `A` activations and
depth `H`: each step goes down into an activation, up out of one, or to the scan iteration before, and none returns to
where an earlier one was. -/
def walkBound (P : Prog) : Nat := ((activations P)[P.root]! + 1) * ((depths P)[P.root]! + 2)

/-- Every reference `Flock.Program.resolve` follows from an activation of the walk ends at a gate: the root has no
parameters (the decoder's refusal), and no scan of no iterations has output leaves, which `resolve` refuses to read. A
scan of no iterations carrying a value is refused here wherever it is, read or not; `analyze` would refuse it only where
a gate's operand reaches it. -/
def resolvable (P : Prog) : Bool :=
  (P.defs[P.root]!).body.params.isEmpty && (List.range P.defs.size).all fun d =>
    (P.defs[d]!).prim || (P.defs[d]!).body.nodes.all fun n => n.form != .scan || n.n != 0 || n.out.leaves == 0

/-- The walk is within `Flock.Steps.FUEL`, with room for `Prog.gate`'s descent. Past it a Program of a few hundred bytes
can need more steps than `Prog.gate` takes: a gate reading the carry of a scan of `2^32` iterations, each a scan of
`2^32` identity steps, needs `2^64`. There `NciCheck.evaluate` refuses the gate (`program`), and this check, which
summarizes each Definition once, would give a verdict. The walk also reads every body as the decoder lays it out
(`bodiesLaidOut`), and every reference it resolves ends at a gate (`resolvable`). -/
def walkable (P : Prog) : Bool :=
  calleesFirst P && bodiesLaidOut P && (resolvable P && walkBound P ≤ Flock.Steps.FUEL / 2)

/-- Per IU `u < n`, its number of gates: those of its root nodes, root node `k` (in IU `own[k]`) holding gates
`[offs[k], offs[k + 1])`. -/
def sizes (offs : Array Nat) (own : Array (Option Nat)) (n : Nat) : Array Nat :=
  (List.range own.size).foldl (fun s k => match own[k]! with
    | some u => s.modify u (· + (offs[k + 1]! - offs[k]!))
    | none => s) (Array.replicate n 0)

/-- One entry of θ, a root node (`inp[k]`: node `k` is an input node; `reg[k]`: its callee is bound at registration),
added to the mask of the entries before it. -/
def thetaStep (inp reg mask : Array Bool) (t : Json) : Except String (Array Bool) := do
  let some k := natOf? t | throw "theta-unknown"
  if k ≥ inp.size then throw "theta-unknown"
  if !inp[k]! then throw "theta-not-input"
  if !reg[k]! then throw "theta-not-registered"
  if mask[k]! then throw "theta-duplicate"
  return mask.set! k true

/-- A root node's owner entry `e`, the node an input node (`inp`) or not, holding gates (`held`) or none: `none` for
`"input"` and for a node without gates, `some u` for IU `u`. -/
def ownerStep (inp held : Bool) (e : Json) : Except String (Option Nat) := do
  let o ← if isInputTag e then pure none else match natOf? e with
    | some u => pure (some u)
    | none => throw "owner-entry"
  if held then
    if o.isSome && inp then throw "input-in-iu"
    if o.isNone && !inp then throw "gate-in-no-iu"
  return if held then o else none

/-- Per IU `u < n`, the segments its root nodes read (`reads`) outside it and outside θ's nodes (`theta`). -/
def intoOf (offs : Array Nat) (own : Array (Option Nat)) (theta : Array Bool) (reads : Array (Array Seg)) (n : Nat) :
    Array (Array Seg) :=
  (List.range own.size).foldl (fun into k => match own[k]! with
    | none => into
    | some u => reads[k]!.foldl (fun into s => match s.ps[0]? with
      | none => into
      | some q =>
        let k' := nodeOf offs q.base.toNat
        if s.count == 0 || own[k']! == some u || theta[k']! then into else into.modify u (·.push s)) into)
    (Array.replicate n #[])

/-- `analyze`'s refusal as `evaluate` gives it: a walk past `WALK` as itself (`unwalked: ` and the reason), any other as
`program`. -/
def analyzeCode (e : String) : String := if e.startsWith UNWALKED then e else "program"

def evaluate (j : Json) : Except String Json := do
  let i ← parse j
  let P ← decodeProgram i.program
  if P.orderViolation.isSome then throw "not-topological"
  if !(rootLaidOut P && inputsAtRoot P) then throw "program"
  if !walkable P then throw "program"
  let reads ← match analyze P with
    | .ok r => pure r
    | .error e => throw (analyzeCode e)
  let body := (P.defs[P.root]!).body
  let K := body.nodes.size
  let offs := body.offsets
  let inp := body.nodes.map (isInputNode P)
  let reg := body.nodes.map fun n => isRegisteredId (P.defs[n.fn]!).key
  let theta ← (List.range i.theta.size).foldlM (fun m t => thetaStep inp reg m i.theta[t]!) (Array.replicate K false)
  if i.owner.size != K then throw "owner-length"
  let own ← (List.range K).foldlM
    (fun own k => own.push <$> ownerStep inp[k]! (offs[k + 1]! - offs[k]! > 0) i.owner[k]!) #[]
  let n := iuCount own
  if n > K then throw "empty-iu"
  let used := usedOf own
  if !(List.range n).all used.contains then throw "empty-iu"
  let size := sizes offs own n
  let into := intoOf offs own theta reads n
  let r := (Array.range n).map fun u => (size[u]!, novelWidth into[u]!)
  return verdict (P.defs[P.root]!).gates r (failure i.gamma i.num i.den r)

/-- A refusal of `j` as JSON: `unwalked` with its reason as the `message`, else as `NciCheck.refusalOf` writes it. -/
def refusal (j : Json) (code : String) : Json :=
  if code.startsWith UNWALKED then
    Json.mkObj [("error", Json.str "unwalked"), ("message", Json.str ((code.drop UNWALKED.length).toString))]
  else NciCheck.refusalOf j code

/-- `evaluate`, with a refusal as `refusal` writes it. -/
def check (j : Json) : Json :=
  match evaluate j with
  | .ok r => r
  | .error code => refusal j code

end NciCheck.Desc
