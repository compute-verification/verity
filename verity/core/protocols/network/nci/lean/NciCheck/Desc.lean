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
* `ret`: its return leaves as pieces, each a run `base + t * stride` of the activation's own gates (of one width) or of
  one parameter's leaves (a pass-through);
* `used`: per parameter, the leaves some gate of it reads, as intervals.

A body's references are resolved through its nodes' outputs with the callees' `ret`, and a pass-through back through the
node's arguments, member by member for a batch or scan node only where the callee's summary isn't one run. Per IU, its
reads are those pieces with the IU's own nodes and θ's taken out, and its novel reads the width of their union.

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

/-- `len` elements, `base + t * stride` for `t < len`. -/
structure Piece where
  tgt : Tgt
  base : Int
  stride : Int
  len : Nat
  deriving Inhabited

def Piece.at (q : Piece) (t : Nat) : Int := q.base + t * q.stride

/-- Elements `[a, b)` of `q`. -/
def Piece.sub (q : Piece) (a b : Nat) : Piece := { q with base := q.at a, len := b - a }

/-- The pieces' elements as intervals of naturals (an order-free view), not necessarily disjoint. Pieces of one stride
`s > 1` and one length `n` whose bases run `b, b + 1, …, b + k − 1` (the rows of a transposed strided part) are `n`
intervals `[b + t·s, b + t·s + k)`, one when `k ≥ s`; any other piece of stride `s > 1` is its `n` elements. -/
def ivsOf (qs : Array Piece) : Array (Nat × Nat) := Id.run do
  let mut out := #[]
  let mut strided : Array (Nat × Nat × Nat) := #[]
  for q in qs do
    if q.len == 0 then continue
    if q.stride == 0 || q.len == 1 then out := out.push (q.base.toNat, q.base.toNat + 1)
    else if q.stride == 1 then out := out.push (q.base.toNat, (q.at (q.len - 1)).toNat + 1)
    else if q.stride == -1 then out := out.push ((q.at (q.len - 1)).toNat, q.base.toNat + 1)
    else
      let lo := if q.stride > 0 then q.base else q.at (q.len - 1)
      strided := strided.push (q.stride.natAbs, q.len, lo.toNat)
  let s := strided.qsort fun x y => x.1 < y.1 || (x.1 == y.1 && (x.2.1 < y.2.1 || (x.2.1 == y.2.1 && x.2.2 < y.2.2)))
  let mut i := 0
  for _ in [0:s.size] do
    if i ≥ s.size then break
    let (st, n, b) := s[i]!
    let mut k := 1
    let mut j := i + 1
    for _ in [i + 1:s.size] do
      let (st', n', b') := s[j]!
      if st' != st || n' != n then break
      if b' + 1 == b + k then j := j + 1
      else if b' == b + k then k := k + 1; j := j + 1
      else break
    if k ≥ st then out := out.push (b, b + (n - 1) * st + k)
    else for t in [0:n] do out := out.push (b + t * st, b + t * st + k)
    i := j
  return out

/-- A sequence of pieces with each one's first position. -/
structure PS where
  ps : Array Piece
  starts : Array Nat
  count : Nat
  deriving Inhabited

def PS.of (ps : Array Piece) : PS :=
  let ps := ps.filter (·.len > 0)
  let (starts, count) := ps.foldl (fun (acc : Array Nat × Nat) q => (acc.1.push acc.2, acc.2 + q.len)) (#[], 0)
  { ps, starts, count }

/-- Positions `[lo, hi)`: the first piece by bisection over the starts, then the pieces it spans. -/
def PS.slice (s : PS) (lo hi : Nat) : Array Piece := Id.run do
  if lo ≥ hi || s.ps.isEmpty then return #[]
  let mut a := 0
  let mut b := s.ps.size
  for _ in [0:s.ps.size] do
    if b - a ≤ 1 then break
    let mid := (a + b) / 2
    if s.starts[mid]! ≤ lo then a := mid else b := mid
  let mut out := #[]
  for i in [a:s.ps.size] do
    let st := s.starts[i]!
    if st ≥ hi then break
    let q := s.ps[i]!
    let en := st + q.len
    if en > lo then out := out.push (q.sub (max lo st - st) (min hi en - st))
  return out

/-- A reference part as pieces: an affine part is one, a strided part one per row unless its rows abut. -/
def ofPart : Part → Array Piece
  | .affine p i b c => #[{ tgt := .ref p i, base := b, stride := 1, len := c }]
  | .strided p i b inner os is c =>
    if c == 0 then #[]
    else if inner == 1 then #[{ tgt := .ref p i, base := b, stride := os, len := c }]
    else if os == (inner : Int) * is then #[{ tgt := .ref p i, base := b, stride := is, len := c }]
    else (Array.range ((c + inner - 1) / inner)).map fun (q : Nat) =>
      { tgt := .ref p i, base := b + (q : Int) * os, stride := is, len := min inner (c - q * inner : Nat) }
  | .explicit it => it.foldl (init := #[]) fun acc r =>
    let one : Piece := { tgt := .ref r.p r.i, base := r.j, stride := 1, len := 1 }
    match acc.back? with
    | some q => if q.tgt == .ref r.p r.i && q.stride == 1 && q.at q.len == r.j then acc.pop.push { q with len := q.len + 1 }
      else acc.push one
    | none => acc.push one

def ofSeq (s : Seq) : PS := PS.of (s.parts.foldl (fun acc q => acc ++ ofPart q) #[])

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
def down (c : Ctx) : Nat → Piece → Except String (Array Piece)
  | 0, _ => throw "references nest deeper than this verifier follows"
  | fuel + 1, q => do
    match q.tgt with
    | .gate _ | .ref true _ => return #[q]
    | .ref false k =>
      if q.len == 0 then return #[]
      let node ← c.node k
      let o : Int := c.body.offsets[k]!
      let callee := c.P.defs[node.fn]!
      match node.form with
      | .prim =>
        let .value w := callee.ret | throw "program"
        return #[{ tgt := .gate w, base := o, stride := 0, len := q.len }]
      | .batch =>
        if callee.prim then
          let .value w := callee.ret | throw "program"
          return #[{ tgt := .gate w, base := o + q.base, stride := q.stride, len := q.len }]
        let L := callee.outLeaves
        let G : Int := callee.gates
        let r := (c.memo[node.fn]!).ret
        if L == 0 then throw "program"
        if q.stride == 1 then
          if r.ps.size == 1 then
            let g := r.ps[0]!
            if let .gate w := g.tgt then
              if g.len == L && (L == 1 || G == L * g.stride) then
                let step : Int := if L == 1 then G else g.stride
                let a := q.base.toNat
                return #[{ tgt := .gate w, base := o + a / L * G + g.at (a % L), stride := step, len := q.len }]
          let a := q.base.toNat
          let b := a + q.len
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
        if q.stride == 1 then fromCallee c fuel k 0 q.base.toNat (q.base.toNat + q.len)
        else
          let mut out := #[]
          for t in [0:q.len] do
            out := out ++ (← fromCallee c fuel k 0 (q.at t).toNat ((q.at t).toNat + 1))
          return out
      | .scan =>
        if node.n == 0 then throw "a scan with no iterations has no output to read"
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
def fromCallee (c : Ctx) : Nat → Nat → Nat → Nat → Nat → Except String (Array Piece)
  | 0, _, _, _, _ => throw "references nest deeper than this verifier follows"
  | fuel + 1, k, m, a, b => do
    let node ← c.node k
    let callee := c.P.defs[node.fn]!
    let o : Int := c.body.offsets[k]! + m * callee.gates
    let mut out := #[]
    for g in (c.memo[node.fn]!).ret.slice a b do
      match g.tgt with
      | .gate _ => out := out.push { g with base := o + g.base }
      | .ref false _ => throw "program"
      | .ref true i => out := out ++ (← throughArg c fuel k m i g)
    return out
termination_by structural fuel _ _ _ _ => fuel

/-- Parameter leaves `g` of member (or iteration) `m` of node `k`'s callee, as what node `k`'s arguments name in the body
(§7, up through a parameter), resolved there. -/
def throughArg (c : Ctx) : Nat → Nat → Nat → Nat → Piece → Except String (Array Piece)
  | 0, _, _, _, _ => throw "references nest deeper than this verifier follows"
  | fuel + 1, k, m, i, g => do
    let node ← c.node k
    let some a := node.args[i]? | throw "program"
    let ap := c.arg k i
    let one (lo hi : Nat) : Except String (Array Piece) := do
      let mut out := #[]
      for q in ap.slice lo hi do out := out ++ (← down c fuel q)
      return out
    let elems : Array (Nat × Nat) :=
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
def nodeReads (c : Ctx) (k : Nat) : Except String (Array Piece) := do
  let node ← c.node k
  let callee := c.P.defs[node.fn]!
  let fuel := Flock.Steps.FUEL
  let resolve (i : Nat) (spans : Array (Nat × Nat)) : Except String (Array Piece) := do
    let ap := c.arg k i
    let mut out := #[]
    for (lo, hi) in spans do
      for q in ap.slice lo hi do out := out ++ (← down c fuel q)
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
  return out.filter fun q => match q.tgt with
    | .gate _ => !(lo ≤ q.base && q.base < hi)
    | _ => true

/-- Definition `d`'s summary, its callees' in `memo`. -/
def summarize (P : Prog) (memo : Array Summary) (d : Nat) : Except String Summary := do
  let defn := P.defs[d]!
  if defn.prim then return { ret := default, used := #[] }
  let body := defn.body
  let c : Ctx := { P, memo, d, args := body.nodes.map fun n => n.args.map (ofSeq ·.seq) }
  let mut ret := #[]
  for q in (ofSeq body.ret).ps do ret := ret ++ (← down c Flock.Steps.FUEL q)
  let mut acc : Array (Array Piece) := Array.replicate body.params.size #[]
  for k in [0:body.nodes.size] do
    for q in ← nodeReads c k do
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
def analyze (P : Prog) : Except String (Array (Array Piece)) := do
  let seen := reached P
  let mut memo : Array Summary := #[]
  for d in [0:P.defs.size] do
    memo := memo.push (← if seen[d]! && d != P.root then summarize P memo d else pure default)
  let body := (P.defs[P.root]!).body
  let c : Ctx := { P, memo, d := P.root, args := body.nodes.map fun n => n.args.map (ofSeq ·.seq) }
  let mut reads := #[]
  for k in [0:body.nodes.size] do
    let rs ← nodeReads c k
    if rs.any fun q => match q.tgt with | .gate _ => false | _ => true then throw "program"
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
  let mut into : Array (Array (Nat × Piece)) := Array.replicate n #[]
  for k in [0:K] do
    let some u := own[k]! | continue
    size := size.modify u (· + gatesOf k)
    for q in reads[k]! do
      let .gate w := q.tgt | continue
      if q.len == 0 then continue
      let k' := nodeOf offs q.base.toNat
      if own[k']! == some u || theta[k']! then continue
      into := into.modify u (·.push (w, q))
  let novel (qs : Array (Nat × Piece)) : Nat :=
    let ws := (qs.map (·.1)).qsort (· < ·) |>.toList.eraseDups
    distinctWidth (ws.toArray.flatMap fun w => (ivsOf (qs.filterMap fun (w', q) => if w' == w then some q else none)).map
      fun (lo, hi) => (lo, hi, w))
  let r := (Array.range n).map fun u => (size[u]!, novel into[u]!)
  return verdict (P.defs[P.root]!).gates r (failure i.gamma i.num i.den r)

/-- `evaluate`, with a refusal as `{"error": code}`. -/
def check (j : Json) : Json :=
  match evaluate j with
  | .ok r => r
  | .error code => Json.mkObj [("error", Json.str code)]

end NciCheck.Desc
