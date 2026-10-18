import Std.Data.HashMap
import Flock.Program
import RemoteAudit.Property.Table
import RemoteAudit.Property.Refs

/-!
# property/v1's pass: run dependence, ranges and classes

`stacking._Pass` over the decoded descriptor. A Definition is walked once per (Definition, its parameters' run
dependence, their ranges) and its tally multiplied through `batch` members and `scan` iterations, so the pass costs
O(the circuit's description), never O(the unrolled circuit). A `scan`'s carry is iterated to a fixpoint, exactly up to
2^16 iterations and then any word. A range is a value's signed interval at its width, `none` any word of it, `EMPTY` the
union's identity; a collection's ranges are segments `[lo, hi)` covering it, at most `MAX_SEGS` for a return.

What a gate computes (`prim`): a constant its value; an s8-family core step its D when D's bound lies in s32 from its
operands' ranges (proven, free), else an unproven step at its site; an affine map its exact result when it never wraps
(free); a gate `RANGES` lists the range its rule derives; every other listed gate one unit of its class when an operand
depends on a run input, a gather its index bits (counted once per distinct index and kind in a body, by `note`) and one
`gathers` unit more when its column depends on one too. An unlisted primitive is refused.

This verdict reads no port declarations: a Definition other than the root that declares ports is refused (`ports`), and
so is a root Call that passes one of its arguments through (`Root`).
-/

namespace RemoteAudit.Property

open Flock.Program

abbrev Rng := Option (Int × Int)
def EMPTY : Rng := some (1, 0)
abbrev Segs := Array (Nat × Nat × Rng)

def MAX_SEGS : Nat := 256
def EXACT_ITERATIONS : Nat := 2 ^ 16

def union (a b : Rng) : Rng :=
  match a, b with
  | none, _ => none
  | _, none => none
  | some (a0, a1), some (b0, b1) =>
    if a0 > a1 then b else if b0 > b1 then a else some (min a0 b0, max a1 b1)

/-- Both ranges at once; any word when they share no value. -/
def meet (a b : Rng) : Rng :=
  match a, b with
  | none, _ => b
  | _, none => a
  | some (a0, a1), some (b0, b1) => let lo := max a0 b0; let hi := min a1 b1; if lo ≤ hi then some (lo, hi) else none

def clip (r : Rng) (bound : Int × Int) : Int × Int :=
  match r with
  | some (r0, r1) => if r0 > r1 then bound else (max r0 bound.1, min r1 bound.2)
  | none => bound

def within (r bound : Int × Int) : Bool := bound.1 ≤ r.1 && r.2 ≤ bound.2

/-- An operand's range in its format; a signed range reaching below 0, read as unsigned, is the whole format. -/
def operand (r : Rng) (fmt : Int × Int) : Int × Int :=
  match r with
  | some (r0, _) => if fmt.1 ≥ 0 && r0 < 0 then fmt else clip r fmt
  | none => clip r fmt

def point (r : Rng) : Option Int := match r with
  | some (a, b) => if a == b then some a else none
  | none => none

def signed (w v : Nat) : Int := if v >>> (w - 1) != 0 then (v : Int) - (2 ^ w : Nat) else v

def F32_POS : Int × Int := (0, 0x7F800000)

private def f32Pos (r : Rng) : Bool := match r with
  | some (a, b) => a ≤ b && within (a, b) F32_POS
  | none => false

/-- `fmaxf` of f32 bit patterns read as s32. -/
def fmaxf (a b : Rng) : Rng :=
  match a, b, f32Pos a, f32Pos b with
  | some (a0, a1), some (b0, b1), true, true => some (max a0 b0, max a1 b1)
  | some (a0, _), _, true, false => some (a0, F32_POS.2)
  | _, some (b0, _), false, true => some (b0, F32_POS.2)
  | _, _, _, _ => none

/-- `fminf` of f32 bit patterns read as s32. -/
def fminf (a b : Rng) : Rng :=
  match a, b, f32Pos a, f32Pos b with
  | some (a0, a1), some (b0, b1), true, true => some (min a0 b0, min a1 b1)
  | some (_, a1), _, true, false => some (S32.1, a1)
  | _, some (_, b1), false, true => some (S32.1, b1)
  | _, _, _, _ => none

/-- `int(f32 bits)` truncated toward zero, for a non-negative pattern below 128.0. -/
def trunc (bits : Int) : Int :=
  let b := bits.toNat
  let e := (b >>> 23) % 256
  if e < 127 then 0 else (((b % 2 ^ 23) + 2 ^ 23) >>> (150 - e) : Nat)

/-- `F32ToU32RzByte`'s low byte of the truncation below 128.0; a NaN, -0 or negative float gives 0. -/
def rzByte (x : Rng) : Rng :=
  match x with
  | some (x0, x1) =>
    if x0 > x1 || x1 ≥ 0x43000000 then none
    else some (if x0 ≥ 0 then trunc x0 else 0, if x1 ≥ 0 then trunc x1 else 0)
  | none => none

def EXP2_LUT127 : Int × Int := (0, 127)

/-! ### Tallies -/

/-- A gather index's identity in a body: its kind and its leaves (`_key`). -/
abbrev IKey := String × String

structure Tally where
  /-- budgeted units, by `BUDGETED` -/
  units : Array Nat := #[0, 0, 0, 0]
  refused : Std.HashMap (String × String) Nat := {}
  family : Nat := 0
  macs : Nat := 0
  affine : Nat := 0
  unproven : Std.HashMap String Nat := {}
  /-- index bits read from the parameters, by key: the index's parts, its bits and kind -/
  index : Std.HashMap IKey (Array Part × Nat × GKind) := {}
  deriving Inhabited

def Cls.slot : Cls → Nat
  | .roundings => 0 | .compareSelect => 1 | .gathers => 2 | .folds => 3 | _ => 0

namespace Tally

def unit (t : Tally) (c : Cls) (n : Nat := 1) : Tally := { t with units := t.units.modify c.slot (· + n) }

def refuse (t : Tally) (code fid : String) (n : Nat := 1) : Tally :=
  { t with refused := t.refused.insert (code, fid) (t.refused.getD (code, fid) 0 + n) }

/-- `Tally.add`: `o`'s units, refusals and steps `m` times (its index stays its own). -/
def add (t o : Tally) (m : Nat := 1) : Tally :=
  { t with
    units := (Array.range 4).map fun k => t.units[k]! + m * o.units[k]!
    refused := o.refused.fold (fun acc k n => acc.insert k (acc.getD k 0 + m * n)) t.refused
    unproven := o.unproven.fold (fun acc k n => acc.insert k (acc.getD k 0 + m * n)) t.unproven
    family := t.family + m * o.family, macs := t.macs + m * o.macs, affine := t.affine + m * o.affine }

end Tally

/-! ### The pass -/

structure St where
  memo : Std.HashMap (Nat × Array Bool × Array Rng) (Segs × Bool × Tally) := {}
  sites : Std.HashMap (String × Nat) ((Int × Int) × (Int × Int)) := {}
  bad : Std.HashMap Nat (Array (Option Nat)) := {}
  /-- each primitive's class (`input` for an input) and operand bits -/
  shapes : Std.HashMap String (Option String × Nat) := {}

abbrev PM := StateT St (Except String)

private def refJ (r : Ref) : String := s!"{if r.p then "p" else "n"},{r.i},{r.j}"

/-- `_key`: an index's leaves when it has at most 64, else its non-empty parts. -/
def indexKey (ps : Array Part) : String :=
  let n := partsCount ps
  if n ≤ 64 then
    "L" ++ ";".intercalate ((Array.range n).toList.filterMap fun k => (partsGet ps k).map refJ)
  else
    "J" ++ ";".intercalate ((ps.filter (·.count > 0)).toList.map fun q => match q with
      | .affine p i b c => s!"a,{p},{i},{b},{c}"
      | .strided p i b inner os is_ c => s!"s,{p},{i},{b},{inner},{os},{is_},{c}"
      | .explicit it => "e," ++ ",".intercalate (it.toList.map refJ))

/-- `_dyn`: whether parts read a collection that depends on a run input. -/
def dynOf (ps : Array Part) (pdyn : Array Bool) (dyns : Array Bool) : Bool :=
  (partsColls ps).any fun (p, i) => if p then pdyn.getD i false else dyns.getD i false

def segUnion (segs : Segs) (lo : Nat := 0) (hi : Option Nat := none) : Rng :=
  segs.foldl (fun out (a, b, r) => if b > lo && (match hi with | some h => a < h | none => true) then union out r else out)
    EMPTY

/-- `_read`: the union of the ranges parts read. -/
def readRng (ps : Array Part) (prng : Array Rng) (segs : Array Segs) : Except String Rng := do
  let mut out := EMPTY
  for ((p, i), ivs) in ← partsIvs ps do
    if p then out := union out (prng.getD i none)
    else
      for (lo, hi) in ivs do out := union out (segUnion (segs.getD i #[]) lo (some hi))
  return out

def merge (segs : Segs) : Segs := segs.foldl (fun out (a, b, r) =>
  match out.back? with
  | some (a0, b0, r0) => if r0 == r && b0 == a then out.pop.push (a0, b, r) else out.push (a, b, r)
  | none => out.push (a, b, r)) #[]

/-- `_read_segs`: a body's return in leaf order; an affine run of a node's leaves keeps that node's segments. -/
def readSegs (ret : Seq) (prng : Array Rng) (segs : Array Segs) : Except String Segs := do
  let mut out : Segs := #[]
  let mut pos := 0
  for q in ret.parts do
    let n := q.count
    if n > 0 then
      match q with
      | .affine false idx base _ =>
        for (a, b, r) in segs.getD idx #[] do
          let lo := max a base
          let hi := min b (base + n)
          if lo < hi then out := out.push (pos + lo - base, pos + hi - base, r)
      | q => out := out.push (pos, pos + n, ← readRng #[q] prng segs)
    pos := pos + n
  let merged := merge out
  return if merged.size > MAX_SEGS then #[(0, pos, segUnion merged)] else merged

/-- `_span`: whether leaves `[pos, pos + n)` of `t` and `[base, base + n)` of `ct` have one width, halving where neither
run has one. -/
partial def span (t : Ty) (pos : Nat) (ct : Ty) (base n : Nat) : Bool :=
  let a := widthOf t pos (pos + n)
  let b := widthOf ct base (base + n)
  if a.isSome || b.isSome || n ≤ 1 then a == b
  else let h := n / 2; span t pos ct base h && span t (pos + h) ct (base + h) (n - h)

/-- `_at_width`: whether each reference names a leaf of the width `t` declares at its position. -/
def atWidth (t : Ty) (seq : Seq) (types : Bool → Nat → Ty) : Except String Bool := do
  let mut pos := 0
  for q in seq.parts do
    let n := q.count
    if n == 0 then continue
    let ok ← match q with
      | .affine p i b _ => pure (span t pos (types p i) b n)
      | q => do
        let w := widthOf t pos (pos + n)
        let ivs ← partsIvs #[q]
        let ok1 := w.isSome && ivs.all fun ((p, i), iv) => iv.all fun (lo, hi) => widthOf (types p i) lo hi == w
        pure (ok1 || (Array.range n).all fun k =>
          let r := q.get k
          widthOf t (pos + k) (pos + k + 1) == widthOf (types r.p r.i) r.j (r.j + 1))
    if !ok then return false
    pos := pos + n
  return true

/-- `_reinterpretations`: the nodes of a body with an argument read at another width, and `none` when its return is. -/
def reinterps (P : Prog) (fi : Nat) : PM (Array (Option Nat)) := do
  if let some hit := (← get).bad[fi]? then return hit
  let body := P.defs[fi]!.body
  let types := fun (p : Bool) (i : Nat) => if p then (body.params.getD i ("", .value 1)).2 else (body.nodes.getD i default).out
  let mut out : Array (Option Nat) := #[]
  for k in [0:body.nodes.size] do
    let nd := body.nodes[k]!
    let mut ok := true
    for a in nd.args do
      if !(← atWidth a.ty a.seq types) then ok := false
    if !ok then out := out.push (some k)
  if !(← atWidth body.retType body.ret types) then out := out.push none
  modify fun s => { s with bad := s.bad.insert fi out }
  return out

private def get! {α} [Inhabited α] (xs : Array α) (i : Nat) (what : String) : Except String α :=
  match xs[i]? with
  | some x => pure x
  | none => throw s!"{what}: no operand {i}"

/-- `_affine`: an affine map's exact result range, or none when it isn't one (a product of two non-constants). -/
def affineRng (fid : String) (rng : Array Rng) : Except String (Option (Int × Int)) := do
  if fid == "Bf16ToF32_v1" then
    let x := clip (← get! rng 0 fid) S16
    return some (x.1 * 65536, x.2 * 65536)
  let a ← get! rng 0 fid
  let b ← get! rng 1 fid
  let x := clip a S32
  let y := clip b S32
  if fid == "I32Add_v1" then return some (x.1 + y.1, x.2 + y.2)
  if fid == "I32Sub_v1" then return some (x.1 - y.2, x.2 - y.1)
  let k := match point a with
    | some k => some k
    | none => point b
  let some k := k | return none
  let o := if (point a).isSome then y else x
  return some (min (k * o.1) (k * o.2), max (k * o.1) (k * o.2))

/-- What one gate computes: its range and tally (`_Pass.prim`). -/
def prim (d : Defn) (dyn : Array Bool) (rng : Array Rng) (where_ : String × Nat) : PM (Rng × Tally) := do
  let fid := d.key
  let cls := classOf fid
  if !(← get).shapes.contains fid then
    modify fun s => { s with shapes := s.shapes.insert fid (cls.map (·.name), d.params.foldl (· + bitsAll ·.2) 0) }
  let t : Tally := {}
  let some cls := cls | return (none, t.refuse "unlisted" fid)
  match cls with
  | .const =>
    return ((constOf? fid).map (fun (w, v) => let x := signed w v; (x, x)), t)
  | .family =>
    let some st := family? fid | throw s!"{fid}: a family class without its step"
    let c := clip (← get! rng 0 fid) S32
    let a := operand (← get! rng 1 fid) st.a
    let b := operand (← get! rng 2 fid) st.b
    let corners := [a.1 * b.1, a.1 * b.2, a.2 * b.1, a.2 * b.2]
    let lo := corners.foldl min (a.1 * b.1)
    let hi := corners.foldl max (a.1 * b.1)
    let s : Int × Int := (c.1 + (st.k : Int) * lo, c.2 + (st.k : Int) * hi)
    if within s S32 then return (some s, { t with family := 1, macs := st.k })
    modify fun S => { S with sites := S.sites.insert where_ (match S.sites[where_]? with
      | some (pc, ps) =>
        let u := fun (x y : Int × Int) => match union (some x) (some y) with
          | some r => r
          | none => x
        (u pc c, u ps s)
      | none => (c, s)) }
    return (some (max s.1 S32.1, min s.2 S32.2), { t with unproven := t.unproven.insert fid 1 })
  | cls =>
    let mut cls := cls
    if AFFINE.contains fid then
      if let some r := ← affineRng fid rng then
        if within r S32 then return (some r, { t with affine := 1 })
      if cls == .affine then cls := .roundings
    let g := if cls == .gathers then gather? fid else none
    let mut t := t
    if dyn.any id && (match g with | none => true | some (_, _, _, l) => l) then
      t := t.unit (if g.isSome then .compareSelect else cls)
    if let some (i, _, _, _) := g then
      if dyn.getD i false && (Array.range dyn.size).any (fun j => j != i && dyn[j]!) then t := t.unit .gathers
    let r : Rng ← if fid == "F32Fmaxf_v1" then (if rng.size == 2 then pure (fmaxf rng[0]! rng[1]!) else throw s!"{fid}: 2 operands")
      else if fid == "F32Fminf_v1" then (if rng.size == 2 then pure (fminf rng[0]! rng[1]!) else throw s!"{fid}: 2 operands")
      else if fid == "F32ToU32RzByte_v1" then (if rng.size == 1 then pure (rzByte rng[0]!) else throw s!"{fid}: 1 operand")
      else if fid == "Exp2Lut127_v1" then (if rng.size == 1 then pure (some EXP2_LUT127) else throw s!"{fid}: 1 operand")
      else pure none
    return (r, t)

/-- `note`: a gather index read in a body. One over the parameters goes up to the caller; one computed here counts its
bits once, in the tally of the first node that reads it. -/
def note (ps : Array Part) (bits : Nat) (kind : GKind) (t : Tally) (fresh : Std.HashMap IKey Nat)
    (up : Std.HashMap IKey (Array Part × Nat × GKind)) :
    Tally × Std.HashMap IKey Nat × Std.HashMap IKey (Array Part × Nat × GKind) :=
  let key : IKey := (kind.name, indexKey ps)
  if (partsColls ps).any (·.1) then
    let prev := match up[key]? with
      | some (_, b, _) => b
      | none => 0
    (t, fresh, up.insert key (ps, max bits prev, kind))
  else
    let had := fresh.getD key 0
    if bits > had then (t.unit .gathers (bits - had), fresh.insert key bits, up) else (t, fresh, up)

mutual

/-- One call of Definition `fi`: its return's segments, its run dependence and its tally (`_Pass.call`), memoized. -/
partial def call (P : Prog) (fi : Nat) (dyn : Array Bool) (rng : Array Rng) (where_ : String × Nat) :
    PM (Segs × Bool × Tally) := do
  let d := P.defs[fi]!
  if d.prim then
    let (r, t) ← prim d dyn rng where_
    return (#[(0, 1, r)], dyn.any id, t)
  let key := (fi, dyn, rng)
  if let some hit := (← get).memo[key]? then return hit
  let (segs, dyns, ts, up) ← walk P fi dyn rng d.key
  let out : Tally := { (ts.foldl (fun acc (_, t) => acc.add t) ({} : Tally)) with index := up }
  let ret ← readSegs d.body.ret rng segs
  let hit := (ret, dynOf d.body.ret.parts dyn dyns, out)
  modify fun s => { s with memo := s.memo.insert key hit }
  return hit

/-- Each node of Definition `fi`'s body with the tally of its gates (a batch's and a scan's whole), its output's segments
and run dependence, and the index bits read from the parameters (`_Pass.walk`). -/
partial def walk (P : Prog) (fi : Nat) (pdyn : Array Bool) (prng : Array Rng) (defId : String) :
    PM (Array Segs × Array Bool × Array (Nat × Tally) × Std.HashMap IKey (Array Part × Nat × GKind)) := do
  let body := P.defs[fi]!.body
  let bad ← reinterps P fi
  let mut fresh : Std.HashMap IKey Nat := {}
  let mut up : Std.HashMap IKey (Array Part × Nat × GKind) := {}
  let mut segs : Array Segs := #[]
  let mut dyns : Array Bool := #[]
  let mut out : Array (Nat × Tally) := #[]
  for k in [0:body.nodes.size] do
    let nd := body.nodes[k]!
    let callee := P.defs[nd.fn]!
    if callee.prim && callee.binding.isSome then
      if !(← get).shapes.contains callee.key then
        modify fun s => { s with shapes := s.shapes.insert callee.key (some "input", 0) }
      segs := segs.push #[(0, nd.out.leaves, none)]
      dyns := dyns.push (callee.binding == some RUN)
      continue
    let ad := nd.args.map fun a => dynOf a.seq.parts pdyn dyns
    let ar ← nd.args.mapM fun a => (readRng a.seq.parts prng segs : PM Rng)
    let mut t : Tally := {}
    let mut s : Segs := #[]
    let mut od := false
    if nd.form == .scan then
      (s, od, t) ← scan P nd ad ar defId k
    else
      let (s0, od0, t0) ← call P nd.fn ad ar (defId, k)
      let batch := nd.form == .batch
      s := if batch then #[(0, nd.out.leaves, segUnion s0)] else s0
      od := od0
      t := ({} : Tally).add t0 (if batch then nd.n else 1)
      for (_, ps, bits, kind) in t0.index.toArray do
        let leaves := (Array.range (partsCount ps)).filterMap (partsGet ps)
        let mapped ← leaves.anyM fun r => do
          if !batch then return false
          match nd.axes[r.i]? with
          | some ax => return ax.isSome
          | none => throw s!"definition '{defId}' node {k}: an index names parameter {r.i}"
        if mapped then t := t.unit .gathers (nd.n * bits)
        else
          let items ← leaves.mapM fun r => do
            let some a := nd.args[r.i]? | throw s!"definition '{defId}' node {k}: an index names parameter {r.i}"
            if r.j < a.seq.count then pure (a.seq.get r.j) else throw s!"definition '{defId}' node {k}: index leaf {r.j}"
          (t, fresh, up) := note #[.explicit items] bits kind t fresh up
      if callee.prim then
        if let some (i, bits, kind, _) := gather? callee.key then
          if ad.getD i false then
            if batch && (nd.axes.getD i none).isSome then t := t.unit .gathers (nd.n * bits)
            else
              let some a := nd.args[i]? | throw s!"definition '{defId}' node {k}: a gather without its index"
              (t, fresh, up) := note a.seq.parts bits kind t fresh up
    if bad.contains (some k) then t := t.refuse "reinterpretation" defId
    segs := segs.push s
    dyns := dyns.push od
    out := out.push (k, t)
  if bad.contains none then out := out.push (body.nodes.size - 1, ({} : Tally).refuse "reinterpretation" defId)
  return (segs, dyns, out, up)

/-- A `scan`: its step called with the carry's range until it is a fixpoint, each iteration's tally counted for the
iterations it stands for (`_Pass.scan`). -/
partial def scan (P : Prog) (nd : Node) (ad : Array Bool) (ar : Array Rng) (defId : String) (k : Nat) :
    PM (Segs × Bool × Tally) := do
  let .tuple fs := nd.out | throw s!"definition '{defId}' node {k}: a scan returns (carry, ys)"
  let cl := (fs.getD 0 (.value 1)).leaves
  let d := #[ad.any id] ++ ad.extract 1 ad.size
  let mut carry := ar.getD 0 none
  let rest := ar.extract 1 ar.size
  let mut ys := EMPTY
  let mut out : Tally := {}
  for i in [0:nd.n] do
    if i == EXACT_ITERATIONS then carry := none
    let (s, _, t) ← call P nd.fn d (#[carry] ++ rest) (defId, k)
    let nxt := union carry (segUnion s 0 (some cl))
    ys := union ys (segUnion s cl none)
    let reps := if nxt == carry then nd.n - i else 1
    out := out.add t reps
    if !t.index.isEmpty then out := out.unit .gathers (reps * t.index.fold (fun n _ (_, b, _) => n + b) 0)
    if nxt == carry then break
    carry := nxt
  return (#[(0, cl, carry), (cl, nd.out.leaves, ys)], d.getD 0 false, out)

end

end RemoteAudit.Property
