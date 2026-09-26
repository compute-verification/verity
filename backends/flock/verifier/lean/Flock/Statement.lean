/-
PROTOCOL.md §16.1, §16.2, §16.5: the circuit statement, set up from its `Tags`, its row leaf (`RowLeaf`), the pinned
circuit file and the public file: Δ, the regions and their claims, the lincheck fold, the statement digest, Σ and `Hello`.
Nothing here evaluates the computation: the region values are the layout's constants, the public digests and outputs, and
hashes of pinned padding constants.
-/
import Flock.RowLeaf
import Flock.Piop
import Flock.Record

namespace Flock

open Lean (Json)

namespace Circuit

/-- The pinned constant column: the constant of the first slot of the first range that is not the mask. -/
def pin (c : Circuit) (leaf : RowLeaf) : V Nat :=
  match c.ranges.find? (·.kind != Kind.mask) with
  | some r => match r.kind with
    | .comp => pure (r.start + leaf.comp.spec.zConst)
    | .net i => pure (r.start + (c.nets.getD i default).2.constPos)
    | .mask => throw "no pin"
  | none => throw "circuit: no range with a constant"

/-- Δ (§16.1), `(row, col)` pairs XORed into A and into B, in the order the statement digest binds: per range and slot,
the slot's constant wired to the pin, then the row leaf's own entries for its slots; then every unit's leaf bits copied from
the message bits that hash them (a zero leaf from a forced-zero row); then every META wire. -/
def delta (c : Circuit) (leaf : RowLeaf) (pin : Nat) : V (Array (Nat × Nat) × Array (Nat × Nat)) := do
  let mut da : Array (Nat × Nat) := #[]
  let mut db : Array (Nat × Nat) := #[]
  for r in c.ranges do
    for s in [0:r.count] do
      let base := r.slot s
      let konst := match r.kind with
        | .comp => some (base + leaf.comp.spec.zConst)
        | .net i => some (base + (c.nets.getD i default).2.constPos)
        | .mask => none
      if let some k := konst then
        if k != pin then
          da := da.push (k, k) |>.push (k, pin)
          db := db.push (k, k) |>.push (k, pin)
      if r.kind == Kind.comp then
        let (la, lb) ← leaf.slotDelta c r s
        da := da ++ la
        db := db ++ lb
  let unit := c.unit
  let ur := c.unitRange
  let leafCols := (portCols #[unit.inGroups.getD 0 default]).map (·.1)
  let starts := (c.ports.foldl (fun (acc : Array Nat × Nat) p => (acc.1.push acc.2, acc.2 + p.words)) (#[], 0)).1
  for k in [0:ur.count] do
    let (g, u) := (k / c.upv, k % c.upv)
    let base := ur.slot k
    let li := c.leavesIn.getD u #[]
    for j in [0:li.size] do
      let lf := li[j]!
      for t in [0:16] do
        let src ← if lf < 0 then pure (base + unit.useful) else do
          let x := lf.toNat
          let some p := (List.range starts.size).reverse.find? (fun q => starts[q]! ≤ x) | throw "a leaf outside the ports"
          pure ((← leaf.messageCol c g p (2 * (x - starts[p]!))) + t)
        let i := base + leafCols.getD j 0 + t
        da := da.push (i, i) |>.push (i, src)
        db := db.push (i, i) |>.push (i, src)
  for g in [0:c.g] do
    for w in c.wires do
      let dc := (portCols (c.nets.getD w.dst.1 default).2.inGroups).getD w.dst.2.2 default
      let sc := (portCols (c.nets.getD w.src.1 default).2.outGroups).getD w.src.2.2 default
      let dst := c.slotCol w.dst.1 g w.dst.2.1 + dc.1
      let src := c.slotCol w.src.1 g w.src.2.1 + sc.1
      for t in [0:dc.2] do
        da := da.push (dst + t, dst + t) |>.push (dst + t, src + t)
        db := db.push (dst + t, dst + t) |>.push (dst + t, src + t)
  return (da, db)

/-- The output region: every unit slot's returned-output words (the public outputs; pinned padding outputs for padding
instances). -/
def outRegion (c : Circuit) (pub : Public) : V Region := do
  let unit := c.unit
  let ur := c.unitRange
  let g0 := unit.outGroups.getD 0 default
  let nOut := c.outWords
  let bits := fun (lo n : Nat) => (Array.range n).map (lo + ·)
  mkRegion c.kLog "out" (bits 0 (7 + Nat.log2 g0.words) ++ bits ur.slotLog (Nat.log2 ur.count)) (ur.start + g0.col * WORD) fun b => Id.run do
    let mut out := ByteArray.empty
    for k in [0:ur.count] do
      let (g, u) := (k / c.upv, k % c.upv)
      let mut w : Array F128 := Array.replicate g0.words 0
      let mut col := 0
      let lo := c.leavesOut.getD u #[]
      for j in [0:lo.size] do
        let o := lo[j]!
        let v := match c.instOf pub.n b g with
          | some i => pub.outputs.getD (i * nOut + o) 0
          | none => c.dummyOutputs.getD o 0
        let wi := col / WORD
        let sh := col % WORD
        let x := w[wi]!
        w := w.set! wi (if sh < 64 then { x with lo := x.lo ||| (v.toUInt64 <<< sh.toUInt64) }
                        else { x with hi := x.hi ||| (v.toUInt64 <<< (sh - 64).toUInt64) })
        col := col + g0.bits.getD j 0
      for x in w do out := F128.pushBytes out x
    return out

end Circuit

/-- The coin source named in `Hello` (§7). -/
inductive Coins where
  | os
  | seed
  deriving BEq, Inhabited

/-- Everything a session needs about the statement. -/
structure Stmt where
  tags : Tags
  leaf : RowLeaf
  c : Circuit
  pub : Public
  n : Nat
  blocksReal : Nat
  nbl : Nat
  m : Nat
  mPts : Nat
  pin : Nat
  regions : Array Region
  da : Array (Nat × Nat)
  db : Array (Nat × Nat)
  digest : ByteArray
  sigma : ByteArray
  hello : String
  deriving Inhabited

def helloOf (tags : Tags) (sigma : ByteArray) (mPts : Nat) (coins : Coins) : V String :=
  let base := [("profile", Json.str "fast100"), ("reps", Lean.toJson (2 : Nat)), ("flavor", Json.str "rs"),
    ("tables", Json.arr #[Json.str tags.table]),
    ("link", Json.mkObj [("sigma", Json.str (hexOf sigma)), ("m", Lean.toJson mPts), ("points", Lean.toJson (2 : Nat))])]
  canon (Json.mkObj (if coins == .seed then base ++ [("coins", Json.str tags.coinScheme)] else base))

/-- `setup` (§3 step 1): parse and check everything, then derive the statement. -/
def Stmt.setup (tags : Tags) (leaf : RowLeaf) (circuitFile publicFile : ByteArray) (coins : Coins) : V Stmt := do
  let c ← Circuit.parse tags circuitFile
  leaf.checkLayout c
  let pub ← Public.load tags c publicFile
  leaf.checkPublic c pub
  Public.checkOutputs c pub
  let n := pub.n
  let blocksReal := (n + c.g - 1) / c.g
  let nbl := max 3 (log2ceil (max blocksReal 1))
  let m := c.kLog + nbl
  if c.kLog > 26 || m > 35 then throw s!"m = {m}: no fast100 schedule (or k_log > 26)"
  let pin ← c.pin leaf
  let (da, db) ← c.delta leaf pin
  let regions := (← leaf.regions c pub).push (← c.outRegion pub)
  let mut h := tags.statement.toUTF8 ++ leaf.comp.digest ++ c.sha ++ leaf.key c ++ (← canon tags.identity).toUTF8
  for x in [n, blocksReal, nbl, m, c.kLog, pin, regions.size, c.g] do h := pushLe64 h x
  for d in [da, db] do
    h := pushLe64 h d.size
    for (i, cc) in d do h := pushLe32 (pushLe32 h i) cc
  let digest := Sha256.hash h
  let sigma := Sha256.hash (tags.sigmaTag.toUTF8 ++ c.sha ++ digest ++ pub.sha)
  let hello ← helloOf tags sigma (PT_LOCAL + nbl) coins
  return { tags, leaf, c, pub, n, blocksReal, nbl, m, mPts := PT_LOCAL + nbl, pin, regions, da, db, digest, sigma, hello }

namespace Stmt

def spec (st : Stmt) : SessionSpec :=
  { table := st.tags.table, domain := st.tags.domain, sigma := st.sigma, hello := st.hello, mPts := st.mPts }

/-- `e[s + 64·u] = lw[s] · eq(ρ, u)` for `u < 2^|ρ|`. -/
def tensor (lw : Array F128) (rho : Array F128) : Array F128 := Id.run do
  let eIn := eqTable rho
  let mut e : Array F128 := Array.mkEmpty (64 * eIn.size)
  for eu in eIn do
    for l in lw do e := e.push (l * eu)
  return e

/-- The lincheck fold (§10, §16.1): `α·A_0ᵀe + B_0ᵀe` for `e = lw ⊗ eq(ρ)`. A slot of `2^sl` bits at position `q` sees
`e[(q << sl) | c] = e_lo[c] · eq(ρ[sl−6 ..], q)` with `e_lo = lw ⊗ eq(ρ[.. sl−6])`, so each slot type is folded once on
`e_lo` and scaled per slot; Δ's entries are added on `e`. -/
def fold (st : Stmt) (alpha : F128) (lw : Array F128) (rho : Array F128) : V (Array F128) := do
  let c := st.c
  let kl := c.kLog
  if rho.size + 6 != kl || lw.size != 64 then throw "lincheck fold: the point is not the block's"
  let mut out : Array F128 := Array.replicate (2 ^ kl) 0
  let mut bases : Array ((Kind × Nat) × Array F128) := #[]
  for r in c.ranges do
    let sl := r.slotLog
    let key := (r.kind, sl)
    let base ← match bases.find? (·.1 == key) with
      | some (_, b) => pure b
      | none => do
        let eLo := tensor lw (rho.extract 0 (sl - 6))
        let sparse ← match r.kind with
          | .comp =>
            if sl != st.leaf.comp.spec.kLog then throw "lincheck fold: a compression slot of the wrong size"
            pure (some (st.leaf.comp.a, st.leaf.comp.b))
          | .net i =>
            let net := (c.nets.getD i default).2
            if net.unitLog != sl then throw "lincheck fold: a unit slot of the wrong size"
            pure (some (net.a, net.b))
          | .mask => pure none
        let b := match sparse with
          | some (a, b) =>
            let fa := a.foldT eLo
            let fb := b.foldT eLo
            (Array.range (2 ^ sl)).map fun i => alpha * fa[i]! + fb[i]!
          | none => eLo.map fun x => (alpha + 1) * x
        bases := bases.push (key, b)
        pure b
    let eHi := eqTable (rho.extract (sl - 6) rho.size)
    for k in [0:r.count] do
      let q := r.pos0 + k
      let s := eHi[q]!
      let at_ := q * 2 ^ sl
      for i in [0:2 ^ sl] do
        out := out.set! (at_ + i) (s * base[i]!)
  let eIn := eqTable rho
  let e := fun (i : Nat) => lw[i % 64]! * eIn[i / 64]!
  for (i, cc) in st.da do out := out.modify cc (· + alpha * e i)
  for (i, cc) in st.db do out := out.modify cc (· + e i)
  return out

def circuitFold (st : Stmt) : CircuitFold := { kLog := st.c.kLog, pin := st.pin, fold := st.fold }

/-- The region's point in `F^m` for a link point `r ∈ F^{m_pts}` (§16.2). -/
def regionPoint (st : Stmt) (reg : Region) (r : Array F128) : Array F128 := Id.run do
  let kl := st.c.kLog
  let mut p : Array F128 := (Array.range st.m).map fun j =>
    if j < kl && (reg.mask >>> j) % 2 == 1 && (reg.fixed >>> j) % 2 == 1 then 1 else 0
  for t in [0:reg.free.size] do p := p.set! reg.free[t]! r[t]!
  for i in [0:st.nbl] do p := p.set! (kl + i) r[PT_LOCAL + i]!
  return p

/-- `Σ_b eq(r[PT_LOCAL ..], b) · Σ_i eq(r[.. |free|], i) · bit_i(bytes(region, b))`. -/
def regionValue (st : Stmt) (reg : Region) (r : Array F128) : V F128 := do
  let nf := reg.free.size
  let wt := eqTable (r.extract 0 nf)
  let eb := eqTable (r.extract PT_LOCAL (PT_LOCAL + st.nbl))
  let mut acc : F128 := 0
  for b in [0:2 ^ st.nbl] do
    let bytes := reg.bytes b
    if bytes.size * 8 != 2 ^ nf then throw s!"region {reg.name}: its bytes are not its free bits"
    let mut s : F128 := 0
    for i in [0:bytes.size] do
      let x := bytes.get! i
      for t in [0:8] do
        if (x >>> t.toUInt8) % 2 == 1 then s := s + wt[8 * i + t]!
    acc := acc + eb[b]! * s
  return acc

/-- The extra claims (§16.2): region-major, then point 0 before point 1. -/
def extraClaims (st : Stmt) (pts : Array (Array F128)) : V (Array Claim) := do
  let mut out := #[]
  for reg in st.regions do
    for r in pts do
      let p := st.regionPoint reg r
      out := out.push { value := ← st.regionValue reg r, skip := eqTable (p.extract 0 6), x := p.extract 6 p.size }
  return out

end Stmt

end Flock
