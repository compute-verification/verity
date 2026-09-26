/-
PROTOCOL.md §16.1, §16.2, §16.5, §16.6: the circuit statement (`verity/flock-netlist/v1`, table `nl`), set up from the pinned
circuit file, the pinned compression circuit and the public file: Δ, the regions and their claims, the lincheck fold,
the statement digest, Σ and `Hello`. Nothing here evaluates the computation: the region values are the layout's constants,
the public digests and outputs, and hashes of pinned padding constants.
-/
import Flock.Circuit
import Flock.Piop
import Flock.Record

namespace Flock

open Lean (Json)

def TAG : String := "verity/flock-netlist/v1"
def SIGMA_TAG : String := "verity/flock-netlist/sigma/v1"
def INST_FORMAT : String := "flock-netlist-inst/v1"
def TABLE : String := "nl"
def PT_LOCAL : Nat := 24
def domainOf (r : Nat) : String := s!"flock-netlist-v1/fast100x2/rep{r}"

/-- What a compression slot computes (block-uniform). -/
inductive CompRole where
  | run (g p c j : Nat)
  | dummyRun (j : Nat)
  | parent (g p k : Nat)
  | dummyParent
  deriving Inhabited, Repr

namespace Circuit

def compRole (c : Circuit) (r : Range) (s : Nat) : CompRole :=
  match r.role with
  | .runs p slots =>
    let nb := (c.ports.getD p default).nb
    match slots.getD (s / nb) none with
    | some (g, ch) => .run g p ch (s % nb)
    | none => .dummyRun (s % nb)
  | .parents p =>
    let ch := (c.ports.getD p default).chunks
    if s % ch == ch - 1 then .dummyParent else .parent (s / ch) p (s % ch)
  | .none => .dummyParent

open Blake3 in
/-- Counter, block length and flags of a compression. -/
def params (c : Circuit) : CompRole → Nat × UInt32 × UInt32
  | .run _ p ch j =>
    let port := c.ports.getD p default
    let fin := j + 1 == port.nb
    (ch, 64, KEYED_HASH ||| (if j == 0 then CHUNK_START else 0)
      ||| (if fin then CHUNK_END ||| (if port.chunks == 1 then ROOT else 0) else 0))
  | .dummyRun j => (0, 64, KEYED_HASH ||| (if j == 0 then CHUNK_START else 0))
  | .parent _ _ k => (0, 64, PARENT ||| KEYED_HASH ||| (if k == 0 then ROOT else 0))
  | .dummyParent => (0, 64, PARENT ||| KEYED_HASH)

/-- Block-local column of the slot holding the last compression of VU `g`'s chunk `ch` of port `p`. -/
def chunkEnd (c : Circuit) (g p ch : Nat) : V Nat := do
  let nb := (c.ports.getD p default).nb
  for r in c.ranges do
    if let .runs rp slots := r.role then
      if rp == p then
        if let some q := slots.findIdx? (· == some (g, ch)) then return r.slot (q * nb + nb - 1)
  throw s!"no run of VU {g} chunk {ch} of port {p}"

def parentsRange (c : Circuit) (p : Nat) : V Range :=
  match c.ranges.find? (fun r => match r.role with | .parents q => q == p | _ => false) with
  | some r => pure r
  | none => throw s!"port {p} has no parent range"

/-- Block-local column of VU `g`'s byte `byte` of port `p`: the message bit where it is hashed. -/
def messageCol (c : Circuit) (g p byte : Nat) : V Nat := do
  let nb := (c.ports.getD p default).nb
  let (ch, rest) := (byte / 1024, byte % 1024)
  let (j, off) := (rest / 64, rest % 64)
  let fin ← c.chunkEnd g p ch
  return fin - (nb - 1 - j) * 2 ^ Comp.kLog + Comp.mBase + 8 * off

/-- The pinned constant column: the constant of the first slot of the first range that is not the mask. -/
def pin (c : Circuit) : V Nat :=
  match c.ranges.find? (·.kind != Kind.mask) with
  | some r => match r.kind with
    | .comp => pure (r.start + Comp.zConst)
    | .net i => pure (r.start + (c.nets.getD i default).2.constPos)
    | .mask => throw "no pin"
  | none => throw "circuit: no range with a constant"

/-- Δ (§16.1): `(row, col)` pairs XORed into A and into B. -/
def delta (c : Circuit) (pin : Nat) : V (Array (Nat × Nat) × Array (Nat × Nat)) := do
  let mut da : Array (Nat × Nat) := #[]
  let mut db : Array (Nat × Nat) := #[]
  for r in c.ranges do
    for s in [0:r.count] do
      let base := r.slot s
      let konst := match r.kind with
        | .comp => some (base + Comp.zConst)
        | .net i => some (base + (c.nets.getD i default).2.constPos)
        | .mask => none
      if let some k := konst then
        if k != pin then
          da := da.push (k, k) |>.push (k, pin)
          db := db.push (k, k) |>.push (k, pin)
      if r.kind != Kind.comp then continue
      match c.compRole r s with
      | .run _ _ _ j | .dummyRun j =>
        if j > 0 then
          for t in [0:256] do
            let i := base + Comp.cvBase + t
            da := da.push (i, i) |>.push (i, base - 2 ^ Comp.kLog + Comp.outLo + t)
      | .parent g p k =>
        let ch := (c.ports.getD p default).chunks
        let pr ← c.parentsRange p
        for (side, h) in [(0, 2 * k + 1), (1, 2 * k + 2)] do
          let child ← if h < ch - 1 then pure (pr.slot (g * ch + h)) else c.chunkEnd g p (h - (ch - 1))
          for t in [0:256] do
            let i := base + Comp.mBase + 256 * side + t
            da := da.push (i, i) |>.push (i, child + Comp.outLo + t)
      | .dummyParent => pure ()
  let unit := c.unit
  let ur := c.unitRange
  let leafCols := (portCols #[unit.inGroups.getD 0 default]).map (·.1)
  let starts := (c.ports.foldl (fun (acc : Array Nat × Nat) p => (acc.1.push acc.2, acc.2 + p.words)) (#[], 0)).1
  for k in [0:ur.count] do
    let (g, u) := (k / c.upv, k % c.upv)
    let base := ur.slot k
    let li := c.leavesIn.getD u #[]
    for j in [0:li.size] do
      let leaf := li[j]!
      for t in [0:16] do
        let src ← if leaf < 0 then pure (base + unit.useful) else do
          let lf := leaf.toNat
          let some p := (List.range starts.size).reverse.find? (fun q => starts[q]! ≤ lf) | throw "a leaf outside the ports"
          pure ((← c.messageCol g p (2 * (lf - starts[p]!))) + t)
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

end Circuit

inductive RegionKind where
  /-- counter, block length and flags of every compression of compression range `ri` -/
  | params (ri : Nat)
  /-- the chaining value of every chain start of compression range `ri` (the key) -/
  | cvIn (ri : Nat)
  /-- the digest of port `p`: the output value of each VU's root compression in range `ri` -/
  | digest (p ri : Nat)
  /-- every unit slot's returned-output words -/
  | out
  deriving Inhabited, Repr

/-- A region (§16.2): `free` in-block bit positions (LSB first); the other bits fixed to `fixed` where `mask` is set. -/
structure Region where
  kind : RegionKind
  free : Array Nat
  mask : Nat
  fixed : Nat
  deriving Inhabited, Repr

def regionsOf (c : Circuit) : V (Array Region) := do
  let kl := c.kLog
  let bits := fun (lo n : Nat) => (Array.range n).map (lo + ·)
  let mk := fun (kind : RegionKind) (free : Array Nat) (fixed : Nat) => do
    let freeMask := free.foldl (fun acc b => acc ||| 2 ^ b) 0
    let mask := (2 ^ kl - 1) &&& ((2 ^ kl - 1) ^^^ freeMask)
    if fixed ≥ 2 ^ kl || fixed &&& freeMask != 0 || free.size > PT_LOCAL || free.any (· ≥ kl) then
      throw s!"a region is not aligned or has more than {PT_LOCAL} free bits"
    pure (⟨kind, free, mask, fixed⟩ : Region)
  let mut out := #[]
  for ri in [0:c.ranges.size] do
    let r := c.ranges[ri]!
    let lr := Nat.log2 r.count
    match r.role with
    | .runs p _ =>
      let port := c.ports.getD p default
      let lnb := Nat.log2 port.nb
      out := out.push (← mk (.params ri) (bits 0 7 ++ bits Comp.kLog lr) (r.start + Comp.tLo))
      out := out.push (← mk (.cvIn ri) (bits 0 8 ++ bits (Comp.kLog + lnb) (lr - lnb)) (r.start + Comp.cvBase))
      if port.chunks == 1 then
        out := out.push (← mk (.digest p ri) (bits 0 8 ++ bits (Comp.kLog + lnb) (lr - lnb))
          (r.start + (port.nb - 1) * 2 ^ Comp.kLog + Comp.outLo))
    | .parents p =>
      let lc := Nat.log2 (c.ports.getD p default).chunks
      out := out.push (← mk (.params ri) (bits 0 7 ++ bits Comp.kLog lr) (r.start + Comp.tLo))
      out := out.push (← mk (.cvIn ri) (bits 0 8 ++ bits Comp.kLog lr) (r.start + Comp.cvBase))
      out := out.push (← mk (.digest p ri) (bits 0 8 ++ bits (Comp.kLog + lc) (lr - lc)) (r.start + Comp.outLo))
    | .none => pure ()
  let unit := c.unit
  let ur := c.unitRange
  let g0 := unit.outGroups.getD 0 default
  out := out.push (← mk .out (bits 0 (7 + Nat.log2 g0.words) ++ bits ur.slotLog (Nat.log2 ur.count)) (ur.start + g0.col * WORD))
  return out

/-- A `flock-netlist-inst/v1` public file (no rows): its header, row digests per instance and port, output words. -/
structure Public where
  header : Json
  n : Nat
  digests : Array ByteArray
  outputs : Array Nat
  /-- SHA-256 of the canonical header (`rows = false`) and every byte after it: what Σ binds -/
  sha : ByteArray
  deriving Inhabited

def Public.load (c : Circuit) (file : ByteArray) : V Public := do
  let some nl := findByte file 10 | throw "public file: no header line"
  let text ← match String.fromUTF8? (file.extract 0 nl) with
    | some s => pure s
    | none => throw "public file: header is not UTF-8"
  let header ← match Json.parse text with
    | .ok j => pure j
    | .error e => throw s!"public file: header: {e}"
  let ctx := "public file"
  if (← J.str header "format" ctx) != INST_FORMAT || (← J.str header "statement" ctx) != TAG then
    throw "public file: not a flock-netlist-inst/v1 file"
  if (← J.str header "netlist_sha256" ctx) != hexOf c.sha then throw "public file: staged for another circuit"
  let n ← J.nat header "instances" ctx
  if (← J.bool header "rows" ctx) then throw "public file: it holds rows (the verifier's public file has none)"
  let np := c.ports.size
  let nOut := c.outWords
  let body := file.extract (nl + 1) file.size
  if body.size != 32 * n * np + 2 * n * nOut then throw "public file: length"
  let digests := (Array.range (n * np)).map fun i => body.extract (32 * i) (32 * i + 32)
  let o := 32 * n * np
  let outputs := (Array.range (n * nOut)).map fun i => (body.get! (o + 2 * i)).toNat + 256 * (body.get! (o + 2 * i + 1)).toNat
  let sha := Sha256.hash ((← canon header).toUTF8 ++ body)
  return { header, n, digests, outputs, sha }

def Public.digest (pub : Public) (i p np : Nat) : ByteArray := pub.digests.getD (i * np + p) ByteArray.empty

/- §16.6: frame-v3 hashing. -/
namespace FV3

def FRAME : ByteArray := "veritor/protocol/merkle/frame/v3".toUTF8.push 0

def hash (tag : String) (parts : List ByteArray) : ByteArray :=
  let t := tag.toUTF8
  Sha256.hash (parts.foldl (fun acc p => acc ++ be64 p.size.toUInt64 ++ p) (FRAME ++ be32 t.size.toUInt32 ++ t))

def uint (x : Nat) : ByteArray :=
  let b := be64 x.toUInt64
  let z := ((List.range 8).find? (fun i => b.get! i != 0)).getD 7
  b.extract z 8

def leaf (dom : ByteArray) (i : Nat) (schema : String) (payload : ByteArray) : ByteArray :=
  hash "leaf" [dom, uint i, uint i, schema.toUTF8, payload]

def root (dom : ByteArray) (leaves : Array ByteArray) : ByteArray := Id.run do
  let n := leaves.size
  let width := nextPow2 n
  let mut level := leaves ++ (Array.range (width - n)).map fun k => hash "pad" [dom, uint (n + k)]
  for depth in [0:log2ceil width] do
    level := (Array.range (level.size / 2)).map fun i => hash "node" [dom, uint depth, uint i, level[2 * i]!, level[2 * i + 1]!]
  return level.getD 0 ByteArray.empty

end FV3

def rowKey (role : String) : ByteArray := Id.run do
  let mut k := "verity/blake3-leaf/v1/".toUTF8 ++ role.toUTF8
  while k.size < 32 do k := k.push 0
  return k.extract 0 32

/-- The frame-v3 roots over the public digests and output words against the header's; the pinned padding digests against
keyed BLAKE3 of zero rows; the row key against the x-row key. -/
def checkPublic (c : Circuit) (pub : Public) : V Unit := do
  let ctx := "public file"
  let fv ← J.get pub.header "frame_v3" ctx
  let get := fun (k p : String) => do unhex (← J.str (← J.get fv k ctx) p s!"{ctx}.frame_v3.{k}")
  if (← unhex (← J.str fv "key" ctx)) != c.key || c.key != rowKey "x" then
    throw "the row key is not the x-row key (verity/blake3-leaf/v1/x)"
  let np := c.ports.size
  for p in [0:np] do
    let port := c.ports[p]!
    let dom ← get "domain_ids" port.name
    if (← J.str (← J.get fv "schemas" ctx) port.name ctx) != "blake3-keyed/row/v2" then
      throw s!"input port {port.name}: schema is not blake3-keyed/row/v2"
    let leaves := (Array.range pub.n).map fun i => FV3.leaf dom i "blake3-keyed/row/v2" (pub.digest i p np)
    if FV3.root dom leaves != (← get "roots" port.name) then
      throw s!"input port {port.name}: the row digests do not hash to the header's frame-v3 root"
    if Blake3.keyedHash c.key (ByteArray.mk (Array.replicate (2 * port.words) 0)) != c.dummyDigests.getD p ByteArray.empty then
      throw s!"the pinned padding digest of port {port.name} is not the zero row's"
  let nOut := c.outWords
  let mut at_ := 0
  for (name, w) in c.outPorts do
    let dom ← get "domain_ids" name
    let words := (Array.range pub.n).foldl (fun acc i => acc ++ (Array.range w).map fun k => pub.outputs.getD (i * nOut + at_ + k) 0) #[]
    let leaves := (Array.range words.size).map fun r =>
      FV3.leaf dom r "u16" (ByteArray.mk #[(words[r]! / 256).toUInt8, (words[r]! % 256).toUInt8])
    if FV3.root dom leaves != (← get "roots" name) then
      throw s!"output port {name}: the output words do not hash to the header's frame-v3 root"
    at_ := at_ + w

/-- The coin source named in `Hello` (§7). -/
inductive Coins where
  | os
  | seed
  deriving BEq, Inhabited

/-- Everything a session needs about the statement. -/
structure Stmt where
  c : Circuit
  comp : Comp
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
  keyWords : Array UInt32
  dummyRunCv : Array (Array UInt32)
  deriving Inhabited

def identity (c : Circuit) : Json :=
  Json.mkObj [("statement", Json.str TAG), ("backend", Json.str "C-interactive"), ("proof_class", Json.str "NON_ZK_PROOF"),
    ("profile", Json.str "fast100"), ("reps", Lean.toJson (2 : Nat)), ("commitments", Json.str "one root, both reps (R1)"),
    ("flavor", Json.str "rs"), ("merkle_hash", Json.str "sha256"), ("leaf_scheme", Json.str c.leafScheme),
    ("hooks", Json.mkObj [("zk", Json.bool false), ("t_pad", Lean.toJson (0 : Nat)), ("masks", Json.str "none"),
      ("sumcheck_mask", Json.str "none"), ("claim_permutation", Json.str "identity")])]

def helloOf (sigma : ByteArray) (mPts : Nat) (coins : Coins) : V String :=
  let base := [("profile", Json.str "fast100"), ("reps", Lean.toJson (2 : Nat)), ("flavor", Json.str "rs"),
    ("tables", Json.arr #[Json.str TABLE]),
    ("link", Json.mkObj [("sigma", Json.str (hexOf sigma)), ("m", Lean.toJson mPts), ("points", Lean.toJson (2 : Nat))])]
  canon (Json.mkObj (if coins == .seed then base ++ [("coins", Json.str "coin-commit/sha256/v1")] else base))

/-- `setup` (§3 step 1): parse and check everything, then derive the statement. -/
def Stmt.setup (circuitFile compFile publicFile : ByteArray) (coins : Coins) : V Stmt := do
  let c ← Circuit.parse circuitFile
  let comp ← Comp.load compFile
  let pub ← Public.load c publicFile
  checkPublic c pub
  let n := pub.n
  let blocksReal := (n + c.g - 1) / c.g
  let nbl := max 3 (log2ceil (max blocksReal 1))
  let m := c.kLog + nbl
  if c.kLog > 26 || m > 35 then throw s!"m = {m}: no fast100 schedule (or k_log > 26)"
  let pin ← c.pin
  let (da, db) ← c.delta pin
  let regions ← regionsOf c
  let keyWords := Blake3.wordsOfBytes c.key 0 8
  let dummyRunCv := c.ports.map fun p => Id.run do
    let mut cv := keyWords
    for j in [0:p.nb] do
      let (ctr, bl, f) := c.params (.dummyRun j)
      cv := (Blake3.compress cv (Array.replicate 16 0) ctr.toUInt64 bl f).extract 0 8
    return cv
  let mut h := TAG.toUTF8 ++ comp.digest ++ c.sha ++ c.key ++ (← canon (identity c)).toUTF8
  for x in [n, blocksReal, nbl, m, c.kLog, pin, regions.size, c.g] do h := pushLe64 h x
  for d in [da, db] do
    h := pushLe64 h d.size
    for (i, cc) in d do h := pushLe32 (pushLe32 h i) cc
  let digest := Sha256.hash h
  let sigma := Sha256.hash (SIGMA_TAG.toUTF8 ++ c.sha ++ digest ++ pub.sha)
  let hello ← helloOf sigma (PT_LOCAL + nbl) coins
  return { c, comp, pub, n, blocksReal, nbl, m, mPts := PT_LOCAL + nbl, pin, regions, da, db, digest, sigma, hello,
           keyWords, dummyRunCv }

namespace Stmt

def spec (st : Stmt) : SessionSpec :=
  { table := TABLE, domain := domainOf, sigma := st.sigma, hello := st.hello, mPts := st.mPts }

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
        let b ← match r.kind with
          | .comp =>
            if sl != Comp.kLog then throw "lincheck fold: a compression slot of the wrong size"
            let fa := st.comp.a.foldT eLo
            let fb := st.comp.b.foldT eLo
            pure ((Array.range (2 ^ sl)).map fun i => alpha * fa[i]! + fb[i]!)
          | .net i =>
            let net := (c.nets.getD i default).2
            if net.unitLog != sl then throw "lincheck fold: a unit slot of the wrong size"
            let fa := net.a.foldT eLo
            let fb := net.b.foldT eLo
            pure ((Array.range (2 ^ sl)).map fun i => alpha * fa[i]! + fb[i]!)
          | .mask => pure (eLo.map fun x => (alpha + 1) * x)
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

def instOf (st : Stmt) (b g : Nat) : Option Nat :=
  let i := b * st.c.g + g
  if i < st.n then some i else none

def cvBytes (cv : Array UInt32) : ByteArray := Blake3.bytesOfWords cv 8

/-- Block `b`'s known bytes of a region, in free-assignment order. -/
def regionBytes (st : Stmt) (reg : RegionKind) (b : Nat) : ByteArray := Id.run do
  let c := st.c
  let np := c.ports.size
  match reg with
  | .params ri =>
    let r := c.ranges.getD ri default
    let mut out := ByteArray.empty
    for s in [0:r.count] do
      let (ctr, bl, f) := c.params (c.compRole r s)
      out := out ++ le64 ctr.toUInt64 ++ le32 bl ++ le32 f
    return out
  | .cvIn ri =>
    let r := c.ranges.getD ri default
    let starts := match r.role with | .runs _ slots => slots.size | _ => r.count
    let mut out := ByteArray.empty
    for _ in [0:starts] do out := out ++ c.key
    return out
  | .digest p ri =>
    let r := c.ranges.getD ri default
    let dig := fun (g : Nat) => match st.instOf b g with
      | some i => st.pub.digest i p np
      | none => c.dummyDigests.getD p ByteArray.empty
    let mut out := ByteArray.empty
    match r.role with
    | .runs _ slots =>
      for x in slots do
        out := out ++ match x with
          | some (g, _) => dig g
          | none => cvBytes (st.dummyRunCv.getD p #[])
    | _ => for g in [0:c.g] do out := out ++ dig g
    return out
  | .out =>
    let unit := c.unit
    let ur := c.unitRange
    let g0 := unit.outGroups.getD 0 default
    let nOut := c.outWords
    let mut out := ByteArray.empty
    for k in [0:ur.count] do
      let (g, u) := (k / c.upv, k % c.upv)
      let mut w : Array F128 := Array.replicate g0.words 0
      let mut col := 0
      let lo := c.leavesOut.getD u #[]
      for j in [0:lo.size] do
        let o := lo[j]!
        let v := match st.instOf b g with
          | some i => st.pub.outputs.getD (i * nOut + o) 0
          | none => c.dummyOutputs.getD o 0
        let wi := col / WORD
        let sh := col % WORD
        let x := w[wi]!
        w := w.set! wi (if sh < 64 then { x with lo := x.lo ||| (v.toUInt64 <<< sh.toUInt64) }
                        else { x with hi := x.hi ||| (v.toUInt64 <<< (sh - 64).toUInt64) })
        col := col + g0.bits.getD j 0
      for x in w do out := F128.pushBytes out x
    return out

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
    let bytes := st.regionBytes reg.kind b
    if bytes.size * 8 != 2 ^ nf then throw "a region's bytes are not its free bits"
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
