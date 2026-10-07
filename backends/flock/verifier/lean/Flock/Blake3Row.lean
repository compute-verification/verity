/-
The keyed-BLAKE3 row leaf (`blake3-keyed/row/v2`), the circuit statement's row leaf at PR #83 up to `19c7269a`: each input
port's row is hashed by BLAKE3 in keyed mode (key `verity/blake3-leaf/v1/x`) in compression slots of the pinned
compression circuit, chunk runs then (for a port of several chunks) a keyed parent tree, and the root's output is the
port's row digest. Regions: every compression's counter, block length and flags; every chain start's chaining value
(the key); every digest (the public row digests; pinned padding values for padding instances).
-/
import Flock.RowLeaf

namespace Flock

namespace Blake3Row

/-- Upstream `b684b12`'s `blake3::build_block_r1cs(3)`: one `2^14`-bit compression slot. -/
def kLog : Nat := 14
def zConst : Nat := 11706
def cvBase : Nat := 0
def outLo : Nat := 256
def mBase : Nat := 512
def tLo : Nat := 1024
def flagsBase : Nat := 1120

def spec : CompSpec :=
  { name := "blake3-compression/flock-b684b12", kLog, mDigest := 17, usefulBits := 11707, zConst,
    header := [("cv_base", cvBase), ("out_lo_base", outLo), ("m_base", mBase), ("t_lo_base", tLo), ("flags_base", flagsBase)],
    inputRows := (List.range 256).map (cvBase + ·) ++ (List.range 512).map (mBase + ·) ++ (List.range (flagsBase + 32 - tLo)).map (tLo + ·),
    pinnedDigest := "3f96431c023dbc354e51d101ab575b2be8cbd33bfcac5e9961d65e8e36d66be3" }

def SCHEMA : String := "blake3-keyed/row/v2"

/-- What a compression slot computes (block-uniform). -/
inductive CompRole where
  | run (g p c j : Nat)
  | dummyRun (j : Nat)
  | parent (g p k : Nat)
  | dummyParent
  deriving Inhabited, Repr

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

/-- Block-local column of the slot holding the last compression of template instance `g`'s chunk `ch` of port `p`. -/
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

/-- Block-local column of template instance `g`'s byte `byte` of port `p`: the message bit where it is hashed. -/
def messageCol (c : Circuit) (g p byte : Nat) : V Nat := do
  let nb := (c.ports.getD p default).nb
  let (ch, rest) := (byte / 1024, byte % 1024)
  let (j, off) := (rest / 64, rest % 64)
  let fin ← chunkEnd c g p ch
  return fin - (nb - 1 - j) * 2 ^ kLog + mBase + 8 * off

/-- A compression's Δ beyond its constant: after a chunk's first, the chaining value is the previous slot's output;
a parent's message is its two children's outputs. A only (B stays the pinned constant). -/
def slotDelta (c : Circuit) (r : Range) (s : Nat) : V (Array (Nat × Nat) × Array (Nat × Nat)) := do
  let base := r.slot s
  let mut da : Array (Nat × Nat) := #[]
  match compRole c r s with
  | .run _ _ _ j | .dummyRun j =>
    if j > 0 then
      for t in [0:256] do
        let i := base + cvBase + t
        da := da.push (i, i) |>.push (i, base - 2 ^ kLog + outLo + t)
  | .parent g p k =>
    let ch := (c.ports.getD p default).chunks
    let pr ← parentsRange c p
    for (side, h) in [(0, 2 * k + 1), (1, 2 * k + 2)] do
      let child ← if h < ch - 1 then pure (pr.slot (g * ch + h)) else chunkEnd c g p (h - (ch - 1))
      for t in [0:256] do
        let i := base + mBase + 256 * side + t
        da := da.push (i, i) |>.push (i, child + outLo + t)
  | .dummyParent => pure ()
  return (da, #[])

/-- The input ports' chunking, and the compression ranges: `2^14`-bit slots whose roles cover every run and parent of
every template instance exactly once. -/
def checkLayout (c : Circuit) : V Unit := do
  for p in c.ports do
    let b := 2 * p.words
    let ok := isPow2 p.nb && p.chunks ≥ 1 && b % 64 == 0 && p.chunks == (b + 1023) / 1024 && p.nb == min b 1024 / 64
      && (p.chunks == 1 || (b % 1024 == 0 && isPow2 p.chunks))
    if !ok then throw s!"port {p.name}: not whole power-of-two chunk runs"
  let mut runs : Array (Nat × Nat × Nat) := #[]
  let mut parents : Array Nat := #[]
  for r in c.ranges do
    if r.kind == Kind.comp && r.slotLog != kLog then throw "a compression range's slots are not the compression circuit's"
    match r.role with
    | .runs p slots =>
      let some port := c.ports[p]? | throw "runs of an unknown port"
      if slots.size * port.nb != r.count || !isPow2 r.count then throw "a runs range's slot list is not its count"
      for x in slots do
        if let some (g, ch) := x then
          if g ≥ c.g || ch ≥ port.chunks || runs.contains (g, p, ch) then throw "a run outside the block's VUs or listed twice"
          runs := runs.push (g, p, ch)
    | .parents p =>
      let some port := c.ports[p]? | throw "parents of an unknown port"
      if port.chunks < 2 || r.count != c.g * port.chunks || parents.contains p then
        throw "a parents range is not one per multi-chunk port"
      parents := parents.push p
    | .none => pure ()
  let want := c.ports.foldl (· + ·.chunks) 0 * c.g
  if runs.size != want || parents.size != (c.ports.filter (·.chunks > 1)).size then
    throw "the runs do not cover every chunk of every VU (or a multi-chunk port has no parent tree)"

def cvBytes (cv : Array UInt32) : ByteArray := Blake3.bytesOfWords cv 8

/-- A padding run's output value: `nb` zero blocks from the key, per port. -/
def dummyRunCv (c : Circuit) (p : Port) : Array UInt32 := Id.run do
  let mut cv := Blake3.wordsOfBytes c.key 0 8
  for j in [0:p.nb] do
    let (ctr, bl, f) := params c (.dummyRun j)
    cv := (Blake3.compress cv (Array.replicate 16 0) ctr.toUInt64 bl f).extract 0 8
  return cv

/-- Params, CvIn and (single-chunk ports) Digest per run range; Params, CvIn and Digest at each parent range. -/
def regions (c : Circuit) (pub : Public) : V (Array Region) := do
  let kl := c.kLog
  let np := c.ports.size
  let bits := fun (lo n : Nat) => (Array.range n).map (lo + ·)
  let paramsBytes := fun (r : Range) => Id.run do
    let mut out := ByteArray.empty
    for s in [0:r.count] do
      let (ctr, bl, f) := params c (compRole c r s)
      out := out ++ le64 ctr.toUInt64 ++ le32 bl ++ le32 f
    return out
  let keys := fun (k : Nat) => (List.replicate k c.key).foldl (· ++ ·) ByteArray.empty
  let dig := fun (p b g : Nat) => match c.instOf pub.n b g with
    | some i => pub.digest i p np
    | none => c.dummyDigests.getD p ByteArray.empty
  let mut out := #[]
  for ri in [0:c.ranges.size] do
    let r := c.ranges[ri]!
    let lr := Nat.log2 r.count
    match r.role with
    | .runs p slots =>
      let port := c.ports.getD p default
      let lnb := Nat.log2 port.nb
      let pb := paramsBytes r
      out := out.push (← mkRegion kl s!"params({ri})" (bits 0 7 ++ bits kLog lr) (r.start + tLo) fun _ => pb)
      let kb := keys slots.size
      out := out.push (← mkRegion kl s!"cv_in({ri})" (bits 0 8 ++ bits (kLog + lnb) (lr - lnb)) (r.start + cvBase) fun _ => kb)
      if port.chunks == 1 then
        let pad := cvBytes (dummyRunCv c port)
        out := out.push (← mkRegion kl s!"digest({p},{ri})" (bits 0 8 ++ bits (kLog + lnb) (lr - lnb))
          (r.start + (port.nb - 1) * 2 ^ kLog + outLo) fun b =>
            slots.foldl (fun acc x => acc ++ match x with | some (g, _) => dig p b g | none => pad) ByteArray.empty)
    | .parents p =>
      let lc := Nat.log2 (c.ports.getD p default).chunks
      let pb := paramsBytes r
      out := out.push (← mkRegion kl s!"params({ri})" (bits 0 7 ++ bits kLog lr) (r.start + tLo) fun _ => pb)
      let kb := keys r.count
      out := out.push (← mkRegion kl s!"cv_in({ri})" (bits 0 8 ++ bits kLog lr) (r.start + cvBase) fun _ => kb)
      out := out.push (← mkRegion kl s!"digest({p},{ri})" (bits 0 8 ++ bits (kLog + lc) (lr - lc)) (r.start + outLo) fun b =>
        (List.range c.g).foldl (fun acc g => acc ++ dig p b g) ByteArray.empty)
    | .none => pure ()
  return out

def rowKey (role : String) : ByteArray :=
  let k := "verity/blake3-leaf/v1/".toUTF8 ++ role.toUTF8
  (k ++ ByteArray.mk (Array.replicate (32 - min 32 k.size) 0)).extract 0 32

/-- The input roots over the public digests (schema `blake3-keyed/row/v2`), the pinned padding digests against keyed BLAKE3
of zero rows, and the row key against the x-row key. -/
def checkPublic (c : Circuit) (pub : Public) : V Unit := do
  let fv ← pub.frameV3
  if (← unhex (← J.str fv "key" "public file.frame_v3")) != c.key || c.key != rowKey "x" then
    throw "the row key is not the x-row key (verity/blake3-leaf/v1/x)"
  let np := c.ports.size
  for p in [0:np] do
    let port := c.ports[p]!
    let dom ← pub.fv3Bytes "domain_ids" port.name
    if (← J.str (← J.get fv "schemas" "public file.frame_v3") port.name "public file.frame_v3.schemas") != SCHEMA then
      throw s!"input port {port.name}: schema is not {SCHEMA}"
    let leaves := (Array.range pub.n).map fun i => FV3.leaf dom i SCHEMA (pub.digest i p np)
    if FV3.root dom leaves != (← pub.fv3Bytes "roots" port.name) then
      throw s!"input port {port.name}: the row digests do not hash to the header's frame-v3 root"
    if Blake3.keyedHash c.key (ByteArray.mk (Array.replicate (2 * port.words) 0)) != c.dummyDigests.getD p ByteArray.empty then
      throw s!"the pinned padding digest of port {port.name} is not the zero row's"

def leaf (comp : Comp) : RowLeaf :=
  { schema := SCHEMA, comp, checkLayout, slotDelta, messageCol, regions, checkPublic, key := (·.key) }

end Blake3Row

end Flock
