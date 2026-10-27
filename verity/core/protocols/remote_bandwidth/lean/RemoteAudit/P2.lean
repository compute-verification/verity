import PousCheck.Merkle
import RemoteAudit.Window

/-!
# A P2 store's registration, receipts and openings: `p2.py`

~~~text
block j       C*'s block j, 2,056 bytes; j = blocks_per_segment·s + v
unit g        blocks 4g … 4g + 3, 8,224 bytes: 257 whole 32-byte sectors; U = ⌊N/4⌋, the plans' extent
digest        SHA512(REGISTRATION_DOMAIN ‖ frame(node) ‖ u32be w ‖ u32be c0 ‖ u32be blocks_per_segment ‖
              u64be segments ‖ frame(expander) ‖ commitment ‖ salt ‖ root ‖ frame(layout) ‖ frame("sketch-sha512") ‖
              u32be k_s)
S_i           k_s rows of 8,224 signs: K = derive(seed, SIGN_DOMAIN, {"tag": "sketch-sha512", "k": k_s, "n": 8224}),
              row c the first 1,028 bytes of K.stream(("row", c)), bit u (byte u / 8, LSB first) 1 → −1
π_(i,j)       Σ_x S_i[c][2056·(j mod 4) + x] · int8(block_j[x]), k_s int32 little-endian
σ_(i,g)       S_i · int8(unit g's bytes) = Σ π over its blocks, k_s int32 little-endian
y_(i,p)       SHA512(Z(SKETCH_RECEIPT_DOMAIN ‖ frame(seed_i)) ‖ u64be g ‖ σ_(i,g)),  g = plan_i[p]
~~~

An opening is checked as `p2.check` checks it (`checkOpening`), its failures `audit.FAILURES`' names: `malformed`,
`receipt-path` (the receipt leaf on the pass's root, as a registered block's), `payload-path` (each of the unit's four
blocks on the registration's vk root, by PoUS's own leaf and RFC 9162 path check, `PousCheck.Merkle.packageLeaf` and
`verifyPath`) and `receipt` (y recomputed from the four blocks). Nothing here decodes a block: vk's leaves are of the
`C*` the verifier decoded to the committed W at PoUS's setup (`PousCheck.P2` is P2's decode).
-/

namespace RemoteAudit.P2

def REGISTRATION_DOMAIN : ByteArray := tag0 "verity/remote-bandwidth/p2-registration/v0"
/-- PoUS's registered point (`codec.P2_POINT`) and its keys' expander. -/
def W : Nat := 16448
def C0 : Nat := 21065
def EXPANDER : String := "sha512-ctr-v4"
def LAYOUT : String := "p2-unit4/v0"
def BLOCK_BYTES : Nat := 2056
def UNIT_BLOCKS : Nat := 4
def UNIT_BYTES : Nat := UNIT_BLOCKS * BLOCK_BYTES
/-- PoUS's salt (`pous.params.SALT_BITS`), what vk's leaves are under. -/
def POUS_SALT_BYTES : Nat := 24

/-- What the store is (`p2.Registration`): the node, P2's geometry at the registered point, and PoUS's setup's
commitment to W, salt and vk root, with sketch-sha512's rows k_s. -/
structure Registration where
  node : ByteArray
  blocksPerSegment : Nat
  segments : Nat
  commitment : ByteArray
  salt : ByteArray
  root : ByteArray
  rows : Nat

/-- A registration `p2.Registration` (with its `codec.P2Codec`) accepts: a node of 1 to 64 bytes, the registered point
and expander, `1 ≤ blocks_per_segment ≤ 2^16` and `1 ≤ segments < 2^32` with at least one unit, a 64-byte
commitment and root, PoUS's 24-byte salt, and 16 or 64 rows. -/
def Registration.make? (node : ByteArray) (w c0 : Nat) (expander : String) (blocksPerSegment segments : Nat)
    (commitment salt root : ByteArray) (rows : Nat) : Except String Registration := do
  if !(1 ≤ node.size && node.size ≤ 64) then throw "a node's id is 1 to 64 bytes"
  if w ≠ W || c0 ≠ C0 || expander ≠ EXPANDER then throw "a P2 store is at PoUS's registered point"
  if blocksPerSegment = 0 || segments = 0 then throw "a P2 store's geometry is positive"
  if blocksPerSegment > 2 ^ 16 || segments ≥ 2 ^ 32 then
    throw "a P2 store's segments are at most 2^16 blocks, and fewer than 2^32"
  if blocksPerSegment * segments < UNIT_BLOCKS then throw "a P2 store is at least one unit of 4 blocks"
  if commitment.size ≠ DIGEST_BYTES then throw "PoUS's commitment to W is 64 bytes"
  if salt.size ≠ POUS_SALT_BYTES then throw "PoUS's salt is 24 bytes"
  if root.size ≠ DIGEST_BYTES then throw "the vk root is 64 bytes"
  if !Kind.sketch.rowsOk rows then throw "sketch-sha512's rows are 16 or 64"
  pure { node, blocksPerSegment, segments, commitment, salt, root, rows }

/-- N, the store's blocks. -/
def Registration.blocks (r : Registration) : Nat := r.blocksPerSegment * r.segments

/-- U, the store's units: what its plans range over. -/
def Registration.units (r : Registration) : Nat := r.blocks / UNIT_BLOCKS

/-- `p2.Registration.digest`. -/
def Registration.digest (r : Registration) : ByteArray :=
  sha512 (REGISTRATION_DOMAIN ++ frameBytes r.node ++ be 4 W ++ be 4 C0 ++ be 4 r.blocksPerSegment ++ be 8 r.segments
    ++ frameStr EXPANDER ++ r.commitment ++ r.salt ++ r.root ++ frameStr LAYOUT ++ frameStr Kind.sketch.name
    ++ be 4 r.rows)

/-- `VerifierKey.check_path(j, block, path)` on the registration's root-only vk: PoUS's leaf of block j under its
salt, on the root of N leaves. -/
def Registration.checkPath (r : Registration) (j : Nat) (block : ByteArray) (path : Array ByteArray) : Bool :=
  block.size == BLOCK_BYTES && PousCheck.Merkle.verifyPath r.root r.blocks j
    (PousCheck.Merkle.packageLeaf r.salt (j / r.blocksPerSegment) (j % r.blocksPerSegment) block) path

/-- `receipts.sign_bits(seed, rows, 8224)`: S_i's rows of 1,028 bytes. -/
def unitSigns (seed : ByteArray) (rows : Nat) : Array ByteArray :=
  let key := FlockSoundness.Randomness.derive seed SIGN_DOMAIN
    [("tag", .str "sketch-sha512"), ("k", .int rows), ("n", .int UNIT_BYTES)]
  (Array.range rows).map fun c => FlockSoundness.Randomness.stream key [.str "row", .int c] (UNIT_BYTES / 8)

private def int8 (x : UInt8) : Int := if x.toNat ≥ 128 then (x.toNat : Int) - 256 else x.toNat

/-- One row's ±1 dot product with `piece` laid at byte `off` of the unit: Σ_x row[off + x] · int8(piece[x]). -/
def dot (row : ByteArray) (off : Nat) (piece : ByteArray) : Int := Id.run do
  let mut s : Int := 0
  for x in [0:piece.size] do
    let u := off + x
    let e := int8 (piece.get! x)
    s := if (row.get! (u / 8)).toNat >>> (u % 8) % 2 == 1 then s - e else s + e
  return s

/-- k_s integers as int32 little-endian. -/
def int32s (xs : Array Int) : ByteArray := xs.foldl (fun acc s => acc ++ le 4 (s % 2 ^ 32).toNat) ByteArray.empty

/-- `Sketcher.block(j, block)`: π, block j's partial at its place in unit j / 4. -/
def blockPartial (signs : Array ByteArray) (j : Nat) (block : ByteArray) : Array Int :=
  signs.map fun row => dot row (BLOCK_BYTES * (j % UNIT_BLOCKS)) block

/-- `Sketcher.total(partials)`: σ, partials summed row by row. -/
def total (rows : Nat) (partials : Array (Array Int)) : Array Int :=
  (Array.range rows).map fun c => partials.foldl (fun s x => s + x.getD c 0) 0

/-- `SketchReceipter.sketch(unit)` at 8,224 bytes: σ of a unit's bytes. -/
def unitSketch (signs : Array ByteArray) (unit : ByteArray) : Array Int := signs.map fun row => dot row 0 unit

/-- `Sketcher.receipt(g, σ)`: y from unit g's sketch, however it was computed. -/
def unitReceipt (seed : ByteArray) (g : Nat) (sigma : ByteArray) : ByteArray :=
  sha512 (receiptPrefix SKETCH_RECEIPT_DOMAIN seed ++ be 8 g ++ sigma)

/-- Receipt (i, p) of a window opened (`p2.Opening`): unit g = plan_i[p]'s four blocks, each block's path to the vk
root, and its receipt leaf's receipts, salt and path to pass i's root. -/
structure Opening where
  pass : Int
  position : Int
  blocks : Array ByteArray
  paths : Array (Array ByteArray)
  leaf : ByteArray
  leafSalt : ByteArray
  receiptPath : Array ByteArray
deriving Inhabited

/-- `checkOpening` with S_i given (`signs`, `unitSigns seed reg.rows`): a window's check derives each pass's rows once,
as `p2.Session._check` keeps a pass's `Sketcher`. -/
def checkOpeningWith (signs : Array ByteArray) (reg : Registration) (seed : ByteArray) (plan : Plan) (root : ByteArray)
    (i p : Nat) (o : Opening) : Option Failure := Id.run do
  if o.pass ≠ i || o.position ≠ p || o.blocks.size ≠ UNIT_BLOCKS || o.blocks.any (·.size ≠ BLOCK_BYTES)
      || o.paths.size ≠ UNIT_BLOCKS || o.paths.any (·.any (·.size ≠ DIGEST_BYTES)) || o.leafSalt.size ≠ SALT_BYTES
      || o.receiptPath.any (·.size ≠ DIGEST_BYTES) then
    return some .malformed
  let g := plan.block p
  let some y := leafReceipt o.leaf p plan.length | return some .malformed
  if !verifyLeaf seed plan root p o.leaf o.leafSalt o.receiptPath then return some .receiptPath
  for b in [0:UNIT_BLOCKS] do
    if !reg.checkPath (UNIT_BLOCKS * g + b) o.blocks[b]! o.paths[b]! then return some .payloadPath
  let unit := o.blocks.foldl (· ++ ·) ByteArray.empty
  if unitReceipt seed g (int32s (unitSketch signs unit)) != y then return some .receipt
  return none

/-- `p2.check`: receipt (i, p)'s opening `o`, position p of pass i under `seed` and `plan`, whose receipt tree's root
is `root`, for `p` a position of the plan: `none` if it checks, else its first failure. -/
def checkOpening (reg : Registration) (seed : ByteArray) (plan : Plan) (root : ByteArray) (i p : Nat) (o : Opening) :
    Option Failure :=
  checkOpeningWith (unitSigns seed reg.rows) reg seed plan root i p o

end RemoteAudit.P2
