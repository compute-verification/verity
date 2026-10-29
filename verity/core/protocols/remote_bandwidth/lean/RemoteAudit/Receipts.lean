import Flock.FV3
import Flock.Hm96
import RemoteAudit.Codec
import RemoteAudit.Plan

/-!
# Receipts, seeds and the receipt tree: `receipts.py`, and `audit.py`'s chain over appends

~~~text
seed_(t,i)    SHA512(SEED_DOMAIN ‖ σ_t ‖ u32be i ‖ chained(root_(t,i−1), head))
chained       root itself without appends (head none), else SHA512(CHAINED_DOMAIN ‖ root ‖ head)
head          SHA512(APPENDS_DOMAIN ‖ head_before (64 zero bytes for none) ‖ u64be first ‖ u64be count ‖ root)
y_(i,p)       sketch-sha512: SHA512(Z(SKETCH_RECEIPT_DOMAIN ‖ frame(seed)) ‖ u64be j ‖ σ_(i,j));
              block-sha512: SHA512(Z(RECEIPT_DOMAIN ‖ frame(seed)) ‖ u64be j ‖ E_j)
S_i           k_s rows of 4096 signs: K = derive(seed, SIGN_DOMAIN, {"tag": "sketch-sha512", "k": k_s, "n": 4096}), row c
              the first 512 bytes of K.stream(("row", c)), bit u (byte u / 8, LSB first) 1 → −1
σ_(i,j)       S_i · E_j over the block's bytes as int8, k_s int32 little-endian
leaf c        y_(16c) ‖ … ‖ y_(16c+15) (the last leaf short), committed as hm96-sha512/row/v2:
              hm96 tree_leaf(key, commit_string(key, sha512/row/v2(leaf), salt))
root_(t,i)    frame-v3-sha512: CommitmentDomain(SHA512(TREE_DOMAIN ‖ seed ‖ plan digest), 0, 0 … ⌈L/16⌉ − 1)
~~~

Receipts here are of a block alone: a registration that binds PoUW's tile outputs (`bind`) is refused before any
receipt is computed (`Registration.make?`).
-/

namespace RemoteAudit

def SEED_DOMAIN : ByteArray := tag0 "verity/remote-bandwidth/seed/v0"
def RECEIPT_DOMAIN : ByteArray := tag0 "verity/remote-bandwidth/receipt/v0"
def SKETCH_RECEIPT_DOMAIN : ByteArray := tag0 "verity/remote-bandwidth/sketch-receipt/v0"
def SIGN_DOMAIN : String := "verity/remote-bandwidth/sketch-signs/v0"
def TREE_DOMAIN : ByteArray := tag0 "verity/remote-bandwidth/receipt-tree/v0"
def APPENDS_DOMAIN : ByteArray := tag0 "verity/remote-bandwidth/appends/v0"
def CHAINED_DOMAIN : ByteArray := tag0 "verity/remote-bandwidth/chained/v0"
def SCHEMA : String := "hm96-sha512/row/v2"
def PER_LEAF : Nat := 16
def SALT_BYTES : Nat := 192
def DIGEST_BYTES : Nat := 64

/-- `receipts.seed(challenge, i, previous)`. -/
def seed (challenge : ByteArray) (i : Nat) (previous : ByteArray) : ByteArray :=
  sha512 (SEED_DOMAIN ++ challenge ++ be 4 i ++ previous)

/-- `audit.chained(previous, head)`. -/
def chained (previous : ByteArray) : Option ByteArray → ByteArray
  | none => previous
  | some head => sha512 (CHAINED_DOMAIN ++ previous ++ head)

/-- `audit.append_head(head, first, count, root)`. -/
def appendHead (head : Option ByteArray) (first count : Nat) (root : ByteArray) : ByteArray :=
  sha512 (APPENDS_DOMAIN ++ head.getD (zeros 64) ++ be 8 first ++ be 8 count ++ root)

/-- A registration's receipt kind (`receipts.KINDS`). -/
inductive Kind
  | sketch
  | block
deriving BEq, Repr

def Kind.name : Kind → String
  | .sketch => "sketch-sha512"
  | .block => "block-sha512"

def Kind.ofName? : String → Option Kind
  | "sketch-sha512" => some .sketch
  | "block-sha512" => some .block
  | _ => none

/-- `receipts.sketch_rows(kind, rows)` for an explicit `rows`: one of 16 and 64 under sketch-sha512, 0 under
block-sha512. -/
def Kind.rowsOk (k : Kind) (rows : Nat) : Bool :=
  match k with
  | .sketch => rows == 16 || rows == 64
  | .block => rows == 0

/-- `receipts.prefix` and `receipts.sketch_prefix`: one SHA-512 block. -/
def receiptPrefix (domain seed : ByteArray) : ByteArray := pad128 (domain ++ frameBytes seed)

/-- `sign_bits(seed, rows)`: `rows` rows of 512 bytes. -/
def signBits (seed : ByteArray) (rows : Nat) : Array ByteArray :=
  let key := FlockSoundness.Randomness.derive seed SIGN_DOMAIN
    [("tag", .str "sketch-sha512"), ("k", .int rows), ("n", .int Codec.BLOCK_BYTES)]
  (Array.range rows).map fun c => FlockSoundness.Randomness.stream key [.str "row", .int c] (Codec.BLOCK_BYTES / 8)

private def int8 (x : UInt8) : Int := if x.toNat ≥ 128 then (x.toNat : Int) - 256 else x.toNat

/-- `SketchReceipter.sketch(block)`: each row's ±1 dot product with the block's bytes as int8, int32 little-endian. -/
def sketch (signs : Array ByteArray) (block : ByteArray) : ByteArray :=
  signs.foldl (fun acc row => Id.run do
    let mut s : Int := 0
    for u in [0:Codec.BLOCK_BYTES] do
      let e := int8 (block.get! u)
      s := if (row.get! (u / 8)).toNat >>> (u % 8) % 2 == 1 then s - e else s + e
    return acc ++ le 4 (s % 2 ^ 32).toNat) ByteArray.empty

/-- A receipter under one pass's seed: `y_(i,j)` of block j's bytes. -/
structure Receipter where
  kind : Kind
  head : ByteArray
  signs : Array ByteArray

/-- `receipts.receipter(kind, seed, rows)`. -/
def receipter (kind : Kind) (seed : ByteArray) (rows : Nat) : Receipter :=
  match kind with
  | .sketch => ⟨kind, receiptPrefix SKETCH_RECEIPT_DOMAIN seed, signBits seed rows⟩
  | .block => ⟨kind, receiptPrefix RECEIPT_DOMAIN seed, #[]⟩

def Receipter.receipt (r : Receipter) (j : Nat) (block : ByteArray) : ByteArray :=
  match r.kind with
  | .sketch => sha512 (r.head ++ be 8 j ++ sketch r.signs block)
  | .block => sha512 (r.head ++ be 8 j ++ block)

/-- `receipts.leaves(positions)`. -/
def leaves (positions : Nat) : Nat := (positions + PER_LEAF - 1) / PER_LEAF

/-- `leaf_receipt(leaf, p, positions)`: position p's receipt inside its leaf's receipts, `none` if they are not that
leaf's size. -/
def leafReceipt (leaf : ByteArray) (p positions : Nat) : Option ByteArray :=
  let c := p / PER_LEAF
  if (leaf.size : Int) ≠ DIGEST_BYTES * min (PER_LEAF : Int) ((positions : Int) - PER_LEAF * c) then none
  else
    let k := p - PER_LEAF * c
    some (leaf.extract (DIGEST_BYTES * k) (DIGEST_BYTES * (k + 1)))

/-- `hidden_digest(x, salt)`: the hm96-sha512 leaf over a row's sha512/row/v2 digest. -/
def hiddenDigest (x salt : ByteArray) : ByteArray := Flock.Hm96.Default512.leaf x salt

/-- `hidden(row, salt)`. -/
def hidden (row salt : ByteArray) : ByteArray := hiddenDigest (Codec.rowDigest row) salt

/-- `receipts.domain(seed, plan)`. -/
def treeDomain (seed : ByteArray) (plan : Plan) : Flock.FV3.Sha512.Domain :=
  .make (sha512 (TREE_DOMAIN ++ seed ++ plan.digest)) 0 0 (leaves plan.length)

/-- `verify_leaf(seed, plan, root, p, leaf, salt, path)`. -/
def verifyLeaf (seed : ByteArray) (plan : Plan) (root : ByteArray) (p : Nat) (leaf salt : ByteArray)
    (path : Array ByteArray) : Bool :=
  let d := treeDomain seed plan
  d.verifyOpening root d.count (p / PER_LEAF) (hidden leaf salt) SCHEMA path

end RemoteAudit
