/-
Halevi–Micali hiding leaves (`verity.commitments.hm96`, PR #88), generic over the hash. For an inner digest `x` of the value
and a salt `y`:

    c    = H(pad(SALT_TAG) ‖ y)          b = x ⊕ M·y,  M[i][j] = key bit (i + j)
    leaf = H(pad(LEAF_TAG ‖ H(key)) ‖ b ‖ c)

where `pad` zero-pads to the hash's block. Bit `t` of a byte string is bit `t mod 8` of byte `⌊t/8⌋`. The verifier computes
the leaf natively from the commit string `b ‖ c`; the key is the statement's, never an opening's.
-/
import Flock.Hash
import Flock.Bytes

namespace Flock

structure Hm96 where
  name : String
  hash : ByteArray → ByteArray
  digestBytes : Nat
  saltBytes : Nat
  /-- the hash's block: the width the tags are padded to -/
  block : Nat
  saltTag : ByteArray
  leafTag : ByteArray
  keyLabel : ByteArray
  deriving Inhabited

namespace Hm96

def pad (b : ByteArray) (n : Nat) : ByteArray := b ++ ByteArray.mk (Array.replicate (n - min n b.size) 0)

def bit (b : ByteArray) (t : Nat) : Bool := ((b.get! (t / 8)).toNat >>> (t % 8)) % 2 == 1

def keyBits (h : Hm96) : Nat := 8 * h.digestBytes + 8 * h.saltBytes - 1
def keyBytes (h : Hm96) : Nat := (h.keyBits + 7) / 8

/-- The pinned key: `H(label ‖ be32 i)` for `i = 0, 1, …`, concatenated to `keyBytes`, the bits above `keyBits` cleared. -/
def keyFromLabel (h : Hm96) : ByteArray := Id.run do
  let mut out := ByteArray.empty
  let mut i := 0
  for _ in [0:(h.keyBytes + h.digestBytes - 1) / h.digestBytes] do
    out := out ++ h.hash (h.keyLabel ++ be32 i.toUInt32)
    i := i + 1
  let k := out.extract 0 h.keyBytes
  let spare := 8 * h.keyBytes - h.keyBits
  return k.set! (h.keyBytes - 1) ((k.get! (h.keyBytes - 1)) &&& (0xFF >>> spare.toUInt8))

/-- `M·y`: bit `i` is `⊕_j key_{i+j} · y_j`, `i < 8·digestBytes`, `j < 8·saltBytes`. -/
def mask (h : Hm96) (key y : ByteArray) : ByteArray := Id.run do
  let ys := (List.range (8 * h.saltBytes)).filter (bit y ·)
  let mut out := ByteArray.mk (Array.replicate h.digestBytes 0)
  for i in [0:8 * h.digestBytes] do
    if ys.foldl (fun acc j => acc ^^ bit key (i + j)) false then
      out := out.set! (i / 8) (out.get! (i / 8) ||| (1 <<< (i % 8).toUInt8))
  return out

def saltDigest (h : Hm96) (y : ByteArray) : ByteArray := h.hash (pad h.saltTag h.block ++ y)

/-- `b ‖ c`, what a backend proves and publishes. -/
def commitString (h : Hm96) (key x y : ByteArray) : ByteArray :=
  let m := h.mask key y
  ByteArray.mk ((Array.range h.digestBytes).map fun i => x.get! i ^^^ m.get! i) ++ h.saltDigest y

def leafPrefix (h : Hm96) (key : ByteArray) : ByteArray := pad (h.leafTag ++ h.hash key) h.block

def treeLeaf (h : Hm96) (key bc : ByteArray) : ByteArray := h.hash (h.leafPrefix key ++ bc)

def leaf (h : Hm96) (key x y : ByteArray) : ByteArray := h.treeLeaf key (h.commitString key x y)

private def tag (s : String) : ByteArray := s.toUTF8.push 0

/-- `hm96-sha256/v1` (PR #88). -/
def sha256 : Hm96 :=
  { name := "hm96-sha256/v1", hash := Sha256.hash, digestBytes := 32, saltBytes := 128, block := 64,
    saltTag := tag "verity/hm96-sha256/salt/v1", leafTag := tag "verity/hm96-sha256/leaf/v1",
    keyLabel := tag "verity/hm96-sha256/key/v1" }

/-- `hm96-sha512/v1` (PR #93): every commitment path of the target statement. -/
def sha512 : Hm96 :=
  { name := "hm96-sha512/v1", hash := Sha512.hash, digestBytes := 64, saltBytes := 192, block := 128,
    saltTag := tag "verity/hm96-sha512/salt/v1", leafTag := tag "verity/hm96-sha512/leaf/v1",
    keyLabel := tag "verity/hm96-sha512/key/v1" }

end Hm96

end Flock
