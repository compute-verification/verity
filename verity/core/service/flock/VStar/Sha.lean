import Flock.HmNets

/-!
# SHA-512 over GF(2) forms

`shaFrom cv bits pre` is `verity_flock.rec_open._sha` over `forms`: SHA-512 of a message whose first `pre` bytes left the
constant chaining value `cv` and whose rest is `bits` (byte-string order, bit `t % 8` of byte `t / 8`), its padding constant
bits, each 1024-bit block placed in word order (`wordBit`) and compressed by `Flock.HmNets.compress`
(`sha512.compress_gadget`); the digest's 512 forms in byte-string order.
-/

namespace VStar

open Flock.HmNets

/-- `sha512.pad_tail`'s length field: the message's bit length, 16 bytes big-endian. -/
def lenBytes (len : Nat) : List UInt8 := (List.range' 0 16).map fun a => (((len * 8) >>> (8 * (15 - a))) % 256).toUInt8

/-- The zero bytes between `0x80` and the length field after a `len`-byte message. -/
def zeroPad (len : Nat) : Nat := (240 - (len + 1) % 128) % 128

/-- `sha512.pad_tail total`: `0x80`, zeros, the length, so that `total` bytes and the tail are whole 128-byte blocks. -/
def padTail (total : Nat) : List UInt8 := [128] ++ List.replicate (zeroPad total) 0 ++ lenBytes total

/-- Bit `t` of a byte string: bit `t % 8` of byte `t / 8`. -/
def bitL (l : List UInt8) (t : Nat) : Bool := (l.getD (t / 8) 0).toNat.testBit (t % 8)

def bitsOf (l : List UInt8) : List Bool := (List.range (8 * l.length)).map (bitL l)

/-- The message after the prefix: `bits`, then the padding's constant bits. -/
def shaMsg (bits : List Lin) (pre : Nat) : List Lin := bits ++ (bitsOf (padTail (pre + bits.length / 8))).map Lin.const

/-- The block at bit `off` in word order: column `p` holds message bit `off + wordBit p` (`wordBit` is an involution). -/
def blockAt (msg : Array Lin) (off : Nat) : List Lin := (List.range 1024).map fun p => msg.getD (off + wordBit p) Lin.zero

/-- One compression of the state's 512 forms by a block's 1024. -/
def shaBlock (st blk : List Lin) : M (List Lin) := do
  let out ← compress (words st 8) (words blk 16)
  pure out.flatten

/-- `n` blocks from block `k` on. -/
def shaBlocks (msg : Array Lin) : Nat → Nat → List Lin → M (List Lin)
  | 0, _, st => pure st
  | n + 1, k, st => do
    let st ← shaBlock st (blockAt msg (1024 * k))
    shaBlocks msg n (k + 1) st

def shaFrom (cv : List (BitVec 64)) (bits : List Lin) (pre : Nat) : M (List Lin) := do
  let msg := (shaMsg bits pre).toArray
  let st ← shaBlocks msg (msg.size / 1024) 0 (cv.map constW).flatten
  let sa := st.toArray
  pure ((List.range 512).map fun t => sa.getD (wordBit t) Lin.zero)

/-- SHA-512's initial chaining value. -/
def iv : List (BitVec 64) := Flock.Sha512.iv.toList.map (·.toBitVec)

end VStar
