/-
The bytes and words a table read's gates hash, on the verifier's builder (`Flock.HmNets`), laid out as
`gates.frame_v3` lays them: a byte is 8 forms least significant first, a word 64 forms least significant first with its
bytes big-endian, and a message's SHA-512 is the chain of compressions from the IV over its padded words (`sha512W`).
It imports only the builder, so a check that a registered read is the builder's net can run it as code.
-/
module

public import Flock.HmNets

@[expose] public section

namespace Accounting.ReadGates

open Flock.HmNets hiding and

/-- A byte of the circuit: its 8 bits' forms, least significant first. -/
abbrev Byte := List Lin

/-- `frame_v3._bytes`: words' bytes, eight a word, big-endian. -/
def bytesW (ws : List Word) : List Byte :=
  ws.flatMap fun w => (List.range 8).map fun b => (w.drop (8 * (7 - b))).take 8

/-- `frame_v3._words`: bytes read back as words, eight a word, big-endian. -/
def wordsB (data : List Byte) : List Word :=
  (List.range (data.length / 8)).map fun k => (List.range 8).flatMap fun b => data.getD (8 * k + 7 - b) []

/-- `forms.const_bits b 8`: a constant byte. -/
def constByte (b : UInt8) : Byte := (List.range 8).map fun i => Lin.const (b.toNat.testBit i)

/-- `frame_v3._const_bytes`. -/
def constBytes (m : List UInt8) : List Byte := m.map constByte

/-- `int.to_bytes(n, "big")`: `v`'s low `8 n` bits as `n` bytes, big-endian. -/
def beBytes (n v : Nat) : List UInt8 := (List.range n).map fun i => (v / 256 ^ (n - 1 - i) % 256).toUInt8

/-- `frame_v3._uint_bytes`: the low `8 n` bits of `bits` (least significant first, zero past their end) as `n` bytes,
big-endian. -/
def uintBytes (bits : List Lin) (n : Nat) : List Byte :=
  let p := bits.take (8 * n) ++ List.replicate (8 * n - min bits.length (8 * n)) Lin.zero
  (List.range n).map fun b => (p.drop (8 * (n - 1 - b))).take 8

/-- `sha512.pad_tail`: SHA-512's padding after a message of `len` bytes: `0x80`, zeros, the 128-bit bit count. -/
def padTail (len : Nat) : List UInt8 :=
  0x80 :: (List.replicate ((128 - (len + 17) % 128) % 128) 0 ++ beBytes 16 (8 * len))

/-- SHA-512's initial chaining value as constant words. -/
def ivW : List Word := Flock.Sha512.iv.toList.map fun v => constW v.toBitVec

/-- `frame_v3._sha512`: the chain of compressions from the IV over the 16-word blocks of `ws`. -/
def sha512W (ws : List Word) : M (List Word) :=
  (List.range ((ws.length + 15) / 16)).foldlM (fun st i => compress st ((ws.drop (16 * i)).take 16)) ivW

end Accounting.ReadGates
