import Flock.Hash
import Flock.Bytes

/-!
# LMS verification (RFC 8554 over SHA-512)

Executable forms of `verity/ml/crypto/lms.py`'s verifier, whose module docstring is the spec: the parameters (`Params`,
`Params.make`: N = 64, or `TOY_N` = 8 when asked for by name; W in {1, 2, 4, 8}; 1 ≤ H ≤ 30; `deployed` = (64, 1, 20)),
the signature's byte format (`Signature.decode`: `u32(q) ‖ C ‖ y_0 ‖ … ‖ y_{P−1} ‖ path_0 ‖ … ‖ path_{H−1}`), the public
key's (`I ‖ T[1]`), and verification (`digits`, `chain`, `leaf`, `interior`, `messageHash`, `rootOf`, `verify`). No
signing. A Python `ValueError` is an `Except.error` here. `H(x)` is `Flock.Sha512.hash` truncated to N bytes.
`Lms.Difftest` holds these to the Python on generated keys and signatures.
-/

namespace Lms

open Flock

abbrev E := Except String

def dPblc : Nat := 0x8080
def dMesg : Nat := 0x8181
def dLeaf : Nat := 0x8282
def dIntr : Nat := 0x8383

/-- The one N besides 64 that `Params.make` admits, and only with `toy`. -/
def toyN : Nat := 8

/-- An LMS parameter set: N the hash length, W the Winternitz width, H the tree height. -/
structure Params where
  N : Nat
  W : Nat
  H : Nat
  deriving BEq

/-- `Params.__post_init__`: N in (8, 64) with `toy`, else 64; W in (1, 2, 4, 8); 1 ≤ H ≤ 30. -/
def Params.make (N W H : Nat) (toy : Bool) : E Params :=
  let sizes := if toy then [toyN, 64] else [64]
  if N ∈ sizes ∧ W ∈ [1, 2, 4, 8] ∧ 1 ≤ H ∧ H ≤ 30 then pure ⟨N, W, H⟩
  else throw s!"LMS parameters are N in {sizes}, W in (1, 2, 4, 8), 1 <= H <= 30, not ({N}, {W}, {H})"

/-- `lms.DEPLOYED`. -/
def deployed : Params := ⟨64, 1, 20⟩

/-- Python's `int.bit_length` on a natural number. -/
def bitLength (n : Nat) : Nat := if n = 0 then 0 else Nat.log2 n + 1

/-- U = ⌈8N / W⌉ message digits. -/
def Params.U (p : Params) : Nat := (8 * p.N + p.W - 1) / p.W

/-- V = ⌈bit_length((2^W − 1)·U) / W⌉ checksum digits. -/
def Params.V (p : Params) : Nat := (bitLength ((2 ^ p.W - 1) * p.U) + p.W - 1) / p.W

/-- P = U + V chains. -/
def Params.P (p : Params) : Nat := p.U + p.V

/-- The checksum's left shift, LS = 16 − V·W. -/
def Params.LS (p : Params) : Nat := 16 - p.V * p.W

/-- `H(x)`: SHA-512, its first N bytes. -/
def Params.hash (p : Params) (data : ByteArray) : ByteArray := (Sha512.hash data).extract 0 p.N

def Params.signatureBytes (p : Params) : Nat := 4 + p.N * (1 + p.P + p.H)

def u32 (x : Nat) : ByteArray := be32 x.toUInt32

def u16 (x : Nat) : ByteArray := ⟨#[(x >>> 8).toUInt8, x.toUInt8]⟩

/-- A signature: the leaf q, the randomizer C, the P chain values y and the H path nodes. -/
structure Signature where
  q : Nat
  C : ByteArray
  y : Array ByteArray
  path : Array ByteArray

/-- `Signature.decode`: exactly `signatureBytes` bytes, `u32(q) ‖ C ‖ y ‖ path`, N bytes each but q. -/
def Signature.decode (p : Params) (data : ByteArray) : E Signature := do
  if data.size ≠ p.signatureBytes then
    throw s!"an LMS signature is {p.signatureBytes} bytes, not {data.size}"
  let n := p.N
  let q := (List.range 4).foldl (fun acc i => acc * 256 + (data.get! i).toNat) 0
  let cut := (List.range (1 + p.P + p.H)).toArray.map fun k => data.extract (4 + n * k) (4 + n * (k + 1))
  pure ⟨q, cut[0]!, cut.extract 1 (1 + p.P), cut.extract (1 + p.P) cut.size⟩

/-- `coef(s, i, w)`: digit i of `s`, w bits wide, most significant first (RFC 8554 §3.1.3). -/
def coef (s : ByteArray) (i w : Nat) : Nat :=
  (2 ^ w - 1) &&& ((s.get! (i * w / 8)).toNat >>> (8 - (w * (i % (8 / w)) + w)))

/-- The P chain lengths a signature reveals: Q's U digits, then its checksum's V (`coef(Q ‖ Cksm(Q))`). -/
def digits (p : Params) (Q : ByteArray) : Array Nat :=
  let w := p.W
  let cksm := ((List.range p.U).foldl (fun acc i => acc + (2 ^ w - 1 - coef Q i w)) 0) <<< p.LS
  let s := Q ++ u16 cksm
  (List.range p.P).toArray.map (coef s · w)

/-- Chain i of leaf q from `tmp`, steps `start … stop − 1`: `tmp ← H(I ‖ u32(q) ‖ u16(i) ‖ u8(j) ‖ tmp)`. -/
def chain (p : Params) (I : ByteArray) (q i : Nat) (tmp : ByteArray) (start stop : Nat) : ByteArray := Id.run do
  let mut tmp := tmp
  for j in [start:stop] do
    tmp := p.hash (I ++ u32 q ++ u16 i ++ ⟨#[j.toUInt8]⟩ ++ tmp)
  return tmp

/-- The one-time public key `K = H(I ‖ u32(q) ‖ u16(D_PBLC) ‖ z_0 ‖ … ‖ z_{P−1})`. -/
def otsPublic (p : Params) (I : ByteArray) (q : Nat) (z : Array ByteArray) : ByteArray :=
  p.hash (z.foldl (· ++ ·) (I ++ u32 q ++ u16 dPblc))

/-- Leaf q, node `2^H + q`: `H(I ‖ u32(2^H + q) ‖ u16(D_LEAF) ‖ K)`. -/
def leaf (p : Params) (I : ByteArray) (q : Nat) (K : ByteArray) : ByteArray :=
  p.hash (I ++ u32 (2 ^ p.H + q) ++ u16 dLeaf ++ K)

/-- Interior node r: `H(I ‖ u32(r) ‖ u16(D_INTR) ‖ left ‖ right)`. -/
def interior (p : Params) (I : ByteArray) (r : Nat) (left right : ByteArray) : ByteArray :=
  p.hash (I ++ u32 r ++ u16 dIntr ++ left ++ right)

/-- `Q = H(I ‖ u32(q) ‖ u16(D_MESG) ‖ C ‖ message)`. -/
def messageHash (p : Params) (I : ByteArray) (q : Nat) (C message : ByteArray) : ByteArray :=
  p.hash (I ++ u32 q ++ u16 dMesg ++ C ++ message)

/-- The root the signature's one-time key and path give (RFC 8554 Algorithms 4b and 6a), refused unless q < 2^H and the
signature has P chain values and H path nodes. -/
def rootOf (p : Params) (I message : ByteArray) (sig : Signature) : E ByteArray := do
  if ¬(sig.q < 2 ^ p.H ∧ sig.y.size = p.P ∧ sig.path.size = p.H) then
    throw "a signature has q below 2^H, P chain values and H path nodes"
  let e := 2 ^ p.W - 1
  let a := digits p (messageHash p I sig.q sig.C message)
  let z := (List.range p.P).toArray.map fun i => chain p I sig.q i sig.y[i]! a[i]! e
  let mut tmp := leaf p I sig.q (otsPublic p I sig.q z)
  let mut r := 2 ^ p.H + sig.q
  for sib in sig.path do
    tmp := if r % 2 = 1 then interior p I (r / 2) sib tmp else interior p I (r / 2) tmp sib
    r := r / 2
  return tmp

/-- Whether `sig` signs `message` under `pub` (`I ‖ T[1]`), refused unless the key is 16 + N bytes. -/
def verify (p : Params) (pub message : ByteArray) (sig : Signature) : E Bool := do
  if pub.size ≠ 16 + p.N then throw s!"an LMS public key is {16 + p.N} bytes, not {pub.size}"
  match rootOf p (pub.extract 0 16) message sig with
  | .ok root => pure (root == pub.extract 16 pub.size)
  | .error _ => pure false

end Lms
