import Lms.Verify

/-!
# LMS signing (RFC 8554 over SHA-512), from the private seed

Executable forms of `verity/ml/crypto/lms.py`'s key derivation and stateless signing: the one-time keys from the seed
(RFC 8554 Appendix A: `x_q[i] = H(I ‖ u32(q) ‖ u16(i) ‖ u8(0xff) ‖ SEED)`), the tree (`key`, `keygen`, with
`I = SHA-512("verity/lms/v0/I" ‖ SEED)[:16]`), the public key `I ‖ T[1]`, a signature's encoding, and `signAt`, the
signature at leaf q (`lms.sign_at`). Which leaf to use is the signer's state, `lms.LeafState.reserve`, which stays with
the host: `signAt` signs at the leaf it is given, and two messages signed at one leaf give the key away.
`Lms.SignDifftest` holds these to the Python on generated keys.
-/

namespace Lms

open Flock

/-- A private key: the parameters, I, the seed and the tree, node r at index r for 1 ≤ r < 2^(H+1) (index 0 unused). -/
structure PrivateKey where
  params : Params
  I : ByteArray
  seed : ByteArray
  tree : Array ByteArray

/-- `x_q[i] = H(I ‖ u32(q) ‖ u16(i) ‖ u8(0xff) ‖ SEED)`. -/
def secret (p : Params) (I seed : ByteArray) (q i : Nat) : ByteArray :=
  p.hash (I ++ u32 q ++ u16 i ++ ⟨#[0xff]⟩ ++ seed)

/-- Leaves `lo … hi − 1`: each one-time key's chains run to their ends, then the one-time public key and the leaf. -/
def leaves (p : Params) (I seed : ByteArray) (lo hi : Nat) : Array ByteArray :=
  let e := 2 ^ p.W - 1
  (List.range (hi - lo)).toArray.map fun j =>
    let q := lo + j
    let z := (List.range p.P).toArray.map fun i => chain p I q i (secret p I seed q i) 0 e
    leaf p I q (otsPublic p I q z)

/-- `lms.key`: the key of identifier `I` (16 bytes) whose one-time keys `seed` derives, with its whole tree. -/
def key (p : Params) (I seed : ByteArray) : E PrivateKey := do
  if I.size ≠ 16 then throw s!"an LMS identifier is 16 bytes, not {I.size}"
  let top := 2 ^ p.H
  let mut tree := Array.replicate top ByteArray.empty ++ leaves p I seed 0 top
  for j in [0:top - 1] do
    let r := top - 1 - j
    tree := tree.set! r (interior p I r tree[2 * r]! tree[2 * r + 1]!)
  pure ⟨p, I, seed, tree⟩

/-- `lms.keygen`: the key `seed` (N bytes) determines, `I = SHA-512("verity/lms/v0/I" ‖ seed)[:16]`. -/
def keygen (p : Params) (seed : ByteArray) : E PrivateKey := do
  if seed.size ≠ p.N then throw s!"an LMS seed is {p.N} bytes, not {seed.size}"
  key p ((Sha512.hash ("verity/lms/v0/I".toUTF8 ++ seed)).extract 0 16) seed

/-- `lms.public_key`: `I ‖ T[1]`. -/
def PrivateKey.publicKey (k : PrivateKey) : ByteArray := k.I ++ k.tree[1]!

/-- `Signature.encode`: `u32(q) ‖ C ‖ y_0 ‖ … ‖ y_{P−1} ‖ path_0 ‖ … ‖ path_{H−1}`. -/
def Signature.encode (s : Signature) : ByteArray :=
  s.path.foldl (· ++ ·) (s.y.foldl (· ++ ·) (u32 s.q ++ s.C))

/-- `lms.sign_at`: the signature of `message` with one-time key q, refused unless q < 2^H. -/
def signAt (k : PrivateKey) (q : Nat) (message : ByteArray) : E Signature := do
  let p := k.params
  if q ≥ 2 ^ p.H then throw s!"q is below 2^{p.H}, not {q}"
  let C := p.hash (k.I ++ u32 q ++ u16 0xFFFD ++ ⟨#[0xff]⟩ ++ k.seed)
  let a := digits p (messageHash p k.I q C message)
  let y := (List.range p.P).toArray.map fun i => chain p k.I q i (secret p k.I k.seed q i) 0 a[i]!
  let path := (List.range p.H).toArray.map fun l => k.tree[((2 ^ p.H + q) >>> l) ^^^ 1]!
  pure ⟨q, C, y, path⟩

end Lms
