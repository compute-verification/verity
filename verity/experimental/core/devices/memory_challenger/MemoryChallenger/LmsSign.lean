import Lms.Verify

/-!
# LMS signing at a given leaf, and the memory challenger role's message

What `Lms.Verify` (the `lms` package, verification only) leaves out of `verity/ml/crypto/lms.py`: the deterministic keys
(`keygen`, `key`: RFC 8554 Appendix A, `x_q[i] = H(I ‖ u32(q) ‖ u16(i) ‖ u8(0xff) ‖ SEED)`, `I = SHA-512("verity/lms/v0/I" ‖
SEED)[:16]`), `sign_at` (the signature at one-time key `q`, with no state), `Signature.encode`, and
`memory_challenger_message`. Signing is pure: the leaf state, which hands out each `q` once, is the outside world's
(`Core.World.reserve`). Separable from the memory challenger: it belongs in the `lms` package beside `Lms.Verify`.
-/

namespace Lms

open Flock

/-- A private key: its parameters, identifier, seed, and tree, node `r` for `1 ≤ r < 2^(H+1)`. The tree is a function so
that a test can stand in a sparse one for a deployed key's 2^20 leaves, as the Python's tests do. -/
structure PrivateKey where
  params : Params
  I : ByteArray
  seed : ByteArray
  tree : Nat → ByteArray

def PrivateKey.root (k : PrivateKey) : ByteArray := k.tree 1

/-- `public_key`: `I ‖ T[1]`. -/
def PrivateKey.publicKey (k : PrivateKey) : ByteArray := k.I ++ k.root

/-- `_x`: chain `i`'s secret start for leaf `q`. -/
def secret (p : Params) (I seed : ByteArray) (q i : Nat) : ByteArray :=
  p.hash (I ++ u32 q ++ u16 i ++ ⟨#[0xff]⟩ ++ seed)

/-- Leaf `q`'s tree node: the one-time public key over its chains run to the end. -/
def leafNode (p : Params) (I seed : ByteArray) (q : Nat) : ByteArray :=
  let e := 2 ^ p.W - 1
  leaf p I q (otsPublic p I q ((Array.range p.P).map fun i => chain p I q i (secret p I seed q i) 0 e))

/-- The whole tree, as an array indexed by node: `2^H` leaves, then the interior nodes down to the root. -/
def treeArray (p : Params) (I seed : ByteArray) : Array ByteArray := Id.run do
  let top := 2 ^ p.H
  let mut t := Array.replicate top ByteArray.empty ++ (Array.range top).map (leafNode p I seed)
  for k in [1:top] do
    let r := top - k
    t := t.set! r (interior p I r t[2 * r]! t[2 * r + 1]!)
  return t

/-- `key`: the key of identifier `I` (16 bytes) whose one-time keys `seed` derives. -/
def key (p : Params) (I seed : ByteArray) : E PrivateKey := do
  if I.size ≠ 16 then throw s!"an LMS identifier is 16 bytes, not {I.size}"
  let t := treeArray p I seed
  pure ⟨p, I, seed, fun r => t.getD r ByteArray.empty⟩

/-- `keygen`: the key `seed` (N secret bytes) determines. -/
def keygen (p : Params) (seed : ByteArray) : E PrivateKey := do
  if seed.size ≠ p.N then throw s!"an LMS seed is {p.N} bytes, not {seed.size}"
  key p ((Sha512.hash ("verity/lms/v0/I".toUTF8 ++ seed)).extract 0 16) seed

/-- `sign_at`: the signature of `message` with one-time key `q`. Two messages signed at one `q` give the key away, so a
signer takes `q` from a leaf state that never repeats one (`Core.World.reserve`). -/
def signAt (k : PrivateKey) (q : Nat) (message : ByteArray) : E Signature := do
  let p := k.params
  if q ≥ 2 ^ p.H then throw s!"q is below 2^{p.H}, not {q}"
  let C := p.hash (k.I ++ u32 q ++ u16 0xFFFD ++ ⟨#[0xff]⟩ ++ k.seed)
  let a := digits p (messageHash p k.I q C message)
  let y := (Array.range p.P).map fun i => chain p k.I q i (secret p k.I k.seed q i) 0 a[i]!
  let mut path := #[]
  let mut r := 2 ^ p.H + q
  for _ in [0:p.H] do
    path := path.push (k.tree (r ^^^ 1))
    r := r / 2
  pure ⟨q, C, y, path⟩

/-- `Signature.encode`: `u32(q) ‖ C ‖ y ‖ path`. -/
def Signature.encode (s : Signature) : ByteArray :=
  s.path.foldl (· ++ ·) (s.y.foldl (· ++ ·) (u32 s.q ++ s.C))

/-- `_field`: a u64 big-endian length, then the bytes. -/
def field (b : ByteArray) : ByteArray := be64 b.size.toUInt64 ++ b

def MEMORY_CHALLENGER_TAG : String := "verity/memory-challenger-record/v1"

/-- `memory_challenger_message(I, session, record)`: `SHA-512([tag] ‖ I ‖ [[session]] ‖ [record])`, each `[·]` a
`field`; the context is the session id framed once, and framed again as the context. -/
def memoryChallengerMessage (I session record : ByteArray) : E ByteArray := do
  if I.size ≠ 16 then throw s!"an LMS identifier is 16 bytes, not {I.size}"
  pure (Sha512.hash (field MEMORY_CHALLENGER_TAG.toUTF8 ++ I ++ field (field session) ++ field record))

end Lms
