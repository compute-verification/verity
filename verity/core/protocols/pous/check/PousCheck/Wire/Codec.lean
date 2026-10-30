import PousCheck.Randomness

/-!
# The decode an audit with verification flag bit 0 checks

P2 v4's maps and keys (`reference.expand_prefix`, `stream`, `keys`, `d`, `sigma`, `decode`), at the spec's prefix
(the model commitment and the 32-byte setup salt):

~~~text
prefix    Z("verity/pous/p2/keys/v4\0" ‖ frame("p2/v4") ‖ frame(prime id) ‖ frame(model) ‖ frame(salt)), Z zero-pads to
          whole 128-byte blocks
base      prefix ‖ u64be(segment) ‖ u64be(local block)
stream_P  SHA512(base ‖ P ‖ u32be 0) ‖ SHA512(base ‖ P ‖ u32be 1) ‖ …                      P ∈ {"K", "t"}
K, t      K the first w/8 bytes of stream_K; t the first w/8-byte word of stream_t below p
D(c)      (d(σ_K(d(c))) − t) mod p, d(z) = ±z² mod p (− for odd z), σ_K(r) = r ⊕ K when below p, else r
~~~

The audit needs only `D` on one block. `PousCheck.Keys` and `PousCheck.P2` (the setup check) state the same maps at both
prefixes; when both are in one tree this module is theirs at the spec's prefix.
-/

namespace PousCheck.Wire.Codec

open Randomness

def KEYS_TAG : ByteArray := "verity/pous/p2/keys/v4".toUTF8.push 0
def PROTOCOL := "p2/v4"

/-- The setup's key-stream prefix, zero-padded to whole 128-byte blocks. -/
def expandPrefix (primeId : String) (model salt : ByteArray) : ByteArray :=
  let body := KEYS_TAG ++ frame (.str PROTOCOL) ++ frame (.str primeId) ++ frame (.bytes model) ++ frame (.bytes salt)
  body ++ ⟨Array.replicate ((128 - body.size % 128) % 128) 0⟩

def be32 (x : Nat) : ByteArray := ⟨(Array.range 4).map fun i => (x >>> (8 * (3 - i))).toUInt8⟩

def natOfBE (b : ByteArray) : Nat := b.foldl (fun acc x => acc * 256 + x.toNat) 0

/-- Bytes `[lo, hi)` of `stream_P` at `base`. -/
def streamRange (base : ByteArray) (purpose : String) (lo hi : Nat) : ByteArray :=
  let head := base ++ purpose.toUTF8
  let first := lo / 64
  ((List.range ((hi + 63) / 64 - first)).foldl (fun acc i => acc ++ Flock.Sha512.hash (head ++ be32 (first + i)))
    ByteArray.empty).extract (lo - 64 * first) (hi - 64 * first)

/-- `t`: the first `n`-byte candidate of `stream_t` from candidate `a` on that is below `p`. The counter is a `u32be`, so
`fuel` is the candidates the stream holds; running out (probability `(c0 / 2^w)^⌊2^38 / n⌋`) is `none`, where Python's
`to_bytes` raises. -/
def below (base : ByteArray) (n p : Nat) : (fuel a : Nat) → Option Nat
  | 0, _ => none
  | fuel + 1, a =>
    let x := natOfBE (streamRange base "t" (a * n) ((a + 1) * n))
    if x < p then some x else below base n p fuel (a + 1)

/-- Block `(segment, block)`'s `(K, t)` for a width of `n` bytes and prime `p`. -/
def keys (primeId : String) (model salt : ByteArray) (n p segment block : Nat) : Option (Nat × Nat) :=
  let base := expandPrefix primeId model salt ++ (Randomness.be8 segment) ++ (Randomness.be8 block)
  (below base n p (64 * 2 ^ 32 / n) 0).map fun t => (natOfBE (streamRange base "K" 0 n), t)

/-- The signed squaring `d(z) = ±z² mod p`, negated for odd `z`. -/
def d (p z : Nat) : Nat := let s := z * z % p; if z % 2 = 0 then s else (p - s) % p

/-- `σ_K(r)`: `r ⊕ K` when that is below `p`, else `r`. -/
def sigma (p r k : Nat) : Nat := let x := r ^^^ k; if x < p then x else r

/-- `D(c) = (d(σ_K(d(c))) − t) mod p`, on `c < p` and `t < p`. -/
def decode (p c k t : Nat) : Nat := (d p (sigma p (d p c) k) + p - t) % p

end PousCheck.Wire.Codec
