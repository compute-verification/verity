import Mathlib
import Definitions.Hash.Basic

/-!
# `hm96-sha512/v1` over a hash (trusted)

The hiding leaf (`verity/ml/commitments/hm96/PROTOCOL.md` §1 and §2, at `hm96-sha512/v1`'s sizes), with the hash a
parameter `H` (SHA-512 is the instance of record; the verifier's code is `Flock.Hm96`). For an inner digest `x`, a
192-byte salt `y` and the verifier's key:

    c    = H(saltPrefix ‖ y)                saltPrefix   = "verity/hm96-sha512/salt/v1\0"                 zero-padded to 128 bytes
    b    = x ⊕ M·y                          M[i][j]      = key bit (i + j),  i < 512, j < 1536
    leaf = H(leafPrefix key ‖ b ‖ c)        leafPrefix k = "verity/hm96-sha512/leaf/v1\0" ‖ H(k)          zero-padded to 128 bytes

Bit `t` of a byte string is bit `t mod 8` of byte `⌊t/8⌋`. The key is the verifier's, bound in every leaf through
`H(key)` in its prefix, never an opening's. `Hm96.hashed` lists the two messages the verifier hashes for an opening
beyond the inner digest: the salt's and the leaf's (`H(key)` is the same for every opening under that key).
-/

namespace Hash.Hm96

/-- The salt's bytes. -/
def SALT_BYTES : ℕ := 192

/-- Bit `t` of a byte string. -/
def bit (s : Msg) (t : ℕ) : Bool := (s.getD (t / 8) 0).toNat / 2 ^ (t % 8) % 2 == 1

/-- Bit `i` of `M·y`: `⊕_j key_{i+j} · y_j` over `j < 1536`. -/
def maskBit (key y : Msg) (i : ℕ) : Bool :=
  ((List.range (8 * SALT_BYTES)).filter (bit y)).foldl (fun acc j => xor acc (bit key (i + j))) false

/-- `M·y`, 64 bytes. -/
def mask (key y : Msg) : Msg :=
  (List.range 64).map fun p => ((List.range 8).foldl (fun acc s => acc + if maskBit key y (8 * p + s) then 2 ^ s else 0) 0).toUInt8

/-- The salt digest's first block. -/
def saltPrefix : Msg := padTo 128 (cstr "verity/hm96-sha512/salt/v1")

/-- The leaf's first block, binding the key. -/
def leafPrefix (H : Msg → Digest) (key : Msg) : Msg := padTo 128 (cstr "verity/hm96-sha512/leaf/v1" ++ (H key).bytes)

/-- The commit string `b ‖ c`. -/
def commitString (H : Msg → Digest) (key : Msg) (x : Digest) (y : Msg) : Msg :=
  (List.ofFn fun i : Fin 64 => x i ^^^ (mask key y).getD i 0) ++ (H (saltPrefix ++ y)).bytes

/-- The hm96 leaf over the inner digest `x` and the salt `y`. -/
def leaf (H : Msg → Digest) (key : Msg) (x : Digest) (y : Msg) : Digest := H (leafPrefix H key ++ commitString H key x y)

/-- The salt's message and the leaf's. -/
def hashed (H : Msg → Digest) (key : Msg) (x : Digest) (y : Msg) : List Msg :=
  [saltPrefix ++ y, leafPrefix H key ++ commitString H key x y]

end Hash.Hm96
