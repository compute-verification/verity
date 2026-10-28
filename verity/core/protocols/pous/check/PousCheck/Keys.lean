import Identity.Sha512
import Randomness
import PousCheck.P2

/-!
P2 v4's per-block keys (`schemes/p2.py`'s `expand_prefix`, `stream`, `below`, `sha512_keys`):

~~~text
prefix   Z(body), Z zero-pads to whole 128-byte blocks
base     prefix ‖ u64be(segment) ‖ u64be(local block)
stream_P SHA512(base ‖ P ‖ u32be 0) ‖ SHA512(base ‖ P ‖ u32be 1) ‖ …            P ∈ {"K", "t"}
K        the first w/8 bytes of stream_K, big-endian
t        the first w/8-byte word of stream_t below p: candidate a is bytes [a·n, (a+1)·n)
~~~

Two prefix bodies, the same streams after them:

- the package's (`expand_prefix`, what `p2-16448/v4` runs): `"verity/pous/p2/keys/v4\0" ‖ frame("p2/v4") ‖
  frame(prime id) ‖ frame(salt)`, its 24-byte salt from the generic `setup.draw_salt`;
- the spec's (`reference.py`, and `vectors.json`): the same with `frame(model commitment) ‖ frame(setup salt)`, 64 and 32 bytes.

The prefix is absorbed once (`Identity.Sha512`'s midstate) and each chunk is one compression after it. The counter is a
`u32be`, so a stream has at most `2^32` chunks: `t` reads candidates while they fit, and a stream with none below `p` among
them is an error here, an `OverflowError` in Python (probability `(c0/2^w)^⌊2^38/n⌋`).
-/

namespace PousCheck.Keys

open PousCheck.Prim Identity
open FlockSoundness.Randomness (frame)

def KEYS_TAG : ByteArray := "verity/pous/p2/keys/v4".toUTF8.push 0
def PROTOCOL_ID : String := "p2/v4"
def SALT_BYTES : Nat := 24

/-- `body` zero-padded to whole 128-byte blocks. -/
def pad128 (body : ByteArray) : ByteArray := body ++ zeros ((128 - body.size % 128) % 128)

/-- The package's prefix (`expand_prefix`): its salt is `SALT_BITS / 8 = 24` bytes. -/
def packagePrefix (q : P2.Params) (salt : ByteArray) : Except String ByteArray := do
  if salt.size != SALT_BYTES then
    throw s!"the salt is {SALT_BYTES} bytes, drawn by the verifier after W is committed"
  return pad128 (KEYS_TAG ++ frame (.str PROTOCOL_ID) ++ frame (.str q.primeId) ++ frame (.bytes salt))

/-- The spec's prefix (`reference.expand_prefix`): the model commitment and the 32-byte setup salt. -/
def specPrefix (q : P2.Params) (model setupSalt : ByteArray) : ByteArray :=
  pad128 (KEYS_TAG ++ frame (.str PROTOCOL_ID) ++ frame (.str q.primeId) ++ frame (.bytes model)
    ++ frame (.bytes setupSalt))

/-- A setup's key expander: the prefix's midstate. -/
structure Expander where
  mid : Sha512.State

def Expander.ofPrefix (pre : ByteArray) : Expander := ⟨Sha512.init.update pre⟩

/-- Chunk `i` of `stream_P` for block `j` of segment `s`. -/
def Expander.chunk (e : Expander) (s j : Nat) (purpose : UInt8) (i : Nat) : ByteArray :=
  (e.mid.update (Flock.be64 s.toUInt64 ++ Flock.be64 j.toUInt64 ++ ByteArray.mk #[purpose]
    ++ Flock.be32 i.toUInt32)).finalize

/-- Bytes `[lo, hi)` of `stream_P`. A chunk's input after the prefix's midstate is 21 bytes, so with the prefix whole
blocks (`pad128`) each chunk is one compression of the same padded block with its last four bytes, the counter, set: the
block is built once per stream. -/
def Expander.range (e : Expander) (s j : Nat) (purpose : UInt8) (lo hi : Nat) : ByteArray := Id.run do
  let first := lo / 64
  let mut out := ByteArray.emptyWithCapacity (hi - lo + 128)
  let tail := e.mid.update (Flock.be64 s.toUInt64 ++ Flock.be64 j.toUInt64 ++ ByteArray.mk #[purpose] ++ Flock.be32 0)
  let mut block := tail.padding
  let at_ := tail.buf.size - 4
  if block.size != 128 then
    for i in [first:(hi + 63) / 64] do out := out ++ e.chunk s j purpose i
  else
    for i in [first:(hi + 63) / 64] do
      for b in [0:4] do block := block.set! (at_ + b) (i >>> (8 * (3 - b))).toUInt8
      out := out ++ Sha512.digest (Flock.Sha512.compress tail.h block 0)
  return out.extract (lo - 64 * first) (hi - 64 * first)

def PURPOSE_K : UInt8 := 0x4b
def PURPOSE_T : UInt8 := 0x74

-- `range` is its chunks in order, after a prefix of whole blocks (one compression each) and after any other (`chunk`)
#guard [128, 3].all fun n => let e := Expander.ofPrefix (zeros n)
  e.range 5 7 PURPOSE_T 30 300 == ((List.range 5).foldl (· ++ e.chunk 5 7 PURPOSE_T ·) .empty).extract 30 300

/-- The chunks a stream may have: its counter is a `u32be`. -/
def STREAM_BYTES : Nat := 64 * 2 ^ 32

/-- `t`: the first `n`-byte candidate of `stream_t` from candidate `a` on that is below `p`, and its index. Each step reads
one more candidate; `fuel` bounds the candidates left in the stream. -/
def Expander.below (e : Expander) (s j n p : Nat) : (fuel a : Nat) → Except String (Nat × Nat)
  | 0, _ => throw s!"stream_t of block ({s}, {j}) has no candidate below p within 2^32 chunks"
  | fuel + 1, a =>
    let x := natOfBE (e.range s j PURPOSE_T (a * n) ((a + 1) * n))
    if x < p then pure (x, a) else e.below s j n p fuel (a + 1)

/-- Block `j` of segment `s`'s keys `(K, t)`, and the index of the candidate `t` is (`t_rejection_index`). -/
def Expander.keys (e : Expander) (q : P2.Params) (s j : Nat) : Except String (Nat × Nat × Nat) := do
  let n := q.blockBytes
  let k := natOfBE (e.range s j PURPOSE_K 0 n)
  let (t, a) ← e.below s j n q.p (STREAM_BYTES / n) 0
  return (k, t, a)

end PousCheck.Keys
