import Flock.Hash
import Flock.Draw

/-!
# `verity.ml.randomness` v2

The derivation the audit's challenges are drawn by (`p2/PROTOCOL.md` Appendix A, `verity/ml/randomness/__init__.py`),
executable on Flock's SHA-512:

~~~text
frame(x)            = tag || u64be(len(body)) || body          tags b (bytes), s (UTF-8), i (minimal big-endian, 0 is 0x00)
frames(index)       = frame(frame(p_1) || … || frame(p_m))
derive(src, d, ctx) = SHA512("verity/randomness/derive/v2\0" || frame(src) || frame(d)
                             || frame(frame(k_1) || frame(ctx[k_1]) || …))[0:32]          keys sorted
block_c(index)      = SHA512("verity/randomness/stream/v2\0" || key || frames(index) || u64be(c))
stream(index, n)    = the first n bytes of block_0 || block_1 || …
uniform(n, j)       = the first width-byte big-endian word of stream(("u", j)) below 2^(8·width) − 2^(8·width) mod n,
                      mod n; width = ⌈(bitlen(n) + 64) / 8⌉
~~~

These are `FlockSoundness.Randomness`'s definitions (`Security/Proofs/Flock/Soundness/Randomness.lean`), which the
program may not import (it is under `Security/`), with `uniform` reading the stream until a word is accepted as the
Python's unbounded stream does. They stand alone, on `Flock.Hash` and `Flock.Draw` only, so that a shared primitives
package can take them over.
-/

namespace PousCheck.Randomness

/-- A part of an index or a context value: bytes, a UTF-8 string, or a nonnegative integer. -/
inductive Part where
  | bytes (b : ByteArray)
  | str (s : String)
  | int (n : Nat)

/-- `x` as 8 bytes, big-endian. -/
def be8 (x : Nat) : ByteArray := ⟨(Array.range 8).map fun i => (x >>> (8 * (7 - i))).toUInt8⟩

/-- `n` as minimal big-endian bytes, `0` as one zero byte. -/
def natBytes (n : Nat) : ByteArray :=
  let len := max 1 (((if n = 0 then 0 else Nat.log2 n + 1) + 7) / 8)
  ⟨(Array.range len).map fun i => (n >>> (8 * (len - 1 - i))).toUInt8⟩

/-- `frame(part)`: `tag || len(body) as 8 bytes || body`, tags `b`, `s` and `i`. -/
def frame : Part → ByteArray
  | .bytes b => ⟨#[98]⟩ ++ be8 b.size ++ b
  | .str s => ⟨#[115]⟩ ++ be8 s.toUTF8.size ++ s.toUTF8
  | .int n => ⟨#[105]⟩ ++ be8 (natBytes n).size ++ natBytes n

/-- `_frames(index)`: the parts framed and concatenated, framed again as bytes. -/
def frames (index : List Part) : ByteArray := frame (.bytes (index.foldl (fun acc p => acc ++ frame p) ByteArray.empty))

def DERIVE : ByteArray := "verity/randomness/derive/v2".toUTF8.push 0
def STREAM : ByteArray := "verity/randomness/stream/v2".toUTF8.push 0

/-- `derive(source, domain, context)`: the context's pairs framed in key order, the 32-byte key SHA-512's first half. -/
def derive (source : ByteArray) (domain : String) (context : List (String × Part)) : ByteArray :=
  let ctx := ((context.toArray.qsort fun a b => a.1 < b.1).toList.foldl
    (fun acc kv => acc ++ frame (.str kv.1) ++ frame kv.2) ByteArray.empty)
  (Flock.Sha512.hash (DERIVE ++ frame (.bytes source) ++ frame (.str domain) ++ frame (.bytes ctx))).extract 0 32

/-- Block `c` of the counter-mode stream at `index`, 64 bytes. -/
def block (key : ByteArray) (index : List Part) (c : Nat) : ByteArray :=
  Flock.Sha512.hash (STREAM ++ key ++ frames index ++ be8 c)

/-- `Key.stream(index, n)`: the first `n` bytes of the stream at `index`. -/
def stream (key : ByteArray) (index : List Part) (n : Nat) : ByteArray :=
  ((List.range ((n + 63) / 64)).foldl (fun acc c => acc ++ block key index c) ByteArray.empty).extract 0 n

/-- `Key._uniform`'s reduction on the stream `s`, from its start: `none` when `s` runs out first. -/
def uniformOn (n : Nat) (s : ByteArray) : Option Nat := (Flock.Draw.uniform n s (s.size + 1) 0).map (·.1)

/-- `Key.uniform(n, index)`: the stream at `("u", *index)` read in doubling prefixes, from one block, until a word is
accepted. Every prefix of the stream is the Python's, so the first accepted word is; a word is refused with probability
below `2^-64`, so a prefix holding one word decides in practice. `none` only for `n = 0` or after 64 doublings. -/
def uniform (key : ByteArray) (n : Nat) (index : List Part) : Option Nat :=
  if n = 0 then none else go 64 ((Flock.Draw.width n + 63) / 64)
where
  go : Nat → Nat → Option Nat
    | 0, _ => none
    | fuel + 1, blocks =>
      match uniformOn n (stream key (.str "u" :: index) (64 * blocks)) with
      | some x => some x
      | none => go fuel (2 * blocks)

end PousCheck.Randomness
