import Flock.Hash
import Flock.Draw

/-!
# `verity.randomness` v1, as a Lean spec

The byte formats and samplers of `packages/verity/src/verity/randomness/__init__.py`, executable, on the verifier's
FIPS 180-4 SHA-256 (`Flock.Sha256.hash`). The Python is tied to them by vectors: the spec reproduces the pinned
`packages/verity/tests/randomness/vectors.json`, and Python must match the spec's own vectors
(`backends/flock/tests/lean/RandomnessVectors.lean`).

* `frame`, `frames`: a part framed as `tag || len(body) as 8 bytes big-endian || body`; an index frames each part,
  concatenates them, and frames the result as bytes.
* `derive`: `SHA-256("verity/randomness/derive/v1\0" || frame(source) || frame(domain) || frame(ctx))`, `ctx` the
  context's `frame(key) || frame(value)` sorted by key.
* `block`, `stream`: block `c` of index `ι` is `SHA-256("verity/randomness/stream/v1\0" || key || frames(ι) || c as 8
  bytes big-endian)`, and `stream key ι n` the first `n` bytes of the blocks' concatenation.
* `uniformOn n s`: `Key._uniform`'s reduction, which is `Flock.Draw.uniform` from the start of `s`: `none` when `s` runs
  out, which the Python's unbounded stream never does.
* `subsetOn k n streams`, `subset`: `Key.subset`'s sparse partial Fisher–Yates, swap `i` drawing `uniformOn (n − i)` on
  its own stream, index `("s", *index, i)`, and the values sorted.
-/

namespace FlockSoundness.Randomness

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

def DERIVE : ByteArray := "verity/randomness/derive/v1".toUTF8.push 0
def STREAM : ByteArray := "verity/randomness/stream/v1".toUTF8.push 0
def SHARD : ByteArray := "verity/randomness/shard/v1".toUTF8.push 0

/-- `derive(source, domain, context)`: the context's pairs framed in key order. -/
def derive (source : ByteArray) (domain : String) (context : List (String × Part)) : ByteArray :=
  let ctx := ((context.toArray.qsort fun a b => a.1 < b.1).toList.foldl
    (fun acc kv => acc ++ frame (.str kv.1) ++ frame kv.2) ByteArray.empty)
  Flock.Sha256.hash (DERIVE ++ frame (.bytes source) ++ frame (.str domain) ++ frame (.bytes ctx))

/-- Block `c` of the counter-mode stream at `index`. -/
def block (key : ByteArray) (index : List Part) (c : Nat) : ByteArray :=
  Flock.Sha256.hash (STREAM ++ key ++ frames index ++ be8 c)

/-- `Key.stream(index, n)`: the first `n` bytes of the stream at `index`. -/
def stream (key : ByteArray) (index : List Part) (n : Nat) : ByteArray :=
  ((List.range ((n + 31) / 32)).foldl (fun acc c => acc ++ block key index c) ByteArray.empty).extract 0 n

/-- `Key.shard(index)`. -/
def shard (key : ByteArray) (index : List Part) : ByteArray := Flock.Sha256.hash (SHARD ++ key ++ frames index)

/-- `Key._uniform`'s reduction on the stream `s`, from its start. -/
def uniformOn (n : Nat) (s : ByteArray) : Option Nat := (Flock.Draw.uniform n s (s.size + 1) 0).map (·.1)

/-- `Key.subset`'s loop, swap `i` drawing from `streams i`. -/
def subsetOn (k n : Nat) (streams : Nat → ByteArray) : Option (Array Nat) := Id.run do
  let mut sw : Std.HashMap Nat Nat := {}
  for i in [0:k] do
    match uniformOn (n - i) (streams i) with
    | none => return none
    | some r =>
      let j := i + r
      let (vi, vj) := (sw.getD i i, sw.getD j j)
      sw := (sw.insert i vj).insert j vi
  return some (((Array.range k).map fun i => sw.getD i i).qsort (· < ·))

/-- The stream of swap `i` of a subset at `index`: `("s", *index, i)`. -/
def swapIndex (index : List Part) (i : Nat) : List Part := .str "s" :: index ++ [.int i]

/-- `Key.subset(k, n, index)`, each swap's stream read to `L` bytes. -/
def subset (key : ByteArray) (k n : Nat) (index : List Part) (L : Nat) : Option (Array Nat) :=
  subsetOn k n fun i => stream key (swapIndex index i) L

/-- `Key.uniform(n, index)`, its stream read to `L` bytes. -/
def uniform (key : ByteArray) (n : Nat) (index : List Part) (L : Nat) : Option Nat :=
  uniformOn n (stream key (.str "u" :: index) L)

/-- `Key.bernoulli(num/den, index)`, its stream read to `L` bytes: `p = 0` and `p = 1` draw nothing. -/
def bernoulli (key : ByteArray) (num den : Nat) (index : List Part) (L : Nat) : Option Bool :=
  if num = 0 then some false else if num = den then some true else
    (uniformOn den (stream key (.str "b" :: index) L)).map (· < num)

end FlockSoundness.Randomness
