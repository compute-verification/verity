import PoubCheck.Basic

/-!
# The store's codec, decoding: `codec.py`

Sloth over one wide word (`verity/word-codec/kv/v0`), what a verifier runs: a codec seed's key, a bound block's offset
`b_j` (under the seed alone, or with its append's coin `c_a`), and `decode`. The prover's `encode` is not here.

~~~text
field      p = 2^n − c: p1024 = 2^1024 − 105, p512 = 2^512 − 569; a word S = n/8 bytes little-endian, W = 4096/S a block
XOF(m)     SHA512(m ‖ u32be 0) ‖ SHA512(m ‖ u32be 1) ‖ …
c_r        LE(XOF(DOMAIN ‖ u8 len(name) ‖ name ‖ seed ‖ u32be r)[0 : S + 8]) mod p, r = 0 … k − 1
τ          the same at r = 2^32 − 1, and 1 in place of 0;  t(i) = τ · i mod p
K          masked: LE(XOF(… ‖ u32be 2^32 − 2)[0 : S]);  σ_K(x) = x ⊕ K if that is below p, else x
b_j        LE(XOF(… ‖ u32be 2^32 − 3 ‖ u64be j ‖ d_j)[0 : 64]) mod p;  with c_a, … ‖ u32be 2^32 − 4 ‖ u64be j ‖ d_j ‖ c_a
ρ⁻¹(y)     y² if y is even, p − y² if y is odd
decode     x ↦ ρ⁻¹(x) − c_r − t(i) − b_j for r = k − 1 … 0; masked, x ↦ σ_K(x) after each round r ≥ 1
d_j        sha512/row/v2 of the payload: SHA512(P(32768) ‖ payload)
~~~

`decode` refuses (`none`) a block with a word at or above p, and, under a bound key, one whose payload does not have
the digest `d_j` it was decoded under. Arguments `codec.py` raises on (a block of another size, an index past
`max_blocks`, a digest or coin of another size) are an `.error`.
-/

namespace PoubCheck.Codec

def BLOCK_BYTES : Nat := 4096
def SEED_BYTES : Nat := 32
def COIN_BYTES : Nat := 32
def TWEAK_INDEX : Nat := 2 ^ 32 - 1
def MASK_INDEX : Nat := 2 ^ 32 - 2
def BIND_INDEX : Nat := 2 ^ 32 - 3
def COIN_INDEX : Nat := 2 ^ 32 - 4
def DOMAIN : ByteArray := tag0 "verity/word-codec/kv/v0"

structure Field where
  name : String
  n : Nat
  c : Nat
deriving BEq, Repr

def Field.p (f : Field) : Nat := 2 ^ f.n - f.c
def Field.wordBytes (f : Field) : Nat := f.n / 8
def Field.blockWords (f : Field) : Nat := BLOCK_BYTES / f.wordBytes

def P1024 : Field := ⟨"p1024", 1024, 105⟩
def P512 : Field := ⟨"p512", 512, 569⟩

def field? (name : String) : Option Field := [P1024, P512].find? (·.name == name)

structure Codec where
  field : Field
  rounds : Nat
  masked : Bool
deriving BEq, Repr

/-- `Codec(field, rounds, masked)`: refused unless the field is one of `FIELDS` and `2 ≤ rounds < 2^32 − 4`. -/
def Codec.make? (fieldName : String) (rounds : Nat) (masked : Bool) : Option Codec := do
  let f ← field? fieldName
  if 2 ≤ rounds ∧ rounds < COIN_INDEX then some ⟨f, rounds, masked⟩ else none

/-- What a registration binds: `p1024/k3/masked`, `p512/k2`. -/
def Codec.name (c : Codec) : String := s!"{c.field.name}/k{c.rounds}" ++ (if c.masked then "/masked" else "")

/-- The most blocks a store under it holds: its words' indices fit a u64. -/
def Codec.maxBlocks (c : Codec) : Nat := 2 ^ 64 / c.field.blockWords

def xof (m : ByteArray) (len : Nat) : ByteArray :=
  ((List.range ((len + 63) / 64)).foldl (fun acc c => acc ++ sha512 (m ++ be 4 c)) ByteArray.empty).extract 0 len

def message (f : Field) (seed : ByteArray) (r : Nat) : ByteArray :=
  let name := f.name.toUTF8
  DOMAIN ++ be 1 name.size ++ name ++ seed ++ be 4 r

def element (f : Field) (seed : ByteArray) (r : Nat) : Nat := leNat (xof (message f seed r) (f.wordBytes + 8)) % f.p

/-- A codec seed's round constants, its tweak multiplier `τ`, masked its mask `K`, and the seed. -/
structure Key where
  codec : Codec
  c : Array Nat
  tau : Nat
  mask : Option Nat
  seed : ByteArray

/-- `codec.key(seed, codec)`, for a 32-byte seed. -/
def key (seed : ByteArray) (codec : Codec) : Key :=
  let f := codec.field
  let tau := element f seed TWEAK_INDEX
  { codec, seed, c := (Array.range codec.rounds).map (element f seed), tau := if tau = 0 then 1 else tau,
    mask := if codec.masked then some (leNat (xof (message f seed MASK_INDEX) f.wordBytes)) else none }

def Key.sigma (k : Key) (x : Nat) : Nat :=
  match k.mask with
  | some m => let v := x ^^^ m; if v < k.codec.field.p then v else x
  | none => x

/-- `b_j`, the offset of block `j`'s bound key from its payload's digest `d`, with its append's coin if given. -/
def binding (d : ByteArray) (j : Nat) (k : Key) (coin : Option ByteArray) : Nat :=
  let f := k.codec.field
  let m := match coin with
    | none => message f k.seed BIND_INDEX ++ be 8 j ++ d
    | some c => message f k.seed COIN_INDEX ++ be 8 j ++ d ++ c
  leNat (xof m 64) % f.p

def rhoInv (y p : Nat) : Nat := let q := y * y % p; if y % 2 = 0 then q else (p - q) % p

/-- `decode_word(y, i, k, b)` for a canonical `y`. -/
def decodeWord (y i : Nat) (k : Key) (b : Nat) : Nat := Id.run do
  let p := k.codec.field.p
  let t := (k.tau * i + b) % p
  let mut x := y
  for s in [0:k.c.size] do
    let r := k.c.size - 1 - s
    x := (rhoInv x p + 2 * p - k.c[r]! - t) % p
    if r ≥ 1 then x := k.sigma x
  return x

/-- `sha512_row_v2_digest(row, 8 · len(row))`: SHA512(P(l) ‖ row), `P(l)` `"verity/sha512-row/v2\0" ‖ u64be l`
zero-padded to 128 bytes. -/
def rowDigest (row : ByteArray) : ByteArray := sha512 (pad128 (tag0 "verity/sha512-row/v2" ++ be 8 (8 * row.size)) ++ row)

/-- `codec.digest(payload)`, `d_j`. -/
def digest (payload : ByteArray) : ByteArray := rowDigest payload

def words (block : ByteArray) (f : Field) : Array Nat :=
  let s := f.wordBytes
  (Array.range f.blockWords).map fun w => leNat (block.extract (s * w) (s * (w + 1)))

/-- `decode(block, j, k, d, coin)`: `bound` is `(d, coin)` for a block under its bound key. -/
def decode (block : ByteArray) (j : Nat) (k : Key) (bound : Option (ByteArray × Option ByteArray)) :
    Except String (Option ByteArray) := do
  let f := k.codec.field
  if j ≥ k.codec.maxBlocks then throw s!"a block index is below {k.codec.maxBlocks}"
  if block.size ≠ BLOCK_BYTES then throw "a block is 4096 bytes"
  let b ← match bound with
    | none => pure 0
    | some (d, coin) => do
      if d.size ≠ 64 then throw "a block's digest is 64 bytes"
      if let some c := coin then
        if c.size ≠ COIN_BYTES then throw "an append's coin is 32 bytes"
      pure (binding d j k coin)
  let ws := words block f
  if ws.any (· ≥ f.p) then return none
  let i0 := f.blockWords * j
  let payload := (Array.range ws.size).foldl (fun acc w => acc ++ le f.wordBytes (decodeWord ws[w]! (i0 + w) k b))
    ByteArray.empty
  match bound with
  | some (d, _) => return if digest payload == d then some payload else none
  | none => return some payload

end PoubCheck.Codec
