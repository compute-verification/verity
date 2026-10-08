import PousCheck.Params
import PousCheck.Prim.Bytes

/-!
P2's parameters and maps (`p2/PROTOCOL.md` §§1–3), as `verity/core/protocols/pous/schemes/p2.py` computes them for
the codec `p2-16448/v4`:

~~~text
p          2^w − c0
d(z)       z² mod p when z is even, (−z²) mod p when z is odd             Python's `sq`
σ_K(r)     r ⊕ K if r ⊕ K < p, else r                                     the rejection branch never reduces mod p
D_j(c)     (d(σ_K(d(c))) − t_j) mod p                                     for canonical c < p only
~~~

A block is `w/8` bytes, unsigned big-endian; a payload is `(w − 16)/8` bytes. `decodeBlock` refuses what `decode_segment`
refuses, in its order: a block of another width, a block `≥ p`, and a result of more than `8·payloadBytes` bits.

Python's `_Field.square` reduces `a²` by two folds of `c0` and one conditional subtraction, which is `a² mod p` only while
`c0² < 2^w − 2·c0` or so; here `square` is the spec's `a² mod p`, and `tests/test_setup_difftest.py` keeps to widths
where the two agree.
-/

namespace PousCheck.P2

/-- The codec `p2-16448/v4` at any width: `segments` segments of `blocksPerSegment` blocks of `w` bits. Only the
counter-mode SHA-512 expander (`sha512-ctr-v4`); v2's ChaCha8 is nonconforming and not ported. -/
structure Params where
  w : Nat
  c0 : Nat
  blocksPerSegment : Nat
  segments : Nat := 1
  deriving Repr, BEq, Inhabited

def EXPANDER : String := "sha512-ctr-v4"

namespace Params

/-- `P2Params.__post_init__`'s checks, with its messages. -/
def validate (q : Params) : Except String Params := do
  for (name, v) in [("w", q.w), ("c0", q.c0), ("blocks_per_segment", q.blocksPerSegment), ("segments", q.segments)] do
    if v < 1 then throw s!"{name} must be a positive integer"
  if q.w % 8 != 0 || q.w ≤ 16 then throw "w is a whole number of bytes above 16 bits"
  if q.c0 % 4 != 1 || q.c0 ≥ 2 ^ 16 then throw "c0 ≡ 1 mod 4 (so p ≡ 3 mod 4) and below 2^16"
  if q.blocksPerSegment > 2 ^ 16 then throw "block indices within a segment are 16 bits (the verifier's leaves)"
  return q

def p (q : Params) : Nat := 2 ^ q.w - q.c0
def blockBytes (q : Params) : Nat := q.w / 8
def payloadBytes (q : Params) : Nat := (q.w - 16) / 8
def blocks (q : Params) : Nat := q.segments * q.blocksPerSegment
def segmentBytes (q : Params) : Nat := q.blocksPerSegment * q.payloadBytes
def wBytes (q : Params) : Nat := q.segments * q.segmentBytes
def primeId (q : Params) : String := s!"p2-p{q.w}-c{q.c0}"
def profile (q : Params) : String := s!"p2-{q.payloadBytes}B"

/-- The registered point, `p2-p16448-c21065` with `2^16` blocks per segment and the certificate's 8 segments. -/
def deployed : Params := ⟨W_BITS, 21065, 2 ^ 16, 8⟩

end Params

/-- `a² mod p`. -/
def square (p a : Nat) : Nat := a * a % p

/-- The signed squaring `d` (Python's `sq`): `a² mod p`, negated when `a` is odd. -/
def d (p a : Nat) : Nat :=
  let s := square p a
  if a % 2 == 1 then (p - s) % p else s

/-- `σ_K(r)`: `r ⊕ K` when that is below `p`, else `r` unchanged. -/
def sigma (p r k : Nat) : Nat :=
  let x := r ^^^ k
  if x < p then x else r

/-- `D(c) = (d(σ_K(d(c))) − t) mod p` on a canonical `c < p` and `t < p`. -/
def decodeValue (p k t c : Nat) : Nat := (d p (sigma p (d p c) k) + p - t) % p

/-- Why a block is not a codeword: `decode_segment`'s three refusals. -/
inductive Refusal where
  | width (j : Nat)
  | noncanonical (j : Nat)
  | payload (j : Nat)
  deriving Repr, BEq, DecidableEq

def Refusal.kind : Refusal → String
  | .width _ => "width"
  | .noncanonical _ => "noncanonical"
  | .payload _ => "payload"

def Refusal.message (q : Params) : Refusal → String
  | .width j => s!"block {j} is not {q.blockBytes} bytes"
  | .noncanonical j => s!"block {j} is not below p"
  | .payload j => s!"block {j} decodes to more than {8 * q.payloadBytes} bits: not a codeword"

/-- Block `j`'s payload under keys `(k, t)`: `payloadBytes` bytes, or why the block is refused. -/
def decodeBlock (q : Params) (j : Nat) (k t : Nat) (block : ByteArray) : Except Refusal ByteArray := do
  if block.size != q.blockBytes then throw (.width j)
  let c := Prim.natOfBE block
  if c ≥ q.p then throw (.noncanonical j)
  let m := decodeValue q.p k t c
  if m >>> (8 * q.payloadBytes) != 0 then throw (.payload j)
  return Prim.beOfNat m q.payloadBytes

end PousCheck.P2
