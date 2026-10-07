/-
PROTOCOL.md §2.1, §2.2, §2.4: the fields.
-/
namespace Flock

/-- GF(2^128) = F2[x]/(x^128 + x^7 + x^2 + x + 1). Bit `i` of `hi·2^64 + lo` is the coefficient of `x^i`. -/
structure F128 where
  lo : UInt64
  hi : UInt64
  deriving BEq, Inhabited, Repr, DecidableEq, Hashable

namespace F128

def zero : F128 := ⟨0, 0⟩
def one : F128 := ⟨1, 0⟩

@[inline] def add (a b : F128) : F128 := ⟨a.lo ^^^ b.lo, a.hi ^^^ b.hi⟩

instance instAdd : Add F128 := ⟨add⟩
instance instOfNatOfNatNat : OfNat F128 0 := ⟨zero⟩
instance instOfNatOfNatNat_1 : OfNat F128 1 := ⟨one⟩

@[inline] def isZero (a : F128) : Bool := a.lo == 0 && a.hi == 0

/-- Bits `i, i + 1, …` of the carry-less product: for each set bit of `bb` (bit `i` of the multiplier) the 128-bit value
`a · x^i` is added into `(lo, hi)`. -/
def clmulGo (a : UInt64) : Nat → Nat → UInt64 → UInt64 → UInt64 → UInt64 × UInt64
  | 0, _, _, lo, hi => (lo, hi)
  | fuel + 1, i, bb, lo, hi =>
    if bb &&& 1 = 1 then
      clmulGo a fuel (i + 1) (bb >>> 1) (lo ^^^ (a <<< i.toUInt64)) (if i = 0 then hi else hi ^^^ (a >>> (64 - i).toUInt64))
    else clmulGo a fuel (i + 1) (bb >>> 1) lo hi

/-- Carry-less product of two 64-bit words, as (low word, high word). -/
def clmul (a b : UInt64) : UInt64 × UInt64 := clmulGo a 64 0 b 0 0

/-- Reduce the 256-bit carry-less product `r3·2^192 + r2·2^128 + r1·2^64 + r0` modulo the GHASH polynomial. -/
@[inline] def reduce (r0 r1 r2 r3 : UInt64) : F128 :=
  let tl := r2 ^^^ (r2 <<< 1) ^^^ (r2 <<< 2) ^^^ (r2 <<< 7)
  let th := r3 ^^^ ((r3 <<< 1) ||| (r2 >>> 63)) ^^^ ((r3 <<< 2) ||| (r2 >>> 62)) ^^^ ((r3 <<< 7) ||| (r2 >>> 57))
  let o := (r3 >>> 63) ^^^ (r3 >>> 62) ^^^ (r3 >>> 57)
  ⟨r0 ^^^ tl ^^^ o ^^^ (o <<< 1) ^^^ (o <<< 2) ^^^ (o <<< 7), r1 ^^^ th⟩

def mul (a b : F128) : F128 :=
  let (p0l, p0h) := clmul a.lo b.lo
  let (p2l, p2h) := clmul a.hi b.hi
  let (ml, mh) := clmul (a.lo ^^^ a.hi) (b.lo ^^^ b.hi)
  let m1l := ml ^^^ p0l ^^^ p2l
  let m1h := mh ^^^ p0h ^^^ p2h
  reduce p0l (p0h ^^^ m1l) (p2l ^^^ m1h) p2h

instance instMul : Mul F128 := ⟨mul⟩

def square (a : F128) : F128 := mul a a

/-- `k` square-and-multiply steps by `a` from `acc`: `acc^(2^k) · a^(2^k − 1)`. -/
def invGo (a : F128) : Nat → F128 → F128
  | 0, acc => acc
  | k + 1, acc => invGo a k (mul (square acc) a)

/-- `a^(2^128 − 2)`: the inverse, with `inv 0 = 0` (§2.1). 2^128 − 2 = 0b111…10: 127 ones, then one zero. -/
def inv (a : F128) : F128 := square (invGo a 127 one)

/-- The 16-byte encoding `le64(lo) ‖ le64(hi)`. -/
def toBytes (a : F128) : ByteArray :=
  ⟨Array.ofFn (n := 16) fun i =>
    if i.val < 8 then (a.lo >>> (8 * i.val).toUInt64).toUInt8 else (a.hi >>> (8 * (i.val - 8)).toUInt64).toUInt8⟩

def pushBytes (out : ByteArray) (a : F128) : ByteArray := out ++ a.toBytes

@[inline] def bit (a : F128) (i : Nat) : Bool :=
  if i < 64 then (a.lo >>> i.toUInt64) &&& 1 == 1 else (a.hi >>> (i - 64).toUInt64) &&& 1 == 1

/-- `x^i` for `i < 128`. -/
def monomial (i : Nat) : F128 :=
  if i < 64 then ⟨(1 : UInt64) <<< i.toUInt64, 0⟩ else ⟨0, (1 : UInt64) <<< (i - 64).toUInt64⟩

private def hexDigit (n : UInt64) : Char :=
  let d := n.toNat
  if d < 10 then Char.ofNat (48 + d) else Char.ofNat (87 + d)

/-- Record hex: `{hi:016x}{lo:016x}` (§5.1). -/
def toHex (a : F128) : String := Id.run do
  let mut s := ""
  for i in [0:16] do s := s.push (hexDigit ((a.hi >>> (4 * (15 - i)).toUInt64) &&& 0xf))
  for i in [0:16] do s := s.push (hexDigit ((a.lo >>> (4 * (15 - i)).toUInt64) &&& 0xf))
  return s

end F128

/-- GF(2^256) = F[u]/(u^2 + u + x^{-1}) (§2.4); an element is `c0 + c1·u`. -/
structure F256 where
  c0 : F128
  c1 : F128
  deriving BEq, Inhabited, Repr, DecidableEq

namespace F256

/-- `x^{-1}` in F. -/
def xinv : F128 := ⟨0x43, 0x8000000000000000⟩

def zero : F256 := ⟨0, 0⟩
def one : F256 := ⟨1, 0⟩
/-- `u`. -/
def u : F256 := ⟨0, 1⟩
def ofF (a : F128) : F256 := ⟨a, 0⟩

@[inline] def add (a b : F256) : F256 := ⟨a.c0 + b.c0, a.c1 + b.c1⟩

def mul (a b : F256) : F256 :=
  let p0 := a.c0 * b.c0
  let p1 := a.c1 * b.c1
  let p2 := (a.c0 + a.c1) * (b.c0 + b.c1)
  ⟨p0 + xinv * p1, p2 + p0⟩

/-- `K · F`, coordinate-wise. -/
@[inline] def smul (a : F256) (s : F128) : F256 := ⟨a.c0 * s, a.c1 * s⟩

instance instAdd : Add F256 := ⟨add⟩
instance instMul : Mul F256 := ⟨mul⟩
instance instOfNatOfNatNat : OfNat F256 0 := ⟨zero⟩
instance instOfNatOfNatNat_1 : OfNat F256 1 := ⟨one⟩

end F256

/-- `φ8(α) = g`, a root of `X^8 + X^4 + X^3 + X + 1` in F (§2.2). -/
def phi8Gen : F128 := ⟨0x6b8330483c2e9849, 0x0dcb364640a222fe⟩

/-- `g^i`. -/
def phi8Pow : Nat → F128
  | 0 => F128.one
  | i + 1 => phi8Pow i * phi8Gen

/-- `Σ_{i < n} bit_i(v) · g^i`. -/
def phi8Sum (v : Nat) : Nat → F128
  | 0 => F128.zero
  | i + 1 => if v.testBit i then phi8Sum v i + phi8Pow i else phi8Sum v i

/-- `φ8(v) = Σ bit_i(v) · g^i` for bytes `v` (§2.2), tabulated. -/
def phi8Table : Array F128 := Array.ofFn (n := 256) fun v => phi8Sum v.val 8

@[inline] def phi8 (v : Nat) : F128 := phi8Table.getD (v % 256) 0

end Flock
