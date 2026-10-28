import Flock.Keccak
import Flock.Hm96

/-!
Flock's `Keccak.shake256` and `Hm96.Default512.leaf` with their words unboxed, for the store binding's payload root
(`Store.expectedRoot`), which salts and hides every block of a whole node's store: Flock's keep Keccak's 25 lanes and
the mask's eight limbs in an `Array UInt64`, which boxes each word a round or a salt byte writes. Here the lanes are a
structure's scalar fields, one allocation a round, and the limbs a loop's arguments. `lean/tests/test_binding.py`
holds them to Python: `store_salt` to `hashlib.shake_256`, and the expected root to `audit.payload_tree`.
-/

namespace PoubCheck.Unboxed

open Flock

/-- Keccak-f[1600]'s state, lane `x + 5y` the field `a(x + 5y)`. -/
structure State where
  a0 : UInt64 := 0
  a1 : UInt64 := 0
  a2 : UInt64 := 0
  a3 : UInt64 := 0
  a4 : UInt64 := 0
  a5 : UInt64 := 0
  a6 : UInt64 := 0
  a7 : UInt64 := 0
  a8 : UInt64 := 0
  a9 : UInt64 := 0
  a10 : UInt64 := 0
  a11 : UInt64 := 0
  a12 : UInt64 := 0
  a13 : UInt64 := 0
  a14 : UInt64 := 0
  a15 : UInt64 := 0
  a16 : UInt64 := 0
  a17 : UInt64 := 0
  a18 : UInt64 := 0
  a19 : UInt64 := 0
  a20 : UInt64 := 0
  a21 : UInt64 := 0
  a22 : UInt64 := 0
  a23 : UInt64 := 0
  a24 : UInt64 := 0

/-- A left rotation by `1 ≤ n ≤ 63`. -/
@[inline] def rotl (x n : UInt64) : UInt64 := (x <<< n) ||| (x >>> (64 - n))

/-- One round of θ, ρ, π, χ and ι (`Keccak.permute`'s, `rc` its round constant): `c`, `d` are θ's column parities and
their mix, `b` lane `y + 5((2x + 3y) mod 5)` the rotated lane `x + 5y`, and χ, ι write the new lanes. -/
def round (s : State) (rc : UInt64) : State :=
  let c0 := s.a0 ^^^ s.a5 ^^^ s.a10 ^^^ s.a15 ^^^ s.a20
  let c1 := s.a1 ^^^ s.a6 ^^^ s.a11 ^^^ s.a16 ^^^ s.a21
  let c2 := s.a2 ^^^ s.a7 ^^^ s.a12 ^^^ s.a17 ^^^ s.a22
  let c3 := s.a3 ^^^ s.a8 ^^^ s.a13 ^^^ s.a18 ^^^ s.a23
  let c4 := s.a4 ^^^ s.a9 ^^^ s.a14 ^^^ s.a19 ^^^ s.a24
  let d0 := c4 ^^^ rotl c1 1
  let d1 := c0 ^^^ rotl c2 1
  let d2 := c1 ^^^ rotl c3 1
  let d3 := c2 ^^^ rotl c4 1
  let d4 := c3 ^^^ rotl c0 1
  let b0 := s.a0 ^^^ d0
  let b1 := rotl (s.a6 ^^^ d1) 44
  let b2 := rotl (s.a12 ^^^ d2) 43
  let b3 := rotl (s.a18 ^^^ d3) 21
  let b4 := rotl (s.a24 ^^^ d4) 14
  let b5 := rotl (s.a3 ^^^ d3) 28
  let b6 := rotl (s.a9 ^^^ d4) 20
  let b7 := rotl (s.a10 ^^^ d0) 3
  let b8 := rotl (s.a16 ^^^ d1) 45
  let b9 := rotl (s.a22 ^^^ d2) 61
  let b10 := rotl (s.a1 ^^^ d1) 1
  let b11 := rotl (s.a7 ^^^ d2) 6
  let b12 := rotl (s.a13 ^^^ d3) 25
  let b13 := rotl (s.a19 ^^^ d4) 8
  let b14 := rotl (s.a20 ^^^ d0) 18
  let b15 := rotl (s.a4 ^^^ d4) 27
  let b16 := rotl (s.a5 ^^^ d0) 36
  let b17 := rotl (s.a11 ^^^ d1) 10
  let b18 := rotl (s.a17 ^^^ d2) 15
  let b19 := rotl (s.a23 ^^^ d3) 56
  let b20 := rotl (s.a2 ^^^ d2) 62
  let b21 := rotl (s.a8 ^^^ d3) 55
  let b22 := rotl (s.a14 ^^^ d4) 39
  let b23 := rotl (s.a15 ^^^ d0) 41
  let b24 := rotl (s.a21 ^^^ d1) 2
  { a0 := b0 ^^^ (~~~b1 &&& b2) ^^^ rc,
    a1 := b1 ^^^ (~~~b2 &&& b3),
    a2 := b2 ^^^ (~~~b3 &&& b4),
    a3 := b3 ^^^ (~~~b4 &&& b0),
    a4 := b4 ^^^ (~~~b0 &&& b1),
    a5 := b5 ^^^ (~~~b6 &&& b7),
    a6 := b6 ^^^ (~~~b7 &&& b8),
    a7 := b7 ^^^ (~~~b8 &&& b9),
    a8 := b8 ^^^ (~~~b9 &&& b5),
    a9 := b9 ^^^ (~~~b5 &&& b6),
    a10 := b10 ^^^ (~~~b11 &&& b12),
    a11 := b11 ^^^ (~~~b12 &&& b13),
    a12 := b12 ^^^ (~~~b13 &&& b14),
    a13 := b13 ^^^ (~~~b14 &&& b10),
    a14 := b14 ^^^ (~~~b10 &&& b11),
    a15 := b15 ^^^ (~~~b16 &&& b17),
    a16 := b16 ^^^ (~~~b17 &&& b18),
    a17 := b17 ^^^ (~~~b18 &&& b19),
    a18 := b18 ^^^ (~~~b19 &&& b15),
    a19 := b19 ^^^ (~~~b15 &&& b16),
    a20 := b20 ^^^ (~~~b21 &&& b22),
    a21 := b21 ^^^ (~~~b22 &&& b23),
    a22 := b22 ^^^ (~~~b23 &&& b24),
    a23 := b23 ^^^ (~~~b24 &&& b20),
    a24 := b24 ^^^ (~~~b20 &&& b21) }

/-- Keccak-f[1600]: `Keccak.RC`'s 24 rounds. -/
def permute (s : State) : State := Id.run do
  let mut s := s
  for rc in Keccak.RC do s := round s rc
  return s

/-- The little-endian lane at byte `o` of `b`. -/
@[inline] def lane (b : ByteArray) (o : Nat) : UInt64 :=
  (b.get! o).toUInt64 ||| (b.get! (o + 1)).toUInt64 <<< 8 ||| (b.get! (o + 2)).toUInt64 <<< 16 |||
    (b.get! (o + 3)).toUInt64 <<< 24 ||| (b.get! (o + 4)).toUInt64 <<< 32 ||| (b.get! (o + 5)).toUInt64 <<< 40 |||
    (b.get! (o + 6)).toUInt64 <<< 48 ||| (b.get! (o + 7)).toUInt64 <<< 56

/-- `x`'s eight bytes, little-endian, after `out`. -/
@[inline] def pushLane (out : ByteArray) (x : UInt64) : ByteArray :=
  (((((((out.push x.toUInt8).push (x >>> 8).toUInt8).push (x >>> 16).toUInt8).push (x >>> 24).toUInt8).push
    (x >>> 32).toUInt8).push (x >>> 40).toUInt8).push (x >>> 48).toUInt8).push (x >>> 56).toUInt8

/-- The rate's 17 lanes of `s` XORed with the block of `p` at byte `o`. -/
def absorb (s : State) (p : ByteArray) (o : Nat) : State :=
  { s with
    a0 := s.a0 ^^^ lane p o, a1 := s.a1 ^^^ lane p (o + 8), a2 := s.a2 ^^^ lane p (o + 16),
    a3 := s.a3 ^^^ lane p (o + 24), a4 := s.a4 ^^^ lane p (o + 32), a5 := s.a5 ^^^ lane p (o + 40),
    a6 := s.a6 ^^^ lane p (o + 48), a7 := s.a7 ^^^ lane p (o + 56), a8 := s.a8 ^^^ lane p (o + 64),
    a9 := s.a9 ^^^ lane p (o + 72), a10 := s.a10 ^^^ lane p (o + 80), a11 := s.a11 ^^^ lane p (o + 88),
    a12 := s.a12 ^^^ lane p (o + 96), a13 := s.a13 ^^^ lane p (o + 104), a14 := s.a14 ^^^ lane p (o + 112),
    a15 := s.a15 ^^^ lane p (o + 120), a16 := s.a16 ^^^ lane p (o + 128) }

/-- The rate's 17 lanes of `s` as bytes, after `out`. -/
def squeeze (s : State) (out : ByteArray) : ByteArray :=
  let out := pushLane (pushLane (pushLane (pushLane out s.a0) s.a1) s.a2) s.a3
  let out := pushLane (pushLane (pushLane (pushLane out s.a4) s.a5) s.a6) s.a7
  let out := pushLane (pushLane (pushLane (pushLane out s.a8) s.a9) s.a10) s.a11
  let out := pushLane (pushLane (pushLane (pushLane out s.a12) s.a13) s.a14) s.a15
  pushLane out s.a16

/-- `Keccak.shake256 msg len`: pad10*1 after the suffix `1111`, absorb at rate 136, squeeze `len` bytes. -/
def shake256 (msg : ByteArray) (len : Nat) : ByteArray := Id.run do
  let mut p := msg.push 0x1F
  for _ in [0:(Keccak.RATE - p.size % Keccak.RATE) % Keccak.RATE] do p := p.push 0
  p := p.set! (p.size - 1) (p.get! (p.size - 1) ||| 0x80)
  let mut s : State := {}
  for blk in [0:p.size / Keccak.RATE] do
    s := permute (absorb s p (Keccak.RATE * blk))
  let mut out := ByteArray.emptyWithCapacity (len + Keccak.RATE)
  for i in [0:(len + Keccak.RATE - 1) / Keccak.RATE] do
    if i > 0 then s := permute s
    out := squeeze s out
  return out.extract 0 len

/-- `Hm96.Default512.mask`'s limbs from salt byte `p` on, `n` bytes of it left: limb `k` XORs in
`tables[8 (256 p + salt[p]) + k]` for each. -/
def maskFrom (salt : ByteArray) :
    Nat → Nat → UInt64 → UInt64 → UInt64 → UInt64 → UInt64 → UInt64 → UInt64 → UInt64 → ByteArray
  | 0, _, a0, a1, a2, a3, a4, a5, a6, a7 =>
    let out := ByteArray.emptyWithCapacity 64
    pushLane (pushLane (pushLane (pushLane (pushLane (pushLane (pushLane (pushLane out a0) a1) a2) a3) a4) a5) a6) a7
  | n + 1, p, a0, a1, a2, a3, a4, a5, a6, a7 =>
    let t := 8 * (256 * p + (salt.get! p).toNat)
    let T := Hm96.Default512.tables
    maskFrom salt n (p + 1) (a0 ^^^ T[t]!) (a1 ^^^ T[t + 1]!) (a2 ^^^ T[t + 2]!) (a3 ^^^ T[t + 3]!) (a4 ^^^ T[t + 4]!)
      (a5 ^^^ T[t + 5]!) (a6 ^^^ T[t + 6]!) (a7 ^^^ T[t + 7]!)

/-- `Hm96.Default512.mask`: `M·salt` as 64 bytes, over the salt's 192 bytes. -/
def mask (salt : ByteArray) : ByteArray := maskFrom salt 192 0 0 0 0 0 0 0 0 0

/-- `Hm96.Default512.leaf x salt`, `receipts.hidden_digest`: the hiding leaf of the digest `x` under `salt`. -/
def hiddenDigest (x salt : ByteArray) : ByteArray :=
  let m := mask salt
  let b := ByteArray.mk ((Array.range 64).map fun i => x.get! i ^^^ m.get! i)
  Sha512.hash (Hm96.Default512.leafPrefix ++ (b ++ Sha512.hash (Hm96.Default512.saltPrefix ++ salt)))

end PoubCheck.Unboxed
