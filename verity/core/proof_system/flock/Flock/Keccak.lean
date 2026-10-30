/-
FIPS 202: Keccak-f[1600] and SHAKE256, for the salts of the values V*'s verifier registers itself
(`verity_flock.rec_residuals.verifier_salts`, Python's `hashlib.shake_256`).
-/
namespace Flock
namespace Keccak

def RC : Array UInt64 := #[0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
  0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
  0x0000000000000088, 0x0000000080008009, 0x000000008000000A, 0x000000008000808B, 0x800000000000008B,
  0x8000000000008089, 0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 0x000000000000800A,
  0x800000008000000A, 0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]

/-- ρ's rotation of lane `x + 5y`. -/
def ROT : Array UInt64 := #[0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14]

def rotl (x n : UInt64) : UInt64 := if n == 0 then x else (x <<< n) ||| (x >>> (64 - n))

/-- Keccak-f[1600] on 25 lanes, lane `x + 5y`: 24 rounds of θ, ρ, π, χ, ι. -/
def permute (a0 : Array UInt64) : Array UInt64 := Id.run do
  let mut a := a0
  for i in [0:24] do
    let c := (Array.range 5).map fun x => a[x]! ^^^ a[x + 5]! ^^^ a[x + 10]! ^^^ a[x + 15]! ^^^ a[x + 20]!
    let d := (Array.range 5).map fun x => c[(x + 4) % 5]! ^^^ rotl c[(x + 1) % 5]! 1
    a := (Array.range 25).map fun k => a[k]! ^^^ d[k % 5]!
    let mut b := Array.replicate 25 (0 : UInt64)
    for x in [0:5] do
      for y in [0:5] do
        b := b.set! (y + 5 * ((2 * x + 3 * y) % 5)) (rotl a[x + 5 * y]! ROT[x + 5 * y]!)
    a := (Array.range 25).map fun k =>
      let x := k % 5
      b[k]! ^^^ ((~~~ b[(x + 1) % 5 + (k - x)]!) &&& b[(x + 2) % 5 + (k - x)]!)
    a := a.set! 0 (a[0]! ^^^ RC[i]!)
  return a

/-- The little-endian lane at byte `off` of `b`. -/
def lane (b : ByteArray) (off : Nat) : UInt64 :=
  (List.range 8).foldl (fun acc j => acc ||| ((b.get! (off + j)).toUInt64 <<< (8 * j).toUInt64)) 0

/-- SHAKE256's rate in bytes. -/
def RATE : Nat := 136

/-- SHAKE256(`msg`), `len` bytes: pad10*1 after the suffix `1111`, absorb at rate 136, squeeze. -/
def shake256 (msg : ByteArray) (len : Nat) : ByteArray := Id.run do
  let mut p := msg.push 0x1F
  for _ in [0:(RATE - p.size % RATE) % RATE] do p := p.push 0
  p := p.set! (p.size - 1) (p.get! (p.size - 1) ||| 0x80)
  let mut a := Array.replicate 25 (0 : UInt64)
  for blk in [0:p.size / RATE] do
    for k in [0:RATE / 8] do a := a.set! k (a[k]! ^^^ lane p (RATE * blk + 8 * k))
    a := permute a
  let mut out := ByteArray.emptyWithCapacity len
  for _ in [0:(len + RATE - 1) / RATE] do
    for k in [0:RATE / 8] do
      for j in [0:8] do out := out.push ((a[k]! >>> (8 * j).toUInt64) &&& 0xff).toUInt8
    a := permute a
  return out.extract 0 len

end Keccak
end Flock
