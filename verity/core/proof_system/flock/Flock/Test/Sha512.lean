/-
The straight-line `Sha512.compress` against the three loops it replaced (`compressLoops`, kept here as the test's
reference only) on 200 chained blocks and on blocks at unaligned offsets, and `Sha512.hash` on the FIPS 180-4 "abc" digest.
-/
import Flock.Bytes
import Flock.Hash

namespace Flock.Test.Sha512

open Flock.Sha512 (k rotr)

/-- `Sha512.compress` as it was: the schedule in an `Array UInt64`, the rounds a `for` loop. -/
def compressLoops (h : Array UInt64) (data : ByteArray) (off : Nat) : Array UInt64 := Id.run do
  let mut w : Array UInt64 := Array.mkEmpty 80
  for i in [0:16] do
    let mut x : UInt64 := 0
    for j in [0:8] do x := (x <<< 8) ||| (data.get! (off + 8 * i + j)).toUInt64
    w := w.push x
  for i in [16:80] do
    let x := w[i - 15]!
    let y := w[i - 2]!
    let s0 := rotr x 1 ^^^ rotr x 8 ^^^ (x >>> 7)
    let s1 := rotr y 19 ^^^ rotr y 61 ^^^ (y >>> 6)
    w := w.push (w[i - 16]! + s0 + w[i - 7]! + s1)
  let mut a := h[0]!
  let mut b := h[1]!
  let mut c := h[2]!
  let mut d := h[3]!
  let mut e := h[4]!
  let mut f := h[5]!
  let mut g := h[6]!
  let mut hh := h[7]!
  for i in [0:80] do
    let s1 := rotr e 14 ^^^ rotr e 18 ^^^ rotr e 41
    let ch := (e &&& f) ^^^ ((~~~e) &&& g)
    let t1 := hh + s1 + ch + k[i]! + w[i]!
    let s0 := rotr a 28 ^^^ rotr a 34 ^^^ rotr a 39
    let maj := (a &&& b) ^^^ (a &&& c) ^^^ (b &&& c)
    let t2 := s0 + maj
    hh := g
    g := f
    f := e
    e := d + t1
    d := c
    c := b
    b := a
    a := t1 + t2
  return #[h[0]! + a, h[1]! + b, h[2]! + c, h[3]! + d, h[4]! + e, h[5]! + f, h[6]! + g, h[7]! + hh]

/-- 128 pseudorandom bytes of seed `s` (xorshift64). -/
def block (s : Nat) : ByteArray := Id.run do
  let mut b := ByteArray.emptyWithCapacity 128
  let mut x : UInt64 := s.toUInt64 * 0x9e3779b97f4a7c15 + 1
  for _ in [0:128] do
    x := x ^^^ (x <<< 13)
    x := x ^^^ (x >>> 7)
    x := x ^^^ (x <<< 17)
    b := b.push x.toUInt8
  return b

/-- Both compressions over the 200 blocks of seeds `0..199`, chained from `iv`: every chaining value agrees. -/
def agreeChained : Bool := Id.run do
  let mut h := Flock.Sha512.iv
  let mut ok := true
  for s in [0:200] do
    let h' := Flock.Sha512.compress h (block s) 0
    ok := ok && h' == compressLoops h (block s) 0
    h := h'
  return ok

#guard agreeChained

#guard
  let data := block 7 ++ block 8 ++ block 9
  [Flock.Sha512.iv, Flock.Sha512.compress Flock.Sha512.iv (block 10) 0].all fun h =>
    [0, 1, 5, 64, 127, 200, 256].all fun off =>
    Flock.Sha512.compress h data off == compressLoops h data off

#guard Flock.hexOf (Flock.Sha512.hash "abc".toUTF8) == "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"

end Flock.Test.Sha512
