/-
SHA-512's compression (FIPS 180-4 §6.4.2) on 64-bit vectors, in the builder's shape (`Flock.HmNets`): the schedule one word at
a time, the rounds from constant `t`, the feed-forward. `Compress` proves the builder's rows force it.
-/
import Flock.HmNets

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

/-- SHA-512's `Ch`. -/
def chV (e f g : BitVec 64) : BitVec 64 := (e &&& f) ^^^ (~~~e &&& g)

/-- SHA-512's `Maj`. -/
def majV (a b c : BitVec 64) : BitVec 64 := (a &&& b) ^^^ (a &&& c) ^^^ (b &&& c)

def bsig0 (x : BitVec 64) : BitVec 64 := x.rotateRight 28 ^^^ x.rotateRight 34 ^^^ x.rotateRight 39
def bsig1 (x : BitVec 64) : BitVec 64 := x.rotateRight 14 ^^^ x.rotateRight 18 ^^^ x.rotateRight 41
def ssig0 (x : BitVec 64) : BitVec 64 := x.rotateRight 1 ^^^ x.rotateRight 8 ^^^ x >>> 7
def ssig1 (x : BitVec 64) : BitVec 64 := x.rotateRight 19 ^^^ x.rotateRight 61 ^^^ x >>> 6

/-- Schedule word `w.length`: `w[t-16] + σ₀ w[t-15] + w[t-7] + σ₁ w[t-2]`. -/
def schedStepV (w : List (BitVec 64)) : List (BitVec 64) :=
  let t := w.length
  w ++ [w.getD (t - 16) 0 + ssig0 (w.getD (t - 15) 0) + w.getD (t - 7) 0 + ssig1 (w.getD (t - 2) 0)]

def scheduleV : Nat → List (BitVec 64) → List (BitVec 64)
  | 0, w => w
  | n + 1, w => scheduleV n (schedStepV w)

def roundV (k wt : BitVec 64) (r : Regs (BitVec 64)) : Regs (BitVec 64) :=
  let t1 := r.h + bsig1 r.e + chV r.e r.f r.g + k + wt
  let t2 := bsig0 r.a + majV r.a r.b r.c
  ⟨t1 + t2, r.a, r.b, r.c, r.d + t1, r.e, r.f, r.g⟩

def roundsV (w : List (BitVec 64)) : Nat → Nat → Regs (BitVec 64) → Regs (BitVec 64)
  | 0, _, r => r
  | n + 1, t, r => roundsV w n (t + 1) (roundV (kAt t) (w.getD t 0) r)

/-- One SHA-512 compression of the 16 message words `msg` from the chaining value `cv`, with the feed-forward. -/
def compressV (cv msg : List (BitVec 64)) : List (BitVec 64) :=
  List.zipWith (· + ·) cv (roundsV (scheduleV 64 msg) 80 0 (Regs.ofList cv)).toList

/-- Bit `p` of a list of words in word order: bit `p % 64` of word `p / 64`. -/
def wbit (ws : List (BitVec 64)) (p : Nat) : Bool := (ws.getD (p / 64) 0).getLsbD (p % 64)

/-- Bit `i` of `M·y` for the salt's bits `y` (`Hm96.mask`): `⊕_{j < 1536} key_{i+j} · y_j`. -/
def maskV (key : ByteArray) (y : Nat → Bool) (i : Nat) : Bool :=
  (List.range 1536).foldl (fun acc j => acc ^^ (Flock.Hm96.bit key (i + j) && y j)) false

end FlockSoundness.Discharge.Hm
