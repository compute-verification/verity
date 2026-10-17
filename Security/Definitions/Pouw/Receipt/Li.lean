import Definitions.Pouw.Receipt.Duo

/-!
# li, a receipt fold (trusted)

li absorbs a word pair `(d₀, d₁)` into a chain's state `y`, one `n`-bit word, with an XOR key `k` and a multiplier `a`
(`benchmarks/pouw/receipts/receipt_folds.py`, `li_step`, at `n = 32`):

    y ← (y ⊕ d₀ ⊕ k)·a + d₁

An MMA's four words are two such half steps, `(d0, d1)` with `(k0, a0)` and then `(d2, d3)` with `(k1, a1)`, so a chain
of MMA steps is a chain of half steps. A thread runs several chains side by side; a checkpoint is the XOR of their states
at the end of a segment, each chain having absorbed that segment's words from the state it started the segment in.
-/

namespace Pouw.Receipt

/-- li's half step: the state after absorbing the word pair `d = (d₀, d₁)` into `y` with XOR key `k` and multiplier
`a`. -/
def liHalf {n : ℕ} (k a y : BitVec n) (d : BitVec n × BitVec n) : BitVec n :=
  (y ^^^ d.1 ^^^ k) * a + d.2

/-- A half step's XOR key, multiplier and word pair. -/
abbrev LiHalfStep (n : ℕ) : Type := BitVec n × BitVec n × (BitVec n × BitVec n)

/-- A half step whose multiplier is odd. -/
abbrev LiOddHalfStep (n : ℕ) : Type := BitVec n × OddKey n × (BitVec n × BitVec n)

/-- The half step an odd one is. -/
def LiOddHalfStep.val {n : ℕ} (h : LiOddHalfStep n) : LiHalfStep n := (h.1, h.2.1.1, h.2.2)

/-- A chain's state after absorbing the half steps `hs`, in order, from the state `y`. -/
def liChain {n : ℕ} (hs : List (LiHalfStep n)) (y : BitVec n) : BitVec n :=
  hs.foldl (fun y h => liHalf h.1 h.2.1 y h.2.2) y

/-- The checkpoint of the chains `cs`, each its half steps and its start state: the XOR of their final states. -/
def liCheckpoint {n : ℕ} (cs : List (List (LiHalfStep n) × BitVec n)) : BitVec n :=
  cs.foldl (fun x c => x ^^^ liChain c.1 c.2) 0

/-- A word whose lowest set bit is bit `v`: `x ≡ 2^v (mod 2^(v+1))`. -/
def LowBit (n v : ℕ) : Type := {x : BitVec n // x.setWidth (v + 1) = BitVec.twoPow (v + 1) v}

end Pouw.Receipt
