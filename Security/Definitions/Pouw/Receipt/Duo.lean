import Mathlib.Data.BitVec

/-!
# duo, a receipt fold (trusted)

duo absorbs one word `d` into a chain's state `(y, z)`, two `n`-bit words, with keys `(k, a, r)`
(`benchmarks/pouw/receipts/receipt_folds.py`, `duo_word`, at `n = 32`):

    u = y ⊕ d ⊕ k;   y ← u·a + z;   z ← rotl(u, r) + y

pfill runs it with multipliers that pass `key_ok` and rotations 5, 11, 17, 29, and commits each chain's whole final
state; a served chain (fgemm, EGEMM, E2E) runs it with the fixed odd multipliers `SERVED_KA` and commits
`y ⊕ rotl(z, 16)`. A run's words are bound when two different runs leave different final states on some start state:
the map from runs to final-state functions is injective (no all-state kernel).
-/

namespace Pouw.Receipt

/-- duo's step: the state `s = (y, z)` after absorbing the word `d` with XOR key `k`, multiplier `a` and rotation `r`. -/
def duoWord {n : ℕ} (k a : BitVec n) (r : ℕ) (s : BitVec n × BitVec n) (d : BitVec n) : BitVec n × BitVec n :=
  ((s.1 ^^^ d ^^^ k) * a + s.2, (s.1 ^^^ d ^^^ k).rotateLeft r + ((s.1 ^^^ d ^^^ k) * a + s.2))

/-- An odd multiplier: a unit mod `2^n`. -/
def OddKey (n : ℕ) : Type := {a : BitVec n // a.toNat % 2 = 1}

/-- A rotation that moves every bit: `0 < r < n`. -/
def Rotation (n : ℕ) : Type := {r : ℕ // 0 < r ∧ r < n}

/-- A rotation that leaves the top three bits clear of its wrap: `0 < r ≤ n − 3`. -/
def ShortRotation (n : ℕ) : Type := {r : ℕ // 0 < r ∧ r + 3 ≤ n}

/-- An odd multiplier that is not `−1` modulo `2^r`. -/
def OddKeyNotNegOne (n r : ℕ) : Type := {a : BitVec n // a.toNat % 2 = 1 ∧ (a.toNat + 1) % 2 ^ r ≠ 0}

/-- Two words `d = (d₁, d₂)` from the state `s`, the first with keys `(k₁, a₁, r₁)` and the second with `(k₂, a₂, r₂)`. -/
def duoPair {n : ℕ} (k₁ a₁ : BitVec n) (r₁ : ℕ) (k₂ a₂ : BitVec n) (r₂ : ℕ) (d : BitVec n × BitVec n)
    (s : BitVec n × BitVec n) : BitVec n × BitVec n :=
  duoWord k₂ a₂ r₂ (duoWord k₁ a₁ r₁ s d.1) d.2

/-- Three words `d = (d₁, d₂, d₃)` from the state `s`, with keys `(kᵢ, aᵢ, rᵢ)` for word `i`: the first word, then the
pair. -/
def duoTriple {n : ℕ} (k₁ a₁ : BitVec n) (r₁ : ℕ) (k₂ a₂ : BitVec n) (r₂ : ℕ) (k₃ a₃ : BitVec n) (r₃ : ℕ)
    (d : BitVec n × BitVec n × BitVec n) (s : BitVec n × BitVec n) : BitVec n × BitVec n :=
  duoPair k₂ a₂ r₂ k₃ a₃ r₃ d.2 (duoWord k₁ a₁ r₁ s d.1)

/-- Four words `d = (d₁, d₂, d₃, d₄)` from the state `s`, with keys `(kᵢ, aᵢ, rᵢ)` for word `i`: the first word, then the
triple. -/
def duoQuad {n : ℕ} (k₁ a₁ : BitVec n) (r₁ : ℕ) (k₂ a₂ : BitVec n) (r₂ : ℕ) (k₃ a₃ : BitVec n) (r₃ : ℕ)
    (k₄ a₄ : BitVec n) (r₄ : ℕ) (d : BitVec n × BitVec n × BitVec n × BitVec n) (s : BitVec n × BitVec n) :
    BitVec n × BitVec n :=
  duoTriple k₂ a₂ r₂ k₃ a₃ r₃ k₄ a₄ r₄ d.2 (duoWord k₁ a₁ r₁ s d.1)

end Pouw.Receipt
