import Definitions.Pouw.Receipt.Duo

/-!
# fx1, sm_120's receipt fold (trusted)

fx1 (SAT-120's design H1 + fx1, decision 83) puts one keyed IMAD in front of duo: word `d` at chain position `(J, e)`
is multiplied by an odd key `m = key_mul(1, (4J + e)·32)` and the product is absorbed by duo's step
(`Definitions/Pouw/Receipt/Duo.lean`) with that position's duo keys:

    d′ = d·m;   u = y ⊕ d′ ⊕ k;   y ← u·a + z;   z ← rotl(u, r) + y

For odd `m`, `d ↦ d·m` is a bijection of the words, so every statement about duo's words carries over to fx1's.
-/

namespace Pouw.Receipt

/-- fx1's step: duo's step (`duoWord`) on the word `d` times the multiplier `m`. -/
def fx1Word {n : ℕ} (k a : BitVec n) (r : ℕ) (m : BitVec n) (s : BitVec n × BitVec n) (d : BitVec n) :
    BitVec n × BitVec n :=
  duoWord k a r s (d * m)

/-- Two words `d = (d₁, d₂)` from the state `s`, the first with keys `(k₁, a₁, r₁, m₁)` and the second with
`(k₂, a₂, r₂, m₂)`. -/
def fx1Pair {n : ℕ} (k₁ a₁ : BitVec n) (r₁ : ℕ) (m₁ : BitVec n) (k₂ a₂ : BitVec n) (r₂ : ℕ) (m₂ : BitVec n)
    (d : BitVec n × BitVec n) (s : BitVec n × BitVec n) : BitVec n × BitVec n :=
  fx1Word k₂ a₂ r₂ m₂ (fx1Word k₁ a₁ r₁ m₁ s d.1) d.2

/-- Three words `d = (d₁, d₂, d₃)` from the state `s`, with keys `(kᵢ, aᵢ, rᵢ, mᵢ)` for word `i`: the first word,
then the pair. -/
def fx1Triple {n : ℕ} (k₁ a₁ : BitVec n) (r₁ : ℕ) (m₁ : BitVec n) (k₂ a₂ : BitVec n) (r₂ : ℕ) (m₂ : BitVec n)
    (k₃ a₃ : BitVec n) (r₃ : ℕ) (m₃ : BitVec n) (d : BitVec n × BitVec n × BitVec n) (s : BitVec n × BitVec n) :
    BitVec n × BitVec n :=
  fx1Pair k₂ a₂ r₂ m₂ k₃ a₃ r₃ m₃ d.2 (fx1Word k₁ a₁ r₁ m₁ s d.1)

/-- Four words `d = (d₁, d₂, d₃, d₄)` from the state `s`, with keys `(kᵢ, aᵢ, rᵢ, mᵢ)` for word `i`: the first
word, then the triple. -/
def fx1Quad {n : ℕ} (k₁ a₁ : BitVec n) (r₁ : ℕ) (m₁ : BitVec n) (k₂ a₂ : BitVec n) (r₂ : ℕ) (m₂ : BitVec n)
    (k₃ a₃ : BitVec n) (r₃ : ℕ) (m₃ : BitVec n) (k₄ a₄ : BitVec n) (r₄ : ℕ) (m₄ : BitVec n)
    (d : BitVec n × BitVec n × BitVec n × BitVec n) (s : BitVec n × BitVec n) : BitVec n × BitVec n :=
  fx1Triple k₂ a₂ r₂ m₂ k₃ a₃ r₃ m₃ k₄ a₄ r₄ m₄ d.2 (fx1Word k₁ a₁ r₁ m₁ s d.1)

/-- A run of `L` words `w` from the state `s`, word `i` with keys `(k i, a i, r i, m i)`: the first word, then the
rest. -/
def fx1Run {n : ℕ} : {L : ℕ} → (k a : Fin L → BitVec n) → (r : Fin L → ℕ) → (m w : Fin L → BitVec n) →
    BitVec n × BitVec n → BitVec n × BitVec n
  | 0, _, _, _, _, _, s => s
  | _ + 1, k, a, r, m, w, s =>
      fx1Run (Fin.tail k) (Fin.tail a) (Fin.tail r) (Fin.tail m) (Fin.tail w) (fx1Word (k 0) (a 0) (r 0) (m 0) s (w 0))

end Pouw.Receipt
