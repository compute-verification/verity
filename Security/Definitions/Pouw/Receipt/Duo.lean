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

`duoRun` absorbs a run of any length one word at a time, `duoQuadWith` absorbs four words with keys bundled as
`DuoQuadKeys`, and `duoCheckpoints` absorbs a run of four-word blocks and returns the state after each block.
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

/-- Four consecutive words' keys `(kᵢ, aᵢ, rᵢ)`. -/
structure DuoQuadKeys (n : ℕ) where
  k₁ : BitVec n
  a₁ : BitVec n
  r₁ : ℕ
  k₂ : BitVec n
  a₂ : BitVec n
  r₂ : ℕ
  k₃ : BitVec n
  a₃ : BitVec n
  r₃ : ℕ
  k₄ : BitVec n
  a₄ : BitVec n
  r₄ : ℕ

/-- Four words' keys that meet `duoQuad`'s binding conditions: odd first, second and third multipliers, a first
rotation `0 < r₁ ≤ n − 3`, `a₂ ≢ −1 (mod 2^r₁)`, and second and third rotations that move every bit. -/
def DuoQuadKeysOk (n : ℕ) : Type :=
  {q : DuoQuadKeys n // q.a₁.toNat % 2 = 1 ∧ 0 < q.r₁ ∧ q.r₁ + 3 ≤ n ∧ q.a₂.toNat % 2 = 1 ∧
    (q.a₂.toNat + 1) % 2 ^ q.r₁ ≠ 0 ∧ (0 < q.r₂ ∧ q.r₂ < n) ∧ q.a₃.toNat % 2 = 1 ∧ (0 < q.r₃ ∧ q.r₃ < n)}

/-- Four words `d` with the keys `q` from the state `s`. -/
def duoQuadWith {n : ℕ} (q : DuoQuadKeys n) (d : BitVec n × BitVec n × BitVec n × BitVec n)
    (s : BitVec n × BitVec n) : BitVec n × BitVec n :=
  duoQuad q.k₁ q.a₁ q.r₁ q.k₂ q.a₂ q.r₂ q.k₃ q.a₃ q.r₃ q.k₄ q.a₄ q.r₄ d s

/-- A run of `m` blocks of four words from the state `s`, checkpointed after every block: the state after block `j`,
for each `j`. Block `j` absorbs the words `w j` with the keys `q j`. -/
def duoCheckpoints {n : ℕ} : {m : ℕ} → (q : Fin m → DuoQuadKeys n) →
    (w : Fin m → BitVec n × BitVec n × BitVec n × BitVec n) → BitVec n × BitVec n → Fin m → BitVec n × BitVec n
  | 0, _, _, _ => Fin.elim0
  | _ + 1, q, w, s =>
      Fin.cons (duoQuadWith (q 0) (w 0) s) (duoCheckpoints (Fin.tail q) (Fin.tail w) (duoQuadWith (q 0) (w 0) s))

/-- A run of `L` words `w` from the state `s`, word `i` with keys `(k i, a i, r i)`: the first word, then the rest
(`duoPair`, `duoTriple` and `duoQuad` at `L = 2, 3, 4`). -/
def duoRun {n : ℕ} : {L : ℕ} → (k a : Fin L → BitVec n) → (r : Fin L → ℕ) → (w : Fin L → BitVec n) →
    BitVec n × BitVec n → BitVec n × BitVec n
  | 0, _, _, _, _, s => s
  | _ + 1, k, a, r, w, s =>
      duoRun (Fin.tail k) (Fin.tail a) (Fin.tail r) (Fin.tail w) (duoWord (k 0) (a 0) (r 0) s (w 0))

end Pouw.Receipt
