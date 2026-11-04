import PouwPearlPlus.Rule
import Proofs.Pouw.PearlC.CoreOps
import Proofs.Pouw.PearlC.TileValues

/-!
# Pearl+'s bridge, part (c), the debit's keys: `key` is the value order, `least` a threshold, `carried` the rounded add

`Rule.wordDebit` replays a word's later promotions on keys rather than on values: `Rule.key` writes a finite FP32 word as
an integer in [−TOP, TOP] in value order (±0 one key), `Rule.least` bisects for the least key at which a monotone predicate
holds, and `Rule.carried x w` is the key of RN(x's value + w's value). `keyVal` is a key's value; the three statements
here are what the debit's replay (`Debit`) reads of them.
-/

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus

/-- The value of a magnitude key `u`, a finite word's low 31 bits: `wordQ`'s field formula at exponent field `u / 2^23`. -/
def magVal (u : ℕ) : ℚ :=
  (((if u / 2 ^ 23 = 0 then 0 else 2 ^ 23) + u % 2 ^ 23 : ℕ) : ℚ) * (2 : ℚ) ^ (((max (u / 2 ^ 23) 1 : ℕ) : ℤ) - 150)

/-- The value of a key. -/
def keyVal (x : ℤ) : ℚ := if x < 0 then -magVal x.natAbs else magVal x.natAbs

/-- **The key is the value's order**: `keyVal` is strictly increasing; a finite word's key is in [−TOP, TOP] and its value
is the key's; and every key in [−TOP, TOP] is a finite word's, `Rule.wordOf`'s. -/
def KeyIsTheValue : Prop :=
  StrictMono keyVal ∧
    (∀ w : ℕ, w < 2 ^ 32 → (w >>> 23) % 256 ≠ 255 →
      wordQ w = some (keyVal (Rule.key w)) ∧ -Rule.TOP ≤ Rule.key w ∧ Rule.key w ≤ Rule.TOP) ∧
    ∀ x : ℤ, -Rule.TOP ≤ x → x ≤ Rule.TOP →
      Rule.wordOf x < 2 ^ 32 ∧ (Rule.wordOf x >>> 23) % 256 ≠ 255 ∧ Rule.key (Rule.wordOf x) = x

/-- **`least` is the threshold**: for a predicate monotone on [−TOP, TOP], `Rule.least` is in [−TOP, TOP + 1], and a key
in [−TOP, TOP] is at least it exactly when the predicate holds there. -/
def LeastIsTheThreshold : Prop :=
  ∀ pred : ℤ → Bool, (∀ x y, -Rule.TOP ≤ x → x ≤ y → y ≤ Rule.TOP → pred x = true → pred y = true) →
    -Rule.TOP ≤ Rule.least pred ∧ Rule.least pred ≤ Rule.TOP + 1 ∧
      ∀ x, -Rule.TOP ≤ x → x ≤ Rule.TOP → (Rule.least pred ≤ x ↔ pred x = true)

/-- **`carried` is the rounded add**: where the check's FP32 add is the semantics' on finite words, a key in [−TOP, TOP]
carried through a finite word of magnitude below 2⁴⁶ (a promotion group's word) stays in [−TOP, TOP], and its value is
`rnAdd` of the two values. -/
def CarriedIsTheAdd : Prop :=
  (∀ a b : ℕ, (wordQ a).isSome → (wordQ b).isSome → Evaluate.Fp32.f32AddV2 a b = TileCircuit.f32Add a b) →
    ∀ (x : ℤ) (w : ℕ) (y : ℚ), -Rule.TOP ≤ x → x ≤ Rule.TOP → wordQ w = some y → |y| < 2 ^ 46 →
      -Rule.TOP ≤ Rule.carried x w ∧ Rule.carried x w ≤ Rule.TOP ∧ keyVal (Rule.carried x w) = rnAdd (keyVal x) y

end Pouw.PearlPlus.Bridge
