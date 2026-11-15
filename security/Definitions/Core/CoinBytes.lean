module

public import Mathlib.Data.Fintype.BigOperators
public import Mathlib.Data.Fintype.EquivFin
public import Mathlib.Data.Fintype.Pi
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# The bytes a sample reads and the coins' bits, one for one

The challenger releases each coin as `8 * B` bits, and a sample reads the coins it was released as bytes: bit `t` of a
coin's bits is bit `t mod 8` of its byte `⌊t/8⌋`. Over the bits of `k` coins, `Fin k → Fin (8 * B) → Bool`, the bytes
are one stream of `k * B` bytes, coin `c`'s byte `q` at `c * B + q` (`byteIdx`). `bitsAt` reads the coins' bits off the
bytes, and `coinBytesEquiv` is that reading as a bijection: it is injective (`bitsAt_injective`, by `bitsAt_spec`)
between two types of one cardinality (`card_coins`).

The lemmas here are the ones `byteIdx` and `coinBytesEquiv` are built from. That the bytes of uniform coin bits are
uniform bytes is proved in `Proofs.Properties.Verity.SampledCoins`.
-/

@[expose] public section

namespace Verity.Sampled

variable {k B : ℕ}

theorem idx_lt (c : Fin k) {q : ℕ} (hq : q < B) : c.val * B + q < k * B :=
  calc c.val * B + q < (c.val + 1) * B := by rw [Nat.add_mul, Nat.one_mul]; omega
    _ ≤ k * B := Nat.mul_le_mul_right _ c.isLt

/-- The byte bit `t` of coin `c` lies in: `c * B + ⌊t/8⌋`. -/
def byteIdx (c : Fin k) (t : Fin (8 * B)) : Fin (k * B) :=
  ⟨c.val * B + t.val / 8, idx_lt c (Nat.div_lt_of_lt_mul t.isLt)⟩

variable (k B) in
/-- **The coins' bits in the bytes `F`**: bit `t` of coin `c` is bit `t mod 8` of its byte `byteIdx c t`. -/
def bitsAt (F : Fin 1 → Fin (k * B) → Fin 256) (c : Fin k) (t : Fin (8 * B)) : Bool :=
  (F 0 (byteIdx c t)).val.testBit (t.val % 8)

theorem bitsAt_spec (F : Fin 1 → Fin (k * B) → Fin 256) (c : Fin k) (p : Fin (k * B)) {q j : ℕ}
    (hp : p.val = c.val * B + q) (hq : q < B) (hj : j < 8) :
    bitsAt k B F c ⟨8 * q + j, by omega⟩ = (F 0 p).val.testBit j := by
  have e : byteIdx c (⟨8 * q + j, by omega⟩ : Fin (8 * B)) = p :=
    Fin.ext (by show c.val * B + (8 * q + j) / 8 = p.val; omega)
  show (F 0 (byteIdx c ⟨8 * q + j, _⟩)).val.testBit ((8 * q + j) % 8) = _
  rw [e, show (8 * q + j) % 8 = j by omega]

variable (k B) in
theorem bitsAt_injective : Function.Injective (bitsAt k B) := by
  intro f g h
  funext i p
  obtain rfl : i = 0 := Subsingleton.elim _ _
  have hB : 0 < B := Nat.pos_of_ne_zero fun h0 => by subst h0; exact absurd p.isLt (by simp)
  have hc : p.val / B < k := Nat.div_lt_of_lt_mul (lt_of_lt_of_eq p.isLt (Nat.mul_comm k B))
  have hp : p.val = (⟨p.val / B, hc⟩ : Fin k).val * B + p.val % B := (Nat.div_add_mod' p.val B).symm
  have hq : p.val % B < B := Nat.mod_lt _ hB
  refine Fin.ext (Nat.eq_of_testBit_eq fun j => ?_)
  rcases Nat.lt_or_ge j 8 with hj | hj
  · rw [← bitsAt_spec f ⟨p.val / B, hc⟩ p hp hq hj, ← bitsAt_spec g ⟨p.val / B, hc⟩ p hp hq hj, h]
  · have h256 : ∀ x : Fin 256, x.val.testBit j = false := fun x =>
      Nat.testBit_lt_two_pow (lt_of_lt_of_le x.isLt (by simpa using Nat.pow_le_pow_right (by norm_num : 0 < 2) hj))
    rw [h256, h256]

variable (k B) in
theorem card_coins : Fintype.card (Fin 1 → Fin (k * B) → Fin 256) = Fintype.card (Fin k → Fin (8 * B) → Bool) := by
  simp only [Fintype.card_fun, Fintype.card_fin, Fintype.card_bool, pow_one]
  rw [show (256 : ℕ) = 2 ^ 8 by norm_num, ← pow_mul, ← pow_mul]
  congr 1
  ring

variable (k B) in
/-- The bytes and the coins' bits, one for one (`bitsAt`). -/
noncomputable def coinBytesEquiv : (Fin 1 → Fin (k * B) → Fin 256) ≃ (Fin k → Fin (8 * B) → Bool) :=
  Equiv.ofBijective (bitsAt k B) ((Fintype.bijective_iff_injective_and_card _).2 ⟨bitsAt_injective k B, card_coins k B⟩)

end Verity.Sampled
