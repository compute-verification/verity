import Mathlib
import Definitions.Accounting.Guarantees.Bandwidth

/-!
# The honest device: the guarantee is not vacuous

A device with no on-chip state, no side sources and no regeneration, whose device memory holds the store as it is,
reads every block once and queries the oracle at it. It is accepted on every store, oracle and sample, and its
ingress is exactly the store, `N·b` bits (`honest_accepts`; the guarantee `Bandwidth.SecurityProofs.HonestPasses`).
-/

namespace Bandwidth.Completeness

open Pous (Bits)
open scoped ENNReal

variable {N b h : ℕ} {α : Type}

/-- Device memory holding the store as it is: bit `i` of block `j` at address `j·b + i`. -/
def layout (E : Payloads N b) (a : ℕ) : Bool :=
  if ha : a < N * b then
    E ⟨a / b, Nat.div_lt_of_lt_mul (by rwa [mul_comm] at ha)⟩
      ⟨a % b, Nat.mod_lt _ (Nat.pos_of_ne_zero (by rintro rfl; simp at ha))⟩
  else false

theorem layout_block (E : Payloads N b) (j : Fin N) (i : Fin b) : layout E (j * b + i) = E j i := by
  have hb : 0 < b := Nat.lt_of_le_of_lt (Nat.zero_le _) i.2
  have hlt : (j : ℕ) * b + i < N * b := by
    have := j.2; have := i.2
    calc (j : ℕ) * b + i < j * b + b := by omega
      _ = (j + 1) * b := by ring
      _ ≤ N * b := Nat.mul_le_mul_right _ (by omega)
  have hdiv : ((j : ℕ) * b + i) / b = j := by
    rw [add_comm, Nat.add_mul_div_right _ _ hb, Nat.div_eq_of_lt i.2, zero_add]
  have hmod : ((j : ℕ) * b + i) % b = i := by
    rw [add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt i.2]
  simp only [layout, hlt, ↓reduceDIte]
  congr 1
  · exact Fin.ext hdiv
  · exact Fin.ext hmod

/-- Read `n` bits of device memory from address `a`, then continue with them. -/
def readBits (a : ℕ) : (n : ℕ) → (Bits n → Chip N b h α) → Chip N b h α
  | 0, k => k Fin.elim0
  | n + 1, k => .ingress a (fun x => readBits (a + 1) n (fun v => k (Fin.cons x v)))

theorem readBits_eq (w : World N b h) (a n : ℕ) (k : Bits n → Chip N b h α) :
    (readBits a n k).run w = (k (fun i => w.mem (a + i))).run w ∧
    (readBits a n k).ingressBits w = (k (fun i => w.mem (a + i))).ingressBits w + n ∧
    (readBits a n k).sideBits w = (k (fun i => w.mem (a + i))).sideBits w ∧
    (readBits a n k).regens w = (k (fun i => w.mem (a + i))).regens w ∧
    (readBits a n k).queries w = (k (fun i => w.mem (a + i))).queries w := by
  induction n generalizing a with
  | zero =>
    have : (fun i : Fin 0 => w.mem (a + i)) = Fin.elim0 := funext (fun i => i.elim0)
    simp [readBits, this]
  | succ n ih =>
    have hc : (Fin.cons (w.mem a) (fun i : Fin n => w.mem (a + 1 + i)) : Bits (n + 1)) =
        fun i : Fin (n + 1) => w.mem (a + i) := by
      funext i
      refine Fin.cases ?_ (fun i => ?_) i
      · simp
      · simp only [Fin.cons_succ, Fin.val_succ]
        congr 1
        omega
    obtain ⟨h1, h2, h3, h4, h5⟩ := ih (a + 1) (fun v => k (Fin.cons (w.mem a) v))
    simp only [readBits, Chip.run, Chip.ingressBits, Chip.sideBits, Chip.regens, Chip.queries]
    rw [h1, h2, h3, h4, h5, hc]
    exact ⟨rfl, by omega, rfl, rfl, rfl⟩

/-- Read each listed block and query the oracle at it, recording the tag. -/
def loop : List (Fin N) → (Fin N → Bits h) → Chip N b h (Fin N → Bits h)
  | [], acc => .done acc
  | j :: js, acc => readBits (j * b) b (fun x => .tag (j, x) (fun y => loop js (Function.update acc j y)))

/-- Block `j` as device memory holds it at `j·b`. -/
def blockAt (w : World N b h) (j : Fin N) : Bits b := fun i => w.mem (j * b + i)

theorem loop_eq (w : World N b h) (js : List (Fin N)) (acc : Fin N → Bits h) :
    (loop js acc).run w = (fun j => if j ∈ js then w.oracle (j, blockAt w j) else acc j) ∧
    (loop js acc).ingressBits w = js.length * b ∧
    (loop js acc).sideBits w = 0 ∧
    (loop js acc).regens w = 0 ∧
    (loop js acc).queries w = js.map (fun j => (j, blockAt w j)) := by
  induction js generalizing acc with
  | nil => simp [loop, Chip.run, Chip.ingressBits, Chip.sideBits, Chip.regens, Chip.queries]
  | cons j js ih =>
    obtain ⟨h1, h2, h3, h4, h5⟩ := readBits_eq w (j * b) b
      (fun x => Chip.tag (j, x) (fun y => loop js (Function.update acc j y)))
    obtain ⟨i1, i2, i3, i4, i5⟩ := ih (Function.update acc j (w.oracle (j, blockAt w j)))
    simp only [loop]
    rw [h1, h2, h3, h4, h5]
    simp only [Chip.run, Chip.ingressBits, Chip.sideBits, Chip.regens, Chip.queries]
    refine ⟨?_, ?_, i3, i4, ?_⟩
    · change (loop js (Function.update acc j (w.oracle (j, blockAt w j)))).run w = _
      rw [i1]
      funext j'
      by_cases hj : j' = j
      · subst hj; simp
      · by_cases hm : j' ∈ js <;> simp [hm, hj]
    · change (loop js (Function.update acc j (w.oracle (j, blockAt w j)))).ingressBits w + b = _
      rw [i2, List.length_cons]
      ring
    · change (j, blockAt w j) :: (loop js (Function.update acc j (w.oracle (j, blockAt w j)))).queries w = _
      rw [i5, List.map_cons]

/-- The honest device: no on-chip state, no side sources, no regeneration; device memory holds the store as it is,
and the chip reads every block once and queries the oracle at it. -/
def honest (N b h : ℕ) : Device N b h 0 0 0 N where
  state _ := Fin.elim0
  mem := layout
  side _ _ := false
  chip _ := loop (List.finRange N) (fun _ _ => false)
  bounded st w := by
    obtain ⟨-, -, h3, h4, h5⟩ := loop_eq w (List.finRange N) (fun _ _ => false)
    simp [h3, h4, h5]

theorem honest_tags (E : Payloads N b) (O : Oracle N b h) (j : Fin N) :
    (honest N b h).tags E O j = O (j, E j) := by
  have := congrFun (loop_eq ((honest N b h).world E O) (List.finRange N) (fun _ _ => false)).1 j
  simp only [List.mem_finRange, ↓reduceIte] at this
  refine this.trans (congrArg _ (Prod.ext rfl (funext (fun i => ?_))))
  exact layout_block E j i

/-- **Completeness.** The honest device is accepted on every store, oracle and sample, and reads exactly the store. -/
theorem honest_accepts {s : ℕ} (E : Payloads N b) (O : Oracle N b h) (I : Fin s → Fin N) :
    Accepts (honest N b h) E O I ∧ (honest N b h).ingress E O = N * b := by
  refine ⟨fun i => honest_tags E O (I i), ?_⟩
  have := (loop_eq ((honest N b h).world E O) (List.finRange N) (fun _ _ => false)).2.1
  simpa [Device.ingress, honest] using this

end Bandwidth.Completeness

theorem Bandwidth.SecurityProofs.HonestPasses : Bandwidth.Guarantees.HonestPasses :=
  fun N b h => ⟨Bandwidth.Completeness.honest N b h, fun E O _ I => Bandwidth.Completeness.honest_accepts E O I⟩
