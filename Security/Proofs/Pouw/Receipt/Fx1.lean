import Definitions.Pouw.Receipt.Fx1
import Proofs.Pouw.Receipt.Run

/-!
# fx1's words are bound wherever duo's are

fx1 is duo on the words times odd multipliers: `fx1Word k a r m s = duoWord k a r s ∘ (· * m)`, and so on for pairs,
triples, quadruples and runs (`fx1Run_eq`). Multiplying by an odd `m` is injective (`mul_odd_cancel`), so each of
duo's injectivity statements composes with it into fx1's, and the state bijection is duo's at the word `d·m`.
-/

namespace Pouw.ReceiptProofs

open Pouw.Receipt

theorem mul_odd_injective {n : ℕ} (m : OddKey n) : Function.Injective (fun d : BitVec n => d * m.1) :=
  fun _ _ h => mul_odd_cancel m.2 h

theorem fx1Word_injective {n : ℕ} (k : BitVec n) (a : OddKey n) (r : ℕ) (m : OddKey n) (s : BitVec n × BitVec n) :
    Function.Injective (fx1Word k a.1 r m.1 s) :=
  (duoWord_injective k a r s).comp (mul_odd_injective m)

theorem fx1Word_state_bijective {n : ℕ} (k a : BitVec n) (r : ℕ) (m d : BitVec n) :
    Function.Bijective (fun s => fx1Word k a r m s d) :=
  duoWord_state_bijective k a r (d * m)

theorem fx1Pair_injective {n : ℕ} (k₁ : BitVec n) (a₁ : OddKey n) (r₁ : Rotation n) (m₁ : OddKey n) (k₂ a₂ : BitVec n)
    (r₂ : ℕ) (m₂ : OddKey n) : Function.Injective (fx1Pair k₁ a₁.1 r₁.1 m₁.1 k₂ a₂ r₂ m₂.1) := by
  intro d e h
  have h' := duoPair_injective k₁ a₁ r₁ k₂ a₂ r₂ (show duoPair k₁ a₁.1 r₁.1 k₂ a₂ r₂ (d.1 * m₁.1, d.2 * m₂.1) =
    duoPair k₁ a₁.1 r₁.1 k₂ a₂ r₂ (e.1 * m₁.1, e.2 * m₂.1) from h)
  simp only [Prod.mk.injEq] at h'
  exact Prod.ext (mul_odd_injective m₁ h'.1) (mul_odd_injective m₂ h'.2)

theorem fx1Triple_injective {n : ℕ} (k₁ : BitVec n) (a₁ : OddKey n) (r₁ : Rotation n) (m₁ : OddKey n) (k₂ : BitVec n)
    (a₂ : OddKey n) (r₂ : Rotation n) (m₂ : OddKey n) (k₃ a₃ : BitVec n) (r₃ : ℕ) (m₃ : OddKey n) :
    Function.Injective (fx1Triple k₁ a₁.1 r₁.1 m₁.1 k₂ a₂.1 r₂.1 m₂.1 k₃ a₃ r₃ m₃.1) := by
  intro d e h
  have h' := duoTriple_injective k₁ a₁ r₁ k₂ a₂ r₂ k₃ a₃ r₃
    (show duoTriple k₁ a₁.1 r₁.1 k₂ a₂.1 r₂.1 k₃ a₃ r₃ (d.1 * m₁.1, d.2.1 * m₂.1, d.2.2 * m₃.1) =
      duoTriple k₁ a₁.1 r₁.1 k₂ a₂.1 r₂.1 k₃ a₃ r₃ (e.1 * m₁.1, e.2.1 * m₂.1, e.2.2 * m₃.1) from h)
  simp only [Prod.mk.injEq] at h'
  exact Prod.ext (mul_odd_injective m₁ h'.1)
    (Prod.ext (mul_odd_injective m₂ h'.2.1) (mul_odd_injective m₃ h'.2.2))

theorem fx1Quad_injective {n : ℕ} (k₁ : BitVec n) (a₁ : OddKey n) (r₁ : ShortRotation n) (m₁ : OddKey n)
    (k₂ : BitVec n) (a₂ : OddKeyNotNegOne n r₁.1) (r₂ : Rotation n) (m₂ : OddKey n) (k₃ : BitVec n) (a₃ : OddKey n)
    (r₃ : Rotation n) (m₃ : OddKey n) (k₄ a₄ : BitVec n) (r₄ : ℕ) (m₄ : OddKey n) :
    Function.Injective (fx1Quad k₁ a₁.1 r₁.1 m₁.1 k₂ a₂.1 r₂.1 m₂.1 k₃ a₃.1 r₃.1 m₃.1 k₄ a₄ r₄ m₄.1) := by
  intro d e h
  have h' := duoQuad_injective k₁ a₁ r₁.1 r₁.2.1 r₁.2.2 k₂ ⟨a₂.1, a₂.2.1⟩ a₂.2.2 r₂ k₃ a₃ r₃ k₄ a₄ r₄
    (show duoQuad k₁ a₁.1 r₁.1 k₂ a₂.1 r₂.1 k₃ a₃.1 r₃.1 k₄ a₄ r₄
        (d.1 * m₁.1, d.2.1 * m₂.1, d.2.2.1 * m₃.1, d.2.2.2 * m₄.1) =
      duoQuad k₁ a₁.1 r₁.1 k₂ a₂.1 r₂.1 k₃ a₃.1 r₃.1 k₄ a₄ r₄
        (e.1 * m₁.1, e.2.1 * m₂.1, e.2.2.1 * m₃.1, e.2.2.2 * m₄.1) from h)
  simp only [Prod.mk.injEq] at h'
  exact Prod.ext (mul_odd_injective m₁ h'.1) (Prod.ext (mul_odd_injective m₂ h'.2.1)
    (Prod.ext (mul_odd_injective m₃ h'.2.2.1) (mul_odd_injective m₄ h'.2.2.2)))

/-- An fx1 run is duo's run on the words times their multipliers. -/
theorem fx1Run_eq {n : ℕ} :
    ∀ {L : ℕ} (k a : Fin L → BitVec n) (r : Fin L → ℕ) (m w : Fin L → BitVec n) (s : BitVec n × BitVec n),
      fx1Run k a r m w s = duoRun k a r (fun j => w j * m j) s
  | 0, _, _, _, _, _, _ => rfl
  | _ + 1, k, a, r, m, w, s => by
    simp only [fx1Run, duoRun]
    exact fx1Run_eq _ _ _ _ _ _

theorem fx1Run_update_injective {n L : ℕ} (k : Fin L → BitVec n) (a : Fin L → OddKey n) (r : Fin L → ℕ)
    (m : Fin L → OddKey n) (s : BitVec n × BitVec n) (w : Fin L → BitVec n) (i : Fin L) :
    Function.Injective (fun x => fx1Run k (fun j => (a j).1) r (fun j => (m j).1) (Function.update w i x) s) := by
  have e : ∀ x, (fun j => Function.update w i x j * (m j).1) = Function.update (fun j => w j * (m j).1) i (x * (m i).1) :=
    fun x => funext fun j => by
      by_cases hj : j = i
      · subst hj; simp only [Function.update_self]
      · simp only [Function.update_of_ne hj]
  intro x y h
  simp only [fx1Run_eq, e] at h
  exact mul_odd_injective (m i) (duoRun_update_injective k a r s (fun j => w j * (m j).1) i h)

end Pouw.ReceiptProofs
