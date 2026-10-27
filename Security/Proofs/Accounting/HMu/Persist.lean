import Definitions.Accounting.Guarantees.HMu

/-!
# Lemma B′ for a persistent pattern, one word a unit

Two word sequences that agree before position `i`, differ at `i` and agree from `i + 1` to the checkpoint `c` give
different states at `c`, for a fold whose step is injective in the word and in the state: the states agree up to `i`,
part at `i + 1` and stay apart. A persistent pattern whose only position before `c` is `i` is such a pair for any two
skipped values.
-/

namespace Accounting.HMuProofs

open Accounting.HMu

variable {S W : Type}

theorem chain_eq_of_agree (step : ℕ → S → W → S) (s₀ : S) (w w' : ℕ → W) (n : ℕ) (h : ∀ k < n, w k = w' k) :
    chain step s₀ w n = chain step s₀ w' n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [chain]
    rw [ih fun k hk => h k (by omega), h n (by omega)]

theorem chain_ne_of_first_diff (step : ℕ → S → W → S) (hs : ∀ n x, Function.Injective (fun s => step n s x))
    (hw : ∀ n s, Function.Injective (step n s)) (s₀ : S) (w w' : ℕ → W) (i : ℕ) (hb : ∀ k < i, w k = w' k)
    (hi : w i ≠ w' i) (k : ℕ) (ha : ∀ n, i < n → n < i + 1 + k → w n = w' n) :
    chain step s₀ w (i + 1 + k) ≠ chain step s₀ w' (i + 1 + k) := by
  induction k with
  | zero =>
    simp only [chain]
    rw [chain_eq_of_agree step s₀ w w' i hb]
    exact fun h => hi (hw i _ h)
  | succ k ih =>
    rw [show i + 1 + (k + 1) = i + 1 + k + 1 by omega]
    simp only [chain]
    rw [ha (i + 1 + k) (by omega) (by omega)]
    exact fun h => ih (fun n h1 h2 => ha n h1 (by omega)) (hs _ _ h)

theorem persistentOneWord : Accounting.Guarantees.HMu.PersistentOneWord := by
  intro S W _ step s₀ w P i c hs hw hi hic hP X X' h
  by_contra hne
  obtain ⟨k, rfl⟩ : ∃ k, c = i + 1 + k := ⟨c - (i + 1), by omega⟩
  have off : ∀ n, n ≠ i → n < i + 1 + k → ∀ Y, persist w P Y n = w n := by
    intro n hn hlt Y
    have : n ∉ P := fun hm => absurd (hP n hm hn) (by omega)
    simp [persist, this]
  refine chain_ne_of_first_diff step hs hw s₀ (persist w P X) (persist w P X') i
    (fun n hn => by rw [off n (by omega) (by omega), off n (by omega) (by omega)]) ?_ k
    (fun n h1 h2 => by rw [off n (by omega) h2, off n (by omega) h2]) h
  simp only [persist, hi, ite_true]
  exact fun h => hne (add_left_cancel h)

end Accounting.HMuProofs
