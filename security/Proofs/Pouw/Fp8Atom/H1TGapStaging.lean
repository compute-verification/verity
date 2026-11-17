module

public import Definitions.Pouw.Fp8Atom.H1TLemmas
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.Fp8Atom.Fp32Proofs

/-! The statements `H1TRunningWords` still needs (staging): none is proved or pinned, and no pin reads them. -/

@[expose] public section

namespace Pouw.Fp8Atom.H1T

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs

def chainFrom (a : ℚ) (cs : List ℚ) : ℚ := cs.foldl rnAdd a

def RnMonoChainGap : Prop :=
  ∀ (a b : ℚ) (cs : List ℚ), Representable a → Representable b → 0 ≤ b → b + 2304 ≤ a →
    (∀ c ∈ cs, 1792 ≤ c ∧ c ≤ 738304) → (∀ n ≤ cs.length, chainFrom a (cs.take n) < 2 ^ 33) →
    chainFrom b cs ≠ chainFrom a cs

/-- The bound `2^33 − 2^29` covers every running word of a legal unit (`run_bound`), and stands in place of
`2^33`. -/
def H1TLastSliceSplit : Prop :=
  ∀ (xs : Fin 29 → Code) (b b' : ℕ → Code) (j j' τ : ℕ) (A A' : ℚ), (∀ l, (xs l).isNaN = false) →
    (∀ r, (b r).isNaN = false) → (∀ r, (b' r).isNaN = false) → tuple j ≠ tuple j' →
    Representable A → Representable A' → |A| < 2 ^ 33 - 2 ^ 29 → |A'| < 2 ^ 33 - 2 ^ 29 →
    ∃ σ : SliceSalt, rnAdd A (wordVal (hopperStep 0 (formSlice xs σ).1 (colLanes b j τ))) ≠
      rnAdd A' (wordVal (hopperStep 0 (formSlice xs σ).1 (colLanes b' j' τ)))

def H1TCrossLate : Prop :=
  ∀ U : Unit, U.Legal → ∀ (i : Fin U.m) (a a' : Fin U.n × Pos U.T), a.1 ≠ a'.1 → Run2 a.2 →
    a'.2.isRight = true → U.T ≤ posIdx a'.2 + 1 → ¬ OppLater a.1.val a'.1.val a.2 a'.2 →
    U.word (i, a) ≠ U.word (i, a')

theorem lt_rnAdd {r c : ℚ} (hr : Representable r) (h0 : r ≠ 0) (hc : 0 ≤ c) (hu : ulp r ≤ c) : r < rnAdd r c := by
  have hle : r ≤ rnAdd r c := by
    have := rnAdd_mono r 0 r c (by linarith)
    rwa [rnAdd_exact r 0 (by simpa using hr), add_zero] at this
  exact lt_of_le_of_ne hle (Ne.symm (rnAdd_moves r c hr h0 (by rwa [abs_of_nonneg hc])))

theorem merge_one_ulp :
    Representable ((2 : ℚ) ^ 32 - 1024) ∧ Representable ((2 : ℚ) ^ 32 - 512) ∧
      rnAdd ((2 : ℚ) ^ 32 - 1024) 1792 = (2 : ℚ) ^ 32 + 1024 ∧
      rnAdd ((2 : ℚ) ^ 32 - 512) 1792 = (2 : ℚ) ^ 32 + 1024 := by
  refine ⟨⟨2 ^ 22 - 1, 10, by norm_num, by norm_num⟩, ⟨2 ^ 23 - 1, 9, by norm_num, by norm_num⟩, ?_, ?_⟩ <;>
    decide +kernel

def zeroSlice : Fin 29 → Code := fun _ => ⟨0, by norm_num⟩

def zeroCol : ℕ → Code := fun _ => ⟨0, by norm_num⟩

def absorbSalt : SliceSalt := ⟨fun _ => false, fun _ => false, pick3 0 0 1⟩

theorem absorb_words :
    wordVal (hopperStep 0 (formSlice zeroSlice absorbSalt).1 (colLanes zeroCol 21 1128)) = 3872 ∧
      wordVal (hopperStep 0 (formSlice zeroSlice absorbSalt).1 (colLanes zeroCol 22 1128)) = 4128 := by
  constructor <;> decide +kernel

theorem absorb_rounds :
    rnAdd (rnAdd (2 ^ 32 + 2 ^ 31) 3872) 3872 = 2 ^ 32 + 2 ^ 31 + 8192 ∧
      rnAdd (rnAdd (2 ^ 32 + 2 ^ 31 + 512) 3872) 3872 = 2 ^ 32 + 2 ^ 31 + 512 + 8192 ∧
      rnAdd (rnAdd (2 ^ 32 + 2 ^ 31) 4128) 4128 = 2 ^ 32 + 2 ^ 31 + 8192 ∧
      rnAdd (rnAdd (2 ^ 32 + 2 ^ 31 + 512) 4128) 4128 = 2 ^ 32 + 2 ^ 31 + 512 + 8192 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> decide +kernel

end Pouw.Fp8Atom.H1T
