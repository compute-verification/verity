import Proofs.Flock.Claims.Recursion.Cost
import Proofs.Flock.Claims.Recursion.Transport

/-!
# The recursion's extraction-based finder

`Be` (`Claims.Recursion.Be`) is `forkExt` at the outer extractor `E_O`, its pair the first collision among the calls
the two outer witnesses record (`Be_eq_forkExt`), so under `ExpectedCR` it finds one with probability at most two runs'
calls and two extractions' over `2^256` (`PrBe_le`).
-/

namespace Claims.Recursion

open Claims

section Finders

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- `Be`'s pair: the first collision among the calls the two outer witnesses record. -/
def bePick (D : Deploy Coin) (x : Msg) (_ : Trace) (w₁ : Option (List Flock.Wit)) (_ : Trace)
    (w₂ : Option (List Flock.Wit)) : OC (Option (Msg × Msg)) :=
  .ret (findColl (D.calls x (w₁.getD []) ++ D.calls x (w₂.getD [])))

theorem Be_eq_forkExt (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (hk : 0 < D.V.k x) : Be H D I DO decO P x hk = forkExt H D.V (EO D I DO decO) (bePick D) P x hk :=
  rfl

/-- **`Be` under `ExpectedCR`**: with every run within `T` calls (both parties') and `E_O` within `N_h`, `Be` finds a
collision with probability at most `(2T + 2N_h)/2^256`. -/
theorem PrBe_le (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) {T Nh : ℕ}
    (hT : ∀ ω r, (run H P D.V x ω r).pLog.length + (run H P D.V x ω r).vLog.length ≤ T)
    (hE : ∀ ω r₀ e, ((EO D I DO decO).eval H P x ω r₀ e).2.1 ≤ Nh)
    (hCR : ∀ hk : 0 < D.V.k x, ExpectedCR H (Be H D I DO decO P x hk)) :
    PrBe H D I DO decO P x ≤ ((2 * T + 2 * Nh : ℕ) : ℝ) / 2 ^ (256 : ℝ) := by
  unfold PrBe
  split
  · rename_i hk
    have h := forkExt_Pr_le H D.V (EO D I DO decO) (bePick D) P x hk (c := 0) hT hE
      (fun _ _ _ _ => by simp [bePick, OC.eval]) (hCR hk)
    simpa only [Be_eq_forkExt, add_zero] using h
  · positivity

end Finders

end Claims.Recursion
