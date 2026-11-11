module

public import Definitions.Claims.Recursion
public import Proofs.Flock.Claims.Recursion.Cost
public import Proofs.Flock.Claims.Recursion.Transport
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.RingTheory.WittVector.IsPoly
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init

/-!
# The recursion's extraction-based finder

`Be` (`Claims.Recursion.Be`) is `forkExt` at the outer extractor `E_O`, its pair the first collision among the calls
the two outer witnesses record (`Be_eq_forkExt`), so under `ExpectedCR` it finds one with probability at most two runs'
calls and two extractions' over `2^256` (`PrBe_le`).
-/

@[expose] public section

namespace Claims.Recursion

open Claims

section Finders

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- `Be`'s pair: the first collision among the calls the two outer witnesses record. -/
def bePick (D : Deploy Coin) (x : Msg) (_ : Transcript) (w₁ : Option (List Flock.Wit)) (_ : Transcript)
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
