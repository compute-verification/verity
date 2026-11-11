module

public import Proofs.Flock.Claims.KS.PlainFirst
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
import Mathlib.Topology.Sheaves.Init

/-!
# The third bound of the plain mixture, from a simulation of the runs

`PlainLiveWrPAt`'s third clause bounds the average over a tape of the live session's probability of a self-clash
(`LiveSelfClash`) by the probability of a one-run event of `V false`'s coins. With the deployed coins as the tape and
`plainCoins` as the coins, `plainMix_le_of_run` gives that form for any live event `pL` and deployed event `E` from the
pointwise implication alone: wherever `pL` holds at the live outcome of `P'` at the tape `(ω, u)` and the live coins
`r'`, `E` holds at `plainCoins (u, r')` (`liveMix_le_Pr`, `Pr_mergeAt`).
-/

@[expose] public section

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Tables Claims.Bridge.Live

local notation "𝔽" => FlockLevel3.GF128

open Classical in
/-- **A deployed event bounds a mixture of live strategies at `plainCoins`**: where the live event `pL` at the live
outcome of `P'` at the tape `(ω, u)` and the live coins `r'` gives `E` at `plainCoins (u, r')`, the average over the
deployed coins `u` of the live session's probability of `pL` against `P'`'s strategies is at most `Pr[E]`. -/
theorem plainMix_le_of_run {Ω : Type} (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 mPts)
    (P' : Claims.Prover (Ω × (Fin ((V false).k x) → Coin))) (hk : N x + 2 = (V false).k x)
    (n : Fin ((V false).k x) → ℕ) (pL : LiveOutJ 1 mPts → Prop) (E : (Fin ((V false).k x) → Coin) → Prop)
    (h : ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') →
      E (plainCoins hk n (u, r'))) :
    (∑ u, (prob pL (liveSession dg 1 mPts (N x))
      (liveStrat (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u))).toReal) /
        Fintype.card (Fin ((V false).k x) → Coin) ≤ Claims.Pr E :=
  liveMix_le_Pr (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x ω pL E (plainCoins hk n)
    (@Pr_mergeAt _ _ _ 𝔽 _ _ hk n E) h

end FlockDeploy.KS
