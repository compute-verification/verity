module

public import Definitions.Pouw.TS.ProofLayer
public import Definitions.Flock.Game
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
import Proofs.Flock.Soundness.Game
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# The bridge's probability steps, over any family of sessions

The proof layer's soundness (`ResourceAccounting.TS.ProofLayer.SoundAt`) is a probability over a window's paths (uniform on
`Paths Ω S`, `Pouw.RelativeSecurity.prob`), per round `w` and proof `i`; Flock's bounds are probabilities over one
session's audit game played by its prover (`FlockSoundness.Game.prob`). This file joins the two for any family of
sessions, with no Flock definition:

* `game_prob_exists_le`: the union bound for a fixed strategy over a finite family of events;
* `SessionSamplingDesign`: read through the session a path runs and its outcome there, the window's coins have the sessions' sampling design;
* `soundAt_of_sessions`: `SoundAt` at `δo + δ` when each unsound proof is off (probability at most `δo`) or its
  session's outcome is in a bad event of probability at most `δ` in every session it runs.
-/

@[expose] public section

namespace Flock.SoundAtBridge

universe u

open FlockSoundness ResourceAccounting.TS
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

/-- **Read through its sessions, the window has the sessions' sampling design**: for every event `E` of a session's outcome, its
probability over the window's paths, at the session `sess x` the path `x` runs and its outcome `out x` there, is at
most the paths' average of its probability in that session's game `G (sess x)` played by its prover `τ (sess x)`. It
holds with equality when, given what the path fixes before the session (its setting and its prover), the session's
own coins are the path's next coins, fresh and uniform. -/
def SessionSamplingDesign {Ω : Type} [Fintype Ω] {S : ℕ} {Sess : Type u} {Out : Sess → Type}
    (G : ∀ s, Game (Out s)) (τ : ∀ s, Game.Strategy (G s)) (sess : Paths Ω S → Sess) (out : ∀ x, Out (sess x)) :
    Prop :=
  ∀ E : ∀ s, Out s → Prop,
    Pouw.RelativeSecurity.prob (fun x => E (sess x) (out x)) ≤
      (∑ x, (Game.prob (E (sess x)) (G (sess x)) (τ (sess x))).toReal) / (Fintype.card (Paths Ω S) : ℝ)

end Flock.SoundAtBridge
