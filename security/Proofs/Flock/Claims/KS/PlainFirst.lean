module

public import Proofs.Flock.Claims.Bridge.Custody
public import Proofs.Flock.Claims.KS.LiveMix
public import Proofs.Flock.Claims.KS.PlainEvent
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
import Proofs.Flock.Claims.KS.Width

/-!
# A simulation of the plain runs, and the record's sample

The plain mixture's first bound (`PlainDrawSim`'s `plain_first_of_simD`) reads a simulation of `P`'s deployed runs:

* `PlainSimRun`, `PlainSim`: coins `Φ (u, r')` of `V false` at the tape `u` and the live coins `r'`, with the sampling design of
  uniform coins, and a prover `P'` of the live session such that wherever `V false` accepts at `Φ (u, r')`, the live
  outcome of `P'` at the tape `(ω, u)` is the run's (`viewOf`);
* `plainCoins`, `plainSim_of_run`: where each deployed coin takes a live coin's first entries and the rest from the
  tape, the sampling design holds, so `PlainSimRun` gives `PlainSim`.

The record's sample is the statement's only up to key order on accepting runs (`RecordDrawEquiv`, `recordDrawEquiv`),
and that is all the checks give (`drawEq_ne`), so the chain reads the session set up at the statement's sample.
-/

@[expose] public section

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Tables Claims.Bridge.Custody Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **`P`'s accepting runs carry the statement's sample up to key order**: on every accepting run of `V false` at `x`
and `ω`, the last message's record has a sample `JsonEquiv` to the statement's, what `checks` compares. -/
def RecordDrawEquiv {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) : Prop :=
  ∀ r : Fin ((V false).k x) → Coin, (Claims.run sha512 P (V false) x ω r).out = .acc →
    ∀ s spec sts link c₁ mids f,
      act false x ((Claims.run sha512 P (V false) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (V false) x ω r).msgs.map Prod.fst).length).map
            (Claims.coinSeq r)) = .decide s spec sts link c₁ mids f →
        Option.Rel JsonEquiv (Exec.recordDraw f.record) s.read.draw

/-- **Every prover's accepting runs carry the statement's sample up to key order**: an accepting decision passes
`checks`, whose sample check is `drawEq` (`checks_draw`). -/
theorem recordDrawEquiv {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) : RecordDrawEquiv P x ω := by
  intro r h s spec sts link c₁ mids f hact
  have hacc := run_acc_next sha512 P (V false) x ω r h
  replace hacc : ((next false x ((Claims.run sha512 P (V false) x ω r).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (V false) x ω r).msgs.map Prod.fst).length).map
        (Claims.coinSeq r))).eval sha512).1 = .acc := hacc
  exact Claims.Bridge.Custody.checks_draw ((next_decide_sha512 hact).1 hacc).2

/-- **`drawEq` passes on different terms**: two objects with the same fields in trees of different shapes are
`JsonEquiv`, so the checks passing does not give the record the statement's sample as the same term. -/
theorem drawEq_ne : ∃ a b : Json, drawEq (some a) (some b) = true ∧ a ≠ b := by
  refine ⟨.obj ⟨⟨.inner 2 "a" .null .leaf (.inner 1 "b" .null .leaf .leaf)⟩⟩,
    .obj ⟨⟨.inner 2 "b" .null (.inner 1 "a" .null .leaf .leaf) .leaf⟩⟩, (drawEq_iff _ _).2 (.some rfl), fun h => ?_⟩
  have e := congrArg (fun j : Json => match j with | .obj ⟨⟨.inner _ k _ _ _⟩⟩ => k | _ => "") h
  exact absurd e (by decide)

/-- **A prover of the live session at one table simulates `P` on the coins `Φ`**: wherever `V false` accepts at
`Φ (u, r')`, the coins of `V false` at the tape `u` and the live coins `r'`, the live outcome of `P'` at the tape
`(ω, u)` and `r'` is the run's (`viewOf`). -/
def PlainSimRun {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) {nMax : ℕ} (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire nMax 1 mPts) {τ : Type} (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin) : Prop :=
  ∀ u r', (Claims.run sha512 P (V false) x ω (Φ (u, r'))).out = .acc →
    ∀ s spec sts link c₁ mids f,
      act false x ((Claims.run sha512 P (V false) x ω (Φ (u, r'))).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (V false) x ω (Φ (u, r'))).msgs.map Prod.fst).length).map
            (Claims.coinSeq (Φ (u, r')))) = .decide s spec sts link c₁ mids f →
        liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r' = viewOf 1 mPts spec link c₁ f

/-- **A prover of the live session at one table simulates `P`**: the coins `Φ (u, r')` have the sampling design of uniform coins,
and `P'` simulates `P` on them (`PlainSimRun`). -/
def PlainSim {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) {nMax : ℕ} (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire nMax 1 mPts) {τ : Type} [Fintype τ] [Nonempty τ]
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin) : Prop :=
  (∀ E : (Fin ((V false).k x) → Coin) → Prop, Claims.Pr (fun p => E (Φ p)) = Claims.Pr E) ∧
    PlainSimRun P x ω mPts N dg w P' Φ

/-- **The deployed coins at the tape `u` and the live coins `r'`**: coin `j` takes its first `n j` field elements from
the live coin `r' j` and the rest from `u j` (`mergeAt`). -/
def plainCoins {x : Claims.Msg} {N : ℕ} (hk : N + 2 = (V false).k x) (n : Fin ((V false).k x) → ℕ)
    (p : (Fin ((V false).k x) → Coin) × (Fin (N + 2) → LCoin 65536)) : Fin ((V false).k x) → Coin :=
  @mergeAt _ _ _ 𝔽 hk n p.1 p.2

/-- **`P'` simulates `P` on `plainCoins`** where it simulates the runs: the merged coins are uniform (`Pr_mergeAt`). -/
theorem plainSim_of_run {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ} {N : Claims.Msg → ℕ}
    {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire 65536 1 mPts}
    {P' : Claims.Prover (Ω × (Fin ((V false).k x) → Coin))} (hk : N x + 2 = (V false).k x)
    (n : Fin ((V false).k x) → ℕ) (h : PlainSimRun P x ω mPts N dg w P' (plainCoins hk n)) :
    PlainSim P x ω mPts N dg w P' (plainCoins hk n) :=
  ⟨fun E => @Pr_mergeAt _ _ _ 𝔽 _ _ hk n E, h⟩

end FlockDeploy.KS
