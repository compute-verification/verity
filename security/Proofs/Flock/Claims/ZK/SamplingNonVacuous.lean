module

public import Definitions.Claims.Sampling
public import Proofs.Flock.Claims.ZK.Indexed
public import Proofs.Flock.Claims.ZK.NonVacuous
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
import Mathlib.Tactic.ENatToNat
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

/-! The sampling `ZK` is not vacuous. -/

@[expose] public section

namespace Claims.Sampling.ZKMal

open Claims Claims.ZKMal

def collSimLeak (u v : Msg) : SimLeak where
  C _ := Unit
  prog _ _ _ := .hash u fun _ => .hash v fun _ => .ret []

theorem not_samplingZK_collSim {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] (H : Msg → Msg) (D : ZkDeploy Coin Col T) (u v : Msg) (εzk γ : Msg → ℝ) (x : Msg)
    (w : Wit) (hP : ∀ w, D.prover w x = quietP (D.Nonce x) (D.Rest x) []) (hS : D.S = collSimLeak u v)
    (hR : Rhon H D.commit D.parse D.partition x w) (hk : 0 < D.V.k x) (hε : εzk x < 1) : ¬ ZK H D εzk γ := fun h => by
  have h2 := (h.2 fun x' => leafLayer H D.commit D.parse x' w).at x
  rw [funext hP, hS] at h2
  exact not_zkmalR_of_out H D.V.k _ [] _ εzk γ x w ⟨rfl, hR, rfl⟩ hk hε
    (fun _ => by simp [Simulator.out, SimLeak.given, collSimLeak, SP.eval]) h2

theorem not_samplingZK_leak {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] (H : Msg → Msg) (D : ZkDeploy Coin Col T) (f : Wit → Msg) (εzk γ : Msg → ℝ) (x : Msg)
    (w₁ w₂ : Wit) (hP : ∀ w, D.prover w x = quietP (D.Nonce x) (D.Rest x) (f w))
    (h₁ : Rhon H D.commit D.parse D.partition x w₁) (h₂ : Rhon H D.commit D.parse D.partition x w₂)
    (hl : leafLayer H D.commit D.parse x w₁ = leafLayer H D.commit D.parse x w₂) (hf : f w₁ ≠ f w₂)
    (hk : 0 < D.V.k x) (hε : εzk x < 1 / 2) : ¬ ZK H D εzk γ := fun h => by
  have h2 := (h.2 fun _ => leafLayer H D.commit D.parse x w₁).at x
  rw [funext hP] at h2
  exact not_zkmalR_leak H D.V.k _ f _ εzk γ x w₁ w₂ ⟨rfl, h₁, rfl⟩ ⟨rfl, h₂, hl.symm⟩ hf hk hε h2

noncomputable def quietZD {Coin Col T : Type} (D₀ : Deploy Coin Col T) (u v : Msg) : ZkDeploy Coin Col T :=
  { D₀ with
    P := fun _ => ⟨fun _ _ _ => .ret []⟩
    Nonce := fun _ => Unit
    Rest := fun _ => Unit
    tapeOf := fun _ _ => Classical.choice D₀.tapeNonempty
    S := collSimLeak u v }

theorem not_forall_samplingZK {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] (H : Msg → Msg) (D₀ : Deploy Coin Col T) (hzk : D₀.flock.zk = true) (x : Msg)
    (w : Wit) (hR : Rhon H D₀.commit D₀.parse D₀.partition x w) (hk : 0 < D₀.V.k x) :
    ¬ ∀ D : ZkDeploy Coin Col T, D.flock.zk = true → ZK H D 0 0 := fun h =>
  not_samplingZK_collSim H (quietZD D₀ [] []) [] [] 0 0 x w (fun _ => rfl) rfl hR hk one_pos (h _ hzk)

end Claims.Sampling.ZKMal
