module

public import Proofs.Flock.Claims.DeployCode
public import Proofs.Flock.Claims.ZK.Congr
public import Proofs.Flock.Claims.ZK.GK
public import Proofs.Flock.Claims.ZK.Statement
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

/-! Flock's `ZK` at the masked record's reference prover. -/

@[expose] public section

namespace Claims.ZKMal

open Claims
open Claims.Flock (allPass)

local notation "𝔽" => FlockLevel3.GF128

theorem ocBind_eq_bind {α β : Type} (f : α → OC β) : ∀ o : OC α, ocBind o f = o.bind f
  | .ret _ => rfl
  | .query q k => by
      simp only [ocBind, OC.bind]
      exact congrArg (OC.query q) (funext fun r => ocBind_eq_bind f (k r))

theorem allOk_eq_allPass (acc : ℕ → Msg → OC Bool) : ∀ (j : ℕ) (ms : List Msg), allOk acc j ms = allPass acc j ms
  | _, [] => rfl
  | j, m :: ms => by
      simp only [allOk, allPass, ocBind_eq_bind, allOk_eq_allPass acc (j + 1) ms]

/-- **The masked record at a committed-coin protocol** `G`: `FlockDeploy.deployZk rel` with `G`'s nonce and tape, its
`Hello`, abort message and opening check, the coin each opening reads as (`coinOf`), the reference prover's tape at
`G`'s (`tapeOf`), and the simulator `S`. -/
noncomputable def zkOf (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit)
    (coinOf : Msg → Msg) (tapeOf : G.Nonce × G.Rand → FlockComplete.TapeZk) (S : Simulator) :
    Claims.Flock.ZkDeploy FlockDeploy.Coin (List 𝔽) Claims.Flock.ZkTrace :=
  { FlockDeploy.deployZk rel with
    Nonce := G.Nonce, Rest := G.Rand, tapeOf := tapeOf, hello := G.hello, abort := G.abort, chk := G.chk,
    coinOf := coinOf, S := S }

def Proto.Honest {Ω : Type} (G : Proto Flock.Wit) (H : Msg → Msg) (P : Flock.Wit → Prover Ω) (coinOf : Msg → Msg)
    (tapeOf : G.Nonce × G.Rand → Ω) : Prop :=
  ∀ x w ν r (ops : List (Msg × Msg)), ops.length ≤ G.K x →
    ((G.msg x w ν r (ops.map fun p => G.dec p.2)).eval H).1 =
        (((P w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf (ν, r))).eval H).1 ∧
      ((G.msg x w ν r (ops.map fun p => G.dec p.2)).eval H).2 ⊆
        (((P w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf (ν, r))).eval H).2

theorem zkOf_prover (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit)
    (coinOf : Msg → Msg) (tapeOf : G.Nonce × G.Rand → FlockComplete.TapeZk) (S : Simulator) (H : Msg → Msg)
    (hG : G.Honest H FlockComplete.ZkTables.honestPR coinOf tapeOf) (w : Flock.Wit) (x : Msg) (hist : List (Msg × Msg))
    (ω : G.Nonce × G.Rand) (hlen : hist.length < G.k x) :
    (((G.prover w).next x hist ω).eval H).1 = ((((zkOf rel G coinOf tapeOf S).prover w).next x hist ω).eval H).1 ∧
      (((G.prover w).next x hist ω).eval H).2 ⊆ ((((zkOf rel G coinOf tapeOf S).prover w).next x hist ω).eval H).2 := by
  cases hist with
  | nil => exact ⟨rfl, List.Subset.refl _⟩
  | cons p ops =>
    have hops : ops.length ≤ G.K x := by
      simp only [List.length_cons, Proto.k] at hlen
      omega
    obtain ⟨a, ρ⟩ := p
    have e₁ : (G.prover w).next x ((a, ρ) :: ops) ω = (allPass (G.chk x ω.1 ρ) 0 (ops.map Prod.snd)).bind
        fun ok => if ok then G.msg x w ω.1 ω.2 (ops.map fun p => G.dec p.2) else .ret G.abort := by
      show ocBind (allOk (G.chk x ω.1 ρ) 0 (ops.map Prod.snd)) _ = _
      rw [ocBind_eq_bind, allOk_eq_allPass]
    have e₂ : ((zkOf rel G coinOf tapeOf S).prover w).next x ((a, ρ) :: ops) ω =
        (allPass (G.chk x ω.1 ρ) 0 (ops.map Prod.snd)).bind fun ok =>
          if ok then (FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf ω)
          else .ret G.abort := rfl
    rw [e₁, e₂, OC.eval_bind, OC.eval_bind]
    obtain ⟨h₁, h₂⟩ := hG x w ω.1 ω.2 ops hops
    cases ((allPass (G.chk x ω.1 ρ) 0 (ops.map Prod.snd)).eval H).1 with
    | true =>
        simp only [↓reduceIte]
        exact ⟨h₁, List.append_subset.mpr
          ⟨List.subset_append_left _ _, List.Subset.trans h₂ (List.subset_append_right _ _)⟩⟩
    | false => exact ⟨rfl, List.Subset.refl _⟩

theorem zk_zkOf (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit)
    (coinOf : Msg → Msg) (tapeOf : G.Nonce × G.Rand → FlockComplete.TapeZk) (S : Simulator) {H : Msg → Msg}
    {εzk γ : Msg → ℝ} (hγ : ∀ x, 0 ≤ γ x)
    (hk : ∀ x w, Flock.R (FlockDeploy.deployZk rel).parse x w → G.k x = (FlockDeploy.deployZk rel).V.k x + 1)
    (hG : G.Honest H FlockComplete.ZkTables.honestPR coinOf tapeOf)
    (h : Claims.Flock.ZKAt H (FlockDeploy.deployZk rel) G.prover S εzk γ) :
    Claims.Flock.ZK H (zkOf rel G coinOf tapeOf S) εzk γ :=
  ⟨h.1, zkmalR_congr_on hγ
    (fun w x hR hist ω hh =>
      (zkOf_prover rel G coinOf tapeOf S H hG w x hist ω (lt_of_lt_of_eq hh (hk x w hR).symm)).1)
    (fun w x hR hist ω hh =>
      (zkOf_prover rel G coinOf tapeOf S H hG w x hist ω (lt_of_lt_of_eq hh (hk x w hR).symm)).2)
    h.2⟩

end Claims.ZKMal
