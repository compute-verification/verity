module

public import Definitions.Pouw.Guarantees.CoinFlow
public import Mathlib.Algebra.DirectSum.Internal
public import Mathlib.Geometry.Euclidean.Projection
public import Mathlib.MeasureTheory.Measure.FiniteMeasure
public import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
public import Mathlib.NumberTheory.LucasLehmer
public import Mathlib.RepresentationTheory.Homological.GroupCohomology.LowDegree
public import Mathlib.RingTheory.FractionalIdeal.Basic
public import Proofs.Pouw.HMu.Luck
import Definitions.Claims
import Mathlib
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
import Proofs.Poub.Tag.Basic
import Proofs.Poub.Tag.LittlewoodOfford
import Proofs.Poub.Tag.Signs
import Std.Tactic.BVDecide.Normalize.Prop

/-! The coin rules, proved on the dataflow model by strong induction over the node order. -/

@[expose] public section

namespace ResourceAccounting.CoinFlowProofs

open ResourceAccounting.CoinFlow ResourceAccounting.HMu Pouw.RelativeSecurity Finset

variable {m : ℕ} {C V : Type}

theorem admits_spec [DecidableEq C] {P : FlowGraph m C V} (h : P.admits = true) :
    P.Topological ∧ P.CoinFlowRule ∧ P.OneUse ∧ P.SinkConfined :=
  of_decide_eq_true h

theorem eval_eq [Inhabited V] (P : FlowGraph m C V) (σ : C → Bool) (n : Fin m) :
    P.eval σ n = P.op n (fun k => if _h : k.1 < n then P.eval σ k.1 else default) (fun c => σ c.1) := by
  rw [FlowGraph.eval]

theorem eval_congr [Inhabited V] (P : FlowGraph m C V) {σ σ' : C → Bool} {g : Fin m}
    (hc : ∀ c ∈ P.coins g, σ c = σ' c) (hr : ∀ k ∈ P.reads g, k < g → P.eval σ k = P.eval σ' k) :
    P.eval σ g = P.eval σ' g := by
  rw [eval_eq, eval_eq]
  congr 1
  · funext k
    split_ifs with h
    · exact hr k.1 k.2 h
    · rfl
  · funext c
    exact hc c.1 c.2

theorem fin_strong {Q : Fin m → Prop} (h : ∀ g, (∀ k, k < g → Q k) → Q g) (g : Fin m) : Q g := by
  suffices ∀ n (g : Fin m), g.val = n → Q g from this g.val g rfl
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro g hg
    exact h g fun k hk => ih k.val (hg ▸ hk) k rfl

theorem mem_noiseCoins {P : FlowGraph m C V} {k : Fin m} {c : C} :
    c ∈ P.noiseCoins k ↔ P.noise k = true ∧ c ∈ P.coins k := by
  unfold FlowGraph.noiseCoins
  by_cases hk : P.noise k = true
  · simp [hk]
  · simp [hk]

theorem viewIndependent : ResourceAccounting.Guarantees.CoinFlow.ViewIndependent := by
  intro m C V _ _ P hP k σ σ' hσ
  obtain ⟨-, -, hone, -⟩ := admits_spec hP
  refine fin_strong fun g ih hd => eval_congr P (fun c hc => hσ c fun hck => ?_) fun r hr hlt =>
    ih r hlt fun hdr => hd (hdr.tail hr)
  obtain ⟨hk, hck⟩ := mem_noiseCoins.mp hck
  have : g = k := (hone k hk).1 c hck g hc
  exact hd (this ▸ Relation.ReflTransGen.refl)

theorem coinsThroughNoise : ResourceAccounting.Guarantees.CoinFlow.CoinsThroughNoise := by
  intro m C V _ _ P hP g σ σ'
  obtain ⟨-, hflow, -, hsink⟩ := admits_spec hP
  revert g
  refine fin_strong fun g ih hg hagree => eval_congr P (fun c hc => ?_) fun r hr hlt => ?_
  · have hn : P.noise g = true := by
      rcases hflow g ⟨c, hc⟩ with h | h
      · exact h
      · rw [hg] at h; exact absurd h (by decide)
    exact hagree g Relation.ReflTransGen.refl c (mem_noiseCoins.mpr ⟨hn, hc⟩)
  · have hr' : P.fold r = false := by
      cases h : P.fold r with
      | false => rfl
      | true => rw [hsink g r hr h] at hg; exact absurd hg (by decide)
    exact ih r hlt hr' fun k hk => hagree k (hk.tail hr)

theorem oneReader : ResourceAccounting.Guarantees.CoinFlow.OneReader := by
  intro m C V _ P hP k r g hk hr hd
  obtain ⟨-, -, hone, -⟩ := admits_spec hP
  rcases Relation.ReflTransGen.cases_head hd with h | ⟨y, hy, hyg⟩
  · exact Or.inl h.symm
  · exact Or.inr ((hone k hk).2 y r hy hr ▸ hyg)

section Prob

theorem prob_comp_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (p : β → Prop) :
    prob (fun a => p (e a)) = prob p := by
  classical
  rw [HMuProofs.prob_eq_card, HMuProofs.prob_eq_card, Fintype.card_congr e]
  congr 1
  rw [← Fintype.card_subtype, ← Fintype.card_subtype]
  exact_mod_cast Fintype.card_congr (e.subtypeEquiv fun _ => Iff.rfl)

variable [DecidableEq C] (S : Finset C)

def glue (ω : ({c // c ∉ S} → Bool) × ({c // c ∈ S} → Bool)) (c : C) : Bool :=
  if h : c ∈ S then ω.2 ⟨c, h⟩ else ω.1 ⟨c, h⟩

def split : (({c // c ∉ S} → Bool) × ({c // c ∈ S} → Bool)) ≃ (C → Bool) where
  toFun := glue S
  invFun σ := (fun c => σ c.1, fun c => σ c.1)
  left_inv ω := by
    refine Prod.ext (funext fun c => ?_) (funext fun c => ?_)
    · simp [glue, c.2]
    · simp [glue, c.2]
  right_inv σ := by
    funext c
    by_cases h : c ∈ S
    · simp [glue, h]
    · simp [glue, h]

theorem glue_out (ω : ({c // c ∉ S} → Bool) × ({c // c ∈ S} → Bool)) (c : {c // c ∉ S}) : glue S ω c.1 = ω.1 c := by
  simp [glue, c.2]

theorem glue_in (ω : ({c // c ∉ S} → Bool) × ({c // c ∈ S} → Bool)) (c : {c // c ∈ S}) : glue S ω c.1 = ω.2 c := by
  simp [glue, c.2]

end Prob

theorem pointMass : ResourceAccounting.Guarantees.CoinFlow.PointMass := by
  intro m C V _ _ _ P hP k O F a v o t
  classical
  let V₀ : ({c // c ∉ P.noiseCoins k} → Bool) → P.View k := fun ω₀ =>
    P.view k (glue (P.noiseCoins k) (ω₀, fun _ => false))
  have hview : ∀ ω, P.view k (split (P.noiseCoins k) ω) = V₀ ω.1 := by
    intro ω
    refine Prod.ext (funext fun g => ?_) (funext fun c => ?_)
    · exact viewIndependent m C V P hP k _ _ (fun c hc => by
        rw [split, Equiv.coe_fn_mk, glue_out _ ω ⟨c, hc⟩, glue_out _ (ω.1, fun _ => false) ⟨c, hc⟩]) g.1 g.2
    · change glue _ ω c.1 = glue _ (ω.1, fun _ => false) c.1
      rw [glue_out, glue_out]
  have hcoins : ∀ ω, (fun j : {c // c ∈ P.noiseCoins k} => split (P.noiseCoins k) ω j.1) = ω.2 :=
    fun ω => funext (glue_in _ ω)
  have key := HMuProofs.coinPointMass ({c // c ∉ P.noiseCoins k} → Bool) O {c // c ∈ P.noiseCoins k}
    (fun ω₀ => F (V₀ ω₀)) (fun ω₀ => a (V₀ ω₀)) (fun ω₀ => v (V₀ ω₀)) o t
  rw [Fintype.card_coe] at key
  rw [← prob_comp_equiv (split (P.noiseCoins k)), ← prob_comp_equiv (split (P.noiseCoins k))]
  simp only [hview, hcoins]
  exact key

end ResourceAccounting.CoinFlowProofs
