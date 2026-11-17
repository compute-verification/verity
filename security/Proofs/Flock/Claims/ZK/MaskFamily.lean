module

public import Proofs.Flock.Claims.ZK.MaskTape
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
import Proofs.Flock.Soundness.CROnly.Hiding

/-! The deployed masked session on a class of statements, and per statement. -/

@[expose] public section

namespace Claims.ZKMal

open Claims FlockSoundness.Merkle FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- instance search over the ZK view's and coins' product types

theorem FibersEq.surjective {α β : Type} [Finite α] [Nonempty α] {f : α → β} (hf : FibersEq f) :
    Function.Surjective f := by
  obtain ⟨a₀⟩ := ‹Nonempty α›
  intro b
  have : Nonempty {a // f a = f a₀} := ⟨⟨a₀, rfl⟩⟩
  have h0 : 0 < Nat.card {a // f a = f a₀} := Nat.card_pos
  rw [hf (f a₀) b] at h0
  obtain ⟨a, ha⟩ := (Nat.card_pos_iff.1 h0).1.some
  exact ⟨a, ha⟩

namespace Masked

theorem jOf_lt_card_of_fibersEq {C : Tabs} (U : TapePlan C) {x : Msg} (h : FibersEq (U.tape x)) :
    jOf x < Fintype.card U.Tape := by
  classical
  have hs := h.surjective
  have hs2 : Function.Surjective fun t => (U.tape x t).2.1 := fun y => by
    obtain ⟨t, ht⟩ := hs ((U.tape x (Classical.arbitrary _)).1, y, (U.tape x (Classical.arbitrary _)).2.2)
    exact ⟨t, congrArg (fun r => r.2.1) ht⟩
  have h1 := Fintype.card_le_of_surjective _ hs2
  let ι : Fin (jOf x) → ZkView.Leaves fun j => (C x j).z := fun j => ⟨j, .inr (.inr 0)⟩
  have h2 := Fintype.card_le_of_injective ι fun a b h => congrArg Sigma.fst h
  rw [Fintype.card_fin] at h2
  have hS : 2 ≤ Fintype.card ZkView.Salt := Fintype.one_lt_card
  rw [Fintype.card_fun] at h1
  calc jOf x < 2 ^ jOf x := Nat.lt_two_pow_self
    _ ≤ 2 ^ Fintype.card (ZkView.Leaves fun j => (C x j).z) := Nat.pow_le_pow_right (by norm_num) h2
    _ ≤ Fintype.card ZkView.Salt ^ Fintype.card (ZkView.Leaves fun j => (C x j).z) := Nat.pow_le_pow_left hS _
    _ ≤ Fintype.card U.Tape := h1

theorem jOf_lt_card_on {C : Tabs} (U : TapePlan C) {A : Msg → Prop} (hT : ∀ x, A x → Live x → FibersEq (U.tape x))
    {x : Msg} (hA : A x) (hx : Live x) : jOf x < Fintype.card U.Tape :=
  jOf_lt_card_of_fibersEq U (hT x hA hx)

theorem not_fibersEq_live {C : Tabs} (hJ : ∀ N, ∃ x, Live x ∧ N ≤ jOf x) (U : TapePlan C) :
    ¬ ∀ x, Live x → FibersEq (U.tape x) := fun hT => by
  obtain ⟨x, hx, hN⟩ := hJ (Fintype.card U.Tape)
  exact absurd (jOf_lt_card_of_fibersEq U (hT x hx)) (not_lt.mpr hN)

end Masked

open Masked

open Classical in
theorem Flock.zk_deployZk_masked_on (H : Msg → Msg) (nr : Msg → ℕ → ℕ) (P : Composed.LigProver)
    (Q : RoundPlan (tabC P)) (U : TapePlan (tabC P)) (A : Msg → Prop) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hc : Q.Causal)
    (hT : ∀ x, A x → Live x → FibersEq (U.tape x))
    (hpre : (session nr (tableData P Q U)).HonestPreOn (fun x => A x ∧ Live x) RZk H coinOfD fun ω => U.tapeZk ω.2)
    (hfin : (session nr (tableData P Q U)).HonestFinOn (fun x => A x ∧ Live x) RZk H coinOfD fun ω => U.tapeZk ω.2)
    (hoff : (session nr (tableData P Q U)).HonestOffOn (fun x => A x ∧ Live x) RZk H coinOfD fun ω => U.tapeZk ω.2) :
    Claims.Flock.ZK H (zkDeployed _root_.Flock.Readable.relation nr (tableData P Q U) t)
      (fun x => if A x ∧ Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL P x * δ₁.toReal + 8 * t * nL P x * δ₁.toReal + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t := by
  exact zk_if_congr (Flock.zk_deployZk_ref_on_rounds H _root_.Flock.Readable.relation nr (tableData P Q U)
    (fun x => A x ∧ Live x) t ht hT1 hδ₁ (tables P Q U nr) (complete P Q U nr _) (dummyReg P Q U nr)
    (RoundPlan.causal hc U nr) (fun x hx => hT x hx.1 hx.2) hpre hfin hoff)
    (fun _ _ hR => and_congr_right fun _ => live_iff hR) fun _ => le_rfl

open Classical in
theorem Flock.zk_deployZk_masked_on_gap (H : Msg → Msg) (nr : Msg → ℕ → ℕ) (P : Composed.LigProver)
    (Q : RoundPlan (tabC P)) (U : TapePlan (tabC P)) (A : Msg → Prop) (t : ℕ) (ht : 0 < t) (hc : Q.Causal)
    (hT : ∀ x, A x → Live x → FibersEq (U.tape x))
    (hpre : (session nr (tableData P Q U)).HonestPreOn (fun x => A x ∧ Live x) RZk H coinOfD fun ω => U.tapeZk ω.2)
    (hfin : (session nr (tableData P Q U)).HonestFinOn (fun x => A x ∧ Live x) RZk H coinOfD fun ω => U.tapeZk ω.2)
    (hoff : (session nr (tableData P Q U)).HonestOffOn (fun x => A x ∧ Live x) RZk H coinOfD fun ω => U.tapeZk ω.2) :
    Claims.Flock.ZK H (zkDeployed _root_.Flock.Readable.relation nr (tableData P Q U) t)
      (fun x => if A x ∧ Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
          8 * t * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t := by
  have h := Flock.zk_deployZk_masked_on H nr P Q U A t ht
    (FlockSoundness.CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash) ENNReal.ofReal_ne_top hc hT hpre hfin hoff
  rwa [ENNReal.toReal_ofReal (FlockSoundness.CROnly.hidingGap_nonneg _ _)] at h

section Family

variable {Coin Col T : Type}

noncomputable def famSim (D : Msg → Claims.Flock.ZkDeploy Coin Col T) : Simulator where
  C x := (D x).S.C x
  fin x := (D x).S.fin x
  ne x := (D x).S.ne x
  prog x := (D x).S.prog x

def ZKfam (H : Msg → Msg) (D : Msg → Claims.Flock.ZkDeploy Coin Col T) (R : Rel Claims.Flock.Wit)
    (εzk γ : Msg → ℝ) : Prop :=
  ∀ x w, R x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = (D x).V.k x + 1 → ∀ (Ev : List (Msg × Out) → Prop),
    |Pr (fun ω : (D x).Nonce × (D x).Rest => Ev (runStar H ((D x).prover w) Vs x ω θ).msgs) -
        Pr (fun c : (famSim D).C x => Ev ((famSim D).out H Vs θ x c))|
      ≤ εzk x + γ x * (Fork H ((D x).prover w) Vs θ x).Pr H

theorem zkFam_const_iff {H : Msg → Msg} {D : Claims.Flock.ZkDeploy Coin Col T} {εzk γ : Msg → ℝ} :
    ZKfam H (fun _ => D) (Claims.Flock.R D.parse) εzk γ ↔
      ZKmalR H (fun x => D.V.k x + 1) D.prover (Claims.Flock.R D.parse) D.S εzk γ :=
  Iff.rfl

theorem zkFam_of {H : Msg → Msg} {D : Msg → Claims.Flock.ZkDeploy Coin Col T} {R : Rel Claims.Flock.Wit}
    {ε : Msg → Msg → ℝ} {ε' γ : Msg → ℝ} (h : ∀ x₀, Claims.Flock.ZK H (D x₀) (ε x₀) γ)
    (hR : ∀ x w, R x w → Claims.Flock.R (D x).parse x w) (hε : ∀ x, ε x x ≤ ε' x) : ZKfam H D R ε' γ :=
  fun x w hRx Θ Vs θ hk Ev => ((h x).2 x w (hR x w hRx) Θ Vs θ hk Ev).trans (by linarith [hε x])

end Family

noncomputable abbrev maskedAt (nr : Msg → ℕ → ℕ) (P : Composed.LigProver) (Q : RoundPlan (tabC P))
    (z : (x₀ : Msg) → Rand (tabC P) x₀ → FlockComplete.TapeZk) (t : ℕ) (x₀ : Msg) :
    Claims.Flock.ZkDeploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace :=
  zkDeployed _root_.Flock.Readable.relation nr (tableData P Q (TapePlan.atMask x₀ (z x₀))) t

theorem ite_self_and_le {S : Prop} [Decidable S] {x : Msg} {E : ℝ} [Decidable (x = x ∧ S)] :
    (if x = x ∧ S then E else 1) ≤ if S then E else 1 := by
  split_ifs with h1 h2 h2
  · exact le_rfl
  · exact absurd h1.2 h2
  · exact absurd ⟨rfl, h2⟩ h1
  · exact le_rfl

open Classical in
theorem Flock.zk_deployZk_masked_atMask_rx (H : Msg → Msg) (nr : Msg → ℕ → ℕ) (P : Composed.LigProver)
    (Q : RoundPlan (tabC P)) (z : (x₀ : Msg) → Rand (tabC P) x₀ → FlockComplete.TapeZk) (t : ℕ) (ht : 0 < t)
    (hc : Q.Causal)
    (hpre : ∀ x₀, (session nr (tableData P Q (TapePlan.atMask x₀ (z x₀)))).HonestPreOn (fun x => x = x₀ ∧ Live x) RZk
      H coinOfD fun ω => (TapePlan.atMask x₀ (z x₀)).tapeZk ω.2)
    (hfin : ∀ x₀, (session nr (tableData P Q (TapePlan.atMask x₀ (z x₀)))).HonestFinOn (fun x => x = x₀ ∧ Live x) RZk
      H coinOfD fun ω => (TapePlan.atMask x₀ (z x₀)).tapeZk ω.2)
    (hoff : ∀ x₀, (session nr (tableData P Q (TapePlan.atMask x₀ (z x₀)))).HonestOffOn (fun x => x = x₀ ∧ Live x) RZk
      H coinOfD fun ω => (TapePlan.atMask x₀ (z x₀)).tapeZk ω.2) :
    ZKfam H (maskedAt nr P Q z t) RZk
      (fun x => if Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
          8 * t * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t :=
  zkFam_of (fun x₀ => Flock.zk_deployZk_masked_atMask_gap H nr P Q x₀ (z x₀) t ht hc (hpre x₀) (hfin x₀) (hoff x₀))
    (fun _ _ h => h) fun _ => ite_self_and_le

open Classical in
theorem Flock.zk_deployZk_masked_atMask_stop_rx (H : Msg → Msg) (nr : Msg → ℕ → ℕ) (P : Composed.LigProver)
    (cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (enc : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((tabC P x j).T (cz x c j)).tab.Cut) → Msg)
    (fin : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((tabC P x j).T (cz x c j)).tab.View) → Msg)
    (z : (x₀ : Msg) → Rand (tabC P) x₀ → FlockComplete.TapeZk) (t : ℕ) (ht : 0 < t)
    (hc : (RoundPlan.stop cz enc fin).Causal)
    (hpre : ∀ x₀, (session nr (tableData P (RoundPlan.stop cz enc fin) (TapePlan.atMask x₀ (z x₀)))).HonestPreOn
      (fun x => x = x₀ ∧ Live x) RZk H coinOfD fun ω => (TapePlan.atMask x₀ (z x₀)).tapeZk ω.2)
    (hfin : ∀ x₀, (session nr (tableData P (RoundPlan.stop cz enc fin) (TapePlan.atMask x₀ (z x₀)))).HonestFinOn
      (fun x => x = x₀ ∧ Live x) RZk H coinOfD fun ω => (TapePlan.atMask x₀ (z x₀)).tapeZk ω.2)
    (hoff : ∀ x, Live x → ∀ w, RZk x w → ∀ r (ops : List (Msg × Msg)), ops.length = refK x →
      ¬ (session nr (tableData P (RoundPlan.stop cz enc fin) (TapePlan.atMask x (z x)))).Admits x
        ((session nr (tableData P (RoundPlan.stop cz enc fin) (TapePlan.atMask x (z x)))).pad
          (ops.map fun p => rd p.2)) →
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2))
        ((TapePlan.atMask x (z x)).tapeZk r)).eval H).1 = []) :
    ZKfam H (maskedAt nr P (RoundPlan.stop cz enc fin) z t) RZk
      (fun x => if Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
          8 * t * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t :=
  zkFam_of (fun x₀ => Flock.zk_deployZk_masked_atMask_stop_gap H nr P cz enc fin x₀ (z x₀) t ht hc (hpre x₀)
      (hfin x₀) fun _ hx => by obtain ⟨rfl, hl⟩ := hx; exact hoff _ hl)
    (fun _ _ h => h) fun _ => ite_self_and_le

end Claims.ZKMal
