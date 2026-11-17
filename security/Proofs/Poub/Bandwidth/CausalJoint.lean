module

public import Proofs.Poub.Bandwidth.Causal
public import Proofs.ResourceAccounting.Actual.Transducer
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Ring.Star
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.Int.Star
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
import Proofs.ResourceAccounting.Actual.Confidence
import Proofs.Poub.Bandwidth.TickedJoint

/-! Bandwidth adds, by counting (Mbw), on a strategy causal from every tick whose chip's view determines `z`. -/

@[expose] public section

namespace Bandwidth

open Pous (Bits pr)
open ResourceAccounting ResourceAccounting.SecurityProofs.Actual
open scoped ENNReal

namespace Causal

namespace Interface

variable {L : Type*} {N b h : ℕ}

def withReplies (I : Interface L N b h) (hbm : Store N b → ℕ → Acts L → ℕ → Bool)
    (side : Store N b → Oracle N b h → ℕ → Acts L → ℕ → Bool) : Interface L N b h :=
  { I with hbm := hbm, side := side }

theorem crosses_withReplies (I : Interface L N b h) (hbm : Store N b → ℕ → Acts L → ℕ → Bool)
    (side : Store N b → Oracle N b h → ℕ → Acts L → ℕ → Bool) {σ : Type*} (t₀ : ℕ)
    (F : ℕ → σ → List (Cross L) → Acts L) (st : σ) (o : List (Ticked.Obs h)) :
    (I.withReplies hbm side).crosses t₀ F st o = I.crosses t₀ F st o := by
  induction o with
  | nil => rfl
  | cons x o ih => simp only [crosses, ih]; rfl

theorem chip_withReplies (I : Interface L N b h) (hbm : Store N b → ℕ → Acts L → ℕ → Bool)
    (side : Store N b → Oracle N b h → ℕ → Acts L → ℕ → Bool) {S : ℕ} (t₀ : ℕ)
    (F : ℕ → Bits S → List (Cross L) → Acts L) (τ : Acts L → Fin N → Bits h) :
    (I.withReplies hbm side).chip t₀ F τ = I.chip t₀ F τ := by
  have hd : ∀ st o, (I.withReplies hbm side).done t₀ F st o = I.done t₀ F st o := fun st o => by
    simp only [done, crosses_withReplies]
  simp only [chip, hd]
  rfl

end Interface

def SplitsJoint {L : Type*} {N b h n : ℕ} (S : (Store N b × Bits n) × Oracle N b h → Actual.Run L) (X : Finset L)
    (t₀ T βs : ℕ) (I : Interface L N b h) (hbm : Bits n → Store N b → ℕ → Acts L → ℕ → Bool)
    (side : Bits n → Store N b → Oracle N b h → ℕ → Acts L → ℕ → Bool) : Prop :=
  ∀ E z O, ∀ i < T, (S ((E, z), O)).crossing (↑X)ᶜ ↑X {t₀ + i} =
    I.cross (t₀ + i) ((S ((E, z), O)).acts ↑X (Set.Ico t₀ (t₀ + i)))
      ((I.withReplies (hbm z) (side z)).obs βs E O (t₀ + i) ((S ((E, z), O)).acts ↑X (Set.Ico t₀ (t₀ + i))))

noncomputable def unpad (M : ℕ) (st : Bits (M + 1)) : List Bool :=
  open Classical in
  if hst : ∃ c : List Bool, c.length ≤ M ∧ pad M c = st then hst.choose else []

theorem unpad_pad {M : ℕ} {c : List Bool} (hc : c.length ≤ M) : unpad M (pad M c) = c := by
  have hst : ∃ c' : List Bool, c'.length ≤ M ∧ pad M c' = pad M c := ⟨c, hc, rfl⟩
  simp only [unpad, hst, dite_true]
  exact pad_inj hst.choose_spec.1 hc hst.choose_spec.2

end Causal

namespace SecurityProofs.Causal

open Bandwidth.Causal

variable {L : Type*} [LinearOrder L] {N b h n : ℕ}

theorem run_eq_joint (S : (Store N b × Bits n) × Oracle N b h → Actual.Run L) (hc : CausalFrom S) (X : Finset L)
    (t₀ T M βs : ℕ) (hM : ∀ r, (S r).mem X t₀ ≤ M)
    (hpre : ∀ e O O', (S (e, O)).content X t₀ = (S (e, O')).content X t₀) (I : Interface L N b h)
    (H : Bits n → Store N b → ℕ → Acts L → ℕ → Bool) (Sd : Bits n → Store N b → Oracle N b h → ℕ → Acts L → ℕ → Bool)
    (hsplit : SplitsJoint S X t₀ T βs I H Sd) (τ : Acts L → Fin N → Bits h) :
    ∃ (F : ℕ → Bits (M + 1) → List (Cross L) → Acts L) (state : Store N b → Bits n → Bits (M + 1))
      (hbm : Bits n → Ticked.Hbm N b) (rest : Bits n → Ticked.Rest N b h),
      (∀ E z, state E z = pad M ((S ((E, z), fun _ _ => false)).content X t₀)) ∧
      (∀ E z O,
        (I.chip t₀ F τ).tags (state E z)
            (Ticked.observe (I.chip t₀ F τ) βs (state E z) (hbm z) (rest z) E O T [] [] []) =
          τ ((S ((E, z), O)).acts ↑X (Set.Ico t₀ (t₀ + T))) ∧
        Ticked.hbmBits (Ticked.observe (I.chip t₀ F τ) βs (state E z) (hbm z) (rest z) E O T [] [] []) =
          ∑ i ∈ Finset.range T, I.width (t₀ + i) ((S ((E, z), O)).acts ↑X (Set.Ico t₀ (t₀ + i))) ∧
        I.crosses t₀ F (state E z) (Ticked.observe (I.chip t₀ F τ) βs (state E z) (hbm z) (rest z) E O T [] [] []) =
          (List.range T).map fun i => (S ((E, z), O)).crossing (↑X)ᶜ ↑X {t₀ + i}) ∧
      ((∀ z E O O' u a, Sd z E O u a = Sd z E O' u a) → ∀ z E O O', rest z E O = rest z E O') := by
  obtain ⟨F, hF⟩ := transducer_bits hc X t₀ (fun r u => (S r).crossing (↑X)ᶜ ↑X {u}) (fun _ _ _ h => h) M hM
  let O₀ : Oracle N b h := fun _ _ => false
  let state : Store N b → Bits n → Bits (M + 1) := fun E z => pad M ((S ((E, z), O₀)).content X t₀)
  refine ⟨F, state, fun z => (I.withReplies (H z) (Sd z)).hbmOf t₀ F (fun E => state E z),
    fun z => (I.withReplies (H z) (Sd z)).restOf t₀ F (fun E => state E z), fun E z => rfl, fun E z O => ?_,
    fun hside z E O O' => ?_⟩
  · have hac : ∀ m ≤ T, F m (state E z) ((List.range m).map fun i => (S ((E, z), O)).crossing (↑X)ᶜ ↑X {t₀ + i}) =
        (S ((E, z), O)).acts ↑X (Set.Ico t₀ (t₀ + m)) := by
      intro m _
      rw [hF ((E, z), O) m]
      simp only [state, hpre (E, z) O₀ O]
    obtain ⟨h1, h2, h3⟩ := (I.withReplies (H z) (Sd z)).observe_eq t₀ T βs F τ (fun E => state E z) E O _ _ hac
      (hsplit E z O) T [] [] [] (by simp) rfl
    rw [Interface.chip_withReplies, Interface.crosses_withReplies] at h1
    rw [Interface.chip_withReplies] at h2 h3
    simp only [List.length_nil, zero_add] at h1 h2 h3
    refine ⟨?_, ?_, ?_⟩
    · change τ (F _ (state E z) (I.crosses t₀ F (state E z) _)) = _
      rw [h1, h2, hac T le_rfl]
    · rw [h3]
      simp only [Ticked.hbmBits, List.map_nil, List.sum_nil, zero_add]
      rfl
    · rw [h1, h2]
  · funext v u j
    simp only [Interface.restOf, Interface.withReplies, hside z E O O']

theorem audit_joint (S : (Store N b × Bits n) × Oracle N b h → Actual.Run L) (hc : CausalFrom S) (X : Finset L)
    (t₀ T M βs qt : ℕ) (hM : ∀ r, (S r).mem X t₀ ≤ M)
    (hpre : ∀ e O O', (S (e, O)).content X t₀ = (S (e, O')).content X t₀) (I : Interface L N b h)
    (hq : ∀ u a, (I.ask u a).length ≤ qt) (H : Bits n → Store N b → ℕ → Acts L → ℕ → Bool)
    (Sd : Bits n → Store N b → Oracle N b h → ℕ → Acts L → ℕ → Bool)
    (hside : ∀ z E O O' u a, Sd z E O u a = Sd z E O' u a) (hsplit : SplitsJoint S X t₀ T βs I H Sd)
    (ζ : List Bool → List (Cross L) → Bits n)
    (hζ : ∀ E z O, ζ ((S ((E, z), O)).content X t₀)
      ((List.range T).map fun i => (S ((E, z), O)).crossing (↑X)ᶜ ↑X {t₀ + i}) = z)
    (τ : Acts L → Fin N → Bits h) (s k lam : ℕ) :
    pr (fun x : (Store N b × Bits n) × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, τ ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + T))) (x.2.2 i) = x.2.1 (x.2.2 i, x.1.1 (x.2.2 i))) ∧
          (∑ i ∈ Finset.range T, I.width (t₀ + i) ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + i)))) + (M + 1) +
            βs * T + lam < k * b + n)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (((T * qt).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  obtain ⟨F, state, hbm, rest, hst, hrun, hro⟩ := run_eq_joint S hc X t₀ T M βs hM hpre I H Sd hsplit τ
  let ζt : Bits (M + 1) → List (Ticked.Obs h) → Bits n := fun st o => ζ (unpad M st) (I.crosses t₀ F st o)
  have hζt : ∀ E z O,
      ζt (state E z) (Ticked.observe (I.chip t₀ F τ) βs (state E z) (hbm z) (rest z) E O T [] [] []) = z := by
    intro E z O
    show ζ (unpad M (state E z)) (I.crosses t₀ F (state E z) _) = z
    rw [(hrun E z O).2.2, hst E z,
      unpad_pad ((content_length_le (S ((E, z), fun _ _ => false)) X t₀).trans (hM _)),
      hpre (E, z) (fun _ _ => false) O]
    exact hζ E z O
  refine (pr_mono fun x hx => ?_).trans
    (Ticked.audit_joint (I.chip t₀ F τ) βs qt T (fun st o => hq _ _) ζt state hbm rest (hro hside) hζt s k lam)
  obtain ⟨⟨E, z⟩, O, Is⟩ := x
  obtain ⟨ht, hr, -⟩ := hrun E z O
  refine ⟨fun i => ?_, ?_⟩
  · rw [ht]
    exact hx.1 i
  · rw [hr]
    exact hx.2

omit [LinearOrder L] in
theorem splits_of_links_joint (S : (Store N b × Bits n) × Oracle N b h → Actual.Run L) {K : Type*} (β : K → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (hα : Alphabet S β link) (X : Finset L) (t₀ T : ℕ)
    (hbmPorts sidePorts orPorts : List (K × L))
    (hports : ∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X →
      (link r g a, (S r).A.x g) ∈ hbmPorts ++ sidePorts ++ orPorts)
    (wd : ℕ → Acts L → ℕ) (hpm : PassiveMemory S link X t₀ T hbmPorts wd)
    (Q : ℕ → Acts L → List (Fin N × Bits b)) (hor : OracleLocation S link X t₀ T orPorts Q) :
    ∃ (I : Interface L N b h) (H : Bits n → Store N b → ℕ → Acts L → ℕ → Bool)
      (Sd : Bits n → Store N b → Oracle N b h → ℕ → Acts L → ℕ → Bool),
      I.width = wd ∧ I.ask = Q ∧ SplitsJoint S X t₀ T (sidePorts.map fun p => β p.1).sum I H Sd ∧
      ((∀ e O O' u, portSignals S link sidePorts (e, O) u = portSignals S link sidePorts (e, O') u) →
        ∀ z E O O' u a, Sd z E O u a = Sd z E O' u a) := by
  obtain ⟨G, hG, hGc⟩ := hpm
  obtain ⟨R, hR⟩ := hor
  set βs := (sidePorts.map fun p => β p.1).sum
  choose fH hfH using fun u a => exists_bits _ _ (hGc u a)
  choose fS hfS using fun u => exists_bits _ βs (portSignals_encard_le S β link hα u sidePorts)
  let H : Bits n → Store N b → ℕ → Acts L → ℕ → Bool := fun z E u a j =>
    if hj : j < wd u a then fH u a (G (E, z) u a) ⟨j, hj⟩ else false
  let Sd : Bits n → Store N b → Oracle N b h → ℕ → Acts L → ℕ → Bool := fun z E O u _ j =>
    if hj : j < βs then fS u (portSignals S link sidePorts ((E, z), O) u) ⟨j, hj⟩ else false
  let I₀ : Interface L N b h :=
    { width := wd
      hbm := fun _ _ _ _ => false
      side := fun _ _ _ _ _ => false
      ask := Q
      cross := fun _ _ _ => ∅ }
  let key : ℕ → (Store N b × Bits n) × Oracle N b h → Acts L × Ticked.Obs h := fun u r =>
    ((S r).acts ↑X (Set.Ico t₀ u),
      (I₀.withReplies (H r.1.2) (Sd r.1.2)).obs βs r.1.1 r.2 u ((S r).acts ↑X (Set.Ico t₀ u)))
  let cross : ℕ → Acts L → Ticked.Obs h → Cross L := fun u a x =>
    Function.extend (key u) (fun r => (S r).crossing (↑X)ᶜ ↑X {u}) (fun _ => ∅) (a, x)
  refine ⟨{ I₀ with cross := cross }, H, Sd, rfl, rfl, fun E z O i hi => ?_, fun h0 z E O O' u a => ?_⟩
  swap
  · funext j
    simp only [Sd, h0 (E, z) O O' u]
  have hf : Function.FactorsThrough (fun r => (S r).crossing (↑X)ᶜ ↑X {t₀ + i}) (key (t₀ + i)) := by
    rintro ⟨⟨E₁, z₁⟩, O₁⟩ ⟨⟨E₂, z₂⟩, O₂⟩ hk
    have hG1 := hG (E₁, z₁) O₁ i hi
    have hG2 := hG (E₂, z₂) O₂ i hi
    have hR1 := hR (E₁, z₁) O₁ i hi
    have hR2 := hR (E₂, z₂) O₂ i hi
    simp only [key] at hk
    generalize (S ((E₁, z₁), O₁)).acts ↑X (Set.Ico t₀ (t₀ + i)) = a at hk hG1 hR1
    generalize (S ((E₂, z₂), O₂)).acts ↑X (Set.Ico t₀ (t₀ + i)) = a₂ at hk hG2 hR2
    obtain ⟨rfl, hx⟩ := Prod.mk.inj hk
    simp only [Interface.obs, Interface.withReplies, Prod.mk.injEq, H, Sd] at hx
    obtain ⟨h1, h2, h3⟩ := hx
    have hH : G (E₁, z₁) (t₀ + i) a = G (E₂, z₂) (t₀ + i) a :=
      hfH _ _ ⟨(E₁, z₁), rfl⟩ ⟨(E₂, z₂), rfl⟩ (bits_inj h1)
    have hS : portSignals S link sidePorts ((E₁, z₁), O₁) (t₀ + i) =
        portSignals S link sidePorts ((E₂, z₂), O₂) (t₀ + i) :=
      hfS _ ⟨_, rfl⟩ ⟨_, rfl⟩ (bits_inj h2)
    have hall : portSignals S link (hbmPorts ++ sidePorts ++ orPorts) ((E₁, z₁), O₁) (t₀ + i) =
        portSignals S link (hbmPorts ++ sidePorts ++ orPorts) ((E₂, z₂), O₂) (t₀ + i) := by
      have e : ∀ r, portSignals S link (hbmPorts ++ sidePorts ++ orPorts) r (t₀ + i) =
          portSignals S link hbmPorts r (t₀ + i) ++ portSignals S link sidePorts r (t₀ + i) ++
            portSignals S link orPorts r (t₀ + i) := fun r => by
        simp only [portSignals, List.map_append]
      rw [e, e, hG1, hG2, hR1, hR2, hH, hS, h3]
    exact crossing_of_ports S link X _ hports (t₀ + i) _ _ hall
  show (S ((E, z), O)).crossing (↑X)ᶜ ↑X {t₀ + i} =
    Function.extend (key (t₀ + i)) (fun r => (S r).crossing (↑X)ᶜ ↑X {t₀ + i}) (fun _ => ∅)
      (key (t₀ + i) ((E, z), O))
  exact (hf.extend_apply (fun _ => ∅) ((E, z), O)).symm

theorem audit_links_joint (S : (Store N b × Bits n) × Oracle N b h → Actual.Run L) (hc : CausalFrom S) {K : Type*}
    (β : K → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (hα : Alphabet S β link) (X : Finset L)
    (t₀ T M qt : ℕ) (hM : ∀ r, (S r).mem X t₀ ≤ M)
    (hpre : ∀ e O O', (S (e, O)).content X t₀ = (S (e, O')).content X t₀) (hbmPorts sidePorts orPorts : List (K × L))
    (hports : ∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X →
      (link r g a, (S r).A.x g) ∈ hbmPorts ++ sidePorts ++ orPorts)
    (wd : ℕ → Acts L → ℕ) (hpm : PassiveMemory S link X t₀ T hbmPorts wd)
    (Q : ℕ → Acts L → List (Fin N × Bits b)) (hq : ∀ u a, (Q u a).length ≤ qt)
    (hor : OracleLocation S link X t₀ T orPorts Q)
    (h0 : ∀ e O O' u, portSignals S link sidePorts (e, O) u = portSignals S link sidePorts (e, O') u)
    (ζ : List Bool → List (Cross L) → Bits n)
    (hζ : ∀ E z O, ζ ((S ((E, z), O)).content X t₀)
      ((List.range T).map fun i => (S ((E, z), O)).crossing (↑X)ᶜ ↑X {t₀ + i}) = z)
    (τ : Acts L → Fin N → Bits h) (s k lam : ℕ) :
    pr (fun x : (Store N b × Bits n) × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, τ ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + T))) (x.2.2 i) = x.2.1 (x.2.2 i, x.1.1 (x.2.2 i))) ∧
          (∑ i ∈ Finset.range T, wd (t₀ + i) ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + i)))) + (M + 1) +
            (sidePorts.map fun p => β p.1).sum * T + lam < k * b + n)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (((T * qt).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  obtain ⟨I, H, Sd, hw, hQ, hs, hside⟩ :=
    splits_of_links_joint S β link hα X t₀ T hbmPorts sidePorts orPorts hports wd hpm Q hor
  subst hw hQ
  exact audit_joint S hc X t₀ T M _ qt hM hpre I hq H Sd (hside h0) hs ζ hζ τ s k lam

theorem audit_capacity_joint (S : (Store N b × Bits n) × Oracle N b h → Actual.Run L) (hc : CausalFrom S) {K : Type*}
    (β : K → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (hα : Alphabet S β link) (X : Finset L)
    (t₀ T M qt : ℕ) (hM : ∀ r, (S r).mem X t₀ ≤ M)
    (hpre : ∀ e O O', (S (e, O)).content X t₀ = (S (e, O')).content X t₀) (hbmPorts sidePorts orPorts : List (K × L))
    (hports : ∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X →
      (link r g a, (S r).A.x g) ∈ hbmPorts ++ sidePorts ++ orPorts)
    (hhc : HearsChip S link X t₀ T hbmPorts)
    (Q : ℕ → Acts L → List (Fin N × Bits b)) (hq : ∀ u a, (Q u a).length ≤ qt)
    (hor : OracleLocation S link X t₀ T orPorts Q)
    (h0 : ∀ e O O' u, portSignals S link sidePorts (e, O) u = portSignals S link sidePorts (e, O') u)
    (ζ : List Bool → List (Cross L) → Bits n)
    (hζ : ∀ E z O, ζ ((S ((E, z), O)).content X t₀)
      ((List.range T).map fun i => (S ((E, z), O)).crossing (↑X)ᶜ ↑X {t₀ + i}) = z)
    (τ : Acts L → Fin N → Bits h) (s k lam : ℕ) :
    pr (fun x : (Store N b × Bits n) × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, τ ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + T))) (x.2.2 i) = x.2.1 (x.2.2 i, x.1.1 (x.2.2 i))) ∧
          (hbmPorts.map fun p => β p.1).sum * T + (M + 1) + (sidePorts.map fun p => β p.1).sum * T + lam <
            k * b + n)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (((T * qt).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  refine (pr_mono fun x hx => ⟨hx.1, ?_⟩).trans (audit_links_joint S hc β link hα X t₀ T M qt hM hpre hbmPorts
    sidePorts orPorts hports _ (passiveMemory_of_hearsChip S β link hα X t₀ T hbmPorts hhc) Q hq hor h0 ζ hζ τ s k lam)
  simp only [Finset.sum_const, Finset.card_range, smul_eq_mul]
  have := hx.2
  nlinarith

end SecurityProofs.Causal

namespace SecurityProofs

/-- **Mbw on a strategy's runs, from the links** (`Bandwidth.Guarantees.Mbw`). -/
theorem Mbw : Guarantees.Mbw := by
  intro L K _ N b h n S hc β link hα X t₀ T M qt hM hpre hbmPorts sidePorts orPorts hports wd hpm Q hq hor h0 ζ hζ τ
    s k lam
  exact Causal.audit_links_joint S hc β link hα X t₀ T M qt hM hpre hbmPorts sidePorts orPorts hports wd hpm Q hq hor
    h0 ζ hζ τ s k lam

/-- **Mbw in the capacity form** (`Bandwidth.Guarantees.MbwCapacity`). -/
theorem MbwCapacity : Guarantees.MbwCapacity := by
  intro L K _ N b h n S hc β link hα X t₀ T M qt hM hpre hbmPorts sidePorts orPorts hports hhc Q hq hor h0 ζ hζ τ s k
    lam
  exact Causal.audit_capacity_joint S hc β link hα X t₀ T M qt hM hpre hbmPorts sidePorts orPorts hports hhc Q hq hor
    h0 ζ hζ τ s k lam

end SecurityProofs

end Bandwidth
