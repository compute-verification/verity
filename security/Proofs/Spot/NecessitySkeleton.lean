module

public import Definitions.Spot.Necessity
public import Mathlib.Data.Fintype.Option
public import Mathlib.Data.Fintype.Prod
public import Proofs.Core.Coins
public import Proofs.Spot.Egress
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Linarith.Frontend
import Mathlib.Tactic.NormNum.Abs
import Mathlib.Tactic.NormNum.DivMod
import Mathlib.Tactic.NormNum.OfScientific
import Proofs.Spot.Determinacy
import Proofs.Spot.Leakage

/-! The skeleton theorem: tier 1 gives the sampling certificate, and each tier-1 item is needed. -/

@[expose] public section

namespace Population.Iop

open Finset
open Verity.Game (iid_toReal)

variable {U T : Type}

section Prob

variable [Fintype U] (π : PMF U) (k : ℕ)

theorem prob_mono {E E' : (Fin k → U) → Prop} (h : ∀ d, E d → E' d) : prob π k E ≤ prob π k E' :=
  Verity.Game.prPMFReal_mono _ h

theorem prob_false : prob π k (fun _ => False) = 0 :=
  Verity.Game.prPMFReal_false _

theorem prob_congr {E E' : (Fin k → U) → Prop} (h : ∀ d, E d ↔ E' d) : prob π k E = prob π k E' :=
  Verity.Game.prPMFReal_congr _ h

theorem prob_eq_sum (E : (Fin k → U) → Prop) [DecidablePred E] :
    prob π k E = ∑ d : Fin k → U, if E d then ∏ i, (π (d i)).toReal else 0 := by
  refine (Verity.Game.prPMFReal_eq_sum _ E).trans (sum_congr rfl fun d _ => ?_)
  rw [iid_toReal]

theorem mass_eq_sum (B : Finset U) : mass π B = ∑ u ∈ B, (π u).toReal :=
  Verity.Game.prPMFReal_mem π B

theorem prob_all (g : U → Prop) [DecidablePred g] :
    prob π k (fun d => ∀ i, g (d i)) = (∑ u ∈ univ.filter g, (π u).toReal) ^ k := by
  classical
  rw [sum_filter, Fintype.sum_pow, prob_eq_sum]
  refine sum_congr rfl fun d _ => ?_
  rw [prod_ite_zero]
  simp

theorem mem_bad {chk : U → T → Prop} {σ : T} {u : U} : u ∈ bad chk σ ↔ ¬ chk u σ := by
  classical
  simp [bad]

theorem disjoint_bad_iff (chk : U → T → Prop) (A : Finset U) (σ : T) :
    Disjoint (bad chk σ) A ↔ ∀ u ∈ A, chk u σ := by
  classical
  simp only [bad, disjoint_left, mem_filter, mem_univ, true_and]
  exact ⟨fun h u hu => by_contra fun hc => h hc hu, fun h u hc hu => hc (h u hu)⟩

theorem mass_bad_le_one (chk : U → T → Prop) (σ : T) : mass π (bad chk σ) ≤ 1 :=
  Verity.Game.prPMFReal_le_one _ _

open Classical in
theorem sum_good (chk : U → T → Prop) (σ : T) :
    ∑ u ∈ univ.filter (fun u => chk u σ), (π u).toReal = 1 - mass π (bad chk σ) := by
  have h := sum_filter_add_sum_filter_not univ (fun u => chk u σ) fun u => (π u).toReal
  rw [Verity.Game.sum_toReal_pmf] at h
  have : mass π (bad chk σ) = ∑ u ∈ univ.filter (fun u => ¬ chk u σ), (π u).toReal := mass_eq_sum π _
  rw [this]
  linarith

theorem prob_accepts (chk : U → T → Prop) (A : Finset U) (σ : T) (hd : Disjoint (bad chk σ) A) :
    prob π k (fun d => Accepts chk A d σ) = (1 - mass π (bad chk σ)) ^ k := by
  classical
  have hA := (disjoint_bad_iff chk A σ).1 hd
  rw [← sum_good π chk σ, ← prob_all]
  exact prob_congr π k fun d => ⟨fun h => h.2, fun h => ⟨hA, h⟩⟩

theorem prob_accepts_eq_zero (chk : U → T → Prop) (A : Finset U) (σ : T) (hd : ¬ Disjoint (bad chk σ) A) :
    prob π k (fun d => Accepts chk A d σ) = 0 := by
  have hA : ¬ ∀ u ∈ A, chk u σ := fun h => hd ((disjoint_bad_iff chk A σ).2 h)
  rw [← prob_false π k]
  exact prob_congr π k fun d => by simp only [Accepts, iff_false, not_and]; exact fun h => absurd h hA

/-- A randomized prover is a mixture of such transcripts, so the bound holds for it too. -/
theorem certificate (chk : U → T → Prop) (A : Finset U) {ε : ℝ} (hε : ε ≤ 1) (σ : T) :
    prob π k (fun d => Accepts chk A d σ ∧ ¬ Certified chk π A ε σ) ≤ (1 - ε) ^ k := by
  have h1e : 0 ≤ 1 - ε := by linarith
  by_cases hc : Certified chk π A ε σ
  · rw [prob_congr π k (E' := fun _ => False) fun d => by simp [hc], prob_false]
    exact pow_nonneg h1e k
  · refine (prob_mono π k fun d h => h.1).trans ?_
    by_cases hd : Disjoint (bad chk σ) A
    · rw [prob_accepts π k chk A σ hd]
      have hm : ε < mass π (bad chk σ) := lt_of_not_ge fun h => hc ⟨hd, h⟩
      exact pow_le_pow_left₀ (by linarith [mass_bad_le_one π chk σ]) (by linarith) k
    · rw [prob_accepts_eq_zero π k chk A σ hd]
      exact pow_nonneg h1e k

theorem inference_iff (chk : U → T → Prop) (A : Finset U) (dmg : T → ℝ) (g δ : ℝ)
    (hδ : 0 ≤ δ) :
    (∀ σ, prob π k (fun d => Accepts chk A d σ ∧ g < dmg σ) ≤ δ) ↔
      ∀ σ, Disjoint (bad chk σ) A → g < dmg σ → (1 - mass π (bad chk σ)) ^ k ≤ δ := by
  constructor
  · intro h σ hd hg
    have := h σ
    rwa [prob_congr π k (E' := fun d => Accepts chk A d σ) fun d => by simp [hg],
      prob_accepts π k chk A σ hd] at this
  · intro h σ
    by_cases hg : g < dmg σ
    · rw [prob_congr π k (E' := fun d => Accepts chk A d σ) fun d => by simp [hg]]
      by_cases hd : Disjoint (bad chk σ) A
      · rw [prob_accepts π k chk A σ hd]
        exact h σ hd hg
      · rw [prob_accepts_eq_zero π k chk A σ hd]
        exact hδ
    · rw [prob_congr π k (E' := fun _ => False) fun d => by simp [hg], prob_false]
      exact hδ

theorem inferable_of_amp {chk : U → T → Prop} {A : Finset U} {c : U → ℝ} {dmg : T → ℝ} {K C : ℝ}
    (hK : 0 ≤ K) (hC : 0 ≤ C) (hamp : AmpLe chk A c dmg K) (hc : ∀ u, c u ≤ C * (π u).toReal) {ε : ℝ}
    (hε : ε ≤ 1) (σ : T) : prob π k (fun d => Accepts chk A d σ ∧ K * C * ε < dmg σ) ≤ (1 - ε) ^ k := by
  refine (prob_mono π k fun d h => ⟨h.1, fun hcert => ?_⟩).trans (certificate π k chk A hε σ)
  have h1 : ∑ u ∈ bad chk σ, c u ≤ C * mass π (bad chk σ) := by
    rw [mass_eq_sum, mul_sum]
    exact sum_le_sum fun u _ => hc u
  have h2 : dmg σ ≤ K * C * ε := by
    calc dmg σ ≤ K * ∑ u ∈ bad chk σ, c u := hamp σ hcert.1
      _ ≤ K * (C * mass π (bad chk σ)) := mul_le_mul_of_nonneg_left h1 hK
      _ ≤ K * (C * ε) := mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hcert.2 hC) hK
      _ = K * C * ε := (mul_assoc _ _ _).symm
  exact absurd h.2 (not_lt.2 h2)

theorem not_inferable_of_massless (chk : U → T → Prop) (A : Finset U) (σ : T)
    (hd : Disjoint (bad chk σ) A) (h0 : mass π (bad chk σ) = 0) :
    prob π k (fun d => Accepts chk A d σ) = 1 := by
  rw [prob_accepts π k chk A σ hd, h0]
  simp

end Prob

theorem accepts_local {Q V : Type} {reads : U → Finset Q} {chk : U → (Q → V) → Prop} (hl : Local reads chk)
    (A : Finset U) {k : ℕ} (d : Fin k → U) {σ τ : Q → V}
    (h : ∀ u, (u ∈ A ∨ ∃ i, d i = u) → ∀ q ∈ reads u, σ q = τ q) :
    Accepts chk A d σ ↔ Accepts chk A d τ := by
  unfold Accepts
  constructor
  · rintro ⟨hA, hd⟩
    exact ⟨fun u hu => (hl u σ τ (h u (Or.inl hu))).1 (hA u hu),
      fun i => (hl _ σ τ (h _ (Or.inr ⟨i, rfl⟩))).1 (hd i)⟩
  · rintro ⟨hA, hd⟩
    exact ⟨fun u hu => (hl u σ τ (h u (Or.inl hu))).2 (hA u hu),
      fun i => (hl _ σ τ (h _ (Or.inr ⟨i, rfl⟩))).2 (hd i)⟩

theorem untestable_of_dense {Q V : Type} (G : (Q → V) → Prop) (R : Finset Q) (dec : (R → V) → Prop)
    (hcomp : ∀ τ : Q → V, G τ → dec fun r : R => τ r) (hdense : ∀ σ : Q → V, ∃ τ, G τ ∧ ∀ q ∈ R, τ q = σ q)
    (σ : Q → V) : dec fun r : R => σ r := by
  obtain ⟨τ, hG, hτ⟩ := hdense σ
  have : (fun r : R => τ r) = fun r : R => σ r := funext fun r => hτ r r.2
  exact this ▸ hcomp τ hG

def checksum {n : ℕ} {V : Type} [AddCommGroup V] (σ : Fin n → V) : Prop := ∑ i, σ i = 0

theorem checksum_dense {n : ℕ} {V : Type} [AddCommGroup V] (R : Finset (Fin n)) (hR : R.card < n)
    (σ : Fin n → V) : ∃ τ, checksum τ ∧ ∀ q ∈ R, τ q = σ q := by
  classical
  obtain ⟨j, hj⟩ : ∃ j, j ∉ R := by
    by_contra h
    push Not at h
    have := card_le_card fun x (_ : x ∈ (univ : Finset (Fin n))) => h x
    simp at this
    omega
  refine ⟨Function.update σ j (σ j - ∑ i, σ i), ?_,
    fun q hq => Function.update_of_ne (by rintro rfl; exact hj hq) _ _⟩
  unfold checksum
  have hsplit := sum_sdiff (f := σ) (singleton_subset_iff.2 (mem_univ j))
  rw [sum_singleton] at hsplit
  rw [sum_update_of_mem (mem_univ j), ← hsplit]
  abel

theorem checksum_untestable {n : ℕ} {V : Type} [AddCommGroup V] (R : Finset (Fin n)) (hR : R.card < n)
    (dec : (R → V) → Prop) (hcomp : ∀ τ : Fin n → V, checksum τ → dec fun r : R => τ r) (σ : Fin n → V) :
    dec fun r : R => σ r :=
  untestable_of_dense checksum R dec hcomp (checksum_dense R hR) σ

theorem adaptive_dodges [Fintype U] [DecidableEq U] {chk : U → T → Prop} (hfree : ∀ B : Finset U, ∃ σ, bad chk σ = B)
    (A : Finset U) (k : ℕ) :
    ∃ σ : (Fin k → U) → T, ∀ d, Accepts chk A d (σ d) ∧ bad chk (σ d) = (A ∪ univ.image d)ᶜ := by
  choose g hg using hfree
  have pass : ∀ (B : Finset U) u, u ∉ B → chk u (g B) := by
    intro B u hu
    classical
    have : u ∉ bad chk (g B) := by rw [hg]; exact hu
    simpa [bad] using this
  exact ⟨fun d => g (A ∪ univ.image d)ᶜ, fun d =>
    ⟨⟨fun u hu => pass _ u (by simp [hu]), fun i => pass _ _ (by simp)⟩, hg _⟩⟩

theorem adaptive_card [Fintype U] [DecidableEq U] (A : Finset U) {k : ℕ} (d : Fin k → U) :
    Fintype.card U ≤ ((A ∪ univ.image d)ᶜ).card + A.card + k := by
  rw [card_compl]
  have h1 := card_union_le A (univ.image d)
  have h2 : (univ.image d).card ≤ k := card_image_le.trans (by simp)
  have h3 := card_le_univ (A ∪ univ.image d)
  omega

theorem late_sample_dodges [Fintype U] [DecidableEq U] (chk : U → T → Prop) (A : Finset U) (σ : T) {u₀ : U}
    (h₀ : chk u₀ σ) (hA : ∀ u ∈ A, chk u σ) (k : ℕ) :
    prob (PMF.pure u₀) k (fun d => Accepts chk A d σ) = 1 := by
  refine not_inferable_of_massless _ k chk A σ ((disjoint_bad_iff chk A σ).2 hA) ?_
  classical
  rw [mass_eq_sum]
  refine sum_eq_zero fun u hu => ?_
  have hu' : ¬ chk u σ := by simpa [bad] using hu
  have : u ≠ u₀ := fun e => hu' (e ▸ h₀)
  simp [this]

theorem flags_free (n : ℕ) (B : Finset (Fin n)) :
    ∃ σ, bad (corr (Dataflow.flags n) (Dataflow.flagsFns n)) σ = B := by
  classical
  refine ⟨fun s => decide (s ∈ B), ?_⟩
  have key : ∀ u, corr (Dataflow.flags n) (Dataflow.flagsFns n) u (fun s => decide (s ∈ B)) ↔ u ∉ B := by
    intro u
    constructor
    · intro h hu
      have := h ⟨u, by simp [Dataflow.flags]⟩
      simp [Dataflow.flagsFns, hu] at this
    · intro hu ⟨s, hs⟩
      have : s = u := by simpa [Dataflow.flags] using hs
      subst this
      simp [Dataflow.flagsFns, hu]
  ext u
  rw [mem_bad, key, not_not]

section Free

/-- Acyclic, and no unit writes an input; only coverage fails. -/
def freeVals (k : ℕ) : Dataflow (Fin k) (Fin k ⊕ Fin k) where
  input := ∅
  reads i := {Sum.inl i}
  writes i := {Sum.inr i}

def freeFns (k : ℕ) {V : Type} : (freeVals k).Fns V := fun i r _ => r ⟨Sum.inl i, mem_singleton_self _⟩

theorem freeVals_acyclic (k : ℕ) : (freeVals k).Acyclic := by
  intro u h
  obtain ⟨b, hb⟩ := Relation.TransGen.head'_iff.1 h
  obtain ⟨s, hs, hs'⟩ := hb.1
  simp [freeVals] at hs hs'
  rw [hs] at hs'
  exact Sum.inr_ne_inl hs'

theorem freeVals_not_covered {k : ℕ} (hk : 0 < k) : ¬ (freeVals k).Covered := by
  intro h
  obtain ⟨u, hu⟩ := h (Sum.inl ⟨0, hk⟩) (by simp [freeVals])
  simp [freeVals] at hu

theorem free_value (k : ℕ) {V : Type} (y : Fin k → V) :
    ∃ σ : Fin k ⊕ Fin k → V, (∀ u, (freeVals k).Correct (freeFns k) σ u) ∧ ∀ i, σ (Sum.inr i) = y i := by
  refine ⟨Sum.elim y y, fun u ⟨s, hs⟩ => ?_, fun i => rfl⟩
  have : s = Sum.inr u := by simpa [freeVals] using hs
  subst this
  rfl

end Free

section Gate

abbrev Gx (k : ℕ) := Option (Bool × Fin k)

def gate (k : ℕ) : Dataflow (Gx k) (Gx k) where
  input := ∅
  reads
    | none => univ
    | some (b, i) => {some (!b, i)}
  writes u := {u}

def gateFns (k : ℕ) : (gate k).Fns Bool
  | none => fun r _ =>
      if ∃ i, r ⟨some (false, i), mem_univ _⟩ = true then !(r ⟨none, mem_univ _⟩) else false
  | some (b, i) => fun r _ => r ⟨some (!b, i), mem_singleton_self _⟩

theorem gate_correct_some {k : ℕ} {σ : Gx k → Bool} {b : Bool} {i : Fin k} :
    (gate k).Correct (gateFns k) σ (some (b, i)) ↔ σ (some (b, i)) = σ (some (!b, i)) := by
  constructor
  · intro h
    exact h ⟨some (b, i), mem_singleton_self _⟩
  · intro h ⟨s, hs⟩
    have : s = some (b, i) := by simpa [gate] using hs
    subst this
    exact h

theorem gate_correct_none {k : ℕ} {σ : Gx k → Bool} :
    (gate k).Correct (gateFns k) σ none ↔
      σ none = if ∃ i, σ (some (false, i)) = true then !(σ none) else false := by
  constructor
  · intro h
    exact h ⟨none, mem_singleton_self _⟩
  · intro h ⟨s, hs⟩
    have : s = none := by simpa [gate] using hs
    subst this
    exact h

theorem gate_determinate (k : ℕ) (a : Gx k → Bool) : DeterminateAt (gate k) (gateFns k) a := by
  refine ⟨fun _ => false, ⟨fun s hs => by simp [gate] at hs, fun u => ?_⟩, ?_⟩
  · rcases u with _ | ⟨b, i⟩
    · rw [gate_correct_none]
      simp
    · rw [gate_correct_some]
  · rintro σ ⟨-, hσ⟩
    have hyz : ∀ b i, σ (some (b, i)) = σ (some (!b, i)) := fun b i => gate_correct_some.1 (hσ _)
    have ht := gate_correct_none.1 (hσ none)
    have hy : ∀ i, σ (some (false, i)) = false := by
      intro i
      by_contra hi
      have hex : ∃ i, σ (some (false, i)) = true := ⟨i, by simpa using hi⟩
      rw [ite_eq_left hex] at ht
      cases h : σ none <;> simp [h] at ht
    have hnone : σ none = false := by
      rw [ht, ite_eq_right]
      simp [hy]
    funext s
    rcases s with _ | ⟨b, i⟩
    · exact hnone
    · cases b
      · exact hy i
      · rw [hyz true i]
        exact hy i

theorem gate_free (k : ℕ) (y : Fin k → Bool) :
    ∃ σ : Gx k → Bool, (gate k).incorrect (gateFns k) σ ⊆ {none} ∧ ∀ i, σ (some (false, i)) = y i := by
  let σ : Gx k → Bool := fun s => match s with
    | none => false
    | some (_, i) => y i
  refine ⟨σ, fun u hu => ?_, fun i => rfl⟩
  rcases u with _ | ⟨b, i⟩
  · exact mem_singleton_self _
  · exfalso
    classical
    have hu' : ¬ (gate k).Correct (gateFns k) σ (some (b, i)) := by simpa [Dataflow.incorrect] using hu
    exact hu' (gate_correct_some.2 rfl)

def gateOut (k : ℕ) : Finset (Gx k) := univ.image fun i => some (false, i)

theorem gate_coneEgress (k : ℕ) : (gate k).coneEgress (gateOut k) {none} = ∅ := by
  classical
  have hdown : ∀ v, Relation.ReflTransGen (gate k).Dep none v → v = none := by
    intro v h
    induction h with
    | refl => rfl
    | tail _ hd ih =>
      subst ih
      obtain ⟨s, hs, hs'⟩ := hd
      have : s = none := by simpa [gate] using hs
      subst this
      rename_i c _
      rcases c with _ | ⟨b, i⟩
      · rfl
      · simp [gate] at hs'
  ext s
  simp only [Dataflow.coneEgress, mem_inter, Dataflow.writesOf, mem_biUnion, notMem_empty, iff_false, not_and]
  intro hs ⟨v, hv, hsv⟩
  have hv' := hdown v (((gate k).mem_down).1 hv |>.elim fun u ⟨hu, h⟩ => by
    rw [mem_singleton] at hu; exact hu ▸ h)
  subst hv'
  have : s = none := by simpa [gate] using hsv
  subst this
  simp [gateOut] at hs

/-- Determinacy at fixed functions, without robustness, bounds neither integrity nor control. -/
theorem gate_egress (k : ℕ) (a : Gx k → Bool) :
    2 ^ k ≤ ((gate k).egress (gateFns k) a ∅ (gateOut k) {none}).card := by
  classical
  let φ : (Fin k → Bool) → (↥(gateOut k) → Bool) := fun y o => match o.1 with
    | none => false
    | some (_, i) => y i
  have hmem : ∀ y, φ y ∈ (gate k).egress (gateFns k) a ∅ (gateOut k) {none} := by
    intro y
    obtain ⟨σ, hw, hσ⟩ := gate_free k y
    simp only [Dataflow.egress, mem_image, mem_filter, mem_univ, true_and]
    refine ⟨σ, ⟨fun s hs => by simp [gate] at hs, hw⟩, funext fun o => ?_⟩
    obtain ⟨o, ho⟩ := o
    obtain ⟨i, -, rfl⟩ := mem_image.1 ho
    exact hσ i
  have hinj : Function.Injective φ := by
    intro y y' h
    funext i
    exact congrFun h ⟨some (false, i), mem_image.2 ⟨i, mem_univ _, rfl⟩⟩
  have := card_le_card_of_injOn φ (s := univ) (fun y _ => hmem y) hinj.injOn
  simpa using this

theorem gate_singleAssignment (k : ℕ) : (gate k).SingleAssignment :=
  fun s => ⟨fun h => by simp [gate] at h, fun _ => ⟨s, by simp [gate], fun u hu => by simpa [gate, eq_comm] using hu⟩⟩

end Gate

section Skeleton

variable {S : Type} (P : Dataflow U S)

theorem bad_corr [Fintype U] {V : Type} (f : P.Fns V) (σ : S → V) : bad (corr P f) σ = P.incorrect f σ := rfl

theorem damaged_sub_coneEgress [Fintype U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V) {honest σ : S → V}
    (hh : ∀ s ∈ P.input, honest s = a s) (hc : ∀ u, P.Correct f honest u) (hσ : ∀ s ∈ P.input, σ s = a s)
    (O : Finset S) : damaged O σ honest ⊆ P.coneEgress O (P.incorrect f σ) := by
  classical
  intro o ho
  simp only [damaged, mem_filter] at ho
  refine (P.mem_coneEgress).2 ⟨ho.1, ?_⟩
  by_contra hno
  exact ho.2 (P.eq_off_cone (P.covered_of_singleAssignment h.1) h.2 f
    (fun t ht => (hσ t ht).trans (hh t ht).symm) le_rfl (by simp [P.incorrect_eq_empty hc]) hno)

theorem card_damaged_le [Fintype U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V) {honest σ : S → V}
    (hh : ∀ s ∈ P.input, honest s = a s) (hc : ∀ u, P.Correct f honest u) (hσ : ∀ s ∈ P.input, σ s = a s)
    (O : Finset S) : (damaged O σ honest).card ≤ ∑ u ∈ P.incorrect f σ, (P.coneEgress O {u}).card :=
  (card_le_card (damaged_sub_coneEgress P h f a hh hc hσ O)).trans (P.card_coneEgress_le O _)

theorem skeleton [Fintype U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V) {honest : S → V}
    (hh : ∀ s ∈ P.input, honest s = a s) (hc : ∀ u, P.Correct f honest u) {π : PMF U}
    (A : Finset U) (k : ℕ) {ε : ℝ} (hε : ε ≤ 1) (O : Finset S) (σ : S → V) (hσ : ∀ s ∈ P.input, σ s = a s) :
    prob π k (fun d => Accepts (corr P f) A d σ ∧ ¬ (Disjoint (P.incorrect f σ) A ∧ mass π (P.incorrect f σ) ≤ ε ∧
      damaged O σ honest ⊆ P.coneEgress O (P.incorrect f σ))) ≤ (1 - ε) ^ k := by
  refine (prob_mono π k
    (E' := fun d => Accepts (corr P f) A d σ ∧ ¬ Certified (corr P f) π A ε σ)
    fun d hd => ⟨hd.1, fun hcert => hd.2 ⟨hcert.1, hcert.2,
      damaged_sub_coneEgress P h f a hh hc hσ O⟩⟩).trans ?_
  exact certificate π k (corr P f) A hε σ

theorem integrity_inferred [Fintype U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V) {honest : S → V}
    (hh : ∀ s ∈ P.input, honest s = a s) (hc : ∀ u, P.Correct f honest u) {π : PMF U}
    (A : Finset U) (k : ℕ) {ε : ℝ} (hε : ε ≤ 1) (O : Finset S) {C : ℝ} (hC : 0 ≤ C)
    (hE : ∀ u, ((P.coneEgress O {u}).card : ℝ) ≤ C * (π u).toReal) (σ : S → V) (hσ : ∀ s ∈ P.input, σ s = a s) :
    prob π k (fun d => Accepts (corr P f) A d σ ∧ C * ε < (damaged O σ honest).card) ≤ (1 - ε) ^ k := by
  classical
  have hamp : AmpLe (corr P f) A (fun u => ((P.coneEgress O {u}).card : ℝ))
      (fun τ => if ∀ s ∈ P.input, τ s = a s then ((damaged O τ honest).card : ℝ) else 0) 1 := by
    intro τ _
    rw [one_mul, bad_corr]
    dsimp only
    split_ifs with hτ
    · exact_mod_cast card_damaged_le P h f a hh hc hτ O
    · exact sum_nonneg fun _ _ => Nat.cast_nonneg _
  have := inferable_of_amp π k zero_le_one hC hamp hE hε σ
  rw [ite_eq_left hσ, one_mul] at this
  exact this

end Skeleton

end Population.Iop
