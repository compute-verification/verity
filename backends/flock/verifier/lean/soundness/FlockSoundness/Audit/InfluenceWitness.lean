import FlockSoundness.Audit.Influence
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Positivity

/-!
# The influence theorems are not vacuous

The statement reviewer's witness (bc-89770364, Phase 19g, X-IC-1): every hypothesis of the influence theorems can be met,
and the conclusions are not trivial.

* `W.lemma47` (`outputs_eq_of_separates`): a four-gate circuit, input `0`, gates `1 = ¬0` and `3 = ¬0` in unit `0`, gate
  `2 = ¬1` in unit `1`, outputs `{2, 3}`, delivered wires `D = {2}`. `K = {1}` separates `B = {0}` from `D`, and is
  narrower than the exits `{1, 3}` (`W.three_mem_exits`). Two different transcripts that both get gate `1` wrong deliver
  the same value on `D`. `W.count` is the count `card_delivered_le` gives.
* `Gen.harm_witness`: the hypotheses of `card_influenceSet_le_harm` hold on any partition and law, with the exits as
  separators and `hb u = card (exits {u})` (`Gen.exits_card_le`).
* `A.influence`, `A.accepts`: a law that draws nothing, a session that always accepts and the honest committed
  transcript meet `KnowledgeSound 0`, `LinkSound 0` and `AnchorsSound 0`, so `audit_influence` applies, and the audit
  accepts with probability `1`.
-/

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit Finset
open scoped ENNReal

namespace FlockSoundness.Audit.InfluenceWitness

namespace W

abbrev ops : Fin 4 → Op 4
  | 0 => .input
  | 1 => .not 0
  | 2 => .not 1
  | 3 => .not 0

abbrev C : Circuit where
  N := 4
  op := ops
  topo := by decide
  outputs := {2, 3}

abbrev unitOf : Fin 4 → Option (Fin 2) := ![none, some 0, some 1, some 0]

abbrev P : Partition C 2 where
  unit := unitOf
  input_iff := by decide

def D : Finset (Fin C.N) := {2}
def B : Finset (Fin 2) := {0}
def K : Finset (Fin C.N) := {1}
def a : Fin C.N → Bool := fun _ => false

theorem hK : K ⊆ P.committed := by
  intro g hg
  simp only [K, mem_singleton] at hg
  subst hg
  exact P.mem_committed_of_read (h := 2) (by decide) (by decide)

theorem hD : D ⊆ P.committed := by
  intro g hg
  simp only [D, mem_singleton] at hg
  subst hg
  exact P.mem_committed_of_output (by decide)

theorem no_reader_3 : ∀ h : Fin C.N, (3 : Fin C.N) ∉ (C.op h).args := by decide

theorem hsep : P.Separates K B D := by
  intro g t u hu huB ht hgt
  simp only [D, mem_singleton] at ht
  simp only [B, mem_singleton] at huB
  subst ht huB
  have hg : g = 1 ∨ g = 3 := by
    clear hgt
    revert hu
    revert g
    decide
  rcases hg with rfl | rfl
  · exact hgt.not_mem (by simp [K])
  · have key : ∀ g t, C.FeedsAvoiding K g t → g = 3 → t = 2 → False := by
      intro g t h
      cases h with
      | refl _ => intro h1 h2; exact absurd (h1.symm.trans h2) (by decide)
      | step _ hx _ => intro h1 _; subst h1; exact no_reader_3 _ hx
    exact key _ _ hgt rfl rfl

/-- The exits of `B` hold gate `3`, which `K` does not: `K` is a narrower separator than the exits. -/
theorem three_mem_exits : (3 : Fin C.N) ∈ P.exits B ∧ (3 : Fin C.N) ∉ K := by
  refine ⟨?_, by simp [K]⟩
  simp only [Partition.exits, mem_filter]
  exact ⟨P.mem_committed_of_output (by decide), 0, by simp [B], by decide⟩

theorem inputs (Z : Fin C.N → Bool) (h0 : Z 0 = false) : C.InputsAgree Z a := by
  intro g hg
  have : g = 0 := by
    revert hg
    revert g
    decide
  subst this
  simpa [a] using h0

theorem correct1 (Z : Fin C.N → Bool) (h1 : Z 1 = false) (h2 : Z 2 = true) : P.Correct Z 1 := by
  intro g hgc hg
  clear hgc
  have : g = 2 := by
    revert hg
    revert g
    decide
  subst this
  rw [P.localEval_of_mem Z (show P.unit 2 = some 1 by decide)]
  show Z 2 = !(P.localEval Z 1 1)
  rw [P.localEval_of_not_mem Z (show P.unit 1 ≠ some 1 by decide), h1, h2]
  rfl

theorem wrong0 (Z : Fin C.N → Bool) (h0 : Z 0 = false) (h1 : Z 1 = false) : (0 : Fin 2) ∈ P.wrong Z := by
  rw [P.mem_wrong]
  intro hc
  have h := hc 1 (hK (by simp [K])) (by decide)
  rw [P.localEval_of_mem Z (show P.unit 1 = some 0 by decide)] at h
  change Z 1 = !(P.localEval Z 0 0) at h
  rw [P.localEval_of_not_mem Z (show P.unit 0 ≠ some 0 by decide), h0, h1] at h
  exact absurd h (by decide)

theorem wrong_sub (Z : Fin C.N → Bool) (h1 : Z 1 = false) (h2 : Z 2 = true) : P.wrong Z ⊆ B := by
  intro u hu
  rw [P.mem_wrong] at hu
  fin_cases u
  · simp [B]
  · exact absurd (correct1 Z h1 h2) hu

def X : Fin C.N → Bool := ![false, false, true, false]
def Y : Fin C.N → Bool := ![false, false, true, true]

/-- **Lemma 4.7 is not vacuous**: two different transcripts, each with a wrong unit, agree on the delivered wire. -/
theorem lemma47 : X 2 = Y 2 ∧ X ≠ Y ∧ P.wrong X ≠ ∅ := by
  refine ⟨P.outputs_eq_of_separates hK hD hsep (inputs X rfl) (inputs Y rfl) (wrong_sub X rfl rfl)
    (wrong_sub Y rfl rfl) ?_ 2 (by simp [D]), ?_, ?_⟩
  · intro g hg
    simp only [K, mem_singleton] at hg
    subst hg
    rfl
  · intro h
    exact absurd (congrFun h 3) (by decide)
  · exact ne_empty_of_mem (wrong0 X rfl rfl)

theorem count : (P.delivered D a B).card ≤ 2 :=
  (P.card_delivered_le hK hD hsep a).trans (by simp [K])

end W

namespace Gen

variable {C : Circuit} {n : ℕ}

/-- The exits of a set are the union of its units' exits. -/
theorem exits_card_le (P : Partition C n) (B : Finset (Fin n)) :
    (P.exits B).card ≤ ∑ u ∈ B, (P.exits {u}).card := by
  classical
  calc (P.exits B).card ≤ (B.biUnion fun u => P.exits {u}).card := by
        refine card_le_card fun g hg => ?_
        simp only [Partition.exits, mem_filter] at hg
        obtain ⟨hgc, u, hu, hgu⟩ := hg
        refine mem_biUnion.2 ⟨u, hu, ?_⟩
        simp only [Partition.exits, mem_filter, mem_singleton]
        exact ⟨hgc, u, rfl, hgu⟩
    _ ≤ ∑ u ∈ B, (P.exits {u}).card := card_biUnion_le

/-- **The harm form is not vacuous**: on any partition and law, the exits and their per-unit widths meet every
hypothesis of `card_influenceSet_le_harm`. -/
theorem harm_witness (P : Partition C n) {D : Finset (Fin C.N)} (hD : D ⊆ P.committed) (L : Law n)
    (a : Fin C.N → Bool) (δ : ℝ≥0∞) :
    (P.influenceSet D L.escape a δ).card ≤
      2 ^ (∑ u, (P.exits {u}).card) * (univ.filter fun B => δ ≤ L.escape B).card :=
  P.card_influenceSet_le_harm hD (fun u => (P.exits {u}).card) P.exits (fun _ => filter_subset _ _)
    (fun B => P.separates_exits B hD) (exits_card_le P) (δ := δ) (H := ∑ u, (P.exits {u}).card)
    (fun B _ => by
      push_cast
      exact sum_le_sum_of_subset_of_nonneg (subset_univ _) fun _ _ _ => by positivity)

end Gen

namespace A

open W

def L : Law 2 := { Ω := Unit, draw := fun _ => ∅ }

def session : Finset (Fin 2) → Unit → Game Bool := fun _ _ => .ret true

/-- The honest transcript on the anchors `a`. -/
def H : Fin C.N → Bool := ![false, true, false, true]

def Xc : (R : Unit) → Cont L Unit session R → (Fin C.N → Bool) := fun _ _ => H

def ext : (S : Finset (Fin 2)) → (R : Unit) → Strategy (session S R) → Fin 2 → Option (Fin C.N → Bool) :=
  fun _ _ _ _ => none

theorem hks : (P.analysis Xc ext).KnowledgeSound 0 := by
  intro R τ tgt
  refine (avg_mono fun ω => ?_).trans (avg_const (0 : ℝ≥0∞)).le
  refine (prob_mono (q := fun _ => False) ?_ _ _).trans (prob_false _ _).le
  rintro ok ⟨_, u, _, hu, _⟩
  exact notMem_empty u hu

theorem hlink : (P.analysis Xc ext).LinkSound 0 := by
  intro R τ tgt
  refine (avg_mono fun ω => ?_).trans (avg_const (0 : ℝ≥0∞)).le
  refine (prob_mono (q := fun _ => False) ?_ _ _).trans (prob_false _ _).le
  rintro ok ⟨_, u, _, hu, _⟩
  exact notMem_empty u hu

theorem hin : Partition.AnchorsSound Xc a 0 := by
  intro σ
  refine (prob_mono (q := fun _ => False) ?_ _ _).trans (prob_false _ _).le
  rintro o ⟨_, hno⟩
  exact hno (inputs H rfl)

/-- **`audit_influence` is not vacuous**: its hypotheses hold at `0`. -/
theorem influence (δ : ℝ≥0∞) (σ : Strategy (audit L Unit session)) :
    prob (fun o => o.1 = true ∧ Partition.out D (Xc (reg σ) (cont σ)) ∉ P.influenceSet D L.escape a δ)
      (audit L Unit session) σ ≤ δ + 0 + 0 + 0 :=
  P.audit_influence hks hlink hin δ σ

/-- The audit accepts with probability `1`. -/
theorem accepts (σ : Strategy (audit L Unit session)) :
    prob (fun o => o.1 = true) (audit L Unit session) σ = 1 := by
  simp [audit, prob, session, Game.map, L]

end A

end FlockSoundness.Audit.InfluenceWitness
