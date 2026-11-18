module

public import Proofs.Spot.Sampling.Influence
public import Mathlib.Data.Fin.VecNotation
public import Mathlib.Tactic.FinCases
public import Mathlib.Tactic.Positivity

/-!
# The influence theorems are not vacuous

Every hypothesis of the influence theorems can be met, together. `W.lemma47`'s conclusion is also not trivial; the
others show only that the hypotheses can be met.

* `W.lemma47` (`outputs_eq_of_separates`): a four-gate circuit, input `0`, gates `1 = ¬0` and `3 = ¬0` in unit `0`, gate
  `2 = ¬1` in unit `1`, outputs `{2, 3}`, delivered wires `D = {2}`. `K = {1}` separates `B = {0}` from `D`, and is
  narrower than the exits `{1, 3}` (`W.three_mem_exits`). Two different transcripts that both get gate `1` incorrect deliver
  the same value on `D`. `W.count` is the count `card_delivered_le` gives.
* `Gen.harm_witness`: the hypotheses of `card_influenceSet_le_harm` hold on any partition and sampling design, with the exits as
  separators and `hb u = card (exits {u})` (`Gen.exits_card_le`).
* `A.influence`, `A.accepts`: a sampling design that samples nothing, a session that always accepts and the honest committed
  transcript meet `KnowledgeSound 0`, `BindSound 0` and `AnchorsSound 0`, so `audit_influence` applies, and the audit
  accepts with probability `1`.
* `Ex.exfil`: `audit_exfiltration`'s hypotheses hold together, for a one-bit message, with `A`'s audit at `0`, `Gen`'s
  exits, the harm bound `Ex.hH` (each unit's exported width), and `k = n = 2`, where `miss 3 = 0 < δ` (`Ex.hk`).
* `N`: a sampling design that samples every unit, a session that always accepts, and a prover that registers the input
  bit `m` and commits `W`'s transcript on it. Every unit is correct and extraction returns the committed transcript, so
  `KnowledgeSound 0` and `BindSound 0` hold; the inputs are not the anchors', so `AnchorsSound 0` fails, and both bounds
  fail: the delivered value lies outside the influence set, and a one-bit message gets through with probability `1`.
* `Partition.audit_influence_honest`, `Partition.audit_exfiltration_honest`: the premises at `A` and `Ex`;
  `Partition.audit_influence_nontrivial`, `Partition.audit_exfiltration_nontrivial`: at `N`.
-/

@[expose] public section

open FlockSoundness FlockSoundness.Game Spot.Sampling Finset
open scoped ENNReal

namespace Spot.Sampling.InfluenceWitness

open FlockSoundness

namespace W

abbrev ops : Fin 4 → Verity.Boolean.Op 4
  | 0 => .input
  | 1 => .not 0
  | 2 => .not 1
  | 3 => .not 0

abbrev C : Verity.Boolean.Circuit where
  N := 4
  op := ops
  topo := by decide
  outputs := {2, 3}

abbrev unitOf : Fin 4 → Option (Fin 2) := ![none, some 0, some 1, some 0]

abbrev P : Partition C 2 where
  unit := unitOf
  input_iff := (by decide : ∀ g : Fin C.N, unitOf g = none ↔ (C.op g).isInput = true)

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

theorem incorrect0 (Z : Fin C.N → Bool) (h0 : Z 0 = false) (h1 : Z 1 = false) : (0 : Fin 2) ∈ P.incorrect Z := by
  rw [P.mem_incorrect]
  intro hc
  have h := hc 1 (hK (by simp [K])) (by decide)
  rw [P.localEval_of_mem Z (show P.unit 1 = some 0 by decide)] at h
  change Z 1 = !(P.localEval Z 0 0) at h
  rw [P.localEval_of_not_mem Z (show P.unit 0 ≠ some 0 by decide), h0, h1] at h
  exact absurd h (by decide)

theorem incorrect_sub (Z : Fin C.N → Bool) (h1 : Z 1 = false) (h2 : Z 2 = true) : P.incorrect Z ⊆ B := by
  intro u hu
  rw [P.mem_incorrect] at hu
  fin_cases u
  · simp [B]
  · exact absurd (correct1 Z h1 h2) hu

def X : Fin C.N → Bool := ![false, false, true, false]
def Y : Fin C.N → Bool := ![false, false, true, true]

/-- **Lemma 4.7 is not vacuous**: two different transcripts, each with an incorrect proof block, agree on the delivered wire. -/
theorem lemma47 : X 2 = Y 2 ∧ X ≠ Y ∧ P.incorrect X ≠ ∅ := by
  refine ⟨P.outputs_eq_of_separates hK hD hsep (inputs X rfl) (inputs Y rfl) (incorrect_sub X rfl rfl)
    (incorrect_sub Y rfl rfl) ?_ 2 (by simp [D]), ?_, ?_⟩
  · intro g hg
    simp only [K, mem_singleton] at hg
    subst hg
    rfl
  · intro h
    exact absurd (congrFun h 3) (by decide)
  · exact ne_empty_of_mem (incorrect0 X rfl rfl)

theorem count : (P.delivered D a B).card ≤ 2 :=
  (P.card_delivered_le hK hD hsep a).trans (by simp [K])

end W

namespace Gen

variable {C : Verity.Boolean.Circuit} {n : ℕ}

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

/-- **The harm form is not vacuous**: on any partition and sampling design, the exits and their per-unit widths meet every
hypothesis of `card_influenceSet_le_harm`. -/
theorem harm_witness (P : Partition C n) {D : Finset (Fin C.N)} (hD : D ⊆ P.committed) (L : SamplingDesign n)
    (a : Fin C.N → Bool) (δ : ℝ≥0∞) :
    (P.influenceSet D L.missProb a δ).card ≤
      2 ^ (∑ u, (P.exits {u}).card) * (univ.filter fun B => δ ≤ L.missProb B).card :=
  P.card_influenceSet_le_harm hD (fun u => (P.exits {u}).card) P.exits (fun _ => filter_subset _ _)
    (fun B => P.separates_exits B hD) (exits_card_le P) (δ := δ) (H := ∑ u, (P.exits {u}).card)
    (fun B _ => by
      push_cast
      exact sum_le_sum_of_subset_of_nonneg (subset_univ _) fun _ _ _ => by positivity)

end Gen

namespace A

open W

def L : SamplingDesign 2 := { Ω := Unit, draw := fun _ => ∅ }

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

theorem hlink : (P.analysis Xc ext).BindSound 0 := by
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
    prob (fun o => o.1 = true ∧ Partition.out D (Xc (reg σ) (cont σ)) ∉ P.influenceSet D L.missProb a δ)
      (audit L Unit session) σ ≤ δ + 0 + 0 + 0 :=
  P.audit_influence hks hlink hin δ σ

/-- The audit accepts with probability `1`. -/
theorem accepts (σ : Strategy (audit L Unit session)) :
    prob (fun o => o.1 = true) (audit L Unit session) σ = 1 := by
  simp [audit, prob, session, Game.map, L]

end A

namespace Ex

/-- `hH`: each unit's harm is its exported width, and the harm bound is their total. -/
theorem hH (δ : ℝ≥0∞) :
    IsHarmBound A.L (fun u => ((W.P.exits {u}).card : ℝ)) δ ((∑ u, (W.P.exits {u}).card : ℕ) : ℝ) :=
  fun B _ => by
    push_cast
    exact sum_le_sum_of_subset_of_nonneg (subset_univ _) fun _ _ _ => by positivity

/-- `hk` at `k = n = 2`: no set of two units holds three, so `miss 3 = 0 < δ`. -/
theorem hk (δ : ℝ≥0∞) (hδ : 0 < δ) : A.L.miss (2 + 1) < δ := by
  have h0 : A.L.miss (2 + 1) = 0 := by
    refine le_antisymm (iSup₂_le fun B hB => ?_) (zero_le)
    have := B.card_le_univ
    simp only [Fintype.card_fin] at this
    exact absurd hB (by omega)
  rw [h0]
  exact hδ

/-- **`audit_exfiltration`'s hypotheses are met together**: `A`'s audit at `0`, the exits as separators with
`Gen.exits_card_le`, the harm bound `hH`, and `hk`, for every `δ > 0` and a one-bit message. -/
theorem exfil (δ : ℝ≥0∞) (hδ : 0 < δ) (σ : Bool → Strategy (audit A.L Unit A.session))
    (g : (W.D → Bool) → Bool) :
    avg (fun m => prob (fun o => o.1 = true ∧ g (Partition.out W.D (A.Xc (reg (σ m)) (cont (σ m)))) = m)
        (audit A.L Unit A.session) (σ m)) ≤
      ((2 ^ (∑ u, (W.P.exits {u}).card) * ∑ j ∈ range (2 + 1), (2 : ℕ).choose j : ℕ) : ℝ≥0∞) /
          (Fintype.card Bool : ℝ≥0∞) + (δ + 0 + 0 + 0) :=
  W.P.audit_exfiltration A.hks A.hlink A.hin W.hD (fun u => (W.P.exits {u}).card) W.P.exits
    (fun _ => filter_subset _ _) (fun B => W.P.separates_exits B W.hD) (Gen.exits_card_le W.P) (hH δ)
    (hk δ hδ) σ g

end Ex

namespace N

open W

def L : SamplingDesign 2 := { Ω := Unit, draw := fun _ => univ }

def session : Finset (Fin 2) → Bool → Game Bool := fun _ _ => .ret true

/-- `W`'s transcript on the input bit `m`. -/
def T (m : Bool) : Fin C.N → Bool := ![m, !m, m, !m]

def Xc : (R : Bool) → Cont L Bool session R → (Fin C.N → Bool) := fun R _ => T R

def ext : (S : Finset (Fin 2)) → (R : Bool) → Strategy (session S R) → Fin 2 → Option (Fin C.N → Bool) :=
  fun _ R _ _ => some (T R)

def σ (m : Bool) : Strategy (audit L Bool session) := ⟨m, fun _ => PUnit.unit⟩

def g (y : D → Bool) : Bool := y ⟨2, by simp [D]⟩

theorem correct (m : Bool) (u : Fin 2) : P.Correct (T m) u := by
  intro x hxc hx
  clear hxc
  have h : (x = 1 ∧ u = 0) ∨ (x = 3 ∧ u = 0) ∨ (x = 2 ∧ u = 1) := by
    revert hx
    revert x
    revert u
    decide
  rcases h with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  · rw [P.localEval_of_mem (T m) (show P.unit 1 = some 0 by decide)]
    show T m 1 = !(P.localEval (T m) 0 0)
    rw [P.localEval_of_not_mem (T m) (show P.unit 0 ≠ some 0 by decide)]
    cases m <;> rfl
  · rw [P.localEval_of_mem (T m) (show P.unit 3 = some 0 by decide)]
    show T m 3 = !(P.localEval (T m) 0 0)
    rw [P.localEval_of_not_mem (T m) (show P.unit 0 ≠ some 0 by decide)]
    cases m <;> rfl
  · rw [P.localEval_of_mem (T m) (show P.unit 2 = some 1 by decide)]
    show T m 2 = !(P.localEval (T m) 1 1)
    rw [P.localEval_of_not_mem (T m) (show P.unit 1 ≠ some 1 by decide)]
    cases m <;> rfl

theorem hks : (P.analysis Xc ext).KnowledgeSound 0 := by
  intro R τ tgt
  refine (avg_mono fun ω => ?_).trans (avg_const (0 : ℝ≥0∞)).le
  refine (prob_mono (q := fun _ => False) ?_ _ _).trans (prob_false _ _).le
  rintro ok ⟨_, u, _, _, hf⟩
  exact hf (T R) rfl (correct R u)

theorem hlink : (P.analysis Xc ext).BindSound 0 := by
  intro R τ tgt
  refine (avg_mono fun ω => ?_).trans (avg_const (0 : ℝ≥0∞)).le
  refine (prob_mono (q := fun _ => False) ?_ _ _).trans (prob_false _ _).le
  rintro ok ⟨_, u, _, _, Y, hY, _, x, _, hne⟩
  cases Option.some.inj hY
  exact hne rfl

open Classical in
theorem prob_audit (p : Bool × Finset (Fin 2) → Prop) (s : Strategy (audit L Bool session)) :
    prob p (audit L Bool session) s = if p (true, univ) then 1 else 0 := by
  simp [audit, prob, session, Game.map, L]

theorem not_hin : ¬ Partition.AnchorsSound Xc a 0 := by
  intro h
  have h1 := h (σ true)
  have hna : ¬ C.InputsAgree (T true) a := fun hia => absurd (hia 0 rfl) (by decide)
  rw [prob_audit] at h1
  simp [reg, σ, Xc, hna] at h1

theorem missProb_ne {B : Finset (Fin 2)} (hB : B ≠ ∅) : L.missProb B = 0 := by
  have hnd : ¬ Disjoint (univ : Finset (Fin 2)) B := fun hd => by
    obtain ⟨x, hx⟩ := Finset.nonempty_iff_ne_empty.2 hB
    exact Finset.disjoint_left.1 hd (mem_univ x) hx
  simp [Verity.SamplingDesign.missProb, Verity.Game.prCoin, Verity.Game.avg, L, hnd]

theorem out_not_mem : Partition.out D (T true) ∉ P.influenceSet D L.missProb a (1 / 2) := by
  intro h
  simp only [Partition.influenceSet, mem_biUnion, mem_filter, mem_univ, true_and] at h
  obtain ⟨B, hB, hy⟩ := h
  by_cases hB0 : B = ∅
  · subst hB0
    simp only [Partition.delivered, mem_filter, mem_univ, true_and] at hy
    obtain ⟨X, hia, hinc, hout⟩ := hy
    have hc : ∀ u, P.Correct X u := fun u => by
      by_contra hw
      exact notMem_empty u (hinc (P.mem_incorrect.2 hw))
    have h0 : X 0 = false := hia 0 rfl
    have h1 : X 1 = !(X 0) := by
      have := hc 0 1 (hK (by simp [K])) (by decide)
      rw [P.localEval_of_mem X (show P.unit 1 = some 0 by decide)] at this
      change X 1 = !(P.localEval X 0 0) at this
      rwa [P.localEval_of_not_mem X (show P.unit 0 ≠ some 0 by decide)] at this
    have h2 : X 2 = !(X 1) := by
      have := hc 1 2 (hD (by simp [D])) (by decide)
      rw [P.localEval_of_mem X (show P.unit 2 = some 1 by decide)] at this
      change X 2 = !(P.localEval X 1 1) at this
      rwa [P.localEval_of_not_mem X (show P.unit 1 ≠ some 1 by decide)] at this
    have h3 : X 2 = true := congrFun hout ⟨2, by simp [D]⟩
    rw [h2, h1, h0] at h3
    exact absurd h3 (by decide)
  · rw [missProb_ne hB0] at hB
    exact absurd hB (by simp)

theorem hH : IsHarmBound L (fun u => ((P.exits {u}).card : ℝ)) (1 / 4) ((0 : ℕ) : ℝ) := by
  intro B hB
  by_cases hB0 : B = ∅
  · subst hB0
    simp
  · rw [missProb_ne hB0] at hB
    exact absurd hB (by simp)

theorem hk : L.miss (0 + 1) < 1 / 4 := by
  have h0 : L.miss (0 + 1) = 0 :=
    le_antisymm (iSup₂_le fun B hB => (missProb_ne (by rintro rfl; simp at hB)).le) (zero_le)
  rw [h0]
  exact ENNReal.div_pos one_ne_zero (by simp)

end N

end Spot.Sampling.InfluenceWitness

namespace Spot.Sampling

open InfluenceWitness

/-- **`audit_influence`'s premises hold together** at `A`: `KnowledgeSound 0`, `BindSound 0` and `AnchorsSound 0`. -/
theorem Partition.audit_influence_honest :
    (W.P.analysis A.Xc A.ext).KnowledgeSound 0 ∧ (W.P.analysis A.Xc A.ext).BindSound 0 ∧
      Partition.AnchorsSound A.Xc W.a 0 :=
  ⟨A.hks, A.hlink, A.hin⟩

/-- **`audit_influence` needs `AnchorsSound`**: at `N`, `KnowledgeSound 0` and `BindSound 0` hold, `AnchorsSound 0`
fails, and the audit accepts with the delivered value outside the influence set at `1/2` with probability `1`. -/
theorem Partition.audit_influence_nontrivial :
    (W.P.analysis N.Xc N.ext).KnowledgeSound 0 ∧ (W.P.analysis N.Xc N.ext).BindSound 0 ∧
      ¬ Partition.AnchorsSound N.Xc W.a 0 ∧
      ¬ prob (fun o => o.1 = true ∧ Partition.out W.D (N.Xc (reg (N.σ true)) (cont (N.σ true))) ∉
          W.P.influenceSet W.D N.L.missProb W.a (1 / 2)) (audit N.L Bool N.session) (N.σ true) ≤
        1 / 2 + 0 + 0 + 0 := by
  refine ⟨N.hks, N.hlink, N.not_hin, ?_⟩
  rw [N.prob_audit]
  split_ifs with hc
  · rw [add_zero, add_zero, add_zero, not_le]
    exact ENNReal.half_lt_self one_ne_zero ENNReal.one_ne_top
  · exact absurd ⟨rfl, N.out_not_mem⟩ hc

/-- **`audit_exfiltration`'s premises hold together** at `A` and `Ex`, at `δ = 1/2`. -/
theorem Partition.audit_exfiltration_honest :
    (W.P.analysis A.Xc A.ext).KnowledgeSound 0 ∧ (W.P.analysis A.Xc A.ext).BindSound 0 ∧
      Partition.AnchorsSound A.Xc W.a 0 ∧
      IsHarmBound A.L (fun u => ((W.P.exits {u}).card : ℝ)) (1 / 2) ((∑ u, (W.P.exits {u}).card : ℕ) : ℝ) ∧
      ∀ B, W.P.Separates (W.P.exits B) B W.D :=
  ⟨A.hks, A.hlink, A.hin, Ex.hH _, fun B => W.P.separates_exits B W.hD⟩

/-- **`audit_exfiltration` needs `AnchorsSound`**: at `N`, with `W`'s exits as separators, harm bound `0` and
`k = 0` at `δ = 1/4`, every other premise holds, `AnchorsSound 0` fails, and a one-bit message carried in the
unanchored input gets through with probability `1`, above the bound `1/2 + 1/4`. -/
theorem Partition.audit_exfiltration_nontrivial :
    (W.P.analysis N.Xc N.ext).KnowledgeSound 0 ∧ (W.P.analysis N.Xc N.ext).BindSound 0 ∧
      ¬ Partition.AnchorsSound N.Xc W.a 0 ∧ W.D ⊆ W.P.committed ∧ (∀ B, W.P.exits B ⊆ W.P.committed) ∧
      (∀ B, W.P.Separates (W.P.exits B) B W.D) ∧
      (∀ B, (W.P.exits B).card ≤ ∑ u ∈ B, (W.P.exits {u}).card) ∧
      IsHarmBound N.L (fun u => ((W.P.exits {u}).card : ℝ)) (1 / 4) ((0 : ℕ) : ℝ) ∧ N.L.miss (0 + 1) < 1 / 4 ∧
      ¬ avg (fun m => prob (fun o => o.1 = true ∧
            N.g (Partition.out W.D (N.Xc (reg (N.σ m)) (cont (N.σ m)))) = m)
          (audit N.L Bool N.session) (N.σ m)) ≤
        ((2 ^ 0 * ∑ j ∈ range (0 + 1), (2 : ℕ).choose j : ℕ) : ℝ≥0∞) / (Fintype.card Bool : ℝ≥0∞) +
          (1 / 4 + 0 + 0 + 0) := by
  refine ⟨N.hks, N.hlink, N.not_hin, W.hD, fun _ => filter_subset _ _, fun B => W.P.separates_exits B W.hD,
    Gen.exits_card_le W.P, N.hH, N.hk, ?_⟩
  have hp : ∀ m, prob (fun o => o.1 = true ∧
      N.g (Partition.out W.D (N.Xc (reg (N.σ m)) (cont (N.σ m)))) = m)
      (audit N.L Bool N.session) (N.σ m) = 1 := fun m => by
    rw [N.prob_audit]
    split_ifs with hc
    · rfl
    · exact absurd ⟨rfl, by cases m <;> rfl⟩ hc
  simp only [hp, avg_const]
  have h4 : (1 / 4 : ℝ≥0∞) < 2⁻¹ := by
    rw [one_div]
    exact ENNReal.inv_lt_inv.2 (by norm_num)
  refine not_le.2 ?_
  calc ((2 ^ 0 * ∑ j ∈ range (0 + 1), (2 : ℕ).choose j : ℕ) : ℝ≥0∞) / (Fintype.card Bool : ℝ≥0∞) +
        (1 / 4 + 0 + 0 + 0) = 2⁻¹ + 1 / 4 := by simp
    _ < 2⁻¹ + 2⁻¹ := ENNReal.add_lt_add_left (by simp) h4
    _ = 1 := ENNReal.inv_two_add_inv_two

end Spot.Sampling
