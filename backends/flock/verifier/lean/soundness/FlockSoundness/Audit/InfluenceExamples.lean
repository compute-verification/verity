import FlockSoundness.Audit.Influence

/-!
# The influence theorems are not vacuous

Each theorem of `Audit/Influence.lean` takes hypotheses that a careless statement could make unsatisfiable. These
witnesses meet them.

* A four-gate circuit: an input (gate 0), gate 1 = ¬0 and gate 3 = ¬0 in unit 0, and gate 2 = ¬1 in unit 1. Its outputs
  are {2, 3}, and only gate 2 is delivered (`D = {2}`). `K = {1}` separates unit 0 from `D`, and it is narrower than
  unit 0's exits {1, 3}, because gate 3 is an output that nobody is delivered (`three_mem_exits`).
  - `lemma47`: `outputs_eq_of_separates` applies to two different transcripts, `X` and `Y`, each with unit 0 wrong.
  - `count`: `card_delivered_le` bounds what they deliver by 2.
* An audit whose law draws nothing and whose session always accepts, with the honest transcript committed: `main`'s
  hypotheses hold at 0, so `audit_influence` applies (`influence_met`), and the audit accepts with probability 1
  (`accepts`).
* `harm_witness`: `card_influenceSet_le_harm`'s hypotheses hold on any partition and law, with the exits as separators,
  each unit's exported width as its harm, and the total exported width as the harm bound.
-/

namespace FlockSoundness.Audit.InfluenceExamples

open Game Finset
open scoped ENNReal

/-- The gates: an input, gate 1 = ¬0, gate 2 = ¬1 and gate 3 = ¬0. -/
def op : Fin 4 → Op 4
  | 0 => .input
  | 1 => .not 0
  | 2 => .not 1
  | 3 => .not 0

/-- The four-gate circuit, with outputs {2, 3}. -/
abbrev C : Circuit where
  N := 4
  op := op
  topo := by decide
  outputs := {2, 3}

/-- Gates 1 and 3 in unit 0, gate 2 in unit 1. -/
def unit : Fin 4 → Option (Fin 2)
  | 0 => none
  | 1 => some 0
  | 2 => some 1
  | 3 => some 0

def P : Partition C 2 where
  unit := unit
  input_iff := by decide

/-- The separator, the delivered wires and the units behind the separator. -/
def K : Finset (Fin 4) := {1}
def D : Finset (Fin 4) := {2}
def B : Finset (Fin 2) := {0}

/-- The anchors: the input is `false`. -/
def a : Fin 4 → Bool := fun _ => false

/-- Gate 1 is wrong (it should be `true`). -/
def X : Fin 4 → Bool
  | 0 => false
  | 1 => false
  | 2 => true
  | 3 => true

/-- Gates 1 and 3 are wrong. -/
def Y : Fin 4 → Bool
  | 0 => false
  | 1 => false
  | 2 => true
  | 3 => false

theorem feeds_cases {K : Finset (Fin C.N)} {g t : Fin C.N} (h : C.FeedsAvoiding K g t) :
    g = t ∨ ∃ h', g ∈ (C.op h').args := by
  induction h with
  | refl _ => exact Or.inl rfl
  | step _ hx _ _ => exact Or.inr ⟨_, hx⟩

theorem one_committed : (1 : Fin 4) ∈ P.committed :=
  P.mem_committed_of_read (g := 1) (h := 2) (by decide) (by decide)

theorem hK : K ⊆ P.committed := by
  intro g hg
  rw [show g = 1 by simpa [K] using hg]
  exact one_committed

theorem hD : D ⊆ P.committed := by
  intro g hg
  rw [show g = 2 by simpa [D] using hg]
  exact P.mem_committed_of_output (by decide)

theorem separates : P.Separates K B D := by
  intro g t u hu huB ht hgt
  obtain rfl : u = 0 := by simpa [B] using huB
  obtain rfl : t = 2 := by simpa [D] using ht
  rcases (by decide : ∀ g : Fin 4, P.unit g = some 0 → g = 1 ∨ g = 3) g hu with rfl | rfl
  · exact hgt.not_mem (by decide)
  · rcases feeds_cases hgt with h | ⟨h', hx⟩
    · exact absurd h (by decide)
    · exact absurd hx (by revert h'; decide)

/-- Unit 1 is correct when gate 2 is the negation of gate 1. -/
theorem correct_one (Z : Fin 4 → Bool) (h : Z 2 = !Z 1) : P.Correct Z 1 := by
  intro g _ hg
  obtain rfl := (by decide : ∀ g : Fin 4, P.unit g = some 1 → g = 2) g hg
  rw [P.localEval_of_mem Z hg]
  change Z 2 = !(P.localEval Z 1 1)
  rw [P.localEval_of_not_mem Z (by decide)]
  exact h

/-- Unit 0 is wrong when gate 1 copies the input instead of negating it. -/
theorem zero_wrong (Z : Fin 4 → Bool) (h : Z 1 = Z 0) : (0 : Fin 2) ∈ P.wrong Z := by
  rw [P.mem_wrong]
  intro hc
  have h1 := hc 1 one_committed (by decide)
  rw [P.localEval_of_mem Z (by decide)] at h1
  change Z 1 = !(P.localEval Z 0 0) at h1
  rw [P.localEval_of_not_mem Z (by decide), h] at h1
  cases hz : Z 0 <;> simp [hz] at h1

theorem inputsAgree_X : C.InputsAgree X a := by unfold Circuit.InputsAgree; decide

theorem inputsAgree_Y : C.InputsAgree Y a := by unfold Circuit.InputsAgree; decide

theorem wrong_sub (Z : Fin 4 → Bool) (h : Z 2 = !Z 1) : P.wrong Z ⊆ B := by
  intro u hu
  rcases (by decide : ∀ u : Fin 2, u = 0 ∨ u = 1) u with rfl | rfl
  · decide
  · exact absurd (correct_one Z h) (P.mem_wrong.1 hu)

/-- **Lemma 4.7's hypotheses are met by two different faulty transcripts**: `K` separates unit 0 from `D`, `X ≠ Y`, unit 0
is wrong in both, and the lemma gives that they deliver the same value. -/
theorem lemma47 : P.Separates K B D ∧ X ≠ Y ∧ (0 : Fin 2) ∈ P.wrong X ∧ (0 : Fin 2) ∈ P.wrong Y ∧ X 2 = Y 2 :=
  ⟨separates, fun h => absurd (congrFun h 3) (by decide), zero_wrong X (by decide), zero_wrong Y (by decide),
    P.outputs_eq_of_separates hK hD separates inputsAgree_X inputsAgree_Y (wrong_sub X (by decide))
      (wrong_sub Y (by decide)) (fun g hg => by rw [show g = 1 by simpa [K] using hg]; decide) 2 (by decide)⟩

/-- **The separator is narrower than the exits**: gate 3 is an exit of unit 0, and not in `K`. -/
theorem three_mem_exits : (3 : Fin 4) ∈ P.exits B ∧ (3 : Fin 4) ∉ K := by
  refine ⟨?_, by decide⟩
  unfold Partition.exits
  rw [mem_filter]
  exact ⟨P.mem_committed_of_output (by decide), 0, by decide, by decide⟩

/-- **The count applies**: what `X`, `Y` and every other such transcript deliver on `D` takes at most 2 values. -/
theorem count : (P.delivered D a B).card ≤ 2 :=
  (P.card_delivered_le hK hD separates a).trans (by decide)

/-! ## An audit that meets `audit_influence`'s hypotheses -/

/-- A law that draws nothing. -/
def nothing : Law 2 where
  Ω := Unit
  draw _ := ∅

/-- A session that always accepts. -/
def session : Finset (Fin 2) → Unit → Game Bool := fun _ _ => .ret true

/-- The committed transcript is the honest evaluation. -/
noncomputable def honest : (R : Unit) → Cont nothing Unit session R → (Fin 4 → Bool) := fun _ _ => C.eval a

def ext : (S : Finset (Fin 2)) → (R : Unit) → Strategy (session S R) → Fin 2 → Option (Fin 4 → Bool) :=
  fun _ _ _ _ => none

theorem knowledgeSound : (P.analysis honest ext).KnowledgeSound 0 := by
  intro R τ tgt
  refine (avg_mono fun ω => (prob_mono (q := fun _ => False) (fun ok h => ?_) _ _).trans
    (prob_false _ _).le).trans (avg_const 0).le
  obtain ⟨-, u, -, hu, -⟩ := h
  exact absurd hu (Finset.notMem_empty u)

theorem linkSound : (P.analysis honest ext).LinkSound 0 := by
  intro R τ tgt
  refine (avg_mono fun ω => (prob_mono (q := fun _ => False) (fun ok h => ?_) _ _).trans
    (prob_false _ _).le).trans (avg_const 0).le
  obtain ⟨-, u, -, hu, -⟩ := h
  exact absurd hu (Finset.notMem_empty u)

theorem anchorsSound : Partition.AnchorsSound (C := C) honest a 0 := by
  intro σ
  exact (prob_mono (q := fun _ => False) (fun o h => h.2 fun g hg => C.eval_input a hg) _ _).trans
    (prob_false _ _).le

/-- **`audit_influence`'s hypotheses are met**, at 0. -/
theorem influence_met (δ : ℝ≥0∞) (σ : Strategy (audit nothing Unit session)) :
    prob (fun o => o.1 = true ∧ Partition.out D (honest (reg σ) (cont σ)) ∉ P.influenceSet D nothing.escape a δ)
      (audit nothing Unit session) σ ≤ δ := by
  simpa using P.audit_influence knowledgeSound linkSound anchorsSound δ σ

/-- **And the audit accepts**, with probability 1, whatever the prover does. -/
theorem accepts (σ : Strategy (audit nothing Unit session)) :
    prob (fun o => o.1 = true) (audit nothing Unit session) σ = 1 := by
  rcases σ with ⟨R, τ⟩
  refine (congrArg avg (funext fun ω => ?_)).trans (avg_const 1)
  rw [prob_map]
  show prob _ (Game.ret true) _ = 1
  simp [prob]

/-! ## The harm form's hypotheses, on any partition and law -/

/-- **`card_influenceSet_le_harm`'s hypotheses are met on any partition and law**: the exits as separators
(`separates_exits`), each unit's exported width as its harm (`card_exits_le`), and the total exported width as the harm
bound. -/
theorem harm_witness {C₀ : Circuit} {n : ℕ} (P₀ : Partition C₀ n) (L : Law n) {D₀ : Finset (Fin C₀.N)}
    (hD₀ : D₀ ⊆ P₀.committed) (a₀ : Fin C₀.N → Bool) (δ : ℝ≥0∞) :
    (P₀.influenceSet D₀ L.escape a₀ δ).card ≤
      2 ^ (∑ u, (P₀.exits {u}).card) * (univ.filter fun B => δ ≤ L.escape B).card :=
  P₀.card_influenceSet_le_harm hD₀ (fun u => (P₀.exits {u}).card) P₀.exits (fun _ => filter_subset _ _)
    (fun B => P₀.separates_exits B hD₀) P₀.card_exits_le
    (fun B _ => by
      show ∑ u ∈ B, ((P₀.exits {u}).card : ℝ) ≤ ((∑ u, (P₀.exits {u}).card : ℕ) : ℝ)
      exact_mod_cast sum_le_sum_of_subset (f := fun u => (P₀.exits {u}).card) (subset_univ B))

end FlockSoundness.Audit.InfluenceExamples
