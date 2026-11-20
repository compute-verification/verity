module

public import Definitions.Compose.Guarantees.Blocks
import Mathlib.Tactic

/-!
# The conversion, sizes over a division, the chain, and the feasible set is not everything

`freeBlockAtMost` proves `FreeBlockAtMost`. `sizeAdds` proves that siblings' sizes add to at most their parent's, and
`sizeExclusive` that declared work does not double as the free blocks' for size. `chain` proves `Chain`.

The policy's check bites: `parent_acyclic` (a division's `parent` is a forest), `oneBlock_fails` (a top block in no
class fails the check) and `feasible_work_le` (under a passing division, a feasible circuit's total work is at most
the inference and free blocks' claimed sizes).

The feasible set is not every actual circuit: `honest_feasible` puts a one-gate circuit in the feasible set of a
one-block division claiming size at most 1, `two_not_feasible` keeps a two-gate circuit out of it at every pricing and
every registered input, and `not_every_feasible` is the test that a division with no function claim does not make every
actual circuit feasible; with blocks the run may leave empty, as before coverage, it fails.
-/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.BlockProperties

open Verity Finset ResourceAccounting
open scoped ENNReal

theorem freeBlockAtMost {L : Type*} {Dv : Division L} (P : Dv.Runs) : P.FreeBlockAtMost := by
  intro K F free use enclosing cap b ε huse hcap hex
  refine le_trans (P.mono _ _ ?_) huse
  rintro r ⟨ha, hn⟩
  refine ⟨ha, fun hb => hn fun i hi => ?_⟩
  have h1 := hb i hi
  have h2 := hcap r i hi
  have h3 := hex r ha i hi
  omega

theorem sizeAdds {L : Type*} {Dv : Division L} {A : Placed L} (B : Dv.Gates A) (π : Fin A.N → ℕ) : B.SizeAdds π := by
  intro j S hS
  unfold Division.Gates.size Placed.work
  rw [← Finset.sum_biUnion]
  · refine Finset.sum_le_sum_of_subset_of_nonneg ?_ fun _ _ _ => Nat.zero_le _
    intro g hg
    obtain ⟨i, hi, hg⟩ := Finset.mem_biUnion.mp hg
    exact B.nest i j (hS i hi) hg
  · intro i hi i' hi' hne
    exact B.disjoint i i' hne (by rw [hS i hi, hS i' hi'])

theorem sizeExclusive {L : Type*} {Dv : Division L} (P : Dv.Runs) (K F I : Finset Dv.ι) (hF : Disjoint F I) :
    P.DeclaredWorkExclusive K F P.size (fun r i => ∑ j ∈ Dv.within I i, P.size r j) P.size := by
  intro r _ i _
  unfold Division.Runs.size
  cases h : P.blocks r with
  | none => simp [h]
  | some B =>
    simp only [h]
    rw [← Finset.sum_union (s₁ := Dv.within F i) (s₂ := Dv.within I i) (Finset.disjoint_filter_filter hF)]
    exact sizeAdds B (P.π r) i _ fun j hj => by
      rcases Finset.mem_union.mp hj with hj | hj <;> exact (Finset.mem_filter.mp hj).2

theorem chain {L : Type*} {Dv : Division L} (P : Dv.Runs) (check : Division L → Bool) (pol : InferenceOnly.Policy)
    {κ : Type} (s : Finset κ) (K : κ → Finset Dv.ι) (p : κ → Dv.ι → Property) (ε : κ → ℝ≥0∞) :
    Chain P check pol s K p ε := by
  intro hcheck hthm hfit hcover hest
  have hsub : ∀ r, P.accepts r ∧ ¬ InferenceOnly.Compliant pol (P.actual r) (P.θ r) →
      ∃ x ∈ s, P.accepts r ∧ ¬ ∀ i ∈ K x, P.Holds r i (p x i) := by
    rintro r ⟨ha, hn⟩
    obtain ⟨hθ, hw, hπ, hsome⟩ := hfit r ha
    obtain ⟨B, hB⟩ := Option.isSome_iff_exists.mp hsome
    by_contra hall
    refine hn (hthm Dv (P.actual r) (P.π r) (P.θ r) hcheck ⟨hθ, hw, hπ, B, fun i q hq => ?_⟩)
    obtain ⟨x, hx, hi, rfl⟩ := hcover i q hq
    by_contra hq
    refine hall ⟨x, hx, ha, fun h => hq ?_⟩
    obtain ⟨B', hB', hh⟩ := h i hi
    rw [hB] at hB'
    cases hB'
    exact hh
  calc P.pr (fun r => P.accepts r ∧ ¬ InferenceOnly.Compliant pol (P.actual r) (P.θ r))
      ≤ P.pr (fun r => ∃ x ∈ s, P.accepts r ∧ ¬ ∀ i ∈ K x, P.Holds r i (p x i)) := P.mono _ _ hsub
    _ ≤ ∑ x ∈ s, P.pr (fun r => P.accepts r ∧ ¬ ∀ i ∈ K x, P.Holds r i (p x i)) := P.union _ _
    _ ≤ ∑ x ∈ s, ε x := sum_le_sum fun x hx => hest x hx

/-- A circuit of `n + 1` one-bit gates: gate 0 an input, every other gate computed from nothing. -/
def line (n : ℕ) : Circuit where
  N := n + 1
  w _ := 1
  isInput g := decide (g.val = 0)
  reads _ := ∅
  reads_lt _ _ h := by simp at h
  input_reads _ _ := rfl
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := ∅

/-- `line n` on one leaf, gate `g` at tick `g`. -/
def placed (n : ℕ) : Placed Unit where
  toCircuit := line n
  x _ := ()
  t g := g.val
  cell _ := 0
  cell_inj a b h := Fin.ext (by simpa using congrArg (fun q => q.2.2) h)

/-- One top server-day block on the one leaf over ticks `[0, 10)`, claimed of size at most 1, no registered width. -/
def oneBlock : Division Unit where
  ι := Unit
  serverDay := {()}
  inference := ∅
  free := ∅
  parent _ := none
  depth _ := 0
  depth_lt _ _ h := by cases h
  X _ := {()}
  win _ := ⟨0, 10⟩
  registered := 0
  claims _ := [.sizeAtMost 1]

/-- **The honest witness**: the one-gate circuit is in the feasible set of `oneBlock`, at unit pricing with no
registered input gates. -/
theorem honest_feasible : oneBlock.FeasibleSet (placed 1) (fun _ => 1) ∅ := by
  refine ⟨by simp, by simp [Circuit.width], fun _ _ => le_rfl, ⟨fun _ => ⟨⟨0, 10⟩, (placed 1).computed⟩,
    fun _ => rfl, ?_, ?_, ?_, ?_, ?_⟩, ?_⟩
  · intro i g hg
    simp only [Placed.region, Finset.mem_filter]
    refine ⟨hg, by simp [oneBlock], ?_⟩
    simp only [Division.ticks, oneBlock, Set.mem_Ico]
    have hg2 : g.val < 2 := g.isLt
    exact ⟨Nat.zero_le _, by simp only [placed]; omega⟩
  · exact fun g hg => ⟨(), rfl, hg⟩
  · intro i j h; simp [oneBlock] at h
  · intro i j h; simp [oneBlock] at h
  · intro i i' h; exact (h rfl).elim
  · intro i q hq
    simp only [oneBlock, List.mem_singleton] at hq
    subst hq
    show (placed 1).work (fun _ => 1) (placed 1).computed ≤ 1
    decide

/-- **A circuit outside the feasible set**: the two-gate circuit is not in the feasible set of `oneBlock`, at any
pricing and any registered input gates: both gates are in the one top block, which then has size at least 2. -/
theorem two_not_feasible (π : Fin (placed 2).N → ℕ) (θ : Finset (Fin (placed 2).N)) :
    ¬ oneBlock.FeasibleSet (placed 2) π θ := by
  rintro ⟨-, -, hπ, B, hB⟩
  have hsize := hB () (.sizeAtMost 1) (by simp [oneBlock])
  have hc : ∀ g : Fin 3, g ≠ 0 → g ∈ (placed 2).computed := by
    intro g hg
    simp only [Circuit.computed, placed, line, decide_eq_false_iff_not, Fin.val_eq_zero_iff]
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, hg⟩
  have hin : ∀ g : Fin 3, g ≠ 0 → g ∈ (B.block ()).gates := by
    intro g hg
    obtain ⟨i, -, h⟩ := B.covers g (hc g hg)
    exact h
  have h12 : ({1, 2} : Finset (Fin 3)) ⊆ (B.block ()).gates := by
    intro g hg
    simp only [Finset.mem_insert, Finset.mem_singleton] at hg
    rcases hg with rfl | rfl
    · exact hin 1 (by decide)
    · exact hin 2 (by decide)
  have hle : ∑ g ∈ ({1, 2} : Finset (Fin 3)), π g ≤ (placed 2).work π (B.block ()).gates :=
    Finset.sum_le_sum_of_subset_of_nonneg h12 fun _ _ _ => Nat.zero_le _
  have h1 := hπ (1 : Fin 3) (hc 1 (by decide))
  have h2 := hπ (2 : Fin 3) (hc 2 (by decide))
  have hpair : ∑ g ∈ ({1, 2} : Finset (Fin 3)), π g = π (1 : Fin 3) + π (2 : Fin 3) := Finset.sum_pair (by decide)
  change (placed 2).work π (B.block ()).gates ≤ 1 at hsize
  omega

/-- **The test**: a division claiming no function does not make every actual circuit feasible. -/
theorem not_every_feasible :
    ¬ ∀ (Dv : Division Unit) (A : Placed Unit) (π : Fin A.N → ℕ) (θ : Finset (Fin A.N)),
      (∀ i, ∀ p ∈ Dv.claims i, ∀ f, p ≠ .function f) → Dv.FeasibleSet A π θ :=
  fun h => two_not_feasible (fun _ => 1) ∅ (h oneBlock (placed 2) _ _ (by simp [oneBlock]))

/-- **`parent` is a forest**: no block lies, at any depth, inside itself. -/
theorem parent_acyclic {L : Type*} (Dv : Division L) (i : Dv.ι) :
    ¬ Relation.TransGen (fun a b => Dv.parent a = some b) i i := by
  have hlt : ∀ a b, Relation.TransGen (fun a b => Dv.parent a = some b) a b → Dv.depth b < Dv.depth a := by
    intro a b h
    induction h with
    | single h => exact Dv.depth_lt _ _ h
    | tail _ h ih => exact lt_trans (Dv.depth_lt _ _ h) ih
  exact fun h => lt_irrefl _ (hlt i i h)

/-- **The check refuses a top block in no class**: `oneBlock`'s one block has no block inside it and is neither an
inference nor a free block, so it fails the inference-only check at every limit. -/
theorem oneBlock_fails (lim : InferenceOnly.Limits) : InferenceOnly.check lim oneBlock = false := by
  unfold InferenceOnly.check
  have h : decide (∀ i, (∀ j, oneBlock.parent j ≠ some i) → i ∈ oneBlock.inference ∨ i ∈ oneBlock.free) = false :=
    decide_eq_false fun h => by
      rcases h () fun _ => by simp [oneBlock] with h | h <;> exact Finset.notMem_empty _ h
  rw [h]
  simp

theorem size_le_of_claim {L : Type*} {Dv : Division L} {A : Placed L} (B : Dv.Gates A) (π : Fin A.N → ℕ)
    (θ : Finset (Fin A.N)) (i : Dv.ι) (c : ℕ) (hB : ∀ p ∈ Dv.claims i, B.Holds π θ i p)
    (h : (Dv.claims i).any (InferenceOnly.sizeAtMost c) = true) : B.size π i ≤ c := by
  obtain ⟨p, hp, hpc⟩ := List.any_eq_true.mp h
  cases p <;> simp only [InferenceOnly.sizeAtMost, decide_eq_true_eq, Bool.false_eq_true] at hpc
  exact le_trans (hB _ hp) hpc

/-- **The check bites**: under a division that passes the inference-only check, every actual circuit in the feasible
set has total work at most `n · c` plus `m · freeSize`, the inference and free blocks' claimed sizes. -/
theorem feasible_work_le {L : Type*} (lim : InferenceOnly.Limits) (Dv : Division L) (A : Placed L)
    (π : Fin A.N → ℕ) (θ : Finset (Fin A.N)) (hc : InferenceOnly.check lim Dv = true)
    (hf : Dv.FeasibleSet A π θ) : A.work π A.computed ≤ lim.n * lim.c + lim.m * lim.freeSize := by
  obtain ⟨-, -, -, B, hB⟩ := hf
  simp only [InferenceOnly.check, Bool.and_eq_true, decide_eq_true_eq] at hc
  obtain ⟨⟨⟨⟨⟨⟨hleaf, -⟩, -⟩, hn⟩, hinf⟩, hm⟩, hfree⟩ := hc
  have hdesc : ∀ k i, Finset.univ.sup Dv.depth - Dv.depth i ≤ k → ∀ g ∈ (B.block i).gates,
      ∃ l, (∀ j, Dv.parent j ≠ some l) ∧ g ∈ (B.block l).gates := by
    intro k
    induction k with
    | zero =>
      intro i hi g hg
      refine ⟨i, fun j hj => ?_, hg⟩
      have h1 := Dv.depth_lt j i hj
      have h2 : Dv.depth j ≤ Finset.univ.sup Dv.depth := Finset.le_sup (Finset.mem_univ j)
      omega
    | succ k ih =>
      intro i hi g hg
      by_cases hl : ∀ j, Dv.parent j ≠ some i
      · exact ⟨i, hl, hg⟩
      · simp only [ne_eq, not_forall, not_not] at hl
        obtain ⟨j, hj⟩ := hl
        obtain ⟨i', hi', hg'⟩ := B.children j i hj g hg
        refine ih i' ?_ g hg'
        have h1 := Dv.depth_lt i' i hi'
        have h2 : Dv.depth i' ≤ Finset.univ.sup Dv.depth := Finset.le_sup (Finset.mem_univ i')
        omega
  have hsub : A.computed ⊆ (Dv.inference ∪ Dv.free).biUnion fun i => (B.block i).gates := by
    intro g hg
    obtain ⟨i, -, hgi⟩ := B.covers g hg
    obtain ⟨l, hl, hgl⟩ := hdesc _ i le_rfl g hgi
    exact Finset.mem_biUnion.mpr ⟨l, Finset.mem_union.mpr (hleaf l hl), hgl⟩
  have hbu : ∀ S : Finset Dv.ι, A.work π (S.biUnion fun i => (B.block i).gates) ≤ ∑ i ∈ S, B.size π i := by
    intro S
    induction S using Finset.induction_on with
    | empty => simp [Placed.work]
    | insert a S ha ih =>
      rw [Finset.biUnion_insert, Finset.sum_insert ha]
      have := Finset.sum_union_inter (s₁ := (B.block a).gates) (s₂ := S.biUnion fun i => (B.block i).gates)
        (f := π)
      unfold Division.Gates.size Placed.work at ih ⊢
      omega
  have hw : A.work π A.computed ≤ A.work π ((Dv.inference ∪ Dv.free).biUnion fun i => (B.block i).gates) :=
    Finset.sum_le_sum_of_subset_of_nonneg hsub fun _ _ _ => Nat.zero_le _
  have hs := Finset.sum_union_inter (s₁ := Dv.inference) (s₂ := Dv.free) (f := fun i => B.size π i)
  have hI : ∑ i ∈ Dv.inference, B.size π i ≤ Dv.inference.card * lim.c := by
    simpa using Finset.sum_le_card_nsmul Dv.inference (fun i => B.size π i) lim.c
      fun i hi => size_le_of_claim B π θ i lim.c (hB i) (hinf i hi).1
  have hF : ∑ i ∈ Dv.free, B.size π i ≤ Dv.free.card * lim.freeSize := by
    simpa using Finset.sum_le_card_nsmul Dv.free (fun i => B.size π i) lim.freeSize
      fun i hi => size_le_of_claim B π θ i lim.freeSize (hB i) (hfree i hi)
  have := hbu (Dv.inference ∪ Dv.free)
  have h1 := Nat.mul_le_mul_right lim.c hn
  have h2 := Nat.mul_le_mul_right lim.freeSize hm
  omega

end ResourceAccounting.SecurityProofs.BlockProperties
