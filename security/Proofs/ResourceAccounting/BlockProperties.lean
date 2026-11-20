module

public import Definitions.Compose.Guarantees.Blocks
import Mathlib.Tactic

/-!
# The conversion, sizes over a division, the chain, and the feasible set is not everything

`freeBlockAtMost` proves `FreeBlockAtMost`. `sizeAdds` proves that siblings' sizes add to at most their parent's, and
`sizeExclusive` that declared work does not double as a free block's for size. `chain` proves `Chain`.

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
  intro K F encl free use enclosing cap b ε hF huse hcap hex
  refine le_trans (P.mono _ _ ?_) huse
  rintro r ⟨ha, hn⟩
  refine ⟨ha, fun hb => hn fun j hj => ?_⟩
  have h1 := hb (encl j) (hF j hj)
  have h2 := hcap r (encl j) (hF j hj)
  have h3 := hex r ha j hj
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

theorem sizeExclusive {L : Type*} {Dv : Division L} (P : Dv.Runs) (F : Finset Dv.ι) (encl : Dv.ι → Dv.ι)
    (hF : ∀ j ∈ F, Dv.parent j = some (encl j) ∧ j ∉ Dv.inference) :
    P.DeclaredWorkExclusive F encl P.size
      (fun r i => ∑ j ∈ Dv.inference.filter (Dv.parent · = some i), P.size r j) P.size := by
  intro r _ j hj
  obtain ⟨hp, hn⟩ := hF j hj
  unfold Division.Runs.size
  cases h : P.blocks r with
  | none => simp
  | some B =>
    simp only
    have hnot : j ∉ Dv.inference.filter (Dv.parent · = some (encl j)) := by simp [hn]
    have := sizeAdds B (P.π r) (encl j) (insert j (Dv.inference.filter (Dv.parent · = some (encl j)))) (by
      intro i hi
      rcases Finset.mem_insert.mp hi with rfl | hi
      · exact hp
      · exact (Finset.mem_filter.mp hi).2)
    rwa [Finset.sum_insert hnot] at this

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
    exact ⟨Nat.zero_le _, by simp only [placed]; omega⟩
  · exact fun g hg => ⟨(), rfl, hg⟩
  · intro i j h; simp [oneBlock] at h
  · intro i j h; simp [oneBlock] at h
  · intro i i' h; exact absurd (Subsingleton.elim i i') h
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
    simp only [Circuit.computed, Finset.mem_filter, Finset.mem_univ, true_and]
    show decide (g.val = 0) = false
    simpa [Fin.ext_iff] using hg
  have hin : ∀ g : Fin 3, g ≠ 0 → g ∈ (B.block ()).gates := by
    intro g hg
    obtain ⟨i, -, h⟩ := B.covers g (hc g hg)
    exact (Subsingleton.elim i ()) ▸ h
  have h12 : ({1, 2} : Finset (Fin 3)) ⊆ (B.block ()).gates := by
    intro g hg
    simp only [Finset.mem_insert, Finset.mem_singleton] at hg
    rcases hg with rfl | rfl
    · exact hin 1 (by decide)
    · exact hin 2 (by decide)
  have hle : ∑ g ∈ ({1, 2} : Finset (Fin 3)), π g ≤ (placed 2).work π (B.block ()).gates :=
    Finset.sum_le_sum_of_subset_of_nonneg h12 fun _ _ _ => Nat.zero_le _
  have h1 := hπ 1 (hc 1 (by decide))
  have h2 := hπ 2 (hc 2 (by decide))
  rw [Finset.sum_pair (by decide)] at hle
  change (placed 2).work π (B.block ()).gates ≤ 1 at hsize
  omega

/-- **The test**: a division claiming no function does not make every actual circuit feasible. -/
theorem not_every_feasible :
    ¬ ∀ (Dv : Division Unit) (A : Placed Unit) (π : Fin A.N → ℕ) (θ : Finset (Fin A.N)),
      (∀ i, ∀ p ∈ Dv.claims i, ∀ f, p ≠ .function f) → Dv.FeasibleSet A π θ :=
  fun h => two_not_feasible (fun _ => 1) ∅ (h oneBlock (placed 2) _ _ (by simp [oneBlock]))

end ResourceAccounting.SecurityProofs.BlockProperties
