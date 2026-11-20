module

public import Definitions.Compose.Guarantees.Blocks
import Mathlib.Tactic

/-!
# The conversion, sizes over the partitions, the chain, and the feasible set is not everything

`freeBlockAtMost` proves `FreeBlockAtMost`. `sizeAdds` proves that the blocks directly inside a server-day block have
sizes adding to at most it, and `sizeExclusive` that a server-day block's free and inference children's work do not
double-count. `chain` proves `Chain`.

The policy's check bites: `feasible_work_le` bounds a realizing run's total work by, for each server-day block, the
smaller of its own capacity and its inference and free blocks' claimed sizes, plus the assumed hidden hardware size.

The feasible set is not every run: `honest_feasible` puts a one-gate circuit's run in the feasible set of a one-block
run claiming size at most 1, and `two_not_feasible` keeps a two-gate circuit's run out of it. `top_blocks_classified`
records that no top block is in no class (the partitions' `kind` is total), so the degenerate case the reviews found
cannot be stated; nothing sits inside an inference block, the finest scale here, so the other cannot be stated either.
-/

@[expose] public section

namespace Verity.Partition

open Finset

variable {C : Circuit} {n : ℕ} (Q : Partition C n)

/-- Distinct units have disjoint gates. -/
theorem gates_disjoint (u v : Fin n) (h : u ≠ v) : Disjoint (Q.gates u) (Q.gates v) := by
  rw [Finset.disjoint_left]
  intro g hu hv
  simp only [gates, mem_filter] at hu hv
  exact h (Option.some_injective _ (hu.2.symm.trans hv.2))

/-- The computed gates are the union of the units' gates. -/
theorem computed_eq_biUnion : C.computed = univ.biUnion Q.gates := by
  ext g
  constructor
  · intro hg
    have hc : C.isInput g = false := by simpa [Circuit.computed] using hg
    have hne : Q.unit g ≠ none := by rw [Q.input_iff]; simp [hc]
    obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.mp hne
    exact mem_biUnion.mpr ⟨u, mem_univ u, by simp [gates, hu]⟩
  · intro hg
    obtain ⟨u, -, hu⟩ := mem_biUnion.mp hg
    simp only [gates, mem_filter] at hu
    have hne : Q.unit g ≠ none := by rw [hu.2]; simp
    have hc : C.isInput g = false := by
      cases hb : C.isInput g with
      | false => rfl
      | true => exact absurd ((Q.input_iff g).mpr hb) hne
    simp [Circuit.computed, hc]

end Verity.Partition

namespace ResourceAccounting.SecurityProofs.BlockProperties

open Verity Finset ResourceAccounting
open scoped ENNReal

variable {L : Type*}

/-! ## Sizes over the partitions -/

/-- A server-day block's gates are the union of the blocks directly inside it. -/
theorem outer_children_cover (P : Runs L) (r : P.Run) (i : Fin (P.nOut r)) :
    (P.outer r).gates i = (P.children r i).biUnion (fun u => (P.inner r).gates u) := by
  ext g
  simp only [Partition.gates, mem_filter, mem_univ, true_and, mem_biUnion, Runs.children]
  rw [(P.ref r).sub g]
  constructor
  · intro h
    obtain ⟨u, hu, hpu⟩ := Option.map_eq_some'.mp h
    exact ⟨u, by simp [hpu], hu⟩
  · rintro ⟨u, hu, hg⟩
    simp only [mem_filter, mem_univ, true_and] at hu
    rw [hg]; simp [hu]

/-- A server-day block's size is the sum of the sizes of the blocks directly inside it. -/
theorem sizeOut_eq_children_sum (P : Runs L) (r : P.Run) (i : Fin (P.nOut r)) :
    P.sizeOut r i = ∑ u ∈ P.children r i, P.sizeIn r u := by
  unfold Runs.sizeOut Runs.sizeIn Placed.work
  rw [outer_children_cover, Finset.sum_biUnion]
  exact fun a _ b _ hab => (P.inner r).gates_disjoint a b hab

/-- The whole circuit's work is the sum of the server-day-or-hidden blocks' sizes. -/
theorem work_eq_outer_sum (P : Runs L) (r : P.Run) :
    (P.actual r).work (P.π r) (P.actual r).computed = ∑ i, P.sizeOut r i := by
  unfold Runs.sizeOut Placed.work
  rw [(P.outer r).computed_eq_biUnion, Finset.sum_biUnion]
  exact fun a _ b _ hab => (P.outer r).gates_disjoint a b hab

/-- The inference and free blocks directly inside a server-day block are the blocks directly inside it. -/
theorem children_split (P : Runs L) (r : P.Run) (i : Fin (P.nOut r)) (hki : P.kind r i = true) :
    P.children r i = P.infChildren r i ∪ P.freeChildren r i := by
  ext u
  simp only [Runs.children, Runs.infChildren, Runs.freeChildren, mem_filter, mem_union, mem_univ, true_and]
  constructor
  · intro hp
    have hnh : P.cls r u ≠ Class.hidden := by
      rw [P.hiddenIff, hp]; simp [hki]
    rcases (by cases h : P.cls r u <;> simp_all : P.cls r u = Class.inference ∨ P.cls r u = Class.free) with h | h
    · exact Or.inl ⟨hp, h⟩
    · exact Or.inr ⟨hp, h⟩
  · rintro (⟨hp, _⟩ | ⟨hp, _⟩) <;> exact hp

theorem infFree_disjoint (P : Runs L) (r : P.Run) (i : Fin (P.nOut r)) :
    Disjoint (P.infChildren r i) (P.freeChildren r i) := by
  rw [Finset.disjoint_left]
  intro u h1 h2
  simp only [Runs.infChildren, Runs.freeChildren, mem_filter] at h1 h2
  rw [h1.2] at h2
  exact absurd h2.2 (by decide)

/-- The blocks directly inside a server-day block, in `S`, have sizes adding to at most it. -/
theorem sizeAdds (P : Runs L) (r : P.Run) (i : Fin (P.nOut r)) (S : Finset (Fin (P.nIn r)))
    (hS : ∀ u ∈ S, (P.ref r).parent u = i) : ∑ u ∈ S, P.sizeIn r u ≤ P.sizeOut r i := by
  rw [sizeOut_eq_children_sum]
  refine Finset.sum_le_sum_of_subset_of_nonneg (fun u hu => ?_) (fun _ _ _ => Nat.zero_le _)
  simp only [Runs.children, mem_filter, mem_univ, true_and]
  exact hS u hu

/-- **Declared work does not double as other work, for size**: a server-day block's free and inference children's work
add up to at most its own size. -/
theorem sizeExclusive (P : Runs L) :
    P.DeclaredWorkExclusive (fun r i => ∑ u ∈ P.freeChildren r i, P.sizeIn r u)
      (fun r i => ∑ u ∈ P.infChildren r i, P.sizeIn r u) P.sizeOut := by
  intro r _ i hki
  rw [sizeOut_eq_children_sum, children_split P r i hki, Finset.sum_union (infFree_disjoint P r i)]
  omega

/-! ## The conversion -/

theorem freeBlockAtMost (P : Runs L) : P.FreeBlockAtMost := by
  intro free use enclosing cap b ε huse hcap hex
  refine le_trans (P.mono _ _ ?_) huse
  rintro r ⟨ha, hn⟩
  refine ⟨ha, fun hb => hn fun i hi => ?_⟩
  have h1 := hb i hi
  have h2 := hcap r i hi
  have h3 := hex r ha i hi
  omega

/-! ## The chain -/

theorem chain (P : Runs L) (check : (P : Runs L) → P.Run → Bool) (pol : InferenceOnly.Policy)
    {κ : Type} (s : Finset κ) (shown : κ → P.Run → Prop) (ε : κ → ℝ≥0∞) :
    Chain P check pol s shown ε := by
  intro hthm hcheck hcover hest
  have hsub : ∀ r, P.accepts r ∧ ¬ InferenceOnly.Compliant pol (P.actual r) (P.θ r) →
      ∃ x ∈ s, P.accepts r ∧ ¬ shown x r := by
    rintro r ⟨ha, hn⟩
    by_contra hall
    push_neg at hall
    exact hn (hthm P r (hcheck r ha) (hcover r ha fun x hx => hall x hx ha))
  calc P.pr (fun r => P.accepts r ∧ ¬ InferenceOnly.Compliant pol (P.actual r) (P.θ r))
      ≤ P.pr (fun r => ∃ x ∈ s, P.accepts r ∧ ¬ shown x r) := P.mono _ _ hsub
    _ ≤ ∑ x ∈ s, P.pr (fun r => P.accepts r ∧ ¬ shown x r) := P.union _ _
    _ ≤ ∑ x ∈ s, ε x := Finset.sum_le_sum fun x hx => hest x hx

/-! ## The check bites -/

/-- A size claim at most `c`, held, bounds the block's size. -/
theorem sizeIn_le_of_claim (P : Runs L) (r : P.Run) (u : Fin (P.nIn r)) (c : ℕ)
    (hcl : ∀ p ∈ P.claimsIn r u, P.HoldsIn r u p)
    (hany : (P.claimsIn r u).any (InferenceOnly.sizeAtMost c) = true) : P.sizeIn r u ≤ c := by
  obtain ⟨p, hp, hpc⟩ := List.any_eq_true.mp hany
  cases p with
  | sizeAtMost sz =>
    simp only [InferenceOnly.sizeAtMost, decide_eq_true_eq] at hpc
    have := hcl _ hp
    simp only [Runs.HoldsIn, Runs.innerBlock, Placed.Holds, Runs.sizeIn] at this ⊢
    omega
  | _ => simp [InferenceOnly.sizeAtMost] at hpc

/-- **The check bites**: under a run that passes the inference-only check and realizes its claims, with each server-day
block's own capacity `serverCap i` (the second cap on the same gates) and the assumed hidden hardware size `lim.hidden`,
the whole circuit's work is at most, for each server-day block, the smaller of its capacity and its inference and free
blocks' claimed sizes, plus the hidden hardware size. -/
theorem feasible_work_le (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run)
    (serverCap : Fin (P.nOut r) → ℕ) (hc : InferenceOnly.check lim P r = true) (hreal : P.Realizes r)
    (hserver : ∀ i, P.kind r i = true → P.sizeOut r i ≤ serverCap i)
    (hhidden : ∑ i ∈ univ.filter (fun i => P.kind r i = false), P.sizeOut r i ≤ lim.hidden) :
    (P.actual r).work (P.π r) (P.actual r).computed ≤
      (∑ i ∈ univ.filter (fun i => P.kind r i = true),
        min (serverCap i) ((P.infChildren r i).card * lim.c + (P.freeChildren r i).card * lim.freeSize)) +
        lim.hidden := by
  simp only [InferenceOnly.check, Bool.and_eq_true, decide_eq_true_eq] at hc
  obtain ⟨⟨⟨⟨_, hinf⟩, _⟩, hfree⟩, _⟩ := hc
  obtain ⟨-, -, -, hcl_in, -⟩ := hreal
  rw [work_eq_outer_sum, ← Finset.sum_filter_add_sum_filter_not univ (fun i => P.kind r i = true) (P.sizeOut r)]
  have hfalse : (univ.filter fun i => ¬ P.kind r i = true) = univ.filter fun i => P.kind r i = false := by
    apply Finset.filter_congr; intro i _; simp [Bool.not_eq_true]
  rw [hfalse]
  refine add_le_add ?_ hhidden
  refine Finset.sum_le_sum fun i hi => ?_
  simp only [mem_filter] at hi
  have hki := hi.2
  refine le_min (hserver i hki) ?_
  rw [sizeOut_eq_children_sum, children_split P r i hki, Finset.sum_union (infFree_disjoint P r i)]
  refine add_le_add ?_ ?_
  · calc ∑ u ∈ P.infChildren r i, P.sizeIn r u ≤ ∑ _u ∈ P.infChildren r i, lim.c :=
          Finset.sum_le_sum fun u hu => by
            simp only [Runs.infChildren, mem_filter] at hu
            exact sizeIn_le_of_claim P r u lim.c (hcl_in u) (hinf u hu.2).1
      _ = (P.infChildren r i).card * lim.c := by rw [Finset.sum_const, smul_eq_mul]
  · calc ∑ u ∈ P.freeChildren r i, P.sizeIn r u ≤ ∑ _u ∈ P.freeChildren r i, lim.freeSize :=
          Finset.sum_le_sum fun u hu => by
            simp only [Runs.freeChildren, mem_filter] at hu
            exact sizeIn_le_of_claim P r u lim.freeSize (hcl_in u) (hfree u hu.2)
      _ = (P.freeChildren r i).card * lim.freeSize := by rw [Finset.sum_const, smul_eq_mul]

/-! ## The feasible set is not everything -/

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

open Classical in
/-- One run: the circuit `A`, no fresh input, one server-day block holding the whole circuit as one inference block,
claimed of size at most 1 and 0 fresh input, with a size claim on the server-day block. -/
noncomputable def demoRuns (A : Placed Unit) : Runs Unit where
  Run := Unit
  pr _ := 0
  mono _ _ _ := le_refl 0
  union _ _ := by simp
  accepts _ := True
  actual _ := A
  π _ _ := 1
  θ _ := ∅
  registered := 0
  nOut _ := 1
  outer _ := Partition.whole A.toCircuit
  kind _ _ := true
  locOut _ _ := {()}
  winOut _ _ := ⟨0, 10⟩
  claimsOut _ _ := [.sizeAtMost 1]
  nIn _ := 1
  inner _ := Partition.whole A.toCircuit
  ref _ := { parent := id, sub := fun g => by cases (Partition.whole A.toCircuit).unit g <;> simp }
  cls _ _ := Class.inference
  locIn _ _ := {()}
  winIn _ _ := ⟨0, 10⟩
  claimsIn _ _ := [.sizeAtMost 1, .inputAtMost [.past, .block, .world] 0]
  hiddenIff _ _ := by decide

/-- **The honest witness**: the one-gate circuit's run realizes the claims of `demoRuns (placed 1)`. -/
theorem honest_feasible : (demoRuns (placed 1)).Realizes () := by
  refine ⟨by simp, by simp [demoRuns, Circuit.width], fun _ _ => le_rfl, ?_, ?_⟩
  · intro u p hp
    fin_cases u
    simp only [demoRuns, List.mem_cons, List.mem_singleton, List.not_mem_nil] at hp
    rcases hp with rfl | rfl
    · show (placed 1).work _ _ ≤ 1
      simp only [demoRuns, Runs.innerBlock, Partition.whole, Partition.gates, Placed.work]
      decide
    · show (placed 1).blockInput _ _ _ _ ≤ 0
      simp only [demoRuns, Runs.innerBlock, Placed.blockInput, Circuit.into, Circuit.width, placed, line]
      decide
  · intro i p hp
    fin_cases i
    simp only [demoRuns, List.mem_singleton] at hp
    subst hp
    show (placed 1).work _ _ ≤ 1
    simp only [demoRuns, Runs.outerBlock, Partition.whole, Partition.gates, Placed.work]
    decide

/-- **A run outside the feasible set**: the two-gate circuit's run does not realize the claims of `demoRuns (placed 2)`,
whose one inference block is then of size at least 2. -/
theorem two_not_feasible : ¬ (demoRuns (placed 2)).Realizes () := by
  rintro ⟨-, -, -, hcl_in, -⟩
  have h := hcl_in 0 (.sizeAtMost 1) (by simp [demoRuns])
  simp only [Runs.HoldsIn, Runs.innerBlock, Placed.Holds, demoRuns, Partition.whole, Partition.gates,
    Placed.work] at h
  revert h
  decide

/-- **The test**: not every circuit's run is in the feasible set. -/
theorem not_every_feasible : ¬ ∀ A : Placed Unit, (demoRuns A).Realizes () :=
  fun h => two_not_feasible (h (placed 2))

/-- **No top block is in no class**: `kind` is total, so the degenerate case of a top block in neither the server-day nor
the hidden class cannot be stated. Nothing sits inside an inference block, the finest scale here, so neither can a block
inside an inference block that is not a proof block. -/
theorem top_blocks_classified (P : Runs L) (r : P.Run) (i : Fin (P.nOut r)) :
    P.kind r i = true ∨ P.kind r i = false := by cases P.kind r i <;> simp

end ResourceAccounting.SecurityProofs.BlockProperties
