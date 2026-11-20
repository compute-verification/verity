module

public import Definitions.Compose.Guarantees.Blocks
public import Definitions.Registration.Physical
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
    have hne : Q.unit g ≠ none := by rw [ne_eq, Q.input_iff]; simp [hc]
    obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.mp hne
    exact mem_biUnion.mpr ⟨u, mem_univ u, by simp [Partition.gates, hu]⟩
  · intro hg
    obtain ⟨u, -, hu⟩ := mem_biUnion.mp hg
    simp only [gates, mem_filter, mem_univ, true_and] at hu
    have hne : Q.unit g ≠ none := by rw [hu]; simp
    have hc : C.isInput g = false := by
      cases hb : C.isInput g with
      | false => rfl
      | true => exact absurd ((Q.input_iff g).mpr hb) hne
    simp only [Circuit.computed, mem_filter, mem_univ, true_and, hc]

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
    cases hu : (P.inner r).unit g with
    | none => rw [hu] at h; simp at h
    | some u =>
      rw [hu] at h
      simp only [Option.map_some, Option.some.injEq] at h
      exact ⟨u, h, rfl⟩
  · rintro ⟨u, hpu, hg⟩
    rw [hg]; simp [hpu]

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

/-- **A server-day block's size comes from the compute rate**, not an assumed figure: its gates lie in its region
(`insideOut`), so its work is at most the location's compute rate `Φ` over the window's length. -/
theorem sizeOut_le_compute (P : Runs L) (r : P.Run) (i : Fin (P.nOut r)) (𝒳 : Set (Finset L)) (Φ : Finset L → ℕ)
    (hcr : (P.actual r).ComputeRate 𝒳 (P.π r) Φ) (hX : P.locOut r i ∈ 𝒳) :
    P.sizeOut r i ≤ Φ (P.locOut r i) * (P.winOut r i).len := by
  have hmono : P.sizeOut r i ≤
      (P.actual r).work (P.π r) ((P.actual r).region (P.locOut r i) ↑((P.winOut r i).ticks)) := by
    unfold Runs.sizeOut Placed.work
    exact Finset.sum_le_sum_of_subset (P.insideOut r i)
  have hticks : ((P.winOut r i).ticks : Set ℕ)
      = Set.Ico (P.winOut r i).start ((P.winOut r i).start + (P.winOut r i).len) := by
    simp [Window.ticks]
  rw [hticks] at hmono
  calc P.sizeOut r i ≤ _ := hmono
    _ ≤ Φ (P.locOut r i) * ((P.winOut r i).start + (P.winOut r i).len - (P.winOut r i).start) :=
        hcr (P.locOut r i) hX _ _
    _ = Φ (P.locOut r i) * (P.winOut r i).len := by rw [Nat.add_sub_cancel_left]

/-- The inference and free blocks directly inside a server-day block are the blocks directly inside it. -/
theorem children_split (P : Runs L) (r : P.Run) (i : Fin (P.nOut r)) (hki : P.kind r i = true) :
    P.children r i = P.infChildren r i ∪ P.freeChildren r i := by
  ext u
  simp only [Runs.children, Runs.infChildren, Runs.freeChildren, mem_filter, mem_union, mem_univ, true_and]
  constructor
  · intro hp
    have hnh : P.cls r u ≠ Class.hidden := by
      rw [ne_eq, P.hiddenIff, hp]; simp [hki]
    have hcls : P.cls r u = Class.inference ∨ P.cls r u = Class.free := by
      cases h : P.cls r u with
      | inference => exact Or.inl rfl
      | free => exact Or.inr rfl
      | hidden => exact absurd h hnh
    rcases hcls with h | h
    · exact Or.inl ⟨hp, h⟩
    · exact Or.inr ⟨hp, h⟩
  · rintro (⟨hp, _⟩ | ⟨hp, _⟩) <;> exact hp

theorem infFree_disjoint (P : Runs L) (r : P.Run) (i : Fin (P.nOut r)) :
    Disjoint (P.infChildren r i) (P.freeChildren r i) := by
  rw [Finset.disjoint_left]
  intro u h1 h2
  simp only [Runs.infChildren, Runs.freeChildren, mem_filter, mem_univ, true_and] at h1 h2
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
  simp only
  rw [sizeOut_eq_children_sum, children_split P r i hki, Finset.sum_union (infFree_disjoint P r i)]
  omega

/-! ## The conversion -/

theorem freeBlockAtMost (P : Runs L) : P.FreeBlockAtMost := by
  intro _what free use enclosing cap b ε huse hcap hex
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
    obtain ⟨M, hM⟩ := hcover r ha fun x hx => hall x hx ha
    exact hn (hthm P r M (hcheck r ha) hM)
  calc P.pr (fun r => P.accepts r ∧ ¬ InferenceOnly.Compliant pol (P.actual r) (P.θ r))
      ≤ P.pr (fun r => ∃ x ∈ s, P.accepts r ∧ ¬ shown x r) := P.mono _ _ hsub
    _ ≤ ∑ x ∈ s, P.pr (fun r => P.accepts r ∧ ¬ shown x r) := P.union _ _
    _ ≤ ∑ x ∈ s, ε x := Finset.sum_le_sum fun x hx => hest x hx

/-- **An empty tolerance realizes**: `RealizesExcept ∅` gives `Realizes` (the claim clauses for every block, dropping the
tolerance term). The converse needs the free quantity to fit the capacity, which `Realizes` alone does not carry, so the
tolerance is a term and not a mere relabelling of the honest path. -/
theorem realizesExcept_empty_realizes (P : Runs L) (r : P.Run) : P.RealizesExcept r ∅ → P.Realizes r := by
  rintro ⟨h1, h2, h3, h4, h5, h6, -, -⟩
  refine ⟨h1, h2, h3, fun u => ?_, h6⟩
  by_cases hc : P.cls r u = Class.inference
  · exact h4 u hc (Finset.notMem_empty u)
  · exact h5 u hc

/-- **`Realizes` gives `RealizesExcept ∅`** once its free blocks' work fits each server-day block's free capacity: the
honest path is the empty tolerance only under the capacity the conversion yields. -/
theorem realizes_realizesExcept (P : Runs L) (r : P.Run)
    (hfit : ∀ i, P.kind r i = true → (∑ u ∈ P.freeChildren r i, P.sizeIn r u) ≤ P.freeCap r i)
    (h : P.Realizes r) : P.RealizesExcept r ∅ := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  refine ⟨h1, h2, h3, fun u _ _ => h4 u, fun u _ => h4 u, h5, fun u hu => absurd hu (Finset.notMem_empty u), ?_⟩
  intro i hi
  simpa using hfit i hi

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

/-- A block's claimed size is one of its claims when it has a size claim. -/
theorem claimedSize_mem : ∀ (l : List Property),
    l.any InferenceOnly.isSize = true → Property.sizeAtMost (InferenceOnly.claimedSize l) ∈ l
  | [], h => by simp at h
  | p :: t, h => by
    cases p with
    | sizeAtMost k => simp [InferenceOnly.claimedSize]
    | function f =>
      simp only [InferenceOnly.claimedSize, List.any_cons, InferenceOnly.isSize, Bool.false_or] at h ⊢
      exact List.mem_cons_of_mem _ (claimedSize_mem t h)
    | inputAtMost s k =>
      simp only [InferenceOnly.claimedSize, List.any_cons, InferenceOnly.isSize, Bool.false_or] at h ⊢
      exact List.mem_cons_of_mem _ (claimedSize_mem t h)
    | outputAtMost k =>
      simp only [InferenceOnly.claimedSize, List.any_cons, InferenceOnly.isSize, Bool.false_or] at h ⊢
      exact List.mem_cons_of_mem _ (claimedSize_mem t h)

/-- A block with a size claim, all held, has size at most its claimed size. -/
theorem sizeIn_le_claimedSize (P : Runs L) (r : P.Run) (u : Fin (P.nIn r))
    (hcl : ∀ p ∈ P.claimsIn r u, P.HoldsIn r u p)
    (hany : (P.claimsIn r u).any InferenceOnly.isSize = true) :
    P.sizeIn r u ≤ InferenceOnly.claimedSize (P.claimsIn r u) := by
  have := hcl _ (claimedSize_mem (P.claimsIn r u) hany)
  simp only [Runs.HoldsIn, Runs.innerBlock, Placed.Holds, Runs.sizeIn] at this ⊢
  exact this

/-- **The check bites, on the path the chain reads**: under a run that passes the inference-only check and realizes its
claims outside a tolerance set `M` (`RealizesExcept`, not `Realizes`), with each server-day block's own capacity
`serverCap i` and the assumed hidden hardware size `lim.hidden`, the whole circuit's work is at most, for each server-day
block, the smaller of its capacity and the count of its inference blocks **outside** `M` at the size cap `lim.c` plus the
free-block capacity `P.freeCap r i` (which bounds the free blocks together with `M`'s actual work). The tolerance `M` is a
bounded term here, not an unbounded quantifier: `M`'s blocks are counted inside the free capacity, not as inference
credit. -/
theorem feasible_work_le (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (serverCap : Fin (P.nOut r) → ℕ)
    (hc : InferenceOnly.check lim P r = true) (hreal : P.RealizesExcept r M)
    (hserver : ∀ i, P.kind r i = true → P.sizeOut r i ≤ serverCap i)
    (hhidden : ∑ i ∈ univ.filter (fun i => P.kind r i = false), P.sizeOut r i ≤ lim.hidden) :
    (P.actual r).work (P.π r) (P.actual r).computed ≤
      (∑ i ∈ univ.filter (fun i => P.kind r i = true),
        min (serverCap i) (((P.infChildren r i).filter (fun u => u ∉ M)).card * lim.c + P.freeCap r i)) +
        lim.hidden := by
  simp only [InferenceOnly.check, Bool.and_eq_true, decide_eq_true_eq] at hc
  obtain ⟨⟨⟨⟨⟨⟨-, hinf⟩, -⟩, -⟩, -⟩, -⟩, -⟩ := hc
  obtain ⟨-, -, -, hcl_out, -, -, -, hfreecap⟩ := hreal
  rw [work_eq_outer_sum, ← Finset.sum_filter_add_sum_filter_not univ (fun i => P.kind r i = true) (P.sizeOut r)]
  have hfalse : (univ.filter fun i => ¬ P.kind r i = true) = univ.filter fun i => P.kind r i = false := by
    apply Finset.filter_congr; intro i _; simp [Bool.not_eq_true]
  rw [hfalse]
  refine add_le_add ?_ hhidden
  refine Finset.sum_le_sum fun i hi => ?_
  simp only [mem_filter] at hi
  have hki := hi.2
  refine le_min (hserver i hki) ?_
  rw [sizeOut_eq_children_sum, children_split P r i hki, Finset.sum_union (infFree_disjoint P r i),
    ← Finset.sum_filter_add_sum_filter_not (P.infChildren r i) (fun u => u ∈ M) (P.sizeIn r)]
  have hout : ∑ u ∈ (P.infChildren r i).filter (fun u => ¬ u ∈ M), P.sizeIn r u
      ≤ ((P.infChildren r i).filter (fun u => u ∉ M)).card * lim.c := by
    calc ∑ u ∈ (P.infChildren r i).filter (fun u => ¬ u ∈ M), P.sizeIn r u
        ≤ ∑ _u ∈ (P.infChildren r i).filter (fun u => ¬ u ∈ M), lim.c := Finset.sum_le_sum fun u hu => by
          simp only [Runs.infChildren, mem_filter, mem_univ, true_and] at hu
          exact sizeIn_le_of_claim P r u lim.c (hcl_out u hu.1.2 hu.2) (hinf u hu.1.2).1
      _ = ((P.infChildren r i).filter (fun u => u ∉ M)).card * lim.c := by rw [Finset.sum_const, smul_eq_mul]
  have hfc := hfreecap i hki
  omega

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
/-- One run of the circuit `placed n`: no fresh input, one server-day block holding the whole circuit as one inference
block, claimed of size at most 1 and 0 fresh input, with a size claim on the server-day block. Its window covers every
tick of `placed n`, so its one block is its region (`insideOut`). -/
noncomputable def demoRuns (n : ℕ) : Runs Unit where
  Run := Unit
  pr _ := 0
  mono _ _ _ := le_refl 0
  union _ _ := by simp
  accepts _ := True
  actual _ := placed n
  π _ _ := 1
  θ _ := ∅
  registered := 0
  nOut _ := 1
  outer _ := Partition.whole (placed n).toCircuit
  kind _ _ := true
  locOut _ _ := {()}
  winOut _ _ := ⟨0, n + 1⟩
  claimsOut _ _ := [.sizeAtMost 1]
  freeCap _ _ := 0
  nIn _ := 1
  inner _ := Partition.whole (placed n).toCircuit
  ref _ := { parent := id, sub := fun g => by cases (Partition.whole (placed n).toCircuit).unit g <;> simp }
  cls _ _ := Class.inference
  locIn _ _ := {()}
  winIn _ _ := ⟨0, n + 1⟩
  claimsIn _ _ := [.sizeAtMost 1, .inputAtMost [.past, .block, .world] 0]
  hiddenIff _ _ := by decide
  insideOut := fun _ i g hg => by
    rw [Partition.gates, mem_filter] at hg
    have hin : (placed n).isInput g = false := by
      by_contra hc
      simp only [Bool.not_eq_false] at hc
      have h2 := hg.2
      simp [Partition.whole, hc] at h2
    simp only [Placed.region, Circuit.computed, Finset.mem_filter, mem_univ, true_and,
      Finset.mem_coe, Window.ticks, Finset.mem_Ico]
    refine ⟨hin, by simp [placed], Nat.zero_le _, ?_⟩
    have := g.isLt
    simp only [placed, line] at this ⊢
    omega
  insideIn := fun _ u g hg => by
    rw [Partition.gates, mem_filter] at hg
    have hin : (placed n).isInput g = false := by
      by_contra hc
      simp only [Bool.not_eq_false] at hc
      have h2 := hg.2
      simp [Partition.whole, hc] at h2
    simp only [Placed.region, Circuit.computed, Finset.mem_filter, mem_univ, true_and,
      Finset.mem_coe, Window.ticks, Finset.mem_Ico]
    refine ⟨hin, by simp [placed], Nat.zero_le _, ?_⟩
    have := g.isLt
    simp only [placed, line] at this ⊢
    omega

/-- **The honest witness**: the one-gate circuit's run realizes the claims of `demoRuns 1`. -/
theorem honest_feasible : (demoRuns 1).Realizes () := by
  refine ⟨?_, ?_, fun _ _ => le_rfl, ?_, ?_⟩
  · intro a ha; exact absurd ha (Finset.notMem_empty a)
  · simp [demoRuns, Circuit.width]
  · intro u p hp
    fin_cases u
    simp only [demoRuns, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl
    · show (placed 1).work _ _ ≤ 1
      simp only [demoRuns, Runs.innerBlock, Partition.whole, Partition.gates, Placed.work]
      decide
    · show (placed 1).blockInput _ _ _ _ ≤ 0
      simp only [demoRuns, Runs.innerBlock, Placed.blockInput, Circuit.into, Circuit.width, placed, line]
      decide
  · intro i p hp
    fin_cases i
    simp only [demoRuns, List.mem_cons, List.not_mem_nil, or_false] at hp
    subst hp
    show (placed 1).work _ _ ≤ 1
    simp only [demoRuns, Runs.outerBlock, Partition.whole, Partition.gates, Placed.work]
    decide

/-- **A run outside the feasible set**: the two-gate circuit's run does not realize the claims of `demoRuns 2`,
whose one inference block is then of size at least 2. -/
theorem two_not_feasible : ¬ (demoRuns 2).Realizes () := by
  rintro ⟨-, -, -, hcl_in, -⟩
  have h := hcl_in ⟨0, by decide⟩ (.sizeAtMost 1) (by simp [demoRuns])
  simp only [Runs.HoldsIn, Runs.innerBlock, Placed.Holds, demoRuns, Partition.whole, Partition.gates,
    Placed.work] at h
  revert h
  decide

/-- **The test**: not every circuit's run is in the feasible set. -/
theorem not_every_feasible : ¬ ∀ n : ℕ, (demoRuns n).Realizes () :=
  fun h => two_not_feasible (h 2)

/-- **A run outside the feasible set with a tolerance**: the two-gate circuit's run does not realize the claims of
`demoRuns 2` even with an empty tolerance, so `not_every_feasible` is of the path the chain reads (`RealizesExcept`), not
only `Realizes`. Its inference block is of size at least 2 and holds no size-1 claim, and its free capacity is 0. -/
theorem two_not_realizesExcept : ¬ (demoRuns 2).RealizesExcept () ∅ :=
  fun h => two_not_feasible (realizesExcept_empty_realizes _ _ h)

/-- **The tolerance must be a term, not a quantifier** (the witness soundness asked for): at `M = all inference blocks`
the inference-claim clause of `RealizesExcept` is vacuous (`u ∉ univ` is false for every `u`), so it binds no inference
block's size, function or input. What then binds their work is the free-capacity clause the mend adds (the free blocks'
sizes and `M`'s inference blocks' sizes together at most `freeCap`), which `feasible_work_le` reads. Were the tolerance an
unbounded quantifier over `M` with only the claim clauses, a run whose inference blocks are as large as the server allows
would be in the feasible set: the old text fails on exactly this `M`. -/
theorem tolerance_term_binds (P : Runs L) (r : P.Run) :
    ∀ u, P.cls r u = Class.inference → u ∉ (Finset.univ : Finset (Fin (P.nIn r))) →
      ∀ p ∈ P.claimsIn r u, P.HoldsIn r u p :=
  fun u _ hu => absurd (Finset.mem_univ u) hu

/-- **No top block is in no class**: `kind` is total, so the degenerate case of a top block in neither the server-day nor
the hidden class cannot be stated. Nothing sits inside an inference block, the finest scale here, so neither can a block
inside an inference block that is not a proof block. -/
theorem top_blocks_classified (P : Runs L) (r : P.Run) (i : Fin (P.nOut r)) :
    P.kind r i = true ∨ P.kind r i = false := by cases P.kind r i <;> simp

/-! ## The chain, with the protocols' rows named as types

`rows_chain` is the chain with each protocol's row fed to `chain` as a typed hypothesis, not named in a docstring. Its
hypotheses are the rows (`DeclaredBlocksRun`; `PouwUseAtLeast` with its premises and the conversion `freeBlockAtMost` for
the free blocks' summed size; `PousUseAtLeast` and `PoubUseAtLeast` with their conversions, their exclusivity premise
open for memory and the link; `WipeInputAtMost`, `NetworkInputAtMost`, `NetworkOutputAtMost` with their premises;
`RegistrationBinds` with `RegistrationSound` and `RegisteredReadsOpenLog`; and the hardware description's capacities
`CapacityAtMost`). Its conclusion is the policy's: an accepted run's actual circuit is compliant except with the seven
rows' soundness errors summed. It is proved from `chain`.

One hypothesis is not a row: `assemble`, the bridge from the rows' per-run conclusions to the feasible set
(`RealizesExcept`). That bridge is the open content the summary records as unproved ("nothing from an accept to
policy-compliant is proved"); until the architecture page names the auditor's source for the wipe's input and that bridge
is argued, it is the one open premise. The deterministic rows carry no error of their own: `AttesterOnEveryLink` and the
capacities feed the error rows, and `RegistrationBinds` feeds `assemble`; they are hypotheses but add no term to the sum.
`NetworkInputAtMost`'s error is `0` (its bound is the covert channels' assumed figure, with no sampled proof). -/
theorem rows_chain (P : Runs L)
    (check : (P : Runs L) → P.Run → Bool) (pol : InferenceOnly.Policy)
    (hthm : InferenceOnly.PolicyTheorem check pol) (hcheck : ∀ r, P.accepts r → check P r = true)
    -- PoUW: use at least, in size, with the conversion to the free blocks' summed size.
    {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : Pouw.PearlC.CostModel Q R S) (Pr : Pouw.PearlC.Protocol Q R S) (TR : Pouw.PearlC.TileRules Q R S)
    (D : Pouw.PearlC.Layout → Prop) (Wd : Pouw.PearlC.Layout → ℕ → Pouw.PearlC.Decl → ℚ) (g : ℝ) (δ : ℕ → ℕ → ℝ)
    (salt : (r : P.Run) → Fin (P.nIn r) → ℕ) (shownPouw : P.Run → Prop)
    (bPouw freePouw capPouw enclosingPouw : (r : P.Run) → Fin (P.nOut r) → ℕ) (εpouw : ℝ≥0∞)
    (pouw : P.PouwUseAtLeast CM Pr TR D Wd g δ salt shownPouw bPouw εpouw)
    (hggap : Pouw.PearlC.GGapSampledDraw CM Pr TR D Wd g δ)
    (hpouwRead : P.RunReads (Pouw.PearlC.GGapSampledDraw CM Pr TR D Wd g δ) shownPouw εpouw)
    (hspend : P.SpendIsWork salt shownPouw bPouw)
    (hcapPouw : P.CapacityAtMost Capacity.computeRate enclosingPouw capPouw)
    (hexclPouw : P.DeclaredWorkExclusive freePouw
      (fun r i => ∑ u ∈ (P.infChildren r i).filter (fun u => salt r u ≤ (P.winIn r u).start), P.sizeIn r u)
      enclosingPouw)
    -- The declared blocks run.
    (f : (r : P.Run) → Fin (P.nIn r) → Circuit) (cDecl kDecl : (r : P.Run) → Fin (P.nIn r) → ℕ)
    (decl : (r : P.Run) → Finset (Fin (P.nIn r)) → ℕ) (tol : ℕ) (εdecl : ℝ≥0∞)
    (declared : P.DeclaredBlocksRun salt f cDecl kDecl decl tol εdecl)
    -- PoUS: use at least, over memory size, with its conversion (its exclusivity premise open for memory).
    (heldS bPous freePous capPous enclosingPous : (r : P.Run) → Fin (P.nOut r) → ℕ) (εpous : ℝ≥0∞)
    (pous : P.PousUseAtLeast heldS bPous εpous)
    (hcapPous : P.CapacityAtMost Capacity.memorySize enclosingPous capPous)
    (hexclPous : P.DeclaredWorkExclusive freePous heldS enclosingPous)
    -- PoUB: use at least, over the link, with its conversion (its exclusivity premise open for the link).
    (readsB bPoub freePoub capPoub enclosingPoub : (r : P.Run) → Fin (P.nOut r) → ℕ) (εpoub : ℝ≥0∞)
    (poub : P.PoubUseAtLeast readsB bPoub εpoub)
    (hcapPoub : P.CapacityAtMost Capacity.linkCapacity enclosingPoub capPoub)
    (hexclPoub : P.DeclaredWorkExclusive freePoub readsB enclosingPoub)
    -- The memory wipe.
    (Gwipe : Prop) (survives : P.Run → Prop) (Ewipe : (r : P.Run) → Fin (P.nOut r) → ℕ) (adv : ℝ≥0∞)
    (wipe : P.WipeInputAtMost Gwipe survives Ewipe adv) (hGwipe : Gwipe)
    (hwipeRead : P.RunReads Gwipe survives (2⁻¹ ^ 40 + adv))
    (hwipeInfo : P.WipeInformationGivesWidth survives (fun r i => Ewipe r i + 8600000000))
    -- Network attestation, input.
    (ingressN egressN bIn covert : (r : P.Run) → Fin (P.nOut r) → ℕ)
    (netIn : P.NetworkInputAtMost ingressN egressN bIn covert)
    (hatt : P.AttesterOnEveryLink ingressN egressN)
    (hingress : ∀ r, P.accepts r → ∀ i, P.kind r i = true → ingressN r i ≤ bIn r i)
    -- Network attestation, output.
    (shownNet : P.Run → Prop) (bOut ingressN' egressN' : (r : P.Run) → Fin (P.nOut r) → ℕ) (εnet : ℝ≥0∞)
    (netOut : P.NetworkOutputAtMost shownNet egressN bOut ingressN' egressN' εnet)
    (hphase : NetTiming.Attester.PhaseGetsBack)
    (hnetRead : P.RunReads NetTiming.Attester.PhaseGetsBack shownNet εnet)
    (hrecord : P.RecordIsOutput shownNet egressN bOut) (hatt' : P.AttesterOnEveryLink ingressN' egressN')
    -- Registration.
    {Root : Type} (log : Finset Root) (opensAt : (r : P.Run) → Fin (P.actual r).N → Root → Prop)
    (reg : P.RegistrationBinds log opensAt) (hregsound : Verity.Guarantees.RegistrationSound)
    (hregreads : P.RegisteredReadsOpenLog log opensAt)
    -- The one open premise: the rows' per-run conclusions assemble to a tolerance set the run realizes outside.
    (assemble : ∀ r, P.accepts r →
      (∀ i, P.kind r i = true → freePouw r i ≤ capPouw r i - bPouw r i) →
      (∃ M ⊆ (univ.filter fun u => P.cls r u = Class.inference ∧ salt r u ≤ (P.winIn r u).start),
        decl r M ≤ tol ∧
          ∀ u ∈ (univ.filter fun u => P.cls r u = Class.inference ∧ salt r u ≤ (P.winIn r u).start) \ M,
            P.HoldsIn r u (.function (f r u)) ∧ P.HoldsIn r u (.sizeAtMost (cDecl r u)) ∧
              P.HoldsIn r u (.inputAtMost [.past, .block, .world] (kDecl r u))) →
      (∀ i, P.kind r i = true → freePous r i ≤ capPous r i - bPous r i) →
      (∀ i, P.kind r i = true → freePoub r i ≤ capPoub r i - bPoub r i) →
      (∀ i, P.kind r i = true → P.HoldsOut r i (.inputAtMost [.past] (Ewipe r i + 8600000000))) →
      (∀ i, P.kind r i = true → P.HoldsOut r i (.inputAtMost [.world] (bIn r i + covert r i))) →
      (∀ i, P.kind r i = true → P.HoldsOut r i (.outputAtMost (bOut r i))) →
      (∀ g ∈ P.θ r, ∃ root ∈ log, opensAt r g root) →
      ∃ M, P.RealizesExcept r M) :
    ExceptWith P.pr P.accepts (fun r => InferenceOnly.Compliant pol (P.actual r) (P.θ r))
      (∑ x : Fin 7, (![εpouw, εdecl, εpous, εpoub, 2⁻¹ ^ 40 + adv, 0, εnet] : Fin 7 → ℝ≥0∞) x) := by
  set shownV : Fin 7 → P.Run → Prop :=
    ![ (fun r => ∀ i, P.kind r i = true → freePouw r i ≤ capPouw r i - bPouw r i),
       (fun r => ∃ M ⊆ (univ.filter fun u => P.cls r u = Class.inference ∧ salt r u ≤ (P.winIn r u).start),
         decl r M ≤ tol ∧
           ∀ u ∈ (univ.filter fun u => P.cls r u = Class.inference ∧ salt r u ≤ (P.winIn r u).start) \ M,
             P.HoldsIn r u (.function (f r u)) ∧ P.HoldsIn r u (.sizeAtMost (cDecl r u)) ∧
               P.HoldsIn r u (.inputAtMost [.past, .block, .world] (kDecl r u))),
       (fun r => ∀ i, P.kind r i = true → freePous r i ≤ capPous r i - bPous r i),
       (fun r => ∀ i, P.kind r i = true → freePoub r i ≤ capPoub r i - bPoub r i),
       (fun r => ∀ i, P.kind r i = true → P.HoldsOut r i (.inputAtMost [.past] (Ewipe r i + 8600000000))),
       (fun r => ∀ i, P.kind r i = true → P.HoldsOut r i (.inputAtMost [.world] (bIn r i + covert r i))),
       (fun r => ∀ i, P.kind r i = true → P.HoldsOut r i (.outputAtMost (bOut r i))) ] with hshownV
  set εV : Fin 7 → ℝ≥0∞ := ![εpouw, εdecl, εpous, εpoub, 2⁻¹ ^ 40 + adv, 0, εnet] with hεV
  have hcover : ∀ r, P.accepts r → (∀ x ∈ Finset.univ, shownV x r) → ∃ M, P.RealizesExcept r M := by
    intro r ha hall
    exact assemble r ha
      (hall 0 (Finset.mem_univ _)) (hall 1 (Finset.mem_univ _)) (hall 2 (Finset.mem_univ _))
      (hall 3 (Finset.mem_univ _)) (hall 4 (Finset.mem_univ _)) (hall 5 (Finset.mem_univ _))
      (hall 6 (Finset.mem_univ _)) (reg hregsound hregreads r ha)
  have hest : ∀ x ∈ Finset.univ, ExceptWith P.pr P.accepts (shownV x) (εV x) := by
    intro x _
    fin_cases x
    · exact (freeBlockAtMost P) Capacity.computeRate freePouw
        (fun r i => ∑ u ∈ (P.infChildren r i).filter (fun u => salt r u ≤ (P.winIn r u).start), P.sizeIn r u)
        enclosingPouw capPouw bPouw εpouw (pouw hggap hpouwRead hspend) hcapPouw hexclPouw
    · exact declared
    · exact (freeBlockAtMost P) Capacity.memorySize freePous heldS enclosingPous capPous bPous εpous
        pous hcapPous hexclPous
    · exact (freeBlockAtMost P) Capacity.linkCapacity freePoub readsB enclosingPoub capPoub bPoub εpoub
        poub hcapPoub hexclPoub
    · exact wipe hGwipe hwipeRead hwipeInfo
    · exact netIn hatt hingress
    · exact netOut hphase hnetRead hrecord hatt'
  exact chain P check pol Finset.univ shownV εV hthm (fun r ha => hcheck r ha) hcover hest

end ResourceAccounting.SecurityProofs.BlockProperties
