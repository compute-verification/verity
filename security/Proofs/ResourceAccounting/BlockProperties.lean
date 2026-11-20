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
  rintro ⟨h1, h2, h3, h4, h5, h6, -⟩
  refine ⟨h1, h2, h3, fun u => ?_, h6⟩
  by_cases hc : P.cls r u = Class.inference
  · exact h4 u hc (Finset.notMem_empty u)
  · exact h5 u hc

/-- **`Realizes` gives `RealizesExcept ∅`**: with no tolerance every block holds its claims, so the inference-claim
clauses split by class. The free-capacity bound is no longer a clause here (`FreeClaimHolds` carries it on its own). -/
theorem realizes_realizesExcept (P : Runs L) (r : P.Run) (h : P.Realizes r) : P.RealizesExcept r ∅ := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  exact ⟨h1, h2, h3, fun u _ _ => h4 u, fun u _ => h4 u, h5, fun u hu => absurd hu (Finset.notMem_empty u)⟩

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

/-- **The check bites, on the path the chain reads**: under a run that passes the inference-only check and realizes its
claims outside a tolerance set `M` (`RealizesExcept`, not `Realizes`), with each server-day block's own capacity
`serverCap i` and the assumed hidden hardware size `lim.hidden`, the whole circuit's work is at most, for each server-day
block, the smaller of its capacity and the count of its inference blocks **outside** `M` at the size cap `lim.c` plus the
free-block limit `lim.freeSize`. The bound is in the limits alone: the free blocks and `M`'s inference blocks sit inside
the committed claim `P.claimedFree r i` (the hypothesis `FreeClaimHolds`), which the check caps at `lim.freeSize` (its
fourth conjunct). The tolerance `M` is a bounded term here, not an unbounded quantifier: `M`'s blocks are counted inside
the free limit, not as inference credit. -/
theorem feasible_work_le (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (serverCap : Fin (P.nOut r) → ℕ)
    (hc : InferenceOnly.check lim P r = true) (hreal : P.RealizesExcept r M)
    (hfree : P.FreeClaimHolds r M)
    (hserver : ∀ i, P.kind r i = true → P.sizeOut r i ≤ serverCap i)
    (hhidden : ∑ i ∈ univ.filter (fun i => P.kind r i = false), P.sizeOut r i ≤ lim.hidden) :
    (P.actual r).work (P.π r) (P.actual r).computed ≤
      (∑ i ∈ univ.filter (fun i => P.kind r i = true),
        min (serverCap i) (((P.infChildren r i).filter (fun u => u ∉ M)).card * lim.c + lim.freeSize)) +
        lim.hidden := by
  simp only [InferenceOnly.check, Bool.and_eq_true, decide_eq_true_eq] at hc
  obtain ⟨⟨⟨⟨⟨-, hinf⟩, -⟩, hfcCheck⟩, -⟩, -⟩ := hc
  obtain ⟨-, -, -, hcl_out, -, -, -⟩ := hreal
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
  have hfc := hfree i hki
  have hfs := hfcCheck i hki
  omega

/-- **The check bites, with no tolerance set in the bound**: the same bound counting every inference block (not only those
outside `M`), so the conclusion names no `M`. From `feasible_work_le` by monotonicity (`(infChildren i).filter (· ∉ M) ⊆
infChildren i`). The count of inference blocks the bound reads is capped by `lim.n` in the check. -/
theorem feasible_work_le_total (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (serverCap : Fin (P.nOut r) → ℕ)
    (hc : InferenceOnly.check lim P r = true) (hreal : P.RealizesExcept r M)
    (hfree : P.FreeClaimHolds r M)
    (hserver : ∀ i, P.kind r i = true → P.sizeOut r i ≤ serverCap i)
    (hhidden : ∑ i ∈ univ.filter (fun i => P.kind r i = false), P.sizeOut r i ≤ lim.hidden) :
    (P.actual r).work (P.π r) (P.actual r).computed ≤
      (∑ i ∈ univ.filter (fun i => P.kind r i = true),
        min (serverCap i) ((P.infChildren r i).card * lim.c + lim.freeSize)) + lim.hidden := by
  refine le_trans (feasible_work_le lim P r M serverCap hc hreal hfree hserver hhidden) (add_le_add ?_ le_rfl)
  refine Finset.sum_le_sum fun i _ => min_le_min le_rfl (add_le_add (Nat.mul_le_mul_right _ ?_) le_rfl)
  exact Finset.card_le_card (Finset.filter_subset _ _)

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
/-- One run of the circuit `placed n` at a free-block capacity `fc`: no fresh input, one server-day block holding the
whole circuit as one inference block, claimed of size at most 1 and 0 fresh input, with a size claim and an input claim on
the server-day block (so the check's two server-day conjuncts can bite). Its window covers every tick of `placed n`, so
its one block is its region (`insideOut`), and its committed free-capacity claim is `fc`. -/
noncomputable def demoRuns (n : ℕ) (fc : ℕ) : Runs Unit where
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
  claimsOut _ _ := [.sizeAtMost n, .inputAtMost [.world] 0]
  claimedFree _ _ := fc
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

/-- **The honest witness**: the one-gate circuit's run realizes the claims of `demoRuns 1 0`. -/
theorem honest_feasible : (demoRuns 1 0).Realizes () := by
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
    rcases hp with rfl | rfl
    · show (placed 1).work _ _ ≤ 1
      simp only [demoRuns, Runs.outerBlock, Partition.whole, Partition.gates, Placed.work]
      decide
    · show (placed 1).blockInput _ _ _ _ ≤ 0
      simp only [demoRuns, Runs.outerBlock, Placed.blockInput, Circuit.into, Circuit.width, placed, line]
      decide

/-- The free blocks of `demoRuns n fc` are none (its one inner block is an inference block). -/
theorem demo_freeChildren_empty (n fc : ℕ) (i : Fin ((demoRuns n fc).nOut ())) :
    (demoRuns n fc).freeChildren () i = ∅ := by
  simp only [Runs.freeChildren]
  apply Finset.filter_false_of_mem
  intro u _
  rw [show (demoRuns n fc).cls () u = Class.inference from rfl]
  exact fun h => absurd h.2 (by decide)

/-- **A run outside the feasible set**: the two-gate circuit's run does not realize the claims of `demoRuns 2 0`,
whose one inference block is then of size at least 2. -/
theorem two_not_feasible : ¬ (demoRuns 2 0).Realizes () := by
  rintro ⟨-, -, -, hcl_in, -⟩
  have h := hcl_in ⟨0, by decide⟩ (.sizeAtMost 1) (by simp [demoRuns])
  simp only [Runs.HoldsIn, Runs.innerBlock, Placed.Holds, demoRuns, Partition.whole, Partition.gates,
    Placed.work] at h
  revert h
  decide

/-- **The test**: not every circuit's run is in the feasible set. -/
theorem not_every_feasible : ¬ ∀ n : ℕ, (demoRuns n 0).Realizes () :=
  fun h => two_not_feasible (h 2)

/-- **A run outside the feasible set with a tolerance**: the two-gate circuit's run does not realize the claims of
`demoRuns 2 0` even with an empty tolerance, so `not_every_feasible` is of the path the chain reads (`RealizesExcept`),
not only `Realizes`. -/
theorem two_not_realizesExcept : ¬ (demoRuns 2 0).RealizesExcept () ∅ :=
  fun h => two_not_feasible (realizesExcept_empty_realizes _ _ h)

/-- **The honest run passes the check and realizes at the empty tolerance**: `demoRuns 1 0` at honest limits (one
inference block of size and input at most 1 and 0, no free block, committed free claim 0) passes `check` and is in the
feasible set with no tolerance. -/
theorem honest_checks :
    InferenceOnly.check ⟨1, 1, 0, 0, 0, 0, 0, 0⟩ (demoRuns 1 0) () = true ∧
      (demoRuns 1 0).RealizesExcept () ∅ :=
  ⟨by decide, realizes_realizesExcept _ _ honest_feasible⟩

/-- **The excluded run as a theorem** (universal, over any `Runs`): a run with a server-day block `i` whose inference
children in the tolerance set `M` are together larger than `lim.freeSize` cannot both pass `check` and have its committed
free-capacity claim hold (`FreeClaimHolds`) — for every choice of the run's other fields. The check reads the committed
claim and caps it at `lim.freeSize`; the claim bounds the free and tolerance work by that number; so the tolerance's work
is capped. With `freeCap` a free field (not a committed claim the check reads) no such theorem is provable, since the field
could be set as large as wanted: this universal statement is the test of the mend. -/
theorem excluded_run (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (i : Fin (P.nOut r)) (hki : P.kind r i = true)
    (hbig : lim.freeSize < ∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u) :
    ¬ (InferenceOnly.check lim P r = true ∧ P.FreeClaimHolds r M) := by
  rintro ⟨hc, hclaim⟩
  simp only [InferenceOnly.check, Bool.and_eq_true, decide_eq_true_eq] at hc
  obtain ⟨⟨⟨⟨⟨-, -⟩, -⟩, hfcCheck⟩, -⟩, -⟩ := hc
  have h1 := hclaim i hki
  have h2 := hfcCheck i hki
  omega

/-- **The demo instance of the excluded run**: the two-gate circuit at `freeSize = 0` with its one size-2 inference block
in `M` cannot pass the check with its claim holding, for any committed claim value `fc` — the committed claim is what the
check reads, so no choice of it rescues the run. -/
theorem excluded_run_demo (fc : ℕ) (M : Finset (Fin ((demoRuns 2 fc).nIn ())))
    (hM : (⟨0, Nat.one_pos⟩ : Fin ((demoRuns 2 fc).nIn ())) ∈ M) :
    ¬ (InferenceOnly.check ⟨1, 1, 0, 0, 0, 0, 0, 0⟩ (demoRuns 2 fc) () = true ∧
        (demoRuns 2 fc).FreeClaimHolds () M) := by
  refine excluded_run _ _ _ M ⟨0, Nat.one_pos⟩ rfl ?_
  have hmem : (⟨0, Nat.one_pos⟩ : Fin ((demoRuns 2 fc).nIn ())) ∈
      ((demoRuns 2 fc).infChildren () ⟨0, Nat.one_pos⟩).filter (· ∈ M) := by
    rw [Finset.mem_filter]
    refine ⟨?_, hM⟩
    simp only [Runs.infChildren, Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨rfl, rfl⟩
  show (0 : ℕ) < ∑ u ∈ ((demoRuns 2 fc).infChildren () ⟨0, Nat.one_pos⟩).filter (· ∈ M),
    (demoRuns 2 fc).sizeIn () u
  calc (0 : ℕ) < (demoRuns 2 fc).sizeIn () ⟨0, Nat.one_pos⟩ := by
        show (0 : ℕ) < (placed 2).work (fun _ => 1)
          ((Partition.whole (placed 2).toCircuit).gates ⟨0, Nat.one_pos⟩)
        decide
    _ ≤ _ := Finset.single_le_sum (fun j _ => Nat.zero_le _) hmem

/-- **The committed claim is what binds**: the same two-gate run with a large committed free claim (`demoRuns 2 2`) is in
the feasible set at `M = all inference blocks` and its claim holds (`FreeClaimHolds`); the check's verdict on it turns only
on its claim conjunct — with `freeSize = 0` the check is false, and with `freeSize = 2` (the only limit changed) it is
true. The committed claim, not a free field, is the number the check reads. -/
theorem claim_binds :
    (demoRuns 2 2).RealizesExcept () Finset.univ ∧ (demoRuns 2 2).FreeClaimHolds () Finset.univ ∧
      InferenceOnly.check ⟨1, 1, 0, 0, 0, 0, 0, 0⟩ (demoRuns 2 2) () = false ∧
      InferenceOnly.check ⟨1, 1, 0, 0, 2, 0, 0, 0⟩ (demoRuns 2 2) () = true := by
  refine ⟨⟨?_, ?_, fun _ _ => le_rfl, ?_, ?_, ?_, ?_⟩, ?_, by decide, by decide⟩
  · intro a ha; exact absurd ha (Finset.notMem_empty a)
  · simp [demoRuns, Circuit.width]
  · intro u _ hu; exact absurd (Finset.mem_univ u) hu
  · intro u hu; exact absurd (by simp [demoRuns] : (demoRuns 2 2).cls () u = Class.inference) hu
  · intro i p hp
    fin_cases i
    simp only [demoRuns, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl
    · show (placed 2).work _ _ ≤ 2
      simp only [demoRuns, Runs.outerBlock, Partition.whole, Partition.gates, Placed.work]; decide
    · show (placed 2).blockInput _ _ _ _ ≤ 0
      simp only [demoRuns, Runs.outerBlock, Placed.blockInput, Circuit.into, Circuit.width, placed, line]
      decide
  · intro u _; simp [demoRuns]
  · intro i _
    rw [demo_freeChildren_empty, Finset.sum_empty, zero_add,
      Finset.filter_true_of_mem (fun u _ => Finset.mem_univ u)]
    fin_cases i
    decide

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

One hypothesis is not a row: `assemble`, the bridge from the rows' per-run conclusions to the feasible set. Its conclusion
is split by name: `FreeClaimHolds r M` (the committed free-capacity claim bounds the free and tolerance work, the part the
conversion gives — capacity minus the use credited outside `M`) and `RealizesExcept r M` (the rest). That bridge is the
open content the summary records as unproved ("nothing from an accept to policy-compliant is proved"); until the
architecture page names the auditor's source for the wipe's input and that bridge is argued, it is the one open premise. The deterministic rows carry no error of their own: `AttesterOnEveryLink` and the
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
    (dsize : (r : P.Run) → Fin (P.nIn r) → ℕ) (tol : ℕ) (εdecl : ℝ≥0∞)
    (declared : P.DeclaredBlocksRun salt f cDecl kDecl dsize tol εdecl)
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
    -- The memory wipe (with its memory-size capacity hypothesis).
    (Gwipe : Prop) (survives : P.Run → Prop) (Ewipe memEnclosing memCap : (r : P.Run) → Fin (P.nOut r) → ℕ)
    (adv : ℝ≥0∞)
    (wipe : P.WipeInputAtMost Gwipe survives Ewipe memEnclosing memCap adv) (hGwipe : Gwipe)
    (hwipeRead : P.RunReads Gwipe survives (2⁻¹ ^ 40 + adv))
    (hwipeCap : P.CapacityAtMost Capacity.memorySize memEnclosing memCap)
    (hwipeInfo : P.WipeInformationGivesWidth survives (fun r i => Ewipe r i + 8600000000))
    -- Network attestation, input (covert inside the attester's assumption).
    (ingressN egressN bIn covert : (r : P.Run) → Fin (P.nOut r) → ℕ)
    (netIn : P.NetworkInputAtMost ingressN egressN bIn covert)
    (hatt : P.AttesterOnEveryLink ingressN egressN covert)
    (hingress : ∀ r, P.accepts r → ∀ i, P.kind r i = true → ingressN r i ≤ bIn r i)
    -- Network attestation, output.
    (shownNet : P.Run → Prop) (bOut ingressN' egressN' covert' : (r : P.Run) → Fin (P.nOut r) → ℕ) (εnet : ℝ≥0∞)
    (netOut : P.NetworkOutputAtMost shownNet egressN bOut ingressN' egressN' covert' εnet)
    (hphase : NetTiming.Attester.PhaseGetsBack)
    (hnetRead : P.RunReads NetTiming.Attester.PhaseGetsBack shownNet εnet)
    (hrecord : P.RecordIsOutput shownNet egressN bOut) (hatt' : P.AttesterOnEveryLink ingressN' egressN' covert')
    -- Registration.
    {Root : Type} (log : Finset Root) (opensAt : (r : P.Run) → Fin (P.actual r).N → Root → Prop)
    (reg : P.RegistrationBinds log opensAt) (hregsound : Verity.Guarantees.RegistrationSound)
    (hregreads : P.RegisteredReadsOpenLog log opensAt)
    -- The one open premise: the rows' per-run conclusions assemble to a tolerance set the run realizes outside.
    (assemble : ∀ r, P.accepts r →
      (∀ i, P.kind r i = true → freePouw r i ≤ capPouw r i - bPouw r i) →
      (∃ M ⊆ (univ.filter fun u => P.cls r u = Class.inference ∧ salt r u ≤ (P.winIn r u).start),
        (∑ u ∈ M, dsize r u) ≤ tol ∧
          ∀ u ∈ (univ.filter fun u => P.cls r u = Class.inference ∧ salt r u ≤ (P.winIn r u).start) \ M,
            P.HoldsIn r u (.function (f r u)) ∧ P.HoldsIn r u (.sizeAtMost (cDecl r u)) ∧
              P.HoldsIn r u (.inputAtMost [.past, .block, .world] (kDecl r u))) →
      (∀ i, P.kind r i = true → freePous r i ≤ capPous r i - bPous r i) →
      (∀ i, P.kind r i = true → freePoub r i ≤ capPoub r i - bPoub r i) →
      (∀ i, P.kind r i = true → P.HoldsOut r i (.inputAtMost [.past] (Ewipe r i + 8600000000))) →
      (∀ i, P.kind r i = true → P.HoldsOut r i (.inputAtMost [.world] (bIn r i + covert r i))) →
      (∀ i, P.kind r i = true → P.HoldsOut r i (.outputAtMost (bOut r i))) →
      (∀ g ∈ P.θ r, ∃ root ∈ log, opensAt r g root) →
      ∃ M, P.FreeClaimHolds r M ∧ P.RealizesExcept r M) :
    ExceptWith P.pr P.accepts (fun r => InferenceOnly.Compliant pol (P.actual r) (P.θ r))
      (∑ x : Fin 7, (![εpouw, εdecl, εpous, εpoub, 2⁻¹ ^ 40 + adv, 0, εnet] : Fin 7 → ℝ≥0∞) x) := by
  set shownV : Fin 7 → P.Run → Prop :=
    ![ (fun r => ∀ i, P.kind r i = true → freePouw r i ≤ capPouw r i - bPouw r i),
       (fun r => ∃ M ⊆ (univ.filter fun u => P.cls r u = Class.inference ∧ salt r u ≤ (P.winIn r u).start),
         (∑ u ∈ M, dsize r u) ≤ tol ∧
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
    obtain ⟨M, -, hreal⟩ := assemble r ha
      (hall 0 (Finset.mem_univ _)) (hall 1 (Finset.mem_univ _)) (hall 2 (Finset.mem_univ _))
      (hall 3 (Finset.mem_univ _)) (hall 4 (Finset.mem_univ _)) (hall 5 (Finset.mem_univ _))
      (hall 6 (Finset.mem_univ _)) (reg hregsound hregreads r ha)
    exact ⟨M, hreal⟩
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
    · exact wipe hGwipe hwipeRead hwipeCap hwipeInfo
    · exact netIn hatt hingress
    · exact netOut hphase hnetRead hrecord hatt'
  exact chain P check pol Finset.univ shownV εV hthm (fun r ha => hcheck r ha) hcover hest

end ResourceAccounting.SecurityProofs.BlockProperties
