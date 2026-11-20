module

public import Definitions.Compose.Guarantees.Blocks
public import Definitions.Registration.Physical
import Mathlib.Tactic

/-!
# The conversion, sizes over the partitions, the chain, and the feasible set is not everything

`freeBlockAtMost` proves `FreeBlockAtMost`. `sizeAdds` proves that the blocks directly inside a server-day block have
sizes adding to at most it, and `sizeExclusive` that a server-day block's free and inference children's work do not
double-count. `conversion_le` is the conversion for size as one inequality, `freeClaimHolds_of_conversion` gives
`FreeClaimHolds` from it, and `conversion_bites` shows it tight at one gate. `conversion_pooled_le` is the same
conversion summed over a run's server-day blocks with the tolerance counted once, `freeClaimHoldsPooled_of_conversion`
gives `FreeClaimHoldsPooled` from it, and `pooled_counts_tol_once` shows the pooled bound below the per-block bounds
summed, on a run with two server-day blocks. `listed_gate_in_serverDay` puts every
computed gate at a listed location in a server-day block, `hidden_work_le` bounds hidden hardware's work by the work at
unlisted locations, and `capacityAtMost_compute` gives the compute capacity from `ComputeRate` over the listed
locations. `chain` proves `Chain`.

The policy's check bites: `feasible_work_le` bounds a realizing run's total work by, for each server-day block, the
smaller of the server-day size limit and its inference and free blocks' limits, plus the assumed hidden hardware size;
`serverInput_le` bounds a server-day block's input from its own past and the outside world by the input limit, and
`input_claim_binds` shows that limit refusing a claim of `2^60` bits.

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

/-- **The compute capacity from the hardware description**: where every run meets `ComputeRate` over the listed locations,
every server-day block's size is at most its location's rate `Φ` times its window's length, its location being listed
(`kindIff`). -/
theorem capacityAtMost_compute (P : Runs L) (Φ : Finset L → ℕ)
    (hcr : ∀ r, (P.actual r).ComputeRate P.listed (P.π r) Φ) :
    P.CapacityAtMost Capacity.computeRate P.sizeOut (fun r i => Φ (P.locOut r i) * (P.winOut r i).len) :=
  fun r i hki => sizeOut_le_compute P r i P.listed Φ (hcr r) ((P.kindIff r i).mp hki)

/-- **Every computed gate at a listed location lies in a server-day block**: the partition puts it in some top block,
whose gates lie at its location (`insideOut`), and a hidden block's location shares no leaf with a listed one. -/
theorem listed_gate_in_serverDay (P : Runs L) (r : P.Run) (g : Fin (P.actual r).N)
    (hg : g ∈ (P.actual r).computed) (X : Finset L) (hX : X ∈ P.listed) (hx : (P.actual r).x g ∈ X) :
    ∃ i, P.kind r i = true ∧ g ∈ (P.outer r).gates i := by
  rw [(P.outer r).computed_eq_biUnion] at hg
  obtain ⟨i, -, hgi⟩ := Finset.mem_biUnion.mp hg
  refine ⟨i, ?_, hgi⟩
  cases hk : P.kind r i with
  | true => rfl
  | false =>
    have hreg := P.insideOut r i hgi
    simp only [Placed.region, Finset.mem_filter] at hreg
    exact absurd hx (Finset.disjoint_left.mp (P.hiddenUnlisted r i hk X hX) hreg.2.1)

/-- **Hidden hardware's work is the work at unlisted locations**: the hidden blocks' sizes add up to at most the work of
the gates no listed location holds. -/
theorem hidden_work_le (P : Runs L) (r : P.Run) :
    ∑ i ∈ univ.filter (fun i => P.kind r i = false), P.sizeOut r i ≤ (P.actual r).work (P.π r) (P.hiddenGates r) := by
  unfold Runs.sizeOut Placed.work
  rw [← Finset.sum_biUnion (fun a _ b _ hab => (P.outer r).gates_disjoint a b hab)]
  refine Finset.sum_le_sum_of_subset fun g hg => ?_
  obtain ⟨i, hi, hgi⟩ := Finset.mem_biUnion.mp hg
  have hk : P.kind r i = false := (Finset.mem_filter.mp hi).2
  have hreg := P.insideOut r i hgi
  simp only [Placed.region, Finset.mem_filter] at hreg
  simp only [Runs.hiddenGates, Finset.mem_filter]
  exact ⟨hreg.1, fun X hX hx => Finset.disjoint_left.mp (P.hiddenUnlisted r i hk X hX) hreg.2.1 hx⟩

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

/-- In one run, inside a server-day block `i`: the free blocks' work, `M`'s inference blocks' work and the work of the
inference blocks after the salt and outside `M` add up to at most `i`'s size. -/
theorem sizeExclusive_run (P : Runs L) (salt : (r : P.Run) → Fin (P.nIn r) → ℕ) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (i : Fin (P.nOut r)) (hki : P.kind r i = true) :
    (∑ u ∈ P.freeChildren r i, P.sizeIn r u) + (∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u) +
      ∑ u ∈ (P.infChildren r i).filter (fun u => salt r u ≤ (P.winIn r u).start ∧ u ∉ M), P.sizeIn r u
      ≤ P.sizeOut r i := by
  rw [sizeOut_eq_children_sum, children_split P r i hki, Finset.sum_union (infFree_disjoint P r i),
    ← Finset.sum_filter_add_sum_filter_not (P.infChildren r i) (· ∈ M) (P.sizeIn r)]
  have hsub : ∑ u ∈ (P.infChildren r i).filter (fun u => salt r u ≤ (P.winIn r u).start ∧ u ∉ M), P.sizeIn r u
      ≤ ∑ u ∈ (P.infChildren r i).filter (fun u => ¬ u ∈ M), P.sizeIn r u :=
    Finset.sum_le_sum_of_subset fun u hu => by
      simp only [Finset.mem_filter] at hu ⊢
      exact ⟨hu.1, hu.2.2⟩
  omega

/-- **Declared work does not double as other work, for size**: inside a server-day block, for every set `M`, the free
blocks' and `M`'s inference blocks' work and the work of the inference blocks after the salt and outside `M` add up to
at most its own size. -/
theorem sizeExclusive (P : Runs L) (salt : (r : P.Run) → Fin (P.nIn r) → ℕ) :
    P.DeclaredWorkExclusive
      (fun r M i => (∑ u ∈ P.freeChildren r i, P.sizeIn r u) +
        ∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u)
      (fun r M i => ∑ u ∈ (P.infChildren r i).filter (fun u => salt r u ≤ (P.winIn r u).start ∧ u ∉ M),
        P.sizeIn r u)
      P.sizeOut :=
  fun r _ M i hki => sizeExclusive_run P salt r M i hki

/-! ## The conversion -/

/-- The conversion's arithmetic for one block: use at least `b`, capacity at least the enclosing quantity, and the free
quantity and the use within it give the free quantity at most `cap − b`. -/
theorem freeBlockAtMost_run {free use enclosing cap b : ℕ} (huse : b ≤ use) (hcap : enclosing ≤ cap)
    (hex : free + use ≤ enclosing) : free ≤ cap - b := by
  omega

theorem freeBlockAtMost (P : Runs L) : P.FreeBlockAtMost := by
  intro _what free use b enclosing cap ε huse hcap hex
  refine le_trans (P.mono _ _ ?_) huse
  rintro r ⟨ha, hn⟩
  exact ⟨ha, fun hb => hn fun M i hi => freeBlockAtMost_run (hb M i hi) (hcap r i hi) (hex r ha M i hi)⟩

/-- `M`'s inference blocks inside one server-day block have committed size claims summing to at most `M`'s. -/
theorem tolChildren_le (P : Runs L) (r : P.Run) (M : Finset (Fin (P.nIn r))) (i : Fin (P.nOut r)) :
    ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤ ∑ u ∈ M, claimedSize (P.claimsIn r u) :=
  Finset.sum_le_sum_of_subset fun _ hu => (Finset.mem_filter.mp hu).2

/-- **The conversion, as one inequality.** In one run, inside a server-day block `i`, for a set `M` of `inner`'s units:
from PoUW's conclusion for `M` (`huse`: `b` less the committed size claims of `M`'s inference blocks in `i` is at most
the work of `i`'s inference blocks after the salt and outside `M`), `i`'s size at most `cap` (`hcap`) and `M`'s committed
size claims at most `tol` (`htol`), the free blocks' work, `M`'s inference blocks' work in `i` and `b` together are at
most `cap + tol`. All in gates at the run's pricing; no subtraction enters the conclusion, so no case is truncated. The
partitions give the rest (`sizeExclusive_run`). -/
theorem conversion_le (P : Runs L) (salt : (r : P.Run) → Fin (P.nIn r) → ℕ) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (i : Fin (P.nOut r)) (hki : P.kind r i = true) (b cap tol : ℕ)
    (huse : b - ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤
      ∑ u ∈ (P.infChildren r i).filter (fun u => salt r u ≤ (P.winIn r u).start ∧ u ∉ M), P.sizeIn r u)
    (hcap : P.sizeOut r i ≤ cap) (htol : ∑ u ∈ M, claimedSize (P.claimsIn r u) ≤ tol) :
    (∑ u ∈ P.freeChildren r i, P.sizeIn r u) + (∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u) + b ≤
      cap + tol := by
  have hex := sizeExclusive_run P salt r M i hki
  have hfree := freeBlockAtMost_run huse hcap hex
  have hM := tolChildren_le P r M i
  omega

/-- **The conversion in the form the check's claim reads**: the free blocks' and `M`'s inference blocks' work in `i` is
at most `cap − b + tol` (from `conversion_le`; `cap + tol − b ≤ cap − b + tol` over the naturals). -/
theorem conversion_le_sub (P : Runs L) (salt : (r : P.Run) → Fin (P.nIn r) → ℕ) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (i : Fin (P.nOut r)) (hki : P.kind r i = true) (b cap tol : ℕ)
    (huse : b - ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤
      ∑ u ∈ (P.infChildren r i).filter (fun u => salt r u ≤ (P.winIn r u).start ∧ u ∉ M), P.sizeIn r u)
    (hcap : P.sizeOut r i ≤ cap) (htol : ∑ u ∈ M, claimedSize (P.claimsIn r u) ≤ tol) :
    (∑ u ∈ P.freeChildren r i, P.sizeIn r u) + (∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u) ≤
      cap - b + tol := by
  have := conversion_le P salt r M i hki b cap tol huse hcap htol
  omega

/-- **The committed free claim holds from the conversion**: in one run, PoUW's conclusion for `M` in every server-day
block, every server-day block's size at most `cap`, `M`'s committed size claims at most `tol`, and every server-day
block's committed free claim at least `cap − b + tol` give `FreeClaimHolds r M`. -/
theorem freeClaimHolds_of_conversion (P : Runs L) (salt : (r : P.Run) → Fin (P.nIn r) → ℕ) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (b cap : Fin (P.nOut r) → ℕ) (tol : ℕ)
    (huse : ∀ i, P.kind r i = true → b i - ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤
      ∑ u ∈ (P.infChildren r i).filter (fun u => salt r u ≤ (P.winIn r u).start ∧ u ∉ M), P.sizeIn r u)
    (hcap : ∀ i, P.kind r i = true → P.sizeOut r i ≤ cap i)
    (htol : ∑ u ∈ M, claimedSize (P.claimsIn r u) ≤ tol)
    (hclaim : ∀ i, P.kind r i = true → cap i - b i + tol ≤ P.claimedFree r i) : P.FreeClaimHolds r M :=
  fun i hki => (conversion_le_sub P salt r M i hki (b i) (cap i) tol (huse i hki) (hcap i hki) htol).trans
    (hclaim i hki)

/-- **The conversion excludes**: in one run, where the free blocks' work, `M`'s inference blocks' work in `i` and `b`
exceed `cap + tol`, one of PoUW's conclusion for `M`, the capacity and the tolerance fails. -/
theorem conversion_excludes (P : Runs L) (salt : (r : P.Run) → Fin (P.nIn r) → ℕ) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (i : Fin (P.nOut r)) (hki : P.kind r i = true) (b cap tol : ℕ)
    (hbig : cap + tol < (∑ u ∈ P.freeChildren r i, P.sizeIn r u) +
      (∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u) + b) :
    ¬ ((b - ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤
        ∑ u ∈ (P.infChildren r i).filter (fun u => salt r u ≤ (P.winIn r u).start ∧ u ∉ M), P.sizeIn r u) ∧
      P.sizeOut r i ≤ cap ∧ ∑ u ∈ M, claimedSize (P.claimsIn r u) ≤ tol) := by
  rintro ⟨huse, hcap, htol⟩
  have := conversion_le P salt r M i hki b cap tol huse hcap htol
  omega

/-! ## The conversion, pooled over the run -/

/-- `M`'s inference blocks inside the blocks of any set `S` of `outer`'s blocks have committed size claims summing to at
most `M`'s: each unit of `inner` has one parent, so no unit is counted in two of them. -/
theorem tolChildren_sum_le (P : Runs L) (r : P.Run) (M : Finset (Fin (P.nIn r))) (S : Finset (Fin (P.nOut r))) :
    ∑ i ∈ S, ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤
      ∑ u ∈ M, claimedSize (P.claimsIn r u) := by
  rw [← Finset.sum_biUnion]
  · refine Finset.sum_le_sum_of_subset fun u hu => ?_
    obtain ⟨i, -, hu⟩ := Finset.mem_biUnion.mp hu
    exact (Finset.mem_filter.mp hu).2
  · intro i _ j _ hij
    rw [Function.onFun, Finset.disjoint_left]
    intro u hi hj
    simp only [Runs.infChildren, Finset.mem_filter, Finset.mem_univ, true_and] at hi hj
    exact hij (hi.1.1.symm.trans hj.1.1)

/-- **The conversion, pooled over the run.** In one run, for a set `M` of `inner`'s units: from PoUW's conclusion for `M`
in every server-day block (`huse`), every server-day block's size at most its `cap` (`hcap`) and `M`'s committed size
claims at most `tol` (`htol`), the sum over the server-day blocks of the free blocks' work, `M`'s inference blocks' work
and `b` is at most the server-day blocks' `cap` summed plus `tol`, once. All in gates at the run's pricing. -/
theorem conversion_pooled_le (P : Runs L) (salt : (r : P.Run) → Fin (P.nIn r) → ℕ) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (b cap : Fin (P.nOut r) → ℕ) (tol : ℕ)
    (huse : ∀ i, P.kind r i = true → b i - ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤
      ∑ u ∈ (P.infChildren r i).filter (fun u => salt r u ≤ (P.winIn r u).start ∧ u ∉ M), P.sizeIn r u)
    (hcap : ∀ i, P.kind r i = true → P.sizeOut r i ≤ cap i)
    (htol : ∑ u ∈ M, claimedSize (P.claimsIn r u) ≤ tol) :
    ∑ i ∈ univ.filter (fun i => P.kind r i = true),
        ((∑ u ∈ P.freeChildren r i, P.sizeIn r u) + (∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u) + b i)
      ≤ (∑ i ∈ univ.filter (fun i => P.kind r i = true), cap i) + tol := by
  have hper : ∀ i ∈ univ.filter (fun i => P.kind r i = true),
      (∑ u ∈ P.freeChildren r i, P.sizeIn r u) + (∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u) + b i ≤
        cap i + ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) := by
    intro i hi
    have hki := (Finset.mem_filter.mp hi).2
    have hex := sizeExclusive_run P salt r M i hki
    have h1 := huse i hki
    have h2 := hcap i hki
    omega
  calc _ ≤ ∑ i ∈ univ.filter (fun i => P.kind r i = true),
        (cap i + ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u)) := Finset.sum_le_sum hper
    _ = (∑ i ∈ univ.filter (fun i => P.kind r i = true), cap i) +
        ∑ i ∈ univ.filter (fun i => P.kind r i = true),
          ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) := Finset.sum_add_distrib
    _ ≤ _ := Nat.add_le_add_left ((tolChildren_sum_le P r M _).trans htol) _

/-- **The pooled conversion in the form a pooled claim reads**: the server-day blocks' free blocks' work and `M`'s
inference blocks' work, summed over the run, are at most the server-day blocks' `cap` summed less their `b` summed, plus
`tol` (gates). No side condition: over the naturals `C + tol − B ≤ (C − B) + tol`. -/
theorem conversion_pooled_le_sub (P : Runs L) (salt : (r : P.Run) → Fin (P.nIn r) → ℕ) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (b cap : Fin (P.nOut r) → ℕ) (tol : ℕ)
    (huse : ∀ i, P.kind r i = true → b i - ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤
      ∑ u ∈ (P.infChildren r i).filter (fun u => salt r u ≤ (P.winIn r u).start ∧ u ∉ M), P.sizeIn r u)
    (hcap : ∀ i, P.kind r i = true → P.sizeOut r i ≤ cap i)
    (htol : ∑ u ∈ M, claimedSize (P.claimsIn r u) ≤ tol) :
    ∑ i ∈ univ.filter (fun i => P.kind r i = true),
        ((∑ u ∈ P.freeChildren r i, P.sizeIn r u) + ∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u)
      ≤ (∑ i ∈ univ.filter (fun i => P.kind r i = true), cap i) -
          (∑ i ∈ univ.filter (fun i => P.kind r i = true), b i) + tol := by
  have h := conversion_pooled_le P salt r M b cap tol huse hcap htol
  rw [Finset.sum_add_distrib] at h
  omega

/-- The per-block free claim gives the pooled one. -/
theorem freeClaimHolds_pooled (P : Runs L) (r : P.Run) (M : Finset (Fin (P.nIn r))) (h : P.FreeClaimHolds r M) :
    P.FreeClaimHoldsPooled r M :=
  Finset.sum_le_sum fun i hi => h i (Finset.mem_filter.mp hi).2

/-- **The pooled free claim holds from the pooled conversion**: in one run, PoUW's conclusion for `M` in every
server-day block, every server-day block's size at most `cap`, `M`'s committed size claims at most `tol`, and the
committed free claims summed at least the `cap`s summed less the `b`s summed plus `tol` give `FreeClaimHoldsPooled r M`. -/
theorem freeClaimHoldsPooled_of_conversion (P : Runs L) (salt : (r : P.Run) → Fin (P.nIn r) → ℕ) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (b cap : Fin (P.nOut r) → ℕ) (tol : ℕ)
    (huse : ∀ i, P.kind r i = true → b i - ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤
      ∑ u ∈ (P.infChildren r i).filter (fun u => salt r u ≤ (P.winIn r u).start ∧ u ∉ M), P.sizeIn r u)
    (hcap : ∀ i, P.kind r i = true → P.sizeOut r i ≤ cap i)
    (htol : ∑ u ∈ M, claimedSize (P.claimsIn r u) ≤ tol)
    (hclaim : (∑ i ∈ univ.filter (fun i => P.kind r i = true), cap i) -
        (∑ i ∈ univ.filter (fun i => P.kind r i = true), b i) + tol ≤
      ∑ i ∈ univ.filter (fun i => P.kind r i = true), P.claimedFree r i) : P.FreeClaimHoldsPooled r M :=
  (conversion_pooled_le_sub P salt r M b cap tol huse hcap htol).trans hclaim

/-- **The per-block cover gives the pooled one** on every accepted run with a server-day block: each block's
`cap − b + tol` summed is at least the `cap`s summed less the `b`s summed plus `tol` once. -/
theorem claimCovers_pooled (P : Runs L) (cap b : (r : P.Run) → Fin (P.nOut r) → ℕ) (tol : ℕ)
    (h : P.ClaimCoversConversion cap b tol) (r : P.Run) (ha : P.accepts r)
    (hne : (univ.filter (fun i => P.kind r i = true)).Nonempty) :
    (∑ i ∈ univ.filter (fun i => P.kind r i = true), cap r i) -
        (∑ i ∈ univ.filter (fun i => P.kind r i = true), b r i) + tol ≤
      ∑ i ∈ univ.filter (fun i => P.kind r i = true), P.claimedFree r i := by
  set S := univ.filter (fun i => P.kind r i = true)
  have hsum : ∑ i ∈ S, (cap r i - b r i + tol) ≤ ∑ i ∈ S, P.claimedFree r i :=
    Finset.sum_le_sum fun i hi => h r ha i (Finset.mem_filter.mp hi).2
  have hsplit : ∑ i ∈ S, cap r i ≤ ∑ i ∈ S, (cap r i - b r i) + ∑ i ∈ S, b r i := by
    rw [← Finset.sum_add_distrib]; exact Finset.sum_le_sum fun i _ => by omega
  have htol : tol ≤ ∑ _i ∈ S, tol := by
    obtain ⟨i, hi⟩ := hne
    exact Finset.single_le_sum (f := fun _ => tol) (fun _ _ => Nat.zero_le _) hi
  rw [Finset.sum_add_distrib] at hsum
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
    obtain ⟨M, hF, hM⟩ := hcover r ha fun x hx => hall x hx ha
    exact hn (hthm P r M (hcheck r ha) hM hF)
  calc P.pr (fun r => P.accepts r ∧ ¬ InferenceOnly.Compliant pol (P.actual r) (P.θ r))
      ≤ P.pr (fun r => ∃ x ∈ s, P.accepts r ∧ ¬ shown x r) := P.mono _ _ hsub
    _ ≤ ∑ x ∈ s, P.pr (fun r => P.accepts r ∧ ¬ shown x r) := P.union _ _
    _ ≤ ∑ x ∈ s, ε x := Finset.sum_le_sum fun x hx => hest x hx

/-- **An empty tolerance realizes**: `RealizesExcept ∅` gives `Realizes` (the claim clauses for every block). The converse
is `realizes_realizesExcept`; neither says anything of the committed free-capacity claim, which `FreeClaimHolds` carries
on its own. -/
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

/-- A claims list with a size claim holds its committed size claim. -/
theorem claimedSize_mem : ∀ (l : List Property),
    l.any InferenceOnly.isSize = true → Property.sizeAtMost (claimedSize l) ∈ l
  | [], h => by simp at h
  | p :: t, h => by
    cases p with
    | sizeAtMost k => simp [claimedSize]
    | _ =>
      simp only [claimedSize, List.any_cons, InferenceOnly.isSize, Bool.false_or] at h ⊢
      exact List.mem_cons_of_mem _ (claimedSize_mem t h)

/-- A block whose claims all hold, among them a size claim, has work at most its committed size claim. -/
theorem work_le_claimedSize {A : Placed L} (π : Fin A.N → ℕ) (θ : Finset (Fin A.N)) (X : Finset L)
    (b : Verity.Block A.toCircuit) (l : List Property) (hcl : ∀ p ∈ l, A.Holds π θ X b p)
    (hany : l.any InferenceOnly.isSize = true) : A.work π b.gates ≤ claimedSize l :=
  hcl _ (claimedSize_mem l hany)

/-- An inference, free or hidden block whose claims all hold has size at most its committed size claim. -/
theorem sizeIn_le_claimedSize (P : Runs L) (r : P.Run) (u : Fin (P.nIn r))
    (hcl : ∀ p ∈ P.claimsIn r u, P.HoldsIn r u p) (hany : (P.claimsIn r u).any InferenceOnly.isSize = true) :
    P.sizeIn r u ≤ claimedSize (P.claimsIn r u) :=
  work_le_claimedSize _ _ _ (P.innerBlock r u) _ hcl hany

/-- A claims list with an input claim from exactly `src` holds its committed input claim from `src`. -/
theorem claimedInput_mem (src : List Source) : ∀ (l : List Property),
    l.any (InferenceOnly.isInputFrom src) = true → Property.inputAtMost src (claimedInput src l) ∈ l
  | [], h => by simp at h
  | p :: t, h => by
    cases p with
    | inputAtMost s k =>
      by_cases hs : s = src
      · subst hs
        simp [claimedInput]
      · simp only [claimedInput, hs, ite_false, List.any_cons, InferenceOnly.isInputFrom, decide_false,
          Bool.false_or] at h ⊢
        exact List.mem_cons_of_mem _ (claimedInput_mem src t h)
    | _ =>
      simp only [claimedInput, List.any_cons, InferenceOnly.isInputFrom, Bool.false_or] at h ⊢
      exact List.mem_cons_of_mem _ (claimedInput_mem src t h)

/-- A block's input from its own past and the outside world together is at most its input from each, summed. -/
theorem blockInput_past_world_le {A : Placed L} (θ : Finset (Fin A.N)) (X : Finset L) (b : Verity.Block A.toCircuit) :
    A.blockInput θ X b [.past, .world] ≤ A.blockInput θ X b [.past] + A.blockInput θ X b [.world] := by
  unfold Placed.blockInput Circuit.width
  simp only [List.toFinset_cons, List.toFinset_nil, insert_empty_eq, Finset.biUnion_insert,
    Finset.singleton_biUnion, Finset.inter_union_distrib_left]
  have := Finset.sum_union_inter (f := A.w)
    (s₁ := A.into b.gates ∩ A.sourceGates θ b.gates X b.win .past)
    (s₂ := A.into b.gates ∩ A.sourceGates θ b.gates X b.win .world)
  omega

/-- **The check bounds a server-day block's size**: under the check and `RealizesExcept`, a server-day block's size is at
most `lim.cS` gates. -/
theorem serverSize_le (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run) (M : Finset (Fin (P.nIn r)))
    (hc : InferenceOnly.check lim P r = true) (hreal : P.RealizesExcept r M) (i : Fin (P.nOut r))
    (hki : P.kind r i = true) : P.sizeOut r i ≤ lim.cS := by
  simp only [InferenceOnly.check, Bool.and_eq_true, decide_eq_true_eq] at hc
  obtain ⟨⟨⟨⟨⟨-, -⟩, -⟩, -⟩, hsz⟩, -⟩ := hc
  obtain ⟨-, -, -, -, -, hout, -⟩ := hreal
  exact (work_le_claimedSize _ _ _ (P.outerBlock r i) _ (hout i) (hsz i hki).1).trans (hsz i hki).2

/-- **The limits bind what comes in**: under the check and `RealizesExcept`, a server-day block's input from its own
past and the outside world together is at most `lim.kS` bits. Its input from another block at its location during its
window (`.block`) is not bounded here. -/
theorem serverInput_le (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run) (M : Finset (Fin (P.nIn r)))
    (hc : InferenceOnly.check lim P r = true) (hreal : P.RealizesExcept r M) (i : Fin (P.nOut r))
    (hki : P.kind r i = true) :
    (P.actual r).blockInput (P.θ r) (P.locOut r i) (P.outerBlock r i) [.past, .world] ≤ lim.kS := by
  simp only [InferenceOnly.check, Bool.and_eq_true, decide_eq_true_eq] at hc
  obtain ⟨⟨⟨⟨⟨-, -⟩, -⟩, -⟩, -⟩, hin⟩ := hc
  obtain ⟨-, -, -, -, -, hout, -⟩ := hreal
  obtain ⟨hp, hw, hsum⟩ := hin i hki
  have h1 : (P.actual r).blockInput (P.θ r) (P.locOut r i) (P.outerBlock r i) [.past] ≤
      claimedInput [.past] (P.claimsOut r i) := hout i _ (claimedInput_mem _ _ hp)
  have h2 : (P.actual r).blockInput (P.θ r) (P.locOut r i) (P.outerBlock r i) [.world] ≤
      claimedInput [.world] (P.claimsOut r i) := hout i _ (claimedInput_mem _ _ hw)
  have := blockInput_past_world_le (P.θ r) (P.locOut r i) (P.outerBlock r i)
  omega

/-- **A server-day block claiming more input than `kS` fails the check**, whatever the rest of the run: its committed
input claims from its own past and from the outside world, summed, above `lim.kS` bits. -/
theorem serverInput_excluded (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run) (i : Fin (P.nOut r))
    (hki : P.kind r i = true)
    (hbig : lim.kS < claimedInput [.past] (P.claimsOut r i) + claimedInput [.world] (P.claimsOut r i)) :
    InferenceOnly.check lim P r = false := by
  by_contra hc
  simp only [Bool.not_eq_false, InferenceOnly.check, Bool.and_eq_true, decide_eq_true_eq] at hc
  obtain ⟨⟨⟨⟨⟨-, -⟩, -⟩, -⟩, -⟩, hin⟩ := hc
  exact absurd (hin i hki).2.2 (by omega)

/-- **The check bites, on the path the chain reads**: under a run that passes the inference-only check and realizes its
claims outside a tolerance set `M` (`RealizesExcept`, not `Realizes`), with the work at unlisted locations at most
`lim.hidden` (assumed), the whole circuit's work is at most, for each server-day block, the smaller of the server-day size limit
`lim.cS` (its committed size claim, which the check caps at `lim.cS`) and the count of its inference blocks **outside**
`M` at the size cap `lim.c` (each block's committed size claim, which the check caps at `lim.c`) plus the free-block limit
`lim.freeSize`. The bound is in the limits alone: the free blocks and `M`'s inference blocks sit inside the committed
claim `P.claimedFree r i` (the hypothesis `FreeClaimHolds`), which the check caps at `lim.freeSize`. The tolerance `M` is
a bounded term here, not an unbounded quantifier: `M`'s blocks are counted inside the free limit, not as inference
credit. -/
theorem feasible_work_le (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run)
    (M : Finset (Fin (P.nIn r)))
    (hc : InferenceOnly.check lim P r = true) (hreal : P.RealizesExcept r M)
    (hfree : P.FreeClaimHolds r M)
    (hhidden : (P.actual r).work (P.π r) (P.hiddenGates r) ≤ lim.hidden) :
    (P.actual r).work (P.π r) (P.actual r).computed ≤
      (∑ i ∈ univ.filter (fun i => P.kind r i = true),
        min lim.cS (((P.infChildren r i).filter (fun u => u ∉ M)).card * lim.c + lim.freeSize)) +
        lim.hidden := by
  have hserver := serverSize_le lim P r M hc hreal
  simp only [InferenceOnly.check, Bool.and_eq_true, decide_eq_true_eq] at hc
  obtain ⟨⟨⟨⟨⟨-, hinf⟩, -⟩, hfcCheck⟩, -⟩, -⟩ := hc
  obtain ⟨-, -, -, hcl_out, -, -, -⟩ := hreal
  rw [work_eq_outer_sum, ← Finset.sum_filter_add_sum_filter_not univ (fun i => P.kind r i = true) (P.sizeOut r)]
  have hfalse : (univ.filter fun i => ¬ P.kind r i = true) = univ.filter fun i => P.kind r i = false := by
    apply Finset.filter_congr; intro i _; simp [Bool.not_eq_true]
  rw [hfalse]
  refine add_le_add ?_ ((hidden_work_le P r).trans hhidden)
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
          exact (sizeIn_le_claimedSize P r u (hcl_out u hu.1.2 hu.2) (hinf u hu.1.2).1).trans (hinf u hu.1.2).2.1
      _ = ((P.infChildren r i).filter (fun u => u ∉ M)).card * lim.c := by rw [Finset.sum_const, smul_eq_mul]
  have hfc := hfree i hki
  have hfs := hfcCheck i hki
  omega

/-- **The check bites, with no tolerance set in the bound**: the same bound counting every inference block (not only those
outside `M`), so the conclusion names no `M`. From `feasible_work_le` by monotonicity (`(infChildren i).filter (· ∉ M) ⊆
infChildren i`). The count of inference blocks the bound reads is capped by `lim.n` in the check. -/
theorem feasible_work_le_total (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run)
    (M : Finset (Fin (P.nIn r)))
    (hc : InferenceOnly.check lim P r = true) (hreal : P.RealizesExcept r M)
    (hfree : P.FreeClaimHolds r M)
    (hhidden : (P.actual r).work (P.π r) (P.hiddenGates r) ≤ lim.hidden) :
    (P.actual r).work (P.π r) (P.actual r).computed ≤
      (∑ i ∈ univ.filter (fun i => P.kind r i = true),
        min lim.cS ((P.infChildren r i).card * lim.c + lim.freeSize)) + lim.hidden := by
  refine le_trans (feasible_work_le lim P r M hc hreal hfree hhidden) (add_le_add ?_ le_rfl)
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
whole circuit as one inference block, claimed of size at most 1 and 0 fresh input, with a size claim and input claims
from its own past and from the outside world on the server-day block (so the check's server-day conjuncts can bite). Its window covers every tick of `placed n`, so
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
  listed := {{()}}
  kindIff _ _ := by simp
  hiddenUnlisted _ _ h := absurd h (by decide)
  winOut _ _ := ⟨0, n + 1⟩
  claimsOut _ _ := [.sizeAtMost n, .inputAtMost [.past] 0, .inputAtMost [.world] 0]
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
    rcases hp with rfl | rfl | rfl
    · show (placed 1).work _ _ ≤ 1
      simp only [demoRuns, Runs.outerBlock, Partition.whole, Partition.gates, Placed.work]
      decide
    all_goals
      show (placed 1).blockInput _ _ _ _ ≤ 0
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

/-- The demos' limits: one inference block of size claim at most 1 gate and fresh input claim at most 0 bits, no free
block, no hidden hardware, and the given free-claim limit (gates), server-day size limit (gates) and server-day input
limit (bits). -/
noncomputable def demoLimits (freeSize cS kS : ℕ) : InferenceOnly.Limits :=
  { n := 1, c := 1, k := 0, m := 0, freeSize := freeSize, cS := cS, kS := kS, hidden := 0, share := 0, allowance := 0 }

/-- **The honest run passes the check and realizes at the empty tolerance**: `demoRuns 1 0` at honest limits (one
inference block of size at most 1 gate and fresh input at most 0 bits, no free block, committed free claim 0 gates, the
server-day block's size at most 1 gate and input at most 0 bits) passes `check` and is in the feasible set with no
tolerance. -/
theorem honest_checks :
    InferenceOnly.check (demoLimits 0 1 0) (demoRuns 1 0) () = true ∧
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
    ¬ (InferenceOnly.check (demoLimits 0 2 0) (demoRuns 2 fc) () = true ∧
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
      InferenceOnly.check (demoLimits 0 2 0) (demoRuns 2 2) () = false ∧
      InferenceOnly.check (demoLimits 2 2 0) (demoRuns 2 2) () = true := by
  refine ⟨⟨?_, ?_, fun _ _ => le_rfl, ?_, ?_, ?_, ?_⟩, ?_, by decide, by decide⟩
  · intro a ha; exact absurd ha (Finset.notMem_empty a)
  · simp [demoRuns, Circuit.width]
  · intro u _ hu; exact absurd (Finset.mem_univ u) hu
  · intro u hu; exact absurd (by simp [demoRuns] : (demoRuns 2 2).cls () u = Class.inference) hu
  · intro i p hp
    fin_cases i
    simp only [demoRuns, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl | rfl
    · show (placed 2).work _ _ ≤ 2
      simp only [demoRuns, Runs.outerBlock, Partition.whole, Partition.gates, Placed.work]; decide
    all_goals
      show (placed 2).blockInput _ _ _ _ ≤ 0
      simp only [demoRuns, Runs.outerBlock, Placed.blockInput, Circuit.into, Circuit.width, placed, line]
      decide
  · intro u _; simp [demoRuns]
  · intro i _
    rw [demo_freeChildren_empty, Finset.sum_empty, zero_add,
      Finset.filter_true_of_mem (fun u _ => Finset.mem_univ u)]
    fin_cases i
    decide

/-- `demoRuns 1 0` with its server-day block's committed input claim from the outside world at `kw` bits. -/
noncomputable def demoRunsInput (kw : ℕ) : Runs Unit :=
  { demoRuns 1 0 with claimsOut := fun _ _ => [.sizeAtMost 1, .inputAtMost [.past] 0, .inputAtMost [.world] kw] }

/-- **The input limit bites, at one bit**: the one-gate run whose server-day block claims `2^60` bits of input from the
outside world realizes its claims (its block reads nothing), and the check refuses it at `kS = 2^60 − 1` bits and passes
it at `kS = 2^60` bits, every other limit the same. -/
theorem input_claim_binds :
    (demoRunsInput (2 ^ 60)).Realizes () ∧
      InferenceOnly.check (demoLimits 0 1 (2 ^ 60 - 1)) (demoRunsInput (2 ^ 60)) () = false ∧
      InferenceOnly.check (demoLimits 0 1 (2 ^ 60)) (demoRunsInput (2 ^ 60)) () = true := by
  refine ⟨?_, serverInput_excluded (demoLimits 0 1 (2 ^ 60 - 1)) (demoRunsInput (2 ^ 60)) () ⟨0, Nat.one_pos⟩ rfl
    (by decide), by decide⟩
  obtain ⟨h1, h2, h3, h4, -⟩ := honest_feasible
  refine ⟨h1, h2, h3, h4, ?_⟩
  intro i p hp
  fin_cases i
  simp only [demoRunsInput, List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl | rfl
  · show (placed 1).work _ _ ≤ 1
    simp only [demoRuns, Runs.outerBlock, Partition.whole, Partition.gates, Placed.work]
    decide
  all_goals
    show (placed 1).blockInput _ _ _ _ ≤ _
    simp only [demoRuns, Runs.outerBlock, Placed.blockInput, Circuit.into, Circuit.width, placed, line]
    decide

/-- **The conversion bites, at one gate**: the two-gate run `demoRuns 2 fc` with its one inference block (work 2 gates,
committed size claim 1 gate) as the tolerance set `M = univ`, salt at tick 0 and capacity 2 gates. At PoUW's bound
`b = 1` and tolerance `tol = 1` PoUW's conclusion, the capacity and the tolerance hold, and the conversion's bound
`cap − b + tol` is 2 gates, exactly `M`'s work; the committed claim `fc = 2` then holds (`FreeClaimHolds`) and `fc = 1`
does not. One more gate of PoUW's bound (`b = 2`) or one less of tolerance (`tol = 0`) and its premise fails. -/
theorem conversion_bites :
    let P2 := demoRuns 2 2
    let i : Fin (P2.nOut ()) := ⟨0, Nat.one_pos⟩
    let Md := ∑ u ∈ (P2.infChildren () i).filter (· ∈ Finset.univ), claimedSize (P2.claimsIn () u)
    let outside := ∑ u ∈ (P2.infChildren () i).filter (fun u => 0 ≤ (P2.winIn () u).start ∧ u ∉ Finset.univ),
      P2.sizeIn () u
    (1 - Md ≤ outside ∧ P2.sizeOut () i ≤ 2 ∧ ∑ u ∈ Finset.univ, claimedSize (P2.claimsIn () u) ≤ 1 ∧
      (∑ u ∈ P2.freeChildren () i, P2.sizeIn () u) +
        (∑ u ∈ (P2.infChildren () i).filter (· ∈ Finset.univ), P2.sizeIn () u) = 2 - 1 + 1) ∧
    P2.FreeClaimHolds () Finset.univ ∧ ¬ (demoRuns 2 1).FreeClaimHolds () Finset.univ ∧
    ¬ (2 - Md ≤ outside) ∧ ¬ (∑ u ∈ Finset.univ, claimedSize (P2.claimsIn () u) ≤ 0) := by
  intro P2 i Md outside
  have hs : ∀ fc, ∑ u ∈ (demoRuns 2 fc).infChildren () ⟨0, Nat.one_pos⟩, (demoRuns 2 fc).sizeIn () u = 2 :=
    fun _ => (by decide : ∑ u ∈ (demoRuns 2 0).infChildren () ⟨0, Nat.one_pos⟩, (demoRuns 2 0).sizeIn () u = 2)
  have hfree : ∀ fc, (demoRuns 2 fc).FreeClaimHolds () Finset.univ ↔ 2 ≤ fc := by
    intro fc
    constructor
    · intro h
      have := h ⟨0, Nat.one_pos⟩ rfl
      rw [demo_freeChildren_empty, Finset.sum_empty, zero_add,
        Finset.filter_true_of_mem (fun u _ => Finset.mem_univ u), hs] at this
      exact this
    · intro h j _
      fin_cases j
      rw [demo_freeChildren_empty, Finset.sum_empty, zero_add,
        Finset.filter_true_of_mem (fun u _ => Finset.mem_univ u)]
      exact (hs fc).le.trans h
  refine ⟨⟨by decide, by decide, by decide, ?_⟩, (hfree 2).mpr le_rfl, fun h => absurd ((hfree 1).mp h) (by decide),
    by decide, by decide⟩
  rw [demo_freeChildren_empty, Finset.sum_empty, zero_add]
  decide

/-- `line 2` in two units: gate 1 in unit 0, gate 2 in unit 1. -/
def cutTwo : Partition (placed 2).toCircuit 2 where
  unit g := if g.val = 0 then none else if g.val = 1 then some 0 else some 1
  input_iff g := by
    simp only [placed, line]
    split_ifs with h0 h1 <;> simp [h0]

/-- Unit `i` of `cutTwo` holds gate `i + 1` alone. -/
theorem cutTwo_gate (i : Fin 2) (g : Fin (placed 2).N) (hg : cutTwo.unit g = some i) : g.val = i.val + 1 := by
  have hlt := g.isLt
  simp only [placed, line] at hlt
  simp only [cutTwo] at hg
  split_ifs at hg with h0 h1
  · obtain rfl := Option.some_injective _ hg
    simp [h1]
  · obtain rfl := Option.some_injective _ hg
    have : (1 : Fin 2).val = 1 := rfl
    omega

/-- One run of the circuit `placed 2` at a free-block capacity `fc`, cut at one location into two server-day blocks, gate
1 during tick 1 and gate 2 during tick 2, each holding its gate as one inference block claimed of size at most 1 gate and
0 bits of fresh input, each server-day block claiming size at most 1 gate and 0 bits of input from its own past and from
the outside world, and each committing the free-capacity claim `fc`. -/
noncomputable def demoCut (fc : ℕ) : Runs Unit where
  Run := Unit
  pr _ := 0
  mono _ _ _ := le_refl 0
  union _ _ := by simp
  accepts _ := True
  actual _ := placed 2
  π _ _ := 1
  θ _ := ∅
  registered := 0
  nOut _ := 2
  outer _ := cutTwo
  kind _ _ := true
  locOut _ _ := {()}
  listed := {{()}}
  kindIff _ _ := by simp
  hiddenUnlisted _ _ h := absurd h (by decide)
  winOut _ i := ⟨i.val + 1, 1⟩
  claimsOut _ _ := [.sizeAtMost 1, .inputAtMost [.past] 0, .inputAtMost [.world] 0]
  claimedFree _ _ := fc
  nIn _ := 2
  inner _ := cutTwo
  ref _ := { parent := id, sub := fun g => by simp }
  cls _ _ := Class.inference
  locIn _ _ := {()}
  winIn _ u := ⟨u.val + 1, 1⟩
  claimsIn _ _ := [.sizeAtMost 1, .inputAtMost [.past, .block, .world] 0]
  hiddenIff _ _ := by simp
  insideOut := fun _ i g hg => by
    rw [Partition.gates, mem_filter] at hg
    have h := cutTwo_gate i g hg.2
    simp only [Placed.region, Circuit.computed, Finset.mem_filter, mem_univ, true_and,
      Finset.mem_coe, Window.ticks, Finset.mem_Ico]
    refine ⟨?_, by simp [placed], ?_, ?_⟩
    · simp only [placed, line, decide_eq_false_iff_not]; omega
    · simp only [placed]; omega
    · simp only [placed]; omega
  insideIn := fun _ u g hg => by
    rw [Partition.gates, mem_filter] at hg
    have h := cutTwo_gate u g hg.2
    simp only [Placed.region, Circuit.computed, Finset.mem_filter, mem_univ, true_and,
      Finset.mem_coe, Window.ticks, Finset.mem_Ico]
    refine ⟨?_, by simp [placed], ?_, ?_⟩
    · simp only [placed, line, decide_eq_false_iff_not]; omega
    · simp only [placed]; omega
    · simp only [placed]; omega

/-- **The pooled bound counts the tolerance once**: the run `demoCut 1`, two server-day blocks of capacity 1 gate each,
PoUW's bound `b = 1` gate in each, both inference blocks (committed size claim 1 gate, work 1 gate each) in the tolerance
set `M = univ`, salt at tick 0 and tolerance `tol = 2` gates. PoUW's conclusion, the capacity and the tolerance hold in
each block; the free and `M`'s work summed over the run is 2 gates, exactly the pooled bound (`2 − 2 + 2`), while the
per-block bounds summed are 4 gates (`(1 − 1 + 2)` twice). The committed claims `1` gate each hold, pooled and per
block, and cover the pooled bound (`ClaimCoversConversionPooled`'s comparison), not the per-block one
(`ClaimCoversConversion`'s, which asks 2 gates of each). -/
theorem pooled_counts_tol_once :
    let P := demoCut 1
    let S := univ.filter (fun i => P.kind () i = true)
    (∀ i, P.kind () i = true →
        1 - ∑ u ∈ (P.infChildren () i).filter (· ∈ univ), claimedSize (P.claimsIn () u) ≤
          ∑ u ∈ (P.infChildren () i).filter (fun u => 0 ≤ (P.winIn () u).start ∧ u ∉ univ), P.sizeIn () u) ∧
      (∀ i, P.kind () i = true → P.sizeOut () i ≤ 1) ∧ ∑ u ∈ univ, claimedSize (P.claimsIn () u) ≤ 2 ∧
      ∑ i ∈ S, ((∑ u ∈ P.freeChildren () i, P.sizeIn () u) +
          ∑ u ∈ (P.infChildren () i).filter (· ∈ univ), P.sizeIn () u) = (∑ _i ∈ S, 1) - (∑ _i ∈ S, 1) + 2 ∧
      ∑ _i ∈ S, (1 - 1 + 2) = 4 ∧
      P.FreeClaimHolds () univ ∧ P.FreeClaimHoldsPooled () univ ∧
      (∑ _i ∈ S, 1) - (∑ _i ∈ S, 1) + 2 ≤ ∑ i ∈ S, P.claimedFree () i ∧
      ¬ ∀ i, P.kind () i = true → 1 - 1 + 2 ≤ P.claimedFree () i := by
  intro P S
  refine ⟨by decide, by decide, by decide, by decide, by decide, ?_, ?_, by decide, by decide⟩
  · intro i _
    fin_cases i <;> decide
  · show _ ≤ _
    decide

/-- The cut witness's limits: at most 2 inference blocks in the run, each of size claim at most 1 gate and fresh input
claim at most 0 bits; no free block; each server-day block's committed free claim at most 1 gate, its size at most 2
gates and its input from its own past and the outside world at most 0 bits; no hidden hardware. -/
noncomputable def cutLimits : InferenceOnly.Limits :=
  { n := 2, c := 1, k := 0, m := 0, freeSize := 1, cS := 2, kS := 0, hidden := 0, share := 0, allowance := 0 }

/-- **Cutting a location's ticks multiplies the per-block limits**: the circuit `placed 2` (2 gates of work at one
location, during ticks 1 and 2) with both its inference blocks in the tolerance set passes `check` at `cutLimits`
(`freeSize` = 1 gate) when cut into two server-day blocks of one tick each (`demoCut 1`), realizing its claims outside
the tolerance set with its committed free claims holding; uncut, as one server-day block (`demoRuns 2 fc`), no committed
claim `fc` and no tolerance set `M` passes `check` at the same limits with `RealizesExcept` and `FreeClaimHolds`. -/
theorem cut_multiplies :
    InferenceOnly.check cutLimits (demoCut 1) () = true ∧ (demoCut 1).RealizesExcept () univ ∧
      (demoCut 1).FreeClaimHolds () univ ∧
      ∀ fc (M : Finset (Fin ((demoRuns 2 fc).nIn ()))),
        ¬ (InferenceOnly.check cutLimits (demoRuns 2 fc) () = true ∧ (demoRuns 2 fc).RealizesExcept () M ∧
          (demoRuns 2 fc).FreeClaimHolds () M) := by
  refine ⟨by decide, ⟨?_, ?_, fun _ _ => le_rfl, ?_, ?_, ?_, ?_⟩, (pooled_counts_tol_once).2.2.2.2.2.1, ?_⟩
  · intro a ha; exact absurd ha (Finset.notMem_empty a)
  · simp [demoCut, Circuit.width]
  · intro u _ hu; exact absurd (Finset.mem_univ u) hu
  · intro u hu; exact absurd (by simp [demoCut] : (demoCut 1).cls () u = Class.inference) hu
  · intro i p hp
    simp only [demoCut, List.mem_cons, List.not_mem_nil, or_false] at hp
    fin_cases i <;> rcases hp with rfl | rfl | rfl
    all_goals first
      | (show (placed 2).work _ _ ≤ 1
         simp only [demoCut, Runs.outerBlock, Partition.gates, Placed.work]; decide)
      | (show (placed 2).blockInput _ _ _ _ ≤ 0
         simp only [demoCut, Runs.outerBlock, Placed.blockInput, Circuit.into, Circuit.width, placed, line]
         decide)
  · intro u _; simp [demoCut]
  · intro fc M ⟨hc, hreal, hfree⟩
    by_cases hM : (⟨0, Nat.one_pos⟩ : Fin ((demoRuns 2 fc).nIn ())) ∈ M
    · refine excluded_run cutLimits (demoRuns 2 fc) () M ⟨0, Nat.one_pos⟩ rfl ?_ ⟨hc, hfree⟩
      have hmem : (⟨0, Nat.one_pos⟩ : Fin ((demoRuns 2 fc).nIn ())) ∈
          ((demoRuns 2 fc).infChildren () ⟨0, Nat.one_pos⟩).filter (· ∈ M) := by
        rw [Finset.mem_filter]
        refine ⟨?_, hM⟩
        simp only [Runs.infChildren, Finset.mem_filter, Finset.mem_univ, true_and]
        exact ⟨rfl, rfl⟩
      calc (1 : ℕ) < (demoRuns 2 fc).sizeIn () ⟨0, Nat.one_pos⟩ := by
            show (1 : ℕ) < (placed 2).work (fun _ => 1)
              ((Partition.whole (placed 2).toCircuit).gates ⟨0, Nat.one_pos⟩)
            decide
        _ ≤ _ := Finset.single_le_sum (fun j _ => Nat.zero_le _) hmem
    · obtain ⟨-, -, -, hcl_in, -⟩ := hreal
      have h : (demoRuns 2 fc).sizeIn () ⟨0, Nat.one_pos⟩ ≤ 1 :=
        hcl_in ⟨0, Nat.one_pos⟩ rfl hM (.sizeAtMost 1) (by simp [demoRuns])
      have h2 : (demoRuns 2 fc).sizeIn () ⟨0, Nat.one_pos⟩ = 2 := by
        show (placed 2).work (fun _ => 1) ((Partition.whole (placed 2).toCircuit).gates ⟨0, Nat.one_pos⟩) = 2
        decide
      omega

/-- **No top block is in no class**: `kind` is total, so the degenerate case of a top block in neither the server-day nor
the hidden class cannot be stated. Nothing sits inside an inference block, the finest scale here, so neither can a block
inside an inference block that is not a proof block. -/
theorem top_blocks_classified (P : Runs L) (r : P.Run) (i : Fin (P.nOut r)) :
    P.kind r i = true ∨ P.kind r i = false := by cases P.kind r i <;> simp

/-! ## The chain, with the protocols' rows named as types

`rows_chain` is the chain with each protocol's row fed to `chain` as a typed hypothesis, not named in a docstring. Its
hypotheses are the rows (`PouwUseAtLeast` with its premises and the compute capacity `CapacityAtMost
Capacity.computeRate P.sizeOut`, which `capacityAtMost_compute` gives from `ComputeRate` over the listed locations; `DeclaredBlocksRun`; `ClaimCoversConversion`; `PousUseAtLeast` and `PoubUseAtLeast`
with their conversions, their exclusivity premise open for memory and the link; `WipeInputAtMost`,
`NetworkInputAtMost`, `NetworkOutputAtMost` with their premises; `RegistrationBinds` with `RegistrationSound` and
`RegisteredReadsOpenLog`). Its conclusion is the policy's: an accepted run's actual circuit is compliant except with the
seven rows' soundness errors summed. It is proved from `chain`.

`FreeClaimHolds r M` is proved, for the set `M` the declared row gives: PoUW's conclusion for that `M`, the compute
capacity, `M`'s committed size claims at most `tol` and `ClaimCoversConversion` give it (`freeClaimHolds_of_conversion`).
One hypothesis is not a row: `assemble`, which takes that `M` and the other rows' per-run conclusions to `RealizesExcept r
M`. It is open because the rows do not give its clauses: that `θ`'s gates are input gates of at most the registered width
and that the pricing charges every computed gate; that the declared row's three claims are each inference block's whole
claims list, and that the inference blocks before their salt (in no row) hold theirs; that the free blocks' claims hold;
and that the server-day and hidden blocks' claims are exactly what the wipe's and network attestation's rows give. The
deterministic rows carry no error of their own: `AttesterOnEveryLink`, the capacities and `ClaimCoversConversion` feed
the error rows or the free claim, and `RegistrationBinds` feeds `assemble`; they are hypotheses but add no term to the
sum. `NetworkInputAtMost`'s error is `0` (its bound is the covert channels' assumed figure, with no sampled proof). -/
theorem rows_chain (P : Runs L)
    (check : (P : Runs L) → P.Run → Bool) (pol : InferenceOnly.Policy)
    (hthm : InferenceOnly.PolicyTheorem check pol) (hcheck : ∀ r, P.accepts r → check P r = true)
    -- PoUW: use at least, in size, for every tolerance set, and the compute capacity.
    {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : Pouw.PearlC.CostModel Q R S) (Pr : Pouw.PearlC.Protocol Q R S) (TR : Pouw.PearlC.TileRules Q R S)
    (D : Pouw.PearlC.Layout → Prop) (Wd : Pouw.PearlC.Layout → ℕ → Pouw.PearlC.Decl → ℚ) (g : ℝ) (δ : ℕ → ℕ → ℝ)
    (salt : (r : P.Run) → Fin (P.nIn r) → ℕ) (shownPouw : P.Run → Prop)
    (bPouw capPouw : (r : P.Run) → Fin (P.nOut r) → ℕ) (εpouw : ℝ≥0∞)
    (pouw : P.PouwUseAtLeast CM Pr TR D Wd g δ salt shownPouw bPouw εpouw)
    (hggap : Pouw.PearlC.GGapSampledDraw CM Pr TR D Wd g δ)
    (hpouwRead : P.RunReads (Pouw.PearlC.GGapSampledDraw CM Pr TR D Wd g δ) shownPouw εpouw)
    (hspend : P.SpendIsWork salt shownPouw bPouw)
    (hcapPouw : P.CapacityAtMost Capacity.computeRate P.sizeOut capPouw)
    -- The declared blocks run, and the committed free claim against the conversion's bound.
    (f : (r : P.Run) → Fin (P.nIn r) → Circuit) (cDecl kDecl : (r : P.Run) → Fin (P.nIn r) → ℕ)
    (tol : ℕ) (εdecl : ℝ≥0∞)
    (declared : P.DeclaredBlocksRun salt f cDecl kDecl tol εdecl)
    (hclaim : P.ClaimCoversConversion capPouw bPouw tol)
    -- PoUS: use at least, over memory size, with its conversion (its exclusivity premise open for memory).
    (heldS bPous freePous capPous enclosingPous : (r : P.Run) → Fin (P.nOut r) → ℕ) (εpous : ℝ≥0∞)
    (pous : P.PousUseAtLeast heldS bPous εpous)
    (hcapPous : P.CapacityAtMost Capacity.memorySize enclosingPous capPous)
    (hexclPous : P.DeclaredWorkExclusive (fun r _ i => freePous r i) (fun r _ i => heldS r i) enclosingPous)
    -- PoUB: use at least, over the link, with its conversion (its exclusivity premise open for the link).
    (readsB bPoub freePoub capPoub enclosingPoub : (r : P.Run) → Fin (P.nOut r) → ℕ) (εpoub : ℝ≥0∞)
    (poub : P.PoubUseAtLeast readsB bPoub εpoub)
    (hcapPoub : P.CapacityAtMost Capacity.linkCapacity enclosingPoub capPoub)
    (hexclPoub : P.DeclaredWorkExclusive (fun r _ i => freePoub r i) (fun r _ i => readsB r i) enclosingPoub)
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
    -- The one open premise: the declared row's set and the other rows' per-run conclusions give `RealizesExcept`.
    (assemble : ∀ r, P.accepts r →
      ∀ M ⊆ (univ.filter fun u => P.cls r u = Class.inference ∧ salt r u ≤ (P.winIn r u).start),
        (∑ u ∈ M, claimedSize (P.claimsIn r u)) ≤ tol →
        (∀ u ∈ (univ.filter fun u => P.cls r u = Class.inference ∧ salt r u ≤ (P.winIn r u).start) \ M,
          P.HoldsIn r u (.function (f r u)) ∧ P.HoldsIn r u (.sizeAtMost (cDecl r u)) ∧
            P.HoldsIn r u (.inputAtMost [.past, .block, .world] (kDecl r u))) →
        (∀ i, P.kind r i = true → freePous r i ≤ capPous r i - bPous r i) →
        (∀ i, P.kind r i = true → freePoub r i ≤ capPoub r i - bPoub r i) →
        (∀ i, P.kind r i = true → P.HoldsOut r i (.inputAtMost [.past] (Ewipe r i + 8600000000))) →
        (∀ i, P.kind r i = true → P.HoldsOut r i (.inputAtMost [.world] (bIn r i + covert r i))) →
        (∀ i, P.kind r i = true → P.HoldsOut r i (.outputAtMost (bOut r i))) →
        (∀ g ∈ P.θ r, ∃ root ∈ log, opensAt r g root) →
        P.RealizesExcept r M) :
    ExceptWith P.pr P.accepts (fun r => InferenceOnly.Compliant pol (P.actual r) (P.θ r))
      (∑ x : Fin 7, (![εpouw, εdecl, εpous, εpoub, 2⁻¹ ^ 40 + adv, 0, εnet] : Fin 7 → ℝ≥0∞) x) := by
  set shownV : Fin 7 → P.Run → Prop :=
    ![ (fun r => ∀ (M : Finset (Fin (P.nIn r))) i, P.kind r i = true →
         bPouw r i - ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤
           ∑ u ∈ (P.infChildren r i).filter (fun u => salt r u ≤ (P.winIn r u).start ∧ u ∉ M), P.sizeIn r u),
       (fun r => ∃ M ⊆ (univ.filter fun u => P.cls r u = Class.inference ∧ salt r u ≤ (P.winIn r u).start),
         (∑ u ∈ M, claimedSize (P.claimsIn r u)) ≤ tol ∧
           ∀ u ∈ (univ.filter fun u => P.cls r u = Class.inference ∧ salt r u ≤ (P.winIn r u).start) \ M,
             P.HoldsIn r u (.function (f r u)) ∧ P.HoldsIn r u (.sizeAtMost (cDecl r u)) ∧
               P.HoldsIn r u (.inputAtMost [.past, .block, .world] (kDecl r u))),
       (fun r => ∀ (_M : Finset (Fin (P.nIn r))) i, P.kind r i = true → freePous r i ≤ capPous r i - bPous r i),
       (fun r => ∀ (_M : Finset (Fin (P.nIn r))) i, P.kind r i = true → freePoub r i ≤ capPoub r i - bPoub r i),
       (fun r => ∀ i, P.kind r i = true → P.HoldsOut r i (.inputAtMost [.past] (Ewipe r i + 8600000000))),
       (fun r => ∀ i, P.kind r i = true → P.HoldsOut r i (.inputAtMost [.world] (bIn r i + covert r i))),
       (fun r => ∀ i, P.kind r i = true → P.HoldsOut r i (.outputAtMost (bOut r i))) ] with hshownV
  set εV : Fin 7 → ℝ≥0∞ := ![εpouw, εdecl, εpous, εpoub, 2⁻¹ ^ 40 + adv, 0, εnet] with hεV
  have hcover : ∀ r, P.accepts r → (∀ x ∈ Finset.univ, shownV x r) →
      ∃ M, P.FreeClaimHolds r M ∧ P.RealizesExcept r M := by
    intro r ha hall
    have h0 := hall 0 (Finset.mem_univ _)
    have h1 := hall 1 (Finset.mem_univ _)
    have h2 := hall 2 (Finset.mem_univ _)
    have h3 := hall 3 (Finset.mem_univ _)
    simp only [hshownV, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val] at h0 h1 h2 h3
    obtain ⟨M, hMsub, htol, hdecl⟩ := h1
    refine ⟨M, freeClaimHolds_of_conversion P salt r M (bPouw r) (capPouw r) tol (h0 M)
      (fun i hki => hcapPouw r i hki) htol (hclaim r ha), ?_⟩
    exact assemble r ha M hMsub htol hdecl (h2 ∅) (h3 ∅)
      (hall 4 (Finset.mem_univ _)) (hall 5 (Finset.mem_univ _)) (hall 6 (Finset.mem_univ _))
      (reg hregsound hregreads r ha)
  have hest : ∀ x ∈ Finset.univ, ExceptWith P.pr P.accepts (shownV x) (εV x) := by
    intro x _
    fin_cases x
    · exact pouw hggap hpouwRead hspend
    · exact declared
    · exact (freeBlockAtMost P) Capacity.memorySize (fun r _ i => freePous r i) (fun r _ i => heldS r i)
        (fun r _ i => bPous r i) enclosingPous capPous εpous pous hcapPous hexclPous
    · exact (freeBlockAtMost P) Capacity.linkCapacity (fun r _ i => freePoub r i) (fun r _ i => readsB r i)
        (fun r _ i => bPoub r i) enclosingPoub capPoub εpoub poub hcapPoub hexclPoub
    · exact wipe hGwipe hwipeRead hwipeCap hwipeInfo
    · exact netIn hatt hingress
    · exact netOut hphase hnetRead hrecord hatt'
  exact chain P check pol Finset.univ shownV εV hthm (fun r ha => hcheck r ha) hcover hest

end ResourceAccounting.SecurityProofs.BlockProperties
