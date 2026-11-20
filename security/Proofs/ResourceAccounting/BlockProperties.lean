module

public import Definitions.Compose.Guarantees.Blocks
public import Definitions.Registration.Physical
public import Proofs.Pous.StreamedWipe.Residual
import Mathlib.Tactic

/-!
# The conversion, sizes over the partitions, the chain, and the feasible set is not everything

`freeBlockAtMost` proves `FreeBlockAtMost`. `sizeAdds` proves that the blocks directly inside a server-day block have
sizes adding to at most it, and `sizeExclusive` that a server-day block's free and inference children's work do not
double-count. `conversion_le` is the conversion for size as one inequality, `freeWorkAtMost_of_conversion` gives
`FreeWorkAtMost` from it on the run's capacity, credited use and tolerance, and `conversion_bites` shows it tight at one
gate. `conversion_pooled_le` is the same conversion summed over a run's server-day blocks with the tolerance counted
once, `freeWorkAtMostPooled_of_conversion` gives `FreeWorkAtMostPooled` from it, and `pooled_counts_tol_once` shows the
pooled bound below the per-block bounds summed, on a run with two server-day blocks. `listed_gate_in_serverDay` puts every
computed gate at a listed location in a server-day block, `hidden_work_le` bounds hidden hardware's work by the work at
unlisted locations, `listed_gate_in_period` puts every computed gate at a listed location inside the period,
`gate_serverDay_or_unlisted` puts every computed gate in a server-day block or at an unlisted location, and
`capacityAtMost_compute` gives the compute capacity from `ComputeRate` over the listed
locations. `chain` proves `Chain`.

The policy's check bites: `feasible_work_le` bounds a realizing run's total work by, for each server-day block, the
smaller of the server-day size limit and its inference and free blocks' limits, plus the assumed hidden hardware size;
`serverSize_le` bounds a server-day block's size by its capacity's limit and `serverInput_le` its input from its own past
plus its attested boundary's day block's input from the outside world by the input limit; `boundary_one_server` gives
back the server-day block's own bound where it is alone behind its boundary that day (`boundaryBlocks_one`); `capacity_binds` and `input_binds` show each refusing alone, at one gate and
at one bit, `covert_decides` a run the covert figure alone decides, and `wipe_count_decides` the wipe's registered
numbers and challenge count deciding the check; `check_wipeApplies` and `wipeApplies_of_check` give the wipe row's side
condition from the check, and `wipe_zero_refused` and `wipe_full_refused` show the check refusing a wipe of 0
challenges and a wipe whose residual is at or above what it fills. `boundary_one_server_output` is
`boundary_one_server` for output, and `wipeInputAtMost` proves the wipe's row at one wipe's error times the period's
wipes. `credited_excluded` refuses a server-day block
whose PoUW credited use is above its capacity, `credited_le_capacity` gives that comparison from `SpendIsWork` and
`ComputeRate`, and `credited_binds` shows it refusing at one gate. Those limits are per server-day block, and
`cut_multiplies` shows a cut into more server-day blocks passing where the uncut run does not; a server-day block is a
listed location and a registered day, so `serverDay_count_le` bounds their number by the listed locations times the days,
and `serverInput_total_le`, `freeWork_total_le` and `feasible_work_le_count` bound the run's totals by that count and the
limits.

The feasible set is not every run: `honest_feasible` puts a one-gate circuit's run in the feasible set of a one-block
run claiming size at most 1, and `two_not_feasible` keeps a two-gate circuit's run out of it. `top_blocks_classified`
records that no top block is in no class (the partitions' `kind` is total), so the degenerate case the reviews found
cannot be stated; nothing sits inside an inference block, the finest scale here, so the other cannot be stated either.
`cls_free_of_before_salt` makes a unit labelled inference that starts before its epoch's challenge tick a free block,
`freeChild_le` puts a free block's work under the free limit, and `salt_label_free` shows both on one run.
`realizesExcept_of_rows` gives `RealizesExcept` from the rows, and `blockBounds_old` shows its server-day clause
giving `OldServerClause`, every committed claim holding, at claims lists whose claims are at least the established bounds.
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

/-- **The compute capacity from the hardware description**: where every run meets `ComputeRate` over the listed locations
at the hardware description's rates (`rate`), every server-day block's size is at most its capacity (`capacity`: its
location's rate times its window's length), its location being listed (`kindIff`). -/
theorem capacityAtMost_compute (P : Runs L) (hcr : ∀ r, (P.actual r).ComputeRate P.listed (P.π r) P.rate) :
    P.CapacityAtMost Capacity.computeRate P.sizeOut P.capacity :=
  fun r i hki => sizeOut_le_compute P r i P.listed P.rate (hcr r) ((P.kindIff r i).mp hki)

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

/-- **Every computed gate at a listed location is computed during the period**: it lies in a server-day block
(`listed_gate_in_serverDay`), whose window is one of the period's registered days (`winDay`), and the days lie within
the period's span. -/
theorem listed_gate_in_period (P : Runs L) (r : P.Run) (g : Fin (P.actual r).N)
    (hg : g ∈ (P.actual r).computed) (X : Finset L) (hX : X ∈ P.listed) (hx : (P.actual r).x g ∈ X) :
    P.period.start ≤ (P.actual r).t g ∧ (P.actual r).t g < P.period.stop := by
  obtain ⟨i, hki, hgi⟩ := listed_gate_in_serverDay P r g hg X hX hx
  have hreg := P.insideOut r i hgi
  have hticks : ((P.winOut r i).ticks : Set ℕ)
      = Set.Ico (P.winOut r i).start ((P.winOut r i).start + (P.winOut r i).len) := by
    simp [Window.ticks]
  rw [hticks] at hreg
  simp only [Placed.region, Finset.mem_filter, Set.mem_Ico] at hreg
  obtain ⟨-, -, hlo, hhi⟩ := hreg
  obtain ⟨hs, hl⟩ := P.winDay r i hki
  simp only [TickSpan.Cut.part] at hs hl
  have hm := P.days.mono (Fin.castSucc_lt_succ (i := P.dayOf r i)).le
  refine ⟨?_, ?_⟩
  · calc P.period.start = P.days.boundary 0 := P.days.first.symm
      _ ≤ P.days.boundary (P.dayOf r i).castSucc := P.days.mono (Fin.zero_le _)
      _ = (P.winOut r i).start := hs.symm
      _ ≤ (P.actual r).t g := hlo
  · calc (P.actual r).t g < (P.winOut r i).start + (P.winOut r i).len := hhi
      _ = P.days.boundary (P.dayOf r i).succ := by rw [hs, hl]; exact Nat.add_sub_of_le hm
      _ ≤ P.days.boundary (Fin.last _) := P.days.mono (Fin.le_last _)
      _ = P.period.stop := P.days.last

open Classical in
/-- **No computed gate escapes both the server-day bounds and the hidden bound**: every computed gate lies in a
server-day block, which `BlockBounds` reads, or is at a leaf no listed location holds (`hiddenGates`, whose work
`lim.hidden` bounds, assumed). -/
theorem gate_serverDay_or_unlisted (P : Runs L) (r : P.Run) (g : Fin (P.actual r).N)
    (hg : g ∈ (P.actual r).computed) :
    (∃ i, P.kind r i = true ∧ g ∈ (P.outer r).gates i) ∨ g ∈ P.hiddenGates r := by
  by_cases h : ∃ X ∈ P.listed, (P.actual r).x g ∈ X
  · obtain ⟨X, hX, hx⟩ := h
    exact Or.inl (listed_gate_in_serverDay P r g hg X hX hx)
  · simp only [not_exists, not_and] at h
    exact Or.inr (Finset.mem_filter.mpr ⟨hg, h⟩)

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

/-- A unit's class is `hidden` exactly when its label is. -/
theorem cls_eq_hidden_iff (P : Runs L) (r : P.Run) (u : Fin (P.nIn r)) :
    P.cls r u = Class.hidden ↔ P.label r u = Class.hidden := by
  unfold Runs.cls
  split_ifs with h
  · rw [h.1]; decide
  · exact Iff.rfl

/-- **A unit's class is `hidden` exactly when its parent is hidden hardware**, from `labelHiddenIff`. -/
theorem cls_hiddenIff (P : Runs L) (r : P.Run) (u : Fin (P.nIn r)) :
    P.cls r u = Class.hidden ↔ P.kind r ((P.ref r).parent u) = false :=
  (cls_eq_hidden_iff P r u).trans (P.labelHiddenIff r u)

/-- **A unit labelled inference whose window starts before its epoch's challenge tick is a free block**, by the class's
definition. -/
theorem cls_free_of_before_salt (P : Runs L) (r : P.Run) (u : Fin (P.nIn r)) (hl : P.label r u = Class.inference)
    (h : (P.winIn r u).start < P.salt r u) : P.cls r u = Class.free := by
  unfold Runs.cls
  rw [if_pos ⟨hl, h⟩]

/-- **An inference block's window starts at or after its epoch's challenge tick**, by the class's definition. -/
theorem cls_inference_salt_le (P : Runs L) (r : P.Run) (u : Fin (P.nIn r)) (h : P.cls r u = Class.inference) :
    P.salt r u ≤ (P.winIn r u).start := by
  by_contra hlt
  have hl : P.label r u = Class.inference := by
    by_cases hc : P.label r u = Class.inference ∧ (P.winIn r u).start < P.salt r u
    · exact hc.1
    · unfold Runs.cls at h
      rwa [if_neg hc] at h
  rw [cls_free_of_before_salt P r u hl (Nat.lt_of_not_le hlt)] at h
  exact absurd h (by decide)

/-- The inference and free blocks directly inside a server-day block are the blocks directly inside it. -/
theorem children_split (P : Runs L) (r : P.Run) (i : Fin (P.nOut r)) (hki : P.kind r i = true) :
    P.children r i = P.infChildren r i ∪ P.freeChildren r i := by
  ext u
  simp only [Runs.children, Runs.infChildren, Runs.freeChildren, mem_filter, mem_union, mem_univ, true_and]
  constructor
  · intro hp
    have hnh : P.cls r u ≠ Class.hidden := by
      rw [ne_eq, cls_hiddenIff, hp]; simp [hki]
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
inference blocks outside `M` add up to at most `i`'s size. -/
theorem sizeExclusive_run (P : Runs L) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (i : Fin (P.nOut r)) (hki : P.kind r i = true) :
    (∑ u ∈ P.freeChildren r i, P.sizeIn r u) + (∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u) +
      ∑ u ∈ (P.infChildren r i).filter (· ∉ M), P.sizeIn r u
      ≤ P.sizeOut r i := by
  rw [sizeOut_eq_children_sum, children_split P r i hki, Finset.sum_union (infFree_disjoint P r i),
    ← Finset.sum_filter_add_sum_filter_not (P.infChildren r i) (· ∈ M) (P.sizeIn r)]
  omega

/-- **Declared work does not double as other work, for size**: inside a server-day block, for every set `M`, the free
blocks' and `M`'s inference blocks' work and the work of the inference blocks outside `M` add up to at most its own
size. -/
theorem sizeExclusive (P : Runs L) :
    P.DeclaredWorkExclusive
      (fun r M i => (∑ u ∈ P.freeChildren r i, P.sizeIn r u) +
        ∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u)
      (fun r M i => ∑ u ∈ (P.infChildren r i).filter (· ∉ M),
        P.sizeIn r u)
      P.sizeOut :=
  fun r _ M i hki => sizeExclusive_run P r M i hki

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
the work of `i`'s inference blocks outside `M`), `i`'s size at most `cap` (`hcap`) and `M`'s committed
size claims at most `tol` (`htol`), the free blocks' work, `M`'s inference blocks' work in `i` and `b` together are at
most `cap + tol`. All in gates at the run's pricing; no subtraction enters the conclusion, so no case is truncated. The
partitions give the rest (`sizeExclusive_run`). -/
theorem conversion_le (P : Runs L) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (i : Fin (P.nOut r)) (hki : P.kind r i = true) (b cap tol : ℕ)
    (huse : b - ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤
      ∑ u ∈ (P.infChildren r i).filter (· ∉ M), P.sizeIn r u)
    (hcap : P.sizeOut r i ≤ cap) (htol : ∑ u ∈ M, claimedSize (P.claimsIn r u) ≤ tol) :
    (∑ u ∈ P.freeChildren r i, P.sizeIn r u) + (∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u) + b ≤
      cap + tol := by
  have hex := sizeExclusive_run P r M i hki
  have hfree := freeBlockAtMost_run huse hcap hex
  have hM := tolChildren_le P r M i
  omega

/-- **The conversion in the form the check reads**: the free blocks' and `M`'s inference blocks' work in `i` is
at most `cap − b + tol` (from `conversion_le`; `cap + tol − b ≤ cap − b + tol` over the naturals). -/
theorem conversion_le_sub (P : Runs L) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (i : Fin (P.nOut r)) (hki : P.kind r i = true) (b cap tol : ℕ)
    (huse : b - ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤
      ∑ u ∈ (P.infChildren r i).filter (· ∉ M), P.sizeIn r u)
    (hcap : P.sizeOut r i ≤ cap) (htol : ∑ u ∈ M, claimedSize (P.claimsIn r u) ≤ tol) :
    (∑ u ∈ P.freeChildren r i, P.sizeIn r u) + (∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u) ≤
      cap - b + tol := by
  have := conversion_le P r M i hki b cap tol huse hcap htol
  omega

/-- **The free work bound holds from the conversion**: in one run, PoUW's conclusion for `M` on the run's credited use in
every server-day block, every server-day block's size at most its capacity and `M`'s committed size claims at most the
run's tolerance give `FreeWorkAtMost r M`. -/
theorem freeWorkAtMost_of_conversion (P : Runs L) (r : P.Run)
    (M : Finset (Fin (P.nIn r)))
    (huse : ∀ i, P.kind r i = true →
      P.credited r i - ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤
        ∑ u ∈ (P.infChildren r i).filter (· ∉ M), P.sizeIn r u)
    (hcap : ∀ i, P.kind r i = true → P.sizeOut r i ≤ P.capacity r i)
    (htol : ∑ u ∈ M, claimedSize (P.claimsIn r u) ≤ P.tol) : P.FreeWorkAtMost r M :=
  fun i hki => conversion_le_sub P r M i hki (P.credited r i) (P.capacity r i) P.tol (huse i hki) (hcap i hki) htol

/-- **The conversion excludes**: in one run, where the free blocks' work, `M`'s inference blocks' work in `i` and `b`
exceed `cap + tol`, one of PoUW's conclusion for `M`, the capacity and the tolerance fails. -/
theorem conversion_excludes (P : Runs L) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (i : Fin (P.nOut r)) (hki : P.kind r i = true) (b cap tol : ℕ)
    (hbig : cap + tol < (∑ u ∈ P.freeChildren r i, P.sizeIn r u) +
      (∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u) + b) :
    ¬ ((b - ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤
        ∑ u ∈ (P.infChildren r i).filter (· ∉ M), P.sizeIn r u) ∧
      P.sizeOut r i ≤ cap ∧ ∑ u ∈ M, claimedSize (P.claimsIn r u) ≤ tol) := by
  rintro ⟨huse, hcap, htol⟩
  have := conversion_le P r M i hki b cap tol huse hcap htol
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
theorem conversion_pooled_le (P : Runs L) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (b cap : Fin (P.nOut r) → ℕ) (tol : ℕ)
    (huse : ∀ i, P.kind r i = true → b i - ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤
      ∑ u ∈ (P.infChildren r i).filter (· ∉ M), P.sizeIn r u)
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
    have hex := sizeExclusive_run P r M i hki
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
theorem conversion_pooled_le_sub (P : Runs L) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (b cap : Fin (P.nOut r) → ℕ) (tol : ℕ)
    (huse : ∀ i, P.kind r i = true → b i - ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤
      ∑ u ∈ (P.infChildren r i).filter (· ∉ M), P.sizeIn r u)
    (hcap : ∀ i, P.kind r i = true → P.sizeOut r i ≤ cap i)
    (htol : ∑ u ∈ M, claimedSize (P.claimsIn r u) ≤ tol) :
    ∑ i ∈ univ.filter (fun i => P.kind r i = true),
        ((∑ u ∈ P.freeChildren r i, P.sizeIn r u) + ∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u)
      ≤ (∑ i ∈ univ.filter (fun i => P.kind r i = true), cap i) -
          (∑ i ∈ univ.filter (fun i => P.kind r i = true), b i) + tol := by
  have h := conversion_pooled_le P r M b cap tol huse hcap htol
  rw [Finset.sum_add_distrib] at h
  omega

/-- **The pooled free work bound holds from the pooled conversion**: in one run, PoUW's conclusion for `M` on the run's
credited use in every server-day block, every server-day block's size at most its capacity and `M`'s committed size
claims at most the run's tolerance give `FreeWorkAtMostPooled r M`: the period's free and tolerance work at most the
run's pooled free bound (`pooledFreeBound`). -/
theorem freeWorkAtMostPooled_of_conversion (P : Runs L) (r : P.Run)
    (M : Finset (Fin (P.nIn r)))
    (huse : ∀ i, P.kind r i = true →
      P.credited r i - ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤
        ∑ u ∈ (P.infChildren r i).filter (· ∉ M), P.sizeIn r u)
    (hcap : ∀ i, P.kind r i = true → P.sizeOut r i ≤ P.capacity r i)
    (htol : ∑ u ∈ M, claimedSize (P.claimsIn r u) ≤ P.tol) : P.FreeWorkAtMostPooled r M :=
  conversion_pooled_le_sub P r M (P.credited r) (P.capacity r) P.tol huse hcap htol

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
is `realizes_realizesExcept`; neither says anything of the free and tolerance work, which `FreeWorkAtMost` bounds on its
own. -/
theorem realizesExcept_empty_realizes (P : Runs L) (r : P.Run) : P.RealizesExcept r ∅ → P.Realizes r := by
  rintro ⟨h1, h2, h3, h4, h5, h6, -⟩
  refine ⟨h1, h2, h3, fun u => ?_, h6⟩
  by_cases hc : P.cls r u = Class.inference
  · exact h4 u hc (Finset.notMem_empty u)
  · exact h5 u hc

/-- **`Realizes` gives `RealizesExcept ∅`**: with no tolerance every block holds its claims, so the inference-claim
clauses split by class. The free work bound is no clause here (`FreeWorkAtMost` bounds it on its own). -/
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

theorem holds_size_mono {A : Placed L} {π : Fin A.N → ℕ} {θ : Finset (Fin A.N)} {X : Finset L}
    {b : Verity.Block A.toCircuit} {k k' : ℕ} (h : A.Holds π θ X b (.sizeAtMost k)) (hk : k ≤ k') :
    A.Holds π θ X b (.sizeAtMost k') := by
  simp only [Placed.Holds] at h ⊢
  exact h.trans hk

theorem holds_input_mono {A : Placed L} {π : Fin A.N → ℕ} {θ : Finset (Fin A.N)} {X : Finset L}
    {b : Verity.Block A.toCircuit} {src : List Source} {k k' : ℕ} (h : A.Holds π θ X b (.inputAtMost src k))
    (hk : k ≤ k') : A.Holds π θ X b (.inputAtMost src k') := by
  simp only [Placed.Holds] at h ⊢
  exact h.trans hk

/-- The check's comparisons, one conjunct each, in its order. -/
theorem check_spec {lim : InferenceOnly.Limits} {P : Runs L} {r : P.Run} (hc : InferenceOnly.check lim P r = true) :
    (univ.filter fun u => P.cls r u = Class.inference).card ≤ lim.n ∧
    (∀ u, P.cls r u = Class.inference →
      (P.claimsIn r u).any InferenceOnly.isSize = true ∧ claimedSize (P.claimsIn r u) ≤ lim.c ∧
        (P.claimsIn r u).any (InferenceOnly.freshAtMost lim.k) = true) ∧
    (univ.filter fun u => P.cls r u = Class.free).card ≤ lim.m ∧
    (∀ i, P.kind r i = true → P.credited r i ≤ P.capacity r i) ∧
    (∀ i, P.kind r i = true → P.capacity r i - P.credited r i + P.tol ≤ lim.freeSize) ∧
    (∀ i, P.kind r i = true → P.capacity r i ≤ lim.cS) ∧
    (∀ i, P.kind r i = true → P.boundaryPastBound r (P.bnd r i) (P.dayOf r i) +
          (P.ingressCount r (P.bnd r i) (P.dayOf r i) + P.covert (P.bnd r i)) ≤ lim.kS) ∧
    (∀ i, P.kind r i = true → P.WipeApplies P.wipeChallenges r i) ∧
    (∀ u, P.cls r u ≠ Class.inference → (P.claimsIn r u).isEmpty = true) := by
  simp only [InferenceOnly.check, Bool.and_eq_true, decide_eq_true_eq] at hc
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩, h7⟩, h8⟩, h9⟩ := hc
  exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9⟩

theorem check_infClaims {lim : InferenceOnly.Limits} {P : Runs L} {r : P.Run}
    (hc : InferenceOnly.check lim P r = true) (u : Fin (P.nIn r)) (hu : P.cls r u = Class.inference) :
    (P.claimsIn r u).any InferenceOnly.isSize = true ∧ claimedSize (P.claimsIn r u) ≤ lim.c ∧
      (P.claimsIn r u).any (InferenceOnly.freshAtMost lim.k) = true :=
  (check_spec hc).2.1 u hu

theorem check_credited {lim : InferenceOnly.Limits} {P : Runs L} {r : P.Run}
    (hc : InferenceOnly.check lim P r = true) (i : Fin (P.nOut r)) (hki : P.kind r i = true) :
    P.credited r i ≤ P.capacity r i :=
  (check_spec hc).2.2.2.1 i hki

theorem check_freeSize {lim : InferenceOnly.Limits} {P : Runs L} {r : P.Run}
    (hc : InferenceOnly.check lim P r = true) (i : Fin (P.nOut r)) (hki : P.kind r i = true) :
    P.capacity r i - P.credited r i + P.tol ≤ lim.freeSize :=
  (check_spec hc).2.2.2.2.1 i hki

theorem check_capacity {lim : InferenceOnly.Limits} {P : Runs L} {r : P.Run}
    (hc : InferenceOnly.check lim P r = true) (i : Fin (P.nOut r)) (hki : P.kind r i = true) :
    P.capacity r i ≤ lim.cS :=
  (check_spec hc).2.2.2.2.2.1 i hki

theorem check_input {lim : InferenceOnly.Limits} {P : Runs L} {r : P.Run}
    (hc : InferenceOnly.check lim P r = true) (i : Fin (P.nOut r)) (hki : P.kind r i = true) :
    P.boundaryPastBound r (P.bnd r i) (P.dayOf r i) +
      (P.ingressCount r (P.bnd r i) (P.dayOf r i) + P.covert (P.bnd r i)) ≤ lim.kS :=
  (check_spec hc).2.2.2.2.2.2.1 i hki

/-- **A run that passes the check meets the wipe's side condition at every server-day block**: its location's label
count and its wipe's challenge count are positive and its wipe residual is below the label count times the label width
(bits), so the wipe row's hypothesis on accepted runs follows from the check passing on them. -/
theorem check_wipeApplies {lim : InferenceOnly.Limits} {P : Runs L} {r : P.Run}
    (hc : InferenceOnly.check lim P r = true) (i : Fin (P.nOut r)) (hki : P.kind r i = true) :
    P.WipeApplies P.wipeChallenges r i :=
  (check_spec hc).2.2.2.2.2.2.2.1 i hki

/-- **The wipe row's side condition from the check**: when the check passes on every accepted run, every server-day
block of every accepted run meets `WipeApplies` at its challenge count. -/
theorem wipeApplies_of_check {lim : InferenceOnly.Limits} {P : Runs L}
    (hcheck : ∀ r, P.accepts r → InferenceOnly.check lim P r = true) :
    ∀ r, P.accepts r → ∀ i, P.kind r i = true → P.WipeApplies P.wipeChallenges r i :=
  fun r ha i hki => check_wipeApplies (hcheck r ha) i hki

theorem check_innerEmpty {lim : InferenceOnly.Limits} {P : Runs L} {r : P.Run}
    (hc : InferenceOnly.check lim P r = true) (u : Fin (P.nIn r)) (hu : P.cls r u ≠ Class.inference) :
    (P.claimsIn r u).isEmpty = true :=
  (check_spec hc).2.2.2.2.2.2.2.2 u hu

/-- **The check bounds a server-day block's size**: under the check and `RealizesExcept`, a server-day block's size is at
most its capacity, which the check caps at `lim.cS` gates. -/
theorem serverSize_le (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run) (M : Finset (Fin (P.nIn r)))
    (hc : InferenceOnly.check lim P r = true) (hreal : P.RealizesExcept r M) (i : Fin (P.nOut r))
    (hki : P.kind r i = true) : P.sizeOut r i ≤ lim.cS := by
  obtain ⟨-, -, -, -, -, hout, -⟩ := hreal
  have h : P.sizeOut r i ≤ P.capacity r i := (hout i hki).1
  exact h.trans (check_capacity hc i hki)

/-- A server-day block's wipe bound is at most the sum over its attested boundary's server-day blocks that day. -/
theorem pastBound_le_boundary (P : Runs L) (r : P.Run) (i : Fin (P.nOut r)) (hki : P.kind r i = true) :
    P.pastBound P.wipeEgress P.wipeChallenges r i ≤ P.boundaryPastBound r (P.bnd r i) (P.dayOf r i) := by
  unfold Runs.boundaryPastBound
  exact Finset.single_le_sum (f := fun j => P.pastBound P.wipeEgress P.wipeChallenges r j)
    (fun _ _ => Nat.zero_le _) (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hki, rfl, rfl⟩)

/-- **The limits bind what comes in**: under the check and `RealizesExcept`, a server-day block's input from its own
past plus its attested boundary's day block's input from the outside world is at most `lim.kS` bits. Its input from
another block at its location during its window (`.block`) is not bounded here, and neither is its own input from the
outside world where its boundary holds other servers: that is bounded only through the union. -/
theorem serverInput_le (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run) (M : Finset (Fin (P.nIn r)))
    (hc : InferenceOnly.check lim P r = true) (hreal : P.RealizesExcept r M) (i : Fin (P.nOut r))
    (hki : P.kind r i = true) :
    (P.actual r).blockInput (P.θ r) (P.locOut r i) (P.outerBlock r i) [.past] +
      (P.actual r).blockInput (P.θ r) (P.boundaryLoc r (P.bnd r i) (P.dayOf r i))
        (P.boundaryBlock r (P.bnd r i) (P.dayOf r i)) [.world] ≤ lim.kS := by
  obtain ⟨-, -, -, -, -, hout, -⟩ := hreal
  obtain ⟨-, hp, hw⟩ := hout i hki
  have h1 : (P.actual r).blockInput (P.θ r) (P.locOut r i) (P.outerBlock r i) [.past] ≤
      P.pastBound P.wipeEgress P.wipeChallenges r i := hp
  have h2 : (P.actual r).blockInput (P.θ r) (P.boundaryLoc r (P.bnd r i) (P.dayOf r i))
      (P.boundaryBlock r (P.bnd r i) (P.dayOf r i)) [.world] ≤
      P.ingressCount r (P.bnd r i) (P.dayOf r i) + P.covert (P.bnd r i) := hw
  have := pastBound_le_boundary P r i hki
  have := check_input hc i hki
  omega

/-- A block's input from the outside world reads its gates and location only, not its window. -/
theorem blockInput_world_gates {A : Placed L} (θ : Finset (Fin A.N)) (X : Finset L) {b b' : Verity.Block A.toCircuit}
    (h : b.gates = b'.gates) : A.blockInput θ X b [.world] = A.blockInput θ X b' [.world] := by
  simp only [Placed.blockInput, List.toFinset_cons, List.toFinset_nil, insert_empty_eq, Finset.singleton_biUnion,
    Placed.sourceGates, h]

/-- **A one-server attested boundary gives back the server-day block's statement**: when server-day block `i` is the
only one behind its attested boundary on its day, its boundary's day block has an input bound from the outside world
exactly when block `i` has it, at every bound `k` (bits). -/
theorem boundary_one_server (P : Runs L) (r : P.Run) (i : Fin (P.nOut r))
    (hone : P.boundaryBlocks r (P.bnd r i) (P.dayOf r i) = {i}) (k : ℕ) :
    P.HoldsBoundary r (P.bnd r i) (P.dayOf r i) (.inputAtMost [.world] k) ↔
      P.HoldsOut r i (.inputAtMost [.world] k) := by
  have hloc : P.boundaryLoc r (P.bnd r i) (P.dayOf r i) = P.locOut r i := by
    simp only [Runs.boundaryLoc, hone, Finset.singleton_biUnion]
  have hg : (P.boundaryBlock r (P.bnd r i) (P.dayOf r i)).gates = (P.outerBlock r i).gates := by
    simp only [Runs.boundaryBlock, Runs.outerBlock, hone, Finset.singleton_biUnion]
  simp only [Runs.HoldsBoundary, Runs.HoldsOut, Placed.Holds, hloc, blockInput_world_gates _ _ hg]

/-- A block's output to the outside world reads its gates and location only, not its window. -/
theorem blockOutput_gates {A : Placed L} (X : Finset L) {b b' : Verity.Block A.toCircuit} (h : b.gates = b'.gates) :
    A.blockOutput X b = A.blockOutput X b' := by
  simp only [Placed.blockOutput, h]

/-- **A one-server attested boundary gives back the server-day block's output statement**: when server-day block `i` is
the only one behind its attested boundary on its day, its boundary's day block has an output bound to the outside world
exactly when block `i` has it, at every bound `k` (bits). -/
theorem boundary_one_server_output (P : Runs L) (r : P.Run) (i : Fin (P.nOut r))
    (hone : P.boundaryBlocks r (P.bnd r i) (P.dayOf r i) = {i}) (k : ℕ) :
    P.HoldsBoundary r (P.bnd r i) (P.dayOf r i) (.outputAtMost k) ↔ P.HoldsOut r i (.outputAtMost k) := by
  have hloc : P.boundaryLoc r (P.bnd r i) (P.dayOf r i) = P.locOut r i := by
    simp only [Runs.boundaryLoc, hone, Finset.singleton_biUnion]
  have hg : (P.boundaryBlock r (P.bnd r i) (P.dayOf r i)).gates = (P.outerBlock r i).gates := by
    simp only [Runs.boundaryBlock, Runs.outerBlock, hone, Finset.singleton_biUnion]
  simp only [Runs.HoldsBoundary, Runs.HoldsOut, Placed.Holds, hloc, blockOutput_gates _ hg]

/-- **A boundary whose listed location is its own holds one server a day**: when no other server-day block's location
shares server-day block `i`'s attested boundary, `i` is the only server-day block behind it on its day (`oneBlock`). -/
theorem boundaryBlocks_one (P : Runs L) (r : P.Run) (i : Fin (P.nOut r)) (hki : P.kind r i = true)
    (hinj : ∀ j, P.kind r j = true → P.bnd r j = P.bnd r i → P.locOut r j = P.locOut r i) :
    P.boundaryBlocks r (P.bnd r i) (P.dayOf r i) = {i} := by
  ext j
  simp only [Runs.boundaryBlocks, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
  constructor
  · rintro ⟨hkj, hb, hd⟩
    exact P.oneBlock r j i hkj hki (hinj j hkj hb) hd
  · rintro rfl
    exact ⟨hki, rfl, rfl⟩

/-- **A server-day block whose capacity is above `cS` fails the check**, whatever the rest of the run (gates at the run's
pricing). -/
theorem capacity_excluded (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run) (i : Fin (P.nOut r))
    (hki : P.kind r i = true) (hbig : lim.cS < P.capacity r i) : InferenceOnly.check lim P r = false := by
  by_contra hc
  exact absurd (check_capacity (by simpa using hc) i hki) (by omega)

/-- **A server-day block whose input bounds sum above `kS` fails the check**, whatever the rest of the run: the wipe's
bounds summed over its attested boundary's server-day blocks that day, plus the boundary's ingress count that day, plus
its covert figure, above `lim.kS` bits. -/
theorem serverInput_excluded (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run) (i : Fin (P.nOut r))
    (hki : P.kind r i = true)
    (hbig : lim.kS < P.boundaryPastBound r (P.bnd r i) (P.dayOf r i) +
      (P.ingressCount r (P.bnd r i) (P.dayOf r i) + P.covert (P.bnd r i))) :
    InferenceOnly.check lim P r = false := by
  by_contra hc
  exact absurd (check_input (by simpa using hc) i hki) (by omega)

/-- **A server-day block credited above its capacity fails the check**, whatever the rest of the run: PoUW's credited
use of `i` above `i`'s capacity, in gates at the run's pricing. -/
theorem credited_excluded (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run) (i : Fin (P.nOut r))
    (hki : P.kind r i = true) (hbig : P.capacity r i < P.credited r i) :
    InferenceOnly.check lim P r = false := by
  by_contra hc
  exact absurd (check_credited (by simpa using hc) i hki) (by omega)

/-- **The rows meet the credited comparison**: in an accepted run where `SpendIsWork` holds (read at the empty tolerance
set) and every run meets `ComputeRate` over the listed locations at the hardware description's rates, every server-day
block's PoUW credited use is at most its capacity (gates at the run's pricing). So the comparison refuses no run of
which both premises are true. -/
theorem credited_le_capacity (P : Runs L) (shown : P.Run → Prop)
    (hspend : P.SpendIsWork shown) (hrate : ∀ r, (P.actual r).ComputeRate P.listed (P.π r) P.rate)
    (r : P.Run) (ha : P.accepts r) (hs : shown r) (i : Fin (P.nOut r)) (hki : P.kind r i = true) :
    P.credited r i ≤ P.capacity r i := by
  have h := hspend r ha hs ∅ i hki
  have h0 : ∑ u ∈ (P.infChildren r i).filter (· ∈ (∅ : Finset (Fin (P.nIn r)))), claimedSize (P.claimsIn r u) = 0 :=
    Finset.sum_eq_zero fun u hu => absurd (Finset.mem_filter.mp hu).2 (Finset.notMem_empty u)
  have h1 : ∑ u ∈ (P.infChildren r i).filter
      (· ∉ (∅ : Finset (Fin (P.nIn r)))), P.sizeIn r u ≤ P.sizeOut r i :=
    sizeAdds P r i _ fun u hu => by
      simp only [Runs.infChildren, Finset.mem_filter, Finset.mem_univ, true_and] at hu
      exact hu.1.1
  have h2 := capacityAtMost_compute P hrate r i hki
  omega

/-- **The check bites, on the path the chain reads**: under a run that passes the inference-only check and realizes its
claims outside a tolerance set `M` (`RealizesExcept`, not `Realizes`), with the work at unlisted locations at most
`lim.hidden` (assumed), the whole circuit's work is at most, for each server-day block, the smaller of the server-day
size limit `lim.cS` (its size is at most its capacity, which the check caps at `lim.cS`) and the count of its inference blocks
**outside** `M` at the size cap `lim.c` (each block's committed size claim, which the check caps at `lim.c`) plus the
free work limit `lim.freeSize`. The bound is in the limits alone: the free blocks' and `M`'s inference blocks' work is
at most capacity − credited use + tolerance (the hypothesis `FreeWorkAtMost`), which the check caps at `lim.freeSize`.
The tolerance `M` is a bounded term here, not an unbounded quantifier: `M`'s blocks are counted inside the free limit,
not as inference credit. -/
theorem feasible_work_le (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run)
    (M : Finset (Fin (P.nIn r)))
    (hc : InferenceOnly.check lim P r = true) (hreal : P.RealizesExcept r M)
    (hfree : P.FreeWorkAtMost r M)
    (hhidden : (P.actual r).work (P.π r) (P.hiddenGates r) ≤ lim.hidden) :
    (P.actual r).work (P.π r) (P.actual r).computed ≤
      (∑ i ∈ univ.filter (fun i => P.kind r i = true),
        min lim.cS (((P.infChildren r i).filter (fun u => u ∉ M)).card * lim.c + lim.freeSize)) +
        lim.hidden := by
  have hserver := serverSize_le lim P r M hc hreal
  have hinf := check_infClaims hc
  have hfcCheck := check_freeSize hc
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
    (hfree : P.FreeWorkAtMost r M)
    (hhidden : (P.actual r).work (P.π r) (P.hiddenGates r) ≤ lim.hidden) :
    (P.actual r).work (P.π r) (P.actual r).computed ≤
      (∑ i ∈ univ.filter (fun i => P.kind r i = true),
        min lim.cS ((P.infChildren r i).card * lim.c + lim.freeSize)) + lim.hidden := by
  refine le_trans (feasible_work_le lim P r M hc hreal hfree hhidden) (add_le_add ?_ le_rfl)
  refine Finset.sum_le_sum fun i _ => min_le_min le_rfl (add_le_add (Nat.mul_le_mul_right _ ?_) le_rfl)
  exact Finset.card_le_card (Finset.filter_subset _ _)

/-! ## How many server-day blocks -/

/-- **A run has at most the listed locations times the days of server-day blocks**: each server-day block is a listed
location (`kindIff`) and a day (`dayOf`), and no two share both (`oneBlock`). `Ls` is any finite family holding every
listed location; listed locations that overlap each count. The count is servers times days only where the listed
locations are pairwise disjoint, which belongs to the hardware description (`Hardware.units` in
`Registration/Physical.lean`); no statement here changes with it. -/
theorem serverDay_count_le (P : Runs L) (r : P.Run) (Ls : Finset (Finset L)) (hLs : ∀ X ∈ P.listed, X ∈ Ls) :
    (univ.filter (fun i => P.kind r i = true)).card ≤ Ls.card * P.nDays := by
  calc (univ.filter (fun i => P.kind r i = true)).card ≤ (Ls ×ˢ (univ : Finset (Fin P.nDays))).card := by
        refine Finset.card_le_card_of_injOn (fun i => (P.locOut r i, P.dayOf r i)) ?_ ?_
        · intro i hi
          have hki := (Finset.mem_filter.mp hi).2
          exact Finset.mem_product.mpr ⟨hLs _ ((P.kindIff r i).mp hki), Finset.mem_univ _⟩
        · intro i hi j hj hij
          simp only [Prod.mk.injEq] at hij
          exact P.oneBlock r i j (Finset.mem_filter.mp hi).2 (Finset.mem_filter.mp hj).2 hij.1 hij.2
    _ = Ls.card * P.nDays := by rw [Finset.card_product, Finset.card_univ, Fintype.card_fin]

/-- **The run's input from the server-day blocks' own past and their boundaries' input from the outside world, in the
limits and the count**: under the check and `RealizesExcept`, each server-day block's input from its own past plus its
attested boundary's day block's input from the outside world, summed over the server-day blocks, is at most the listed
locations times the days times `lim.kS` bits. The count is servers times days only where the listed locations are pairwise disjoint, which
belongs to the hardware description (`Hardware.units` in `Registration/Physical.lean`); no statement here changes with
it. -/
theorem serverInput_total_le (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run) (M : Finset (Fin (P.nIn r)))
    (hc : InferenceOnly.check lim P r = true) (hreal : P.RealizesExcept r M)
    (Ls : Finset (Finset L)) (hLs : ∀ X ∈ P.listed, X ∈ Ls) :
    ∑ i ∈ univ.filter (fun i => P.kind r i = true),
        ((P.actual r).blockInput (P.θ r) (P.locOut r i) (P.outerBlock r i) [.past] +
          (P.actual r).blockInput (P.θ r) (P.boundaryLoc r (P.bnd r i) (P.dayOf r i))
            (P.boundaryBlock r (P.bnd r i) (P.dayOf r i)) [.world]) ≤
      Ls.card * P.nDays * lim.kS := by
  calc _ ≤ ∑ _i ∈ univ.filter (fun i => P.kind r i = true), lim.kS :=
        Finset.sum_le_sum fun i hi => serverInput_le lim P r M hc hreal i (Finset.mem_filter.mp hi).2
    _ = (univ.filter (fun i => P.kind r i = true)).card * lim.kS := by rw [Finset.sum_const, smul_eq_mul]
    _ ≤ _ := Nat.mul_le_mul_right _ (serverDay_count_le P r Ls hLs)

/-- **The run's free and tolerance work, in the limits and the count**: under the check and `FreeWorkAtMost`, the
server-day blocks' free blocks' work and `M`'s inference blocks' work, summed over the run, are at most the listed
locations times the days times `lim.freeSize` gates. The count is servers times days only where the listed locations are
pairwise disjoint, which belongs to the hardware description (`Hardware.units` in `Registration/Physical.lean`); no
statement here changes with it. -/
theorem freeWork_total_le (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run) (M : Finset (Fin (P.nIn r)))
    (hc : InferenceOnly.check lim P r = true) (hfree : P.FreeWorkAtMost r M)
    (Ls : Finset (Finset L)) (hLs : ∀ X ∈ P.listed, X ∈ Ls) :
    ∑ i ∈ univ.filter (fun i => P.kind r i = true),
        ((∑ u ∈ P.freeChildren r i, P.sizeIn r u) + ∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u) ≤
      Ls.card * P.nDays * lim.freeSize := by
  have hfcCheck := check_freeSize hc
  calc _ ≤ ∑ _i ∈ univ.filter (fun i => P.kind r i = true), lim.freeSize :=
        Finset.sum_le_sum fun i hi => by
          have hki := (Finset.mem_filter.mp hi).2
          exact (hfree i hki).trans (hfcCheck i hki)
    _ = (univ.filter (fun i => P.kind r i = true)).card * lim.freeSize := by rw [Finset.sum_const, smul_eq_mul]
    _ ≤ _ := Nat.mul_le_mul_right _ (serverDay_count_le P r Ls hLs)

/-- **The check bites, in the limits and the count**: `feasible_work_le_total`'s sum has at most the listed locations
times the days terms, each at most `lim.cS` gates, so the whole circuit's work is at most that product times `lim.cS`
plus `lim.hidden` (assumed). The count is servers times days only where the listed locations are pairwise disjoint,
which belongs to the hardware description (`Hardware.units` in `Registration/Physical.lean`); no statement here changes
with it. -/
theorem feasible_work_le_count (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run)
    (M : Finset (Fin (P.nIn r)))
    (hc : InferenceOnly.check lim P r = true) (hreal : P.RealizesExcept r M)
    (hfree : P.FreeWorkAtMost r M)
    (hhidden : (P.actual r).work (P.π r) (P.hiddenGates r) ≤ lim.hidden)
    (Ls : Finset (Finset L)) (hLs : ∀ X ∈ P.listed, X ∈ Ls) :
    (P.actual r).work (P.π r) (P.actual r).computed ≤ Ls.card * P.nDays * lim.cS + lim.hidden := by
  refine (feasible_work_le_total lim P r M hc hreal hfree hhidden).trans (Nat.add_le_add_right ?_ _)
  calc _ ≤ ∑ _i ∈ univ.filter (fun i => P.kind r i = true), lim.cS := Finset.sum_le_sum fun i _ => min_le_left _ _
    _ = (univ.filter (fun i => P.kind r i = true)).card * lim.cS := by rw [Finset.sum_const, smul_eq_mul]
    _ ≤ _ := Nat.mul_le_mul_right _ (serverDay_count_le P r Ls hLs)

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

/-- The span `p` as one day. -/
def oneDay (p : TickSpan) : p.Cut 1 where
  boundary j := p.start + j.val * p.len
  mono _ _ h := Nat.add_le_add_left (Nat.mul_le_mul_right _ h) _
  first := by simp
  last := by simp [TickSpan.stop]

open Classical in
/-- One run of the circuit `placed n` at credited use `cr` gates and tolerance `tl` gates: no fresh input, one server-day
block holding the whole circuit as one inference block, claimed of size at most 1 gate and 0 bits of fresh input; the
server-day block's wipe egress, its attested boundary's ingress count and covert figure are 0 bits, and its wipe is one
label of 4 bits at one challenge and soundness exponent 0 bits, so its wipe bound is `demoPast` bits. The server-day block's window is ticks 1 to `n`, where `placed n` computes its `n` gates,
one a tick; its location's rate is 1 gate a tick, so its capacity is `n` gates, its work. -/
noncomputable def demoRuns (n cr tl : ℕ) : Runs Unit where
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
  winOut _ _ := ⟨1, n⟩
  period := ⟨1, n⟩
  nDays := 1
  days := oneDay _
  dayOf _ _ := 0
  winDay _ _ _ := by simp [TickSpan.Cut.part, oneDay]
  oneBlock _ i j _ _ _ _ := Subsingleton.elim (α := Fin 1) i j
  rate _ := 1
  credited _ _ := cr
  tol := tl
  wipeLabels _ := 1
  labelBits _ := 4
  uncovered _ := 0
  wipeError := 0
  wipeEgress _ _ := 0
  wipeChallenges _ _ := 1
  boundary _ := 0
  ingressCount _ _ _ := 0
  covert _ := 0
  nIn _ := 1
  inner _ := Partition.whole (placed n).toCircuit
  ref _ := { parent := id, sub := fun g => by cases (Partition.whole (placed n).toCircuit).unit g <;> simp }
  label _ _ := Class.inference
  locIn _ _ := {()}
  winIn _ _ := ⟨0, n + 1⟩
  claimsIn _ _ := [.sizeAtMost 1, .inputAtMost [.past, .block, .world] 0]
  labelHiddenIff _ _ := by decide
  salt _ _ := 0
  insideOut := fun _ i g hg => by
    rw [Partition.gates, mem_filter] at hg
    have hin : (placed n).isInput g = false := by
      by_contra hc
      simp only [Bool.not_eq_false] at hc
      have h2 := hg.2
      simp [Partition.whole, hc] at h2
    simp only [Placed.region, Circuit.computed, Finset.mem_filter, mem_univ, true_and,
      Finset.mem_coe, Window.ticks, Finset.mem_Ico]
    refine ⟨hin, by simp [placed], ?_, ?_⟩
    all_goals
      have := g.isLt
      simp only [placed, line, decide_eq_false_iff_not] at this hin ⊢
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

/-- The demos' wipe bound on a server-day block's input from its own past, in bits: the wipe residual at one label of
4 bits, one challenge and soundness exponent 0 bits, with no egress and no uncovered storage; it is 2 bits, below the
4 bits the wipe fills, so the check's wipe comparison passes it. -/
def demoPast : ℕ := Pous.StreamedWipe.wipeResidual 1 1 4 0

/-- The one-gate demo's server-day block and its attested boundary's day block hold the three bounds. -/
theorem demo_blockBounds_one : (demoRuns 1 1 0).BlockBounds () := by
  intro i hki
  have hone := boundaryBlocks_one _ _ i hki fun _ _ _ => rfl
  refine ⟨?_, holds_input_mono (k := 0) ?_ (Nat.zero_le _),
    (boundary_one_server _ _ i hone _).mpr (holds_input_mono (k := 0) ?_ (Nat.zero_le _))⟩
  all_goals clear hone hki
  all_goals fin_cases i
  · show (placed 1).work _ _ ≤ 1
    simp only [demoRuns, Runs.outerBlock, Partition.whole, Partition.gates, Placed.work]
    decide
  all_goals
    show (placed 1).blockInput _ _ _ _ ≤ 0
    simp only [demoRuns, Runs.outerBlock, Placed.blockInput, Circuit.into, Circuit.width, placed, line]
    decide

/-- The two-gate demo's server-day block and its attested boundary's day block hold the three bounds. -/
theorem demo_blockBounds_two : (demoRuns 2 1 1).BlockBounds () := by
  intro i hki
  have hone := boundaryBlocks_one _ _ i hki fun _ _ _ => rfl
  refine ⟨?_, holds_input_mono (k := 0) ?_ (Nat.zero_le _),
    (boundary_one_server _ _ i hone _).mpr (holds_input_mono (k := 0) ?_ (Nat.zero_le _))⟩
  all_goals clear hone hki
  all_goals fin_cases i
  · show (placed 2).work _ _ ≤ 2
    simp only [demoRuns, Runs.outerBlock, Partition.whole, Partition.gates, Placed.work]
    decide
  all_goals
    show (placed 2).blockInput _ _ _ _ ≤ 0
    simp only [demoRuns, Runs.outerBlock, Placed.blockInput, Circuit.into, Circuit.width, placed, line]
    decide

/-- **The honest witness**: the one-gate circuit's run realizes the claims of `demoRuns 1 1 0`. -/
theorem honest_feasible : (demoRuns 1 1 0).Realizes () := by
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
  · exact demo_blockBounds_one

/-- The free blocks of `demoRuns n cr tl` are none (its one inner block is an inference block). -/
theorem demo_freeChildren_empty (n cr tl : ℕ) (i : Fin ((demoRuns n cr tl).nOut ())) :
    (demoRuns n cr tl).freeChildren () i = ∅ := by
  simp only [Runs.freeChildren]
  apply Finset.filter_false_of_mem
  intro u _
  rw [show (demoRuns n cr tl).cls () u = Class.inference from rfl]
  exact fun h => absurd h.2 (by decide)

/-- **A run outside the feasible set**: the two-gate circuit's run does not realize the claims of `demoRuns 2 0 0`,
whose one inference block is then of size at least 2. -/
theorem two_not_feasible : ¬ (demoRuns 2 0 0).Realizes () := by
  rintro ⟨-, -, -, hcl_in, -⟩
  have h := hcl_in ⟨0, by decide⟩ (.sizeAtMost 1) (by simp [demoRuns])
  simp only [Runs.HoldsIn, Runs.innerBlock, Placed.Holds, demoRuns, Partition.whole, Partition.gates,
    Placed.work] at h
  revert h
  decide

/-- **The test**: not every circuit's run is in the feasible set. -/
theorem not_every_feasible : ¬ ∀ n : ℕ, (demoRuns n 0 0).Realizes () :=
  fun h => two_not_feasible (h 2)

/-- **A run outside the feasible set with a tolerance**: the two-gate circuit's run does not realize the claims of
`demoRuns 2 0 0` even with an empty tolerance, so `not_every_feasible` is of the path the chain reads (`RealizesExcept`),
not only `Realizes`. -/
theorem two_not_realizesExcept : ¬ (demoRuns 2 0 0).RealizesExcept () ∅ :=
  fun h => two_not_feasible (realizesExcept_empty_realizes _ _ h)

/-- The demos' limits: one inference block of size claim at most 1 gate and fresh input claim at most 0 bits, no free
block, no hidden hardware, and the given free work limit (gates) and server-day capacity limit (gates); the input limit
of an attested boundary's day block is `demoPast + kS` bits, `kS` bits above the demos' one wipe bound. -/
noncomputable def demoLimits (freeSize cS kS : ℕ) : InferenceOnly.Limits :=
  { n := 1, c := 1, k := 0, m := 0, freeSize := freeSize, cS := cS, kS := demoPast + kS, hidden := 0, share := 0,
    allowance := 0 }

/-- **The honest run passes the check and realizes at the empty tolerance**: `demoRuns 1 1 0` (capacity 1 gate, credited
use 1 gate, tolerance 0 gates) at honest limits (one inference block of size at most 1 gate and fresh input at most 0
bits, no free block, free work at most 0 gates, the server-day block's capacity at most 1 gate and its input bounds at
most `demoPast` bits)
passes `check` and is in the feasible set with no tolerance. -/
theorem honest_checks :
    InferenceOnly.check (demoLimits 0 1 0) (demoRuns 1 1 0) () = true ∧
      (demoRuns 1 1 0).RealizesExcept () ∅ :=
  ⟨by decide, realizes_realizesExcept _ _ honest_feasible⟩

/-- **The excluded run as a theorem** (universal, over any `Runs`): a run with a server-day block `i` whose inference
children in the tolerance set `M` are together larger than `lim.freeSize` cannot both pass `check` and have its free work
bound hold (`FreeWorkAtMost`), for every choice of the run's other fields: the check caps capacity − credited use +
tolerance at `lim.freeSize`, and `FreeWorkAtMost` bounds the free and tolerance work by that number. -/
theorem excluded_run (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run)
    (M : Finset (Fin (P.nIn r))) (i : Fin (P.nOut r)) (hki : P.kind r i = true)
    (hbig : lim.freeSize < ∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u) :
    ¬ (InferenceOnly.check lim P r = true ∧ P.FreeWorkAtMost r M) := by
  rintro ⟨hc, hfree⟩
  have h1 := hfree i hki
  have h2 := check_freeSize hc i hki
  omega

/-- **The excluded run, on the rows**: a run with a server-day block `i` whose inference children in the tolerance set
`M` are together larger than `lim.freeSize` fails the check or contradicts a row: PoUW's conclusion for that `M` on the
run's credited use at `i`, `i`'s size at most its capacity, or `M`'s committed size claims at most the run's
tolerance. -/
theorem excluded_run_rows (lim : InferenceOnly.Limits) (P : Runs L)    (r : P.Run) (M : Finset (Fin (P.nIn r))) (i : Fin (P.nOut r)) (hki : P.kind r i = true)
    (hbig : lim.freeSize < ∑ u ∈ (P.infChildren r i).filter (· ∈ M), P.sizeIn r u) :
    InferenceOnly.check lim P r = false ∨
      ¬ ((P.credited r i - ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤
          ∑ u ∈ (P.infChildren r i).filter (· ∉ M), P.sizeIn r u) ∧
        P.sizeOut r i ≤ P.capacity r i ∧ ∑ u ∈ M, claimedSize (P.claimsIn r u) ≤ P.tol) := by
  by_cases hc : InferenceOnly.check lim P r = true
  · refine Or.inr fun ⟨huse, hcap, htol⟩ => ?_
    have h := conversion_le_sub P r M i hki (P.credited r i) (P.capacity r i) P.tol huse hcap htol
    have h2 := check_freeSize hc i hki
    omega
  · exact Or.inl (by simpa using hc)

/-- **The demo instance of the excluded run**: the two-gate circuit at `freeSize = 0` with its one size-2 inference block
in `M` cannot pass the check with its free work bound holding, for any credited use `cr` and tolerance `tl`. -/
theorem excluded_run_demo (cr tl : ℕ) (M : Finset (Fin ((demoRuns 2 cr tl).nIn ())))
    (hM : (⟨0, Nat.one_pos⟩ : Fin ((demoRuns 2 cr tl).nIn ())) ∈ M) :
    ¬ (InferenceOnly.check (demoLimits 0 2 0) (demoRuns 2 cr tl) () = true ∧
        (demoRuns 2 cr tl).FreeWorkAtMost () M) := by
  refine excluded_run _ _ _ M ⟨0, Nat.one_pos⟩ rfl ?_
  have hmem : (⟨0, Nat.one_pos⟩ : Fin ((demoRuns 2 cr tl).nIn ())) ∈
      ((demoRuns 2 cr tl).infChildren () ⟨0, Nat.one_pos⟩).filter (· ∈ M) := by
    rw [Finset.mem_filter]
    refine ⟨?_, hM⟩
    simp only [Runs.infChildren, Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨rfl, rfl⟩
  show (0 : ℕ) < ∑ u ∈ ((demoRuns 2 cr tl).infChildren () ⟨0, Nat.one_pos⟩).filter (· ∈ M),
    (demoRuns 2 cr tl).sizeIn () u
  calc (0 : ℕ) < (demoRuns 2 cr tl).sizeIn () ⟨0, Nat.one_pos⟩ := by
        show (0 : ℕ) < (placed 2).work (fun _ => 1)
          ((Partition.whole (placed 2).toCircuit).gates ⟨0, Nat.one_pos⟩)
        decide
    _ ≤ _ := Finset.single_le_sum (fun j _ => Nat.zero_le _) hmem

/-- The two-gate demo's one inference block does 2 gates of work, whatever its credited use and tolerance. -/
theorem demo_two_work (cr tl : ℕ) :
    ∑ u ∈ (demoRuns 2 cr tl).infChildren () ⟨0, Nat.one_pos⟩, (demoRuns 2 cr tl).sizeIn () u = 2 :=
  (by decide : ∑ u ∈ (demoRuns 2 0 0).infChildren () ⟨0, Nat.one_pos⟩, (demoRuns 2 0 0).sizeIn () u = 2)

/-- The two-gate demo's free work bound at `M = univ` holds exactly when its 2 gates of work are at most capacity
(2 gates) − credited use + tolerance. -/
theorem demo_freeWork_iff (cr tl : ℕ) : (demoRuns 2 cr tl).FreeWorkAtMost () univ ↔ 2 ≤ 2 - cr + tl := by
  constructor
  · intro h
    have := h ⟨0, Nat.one_pos⟩ rfl
    rw [demo_freeChildren_empty, Finset.sum_empty, zero_add,
      Finset.filter_true_of_mem (fun u _ => Finset.mem_univ u), demo_two_work] at this
    exact this
  · intro h j _
    fin_cases j
    rw [demo_freeChildren_empty, Finset.sum_empty, zero_add,
      Finset.filter_true_of_mem (fun u _ => Finset.mem_univ u)]
    exact (demo_two_work cr tl).le.trans h

/-- **The free bound is what binds, at one gate**: the two-gate run at credited use 1 gate and tolerance 1 gate
(`demoRuns 2 1 1`, capacity 2 gates, so capacity − credited use + tolerance is 2 gates) is in the feasible set at
`M = all inference blocks` with its free work bound holding (`FreeWorkAtMost`). Every limit but `freeSize` the same, the
check refuses it at `freeSize = 1` gate and passes it at `freeSize = 2` gates; at `freeSize = 2` gates, one gate less of
credited use (`demoRuns 2 0 1`) or one more of tolerance (`demoRuns 2 1 2`) and the check refuses. -/
theorem freeBound_binds :
    (demoRuns 2 1 1).RealizesExcept () univ ∧ (demoRuns 2 1 1).FreeWorkAtMost () univ ∧
      InferenceOnly.check (demoLimits 1 2 0) (demoRuns 2 1 1) () = false ∧
      InferenceOnly.check (demoLimits 2 2 0) (demoRuns 2 1 1) () = true ∧
      InferenceOnly.check (demoLimits 2 2 0) (demoRuns 2 0 1) () = false ∧
      InferenceOnly.check (demoLimits 2 2 0) (demoRuns 2 1 2) () = false := by
  refine ⟨⟨?_, ?_, fun _ _ => le_rfl, ?_, ?_, ?_, ?_⟩, (demo_freeWork_iff 1 1).mpr (by decide), by decide, by decide,
    by decide, by decide⟩
  · intro a ha; exact absurd ha (Finset.notMem_empty a)
  · simp [demoRuns, Circuit.width]
  · intro u _ hu; exact absurd (Finset.mem_univ u) hu
  · intro u hu; exact absurd (rfl : (demoRuns 2 1 1).cls () u = Class.inference) hu
  · exact demo_blockBounds_two
  · intro u _; rfl

/-- `demoRuns 1 1 0` with its one attested boundary's ingress count at `ig` bits each day and its covert figure for the
period at `cv` bits (wipe egress 0 bits). -/
noncomputable def demoIn (ig cv : ℕ) : Runs Unit :=
  { demoRuns 1 1 0 with ingressCount := fun _ _ _ => ig, covert := fun _ => cv }

/-- **The capacity comparison bites alone, at one gate**: `demoRuns 2 2 0` (capacity 2 gates, credited use 2 gates)
fails the check at `cS` = 1 gate and passes it at `cS` = 2 gates, every other limit the same; no other comparison reads
`cS`. -/
theorem capacity_binds :
    InferenceOnly.check (demoLimits 0 1 0) (demoRuns 2 2 0) () = false ∧
      InferenceOnly.check (demoLimits 0 2 0) (demoRuns 2 2 0) () = true :=
  ⟨capacity_excluded (demoLimits 0 1 0) (demoRuns 2 2 0) () ⟨0, Nat.one_pos⟩ rfl (by decide), by decide⟩

/-- **The input comparison bites alone, at one bit**: `demoIn 1 0` (wipe egress 0 bits, ingress count 1 bit, covert
figure 0 bits) fails the check at `kS = demoPast` bits and passes it at one bit more, every other limit the same;
no other comparison reads `kS`. -/
theorem input_binds :
    InferenceOnly.check (demoLimits 0 1 0) (demoIn 1 0) () = false ∧
      InferenceOnly.check (demoLimits 0 1 1) (demoIn 1 0) () = true :=
  ⟨serverInput_excluded (demoLimits 0 1 0) (demoIn 1 0) () ⟨0, Nat.one_pos⟩ rfl (by decide), by decide⟩

/-- **The covert figure alone decides**: at the same limits (`kS = demoPast` bits) and the same counts (wipe egress
0 bits, ingress count 0 bits), the run passes the check when its attested boundary's covert figure for the period is
0 bits (`demoIn 0 0`) and fails it at 1 bit (`demoIn 0 1`). -/
theorem covert_decides :
    InferenceOnly.check (demoLimits 0 1 0) (demoIn 0 0) () = true ∧
      InferenceOnly.check (demoLimits 0 1 0) (demoIn 0 1) () = false :=
  ⟨by decide, serverInput_excluded (demoLimits 0 1 0) (demoIn 0 1) () ⟨0, Nat.one_pos⟩ rfl (by decide)⟩

/-- `demoRuns 1 1 0` with its location's wipe at `N` labels of 512 bits, soundness exponent 40 bits and `k` challenges
(wipe egress 0 bits, no uncovered storage). -/
noncomputable def demoWipe (N k : ℕ) : Runs Unit :=
  { demoRuns 1 1 0 with
    wipeLabels := fun _ => N
    labelBits := fun _ => 512
    wipeError := 40
    wipeChallenges := fun _ _ => k }

/-- **The wipe's registered numbers and its challenge count decide**: at 512-bit labels and soundness exponent 40 bits,
the wipe residual is 8,549,712,478 bits at 40,625,000,000 labels and 74,000 challenges and 16,666,681,461 bits at the
served H200 wipe's 2,319,974,400 labels and 2,152 challenges, so at `kS` = 10,000,000,000 bits, every other limit the
honest one, the first run passes the check and the second fails it. -/
theorem wipe_count_decides :
    Pous.StreamedWipe.wipeResidual 40625000000 74000 512 40 = 8549712478 ∧
      Pous.StreamedWipe.wipeResidual 2319974400 2152 512 40 = 16666681461 ∧
      InferenceOnly.check { demoLimits 0 1 0 with kS := 10000000000 } (demoWipe 40625000000 74000) () = true ∧
      InferenceOnly.check { demoLimits 0 1 0 with kS := 10000000000 } (demoWipe 2319974400 2152) () = false := by
  decide

/-- **A wipe that asked no challenge is refused by computation**: at 40,625,000,000 labels of 512 bits, soundness
exponent 40 bits and 0 challenges, the wipe's guarantee does not apply at the block's numbers and the check refuses the
run at `kS` = 10,000,000,000 bits, every other limit the honest one. -/
theorem wipe_zero_refused :
    ¬ (demoWipe 40625000000 0).WipeApplies (demoWipe 40625000000 0).wipeChallenges () ⟨0, Nat.one_pos⟩ ∧
      InferenceOnly.check { demoLimits 0 1 0 with kS := 10000000000 } (demoWipe 40625000000 0) () = false := by
  decide

/-- **A wipe whose residual is at or above what it fills is refused by computation**: at one label of 512 bits, one
challenge and soundness exponent 40 bits, the wipe residual is 14,406 bits, at least the 512 bits the wipe fills, so
the wipe's guarantee does not apply; the check refuses the run at `kS` = 10,000,000,000 bits, every other limit the
honest one, though its boundary's input bounds (14,406 bits) are far below `kS`. -/
theorem wipe_full_refused :
    Pous.StreamedWipe.wipeResidual 1 1 512 40 = 14406 ∧
      ¬ (demoWipe 1 1).WipeApplies (demoWipe 1 1).wipeChallenges () ⟨0, Nat.one_pos⟩ ∧
      (demoWipe 1 1).boundaryPastBound () 0 ⟨0, Nat.one_pos⟩ +
          ((demoWipe 1 1).ingressCount () 0 ⟨0, Nat.one_pos⟩ + (demoWipe 1 1).covert 0) ≤ 10000000000 ∧
      InferenceOnly.check { demoLimits 0 1 0 with kS := 10000000000 } (demoWipe 1 1) () = false := by
  decide

/-- **The credited comparison bites, at one gate**: `demoRuns 1 1 0` (capacity 1 gate, credited use 1 gate) passes the
check at the honest limits, and `demoRuns 1 2 0` (credited use 2 gates, one above its capacity) fails it, every limit the
same. The free comparison alone passes `demoRuns 1 2 0`: its capacity less its credited use plus its tolerance is
`1 − 2 + 0 = 0` gates over the naturals, at most `freeSize = 0` gates. -/
theorem credited_binds :
    InferenceOnly.check (demoLimits 0 1 0) (demoRuns 1 1 0) () = true ∧
      InferenceOnly.check (demoLimits 0 1 0) (demoRuns 1 2 0) () = false ∧
      (demoRuns 1 2 0).capacity () ⟨0, Nat.one_pos⟩ - (demoRuns 1 2 0).credited () ⟨0, Nat.one_pos⟩ +
        (demoRuns 1 2 0).tol ≤ (demoLimits 0 1 0).freeSize :=
  ⟨by decide, credited_excluded (demoLimits 0 1 0) (demoRuns 1 2 0) () ⟨0, Nat.one_pos⟩ rfl (by decide), by decide⟩

/-- **The conversion bites, at one gate**: the two-gate run `demoRuns 2 1 1` with its one inference block (work 2 gates,
committed size claim 1 gate) as the tolerance set `M = univ`. At credited use 1 gate and tolerance 1
gate PoUW's conclusion, the capacity (2 gates) and the tolerance hold, and the conversion's bound capacity − credited use
+ tolerance is 2 gates, exactly `M`'s work, so `FreeWorkAtMost` holds. One gate more of credited use (`demoRuns 2 2 1`)
or one less of tolerance (`demoRuns 2 1 0`) and `FreeWorkAtMost` fails, and so does the premise it changes. -/
theorem conversion_bites :
    let P := demoRuns 2 1 1
    let i : Fin (P.nOut ()) := ⟨0, Nat.one_pos⟩
    let Md := ∑ u ∈ (P.infChildren () i).filter (· ∈ univ), claimedSize (P.claimsIn () u)
    let outside := ∑ u ∈ (P.infChildren () i).filter (· ∉ univ),
      P.sizeIn () u
    (P.credited () i - Md ≤ outside ∧ P.sizeOut () i ≤ P.capacity () i ∧
      ∑ u ∈ univ, claimedSize (P.claimsIn () u) ≤ P.tol ∧
      (∑ u ∈ P.freeChildren () i, P.sizeIn () u) +
        (∑ u ∈ (P.infChildren () i).filter (· ∈ univ), P.sizeIn () u) = P.capacity () i - P.credited () i + P.tol) ∧
    P.FreeWorkAtMost () univ ∧ ¬ (demoRuns 2 2 1).FreeWorkAtMost () univ ∧
    ¬ (demoRuns 2 1 0).FreeWorkAtMost () univ ∧
    ¬ (2 - Md ≤ outside) ∧ ¬ (∑ u ∈ univ, claimedSize (P.claimsIn () u) ≤ 0) := by
  intro P i Md outside
  refine ⟨⟨by decide, by decide, by decide, ?_⟩, (demo_freeWork_iff 1 1).mpr (by decide),
    fun h => absurd ((demo_freeWork_iff 2 1).mp h) (by decide),
    fun h => absurd ((demo_freeWork_iff 1 0).mp h) (by decide), by decide, by decide⟩
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

/-- The span of ticks 1 and 2, cut into its two ticks. -/
def twoDays : (⟨1, 2⟩ : TickSpan).Cut 2 where
  boundary j := 1 + j.val
  mono _ _ h := Nat.add_le_add_left h _
  first := rfl
  last := rfl

/-- One run of the circuit `placed 2` at credited use `cr` gates in each server-day block and tolerance `tl` gates, cut at
one location into two server-day blocks, gate 1 during tick 1 and gate 2 during tick 2, each holding its gate as one
inference block claimed of size at most 1 gate and 0 bits of fresh input, each server-day block's wipe egress, and the
one attested boundary's ingress count and covert figure, 0 bits, each wipe bound `demoPast` bits. Its location's rate is 1 gate a tick, so each
server-day block's capacity is 1 gate. Its period, ticks 1 and 2, is registered as two days of one tick, one server-day
block each. -/
noncomputable def demoCut (cr tl : ℕ) : Runs Unit where
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
  period := ⟨1, 2⟩
  nDays := 2
  days := twoDays
  dayOf _ i := i
  winDay _ i _ := by
    refine ⟨?_, ?_⟩ <;> simp only [TickSpan.Cut.part, twoDays, Fin.coe_castSucc, Fin.val_succ] <;> omega
  oneBlock _ _ _ _ _ _ h := h
  rate _ := 1
  credited _ _ := cr
  tol := tl
  wipeLabels _ := 1
  labelBits _ := 4
  uncovered _ := 0
  wipeError := 0
  wipeEgress _ _ := 0
  wipeChallenges _ _ := 1
  boundary _ := 0
  ingressCount _ _ _ := 0
  covert _ := 0
  nIn _ := 2
  inner _ := cutTwo
  ref _ := { parent := id, sub := fun g => by simp }
  label _ _ := Class.inference
  locIn _ _ := {()}
  winIn _ u := ⟨u.val + 1, 1⟩
  claimsIn _ _ := [.sizeAtMost 1, .inputAtMost [.past, .block, .world] 0]
  labelHiddenIff _ _ := by simp
  salt _ _ := 0
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

/-- **The pooled bound counts the tolerance once**: the run `demoCut 1 2`, two server-day blocks of capacity 1 gate each,
credited use 1 gate in each, both inference blocks (committed size claim 1 gate, work 1 gate each) in the tolerance set
`M = univ` and tolerance 2 gates. PoUW's conclusion, the capacity and the tolerance hold in each block;
the free and `M`'s work summed over the run is 2 gates, exactly the pooled free bound (`2 − 2 + 2`), while the per-block
bounds summed are 4 gates (`(1 − 1 + 2)` twice). At one gate less of tolerance (`demoCut 1 1`) the per-block bound still
holds (`1 ≤ 1 − 1 + 1` in each) and the pooled one fails (`2 > 2 − 2 + 1`). -/
theorem pooled_counts_tol_once :
    let P := demoCut 1 2
    let S := univ.filter (fun i => P.kind () i = true)
    (∀ i, P.kind () i = true →
        P.credited () i - ∑ u ∈ (P.infChildren () i).filter (· ∈ univ), claimedSize (P.claimsIn () u) ≤
          ∑ u ∈ (P.infChildren () i).filter (· ∉ univ), P.sizeIn () u) ∧
      (∀ i, P.kind () i = true → P.sizeOut () i ≤ P.capacity () i) ∧
      ∑ u ∈ univ, claimedSize (P.claimsIn () u) ≤ P.tol ∧
      ∑ i ∈ S, ((∑ u ∈ P.freeChildren () i, P.sizeIn () u) +
          ∑ u ∈ (P.infChildren () i).filter (· ∈ univ), P.sizeIn () u) = P.pooledFreeBound () ∧
      P.pooledFreeBound () = 2 ∧ ∑ i ∈ S, (P.capacity () i - P.credited () i + P.tol) = 4 ∧
      P.FreeWorkAtMost () univ ∧ P.FreeWorkAtMostPooled () univ ∧
      (demoCut 1 1).FreeWorkAtMost () univ ∧ ¬ (demoCut 1 1).FreeWorkAtMostPooled () univ := by
  intro P S
  refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, ?_, ?_, ?_, ?_⟩
  · unfold Runs.FreeWorkAtMost; decide
  · unfold Runs.FreeWorkAtMostPooled; decide
  · unfold Runs.FreeWorkAtMost; decide
  · unfold Runs.FreeWorkAtMostPooled; decide

/-- The cut witness's limits: at most 2 inference blocks in the run, each of size claim at most 1 gate and fresh input
claim at most 0 bits; no free block; each server-day block's capacity − credited use + tolerance at most 1 gate, its
capacity at most 2 gates and each attested boundary's day block's input bounds at most `demoPast` bits; no hidden
hardware. -/
noncomputable def cutLimits : InferenceOnly.Limits :=
  { n := 2, c := 1, k := 0, m := 0, freeSize := 1, cS := 2, kS := demoPast, hidden := 0, share := 0,
    allowance := 0 }

/-- **Cutting a location's ticks multiplies the per-block limits**: the circuit `placed 2` (2 gates of work at one
location, during ticks 1 and 2) with both its inference blocks in the tolerance set passes `check` at `cutLimits`
(`freeSize` = 1 gate) when cut into two server-day blocks of one tick each at credited use 1 gate and tolerance 1 gate
(`demoCut 1 1`), realizing its claims outside the tolerance set with its free work bound holding; uncut, as one
server-day block (`demoRuns 2 cr tl`), no credited use `cr`, tolerance `tl` or tolerance set `M` passes `check` at the
same limits with `RealizesExcept` and `FreeWorkAtMost`. The witness is of the check and the per-run facts the policy
theorem reads: the cut run's tolerance (1 gate) is below `M`'s committed size claims (2 gates), so the declared row's
premise fails for this `M`. The cut is the registered days' (`demoCut`'s period is two days), not the prover's: a run has
at most the listed locations times the days of server-day blocks (`serverDay_count_le`), so the per-block limits
multiply by that count and no more. -/
theorem cut_multiplies :
    InferenceOnly.check cutLimits (demoCut 1 1) () = true ∧ (demoCut 1 1).RealizesExcept () univ ∧
      (demoCut 1 1).FreeWorkAtMost () univ ∧
      ∀ cr tl (M : Finset (Fin ((demoRuns 2 cr tl).nIn ()))),
        ¬ (InferenceOnly.check cutLimits (demoRuns 2 cr tl) () = true ∧ (demoRuns 2 cr tl).RealizesExcept () M ∧
          (demoRuns 2 cr tl).FreeWorkAtMost () M) := by
  refine ⟨by decide, ⟨?_, ?_, fun _ _ => le_rfl, ?_, ?_, ?_, ?_⟩, (pooled_counts_tol_once).2.2.2.2.2.2.2.2.1, ?_⟩
  · intro a ha; exact absurd ha (Finset.notMem_empty a)
  · simp [demoCut, Circuit.width]
  · intro u _ hu; exact absurd (Finset.mem_univ u) hu
  · intro u hu; exact absurd (by simp [demoCut, Runs.cls] : (demoCut 1 1).cls () u = Class.inference) hu
  · intro i hki
    have hone := boundaryBlocks_one _ _ i hki fun _ _ _ => rfl
    refine ⟨?_, holds_input_mono (k := 0) ?_ (Nat.zero_le _),
      (boundary_one_server _ _ i hone _).mpr (holds_input_mono (k := 0) ?_ (Nat.zero_le _))⟩
    all_goals clear hone hki
    all_goals fin_cases i
    all_goals first
      | (show (placed 2).work _ _ ≤ 1
         simp only [demoCut, Runs.outerBlock, Partition.gates, Placed.work]; decide)
      | (show (placed 2).blockInput _ _ _ _ ≤ 0
         simp only [demoCut, Runs.outerBlock, Placed.blockInput, Circuit.into, Circuit.width, placed, line]
         decide)
  · intro u _; simp [demoCut, Runs.cls]
  · intro cr tl M ⟨hc, hreal, hfree⟩
    by_cases hM : (⟨0, Nat.one_pos⟩ : Fin ((demoRuns 2 cr tl).nIn ())) ∈ M
    · refine excluded_run cutLimits (demoRuns 2 cr tl) () M ⟨0, Nat.one_pos⟩ rfl ?_ ⟨hc, hfree⟩
      have hmem : (⟨0, Nat.one_pos⟩ : Fin ((demoRuns 2 cr tl).nIn ())) ∈
          ((demoRuns 2 cr tl).infChildren () ⟨0, Nat.one_pos⟩).filter (· ∈ M) := by
        rw [Finset.mem_filter]
        refine ⟨?_, hM⟩
        simp only [Runs.infChildren, Finset.mem_filter, Finset.mem_univ, true_and]
        exact ⟨rfl, rfl⟩
      calc (1 : ℕ) < (demoRuns 2 cr tl).sizeIn () ⟨0, Nat.one_pos⟩ := by
            show (1 : ℕ) < (placed 2).work (fun _ => 1)
              ((Partition.whole (placed 2).toCircuit).gates ⟨0, Nat.one_pos⟩)
            decide
        _ ≤ _ := Finset.single_le_sum (fun j _ => Nat.zero_le _) hmem
    · obtain ⟨-, -, -, hcl_in, -⟩ := hreal
      have h : (demoRuns 2 cr tl).sizeIn () ⟨0, Nat.one_pos⟩ ≤ 1 :=
        hcl_in ⟨0, Nat.one_pos⟩ rfl hM (.sizeAtMost 1) (by simp [demoRuns])
      have h2 : (demoRuns 2 cr tl).sizeIn () ⟨0, Nat.one_pos⟩ = 2 := by
        show (placed 2).work (fun _ => 1) ((Partition.whole (placed 2).toCircuit).gates ⟨0, Nat.one_pos⟩) = 2
        decide
      omega

/-- **No top block is in no class**: `kind` is total, so the degenerate case of a top block in neither the server-day nor
the hidden class cannot be stated. Nothing sits inside an inference block, the finest scale here, so neither can a block
inside an inference block that is not a proof block. -/
theorem top_blocks_classified (P : Runs L) (r : P.Run) (i : Fin (P.nOut r)) :
    P.kind r i = true ∨ P.kind r i = false := by cases P.kind r i <;> simp

/-! ## A unit labelled inference before its epoch's challenge tick -/

/-- **A free block's work is under the free bound**: under the check and `FreeWorkAtMost`, every free block directly
inside a server-day block has size at most `lim.freeSize` gates. -/
theorem freeChild_le (lim : InferenceOnly.Limits) (P : Runs L) (r : P.Run) (M : Finset (Fin (P.nIn r)))
    (hc : InferenceOnly.check lim P r = true) (hfree : P.FreeWorkAtMost r M) (i : Fin (P.nOut r))
    (hki : P.kind r i = true) (u : Fin (P.nIn r)) (hu : u ∈ P.freeChildren r i) : P.sizeIn r u ≤ lim.freeSize := by
  have h1 : P.sizeIn r u ≤ ∑ v ∈ P.freeChildren r i, P.sizeIn r v :=
    Finset.single_le_sum (fun _ _ => Nat.zero_le _) hu
  have h2 := hfree i hki
  have h3 := check_freeSize hc i hki
  omega

/-- `demoRuns 1 1 0` with its epoch's challenge tick at tick `s`: its one unit is labelled inference, its window from
tick 0. -/
noncomputable def demoSalt (s : ℕ) : Runs Unit := { demoRuns 1 1 0 with salt := fun _ _ => s }

/-- **A unit labelled inference before its epoch's challenge tick is a free block, and the free bound reads it**: in
`demoSalt 1` the one unit is labelled inference and its window starts at tick 0, before its epoch's challenge tick at
tick 1, so its class is free. Its 1 gate of work is then above its server-day block's capacity (1 gate) less credited use
(1 gate) plus tolerance (0 gates), so `FreeWorkAtMost` fails for every tolerance set, and the check refuses it at
`demoLimits 0 1 0`. With the class read from the label, as the class field before this definition was, `demoSalt 1`
reads as `demoSalt 0` does: an inference block, passing the check with `FreeWorkAtMost` at the empty tolerance set. That
difference is why the class is derived. -/
theorem salt_label_free :
    (demoSalt 1).label () ⟨0, Nat.one_pos⟩ = Class.inference ∧
      ((demoSalt 1).winIn () ⟨0, Nat.one_pos⟩).start < (demoSalt 1).salt () ⟨0, Nat.one_pos⟩ ∧
      (demoSalt 1).cls () ⟨0, Nat.one_pos⟩ = Class.free ∧
      (∀ M, ¬ (demoSalt 1).FreeWorkAtMost () M) ∧
      InferenceOnly.check (demoLimits 0 1 0) (demoSalt 1) () = false ∧
      (demoSalt 0).cls () ⟨0, Nat.one_pos⟩ = Class.inference ∧
      (demoSalt 0).FreeWorkAtMost () ∅ ∧
      InferenceOnly.check (demoLimits 0 1 0) (demoSalt 0) () = true := by
  refine ⟨rfl, by decide, by decide, ?_, by decide, by decide, ?_, by decide⟩
  · intro M h
    have h1 := h ⟨0, Nat.one_pos⟩ rfl
    have h2 : ∑ u ∈ (demoSalt 1).freeChildren () ⟨0, Nat.one_pos⟩, (demoSalt 1).sizeIn () u = 1 := by decide
    have h3 : (demoSalt 1).capacity () ⟨0, Nat.one_pos⟩ - (demoSalt 1).credited () ⟨0, Nat.one_pos⟩ +
        (demoSalt 1).tol = 0 := by decide
    omega
  · unfold Runs.FreeWorkAtMost; decide

/-! ## The rows give the feasible set with a tolerance -/

theorem not_mem_of_isEmpty {α : Type*} {l : List α} (h : l.isEmpty = true) (p : α) : p ∉ l := by
  cases l with
  | nil => simp
  | cons _ _ => simp at h

/-- **The rows give the feasible set with a tolerance**: an accepted run that passes the check, with the declared row's
tolerance set `M` and its conclusion outside `M`, the wipe's row's bound on every server-day block's input from its own
past, network attestation's row's bound on its attested boundary's day block's input from the outside world (the
boundary's ingress count that day plus its covert figure), and `ComputeRate` over the listed locations at the hardware description's rates, which bounds every
server-day block's size by its capacity, realizes its claims outside `M` (`RealizesExcept r M`), under four named
premises: `RegisteredReadsOpenLog` (`θ`'s gates are input gates), `RegisteredWidthAtMost`, `PricingChargesEveryGate`
and `InferenceClaimsDeclared`. The check gives that a unit of `inner` that is not an inference block carries no claim. -/
theorem realizesExcept_of_rows (lim : InferenceOnly.Limits) (P : Runs L) {Root : Type} (log : Finset Root)
    (opensAt : (r : P.Run) → Fin (P.actual r).N → Root → Prop)
    (f : (r : P.Run) → Fin (P.nIn r) → Circuit) (cDecl kDecl : (r : P.Run) → Fin (P.nIn r) → ℕ)
    (hrate : ∀ r, (P.actual r).ComputeRate P.listed (P.π r) P.rate)
    (hreads : P.RegisteredReadsOpenLog log opensAt) (hwidth : P.RegisteredWidthAtMost)
    (hprice : P.PricingChargesEveryGate) (hinfC : P.InferenceClaimsDeclared f cDecl kDecl)
    (r : P.Run) (ha : P.accepts r) (hc : InferenceOnly.check lim P r = true)
    (M : Finset (Fin (P.nIn r))) (hMsub : M ⊆ univ.filter fun u => P.cls r u = Class.inference)
    (hdecl : ∀ u ∈ (univ.filter fun u => P.cls r u = Class.inference) \ M,
      P.HoldsIn r u (.function (f r u)) ∧ P.HoldsIn r u (.sizeAtMost (cDecl r u)) ∧
        P.HoldsIn r u (.inputAtMost [.past, .block, .world] (kDecl r u)))
    (hpast : ∀ i, P.kind r i = true →
      P.HoldsOut r i (.inputAtMost [.past] (P.pastBound P.wipeEgress P.wipeChallenges r i)))
    (hworld : ∀ i, P.kind r i = true → P.HoldsBoundary r (P.bnd r i) (P.dayOf r i)
      (.inputAtMost [.world] (P.ingressCount r (P.bnd r i) (P.dayOf r i) + P.covert (P.bnd r i)))) :
    P.RealizesExcept r M := by
  refine ⟨fun a hamem => (hreads r ha a hamem).1, hwidth r ha, hprice r, ?_, ?_, ?_,
    fun u hu => (Finset.mem_filter.mp (hMsub hu)).2⟩
  · intro u hu huM p hp
    obtain ⟨hf, hs, hi⟩ := hdecl u (Finset.mem_sdiff.mpr ⟨Finset.mem_filter.mpr ⟨Finset.mem_univ u, hu⟩, huM⟩)
    rcases hinfC r ha u hu p hp with rfl | ⟨s, rfl, hcs⟩ | ⟨s, rfl, hks⟩
    · exact hf
    · exact holds_size_mono hs hcs
    · exact holds_input_mono hi hks
  · intro u hu p hp
    exact absurd hp (not_mem_of_isEmpty (check_innerEmpty hc u hu) p)
  · intro i hki
    exact ⟨holds_size_mono (k := P.capacity r i) (capacityAtMost_compute P hrate r i hki) le_rfl, hpast i hki,
      hworld i hki⟩

/-! ## The server-day clause over committed claims

`OldServerClause` is a server-day clause over committed claims lists, one per top block: every claim holds.
`ClaimsAtLeastBounds` holds of claims lists whose every claim is at least a bound the rows establish.
`blockBounds_old` proves that `BlockBounds` gives `OldServerClause` at such lists. -/

/-- **The server-day clause over committed claims**, a claims list `cl i` per top block: every claim holds. -/
def OldServerClause (P : Runs L) (r : P.Run) (cl : Fin (P.nOut r) → List Property) : Prop :=
  ∀ i, ∀ p ∈ cl i, P.HoldsOut r i p

/-- **Claims at least the established bounds**: a hidden-hardware block carries none, and every claim of a server-day
block is a size claim of at least its capacity (gates), an input claim from exactly its own past of at least the wipe's
bound, or, for a block alone behind its attested boundary that day, one from exactly the outside world of at least the
boundary's ingress count that day plus its covert figure (bits). -/
def ClaimsAtLeastBounds (P : Runs L) (r : P.Run) (cl : Fin (P.nOut r) → List Property) : Prop :=
  ∀ i, (P.kind r i = false → cl i = []) ∧
    (P.kind r i = true → ∀ p ∈ cl i,
      (∃ s, p = .sizeAtMost s ∧ P.capacity r i ≤ s) ∨
        (∃ s, p = .inputAtMost [.past] s ∧ P.pastBound P.wipeEgress P.wipeChallenges r i ≤ s) ∨
        (P.boundaryBlocks r (P.bnd r i) (P.dayOf r i) = {i} ∧ ∃ s, p = .inputAtMost [.world] s ∧
          P.ingressCount r (P.bnd r i) (P.dayOf r i) + P.covert (P.bnd r i) ≤ s))

/-- **The new server-day clause gives the old one**: a run whose server-day blocks and attested boundaries' day blocks
hold the three bounds holds every claim of any claims lists whose claims are at least those bounds; a claim of input
from the outside world only for a block alone behind its attested boundary that day. -/
theorem blockBounds_old (P : Runs L) (r : P.Run) (cl : Fin (P.nOut r) → List Property)
    (h : P.BlockBounds r) (hcl : ClaimsAtLeastBounds P r cl) : OldServerClause P r cl := by
  intro i p hp
  cases hki : P.kind r i with
  | false => rw [(hcl i).1 hki] at hp; simp at hp
  | true =>
    obtain ⟨hs, hpa, hw⟩ := h i hki
    rcases (hcl i).2 hki p hp with ⟨s, rfl, hle⟩ | ⟨s, rfl, hle⟩ | ⟨hone, s, rfl, hle⟩
    · exact holds_size_mono hs hle
    · exact holds_input_mono hpa hle
    · exact (boundary_one_server P r i hone _).mp (holds_input_mono hw hle)

/-! ## The memory wipe's row, summed over the period's wipes -/

/-- **The wipe's row holds from its premises, at one wipe's error times the wipes**: if each wipe, a location of `Ls`
and a day, leaves its prior content under its bound except with probability at most `2^-s + adv`, and every accepted run
where all of them do has every server-day block's input from its own past under `pastBound E k` (bits), then that fails
on an accepted run with probability at most `|Ls| · nDays · (2^-s + adv)`: the union bound over the wipes (`Runs.union`).
-/
theorem wipeInputAtMost (P : Runs L) (G : Prop) (Ls : Finset (Finset L))
    (survives : Finset L → Fin P.nDays → P.Run → Prop)
    (E k memEnclosing memCap : (r : P.Run) → Fin (P.nOut r) → ℕ) (adv : ℝ≥0∞) :
    P.WipeInputAtMost G Ls survives E k memEnclosing memCap adv := by
  intro hG hread _ _ _ _ hinfo
  classical
  let e := Ls.equivFin
  have hsub : ∀ r, (P.accepts r ∧
      ¬ ∀ i, P.kind r i = true → P.HoldsOut r i (.inputAtMost [.past] (P.pastBound E k r i))) →
      ∃ x ∈ (univ : Finset (Fin Ls.card × Fin P.nDays)), P.accepts r ∧ ¬ survives (e.symm x.1).1 x.2 r := by
    rintro r ⟨ha, hn⟩
    by_contra hall
    push Not at hall
    refine hn (hinfo r ha fun X hX d => ?_)
    have h := hall (e ⟨X, hX⟩, d) (Finset.mem_univ _) ha
    simpa using h
  unfold ExceptWith
  calc P.pr _ ≤ P.pr (fun r => ∃ x ∈ (univ : Finset (Fin Ls.card × Fin P.nDays)),
          P.accepts r ∧ ¬ survives (e.symm x.1).1 x.2 r) := P.mono _ _ hsub
    _ ≤ ∑ x ∈ (univ : Finset (Fin Ls.card × Fin P.nDays)),
          P.pr (fun r => P.accepts r ∧ ¬ survives (e.symm x.1).1 x.2 r) :=
        P.union _ (fun (x : Fin Ls.card × Fin P.nDays) r => P.accepts r ∧ ¬ survives (e.symm x.1).1 x.2 r)
    _ ≤ ∑ _x ∈ (univ : Finset (Fin Ls.card × Fin P.nDays)), ((2⁻¹ : ℝ≥0∞) ^ P.wipeError + adv) :=
        Finset.sum_le_sum fun x _ => hread (e.symm x.1).1 (e.symm x.1).2 x.2 hG
    _ = ((Ls.card * P.nDays : ℕ) : ℝ≥0∞) * ((2⁻¹ : ℝ≥0∞) ^ P.wipeError + adv) := by
        rw [Finset.sum_const, nsmul_eq_mul, Finset.card_univ, Fintype.card_prod, Fintype.card_fin,
          Fintype.card_fin]

/-! ## The chain, with the protocols' rows named as types

`rows_chain` is the chain with each protocol's row fed to `chain` as a typed hypothesis, not named in a docstring. Its
hypotheses are the rows (`PouwUseAtLeast` on the run's credited use with its premises, and `ComputeRate` over the listed
locations at the hardware description's rates, from which `capacityAtMost_compute` gives each server-day block's size at
most its capacity; `DeclaredBlocksRun` at the run's tolerance; `PousUseAtLeast` and `PoubUseAtLeast` with their
conversions, their exclusivity premise open for memory and the link; `WipeInputAtMost`, `NetworkInputAtMost`,
`NetworkOutputAtMost` with their premises), `RegisteredReadsOpenLog`, and three named premises the rows do not give:
`RegisteredWidthAtMost`, `PricingChargesEveryGate` and `InferenceClaimsDeclared`. The wipe's row is proved from its
premises (`wipeInputAtMost`), read at the run's wipe egress and challenge count with its guarantee proved
(`PousStreamedWipe.streamed_wipe_residual`), its side condition from the check (`wipeApplies_of_check`), and its error
one wipe's times the wipes, `|Ls| · nDays`; network attestation's input row is read at the run's ingress count for
each attested boundary and day, and its output row bounds each attested boundary's day block, so the bounds
`BlockBounds` and the check read are the ones those rows establish. What those bounds rest on: `wipeEgress`, `wipeChallenges` and
`ingressCount` are tied to no attester's signed output yet; the covert figure is assumed; `lim.hidden` is assumed and no
protocol holds it; and no instance of `Runs` exists, so nothing about a real run is proved by this file. Its check is the inference-only check. Its
conclusion is the policy's: an accepted run's actual circuit is compliant except with the seven rows' soundness errors
summed. It is proved from `chain`.

For the set `M` the declared row gives, `FreeWorkAtMost r M` follows from PoUW's conclusion for that `M`, the capacity
and `M`'s committed size claims at most the run's tolerance (`freeWorkAtMost_of_conversion`), and `RealizesExcept r M`
from the check, the declared row, the wipe's and network attestation's input rows and the named premises
(`realizesExcept_of_rows`). PoUS's, PoUB's and network attestation's output rows feed no step of either: their errors
stay in the sum. The deterministic rows carry no error of their own: `AttesterOnEveryLink` and the capacities feed the
error rows or the free work bound; they are hypotheses but add no term to the sum. `NetworkInputAtMost`'s error is `0`
(its bound is the covert channels' assumed figure, with no sampled proof). -/
theorem rows_chain (P : Runs L)
    (lim : InferenceOnly.Limits) (pol : InferenceOnly.Policy)
    (hthm : InferenceOnly.PolicyTheorem (InferenceOnly.check (L := L) lim) pol)
    (hcheck : ∀ r, P.accepts r → InferenceOnly.check lim P r = true)
    -- PoUW: use at least, in size, for every tolerance set, and the compute capacity.
    {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : Pouw.PearlC.CostModel Q R S) (Pr : Pouw.PearlC.Protocol Q R S) (TR : Pouw.PearlC.TileRules Q R S)
    (D : Pouw.PearlC.Layout → Prop) (Wd : Pouw.PearlC.Layout → ℕ → Pouw.PearlC.Decl → ℚ) (g : ℝ) (δ : ℕ → ℕ → ℝ)
    (shownPouw : P.Run → Prop) (εpouw : ℝ≥0∞)
    (pouw : P.PouwUseAtLeast CM Pr TR D Wd g δ shownPouw εpouw)
    (hggap : Pouw.PearlC.GGapSampledDraw CM Pr TR D Wd g δ)
    (hpouwRead : P.RunReads (Pouw.PearlC.GGapSampledDraw CM Pr TR D Wd g δ) shownPouw εpouw)
    (hspend : P.SpendIsWork shownPouw)
    (hrate : ∀ r, (P.actual r).ComputeRate P.listed (P.π r) P.rate)
    -- The declared blocks run, at the run's tolerance.
    (f : (r : P.Run) → Fin (P.nIn r) → Circuit) (cDecl kDecl : (r : P.Run) → Fin (P.nIn r) → ℕ)
    (εdecl : ℝ≥0∞)
    (declared : P.DeclaredBlocksRun f cDecl kDecl εdecl)
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
    -- The memory wipe, read once for each wipe (with its memory-size capacity hypothesis).
    (Ls : Finset (Finset L)) (hLs : ∀ X ∈ P.listed, X ∈ Ls)
    (survives : Finset L → Fin P.nDays → P.Run → Prop) (memEnclosing memCap : (r : P.Run) → Fin (P.nOut r) → ℕ)
    (adv : ℝ≥0∞)
    (hwipeRead : ∀ X ∈ Ls, ∀ d,
      P.RunReads Pous.Guarantees.StreamedWipeResidual (survives X d) (2⁻¹ ^ P.wipeError + adv))
    (hwipeCap : P.CapacityAtMost Capacity.memorySize memEnclosing memCap)
    (hwipeCover : ∀ r i, P.kind r i = true →
      memCap r i ≤ P.wipeLabels (P.locOut r i) * P.labelBits (P.locOut r i) + P.uncovered (P.locOut r i))
    (hwipeInfo : P.WipeInformationGivesWidth (fun r => ∀ X ∈ Ls, ∀ d, survives X d r)
      (P.pastBound P.wipeEgress P.wipeChallenges))
    -- Network attestation, input, at the run's ingress count for each attested boundary and day.
    (egressN : (r : P.Run) → ℕ → Fin P.nDays → ℕ)
    (netIn : P.NetworkInputAtMost P.ingressCount P.ingressCount egressN)
    (hatt : P.AttesterOnEveryLink P.ingressCount egressN)
    -- Network attestation, output, for each attested boundary and day.
    (shownNet : P.Run → Prop) (bOut egressN' ingressN' : (r : P.Run) → ℕ → Fin P.nDays → ℕ) (εnet : ℝ≥0∞)
    (netOut : P.NetworkOutputAtMost shownNet egressN bOut egressN' ingressN' εnet)
    (hphase : NetTiming.Attester.PhaseGetsBack)
    (hnetRead : P.RunReads NetTiming.Attester.PhaseGetsBack shownNet εnet)
    (hrecord : P.RecordIsOutput shownNet egressN bOut) (hatt' : P.AttesterOnEveryLink ingressN' egressN')
    -- Registration: the registered input gates.
    {Root : Type} (log : Finset Root) (opensAt : (r : P.Run) → Fin (P.actual r).N → Root → Prop)
    (hregreads : P.RegisteredReadsOpenLog log opensAt)
    -- The named premises the rows do not give.
    (hwidth : P.RegisteredWidthAtMost) (hprice : P.PricingChargesEveryGate)
    (hinfC : P.InferenceClaimsDeclared f cDecl kDecl) :
    ExceptWith P.pr P.accepts (fun r => InferenceOnly.Compliant pol (P.actual r) (P.θ r))
      (∑ x : Fin 7, (![εpouw, εdecl, εpous, εpoub,
        ((Ls.card * P.nDays : ℕ) : ℝ≥0∞) * (2⁻¹ ^ P.wipeError + adv), 0, εnet] : Fin 7 → ℝ≥0∞) x) := by
  set shownV : Fin 7 → P.Run → Prop :=
    ![ (fun r => ∀ (M : Finset (Fin (P.nIn r))) i, P.kind r i = true →
         P.credited r i - ∑ u ∈ (P.infChildren r i).filter (· ∈ M), claimedSize (P.claimsIn r u) ≤
           ∑ u ∈ (P.infChildren r i).filter (· ∉ M), P.sizeIn r u),
       (fun r => ∃ M ⊆ (univ.filter fun u => P.cls r u = Class.inference),
         (∑ u ∈ M, claimedSize (P.claimsIn r u)) ≤ P.tol ∧
           ∀ u ∈ (univ.filter fun u => P.cls r u = Class.inference) \ M,
             P.HoldsIn r u (.function (f r u)) ∧ P.HoldsIn r u (.sizeAtMost (cDecl r u)) ∧
               P.HoldsIn r u (.inputAtMost [.past, .block, .world] (kDecl r u))),
       (fun r => ∀ (_M : Finset (Fin (P.nIn r))) i, P.kind r i = true → freePous r i ≤ capPous r i - bPous r i),
       (fun r => ∀ (_M : Finset (Fin (P.nIn r))) i, P.kind r i = true → freePoub r i ≤ capPoub r i - bPoub r i),
       (fun r => ∀ i, P.kind r i = true →
         P.HoldsOut r i (.inputAtMost [.past] (P.pastBound P.wipeEgress P.wipeChallenges r i))),
       (fun r => ∀ i, P.kind r i = true → P.HoldsBoundary r (P.bnd r i) (P.dayOf r i)
         (.inputAtMost [.world] (P.ingressCount r (P.bnd r i) (P.dayOf r i) + P.covert (P.bnd r i)))),
       (fun r => ∀ i, P.kind r i = true → P.HoldsBoundary r (P.bnd r i) (P.dayOf r i)
         (.outputAtMost (bOut r (P.bnd r i) (P.dayOf r i)))) ] with hshownV
  set εV : Fin 7 → ℝ≥0∞ := ![εpouw, εdecl, εpous, εpoub,
    ((Ls.card * P.nDays : ℕ) : ℝ≥0∞) * (2⁻¹ ^ P.wipeError + adv), 0, εnet] with hεV
  have hcover : ∀ r, P.accepts r → (∀ x ∈ Finset.univ, shownV x r) →
      ∃ M, P.FreeWorkAtMost r M ∧ P.RealizesExcept r M := by
    intro r ha hall
    have h0 := hall 0 (Finset.mem_univ _)
    have h1 := hall 1 (Finset.mem_univ _)
    simp only [hshownV, Matrix.cons_val_zero, Matrix.cons_val_one] at h0 h1
    obtain ⟨M, hMsub, htol, hdecl⟩ := h1
    refine ⟨M, freeWorkAtMost_of_conversion P r M (h0 M)
      (fun i hki => capacityAtMost_compute P hrate r i hki) htol, ?_⟩
    exact realizesExcept_of_rows lim P log opensAt f cDecl kDecl hrate hregreads hwidth hprice hinfC r ha
      (hcheck r ha) M hMsub hdecl (hall 4 (Finset.mem_univ _)) (hall 5 (Finset.mem_univ _))
  have hest : ∀ x ∈ Finset.univ, ExceptWith P.pr P.accepts (shownV x) (εV x) := by
    intro x _
    fin_cases x
    · exact pouw hggap hpouwRead hspend
    · exact declared
    · exact (freeBlockAtMost P) Capacity.memorySize (fun r _ i => freePous r i) (fun r _ i => heldS r i)
        (fun r _ i => bPous r i) enclosingPous capPous εpous pous hcapPous hexclPous
    · exact (freeBlockAtMost P) Capacity.linkCapacity (fun r _ i => freePoub r i) (fun r _ i => readsB r i)
        (fun r _ i => bPoub r i) enclosingPoub capPoub εpoub poub hcapPoub hexclPoub
    · exact wipeInputAtMost P Pous.Guarantees.StreamedWipeResidual Ls survives P.wipeEgress P.wipeChallenges memEnclosing memCap adv
        PousStreamedWipe.streamed_wipe_residual hwipeRead hwipeCap hwipeCover (wipeApplies_of_check hcheck) hLs
        hwipeInfo
    · exact netIn hatt fun _ _ _ _ => le_rfl
    · exact netOut hphase hnetRead hrecord hatt'
  exact chain P (InferenceOnly.check lim) pol Finset.univ shownV εV hthm (fun r ha => hcheck r ha) hcover hest

end ResourceAccounting.SecurityProofs.BlockProperties
