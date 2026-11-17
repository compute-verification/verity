module

public import Properties.ResourceAccounting.Requirement
public import Mathlib.Algebra.GroupWithZero.Nat
import Mathlib.Algebra.Order.BigOperators.Group.LocallyFinite
import Std.Data.HashSet.Lemmas

/-! The requirement code's lists, sums and placements equal the definition's sets and sums. -/

@[expose] public section

namespace Properties.Requirement

open CircuitRequirement

theorem contains_iff {l : List ℕ} {a : ℕ} : (Std.HashSet.ofList l).contains a = true ↔ a ∈ l := by
  simp [Std.HashSet.contains_ofList]

theorem not_contains_iff {l : List ℕ} {a : ℕ} : (!(Std.HashSet.ofList l).contains a) = true ↔ a ∉ l := by
  simp [Std.HashSet.contains_ofList]

theorem mem_distinct {l : List ℕ} {a : ℕ} : a ∈ distinct l ↔ a ∈ l := by
  simp [distinct, List.mem_mergeSort, Std.HashSet.mem_toList, Std.HashSet.mem_ofList]

theorem subsetOf_iff {a b : List ℕ} : subsetOf a b = true ↔ a.toFinset ⊆ b.toFinset := by
  simp only [subsetOf, List.all_eq_true, contains_iff, Finset.subset_iff, List.mem_toFinset]

theorem sum_range' (f : ℕ → ℕ) (start len : ℕ) :
    ((List.range' start len).map f).sum = ∑ t ∈ Finset.Ico start (start + len), f t := by
  induction len generalizing start with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_eq_sum_Ico_succ_bot (by omega), List.range'_succ, List.map_cons, List.sum_cons, ih,
      show start + 1 + n = start + (n + 1) by omega]

theorem resident_eq (s : CircuitRequirement.Store) (t : ℕ) : (s.resident t).toFinset = (store s).resident t := by
  ext j
  simp [Store.resident, mem_distinct, ResourceAccounting.Requirement.Store.resident, store]
  tauto

section Instance

variable (i : Instance)

theorem mem_ran {v : ℕ} : v ∈ i.ran ↔ v < i.width.size ∧ (v ∈ i.registered ∨ (i.step.getD v none).isSome) := by
  simp [Instance.ran]

theorem ran_nodup : i.ran.Nodup := List.nodup_range.filter _

theorem mem_live {t v : ℕ} : v ∈ i.live t ↔
    v ∈ i.ran ∧ (v ∈ i.registered ∨ i.stepOf v < t) ∧ ∃ g ∈ i.ran, t ≤ i.stepOf g ∧ v ∈ i.opsOf g := by
  simp only [Instance.live, List.mem_filter, Bool.and_eq_true, Bool.or_eq_true, contains_iff, decide_eq_true_eq,
    List.mem_flatMap, and_assoc]

theorem mem_reads {t v : ℕ} : v ∈ i.reads t ↔
    v ∈ i.ran ∧ (v ∈ i.registered ∨ i.stepOf v < t) ∧ ∃ g ∈ i.ran, i.stepOf g = t ∧ v ∈ i.usesOf g := by
  simp only [Instance.reads, List.mem_filter, Bool.and_eq_true, Bool.or_eq_true, contains_iff, decide_eq_true_eq,
    List.mem_flatMap, and_assoc]

theorem live_nodup (t : ℕ) : (i.live t).Nodup := by
  simp only [Instance.live]
  exact (ran_nodup i).filter _

theorem reads_nodup (t : ℕ) : (i.reads t).Nodup := by
  simp only [Instance.reads]
  exact (ran_nodup i).filter _

theorem topological_iff : i.topological = true ↔ (wiring i).Topological := by
  simp only [Instance.topological, List.all_eq_true, List.mem_range, decide_eq_true_eq,
    ResourceAccounting.Requirement.Wiring.Topological, wiring, List.mem_toFinset]
  refine ⟨fun h g v hv => ?_, fun h g _ v hv => h g v hv⟩
  by_cases hg : g < i.ops.size
  · exact h g hg v hv
  · simp [Instance.opsOf, Array.getD, hg] at hv

theorem follows_iff : i.follows = true ↔ (execution i).Follows (wiring i) := by
  simp only [Instance.follows, Bool.and_eq_true, List.all_eq_true, contains_iff, Bool.or_eq_true, decide_eq_true_eq,
    ResourceAccounting.Requirement.Execution.Follows, execution, wiring, Finset.subset_iff, List.mem_toFinset,
    List.contains_iff_mem]

theorem live_eq (t : ℕ) : (i.live t).toFinset = ResourceAccounting.Requirement.live (wiring i) (execution i) t := by
  ext v
  simp only [List.mem_toFinset, mem_live, ResourceAccounting.Requirement.live, Finset.mem_filter, wiring, execution]

theorem reads_eq (t : ℕ) : (i.reads t).toFinset = ResourceAccounting.Requirement.reads (wiring i) (execution i) t := by
  ext v
  simp only [List.mem_toFinset, mem_reads, ResourceAccounting.Requirement.reads, Finset.mem_filter, wiring, execution]

theorem bits_eq {S : List ℕ} (h : S.Nodup) : i.bits S = ResourceAccounting.Requirement.bits (wiring i) S.toFinset := by
  rw [Instance.bits, ResourceAccounting.Requirement.bits, List.sum_toFinset _ h]
  rfl

theorem liveBits_eq (t : ℕ) : i.liveBits t = ResourceAccounting.Requirement.liveBits (wiring i) (execution i) t := by
  rw [Instance.liveBits, bits_eq i (live_nodup i t), live_eq, ResourceAccounting.Requirement.liveBits]

theorem readBits_eq (t : ℕ) : i.readBits t = ResourceAccounting.Requirement.readBits (wiring i) (execution i) t := by
  rw [Instance.readBits, bits_eq i (reads_nodup i t), reads_eq, ResourceAccounting.Requirement.readBits]

theorem held_eq (start len : ℕ) :
    i.held start len = ResourceAccounting.Requirement.held (wiring i) (execution i) ⟨start, len⟩ := by
  rw [Instance.held, sum_range', ResourceAccounting.Requirement.held, ResourceAccounting.Window.steps]
  exact Finset.sum_congr rfl fun t _ => liveBits_eq i t

theorem read_eq (start len : ℕ) :
    i.read start len = ResourceAccounting.Requirement.read (wiring i) (execution i) ⟨start, len⟩ := by
  rw [Instance.read, sum_range', ResourceAccounting.Requirement.read, ResourceAccounting.Window.steps]
  exact Finset.sum_congr rfl fun t _ => readBits_eq i t

theorem blocks_eq (S : List ℕ) : (i.blocks S).toFinset = ResourceAccounting.Requirement.blocks (layout i) S.toFinset := by
  ext j
  simp [Instance.blocks, mem_distinct, ResourceAccounting.Requirement.blocks, layout]

theorem bitsIn_eq {S : List ℕ} (h : S.Nodup) (j : ℕ) :
    i.bitsIn S j = ResourceAccounting.Requirement.bitsIn (wiring i) (layout i) S.toFinset j := by
  rw [Instance.bitsIn, bits_eq i (h.filter _), ResourceAccounting.Requirement.bitsIn, List.toFinset_filter]
  simp only [decide_eq_true_eq]
  rfl

theorem liveBlocks_eq (t : ℕ) :
    (i.liveBlocks t).toFinset = ResourceAccounting.Requirement.liveBlocks (wiring i) (execution i) (layout i) t := by
  rw [Instance.liveBlocks, blocks_eq, live_eq, ResourceAccounting.Requirement.liveBlocks]

theorem readBlocks_eq (t : ℕ) :
    (i.readBlocks t).toFinset = ResourceAccounting.Requirement.readBlocks (wiring i) (execution i) (layout i) t := by
  rw [Instance.readBlocks, blocks_eq, reads_eq, ResourceAccounting.Requirement.readBlocks]

theorem liveIn_eq (t j : ℕ) : i.liveIn t j = ResourceAccounting.Requirement.liveIn (wiring i) (execution i) (layout i) t j := by
  rw [Instance.liveIn, bitsIn_eq i (live_nodup i t), live_eq, ResourceAccounting.Requirement.liveIn]

theorem readIn_eq (t j : ℕ) : i.readIn t j = ResourceAccounting.Requirement.readIn (wiring i) (execution i) (layout i) t j := by
  rw [Instance.readIn, bitsIn_eq i (reads_nodup i t), reads_eq, ResourceAccounting.Requirement.readIn]

theorem before_trans (a b c : ℕ) : i.before a b = true → i.before b c = true → i.before a c = true := by
  intro h1 h2
  simp only [Instance.before, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at h1 h2 ⊢
  omega

theorem before_total (a b : ℕ) : (i.before a b || i.before b a) = true := by
  simp only [Instance.before, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq]
  omega

def Chained (u v : ℕ) : Prop := i.before u v = true ∧ (i.blockOf u = i.blockOf v → i.endOf u ≤ i.offsetOf v)

theorem chained_trans {a b c : ℕ} : Chained i a b → Chained i b c → Chained i a c := by
  rintro ⟨hab, h1⟩ ⟨hbc, h2⟩
  refine ⟨before_trans i a b c hab hbc, fun h => ?_⟩
  simp only [Instance.before, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at hab hbc h1 h2 ⊢
  simp only [Instance.endOf] at *
  have hb : i.blockOf a = i.blockOf b ∧ i.blockOf b = i.blockOf c := by omega
  have := h1 hb.1
  have := h2 hb.2
  omega

theorem apart_pairwise : ∀ {L : List ℕ}, L.Pairwise (fun a b => i.before a b = true) → i.apart L = true →
    L.Pairwise (Chained i)
  | [], _, _ => List.Pairwise.nil
  | [_], _, _ => List.pairwise_singleton _ _
  | a :: b :: rest, hs, ha => by
    simp only [Instance.apart, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq] at ha
    have ih := apart_pairwise hs.of_cons ha.2
    have hab : Chained i a b := ⟨List.rel_of_pairwise_cons hs (by simp), fun h => ha.1.resolve_left (not_not.mpr h)⟩
    refine List.Pairwise.cons (fun x hx => ?_) ih
    rcases List.mem_cons.mp hx with rfl | hx
    · exact hab
    · exact chained_trans i hab (List.rel_of_pairwise_cons ih hx)

theorem apart_of : ∀ {L : List ℕ}, L.Nodup → L.Pairwise (fun a b => i.before a b = true) →
    (∀ u ∈ L, ∀ v ∈ L, u ≠ v → i.blockOf u = i.blockOf v → i.endOf u ≤ i.offsetOf v ∨ i.endOf v ≤ i.offsetOf u) →
    i.apart L = true
  | [], _, _, _ => by simp [Instance.apart]
  | [_], _, _, _ => by simp [Instance.apart]
  | a :: b :: rest, hn, hs, hp => by
    simp only [Instance.apart, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq]
    refine ⟨?_, apart_of hn.of_cons hs.of_cons fun u hu v hv =>
      hp u (List.mem_cons_of_mem _ hu) v (List.mem_cons_of_mem _ hv)⟩
    have hne : a ≠ b := fun h => (List.nodup_cons.mp hn).1 (h ▸ by simp)
    have hab := List.rel_of_pairwise_cons hs (by simp : b ∈ b :: rest)
    by_cases hbl : i.blockOf a = i.blockOf b
    · right
      have := hp a (by simp) b (by simp) hne hbl
      simp only [Instance.before, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at hab
      simp only [Instance.endOf] at hab this ⊢
      omega
    · exact Or.inl hbl

theorem packs_iff {S : List ℕ} (h : S.Nodup) (b : ℕ) :
    i.packs b S = true ↔ ResourceAccounting.Requirement.Packs (wiring i) (layout i) b S.toFinset := by
  have hperm := List.mergeSort_perm S i.before
  have hsort := List.pairwise_mergeSort (before_trans i) (before_total i) S
  have hmem : ∀ x, x ∈ S.mergeSort i.before ↔ x ∈ S := fun x => hperm.mem_iff
  simp only [Instance.packs, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq, ResourceAccounting.Requirement.Packs,
    List.mem_toFinset, wiring, layout]
  constructor
  · rintro ⟨hB, ha⟩ u hu
    refine ⟨hB u hu, fun v hv huv hbl => ?_⟩
    have hq := (apart_pairwise i hsort ha).imp (R := Chained i) (S := fun a b => Chained i a b ∨ Chained i b a) Or.inl
    have : Std.Symm fun a b => Chained i a b ∨ Chained i b a := ⟨fun _ _ h => h.symm⟩
    rcases hq.forall ((hmem u).2 hu) ((hmem v).2 hv) huv with h | h
    · exact Or.inl (h.2 hbl)
    · exact Or.inr (h.2 hbl.symm)
  · intro hp
    exact ⟨fun u hu => (hp u hu).1, apart_of i (hperm.nodup_iff.mpr h) hsort fun u hu v hv huv hbl =>
      (hp u ((hmem u).1 hu)).2 v ((hmem v).1 hv) huv hbl⟩

theorem carries_iff (s : CircuitRequirement.Store) (S : List ℕ) :
    i.carries s S = true ↔
      ResourceAccounting.Requirement.Carries (wiring i) (execution i) (layout i) (store s) S.toFinset := by
  simp [Instance.carries, ResourceAccounting.Requirement.Carries, wiring, execution, layout, store]

theorem incomplete_eq (t : ℕ) :
    (i.incomplete t).toFinset = ResourceAccounting.Requirement.incomplete (wiring i) (execution i) (layout i) t := by
  ext j
  simp only [Instance.incomplete, List.mem_toFinset, mem_distinct, List.mem_map, List.mem_filter, Bool.and_eq_true,
    Bool.or_eq_true, not_contains_iff, decide_eq_true_eq, ResourceAccounting.Requirement.incomplete, Finset.mem_image,
    Finset.mem_filter, wiring, execution, layout]

theorem mem_complete {S : List ℕ} {t v : ℕ} : v ∈ i.complete S t ↔ v ∈ S ∧ i.blockOf v ∉ i.incomplete t := by
  simp only [Instance.complete, List.mem_filter, not_contains_iff]

theorem credited_nodup (t : ℕ) : (i.credited t).Nodup := (live_nodup i t).filter _

theorem creditedReads_nodup (t : ℕ) : (i.creditedReads t).Nodup := (reads_nodup i t).filter _

theorem credited_eq (t : ℕ) :
    (i.credited t).toFinset = ResourceAccounting.Requirement.credited (wiring i) (execution i) (layout i) t := by
  ext v
  rw [ResourceAccounting.Requirement.credited, ← live_eq, ← incomplete_eq, Instance.credited, List.mem_toFinset, mem_complete,
    Finset.mem_filter, List.mem_toFinset, List.mem_toFinset]
  rfl

theorem creditedReads_eq (t : ℕ) :
    (i.creditedReads t).toFinset = ResourceAccounting.Requirement.creditedReads (wiring i) (execution i) (layout i) t := by
  ext v
  rw [ResourceAccounting.Requirement.creditedReads, ← reads_eq, ← incomplete_eq, Instance.creditedReads, List.mem_toFinset,
    mem_complete, Finset.mem_filter, List.mem_toFinset, List.mem_toFinset]
  rfl

theorem creditedBits_eq (t : ℕ) :
    i.creditedBits t = ResourceAccounting.Requirement.creditedBits (wiring i) (execution i) (layout i) t := by
  rw [Instance.creditedBits, bits_eq i (credited_nodup i t), credited_eq, ResourceAccounting.Requirement.creditedBits]

theorem creditedReadBits_eq (t : ℕ) :
    i.creditedReadBits t = ResourceAccounting.Requirement.creditedReadBits (wiring i) (execution i) (layout i) t := by
  rw [Instance.creditedReadBits, bits_eq i (creditedReads_nodup i t), creditedReads_eq,
    ResourceAccounting.Requirement.creditedReadBits]

theorem storeMatches_iff (s : CircuitRequirement.Store) (b t : ℕ) :
    i.storeMatches s b t = true ↔
      ResourceAccounting.Requirement.StoreMatches (wiring i) (execution i) (layout i) (store s) b t := by
  simp only [Instance.storeMatches, Bool.and_eq_true, subsetOf_iff, resident_eq, blocks_eq, packs_iff i (live_nodup i t),
    carries_iff, live_eq, credited_eq, ResourceAccounting.Requirement.StoreMatches, and_assoc]

theorem planMatches_iff (plan : List ℕ) (b t : ℕ) :
    i.planMatches plan b t = true ↔
      ResourceAccounting.Requirement.PlanMatches (wiring i) (execution i) (layout i) plan.toFinset b t := by
  simp only [Instance.planMatches, Bool.and_eq_true, subsetOf_iff, blocks_eq, packs_iff i (reads_nodup i t), reads_eq,
    creditedReads_eq, ResourceAccounting.Requirement.PlanMatches]

theorem distinct_nodup (l : List ℕ) : (distinct l).Nodup :=
  (List.mergeSort_perm _ _).nodup_iff.mpr ((Std.HashSet.ofList l).distinct_toList.imp fun h => by simpa using h)

variable (p : CircuitRequirement.Placement) (X : List ℕ)

theorem mem_computed {g : ℕ} : g ∈ i.computed ↔ g ∈ i.ran ∧ g ∉ i.registered ∧ g ∉ i.inputs := by
  simp only [Instance.computed, List.mem_filter, Bool.and_eq_true, not_contains_iff]

theorem mem_region {lo hi g : ℕ} : g ∈ i.region p X lo hi ↔
    g ∈ i.ran ∧ g ∉ i.registered ∧ g ∉ i.inputs ∧ p.leafOf g ∈ X ∧ lo ≤ i.tick p g ∧ i.tick p g < hi := by
  simp only [Instance.region, List.mem_filter, mem_computed, Bool.and_eq_true, contains_iff, decide_eq_true_eq,
    and_assoc]

theorem region_nodup (lo hi : ℕ) : (i.region p X lo hi).Nodup := by
  simp only [Instance.region, Instance.computed]
  exact ((ran_nodup i).filter _).filter _

theorem region_eq (lo hi : ℕ) : (i.region p X lo hi).toFinset =
    ResourceAccounting.Requirement.region (wiring i) (placement i p) X.toFinset (Set.Ico lo hi) := by
  ext g
  simp only [List.mem_toFinset, mem_region, ResourceAccounting.Requirement.region, Finset.mem_filter, wiring, placement,
    execution, ResourceAccounting.Requirement.Placement.tick, Instance.tick, Set.mem_Ico]

theorem stepRegion_eq (start len : ℕ) : (i.stepRegion p X start len).toFinset =
    ResourceAccounting.Requirement.stepRegion (wiring i) (placement i p) X.toFinset ⟨start, len⟩ := by
  ext g
  simp only [Instance.stepRegion, List.mem_toFinset, List.mem_filter, mem_computed, Bool.and_eq_true, contains_iff,
    decide_eq_true_eq, ResourceAccounting.Requirement.stepRegion, Finset.mem_filter, wiring, placement, execution,
    ResourceAccounting.Window.steps, Finset.mem_Ico, and_assoc]

theorem mem_into {S : List ℕ} {a : ℕ} : a ∈ i.into S ↔ (∃ g ∈ S, a ∈ i.usesOf g) ∧ a ∉ S := by
  simp only [Instance.into, mem_distinct, List.mem_filter, List.mem_flatMap, not_contains_iff]

theorem ingress_eq (lo hi : ℕ) :
    i.ingress p X lo hi = ResourceAccounting.Requirement.ingress (wiring i) (placement i p) X.toFinset (Set.Ico lo hi) := by
  have h : ((i.into (i.region p X lo hi)).filter fun a => !(Std.HashSet.ofList X).contains (p.leafOf a)).Nodup :=
    (distinct_nodup _).filter _
  show i.bits _ = _
  rw [bits_eq i h, ResourceAccounting.Requirement.ingress, ← region_eq]
  congr 1
  ext a
  simp only [List.mem_toFinset, List.mem_filter, not_contains_iff, mem_into, Finset.mem_filter,
    ResourceAccounting.Requirement.into, Finset.mem_sdiff, Finset.mem_biUnion, wiring, placement]

theorem credits_eq (price : Array ℕ) (marked : List ℕ) (lo hi : ℕ) :
    i.credits p price X marked lo hi = ResourceAccounting.Requirement.work (fun g => price.getD g 0)
      (ResourceAccounting.Requirement.region (wiring i) (placement i p) X.toFinset (Set.Ico lo hi) \ marked.toFinset) := by
  have h : ((i.region p X lo hi).filter fun g => !(Std.HashSet.ofList marked).contains g).Nodup :=
    (region_nodup i p X lo hi).filter _
  show Instance.work price _ = _
  rw [Instance.work, ← List.sum_toFinset _ h, ResourceAccounting.Requirement.work, ← region_eq]
  congr 1
  ext g
  simp only [List.mem_toFinset, List.mem_filter, not_contains_iff, Finset.mem_sdiff]

end Instance

theorem topological : Topological := topological_iff
theorem follows : Follows := follows_iff
theorem live : Live := live_eq
theorem reads : Reads := reads_eq
theorem liveBits : LiveBits := liveBits_eq
theorem readBits : ReadBits := readBits_eq
theorem held : Held := held_eq
theorem read : Read := read_eq
theorem liveBlocks : LiveBlocks := liveBlocks_eq
theorem liveIn : LiveIn := liveIn_eq
theorem readBlocks : ReadBlocks := readBlocks_eq
theorem readIn : ReadIn := readIn_eq
theorem resident : Resident := resident_eq
theorem incomplete : Incomplete := incomplete_eq
theorem credited : Credited := credited_eq
theorem creditedReads : CreditedReads := creditedReads_eq
theorem creditedBits : CreditedBits := creditedBits_eq
theorem creditedReadBits : CreditedReadBits := creditedReadBits_eq
theorem storeMatches : StoreMatches := storeMatches_iff
theorem planMatches : PlanMatches := planMatches_iff
theorem region : Region := region_eq
theorem stepRegion : StepRegion := stepRegion_eq
theorem ingress : Ingress := ingress_eq
theorem credits : Credits := fun i p price X marked lo hi => credits_eq i p X price marked lo hi

end Properties.Requirement
