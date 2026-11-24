module

public import Definitions.Core.Hardware
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset

/-!
# What a machine's limits leave outside a permitted part

For a computation `X ∈ feasible M T`:

* its gates use at most `T` cycles of each resource (`used_le`), so if a permitted part `S` uses `x` of some
  resources, the gates outside `S` use at most what is left (`work_outside`), and number at most that where every gate
  uses at least one (`gates_outside`);
* a live value is held at some node (`live_subset`), so if the nodes hold `x` bits of `S`, the live values outside `S`
  have at most the nodes' state less `x` bits (`memory_outside`, `live_outside`);
* a value that leaves a set of nodes is carried by a link out of it (`leaving_subset`), so if those links carry `x`
  bits of `S`, the values outside `S` that leave have at most the links' capacity less `x` bits (`bandwidth_outside`,
  `leaving_outside`).

So no feasible computation both uses `x` inside a permitted part and needs more than the rest outside every one
(`work_excludes`, `memory_excludes`, `bandwidth_excludes`).
-/

@[expose] public section

namespace Verity.Hardware

open Finset

theorem width_inter_add_diff (C : Circuit) (A S : Finset (Fin C.N)) :
    C.width (A ∩ S) + C.width (A \ S) = C.width A := by
  have h := sum_sdiff (f := C.w) (inter_subset_left : A ∩ S ⊆ A)
  rw [sdiff_inter_self_left] at h
  simp only [Circuit.width]
  omega

theorem width_inter_add_compl (C : Circuit) (A S : Finset (Fin C.N)) :
    C.width (A ∩ S) + C.width (A ∩ Sᶜ) = C.width A := by
  rw [← sdiff_eq_inter_compl]
  exact width_inter_add_diff C A S

theorem width_biUnion_le (C : Circuit) {α : Type} (s : Finset α) (f : α → Finset (Fin C.N)) :
    C.width (s.biUnion f) ≤ ∑ x ∈ s, C.width (f x) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp [Circuit.width]
  | insert a s ha ih =>
    rw [biUnion_insert, sum_insert ha]
    have h := sum_union_inter (s₁ := f a) (s₂ := s.biUnion f) (f := C.w)
    simp only [Circuit.width] at ih ⊢
    omega

namespace Computation

variable {ι : Type} {M : Machine ι} {T : ℕ} (X : Computation M T)

/-- **A resource's use over the run**: the gates use at most `T` cycles of it. -/
theorem cost_le (hW : X.Within) (e : M.Res) : ∑ g ∈ X.C.computed, X.cost e g ≤ (M.node e.1).cap e.2 * T := by
  classical
  have hmaps : ∀ g ∈ X.C.computed, X.cycle g ∈ range (T + 1) := fun g hg =>
    mem_range.mpr (Nat.lt_succ_of_le (X.cycle_le g (mem_filter.mp hg).2))
  have h0 : X.gatesAt 0 = ∅ := by
    refine filter_eq_empty_iff.mpr fun g hg h => ?_
    have := X.cycle_pos g (mem_filter.mp hg).2
    omega
  rw [← sum_fiberwise_of_maps_to hmaps, sum_range_succ']
  have hz : ∑ g ∈ X.C.computed with X.cycle g = 0, X.cost e g = 0 := by
    change ∑ g ∈ X.gatesAt 0, X.cost e g = 0
    rw [h0, sum_empty]
  rw [hz, add_zero]
  calc ∑ c ∈ range T, ∑ g ∈ X.C.computed with X.cycle g = c + 1, X.cost e g
      ≤ ∑ _c ∈ range T, (M.node e.1).cap e.2 := sum_le_sum fun c _ => hW.work (c + 1) e
    _ = (M.node e.1).cap e.2 * T := by rw [sum_const, card_range, smul_eq_mul, mul_comm]

theorem used_add_compl (R : Finset M.Res) (S : Finset (Fin X.C.N)) :
    X.used R S + X.used R Sᶜ = ∑ e ∈ R, ∑ g ∈ X.C.computed, X.cost e g := by
  simp only [used]
  rw [← sum_add_distrib]
  refine sum_congr rfl fun e _ => ?_
  have h := sum_sdiff (f := X.cost e) (inter_subset_left : X.C.computed ∩ S ⊆ X.C.computed)
  rw [sdiff_inter_self_left, sdiff_eq_inter_compl] at h
  omega

/-- **The resources' use over the run**: the gates use at most `T` cycles of the resources `R`. -/
theorem used_le (hW : X.Within) (R : Finset M.Res) (S : Finset (Fin X.C.N)) :
    X.used R S + X.used R Sᶜ ≤ (∑ e ∈ R, (M.node e.1).cap e.2) * T := by
  rw [used_add_compl, sum_mul]
  exact sum_le_sum fun e _ => X.cost_le hW e

/-- **Work outside a permitted part**: if `S` uses `x` of the resources `R`, the gates outside `S` use at most those
resources' `T` cycles less `x`. -/
theorem work_outside (hW : X.Within) (R : Finset M.Res) {S : Finset (Fin X.C.N)} {x : ℕ} (hx : x ≤ X.used R S) :
    X.used R Sᶜ ≤ (∑ e ∈ R, (M.node e.1).cap e.2) * T - x := by
  have h := X.used_le hW R S
  omega

/-- **Gates outside a permitted part**: where every computed gate uses at least one of the resources `R`, the computed
gates outside `S` number at most what they use. -/
theorem gates_outside (R : Finset M.Res) (S : Finset (Fin X.C.N))
    (h1 : ∀ g ∈ X.C.computed, 1 ≤ ∑ e ∈ R, X.cost e g) : X.C.size (X.C.computed \ S) ≤ X.used R Sᶜ := by
  simp only [used, Circuit.size]
  rw [sum_comm, sdiff_eq_inter_compl, card_eq_sum_ones]
  exact sum_le_sum fun g hg => h1 g (mem_inter.mp hg).1

/-- A value held at the end of cycle `c + j` that is an input or was computed by cycle `c` is held at some node at the
end of cycle `c`. -/
theorem held_back (hO : X.Order) {c : ℕ} {a : Fin X.C.N} (ha : X.C.isInput a = false → X.cycle a ≤ c) :
    ∀ j p, a ∈ X.hold (c + j) p → ∃ q, a ∈ X.hold c q := by
  intro j
  induction j with
  | zero => exact fun p h => ⟨p, h⟩
  | succ j ih =>
    intro p h
    rcases hO.kept (c + j) p a h with ⟨hi, hc, -⟩ | h' | ⟨q, hq⟩
    · have := ha hi
      omega
    · exact ih p h'
    · exact ih q (hO.sent (c + j) q p hq)

/-- **A live value is held** at some node. -/
theorem live_subset [Fintype ι] (hO : X.Order) (c : ℕ) : X.live c ⊆ univ.biUnion (X.hold c) := by
  intro a ha
  simp only [live, mem_filter, mem_univ, true_and] at ha
  obtain ⟨hac, g, hg, hr, hc⟩ := ha
  simp only [mem_biUnion, mem_univ, true_and]
  rcases hO.reads g hg a hr with ⟨hi, he, -⟩ | h
  · have := hac hi
    omega
  · obtain ⟨j, hj⟩ : ∃ j, X.cycle g - 1 = c + j := ⟨X.cycle g - 1 - c, by omega⟩
    rw [hj] at h
    exact X.held_back hO hac j _ h

/-- **Memory outside a permitted part**: if the nodes `A` hold `x` bits of `S` at the end of cycle `c`, they hold at
most their state less `x` bits outside it. -/
theorem memory_outside (hW : X.Within) (A : Finset ι) {S : Finset (Fin X.C.N)} {x c : ℕ} (hx : x ≤ X.held A c S) :
    X.held A c Sᶜ ≤ ∑ p ∈ A, (M.node p).state - x := by
  have h : X.held A c S + X.held A c Sᶜ ≤ ∑ p ∈ A, (M.node p).state := by
    simp only [held]
    rw [← sum_add_distrib]
    exact sum_le_sum fun p _ => (width_inter_add_compl X.C _ S).le.trans (hW.state c p)
  omega

/-- **Live values outside a permitted part**: if the machine holds `x` bits of `S` at the end of cycle `c`, the live
values outside `S` have at most its state less `x` bits. -/
theorem live_outside [Fintype ι] (hO : X.Order) (hW : X.Within) {S : Finset (Fin X.C.N)} {x c : ℕ}
    (hx : x ≤ X.held univ c S) : X.C.width (X.live c \ S) ≤ ∑ p, (M.node p).state - x := by
  classical
  have hsub : X.live c \ S ⊆ univ.biUnion fun p => X.hold c p ∩ Sᶜ := by
    intro a ha
    obtain ⟨hl, hS⟩ := mem_sdiff.mp ha
    obtain ⟨p, -, hp⟩ := mem_biUnion.mp (X.live_subset hO c hl)
    exact mem_biUnion.mpr ⟨p, mem_univ p, mem_inter.mpr ⟨hp, mem_compl.mpr hS⟩⟩
  exact ((sum_le_sum_of_subset hsub).trans (width_biUnion_le X.C _ _)).trans (X.memory_outside hW univ hx)

/-- A value that leaves `U`, held at a node outside `U`, was carried there by a link out of `U`. -/
theorem sent_of_hold (hO : X.Order) (U : Finset ι) {a : Fin X.C.N} (hc : X.C.isInput a = false → X.place a ∈ U)
    (hi : X.C.isInput a = true → ∀ p ∉ U, a ∉ X.hold 0 p) :
    ∀ c q, q ∉ U → a ∈ X.hold c q → ∃ c' < c, ∃ p ∈ U, ∃ q' ∉ U, a ∈ X.sent c' p q' := by
  intro c
  induction c with
  | zero => exact fun q hq h => absurd h (hi (hO.start q a h) q hq)
  | succ c ih =>
    intro q hq h
    rcases hO.kept c q a h with ⟨hin, -, hp⟩ | h' | ⟨p, hp⟩
    · exact absurd (hp ▸ hc hin) hq
    · obtain ⟨c', hc', r⟩ := ih q hq h'
      exact ⟨c', Nat.lt_succ_of_lt hc', r⟩
    · by_cases hpU : p ∈ U
      · exact ⟨c, Nat.lt_succ_self c, p, hpU, q, hq, hp⟩
      · obtain ⟨c', hc', r⟩ := ih p hpU (hO.sent c p q hp)
        exact ⟨c', Nat.lt_succ_of_lt hc', r⟩

/-- **A value that leaves is carried**: by a link out of `U`, in one of the `T` cycles. -/
theorem leaving_subset [Fintype ι] [DecidableEq ι] (hO : X.Order) (U : Finset ι) :
    X.leaving U ⊆ (U ×ˢ Uᶜ).biUnion fun e => (range T).biUnion fun c => X.sent c e.1 e.2 := by
  intro a ha
  simp only [leaving, mem_filter, mem_univ, true_and] at ha
  obtain ⟨hc, hi, g, hg, hgU, hr⟩ := ha
  have hheld : a ∈ X.hold (X.cycle g - 1) (X.place g) := by
    rcases hO.reads g hg a hr with ⟨hin, -, hp⟩ | h
    · exact absurd (hp ▸ hc hin) hgU
    · exact h
  obtain ⟨c', hc', p, hp, q, hq, hs⟩ := X.sent_of_hold hO U hc hi _ _ hgU hheld
  have hT := X.cycle_le g hg
  exact mem_biUnion.mpr ⟨(p, q), mem_product.mpr ⟨hp, mem_compl.mpr hq⟩,
    mem_biUnion.mpr ⟨c', mem_range.mpr (by omega), hs⟩⟩

/-- **Bandwidth outside a permitted part**: if the links `E` carry `x` bits of `S` over the `T` cycles, they carry at
most their capacity less `x` bits outside it. -/
theorem bandwidth_outside (hW : X.Within) (E : Finset (ι × ι)) {S : Finset (Fin X.C.N)} {x : ℕ}
    (hx : x ≤ X.carried E S) : X.carried E Sᶜ ≤ (∑ e ∈ E, M.link e.1 e.2) * T - x := by
  have h : X.carried E S + X.carried E Sᶜ ≤ ∑ e ∈ E, ∑ _c ∈ range T, M.link e.1 e.2 := by
    simp only [carried]
    rw [← sum_add_distrib]
    refine sum_le_sum fun e _ => ?_
    rw [← sum_add_distrib]
    exact sum_le_sum fun c _ => (width_inter_add_compl X.C _ S).le.trans (hW.link c e.1 e.2)
  simp only [sum_const, card_range, smul_eq_mul] at h
  rw [← Finset.mul_sum, mul_comm] at h
  omega

/-- **Values outside a permitted part that leave**: if the links out of `U` carry `x` bits of `S` over the `T` cycles,
the values outside `S` that leave `U` have at most those links' capacity less `x` bits. -/
theorem leaving_outside [Fintype ι] [DecidableEq ι] (hO : X.Order) (hW : X.Within) (U : Finset ι)
    {S : Finset (Fin X.C.N)} {x : ℕ} (hx : x ≤ X.carried (U ×ˢ Uᶜ) S) :
    X.C.width (X.leaving U \ S) ≤ (∑ e ∈ U ×ˢ Uᶜ, M.link e.1 e.2) * T - x := by
  have hsub : X.leaving U \ S ⊆
      (U ×ˢ Uᶜ).biUnion fun e => (range T).biUnion fun c => X.sent c e.1 e.2 ∩ Sᶜ := by
    intro a ha
    obtain ⟨hl, hS⟩ := mem_sdiff.mp ha
    obtain ⟨e, he, h⟩ := mem_biUnion.mp (X.leaving_subset hO U hl)
    obtain ⟨c, hc, h⟩ := mem_biUnion.mp h
    exact mem_biUnion.mpr ⟨e, he, mem_biUnion.mpr ⟨c, hc, mem_inter.mpr ⟨h, mem_compl.mpr hS⟩⟩⟩
  refine ((sum_le_sum_of_subset hsub).trans ((width_biUnion_le X.C _ _).trans ?_)).trans
    (X.bandwidth_outside hW (U ×ˢ Uᶜ) hx)
  exact sum_le_sum fun e _ => width_biUnion_le X.C _ _

/-- **Blocks outside a permitted part**: of pairwise disjoint sets of computed gates, MACs say, those not inside `S`
number at most the computed gates outside `S`. -/
theorem blocks_outside {κ : Type} (B : Finset κ) (m : κ → Finset (Fin X.C.N)) (hm : ∀ b ∈ B, m b ⊆ X.C.computed)
    (hd : ∀ b ∈ B, ∀ b' ∈ B, b ≠ b' → Disjoint (m b) (m b')) (S : Finset (Fin X.C.N)) :
    (B.filter fun b => ¬ m b ⊆ S).card ≤ X.C.size (X.C.computed \ S) := by
  classical
  rcases isEmpty_or_nonempty (Fin X.C.N) with hE | hN
  · have h0 : (B.filter fun b => ¬ m b ⊆ S) = ∅ :=
      filter_eq_empty_iff.mpr fun b _ h => h (by rw [eq_empty_of_isEmpty (m b)]; exact empty_subset _)
    rw [h0]
    exact Nat.zero_le _
  have hex : ∀ b ∈ B.filter fun b => ¬ m b ⊆ S, ∃ g, g ∈ m b ∧ g ∉ S := fun b hb =>
    not_subset.mp (mem_filter.mp hb).2
  choose! f hf using hex
  refine card_le_card_of_injOn f (fun b hb => ?_) fun b hb b' hb' e => ?_
  · exact mem_sdiff.mpr ⟨hm b (mem_filter.mp hb).1 (hf b hb).1, (hf b hb).2⟩
  · by_contra hne
    exact disjoint_left.mp (hd b (mem_filter.mp hb).1 b' (mem_filter.mp hb').1 hne) (hf b hb).1 (e ▸ (hf b' hb').1)

end Computation

variable {ι : Type} {M : Machine ι} {T : ℕ} (P : Permitted)

/-- **Work**: no computation the machine can run uses `x` of the resources `R` inside a permitted part and more than
the resources' remainder outside every one. -/
theorem work_excludes (R : Finset M.Res) (x : ℕ) :
    needsWork P R ((∑ e ∈ R, (M.node e.1).cap e.2) * T - x) ∩ usesWork P R x ∩ feasible M T =
      (∅ : Set (Computation M T)) := by
  ext X
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false]
  rintro ⟨⟨hB, S, hP, hx⟩, -, hW⟩
  exact absurd (X.work_outside hW R hx) (not_le.mpr (hB S hP))

/-- **Memory**: no computation the machine can run holds `x` bits of a permitted part at the nodes `A` at the end of
every cycle and, at the end of some cycle, more than their remaining state outside any. -/
theorem memory_excludes (A : Finset ι) (x : ℕ) :
    needsMemory P A (∑ p ∈ A, (M.node p).state - x) ∩ usesMemory P A x ∩ feasible M T =
      (∅ : Set (Computation M T)) := by
  ext X
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false]
  rintro ⟨⟨hB, S, hP, hx⟩, -, hW⟩
  obtain ⟨c, hc, hlt⟩ := hB S hP
  exact absurd (X.memory_outside hW A (hx c hc)) (not_le.mpr hlt)

/-- **Bandwidth**: no computation the machine can run carries `x` bits of a permitted part on the links `E` and more
than the links' remaining capacity outside any. -/
theorem bandwidth_excludes (E : Finset (ι × ι)) (x : ℕ) :
    needsBandwidth P E ((∑ e ∈ E, M.link e.1 e.2) * T - x) ∩ usesBandwidth P E x ∩ feasible M T =
      (∅ : Set (Computation M T)) := by
  ext X
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false]
  rintro ⟨⟨hB, S, hP, hx⟩, -, hW⟩
  exact absurd (X.bandwidth_outside hW E hx) (not_le.mpr (hB S hP))

/-! ### A server of GPUs -/

section Server

variable {n : ℕ} (die mem host outside : Node) (read write bus peer net : ℕ)

/-- A server's dies together read their device memories at `n · read` bits a cycle. -/
theorem server_reads :
    ∑ e ∈ reads n, (server n die mem host outside read write bus peer net).link e.1 e.2 = n * read := by
  rw [reads, sum_image fun i _ j _ h => by simpa using congrArg Prod.fst h]
  simp [server]

/-- A server's device memories together hold `n` times one's state. -/
theorem server_mems :
    ∑ p ∈ mems n, ((server n die mem host outside read write bus peer net).node p).state = n * mem.state := by
  rw [mems, sum_image fun i _ j _ h => by simpa using h]
  simp [server]

end Server

end Verity.Hardware
