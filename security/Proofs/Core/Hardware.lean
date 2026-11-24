module

public import Definitions.Core.Hardware
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset

/-!
# What a machine's limits leave outside a permitted part

For a computation `X ∈ feasible M T`: a node whose resources admit at most `W` gates computes at most `W` a cycle
(`gatesAt_card`), so `X` has at most `T · Σ W` computed gates (`work`); a value read after the cycle it was computed in is held at some node at the
end of every cycle between (`held_between`). If a permitted part `S` uses `x` of a resource, what lies outside `S` uses
at most the machine's limit less `x` (`work_outside`, `units_outside`, `memory_outside`, `bandwidth_outside`), so no feasible
computation both uses `x` inside a permitted part and needs more than the rest outside every one (`work_excludes`,
`units_excludes`, `memory_excludes`, `bandwidth_excludes`).
-/

@[expose] public section

namespace Verity.Hardware

open Finset

/-- A copy of `K` has as many gates as `K` has computed gates. -/
theorem Copy.card {K C : Circuit} {G : Finset (Fin C.N)} (c : Copy K C G) : G.card = K.size K.computed := by
  rw [← c.onto, Circuit.size]
  refine card_image_of_injOn fun g hg h hh e => ?_
  exact c.inj g h (mem_filter.mp hg).2 (mem_filter.mp hh).2 e

theorem width_inter_add_diff (C : Circuit) (A S : Finset (Fin C.N)) :
    C.width (A ∩ S) + C.width (A \ S) = C.width A := by
  have h := sum_sdiff (f := C.w) (inter_subset_left : A ∩ S ⊆ A)
  rw [sdiff_inter_self_left] at h
  simp only [Circuit.width]
  omega

namespace Computation

variable {ι : Type} {M : Machine ι} {T : ℕ} (X : Computation M T)

/-- **A node's work in a cycle**: at most `W` gates, when its resources admit no more. -/
theorem gatesAt_card (hW : X.Within) (c : ℕ) (p : ι) {W : ℕ} (hG : (M.node p).GatesWithin W) :
    (X.gatesAt c p).card ≤ W := by
  classical
  set n := M.node p
  have hcount : ∀ j, ∑ i, n.use j i * ((X.tiles c p).filter fun x => x.1 = i).card ≤ n.cap j := by
    intro j
    refine le_trans (le_of_eq ?_) (hW.work c p j)
    rw [← sum_fiberwise (X.tiles c p) Prod.fst fun x => n.use j x.1]
    refine sum_congr rfl fun i _ => ?_
    rw [sum_congr rfl fun x hx => by rw [(mem_filter.mp hx).2], sum_const, smul_eq_mul, mul_comm]
  calc (X.gatesAt c p).card = ((X.tiles c p).biUnion Prod.snd).card := by rw [hW.cover]
    _ ≤ ∑ x ∈ X.tiles c p, x.2.card := card_biUnion_le
    _ = ∑ x ∈ X.tiles c p, (n.unit x.1).size (n.unit x.1).computed :=
        sum_congr rfl fun x hx => (hW.copy c p x hx).some.card
    _ = ∑ i, ∑ x ∈ X.tiles c p with x.1 = i, (n.unit x.1).size (n.unit x.1).computed :=
        (sum_fiberwise (X.tiles c p) Prod.fst _).symm
    _ = ∑ i, ((X.tiles c p).filter fun x => x.1 = i).card * (n.unit i).size (n.unit i).computed := by
        refine sum_congr rfl fun i _ => ?_
        rw [sum_congr rfl fun x hx => by rw [(mem_filter.mp hx).2], sum_const, smul_eq_mul]
    _ ≤ W := hG _ hcount

/-- **A machine's work**: `T` cycles compute at most `T` times what the nodes compute in one. -/
theorem work [Fintype ι] (hW : X.Within) {W : ι → ℕ} (hG : ∀ p, (M.node p).GatesWithin (W p)) :
    X.C.size X.C.computed ≤ T * ∑ p, W p := by
  classical
  have hsub : X.C.computed ⊆ (range T).biUnion fun c => univ.biUnion fun p => X.gatesAt (c + 1) p := by
    intro g hg
    have hi := (mem_filter.mp hg).2
    have h1 := X.cycle_pos g hi
    have h2 := X.cycle_le g hi
    simp only [mem_biUnion, mem_range, mem_univ, true_and, gatesAt, mem_filter]
    exact ⟨X.cycle g - 1, by omega, X.place g, hg, by omega, rfl⟩
  calc X.C.size X.C.computed
      ≤ ((range T).biUnion fun c => univ.biUnion fun p => X.gatesAt (c + 1) p).card := card_le_card hsub
    _ ≤ ∑ c ∈ range T, (univ.biUnion fun p => X.gatesAt (c + 1) p).card := card_biUnion_le
    _ ≤ ∑ _c ∈ range T, ∑ p, W p :=
        sum_le_sum fun c _ => card_biUnion_le.trans (sum_le_sum fun p _ => X.gatesAt_card hW (c + 1) p (hG p))
    _ = T * ∑ p, W p := by rw [sum_const, card_range, smul_eq_mul]

/-- **Resources outside a permitted part**: if the copies inside `S` use `x` of the resources `R`, the other copies
use at most those resources' `T` cycles less `x`. -/
theorem units_outside (hW : X.Within) (R : Finset ((p : ι) × Fin (M.node p).m)) {S : Finset (Fin X.C.N)} {x : ℕ}
    (hx : x ≤ X.usedIn R S) : X.usedOut R S ≤ (∑ e ∈ R, (M.node e.1).cap e.2) * T - x := by
  classical
  have h : X.usedIn R S + X.usedOut R S ≤ ∑ e ∈ R, ∑ _c ∈ range T, (M.node e.1).cap e.2 := by
    simp only [usedIn, usedOut]
    rw [← sum_add_distrib]
    refine sum_le_sum fun e _ => ?_
    rw [← sum_add_distrib]
    refine sum_le_sum fun c _ => ?_
    rw [sum_filter_add_sum_filter_not]
    exact hW.work (c + 1) e.1 e.2
  simp only [sum_const, card_range, smul_eq_mul] at h
  rw [← Finset.mul_sum, mul_comm] at h
  omega

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

/-- **A live value is held**: a value that is an input or was computed by cycle `c`, read by a gate computed after
cycle `c`, is held at some node at the end of cycle `c`. -/
theorem held_between (hO : X.Order) {c : ℕ} {a g : Fin X.C.N} (hg : X.C.isInput g = false) (hr : a ∈ X.C.reads g)
    (ha : X.C.isInput a = false → X.cycle a ≤ c) (hc : c < X.cycle g) : ∃ q, a ∈ X.hold c q := by
  rcases hO.reads g hg a hr with ⟨hi, he, -⟩ | h
  · have := ha hi
    omega
  · obtain ⟨j, hj⟩ : ∃ j, X.cycle g - 1 = c + j := ⟨X.cycle g - 1 - c, by omega⟩
    rw [hj] at h
    exact X.held_back hO ha j _ h

/-- **Work outside a permitted part**: if `S` has `x` computed gates, at most the machine's work less `x` lie outside. -/
theorem work_outside [Fintype ι] (hW : X.Within) {W : ι → ℕ} (hG : ∀ p, (M.node p).GatesWithin (W p))
    {S : Finset (Fin X.C.N)} {x : ℕ} (hx : x ≤ X.C.size (X.C.computed ∩ S)) :
    X.C.size (X.C.computed \ S) ≤ T * ∑ p, W p - x := by
  have h := X.work hW hG
  have hsplit := card_inter_add_card_sdiff X.C.computed S
  simp only [Circuit.size] at h hx ⊢
  omega

theorem width_compl (A S : Finset (Fin X.C.N)) : X.C.width (A ∩ Sᶜ) = X.C.width (A \ S) := by
  rw [← sdiff_eq_inter_compl]

/-- **Memory outside a permitted part**: if the nodes `A` hold `x` bits of `S` at the end of cycle `c`, they hold at
most their state less `x` bits outside it. -/
theorem memory_outside (hW : X.Within) (A : Finset ι) {S : Finset (Fin X.C.N)} {x c : ℕ} (hx : x ≤ X.held A c S) :
    X.held A c Sᶜ ≤ ∑ p ∈ A, (M.node p).state - x := by
  have h : X.held A c S + X.held A c Sᶜ ≤ ∑ p ∈ A, (M.node p).state := by
    simp only [held]
    rw [← sum_add_distrib]
    refine sum_le_sum fun p _ => ?_
    rw [X.width_compl]
    exact (width_inter_add_diff X.C _ S).le.trans (hW.state c p)
  omega

/-- **Bandwidth outside a permitted part**: if the links `E` carry `x` bits of `S` over the `T` cycles, they carry at
most their capacity less `x` bits outside it. -/
theorem bandwidth_outside (hW : X.Within) (E : Finset (ι × ι)) {S : Finset (Fin X.C.N)} {x : ℕ}
    (hx : x ≤ X.carried E S) : X.carried E Sᶜ ≤ (∑ e ∈ E, M.link e.1 e.2) * T - x := by
  have h : X.carried E S + X.carried E Sᶜ ≤ ∑ e ∈ E, ∑ _c ∈ range T, M.link e.1 e.2 := by
    simp only [carried]
    rw [← sum_add_distrib]
    refine sum_le_sum fun e _ => ?_
    rw [← sum_add_distrib]
    refine sum_le_sum fun c _ => ?_
    rw [X.width_compl]
    exact (width_inter_add_diff X.C _ S).le.trans (hW.link c e.1 e.2)
  simp only [sum_const, card_range, smul_eq_mul] at h
  rw [← Finset.mul_sum, mul_comm] at h
  omega

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

/-- **Work**: no computation the machine can run uses `x` gates in a permitted part and needs more than the machine's
remaining work outside every one. -/
theorem work_excludes [Fintype ι] {W : ι → ℕ} (hG : ∀ p, (M.node p).GatesWithin (W p)) (x : ℕ) :
    needsWork P (T * ∑ p, W p - x) ∩ usesWork P x ∩ feasible M T = (∅ : Set (Computation M T)) := by
  ext X
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false]
  rintro ⟨⟨hB, S, hP, hx⟩, -, hW⟩
  exact absurd (X.work_outside hW hG hx) (not_le.mpr (hB S hP))

/-- **Resources**: no computation the machine can run uses `x` of the resources `R` in copies inside a permitted part
and more than the resources' remainder in copies outside every one. -/
theorem units_excludes (R : Finset ((p : ι) × Fin (M.node p).m)) (x : ℕ) :
    needsUnits P R ((∑ e ∈ R, (M.node e.1).cap e.2) * T - x) ∩ usesUnits P R x ∩ feasible M T =
      (∅ : Set (Computation M T)) := by
  ext X
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false]
  rintro ⟨⟨hB, S, hP, hx⟩, -, hW⟩
  exact absurd (X.units_outside hW R hx) (not_le.mpr (hB S hP))

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

/-- A server's dies together have `n` times one die's resource `j`. -/
theorem server_dies (j : Fin die.m) :
    ∑ e ∈ univ.image fun i : Fin n =>
        (⟨Loc.die i, j⟩ : (p : Loc n) × Fin ((server n die mem host outside read write bus peer net).node p).m),
      ((server n die mem host outside read write bus peer net).node e.1).cap e.2 = n * die.cap j := by
  rw [sum_image fun i _ k _ h => by simpa using congrArg Sigma.fst h]
  simp [server]

end Server

end Verity.Hardware
