module

public import Definitions.Core.Hardware
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# What a machine's limits leave outside a permitted part

For a computation `X ∈ feasible M T`: a node computes at most `rate` gates a cycle (`gatesAt_card`), so `X` has at
most `T · Σ rate` computed gates (`work`); a value read after the cycle it was computed in is held at some node at the
end of every cycle between (`held_between`). If a permitted part `S` uses `x` of a resource, what lies outside `S` uses
at most the machine's limit less `x` (`work_outside`, `memory_outside`, `bandwidth_outside`), so no feasible
computation both uses `x` inside a permitted part and needs more than the rest outside every one (`work_excludes`,
`memory_excludes`, `bandwidth_excludes`).
-/

@[expose] public section

namespace Verity.Hardware

open Finset

/-- A copy of `K` has as many gates as `K` has computed gates. -/
theorem Copy.card {K C : Circuit} {G : Finset (Fin C.N)} (c : Copy K C G) : G.card = K.size K.computed := by
  rw [← c.onto, Circuit.size]
  refine card_image_of_injOn fun g hg h hh e => ?_
  exact c.inj g h (mem_filter.mp hg).2 (mem_filter.mp hh).2 e

/-- Gates tiled by a node's units number at most the node's rate. -/
theorem Tiling.card {C : Circuit} {G : Finset (Fin C.N)} {n : Node} (t : Tiling C G n) : G.card ≤ n.rate := by
  calc G.card = (t.tiles.biUnion Prod.snd).card := by rw [t.cover]
    _ ≤ ∑ x ∈ t.tiles, x.2.card := card_biUnion_le
    _ = ∑ x ∈ t.tiles, (n.unit x.1).size (n.unit x.1).computed :=
        sum_congr rfl fun x hx => (t.copy x hx).some.card
    _ = ∑ i, ∑ x ∈ t.tiles with x.1 = i, (n.unit x.1).size (n.unit x.1).computed :=
        (sum_fiberwise t.tiles Prod.fst _).symm
    _ = ∑ i, (t.tiles.filter fun x => x.1 = i).card * (n.unit i).size (n.unit i).computed := by
        refine sum_congr rfl fun i _ => ?_
        rw [sum_congr rfl fun x hx => by rw [(mem_filter.mp hx).2], sum_const, smul_eq_mul]
    _ ≤ n.rate := sum_le_sum fun i _ => Nat.mul_le_mul_right _ (t.count i)

theorem width_inter_add_diff (C : Circuit) (A S : Finset (Fin C.N)) :
    C.width (A ∩ S) + C.width (A \ S) = C.width A := by
  have h := sum_sdiff (f := C.w) (inter_subset_left : A ∩ S ⊆ A)
  rw [sdiff_inter_self_left] at h
  simp only [Circuit.width]
  omega

namespace Computation

variable {ι : Type} {T : ℕ} (X : Computation ι T) {M : Machine ι}

/-- **A node's work in a cycle** is at most its rate. -/
theorem gatesAt_card (hW : X.Within M) (c : ℕ) (p : ι) : (X.gatesAt c p).card ≤ (M.node p).rate :=
  (hW.work c p).some.card

/-- **A machine's work**: `T` cycles compute at most `T` times the nodes' rates. -/
theorem work [Fintype ι] (hW : X.Within M) : X.C.size X.C.computed ≤ T * ∑ p, (M.node p).rate := by
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
    _ ≤ ∑ _c ∈ range T, ∑ p, (M.node p).rate :=
        sum_le_sum fun c _ => card_biUnion_le.trans (sum_le_sum fun p _ => X.gatesAt_card hW (c + 1) p)
    _ = T * ∑ p, (M.node p).rate := by rw [sum_const, card_range, smul_eq_mul]

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
theorem work_outside [Fintype ι] (hW : X.Within M) {S : Finset (Fin X.C.N)} {x : ℕ}
    (hx : x ≤ X.C.size (X.C.computed ∩ S)) :
    X.C.size (X.C.computed \ S) ≤ T * ∑ p, (M.node p).rate - x := by
  have h := X.work hW
  have hsplit := card_inter_add_card_sdiff X.C.computed S
  simp only [Circuit.size] at h hx ⊢
  omega

/-- **Memory outside a permitted part**: if the nodes `A` hold `x` bits of `S` at the end of cycle `c`, they hold at
most their state less `x` bits outside it. -/
theorem memory_outside (hW : X.Within M) (A : Finset ι) {S : Finset (Fin X.C.N)} {x c : ℕ}
    (hx : x ≤ ∑ p ∈ A, X.C.width (X.hold c p ∩ S)) :
    ∑ p ∈ A, X.C.width (X.hold c p \ S) ≤ ∑ p ∈ A, (M.node p).state - x := by
  have h : ∑ p ∈ A, (X.C.width (X.hold c p ∩ S) + X.C.width (X.hold c p \ S)) ≤ ∑ p ∈ A, (M.node p).state :=
    sum_le_sum fun p _ => (width_inter_add_diff X.C _ S).le.trans (hW.state c p)
  rw [sum_add_distrib] at h
  omega

/-- **Bandwidth outside a permitted part**: if the link from `p` to `q` carries `x` bits of `S` over the `T` cycles, it
carries at most its capacity less `x` bits outside it. -/
theorem bandwidth_outside (hW : X.Within M) (p q : ι) {S : Finset (Fin X.C.N)} {x : ℕ}
    (hx : x ≤ ∑ c ∈ range T, X.C.width (X.sent c p q ∩ S)) :
    ∑ c ∈ range T, X.C.width (X.sent c p q \ S) ≤ M.link p q * T - x := by
  have h : ∑ c ∈ range T, (X.C.width (X.sent c p q ∩ S) + X.C.width (X.sent c p q \ S)) ≤
      ∑ _c ∈ range T, M.link p q :=
    sum_le_sum fun c _ => (width_inter_add_diff X.C _ S).le.trans (hW.link c p q)
  rw [sum_add_distrib, sum_const, card_range, smul_eq_mul, mul_comm] at h
  omega

end Computation

variable {ι : Type} {T : ℕ} (M : Machine ι) (P : Permitted)

/-- **Work**: no computation the machine can run uses `x` gates in a permitted part and needs more than the machine's
remaining work outside every one. -/
theorem work_excludes [Fintype ι] (x : ℕ) :
    needsWork P (T * ∑ p, (M.node p).rate - x) ∩ usesWork P x ∩ feasible M T = (∅ : Set (Computation ι T)) := by
  ext X
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false]
  rintro ⟨⟨hB, S, hP, hx⟩, -, hW⟩
  exact absurd (X.work_outside hW hx) (not_le.mpr (hB S hP))

/-- **Memory**: no computation the machine can run holds `x` bits of a permitted part at the nodes `A` at the end of
every cycle and, at the end of some cycle, more than their remaining state outside any. -/
theorem memory_excludes (A : Finset ι) (x : ℕ) :
    needsMemory P A (∑ p ∈ A, (M.node p).state - x) ∩ usesMemory P A x ∩ feasible M T =
      (∅ : Set (Computation ι T)) := by
  ext X
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false]
  rintro ⟨⟨hB, S, hP, hx⟩, -, hW⟩
  obtain ⟨c, hc, hlt⟩ := hB S hP
  exact absurd (X.memory_outside hW A (hx c hc)) (not_le.mpr hlt)

/-- **Bandwidth**: no computation the machine can run carries `x` bits of a permitted part on the link from `p` to `q`
and more than the link's remaining capacity outside any. -/
theorem bandwidth_excludes (p q : ι) (x : ℕ) :
    needsBandwidth P p q (M.link p q * T - x) ∩ usesBandwidth P p q x ∩ feasible M T =
      (∅ : Set (Computation ι T)) := by
  ext X
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false]
  rintro ⟨⟨hB, S, hP, hx⟩, -, hW⟩
  exact absurd (X.bandwidth_outside hW p q hx) (not_le.mpr (hB S hP))

end Verity.Hardware
