import Definitions.Nci.Guarantees.Products
import Proofs.Core.Combinatorics
import Proofs.Nci.Formalism.Slowdown

/-!
# Products of two fresh operands: the cap, its word-level form and the slowdown

For a partition `P` into `m` inference blocks, block `u` with gates `I_u` and reads `U_u = in(I_u) \ θ`, the four facts
the amortized cap (`Verity.Combinatorics.sum_card_le`) reads, with `T_u` the products whose MAC lies in `I_u`:

1. **Operands** (`mul_card_le_cones`): `b_A |π₁₂ T_u| + b_B |π₂₃ T_u| ≤ |U_u ∩ cones|`. A whole MAC reads every bit of
   its operands, so each operand it touches is held or read in full, and widths as passed charge `b_A` (or `b_B`) bits
   of its cone; the cones are disjoint.
2. **Reads** (`card_add_card_le`): `|U_u ∩ cones| + |U_u ∩ macs| ≤ k`, the cones and the MACs being disjoint.
3. **Partial sums** (`partial_sums`): `b_C Σ_u |π₁₃ T_u| ≤ E + Σ_u |U_u ∩ macs|`. A block holding a whole MAC of `C_ij`
   hands `b_C` bits of `C_ij`'s gates on or holds them as `C_ij`'s final bits; what a block hands on another reads.
4. **Splits** (`card_le_whole_add`): `N ≤ Σ_u |T_u| + Σ_u |U_u ∩ macs|`. A MAC no block holds whole is split, and a
   read across the split is a read of a MAC's gate.

`Cap` takes the cap at the weight `s = 1` on the MACs' reads, which pays for the splits; `WordCap` has no splits and
takes it at `s = 0` and `μ*`. `ChainCap` replaces fact 3 with its per-block form along chains (`chain_sums`):
`b_C |π₁₃ T_u| ≤ |U_u ∩ macs| + b_C s_u`, `s_u` the entries whose first MAC is in `I_u` (`starts`). An entry `I_u`
touches but doesn't start has a first MAC in `I_u` whose predecessor lies in another block, and the `b_C` gates of the
predecessor it reads are reads of `I_u` (`le_card_entry`); each first MAC lies in one block, `Σ_u s_u ≤ ni nj`
(`sum_card_starts_le`). `Bound` divides `Cap` by `|C| / c` at each partition. The `example` at the end gives the one
slowdown (`Nci.Formalism.Guarantees.Slowdown`) the products kind as a kind `J`, its `Λ_J` the products kind's bound.
-/

namespace Nci.Products

open Finset
open Verity (Declaration Partition)
open Nci.Formalism

variable {d : Declaration}

/-- A computed gate that a gate of `I` reads and `I` doesn't hold is read by `I` beyond `θ`. -/
theorem mem_unregistered {I : Finset (Fin d.D.N)} {g h : Fin d.D.N} (hg : g ∈ I) (hh : h ∉ I)
    (hr : h ∈ d.D.reads g) (hc : h ∈ d.D.computed) : h ∈ d.unregistered I := by
  simp only [Declaration.unregistered, Verity.Circuit.into, mem_sdiff, mem_filter, mem_univ, true_and]
  refine ⟨⟨hh, g, hg, hr⟩, fun hθ => ?_⟩
  have hi := d.θ_input h hθ
  simp only [Verity.Circuit.computed, mem_filter, mem_univ, true_and] at hc
  rw [hc] at hi
  exact Bool.false_ne_true hi

/-- A gate a gate of `I` reads is held by `I` or read by it. -/
theorem mem_union_into {I : Finset (Fin d.D.N)} {g a : Fin d.D.N} (hg : g ∈ I) (hr : a ∈ d.D.reads g) :
    a ∈ I ∪ d.D.into I := by
  by_cases ha : a ∈ I
  · exact mem_union_left _ ha
  · refine mem_union_right _ ?_
    simp only [Verity.Circuit.into, mem_filter, mem_univ, true_and]
    exact ⟨ha, g, hg, hr⟩

/-- Every computed gate is in a block. -/
theorem exists_mem_gates {m : ℕ} (P : Partition d.D m) {g : Fin d.D.N} (hg : g ∈ d.D.computed) :
    ∃ u, g ∈ P.gates u := by
  simp only [Verity.Circuit.computed, mem_filter, mem_univ, true_and] at hg
  have hne : P.unit g ≠ none := fun h => by
    rw [P.input_iff, hg] at h
    exact Bool.false_ne_true h
  obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.1 hne
  exact ⟨u, P.mem_gates.2 hu⟩

/-- A gate is in at most one block. -/
theorem eq_of_mem_gates {m : ℕ} (P : Partition d.D m) {g : Fin d.D.N} {u v : Fin m} (hu : g ∈ P.gates u)
    (hv : g ∈ P.gates v) : u = v :=
  Option.some_injective _ ((P.mem_gates.1 hu).symm.trans (P.mem_gates.1 hv))

namespace Standard

variable {ni nl nj : ℕ} (S : Standard d ni nl nj)

/-- The products whose MAC lies in `I`, a block's `T`. -/
def whole (I : Finset (Fin d.D.N)) : Finset (Fin ni × Fin nl × Fin nj) :=
  univ.filter fun p => S.mac p.1 p.2.1 p.2.2 ⊆ I

theorem mac_subset_macs (i : Fin ni) (l : Fin nl) (j : Fin nj) : S.mac i l j ⊆ S.macs := fun _ hg =>
  mem_biUnion.2 ⟨(i, l, j), mem_univ _, hg⟩

theorem entry_subset_macs (i : Fin ni) (j : Fin nj) : S.entry i j ⊆ S.macs := fun g hg => by
  obtain ⟨l, -, hl⟩ := mem_biUnion.1 hg
  exact S.mac_subset_macs i l j hl

theorem entry_computed (hS : S.Algorithm) (i : Fin ni) (j : Fin nj) : S.entry i j ⊆ d.D.computed := fun g hg => by
  obtain ⟨l, -, hl⟩ := mem_biUnion.1 hg
  exact hS.mac_computed i l j hl

theorem entry_disjoint (hS : S.Algorithm) {e e' : Fin ni × Fin nj} (h : e ≠ e') :
    Disjoint (S.entry e.1 e.2) (S.entry e'.1 e'.2) := by
  refine (disjoint_biUnion_left _ _ _).2 fun l _ => (disjoint_biUnion_right _ _ _).2 fun l' _ =>
    hS.mac_disjoint (e.1, l, e.2) (e'.1, l', e'.2) ?_
  intro heq
  apply h
  simp only [Prod.mk.injEq] at heq
  exact Prod.ext heq.1 heq.2.2

/-- **Operands, `A`**: a block's whole MACs touch at most `|U ∩ ⋃ coneA| / b_A` entries of `A`. -/
theorem mul_card_le_coneA (hS : S.Algorithm) {c k bA bB bC : ℕ} (hW : S.Passed c k bA bB bC)
    {I : Finset (Fin d.D.N)} (hI : InferenceBlock d c k I) :
    bA * ((S.whole I).image fun x => (x.1, x.2.1)).card ≤
      (d.unregistered I ∩ univ.biUnion fun p : Fin ni × Fin nl => S.coneA p.1 p.2).card := by
  calc bA * ((S.whole I).image fun x => (x.1, x.2.1)).card
      = ∑ _p ∈ (S.whole I).image fun x => (x.1, x.2.1), bA := by rw [sum_const, smul_eq_mul, mul_comm]
    _ ≤ ∑ p ∈ (S.whole I).image fun x => (x.1, x.2.1), (d.unregistered I ∩ S.coneA p.1 p.2).card := by
        refine sum_le_sum fun p hp => ?_
        obtain ⟨⟨i, l, j⟩, hx, rfl⟩ := mem_image.1 hp
        have hsub : S.mac i l j ⊆ I := (mem_filter.1 hx).2
        refine hW.operandA I hI i l fun a ha => ?_
        obtain ⟨g, hg, hr⟩ := hS.readsA i l j a ha
        exact mem_union_into (hsub hg) hr
    _ = (((S.whole I).image fun x => (x.1, x.2.1)).biUnion fun p => d.unregistered I ∩ S.coneA p.1 p.2).card := by
        refine (card_biUnion fun p _ q _ hpq => ?_).symm
        exact (hS.coneA_disjoint p q hpq).mono inter_subset_right inter_subset_right
    _ ≤ _ := card_le_card fun g hg => by
        obtain ⟨p, -, hgp⟩ := mem_biUnion.1 hg
        rw [mem_inter] at hgp ⊢
        exact ⟨hgp.1, mem_biUnion.2 ⟨p, mem_univ _, hgp.2⟩⟩

/-- **Operands, `B`**: a block's whole MACs touch at most `|U ∩ ⋃ coneB| / b_B` entries of `B`. -/
theorem mul_card_le_coneB (hS : S.Algorithm) {c k bA bB bC : ℕ} (hW : S.Passed c k bA bB bC)
    {I : Finset (Fin d.D.N)} (hI : InferenceBlock d c k I) :
    bB * ((S.whole I).image fun x => (x.2.1, x.2.2)).card ≤
      (d.unregistered I ∩ univ.biUnion fun p : Fin nl × Fin nj => S.coneB p.1 p.2).card := by
  calc bB * ((S.whole I).image fun x => (x.2.1, x.2.2)).card
      = ∑ _p ∈ (S.whole I).image fun x => (x.2.1, x.2.2), bB := by rw [sum_const, smul_eq_mul, mul_comm]
    _ ≤ ∑ p ∈ (S.whole I).image fun x => (x.2.1, x.2.2), (d.unregistered I ∩ S.coneB p.1 p.2).card := by
        refine sum_le_sum fun p hp => ?_
        obtain ⟨⟨i, l, j⟩, hx, rfl⟩ := mem_image.1 hp
        have hsub : S.mac i l j ⊆ I := (mem_filter.1 hx).2
        refine hW.operandB I hI l j fun b hb => ?_
        obtain ⟨g, hg, hr⟩ := hS.readsB i l j b hb
        exact mem_union_into (hsub hg) hr
    _ = (((S.whole I).image fun x => (x.2.1, x.2.2)).biUnion fun p => d.unregistered I ∩ S.coneB p.1 p.2).card := by
        refine (card_biUnion fun p _ q _ hpq => ?_).symm
        exact (hS.coneB_disjoint p q hpq).mono inter_subset_right inter_subset_right
    _ ≤ _ := card_le_card fun g hg => by
        obtain ⟨p, -, hgp⟩ := mem_biUnion.1 hg
        rw [mem_inter] at hgp ⊢
        exact ⟨hgp.1, mem_biUnion.2 ⟨p, mem_univ _, hgp.2⟩⟩

theorem coneA_coneB_disjoint (hS : S.Algorithm) :
    Disjoint (univ.biUnion fun p : Fin ni × Fin nl => S.coneA p.1 p.2)
      (univ.biUnion fun p : Fin nl × Fin nj => S.coneB p.1 p.2) :=
  (disjoint_biUnion_left _ _ _).2 fun p _ => (disjoint_biUnion_right _ _ _).2 fun q _ =>
    hS.cones_disjoint p.1 p.2 q.1 q.2

/-- **Operands**: `b_A |π₁₂ T| + b_B |π₂₃ T| ≤ |U ∩ cones|` for a block's whole MACs `T`. -/
theorem mul_card_le_cones (hS : S.Algorithm) {c k bA bB bC : ℕ} (hW : S.Passed c k bA bB bC)
    {I : Finset (Fin d.D.N)} (hI : InferenceBlock d c k I) :
    bA * ((S.whole I).image fun x => (x.1, x.2.1)).card + bB * ((S.whole I).image fun x => (x.2.1, x.2.2)).card ≤
      (d.unregistered I ∩ S.cones).card := by
  rw [Standard.cones, inter_union_distrib_left, card_union_of_disjoint
    ((S.coneA_coneB_disjoint hS).mono inter_subset_right inter_subset_right)]
  exact add_le_add (S.mul_card_le_coneA hS hW hI) (S.mul_card_le_coneB hS hW hI)

/-- **Reads**: a block reads its cones' and its MACs' gates within its `k`. -/
theorem card_add_card_le (hS : S.Algorithm) {c k : ℕ} {I : Finset (Fin d.D.N)} (hI : InferenceBlock d c k I) :
    (d.unregistered I ∩ S.cones).card + (d.unregistered I ∩ S.macs).card ≤ k := by
  rw [← card_union_of_disjoint (hS.cones_macs.mono inter_subset_right inter_subset_right)]
  exact (card_le_card (union_subset inter_subset_left inter_subset_left)).trans hI.2.2

/-- What the blocks hand on of `C_ij`'s gates, other blocks read. -/
theorem sum_card_handed_le (hS : S.Algorithm) {m : ℕ} (P : Partition d.D m) (e : Fin ni × Fin nj) :
    ∑ u, (handed d.D (S.entry e.1 e.2) (P.gates u)).card ≤
      ∑ u, (d.unregistered (P.gates u) ∩ S.entry e.1 e.2).card := by
  rw [← card_biUnion fun u _ v _ huv => ?_]
  · refine (card_le_card fun h hh => ?_).trans card_biUnion_le
    obtain ⟨u, -, hu⟩ := mem_biUnion.1 hh
    simp only [handed, mem_filter, mem_inter] at hu
    obtain ⟨⟨huI, hhe⟩, g, hge, hgu, hr⟩ := hu
    obtain ⟨v, hv⟩ := exists_mem_gates P (S.entry_computed hS e.1 e.2 hge)
    have hhv : h ∉ P.gates v := fun hhv => hgu (eq_of_mem_gates P hhv huI ▸ hv)
    exact mem_biUnion.2 ⟨v, mem_univ _, mem_inter.2
      ⟨mem_unregistered hv hhv hr (P.gates_subset_computed u huI), hhe⟩⟩
  · refine disjoint_left.2 fun h hu hv => huv ?_
    simp only [handed, mem_filter, mem_inter] at hu hv
    exact eq_of_mem_gates P hu.1.1 hv.1.1

/-- A block's reads of the entries' gates are its reads of the MACs' gates. -/
theorem sum_card_entry_le (hS : S.Algorithm) (U : Finset (Fin d.D.N)) :
    ∑ e : Fin ni × Fin nj, (U ∩ S.entry e.1 e.2).card ≤ (U ∩ S.macs).card := by
  rw [← card_biUnion fun e _ e' _ h => (S.entry_disjoint hS h).mono inter_subset_right inter_subset_right]
  exact card_le_card (biUnion_subset.2 fun e _ =>
    inter_subset_inter_left (S.entry_subset_macs e.1 e.2))

/-- **Partial sums**: `b_C Σ_u |π₁₃ T_u| ≤ E + Σ_u |U_u ∩ macs|`. -/
theorem partial_sums (hS : S.Algorithm) {c k bA bB bC m : ℕ} (hW : S.Passed c k bA bB bC) {P : Partition d.D m}
    (hP : P.IsCK d c k) :
    bC * ∑ u, ((S.whole (P.gates u)).image fun x => (x.1, x.2.2)).card ≤
      S.outBits + ∑ u, (d.unregistered (P.gates u) ∩ S.macs).card := by
  have hblock : ∀ u, bC * ((S.whole (P.gates u)).image fun x => (x.1, x.2.2)).card ≤
      ∑ e : Fin ni × Fin nj,
        ((handed d.D (S.entry e.1 e.2) (P.gates u)).card + (P.gates u ∩ S.out e.1 e.2).card) := by
    intro u
    calc bC * ((S.whole (P.gates u)).image fun x => (x.1, x.2.2)).card
        = ∑ _e ∈ (S.whole (P.gates u)).image fun x => (x.1, x.2.2), bC := by
          rw [sum_const, smul_eq_mul, mul_comm]
      _ ≤ ∑ e ∈ (S.whole (P.gates u)).image fun x => (x.1, x.2.2),
            ((handed d.D (S.entry e.1 e.2) (P.gates u)).card + (P.gates u ∩ S.out e.1 e.2).card) := by
          refine sum_le_sum fun e he => ?_
          obtain ⟨⟨i, l, j⟩, hx, rfl⟩ := mem_image.1 he
          exact hW.partialSum _ (inferenceBlock_gates hP u) i l j (mem_filter.1 hx).2
      _ ≤ _ := sum_le_sum_of_subset (subset_univ _)
  calc bC * ∑ u, ((S.whole (P.gates u)).image fun x => (x.1, x.2.2)).card
      = ∑ u, bC * ((S.whole (P.gates u)).image fun x => (x.1, x.2.2)).card := mul_sum _ _ _
    _ ≤ ∑ u, ∑ e : Fin ni × Fin nj,
          ((handed d.D (S.entry e.1 e.2) (P.gates u)).card + (P.gates u ∩ S.out e.1 e.2).card) :=
        sum_le_sum fun u _ => hblock u
    _ = ∑ e : Fin ni × Fin nj, ∑ u, (P.gates u ∩ S.out e.1 e.2).card +
          ∑ e : Fin ni × Fin nj, ∑ u, (handed d.D (S.entry e.1 e.2) (P.gates u)).card := by
        rw [sum_comm, ← sum_add_distrib]
        exact sum_congr rfl fun e _ => by rw [sum_add_distrib, add_comm]
    _ ≤ S.outBits + ∑ e : Fin ni × Fin nj, ∑ u, (d.unregistered (P.gates u) ∩ S.entry e.1 e.2).card := by
        refine add_le_add (le_of_eq ?_) (sum_le_sum fun e _ => S.sum_card_handed_le hS P e)
        exact sum_congr rfl fun e _ =>
          sum_card_inter_gates P ((hS.out_entry e.1 e.2).trans (S.entry_computed hS e.1 e.2))
    _ ≤ S.outBits + ∑ u, (d.unregistered (P.gates u) ∩ S.macs).card := by
        rw [sum_comm]
        exact add_le_add le_rfl (sum_le_sum fun u _ => S.sum_card_entry_le hS _)

/-- A block's reads of the MACs' gates, MAC by MAC. -/
theorem sum_card_mac_le (hS : S.Algorithm) (U : Finset (Fin d.D.N)) :
    ∑ p : Fin ni × Fin nl × Fin nj, (U ∩ S.mac p.1 p.2.1 p.2.2).card ≤ (U ∩ S.macs).card := by
  rw [← card_biUnion fun p _ q _ h => (hS.mac_disjoint p q h).mono inter_subset_right inter_subset_right]
  exact card_le_card (biUnion_subset.2 fun p _ => inter_subset_inter_left (S.mac_subset_macs _ _ _))

/-- The blocks' whole MACs, MAC by MAC. -/
theorem sum_card_whole {m : ℕ} (P : Partition d.D m) :
    ∑ u, (S.whole (P.gates u)).card =
      ∑ p : Fin ni × Fin nl × Fin nj, (univ.filter fun u => S.mac p.1 p.2.1 p.2.2 ⊆ P.gates u).card := by
  simp only [Standard.whole, card_filter]
  exact sum_comm

/-- **A split MAC is read across**: a MAC that no block holds whole has a gate that some block reads. -/
theorem split_read (hS : S.Algorithm) {m : ℕ} (P : Partition d.D m) (i : Fin ni) (l : Fin nl) (j : Fin nj)
    (h : ∀ u, ¬ S.mac i l j ⊆ P.gates u) : ∃ v, (d.unregistered (P.gates v) ∩ S.mac i l j).Nonempty := by
  obtain ⟨g₀, hg₀⟩ := hS.mac_nonempty i l j
  obtain ⟨u, hu⟩ := exists_mem_gates P (hS.mac_computed i l j hg₀)
  have hX : P.gates u ∩ S.mac i l j ≠ S.mac i l j := fun heq => h u (by
    rw [← heq]
    exact inter_subset_left)
  obtain ⟨a, ha, g, hg, hr, hag⟩ := hS.mac_connected i l j _ inter_subset_right ⟨g₀, mem_inter.2 ⟨hu, hg₀⟩⟩ hX
  have hac := hS.mac_computed i l j ha
  by_cases hau : a ∈ P.gates u
  · have hgu : g ∉ P.gates u := fun hgu => (hag.1 (mem_inter.2 ⟨hau, ha⟩)) (mem_inter.2 ⟨hgu, hg⟩)
    obtain ⟨v, hv⟩ := exists_mem_gates P (hS.mac_computed i l j hg)
    have hav : a ∉ P.gates v := fun hav => hgu (eq_of_mem_gates P hav hau ▸ hv)
    exact ⟨v, a, mem_inter.2 ⟨mem_unregistered hv hav hr hac, ha⟩⟩
  · have hgu : g ∈ P.gates u := by
      by_contra hgu
      exact hau (mem_inter.1 (hag.2 fun hgX => hgu (mem_inter.1 hgX).1)).1
    exact ⟨u, a, mem_inter.2 ⟨mem_unregistered hgu hau hr hac, ha⟩⟩

/-- **Splits**: `N ≤ Σ_u |T_u| + Σ_u |U_u ∩ macs|`, each MAC whole in a block or read across a split. -/
theorem card_le_whole_add (hS : S.Algorithm) {m : ℕ} (P : Partition d.D m) :
    ni * nl * nj ≤ ∑ u, (S.whole (P.gates u)).card + ∑ u, (d.unregistered (P.gates u) ∩ S.macs).card := by
  have hp : ∀ p : Fin ni × Fin nl × Fin nj, 1 ≤ (univ.filter fun u => S.mac p.1 p.2.1 p.2.2 ⊆ P.gates u).card +
      ∑ u, (d.unregistered (P.gates u) ∩ S.mac p.1 p.2.1 p.2.2).card := by
    rintro ⟨i, l, j⟩
    dsimp only
    by_cases h : ∃ u, S.mac i l j ⊆ P.gates u
    · obtain ⟨u, hu⟩ := h
      exact le_add_right (card_pos.2 ⟨u, mem_filter.2 ⟨mem_univ _, hu⟩⟩)
    · obtain ⟨v, hv⟩ := S.split_read hS P i l j fun u hu => h ⟨u, hu⟩
      refine le_add_left ((card_pos.2 hv).trans_le ?_)
      exact single_le_sum (f := fun u => (d.unregistered (P.gates u) ∩ S.mac i l j).card)
        (fun _ _ => Nat.zero_le _) (mem_univ v)
  calc ni * nl * nj = ∑ _p : Fin ni × Fin nl × Fin nj, 1 := by
        rw [sum_const, card_univ, Fintype.card_prod, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin,
          Fintype.card_fin, smul_eq_mul, mul_one, mul_assoc]
    _ ≤ ∑ p : Fin ni × Fin nl × Fin nj, ((univ.filter fun u => S.mac p.1 p.2.1 p.2.2 ⊆ P.gates u).card +
          ∑ u, (d.unregistered (P.gates u) ∩ S.mac p.1 p.2.1 p.2.2).card) := sum_le_sum fun p _ => hp p
    _ = ∑ u, (S.whole (P.gates u)).card +
          ∑ u, ∑ p : Fin ni × Fin nl × Fin nj, (d.unregistered (P.gates u) ∩ S.mac p.1 p.2.1 p.2.2).card := by
        rw [sum_add_distrib, S.sum_card_whole P, sum_comm (s := univ) (t := univ)]
    _ ≤ _ := add_le_add le_rfl (sum_le_sum fun u _ => S.sum_card_mac_le hS _)

/-- With every MAC in one block, `N ≤ Σ_u |T_u|`. -/
theorem card_le_whole {m : ℕ} (P : Partition d.D m) (h : S.Unsplit P) :
    ni * nl * nj ≤ ∑ u, (S.whole (P.gates u)).card := by
  rw [S.sum_card_whole P]
  calc ni * nl * nj = ∑ _p : Fin ni × Fin nl × Fin nj, 1 := by
        rw [sum_const, card_univ, Fintype.card_prod, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin,
          Fintype.card_fin, smul_eq_mul, mul_one, mul_assoc]
    _ ≤ _ := sum_le_sum fun p _ => by
        obtain ⟨u, hu⟩ := h p.1 p.2.1 p.2.2
        exact card_pos.2 ⟨u, mem_filter.2 ⟨mem_univ _, hu⟩⟩

/-- The entries whose chain starts in `I`, their first MAC in `I`. -/
def starts (I : Finset (Fin d.D.N)) : Finset (Fin ni × Fin nj) :=
  univ.filter fun e => ∃ l : Fin nl, l.val = 0 ∧ S.mac e.1 l e.2 ⊆ I

/-- **Each chain starts once**: `Σ_u s_u ≤ ni nj`, an entry's first MAC lying in at most one block. -/
theorem sum_card_starts_le (hS : S.Algorithm) {m : ℕ} (P : Partition d.D m) :
    ∑ u, (S.starts (P.gates u)).card ≤ ni * nj := by
  have h1 : ∀ e : Fin ni × Fin nj, (univ.filter fun u => e ∈ S.starts (P.gates u)).card ≤ 1 := by
    intro e
    refine card_le_one.2 fun u hu v hv => ?_
    simp only [starts, mem_filter, mem_univ, true_and] at hu hv
    obtain ⟨l, hl, hlu⟩ := hu
    obtain ⟨l', hl', hlv⟩ := hv
    obtain rfl : l = l' := Fin.ext (hl.trans hl'.symm)
    obtain ⟨g, hg⟩ := hS.mac_nonempty e.1 l e.2
    exact eq_of_mem_gates P (hlu hg) (hlv hg)
  calc ∑ u, (S.starts (P.gates u)).card
      = ∑ e : Fin ni × Fin nj, (univ.filter fun u => e ∈ S.starts (P.gates u)).card := by
        simp only [starts, card_filter, mem_filter, mem_univ, true_and]
        exact sum_comm
    _ ≤ ∑ _e : Fin ni × Fin nj, 1 := sum_le_sum fun e _ => h1 e
    _ = ni * nj := by rw [sum_const, card_univ, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin, smul_eq_mul,
        mul_one]

/-- **A chain read into a block**: a block that holds a MAC of `C_ij` and doesn't start its chain reads `b_C` of
`C_ij`'s gates, the partial sum of the MAC before the first one it holds. -/
theorem le_card_entry (hS : S.Algorithm) {bC : ℕ} (hch : S.Chained bC) {m : ℕ} {P : Partition d.D m}
    (hP : S.Unsplit P) (u : Fin m) (e : Fin ni × Fin nj) (he : ∃ l, S.mac e.1 l e.2 ⊆ P.gates u)
    (hs : e ∉ S.starts (P.gates u)) : bC ≤ (d.unregistered (P.gates u) ∩ S.entry e.1 e.2).card := by
  have hne : (univ.filter fun l : Fin nl => S.mac e.1 l e.2 ⊆ P.gates u).Nonempty := by
    obtain ⟨l, hl⟩ := he
    exact ⟨l, mem_filter.2 ⟨mem_univ _, hl⟩⟩
  obtain ⟨l₁, hl₁m, hl₁⟩ : ∃ l₁, l₁ ∈ (univ.filter fun l : Fin nl => S.mac e.1 l e.2 ⊆ P.gates u) ∧
      ∀ l ∈ (univ.filter fun l : Fin nl => S.mac e.1 l e.2 ⊆ P.gates u), l₁ ≤ l :=
    ⟨_, min'_mem _ hne, fun l hl => min'_le _ l hl⟩
  have hl₁u : S.mac e.1 l₁ e.2 ⊆ P.gates u := (mem_filter.1 hl₁m).2
  have h0 : l₁.val ≠ 0 := fun h0 => hs (mem_filter.2 ⟨mem_univ _, l₁, h0, hl₁u⟩)
  have hlt : l₁.val - 1 < nl := by omega
  have hnot : ¬ S.mac e.1 ⟨l₁.val - 1, hlt⟩ e.2 ⊆ P.gates u := fun h => by
    have : l₁.val ≤ l₁.val - 1 := Fin.le_def.1 (hl₁ _ (mem_filter.2 ⟨mem_univ _, h⟩))
    omega
  obtain ⟨v, hv⟩ := hP e.1 ⟨l₁.val - 1, hlt⟩ e.2
  refine (hch e.1 l₁ ⟨l₁.val - 1, hlt⟩ e.2 (Nat.sub_add_cancel (Nat.one_le_iff_ne_zero.2 h0))).trans
    (card_le_card fun h hh => ?_)
  obtain ⟨hh', g, hg, hr⟩ := mem_filter.1 hh
  have hhu : h ∉ P.gates u := fun hhu => hnot (by
    rw [eq_of_mem_gates P hhu (hv hh')]
    exact hv)
  exact mem_inter.2 ⟨mem_unregistered (hl₁u hg) hhu hr (hS.mac_computed _ _ _ hh'),
    mem_biUnion.2 ⟨_, mem_univ _, hh'⟩⟩

/-- **Partial sums along chains**: `b_C |π₁₃ T_u| ≤ |U_u ∩ macs| + b_C s_u`, each entry a block touches starting its
chain there or reading a partial sum. -/
theorem chain_sums (hS : S.Algorithm) {bC : ℕ} (hch : S.Chained bC) {m : ℕ} {P : Partition d.D m}
    (hP : S.Unsplit P) (u : Fin m) :
    bC * ((S.whole (P.gates u)).image fun x => (x.1, x.2.2)).card ≤
      (d.unregistered (P.gates u) ∩ S.macs).card + bC * (S.starts (P.gates u)).card := by
  have hZ : ((S.whole (P.gates u)).image fun x => (x.1, x.2.2)).card ≤ (S.starts (P.gates u)).card +
      (((S.whole (P.gates u)).image fun x => (x.1, x.2.2)).filter fun e => e ∉ S.starts (P.gates u)).card := by
    rw [← card_filter_add_card_filter_not (s := (S.whole (P.gates u)).image fun x => (x.1, x.2.2))
      (fun e => e ∈ S.starts (P.gates u))]
    exact add_le_add (card_le_card fun e he => (mem_filter.1 he).2) le_rfl
  have hread : bC * (((S.whole (P.gates u)).image fun x => (x.1, x.2.2)).filter
      fun e => e ∉ S.starts (P.gates u)).card ≤ (d.unregistered (P.gates u) ∩ S.macs).card := by
    calc bC * (((S.whole (P.gates u)).image fun x => (x.1, x.2.2)).filter fun e => e ∉ S.starts (P.gates u)).card
        = ∑ _e ∈ ((S.whole (P.gates u)).image fun x => (x.1, x.2.2)).filter
            (fun e => e ∉ S.starts (P.gates u)), bC := by rw [sum_const, smul_eq_mul, mul_comm]
      _ ≤ ∑ e ∈ ((S.whole (P.gates u)).image fun x => (x.1, x.2.2)).filter (fun e => e ∉ S.starts (P.gates u)),
            (d.unregistered (P.gates u) ∩ S.entry e.1 e.2).card := by
          refine sum_le_sum fun e he => ?_
          obtain ⟨hez, hes⟩ := mem_filter.1 he
          obtain ⟨⟨i, l, j⟩, hx, rfl⟩ := mem_image.1 hez
          exact S.le_card_entry hS hch hP u (i, j) ⟨l, (mem_filter.1 hx).2⟩ hes
      _ ≤ ∑ e : Fin ni × Fin nj, (d.unregistered (P.gates u) ∩ S.entry e.1 e.2).card :=
          sum_le_sum_of_subset (subset_univ _)
      _ ≤ _ := S.sum_card_entry_le hS _
  calc bC * ((S.whole (P.gates u)).image fun x => (x.1, x.2.2)).card
      ≤ bC * ((S.starts (P.gates u)).card +
          (((S.whole (P.gates u)).image fun x => (x.1, x.2.2)).filter fun e => e ∉ S.starts (P.gates u)).card) :=
        Nat.mul_le_mul_left _ hZ
    _ = bC * (S.starts (P.gates u)).card +
          bC * (((S.whole (P.gates u)).image fun x => (x.1, x.2.2)).filter fun e => e ∉ S.starts (P.gates u)).card :=
        mul_add _ _ _
    _ ≤ _ := by
        rw [add_comm]
        exact add_le_add hread le_rfl

/-- There are at most `|C|` products: their MACs are disjoint nonempty sets of computed gates. -/
theorem card_le_computed (hS : S.Algorithm) : ni * nl * nj ≤ d.D.computed.card := by
  calc ni * nl * nj = ∑ _p : Fin ni × Fin nl × Fin nj, 1 := by
        rw [sum_const, card_univ, Fintype.card_prod, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin,
          Fintype.card_fin, smul_eq_mul, mul_one, mul_assoc]
    _ ≤ ∑ p : Fin ni × Fin nl × Fin nj, (S.mac p.1 p.2.1 p.2.2).card :=
        sum_le_sum fun p _ => card_pos.2 (hS.mac_nonempty _ _ _)
    _ = S.macs.card := (card_biUnion fun p _ q _ h => hS.mac_disjoint p q h).symm
    _ ≤ _ := card_le_card (biUnion_subset.2 fun p _ => hS.mac_computed _ _ _)

end Standard

namespace SecurityProofs

open Standard

theorem Cap : Guarantees.Cap := by
  intro d ni nl nj S c k bA bB bC m P μ hS hW hP hA hB hC hμ
  have key := Verity.Combinatorics.sum_card_le (fun u => S.whole (P.gates u))
    (fun u => ((d.unregistered (P.gates u) ∩ S.cones).card : ℝ)) (fun _ => 0)
    (fun u => ((d.unregistered (P.gates u) ∩ S.macs).card : ℝ)) (k := k) (bA := bA) (bB := bB) (bC := bC)
    (s := 1) (E := S.outBits) (by exact_mod_cast hA) (by exact_mod_cast hB) (by exact_mod_cast hC) hμ
    (fun _ => Nat.cast_nonneg _) (fun _ => le_rfl) (fun _ => Nat.cast_nonneg _)
    (fun u => by
      have h := S.mul_card_le_cones hS hW (inferenceBlock_gates hP u)
      simp only [add_zero]
      exact_mod_cast h)
    (fun u => by exact_mod_cast S.card_add_card_le hS (inferenceBlock_gates hP u))
    (by exact_mod_cast S.partial_sums hS hW hP)
  simp only [add_zero, one_mul, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul] at key
  have hN : ((ni * nl * nj : ℕ) : ℝ) ≤ ∑ u, ((S.whole (P.gates u)).card : ℝ) +
      ∑ u, ((d.unregistered (P.gates u) ∩ S.macs).card : ℝ) := by
    exact_mod_cast S.card_le_whole_add hS P
  exact hN.trans key

theorem WordCap : Guarantees.WordCap := by
  intro d ni nl nj S c k bA bB bC m P hS hW hP hwhole hk hA hB hC
  have key := Verity.Combinatorics.sum_card_le_cap (fun u => S.whole (P.gates u))
    (fun u => ((d.unregistered (P.gates u) ∩ S.cones).card : ℝ))
    (fun u => ((d.unregistered (P.gates u) ∩ S.macs).card : ℝ)) (k := k) (bA := bA) (bB := bB) (bC := bC)
    (E := S.outBits) (by exact_mod_cast hk) (by exact_mod_cast hA) (by exact_mod_cast hB) (by exact_mod_cast hC)
    (fun _ => Nat.cast_nonneg _) (fun _ => Nat.cast_nonneg _)
    (fun u => by exact_mod_cast S.mul_card_le_cones hS hW (inferenceBlock_gates hP u))
    (fun u => by exact_mod_cast S.card_add_card_le hS (inferenceBlock_gates hP u))
    (by exact_mod_cast S.partial_sums hS hW hP)
  rw [Fintype.card_fin] at key
  have hN : ((ni * nl * nj : ℕ) : ℝ) ≤ ∑ u, ((S.whole (P.gates u)).card : ℝ) := by
    exact_mod_cast S.card_le_whole P hwhole
  exact hN.trans key

theorem ChainCap : Guarantees.ChainCap := by
  intro d ni nl nj S c k bA bB bC m P hS hW hch hP hwhole hA hB hC
  have key := Verity.Combinatorics.sum_card_le_chain (fun u => S.whole (P.gates u))
    (fun u => ((d.unregistered (P.gates u) ∩ S.cones).card : ℝ))
    (fun u => ((d.unregistered (P.gates u) ∩ S.macs).card : ℝ)) (fun u => ((S.starts (P.gates u)).card : ℝ))
    (k := k) (bA := bA) (bB := bB) (bC := bC) (O := ((ni * nj : ℕ) : ℝ)) (Nat.cast_nonneg _)
    (by exact_mod_cast hA) (by exact_mod_cast hB) (by exact_mod_cast hC) (fun _ => Nat.cast_nonneg _)
    (fun _ => Nat.cast_nonneg _)
    (fun u => by exact_mod_cast S.mul_card_le_cones hS hW (inferenceBlock_gates hP u))
    (fun u => by exact_mod_cast S.card_add_card_le hS (inferenceBlock_gates hP u))
    (fun u => by exact_mod_cast S.chain_sums hS hch hwhole u)
    (by exact_mod_cast S.sum_card_starts_le hS P)
  rw [Fintype.card_fin] at key
  have hN : ((ni * nl * nj : ℕ) : ℝ) ≤ ∑ u, ((S.whole (P.gates u)).card : ℝ) := by
    exact_mod_cast S.card_le_whole P hwhole
  exact hN.trans key

theorem Bound : Guarantees.Bound := by
  intro d ni nl nj S c k bA bB bC μ hS hW hA hB hC hμ m P hP
  have hcap := Cap d ni nl nj S c k bA bB bC m P μ hS hW hP hA hB hC hμ
  unfold Standard.bound
  have hK0 : 0 ≤ capAt k bA bB bC μ := by
    unfold capAt
    exact le_max_of_le_right (mul_nonneg (by linarith) (Nat.cast_nonneg _))
  have hNC : ((ni * nl * nj : ℕ) : ℝ) ≤ d.D.computed.card := by exact_mod_cast S.card_le_computed hS
  have hE : 0 ≤ μ * S.outBits := mul_nonneg hμ.le (Nat.cast_nonneg _)
  rcases le_or_gt (((ni * nl * nj : ℕ) : ℝ) - μ * S.outBits) 0 with hW0 | hW0
  · rw [ENNReal.ofReal_of_nonpos hW0, ENNReal.zero_div, ENNReal.zero_div]
    exact zero_le
  have hKpos : 0 < capAt k bA bB bC μ := by
    by_contra hK
    have hK' : capAt k bA bB bC μ = 0 := le_antisymm (not_lt.1 hK) hK0
    rw [hK', mul_zero, zero_add] at hcap
    linarith
  have hCpos : (0 : ℝ) < d.D.computed.card := by linarith
  rcases Nat.eq_zero_or_pos c with rfl | hc
  · rw [Nat.cast_zero, ENNReal.div_zero (ENNReal.ofReal_pos.2 hKpos).ne', ENNReal.div_top]
    exact zero_le
  have hcR : (0 : ℝ) < c := by exact_mod_cast hc
  rw [partitionSlowdown, ← ENNReal.ofReal_natCast d.D.computed.card, ← ENNReal.ofReal_natCast c,
    ← ENNReal.ofReal_natCast (m * c), ← ENNReal.ofReal_div_of_pos hCpos, ← ENNReal.ofReal_div_of_pos hcR,
    ← ENNReal.ofReal_div_of_pos (div_pos hKpos hcR), ← ENNReal.ofReal_div_of_pos hCpos]
  refine ENNReal.ofReal_le_ofReal ?_
  rw [div_le_div_iff₀ (div_pos hKpos hcR) hCpos, div_mul_cancel₀ _ hCpos.ne']
  calc ((ni * nl * nj : ℕ) : ℝ) - μ * S.outBits ≤ m * capAt k bA bB bC μ := by linarith
    _ = ((m * c : ℕ) : ℝ) * (capAt k bA bB bC μ / c) := by
        rw [Nat.cast_mul, mul_assoc, mul_div_cancel₀ _ hcR.ne']

open Nci.Formalism.Assumptions in
open scoped NNReal ENNReal in
example (d : Declaration) (L : Type) [Fintype L] (A : Annotation d L) (c k : ℕ) (E : Type)
    (e : Finset (Fin d.D.N) → L → E) (𝒲 : Finset (Finset (Fin d.D.N))) (X : Finset (Fin d.D.N)) (capX : ℝ≥0)
    (κ : Type) (B : κ → Finset (Fin d.D.N)) (cap : κ → ℝ≥0) {ni nl nj : ℕ} (S : Standard d ni nl nj)
    (bA bB bC : ℕ) (μ : ℝ) (m : ℕ) (P : Partition d.D m) (hA : A.Valid) (hcut : CutCost A c k e 𝒲)
    (hX : A.TokenCapAt c k e 𝒲 X capX) (hcaps : Caps d c k B cap) (hS : S.Algorithm)
    (hW : S.Passed c k bA bB bC) (h0A : 0 < bA) (h0B : 0 < bB) (h0C : 0 < bC) (hμ : 0 < μ) (hP : P.IsCK d c k) :
    max (kindBound d c X capX) (max (⨆ K, kindBound d c (B K) (cap K)) (⨆ _ : Unit, S.bound c k bA bB bC μ)) ≤
      partitionSlowdown d m c :=
  Nci.Formalism.SecurityProofs.Slowdown d L A c k E e 𝒲 X capX κ B cap Unit (fun _ => S.bound c k bA bB bC μ) m P
    hA hcut hX hcaps (fun _ => Bound d ni nl nj S c k bA bB bC μ hS hW h0A h0B h0C hμ) hP

end SecurityProofs

end Nci.Products
