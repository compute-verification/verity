import Definitions.Nci.Guarantees.Products.NoCell
import Proofs.Nci.Products.Cap

/-!
# Products of two fresh operands without the word level: the cap and its bound on `ρ`

For a partition `P` into `m` inference blocks, block `u` with gates `I_u` and reads `U_u = in(I_u) \ θ`, cells of at
most `w` MACs (`Standard.Cells`), `T_u` the MACs of the cells `I_u` holds whole (`Standard.cellWhole`), and the sets
`start_ij` that partial sums received at their width (`Standard.Received`) give:

1. **Splits** (`card_le_cellWhole_add`): `N ≤ Σ_u |T_u| + w Σ_u |U_u ∩ macs|`. A cell no block holds whole has a gate
   some block reads across the split (`cell_split_read`), and a block's reads of the MACs' gates lie in cells of at most
   `w` MACs each (`sum_card_cell_le`). Each block reads at most `k` of the MACs' gates (`card_add_card_le`), so
   `Σ_u |U_u ∩ macs| ≤ m k`.
2. **Operands** (`mul_card_le_cones_cell`): `b_A |π₁₂ T_u| + b_B |π₂₃ T_u| ≤ |U_u ∩ cones|`, each MAC of a whole cell
   being whole.
3. **Partial sums received** (`received_sums`): `b_C |π₁₃ T_u| ≤ |U_u ∩ macs| + Σ_ij |I_u ∩ start_ij|`, each entry a
   block touches through a whole cell paying `b_C` bits of reads of its gates less the start's gates the block holds.
4. **Each start held once** (`sum_sum_card_inter_le`): `Σ_u Σ_ij |I_u ∩ start_ij| ≤ ni nj b_C`, the blocks being
   disjoint and each start at most `b_C` gates.

With `s_u = Σ_ij |I_u ∩ start_ij| / b_C`, facts 3 and 4 are the cap along chains' (`Verity.Combinatorics.sum_card_le_chain`)
per-block partial sums and its `Σ_u s_u ≤ ni nj`, so `Σ_u |T_u| ≤ m α + β √(m ni nj)`, and fact 1 adds `m w k`
(`NoCellCap`). `NoCellBound` inverts the cap: `m Γ + B √m ≥ N` gives `m ≥ chainBlocks N Γ B` (`chainBlocks_le`).
-/

namespace Nci.Products

open Finset
open Verity (Declaration Partition)
open Nci.Formalism
open scoped ENNReal

variable {d : Declaration}

/-- A partition's blocks hold at most all of a set's gates. -/
theorem sum_card_inter_gates_le {m : ℕ} (P : Partition d.D m) (X : Finset (Fin d.D.N)) :
    ∑ u, (P.gates u ∩ X).card ≤ X.card :=
  calc ∑ u, (P.gates u ∩ X).card = ∑ u, (P.gates u ∩ (d.D.computed ∩ X)).card :=
        sum_congr rfl fun u _ => by rw [← inter_assoc, inter_eq_left.2 (P.gates_subset_computed u)]
    _ = (d.D.computed ∩ X).card := sum_card_inter_gates P inter_subset_left
    _ ≤ X.card := card_le_card inter_subset_right

/-- **Each start held once**: sets of at most `b_C` gates, one an entry, have at most `ni nj b_C` of their gates in a
partition's blocks. -/
theorem sum_sum_card_inter_le {ni nj bC : ℕ} {start : Fin ni → Fin nj → Finset (Fin d.D.N)}
    (hcard : ∀ i j, (start i j).card ≤ bC) {m : ℕ} (P : Partition d.D m) :
    ∑ u, ∑ e : Fin ni × Fin nj, (P.gates u ∩ start e.1 e.2).card ≤ ni * nj * bC := by
  rw [sum_comm]
  calc ∑ e : Fin ni × Fin nj, ∑ u, (P.gates u ∩ start e.1 e.2).card
      ≤ ∑ _e : Fin ni × Fin nj, bC := sum_le_sum fun e _ => (sum_card_inter_gates_le P _).trans (hcard e.1 e.2)
    _ = ni * nj * bC := by
        rw [sum_const, card_univ, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin, smul_eq_mul]

namespace Standard

variable {ni nl nj : ℕ} (S : Standard d ni nl nj) (cell : Fin ni → Fin nl → Fin nj → ℕ)

/-- The MACs whose cell lies whole in `I`, a block's `T`. -/
def cellWhole (I : Finset (Fin d.D.N)) : Finset (Fin ni × Fin nl × Fin nj) :=
  univ.filter fun p => S.cellGates cell (cell p.1 p.2.1 p.2.2) ⊆ I

theorem mac_subset_cellGates (i : Fin ni) (l : Fin nl) (j : Fin nj) : S.mac i l j ⊆ S.cellGates cell (cell i l j) :=
  fun _ hg => mem_biUnion.2 ⟨(i, l, j), mem_filter.2 ⟨mem_univ _, rfl⟩, hg⟩

/-- A MAC of a whole cell is whole. -/
theorem cellWhole_subset_whole (I : Finset (Fin d.D.N)) : S.cellWhole cell I ⊆ S.whole I := fun p hp =>
  mem_filter.2 ⟨mem_univ _, (S.mac_subset_cellGates cell p.1 p.2.1 p.2.2).trans (mem_filter.1 hp).2⟩

theorem cellGates_subset_macs (z : ℕ) : S.cellGates cell z ⊆ S.macs :=
  biUnion_subset.2 fun _ _ => S.mac_subset_macs _ _ _

theorem cellGates_computed (hS : S.Wiring) (z : ℕ) : S.cellGates cell z ⊆ d.D.computed :=
  biUnion_subset.2 fun _ _ => hS.mac_computed _ _ _

theorem cellGates_disjoint (hS : S.Wiring) {z z' : ℕ} (h : z ≠ z') :
    Disjoint (S.cellGates cell z) (S.cellGates cell z') :=
  (disjoint_biUnion_left _ _ _).2 fun p hp => (disjoint_biUnion_right _ _ _).2 fun q hq =>
    hS.mac_disjoint p q fun hpq => h (by
      subst hpq
      exact (mem_filter.1 hp).2.symm.trans (mem_filter.1 hq).2)

/-- **Each MAC its own cell**: the standard algorithm, with `cell` putting each MAC in a cell of its own, has cells of
one MAC each. -/
theorem cells_one (hS : S.Algorithm)
    (hcell : ∀ p q : Fin ni × Fin nl × Fin nj, cell p.1 p.2.1 p.2.2 = cell q.1 q.2.1 q.2.2 → p = q) :
    S.Cells cell 1 where
  card_le z := card_le_one.2 fun p hp q hq => hcell p q ((mem_filter.1 hp).2.trans (mem_filter.1 hq).2.symm)
  connected z X hX hne hXne := by
    obtain ⟨g, hg⟩ := hne
    obtain ⟨p, hp, -⟩ := mem_biUnion.1 (hX hg)
    have hz : S.cellGates cell z = S.mac p.1 p.2.1 p.2.2 := by
      refine Subset.antisymm (biUnion_subset.2 fun q hq => ?_) fun h hh => mem_biUnion.2 ⟨p, hp, hh⟩
      obtain rfl := hcell q p ((mem_filter.1 hq).2.trans (mem_filter.1 hp).2.symm)
      exact fun _ h => h
    rw [hz] at hX hXne ⊢
    exact hS.mac_connected p.1 p.2.1 p.2.2 X hX ⟨g, hg⟩ hXne

/-- **A split cell is read across**: a cell that no block holds whole has a gate that some block reads. -/
theorem cell_split_read (hS : S.Wiring) {w : ℕ} (hc : S.Cells cell w) {m : ℕ} (P : Partition d.D m) (i : Fin ni)
    (l : Fin nl) (j : Fin nj) (h : ∀ u, ¬ S.cellGates cell (cell i l j) ⊆ P.gates u) :
    ∃ v, (d.unregistered (P.gates v) ∩ S.cellGates cell (cell i l j)).Nonempty := by
  obtain ⟨g, hg⟩ := hS.mac_nonempty i l j
  exact split_read_of_connected P (S.cellGates_computed cell hS _) ⟨g, S.mac_subset_cellGates cell i l j hg⟩
    (hc.connected (cell i l j)) h

/-- A block's reads of the MACs' cells, MAC by MAC, are at most `w` times its reads of the MACs' gates. -/
theorem sum_card_cell_le (hS : S.Wiring) {w : ℕ} (hc : S.Cells cell w) (U : Finset (Fin d.D.N)) :
    ∑ p : Fin ni × Fin nl × Fin nj, (U ∩ S.cellGates cell (cell p.1 p.2.1 p.2.2)).card ≤ w * (U ∩ S.macs).card := by
  let g : Fin ni × Fin nl × Fin nj → ℕ := fun p => cell p.1 p.2.1 p.2.2
  calc ∑ p : Fin ni × Fin nl × Fin nj, (U ∩ S.cellGates cell (cell p.1 p.2.1 p.2.2)).card
      = ∑ z ∈ univ.image g, (univ.filter fun p => g p = z).card • (U ∩ S.cellGates cell z).card :=
        sum_comp (fun z => (U ∩ S.cellGates cell z).card) g
    _ ≤ ∑ z ∈ univ.image g, w * (U ∩ S.cellGates cell z).card :=
        sum_le_sum fun z _ => by
          rw [smul_eq_mul]
          exact Nat.mul_le_mul_right _ (hc.card_le z)
    _ = w * ((univ.image g).biUnion fun z => U ∩ S.cellGates cell z).card := by
        rw [← mul_sum, card_biUnion fun z _ z' _ h =>
          (S.cellGates_disjoint cell hS h).mono inter_subset_right inter_subset_right]
    _ ≤ w * (U ∩ S.macs).card :=
        Nat.mul_le_mul_left _ (card_le_card (biUnion_subset.2 fun z _ =>
          inter_subset_inter_left (S.cellGates_subset_macs cell z)))

/-- The blocks' MACs of whole cells, MAC by MAC. -/
theorem sum_card_cellWhole {m : ℕ} (P : Partition d.D m) :
    ∑ u, (S.cellWhole cell (P.gates u)).card = ∑ p : Fin ni × Fin nl × Fin nj,
      (univ.filter fun u => S.cellGates cell (cell p.1 p.2.1 p.2.2) ⊆ P.gates u).card := by
  simp only [Standard.cellWhole, card_filter]
  exact sum_comm

/-- **Splits**: `N ≤ Σ_u |T_u| + w Σ_u |U_u ∩ macs|`, each MAC's cell whole in a block or read across a split. -/
theorem card_le_cellWhole_add (hS : S.Wiring) {w : ℕ} (hc : S.Cells cell w) {m : ℕ} (P : Partition d.D m) :
    ni * nl * nj ≤
      ∑ u, (S.cellWhole cell (P.gates u)).card + w * ∑ u, (d.unregistered (P.gates u) ∩ S.macs).card := by
  have hp : ∀ p : Fin ni × Fin nl × Fin nj,
      1 ≤ (univ.filter fun u => S.cellGates cell (cell p.1 p.2.1 p.2.2) ⊆ P.gates u).card +
        ∑ u, (d.unregistered (P.gates u) ∩ S.cellGates cell (cell p.1 p.2.1 p.2.2)).card := by
    rintro ⟨i, l, j⟩
    dsimp only
    by_cases h : ∃ u, S.cellGates cell (cell i l j) ⊆ P.gates u
    · obtain ⟨u, hu⟩ := h
      exact le_add_right (card_pos.2 ⟨u, mem_filter.2 ⟨mem_univ _, hu⟩⟩)
    · obtain ⟨v, hv⟩ := S.cell_split_read cell hS hc P i l j fun u hu => h ⟨u, hu⟩
      refine le_add_left ((card_pos.2 hv).trans_le ?_)
      exact single_le_sum (f := fun u => (d.unregistered (P.gates u) ∩ S.cellGates cell (cell i l j)).card)
        (fun _ _ => Nat.zero_le _) (mem_univ v)
  calc ni * nl * nj = ∑ _p : Fin ni × Fin nl × Fin nj, 1 := by
        rw [sum_const, card_univ, Fintype.card_prod, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin,
          Fintype.card_fin, smul_eq_mul, mul_one, mul_assoc]
    _ ≤ ∑ p : Fin ni × Fin nl × Fin nj,
          ((univ.filter fun u => S.cellGates cell (cell p.1 p.2.1 p.2.2) ⊆ P.gates u).card +
            ∑ u, (d.unregistered (P.gates u) ∩ S.cellGates cell (cell p.1 p.2.1 p.2.2)).card) :=
        sum_le_sum fun p _ => hp p
    _ = ∑ u, (S.cellWhole cell (P.gates u)).card + ∑ u, ∑ p : Fin ni × Fin nl × Fin nj,
          (d.unregistered (P.gates u) ∩ S.cellGates cell (cell p.1 p.2.1 p.2.2)).card := by
        rw [sum_add_distrib, S.sum_card_cellWhole cell P, sum_comm (s := univ) (t := univ)]
    _ ≤ _ := by
        rw [mul_sum]
        exact add_le_add le_rfl (sum_le_sum fun u _ => S.sum_card_cell_le cell hS hc _)

/-- **Operands**: `b_A |π₁₂ T| + b_B |π₂₃ T| ≤ |U ∩ cones|` for the MACs `T` of a block's whole cells. -/
theorem mul_card_le_cones_cell (hS : S.Wiring) {c k bA bB bC : ℕ} (hW : S.Passed c k bA bB bC)
    {I : Finset (Fin d.D.N)} (hI : InferenceBlock d c k I) :
    bA * ((S.cellWhole cell I).image fun x => (x.1, x.2.1)).card +
        bB * ((S.cellWhole cell I).image fun x => (x.2.1, x.2.2)).card ≤
      (d.unregistered I ∩ S.cones).card :=
  (add_le_add (Nat.mul_le_mul_left _ (card_le_card (image_subset_image (S.cellWhole_subset_whole cell I))))
    (Nat.mul_le_mul_left _ (card_le_card (image_subset_image (S.cellWhole_subset_whole cell I))))).trans
    (S.mul_card_le_cones hS hW hI)

/-- **Partial sums received**: `b_C |π₁₃ T_u| ≤ |U_u ∩ macs| + Σ_ij |I_u ∩ start_ij|`, each entry a block touches
through a whole cell paying `b_C` bits of reads of its gates, less its start's gates the block holds. -/
theorem received_sums (hS : S.Wiring) {c k bC : ℕ} {start : Fin ni → Fin nj → Finset (Fin d.D.N)}
    (hread : ∀ I, InferenceBlock d c k I → ∀ i l j, S.cellGates cell (cell i l j) ⊆ I →
      bC ≤ (d.unregistered I ∩ S.entry i j).card + (I ∩ start i j).card)
    {m : ℕ} {P : Partition d.D m} (hP : P.IsCK d c k) (u : Fin m) :
    bC * ((S.cellWhole cell (P.gates u)).image fun x => (x.1, x.2.2)).card ≤
      (d.unregistered (P.gates u) ∩ S.macs).card + ∑ e : Fin ni × Fin nj, (P.gates u ∩ start e.1 e.2).card :=
  calc bC * ((S.cellWhole cell (P.gates u)).image fun x => (x.1, x.2.2)).card
      = ∑ _e ∈ (S.cellWhole cell (P.gates u)).image fun x => (x.1, x.2.2), bC := by
        rw [sum_const, smul_eq_mul, mul_comm]
    _ ≤ ∑ e ∈ (S.cellWhole cell (P.gates u)).image fun x => (x.1, x.2.2),
          ((d.unregistered (P.gates u) ∩ S.entry e.1 e.2).card + (P.gates u ∩ start e.1 e.2).card) := by
        refine sum_le_sum fun e he => ?_
        obtain ⟨⟨i, l, j⟩, hx, rfl⟩ := mem_image.1 he
        exact hread _ (inferenceBlock_gates hP u) i l j (mem_filter.1 hx).2
    _ ≤ ∑ e : Fin ni × Fin nj,
          ((d.unregistered (P.gates u) ∩ S.entry e.1 e.2).card + (P.gates u ∩ start e.1 e.2).card) :=
        sum_le_sum_of_subset (subset_univ _)
    _ = ∑ e : Fin ni × Fin nj, (d.unregistered (P.gates u) ∩ S.entry e.1 e.2).card +
          ∑ e : Fin ni × Fin nj, (P.gates u ∩ start e.1 e.2).card := sum_add_distrib
    _ ≤ _ := add_le_add (S.sum_card_entry_le hS _) le_rfl

end Standard

/-- **The fewest blocks**: `N ≤ m Γ + B √m` with `N, Γ, B, m ≥ 0` gives `chainBlocks N Γ B ≤ m`. -/
theorem chainBlocks_le {N Γ B m : ℝ} (hN : 0 ≤ N) (hΓ : 0 ≤ Γ) (hB : 0 ≤ B) (hm : 0 ≤ m)
    (h : N ≤ m * Γ + B * √m) : chainBlocks N Γ B ≤ m := by
  unfold chainBlocks
  have hR0 : 0 ≤ √(B ^ 2 + 4 * Γ * N) := Real.sqrt_nonneg _
  have hs0 : 0 ≤ √m := Real.sqrt_nonneg _
  have hs2 : √m ^ 2 = m := Real.sq_sqrt hm
  rcases eq_or_lt_of_le (add_nonneg hR0 hB) with hD | hD
  · rw [← hD, div_zero, zero_pow two_ne_zero]
    exact hm
  have hs : 2 * N / (√(B ^ 2 + 4 * Γ * N) + B) ≤ √m := by
    rw [div_le_iff₀ hD]
    have hN' : N ≤ Γ * √m ^ 2 + B * √m := by rw [hs2]; linarith
    have hR : √(B ^ 2 + 4 * Γ * N) ≤ 2 * Γ * √m + B := by
      rw [← Real.sqrt_sq (by positivity : 0 ≤ 2 * Γ * √m + B)]
      exact Real.sqrt_le_sqrt (by nlinarith)
    have hRB : B ≤ √(B ^ 2 + 4 * Γ * N) :=
      calc B = √(B ^ 2) := (Real.sqrt_sq hB).symm
        _ ≤ _ := Real.sqrt_le_sqrt (by nlinarith)
    rcases eq_or_lt_of_le hΓ with hΓ0 | hΓ0
    · subst hΓ0
      nlinarith [mul_le_mul_of_nonneg_left hRB hs0]
    · nlinarith [mul_le_mul_of_nonneg_right hR hD.le, Real.sq_sqrt (by positivity : 0 ≤ B ^ 2 + 4 * Γ * N)]
  calc (2 * N / (√(B ^ 2 + 4 * Γ * N) + B)) ^ 2 ≤ √m ^ 2 :=
        pow_le_pow_left₀ (div_nonneg (by linarith) hD.le) hs 2
    _ = m := hs2

namespace SecurityProofs

open Standard

theorem NoCellCap : Guarantees.NoCellCap := by
  intro d ni nl nj S cell w c k bA bB bC m P hS hc hW hR hP hA hB hC
  obtain ⟨start, hcard, hread⟩ := hR
  have hCpos : (0 : ℝ) < bC := by exact_mod_cast hC
  have key := Verity.Combinatorics.sum_card_le_chain (fun u => S.cellWhole cell (P.gates u))
    (fun u => ((d.unregistered (P.gates u) ∩ S.cones).card : ℝ))
    (fun u => ((d.unregistered (P.gates u) ∩ S.macs).card : ℝ))
    (fun u => ((∑ e : Fin ni × Fin nj, (P.gates u ∩ start e.1 e.2).card : ℕ) : ℝ) / bC)
    (k := k) (bA := bA) (bB := bB) (bC := bC) (O := ((ni * nj : ℕ) : ℝ)) (Nat.cast_nonneg _)
    (by exact_mod_cast hA) (by exact_mod_cast hB) hCpos (fun _ => Nat.cast_nonneg _)
    (fun _ => div_nonneg (Nat.cast_nonneg _) hCpos.le)
    (fun u => by exact_mod_cast S.mul_card_le_cones_cell cell hS hW (inferenceBlock_gates hP u))
    (fun u => by exact_mod_cast S.card_add_card_le hS (inferenceBlock_gates hP u))
    (fun u => by
      have h := S.received_sums cell hS hread hP u
      rw [mul_div_cancel₀ _ hCpos.ne']
      exact_mod_cast h)
    (by
      rw [← sum_div, div_le_iff₀ hCpos]
      exact_mod_cast sum_sum_card_inter_le hcard P)
  rw [Fintype.card_fin] at key
  have hsplit : (w : ℝ) * ∑ u, ((d.unregistered (P.gates u) ∩ S.macs).card : ℝ) ≤ m * (w * k) := by
    calc (w : ℝ) * ∑ u, ((d.unregistered (P.gates u) ∩ S.macs).card : ℝ) ≤ w * ∑ _u : Fin m, (k : ℝ) :=
          mul_le_mul_of_nonneg_left (sum_le_sum fun u _ => by
            exact_mod_cast (Nat.le_add_left _ _).trans (S.card_add_card_le hS (inferenceBlock_gates hP u)))
            (Nat.cast_nonneg _)
      _ = m * (w * k) := by rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
  have hN : ((ni * nl * nj : ℕ) : ℝ) ≤ ∑ u, ((S.cellWhole cell (P.gates u)).card : ℝ) +
      (w : ℝ) * ∑ u, ((d.unregistered (P.gates u) ∩ S.macs).card : ℝ) := by
    exact_mod_cast S.card_le_cellWhole_add cell hS hc P
  rw [mul_add]
  linarith

theorem NoCellBound : Guarantees.NoCellBound := by
  intro d ni nl nj S cell w c k bA bB bC hS hc hW hR hA hB hC m P hP
  have hcap := NoCellCap d ni nl nj S cell w c k bA bB bC m P hS hc hW hR hP hA hB hC
  have hα : 0 ≤ (k : ℝ) / 3 * √((k : ℝ) / 3) / √((bA : ℝ) * bB * bC) + w * k := by positivity
  have hβ : 0 ≤ (k : ℝ) / 2 / √((bA : ℝ) * bB) * √((ni * nj : ℕ) : ℝ) := by positivity
  have hm : chainBlocks ((ni * nl * nj : ℕ) : ℝ) ((k : ℝ) / 3 * √((k : ℝ) / 3) / √((bA : ℝ) * bB * bC) + w * k)
      ((k : ℝ) / 2 / √((bA : ℝ) * bB) * √((ni * nj : ℕ) : ℝ)) ≤ m := by
    refine chainBlocks_le (Nat.cast_nonneg _) hα hβ (Nat.cast_nonneg _) ?_
    rw [Real.sqrt_mul (Nat.cast_nonneg (α := ℝ) m) ((ni * nj : ℕ) : ℝ)] at hcap
    linarith
  unfold noCellBound partitionSlowdown
  refine ENNReal.div_le_div_right ?_ _
  rw [show ((m * c : ℕ) : ℝ≥0∞) = (m : ℝ≥0∞) * c from Nat.cast_mul m c, ← ENNReal.ofReal_natCast m]
  gcongr

end SecurityProofs

end Nci.Products
