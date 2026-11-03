import Definitions.Nci.Guarantees.Products.Whole
import Proofs.Nci.Products.NoCell

/-!
# Products of two fresh operands, made whole through chains: the class bound, the cap and its bound on `ρ`

For an inference block `I` with reads `U = in(I) \ θ`:

1. **Loomis–Whitney on whole chains** (`Standard.sq_mul_card_wholeOut_le`): `K² |wholeOut I| ≤ |hadA I| |hadB I|`, the
   triples `(i, l, j)` of the entries whose chain `I` holds projecting into the operand entries `I` has.
2. **Paid values** (`mul_card_le_card_inter`): distinct values are disjoint (`System.Values`), so `I` reads `b` bits of
   at most `|U| / b ≤ k / b` of them.
3. **Made values** (`System.card_le_sum_fanout`): a value made whole from an output entry `x` is one of the at most
   `A_{cls x, j}` values of class `j` that `x` makes (`System.Matched`).
4. **The class bound** (`SecurityProofs.ClassBound`), by induction on nesting depth `t` (`System.classBound_lt`): class
   `j`'s products less deep than `t + 1` have `K_j² Σ |wholeOut| ≤ c_j H²` by 1 and P-share, `H` their had values, each
   paid or made from an output entry less deep than `t` (P-width extended, P-match's depth), so
   `H ≤ (k/b) (1 + Σ_i A_ij μ_i)` by 2, 3 and the bound below `t`; the post-fixed point closes the step.
5. **The cap** (`SecurityProofs.WholeCap`): the operand entries of a block's whole MACs of product `q` have distinct
   values (`Standard.card_add_card_le_hadCones`), each paid or made, so `b |π₁₂ T| + b |π₂₃ T| ≤ |U ∩ cones| + e k`
   (`System.ops_le`) by 2, 3 and the class bound. With `|U ∩ cones| + |U ∩ macs| ≤ k`, the cap along chains
   (`Verity.Combinatorics.sum_card_le_chain`) runs at `k' = k (1 + e)`; the split MACs and the received partial sums
   are as in `NoCellCap`.
6. **The bound** (`SecurityProofs.WholeBound`): each product's cap inverted by `chainBlocks_le`, then the max over `G`.
-/

namespace Nci.Products

open Finset
open Verity (Declaration Partition)
open Nci.Formalism
open scoped ENNReal

variable {d : Declaration}

/-- **Paid values**: pairwise disjoint values, each with at least `b` gates of `U`, number at most `|U ∩ ⋃ V| / b`. -/
theorem mul_card_le_card_inter {α : Type*} [DecidableEq α] {V : Finset (Finset α)} {U : Finset α} {b : ℕ}
    (hd : ∀ v ∈ V, ∀ w ∈ V, v ≠ w → Disjoint v w) (hb : ∀ v ∈ V, b ≤ (U ∩ v).card) :
    b * V.card ≤ (U ∩ V.biUnion id).card :=
  calc b * V.card = ∑ _v ∈ V, b := by rw [sum_const, smul_eq_mul, mul_comm]
    _ ≤ ∑ v ∈ V, (U ∩ v).card := sum_le_sum hb
    _ = (V.biUnion fun v => U ∩ v).card :=
        (card_biUnion fun v hv w hw h => (hd v hv w hw h).mono inter_subset_right inter_subset_right).symm
    _ ≤ (U ∩ V.biUnion id).card :=
        card_le_card (biUnion_subset.2 fun v hv => inter_subset_inter_left (subset_biUnion_of_mem id hv))

namespace Standard

variable {ni nl nj : ℕ} (S : Standard d ni nl nj)

/-- A block's whole MACs read entries of `A` it has. -/
theorem image_whole_subset_hadA (hS : S.Algorithm) (I : Finset (Fin d.D.N)) :
    (S.whole I).image (fun x => (x.1, x.2.1)) ⊆ S.hadA I := fun e he => by
  obtain ⟨⟨i, l, j⟩, hx, rfl⟩ := mem_image.1 he
  refine mem_filter.2 ⟨mem_univ _, fun a ha => ?_⟩
  obtain ⟨g, hg, hr⟩ := hS.readsA i l j a ha
  exact mem_union_into ((mem_filter.1 hx).2 hg) hr

/-- A block's whole MACs read entries of `B` it has. -/
theorem image_whole_subset_hadB (hS : S.Algorithm) (I : Finset (Fin d.D.N)) :
    (S.whole I).image (fun x => (x.2.1, x.2.2)) ⊆ S.hadB I := fun e he => by
  obtain ⟨⟨i, l, j⟩, hx, rfl⟩ := mem_image.1 he
  refine mem_filter.2 ⟨mem_univ _, fun a ha => ?_⟩
  obtain ⟨g, hg, hr⟩ := hS.readsB i l j a ha
  exact mem_union_into ((mem_filter.1 hx).2 hg) hr

/-- **Loomis–Whitney on whole chains**: `nl² |wholeOut I| ≤ |hadA I| |hadB I|`. -/
theorem sq_mul_card_wholeOut_le (hS : S.Algorithm) (I : Finset (Fin d.D.N)) :
    nl ^ 2 * (S.wholeOut I).card ≤ (S.hadA I).card * (S.hadB I).card := by
  let V : Finset (Fin ni × Fin nl × Fin nj) :=
    (S.wholeOut I ×ˢ (univ : Finset (Fin nl))).image fun y => (y.1.1, y.2, y.1.2)
  have hV : V.card = (S.wholeOut I).card * nl := by
    rw [card_image_of_injective _ fun y y' h => ?_, card_product, card_univ, Fintype.card_fin]
    obtain ⟨⟨i, j⟩, l⟩ := y
    obtain ⟨⟨i', j'⟩, l'⟩ := y'
    simp only [Prod.mk.injEq] at h
    obtain ⟨rfl, rfl, rfl⟩ := h
    rfl
  have hentry : ∀ x ∈ V, S.mac x.1 x.2.1 x.2.2 ⊆ I := fun x hx => by
    obtain ⟨⟨⟨i, j⟩, l⟩, hy, rfl⟩ := mem_image.1 hx
    exact fun g hg => (mem_filter.1 (mem_product.1 hy).1).2 (mem_biUnion.2 ⟨l, mem_univ _, hg⟩)
  have h12 : V.image (fun x => (x.1, x.2.1)) ⊆ S.hadA I := fun e he => by
    obtain ⟨x, hx, rfl⟩ := mem_image.1 he
    refine mem_filter.2 ⟨mem_univ _, fun a ha => ?_⟩
    obtain ⟨g, hg, hr⟩ := hS.readsA x.1 x.2.1 x.2.2 a ha
    exact mem_union_into (hentry x hx hg) hr
  have h23 : V.image (fun x => (x.2.1, x.2.2)) ⊆ S.hadB I := fun e he => by
    obtain ⟨x, hx, rfl⟩ := mem_image.1 he
    refine mem_filter.2 ⟨mem_univ _, fun a ha => ?_⟩
    obtain ⟨g, hg, hr⟩ := hS.readsB x.1 x.2.1 x.2.2 a ha
    exact mem_union_into (hentry x hx hg) hr
  have h13 : V.image (fun x => (x.1, x.2.2)) ⊆ S.wholeOut I := fun e he => by
    obtain ⟨x, hx, rfl⟩ := mem_image.1 he
    obtain ⟨⟨⟨i, j⟩, l⟩, hy, rfl⟩ := mem_image.1 hx
    exact (mem_product.1 hy).1
  have hLW := Verity.Combinatorics.card_sq_le_proj V
  rw [hV] at hLW
  have hP : (V.image fun x => (x.1, x.2.1)).card * (V.image fun x => (x.2.1, x.2.2)).card *
      (V.image fun x => (x.1, x.2.2)).card ≤ (S.hadA I).card * (S.hadB I).card * (S.wholeOut I).card :=
    Nat.mul_le_mul (Nat.mul_le_mul (card_le_card h12) (card_le_card h23)) (card_le_card h13)
  rcases Nat.eq_zero_or_pos (S.wholeOut I).card with h0 | hpos
  · rw [h0, mul_zero]
    exact Nat.zero_le _
  · refine Nat.le_of_mul_le_mul_right ?_ hpos
    calc nl ^ 2 * (S.wholeOut I).card * (S.wholeOut I).card = ((S.wholeOut I).card * nl) ^ 2 := by ring
      _ ≤ _ := hLW.trans hP

/-- The values of the operand entries a block has are values of the product. -/
theorem hadCones_subset_values (I : Finset (Fin d.D.N)) : S.hadCones I ⊆ S.values :=
  union_subset_union (image_subset_image (subset_univ _)) (image_subset_image (subset_univ _))

/-- A value of the product is one of its cones. -/
theorem subset_cones_of_mem_values {v : Finset (Fin d.D.N)} (hv : v ∈ S.values) : v ⊆ S.cones := by
  rcases mem_union.1 hv with hv | hv
  · obtain ⟨e, -, rfl⟩ := mem_image.1 hv
    exact (subset_biUnion_of_mem (fun p : Fin ni × Fin nl => S.coneA p.1 p.2) (mem_univ e)).trans subset_union_left
  · obtain ⟨e, -, rfl⟩ := mem_image.1 hv
    exact (subset_biUnion_of_mem (fun p : Fin nl × Fin nj => S.coneB p.1 p.2) (mem_univ e)).trans subset_union_right

/-- **Distinct entries, distinct values**: entries `X` of `A` and `Y` of `B` that `I` has number at most its had
values, the cones being nonempty and pairwise disjoint. -/
theorem card_add_card_le_hadCones (hS : S.Algorithm) (hne : ∀ v ∈ S.values, v.Nonempty) {I : Finset (Fin d.D.N)}
    {X : Finset (Fin ni × Fin nl)} {Y : Finset (Fin nl × Fin nj)} (hX : X ⊆ S.hadA I) (hY : Y ⊆ S.hadB I) :
    X.card + Y.card ≤ (S.hadCones I).card := by
  have hneA : ∀ e : Fin ni × Fin nl, (S.coneA e.1 e.2).Nonempty := fun e =>
    hne _ (mem_union_left _ (mem_image_of_mem _ (mem_univ e)))
  have hneB : ∀ e : Fin nl × Fin nj, (S.coneB e.1 e.2).Nonempty := fun e =>
    hne _ (mem_union_right _ (mem_image_of_mem _ (mem_univ e)))
  have hiA : Set.InjOn (fun e : Fin ni × Fin nl => S.coneA e.1 e.2) X := fun e _ e' _ h => by
    by_contra hee
    have hd := hS.coneA_disjoint e e' hee
    rw [show S.coneA e.1 e.2 = S.coneA e'.1 e'.2 from h, disjoint_self, bot_eq_empty] at hd
    exact (hneA e').ne_empty hd
  have hiB : Set.InjOn (fun e : Fin nl × Fin nj => S.coneB e.1 e.2) Y := fun e _ e' _ h => by
    by_contra hee
    have hd := hS.coneB_disjoint e e' hee
    rw [show S.coneB e.1 e.2 = S.coneB e'.1 e'.2 from h, disjoint_self, bot_eq_empty] at hd
    exact (hneB e').ne_empty hd
  have hd : Disjoint (X.image fun e => S.coneA e.1 e.2) (Y.image fun e => S.coneB e.1 e.2) := by
    rw [disjoint_left]
    intro v hvA hvB
    obtain ⟨e, -, rfl⟩ := mem_image.1 hvA
    obtain ⟨e', -, he'⟩ := mem_image.1 hvB
    have h := hS.cones_disjoint e.1 e.2 e'.1 e'.2
    rw [he', disjoint_self, bot_eq_empty] at h
    exact (hneA e).ne_empty h
  calc X.card + Y.card = (X.image fun e => S.coneA e.1 e.2).card + (Y.image fun e => S.coneB e.1 e.2).card := by
        rw [card_image_of_injOn hiA, card_image_of_injOn hiB]
    _ = ((X.image fun e => S.coneA e.1 e.2) ∪ Y.image fun e => S.coneB e.1 e.2).card :=
        (card_union_of_disjoint hd).symm
    _ ≤ (S.hadCones I).card := card_le_card (union_subset_union (image_subset_image hX) (image_subset_image hY))

end Standard

namespace System

variable {ι κ : Type} [Fintype ι] [DecidableEq ι] (sys : System d ι κ)

omit [Fintype ι] [DecidableEq ι] in
/-- Two distinct values of the system are disjoint. -/
theorem disjoint_of_ne (hV : sys.Values) {p q : ι} {v w : Finset (Fin d.D.N)} (hv : v ∈ (sys.prod p).values)
    (hw : w ∈ (sys.prod q).values) (h : v ≠ w) : Disjoint v w :=
  (hV p q v hv w hw).2.resolve_left h

/-- **Made values**: values of class `j`, each made from an output entry in `X`, number at most
`Σ_{x ∈ X} A_{cls x, j}`. -/
theorem card_le_sum_fanout [DecidableEq κ] {A : κ → κ → ℝ} (hM : sys.Matched A) (j : κ)
    (X : Finset ((q : ι) × Fin (sys.ni q) × Fin (sys.nj q))) {V : Finset (Finset (Fin d.D.N))}
    (hV : ∀ v ∈ V, v ∈ sys.classValues j ∧ ∃ x ∈ sys.src v, x ∈ X) :
    (V.card : ℝ) ≤ ∑ x ∈ X, A (sys.cls x.1) j := by
  have hsub : V ⊆ X.biUnion fun x => (sys.classValues j).filter fun v => x ∈ sys.src v := fun v hv => by
    obtain ⟨hvj, x, hxs, hxX⟩ := hV v hv
    exact mem_biUnion.2 ⟨x, hxX, mem_filter.2 ⟨hvj, hxs⟩⟩
  calc (V.card : ℝ) ≤ ((∑ x ∈ X, ((sys.classValues j).filter fun v => x ∈ sys.src v).card : ℕ) : ℝ) := by
        exact_mod_cast (card_le_card hsub).trans card_biUnion_le
    _ = ∑ x ∈ X, (((sys.classValues j).filter fun v => x ∈ sys.src v).card : ℝ) := Nat.cast_sum _ _
    _ ≤ _ := sum_le_sum fun x _ => hM.2 x j

omit [Fintype ι] [DecidableEq ι] in
/-- The output entries of the products `s`, each weighted by its class, are the classes' sums. -/
theorem sum_sigma_wholeOut [Fintype κ] [DecidableEq κ] (s : Finset ι) (f : κ → ℝ) (I : Finset (Fin d.D.N)) :
    ∑ x ∈ s.sigma (fun q => (sys.prod q).wholeOut I), f (sys.cls x.1) =
      ∑ i, f i * ((∑ q ∈ s.filter (fun q => sys.cls q = i), ((sys.prod q).wholeOut I).card : ℕ) : ℝ) := by
  rw [sum_sigma]
  simp only [sum_const, nsmul_eq_mul]
  rw [← sum_fiberwise s sys.cls]
  refine sum_congr rfl fun i _ => ?_
  rw [Nat.cast_sum, mul_sum]
  refine sum_congr rfl fun q hq => ?_
  rw [show f (sys.cls q) = f i from congrArg f (mem_filter.1 hq).2, mul_comm]

omit [Fintype ι] [DecidableEq ι] in
/-- A post-fixed point is nonnegative. -/
theorem nonneg_of_postFixed [Fintype κ] {k b : ℕ} {cs : κ → ℝ} {A : κ → κ → ℝ} {μ : κ → ℝ} (hcs : ∀ j, 0 ≤ cs j)
    (hμ : sys.PostFixed k b cs A μ) (j : κ) : 0 ≤ μ j :=
  (mul_nonneg (div_nonneg (mul_nonneg (hcs j) (Nat.cast_nonneg k)) (by positivity)) (sq_nonneg _)).trans (hμ j)

/-- **The class bound below depth `t`**: in an inference block, each class `i`'s products less deep than `t` have the
whole chains of at most `μ_i k / b` output entries. -/
theorem classBound_lt [Fintype κ] [DecidableEq κ] {c k b : ℕ} {cs : κ → ℝ} {A : κ → κ → ℝ} {μ : κ → ℝ}
    (hS : ∀ p, (sys.prod p).Algorithm) (hV : sys.Values) (hsh : sys.Shared c k cs) (hW : sys.WidthExtended c k b)
    (hM : sys.Matched A) (hb : 0 < b) (hcs : ∀ j, 0 ≤ cs j) (hA : ∀ i j, 0 ≤ A i j) (hμ : sys.PostFixed k b cs A μ)
    {I : Finset (Fin d.D.N)} (hI : InferenceBlock d c k I) (t : ℕ) (i : κ) :
    ((∑ q ∈ (univ.filter fun q => sys.depth q < t).filter (fun q => sys.cls q = i),
      ((sys.prod q).wholeOut I).card : ℕ) : ℝ) ≤ μ i * k / b := by
  have hb' : (0 : ℝ) < b := by exact_mod_cast hb
  induction t generalizing i with
  | zero =>
    rw [filter_false_of_mem fun q _ => Nat.not_lt_zero _, filter_empty, sum_empty, Nat.cast_zero]
    exact div_nonneg (mul_nonneg (sys.nonneg_of_postFixed hcs hμ i) (Nat.cast_nonneg k)) hb'.le
  | succ t ih =>
    set F := (univ.filter fun q => sys.depth q < t + 1).filter (fun q => sys.cls q = i) with hF
    have hFi : ∀ p ∈ F, sys.cls p = i := fun p hp => (mem_filter.1 hp).2
    have hLW : sys.K i ^ 2 * ∑ p ∈ F, ((sys.prod p).wholeOut I).card ≤
        ∑ p ∈ F, ((sys.prod p).hadA I).card * ((sys.prod p).hadB I).card := by
      rw [mul_sum]
      refine sum_le_sum fun p hp => ?_
      rw [← congrArg sys.K (hFi p hp)]
      exact (sys.prod p).sq_mul_card_wholeOut_le (hS p) I
    have hmemH : ∀ v ∈ sys.hadValues F I, ∃ p ∈ F, v ∈ (sys.prod p).hadCones I := fun v hv => mem_biUnion.1 hv
    have hpaid : b * ((sys.hadValues F I).filter fun v => b ≤ (d.unregistered I ∩ v).card).card ≤ k := by
      refine (mul_card_le_card_inter (fun v hv w hw h => ?_) fun v hv => (mem_filter.1 hv).2).trans
        ((card_le_card inter_subset_left).trans hI.2.2)
      obtain ⟨p, -, hvp⟩ := hmemH v (mem_filter.1 hv).1
      obtain ⟨p', -, hwp⟩ := hmemH w (mem_filter.1 hw).1
      exact sys.disjoint_of_ne hV ((sys.prod p).hadCones_subset_values I hvp)
        ((sys.prod p').hadCones_subset_values I hwp) h
    have hmade : (((sys.hadValues F I).filter fun v => ¬ b ≤ (d.unregistered I ∩ v).card).card : ℝ) ≤
        ∑ x ∈ (univ.filter fun q => sys.depth q < t).sigma (fun q => (sys.prod q).wholeOut I),
          A (sys.cls x.1) i := by
      refine sys.card_le_sum_fanout hM i _ fun v hv => ?_
      obtain ⟨hvH, hvb⟩ := mem_filter.1 hv
      obtain ⟨p, hpF, hvp⟩ := hmemH v hvH
      have hvv := (sys.prod p).hadCones_subset_values I hvp
      refine ⟨mem_biUnion.2 ⟨p, mem_filter.2 ⟨mem_univ _, hFi p hpF⟩, hvv⟩, ?_⟩
      obtain ⟨x, hxs, hx⟩ := (hW I hI p v hvp).resolve_left hvb
      have hdx : sys.depth x.1 < sys.depth p := hM.1 p v hvv x hxs
      have hdp : sys.depth p < t + 1 := (mem_filter.1 (mem_filter.1 hpF).1).2
      exact ⟨x, hxs, mem_sigma.2 ⟨mem_filter.2 ⟨mem_univ _, by omega⟩, mem_filter.2 ⟨mem_univ _, hx⟩⟩⟩
    have hX : ∑ x ∈ (univ.filter fun q => sys.depth q < t).sigma (fun q => (sys.prod q).wholeOut I),
        A (sys.cls x.1) i ≤ (k : ℝ) / b * wholeFactor A μ i := by
      rw [sys.sum_sigma_wholeOut _ (fun i' => A i' i) I, wholeFactor, mul_sum]
      refine sum_le_sum fun i' _ => ?_
      calc A i' i * _ ≤ A i' i * (μ i' * k / b) := mul_le_mul_of_nonneg_left (ih i') (hA i' i)
        _ = (k : ℝ) / b * (A i' i * μ i') := by ring
    have hH : ((sys.hadValues F I).card : ℝ) ≤ (k : ℝ) / b * (1 + wholeFactor A μ i) := by
      have hp : (((sys.hadValues F I).filter fun v => b ≤ (d.unregistered I ∩ v).card).card : ℝ) ≤
          (k : ℝ) / b := by
        rw [le_div_iff₀ hb', mul_comm]
        exact_mod_cast hpaid
      rw [← card_filter_add_card_filter_not (fun v => b ≤ (d.unregistered I ∩ v).card), Nat.cast_add, mul_add,
        mul_one]
      linarith
    have hsum : (sys.K i : ℝ) ^ 2 * ((∑ p ∈ F, ((sys.prod p).wholeOut I).card : ℕ) : ℝ) ≤
        cs i * ((sys.hadValues F I).card : ℝ) ^ 2 := by
      have h1 : ((sys.K i ^ 2 * ∑ p ∈ F, ((sys.prod p).wholeOut I).card : ℕ) : ℝ) ≤
          ((∑ p ∈ F, ((sys.prod p).hadA I).card * ((sys.prod p).hadB I).card : ℕ) : ℝ) := Nat.cast_le.2 hLW
      rw [Nat.cast_mul, Nat.cast_pow] at h1
      exact h1.trans (hsh I hI i F hFi)
    have hKpos : (0 : ℝ) < (sys.K i : ℝ) ^ 2 := by
      have := sys.K_pos i
      positivity
    have hK0 : (sys.K i : ℝ) ≠ 0 := by exact_mod_cast (sys.K_pos i).ne'
    refine le_of_mul_le_mul_left ?_ hKpos
    calc (sys.K i : ℝ) ^ 2 * ((∑ p ∈ F, ((sys.prod p).wholeOut I).card : ℕ) : ℝ)
        ≤ cs i * ((sys.hadValues F I).card : ℝ) ^ 2 := hsum
      _ ≤ cs i * ((k : ℝ) / b * (1 + wholeFactor A μ i)) ^ 2 :=
          mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (Nat.cast_nonneg _) hH 2) (hcs i)
      _ = (sys.K i : ℝ) ^ 2 * (cs i * k / ((sys.K i : ℝ) ^ 2 * b) * (1 + wholeFactor A μ i) ^ 2 * ((k : ℝ) / b)) := by
          field_simp
      _ ≤ (sys.K i : ℝ) ^ 2 * (μ i * ((k : ℝ) / b)) :=
          mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right (hμ i) (by positivity)) hKpos.le
      _ = (sys.K i : ℝ) ^ 2 * (μ i * k / b) := by rw [mul_div_assoc]

end System

namespace SecurityProofs

theorem ClassBound : Guarantees.ClassBound := by
  intro d ι κ _ _ _ _ sys c k b cs A μ I j hS hV hsh hW hM hb hcs hA hμ hI
  have h := sys.classBound_lt hS hV hsh hW hM hb hcs hA hμ hI (univ.sup sys.depth + 1) j
  rwa [filter_true_of_mem fun q _ => Nat.lt_succ_of_le (le_sup (f := sys.depth) (mem_univ q))] at h

end SecurityProofs

namespace System

variable {ι κ : Type} [Fintype ι] [DecidableEq ι] (sys : System d ι κ)

/-- **Operands paid or made whole**: a block's whole MACs `T` of product `q` have
`b |π₁₂ T| + b |π₂₃ T| ≤ |U ∩ cones| + e k`, `e` the class's whole-making factor. -/
theorem ops_le [Fintype κ] [DecidableEq κ] {c k b : ℕ} {cs : κ → ℝ} {A : κ → κ → ℝ} {μ : κ → ℝ}
    (hS : ∀ p, (sys.prod p).Algorithm) (hV : sys.Values) (hsh : sys.Shared c k cs) (hW : sys.WidthExtended c k b)
    (hM : sys.Matched A) (hb : 0 < b) (hcs : ∀ j, 0 ≤ cs j) (hA : ∀ i j, 0 ≤ A i j) (hμ : sys.PostFixed k b cs A μ)
    {I : Finset (Fin d.D.N)} (hI : InferenceBlock d c k I) (q : ι) :
    (b : ℝ) * (((sys.prod q).whole I).image fun x => (x.1, x.2.1)).card +
        (b : ℝ) * (((sys.prod q).whole I).image fun x => (x.2.1, x.2.2)).card ≤
      ((d.unregistered I ∩ (sys.prod q).cones).card : ℝ) + k * wholeFactor A μ (sys.cls q) := by
  have hb' : (0 : ℝ) < b := by exact_mod_cast hb
  have hne : ∀ v ∈ (sys.prod q).values, v.Nonempty := fun v hv => (hV q q v hv v hv).1
  have h1 := (sys.prod q).card_add_card_le_hadCones (hS q) hne ((sys.prod q).image_whole_subset_hadA (hS q) I)
    ((sys.prod q).image_whole_subset_hadB (hS q) I)
  have hpaid : b * (((sys.prod q).hadCones I).filter fun v => b ≤ (d.unregistered I ∩ v).card).card ≤
      (d.unregistered I ∩ (sys.prod q).cones).card := by
    refine (mul_card_le_card_inter (fun v hv w hw h => ?_) fun v hv => (mem_filter.1 hv).2).trans
      (card_le_card (inter_subset_inter_left (biUnion_subset.2 fun v hv =>
        (sys.prod q).subset_cones_of_mem_values ((sys.prod q).hadCones_subset_values I (mem_filter.1 hv).1))))
    exact sys.disjoint_of_ne hV ((sys.prod q).hadCones_subset_values I (mem_filter.1 hv).1)
      ((sys.prod q).hadCones_subset_values I (mem_filter.1 hw).1) h
  have hmade : ((((sys.prod q).hadCones I).filter fun v => ¬ b ≤ (d.unregistered I ∩ v).card).card : ℝ) ≤
      ∑ x ∈ univ.sigma (fun q' => (sys.prod q').wholeOut I), A (sys.cls x.1) (sys.cls q) := by
    refine sys.card_le_sum_fanout hM _ _ fun v hv => ?_
    obtain ⟨hvH, hvb⟩ := mem_filter.1 hv
    refine ⟨mem_biUnion.2 ⟨q, mem_filter.2 ⟨mem_univ _, rfl⟩, (sys.prod q).hadCones_subset_values I hvH⟩, ?_⟩
    obtain ⟨x, hxs, hx⟩ := (hW I hI q v hvH).resolve_left hvb
    exact ⟨x, hxs, mem_sigma.2 ⟨mem_univ _, mem_filter.2 ⟨mem_univ _, hx⟩⟩⟩
  have hX : ∑ x ∈ univ.sigma (fun q' => (sys.prod q').wholeOut I), A (sys.cls x.1) (sys.cls q) ≤
      (k : ℝ) / b * wholeFactor A μ (sys.cls q) := by
    rw [sys.sum_sigma_wholeOut _ (fun i => A i (sys.cls q)) I, wholeFactor, mul_sum]
    refine sum_le_sum fun i _ => ?_
    have hc := SecurityProofs.ClassBound d ι κ sys c k b cs A μ I i hS hV hsh hW hM hb hcs hA hμ hI
    calc A i (sys.cls q) * _ ≤ A i (sys.cls q) * (μ i * k / b) := mul_le_mul_of_nonneg_left hc (hA _ _)
      _ = (k : ℝ) / b * (A i (sys.cls q) * μ i) := by ring
  rw [← card_filter_add_card_filter_not (s := (sys.prod q).hadCones I)
    (fun v => b ≤ (d.unregistered I ∩ v).card)] at h1
  have h1' : ((((sys.prod q).whole I).image fun x => (x.1, x.2.1)).card : ℝ) +
      (((sys.prod q).whole I).image fun x => (x.2.1, x.2.2)).card ≤
      ((((sys.prod q).hadCones I).filter fun v => b ≤ (d.unregistered I ∩ v).card).card : ℝ) +
        ((((sys.prod q).hadCones I).filter fun v => ¬ b ≤ (d.unregistered I ∩ v).card).card : ℝ) := by
    exact_mod_cast h1
  have hpaid' : (b : ℝ) * ((((sys.prod q).hadCones I).filter fun v => b ≤ (d.unregistered I ∩ v).card).card : ℝ) ≤
      ((d.unregistered I ∩ (sys.prod q).cones).card : ℝ) := by
    exact_mod_cast hpaid
  have hbX : (b : ℝ) * ((k : ℝ) / b * wholeFactor A μ (sys.cls q)) = k * wholeFactor A μ (sys.cls q) := by
    field_simp
  linarith [mul_le_mul_of_nonneg_left h1' hb'.le, mul_le_mul_of_nonneg_left (hmade.trans hX) hb'.le]

end System

namespace SecurityProofs

theorem WholeCap : Guarantees.WholeCap := by
  intro d ι κ _ _ _ _ sys c k b bC cs A μ q m P hS hV hsh hW hM hR hP hb hC hcs hA hμ
  obtain ⟨start, hcard, hread⟩ := hR
  have hCpos : (0 : ℝ) < bC := by exact_mod_cast hC
  have hbpos : (0 : ℝ) < b := by exact_mod_cast hb
  have he : 0 ≤ wholeFactor A μ (sys.cls q) :=
    sum_nonneg fun i _ => mul_nonneg (hA _ _) (sys.nonneg_of_postFixed hcs hμ i)
  have key := Verity.Combinatorics.sum_card_le_chain (fun u => (sys.prod q).whole (P.gates u))
    (fun u => ((d.unregistered (P.gates u) ∩ (sys.prod q).cones).card : ℝ) + k * wholeFactor A μ (sys.cls q))
    (fun u => ((d.unregistered (P.gates u) ∩ (sys.prod q).macs).card : ℝ))
    (fun u => ((∑ e : Fin (sys.ni q) × Fin (sys.nj q), (P.gates u ∩ start e.1 e.2).card : ℕ) : ℝ) / bC)
    (k := (k : ℝ) * (1 + wholeFactor A μ (sys.cls q))) (bA := b) (bB := b) (bC := bC)
    (O := ((sys.ni q * sys.nj q : ℕ) : ℝ)) (mul_nonneg (Nat.cast_nonneg _) (by linarith)) hbpos hbpos hCpos
    (fun _ => Nat.cast_nonneg _) (fun _ => div_nonneg (Nat.cast_nonneg _) hCpos.le)
    (fun u => sys.ops_le hS hV hsh hW hM hb hcs hA hμ (inferenceBlock_gates hP u) q)
    (fun u => by
      have h := (sys.prod q).card_add_card_le (hS q) (inferenceBlock_gates hP u)
      have h' : ((d.unregistered (P.gates u) ∩ (sys.prod q).cones).card : ℝ) +
          ((d.unregistered (P.gates u) ∩ (sys.prod q).macs).card : ℝ) ≤ k := by exact_mod_cast h
      linarith)
    (fun u => by
      have h := (sys.prod q).received_sums (hS q) hread hP u
      rw [mul_div_cancel₀ _ hCpos.ne']
      exact_mod_cast h)
    (by
      rw [← sum_div, div_le_iff₀ hCpos]
      exact_mod_cast sum_sum_card_inter_le hcard P)
  rw [Fintype.card_fin] at key
  have hsplit : ∑ u, ((d.unregistered (P.gates u) ∩ (sys.prod q).macs).card : ℝ) ≤ m * k := by
    calc ∑ u, ((d.unregistered (P.gates u) ∩ (sys.prod q).macs).card : ℝ) ≤ ∑ _u : Fin m, (k : ℝ) :=
          sum_le_sum fun u _ => by
            exact_mod_cast (Nat.le_add_left _ _).trans
              ((sys.prod q).card_add_card_le (hS q) (inferenceBlock_gates hP u))
      _ = m * k := by rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hN : ((sys.ni q * sys.K (sys.cls q) * sys.nj q : ℕ) : ℝ) ≤
      ∑ u, (((sys.prod q).whole (P.gates u)).card : ℝ) +
        ∑ u, ((d.unregistered (P.gates u) ∩ (sys.prod q).macs).card : ℝ) := by
    exact_mod_cast (sys.prod q).card_le_whole_add (hS q) P
  rw [mul_add]
  linarith

theorem WholeBound : Guarantees.WholeBound := by
  intro d ι κ _ _ _ _ sys c k b bC cs A μ G hS hV hsh hW hM hR hb hC hcs hA hμ m P hP
  refine Finset.sup_le fun q hq => ?_
  have hcap := WholeCap d ι κ sys c k b bC cs A μ q m P hS hV hsh hW hM (hR q hq) hP hb hC hcs hA hμ
  have he : 0 ≤ wholeFactor A μ (sys.cls q) :=
    sum_nonneg fun i _ => mul_nonneg (hA _ _) (sys.nonneg_of_postFixed hcs hμ i)
  have hk' : 0 ≤ (k : ℝ) * (1 + wholeFactor A μ (sys.cls q)) := mul_nonneg (Nat.cast_nonneg _) (by linarith)
  have hα : 0 ≤ (k : ℝ) * (1 + wholeFactor A μ (sys.cls q)) / 3 *
      √((k : ℝ) * (1 + wholeFactor A μ (sys.cls q)) / 3) / √((b : ℝ) * b * bC) + k :=
    add_nonneg (div_nonneg (mul_nonneg (div_nonneg hk' (by norm_num)) (Real.sqrt_nonneg _)) (Real.sqrt_nonneg _))
      (Nat.cast_nonneg _)
  have hβ : 0 ≤ (k : ℝ) * (1 + wholeFactor A μ (sys.cls q)) / 2 / √((b : ℝ) * b) *
      √((sys.ni q * sys.nj q : ℕ) : ℝ) :=
    mul_nonneg (div_nonneg (div_nonneg hk' (by norm_num)) (Real.sqrt_nonneg _)) (Real.sqrt_nonneg _)
  have hm : chainBlocks ((sys.ni q * sys.K (sys.cls q) * sys.nj q : ℕ) : ℝ)
      ((k : ℝ) * (1 + wholeFactor A μ (sys.cls q)) / 3 * √((k : ℝ) * (1 + wholeFactor A μ (sys.cls q)) / 3) /
        √((b : ℝ) * b * bC) + k)
      ((k : ℝ) * (1 + wholeFactor A μ (sys.cls q)) / 2 / √((b : ℝ) * b) * √((sys.ni q * sys.nj q : ℕ) : ℝ)) ≤
      m := by
    refine chainBlocks_le (Nat.cast_nonneg _) hα hβ (Nat.cast_nonneg _) ?_
    rw [Real.sqrt_mul (Nat.cast_nonneg (α := ℝ) m) ((sys.ni q * sys.nj q : ℕ) : ℝ)] at hcap
    linarith
  unfold wholeBound partitionSlowdown
  refine ENNReal.div_le_div_right ?_ _
  rw [show ((m * c : ℕ) : ℝ≥0∞) = (m : ℝ≥0∞) * c from Nat.cast_mul m c, ← ENNReal.ofReal_natCast m]
  gcongr

end SecurityProofs

end Nci.Products
