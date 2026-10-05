import Proofs.Flock.Soundness.LincheckPhase
import Proofs.Flock.Soundness.OpeningPhase
import Proofs.Flock.Soundness.ListSize
import Mathlib.Data.Nat.BitIndices

/-!
# The link points (PROTOCOL.md §16.2)

`link_sound`: if the committed values do not satisfy the statement, the two link points leave some
rep start undoomed with probability at most `linkError`. For a codeword near the committed table
whose witness satisfies the R1CS and the pin, some region fails; that region's claim, minus its
public value, is a nonzero polynomial in the link point of degree at most `|free| + nbl ≤ m_pts`
(`regionPoly`). It is nonzero because at the Boolean point of the failing region bit it is the
witness bit minus the public bit (`eval_regionPoly_bool`). Two independent points square the
probability, and a union bound runs over the list and the regions.

The layout of the point (`Statement.LinkLayout`): each region's free bits are distinct and read
link coordinates below `PT_LOCAL`, and the block coordinates `PT_LOCAL …` fit inside the point.
-/

namespace FlockSoundness

open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- The layout of the link points (§16.2): each region's free bits are distinct and read link
coordinates below `PT_LOCAL`, and the block coordinates `PT_LOCAL …` fit inside the point. -/
def Model.Statement.LinkLayout (S : Statement) (ptLocal mPts : ℕ) : Prop :=
  (∀ R ∈ S.regions, R.free.Nodup ∧ R.free.length ≤ ptLocal) ∧ ptLocal + S.nbl ≤ mPts

/-! ## The bits of a region address -/

theorem testBit_sum_two_pow (s : Finset ℕ) (n : ℕ) :
    (∑ i ∈ s, 2 ^ i).testBit n = decide (n ∈ s) := by
  have h : ∑ i ∈ s, 2 ^ i = ((Finset.sort s).map fun i => 2 ^ i).sum := by
    conv_lhs => rw [← Finset.sort_toFinset (s := s) (r := fun a b => a ≤ b)]
    rw [List.sum_toFinset _ (Finset.sort_nodup (s := s) (r := fun a b => a ≤ b))]
  rw [Bool.eq_iff_iff, decide_eq_true_iff, ← Nat.mem_bitIndices, h,
    Nat.bitIndices_sum_map_two_pow (Finset.sortedLT_sort s), Finset.mem_sort]

/-- The set of bits of `R.inBlock i`: the free positions where `i` has a 1, and the fixed ones. -/
def Model.Region.bitSet {kLog nbl : ℕ} (R : Region kLog nbl) (i : ℕ) : Finset ℕ :=
  (Finset.univ.filter fun t : Fin R.free.length => i.testBit t).image (fun t => (R.free.get t).val)
    ∪ (Finset.univ.filter fun j : Fin kLog => j ∉ R.free ∧ R.fixedBits j = true).image (·.val)

theorem inBlock_eq_sum {kLog nbl : ℕ} (R : Region kLog nbl) (hR : R.free.Nodup) (i : ℕ) :
    R.inBlock i = ∑ n ∈ R.bitSet i, 2 ^ n := by
  classical
  have hdisj : Disjoint
      ((Finset.univ.filter fun t : Fin R.free.length => i.testBit t).image
        (fun t => (R.free.get t).val))
      ((Finset.univ.filter fun j : Fin kLog => j ∉ R.free ∧ R.fixedBits j = true).image (·.val)) := by
    rw [Finset.disjoint_left]
    intro n hn hn'
    simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and] at hn hn'
    obtain ⟨t, _, rfl⟩ := hn
    obtain ⟨j, ⟨hj, _⟩, hjt⟩ := hn'
    exact hj (by rw [Fin.ext hjt]; exact List.get_mem _ _)
  rw [Model.Region.bitSet, Finset.sum_union hdisj, Finset.sum_image, Finset.sum_image]
  · simp only [Model.Region.inBlock, Finset.sum_filter, List.get_eq_getElem]
  · intro j _ j' _ h; exact Fin.ext h
  · intro t _ t' _ h
    exact (List.Nodup.get_inj_iff hR).1 (Fin.ext h)

/-- The bits of a region address: `i`'s bit at a free position, `fixedBits` elsewhere. -/
theorem testBit_inBlock {kLog nbl : ℕ} (R : Region kLog nbl) (hR : R.free.Nodup) (i : ℕ)
    (j : Fin kLog) :
    (R.inBlock i).testBit j =
      match R.free.idxOf? j with
      | some t => i.testBit t
      | none => R.fixedBits j := by
  classical
  rw [inBlock_eq_sum R hR, testBit_sum_two_pow]
  split
  · rename_i t ht
    obtain ⟨htl, hget, _⟩ := List.idxOf?_eq_some_iff.1 ht
    simp only [Model.Region.bitSet, Finset.mem_union, Finset.mem_image, Finset.mem_filter,
      Finset.mem_univ, true_and]
    have hjmem : j ∈ R.free := by rw [← hget]; exact List.getElem_mem htl
    rw [Bool.eq_iff_iff, decide_eq_true_iff]
    constructor
    · rintro (⟨t', ht', hjt⟩ | ⟨j', ⟨hj', _⟩, hjj⟩)
      · have : t' = ⟨t, htl⟩ := by
          apply (List.Nodup.get_inj_iff hR).1
          have h2 : R.free.get ⟨t, htl⟩ = j := by simpa using hget
          rw [h2]
          exact Fin.ext hjt
        rw [this] at ht'
        exact ht'
      · exact absurd (by rw [Fin.ext hjj]; exact hjmem) hj'
    · intro hi
      exact Or.inl ⟨⟨t, htl⟩, hi, by simp [hget]⟩
  · rename_i hnone
    have hj : j ∉ R.free := List.idxOf?_eq_none_iff.1 hnone
    simp only [Model.Region.bitSet, Finset.mem_union, Finset.mem_image, Finset.mem_filter,
      Finset.mem_univ, true_and]
    rw [Bool.eq_iff_iff, decide_eq_true_iff]
    constructor
    · rintro (⟨t', _, hjt⟩ | ⟨j', ⟨_, hfix⟩, hjj⟩)
      · exact absurd (by rw [← Fin.ext hjt]; exact List.get_mem _ _) hj
      · rw [← Fin.ext hjj]; exact hfix
    · intro hfix
      exact Or.inr ⟨j, ⟨hj, hfix⟩, rfl⟩

theorem testBit_inBlock_ge {kLog nbl : ℕ} (R : Region kLog nbl) (hR : R.free.Nodup) (i n : ℕ)
    (hn : kLog ≤ n) : (R.inBlock i).testBit n = false := by
  classical
  rw [inBlock_eq_sum R hR, testBit_sum_two_pow, decide_eq_false_iff_not]
  simp only [Model.Region.bitSet, Finset.mem_union, Finset.mem_image, Finset.mem_filter,
    Finset.mem_univ, true_and, not_or, not_exists, not_and]
  exact ⟨fun t _ h => by have := (R.free.get t).isLt; omega, fun j _ h => by have := j.isLt; omega⟩

/-! ## A region claim as a polynomial in the link point -/

/-- A region claim evaluates to `Σ_a eq(p, a)·z[a]` at its point `p`. -/
theorem regionClaim_eval (S : Statement) (R : Region S.kLog S.nbl) (ptLocal mPts : ℕ)
    (r : Fin mPts → F) (z : Witness S.m) :
    (regionClaim S R ptLocal mPts r).eval z =
      ∑ a : Fin (2 ^ S.m), eqAt (regionPoint S R ptLocal mPts r) a.val * ((z a).val : F) := by
  have h6 : 6 ≤ S.m := le_trans S.hk6 S.hk
  set p := regionPoint S R ptLocal mPts r with hp
  have hskip : (regionClaim S R ptLocal mPts r).skip =
      fun s : Fin 64 => eqAt (fun j : Fin 6 => p ⟨j.val, by omega⟩) s.val := by
    funext s
    simp only [regionClaim, eqAt, ← hp]
    refine Finset.prod_congr rfl fun j _ => ?_
    simp only [dif_pos (show j.val < S.m by omega)]
  have hx : (regionClaim S R ptLocal mPts r).x =
      fun i : Fin (S.m - 6) => p ⟨6 + i.val, by omega⟩ := by
    funext i
    simp only [regionClaim, ← hp]
    rw [dif_pos (show i.val + 6 < S.m by omega)]
    exact congrArg p (Fin.ext (by simp only; omega))
  unfold Claim.eval
  rw [hskip, hx]
  refine Finset.sum_congr rfl fun a _ => ?_
  congr 1
  have hsplit := eqAt_split_at 6 (S.m - 6) (by omega) p (a.val % 64) (a.val / 64)
    (by norm_num; exact Nat.mod_lt _ (by norm_num))
  rw [show a.val % 64 + 2 ^ 6 * (a.val / 64) = a.val by omega] at hsplit
  rw [hsplit]

/-- Coordinate `j` of a region's point as a polynomial in the link coordinates. -/
noncomputable def pointPoly (S : Statement) (R : Region S.kLog S.nbl) (ptLocal mPts : ℕ)
    (j : Fin S.m) : MvPolynomial (Fin mPts) F :=
  if hj : j.val < S.kLog then
    match R.free.idxOf? ⟨j.val, hj⟩ with
    | some t => if ht : t < mPts then MvPolynomial.X ⟨t, ht⟩ else 0
    | none => if R.fixedBits ⟨j.val, hj⟩ then 1 else 0
  else
    if hi : ptLocal + (j.val - S.kLog) < mPts then
      MvPolynomial.X ⟨ptLocal + (j.val - S.kLog), hi⟩ else 0

theorem eval_pointPoly (S : Statement) (R : Region S.kLog S.nbl) (ptLocal mPts : ℕ)
    (r : Fin mPts → F) (j : Fin S.m) :
    MvPolynomial.eval r (pointPoly S R ptLocal mPts j) = regionPoint S R ptLocal mPts r j := by
  unfold pointPoly regionPoint
  by_cases hj : j.val < S.kLog
  · rw [dif_pos hj, dif_pos hj]
    rcases h : R.free.idxOf? ⟨j.val, hj⟩ with _ | t
    · simp only [h]
      split_ifs <;> simp
    · simp only [h]
      split_ifs <;> simp
  · rw [dif_neg hj, dif_neg hj]
    dsimp only
    split_ifs <;> simp

/-- Whether coordinate `j` of the point reads a link coordinate. -/
def pointVar (S : Statement) (R : Region S.kLog S.nbl) (j : Fin S.m) : ℕ :=
  if hj : j.val < S.kLog then (if (⟨j.val, hj⟩ : Fin S.kLog) ∈ R.free then 1 else 0) else 1

theorem totalDegree_pointFactor_le (S : Statement) (R : Region S.kLog S.nbl) (ptLocal mPts : ℕ)
    (b : Bool) (j : Fin S.m) :
    (if b then pointPoly (F := F) S R ptLocal mPts j else 1 + pointPoly S R ptLocal mPts j).totalDegree
      ≤ pointVar S R j := by
  have hP : (pointPoly (F := F) S R ptLocal mPts j).totalDegree ≤ pointVar S R j := by
    unfold pointPoly pointVar
    by_cases hj : j.val < S.kLog
    · rw [dif_pos hj, dif_pos hj]
      rcases h : R.free.idxOf? ⟨j.val, hj⟩ with _ | t
      · simp only [h]
        split_ifs <;> simp
      · have hmem : (⟨j.val, hj⟩ : Fin S.kLog) ∈ R.free := by
          have := List.isSome_idxOf? (l := R.free) (a := ⟨j.val, hj⟩)
          rw [h] at this; simpa using this
        rw [if_pos hmem]
        simp only [h]
        split_ifs
        · rw [MvPolynomial.totalDegree_X]
        · simp
    · rw [dif_neg hj, dif_neg hj]
      split_ifs
      · rw [MvPolynomial.totalDegree_X]
      · simp
  cases b
  · simp only [Bool.false_eq_true, if_false]
    exact (MvPolynomial.totalDegree_add _ _).trans (max_le (by simp) hP)
  · simpa using hP

theorem sum_pointVar_le (S : Statement) (R : Region S.kLog S.nbl) :
    ∑ j : Fin S.m, pointVar S R j ≤ R.free.length + S.nbl := by
  classical
  have hm : S.kLog + S.nbl = S.m := by simp only [Statement.nbl]; have := S.hk; omega
  rw [← (finCongr hm).sum_comp, Fin.sum_univ_add]
  have h1 : ∑ j : Fin S.kLog, pointVar S R ((finCongr hm) (Fin.castAdd S.nbl j)) =
      (Finset.univ.filter fun j : Fin S.kLog => j ∈ R.free).card := by
    rw [Finset.card_filter]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp only [pointVar, finCongr_apply, Fin.val_cast, Fin.val_castAdd, dif_pos j.isLt]
  have h2 : ∑ i : Fin S.nbl, pointVar S R ((finCongr hm) (Fin.natAdd S.kLog i)) = S.nbl := by
    have hone : ∀ i : Fin S.nbl, pointVar S R ((finCongr hm) (Fin.natAdd S.kLog i)) = 1 :=
      fun i => by
        simp only [pointVar, finCongr_apply, Fin.val_cast, Fin.val_natAdd]
        rw [dif_neg (by omega)]
    simp only [hone, Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul, mul_one]
  rw [h1, h2]
  have h3 : (Finset.univ.filter fun j : Fin S.kLog => j ∈ R.free).card ≤ R.free.length := by
    calc (Finset.univ.filter fun j : Fin S.kLog => j ∈ R.free).card
        ≤ R.free.toFinset.card := Finset.card_le_card fun j hj => by simpa using hj
      _ ≤ R.free.length := List.toFinset_card_le _
  omega

/-- Link coordinate `c` as a polynomial (0 beyond the point, as the model reads it). -/
noncomputable def coordPoly (mPts c : ℕ) : MvPolynomial (Fin mPts) F :=
  if hc : c < mPts then MvPolynomial.X ⟨c, hc⟩ else 0

theorem eval_coordPoly (mPts c : ℕ) (r : Fin mPts → F) :
    MvPolynomial.eval r (coordPoly mPts c) = if hc : c < mPts then r ⟨c, hc⟩ else 0 := by
  unfold coordPoly; split_ifs <;> simp

theorem totalDegree_coordFactor_le (mPts c : ℕ) (b : Bool) :
    (if b = true then coordPoly (F := F) mPts c else 1 + coordPoly mPts c).totalDegree ≤ 1 := by
  have h : (coordPoly (F := F) mPts c).totalDegree ≤ 1 := by
    unfold coordPoly; split_ifs
    · rw [MvPolynomial.totalDegree_X]
    · simp
  split_ifs
  · exact h
  · exact (MvPolynomial.totalDegree_add _ _).trans (max_le (by simp) h)

/-- A region claim minus its public value, as a polynomial in the link point. -/
noncomputable def regionPoly (S : Statement) (R : Region S.kLog S.nbl) (ptLocal mPts : ℕ)
    (z : Witness S.m) : MvPolynomial (Fin mPts) F :=
  (∑ a : Fin (2 ^ S.m), MvPolynomial.C ((z a).val : F) *
      ∏ j : Fin S.m, (if a.val.testBit j then pointPoly S R ptLocal mPts j
        else 1 + pointPoly S R ptLocal mPts j))
  - ∑ b : Fin (2 ^ S.nbl), (∏ i : Fin S.nbl,
        (if b.val.testBit i then coordPoly mPts (ptLocal + i.val)
          else 1 + coordPoly mPts (ptLocal + i.val)))
      * ∑ i ∈ Finset.range (2 ^ R.free.length), (∏ t : Fin R.free.length,
          (if i.testBit t then coordPoly mPts t.val else 1 + coordPoly mPts t.val))
        * MvPolynomial.C ((R.bits b i).val : F)

theorem eval_regionPoly (S : Statement) (R : Region S.kLog S.nbl) (ptLocal mPts : ℕ)
    (z : Witness S.m) (r : Fin mPts → F) :
    MvPolynomial.eval r (regionPoly S R ptLocal mPts z) =
      (regionClaim S R ptLocal mPts r).eval z - regionValue S R ptLocal mPts r := by
  rw [regionClaim_eval]
  simp only [regionPoly, map_sub, map_sum, map_mul, map_prod, MvPolynomial.eval_C]
  congr 1
  · refine Finset.sum_congr rfl fun a _ => ?_
    rw [mul_comm]
    congr 1
    unfold eqAt
    refine Finset.prod_congr rfl fun j _ => ?_
    split_ifs <;> simp [eval_pointPoly]
  · unfold regionValue
    dsimp only
    refine Finset.sum_congr rfl fun b _ => ?_
    congr 1
    · refine Finset.prod_congr rfl fun i _ => ?_
      by_cases hc : ptLocal + i.val < mPts
      · split_ifs <;> simp [eval_coordPoly, hc]
      · split_ifs <;> simp [eval_coordPoly, hc]
    · refine Finset.sum_congr rfl fun i _ => ?_
      congr 1
      refine Finset.prod_congr rfl fun t _ => ?_
      by_cases hc : t.val < mPts
      · split_ifs <;> simp [eval_coordPoly, hc]
      · split_ifs <;> simp [eval_coordPoly, hc]

theorem totalDegree_regionPoly_le (S : Statement) (R : Region S.kLog S.nbl) (ptLocal mPts : ℕ)
    (z : Witness S.m) :
    (regionPoly (F := F) S R ptLocal mPts z).totalDegree ≤ R.free.length + S.nbl := by
  unfold regionPoly
  refine (MvPolynomial.totalDegree_sub _ _).trans (max_le ?_ ?_)
  · refine (MvPolynomial.totalDegree_finsetSum _ _).trans (Finset.sup_le fun a _ => ?_)
    refine (MvPolynomial.totalDegree_mul _ _).trans ?_
    rw [MvPolynomial.totalDegree_C, zero_add]
    refine (MvPolynomial.totalDegree_finsetProd _ _).trans ?_
    exact (Finset.sum_le_sum fun j _ => totalDegree_pointFactor_le S R ptLocal mPts _ j).trans
      (sum_pointVar_le S R)
  · refine (MvPolynomial.totalDegree_finsetSum _ _).trans (Finset.sup_le fun b _ => ?_)
    refine (MvPolynomial.totalDegree_mul _ _).trans ?_
    have h1 : (∏ i : Fin S.nbl, (if b.val.testBit i then coordPoly (F := F) mPts (ptLocal + i.val)
        else 1 + coordPoly mPts (ptLocal + i.val))).totalDegree ≤ S.nbl := by
      refine (MvPolynomial.totalDegree_finsetProd _ _).trans ?_
      calc ∑ i : Fin S.nbl, (if b.val.testBit i then coordPoly (F := F) mPts (ptLocal + i.val)
            else 1 + coordPoly mPts (ptLocal + i.val)).totalDegree
          ≤ ∑ _i : Fin S.nbl, 1 :=
            Finset.sum_le_sum fun i _ => totalDegree_coordFactor_le mPts _ _
        _ = S.nbl := by simp
    have h2 : (∑ i ∈ Finset.range (2 ^ R.free.length), (∏ t : Fin R.free.length,
        (if i.testBit t then coordPoly (F := F) mPts t.val else 1 + coordPoly mPts t.val))
          * MvPolynomial.C ((R.bits b i).val : F)).totalDegree ≤ R.free.length := by
      refine (MvPolynomial.totalDegree_finsetSum _ _).trans (Finset.sup_le fun i _ => ?_)
      refine (MvPolynomial.totalDegree_mul _ _).trans ?_
      rw [MvPolynomial.totalDegree_C, add_zero]
      refine (MvPolynomial.totalDegree_finsetProd _ _).trans ?_
      calc ∑ t : Fin R.free.length, (if i.testBit t then coordPoly (F := F) mPts t.val
            else 1 + coordPoly mPts t.val).totalDegree
          ≤ ∑ _t : Fin R.free.length, 1 :=
            Finset.sum_le_sum fun t _ => totalDegree_coordFactor_le mPts _ _
        _ = R.free.length := by simp
    omega

/-! ## At the Boolean point of a failing region bit -/

/-- The Boolean link point of region bit `i₀` in block `o₀`. -/
def boolPoint (S : Statement) (R : Region S.kLog S.nbl) (ptLocal mPts : ℕ) (o₀ : Fin (2 ^ S.nbl))
    (i₀ : ℕ) : Fin mPts → F := fun c =>
  if c.val < R.free.length then (if i₀.testBit c.val then 1 else 0)
  else if ptLocal ≤ c.val ∧ c.val < ptLocal + S.nbl then
    (if o₀.val.testBit (c.val - ptLocal) then 1 else 0)
  else 0

theorem eval_regionPoly_bool (S : Statement) (R : Region S.kLog S.nbl) (ptLocal mPts : ℕ)
    [CharP F 2] (hR : R.free.Nodup) (hfree : R.free.length ≤ ptLocal) (hpt : ptLocal + S.nbl ≤ mPts)
    (z : Witness S.m) (o₀ : Fin (2 ^ S.nbl)) (i₀ : ℕ) (hi₀ : i₀ < 2 ^ R.free.length)
    (h₀ : R.inBlock i₀ < 2 ^ S.kLog) :
    MvPolynomial.eval (boolPoint (F := F) S R ptLocal mPts o₀ i₀) (regionPoly S R ptLocal mPts z) =
      ((S.block z o₀ ⟨R.inBlock i₀, h₀⟩).val : F) - ((R.bits o₀ i₀).val : F) := by
  classical
  set r := boolPoint (F := F) S R ptLocal mPts o₀ i₀ with hr
  set a₀ := S.addr o₀ ⟨R.inBlock i₀, h₀⟩ with ha₀
  have hnbl : S.nbl = S.m - S.kLog := rfl
  have hbit : ∀ n, a₀.val.testBit n =
      if n < S.kLog then (R.inBlock i₀).testBit n else o₀.val.testBit (n - S.kLog) := fun n => by
    simp only [ha₀, Statement.addr]
    rw [mul_comm, Nat.testBit_two_pow_mul_add _ h₀]
  -- the claim's point at `r` is the bits of `a₀`
  have hp : regionPoint S R ptLocal mPts r = fun j : Fin S.m => if a₀.val.testBit j then 1 else 0 := by
    funext j
    rw [hbit]
    unfold regionPoint
    by_cases hj : j.val < S.kLog
    · rw [dif_pos hj, if_pos hj]
      have htb := testBit_inBlock R hR i₀ ⟨j.val, hj⟩
      simp only at htb
      rw [htb]
      rcases h : R.free.idxOf? ⟨j.val, hj⟩ with _ | t
      · simp only [h] <;> split_ifs <;> simp_all
      · obtain ⟨htl, _, _⟩ := List.idxOf?_eq_some_iff.1 h
        simp only [h]
        rw [dif_pos (show t < mPts by omega)]
        simp only [hr, boolPoint, if_pos htl]
    · rw [dif_neg hj, if_neg hj]
      dsimp only
      have hjm := j.isLt
      rw [dif_pos (show ptLocal + (j.val - S.kLog) < mPts by omega)]
      simp only [hr, boolPoint]
      rw [if_neg (show ¬ (ptLocal + (j.val - S.kLog) < R.free.length) by omega),
        if_pos (show ptLocal ≤ ptLocal + (j.val - S.kLog) ∧
          ptLocal + (j.val - S.kLog) < ptLocal + S.nbl by omega),
        show ptLocal + (j.val - S.kLog) - ptLocal = j.val - S.kLog by omega]
  -- the region value at `r` is the public bit
  have hval : regionValue S R ptLocal mPts r = ((R.bits o₀ i₀).val : F) := by
    unfold regionValue
    dsimp only
    have hb : ∀ b : Fin (2 ^ S.nbl), (∏ i : Fin S.nbl,
        (if b.val.testBit i then (if hc : ptLocal + i.val < mPts then r ⟨ptLocal + i.val, hc⟩ else 0)
          else 1 + (if hc : ptLocal + i.val < mPts then r ⟨ptLocal + i.val, hc⟩ else 0))) =
        if b = o₀ then 1 else 0 := fun b => by
      have := eqAt_bool (R := F) (n := S.nbl) b.val o₀.val b.isLt o₀.isLt
      rw [show (if b = o₀ then (1 : F) else 0) = if b.val = o₀.val then 1 else 0 by
        simp [Fin.ext_iff], ← this]
      unfold eqAt
      refine Finset.prod_congr rfl fun i _ => ?_
      rw [dif_pos (show ptLocal + i.val < mPts by omega)]
      simp only [hr, boolPoint]
      rw [if_neg (show ¬ (ptLocal + i.val < R.free.length) by omega),
        if_pos (show ptLocal ≤ ptLocal + i.val ∧ ptLocal + i.val < ptLocal + S.nbl by omega),
        show ptLocal + i.val - ptLocal = i.val by omega]
    have hi : ∀ i ∈ Finset.range (2 ^ R.free.length), (∏ t : Fin R.free.length,
        (if i.testBit t then (if ht : t.val < mPts then r ⟨t.val, ht⟩ else 0)
          else 1 + (if ht : t.val < mPts then r ⟨t.val, ht⟩ else 0))) =
        if i = i₀ then 1 else 0 := fun i hi => by
      rw [← eqAt_bool (R := F) (n := R.free.length) i i₀ (Finset.mem_range.1 hi) hi₀]
      unfold eqAt
      refine Finset.prod_congr rfl fun t _ => ?_
      rw [dif_pos (show t.val < mPts by omega)]
      simp only [hr, boolPoint, if_pos t.isLt]
    simp only [hb]
    rw [Finset.sum_eq_single o₀ (fun b _ hb => by simp [hb]) (by simp)]
    simp only [if_true, one_mul]
    rw [Finset.sum_congr rfl fun i hi' => by rw [hi i hi'], Finset.sum_eq_single i₀
      (fun i _ hi => by simp [hi]) (fun h => absurd (Finset.mem_range.2 hi₀) h)]
    simp
  rw [eval_regionPoly, hval, regionClaim_eval, hp]
  congr 1
  rw [Finset.sum_eq_single a₀]
  · rw [eqAt_bool a₀.val a₀.val a₀.isLt a₀.isLt, if_pos rfl, one_mul]
    rfl
  · intro a _ ha
    rw [eqAt_bool a.val a₀.val a.isLt a₀.isLt, if_neg (fun h => ha (Fin.ext h)), zero_mul]
  · simp

/-! ## The link points -/

/-- A region fails for a witness: some region bit differs from its public value. -/
def RegionFails (S : Statement) (R : Region S.kLog S.nbl) (z : Witness S.m) : Prop :=
  ∃ (o : Fin (2 ^ S.nbl)) (i : ℕ) (_hi : i < 2 ^ R.free.length) (h : R.inBlock i < 2 ^ S.kLog),
    S.block z o ⟨R.inBlock i, h⟩ ≠ R.bits o i

omit [Fintype F] [DecidableEq F] in
theorem val_sub_val_ne_zero (x y : ZMod 2) (hxy : x ≠ y) :
    ((x.val : ℕ) : F) - ((y.val : ℕ) : F) ≠ 0 := by
  fin_cases x <;> fin_cases y
  · exact absurd rfl hxy
  · show ((0 : ℕ) : F) - ((1 : ℕ) : F) ≠ 0
    simp
  · show ((1 : ℕ) : F) - ((0 : ℕ) : F) ≠ 0
    simp
  · exact absurd rfl hxy

/-- **One region, one point.** A failing region's claim holds at a uniform link point with
probability at most `m_pts/|F|`. -/
theorem prCoin_regionClaim_le (A : Arith F K) (hA : A.Correct) (S : Statement)
    (R : Region S.kLog S.nbl) (ptLocal mPts : ℕ) (hR : R.free.Nodup)
    (hfree : R.free.length ≤ ptLocal) (hpt : ptLocal + S.nbl ≤ mPts) (z : Witness S.m)
    (hfail : RegionFails S R z) :
    prCoin (fun r : Fin mPts → F => (regionClaim S R ptLocal mPts r).Holds z) ≤
      (mPts : ℝ≥0∞) / Fintype.card F := by
  have := hA.charTwo
  obtain ⟨o₀, i₀, hi₀, h₀, hne⟩ := hfail
  have hQ : regionPoly (F := F) S R ptLocal mPts z ≠ 0 := by
    intro h0
    have h := eval_regionPoly_bool (F := F) S R ptLocal mPts hR hfree hpt z o₀ i₀ hi₀ h₀
    rw [h0, map_zero] at h
    exact val_sub_val_ne_zero _ _ hne h.symm
  calc prCoin (fun r : Fin mPts → F => (regionClaim S R ptLocal mPts r).Holds z)
      = prCoin (fun r : Fin mPts → F =>
          MvPolynomial.eval r (regionPoly S R ptLocal mPts z) = 0) := by
        congr 1
        funext r
        rw [eval_regionPoly, sub_eq_zero]
        rfl
    _ ≤ ((regionPoly (F := F) S R ptLocal mPts z).totalDegree : ℝ≥0∞) / Fintype.card F :=
        prCoin_mvEval_eq_zero_le _ hQ
    _ ≤ (mPts : ℝ≥0∞) / Fintype.card F := by
        have := totalDegree_regionPoly_le (F := F) S R ptLocal mPts z
        gcongr
        exact_mod_cast (by omega : (regionPoly (F := F) S R ptLocal mPts z).totalDegree ≤ mPts)

/-- Two independent uniform points. -/
theorem prCoin_pair_le {α : Type} [Fintype α] (P : α → Prop) :
    prCoin (fun pts : Fin 2 → α => P (pts 0) ∧ P (pts 1)) ≤ prCoin P * prCoin P := by
  classical
  refine le_of_eq ?_
  simp only [prCoin, avg]
  rw [← (piFinTwoEquiv fun _ => α).symm.sum_comp, Fintype.sum_prod_type, Finset.sum_mul_sum]
  refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun y _ => ?_
  simp only [piFinTwoEquiv_symm_apply, Fin.cons_zero, Fin.cons_one, Fintype.card_fun,
    Fintype.card_fin, Nat.cast_pow]
  rw [pow_two, ENNReal.mul_inv (Or.inr (ENNReal.natCast_ne_top _))
    (Or.inl (ENNReal.natCast_ne_top _))]
  by_cases hx : P x <;> by_cases hy : P y <;> simp [hx, hy, mul_comm]

/-- The link event for one codeword and one region: the region fails, yet both link points'
claims hold. -/
def LinkBad (S : Statement) (R : Region S.kLog S.nbl) (ptLocal mPts : ℕ) (z : Witness S.m)
    (pts : Fin 2 → Fin mPts → F) : Prop :=
  RegionFails S R z ∧ (regionClaim S R ptLocal mPts (pts 0)).Holds z ∧
    (regionClaim S R ptLocal mPts (pts 1)).Holds z

theorem prCoin_linkBad_le (A : Arith F K) (hA : A.Correct) (S : Statement)
    (R : Region S.kLog S.nbl) (ptLocal mPts : ℕ) (hlay : S.LinkLayout ptLocal mPts)
    (hR : R ∈ S.regions) (z : Witness S.m) :
    prCoin (LinkBad (F := F) S R ptLocal mPts z) ≤ ((mPts : ℝ≥0∞) / Fintype.card F) ^ 2 := by
  by_cases hf : RegionFails S R z
  · obtain ⟨hnd, hfr⟩ := hlay.1 R hR
    have h1 := prCoin_regionClaim_le A hA S R ptLocal mPts hnd hfr hlay.2 z hf
    calc prCoin (LinkBad (F := F) S R ptLocal mPts z)
        ≤ prCoin (fun pts : Fin 2 → Fin mPts → F =>
            (fun r : Fin mPts → F => (regionClaim S R ptLocal mPts r).Holds z) (pts 0) ∧
              (fun r : Fin mPts → F => (regionClaim S R ptLocal mPts r).Holds z) (pts 1)) :=
          prCoin_mono fun pts h => h.2
      _ ≤ prCoin (fun r : Fin mPts → F => (regionClaim S R ptLocal mPts r).Holds z) *
            prCoin (fun r : Fin mPts → F => (regionClaim S R ptLocal mPts r).Holds z) :=
          prCoin_pair_le (fun r : Fin mPts → F => (regionClaim S R ptLocal mPts r).Holds z)
      _ ≤ ((mPts : ℝ≥0∞) / Fintype.card F) * ((mPts : ℝ≥0∞) / Fintype.card F) :=
          mul_le_mul' h1 h1
      _ = _ := (sq _).symm
  · calc prCoin (LinkBad (F := F) S R ptLocal mPts z)
        ≤ prCoin (fun _ : Fin 2 → Fin mPts → F => False) := prCoin_mono fun pts h => hf h.1
      _ = 0 := by simp [prCoin, avg]
      _ ≤ _ := zero_le

open Classical in
/-- **The link points** (§16.2), for a list `W` of at most `L` messages none of which packs a
satisfying witness: the link points leave some rep start undoomed over `W` with probability at
most `L·|regions|·(m_pts/|F|)²`. -/
theorem link_sound_of_list (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (ptLocal mPts : ℕ) (W : Set (Message F sch.k0 sch.level₀.logCols))
    (hC : ¬ ∃ M ∈ W, S.Satisfies (wit A S sch M))
    (hlay : S.LinkLayout ptLocal mPts) (L : ℕ)
    (hL : (Finset.univ.filter fun M => M ∈ W).card ≤ L) :
    prCoin (fun pts : Fin 2 → Fin mPts → F =>
        ¬ RepDoomedW A S sch W (extraClaims S ptLocal mPts pts)) ≤
      L * (S.regions.length * ((mPts : ℝ≥0∞) / Fintype.card F) ^ 2) := by
  classical
  set s := Finset.univ.filter fun M => M ∈ W
  calc prCoin (fun pts : Fin 2 → Fin mPts → F =>
          ¬ RepDoomedW A S sch W (extraClaims S ptLocal mPts pts))
      ≤ prCoin (fun pts => ∃ M ∈ s, ∃ R ∈ S.regions.toFinset,
          LinkBad S R ptLocal mPts (wit A S sch M) pts) := by
        refine prCoin_mono fun pts h => ?_
        simp only [RepDoomedW, not_forall, not_not] at h
        obtain ⟨M, hMW, hr1cs, hpin, hext⟩ := h
        have hsat : ¬ S.Satisfies (wit A S sch M) := fun hs => hC ⟨M, hMW, hs⟩
        have hreg : ¬ ∀ R ∈ S.regions, ∀ (o : Fin (2 ^ S.nbl)) (i : ℕ)
            (_hi : i < 2 ^ R.free.length), ∀ h : R.inBlock i < 2 ^ S.kLog,
              S.block (wit A S sch M) o ⟨R.inBlock i, h⟩ = R.bits o i :=
          fun hr => hsat ⟨hr1cs, hpin, hr⟩
        push Not at hreg
        obtain ⟨R, hR, o, i, hi, hib, hne⟩ := hreg
        refine ⟨M, by simp only [s, Finset.mem_filter, Finset.mem_univ, true_and]; exact hMW,
          R, List.mem_toFinset.2 hR, ⟨o, i, hi, hib, hne⟩, hext _ ?_, hext _ ?_⟩
        · exact List.mem_flatMap.2 ⟨R, hR, by simp⟩
        · exact List.mem_flatMap.2 ⟨R, hR, by simp⟩
    _ ≤ ∑ M ∈ s, prCoin (fun pts => ∃ R ∈ S.regions.toFinset,
          LinkBad S R ptLocal mPts (wit A S sch M) pts) :=
        prCoin_exists_le _ _
    _ ≤ ∑ M ∈ s, ∑ R ∈ S.regions.toFinset,
          prCoin (LinkBad S R ptLocal mPts (wit A S sch M)) :=
        Finset.sum_le_sum fun M _ => prCoin_exists_le _ _
    _ ≤ ∑ _M ∈ s, ∑ _R ∈ S.regions.toFinset, ((mPts : ℝ≥0∞) / Fintype.card F) ^ 2 :=
        Finset.sum_le_sum fun M _ => Finset.sum_le_sum fun R hR =>
          prCoin_linkBad_le A hA S R ptLocal mPts hlay (List.mem_toFinset.1 hR) _
    _ ≤ L * (S.regions.length * ((mPts : ℝ≥0∞) / Fintype.card F) ^ 2) := by
        simp only [Finset.sum_const, nsmul_eq_mul]
        gcongr
        all_goals first | exact_mod_cast hL | exact_mod_cast List.toFinset_card_le _

open Classical in
/-- **The link points** for the codewords near the committed table: `link_sound_of_list` at
`closeMsgs`. -/
theorem link_sound_of_card (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (ptLocal mPts : ℕ) (C : Oracle F) (hC : ¬ Committed A S sch C)
    (hlay : S.LinkLayout ptLocal mPts) (L : ℕ)
    (hL : (Finset.univ.filter fun M => M ∈ closeMsgs A sch C).card ≤ L) :
    prCoin (fun pts : Fin 2 → Fin mPts → F =>
        ¬ RepDoomed A S sch C (extraClaims S ptLocal mPts pts)) ≤
      L * (S.regions.length * ((mPts : ℝ≥0∞) / Fintype.card F) ^ 2) := by
  simp only [repDoomed_iff]
  exact link_sound_of_list A hA S sch ptLocal mPts _ (fun ⟨M, hM, hs⟩ => hC ⟨M, hM, hs⟩) hlay L hL

end FlockSoundness
