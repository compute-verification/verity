import Proofs.Flock.Soundness.LigeritoWeights

/-!
# How Ligerito's weights fold (PROTOCOL.md §13.3, §13.6)

Every term of the final check's weight is a constant times a multilinear extension read at the
point `fixed ++ bits(y)`: the fixed challenges so far, then the bits of the still-open variables.
Folding the open variables' lowest bits by challenges `ch` replaces those bits by `ch`
(`foldLoL_mle_bits`), and the split coordinate's weight `(b ? u : 1)` folds to `cff(c)`
(`foldLoL_split`). These two facts carry the claim's weight from one level to the next.
-/

namespace FlockSoundness

open Game Model Accounting
open scoped ENNReal

section Fold

variable {L : Type} [Field L]

theorem foldLo_eq' (t : L) (u : ℕ → L) (y : ℕ) :
    foldLo t u y = (1 + t) * u (2 * y) + t * u (2 * y + 1) := by
  unfold foldLo; ring

/-- Folding by the challenges `ch` is the `eq(ch, ·)`-weighted sum over the folded bits. -/
theorem foldLoL_eq_sum : ∀ (ch : List L) (u : ℕ → L) (y : ℕ),
    foldLoL ch u y = ∑ x ∈ Finset.range (2 ^ ch.length), eqList ch x * u (x + 2 ^ ch.length * y)
  | [], u, y => by simp [foldLoL, eqList]
  | r :: rs, u, y => by
      rw [foldLoL, foldLoL_eq_sum rs, List.length_cons, sum_range_pairs]
      refine Finset.sum_congr rfl fun x _ => ?_
      rw [foldLo_eq', eqList_cons, eqList_cons]
      have h0 : (2 * x).testBit 0 = false := by simp [Nat.testBit_zero]
      have h1 : (2 * x + 1).testBit 0 = true := by simp [Nat.testBit_zero]
      rw [h0, h1, show 2 * x / 2 = x by omega, show (2 * x + 1) / 2 = x by omega]
      simp only [Bool.false_eq_true, if_false, if_true]
      rw [show 2 * (x + 2 ^ rs.length * y) + 1 = 2 * x + 1 + 2 ^ (rs.length + 1) * y by ring,
        show 2 * (x + 2 ^ rs.length * y) = 2 * x + 2 ^ (rs.length + 1) * y by ring]
      ring

theorem foldLoL_add (ch : List L) (u v : ℕ → L) (y : ℕ) :
    foldLoL ch (fun z => u z + v z) y = foldLoL ch u y + foldLoL ch v y := by
  simp only [foldLoL_eq_sum, mul_add, Finset.sum_add_distrib]

theorem foldLoL_mul_left (ch : List L) (a : L) (u : ℕ → L) (y : ℕ) :
    foldLoL ch (fun z => a * u z) y = a * foldLoL ch u y := by
  simp only [foldLoL_eq_sum, Finset.mul_sum]
  exact Finset.sum_congr rfl fun x _ => by ring

theorem foldLoL_sum {ι : Type} (s : Finset ι) (ch : List L) (u : ι → ℕ → L) (y : ℕ) :
    foldLoL ch (fun z => ∑ i ∈ s, u i z) y = ∑ i ∈ s, foldLoL ch (u i) y := by
  simp only [foldLoL_eq_sum, Finset.mul_sum]
  exact Finset.sum_comm

theorem foldLoL_list_sum {ι : Type} (l : List ι) (ch : List L) (u : ι → ℕ → L) (y : ℕ) :
    foldLoL ch (fun z => (l.map fun i => u i z).sum) y = (l.map fun i => foldLoL ch (u i) y).sum := by
  induction l with
  | nil => simp [foldLoL_eq_sum]
  | cons a l ih =>
      simp only [List.map_cons, List.sum_cons]
      rw [foldLoL_add ch (u a) (fun z => (l.map fun i => u i z).sum), ih]

theorem foldLoL_congr (ch : List L) (u v : ℕ → L) (y : ℕ)
    (h : ∀ x < 2 ^ ch.length, u (x + 2 ^ ch.length * y) = v (x + 2 ^ ch.length * y)) :
    foldLoL ch u y = foldLoL ch v y := by
  simp only [foldLoL_eq_sum]
  exact Finset.sum_congr rfl fun x hx => by rw [h x (Finset.mem_range.1 hx)]

end Fold

/-- **The split coordinate folds to `cff`.** The weight `(b ? u : 1)·w(y/2)` of a split table,
folded first at `c`, is `cff(c)` times `w` folded by the rest. -/
theorem foldLoL_split {F K : Type} [Field F] [Field K] (A : Arith F K) (c : K) (cs : List K)
    (w : ℕ → K) (y : ℕ) :
    foldLoL (c :: cs) (fun z => (if z.testBit 0 then A.u else 1) * w (z / 2)) y =
      cff A c * foldLoL cs w y := by
  rw [foldLoL, ← foldLoL_mul_left]
  congr 1
  funext z
  have h0 : (2 * z).testBit 0 = false := by simp [Nat.testBit_zero]
  have h1 : (2 * z + 1).testBit 0 = true := by simp [Nat.testBit_zero]
  simp only [foldLo, h0, h1, show 2 * z / 2 = z by omega, show (2 * z + 1) / 2 = z by omega,
    Bool.false_eq_true, if_false, if_true, cff]
  ring

/-! ## Points with open bits -/

section Points

variable {K : Type} [Field K]

theorem eqList_append (l₁ l₂ : List K) (j : ℕ) :
    eqList (l₁ ++ l₂) j = eqList l₁ j * eqList l₂ (j / 2 ^ l₁.length) := by
  induction l₁ generalizing j with
  | nil => simp [eqList]
  | cons a l ih =>
      rw [List.cons_append, eqList_cons, eqList_cons, ih, List.length_cons, pow_succ,
        Nat.div_div_eq_div_mul, mul_comm (2 ^ l.length) 2]
      ring

theorem bitsK_succ_two_mul (m z : ℕ) :
    bitsK (K := K) (m + 1) (2 * z) = 0 :: bitsK m z := by
  simp only [bitsK, List.range_succ_eq_map, List.map_cons, List.map_map]
  congr 1
  · simp [Nat.testBit_zero]
  · refine List.map_congr_left fun j _ => ?_
    simp only [Function.comp, Nat.testBit_succ, show 2 * z / 2 = z by omega]

theorem bitsK_succ_two_mul_add_one (m z : ℕ) :
    bitsK (K := K) (m + 1) (2 * z + 1) = 1 :: bitsK m z := by
  simp only [bitsK, List.range_succ_eq_map, List.map_cons, List.map_map]
  congr 1
  · simp [Nat.testBit_zero]
  · refine List.map_congr_left fun j _ => ?_
    simp only [Function.comp, Nat.testBit_succ, show (2 * z + 1) / 2 = z by omega]

theorem length_bitsK (m y : ℕ) : (bitsK (K := K) m y).length = m := by simp [bitsK]

/-- **Folding open bits.** Reading `eq(·, w)` at the open bits of the point and folding them by
`ch` reads it at `ch` followed by the still-open bits. -/
theorem foldLoL_eqList_bits [CharP K 2] : ∀ (ch : List K) (m : ℕ), ch.length ≤ m →
    ∀ (w x : ℕ), foldLoL ch (fun y => eqList (bitsK m y) w) x =
      eqList (ch ++ bitsK (m - ch.length) x) w
  | [], m, _, w, x => by simp [foldLoL]
  | c :: cs, m, hm, w, x => by
      obtain ⟨m', rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by simp at hm; omega⟩
      rw [foldLoL]
      have hstep : foldLo c (fun y => eqList (bitsK (m' + 1) y) w) =
          fun z => (if w.testBit 0 then c else 1 + c) * eqList (bitsK m' z) (w / 2) := by
        funext z
        rw [foldLo, bitsK_succ_two_mul, bitsK_succ_two_mul_add_one, eqList_cons, eqList_cons]
        by_cases hw : w.testBit 0
        · simp only [hw, if_true]; ring
        · simp only [hw, Bool.false_eq_true, if_false]
          have h11 : (1 : K) + 1 = 0 := Rounds.add_self 1
          rw [h11]; ring
      rw [hstep, foldLoL_mul_left, foldLoL_eqList_bits cs m' (by simp at hm; omega), List.cons_append,
        eqList_cons]
      congr 3
      simp only [List.length_cons, Nat.add_sub_add_right]

/-- The same with a fixed prefix of challenges. -/
theorem foldLoL_eqList_fixed_bits [CharP K 2] (fixed ch : List K) (m : ℕ) (hm : ch.length ≤ m)
    (w x : ℕ) :
    foldLoL ch (fun y => eqList (fixed ++ bitsK m y) w) x =
      eqList (fixed ++ ch ++ bitsK (m - ch.length) x) w := by
  simp only [eqList_append]
  rw [foldLoL_mul_left, foldLoL_eqList_bits ch m hm, eqList_append, List.length_append, pow_add,
    ← Nat.div_div_eq_div_mul]
  ring

/-- A multilinear extension read at a point: `Σ_w c(w)·eq(p, w)`. -/
def mleAt (N : ℕ) (c : ℕ → K) (p : List K) : K :=
  ∑ w ∈ Finset.range (2 ^ N), c w * eqList p w

/-- **Folding a multilinear extension's open bits.** -/
theorem foldLoL_mle_bits [CharP K 2] (N : ℕ) (c : ℕ → K) (fixed ch : List K) (m : ℕ)
    (hm : ch.length ≤ m) (x : ℕ) :
    foldLoL ch (fun y => mleAt N c (fixed ++ bitsK m y)) x =
      mleAt N c (fixed ++ ch ++ bitsK (m - ch.length) x) := by
  unfold mleAt
  rw [foldLoL_sum]
  exact Finset.sum_congr rfl fun w _ => by
    rw [foldLoL_mul_left, foldLoL_eqList_fixed_bits fixed ch m hm]

end Points

/-! ## List products as range products -/

section Lists

variable {K : Type} [Field K]

theorem prod_map_range (n : ℕ) (f : ℕ → K) :
    ((List.range n).map f).prod = ∏ j ∈ Finset.range n, f j := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.map_append, List.prod_append, ih, Finset.prod_range_succ]; simp

theorem prod_map_zip (h : K × K → K) : ∀ (l₁ l₂ : List K),
    ((List.zip l₁ l₂).map h).prod =
      ∏ i ∈ Finset.range (min l₁.length l₂.length), h (l₁.getD i 0, l₂.getD i 0)
  | [], _ => by simp
  | _ :: _, [] => by simp
  | a :: l₁, b :: l₂ => by
      rw [List.zip_cons_cons, List.map_cons, List.prod_cons, prod_map_zip h l₁ l₂,
        List.length_cons, List.length_cons, Nat.add_min_add_right, Finset.prod_range_succ']
      simp [mul_comm]

theorem prod_map_zipIdx (h : K × ℕ → K) : ∀ (l : List K) (k : ℕ),
    ((l.zipIdx k).map h).prod = ∏ i ∈ Finset.range l.length, h (l.getD i 0, k + i)
  | [], _ => by simp
  | a :: l, k => by
      rw [List.zipIdx_cons, List.map_cons, List.prod_cons, prod_map_zipIdx h l (k + 1),
        List.length_cons, Finset.prod_range_succ']
      simp only [List.getD_cons_succ, List.getD_cons_zero, add_zero]
      rw [mul_comm]
      congr 1
      exact Finset.prod_congr rfl fun i _ => by rw [show k + 1 + i = k + (i + 1) by ring]

theorem getD_bitsK (m y j : ℕ) (hj : j < m) :
    (bitsK (K := K) m y).getD j 0 = if y.testBit j then 1 else 0 := by
  simp [bitsK, List.getD_eq_getElem?_getD, hj]

/-- The agreement product `Π_i (1 + z_i + p_i)` of two points of equal length is `Σ_w eq(z,w)·eq(p,w)`. -/
theorem sum_eqList_eqList [CharP K 2] : ∀ (z p : List K), z.length = p.length →
    ∑ w ∈ Finset.range (2 ^ p.length), eqList z w * eqList p w =
      ∏ i ∈ Finset.range p.length, (1 + z.getD i 0 + p.getD i 0)
  | [], [], _ => by simp [eqList]
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | a :: z, c :: p, h => by
      rw [List.length_cons, sum_range_pairs, Finset.prod_range_succ']
      simp only [List.getD_cons_succ, List.getD_cons_zero]
      rw [← sum_eqList_eqList z p (by simpa using h), Finset.sum_mul]
      refine Finset.sum_congr rfl fun w _ => ?_
      have h0 : (2 * w).testBit 0 = false := by simp [Nat.testBit_zero]
      have h1 : (2 * w + 1).testBit 0 = true := by simp [Nat.testBit_zero]
      simp only [eqList_cons, h0, h1, show 2 * w / 2 = w by omega, show (2 * w + 1) / 2 = w by omega,
        Bool.false_eq_true, if_false, if_true, List.getD_cons_zero]
      have h2 : (2 : K) = 0 := CharTwo.two_eq_zero
      linear_combination (a * c * eqList z w * eqList p w) * h2

end Lists

/-! ## The final check's weight at any stage -/

section Res

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- The challenges on the original variables so far. -/
def origOf (initial : List K) (levels : List (List K)) : List K :=
  initial ++ (levels.map List.tail).flatten

/-- An OOD claim's fixed challenges: those on its table's variables so far. -/
def oodFixed (initial : List K) (levels : List (List K)) (o : OodCtx F) : List K :=
  match o.split with
  | none => origOf initial levels
  | some a => (levels.getD a []) ++ ((levels.drop (a + 1)).map List.tail).flatten

def oodScale (A : Arith F K) (levels : List (List K)) (o : OodCtx F) : K :=
  match o.split with
  | none => cffFrom A levels 0
  | some a => cffFrom A levels (a + 1)

/-- An OOD claim's term of the final check's weight. -/
def oodTerm (A : Arith F K) (initial : List K) (levels : List (List K)) (ext : ℕ) (o : OodCtx F)
    (y : ℕ) : K :=
  A.embed o.β * oodScale A levels o
    * ((List.zip (o.z.map A.embed) (oodFixed initial levels o)).map fun (zt, ft) => 1 + zt + ft).prod
    * ((List.range ext).map fun j =>
        let zj := A.embed (o.z.getD ((oodFixed initial levels o).length + j) 0)
        if y.testBit j then zj else 1 + zj).prod

/-- A consistency claim's fixed challenges: those on its folded columns so far. -/
def consFixed (levels : List (List K)) (c : ConsCtx F) : List K :=
  ((levels.drop c.start).map List.tail).flatten

/-- A consistency claim's term of the final check's weight. -/
def consTerm (A : Arith F K) (levels : List (List K)) (ext : ℕ) (c : ConsCtx F) (y : ℕ) : K :=
  A.embed c.β * cffFrom A levels c.start
    * ((c.positions.zipIdx).map fun (p, j) =>
        A.embed (eqList c.α j)
          * (((consFixed levels c).zipIdx).map fun (fi, i) =>
              1 + fi * (1 + A.embed (A.What i (A.omega p)))).prod
          * ((List.range ext).map fun i =>
              if y.testBit i then A.embed (A.What ((consFixed levels c).length + i) (A.omega p))
              else 1).prod).sum

/-- **The claim's weight on the open variables** at the stage where `levels` have been folded:
the final check's `res` with the lists so far. -/
def resAt (A : Arith F K) (n : ℕ) (b : ℕ → F) (initial : List K) (levels : List (List K))
    (oods : List (OodCtx F)) (cons : List (ConsCtx F)) (ext : ℕ) (y : ℕ) : K :=
  bHat A n b (origOf initial levels ++ bitsK ext y) * cffFrom A levels 0
    + (oods.map fun o => oodTerm A initial levels ext o y).sum
    + (cons.map fun c => consTerm A levels ext c y).sum

theorem finalCheck_eq_resAt (A : Arith F K) (n : ℕ) (b : ℕ → F) (initial : List K)
    (levels : List (List K)) (oods : List (OodCtx F)) (cons : List (ConsCtx F)) (ext : ℕ)
    (yr : Fin (2 ^ (ext + 1)) → F) (s : K) :
    finalCheck A n b initial levels oods cons ext yr s =
      (decide ((origOf initial levels).length + ext = n)
        && decide ((∑ y ∈ Finset.range (2 ^ ext),
          resAt A n b initial levels oods cons ext y *
            (A.embed (yr ⟨(2 * y) % 2 ^ (ext + 1), Nat.mod_lt _ (Nat.two_pow_pos _)⟩)
              + A.u * A.embed (yr ⟨(2 * y + 1) % 2 ^ (ext + 1), Nat.mod_lt _ (Nat.two_pow_pos _)⟩)))
            = s)) := rfl

/-! ### Each term as a multilinear extension read at `fixed ++ bits(y)` -/

theorem oodTerm_eq_mle [CharP K 2] (A : Arith F K) (initial : List K) (levels : List (List K))
    (ext : ℕ) (o : OodCtx F) (y : ℕ)
    (hz : o.z.length = (oodFixed initial levels o).length + ext) :
    oodTerm A initial levels ext o y =
      A.embed o.β * oodScale A levels o *
        mleAt o.z.length (eqList (o.z.map A.embed)) (oodFixed initial levels o ++ bitsK ext y) := by
  set fx := oodFixed initial levels o
  have hP : (fx ++ bitsK (K := K) ext y).length = o.z.length := by
    rw [List.length_append, length_bitsK, hz]
  unfold oodTerm mleAt
  rw [← hP, sum_eqList_eqList _ _ (by rw [List.length_map, hP]), mul_assoc, prod_map_zip, prod_map_range,
    List.length_map, List.length_append, length_bitsK, show min o.z.length fx.length = fx.length by omega,
    Finset.prod_range_add]
  congr 1
  congr 1
  · refine Finset.prod_congr rfl fun i hi => ?_
    rw [List.getD_append _ _ _ _ (Finset.mem_range.1 hi)]
  · refine Finset.prod_congr rfl fun j hj => ?_
    have hj' := Finset.mem_range.1 hj
    rw [List.getD_append_right _ _ _ _ (by omega), show fx.length + j - fx.length = j by omega,
      getD_bitsK ext y j hj']
    have h2 : (2 : K) = 0 := CharTwo.two_eq_zero
    have hzj : (o.z.map A.embed).getD (fx.length + j) 0 = A.embed (o.z.getD (fx.length + j) 0) := by
      simp only [List.getD_eq_getElem?_getD, List.getElem?_map]
      cases o.z[fx.length + j]? <;> simp
    rw [hzj]
    split_ifs
    · linear_combination -h2
    · rw [add_zero]

theorem consTerm_eq_mle [CharP K 2] (A : Arith F K) (levels : List (List K)) (ext : ℕ)
    (c : ConsCtx F) (y : ℕ) :
    consTerm A levels ext c y = A.embed c.β * cffFrom A levels c.start *
      ((c.positions.zipIdx).map fun pj => A.embed (eqList c.α pj.2) *
        mleAt ((consFixed levels c).length + ext)
          (fun w => A.embed (A.Xhat w ((consFixed levels c).length + ext) (A.omega pj.1)))
          (consFixed levels c ++ bitsK ext y)).sum := by
  unfold consTerm
  congr 1
  refine congrArg List.sum (List.map_congr_left fun pj _ => ?_)
  obtain ⟨p, j⟩ := pj
  set fx := consFixed levels c
  have hP : (fx ++ bitsK (K := K) ext y).length = fx.length + ext := by
    rw [List.length_append, length_bitsK]
  dsimp only
  rw [mul_assoc]
  congr 1
  unfold mleAt
  rw [← hP]
  beta_reduce
  simp_rw [mul_comm (A.embed (A.Xhat _ _ _)) (eqList _ _)]
  rw [sum_eqList_Xhat, hP, Finset.prod_range_add, prod_map_zipIdx, prod_map_range]
  congr 1
  · refine Finset.prod_congr rfl fun i hi => ?_
    rw [List.getD_append _ _ _ _ (Finset.mem_range.1 hi), zero_add]
  · refine Finset.prod_congr rfl fun j hj => ?_
    have hj' := Finset.mem_range.1 hj
    rw [List.getD_append_right _ _ _ _ (by omega), show fx.length + j - fx.length = j by omega,
      getD_bitsK ext y j hj']
    have h2 : (2 : K) = 0 := CharTwo.two_eq_zero
    split_ifs
    · linear_combination -h2
    · ring

/-! ### Appending a level's challenges -/

theorem origOf_snoc (initial : List K) (levels : List (List K)) (ch : List K) :
    origOf initial (levels ++ [ch]) = origOf initial levels ++ ch.tail := by
  simp [origOf, List.map_append, List.flatten_append, List.append_assoc]

theorem cffFrom_snoc (A : Arith F K) (levels : List (List K)) (c : K) (cs : List K) (a : ℕ)
    (ha : a ≤ levels.length) :
    cffFrom A (levels ++ [c :: cs]) a = cffFrom A levels a * cff A c := by
  simp [cffFrom, List.drop_append_of_le_length ha, List.map_append, List.prod_append]

theorem cffFrom_snoc_top (A : Arith F K) (levels : List (List K)) (ch : List K) :
    cffFrom A (levels ++ [ch]) (levels.length + 1) = 1 := by
  simp [cffFrom, List.drop_eq_nil_of_le]

/-- An OOD claim introduced before this stage. -/
def OodOld (levels : List (List K)) (o : OodCtx F) : Prop :=
  o.split = none ∨ ∃ a < levels.length, o.split = some a

theorem oodFixed_snoc (initial : List K) (levels : List (List K)) (ch : List K) (o : OodCtx F)
    (ho : OodOld levels o) :
    oodFixed initial (levels ++ [ch]) o = oodFixed initial levels o ++ ch.tail := by
  rcases ho with h | ⟨a, ha, h⟩
  · simp only [oodFixed, h, origOf_snoc]
  · simp only [oodFixed, h, List.getD_eq_getElem?_getD, List.getElem?_append_left ha,
      List.drop_append_of_le_length (show a + 1 ≤ levels.length by omega), List.map_append,
      List.flatten_append, List.map_cons, List.map_nil, List.flatten_cons, List.flatten_nil,
      List.append_nil, List.append_assoc]

theorem oodScale_snoc (A : Arith F K) (levels : List (List K)) (c : K) (cs : List K)
    (o : OodCtx F) (ho : OodOld levels o) :
    oodScale A (levels ++ [c :: cs]) o = oodScale A levels o * cff A c := by
  rcases ho with h | ⟨a, ha, h⟩
  · simp only [oodScale, h, cffFrom_snoc A levels c cs 0 (Nat.zero_le _)]
  · simp only [oodScale, h, cffFrom_snoc A levels c cs (a + 1) (by omega)]

theorem oodFixed_new (initial : List K) (levels : List (List K)) (ch : List K) (o : OodCtx F)
    (ho : o.split = some levels.length) : oodFixed initial (levels ++ [ch]) o = ch := by
  simp [oodFixed, ho, List.drop_eq_nil_of_le]

theorem oodScale_new (A : Arith F K) (levels : List (List K)) (ch : List K) (o : OodCtx F)
    (ho : o.split = some levels.length) : oodScale A (levels ++ [ch]) o = 1 := by
  simp only [oodScale, ho, cffFrom_snoc_top]

theorem consFixed_snoc (levels : List (List K)) (ch : List K) (c : ConsCtx F)
    (hc : c.start ≤ levels.length) :
    consFixed (levels ++ [ch]) c = consFixed levels c ++ ch.tail := by
  simp [consFixed, List.drop_append_of_le_length hc, List.map_append, List.flatten_append]

/-! ### Each term across one level -/

theorem fold_base [CharP K 2] (A : Arith F K) (n : ℕ) (b : ℕ → F) (initial : List K)
    (levels : List (List K)) (c : K) (cs : List K) (ext ext' : ℕ) (hext : ext = cs.length + ext')
    (y' : ℕ) :
    cff A c * foldLoL cs (fun y => bHat A n b (origOf initial levels ++ bitsK ext y) *
        cffFrom A levels 0) y' =
      bHat A n b (origOf initial (levels ++ [c :: cs]) ++ bitsK ext' y') *
        cffFrom A (levels ++ [c :: cs]) 0 := by
  have h := foldLoL_mle_bits n (fun y => A.embed (b y)) (origOf initial levels) cs ext (by omega) y'
  simp only [mleAt] at h
  simp only [bHat]
  rw [show (fun y => (∑ w ∈ Finset.range (2 ^ n), A.embed (b w) *
        eqList (origOf initial levels ++ bitsK ext y) w) * cffFrom A levels 0) =
      fun y => cffFrom A levels 0 * ∑ w ∈ Finset.range (2 ^ n), A.embed (b w) *
        eqList (origOf initial levels ++ bitsK ext y) w from funext fun _ => mul_comm _ _,
    foldLoL_mul_left, h, origOf_snoc, cffFrom_snoc A levels c cs 0 (Nat.zero_le _),
    show ext - cs.length = ext' by omega]
  simp only [List.tail_cons]
  ring

theorem fold_oodOld [CharP K 2] (A : Arith F K) (initial : List K) (levels : List (List K))
    (c : K) (cs : List K) (ext ext' : ℕ) (hext : ext = cs.length + ext') (o : OodCtx F)
    (ho : OodOld levels o) (hz : o.z.length = (oodFixed initial levels o).length + ext) (y' : ℕ) :
    cff A c * foldLoL cs (oodTerm A initial levels ext o) y' =
      oodTerm A initial (levels ++ [c :: cs]) ext' o y' := by
  have hz' : o.z.length = (oodFixed initial (levels ++ [c :: cs]) o).length + ext' := by
    rw [oodFixed_snoc initial levels _ o ho, List.length_append, List.tail_cons]; omega
  rw [oodTerm_eq_mle A initial (levels ++ [c :: cs]) ext' o y' hz',
    show oodTerm A initial levels ext o = fun y => A.embed o.β * oodScale A levels o *
      mleAt o.z.length (eqList (o.z.map A.embed)) (oodFixed initial levels o ++ bitsK ext y) from
      funext fun y => oodTerm_eq_mle A initial levels ext o y hz,
    foldLoL_mul_left, foldLoL_mle_bits _ _ _ cs ext (by omega), oodFixed_snoc initial levels _ o ho,
    oodScale_snoc A levels c cs o ho, show ext - cs.length = ext' by omega]
  simp only [List.tail_cons]
  ring

theorem fold_cons [CharP K 2] (A : Arith F K) (levels : List (List K)) (c : K) (cs : List K)
    (ext ext' : ℕ) (hext : ext = cs.length + ext') (cc : ConsCtx F) (hc : cc.start ≤ levels.length)
    (y' : ℕ) :
    cff A c * foldLoL cs (consTerm A levels ext cc) y' = consTerm A (levels ++ [c :: cs]) ext' cc y' := by
  have hN : (consFixed (levels ++ [c :: cs]) cc).length + ext' = (consFixed levels cc).length + ext := by
    rw [consFixed_snoc levels _ cc hc, List.length_append, List.tail_cons]; omega
  rw [consTerm_eq_mle A (levels ++ [c :: cs]) ext' cc y', hN,
    show consTerm A levels ext cc = fun y => A.embed cc.β * cffFrom A levels cc.start *
      ((cc.positions.zipIdx).map fun pj => A.embed (eqList cc.α pj.2) *
        mleAt ((consFixed levels cc).length + ext)
          (fun w => A.embed (A.Xhat w ((consFixed levels cc).length + ext) (A.omega pj.1)))
          (consFixed levels cc ++ bitsK ext y)).sum from
      funext fun y => consTerm_eq_mle A levels ext cc y,
    foldLoL_mul_left, foldLoL_list_sum, consFixed_snoc levels _ cc hc,
    cffFrom_snoc A levels c cs cc.start hc]
  simp only [foldLoL_mul_left]
  rw [show ext' = ext - cs.length by omega]
  simp only [foldLoL_mle_bits _ _ _ cs ext (show cs.length ≤ ext by omega), List.tail_cons]
  ring

/-- `eq(bits(y), w)` on the hypercube is the indicator of `w = y`. -/
theorem eqList_bitsK [CharP K 2] (N y w : ℕ) (hy : y < 2 ^ N) (hw : w < 2 ^ N) :
    eqList (bitsK (K := K) N y) w = if w = y then 1 else 0 := by
  unfold eqList
  rw [prod_map_zipIdx, length_bitsK]
  simp only [zero_add]
  have h2 : (1 : K) + 1 = 0 := Rounds.add_self 1
  by_cases hwy : w = y
  · subst hwy
    rw [if_pos rfl]
    refine Finset.prod_eq_one fun i hi => ?_
    rw [getD_bitsK N w i (Finset.mem_range.1 hi)]
    split_ifs <;> simp_all
  · rw [if_neg hwy]
    obtain ⟨i, hi⟩ : ∃ i, w.testBit i ≠ y.testBit i := by
      by_contra hno
      exact hwy (Nat.eq_of_testBit_eq fun i => by
        by_contra h
        exact hno ⟨i, h⟩)
    have hiN : i < N := by
      by_contra hge'
      have hge : N ≤ i := Nat.le_of_not_lt hge'
      have h1 : w.testBit i = false := Nat.testBit_eq_false_of_lt (lt_of_lt_of_le hw
        (Nat.pow_le_pow_right (by norm_num) hge))
      have h2' : y.testBit i = false := Nat.testBit_eq_false_of_lt (lt_of_lt_of_le hy
        (Nat.pow_le_pow_right (by norm_num) hge))
      exact hi (h1.trans h2'.symm)
    refine Finset.prod_eq_zero (Finset.mem_range.2 hiN) ?_
    rw [getD_bitsK N y i hiN]
    cases hw' : w.testBit i <;> cases hy' : y.testBit i <;> simp_all

theorem fold_oodNew [CharP K 2] (A : Arith F K) (initial : List K) (levels : List (List K))
    (ch : List K) (ext' : ℕ) (o : OodCtx F) (ho : o.split = some levels.length)
    (hz : o.z.length = ch.length + ext') (y' : ℕ) (hy' : y' < 2 ^ ext') :
    foldLoL ch (fun y => A.embed o.β * eqList (o.z.map A.embed) y) y' =
      oodTerm A initial (levels ++ [ch]) ext' o y' := by
  have hz' : o.z.length = (oodFixed initial (levels ++ [ch]) o).length + ext' := by
    rw [oodFixed_new initial levels ch o ho, hz]
  rw [oodTerm_eq_mle A initial (levels ++ [ch]) ext' o y' hz', oodFixed_new initial levels ch o ho,
    oodScale_new A levels ch o ho, foldLoL_mul_left, mul_one]
  congr 1
  have hself : ∀ y < 2 ^ o.z.length, eqList (o.z.map A.embed) y =
      mleAt o.z.length (eqList (o.z.map A.embed)) ([] ++ bitsK o.z.length y) := by
    intro y hy
    unfold mleAt
    rw [List.nil_append, Finset.sum_eq_single_of_mem y (Finset.mem_range.2 hy)]
    · rw [eqList_bitsK _ y y hy hy, if_pos rfl, mul_one]
    · intro w hw hwy
      rw [eqList_bitsK _ y w hy (Finset.mem_range.1 hw), if_neg hwy, mul_zero]
  rw [foldLoL_congr ch _ (fun y => mleAt o.z.length (eqList (o.z.map A.embed))
      ([] ++ bitsK o.z.length y)) y' fun x hx => hself _ (by
        rw [hz, pow_add]
        calc x + 2 ^ ch.length * y' < 2 ^ ch.length + 2 ^ ch.length * y' := by omega
          _ = 2 ^ ch.length * (y' + 1) := by ring
          _ ≤ 2 ^ ch.length * 2 ^ ext' := Nat.mul_le_mul_left _ hy'),
    foldLoL_mle_bits _ _ [] ch o.z.length (by omega), List.nil_append,
    show o.z.length - ch.length = ext' by omega]

/-! ### One level -/

/-- **The weight on a split table** before its lanes are folded: the previous stage's weight on
the split coordinate's two sides, plus the OOD claims made on this table. -/
def preW (A : Arith F K) (n : ℕ) (b : ℕ → F) (initial : List K) (levels : List (List K))
    (oods : List (OodCtx F)) (cons : List (ConsCtx F)) (ext : ℕ) (y : ℕ) : K :=
  (if y.testBit 0 then A.u else 1) *
      resAt A n b initial levels (oods.filter fun o => o.split ≠ some levels.length) cons ext (y / 2)
    + ((oods.filter fun o => o.split = some levels.length).map fun o =>
        A.embed o.β * eqList (o.z.map A.embed) y).sum

/-- The lists' shape after a level's lanes are folded. -/
structure PostOK (initial : List K) (levels : List (List K)) (oods : List (OodCtx F))
    (cons : List (ConsCtx F)) (ext : ℕ) : Prop where
  ood : ∀ o ∈ oods, OodOld levels o ∧ o.z.length = (oodFixed initial levels o).length + ext
  cons : ∀ c ∈ cons, c.start ≤ levels.length

/-- The lists' shape before a level's lanes are folded: the OOD claims on this table are new. -/
structure PreOK (initial : List K) (levels : List (List K)) (oods : List (OodCtx F))
    (cons : List (ConsCtx F)) (ext : ℕ) : Prop where
  ood : ∀ o ∈ oods, (o.split = some levels.length ∧ o.z.length = ext + 1) ∨
    (OodOld levels o ∧ o.z.length = (oodFixed initial levels o).length + ext)
  cons : ∀ c ∈ cons, c.start ≤ levels.length

theorem list_sum_filter_split {ι : Type} (l : List ι) (p : ι → Bool) (f : ι → K) :
    (l.map f).sum = ((l.filter p).map f).sum + ((l.filter fun i => !p i).map f).sum := by
  induction l with
  | nil => simp
  | cons a l ih => cases h : p a <;> simp [List.filter_cons, h, ih] <;> ring

theorem list_sum_mul_left {ι : Type} (l : List ι) (a : K) (f : ι → K) :
    a * (l.map f).sum = (l.map fun i => a * f i).sum := by
  induction l with
  | nil => simp
  | cons x l ih => simp [mul_add, ih]

/-- **A level's fold identity.** Folding the split table's lanes carries the weight to the next
stage's `resAt`. -/
theorem foldLoL_preW [CharP K 2] (A : Arith F K) (n : ℕ) (b : ℕ → F) (initial : List K)
    (levels : List (List K)) (oods : List (OodCtx F)) (cons : List (ConsCtx F)) (ext : ℕ)
    (c : K) (cs : List K) (ext' : ℕ) (hext : ext = cs.length + ext')
    (hok : PreOK initial levels oods cons ext) (y' : ℕ) (hy' : y' < 2 ^ ext') :
    foldLoL (c :: cs) (preW A n b initial levels oods cons ext) y' =
      resAt A n b initial (levels ++ [c :: cs]) oods cons ext' y' := by
  unfold preW
  rw [foldLoL_add, foldLoL_split, foldLoL_list_sum]
  unfold resAt
  rw [foldLoL_add, foldLoL_add, foldLoL_list_sum, foldLoL_list_sum, mul_add, mul_add,
    fold_base A n b initial levels c cs ext ext' hext, list_sum_mul_left, list_sum_mul_left,
    list_sum_filter_split oods (fun o => decide (o.split = some levels.length))]
  have hold : ∀ o ∈ oods.filter (fun o => o.split ≠ some levels.length),
      cff A c * foldLoL cs (fun y => oodTerm A initial levels ext o y) y' =
        oodTerm A initial (levels ++ [c :: cs]) ext' o y' := by
    intro o ho
    obtain ⟨hmem, hne⟩ := List.mem_filter.1 ho
    rcases hok.ood o hmem with ⟨hs, _⟩ | ⟨hold, hz⟩
    · exact absurd hs (by simpa using hne)
    · exact fold_oodOld A initial levels c cs ext ext' hext o hold hz y'
  have hnew : ∀ o ∈ oods.filter (fun o => o.split = some levels.length),
      foldLoL (c :: cs) (fun y => A.embed o.β * eqList (o.z.map A.embed) y) y' =
        oodTerm A initial (levels ++ [c :: cs]) ext' o y' := by
    intro o ho
    obtain ⟨hmem, heq⟩ := List.mem_filter.1 ho
    have heq' : o.split = some levels.length := by simpa using heq
    rcases hok.ood o hmem with ⟨_, hz⟩ | ⟨hold, _⟩
    · exact fold_oodNew A initial levels (c :: cs) ext' o heq' (by simp; omega) y' hy'
    · rcases hold with h | ⟨a, ha, h⟩ <;> simp_all
  have hcons : ∀ cc ∈ cons, cff A c * foldLoL cs (fun y => consTerm A levels ext cc y) y' =
      consTerm A (levels ++ [c :: cs]) ext' cc y' := fun cc hcc =>
    fold_cons A levels c cs ext ext' hext cc (hok.cons cc hcc) y'
  rw [List.map_congr_left hold, List.map_congr_left hnew, List.map_congr_left hcons]
  have hfilt : oods.filter (fun o => !decide (o.split = some levels.length)) =
      oods.filter (fun o => o.split ≠ some levels.length) := by
    refine List.filter_congr fun o _ => ?_
    simp
  rw [hfilt]
  ring

theorem postOK_of_preOK (initial : List K) (levels : List (List K)) (oods : List (OodCtx F))
    (cons : List (ConsCtx F)) (ext : ℕ) (c : K) (cs : List K) (ext' : ℕ)
    (hext : ext = cs.length + ext') (hok : PreOK initial levels oods cons ext) :
    PostOK initial (levels ++ [c :: cs]) oods cons ext' where
  ood o ho := by
    rcases hok.ood o ho with ⟨hs, hz⟩ | ⟨hold, hz⟩
    · refine ⟨Or.inr ⟨levels.length, by simp, hs⟩, ?_⟩
      rw [oodFixed_new initial levels _ o hs, hz]; simp; omega
    · refine ⟨?_, ?_⟩
      · rcases hold with h | ⟨a, ha, h⟩
        · exact Or.inl h
        · exact Or.inr ⟨a, by simp; omega, h⟩
      · rw [oodFixed_snoc initial levels _ o hold, hz, List.length_append, List.tail_cons]; omega
  cons cc hcc := by simp; have := hok.cons cc hcc; omega

theorem preOK_of_postOK (initial : List K) (levels : List (List K)) (oods : List (OodCtx F))
    (cons : List (ConsCtx F)) (ext : ℕ) (hok : PostOK initial levels oods cons ext)
    (new : List (OodCtx F))
    (hnew : ∀ o ∈ new, o.split = some levels.length ∧ o.z.length = ext + 1)
    (cc : ConsCtx F) (hcc : cc.start = levels.length) :
    PreOK initial levels (oods ++ new) (cons ++ [cc]) ext where
  ood o ho := by
    rcases List.mem_append.1 ho with h | h
    · exact Or.inr (hok.ood o h)
    · exact Or.inl (hnew o h)
  cons c hc := by
    rcases List.mem_append.1 hc with h | h
    · exact hok.cons c h
    · rw [List.mem_singleton.1 h, hcc]

/-- The weight's two parts: its sum against a split table is the previous stage's claim on the
table's other side plus the new OOD claims. -/
theorem sum_preW (A : Arith F K) (n : ℕ) (b : ℕ → F) (initial : List K)
    (levels : List (List K)) (oods : List (OodCtx F)) (cons : List (ConsCtx F)) (ext : ℕ)
    (g : ℕ → K) :
    ∑ y ∈ Finset.range (2 ^ (ext + 1)), g y * preW A n b initial levels oods cons ext y =
      ∑ x ∈ Finset.range (2 ^ ext), (g (2 * x) + A.u * g (2 * x + 1)) *
          resAt A n b initial levels (oods.filter fun o => o.split ≠ some levels.length) cons ext x
        + ((oods.filter fun o => o.split = some levels.length).map fun o =>
            A.embed o.β * ∑ y ∈ Finset.range (2 ^ (ext + 1)), g y * eqList (o.z.map A.embed) y).sum := by
  unfold preW
  simp only [mul_add, Finset.sum_add_distrib]
  congr 1
  · rw [sum_range_pairs]
    refine Finset.sum_congr rfl fun x _ => ?_
    have h0 : (2 * x).testBit 0 = false := by simp [Nat.testBit_zero]
    have h1 : (2 * x + 1).testBit 0 = true := by simp [Nat.testBit_zero]
    simp only [h0, h1, show 2 * x / 2 = x by omega, show (2 * x + 1) / 2 = x by omega,
      Bool.false_eq_true, if_false, if_true]
    ring
  · induction (oods.filter fun o => o.split = some levels.length) with
    | nil => simp
    | cons o l ih =>
        simp only [List.map_cons, List.sum_cons, mul_add, Finset.sum_add_distrib, ih, Finset.mul_sum]
        congr 1
        exact Finset.sum_congr rfl fun y _ => by ring

/-- A consistency claim made at this stage reads the fold's encoding: `Σ_j eq(α)_j·X̂_y(ω_{p_j})`. -/
theorem consTerm_new (A : Arith F K) (levels : List (List K)) (ext : ℕ) (cc : ConsCtx F)
    (hcc : cc.start = levels.length) (y : ℕ) :
    consTerm A levels ext cc y = A.embed cc.β *
      ((cc.positions.zipIdx).map fun pj =>
        A.embed (eqList cc.α pj.2) * A.embed (A.Xhat y ext (A.omega pj.1))).sum := by
  have hfx : consFixed levels cc = [] := by simp [consFixed, hcc, List.drop_eq_nil_of_le]
  have hsc : cffFrom A levels cc.start = 1 := by simp [cffFrom, hcc, List.drop_eq_nil_of_le]
  unfold consTerm
  rw [hsc, mul_one, hfx]
  congr 1
  refine congrArg List.sum (List.map_congr_left fun pj _ => ?_)
  obtain ⟨p, j⟩ := pj
  simp only [List.zipIdx_nil, List.map_nil, List.prod_nil, mul_one, List.length_nil, zero_add]
  congr 1
  rw [prod_map_range, Arith.Xhat, map_prod, ← Fin.prod_univ_eq_prod_range]
  exact Finset.prod_congr rfl fun i _ => by split_ifs <;> simp

theorem resAt_cons_append (A : Arith F K) (n : ℕ) (b : ℕ → F) (initial : List K)
    (levels : List (List K)) (oods : List (OodCtx F)) (cons cons' : List (ConsCtx F)) (ext y : ℕ) :
    resAt A n b initial levels oods (cons ++ cons') ext y =
      resAt A n b initial levels oods cons ext y + (cons'.map fun c => consTerm A levels ext c y).sum := by
  simp [resAt, List.map_append, List.sum_append, add_assoc]

end Res

end FlockSoundness
