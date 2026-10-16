import Proofs.Flock.Claims.GS.Elim
import Proofs.Flock.Claims.GS.Basic
import Proofs.Flock.Claims.Decode

/-!
# Guruswami–Sudan interpolation by Gaussian elimination, counted

At `N` points `(x, y)` with distinct `x`, code dimension `k` and agreement threshold `t` in the regime
`(k+1)·N < t²`, `t ≤ N` (`Reg`), the interpolation step finds `Q(X, Y) ≠ 0` of `(1, k−1)`-weighted degree at most
`D = (k+1)·t − 1` (`degX`) vanishing to order `m = k+1` at every point. The unknowns are the coefficients of the first
`R + 1` monomials of weighted degree at most `D` (`monos`, `sel`), `R = N·m(m+1)/2` (`neqs`) the number of
constraints: one per point and per `(s, t)` with `s + t < m` (`ders`), the `(s, t)` coefficient of `Q(X + x, Y + y)`
(ArkLib's `evalConstraint`), whose entry at the monomial `X^i Y^j` is `x^(i−s)·C(i,s)·y^(j−t)·C(j,t)` (`entry`,
`evalConstraint_monomial`). Gaussian elimination (`kernel`) solves the system, which has more unknowns than
equations (`neqs_lt_numVars`, from ArkLib's count of the unknowns), and `interpTab` is the solution as a coefficient
table on the box `i ≤ D`, `j ≤ L` (`degY`).

`interpPoly_eval_eq_zero` is Guruswami–Sudan's key property, from ArkLib's lemmas (`orderAt_eval_ge`,
`roots_le_degree_of_deg_lt_roots`): `Q(X, P(X)) = 0` for every `P` of degree below `k` that agrees with at least `t`
of the points. `interpCost` counts the step's field operations, `x^e` as `e` multiplications and a binomial
coefficient's image in the field as one operation (`entryCost`), with the elimination's count (`kernelCost`); in the
regime it is at most `20·N^9` (`interpCost_le`).
-/

namespace Claims.GS

open Polynomial

/-! ## The unknowns and the constraints -/

/-- The monomials `X^i Y^j` of `(1, k−1)`-weighted degree at most `D`, as pairs `(i, j)`: ArkLib's
`weightBoundIndices` as a list. -/
def monos (k D : ℕ) : List (ℕ × ℕ) :=
  (List.range (D + 1) ×ˢ List.range (D + 1)).filter fun x => decide (x.1 + (k - 1) * x.2 ≤ D)

theorem mem_monos {k D : ℕ} {x : ℕ × ℕ} : x ∈ monos k D ↔ x.1 ≤ D ∧ x.2 ≤ D ∧ x.1 + (k - 1) * x.2 ≤ D := by
  obtain ⟨a, b⟩ := x
  simp only [monos, List.mem_filter, List.mem_product, List.mem_range, decide_eq_true_eq]
  omega

theorem nodup_monos (k D : ℕ) : (monos k D).Nodup :=
  ((List.nodup_range).product List.nodup_range).filter _

theorem length_monos (k D : ℕ) : (monos k D).length = GuruswamiSudan.numVars k D := by
  rw [GuruswamiSudan.numVars, ← List.toFinset_card_of_nodup (nodup_monos k D)]
  congr 1
  ext x
  simp only [List.mem_toFinset, mem_monos, GuruswamiSudan.weightBoundIndices, Finset.mem_filter,
    Finset.product_eq_sprod, Finset.mem_product, Finset.mem_range]
  omega

/-- The constraint indices `(s, t)`, `s + t < m`: ArkLib's `constraintIndices` as a list. -/
def ders (m : ℕ) : List (ℕ × ℕ) :=
  (List.range m ×ˢ List.range m).filter fun x => decide (x.1 + x.2 < m)

theorem mem_ders {m : ℕ} {x : ℕ × ℕ} : x ∈ ders m ↔ x.1 + x.2 < m := by
  obtain ⟨a, b⟩ := x
  simp only [ders, List.mem_filter, List.mem_product, List.mem_range, decide_eq_true_eq]
  omega

theorem length_ders (m : ℕ) : (ders m).length = m * (m + 1) / 2 := by
  have hnd : (ders m).Nodup := ((List.nodup_range).product List.nodup_range).filter _
  rw [← GuruswamiSudan.card_constraintIndices, ← List.toFinset_card_of_nodup hnd]
  congr 1
  ext x
  simp only [List.mem_toFinset, mem_ders, GuruswamiSudan.constraintIndices, Finset.mem_filter,
    Finset.product_eq_sprod, Finset.mem_product, Finset.mem_range]
  omega

/-- The `(s, t)` coefficient of `X^i Y^j` shifted to `(x, y)`: `x^(i−s)·C(i,s)·y^(j−t)·C(j,t)`. -/
def entry {F : Type} [Field F] (x y : F) (st ij : ℕ × ℕ) : F :=
  x ^ (ij.1 - st.1) * (ij.1.choose st.1 : F) * (y ^ (ij.2 - st.2) * (ij.2.choose st.2 : F))

/-- `entry`'s field operations: `x^e` as `e` multiplications, each binomial coefficient's image in the field as one, and
three products. -/
def entryCost (st ij : ℕ × ℕ) : ℕ := (ij.1 - st.1) + (ij.2 - st.2) + 5

theorem evalConstraint_monomial {F : Type} [Field F] (x y : F) (s t i j : ℕ) :
    GuruswamiSudan.evalConstraint x y s t (monomial j (monomial i (1 : F))) = entry x y (s, t) (i, j) := by
  change ((Polynomial.Bivariate.shift (monomial j (monomial i (1 : F))) x y).coeff t).coeff s = _
  simp only [Polynomial.Bivariate.shift, monomial_comp, Polynomial.map_mul, Polynomial.map_pow, map_C,
    coe_compRingHom_apply, C_1, one_mul, coeff_C_mul, entry]
  simp only [Polynomial.map_add, Polynomial.map_X, Polynomial.map_C, coe_compRingHom_apply, C_comp, coeff_X_add_C_pow]
  rw [← C_pow, ← map_natCast C, ← C_mul, coeff_mul_C, coeff_X_add_C_pow]

/-! ## The system and its solution -/

/-- The unknowns: the first `n` monomials of weighted degree at most `D`. -/
def sel (k D n : ℕ) (u : Fin n) : ℕ × ℕ := (monos k D).getD u (0, 0)

theorem sel_mem {k D n : ℕ} (hn : n ≤ GuruswamiSudan.numVars k D) (u : Fin n) : sel k D n u ∈ monos k D := by
  rw [sel, List.getD_eq_getElem _ _ (by rw [length_monos]; omega)]
  exact List.getElem_mem _

theorem sel_injective {k D n : ℕ} (hn : n ≤ GuruswamiSudan.numVars k D) : Function.Injective (sel k D n) := by
  intro u v h
  have hu : u.val < (monos k D).length := by rw [length_monos]; omega
  have hv : v.val < (monos k D).length := by rw [length_monos]; omega
  simp only [sel, List.getD_eq_getElem _ _ hu, List.getD_eq_getElem _ _ hv] at h
  exact Fin.ext ((nodup_monos k D).getElem_inj_iff.1 h)

/-- The table with `c u` at `sel u` and zero elsewhere, by lookup (no field operation). -/
def tabOf {F : Type} [Field F] {n : ℕ} (sel : Fin n → ℕ × ℕ) (c : Fin n → F) (i j : ℕ) : F :=
  match (List.finRange n).find? (fun u => decide (sel u = (i, j))) with
  | some u => c u
  | none => 0

theorem tabOf_eq {F : Type} [Field F] {n : ℕ} {sel : Fin n → ℕ × ℕ} (hs : Function.Injective sel) (c : Fin n → F)
    (i j : ℕ) : tabOf sel c i j = ∑ u, if sel u = (i, j) then c u else 0 := by
  unfold tabOf
  split
  · rename_i u hu
    have hsu : sel u = (i, j) := by simpa using List.find?_some hu
    rw [Finset.sum_eq_single u (fun v _ hv => ite_eq_right fun h => hv (hs (h.trans hsu.symm))) (by simp), ite_eq_left hsu]
  · rename_i hu
    refine (Finset.sum_eq_zero fun v _ => ite_eq_right fun h => ?_).symm
    have := List.find?_eq_none.1 hu v (List.mem_finRange v)
    simp [h] at this

/-- The interpolation system: at each point `(x, y)` and each `(s, t)` with `s + t < m`, the row `u ↦ entry x y (s, t)
(sel u)`. -/
def eqns {F : Type} [Field F] {n : ℕ} (sel : Fin n → ℕ × ℕ) (m : ℕ) (pts : List (F × F)) : List (Fin n → F) :=
  pts.flatMap fun p => (ders m).map fun st u => entry p.1 p.2 st (sel u)

theorem length_eqns {F : Type} [Field F] {n : ℕ} (sel : Fin n → ℕ × ℕ) (m : ℕ) (pts : List (F × F)) :
    (eqns sel m pts).length = pts.length * (m * (m + 1) / 2) := by
  simp [eqns, List.length_flatMap, length_ders]

/-- The system's field operations: `entryCost` for every entry of every row. -/
def eqnsCost {n : ℕ} (sel : Fin n → ℕ × ℕ) (m N : ℕ) : ℕ :=
  N * ((ders m).map fun st => ∑ u, entryCost st (sel u)).sum

/-- The multiplicity `k + 1`. -/
def mult (k : ℕ) : ℕ := k + 1

/-- The `X`-degree bound `D = (k+1)·t − 1`, the weighted-degree bound. -/
def degX (k t : ℕ) : ℕ := (k + 1) * t - 1

/-- The `Y`-degree bound `L`: `D/(k−1)` for `k ≥ 2`, else `D`. -/
def degY (k t : ℕ) : ℕ := if 2 ≤ k then degX k t / (k - 1) else degX k t

/-- The number of constraints `R = N·m(m+1)/2`. -/
def neqs (k N : ℕ) : ℕ := N * ((k + 1) * (k + 2) / 2)

/-- **The regime**: `(k+1)·N < t²` and `t ≤ N`. -/
def Reg (k N t : ℕ) : Prop := (k + 1) * N < t * t ∧ t ≤ N

instance (k N t : ℕ) : Decidable (Reg k N t) := by unfold Reg; infer_instance

theorem Reg.pos {k N t : ℕ} (h : Reg k N t) : 0 < t := by
  rcases Nat.eq_zero_or_pos t with rfl | ht
  · exact absurd h.1 (by simp)
  · exact ht

theorem Reg.k_lt {k N t : ℕ} (h : Reg k N t) : k + 1 < N := by
  have h1 := h.1
  have h2 : t * t ≤ N * N := Nat.mul_le_mul h.2 h.2
  by_contra hc
  have : N * N ≤ (k + 1) * N := Nat.mul_le_mul_right _ (by omega)
  omega

theorem degX_succ {k N t : ℕ} (h : Reg k N t) : degX k t + 1 = (k + 1) * t := by
  have := h.pos
  unfold degX
  have : 1 ≤ (k + 1) * t := Nat.one_le_iff_ne_zero.2 (by positivity)
  omega

/-- **The system has more unknowns than equations** in the regime, by ArkLib's count of the unknowns. -/
theorem neqs_lt_numVars {k N t : ℕ} (h : Reg k N t) : neqs k N < GuruswamiSudan.numVars k (degX k t) := by
  have hD := degX_succ h
  have hN : 0 < N := lt_of_lt_of_le h.pos h.2
  have hc : GuruswamiSudan.numConstraints N (k + 1) = neqs k N := by
    rw [GuruswamiSudan.numConstraints, GuruswamiSudan.card_constraintIndices]; rfl
  rcases Nat.lt_or_ge 1 k with hk | hk
  · rw [← hc]
    refine GuruswamiSudan.numVars_gt_numConstraints_of_mul_lt hk ?_
    obtain ⟨j, rfl⟩ : ∃ j, k = j + 2 := ⟨k - 2, by omega⟩
    set D := degX (j + 2) t
    have ht := h.1
    have key : (j + 1) * N * (j + 3) * (j + 4) < D * (D + 2) := by
      have e1 : D * (D + 2) + 1 = (j + 3) * t * ((j + 3) * t) := by
        rw [← hD]; ring
      have e2 : (j + 3) * (j + 3) * ((j + 3) * N + 1) ≤ (j + 3) * (j + 3) * (t * t) :=
        Nat.mul_le_mul_left _ (by linarith)
      nlinarith
    have : ((((j + 1) * N * (j + 3) * (j + 4) : ℕ)) : ℝ) < ((D * (D + 2) : ℕ) : ℝ) := by exact_mod_cast key
    push_cast at this ⊢
    linarith
  · rw [GuruswamiSudan.numVars_eq_sq hk, hD]
    interval_cases k
    · simp only [neqs]; nlinarith [h.1]
    · simp only [neqs]; nlinarith [h.1]

/-- The number of unknowns, `R + 1`. -/
def nvars (k N : ℕ) : ℕ := neqs k N + 1

theorem nvars_le {k N t : ℕ} (h : Reg k N t) : nvars k N ≤ GuruswamiSudan.numVars k (degX k t) :=
  neqs_lt_numVars h

/-- The interpolation system at `N = pts.length` points. -/
def interpEqns {F : Type} [Field F] (k t : ℕ) (pts : List (F × F)) : List (Fin (nvars k pts.length) → F) :=
  eqns (sel k (degX k t) (nvars k pts.length)) (mult k) pts

/-- **The solution**: a kernel vector of the interpolation system, by Gaussian elimination. -/
def interpCoeffs {F : Type} [Field F] [DecidableEq F] (k t : ℕ) (pts : List (F × F)) :
    Fin (nvars k pts.length) → F :=
  kernel _ (interpEqns k t pts)

/-- **The interpolation polynomial's table**: the solution at its monomials. -/
def interpTab {F : Type} [Field F] [DecidableEq F] (k t : ℕ) (pts : List (F × F)) : ℕ → ℕ → F :=
  tabOf (sel k (degX k t) (nvars k pts.length)) (interpCoeffs k t pts)

/-- **The interpolation step's field operations**: the system's entries and the elimination. -/
def interpCost {F : Type} [Field F] [DecidableEq F] (k t : ℕ) (pts : List (F × F)) : ℕ :=
  eqnsCost (sel k (degX k t) (nvars k pts.length)) (mult k) pts.length +
    kernelCost (nvars k pts.length) (interpEqns k t pts)

/-- The interpolation polynomial `Q(X, Y)`. -/
noncomputable def interpPoly {F : Type} [Field F] [DecidableEq F] (k t : ℕ) (pts : List (F × F)) : F[X][X] :=
  toBi (interpTab k t pts) (degX k t) (degY k t)

theorem length_interpEqns {F : Type} [Field F] (k t : ℕ) (pts : List (F × F)) :
    (interpEqns k t pts).length = neqs k pts.length := by
  rw [interpEqns, length_eqns, neqs, mult]

theorem sel_bound {k t N : ℕ} (h : Reg k N t) (u : Fin (nvars k N)) :
    (sel k (degX k t) (nvars k N) u).1 + (k - 1) * (sel k (degX k t) (nvars k N) u).2 ≤ degX k t ∧
      (sel k (degX k t) (nvars k N) u).1 ≤ degX k t ∧ (sel k (degX k t) (nvars k N) u).2 ≤ degY k t := by
  obtain ⟨h1, h2, h3⟩ := mem_monos.1 (sel_mem (nvars_le h) u)
  refine ⟨h3, h1, ?_⟩
  unfold degY
  split_ifs with hk
  · refine (Nat.le_div_iff_mul_le (by omega)).2 ?_
    rw [Nat.mul_comm]
    omega
  · exact h2

theorem interpTab_ne_zero {F : Type} [Field F] [DecidableEq F] {k t : ℕ} {pts : List (F × F)}
    (h : Reg k pts.length t) {i j : ℕ} (hij : interpTab k t pts i j ≠ 0) :
    ∃ u, sel k (degX k t) (nvars k pts.length) u = (i, j) := by
  rw [interpTab, tabOf_eq (sel_injective (nvars_le h))] at hij
  obtain ⟨u, _, hu⟩ := Finset.exists_ne_zero_of_sum_ne_zero hij
  exact ⟨u, by by_contra hc; exact hu (ite_eq_right hc)⟩

/-- The table lies in the box `i ≤ D`, `j ≤ L`. -/
theorem interpTab_boxed {F : Type} [Field F] [DecidableEq F] {k t : ℕ} {pts : List (F × F)}
    (h : Reg k pts.length t) : Boxed (interpTab k t pts) (degX k t) (degY k t) := by
  intro i j hij
  obtain ⟨u, hu⟩ := interpTab_ne_zero h hij
  have := sel_bound h u
  rw [hu] at this
  exact ⟨this.2.1, this.2.2⟩

/-- The table has `(1, k−1)`-weighted degree at most `D`. -/
theorem interpTab_wdeg {F : Type} [Field F] [DecidableEq F] {k t : ℕ} {pts : List (F × F)}
    (h : Reg k pts.length t) {i j : ℕ} (hij : interpTab k t pts i j ≠ 0) : i + (k - 1) * j ≤ degX k t := by
  obtain ⟨u, hu⟩ := interpTab_ne_zero h hij
  have := (sel_bound h u).1
  rwa [hu] at this

theorem interpPoly_eq_sum {F : Type} [Field F] [DecidableEq F] {k t : ℕ} {pts : List (F × F)}
    (h : Reg k pts.length t) :
    interpPoly k t pts = ∑ u, interpCoeffs k t pts u •
      monomial (sel k (degX k t) (nvars k pts.length) u).2
        (monomial (sel k (degX k t) (nvars k pts.length) u).1 (1 : F)) := by
  ext j i
  rw [interpPoly, coeff_toBi_of_boxed (interpTab_boxed h), interpTab,
    tabOf_eq (sel_injective (nvars_le h))]
  simp only [finsetSum_coeff, coeff_smul, coeff_monomial, smul_eq_mul]
  refine Finset.sum_congr rfl fun u _ => ?_
  by_cases hu : sel k (degX k t) (nvars k pts.length) u = (i, j)
  · simp [hu]
  · rw [ite_eq_right hu]
    by_cases hj : (sel k (degX k t) (nvars k pts.length) u).2 = j
    · rw [ite_eq_left hj, coeff_monomial, ite_eq_right, mul_zero]
      exact fun hi => hu (Prod.ext hi hj)
    · simp [hj]

/-- **The solution is nonzero**, so `Q ≠ 0`. -/
theorem interpPoly_ne_zero {F : Type} [Field F] [DecidableEq F] {k t : ℕ} {pts : List (F × F)}
    (h : Reg k pts.length t) : interpPoly k t pts ≠ 0 := by
  have hc := kernel_ne_zero (nvars k pts.length) (interpEqns k t pts)
    (by rw [length_interpEqns, nvars]; omega)
  obtain ⟨u, hu⟩ := Function.ne_iff.1 hc
  intro h0
  have := congrArg (fun Q : F[X][X] => (Q.coeff (sel k (degX k t) (nvars k pts.length) u).2).coeff
    (sel k (degX k t) (nvars k pts.length) u).1) h0
  simp only [interpPoly, coeff_toBi_of_boxed (interpTab_boxed h), coeff_zero] at this
  rw [interpTab, tabOf_eq (sel_injective (nvars_le h)),
    Finset.sum_eq_single u (fun v _ hv => ite_eq_right fun e => hv (sel_injective (nvars_le h) e)) (by simp),
    ite_eq_left rfl] at this
  exact hu this

/-- `Q` vanishes to order `k + 1` at each point. -/
theorem interpPoly_hasOrderAt {F : Type} [Field F] [DecidableEq F] {k t : ℕ} {pts : List (F × F)}
    (h : Reg k pts.length t) {p : F × F} (hp : p ∈ pts) : GuruswamiSudan.HasOrderAt (interpPoly k t pts) p.1 p.2 (k + 1) := by
  intro s r hsr
  have hker := kernel_mem (nvars k pts.length) (interpEqns k t pts) (by rw [length_interpEqns, nvars]; omega)
  have hmem : (fun u => entry p.1 p.2 (s, r) (sel k (degX k t) (nvars k pts.length) u)) ∈ interpEqns k t pts := by
    simp only [interpEqns, eqns, List.mem_flatMap, List.mem_map]
    exact ⟨p, hp, (s, r), mem_ders.2 hsr, rfl⟩
  have := hker _ hmem
  change GuruswamiSudan.evalConstraint p.1 p.2 s r (interpPoly k t pts) = 0
  rw [interpPoly_eq_sum h, map_sum]
  simp only [map_smul, evalConstraint_monomial, smul_eq_mul]
  rw [← this]
  exact Finset.sum_congr rfl fun u _ => mul_comm _ _

/-- `Q(X, g(X))` has degree at most `D` for `g` of degree at most `w`, when `P` has `(1, w)`-weighted degree at most
`D`. -/
theorem natDegree_eval_toBi_le {F : Type} [Field F] {P : ℕ → ℕ → F} {D L w : ℕ}
    (hw : ∀ i j, P i j ≠ 0 → i + w * j ≤ D) {g : F[X]} (hg : g.natDegree ≤ w) :
    ((toBi P D L).eval g).natDegree ≤ D := by
  rw [eval_toBi]
  refine natDegree_sum_le_of_forall_le _ _ fun j _ => ?_
  rw [rowX, Finset.sum_mul]
  refine natDegree_sum_le_of_forall_le _ _ fun i _ => ?_
  by_cases h0 : P i j = 0
  · simp [h0]
  · refine (natDegree_mul_le).trans ?_
    have h1 := natDegree_monomial_le (P i j) (m := i)
    have h2 : (g ^ j).natDegree ≤ j * w := natDegree_pow_le.trans (Nat.mul_le_mul_left j hg)
    have := hw i j h0
    nlinarith

/-- **Guruswami–Sudan's key property**: `Q(X, P(X)) = 0` for every `P` of degree below `k` that agrees with at least
`t` of the points, at points with distinct `x`, in the regime. -/
theorem interpPoly_eval_eq_zero {F : Type} [Field F] [DecidableEq F] {k t : ℕ} {pts : List (F × F)}
    (hnd : (pts.map Prod.fst).Nodup) (h : Reg k pts.length t) {P : F[X]} (hP : P.natDegree < k)
    (ht : t ≤ (pts.filter fun p => decide (P.eval p.1 = p.2)).length) : (interpPoly k t pts).eval P = 0 := by
  set Q := interpPoly k t pts
  have hdeg : (Q.eval P).natDegree ≤ degX k t :=
    natDegree_eval_toBi_le (fun i j hij => interpTab_wdeg h hij) (by omega)
  let ωs : Fin pts.length ↪ F := ⟨fun i => (pts.get i).1, fun i j hij => Fin.ext <|
    (List.Nodup.getElem_inj_iff hnd (i := i.val) (j := j.val) (hi := by simp) (hj := by simp)).1 (by simpa using hij)⟩
  let A := Finset.univ.filter fun i : Fin pts.length => P.eval (pts.get i).1 = (pts.get i).2
  have hA : t ≤ A.card := by rw [card_filter_get (fun p : F × F => P.eval p.1 = p.2) pts]; exact ht
  by_contra hne
  have hroots : ∀ i ∈ A, k + 1 ≤ (Q.eval P).rootMultiplicity (ωs i) := by
    intro i hi
    have hi' : P.eval (pts.get i).1 = (pts.get i).2 := (Finset.mem_filter.1 hi).2
    have hord := interpPoly_hasOrderAt h (List.get_mem pts i)
    rw [← hi'] at hord
    exact (GuruswamiSudan.orderAt_eval_ge Q P _ _ hord).resolve_left hne
  have hlt : (Q.eval P).natDegree < (k + 1) * A.card := by
    have := degX_succ h
    have : (k + 1) * t ≤ (k + 1) * A.card := Nat.mul_le_mul_left _ hA
    omega
  exact hne (GuruswamiSudan.roots_le_degree_of_deg_lt_roots _ (k + 1) A hroots hlt)

/-! ## The count -/

theorem entryCost_le {k t N : ℕ} (h : Reg k N t) (st : ℕ × ℕ) (u : Fin (nvars k N)) :
    entryCost st (sel k (degX k t) (nvars k N) u) ≤ 2 * degX k t + 5 := by
  have := sel_bound h u
  unfold entryCost
  have : (sel k (degX k t) (nvars k N) u).2 ≤ degX k t := by
    unfold degY at this; split_ifs at this
    · exact this.2.2.trans (Nat.div_le_self _ _)
    · exact this.2.2
  omega

theorem eqnsCost_le {k t N : ℕ} (h : Reg k N t) :
    eqnsCost (sel k (degX k t) (nvars k N)) (mult k) N ≤ neqs k N * nvars k N * (2 * degX k t + 5) := by
  unfold eqnsCost neqs
  have hs : ((ders (mult k)).map fun st => ∑ u, entryCost st (sel k (degX k t) (nvars k N) u)).sum ≤
      (ders (mult k)).length * (nvars k N * (2 * degX k t + 5)) := by
    refine (List.sum_le_length_nsmul _ _ fun x hx => ?_).trans_eq (by rw [List.length_map, smul_eq_mul])
    obtain ⟨st, _, rfl⟩ := List.mem_map.1 hx
    calc ∑ u, entryCost st (sel k (degX k t) (nvars k N) u) ≤ ∑ _u : Fin (nvars k N), (2 * degX k t + 5) :=
          Finset.sum_le_sum fun u _ => entryCost_le h st u
      _ = nvars k N * (2 * degX k t + 5) := by simp
  rw [length_ders, mult] at hs
  have : (k + 1) * (k + 1 + 1) / 2 = (k + 1) * (k + 2) / 2 := rfl
  rw [this] at hs
  calc N * _ ≤ N * ((k + 1) * (k + 2) / 2 * (nvars k N * (2 * degX k t + 5))) := Nat.mul_le_mul_left _ hs
    _ = _ := by ring

theorem neqs_le {k N : ℕ} (hk : k + 1 < N) : neqs k N ≤ N ^ 3 := by
  unfold neqs
  have : (k + 1) * (k + 2) / 2 ≤ N * N := (Nat.div_le_self _ _).trans (Nat.mul_le_mul (by omega) (by omega))
  calc N * ((k + 1) * (k + 2) / 2) ≤ N * (N * N) := Nat.mul_le_mul_left _ this
    _ = N ^ 3 := by ring

theorem degX_le {k N t : ℕ} (h : Reg k N t) : degX k t + 1 ≤ N * N := by
  rw [degX_succ h]
  exact Nat.mul_le_mul (by have := h.k_lt; omega) h.2

/-- **The interpolation step's count**: at most `20·N^9` field operations in the regime. -/
theorem interpCost_le {F : Type} [Field F] [DecidableEq F] {k t : ℕ} {pts : List (F × F)}
    (h : Reg k pts.length t) : interpCost k t pts ≤ 20 * pts.length ^ 9 := by
  set N := pts.length
  have hR := neqs_le h.k_lt
  have hD := degX_le h
  have hN : 1 ≤ N := by have := h.k_lt; omega
  have h1 := eqnsCost_le (N := pts.length) h
  have h2 := kernelCost_le (nvars k N) (interpEqns k t pts)
  rw [length_interpEqns] at h2
  unfold interpCost
  have hv : nvars k N ≤ 2 * N ^ 3 := by unfold nvars; nlinarith [Nat.one_le_pow 3 N hN]
  have e1 : neqs k N * nvars k N * (2 * degX k t + 5) ≤ N ^ 3 * (2 * N ^ 3) * (7 * N ^ 2) := by
    have : 2 * degX k t + 5 ≤ 7 * N ^ 2 := by nlinarith
    exact Nat.mul_le_mul (Nat.mul_le_mul hR hv) this
  have e2 : neqs k N * nvars k N * (2 * neqs k N + 1) ≤ N ^ 3 * (2 * N ^ 3) * (3 * N ^ 3) := by
    have : 2 * neqs k N + 1 ≤ 3 * N ^ 3 := by nlinarith [Nat.one_le_pow 3 N hN]
    exact Nat.mul_le_mul (Nat.mul_le_mul hR hv) this
  have hN9 : N ^ 8 ≤ N ^ 9 := Nat.pow_le_pow_right hN (by norm_num)
  calc _ ≤ N ^ 3 * (2 * N ^ 3) * (7 * N ^ 2) + N ^ 3 * (2 * N ^ 3) * (3 * N ^ 3) := Nat.add_le_add (h1.trans e1) (h2.trans e2)
    _ = 14 * N ^ 8 + 6 * N ^ 9 := by ring
    _ ≤ 20 * N ^ 9 := by nlinarith

end Claims.GS
