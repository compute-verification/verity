import Proofs.Flock.Claims.GS.Basic

/-!
# The Y-roots of `Q(X, Y)` of degree below `k`: Roth–Ruckenstein, counted

`rr rf D L T k` lists the coefficient lists (length `k`) of the polynomials `P` with `deg P < k` and `Q(X, P(X)) = 0`,
for `Q ≠ 0` given as its coefficient table `T` on the box `i ≤ D`, `j ≤ L` (`T` stores row `i` as the list of
`Q`'s coefficients at `X^i Y^j`, `j ≤ L`; `mkTab`, `tget`). A univariate root finder `rf` (`RootFinder`: the distinct
roots of a nonzero polynomial given by its coefficient list, with its operation count) does the field work.

The search is breadth first over the coefficients of `P`, lowest first (Roth and Ruckenstein's recursion). A node is a
prefix `P_0, …, P_{d−1}` and a table `Q_d`. Its lowest nonzero row `v` (`lowRow`) splits `Q_d = X^v·Q_v`
(`tpoly_eq_mul`); the next coefficient is a root `a` of row `v` read as a polynomial in `Y`, which is `Q_v(0, Y)`, and
the child for `a` is `Q_v(X, XY + a)` (`childTab`, `shiftEntry`). If `Q_d(X, g) = 0` with `g = a + X·g'`, then
`Q_v(0, a) = 0` and the child vanishes at `g'`. Its table is exact on the box as long as `Q_d` has
`(1, w)`-weighted degree at most `D` with `w ≥ 1` (`tpoly_childTab`), and the weight drops by one per level, from
`k − 1` at the root, so the first `k − 1` levels are exact (`mem_rr`, completeness).

Each level after the root has at most `L` nodes (`length_level_le`), so the output has at most `L + 1` prefixes
(`length_rr_le`), distinct and each of `k` coefficients (`rr_shape`): the root multiplicity `μ` of `a` in a node's row
polynomial bounds the degree of its child's row polynomial (`phi_child_le`), and the multiplicities of a polynomial's
distinct roots sum to at most its degree (`expand_le`). The operation count (`rrCost`) is at most
`k·(L + 1)·((D + 1)(L + 1) + ρ(L + 1) + (D + 1)(L + 1)²(L + 4))`, with `ρ` the root finder's bound (`rrCost_le`).
-/

namespace Claims.GS

open Polynomial

/-! ## Tables stored as rows -/

/-- Entry `(i, j)` of a table stored as rows. -/
def tget {F : Type} [Zero F] (T : List (List F)) (i j : ℕ) : F := (T.getD i []).getD j 0

/-- The table of `f` on the box `i ≤ D`, `j ≤ L`, stored as `D + 1` rows of `L + 1` entries. -/
def mkTab {F : Type} (D L : ℕ) (f : ℕ → ℕ → F) : List (List F) :=
  (List.range (D + 1)).map fun i => (List.range (L + 1)).map fun j => f i j

theorem getD_mkTab {F : Type} (D L : ℕ) (f : ℕ → ℕ → F) (i : ℕ) :
    (mkTab D L f).getD i [] = if i ≤ D then (List.range (L + 1)).map fun j => f i j else [] := by
  unfold mkTab
  split_ifs with h
  · rw [List.getD_eq_getElem _ _ (by simp; omega)]; simp
  · rw [List.getD_eq_default _ _ (by simp; omega)]

theorem tget_mkTab {F : Type} [Zero F] (D L : ℕ) (f : ℕ → ℕ → F) (i j : ℕ) :
    tget (mkTab D L f) i j = if i ≤ D ∧ j ≤ L then f i j else 0 := by
  unfold tget
  rw [getD_mkTab]
  by_cases hi : i ≤ D
  · rw [ite_eq_left hi]
    by_cases hj : j ≤ L
    · rw [List.getD_eq_getElem _ _ (by simp; omega)]; simp [hi, hj]
    · rw [List.getD_eq_default _ _ (by simp; omega)]; simp [hj]
  · simp [hi]

theorem boxed_mkTab {F : Type} [Field F] (D L : ℕ) (f : ℕ → ℕ → F) : Boxed (tget (mkTab D L f)) D L := by
  intro i j h
  rw [tget_mkTab] at h
  by_contra hc
  exact h (ite_eq_right hc)

theorem length_getD_mkTab_le {F : Type} (D L : ℕ) (f : ℕ → ℕ → F) (i : ℕ) :
    ((mkTab D L f).getD i []).length ≤ L + 1 := by
  rw [getD_mkTab]; split_ifs <;> simp

/-! ## The lowest nonzero row -/

/-- The lowest row of the box with a nonzero entry: `Q = X^v·Q_v` with `Q_v(0, Y) ≠ 0`. A zero test per entry. -/
def lowRow {F : Type} [Zero F] [DecidableEq F] (D L : ℕ) (T : List (List F)) : Option ℕ :=
  (List.range (D + 1)).find? fun i => (List.range (L + 1)).any fun j => decide (tget T i j ≠ 0)

theorem lowRow_some {F : Type} [Zero F] [DecidableEq F] {D L : ℕ} {T : List (List F)} {v : ℕ}
    (h : lowRow D L T = some v) :
    v ≤ D ∧ (∃ j ≤ L, tget T v j ≠ 0) ∧ ∀ i < v, ∀ j ≤ L, tget T i j = 0 := by
  rw [lowRow, List.find?_range_eq_some] at h
  obtain ⟨h1, h2, h3⟩ := h
  simp only [List.any_eq_true, List.mem_range, decide_eq_true_eq] at h1
  refine ⟨by simpa [Nat.lt_succ_iff] using h2, ?_, ?_⟩
  · obtain ⟨j, hj, hne⟩ := h1
    exact ⟨j, by omega, hne⟩
  · intro i hi j hj
    have := h3 i hi
    simp only [Bool.not_eq_eq_eq_not, Bool.not_true, List.any_eq_false, List.mem_range] at this
    simpa using this j (by omega)

theorem lowRow_none {F : Type} [Zero F] [DecidableEq F] {D L : ℕ} {T : List (List F)} (h : lowRow D L T = none) :
    ∀ i ≤ D, ∀ j ≤ L, tget T i j = 0 := by
  rw [lowRow, List.find?_eq_none] at h
  intro i hi j hj
  have := h i (List.mem_range.2 (by omega))
  simp only [List.any_eq_true, List.mem_range, decide_eq_true_eq, not_exists, not_and, not_not] at this
  exact this j (by omega)

theorem lowRow_le {F : Type} [Zero F] [DecidableEq F] {D L : ℕ} {T : List (List F)} {i j : ℕ}
    (hne : tget T i j ≠ 0) (hi : i ≤ D) (hj : j ≤ L) : ∃ v, lowRow D L T = some v ∧ v ≤ i := by
  cases h : lowRow D L T with
  | none => exact absurd (lowRow_none h i hi j hj) hne
  | some v =>
    refine ⟨v, rfl, ?_⟩
    by_contra hlt
    exact hne ((lowRow_some h).2.2 i (by omega) j hj)

/-! ## The child `Q_v(X, XY + a)` -/

/-- Entry `(u, l)` of `Q_v(X, XY + a)`, where `Q_v(X, Y) = Q(X, Y)/X^v`: `Σ_j Q_{u−l+v, j}·C(j, l)·a^(j−l)` for
`l ≤ u`. -/
def shiftEntry {F : Type} [Field F] (L : ℕ) (T : List (List F)) (v : ℕ) (a : F) (u l : ℕ) : F :=
  if l ≤ u then ∑ j ∈ Finset.range (L + 1), tget T (u - l + v) j * (j.choose l : F) * a ^ (j - l) else 0

/-- The child's table on the box. -/
def childTab {F : Type} [Field F] (D L : ℕ) (T : List (List F)) (v : ℕ) (a : F) : List (List F) :=
  mkTab D L (shiftEntry L T v a)

/-- `childTab`'s field operations: at each of the `(D+1)(L+1)` entries, `L + 1` terms, each a power (`j − l ≤ L`
multiplications), a binomial coefficient's image and two products, and their sum. -/
def childCost (D L : ℕ) : ℕ := (D + 1) * (L + 1) * ((L + 1) * (L + 4))

/-- The bivariate polynomial of a table on the box. -/
noncomputable def tpoly {F : Type} [Field F] (D L : ℕ) (T : List (List F)) : F[X][X] := toBi (tget T) D L

/-- `Q_v = Q / X^v`. -/
noncomputable def lowPart {F : Type} [Field F] (D L : ℕ) (T : List (List F)) (v : ℕ) : F[X][X] :=
  toBi (fun i j => tget T (i + v) j) D L

/-- Row `v` read as a polynomial in `Y`. -/
noncomputable def rowPolyY {F : Type} [Field F] (T : List (List F)) (v : ℕ) : F[X] := toPolyL (T.getD v [])

/-- **`Q = X^v·Q_v`** at the lowest nonzero row `v`. -/
theorem tpoly_eq_mul {F : Type} [Field F] [DecidableEq F] {D L : ℕ} {T : List (List F)} (hB : Boxed (tget T) D L)
    {v : ℕ} (hv : lowRow D L T = some v) : tpoly D L T = C (X ^ v) * lowPart D L T v := by
  obtain ⟨hvD, _, hlow⟩ := lowRow_some hv
  ext j i
  rw [tpoly, coeff_toBi_of_boxed hB, coeff_C_mul, coeff_X_pow_mul', lowPart, coeff_toBi]
  by_cases hiv : v ≤ i
  · rw [ite_eq_left hiv, Nat.sub_add_cancel hiv]
    split_ifs with h
    · rfl
    · by_contra hne
      have := hB i j hne
      exact h ⟨by omega, this.2⟩
  · rw [ite_eq_right hiv]
    by_cases hj : j ≤ L
    · exact hlow i (by omega) j hj
    · by_contra hne
      exact hj (hB i j hne).2

theorem lowPart_ne_zero {F : Type} [Field F] [DecidableEq F] {D L : ℕ} {T : List (List F)}
    (hB : Boxed (tget T) D L) {v : ℕ} (hv : lowRow D L T = some v) (hT : tpoly D L T ≠ 0) :
    lowPart D L T v ≠ 0 := by
  intro h
  rw [tpoly_eq_mul hB hv, h, mul_zero] at hT
  exact hT rfl

theorem coeff_shiftY_pow {F : Type} [Field F] (a : F) (j l : ℕ) :
    ((C X * X + C (C a) : F[X][X]) ^ j).coeff l = C (a ^ (j - l) * (j.choose l : F)) * X ^ l := by
  rw [add_pow, finsetSum_coeff]
  have hterm : ∀ m ∈ Finset.range (j + 1), ((C X * X) ^ m * C (C a) ^ (j - m) * (j.choose m : F[X][X])).coeff l =
      if l = m then C (a ^ (j - m) * (j.choose m : F)) * X ^ m else 0 := by
    intro m _
    have : (C X * X) ^ m * C (C a) ^ (j - m) * (j.choose m : F[X][X]) =
        C (C (a ^ (j - m) * (j.choose m : F)) * X ^ m) * X ^ m := by
      rw [← C_eq_natCast, ← C_eq_natCast, mul_pow, ← C_pow, ← C_pow, ← C_pow]
      simp only [map_mul, map_pow, map_natCast]
      ring
    rw [this, coeff_C_mul_X_pow]
  rw [Finset.sum_congr rfl hterm, Finset.sum_ite_eq]
  split_ifs with h
  · rfl
  · rw [Finset.mem_range, not_lt] at h
    rw [Nat.choose_eq_zero_of_lt (by omega)]
    simp

/-- The coefficients of `S(X, XY + a)`. -/
theorem coeff_comp_shiftY {F : Type} [Field F] (S : F[X][X]) (L : ℕ) (hS : S.natDegree < L + 1) (a : F) (u l : ℕ) :
    ((S.comp (C X * X + C (C a))).coeff l).coeff u =
      if l ≤ u then ∑ j ∈ Finset.range (L + 1), (S.coeff j).coeff (u - l) * (j.choose l : F) * a ^ (j - l)
      else 0 := by
  conv_lhs => rw [S.as_sum_range' (L + 1) hS]
  rw [sum_comp, finsetSum_coeff, finsetSum_coeff]
  have hterm : ∀ j ∈ Finset.range (L + 1), (((monomial j (S.coeff j)).comp (C X * X + C (C a))).coeff l).coeff u =
      if l ≤ u then (S.coeff j).coeff (u - l) * (j.choose l : F) * a ^ (j - l) else 0 := by
    intro j _
    rw [monomial_comp, coeff_C_mul, coeff_shiftY_pow, ← mul_assoc, mul_comm (S.coeff j), mul_assoc, coeff_C_mul,
      coeff_mul_X_pow']
    split_ifs <;> ring
  rw [Finset.sum_congr rfl hterm]
  split_ifs <;> simp

theorem natDegree_toBi_lt {F : Type} [Field F] (P : ℕ → ℕ → F) (D L : ℕ) : (toBi P D L).natDegree < L + 1 := by
  refine Nat.lt_succ_of_le (natDegree_le_iff_coeff_eq_zero.2 fun j hj => ?_)
  ext i
  rw [coeff_toBi, ite_eq_right (by omega), coeff_zero]

/-- **The child's table is `Q_v(X, XY + a)`** when `Q` has `(1, w)`-weighted degree at most `D`, `w ≥ 1`. -/
theorem tpoly_childTab {F : Type} [Field F] {D L w : ℕ} {T : List (List F)}
    (hW : ∀ i j, tget T i j ≠ 0 → i + w * j ≤ D) (hw : 1 ≤ w) (v : ℕ) (a : F) :
    tpoly D L (childTab D L T v a) = (lowPart D L T v).comp (C X * X + C (C a)) := by
  ext l u
  rw [coeff_comp_shiftY (lowPart D L T v) L (natDegree_toBi_lt _ _ _), tpoly, coeff_toBi, childTab, tget_mkTab]
  simp only [lowPart, coeff_toBi]
  by_cases hlu : l ≤ u
  · rw [ite_eq_left hlu]
    by_cases hbox : u ≤ D ∧ l ≤ L
    · rw [ite_eq_left hbox, ite_eq_left hbox, shiftEntry, ite_eq_left hlu]
      refine Finset.sum_congr rfl fun j hj => ?_
      rw [ite_eq_left ⟨by omega, by simpa [Nat.lt_succ_iff] using hj⟩]
    · rw [ite_eq_right hbox]
      refine (Finset.sum_eq_zero fun j hj => ?_).symm
      split_ifs with h
      · by_cases hc : (j.choose l : F) = 0
        · rw [hc]; ring
        · have hlj : l ≤ j := by
            by_contra hlt
            exact hc (by rw [Nat.choose_eq_zero_of_lt (by omega)]; simp)
          by_cases ht : tget T (u - l + v) j = 0
          · rw [ht]; ring
          · have := hW _ _ ht
            have hjL : j ≤ L := by simpa [Nat.lt_succ_iff] using hj
            exfalso
            apply hbox
            refine ⟨?_, by omega⟩
            have e : u - l + l = u := Nat.sub_add_cancel hlu
            have e2 : 1 * j ≤ w * j := Nat.mul_le_mul_right _ hw
            omega
      · ring
  · rw [ite_eq_right hlu]
    split_ifs with h
    · rw [shiftEntry, ite_eq_right hlu]
    · rfl

/-- The child's weighted degree drops by one. -/
theorem wdeg_childTab {F : Type} [Field F] {D L w : ℕ} {T : List (List F)}
    (hW : ∀ i j, tget T i j ≠ 0 → i + w * j ≤ D) (hw : 1 ≤ w) (v : ℕ) (a : F) :
    ∀ u l, tget (childTab D L T v a) u l ≠ 0 → u + (w - 1) * l ≤ D := by
  intro u l h
  rw [childTab, tget_mkTab] at h
  split_ifs at h with hbox
  · rw [shiftEntry] at h
    split_ifs at h with hlu
    · obtain ⟨j, hj, hne⟩ := Finset.exists_ne_zero_of_sum_ne_zero h
      have h1 : tget T (u - l + v) j ≠ 0 := fun h0 => hne (by rw [h0]; ring)
      have h2 : (j.choose l : F) ≠ 0 := fun h0 => hne (by rw [h0]; ring)
      have hlj : l ≤ j := by
        by_contra hlt
        exact h2 (by rw [Nat.choose_eq_zero_of_lt (by omega)]; simp)
      have := hW _ _ h1
      obtain ⟨w', rfl⟩ : ∃ w', w = w' + 1 := ⟨w - 1, by omega⟩
      simp only [Nat.add_sub_cancel]
      have e : u - l + l = u := Nat.sub_add_cancel hlu
      have e2 : (w' + 1) * l ≤ (w' + 1) * j := Nat.mul_le_mul_left _ hlj
      nlinarith
    · exact absurd rfl h
  · exact absurd rfl h

theorem tpoly_childTab_ne_zero {F : Type} [Field F] [DecidableEq F] {D L w : ℕ} {T : List (List F)}
    (hB : Boxed (tget T) D L) (hW : ∀ i j, tget T i j ≠ 0 → i + w * j ≤ D) (hw : 1 ≤ w) {v : ℕ}
    (hv : lowRow D L T = some v) (hT : tpoly D L T ≠ 0) (a : F) : tpoly D L (childTab D L T v a) ≠ 0 := by
  rw [tpoly_childTab hW hw]
  intro h
  rcases comp_eq_zero_iff.1 h with h0 | ⟨_, hc⟩
  · exact lowPart_ne_zero hB hv hT h0
  · have := congrArg (fun q : F[X][X] => q.coeff 1) hc
    simp at this

/-- The child vanishes at `g'` when `Q` vanishes at `a + X·g'`. -/
theorem eval_childTab {F : Type} [Field F] [DecidableEq F] {D L w : ℕ} {T : List (List F)}
    (hB : Boxed (tget T) D L) (hW : ∀ i j, tget T i j ≠ 0 → i + w * j ≤ D) (hw : 1 ≤ w) {v : ℕ}
    (hv : lowRow D L T = some v) {g : F[X]} (hg : (tpoly D L T).eval g = 0) :
    (tpoly D L (childTab D L T v (g.coeff 0))).eval g.divX = 0 := by
  rw [tpoly_childTab hW hw, eval_comp]
  have : (C X * X + C (C (g.coeff 0)) : F[X][X]).eval g.divX = g := by
    simp only [eval_add, eval_mul, eval_C, eval_X]
    exact X_mul_divX_add g
  rw [this]
  rw [tpoly_eq_mul hB hv, eval_mul, eval_C] at hg
  exact (mul_eq_zero.1 hg).resolve_left (pow_ne_zero _ X_ne_zero)

/-- The next coefficient is a root of row `v`: `Q_v(0, g(0)) = 0`. -/
theorem rowPolyY_isRoot {F : Type} [Field F] [DecidableEq F] {D L : ℕ} {T : List (List F)}
    (hB : Boxed (tget T) D L) {v : ℕ} (hv : lowRow D L T = some v) {g : F[X]} (hg : (tpoly D L T).eval g = 0) :
    (rowPolyY T v).IsRoot (g.coeff 0) := by
  rw [tpoly_eq_mul hB hv, eval_mul, eval_C] at hg
  have h0 := congrArg (fun p : F[X] => p.coeff 0) ((mul_eq_zero.1 hg).resolve_left (pow_ne_zero _ X_ne_zero))
  simp only [coeff_zero] at h0
  have hvD := (lowRow_some hv).1
  rw [eval_eq_sum_range' (p := lowPart D L T v) (natDegree_toBi_lt _ D L), finsetSum_coeff] at h0
  rw [IsRoot, rowPolyY, eval_eq_sum_range' (n := L + 1) ?_]
  · rw [← h0]
    refine Finset.sum_congr rfl fun j hj => ?_
    rw [mul_coeff_zero, coeff_zero_eq_eval_zero (g ^ j), eval_pow, ← coeff_zero_eq_eval_zero, lowPart, coeff_toBi,
      ite_eq_left ⟨by omega, by simpa [Nat.lt_succ_iff] using hj⟩, coeff_toPolyL, zero_add]
    rfl
  · refine Nat.lt_succ_of_le (natDegree_le_iff_coeff_eq_zero.2 fun j hj => ?_)
    rw [coeff_toPolyL]
    by_contra hne
    exact absurd (hB v j hne).2 (by omega)

theorem exists_lowRow {F : Type} [Field F] [DecidableEq F] {D L : ℕ} {T : List (List F)} (hT : tpoly D L T ≠ 0) :
    ∃ v, lowRow D L T = some v := by
  obtain ⟨j, hj⟩ := Polynomial.ext_iff.not.1 hT |> not_forall.1
  obtain ⟨i, hi⟩ := Polynomial.ext_iff.not.1 hj |> not_forall.1
  rw [tpoly, coeff_toBi, coeff_zero, coeff_zero] at hi
  split_ifs at hi with hbox
  · obtain ⟨v, hv, _⟩ := lowRow_le hi hbox.1 hbox.2
    exact ⟨v, hv⟩
  · exact absurd rfl hi

theorem rowPolyY_ne_zero {F : Type} [Field F] [DecidableEq F] {D L : ℕ} {T : List (List F)} {v : ℕ}
    (hv : lowRow D L T = some v) : rowPolyY T v ≠ 0 := by
  obtain ⟨_, ⟨j, _, hj⟩, _⟩ := lowRow_some hv
  intro h
  have := congrArg (fun p : F[X] => p.coeff j) h
  simp only [rowPolyY, coeff_toPolyL, coeff_zero] at this
  exact hj this

/-! ## The search -/

/-- **A univariate root finder**: the distinct roots of a nonzero polynomial given by its coefficient list, lowest
degree first, with its field-operation count, at most `bound` at the list's length. -/
structure RootFinder (F : Type) [Field F] where
  run : List F → List F
  cost : List F → ℕ
  nodup : ∀ cs, toPolyL cs ≠ 0 → (run cs).Nodup
  mem_iff : ∀ cs, toPolyL cs ≠ 0 → ∀ a, a ∈ run cs ↔ (toPolyL cs).IsRoot a
  bound : Polynomial ℕ
  cost_le : ∀ cs, cost cs ≤ bound.eval cs.length

/-- A node of the search: the coefficients found so far, lowest first, and the table. -/
abbrev Node (F : Type) := List F × List (List F)

/-- A node's children: one per root `a` of its lowest nonzero row, with table `Q_v(X, XY + a)`. -/
def expand {F : Type} [Field F] [DecidableEq F] (rf : RootFinder F) (D L : ℕ) (nd : Node F) : List (Node F) :=
  match lowRow D L nd.2 with
  | none => []
  | some v => (rf.run (nd.2.getD v [])).map fun a => (nd.1 ++ [a], childTab D L nd.2 v a)

/-- `expand`'s field operations: a zero test per entry of the box, the root finder, and each child's table. -/
def expandCost {F : Type} [Field F] [DecidableEq F] (rf : RootFinder F) (D L : ℕ) (nd : Node F) : ℕ :=
  (D + 1) * (L + 1) + match lowRow D L nd.2 with
    | none => 0
    | some v => rf.cost (nd.2.getD v []) + (rf.run (nd.2.getD v [])).length * childCost D L

/-- Level `d` of the search, from the root `([], T)`. -/
def level {F : Type} [Field F] [DecidableEq F] (rf : RootFinder F) (D L : ℕ) (T : List (List F)) :
    ℕ → List (Node F)
  | 0 => [([], T)]
  | d + 1 => (level rf D L T d).flatMap (expand rf D L)

/-- **Roth–Ruckenstein**: the prefixes at level `k`. -/
def rr {F : Type} [Field F] [DecidableEq F] (rf : RootFinder F) (D L : ℕ) (T : List (List F)) (k : ℕ) :
    List (List F) :=
  (level rf D L T k).map Prod.fst

/-- `rr`'s field operations: every expansion of levels `0, …, k − 1`. -/
def rrCost {F : Type} [Field F] [DecidableEq F] (rf : RootFinder F) (D L : ℕ) (T : List (List F)) : ℕ → ℕ
  | 0 => 0
  | d + 1 => rrCost rf D L T d + ((level rf D L T d).map (expandCost rf D L)).sum

theorem mem_level_succ {F : Type} [Field F] [DecidableEq F] {rf : RootFinder F} {D L : ℕ} {T : List (List F)}
    {d : ℕ} {nd : Node F} (hnd : nd ∈ level rf D L T d) {v : ℕ} (hv : lowRow D L nd.2 = some v) {a : F}
    (ha : a ∈ rf.run (nd.2.getD v [])) : (nd.1 ++ [a], childTab D L nd.2 v a) ∈ level rf D L T (d + 1) := by
  rw [level, List.mem_flatMap]
  refine ⟨nd, hnd, ?_⟩
  rw [expand, hv]
  exact List.mem_map.2 ⟨a, ha, rfl⟩

theorem shape_level {F : Type} [Field F] [DecidableEq F] {rf : RootFinder F} {D L : ℕ} {f0 : ℕ → ℕ → F} :
    ∀ d, ∀ nd ∈ level rf D L (mkTab D L f0) d, nd.1.length = d ∧ ∃ f, nd.2 = mkTab D L f
  | 0, nd, h => by
    simp only [level, List.mem_singleton] at h
    subst h
    exact ⟨rfl, f0, rfl⟩
  | d + 1, nd, h => by
    rw [level, List.mem_flatMap] at h
    obtain ⟨p, hp, hc⟩ := h
    have ih := shape_level d p hp
    rw [expand] at hc
    split at hc
    · simp at hc
    · obtain ⟨a, _, rfl⟩ := List.mem_map.1 hc
      exact ⟨by simp [ih.1], _, rfl⟩

theorem coeff_iterate_divX {F : Type} [Field F] (P : F[X]) : ∀ (d n : ℕ), (divX^[d] P).coeff n = P.coeff (n + d)
  | 0, n => rfl
  | d + 1, n => by
    rw [Function.iterate_succ_apply', coeff_divX, coeff_iterate_divX P d (n + 1)]
    congr 1
    omega

theorem coeffList_succ {F : Type} [Field F] (d : ℕ) (P : F[X]) :
    coeffList (d + 1) P = coeffList d P ++ [P.coeff d] := by
  simp [coeffList, List.range_succ]

/-- **Roth–Ruckenstein is complete**: from a nonzero `Q` of `(1, k−1)`-weighted degree at most `D` on the box, every
`P` of degree below `k` with `Q(X, P(X)) = 0` is listed, as its first `k` coefficients. -/
theorem mem_rr {F : Type} [Field F] [DecidableEq F] (rf : RootFinder F) {D L k : ℕ} {f0 : ℕ → ℕ → F}
    (hT : tpoly D L (mkTab D L f0) ≠ 0) (hW : ∀ i j, tget (mkTab D L f0) i j ≠ 0 → i + (k - 1) * j ≤ D)
    {P : F[X]} (hP : P.natDegree < k) (hE : (tpoly D L (mkTab D L f0)).eval P = 0) :
    coeffList k P ∈ rr rf D L (mkTab D L f0) k := by
  have claim : ∀ d < k, ∃ T', (coeffList d P, T') ∈ level rf D L (mkTab D L f0) d ∧ tpoly D L T' ≠ 0 ∧
      (∀ i j, tget T' i j ≠ 0 → i + (k - 1 - d) * j ≤ D) ∧ (tpoly D L T').eval (divX^[d] P) = 0 := by
    intro d
    induction d with
    | zero => exact fun _ => ⟨mkTab D L f0, by simp [level, coeffList], hT, by simpa using hW, hE⟩
    | succ d ih =>
      intro hd
      obtain ⟨T', hmem, hne, hW', hE'⟩ := ih (by omega)
      obtain ⟨-, f, hf⟩ := shape_level d _ hmem
      have hB : Boxed (tget T') D L := by change T' = mkTab D L f at hf; rw [hf]; exact boxed_mkTab D L f
      obtain ⟨v, hv⟩ := exists_lowRow hne
      have hroot := rowPolyY_isRoot hB hv hE'
      have ha : (divX^[d] P).coeff 0 ∈ rf.run (T'.getD v []) :=
        (rf.mem_iff _ (rowPolyY_ne_zero hv) _).2 hroot
      have hw : 1 ≤ k - 1 - d := by omega
      refine ⟨childTab D L T' v ((divX^[d] P).coeff 0), ?_, tpoly_childTab_ne_zero hB hW' hw hv hne _, ?_, ?_⟩
      · have e : coeffList (d + 1) P = coeffList d P ++ [(divX^[d] P).coeff 0] := by
          rw [coeff_iterate_divX, zero_add, coeffList_succ]
        rw [e]
        exact mem_level_succ hmem hv ha
      · intro i j hij
        have := wdeg_childTab hW' hw v _ i j hij
        rwa [show k - 1 - d - 1 = k - 1 - (d + 1) by omega] at this
      · rw [Function.iterate_succ_apply']
        exact eval_childTab hB hW' hw hv hE'
  obtain ⟨k', rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
  obtain ⟨T', hmem, hne, -, hE'⟩ := claim k' (by omega)
  obtain ⟨-, f, hf⟩ := shape_level k' _ hmem
  have hB : Boxed (tget T') D L := by change T' = mkTab D L f at hf; rw [hf]; exact boxed_mkTab D L f
  obtain ⟨v, hv⟩ := exists_lowRow hne
  have ha : (divX^[k'] P).coeff 0 ∈ rf.run (T'.getD v []) :=
    (rf.mem_iff _ (rowPolyY_ne_zero hv) _).2 (rowPolyY_isRoot hB hv hE')
  have e : coeffList (k' + 1) P = coeffList k' P ++ [(divX^[k'] P).coeff 0] := by
    rw [coeff_iterate_divX, zero_add, coeffList_succ]
  rw [rr, e]
  exact List.mem_map.2 ⟨_, mem_level_succ hmem hv ha, rfl⟩

/-! ## The size of each level -/

theorem sum_map_flatMap {α β : Type} (l : List α) (f : α → List β) (g : β → ℕ) :
    ((l.flatMap f).map g).sum = (l.map fun a => ((f a).map g).sum).sum := by
  induction l with
  | nil => rfl
  | cons a l ih => simp [List.flatMap_cons, ih]


/-- A node's potential: the degree of its lowest nonzero row's polynomial (`0` at a zero table). -/
noncomputable def phi {F : Type} [Field F] [DecidableEq F] (D L : ℕ) (nd : Node F) : ℕ :=
  match lowRow D L nd.2 with
  | none => 0
  | some v => (rowPolyY nd.2 v).natDegree

/-- The coefficients of `p(X + a)`. -/
theorem coeff_comp_X_add_C {F : Type} [Field F] (p : F[X]) (L : ℕ) (hp : p.natDegree < L + 1) (a : F) (l : ℕ) :
    (p.comp (X + C a)).coeff l = ∑ j ∈ Finset.range (L + 1), p.coeff j * (j.choose l : F) * a ^ (j - l) := by
  conv_lhs => rw [p.as_sum_range' (L + 1) hp]
  rw [sum_comp, finsetSum_coeff]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [monomial_comp, coeff_C_mul, coeff_X_add_C_pow]
  ring

theorem natDegree_rowPolyY_lt {F : Type} [Field F] (D L : ℕ) (f : ℕ → ℕ → F) (v : ℕ) :
    (rowPolyY (mkTab D L f) v).natDegree < L + 1 := by
  refine Nat.lt_succ_of_le (natDegree_le_iff_coeff_eq_zero.2 fun j hj => ?_)
  rw [rowPolyY, coeff_toPolyL]
  exact List.getD_eq_default _ _ (by have := length_getD_mkTab_le D L f v; omega)

/-- **The child's row polynomial has degree at most the root's multiplicity.** -/
theorem phi_child_le {F : Type} [Field F] [DecidableEq F] {D L : ℕ} (hLD : L ≤ D) (pre : List F)
    (f : ℕ → ℕ → F) {v : ℕ} (hv : lowRow D L (mkTab D L f) = some v) (a : F) :
    phi D L (pre ++ [a], childTab D L (mkTab D L f) v a) ≤ (rowPolyY (mkTab D L f) v).rootMultiplicity a := by
  have hr : rowPolyY (mkTab D L f) v ≠ 0 := rowPolyY_ne_zero hv
  have hrL := natDegree_rowPolyY_lt D L f v
  have hμ : (rowPolyY (mkTab D L f) v).rootMultiplicity a ≤ L := by
    rw [← count_roots]
    exact (Multiset.count_le_card _ _).trans ((card_roots' _).trans (Nat.lt_succ_iff.1 hrL))
  have hcomp : (rowPolyY (mkTab D L f) v).comp (X + C a) ≠ 0 := by
    intro h
    rcases comp_eq_zero_iff.1 h with h0 | ⟨_, hc⟩
    · exact hr h0
    · have := congrArg (fun q : F[X] => q.coeff 1) hc
      simp at this
  have hentry : tget (childTab D L (mkTab D L f) v a) ((rowPolyY (mkTab D L f) v).rootMultiplicity a)
      ((rowPolyY (mkTab D L f) v).rootMultiplicity a) ≠ 0 := by
    rw [childTab, tget_mkTab, ite_eq_left ⟨by omega, hμ⟩, shiftEntry, ite_eq_left le_rfl, Nat.sub_self, zero_add]
    have : ((rowPolyY (mkTab D L f) v).comp (X + C a)).coeff ((rowPolyY (mkTab D L f) v).rootMultiplicity a) ≠ 0 := by
      have h := trailingCoeff_nonzero_iff_nonzero.2 hcomp
      rwa [trailingCoeff, ← rootMultiplicity_eq_natTrailingDegree] at h
    rw [coeff_comp_X_add_C _ L hrL] at this
    refine fun h0 => this (h0 ▸ Finset.sum_congr rfl fun j _ => ?_)
    rw [rowPolyY, coeff_toPolyL]
    rfl
  obtain ⟨v', hv', hv'μ⟩ := lowRow_le (D := D) (L := L) hentry (by omega) hμ
  have hphi : phi D L (pre ++ [a], childTab D L (mkTab D L f) v a) =
      (rowPolyY (childTab D L (mkTab D L f) v a) v').natDegree := by
    unfold phi
    simp only
    rw [hv']
  rw [hphi]
  refine (natDegree_le_iff_coeff_eq_zero.2 fun l hl => ?_).trans hv'μ
  rw [rowPolyY, coeff_toPolyL]
  change tget (childTab D L (mkTab D L f) v a) v' l = 0
  rw [childTab, tget_mkTab]
  split_ifs
  · rw [shiftEntry, ite_eq_right (by omega)]
  · rfl

/-- **A node's children**: at most its potential, and so is the sum of theirs. -/
theorem expand_le {F : Type} [Field F] [DecidableEq F] (rf : RootFinder F) {D L : ℕ} (hLD : L ≤ D) (pre : List F)
    (f : ℕ → ℕ → F) :
    (expand rf D L (pre, mkTab D L f)).length ≤ phi D L (pre, mkTab D L f) ∧
      ((expand rf D L (pre, mkTab D L f)).map (phi D L)).sum ≤ phi D L (pre, mkTab D L f) := by
  unfold expand phi
  split
  · simp
  · rename_i v hv
    simp only
    set r := rowPolyY (mkTab D L f) v
    have hr : r ≠ 0 := rowPolyY_ne_zero hv
    set l := rf.run ((mkTab D L f).getD v [])
    have hnd : l.Nodup := rf.nodup _ hr
    have hmem : ∀ a, a ∈ l ↔ r.IsRoot a := rf.mem_iff _ hr
    classical
    have hsum : (l.map fun a => r.rootMultiplicity a).sum ≤ r.natDegree := by
      rw [← List.sum_toFinset _ hnd]
      have hfs : l.toFinset = r.roots.toFinset := by
        ext a
        rw [List.mem_toFinset, hmem, Multiset.mem_toFinset, mem_roots hr]
      rw [hfs]
      calc ∑ a ∈ r.roots.toFinset, r.rootMultiplicity a = ∑ a ∈ r.roots.toFinset, r.roots.count a :=
            Finset.sum_congr rfl fun a _ => (count_roots r).symm
        _ = r.roots.card := Multiset.toFinset_sum_count_eq _
        _ ≤ r.natDegree := card_roots' r
    constructor
    · rw [List.length_map]
      refine le_trans ?_ hsum
      calc l.length = (l.map fun _ => 1).sum := by simp
        _ ≤ (l.map fun a => r.rootMultiplicity a).sum :=
          List.sum_le_sum fun a ha => (rootMultiplicity_pos hr).2 ((hmem a).1 ha)
    · rw [List.map_map]
      refine le_trans (List.sum_le_sum fun a ha => ?_) hsum
      exact phi_child_le hLD pre f hv a

/-- **Each level has at most `L` nodes after the root**, by the potential. -/
theorem length_level_le {F : Type} [Field F] [DecidableEq F] (rf : RootFinder F) {D L : ℕ} (hLD : L ≤ D)
    (f0 : ℕ → ℕ → F) : ∀ d, (level rf D L (mkTab D L f0) (d + 1)).length ≤ L ∧
      ((level rf D L (mkTab D L f0) (d + 1)).map (phi D L)).sum ≤ L := by
  have hstep : ∀ d, (level rf D L (mkTab D L f0) (d + 1)).length ≤ ((level rf D L (mkTab D L f0) d).map (phi D L)).sum ∧
      ((level rf D L (mkTab D L f0) (d + 1)).map (phi D L)).sum ≤
        ((level rf D L (mkTab D L f0) d).map (phi D L)).sum := by
    intro d
    rw [level, List.length_flatMap, sum_map_flatMap]
    constructor
    · refine List.sum_le_sum fun nd hnd => ?_
      obtain ⟨-, f, hf⟩ := shape_level d nd hnd
      obtain ⟨pre, T⟩ := nd
      simp only at hf
      subst hf
      exact (expand_le rf hLD pre f).1
    · refine List.sum_le_sum fun nd hnd => ?_
      obtain ⟨-, f, hf⟩ := shape_level d nd hnd
      obtain ⟨pre, T⟩ := nd
      simp only at hf
      subst hf
      exact (expand_le rf hLD pre f).2
  have h0 : ((level rf D L (mkTab D L f0) 0).map (phi D L)).sum ≤ L := by
    simp only [level, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, phi]
    split
    · exact Nat.zero_le _
    · exact Nat.lt_succ_iff.1 (natDegree_rowPolyY_lt D L f0 _)
  intro d
  induction d with
  | zero => exact ⟨(hstep 0).1.trans h0, (hstep 0).2.trans h0⟩
  | succ d ih => exact ⟨(hstep (d + 1)).1.trans ih.2, (hstep (d + 1)).2.trans ih.2⟩

theorem length_level_le' {F : Type} [Field F] [DecidableEq F] (rf : RootFinder F) {D L : ℕ} (hLD : L ≤ D)
    (f0 : ℕ → ℕ → F) (d : ℕ) : (level rf D L (mkTab D L f0) d).length ≤ L + 1 := by
  cases d with
  | zero => simp [level]
  | succ d => exact (length_level_le rf hLD f0 d).1.trans (Nat.le_succ L)

/-- The output has at most `L + 1` prefixes. -/
theorem length_rr_le {F : Type} [Field F] [DecidableEq F] (rf : RootFinder F) {D L : ℕ} (hLD : L ≤ D)
    (f0 : ℕ → ℕ → F) (k : ℕ) : (rr rf D L (mkTab D L f0) k).length ≤ L + 1 := by
  rw [rr, List.length_map]
  exact length_level_le' rf hLD f0 k

/-! ## The output's shape -/

theorem nodup_level {F : Type} [Field F] [DecidableEq F] (rf : RootFinder F) {D L : ℕ} (f0 : ℕ → ℕ → F) :
    ∀ d, ((level rf D L (mkTab D L f0) d).map Prod.fst).Nodup
  | 0 => by simp [level]
  | d + 1 => by
    have ih := nodup_level rf (D := D) (L := L) f0 d
    rw [level, List.map_flatMap, List.nodup_flatMap]
    constructor
    · intro nd _
      rw [expand]
      split
      · simp
      · rename_i v hv
        rw [List.map_map]
        refine (rf.nodup _ (rowPolyY_ne_zero hv)).map fun a b h => ?_
        simpa using h
    · have hp := List.pairwise_map.1 ih
      refine hp.imp_of_mem fun {nd nd'} hnd hnd' hne => ?_
      simp only [Function.onFun]
      rw [List.disjoint_left]
      intro x hx hx'
      have hl := (shape_level d nd hnd).1
      have hl' := (shape_level d nd' hnd').1
      rw [expand] at hx hx'
      split at hx
      · simp at hx
      split at hx'
      · simp at hx'
      rw [List.map_map] at hx hx'
      obtain ⟨a, _, rfl⟩ := List.mem_map.1 hx
      obtain ⟨a', _, h'⟩ := List.mem_map.1 hx'
      simp only [Function.comp] at h'
      exact hne (List.append_inj h' (by rw [hl, hl'])).1.symm

/-- **The output lists distinct prefixes, each of `k` coefficients.** -/
theorem rr_shape {F : Type} [Field F] [DecidableEq F] (rf : RootFinder F) {D L : ℕ} (f0 : ℕ → ℕ → F) (k : ℕ) :
    (rr rf D L (mkTab D L f0) k).Nodup ∧ ∀ cs ∈ rr rf D L (mkTab D L f0) k, cs.length = k := by
  refine ⟨nodup_level rf f0 k, fun cs hcs => ?_⟩
  obtain ⟨nd, hnd, rfl⟩ := List.mem_map.1 hcs
  exact (shape_level k nd hnd).1

/-! ## The count -/

theorem eval_mono_nat (p : Polynomial ℕ) {a b : ℕ} (h : a ≤ b) : p.eval a ≤ p.eval b := by
  rw [eval_eq_sum, eval_eq_sum, Polynomial.sum_def, Polynomial.sum_def]
  exact Finset.sum_le_sum fun n _ => Nat.mul_le_mul_left _ (Nat.pow_le_pow_left h n)

theorem expandCost_le {F : Type} [Field F] [DecidableEq F] (rf : RootFinder F) (D L : ℕ) (pre : List F)
    (f : ℕ → ℕ → F) :
    expandCost rf D L (pre, mkTab D L f) ≤ (D + 1) * (L + 1) + rf.bound.eval (L + 1) +
      (expand rf D L (pre, mkTab D L f)).length * childCost D L := by
  unfold expandCost expand
  split
  · simp
  · rename_i v _
    simp only [List.length_map]
    have := (rf.cost_le _).trans (eval_mono_nat rf.bound (length_getD_mkTab_le D L f v))
    linarith

/-- **Roth–Ruckenstein's count**: at most `k·(L + 1)·((D + 1)(L + 1) + ρ(L + 1) + childCost D L)`. -/
theorem rrCost_le {F : Type} [Field F] [DecidableEq F] (rf : RootFinder F) {D L : ℕ} (hLD : L ≤ D)
    (f0 : ℕ → ℕ → F) (k : ℕ) :
    rrCost rf D L (mkTab D L f0) k ≤ k * ((L + 1) * ((D + 1) * (L + 1) + rf.bound.eval (L + 1) + childCost D L)) := by
  induction k with
  | zero => simp [rrCost]
  | succ d ih =>
    rw [rrCost]
    have hlev : ((level rf D L (mkTab D L f0) d).map (expandCost rf D L)).sum ≤
        (L + 1) * ((D + 1) * (L + 1) + rf.bound.eval (L + 1)) +
          (level rf D L (mkTab D L f0) (d + 1)).length * childCost D L := by
      rw [level, List.length_flatMap]
      calc ((level rf D L (mkTab D L f0) d).map (expandCost rf D L)).sum
          ≤ ((level rf D L (mkTab D L f0) d).map fun nd =>
              (D + 1) * (L + 1) + rf.bound.eval (L + 1) + (expand rf D L nd).length * childCost D L).sum := by
            refine List.sum_le_sum fun nd hnd => ?_
            obtain ⟨-, f, hf⟩ := shape_level d nd hnd
            obtain ⟨pre, T⟩ := nd
            simp only at hf
            subst hf
            exact expandCost_le rf D L pre f
        _ = (level rf D L (mkTab D L f0) d).length * ((D + 1) * (L + 1) + rf.bound.eval (L + 1)) +
              ((level rf D L (mkTab D L f0) d).map fun nd => (expand rf D L nd).length).sum * childCost D L := by
            rw [List.sum_map_add, List.sum_map_mul_right, List.map_const', List.sum_replicate, smul_eq_mul]
        _ ≤ (L + 1) * ((D + 1) * (L + 1) + rf.bound.eval (L + 1)) +
              ((level rf D L (mkTab D L f0) d).map fun nd => (expand rf D L nd).length).sum * childCost D L :=
            Nat.add_le_add_right (Nat.mul_le_mul_right _ (length_level_le' rf hLD f0 d)) _
    have h2 : (level rf D L (mkTab D L f0) (d + 1)).length ≤ L + 1 := length_level_le' rf hLD f0 (d + 1)
    have h3 : (level rf D L (mkTab D L f0) (d + 1)).length * childCost D L ≤ (L + 1) * childCost D L :=
      Nat.mul_le_mul_right _ h2
    nlinarith

end Claims.GS
