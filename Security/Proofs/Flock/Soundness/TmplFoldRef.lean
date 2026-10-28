import Proofs.Flock.Level3.FoldedBase
import Proofs.Flock.Soundness.SparseRows
import Flock.TmplFold
import Flock.NetAt

/-!
# The template fold's reference, in closed form

`Tmpl.foldRef` is the lincheck fold of a templated unit summed over its unrolled rows at their committed positions
`π = Tmpl.posMap`: entry `s < 64` is `Σ_x Σ_{v ∈ A_x} [π v mod 64 = s] · α · e(π x) · g(π v)` plus the same over `B`
without `α`, with `e(q) = lw[q mod 64] · eq(ρ, ⌊q/64⌋)` and `g(q) = eq(r, ⌊q/64⌋)`.

- `foldRef_eq_rows`: its loops are a `foldl` over the rows (`foldRefRows`).
- `of_eqAt`, `eqTail_halves`: the fold point's `eq` tables, read in two halves, are `eqTable`'s entries.
-/

open Flock Flock.Tmpl FlockLevel3 GF128

namespace FlockSoundness.TmplFold

open FlockSoundness.SparseRows

export FlockSoundness.SparseRows (except_bind_ok)

theorem size_le_pow_log2ceil (n : ℕ) : n ≤ 2 ^ log2ceil n := by
  unfold log2ceil
  split
  · simp only [Nat.pow_zero]; omega
  · have := Nat.lt_log2_self (n := n - 1)
    omega

/-! ## Loops that only yield -/

theorem list_forIn_ok {α β ε : Type} (l : List α) (b : β) (f : α → β → β) :
    forIn (m := Except ε) l b (fun a s => Except.ok (ForInStep.yield (f a s))) =
      Except.ok (l.foldl (fun s a => f a s) b) := by
  induction l generalizing b with
  | nil => rfl
  | cons a l ih => rw [List.forIn_cons]; exact ih _

theorem array_forIn_ok {α β ε : Type} (l : Array α) (b : β) (f : α → β → β) :
    forIn (m := Except ε) l b (fun a s => Except.ok (ForInStep.yield (f a s))) =
      Except.ok (l.foldl (fun s a => f a s) b) := by
  rw [← Array.forIn_toList, list_forIn_ok, Array.foldl_toList]

/-! ## `foldRef` as a fold over the rows -/

/-- One logical row's entries added at their positions: `A`'s weighted by `α`, `B`'s by 1. -/
def rowStep (pt : FoldPt) (π : Array ℕ) (ra rb : Array (Array ℕ)) (p : Array F128) (x : ℕ) : Array F128 :=
  let ex := pt.e (π.getD x 0)
  let p := (ra.getD x #[]).foldl (fun p v => addAt p (π.getD v 0) (pt.alpha * ex * pt.g (π.getD v 0))) p
  (rb.getD x #[]).foldl (fun p v => addAt p (π.getD v 0) (ex * pt.g (π.getD v 0))) p

/-- `foldRef`'s loops, on rows `ra`, `rb` at positions `π`. -/
def foldRefRows (pt : FoldPt) (π : Array ℕ) (ra rb : Array (Array ℕ)) : Array F128 :=
  (List.range ra.size).foldl (rowStep pt π ra rb) (Array.replicate 64 0)

theorem foldRef_eq_rows (t : Tmpl) (lays : Array Lay) (alpha : F128) (lw rho r : Array F128) :
    foldRef t lays alpha lw rho r = (do
      let acc ← t.rows
      pure (foldRefRows { alpha, lw, rho, r, qc := 0 } (posMap t lays) acc.ra acc.rb)) := by
  unfold foldRef foldRefRows rowStep
  cases t.rows with
  | error e => rfl
  | ok acc =>
    simp only [bind, Except.bind, pure, Except.pure]
    simp [Std.Legacy.Range.forIn_eq_forIn_range', array_forIn_ok, list_forIn_ok, List.range_eq_range']

/-! ## `addAt`, read back -/

theorem addAt_size (p : Array F128) (q : ℕ) (v : F128) : (addAt p q v).size = p.size := by simp [addAt]

theorem addAt_get (p : Array F128) (q : ℕ) (v : F128) {s : ℕ} (hs : s < p.size) :
    of ((addAt p q v).getD s 0) = of (p.getD s 0) + if q % 64 = s then of v else 0 := by
  unfold addAt
  by_cases h : q % 64 = s
  · subst h
    simp [Array.getD_eq_getD_getElem?, hs, Array.getElem_modify_self, of_add]
  · simp [Array.getD_eq_getD_getElem?, Array.getElem?_modify, h]

theorem foldl_addAt_size (l : List ℕ) (qf : ℕ → ℕ) (w : ℕ → F128) (p : Array F128) :
    (l.foldl (fun p v => addAt p (qf v) (w v)) p).size = p.size := by
  induction l generalizing p with
  | nil => rfl
  | cons a l ih => simp only [List.foldl_cons]; rw [ih, addAt_size]

theorem foldl_addAt_get (l : List ℕ) (qf : ℕ → ℕ) (w : ℕ → F128) (p : Array F128) {s : ℕ} (hs : s < p.size) :
    of ((l.foldl (fun p v => addAt p (qf v) (w v)) p).getD s 0) =
      of (p.getD s 0) + (l.map fun v => if qf v % 64 = s then of (w v) else 0).sum := by
  induction l generalizing p with
  | nil => simp
  | cons a l ih =>
    simp only [List.foldl_cons, List.map_cons, List.sum_cons]
    rw [ih _ (by rw [addAt_size]; exact hs), addAt_get p _ _ hs, add_assoc]

/-! ## `foldRef`'s entries -/

/-- Logical row `x`'s share of entry `s`. -/
noncomputable def rowTerm (pt : FoldPt) (π : Array ℕ) (ra rb : Array (Array ℕ)) (s x : ℕ) : GF128 :=
  ((ra.getD x #[]).toList.map fun v =>
      if π.getD v 0 % 64 = s then of (pt.alpha * pt.e (π.getD x 0) * pt.g (π.getD v 0)) else 0).sum +
    ((rb.getD x #[]).toList.map fun v =>
      if π.getD v 0 % 64 = s then of (pt.e (π.getD x 0) * pt.g (π.getD v 0)) else 0).sum

section
variable (pt : FoldPt) (π : Array ℕ) (ra rb : Array (Array ℕ))

theorem rowStep_size (p : Array F128) (x : ℕ) : (rowStep pt π ra rb p x).size = p.size := by
  unfold rowStep
  dsimp only
  simp only [← Array.foldl_toList]
  rw [foldl_addAt_size, foldl_addAt_size]

theorem rowStep_get (p : Array F128) (x : ℕ) {s : ℕ} (hs : s < p.size) :
    of ((rowStep pt π ra rb p x).getD s 0) = of (p.getD s 0) + rowTerm pt π ra rb s x := by
  unfold rowStep rowTerm
  dsimp only
  simp only [← Array.foldl_toList]
  rw [foldl_addAt_get _ _ _ _ (by rw [foldl_addAt_size]; exact hs), foldl_addAt_get _ _ _ _ hs, add_assoc]

theorem foldl_rowStep_get (l : List ℕ) (p : Array F128) {s : ℕ} (hs : s < p.size) :
    of ((l.foldl (rowStep pt π ra rb) p).getD s 0) = of (p.getD s 0) + (l.map (rowTerm pt π ra rb s)).sum := by
  induction l generalizing p with
  | nil => simp
  | cons a l ih =>
    simp only [List.foldl_cons, List.map_cons, List.sum_cons]
    rw [ih _ (by rw [rowStep_size]; exact hs), rowStep_get _ _ _ _ _ _ hs, add_assoc]

/-- **`foldRef`'s entry `s`**: every logical row's share, summed. -/
theorem foldRefRows_size : (foldRefRows pt π ra rb).size = 64 := by
  unfold foldRefRows
  suffices h : ∀ (l : List ℕ) (p : Array F128), (l.foldl (rowStep pt π ra rb) p).size = p.size by
    rw [h]; simp
  intro l
  induction l with
  | nil => intro p; rfl
  | cons a l ih => intro p; rw [List.foldl_cons, ih, rowStep_size]

theorem foldRefRows_get {s : ℕ} (hs : s < 64) :
    of ((foldRefRows pt π ra rb).getD s 0) = ((List.range ra.size).map (rowTerm pt π ra rb s)).sum := by
  unfold foldRefRows
  rw [foldl_rowStep_get _ _ _ _ _ _ (by simpa using hs)]
  simp [Array.getD_eq_getD_getElem?, hs, ← zero_eq]

end

/-! ## The fold point's `eq` tables -/

/-- `Tmpl.eqAt r lo n u` is `eq(r[lo .. lo + n], u)` below `2^n`. -/
theorem of_eqAt (r : Array F128) (lo : ℕ) (n u : ℕ) (hu : u < 2 ^ n) :
    of (Tmpl.eqAt r lo n u) = FlockLevel3.eqAt (fun j : Fin n => of (r.getD (lo + j.val) 0)) u := by
  unfold Tmpl.eqAt
  simp only [show ¬ 2 ^ n ≤ u from Nat.not_le.mpr hu, ↓reduceIte]
  clear hu
  induction n with
  | zero => simp [FlockLevel3.eqAt, ← one_eq]
  | succ n ih =>
    simp only [Nat.fold_succ]
    rw [of_mul, ih]
    unfold FlockLevel3.eqAt
    rw [Fin.prod_univ_castSucc]
    simp only [Fin.val_castSucc, Fin.val_last]
    congr 1
    by_cases hb : u.testBit n
    · simp [hb]
    · simp [hb, of_add, ← one_eq, add_comm]

/-- `eq(r, u)` read in two halves (`halves`, `eqTail`) is `eqTable r`'s entry `u`. -/
theorem eqTail_halves (r : Array F128) (u : ℕ) (hu : u < 2 ^ r.size) :
    of (Tmpl.eqTail (Tmpl.halves r) 0 u) = of (eqTable r)[u]! := by
  have hh : r.size / 2 ≤ r.size := Nat.div_le_self _ _
  have h1 : u % 2 ^ (r.size / 2) < 2 ^ (r.size / 2) := Nat.mod_lt _ (Nat.two_pow_pos _)
  have h2 : u / 2 ^ (r.size / 2) < 2 ^ (r.size - r.size / 2) := by
    rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _), ← Nat.pow_add, Nat.sub_add_cancel hh]
    exact hu
  have hT : (Tmpl.halves r).getD 0 (0, #[], #[]) =
      (r.size / 2, (Array.range (2 ^ (r.size / 2))).map (Tmpl.eqAt r 0 (r.size / 2)),
        (Array.range (2 ^ (r.size - r.size / 2))).map (Tmpl.eqAt r (r.size / 2) (r.size - r.size / 2))) := by
    simp [Tmpl.halves]
  unfold Tmpl.eqTail
  rw [hT]
  dsimp only
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range, h1, h2, ↓reduceIte,
    Option.map_some, Option.getD_some]
  rw [of_mul, of_eqAt r 0 _ _ h1, of_eqAt r _ _ _ h2, eqTable_eqAt r u hu]
  have e1 : FlockLevel3.eqAt (fun j : Fin r.size => of r[j.val]!) u =
      FlockLevel3.eqAt (fun j : Fin (r.size / 2 + (r.size - r.size / 2)) => of (r.getD j.val 0)) u := by
    unfold FlockLevel3.eqAt
    refine Fintype.prod_equiv (finCongr (by omega)) _ _ (fun j => ?_)
    simp only [finCongr_apply, Fin.val_cast]
    rw [getElem!_pos r j.val j.isLt, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem j.isLt]
    rfl
  rw [e1, eqAt_split]
  simp only [Fin.val_castAdd, Fin.val_natAdd, Nat.zero_add]

/-! ## The committed side: `project ∘ baseOf` -/

theorem list_sum_range {β : Type} [AddCommMonoid β] (f : ℕ → β) : ∀ l : List ℕ,
    (l.map f).sum = ∑ t ∈ Finset.range l.length, f (l.getD t 0)
  | [] => by simp
  | a :: l => by
    rw [List.map_cons, List.sum_cons, list_sum_range f l, List.length_cons, Finset.sum_range_succ']
    simp [add_comm]

theorem project_get (base eLo : Array F128) {s : ℕ} (hs : s < 64) :
    of (Stmt.project base eLo)[s]! = ∑ u ∈ Finset.range eLo.size, of base[s + 64 * u]! * of eLo[u]! := by
  unfold Stmt.project
  rw [getElem!_pos _ s (by simp [hs])]
  simp only [Array.getElem_ofFn]
  rw [sumUpTo_eq]
  simp only [of_mul]

theorem tensor_get (lw rho : Array F128) (hlw : lw.size = 64) {p : ℕ} (hp : p < 64 * 2 ^ rho.size) :
    of (Stmt.tensor lw rho)[p]! = of (lw.getD (p % 64) 0 * Tmpl.eqTail (Tmpl.halves rho) 0 (p / 64)) := by
  have hq : p / 64 < 2 ^ rho.size := by omega
  have hsz : p < (Stmt.tensor lw rho).size := by simp [Stmt.tensor, hlw, eqTable_size]; omega
  rw [getElem!_pos _ p hsz]
  simp only [Stmt.tensor, Array.getElem_ofFn, hlw]
  rw [of_mul, of_mul, eqTail_halves rho _ hq]
  congr 1

theorem sparse_ofRows_dims {rows : Array (Array ℕ)} {width : ℕ} {m : Sparse} (h : Sparse.ofRows rows width = .ok m) :
    m.rows = rows.size ∧ m.width = width := by
  unfold Sparse.ofRows at h
  simp only at h
  obtain ⟨st, -, h⟩ := except_bind_ok h
  cases h
  exact ⟨rfl, rfl⟩

/-- Row `p` of a matrix `Sparse.ofRows` built: its length and its columns are `rows[p]`'s. -/
theorem sparse_col {rows : Array (Array ℕ)} {width : ℕ} {m : Sparse} (h : Sparse.ofRows rows width = .ok m)
    (hw : width ≤ 2 ^ 32) {p : ℕ} (hp : p < rows.size) :
    m.len[p]! = (rows.getD p #[]).size ∧ ∀ t < (rows.getD p #[]).size, m.col p t = (rows.getD p #[])[t]! := by
  obtain ⟨-, hs, hl, -⟩ := ofRows_spec h
  have hsz : p < m.start.size := by
    have := congrArg List.length hs
    simp only [Array.length_toList, rowStarts_length] at this
    omega
  have hlsz : p < m.len.size := by
    have := congrArg List.length hl
    simp only [Array.length_toList, List.length_map] at this
    omega
  have hlen : m.len.getD p 0 = (rows.getD p #[]).size := by
    simp only [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hl, List.getElem?_map]
    cases rows.toList[p]? <;> rfl
  have hrow := ofRows_row h hw p
  refine ⟨?_, fun t ht => ?_⟩
  · rw [getElem!_pos _ p hlsz]
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlsz] at hlen
    exact hlen
  · have ht' : t < (m.row p).size := by rw [hrow]; exact ht
    have e : (m.row p)[t]! = m.col p t := by
      rw [getElem!_pos _ t ht']
      simp only [Sparse.row, Sparse.col, Array.getElem_map, Array.getElem_range, Array.getD_eq_getD_getElem?,
        Array.getElem?_eq_getElem hsz, Option.getD_some, getElem!_pos m.start p hsz]
    rw [← e, hrow]

/-- `(Xᵀe)[i']` for a matrix `Sparse.ofRows` built, over its rows. -/
theorem sparseT_rows {rows : Array (Array ℕ)} {W : ℕ} {m : Sparse} (h : Sparse.ofRows rows W = .ok m)
    (hw : W ≤ 2 ^ 32) (eLo : Array F128) (hN : rows.size ≤ eLo.size) (i' : ℕ) :
    sparseT m eLo i' = ∑ p ∈ Finset.range rows.size, ∑ t ∈ Finset.range (rows.getD p #[]).size,
      if (rows.getD p #[])[t]! = i' then of eLo[p]! else 0 := by
  unfold sparseT
  rw [(sparse_ofRows_dims h).1, Nat.min_eq_left hN]
  refine Finset.sum_congr rfl fun p hp => ?_
  obtain ⟨hl, hc⟩ := sparse_col h hw (Finset.mem_range.mp hp)
  rw [hl]
  refine Finset.sum_congr rfl fun t ht => ?_
  rw [hc t (Finset.mem_range.mp ht)]

/-- A column's share of entry `s` over the slot's `M` words. -/
theorem sum_slot (M c s : ℕ) (hs : s < 64) (hc : c < 64 * M) (f : ℕ → GF128) :
    ∑ u ∈ Finset.range M, (if c = s + 64 * u then f u else 0) = if c % 64 = s then f (c / 64) else 0 := by
  by_cases h : c % 64 = s
  · rw [if_pos h, Finset.sum_eq_single (c / 64)]
    · rw [if_pos (show c = s + 64 * (c / 64) by omega)]
    · intro u _ hu
      rw [if_neg (show ¬ c = s + 64 * u by omega)]
    · intro hm
      exact absurd (Finset.mem_range.mpr (by omega)) hm
  · rw [if_neg h]
    exact Finset.sum_eq_zero fun u _ => if_neg (show ¬ c = s + 64 * u by omega)

/-- One matrix's share of `project ∘ baseOf`, regrouped by row. -/
theorem side_sum {rows : Array (Array ℕ)} {W M : ℕ} {m : Sparse} (h : Sparse.ofRows rows W = .ok m)
    (hw : W ≤ 2 ^ 32) (hW : W = 64 * M) (eLo : Array F128) (hN : rows.size ≤ eLo.size) (w : GF128)
    (g : ℕ → GF128) {s : ℕ} (hs : s < 64) :
    ∑ u ∈ Finset.range M, w * sparseT m eLo (s + 64 * u) * g u =
      ∑ p ∈ Finset.range rows.size, ((rows.getD p #[]).toList.map fun c =>
        if c % 64 = s then w * of eLo[p]! * g (c / 64) else 0).sum := by
  have hcol := (ofRows_spec h).2.2.2
  have e1 : ∀ u ∈ Finset.range M, w * sparseT m eLo (s + 64 * u) * g u =
      ∑ p ∈ Finset.range rows.size, ∑ t ∈ Finset.range (rows.getD p #[]).size,
        if (rows.getD p #[])[t]! = s + 64 * u then w * of eLo[p]! * g u else 0 := by
    intro u _
    rw [sparseT_rows h hw eLo hN]
    refine ((congrArg (· * g u) (Finset.mul_sum _ _ _)).trans (Finset.sum_mul _ _ _)).trans
      (Finset.sum_congr rfl fun p _ => ?_)
    refine ((congrArg (· * g u) (Finset.mul_sum _ _ _)).trans (Finset.sum_mul _ _ _)).trans
      (Finset.sum_congr rfl fun t _ => ?_)
    split
    · rfl
    · exact (congrArg (· * g u) (mul_zero w)).trans (zero_mul _)
  rw [Finset.sum_congr rfl e1, Finset.sum_comm]
  refine Finset.sum_congr rfl fun p hp => ?_
  rw [Finset.sum_comm, list_sum_range, Array.length_toList]
  refine Finset.sum_congr rfl fun t ht => ?_
  have hp' := Finset.mem_range.mp hp
  have ht' := Finset.mem_range.mp ht
  have hpt : (rows.getD p #[])[t]! < 64 * M := by
    rw [← hW]
    refine hcol (rows.getD p #[]) ?_ _ ?_
    · simp only [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hp', Option.getD_some]
      exact Array.getElem_mem_toList hp'
    · rw [getElem!_pos _ t ht']
      exact Array.getElem_mem_toList ht'
  have e : (rows.getD p #[]).toList.getD t 0 = (rows.getD p #[])[t]! := by
    rw [getElem!_pos _ t ht', List.getD_eq_getElem?_getD, Array.getElem?_toList, Array.getElem?_eq_getElem ht']
    rfl
  rw [e]
  exact sum_slot M _ s hs hpt (fun u => w * of eLo[p]! * g u)

/-- **`project ∘ baseOf` on matrices `Sparse.ofRows` built**, regrouped by row: row `p`'s columns `c`, each at entry
`c mod 64` with the row's `e_lo` weight and `eq(r, ⌊c/64⌋)`. -/
theorem project_baseOf_rows {rowsA rowsB : Array (Array ℕ)} {sl : ℕ} {A B : Sparse}
    (hA : Sparse.ofRows rowsA (2 ^ sl) = .ok A) (hB : Sparse.ofRows rowsB (2 ^ sl) = .ok B) (hsl : sl ≤ 32)
    (alpha : F128) (lw rho r : Array F128) (hlw : lw.size = 64) (hrho : rho.size + 6 = sl) (hr : r.size = rho.size)
    (hNA : rowsA.size ≤ 2 ^ sl) (hNB : rowsB.size = rowsA.size) {s : ℕ} (hs : s < 64) :
    of (Stmt.project (Stmt.baseOf alpha (Stmt.tensor lw rho) A B none sl) (eqTable r))[s]! =
      ∑ p ∈ Finset.range rowsA.size,
        (((rowsA.getD p #[]).toList.map fun c =>
            if c % 64 = s then of alpha * of (Stmt.tensor lw rho)[p]! * of (eqTable r)[c / 64]! else 0).sum +
          ((rowsB.getD p #[]).toList.map fun c =>
            if c % 64 = s then 1 * of (Stmt.tensor lw rho)[p]! * of (eqTable r)[c / 64]! else 0).sum) := by
  subst hrho
  have hW : 2 ^ (rho.size + 6) = 64 * 2 ^ r.size := by rw [hr, pow_add]; ring
  have hE : (Stmt.tensor lw rho).size = 2 ^ (rho.size + 6) := by
    rw [hW, hr]
    simp [Stmt.tensor, hlw, eqTable_size]
  have hw : 2 ^ (rho.size + 6) ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) hsl
  have hAw := (sparse_ofRows_dims hA).2
  have hBw := (sparse_ofRows_dims hB).2
  rw [project_get _ _ hs, eqTable_size]
  have step : ∀ u ∈ Finset.range (2 ^ r.size),
      of (Stmt.baseOf alpha (Stmt.tensor lw rho) A B none (rho.size + 6))[s + 64 * u]! * of (eqTable r)[u]! =
        of alpha * sparseT A (Stmt.tensor lw rho) (s + 64 * u) * of (eqTable r)[u]! +
          1 * sparseT B (Stmt.tensor lw rho) (s + 64 * u) * of (eqTable r)[u]! := by
    intro u hu
    have hi : s + 64 * u < 2 ^ (rho.size + 6) := by
      have := Finset.mem_range.mp hu
      rw [hW]
      omega
    rw [baseOf_get _ _ _ _ _ _ _ hi (by rw [hAw]; exact hi) (by rw [hBw]; exact hi)]
    simp [lookupTerm, add_mul]
  rw [Finset.sum_congr rfl step, Finset.sum_add_distrib,
    side_sum hA hw hW _ (by rw [hE]; exact hNA) (of alpha) (fun u => of (eqTable r)[u]!) hs,
    side_sum hB hw hW _ (by rw [hE, hNB]; exact hNA) 1 (fun u => of (eqTable r)[u]!) hs, hNB,
    ← Finset.sum_add_distrib]

/-! ## The committed rows, back to the logical ones -/

/-- A sum over the committed rows (`Net.placed`) is the sum over the logical rows, each at its position: `pos` is
one-to-one, and the gaps are empty. -/
theorem sum_placed {useful inCols size : ℕ} {pos : Array ℕ} {inv : Array (Option ℕ)}
    (h : Net.posOk useful inCols size pos inv = true) (ra rb : Array (Array ℕ))
    (F : ℕ → Array ℕ → Array ℕ → GF128) (hF : ∀ p, F p #[] #[] = 0) :
    ∑ p ∈ Finset.range size, F p ((Net.placed pos inv ra).getD p #[]) ((Net.placed pos inv rb).getD p #[]) =
      ∑ x ∈ Finset.range useful,
        F (pos.getD x 0) ((ra.getD x #[]).map (pos.getD · 0)) ((rb.getD x #[]).map (pos.getD · 0)) := by
  have e : ∀ x ∈ Finset.range useful,
      F (pos.getD x 0) ((ra.getD x #[]).map (pos.getD · 0)) ((rb.getD x #[]).map (pos.getD · 0)) =
        F (pos.getD x 0) ((Net.placed pos inv ra).getD (pos.getD x 0) #[])
          ((Net.placed pos inv rb).getD (pos.getD x 0) #[]) := fun x hx => by
    rw [Net.placed_at h ra (Finset.mem_range.mp hx), Net.placed_at h rb (Finset.mem_range.mp hx)]
  have hinj : Set.InjOn (fun x => pos.getD x 0) (Finset.range useful : Set ℕ) := by
    intro x hx y hy he
    exact Net.posOk_inj h (by simpa using hx) (by simpa using hy) he
  refine Eq.trans ?_ ((Finset.sum_congr rfl e).trans
    (Finset.sum_image (f := fun p => F p ((Net.placed pos inv ra).getD p #[]) ((Net.placed pos inv rb).getD p #[]))
      hinj).symm).symm
  refine (Finset.sum_subset (fun p hp => ?_) fun p _ hp => ?_).symm
  · obtain ⟨x, hx, rfl⟩ := Finset.mem_image.mp hp
    exact Finset.mem_range.mpr (Net.posOk_lt h (Finset.mem_range.mp hx))
  · have hn : ∀ x < useful, pos.getD x 0 ≠ p := fun x hx he =>
      hp (Finset.mem_image.mpr ⟨x, Finset.mem_range.mpr hx, he⟩)
    show F p ((Net.placed pos inv ra).getD p #[]) ((Net.placed pos inv rb).getD p #[]) = 0
    rw [Net.placed_gap h ra hn, Net.placed_gap h rb hn, hF]

/-! ## The unrolled rows' count -/

theorem write_size {acc acc' : Acc} {c : ℕ} {a b : Array ℕ} (h : write acc c a b = .ok acc') :
    acc'.ra.size = acc.ra.size ∧ acc'.rb.size = acc.rb.size := by
  unfold write at h
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  · cases h
    simp

theorem foldlM_inv {α : Type} (P : Acc → Prop) (f : Acc → α → V Acc)
    (hf : ∀ acc a acc', f acc a = .ok acc' → P acc → P acc') :
    ∀ (l : List α) (acc acc' : Acc), l.foldlM f acc = .ok acc' → P acc → P acc'
  | [], acc, acc', h, hp => by
    rw [List.foldlM_nil] at h
    cases h
    exact hp
  | a :: l, acc, acc', h, hp => by
    rw [List.foldlM_cons] at h
    obtain ⟨b, hb, h⟩ := except_bind_ok h
    exact foldlM_inv P f hf l b acc' h (hf acc a b hb hp)

theorem place_size (defs : Array Def) (k : ℕ) : ∀ (fuel d o : ℕ) (inp : List ℕ) (acc acc' : Acc),
    place defs k fuel d o inp acc = .ok acc' → acc'.ra.size = acc.ra.size ∧ acc'.rb.size = acc.rb.size
  | 0, d, o, inp, acc, acc', h => by
    rw [place] at h
    simp [throw, throwThe, MonadExceptOf.throw] at h
  | fuel + 1, d, o, inp, acc, acc', h => by
    rw [place] at h
    cases hd : defs[d]? with
    | none => simp [hd, throw, throwThe, MonadExceptOf.throw] at h
    | some df =>
      simp only [hd] at h
      obtain ⟨acc1, h1, h2⟩ := except_bind_ok h
      let P : Acc → Prop := fun a => a.ra.size = acc.ra.size ∧ a.rb.size = acc.rb.size
      have p1 : P acc1 := foldlM_inv P _ (fun a _ b hab ha => by
          have e := write_size hab
          exact ⟨e.1.trans ha.1, e.2.trans ha.2⟩) _ acc acc1 h1 ⟨rfl, rfl⟩
      exact foldlM_inv P _ (fun a _ b hab ha => foldlM_inv P _ (fun a' _ b' hab' ha' => by
          have e := place_size defs k fuel _ _ _ a' b' hab'
          exact ⟨e.1.trans ha'.1, e.2.trans ha'.2⟩) _ a b hab ha) _ acc1 acc' h2 p1

/-- `Tmpl.rows` has one row per column of the root. -/
theorem rows_size {t : Tmpl} {acc : Acc} (h : t.rows = .ok acc) :
    acc.ra.size = (t.defs.getD t.root default).cols ∧ acc.rb.size = (t.defs.getD t.root default).cols := by
  unfold Tmpl.rows at h
  obtain ⟨a, ha, h⟩ := except_bind_ok h
  have h1 := place_size _ _ _ _ _ _ _ _ ha
  have h2 := write_size h
  simp only [Acc.empty, Array.size_replicate] at h1
  exact ⟨h2.1.trans h1.1, h2.2.trans h1.2⟩

/-! ## The fold's reference is `project ∘ baseOf` -/

theorem getElem!_getD (p : Array F128) {s : ℕ} (hs : s < p.size) : p[s]! = p.getD s 0 := by
  rw [getElem!_pos p s hs, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hs, Option.getD_some]

theorem list_range_sum {β : Type} [AddCommMonoid β] (f : ℕ → β) (n : ℕ) :
    ((List.range n).map f).sum = ∑ x ∈ Finset.range n, f x := by
  induction n with
  | zero => simp
  | succ n ih => simp [List.range_succ, Finset.sum_range_succ, ih]

/-- **The template fold's reference is `project ∘ baseOf`** at the matrices `unrollAt` commits: entry `s < 64` of
`foldRef` is entry `s` of `project (baseOf α (lw ⊗ eq(ρ, ·)) A B none unitLog) (eq(r, ·))`. -/
theorem foldRef_eq_project {t : Tmpl} {lays : Array Lay} {alpha : F128} {lw rho r P : Array F128} {n' : Net}
    (hP : foldRef t lays alpha lw rho r = .ok P) (hU : Flock.Tmpl.unrollAt t lays = .ok n')
    (hlw : lw.size = 64) (hrho : rho.size + 6 = n'.unitLog) (hr : r.size = rho.size) (hsl : n'.unitLog ≤ 32)
    {s : ℕ} (hs : s < 64) :
    of P[s]! = of (Stmt.project (Stmt.baseOf alpha (Stmt.tensor lw rho) n'.a n'.b none n'.unitLog) (eqTable r))[s]! := by
  rw [foldRef_eq_rows] at hP
  obtain ⟨acc, hR, hP⟩ := except_bind_ok hP
  cases hP
  unfold Flock.Tmpl.unrollAt at hU
  obtain ⟨acc2, hR2, hU⟩ := except_bind_ok hU
  rw [hR] at hR2
  cases hR2
  obtain ⟨n, -, -, -, hul, ha, hb, -, -, -, -, -, hpos, -, -⟩ := Net.ofRowsAt_ok hU
  generalize posMap t lays = π at ha hb hpos hul ⊢
  have hfit : π.getD t.constPos 0 + 1 ≤ 2 ^ n'.unitLog := by
    rw [hul]
    exact (size_le_pow_log2ceil _).trans (Nat.pow_le_pow_right (by norm_num) (le_max_right _ _))
  generalize Net.invOf π (π.getD t.constPos 0 + 1) = inv at ha hb hpos
  generalize π.getD t.constPos 0 + 1 = size at hpos hfit
  obtain ⟨-, hinv, -, -, -⟩ := Net.posOk_parts hpos
  have hNA : (Net.placed π inv acc.ra).size ≤ 2 ^ n'.unitLog := by rw [Net.placed_size, hinv]; exact hfit
  have hNB : (Net.placed π inv acc.rb).size = (Net.placed π inv acc.ra).size := by
    rw [Net.placed_size, Net.placed_size]
  rw [project_baseOf_rows ha hb hsl alpha lw rho r hlw hrho hr hNA hNB hs, Net.placed_size, hinv]
  refine Eq.trans ?_ (sum_placed hpos acc.ra acc.rb (fun p ra' rb' =>
      (ra'.toList.map fun c =>
          if c % 64 = s then of alpha * of (Stmt.tensor lw rho)[p]! * of (eqTable r)[c / 64]! else 0).sum +
        (rb'.toList.map fun c =>
          if c % 64 = s then 1 * of (Stmt.tensor lw rho)[p]! * of (eqTable r)[c / 64]! else 0).sum)
      fun p => by simp).symm
  rw [getElem!_getD _ (by rw [foldRefRows_size]; exact hs), foldRefRows_get _ _ _ _ hs, list_range_sum,
    (rows_size hR).1]
  refine Finset.sum_congr rfl fun x hx => ?_
  have hx' := Finset.mem_range.mp hx
  have hpx : π.getD x 0 < 64 * 2 ^ rho.size := by
    have := Net.posOk_lt hpos hx'
    have h64 : 2 ^ n'.unitLog = 64 * 2 ^ rho.size := by rw [← hrho, pow_add]; ring
    omega
  have hcol : ∀ (rows : Array (Array ℕ)) (m : Sparse),
      Sparse.ofRows (Net.placed π inv rows) (2 ^ n'.unitLog) = .ok m →
      ∀ v ∈ (rows.getD x #[]).toList, π.getD v 0 / 64 < 2 ^ r.size := by
    intro rows m hm v hv
    have hrow := Net.placed_at hpos rows hx'
    have hlt : π.getD x 0 < (Net.placed π inv rows).size := by
      rw [Net.placed_size, hinv]; exact Net.posOk_lt hpos hx'
    have hmem : (Net.placed π inv rows).getD (π.getD x 0) #[] ∈ (Net.placed π inv rows).toList := by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some]
      exact Array.getElem_mem_toList hlt
    have hc := (ofRows_spec hm).2.2.2 _ hmem (π.getD v 0) (by
      rw [hrow, Array.toList_map]; exact List.mem_map.mpr ⟨v, hv, rfl⟩)
    have h64 : 2 ^ n'.unitLog = 64 * 2 ^ r.size := by rw [← hrho, hr, pow_add]; ring
    omega
  unfold rowTerm
  simp only [Array.toList_map, List.map_map, Function.comp_def]
  refine congrArg₂ (· + ·) (congrArg List.sum (List.map_congr_left fun v hv => ?_))
    (congrArg List.sum (List.map_congr_left fun v hv => ?_))
  · by_cases hc : π.getD v 0 % 64 = s
    · simp only [hc, ↓reduceIte]
      rw [of_mul, of_mul, tensor_get lw rho hlw hpx, ← eqTail_halves r _ (hcol acc.ra n'.a ha v hv)]
      all_goals rfl
    · simp only [hc, ↓reduceIte]
  · by_cases hc : π.getD v 0 % 64 = s
    · simp only [hc, ↓reduceIte]
      rw [of_mul, tensor_get lw rho hlw hpx, ← eqTail_halves r _ (hcol acc.rb n'.b hb v hv), one_mul]
      all_goals rfl
    · simp only [hc, ↓reduceIte]

end FlockSoundness.TmplFold
