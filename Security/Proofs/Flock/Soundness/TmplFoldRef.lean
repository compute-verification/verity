import Proofs.Flock.Level3.Folded
import Proofs.Flock.Soundness.ExecRows
import Flock.TmplFold
import Flock.NetAt

/-!
# The template fold's reference, in closed form (PRF-133, part 3)

`Tmpl.foldRef` is the lincheck fold of a templated unit summed over its unrolled rows at their committed positions
`π = Tmpl.posMap`: entry `s < 64` is `Σ_x Σ_{v ∈ A_x} [π v mod 64 = s] · α · e(π x) · g(π v)` plus the same over `B`
without `α`, with `e(q) = lw[q mod 64] · eq(ρ, ⌊q/64⌋)` and `g(q) = eq(r, ⌊q/64⌋)`.

- `foldRef_eq_rows`: its loops are a `foldl` over the rows (`foldRefRows`).
- `of_eqAt`, `eqTail_halves`: the fold point's `eq` tables, read in two halves, are `eqTable`'s entries.
-/

open Flock Flock.Tmpl FlockLevel3 GF128

namespace FlockSoundness.TmplFold

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

end FlockSoundness.TmplFold
