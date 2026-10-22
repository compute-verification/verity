import Proofs.Flock.Level3.Folded
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
    simp [Array.getD_eq_getD_getElem?, Array.getElem?_modify, hs, of_add]
  · simp [Array.getD_eq_getD_getElem?, Array.getElem?_modify, h]

/-! ## The fold point's `eq` tables -/

/-- `Tmpl.eqAt r lo n u` is `eq(r[lo .. lo + n], u)` below `2^n`. -/
theorem of_eqAt (r : Array F128) (lo : ℕ) (n u : ℕ) (hu : u < 2 ^ n) :
    of (Tmpl.eqAt r lo n u) = FlockLevel3.eqAt (fun j : Fin n => of (r.getD (lo + j.val) 0)) u := by
  unfold Tmpl.eqAt
  rw [if_neg (Nat.not_le.mpr hu)]
  clear hu
  induction n with
  | zero => simp [FlockLevel3.eqAt, ← one_eq]
  | succ n ih =>
    simp only [Nat.fold_succ]
    rw [of_mul, ih]
    unfold FlockLevel3.eqAt
    rw [Fin.prod_univ_castSucc]
    simp only [Fin.coe_castSucc, Fin.val_last]
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
    simp only [finCongr_apply, Fin.coe_cast]
    rw [getElem!_pos r j.val j.isLt, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem j.isLt]
    rfl
  rw [e1, eqAt_split]
  simp only [Fin.coe_castAdd, Fin.coe_natAdd, Nat.zero_add]

end FlockSoundness.TmplFold
