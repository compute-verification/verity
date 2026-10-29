import Proofs.Flock.Soundness.TmplFoldClassOk
import Proofs.Flock.Soundness.TmplPosMap

/-!
# The template fold is its reference

`fold_refVal` has the unit's fold as the root copy's `refVal`, which reads every column at `A + B · posIn`; `foldRef_tree`
has the reference as the call tree's `placeSum`, which reads every column at `π = posMap`. They agree wherever each copy
`(d, o)` of the unrolled unit, and every copy below it, has its own column `y` at `π[o + y] = A + B · posIn y`
(`TreeAt`), which `posMap_tree` proves whenever every Definition's layout is in order (`DefPos`).

- `refVal_place`: a copy's `refVal`, its inputs read through `π`, is its `placeSum`.
- `fold_eq_foldRef`: the fold and its reference agree entry by entry.
-/

open Flock Flock.Tmpl FlockLevel3 GF128

namespace FlockSoundness.TmplFold

theorem getD_map_at (l : List ℕ) (f : ℕ → ℕ) (i k d : ℕ) (hd : f k = d) : (l.map f).getD i d = f (l.getD i k) := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases l[i]? <;> simp [hd]

/-- Where a copy at `(A, B)` reads reference `v` is where `π` puts the column `Def.ref` names. -/
theorem readAt_ref (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (π : Array ℕ) {d o A B : ℕ} (inp : List ℕ)
    (hk : π.getD t.constPos 0 = pt.qc)
    (hcopy : ∀ y < (t.defs.getD d default).cols,
      π.getD (o + y) 0 = A + B * Flock.Tmpl.posIn t lays (t.defs.size + 1) d y) (v : ℕ) :
    readAt t lays pt d A B (inp.map fun x => π.getD x 0) v = π.getD ((t.defs.getD d default).ref o inp t.constPos v) 0 := by
  unfold readAt Def.ref
  split_ifs with h1 h2
  · exact (hcopy v h1).symm
  · exact getD_map_at inp (fun x => π.getD x 0) _ t.constPos _ hk
  · exact hk.symm

/-- A row of the copy at column `x`, its references read at `R v = π[F v]`: its `rowVal` is its `rowShare`. -/
theorem row_place (pt : FoldPt) (π : Array ℕ) (s x : ℕ) (R F : ℕ → ℕ) (r : ℕ × Array ℕ × Array ℕ)
    (hR : ∀ v, R v = π.getD (F v) 0) :
    rowVal pt s (pt.e (π.getD x 0)) R r = rowShare (termA pt π s) (termB pt π s) x (r.2.1.map F) (r.2.2.map F) := by
  have keyA : ∀ v, of (pt.alpha * pt.e (π.getD x 0)) * Gs pt s (R v) = termA pt π s x (F v) := by
    intro v
    rw [hR v]
    unfold Gs termA
    split_ifs <;> simp only [of_mul, mul_zero]
  have keyB : ∀ v, of ((1 : F128) * pt.e (π.getD x 0)) * Gs pt s (R v) = termB pt π s x (F v) := by
    intro v
    rw [hR v]
    unfold Gs termB
    split_ifs <;> simp only [of_mul, ← one_eq, one_mul, mul_zero]
  unfold rowVal rowShare
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, Array.toList_map, List.map_map]
  unfold grpVal
  exact congrArg₂ (· + ·) (congrArg List.sum (List.map_congr_left fun v _ => keyA v))
    (congrArg List.sum (List.map_congr_left fun v _ => keyB v))

theorem list_sum_getD {α β : Type} [Inhabited α] [AddCommMonoid β] (f : α → β) : ∀ l : List α,
    (l.map f).sum = ∑ i ∈ Finset.range l.length, f (l.getD i default)
  | [] => by simp
  | a :: l => by
    rw [List.map_cons, List.sum_cons, list_sum_getD f l, List.length_cons, Finset.sum_range_succ']
    simp [add_comm]

/-- **A copy's `refVal` is its `placeSum`**: Definition `d`'s copy at offset `o`, at `(A, B)` under `π`, its inputs the
columns `inp` read through `π`. -/
theorem refVal_place (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (π : Array ℕ) (s : ℕ)
    (hk : π.getD t.constPos 0 = pt.qc)
    (hfit : ∀ d, ∀ r ∈ (t.defs.getD d default).rows.toList, r.1 < (t.defs.getD d default).cols) :
    ∀ fuel d o A B (inp : List ℕ), TreeAt t lays π fuel d o A B →
      refVal t lays pt s fuel d A B (inp.map fun x => π.getD x 0) =
        placeSum (termA pt π s) (termB pt π s) t.defs t.constPos fuel d o inp
  | 0, _, _, _, _, _, _ => by simp [refVal, placeSum]
  | fuel + 1, d, o, A, B, inp, ⟨hcopy, hcalls⟩ => by
    have hR := readAt_ref t lays pt π inp hk hcopy
    cases hd : t.defs[d]? with
    | none =>
      have h0 : t.defs.getD d default = default := by simp [Array.getD_eq_getD_getElem?, hd]
      simp only [refVal, placeSum, hd]
      rw [h0]
      simp [show (default : Def).rows = #[] from rfl, show (default : Def).calls = #[] from rfl]
    | some df =>
      have hdf : t.defs.getD d default = df := by simp [Array.getD_eq_getD_getElem?, hd]
      simp only [refVal, placeSum, hd]
      subst hdf
      refine congrArg₂ (· + ·) (congrArg List.sum (List.map_congr_left fun r hr => ?_)) ?_
      · rw [← hcopy r.1 (hfit d r hr)]
        exact row_place pt π s (o + r.1) _ _ r hR
      rw [list_sum_getD, Array.length_toList]
      refine Finset.sum_congr rfl fun i hi => ?_
      have hi := Finset.mem_range.mp hi
      have hc : (t.defs.getD d default).calls.toList.getD i default = (t.defs.getD d default).calls.getD i default := by
        simp [List.getD_eq_getElem?_getD, Array.getD_eq_getD_getElem?]
      rw [hc, list_range_sum]
      refine Finset.sum_congr rfl fun j hj => ?_
      have hb : ((t.defs.getD d default).calls.getD i default).bound (readAt t lays pt d A B (inp.map fun x => π.getD x 0)) j =
          (((t.defs.getD d default).calls.getD i default).bound ((t.defs.getD d default).ref o inp t.constPos) j).map
            fun x => π.getD x 0 := by
        rw [bound_map ((t.defs.getD d default).calls.getD i default) (readAt t lays pt d A B (inp.map fun x => π.getD x 0)),
          bound_map ((t.defs.getD d default).calls.getD i default) ((t.defs.getD d default).ref o inp t.constPos), List.map_map]
        exact List.map_congr_left fun v _ => hR v
      rw [hb]
      exact refVal_place t lays pt π s hk hfit fuel _ _ _ _ _ (hcalls i hi j (Finset.mem_range.mp hj))

/-- **The template fold is its reference**: entry by entry, while every Definition's layout is in order (`DefPos`), the
lane bits fit the point, and the root reads no inputs and has a constant. -/
theorem fold_eq_foldRef {t : Tmpl} {lays : Array Lay} {alpha : F128} {lw rho r P P' : Array F128} {w : ℕ}
    (hP : fold t lays alpha lw rho r = .ok (P, w)) (hP' : foldRef t lays alpha lw rho r = .ok P')
    (hlay : LayOk t lays) (hins : (t.defs.getD t.root default).ins = 0) (hcols : 0 < (t.defs.getD t.root default).cols)
    (hfit : ∀ d, ∀ r ∈ (t.defs.getD d default).rows.toList, r.1 < (t.defs.getD d default).cols)
    (hf1 : lanes t lays (t.defs.size + 1) t.root ≤ 6 + rho.size) (hf2 : lanes t lays (t.defs.size + 1) t.root ≤ 6 + r.size)
    (hpos : ∀ d < t.defs.size, DefPos t lays d) (hroot : t.root < t.defs.size) {s : ℕ} (hs : s < 64) :
    of P[s]! = of P'[s]! := by
  have htree := posMap_tree t lays hpos hroot
  clear hpos hroot
  rw [fold_refVal hP hlay hins hf1 hf2 hs, foldRef_tree hP' hs]
  have hk : (posMap t lays).getD t.constPos 0 = posOf t lays ((t.defs.getD t.root default).cols - 1) := by
    have h1 := htree
    rw [TreeAt] at h1
    have h := h1.1 _ (Nat.sub_lt hcols Nat.one_pos)
    simpa [Tmpl.constPos, posOf] using h
  refine congrArg₂ (· + ·) ((refVal_place t lays { alpha, lw, rho, r, qc := posOf t lays ((t.defs.getD t.root default).cols - 1) }
    (posMap t lays) s hk hfit _ _ 0 0 1 [] htree).trans rfl) ?_
  unfold rowShare termA termB
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
  rw [hk]
  split_ifs
  · simp only [FoldPt.e, FoldPt.g, of_mul, of_add, ← one_eq]
    ring
  · rw [add_zero]

end FlockSoundness.TmplFold
