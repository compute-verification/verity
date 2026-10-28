import Proofs.Flock.Soundness.TmplSplitEq

/-!
# A templated unit slot's fold is the flat slot's

`Tmpl.foldUnit` is `foldSplit` behind the checks a verifier can make in `O(1)`: the point is the slot's, and the root is
one chunk of 64 to `2^sl` columns. On a well-formed template at `layoutM`'s layouts, whose `unrollAt` net has `sl` bits,
it is the fold the flat statement computes on that net, `project (baseOf α (lw ⊗ eq(ρ, ·)) A B none sl) (eqTable r)`
(`foldUnit_eq_project`). The split fold reaches the reference without `fold` (`foldSplit_eq_foldRef`), so no lane count
against the point is needed: the walk refuses a run of lanes wider than its chunk.
-/

open Flock Flock.Tmpl FlockLevel3 GF128

namespace FlockSoundness.TmplFold

/-- **The split fold is the reference**, entry by entry, without `fold`. -/
theorem foldSplit_eq_foldRef {t : Tmpl} {lays : Array Lay} {alpha : F128} {lw rho r P P' : Array F128}
    {w1 w2 w3 : ℕ} (hS : foldSplit t lays alpha lw rho r = .ok (P, w1, w2, w3))
    (hP' : foldRef t lays alpha lw rho r = .ok P') (hlay : LayOk t lays)
    (hcall : ∀ d i, i < (t.defs.getD d default).calls.size → ((t.defs.getD d default).calls.getD i default).def_ < d)
    (hins : (t.defs.getD t.root default).ins = 0) (hcols : 0 < (t.defs.getD t.root default).cols)
    (hfit : ∀ d, ∀ r ∈ (t.defs.getD d default).rows.toList, r.1 < (t.defs.getD d default).cols)
    (hpos : ∀ d < t.defs.size, DefPos t lays d) (hroot : t.root < t.defs.size)
    (hal : 6 ≤ log2ceil (lays.getD t.root default).size)
    (hr1 : log2ceil (lays.getD t.root default).size - 6 ≤ rho.size)
    (hr2 : log2ceil (lays.getD t.root default).size - 6 ≤ r.size) {s : ℕ} (hs : s < 64) :
    of P[s]! = of P'[s]! := by
  have htree := posMap_tree t lays hpos hroot
  rw [foldSplit_refVal hS hlay hcall hroot hins hal hr1 hr2 hs, foldRef_tree hP' hs]
  clear hpos hroot
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
  · simp

/-- **A templated unit slot's fold is the flat slot's**: `foldUnit` on a well-formed template at `layoutM`'s layouts is
the flat statement's fold of the slot on `unrollAt`'s net of `sl` bits, entry by entry. -/
theorem foldUnit_eq_project {t : Tmpl} {mode : Mode} {lays : Array Lay} {sl : ℕ} {alpha : F128}
    {lw rho r P : Array F128} {n' : Net} (hwf : t.wellFormed = true) (hL : layoutM t mode = .ok lays)
    (hU : Flock.Tmpl.unrollAt t lays = .ok n') (hsl : n'.unitLog = sl)
    (hF : foldUnit t lays sl alpha lw rho r = .ok P) {s : ℕ} (hs : s < 64) :
    of P[s]! = of (Stmt.project (Stmt.baseOf alpha (Stmt.tensor lw rho) n'.a n'.b none sl) (eqTable r))[s]! := by
  unfold foldUnit at hF
  simp only [bind, Except.bind, pure, Except.pure] at hF
  split at hF
  · simp at hF
  rename_i hpt
  split at hF
  · simp at hF
  rename_i hroot'
  split at hF
  · simp at hF
  rename_i v hS
  simp only [Except.ok.injEq] at hF
  subst hF
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq, not_or, not_not, not_lt] at hpt hroot'
  obtain ⟨⟨⟨hlw, hrho⟩, hr⟩, hsl32⟩ := hpt
  obtain ⟨hal, hla⟩ := hroot'
  subst hsl
  have hrows : ∃ acc, t.rows = .ok acc := by
    unfold Flock.Tmpl.unrollAt at hU
    cases h : t.rows with
    | error e => rw [h] at hU; simp [bind, Except.bind] at hU
    | ok acc => exact ⟨acc, rfl⟩
  obtain ⟨acc, hacc⟩ := hrows
  have hR : foldRef t lays alpha lw rho r =
      .ok (FlockSoundness.TmplFold.foldRefRows { alpha, lw, rho, r, qc := 0 } (posMap t lays) acc.ra acc.rb) := by
    rw [foldRef_eq_rows, hacc]
    rfl
  have hcall : ∀ d i, i < (t.defs.getD d default).calls.size →
      ((t.defs.getD d default).calls.getD i default).def_ < d := fun d i hi => by
    by_cases hd : d < t.defs.size
    · exact (wf_call hwf d hd i hi).1
    · rw [getD_out hd] at hi
      exact absurd hi (Nat.not_lt_zero _)
  have hlay : LayOk t lays :=
    ⟨fun d i j hi => wf_bound hwf d i j hi, layouts_lanes (layoutM_layouts hL) fun d hd i hi => (wf_call hwf d hd i hi).2⟩
  obtain ⟨P, w1, w2, w3⟩ := v
  rw [foldSplit_eq_foldRef hS hR hlay hcall (wf_tmpl hwf).2.1 (wf_tmpl hwf).2.2.1 (wf_fit hwf)
      (layouts_defPos (layoutM_layouts hL) (wf_call hwf) (wf_small hwf)) (wf_tmpl hwf).1 hal (by omega) (by omega) hs,
    foldRef_eq_project hR hU hlw (by omega) (by omega) hsl32 hs]

/-- **A templated unit slot's fold is the reference**: `foldUnit` on a well-formed template at `layoutM`'s layouts is,
entry by entry, `foldRef`, the sum over the unrolled rows at their committed positions. -/
theorem foldUnit_eq_foldRef {t : Tmpl} {mode : Mode} {lays : Array Lay} {sl : ℕ} {alpha : F128}
    {lw rho r P P' : Array F128} (hwf : t.wellFormed = true) (hL : layoutM t mode = .ok lays)
    (hF : foldUnit t lays sl alpha lw rho r = .ok P) (hR : foldRef t lays alpha lw rho r = .ok P') {s : ℕ}
    (hs : s < 64) : of P[s]! = of P'[s]! := by
  unfold foldUnit at hF
  simp only [bind, Except.bind, pure, Except.pure] at hF
  split at hF
  · simp at hF
  rename_i hpt
  split at hF
  · simp at hF
  rename_i hroot'
  split at hF
  · simp at hF
  rename_i v hS
  simp only [Except.ok.injEq] at hF
  subst hF
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq, not_or, not_not, not_lt] at hpt hroot'
  obtain ⟨⟨⟨_hlw, _hrho⟩, _hr⟩, _hsl32⟩ := hpt
  obtain ⟨hal, _hla⟩ := hroot'
  have hcall : ∀ d i, i < (t.defs.getD d default).calls.size →
      ((t.defs.getD d default).calls.getD i default).def_ < d := fun d i hi => by
    by_cases hd : d < t.defs.size
    · exact (wf_call hwf d hd i hi).1
    · rw [getD_out hd] at hi
      exact absurd hi (Nat.not_lt_zero _)
  have hlay : LayOk t lays :=
    ⟨fun d i j hi => wf_bound hwf d i j hi, layouts_lanes (layoutM_layouts hL) fun d hd i hi => (wf_call hwf d hd i hi).2⟩
  obtain ⟨P, w1, w2, w3⟩ := v
  exact foldSplit_eq_foldRef hS hR hlay hcall (wf_tmpl hwf).2.1 (wf_tmpl hwf).2.2.1 (wf_fit hwf)
    (layouts_defPos (layoutM_layouts hL) (wf_call hwf) (wf_small hwf)) (wf_tmpl hwf).1 hal (by omega) (by omega) hs

end FlockSoundness.TmplFold
