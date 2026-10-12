import Proofs.Flock.VBridge.OfVStar

/-!
# A part of S, renumbered

A part of `GfResiduals`' S keeps some of its residuals (`R`, in order) and the operands they multiply (`used`), with `w`'s,
`v`'s and the operands' indices renumbered by `fw`, `fv` and `fo` (`renS`). Whatever the maps, the part's residual `k` is
S's residual `R[k]` (`resVal_renS`, with `res_renS`), when:

* every operand a kept residual multiplies is kept where `fo` puts it (`used[fo k] = k`);
* every term a kept residual reads, added or in an operand it multiplies, reads under the maps what it read
  (`TermAgrees`): the part's element `fw i` of `w` is S's element `i`, its `fv j` of `v` is S's `j`, and a `wt` term's
  block of 128 elements at `fw i` is S's at `i`, its bit below 128.

The maps are the part's to choose. `VStar.parts` builds its own with HashMaps, but nothing here reads them.
-/

namespace FlockVBridge.OfVStar

open Flock

def renTerm (fw fv : ℕ → ℕ) : VStar.Term → VStar.Term
  | .w i => .w (fw i)
  | .v j => .v (fv j)
  | .wt i u => .wt (fw i) u

def renRes (fw fv fo : ℕ → ℕ) (r : VStar.Res) : VStar.Res :=
  ⟨r.mul.map fun ab => (fo ab.1, fo ab.2), r.add.map (renTerm fw fv)⟩

/-- **A part of S**: its ports of `nw` and `nv` elements, S's residuals `R` and the operands `used`, renumbered. -/
def renS (S : VStar.GfS) (nw nv : ℕ) (R used : List ℕ) (fw fv fo : ℕ → ℕ) : VStar.GfS :=
  ⟨nw, nv, used.map fun k => (S.ops.getD k []).map (renTerm fw fv), R.map fun r => renRes fw fv fo (S.res.getD r default)⟩

/-- A term reads under the maps what it read. -/
def TermAgrees (w v w' v' : ℕ → F128) (fw fv : ℕ → ℕ) : VStar.Term → Prop
  | .w i => w' (fw i) = w i
  | .v j => v' (fv j) = v j
  | .wt i u => u < 128 ∧ ∀ j < 128, w' (fw i + j) = w (i + j)

/-- What `resVal_renS` needs of a part's maps at S's ports `w`, `v` and the part's `w'`, `v'`. -/
structure PartAgrees (S : VStar.GfS) (R used : List ℕ) (fw fv fo : ℕ → ℕ) (w v w' v' : ℕ → F128) : Prop where
  ops : ∀ r ∈ R, ∀ ab ∈ (S.res.getD r default).mul, ∀ k ∈ [ab.1, ab.2], ∃ h : fo k < used.length, used[fo k] = k
  add : ∀ r ∈ R, ∀ t ∈ (S.res.getD r default).add, TermAgrees w v w' v' fw fv t
  opTerms : ∀ r ∈ R, ∀ ab ∈ (S.res.getD r default).mul, ∀ k ∈ [ab.1, ab.2], ∀ t ∈ S.ops.getD k [],
    TermAgrees w v w' v' fw fv t

theorem foldl_add_congr {β : Type} {f g : β → F128} :
    ∀ {l : List β} (a : F128), (∀ x ∈ l, f x = g x) → (l.map f).foldl (· + ·) a = (l.map g).foldl (· + ·) a
  | [], _, _ => rfl
  | x :: l, a, h => by
    simp only [List.map_cons, List.foldl_cons]
    rw [h x (by simp)]
    exact foldl_add_congr _ fun y hy => h y (by simp [hy])

theorem ofBits_congr {f g : ℕ → Bool} (h : ∀ j < 128, f j = g j) : ofBits f = ofBits g := by
  unfold ofBits
  congr 1
  exact List.map_congr_left fun j hj => by rw [h j (List.mem_range.1 hj)]

theorem termVal_renTerm {w v w' v' : ℕ → F128} {fw fv : ℕ → ℕ} {t : VStar.Term} (h : TermAgrees w v w' v' fw fv t) :
    termVal w' v' (term (renTerm fw fv t)) = termVal w v (term t) := by
  cases t with
  | w i => exact h
  | v j => exact h
  | wt i u =>
    obtain ⟨hu, hb⟩ := h
    simp only [renTerm, term, termVal]
    refine ofBits_congr fun j hj => ?_
    rw [show 128 * (fw i + j) + u = 128 * (fw i + j) + u from rfl, wbit_elem _ _ hu, wbit_elem _ _ hu, hb j hj]

theorem opVal_renS {S : VStar.GfS} {nw nv : ℕ} {R used : List ℕ} {fw fv fo : ℕ → ℕ} {w v w' v' : ℕ → F128} {k : ℕ}
    (hk : ∃ h : fo k < used.length, used[fo k] = k) (ht : ∀ t ∈ S.ops.getD k [], TermAgrees w v w' v' fw fv t) :
    opVal (gfS (renS S nw nv R used fw fv fo)) (termVal w' v') (fo k) = opVal (gfS S) (termVal w v) k := by
  obtain ⟨hlt, hk⟩ := hk
  unfold opVal gfS renS
  simp only [List.map_map]
  rw [List.getD_eq_getElem _ _ (by simpa using hlt), List.getElem_map, hk]
  have hg : (S.ops.map (List.map term)).getD k [] = (S.ops.getD k []).map term := by
    by_cases hko : k < S.ops.length
    · rw [List.getD_eq_getElem _ _ (by simpa using hko), List.getElem_map, List.getD_eq_getElem _ _ hko]
    · rw [List.getD_eq_default _ _ (by simpa using hko), List.getD_eq_default _ _ (by simpa using hko)]
      rfl
  rw [hg]
  simp only [Function.comp_def, List.map_map]
  exact foldl_add_congr _ fun t htm => termVal_renTerm (ht t htm)

theorem res_renS (S : VStar.GfS) (nw nv : ℕ) (R used : List ℕ) (fw fv fo : ℕ → ℕ) :
    (renS S nw nv R used fw fv fo).res = R.map fun r => renRes fw fv fo (S.res.getD r default) := rfl

theorem addVal_renRes {w v w' v' : ℕ → F128} {fw fv fo : ℕ → ℕ} {r : VStar.Res}
    (h : ∀ t ∈ r.add, TermAgrees w v w' v' fw fv t) :
    ((res (renRes fw fv fo r)).add.map (termVal w' v')).foldl (· + ·) 0 = ((res r).add.map (termVal w v)).foldl (· + ·) 0 := by
  simp only [res, renRes, Function.comp_def, List.map_map]
  exact foldl_add_congr _ fun t ht => termVal_renTerm (h t ht)

theorem mulVals_renRes {S : VStar.GfS} {nw nv : ℕ} {R used : List ℕ} {fw fv fo : ℕ → ℕ} {w v w' v' : ℕ → F128}
    {r : VStar.Res} (hops : ∀ ab ∈ r.mul, ∀ k ∈ [ab.1, ab.2], ∃ h : fo k < used.length, used[fo k] = k)
    (hterms : ∀ ab ∈ r.mul, ∀ k ∈ [ab.1, ab.2], ∀ t ∈ S.ops.getD k [], TermAgrees w v w' v' fw fv t) :
    (res (renRes fw fv fo r)).mul.map
        (fun ab => opVal (gfS (renS S nw nv R used fw fv fo)) (termVal w' v') ab.1 *
          opVal (gfS (renS S nw nv R used fw fv fo)) (termVal w' v') ab.2) =
      (res r).mul.map (fun ab => opVal (gfS S) (termVal w v) ab.1 * opVal (gfS S) (termVal w v) ab.2) := by
  simp only [res, renRes, List.map_map, Function.comp_def]
  refine List.map_congr_left fun ab hab => ?_
  have h1 := opVal_renS (nw := nw) (nv := nv) (R := R) (hops ab hab ab.1 (by simp)) (hterms ab hab ab.1 (by simp))
  have h2 := opVal_renS (nw := nw) (nv := nv) (R := R) (hops ab hab ab.2 (by simp)) (hterms ab hab ab.2 (by simp))
  rw [h1, h2]

/-- **A part's residual is S's**: a residual `r` of S that the part keeps, renumbered (`renRes`), has at the part's ports
`w'`, `v'` its value at `w`, `v`, when the maps agree (`PartAgrees`). -/
theorem resVal_renS {S : VStar.GfS} {nw nv : ℕ} {R used : List ℕ} {fw fv fo : ℕ → ℕ} {w v w' v' : ℕ → F128}
    (h : PartAgrees S R used fw fv fo w v w' v') {r : ℕ} (hR : r ∈ R) :
    resVal (gfS (renS S nw nv R used fw fv fo)) (termVal w' v') (res (renRes fw fv fo (S.res.getD r default))) =
      resVal (gfS S) (termVal w v) (res (S.res.getD r default)) := by
  unfold resVal
  rw [addVal_renRes (h.add r hR), mulVals_renRes (h.ops r hR) (h.opTerms r hR)]

end FlockVBridge.OfVStar
