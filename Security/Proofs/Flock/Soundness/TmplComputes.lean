import Proofs.Flock.Soundness.Types.Dag
import Flock.TmplComputes

/-!
# A template Definition whose rows are a typed layout's computes its type

`Flock.TmplValue.defComputes_of_rows` proves `Tmpl.DefComputes` from any relational fact about a Definition's rows. The
typed path has that fact for every layout `deriveChecked` accepts: `Types.Dag.layout_sound`, whose rows
(`Flock.DeriveAll.fullRow`) satisfied with the constant at 1 carry `evalT` of the layout's type on its outputs. So a
Definition whose rows, in its own references, are a checked layout's, with the layout's constant at the Definition's
constant reference, its inputs at the Definition's inputs and its outputs among the Definition's columns, computes its
type's `evalT` on those outputs (`defComputes_of_layout`). No verifier path reads this file.
-/

open Flock Flock.Tmpl Flock.DeriveAll FlockSoundness.Types.Dag

namespace FlockSoundness.TmplValue

/-- **A Definition whose rows are a checked layout's computes the layout's type.** -/
theorem defComputes_of_layout {t : Tmpl} {d : ℕ} (hd : d < t.defs.size)
    (hord : ∀ r ∈ (t.defs.getD d default).rows.toList, ∀ w ∈ r.2.1.toList ++ r.2.2.toList,
      (t.defs.getD d default).cols ≤ w ∨ w < r.1)
    (hcols : ∀ r ∈ (t.defs.getD d default).rows.toList, r.1 < (t.defs.getD d default).cols)
    {types : String → Option CircuitType} {ls : Array Layout} {info : String → Option TableInfo}
    {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}
    (hdc : deriveChecked types ls info words unit = .ok done) {j : ℕ} (hj : j ≠ unit) {ty : CircuitType}
    (ht : (ls.getD j default).type.bind types = some ty) (F : ℕ) (hF : j < F)
    (hrows : fullRow words (done.getD j default) = (t.defs.getD d default).rowsAt)
    (hconst : (done.getD j default).const = (t.defs.getD d default).cols + (t.defs.getD d default).ins)
    (hin : ∀ i < ty.inBits, (done.getD j default).inCols.getD i 0 = (t.defs.getD d default).cols + i)
    (hinB : ty.inBits ≤ (t.defs.getD d default).ins)
    (hout : ∀ o < ty.outputs.size, (done.getD j default).outCols.getD o 0 < (t.defs.getD d default).cols) :
    t.DefComputes d ((List.range ty.outputs.size).map fun o => (done.getD j default).outCols.getD o 0)
      (evalT types words F ty) := by
  have hg : ∀ o < ty.outputs.size,
      ((List.range ty.outputs.size).map fun o => (done.getD j default).outCols.getD o 0).getD o 0 =
        (done.getD j default).outCols.getD o 0 := fun o ho => by
    rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range ho]
    rfl
  refine Flock.TmplValue.defComputes_of_rows hd hord hcols (fun o ho => ?_)
    (fun x y hxy => evalT_congr types words F ty fun i hi => hxy i (by omega)) ?_
  · simp only [List.length_map, List.length_range] at ho
    rw [hg o ho]
    exact hout o ho
  · intro z hz h1 o ho
    simp only [List.length_map, List.length_range] at ho
    have hsat : SatAt (fullRow words (done.getD j default)) z := by
      rw [hrows]
      exact hz
    rw [hg o ho, layout_sound hdc hj ht z hsat (by rw [hconst]; exact h1) F hF o ho]
    exact congrArg (fun l => l.getD o false) (evalT_congr types words F ty
      (x := fun i => z ((done.getD j default).inCols.getD i 0)) (y := fun i => z ((t.defs.getD d default).cols + i))
      fun i hi => congrArg z (hin i hi))

end FlockSoundness.TmplValue
