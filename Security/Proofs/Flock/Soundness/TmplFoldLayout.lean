import Proofs.Flock.Soundness.TmplFoldPos
import Proofs.Flock.Soundness.TmplLayOk

/-!
# The template fold is its reference on `layoutM`'s layouts (PRF-133, part 3, stage 2)

`fold_eq_foldRef` with its layout hypotheses discharged by `layoutM` (`layouts_defPos`, `layouts_lanes`). What remains is
what `parse` checks of the template, each call's binding of its callee's inputs among them, and the column and call counts
below `2^64` that `locate`'s halving assumes.
-/

open Flock Flock.Tmpl FlockLevel3 GF128

namespace FlockSoundness.TmplFold

theorem fold_eq_foldRef_layout {t : Tmpl} {mode : Mode} {lays : Array Lay} {alpha : F128} {lw rho r P P' : Array F128}
    {w : ℕ} (hL : layoutM t mode = .ok lays)
    (hP : fold t lays alpha lw rho r = .ok (P, w)) (hP' : foldRef t lays alpha lw rho r = .ok P')
    (hbound : ∀ d i j, i < (t.defs.getD d default).calls.size →
      (((t.defs.getD d default).calls.getD i default).bound id j).length =
        (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).ins)
    (hcall : ∀ d < t.defs.size, ∀ i < (t.defs.getD d default).calls.size,
      ((t.defs.getD d default).calls.getD i default).def_ < d ∧
        ((t.defs.getD d default).calls.getD i default).base + ((t.defs.getD d default).calls.getD i default).count *
          (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols ≤ (t.defs.getD d default).cols)
    (hsmall : ∀ d < t.defs.size, (t.defs.getD d default).cols < 2 ^ 64 ∧ (t.defs.getD d default).calls.size < 2 ^ 64)
    (hroot : t.root < t.defs.size) (hins : (t.defs.getD t.root default).ins = 0)
    (hcols : 0 < (t.defs.getD t.root default).cols)
    (hfit : ∀ d, ∀ r ∈ (t.defs.getD d default).rows.toList, r.1 < (t.defs.getD d default).cols)
    (hf1 : lanes t lays (t.defs.size + 1) t.root ≤ 6 + rho.size) (hf2 : lanes t lays (t.defs.size + 1) t.root ≤ 6 + r.size)
    {s : ℕ} (hs : s < 64) : of P[s]! = of P'[s]! :=
  fold_eq_foldRef hP hP' ⟨hbound, layouts_lanes (layoutM_layouts hL) fun d hd i hi => (hcall d hd i hi).2⟩ hins hcols
    hfit hf1 hf2 (layouts_defPos (layoutM_layouts hL) hcall hsmall) hroot hs

end FlockSoundness.TmplFold
