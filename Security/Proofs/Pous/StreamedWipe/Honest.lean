import Mathlib.Logic.Equiv.Fin.Basic
import Definitions.Pous.Model.StreamedWipe

/-!
# The streamed-label wipe: the honest server is accepted

The server that keeps every label, `M = N · b` bits and no egress, answers every challenge after any history, so
`Accepts` holds on every run (`honest_accepts`): the guarantee's acceptance is not vacuous.
-/

namespace PousStreamedWipe

open Pous Pous.StreamedWipe

/-- The honest server: its memory is the labels, bit `t` of label `i` at `finProdFinEquiv (i, t)`. It keeps nothing of
the prior content. -/
def honest (N k b q : ℕ) : Prover N k b q (N * b) 0 where
  fill _ L := (fun j => L (finProdFinEquiv.symm j).1 (finProdFinEquiv.symm j).2, fun _ => false)
  answer σ _ _ i := fun t => σ (finProdFinEquiv (i, t))
  keep _ _ _ := fun _ => false

theorem honest_accepts {N k b q : ℕ} (z : Bits q) (L : Fin N → Bits b) (c : Fin k → Fin N) :
    Accepts (honest N k b q) z L c := by
  intro j
  funext t
  show L (finProdFinEquiv.symm (finProdFinEquiv (c j, t))).1 (finProdFinEquiv.symm (finProdFinEquiv (c j, t))).2 =
    L (c j) t
  rw [Equiv.symm_apply_apply]

end PousStreamedWipe
