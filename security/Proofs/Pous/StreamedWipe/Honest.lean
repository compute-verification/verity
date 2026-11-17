module

public import Mathlib.Logic.Equiv.Fin.Basic
public import Definitions.Pous.Model.StreamedWipe
import Mathlib.Basic.ENNReal.Real

/-! The honest server, which keeps every label, is accepted on every run, so the streamed-label wipe is not vacuous. -/

@[expose] public section

namespace PousStreamedWipe

open Pous Pous.StreamedWipe

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
