import Proofs.Flock.Claims.ZK.FinishSwSv
import Proofs.Flock.Level3.ArithFacts

/-!
# The switched source's masked values

`FinishSrc`'s `repCut_fw` asks of its source the masked values `vals + h` below `h_ab` (`hmk`). At the switched prover's
source (`swSrcE`), where table `i`'s rows hold at its base, rep `r`'s masked value `n` is the worker's clear value there
plus the base's pad, each as the firewall's words (`GF128.val`).

* **`Masked.swSrcE_masked_of_rowsAt`**: that reading, from the switched cut's masked values being the worker's view's
  (`swCutE_masked_of_rowsAt`), whose masked values are `v + h` (`Table.view`, at `Table.onVals`'s values).
-/

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **Where table `i`'s rows hold at its base, the switched source's masked values are the worker's values plus the
base's pads**, as the firewall's words. -/
theorem swSrcE_masked_of_rowsAt (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (i : ℕ) (hi : i < jOf x) (r : ℕ)
    (hr : r < 2)
    (hrows : ((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).tab.onVals.RowsAt (wtOf PE x w ⟨i, hi⟩, v ⟨i, hi⟩)
      (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).randOf ((UE.tape x τ).1 ⟨i, hi⟩) 0).1
      (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).randOf ((UE.tape x τ).1 ⟨i, hi⟩) 0).2.1.2.1)
    (n : ℕ) (hn : n < ((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).tab.nE) :
    ((swSrcE x c w v τ i).rep r).masked n =
      FlockLevel3.GF128.val (v ⟨i, hi⟩ ⟨r, hr⟩ (wtOf PE x w ⟨i, hi⟩)
          (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).randOf ((UE.tape x τ).1 ⟨i, hi⟩) 0).1
          (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).randOf ((UE.tape x τ).1 ⟨i, hi⟩) 0).2.1.2.1 ⟨n, hn⟩) +
        FlockLevel3.GF128.val
          (((((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).randOf ((UE.tape x τ).1 ⟨i, hi⟩) 0).2.2.1
            ⟨r, hr⟩).1 ⟨n, hn⟩) := by
  rw [← FlockLevel3.val_add]
  unfold swSrcE tabSrcCut
  simp only [dif_pos hi, dif_pos hr, repSrcCut, dif_pos hn]
  exact congrArg FlockLevel3.GF128.val
    (congrFun (congrFun (swCutE_masked_of_rowsAt x c w v τ ⟨i, hi⟩ hrows) ⟨r, hr⟩) ⟨n, hn⟩)

end Masked
end Claims.ZKMal
