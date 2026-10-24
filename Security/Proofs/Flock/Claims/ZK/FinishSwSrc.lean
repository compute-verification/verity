import Proofs.Flock.Claims.ZK.Split
import Proofs.Flock.Claims.ZK.Blind

/-!
# The switched prover's rep source

`FinishSrc` holds the firewall's rep to the rep from a source (`FlockComplete.repCut`) at premises on that source. For
the honest prover the source is `srcE`, what the session reads from its cut (`tabSrcCut` of `cutE`). For `FinishIs` it
is the switched prover's at the workers' values: the same reading of the switched cut (`RefSession.swCutOf`).

* `swCutE`, `swCutOf_eq`: the switched prover's cut at the opened coins, on `w`'s table witnesses, the workers' values
  `v` and the tape `τ`, as `cutE` has the honest prover's.
* `swSrcE`: what the session reads from it (`tabSrcCut`), as `srcE` reads `cutE`.
* `swCutE_masked`: the switched cut's masked values are `Table.swPre`'s at the table's base.
* `Table.swPre_masked`: where the worker's rows hold (`RowsAt`), they are the worker's own, its view's at its clear
  values (`Table.onVals`).
* **`swCutE_masked_of_rowsAt`**: so where table `j`'s rows hold at its base, the switched cut's masked values are the
  worker's view's.
-/

/-- **Where the worker's rows hold, the switched pre-final view's masked values are its own.** -/
theorem FlockSoundness.ZK.Table.swPre_masked {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (w : W) (v : T.Vals) (ω : T.Rand)
    (hr : T.onVals.RowsAt (w, v) ω.1 ω.2.1.2.1) : (T.swPre w v ω).masked = (T.onVals.view (w, v) ω).masked := by
  rw [FlockSoundness.ZK.Table.swPre, if_pos hr]
  rfl

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

/-- The switched cut's masked values are `Table.swPre`'s at the base. -/
theorem swCutAt_masked {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (v : ∀ j, ((C j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) (lv : ZkView.Leaves (fun j => (C j).z) → ByteArray) (j : Fin J) :
    (swCutAt C cz wt v ω lv j).masked =
      (((C j).T (cz j)).tab.swPre (wt j) (v j) (((C j).T (cz j)).randOf (ω j) 0)).masked := rfl

namespace Masked

/-- **The switched prover's cut at the opened coins `c`**, on `w`'s table witnesses, the workers' values `v` and the
tape `τ` (`RefSession.swCutOf`), as `cutE` has the honest prover's. -/
noncomputable def swCutE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) :
    ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Cut :=
  swCutAt (tabC PE x) (czExe PE x c) (wtOf PE x w) v (UE.tape x τ).1
    (swExeOf (tabC PE x) (czExe PE x c) (wtOf PE x w) v (UE.tape x τ).1 (UE.tape x τ).2.1)

theorem swCutOf_eq (nr : Msg → ℕ → ℕ) (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) :
    (session nr (tableData PE QE UE)).swCutOf x c (wtOf PE x w) v τ = swCutE x c w v τ := rfl

/-- **What the deployed session reads from the switched cut** (`tabSrcCut`), as `srcE` reads `cutE`. -/
noncomputable def swSrcE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) : ℕ → FlockComplete.TabSrc :=
  tabSrcCut PE x (czExe PE x c) (swCutE x c w v τ)

/-- The switched cut's masked values are `Table.swPre`'s at the table's base. -/
theorem swCutE_masked (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) :
    (swCutE x c w v τ j).masked =
      (((tabC PE x j).T (czExe PE x c j)).tab.swPre (wtOf PE x w j) (v j)
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0)).masked := by
  unfold swCutE
  exact swCutAt_masked _ _ _ _ _ _ j

/-- **Where table `j`'s rows hold at its base, the switched cut's masked values are the worker's view's.** -/
theorem swCutE_masked_of_rowsAt (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x))
    (hr : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1) :
    (swCutE x c w v τ j).masked =
      (((tabC PE x j).T (czExe PE x c j)).tab.onVals.view (wtOf PE x w j, v j)
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0)).masked := by
  rw [swCutE_masked]
  exact Table.swPre_masked _ _ _ _ hr

end Masked
end Claims.ZKMal
