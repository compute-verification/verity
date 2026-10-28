import Proofs.Flock.Claims.ZK.FinishSwMk

/-!
# The switched source's masked values, with or without the rows

The switched prover's pre-final view (`Table.swPre`) is the worker's view where its rows hold, and elsewhere the honest
prover's on the pads moved by Lemma A's amounts (`Table.shift`, `Table.padShift`), which moves each exposed pad by the
difference between the worker's clear value and the honest one. Either way its masked values are the worker's clear
values plus the drawn pads, so the readings `FinishSwMk` takes where the rows hold hold everywhere:

* **`Table.swPre_masked_any`**: the switched pre-final view's masked values are the worker's view's, at any values.
* `Masked.swCutE_masked_any`, `Masked.swSrcE_masked_any`, **`Masked.swSrcE_masked_pad_any`**: so the switched source's
  masked value at a sent index is the worker's clear value plus the tape's pad, and below `h_ab` plus the session's pad
  array's entry (`padsOf`), with no hypothesis on the rows.
-/

/-- **The switched pre-final view's masked values are the worker's view's**, whether or not its rows hold: where they
fail, the honest values on the moved pads add up to the worker's on the drawn ones. -/
theorem FlockSoundness.ZK.Table.swPre_masked_any {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (w : W) (v : T.Vals) (ω : T.Rand) :
    (T.swPre w v ω).masked = (T.onVals.view (w, v) ω).masked := by
  by_cases hr : T.onVals.RowsAt (w, v) ω.1 ω.2.1.2.1
  · exact T.swPre_masked w v ω hr
  · rw [FlockSoundness.ZK.Table.swPre, if_neg hr]
    funext r i
    show T.vals r w ω.1 ω.2.1.2.1 i + ((ω.2.2.1 r).1 i + (v r w ω.1 ω.2.1.2.1 i - T.vals r w ω.1 ω.2.1.2.1 i)) =
      v r w ω.1 ω.2.1.2.1 i + (ω.2.2.1 r).1 i
    ring

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- The switched cut's masked values are the worker's view's, at any values. -/
theorem swCutE_masked_any (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) :
    (swCutE x c w v τ j).masked =
      (((tabC PE x j).T (czExe PE x c j)).tab.onVals.view (wtOf PE x w j, v j)
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0)).masked := by
  rw [swCutE_masked]
  exact Table.swPre_masked_any _ _ _ _

/-- **The switched source's masked value at a sent index** `k < n_E` is the worker's clear value there plus the tape's
pad `k` of table `j`'s rep `r`, with no hypothesis on the rows (`swSrcE_masked` at `swCutE_masked_any`). -/
theorem swSrcE_masked_any (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2)
    {k : ℕ} (hk : k < (tabC PE x j).z.nE) :
    ((swSrcE x c w v τ j.val).rep r.val).masked k =
      FlockLevel3.GF128.val (v j r (wtOf PE x w j)
          (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
          (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1 ⟨k, hk⟩) +
        (UE.tapeZk τ).h j.val r.val k := by
  obtain ⟨t, rfl⟩ : ∃ t' : Fin Claims.Flock.zkTableCap → ExeCell, t' = τ := ⟨τ, rfl⟩
  have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
  have hz := fits_zkTab PE (zkTab x j)
  have hE64 : (tabC PE x j).z.nE ≤ 2 ^ 64 := hz.nE
  have tH : ∀ q : Fin (2 ^ 64 + 4), FlockLevel3.GF128.of ((UE.tapeZk t).h j.val r.val q.val) =
      (t ⟨j.val, hc⟩).1.2.2.1 r q := fun q => by
    show FlockLevel3.GF128.of (fAt 0 (fun r => fAt 0 ((cellAt t j.val).1.2.2.1 r) q.val) r.val) = _
    rw [cellAt_lt t hc, fAt_lt _ _ r.isLt]
    exact fAt_lt _ _ q.isLt
  unfold swSrcE tabSrcCut
  rw [dite_eq_left j.isLt]
  show (if hr : r.val < 2 then repSrcCut PE x (czExe PE x c) (swCutE x c w v t) j ⟨r.val, hr⟩ else default).masked k =
    _
  rw [dite_eq_left r.isLt]
  show (if h : k < ((tabC PE x j).T (czExe PE x c j)).tab.nE then
      FlockLevel3.GF128.val ((swCutE x c w v t j).masked r ⟨k, h⟩) else 0) = _
  rw [dite_eq_left (show k < ((tabC PE x j).T (czExe PE x c j)).tab.nE from hk), swCutE_masked_any x c w v t j]
  show FlockLevel3.GF128.val (v j r (wtOf PE x w j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x t).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x t).1 j) 0).2.1.2.1 ⟨k, hk⟩ +
    (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inl ⟨k, hk⟩))) = _
  rw [FlockLevel3.val_add]
  congr 1
  exact (congrArg FlockLevel3.GF128.val (tH ⟨k, by omega⟩)).symm

/-- **The switched source's masked value below `h_ab`** is the worker's clear value plus the session's pad
(`padsOf`), with no hypothesis on the rows (`swSrcE_masked_pad` at `swSrcE_masked_any`). -/
theorem swSrcE_masked_pad_any (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2)
    {k : ℕ} (hkE : k < (tabC PE x j).z.nE) :
    ((swSrcE x c w v τ j.val).rep r.val).masked k =
      FlockLevel3.GF128.val (v j r (wtOf PE x w j)
          (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
          (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1 ⟨k, hkE⟩) +
        (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[k]! := by
  have hk : k < (layE x j).hab := by rw [← nE_eq_hab x j]; exact hkE
  have hlen : (layE x j).len = (layE x j).tw + 1 := rfl
  have htw : (layE x j).tw = (layE x j).hab + 3 := rfl
  rw [swSrcE_masked_any x c w v τ j r hkE,
    FlockComplete.padsOf_get (UE.tapeZk τ) j.val r.val (layE x j) k (by omega) (by omega) (by omega)]

end Masked
end Claims.ZKMal
