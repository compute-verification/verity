import Proofs.Flock.Claims.ZK.FinishSwSv
import Proofs.Flock.Level3.ArithFacts
import Proofs.Flock.Claims.ZK.PiopSrcVals

/-!
# The switched source's masked values

`FinishSrc`'s `repCut_fw` asks of its source the masked values `vals + h` below `zp 64` (`hmk`), and level 0's opening's
from `zp 64` to `h_ab` (`hlig`). At the switched prover's
source (`swSrcE`), where table `i`'s rows hold at its base, rep `r`'s masked value `n` is the worker's clear value there
plus the base's pad, each as the firewall's words (`GF128.val`).

* **`Masked.swSrcE_masked_of_rowsAt`**: that reading, from the switched cut's masked values being the worker's view's
  (`swCutE_masked_of_rowsAt`), whose masked values are `v + h` (`Table.view`, at `Table.onVals`'s values).
* `Masked.swSrcE_masked`, **`Masked.swSrcE_masked_pad`**: the pad there is the tape's, and below `h_ab` it is the
  session's pad array's entry (`padsOf`), as `srcE_masked` and `srcE_masked_pad` have the honest source's: the switched
  source's masked values are `vals + pads.h` at the worker's values and the session's pads.
* `Masked.valsFw`, **`Masked.swSrcE_hmk`**, **`Masked.swSrcE_l0`**: at the firewall's clear values by pad index, the
  passed values below `zp 64` and level 0's opening from there on: below `zp 64`, `repCut_fw`'s `hmk` with the passed
  values as `vals` and the session's pads as `pads`; from `zp 64` to `h_ab`, the opening's values plus those pads.
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

/-- **The switched source's masked value at a sent index** `k < n_E`, where table `j`'s rows hold at its base, is the
worker's clear value there plus the tape's pad `k` of table `j`'s rep `r`, as `srcE_masked` has the honest source's. -/
theorem swSrcE_masked (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2)
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1)
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
  rw [dite_eq_left (show k < ((tabC PE x j).T (czExe PE x c j)).tab.nE from hk), swCutE_masked_of_rowsAt x c w v t j hrows]
  show FlockLevel3.GF128.val (v j r (wtOf PE x w j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x t).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x t).1 j) 0).2.1.2.1 ⟨k, hk⟩ +
    (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inl ⟨k, hk⟩))) = _
  rw [FlockLevel3.val_add]
  congr 1
  exact (congrArg FlockLevel3.GF128.val (tH ⟨k, by omega⟩)).symm

/-- At table `j`, `n_E` is the layout's `h_ab`. -/
theorem nE_eq_hab (x : Msg) (j : Fin (jOf x)) : (tabC PE x j).z.nE = (layE x j).hab := by
  show (layE x j).len - 4 = (layE x j).hab
  simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
  omega

/-- **The switched source's masked value below `h_ab`**, where table `j`'s rows hold at its base, is the worker's clear
value plus the session's pad (`padsOf`), as `srcE_masked_pad` has the honest source's. Below `h_ab` is below `n_E`
(`nE_eq_hab`). -/
theorem swSrcE_masked_pad (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2)
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1)
    {k : ℕ} (hkE : k < (tabC PE x j).z.nE) :
    ((swSrcE x c w v τ j.val).rep r.val).masked k =
      FlockLevel3.GF128.val (v j r (wtOf PE x w j)
          (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
          (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1 ⟨k, hkE⟩) +
        (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[k]! := by
  have hk : k < (layE x j).hab := by rw [← nE_eq_hab x j]; exact hkE
  have hlen : (layE x j).len = (layE x j).tw + 1 := rfl
  have htw : (layE x j).tw = (layE x j).hab + 3 := rfl
  rw [swSrcE_masked x c w v τ j r hrows hkE,
    FlockComplete.padsOf_get (UE.tapeZk τ) j.val r.val (layE x j) k (by omega) (by omega) (by omega)]

/-- **The firewall's clear values by pad index**: table `j`'s rep `r` clear value `k` is, below `zp 64`, word `k` of
`A j r`, as the firewall reads a rep's passed values (`Finish.RepIn.vals`, `zp 64` words), and from `zp 64` on `B j r k`,
level 0's opening, which the firewall computes itself; at every witness, mask slot and extra lanes. -/
noncomputable def valsFw (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin)
    (A : Fin (jOf x) → Fin 2 → Array _root_.Flock.F128) (B : Fin (jOf x) → Fin 2 → ℕ → _root_.Flock.F128) :
    ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals :=
  fun j r _ _ _ k => FlockLevel3.GF128.of (if k.val < (layE x j).zp 64 then (A j r)[k.val]! else B j r k.val)

/-- `zp 64` is below `h_ab`. -/
theorem zp64_lt_hab (x : Msg) (j : Fin (jOf x)) : (layE x j).zp 64 < (layE x j).hab := by
  simp only [_root_.Flock.Zk.PadLayout.hab]
  omega

/-- **`repCut_fw`'s `hmk` at the switched source**: at the firewall's clear values (`valsFw`), where table `j`'s rows
hold at its base, rep `r`'s masked values below `zp 64` are `A j r` plus the session's pads (`padsOf`). -/
theorem swSrcE_hmk (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (A : Fin (jOf x) → Fin 2 → Array _root_.Flock.F128) (B : Fin (jOf x) → Fin 2 → ℕ → _root_.Flock.F128)
    (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2)
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, valsFw x c A B j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1) :
    ∀ k < (layE x j).zp 64, ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val).masked k =
      (A j r)[k]! + (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[k]! := by
  intro k hk
  have hkE : k < (tabC PE x j).z.nE := by rw [nE_eq_hab x j]; exact lt_trans hk (zp64_lt_hab x j)
  rw [swSrcE_masked_pad x c w (valsFw x c A B) τ j r hrows hkE]
  show (if k < (layE x j).zp 64 then (A j r)[k]! else B j r k) + _ = _
  rw [ite_eq_left hk]

/-- **Level 0's opening at the switched source**: at the firewall's clear values (`valsFw`), where table `j`'s rows hold
at its base, rep `r`'s masked values from `zp 64` to `h_ab` are `B j r` plus the session's pads (`padsOf`). -/
theorem swSrcE_l0 (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (A : Fin (jOf x) → Fin 2 → Array _root_.Flock.F128) (B : Fin (jOf x) → Fin 2 → ℕ → _root_.Flock.F128)
    (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2)
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, valsFw x c A B j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1) :
    ∀ k, (layE x j).zp 64 ≤ k → k < (layE x j).hab → ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val).masked k =
      B j r k + (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[k]! := by
  intro k hk0 hk
  have hkE : k < (tabC PE x j).z.nE := by rw [nE_eq_hab x j]; exact hk
  rw [swSrcE_masked_pad x c w (valsFw x c A B) τ j r hrows hkE]
  show (if k < (layE x j).zp 64 then (A j r)[k]! else B j r k) + _ = _
  rw [ite_eq_right (Nat.not_lt.mpr hk0)]

end Masked
end Claims.ZKMal
