import Proofs.Flock.Claims.ZK.PiopSrcVals
import Proofs.Flock.Claims.Complete.Replay.RingSwitchAlgebra

/-!
# The deployed source's lincheck values at the opened coins

`PiopSrcE`'s lincheck fields: at the deployed source (`srcE`), table `j`'s rep `r` masks at the lincheck's layout
indices the plain prover's messages and `z_partial` at `lcCoins` after the zerocheck's `3 + (m − 6)` coins, plus the
session's pads.

* `Replay.Lc.combT_lcU`, `zrT_lcV`: the prover's halved circuit and witness columns after `j` rounds are the model's
  `lcU`, `lcV` folded at the first `j` round coins (`foldl_halve`, `FoldRealizes.fold`, `zr0_hcol`), so its round-`j`
  message is the model's (`msg_lcMsg`).
* `Masked.rhoF_out`, `tsF_lc`, `lcH_exe`: the draw's residual and round coins are the zerocheck's output `ρ` and the
  `lcCoins` round coins, so the cut's honest lincheck message is `lcMsg` at the executable's coins.
* `Masked.src_lc0`, `src_lc1`, `src_zp`: `PiopSrc`'s lincheck fields at the deployed source.
-/

namespace FlockComplete.Replay.Lc

open Flock FlockLevel3 FlockSoundness FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128

/-- **The prover's halved circuit after `j` rounds is the model's `lcU`** folded at the first `j` round coins. -/
theorem combT_lcU (S : Model.Statement) (c : CircuitFold) (hc : FoldRealizes c S) (zc : ZcOut)
    (hrho : zc.rho.size = S.m - 6) (cs : Challenges) (full : Array F128) (hf : full.size = 2 ^ S.kLog)
    (hfold : c.fold (cs.ca 0) (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) = .ok full) (j : ℕ) {k : ℕ}
    (hk : k < 2 ^ S.kLog / 2 ^ j) :
    GF128.of (combT c full cs j)[k]! = Model.foldAll ((pushes cs.τ j).toList.map fOf) (2 ^ S.kLog)
      (Discharge.Composed.Lincheck.lcU S zc.z (cs.ca 0) (cs.cb 0) zc.rho) k := by
  have hk6 := S.hk6
  have hkm := S.hk
  have hK : c.kLog = S.kLog := hc.kLog
  have hρsz : (zc.rho.extract 0 (c.kLog - 6)).size = S.kLog - 6 := by
    rw [Array.size_extract, hrho, hK]; omega
  have hρv : vecOf (S.kLog - 6) (zc.rho.extract 0 (c.kLog - 6)) = fun j : Fin (S.kLog - 6) => fOf zc.rho[j.val]! := by
    funext j
    simp only [vecOf]
    rw [getElem!_pos _ _ (by rw [hρsz]; exact j.isLt), getElem!_pos _ _ (by rw [hrho]; omega),
      Array.getElem_extract]
    simp
  show fOf (combT c full cs j)[k]! = _
  rw [← foldl_pushes (c := c) (full := full) (cs := cs) j, ← Array.foldl_toList,
    foldl_halve _ _ k (by rw [Array.size_modify, hf, Array.length_toList, pushes_size]; exact hk),
    Array.size_modify, hf]
  refine foldAll_congr _ _ _ _ (fun col hcol => ?_) k
    (by rw [List.length_map, Array.length_toList, pushes_size]; exact hk)
  show GF128.of (full.modify c.pin (· + cs.cb 0))[col]! = _
  rw [modify_get! full c.pin col _ (by rw [hf]; exact hcol), hc.pin]
  have hsum : GF128.of full[col]! = ∑ i : Fin (2 ^ S.kLog),
      execArith.lagS (fOf zc.z) ⟨i.val % 64, Nat.mod_lt _ (by norm_num)⟩ *
        Model.eqAt (fun j : Fin (S.kLog - 6) => fOf zc.rho[j.val]!) (i.val / 64) *
          (fOf (cs.ca 0) * ((S.A₀ i ⟨col, hcol⟩).val : 𝔽) + ((S.B₀ i ⟨col, hcol⟩).val : 𝔽)) := by
    have hfr := hc.fold (cs.ca 0) (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) full (skipWeights_size _)
      hρsz hfold ⟨col, hcol⟩
    rw [hρv] at hfr
    exact hfr
  unfold Discharge.Composed.Lincheck.lcU Model.comb
  rw [dite_eq_left hcol]
  by_cases hp : S.pin.val = col
  · rw [ite_eq_left hp, ite_eq_left hp.symm]
    show GF128.of (full[col]! + cs.cb 0) = _
    rw [of_add, hsum]
  · rw [ite_eq_right hp, ite_eq_right (Ne.symm hp), add_zero, hsum]

/-- **The prover's halved witness columns after `j` rounds are the model's `lcV`** folded at the first `j` round
coins. -/
theorem zrT_lcV (S : Model.Statement) (c : CircuitFold) (hK : c.kLog = S.kLog) (tr : Trace) (zc : ZcOut)
    (hrho : zc.rho.size = S.m - 6) (w : Model.Witness S.m)
    (hz : ∀ a : Fin (2 ^ S.m), Zc.bitF tr.z a.val = ((w a).val : 𝔽)) (cs : Challenges) (j : ℕ) {k : ℕ}
    (hk : k < 2 ^ S.kLog / 2 ^ j) :
    GF128.of (zrT c S.m tr zc cs j)[k]! = Model.foldAll ((pushes cs.τ j).toList.map fOf) (2 ^ S.kLog)
      (Discharge.Composed.Lincheck.lcV S zc.rho w) k := by
  have hzs : (zr0 c S.m tr zc).size = 2 ^ S.kLog := by simp [zr0, hK]
  show fOf (zrT c S.m tr zc cs j)[k]! = _
  rw [← zrT_foldl, ← Array.foldl_toList,
    foldl_halve _ _ k (by rw [hzs, Array.length_toList, pushes_size]; exact hk), hzs]
  exact foldAll_congr _ _ _ (Discharge.Composed.Lincheck.lcV S zc.rho w)
    (fun i hi => zr0_hcol S c hK tr zc hrho w hz (fun j => fOf zc.rho[S.kLog - 6 + j.val]!) (fun _ => rfl) i hi) k
    (by rw [List.length_map, Array.length_toList, pushes_size]; exact hk)

/-- **The prover's round-`j` lincheck message is the model's** (`lcMsg`) on `lcU`, `lcV` at its round coins. -/
theorem msg_lcMsg (S : Model.Statement) (c : CircuitFold) (hc : FoldRealizes c S) (tr : Trace) (zc : ZcOut)
    (hrho : zc.rho.size = S.m - 6) (w : Model.Witness S.m)
    (hz : ∀ a : Fin (2 ^ S.m), Zc.bitF tr.z a.val = ((w a).val : 𝔽)) (cs : Challenges) (full : Array F128)
    (hf : full.size = 2 ^ S.kLog)
    (hfold : c.fold (cs.ca 0) (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) = .ok full) {j : ℕ}
    (hj : j < S.kLog - 6) :
    (GF128.of (msg c S.m tr zc full cs j).1, GF128.of (msg c S.m tr zc full cs j).2) =
      Discharge.Composed.Lincheck.lcMsg (2 ^ S.kLog)
        (Discharge.Composed.Lincheck.lcU S zc.z (cs.ca 0) (cs.cb 0) zc.rho)
        (Discharge.Composed.Lincheck.lcV S zc.rho w) ((pushes cs.τ (S.kLog - 6)).toList.map fOf) j := by
  have hK : c.kLog = S.kLog := hc.kLog
  have hf' : full.size = 2 ^ c.kLog := by rw [hK]; exact hf
  have h2 : 2 ^ S.kLog / 2 ^ j = 2 * (2 ^ S.kLog / 2 ^ (j + 1)) := half_pow (by omega)
  have hH : (combT c full cs j).size / 2 = 2 ^ S.kLog / 2 ^ (j + 1) := by
    rw [combT_size hf' j, hK, h2]
    exact Nat.mul_div_cancel_left _ (by norm_num)
  have hA : ∀ k, k < 2 ^ S.kLog / 2 ^ j → GF128.of (combT c full cs j)[k]! =
      Model.foldAll ((pushes cs.τ j).toList.map fOf) (2 ^ S.kLog)
        (Discharge.Composed.Lincheck.lcU S zc.z (cs.ca 0) (cs.cb 0) zc.rho) k :=
    fun k hk => combT_lcU S c hc zc hrho cs full hf hfold j hk
  have hV : ∀ k, k < 2 ^ S.kLog / 2 ^ j → GF128.of (zrT c S.m tr zc cs j)[k]! =
      Model.foldAll ((pushes cs.τ j).toList.map fOf) (2 ^ S.kLog) (Discharge.Composed.Lincheck.lcV S zc.rho w) k :=
    fun k hk => zrT_lcV S c hK tr zc hrho w hz cs j hk
  have htake : ((pushes cs.τ (S.kLog - 6)).toList.map fOf).take j = (pushes cs.τ j).toList.map fOf := by
    refine List.ext_getElem ?_ fun i h₁ h₂ => ?_
    · simp only [List.length_take, List.length_map, Array.length_toList, pushes_size]
      omega
    · simp [pushes_eq_map]
  unfold Discharge.Composed.Lincheck.lcMsg Discharge.Composed.Lincheck.roundMsg
  rw [htake]
  simp only [msg]
  rw [hH]
  refine Prod.ext ?_ ?_
  · dsimp only
    rw [Zc.of_sumF]
    refine Finset.sum_congr rfl fun k hk => ?_
    have hk' := Finset.mem_range.1 hk
    rw [of_mul, hA (k + 2 ^ S.kLog / 2 ^ (j + 1)) (by rw [h2]; omega),
      hV (k + 2 ^ S.kLog / 2 ^ (j + 1)) (by rw [h2]; omega)]
  · dsimp only
    rw [Zc.of_sumF]
    refine Finset.sum_congr rfl fun k hk => ?_
    have hk' := Finset.mem_range.1 hk
    rw [of_mul, of_add, of_add, hA k (by rw [h2]; omega), hA (k + 2 ^ S.kLog / 2 ^ (j + 1)) (by rw [h2]; omega),
      hV k (by rw [h2]; omega), hV (k + 2 ^ S.kLog / 2 ^ (j + 1)) (by rw [h2]; omega)]
    ring

end FlockComplete.Replay.Lc

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

local notation "𝔽" => FlockLevel3.GF128

/-- Table `j`'s rep `r` lincheck coins at the opened coins, after the zerocheck's `3 + (m − 6)`. -/
noncomputable abbrev lcE {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) (r : Fin 2) :
    FlockComplete.Replay.Lc.Challenges :=
  FlockComplete.lcCoins (fun k i => slotAt c k i)
    (repBase x (2 * j.val + r.val) + 3 + ((zkTab x j).table.S.m - 6)) ((zkTab x j).table.S.kLog - 6)

/-- An array of `n` entries, each `f k`, is `pushes f n`. -/
theorem arr_eq_pushes {α : Type} [Inhabited α] {a : Array α} {f : ℕ → α} {n : ℕ} (hs : a.size = n)
    (h : ∀ k < n, a[k]! = f k) : a = FlockComplete.Replay.pushes f n := by
  refine Array.ext (hs.trans (FlockComplete.Replay.pushes_size f n).symm) fun k h1 h2 => ?_
  rw [← getElem!_pos a k h1, ← getElem!_pos (FlockComplete.Replay.pushes f n) k h2, h k (lt_of_lt_of_eq h1 hs),
    FlockComplete.Replay.pushes_get f n k (lt_of_lt_of_eq h1 hs)]

/-- **The draw's residual coins are the zerocheck's output `ρ`** at the opened coins. -/
theorem rhoF_out {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) (r : Fin 2)
    (tr : FlockComplete.Trace) :
    Composed.rhoF (zkTab x j).table (stAt x j) (czExe PE x c j) r =
      (FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) tr).rho := by
  show _ = FlockComplete.Replay.pushes (zcE c j r).ρ ((zkTab x j).table.S.m - 6)
  exact arr_eq_pushes (by unfold Composed.rhoF; exact Array.size_ofFn) (rhoF_czExe c j r)

/-- **The draw's round coins are the lincheck's** (`lcCoins`) at the opened coins. -/
theorem tsF_lc {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) (r : Fin 2) :
    Composed.tsF (zkTab x j).table (stAt x j) (czExe PE x c j) r =
      FlockComplete.Replay.pushes (lcE c j r).τ ((zkTab x j).table.S.kLog - 6) := by
  refine arr_eq_pushes (by unfold Composed.tsF; exact Array.size_ofFn) fun k hk => ?_
  unfold Composed.tsF
  rw [getElem!_pos _ k (by rw [Array.size_ofFn]; exact hk), Array.getElem_ofFn]
  show slotAt c (repBase x (2 * j.val + r.val) + (5 + ((stAt x j).m - 6) + k)) 0 =
    slotAt c (repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6) + 2 + k) 0
  exact congrArg (fun q => slotAt c q 0) (by omega)

/-- **The cut's honest lincheck message is `lcMsg` at the executable's coins**: the zerocheck's output `ζ` and `ρ`,
and `lcCoins`' `α`, `β` and round coins. -/
theorem lcH_exe {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) (tr : FlockComplete.Trace) (i : ℕ) :
    Composed.lcH (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwE x w τ j) i =
      Composed.Lincheck.lcMsg (2 ^ (zkTab x j).table.S.kLog)
        (Composed.Lincheck.lcU (zkTab x j).table.S
          (FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) tr).z ((lcE c j r).ca 0) ((lcE c j r).cb 0)
          (FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) tr).rho)
        (Composed.Lincheck.lcV (zkTab x j).table.S
          (FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) tr).rho (zwE x w τ j))
        ((FlockComplete.Replay.pushes (lcE c j r).τ ((zkTab x j).table.S.kLog - 6)).toList.map
          FlockSoundness.Refine.fOf) i := by
  have hz : Composed.zcF (zkTab x j).table (stAt x j) (czExe PE x c j) r =
      (FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) tr).z := rfl
  have hα : (czExe PE x c j).rest.lcAlpha r = (lcE c j r).ca 0 :=
    show slotAt c (repBase x (2 * j.val + r.val) + (3 + ((stAt x j).m - 6))) 0 =
      slotAt c (repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6)) 0 from
    congrArg (fun q => slotAt c q 0) (Nat.add_assoc _ _ _).symm
  have hβ : (czExe PE x c j).rest.lcBeta r = (lcE c j r).cb 0 :=
    show slotAt c (repBase x (2 * j.val + r.val) + (4 + ((stAt x j).m - 6))) 0 =
      slotAt c (repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6) + 1) 0 from
    congrArg (fun q => slotAt c q 0) (by omega)
  unfold Composed.lcH
  rw [hz, hα, hβ, rhoF_out c j r tr, tsF_lc c j r]

/-- Every lincheck message index is below `h_ab`. -/
theorem lc_lt_hab (L : _root_.Flock.Zk.PadLayout) {i t : ℕ} (hi : i < L.nLc) (ht : t < 2) : L.lc i t < L.hab := by
  simp only [_root_.Flock.Zk.PadLayout.hab, _root_.Flock.Zk.PadLayout.zp, _root_.Flock.Zk.PadLayout.lc]
  omega

/-- Every `z_partial` index is below `h_ab`. -/
theorem zp_lt_hab (L : _root_.Flock.Zk.PadLayout) {i : ℕ} (hi : i < 64) : L.zp i < L.hab := by
  simp only [_root_.Flock.Zk.PadLayout.hab, _root_.Flock.Zk.PadLayout.zp]
  omega

/-- The session's trace's `z` bits are `w`'s share with the tape's mask words (`trsE_bits`). -/
theorem bits_z {x : Msg} {w : Claims.Flock.Wit} (hw : RZk x w) (τ : UE.Tape) (j : Fin (jOf x)) :
    ∀ a : Fin (2 ^ (zkTab x j).table.S.m),
      FlockComplete.Replay.Zc.bitF ((trsE x w (UE.tapeZk τ))[j.val]!).z a.val = ((zwE x w τ j a).val : 𝔽) :=
  fun a => by
    rw [(trsE_bits hw τ j).2.2]
    unfold Composed.cT
    exact dite_eq_left a.isLt

/-! ## `PiopSrc`'s lincheck fields at the deployed source -/

section Lincheck

variable {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) {w : Claims.Flock.Wit} (hw : RZk x w) (τ : UE.Tape)
  (j : Fin (jOf x)) (r : Fin 2)

include hw

/-- **The cut's honest lincheck message is the prover's** at the opened coins, wherever the circuit fold returns
`full`. -/
theorem lcH_msg {full : Array _root_.Flock.F128}
    (hfold : (stAt x j).circuitFold.fold ((lcE c j r).ca 0)
      (_root_.Flock.skipWeights
        (FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) ((trsE x w (UE.tapeZk τ))[j.val]!)).z)
      ((FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) ((trsE x w (UE.tapeZk τ))[j.val]!)).rho.extract 0
        ((stAt x j).circuitFold.kLog - 6)) = .ok full)
    {i : ℕ} (hi : i < (stAt x j).c.kLog - 6) :
    Composed.lcH (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwE x w τ j) i =
      (FlockLevel3.GF128.of (FlockComplete.Replay.Lc.msg (stAt x j).circuitFold (zkTab x j).table.S.m
          ((trsE x w (UE.tapeZk τ))[j.val]!)
          (FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) ((trsE x w (UE.tapeZk τ))[j.val]!)) full
          (lcE c j r) i).1,
        FlockLevel3.GF128.of (FlockComplete.Replay.Lc.msg (stAt x j).circuitFold (zkTab x j).table.S.m
          ((trsE x w (UE.tapeZk τ))[j.val]!)
          (FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) ((trsE x w (UE.tapeZk τ))[j.val]!)) full
          (lcE c j r) i).2) := by
  have hm := FlockComplete.Replay.Lc.msg_lcMsg (zkTab x j).table.S (stAt x j).circuitFold
    (FlockSoundness.Refine.stmtOf_fold (stAt x j) (zkTab x j).wf.1 (FlockSoundness.Refine.regionsOf (stAt x j)))
    ((trsE x w (UE.tapeZk τ))[j.val]!)
    (FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) ((trsE x w (UE.tapeZk τ))[j.val]!))
    (FlockComplete.Replay.pushes_size _ _) (zwE x w τ j) (bits_z hw τ j) (lcE c j r) full
    (FlockLevel3.fold_size (stAt x j) _ _ _ full hfold) hfold hi
  exact (lcH_exe c w τ j r ((trsE x w (UE.tapeZk τ))[j.val]!) i).trans hm.symm

theorem src_lc0 {full : Array _root_.Flock.F128}
    (hfold : (stAt x j).circuitFold.fold ((lcE c j r).ca 0)
      (_root_.Flock.skipWeights
        (FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) ((trsE x w (UE.tapeZk τ))[j.val]!)).z)
      ((FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) ((trsE x w (UE.tapeZk τ))[j.val]!)).rho.extract 0
        ((stAt x j).circuitFold.kLog - 6)) = .ok full)
    {i : ℕ} (hi : i < (stAt x j).c.kLog - 6) :
    ((srcE x c w τ j.val).rep r.val).masked ((layE x j).lc i 0) =
      (FlockComplete.Replay.Lc.msg (stAt x j).circuitFold (zkTab x j).table.S.m ((trsE x w (UE.tapeZk τ))[j.val]!)
        (FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) ((trsE x w (UE.tapeZk τ))[j.val]!)) full
        (lcE c j r) i).1 +
      (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[(layE x j).lc i 0]! := by
  refine (srcE_masked_pad x c w τ j r (lc_lt_hab (layE x j) hi (by decide))).trans (congrArg (· + _) ?_)
  have e := Composed.honestAt_lc (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwE x w τ j)
    (((UE.tape x τ).1 j).2.1.2.1 r) i 0 hi (by decide)
  rw [ite_eq_left rfl, lcH_msg c hw τ j r hfold hi] at e
  exact congrArg FlockLevel3.GF128.val e

theorem src_lc1 {full : Array _root_.Flock.F128}
    (hfold : (stAt x j).circuitFold.fold ((lcE c j r).ca 0)
      (_root_.Flock.skipWeights
        (FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) ((trsE x w (UE.tapeZk τ))[j.val]!)).z)
      ((FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) ((trsE x w (UE.tapeZk τ))[j.val]!)).rho.extract 0
        ((stAt x j).circuitFold.kLog - 6)) = .ok full)
    {i : ℕ} (hi : i < (stAt x j).c.kLog - 6) :
    ((srcE x c w τ j.val).rep r.val).masked ((layE x j).lc i 1) =
      (FlockComplete.Replay.Lc.msg (stAt x j).circuitFold (zkTab x j).table.S.m ((trsE x w (UE.tapeZk τ))[j.val]!)
        (FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) ((trsE x w (UE.tapeZk τ))[j.val]!)) full
        (lcE c j r) i).2 +
      (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[(layE x j).lc i 1]! := by
  refine (srcE_masked_pad x c w τ j r (lc_lt_hab (layE x j) hi (by decide))).trans (congrArg (· + _) ?_)
  have e := Composed.honestAt_lc (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwE x w τ j)
    (((UE.tape x τ).1 j).2.1.2.1 r) i 1 hi (by decide)
  rw [ite_eq_right (show ¬ (1 : ℕ) = 0 by decide), lcH_msg c hw τ j r hfold hi] at e
  exact congrArg FlockLevel3.GF128.val e

theorem src_zp {i : ℕ} (hi : i < 64) :
    ((srcE x c w τ j.val).rep r.val).masked ((layE x j).zp i) =
      (FlockComplete.Replay.Lc.zrT (stAt x j).circuitFold (zkTab x j).table.S.m ((trsE x w (UE.tapeZk τ))[j.val]!)
        (FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) ((trsE x w (UE.tapeZk τ))[j.val]!))
        (lcE c j r) ((zkTab x j).table.S.kLog - 6))[i]! +
      (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[(layE x j).zp i]! := by
  refine (srcE_masked_pad x c w τ j r (zp_lt_hab (layE x j) hi)).trans (congrArg (· + _) ?_)
  have e := Composed.honestAt_zp (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwE x w τ j)
    (((UE.tape x τ).1 j).2.1.2.1 r) i hi
  have hz : Composed.zpH (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwE x w τ j) i =
      FlockSoundness.Model.foldAll
        ((FlockComplete.Replay.pushes (lcE c j r).τ ((zkTab x j).table.S.kLog - 6)).toList.map
          FlockSoundness.Refine.fOf) (2 ^ (zkTab x j).table.S.kLog)
        (Composed.Lincheck.lcV (zkTab x j).table.S
          (FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) ((trsE x w (UE.tapeZk τ))[j.val]!)).rho
          (zwE x w τ j)) i := by
    unfold Composed.zpH
    rw [rhoF_out c j r ((trsE x w (UE.tapeZk τ))[j.val]!), tsF_lc c j r]
  have h6 := (zkTab x j).table.S.hk6
  have h64 : 2 ^ (zkTab x j).table.S.kLog / 2 ^ ((zkTab x j).table.S.kLog - 6) = 64 := by
    rw [Nat.pow_div (by omega) (by norm_num),
      show (zkTab x j).table.S.kLog - ((zkTab x j).table.S.kLog - 6) = 6 by omega]
  have hv := FlockComplete.Replay.Lc.zrT_lcV (zkTab x j).table.S (stAt x j).circuitFold rfl
    ((trsE x w (UE.tapeZk τ))[j.val]!)
    (FlockComplete.Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) ((trsE x w (UE.tapeZk τ))[j.val]!))
    (FlockComplete.Replay.pushes_size _ _) (zwE x w τ j) (bits_z hw τ j) (lcE c j r)
    ((zkTab x j).table.S.kLog - 6) (k := i) (by rw [h64]; exact hi)
  exact congrArg FlockLevel3.GF128.val ((e.trans hz).trans hv.symm)

end Lincheck

end Masked

end Claims.ZKMal
