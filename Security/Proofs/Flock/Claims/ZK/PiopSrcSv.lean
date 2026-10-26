import Proofs.Flock.Claims.ZK.PiopSrcLc
import Proofs.Flock.Claims.ZK.SentE

/-!
# The deployed source's switched vectors at the opened coins

`PiopSrcE`'s ring-switch field: at the deployed source (`srcE`), table `j`'s rep `r` switched vector `k` is the plain
prover's `switchVec` of claim `k` at the opened coins wherever the rep sends it before the stop.

* `Replay.Rs.switchVec_shat`: entry `v` of a claim's switched vector of a trace whose bits are `w`'s is `shat` of `w` at
  the claim (`switch_eval`'s body).
* `Masked.extras_x`: extra claim `i` is region `i / 2`'s at link point `i % 2` (`extraClaims_ok`).
* `Masked.src_sv_abc`: switched vectors `0` and `1`, whatever claims follow the zerocheck's and lincheck's.
* `Masked.src_sv`: `PiopSrc`'s ring-switch field at the deployed source. Masked claim `2r + k` (`k < 2`) is sent where
  its rank check is seen (`famPasses_of`), and its `ŝ` is `shat` at `z = (w, u)` (`sHat_shat`) at the zerocheck's and
  lincheck's coins (`rhoF_out`, `rF_czExe`, `tsF_lc`); region claim `k − 2`'s is the witness's, the region's words off
  the mask slot (`regions_off_of_words`, `zWith_off`), at the link points of round `0`'s coin.
-/

namespace FlockComplete.Replay.Rs

open Flock FlockLevel3 FlockSoundness FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128

/-- **Entry `v` of a claim's switched vector**, on a trace whose bits are the witness's, is the witness's `shat` at the
claim. -/
theorem switchVec_shat {m : ℕ} (hm : 7 ≤ m) (c : Claim) (hx : c.x.size = m - 6) (tr : Trace) (w : Model.Witness m)
    (hz : ∀ a : Fin (2 ^ m), Zc.bitF tr.z a.val = ((w a).val : 𝔽)) (v : Fin 128) :
    GF128.of (switchVec tr c)[v.val]! = shat hm (Discharge.Composed.RingSwitch.claimOf m c) w v := by
  have hsz : (c.x.extract 1 c.x.size).size = m - 7 := by
    rw [Array.size_extract, Nat.min_self, hx]
    omega
  unfold switchVec
  dsimp only
  rw [Discharge.Composed.Zerocheck.get_map_range _ v.isLt, Zc.of_sumF, eqTable_size, hsz, Finset.sum_range]
  unfold shat
  refine Finset.sum_congr rfl fun y _ => ?_
  have he : GF128.of (eqTable (c.x.extract 1 c.x.size))[y.val]! =
      Model.eqAt ((Discharge.Composed.RingSwitch.claimOf m c).tail hm) y.val := by
    rw [Zc.of_eqTable_extract c.x 1 c.x.size rfl (by omega) y.val
        (by rw [hx, show m - 6 - 1 = m - 7 by omega]; exact y.isLt),
      Discharge.Composed.RingSwitch.eqAt_range ((Discharge.Composed.RingSwitch.claimOf m c).tail hm)
        (fun i => fOf c.x[i + 1]!) fun j => rfl, hx, show m - 6 - 1 = m - 7 by omega]
    unfold Discharge.Composed.Zerocheck.eqR
    refine Finset.prod_congr rfl fun j _ => ?_
    rw [Nat.add_comm 1 j]
    rfl
  rw [← hz (addr128 hm y v), ← he]
  unfold Zc.bitF
  rw [show (addr128 hm y v).val = 128 * y.val + v.val from rfl]
  cases tr.z[128 * y.val + v.val]!
  · rw [ite_eq_right Bool.false_ne_true, ite_eq_right Bool.false_ne_true, mul_zero]
    exact GF128.zero_eq.symm
  · rw [ite_eq_left rfl, ite_eq_left rfl, mul_one]

theorem switchVec_size (tr : Trace) (c : Claim) : (switchVec tr c).size = 128 := by
  unfold switchVec
  simp

end FlockComplete.Replay.Rs

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

local notation "𝔽" => FlockLevel3.GF128

/-- **Extra claim `i` is region `i / 2`'s at link point `i % 2`**, its point's last `m − 6` coordinates
(`extraClaims_ok`'s lists). -/
theorem extras_x (st : _root_.Flock.Stmt) (p0 p1 : Array _root_.Flock.F128) (L : List _root_.Flock.Region)
    (ess : List (List _root_.Flock.Claim))
    (h : List.Forall₂ (fun reg es => List.Forall₂ (fun r e => ∃ v, st.regionValue reg r = .ok v ∧
      e = FlockSoundness.Refine.regionClaimOf st reg r v) [p0, p1] es) L ess) :
    ∀ i < 2 * L.length, (ess.flatten[i]!).x =
      (st.regionPoint L[i / 2]! (if i % 2 = 0 then p0 else p1)).extract 6
        (st.regionPoint L[i / 2]! (if i % 2 = 0 then p0 else p1)).size := by
  induction h with
  | nil => intro i hi; simp at hi
  | @cons reg es L' ess' hes _ ih =>
    intro i hi
    rcases hes with _ | ⟨⟨v0, -, rfl⟩, hrest⟩
    rcases hrest with _ | ⟨⟨v1, -, rfl⟩, hnil⟩
    rcases hnil
    rcases Nat.lt_or_ge i 2 with h2 | h2
    · interval_cases i <;> simp [FlockSoundness.Refine.regionClaimOf]
    · obtain ⟨i', rfl⟩ : ∃ i', i = i' + 2 := ⟨i - 2, by omega⟩
      rw [show (i' + 2) / 2 = i' / 2 + 1 by omega, show (i' + 2) % 2 = i' % 2 by omega]
      simp only [List.flatten_cons, List.cons_append, List.nil_append, List.getElem!_cons_succ]
      exact ih i' (by simp only [List.length_cons] at hi; omega)

/-- A point's last coordinates past the first seven. -/
theorem extract_six_one {α : Type} (a : Array α) : ((a.extract 6 a.size).extract 1).toList = a.toList.drop 7 := by
  rw [Composed.extract_one_toList, Array.toList_extract, List.extract_eq_take_drop,
    List.take_of_length_le (by rw [List.length_drop, Array.length_toList])]
  simp only [List.drop_drop]

section Switch

variable {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) {w : Claims.Flock.Wit} (hw : RZk x w) (τ : UE.Tape)
  (j : Fin (jOf x)) (r : Fin 2)

include hw

/-- **Masked claim `2r + k`'s `ŝ` is the prover's switched vector** of a claim whose point past its first coordinate is
the rank check's point `2r + k`: `ν + maskHom` is `shat` at `z = (w, u)` (`sHat_shat`, `shat_eqWord`), and so is the
switched vector of the session's trace (`switchVec_shat`, `bits_z`). -/
theorem sv_masked (hm7 : 7 ≤ (stAt x j).m) (k : ℕ) (h4 : 2 * r.val + k < _root_.Flock.Zk.RANK_POINTS)
    (cl : _root_.Flock.Claim) (hx6 : cl.x.size = (stAt x j).m - 6)
    (hpt : Composed.pointsOf ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j))
      ⟨2 * r.val + k, h4⟩ = cl.x.toList.drop 1) :
    (Array.ofFn fun v => FlockLevel3.GF128.val
      ((((tabC PE x j).T (czExe PE x c j)).tab.nu (wtOf PE x w j) +
        ((tabC PE x j).T (czExe PE x c j)).tab.maskHom ((UE.tape x τ).1 j).1) ⟨2 * r.val + k, h4⟩ v)) =
      FlockComplete.switchVec (trsE x w (UE.tapeZk τ))[j.val]! cl := by
  obtain ⟨hnd, hlt⟩ := Composed.maskWordsOf_ok (stAt x j) (zkTab x j).mask.m (zkTab x j).mask.mask
  apply Array.ext
  · rw [Array.size_ofFn, FlockComplete.Replay.Rs.switchVec_size]
  · intro v hv hv'
    have hv128 : v < 128 := by rwa [Array.size_ofFn] at hv
    rw [Array.getElem_ofFn]
    have e1 := FlockComplete.Replay.Rs.switchVec_shat hm7 cl hx6 _ (zwE x w τ j) (bits_z hw τ j) ⟨v, hv128⟩
    rw [getElem!_pos _ v hv'] at e1
    have hp : (cl.x.extract 1).toList = Composed.pointsOf ((czExe PE x c j).rest :
        Composed.RestOf (zkTab x j).table (stAt x j)) ⟨2 * r.val + k, h4⟩ := by
      rw [Composed.extract_one_toList, hpt]
    refine (e1.trans ((Composed.RingSwitch.shat_eqWord hm7 cl hx6 _ _).trans ?_)).symm
    rw [hp]
    exact (Composed.sHat_shat (stAt x j) hm7 (wtOf PE x w j) rfl ((UE.tape x τ).1 j).1 hnd hlt
      (fun y => FlockSoundness.Refine.fOf (Composed.eqWord (Composed.pointsOf ((czExe PE x c j).rest :
        Composed.RestOf (zkTab x j).table (stAt x j)) ⟨2 * r.val + k, h4⟩) y)) ⟨v, hv128⟩).symm

/-- **Region claim `i`'s `ŝ` is the prover's switched vector** of a claim whose point past its first coordinate is the
region's point at the link: off the mask slot's words `z = (w, u)` is the witness (`zWith_off`), and on them the region's
point weighs nothing (`regions_off_of_words`). -/
theorem sv_region (hm7 : 7 ≤ (stAt x j).m) (i : Fin (2 * (stAt x j).regions.size)) (cl : _root_.Flock.Claim)
    (hx6 : cl.x.size = (stAt x j).m - 6)
    (hpt : (cl.x.extract 1).toList = Composed.regionPtOf (stAt x j)
      ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).link i) :
    (Array.ofFn fun v => FlockLevel3.GF128.val
      (((tabC PE x j).T (czExe PE x c j)).tab.sReg (wtOf PE x w j) i v)) =
      FlockComplete.switchVec (trsE x w (UE.tapeZk τ))[j.val]! cl := by
  have hwo := Composed.avoidsMask_words (FailClosed.avoidsMask_of_shapeOk (zkTab x j).wf.2 (zkTab x j).hzk)
  apply Array.ext
  · rw [Array.size_ofFn, FlockComplete.Replay.Rs.switchVec_size]
  · intro v hv hv'
    have hv128 : v < 128 := by rwa [Array.size_ofFn] at hv
    rw [Array.getElem_ofFn]
    have e1 := FlockComplete.Replay.Rs.switchVec_shat hm7 cl hx6 _ (zwE x w τ j) (bits_z hw τ j) ⟨v, hv128⟩
    rw [getElem!_pos _ v hv'] at e1
    refine (e1.trans ((Composed.RingSwitch.shat_eqWord hm7 cl hx6 _ _).trans ?_)).symm
    rw [hpt]
    show _ = ∑ y : Fin (2 ^ ((stAt x j).m - 7)), FlockSoundness.Refine.fOf (Composed.eqWord
      (Composed.regionPtOf (stAt x j) ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).link i)
        y.val) * FlockSoundness.ZK.Table.castF (F := 𝔽) (Composed.wordBit (wtOf PE x w j) y.val v)
    refine Finset.sum_congr rfl fun y _ => ?_
    by_cases hy : y.val ∈ Composed.maskWordsOf (stAt x j)
    · rw [Composed.regions_off_of_words (zkTab x j).mask.m hwo _ i y.val hy, FlockSoundness.Refine.fOf_zero,
        zero_mul, zero_mul]
    · refine congrArg (FlockSoundness.Refine.fOf (Composed.eqWord (Composed.regionPtOf (stAt x j)
        ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).link i) y.val) * ·) ?_
      have hz := Composed.zWith_off (stAt x j) hm7 (wtOf PE x w j) rfl ((UE.tape x τ).1 j).1 y ⟨v, hv128⟩ hy
      rw [Composed.castF_eq]
      unfold Composed.wordBit
      rw [dite_eq_left (show 128 * y.val + v < 2 ^ (tabC PE x j).S.m from
        (FlockSoundness.addr128 hm7 y ⟨v, hv128⟩).isLt)]
      exact congrArg (fun b : ZMod 2 => ((b.val : ℕ) : 𝔽)) hz

/-- **The rep's first two switched vectors at the deployed source**: wherever the rep's ring-switch body `k < 2` is sent
before the stop, switched vector `k` is the prover's of the zerocheck's or lincheck's claim at the opened coins, whatever
claims follow them. -/
theorem src_sv_abc (hx : Live x) {n : ℕ} (hn : ¬ Refused x c n) {s spec}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (e : Array _root_.Flock.Claim) :
    ∀ k < 2, repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6) + 3 + ((stAt x j).circuitFold.kLog - 6) + k ≤ n →
      ((srcE x c w τ j.val).rep r.val).sv k =
        FlockComplete.switchVec (trsE x w (UE.tapeZk τ))[j.val]!
          (_root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog
            (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) (trsE x w (UE.tapeZk τ))[j.val]!)
            (FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m (trsE x w (UE.tapeZk τ))[j.val]!
              (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) (trsE x w (UE.tapeZk τ))[j.val]!)
              (lcE c j r)) ++ e)[k]! := by
  intro k hk2 hkn
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  have hst : (stsOf x)[j.val]! = stAt x j := getElem!_pos (stsOf x) j.val hj
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hm14 : 14 ≤ (stAt x j).m := le_trans (by norm_num) (hall j.val hj).1
  have hm7 : 7 ≤ (stAt x j).m := by omega
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hkm : (stAt x j).c.kLog ≤ (stAt x j).m := (zkTab x j).wf.1.hk
  have hfk : (stAt x j).circuitFold.kLog = (stAt x j).c.kLog := rfl
  set tr := (trsE x w (UE.tapeZk τ))[j.val]! with htr
  set zc := FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr with hzc
  set lc := FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr zc (lcE c j r) with hlc
  set A := _root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog zc lc with hA
  have hA2 : A.size = 2 := rfl
  unfold srcE tabSrcCut
  rw [dite_eq_left j.isLt]
  show (if hr : r.val < 2 then repSrcCut PE x (czExe PE x c) (cutE x c w τ) j ⟨r.val, hr⟩ else default).sv k = _
  rw [dite_eq_left r.isLt]
  dsimp only [repSrcCut]
  rw [ite_eq_left hk2]
  have h4 : 2 * r.val + k < _root_.Flock.Zk.RANK_POINTS := by
    have := r.isLt; simp only [_root_.Flock.Zk.RANK_POINTS]; omega
  rw [dite_eq_left (show 2 * r.val + k < ((tabC PE x j).T (czExe PE x c j)).tab.nC from h4)]
  have hrank : guardSlot x j.val r.val .opening ≤ n := by
    show rankSlot x j.val r.val ≤ n
    unfold rankSlot
    rw [hst]
    omega
  have hpass := famPasses_of (pointReads_czExe PE) (coinReads_czExe PE) hx hn hsx j r .opening hrank
  have hrep : ((tabC PE x j).T (czExe PE x c j)).rep ⟨2 * r.val + k, h4⟩ = r :=
    Composed.rep_of_lt r (2 * r.val + k) (by omega)
  have hs : (cutE x c w τ j).sHat ⟨2 * r.val + k, h4⟩ =
      some ((((tabC PE x j).T (czExe PE x c j)).tab.nu (wtOf PE x w j) +
        ((tabC PE x j).T (czExe PE x c j)).tab.maskHom ((UE.tape x τ).1 j).1) ⟨2 * r.val + k, h4⟩) := by
    have hpass' : famPasses (fun j => ((tabC PE x j).T (czExe PE x c j)).tab)
        (fun j => ((tabC PE x j).T (czExe PE x c j)).rep) j
        (((tabC PE x j).T (czExe PE x c j)).rep ⟨2 * r.val + k, h4⟩) .opening := by
      rw [hrep]; exact hpass
    exact ite_eq_left hpass'
  rw [hs]
  dsimp only
  rw [FlockComplete.Replay.get_left (show k < A.size by rw [hA2]; exact hk2)]
  have hρ := rhoF_out c j r tr
  have hR := rF_czExe c hsx j r
  have hT := congrArg Array.reverse (tsF_lc c j r)
  have hpt : Composed.pointsOf ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j))
      ⟨2 * r.val + k, h4⟩ = (A[k]!).x.toList.drop 1 := by
    show ((Composed.claimsOf ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j))
      (if 2 * r.val + k < 2 then 0 else 1))[(2 * r.val + k) % 2]!).x.toList.drop 1 = _
    rw [show (2 * r.val + k) % 2 = k by omega, Composed.rep_of_lt r (2 * r.val + k) (by omega)]
    unfold Composed.claimsOf
    interval_cases k
    · exact congrArg (fun a : Array _root_.Flock.F128 => a.toList.drop 1)
        (FlockComplete.Replay.abc_x (z₁ := ⟨_, _, _, _, _, _⟩) (z₂ := zc) (l₁ := ⟨_, _, _⟩) (l₂ := lc) hρ hR hT).1
    · exact congrArg (fun a : Array _root_.Flock.F128 => a.toList.drop 1)
        (FlockComplete.Replay.abc_x (z₁ := ⟨_, _, _, _, _, _⟩) (z₂ := zc) (l₁ := ⟨_, _, _⟩) (l₂ := lc) hρ hR hT).2
  have hx6 : (A[k]!).x.size = (stAt x j).m - 6 := by
    interval_cases k
    · show ((FlockComplete.Replay.pushes (lcE c j r).τ ((stAt x j).circuitFold.kLog - 6)).reverse ++
        (FlockComplete.Replay.pushes (zcE c j r).ρ ((stAt x j).m - 6)).extract ((stAt x j).c.kLog - 6)
          ((stAt x j).m - 6)).size = (stAt x j).m - 6
      rw [Array.size_append, Array.size_reverse, FlockComplete.Replay.pushes_size, Array.size_extract,
        FlockComplete.Replay.pushes_size, Nat.min_self]
      omega
    · exact FlockComplete.Replay.Zc.r_size (stAt x j).m (by omega) (zcE c j r)
  exact sv_masked c hw τ j r hm7 k h4 _ hx6 hpt

/-- **`PiopSrc`'s ring-switch field at the deployed source**: wherever the rep's ring-switch body `k` is sent before the
stop, switched vector `k` is the prover's of claim `k`, the zerocheck's and lincheck's claims at the opened coins, then
the statement's extra claims at the link points of round `0`'s coin. -/
theorem src_sv (hx : Live x) {n : ℕ} (hn : ¬ Refused x c n) {s spec}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) {extras : Array _root_.Flock.Claim}
    (he : (stAt x j).extraClaims (ptsE c spec) = .ok extras) :
    ∀ k < (_root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog
        (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) (trsE x w (UE.tapeZk τ))[j.val]!)
        (FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m (trsE x w (UE.tapeZk τ))[j.val]!
          (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) (trsE x w (UE.tapeZk τ))[j.val]!)
          (lcE c j r)) ++ extras).size,
      repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6) + 3 + ((stAt x j).circuitFold.kLog - 6) + k ≤ n →
      ((srcE x c w τ j.val).rep r.val).sv k =
        FlockComplete.switchVec (trsE x w (UE.tapeZk τ))[j.val]!
          (_root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog
            (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) (trsE x w (UE.tapeZk τ))[j.val]!)
            (FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m (trsE x w (UE.tapeZk τ))[j.val]!
              (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) (trsE x w (UE.tapeZk τ))[j.val]!)
              (lcE c j r)) ++ extras)[k]! := by
  intro k hk hkn
  rcases Nat.lt_or_ge k 2 with hk2 | hk2
  · exact src_sv_abc c hw τ j r hx hn hsx extras k hk2 hkn
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  have hst : (stsOf x)[j.val]! = stAt x j := getElem!_pos (stsOf x) j.val hj
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hm14 : 14 ≤ (stAt x j).m := le_trans (by norm_num) (hall j.val hj).1
  have hm7 : 7 ≤ (stAt x j).m := by omega
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hkm : (stAt x j).c.kLog ≤ (stAt x j).m := (zkTab x j).wf.1.hk
  have hfk : (stAt x j).circuitFold.kLog = (stAt x j).c.kLog := rfl
  set tr := (trsE x w (UE.tapeZk τ))[j.val]! with htr
  set zc := FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr with hzc
  set lc := FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr zc (lcE c j r) with hlc
  set A := _root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog zc lc with hA
  have hA2 : A.size = 2 := rfl
  unfold srcE tabSrcCut
  rw [dite_eq_left j.isLt]
  show (if hr : r.val < 2 then repSrcCut PE x (czExe PE x c) (cutE x c w τ) j ⟨r.val, hr⟩ else default).sv k = _
  rw [dite_eq_left r.isLt]
  dsimp only [repSrcCut]
  · rw [ite_eq_right (by omega : ¬ k < 2)]
    set ρ := ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) with hρdef
    obtain ⟨hpts, hmp⟩ := HiddenExec.setupTables_pointsAny (FlockComplete.ZkTables.setupTables_of hx.1 hsx)
    have hmP : (stAt x j).mPts = spec.mPts := hmp j.val hj
    have hmx : mPtsOf x = spec.mPts := by unfold mPtsOf; rw [hsx]
    have hpsz : (ptsE c spec).size = 2 := by unfold ptsE; simp [hpts]
    have hesz : extras.size = (stAt x j).regions.size * 2 := by rw [FlockDeploy.extraClaims_size he, hpsz]
    have hkr : k - 2 < 2 * (stAt x j).regions.size := by
      rw [Array.size_append, hA2, hesz] at hk; omega
    rw [dite_eq_left (show k - 2 < ((tabC PE x j).T (czExe PE x c j)).tab.nR from hkr)]
    rw [FlockComplete.Replay.get_right (show A.size ≤ k by rw [hA2]; omega), hA2]
    obtain ⟨ess, hess, heq⟩ := FlockSoundness.Refine.extraClaims_ok (stAt x j) (ptsE c spec) extras he
    have hlink : (ptsE c spec).toList = [Array.ofFn (ρ.link 0), Array.ofFn (ρ.link 1)] := by
      have hp : ∀ p : Fin 2, (Array.range spec.mPts).map (fun i => slotAt c 0 (p.val * spec.mPts + i)) =
          Array.ofFn (ρ.link p) := by
        intro p
        apply Array.ext
        · simp [hmP]
        · intro i h1 h2
          rw [Array.getElem_map, Array.getElem_range, Array.getElem_ofFn]
          show slotAt c 0 (p.val * spec.mPts + i) = slotAt c 0 (p.val * mPtsOf x + i)
          rw [hmx]
      apply List.ext_getElem
      · simp [ptsE, hpts]
      · intro i h1 h2
        have hi2 : i < 2 := by simpa [ptsE, hpts] using h1
        rw [Array.getElem_toList]
        simp only [ptsE, Array.getElem_map, Array.getElem_range]
        interval_cases i
        · exact hp 0
        · exact hp 1
    rw [hlink] at hess
    have hxe := extras_x (stAt x j) _ _ _ ess hess (k - 2) (by rw [Array.length_toList]; exact hkr)
    have hget : extras[k - 2]! = ess.flatten[k - 2]! := by
      rw [← heq, getElem!_def, getElem!_def, Array.getElem?_toList]
    rw [← hget] at hxe
    have hreg : (stAt x j).regions.toList[(k - 2) / 2]! = (stAt x j).regions[(k - 2) / 2]! := by
      rw [getElem!_def, getElem!_def, Array.getElem?_toList]
    have hif : (if (k - 2) % 2 = 0 then Array.ofFn (ρ.link 0) else Array.ofFn (ρ.link 1)) =
        Array.ofFn (ρ.link (if (k - 2) % 2 = 0 then 0 else 1)) := by
      split <;> rfl
    rw [hreg, hif] at hxe
    have hx6 : (extras[k - 2]!).x.size = (stAt x j).m - 6 := by
      rw [hxe, Array.size_extract, FlockSoundness.Refine.regionPoint_size]; omega
    have hpt : ((extras[k - 2]!).x.extract 1).toList = Composed.regionPtOf (stAt x j) ρ.link ⟨k - 2, hkr⟩ := by
      rw [hxe, extract_six_one]
      rfl
    exact sv_region c hw τ j hm7 ⟨k - 2, hkr⟩ _ hx6 hpt

end Switch

end Masked

end Claims.ZKMal
