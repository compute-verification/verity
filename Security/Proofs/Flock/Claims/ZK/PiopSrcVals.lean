import Proofs.Flock.Claims.ZK.SimPiopOf
import Proofs.Flock.Claims.ZK.Round3Vals
import Proofs.Flock.Claims.ZK.OneTape
import Proofs.Flock.Claims.Complete.ZkTables.RunsGo
import Proofs.Flock.Claims.Complete.ZkTables.Trace
import Proofs.Flock.Claims.Complete.Zk.PadAlgebra
import Proofs.Flock.Soundness.Discharge.Composed.Inner.Holds
import Proofs.Flock.Level3.ArithFacts

/-!
# The deployed source's zerocheck values at the opened coins

`PiopSrcE`'s zerocheck fields: at the deployed source (`srcE`), table `j`'s rep `r` masks at the zerocheck's layout
indices the plain prover's messages at `zcCoins` from the rep's first round, plus the session's pads.

* `Replay.Zc.bitF_eq`, `rounds_honest`, `finalA_honest`, `finalB_honest`, `finalC_honest`: the soundness analysis's
  honest zerocheck on the trace's bits sends the executable's residual rounds and finals (`msg_eq`, `of_tab`, `c_eval`).
* `Masked.srcE_masked_pad`: the source's masked value at a sent index `k < h_ab` is the clear value the cut's view masks
  (`Composed.honestAt`) plus the session's pad `k` (`padsOf`), the tape's cell read back (`cellAt`, `fAt`).
* `Masked.trsE_bits`: the model's tables (`aT`, `bT`, `cT`) are the session's trace's bits (`trace_bits`, `zOf_ofLayout`),
  so the draw's honest zerocheck is `Zerocheck.honest` on them at the opened coins (`zcH_exe`: `rF_czExe`, the outer coins
  of the round after the rep's first, and `rhoF_czExe`).
* `Masked.src_r1ab`, `src_r1c`, `src_ml0`, `src_ml1`, `src_fa`, `src_fb`, `src_fc`: `PiopSrc`'s zerocheck fields at the
  deployed source.
-/

namespace FlockComplete.Replay.Zc

open Flock FlockLevel3 FlockSoundness.Discharge.Composed.Zerocheck

local notation "𝔽" => FlockLevel3.GF128

/-- A bit table of `2^M` entries as field values is any table agreeing with it below `2^M` and `0` past it. -/
theorem bitF_eq {bits : Array Bool} {M : ℕ} (hs : bits.size = 2 ^ M) {f : ℕ → 𝔽}
    (hlt : ∀ i, i < 2 ^ M → bitF bits i = f i) (hge : ∀ i, ¬ i < 2 ^ M → f i = 0) : bitF bits = f := by
  funext i
  by_cases hi : i < 2 ^ M
  · exact hlt i hi
  · rw [hge i hi]
    unfold bitF
    rw [getElem!_neg bits i (by rw [hs]; exact hi)]
    rfl

/-- **The honest residual round `j` is the prover's** on the trace's `A·z`, `B·z` bits (`msg_eq`). -/
theorem rounds_honest (tr : Trace) (n : ℕ) (r : Array F128) (hr : r.size = n) (z : F128) (ρ : ℕ → F128)
    (rho : Array F128) (hρ : ∀ k < n, rho[k]! = ρ k) (c : ℕ → 𝔽) {j : ℕ} (hj : j < n) :
    (honest n (bitF tr.az) (bitF tr.bz) c r rho z).rounds[j]! = msg tr n r z ρ j := by
  simp only [honest]
  rw [get_map_range _ hj]
  exact (msg_eq tr n r hr z ρ rho hρ j hj).symm

/-- **The honest `final_a` is the prover's**: its `A·z` table folded through every residual round (`of_tab`). -/
theorem finalA_honest (tr : Trace) (n : ℕ) (z : F128) (ρ : ℕ → F128) (rho : Array F128)
    (hρ : ∀ k < n, rho[k]! = ρ k) (b c : ℕ → 𝔽) (r : Array F128) :
    (honest n (bitF tr.az) b c r rho z).finalA = (tab n z ρ tr.az n)[0]! := by
  apply eq_of_of
  rw [of_tab n z ρ tr.az rho hρ n le_rfl 0 (by simp)]
  rfl

/-- **The honest `final_b` is the prover's**: its `B·z` table folded through every residual round (`of_tab`). -/
theorem finalB_honest (tr : Trace) (n : ℕ) (z : F128) (ρ : ℕ → F128) (rho : Array F128)
    (hρ : ∀ k < n, rho[k]! = ρ k) (a c : ℕ → 𝔽) (r : Array F128) :
    (honest n a (bitF tr.bz) c r rho z).finalB = (tab n z ρ tr.bz n)[0]! := by
  apply eq_of_of
  rw [of_tab n z ρ tr.bz rho hρ n le_rfl 0 (by simp)]
  rfl

/-- **The honest `final_c` is the prover's**: `round1_c` evaluated at `z` (`r1c_eq`, `c_eval`). -/
theorem finalC_honest (tr : Trace) (n : ℕ) (r : Array F128) (hrs : r.size = n) (a b : ℕ → 𝔽) (rho : Array F128)
    (z : F128) :
    (honest n a b (bitF tr.z) r rho z).finalC =
      _root_.Flock.dot (_root_.Flock.lagrange _root_.Flock.nodesL _root_.Flock.den6Inv z) (r1c tr n r) := by
  rw [r1c_eq tr n r hrs a b rho z]
  exact (c_eval n a b (bitF tr.z) r rho z).symm

end FlockComplete.Replay.Zc

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

local notation "𝔽" => FlockLevel3.GF128

/-- Table `j`'s pad layout at the executable's schedule. -/
noncomputable abbrev layE (x : Msg) (j : Fin (jOf x)) : _root_.Flock.Zk.PadLayout :=
  _root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0

/-- Table `j`'s witness as the cut's view reads it: `w`'s share with the tape's mask words (`Composed.zWith`). -/
noncomputable abbrev zwE (x : Msg) (w : Claims.Flock.Wit) (τ : UE.Tape) (j : Fin (jOf x)) :
    FlockSoundness.Model.Witness (tabC PE x j).S.m :=
  Composed.zWith (stAt x j) (wtOf PE x w j) ((UE.tape x τ).1 j).1

/-- Table `j`'s rep `r` zerocheck coins at the opened coins, from the rep's first round. -/
abbrev zcE {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) (r : Fin 2) :
    FlockComplete.Replay.Zc.Challenges :=
  FlockComplete.zcCoins (fun k i => slotAt c k i) (repBase x (2 * j.val + r.val))

/-- The session's setup of table `j` is its statement's. -/
theorem tabsOf_live (x : Msg) (j : Fin (jOf x)) :
    (FlockDeploy.tabsOf (stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (stAt x j) := by
  rw [getElem!_pos _ _ (by simpa [FlockDeploy.tabsOf] using lt_of_jOf j)]
  simp [FlockDeploy.tabsOf, stAt]

/-- Every zerocheck index is below `h_ab`. -/
theorem lt_hab (L : _root_.Flock.Zk.PadLayout) {k : ℕ} (hk : k ≤ L.fc) : k < L.hab := by
  simp only [_root_.Flock.Zk.PadLayout.hab, _root_.Flock.Zk.PadLayout.zp, _root_.Flock.Zk.PadLayout.lc]
  omega

/-- **The deployed source's masked value at a sent index** `k < n_E` is the clear value the cut's view masks
(`Composed.honestAt`) plus the tape's pad `k` of table `j`'s rep `r`. -/
theorem srcE_masked (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) {k : ℕ} (hk : k < (tabC PE x j).z.nE) :
    ((srcE x c w τ j.val).rep r.val).masked k =
      FlockLevel3.GF128.val (Composed.honestAt (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwE x w τ j)
        (((UE.tape x τ).1 j).2.1.2.1 r) k) + (UE.tapeZk τ).h j.val r.val k := by
  obtain ⟨t, rfl⟩ : ∃ t' : Fin Claims.Flock.zkTableCap → ExeCell, t' = τ := ⟨τ, rfl⟩
  have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
  have hz := fits_zkTab PE (zkTab x j)
  have hE64 : (tabC PE x j).z.nE ≤ 2 ^ 64 := hz.nE
  have tH : ∀ q : Fin (2 ^ 64 + 4), FlockLevel3.GF128.of ((UE.tapeZk t).h j.val r.val q.val) =
      (t ⟨j.val, hc⟩).1.2.2.1 r q := fun q => by
    show FlockLevel3.GF128.of (fAt 0 (fun r => fAt 0 ((cellAt t j.val).1.2.2.1 r) q.val) r.val) = _
    rw [cellAt_lt t hc, fAt_lt _ _ r.isLt]
    exact fAt_lt _ _ q.isLt
  unfold srcE tabSrcCut
  rw [dite_eq_left j.isLt]
  show (if hr : r.val < 2 then repSrcCut PE x (czExe PE x c) (cutE x c w t) j ⟨r.val, hr⟩ else default).masked k = _
  rw [dite_eq_left r.isLt]
  show (if h : k < ((tabC PE x j).T (czExe PE x c j)).tab.nE then
      FlockLevel3.GF128.val ((cutE x c w t j).masked r ⟨k, h⟩) else 0) = _
  rw [dite_eq_left (show k < ((tabC PE x j).T (czExe PE x c j)).tab.nE from hk)]
  show FlockLevel3.GF128.val (Composed.honestAt (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwE x w t j)
      (((UE.tape x t).1 j).2.1.2.1 r) k + (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inl ⟨k, hk⟩))) = _
  rw [FlockLevel3.val_add]
  congr 1
  exact (congrArg FlockLevel3.GF128.val (tH ⟨k, by omega⟩)).symm

/-- **The deployed source's masked value below `h_ab`** is the clear value plus the session's pad (`padsOf`). -/
theorem srcE_masked_pad (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) {k : ℕ} (hk : k < (layE x j).hab) :
    ((srcE x c w τ j.val).rep r.val).masked k =
      FlockLevel3.GF128.val (Composed.honestAt (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwE x w τ j)
        (((UE.tape x τ).1 j).2.1.2.1 r) k) + (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[k]! := by
  have hE : (tabC PE x j).z.nE = (layE x j).hab := by
    show (layE x j).len - 4 = (layE x j).hab
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
    omega
  have hlen : (layE x j).len = (layE x j).tw + 1 := rfl
  have htw : (layE x j).tw = (layE x j).hab + 3 := rfl
  rw [srcE_masked x c w τ j r (k := k) (by omega),
    FlockComplete.padsOf_get (UE.tapeZk τ) j.val r.val (layE x j) k (by omega) (by omega) (by omega)]

/-- **The draw's zerocheck vector is the opened coins'**: the pinned coordinates, then the outer coins of the round
after the rep's first (`outerAt_eq`). -/
theorem rF_czExe {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) {s spec}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2) :
    Composed.rF (zkTab x j).table (stAt x j) (czExe PE x c j) r = (zcE c j r).r (zkTab x j).table.S.m := by
  show _root_.Flock.pinned ++ Array.ofFn (fun i : Fin ((stAt x j).m - 13) => outerAt c j.val r.val i.val) =
    _root_.Flock.pinned ++ (Array.range ((zkTab x j).table.S.m - 13)).map
      fun i => slotAt c (repBase x (2 * j.val + r.val) + 1) i
  refine congrArg (_root_.Flock.pinned ++ ·) ?_
  ext i h1 h2
  · simp only [Array.size_ofFn, Array.size_map, Array.size_range]
    rfl
  · simp only [Array.getElem_ofFn, Array.getElem_map, Array.getElem_range]
    exact outerAt_eq c hsx j.val r.val i

/-- **The draw's residual coins are the opened coins'**, coin `0` of the rep's rounds `3 …`. -/
theorem rhoF_czExe {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) (r : Fin 2) :
    ∀ k < (zkTab x j).table.S.m - 6,
      (Composed.rhoF (zkTab x j).table (stAt x j) (czExe PE x c j) r)[k]! = (zcE c j r).ρ k := fun k hk => by
  unfold Composed.rhoF
  rw [getElem!_pos _ k (by rw [Array.size_ofFn]; exact hk), Array.getElem_ofFn]
  exact congrArg (fun q => slotAt c q 0) (Nat.add_assoc (repBase x (2 * j.val + r.val)) 3 k).symm

/-- **The model's tables are the session's trace's bits**: at a witness in the deployed relation, table `j`'s `A·z`,
`B·z` and `z` tables on `w`'s share with the tape's mask words (`zwE`) are the bits of its trace (`trace_bits`, the share
as `zOf_ofLayout` reads it), each `0` past `2^m`. -/
theorem trsE_bits {x : Msg} {w : Claims.Flock.Wit} (hw : RZk x w) (τ : UE.Tape) (j : Fin (jOf x)) :
    FlockComplete.Replay.Zc.bitF ((trsE x w (UE.tapeZk τ))[j.val]!).az = Composed.aT (zkTab x j).table (zwE x w τ j) ∧
      FlockComplete.Replay.Zc.bitF ((trsE x w (UE.tapeZk τ))[j.val]!).bz =
        Composed.bT (zkTab x j).table (zwE x w τ j) ∧
      FlockComplete.Replay.Zc.bitF ((trsE x w (UE.tapeZk τ))[j.val]!).z =
        Composed.cT (zkTab x j).table (zwE x w τ j) := by
  obtain ⟨t, rfl⟩ : ∃ t' : Fin Claims.Flock.zkTableCap → ExeCell, t' = τ := ⟨τ, rfl⟩
  have hj := lt_of_jOf j
  obtain ⟨s, spec, hset⟩ := setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨p, hp, -, ht, hpi, hpe, -, -, -⟩ := _root_.Flock.Readable.parsed_zk (live_of_lt j).1 hw hset
  have hs : p.toRead.tablesOf = some (stsOf x) := by
    rw [← hpe]
    exact _root_.Flock.Readable.tablesOf_of_parses hp
  obtain ⟨-, -, hmat⟩ := _root_.Flock.Readable.tableAt_model hs ht hpi hj
  obtain ⟨bz, ba, bb⟩ := FlockComplete.ZkTables.trace_bits (stAt x j) (zkTab x j).wf.1 hmat
    (_root_.Flock.Readable.tableWit (stsOf x).size (FlockComplete.ZkTables.maskWOf x w (UE.tapeZk t).mask) j.val)
  have htr : (trsE x w (UE.tapeZk t))[j.val]! = FlockComplete.traceOf (stAt x j)
      (_root_.Flock.Readable.tableWit (stsOf x).size (FlockComplete.ZkTables.maskWOf x w (UE.tapeZk t).mask)
        j.val).hid.toArray 0 :=
    FlockComplete.ZkTables.traces_getElem _ _ hj
  have hzw : _root_.Flock.Readable.zOf 1 (_root_.Flock.Readable.tableWit (stsOf x).size
      (FlockComplete.ZkTables.maskWOf x w (UE.tapeZk t).mask) j.val) 0 (stAt x j).m = zwE x w t j :=
    (_root_.Flock.Readable.zOf_tableWit hj _ _).trans (zOf_ofLayout (TapeLayout.exe PE) exeTape x t w j)
  have hm : (zkTab x j).table.S.m = (stAt x j).m := rfl
  have hsz : ((trsE x w (UE.tapeZk t))[j.val]!).z.size = 2 ^ (zkTab x j).table.S.m ∧
      ((trsE x w (UE.tapeZk t))[j.val]!).az.size = 2 ^ (zkTab x j).table.S.m ∧
      ((trsE x w (UE.tapeZk t))[j.val]!).bz.size = 2 ^ (zkTab x j).table.S.m := by
    rw [htr, hm]
    simp [FlockComplete.traceOf, FlockComplete.mulBlocks]
  refine ⟨FlockComplete.Replay.Zc.bitF_eq hsz.2.1 (fun i hi => ?_) (fun i hi => ?_),
    FlockComplete.Replay.Zc.bitF_eq hsz.2.2 (fun i hi => ?_) (fun i hi => ?_),
    FlockComplete.Replay.Zc.bitF_eq hsz.1 (fun i hi => ?_) (fun i hi => ?_)⟩
  · rw [htr]
    unfold Composed.aT
    rw [dite_eq_left hi]
    exact (ba ⟨i, hi⟩).trans (congrArg (fun z : FlockSoundness.Model.Witness (zkTab x j).table.S.m =>
      ((FlockSoundness.aVec (zkTab x j).table.S z ⟨i, hi⟩).val : 𝔽)) hzw)
  · unfold Composed.aT
    exact dite_eq_right hi
  · rw [htr]
    unfold Composed.bT
    rw [dite_eq_left hi]
    exact (bb ⟨i, hi⟩).trans (congrArg (fun z : FlockSoundness.Model.Witness (zkTab x j).table.S.m =>
      ((FlockSoundness.bVec (zkTab x j).table.S z ⟨i, hi⟩).val : 𝔽)) hzw)
  · unfold Composed.bT
    exact dite_eq_right hi
  · rw [htr]
    unfold Composed.cT
    rw [dite_eq_left hi]
    exact (bz ⟨i, hi⟩).trans (congrArg (fun z : FlockSoundness.Model.Witness (zkTab x j).table.S.m =>
      ((z ⟨i, hi⟩).val : 𝔽)) hzw)
  · unfold Composed.cT
    exact dite_eq_right hi

/-- **The draw's honest zerocheck is the executable's**: `Zerocheck.honest` on the session's trace's bits, at the opened
coins' zerocheck vector (`trsE_bits`, `rF_czExe`). -/
theorem zcH_exe {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) {w : Claims.Flock.Wit} (hw : RZk x w) (τ : UE.Tape)
    {s spec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2) :
    Composed.zcH (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwE x w τ j) =
      Composed.Zerocheck.honest ((zkTab x j).table.S.m - 6)
        (FlockComplete.Replay.Zc.bitF ((trsE x w (UE.tapeZk τ))[j.val]!).az)
        (FlockComplete.Replay.Zc.bitF ((trsE x w (UE.tapeZk τ))[j.val]!).bz)
        (FlockComplete.Replay.Zc.bitF ((trsE x w (UE.tapeZk τ))[j.val]!).z)
        ((zcE c j r).r (zkTab x j).table.S.m) (Composed.rhoF (zkTab x j).table (stAt x j) (czExe PE x c j) r)
        (Composed.zcF (zkTab x j).table (stAt x j) (czExe PE x c j) r) := by
  obtain ⟨ha, hb, hz⟩ := trsE_bits hw τ j
  unfold Composed.zcH
  rw [← ha, ← hb, ← hz, rF_czExe c hsx j r]

/-! ## `PiopSrc`'s zerocheck fields at the deployed source -/

section Fields

variable {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) {w : Claims.Flock.Wit} (hw : RZk x w) (τ : UE.Tape) {s spec}
  (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)

include hw hsx

theorem src_r1ab {i : ℕ} (hi : i < 64) :
    ((srcE x c w τ j.val).rep r.val).masked ((layE x j).r1ab i) =
      (FlockComplete.Replay.Zc.r1ab ((trsE x w (UE.tapeZk τ))[j.val]!) ((zkTab x j).table.S.m - 6)
        ((zcE c j r).r (zkTab x j).table.S.m))[i]! +
      (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[(layE x j).r1ab i]! := by
  refine (srcE_masked_pad x c w τ j r (lt_hab (layE x j) ?_)).trans (congrArg (· + _) ?_)
  · simp only [layE, _root_.Flock.Zk.PadLayout.new, _root_.Flock.Zk.PadLayout.r1ab, _root_.Flock.Zk.PadLayout.fc,
      _root_.Flock.Zk.PadLayout.fa]
    omega
  · have e := Composed.honestAt_r1ab (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwE x w τ j)
      (((UE.tape x τ).1 j).2.1.2.1 r) i hi
    rw [zcH_exe c hw τ hsx j r, ← FlockComplete.Replay.Zc.r1ab_eq ((trsE x w (UE.tapeZk τ))[j.val]!) _ _
      (FlockComplete.Replay.Zc.r_size (zkTab x j).table.S.m (zkTab x j).table.hm (zcE c j r))] at e
    exact congrArg FlockLevel3.GF128.val e

theorem src_r1c {i : ℕ} (hi : i < 64) :
    ((srcE x c w τ j.val).rep r.val).masked ((layE x j).r1c i) =
      (FlockComplete.Replay.Zc.r1c ((trsE x w (UE.tapeZk τ))[j.val]!) ((zkTab x j).table.S.m - 6)
        ((zcE c j r).r (zkTab x j).table.S.m))[i]! +
      (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[(layE x j).r1c i]! := by
  refine (srcE_masked_pad x c w τ j r (lt_hab (layE x j) ?_)).trans (congrArg (· + _) ?_)
  · simp only [layE, _root_.Flock.Zk.PadLayout.new, _root_.Flock.Zk.PadLayout.r1c, _root_.Flock.Zk.PadLayout.fc,
      _root_.Flock.Zk.PadLayout.fa]
    omega
  · have e := Composed.honestAt_r1c (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwE x w τ j)
      (((UE.tape x τ).1 j).2.1.2.1 r) i hi
    rw [zcH_exe c hw τ hsx j r, ← FlockComplete.Replay.Zc.r1c_eq ((trsE x w (UE.tapeZk τ))[j.val]!) _ _
      (FlockComplete.Replay.Zc.r_size (zkTab x j).table.S.m (zkTab x j).table.hm (zcE c j r))] at e
    exact congrArg FlockLevel3.GF128.val e

theorem src_ml0 {i : ℕ} (hi : i < (stAt x j).m - 6) :
    ((srcE x c w τ j.val).rep r.val).masked ((layE x j).ml i 0) =
      (FlockComplete.Replay.Zc.msg ((trsE x w (UE.tapeZk τ))[j.val]!) ((zkTab x j).table.S.m - 6)
        ((zcE c j r).r (zkTab x j).table.S.m) (zcE c j r).z (zcE c j r).ρ i).1 +
      (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[(layE x j).ml i 0]! := by
  refine (srcE_masked_pad x c w τ j r (lt_hab (layE x j) ?_)).trans (congrArg (· + _) ?_)
  · simp only [layE, _root_.Flock.Zk.PadLayout.new, _root_.Flock.Zk.PadLayout.ml, _root_.Flock.Zk.PadLayout.fc,
      _root_.Flock.Zk.PadLayout.fa]
    omega
  · have e := Composed.honestAt_ml (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwE x w τ j)
      (((UE.tape x τ).1 j).2.1.2.1 r) i 0 hi (by decide)
    rw [ite_eq_left rfl, zcH_exe c hw τ hsx j r,
      FlockComplete.Replay.Zc.rounds_honest ((trsE x w (UE.tapeZk τ))[j.val]!) ((zkTab x j).table.S.m - 6) _
        (FlockComplete.Replay.Zc.r_size (zkTab x j).table.S.m (zkTab x j).table.hm (zcE c j r)) _ (zcE c j r).ρ _
        (rhoF_czExe c j r) _ hi] at e
    exact congrArg FlockLevel3.GF128.val e

theorem src_ml1 {i : ℕ} (hi : i < (stAt x j).m - 6) :
    ((srcE x c w τ j.val).rep r.val).masked ((layE x j).ml i 1) =
      (FlockComplete.Replay.Zc.msg ((trsE x w (UE.tapeZk τ))[j.val]!) ((zkTab x j).table.S.m - 6)
        ((zcE c j r).r (zkTab x j).table.S.m) (zcE c j r).z (zcE c j r).ρ i).2 +
      (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[(layE x j).ml i 1]! := by
  refine (srcE_masked_pad x c w τ j r (lt_hab (layE x j) ?_)).trans (congrArg (· + _) ?_)
  · simp only [layE, _root_.Flock.Zk.PadLayout.new, _root_.Flock.Zk.PadLayout.ml, _root_.Flock.Zk.PadLayout.fc,
      _root_.Flock.Zk.PadLayout.fa]
    omega
  · have e := Composed.honestAt_ml (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwE x w τ j)
      (((UE.tape x τ).1 j).2.1.2.1 r) i 1 hi (by decide)
    rw [ite_eq_right (show ¬ (1 : ℕ) = 0 by decide), zcH_exe c hw τ hsx j r,
      FlockComplete.Replay.Zc.rounds_honest ((trsE x w (UE.tapeZk τ))[j.val]!) ((zkTab x j).table.S.m - 6) _
        (FlockComplete.Replay.Zc.r_size (zkTab x j).table.S.m (zkTab x j).table.hm (zcE c j r)) _ (zcE c j r).ρ _
        (rhoF_czExe c j r) _ hi] at e
    exact congrArg FlockLevel3.GF128.val e

theorem src_fa :
    ((srcE x c w τ j.val).rep r.val).masked (layE x j).fa =
      (FlockComplete.Replay.Zc.tab ((zkTab x j).table.S.m - 6) (zcE c j r).z (zcE c j r).ρ
        ((trsE x w (UE.tapeZk τ))[j.val]!).az ((zkTab x j).table.S.m - 6))[0]! +
      (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[(layE x j).fa]! := by
  refine (srcE_masked_pad x c w τ j r (lt_hab (layE x j) ?_)).trans (congrArg (· + _) ?_)
  · simp only [_root_.Flock.Zk.PadLayout.fc]
    omega
  · have e := Composed.honestAt_fa (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwE x w τ j)
      (((UE.tape x τ).1 j).2.1.2.1 r)
    rw [zcH_exe c hw τ hsx j r,
      FlockComplete.Replay.Zc.finalA_honest ((trsE x w (UE.tapeZk τ))[j.val]!) ((zkTab x j).table.S.m - 6) _
        (zcE c j r).ρ _ (rhoF_czExe c j r)] at e
    exact congrArg FlockLevel3.GF128.val e

theorem src_fb :
    ((srcE x c w τ j.val).rep r.val).masked (layE x j).fb =
      (FlockComplete.Replay.Zc.tab ((zkTab x j).table.S.m - 6) (zcE c j r).z (zcE c j r).ρ
        ((trsE x w (UE.tapeZk τ))[j.val]!).bz ((zkTab x j).table.S.m - 6))[0]! +
      (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[(layE x j).fb]! := by
  refine (srcE_masked_pad x c w τ j r (lt_hab (layE x j) ?_)).trans (congrArg (· + _) ?_)
  · simp only [_root_.Flock.Zk.PadLayout.fb, _root_.Flock.Zk.PadLayout.fc]
    omega
  · have e := Composed.honestAt_fb (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwE x w τ j)
      (((UE.tape x τ).1 j).2.1.2.1 r)
    rw [zcH_exe c hw τ hsx j r,
      FlockComplete.Replay.Zc.finalB_honest ((trsE x w (UE.tapeZk τ))[j.val]!) ((zkTab x j).table.S.m - 6) _
        (zcE c j r).ρ _ (rhoF_czExe c j r)] at e
    exact congrArg FlockLevel3.GF128.val e

theorem src_fc :
    ((srcE x c w τ j.val).rep r.val).masked (layE x j).fc =
      _root_.Flock.dot (_root_.Flock.lagrange _root_.Flock.nodesL _root_.Flock.den6Inv (zcE c j r).z)
        (FlockComplete.Replay.Zc.r1c ((trsE x w (UE.tapeZk τ))[j.val]!) ((zkTab x j).table.S.m - 6)
          ((zcE c j r).r (zkTab x j).table.S.m)) +
      (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[(layE x j).fc]! := by
  refine (srcE_masked_pad x c w τ j r (lt_hab (layE x j) le_rfl)).trans (congrArg (· + _) ?_)
  have e := Composed.honestAt_fc (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwE x w τ j)
    (((UE.tape x τ).1 j).2.1.2.1 r)
  rw [zcH_exe c hw τ hsx j r,
    FlockComplete.Replay.Zc.finalC_honest ((trsE x w (UE.tapeZk τ))[j.val]!) ((zkTab x j).table.S.m - 6) _
      (FlockComplete.Replay.Zc.r_size (zkTab x j).table.S.m (zkTab x j).table.hm (zcE c j r))] at e
  exact congrArg FlockLevel3.GF128.val e

end Fields

end Masked

end Claims.ZKMal
