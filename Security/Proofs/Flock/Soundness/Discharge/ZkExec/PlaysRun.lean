import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysRep
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsCut
import Proofs.Flock.Soundness.Discharge.ZkExec.Inner

/-!
# A `--zk` rep's `RepRun`

`zk_rep_run`: `zk_rep_plays`'s run, cut by `zk_cut` along `repShapeZ`, is a `RepRun` for the decoder `decZ`. The cut's
size hypotheses come from the run itself (`repProg_sizes`: `Zk.ligerito`'s shape check on `ybar`, `verifyInner`'s
on `τ`'s commitment and `Y`), and `repShapeZ` is `shapeZE` at the executable's dimensions (`ZkTabSeam`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

theorem ligerito_ybar (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray)
    (p : LigeritoProof) (basisAt : Array F256 → Nat → Array F256) (zl : Zk.Level0) {t t' : Flock.Transcript}
    {rec : Zk.L0Rec} (h : Zk.ligerito ms sch T cap p basisAt zl t = .ok (rec, t')) :
    zl.msgs.ybar.size = zl.tPad := by
  unfold Zk.ligerito at h
  try dsimp only at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨hc, -⟩ := check_bind h
  simp only [Bool.and_eq_true, beq_iff_eq] at hc
  exact hc.1

/-- **The sizes a run of `repProg` checks**: `ybar`'s `t_pad` elements (`Zk.ligerito`), `τ`'s 64-byte commitment and
`Y`'s `len + K_PAD` words (`verifyInner`). -/
theorem repProg_sizes (st : Setup) (sch : Flock.Schedule) (tPad r : ℕ) (extras : Array Flock.Claim)
    (zp : Zk.ZkProof) {t tr : Flock.Transcript}
    (hrun : (repProg st sch tPad r extras zp).run t = .ok ((), tr)) :
    zp.level0.ybar.size = tPad ∧ zp.inner.tauCommit.size = 64 ∧
      zp.inner.y.size = (layZ st sch).len + Zk.K_PAD := by
  unfold repProg at hrun
  rw [StateT.run] at hrun
  try dsimp only at hrun
  obtain ⟨rp, t₁, h₁, hrun⟩ := bind_ok hrun
  try dsimp only at hrun
  obtain ⟨⟨Tb, sws⟩, t₂, h₂, hrun⟩ := bind_ok hrun
  obtain ⟨rc, t₃, h₃, hrun⟩ := bind_ok hrun
  cases hv : Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rc zp.proof.ligerito zp.level0 with
  | error e =>
    rw [hv] at hrun
    try dsimp only at hrun
    obtain ⟨_, _, h₄, _⟩ := bind_ok hrun
    cases h₄
  | ok lg =>
  rw [hv] at hrun
  try dsimp only at hrun
  obtain ⟨lg', t₄, h₄, h₅⟩ := bind_ok hrun
  obtain ⟨hy, -, -, -, -, htau, -⟩ := verifyInner_ok _ _ _ _ _ _ h₅
  exact ⟨ligerito_ybar _ _ _ _ _ _ _ h₃, htau, hy⟩

theorem zk_rep_run : ZkRepRun := by
  intro mPts Sl Hs T pts r st sch tPad extras bytes zp rounds dom tr hsch hdl hseam hlen hext hdec hroot hwf hret
    hrun hk
  obtain ⟨pend, hpl, hne⟩ := zk_rep_plays Hs T pts r st sch tPad extras zp rounds dom tr hsch hseam hlen hext hwf
    hrun hk
  obtain ⟨Tv, hl, hR⟩ := zk_rep_frames st sch tPad r.val extras zp rounds dom tr hsch hrun hk hret
  obtain ⟨hyb, htau, hy⟩ := repProg_sizes _ _ _ _ _ _ hrun
  have hcut := zk_cut st sch tPad extras bytes zp Tv dom rounds hsch hdl hdec hroot hyb htau hy hl hR
  have hσ : repShapeZ 𝔽 𝕂 Dig T (Model.extraClaims T.S T.ptLocal mPts pts).length =
      shapeZE st.m st.kLog (2 + extras.size) (schedOf sch) tPad (layZ st sch).len := by
    unfold repShapeZ shapeZE
    rw [← hlen, hseam.hm, hseam.hk, hseam.hsch, hseam.he, hseam.ht, hseam.hnp, hseam.hnP, hseam.hq,
      Nat.add_comm extras.size 2]
  refine ⟨pend, ⟨fun rd hrd => ?_, _, hpl, ?_⟩, hne⟩
  · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hrd
    have hi' : i < rounds.size := by simpa using hi
    have := hwf i hi'
    rw [getElem!_pos rounds i hi'] at this
    simpa using this
  · rw [hσ]
    exact hcut

end FlockSoundness.Discharge.ZkExec
