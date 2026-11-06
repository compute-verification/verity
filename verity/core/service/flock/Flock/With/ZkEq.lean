/-
The `--zk` copies under the deployed hashes: run in `V` with `Hashes.ofSha512 Sha512.hash`, each `Zk.…With` is the compiled
verifier's function it copies, so `Zk.verifyWith Sha512.hash` is `Zk.verify` for every session whose tables' Merkle schemes
are the registry's.
-/
import Flock.With.Eq
import Flock.With.Zk

namespace Flock

open Transcript

namespace With

theorem leafWith_std (x salt : ByteArray) : Hm96.Default512.leafWith std x salt = pure (Hm96.Default512.leaf x salt) := rfl

theorem monadLift_TMW_pure {α : Type} (x : α) : (monadLift (pure x : V α) : TMW V α) = pure x := rfl

theorem zk_replayWith_std (st : Setup) (lay : Zk.PadLayout) (p : Proof) (root : ByteArray) :
    Zk.replayWith std st lay p root = Zk.replay st lay p root := by
  unfold Zk.replayWith Zk.replay
  simp only [liftTM_V, sampleWith_std, sampleVecWith_std]
  rfl

theorem zk_tauCommitmentWith_std (tau : F128) (salt : ByteArray) :
    Zk.tauCommitmentWith std tau salt = pure (Zk.tauCommitment tau salt) := rfl

theorem zk_verifyInnerWith_std (lay : Zk.PadLayout) (pr : Zk.InnerProof) (finalC : F128) (cons : Array (String × Zk.Lin)) :
    Zk.verifyInnerWith std lay pr finalC cons = Zk.verifyInner lay pr finalC cons := by
  unfold Zk.verifyInnerWith Zk.verifyInner
  simp only [liftTM_V, sampleWith_std, sampleVecWith_std, zk_tauCommitmentWith_std, std_sha512, leafWith_std,
    hm96With_std, ofScheme_climb, monadLift_TMW_pure, pure_bind]

theorem zk_queryLevel0With_std (ms : MerkleScheme) (h : ms.Named) (p : LigeritoProof) (lv : Level) (cap : Array ByteArray)
    (fold : Array F256) : Zk.queryLevel0With std ms p lv cap fold = Zk.queryLevel0 ms p lv cap fold := by
  unfold Zk.queryLevel0With Zk.queryLevel0
  simp only [liftLM_V, liftW_V, sampleVecWith_std, opsWith_std ms h, ofScheme_merkleCheck, monadLift_LMW_bind]
  rfl

theorem zk_ligeritoWith_std (ms : MerkleScheme) (h : ms.Named) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (p : LigeritoProof) (basisAt : Array F256 → Nat → Array F256) (zl : Zk.Level0) :
    Zk.ligeritoWith std ms sch T cap p basisAt zl = Zk.ligerito ms sch T cap p basisAt zl := by
  unfold Zk.ligeritoWith Zk.ligerito
  simp only [liftTM_V, liftLM_V, liftW_V, sampleWith_std, sampleVecWith_std, sampleKWith_std, challengeWith_std,
    oodIntroWith_std, queryLevelWith_std ms h, zk_queryLevel0With_std ms h]
  rfl

theorem zk_verifyRepWith_std (spec : SessionSpec) (st : Setup) (h : st.merkle.Named) (sch : Schedule) (tPad : Nat)
    (sess : RunRecord) (i r : Nat) (bytes : ByteArray) :
    Zk.verifyRepWith std spec st sch tPad sess i r bytes = Zk.verifyRep spec st sch tPad sess i r bytes := by
  unfold Zk.verifyRepWith Zk.verifyRep
  simp only [monadLift_self, zk_replayWith_std, ringSwitchWith_std, zk_ligeritoWith_std _ h, zk_verifyInnerWith_std]
  rfl

end With

/-- `Zk.verify` is `Zk.verifyWith Sha512.hash` when every table's Merkle scheme is the registry's of its id. -/
theorem Zk.verify_eq_verifyWith (spec : SessionSpec) (tabs : Array Setup) (record : String) (proofs : Array ByteArray)
    (hN : ∀ st ∈ tabs, st.merkle.Named) :
    Zk.verify spec tabs record proofs = Zk.verifyWith Sha512.hash spec tabs record proofs := by
  have hN' := With.named_getElem! tabs hN
  unfold Zk.verifyWith Zk.verifyWithM Zk.verify
  simp only [monadLift_self, With.decodeWith_std, With.dgWith_std, pure_bind, With.zk_verifyRepWith_std _ _ (hN' _)]

end Flock
