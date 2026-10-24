/-
The `--zk` verifier's `…With` functions (`Flock.With.Zk`) under a simulation (`Flock.With.Sim`): run with two oracles that
`S` relates hash by hash, each function's two runs are related by `S` (or by its lift to the transcript and the Ligerito
state).
-/
import Flock.With.SimVerify
import Flock.With.Zk

namespace Flock.With

open Transcript

variable {M N : Type → Type} [Monad M] [Monad N] [MonadExceptOf String M] [MonadExceptOf String N] [MonadLiftT V M]
  [MonadLiftT V N]

theorem zk_replayWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (st : Setup) (lay : Zk.PadLayout) (p : Proof) (root : ByteArray) :
    (S.st Transcript).R (Zk.replayWith hs st lay p root) (Zk.replayWith hs' st lay p root) := by
  unfold Zk.replayWith
  sim_walk [sampleWith_sim S hh, sampleVecWith_sim S hh]

theorem zk_tauCommitmentWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (tau : F128) (salt : ByteArray) : S.R (Zk.tauCommitmentWith hs tau salt) (Zk.tauCommitmentWith hs' tau salt) := by
  unfold Zk.tauCommitmentWith
  sim_walk [leafWith_sim S hh]

theorem zk_verifyInnerWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (lay : Zk.PadLayout) (pr : Zk.InnerProof) (finalC : F128) (cons : Array (String × Zk.Lin)) :
    (S.st Transcript).R (Zk.verifyInnerWith hs lay pr finalC cons) (Zk.verifyInnerWith hs' lay pr finalC cons) := by
  unfold Zk.verifyInnerWith
  sim_walk [sampleWith_sim S hh, sampleVecWith_sim S hh, zk_tauCommitmentWith_sim S hh, leafWith_sim S hh,
    climbWith_sim S (opsWith_sim S hh MerkleScheme.hm96Sha512),
    climbWith_sim S ⟨fun _ _ => hh.sha512 _, fun _ salt => S.bind (hh.sha512 _) fun x => leafWith_sim S hh x salt⟩]

theorem zk_queryLevel0With_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (ms : MerkleScheme) (p : LigeritoProof) (lv : Level) (cap : Array ByteArray) (fold : Array F256) :
    ((S.st Transcript).st LState).R (Zk.queryLevel0With hs ms p lv cap fold) (Zk.queryLevel0With hs' ms p lv cap fold) := by
  unfold Zk.queryLevel0With
  sim_walk [liftW_sim, liftLM_sim, sampleVecWith_sim S hh, merkleCheckWith_sim S (opsWith_sim S hh ms)]

theorem zk_ligeritoWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (ms : MerkleScheme) (sch : Schedule) (T : F128) (cap : Array ByteArray) (p : LigeritoProof)
    (basisAt : Array F256 → Nat → Array F256) (zl : Zk.Level0) :
    (S.st Transcript).R (Zk.ligeritoWith hs ms sch T cap p basisAt zl) (Zk.ligeritoWith hs' ms sch T cap p basisAt zl) := by
  unfold Zk.ligeritoWith
  sim_walk [liftW_sim, liftLM_sim, sampleWith_sim S hh, sampleVecWith_sim S hh, sampleKWith_sim S hh,
    challengeWith_sim S hh, oodIntroWith_sim S hh, queryLevelWith_sim S hh, zk_queryLevel0With_sim S hh]

theorem zk_verifyRepWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (spec : SessionSpec) (st : Setup) (sch : Schedule) (tPad : Nat) (sess : Session) (i r : Nat) (bytes : ByteArray) :
    S.R (Zk.verifyRepWith hs spec st sch tPad sess i r bytes) (Zk.verifyRepWith hs' spec st sch tPad sess i r bytes) := by
  unfold Zk.verifyRepWith
  sim_walk [zk_replayWith_sim S hh, ringSwitchWith_sim S hh, zk_ligeritoWith_sim S hh, zk_verifyInnerWith_sim S hh]

theorem zk_verifyWithM_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] {hs : Flock.Hashes M} {hs' : Flock.Hashes N}
    (hh : S.Hashes hs hs') (spec : SessionSpec) (tabs : Array Setup) (record : String) (proofs : Array ByteArray) :
    S.R (Zk.verifyWithM hs spec tabs record proofs) (Zk.verifyWithM hs' spec tabs record proofs) := by
  unfold Zk.verifyWithM
  sim_walk [decodeWith_sim S hh, dgWith_sim S hh, zk_verifyRepWith_sim S hh]

theorem zk_verifyJWithM_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] {hs : Flock.Hashes M} {hs' : Flock.Hashes N}
    (hh : S.Hashes hs hs') (spec : SessionSpec) (tabs : Array Setup) (j : Lean.Json) (proofs : Array ByteArray) :
    S.R (Zk.verifyJWithM hs spec tabs j proofs) (Zk.verifyJWithM hs' spec tabs j proofs) := by
  unfold Zk.verifyJWithM
  sim_walk [decodeJWith_sim S hh, dgWith_sim S hh, zk_verifyRepWith_sim S hh]

end Flock.With
