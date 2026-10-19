import Proofs.Flock.Claims.ZK.SplitInner
import Proofs.Flock.Claims.ZK.SimSession
import Proofs.Flock.Claims.Complete.ZkTables.RunsGo
import Proofs.Flock.Claims.Complete.ZkTables.MaskFree
import Proofs.Flock.Claims.Complete.Zk.Level0Premises

/-!
# `InnerBatched`: the batched constraint at the honest pads
-/

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

open FlockComplete Flock.With

/-- **The phases hybrids 2 and 3 share, answered by `asksCut`, that return** are those answered by `asksS` at the same
coins. -/
theorem innerPre_cut {coin : ℕ → ℕ → _root_.Flock.F128} {n : ℕ} {resp : ℕ → Msg}
    (hr : ∀ k, coinOf (resp k) = coin k) (hs : _root_.Flock.Hashes FlockComplete.PS) (st : _root_.Flock.Setup)
    (sch : _root_.Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (tr : FlockComplete.Trace)
    (t : Array _root_.Flock.F128) (cm0 : Commit) (extras : Array _root_.Flock.Claim) (pads : PadsZk)
    (ic : InnerCommit) (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List _root_.Flock.F128)) (words : List ℕ) :
    (cutS.st _root_.Flock.Transcript).R
      (SimInner.innerPre (asksCut coin n) hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce
        prior words)
      (SimInner.innerPre (asksS resp) hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce
        prior words) := by
  unfold SimInner.innerPre piopPZk rankCheckP
  sim_walk [zerocheckPZk_cut hr, lincheckPZk_cut hr, ringSwitchP_cut hr, ligeritoPZk_cut hr, cutS_reflT, cutS_refl]

end Masked

end Claims.ZKMal
