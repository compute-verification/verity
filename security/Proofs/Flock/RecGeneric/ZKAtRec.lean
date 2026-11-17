module

public import Proofs.Flock.RecGeneric.ZKSessions
public import Proofs.Flock.RecGeneric.ZKServerRec
public import Proofs.Flock.Claims.Recursion.Deploy

/-! The recursion's zero knowledge with Flock's ZK server as each of `deployRec`'s V\* sessions (`GenericRecursiveZKAtRecServer`), a run's outer view only where `Xs x` holds its V\* statements; definitions, no proofs. -/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.SamplingGeneric

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

noncomputable abbrev recServerSess (rel : FlockZk.Parser) (T : ZKMal.CoinLayer.TableData)
    (vs : (FlockZk.Sess T).SwVals) (L : FlockZk.Release T) (wk : L.toSrv.Worker) (pc : FlockZk.PC T)
    (fc : FlockZk.FC T) (t : ℕ) (Xs : Msg → Finset Msg) : ZkSess :=
  ZkSess.ofZKOut (FlockZk.LiveRelAt (RecDeploy.deployZkRec rel) T vs L wk pc fc)
    (FlockZk.roundsAt (RecDeploy.deployZkRec rel)) (FlockZk.Tape T) L.toSrv.Chk
    (fun w => L.toSrv.proverD L.fillOf (FlockZk.roundsAt (RecDeploy.deployZkRec rel)) wk w)
    (ZKServer.outD L.toSrv.stop (FlockZk.roundsAt (RecDeploy.deployZkRec rel)))
    (FlockZk.simOutAt (RecDeploy.deployZkRec rel) T L pc fc t) Xs

noncomputable def recLayout (S : ZkSess) (b : RecDeploy.Brings) (msOf : Msg → List (Msg × Out)) : Sessions S where
  n := RecDeploy.nS
  xs x xO s := RecDeploy.vstarAt b.st x (msOf xO) s
  ko x := ∑ s ∈ Finset.range (RecDeploy.nS x), ((RecDeploy.nums b.st x s).rounds + 1)

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims Claims.RecGeneric Claims.SamplingGeneric FlockSoundness.Discharge
open scoped NNReal ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

def GenericRecursiveZKAtRecServer : Prop :=
  ∀ (rel : FlockZk.Parser) (T : ZKMal.CoinLayer.TableData) (vs : (FlockZk.Sess T).SwVals) (t : ℕ) (δ₁ : ℝ≥0),
    0 < t → Claims.Hm96Hiding ZkView.My ZkView.saltHash (δ₁ : ℝ≥0∞) →
    (FlockZk.Sess T).Tables → (FlockZk.Sess T).Complete (Claims.Flock.R (RecDeploy.deployZkRec rel).parse) →
    (FlockZk.Sess T).DummyReg → (FlockZk.Sess T).Causal → (FlockZk.Sess T).SwCausal vs → (FlockZk.Sess T).TapeOK →
    ∀ (L₀ L : FlockZk.Release T), L = ZKServer.LeanRelease.loop ZKServer.Loop.loopNr T L₀ →
    ∀ (wk : L.toSrv.Worker) (pc : FlockZk.PC T) (fc : FlockZk.FC T) (Xs : Msg → Finset Msg) (b : RecDeploy.Brings)
      (msOf : Msg → List (Msg × Out)) (enc : ZkView.Bits × ByteArray → Msg) {W : Type}
      (F : FirstPart W (Leaves.hm96 enc) (ℕ → (recServerSess rel T vs L wk pc fc t Xs).WS)),
      ZKOut (ZkRelease.ofSessions F (recLayout (recServerSess rel T vs L wk pc fc t Xs) b msOf)).N
        (ZkRelease.ofSessions F (recLayout (recServerSess rel T vs L wk pc fc t Xs) b msOf)).Ω FlockZk.sha
        (ZkRelease.ofSessions F (recLayout (recServerSess rel T vs L wk pc fc t Xs) b msOf)).k
        (ZkRelease.ofSessions F (recLayout (recServerSess rel T vs L wk pc fc t Xs) b msOf)).prover (fun _ _ => True)
        (ZkRelease.ofSessions F (recLayout (recServerSess rel T vs L wk pc fc t Xs) b msOf)).outcome
        (ZkRelease.ofSessions F (recLayout (recServerSess rel T vs L wk pc fc t Xs) b msOf)).sim
        ((ZkRelease.ofSessions F (recLayout (recServerSess rel T vs L wk pc fc t Xs) b msOf)).err FlockZk.sha
          ((recLayout (recServerSess rel T vs L wk pc fc t Xs) b msOf).err FlockZk.sha
            fun _ => FlockZk.errOut T vs L pc fc t (δ₁ : ℝ≥0∞)) (δ₁ : ℝ))

end Flock.Guarantees
