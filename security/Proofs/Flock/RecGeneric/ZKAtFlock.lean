module

public import Proofs.Flock.RecGeneric.ZKSessions
public import Proofs.Flock.SamplingGeneric.FlockZk

/-! The recursion's zero knowledge with Flock's ZK server at its masked deployment as each session, whose release refuses at V\* statements (`GenericRecursiveZKAtFlockServer`); definitions, no proofs. -/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.SamplingGeneric

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

noncomputable abbrev flockServerSess (rel : FlockZk.Parser) (T : ZKMal.CoinLayer.TableData)
    (vs : (FlockZk.Sess T).SwVals) (L : FlockZk.Release T) (wk : L.toSrv.Worker) (pc : FlockZk.PC T)
    (fc : FlockZk.FC T) (t : ℕ) (Xs : Msg → Finset Msg) : ZkSess :=
  ZkSess.ofZKOut (FlockZk.LiveRel rel T vs L wk pc fc) (FlockZk.rounds rel) (FlockZk.Tape T) L.toSrv.Chk
    (fun w => L.toSrv.proverD L.fillOf (FlockZk.rounds rel) wk w) (ZKServer.outD L.toSrv.stop (FlockZk.rounds rel))
    (FlockZk.simOut rel T L pc fc t) Xs

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims Claims.RecGeneric Claims.SamplingGeneric FlockSoundness.Discharge
open scoped NNReal ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

def GenericRecursiveZKAtFlockServer : Prop :=
  ∀ (rel : FlockZk.Parser) (T : ZKMal.CoinLayer.TableData) (vs : (FlockZk.Sess T).SwVals) (t : ℕ) (δ₁ : ℝ≥0),
    0 < t → Claims.Hm96Hiding ZkView.My ZkView.saltHash (δ₁ : ℝ≥0∞) →
    (FlockZk.Sess T).Tables → (FlockZk.Sess T).Complete (Claims.Flock.R (FlockDeploy.deployZk rel).parse) →
    (FlockZk.Sess T).DummyReg → (FlockZk.Sess T).Causal → (FlockZk.Sess T).SwCausal vs → (FlockZk.Sess T).TapeOK →
    ∀ (L₀ L : FlockZk.Release T), L = ZKServer.LeanRelease.loop ZKServer.Loop.loopNr T L₀ →
    ∀ (wk : L.toSrv.Worker) (pc : FlockZk.PC T) (fc : FlockZk.FC T) (Xs : Msg → Finset Msg)
      (enc : ZkView.Bits × ByteArray → Msg) {W : Type}
      (F : FirstPart W (Leaves.hm96 enc) (ℕ → (flockServerSess rel T vs L wk pc fc t Xs).WS))
      (Ls : Sessions (flockServerSess rel T vs L wk pc fc t Xs)),
      Ls.Shaped →
        ZKOut (ZkRelease.ofSessions F Ls).N (ZkRelease.ofSessions F Ls).Ω FlockZk.sha (ZkRelease.ofSessions F Ls).k
          (ZkRelease.ofSessions F Ls).prover (fun _ _ => True) (ZkRelease.ofSessions F Ls).outcome
          (ZkRelease.ofSessions F Ls).sim
          ((ZkRelease.ofSessions F Ls).err FlockZk.sha
            (Ls.err FlockZk.sha fun _ => FlockZk.errOut T vs L pc fc t (δ₁ : ℝ≥0∞)) (δ₁ : ℝ))

end Flock.Guarantees
