module

public import Proofs.Flock.SamplingGeneric.FlockZk
public import Proofs.Flock.Claims.Recursion.DeployOuter

/-! Flock's ZK server is zero knowledge per outcome at the recursive outer record (`server_zkOutRec`). -/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.SamplingGeneric
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

theorem VO_k_of_R {rel : FlockZk.Parser} {x : Msg} {w : Claims.Flock.Wit}
    (hR : Claims.Flock.R (RecDeploy.deployZkRec rel).parse x w) : RecDeploy.VO.k x = ZKMal.refK x + 1 := by
  obtain ⟨s, hs, -⟩ := hR
  obtain ⟨r, hr⟩ := RecDeploy.read_of_parseOf (rel := rel) (rd := RecDeploy.readOfO) hs
  obtain ⟨u, hu⟩ := RecDeploy.decodeRec_of_readOfO hr
  have hf : FlockDeploy.setupOf true x = none := RecDeploy.setupOf_none (RecDeploy.decode_none hu)
  rcases hso : RecDeploy.setupO x with _ | ⟨s', spec, sts⟩
  · have h1 : RecDeploy.VO.k x = 1 := by simp only [RecDeploy.VO, hso]
    have h2 := ZKMal.refK_none (ZKMal.sessOf_none hf hso)
    omega
  · have h1 : RecDeploy.VO.k x = 2 + FlockDeploy.rounds true spec sts := by simp only [RecDeploy.VO, hso]
    have h2 := ZKMal.refK_of_sess (ZKMal.sessOf_rec hso)
    omega

theorem server_zkOutRec (rel : FlockZk.Parser) (T : ZKMal.CoinLayer.TableData) (vs : (FlockZk.Sess T).SwVals)
    (t : ℕ) (ht : 0 < t) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding FlockSoundness.Discharge.ZkView.My FlockSoundness.Discharge.ZkView.saltHash δ₁)
    (hδ₁ : δ₁ ≠ ⊤) (hS : (FlockZk.Sess T).Tables)
    (hV : (FlockZk.Sess T).Complete (Claims.Flock.R (RecDeploy.deployZkRec rel).parse))
    (hD : (FlockZk.Sess T).DummyReg) (hc : (FlockZk.Sess T).Causal) (hsc : (FlockZk.Sess T).SwCausal vs)
    (hT : (FlockZk.Sess T).TapeOK) (L₀ L : FlockZk.Release T)
    (hL : L = ZKServer.LeanRelease.loop ZKServer.Loop.loopNr T L₀) (wk : L.toSrv.Worker) (pc : FlockZk.PC T)
    (fc : FlockZk.FC T) :
    Claims.ZKOut (fun _ => FlockZk.Tape T) (fun _ => L.toSrv.Chk) FlockZk.sha
      (FlockZk.roundsAt (RecDeploy.deployZkRec rel))
      (fun w _ => L.toSrv.proverD L.fillOf (FlockZk.roundsAt (RecDeploy.deployZkRec rel)) wk w)
      (FlockZk.LiveRelAt (RecDeploy.deployZkRec rel) T vs L wk pc fc)
      (ZKServer.outD L.toSrv.stop (FlockZk.roundsAt (RecDeploy.deployZkRec rel)))
      (FlockZk.simOutAt (RecDeploy.deployZkRec rel) T L pc fc t) (FlockZk.errOut T vs L pc fc t δ₁) :=
  FlockZk.server_zkOutAt (RecDeploy.deployZkRec rel) rfl (fun _ _ hR => VO_k_of_R hR) T vs t ht hT1 hδ₁ hS hV hD hc
    hsc hT L₀ L hL wk pc fc

end Claims.RecGeneric
