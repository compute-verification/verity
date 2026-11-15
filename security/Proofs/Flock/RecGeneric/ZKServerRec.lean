module

public import Proofs.Flock.SamplingGeneric.FlockZk
public import Proofs.Flock.Claims.Recursion.DeployOuter

/-!
# Flock's ZK server at the recursive outer record

The ZK server's session reads the masked session a statement sets up, Flock's or a V\* part's (`ZKMal.sessOf`). A
statement the recursive outer record (`RecDeploy.deployZkRec`) parses is a V\* part's, so it sets up no session of
Flock's (`RecDeploy.decode_none`), and the session's rounds there are the recursive setup's (`ZKMal.sessOf_rec`): the
outer verifier's rounds are the session's and its last message (`VO_k_of_R`). So Flock's ZK server is zero knowledge per
outcome at the outer record, at the same release of the session loop (`server_zkOutRec`, `FlockZk.server_zkOutAt` there),
on the statements of the V\* sessions `RecDeploy.deployRec` stages.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.SamplingGeneric
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **At a statement the recursive outer record relates, its verifier's rounds are the session's and its last
message** (`RecDeploy.VO.k x = refK x + 1`): the statement is a V\* part's, so it sets up no session of Flock's
(`RecDeploy.decode_none`), and `refK` reads the recursive setup (`ZKMal.sessOf_rec`), `0` where there is none. -/
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

/-- **Flock's ZK server is zero knowledge per outcome at the recursive outer record** (`RecDeploy.deployZkRec rel`):
`FlockZk.server_zkOutAt` there, whose rounds on its relation are the session's (`VO_k_of_R`). The server is the session
loop's release `L` at the coin layer's tables `T`, as in `FlockZk.server_zkOut`; at a V\* part's statement it serves the
recursive setup's masked session. Its premises are `server_zkOut`'s, with the session's completeness on the outer
record's relation. -/
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
