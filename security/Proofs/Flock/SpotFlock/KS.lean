module

public import Proofs.Flock.SpotFlock.Facts
public import Proofs.Flock.SpotFlock.Coins
public import Proofs.Flock.SpotFlock.Cap
public import Proofs.Flock.SpotFlock.Sessions
public import Proofs.Flock.SamplingGeneric.TwoAt
public import Proofs.Flock.Claims.Sampling.Deployed
public import Proofs.Flock.Claims.NumbersDeploy
import Proofs.Flock.SamplingGeneric.TwoKSAt
import Proofs.Flock.SpotFlock.RevealsLe

/-! The two-stage compiler's premises at SPOT[Flock]'s code, and its knowledge soundness there from the sessions'
openings (`ksTwo_spot`, `ksAtTwo_spot`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling

variable (h : Held)

theorem regMatchesTwo_code : SamplingDeploy.RegMatchesTwo ((base h).atOrder (order h)) (reg h) (two h) := regMatchesTwo h

theorem refusesTwo_code (H : Msg → Msg) : ((base h).atOrder (order h)).RefusesTwo H (two h) := refusesTwo h H

theorem interiorFixed_code (H : Msg → Msg) : ((base h).atOrder (order h)).InteriorFixed H iroots :=
  interiorFixed h H

theorem sessionsIn_at (x : Msg) (u : ℕ) (v : Seen) (xS : Msg) (vS : Seen)
    (hv : (code h).session x u v = some (xS, vS)) :
    (∃ s spec sts, FlockDeploy.ZkIn xS s spec sts ∧ sts.size ≤ (code h).tables x) ∧
      FlockDeploy.N₀ true xS = n₀ h (templateOf h.ms u) := by
  obtain ⟨⟨s, spec, sts, hset, h1, h2, h3, h4, hn⟩, hshape⟩ := sessionsIn h x u v xS vS hv
  refine ⟨⟨s, spec, sts, ⟨hset, h1, h2, h3, fun st hst => ?_⟩, hn⟩, ?_⟩
  · obtain ⟨a, b, c, d, e⟩ := h4 st hst
    exact ⟨a, b, c, d, e⟩
  · show FlockDeploy.N₀ true xS =
      ((shapeOf (h.xS₀ (templateOf h.ms u))).map fun p => (2 : ℝ) ^ (Claims.Flock.zkLevel0 p.1 p.2).d).sum
    rw [← hshape]
    unfold FlockDeploy.N₀ shapeOf
    rw [FlockDeploy.tables_eq hset, hset]
    simp [FlockDeploy.tableLen, List.map_map, Function.comp_def]

theorem sessionsIn_code : SamplingDeploy.SessionsIn (code h) := sessionsIn_at h

theorem ksTwo_spot (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h) (rs h)
      (cap h)).KS FlockDeploy.sha512 :=
  SamplingDeploy.ksTwo_stage (base h) (reg h) (two h) (fun _ => L) (fun _ => h.stage * L) (fun _ => coinByte)
    (fun _ => h.stage) (stageOrder h) (fun _ => stageByte) iroots (cap h) callsUniform (stageUniform h.stage)
    (regMatchesTwo_code h) (refusesTwo_code h _) (replayCap h _) (interiorFixed_code h _)
    (sessionRunsTo h _root_.Flock.Readable.relation FlockDeploy.sha512) (sessionsIn_code h) hopens

theorem sessionsTwo_spot (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
      (rs h) (cap h)).Sessions FlockDeploy.sha512 :=
  SamplingDeploy.sessionsTwo_stage (base h) (reg h) (two h) (fun _ => L) (fun _ => h.stage * L) (fun _ => coinByte)
    (fun _ => h.stage) (stageOrder h) (fun _ => stageByte) iroots (cap h) callsUniform (stageUniform h.stage)
    (regMatchesTwo_code h) (refusesTwo_code h _) (replayCap h _) (interiorFixed_code h _)
    (sessionRunsTo h _root_.Flock.Readable.relation FlockDeploy.sha512) (sessionsIn_code h) hopens

theorem ksAtTwo_spot (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
      (rs h) (cap h)).KSAt FlockDeploy.sha512 :=
  Claims.SamplingGeneric.compilerKSAtTwo FlockDeploy.Coin _ _ FlockDeploy.sha512
    (Claims.SamplingGeneric.flockArgKS FlockDeploy.sha512 _ _root_.Flock.SecurityProofs.MaskedKS
      (Claims.Sampling.listWithin_deployZk _)) (SamplingDeploy.commit_binds (code h) _) (sessionsTwo_spot h hopens)
    (SamplingDeploy.openCallsLe_compiled _ (code h) iroots _ (revealsLe h))

end SpotFlock
