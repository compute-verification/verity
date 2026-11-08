module

public import Proofs.Flock.SpotFlock.Facts
public import Proofs.Flock.SpotFlock.Coins
public import Proofs.Flock.SpotFlock.Sessions
public import Proofs.Flock.SamplingGeneric.TwoAt
public import Proofs.Flock.Claims.Sampling.Deployed
public import Proofs.Flock.Claims.NumbersDeploy

/-!
# SPOT[Flock]'s two-stage verifier: the compiler's premises at its code

`SpotFlock.Facts`'s facts as the premises the two-stage compiler's theorem (`ksTwo_atOrder`) names, at
`SpotFlock.code h`:

* `regMatchesTwo_code`, `refusesTwo_code`, `interiorFixed_code`: `RegMatchesTwo`, `Code.RefusesTwo` and
  `Code.InteriorFixed`;
* `sessionsIn_code`: `SessionsIn`, the reference statement being in Flock's masked scope (`Held.xS₀_ok`): every
  session's statement has its shapes, so its `N₀` is the code's;
* `ksTwo_spot`: `CompiledTwo.KS` at Flock's masked record and SHA-512, from the sessions' openings
  (`SessionOpensTwo`); A3 at the coins' bytes is `SpotFlock.callsUniform` and the sessions' runs are
  `SpotFlock.sessionRuns`.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling

variable (h : Held)

theorem regMatchesTwo_code : SamplingDeploy.RegMatchesTwo ((base h).atOrder (order h)) (reg h) (two h) := regMatchesTwo h

theorem refusesTwo_code (H : Msg → Msg) : ((base h).atOrder (order h)).RefusesTwo H (two h) := refusesTwo h H

theorem interiorFixed_code (H : Msg → Msg) : ((base h).atOrder (order h)).InteriorFixed H iroots :=
  interiorFixed h H

theorem sessionsIn_code : SamplingDeploy.SessionsIn (code h) := by
  intro x u v xS vS hv
  obtain ⟨⟨s, spec, sts, hset, h1, h2, h3, h4, hn⟩, hshape⟩ := sessionsIn h x u v xS vS hv
  refine ⟨⟨s, spec, sts, ⟨hset, h1, h2, h3, fun st hst => ?_⟩, hn⟩, ?_⟩
  · obtain ⟨a, b, c, d, e⟩ := h4 st hst
    exact ⟨a, b, c, d, e⟩
  · show FlockDeploy.N₀ true xS = ((shapeOf h.xS₀).map fun m => (2 : ℝ) ^ (Claims.Flock.zkLevel0 m).d).sum
    rw [← hshape]
    unfold FlockDeploy.N₀ shapeOf
    rw [FlockDeploy.tables_eq hset, hset]
    simp [FlockDeploy.tableLen, List.map_map, Function.comp_def]

/-- **SPOT[Flock]'s two-stage knowledge soundness at its code**: `ksTwo_atOrder` at `SpotFlock.code h`, A3 at its
coins' bytes, its code's facts and its sessions' runs proved, from the sessions' openings. -/
theorem ksTwo_spot (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h) (rs h)).KS
      FlockDeploy.sha512 :=
  SamplingDeploy.ksTwo_atOrder (base h) (reg h) (two h) (fun _ => L) (fun _ => L) (order h) (fun _ => coinByte)
    (fun _ => coinByte) iroots callsUniform callsUniform (regMatchesTwo_code h) (refusesTwo_code h _)
    (interiorFixed_code h _) (sessionRuns h _root_.Flock.Readable.relation FlockDeploy.sha512) (sessionsIn_code h)
    hopens

end SpotFlock
