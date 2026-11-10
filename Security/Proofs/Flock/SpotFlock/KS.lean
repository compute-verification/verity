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

/-!
# SPOT[Flock]'s two-stage verifier: the compiler's premises at its code

`SpotFlock.Facts`'s facts as the premises the two-stage compiler's theorem (`ksTwo_atOrder`) names, at
`SpotFlock.code h`:

* `regMatchesTwo_code`, `refusesTwo_code`, `interiorFixed_code`: `RegMatchesTwo`, `Code.RefusesTwo` and
  `Code.InteriorFixed`; `replayCap` (`SpotFlock.Cap`) is `Code.ReplayCap` at `SpotFlock.cap h`;
* `sessionsIn_at`, `sessionsIn_code`: each unit's session in its template's scope, the reference statements being in
  Flock's masked scope (`Held.xS₀_ok`): a unit's session statement has its template's reference statement's shapes, so
  its `N₀` is that template's (`n₀`), the code's `unitN₀` at the unit (`SessionsIn`);
* `ksTwo_spot`: `CompiledTwo.KS` at Flock's masked record and SHA-512, from the sessions' openings
  (`SessionOpensTwo`); A3 at the coins' bytes is `SpotFlock.callsUniform`, and the run's acceptance gives each unit's
  session's one way (`SpotFlock.sessionRunsTo`);
* `sessionsTwo_spot`: the two-stage contract from the same openings (`sessionsTwo_atOrder`);
* `ksAtTwo_spot`: `CompiledTwo.KSAt`, KS with each hash term at a budget on one run's SHA-512 work
  (`compilerKSAtTwo`), from the same openings alone, the opening checks' calls bounded by the code's reveal count
  (`SpotFlock.revealsLe`, `openCallsLe_compiled`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling

variable (h : Held)

theorem regMatchesTwo_code : SamplingDeploy.RegMatchesTwo ((base h).atOrder (order h)) (reg h) (two h) := regMatchesTwo h

theorem refusesTwo_code (H : Msg → Msg) : ((base h).atOrder (order h)).RefusesTwo H (two h) := refusesTwo h H

theorem interiorFixed_code (H : Msg → Msg) : ((base h).atOrder (order h)).InteriorFixed H iroots :=
  interiorFixed h H

/-- **Each unit's session is in its template's scope**: its statement sets up in Flock's masked scope within the code's
tables, and its `N₀` is its template's reference statement's (`n₀`). -/
theorem sessionsIn_at (x : Msg) (u : ℕ) (v : Seen) (xS : Msg) (vS : Seen)
    (hv : (code h).session x u v = some (xS, vS)) :
    (∃ s spec sts, FlockDeploy.ZkIn xS s spec sts ∧ sts.size ≤ (code h).tables x) ∧
      FlockDeploy.N₀ true xS = n₀ h (templateOf h.ms u) := by
  obtain ⟨⟨s, spec, sts, hset, h1, h2, h3, h4, hn⟩, hshape⟩ := sessionsIn h x u v xS vS hv
  refine ⟨⟨s, spec, sts, ⟨hset, h1, h2, h3, fun st hst => ?_⟩, hn⟩, ?_⟩
  · obtain ⟨a, b, c, d, e⟩ := h4 st hst
    exact ⟨a, b, c, d, e⟩
  · show FlockDeploy.N₀ true xS =
      ((shapeOf (h.xS₀ (templateOf h.ms u))).map fun m => (2 : ℝ) ^ (Claims.Flock.zkLevel0 m).d).sum
    rw [← hshape]
    unfold FlockDeploy.N₀ shapeOf
    rw [FlockDeploy.tables_eq hset, hset]
    simp [FlockDeploy.tableLen, List.map_map, Function.comp_def]

/-- **Every session is in its unit's scope** (`SessionsIn`): the code's `unitN₀` is each unit's template's `N₀`. -/
theorem sessionsIn_code : SamplingDeploy.SessionsIn (code h) := sessionsIn_at h

/-- **SPOT[Flock]'s two-stage knowledge soundness at its code**: `ksTwo_atOrder` at `SpotFlock.code h`, A3 at its
coins' bytes, its code's facts, its sessions' runs one way and their numbers proved, from the sessions' openings.
Its inner count is the replay cap (`SpotFlock.cap`). -/
theorem ksTwo_spot (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h) (rs h)
      (cap h)).KS FlockDeploy.sha512 :=
  SamplingDeploy.ksTwo_atOrder (base h) (reg h) (two h) (fun _ => L) (fun _ => L) (order h) (fun _ => coinByte)
    (fun _ => coinByte) iroots (cap h) callsUniform callsUniform (regMatchesTwo_code h) (refusesTwo_code h _)
    (replayCap h _) (interiorFixed_code h _) (sessionRunsTo h _root_.Flock.Readable.relation FlockDeploy.sha512)
    (sessionsIn_code h) hopens

/-- **SPOT[Flock]'s two-stage contract at its code**: `sessionsTwo_atOrder` at `ksTwo_spot`'s premise, A3 at its
coins' bytes by `callsUniform`. -/
theorem sessionsTwo_spot (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
      (rs h) (cap h)).Sessions FlockDeploy.sha512 :=
  SamplingDeploy.sessionsTwo_atOrder (base h) (reg h) (two h) (fun _ => L) (fun _ => L) (order h) (fun _ => coinByte)
    (fun _ => coinByte) iroots (cap h) callsUniform callsUniform (regMatchesTwo_code h) (refusesTwo_code h _)
    (replayCap h _) (interiorFixed_code h _) (sessionRunsTo h _root_.Flock.Readable.relation FlockDeploy.sha512)
    (sessionsIn_code h) hopens

/-- **SPOT[Flock]'s two-stage knowledge soundness at a time budget**: `compilerKSAtTwo` at `ksTwo_spot`'s premises
(`sessionsTwo_spot`), with Flock's KS and binding as in `ksTwo_masked`, and the opening checks' count from the code's
reveal count (`revealsLe`, `openCallsLe_compiled`). -/
theorem ksAtTwo_spot (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
      (rs h) (cap h)).KSAt FlockDeploy.sha512 :=
  Claims.SamplingGeneric.compilerKSAtTwo FlockDeploy.Coin _ _ FlockDeploy.sha512
    (Claims.SamplingGeneric.flockArgKS FlockDeploy.sha512 _ _root_.Flock.SecurityProofs.MaskedKS
      (Claims.Sampling.listWithin_deployZk _)) (SamplingDeploy.commit_binds (code h) _) (sessionsTwo_spot h hopens)
    (SamplingDeploy.openCallsLe_compiled _ (code h) iroots _ (revealsLe h))

end SpotFlock
