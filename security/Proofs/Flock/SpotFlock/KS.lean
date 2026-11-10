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
* `sessionsIn_code`: `SessionsIn`, the reference statement being in Flock's masked scope (`Held.xS₀_ok`): every
  session's statement has its shapes, so its `N₀` is the code's;
* `ksTwo_spot`: `CompiledTwo.KS` at Flock's masked record and SHA-512, from the sessions' openings
  (`SessionOpensTwo`); A3 at the coins' bytes is `SpotFlock.callsUniform` and the sessions' runs are
  `SpotFlock.sessionRuns`.
* `sessionsTwo_spot`: the two-stage contract from the same openings (`sessionsTwo_atOrder`).
* `ksTwo_spot_one`, `efficient_spot_one`: one session a run. The session doesn't read the unit (`SpotFlock.session`),
  so every slot of a sample has one key (`SamplingDeploy.compiledTwoOne`) and the extractor runs Flock's once a run:
  KS there, and its reruns, at most `1 + k_out + k_out·k_in·(1 + e_s)` in expectation. The code's reveal count
  (`SpotFlock.revealsLe`) bounds the opening checks' calls (`openCallsLe_compiled`).
* `ksAtTwo_spot`, `ksAtTwo_spot_one`: `CompiledTwo.KSAt`, KS with each hash term at a budget on one run's SHA-512 work
  (`compilerKSAtTwo`), from the same openings alone, the opening checks' calls bounded by the code's reveal count; and
  the same at one session a run.
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
coins' bytes, its code's facts and its sessions' runs proved, from the sessions' openings. Its inner count is the
replay cap (`SpotFlock.cap`). -/
theorem ksTwo_spot (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h) (rs h)
      (cap h)).KS FlockDeploy.sha512 :=
  SamplingDeploy.ksTwo_atOrder (base h) (reg h) (two h) (fun _ => L) (fun _ => L) (order h) (fun _ => coinByte)
    (fun _ => coinByte) iroots (cap h) callsUniform callsUniform (regMatchesTwo_code h) (refusesTwo_code h _)
    (replayCap h _) (interiorFixed_code h _) (sessionRuns h _root_.Flock.Readable.relation FlockDeploy.sha512).to
    (sessionsIn_code h) hopens

/-- **SPOT[Flock]'s two-stage contract at its code**: `sessionsTwo_atOrder` at `ksTwo_spot`'s premise, A3 at its
coins' bytes by `callsUniform`. -/
theorem sessionsTwo_spot (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
      (rs h) (cap h)).Sessions FlockDeploy.sha512 :=
  SamplingDeploy.sessionsTwo_atOrder (base h) (reg h) (two h) (fun _ => L) (fun _ => L) (order h) (fun _ => coinByte)
    (fun _ => coinByte) iroots (cap h) callsUniform callsUniform (regMatchesTwo_code h) (refusesTwo_code h _)
    (replayCap h _) (interiorFixed_code h _) (sessionRuns h _root_.Flock.Readable.relation FlockDeploy.sha512).to
    (sessionsIn_code h) hopens

/-- **SPOT[Flock] with one session a run is knowledge sound**: its session doesn't read the unit
(`SpotFlock.session`), so every slot of a sample has one key and the extractor runs Flock's once a run (`ksTwo_one`
at `sessionsTwo_spot`). -/
theorem ksTwo_spot_one (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    (SamplingDeploy.compiledTwoOne (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
      (rs h) (cap h)).KS FlockDeploy.sha512 :=
  SamplingDeploy.ksTwo_one (code h) iroots (two h) (rs h) (cap h) (sessionsTwo_spot h hopens) fun _ _ _ _ => rfl

/-- **SPOT[Flock]'s extractor with one session a run**: at most `1 + k_out + k_out·k_in·(1 + e_s)` reruns in
expectation and `k_out·k_in·n_PU·openCalls` own SHA-512 calls (`compilerEfficientTwo` at `sessionsTwo_one`), the
opening checks' count from the code's reveal count (`revealsLe`). -/
theorem efficient_spot_one (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    (SamplingDeploy.compiledTwoOne (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
      (rs h) (cap h)).Efficient FlockDeploy.sha512 :=
  Claims.SamplingGeneric.compilerEfficientTwo FlockDeploy.Coin _ _ FlockDeploy.sha512
    (SamplingDeploy.sessionsTwo_one _ (code h) iroots (two h) (rs h) (cap h) _ (sessionsTwo_spot h hopens)
      fun _ _ _ _ => rfl)
    (SamplingDeploy.openCallsLe_compiled _ (code h) iroots _ (revealsLe h))

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

/-- **SPOT[Flock] with one session a run, at a time budget**: `ksAtTwo_spot`'s statement at
`SamplingDeploy.compiledTwoOne`, whose work counts one session a run (`compilerKSAtTwo` at `sessionsTwo_one`). -/
theorem ksAtTwo_spot_one (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    (SamplingDeploy.compiledTwoOne (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
      (rs h) (cap h)).KSAt FlockDeploy.sha512 :=
  Claims.SamplingGeneric.compilerKSAtTwo FlockDeploy.Coin _ _ FlockDeploy.sha512
    (Claims.SamplingGeneric.flockArgKS FlockDeploy.sha512 _ _root_.Flock.SecurityProofs.MaskedKS
      (Claims.Sampling.listWithin_deployZk _)) (SamplingDeploy.commit_binds (code h) _)
    (SamplingDeploy.sessionsTwo_one _ (code h) iroots (two h) (rs h) (cap h) _ (sessionsTwo_spot h hopens)
      fun _ _ _ _ => rfl)
    (SamplingDeploy.openCallsLe_compiled _ (code h) iroots _ (revealsLe h))

end SpotFlock
