module

public import Definitions.Claims.SpotFlock
public import Proofs.Flock.ZkAny.DeferLoop
public import Proofs.Flock.SamplingGeneric.TwoDeployed
import Proofs.Flock.Masked
import Proofs.Flock.SamplingGeneric
import Proofs.Flock.SamplingGeneric.Two
import Proofs.Flock.SamplingGeneric.TwoCount
import Proofs.Flock.SamplingGeneric.ZK
import Proofs.Flock.SessionLoop.TablesSide

/-!
# SPOT-Flock at the deployment: L1's export, L3's, and the deployed guarantees

The layers and their interfaces are `Claims.SpotFlock`'s. Here they are at the deployment.

* **L1** (`Flock.Spot`): `arg p`, Flock at the deployed masked record as a proof system for proof blocks with the ZK
  server's sessions at the deployment `p`; `arg_KS` (from `MaskedKS`) and `arg_ZK` (from `Loop.zkOut_loopDeployed`).
* **L3** (`SpotFlock`): `deployed p` and `deployedTwo p`, the deployed sampled verifier with its Flock sessions as a
  masked compiled verifier and a two-stage compiled verifier over `arg p`; `refines` and `refinesTwo`, the compiler's
  contract at them.
* **The deployed guarantees** (`SpotFlock.ks`, `ksReg`, `ksCount`, `bits`, `zk`, `ksTwo`, `exfilTwo`): each L2's theorem
  at L1's export and L3's, in one line. `Claims.SpotFlock.composes` and `composesTwo` are the same composition at any
  proof system and compiled verifier.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- the release's session types, as in `DeferLoop`

namespace Claims.SpotFlock

open Claims.SamplingGeneric

theorem composes : Composes :=
  fun Coin _ _ _ A L H h1 h3 =>
    ⟨compilerKS Coin A.toArg L.toCompiled H h1.ks h3.binds h3.sessions,
      compilerKSReg Coin A.toArg L.toCompiled H h1.ks h3.binds h3.sessions,
      compilerKSCount Coin A.toArg L.toCompiled H h1.ks h3.binds h3.sessions,
      compilerBits Coin A.toArg L.toCompiled H h1.ks h3.binds h3.sessions h3.openCalls,
      compilerZK Coin A L H h1.zk h3.zkSessions⟩

theorem composesTwo : ComposesTwo :=
  fun Coin _ _ _ A L Z H h1 h3 =>
    ⟨compilerKSTwo Coin A.toArg L H h1.ks h3.binds h3.sessions,
      compilerExfilTwo Coin A.toArg L H h1.ks h3.binds h3.sessions,
      compilerZK Coin A Z H h1.zk h3.zkSessions⟩

end Claims.SpotFlock

/-! ## L1: Flock -/

namespace Flock.Spot

open Claims Claims.SamplingGeneric

/-- The deployed masked record, the one Flock's sessions run at. -/
noncomputable abbrev D := FlockDeploy.deployZk _root_.Flock.Readable.relation

/-- **The ZK server's deployment**: the session's switched values `vs`, the repetitions `t > 0`, the loop's release
`L₀` and its sources `src`, and the worker `wk`. -/
structure Server where
  vs : Claims.ZKServer.Loop.finS.SwVals
  t : ℕ
  ht : 0 < t
  L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp
  src : Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
    _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)
  wk : (Claims.ZKServer.Loop.deferL L₀ src).toSrv.Worker

/-- The release the deployment's sessions run at: the loop's own frames, link and `Finish` on its sources. -/
noncomputable abbrev Server.rel (p : Server) : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp :=
  Claims.ZKServer.Loop.deferL p.L₀ p.src

/-- **L1's relation**: Flock's at the deployed record, at a statement in the masked verifier's scope that the loop
serves (`SessionLoopZKDeployed`'s domain premises). -/
def R : Rel Claims.Flock.Wit := fun x w =>
  Claims.Flock.R D.parse x w ∧ Claims.Flock.Scope D x ∧ Claims.ZKServer.Loop.Serves x

/-- **L1's export**: Flock at the deployed record as a proof system for proof blocks, `E^rep` at its decoder
(`Arg.ofFlock`) at the relation `R`, with the ZK server's sessions at the deployment `p`: `V.k + 1` rounds (`Hello`
first), the deferred server's prover on the session's nonce and tape and the worker's challenges, its outcome (`outD`:
the verdict, and whether it stopped at its `Finish`), the simulator given the outcome (`FlockZk.simOut`) and the error
at hm96-sha512's hiding gap at the pinned key (`FlockZk.errOut`). -/
noncomputable def arg (p : Server) : ZkArg FlockDeploy.Coin where
  toArg := { Arg.ofFlock D with R := R }
  kz := FlockZk.rounds _root_.Flock.Readable.relation
  Nonce := Claims.ZKServer.Loop.Tp
  Rest := p.rel.toSrv.Chk
  P w := p.rel.toSrv.proverD p.rel.fillOf (FlockZk.rounds _root_.Flock.Readable.relation) p.wk w
  St := Option Bool × Bool
  out := Claims.ZKServer.outD p.rel.toSrv.stop (FlockZk.rounds _root_.Flock.Readable.relation)
  S := FlockZk.simOut _root_.Flock.Readable.relation Claims.ZKMal.Masked.dataE p.rel Claims.ZKServer.Loop.progOf
    Claims.ZKServer.Loop.freshOf p.t
  εzk := FlockZk.errOut Claims.ZKMal.Masked.dataE p.vs p.rel Claims.ZKServer.Loop.progOf Claims.ZKServer.Loop.freshOf
    p.t (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap FlockSoundness.Discharge.ZkView.My
      FlockSoundness.Discharge.ZkView.saltHash))

/-- The knowledge-soundness hash and the ZK server's are one SHA-512 on messages. -/
theorem sha512_eq_shaH : FlockDeploy.sha512 = Claims.ZKMal.CoinOpen.shaH := by
  funext m
  rfl

/-- **Flock is knowledge sound at its own relation**: `flockArgKS` at the deployed record, from `MaskedKS` and the
deployed decoder's list bound (`listWithin_deployZk`). -/
theorem ofFlock_KS : (Arg.ofFlock D).KS FlockDeploy.sha512 :=
  flockArgKS FlockDeploy.sha512 D _root_.Flock.SecurityProofs.MaskedKS (Claims.Sampling.listWithin_deployZk _)

/-- **L1's knowledge soundness**: `ofFlock_KS` at `R`, which is Flock's relation at the statements in scope that the
loop serves; it reads the masked verifier refusing every other statement. -/
theorem arg_KS (p : Server) : (arg p).toArg.KS FlockDeploy.sha512 := by
  sorry

/-- **L1's code facts for zero knowledge** at the deployment `p`: the switched values are causal, and at every
statement and witness in `R` the loop's inner round 0 and the live and stand-in rounds' bytes are the session's. -/
structure ZkCode (p : Server) : Prop where
  swCausal : Claims.ZKServer.Loop.finS.SwCausal p.vs
  round0 : ∀ x w, R x w →
    _root_.Flock.Guarantees.LoopInnerRound0 Claims.ZKMal.Masked.dataE p.L₀ (Claims.ZKServer.Loop.realOf p.src) x w
  live : ∀ x w, R x w →
    Claims.ZKServer.LeanRelease.LiveRoundEq Claims.ZKServer.Loop.finS p.vs p.rel Claims.ZKMal.CoinOpen.shaH p.wk x w
  fill : ∀ x w, R x w →
    Claims.ZKServer.LeanRelease.FillRoundEq Claims.ZKServer.Loop.finS p.vs p.rel Claims.ZKMal.CoinOpen.shaH p.wk x w

/-- **L1's trust item for zero knowledge** at the deployment `p`: at every statement and witness in `R`, the loop's
inputs are the tape's (`Loop.Src`). -/
def Src (p : Server) : Prop := ∀ x w, R x w → Claims.ZKServer.Loop.Src p.L₀ p.src x w

/-- **L1's zero knowledge**: `Loop.zkOut_loopDeployed` at the deployment, its table side at the deployed tables
(`tables_dataE`, `complete_dataE`, `dummyReg_dataE`, `causal_dataE`, `tapeOK_dataE`), and its relation `DeferRel` from
`R`, the code facts and the sources. -/
theorem arg_ZK (p : Server) (hc : ZkCode p) (hs : Src p) : (arg p).ZK FlockDeploy.sha512 := by
  rw [sha512_eq_shaH]
  intro x w hR
  have hR' : R x w := hR
  obtain ⟨hF, hSc, hSv⟩ := hR'
  exact Claims.ZKServer.Loop.zkOut_loopDeployed p.vs p.t p.ht Claims.ZKServer.Loop.tables_dataE
    Claims.ZKServer.Loop.complete_dataE Claims.ZKServer.Loop.dummyReg_dataE Claims.ZKServer.Loop.causal_dataE
    hc.swCausal Claims.ZKServer.Loop.tapeOK_dataE p.L₀ p.src p.wk x w
    ⟨hF, hSv, hc.round0 x w hR, hSc, hs x w hR, hc.live x w hR, hc.fill x w hR⟩

/-- **L1's export at the deployment `p`**: `arg p` is knowledge sound and zero knowledge at SHA-512. -/
theorem exports (p : Server) (hc : ZkCode p) (hs : Src p) : Claims.SpotFlock.Exports FlockDeploy.sha512 (arg p) :=
  ⟨arg_KS p, arg_ZK p hc hs⟩

end Flock.Spot

/-! ## L3: the deployment -/

namespace SpotFlock

open Claims Claims.SamplingGeneric

/-- **The deployed sampled verifier as a `Code`**: the deployed Lean verifier's run (`SampledProofs.Run.run`), with its
Flock sessions' statements and verdicts read off the sessions. -/
noncomputable def code : SamplingDeploy.Code := sorry

/-- The roots the deployed verifier's runs show for a proof block's interior values. -/
noncomputable def iroots : Msg → Claims.Seen → ℕ → List Msg := sorry

/-- The two-stage design the deployed verifier reads off a statement. -/
noncomputable def two : Claims.Sampling.AuditStatement → Option Claims.Sampling.Design.Two := sorry

/-- The replay sample the deployed verifier draws from its coins. -/
noncomputable def rs : (x : Msg) → (Fin (code.V.k x) → FlockDeploy.Coin) → List ℕ := sorry

/-- The deployed sampled verifier as a compiled verifier over L1's proof system: `SamplingDeploy.compiled` at the
deployed record, which reads `arg p`'s shapes and witnesses and not its relation. -/
noncomputable def compiled (p : Flock.Spot.Server) : Compiled FlockDeploy.Coin (Flock.Spot.arg p).toArg :=
  { SamplingDeploy.compiled Flock.Spot.D code iroots with }

/-- **L3's export, one stage**: the deployed sampled verifier with its Flock sessions as a masked compiled verifier
over `arg p`, with the framing the deployed run gives: which history is which session's, the rounds before the
sessions, each session's rounds, the witness the reference prover gives it, and its tape. -/
noncomputable def deployed (p : Flock.Spot.Server) : ZkCompiled FlockDeploy.Coin (Flock.Spot.arg p) where
  toCompiled := compiled p
  step := sorry
  k₀ := sorry
  M := sorry
  n := sorry
  wit := sorry
  CB _ _ := (Flock.Spot.arg p).S.C sorry
  tapeOf := sorry

/-- **L3's export, two stages**: the deployed sampled verifier as a two-stage compiled verifier over `arg p`, at its
two-stage design and replay sample. -/
noncomputable def deployedTwo (p : Flock.Spot.Server) : CompiledTwo FlockDeploy.Coin (Flock.Spot.arg p).toArg where
  toCompiled := compiled p
  two := two
  replaySample := rs

/-- **L3's refinement, one stage**: `binds` is `SamplingDeploy.commit_binds`; `sessions` is
`SamplingDeploy.sessions_compiled` at `code`'s facts (`Code.DrawLaw`, `Code.Refuses`, `Code.InteriorFixed`,
`Code.DrawFixed`) and its sessions' runs, numbers and openings (`SessionRuns`, `SessionsIn`, `SessionOpens`), read at
`R`; `openCalls` is `SamplingDeploy.openCallsLe_compiled` at `RevealsLe`; `zkSessions` is the framing's five facts. -/
theorem refines (p : Flock.Spot.Server) : Claims.SpotFlock.Refines FlockDeploy.sha512 (deployed p) := by
  sorry

/-- **L3's refinement, two stages**: one compiled verifier, its binding and zero-knowledge contract `refines`'; the
two-stage contract is `SamplingDeploy.sessionsTwo_compiled` at `code`'s two-stage facts (`Code.ReplayDraws`,
`Code.SecondDraws`, `Code.RefusesTwo`, `Code.InteriorFixed`, `Code.DrawFixed`) and its sessions' runs, numbers and
openings at a replay block (`SessionRuns`, `SessionsIn`, `SessionOpensTwo`), read at `R`. -/
theorem refinesTwo (p : Flock.Spot.Server) :
    Claims.SpotFlock.RefinesTwo FlockDeploy.sha512 (deployedTwo p) (deployed p) where
  base := rfl
  binds := (refines p).binds
  sessions := sorry
  zkSessions := (refines p).zkSessions

/-! ## The deployed guarantees, each L2's theorem at L1's export and L3's -/

/-- **The deployed sampled verifier is knowledge sound at SHA-512**. -/
theorem ks (p : Flock.Spot.Server) : (deployed p).toCompiled.KS FlockDeploy.sha512 :=
  compilerKS _ _ _ _ (Flock.Spot.arg_KS p) (refines p).binds (refines p).sessions

/-- **The committed form**. -/
theorem ksReg (p : Flock.Spot.Server) : (deployed p).toCompiled.KSReg FlockDeploy.sha512 :=
  compilerKSReg _ _ _ _ (Flock.Spot.arg_KS p) (refines p).binds (refines p).sessions

/-- **The count form**. -/
theorem ksCount (p : Flock.Spot.Server) : (deployed p).toCompiled.KSCount FlockDeploy.sha512 :=
  compilerKSCount _ _ _ _ (Flock.Spot.arg_KS p) (refines p).binds (refines p).sessions

/-- **Bits**, with the opening checks' count. -/
theorem bits (p : Flock.Spot.Server) : (deployed p).toCompiled.Bits FlockDeploy.sha512 :=
  compilerBits _ _ _ _ (Flock.Spot.arg_KS p) (refines p).binds (refines p).sessions (refines p).openCalls

/-- **The deployed sampled verifier is zero knowledge with a public outcome**, relative to the published leaf layer, at
L1's code facts and sources. -/
theorem zk (p : Flock.Spot.Server) (hc : Flock.Spot.ZkCode p) (hs : Flock.Spot.Src p) :
    (deployed p).ZK FlockDeploy.sha512 :=
  compilerZK _ _ _ _ (Flock.Spot.arg_ZK p hc hs) (refines p).zkSessions

/-- **The deployed two-stage sampled verifier is knowledge sound at SHA-512**. -/
theorem ksTwo (p : Flock.Spot.Server) : (deployedTwo p).KS FlockDeploy.sha512 :=
  compilerKSTwo _ _ _ _ (Flock.Spot.arg_KS p) (refinesTwo p).binds (refinesTwo p).sessions

/-- **The deployed two-stage sampled verifier's exfiltration bound**. -/
theorem exfilTwo (p : Flock.Spot.Server) : (deployedTwo p).Exfil FlockDeploy.sha512 :=
  compilerExfilTwo _ _ _ _ (Flock.Spot.arg_KS p) (refinesTwo p).binds (refinesTwo p).sessions

end SpotFlock
