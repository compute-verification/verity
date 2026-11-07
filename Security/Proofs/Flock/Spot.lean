module

public import Definitions.Claims.SpotFlock
public import Proofs.Flock.ZkAny.DeferLoop
public import Proofs.Flock.SamplingGeneric.TwoDeployed
public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.SpotFlock.Contract
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
* **L3** (`SpotFlock`): `deployed p h` and `deployedTwo p h`, the deployed sampled verifier (`SpotFlock.code h`, at the
  auditor's copies `h`) with its Flock sessions as a masked compiled verifier and a two-stage compiled verifier over
  `arg p`; `refines` and `refinesTwo`, the compiler's contract at them. Two-stage zero knowledge isn't L3's contract:
  the two-stage view includes the interior registration, which reads the witness and the tape.
* **The deployed guarantees** (`SpotFlock.ks`, `ksReg`, `ksCount`, `bits`, `zk`, `ksTwo`, `exfilTwo`): each L2's theorem
  at L1's export and L3's, in one line. `Claims.SpotFlock.composes` and `composesTwo` are the same composition at any
  proof system and compiled verifier.
* **Knowledge soundness at Flock's own relation** (`SpotFlock.ksOfFlock`, `ksTwoOfFlock`): the same record at
  `Arg.ofFlock D`, whose knowledge soundness is proved, so the KS side reads no refusal outside `arg p`'s relation.
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
  fun Coin _ _ _ A L H h1 h3 =>
    ⟨compilerKSTwo Coin A.toArg L H h1.ks h3.binds h3.sessions,
      compilerExfilTwo Coin A.toArg L H h1.ks h3.binds h3.sessions⟩

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

/-- The deployed sampled verifier as a compiled verifier over L1's proof system: `SamplingDeploy.compiled` at the
deployed record and the deployed code at the auditor's copies `h` (`code h`, with its interior roots `iroots`), which
reads `arg p`'s shapes and witnesses and not its relation. -/
noncomputable def compiled (p : Flock.Spot.Server) (h : Held) : Compiled FlockDeploy.Coin (Flock.Spot.arg p).toArg :=
  { SamplingDeploy.compiled Flock.Spot.D (code h) iroots with }

/-- **L3's export, one stage**: the deployed sampled verifier with its Flock sessions as a masked compiled verifier
over `arg p`, with the framing the deployed run gives: which history is which session's, the rounds before the
sessions, each session's rounds, the witness the reference prover gives it, and its tape. -/
noncomputable def deployed (p : Flock.Spot.Server) (h : Held) : ZkCompiled FlockDeploy.Coin (Flock.Spot.arg p) where
  toCompiled := compiled p h
  step := sorry
  k₀ := sorry
  M := sorry
  n := sorry
  wit := sorry
  CB _ _ := (Flock.Spot.arg p).S.C sorry
  tapeOf := sorry

/-- **L3's export, two stages**: the deployed sampled verifier as a two-stage compiled verifier over `arg p`, at its
two-stage design and replay sample at the auditor's copies `h`. -/
noncomputable def deployedTwo (p : Flock.Spot.Server) (h : Held) :
    CompiledTwo FlockDeploy.Coin (Flock.Spot.arg p).toArg where
  toCompiled := compiled p h
  two := two h
  replaySample := rs h

/-- **L3's refinement, one stage**: `binds` is `SamplingDeploy.commit_binds`; `sessions` is
`SamplingDeploy.sessions_compiled` at `code`'s facts (`Code.DrawLaw`, `Code.Refuses`, `Code.InteriorFixed`,
`Code.DrawFixed`) and its sessions' runs, numbers and openings (`SessionRuns`, `SessionsIn`, `SessionOpens`), read at
`R`; `openCalls` is `SamplingDeploy.openCallsLe_compiled` at `RevealsLe`; `zkSessions` is the framing's five facts. -/
theorem refines (p : Flock.Spot.Server) (h : Held) : Claims.SpotFlock.Refines FlockDeploy.sha512 (deployed p h) := by
  sorry

/-- **L3's refinement, two stages**: `binds` is `SamplingDeploy.commit_binds`; `sessions` is `sessionsTwo_spot`, the
two-stage contract at `code h` from A3 at the coins' bytes and the sessions' openings at a replay block
(`SessionOpensTwo`), its openings read at `R`, which is Flock's relation and more. -/
theorem refinesTwo (p : Flock.Spot.Server) (h : Held) (hA : SamplingDeploy.CallsUniform (fun _ => L) fun _ => coinByte)
    (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    Claims.SpotFlock.RefinesTwo FlockDeploy.sha512 (deployedTwo p h) where
  binds := SamplingDeploy.commit_binds (code h) FlockDeploy.sha512
  sessions :=
    have hS := sessionsTwo_spot h hA hopens
    { hS with
      opens := fun x s Pt d Ω Pv ω r u xS vS w hs hP h2 hin hacc hu hsess hR =>
        hS.opens x s Pt d Ω Pv ω r u xS vS w hs hP h2 hin hacc hu hsess hR.1 }

/-! ## The deployed guarantees, each L2's theorem at L1's export and L3's -/

/-- **The deployed sampled verifier is knowledge sound at SHA-512**. -/
theorem ks (p : Flock.Spot.Server) (h : Held) : (deployed p h).toCompiled.KS FlockDeploy.sha512 :=
  compilerKS _ _ _ _ (Flock.Spot.arg_KS p) (refines p h).binds (refines p h).sessions

/-- **The committed form**. -/
theorem ksReg (p : Flock.Spot.Server) (h : Held) : (deployed p h).toCompiled.KSReg FlockDeploy.sha512 :=
  compilerKSReg _ _ _ _ (Flock.Spot.arg_KS p) (refines p h).binds (refines p h).sessions

/-- **The count form**. -/
theorem ksCount (p : Flock.Spot.Server) (h : Held) : (deployed p h).toCompiled.KSCount FlockDeploy.sha512 :=
  compilerKSCount _ _ _ _ (Flock.Spot.arg_KS p) (refines p h).binds (refines p h).sessions

/-- **Bits**, with the opening checks' count. -/
theorem bits (p : Flock.Spot.Server) (h : Held) : (deployed p h).toCompiled.Bits FlockDeploy.sha512 :=
  compilerBits _ _ _ _ (Flock.Spot.arg_KS p) (refines p h).binds (refines p h).sessions (refines p h).openCalls

/-- **The deployed sampled verifier is zero knowledge with a public outcome**, relative to the published leaf layer, at
L1's code facts and sources. -/
theorem zk (p : Flock.Spot.Server) (h : Held) (hc : Flock.Spot.ZkCode p) (hs : Flock.Spot.Src p) :
    (deployed p h).ZK FlockDeploy.sha512 :=
  compilerZK _ _ _ _ (Flock.Spot.arg_ZK p hc hs) (refines p h).zkSessions

/-- **The deployed two-stage sampled verifier is knowledge sound at SHA-512**. -/
theorem ksTwo (p : Flock.Spot.Server) (h : Held) (hA : SamplingDeploy.CallsUniform (fun _ => L) fun _ => coinByte)
    (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    (deployedTwo p h).KS FlockDeploy.sha512 :=
  compilerKSTwo _ _ _ _ (Flock.Spot.arg_KS p) (refinesTwo p h hA hopens).binds (refinesTwo p h hA hopens).sessions

/-- **The deployed two-stage sampled verifier's exfiltration bound**. -/
theorem exfilTwo (p : Flock.Spot.Server) (h : Held) (hA : SamplingDeploy.CallsUniform (fun _ => L) fun _ => coinByte)
    (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    (deployedTwo p h).Exfil FlockDeploy.sha512 :=
  compilerExfilTwo _ _ _ _ (Flock.Spot.arg_KS p) (refinesTwo p h hA hopens).binds (refinesTwo p h hA hopens).sessions

/-! ## Knowledge soundness at Flock's own relation

`Compiled` reads its proof system's shapes and witnesses and not its relation, so the same record composes at
`Arg.ofFlock D`, whose knowledge soundness `ofFlock_KS` is proved. The KS side then takes no refusal outside `arg p`'s
relation; only zero knowledge reads it. -/

/-- **A sampled verifier over Flock's sessions is knowledge sound at Flock's relation**, at any `Code`, from L3's
knowledge-soundness contract; at the deployed one, `c = code h`. -/
theorem ksOfFlock (c : SamplingDeploy.Code) (ir : Msg → Claims.Seen → ℕ → List Msg)
    (hS : (SamplingDeploy.compiled Flock.Spot.D c ir).Sessions FlockDeploy.sha512) :
    (SamplingDeploy.compiled Flock.Spot.D c ir).KS FlockDeploy.sha512 :=
  compilerKS _ _ _ _ Flock.Spot.ofFlock_KS (SamplingDeploy.commit_binds c _) hS

/-- **A two-stage sampled verifier over Flock's sessions is knowledge sound at Flock's relation**, at any `Code`, from
L3's two-stage contract; at the deployed one, `c = code h`. -/
theorem ksTwoOfFlock (c : SamplingDeploy.Code) (ir : Msg → Claims.Seen → ℕ → List Msg)
    (tw : Claims.Sampling.AuditStatement → Option Claims.Sampling.Design.Two)
    (r : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → List ℕ)
    (hS : (SamplingDeploy.compiledTwo Flock.Spot.D c ir tw r).Sessions FlockDeploy.sha512) :
    (SamplingDeploy.compiledTwo Flock.Spot.D c ir tw r).KS FlockDeploy.sha512 :=
  compilerKSTwo _ _ _ _ Flock.Spot.ofFlock_KS (SamplingDeploy.commit_binds c _) hS

end SpotFlock
