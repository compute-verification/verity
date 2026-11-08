module

public import Proofs.Flock.ZkAny.Finish
public import Proofs.Flock.ZkAny.RelayOut
public import Proofs.Flock.Claims.ZK.FrameZK
public import Proofs.Flock.Soundness.CROnly.Hiding
public import Proofs.Flock.Soundness.ZK.RealLeaves
public import Proofs.Flock.SessionLoop.Statement
public import Proofs.Flock.Claims.ZK.CausalExe
public import Proofs.Flock.Claims.ZK.Stopped
public import Proofs.Flock.Claims.ZK.OneTape
public import Proofs.Flock.Claims.ZK.ExeLayout
public import Definitions.Flock.ZK.AnyValues

/-!
# Flock's zero knowledge at any values, per outcome, at the session loop's own release and at the deployed held relay

`Flock.Guarantees.SessionLoopZKAny` states the session loop's release (`LeanRelease.ofLoop`, at the stops its own
frames' refusals give, `LeanRelease.rfOf`) as `Claims.ZKOut` at its `Finish` verdict, for every worker and any values
whose unswitched round encoding is causal (`RefSession.UnCausal`), at every relation at whose statements the session's
table witnesses are valid: no `Flock.R`, no premise that the workers' rows hold anywhere but where the release goes on
at its `Finish`, and no premise that their switched round encoding is causal. Before the `Finish` the release is the
unswitched prover's on both branches, and it splits only at the `Finish`: where it stops, the simulator is the refusing
prover's cut at the `Finish`; where it goes on, the honest prover's Goldreich–Kahan simulator, through the go prover
`RefSession.goP`.

At the deployed held relay (`Flock.Firewall.Session.relay` at `hold = true`, as `Relay.out`, each session held as
`core-firewall session` holds it), which sends the setup's control frames at their places and the model's frames at
theirs, the same holds per outcome (`FlockSoundness.ZK.PerOutcome`) at the pair of the verdict and how many frames left
(`Relay.relayLen`, at most the setup's public bound `Relay.relayB`), at the same error (`Relay.relay_zkOut_env`):
`ZKOutRelay`, which costs `relayB spec sts + 1` outcomes per verdict. Its simulator encodes the model's simulated
messages by the public encoding at the setup (`Relay.encOf`), so it reads nothing the witness fixes.

Premises: the session's Props at the loop's tables (`Tables`, `Complete`, `DummyReg`, `Causal`, `TapeOK`), `UnCausal`
at the values, and the transport's trusts at every statement of the relation (`LeanRelease.TrustsStop`: the link, live
and fill rounds at the unswitched prover, the `Finish`, and the rows where it goes on); at the relay, what its host
gives it (`RelayEnvAt`) and that what it read at each model slot is the model's message there (`RelayReads`). The
loop's stops, coin check, `Hello` and last round are theorems at `ofLoop` (`stopsAt_of_readsCoins`,
`stopsCausal_rfOf`, `checkIs_coinLayer`, `helloEq_of_helloIs`, `loop_last`), and hm96's hiding is at its gap at the
pinned key (`CROnly.hm96Hiding_gap`). Scope: the released bytes and their order, not their times.

* `LeanRelease.TrustsStop`: the transport's trusts at the unswitched prover, at every statement of a relation.
* `RelayEnvAt`, `RelayReads`, `ZKOutRelay`, `zkOutRelay_of_zkOut`: what the held relay reads from its host and at its
  model slots, `ZKOut` read at the deployed held relay, and that `ZKOut` gives it.
* `Loop.anySim`, `Loop.anyErr`: the session loop's simulator given the verdict and its error per outcome.
* `Loop.ZkAnyAt`: `ZKOut` and `ZKOutRelay` at one release of the loop, the conclusion of both guarantees.
* **`Flock.SecurityProofs.SessionLoopZKAny`**: `LeanRelease.zk_finish_out` at `ofLoop`, and `zkOutRelay_of_zkOut`.
* **`Flock.SecurityProofs.SessionLoopZKAnyDeployed`**: `SessionLoopZKAny` at the deployed tables (`Masked.dataE`) and
  the loop's own link and `Finish` (`LeanRelease.ofLoopLink`, `SessionLoopZKDeployed`'s release), where the tables' side
  holds. It is the deployed statement without `SessionLoopZKDeployed`'s valid witness: any values, no `Flock.R`, no
  `RowsOK`.

The ZK server never stops for good after failed sessions: a worker chooses its sessions' verdicts, so what they reveal is
bounded per outcome of each session (its `Finish` verdict, and at the held relay how many frames left), and is not
capped over a monitoring period.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **What the deployed held relay reads from its host, at the setup `(spec, sts)`**: at each of the protocol's
randomness `ω`, every request of `ins ω` carries the setup, and every barrier it waits at returns. -/
def RelayEnvAt {Ω : Type} (spec : _root_.Flock.SessionSpec) (sts : Array _root_.Flock.Stmt)
    (ins : Ω → List (ByteArray × Relay.Env)) : Prop :=
  ∀ ω, ∀ p ∈ ins ω, p.2.setup = .ok (spec, sts) ∧ ∀ why, (p.2.barrier (some why)).isSome

/-- **What the deployed held relay read at each model slot is the model's message there** (`Relay.EnvPre`): at each of
the protocol's randomness `ω`, the model's messages `model ω`, in order. -/
def RelayReads {Ω : Type} (spec : _root_.Flock.SessionSpec) (ins : Ω → List (ByteArray × Relay.Env))
    (model : Ω → List Msg) : Prop :=
  ∀ ω, Relay.EnvPre spec [] (Relay.relaySteps true (ins ω)) (model ω)

/-- **`ZKOut` read at the deployed held relay**: at every setup `(spec, sts)`, for `(x, w) ∈ R`, every verifier `V*`
of the protocol's `k(x)` rounds and tape `θ`, and the requests `ins ω` the relay reads at each of the protocol's
randomness `ω`, where the relay's host gives it the setup (`RelayEnvAt`) and what it read at each model slot is the
protocol's message there (`RelayReads`), the relay's release (`Relay.out`) is simulated per outcome at the pair of the
protocol's outcome and how many frames left (`Relay.relayLen`, at most `Relay.relayB spec sts`), by the simulator given
the protocol's outcome, its messages encoded at the setup (`Relay.encOf`), woven with the setup's control frames and
cut there, within `ε` in `ℝ≥0∞`. -/
def ZKOutRelay {W St : Type} (N Ω : Msg → Type) [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)] [∀ x, Fintype (Ω x)]
    [∀ x, Nonempty (Ω x)] (H : Msg → Msg) (k : Msg → ℕ) (P : W → (x : Msg) → Prover (N x × Ω x)) (R : Rel W)
    (out : Msg → List (Msg × Out) → St) (S : SimOut St) (ε : (x : Msg) → W → (Θ : Type) → VStar Θ → Θ → St → ℝ) :
    Prop :=
  ∀ (spec : _root_.Flock.SessionSpec) (sts : Array _root_.Flock.Stmt) (x : Msg) (w : W), R x w →
    ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = k x → ∀ ins : N x × Ω x → List (ByteArray × Relay.Env),
    RelayEnvAt spec sts ins → RelayReads spec ins (fun ω => (runStar H (P w x) Vs x ω θ).msgs.map (·.1)) →
    FlockSoundness.ZK.PerOutcome
      (fun ω => (out x (runStar H (P w x) Vs x ω θ).msgs, Relay.relayLen true (Relay.relayB spec sts) (ins ω)))
      (fun ω => Relay.out true {} {} (ins ω))
      (fun o (c : S.C x) => (Relay.weave (Relay.ctlOf (.ok (spec, sts))) (Relay.tmplOf (.ok (spec, sts)))
        (Relay.encOf spec sts (((S.given o.1).out H Vs θ x c).map (·.1)))).take o.2)
      (fun o => ENNReal.ofReal (ε x w Θ Vs θ o.1))

/-- **`ZKOut` gives `ZKOutRelay`** (`Relay.relay_zkOut_env`). -/
theorem zkOutRelay_of_zkOut {W St : Type} {N Ω : Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)]
    [∀ x, Fintype (Ω x)] [∀ x, Nonempty (Ω x)] {H : Msg → Msg} {k : Msg → ℕ} {P : W → (x : Msg) → Prover (N x × Ω x)}
    {R : Rel W} {out : Msg → List (Msg × Out) → St} {S : SimOut St}
    {ε : (x : Msg) → W → (Θ : Type) → VStar Θ → Θ → St → ℝ} (h : ZKOut N Ω H k P R out S ε) :
    ZKOutRelay N Ω H k P R out S ε :=
  fun spec sts _ _ hR _ Vs θ hk ins hE he =>
    Relay.relay_zkOut_env h hR Vs θ hk spec sts ins (fun ω p hp => (hE ω p hp).1) (fun ω p hp => (hE ω p hp).2) he

namespace LeanRelease

/-- **The transport's trusts at the unswitched prover, at every statement of `Rl`**: at each `(x, w) ∈ Rl`, the
ZK server's own link, the worker's frame rounds and the stand-in frame rounds are the unswitched prover's
(`LinkRoundEqStop`, `LiveRoundEqStop`, `FillRoundEqStop`), the ZK server's own `Finish` is the switched prover's last
message (`FinalRoundEq`), and the `Finish` goes on only where the workers' rows hold at the sampled pads
(`FinishRows`). -/
def TrustsStop (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (Rl : Rel Flock.Wit) : Prop :=
  ∀ x w, Rl x w → LinkRoundEqStop S vs L x w ∧ LiveRoundEqStop S vs L H wk x w ∧ FillRoundEqStop S vs L H wk x w ∧
    FinalRoundEq S vs L H wk x w ∧ FinishRows S vs L H wk x w

end LeanRelease

namespace Loop

/-- The session loop's session at the table data `T`, with the stops of its own frames' refusals. -/
noncomputable def anyStops (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit ((CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit → (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand →
      ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)) : RefSession :=
  (CoinLayer.session loopNr T).withStops
    (LeanRelease.rfOf (CoinLayer.session loopNr T) (LeanRelease.ofLoop T L₀ real) progOf freshOf readCoins)
    (LeanRelease.ofLoop T L₀ real).stop

/-- **The session loop's simulator given its `Finish` verdict**: where it goes on, the honest prover's Goldreich–Kahan
simulator at `t` repetitions; else the refusing prover's, cut at the `Finish` (`verdictSim`). Neither reads a worker's
values. -/
noncomputable def anySim (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit ((CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit → (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand →
      ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)) (t : ℕ) : SimOut Bool :=
  verdictSim
    (((anyStops T L₀ real).refP.withSim fun x _ c => (anyStops T L₀ real).coinModel x c).simulatorE t (t * 3 ^ t))
    ((anyStops T L₀ real).offP.simulatorE t (t * 3 ^ t)) (fun x => refK x + 2) (LeanRelease.ofLoop T L₀ real).stop

/-- **The error per outcome**, with `n_L(x)` the session's leaf count and `δ₁` hm96's hiding gap at the pinned key:
where the release goes on, `8·n_L·δ₁ + 24t·n_L·δ₁ + 2·3^{-t}` plus `4t·3^t` times the go prover's collision finder's
success; where it stops, `8·n_L·δ₁ + 8t·n_L·δ₁ + 2·3^{-t}` plus `4t·3^t` times the unswitched prover's. -/
noncomputable def anyErr (T : CoinLayer.TableData) (vs : (CoinLayer.session loopNr T).SwVals)
    (L₀ : LeanRelease Flock.Wit ((CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit → (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand →
      ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)) (t : ℕ) :
    (x : Msg) → Flock.Wit → (Θ : Type) → VStar Θ → Θ → Bool → ℝ :=
  fun x w _Θ Vs θ o =>
    if o then
      8 * (CoinLayer.session loopNr T).nLeaves x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
        24 * t * (CoinLayer.session loopNr T).nLeaves x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
        2 * (1 / 3) ^ t +
        4 * t * 3 ^ t * (Fork CoinOpen.shaH (((anyStops T L₀ real).goP vs).prover w) Vs θ x).Pr CoinOpen.shaH
    else
      8 * (CoinLayer.session loopNr T).nLeaves x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
        8 * t * (CoinLayer.session loopNr T).nLeaves x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
        2 * (1 / 3) ^ t +
        4 * t * 3 ^ t * (Fork CoinOpen.shaH (((anyStops T L₀ real).unP vs).prover w) Vs θ x).Pr CoinOpen.shaH

/-- **Zero knowledge at any values, per outcome, at one release of the session loop**: the release `ofLoop T L₀ real`
at the worker `wk` meets `Claims.ZKOut` at the statements of `Rl` against every `V*` of its `refK x + 2` rounds, at its
`Finish` verdict (`goesOnAt` at its last round), at `anySim` within `anyErr`, and so `ZKOutRelay` at the deployed held
relay. -/
def ZkAnyAt (T : CoinLayer.TableData) (vs : (CoinLayer.session loopNr T).SwVals)
    (L₀ : LeanRelease Flock.Wit ((CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit → (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand →
      ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (LeanRelease.ofLoop T L₀ real).toSrv.Worker) (Rl : Rel Flock.Wit) (t : ℕ) : Prop :=
  ZKOut (fun _ => (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand)
      (fun _ => (LeanRelease.ofLoop T L₀ real).toSrv.Chk) CoinOpen.shaH (fun x => refK x + 2)
      (fun w _ => (LeanRelease.ofLoop T L₀ real).toSrv.proverD (LeanRelease.ofLoop T L₀ real).fillOf
        (fun x => refK x + 2) wk w) Rl
      (fun x τ => goesOnAt (LeanRelease.ofLoop T L₀ real).stop (refK x + 1) τ) (anySim T L₀ real t)
      (anyErr T vs L₀ real t) ∧
    ZKOutRelay (fun _ => (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand)
      (fun _ => (LeanRelease.ofLoop T L₀ real).toSrv.Chk) CoinOpen.shaH (fun x => refK x + 2)
      (fun w _ => (LeanRelease.ofLoop T L₀ real).toSrv.proverD (LeanRelease.ofLoop T L₀ real).fillOf
        (fun x => refK x + 2) wk w) Rl
      (fun x τ => goesOnAt (LeanRelease.ofLoop T L₀ real).stop (refK x + 1) τ) (anySim T L₀ real t)
      (anyErr T vs L₀ real t)

end Loop

/-- **Flock's zero knowledge at any values, per outcome, at the session loop's own release and at the deployed held
relay.** At every table data `T`, values `vs` whose unswitched round encoding is causal, repetition count `0 < t` and
relation `Rl` at whose statements the session's table witnesses are valid, where the session meets the tables' side
(`Tables`, `DummyReg`, `Causal`, `TapeOK`), at every `L₀` (the release's inner rounds), tables `real` and worker `wk`
where the transport's trusts hold at the statements of `Rl` (`LeanRelease.TrustsStop`), the session loop's release
(`LeanRelease.ofLoop`) against every `V*` of its `refK x + 2` rounds meets `Claims.ZKOut` at its `Finish` verdict
(`goesOnAt` at its last round), at `Loop.anySim` within `Loop.anyErr`, and so `ZKOutRelay` at the deployed held relay.
-/
def _root_.Flock.Guarantees.SessionLoopZKAny : Prop :=
  ∀ (T : CoinLayer.TableData) (vs : (CoinLayer.session Loop.loopNr T).SwVals) (t : ℕ), 0 < t →
    ∀ (Rl : Rel Flock.Wit),
    (CoinLayer.session Loop.loopNr T).Tables → (CoinLayer.session Loop.loopNr T).Complete Rl →
    (CoinLayer.session Loop.loopNr T).DummyReg → (CoinLayer.session Loop.loopNr T).Causal →
    (CoinLayer.session Loop.loopNr T).UnCausal vs → (CoinLayer.session Loop.loopNr T).TapeOK →
    ∀ (L₀ : LeanRelease Flock.Wit
        ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand))
      (real : Msg → Flock.Wit →
        (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
        ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
      (wk : (LeanRelease.ofLoop T L₀ real).toSrv.Worker),
    LeanRelease.TrustsStop (CoinLayer.session Loop.loopNr T) vs (LeanRelease.ofLoop T L₀ real) CoinOpen.shaH wk Rl →
    Loop.ZkAnyAt T vs L₀ real wk Rl t

/-- **Flock's zero knowledge at any values, per outcome, at the deployed tables and the session loop's own link and
`Finish`**: `SessionLoopZKAny` at `Masked.dataE`, at `SessionLoopZKDeployed`'s release (`LeanRelease.ofLoopLink` at the
loop's own tables, `Loop.realOf src`), where the session's other Props hold. In place of `SessionLoopZKDeployed`'s valid
witness (`Flock.R`, through which the workers' rows hold at the values, `Loop.RowsOK`) and its switched causality, it is
at any values whose unswitched round encoding is causal (`UnCausal`), at every relation at whose statements the
session's table witnesses are valid, where the transport's trusts hold at those statements (`LeanRelease.TrustsStop`).
The ZK server never stops for good after failed sessions, so what the verdicts reveal is bounded per outcome of each
session, and is not capped over a monitoring period. -/
def _root_.Flock.Guarantees.SessionLoopZKAnyDeployed : Prop :=
  ∀ (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (t : ℕ), 0 < t →
    ∀ (Rl : Rel Flock.Wit), (CoinLayer.session Loop.loopNr Masked.dataE).Complete Rl →
    (CoinLayer.session Loop.loopNr Masked.dataE).UnCausal vs →
    ∀ (L₀ : LeanRelease Flock.Wit
        ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
      (src : Msg → Flock.Wit →
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
        ℕ → _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
      (wk : (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Worker),
    LeanRelease.TrustsStop (CoinLayer.session Loop.loopNr Masked.dataE) vs
      (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)) CoinOpen.shaH wk Rl →
    Loop.ZkAnyAt Masked.dataE vs (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) wk Rl t

end Claims.ZKServer

namespace Flock.SecurityProofs

open Claims Claims.ZKMal Claims.ZKServer
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **Flock's zero knowledge at any values, per outcome, at the session loop's own release and at the deployed held
relay**: `LeanRelease.zk_finish_out` at `ofLoop`, with the loop's stops, check, `Hello` and last round from its own
lemmas and hm96's hiding at its gap, and `zkOutRelay_of_zkOut`. -/
theorem SessionLoopZKAny : Flock.Guarantees.SessionLoopZKAny := by
  intro T vs t ht Rl hS hV hD hc huc hT L₀ real wk hTr
  have hRd : ∀ x, LeanRelease.ReadsCoins (CoinLayer.session Loop.loopNr T) (LeanRelease.ofLoop T L₀ real) Loop.progOf
      Loop.freshOf Loop.readCoins x := fun x =>
    ⟨LeanRelease.ofLoop_progReads T L₀ real x,
      LeanRelease.loop_coinsRead Loop.loopNr T (LeanRelease.withFrames T L₀ real) x⟩
  have hC : ∀ x, LeanRelease.CheckIs (CoinLayer.session Loop.loopNr T) (LeanRelease.ofLoop T L₀ real) CoinOpen.shaH x :=
    fun x => LeanRelease.checkIs_coinLayer Loop.loopNr T (LeanRelease.ofLoop T L₀ real) x
      (LeanRelease.loop_helloParses Loop.loopNr T (LeanRelease.withFrames T L₀ real) x)
      (LeanRelease.loop_reqsFollow T (LeanRelease.withFrames T L₀ real) x)
  have hZ := LeanRelease.zk_finish_out (CoinLayer.session Loop.loopNr T)
    (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) (LeanRelease.ofLoop T L₀ real) Loop.progOf Loop.freshOf
      Loop.readCoins) vs (LeanRelease.ofLoop T L₀ real) CoinOpen.shaH wk Rl (fun _ => True) t ht
    (FlockSoundness.CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash) ENNReal.ofReal_ne_top hS hV hD hc
    (LeanRelease.stopsCausal_rfOf _ (LeanRelease.ofLoop T L₀ real) Loop.progOf Loop.freshOf Loop.readCoins) huc
    (fun x _ => hT x) (fun _ _ _ => trivial)
    (fun x => LeanRelease.loop_last Loop.loopNr T (LeanRelease.withFrames T L₀ real) x)
    (fun x => LeanRelease.stopsAt_of_readsCoins _ (LeanRelease.ofLoop T L₀ real) CoinOpen.shaH Loop.progOf Loop.freshOf
      Loop.readCoins x (hRd x) (hC x)) hC
    (fun x => LeanRelease.helloEq_of_helloIs Loop.loopNr T (LeanRelease.ofLoop T L₀ real) x
      (LeanRelease.loop_helloIs Loop.loopNr T (LeanRelease.withFrames T L₀ real) x))
    (fun x w h => (hTr x w h).1) (fun x w h => (hTr x w h).2.1) (fun x w h => (hTr x w h).2.2.1)
    (fun x w h => (hTr x w h).2.2.2.1) (fun x w h => (hTr x w h).2.2.2.2)
  rw [ENNReal.toReal_ofReal (FlockSoundness.CROnly.hidingGap_nonneg ZkView.My ZkView.saltHash)] at hZ
  unfold Loop.ZkAnyAt
  exact ⟨hZ, zkOutRelay_of_zkOut hZ⟩

/-- **Flock's zero knowledge at any values, per outcome, at the deployed tables and the session loop's own link and
`Finish`**: `SessionLoopZKAny` at `Masked.dataE` and `withFiles (withLink L₀ src) src`, where every table is one it sets
up (`Masked.tables`), the dummy's table witnesses carry the public region data (`Masked.dummyReg`), the stopped plan is
causal (`causal_encStop` at `causalExe`) and the tape is read with fibers of one size (`fibersEq_ofLayout`). -/
theorem SessionLoopZKAnyDeployed : Flock.Guarantees.SessionLoopZKAnyDeployed :=
  fun vs t ht Rl hV huc L₀ src wk hTr =>
    SessionLoopZKAny Masked.dataE vs t ht Rl (Masked.tables _ _ _ _) hV (Masked.dummyReg _ _ _ _)
      (Masked.RoundPlan.causal (Masked.causal_encStop (cz := Masked.czExe Masked.PE) (enc := Masked.encExe Masked.PE)
        (fin := Masked.finExe Masked.PE) Masked.causalExe) _ _) huc
      ((CoinLayer.session Loop.loopNr Masked.dataE).tapeOKOn_iff.1
        ((CoinLayer.session Loop.loopNr Masked.dataE).tapeOKOn_of_tape _ fun x _ =>
          Masked.fibersEq_ofLayout (Masked.TapeLayout.exe Masked.PE) Masked.exeTape x))
      (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) wk hTr

end Flock.SecurityProofs
