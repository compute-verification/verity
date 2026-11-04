module

public import Proofs.Flock.ZkAny.Finish
public import Proofs.Flock.Claims.ZK.FrameZK
public import Proofs.Flock.Claims.Sampling.Draws
public import Proofs.Flock.Soundness.CROnly.Hiding
public import Proofs.Flock.Soundness.ZK.RealLeaves
public import Proofs.Core.Game.Basic
public import Definitions.Flock.ZK.AnyValues

/-!
# C-Flock's zero knowledge at any values, per outcome, at the session loop's own release and at a relay

`Flock.Guarantees.SessionLoopZKAny` states the session loop's release (`LeanRelease.ofLoop`, at the stops its own frames'
refusals give, `LeanRelease.rfOf`) as `Claims.ZKOut` at its `Finish` verdict, for every worker and any values whose
unswitched round encoding is causal (`RefSession.UnCausal`), at every relation at whose statements the session's table
witnesses are valid: no `Flock.R`, no premise that the workers' rows hold anywhere but where the release goes on at its
`Finish`, and no premise that their switched round encoding is causal. Before the `Finish` the release is the unswitched
prover's on both branches, and it splits only at the `Finish`: where it stops, the simulator is the refusing prover's cut
at the `Finish`; where it goes on, the honest prover's Goldreich–Kahan simulator, through the go prover `RefSession.goP`.

At a relay that sends the model's release woven with frames of its own, the same holds per outcome
(`FlockSoundness.ZK.PerOutcome`) at the pair of the verdict and a public outcome `pos ∈ Q` (at the deployed relay, where
the release ended and what the deployed `Finish` reveals on a stop), wherever the relay's bytes are a public function
`g`, fixed before the witness, of `pos` and the model's release (`RelayLift`): `ZKOutRelay`, which costs `2·|Q|`
outcomes.

Premises: the session's Props at the loop's tables (`Tables`, `Complete`, `DummyReg`, `Causal`, `TapeOK`), `UnCausal`
at the values, and the transport's trusts at every statement of the relation (`LeanRelease.TrustsStop`: the link, live
and fill rounds at the unswitched prover, the `Finish`, and the rows where it goes on). The loop's stops, coin check,
`Hello` and last round are theorems at `ofLoop` (`stopsAt_of_readsCoins`, `stopsCausal_rfOf`, `checkIs_coinLayer`,
`helloEq_of_helloIs`, `loop_last`), and hm96's hiding is at its gap at the pinned key (`CROnly.hm96Hiding_gap`). Scope:
the released bytes and their order, not their times.

* `LeanRelease.TrustsStop`: the transport's trusts at the unswitched prover, at every statement of a relation.
* `RelayLift`, `ZKOutRelay`, `zkOutRelay_of_zkOut`: a relay's bytes as a public function of a public outcome and the
  model's release, `ZKOut` read at such a relay, and that `ZKOut` gives it.
* `Loop.anySim`, `Loop.anyErr`: the session loop's simulator given the verdict and its error per outcome.
* **`Flock.SecurityProofs.SessionLoopZKAny`**: `LeanRelease.zk_finish_out` at `ofLoop`, and `zkOutRelay_of_zkOut`.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **A relay's bytes are a public function of a public outcome and the model's release**: at each of the release's
randomness `ω`, the bytes `rel ω` the relay releases are `g` at the outcome `pos ω` of the model's release `model ω`. At
the deployed relay, which sends the setup's control frames at their fixed places and the model's frames, encoded, at
theirs, over the schedule cut where the release ended, `pos` is that cut and `g` weaves the control frames into the
encoded model frames there. -/
def RelayLift {Ω B Q : Type} (rel : Ω → B) (pos : Ω → Q) (g : Q → List (Msg × Out) → B)
    (model : Ω → List (Msg × Out)) : Prop :=
  ∀ ω, rel ω = g (pos ω) (model ω)

/-- **`ZKOut` read at a relay**: for `(x, w) ∈ R`, every verifier `V*` of the protocol's `k(x)` rounds and tape `θ`,
and every public `g`, fixed before the witness, wherever the bytes `rel` a relay releases are `g x` at a public outcome
`pos` of the protocol's transcript (`RelayLift`), they are simulated per outcome at the pair of the protocol's outcome
and `pos`, by the simulator given the protocol's outcome read through `g x` at `pos`, within `ε` in `ℝ≥0∞`. -/
def ZKOutRelay {W St : Type} (N Ω : Msg → Type) [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)] [∀ x, Fintype (Ω x)]
    [∀ x, Nonempty (Ω x)] (H : Msg → Msg) (k : Msg → ℕ) (P : W → (x : Msg) → Prover (N x × Ω x)) (R : Rel W)
    (out : Msg → List (Msg × Out) → St) (S : SimOut St) (ε : (x : Msg) → W → (Θ : Type) → VStar Θ → Θ → St → ℝ) :
    Prop :=
  ∀ {B Q : Type} (g : Msg → Q → List (Msg × Out) → B) (x : Msg) (w : W), R x w →
    ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = k x → ∀ (rel : N x × Ω x → B) (pos : N x × Ω x → Q),
      RelayLift rel pos (g x) (fun ω => (runStar H (P w x) Vs x ω θ).msgs) →
      FlockSoundness.ZK.PerOutcome (fun ω => (out x (runStar H (P w x) Vs x ω θ).msgs, pos ω)) rel
        (fun o (c : S.C x) => g x o.2 ((S.given o.1).out H Vs θ x c)) (fun o => ENNReal.ofReal (ε x w Θ Vs θ o.1))

/-- A bound in `Claims.Pr` is one in `prCoin`, its error read in `ℝ≥0∞`. -/
private theorem prCoin_le_of_Pr {α β : Type} [Fintype α] [Nonempty α] [Fintype β] [Nonempty β] {p : α → Prop}
    {q : β → Prop} {e : ℝ} (h : Claims.Pr p ≤ Claims.Pr q + e) :
    Verity.Game.prCoin p ≤ Verity.Game.prCoin q + ENNReal.ofReal e :=
  calc Verity.Game.prCoin p = ENNReal.ofReal (Claims.Pr p) := by
        rw [Claims.Sampling.Pr_eq_toReal_prCoin, ENNReal.ofReal_toReal (FlockSoundness.ZK.prCoin_ne_top p)]
    _ ≤ ENNReal.ofReal (Claims.Pr q + e) := ENNReal.ofReal_le_ofReal h
    _ ≤ ENNReal.ofReal (Claims.Pr q) + ENNReal.ofReal e := ENNReal.ofReal_add_le
    _ = Verity.Game.prCoin q + ENNReal.ofReal e := by
        rw [Claims.Sampling.Pr_eq_toReal_prCoin, ENNReal.ofReal_toReal (FlockSoundness.ZK.prCoin_ne_top q)]

/-- **`ZKOut` gives `ZKOutRelay`**: the relay's event at an outcome pair is the protocol's at its outcome, read through
`g x` at the second. -/
theorem zkOutRelay_of_zkOut {W St : Type} {N Ω : Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)]
    [∀ x, Fintype (Ω x)] [∀ x, Nonempty (Ω x)] {H : Msg → Msg} {k : Msg → ℕ} {P : W → (x : Msg) → Prover (N x × Ω x)}
    {R : Rel W} {out : Msg → List (Msg × Out) → St} {S : SimOut St}
    {ε : (x : Msg) → W → (Θ : Type) → VStar Θ → Θ → St → ℝ} (h : ZKOut N Ω H k P R out S ε) :
    ZKOutRelay N Ω H k P R out S ε := by
  intro B Q g x w hR Θ Vs θ hk rel pos hfr
  rintro ⟨o, q⟩ E
  have hm : Verity.Game.prCoin (fun ω : N x × Ω x =>
        (out x (runStar H (P w x) Vs x ω θ).msgs, pos ω) = (o, q) ∧ E (rel ω)) ≤
      Verity.Game.prCoin (fun ω : N x × Ω x =>
        out x (runStar H (P w x) Vs x ω θ).msgs = o ∧ E (g x q (runStar H (P w x) Vs x ω θ).msgs)) :=
    Verity.Game.prCoin_mono fun ω hω => by
      obtain ⟨h1, h2⟩ := Prod.mk.inj hω.1
      refine ⟨h1, ?_⟩
      rw [← h2, ← hfr ω]
      exact hω.2
  exact hm.trans (prCoin_le_of_Pr (h x w hR Θ Vs θ hk o fun τ => E (g x q τ)))

namespace LeanRelease

/-- **The transport's trusts at the unswitched prover, at every statement of `Rl`**: at each `(x, w) ∈ Rl`, the
firewall's own link, the worker's frame rounds and the stand-in frame rounds are the unswitched prover's
(`LinkRoundEqStop`, `LiveRoundEqStop`, `FillRoundEqStop`), the firewall's own `Finish` is the switched prover's last
message (`FinalRoundEq`), and the `Finish` goes on only where the workers' rows hold at the drawn pads (`FinishRows`). -/
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

end Loop

/-- **C-Flock's zero knowledge at any values, per outcome, at the session loop's own release and at a relay.** At every
table data `T`, values `vs` whose unswitched round encoding is causal, repetition count `0 < t` and relation `Rl` at
whose statements the session's table witnesses are valid, where the session meets the tables' side (`Tables`,
`DummyReg`, `Causal`, `TapeOK`), at every `L₀` (the release's inner rounds), tables `real` and worker `wk` where the
transport's trusts hold at the statements of `Rl` (`LeanRelease.TrustsStop`), the session loop's release
(`LeanRelease.ofLoop`) against every `V*` of its `refK x + 2` rounds meets `Claims.ZKOut` at its `Finish` verdict
(`goesOnAt` at its last round), at `Loop.anySim` within `Loop.anyErr`, and so `ZKOutRelay` at a relay. -/
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
    ZKOut (fun _ => (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand)
        (fun _ => (LeanRelease.ofLoop T L₀ real).toSrv.Chk) CoinOpen.shaH (fun x => refK x + 2)
        (fun w _ => (LeanRelease.ofLoop T L₀ real).toSrv.proverD (LeanRelease.ofLoop T L₀ real).fillOf
          (fun x => refK x + 2) wk w) Rl
        (fun x τ => goesOnAt (LeanRelease.ofLoop T L₀ real).stop (refK x + 1) τ) (Loop.anySim T L₀ real t)
        (Loop.anyErr T vs L₀ real t) ∧
      ZKOutRelay (fun _ => (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand)
        (fun _ => (LeanRelease.ofLoop T L₀ real).toSrv.Chk) CoinOpen.shaH (fun x => refK x + 2)
        (fun w _ => (LeanRelease.ofLoop T L₀ real).toSrv.proverD (LeanRelease.ofLoop T L₀ real).fillOf
          (fun x => refK x + 2) wk w) Rl
        (fun x τ => goesOnAt (LeanRelease.ofLoop T L₀ real).stop (refK x + 1) τ) (Loop.anySim T L₀ real t)
        (Loop.anyErr T vs L₀ real t)

end Claims.ZKServer

namespace Flock.SecurityProofs

open Claims Claims.ZKMal Claims.ZKServer
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **C-Flock's zero knowledge at any values, per outcome, at the session loop's own release and at a relay**:
`LeanRelease.zk_finish_out` at `ofLoop`, with the loop's stops, check, `Hello` and last round from its own lemmas and
hm96's hiding at its gap, and `zkOutRelay_of_zkOut`. -/
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
  exact ⟨hZ, zkOutRelay_of_zkOut hZ⟩

end Flock.SecurityProofs
