import Proofs.Flock.Claims.ZK.FrameAdmits

/-!
# C-Flock's zero knowledge at the session loop's own frames (trusted)

`SessionLoopZK` is the session loop's zero-knowledge bound (Stage B, `note:20261022T0001Z-report-zkfin-stageb`) as a
guarantee: at the deployed relation (`Flock.Readable.relation`), at the release whose transport state is the loop's own
frames (`LeanRelease.ofLoop`, each rep's frame driven on `Drive.reqs` from the decoded coins), at every statement `x` the
loop serves (`Loop.Serves`: its `Hello` leaves at `x`'s setup, which it refuses where the setup's sizes are past its
frames', `Session.sizesOk`), and at the hiding gap of hm96-sha512's pinned key (`hidingGap My saltHash`, with no hiding
premise). It concludes that the deployed frame's state is the loop's but for `stopped` (`ProgChecks`) and the bound.

What it takes of the loop beyond the code is named, each an obligation:

* `LoopTabsReal`: the tables `real` answers are the session loop's own (`Count.RealTabs`: `Finish`'s `level0` request,
  `tabOfLevel0` at a level 0 `Level0.ofDraws` drew).
* `LoopInnerRound0`: `L₀`'s round at the inner proof's round 0 is the firewall's own at some 16-byte `final_c'` (Rust's
  firewall sends that round, `check_self`, at the inner proof's label).
* `TablesAdmitUnrefused`, on the tables `T` alone: where no coin fault shows (`Masked.Refused`), every table admits the
  coins. `SessionLoopZK` takes it; at the deployed tables it is proved (`LeanRelease.tablesAdmitUnrefused_dataE`), so
  `SessionLoopZKDeployed` (`SessionLoopZKAt Masked.dataE`) does not.

The transport's other trusts (`LeanRelease.LinkRoundEq`, `FinishIs`, `FrameUnrefused`, `LiveRoundEq`, `FillRoundEq`) and
the tables' side (`CoinLayer` `Tables`, `Complete`, `DummyReg`, `Causal`, `SwCausal`, `TapeOK`) are the statement's
premises, with `FinishIs` and `FrameUnrefused` in place of the `FinalRoundEq` `Flock.zk_server_lean_loop` takes
(`LeanRelease.finalRoundEq_ofLoop`). The proofs are `Flock.SecurityProofs.SessionLoopZK` and `SessionLoopZKDeployed`
(`Proofs.Flock.SessionLoop`), `Flock.zk_server_lean_ofLoop` at the gap.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker labelFrame)
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

/-- Obligation (the loop's tables): at every tape `ρ`, the tables `real` answers at `x` and the witness `w` are the
session loop's own at `x`'s setup (`Count.RealTabs`: `Finish`'s `level0` request, `tabOfLevel0` at a level 0
`Level0.ofDraws` drew). -/
def _root_.Flock.Guarantees.LoopTabsReal (T : CoinLayer.TableData)
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ρ : (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand)
    (s0 : _) (spec : SessionSpec) (sts : Array Stmt),
    FlockDeploy.setupOf true x = some (s0, spec, sts) → Count.RealTabs sts (real x w ρ)

/-- Obligation (the inner proof's round 0): where history `h` stands at a rep stream `k`'s first round past its
scheduled rounds `rsched` (the opening's, on `real`'s table and salts), `L₀`'s round there is the firewall's own at some
16-byte `final_c'`: the inner proof's label frame, then `final_c'` between its headers (Rust's firewall sends that round,
`check_self`). -/
def _root_.Flock.Guarantees.LoopInnerRound0 (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ρ : (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand)
    (h : List (Msg × Msg)) (vals : _) (s0 : _) (spec : SessionSpec) (sts : Array Stmt) (q k j : ℕ)
    (tr : _root_.Flock.Firewall.Finish.Tab) (hr : Array ByteArray) (rsched : Array (ByteArray × ℕ)),
    FlockDeploy.setupOf true x = some (s0, spec, sts) → h.length = q + 2 →
    CoinLayer.locate (Loop.roundsOf spec sts) 0 q = (k, j + 1) → real x w ρ k = .ok (tr, hr) →
    Loop.schedSoFar tr (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]! (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]!
        (Loop.wordsAt x (h.tail.map fun p => CoinLayer.rd p.2) 0))) hr
      ((Array.range (j + 1)).map
        fun i => Loop.wordsAt x (h.tail.map fun p => CoinLayer.rd p.2) (q - (j + 1) + i + 1)) = .ok rsched →
    rsched.size = j + 1 →
    ∃ w' : ByteArray, w'.size = 16 ∧ L₀.scheduled x w ρ h vals =
      (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1).data.toList

/-- **C-Flock's zero knowledge at the session loop's own frames, at the tables `T`.** At the deployed relation, where
`T`'s session meets the tables' side (`Tables`, `Complete` at the relation, `DummyReg`, `Causal`, `SwCausal` at `vs`,
`TapeOK`), at every repetition count `0 < t`, every `L₀` and `real` (the release's inner rounds and the rep's tables) and
worker `wk`, at every statement `x` the loop serves (`Loop.Serves`) with a witness `w` under the relation, where the
loop's obligations (`LoopTabsReal`, `LoopInnerRound0`) and the transport's round trusts hold: the deployed frame's state
is the loop's but for `stopped` (`ProgChecks`), and against every `V*` of one more round than the verifier, the
probability of any event of the real view at `ofLoop` is at most twice the simulator's, plus
$$8\,n_L(x)\,\delta_1 + 8\,t\,n_L(x)\,\delta_1 + 2 \cdot 3^{-t}$$
with `n_L(x)` the session's leaf count and `δ₁` the hm96 salted-leaf hiding gap at the pinned key, plus `4·t·3^t` times
the fork probability. -/
def _root_.Flock.Guarantees.SessionLoopZKAt (T : CoinLayer.TableData) : Prop :=
  ∀ (vs : (CoinLayer.session Loop.loopNr T).SwVals) (t : ℕ), 0 < t →
    (CoinLayer.session Loop.loopNr T).Tables →
    (CoinLayer.session Loop.loopNr T).Complete
      (Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse) →
    (CoinLayer.session Loop.loopNr T).DummyReg → (CoinLayer.session Loop.loopNr T).Causal →
    (CoinLayer.session Loop.loopNr T).SwCausal vs → (CoinLayer.session Loop.loopNr T).TapeOK →
    ∀ (L₀ : LeanRelease Flock.Wit
        ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand))
      (real : Msg → Flock.Wit →
        (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
        ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
      (wk : (LeanRelease.ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit),
    Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w → Loop.Serves x →
    _root_.Flock.Guarantees.LoopTabsReal T real x w → _root_.Flock.Guarantees.LoopInnerRound0 T L₀ real x w →
    LeanRelease.LinkRoundEq (CoinLayer.session Loop.loopNr T) vs (LeanRelease.ofLoop T L₀ real) x w →
    LeanRelease.FinishIs (CoinLayer.session Loop.loopNr T) vs (LeanRelease.ofLoop T L₀ real) CoinOpen.shaH wk x w →
    LeanRelease.FrameUnrefused Loop.loopNr T (LeanRelease.ofLoop T L₀ real) CoinOpen.shaH wk x w →
    LeanRelease.LiveRoundEq (CoinLayer.session Loop.loopNr T) vs (LeanRelease.ofLoop T L₀ real) CoinOpen.shaH wk x w →
    LeanRelease.FillRoundEq (CoinLayer.session Loop.loopNr T) vs (LeanRelease.ofLoop T L₀ real) CoinOpen.shaH wk x w →
    ∀ {Θ : Type} (Vs : VStar Θ) (θ : Θ),
    Vs.k x = (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1 →
    ∀ (Ev : List ((Msg × Out) × ℕ) → Prop),
    (LeanRelease.ofLoop T L₀ real).ProgChecks x w ∧
    Pr (fun ω : ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand) ×
          (LeanRelease.ofLoop T L₀ real).toSrv.Chk =>
        Ev ((LeanRelease.ofLoop T L₀ real).toSrv.timed x
          (runStar CoinOpen.shaH ((LeanRelease.ofLoop T L₀ real).toSrv.proverD (LeanRelease.ofLoop T L₀ real).fillOf
            (fun x => (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1) wk w) Vs x ω θ).msgs)) ≤
      2 * (Pr (fun c : (leaky1 ((((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) (LeanRelease.ofLoop T L₀ real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoop T L₀ real).stop).refP.withSim fun x _ c =>
              ((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) (LeanRelease.ofLoop T L₀ real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoop T L₀ real).stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1)
            (LeanRelease.ofLoop T L₀ real).toSrv.stop).C x =>
          Ev ((LeanRelease.ofLoop T L₀ real).toSrv.timed x ((leaky1 ((((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) (LeanRelease.ofLoop T L₀ real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoop T L₀ real).stop).refP.withSim fun x _ c =>
              ((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) (LeanRelease.ofLoop T L₀ real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoop T L₀ real).stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1)
            (LeanRelease.ofLoop T L₀ real).toSrv.stop).out CoinOpen.shaH Vs θ x c))) +
          (8 * ((CoinLayer.session Loop.loopNr T).withStops (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T)
              (LeanRelease.ofLoop T L₀ real) Loop.progOf Loop.freshOf Loop.readCoins)
              (LeanRelease.ofLoop T L₀ real).stop).nLeaves x *
              FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
            8 * t * ((CoinLayer.session Loop.loopNr T).withStops (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T)
              (LeanRelease.ofLoop T L₀ real) Loop.progOf Loop.freshOf Loop.readCoins)
              (LeanRelease.ofLoop T L₀ real).stop).nLeaves x *
              FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t *
            (Fork CoinOpen.shaH ((((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) (LeanRelease.ofLoop T L₀ real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoop T L₀ real).stop).swP vs).prover w) Vs θ x).Pr
              CoinOpen.shaH)

/-- **C-Flock's zero knowledge at the session loop's own frames**, at every table data `T` that admits unrefused coins
(`TablesAdmitUnrefused`): `SessionLoopZKAt T`. -/
def _root_.Flock.Guarantees.SessionLoopZK : Prop :=
  ∀ T : CoinLayer.TableData, _root_.Flock.Guarantees.TablesAdmitUnrefused T → _root_.Flock.Guarantees.SessionLoopZKAt T

/-- **C-Flock's zero knowledge at the session loop's own frames, at the deployed tables** (`Masked.dataE`):
`SessionLoopZKAt Masked.dataE`, with no premise on the tables beyond `SessionLoopZKAt`'s. -/
def _root_.Flock.Guarantees.SessionLoopZKDeployed : Prop :=
  _root_.Flock.Guarantees.SessionLoopZKAt Masked.dataE

end Claims.ZKServer
