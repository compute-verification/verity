import Proofs.Flock.Claims.ZK.FrameAdmits

/-!
# C-Flock's zero knowledge at the session loop's own frames (trusted)

`SessionLoopZK` is the session loop's zero-knowledge bound (Stage B, `note:20261022T0001Z-report-zkfin-stageb`) as a
guarantee: at the deployed relation (`Flock.Readable.relation`), at the release whose transport state is the loop's own
frames (`LeanRelease.ofLoop`, each rep's frame driven on `Drive.reqs` from the decoded coins), at every statement `x` the
loop serves (`Loop.Serves`: its `Hello` leaves at `x`'s setup, which it refuses where the setup's sizes are past its
frames', `Session.sizesOk`), and at the hiding gap of hm96-sha512's pinned key (`hidingGap My saltHash`, with no hiding
premise). It concludes that the deployed frame's state is the loop's but for `stopped` (`ProgChecks`) and the bound.

Each rep's table is the loop's own (`Loop.realOf`: `tabOfLevel0` at the level 0 `Level0.ofDraws` draws), at every
level-0 source `src` (the host's packed witness, the bytes drawn for level 0 and the salts for levels ≥ 1). What it
takes of the loop beyond the code is named, each an obligation:

* `LoopInnerRound0`: `L₀`'s round at the inner proof's round 0 is the firewall's own at some 16-byte `final_c'` (Rust's
  firewall sends that round, `check_self`, at the inner proof's label).
* `TablesAdmitUnrefused`, on the tables `T` alone: where, at a `Live` statement, no coin fault shows (`Masked.Refused`),
  every table admits the coins. `SessionLoopZK` takes it; at the deployed tables it is proved
  (`LeanRelease.tablesAdmitUnrefused_dataE`), so `SessionLoopZKDeployed` does not.

`SessionLoopZKDeployed` is `SessionLoopZKAt Masked.dataE` at the loop's own link (`LeanRelease.ofLoopLink`: `withLink`,
the roots of the level 0 the loop drew, `Loop.linkOf`, as its `Commit` carries them). In place of `LinkRoundEq` it
takes the masked verifier's scope (`Claims.Flock.Scope`) and `Loop.LinkSrc` (`flock-firewall-level0-src`): the level 0
the loop draws has the tape's inputs.

The transport's other trusts (`LeanRelease.LinkRoundEq`, `FinishIs`, `LiveRoundEq`, `FillRoundEq`) and the tables' side
(`CoinLayer` `Tables`, `Complete`, `DummyReg`, `Causal`, `SwCausal`, `TapeOK`) are the statement's premises, with
`FinishIs` in place of the `FinalRoundEq` `Flock.zk_server_lean_loop` takes (`LeanRelease.finalRoundEq_ofLoop`, with the
loop's frames refusing every coin fault, `LeanRelease.frameUnrefused_ofLoop`). The proofs are
`Flock.SecurityProofs.SessionLoopZK` and `SessionLoopZKDeployed` (`Proofs.Flock.SessionLoop`),
`Flock.zk_server_lean_ofLoop` at the gap.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker labelFrame)
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

/-- **The session loop's own tables** at the level-0 source `src`: rep stream `k`'s table is table `k / 2`'s
`tabOfLevel0` at the level 0 `Level0.ofDraws` draws at its `m` from `src`'s packed witness and drawn bytes, with `src`'s
salts for levels ≥ 1 (`Frames.placeOf` on the loop's `Level0`, which `Level0.draw` drew on the host's packed witness and
the operating system's bytes, and `Level0.commit` binds at the table's `m`; `Finish.serveLoop`'s salts). -/
def Loop.realOf {R : Type} (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (ρ : R) (k : ℕ) : V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray) :=
  match FlockDeploy.setupOf true x with
  | some (_, _, sts) => do
    let (g, d, hi) ← src x w ρ k
    let l ← _root_.Flock.Firewall.Level0.ofDraws sts[k / 2]!.m g d
    let t ← _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit sts[k / 2]!) l
    pure (t, hi)
  | none => .error "no session"

/-- **The level 0 the loop draws for table `j`** at the source `src`: `Level0.ofDraws` at table `j`'s `m` on the packed
witness and drawn bytes `src` gives rep stream `2j` (one level 0 a table, as `Loop.realOf` reads it). -/
def Loop.level0Of {R : Type} (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (ρ : R) (sts : Array Stmt) (j : ℕ) : V _root_.Flock.Firewall.Level0.Level0 := do
  let (g, d, _) ← src x w ρ (2 * j)
  _root_.Flock.Firewall.Level0.ofDraws sts[j]!.m g d

/-- **The loop's own link** at the source `src`: each table's drawn level-0 root, with no public words, as the loop's
`Commit` carries them (`Session.relay`'s `commitFrame` of the roots `Level0.ofSession` bound); the empty root where the
draw is refused, which the loop's `Commit` check refuses. -/
def Loop.linkOf {R : Type} (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (ρ : R) : Msg :=
  match FlockDeploy.setupOf true x with
  | some (_, _, sts) =>
    0 :: FlockDeploy.linkBody
      (List.ofFn fun j : Fin sts.size => ((Loop.level0Of src x w ρ sts j.val).toOption.map (·.root)).getD .empty)
      (List.replicate sts.size ByteArray.empty)
  | none => []

/-- `L₀` with the loop's own link at the source `src` (`Loop.linkOf`): the link the loop's `Commit` carries, which it
builds itself from the level 0 it drew. -/
def LeanRelease.withLink {R : Type} (L₀ : LeanRelease Flock.Wit R)
    (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray)) : LeanRelease Flock.Wit R :=
  { L₀ with link := fun x w ρ _ => Loop.linkOf src x w ρ }

/-- **The release at the session loop's own frames and link**, at the deployed tables: `ofLoop` at `withLink L₀ src`. -/
noncomputable def LeanRelease.ofLoopLink
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)) :
    LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) :=
  LeanRelease.ofLoop Masked.dataE (LeanRelease.withLink L₀ src) real

/-- `flock-firewall-level0-src` (a trust in the host's inputs and the operating system's bytes, which the loop reads and
does not check). **The loop draws the tape's level 0** at `x`: at every nonce and tape, table `j`'s drawn level 0
(`Loop.level0Of`) has `Masked.linkIn`'s inputs, the witness's packed table at the tape's mask words (what the host
hands the loop, as `hg` takes it for the finish) and the tape's extra lanes, lane paddings and level-0 salts (the bytes
`Level0.draw` reads from `IO.getRandomBytes`). It says nothing of how a root is committed: the root is `Link.root` of
those inputs by the code (`Level0.ofDraws_ok`). -/
def Loop.LinkSrc (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (j : Fin (Masked.jOf x)),
    ∃ l, Loop.level0Of src x w (ν, t) (Masked.stsOf x) j.val = .ok l ∧ l.toIn = Masked.linkIn x w t j

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
`TapeOK`), at every repetition count `0 < t`, every `L₀` (the release's inner rounds), every level-0 source `src` (the
rep's tables `real` are the loop's own, `Loop.realOf src`) and worker `wk`, at every statement `x` the loop serves
(`Loop.Serves`) with a witness `w` under the relation, where the loop's obligation (`LoopInnerRound0`) and the
transport's round trusts hold: the deployed frame's state
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
      (src : Msg → Flock.Wit →
        (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
        ℕ → V (Array F128 × ByteArray × Array ByteArray))
      (real : Msg → Flock.Wit →
        (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
        ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)),
    real = Loop.realOf src →
    ∀ (wk : (LeanRelease.ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit),
    Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w → Loop.Serves x →
    _root_.Flock.Guarantees.LoopInnerRound0 T L₀ real x w →
    LeanRelease.LinkRoundEq (CoinLayer.session Loop.loopNr T) vs (LeanRelease.ofLoop T L₀ real) x w →
    LeanRelease.FinishIs (CoinLayer.session Loop.loopNr T) vs (LeanRelease.ofLoop T L₀ real) CoinOpen.shaH wk x w →
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

/-- **C-Flock's zero knowledge at the session loop's own frames, at the deployed tables** (`Masked.dataE`), at the
loop's own link: `SessionLoopZKAt Masked.dataE` at the release `withLink L₀ src`, whose link is the one the loop's
`Commit` carries (`Loop.linkOf src`), with no premise on the tables beyond `SessionLoopZKAt`'s. In place of
`LinkRoundEq` it takes that `x` is in the masked verifier's scope (`Claims.Flock.Scope`, the statement check
`flock-verify verify --zk` runs, `ScopeMatches`) and that the loop draws the tape's level 0 (`Loop.LinkSrc`). The scope
is a premise because the loop does not run that check, and cannot run all of it: the check reads the verifier's own
`--public-inputs` and `--registered`, which the loop never sees. At a statement outside it the model's session has no
tables, so its link carries no root while the loop's `Commit` carries each table's. -/
def _root_.Flock.Guarantees.SessionLoopZKDeployed : Prop :=
  ∀ (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (t : ℕ), 0 < t →
    (CoinLayer.session Loop.loopNr Masked.dataE).Tables →
    (CoinLayer.session Loop.loopNr Masked.dataE).Complete
      (Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse) →
    (CoinLayer.session Loop.loopNr Masked.dataE).DummyReg → (CoinLayer.session Loop.loopNr Masked.dataE).Causal →
    (CoinLayer.session Loop.loopNr Masked.dataE).SwCausal vs → (CoinLayer.session Loop.loopNr Masked.dataE).TapeOK →
    ∀ (L₀ : LeanRelease Flock.Wit
        ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
      (src : Msg → Flock.Wit →
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
        ℕ → V (Array F128 × ByteArray × Array ByteArray))
      (real : Msg → Flock.Wit →
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
        ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)),
    real = Loop.realOf src →
    ∀ (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit),
    Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w → Loop.Serves x →
    _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ real x w →
    Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x → Loop.LinkSrc src x w →
    LeanRelease.FinishIs (CoinLayer.session Loop.loopNr Masked.dataE) vs (LeanRelease.ofLoopLink L₀ src real) CoinOpen.shaH wk x w →
    LeanRelease.LiveRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs (LeanRelease.ofLoopLink L₀ src real) CoinOpen.shaH wk x w →
    LeanRelease.FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs (LeanRelease.ofLoopLink L₀ src real) CoinOpen.shaH wk x w →
    ∀ {Θ : Type} (Vs : VStar Θ) (θ : Θ),
    Vs.k x = (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1 →
    ∀ (Ev : List ((Msg × Out) × ℕ) → Prop),
    (LeanRelease.ofLoopLink L₀ src real).ProgChecks x w ∧
    Pr (fun ω : ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) ×
          (LeanRelease.ofLoopLink L₀ src real).toSrv.Chk =>
        Ev ((LeanRelease.ofLoopLink L₀ src real).toSrv.timed x
          (runStar CoinOpen.shaH ((LeanRelease.ofLoopLink L₀ src real).toSrv.proverD (LeanRelease.ofLoopLink L₀ src real).fillOf
            (fun x => (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1) wk w) Vs x ω θ).msgs)) ≤
      2 * (Pr (fun c : (leaky1 ((((CoinLayer.session Loop.loopNr Masked.dataE).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE) (LeanRelease.ofLoopLink L₀ src real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoopLink L₀ src real).stop).refP.withSim fun x _ c =>
              ((CoinLayer.session Loop.loopNr Masked.dataE).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE) (LeanRelease.ofLoopLink L₀ src real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoopLink L₀ src real).stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1)
            (LeanRelease.ofLoopLink L₀ src real).toSrv.stop).C x =>
          Ev ((LeanRelease.ofLoopLink L₀ src real).toSrv.timed x ((leaky1 ((((CoinLayer.session Loop.loopNr Masked.dataE).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE) (LeanRelease.ofLoopLink L₀ src real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoopLink L₀ src real).stop).refP.withSim fun x _ c =>
              ((CoinLayer.session Loop.loopNr Masked.dataE).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE) (LeanRelease.ofLoopLink L₀ src real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoopLink L₀ src real).stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1)
            (LeanRelease.ofLoopLink L₀ src real).toSrv.stop).out CoinOpen.shaH Vs θ x c))) +
          (8 * ((CoinLayer.session Loop.loopNr Masked.dataE).withStops (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE)
              (LeanRelease.ofLoopLink L₀ src real) Loop.progOf Loop.freshOf Loop.readCoins)
              (LeanRelease.ofLoopLink L₀ src real).stop).nLeaves x *
              FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
            8 * t * ((CoinLayer.session Loop.loopNr Masked.dataE).withStops (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE)
              (LeanRelease.ofLoopLink L₀ src real) Loop.progOf Loop.freshOf Loop.readCoins)
              (LeanRelease.ofLoopLink L₀ src real).stop).nLeaves x *
              FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t *
            (Fork CoinOpen.shaH ((((CoinLayer.session Loop.loopNr Masked.dataE).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE) (LeanRelease.ofLoopLink L₀ src real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoopLink L₀ src real).stop).swP vs).prover w) Vs θ x).Pr
              CoinOpen.shaH)

end Claims.ZKServer
