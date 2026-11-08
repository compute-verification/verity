module

public import Proofs.Flock.SessionLoop.L0Fixed
public import Proofs.Flock.SessionLoop.L0FixedHonest
public import Proofs.Flock.SessionLoop.RoundOff
public import Proofs.Flock.SessionLoop.RoundSwitch
public import Proofs.Flock.SessionLoop.RoundLig
public import Proofs.Flock.SessionLoop.SchedSlot

/-!
# `LiveSched` for any worker, at its own histories

`LeanRelease.LiveSched` asks its equation after every live history, but `live` checks the worker's values at each
prefix, never the history's messages: only `ZkServer.OwnHist` ties each message to the deferred server's frame of the
worker's round there. The deferred server meets no other history (`ZkServer.RefDOwn`, `split_deferred`'s `hown`), so the
equation is asked there only, as `FinalRoundEq` asks its own:

* **`Loop.TakesValsAt`**: `Loop.TakesVals` after one history: each word the frame frees in the worker's round is `vs`'s
  value there. `TakesVals` is it after every history (`Loop.takesVals_iff`).
* **`Loop.TakesValsOwn`**: it after every history of the worker's own. A worker whose rounds read the history's messages
  can meet it and not `TakesVals`; a worker that meets `TakesVals` meets it (`Loop.takesValsOwn_of_takesVals`), the honest
  worker `Loop.valsWk honestVals` among them (`Loop.takesValsOwn_honestVals`).
* **`LeanRelease.LiveSchedOwn`**: `LiveSched` with `OwnHist` among its hypotheses, in `FinalRoundEq`'s place; `LiveSched`
  gives it (`LeanRelease.liveSchedOwn_of_liveSched`).
* **`LeanRelease.liveSchedAt_ofLoopLink`**: `LiveSched`'s equation after one history (`LeanRelease.LiveSchedAt`;
  `LiveSched` is it after every history, `LeanRelease.liveSched_iff`), from `TakesValsAt` at that history alone, at
  `SessionLoopZKDeployed`'s premises on the session, the statement and the loop's inputs, where the rows hold at `vs`
  (`Loop.RowsOK`) and its level-0 words are the ZK server's own (`Loop.L0Fixed`). The history stands at a round of some
  rep `2j + r` before its inner proof's (`sched_slot`), which is its first (`release_sched_zcBind`), in its PIOP
  (`roundFrames_piop`), in the ring switch from its rank check's (`release_sched_switch`), the opening's binding
  (`release_sched_openBind`), level 0's (`release_sched_l0`, `release_sched_close`) or Ligerito's past it
  (`release_sched_lig`). No round frees a level-0 word, so level 0's release is the ZK server's body whatever the worker
  sends, and the switched message's is `vs`'s: `L0Fixed` is what equates them.
* **`LeanRelease.liveSchedOwn_ofLoopLink`**: for every worker `wk` and every `vs` it takes at its own histories
  (`TakesValsOwn`), `LiveSchedOwn` holds: `liveSchedAt_ofLoopLink` at each. A worker that takes no such `vs`'s values
  is outside it. `LeanRelease.liveSchedOwn_honestVals`: its premises on the worker and the values hold at the honest
  worker and values; `LeanRelease.liveSched_honestVals`: there `LiveSched` holds after every history.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal

open _root_.Flock (V F128)
open _root_.Flock.Firewall.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Loop

/-- **The worker's free words are `vs`'s values at its own histories**: `TakesValsAt` after every history each of whose
messages is the deferred server's frame of the worker's round before it (`ZkServer.OwnHist`), the only histories the
deferred server meets. -/
def TakesValsOwn (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)),
    L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) → TakesValsAt vs L wk x w ν t κ m₀ ρ₀ ops

theorem takesValsOwn_of_takesVals {vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals}
    {L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)}
    {wk : L.toSrv.Worker} {x : Msg} {w : Flock.Wit} (h : TakesVals vs L wk x w) : TakesValsOwn vs L wk x w :=
  fun ν t κ m₀ ρ₀ ops _ => h ν t κ m₀ ρ₀ ops

/-- **The honest worker takes the honest values at its own histories**: `valsWk honestVals` at the session loop
(`takes_honestVals_ofLoopLink`). -/
theorem takesValsOwn_honestVals
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit) :
    TakesValsOwn (CoinLayer.session Loop.loopNr Masked.dataE).honestVals (LeanRelease.ofLoopLink L₀ src real)
      (valsWk (CoinLayer.session Loop.loopNr Masked.dataE).honestVals (LeanRelease.ofLoopLink L₀ src real)) x w :=
  takesValsOwn_of_takesVals (takes_honestVals_ofLoopLink L₀ src real).1

end Loop

namespace LeanRelease

/-- **`LiveSched` at the worker's own histories**: `LiveSched`'s equation after every history it reads at which the loop
schedules its own round `r`, each of whose messages is the deferred server's frame of the worker's round before it
(`ZkServer.OwnHist`), in `FinalRoundEq`'s place. -/
def LiveSchedOwn (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg) (r : ByteArray),
    0 < ops.length → ops.length < refK x →
    (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    (ofLoopLink L₀ src real).toSrv.OwnHist (ofLoopLink L₀ src real).fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    Loop.schedOf (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) = some r →
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom ((CoinLayer.session Loop.loopNr Masked.dataE).swMsgAt x
        ((CoinLayer.session Loop.loopNr Masked.dataE).wt x w) (vs x w ν) t
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2))

theorem liveSchedOwn_of_liveSched {vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals}
    {L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)}
    {src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray)}
    {real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray)}
    {wk : (ofLoopLink L₀ src real).toSrv.Worker} {x : Msg} {w : Flock.Wit} (h : LiveSched vs L₀ src real wk x w) :
    LiveSchedOwn vs L₀ src real wk x w :=
  fun ν t κ m₀ ρ₀ ops v r hpos hlt hl hok href _ hv hsc => h ν t κ m₀ ρ₀ ops v r hpos hlt hl hok href hv hsc

/-- **`LiveSched`'s equation after one history** `(m₀, ρ₀) :: ops` at the tape `(ν, t)` and the check coins `κ`. -/
def LiveSchedAt (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) : Prop :=
  ∀ (v : Msg) (r : ByteArray),
    0 < ops.length → ops.length < refK x →
    (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    Loop.schedOf (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) = some r →
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom ((CoinLayer.session Loop.loopNr Masked.dataE).swMsgAt x
        ((CoinLayer.session Loop.loopNr Masked.dataE).wt x w) (vs x w ν) t
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2))

/-- `LiveSched` is `LiveSchedAt` after every history. -/
theorem liveSched_iff (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit) :
    LiveSched vs L₀ src real wk x w ↔ ∀ ν t κ m₀ ρ₀ ops, LiveSchedAt vs L₀ src real wk x w ν t κ m₀ ρ₀ ops :=
  Iff.rfl

/-- **`LiveSched` after one history, from the worker's free words there**: at `SessionLoopZKDeployed`'s release and
premises (as in `liveSchedOwn_ofLoopLink`), where the rows hold at `vs` and its level-0 words are the ZK server's own
(`Loop.L0Fixed`), after any history at which the worker's free words are `vs`'s (`Loop.TakesValsAt`), `LiveSched`'s
equation holds there. The release at round `n` reads the worker's round `n`; the frame's state before it reads the
history only through `live` and the coins. -/
theorem liveSchedAt_ofLoopLink (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (hreal : real = Loop.realOf src) (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hTab : (CoinLayer.session Loop.loopNr Masked.dataE).Tables)
    (hTape : (CoinLayer.session Loop.loopNr Masked.dataE).TapeOK)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) (hS : Loop.Src L₀ src x w)
    (hRows : Loop.RowsOK vs x w) (hL0 : Loop.L0Fixed vs x w)
    (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (hTV : Loop.TakesValsAt vs (ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops) :
    LiveSchedAt vs L₀ src real wk x w ν t κ m₀ ρ₀ ops := by
  intro v b hpos hlt hl hok href hv hsc
  have hx : Masked.Live x := Masked.live_of hSc hR
  obtain ⟨j, r, h1, h2⟩ := sched_slot (withFiles (withLink L₀ src) src) real wk hx hSv (ν, t) κ m₀ ρ₀ ops
    (fun _ _ _ hs => by rw [hreal]; exact Loop.realTabs_of src x w (ν, t) hs) hpos hlt hl hsc
  have hj := Masked.lt_of_jOf j
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨-, -, -, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst]
    exact (Masked.zkTab x j).hz
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hj).1
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  -- the rep's rounds: its first, the PIOP, the ring switch from the rank check's, then past the PIOP
  have eS : ∀ f : _root_.Flock.Schedule → ℕ, Masked.repSlot x (2 * j.val + r.val) f =
      Masked.repBase x (2 * j.val + r.val) + FlockComplete.piopRoundsZk spec (Masked.stAt x j) +
        f (Masked.zkTab x j).sch := by
    intro f
    rw [Masked.repSlot_eq hsx hz, ← Masked.repBase_eq hsx, hst]
  have eR : Masked.rankSlot x j.val r.val =
      Masked.repBase x (2 * j.val + r.val) + ((Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6) := by
    unfold Masked.rankSlot
    rw [getElem!_pos (Masked.stsOf x) j.val hj]
    try rfl
  have eP : FlockComplete.piopRoundsZk spec (Masked.stAt x j) =
      ((Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6) +
        (2 + spec.points * (Masked.stAt x j).regions.size + 1) := rfl
  simp only [eS, Masked.innerOff] at h2
  rcases Nat.lt_or_ge ops.length (Masked.rankSlot x j.val r.val) with hA | hA
  · rcases Nat.eq_or_lt_of_le h1 with h0 | h0
    · exact release_sched_zcBind vs L₀ src real hreal wk hR hSv hSc hS hTab hTape hRows ν t κ m₀ ρ₀ ops hTV hpos
        hlt hl hok href hv j r h0.symm
    · exact release_of_roundFrames _ x (ν, t) _ false v _ (roundFrames_piop vs L₀ src real hreal wk hR hSv hSc hS ν t
        κ m₀ ρ₀ ops hTV hpos hlt hl hok href hv j r (jj := ops.length - Masked.repBase x (2 * j.val + r.val))
        (by omega) (by omega) (by rw [eR] at hA; unfold piopNR; omega))
  · rcases Nat.lt_or_ge ops.length (Masked.repSlot x (2 * j.val + r.val) fun _ => 0) with hB | hB
    · exact release_sched_switch vs L₀ src real hreal wk hR hSv hSc hS ν t κ m₀ ρ₀ ops hTV hpos hlt hl hok href hv j r
        (k := ops.length - Masked.rankSlot x j.val r.val) (by omega) hB
    · simp only [eS] at hB
      obtain ⟨i, hi⟩ : ∃ i, ops.length = Masked.repSlot x (2 * j.val + r.val) fun _ => i :=
        ⟨ops.length - (Masked.repBase x (2 * j.val + r.val) + FlockComplete.piopRoundsZk spec (Masked.stAt x j)),
          by simp only [eS]; omega⟩
      have hi' := hi
      simp only [eS] at hi'
      by_cases hl0 : i ≤ (Masked.zkTab x j).sch.k0 + 3
      · rcases Nat.eq_zero_or_pos i with rfl | hi0
        · exact release_sched_openBind vs L₀ src real hreal wk hR hSv hSc hS ν t κ m₀ ρ₀ ops hTV hpos hlt hl hok href
            hv j r hi
        · exact release_sched_l0 vs L₀ src real hreal wk hR hSv hSc hS hL0 ν t κ m₀ ρ₀ ops hTV hpos hlt hl hok href hv
            j r hi hi0 hl0
      · by_cases hc : i = (Masked.zkTab x j).sch.k0 + 4
        · exact release_sched_close vs L₀ src real hreal wk hR hSv hSc hS ν t κ m₀ ρ₀ ops hTV hpos hlt hl hok href hv
            j r (by simp only [eS]; omega)
        · exact release_sched_lig vs L₀ src real hreal wk hR hSv hSc hS ν t κ m₀ ρ₀ ops hTV hpos hlt hl hok href hv j r
            hi (by omega) (by omega)

/-- **`LiveSched` for any worker, at its own histories**: at `SessionLoopZKDeployed`'s release (`ofLoopLink` at the
loop's own tables, `real = Loop.realOf src`), at its premises on the session (`Tables`, `TapeOK`), the statement (in the
masked verifier's scope, under the relation, served by the loop) and the loop's inputs (`Loop.Src`), every worker `wk`
whose free words are `vs`'s values at its own histories (`Loop.TakesValsOwn`), where the rows hold at `vs`
(`Loop.RowsOK`) and its level-0 words are the ZK server's own (`Loop.L0Fixed`: no round frees a level-0 word, so the
release there is the ZK server's body whatever the worker sends), meets `LiveSchedOwn` at `vs`: `liveSchedAt_ofLoopLink`
at each own history. -/
theorem liveSchedOwn_ofLoopLink (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (hreal : real = Loop.realOf src) (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hTab : (CoinLayer.session Loop.loopNr Masked.dataE).Tables)
    (hTape : (CoinLayer.session Loop.loopNr Masked.dataE).TapeOK)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) (hS : Loop.Src L₀ src x w)
    (hRows : Loop.RowsOK vs x w) (hL0 : Loop.L0Fixed vs x w)
    (hTV : Loop.TakesValsOwn vs (ofLoopLink L₀ src real) wk x w) :
    LiveSchedOwn vs L₀ src real wk x w :=
  fun ν t κ m₀ ρ₀ ops v r hpos hlt hl hok href hown hv hsc =>
    liveSchedAt_ofLoopLink vs L₀ src real hreal wk hTab hTape hR hSv hSc hS hRows hL0 ν t κ m₀ ρ₀ ops
      (hTV ν t κ m₀ ρ₀ ops hown) v r hpos hlt hl hok href hv hsc

/-- **At the honest worker and values**: `liveSchedOwn_ofLoopLink` at `valsWk honestVals`, whose premises on the worker
and the values hold there (`Loop.takesValsOwn_honestVals`, `Loop.rowsOK_honestVals`, `Loop.l0Fixed_honestVals`). -/
theorem liveSchedOwn_honestVals
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (hreal : real = Loop.realOf src) {x : Msg} {w : Flock.Wit}
    (hTab : (CoinLayer.session Loop.loopNr Masked.dataE).Tables)
    (hTape : (CoinLayer.session Loop.loopNr Masked.dataE).TapeOK)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) (hS : Loop.Src L₀ src x w) :
    LiveSchedOwn (CoinLayer.session Loop.loopNr Masked.dataE).honestVals L₀ src real
      (Loop.valsWk (CoinLayer.session Loop.loopNr Masked.dataE).honestVals (ofLoopLink L₀ src real)) x w :=
  liveSchedOwn_ofLoopLink _ L₀ src real hreal _ hTab hTape hR hSv hSc hS (Loop.rowsOK_honestVals hR)
    (Loop.l0Fixed_honestVals x w (Masked.live_of hSc hR) hR) (Loop.takesValsOwn_honestVals L₀ src real x w)

/-- **`LiveSched` at the honest worker and values**: `liveSchedAt_ofLoopLink` after every history, at which the honest
worker takes the honest values (`Loop.takes_honestVals_ofLoopLink`), where the rows hold (`Loop.rowsOK_honestVals`) and
the level-0 words are the ZK server's own (`Loop.l0Fixed_honestVals`). -/
theorem liveSched_honestVals
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (hreal : real = Loop.realOf src) {x : Msg} {w : Flock.Wit}
    (hTab : (CoinLayer.session Loop.loopNr Masked.dataE).Tables)
    (hTape : (CoinLayer.session Loop.loopNr Masked.dataE).TapeOK)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) (hS : Loop.Src L₀ src x w) :
    LiveSched (CoinLayer.session Loop.loopNr Masked.dataE).honestVals L₀ src real
      (Loop.valsWk (CoinLayer.session Loop.loopNr Masked.dataE).honestVals (ofLoopLink L₀ src real)) x w :=
  fun ν t κ m₀ ρ₀ ops => liveSchedAt_ofLoopLink _ L₀ src real hreal _ hTab hTape hR hSv hSc hS
    (Loop.rowsOK_honestVals hR) (Loop.l0Fixed_honestVals x w (Masked.live_of hSc hR) hR) ν t κ m₀ ρ₀ ops
    ((Loop.takes_honestVals_ofLoopLink L₀ src real).1 ν t κ m₀ ρ₀ ops)

end LeanRelease

end Claims.ZKServer
