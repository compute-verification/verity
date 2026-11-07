module

public import Proofs.Flock.Claims.ZK.Refine
public import Proofs.Flock.Claims.ZK.FrameRounds
public import Proofs.Flock.Soundness.ZK.Session

/-!
# The release before the `Finish` is the unswitched cut

The ZK server masks each worker round with the pads its frame drew, whatever the worker's values. The switched prover
(`RefSession.swP`) commits its pads tree over moved pads where a table's rows fail at the worker's values
(`Table.swPre`), so the transport premises at it (`LeanRelease.LinkRoundEq`, `LiveRoundEq`, `FillRoundEq`) are false
at exactly the witnesses zero knowledge at any values is about. Here the release before the `Finish` is held to the
**unswitched** prover instead: the worker's values on the sampled pads, rows or not.

* `Table.unPre`: a table's unswitched pre-final view, the table at clear values (`Table.onVals`) at the worker's values,
  as one of the table's. Where the rows hold it is the switched one (`Table.swPre_of_rows`); at the honest values it is
  the honest one (`Table.unPre_vals`). `Table.un_prefinal_cut` is Lemma A at it: cut to what the coins admit, it is
  distributed as the dummy's pre-final view at any values.
* `Coined.unPreWith`, `Coined.unContent`, `unCutAt`, `unDigOf`, `unExeOf`: the unswitched prover at the session's tables
  with the executable's leaves, as `swPreWith`, `swContent`, `swCutAt`, `swDigOf`, `swExeOf` have the switched one;
  `card_unCutAt_ideal`, Lemma A at them, at ideal leaves.
* `RefSession.unCutOf`, `RefSession.unMsgAt`, `RefSession.unP`: the unswitched reference prover at the workers' values
  `vs`: its message before the last coin is the round encoding `enc` of the unswitched cut, and after it the refusal
  `off`, so it never sends a final message. `RefSession.swMsgAt_of_rows`: before the last coin it is the switched
  prover's where the rows hold at the tape (`RefSession.RowsAtTape`); `RefSession.unMsgAt_vals`: at the honest values,
  `refP`'s.
* `LeanRelease.LinkRoundEqStop`, `LiveRoundEqStop`, `FillRoundEqStop`: **the per-round byte equalities at the unswitched
  prover**, at the link and at the worker's and the stand-in frame rounds, each asked at the deferred release's own
  histories (`ZkServer.OwnHist`) as `FinalRoundEq` asks its own. Where the rows hold they are `LinkRoundEq`,
  `LiveRoundEq` and `FillRoundEq` (`LeanRelease.liveRoundEqStop_of_rows`, …). The frame rounds' two follow, as
  today's do, from the device's frames through the frame's masks being the session's unswitched message
  (`LiveFramesStop`, `LiveHonestStop`, `FillFramesStop`; `LeanRelease.liveRoundEqStop_of_frames`, …).
* `ZkServer.RefDBelow`, `ZkServer.take_deferred`, `ZkServer.stop_deferred`: `RefDOwn` below the last round alone gives
  the deferred server's first `n x - 1` messages as `P`'s, and, where it stops at its last round, its whole release as
  `P`'s run cut there.
* `LeanRelease.refDBelow_unP`: at the session with the frame's stops (`RefSession.withStops`), `RefDBelow` holds at its
  unswitched prover from `HelloEq`, the stops being the frame's (`StopsAt`), the release refusing a failed opening
  (`OpeningsRefused`) and the three equalities.
* **`LeanRelease.release_preFinish`, `LeanRelease.release_stop`**: for every worker and any values, the deferred
  release's messages before the `Finish` are the unswitched prover's run, and where the session stops at its `Finish`
  the release is that run with its last message the stop (`release_preFinish_runStar`, `release_stop_runStar` at a
  verifier's run). No premise reads the relation, `RowsAt` or `RowsOK`.
-/

@[expose] public section

namespace FlockSoundness.ZK.Table

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The unswitched prover's pre-final view**: the worker's clear values `v` on the sampled pads, whether or not its rows
hold (`swPre` without its switch). -/
noncomputable def unPre {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : Table F K W Salt Seed D Hi VL) (w : W) (v : T.Vals) (ω : T.Rand) : T.PreView :=
  T.toBasePre (T.onVals.preView (w, v) ω)

/-- Where the rows hold, the switched pre-final view is the unswitched one. -/
theorem swPre_of_rows {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : Table F K W Salt Seed D Hi VL) (w : W) (v : T.Vals) (ω : T.Rand)
    (hr : T.onVals.RowsAt (w, v) ω.1 ω.2.1) : T.swPre w v ω = T.unPre w v ω := by
  unfold swPre
  exact ite_eq_left hr

/-- At the honest values the unswitched pre-final view is the table's. -/
theorem unPre_vals {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : Table F K W Salt Seed D Hi VL) (w : W) (ω : T.Rand) : T.unPre w T.vals ω = T.preView w ω :=
  T.toBasePre_preView w ω

/-- **Lemma A at the unswitched prover**: for a witness `w` and a dummy `w0` carrying the public region data, and
messages `S` the coins admit, the unswitched pre-final view at any clear values `v`, cut to `S`, and the dummy's at the
honest ones have the same distribution (Lemma A at `onVals`). -/
theorem un_prefinal_cut {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt]
    [AddCommGroup Seed] [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype Salt] [DecidableEq Salt]
    [Fintype Seed] [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi]
    (T : Table F K W Salt Seed D Hi VL) (S : T.Sent) (hS : T.Admits S) (w w0 : W) (hreg : T.sReg w = T.pubReg)
    (hreg0 : T.sReg w0 = T.pubReg) (v : T.Vals) :
    SameDist (fun ω => T.cut S (T.unPre w v ω)) (fun ω => T.cut S (T.preView w0 ω)) :=
  ((T.onVals.table_prefinal_indep_cut (T.valsSent S) hS (w, v) (w0, T.vals) hreg hreg0).comp T.toBaseCut).of_eq
    (fun _ => rfl) fun ω => by
      rw [← T.toBasePre_preView w0 ω]
      rfl

end FlockSoundness.ZK.Table

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-! ## The unswitched prover at the session's tables -/

/-- **The unswitched prover's pre-final view with its leaves given by position** (`lv`): `Table.unPre`, as
`Coined.preWith` has `Table.preView`. -/
noncomputable def _root_.FlockSoundness.Discharge.ZkView.Coined.unPreWith {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (b : ZkView.Base z Hi)
    (lv : ZkView.LeafPos z → ByteArray) : T.tab.PreView :=
  { T.tab.unPre w v (T.randOf b 0) with
    leaves0 := (fun q => lv (.inl (T.s0 (.inl q))), fun q => lv (.inl (T.s0 (.inr q))))
    tauCom := fun r => lv (.inr (.inr r))
    leavesP := fun r =>
      (fun k => lv (.inr (.inl (r, T.sP r (.inl k)))), fun k => lv (.inr (.inl (r, T.sP r (.inr k))))) }

/-- At ideal leaves `H(U, c(y))`, the unswitched pre-final view with its leaves given by position is `Table.unPre`. -/
theorem _root_.FlockSoundness.Discharge.ZkView.Coined.unPreWith_ideal {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (b : ZkView.Base z Hi)
    (u : ZkView.LeafPos z → ZkView.Bits × ZkView.Salt) :
    T.unPreWith w v b (fun p => ZkView.leafH ((u p).1, ZkView.saltHash (u p).2)) = T.tab.unPre w v (T.randOf b u) :=
  rfl

theorem _root_.FlockSoundness.Discharge.ZkView.Coined.swPreWith_of_rows {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (b : ZkView.Base z Hi)
    (lv : ZkView.LeafPos z → ByteArray) (hr : T.tab.onVals.RowsAt (w, v) b.1 b.2.1) :
    T.swPreWith w v b lv = T.unPreWith w v b lv := by
  unfold ZkView.Coined.swPreWith ZkView.Coined.unPreWith
  rw [T.tab.swPre_of_rows w v (T.randOf b 0) hr]

theorem _root_.FlockSoundness.Discharge.ZkView.Coined.unPreWith_vals {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (b : ZkView.Base z Hi)
    (lv : ZkView.LeafPos z → ByteArray) : T.unPreWith w T.tab.vals b lv = T.preWith w b lv := by
  unfold ZkView.Coined.unPreWith ZkView.Coined.preWith
  rw [T.tab.unPre_vals]

/-- **What each of the unswitched prover's pre-final leaves commits**: `v`'s columns and `τ` on the sampled pads. -/
noncomputable def _root_.FlockSoundness.Discharge.ZkView.Coined.unContent {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (b : ZkView.Base z Hi) :
    ZkView.LeafPos z → ByteArray :=
  T.onVals.content (w, v) b

theorem _root_.FlockSoundness.Discharge.ZkView.Coined.unContent_size {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (b : ZkView.Base z Hi)
    (p : ZkView.LeafPos z) : (T.unContent w v b p).size = 64 :=
  T.onVals.content_size _ _ _

theorem _root_.FlockSoundness.Discharge.ZkView.Coined.swContent_of_rows {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (b : ZkView.Base z Hi)
    (hr : T.tab.onVals.RowsAt (w, v) b.1 b.2.1) : T.swContent w v b = T.unContent w v b := by
  unfold ZkView.Coined.swContent
  exact ite_eq_left hr

theorem _root_.FlockSoundness.Discharge.ZkView.Coined.unContent_vals {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (b : ZkView.Base z Hi) :
    T.unContent w T.tab.vals b = T.content w b :=
  T.onVals_content_vals w b

/-- **The unswitched prover's pre-final view at coins `cz`, cut where the session stops** (`famSent`), at table
witnesses `wt`, the workers' values `v`, bases `ω` and the pre-final leaves `lv` by position (`Coined.unPreWith`), as
`swCutAt` has the switched prover's. -/
noncomputable def unCutAt {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (v : ∀ j, ((C j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) (lv : ZkView.Leaves (fun j => (C j).z) → ByteArray) :
    ∀ j, ((C j).T (cz j)).tab.Cut :=
  fun j => ((C j).T (cz j)).tab.cut (famSent (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep) j)
    (((C j).T (cz j)).unPreWith (wt j) (v j) (ω j) fun p => lv ⟨j, p⟩)

/-- The digest each of the unswitched prover's pre-final leaves commits, as bits (`Coined.unContent`). -/
noncomputable def unDigOf {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (v : ∀ j, ((C j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) (l : ZkView.Leaves (fun j => (C j).z)) : ZkView.Bits :=
  ZkView.bitsOf (((C l.1).T (cz l.1)).unContent (wt l.1) (v l.1) (ω l.1) l.2)

/-- **The executable's pre-final leaves at the unswitched prover**: `hm96-sha512/v1`'s leaf of each `Coined.unContent`
under its own salt. -/
noncomputable def unExeOf {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (v : ∀ j, ((C j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) (y : ZkView.Leaves (fun j => (C j).z) → ZkView.Salt) :
    ZkView.Leaves (fun j => (C j).z) → ByteArray :=
  fun l => _root_.Flock.Hm96.Default512.leaf (((C l.1).T (cz l.1)).unContent (wt l.1) (v l.1) (ω l.1) l.2)
    (ZkView.saltBytes (y l))

/-- The unswitched prover's leaves are `H(dig ⊕ M·y, c(y))` (`ZkView.leafSplit`). -/
theorem unExeOf_eq {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (v : ∀ j, ((C j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) (y : ZkView.Leaves (fun j => (C j).z) → ZkView.Salt) :
    unExeOf C cz wt v ω y =
      fun l => ZkView.leafH (unDigOf C cz wt v ω l + ZkView.My (y l), ZkView.saltHash (y l)) :=
  funext fun l => by
    rw [unExeOf, ZkView.leafSplit _ _ (((C l.1).T (cz l.1)).unContent_size _ _ _ _)]
    rfl

/-- **The rows hold at every table's base**: each table's clear values `v j` pass its inner constraint at every pad, at
the mask slot and level 0's hiding of `ω j` (`Table.RowsAt`). -/
def RowsAtAll {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (v : ∀ j, ((C j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) : Prop :=
  ∀ j, ((C j).T (cz j)).tab.onVals.RowsAt (wt j, v j) (ω j).1 (ω j).2.1

theorem swCutAt_of_rows {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (v : ∀ j, ((C j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) (lv : ZkView.Leaves (fun j => (C j).z) → ByteArray)
    (hr : RowsAtAll C cz wt v ω) : swCutAt C cz wt v ω lv = unCutAt C cz wt v ω lv := by
  funext j
  simp only [swCutAt, unCutAt, ZkView.Coined.swPreWith_of_rows _ _ _ _ _ (hr j)]

theorem swDigOf_of_rows {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (v : ∀ j, ((C j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) (hr : RowsAtAll C cz wt v ω) :
    swDigOf C cz wt v ω = unDigOf C cz wt v ω := by
  funext l
  simp only [swDigOf, unDigOf, ZkView.Coined.swContent_of_rows _ _ _ _ (hr l.1)]

theorem swExeOf_of_rows {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (v : ∀ j, ((C j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) (y : ZkView.Leaves (fun j => (C j).z) → ZkView.Salt)
    (hr : RowsAtAll C cz wt v ω) : swExeOf C cz wt v ω y = unExeOf C cz wt v ω y := by
  funext l
  simp only [swExeOf, unExeOf, ZkView.Coined.swContent_of_rows _ _ _ _ (hr l.1)]

theorem unCutAt_vals {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (ω : ∀ j, ZkView.Base (C j).z (C j).Hi)
    (lv : ZkView.Leaves (fun j => (C j).z) → ByteArray) :
    unCutAt C cz wt (fun j => ((C j).T (cz j)).tab.vals) ω lv = cutAt C cz wt ω lv := by
  funext j
  simp only [unCutAt, cutAt, ZkView.Coined.unPreWith_vals]

theorem unDigOf_vals {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) :
    unDigOf C cz wt (fun j => ((C j).T (cz j)).tab.vals) ω = digOf C cz wt ω := by
  funext l
  simp only [unDigOf, digOf, ZkView.Coined.unContent_vals]

theorem unExeOf_vals {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (ω : ∀ j, ZkView.Base (C j).z (C j).Hi)
    (y : ZkView.Leaves (fun j => (C j).z) → ZkView.Salt) :
    unExeOf C cz wt (fun j => ((C j).T (cz j)).tab.vals) ω y = exeOf C cz wt ω y := by
  funext l
  simp only [unExeOf, exeOf, ZkView.Coined.unContent_vals]
  rfl

open Finset in
/-- **Lemma A at the unswitched prover, for tables of their own types** (`Table.un_prefinal_cut` at each table):
messages that are a function of the session's cut pre-view, on the tables' randomness read through a map with fibers of
one size, take every value as often at the unswitched prover on `w`, whatever values `v` the workers supply, as at the
honest prover on `w₀`, both carrying the public region data. -/
theorem famCard_unCut_eq {F K Salt Seed D : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt]
    [AddCommGroup Seed] [Fintype F] [DecidableEq F] [Fintype Salt] [DecidableEq Salt] [Fintype Seed]
    [DecidableEq Seed] {J : ℕ} {Wf Hi VL : Fin J → Type} [∀ j, AddCommGroup (Hi j)] [∀ j, Fintype (Hi j)]
    [∀ j, DecidableEq (Hi j)] {Ω V : Type} [Fintype Ω] [DecidableEq V]
    (T : ∀ j, FlockSoundness.ZK.Table F K (Wf j) Salt Seed D (Hi j) (VL j)) (rep : ∀ j, Fin (T j).nC → Fin 2)
    (e : Ω → ∀ j, (T j).Rand) {k : ℕ} (he : ∀ r, Nat.card {x // e x = r} = k) (out : (∀ j, (T j).Cut) → V)
    (w w₀ : ∀ j, Wf j) (v : ∀ j, (T j).Vals) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg)
    (hreg₀ : ∀ j, (T j).sReg (w₀ j) = (T j).pubReg) (σ : V) :
    #{x | out (fun j => (T j).cut (famSent T rep j) ((T j).unPre (w j) (v j) (e x j))) = σ}
      = #{x | out (fun j => (T j).cut (famSent T rep j) ((T j).preView (w₀ j) (e x j))) = σ} := by
  classical
  have : Nonempty (∀ j, (T j).Rand) := ⟨fun _ => 0⟩
  rw [FlockSoundness.ZK.card_comp_of_fiber e he
      (fun r => out fun j => (T j).cut (famSent T rep j) ((T j).unPre (w j) (v j) (r j))),
    FlockSoundness.ZK.card_comp_of_fiber e he
      (fun r => out fun j => (T j).cut (famSent T rep j) ((T j).preView (w₀ j) (r j)))]
  congr 1
  exact ((FlockSoundness.ZK.SameDist.pi
    (fun j => fun ω => (T j).cut (famSent T rep j) ((T j).unPre (w j) (v j) ω))
    (fun j => fun ω => (T j).cut (famSent T rep j) ((T j).preView (w₀ j) ω))
    fun j => (T j).un_prefinal_cut (famSent T rep j) (famSent_admits T rep j) (w j) (w₀ j) (hreg j) (hreg₀ j)
      (v j)).comp out).card_eq σ

open Finset in
/-- **World `W₀` at the unswitched prover, at the session's tables**: at ideal leaves `H(U, c(y))`, any function of
the cut view takes every value as often at the unswitched prover on `wt`, whatever values `v` the workers supply, as at
the honest one on `wt₀`, both carrying the public region data (`famCard_unCut_eq`, `Coined.unPreWith_ideal`). -/
theorem card_unCutAt_ideal {J : ℕ} {V : Type} [DecidableEq V] (C : Fin J → Composed.CoinedSetup)
    (cz : ∀ j, Composed.CoinsZK (C j).Rest) (wt wt₀ : ∀ j, FlockSoundness.Model.Witness (C j).S.m)
    (v : ∀ j, ((C j).T (cz j)).tab.Vals) (hreg : ∀ j, ((C j).T (cz j)).tab.sReg (wt j) = ((C j).T (cz j)).tab.pubReg)
    (hreg₀ : ∀ j, ((C j).T (cz j)).tab.sReg (wt₀ j) = ((C j).T (cz j)).tab.pubReg)
    (out : (∀ j, ((C j).T (cz j)).tab.Cut) → V) (σ : V) :
    #{u : (∀ j, ZkView.Base (C j).z (C j).Hi) × (ZkView.Leaves (fun j => (C j).z) → ZkView.Bits × ZkView.Salt) |
        out (unCutAt C cz wt v u.1 fun l => ZkView.leafH ((u.2 l).1, ZkView.saltHash (u.2 l).2)) = σ}
      = #{u : (∀ j, ZkView.Base (C j).z (C j).Hi) × (ZkView.Leaves (fun j => (C j).z) → ZkView.Bits × ZkView.Salt) |
        out (cutAt C cz wt₀ u.1 fun l => ZkView.leafH ((u.2 l).1, ZkView.saltHash (u.2 l).2)) = σ} := by
  have e : ∀ u : (∀ j, ZkView.Base (C j).z (C j).Hi) × (ZkView.Leaves (fun j => (C j).z) → ZkView.Bits × ZkView.Salt),
      unCutAt C cz wt v u.1 (fun l => ZkView.leafH ((u.2 l).1, ZkView.saltHash (u.2 l).2)) =
        fun j => ((C j).T (cz j)).tab.cut
          (famSent (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep) j)
          (((C j).T (cz j)).tab.unPre (wt j) (v j) (famRandAt (fun j => (C j).T (cz j)) u j)) :=
    fun u => funext fun j => congrArg _ (((C j).T (cz j)).unPreWith_ideal (wt j) (v j) (u.1 j) fun p => u.2 ⟨j, p⟩)
  simp only [e]
  exact famCard_unCut_eq (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep)
    (famRandAt fun j => (C j).T (cz j))
    (fun r => ZkView.card_fiber_of_bijective (famRandAt_bijective fun j => (C j).T (cz j)) r) out wt wt₀ v hreg hreg₀ σ

/-! ## The unswitched prover at the reference session -/

namespace RefSession

/-- **The unswitched prover's cut pre-final view at coins `c`**, on table witnesses `wt`, values `v` read from what the
workers see at tape `t`, every pre-final leaf the executable's under its own salt (`swCutOf` without the switch). -/
noncomputable def unCutOf (S : RefSession) (x : Msg) (c : Fin (refK x) → S.Coin)
    (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (v : (∀ j, ZkView.Seen (S.C x j).z) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals) (t : S.Tape) :
    ∀ j, ((S.C x j).T (S.cz x c j)).tab.Cut :=
  unCutAt (S.C x) (S.cz x c) wt (v (S.seenOf x t)) (S.eP x t).1
    (unExeOf (S.C x) (S.cz x c) wt (v (S.seenOf x t)) (S.eP x t).1 (S.eP x t).2)

/-- **The unswitched prover's message after the coins `cs`** at the values `v`: before the last coin the round encoding
of the unswitched cut, as the switched prover's (`swMsgAt`), and after it the refusal `off`, so it never sends a final
message. -/
noncomputable def unMsgAt (S : RefSession) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (v : (c : Fin (refK x) → S.Coin) → (∀ j, ZkView.Seen (S.C x j).z) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals)
    (t : S.Tape) (cs : List S.Coin) : Msg :=
  if cs.length < refK x then S.enc x (S.pad cs) cs.length (S.unCutOf x (S.pad cs) wt (v (S.pad cs)) t)
  else S.off x (S.pad cs) (S.unCutOf x (S.pad cs) wt (v (S.pad cs)) t)

/-- **The unswitched reference prover** at the workers' values `vs`: `refP` with its messages on each witness
`unMsgAt`; its dummy, coins, nonce, tape and opening check `refP`'s. -/
noncomputable def unP (S : RefSession) (vs : S.SwVals) : Proto Flock.Wit :=
  { S.refP with msg := fun x w ν t cs => .ret (S.unMsgAt x (S.wt x w) (vs x w ν) t cs) }

/-- **The unswitched round encoding reads the coins before `i` only**, whatever the workers see and each pre-final
leaf's pair, at every witness and nonce (`SwCausal` at the unswitched prover). -/
def UnCausal (S : RefSession) (vs : S.SwVals) : Prop :=
  ∀ (x : Msg) (w : Flock.Wit) (ν : S.Nonce) (ω : ∀ j, ZkView.Base (S.C x j).z (S.C x j).Hi)
    (s : ∀ j, ZkView.Seen (S.C x j).z) (p : ZkView.Leaves (fun j => (S.C x j).z) → ZkView.Bits × ByteArray)
    (c c' : Fin (refK x) → S.Coin) (i : ℕ), (∀ j : Fin (refK x), j.val < i → c j = c' j) →
    S.enc x c i (unCutAt (S.C x) (S.cz x c) (S.wt x w) (vs x w ν c s) ω fun l =>
        ZkView.leafH (unDigOf (S.C x) (S.cz x c) (S.wt x w) (vs x w ν c s) ω l + (p l).1, (p l).2)) =
      S.enc x c' i (unCutAt (S.C x) (S.cz x c') (S.wt x w) (vs x w ν c' s) ω fun l =>
        ZkView.leafH (unDigOf (S.C x) (S.cz x c') (S.wt x w) (vs x w ν c' s) ω l + (p l).1, (p l).2))

/-- **Round `i`'s unswitched encoding reads the coins before `i` only** (`UnCausal` at the executable's leaves). -/
theorem enc_unCutOf (S : RefSession) (vs : S.SwVals) (x : Msg) (w : Flock.Wit) (ν : S.Nonce) (hc : S.UnCausal vs)
    (t : S.Tape) (c c' : Fin (refK x) → S.Coin) (i : ℕ) (h : ∀ k : Fin (refK x), k.val < i → c k = c' k) :
    S.enc x c i (S.unCutOf x c (S.wt x w) (vs x w ν c) t) =
      S.enc x c' i (S.unCutOf x c' (S.wt x w) (vs x w ν c') t) := by
  have e := hc x w ν (S.eP x t).1 (S.seenOf x t)
    (fun l => (ZkView.My ((S.eP x t).2 l), ZkView.saltHash ((S.eP x t).2 l))) c c' i h
  unfold unCutOf
  rw [unExeOf_eq, unExeOf_eq]
  convert e using 4

/-- **The rows hold at the tape** at the coins `c`: `RowsAtAll` at the tables' bases the tape `t` samples, at the values
read from what the workers see there. -/
def RowsAtTape (S : RefSession) (x : Msg) (c : Fin (refK x) → S.Coin)
    (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (v : (∀ j, ZkView.Seen (S.C x j).z) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals) (t : S.Tape) : Prop :=
  RowsAtAll (S.C x) (S.cz x c) wt (v (S.seenOf x t)) (S.eP x t).1

theorem swCutOf_of_rows (S : RefSession) (x : Msg) (c : Fin (refK x) → S.Coin)
    (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (v : (∀ j, ZkView.Seen (S.C x j).z) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals)
    (t : S.Tape) (hr : S.RowsAtTape x c wt v t) : S.swCutOf x c wt v t = S.unCutOf x c wt v t := by
  unfold swCutOf unCutOf
  rw [swExeOf_of_rows _ _ _ _ _ _ hr, swCutAt_of_rows _ _ _ _ _ _ hr]

/-- **Before the last coin, where the rows hold at the tape, the switched prover's message is the unswitched one's.** -/
theorem swMsgAt_of_rows (S : RefSession) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (v : (c : Fin (refK x) → S.Coin) → (∀ j, ZkView.Seen (S.C x j).z) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals)
    (t : S.Tape) (cs : List S.Coin) (hlt : cs.length < refK x) (hr : S.RowsAtTape x (S.pad cs) wt (v (S.pad cs)) t) :
    S.swMsgAt x wt v t cs = S.unMsgAt x wt v t cs := by
  unfold swMsgAt unMsgAt
  rw [ite_eq_left hlt, ite_eq_left hlt, S.swCutOf_of_rows x _ wt _ t hr]

/-- **Before the last coin, at the honest values, the unswitched prover's message is `refP`'s.** -/
theorem unMsgAt_vals (S : RefSession) (x : Msg) (w : Flock.Wit) (ν : S.Nonce)
    (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m) (t : S.Tape) (cs : List S.Coin) (hlt : cs.length < refK x) :
    S.unMsgAt x wt (S.honestVals x w ν) t cs = S.msgAt x wt t cs := by
  have hc : ∀ c, S.unCutOf x c wt (S.honestVals x w ν c) t = S.cutOf x c wt t := fun c => by
    unfold unCutOf cutOf honestVals
    rw [unExeOf_vals, unCutAt_vals]
  unfold unMsgAt msgAt
  rw [ite_eq_left hlt, ite_eq_left hlt, hc]

/-- **At the honest values `UnCausal` is `Causal`.** -/
theorem unCausal_vals (S : RefSession) (hc : S.Causal) : S.UnCausal S.honestVals := by
  intro x w ν ω _ p c c' i h
  unfold honestVals
  rw [unDigOf_vals, unDigOf_vals, unCutAt_vals, unCutAt_vals]
  exact hc x (S.wt x w) ω p c c' i h

theorem withStops_unCausal (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (vs : S.SwVals) (hRf : S.StopsCausal Rf) (hc : S.UnCausal vs) : (S.withStops Rf stop).UnCausal vs := by
  intro x w ν ω s p c c' i h
  show (if Rf x c i = true then stop else if i = 0 then S.enc x c i _ else dropDom (S.enc x c i _)) =
    (if Rf x c' i = true then stop else if i = 0 then S.enc x c' i _ else dropDom (S.enc x c' i _))
  rw [hRf x c c' i h]
  exact congrArg (fun m => if Rf x c' i = true then stop else if i = 0 then m else dropDom m) (hc x w ν ω s p c c' i h)

/-- **Where the rule stops the coins, the session's unswitched message is the stop**, before the last coin and after
it. -/
theorem unMsgAt_withStops_stop (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (v : (c : Fin (refK x) → S.Coin) → (∀ j, ZkView.Seen (S.C x j).z) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals)
    (t : S.Tape) (cs : List S.Coin) (hR : Rf x (S.pad cs) (min cs.length (refK x)) = true) :
    (S.withStops Rf stop).unMsgAt x wt v t cs = stop := by
  by_cases h : cs.length < refK x
  · rw [Nat.min_eq_left h.le] at hR
    show (if cs.length < refK x then (if Rf x (S.pad cs) cs.length = true then stop else _) else _) = stop
    rw [ite_eq_left h, ite_eq_left hR]
  · rw [Nat.min_eq_right (Nat.le_of_not_lt h)] at hR
    show (if cs.length < refK x then _ else (if Rf x (S.pad cs) (refK x) = true then stop else _)) = stop
    rw [ite_eq_right h, ite_eq_left hR]

/-- **Where the rule doesn't stop the coins, the session's unswitched link is `S`'s.** -/
theorem unMsgAt_withStops_link (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (v : (c : Fin (refK x) → S.Coin) → (∀ j, ZkView.Seen (S.C x j).z) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals)
    (t : S.Tape) (cs : List S.Coin) (hR : Rf x (S.pad cs) cs.length = false) (h0 : cs.length = 0)
    (h : cs.length < refK x) :
    (S.withStops Rf stop).unMsgAt x wt v t cs = S.unMsgAt x wt v t cs := by
  show (if cs.length < refK x then
      (if Rf x (S.pad cs) cs.length = true then stop
        else if cs.length = 0 then S.enc x (S.pad cs) cs.length (S.unCutOf x (S.pad cs) wt (v (S.pad cs)) t)
        else dropDom (S.enc x (S.pad cs) cs.length (S.unCutOf x (S.pad cs) wt (v (S.pad cs)) t))) else _) =
    (if cs.length < refK x then S.enc x (S.pad cs) cs.length (S.unCutOf x (S.pad cs) wt (v (S.pad cs)) t) else _)
  rw [ite_eq_left h, ite_eq_left h, hR, ite_eq_right Bool.false_ne_true, ite_eq_left h0]

/-- **Where the rule doesn't stop the coins, the session's unswitched message at a frame round is `S`'s without its
leading domain frame.** -/
theorem unMsgAt_withStops_frame (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (v : (c : Fin (refK x) → S.Coin) → (∀ j, ZkView.Seen (S.C x j).z) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals)
    (t : S.Tape) (cs : List S.Coin) (hR : Rf x (S.pad cs) cs.length = false) (h0 : 0 < cs.length)
    (h : cs.length < refK x) :
    (S.withStops Rf stop).unMsgAt x wt v t cs = dropDom (S.unMsgAt x wt v t cs) := by
  show (if cs.length < refK x then
      (if Rf x (S.pad cs) cs.length = true then stop
        else if cs.length = 0 then S.enc x (S.pad cs) cs.length (S.unCutOf x (S.pad cs) wt (v (S.pad cs)) t)
        else dropDom (S.enc x (S.pad cs) cs.length (S.unCutOf x (S.pad cs) wt (v (S.pad cs)) t))) else _) =
    dropDom (if cs.length < refK x then S.enc x (S.pad cs) cs.length (S.unCutOf x (S.pad cs) wt (v (S.pad cs)) t)
      else _)
  rw [ite_eq_left h, ite_eq_left h, hR, ite_eq_right Bool.false_ne_true, ite_eq_right (by omega)]

/-- The unswitched prover's first message is the session's `Hello`. -/
theorem unP_next_nil (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce)
    (t : S.Tape) : ((((S.unP vs).prover w).next x [] (ν, t)).eval H).1 = S.hello x ν := by
  show ((strat ((S.unP vs).hello x ν) (S.unP vs).abort (S.unP vs).dec ((S.unP vs).chk x ν)
    ((S.unP vs).msg x w ν t) []).eval H).1 = _
  rw [strat_eval_nil]
  rfl

/-- After `Hello`, the unswitched prover sends its message at the coins the openings open while they pass the
coin-tree check, else its `abort` (`strat_eval_cons`). -/
theorem unP_next_cons (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce)
    (t : S.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) :
    ((((S.unP vs).prover w).next x ((m₀, ρ₀) :: ops) (ν, t)).eval H).1 =
      if ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 then
        S.unMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2)
      else S.abort := by
  show ((strat ((S.unP vs).hello x ν) (S.unP vs).abort (S.unP vs).dec ((S.unP vs).chk x ν)
    ((S.unP vs).msg x w ν t) ((m₀, ρ₀) :: ops)).eval H).1 = _
  rw [strat_eval_cons]
  rfl

/-- `unP_next_nil` at the session with the frame's stops, in `S`'s own terms. -/
theorem withStops_unP_next_nil (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce) (t : S.Tape) :
    (((((S.withStops Rf stop).unP vs).prover w).next x [] (ν, t)).eval H).1 = S.hello x ν :=
  (S.withStops Rf stop).unP_next_nil vs H x w ν t

/-- `unP_next_cons` at the session with the frame's stops, in `S`'s own terms: its `abort` is the stop. -/
theorem withStops_unP_next_cons (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce) (t : S.Tape) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) :
    (((((S.withStops Rf stop).unP vs).prover w).next x ((m₀, ρ₀) :: ops) (ν, t)).eval H).1 =
      if ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 then
        (S.withStops Rf stop).unMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2)
      else stop :=
  (S.withStops Rf stop).unP_next_cons vs H x w ν t m₀ ρ₀ ops

end RefSession

end Claims.ZKMal

namespace Claims.ZKServer

open Claims Claims.ZKMal

/-! ## The deferred server below its last round -/

namespace ZkServer

/-- **`RefDOwn` below the last round**: on the deferred server's own runs, its frame below its last round `n x - 1` is
`P`'s message. It says nothing of the last round. -/
def RefDBelow {W R : Type} (S : ZkServer W R) (H : Msg → Msg) (P : W → Prover R)
    (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val) (n : Msg → ℕ) (wk : S.Worker) (x : Msg) (w : W) :
    Prop :=
  ∀ ρ κ h, S.OwnHist fill wk x w ρ κ h → h.length + 1 < n x →
    (((P w).next x h ρ).eval H).1 = S.frameD fill wk x w ρ κ h

theorem RefDOwn.below {W R : Type} {S : ZkServer W R} {H : Msg → Msg} {P : W → Prover R}
    {fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val} {n : Msg → ℕ} {wk : S.Worker} {x : Msg} {w : W}
    (hRef : S.RefDOwn H P fill n wk x w) : S.RefDBelow H P fill n wk x w :=
  fun ρ κ h hown hlt => hRef ρ κ h hown (Or.inl hlt)

/-- **`P`'s run is the deferred server's own below the last round**: each prefix of at most `n x - 1` of its rounds. -/
theorem ownHist_below {W R : Type} (S : ZkServer W R) (H : Msg → Msg) (P : W → Prover R)
    (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val) (n : Msg → ℕ) (x : Msg) (w : W) (wk : S.Worker)
    (hRef : S.RefDBelow H P fill n wk x w) (ask : List Msg → Out) (ρ : R) (κ : S.Chk) :
    ∀ i, i + 1 ≤ n x →
      S.OwnHist fill wk x w ρ κ (histOf ((interact ask (fun h => (((P w).next x h ρ).eval H).1) (n x) []).take i)) := by
  set pm : List (Msg × Msg) → Msg := fun h => (((P w).next x h ρ).eval H).1 with hpm
  set τ := interact ask pm (n x) [] with hτ
  intro i
  induction i using Nat.strong_induction_on with
  | _ i ih =>
    intro hiN j hj
    have hjl : j < τ.length := by rw [length_histOf_take] at hj; omega
    have hji : j < i := by rw [length_histOf_take] at hj; omega
    have hg := interact_getElem_fst ask pm (n x) [] j hjl
    rw [← hτ, List.nil_append, List.getElem?_eq_getElem hjl] at hg
    have hg' : (τ[j]'hjl).1 = pm (histOf (τ.take j)) := Option.some.inj hg
    have hm : ((histOf (τ.take i))[j]'hj).1 = (τ[j]'hjl).1 := by simp [histOf]
    rw [hm, histOf_take_take τ hji.le, hg']
    exact hRef ρ κ _ (ih j hji (by omega)) (by rw [length_histOf_take]; omega)

/-- **Below its last round the deferred server releases `P`'s run**: its first `n x - 1` messages and answers are
`P`'s, at every worker, check coins and tape, from `RefDBelow` alone. -/
theorem take_deferred {W R : Type} (S : ZkServer W R) (H : Msg → Msg) (P : W → Prover R)
    (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val) (n : Msg → ℕ) (x : Msg) (w : W) (wk : S.Worker)
    (hRef : S.RefDBelow H P fill n wk x w) (ask : List Msg → Out) (ρ : R) (κ : S.Chk) :
    (interact ask (S.nextD fill n wk x w ρ κ) (n x) []).take (n x - 1) =
      (interact ask (fun h => (((P w).next x h ρ).eval H).1) (n x) []).take (n x - 1) := by
  set pm : List (Msg × Msg) → Msg := fun h => (((P w).next x h ρ).eval H).1 with hpm
  rcases Nat.lt_or_ge (n x) 2 with hN | hN
  · rw [show n x - 1 = 0 by omega, List.take_zero, List.take_zero]
  · obtain ⟨j, hj⟩ : ∃ j, n x - 1 = j + 1 := ⟨n x - 2, by omega⟩
    rw [hj]
    refine interact_take_congr ask pm (S.nextD fill n wk x w ρ κ) (n x) [] j fun i hi hlt => ?_
    simp only [List.nil_append]
    have hl : (histOf ((interact ask pm (n x) []).take i)).length = i := by rw [length_histOf_take]; omega
    unfold nextD
    rw [ite_eq_left (Or.inl (by rw [hl]; omega))]
    exact (hRef ρ κ _ (S.ownHist_below H P fill n x w wk hRef ask ρ κ i (by omega)) (by rw [hl]; omega)).symm

/-- **Where the deferred server stops at its last round, it releases `P`'s run cut there**: if the run is not live at
its last round (the worker failed a round, or the server refuses that round), the release is `P`'s run with its last
message the stop, from `RefDBelow` alone. -/
theorem stop_deferred {W R : Type} (S : ZkServer W R) (H : Msg → Msg) (P : W → Prover R)
    (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val) (n : Msg → ℕ) (x : Msg) (w : W) (wk : S.Worker)
    (hRef : S.RefDBelow H P fill n wk x w) (ask : List Msg → Out) (ρ : R) (κ : S.Chk)
    (hd : S.live wk x w ρ κ
      (histOf ((interact ask (fun h => (((P w).next x h ρ).eval H).1) (n x) []).take (n x - 1))) = false) :
    interact ask (S.nextD fill n wk x w ρ κ) (n x) [] =
      cutT ask S.stop (n x - 1) (n x) (interact ask (fun h => (((P w).next x h ρ).eval H).1) (n x) []) := by
  classical
  set pm : List (Msg × Msg) → Msg := fun h => (((P w).next x h ρ).eval H).1 with hpm
  set N := n x with hN
  set τ := interact ask pm N [] with hτ
  rw [← interact_stopAt]
  refine interact_congr ask (stopAt pm (N - 1) S.stop) (S.nextD fill n wk x w ρ κ) N [] fun i hi => ?_
  simp only [List.nil_append]
  set σ := interact ask (stopAt pm (N - 1) S.stop) N [] with hσ
  have hσn : σ.length ≤ N := interact_length_le ask _ N []
  have hst : σ.take (N - 1) = τ.take (N - 1) := stopAt_take ask pm S.stop (N - 1) N
  have hgl : (histOf (σ.take i)).length = i := by rw [length_histOf_take]; omega
  by_cases hlast : i + 1 < N
  · have hσt : σ.take i = τ.take i := by
      have e1 : σ.take i = (σ.take (N - 1)).take i := by rw [List.take_take, Nat.min_eq_left (by omega)]
      have e2 : τ.take i = (τ.take (N - 1)).take i := by rw [List.take_take, Nat.min_eq_left (by omega)]
      rw [e1, hst, ← e2]
    rw [stopAt_of_lt (by rw [hgl]; omega)]
    unfold nextD
    rw [ite_eq_left (Or.inl (by rw [hgl]; exact hlast)), hσt]
    exact (hRef ρ κ _ (S.ownHist_below H P fill n x w wk hRef ask ρ κ i (by omega))
      (by rw [← hσt, hgl]; exact hlast)).symm
  · have hiN : i = N - 1 := by omega
    rw [stopAt_of_ge (by rw [hgl]; omega)]
    unfold nextD
    rw [ite_eq_right]
    rintro (h1 | h1)
    · rw [hgl] at h1; omega
    · rw [hiN, hst, hd] at h1; exact Bool.false_ne_true h1

end ZkServer

/-! ## The release at the unswitched prover -/

namespace LeanRelease

/-- **Where the frame refuses the coins, the deferred frame past the `Hello` is the stop**, whatever the worker
sends. -/
theorem frameD_of_refusesCoins {W R : Type} (L : LeanRelease W R)
    (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → L.toSrv.Val) (wk : L.toSrv.Worker) (x : Msg) (w : W)
    (ρ : R) (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length ≠ .own)
    (hr : L.refusesCoins x ρ h = true) : L.toSrv.frameD fill wk x w ρ κ h = L.stop := by
  cases hkk : L.kindAt x h.length with
  | own => exact absurd hkk hk
  | fixed => unfold kindAt at hkk; split_ifs at hkk
  | recomputed =>
    rw [L.frameD_recomp fill wk x w ρ κ h hkk]
    unfold ownMsg
    rw [ite_eq_left hr]
  | masked =>
    cases hl : L.toSrv.live wk x w ρ κ h with
    | true =>
      obtain ⟨v, hv, -, -, -⟩ := L.live_some wk x w ρ κ h hkk hl
      have hu : L.toSrv.wkD fill wk κ x w ρ h 0 = some (v, false) := by
        simp only [ZkServer.wkD, hl, ↓reduceIte, hv]
        rfl
      rw [L.frameD_toSrv fill wk x w ρ κ h hkk hu]
      exact L.release_of_refusesCoins x ρ h _ _ hr
    | false =>
      have hu : L.toSrv.wkD fill wk κ x w ρ h 0 = some (fill x h.length 0 w ρ h) := by
        simp only [ZkServer.wkD, hl, Bool.false_eq_true, ↓reduceIte]
      rw [L.frameD_toSrv fill wk x w ρ κ h hkk hu]
      exact L.release_of_refusesCoins x ρ h _ _ hr

/-- **The link is the session's unswitched link**: after `Hello`'s answer, where the protocol has coins and the frame
takes the answer's, the ZK server's own link is the session's unswitched message at the tape, the roots over the sampled
pads and the workers' values whatever they are. `LinkRoundEq` with the unswitched prover. -/
def LinkRoundEqStop (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (m₀ ρ₀ : Msg), 0 < refK x → L.refusesCoins x (ν, t) [(m₀, ρ₀)] = false →
    L.link x w (ν, t) [(m₀, ρ₀)] = S.unMsgAt x (S.wt x w) (vs x w ν) t []

/-- **The live frame rounds' bytes at the unswitched prover**: after every history of `0 < ops.length < refK x`
openings whose rounds all passed, whose openings pass, whose last coins the frame takes and whose messages are the
deferred release's own (`ZkServer.OwnHist`), the frame's release of the worker's round is the session's unswitched
message at the coins the openings open without its leading domain frame. `LiveRoundEq` with the unswitched prover,
asked at the release's own histories alone. -/
def LiveRoundEqStop (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    L.release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom (S.unMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2))

/-- **The stand-in frame rounds' bytes at the unswitched prover**: after every history of `0 < ops.length < refK x`
openings on which a round failed, whose openings pass, whose last coins the frame takes and whose messages are the
deferred release's own, the frame's release of the scheduled round, at the values the ZK server's transcript reports and
as a stopped rep's, is the session's unswitched message at the coins the openings open without its leading domain frame.
`FillRoundEq` with the unswitched prover, asked at the release's own histories alone. -/
def FillRoundEqStop (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = false →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) →
    L.release x (ν, t) ((m₀, ρ₀) :: ops) true
        (L.scheduled x w (ν, t) ((m₀, ρ₀) :: ops) (computedOf (L.computed x w (ν, t) ((m₀, ρ₀) :: ops)))) =
      dropDom (S.unMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2))

/-- **Where the rows hold, `LinkRoundEq` is `LinkRoundEqStop`.** -/
theorem linkRoundEqStop_of_rows (S : RefSession) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (x : Msg) (w : Flock.Wit)
    (hr : ∀ ν t, S.RowsAtTape x (S.pad []) (S.wt x w) (vs x w ν (S.pad [])) t) (hLk : LinkRoundEq S vs L x w) :
    LinkRoundEqStop S vs L x w := fun ν t m₀ ρ₀ hk hrf =>
  (hLk ν t m₀ ρ₀ hk hrf).trans (S.swMsgAt_of_rows x (S.wt x w) (vs x w ν) t [] hk (hr ν t))

/-- **Where the rows hold, `LiveRoundEq` gives `LiveRoundEqStop`.** -/
theorem liveRoundEqStop_of_rows (S : RefSession) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg)
    (w : Flock.Wit) (hr : ∀ ν t (cs : List S.Coin), S.RowsAtTape x (S.pad cs) (S.wt x w) (vs x w ν (S.pad cs)) t)
    (hE : LiveRoundEq S vs L H wk x w) : LiveRoundEqStop S vs L H wk x w :=
  fun ν t κ m₀ ρ₀ ops v h0 hlt hl hok hrf _ hv => (hE ν t κ m₀ ρ₀ ops v h0 hlt hl hok hrf hv).trans
    (congrArg dropDom (S.swMsgAt_of_rows x (S.wt x w) (vs x w ν) t _ (by simpa using hlt) (hr ν t _)))

/-- **Where the rows hold, `FillRoundEq` gives `FillRoundEqStop`.** -/
theorem fillRoundEqStop_of_rows (S : RefSession) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg)
    (w : Flock.Wit) (hr : ∀ ν t (cs : List S.Coin), S.RowsAtTape x (S.pad cs) (S.wt x w) (vs x w ν (S.pad cs)) t)
    (hE : FillRoundEq S vs L H wk x w) : FillRoundEqStop S vs L H wk x w :=
  fun ν t κ m₀ ρ₀ ops h0 hlt hl hok hrf _ => (hE ν t κ m₀ ρ₀ ops h0 hlt hl hok hrf).trans
    (congrArg dropDom (S.swMsgAt_of_rows x (S.wt x w) (vs x w ν) t _ (by simpa using hlt) (hr ν t _)))

/-- **The live frame rounds in frames at the unswitched prover**: at every live frame round `LiveRoundEqStop` reads,
the worker's round and the session's unswitched message are a frame round in frames of a rep that has not stopped
(`RoundFrames`). `LiveFrames` with the unswitched prover, at the release's own histories. -/
def LiveFramesStop (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    L.RoundFrames x (ν, t) ((m₀, ρ₀) :: ops) false v
      (S.unMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2))

/-- **`LiveRoundEqStop` from the live frame rounds in frames at the unswitched prover.** -/
theorem liveRoundEqStop_of_frames (S : RefSession) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg)
    (w : Flock.Wit) (hF : LiveFramesStop S vs L H wk x w) : LiveRoundEqStop S vs L H wk x w :=
  fun ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr ho hv =>
    L.release_of_roundFrames x _ _ false v _ (hF ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr ho hv)

/-- **The live frame rounds of a device at the unswitched prover**: at every live frame round `LiveRoundEqStop` reads,
the worker's round is the session's unswitched message with each masked value's pad added back (`RoundHonest`): the
worker's values `vs` on the sampled pads, whether or not its rows hold. `LiveHonest` with the unswitched prover, at the
release's own histories. -/
def LiveHonestStop (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    L.RoundHonest x (ν, t) ((m₀, ρ₀) :: ops) v (S.unMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2))

/-- **`LiveRoundEqStop` from a device's live frame rounds at the unswitched prover.** -/
theorem liveRoundEqStop_of_honest (S : RefSession) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg)
    (w : Flock.Wit) (hF : LiveHonestStop S vs L H wk x w) : LiveRoundEqStop S vs L H wk x w :=
  liveRoundEqStop_of_frames S vs L H wk x w fun ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr ho hv =>
    L.roundFrames_of_honest x _ _ v _ (hF ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr ho hv)

/-- **The stand-in frame rounds in frames at the unswitched prover**: at every stand-in frame round `FillRoundEqStop`
reads, the ZK server's scheduled round, at the values its transcript reports, and the session's unswitched message are
a frame round in frames of a stopped rep (`RoundFrames`). `FillFrames` with the unswitched prover, at the release's own
histories. -/
def FillFramesStop (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = false →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) →
    L.RoundFrames x (ν, t) ((m₀, ρ₀) :: ops) true
      (L.scheduled x w (ν, t) ((m₀, ρ₀) :: ops) (computedOf (L.computed x w (ν, t) ((m₀, ρ₀) :: ops))))
      (S.unMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2))

/-- **`FillRoundEqStop` from the stand-in frame rounds in frames at the unswitched prover.** -/
theorem fillRoundEqStop_of_frames (S : RefSession) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg)
    (w : Flock.Wit) (hF : FillFramesStop S vs L H wk x w) : FillRoundEqStop S vs L H wk x w :=
  fun ν t κ m₀ ρ₀ ops h0 hlt hl hok hr ho =>
    L.release_of_roundFrames x _ _ true _ _ (hF ν t κ m₀ ρ₀ ops h0 hlt hl hok hr ho)

/-- **The deferred release refines the unswitched prover below the `Finish`**: at the session with the frame's stops,
a last round `n x` one past the `Finish` and the `Finish` one past the last coin, on the deferred release's own
histories below the `Finish` its frame is the unswitched prover's message, from the server's `Hello` being the
session's (`HelloEq`), the stops being the frame's (`StopsAt`), the release refusing a failed opening
(`OpeningsRefused`), and the link and the worker's and stand-in frame rounds being the session's unswitched messages
(`LinkRoundEqStop`, `LiveRoundEqStop`, `FillRoundEqStop`). No premise reads the relation or the workers' rows. -/
theorem refDBelow_unP (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (n : Msg → ℕ) (wk : L.toSrv.Worker)
    (x : Msg) (w : Flock.Wit) (hn : n x = refK x + 2) (hlast : L.last x = refK x + 1) (hR : StopsAt S Rf L H x)
    (hO : OpeningsRefused S L H x) (hH : HelloEq S L x) (hLk : LinkRoundEqStop S vs L x w)
    (hE : LiveRoundEqStop S vs L H wk x w) (hEf : FillRoundEqStop S vs L H wk x w) :
    L.toSrv.RefDBelow H ((S.withStops Rf L.stop).unP vs).prover L.fillOf n wk x w := by
  intro ρ κ h hown hlt
  obtain ⟨ν, t⟩ := ρ
  cases h with
  | nil =>
    rw [L.frameD_own L.fillOf wk x w _ κ [] (kindAt_zero L x)]
    exact (S.withStops_unP_next_nil Rf L.stop vs H x w ν t).trans (hH ν t).symm
  | cons p ops =>
    obtain ⟨m₀, ρ₀⟩ := p
    have hlen : ops.length < refK x := by
      simp only [List.length_cons, hn] at hlt
      omega
    have hk0 : L.kindAt x ((m₀, ρ₀) :: ops).length ≠ .own := fun e => by
      have := kindAt_own e
      simp only [List.length_cons] at this
      omega
    refine (S.withStops_unP_next_cons Rf L.stop vs H x w ν t m₀ ρ₀ ops).trans ?_
    cases hr : L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) with
    | true =>
      rw [L.frameD_of_refusesCoins L.fillOf wk x w _ κ _ hk0 hr]
      split
      next hok =>
        exact S.unMsgAt_withStops_stop Rf L.stop x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2)
          ((hR ν t m₀ ρ₀ ops hlen.le hok).symm.trans hr)
      next => rfl
    | false =>
      have hok : ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true := by
        cases hb : ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 with
        | true => rfl
        | false =>
          rw [hO ν t m₀ ρ₀ ops hlen.le hb] at hr
          cases hr
      have hRf : Rf x (S.pad (ops.map fun p => S.decOf p.2)) (ops.map fun p => S.decOf p.2).length = false := by
        have e := (hR ν t m₀ ρ₀ ops hlen.le hok).symm.trans hr
        simpa only [List.length_map, Nat.min_eq_left hlen.le] using e
      rw [ite_eq_left hok]
      cases hk : L.kindAt x ((m₀, ρ₀) :: ops).length with
      | own => exact absurd hk hk0
      | fixed => unfold kindAt at hk; split_ifs at hk
      | recomputed =>
        rw [L.frameD_recomp L.fillOf wk x w _ κ _ hk]
        have hk' := kindAt_recomputed hk
        simp only [List.length_cons, hlast] at hk'
        have h0 : ops = [] := List.eq_nil_of_length_eq_zero (by omega)
        subst h0
        have hK : 0 < refK x := by simpa using hlen
        refine (S.unMsgAt_withStops_link Rf L.stop x (S.wt x w) (vs x w ν) t [] (by simpa using hRf) rfl
          (by simpa using hK)).trans ?_
        unfold ownMsg
        rw [hr]
        simp only [Bool.false_eq_true, ↓reduceIte, List.length_cons, List.length_nil, hlast]
        rw [ite_eq_right (by omega)]
        exact (hLk ν t m₀ ρ₀ hK hr).symm
      | masked =>
        obtain ⟨h0, -⟩ := ops_of_masked hlast hk
        refine (S.unMsgAt_withStops_frame Rf L.stop x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2) hRf
          (by simpa using h0) (by simpa using hlen)).trans ?_
        cases hl : L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) with
        | true =>
          obtain ⟨v, hv, -, -, -⟩ := L.live_some wk x w _ κ _ hk hl
          have hu : L.toSrv.wkD L.fillOf wk κ x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) := by
            simp only [ZkServer.wkD, hl, ↓reduceIte, hv]
            rfl
          rw [L.frameD_toSrv L.fillOf wk x w _ κ _ hk hu]
          exact (hE ν t κ m₀ ρ₀ ops v h0 hlen hl hok hr hown hv).symm
        | false =>
          have hu : L.toSrv.wkD L.fillOf wk κ x w (ν, t) ((m₀, ρ₀) :: ops) 0 =
              some (L.fillOf x ((m₀, ρ₀) :: ops).length 0 w (ν, t) ((m₀, ρ₀) :: ops)) := by
            simp only [ZkServer.wkD, hl, Bool.false_eq_true, ↓reduceIte]
          rw [L.frameD_toSrv L.fillOf wk x w _ κ _ hk hu]
          exact (hEf ν t κ m₀ ρ₀ ops h0 hlen hl hok hr hown).symm

/-- **The release before the `Finish` is the unswitched cut.** At the session with the frame's stops, for every
worker `wk` and any values `vs`, valid or not, the deferred release's messages and answers before the `Finish` (its
first `n x - 1`, `n x` one past the `Finish`) are the unswitched prover's run, against every verifier `ask`, at every
tape and check coins: the `Hello`, the link and every frame round, worker's or stand-in, committed over the sampled pads
(`refDBelow_unP`, `ZkServer.take_deferred`). -/
theorem release_preFinish (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (n : Msg → ℕ) (wk : L.toSrv.Worker)
    (x : Msg) (w : Flock.Wit) (hn : n x = refK x + 2) (hlast : L.last x = refK x + 1) (hR : StopsAt S Rf L H x)
    (hC : CheckIs S L H x) (hH : HelloEq S L x) (hLk : LinkRoundEqStop S vs L x w)
    (hE : LiveRoundEqStop S vs L H wk x w) (hEf : FillRoundEqStop S vs L H wk x w) (ask : List Msg → Out)
    (ρ : S.refP.Nonce × S.refP.Rand) (κ : L.toSrv.Chk) :
    (interact ask (L.toSrv.nextD L.fillOf n wk x w ρ κ) (n x) []).take (n x - 1) =
      (interact ask (fun h => (((((S.withStops Rf L.stop).unP vs).prover w).next x h ρ).eval H).1) (n x)
        []).take (n x - 1) :=
  L.toSrv.take_deferred H _ L.fillOf n x w wk
    (L.refDBelow_unP S Rf vs H n wk x w hn hlast hR (openingsRefused_of_checkIs S L H x hC) hH hLk hE hEf) ask ρ κ

/-- **On the stop branch the release is the unswitched cut, then the stop.** Where the session stops at its
`Finish` (the run is not live there: a round failed, or the ZK server's own `Finish` stops), the deferred release is
the unswitched prover's run with its last message the stop (`ZkServer.stop_deferred`). -/
theorem release_stop (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (n : Msg → ℕ) (wk : L.toSrv.Worker)
    (x : Msg) (w : Flock.Wit) (hn : n x = refK x + 2) (hlast : L.last x = refK x + 1) (hR : StopsAt S Rf L H x)
    (hC : CheckIs S L H x) (hH : HelloEq S L x) (hLk : LinkRoundEqStop S vs L x w)
    (hE : LiveRoundEqStop S vs L H wk x w) (hEf : FillRoundEqStop S vs L H wk x w) (ask : List Msg → Out)
    (ρ : S.refP.Nonce × S.refP.Rand) (κ : L.toSrv.Chk)
    (hd : L.toSrv.live wk x w ρ κ (histOf ((interact ask
      (fun h => (((((S.withStops Rf L.stop).unP vs).prover w).next x h ρ).eval H).1) (n x) []).take (n x - 1))) =
        false) :
    interact ask (L.toSrv.nextD L.fillOf n wk x w ρ κ) (n x) [] =
      cutT ask L.stop (n x - 1) (n x)
        (interact ask (fun h => (((((S.withStops Rf L.stop).unP vs).prover w).next x h ρ).eval H).1) (n x) []) :=
  L.toSrv.stop_deferred H _ L.fillOf n x w wk
    (L.refDBelow_unP S Rf vs H n wk x w hn hlast hR (openingsRefused_of_checkIs S L H x hC) hH hLk hE hEf) ask ρ κ hd

/-- **`release_stop` at a verifier's run**: against `V*` with `n x` rounds, on the stop branch the deferred release's
messages are the unswitched prover's run against `V*`, its last message the stop. -/
theorem release_stop_runStar {Θ : Type} (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool)
    (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (hn : n x = refK x + 2) (hlast : L.last x = refK x + 1)
    (hR : StopsAt S Rf L H x) (hC : CheckIs S L H x) (hH : HelloEq S L x) (hLk : LinkRoundEqStop S vs L x w)
    (hE : LiveRoundEqStop S vs L H wk x w) (hEf : FillRoundEqStop S vs L H wk x w) (Vs : VStar Θ) (θ : Θ)
    (hk : Vs.k x = n x) (ω : (S.refP.Nonce × S.refP.Rand) × L.toSrv.Chk)
    (hd : L.toSrv.live wk x w ω.1 ω.2
      (histOf ((runStar H (((S.withStops Rf L.stop).unP vs).prover w) Vs x ω.1 θ).msgs.take (n x - 1))) = false) :
    (runStar H (L.toSrv.proverD L.fillOf n wk w) Vs x ω θ).msgs =
      cutT (fun as => ((Vs.next x as θ).eval H).1) L.stop (n x - 1) (n x)
        (runStar H (((S.withStops Rf L.stop).unP vs).prover w) Vs x ω.1 θ).msgs := by
  have hP : (runStar H (((S.withStops Rf L.stop).unP vs).prover w) Vs x ω.1 θ).msgs =
      interact (fun as => ((Vs.next x as θ).eval H).1)
        (fun h => (((((S.withStops Rf L.stop).unP vs).prover w).next x h ω.1).eval H).1) (n x) [] := by
    rw [← hk]
    exact runStar_msgs H (((S.withStops Rf L.stop).unP vs).prover w) Vs x ω.1 θ
  have hsrv : (runStar H (L.toSrv.proverD L.fillOf n wk w) Vs x ω θ).msgs =
      interact (fun as => ((Vs.next x as θ).eval H).1) (L.toSrv.nextD L.fillOf n wk x w ω.1 ω.2) (n x) [] := by
    rw [runStar_msgs, hk]
    rfl
  rw [hsrv, hP]
  rw [hP] at hd
  exact L.release_stop S Rf vs H n wk x w hn hlast hR hC hH hLk hE hEf _ ω.1 ω.2 hd

/-- **`release_preFinish` at a verifier's run**: against `V*` with `n x` rounds, the deferred release's messages before
the `Finish` are the unswitched prover's run against `V*`, on either branch. -/
theorem release_preFinish_runStar {Θ : Type} (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool)
    (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (hn : n x = refK x + 2) (hlast : L.last x = refK x + 1)
    (hR : StopsAt S Rf L H x) (hC : CheckIs S L H x) (hH : HelloEq S L x) (hLk : LinkRoundEqStop S vs L x w)
    (hE : LiveRoundEqStop S vs L H wk x w) (hEf : FillRoundEqStop S vs L H wk x w) (Vs : VStar Θ) (θ : Θ)
    (hk : Vs.k x = n x) (ω : (S.refP.Nonce × S.refP.Rand) × L.toSrv.Chk) :
    (runStar H (L.toSrv.proverD L.fillOf n wk w) Vs x ω θ).msgs.take (n x - 1) =
      (runStar H (((S.withStops Rf L.stop).unP vs).prover w) Vs x ω.1 θ).msgs.take (n x - 1) := by
  have hP : (runStar H (((S.withStops Rf L.stop).unP vs).prover w) Vs x ω.1 θ).msgs =
      interact (fun as => ((Vs.next x as θ).eval H).1)
        (fun h => (((((S.withStops Rf L.stop).unP vs).prover w).next x h ω.1).eval H).1) (n x) [] := by
    rw [← hk]
    exact runStar_msgs H (((S.withStops Rf L.stop).unP vs).prover w) Vs x ω.1 θ
  have hsrv : (runStar H (L.toSrv.proverD L.fillOf n wk w) Vs x ω θ).msgs =
      interact (fun as => ((Vs.next x as θ).eval H).1) (L.toSrv.nextD L.fillOf n wk x w ω.1 ω.2) (n x) [] := by
    rw [runStar_msgs, hk]
    rfl
  rw [hsrv, hP]
  exact L.release_preFinish S Rf vs H n wk x w hn hlast hR hC hH hLk hE hEf _ ω.1 ω.2

end LeanRelease

end Claims.ZKServer
