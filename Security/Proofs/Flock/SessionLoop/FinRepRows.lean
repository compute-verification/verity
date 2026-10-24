import Proofs.Flock.SessionLoop.FinRepOpen
import Proofs.Flock.SessionLoop.FinRepXs
import Proofs.Flock.SessionLoop.Tables
import Proofs.Flock.Claims.ZK.FinishFwRep
import Proofs.Flock.Claims.ZK.FrameUnrefused

/-!
# `FinRepRows`: every table's rows hold at its base

At a `Finish` `FinishIs` reads, table `j`'s rows hold at its base (`FinRows`) where the firewall's run on each of its
reps fixes them (`Composed.rowsAt_of_fw`). The firewall's rep stream `2j + r` is `repFwC` answered by the opened coins
from the rep's first round (`Loop.finRepCut`), on the loop's table and request (`Loop.RepAt`), and `Masked.fwRows_at`
reads its rows off that run:

* the table, schedule and packed table are the session's (`Loop.finRepTab`, `Masked.schedsPZk_live`,
  `Loop.Level0Src.realOf`), the pads `h_ab = h_a·h_b` and `h_w = h_x·h_y` at the table's layout (`Loop.finRepPads`), and
  the request carries as many extra claims as the statement (`Loop.linkPoints_wordsAt`);
* no coin fault shows at the opened coins (`LeanRelease.frameUnrefused_ofLoop`, at a statement the loop serves).

What `fwRows_at` asks of the run beyond that is stated here, at the run (`Loop.RepRun`), as four facts the proof
consumes, none a premise of a guarantee:

* `Loop.FinRepMsgs`: below `zp 64`, the workers' clear values are the request's;
* `Loop.FinRepLanes`: level 0's out-of-domain value, lane messages and pair's sums are the masked values;
* `Loop.FinRepTarget`: the ring switch's `T` is the opening's (`Composed.openT`);
* `Loop.FinRepRestart`: level 0's restart is `Σ y₁'·bFold`.

**`Loop.finRepRows`**: in the masked scope, with a witness, at a statement the loop serves whose level 0 the source draws
from the tape (`Loop.Level0Src`), the four give `FinRepRows`.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge
open FlockSoundness.Refine (fOf gOf)
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

/-- Table `j` at the opened coins. -/
noncomputable abbrev tabAt (x : Msg) (ops : List (Msg × Msg)) (j : Fin (Masked.jOf x)) :=
  (Masked.tabC Masked.PE x j).T (Masked.czExe Masked.PE x (finCoins x ops) j)

/-- Table `j`'s randomness from the tape's base, its mask slot `.1` and extra lanes `.2.1.2.1`. -/
noncomputable abbrev randAtT (x : Msg) (ops : List (Msg × Msg)) (t : Masked.UE.Tape) (j : Fin (Masked.jOf x)) :=
  (tabAt x ops j).randOf ((Masked.UE.tape x t).1 j) 0

/-- A history at the rep's first round, as many entries long as the rounds before it. -/
def histAt (n : ℕ) : Array Msg := Array.replicate n default

theorem go_of_fst {α β : Type} {z : α × β} {a : α} (h : z.1 = a) : z = (a, z.2) := Prod.ext h rfl

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (real : Msg → Flock.Wit → Tp → ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **The firewall's rep stream `k` at a `Finish` `FinishIs` reads**: on the loop's table `tb` and request `q`
(`RepAt`), at the pads commitment `ic'` of the request's pads, `repFwC` answered by the opened coins from a history `l`
at the rep's first round returns the proof `p`, transcript `tr` and history `l'`. -/
def RepRun (ν : finS.Nonce) (t : finS.Tape) (κ : (LeanRelease.ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (s0 : FlockDeploy.Ins) (spec : SessionSpec) (k : ℕ) (tb : _root_.Flock.Firewall.Finish.Tab)
    (hi : Array ByteArray) (q : _root_.Flock.Firewall.Finish.RepIn) (f : ByteArray)
    (ic' : _root_.Flock.Firewall.Finish.InnerCommit) (l : Array Msg) (p : _root_.Flock.Zk.ZkProof)
    (tr : _root_.Flock.Transcript) (l' : Array Msg) : Prop :=
  FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops ∧
    FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) ∧
    RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec k tb hi q f ∧
    (_root_.Flock.Firewall.Finish.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
      (_root_.Flock.Firewall.Finish.padsOf q.draws) (_root_.Flock.Firewall.Finish.saltAt q.draws.salts) :
      V _root_.Flock.Firewall.Finish.InnerCommit) = .ok ic' ∧
    l.size = Masked.repBase x k ∧
    ((_root_.Flock.Firewall.Finish.repFwC
        (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
        (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) tb.st tb.sch tb.tPad (k % 2)
        (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 tb.g tb.cm0.toC
        (_root_.Flock.Firewall.Finish.claimsOf q.xs) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic'.toC q.vals
        tb.extra tb.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run
        (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l =
      (.ok (p, tr), l')

/-- **Consumed: the clear values below `zp 64` are the request's**: at table `j`'s rep `r`'s run, below the layout's
`zp 64` the workers' clear values `vs`, masked at the tape's mask slot and extra lanes, are the request's `q.vals`. -/
def FinRepMsgs : Prop :=
  ∀ ν t κ m₀ ρ₀ ops s0 spec (j : Fin (Masked.jOf x)) (r : Fin 2) tb hi q f ic' l p tr l',
    RepRun L₀ src real wk x w ν t κ m₀ ρ₀ ops s0 spec (2 * j.val + r.val) tb hi q f ic' l p tr l' →
    ∀ i < (Composed.layOf (Masked.zkTab x j).table (Masked.stAt x j)).zp 64,
      (Composed.msgAt ((tabAt x ops j).tab.onVals.vals r (Masked.wtOf Masked.PE x w j, vs x w ν (finCoins x ops) j)
        (randAtT x ops t j).1 (randAtT x ops t j).2.1.2.1) i).val = q.vals[i]!

/-- **Consumed: level 0's values are the masked values**: at table `j`'s rep `r`'s run, the returned proof's level 0
(its out-of-domain value, lane messages and pair's sums) is the workers' clear values plus the request's pads
(`Masked.LaneVals`). -/
def FinRepLanes : Prop :=
  ∀ ν t κ m₀ ρ₀ ops s0 spec (j : Fin (Masked.jOf x)) (r : Fin 2) tb hi q f ic' l p tr l',
    RepRun L₀ src real wk x w ν t κ m₀ ρ₀ ops s0 spec (2 * j.val + r.val) tb hi q f ic' l p tr l' →
    Masked.LaneVals (Masked.zkTab x j).table (Masked.stAt x j)
      (fun i => (tabAt x ops j).tab.onVals.vals r (Masked.wtOf Masked.PE x w j, vs x w ν (finCoins x ops) j)
        (randAtT x ops t j).1 (randAtT x ops t j).2.1.2.1 i +
        fOf (_root_.Flock.Firewall.Finish.padsOf q.draws).h[i.val]!)
      p.proof.ligerito p.level0

/-- **Consumed: the ring switch's `T` is the opening's**: at table `j`'s rep `r`'s run, wherever its zerocheck,
lincheck and ring switch return, the ring switch's `T` is `Composed.openT` at the views' `ν + maskHom`. -/
def FinRepTarget : Prop :=
  ∀ ν t κ m₀ ρ₀ ops s0 spec (j : Fin (Masked.jOf x)) (r : Fin 2) tb hi q f ic' l p tr l',
    RepRun L₀ src real wk x w ν t κ m₀ ρ₀ ops s0 spec (2 * j.val + r.val) tb hi q f ic' l p tr l' →
    ∀ zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
          tb.st.m tb.st.digest (FlockComplete.flat tb.cm0.toC.cap) ic'.toC.root
          (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0)
          fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).h[i]!).run
          (_root_.Flock.Transcript.init #[]
            (spec.domain spec.tables[(2 * j.val + r.val) / 2]! ((2 * j.val + r.val) % 2)).toUTF8)).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
          tb.st.fold tb.st.m (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0)
          (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).h[i]!) zc).run t₁).go l₁ =
        (.ok ((lp, lc), t₂), l₂) →
      ((FlockComplete.ringSwitchCut (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
          (fun k => _root_.Flock.Firewall.Finish.switchVec tb.g
            (_root_.Flock.abcClaims tb.st.m tb.st.kLog zc lc ++ _root_.Flock.Firewall.Finish.claimsOf q.xs)[k]!)
          (_root_.Flock.abcClaims tb.st.m tb.st.kLog zc lc ++ _root_.Flock.Firewall.Finish.claimsOf q.xs)).run t₂).go l₂ =
        (.ok ((ss, T, sws), t₃), l₃) →
      fOf T = Composed.openT (Masked.zkTab x j).table (Masked.stAt x j) (Masked.czExe Masked.PE x (finCoins x ops) j) r
        ((tabAt x ops j).tab.onVals.nu (Masked.wtOf Masked.PE x w j, vs x w ν (finCoins x ops) j) +
          (tabAt x ops j).tab.onVals.maskHom (randAtT x ops t j).1)

/-- **Consumed: level 0's restart is `Σ y₁'·bFold`**: at table `j`'s rep `r`'s run, the returned proof's level-0
restart `t'` is `Σ_col y₁'(col)·bFold(col)` at the workers' clear values and the tape's mask slot and extra lanes. -/
def FinRepRestart : Prop :=
  ∀ ν t κ m₀ ρ₀ ops s0 spec (j : Fin (Masked.jOf x)) (r : Fin 2) tb hi q f ic' l p tr l',
    RepRun L₀ src real wk x w ν t κ m₀ ρ₀ ops s0 spec (2 * j.val + r.val) tb hi q f ic' l p tr l' →
    gOf p.level0.tPrime = ∑ col, (tabAt x ops j).tab.onVals.y1'
      (Masked.wtOf Masked.PE x w j, vs x w ν (finCoins x ops) j) (randAtT x ops t j).1 (randAtT x ops t j).2.1.2.1 r col *
        Composed.bFold (Masked.zkTab x j).table (Masked.stAt x j) (Masked.czExe Masked.PE x (finCoins x ops) j) r col.val

/-- **Where the loop's `Finish` returns, the loop proved every rep**: at a `Finish` `FinishIs` reads, `filesE` returned
(`finStops`), so at the statement's setup its proofs did. -/
theorem proofsOf_ok_of_ctx {ν : finS.Nonce} {t : finS.Tape} {κ m₀ ρ₀ ops}
    (hctx : FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops) {s0 : FlockDeploy.Ins}
    {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) :
    ∃ files, proofsOf src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) = .ok files := by
  have hs := hctx.2.2.2.2.1
  have hs' : (filesE src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops)).isOk = true := by
    change (!(filesE src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops)).isOk) = false at hs
    simpa using hs
  cases hfe : filesE src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops) with
  | error e => rw [hfe] at hs'; cases hs'
  | ok fp =>
  unfold filesE at hfe
  rw [hsx] at hfe
  dsimp only at hfe
  cases hp : proofsOf src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) with
  | error e => rw [hp] at hfe; cases hfe
  | ok files => exact ⟨files, rfl⟩

/-- **`FinRepRows` from the four consumed facts**: in the masked scope, with a witness, at a statement the loop serves
(no coin fault shows at the opened coins, `frameUnrefused_ofLoop`) whose level 0 the source draws from the tape
(`Level0Src`), table `j`'s rows hold at its base: at each rep `r`, the loop's rep stream `2j + r` (`proofsOf_at`,
`proofFile_ok`) runs as `repFwC` at the opened coins (`finRepCut`) on the session's table, schedule and packed table
(`finRepTab`, `schedsPZk_live`, `Level0Src.realOf`) at pads of the table's layout (`finRepPads`) and as many extra
claims as the statement (`linkPoints_wordsAt`), so `Masked.fwRows_at` reads its rows off the run, and
`Composed.rowsAt_of_fw` decides them. -/
theorem finRepRows (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Loop.Serves x)
    (hS : Loop.Level0Src src x w) (hM : FinRepMsgs vs L₀ src real wk x w) (hV : FinRepLanes vs L₀ src real wk x w)
    (hT : FinRepTarget vs L₀ src real wk x w) (hTP : FinRepRestart vs L₀ src real wk x w) :
    FinRepRows vs L₀ src real wk x w := by
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  have hx : Masked.Live x := Masked.live_of hsc hR
  intro ν t κ m₀ ρ₀ ops hctx j
  have hj : j.val < (Masked.stsOf x).size := Masked.lt_of_jOf j
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨-, -, hTs, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hnr : ¬ Masked.Refused x (finCoins x ops) (refK x) :=
    LeanRelease.frameUnrefused_ofLoop Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real
      CoinOpen.shaH wk x w hServe hx ν t κ m₀ ρ₀ ops hctx.1 hctx.2.1 hctx.2.2.1 hctx.2.2.2.1
  obtain ⟨files, hp⟩ := proofsOf_ok_of_ctx L₀ src real wk x w hctx hsx
  obtain ⟨schs, hschs, hschZ⟩ := Masked.schedsPZk_live hx
  refine Composed.rowsAt_of_fw Masked.PE (Masked.zkTab x j).table (Masked.stAt x j)
    (Masked.czExe Masked.PE x (finCoins x ops) j) (Masked.wtOf Masked.PE x w j, vs x w ν (finCoins x ops) j)
    (randAtT x ops t j).1 (randAtT x ops t j).2.1.2.1 fun r => ?_
  -- rep stream `2j + r`
  have hk : 2 * j.val + r.val < 2 * spec.tables.size := by
    have := r.isLt
    omega
  have hk2 : (2 * j.val + r.val) / 2 = j.val := by
    have := r.isLt
    omega
  have hkr : (2 * j.val + r.val) % 2 = r.val := by
    have := r.isLt
    omega
  have hrs : spec.repStreams.size = 2 * spec.tables.size := FlockDeploy.repStreams_size spec
  obtain ⟨tb, hi, q, hRep⟩ := proofsOf_at hp (k := 2 * j.val + r.val) (by rw [hrs]; exact hk)
  obtain ⟨ic', p, tr, hic', hrun, -⟩ := _root_.Flock.Firewall.Finish.proofFile_ok hRep.2.2
  have hl : (histAt (Masked.repBase x (2 * j.val + r.val))).size = Masked.repBase x (2 * j.val + r.val) :=
    Array.size_replicate
  have hgo := go_of_fst (finRepCut L₀ src real wk x w hsc ν t κ m₀ ρ₀ ops hctx s0 spec hsx _ hk tb hi q _ hRep ic' p tr
    hic' hrun _ hl)
  have hrr : RepRun L₀ src real wk x w ν t κ m₀ ρ₀ ops s0 spec (2 * j.val + r.val) tb hi q _ ic' _ p tr _ :=
    ⟨hctx, hsx, hRep, hic', hl, hgo⟩
  have hM' := hM ν t κ m₀ ρ₀ ops s0 spec j r tb hi q _ ic' _ p tr _ hrr
  have hV' := hV ν t κ m₀ ρ₀ ops s0 spec j r tb hi q _ ic' _ p tr _ hrr
  have hT' := hT ν t κ m₀ ρ₀ ops s0 spec j r tb hi q _ ic' _ p tr _ hrr
  have hTP' := hTP ν t κ m₀ ρ₀ ops s0 spec j r tb hi q _ ic' _ p tr _ hrr
  -- the session's table, schedule and packed table
  obtain ⟨htb, hsb⟩ := finRepTab (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hschs hk hRep
  rw [hk2] at htb hsb
  have hst : tb.st = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) := htb.symm.trans (Masked.tabsOf_stAt x j)
  have hsch : tb.sch = (Masked.zkTab x j).sch := (Prod.mk.inj (hsb.symm.trans (hschZ j))).1
  obtain ⟨-, -, -, hg⟩ := hS.realOf ν t j r hRep.1
  -- the pads at the table's layout
  obtain ⟨-, hps, hab, htw⟩ := finRepPads L₀ src real wk x w hsc ν t κ m₀ ρ₀ ops hctx s0 spec hsx _ hk tb hi q _ hRep
  have hlay : _root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0 =
      Composed.layOf (Masked.zkTab x j).table (Masked.stAt x j) := by
    rw [hst, hsch]
    rfl
  rw [hlay] at hps hab htw
  -- as many extra claims as the statement
  have hxsz : (_root_.Flock.Firewall.Finish.claimsOf q.xs).size = spec.points * (Masked.stAt x j).regions.size := by
    have hq := hRep.2.1
    unfold repInOf at hq
    obtain ⟨⟨lay, ms⟩, -, hq⟩ := bok hq
    have hqx : q.xs = _root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
          (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0)) := by
      cases hq
      rfl
    rw [linkPoints_wordsAt x w hsc hR hctx.1 hsx hk m₀ ρ₀, hk2,
      show (Masked.stsOf x)[j.val]! = Masked.stAt x j from getElem!_pos (Masked.stsOf x) j.val hj] at hqx
    simp only [_root_.Flock.Firewall.Finish.claimsOf, Array.size_map]
    rw [hqx]
    unfold _root_.Flock.Firewall.Drive.extras
    rw [flatMap_size_const _ _ _ fun _ => Array.size_map]
    simp only [viewPoints, Array.size_map, Array.size_range]
    exact Nat.mul_comm _ _
  -- the run at table `j`'s rep `r`
  rw [hkr, hk2, hsch] at hgo
  rw [hkr, hk2, hsch] at hT'
  have ht₀ : (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8).rounds = #[] := rfl
  exact Masked.fwRows_at x hx w hR t (finCoins x ops) (refK x) hnr s0 spec hsx j r (vs x w ν (finCoins x ops))
    (_root_.Flock.Firewall.Finish.claimsOf q.xs) hxsz tb.st hst tb.tPad _ tb.g hg tb.cm0.toC
    (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic'.toC q.vals tb.extra tb.lanePad _ hps hab htw ht₀ hl hgo
    hM' hV' hT' hTP'

end Loop

end Claims.ZKServer
