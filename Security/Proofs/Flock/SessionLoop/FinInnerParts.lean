import Proofs.Flock.SessionLoop.FinInner

/-!
# The loop's finish at a rep's inner rounds on the worker's own histories, in parts

`LeanRelease.finOf_finalC_own` and `LeanRelease.finOf_tau_own` read the rounds the loop's finish hands a rep's frame
(`Loop.finOf`) at a history `LiveInnerOwn` reads at rep `2j + r`'s inner round `e` (`Loop.InnerHist`), at the release
`withInner` on those rounds (`Loop.relIn`). Each part is named here, none a premise of a guarantee:

* **`Loop.innerHist_fin`**: there the loop's table and request for rep stream `2j + r` return; the table's pad layout
  and the round's configuration are table `j`'s (`Masked.layE`) and the configuration's pads the tape's; and where the
  finish's live inner proof (`finInner`) returns at the request with `final_c = a` (`askOf`) and at no other `final_c`,
  `finOf` is its rounds. The history's next round is the stream's round `nSched + e`, past its scheduled rounds
  (`innerAt_slot`).
* **`Loop.innerReq_own`**: the request's values below `zp 64` but `final_c`'s are `vs`'s (`vsWord`), and its pads the
  tape's. The history's rounds of the rep's PIOP are the frame's releases of the worker's rounds (`OwnHist`), which carry
  `vs`'s values under the pads (`TakesValsOwn`).
* **`Loop.innerReq_coins`**: the request's coins of the rep's rounds before the history's next are the opened coins from
  the rep's first round (`Masked.repBase`).
* **`Loop.finInner_fc`**: where `finInner` returns, its `final_c'` is the request's `final_c` under its pad
  (`Loop.repCons_rets`, over `Loop.Rets`: what a run that returns returns).
* **`Loop.finInner_accepts_own`**: `finInner` returns at the request with `final_c = a` only at `vs`'s `final_c`: its
  `zerocheck c_eval` constraint holds on the pads (`consWhy`), which at the request's round-1c values and the opened
  coins' zerocheck point fixes `a` (`Loop.FinalCFixed`).
* **`Loop.finInner_returns_own`**: it returns there.
* **`Loop.finInner_tau_own`**: at the inner proof's round 2, its `τ` commitment is `vs`'s (`Loop.vsTau`), where the rows
  hold (`Loop.RowsOK`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec TMW)
open _root_.Flock.Firewall.Frame (INNER_ROUNDS)
open _root_.Flock.Firewall.Finish (Tab RepIn Asks)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Loop

/-- **The session loop's release at its finish's inner rounds**: `withInner` on `finOf` at `L₁`'s frame draws, with the
loop's own link and `Finish` (`ofLoopLink`). -/
noncomputable abbrev relIn (L₁ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray)) : LeanRelease Flock.Wit Tp :=
  LeanRelease.ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src real

/-- **A history `LiveInnerOwn` reads at rep `2j + r`'s inner round `e`**, at the release `L`: after `Hello` `(m₀, ρ₀)`
and `0 < ops.length < refK x` openings, live, whose openings pass, no coin refused, the release's own run
(`ZkServer.OwnHist`), at which the worker sends a message, whose next round is an inner proof's (`InnerAt`), rep
`2j + r`'s round `innerOff e`. -/
def InnerHist (L : LeanRelease Flock.Wit Tp) (wk : L.toSrv.Worker)
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray)) (x : Msg) (w : Flock.Wit) (ν : finS.Nonce)
    (t : finS.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (e : ℕ) : Prop :=
  0 < ops.length ∧ ops.length < refK x ∧ L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true ∧
    ((allOk (finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true ∧
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false ∧
    L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) ∧
    (∃ v, wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) ∧
    InnerAt (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) ∧
    L.rep x (ops.length + 1) = 2 * j.val + r.val ∧
    ops.length = Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff e)

theorem v_bind_ok {α β : Type} {x : V α} {f : α → V β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-- **A rep's inner rounds are its last**: at a `Live` statement, rep stream `k` has at least the inner proof's rounds,
and its scheduled ones (`nSched`) end at its inner round 0 (`Masked.repSlot … (Masked.innerOff 0)`). -/
theorem repSlot_nSched {x : Msg} (hx : Masked.Live x) {s0 : FlockDeploy.Ins} {spec : SessionSpec}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) {k : ℕ} (hk : k < spec.repStreams.size) :
    INNER_ROUNDS ≤ (roundsOf spec (Masked.stsOf x)).getD k 0 ∧
      Masked.repBase x k + nSched spec (Masked.stsOf x) k = Masked.repSlot x k (Masked.innerOff 0) := by
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hrsz : spec.repStreams.size = 2 * (Masked.stsOf x).size := by rw [FlockDeploy.repStreams_size, hspec]
  have hj : k / 2 < (Masked.stsOf x).size := by omega
  obtain ⟨y⟩ := hx.2 (k / 2) hj
  have hst : (Masked.stsOf x)[k / 2]! = (Masked.stsOf x)[k / 2] := getElem!_pos _ (k / 2) hj
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[k / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]
    exact y.hz
  have hk6 : 6 ≤ (Masked.stsOf x)[k / 2]!.c.kLog := by
    rw [hst]
    exact y.wf.1.hk6
  have hrq : (roundsOf spec (Masked.stsOf x)).getD k 0 =
      FlockComplete.piopRoundsZk spec (Masked.stsOf x)[k / 2]! + (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) + 4 := by
    simp only [roundsOf, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hk, Option.map_some,
      Option.getD_some]
    have e1 : (Masked.stsOf x)[k / 2]!.m + (Masked.stsOf x)[k / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[k / 2]!.m + ((Masked.stsOf x)[k / 2]!.c.kLog - 6) := by omega
    rw [repRounds_true]
    simp only [hz]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
  have hsl : Masked.repSlot x k (Masked.innerOff 0) = FlockComplete.repStartZk spec (Masked.stsOf x) k +
      FlockComplete.piopRoundsZk spec (Masked.stsOf x)[k / 2]! + (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1) + 0) := by
    unfold Masked.repSlot
    rw [hs]
    simp only [hz]
    rfl
  rw [Masked.repBase_eq hs, hsl]
  unfold nSched
  rw [hrq]
  have hI : INNER_ROUNDS = 4 := rfl
  refine ⟨by omega, by omega⟩

/-- **Rep stream `k`'s first round is past the link**: its first rep round (`streamAt`) is one before its first session
round (`Masked.repBase`). -/
theorem streamAt_repBase {x : Msg} {s0 : FlockDeploy.Ins} {spec : SessionSpec}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) {k : ℕ} (hk : k < spec.repStreams.size) :
    streamAt (roundsOf spec (Masked.stsOf x)) k + 1 = Masked.repBase x k := by
  rw [Masked.repBase_eq hs k]
  unfold streamAt roundsOf FlockComplete.repStartZk
  rw [take_map_range _ (by omega)]
  omega

variable (L₁ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray))

/-- **The finish at a rep's inner round**: at a history `InnerHist` holds at with `e < 4`, on the loop's own tables at a
statement in the masked scope with a witness that the loop serves, the loop's table and request for rep stream `2j + r`
return; the table's pad layout and the round's configuration are table `j`'s, the configuration's pads the tape's
(`Loop.RepSrc`); and where `finInner` on them returns at `final_c = a` after the stream's `nSched + e` rounds of coins,
and at no other `final_c`, `finOf` is its round 0 and round 2. -/
theorem innerHist_fin (hreal : real = realOf src) (x : Msg) (w : Flock.Wit)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Src (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src x w)
    (wk : (relIn L₁ src real).toSrv.Worker) {ν : finS.Nonce} {t : finS.Tape} {κ : (relIn L₁ src real).toSrv.Chk}
    {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)} {j : Fin (Masked.jOf x)} {r : Fin 2} {e : ℕ} (he : e < 4)
    (hH : InnerHist (relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r e) :
    ∃ (s0 : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt) (tr : Tab) (hi : Array ByteArray) (q₀ : RepIn),
      FlockDeploy.setupOf true x = some (s0, spec, sts) ∧
      real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hi) ∧
      repInOf x ((m₀, ρ₀) :: ops) spec sts (2 * j.val + r.val) (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀ ∧
      _root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0 = Masked.layE x j ∧
      ((relIn L₁ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay = Masked.layE x j ∧
      ((relIn L₁ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).h =
        (FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j)).h ∧
      ∀ (a fc' : F128) (tc : ByteArray),
        finInner tr ((2 * j.val + r.val) % 2) (domOf spec (2 * j.val + r.val)) (nSched spec sts (2 * j.val + r.val))
          (askOf q₀ (Masked.layE x j).fc (nSched spec sts (2 * j.val + r.val) + e) a) = .ok (fc', tc) →
        (∀ a', (finInner tr ((2 * j.val + r.val) % 2) (domOf spec (2 * j.val + r.val))
          (nSched spec sts (2 * j.val + r.val))
          (askOf q₀ (Masked.layE x j).fc (nSched spec sts (2 * j.val + r.val) + e) a')).isOk = true → a' = a) →
        finOf real L₁.draws x w (ν, t) ((m₀, ρ₀) :: ops) = #[inner0 fc'.toBytes, .empty, inner2 tc, .empty] := by
  obtain ⟨h0, hlt, hl, -, -, -, ⟨v, hv⟩, hI, hrep, hslot⟩ := hH
  have hx : Masked.Live x := Masked.live_of hsc hR
  obtain ⟨s0, spec, sts, q, k, i, tr, hi, rsched, hxs, hq, hloc, hrk, hrs, hle⟩ := hI
  have hq1 : ops.length = q + 1 := by rw [← hq, List.length_map]
  have hsts : Masked.stsOf x = sts := by
    unfold Masked.stsOf
    rw [hxs]
  subst hsts
  have hRT : Count.RealTabs (Masked.stsOf x) (real x w (ν, t)) := by
    subst hreal
    exact realTabs_of src x w (ν, t) hxs
  obtain ⟨-, -, -, -, -, -, -, hi0, -, -⟩ := innerAt_slot hx hSv hxs hq
    (by rw [List.length_map]; exact hlt) hloc hRT hrk hrs hle
  -- the history's rep is the stream's, `2j + r`
  have hrep' : (relIn L₁ src real).rep x (ops.length + 1) = k := by
    have e : (relIn L₁ src real).rep x (ops.length + 1) = (CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).1 := by
      show Loop.repOf x (ops.length + 1) = _
      rw [show ops.length + 1 = q + 2 by omega, LeanRelease.repOf_add_two hxs]
    rw [e, hloc]
  have hkj : k = 2 * j.val + r.val := hrep'.symm.trans hrep
  subst hkj
  have hr2 := r.isLt
  have hjs := Masked.lt_of_jOf j
  have hk2 : (2 * j.val + r.val) / 2 = j.val := by omega
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hxs
  have hrsz : spec.repStreams.size = 2 * (Masked.stsOf x).size := by rw [FlockDeploy.repStreams_size, hspec]
  have hkk : 2 * j.val + r.val < spec.repStreams.size := by omega
  -- the round is the stream's `nSched + e`
  obtain ⟨-, hbase⟩ := repSlot_nSched hx hxs hkk
  have hstream := streamAt_repBase hxs hkk
  unfold streamAt at hstream
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  have hsum : (roundsOf spec (Masked.stsOf x)).sum + 1 = refK x := by
    rw [Masked.refK_eq hxs, ← FlockComplete.ZkTables.repStart_last, ← hrsz]
    unfold FlockComplete.repStartZk roundsOf
    omega
  have hinv := locate_inv (roundsOf spec (Masked.stsOf x)) 0 q (by omega)
  rw [hloc] at hinv
  simp only [Nat.sub_zero] at hinv
  obtain ⟨hqs, -⟩ := hinv
  have hoff := Masked.repSlot_innerOff hxs j r e
  have hie : i = nSched spec (Masked.stsOf x) (2 * j.val + r.val) + e := by omega
  -- the table's layout is table `j`'s
  obtain ⟨g, d, l, -, htr⟩ := hRT _ tr hi hrk
  obtain ⟨hts, hsch⟩ := Count.tabOfLevel0_sched htr
  have hst' : (Masked.stsOf x)[j.val]! = Masked.stAt x j := getElem!_pos (Masked.stsOf x) j.val hjs
  have hsc' : _root_.Flock.Zk.schedule (Masked.stAt x j).m = .ok (tr.sch, tr.tPad) := by
    have e : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m = .ok (tr.sch, tr.tPad) := hsch
    rwa [hk2, hst'] at e
  rw [(Masked.zkTab x j).hz] at hsc'
  have htsch : tr.sch = (Masked.zkTab x j).sch := (Prod.mk.inj (Except.ok.inj hsc')).1.symm
  have htst : tr.st = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) := by
    rw [hts, hk2, hst']
  have hLayT : _root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0 = Masked.layE x j := by
    rw [htst, htsch]
    rfl
  -- the rep's frame starts, so the request returns
  have hlast : (relIn L₁ src real).last x = refK x + 1 := LeanRelease.loop_last loopNr Masked.dataE _ x
  have hkd : (relIn L₁ src real).kindAt x ((m₀, ρ₀) :: ops).length = .masked :=
    LeanRelease.kindAt_masked_of _ (by rw [List.length_cons]; omega) (by rw [hlast, List.length_cons]; omega)
  obtain ⟨m, -, -, ha, -⟩ :=
    LeanRelease.check_of_live (relIn L₁ src real) wk x w (ν, t) κ ((m₀, ρ₀) :: ops) v hl hkd hv
  have hloc' : CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q = (2 * j.val + r.val, i - 1 + 1) := by
    rw [Nat.sub_add_cancel hi0]
    exact hloc
  obtain ⟨lay, mask, hst0⟩ := LeanRelease.ofLoop_start_at Masked.dataE
    (LeanRelease.withFiles (LeanRelease.withLink (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src)
      src) real x (ν, t) (m₀, ρ₀) ops ha rfl hxs hq1 hloc'
  obtain ⟨q₀, hq₀⟩ : ∃ q₀, repInOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) (2 * j.val + r.val)
      (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀ := by
    unfold repInOf
    simp only [hst0]
    exact ⟨_, rfl⟩
  -- the configuration's layout and pads are the tape's
  have hst := hst0
  rw [hk2] at hst
  have hlay := LeanRelease.lay_ofLoopLink (L₀ := LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src
    real j r hst
  have hd := LeanRelease.draws_ofLoopLink (L₀ := LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src
    real
  have hz := hSrc.2.size (relIn L₁ src real) hd ν t j r hlay
  have hpads := hSrc.2.pads (relIn L₁ src real) hd ν t j r hlay (Masked.UE.tapeZk t)
  obtain ⟨hHc, -, -⟩ := cfg_tapeOfDraws (relIn L₁ src real) x (ν, t) ((m₀, ρ₀) :: ops) (Masked.UE.tapeZk t)
    Loop.frepOf j.val r.val hrep hz
  rw [hlay, ← hpads] at hHc
  have hLay : ((relIn L₁ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay = Masked.layE x j := by
    show (relIn L₁ src real).lay x ((relIn L₁ src real).rep x (ops.length + 1)) = _
    rw [hrep]
    exact hlay
  refine ⟨s0, spec, Masked.stsOf x, tr, hi, q₀, hxs, hrk, hq₀, hLayT, hLay, hHc, ?_⟩
  -- `finOf` is `finInner` at the `final_c` it accepts
  intro a fc' tc hfi huniq
  have hok : (finInner tr ((2 * j.val + r.val) % 2) (domOf spec (2 * j.val + r.val))
      (nSched spec (Masked.stsOf x) (2 * j.val + r.val))
      (askOf q₀ (Masked.layE x j).fc (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + e) a)).isOk = true := by
    rw [hfi] <;> rfl
  unfold finOf
  simp only [hxs, List.tail_cons, hq1]
  generalize CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q = L at hloc ⊢
  subst hloc hie
  simp only [hrk, hq₀]
  rw [hLayT]
  split
  · rename_i hA
    rw [huniq _ hA.choose_spec, hfi] <;> rfl
  · rename_i hn
    exact (hn ⟨a, hok⟩).elim

/-- **The loop's request at a rep's inner round, on the worker's own history**: at a history `InnerHist` holds at, where
the worker takes `vs`'s values at its own histories (`TakesValsOwn`) and `vs` is causal, the request for rep stream
`2j + r` (`repInOf`) has `zp 64` values, `vs`'s below it but at `final_c`'s pad (`vsWord`, at the coins the openings
open), and the tape's pads. -/
theorem innerReq_own (vs : finS.SwVals) (hreal : real = realOf src) (x : Msg) (w : Flock.Wit)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Src (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src x w)
    (hcaus : finS.SwCausal vs) (wk : (relIn L₁ src real).toSrv.Worker)
    (hV : TakesValsOwn vs (relIn L₁ src real) wk x w) {ν : finS.Nonce} {t : finS.Tape}
    {κ : (relIn L₁ src real).toSrv.Chk} {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)} {j : Fin (Masked.jOf x)} {r : Fin 2}
    {e : ℕ} (he : e < 4) (hH : InnerHist (relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r e)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {tr : Tab} {hi : Array ByteArray} {q₀ : RepIn}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hr : real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hi))
    (hq : repInOf x ((m₀, ρ₀) :: ops) spec sts (2 * j.val + r.val) (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀) :
    q₀.vals.size = (Masked.layE x j).zp 64 ∧
      (∀ i < (Masked.layE x j).zp 64, i ≠ (Masked.layE x j).fc →
        q₀.vals[i]! = vsWord vs x w ν (finS.pad (ops.map fun p => finS.decOf p.2)) t j r i) ∧
      (_root_.Flock.Firewall.Finish.padsOf q₀.draws).toC =
        FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j) := by
  sorry

/-- **The request's coins** at a history `InnerHist` holds at: it holds a round of coins for each of rep stream
`2j + r`'s rounds, and below each round's count its rounds before the history's next, `nSched + e` of them, are the
opened coins from the rep's first round (`Masked.repBase`). As `finRepCoins` at a `Finish`, below the history's
length. -/
theorem innerReq_coins (x : Msg) (w : Flock.Wit)
    (wk : (relIn L₁ src real).toSrv.Worker) {ν : finS.Nonce} {t : finS.Tape} {κ : (relIn L₁ src real).toSrv.Chk}
    {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)} {j : Fin (Masked.jOf x)} {r : Fin 2} {e : ℕ} (he : e < 4)
    (hH : InnerHist (relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r e)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {hi : Array ByteArray} {q₀ : RepIn}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hq : repInOf x ((m₀, ρ₀) :: ops) spec sts (2 * j.val + r.val) (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀) :
    q₀.coins.size = (roundsOf spec sts).getD (2 * j.val + r.val) 0 ∧
      nSched spec sts (2 * j.val + r.val) + e < q₀.coins.size ∧
      ∀ k' < nSched spec sts (2 * j.val + r.val) + e, ∀ i < (q₀.coins[k']!).size,
        Masked.slotAt (finCoins x ops) (Masked.repBase x (2 * j.val + r.val) + k') i = (q₀.coins[k']!)[i]! := by
  obtain ⟨-, hlt, -, -, -, -, -, -, -, hslot⟩ := hH
  have hx := Masked.live_of_lt j
  have hsts : Masked.stsOf x = sts := by
    unfold Masked.stsOf
    rw [hs]
  subst hsts
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hrsz : spec.repStreams.size = 2 * (Masked.stsOf x).size := by rw [FlockDeploy.repStreams_size, hspec]
  have hjs := Masked.lt_of_jOf j
  have hr2 := r.isLt
  rw [Masked.repSlot_innerOff hs j r e] at hslot
  generalize hk2 : 2 * j.val + r.val = k at hslot hq ⊢
  have hk : k < spec.repStreams.size := by omega
  obtain ⟨h4, hbase⟩ := repSlot_nSched hx hs hk
  rw [← hbase] at hslot
  have hI : INNER_ROUNDS = 4 := rfl
  unfold repInOf at hq
  obtain ⟨⟨lay, ms⟩, -, hq⟩ := v_bind_ok hq
  have hqc : q₀.coins =
      ((seenRounds x ((m₀, ρ₀) :: ops) (roundsOf spec (Masked.stsOf x)) k).map fun r => r.2.2).toArray := by
    cases hq
    rfl
  have hsz : q₀.coins.size = (roundsOf spec (Masked.stsOf x)).getD k 0 := by
    rw [hqc, List.size_toArray, List.length_map]
    simp only [seenRounds, List.length_map, List.length_range]
  have hstream := streamAt_repBase hs hk
  have hns : nSched spec (Masked.stsOf x) k + INNER_ROUNDS = (roundsOf spec (Masked.stsOf x)).getD k 0 := by
    unfold nSched
    omega
  refine ⟨hsz, by omega, fun k' hk' i hi => ?_⟩
  have hkc : k' < q₀.coins.size := by omega
  have hQ : Masked.repBase x k + k' < ops.length := by omega
  have hQK : Masked.repBase x k + k' < refK x := by omega
  have hent : q₀.coins[k']! = wordsAt x (ops.map fun p => CoinLayer.rd p.2) (Masked.repBase x k + k') := by
    rw [getElem!_pos q₀.coins k' hkc]
    simp only [hqc, List.getElem_toArray, List.getElem_map]
    unfold seenRounds
    simp only [List.getElem_map, List.getElem_range, List.tail_cons]
    congr 1
    omega
  have hw : wordsAt x (ops.map fun p => CoinLayer.rd p.2) (Masked.repBase x k + k') =
      (CoinLayer.rd ops[Masked.repBase x k + k'].2).vals
        (min (loopNr x (Masked.repBase x k + k')) _root_.Flock.Firewall.Frames.COIN_MAX) := by
    unfold wordsAt
    rw [List.getElem?_map, List.getElem?_eq_getElem hQ]
    rfl
  rw [hent, hw] at hi ⊢
  rw [getElem!_pos _ i hi]
  simp only [FlockDeploy.Coin.vals, Array.getElem_map, Array.getElem_range]
  unfold Masked.slotAt
  rw [dif_pos hQK]
  show ((ops.map fun p => CoinLayer.rd p.2).getD (Masked.repBase x k + k') default).at i = _
  rw [List.getD_eq_getElem _ _ (by simpa using hQ), List.getElem_map]

/-- **`x` returns only values `P` holds of**: from every transcript and round, a run of `x` that returns, returns one. -/
def Rets {α : Type} (P : α → Prop) (x : TMW (StateT ℕ V) α) : Prop :=
  ∀ tr k a tr' k', (x.run tr).run k = .ok ((a, tr'), k') → P a

theorem rets_pure {α : Type} {P : α → Prop} {a : α} (h : P a) : Rets P (pure a) := by
  intro tr k b tr' k' e
  cases e
  exact h

theorem rets_throw {α : Type} {P : α → Prop} (e : String) : Rets P (throw e : TMW (StateT ℕ V) α) := by
  intro tr k b tr' k' h
  cases h

theorem rets_bind {α β : Type} {P : β → Prop} {x : TMW (StateT ℕ V) α} {f : α → TMW (StateT ℕ V) β}
    (h : ∀ a, Rets P (f a)) : Rets P (x >>= f) := by
  intro tr k b tr' k' e
  obtain ⟨a, tr₁, k₁, -, e2⟩ := _root_.Flock.Firewall.Finish.tmw_bind_ok e
  exact h a tr₁ k₁ b tr' k' e2

theorem rets_bind' {α β : Type} {P : α → Prop} {Q : β → Prop} {x : TMW (StateT ℕ V) α}
    {f : α → TMW (StateT ℕ V) β} (hx : Rets P x) (h : ∀ a, P a → Rets Q (f a)) : Rets Q (x >>= f) := by
  intro tr k b tr' k' e
  obtain ⟨a, tr₁, k₁, e1, e2⟩ := _root_.Flock.Firewall.Finish.tmw_bind_ok e
  exact h a (hx tr k a tr₁ k₁ e1) tr₁ k₁ b tr' k' e2

/-- The masked zerocheck's proof carries the masked value at `final_c`'s pad. -/
theorem zerocheckFw_rets (io : Asks (StateT ℕ V)) (m : ℕ) (digest cap root : ByteArray)
    (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) :
    Rets (fun r => r.1.finalC = mk lay.fc) (_root_.Flock.Firewall.Finish.zerocheckFw io m digest cap root lay mk) := by
  unfold _root_.Flock.Firewall.Finish.zerocheckFw
  repeat (first | refine rets_bind fun _ => ?_ | exact rets_pure rfl)

/-- A rep's PIOP's zerocheck proof carries the masked value at `final_c`'s pad. -/
theorem repPiop_rets (io : Asks (StateT ℕ V)) (st : _root_.Flock.Setup) (lay : _root_.Flock.Zk.PadLayout)
    (cap root : ByteArray) (extras : Array _root_.Flock.Claim) (mk : ℕ → F128) :
    Rets (fun r => r.1.finalC = mk lay.fc) (_root_.Flock.Firewall.Finish.repPiop io st lay cap root extras mk) := by
  unfold _root_.Flock.Firewall.Finish.repPiop
  refine rets_bind' (zerocheckFw_rets io st.m st.digest cap root lay mk) ?_
  rintro ⟨zp, zc⟩ hr
  refine rets_bind ?_
  rintro ⟨lp, lc⟩
  exact rets_pure hr

/-- **`repCons`'s proof carries the passed `final_c` under its pad**: its zerocheck proof's `finalC` is the PIOP's masked
value at `final_c`'s pad. -/
theorem repCons_rets (io : Asks (StateT ℕ V))
    (commit : _root_.Flock.Level → Array F128 → (ℕ → ByteArray) → StateT ℕ V _root_.Flock.Firewall.Finish.Commit)
    (st : _root_.Flock.Setup) (sch : _root_.Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (g : Array F128)
    (cm0 : _root_.Flock.Firewall.Finish.Commit) (extras : Array _root_.Flock.Claim)
    (pads : _root_.Flock.Firewall.Finish.PadsZk) (ic : _root_.Flock.Firewall.Finish.InnerCommit) (vals : Array F128)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) :
    Rets (fun r => r.1.zerocheck.finalC = vals[(_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0).fc]! +
        pads.h[(_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0).fc]!)
      (_root_.Flock.Firewall.Finish.repCons io commit st sch tPad rep dom g cm0 extras pads ic vals extra lanePad
        salt) := by
  unfold _root_.Flock.Firewall.Finish.repCons
  refine rets_bind' (repPiop_rets io st _ _ _ extras _) ?_
  rintro ⟨zp, lp, claims⟩ hr
  refine rets_bind ?_
  rintro ⟨ss, lg, zl, rec⟩
  refine rets_bind fun _ => ?_
  refine rets_bind ?_
  rintro ⟨rp, _⟩
  refine rets_bind fun _ => ?_
  exact rets_pure hr

/-- **`finInner` returns the request's `final_c` under its pad**: the masked `final_c` its round 0 absorbs is
`repCons`'s zerocheck proof's, the PIOP's masked value at `final_c`'s pad. -/
theorem finInner_fc {t : Tab} {rep : ℕ} {dom : ByteArray} {n : ℕ} {q : RepIn} {fc' : F128} {tc : ByteArray}
    (h : finInner t rep dom n q = .ok (fc', tc)) :
    fc' = q.vals[(_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).fc]! +
      (_root_.Flock.Firewall.Finish.padsOf q.draws).h[(_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).fc]! := by
  have key : ∀ {c : F128} {x : TMW (StateT ℕ V) (F128 × ByteArray)} {tr : _root_.Flock.Transcript} {k : ℕ}
      {a : F128 × ByteArray} {tr' : _root_.Flock.Transcript} {k' : ℕ},
      (x.run tr).run k = .ok ((a, tr'), k') → Rets (fun r => r.1 = c) x → a.1 = c :=
    fun e hx => hx _ _ _ _ _ e
  unfold finInner at h
  obtain ⟨ic, -, h⟩ := v_bind_ok h
  obtain ⟨⟨⟨r, tr'⟩, k'⟩, hrun, h⟩ := v_bind_ok h
  cases h
  refine key hrun (rets_bind' (repCons_rets _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) ?_)
  rintro ⟨p, zl, cons⟩ hp
  repeat (first | refine rets_bind fun _ => ?_ | exact rets_pure hp | exact rets_throw _ | split | dsimp only)

/-- **The finish accepts the request only at `vs`'s `final_c`**: at a history `InnerHist` holds at, on the worker's own
values (`innerReq_own`), where `finInner` returns at the request with `final_c = a`, its `zerocheck c_eval` constraint
holds on the pads (`consWhy`): `a` is the c_eval of the request's round-1c values at the replay's zerocheck point, the
opened coins' (`zcE`), which is `vs`'s `final_c` where no outer coin is `1` (`Loop.FinalCFixed`,
`outer_ne_one_ofLoopLink`). -/
theorem finInner_accepts_own (vs : finS.SwVals) (hreal : real = realOf src) (x : Msg) (w : Flock.Wit)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Src (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src x w)
    (hcaus : finS.SwCausal vs) (wk : (relIn L₁ src real).toSrv.Worker)
    (hV : TakesValsOwn vs (relIn L₁ src real) wk x w) (hC : FinalCFixed vs x w) {ν : finS.Nonce} {t : finS.Tape}
    {κ : (relIn L₁ src real).toSrv.Chk} {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)} {j : Fin (Masked.jOf x)} {r : Fin 2}
    {e : ℕ} (he : e < 4) (hH : InnerHist (relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r e)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {tr : Tab} {hi : Array ByteArray} {q₀ : RepIn}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hr : real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hi))
    (hq : repInOf x ((m₀, ρ₀) :: ops) spec sts (2 * j.val + r.val) (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀)
    (a : F128)
    (hok : (finInner tr ((2 * j.val + r.val) % 2) (domOf spec (2 * j.val + r.val)) (nSched spec sts (2 * j.val + r.val))
      (askOf q₀ (Masked.layE x j).fc (nSched spec sts (2 * j.val + r.val) + e) a)).isOk = true) :
    a = vsWord vs x w ν (finS.pad (ops.map fun p => finS.decOf p.2)) t j r (Masked.layE x j).fc := by
  sorry

/-- **The finish accepts the request at `vs`'s `final_c`**: at a history `InnerHist` holds at with `e < 4`, `finInner`
returns at the request with `final_c` `vs`'s. Its `repCons` returns at the stream's scheduled coins (the session's cut of
the rep returns there and `repCons` with it, `repCons_asksPad`), taking `nSched` rounds; its `zerocheck c_eval`
constraint holds on the pads at `vs`'s `final_c` (`Loop.FinalCFixed`), so `consWhy` passes; and `innerP` returns on the
coins past them, the stream's and then ones. -/
theorem finInner_returns_own (vs : finS.SwVals) (hreal : real = realOf src) (x : Msg) (w : Flock.Wit)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Src (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src x w)
    (hcaus : finS.SwCausal vs) (wk : (relIn L₁ src real).toSrv.Worker)
    (hV : TakesValsOwn vs (relIn L₁ src real) wk x w) (hC : FinalCFixed vs x w) (hrows : RowsOK vs x w)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (relIn L₁ src real).toSrv.Chk} {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)}
    {j : Fin (Masked.jOf x)} {r : Fin 2} {e : ℕ} (he : e < 4)
    (hH : InnerHist (relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r e)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {tr : Tab} {hi : Array ByteArray} {q₀ : RepIn}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hr : real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hi))
    (hq : repInOf x ((m₀, ρ₀) :: ops) spec sts (2 * j.val + r.val) (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀) :
    ∃ (fc' : F128) (tc : ByteArray),
      finInner tr ((2 * j.val + r.val) % 2) (domOf spec (2 * j.val + r.val)) (nSched spec sts (2 * j.val + r.val))
        (askOf q₀ (Masked.layE x j).fc (nSched spec sts (2 * j.val + r.val) + e)
          (vsWord vs x w ν (finS.pad (ops.map fun p => finS.decOf p.2)) t j r (Masked.layE x j).fc)) =
        .ok (fc', tc) := by
  sorry

/-- **The finish's `τ` commitment at a rep's inner round 2**: at a history `InnerHist` holds at with `e = 2`, where the
rows hold (`Loop.RowsOK`), where `finInner` returns at the request, its `τ` commitment is `vs`'s (`Loop.vsTau`): it
accepts only at `vs`'s `final_c` (`finInner_accepts_own`), and its inner proof commits to `InZ.tau` over its constraints
at the coins of the inner proof's rounds 0 and 1, the stream's, which where the request's values are `vs`'s and its pads
the tape's are the switched view's first rows (`Composed.tauFw_at_coef`, `Masked.swTau_of_rows`). -/
theorem finInner_tau_own (vs : finS.SwVals) (hreal : real = realOf src) (x : Msg) (w : Flock.Wit)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Src (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src x w)
    (hcaus : finS.SwCausal vs) (wk : (relIn L₁ src real).toSrv.Worker)
    (hV : TakesValsOwn vs (relIn L₁ src real) wk x w) (hC : FinalCFixed vs x w) (hrows : RowsOK vs x w)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (relIn L₁ src real).toSrv.Chk} {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)}
    {j : Fin (Masked.jOf x)} {r : Fin 2} (hH : InnerHist (relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r 2)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {tr : Tab} {hi : Array ByteArray} {q₀ : RepIn}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hr : real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hi))
    (hq : repInOf x ((m₀, ρ₀) :: ops) spec sts (2 * j.val + r.val) (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀)
    {a fc' : F128} {tc : ByteArray}
    (hok : finInner tr ((2 * j.val + r.val) % 2) (domOf spec (2 * j.val + r.val)) (nSched spec sts (2 * j.val + r.val))
      (askOf q₀ (Masked.layE x j).fc (nSched spec sts (2 * j.val + r.val) + 2) a) = .ok (fc', tc)) :
    tc = vsTau vs x w ν (finS.pad (ops.map fun p => finS.decOf p.2)) t j r := by
  sorry

end Loop

end Claims.ZKServer
