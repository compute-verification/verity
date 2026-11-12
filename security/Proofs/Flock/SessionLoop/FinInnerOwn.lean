module

public import Proofs.Flock.SessionLoop.FinInnerParts
public import Proofs.Flock.SessionLoop.FinInnerRet
public import Proofs.Flock.SessionLoop.InnerUnrefused

/-!
# The loop's finish returns at a rep's inner rounds on the worker's own histories

`Loop.finInner_returns_own`: at a history `LiveInnerOwn` reads at rep `2j + r`'s inner round `e` (`Loop.InnerHist`), the
finish's live inner proof (`finInner`) returns at the request with `final_c` `vs`'s, by `Loop.finInner_ok_at`:

* the request's coins of the rep's rounds before the history's next count the rep's plan (`loopNr_rep`), which is
  `repCons`'s counts and then the inner proof's four (`repPlan_cons`), so they start with `repCons`'s;
* its pair's coins (`ρ₀, ρ₁`, the plan's two-coin rounds `Count.repPlan_rho`) are the opened coins', which no refusal
  shows dependent (`innerAt_unrefused`), and its `β` is the opened coin, which none shows zero, where the request holds
  it, and past the request's coins a one;
* its value at `final_c` is `vs`'s, which where no outer coin is `1` is the c_eval of its round-1c values, `vs`'s
  (`innerReq_own`), at the opened coins' zerocheck point (`Loop.FinalCFixed`).
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Firewall.Frame (INNER_ROUNDS)
open _root_.Flock.Firewall.Finish (Tab RepIn)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Loop

variable (L₁ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray))

set_option maxHeartbeats 2000000 in
/-- **The finish returns at the request with `vs`'s `final_c`**: at a history `InnerHist` holds at with `e < 4`,
`finInner` returns at the request with `final_c` `vs`'s (`finInner_ok_at`). The request's coins before the history's
next count the rep's plan, `repCons`'s rounds then the inner proof's; its pair's coins and `β` are the opened coins',
not refused (`innerAt_unrefused`); and its value at `final_c` is the c_eval of its round-1c values at the opened coins'
zerocheck point (`Loop.FinalCFixed`, `innerReq_own`). -/
theorem finInner_returns_own (vs : finS.SwVals) (hreal : real = realOf src) (x : Msg) (w : Flock.Wit)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Src (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src x w)
    (hcaus : finS.SwCausalOn (SwOk Masked.dataE) vs) (wk : (relIn L₁ src real).toSrv.Worker)
    (hSch : LeanRelease.LiveSchedOwn vs (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src real wk
      x w) (hC : FinalCFixed vs x w)
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
          (vsWord vs x w ν (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) t j r (Masked.layE x j).fc)) =
        .ok (fc', tc) := by
  have hH' := hH
  obtain ⟨hpos, hlt, hl, -, hrf, -, ⟨v, hv⟩, hI, -, hslot⟩ := hH'
  have hx : Masked.Live x := Masked.live_of hsc hR
  have hsts : Masked.stsOf x = sts := by
    unfold Masked.stsOf
    rw [hs]
  subst hsts
  have hjs := Masked.lt_of_jOf j
  have hr2 := r.isLt
  have hk2 : (2 * j.val + r.val) / 2 = j.val := by omega
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hrsz : spec.repStreams.size = 2 * (Masked.stsOf x).size := by rw [FlockDeploy.repStreams_size, hspec]
  have hkk : 2 * j.val + r.val < spec.repStreams.size := by omega
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hjs).1
  have hm35 : (Masked.stAt x j).m ≤ 35 := (htabs j.val hjs).2.1
  -- the table is table `j`'s, at its `--zk` schedule
  have hRT : Count.RealTabs (Masked.stsOf x) (real x w (ν, t)) := by
    subst hreal
    exact realTabs_of src x w (ν, t) hs
  obtain ⟨_, _, _, -, htr⟩ := hRT _ tr hi hr
  obtain ⟨hts, hsch⟩ := Count.tabOfLevel0_sched htr
  have hst' : (Masked.stsOf x)[j.val]! = Masked.stAt x j := getElem!_pos (Masked.stsOf x) j.val hjs
  have hsch' : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m = .ok (tr.sch, tr.tPad) := hsch
  have hz : _root_.Flock.Zk.schedule (Masked.stAt x j).m = .ok (tr.sch, tr.tPad) := by
    have e : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m = .ok (tr.sch, tr.tPad) := hsch
    rwa [hk2, hst'] at e
  obtain ⟨hn6, hq6⟩ := FlockComplete.JsonRT.zkSchedule_small hm25 hm35 hz
  have hn6' : tr.sch.n + 1 ≤ 65536 := hn6
  have hq6' : ∀ i, (tr.sch.levels.getD i default).q + 1 ≤ 65536 := hq6
  have hz' := hz
  rw [(Masked.zkTab x j).hz] at hz'
  have htsch : tr.sch = (Masked.zkTab x j).sch := (Prod.mk.inj (Except.ok.inj hz')).1.symm
  have htst : tr.st = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) := by
    rw [hts, hk2, hst']
  have hLayT : _root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0 = Masked.layE x j := by
    rw [htst, htsch]
    rfl
  have hms : tr.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    rw [htst]
    exact (htabs j.val hjs).2.2
  have e7 : tr.st.m = (Masked.stAt x j).m := congrArg (fun s : _root_.Flock.Setup => s.m) htst
  have e6 : tr.st.fold.kLog = (Masked.stAt x j).c.kLog := congrArg (fun s : _root_.Flock.Setup => s.fold.kLog) htst
  have e8 : tr.st.kLog = (Masked.stAt x j).c.kLog := congrArg (fun s : _root_.Flock.Setup => s.kLog) htst
  -- the table's statement is a parsed one: its fold returns
  obtain ⟨⟨hwf, -⟩, hslotT⟩ : FlockComplete.ZkTables.StmtT (Masked.stAt x j) :=
    _root_.Flock.Readable.stmtT_of_scope hsc hs hjs
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := hwf.hk6
  have hfoldT : ∀ (a b : F128) (lw ρ ts : Array F128), lw.size = 64 → ρ.size = tr.st.fold.kLog - 6 →
      ts.size = tr.st.fold.kLog - 6 → ∃ y, tr.st.fold.folded a b lw ρ ts = .ok y ∧ y.size = 64 := by
    rw [htst]
    exact fun a b lw ρ ts (hlw : lw.size = 64) (hρ : ρ.size = (Masked.stAt x j).c.kLog - 6)
        (hts' : ts.size = (Masked.stAt x j).c.kLog - 6) => by
      obtain ⟨v', hv'⟩ := FlockComplete.JsonRT.fold_ok _ hk6 hslotT a hlw hρ
      exact ⟨_, FlockLevel3.folded_ok _ a b lw ρ ts v' hv',
        _root_.Flock.RecAlgebra.folded_size (st := Masked.stAt x j) (FlockLevel3.folded_ok _ a b lw ρ ts v' hv')
          (by omega)⟩
  have hfT : ∀ (a : F128) (lw ρ : Array F128), lw.size = 64 → ρ.size = tr.st.fold.kLog - 6 →
      ∃ y, tr.st.fold.fold a lw ρ = .ok y := by
    rw [htst]
    exact fun a lw ρ hlw hρ => FlockComplete.JsonRT.fold_ok _ hk6 hslotT a hlw hρ
  -- the request: its values but `final_c`'s are `vs`'s, its coins the opened ones from the rep's base
  obtain ⟨hvsz, hvals, -⟩ := innerReq_own L₁ src real vs hreal x w hR hSv hsc hSrc hcaus wk hSch he hH hs hr hq
  obtain ⟨-, hnc, hcoins⟩ := innerReq_coins L₁ src real x w wk he hH hs hq
  rw [show finCoins x ops = finS.pad (K := refK x) (ops.map fun (p : Msg × Msg) => finS.decOf p.2) from rfl]
    at hcoins
  obtain ⟨-, hbase⟩ := repSlot_nSched hx hs hkk
  have hslot' : ops.length =
      Masked.repBase x (2 * j.val + r.val) + nSched spec (Masked.stsOf x) (2 * j.val + r.val) + e := by
    rw [hslot, Masked.repSlot_innerOff hs j r e, ← hbase]
  have hnr := LeanRelease.innerAt_unrefused L₁ src real (finOf real L₁.draws) x w hx hSv
    (fun ρ s0' spec' sts' h => by rw [hreal]; exact realTabs_of src x w ρ h) wk ν t κ m₀ ρ₀ ops v hpos hlt hl hrf hv hI
  have hout := LeanRelease.outer_ne_one_ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws))
    src real wk hSv (ν, t) κ m₀ ρ₀ ops hlt.le hl hrf j
  obtain ⟨Q, hQ⟩ : ∃ Q, Q = askOf q₀ (Masked.layE x j).fc (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + e)
      (vsWord vs x w ν (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) t j r (Masked.layE x j).fc) :=
    ⟨_, rfl⟩
  rw [← hQ]
  have hQc : Q.coins = q₀.coins.extract 0 (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + e) := by
    rw [hQ]; rfl
  have hQsz : Q.coins.size = nSched spec (Masked.stsOf x) (2 * j.val + r.val) + e := by
    rw [hQc, Array.size_extract]
    omega
  have hQv : Q.vals = q₀.vals.set! (Masked.layE x j).fc
      (vsWord vs x w ν (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) t j r (Masked.layE x j).fc) := by
    rw [hQ]; rfl
  have hQx : Q.xs = q₀.xs := by
    rw [hQ]; rfl
  have hQk : ∀ k < nSched spec (Masked.stsOf x) (2 * j.val + r.val) + e, Q.coins[k]! = q₀.coins[k]! := by
    intro k hk
    rw [hQc]
    exact extract_get! hk (by omega)
  have hpc : ∀ k < nSched spec (Masked.stsOf x) (2 * j.val + r.val) + e,
      padCoin Q.coins k = fun i => (q₀.coins[k]!)[i]! := by
    intro k hk
    rw [hQc]
    exact padCoin_extract hk (by omega)
  -- as many extra claims as the statement's regions at each link point, below `2^64`
  have hxs : (_root_.Flock.Firewall.Finish.claimsOf Q.xs).size = spec.points * (Masked.stAt x j).regions.size := by
    simp only [_root_.Flock.Firewall.Finish.claimsOf, Array.size_map]
    rw [hQx]
    exact innerReq_xs hx hSv hpos hs j r hq
  obtain ⟨hpts, -, hregs⟩ := Masked.live_bounds hx hs
  have hreg : (Masked.stAt x j).regions.size ≤ 1024 := hregs j.val hjs
  have hc65 : 2 + spec.points * (Masked.stAt x j).regions.size ≤ 65536 := by
    rw [hpts]
    omega
  have hxb : (_root_.Flock.Firewall.Finish.claimsOf Q.xs).size + 2 < 2 ^ 64 :=
    lt_of_le_of_lt (by rw [hxs, hpts]; omega) (by norm_num : (2050 : ℕ) < 2 ^ 64)
  -- the rep's plan: `repCons`'s rounds, `nSched` of them, then the inner proof's four
  have hPsz : (_root_.Flock.Firewall.Session.repPlan spec (Masked.stAt x j)).size =
      ((Masked.stAt x j).m + ((Masked.stAt x j).c.kLog - 6)) +
        ((2 + spec.points * (Masked.stAt x j).regions.size) + 1) +
        (tr.sch.k0 + 7 + 11 * (tr.sch.levels.size - 1)) + 4 := by
    rw [repPlan_size, repRounds_true]
    simp only [hz]
  have hrq : (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 =
      (_root_.Flock.Firewall.Session.repPlan spec (Masked.stAt x j)).size := by
    rw [repPlan_size, ← Masked.stsOf_half x j r]
    simp only [roundsOf, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hkk, Option.map_some,
      Option.getD_some]
  have hI4 : INNER_ROUNDS = 4 := rfl
  have hnS : nSched spec (Masked.stsOf x) (2 * j.val + r.val) =
      ((Masked.stAt x j).m + ((Masked.stAt x j).c.kLog - 6)) +
        ((2 + spec.points * (Masked.stAt x j).regions.size) + 1) +
        (tr.sch.k0 + 7 + 11 * (tr.sch.levels.size - 1)) := by
    unfold nSched
    rw [hrq, hPsz]
    omega
  have hP' : (_root_.Flock.Firewall.Session.repPlan spec (Masked.stAt x j)).toList =
      Count.consCounts tr.st tr.sch (_root_.Flock.Firewall.Finish.claimsOf Q.xs).size ++
        [1, _root_.Flock.Firewall.Session.INNER_CONS, 1, _root_.Flock.Zk.Q] := by
    rw [htst, hxs]
    exact repPlan_cons hz
  have hcl : (Count.consCounts tr.st tr.sch (_root_.Flock.Firewall.Finish.claimsOf Q.xs).size).length =
      nSched spec (Masked.stsOf x) (2 * j.val + r.val) := by
    have h := congrArg List.length hP'
    rw [Array.length_toList, List.length_append, hPsz] at h
    simp only [List.length_cons, List.length_nil] at h
    omega
  have hsize : ∀ k < nSched spec (Masked.stsOf x) (2 * j.val + r.val) + e,
      (Q.coins[k]!).size = (_root_.Flock.Firewall.Session.repPlan spec (Masked.stAt x j)).getD k 0 := by
    intro k hk
    rw [hQk k hk, (hcoins k hk).1, Masked.repBase_eq hs, loopNr_rep hs hkk (i := k) (by rw [hrq, hPsz]; omega),
      Masked.stsOf_half x j r]
    exact Nat.min_eq_left (repPlan_le hz hm35 hn6' hq6' hc65 k)
  have hpre : Count.consCounts tr.st tr.sch (_root_.Flock.Firewall.Finish.claimsOf Q.xs).size <+:
      Q.coins.toList.map Array.size :=
    prefix_of_sizes hP' (by rw [hcl, hQsz]; omega) fun k hk => hsize k (by rw [hcl] at hk; omega)
  -- the pair's coins: the plan's two-coin rounds, the opened coins at the rep's `ρ₀, ρ₁` rounds, independent
  obtain ⟨L, hL⟩ : ∃ L, L = FlockComplete.CutEq.ligAt 0 tr.st (_root_.Flock.Firewall.Finish.claimsOf Q.xs) :=
    ⟨_, rfl⟩
  have hLig : L = 3 + ((Masked.stAt x j).m - 6) + (3 + ((Masked.stAt x j).c.kLog - 6)) +
      (2 + spec.points * (Masked.stAt x j).regions.size) + 1 := by
    have h : L = 0 + 3 + (tr.st.m - 6) + (3 + (tr.st.fold.kLog - 6)) +
        (2 + (_root_.Flock.Firewall.Finish.claimsOf Q.xs).size) + 1 := hL
    rw [e7, e6, hxs] at h
    omega
  have eρ : Masked.rhoSlot x j.val r.val = Masked.repBase x (2 * j.val + r.val) + (L + 2 + tr.sch.k0) := by
    unfold Masked.rhoSlot
    rw [Masked.repSlot_eq hs hsch', Masked.repBase_eq hs, Masked.stsOf_half x j r, hLig]
    unfold FlockComplete.piopRoundsZk
    omega
  have hfl := congrArg List.length
    (Count.piop_list (kLog := (Masked.stAt x j).c.kLog) (by omega : 14 ≤ (Masked.stAt x j).m))
  simp only [List.length_cons, List.length_append, List.length_replicate, List.length_nil] at hfl
  have hρc : ∀ k e', e' < 2 → k = L + 2 + tr.sch.k0 + e' →
      (_root_.Flock.Firewall.Session.repPlan spec (Masked.stAt x j)).getD k 0 = 2 := by
    intro k e' he' hk
    rw [show k = ((_root_.Flock.Firewall.Session.replayCoins (Masked.stAt x j).m
        (Masked.stAt x j).c.kLog).toList.filter (· != 0)).length +
        ((2 + spec.points * (Masked.stAt x j).regions.size) + 1 + (tr.sch.k0 + 2 + e')) by omega]
    exact Count.repPlan_rho hz he'
  have hA : ∀ i < 2, (Q.coins[L + 2 + tr.sch.k0]!)[i]! =
      Masked.slotAt (finS.pad (K := refK x) (ops.map fun (p : Msg × Msg) => finS.decOf p.2))
        (Masked.rhoSlot x j.val r.val) i := by
    intro i hi
    have hk : L + 2 + tr.sch.k0 < nSched spec (Masked.stsOf x) (2 * j.val + r.val) + e := by omega
    have hsz : i < (q₀.coins[L + 2 + tr.sch.k0]!).size := by
      rw [← hQk _ hk, hsize _ hk, hρc (L + 2 + tr.sch.k0) 0 (by norm_num) (by omega)]
      exact hi
    rw [hQk _ hk, eρ]
    exact ((hcoins _ hk).2 i hsz).symm
  have hB : ∀ i < 2, (Q.coins[L + 3 + tr.sch.k0]!)[i]! =
      Masked.slotAt (finS.pad (K := refK x) (ops.map fun (p : Msg × Msg) => finS.decOf p.2))
        (Masked.rhoSlot x j.val r.val + 1) i := by
    intro i hi
    have hk : L + 3 + tr.sch.k0 < nSched spec (Masked.stsOf x) (2 * j.val + r.val) + e := by omega
    have hsz : i < (q₀.coins[L + 3 + tr.sch.k0]!).size := by
      rw [← hQk _ hk, hsize _ hk, hρc (L + 3 + tr.sch.k0) 1 (by norm_num) (by omega)]
      exact hi
    rw [hQk _ hk, show Masked.rhoSlot x j.val r.val + 1 =
      Masked.repBase x (2 * j.val + r.val) + (L + 3 + tr.sch.k0) by omega]
    exact ((hcoins _ hk).2 i hsz).symm
  have hρ' : _root_.Flock.Zk.rhoIndependent
      (Masked.rhoAt (finS.pad (K := refK x) (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) j.val r.val) = true := by
    cases hb : _root_.Flock.Zk.rhoIndependent
      (Masked.rhoAt (finS.pad (K := refK x) (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) j.val r.val)
    · exact (hnr (.inr ⟨j.val, hjs, r.val, hr2, .inr (.inl ⟨by omega, hb⟩)⟩)).elim
    · rfl
  have hρQ : _root_.Flock.Zk.rhoIndependent
      ((⟨(Q.coins[FlockComplete.CutEq.ligAt 0 tr.st (_root_.Flock.Firewall.Finish.claimsOf Q.xs) + 2 +
            tr.sch.k0]!)[0]!,
          (Q.coins[FlockComplete.CutEq.ligAt 0 tr.st (_root_.Flock.Firewall.Finish.claimsOf Q.xs) + 2 +
            tr.sch.k0]!)[1]!⟩ : _root_.Flock.F256),
        (⟨(Q.coins[FlockComplete.CutEq.ligAt 0 tr.st (_root_.Flock.Firewall.Finish.claimsOf Q.xs) + 3 +
            tr.sch.k0]!)[0]!,
          (Q.coins[FlockComplete.CutEq.ligAt 0 tr.st (_root_.Flock.Firewall.Finish.claimsOf Q.xs) + 3 +
            tr.sch.k0]!)[1]!⟩ : _root_.Flock.F256)) = true := by
    rw [← hL, hA 0 (by norm_num), hA 1 (by norm_num), hB 0 (by norm_num), hB 1 (by norm_num)]
    exact hρ'
  -- the value at `final_c`: `vs`'s, the c_eval of the round-1c values at the opened coins' zerocheck point
  have hfcz : (Masked.layE x j).fc < (Masked.layE x j).zp 64 := by
    unfold _root_.Flock.Zk.PadLayout.zp _root_.Flock.Zk.PadLayout.lc
    omega
  have hfa : Q.vals[(Masked.layE x j).fc]! =
      vsWord vs x w ν (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) t j r (Masked.layE x j).fc := by
    rw [hQ]
    exact askOf_vals_fc _ _ (by rw [hvsz]; exact hfcz)
  have hR1 : ((Array.range 64).map fun i => Q.vals[(Masked.layE x j).r1c i]!) =
      (Array.range 64).map fun i =>
        vsWord vs x w ν (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) t j r ((Masked.layE x j).r1c i) := by
    refine Array.map_congr_left fun i hi => ?_
    have hi' : i < 64 := by simpa using hi
    have hlt' : (Masked.layE x j).r1c i < (Masked.layE x j).fc := by
      unfold _root_.Flock.Zk.PadLayout.r1c _root_.Flock.Zk.PadLayout.fc _root_.Flock.Zk.PadLayout.fa
      omega
    rw [hQv, Array.getElem!_set!_ne q₀.vals (Masked.layE x j).fc ((Masked.layE x j).r1c i) _ (by omega)]
    exact hvals _ (by omega) (by omega)
  have hNS := piopNR_le_nSched hx hs j r
  have hP6 : 6 ≤ piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog := by
    unfold piopNR
    omega
  have hNR2 := loopNr_piop_at hx hs j r (jj := 2) (by omega)
  have hpos2 : 0 < (q₀.coins[2]!).size := by
    rw [(hcoins 2 (by omega)).1, hNR2]
    unfold piopN _root_.Flock.Firewall.Frames.COIN_MAX
    split_ifs <;> omega
  have hz2 : padCoin Q.coins 2 0 =
      (Masked.zcE (finS.pad (K := refK x) (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) j r).z := by
    rw [hpc 2 (by omega)]
    exact ((hcoins 2 (by omega)).2 0 hpos2).symm
  have hvalQ : Q.vals[(_root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0).fc]! =
      _root_.Flock.dot (_root_.Flock.lagrange _root_.Flock.nodesL _root_.Flock.den6Inv (padCoin Q.coins 2 0))
        ((Array.range 64).map fun i =>
          Q.vals[(_root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0).r1c i]!) := by
    rw [hLayT, hfa, hR1, hz2]
    exact hC ν (finS.pad (K := refK x) (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) t j hout r
  -- the inner proof's rounds the request holds count the plan's
  have hpl : ∀ i < 4, (_root_.Flock.Firewall.Session.repPlan spec (Masked.stAt x j)).getD
      (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + i) 0 =
      [1, _root_.Flock.Firewall.Session.INNER_CONS, 1, _root_.Flock.Zk.Q].getD i 0 := by
    intro i hi
    have hle : (Count.consCounts tr.st tr.sch (_root_.Flock.Firewall.Finish.claimsOf Q.xs).size).length ≤
        nSched spec (Masked.stsOf x) (2 * j.val + r.val) + i := by
      rw [hcl]
      omega
    rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hP', List.getElem?_append_right hle, hcl,
      Nat.add_sub_cancel_left, List.getD_eq_getElem?_getD]
  have hIn : ∀ i < 4, ∀ cs, Q.coins[nSched spec (Masked.stsOf x) (2 * j.val + r.val) + i]? = some cs →
      cs.size = [1, _root_.Flock.Firewall.Session.INNER_CONS, 1, _root_.Flock.Zk.Q].getD i 0 := by
    intro i hi cs hcs
    by_cases hie : i < e
    · have hk : nSched spec (Masked.stsOf x) (2 * j.val + r.val) + i <
          nSched spec (Masked.stsOf x) (2 * j.val + r.val) + e := by omega
      rw [getElem?_pos Q.coins (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + i) (by omega)] at hcs
      obtain rfl := Option.some.inj hcs
      rw [← getElem!_pos Q.coins (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + i) (by omega), hsize _ hk,
        hpl i hi]
    · rw [getElem?_neg Q.coins (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + i) (by omega)] at hcs
      cases hcs
  have h0 : ∀ cs, Q.coins[nSched spec (Masked.stsOf x) (2 * j.val + r.val)]? = some cs → cs.size = 1 :=
    hIn 0 (by norm_num)
  have h1 : ∀ cs, Q.coins[nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 1]? = some cs →
      cs.size = _root_.Flock.Firewall.Session.INNER_CONS := hIn 1 (by norm_num)
  have h2 : ∀ cs, Q.coins[nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 2]? = some cs → cs.size = 1 :=
    hIn 2 (by norm_num)
  have h3 : ∀ cs, Q.coins[nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 3]? = some cs →
      cs.size = _root_.Flock.Zk.Q := hIn 3 (by norm_num)
  -- `β`: the opened coin at the rep's `β` round, not zero, where the request holds it, and a one past it
  have eβ : Masked.betaSlot x j.val r.val =
      Masked.repBase x (2 * j.val + r.val) + (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 2) := by
    have h1 : Masked.betaSlot x j.val r.val = Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff 2) := rfl
    have h2 := Masked.repSlot_innerOff hs j r 2
    omega
  have hβQ : (padCoin Q.coins (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 2) 0).isZero = false := by
    by_cases he3 : e = 3
    · have hk : nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 2 <
          nSched spec (Masked.stsOf x) (2 * j.val + r.val) + e := by omega
      have hsz : 0 < (q₀.coins[nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 2]!).size := by
        rw [← hQk _ hk, hsize _ hk, hpl 2 (by norm_num)]
        exact Nat.one_pos
      rw [hpc _ hk]
      show ((q₀.coins[nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 2]!)[0]!).isZero = false
      rw [← (hcoins _ hk).2 0 hsz, ← eβ]
      show (Masked.betaAt (finS.pad (K := refK x) (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) j.val
        r.val).isZero = false
      cases hb : (Masked.betaAt (finS.pad (K := refK x) (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) j.val
        r.val).isZero
      · rfl
      · exact (hnr (.inr ⟨j.val, hjs, r.val, hr2, .inr (.inr ⟨by omega, hb⟩)⟩)).elim
    · unfold padCoin
      rw [getElem?_neg Q.coins (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 2) (by omega)]
      decide
  have hzT : _root_.Flock.Zk.schedule tr.st.m = .ok (tr.sch, tr.tPad) := by
    rw [e7]
    exact hz
  exact finInner_ok_at (by rw [e7]; exact hm25) (by rw [e7]; exact hm35) (by rw [e6, e7]; exact hwf.hk)
    (e8.trans e6.symm) hms hfoldT hfT hzT hxb hpre hcl (by omega) hρQ hvalQ h0 h1 h2 h3 hβQ

end Loop

end Claims.ZKServer
