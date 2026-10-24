import Proofs.Flock.SessionLoop.FinRepXs
import Proofs.Flock.Claims.ZK.L0PinLoop
import Proofs.Flock.Claims.ZK.L0PinCut

/-!
# Level 0's values at the views' source

At (c.3)'s context, the views' source (`Masked.swSrcE` at the workers' values `vs`) holds, from `zp 64` to `h_ab`, level
0's clear values at the opened coins plus the request's pads (`Finish.padsOf q.draws`): the OOD reply, each lane round's
pair and the folded weight's products with the extra lanes (`l0Clear`), at their layout positions.

* **The loop's level 0** (`Loop.l0_loop`): table `j`'s rep `r`'s rounds past its binding send, tagged `0`, level 0's
  bodies at the opened coins from `ligP0`, the request's pads and the basis of its switched vectors at those coins
  (`Masked.ligBasis`): `LeanRelease.ofLoop_l0` at the loop's table (`realOf`), its coins the opened coins
  (`l0Body_shift`, `words_slot`, `loopNr_l0`) and its switched vectors the session's (`sws_of_pts`, `loopNr_piop`).
* **`Loop.swSrcE_l0_of_cut`**: where the cut session's bodies there are the cut opening's from the views' source, that
  source's masked values are `l0Clear` plus the pads (`masked_of_cutL0`): the history's messages are the cut session's
  (`msg_runCut`), round messages without a domain frame (`dropDom_l0Body`).
* **`Loop.swSrcE_l0_live`**: the same with the cut opening's bodies from the earlier reps' returns
  (`Masked.runCut_cutL0_of_ret`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

/-- **Level 0's round messages carry no domain frame**: bodies `1` to `k₀ + 2` lead with the OOD reply or a pair. -/
theorem dropDom_l0Body {coin : ℕ → ℕ → F128} {p0 : ℕ} {P : List UInt8} {T : F128} {cap : ByteArray}
    {sch : _root_.Flock.Schedule} {t b : Array F128} {lay : _root_.Flock.Zk.PadLayout} {h : Array F128} {rep : ℕ}
    {extra : ℕ → ℕ → F128} {nonce : ℕ → ℕ → UInt64} {i : ℕ} (h1 : 1 ≤ i) (h2 : i ≤ sch.k0 + 2) :
    dropDom (0 :: FlockComplete.LigLock.l0BodyAt coin p0 P T cap sch t b lay h rep extra nonce i) =
      0 :: FlockComplete.LigLock.l0BodyAt coin p0 P T cap sch t b lay h rep extra nonce i := by
  rcases Nat.lt_or_ge i 2 with hi | hi
  · obtain rfl : i = 1 := by omega
    rw [FlockComplete.LigLock.l0BodyAt_one]
    exact dropDom_noDom _ _ _ fun f hf => by
      simp only [oodFr, List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl <;> exact True.intro
  · rcases Nat.lt_or_ge i (sch.k0 + 2) with hk | hk
    · obtain ⟨i', rfl⟩ : ∃ i', i = i' + 2 := ⟨i - 2, by omega⟩
      rw [FlockComplete.LigLock.l0BodyAt_msg (by omega)]
      exact dropDom_noDom _ _ _ fun f hf => by
        simp only [pairFr, List.mem_cons, List.not_mem_nil, or_false] at hf
        rcases hf with rfl | rfl <;> exact True.intro
    · obtain rfl : i = sch.k0 + 2 := by omega
      rw [FlockComplete.LigLock.l0BodyAt_e]
      exact dropDom_noDom _ _ _ fun f hf => by
        simp only [pairFr, List.mem_cons, List.not_mem_nil, or_false] at hf
        rcases hf with rfl | rfl <;> exact True.intro

/-- **The loop's words are the opened coins** below the round's count. -/
theorem words_slot {x : Msg} {ops : List (Msg × Msg)} (hl : ops.length = refK x) {p e : ℕ} (hp : p < ops.length)
    (he : e < loopNr x p) (hc : e < 65536) :
    (wordsAt x (ops.map fun q => ZKMal.CoinLayer.rd q.2) p)[e]! = Masked.slotAt (finCoins x ops) p e := by
  have hs := wordsAt_size x ops hp le_rfl
  rw [List.take_length] at hs
  have h := wordsAt_slot Masked.dataE x ops hl hp le_rfl (e := e)
    (by rw [List.take_length, hs]; exact Nat.lt_min.mpr ⟨he, hc⟩)
  rw [List.take_length] at h
  exact h

/-- **The loop asks rep `q`'s rounds through level 0 for its plan's counts.** -/
theorem loopNr_plan {x : Msg} {s : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, sts)) {q : ℕ} (hq : q < spec.repStreams.size)
    {sch : _root_.Flock.Schedule} {tp : ℕ} (hz : _root_.Flock.Zk.schedule sts[q / 2]!.m = .ok (sch, tp))
    (hk : 6 ≤ sts[q / 2]!.c.kLog) {i : ℕ} (hi : i ≤ FlockComplete.piopRoundsZk spec sts[q / 2]! + sch.k0 + 2) :
    loopNr x (FlockComplete.repStartZk spec sts q + i) =
      (_root_.Flock.Firewall.Session.repPlan spec sts[q / 2]!).getD i 0 := by
  refine loopNr_rep hsx hq (i := i) ?_
  have e1 : sts[q / 2]!.m + sts[q / 2]!.c.kLog - 6 = sts[q / 2]!.m + (sts[q / 2]!.c.kLog - 6) := by omega
  have hrR : FlockDeploy.repRounds true spec sts[q / 2]! =
      FlockComplete.piopRoundsZk spec sts[q / 2]! + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) + 4 := by
    rw [repRounds_true]
    simp only [hz]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
  simp only [roundsOf, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hq, Option.map_some,
    Option.getD_some]
  rw [hrR]
  omega

/-- **The loop's counts in level 0**: `z`'s `n`, then one, then two for each lane round and the extra lanes. -/
theorem loopNr_l0 {x : Msg} {s : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, sts)) {q : ℕ} (hq : q < spec.repStreams.size)
    {sch : _root_.Flock.Schedule} {tp : ℕ} (hz : _root_.Flock.Zk.schedule sts[q / 2]!.m = .ok (sch, tp))
    (hm : 14 ≤ sts[q / 2]!.m) (hk : 6 ≤ sts[q / 2]!.c.kLog) {i : ℕ} (hi : i ≤ sch.k0 + 2) :
    loopNr x (FlockComplete.repStartZk spec sts q + FlockComplete.piopRoundsZk spec sts[q / 2]! + i) =
      if i = 0 then sch.n else l0Nc i := by
  have hL := congrArg List.length (Count.piop_list (kLog := sts[q / 2]!.c.kLog) hm)
  simp only [List.length_cons, List.length_append, List.length_replicate, List.length_nil] at hL
  rw [Nat.add_assoc, loopNr_plan hsx hq hz hk (by omega), show FlockComplete.piopRoundsZk spec sts[q / 2]! + i =
    ((_root_.Flock.Firewall.Session.replayCoins sts[q / 2]!.m sts[q / 2]!.c.kLog).toList.filter (· != 0)).length +
      ((2 + spec.points * sts[q / 2]!.regions.size) + 1 + i) by unfold FlockComplete.piopRoundsZk; omega]
  exact Count.repPlan_l0 hz hi

/-- **The loop's counts in the algebraic phase and ring switch**: the zerocheck's `m − 13` and ones, the lincheck's
ones, each claim's `7` and the claims' batching. -/
theorem loopNr_piop {x : Msg} {s : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, sts)) {q : ℕ} (hq : q < spec.repStreams.size)
    {sch : _root_.Flock.Schedule} {tp : ℕ} (hz : _root_.Flock.Zk.schedule sts[q / 2]!.m = .ok (sch, tp))
    (hm : 14 ≤ sts[q / 2]!.m) (hk : 6 ≤ sts[q / 2]!.c.kLog) :
    loopNr x (FlockComplete.repStartZk spec sts q + 1) = sts[q / 2]!.m - 13 ∧
      (∀ i < sts[q / 2]!.m - 6, loopNr x (FlockComplete.repStartZk spec sts q + 3 + i) = 1) ∧
      (∀ i < sts[q / 2]!.c.kLog - 6,
        loopNr x (FlockComplete.repStartZk spec sts q + 3 + (sts[q / 2]!.m - 6) + 2 + i) = 1) ∧
      (∀ k < 2 + spec.points * sts[q / 2]!.regions.size,
        loopNr x (FlockComplete.repStartZk spec sts q + 3 + (sts[q / 2]!.m - 6) + 3 + (sts[q / 2]!.c.kLog - 6) + k) =
          7) ∧
      loopNr x (FlockComplete.repStartZk spec sts q + 3 + (sts[q / 2]!.m - 6) + 3 + (sts[q / 2]!.c.kLog - 6) +
        (2 + spec.points * sts[q / 2]!.regions.size)) = 2 + spec.points * sts[q / 2]!.regions.size := by
  obtain ⟨h1, h3, h5, h7, hN⟩ := Count.repPlan_piop (spec := spec) hz hm
  have hP : ∀ i, i ≤ FlockComplete.piopRoundsZk spec sts[q / 2]! + sch.k0 + 2 →
      loopNr x (FlockComplete.repStartZk spec sts q + i) =
        (_root_.Flock.Firewall.Session.repPlan spec sts[q / 2]!).getD i 0 :=
    fun i hi => loopNr_plan hsx hq hz hk hi
  unfold FlockComplete.piopRoundsZk at hP
  refine ⟨?_, fun i hi => ?_, fun i hi => ?_, fun k hk' => ?_, ?_⟩
  · rw [hP 1 (by omega)]
    exact h1
  · rw [Nat.add_assoc, hP (3 + i) (by omega)]
    exact h3 i hi
  · rw [show FlockComplete.repStartZk spec sts q + 3 + (sts[q / 2]!.m - 6) + 2 + i =
      FlockComplete.repStartZk spec sts q + (3 + (sts[q / 2]!.m - 6) + 2 + i) by omega, hP _ (by omega)]
    exact h5 i hi
  · rw [show FlockComplete.repStartZk spec sts q + 3 + (sts[q / 2]!.m - 6) + 3 + (sts[q / 2]!.c.kLog - 6) + k =
      FlockComplete.repStartZk spec sts q + (3 + (sts[q / 2]!.m - 6) + 3 + (sts[q / 2]!.c.kLog - 6) + k) by omega,
      hP _ (by omega)]
    exact h7 k hk'
  · rw [show FlockComplete.repStartZk spec sts q + 3 + (sts[q / 2]!.m - 6) + 3 + (sts[q / 2]!.c.kLog - 6) +
        (2 + spec.points * sts[q / 2]!.regions.size) =
      FlockComplete.repStartZk spec sts q + (3 + (sts[q / 2]!.m - 6) + 3 + (sts[q / 2]!.c.kLog - 6) +
        (2 + spec.points * sts[q / 2]!.regions.size)) by omega, hP _ (by omega)]
    exact hN

/-- **Level 0's bodies at shifted coins**: the bodies at the coins from round `R₀` read as from `0` are the bodies at
the coins `C` from `R₀`, where the two agree below each level-0 round's count. -/
theorem l0Body_shift {W C : ℕ → ℕ → F128} {R0 R1 : ℕ} (hR : R0 = R1) {P : List UInt8} {T : F128} {cap : ByteArray}
    {sch : _root_.Flock.Schedule} {t b : Array F128} {lay : _root_.Flock.Zk.PadLayout} {h : Array F128} {rep : ℕ}
    {extra : ℕ → ℕ → F128} {nonce : ℕ → ℕ → UInt64}
    (hW : ∀ a e, a ≤ sch.k0 + 2 → e < (if a = 0 then sch.n else l0Nc a) → W (R1 + a) e = C (R1 + a) e) (i : ℕ) :
    FlockComplete.LigLock.l0BodyAt (fun p e => W (R0 + p) e) 0 P T cap sch t b lay h rep extra nonce i =
      FlockComplete.LigLock.l0BodyAt C R1 P T cap sch t b lay h rep extra nonce i := by
  subst hR
  refine FlockComplete.LigLock.Fed.l0BodyAt_congr (fun i hi => ?_) ?_ (fun j hj i hi => ?_) i
  · exact hW 0 i (by omega) (by simpa using hi)
  · exact hW 1 0 (by omega) (by simp [l0Nc])
  · show W (R0 + (0 + 2 + j)) i = C (R0 + 2 + j) i
    rw [show R0 + (0 + 2 + j) = R0 + (2 + j) by omega, show R0 + 2 + j = R0 + (2 + j) by omega]
    exact hW (2 + j) i (by omega) (by rw [if_neg (by omega)]; unfold l0Nc; split <;> omega)

/-- `claimsOf`'s coordinates are the claims'. -/
theorem claimsOf_x {e : Array _root_.Flock.Claim} {xs : Array (Array F128)} (h : e.map (·.x) = xs) (k : ℕ) :
    (_root_.Flock.Firewall.Finish.claimsOf xs)[k]!.x = e[k]!.x := by
  subst h
  by_cases hk : k < e.size
  · rw [getElem!_pos _ k (by simp [_root_.Flock.Firewall.Finish.claimsOf, hk]), getElem!_pos e k hk]
    simp [_root_.Flock.Firewall.Finish.claimsOf]
  · rw [getElem!_neg _ k (by simp [_root_.Flock.Firewall.Finish.claimsOf, hk]), getElem!_neg e k hk]

/-- **The switched vectors at the loop's coins are the session's**: the ring switch on the zerocheck's and lincheck's
claims at the points of `W` from `R`, then `claimsOf xs`, at `W`'s ring-switch coins from `L₁`, is the ring switch on
the closed-form PIOP's claims at the coins `C` from `R`, then claims at `xs`, at `C`'s coins, where `W` and `C` agree
on what each round reads. -/
theorem sws_of_pts {st : _root_.Flock.Setup} {W C : ℕ → ℕ → F128} {R L₁ L₂ : ℕ} {zc : _root_.Flock.ZcOut}
    {lc : _root_.Flock.LcOut} {xs : Array (Array F128)} {extras : Array _root_.Flock.Claim} (tr : FlockComplete.Trace)
    (hpts : FlockComplete.LigLock.Fed.PtsAt st.m st.fold.kLog W R zc lc)
    (h1 : ∀ i < st.m - 13, W (R + 1) i = C (R + 1) i)
    (h3 : ∀ i < st.m - 6, W (R + 3 + i) 0 = C (R + 3 + i) 0)
    (h5 : ∀ i < st.fold.kLog - 6, W (R + 3 + (st.m - 6) + 2 + i) 0 = C (R + 3 + (st.m - 6) + 2 + i) 0)
    (hL : L₁ = L₂) (hr : ∀ k < 2 + xs.size, ∀ i < 7, W (L₁ + k) i = C (L₁ + k) i)
    (hg : ∀ i < 2 + xs.size, W (L₁ + (2 + xs.size)) i = C (L₁ + (2 + xs.size)) i) (hx : extras.map (·.x) = xs) :
    FlockComplete.Replay.Rs.sws (_root_.Flock.abcClaims st.m st.kLog zc lc ++ _root_.Flock.Firewall.Finish.claimsOf xs)
        ⟨fun k e => W (L₁ + k) e, fun e => W (L₁ + (2 + xs.size)) e⟩ =
      FlockComplete.Replay.Rs.sws (_root_.Flock.abcClaims st.m st.kLog
          (FlockComplete.Replay.Zc.out st.m (FlockComplete.zcCoins C R) tr)
          (FlockComplete.Replay.Lc.out st.fold st.m tr (FlockComplete.Replay.Zc.out st.m (FlockComplete.zcCoins C R) tr)
            (FlockComplete.lcCoins C (R + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras)
        ⟨fun k => C (L₂ + k), C (L₂ + (2 + extras.size))⟩ := by
  subst hL
  have hsz : extras.size = xs.size := by rw [← hx, Array.size_map]
  have hpts' := FlockComplete.LigLock.Fed.ptsAt_congr h1 h3 h5 hpts
  have hcs : (_root_.Flock.abcClaims st.m st.kLog zc lc ++ _root_.Flock.Firewall.Finish.claimsOf xs).size =
      2 + xs.size := by
    rw [Array.size_append, FlockComplete.abcClaims_size]
    simp only [_root_.Flock.Firewall.Finish.claimsOf, Array.size_map]
  refine (FlockComplete.LigLock.Fed.sws_coins_congr _
      (cs' := ⟨fun k => C (L₁ + k), C (L₁ + (2 + extras.size))⟩) (fun k hk i hi => ?_) (fun k hk => ?_)).trans
    (FlockComplete.Replay.Rs.sws_congr ?_ (FlockComplete.LigLock.Fed.ptsAt_claims_x hpts'
      (FlockComplete.LigLock.Fed.ptsAt_out st.m st.fold C R tr) (fun k => claimsOf_x hx k)) _)
  · rw [hcs] at hk
    exact hr k hk i hi
  · rw [hcs] at hk
    show W (L₁ + (2 + xs.size)) k = C (L₁ + (2 + extras.size)) k
    rw [hsz]
    exact hg k hk
  · rw [hcs, Array.size_append, FlockComplete.abcClaims_size, hsz]

/-- The loop's own tables are `RealTabs` (`realTabs_realOf`'s proof). -/
theorem realTabs_of {R : Type}
    (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (ρ : R) {s0 : _} {spec : SessionSpec} {sts : Array Stmt}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts)) : Count.RealTabs sts (realOf src x w ρ) := by
  intro k tr hr h
  unfold realOf at h
  rw [hs] at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i a _
    obtain ⟨g, d, hi⟩ := a
    dsimp only at h
    split at h
    · cases h
    · rename_i l hl
      split at h
      · cases h
      · rename_i t ht
        cases h
        exact ⟨g, d, l, hl, ht⟩

variable (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Worker) (x : Msg) (w : Flock.Wit)

set_option maxHeartbeats 1000000 in
/-- **The loop's level 0 at the opened coins**: at (c.3)'s context on the loop's tables, table `j`'s rep `r`'s rounds
`ligP0 + 1 + i` (`1 ≤ i ≤ k₀ + 2`) send, tagged `0`, level 0's bodies at the opened coins from `ligP0`, its table, the
basis of its switched vectors at the opened coins (`Masked.ligBasis`), its layout (`Masked.layE`) and its request's
pads. -/
theorem l0_loop (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hL0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Chk} {m₀ ρ₀ : Msg}
    {ops : List (Msg × Msg)} (hF : FinCtx (LeanRelease.ofLoopLink L₀ src (realOf src)) wk x w ν t κ m₀ ρ₀ ops)
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    (j : Fin (Masked.jOf x)) (r : Fin 2) {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray}
    {q : _root_.Flock.Firewall.Finish.RepIn} {f : ByteArray}
    (hRep : RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec (2 * j.val + r.val) tb hi q f)
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (viewPoints x ops spec) = .ok extras) :
    ∃ T₁, Masked.ligP0 x j r + (Masked.zkTab x j).sch.k0 + 3 ≤ ops.length ∧
      ∀ i, 1 ≤ i → i ≤ (Masked.zkTab x j).sch.k0 + 2 →
        ((m₀, ρ₀) :: ops)[Masked.ligP0 x j r + 1 + i]?.map Prod.fst =
          some (0 :: FlockComplete.LigLock.l0BodyAt (Masked.ligCoin (finCoins x ops)) (Masked.ligP0 x j r) [] T₁
            (FlockComplete.flat tb.cm0.toC.cap) (Masked.zkTab x j).sch tb.g
            (Masked.ligBasis x (finCoins x ops) w t j r extras) (Masked.layE x j)
            (_root_.Flock.Firewall.Finish.padsOf q.draws).h r.val tb.extra (fun _ _ => 0) i) := by
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  have hx := Masked.live_of hsc hR
  have hj := Masked.lt_of_jOf j
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hq : 2 * j.val + r.val < 2 * spec.tables.size := by rw [hspec]; omega
  have hq2 : 2 * j.val + r.val < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hjj : (2 * j.val + r.val) / 2 = j.val := by omega
  have hmod : (2 * j.val + r.val) % 2 = r.val := by omega
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst]
    exact (Masked.zkTab x j).hz
  have hm25 : 25 ≤ (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m := by
    rw [hst]
    exact (htabs j.val hj).1
  have hk6 : 6 ≤ (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.c.kLog := by
    rw [hst]
    exact (Masked.zkTab x j).wf.1.hk6
  have hm35 : (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m ≤ 35 := by
    rw [hst]
    exact (htabs j.val hj).2.1
  obtain ⟨hpts2, -, hregs⟩ := Masked.live_bounds hx hsx
  have hpr : spec.points * (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.regions.size ≤ 2048 := by
    rw [hst, hpts2]
    have := hregs j.val hj
    exact Nat.le_trans (Nat.mul_le_mul_left 2 this) (by norm_num)
  have hn64 := Count.sched_n hz
  -- the loop's level 0
  have hP := LeanRelease.ofLoop_progChecks_sizes Masked.dataE
    (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (realOf src) x w (Loop.sizesOk_of_serves hServe)
    (fun ρ _ _ _ hs => realTabs_of src x w ρ hs) hL0
  obtain ⟨tr, hr', T₀, T₁, sws, hreal, hlen, -, hbody, hJeq, zc, lc, hpts, hsws⟩ :=
    LeanRelease.ofLoop_l0 Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (realOf src) wk x w
      hx hServe (ν, t) κ m₀ ρ₀ ops hF.1 hF.2.1 hF.2.2.2.1 hF.2.2.2.2.2 hP (realTabs_of src x w (ν, t) hsx) hsx hj
      r.isLt rfl
  rw [hRep.1, Except.ok.injEq, Prod.mk.injEq] at hreal
  obtain ⟨rfl, rfl⟩ := hreal
  -- its table, schedule and layout are the session's
  obtain ⟨schs, hschs, hschj⟩ := Masked.schedsPZk_live hx
  obtain ⟨htabst, hschtb⟩ := finRepTab (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hschs hq hRep
  rw [hjj] at htabst hschtb
  have htbsch : tb.sch = (Masked.zkTab x j).sch := by
    have e := hschtb.symm.trans (hschj j)
    simp only [Prod.mk.injEq] at e
    exact e.1
  have hj' : j.val < (FlockDeploy.tabsOf (Masked.stsOf x)).size := by simpa [FlockDeploy.tabsOf] using hj
  have htabj : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) := by
    rw [getElem!_pos _ j.val hj']
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    rfl
  have hts : tb.st = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) := htabst.symm.trans htabj
  have hm1 : (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m = tb.st.m := by rw [hst, hts]; rfl
  have hk1 : (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.c.kLog = tb.st.fold.kLog := by rw [hst, hts]; rfl
  have hlay : _root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog (Masked.zkTab x j).sch.k0 = Masked.layE x j := by
    rw [hts]
    rfl
  have hP0 : Masked.ligP0 x j r = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! :=
    (Masked.repSlot_eq hsx (q := 2 * j.val + r.val) hz (fun _ => 0)).trans (Nat.add_zero _)
  have hPR : FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! =
      (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m + (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.c.kLog - 6 +
        (2 + spec.points * (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.regions.size + 1) := rfl
  -- the opened coins
  have hWC : ∀ p e, p < ops.length → e < loopNr x p → e < 65536 →
      (wordsAt x (ops.map fun q => ZKMal.CoinLayer.rd q.2) p)[e]! = Masked.ligCoin (finCoins x ops) p e :=
    fun p e hp he hc => words_slot hF.1 hp he hc
  -- the draws
  have hdraws : ((LeanRelease.ofLoop Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src)
      (realOf src)).drawsAt x (ν, t) (2 * j.val + r.val)) = q.draws := by
    obtain ⟨-, hq', -⟩ := hRep
    unfold repInOf at hq'
    obtain ⟨⟨lay, ms⟩, hds, hq'⟩ := bok hq'
    have hso : startOf x (2 * j.val + r.val) = .ok (lay, ms) := by
      unfold startOf
      rw [hsx]
      exact hds
    cases hq'
    show _root_.Flock.Firewall.Frame.Draws.ofBytes
      (((startOf x (2 * j.val + r.val)).toOption.map (·.1)).getD default) (L₀.draws (ν, t) (2 * j.val + r.val)) = _
    rw [hso]
    rfl
  -- the extra claims' coordinates
  have hqx : q.xs = _root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
      (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
        (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0)) := by
    obtain ⟨-, hq', -⟩ := hRep
    unfold repInOf at hq'
    obtain ⟨⟨lay, ms⟩, -, hq'⟩ := bok hq'
    cases hq'
    rfl
  have hext' : (FlockDeploy.tabsOf (Masked.stsOf x))[(2 * j.val + r.val) / 2]!.extra (viewPoints x ops spec) =
      .ok extras := by rw [hjj]; exact hext
  have hxs := finRepXs L₀ src (realOf src) wk x w hsc hR ν t κ m₀ ρ₀ ops hF s0 spec hsx _ hq extras hext' tb hi q f
    hRep
  rw [hqx] at hxs
  -- the switched vectors
  obtain ⟨hp1, hp3, hp5, hp7, hpN⟩ := loopNr_piop hsx hq2 hz (by omega) hk6
  simp only [hm1, hk1] at hp1 hp3 hp5 hp7 hpN
  obtain ⟨Rz, hRz⟩ : ∃ Rz, FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j.val + r.val) = Rz := ⟨_, rfl⟩
  obtain ⟨XS, hXS⟩ : ∃ XS, _root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
      (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
        (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0)) = XS := ⟨_, rfl⟩
  rw [hRz] at hp1 hp3 hp5 hp7 hpN hpts hsws hJeq
  rw [hXS] at hsws hJeq hxs
  have hXn : XS.size = spec.points * (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.regions.size := by
    rw [hPR] at hJeq
    omega
  have h1 : ∀ i < tb.st.m - 13, (wordsAt x (ops.map fun q => ZKMal.CoinLayer.rd q.2) (Rz + 1))[i]! =
      Masked.ligCoin (finCoins x ops) (Rz + 1) i :=
    fun i hi => hWC _ _ (by omega) (by rw [hp1]; omega) (by omega)
  have h3 : ∀ i < tb.st.m - 6, (wordsAt x (ops.map fun q => ZKMal.CoinLayer.rd q.2) (Rz + 3 + i))[0]! =
      Masked.ligCoin (finCoins x ops) (Rz + 3 + i) 0 :=
    fun i hi => hWC _ _ (by omega) (by rw [hp3 i hi]; omega) (by omega)
  have h5 : ∀ i < tb.st.fold.kLog - 6,
      (wordsAt x (ops.map fun q => ZKMal.CoinLayer.rd q.2) (Rz + 3 + (tb.st.m - 6) + 2 + i))[0]! =
        Masked.ligCoin (finCoins x ops) (Rz + 3 + (tb.st.m - 6) + 2 + i) 0 :=
    fun i hi => hWC _ _ (by omega) (by rw [hp5 i hi]; omega) (by omega)
  have hr : ∀ k < 2 + XS.size, ∀ i < 7,
      (wordsAt x (ops.map fun q => ZKMal.CoinLayer.rd q.2) (Rz + (3 + (tb.st.m - 6) + 3 + (tb.st.fold.kLog - 6)) + k))[i]! =
        Masked.ligCoin (finCoins x ops) (Rz + (3 + (tb.st.m - 6) + 3 + (tb.st.fold.kLog - 6)) + k) i := by
    intro k hk i hi
    have e7 := hp7 k (by omega)
    rw [show Rz + 3 + (tb.st.m - 6) + 3 + (tb.st.fold.kLog - 6) + k =
      Rz + (3 + (tb.st.m - 6) + 3 + (tb.st.fold.kLog - 6)) + k by omega] at e7
    exact hWC _ _ (by omega) (by rw [e7]; omega) (by omega)
  have hg : ∀ i < 2 + XS.size,
      (wordsAt x (ops.map fun q => ZKMal.CoinLayer.rd q.2)
        (Rz + (3 + (tb.st.m - 6) + 3 + (tb.st.fold.kLog - 6)) + (2 + XS.size)))[i]! =
        Masked.ligCoin (finCoins x ops) (Rz + (3 + (tb.st.m - 6) + 3 + (tb.st.fold.kLog - 6)) + (2 + XS.size)) i := by
    intro i hi
    have eN := hpN
    rw [show Rz + 3 + (tb.st.m - 6) + 3 + (tb.st.fold.kLog - 6) +
        (2 + spec.points * (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.regions.size) =
      Rz + (3 + (tb.st.m - 6) + 3 + (tb.st.fold.kLog - 6)) + (2 + XS.size) by omega] at eN
    exact hWC _ _ (by omega) (by rw [eN]; omega) (by omega)
  have e := sws_of_pts (st := tb.st) (W := fun p e => (wordsAt x (ops.map fun q => ZKMal.CoinLayer.rd q.2) p)[e]!)
    (C := Masked.ligCoin (finCoins x ops)) (R := Rz) (L₁ := Rz + (3 + (tb.st.m - 6) + 3 + (tb.st.fold.kLog - 6)))
    (L₂ := Rz + 3 + (tb.st.m - 6) + 3 + (tb.st.fold.kLog - 6)) (zc := zc) (lc := lc) (xs := XS) (extras := extras)
    (Masked.trsE x w (Masked.UE.tapeZk t))[j.val]! hpts h1 h3 h5 (by omega) hr hg hxs
  beta_reduce at e
  have hbasis : FlockComplete.basisVals sws (tb.st.m - 7) = Masked.ligBasis x (finCoins x ops) w t j r extras := by
    have hzc : ∀ L, FlockComplete.Replay.Zc.coinsAt (Masked.ligRespE (finCoins x ops)) L =
        FlockComplete.zcCoins (Masked.ligCoin (finCoins x ops)) L := fun L => Masked.zcCoins_ligRespE _ L
    have hlc : ∀ L n, FlockComplete.Replay.Lc.coinsAt (Masked.ligRespE (finCoins x ops)) L n =
        FlockComplete.lcCoins (Masked.ligCoin (finCoins x ops)) L n := fun L n => Masked.lcCoins_ligRespE _ L n
    have hrc : ∀ L n, FlockComplete.Replay.Rs.coinsAt (Masked.ligRespE (finCoins x ops)) L n =
        ⟨fun k => Masked.ligCoin (finCoins x ops) (L + k), Masked.ligCoin (finCoins x ops) (L + n)⟩ :=
      fun L n => Masked.rsCoins_ligRespE _ L n
    unfold Masked.ligBasis Masked.ligSt Masked.swsE
    rw [htabst, hzc, hlc, hrc, Masked.repBase_eq hsx, hRz]
    refine congrArg (fun s => FlockComplete.basisVals s (tb.st.m - 7)) ?_
    rw [hsws]
    refine Eq.trans ?_ (e.trans ?_)
    · with_reducible rfl
    · with_reducible rfl
  -- level 0's bodies
  refine ⟨T₁, by rw [hP0, ← htbsch]; omega, fun i h1 h2 => ?_⟩
  rw [hP0, hbody i h1 (by rw [htbsch]; exact h2)]
  refine congrArg some (congrArg (List.cons 0) ?_)
  rw [htbsch, hlay, hbasis, hmod, hdraws]
  -- every implicit named: unifying against the goal unfolds `l0BodyAt`
  have e := l0Body_shift (W := fun p e => (wordsAt x (ops.map fun q => ZKMal.CoinLayer.rd q.2) p)[e]!)
    (C := Masked.ligCoin (finCoins x ops))
    (R0 := FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! + 1 - 1)
    (R1 := FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]!)
    (P := []) (T := T₁) (cap := FlockComplete.flat tb.cm0.toC.cap) (sch := (Masked.zkTab x j).sch) (t := tb.g)
    (b := Masked.ligBasis x (finCoins x ops) w t j r extras) (lay := Masked.layE x j)
    (h := (_root_.Flock.Firewall.Finish.padsOf q.draws).h) (rep := r.val) (extra := tb.extra) (nonce := fun _ _ => 0)
    (by omega) ?hW i
  beta_reduce at e
  refine Eq.trans ?_ (e.trans ?_)
  · rw [show q.draws.h = (_root_.Flock.Firewall.Finish.padsOf q.draws).h from rfl]
  · with_reducible rfl
  case hW =>
    intro a e ha he
    have hle : (if a = 0 then (Masked.zkTab x j).sch.n else l0Nc a) ≤ 64 := by
      unfold l0Nc
      split_ifs <;> omega
    refine hWC _ _ (by rw [← htbsch] at ha; omega) ?_ (by omega)
    rw [loopNr_l0 hsx hq2 hz (by omega) hk6 ha]
    exact he

set_option maxHeartbeats 1000000 in
/-- **Level 0's values at the views' source, from the cut session's bodies**: at (c.3)'s context on the loop's tables,
where the cut session's bodies `ligP0 + i` (`1 ≤ i ≤ k₀ + 2`) from the views' source are the cut opening's from that
source at the opened coins, table `j`'s rep `r`'s masked values from `zp 64` to `h_ab` are level 0's clear values
(`l0Clear`) plus the request's pads (`masked_of_cutL0`): the history's messages there are the cut session's bodies
(`msg_runCut`) and the loop's (`l0_loop`). -/
theorem swSrcE_l0_of_cut (vs : finS.SwVals)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hL0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Chk} {m₀ ρ₀ : Msg}
    {ops : List (Msg × Msg)} (hF : FinCtx (LeanRelease.ofLoopLink L₀ src (realOf src)) wk x w ν t κ m₀ ρ₀ ops)
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    (j : Fin (Masked.jOf x)) (r : Fin 2) {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray}
    {q : _root_.Flock.Firewall.Finish.RepIn} {f : ByteArray}
    (hRep : RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec (2 * j.val + r.val) tb hi q f)
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (viewPoints x ops spec) = .ok extras)
    {P : List UInt8} {T : F128} {cap : ByteArray}
    (hcut : ∀ i, 1 ≤ i → i ≤ (Masked.zkTab x j).sch.k0 + 2 → Masked.ligP0 x j r + i ≤ refK x →
      (Masked.runCut x (finCoins x ops) (Masked.ligP0 x j r + i) spec
          (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t)).2[Masked.ligP0 x j r + i]! =
        FlockComplete.LigLock.l0BodyAt (Masked.ligCoin (finCoins x ops)) (Masked.ligP0 x j r) P T cap
          (Masked.zkTab x j).sch tb.g (Masked.ligBasis x (finCoins x ops) w t j r extras) (Masked.layE x j)
          (FlockComplete.LigLock.padsL0 (Masked.ligCoin (finCoins x ops)) (Masked.ligP0 x j r) (Masked.zkTab x j).sch
            tb.g (Masked.ligBasis x (finCoins x ops) w t j r extras) (Masked.layE x j) tb.extra r.val
            ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val))
          r.val tb.extra (fun _ _ => 0) i)
    {k : ℕ} (hlo : (Masked.layE x j).zp 64 ≤ k) (hhi : k < (Masked.layE x j).hab) :
    ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val).masked k =
      FlockComplete.LigLock.l0Clear (Masked.ligCoin (finCoins x ops)) (Masked.ligP0 x j r) (Masked.zkTab x j).sch tb.g
          (Masked.ligBasis x (finCoins x ops) w t j r extras) (Masked.layE x j) tb.extra r.val k +
        (_root_.Flock.Firewall.Finish.padsOf q.draws).h[k]! := by
  obtain ⟨T₁, hlen, hbody⟩ := l0_loop L₀ src wk x w hsc hR hServe hL0 hF hsx j r hRep hext
  have hlast : (LeanRelease.ofLoopLink L₀ src (realOf src)).last x = refK x + 1 :=
    LeanRelease.loop_last loopNr Masked.dataE _ x
  have hl := hF.1
  have hsrc : Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t =
      Masked.tabSrcCut Masked.PE x (Masked.czExe Masked.PE x (finCoins x ops))
        (finS.swCutOf x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t) := rfl
  refine FlockComplete.LigLock.masked_of_cutL0 (P := P) (T := T) (cap := cap) (P' := []) (T' := T₁)
    (cap' := FlockComplete.flat tb.cm0.toC.cap)
    (nonce := fun _ _ => 0) rfl rfl (fun i h1 h2 => ?_) hlo hhi
  have hlt : Masked.ligP0 x j r + 1 + i < ((m₀, ρ₀) :: ops).length := by
    simp only [List.length_cons]
    omega
  obtain ⟨s', spec', hsx', hm⟩ := msg_runCut vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w hE
    hc hlast ν t κ m₀ ρ₀ ops hF.1 hF.2.1 hF.2.2.1 hF.2.2.2.2.2 (Masked.ligP0 x j r + 1 + i) (by omega) (by omega)
  rw [hsx] at hsx'
  cases hsx'
  rw [← hsrc, show Masked.ligP0 x j r + 1 + i - 1 = Masked.ligP0 x j r + i by omega, hcut i h1 h2 (by omega),
    dropDom_l0Body h1 h2] at hm
  have hb : (((m₀, ρ₀) :: ops)[Masked.ligP0 x j r + 1 + i]'hlt).1 =
      0 :: FlockComplete.LigLock.l0BodyAt (Masked.ligCoin (finCoins x ops)) (Masked.ligP0 x j r) [] T₁
        (FlockComplete.flat tb.cm0.toC.cap) (Masked.zkTab x j).sch tb.g
        (Masked.ligBasis x (finCoins x ops) w t j r extras) (Masked.layE x j)
        (_root_.Flock.Firewall.Finish.padsOf q.draws).h r.val tb.extra (fun _ _ => 0) i := by
    have e := hbody i h1 h2
    rw [List.getElem?_eq_getElem hlt] at e
    exact Option.some.inj e
  exact (List.cons.inj (hm.symm.trans hb)).2

set_option maxHeartbeats 1000000 in
/-- **Level 0's values at the views' source** (`Loop.swSrcE_l0_live`): at (c.3)'s context on the loop's tables, where
the earlier reps return from the views' source cut at the opened coins and table `j`'s rep `r`'s PIOP returns
(`Masked.runCut_cutL0_of_ret`'s premises), its masked values from `zp 64` to `h_ab` are level 0's clear values at the
opened coins (`l0Clear`: the OOD reply, each lane round's pair and the folded weight's products with the extra lanes)
plus the request's pads. -/
theorem swSrcE_l0_live (vs : finS.SwVals)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hL0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Chk} {m₀ ρ₀ : Msg}
    {ops : List (Msg × Msg)} (hF : FinCtx (LeanRelease.ofLoopLink L₀ src (realOf src)) wk x w ν t κ m₀ ρ₀ ops)
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : _root_.Flock.V _) = .ok schs)
    {cms : Array FlockComplete.Commit}
    (hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut Masked.hsE (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle
        (schs[i]!.1.levels.getD 0 default)
        (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t i)).go l = (.ok cms, l))
    (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hexts : ∀ k < 2 * j.val + r.val, ∃ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras)
    (hprev : ∀ k < 2 * j.val + r.val, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras → ∀ ic,
      (∀ l, (FlockComplete.padsTreeCut Masked.hsE
        (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.m
          (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.kLog schs[k / 2]!.1.k0).len +
          _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
        ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t (k / 2)).rep (k % 2))).go l =
          (.ok ic, l)) →
      ∀ l : Array Msg, l.size = Masked.repBase x k →
      ∃ p tr l', ((FlockComplete.repCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) (refK x)) Masked.hsE
          (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! schs[k / 2]!.1 (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 cms[k / 2]! extras ic
          ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t (k / 2)).rep (k % 2))
          (Masked.priorE (finCoins x ops) (k / 2) (k % 2)) (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[k / 2]!).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l =
          (.ok (p, tr), l') ∧ l'.size = Masked.repBase x (k + 1))
    {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray} {q : _root_.Flock.Firewall.Finish.RepIn}
    {f : ByteArray} (hRep : RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec (2 * j.val + r.val) tb hi q f)
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (viewPoints x ops spec) = .ok extras)
    {ic : FlockComplete.InnerCommit}
    (hic : ∀ l, (FlockComplete.padsTreeCut Masked.hsE
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val)).go l = (.ok ic, l))
    {l : Array Msg} (hl : l.size = Masked.repBase x (2 * j.val + r.val))
    {res} {tr₁ : _root_.Flock.Transcript} {l₁ : Array Msg}
    (hpiop : ((FlockComplete.piopCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) (refK x))
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!
      (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0)
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val)
      (FlockComplete.flat cms[j.val]!.cap) ic.root extras (Masked.priorE (finCoins x ops) j.val r.val)
      (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[j.val]!).run
      (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8)).go l =
      (.ok (res, tr₁), l₁))
    {k : ℕ} (hlo : (Masked.layE x j).zp 64 ≤ k) (hhi : k < (Masked.layE x j).hab) :
    ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val).masked k =
      FlockComplete.LigLock.l0Clear (Masked.ligCoin (finCoins x ops)) (Masked.ligP0 x j r) (Masked.zkTab x j).sch tb.g
          (Masked.ligBasis x (finCoins x ops) w t j r extras) (Masked.layE x j) tb.extra r.val k +
        (_root_.Flock.Firewall.Finish.padsOf q.draws).h[k]! :=
  swSrcE_l0_of_cut L₀ src wk x w vs hsc hR hServe hL0 hE hc hF hsx j r hRep hext
    (P := tr₁.pending.data.toList) (T := res.2.2.2.1) (cap := FlockComplete.flat cms[j.val]!.cap)
    (fun i _ h2 hn => by
      have e := Masked.runCut_cutL0_of_ret (Masked.live_of hsc hR) w t (finCoins x ops)
        (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t) hsx hs hcms j r hexts hprev hext hic hl
        hpiop tb.g tb.extra (n := Masked.ligP0 x j r + i) (Nat.le_add_right _ _) (by omega) hn
      rwa [Nat.add_sub_cancel_left] at e)
    hlo hhi

end Loop

end Claims.ZKServer
