module

public import Proofs.Flock.SessionLoop.RoundPiop
public import Proofs.Flock.SessionLoop.RoundSwitchSv
public import Proofs.Flock.SessionLoop.RoundOffCap
public import Proofs.Flock.SessionLoop.PcsRank
public import Proofs.Flock.Claims.ZK.FinishFwTarget
public import Proofs.Flock.Claims.ZK.PiopFree

/-!
# The opening's binding at a rep's round after its PIOP

At rep stream `2j + r`'s round after its algebraic phase and PIOP (`Masked.repSlot` `0`, `Masked.ligP0`), the frame
releases the opening's binding (`openBind`) at the ring switch's target `T` and level 0's cap, and the switched session
sends the same binding: both targets are `Σ_k γ_k·dot (eq r''_k) (t(s_k))` at the opened coins from `Masked.rankSlot`,
and the switched vectors agree at every claim (`Loop.switch_sv`).

* `FlockComplete.rsTarget`: that target at the coins `coin` from `L`; `piopCut_T`: a returned cut PIOP's target.
* `Bind.sched_openBind`: the loop schedules the binding at the target of its zerocheck's and lincheck's claims.
* `LeanRelease.unrefused_open`: the coins show no refusal by the round.
* `Masked.runCut_openBody`, `Loop.enc_openBind`: the switched session sends the binding there, tagged `0`.
* `Loop.rsTarget_raw`: the target at the schedule's coins past the algebraic phase is the one at the opened coins.
* `LeanRelease.openBind_frame`: the frame releases the loop's binding as it came (`roundFree_openBind`).
* **`LeanRelease.release_sched_openBind_pf`**: `release_sched_openBind`'s statement.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockComplete

open Flock Transcript Replay

/-- **The ring switch's target at coins from `L`**: `Σ_k γ_k·dot (eq r''_k) (t(s_k))` over `N` claims, claim `k`'s
`r''` at `L + k` and the batching `γ` at `L + N`. -/
def rsTarget (coin : ℕ → ℕ → F128) (L : ℕ) (sv : ℕ → Array F128) (N : ℕ) : F128 :=
  dot ((Array.range N).map (coin (L + N)))
    (pushes (fun k => dot (eqTable ((Array.range 7).map (coin (L + k)))) (FlockSoundness.Refine.tOf (sv k))) N)

theorem rsTarget_congr {coin coin' : ℕ → ℕ → F128} {L L' N : ℕ} {sv sv' : ℕ → Array F128}
    (hg : ∀ i < N, coin (L + N) i = coin' (L' + N) i) (hr : ∀ k < N, ∀ i < 7, coin (L + k) i = coin' (L' + k) i)
    (hs : ∀ k < N, sv k = sv' k) : rsTarget coin L sv N = rsTarget coin' L' sv' N := by
  unfold rsTarget
  have e1 : (Array.range N).map (coin (L + N)) = (Array.range N).map (coin' (L' + N)) :=
    Array.map_congr_left fun i hi => hg i (by simpa using hi)
  have e2 : pushes (fun k => dot (eqTable ((Array.range 7).map (coin (L + k)))) (FlockSoundness.Refine.tOf (sv k))) N =
      pushes (fun k => dot (eqTable ((Array.range 7).map (coin' (L' + k)))) (FlockSoundness.Refine.tOf (sv' k))) N :=
    pushes_congr fun k hk => by
      have e3 : (Array.range 7).map (coin (L + k)) = (Array.range 7).map (coin' (L' + k)) :=
        Array.map_congr_left fun i hi => hr k hk i (by simpa using hi)
      rw [e3, hs k hk]
  rw [e1, e2]

/-- **A returned cut PIOP's ring-switch target** is `rsTarget` at the coins past its zerocheck's `3 + (m − 6)` and
lincheck's `3 + (k_log − 6)` rounds, over the source's switched vectors, with nothing pending after it. -/
theorem piopCut_T {coin : ℕ → ℕ → F128} {n : ℕ} {st : Setup} (hm : 14 ≤ st.m) {lay : Zk.PadLayout}
    {src : RepSrc} {cap root : ByteArray} {extras : Array Claim} {prior : Array (List F128)} {words : List ℕ}
    {t₀ : Transcript} {l : Array Claims.Msg} {res t₃ l₃}
    (hN : l.size + 3 + (st.m - 6) + (3 + (st.fold.kLog - 6)) + (2 + extras.size) + 1 ≤ n)
    (h : ((piopCut (asksCut coin n) st lay src cap root extras prior words).run t₀).go l = (.ok (res, t₃), l₃)) :
    res.2.2.2.1 = rsTarget coin (l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6)) src.sv (extras.size + 2) ∧
      t₃.pending = ByteArray.empty := by
  obtain ⟨zp, lp, ss, T, sws⟩ := res
  obtain ⟨zc, t₁, l₁, lc, t₂, l₂, h₁, h₂, -, h₃⟩ := piopCut_split_rank h
  obtain ⟨-, hl₁⟩ := zerocheckCut_at coin n hm _ _ _ _ _ h₁
  obtain ⟨-, hl₂⟩ := lincheckCut_at coin n st.fold st.m lay src.masked zc h₂
  have hsz : (abcClaims st.m st.kLog zc lc ++ extras).size = extras.size + 2 := by
    rw [Array.size_append, abcClaims_size]
    omega
  obtain ⟨res₃, t₃', L₃, e₃, -, hp₃⟩ := Claims.ZKServer.Switch.ringSwitchCut_bodies coin n src.sv
    (abcClaims st.m st.kLog zc lc ++ extras) (by rw [hsz]; omega) t₂ l₂ (by rw [hsz]; omega)
  have e₃' := e₃.symm.trans h₃
  simp only [Prod.mk.injEq, Except.ok.injEq] at e₃'
  obtain ⟨⟨-, ht₃⟩, -⟩ := e₃'
  have hT := ringSwitchCut_T coin n src.sv _ (by rw [hsz]; omega) h₃
  rw [hsz, hl₂, hl₁] at hT
  refine ⟨hT, ?_⟩
  rw [← ht₃]
  exact hp₃

end FlockComplete

namespace Claims.ZKServer

namespace Bind

open _root_.Flock (F128 Transcript TMW)
open _root_.Flock.Firewall.Frame (Cfg Masker)
open _root_.Flock.Firewall.Drive (Step OpenM stepAsks openStep)
open _root_.Flock.Firewall.Finish (repOpening repPiop commitFw claimsOf ringSwitchP_eq switchVec)
open FlockSoundness.Refine (Frame encs)
open FlockComplete (PS asksCut ringSwitchCut)
open FlockComplete.Replay (bind_go liftTM_ok)
open FlockComplete.LigLock.Fed (feedR sentQ pendC feed_eq CoinAt repOpening_feed cutAsks cutAsks_toC openStep_sent)
open Switch (rsBody)
open Loop

/-- **The loop schedules the opening's binding at its own target** (`sched_binding`): the target is `rsTarget` at the
schedule's coins past the algebraic phase, over the table's switched vectors of its zerocheck's and lincheck's claims,
which are at the points of the algebraic phase's coins (`repPiop_rec`), and the extra claims. -/
theorem sched_openBind {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr : Array (Array F128)} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr = .ok sched) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64)
    (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63)
    (hn0 : t.sch.n ≠ 0) (hcr : cr.size = alg.size + (xs.size + 2) + 1) (hm14 : 14 ≤ t.st.m) {p : ℕ}
    (hp : p = alg.size + (xs.size + 2) + 1) :
    p < sched.size ∧ ∃ zc lc,
      FlockComplete.LigLock.Fed.PtsAt t.st.m t.st.fold.kLog
        (FlockComplete.LigLock.Fed.recCoin (cr.extract 0 alg.size)) 0 zc lc ∧
      ∃ T, T = FlockComplete.rsTarget (fun j i => ((cr.extract alg.size cr.size).toList.getD j #[])[i]!) 0
          (fun k => switchVec t.g (_root_.Flock.abcClaims t.st.m t.st.kLog zc lc ++ claimsOf xs)[k]!)
          (xs.size + 2) ∧
        (sched.getD p (.empty, 0)).1.data.toList =
          encs (openBind T (FlockComplete.flat t.cm0.toC.cap)) ++ (Frame.sq 2 t.sch.n).enc ∧
        (sched.getD p (.empty, 0)).2 = t.sch.n := by
  subst hp
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal hs
  have hAle : alg.size ≤ sched.size := by rw [hrest, Array.size_append]; omega
  obtain ⟨hdrop, hnf⟩ := schedSoFar_sentQ hal hs (by omega)
  have hcsl : (cr.extract alg.size cr.size).toList.length = xs.size + 2 + 1 := by
    simp only [Array.length_toList, Array.size_extract]
    omega
  have hlenF := feed_part _ _ _
    (Count.openStep_end t rep dom xs hi (cr.extract 0 alg.size) hdom hdg hcn hm hx hk hn hq)
    (by rw [hcsl]; exact (openList_bind t.sch (by omega) hn0).1) hnf
  rw [hcsl] at hlenF
  have hsz : sched.size = alg.size + (xs.size + 2) + 2 := by
    have := congrArg List.length hdrop
    simp only [List.length_drop, List.length_map, Array.length_toList] at this
    omega
  refine ⟨by omega, ?_⟩
  rw [show alg.size + (xs.size + 2) + 1 = alg.size + (xs.size + 2 + 1) by omega]
  have hcnt := sched_count hal hs hdom hdg hcn hm hx hk hn hq (i := xs.size + 2 + 1) (by omega)
  rw [(openList_bind t.sch (by omega) hn0).2] at hcnt
  have hc := coinAt_getD (cr.extract alg.size cr.size).toList
  obtain ⟨r, hp, -, hL0⟩ := FlockComplete.LigLock.Fed.openStep_l0 t rep dom xs hi (cr.extract 0 alg.size) hdom hdg
    hcn hm hmk (cr.extract alg.size cr.size).toList
    (fun j i => ((cr.extract alg.size cr.size).toList.getD j #[])[i]!) #[] hc
    (by intro h0; rw [h0] at hlenF; simp at hlenF)
  have hna : r.1.1.2.2.size = xs.size + 2 := by
    have := Count.piop_na (cr.extract 0 alg.size) t.st (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
      (_root_.Flock.Firewall.Finish.flat t.cm0.cap) ByteArray.empty (claimsOf xs) (fun _ => 0) hdg hcn (by decide) hm _
      (Sched.trAlike_init dom hdom) #[]
    rw [hp] at this
    have h' : r.1.1.2.2.size = (claimsOf xs).size + 2 := this
    simpa only [claimsOf, Array.size_map] using h'
  obtain ⟨zc, lc, hcl, hpts⟩ := FlockComplete.LigLock.Fed.repPiop_rec _ _ _ _ _ _ _ hm14 _ hp
  obtain ⟨⟨ss, T, sws⟩, t', L', hL, hLl, ht'⟩ := Switch.ringSwitchCut_bodies
    (fun j i => ((cr.extract alg.size cr.size).toList.getD j #[])[i]!) (xs.size + 2 + 1)
    (fun k => switchVec t.g r.1.1.2.2[k]!) r.1.1.2.2 (by omega) r.1.2 #[] (by simp [hna])
  have e := FlockComplete.ringSwitchCut_T _ _ _ _ (by rw [hna]; omega) hL
  rw [hna] at e
  rw [hcl] at e
  have hLs : L'.size = xs.size + 2 + 1 := by
    have := congrArg List.length hLl
    simp only [Array.length_toList, Array.toList_empty, List.nil_append, List.length_map, List.length_range,
      hna] at this
    exact this
  have key := hL0 (j := 0) (ss := ss) (T := T) (sws := sws) (tr₂ := t') (l₂ := L') (by omega) hk hn0 (by simp)
    (by rw [hLs, Nat.add_zero]; exact hL) (by rw [hLs]; simp only [Array.size_empty, Nat.zero_add]; omega)
  rw [hLl, hLs] at key
  simp only [Array.toList_empty, List.nil_append, Array.size_empty, Nat.sub_zero, Nat.add_zero, hna] at key
  have hk2 := List.getElem_of_eq key (i := xs.size + 2 + 1)
    (by simp only [List.length_map, List.length_take]; omega)
  rw [List.getElem_append_right (by simp)] at hk2
  simp only [List.getElem_map, List.getElem_take, List.length_map, List.length_range, Nat.sub_self,
    List.getElem_range] at hk2
  rw [FlockComplete.LigLock.l0BodyAt_zero, ht', empty_toList, List.nil_append] at hk2
  refine ⟨zc, lc, hpts, T, e.trans rfl, ?_, hcnt⟩
  rw [sched_entry hdrop (by omega)]
  unfold Claims.toMsg at hk2
  rw [hk2, openBind]

end Bind

open Claims Claims.ZKMal
open _root_.Flock (V F128 SessionSpec)
open _root_.Flock.Firewall.Finish (Tab)
open FlockSoundness.Refine (Frame encs)

namespace LeanRelease

open Loop in
/-- **The coins show no refusal by a rep's round after its PIOP**: at a live history of `Masked.repSlot` `0` of rep
stream `2j + r` openings, no outer coin is `1` (`outer_unrefused`), no rank check fails (`rank_unrefused_sched`), and no
`ρ` or `β` check of a rep before it fails (`rho_unrefused`, `beta_unrefused`); the rep's own `ρ` and `β` are later. -/
theorem unrefused_open (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit) (hx : Masked.Live x) (hSv : Serves x)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (κ : (ofLoop T L₀ real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length ≤ refK x)
    (hlive : (ofLoop T L₀ real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true)
    (hr : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false)
    {j r : ℕ} (hj : j < (Masked.stsOf x).size) (hr2 : r < 2)
    (hn : ops.length = Masked.repSlot x (2 * j + r) fun _ => 0) :
    ¬ Masked.Refused x ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) ops.length := by
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨y⟩ := hx.2 j hj
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j + r) / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [show (2 * j + r) / 2 = j by omega, getElem!_pos (Masked.stsOf x) j hj]
    exact y.hz
  have Sq := Masked.slots_order hx hsx hj hr2
  have eP : ops.length = Masked.repBase x (2 * j + r) +
      FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j + r) / 2]! := by
    rw [hn, Masked.repSlot_eq hsx hz, ← Masked.repBase_eq hsx]
    try dsimp only
    try omega
  have hρl : ops.length < Masked.rhoSlot x j r := by
    unfold Masked.rhoSlot
    rw [Masked.repSlot_eq hsx hz, eP, ← Masked.repBase_eq hsx]
    try dsimp only
    omega
  have hRl : Masked.rankSlot x j r < ops.length := by
    rw [eP]
    show Masked.repBase x (2 * j + r) + ((Masked.stsOf x)[j]!.m + (Masked.stsOf x)[j]!.c.kLog - 6) < _
    rw [show (2 * j + r) / 2 = j by omega]
    unfold FlockComplete.piopRoundsZk
    try dsimp only
    omega
  have hcase : ∀ {j' r' : ℕ}, j' < (Masked.stsOf x).size → r' < 2 →
      Masked.betaSlot x j' r' < ops.length ∨ (j' = j ∧ r' = r) ∨
        ops.length < Masked.repBase x (2 * j' + r') := by
    intro j' r' hj' hr'
    have Sp := Masked.slots_order hx hsx hj' hr'
    rcases Nat.lt_trichotomy (2 * j' + r') (2 * j + r) with h | h | h
    · left
      have hm := FlockComplete.ZkTables.repStart_mono spec (Masked.stsOf x)
        (show 2 * j' + r' + 1 ≤ 2 * j + r by omega)
      rw [← Masked.repBase_eq hsx, ← Masked.repBase_eq hsx] at hm
      omega
    · right
      left
      omega
    · right
      right
      have hm := FlockComplete.ZkTables.repStart_mono spec (Masked.stsOf x)
        (show 2 * j + r + 1 ≤ 2 * j' + r' by omega)
      rw [← Masked.repBase_eq hsx, ← Masked.repBase_eq hsx] at hm
      omega
  rintro (⟨j', hj', r', hr', ho, e, he, h1⟩ | ⟨j', hj', r', hr', hth⟩)
  · exact outer_unrefused T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj' hr' ho he h1
  · have Sp := Masked.slots_order hx hsx hj' hr'
    rcases hth with ⟨-, -, -, hrs, hbad⟩ | ⟨hρ, hd⟩ | ⟨hβ, hz'⟩
    · refine rank_unrefused_sched T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj' hr' ?_ hbad
      rcases hcase hj' hr' with h | ⟨h1, h2⟩ | h
      · omega
      · subst h1 h2
        exact hRl
      · omega
    · rcases hcase hj' hr' with h | ⟨h1, h2⟩ | h
      · rw [rho_unrefused T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj' hr' h] at hd
        exact Bool.noConfusion hd
      · subst h1 h2
        omega
      · omega
    · rw [beta_unrefused T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj' hr' hβ] at hz'
      exact Bool.noConfusion hz'

end LeanRelease

end Claims.ZKServer

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

set_option maxHeartbeats 1000000 in
/-- **The cut session at rep `2j + r`'s round after its PIOP, on the opened coins alone**: where the coins show no
refusal by it, the cut session stopped there sends the opening's binding at the PIOP's target, which is `rsTarget` at
the opened coins from `rankSlot` over the source's switched vectors (`piopCut_T`), and level 0's cap: level 0's body `0`
(`runCut_cutL0_of_ret`, `l0BodyAt_zero`), the PIOP returning as the rank check passes (`rank_at_unrefused`). -/
theorem runCut_openBody {x : Msg} (hx : Live x) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (src : ℕ → FlockComplete.TabSrc) {s0 : _}
    {spec : _root_.Flock.SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (stsOf x)) : _root_.Flock.V _) = .ok schs)
    {cms : Array FlockComplete.Commit}
    (hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
          _root_.Flock.Hashes FlockComplete.PS)
        (FlockDeploy.tabsOf (stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default) (src i)).go l = (.ok cms, l))
    (j : Fin (jOf x)) (r : Fin 2) (hN : ¬ Refused x c (ligP0 x j r))
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    {ic : FlockComplete.InnerCommit}
    (hic : ∀ l, (FlockComplete.padsTreeCut hsE
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((src j.val).rep r.val)).go l = (.ok ic, l)) :
    ligP0 x j r < (runCut x c (ligP0 x j r) spec src).2.size ∧
    (runCut x c (ligP0 x j r) spec src).2[ligP0 x j r]! =
      FlockSoundness.Refine.encs (Claims.ZKServer.openBind
        (FlockComplete.rsTarget (ligCoin c) (rankSlot x j.val r.val) ((src j.val).rep r.val).sv (extras.size + 2))
        (FlockComplete.flat cms[j.val]!.cap)) ++ (FlockSoundness.Refine.Frame.sq 2 (zkTab x j).sch.n).enc := by
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have htab' : (FlockDeploy.tabsOf (stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (stsOf x)[j.val]! := by
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) j.val (by simpa [FlockDeploy.tabsOf] using hj),
      getElem!_pos (stsOf x) j.val hj]
    simp [FlockDeploy.tabsOf]
  have hmS : (FlockDeploy.tabsOf (stsOf x))[j.val]!.m = (stsOf x)[j.val]!.m := by rw [htab']; rfl
  have hkS : (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog = (stsOf x)[j.val]!.c.kLog := by rw [htab']; rfl
  have hfS : (FlockDeploy.tabsOf (stsOf x))[j.val]!.fold.kLog = (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog := by
    rw [htab']; rfl
  have hm25 : 25 ≤ (stsOf x)[j.val]!.m := by
    rw [getElem!_pos (stsOf x) j.val hj]
    exact (hall _ hj).1
  have hT : FlockComplete.ZkTables.StmtT (stsOf x)[j.val]! := by
    rw [getElem!_pos (stsOf x) j.val hj]
    exact _root_.Flock.Readable.stmtT_of_scope hx.1 hsx hj
  obtain ⟨⟨hwf, -⟩, hslot⟩ := hT
  have hk6 := hwf.hk6
  have hkm := hwf.hk
  have hst : (stsOf x)[j.val]! = stAt x j := getElem!_pos _ _ _
  have hz : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [show (2 * j.val + r.val) / 2 = j.val by omega, hst]
    exact (zkTab x j).hz
  have hext' := hext
  have hes : extras.size = spec.points * (stsOf x)[j.val]!.regions.size := by
    rw [htab'] at hext'
    rw [FlockComplete.JsonRT.extraClaims_size hext']
    simp [Nat.mul_comm, ptsE]
  have hβ := betaSlot_lt_repBase hx hsx (q := 2 * j.val + r.val) (by omega)
  rw [show (2 * j.val + r.val) / 2 = j.val by omega, show (2 * j.val + r.val) % 2 = r.val by omega] at hβ
  have eβ : betaSlot x j.val r.val = repBase x (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (stsOf x)[j.val]! +
        ((zkTab x j).sch.k0 + 7 + 11 * ((zkTab x j).sch.levels.size - 1) + 2) := by
    unfold betaSlot
    rw [repSlot_eq hsx hz, ← repBase_eq hsx, show (2 * j.val + r.val) / 2 = j.val by omega]
  have eP0 : ligP0 x j r = repBase x (2 * j.val + r.val) + FlockComplete.piopRoundsZk spec (stsOf x)[j.val]! := by
    show repSlot x (2 * j.val + r.val) (fun _ => 0) = _
    rw [repSlot_eq hsx hz, ← repBase_eq hsx, show (2 * j.val + r.val) / 2 = j.val by omega]
    try rfl
  have eP : FlockComplete.piopRoundsZk spec (stsOf x)[j.val]! =
      ((stsOf x)[j.val]!.m + (stsOf x)[j.val]!.c.kLog - 6) +
        (2 + spec.points * (stsOf x)[j.val]!.regions.size + 1) := rfl
  have hK : repBase x (2 * j.val + r.val + 1) ≤ refK x := by
    rw [repBase_eq hsx, refK_eq hsx, ← FlockComplete.ZkTables.repStart_last]
    exact FlockComplete.ZkTables.repStart_mono _ _ (by omega)
  have hR : rankSlot x j.val r.val =
      repBase x (2 * j.val + r.val) + ((stsOf x)[j.val]!.m + (stsOf x)[j.val]!.c.kLog - 6) := rfl
  have hrank0 : rankSlot x j.val 0 ≤ rankSlot x j.val r.val := by
    unfold rankSlot
    rw [repBase_eq hsx, repBase_eq hsx]
    have := FlockComplete.ZkTables.repStart_mono spec (stsOf x) (show 2 * j.val + 0 ≤ 2 * j.val + r.val by omega)
    omega
  have hm14 : 14 ≤ (FlockDeploy.tabsOf (stsOf x))[j.val]!.m := by rw [hmS]; omega
  have hkm' : (FlockDeploy.tabsOf (stsOf x))[j.val]!.fold.kLog ≤ (FlockDeploy.tabsOf (stsOf x))[j.val]!.m := by
    rw [hfS, hkS, hmS]
    exact hkm
  obtain ⟨l, hl⟩ : ∃ l : Array Msg, l.size = repBase x (2 * j.val + r.val) :=
    ⟨Array.replicate _ [], Array.size_replicate⟩
  have hNb : l.size + 3 + ((FlockDeploy.tabsOf (stsOf x))[j.val]!.m - 6) +
      (3 + ((FlockDeploy.tabsOf (stsOf x))[j.val]!.fold.kLog - 6)) + (2 + extras.size) + 1 ≤ refK x := by
    rw [hfS, hkS, hmS, hl, hes]
    omega
  obtain ⟨res, t₃, l₃, hP, -⟩ := FlockComplete.CutFree.piopCut_ret (ligCoin c) (refK x) hm14 hkm'
    (fun a lw ρ hlw hρ => by
      rw [htab'] at hρ ⊢
      exact FlockComplete.JsonRT.fold_ok _ hk6 hslot a hlw hρ)
    (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0)
    ((src j.val).rep r.val) (FlockComplete.flat cms[j.val]!.cap) ic.root extras (priorE c j.val r.val)
    (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!
    (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8) l hNb
    (fun _ _ _ _ _ _ _ _ h₁ h₂ => by
      rw [FlockComplete.MaskRank.masksOf_getElem! hj]
      exact rank_at_unrefused hx hN hj hr2 (by omega) (by omega) (ligCoin c) (refK x) (fun _ _ _ => rfl)
        (FlockDeploy.tabsOf (stsOf x))[j.val]! hmS hkS hfS _ _ _ _
        (t₀ := _root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8) rfl hl h₁ h₂)
  obtain ⟨hTe, hpend⟩ := FlockComplete.piopCut_T hm14 hNb hP
  have hI : l.size + 3 + ((FlockDeploy.tabsOf (stsOf x))[j.val]!.m - 6) + 3 +
      ((FlockDeploy.tabsOf (stsOf x))[j.val]!.fold.kLog - 6) = rankSlot x j.val r.val := by
    rw [hR, hfS, hkS, hmS, hl]
    omega
  have hkb : ∀ k < 2 * j.val + r.val, betaSlot x (k / 2) (k % 2) < ligP0 x j r := by
    intro k hk
    have h1 := betaSlot_lt_repBase hx hsx (q := k) (by omega)
    have h2 : repBase x (k + 1) ≤ repBase x (2 * j.val + r.val) := by
      rw [repBase_eq hsx, repBase_eq hsx]
      exact FlockComplete.ZkTables.repStart_mono _ _ (by omega)
    omega
  have hv := runCut_cutL0_of_ret hx w τ c src hsx hs hcms j r (fun k hk => extra_ok hx hsx c (by omega))
    (fun k hk extras' hxe ic' _ l' hl' => repCut_ret hx c hN hsx hs (by omega) (hkb k hk) _ _ hxe ic' _ hl')
    hext hic hl hP #[] (fun _ _ => 0) (n := ligP0 x j r) le_rfl (by omega) (by omega)
  have hbody : (runCut x c (ligP0 x j r) spec src).2[ligP0 x j r]! =
      FlockSoundness.Refine.encs (Claims.ZKServer.openBind
        (FlockComplete.rsTarget (ligCoin c) (rankSlot x j.val r.val) ((src j.val).rep r.val).sv (extras.size + 2))
        (FlockComplete.flat cms[j.val]!.cap)) ++ (FlockSoundness.Refine.Frame.sq 2 (zkTab x j).sch.n).enc := by
    rw [hv, Nat.sub_self, FlockComplete.LigLock.l0BodyAt_zero, hpend, Claims.ZKServer.empty_toList, List.nil_append,
      hTe, hI]
    rfl
  refine ⟨?_, hbody⟩
  by_contra hge
  have e := congrArg List.length
    ((getElem!_neg (runCut x c (ligP0 x j r) spec src).2 (ligP0 x j r) hge).symm.trans hbody)
  rw [List.length_append, FlockSoundness.Refine.Frame.enc_length] at e
  exact absurd e (by show ¬ (0 = _); omega)

end Masked

end Claims.ZKMal

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 SessionSpec)
open _root_.Flock.Firewall.Finish (Tab)
open FlockSoundness.Refine (Frame encs)

namespace Loop

set_option maxHeartbeats 1000000 in
/-- **The switched session's round after rep stream `2j + r`'s PIOP**: where the coins show no refusal by it, the
session sends the opening's binding at the switched source's ring-switch target from `rankSlot` and level 0's cap,
tagged `0` (`Masked.runCut_openBody`: the views' level-0 commitments and the pads commitment return quietly). -/
theorem enc_openBind (vs : finS.SwVals) (w : Flock.Wit) {x : Msg} (hx : Masked.Live x) {s0 : FlockDeploy.Ins}
    {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) (ν : finS.Nonce)
    (t : finS.Tape) (C : Fin (refK x) → finS.Coin) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hN : ¬ Masked.Refused x C (Masked.ligP0 x j r)) {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (Masked.ptsE C spec) = .ok extras) :
    ∃ T, T = FlockComplete.rsTarget (Masked.ligCoin C) (Masked.rankSlot x j.val r.val)
        ((Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val).rep r.val).sv (extras.size + 2) ∧
      finS.enc x C (Masked.ligP0 x j r) (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t) =
        0 :: (encs (openBind T (FlockComplete.flat (FlockComplete.layerOf (fun a b => _root_.Flock.Sha512.hash (a ++ b))
          (Masked.lv0 x j).d (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val).leaf0
          ((Masked.lv0 x j).d - (Masked.lv0 x j).cap)))) ++ (Frame.sq 2 (Masked.zkTab x j).sch.n).enc) := by
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨schs, hs, hschs⟩ := Masked.schedsPZk_live hx
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hms : ∀ i < (FlockDeploy.tabsOf (Masked.stsOf x)).size,
      (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    intro i hi
    have hi' : i < (Masked.stsOf x).size := by simpa [FlockDeploy.tabsOf] using hi
    rw [getElem!_pos (FlockDeploy.tabsOf (Masked.stsOf x)) i hi]
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    exact (hall i hi').2.2
  let cmOf : ℕ → FlockComplete.Commit := fun i =>
    Classical.choose (FlockComplete.commit0Cut_cap (schs[i]!.1.levels.getD 0 default)
      (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i))
  have hcm : ∀ i, (FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
      _root_.Flock.Hashes V) _root_.Flock.MerkleScheme.hm96Sha512 (schs[i]!.1.levels.getD 0 default)
      (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i) : V _) = .ok (cmOf i) := fun i =>
    (Classical.choose_spec (FlockComplete.commit0Cut_cap (schs[i]!.1.levels.getD 0 default)
      (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i))).1
  have hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
          _root_.Flock.Hashes FlockComplete.PS)
        (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
        (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i)).go l =
      (.ok ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).map cmOf), l) := by
    intro l
    have hq : FlockComplete.quietS.R
        ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
          FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes V)
            (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
            (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i))
        ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
          FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
              _root_.Flock.Hashes FlockComplete.PS)
            (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
            (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i)) :=
      _root_.Flock.With.Sim.mapM_array FlockComplete.quietS _ fun i =>
        FlockComplete.CutEq.commit0Cut_quiet _ _ _ _
    rw [hq l, FlockComplete.mapM_ok_mem cmOf _ fun i hi => ?_]
    rw [hms i (by simpa using hi)]
    exact hcm i
  obtain ⟨ic, hic0⟩ := FlockComplete.padsTreeCut_ok
    (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0).len +
      _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
    ((Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val).rep r.val)
  have hrc := Masked.runCut_openBody hx w t C (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t) hsx hs hcms j r hN hext (ic := ic)
    (fun l => by rw [FlockComplete.CutEq.padsTreeCut_quiet _ _ _ l, hic0])
  have hsrc : Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t =
      Masked.tabSrcCut Masked.PE x (Masked.czExe Masked.PE x C) (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t) :=
    rfl
  have hj' : j.val < (FlockDeploy.tabsOf (Masked.stsOf x)).size := by simpa [FlockDeploy.tabsOf] using hj
  have hci : ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).map cmOf)[j.val]! = cmOf j.val := by
    rw [getElem!_pos ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).map cmOf) j.val (by simpa using hj'),
      Array.getElem_map, Array.getElem_range]
  have hcap' : (cmOf j.val).cap = FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r))
      (schs[j.val]!.1.levels.getD 0 default).d (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val).leaf0
      ((schs[j.val]!.1.levels.getD 0 default).d - (schs[j.val]!.1.levels.getD 0 default).cap) :=
    (Classical.choose_spec (FlockComplete.commit0Cut_cap (schs[j.val]!.1.levels.getD 0 default)
      (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val))).2
  have hlv : schs[j.val]!.1.levels.getD 0 default = Masked.lv0 x j := by
    rw [hschs j]
    try rfl
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [Masked.stsOf_half]
    exact (Masked.zkTab x j).hz
  have h0 : Masked.ligP0 x j r ≠ 0 := by
    show Masked.repSlot x (2 * j.val + r.val) (fun _ => 0) ≠ 0
    rw [Masked.repSlot_eq hsx hz]
    have : 1 ≤ FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j.val + r.val) := by
      unfold FlockComplete.repStartZk
      omega
    omega
  refine ⟨_, rfl, ?_⟩
  rw [enc_finS_of C h0 (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t) hsx hN
    (by rw [← hsrc]; exact hrc.1), ← hsrc, hrc.2, hci, hcap', hlv]

/-- **The ring switch's target at the schedule's coins past `A` is the one at the opened coins from `R + A`**
(`coin_raw`). -/
theorem rsTarget_raw (W : ℕ → Array F128) {R A C L N : ℕ} (sv : ℕ → Array F128) (hN : A + N < C)
    (hL : L = R + A) :
    FlockComplete.rsTarget (fun j i => ((((Array.range C).map fun q => W (R + q)).extract A
        ((Array.range C).map fun q => W (R + q)).size).toList.getD j #[])[i]!) 0 sv N =
      FlockComplete.rsTarget (fun p e => (W p)[e]!) L sv N := by
  subst hL
  refine FlockComplete.rsTarget_congr (fun e _ => ?_) (fun k hk e _ => ?_) (fun _ _ => rfl)
  · simp only [Nat.zero_add]
    rw [coin_raw W (R := R) (show A + N < C by omega)]
    exact congrArg (fun q => (W q)[e]!) (Nat.add_assoc R A N).symm
  · simp only [Nat.zero_add]
    rw [coin_raw W (R := R) (show A + k < C by omega)]
    exact congrArg (fun q => (W q)[e]!) (Nat.add_assoc R A k).symm

end Loop

namespace LeanRelease

open Loop in
set_option maxHeartbeats 4000000 in
/-- **The frame releases the opening's binding as the loop schedules it**: at a live history whose next round is rep
stream `2j + r`'s round after its PIOP (`Masked.repSlot` `0`), on the rep's own tables where `real` answers the loop's,
the loop's table there is of the rep's setup and the rep's schedule, its zerocheck's and lincheck's claims are at the
points of the openings' words from the rep's start, and the release is the opening's binding at the target `rsTarget`
of the openings' words from `rankSlot` over the table's switched vectors of those claims and the extra claims, and the
table's cap. The frame is past the algebraic phase there (`Bind.repState_opened`); the loop scheduled the binding
(`Bind.sched_openBind`) and asks for `z`'s coins (`Loop.loopNr_l0`), and the check passes it as it came
(`roundFree_openBind`). -/
theorem openBind_frame (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoop Masked.dataE L₁ real).toSrv.Worker) {x : Msg} {w : Flock.Wit} (hx : Masked.Live x)
    (hSv : Loop.Serves x) (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoop Masked.dataE L₁ real).toSrv.Chk)
    (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hRT : Count.RealTabs (Masked.stsOf x) (real x w (ν, t)))
    (hlt : ops.length < refK x)
    (hl : (ofLoop Masked.dataE L₁ real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true) {v : Msg}
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hn : ops.length = Masked.repSlot x (2 * j.val + r.val) fun _ => 0) :
    ∃ (tr : Tab) (hir : Array ByteArray) (zc : _root_.Flock.ZcOut) (lc : _root_.Flock.LcOut) (T : F128),
      real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hir) ∧
      tr.st = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) ∧
      tr.sch = (Masked.zkTab x j).sch ∧
      FlockComplete.LigLock.Fed.PtsAt tr.st.m tr.st.fold.kLog
        (fun p e => (wordsAt x (ops.map fun p => CoinLayer.rd p.2) p)[e]!) (Masked.repBase x (2 * j.val + r.val))
        zc lc ∧
      T = FlockComplete.rsTarget (fun p e => (wordsAt x (ops.map fun p => CoinLayer.rd p.2) p)[e]!)
        (Masked.rankSlot x j.val r.val)
        (fun k => _root_.Flock.Firewall.Finish.switchVec tr.g
          (_root_.Flock.abcClaims tr.st.m tr.st.kLog zc lc ++ _root_.Flock.Firewall.Finish.claimsOf
            (_root_.Flock.Firewall.Drive.extras (Masked.stAt x j) (_root_.Flock.Firewall.Drive.linkPoints
              (Masked.stAt x j) (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0))))[k]!)
        ((_root_.Flock.Firewall.Drive.extras (Masked.stAt x j) (_root_.Flock.Firewall.Drive.linkPoints
          (Masked.stAt x j) (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0))).size + 2) ∧
      (ofLoop Masked.dataE L₁ real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
        0 :: (encs (openBind T (FlockComplete.flat tr.cm0.toC.cap)) ++ (Frame.sq 2 tr.sch.n).enc) := by
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hq2 : 2 * j.val + r.val < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  have hstj : (Masked.stsOf x)[j.val]! = Masked.stAt x j := getElem!_pos (Masked.stsOf x) j.val hj
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst]
    exact (Masked.zkTab x j).hz
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hj).1
  have hm35 : (Masked.stAt x j).m ≤ 35 := (htabs j.val hj).2.1
  have hmk : (Masked.stAt x j).merkle = _root_.Flock.MerkleScheme.hm96Sha512 := (htabs j.val hj).2.2
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  have hrk : (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 =
      FlockDeploy.repRounds true spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! := by
    unfold roundsOf
    rw [List.getD_eq_getElem _ _ (by simp only [List.length_map, List.length_range]; omega)]
    simp
  -- the round's place: `A + k` into the rep's rounds, `A` the algebraic phase's, `k` past the ring switch
  obtain ⟨A, hAd⟩ : ∃ A, A = (Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6 := ⟨_, rfl⟩
  obtain ⟨k, hkk⟩ : ∃ k, k = 3 + spec.points * (Masked.stAt x j).regions.size := ⟨_, rfl⟩
  have hRk : Masked.rankSlot x j.val r.val = Masked.repBase x (2 * j.val + r.val) + A := by
    rw [hAd]
    show Masked.repBase x (2 * j.val + r.val) + ((Masked.stsOf x)[j.val]!.m + (Masked.stsOf x)[j.val]!.c.kLog - 6) = _
    rw [hstj]
  have hRS : Masked.repSlot x (2 * j.val + r.val) (fun _ => 0) =
      Masked.repBase x (2 * j.val + r.val) + A + (3 + spec.points * (Masked.stAt x j).regions.size) := by
    rw [Masked.repSlot_eq hsx hz, hst, ← Masked.repBase_eq hsx (2 * j.val + r.val), hAd]
    unfold FlockComplete.piopRoundsZk
    try dsimp only
    omega
  have hnk : ops.length = Masked.repBase x (2 * j.val + r.val) + A + k := by rw [hn, hRS, hkk]
  have hbs1 : 1 ≤ Masked.repBase x (2 * j.val + r.val) := by
    rw [Masked.repBase_eq hsx]
    unfold FlockComplete.repStartZk
    omega
  have hpos : 0 < ops.length := by omega
  -- the extra claims: the table's regions at each link point
  obtain ⟨hmp, hmp0, hpc⟩ := Masked.live_links hx hsx hq2
  have hmp' : (Masked.stAt x j).mPts = spec.mPts := by rw [← hst]; exact hmp
  have hl0 : loopNr x 0 = spec.points * spec.mPts := by
    show (reqOf x 0).2 = _
    rw [reqOf_eq hsx]
    rfl
  have hW0 : (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0).size = spec.points * spec.mPts := by
    have e := wordsAt_size x ops (s := ops.length) (p := 0) hpos le_rfl
    rw [List.take_length] at e
    rw [e, hl0, Nat.min_eq_left hpc]
  have hXS : (_root_.Flock.Firewall.Drive.extras (Masked.stAt x j) (_root_.Flock.Firewall.Drive.linkPoints
      (Masked.stAt x j) (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0))).size =
      spec.points * (Masked.stAt x j).regions.size := by
    rw [Count.extras_size, Count.linkPoints_size, hW0, hmp', Nat.mul_div_cancel _ hmp0]
    exact Nat.mul_comm _ _
  obtain ⟨-, hdom', hxr, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec _ hsx) hq2
  rw [hst] at hxr
  have hXb : (_root_.Flock.Firewall.Drive.extras (Masked.stAt x j) (_root_.Flock.Firewall.Drive.linkPoints
      (Masked.stAt x j) (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0))).size + 2 < 2 ^ 64 := by
    rw [Count.extras_size]
    have := Nat.mul_le_mul_left (Masked.stAt x j).regions.size
      ((Loop.linkPoints_size_le (Masked.stAt x j) (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0)).trans
        (show _ ≤ _root_.Flock.Firewall.Frames.COIN_MAX by rw [hW0]; exact hpc))
    omega
  -- the located round
  obtain ⟨NR, hNd⟩ : ∃ NR, NR = FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! +
      ((Masked.zkTab x j).sch.k0 + 7 + 11 * ((Masked.zkTab x j).sch.levels.size - 1)) := ⟨_, rfl⟩
  have hR4 : (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 = NR + 4 := by
    have e1 : (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m + (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m + ((Masked.stsOf x)[(2 * j.val + r.val) / 2]!.c.kLog - 6) := by
      rw [hst]; omega
    rw [hrk, repRounds_true]
    simp only [hz]
    rw [hNd]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
  have hR0 : A + k < (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 := by
    rw [hR4, hNd, hst]
    unfold FlockComplete.piopRoundsZk
    omega
  obtain ⟨i, hp⟩ : ∃ i, ops.length = i + 2 := ⟨ops.length - 2, by omega⟩
  obtain ⟨a, ha⟩ : ∃ a, A + k = a + 1 := ⟨A + k - 1, by omega⟩
  have hloc : ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1) = (2 * j.val + r.val, a + 1) := by
    have e := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j.val + r.val) (a + 1)
      (by rw [hrsl]; exact hq2) (by omega)
    rw [Nat.zero_add] at e
    rw [← e]
    congr 1
    have hbase : Masked.repBase x (2 * j.val + r.val) =
        1 + ((roundsOf spec (Masked.stsOf x)).take (2 * j.val + r.val)).sum := by
      rw [Masked.repBase_eq hsx, repStartZk_eq spec _ hq2.le]
    omega
  have hj1 : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).2 = a + 1 := by rw [hloc]
  -- the frame's check
  have hlast : (ofLoop Masked.dataE L₁ real).last x = refK x + 1 :=
    loop_last loopNr Masked.dataE (withFrames Masked.dataE L₁ real) x
  have hkd : (ofLoop Masked.dataE L₁ real).kindAt x ((m₀, ρ₀) :: ops).length = .masked :=
    kindAt_masked_of _ (by simp only [List.length_cons]; omega) (by rw [hlast, List.length_cons]; omega)
  obtain ⟨m, out, s', hA, hck⟩ :=
    check_of_live (ofLoop Masked.dataE L₁ real) wk x w (ν, t) κ ((m₀, ρ₀) :: ops) v hl hkd hv
  obtain ⟨ho, ht, -⟩ := answered_ok hA
  have hA' := hA
  rw [ofLoop_answered_same Masked.dataE L₁ real x (ν, t) (m₀, ρ₀) ops hsx (q := i + 1) (by omega) hj1 ho ht] at hA'
  replace hA' := (bind_rankStop_ok hA').1
  -- the loop's placeholder table and its state there
  have hcs : (ops.map fun p => ZKMal.CoinLayer.rd p.2).length = i + 2 := by rw [List.length_map]; omega
  have ho' : (progOf x (ops.map fun p => ZKMal.CoinLayer.rd p.2)).owed ≠ none := fun h => answer_owed_none _ h hA'
  obtain ⟨c', tb, dom, xs, his, hph0⟩ := progOf_placeholder x _ hsx hcs ho'
  rw [locate_pred _ 0 i a hj1, hloc] at hph0
  dsimp only at hph0
  have hph : placeholder spec (Masked.stsOf x) (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).1
      (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', tb, dom, xs, his) := by
    rw [hloc]
    exact hph0
  rw [progOf_at x _ hsx (q := i + 1) (by rw [List.length_map]; omega) hj1 hph, hloc] at hA'
  dsimp only at hA'
  obtain ⟨alg, -, hal, -⟩ := repState_ok hA'
  obtain ⟨hts', ⟨tp, htp⟩, hxs⟩ := Count.placeholder_tab hph0
  obtain ⟨hm36, hk0, hnn, hq⟩ := Count.sched_facts htp
  obtain ⟨lay, mask, hds, hc', hdm, -⟩ := placeholder_ok hph0
  rw [hst] at hts' hxs
  have htm : tb.st.m = (Masked.stAt x j).m := by rw [hts']; rfl
  have htk : tb.st.fold.kLog = (Masked.stAt x j).c.kLog := by rw [hts']; rfl
  have hdg : tb.st.digest.size < 2 ^ 64 := by
    rw [hts']
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hmb : tb.st.m < 2 ^ 64 := by rw [htm]; exact lt_of_le_of_lt hm35 (by norm_num)
  have hmkt : tb.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by rw [hts']; exact hmk
  have hcc : ∀ p, p < c'.order.size → c'.order[p]! < c'.h.size := by
    rw [hc']
    exact Claims.ZKServer.ofDraws_order _ _ _
  have hAA : alg.size = A := by
    rw [(Ranked.alg_cnt (by rw [htm]; omega) (by rw [htk]; omega) hmb hal).1, htm, htk, hAd]
  -- past the ring switch the frame is outside every masked phase, the algebraic phase done
  have hA'' := hA'
  rw [show a + 1 = alg.size + (xs.size + 2) + 1 by rw [hxs, hXS]; omega] at hA''
  obtain ⟨hdA, honA, -, hl0A, hiA, -, -⟩ := Bind.repState_opened hcc hal (by rw [htm]; omega) hmb hdg
    (placeholder_cap hph0) (hdm ▸ hdom') hmkt (by rw [hxs]; exact hXb) hk0 hnn hq
    (by rw [Array.size_map, Array.size_range]) hA''
  have hlig : ((ofLoop Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).lig = true := by
    rw [(ofLoop_cfg_place Masked.dataE L₁ real x (ν, t) ((m₀, ρ₀) :: ops) hsx (q := i + 1)
      (by simp only [List.length_cons]; omega) hloc hph0).1.lig, hc']
    obtain ⟨sch1, tp1, hsch1, hlay1⟩ := start_lay hds
    show decide (lay.k0 > 0) = true
    rw [hlay1]
    exact decide_eq_true (Count.sched_facts hsch1).2.1
  -- the loop's own round there
  obtain ⟨tr, hir, sched', hre, hs'⟩ := ofLoop_sched_live Masked.dataE L₁ real wk x w (ν, t) κ (m₀, ρ₀) ops hl hkd
    hsx (q := i + 1) (by omega)
  have hre' : real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hir) := by
    rw [hloc] at hre
    exact hre
  obtain ⟨hT1, hT2, -, -, -, hcapR⟩ := Count.real_tabAlike hRT hph0 hre'
  have hs'' : schedSoFar tr ((2 * j.val + r.val) % 2)
      (spec.domain (spec.tables[(2 * j.val + r.val) / 2]!) ((2 * j.val + r.val) % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
          (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0))) hir
      ((Array.range (a + 1)).map fun q =>
        wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (i + 1 - (a + 1) + q + 1)) = .ok sched' := by
    rw [hloc] at hs'
    exact hs'
  rw [hst] at hs''
  have hcr : ((Array.range (a + 1)).map fun q =>
      wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (i + 1 - (a + 1) + q + 1)) =
      (Array.range (A + k)).map fun q =>
        wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (Masked.repBase x (2 * j.val + r.val) + q) := by
    rw [ha]
    refine Array.map_congr_left fun q _ => ?_
    congr 1
    omega
  rw [hcr] at hs''
  obtain ⟨alg', hal'⟩ := schedSoFar_ok_rounds hs''
  have htrm : tr.st.m = (Masked.stAt x j).m := by rw [hT1, htm]
  have htrk : tr.st.fold.kLog = (Masked.stAt x j).c.kLog := by rw [hT1, htk]
  have hmT : 14 ≤ tr.st.m := by rw [htrm]; omega
  have hmT' : tr.st.m < 2 ^ 64 := by rw [htrm]; exact lt_of_le_of_lt hm35 (by norm_num)
  have hAA' : alg'.size = A := by
    rw [(Ranked.alg_cnt hmT (by rw [htrk]; omega) hmT' hal').1, htrm, htrk, hAd]
  have hsch : tb.sch = (Masked.zkTab x j).sch := (Prod.mk.inj (Except.ok.inj (htp.symm.trans hz))).1
  have htrz : tr.sch = (Masked.zkTab x j).sch := hT2.trans hsch
  have hn0r : tr.sch.n ≠ 0 := by
    rw [hT2]
    have := (Count.sched_rho htp).1
    omega
  have hn64 : tr.sch.n < 2 ^ 64 := by rw [hT2]; exact lt_trans hnn (by norm_num)
  obtain ⟨hssz, zc, lc, hptsR, T, hTe, hbody, -⟩ := Bind.sched_openBind hal' hs'' hdom' (by rw [hT1]; exact hdg) hcapR
    hmT' (by rw [hT1]; exact hmkt) hXb (by rw [hT2]; exact hk0) (by rw [hT2]; exact hnn) (by rw [hT2]; exact hq)
    hn0r (by rw [Array.size_map, Array.size_range, hAA', hXS]; omega) hmT (p := a + 1) (by rw [hAA', hXS]; omega)
  have hlt' : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).2 < sched'.size := by
    rw [hj1]
    exact hssz
  have hcoins : (ofLoop Masked.dataE L₁ real).coins x ((m₀, ρ₀) :: ops).length = tr.sch.n := by
    show loopNr x (((m₀, ρ₀) :: ops).length - 1) = _
    rw [List.length_cons, Nat.add_sub_cancel, hn, Masked.repSlot_eq hsx hz, htrz]
    exact (loopNr_l0 hsx hq2 hz (by rw [hst]; omega) (by rw [hst]; exact hk6) (i := 0) (Nat.zero_le _)).trans
      (if_pos rfl)
  rw [ofLoop_scheduled Masked.dataE L₁ real x w (ν, t) (m₀, ρ₀) ops _ hsx (q := i + 1) (by omega) hre hs' hlt', hj1,
    hbody, hcoins] at hck
  obtain ⟨o, hro, ho⟩ := roundFree_openBind ((ofLoop Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops))
    { m with stopped := false } T (FlockComplete.flat tr.cm0.toC.cap) tr.sch.n honA hdA hlig hl0A hiA hcapR hn0r
    hn64
  have hvb := passes_eq hro (check_none_passes hck)
  have hrel := release_of_roundFree (ofLoop Masked.dataE L₁ real) x (ν, t) ((m₀, ρ₀) :: ops) false v hA hcoins
    (out := o) (fr := #[]) (by rw [hvb]; exact hro)
  refine ⟨tr, hir, zc, lc, T, hre', hT1.trans hts', htrz,
    pts_raw (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2)) (R := Masked.repBase x (2 * j.val + r.val))
      (by rw [hAA', htrm, htrk, hAd]; omega) (by rw [hAA']; omega) hptsR, ?_, by rw [hrel, ho]⟩
  rw [hTe]
  exact rsTarget_raw (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2)) (R := Masked.repBase x (2 * j.val + r.val))
    (A := alg'.size) (C := A + k) _ (by omega) (by omega)

open Loop in
set_option maxHeartbeats 2000000 in
/-- **The opening's binding**: at rep stream `2j + r`'s round after its algebraic phase and PIOP
(`Masked.repSlot` `0`), the release of the worker's round is the switched session's message without its leading domain
frame. The frame releases the loop's binding (`openBind_frame`) and the session sends its own (`Loop.enc_openBind`):
the caps are level 0's of the switched source (`Loop.cap_zcBind`), and the targets agree, the coins as the openings'
words (`Loop.wordsAt_slot`) and the switched vectors at every claim (`Loop.switch_sv`). -/
theorem release_sched_openBind_pf (vs : Loop.finS.SwVals) (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (hreal : real = Loop.realOf src)
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) (hS : Loop.Src L₀ src x w)
    (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (hTV : Loop.TakesValsAt vs (ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops) {v : Msg}
    (hpos : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk (Loop.finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true)
    (href : (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hn : ops.length = Masked.repSlot x (2 * j.val + r.val) fun _ => 0) :
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom (Loop.finS.swMsgAt x (Loop.finS.wt x w) (vs x w ν) t (ops.map fun p => Loop.finS.decOf p.2)) := by
  have hx : Masked.Live x := Masked.live_of hSc hR
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hSc hsx
  have hq2 : 2 * j.val + r.val < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  have hstj : (Masked.stsOf x)[j.val]! = Masked.stAt x j := getElem!_pos (Masked.stsOf x) j.val hj
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst]
    exact (Masked.zkTab x j).hz
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hj).1
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  have hRk : Masked.rankSlot x j.val r.val =
      Masked.repBase x (2 * j.val + r.val) + ((Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6) := by
    show Masked.repBase x (2 * j.val + r.val) + ((Masked.stsOf x)[j.val]!.m + (Masked.stsOf x)[j.val]!.c.kLog - 6) = _
    rw [hstj]
  have hRS : ops.length = Masked.repBase x (2 * j.val + r.val) +
      ((Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6) + (3 + spec.points * (Masked.stAt x j).regions.size) := by
    rw [hn, Masked.repSlot_eq hsx hz, hst, ← Masked.repBase_eq hsx (2 * j.val + r.val)]
    unfold FlockComplete.piopRoundsZk
    try dsimp only
    omega
  obtain ⟨tr, hir, zc, lc, TL, hre, hst', htrz, hpts, hTL, hrel⟩ :=
    openBind_frame (withFiles (withLink L₀ src) src) real wk hx hSv ν t κ m₀ ρ₀ ops
      (by rw [hreal]; exact Loop.realTabs_of src x w (ν, t) hsx) hlt hl hv j r hn
  have hrel' : (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      0 :: (encs (openBind TL (FlockComplete.flat tr.cm0.toC.cap)) ++ (Frame.sq 2 tr.sch.n).enc) := hrel
  have hL0 : Loop.Level0Src src x w := (show Loop.Level0Src src x w ∧ Loop.RepSrc L₀ src x w from hS).1
  have hN0 := unrefused_open Masked.dataE (withFiles (withLink L₀ src) src) real wk x w hx hSv (ν, t) κ m₀ ρ₀ ops
    hlt.le hl href hj hr2 hn
  -- `Masked.swSrcE` at `finS.pad (ops.map …)` itself exhausts `whnf`; at a variable list it elaborates
  generalize hcs : (ops.map fun p => Loop.finS.decOf p.2) = cs
  rw [hcs] at hN0
  have hcl' : cs.length = Masked.ligP0 x j r := by rw [← hcs, List.length_map]; exact hn
  have hlt' : cs.length < refK x := by rw [← hcs, List.length_map]; exact hlt
  obtain ⟨extras, hext⟩ := Masked.extra_ok hx hsx (Loop.finS.pad (K := refK x) cs) hj
  have heS := hext
  rw [Masked.tabsOf_stAt] at heS
  have hes : extras.size = spec.points * (Masked.stAt x j).regions.size := by
    rw [FlockComplete.JsonRT.extraClaims_size heS]
    simp [Nat.mul_comm, Masked.ptsE]
  obtain ⟨hpts2, -, hregs⟩ := Masked.live_bounds hx hsx
  have hreg : (Masked.stAt x j).regions.size ≤ 1024 := hregs j.val hj
  have hNb : spec.points * (Masked.stAt x j).regions.size ≤ 2048 := by rw [hpts2]; omega
  obtain ⟨TR, hTR, henc⟩ := Loop.enc_openBind vs w hx hsx ν t (Loop.finS.pad (K := refK x) cs) j r
    (by have := hN0; rwa [hn] at this) hext
  rw [hreal] at hre
  have hcap := Loop.cap_zcBind hL0 ν t (Loop.finS.pad (K := refK x) cs)
    (vs x w ν (Loop.finS.pad (K := refK x) cs) (Loop.finS.seenOf x t)) j r hre
  have hcap2 : tr.cm0.toC.cap = _ := hcap
  -- the targets agree: the coins are the openings' words, the switched vectors the session's
  have hT : TL = TR := by
    have hsw := Loop.switch_sv vs hR hx hsx hL0 ν t ops hpos hlt.le hcs j r hre hst' hpts (k := 0) (by omega) hN0
      hext
    rw [hTL, hTR, hsw.1]
    have hCM : _root_.Flock.Firewall.Frames.COIN_MAX = 65536 := rfl
    have hW : ∀ p e, p < ops.length → e < min (loopNr x p) _root_.Flock.Firewall.Frames.COIN_MAX →
        (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) p)[e]! =
          Masked.slotAt (Loop.finS.pad (K := refK x) cs) p e := by
      intro p e hp he
      have hw := wordsAt_slot Masked.dataE x ops hlt.le (s := ops.length) (p := p) (e := e) hp le_rfl
        (by rw [wordsAt_size x ops hp le_rfl]; exact he)
      rw [List.take_length] at hw
      exact hw.trans (congrArg (fun c => Masked.slotAt (Loop.finS.pad (K := refK x) c) p e) hcs)
    obtain ⟨-, -, -, h7, h8⟩ := loopNr_piop hsx hq2 hz (by rw [hst]; omega) (by rw [hst]; exact hk6)
    rw [hst, ← Masked.repBase_eq hsx (2 * j.val + r.val)] at h7 h8
    refine FlockComplete.rsTarget_congr (fun e he => ?_) (fun k' hk' e he => ?_) (fun k' hk' =>
      (Loop.switch_sv vs hR hx hsx hL0 ν t ops hpos hlt.le hcs j r hre hst' hpts (k := k') (by omega) hN0
        hext).2 hk')
    · refine hW _ _ (by omega) ?_
      rw [show Masked.rankSlot x j.val r.val + (extras.size + 2) = Masked.repBase x (2 * j.val + r.val) + 3 +
          ((Masked.stAt x j).m - 6) + 3 + ((Masked.stAt x j).c.kLog - 6) +
          (2 + spec.points * (Masked.stAt x j).regions.size) by omega, h8, hCM, Nat.min_eq_left (by omega)]
      omega
    · refine hW _ _ (by omega) ?_
      rw [show Masked.rankSlot x j.val r.val + k' = Masked.repBase x (2 * j.val + r.val) + 3 +
          ((Masked.stAt x j).m - 6) + 3 + ((Masked.stAt x j).c.kLog - 6) + k' by omega, h7 k' (by omega), hCM]
      exact lt_of_lt_of_le he (by decide)
  have hM : ∀ (T : F128) (cap : ByteArray), ∀ f ∈ openBind T cap, noDom f := by
    intro T cap f hf
    simp only [openBind, List.mem_cons, List.not_mem_nil, or_false] at hf
    rcases hf with rfl | rfl | rfl | rfl <;> trivial
  rw [Loop.swMsgAt_enc vs w ν t cs hlt', hcl', henc, dropDom_noDom _ _ _ (hM _ _), hrel', hcap2, htrz, hT]

end LeanRelease

end Claims.ZKServer
