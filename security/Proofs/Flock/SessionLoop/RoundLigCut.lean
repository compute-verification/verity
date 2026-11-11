module

public import Proofs.Flock.SessionLoop.CutInnerLive
public import Proofs.Flock.Claims.ZK.L0PinCoins
public import Proofs.Flock.SessionLoop.FinRounds
public import Proofs.Flock.Claims.ZK.CutPrefix
public import Proofs.Flock.Claims.ZK.FinExe
public import Proofs.Flock.SessionLoop.CutPiopLive
public import Proofs.Flock.Claims.ZK.RepRet
public import Proofs.Flock.Claims.ZK.L0PinCut
public import Proofs.Flock.Claims.ZK.FinishRank

/-!
# The cut session at a rep's Ligerito rounds past level 0, on the opened coins alone

Cut at a round `n` of rep `2j + r`'s Ligerito opening after level 0's close (from `ligP0 + k₀ + 4` to the last before
the inner proof), the rep's PIOP returns at `ligP0` (`piopCut_ret`) and its Ligerito opening, which returns past `n`
when cut later (`ligeritoCut_ret`), throws at `n` with `n + 1` bodies (`ligeritoCut_cut_agree`, `prefix_of_agree'`). The
cut session stopped at `n` sent there what that opening sent (`runCut_cutRep`, carrying the log's size).

* `CutFree.repCut_lig_stop`: the rep from a source, cut at such a round, throws there in its Ligerito opening, after
  its PIOP returned at `ligAt`, wherever its rank check passes and its pair's coins are independent.
* `Masked.repCut_lig_stop`: that stop at a `Live` statement's table `j`'s rep `r`, answered by the opened coins, where
  the coins below the cut show no refusal, with the PIOP's basis `ligBasis` (`piopCut_lig`).
* `Masked.runCut_cutRep_size`, `Masked.runCut_cutRep_size_of_ret`: `runCut_cutRep` and its `_of_ret` form, with the cut
  session's log reaching `n`.
* **`Masked.runCut_lig`**: the cut session stopped at such a round of rep `2j + r` reached it and sent there what the
  rep's Ligerito opening, cut there, sent, from the end of its PIOP at `ligP0` with the basis `ligBasis`.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockComplete.CutFree

open Flock Transcript Replay

set_option maxHeartbeats 2000000 in
/-- **The cut's rep at a round of its Ligerito opening**: answered by `asksCut` cut at a round `n` of the opening
(from its first, `ligAt`, up to the last), at a `--zk` schedule of `m ∈ [25, 35]` and a fold defined at every point
of the lincheck's shape, from any source, level-0 commitment and pads commitment, the rep's PIOP returns at `ligAt` and
its Ligerito opening, cut at `n`, throws there with `n + 1` bodies, as the rep does, wherever its rank check passes and
its pair's coins are independent. -/
theorem repCut_lig_stop (coin : ℕ → ℕ → F128) {resp : ℕ → Claims.Msg} (hr : ∀ k, coinOf (resp k) = coin k)
    {st : Setup} (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) (hkm : st.fold.kLog ≤ st.m)
    (hf : ∀ (a : F128) (lw ρ : Array F128), lw.size = 64 → ρ.size = st.fold.kLog - 6 →
      ∃ v, st.fold.fold a lw ρ = .ok v)
    {sch : Schedule} {tPad : ℕ} (hz : Zk.schedule st.m = .ok (sch, tPad)) (rep : ℕ) (dom : ByteArray) (cm0 : Commit)
    (extras : Array Claim) (ic : InnerCommit) (src : RepSrc) (prior : Array (List F128)) (words : List ℕ)
    (l : Array Claims.Msg) {n : ℕ} (h1 : CutEq.ligAt l.size st extras ≤ n)
    (h2 : n < CutEq.ligAt l.size st extras + (sch.k0 + 7 + 11 * (sch.levels.size - 1)))
    (hrk : ∀ zp zc t₁ l₁ lp lc t₂ l₂,
      ((zerocheckCut (asksCut coin n) st.m st.digest (flat cm0.cap) ic.root (Zk.PadLayout.new st.m st.kLog sch.k0)
          src.masked).run (Transcript.init #[] dom)).go l = (.ok ((zp, zc), t₁), l₁) →
        ((lincheckCut (asksCut coin n) st.fold st.m (Zk.PadLayout.new st.m st.kLog sch.k0) src.masked zc).run t₁).go
          l₁ = (.ok ((lp, lc), t₂), l₂) →
        MaskRank.maskRankOk (prior ++ MaskRank.roundPoints st.m st.kLog t₂.rounds) words = true)
    (hρ : Zk.rhoIndependent
      ((⟨coin (CutEq.ligAt l.size st extras + 2 + sch.k0) 0, coin (CutEq.ligAt l.size st extras + 2 + sch.k0) 1⟩ :
          F256),
        (⟨coin (CutEq.ligAt l.size st extras + 3 + sch.k0) 0, coin (CutEq.ligAt l.size st extras + 3 + sch.k0) 1⟩ :
          F256)) = true) :
    ∃ zp lp ss T sws t₃ l₃ e L,
      ((piopCut (asksCut coin n) st (Zk.PadLayout.new st.m st.kLog sch.k0) src (flat cm0.cap) ic.root extras prior
          words).run (Transcript.init #[] dom)).go l = (.ok ((zp, lp, ss, T, sws), t₃), l₃) ∧
      l₃.size = CutEq.ligAt l.size st extras ∧
      ((ligeritoCut (asksCut coin n) (Hashes.ofSha512 Sha512.hash) st.merkle sch T cm0 (basisVals sws (st.m - 7))
          (Zk.PadLayout.new st.m st.kLog sch.k0) src).run t₃).go l₃ = (.error e, L) ∧
      L.size = n + 1 ∧
      ((repCut (asksCut coin n) (Hashes.ofSha512 Sha512.hash) st sch rep dom cm0 extras ic src prior words).run
          (Transcript.init #[] dom)).go l = (.error e, L) := by
  have hm : 14 ≤ st.m := by omega
  have hLA : CutEq.ligAt l.size st extras = l.size + 3 + (st.m - 6) + (3 + (st.fold.kLog - 6)) +
      (2 + extras.size) + 1 := rfl
  obtain ⟨⟨zp, lp, ss, T, sws⟩, t₃, l₃, hP, hsz₃⟩ := piopCut_ret coin n hm hkm hf
    (Zk.PadLayout.new st.m st.kLog sch.k0) src (flat cm0.cap) ic.root extras prior words (Transcript.init #[] dom) l
    (by omega) hrk
  obtain ⟨lg, zl, rec, t₄, l₄, h₄, hsz₄, -⟩ := ligeritoCut_ret coin
    (l₃.size + (sch.k0 + 7 + 11 * (sch.levels.size - 1))) hr h25 h35 hz st.merkle T cm0
    (basisVals sws (st.m - 7)) (Zk.PadLayout.new st.m st.kLog sch.k0) src t₃ l₃ le_rfl
    (by rw [hsz₃]; exact hρ)
  obtain ⟨e, L, hL, hLs, -⟩ := CutPrefix.prefix_of_agree'
    (CutPrefix.ligeritoCut_cut_agree coin (show n ≤ l₃.size + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) by omega)
      st.merkle sch T cm0 (basisVals sws (st.m - 7)) (Zk.PadLayout.new st.m st.kLog sch.k0) src t₃)
    (by omega) h₄ (by omega)
  refine ⟨zp, lp, ss, T, sws, t₃, l₃, e, L, hP, by omega, hL, hLs, ?_⟩
  unfold repCut
  try dsimp only
  rw [bind_go hP]
  try dsimp only
  exact FlockComplete.bind_go_err hL

end FlockComplete.CutFree

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

set_option maxHeartbeats 1000000 in
/-- **A rep stopped at a round of its Ligerito opening past level 0's close, on the opened coins alone**: at a `Live`
statement whose coins below the cut `n` (from `ligP0 + k₀ + 4` up to the opening's last round) show no refusal, the cut
rep answered by the opened coins from its first send `repBase x (2j + r)`, from any source, level-0 commitment and pads
commitment, returns from its PIOP at `ligP0` with the basis `ligBasis` (at any witness and tape) and throws at `n` in
its Ligerito opening, with `n + 1` bodies. Its rank check passes (`rank_at_unrefused`) and its pair's coins are
independent, both below the cut. -/
theorem repCut_lig_stop {x : Msg} (hx : Live x) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) {s0} {spec : _root_.Flock.SessionSpec}
    (hsx : FlockDeploy.setupOf true x = some (s0, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2) {n : ℕ}
    (h1 : ligP0 x j r + (zkTab x j).sch.k0 + 4 ≤ n)
    (h2 : n < ligP0 x j r + ((zkTab x j).sch.k0 + 7 + 11 * ((zkTab x j).sch.levels.size - 1)))
    (hn : ¬ Refused x c n) (dom : ByteArray) (cm0 : FlockComplete.Commit) {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    (ic : FlockComplete.InnerCommit) (src : FlockComplete.RepSrc) {l : Array Msg}
    (hl : l.size = repBase x (2 * j.val + r.val)) :
    ∃ T sws t₃ l₃ e L, l₃.size = ligP0 x j r ∧
      FlockComplete.basisVals sws ((FlockDeploy.tabsOf (stsOf x))[j.val]!.m - 7) = ligBasis x c w τ j r extras ∧
      ((FlockComplete.ligeritoCut (FlockComplete.asksCut (ligCoin c) n) hsE
          (FlockDeploy.tabsOf (stsOf x))[j.val]!.merkle (zkTab x j).sch T cm0
          (FlockComplete.basisVals sws ((FlockDeploy.tabsOf (stsOf x))[j.val]!.m - 7))
          (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
            (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0) src).run t₃).go l₃ = (.error e, L) ∧
      L.size = n + 1 ∧
      ((FlockComplete.repCut (FlockComplete.asksCut (ligCoin c) n) hsE
        (FlockDeploy.tabsOf (stsOf x))[j.val]! (zkTab x j).sch r.val dom cm0 extras ic src
        (priorE c j.val r.val) (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!).run
        (_root_.Flock.Transcript.init #[] dom)).go l = (.error e, L) := by
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  have hr := r.isLt
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hst : (stsOf x)[j.val]! = stAt x j := getElem!_pos _ _ _
  have hz₀ : _root_.Flock.Zk.schedule (stsOf x)[j.val]!.m = .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [hst]
    exact (zkTab x j).hz
  have hz₁ : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [show (2 * j.val + r.val) / 2 = j.val by omega]
    exact hz₀
  have htab : (FlockDeploy.tabsOf (stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (stsOf x)[j.val]! := by
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) j.val (by simpa [FlockDeploy.tabsOf] using hj),
      getElem!_pos (stsOf x) j.val hj]
    simp [FlockDeploy.tabsOf]
  have hext₀ := hext
  rw [htab] at hext ⊢
  have h25 : 25 ≤ (stsOf x)[j.val]!.m := by
    rw [getElem!_pos (stsOf x) j.val hj]
    exact (hall _ hj).1
  have h35 : (stsOf x)[j.val]!.m ≤ 35 := by
    rw [getElem!_pos (stsOf x) j.val hj]
    exact (hall _ hj).2.1
  have hT : FlockComplete.ZkTables.StmtT (stsOf x)[j.val]! := by
    rw [getElem!_pos (stsOf x) j.val hj]
    exact _root_.Flock.Readable.stmtT_of_scope hx.1 hsx hj
  obtain ⟨⟨hwf, -⟩, hslot⟩ := hT
  have hk6 := hwf.hk6
  have hkm := hwf.hk
  have hes : extras.size = spec.points * (stsOf x)[j.val]!.regions.size := by
    rw [FlockComplete.JsonRT.extraClaims_size hext]
    simp [Nat.mul_comm, ptsE]
  have hbase : l.size = FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) := by rw [hl, repBase_eq hsx]
  have hLig : FlockComplete.CutEq.ligAt l.size (_root_.Flock.Setup.ofCircuit (stsOf x)[j.val]!) extras =
      l.size + 3 + ((stsOf x)[j.val]!.m - 6) + (3 + ((stsOf x)[j.val]!.c.kLog - 6)) + (2 + extras.size) + 1 := rfl
  have eP : ligP0 x j r = FlockComplete.CutEq.ligAt l.size (_root_.Flock.Setup.ofCircuit (stsOf x)[j.val]!) extras := by
    show repSlot x (2 * j.val + r.val) (fun _ => 0) = _
    rw [repSlot_eq hsx hz₁, ← hbase, show (2 * j.val + r.val) / 2 = j.val by omega, hLig]
    unfold FlockComplete.piopRoundsZk
    omega
  have eρ : rhoSlot x j.val r.val =
      FlockComplete.CutEq.ligAt l.size (_root_.Flock.Setup.ofCircuit (stsOf x)[j.val]!) extras + 2 +
        (zkTab x j).sch.k0 := by
    unfold rhoSlot
    rw [repSlot_eq hsx hz₁, ← hbase, show (2 * j.val + r.val) / 2 = j.val by omega, hLig]
    unfold FlockComplete.piopRoundsZk
    omega
  have hrank : rankSlot x j.val r.val = l.size + ((stsOf x)[j.val]!.m + (stsOf x)[j.val]!.c.kLog - 6) := by
    unfold rankSlot
    rw [← hl]
  have hrank0 : rankSlot x j.val 0 ≤ rankSlot x j.val r.val := by
    unfold rankSlot
    rw [repBase_eq hsx, repBase_eq hsx]
    have := FlockComplete.ZkTables.repStart_mono spec (stsOf x) (show 2 * j.val + 0 ≤ 2 * j.val + r.val by omega)
    omega
  have hρ' : _root_.Flock.Zk.rhoIndependent (rhoAt c j.val r.val) = true := by
    cases hb : _root_.Flock.Zk.rhoIndependent (rhoAt c j.val r.val)
    · exact (hn (.inr ⟨j.val, hj, r.val, hr, .inr (.inl ⟨by omega, hb⟩)⟩)).elim
    · rfl
  obtain ⟨zp, lp, ss, T, sws, t₃, l₃, e, L, hP, -, hL, hLs, hR⟩ :=
    FlockComplete.CutFree.repCut_lig_stop (ligCoin c) (coinOf_respE c)
      (st := _root_.Flock.Setup.ofCircuit (stsOf x)[j.val]!) h25 h35 hkm
      (fun a lw ρ hlw hρ => FlockComplete.JsonRT.fold_ok _ hk6 hslot a hlw hρ)
      hz₀ r.val dom cm0 extras ic src (priorE c j.val r.val) (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]! l
      (n := n) (by omega) (by omega)
      (fun _ _ _ _ _ _ _ _ h₁ h₂ => by
        rw [FlockComplete.MaskRank.masksOf_getElem! hj]
        exact rank_at_unrefused hx hn hj hr (by omega) (by omega) (ligCoin c) _ (fun _ _ _ => rfl)
          (_root_.Flock.Setup.ofCircuit (stsOf x)[j.val]!) rfl rfl rfl _ _ _ _
          (t₀ := _root_.Flock.Transcript.init #[] dom) rfl hl h₁ h₂)
      (by
        rw [show FlockComplete.CutEq.ligAt l.size (_root_.Flock.Setup.ofCircuit (stsOf x)[j.val]!) extras + 3 +
            (zkTab x j).sch.k0 = rhoSlot x j.val r.val + 1 by omega, ← eρ]
        exact hρ')
  obtain ⟨hl₃, -, hb⟩ := piopCut_lig hx w τ c n hsx j r hext₀ (_root_.Flock.Setup.ofCircuit (stsOf x)[j.val]!)
    (by rw [hst]) src _ _ _ _ hl hP
  exact ⟨T, sws, t₃, l₃, e, L, hl₃, hb, hL, hLs, hR⟩

set_option maxHeartbeats 1000000 in
/-- **The cut session at rep `2j + r`'s stop, reaching it**: `runCut_cutRep`, where the rep's log `L` reaches `n`, with
the cut session's log reaching `n` too. -/
theorem runCut_cutRep_size {x : Msg} (hx : Live x) (c : Fin (refK x) → FlockDeploy.Coin)
    (src : ℕ → FlockComplete.TabSrc) {s0 : _}
    {spec : _root_.Flock.SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (stsOf x)) : _root_.Flock.V _) = .ok schs)
    {cms : Array FlockComplete.Commit}
    (hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
          _root_.Flock.Hashes FlockComplete.PS)
        (FlockDeploy.tabsOf (stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default) (src i)).go l = (.ok cms, l))
    (j : Fin (jOf x)) (r : Fin 2)
    (hexts : ∀ k < 2 * j.val + r.val, ∃ extras,
      (FlockDeploy.tabsOf (stsOf x))[k / 2]!.extra (ptsE c spec) = .ok extras)
    (hprev : ∀ k < 2 * j.val + r.val, ∀ extras,
      (FlockDeploy.tabsOf (stsOf x))[k / 2]!.extra (ptsE c spec) = .ok extras → ∀ ic,
      (∀ l, (FlockComplete.padsTreeCut hsE
        (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[k / 2]!.m
          (FlockDeploy.tabsOf (stsOf x))[k / 2]!.kLog schs[k / 2]!.1.k0).len +
          _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
        ((src (k / 2)).rep (k % 2))).go l = (.ok ic, l)) →
      ∀ l : Array Msg, l.size = repBase x k →
      ∃ p tr l', ((FlockComplete.repCut (FlockComplete.asksCut (ligCoin c) (refK x)) hsE
          (FlockDeploy.tabsOf (stsOf x))[k / 2]! schs[k / 2]!.1 (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 cms[k / 2]! extras ic ((src (k / 2)).rep (k % 2))
          (priorE c (k / 2) (k % 2)) (FlockComplete.MaskRank.masksOf (stsOf x))[k / 2]!).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l =
          (.ok (p, tr), l') ∧ l'.size = repBase x (k + 1) ∧
        (k % 2 = 0 → FlockComplete.MaskRank.roundPoints (FlockDeploy.tabsOf (stsOf x))[k / 2]!.m
          (FlockDeploy.tabsOf (stsOf x))[k / 2]!.kLog tr.rounds = ptsOpen c (stsOf x)[k / 2]! (2 * (k / 2))))
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    {ic : FlockComplete.InnerCommit}
    (hic : ∀ l, (FlockComplete.padsTreeCut hsE
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((src j.val).rep r.val)).go l = (.ok ic, l))
    {l : Array Msg} (hl : l.size = repBase x (2 * j.val + r.val))
    {n : ℕ} {e : String} {L : Array Msg}
    (hR : ((FlockComplete.repCut (FlockComplete.asksCut (ligCoin c) n) hsE
        (FlockDeploy.tabsOf (stsOf x))[j.val]! (zkTab x j).sch r.val (spec.domain spec.tables[j.val]! r.val).toUTF8
        cms[j.val]! extras ic ((src j.val).rep r.val) (priorE c j.val r.val)
        (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!).run
        (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8)).go l = (.error e, L))
    (hLn : n < L.size) (hn0 : repBase x (2 * j.val + r.val) ≤ n) (hnN : n ≤ refK x) :
    n < (runCut x c n spec src).2.size ∧ (runCut x c n spec src).2[n]! = L[n]! := by
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hJ : jOf x = (stsOf x).size := by
    unfold jOf
    exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  have hk : 2 * j.val + r.val < 2 * spec.tables.size := by
    have := r.isLt
    omega
  obtain ⟨schs', hs', hschs⟩ := schedsPZk_live hx
  have es : schs' = schs := Except.ok.inj (hs'.symm.trans hs)
  have esch : schs[j.val]! = ((zkTab x j).sch, (zkTab x j).tPad) := es ▸ hschs j
  have h1 : 1 ≤ repBase x (2 * j.val + r.val) := by
    rw [repBase_eq hsx]
    unfold FlockComplete.repStartZk
    omega
  have hmono : ∀ a b, a ≤ b → repBase x a ≤ repBase x b := fun a b hab => by
    rw [repBase_eq hsx, repBase_eq hsx]
    exact FlockComplete.ZkTables.repStart_mono _ _ hab
  have hpri : ∀ (k : ℕ) (streams : Array (Array _root_.Flock.Round)),
      (k % 2 = 1 → FlockComplete.MaskRank.roundPoints (FlockDeploy.tabsOf (stsOf x))[k / 2]!.m
        (FlockDeploy.tabsOf (stsOf x))[k / 2]!.kLog streams.back! = ptsOpen c (stsOf x)[k / 2]! (2 * (k / 2))) →
      FlockComplete.priorOf (FlockDeploy.tabsOf (stsOf x))[k / 2]! streams (k % 2) = priorE c (k / 2) (k % 2) := by
    intro k b hb
    unfold FlockComplete.priorOf priorE
    by_cases h1 : k % 2 = 1
    · rw [ite_eq_left_of_eq_true _ _ (eq_true h1), ite_eq_left_of_eq_true _ _ (eq_true h1)]
      exact hb h1
    · rw [ite_eq_right_of_eq_false _ _ (eq_false h1), ite_eq_right_of_eq_false _ _ (eq_false h1)]
  unfold runCut
  rw [FlockComplete.finSessionCut_eq, FlockComplete.go_bind, FlockComplete.schedsPZk_quiet, hs]
  dsimp only
  rw [FlockComplete.go_bind, hcms]
  dsimp only
  rw [FlockComplete.go_bind, FlockComplete.finAsksCut_link_go,
    ite_eq_right_of_eq_false _ _ (eq_false (by simp only [Array.size_empty]; omega))]
  try dsimp only
  unfold FlockComplete.afterLinkCut
  try simp only [Array.size_empty]
  try dsimp only
  rw [← Array.forIn_toList, FlockDeploy.repStreams_toList]
  refine Claims.ZKServer.Loop.forIn_bind_err (fun L' => n < L'.size ∧ L'[n]! = L[n]!)
    (fun k b l => l.size = repBase x k ∧ (k % 2 = 1 →
      FlockComplete.MaskRank.roundPoints (FlockDeploy.tabsOf (stsOf x))[k / 2]!.m
        (FlockDeploy.tabsOf (stsOf x))[k / 2]!.kLog b.1.back! = ptsOpen c (stsOf x)[k / 2]! (2 * (k / 2))))
    (2 * j.val + r.val) (by simpa using hk) ?_ ⟨?_, fun h => absurd h (by norm_num)⟩ ?_
  · -- the reps before `2j + r` return by `n`, as at the ZK server's stop
    rintro k hk' ⟨streams, proofs⟩ ls ⟨hls, hpts⟩
    obtain ⟨extras', hxe'⟩ := hexts k hk'
    obtain ⟨ic', hic'⟩ := padsTreeCut_go
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[k / 2]!.m
        (FlockDeploy.tabsOf (stsOf x))[k / 2]!.kLog schs[k / 2]!.1.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((src (k / 2)).rep (k % 2))
    obtain ⟨p', tr', l'', hrep, hsz, hrp⟩ := hprev k hk' extras' hxe' ic' hic' ls hls
    have hle : repBase x (k + 1) ≤ repBase x (2 * j.val + r.val) := hmono _ _ (by omega)
    have hle' : repBase x k ≤ repBase x (k + 1) := hmono _ _ (by omega)
    have hcutn := FlockComplete.CutPrefix.eq_of_agree
      (FlockComplete.CutPrefix.repCut_cut_agree _ hnN _ _ _ _ _ _ _ _ _ _ _) (by omega) hrep (by omega)
    have hp := hpri k streams hpts
    unfold ptsE at hxe'
    refine ⟨?_, l'', ?_, hsz, ?_⟩
    rotate_left
    · simp only [List.getElem_map, List.getElem_range]
      try dsimp only
      rw [FlockComplete.go_bind, FlockComplete.go_liftV, hxe']
      try dsimp only
      rw [FlockComplete.go_bind, hic']
      try dsimp only
      rw [hp, FlockComplete.go_bind]
      try simp only [StateT.run] at hcutn ⊢
      rw [hcutn]
      try dsimp only
      rfl
    · intro h1
      have hk2 : (k + 1) / 2 = k / 2 := by omega
      rw [hk2]
      show FlockComplete.MaskRank.roundPoints _ _ (streams.push tr'.rounds).back! = _
      rw [Array.back!_push]
      exact hrp (by omega)
  · rw [repBase_eq hsx]
    simp [FlockComplete.repStartZk]
  · -- rep `2j + r` stops at `n`
    rintro ⟨streams, proofs⟩ l₀ ⟨hl₀, hpts⟩
    have hp := hpri (2 * j.val + r.val) streams hpts
    have e2 : (2 * j.val + r.val) / 2 = j.val := by omega
    have e3 : (2 * j.val + r.val) % 2 = r.val := by omega
    simp only [e2, e3] at hp
    obtain ⟨bs, hbs, hR0⟩ := FlockComplete.CutPrefix.go_of_shift
      (FlockComplete.CutPrefix.repCut_shift _ _ _ _ _ _ _ _ _ _ _ _ _) (hl.trans hl₀.symm) hR
    refine ⟨e, l₀ ++ bs, ?_, ?_⟩
    · simp only [List.getElem_map, List.getElem_range]
      simp only [e2, e3]
      try dsimp only
      rw [esch]
      try dsimp only
      unfold ptsE at hext
      rw [FlockComplete.go_bind, FlockComplete.go_liftV, hext]
      try dsimp only
      rw [FlockComplete.go_bind, hic]
      try dsimp only
      rw [hp, FlockComplete.go_bind]
      try simp only [StateT.run] at hR0 ⊢
      rw [hR0]
    · have hsz : (l₀ ++ bs).size = L.size := by
        rw [hbs, Array.size_append, Array.size_append, hl, hl₀]
      refine ⟨by omega, ?_⟩
      rw [hbs]
      exact Claims.ZKServer.Loop.getElem!_append_shift (hl₀.trans hl.symm) (by omega)

/-- **The cut session at rep `2j + r`'s stop, reaching it, from the earlier reps' returns alone**:
`runCut_cutRep_size` with rep `0`'s claim points at the opened coins read off its return (`repCut_points`). -/
theorem runCut_cutRep_size_of_ret {x : Msg} (hx : Live x) (c : Fin (refK x) → FlockDeploy.Coin)
    (src : ℕ → FlockComplete.TabSrc) {s0 : _}
    {spec : _root_.Flock.SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (stsOf x)) : _root_.Flock.V _) = .ok schs)
    {cms : Array FlockComplete.Commit}
    (hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
          _root_.Flock.Hashes FlockComplete.PS)
        (FlockDeploy.tabsOf (stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default) (src i)).go l = (.ok cms, l))
    (j : Fin (jOf x)) (r : Fin 2)
    (hexts : ∀ k < 2 * j.val + r.val, ∃ extras,
      (FlockDeploy.tabsOf (stsOf x))[k / 2]!.extra (ptsE c spec) = .ok extras)
    (hprev : ∀ k < 2 * j.val + r.val, ∀ extras,
      (FlockDeploy.tabsOf (stsOf x))[k / 2]!.extra (ptsE c spec) = .ok extras → ∀ ic,
      (∀ l, (FlockComplete.padsTreeCut hsE
        (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[k / 2]!.m
          (FlockDeploy.tabsOf (stsOf x))[k / 2]!.kLog schs[k / 2]!.1.k0).len +
          _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
        ((src (k / 2)).rep (k % 2))).go l = (.ok ic, l)) →
      ∀ l : Array Msg, l.size = repBase x k →
      ∃ p tr l', ((FlockComplete.repCut (FlockComplete.asksCut (ligCoin c) (refK x)) hsE
          (FlockDeploy.tabsOf (stsOf x))[k / 2]! schs[k / 2]!.1 (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 cms[k / 2]! extras ic ((src (k / 2)).rep (k % 2))
          (priorE c (k / 2) (k % 2)) (FlockComplete.MaskRank.masksOf (stsOf x))[k / 2]!).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l =
          (.ok (p, tr), l') ∧ l'.size = repBase x (k + 1))
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    {ic : FlockComplete.InnerCommit}
    (hic : ∀ l, (FlockComplete.padsTreeCut hsE
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((src j.val).rep r.val)).go l = (.ok ic, l))
    {l : Array Msg} (hl : l.size = repBase x (2 * j.val + r.val))
    {n : ℕ} {e : String} {L : Array Msg}
    (hR : ((FlockComplete.repCut (FlockComplete.asksCut (ligCoin c) n) hsE
        (FlockDeploy.tabsOf (stsOf x))[j.val]! (zkTab x j).sch r.val (spec.domain spec.tables[j.val]! r.val).toUTF8
        cms[j.val]! extras ic ((src j.val).rep r.val) (priorE c j.val r.val)
        (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!).run
        (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8)).go l = (.error e, L))
    (hLn : n < L.size) (hn0 : repBase x (2 * j.val + r.val) ≤ n) (hnN : n ≤ refK x) :
    n < (runCut x c n spec src).2.size ∧ (runCut x c n spec src).2[n]! = L[n]! := by
  refine runCut_cutRep_size hx c src hsx hs hcms j r hexts (fun k hk extras' hxe ic' hic' l' hl' => ?_) hext hic hl
    hR hLn hn0 hnN
  obtain ⟨p, tr, l'', hrep, hsz⟩ := hprev k hk extras' hxe ic' hic' l' hl'
  refine ⟨p, tr, l'', hrep, hsz, fun h0 => ?_⟩
  have := r.isLt
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  exact repCut_points hx c (refK x) (i := k / 2) (by omega) _ _ _ _ _ _ _ _ _
    (by rw [show 2 * (k / 2) = k by omega]; exact hl') hrep

set_option maxHeartbeats 1000000 in
/-- **The cut session at a round of rep `2j + r`'s Ligerito opening past level 0's close, on the opened coins alone**:
where the coins below the cut `n` (from `ligP0 + k₀ + 4` up to the opening's last round) show no refusal, the cut
session stopped at `n` reached it and sent there what the rep's Ligerito opening, cut at `n`, sent from the end of its
PIOP at `ligP0`, with the PIOP's basis `ligBasis` (at any witness and tape), level 0's commitment and the rep's source:
the earlier reps return (`repCut_ret`, `runCut_cutRep_size_of_ret`) and the rep stops there (`repCut_lig_stop`). -/
theorem runCut_lig {x : Msg} (hx : Live x) (w : Claims.Flock.Wit) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin)
    (src : ℕ → FlockComplete.TabSrc) {s0 : _}
    {spec : _root_.Flock.SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (stsOf x)) : _root_.Flock.V _) = .ok schs)
    {cms : Array FlockComplete.Commit}
    (hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
          _root_.Flock.Hashes FlockComplete.PS)
        (FlockDeploy.tabsOf (stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default) (src i)).go l = (.ok cms, l))
    (j : Fin (jOf x)) (r : Fin 2) {n : ℕ}
    (h1 : ligP0 x j r + (zkTab x j).sch.k0 + 4 ≤ n)
    (h2 : n < ligP0 x j r + ((zkTab x j).sch.k0 + 7 + 11 * ((zkTab x j).sch.levels.size - 1)))
    (hN : ¬ Refused x c n) {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras) :
    ∃ T sws t₃ l₃ e L, l₃.size = ligP0 x j r ∧
      FlockComplete.basisVals sws ((FlockDeploy.tabsOf (stsOf x))[j.val]!.m - 7) = ligBasis x c w τ j r extras ∧
      ((FlockComplete.ligeritoCut (FlockComplete.asksCut (ligCoin c) n) hsE
          (FlockDeploy.tabsOf (stsOf x))[j.val]!.merkle (zkTab x j).sch T cms[j.val]!
          (FlockComplete.basisVals sws ((FlockDeploy.tabsOf (stsOf x))[j.val]!.m - 7))
          (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
            (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0) ((src j.val).rep r.val)).run t₃).go l₃ =
        (.error e, L) ∧
      n < (runCut x c n spec src).2.size ∧ (runCut x c n spec src).2[n]! = L[n]! := by
  have hjs : j.val < (stsOf x).size := lt_of_jOf j
  have hr2 := r.isLt
  have hz₁ : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [show (2 * j.val + r.val) / 2 = j.val by omega, getElem!_pos (stsOf x) j.val hjs]
    exact (zkTab x j).hz
  have hβ := betaSlot_lt_repBase hx hsx (q := 2 * j.val + r.val) (by omega)
  rw [show (2 * j.val + r.val) / 2 = j.val by omega, show (2 * j.val + r.val) % 2 = r.val by omega] at hβ
  have hP0 : ligP0 x j r = repBase x (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (stsOf x)[(2 * j.val + r.val) / 2]! := by
    rw [repBase_eq hsx]
    exact (repSlot_eq hsx (q := 2 * j.val + r.val) hz₁ (fun _ => 0)).trans (Nat.add_zero _)
  have hβs : betaSlot x j.val r.val = repBase x (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (stsOf x)[(2 * j.val + r.val) / 2]! +
        ((zkTab x j).sch.k0 + 7 + 11 * ((zkTab x j).sch.levels.size - 1) + 2) := by
    unfold betaSlot
    rw [repBase_eq hsx]
    exact repSlot_eq hsx hz₁ _
  have hK : repBase x (2 * j.val + r.val + 1) ≤ refK x := by
    rw [repBase_eq hsx, refK_eq hsx, ← FlockComplete.ZkTables.repStart_last]
    exact FlockComplete.ZkTables.repStart_mono _ _ (by omega)
  have hn0 : repBase x (2 * j.val + r.val) ≤ n := by omega
  have hnN : n ≤ refK x := by omega
  obtain ⟨ic, hic⟩ := padsTreeCut_go
    (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0).len +
      _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
    ((src j.val).rep r.val)
  obtain ⟨l, hl⟩ : ∃ l : Array Msg, l.size = repBase x (2 * j.val + r.val) :=
    ⟨Array.replicate _ [], Array.size_replicate⟩
  obtain ⟨T, sws, t₃, l₃, e, L, hl₃, hb, hL, hLs, hR⟩ := repCut_lig_stop hx w τ c hsx j r h1 h2 hN
    (spec.domain spec.tables[j.val]! r.val).toUTF8 cms[j.val]! hext ic ((src j.val).rep r.val) hl
  refine ⟨T, sws, t₃, l₃, e, L, hl₃, hb, hL, runCut_cutRep_size_of_ret hx c src hsx hs hcms j r
    (fun k hk => extra_ok hx hsx c (by omega)) (fun k hk extras' hxe ic' _ l' hl' => ?_) hext hic hl hR
    (by omega) hn0 hnN⟩
  have hkb : betaSlot x (k / 2) (k % 2) < n := by
    have h1 := betaSlot_lt_repBase hx hsx (q := k) (by omega)
    have h2 : repBase x (k + 1) ≤ repBase x (2 * j.val + r.val) := by
      rw [repBase_eq hsx, repBase_eq hsx]
      exact FlockComplete.ZkTables.repStart_mono _ _ (by omega)
    omega
  exact repCut_ret hx c hN hsx hs (by omega) hkb _ _ hxe ic' _ hl'

end Masked

end Claims.ZKMal
