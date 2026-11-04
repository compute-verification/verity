import Proofs.Flock.Claims.ZK.FinishSwPts
import Proofs.Flock.Claims.ZK.FinishSwBit
import Proofs.Flock.Claims.ZK.FinishSwLigE
import Proofs.Flock.Claims.ZK.FinishRank
import Proofs.Flock.Claims.ZK.FinishSrc

/-!
# The firewall's rep is the switched source's cut

`FinishSrc.repCut_fw` takes a rep source holding four things at the firewall's runs: the masked values (`hmk`), the
rank check and switched vectors at its zerocheck and lincheck (`hpiop`), level 0's opening at its PIOP (`hlig`), and the
inner proof's at its Ligerito and constraints (`hinner`). Here the switched prover's source (`swSrcE`) at the firewall's
clear values (`valsFw`) is shown to hold all four at table `j`'s rep `r`, at the firewall's request `q`, the opened
coins (`ligCoin`) and the packed table of the witness's trace (`ligTab`). So the firewall's rep returns, up to
`eraseZ`, what the cut returns from the switched source.

* `repFwC_piopFw`: where the firewall's rep returned, so did its PIOP.
* `zcCoins_ligRespE`, `lcCoins_ligRespE`, `rsCoins_ligRespE`: the closed forms' coins at the replies `ligRespE c` are
  the opened coins.
* **`swSrcE_repCut`**: `repCut_fw` at the switched source. `hmk` is `swSrcE_hmk_frame`. `hpiop`'s rank half is
  `rank_at_admits`, and its switched vectors are the prover's (`swSrc_sv`), whose claims have the firewall's points
  (`zerocheckCut_at`, `lincheckCut_at`, `claims_x`) and `m − 6` coordinates (`claims_xsize'`), read off the packed
  table as off the trace (`switchVec_packed`). `hlig` is `swSrcE_ligHyp_E`, the firewall's PIOP having sent
  `piopRoundsZk` bodies from the rep's base and switched `swsE` (`piopFw_at`). `hinner` is `swSrcE_hinner_sv`.

Premises: the firewall's packed table is the witness's (`hg`), a trust in the host's inputs as `LinkIs` is for the link;
the opened coins are admitted; and level 0's clear values are `l0Clear` at the coins (`hB`, as `swSrcE_ligHyp_E` takes
it). That the rep's base is before the stop comes from the firewall's PIOP having returned (`repFwC_piopFw`, `piopFw_at`'s
bound), and that no refusal shows by round `n` from the admitted coins (`not_refused_of_admits`).
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Flock.Firewall.Finish
open Flock Transcript

/-- **A returned rep ran its PIOP**: where `repFwC` returned from `l`, so did `piopFw` at its arguments from `l`. -/
theorem repFwC_piopFw {coin : ℕ → ℕ → F128} {n : ℕ} {st : Setup} {sch : Flock.Schedule} {tPad rep : ℕ}
    {dom : ByteArray} {g : Array F128} {cm0 : FlockComplete.Commit} {extras : Array Claim}
    {pads : FlockComplete.PadsZk} {ic : FlockComplete.InnerCommit} {vals : Array F128} {extra lanePad : ℕ → ℕ → F128}
    {salt : ℕ → ℕ → ByteArray} {t₀ : Transcript} {l : Array Claims.Msg} {p : Zk.ZkProof} {t : Transcript}
    {l' : Array Claims.Msg}
    (h : ((repFwC (FlockComplete.asksCut coin n) (Hashes.ofSha512 Sha512.hash) st sch tPad rep dom g cm0 extras pads ic
      vals extra lanePad salt).run t₀).go l = (.ok (p, t), l')) :
    ∃ zp lp ss T sws t₃ l₃, ((piopFw (FlockComplete.asksCut coin n) st (Zk.PadLayout.new st.m st.kLog sch.k0) g
      (fun i => vals[i]! + pads.h[i]!) (FlockComplete.flat cm0.cap) ic.root extras).run t₀).go l =
        (.ok ((zp, lp, ss, T, sws), t₃), l₃) := by
  dsimp only [repFwC] at h
  obtain ⟨⟨zp, zc⟩, t₁, l₁, h₁, h⟩ := run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lp, lc⟩, t₂, l₂, h₂, h⟩ := run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨ss, T, sws⟩, t₃, l₃, h₃, -⟩ := run_bind_ok h
  refine ⟨zp, lp, ss, T, sws, t₃, l₃, ?_⟩
  unfold piopFw
  rw [FlockComplete.Replay.bind_go h₁]
  try dsimp only
  rw [FlockComplete.Replay.bind_go h₂]
  try dsimp only
  rw [FlockComplete.Replay.bind_go h₃]
  rfl

end Flock.Firewall.Finish

namespace FlockComplete.LigLock

/-- **`LigHyp` reads levels ≥ 1's salts only on the schedule**: it holds at salts that agree with its own at levels `1`
to `R` and the positions of their trees. -/
theorem LigHyp.ofSalt {coin : ℕ → ℕ → _root_.Flock.F128} {n p0 : ℕ} {ms : _root_.Flock.MerkleScheme}
    {sch : _root_.Flock.Schedule} {t b : Array _root_.Flock.F128} {lay : _root_.Flock.Zk.PadLayout}
    {h : Array _root_.Flock.F128} {rep tPad : ℕ} {extra lanePad : ℕ → ℕ → _root_.Flock.F128}
    {salt salt' : ℕ → ℕ → ByteArray} {src : FlockComplete.RepSrc}
    (H : LigHyp coin n p0 ms sch t b lay h rep tPad extra lanePad salt src)
    (hs : ∀ i, 1 ≤ i → i < sch.levels.size → ∀ q < 2 ^ (sch.levels.getD i default).d, salt' i q = salt i q) :
    LigHyp coin n p0 ms sch t b lay h rep tPad extra lanePad salt' src :=
  { H with salt := fun a b' i h1 h2 q hq => by rw [hs i h1 h2 q hq]; exact H.salt a b' i h1 h2 q hq }

end FlockComplete.LigLock

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

theorem zcCoins_ligRespE {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (L : ℕ) :
    FlockComplete.Replay.Zc.coinsAt (ligRespE c) L = FlockComplete.zcCoins (ligCoin c) L := by
  unfold FlockComplete.Replay.Zc.coinsAt FlockComplete.zcCoins
  simp only [coinOf_ligRespE]

theorem lcCoins_ligRespE {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (L n : ℕ) :
    FlockComplete.Replay.Lc.coinsAt (ligRespE c) L n = FlockComplete.lcCoins (ligCoin c) L n := by
  unfold FlockComplete.Replay.Lc.coinsAt FlockComplete.lcCoins
  simp only [coinOf_ligRespE]

theorem rsCoins_ligRespE {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (L n : ℕ) :
    FlockComplete.Replay.Rs.coinsAt (ligRespE c) L n = ⟨fun k => ligCoin c (L + k), ligCoin c (L + n)⟩ := by
  unfold FlockComplete.Replay.Rs.coinsAt
  simp only [coinOf_ligRespE]

/-- **The firewall's rep is the switched source's cut**: at a live statement, a witness, a tape, admitted opened coins,
any stop `n`, the session's setup, table `j` and its extra claims, and at
`pads_frame`'s premises at the firewall's request `q`, where table `j`'s rows hold at its base, level 0's part of the
firewall's clear values is `l0Clear` at the coins, the firewall's packed table is the witness's (`hg`), and its levels ≥ 1
salts are the tape's on the schedule (`hsalt`, all `LigHyp` reads of them, `LigHyp.ofSalt`): where the
firewall's rep returns a proof `p` from the log `l` at the rep's base, the cut from the switched source returns the same
transcript and log and a proof `p'` with `eraseZ p' = eraseZ p`. -/
theorem swSrcE_repCut {W R : Type} (Lr : Claims.ZKServer.LeanRelease W R) (ρ : R) (tz : FlockComplete.TapeZk)
    (frep : ℕ → ℕ → ℕ) (x : Msg) (hx : Live x) (w : Claims.Flock.Wit) (hw : RZk x w) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (hAd : ∀ j, (tabC PE x j).Admits (czExe PE x c j)) (n : ℕ)
    (s : _) (spec : _root_.Flock.SessionSpec)
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (extras : Array _root_.Flock.Claim) (he : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    (A : Fin (jOf x) → Fin 2 → Array _root_.Flock.F128) (B : Fin (jOf x) → Fin 2 → ℕ → _root_.Flock.F128)
    (q : _root_.Flock.Firewall.Finish.RepIn) (hA : A j r = q.vals)
    (hB : B j r = FlockComplete.LigLock.l0Clear (ligCoin c) (ligP0 x j r) (zkTab x j).sch (ligTab x w τ j)
      (ligBasis x c w τ j r extras) (layE x j) ((UE.tapeZk τ).extra j.val) r.val)
    (hq : q.draws = Lr.drawsAt x ρ (frep j.val r.val)) (hlayR : Lr.lay x (frep j.val r.val) = layE x j)
    (hz : (Lr.draws ρ (frep j.val r.val)).size =
      _root_.Flock.Firewall.Frame.drawBytes (Lr.lay x (frep j.val r.val)))
    (htape : ∀ k < (layE x j).hab, (UE.tapeZk τ).h j.val r.val k =
      (Claims.ZKServer.tapeOfDraws tz (Lr.lay x) (Lr.draws ρ) frep).h j.val r.val k)
    (hpads : FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j) =
      FlockComplete.padsOf (Claims.ZKServer.tapeOfDraws tz (Lr.lay x) (Lr.draws ρ) frep) j.val r.val (layE x j))
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, valsFw x c A B j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1)
    (st : _root_.Flock.Setup) (hst : st = _root_.Flock.Setup.ofCircuit (stAt x j)) (dom : ByteArray)
    (g : Array _root_.Flock.F128) (hg : g = ligTab x w τ j) (cm0 : FlockComplete.Commit)
    (ic : FlockComplete.InnerCommit) (salt : ℕ → ℕ → ByteArray)
    (hsalt : ∀ i, 1 ≤ i → i < (zkTab x j).sch.levels.size → ∀ q < 2 ^ ((zkTab x j).sch.levels.getD i default).d,
      salt i q = (UE.tapeZk τ).salt j.val r.val i q)
    {t₀ : _root_.Flock.Transcript} (ht₀ : t₀.rounds = #[]) {l : Array Msg}
    (hl : l.size = repBase x (2 * j.val + r.val)) {p : _root_.Flock.Zk.ZkProof} {t : _root_.Flock.Transcript}
    {l' : Array Msg}
    (h : ((_root_.Flock.Firewall.Finish.repFwC (FlockComplete.asksCut (ligCoin c) n)
      (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) st (zkTab x j).sch (zkTab x j).tPad r.val dom g cm0
      extras (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic q.vals ((UE.tapeZk τ).extra j.val)
      ((UE.tapeZk τ).lanePad j.val) salt).run t₀).go l = (.ok (p, t), l')) :
    ∃ p', ((FlockComplete.repCut (FlockComplete.asksCut (ligCoin c) n)
        (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) st (zkTab x j).sch r.val dom
        (FlockComplete.eraseC cm0) extras ic ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val)
        (priorE c j.val r.val) (Composed.maskWordsOf (stsOf x)[j.val]!)).run t₀).go l = (.ok (p', t), l') ∧
      FlockComplete.eraseZ p' = FlockComplete.eraseZ p := by
  subst hg
  have hnr : ¬ Refused x c n := not_refused_of_admits hx hAd n
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  have hstj : (stsOf x)[j.val]! = stAt x j := getElem!_pos (stsOf x) j.val hj
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hm25 : 25 ≤ (stAt x j).m := (hall j.val hj).1
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hkm : (stAt x j).c.kLog ≤ (stAt x j).m := (zkTab x j).wf.1.hk
  have hfk : (stAt x j).circuitFold.kLog = (stAt x j).c.kLog := rfl
  have hSm : st.m = (stAt x j).m := by
    subst hst
    rfl
  have hSk : st.kLog = (stAt x j).c.kLog := by
    subst hst
    rfl
  have hkL : st.fold.kLog = st.kLog := by
    subst hst
    rfl
  have hm14 : 14 ≤ st.m := by
    rw [hSm]
    omega
  have hmerk : st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    subst hst
    exact (hall j.val hj).2.2
  have hLyE : _root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0 = layE x j := by
    subst hst
    rfl
  have heS : (stAt x j).extraClaims (ptsE c spec) = .ok extras := by
    rw [tabsOf_stAt] at he
    exact he
  obtain ⟨hpts, -⟩ := HiddenExec.setupTables_pointsAny (FlockComplete.ZkTables.setupTables_of hx.1 hs)
  have hpsz : (ptsE c spec).size = 2 := by
    unfold ptsE
    simp [hpts]
  have hesz : extras.size = spec.points * (stAt x j).regions.size := by
    rw [FlockDeploy.extraClaims_size heS, hpsz, hpts, Nat.mul_comm]
  have hzq : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [stsOf_half]
    exact (zkTab x j).hz
  -- rep `r`'s PIOP: `m + k_log − 6` bodies of zerocheck and lincheck, then `2 + |extras| + 1` of ring switch
  have hP : ligP0 x j r =
      repBase x (2 * j.val + r.val) + ((stAt x j).m + (stAt x j).c.kLog - 6) + (2 + extras.size + 1) := by
    show repSlot x (2 * j.val + r.val) (fun _ => 0) = _
    rw [repSlot_eq hs hzq, stsOf_half, ← repBase_eq hs (2 * j.val + r.val)]
    unfold FlockComplete.piopRoundsZk
    rw [hesz]
    omega
  -- the firewall's PIOP returned, so it sent all its bodies before the stop
  have hP0 : ligP0 x j r ≤ n := by
    obtain ⟨_, _, _, _, _, _, _, hpf⟩ := _root_.Flock.Firewall.Finish.repFwC_piopFw h
    obtain ⟨-, hl₃, hn₃⟩ := FlockComplete.piopFw_at hm14 hpf (trsE x w (UE.tapeZk τ))[j.val]!
    omega
  have hP0' := hP0
  rw [hP] at hP0'
  have hln : l.size ≤ n := by
    rw [hl]
    omega
  -- the masked values below `zp 64`
  have hmk : ∀ i < (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0).zp 64,
      ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val).masked i =
        q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]! := fun i hi =>
    swSrcE_hmk_frame Lr ρ tz frep x c w A B τ j r q hA hq hlayR hz htape hrows i (by rw [← hLyE]; exact hi)
  -- the switched vectors at the firewall's zerocheck and lincheck
  have hsv : ∀ zp zc t₁ l₁ lp lc t₂ l₂,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut (ligCoin c) n) st.m st.digest (FlockComplete.flat cm0.cap)
          ic.root (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut (ligCoin c) n) st.fold st.m
          (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!) zc).run t₁).go l₁ =
        (.ok ((lp, lc), t₂), l₂) →
      ∀ k < (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras).size,
        ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val).sv k =
          _root_.Flock.Firewall.Finish.switchVec (ligTab x w τ j)
            (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras)[k]! := by
    intro zp zc t₁ l₁ lp lc t₂ l₂ h₁ h₂ k hk
    obtain ⟨hzc, hl₁⟩ := FlockComplete.zerocheckCut_at (ligCoin c) n hm14 _ _ _ _ _ h₁
    obtain ⟨hlc, -⟩ := FlockComplete.lincheckCut_at (ligCoin c) n _ _ _ _ _ h₂
    have hk2 : k < 2 + extras.size := by
      rw [Array.size_append, FlockComplete.abcClaims_size] at hk
      exact hk
    subst hst
    set tr := (trsE x w (UE.tapeZk τ))[j.val]! with htr
    have hzE : FlockComplete.zcCoins (ligCoin c) l.size = zcE c j r := by
      rw [hl]
    have hlE : FlockComplete.lcCoins (ligCoin c) l₁.size
        ((_root_.Flock.Setup.ofCircuit (stAt x j)).fold.kLog - 6) = lcE c j r := by
      rw [hl₁, hl]
      rfl
    rw [hzE] at hzc
    rw [hlE] at hlc
    have hab := FlockComplete.Replay.abc_x (m := (_root_.Flock.Setup.ofCircuit (stAt x j)).m)
      (kLog := (_root_.Flock.Setup.ofCircuit (stAt x j)).kLog) (z₁ := zc) (l₁ := lc)
      (z₂ := FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr)
      (l₂ := FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr
        (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr) (lcE c j r)) hzc.2.1 hzc.2.2 hlc.2
    have hxk := FlockComplete.Replay.claims_x (e := extras) (FlockComplete.abcClaims_size _ _ _ _)
      (FlockComplete.abcClaims_size _ _ _ _) hab.1 hab.2 k
    have hkE : k < (_root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog
        (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr)
        (FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr
          (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr) (lcE c j r)) ++ extras).size := by
      rw [Array.size_append, FlockComplete.abcClaims_size]
      exact hk2
    have hmem : ∀ (a : Array _root_.Flock.Claim) (i : ℕ), i < a.size → a[i]! ∈ a := fun a i hi => by
      rw [getElem!_pos a i hi]
      exact Array.getElem_mem hi
    have hx6E := FlockComplete.Replay.claims_xsize' (stAt x j) hkm (by omega) tr (zcE c j r) (lcE c j r)
      (ptsE c spec) extras heS _ (hmem _ k hkE)
    have hx6 : ((_root_.Flock.abcClaims (_root_.Flock.Setup.ofCircuit (stAt x j)).m
        (_root_.Flock.Setup.ofCircuit (stAt x j)).kLog zc lc ++ extras)[k]!).x.size = (stAt x j).m - 6 := by
      rw [hxk]
      exact hx6E
    have hpk : ligTab x w τ j = FlockComplete.packed tr (stAt x j).m := by
      show FlockComplete.packed tr (FlockDeploy.tabsOf (stsOf x))[j.val]!.m = _
      rw [tabsOf_stAt]
      rfl
    rw [hpk, FlockComplete.switchVec_packed (by omega) tr _ hx6, FlockComplete.Replay.switchVec_x hxk]
    exact swSrc_sv c hw τ j r (valsFw x c A B) hx hnr hs heS k hkE (by rw [hfk]; omega)
  -- the rank check
  have hpiop : ∀ zp zc t₁ l₁ lp lc t₂ l₂,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut (ligCoin c) n) st.m st.digest (FlockComplete.flat cm0.cap)
          ic.root (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut (ligCoin c) n) st.fold st.m
          (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!) zc).run t₁).go l₁ =
        (.ok ((lp, lc), t₂), l₂) →
      FlockComplete.MaskRank.maskRankOk
          (priorE c j.val r.val ++ FlockComplete.MaskRank.roundPoints st.m st.kLog t₂.rounds)
          (Composed.maskWordsOf (stsOf x)[j.val]!) = true ∧
        ∀ k < (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras).size,
          ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val).sv k =
            _root_.Flock.Firewall.Finish.switchVec (ligTab x w τ j)
              (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras)[k]! := fun zp zc t₁ l₁ lp lc t₂ l₂ h₁ h₂ =>
    ⟨rank_at_admits hx hAd hj r.isLt (ligCoin c) n (fun _ _ _ => rfl) st (by rw [hstj, hSm]) (by rw [hstj, hSk]) hkL
      _ _ _ _ ht₀ hl h₁ h₂, hsv zp zc t₁ l₁ lp lc t₂ l₂ h₁ h₂⟩
  refine _root_.Flock.Firewall.Finish.repCut_fw (ligCoin c) n st hmerk (zkTab x j).sch (zkTab x j).tPad r.val dom
    (ligTab x w τ j) cm0 extras (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic q.vals
    ((UE.tapeZk τ).extra j.val) ((UE.tapeZk τ).lanePad j.val) salt
    ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val) (priorE c j.val r.val)
    (Composed.maskWordsOf (stsOf x)[j.val]!) t₀ l hm14 hln hkL hmk hpiop ?_
    (swSrcE_hinner_sv Lr ρ tz frep x hx w τ c n hnr s spec hs j r A B q hA hq hlayR hz htape hpads hrows st hst
      (zkTab x j).tPad dom (ligTab x w τ j) cm0 extras hesz ic _ _ _ ht₀ hl hsv) h
  -- level 0's opening: the firewall's PIOP ends at `ligP0` and switched `swsE`
  intro zp lp ss T sws t₃ l₃ h₃
  obtain ⟨hsws, hl₃, -⟩ := FlockComplete.piopFw_at hm14 h₃ (trsE x w (UE.tapeZk τ))[j.val]!
  have hl3 : l₃.size = ligP0 x j r := by
    rw [hl₃, hl, hP, hkL, hSm, hSk, hesz]
    omega
  have hsE : sws = swsE x c w τ j r extras := by
    rw [hsws]
    unfold swsE
    rw [tabsOf_stAt, ← hst, hl, zcCoins_ligRespE, lcCoins_ligRespE, rsCoins_ligRespE]
  rw [hl3, hsE]
  have H := swSrcE_ligHyp_E Lr ρ tz frep x hx w hw τ c n hnr s spec hs j r extras he A B q hB hq hlayR hz htape hrows
  unfold ligBasis at H
  rw [show ligSt x j = st from (tabsOf_stAt x j).trans hst.symm, ← hLyE] at H
  exact H.ofSalt hsalt

end Masked
end Claims.ZKMal
