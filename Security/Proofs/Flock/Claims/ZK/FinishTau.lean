import Proofs.Flock.Claims.ZK.FinishRows
import Proofs.Flock.Claims.ZK.FinishLane
import Proofs.Flock.Claims.ZK.FinishZc
import Proofs.Flock.Claims.ZK.FinishRank
import Proofs.Flock.Claims.ZK.FinishSrc
import Proofs.Flock.Claims.ZK.PiopSrcLc
import Proofs.Flock.Claims.ZK.SimLig
import Proofs.Flock.Claims.Complete.Zk.ConsReplay
import Proofs.Flock.Claims.Complete.Zk.Json.Session

/-!
# The firewall's seven rows at a worker's values

`FinishStopAt.rowsAt_of_cons` asks that the firewall's constraints be the first seven rows on the view's masked values,
constants included. Here they are at the firewall's own run (`repFwC` answered by `asksCut` from a log `l`
and no rounds): its zerocheck and lincheck send the masked values `vals + h` and leave rounds answered by the coins at
their sends, which the rounds after them keep (`growS`), so the replay reads its masked proof as the view's
(`FinishZc`, `FinishRows`' `zcOf_eq`, …) under the coins `coin (l.size + j)`, and `FinishRows.replay_rows` applies; its
level 0's two lane rows are `FinishLane.lanesAt`'s where its Ligerito proof is read at the view (`LaneAt`), and
`TauLig.lanesA`'s by coefficient where only its record's coins are the model's (`LaneCoins`).

* **The rounds after the lincheck** (`ringSwitchCut_grows`, `ligeritoPZk_grows`): the ring switch and Ligerito only add
  rounds, each answered by its send's coin, as the zerocheck and lincheck do (`FinishRank`).
* **`cut_coins`**: from no rounds, the zerocheck and lincheck leave `m + k_log − 6` rounds, round `j` answered by
  `coin (l.size + j)`, round `1` with `m − 13` coins.
* **`lincheckCut_fold`**: a lincheck that returns had its circuit's fold defined at its `α`.
* **`tauFw`**: where `repFwC` returns, its constraints pass on the pads and are `rowsOf`'s first seven, at coins that
  are the model's (`hz`, …, `hr'`), masked values that are the view's below `zp 64` (`hmk`), folds that realize the
  statement (`hfold`), switched vectors that are the view's `ŝ` (`hsw`) and a level 0 read at the view (`hlane`).
  `tauFw_at` is it at the run's steps, `tauFw_at_replay` its replay's five rows, and **`tauFw_at_coef`** its rows by
  coefficient where level 0's record holds the model's coins (`LaneCoins`), as `FinishSwFw` uses it.
-/

namespace FlockComplete.RankAt

open Flock Transcript FlockComplete FlockDeploy With

/-- Round `j` of a prefix is the round's. -/
theorem prefix_get {A B : Array Round} (h : Replay.Prefix A B) {j : ℕ} (hj : j < A.size) : B[j]! = A[j]! := by
  obtain ⟨C, rfl⟩ := h
  rw [getElem!_pos _ j (by rw [Array.size_append]; omega), Array.getElem_append_left hj, ← getElem!_pos A j hj]

/-- A round answered by `coin (L + j)` has `coin (L + j) 0` first. -/
theorem coin_getD {coin : ℕ → ℕ → F128} {L : ℕ} {R : Array Round} (hR : CoinsAt coin L R) {j : ℕ} (hj : j < R.size) :
    (R[j]!.coins).getD 0 0 = coin (L + j) 0 := by
  rw [(hR j hj).2, Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range]
  simp [(hR j hj).1]

section Asks

variable (coin : ℕ → ℕ → F128) (n L : ℕ)

theorem absorbK_grows (v : F256) : (growS coin L).R (liftTM (absorbK v) : TMW PS Unit) (liftTM (absorbK v)) :=
  liftTM_grows coin L _ fun _ _ _ h => by cases h; rfl

theorem pow_grows (v : UInt64) : (growS coin L).R (liftTM (pow v) : TMW PS Unit) (liftTM (pow v)) :=
  liftTM_grows coin L _ fun _ _ _ h => by cases h; rfl

theorem commitP_grows (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (g : Array F128)
    (salt : ℕ → ByteArray) :
    (growS coin L).R (StateT.lift (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt) : TMW PS Commit)
      (StateT.lift (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt)) := by
  intro t l a t' l' e
  change ((commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt) >>= fun a => Pure.pure (a, t)).go l = _ at e
  obtain ⟨b, l₁, e1, e2⟩ := JsonRT.go_bind_ok e
  have := JsonRT.Still.go (JsonRT.Still.of_pureS (JsonRT.commitP_pureS h ms lv g salt)) e1
  subst this
  cases e2
  exact ⟨Replay.Prefix.refl _, id⟩

theorem liftT_grows {α : Type} {x y : TMW PS α} (h : (growS coin L).R x y) :
    ((growS coin L).st LigState).R (liftT x) (liftT y) :=
  With.Sim.lift_st _ h

theorem sampleKP_grows : (growS coin L).R (sampleKP (asksCut coin n)) (sampleKP (asksCut coin n)) := by
  unfold sampleKP
  sim_walk [squeezeP_grows coin n L]

theorem sendMsg_grows (m : F256 × F256) :
    ((growS coin L).st LigState).R (sendMsg m : LP PS Unit) (sendMsg m) := by
  unfold sendMsg
  sim_walk [liftT_grows coin L, absorbK_grows coin L]

theorem powP_grows (nonce : ℕ → ℕ → UInt64) (kind : ℕ) :
    ((growS coin L).st LigState).R (powP nonce kind : LP PS Unit) (powP nonce kind) := by
  unfold powP
  sim_walk [liftT_grows coin L, pow_grows coin L]

theorem challengeP_grows (split : Bool) :
    ((growS coin L).st LigState).R (challengeP (asksCut coin n) split) (challengeP (asksCut coin n) split) := by
  unfold challengeP
  sim_walk [liftT_grows coin L, sampleKP_grows coin n L, sendMsg_grows coin L]

theorem oodIntroP_grows (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ) :
    ((growS coin L).st LigState).R (oodIntroP (asksCut coin n) nonce g dim) (oodIntroP (asksCut coin n) nonce g dim) := by
  unfold oodIntroP sampleVecP
  sim_walk [liftT_grows coin L, squeezeP_grows coin n L, sampleP_grows coin n L, sendMsg_grows coin L,
    powP_grows coin L, absorbF_grows coin L]

theorem queryP_grows (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit) :
    ((growS coin L).st LigState).R (queryP (asksCut coin n) nonce ms lv cm) (queryP (asksCut coin n) nonce ms lv cm) := by
  unfold queryP sampleVecP
  sim_walk [liftT_grows coin L, squeezeP_grows coin n L, powP_grows coin L]

theorem ringSwitchCut_grows (sv : ℕ → Array F128) (claims : Array Claim) :
    (growS coin L).R (ringSwitchCut (asksCut coin n) sv claims) (ringSwitchCut (asksCut coin n) sv claims) := by
  unfold ringSwitchCut sampleVecP
  sim_walk [squeezeP_grows coin n L, label_grows coin L, absorbSlice_grows coin L]

theorem ligeritoPZk_grows (h : ByteArray → ByteArray) (ms : MerkleScheme) (sch : Schedule) (T : F128) (cm0 : Commit)
    (t b : Array F128) (lay : Zk.PadLayout) (hp : Array F128) (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    (growS coin L).R
      (ligeritoPZk (asksCut coin n) (Hashes.ofSha512 h) ms sch T cm0 t b lay hp rep tPad extra lanePad salt nonce)
      (ligeritoPZk (asksCut coin n) (Hashes.ofSha512 h) ms sch T cm0 t b lay hp rep tPad extra lanePad salt nonce) := by
  unfold ligeritoPZk sampleVecP
  sim_walk [liftT_grows coin L, squeezeP_grows coin n L, sampleP_grows coin n L, sampleKP_grows coin n L,
    sendMsg_grows coin L, powP_grows coin L, challengeP_grows coin n L, oodIntroP_grows coin n L,
    queryP_grows coin n L, commitP_grows coin L h, label_grows coin L, absorbF_grows coin L, absorbBytes_grows coin L,
    absorbK_grows coin L, absorbSlice_grows coin L]

/-- **The rounds a zerocheck and lincheck leave from no rounds**: `m + k_log − 6` of them, round `j` answered by
`coin (l.size + j)`, round `1` with `m − 13` coins. -/
theorem cut_coins {m : ℕ} (hm : 14 ≤ m) {fold : CircuitFold} (hk : 6 ≤ fold.kLog) {digest cap root : ByteArray}
    {lay : Zk.PadLayout} {mk : ℕ → F128} {t₀ : Transcript} (ht₀ : t₀.rounds = #[]) {l : Array Claims.Msg}
    {zp : ZerocheckProof} {zc : ZcOut} {t₁ : Transcript} {l₁ : Array Claims.Msg} {lp : LincheckProof} {lc : LcOut}
    {t₂ : Transcript} {l₂ : Array Claims.Msg}
    (h₁ : ((zerocheckCut (asksCut coin n) m digest cap root lay mk).run t₀).go l = (.ok ((zp, zc), t₁), l₁))
    (h₂ : ((lincheckCut (asksCut coin n) fold m lay mk zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂)) :
    CoinsAt coin l.size t₂.rounds ∧ t₂.rounds.size = m + fold.kLog - 6 ∧ t₂.rounds[1]!.n = m - 13 := by
  have i0 : Inv coin l.size t₀ l := ⟨fun j hj => absurd hj (by simp [ht₀]), by simp [ht₀]⟩
  obtain ⟨-, hi1⟩ := zerocheckCut_grows coin n l.size m digest cap root lay mk t₀ l _ t₁ l₁ h₁
  obtain ⟨p2, hi2⟩ := lincheckCut_grows coin n l.size fold m lay mk zc t₁ l₁ _ t₂ l₂ h₂
  obtain ⟨-, hs1⟩ := hi1 i0
  obtain ⟨hc2, hs2⟩ := hi2 (hi1 i0)
  obtain ⟨hl, hz1⟩ := zerocheckCut_round1 coin n hm digest cap root lay mk h₁
  obtain ⟨e1, e1n, -⟩ := (CutEq.zerocheckCut_eqAt coin n hm digest cap root lay mk mk t₀ (l := l) rfl (by omega)
    (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ => ⟨rfl, rfl, rfl⟩)).2 _ _ h₁
  obtain ⟨e2, -, -⟩ := (CutEq.lincheckCut_eqAt coin n fold m lay mk mk zc t₁ (l := l₁) rfl (by omega)
    (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ _ _ => rfl)).2 _ _ h₂
  have hsz : t₂.rounds.size = m + fold.kLog - 6 := by omega
  refine ⟨hc2, hsz, ?_⟩
  obtain ⟨C, hC⟩ := p2
  have h1s : 1 < t₁.rounds.size := by omega
  rw [hC, getElem!_pos _ 1 (by rw [Array.size_append]; omega), Array.getElem_append_left h1s,
    ← getElem!_pos t₁.rounds 1 h1s]
  simpa [ht₀] using hz1

/-- **A lincheck that returns had its fold defined** at its `α`, the coin of its first send. -/
theorem lincheckCut_fold {c : CircuitFold} {m : ℕ} {lay : Zk.PadLayout} {mk : ℕ → F128} {zc : ZcOut}
    {t t' : Transcript} {l l' : Array Claims.Msg} {res : LincheckProof × LcOut}
    (h : ((lincheckCut (asksCut coin n) c m lay mk zc).run t).go l = (.ok (res, t'), l')) :
    ∃ w, c.fold (coin l.size 0) (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) = .ok w := by
  unfold lincheckCut at h
  rw [Replay.bind_go (Replay.liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)] at h
  obtain ⟨a, t₁, l₁, h1, h⟩ := run_bind_ok h
  obtain ⟨-, rfl, -⟩ := sampleP_cut_inv coin h1
  obtain ⟨_, _, _, -, h⟩ := run_bind_ok h
  rcases e : c.fold (coin l.size 0) (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) with err | w
  · simp only [e] at h
    obtain ⟨_, _, _, hthrow, -⟩ := run_bind_ok h
    rw [Flock.Firewall.Finish.throw_run_go] at hthrow
    cases hthrow
  · exact ⟨w, rfl⟩

end Asks

end FlockComplete.RankAt

namespace FlockSoundness.Discharge.Composed
open Flock Model ZkSession ZkView ZK Refine

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-- **The replay's five rows at the firewall's own run's steps**: the replay's three and the two ring-switch rows are
`rowsOf`'s first five, at the zerocheck, lincheck, ring switch and Ligerito runs and the replay that `repCut_fw`'s
`hinner` is handed (`h₁`, …, `hrp`), under `tauFw`'s premises on the coins, masked values, fold and switched vectors. -/
theorem tauFw_at_replay {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (stm : Stmt) (c : CoinsZK (RestOf Tz stm)) (r : Fin 2) {nE : ℕ}
    (v : Fin nE → 𝔽) {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) (T' : 𝕂) (ρσ : 𝔽 × 𝔽)
    (coin : ℕ → ℕ → Flock.F128) (n : ℕ) (st : Flock.Setup) (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (g : Array Flock.F128) (cm0 : FlockComplete.Commit) (extras : Array Flock.Claim) (pads : FlockComplete.PadsZk)
    (ic : FlockComplete.InnerCommit) (vals : Array Flock.F128) (extra lanePad : ℕ → ℕ → Flock.F128)
    (salt : ℕ → ℕ → ByteArray) {t₀ : Flock.Transcript} (ht₀ : t₀.rounds = #[]) {l : Array _root_.Claims.Msg}
    (hm : 14 ≤ st.m) (hk6 : 6 ≤ st.kLog) (hSm : st.m = stm.m) (hSk : st.kLog = stm.c.kLog)
    (hkL : st.fold.kLog = st.kLog) (hlay : layOf Tz stm = Zk.PadLayout.new st.m st.kLog sch.k0)
    (hmk : ∀ k < (layOf Tz stm).zp 64, (msgAt v k).val = vals[k]! + pads.h[k]!)
    (hz : coin (l.size + 2) 0 = zcF Tz stm c r)
    (hr : pinned ++ (Array.range (st.m - 13)).map (coin (l.size + 1)) = rF Tz stm c r)
    (hρ : rhoF Tz stm c r = FlockComplete.Replay.pushes (fun j => coin (l.size + (3 + j)) 0) (st.m - 6))
    (hα : coin (l.size + (3 + (st.m - 6))) 0 = c.rest.lcAlpha r)
    (hβ : coin (l.size + (4 + (st.m - 6))) 0 = c.rest.lcBeta r)
    (hτ : tsF Tz stm c r =
      FlockComplete.Replay.pushes (fun j => coin (l.size + (5 + (st.m - 6) + j)) 0) (st.kLog - 6))
    (hr' : coin (l.size + (5 + (st.m - 6) + (st.kLog - 6))) 0 = c.rest.rPrime r)
    (hfold : ∀ combv, st.fold.folded (c.rest.lcAlpha r) (c.rest.lcBeta r) (skipWeights (zcF Tz stm c r))
      ((rhoF Tz stm c r).extract 0 (stm.c.kLog - 6)) (tsF Tz stm c r) = .ok combv → combv = combvOf Tz stm c r)
    (hsw : ∀ zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut coin n) st.m st.digest (FlockComplete.flat cm0.cap) ic.root
          (Zk.PadLayout.new st.m st.kLog sch.k0) fun i => vals[i]! + pads.h[i]!).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut coin n) st.fold st.m (Zk.PadLayout.new st.m st.kLog sch.k0)
          (fun i => vals[i]! + pads.h[i]!) zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂) →
      ((FlockComplete.ringSwitchCut (FlockComplete.asksCut coin n)
          (fun k => Flock.Firewall.Finish.switchVec g (abcClaims st.m st.kLog zc lc ++ extras)[k]!)
          (abcClaims st.m st.kLog zc lc ++ extras)).run t₂).go l₂ = (.ok ((ss, T, sws), t₃), l₃) →
      ss[0]! = shArr sh (2 * r.val) ∧ ss[1]! = shArr sh (2 * r.val + 1))
    {zp : ZerocheckProof} {zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃ lg zl rec t₄ l₄ rp tt}
    (h₁ : ((FlockComplete.zerocheckCut (FlockComplete.asksCut coin n) st.m st.digest (FlockComplete.flat cm0.cap)
        ic.root (Zk.PadLayout.new st.m st.kLog sch.k0) fun i => vals[i]! + pads.h[i]!).run t₀).go l =
      (.ok ((zp, zc), t₁), l₁))
    (h₂ : ((FlockComplete.lincheckCut (FlockComplete.asksCut coin n) st.fold st.m
        (Zk.PadLayout.new st.m st.kLog sch.k0) (fun i => vals[i]! + pads.h[i]!) zc).run t₁).go l₁ =
      (.ok ((lp, lc), t₂), l₂))
    (h₃ : ((FlockComplete.ringSwitchCut (FlockComplete.asksCut coin n)
        (fun k => Flock.Firewall.Finish.switchVec g (abcClaims st.m st.kLog zc lc ++ extras)[k]!)
        (abcClaims st.m st.kLog zc lc ++ extras)).run t₂).go l₂ = (.ok ((ss, T, sws), t₃), l₃))
    (h₄ : ((FlockComplete.ligeritoPZk (FlockComplete.asksCut coin n) (Flock.Hashes.ofSha512 Flock.Sha512.hash)
        Flock.MerkleScheme.hm96Sha512 sch T cm0 g (FlockComplete.basisVals sws (st.m - 7))
        (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h rep tPad extra lanePad salt (fun _ _ => 0)).run t₃).go l₃ =
      (.ok ((lg, zl, rec), t₄), l₄))
    (hrp : (Zk.replay st (Zk.PadLayout.new st.m st.kLog sch.k0)
        { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp, ringSwitches := ss,
          ligerito := lg } ic.root).run (Flock.Transcript.init t₄.rounds dom) = .ok (rp, tt))
    (lgc : Array (String × Zk.Lin)) :
    ∀ k (hk : k < 5), (ofZk (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]!
        ss[0]!)), ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)[k]!.2 :
        ZkSession.Lin 𝔽 (nE + 4)) = rowsOf Tz stm c r v sh T' ρσ (Fin.castAdd 3 ⟨k, by omega⟩) := by
  intro k hk
  -- the rounds the replay reads
  obtain ⟨hc2, hsz2, hr1⟩ := FlockComplete.RankAt.cut_coins coin n hm (by rw [hkL]; exact hk6) ht₀ h₁ h₂
  have hS2 : t₂.rounds.size = st.m + st.kLog - 6 := by rw [hsz2, hkL]
  have hpre := (FlockComplete.RankAt.prefix_of_grows coin 0
    (FlockComplete.RankAt.ringSwitchCut_grows coin n 0 _ _) h₃).trans
    (FlockComplete.RankAt.prefix_of_grows coin 0
      (FlockComplete.RankAt.ligeritoPZk_grows coin n 0 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) h₄)
  have hcT : ∀ j < t₂.rounds.size,
      Lincheck.coinAt (Flock.Transcript.init t₄.rounds dom) j = coin (l.size + j) 0 := fun j hj => by
    show (t₄.rounds[0 + j]!).coins.getD 0 0 = _
    rw [Nat.zero_add, FlockComplete.RankAt.prefix_get hpre hj]
    exact FlockComplete.RankAt.coin_getD hc2 hj
  have ez : ZerocheckMasked.zAt (Flock.Transcript.init t₄.rounds dom) = zcF Tz stm c r :=
    (hcT 2 (by omega)).trans hz
  have er : pinned ++ ZerocheckMasked.rOuterAt (Flock.Transcript.init t₄.rounds dom) = rF Tz stm c r := by
    rw [← hr]
    congr 1
    show (t₄.rounds[0 + 1]!).coins = _
    rw [Nat.zero_add, FlockComplete.RankAt.prefix_get hpre (by omega), (hc2 1 (by omega)).2, hr1]
  have eρ : ZerocheckMasked.rhoAt (Flock.Transcript.init t₄.rounds dom) (st.m - 6) = rhoF Tz stm c r := by
    rw [hρ]
    exact FlockComplete.Replay.rhoAt_eq _ _ _ fun j hj => hcT (3 + j) (by omega)
  have eρ' : Lincheck.rhoOf st.m (Flock.Transcript.init t₄.rounds dom) = rhoF Tz stm c r := by
    rw [hρ]
    refine _root_.Claims.ZKMal.Masked.arr_eq_pushes (by simp only [Lincheck.rhoOf, Array.size_ofFn])
      fun j hj => ?_
    rw [Lincheck.rhoOf, getElem!_pos _ j (by rw [Array.size_ofFn]; exact hj), Array.getElem_ofFn]
    exact hcT (3 + j) (by omega)
  have eα : Lincheck.coinAt (Flock.Transcript.init t₄.rounds dom) (3 + (st.m - 6)) = c.rest.lcAlpha r :=
    (hcT _ (by omega)).trans hα
  have eβ : Lincheck.coinAt (Flock.Transcript.init t₄.rounds dom) (4 + (st.m - 6)) = c.rest.lcBeta r :=
    (hcT _ (by omega)).trans hβ
  have eτ : Lincheck.tsOf st.m st.kLog (Flock.Transcript.init t₄.rounds dom) = tsF Tz stm c r := by
    rw [hτ]
    refine _root_.Claims.ZKMal.Masked.arr_eq_pushes (by simp only [Lincheck.tsOf, Array.size_ofFn])
      fun j hj => ?_
    rw [Lincheck.tsOf, getElem!_pos _ j (by rw [Array.size_ofFn]; exact hj), Array.getElem_ofFn]
    exact hcT (5 + (st.m - 6) + j) (by omega)
  have er' : Lincheck.coinAt (Flock.Transcript.init t₄.rounds dom) (5 + (st.m - 6) + (st.kLog - 6)) =
      c.rest.rPrime r :=
    (hcT _ (by omega)).trans hr'
  -- the masked proof the replay reads
  have hl0 := (FlockComplete.RankAt.zerocheckCut_round1 coin n hm _ _ _ _ _ h₁).1
  obtain ⟨hl1, hl1n, -⟩ := (FlockComplete.CutEq.zerocheckCut_eqAt coin n hm st.digest (FlockComplete.flat cm0.cap)
    ic.root (Zk.PadLayout.new st.m st.kLog sch.k0) (fun i => vals[i]! + pads.h[i]!) (fun i => vals[i]! + pads.h[i]!)
    t₀ (l := l) rfl hl0.le (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ => ⟨rfl, rfl, rfl⟩)).2 _ _ h₁
  obtain ⟨hl2, hl2n, -⟩ := (FlockComplete.CutEq.lincheckCut_eqAt coin n st.fold st.m
    (Zk.PadLayout.new st.m st.kLog sch.k0) (fun i => vals[i]! + pads.h[i]!) (fun i => vals[i]! + pads.h[i]!) zc t₁
    (l := l₁) rfl (by omega) (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ _ _ => rfl)).2 _ _ h₂
  obtain ⟨zc', t₁', L₁', hZ⟩ := _root_.Claims.ZKServer.zerocheckCut_proof coin n hm st.digest
    (FlockComplete.flat cm0.cap) ic.root (Zk.PadLayout.new st.m st.kLog sch.k0) (fun i => vals[i]! + pads.h[i]!) t₀ l
    (by omega)
  rw [h₁] at hZ
  simp only [Prod.mk.injEq, Except.ok.injEq] at hZ
  obtain ⟨⟨⟨hzp, -⟩, -⟩, -⟩ := hZ
  obtain ⟨lc', t₂', L₂', hLc⟩ := _root_.Claims.ZKServer.lincheckCut_proof coin n st.fold st.m
    (Zk.PadLayout.new st.m st.kLog sch.k0) (fun i => vals[i]! + pads.h[i]!) zc t₁ l₁ (by omega)
    (FlockComplete.RankAt.lincheckCut_fold coin n h₂)
  rw [h₂] at hLc
  simp only [Prod.mk.injEq, Except.ok.injEq] at hLc
  obtain ⟨⟨⟨hlp, -⟩, -⟩, -⟩ := hLc
  have hzc : zp = zcOf Tz stm v := by
    rw [hzp, zcOf_eq Tz stm v (fun i => vals[i]! + pads.h[i]!) hmk, hlay, ← hSm]
  have hrd : lp.rounds = lcRoundsOf Tz stm v := by
    rw [hlp, lcRoundsOf_eq Tz stm v (fun i => vals[i]! + pads.h[i]!) hmk, hlay, hkL, hSk]
  have hzp' : lp.zPartial = zPartialOf Tz stm v := by
    rw [hlp, zPartialOf_eq Tz stm v (fun i => vals[i]! + pads.h[i]!) hmk, hlay]
  obtain ⟨hs0, hs1⟩ := hsw zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃ h₁ h₂ h₃
  have hrp' : Zk.replay st (layOf Tz stm)
      { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp, ringSwitches := ss,
        ligerito := lg } ic.root (Flock.Transcript.init t₄.rounds dom) = .ok (rp, tt) := by
    rw [hlay]
    exact hrp
  exact replay_rows Tz stm c r v sh T' ρσ st _ ic.root hm hSm hSk hrp' hzc hrd hzp' ez er eρ eρ' eα eβ eτ er'
    hfold ss hs0 hs1 lgc k hk

/-- **The firewall's seven rows at its own run's steps**: `tauFw`'s rows, at the zerocheck, lincheck, ring switch and
Ligerito runs, the replay and level 0's constraints that `repCut_fw`'s `hinner` is handed (`h₁`, …, `hlgc`), under
`tauFw`'s premises on the coins, masked values, fold, switched vectors and level 0. -/
theorem tauFw_at {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (stm : Stmt) (c : CoinsZK (RestOf Tz stm)) (r : Fin 2) {nE : ℕ}
    (v : Fin nE → 𝔽) {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) (T' : 𝕂) (ρσ : 𝔽 × 𝔽)
    (coin : ℕ → ℕ → Flock.F128) (n : ℕ) (st : Flock.Setup) (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (g : Array Flock.F128) (cm0 : FlockComplete.Commit) (extras : Array Flock.Claim) (pads : FlockComplete.PadsZk)
    (ic : FlockComplete.InnerCommit) (vals : Array Flock.F128) (extra lanePad : ℕ → ℕ → Flock.F128)
    (salt : ℕ → ℕ → ByteArray) {t₀ : Flock.Transcript} (ht₀ : t₀.rounds = #[]) {l : Array _root_.Claims.Msg}
    (hm : 14 ≤ st.m) (hk6 : 6 ≤ st.kLog) (hSm : st.m = stm.m) (hSk : st.kLog = stm.c.kLog)
    (hkL : st.fold.kLog = st.kLog) (hlay : layOf Tz stm = Zk.PadLayout.new st.m st.kLog sch.k0)
    (hmk : ∀ k < (layOf Tz stm).zp 64, (msgAt v k).val = vals[k]! + pads.h[k]!)
    (hz : coin (l.size + 2) 0 = zcF Tz stm c r)
    (hr : pinned ++ (Array.range (st.m - 13)).map (coin (l.size + 1)) = rF Tz stm c r)
    (hρ : rhoF Tz stm c r = FlockComplete.Replay.pushes (fun j => coin (l.size + (3 + j)) 0) (st.m - 6))
    (hα : coin (l.size + (3 + (st.m - 6))) 0 = c.rest.lcAlpha r)
    (hβ : coin (l.size + (4 + (st.m - 6))) 0 = c.rest.lcBeta r)
    (hτ : tsF Tz stm c r =
      FlockComplete.Replay.pushes (fun j => coin (l.size + (5 + (st.m - 6) + j)) 0) (st.kLog - 6))
    (hr' : coin (l.size + (5 + (st.m - 6) + (st.kLog - 6))) 0 = c.rest.rPrime r)
    (hfold : ∀ combv, st.fold.folded (c.rest.lcAlpha r) (c.rest.lcBeta r) (skipWeights (zcF Tz stm c r))
      ((rhoF Tz stm c r).extract 0 (stm.c.kLog - 6)) (tsF Tz stm c r) = .ok combv → combv = combvOf Tz stm c r)
    (hsw : ∀ zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut coin n) st.m st.digest (FlockComplete.flat cm0.cap) ic.root
          (Zk.PadLayout.new st.m st.kLog sch.k0) fun i => vals[i]! + pads.h[i]!).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut coin n) st.fold st.m (Zk.PadLayout.new st.m st.kLog sch.k0)
          (fun i => vals[i]! + pads.h[i]!) zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂) →
      ((FlockComplete.ringSwitchCut (FlockComplete.asksCut coin n)
          (fun k => Flock.Firewall.Finish.switchVec g (abcClaims st.m st.kLog zc lc ++ extras)[k]!)
          (abcClaims st.m st.kLog zc lc ++ extras)).run t₂).go l₂ = (.ok ((ss, T, sws), t₃), l₃) →
      ss[0]! = shArr sh (2 * r.val) ∧ ss[1]! = shArr sh (2 * r.val + 1))
    (hlen : (layOf Tz stm).len = nE + 4)
    (hlane : ∀ zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃ lg zl rec t₄ l₄,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut coin n) st.m st.digest (FlockComplete.flat cm0.cap) ic.root
          (Zk.PadLayout.new st.m st.kLog sch.k0) fun i => vals[i]! + pads.h[i]!).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut coin n) st.fold st.m (Zk.PadLayout.new st.m st.kLog sch.k0)
          (fun i => vals[i]! + pads.h[i]!) zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂) →
      ((FlockComplete.ringSwitchCut (FlockComplete.asksCut coin n)
          (fun k => Flock.Firewall.Finish.switchVec g (abcClaims st.m st.kLog zc lc ++ extras)[k]!)
          (abcClaims st.m st.kLog zc lc ++ extras)).run t₂).go l₂ = (.ok ((ss, T, sws), t₃), l₃) →
      ((FlockComplete.ligeritoPZk (FlockComplete.asksCut coin n) (Flock.Hashes.ofSha512 Flock.Sha512.hash)
          Flock.MerkleScheme.hm96Sha512 sch T cm0 g (FlockComplete.basisVals sws (st.m - 7))
          (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h rep tPad extra lanePad salt (fun _ _ => 0)).run t₃).go l₃ =
        (.ok ((lg, zl, rec), t₄), l₄) →
      _root_.Claims.ZKMal.Masked.LaneAt Tz stm c r v sh T' lg rec zl)
    {zp : ZerocheckProof} {zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃ lg zl rec t₄ l₄ rp tt lgc}
    (h₁ : ((FlockComplete.zerocheckCut (FlockComplete.asksCut coin n) st.m st.digest (FlockComplete.flat cm0.cap)
        ic.root (Zk.PadLayout.new st.m st.kLog sch.k0) fun i => vals[i]! + pads.h[i]!).run t₀).go l =
      (.ok ((zp, zc), t₁), l₁))
    (h₂ : ((FlockComplete.lincheckCut (FlockComplete.asksCut coin n) st.fold st.m
        (Zk.PadLayout.new st.m st.kLog sch.k0) (fun i => vals[i]! + pads.h[i]!) zc).run t₁).go l₁ =
      (.ok ((lp, lc), t₂), l₂))
    (h₃ : ((FlockComplete.ringSwitchCut (FlockComplete.asksCut coin n)
        (fun k => Flock.Firewall.Finish.switchVec g (abcClaims st.m st.kLog zc lc ++ extras)[k]!)
        (abcClaims st.m st.kLog zc lc ++ extras)).run t₂).go l₂ = (.ok ((ss, T, sws), t₃), l₃))
    (h₄ : ((FlockComplete.ligeritoPZk (FlockComplete.asksCut coin n) (Flock.Hashes.ofSha512 Flock.Sha512.hash)
        Flock.MerkleScheme.hm96Sha512 sch T cm0 g (FlockComplete.basisVals sws (st.m - 7))
        (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h rep tPad extra lanePad salt (fun _ _ => 0)).run t₃).go l₃ =
      (.ok ((lg, zl, rec), t₄), l₄))
    (hrp : (Zk.replay st (Zk.PadLayout.new st.m st.kLog sch.k0)
        { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp, ringSwitches := ss,
          ligerito := lg } ic.root).run (Flock.Transcript.init t₄.rounds dom) = .ok (rp, tt))
    (hlgc : Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rec lg zl = .ok lgc) :
    ∀ k (hk : k < 7), (ofZk (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]!
        ss[0]!)), ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)[k]!.2 :
        ZkSession.Lin 𝔽 (nE + 4)) = rowsOf Tz stm c r v sh T' ρσ (Fin.castAdd 3 ⟨k, by omega⟩) := by
  intro k hk
  by_cases hk5 : k < 5
  · exact tauFw_at_replay Tz stm c r v sh T' ρσ coin n st sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt
      ht₀ hm hk6 hSm hSk hkL hlay hmk hz hr hρ hα hβ hτ hr' hfold hsw h₁ h₂ h₃ h₄ hrp lgc k hk5
  obtain ⟨k', rfl⟩ : ∃ k' : Fin 2, k = 5 + k'.val := ⟨⟨k - 5, by omega⟩, by simp only; omega⟩
  have hrp' : Zk.replay st (layOf Tz stm)
      { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp, ringSwitches := ss,
        ligerito := lg } ic.root (Flock.Transcript.init t₄.rounds dom) = .ok (rp, tt) := by
    rw [hlay]
    exact hrp
  have hL := hlane zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃ lg zl rec t₄ l₄ h₁ h₂ h₃ h₄
  have hlgc' : Zk.ligeritoConstraints (layOf Tz stm) rec lg zl = .ok lgc := by
    rw [hlay]
    exact hlgc
  obtain ⟨hsz2, hrow⟩ := _root_.Claims.ZKMal.Masked.lanesAt Tz stm c r v sh T' hlen lg rec zl hL hlgc'
  have h3 := ZkExec.replay_cons_size _ _ _ _ hrp'
  rw [ZkExec.append_get!_right _ _ (j := k'.val) (by rw [Array.size_append, h3]; rfl) (by rw [hsz2]; exact k'.isLt),
    hrow k']
  fin_cases k' <;> rfl

/-- **The firewall's seven rows at its own run's steps, by coefficient**: `tauFw_at`'s coefficients, with level 0's
two rows of the layout's length, where level 0's record holds the model's coins (`LaneCoins`) rather than all of
level 0 being read at the view (`LaneAt`): `lanesA` reads only those. -/
theorem tauFw_at_coef {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (stm : Stmt) (c : CoinsZK (RestOf Tz stm)) (r : Fin 2)
    {nE : ℕ} (v : Fin nE → 𝔽) {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) (T' : 𝕂) (ρσ : 𝔽 × 𝔽)
    (coin : ℕ → ℕ → Flock.F128) (n : ℕ) (st : Flock.Setup) (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (g : Array Flock.F128) (cm0 : FlockComplete.Commit) (extras : Array Flock.Claim) (pads : FlockComplete.PadsZk)
    (ic : FlockComplete.InnerCommit) (vals : Array Flock.F128) (extra lanePad : ℕ → ℕ → Flock.F128)
    (salt : ℕ → ℕ → ByteArray) {t₀ : Flock.Transcript} (ht₀ : t₀.rounds = #[]) {l : Array _root_.Claims.Msg}
    (hm : 14 ≤ st.m) (hk6 : 6 ≤ st.kLog) (hSm : st.m = stm.m) (hSk : st.kLog = stm.c.kLog)
    (hkL : st.fold.kLog = st.kLog) (hlay : layOf Tz stm = Zk.PadLayout.new st.m st.kLog sch.k0)
    (hmk : ∀ k < (layOf Tz stm).zp 64, (msgAt v k).val = vals[k]! + pads.h[k]!)
    (hz : coin (l.size + 2) 0 = zcF Tz stm c r)
    (hr : pinned ++ (Array.range (st.m - 13)).map (coin (l.size + 1)) = rF Tz stm c r)
    (hρ : rhoF Tz stm c r = FlockComplete.Replay.pushes (fun j => coin (l.size + (3 + j)) 0) (st.m - 6))
    (hα : coin (l.size + (3 + (st.m - 6))) 0 = c.rest.lcAlpha r)
    (hβ : coin (l.size + (4 + (st.m - 6))) 0 = c.rest.lcBeta r)
    (hτ : tsF Tz stm c r =
      FlockComplete.Replay.pushes (fun j => coin (l.size + (5 + (st.m - 6) + j)) 0) (st.kLog - 6))
    (hr' : coin (l.size + (5 + (st.m - 6) + (st.kLog - 6))) 0 = c.rest.rPrime r)
    (hfold : ∀ combv, st.fold.folded (c.rest.lcAlpha r) (c.rest.lcBeta r) (skipWeights (zcF Tz stm c r))
      ((rhoF Tz stm c r).extract 0 (stm.c.kLog - 6)) (tsF Tz stm c r) = .ok combv → combv = combvOf Tz stm c r)
    (hsw : ∀ zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut coin n) st.m st.digest (FlockComplete.flat cm0.cap) ic.root
          (Zk.PadLayout.new st.m st.kLog sch.k0) fun i => vals[i]! + pads.h[i]!).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut coin n) st.fold st.m (Zk.PadLayout.new st.m st.kLog sch.k0)
          (fun i => vals[i]! + pads.h[i]!) zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂) →
      ((FlockComplete.ringSwitchCut (FlockComplete.asksCut coin n)
          (fun k => Flock.Firewall.Finish.switchVec g (abcClaims st.m st.kLog zc lc ++ extras)[k]!)
          (abcClaims st.m st.kLog zc lc ++ extras)).run t₂).go l₂ = (.ok ((ss, T, sws), t₃), l₃) →
      ss[0]! = shArr sh (2 * r.val) ∧ ss[1]! = shArr sh (2 * r.val + 1))
    (hlen : (layOf Tz stm).len = nE + 4)
    {zp : ZerocheckProof} {zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃ lg zl rec t₄ l₄ rp tt lgc}
    (h₁ : ((FlockComplete.zerocheckCut (FlockComplete.asksCut coin n) st.m st.digest (FlockComplete.flat cm0.cap)
        ic.root (Zk.PadLayout.new st.m st.kLog sch.k0) fun i => vals[i]! + pads.h[i]!).run t₀).go l =
      (.ok ((zp, zc), t₁), l₁))
    (h₂ : ((FlockComplete.lincheckCut (FlockComplete.asksCut coin n) st.fold st.m
        (Zk.PadLayout.new st.m st.kLog sch.k0) (fun i => vals[i]! + pads.h[i]!) zc).run t₁).go l₁ =
      (.ok ((lp, lc), t₂), l₂))
    (h₃ : ((FlockComplete.ringSwitchCut (FlockComplete.asksCut coin n)
        (fun k => Flock.Firewall.Finish.switchVec g (abcClaims st.m st.kLog zc lc ++ extras)[k]!)
        (abcClaims st.m st.kLog zc lc ++ extras)).run t₂).go l₂ = (.ok ((ss, T, sws), t₃), l₃))
    (h₄ : ((FlockComplete.ligeritoPZk (FlockComplete.asksCut coin n) (Flock.Hashes.ofSha512 Flock.Sha512.hash)
        Flock.MerkleScheme.hm96Sha512 sch T cm0 g (FlockComplete.basisVals sws (st.m - 7))
        (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h rep tPad extra lanePad salt (fun _ _ => 0)).run t₃).go l₃ =
      (.ok ((lg, zl, rec), t₄), l₄))
    (hrp : (Zk.replay st (Zk.PadLayout.new st.m st.kLog sch.k0)
        { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp, ringSwitches := ss,
          ligerito := lg } ic.root).run (Flock.Transcript.init t₄.rounds dom) = .ok (rp, tt))
    (hlgc : Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rec lg zl = .ok lgc)
    (hL : _root_.Claims.ZKMal.Masked.LaneCoins Tz stm c r rec) :
    lgc.size = 2 ∧ (∀ k : Fin 2, (lgc[k.val]!).2.a.size = nE + 4) ∧
      ∀ k (hk : k < 7) (i : Fin (nE + 4)),
        (ofZk (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! ss[0]!)),
          ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)[k]!.2 :
          ZkSession.Lin 𝔽 (nE + 4)).coef i = (rowsOf Tz stm c r v sh T' ρσ (Fin.castAdd 3 ⟨k, by omega⟩)).coef i := by
  have hrp' : Zk.replay st (layOf Tz stm)
      { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp, ringSwitches := ss,
        ligerito := lg } ic.root (Flock.Transcript.init t₄.rounds dom) = .ok (rp, tt) := by
    rw [hlay]
    exact hrp
  have hlgc' : Zk.ligeritoConstraints (layOf Tz stm) rec lg zl = .ok lgc := by
    rw [hlay]
    exact hlgc
  obtain ⟨hsz2, hrow⟩ := _root_.Claims.ZKMal.Masked.lanesA Tz stm c r v sh T' hlen lg rec zl hL.ood hL.lane hL.rho
    hlgc'
  refine ⟨hsz2, fun k => (hrow k).1, fun k hk i => ?_⟩
  by_cases hk5 : k < 5
  · exact congrArg (fun ℓ => ZkSession.Lin.coef ℓ i) (tauFw_at_replay Tz stm c r v sh T' ρσ coin n st sch tPad rep
      dom g cm0 extras pads ic vals extra lanePad salt ht₀ hm hk6 hSm hSk hkL hlay hmk hz hr hρ hα hβ hτ hr' hfold hsw
      h₁ h₂ h₃ h₄ hrp lgc k hk5)
  obtain ⟨k', rfl⟩ : ∃ k' : Fin 2, k = 5 + k'.val := ⟨⟨k - 5, by omega⟩, by simp only; omega⟩
  have h3 := ZkExec.replay_cons_size _ _ _ _ hrp'
  rw [ZkExec.append_get!_right _ _ (j := k'.val) (by rw [Array.size_append, h3]; rfl) (by rw [hsz2]; exact k'.isLt)]
  refine ((hrow k').2 i).trans ?_
  fin_cases k' <;> rfl

/-- **The firewall's seven rows are `rowsOf`'s**: where `repFwC`, answered by `asksCut` from a log `l` and no rounds,
returns, its constraints pass on the pads, and the replay's three, the two ring-switch rows and level 0's two lane rows
are `rowsOf`'s first seven, constants included, wherever the coins from `l.size` on are the model's (`hz`, …, `hr'`),
the masked values `vals + h` are the view's below `zp 64` (`hmk`), the fold folds as the model's (`hfold`), the ring
switch's vectors at the `ab` and `c` claims are the view's `ŝ` (`hsw`) and the Ligerito run's proof, record and
messages are read at the view (`hlane`, `LaneAt`). -/
theorem tauFw {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (stm : Stmt) (c : CoinsZK (RestOf Tz stm)) (r : Fin 2) {nE : ℕ}
    (v : Fin nE → 𝔽) {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) (T' : 𝕂) (ρσ : 𝔽 × 𝔽)
    (coin : ℕ → ℕ → Flock.F128) (n : ℕ) (st : Flock.Setup) (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (g : Array Flock.F128) (cm0 : FlockComplete.Commit) (extras : Array Flock.Claim) (pads : FlockComplete.PadsZk)
    (ic : FlockComplete.InnerCommit) (vals : Array Flock.F128) (extra lanePad : ℕ → ℕ → Flock.F128)
    (salt : ℕ → ℕ → ByteArray) {t₀ : Flock.Transcript} (ht₀ : t₀.rounds = #[]) {l : Array _root_.Claims.Msg}
    (hm : 14 ≤ st.m) (hk6 : 6 ≤ st.kLog) (hSm : st.m = stm.m) (hSk : st.kLog = stm.c.kLog)
    (hkL : st.fold.kLog = st.kLog) (hlay : layOf Tz stm = Zk.PadLayout.new st.m st.kLog sch.k0)
    (hmk : ∀ k < (layOf Tz stm).zp 64, (msgAt v k).val = vals[k]! + pads.h[k]!)
    (hz : coin (l.size + 2) 0 = zcF Tz stm c r)
    (hr : pinned ++ (Array.range (st.m - 13)).map (coin (l.size + 1)) = rF Tz stm c r)
    (hρ : rhoF Tz stm c r = FlockComplete.Replay.pushes (fun j => coin (l.size + (3 + j)) 0) (st.m - 6))
    (hα : coin (l.size + (3 + (st.m - 6))) 0 = c.rest.lcAlpha r)
    (hβ : coin (l.size + (4 + (st.m - 6))) 0 = c.rest.lcBeta r)
    (hτ : tsF Tz stm c r =
      FlockComplete.Replay.pushes (fun j => coin (l.size + (5 + (st.m - 6) + j)) 0) (st.kLog - 6))
    (hr' : coin (l.size + (5 + (st.m - 6) + (st.kLog - 6))) 0 = c.rest.rPrime r)
    (hfold : ∀ combv, st.fold.folded (c.rest.lcAlpha r) (c.rest.lcBeta r) (skipWeights (zcF Tz stm c r))
      ((rhoF Tz stm c r).extract 0 (stm.c.kLog - 6)) (tsF Tz stm c r) = .ok combv → combv = combvOf Tz stm c r)
    (hsw : ∀ zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut coin n) st.m st.digest (FlockComplete.flat cm0.cap) ic.root
          (Zk.PadLayout.new st.m st.kLog sch.k0) fun i => vals[i]! + pads.h[i]!).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut coin n) st.fold st.m (Zk.PadLayout.new st.m st.kLog sch.k0)
          (fun i => vals[i]! + pads.h[i]!) zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂) →
      ((FlockComplete.ringSwitchCut (FlockComplete.asksCut coin n)
          (fun k => Flock.Firewall.Finish.switchVec g (abcClaims st.m st.kLog zc lc ++ extras)[k]!)
          (abcClaims st.m st.kLog zc lc ++ extras)).run t₂).go l₂ = (.ok ((ss, T, sws), t₃), l₃) →
      ss[0]! = shArr sh (2 * r.val) ∧ ss[1]! = shArr sh (2 * r.val + 1))
    (hlen : (layOf Tz stm).len = nE + 4)
    (hlane : ∀ zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃ lg zl rec t₄ l₄,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut coin n) st.m st.digest (FlockComplete.flat cm0.cap) ic.root
          (Zk.PadLayout.new st.m st.kLog sch.k0) fun i => vals[i]! + pads.h[i]!).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut coin n) st.fold st.m (Zk.PadLayout.new st.m st.kLog sch.k0)
          (fun i => vals[i]! + pads.h[i]!) zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂) →
      ((FlockComplete.ringSwitchCut (FlockComplete.asksCut coin n)
          (fun k => Flock.Firewall.Finish.switchVec g (abcClaims st.m st.kLog zc lc ++ extras)[k]!)
          (abcClaims st.m st.kLog zc lc ++ extras)).run t₂).go l₂ = (.ok ((ss, T, sws), t₃), l₃) →
      ((FlockComplete.ligeritoPZk (FlockComplete.asksCut coin n) (Flock.Hashes.ofSha512 Flock.Sha512.hash)
          Flock.MerkleScheme.hm96Sha512 sch T cm0 g (FlockComplete.basisVals sws (st.m - 7))
          (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h rep tPad extra lanePad salt (fun _ _ => 0)).run t₃).go l₃ =
        (.ok ((lg, zl, rec), t₄), l₄) →
      _root_.Claims.ZKMal.Masked.LaneAt Tz stm c r v sh T' lg rec zl)
    {p : Zk.ZkProof} {t : Flock.Transcript} {l' : Array _root_.Claims.Msg}
    (h : ((Flock.Firewall.Finish.repFwC (FlockComplete.asksCut coin n) (Flock.Hashes.ofSha512 Flock.Sha512.hash) st sch
      tPad rep dom g cm0 extras pads ic vals extra lanePad salt).run t₀).go l = (.ok (p, t), l')) :
    ∃ (rp : Zk.Replayed) (ss : Array (Array Flock.F128)) (lgc : Array (String × Zk.Lin)),
      Flock.Firewall.Finish.failingOn (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]!
          ss[0]!)), ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc) pads.h = #[] ∧
      ∀ k (hk : k < 7), (ofZk (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]!
          ss[0]!)), ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)[k]!.2 :
          ZkSession.Lin 𝔽 (nE + 4)) = rowsOf Tz stm c r v sh T' ρσ (Fin.castAdd 3 ⟨k, by omega⟩) := by
  dsimp only [Flock.Firewall.Finish.repFwC] at h
  obtain ⟨⟨zp, zc⟩, t₁, l₁, h₁, h⟩ := Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lp, lc⟩, t₂, l₂, h₂, h⟩ := Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨ss, T, sws⟩, t₃, l₃, h₃, h⟩ := Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lg, zl, rec⟩, t₄, l₄, h₄, h⟩ := Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨tr', t₅, l₅, h₅, hg⟩ := Flock.Firewall.Finish.run_bind_ok h
  rw [Flock.Firewall.Finish.get_run_go] at h₅
  simp only [Prod.mk.injEq, Except.ok.injEq] at h₅
  obtain ⟨⟨htr, ht₅⟩, hl₅⟩ := h₅
  subst tr' t₅ l₅
  try dsimp only at hg
  obtain ⟨⟨rp, tt⟩, t₆, l₆, h₆, hr₆⟩ := Flock.Firewall.Finish.run_bind_ok hg
  obtain ⟨hrp, ht₆, hl₆⟩ := Flock.Firewall.Finish.liftV_run_ok h₆
  subst t₆ l₆
  try dsimp only at hr₆
  obtain ⟨lgc, t₇, l₇, h₇, hc⟩ := Flock.Firewall.Finish.run_bind_ok hr₆
  obtain ⟨hlgc, ht₇, hl₇⟩ := Flock.Firewall.Finish.liftV_run_ok h₇
  subst t₇ l₇
  try dsimp only at hc
  split at hc
  · obtain ⟨_, _, _, h', _⟩ := Flock.Firewall.Finish.run_bind_ok hc
    rw [Flock.Firewall.Finish.throw_run_go] at h'
    cases h'
  rename_i hf
  exact ⟨rp, ss, lgc, by simpa using hf, tauFw_at Tz stm c r v sh T' ρσ coin n st sch tPad rep dom g cm0 extras pads ic
    vals extra lanePad salt ht₀ hm hk6 hSm hSk hkL hlay hmk hz hr hρ hα hβ hτ hr' hfold hsw hlen hlane h₁ h₂ h₃ h₄ hrp
    hlgc⟩

end FlockSoundness.Discharge.Composed
