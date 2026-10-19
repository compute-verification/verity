import Proofs.Flock.Claims.ZK.SplitInner
import Proofs.Flock.Claims.ZK.SimSession

/-!
# The shared phases end at the rep's `ε` slot (`InnerSlot`)

`innerSlot : InnerSlot`: where the phases hybrids 2 and 3 share (`innerPreE`) return from a rep's first round
(`repBase`), they have sent the rep's PIOP rounds (`piopRoundsZk`) and Ligerito's `k₀ + 7 + 11 (levels − 1)`
(`innerOff 0`), which is `repSlot`'s count.

* **Answered by `asksS`** (`piopPZk_sends`, `innerPre_sends`): the PIOP sends `piopZk`, from ZkTables' step counts
  (`zerocheckPZk_go`, `lincheckPZk_go`, `ringSwitchP_go`; the rank check sends nothing), and Ligerito its
  `ligeritoPZk_exact` count; the replay and `ligeritoConstraints` send nothing.
* **Answered by `asksCut`** (`piopPZk_cut`, `innerPre_cut`): a run that returns is the run answered by `asksS` at the
  opened coins' bytes (`respOf`), as for a whole rep (`repPZk_cut`).
-/

namespace FlockComplete

open Flock Transcript Replay Replay.Lg

/-- **A masked PIOP answered by `asksS` that returns sent `piopZk`**: the zerocheck's `3 + (m − 6)`, the lincheck's
`3 + (k_log − 6)` and the ring switch's `claims + 1`. -/
theorem piopPZk_sends (resp : ℕ → Claims.Msg) (st : Setup) (lay : Zk.PadLayout) (hv : Array F128) (tr : Trace)
    (cap root : ByteArray) (extras : Array Claim) (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m)
    {t t' : Transcript} {l l' : Array Claims.Msg}
    {a : ZerocheckProof × LincheckProof × Array (Array F128) × F128 × Array Switched}
    (h : ((piopPZk (asksS resp) st lay hv tr cap root extras prior words).run t).go l = (.ok (a, t'), l')) :
    l'.size = l.size + piopZk st extras := by
  obtain ⟨l₁, hl₁, hz⟩ := zerocheckPZk_go resp hm st.digest cap root lay hv tr t l
  unfold piopPZk at h
  rw [bind_go hz] at h
  dsimp only at h
  have h' := h
  rw [StateT.run_bind] at h'
  obtain ⟨_, _, h₁, -⟩ := JsonRT.go_bind_ok h'
  obtain ⟨full, hfold⟩ := lincheckPZk_fold resp _ _ _ _ _ _ _ _ h₁
  obtain ⟨l₂, hl₂, hl⟩ := lincheckPZk_go resp st.fold st.m lay hv tr
    (Zc.out st.m (Zc.coinsAt resp l.size) tr) full _ l₁ hfold
  rw [bind_go hl] at h
  dsimp only at h
  rw [StateT.run_bind] at h
  obtain ⟨⟨u, t₃⟩, l₃', hk, h⟩ := JsonRT.go_bind_ok h
  obtain ⟨ht₃, hl₃⟩ := rankCheckP_go hk
  subst t₃ l₃'
  obtain ⟨l₃, hl₃, hrs⟩ := ringSwitchP_go resp tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) (Lc.coinsAt resp l₁.size (st.fold.kLog - 6))) ++
      extras) (by simp [abcClaims]) _ l₂
  rw [bind_go hrs] at h
  simp only [StateT.run_pure, JsonRT.go_pure, Prod.mk.injEq] at h
  have hsz : (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) (Lc.coinsAt resp l₁.size (st.fold.kLog - 6))) ++
      extras).size = 2 + extras.size := by simp [abcClaims, Nat.add_comm]
  rw [← h.2]
  unfold piopZk
  omega

/-- **The shared phases answered by `asksS` that return sent `piopZk` and Ligerito's `k₀ + 7 + 11 R`**; the replay and
`ligeritoConstraints` send nothing. -/
theorem innerPre_sends (resp : ℕ → Claims.Msg) (hsh : ByteArray → ByteArray) (st : Setup) (sch : Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (tr : Trace) (tb : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m) (hn : sch.n ≠ 0)
    (hk0 : 1 ≤ sch.k0) (hL : 2 ≤ sch.levels.size)
    (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0) {t t' : Transcript} {l l' : Array Claims.Msg}
    {o : F128 × Proof × Zk.Level0Msgs × Array (String × Zk.Lin)}
    (h : ((SimInner.innerPre (asksS resp) (Hashes.ofSha512 hsh) st sch tPad rep dom tr tb cm0 extras pads ic extra
      lanePad salt nonce prior words).run t).go l = (.ok (o, t'), l')) :
    l'.size = l.size + piopZk st extras + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) := by
  unfold SimInner.innerPre at h
  rw [StateT.run_bind] at h
  obtain ⟨⟨⟨zp, lp, ss, T, sws⟩, t₁⟩, l₁, h1, h⟩ := JsonRT.go_bind_ok h
  have e1 := piopPZk_sends resp _ _ _ _ _ _ _ _ _ hm h1
  dsimp only at h
  rw [StateT.run_bind] at h
  obtain ⟨⟨⟨lg, zl, rec⟩, t₂⟩, l₂, h2, h⟩ := JsonRT.go_bind_ok h
  obtain ⟨hl₂, -⟩ := ligeritoPZk_exact resp hsh st.merkle sch _ cm0 _ _ _ _ rep tPad extra lanePad salt nonce hn
    hk0 hL hq h2
  dsimp only at h
  simp only [pure_bind] at h
  rw [bind_go (show ((get : TMW PS Transcript).run t₂).go l₂ = (.ok (t₂, t₂), l₂) from rfl)] at h
  obtain ⟨⟨rp, x⟩, -, h⟩ := liftV_bind_ok h
  dsimp only at h
  obtain ⟨lgc, -, h⟩ := liftV_bind_ok h
  simp only [StateT.run_pure, JsonRT.go_pure, Prod.mk.injEq] at h
  rw [← h.2, hl₂, e1]

end FlockComplete

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

open FlockComplete Flock.With

section Bridge

variable {coin : ℕ → ℕ → _root_.Flock.F128} {n : ℕ} {resp : ℕ → Msg}

/-- **A masked PIOP answered by `asksCut` that returns** is the PIOP answered by `asksS` at the same coins. -/
theorem piopPZk_cut (hr : ∀ k, coinOf (resp k) = coin k) (st : _root_.Flock.Setup) (lay : _root_.Flock.Zk.PadLayout)
    (h : Array _root_.Flock.F128) (tr : FlockComplete.Trace) (cap root : ByteArray) (extras : Array _root_.Flock.Claim)
    (prior : Array (List _root_.Flock.F128)) (words : List ℕ) :
    (cutS.st _root_.Flock.Transcript).R (piopPZk (asksCut coin n) st lay h tr cap root extras prior words)
      (piopPZk (asksS resp) st lay h tr cap root extras prior words) := by
  unfold piopPZk rankCheckP
  sim_walk [zerocheckPZk_cut hr, lincheckPZk_cut hr, ringSwitchP_cut hr, cutS_reflT, cutS_refl]

/-- **The shared phases answered by `asksCut` that return** are the phases answered by `asksS` at the same coins. -/
theorem innerPre_cut (hr : ∀ k, coinOf (resp k) = coin k) (hs : _root_.Flock.Hashes FlockComplete.PS)
    (st : _root_.Flock.Setup) (sch : _root_.Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (tr : FlockComplete.Trace)
    (t : Array _root_.Flock.F128) (cm0 : Commit) (extras : Array _root_.Flock.Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List _root_.Flock.F128)) (words : List ℕ) :
    (cutS.st _root_.Flock.Transcript).R
      (SimInner.innerPre (asksCut coin n) hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior
        words)
      (SimInner.innerPre (asksS resp) hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior
        words) := by
  unfold SimInner.innerPre
  sim_walk [piopPZk_cut hr, ligeritoPZk_cut hr, cutS_reflT, cutS_refl]

end Bridge

/-- **The shared phases end at the rep's `ε` slot**: from the rep's first round, `innerPreE` that returns has sent the
rep's PIOP rounds and Ligerito's, `repSlot`'s count at `innerOff 0`. -/
theorem innerSlot : InnerSlot := by
  rintro x w τ c n j r cons L ⟨hx, hw, hn, s, spec, hsx, dom, cm0, ic, extras, he, l, hl, o, t', l', hrun, -, rfl⟩
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hJ : jOf x = (stsOf x).size := by
    unfold jOf
    simp only [hx, ↓reduceIte]
  have hi : j.val < (stsOf x).size := hJ ▸ j.isLt
  have hst : (stsOf x)[j.val]! = stAt x j := by
    unfold stAt
    exact getElem!_pos _ _ _
  have hv : _root_.Flock.Zk.schedule (stsOf x)[j.val]!.m = .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [hst]
    exact (zkTab x j).hz
  have hk6 : 6 ≤ (stsOf x)[j.val]!.c.kLog := by
    rw [hst]
    exact (zkTab x j).wf.1.hk6
  have hm : 14 ≤ (stsOf x)[j.val]!.m := by
    have := (hall j.val hi).1
    rw [getElem!_pos (stsOf x) j.val hi]
    omega
  have htab : (FlockDeploy.tabsOf (stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (stsOf x)[j.val]! := by
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) j.val (by rw [FlockComplete.ZkTables.tabsOf_size]; exact hi),
      getElem!_pos (stsOf x) j.val hi]
    simp [FlockDeploy.tabsOf]
  unfold innerPreE ioE at hrun
  have e := innerPre_cut (coin := fun k i => slotAt c k i) (n := n) (resp := respOf c) (fun k => coinOf_respOf c k)
    hsE _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ (_root_.Flock.Transcript.init #[] dom) l _ l' hrun
  rw [htab] at e he
  obtain ⟨-, -, hsm, hk0, -, hdim, hlv, -, -⟩ := FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ hv
  obtain ⟨hn7, hL2, -, -, -, -, -⟩ := hdim
  have hcount := FlockComplete.innerPre_sends _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hm (by rw [hn7, hsm]; omega) hk0
    hL2 (fun i hi => (hlv i hi).q_ne_zero) e
  have hx' : extras.size = spec.points * (stsOf x)[j.val]!.regions.size := by
    rw [JsonRT.extraClaims_size he]
    simp [ptsE, Nat.mul_comm]
  have e6 : (_root_.Flock.Setup.ofCircuit (stsOf x)[j.val]!).fold.kLog = (stsOf x)[j.val]!.c.kLog := rfl
  have e7 : (_root_.Flock.Setup.ofCircuit (stsOf x)[j.val]!).m = (stsOf x)[j.val]!.m := rfl
  unfold innerSlotE repSlot
  rw [hsx]
  dsimp only
  rw [show (2 * j.val + r.val) / 2 = j.val by have := r.isLt; omega, hv]
  dsimp only
  rw [hcount, hl, repBase_eq hsx]
  unfold piopZk FlockComplete.piopRoundsZk innerOff
  rw [e6, e7, hx']
  omega

end Masked

end Claims.ZKMal
