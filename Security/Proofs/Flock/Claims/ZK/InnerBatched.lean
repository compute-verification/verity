import Proofs.Flock.Claims.ZK.SplitInner
import Proofs.Flock.Claims.ZK.SimSession
import Proofs.Flock.Claims.Complete.ZkTables.RunsGo
import Proofs.Flock.Claims.Complete.ZkTables.MaskFree
import Proofs.Flock.Claims.Complete.Zk.Level0Premises

/-!
# `InnerBatched`: the batched constraint at the honest pads
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockComplete.Replay

/-- **A returning masked PIOP (`piopPZk`), in closed form**: zerocheck, lincheck and ring switch at the replies' coins,
the rank check keeping the transcript and the sends. -/
theorem piopPZk_split (resp : ℕ → Claims.Msg) (st : Setup) (lay : Zk.PadLayout) (hp : Array F128) (tr : Trace)
    (cap root : ByteArray) (extras : Array Claim) (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m)
    (t : Transcript) (l : Array Claims.Msg)
    {o : ZerocheckProof × LincheckProof × Array (Array F128) × F128 × Array Switched} {t' : Transcript}
    {l' : Array Claims.Msg}
    (h : ((piopPZk (asksS resp) st lay hp tr cap root extras prior words).run t).go l = (.ok (o, t'), l')) :
    ∃ full : Array F128,
      st.fold.fold (coinOf (resp (l.size + 3 + (st.m - 6))) 0) (skipWeights (zcR resp st tr l.size).z)
        ((zcR resp st tr l.size).rho.extract 0 (st.fold.kLog - 6)) = .ok full ∧
      l'.size = l.size + piopSends st extras ∧
      o = (ZcZ.proof lay hp st.m (Zc.coinsAt resp l.size) tr,
        LcZ.proof lay hp st.fold st.m tr (zcR resp st tr l.size) full (lcCoinsR resp st l.size),
        Rs.proof tr (claimsR resp st tr l.size extras),
        Rs.T tr (claimsR resp st tr l.size extras) (rsCoinsR resp st l.size extras),
        Rs.sws (claimsR resp st tr l.size extras) (rsCoinsR resp st l.size extras)) ∧
      t' = tPiop resp st tr l.size lay hp st.digest cap root full extras t := by
  obtain ⟨l₁, hl₁, hz⟩ := zerocheckPZk_go resp hm st.digest cap root lay hp tr t l
  unfold piopPZk at h
  rw [bind_go hz] at h
  dsimp only at h
  have h' := h
  rw [StateT.run_bind] at h'
  obtain ⟨_, _, h₁, -⟩ := JsonRT.go_bind_ok h'
  obtain ⟨full, hfold⟩ := lincheckPZk_fold resp _ _ _ _ _ _ _ _ h₁
  obtain ⟨l₂, hl₂, hl⟩ := lincheckPZk_go resp st.fold st.m lay hp tr
    (Zc.out st.m (Zc.coinsAt resp l.size) tr) full _ l₁ hfold
  rw [bind_go hl] at h
  dsimp only at h
  rw [StateT.run_bind] at h
  obtain ⟨⟨u, t₃⟩, l₃', hk, h⟩ := JsonRT.go_bind_ok h
  obtain ⟨ht₃, hl₃'⟩ := rankCheckP_go hk
  subst t₃ l₃'
  obtain ⟨l₃, hl₃, hrs⟩ := ringSwitchP_go resp tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) (Lc.coinsAt resp l₁.size (st.fold.kLog - 6))) ++
      extras) (by simp [abcClaims]) _ l₂
  rw [bind_go hrs] at h
  simp only [StateT.run_pure, JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at h
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
  have e₁ : l₁.size = l.size + 3 + (st.m - 6) := hl₁
  have e₂ : ∀ L, (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
      (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) (Lc.coinsAt resp L (st.fold.kLog - 6))) ++
        extras).size = 2 + extras.size := fun _ => by simp [abcClaims]
  rw [e₁] at hl₂ hfold ⊢
  refine ⟨full, hfold, ?_, ?_, ?_⟩
  · rw [hl₃, e₂, hl₂]
    unfold piopSends
    omega
  · rw [show l₂.size = l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6) by omega, e₂]
    rfl
  · rw [show l₂.size = l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6) by omega, e₂]
    rfl

/-- **The shared phases (`innerPre`) that return, stage by stage**: the masked PIOP in closed form (`piopPZk_split`),
then the masked Ligerito from `tPiop`, the verifier's replay of the proof with level 0's openings erased, and level 0's
lane constraints; the constraints they return are the replay's, the two ring-switch rows and the lanes'. -/
theorem innerPre_split (resp : ℕ → Claims.Msg) (hsh : ByteArray → ByteArray) (st : Setup) (sch : Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (tr : Trace) (tb : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m) (t : Transcript)
    (l : Array Claims.Msg) {o : F128 × Proof × Zk.Level0Msgs × Array (String × Zk.Lin)} {t' : Transcript}
    {l' : Array Claims.Msg}
    (h : ((SimInner.innerPre (asksS resp) (Hashes.ofSha512 hsh) st sch tPad rep dom tr tb cm0 extras pads ic extra
      lanePad salt nonce prior words).run t).go l = (.ok (o, t'), l')) :
    ∃ (full : Array F128) (lL : Array Claims.Msg) (lg : LigeritoProof) (zl : Zk.Level0Msgs) (rec : Zk.L0Rec)
      (tI : Transcript) (lI : Array Claims.Msg) (rp : Zk.Replayed) (x : Transcript) (lgc : Array (String × Zk.Lin)),
      st.fold.fold (coinOf (resp (l.size + 3 + (st.m - 6))) 0) (skipWeights (zcR resp st tr l.size).z)
        ((zcR resp st tr l.size).rho.extract 0 (st.fold.kLog - 6)) = .ok full ∧
      lL.size = l.size + piopSends st extras ∧
      ((ligeritoPZk (asksS resp) (Hashes.ofSha512 hsh) st.merkle sch
        (Rs.T tr (claimsR resp st tr l.size extras) (rsCoinsR resp st l.size extras)) cm0 tb
        (basisVals (Rs.sws (claimsR resp st tr l.size extras) (rsCoinsR resp st l.size extras)) (st.m - 7))
        (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h rep tPad extra lanePad salt nonce).run
          (tPiop resp st tr l.size (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h st.digest (flat cm0.cap) ic.root full
            extras t)).go lL = (.ok ((lg, zl, rec), tI), lI) ∧
      (Zk.replay st (Zk.PadLayout.new st.m st.kLog sch.k0)
        (proofR resp st tr l.size sch (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h cm0 full extras (eraseL lg))
          ic.root).run (Transcript.init tI.rounds dom) = .ok (rp, x) ∧
      Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rec lg zl = .ok lgc ∧
      o.2.2.2 = rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]!
            (Rs.proof tr (claimsR resp st tr l.size extras))[0]!)),
          ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]!
            (Rs.proof tr (claimsR resp st tr l.size extras))[1]!))] ++ lgc := by
  unfold SimInner.innerPre at h
  rw [StateT.run_bind] at h
  obtain ⟨⟨po, t₁⟩, l₁, hp, h⟩ := JsonRT.go_bind_ok h
  obtain ⟨full, hfold, hl₁, rfl, rfl⟩ := piopPZk_split resp st _ pads.h tr _ _ extras prior words hm t l hp
  dsimp only at h
  rw [StateT.run_bind] at h
  obtain ⟨⟨⟨lg, zl, rec⟩, tI⟩, lI, hlig, h⟩ := JsonRT.go_bind_ok h
  dsimp only at h
  rw [bind_go (pure_go _ _ _)] at h
  dsimp only at h
  rw [StateT.run_bind] at h
  obtain ⟨⟨tg, tg'⟩, lg₂, hget, h⟩ := JsonRT.go_bind_ok h
  rw [get_go] at hget
  simp only [Prod.mk.injEq, Except.ok.injEq] at hget
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := hget
  dsimp only at h
  obtain ⟨⟨rp, x⟩, hre, h⟩ := liftV_bind_ok h
  dsimp only at h
  obtain ⟨lgc, hlgc, h⟩ := liftV_bind_ok h
  simp only [StateT.run_pure, JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at h
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
  rw [Claims.ZKMal.Masked.ligeritoConstraints_eraseL] at hlgc
  exact ⟨full, l₁, lg, zl, rec, tI, lI, rp, x, lgc, hfold, hl₁, hlig, hre, hlgc, rfl⟩

end FlockComplete.ZkTables

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

open FlockComplete Flock.With

/-- **The phases hybrids 2 and 3 share, answered by `asksCut`, that return** are those answered by `asksS` at the same
coins. -/
theorem innerPre_cut {coin : ℕ → ℕ → _root_.Flock.F128} {n : ℕ} {resp : ℕ → Msg}
    (hr : ∀ k, coinOf (resp k) = coin k) (hs : _root_.Flock.Hashes FlockComplete.PS) (st : _root_.Flock.Setup)
    (sch : _root_.Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (tr : FlockComplete.Trace)
    (t : Array _root_.Flock.F128) (cm0 : Commit) (extras : Array _root_.Flock.Claim) (pads : PadsZk)
    (ic : InnerCommit) (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List _root_.Flock.F128)) (words : List ℕ) :
    (cutS.st _root_.Flock.Transcript).R
      (SimInner.innerPre (asksCut coin n) hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce
        prior words)
      (SimInner.innerPre (asksS resp) hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce
        prior words) := by
  unfold SimInner.innerPre piopPZk rankCheckP
  sim_walk [zerocheckPZk_cut hr, lincheckPZk_cut hr, ringSwitchP_cut hr, ligeritoPZk_cut hr, cutS_reflT, cutS_refl]

end Masked

end Claims.ZKMal
