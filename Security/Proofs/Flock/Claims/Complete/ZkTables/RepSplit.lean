import Proofs.Flock.Claims.Complete.Zk.RingSwitch
import Proofs.Flock.Claims.Complete.Zk.Replay

/-!
# A returning masked rep, stage by stage

`repPZk_split`: a masked rep (`repPZk`) that returns, run in the pure session from the bodies `l`, ran its zerocheck,
lincheck and ring switch in closed form at the replies' coins (`zerocheckPZk_go`, `lincheckPZk_go`, `ringSwitchP_go`),
its rank check between them keeping the transcript and the sends (`rankCheckP_go`), ending at `tPiop` with
`|l| + piopSends` bodies sent; then its masked Ligerito from there, on the ring switch's `T` and
batched basis; then, on the transcript the Ligerito left, the verifier's own replay of its rounds, level 0's lane
constraints of the Ligerito's record, and its inner proof on every constraint, which returned the rep's inner proof
and its final transcript and bodies. The rep's proof is the stages' outputs.
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockComplete.Replay

section Defs

variable (resp : ℕ → Claims.Msg) (st : Setup) (tr : Trace) (L : ℕ)

/-- The masked rep's zerocheck output from `L` bodies (the plain prover's). -/
def zcR : ZcOut := Zc.out st.m (Zc.coinsAt resp L) tr

/-- Its lincheck's coins, from the bodies after the zerocheck's. -/
def lcCoinsR : Lc.Challenges := Lc.coinsAt resp (L + 3 + (st.m - 6)) (st.fold.kLog - 6)

/-- Its lincheck output. -/
def lcR : LcOut := Lc.out st.fold st.m tr (zcR resp st tr L) (lcCoinsR resp st L)

/-- The claims it ring-switches: the zerocheck's and lincheck's two, then the extra claims. -/
def claimsR (extras : Array Claim) : Array Claim :=
  abcClaims st.m st.kLog (zcR resp st tr L) (lcR resp st tr L) ++ extras

/-- Its ring switch's coins, from the bodies after the lincheck's. -/
def rsCoinsR (extras : Array Claim) : Rs.Challenges :=
  Rs.coinsAt resp (L + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6)) (2 + extras.size)

/-- A masked rep's sends before its Ligerito. -/
def piopSends (extras : Array Claim) : ℕ := 3 + (st.m - 6) + 3 + (st.fold.kLog - 6) + (2 + extras.size + 1)

/-- **The masked rep's transcript after its ring switch**, from `t`, with the lincheck's fold `full`. -/
def tPiop (lay : Zk.PadLayout) (hp : Array F128) (digest cap root : ByteArray) (full : Array F128)
    (extras : Array Claim) (t : Transcript) : Transcript :=
  Rs.tEnd tr (claimsR resp st tr L extras) (rsCoinsR resp st L extras)
    (LcZ.tEnd lay hp st.fold st.m tr (zcR resp st tr L) full (lcCoinsR resp st L)
      (ZcZ.tEnd lay hp st.m digest cap root (Zc.coinsAt resp L) tr t))

/-- **The masked rep's proof before its Ligerito**, its Ligerito proof `lg`. -/
def proofR (sch : Schedule) (lay : Zk.PadLayout) (hp : Array F128) (cm0 : Commit) (full : Array F128)
    (extras : Array Claim) (lg : LigeritoProof) : Proof :=
  { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := ZcZ.proof lay hp st.m (Zc.coinsAt resp L) tr,
    lincheck := LcZ.proof lay hp st.fold st.m tr (zcR resp st tr L) full (lcCoinsR resp st L),
    ringSwitches := Rs.proof tr (claimsR resp st tr L extras), ligerito := lg }

end Defs

theorem get_go (t : Transcript) (l : Array Claims.Msg) :
    ((get : TMW PS Transcript).run t).go l = (.ok (t, t), l) := rfl

theorem liftV_bind_ok {α β : Type} {v : V α} {f : α → TMW PS β} {t : Transcript} {l l' : Array Claims.Msg}
    {b : β} {t' : Transcript}
    (h : ((StateT.lift (monadLift v : PS α) >>= f).run t).go l = (.ok (b, t'), l')) :
    ∃ a, v = .ok a ∧ ((f a).run t).go l = (.ok (b, t'), l') := by
  rw [StateT.run_bind] at h
  obtain ⟨⟨a, t₁⟩, l₁, e1, e2⟩ := JsonRT.go_bind_ok h
  change ((monadLift v : PS α) >>= fun a => Pure.pure (a, t)).go l = _ at e1
  obtain ⟨c, l₂, e3, e4⟩ := JsonRT.go_bind_ok e1
  rw [JsonRT.monadLift_go] at e3
  simp only [Prod.mk.injEq] at e3
  obtain ⟨hv, rfl⟩ := e3
  simp only [JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at e4
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := e4
  exact ⟨c, hv, e2⟩

/-- **A returning masked rep, stage by stage.** -/
theorem repPZk_split (resp : ℕ → Claims.Msg) (hsh : ByteArray → ByteArray) (st : Setup) (sch : Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (tr : Trace) (tb : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m) (t : Transcript)
    (l : Array Claims.Msg) {P : Zk.ZkProof} {t' : Transcript} {l' : Array Claims.Msg}
    (h : ((repPZk (asksS resp) (Hashes.ofSha512 hsh) st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt
      nonce prior words).run t).go l = (.ok (P, t'), l')) :
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
        (proofR resp st tr l.size sch (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h cm0 full extras lg) ic.root).run
          (Transcript.init tI.rounds dom) = .ok (rp, x) ∧
      Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rec lg zl = .ok lgc ∧
      ((innerP (asksS resp) (Hashes.ofSha512 hsh) (Zk.PadLayout.new st.m st.kLog sch.k0) pads ic
        (ZcZ.proof (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h st.m (Zc.coinsAt resp l.size) tr).finalC
        (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]!
            (Rs.proof tr (claimsR resp st tr l.size extras))[0]!)),
          ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]!
            (Rs.proof tr (claimsR resp st tr l.size extras))[1]!))] ++ lgc)).run tI).go lI =
        (.ok (P.inner, t'), l') ∧
      P.proof = proofR resp st tr l.size sch (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h cm0 full extras lg ∧
      P.level0 = zl := by
  obtain ⟨l₁, hl₁, hz⟩ := zerocheckPZk_go resp hm st.digest (flat cm0.cap) ic.root
    (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h tr t l
  unfold repPZk at h
  rw [bind_go hz] at h
  dsimp only at h
  have h' := h
  rw [StateT.run_bind] at h'
  obtain ⟨_, _, h₁, -⟩ := JsonRT.go_bind_ok h'
  obtain ⟨full, hfold⟩ := lincheckPZk_fold resp _ _ _ _ _ _ _ _ h₁
  obtain ⟨l₂, hl₂, hl⟩ := lincheckPZk_go resp st.fold st.m (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h tr
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
  dsimp only at h
  rw [StateT.run_bind] at h
  obtain ⟨⟨⟨lg, zl, rec⟩, tI⟩, lI, hlg, h⟩ := JsonRT.go_bind_ok h
  dsimp only at h
  rw [StateT.run_bind] at h
  obtain ⟨⟨tg, tg'⟩, lg', hget, h⟩ := JsonRT.go_bind_ok h
  rw [get_go] at hget
  simp only [Prod.mk.injEq, Except.ok.injEq] at hget
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := hget
  dsimp only at h
  obtain ⟨⟨rp, x⟩, hre, h⟩ := liftV_bind_ok h
  dsimp only at h
  obtain ⟨lgc, hlgc, h⟩ := liftV_bind_ok h
  rw [StateT.run_bind] at h
  obtain ⟨⟨inner, tE⟩, lE, hin, h⟩ := JsonRT.go_bind_ok h
  simp only [StateT.run_pure, JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at h
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
  have e₁ : l₁.size = l.size + 3 + (st.m - 6) := hl₁
  have e₂ : ∀ L, (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
      (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) (Lc.coinsAt resp L (st.fold.kLog - 6))) ++
        extras).size = 2 + extras.size := fun _ => by simp [abcClaims]
  rw [e₁] at hl₂ hlg hre hin hfold ⊢
  refine ⟨full, l₃, lg, zl, rec, tI, lI, rp, x, lgc, hfold, ?_, ?_, hre, hlgc, ?_, rfl, rfl⟩
  · rw [hl₃, e₂, hl₂]
    unfold piopSends
    omega
  · rw [show l₂.size = l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6) by omega, e₂] at hlg
    exact hlg
  · exact hin

end FlockComplete.ZkTables
