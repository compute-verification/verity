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

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT FlockComplete.Replay FlockSoundness.Refine

/-- **A table and its honest trace, as the replay's algebra reads them**: `TableT` without the readiness check's
`RegionsFit`, which `TableT.model` and `replay_holds` never read. -/
def TableM (st : Flock.Stmt) (tr : Trace) : Prop :=
  25 ≤ st.m ∧ st.m ≤ 35 ∧ st.merkle = MerkleScheme.hm96Sha512 ∧
    ∃ (p : Readable.Parsed) (w : Claims.Flock.Wit), p.sts = #[st] ∧
      Claims.Flock.RelS (Readable.toReadable Readable.relation p) w ∧ tr = traceOf st w.hid.toArray 0 ∧ StmtT st

/-- `TableT.model` at `TableM`. -/
theorem TableM.model {st : Flock.Stmt} {tr : Trace} (hT : TableM st tr) :
    ∃ (w : Claims.Flock.Wit) (hwf : StmtWF st), RegionsWF st ∧
      (∀ a b : Fin (2 ^ st.c.kLog), FlockLevel3.placedA st a b = matA st a.val b.val ∧
        FlockLevel3.placedB st a b = matB st a.val b.val) ∧
      (stmtOf st hwf (regionsOf st)).Satisfies (Readable.zOf 1 w 0 st.m) ∧ tr = traceOf st w.hid.toArray 0 ∧
      (∀ x < 2 ^ st.m, (tr.az[x]! && tr.bz[x]!) = tr.z[x]!) ∧ ∀ r ∈ st.c.ranges.toList, SlotOk st.c r := by
  obtain ⟨-, -, -, p, w, hpe, hR, htr, ⟨hwf, hrw, hmat⟩, hslot⟩ := hT
  obtain ⟨hf, hb⟩ := holds_only hpe hR
  have h0 : 0 < p.sts.size := by simp [hpe]
  obtain ⟨hf', hsat⟩ := (Readable.relS_relation p w).1 hR 0 h0
  have e0 : p.sts[0]! = st := by simp [hpe]
  have e1 : p.sts.size = 1 := by simp [hpe]
  revert hsat hf'
  rw [e0, e1]
  intro hf' hsat
  obtain ⟨h1, h2, h3⟩ := hsat
  have hA : ((Readable.Tab.ofStmt st).toModel hf'.2.2).A₀ = (stmtOf st hwf (regionsOf st)).A₀ :=
    funext fun a => funext fun b => (hmat a b).1
  have hB : ((Readable.Tab.ofStmt st).toModel hf'.2.2).B₀ = (stmtOf st hwf (regionsOf st)).B₀ :=
    funext fun a => funext fun b => (hmat a b).2
  refine ⟨w, hwf, hrw, hmat, ⟨fun o j => ?_, h2, h3⟩, htr, fun x hx => ?_, hslot⟩
  · have := h1 o j
    rw [hA, hB] at this
    exact this
  · subst htr
    exact traceOf_r1cs hf hb hx

/-- `replay_holds` at `TableM`. -/
theorem replay_holdsM {st : Flock.Stmt} {tr : Trace} (hT : TableM st tr) (lay : Zk.PadLayout) (h : Array F128)
    (hsz : h.size = lay.len) (hab : h[lay.hab]! = h[lay.fa]! * h[lay.fb]!) (hn : st.m - 6 ≤ lay.nMlv)
    (hn' : st.c.kLog - 6 ≤ lay.nLc) (zcs : Zc.Coins) (lcs : Lc.Coins) (full : Array F128)
    (hc1 : ∀ j < st.m - 13, zcs.c1 j ≠ 1)
    (hfold : st.circuitFold.fold (lcs.ca 0) (skipWeights (Zc.out st.m zcs tr).z)
      ((Zc.out st.m zcs tr).rho.extract 0 (st.c.kLog - 6)) = .ok full)
    (pts : Array (Array F128)) (hpts : ∀ a ∈ pts, a.size = st.mPts) (e : Array Claim)
    (he : st.extraClaims pts = .ok e) (p : Proof) (root : ByteArray)
    (hpz : p.zerocheck = ZcZ.proof lay h st.m zcs tr)
    (hpl : p.lincheck = LcZ.proof lay h st.circuitFold st.m tr (Zc.out st.m zcs tr) full lcs)
    (digest cap padsRoot : ByteArray) (t₀ t : Transcript) (hk₀ : t₀.k = t₀.rounds.size) (ht : t.k = t₀.k)
    (hR : Agrees t.rounds (LcZ.tEnd lay h st.circuitFold st.m tr (Zc.out st.m zcs tr) full lcs
      (ZcZ.tEnd lay h st.m digest cap padsRoot zcs tr t₀)).rounds 0)
    {rp : Zk.Replayed} {t' : Transcript} (hrun : Zk.replay (Setup.ofCircuit st) lay p root t = .ok (rp, t')) :
    ConsHold (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]!
        (Rs.proof tr (abcClaims st.m st.c.kLog (Zc.out st.m zcs tr)
          (Lc.out st.circuitFold st.m tr (Zc.out st.m zcs tr) lcs) ++ e))[0]!)),
      ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]!
        (Rs.proof tr (abcClaims st.m st.c.kLog (Zc.out st.m zcs tr)
          (Lc.out st.circuitFold st.m tr (Zc.out st.m zcs tr) lcs) ++ e))[1]!))]) h := by
  have h25 := hT.1
  obtain ⟨w, hwf, hrw, hmat, hsat, htr, hR1, -⟩ := hT.model
  obtain ⟨hz, ha, hb⟩ := trace_bits st hwf hmat w
  rw [← htr] at hz ha hb
  have hsw := claims_switch st hwf hrw (by omega) _ hsat _ hz zcs lcs pts (fun a ha => hpts a (by simpa using ha))
  rw [he] at hsw
  obtain ⟨comb, hcomb, hcsz, hlc⟩ := Lc.lcAlgebra_of (stmtOf st hwf (regionsOf st)) st.circuitFold
    (stmtOf_fold st hwf _) (fun α β lw ρ ts v h => ⟨_, FlockLevel3.folded_ok st α β lw ρ ts v h⟩) st.m rfl _ hsat.2.1
    _ hz ha hb zcs lcs full hfold
  have hcs : (abcClaims st.m st.c.kLog (Zc.out st.m zcs tr)
      (Lc.out st.circuitFold st.m tr (Zc.out st.m zcs tr) lcs) ++ e).size = 2 + e.size := by simp [abcClaims]
  refine replay_consHold_agrees (Setup.ofCircuit st) lay h p root (show 14 ≤ st.m by omega) hrun hsz st.circuitFold
    _ zcs lcs full rfl rfl hpz hpl hn hn' hab digest cap padsRoot t₀ hk₀ ht hR
    (Zc.fold_final (show 14 ≤ st.m by omega) zcs _ hR1 hc1) ⟨comb, hcomb, hcsz, hlc⟩ _ _ ?_ ?_
  · rw [Rs.proof_get 0 (by omega), abc_get0]
    exact hsw _ (abClaim_mem _ _ _ _ _)
  · rw [Rs.proof_get 1 (by omega), abc_get1]
    exact hsw _ (cClaim_mem _ _ _ _ _)

/-- **The shared phases' constraints vanish at the honest pads**: at a table and its honest trace (`TableM`) and its
`--zk` schedule `v`, wherever the shared phases (`innerPre`), answered by `asksS resp` from the empty transcript, return,
at outer coins that are not `1` and extra claims at points of the table's `m_pts`, every constraint they hand the inner
proof (the replay's, the two ring-switch rows and the lanes') vanishes at the rep's pads (`replay_holdsM`,
`lanes_hold`). -/
theorem innerPre_holds (resp : ℕ → Claims.Msg) (hsh : ByteArray → ByteArray) {st : Flock.Stmt} {tr : Trace}
    (hT : TableM st tr) {v : Schedule × ℕ} (hv : Zk.schedule st.m = .ok v) (rep : ℕ) (dom : ByteArray)
    (cm0 : Commit) (pts : Array (Array F128)) (hpts : ∀ a ∈ pts, a.size = st.mPts) (e : Array Claim)
    (he : st.extraClaims pts = .ok e) (tape : TapeZk) (i : ℕ) (ic : InnerCommit) (prior : Array (List F128))
    (words : List ℕ) (l : Array Claims.Msg) (hc1 : ∀ k < st.m - 13, coinOf (resp (l.size + 1)) k ≠ 1)
    {o : F128 × Proof × Zk.Level0Msgs × Array (String × Zk.Lin)} {t' : Transcript} {l' : Array Claims.Msg}
    (h : ((SimInner.innerPre (asksS resp) (Hashes.ofSha512 hsh) (Setup.ofCircuit st) v.1 v.2 rep dom tr
      (packed tr (Setup.ofCircuit st).m) cm0 e
      (padsOf tape i rep (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0)) ic
      (tape.extra i) (tape.lanePad i) (tape.salt i rep) (tape.nonce i rep) prior words).run
        (Transcript.init #[] dom)).go l = (.ok (o, t'), l')) :
    ConsHold o.2.2.2
      (padsOf tape i rep (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0)).h := by
  have h25 := hT.1
  have h35 := hT.2.1
  have hmerk : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 := hT.2.2.1
  obtain ⟨w, hwf, -, -, -, -, -, -⟩ := hT.model
  have hkm := hwf.hk
  have hm14 : 14 ≤ (Setup.ofCircuit st).m := by show 14 ≤ st.m; omega
  have hn : v.1.n = st.m - 7 := zkSchedule_n hv
  have hkn : v.1.k0 ≤ v.1.n := zkSchedule_k0n h25 h35 hv
  set lay := Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0 with hlay
  set pads := padsOf tape i rep lay with hpads
  obtain ⟨full, lL, lg, zl, rec, tI, lI, rp, x, lgc, hfold, -, hlig, hre, hlgc, ho⟩ :=
    innerPre_split resp hsh (Setup.ofCircuit st) v.1 v.2 rep dom tr (packed tr (Setup.ofCircuit st).m) cm0 e pads ic
      (tape.extra i) (tape.lanePad i) (tape.salt i rep) (tape.nonce i rep) prior words hm14 _ l h
  rw [ho]
  have htb : (packed tr (Setup.ofCircuit st).m).size = 2 ^ v.1.n := by simp [packed, hn]; rfl
  have hb : (basisVals (Rs.sws (claimsR resp (Setup.ofCircuit st) tr l.size e)
      (rsCoinsR resp (Setup.ofCircuit st) l.size e)) ((Setup.ofCircuit st).m - 7)).size = 2 ^ v.1.n := by
    simp [basisVals, hn]; rfl
  rw [hmerk] at hlig
  have hLF := L0.lanes resp hsh _ v _ cm0 _ _ lay pads.h rep (tape.extra i) (tape.lanePad i) (tape.salt i rep)
    (tape.nonce i rep) _ tI lL lI lg zl rec h25 h35 hv htb hb hlig
  have hx : ∀ c ∈ claimsR resp (Setup.ofCircuit st) tr l.size e, c.x.size = st.m - 6 :=
    claims_xsize' st hkm (by omega) tr (Zc.coinsAt resp l.size) (lcCoinsR resp (Setup.ofCircuit st) l.size) pts e he
  have hpre : Prefix (LcZ.tEnd lay pads.h st.circuitFold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
      (lcCoinsR resp (Setup.ofCircuit st) l.size) (ZcZ.tEnd lay pads.h st.m (Setup.ofCircuit st).digest (flat cm0.cap)
        ic.root (Zc.coinsAt resp l.size) tr (Transcript.init #[] dom))).rounds tI.rounds :=
    (Rs.prefix_loop (i := 0) _ (Nat.zero_le _)).trans
      (rec_pre (JsonRT.ligeritoPZk_rec resp hsh _ _ _ _ _ _ _ _ _ _ _ _ _ _) _ _ _ _ hlig)
  have hR := (Agrees.self tI.rounds 0).of_prefix hpre
  exact ConsHold.append
    (replay_holdsM hT lay pads.h (padsOf_size ..) (padsOf_hab ..) le_rfl le_rfl (Zc.coinsAt resp l.size)
      (lcCoinsR resp (Setup.ofCircuit st) l.size) full hc1 hfold pts hpts e he
      (proofR resp (Setup.ofCircuit st) tr l.size v.1 lay pads.h cm0 full e (eraseL lg)) ic.root rfl rfl
      (Setup.ofCircuit st).digest (flat cm0.cap) ic.root (Transcript.init #[] dom) (Transcript.init tI.rounds dom)
      rfl rfl hR hre)
    (lanes_hold hLF rfl rfl htb hb hkn (by omega) (Lg.rsSum_eq tr _ _ _ hx) (padsOf_size ..) hlgc)

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
