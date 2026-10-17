import Proofs.Flock.Claims.Complete.ZkTables.RepSplit
import Proofs.Flock.Claims.Complete.ZkTables.Level0
import Proofs.Flock.Claims.Complete.ZkTables.Table
import Proofs.Flock.Claims.Complete.Replay.LigeritoBasis
import Proofs.Flock.Claims.Complete.Replay.LigeritoSum
import Proofs.Flock.Claims.Complete.Zk.InnerGo

/-!
# The verifier's view of a returning masked rep

* `ringSwitchWith_checks`: a verifier's ring switch that returns met every claim's check, so its claims meet their
  values on the proof's switched vectors. On the prover's ring switches that is each claim's identity, the extra
  claims' included, with no appeal to the relation (`rs_eq`): the verifier's ring switch, on the replay's claims and
  rounds carrying the prover's, returns the prover's `T` and switched claims, where the prover's stopped.
-/

namespace FlockComplete.Replay

open Flock Transcript FlockComplete

theorem check_true {c : Bool} {why : String} {t t' : Transcript} {a : Unit} (e : Flock.check c why t = .ok (a, t')) :
    c = true := by
  cases c
  · cases e
  · rfl

/-- **A verifier's ring switch that returns met every claim's check.** -/
theorem ringSwitchWith_checks (hs : Hashes V) (m : ℕ) (claims : Array Claim) (rs : Array (Array F128))
    (t : Transcript) :
    RetV (ringSwitchWith hs m claims rs) t fun _ _ =>
      ∀ k < claims.size, dot (FlockSoundness.Refine.wOf claims[k]!) rs[k]! = claims[k]!.value := by
  unfold ringSwitchWith
  simp only [Flock.With.liftTM_V]
  refine RetV.ok_then (Flock.RecAlgebra.label_eq _ _) ?_
  refine RetV.bind (RetV.mono (RetV.forIn_range _ _
    (fun j _ _ => ∀ k < j, dot (FlockSoundness.Refine.wOf claims[k]!) rs[k]! = claims[k]!.value) ?_ _ _
    fun k hk => absurd hk (Nat.not_lt_zero k)) fun b t₁ h₁ => ?_)
  · intro i b t _ hi
    refine RetV.ok_then (Flock.RecAlgebra.label_eq _ _) ?_
    refine RetV.ok_then (Flock.RecAlgebra.absorbSlice_eq _ _) ?_
    refine RetV.bind fun u t₂ e => ?_
    have hc := beq_iff_eq.mp (check_true e)
    refine RetV.bind fun _ _ _ => RetV.pure ⟨_, rfl, fun k hk => ?_⟩
    rcases Nat.lt_succ_iff_lt_or_eq.1 hk with hk | rfl
    · exact hi k hk
    · exact hc
  · exact RetV.bind fun _ _ _ => RetV.pure h₁

namespace Rs

variable {tr : Trace} {c₁ c₂ : Array Claim}

theorem betas_congr (hn : c₁.size = c₂.size) (hx : ∀ k : ℕ, c₁[k]!.x = c₂[k]!.x) (cs : Coins) :
    betas tr c₁ cs = betas tr c₂ cs := by
  unfold betas
  rw [hn]
  exact pushes_congr fun k _ => by rw [switch_congr hx k]

theorem T_congr (hn : c₁.size = c₂.size) (hx : ∀ k : ℕ, c₁[k]!.x = c₂[k]!.x) (cs : Coins) :
    T tr c₁ cs = T tr c₂ cs := by
  unfold T
  rw [hn, betas_congr hn hx]

theorem perClaim_congr (hn : c₁.size = c₂.size) (hx : ∀ k : ℕ, c₁[k]!.x = c₂[k]!.x) (cs : Coins) :
    perClaim c₁ cs = perClaim c₂ cs := by
  unfold perClaim
  rw [hn]
  exact pushes_congr fun k _ => by rw [hx k]

theorem sws_congr (hn : c₁.size = c₂.size) (hx : ∀ k : ℕ, c₁[k]!.x = c₂[k]!.x) (cs : Coins) :
    sws c₁ cs = sws c₂ cs := by
  unfold sws
  rw [hn, perClaim_congr hn hx]

end Rs

/-- **The verifier's ring switch returns the prover's**: after a masked replay that returned on rounds carrying the
prover's through its ring switch, a ring switch that returns gives the prover's `T` and switched claims and stops
where the prover's did. Its checks passing is each claim's identity on the prover's switched vectors. -/
theorem rs_eq (hs : Hashes V) (st : Setup) (hk : st.fold.kLog = st.kLog) (tr : Trace) (e : Array Claim)
    (lay : Zk.PadLayout) (hp : Array F128) (digest cap root : ByteArray) (cz : Zc.Coins)
    (full : Array F128) (cl : Lc.Coins) (crs : Rs.Coins) (t₀ : Transcript) (h₀ : t₀.k = 0)
    (hs₀ : t₀.rounds.size = 0) (R : Array Round) (rp : Zk.Replayed) (prf : Proof)
    (hR : Agrees R (Rs.tEnd tr (abcClaims st.m st.kLog (Zc.out st.m cz tr)
      (Lc.out st.fold st.m tr (Zc.out st.m cz tr) cl) ++ e) crs
      (LcZ.tEnd lay hp st.fold st.m tr (Zc.out st.m cz tr) full cl
        (ZcZ.tEnd lay hp st.m digest cap root cz tr t₀))).rounds 0)
    (hcl : rp.claims = abcClaims st.m st.kLog (zcAt st.m R 0) (lcAt st.m st.kLog R 0))
    (hsw : prf.ringSwitches = Rs.proof tr (abcClaims st.m st.kLog (Zc.out st.m cz tr)
      (Lc.out st.fold st.m tr (Zc.out st.m cz tr) cl) ++ e))
    {sw : (F128 × Array Switched) × Transcript}
    (hret : (ringSwitchWith hs st.m (ringClaimsZk rp prf ++ e) prf.ringSwitches).run
      ⟨R, 0 + 3 + (st.m - 6) + 3 + (st.kLog - 6), ByteArray.empty⟩ = .ok sw) :
    sw = ((Rs.T tr (abcClaims st.m st.kLog (Zc.out st.m cz tr) (Lc.out st.fold st.m tr (Zc.out st.m cz tr) cl) ++ e)
        crs, Rs.sws (abcClaims st.m st.kLog (Zc.out st.m cz tr) (Lc.out st.fold st.m tr (Zc.out st.m cz tr) cl) ++ e)
        crs),
      ⟨R, (Rs.tEnd tr (abcClaims st.m st.kLog (Zc.out st.m cz tr)
        (Lc.out st.fold st.m tr (Zc.out st.m cz tr) cl) ++ e) crs
        (LcZ.tEnd lay hp st.fold st.m tr (Zc.out st.m cz tr) full cl
          (ZcZ.tEnd lay hp st.m digest cap root cz tr t₀))).k, ByteArray.empty⟩) := by
  have hAL := hR.of_prefix (Rs.prefix_loop (i := 0) _ (Nat.zero_le _))
  have hAZ := hAL.of_prefix (LcZ.prefix_start _)
  have hzt : (ZcZ.tEnd lay hp st.m digest cap root cz tr t₀).k =
      (ZcZ.tEnd lay hp st.m digest cap root cz tr t₀).rounds.size := by
    rw [ZcZ.tEnd_k, ZcZ.tEnd_size, h₀, hs₀]
  obtain ⟨hρ, hr⟩ := zc_coins t₀ (by rw [h₀, hs₀]) hAZ
  have hi := lc_coins _ hzt hAL 0 (by rw [ZcZ.tEnd_k, h₀])
  rw [h₀] at hρ hr
  rw [hk] at hi
  obtain ⟨hx₀, hx₁⟩ := abc_x (m := st.m) (kLog := st.kLog) hρ hr hi
  rw [← hcl] at hx₀ hx₁
  have hsz : (abcClaims st.m st.kLog (Zc.out st.m cz tr) (Lc.out st.fold st.m tr (Zc.out st.m cz tr) cl)).size = 2 :=
    rfl
  have hrc : (ringClaimsZk rp prf).size = 2 := rfl
  have hx := claims_x (e := e) hsz hrc (by rw [← hx₀]; rfl) (by rw [← hx₁]; rfl)
  have hn : (abcClaims st.m st.kLog (Zc.out st.m cz tr) (Lc.out st.fold st.m tr (Zc.out st.m cz tr) cl) ++ e).size =
      (ringClaimsZk rp prf ++ e).size := by
    simp only [Array.size_append, hsz]
    rfl
  have hlt : (LcZ.tEnd lay hp st.fold st.m tr (Zc.out st.m cz tr) full cl
      (ZcZ.tEnd lay hp st.m digest cap root cz tr t₀)).k = (LcZ.tEnd lay hp st.fold st.m tr (Zc.out st.m cz tr) full
      cl (ZcZ.tEnd lay hp st.m digest cap root cz tr t₀)).rounds.size := by
    rw [LcZ.tEnd_k, LcZ.tEnd_size, hzt]
  have hchk := ringSwitchWith_checks hs st.m _ _ _ _ _ hret
  have hval : ∀ k < (ringClaimsZk rp prf ++ e).size, dot (FlockSoundness.Refine.wOf (ringClaimsZk rp prf ++ e)[k]!)
      (switchVec tr (ringClaimsZk rp prf ++ e)[k]!) = (ringClaimsZk rp prf ++ e)[k]!.value := by
    intro k hk'
    have := hchk k hk'
    rwa [hsw, Rs.proof_congr hn hx, Rs.proof_get k hk'] at this
  have key := ringSwitchWith_of hs st.m tr (ringClaimsZk rp prf ++ e) crs _ hlt
    (by simp only [Array.size_append, hrc]; omega) (by rw [← Rs.tEnd_congr hn hx]; exact hR.mono (Nat.zero_le _)) hval
  rw [LcZ.tEnd_k, ZcZ.tEnd_k, h₀, hk] at key
  rw [hsw, Rs.proof_congr hn hx] at hret
  have e2 := Except.ok.inj (hret.symm.trans key)
  rw [e2, Rs.T_congr hn hx, Rs.sws_congr hn hx, Rs.tEnd_congr hn hx]
  rfl

/-- **Every claim a rep opens has `m − 6` coordinates**, for a table with `k_log ≤ m`, at any link points: the extra
claims' points are region points, of `m` coordinates (`regionPoint_size`). -/
theorem claims_xsize' (st : Stmt) (hk : st.c.kLog ≤ st.m) (hm : 13 ≤ st.m) (tr : Trace) (zcs : Zc.Coins)
    (cs : Lc.Coins) (points : Array (Array F128)) (extras : Array Claim) (hex : st.extraClaims points = .ok extras) :
    ∀ c ∈ abcClaims st.m st.c.kLog (Zc.out st.m zcs tr) (Lc.out st.circuitFold st.m tr (Zc.out st.m zcs tr) cs) ++
        extras, c.x.size = st.m - 6 := by
  intro c hc
  rcases Array.mem_append.1 hc with hc | hc
  · simp only [abcClaims, List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hc
    rcases hc with rfl | rfl
    · show ((pushes cs.τ (st.c.kLog - 6)).reverse ++
        (pushes zcs.ρ (st.m - 6)).extract (st.c.kLog - 6) (st.m - 6)).size = st.m - 6
      rw [Array.size_append, Array.size_reverse, pushes_size, Array.size_extract, pushes_size, Nat.min_self]
      omega
    · exact Zc.r_size st.m hm zcs
  · obtain ⟨ess, hess, heq⟩ := FlockSoundness.Refine.extraClaims_ok st points extras hex
    have hc' : c ∈ ess.flatten := by
      rw [← heq]
      exact Array.mem_def.1 hc
    obtain ⟨reg, -, r, -, v, -, rfl⟩ := mem_of_forall₂ hess hc'
    simp only [FlockSoundness.Refine.regionClaimOf, Array.size_extract, FlockSoundness.Refine.regionPoint_size,
      Nat.min_self]

/-! ## The verifier's replay is the prover's -/

theorem recs_bytes {resp : ℕ → Claims.Msg} :
    ∀ {i : ℕ} {rs : List Round} {bs : List Claims.Msg}, JsonRT.Recs resp i rs bs → ∀ r ∈ rs, r.bytes.isSome
  | _, [], [], _, _, hr => by simp at hr
  | _, _ :: _, _ :: _, h, r, hr => by
    rcases List.mem_cons.1 hr with rfl | hr
    · obtain ⟨c, hc, -⟩ := h.1
      simp [hc]
    · exact recs_bytes h.2 r hr
  | _, [], _ :: _, h, _, _ => h.elim
  | _, _ :: _, [], h, _, _ => h.elim

/-- **The verifier's masked replay is the prover's**: on rounds `R` that carry the prover's rounds `P`, each of which
carries its sent bytes, the verifier's replay under any hash returns what the prover's own replay (`Zk.replay`, the
deployed hash's) returned on `P`: it reads only rounds' coins and positions (`replayWith_xfer`). -/
theorem replay_bridge (hs : Hashes V) (st : Setup) (lay : Zk.PadLayout) (prf : Proof) (root : ByteArray)
    {P R : Array Round} (d : ByteArray) (hb : ∀ j (h : j < P.size), P[j].bytes.isSome) (hAg : Agrees R P 0)
    {rp : Zk.Replayed} {x : Transcript} (hre : (Zk.replay st lay prf root).run (Transcript.init P d) = .ok (rp, x))
    {q : Zk.Replayed × Transcript} (hq : (Zk.replayWith hs st lay prf root).run (Transcript.init R d) = .ok q) :
    q.1 = rp := by
  rw [← With.zk_replayWith_std] at hre
  obtain ⟨u', e, -⟩ := replayWith_xfer With.std hs st lay prf root (Transcript.init P d) (Transcript.init R d) rp x
    ⟨rfl, rfl, fun j h => ⟨hb j h, hAg j h (Nat.zero_le _)⟩⟩ hre
  obtain rfl := Except.ok.inj (hq.symm.trans e)
  rfl

/-! ## The prover's inner proof that returned -/

/-- **A masked inner proof that returned** passed its two refusals at the replies' coins, and is `InZ.proof` with
transcript `InZ.tEnd` (`innerP_go`'s converse). -/
theorem innerP_ret (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (lay : Zk.PadLayout) (p : PadsZk)
    (ic : InnerCommit) (finalC : F128) (cons : Array (String × Zk.Lin)) (t : Transcript) (l : Array Claims.Msg)
    {a : Zk.InnerProof} {t' : Transcript} {l' : Array Claims.Msg}
    (e : ((innerP (asksS resp) (Hashes.ofSha512 h) lay p ic finalC cons).run t).go l = (.ok (a, t'), l')) :
    ((InZ.lin lay p cons (InZ.coinsAt resp l.size)).c +
      dot (InZ.lin lay p cons (InZ.coinsAt resp l.size)).a p.h).isZero = true ∧
    (coinOf (resp (l.size + 2)) 0).isZero = false ∧
    a = InZ.proof lay p cons h ic (InZ.coinsAt resp l.size) ∧
    t' = InZ.tEnd lay p finalC cons h (InZ.coinsAt resp l.size) t := by
  by_cases hz : ((InZ.lin lay p cons (InZ.coinsAt resp l.size)).c +
      dot (InZ.lin lay p cons (InZ.coinsAt resp l.size)).a p.h).isZero = true
  · by_cases hb : (coinOf (resp (l.size + 2)) 0).isZero = false
    · obtain ⟨l'', -, e'⟩ := innerP_go resp h lay p ic finalC cons t l hz hb
      rw [e'] at e
      simp only [Prod.mk.injEq, Except.ok.injEq] at e
      exact ⟨hz, hb, e.1.1.symm, e.1.2.symm⟩
    · exfalso
      unfold innerP at e
      rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _)] at e
      rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) _)] at e
      rw [bind_go (sampleP_go resp _ _)] at e
      dsimp only at e
      rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbSlice_eq _ _) _)] at e
      rw [bind_go (sampleVecP_go resp (by simp [Zk.triple]) _ _)] at e
      simp only [InZ.lin, InZ.all, InZ.rho, InZ.sigma, InZ.coinsAt] at hz
      simp only [Array.size_push, hz, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at e
      rw [tauCommitmentWith_ofSha512] at e
      rw [bind_go (x := StateT.lift (pure _)) rfl] at e
      rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) _)] at e
      rw [bind_go (sampleP_go resp _ _)] at e
      rw [Bool.not_eq_false] at hb
      simp only [Array.size_push, hb, ↓reduceIte] at e
      cases e
  · exfalso
    unfold innerP at e
    rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _)] at e
    rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) _)] at e
    rw [bind_go (sampleP_go resp _ _)] at e
    dsimp only at e
    rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbSlice_eq _ _) _)] at e
    rw [bind_go (sampleVecP_go resp (by simp [Zk.triple]) _ _)] at e
    rw [Bool.not_eq_true] at hz
    simp only [InZ.lin, InZ.all, InZ.rho, InZ.sigma, InZ.coinsAt] at hz
    simp only [Array.size_push, hz, Bool.not_false, ↓reduceIte] at e
    cases e

/-- **The verifier's inner check passes on the masked prover's inner proof** where its batched constraint holds at
its coins (`verifyInnerWith_of` with the batched constraint for `ConsHold`). -/
theorem verifyInnerWith_of' (h : ByteArray → ByteArray) (h64 : ∀ b, (h b).size = 64) (lay : Zk.PadLayout)
    (tape : TapeZk) (i r : ℕ) (salt : ℕ → ByteArray) {ic : InnerCommit}
    (hic : innerCommitP (Hashes.ofSha512 h : Hashes V) (padsOf tape i r lay) salt = .ok ic) (finalC : F128)
    {cons : Array (String × Zk.Lin)} (cs : InZ.Coins)
    (hz : ((InZ.lin lay (padsOf tape i r lay) cons cs).c +
      dot (InZ.lin lay (padsOf tape i r lay) cons cs).a (padsOf tape i r lay).h).isZero = true)
    (hb : (cs.cb 0).isZero = false) (t : Transcript) (hk : t.k = t.rounds.size) {R : Array Round}
    (hR : Agrees R (InZ.tEnd lay (padsOf tape i r lay) finalC cons h cs t).rounds t.k) :
    (Zk.verifyInnerWith (Hashes.ofSha512 h : Hashes V) lay (InZ.proof lay (padsOf tape i r lay) cons h ic cs) finalC
      cons) ⟨R, t.k, t.pending⟩ = .ok ((), ⟨R, t.k + 4, ByteArray.empty⟩) := by
  set p := padsOf tape i r lay with hp
  have ht := innerCommitP_tree hic
  have hN : ic.ch.size = (lay.len + Zk.K_PAD).nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE := by
    rw [ht.ch, padsOf_symbols]
  have c1 : Carries R t.k (sent (InZ.t₁ finalC t) 1 1 cs.ce) := by
    have := hR.at_push (A := (InZ.t₁ finalC t).rounds) (x := sent (InZ.t₁ finalC t) 1 1 cs.ce)
      (((Prefix.push _ _).trans (Prefix.push _ _)).trans (Prefix.push _ _)) (by simp [InZ.t₁, hk])
    rwa [show (InZ.t₁ finalC t).rounds.size = t.k from hk.symm] at this
  have c2 : Carries R (t.k + 1) (sent (InZ.t₃ lay p finalC cs t) 2 (InZ.all lay p cons cs).size cs.cg) := by
    have := hR.at_push (A := (InZ.t₃ lay p finalC cs t).rounds)
      (x := sent (InZ.t₃ lay p finalC cs t) 2 (InZ.all lay p cons cs).size cs.cg)
      ((Prefix.push _ _).trans (Prefix.push _ _)) (by simp [InZ.t₃, InZ.t₂, InZ.t₁, afterSq, hk])
    rwa [show (InZ.t₃ lay p finalC cs t).rounds.size = t.k + 1 by simp [InZ.t₃, InZ.t₂, InZ.t₁, afterSq, hk]] at this
  have c3 : Carries R (t.k + 2) (sent (InZ.t₅ lay p finalC cons h cs t) 1 1 cs.cb) := by
    have := hR.at_push (A := (InZ.t₅ lay p finalC cons h cs t).rounds)
      (x := sent (InZ.t₅ lay p finalC cons h cs t) 1 1 cs.cb)
      (Prefix.push _ _) (by simp [InZ.t₅, InZ.t₄, InZ.t₃, InZ.t₂, InZ.t₁, afterSq, hk]; omega)
    rwa [show (InZ.t₅ lay p finalC cons h cs t).rounds.size = t.k + 2 by
      simp [InZ.t₅, InZ.t₄, InZ.t₃, InZ.t₂, InZ.t₁, afterSq, hk]] at this
  have c4 : Carries R (t.k + 3) (sent (InZ.t₇ lay p finalC cons h cs t) 2 Zk.Q cs.cq) := by
    have := hR.at_push (A := (InZ.t₇ lay p finalC cons h cs t).rounds)
      (x := sent (InZ.t₇ lay p finalC cons h cs t) 2 Zk.Q cs.cq)
      (Prefix.refl _) (by simp [InZ.t₇, InZ.t₆, InZ.t₅, InZ.t₄, InZ.t₃, InZ.t₂, InZ.t₁, afterSq, hk]; omega)
    rwa [show (InZ.t₇ lay p finalC cons h cs t).rounds.size = t.k + 3 by
      simp [InZ.t₇, InZ.t₆, InZ.t₅, InZ.t₄, InZ.t₃, InZ.t₂, InZ.t₁, afterSq, hk]] at this
  have hrows := innerCommitP_rows hic
  have hpow : ic.ch.size = 2 ^ Nat.log2 ic.ch.size := by rw [ht.ch]; exact innerCols_pow _
  unfold Zk.verifyInnerWith
  simp only [Flock.With.liftTM_V]
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq ?sz _ _)) ?_
  case sz =>
    simp only [InZ.proof, InZ.y, InZ.qs, Array.size_map, Array.size_range, padsOf_symbols,
      JsonRT.innerCommitP_root h64 _ _ hic, tcOf_size h64, padsOf_tauSalt_size, hp, beq_self_eq_true, Bool.and_self]
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.label_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbF_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (sampleWith_sent_at _ _ (by rfl) (by rfl) (by rfl) c1)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbSlice_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (sampleVecWith_sent_at _ (by simp [Zk.triple]) _ (by rfl)
    (by rfl) (by rfl) c2)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbBytes_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (sampleWith_sent_at _ _ (by rfl) (by rfl) (by rfl) c3)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq (by rw [hb]; rfl) _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbSlice_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (sampleVecWith_sent_at _ (by decide) _ (by rfl) (by rfl) (by rfl)
    c4)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (lift_ok (a := InZ.tcOf h (InZ.tau lay p cons cs) p.tauSalt) rfl _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq (bytes_beq_refl _) _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq
    (innerY_batched (InZ.lin lay p cons cs) p.h p.mu p.padH p.padMu (cs.cb 0)
      (by rw [InZ.lin, FlockSoundness.Discharge.Composed.ZerocheckMasked.size_combo, hp, padsOf_size])
      (by rw [hp]; exact padsOf_mu_size tape i r lay) hz) _ _)) ?_
  have ht' : InnerTree (Hashes.ofSha512 h) ic.ch.size ic := by rw [ht.ch]; exact ht
  have hqs : (InZ.qs ic cs).size = Zk.Q := by simp [InZ.qs]
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (forIn_keep _ Zk.Q _ fun k hk' => ?_)) rfl
  have hqk : (InZ.qs ic cs)[k]! = (cs.cq k).lo.toNat % ic.ch.size := qs_get hk'
  have hop : (InZ.proof lay p cons h ic cs).opened[k]! =
      (ic.ch[(cs.cq k).lo.toNat % ic.ch.size]!, ic.cm[(cs.cq k).lo.toNat % ic.ch.size]!) := by
    show ((InZ.qs ic cs).map _)[k]! = _
    rw [map_get _ _ (hqs ▸ hk'), hqk]
  have hsl : (InZ.proof lay p cons h ic cs).salts[k]! = ic.salts[(cs.cq k).lo.toNat % ic.ch.size]! := by
    show ((InZ.qs ic cs).map _)[k]! = _
    rw [map_get _ _ (hqs ▸ hk'), hqk]
  have hpa : (InZ.proof lay p cons h ic cs).paths[k]! = (Array.range (Nat.log2 ic.ch.size)).map
      fun k' => (ic.layers[k']!)[(((cs.cq k).lo.toNat % ic.ch.size) >>> k') ^^^ 1]! := by
    show ((InZ.qs ic cs).map _)[k]! = _
    rw [map_get _ _ (hqs ▸ hk'), hqk]
  rw [qs_get hk', ← hN, hop, hsl, hpa]
  set j := (cs.cq k).lo.toNat % ic.ch.size with hj
  have hjn : j < ic.ch.size := Nat.mod_lt _ (by rw [hpow]; exact Nat.two_pow_pos _)
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq ?col _ _)) ?_
  case col =>
    have hcol := innerY_column (p.h ++ p.padH) (p.mu ++ p.padMu) (cs.cb 0) (by simp [hp, padsOf]) _ _
      (show j < (p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE by rw [← ht.ch]; exact hjn)
    rw [← hrows.1, ← hrows.2] at hcol
    exact hcol
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq (by simp) _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (lift_ok rfl _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (lift_ok (ht'.leaf _ hjn) _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (lift_ok (innerTree_climb ht' hpow hjn) _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq (bytes_beq_refl _) _ _)) ?_
  rfl

theorem foldl_size_const {β : Type} {f : Array F256 → β → Array F256} {N : ℕ} (hf : ∀ a b, (f a b).size = N) :
    ∀ (l : List β) (a : Array F256), a.size = N → (l.foldl f a).size = N
  | [], _, h => h
  | b :: l, a, _ => foldl_size_const hf l (f a b) (hf a b)

theorem basis_size (sws : Array Switched) (q : Array F256) (e : ℕ) : (basis sws q e).size = 2 ^ e := by
  unfold basis
  rw [← Array.foldl_toList]
  exact foldl_size_const (fun _ _ => by simp) _ _ (by simp)

end FlockComplete.Replay
