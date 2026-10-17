import Proofs.Flock.Claims.Complete.ZkTables.RepSplit
import Proofs.Flock.Claims.Complete.ZkTables.Level0
import Proofs.Flock.Claims.Complete.ZkTables.Table
import Proofs.Flock.Claims.Complete.Replay.LigeritoBasis
import Proofs.Flock.Claims.Complete.Replay.LigeritoSum

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

theorem foldl_size_const {β : Type} {f : Array F256 → β → Array F256} {N : ℕ} (hf : ∀ a b, (f a b).size = N) :
    ∀ (l : List β) (a : Array F256), a.size = N → (l.foldl f a).size = N
  | [], _, h => h
  | b :: l, a, _ => foldl_size_const hf l (f a b) (hf a b)

theorem basis_size (sws : Array Switched) (q : Array F256) (e : ℕ) : (basis sws q e).size = 2 ^ e := by
  unfold basis
  rw [← Array.foldl_toList]
  exact foldl_size_const (fun _ _ => by simp) _ _ (by simp)

end FlockComplete.Replay
