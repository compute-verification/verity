import Proofs.Flock.Claims.Complete.ZkTables.View
import Proofs.Flock.Claims.Complete.Zk.LigeritoPos

/-!
# The masked prover's rep: its round count

* `zkSchedule_q`: every level of a `--zk` schedule draws queries.
* `rep_rounds`: given level 0's verifier premise (`Level0.Verifier`), a masked rep that returns records `repRoundsT`
  rounds. The verifier's masked Ligerito, on the prover's rounds, ends where the prover's did (`Level0.Verifier`),
  which is `k_0 + 7 + 11 R` rounds on (`zkLigeritoWith_pos`), and the inner proof adds four.
-/

namespace FlockComplete.Replay

open Flock Transcript FlockComplete

theorem zkSchedule_q {m : ℕ} {v : Schedule × ℕ} (h : Zk.schedule m = .ok v) :
    ∀ i < v.1.levels.size, (v.1.levels.getD i default).q ≠ 0 := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, (match Zk.schedule m.val with
        | .ok v => decide (∀ i < v.1.levels.size, (v.1.levels.getD i default).q ≠ 0)
        | .error _ => true) = true := by
      decide +kernel
    have := key ⟨m, hm⟩
    simp only [h] at this
    simpa using this
  · exfalso
    unfold Zk.schedule at h
    obtain ⟨q0, hq0, -⟩ := except_ok_bind h
    unfold Zk.queries0 at hq0
    simp [show ¬m ≤ 24 by omega, show m ≠ 25 by omega, show m ≠ 26 by omega, show m ≠ 27 by omega,
      show ¬m ≤ 31 by omega, show ¬m ≤ 35 by omega] at hq0

theorem Agrees.self (P : Array Round) (k₀ : ℕ) : Agrees P P k₀ :=
  fun j h _ => ⟨P[j], Array.getElem?_eq_getElem h, rfl, rfl, rfl⟩

theorem InZ.tEnd_size (lay : Zk.PadLayout) (p : PadsZk) (finalC : F128) (cons : Array (String × Zk.Lin))
    (h : ByteArray → ByteArray) (cs : InZ.Challenges) (t : Transcript) :
    (InZ.tEnd lay p finalC cons h cs t).rounds.size = t.rounds.size + 4 := by
  simp [InZ.tEnd, InZ.t₇, InZ.t₆, InZ.t₅, InZ.t₄, InZ.t₃, InZ.t₂, InZ.t₁, afterSq]

end FlockComplete.Replay

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy FlockComplete.Replay

variable {H : Claims.Msg → Claims.Msg}

/-- **A masked rep that returns records `repRoundsT` rounds**, given level 0's verifier premise. -/
theorem rep_rounds (hV : Level0.Verifier) (h64 : Claims.Out64 H) (st : Flock.Stmt) (tr : Trace) (hT : TableT st tr)
    (ω : TapeZk) (i : ℕ) (d : ByteArray) {k : ℕ} (r : Fin k → Coin) (v : Schedule × ℕ) (cm : Commit)
    (pts : Array (Array F128)) (e : Array Claim) (rep : ℕ) (ic : InnerCommit)
    (hin : RepIn H st tr ω i v cm pts e rep ic) (hrep : rep < 2) (prior : Array (List F128)) (ws : List ℕ)
    (l : Array Claims.Msg) (p : Zk.ZkProof) (t' : Transcript) (l' : Array Claims.Msg)
    (hgo : (repT H r st tr ω i d v cm e rep ic prior ws).go l = (.ok (p, t'), l')) :
    t'.rounds.size = repRoundsT st v e := by
  obtain ⟨h25, h35, hms, -, p0, w, hpe, hR0, -⟩ := hT
  obtain ⟨hf, -⟩ := holds_only hpe hR0
  have hkm : st.c.kLog ≤ st.m := hf.2.2.2.1.trans_eq hf.1
  have hm14 : 14 ≤ (Setup.ofCircuit st).m := by show 14 ≤ st.m; omega
  have hmerk : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 := hms
  set resp : ℕ → Claims.Msg := fun j => (Claims.coinSeq r j).bytes with hresp
  set lay := Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0 with hlay
  set pads := padsOf ω i rep lay with hpads
  unfold repT at hgo
  obtain ⟨full, lL, lg, zl, rec, tI, lI, rp, x, lgc, hfold, hlL, hlig, hre, hlgc, hinn, hP, hzl⟩ :=
    repPZk_split resp (Claims.hOf H) (Setup.ofCircuit st) v.1 v.2 rep d tr (packed tr (Setup.ofCircuit st).m) cm e
      pads ic (ω.extra i) (ω.lanePad i) (ω.salt i rep) (ω.nonce i rep) prior ws hm14 _ l hgo
  have hn : v.1.n = st.m - 7 := zkSchedule_n hin.schedule
  have hx : ∀ c ∈ claimsR resp (Setup.ofCircuit st) tr l.size e, c.x.size = st.m - 6 :=
    claims_xsize' st hkm (by omega) tr (Zc.coinsAt resp l.size) (lcCoinsR resp (Setup.ofCircuit st) l.size) pts e
      hin.extra
  have hcs : (claimsR resp (Setup.ofCircuit st) tr l.size e).size = 2 + e.size := by simp [claimsR, abcClaims]
  have htk : (tPiop resp (Setup.ofCircuit st) tr l.size lay pads.h (Setup.ofCircuit st).digest (flat cm.cap) ic.root
      full e (Transcript.init #[] d)).k = (tPiop resp (Setup.ofCircuit st) tr l.size lay pads.h
      (Setup.ofCircuit st).digest (flat cm.cap) ic.root full e (Transcript.init #[] d)).rounds.size := by
    simp only [tPiop, Rs.tEnd_k, Rs.tEnd_size, LcZ.tEnd_k, LcZ.tEnd_size, ZcZ.tEnd_k, ZcZ.tEnd_size]
    rfl
  have hpk : (tPiop resp (Setup.ofCircuit st) tr l.size lay pads.h (Setup.ofCircuit st).digest (flat cm.cap) ic.root
      full e (Transcript.init #[] d)).k = 3 + (st.m - 6) + 3 + (st.c.kLog - 6) + (2 + e.size) + 1 := by
    simp only [tPiop, Rs.tEnd_k, LcZ.tEnd_k, ZcZ.tEnd_k, hcs]
    show 0 + 3 + (st.m - 6) + 3 + (st.c.kLog - 6) + (2 + e.size) + 1 = _
    omega
  have hLk : lL.size = l.size + (tPiop resp (Setup.ofCircuit st) tr l.size lay pads.h (Setup.ofCircuit st).digest
      (flat cm.cap) ic.root full e (Transcript.init #[] d)).k := by
    rw [hlL, hpk]
    have : piopSends (Setup.ofCircuit st) e = 3 + (st.m - 6) + 3 + (st.c.kLog - 6) + (2 + e.size + 1) := rfl
    omega
  obtain ⟨hLgV, htIk⟩ := hV resp (Claims.hOf H) (Setup.ofCircuit st).m v
    (Rs.T tr (claimsR resp (Setup.ofCircuit st) tr l.size e) (rsCoinsR resp (Setup.ofCircuit st) l.size e)) cm
    (packed tr (Setup.ofCircuit st).m)
    (basisVals (Rs.sws (claimsR resp (Setup.ofCircuit st) tr l.size e) (rsCoinsR resp (Setup.ofCircuit st) l.size e))
      ((Setup.ofCircuit st).m - 7))
    lay pads.h rep (ω.extra i) (ω.lanePad i) (ω.salt i 0 0) (ω.salt i rep) (ω.nonce i rep)
    (basis (Rs.sws (claimsR resp (Setup.ofCircuit st) tr l.size e) (rsCoinsR resp (Setup.ofCircuit st) l.size e)))
    l.size _ tI lL lI lg zl rec tI.rounds (JsonRT.hOf_size h64) h25 h35 hin.schedule hrep
    (by simp [packed, hn]; rfl) (by simp [basisVals, hn]; rfl) (Lg.rsSum_eq tr _ _ _ hx)
    (by rw [← hmerk]; exact hin.commit)
    (fun q e' y hqe hy => by rw [hn] at hqe ⊢; exact Lg.basis_sws _ _ _ hx q e' y hqe hy)
    (fun q e' => basis_size _ q e') htk hLk (by rw [hmerk] at hlig; exact hlig) (Agrees.self _ _)
  obtain ⟨hk0, h2⟩ := zkSchedule_shape hin.schedule
  obtain ⟨-, -, -, -, -, hend, -⟩ := FlockComplete.zkLigeritoWith_pos _ _ v.1 _ _ _ _ _ (by omega) hk0 h2
    (zkSchedule_q hin.schedule) hLgV
  have htI : tI.k = (tPiop resp (Setup.ofCircuit st) tr l.size lay pads.h (Setup.ofCircuit st).digest (flat cm.cap)
      ic.root full e (Transcript.init #[] d)).k + (v.1.k0 + 7 + 11 * (v.1.levels.size - 1)) :=
    congrArg Transcript.k hend
  obtain ⟨-, -, -, ht'⟩ := innerP_ret resp (Claims.hOf H) _ _ _ _ _ _ _ hinn
  rw [ht', InZ.tEnd_size, ← htIk, htI, hpk]
  unfold repRoundsT
  omega

end FlockComplete.ZkTables
