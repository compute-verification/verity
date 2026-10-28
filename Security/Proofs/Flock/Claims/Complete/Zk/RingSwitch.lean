import Proofs.Flock.Claims.Complete.Zk.PiopGo
import Proofs.Flock.Claims.Complete.Zk.ReplayPos
import Proofs.Flock.Claims.Complete.Zk.Replay

/-!
# The masked ring switch replays (`RingSwitchZk`)

`ringSwitchZk_goal`: at every honest masked run, the verifier's ring switch on the replay's claims, each valued by its
check, and the extra claims returns, given the extra claims' algebra (`RsExtrasZk`).

* **The prover** (`repPZk_rs`): a masked rep that returns ran zerocheck and lincheck in closed form
  (`zerocheckPZk_go`, `lincheckPZk_go`, the circuit fold returning since the rep did, `lincheckPZk_fold`), then the
  plain ring switch on its claims (`ringSwitchP_go`); its final rounds begin with those rounds.
* **The verifier**: its replay stopped where the prover's lincheck did, and its claims' points are the prover's
  claims' (`replayWith_ret`, at the prover's coins the stream carries, `zc_coins`, `lc_coins`). The ring switch reads
  claims only through their points (`Rs.tEnd_congr`, `Rs.proof_congr`), so it replays the prover's
  (`ringSwitchWith_of`, `rs_of`): the two checked claims meet their values by definition (`Zk.claimCheck`), the extra
  claims by `RsExtrasZk`.
-/

namespace FlockComplete.Replay

open Flock Transcript

namespace ZcZ

variable {lay : Zk.PadLayout} {h : Array F128} {m : ℕ} {digest cap root : ByteArray} {cs : Zc.Challenges} {tr : Trace}

theorem loopT_k (t₀ : Transcript) (i : ℕ) : (loopT lay h m cs tr t₀ i).k = t₀.k + i := by
  induction i with
  | zero => rfl
  | succ i ih => simp only [loopT, afterSq, ih]; omega

theorem loopT_size (t₀ : Transcript) (i : ℕ) : (loopT lay h m cs tr t₀ i).rounds.size = t₀.rounds.size + i := by
  induction i with
  | zero => rfl
  | succ i ih => simp only [loopT, afterSq, Array.size_push, ih]; omega

theorem loopT_prefix (t₀ : Transcript) (i : ℕ) :
    ∀ j, Prefix (loopT lay h m cs tr t₀ i).rounds (loopT lay h m cs tr t₀ (i + j)).rounds
  | 0 => Prefix.refl _
  | j + 1 => (loopT_prefix t₀ i j).trans (Prefix.push _ _)

theorem tEnd_k (t : Transcript) : (tEnd lay h m digest cap root cs tr t).k = t.k + 3 + (m - 6) := by
  simp only [tEnd, loopT_k, t₅, t₄, t₃, t₂, t₁, afterSq]

theorem tEnd_size (t : Transcript) :
    (tEnd lay h m digest cap root cs tr t).rounds.size = t.rounds.size + 3 + (m - 6) := by
  simp only [tEnd, loopT_size, t₅, t₄, t₃, t₂, t₁, afterSq, Array.size_push]

theorem prefix_loop (t : Transcript) {i : ℕ} (hi : i ≤ m - 6) :
    Prefix (loopT lay h m cs tr (t₅ lay h m digest cap root cs tr t) i).rounds
      (tEnd lay h m digest cap root cs tr t).rounds := by
  have := loopT_prefix (lay := lay) (h := h) (m := m) (cs := cs) (tr := tr) (t₅ lay h m digest cap root cs tr t) i
    (m - 6 - i)
  rwa [show i + (m - 6 - i) = m - 6 by omega] at this

end ZcZ

namespace LcZ

variable {lay : Zk.PadLayout} {h : Array F128} {c : CircuitFold} {m : ℕ} {tr : Trace} {zc : ZcOut}
  {full : Array F128} {cs : Lc.Challenges}

theorem loopT_k (t : Transcript) (i : ℕ) : (loopT lay h c m tr zc full cs t i).k = t.k + 2 + i := by
  induction i with
  | zero => rfl
  | succ i ih => simp only [loopT, afterSq, ih]; omega

theorem loopT_size (t : Transcript) (i : ℕ) :
    (loopT lay h c m tr zc full cs t i).rounds.size = t.rounds.size + 2 + i := by
  induction i with
  | zero => simp [loopT, Lc.t₃, Lc.t₂, Lc.t₁, afterSq]
  | succ i ih => simp only [loopT, afterSq, Array.size_push, ih]; omega

theorem loopT_prefix (t : Transcript) (i : ℕ) :
    ∀ j, Prefix (loopT lay h c m tr zc full cs t i).rounds (loopT lay h c m tr zc full cs t (i + j)).rounds
  | 0 => Prefix.refl _
  | j + 1 => (loopT_prefix t i j).trans (Prefix.push _ _)

theorem prefix_loop (t : Transcript) {i : ℕ} (hi : i ≤ c.kLog - 6) :
    Prefix (loopT lay h c m tr zc full cs t i).rounds (tEnd lay h c m tr zc full cs t).rounds := by
  have := loopT_prefix (lay := lay) (h := h) (c := c) (m := m) (tr := tr) (zc := zc) (full := full) (cs := cs) t i
    (c.kLog - 6 - i)
  rw [show i + (c.kLog - 6 - i) = c.kLog - 6 by omega] at this
  exact this.trans (Prefix.push _ _)

theorem prefix_start (t : Transcript) : Prefix t.rounds (tEnd lay h c m tr zc full cs t).rounds :=
  ((Prefix.push _ _).trans (Prefix.push _ _)).trans (prefix_loop (i := 0) t (Nat.zero_le _))

theorem tEnd_k (t : Transcript) : (tEnd lay h c m tr zc full cs t).k = t.k + 3 + (c.kLog - 6) := by
  simp only [tEnd, t₄, afterSq, loopT_k]; omega

theorem tEnd_size (t : Transcript) :
    (tEnd lay h c m tr zc full cs t).rounds.size = t.rounds.size + 3 + (c.kLog - 6) := by
  simp only [tEnd, t₄, afterSq, Array.size_push, loopT_size]; omega

end LcZ

/-! ## The coins the verifier's claims read -/

theorem coins_of {R : Array Round} {j : ℕ} {ρ : Round} (h : Carries R j ρ) : (R[j]!).coins = ρ.coins := by
  obtain ⟨r, hr, -, hc, -⟩ := h
  rw [getElem!_def, hr]
  exact hc

/-- **Zerocheck's points at the stream**: on rounds carrying the masked prover's zerocheck from `t.k`, the replay's
`ρ` and `r` are the prover's. -/
theorem zc_coins {lay : Zk.PadLayout} {h : Array F128} {m : ℕ} {digest cap root : ByteArray} {cs : Zc.Challenges}
    {tr : Trace} (t : Transcript) (hk : t.k = t.rounds.size) {R : Array Round}
    (hR : Agrees R (ZcZ.tEnd lay h m digest cap root cs tr t).rounds 0) :
    (zcAt m R t.k).rho = (Zc.out m cs tr).rho ∧ (zcAt m R t.k).rRest = (Zc.out m cs tr).rRest := by
  refine ⟨pushes_congr fun i hi => ?_, ?_⟩
  · have e : (ZcZ.loopT lay h m cs tr (ZcZ.t₅ lay h m digest cap root cs tr t) i).rounds.size = t.k + 3 + i := by
      rw [ZcZ.loopT_size]
      simp only [ZcZ.t₅, ZcZ.t₄, ZcZ.t₃, ZcZ.t₂, ZcZ.t₁, afterSq, Array.size_push, hk]
    have hc := hR.at_push (ZcZ.prefix_loop (i := i + 1) t (by omega)) (Nat.zero_le _)
    rw [e] at hc
    simp only [coinAt, coins_of hc, sent, Zc.Challenges.ρ]
    simp
  · have e : (ZcZ.t₂ digest cap root cs t).rounds.size = t.k + 1 := by
      simp only [ZcZ.t₂, ZcZ.t₁, afterSq, Array.size_push, hk]
    have hp : Prefix ((ZcZ.t₂ digest cap root cs t).rounds.push
        (sent (ZcZ.t₂ digest cap root cs t) 2 (m - 13) cs.c1)) (ZcZ.tEnd lay h m digest cap root cs tr t).rounds :=
      ((Prefix.push _ _).trans (ZcZ.prefix_loop (i := 0) t (Nat.zero_le _)))
    have hc := hR.at_push hp (Nat.zero_le _)
    rw [e] at hc
    simp only [zcAt, Zc.out, Zc.Challenges.r, coins_of hc, sent]

/-- **Lincheck's points at the stream**: on rounds carrying the masked prover's lincheck from `t.k`, the replay's
`r_in` is the prover's. -/
theorem lc_coins {lay : Zk.PadLayout} {h : Array F128} {c : CircuitFold} {m : ℕ} {tr : Trace} {zc : ZcOut}
    {full : Array F128} {cs : Lc.Challenges} (t : Transcript) (hk : t.k = t.rounds.size) {R : Array Round}
    (hR : Agrees R (LcZ.tEnd lay h c m tr zc full cs t).rounds 0) (k₀ : ℕ) (hk₀ : k₀ + 3 + (m - 6) = t.k) :
    (lcAt m c.kLog R k₀).rIn = (Lc.out c m tr zc cs).rIn := by
  simp only [lcAt, Lc.out]
  congr 1
  refine pushes_congr fun i hi => ?_
  have e : (LcZ.loopT lay h c m tr zc full cs t i).rounds.size = k₀ + 3 + (m - 6) + 2 + i := by
    rw [LcZ.loopT_size, hk₀, hk]
  have hc := hR.at_push (LcZ.prefix_loop (i := i + 1) t (by omega)) (Nat.zero_le _)
  rw [e] at hc
  simp only [coinAt, coins_of hc, sent, Lc.Challenges.τ]
  simp

/-! ## The prover's rep -/

/-- **A masked lincheck that returns had its circuit fold return.** -/
theorem lincheckPZk_fold (resp : ℕ → Claims.Msg) (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (h : Array F128)
    (tr : Trace) (zc : ZcOut) (t : Transcript) (l : Array Claims.Msg) {a : (LincheckProof × LcOut) × Transcript}
    {l' : Array Claims.Msg} (e : ((lincheckPZk (asksS resp) c m lay h tr zc).run t).go l = (.ok a, l')) :
    ∃ full, c.fold (coinOf (resp l.size) 0) (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) = .ok full := by
  unfold lincheckPZk at e
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _)] at e
  rw [bind_go (sampleP_go resp _ _)] at e
  rw [bind_go (sampleP_go resp _ _)] at e
  dsimp only at e
  simp only [Array.size_push] at e
  rcases hf : c.fold (coinOf (resp l.size) 0) (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) with err | full
  · rw [hf, StateT.run_bind, FlockComplete.go_bind] at e
    cases e
  · exact ⟨full, rfl⟩

/-- A step that only adds rounds. -/
theorem rec_pre {resp : ℕ → Claims.Msg} {α : Type} {y : TMW PS α} (hy : (JsonRT.recS resp).R y y) (t : Transcript) :
    Ret (y.run t) fun pt => Prefix t.rounds pt.2.rounds := by
  intro l a l' e
  obtain ⟨rs, bs, hrs, -, -⟩ := JsonRT.Rec.go hy e
  exact ⟨rs.toArray, Array.toList_inj.1 (by simp [hrs])⟩

/-- **The rank check, where it returns, keeps the transcript and the sends.** -/
theorem rankCheckP_go {m kLog : ℕ} {prior : Array (List F128)} {words : List ℕ} {t t' : Transcript}
    {l l' : Array Claims.Msg} {u : Unit}
    (h : ((rankCheckP m kLog prior words : TMW PS Unit).run t).go l = (.ok (u, t'), l')) : t' = t ∧ l' = l := by
  unfold rankCheckP at h
  simp only [StateT.run_bind, StateT.run_get, pure_bind] at h
  split at h
  · simp only [StateT.run_pure, JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at h
    exact ⟨h.1.2.symm, h.2.symm⟩
  · change ((throw _ : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at h
    obtain ⟨_, _, h1, -⟩ := JsonRT.go_bind_ok h
    cases h1

/-- **A masked rep that returns** ran zerocheck, lincheck, the rank check and the ring switch on its claims in closed
form: its ring switches are the prover's on those claims, and its final rounds begin with the rounds through the ring
switch. -/
theorem repPZk_rs (resp : ℕ → Claims.Msg) (hsh : ByteArray → ByteArray) (st : Setup) (sch : Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (tr : Trace) (tb : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m) (t : Transcript)
    (l : Array Claims.Msg) {P : Zk.ZkProof} {t' : Transcript} {l' : Array Claims.Msg}
    (h : ((repPZk (asksS resp) (Hashes.ofSha512 hsh) st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt
      nonce prior words).run t).go l = (.ok (P, t'), l')) :
    ∃ full, P.proof.ringSwitches = Rs.proof tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
        (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
          (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras) ∧
      Prefix (Rs.tEnd tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
        (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
          (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras)
        (Rs.coinsAt resp (l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6)) (2 + extras.size))
        (LcZ.tEnd (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h st.fold st.m tr
          (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
          (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))
          (ZcZ.tEnd (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h st.m st.digest (flat cm0.cap) ic.root
            (Zc.coinsAt resp l.size) tr t))).rounds t'.rounds := by
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
  obtain ⟨ht₃, hl₃⟩ := rankCheckP_go hk
  subst t₃ l₃'
  obtain ⟨l₃, -, hrs⟩ := ringSwitchP_go resp tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) (Lc.coinsAt resp l₁.size (st.fold.kLog - 6))) ++
      extras) (by simp [abcClaims]) _ l₂
  rw [bind_go hrs] at h
  refine ⟨full, ?_⟩
  rw [← hl₁, show 2 + extras.size = (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) (Lc.coinsAt resp l₁.size (st.fold.kLog - 6))) ++
      extras).size by simp [abcClaims], ← hl₂]
  refine (?_ : Ret _ fun pt : Zk.ZkProof × Transcript => pt.1.proof.ringSwitches = _ ∧ Prefix _ pt.2.rounds)
    l₃ (P, t') l' h
  refine Replay.Ret.run_bind (rec_pre (JsonRT.ligeritoPZk_rec resp hsh _ _ _ _ _ _ _ _ _ _ _ _ _ _) _)
    fun ⟨lg, zl, rec⟩ s₄ h4 => ?_
  refine Replay.Ret.run_bind (get_ret s₄) fun tr' s₅ ⟨h1, h2⟩ => ?_
  refine Replay.Ret.run_bind (liftV_ret _ s₅) fun ⟨rp, x⟩ s₆ ⟨_, h3⟩ => ?_
  refine Replay.Ret.run_bind (liftV_ret _ s₆) fun lgc s₇ ⟨_, h5⟩ => ?_
  refine Replay.Ret.run_bind (innerP_pre resp hsh _ _ _ _ _ s₇) fun inner s₈ h6 => ?_
  rw [StateT.run_pure]
  refine Ret.pure ⟨rfl, ?_⟩
  dsimp only at h2 h3 h5 ⊢
  rw [h5, h3, h2] at h6
  exact h4.trans h6

/-! ## The ring switch reads its claims' points only -/

namespace Rs

variable {tr : Trace} {c₁ c₂ : Array Claim}

theorem switch_congr (hx : ∀ k : ℕ, c₁[k]!.x = c₂[k]!.x) (k : ℕ) : switchVec tr c₁[k]! = switchVec tr c₂[k]! := by
  unfold switchVec
  rw [hx k]

theorem proof_congr (hn : c₁.size = c₂.size) (hx : ∀ k : ℕ, c₁[k]!.x = c₂[k]!.x) : proof tr c₁ = proof tr c₂ := by
  unfold proof
  rw [hn]
  exact pushes_congr fun k _ => switch_congr hx k

theorem loopT_congr (hx : ∀ k : ℕ, c₁[k]!.x = c₂[k]!.x) (cs : Challenges) (t : Transcript) :
    ∀ k, loopT tr c₁ cs t k = loopT tr c₂ cs t k
  | 0 => rfl
  | k + 1 => by simp only [loopT, absorbed, loopT_congr hx cs t k, switch_congr hx k]

theorem tEnd_congr (hn : c₁.size = c₂.size) (hx : ∀ k : ℕ, c₁[k]!.x = c₂[k]!.x) (cs : Challenges) (t : Transcript) :
    tEnd tr c₁ cs t = tEnd tr c₂ cs t := by
  unfold tEnd
  rw [hn, loopT_congr hx]

end Rs

theorem get_left {A e : Array Claim} {k : ℕ} (hk : k < A.size) : (A ++ e)[k]! = A[k]! := by
  rw [getElem!_pos (A ++ e) k (by simp; omega), getElem!_pos A k hk, Array.getElem_append_left hk]

theorem get_right {A e : Array Claim} {k : ℕ} (hk : A.size ≤ k) : (A ++ e)[k]! = e[k - A.size]! := by
  rw [getElem!_def, getElem!_def, Array.getElem?_append_right hk]

theorem claims_x {A B e : Array Claim} (hA : A.size = 2) (hB : B.size = 2) (h₀ : A[0]!.x = B[0]!.x)
    (h₁ : A[1]!.x = B[1]!.x) : ∀ k : ℕ, (A ++ e)[k]!.x = (B ++ e)[k]!.x
  | 0 => by rw [get_left (by omega), get_left (by omega)]; exact h₀
  | 1 => by rw [get_left (by omega), get_left (by omega)]; exact h₁
  | k + 2 => by rw [get_right (by omega), get_right (by omega), hA, hB]

theorem switchVec_x {tr : Trace} {a b : Claim} (h : a.x = b.x) : switchVec tr a = switchVec tr b := by
  unfold switchVec
  rw [h]

theorem abc_x {m kLog : ℕ} {z₁ z₂ : ZcOut} {l₁ l₂ : LcOut} (hρ : z₁.rho = z₂.rho) (hr : z₁.rRest = z₂.rRest)
    (hi : l₁.rIn = l₂.rIn) :
    (abcClaims m kLog z₁ l₁)[0]!.x = (abcClaims m kLog z₂ l₂)[0]!.x ∧
      (abcClaims m kLog z₁ l₁)[1]!.x = (abcClaims m kLog z₂ l₂)[1]!.x := by
  refine ⟨?_, ?_⟩ <;> simp [abcClaims, hρ, hr, hi]

/-- **The verifier's ring switch returns** after a masked replay that returned on rounds carrying the prover's through
its ring switch, if the extra claims meet their values. -/
theorem rs_of (hs : Hashes V) (st : Setup) (hk : st.fold.kLog = st.kLog) (tr : Trace) (e : Array Claim)
    (lay : Zk.PadLayout) (hp : Array F128) (digest cap root : ByteArray) (cz : Zc.Challenges)
    (full : Array F128) (cl : Lc.Challenges) (crs : Rs.Challenges) (t₀ : Transcript) (h₀ : t₀.k = 0)
    (hs₀ : t₀.rounds.size = 0) (R : Array Round) (rp : Zk.Replayed) (prf : Proof)
    (hR : Agrees R (Rs.tEnd tr (abcClaims st.m st.kLog (Zc.out st.m cz tr)
      (Lc.out st.fold st.m tr (Zc.out st.m cz tr) cl) ++ e) crs
      (LcZ.tEnd lay hp st.fold st.m tr (Zc.out st.m cz tr) full cl
        (ZcZ.tEnd lay hp st.m digest cap root cz tr t₀))).rounds 0)
    (hcl : rp.claims = abcClaims st.m st.kLog (zcAt st.m R 0) (lcAt st.m st.kLog R 0))
    (hsw : prf.ringSwitches = Rs.proof tr (abcClaims st.m st.kLog (Zc.out st.m cz tr)
      (Lc.out st.fold st.m tr (Zc.out st.m cz tr) cl) ++ e))
    (hX : ∀ k < e.size, dot (FlockSoundness.Refine.wOf e[k]!) (switchVec tr e[k]!) = e[k]!.value) :
    ∃ v, (ringSwitchWith hs st.m (ringClaimsZk rp prf ++ e) prf.ringSwitches).run
      ⟨R, 0 + 3 + (st.m - 6) + 3 + (st.kLog - 6), ByteArray.empty⟩ = .ok v := by
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
  have key := ringSwitchWith_of hs st.m tr (ringClaimsZk rp prf ++ e) crs _ hlt
    (by simp only [Array.size_append, hrc]; omega) (by rw [← Rs.tEnd_congr hn hx]; exact hR.mono (Nat.zero_le _)) ?hval
  · rw [LcZ.tEnd_k, ZcZ.tEnd_k, h₀, hk] at key
    rw [hsw, Rs.proof_congr hn hx]
    exact ⟨_, key⟩
  intro k hk'
  have hpc : ∀ j < 2, (abcClaims st.m st.kLog (Zc.out st.m cz tr) (Lc.out st.fold st.m tr (Zc.out st.m cz tr) cl) ++
      e)[j]!.x = rp.claims[j]!.x := by
    intro j hj
    rw [get_left (by rw [hsz]; omega)]
    rcases j with _ | _ | j
    · exact hx₀.symm
    · exact hx₁.symm
    · omega
  rcases k with _ | _ | k
  · rw [get_left (by rw [hrc]; omega)]
    show dot (FlockSoundness.Refine.wOf rp.claims[0]!) (switchVec tr rp.claims[0]!) =
      Zk.claimCheck rp.claims[0]! prf.ringSwitches[0]!
    rw [hsw, Rs.proof_get 0 (by simp only [Array.size_append, hsz]; omega), switchVec_x (hpc 0 (by omega))]
    rfl
  · rw [get_left (by rw [hrc]; omega)]
    show dot (FlockSoundness.Refine.wOf rp.claims[1]!) (switchVec tr rp.claims[1]!) =
      Zk.claimCheck rp.claims[1]! prf.ringSwitches[1]!
    rw [hsw, Rs.proof_get 1 (by simp only [Array.size_append, hsz]; omega), switchVec_x (hpc 1 (by omega))]
    rfl
  · rw [get_right (by rw [hrc]; omega), hrc, show k + 2 - 2 = k by omega]
    refine hX k ?_
    simp only [Array.size_append, hrc] at hk'
    omega

end FlockComplete.Replay

namespace FlockComplete

open Flock Transcript FlockComplete.Replay

variable (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)

/-- **The extra claims' ring-switch algebra**: at every honest masked run, each extra claim's switched vector of the
table's trace meets its value. -/
def RsExtrasZk : Prop :=
  ∀ o : RunZk H rel, ∀ i < o.tabs.size, ∀ k < (o.extras i).size,
    dot (FlockSoundness.Refine.wOf (o.extras i)[k]!) (switchVec (tracesOf o.sts o.w)[i]! (o.extras i)[k]!) =
      (o.extras i)[k]!.value

variable {H rel}

/-- **The masked ring switch replays** (`RingSwitchZk`) at every honest masked run, given the extra claims' algebra. -/
theorem ringSwitchZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : JsonRT.StmtOkZk H rel)
    (hP : JsonRT.ProverOkZk H rel) (hR : JsonRT.RepShapedZk H rel) (hX : RsExtrasZk H rel) : RingSwitchZk H rel := by
  intro o i hi r hr
  have hRp := replay_goal S h64 hS hP hR o i hi r hr
  obtain ⟨st, v, cm, e, streams, htabs, hsch, hss, hbytes, hstream, hext, h25, hrep⟩ :=
    JsonRT.RunZk.walk S h64 hS hP hR o
  have hi₀ := hi
  rw [htabs] at hi₀
  obtain rfl : i = 0 := by simpa using hi₀
  obtain ⟨p, ic, l, t', l', prior, ws, -, hzp, hst', hrun⟩ := hrep r hr
  have hT : o.tabs[0]! = Setup.ofCircuit st := by rw [htabs]; rfl
  have hm : 14 ≤ (Setup.ofCircuit st).m := by show 14 ≤ st.m; omega
  obtain ⟨full, hsw, hpre⟩ := repPZk_rs (hm := hm) (h := hrun)
  have hAg : Agrees (o.stream 0 r).rounds t'.rounds 0 := by
    unfold Agrees
    intro j hj _
    have hmem : t'.rounds[j] ∈ streams[r]! := by
      rw [hst']
      exact Array.getElem_mem hj
    obtain ⟨c, hc⟩ := hbytes _ (by rw [getElem!_pos streams r (by omega)]; exact Array.getElem_mem _) _ hmem
    refine ⟨JsonRT.roundRec o.spec (Claims.hOf H) t'.rounds[j], ?_, rfl, rfl, by simp [JsonRT.roundRec, hc]⟩
    rw [hstream r hr, hst']
    simp [hj]
  refine Replay.ok_val ?_
  have hm' : 14 ≤ o.tabs[0]!.m := by rw [hT]; exact hm
  generalize o.rp 0 r = q at hRp ⊢
  obtain ⟨ht, hcl⟩ := replayWith_ret o.hs o.tabs[0]! hm' (o.lay 0) (o.zp 0 r).proof (o.zp 0 r).inner.root
    (o.stream 0 r).rounds 0 _ q.1 q.2 hRp
  rw [ht, hext, hzp, hT]
  rw [hT] at hcl
  refine rs_of (hs := o.hs) (st := Setup.ofCircuit st) (hk := rfl) (h₀ := rfl) (hs₀ := rfl)
    (hR := hAg.of_prefix hpre) (hcl := hcl) (hsw := hsw) (hX := fun k hk => ?_)
  have := hX o 0 hi k (by rw [hext]; exact hk)
  rwa [hext] at this

end FlockComplete
