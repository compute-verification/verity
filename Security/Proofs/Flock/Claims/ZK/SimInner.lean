import Proofs.Flock.Claims.ZK.Split
import Proofs.Flock.Claims.Complete.Zk.Exact

/-!
# Hybrid 2 against hybrid 3: a rep's inner proof from the cut (`SimInner`)

`repMix 2` and `repMix 3` share every phase up to the inner proof (`innerPre`, `repMix_eq`). There `repMix 2` runs
`innerCut` on the cut's source and `repMix 3` the session's `innerP` on the honest pads. Both absorb the label and
`final_c`, then send four rounds, `ε`, the batching coins, `β` and the query coins, each answered by `asksCut`
(`squeezeCut_go`, `sampleCut_go`): the send is recorded, then refused at or past round `n`, else answered by its
round's coin.

* **`innerCut_alike`**: where the source carries what the session absorbs at each round the session reaches
  (`InnerSrc`), the two inner proofs send the same bodies, leave the same transcript and agree up to `eraseIP`.
  `InnerSrc` asks for:
  - the triple's openings `ρ`, `σ` once `ε` is answered;
  - the batched constraint at the honest pads, and `τ`'s commitment, once the batching coins are;
  - `Y` once `β` is, where `β ≠ 0`.
* **`simInner_of`**: `SimInner` from `InnerSrcE`, the deployed instance of `InnerSrc` at the cut's source, wherever the
  shared phases reach the inner proof.
* **`innerPre_slot`**: where they do, the inner proof begins at `repSlot … (innerOff 0)`, the round the cut reads `ε`
  at: `innerPre_size` counts the shared phases' sends at `asksCut` through the exact counts at `asksS` (`okS`).
* **`innerSrcE_triple`**: `InnerSrcE`'s `triple` there: the cut's `inner r` is the session's `ρ_in`, `σ_in` at the
  honest pads (`innerTripleE`).
* **`innerYE`**: wherever the cut sent `Y₁`, its `Y` read in `yAt`'s order is the session's `h ‖ pad_h + β·(μ ‖ pad_μ)`
  at the honest pads, `β` the coin two rounds after `ε`'s.
-/

namespace FlockComplete.SimInner

open Flock Transcript Replay

/-! ## Steps in the pure session -/

/-- A bind whose first step both sides share: the continuations need only agree where that step returns. -/
theorem go_bind_congr_r {α β : Type} {x : TMW PS α} {f g : α → TMW PS β} {t : Transcript} {l : Array Claims.Msg}
    (h : ∀ a t' l', (x.run t).go l = (.ok (a, t'), l') → ((f a).run t').go l' = ((g a).run t').go l') :
    ((x >>= f).run t).go l = ((x >>= g).run t).go l := by
  rw [StateT.run_bind, StateT.run_bind, go_bind, go_bind]
  match hx : (x.run t).go l with
  | (.ok (a, t'), l') => exact h a t' l' hx
  | (.error e, l') => rfl

theorem ite_ok {α τ σ : Type} {c : Prop} [Decidable c] {e : String} {v a : α} {T T' : τ} {L L' : σ}
    (h : ((if c then .error e else .ok (v, T) : Except String (α × τ)), L) = (.ok (a, T'), L')) :
    ¬ c ∧ v = a ∧ T = T' ∧ L = L' := by
  by_cases hc : c
  · simp [hc] at h
  · simp only [hc, ↓reduceIte, Prod.mk.injEq, Except.ok.injEq] at h
    exact ⟨hc, h.1.1, h.1.2, h.2⟩

theorem liftTM_ok_log {α : Type} {x : TM α} {t t' : Transcript} {a : α} {l l' : Array Claims.Msg}
    (h : ((liftTM x : TMW PS α).run t).go l = (.ok (a, t'), l')) : l = l' := by
  rw [liftTM_go] at h
  exact (Prod.mk.inj h).2

/-- **A send answered by `asksCut`**: recorded, then refused at or past round `n`, else answered by round `l.size`'s
coin. -/
theorem cutRound_go (coin : ℕ → ℕ → F128) (n : ℕ) (c : ByteArray) (l : Array Claims.Msg) :
    ((asksCut coin n).round c).go l =
      (if n ≤ l.size then .error "past the openings" else .ok (coin l.size), l.push (Claims.toMsg c)) := by
  have e : ((asksCut coin n).round c).go l =
      (if n ≤ l.size then (throw "past the openings" : PS (ℕ → F128)) else pure (coin l.size)).go
        (l.push (Claims.toMsg c)) := rfl
  rw [e]
  by_cases hn : n ≤ l.size <;> simp only [hn, ↓reduceIte] <;> rfl

theorem squeezeCut_go (coin : ℕ → ℕ → F128) (n : ℕ) (kind : UInt8) {m : ℕ} (hm : m ≠ 0) (t : Transcript)
    (l : Array Claims.Msg) :
    ((squeezeP (asksCut coin n) kind m).run t).go l =
      (if n ≤ l.size then .error "past the openings"
        else .ok ((Array.range m).map (coin l.size), afterSq t kind m (coin l.size)),
        l.push (Claims.toMsg (framed t kind m))) := by
  simp only [squeezeP, StateT.run, hm, ite_false]
  rw [go_bind, cutRound_go]
  by_cases hn : n ≤ l.size
  · simp only [hn, ↓reduceIte]
  · simp only [hn, ↓reduceIte]
    rfl

theorem sampleVecCut_go (coin : ℕ → ℕ → F128) (n : ℕ) {m : ℕ} (hm : m ≠ 0) (t : Transcript)
    (l : Array Claims.Msg) :
    ((sampleVecP (asksCut coin n) m).run t).go l =
      (if n ≤ l.size then .error "past the openings"
        else .ok ((Array.range m).map (coin l.size), afterSq t 2 m (coin l.size)),
        l.push (Claims.toMsg (framed t 2 m))) :=
  squeezeCut_go coin n 2 hm t l

theorem sampleCut_go (coin : ℕ → ℕ → F128) (n : ℕ) (t : Transcript) (l : Array Claims.Msg) :
    ((sampleP (asksCut coin n)).run t).go l =
      (if n ≤ l.size then .error "past the openings" else .ok (coin l.size 0, afterSq t 1 1 (coin l.size)),
        l.push (Claims.toMsg (framed t 1 1))) := by
  by_cases hn : n ≤ l.size
  · unfold sampleP
    rw [StateT.run_bind, go_bind, squeezeCut_go coin n 1 one_ne_zero]
    simp only [hn, ↓reduceIte]
  · have h1 := squeezeCut_go coin n 1 one_ne_zero t l
    simp only [hn, ↓reduceIte] at h1
    unfold sampleP
    rw [bind_go h1]
    simp only [hn, ↓reduceIte]
    show (Except.ok (_, _), _) = _
    simp

/-! ## The inner proof from the source against the session's -/

/-- The inner proof's coins at `asksCut`'s rounds `L` to `L + 3`. -/
def cutCoins (coin : ℕ → ℕ → F128) (L : ℕ) : InZ.Challenges := ⟨coin L, coin (L + 1), coin (L + 2), coin (L + 3)⟩

/-- **What the source carries for an inner proof begun after `L` bodies**, against the honest pads `p` and the
constraints `cons`, at the coins `asksCut coin n` answers (round `k < n`). Each holds where its round is answered:
* the triple's openings `ρ`, `σ` once `ε` (round `L`) is;
* once the batching coins (round `L + 1`) are, the batched constraint at the pads, and `τ`'s commitment through `h`;
* `Y` once `β` (round `L + 2`) is, and is not `0`. -/
structure InnerSrc (coin : ℕ → ℕ → F128) (n : ℕ) (h : ByteArray → ByteArray) (lay : Zk.PadLayout) (p : PadsZk)
    (cons : Array (String × Zk.Lin)) (src : RepSrc) (L : ℕ) : Prop where
  triple : L < n → src.rhoIn = InZ.rho lay p (cutCoins coin L) ∧ src.sigmaIn = InZ.sigma lay p
  batched : L + 1 < n →
    ((InZ.lin lay p cons (cutCoins coin L)).c + dot (InZ.lin lay p cons (cutCoins coin L)).a p.h).isZero = true
  tauCom : L + 1 < n → src.tauCom = InZ.tcOf h (InZ.tau lay p cons (cutCoins coin L)) p.tauSalt
  y : L + 2 < n → (coin (L + 2) 0).isZero = false → src.y = InZ.y p (cutCoins coin L)

/-- **The inner proof from the source sends what the session's sends**: at `asksCut`, where the source carries the
session's values (`InnerSrc`), `innerCut` and `innerP` send the same bodies, leave the same transcript, and return inner
proofs equal up to `eraseIP`, so any continuation that reads its inner proof through `eraseIP` agrees. -/
theorem innerCut_alike {β : Type} {coin : ℕ → ℕ → F128} {n : ℕ} {h : ByteArray → ByteArray} {lay : Zk.PadLayout}
    {p : PadsZk} {ic : InnerCommit} {finalC : F128} {src : RepSrc} {cons : Array (String × Zk.Lin)}
    {f : Zk.InnerProof → TMW PS β} (hf : ∀ i i', eraseIP i = eraseIP i' → f i = f i') {t : Transcript}
    {l : Array Claims.Msg} (hsrc : InnerSrc coin n h lay p cons src l.size) :
    ((innerCut (asksCut coin n) lay ic finalC src cons >>= f).run t).go l =
      ((innerP (asksCut coin n) (Hashes.ofSha512 h) lay p ic finalC cons >>= f).run t).go l := by
  obtain ⟨hrs, hbat, htc, hy⟩ := hsrc
  unfold innerCut innerP
  simp only [bind_assoc]
  refine go_bind_congr_r fun _ t₁ l₁ h₁ => ?_
  obtain rfl := liftTM_ok_log h₁
  refine go_bind_congr_r fun _ t₂ l₂ h₂ => ?_
  obtain rfl := liftTM_ok_log h₂
  refine go_bind_congr_r fun eps t₃ l₃ h₃ => ?_
  obtain ⟨hn₀, rfl, rfl, rfl⟩ := ite_ok ((sampleCut_go coin n t₂ l).symm.trans h₃)
  obtain ⟨hρ, hσ⟩ := hrs (by omega)
  simp only [InZ.rho, InZ.sigma, cutCoins] at hρ hσ
  rw [hρ, hσ]
  refine go_bind_congr_r fun _ t₄ l₄ h₄ => ?_
  obtain rfl := liftTM_ok_log h₄
  refine go_bind_congr_r fun gs t₅ l₅ h₅ => ?_
  obtain ⟨hn₁, rfl, rfl, rfl⟩ := ite_ok ((sampleVecCut_go coin n (by simp [Zk.triple]) t₄ _).symm.trans h₅)
  simp only [Array.size_push] at hn₁ ⊢
  have hz := hbat (by omega)
  have hτ := htc (by omega)
  simp only [InZ.lin, InZ.all, InZ.tau, InZ.rho, InZ.sigma, cutCoins] at hz hτ
  simp only [hz, Bool.not_true, Bool.false_eq_true, ↓reduceIte, bind_assoc]
  rw [tauCommitmentWith_ofSha512]
  rw [bind_go (x := StateT.lift (pure _)) rfl]
  rw [hτ]
  refine go_bind_congr_r fun _ t₆ l₆ h₆ => ?_
  obtain rfl := liftTM_ok_log h₆
  refine go_bind_congr_r fun beta t₇ l₇ h₇ => ?_
  obtain ⟨hn₂, rfl, rfl, rfl⟩ := ite_ok ((sampleCut_go coin n t₆ _).symm.trans h₇)
  simp only [Array.size_push, Nat.add_assoc, Nat.reduceAdd] at hn₂ ⊢
  by_cases hb : (coin (l.size + 2) 0).isZero = true
  · simp only [hb, ↓reduceIte, bind_assoc]
    rfl
  · have hY := hy (by omega) (by simpa using hb)
    simp only [InZ.y, cutCoins] at hY
    simp only [hb, Bool.false_eq_true, ↓reduceIte, bind_assoc]
    rw [hY]
    refine go_bind_congr_r fun _ t₈ l₈ h₈ => ?_
    obtain rfl := liftTM_ok_log h₈
    refine go_bind_congr_r fun _ t₉ l₉ _ => ?_
    simp only [pure_bind]
    exact congrArg (fun z : TMW PS β => (z.run t₉).go l₉) (hf _ _ rfl)

/-! ## The phases hybrids 2 and 3 share -/

/-- **The phases `repMix k` shares for every `k ≥ 2`**, up to its inner proof: the session's PIOP and Ligerito
(level 0's opening erased), the replay and the constraints. Returns `final_c`, the proof, level 0's messages and the
constraints. -/
def innerPre (io : Asks PS) (hs : Hashes PS) (st : Setup) (sch : Schedule) (tPad rep : Nat) (dom : ByteArray)
    (tr : Trace) (t : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : Nat → Nat → F128) (salt : Nat → Nat → ByteArray) (nonce : Nat → Nat → UInt64)
    (prior : Array (List F128)) (words : List Nat) :
    TMW PS (F128 × Proof × Zk.Level0Msgs × Array (String × Zk.Lin)) := do
  let lay := Zk.PadLayout.new st.m st.kLog sch.k0
  let (zp, lp, ss, T, sws) ← piopPZk io st lay pads.h tr (flat cm0.cap) ic.root extras prior words
  let (lg, zl, rec) ← do
      let (lg, zl, rec) ← ligeritoPZk io hs st.merkle sch T cm0 t (basisVals sws (st.m - 7)) lay pads.h rep tPad extra
        lanePad salt nonce
      pure (eraseL lg, zl, rec)
  let p : Proof := { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp,
                     ringSwitches := ss, ligerito := lg }
  let tr' ← get
  let (rp, _) ← StateT.lift (monadLift ((Zk.replay st lay p ic.root).run (Transcript.init tr'.rounds dom)) :
    PS (Zk.Replayed × Transcript))
  let ccAb := Zk.claimCheck rp.claims[0]! ss[0]!
  let ccC := Zk.claimCheck rp.claims[1]! ss[1]!
  let lgc ← StateT.lift (monadLift (Zk.ligeritoConstraints lay rec lg zl) : PS (Array (String × Zk.Lin)))
  return (zp.finalC, p, zl,
    rp.cons ++ #[("ring switch ab", rp.abValue.addConst ccAb), ("ring switch c", rp.cValue.addConst ccC)] ++ lgc)

/-- **`repMix k` for `k ≥ 2`**: the shared phases (`innerPre`), then the inner proof, from the source at `k = 2` and the
session's at `k ≥ 3`. -/
theorem repMix_eq (k : ℕ) (hk : 2 ≤ k) (io : Asks PS) (hs : Hashes PS) (st : Setup) (sch : Schedule) (tPad rep : Nat)
    (dom : ByteArray) (tr : Trace) (t : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk)
    (ic : InnerCommit) (extra lanePad : Nat → Nat → F128) (salt : Nat → Nat → ByteArray)
    (nonce : Nat → Nat → UInt64) (prior : Array (List F128)) (words : List Nat) (src : RepSrc) :
    repMix k io hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior words src =
      innerPre io hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior words >>= fun o =>
        (if k ≤ 2 then innerCut io (Zk.PadLayout.new st.m st.kLog sch.k0) ic o.1 src o.2.2.2
          else innerP io hs (Zk.PadLayout.new st.m st.kLog sch.k0) pads ic o.1 o.2.2.2) >>= fun inner =>
        pure (eraseZ { proof := o.2.1, level0 := o.2.2.1, inner }) := by
  unfold repMix innerPre
  simp only [show k ≠ 0 by omega, show ¬ k ≤ 1 by omega, ↓reduceIte, bind_assoc]
  refine bind_congr fun ⟨zp, lp, ss, T, sws⟩ => ?_
  simp only [pure_bind]
  refine bind_congr fun ⟨lg, zl, rec⟩ => ?_
  dsimp only
  refine bind_congr fun tr' => ?_
  refine bind_congr fun ⟨rp, _⟩ => ?_
  refine bind_congr fun lgc => ?_
  by_cases hk2 : k ≤ 2 <;> simp only [hk2, ↓reduceIte]

/-! ## How many bodies the shared phases send

`asksCut` answers what `asksS` would where it answers at all, so an ok run at `asksCut` is the same run at `asksS` on
responses that carry the cut's coins (`okS`), and the exact counts of `Exact` carry over. -/

section Slots

open With

/-- **Where the left run returns, the right returns the same**, with the same log. -/
def okS : With.Sim PS PS where
  R x y := ∀ l a l', x.go l = (.ok a, l') → y.go l = (.ok a, l')
  pure _ _ _ _ h := h
  bind {α β x y f g} hxy hfg l b l'' h := by
    obtain ⟨a, l', h1, h2⟩ := JsonRT.go_bind_ok h
    rw [go_bind, hxy _ _ _ h1]
    exact hfg a _ _ _ h2
  throw _ _ _ _ h := by cases h
  lift _ _ _ _ h := h

theorem okS_refl {α : Type} (x : PS α) : okS.R x x := fun _ _ _ h => h

theorem okS_refl_st {α σ : Type} (x : StateT σ PS α) : (okS.st σ).R x x := fun _ => okS_refl _

theorem okS_refl_st2 {α σ τ : Type} (x : StateT τ (StateT σ PS) α) : ((okS.st σ).st τ).R x x :=
  fun _ => okS_refl_st _

theorem liftT_ok {α : Type} {x y : TMW PS α} (h : (okS.st Transcript).R x y) :
    ((okS.st Transcript).st LigState).R (liftT x) (liftT y) :=
  Sim.lift_st _ h

variable (coin : ℕ → ℕ → F128) (n : ℕ) (resp : ℕ → Claims.Msg) (hc : ∀ k, coinOf (resp k) = coin k)
include hc

/-- A round `asksCut` answers, `asksS` answers alike. -/
theorem okS_round (c : ByteArray) : okS.R ((asksCut coin n).round c) ((asksS resp).round c) := by
  intro l a l' h
  have e0 : ((asksCut coin n).round c).go l =
      (if n ≤ l.size then (throw "past the openings" : PS (ℕ → F128)) else pure (coin l.size)).go
        (l.push (Claims.toMsg c)) := rfl
  rw [e0] at h
  by_cases hn : n ≤ l.size
  · rw [ite_eq_left hn] at h
    cases h
  · rw [ite_eq_right hn] at h
    change (Except.ok (coin l.size), l.push (Claims.toMsg c)) = _ at h
    obtain ⟨⟨⟩, rfl⟩ := Prod.mk.inj h
    show (Except.ok (coinOf (resp l.size)), l.push (Claims.toMsg c)) = _
    rw [hc]

theorem squeezeP_ok (kind : UInt8) (m : ℕ) :
    (okS.st Transcript).R (squeezeP (asksCut coin n) kind m) (squeezeP (asksS resp) kind m) := by
  intro t
  unfold squeezeP
  sim_walk [okS_round coin n resp hc]

theorem sampleP_ok : (okS.st Transcript).R (sampleP (asksCut coin n)) (sampleP (asksS resp)) := by
  unfold sampleP
  sim_walk [squeezeP_ok coin n resp hc]

theorem sampleKP_ok : (okS.st Transcript).R (sampleKP (asksCut coin n)) (sampleKP (asksS resp)) := by
  unfold sampleKP
  sim_walk [squeezeP_ok coin n resp hc]

theorem zerocheckPZk_ok (m : ℕ) (digest cap root : ByteArray) (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) :
    (okS.st Transcript).R (zerocheckPZk (asksCut coin n) m digest cap root lay h tr)
      (zerocheckPZk (asksS resp) m digest cap root lay h tr) := by
  unfold zerocheckPZk sampleVecP
  sim_walk [squeezeP_ok coin n resp hc, sampleP_ok coin n resp hc, okS_refl_st]

theorem lincheckPZk_ok (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) (zc : ZcOut) :
    (okS.st Transcript).R (lincheckPZk (asksCut coin n) c m lay h tr zc)
      (lincheckPZk (asksS resp) c m lay h tr zc) := by
  unfold lincheckPZk
  sim_walk [sampleP_ok coin n resp hc, okS_refl_st]

theorem ringSwitchP_ok (tr : Trace) (claims : Array Claim) :
    (okS.st Transcript).R (ringSwitchP (asksCut coin n) tr claims) (ringSwitchP (asksS resp) tr claims) := by
  unfold ringSwitchP sampleVecP
  sim_walk [squeezeP_ok coin n resp hc, okS_refl_st]

theorem piopPZk_ok (st : Setup) (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) (cap root : ByteArray)
    (extras : Array Claim) (prior : Array (List F128)) (words : List ℕ) :
    (okS.st Transcript).R (piopPZk (asksCut coin n) st lay h tr cap root extras prior words)
      (piopPZk (asksS resp) st lay h tr cap root extras prior words) := by
  unfold piopPZk
  sim_walk [zerocheckPZk_ok coin n resp hc, lincheckPZk_ok coin n resp hc, ringSwitchP_ok coin n resp hc, okS_refl_st]

theorem challengeP_ok (split : Bool) :
    ((okS.st Transcript).st LigState).R (challengeP (asksCut coin n) split) (challengeP (asksS resp) split) := by
  unfold challengeP
  sim_walk [liftT_ok, sampleKP_ok coin n resp hc, okS_refl_st2]

theorem oodIntroP_ok (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ) :
    ((okS.st Transcript).st LigState).R (oodIntroP (asksCut coin n) nonce g dim)
      (oodIntroP (asksS resp) nonce g dim) := by
  unfold oodIntroP sampleVecP
  sim_walk [liftT_ok, squeezeP_ok coin n resp hc, sampleP_ok coin n resp hc, okS_refl_st2]

theorem queryP_ok (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit) :
    ((okS.st Transcript).st LigState).R (queryP (asksCut coin n) nonce ms lv cm)
      (queryP (asksS resp) nonce ms lv cm) := by
  unfold queryP sampleVecP
  sim_walk [liftT_ok, squeezeP_ok coin n resp hc, okS_refl_st2]

theorem ligeritoPZk_ok (hs : Hashes PS) (ms : MerkleScheme) (sch : Schedule) (T : F128) (cm0 : Commit)
    (t b : Array F128) (lay : Zk.PadLayout) (h : Array F128) (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    (okS.st Transcript).R (ligeritoPZk (asksCut coin n) hs ms sch T cm0 t b lay h rep tPad extra lanePad salt nonce)
      (ligeritoPZk (asksS resp) hs ms sch T cm0 t b lay h rep tPad extra lanePad salt nonce) := by
  unfold ligeritoPZk sampleVecP
  sim_walk [liftT_ok, squeezeP_ok coin n resp hc, sampleP_ok coin n resp hc, sampleKP_ok coin n resp hc,
    challengeP_ok coin n resp hc, oodIntroP_ok coin n resp hc, queryP_ok coin n resp hc, okS_refl, okS_refl_st,
    okS_refl_st2]

theorem innerPre_ok (hs : Hashes PS) (st : Setup) (sch : Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (tr : Trace) (tb : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List F128)) (words : List ℕ) :
    (okS.st Transcript).R
      (innerPre (asksCut coin n) hs st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt nonce prior words)
      (innerPre (asksS resp) hs st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt nonce prior words) := by
  unfold innerPre
  sim_walk [piopPZk_ok coin n resp hc, ligeritoPZk_ok coin n resp hc, okS_refl, okS_refl_st]

omit hc in
/-- **The PIOP's exact send count at `asksS`**: `piopZk`. -/
theorem piopPZk_exact (st : Setup) (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) (cap root : ByteArray)
    (extras : Array Claim) (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m) {t t' : Transcript}
    {l l' : Array Claims.Msg} {o : ZerocheckProof × LincheckProof × Array (Array F128) × F128 × Array Switched}
    (hrun : ((piopPZk (asksS resp) st lay h tr cap root extras prior words).run t).go l = (.ok (o, t'), l')) :
    l'.size = l.size + piopZk st extras := by
  obtain ⟨l₁, hl₁, hz⟩ := zerocheckPZk_go resp hm st.digest cap root lay h tr t l
  unfold piopPZk at hrun
  rw [bind_go hz] at hrun
  dsimp only at hrun
  have h' := hrun
  rw [StateT.run_bind] at h'
  obtain ⟨_, _, h₁, -⟩ := JsonRT.go_bind_ok h'
  obtain ⟨full, hfold⟩ := lincheckPZk_fold resp _ _ _ _ _ _ _ _ h₁
  obtain ⟨l₂, hl₂, hl⟩ := lincheckPZk_go resp st.fold st.m lay h tr _ full _ l₁ hfold
  rw [bind_go hl] at hrun
  dsimp only at hrun
  rw [StateT.run_bind] at hrun
  obtain ⟨⟨u, t₃⟩, l₃', hk, hrun⟩ := JsonRT.go_bind_ok hrun
  obtain ⟨ht₃, hl₃⟩ := rankCheckP_go hk
  subst t₃ l₃'
  obtain ⟨l₃, hl₃, hrs⟩ := ringSwitchP_go resp tr (abcClaims st.m st.kLog _ _ ++ extras) (by simp [abcClaims]) _ l₂
  rw [bind_go hrs] at hrun
  obtain ⟨-, rfl⟩ := Prod.mk.inj hrun
  have hsz : (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) (Lc.coinsAt resp l₁.size (st.fold.kLog - 6))) ++
      extras).size = 2 + extras.size := by simp [abcClaims, Nat.add_comm]
  unfold piopZk
  omega

omit hc in
/-- **The shared phases' exact send count at `asksS`**: the PIOP's `piopZk`, then Ligerito's `k₀ + 7 + 11 R`; the
replay and the constraints send nothing. -/
theorem innerPre_exact (hsh : ByteArray → ByteArray) (st : Setup) (sch : Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (tr : Trace) (tb : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m) (hn : sch.n ≠ 0) (hk0 : 1 ≤ sch.k0)
    (hL : 2 ≤ sch.levels.size) (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0) {t t' : Transcript}
    {l l' : Array Claims.Msg} {o : F128 × Proof × Zk.Level0Msgs × Array (String × Zk.Lin)}
    (h : ((innerPre (asksS resp) (Hashes.ofSha512 hsh) st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt
      nonce prior words).run t).go l = (.ok (o, t'), l')) :
    l'.size = l.size + piopZk st extras + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) := by
  unfold innerPre at h
  rw [StateT.run_bind] at h
  obtain ⟨⟨⟨zp, lp, ss, T, sws⟩, t₁⟩, l₁, h1, h⟩ := JsonRT.go_bind_ok h
  have e1 := piopPZk_exact resp st _ _ tr _ _ extras prior words hm h1
  dsimp only at h
  rw [StateT.run_bind] at h
  obtain ⟨⟨⟨lg, zl, rec⟩, t₂⟩, l₂, h2, h⟩ := JsonRT.go_bind_ok h
  have e2 := (ligeritoPZk_exact resp hsh st.merkle sch _ cm0 _ _ _ _ rep tPad extra lanePad salt nonce hn hk0 hL hq
    h2).1
  dsimp only at h
  rw [bind_go (pure_go _ t₂ l₂)] at h
  dsimp only at h
  rw [bind_go (show ((get : TMW PS Transcript).run t₂).go l₂ = (.ok (t₂, t₂), l₂) from rfl)] at h
  obtain ⟨⟨rp, x⟩, -, h⟩ := liftV_bind_ok h
  dsimp only at h
  obtain ⟨lgc, -, h⟩ := liftV_bind_ok h
  obtain ⟨-, rfl⟩ := Prod.mk.inj h
  omega

/-- **The PIOP's send count at `asksCut`**, where it returns: `piopZk`. -/
theorem piopPZk_size (st : Setup) (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) (cap root : ByteArray)
    (extras : Array Claim) (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m) {t t' : Transcript}
    {l l' : Array Claims.Msg} {o : ZerocheckProof × LincheckProof × Array (Array F128) × F128 × Array Switched}
    (hrun : ((piopPZk (asksCut coin n) st lay h tr cap root extras prior words).run t).go l = (.ok (o, t'), l')) :
    l'.size = l.size + piopZk st extras :=
  piopPZk_exact resp st lay h tr cap root extras prior words hm
    (piopPZk_ok coin n resp hc st lay h tr cap root extras prior words t l _ _ hrun)

/-- **The shared phases' send count at `asksCut`**, where they return: `piopZk`, then Ligerito's `k₀ + 7 + 11 R`. -/
theorem innerPre_size (hsh : ByteArray → ByteArray) (st : Setup) (sch : Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (tr : Trace) (tb : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m) (hn : sch.n ≠ 0) (hk0 : 1 ≤ sch.k0)
    (hL : 2 ≤ sch.levels.size) (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0) {t t' : Transcript}
    {l l' : Array Claims.Msg} {o : F128 × Proof × Zk.Level0Msgs × Array (String × Zk.Lin)}
    (h : ((innerPre (asksCut coin n) (Hashes.ofSha512 hsh) st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad
      salt nonce prior words).run t).go l = (.ok (o, t'), l')) :
    l'.size = l.size + piopZk st extras + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) :=
  innerPre_exact resp hsh st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt nonce prior words hm hn hk0
    hL hq (innerPre_ok coin n resp hc _ st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt nonce prior
      words t l _ _ h)

end Slots

end FlockComplete.SimInner

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- Table `j`'s pads layout at the deployed instance. -/
noncomputable abbrev innerLayE (x : Msg) (j : Fin (jOf x)) : _root_.Flock.Zk.PadLayout :=
  _root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog
    (zkTab x j).sch.k0

/-- Table `j`'s rep `r` honest pads at the tape `τ`. -/
noncomputable abbrev innerPadsE (x : Msg) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) : FlockComplete.PadsZk :=
  FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (innerLayE x j)

/-- **The phases hybrids 2 and 3 share** at the deployed instance (`mixE`'s arguments). -/
noncomputable def innerPreE (x : Msg) (w : Claims.Flock.Wit) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin)
    (n : ℕ) (j : Fin (jOf x)) (r : Fin 2) (dom : ByteArray) (cm0 : FlockComplete.Commit)
    (extras : Array _root_.Flock.Claim) (ic : FlockComplete.InnerCommit) :=
  FlockComplete.SimInner.innerPre (ioE c n) hsE
    (FlockDeploy.tabsOf (stsOf x))[j.val]! (zkTab x j).sch (zkTab x j).tPad r.val dom
    (trsE x w (UE.tapeZk τ))[j.val]!
    (FlockComplete.packed (trsE x w (UE.tapeZk τ))[j.val]! (FlockDeploy.tabsOf (stsOf x))[j.val]!.m) cm0 extras
    (innerPadsE x τ j r) ic
    ((UE.tapeZk τ).extra j.val) ((UE.tapeZk τ).lanePad j.val) ((UE.tapeZk τ).salt j.val r.val)
    ((UE.tapeZk τ).nonce j.val r.val) (priorE c j.val r.val) (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!

/-- The opened coins as the session's responses: round `k`'s coin's bytes, and a zero coin's past the opened rounds. -/
noncomputable def respE {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (k : ℕ) : Msg :=
  if h : k < refK x then (c ⟨k, h⟩).bytes else (default : FlockDeploy.Coin).bytes

theorem coinOf_respE {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (k : ℕ) :
    FlockComplete.coinOf (respE c k) = fun i => slotAt c k i := by
  funext i
  by_cases h : k < refK x
  · simp only [respE, slotAt, h, ↓reduceDIte, FlockComplete.JsonRT.coinOf_bytes]
  · simp only [respE, slotAt, h, ↓reduceDIte]
    rw [FlockComplete.JsonRT.coinOf_bytes]
    unfold FlockDeploy.Coin.at
    split <;> rfl

/-- **Where the shared phases leave table `j`'s rep `r`**: begun at the rep's base, a run of `innerPreE` that returns
has sent the rep's PIOP and Ligerito, so its inner proof begins at `repSlot … (innerOff 0)`, the round of `ε` the cut
reads. -/
theorem innerPre_slot {x : Msg} {s spec} (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (w : Claims.Flock.Wit) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) (j : Fin (jOf x)) (r : Fin 2)
    (dom : ByteArray) (cm0 : FlockComplete.Commit) (ic : FlockComplete.InnerCommit)
    (extras : Array _root_.Flock.Claim)
    (he : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    {l : Array Msg} (hl : l.size = repBase x (2 * j.val + r.val)) {o t' l'}
    (ho : ((innerPreE x w τ c n j r dom cm0 extras ic).run (_root_.Flock.Transcript.init #[] dom)).go l =
      (.ok (o, t'), l')) :
    l'.size = repSlot x (2 * j.val + r.val) (innerOff 0) := by
  have hj := lt_of_jOf j
  have hr := r.isLt
  have hst : (stsOf x)[j.val]! = stAt x j := getElem!_pos (stsOf x) j.val hj
  have htab : (FlockDeploy.tabsOf (stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (stAt x j) := by
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) j.val (by simpa [FlockDeploy.tabsOf] using hj)]
    simp [FlockDeploy.tabsOf]
    rfl
  have hz := (zkTab x j).hz
  obtain ⟨h25, -, hsm, hk0, -, ⟨hn7, h2, -⟩, hlv, -, -⟩ := ZkExec.zk_schedule_seams _ _ _ hz
  have hk6 := (zkTab x j).wf.1.hk6
  have hx : extras.size = spec.points * (stAt x j).regions.size := by
    rw [htab] at he
    rw [FlockDeploy.extraClaims_size (st := stAt x j) he, Nat.mul_comm]
    simp [ptsE]
  unfold innerPreE ioE at ho
  have hsz := FlockComplete.SimInner.innerPre_size (fun k i => slotAt c k i) n (respE c) (coinOf_respE c)
    _root_.Flock.Sha512.hash _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ (by rw [htab]; show 14 ≤ (stAt x j).m; omega)
    (by omega) hk0 h2 (fun i hi => (hlv i hi).q_ne_zero) ho
  rw [htab] at hsz
  unfold repSlot
  rw [hs]
  simp only [show (2 * j.val + r.val) / 2 = j.val by omega, hst, hz]
  rw [hsz, hl, repBase_eq hs]
  unfold FlockComplete.piopZk FlockComplete.piopRoundsZk innerOff
  have hm' : (_root_.Flock.Setup.ofCircuit (stAt x j)).m = (stAt x j).m := rfl
  have hk' : (_root_.Flock.Setup.ofCircuit (stAt x j)).fold.kLog = (stAt x j).c.kLog := rfl
  rw [hm', hk', hx]
  omega

theorem val_mul_add (a b d : _root_.Flock.F128) :
    FlockLevel3.GF128.val (FlockSoundness.Refine.fOf a * FlockSoundness.Refine.fOf b + FlockSoundness.Refine.fOf d) =
      a * b + d := by
  rw [FlockLevel3.GF128.mul_eq, FlockLevel3.GF128.add_eq]
  rfl

theorem val_add_of (b d : _root_.Flock.F128) :
    FlockLevel3.GF128.val (FlockSoundness.Refine.fOf b + FlockSoundness.Refine.fOf d) = b + d := by
  rw [FlockLevel3.GF128.add_eq]
  rfl

/-- The tape's pad `k` of table `i`'s rep `r` is the cell's, at its index. -/
theorem tapeH_eq (τ : UE.Tape) {i : ℕ} (hc : i < Claims.Flock.zkTableCap) (r : Fin 2) (k : Fin (2 ^ 64 + 4)) :
    FlockSoundness.Refine.fOf ((UE.tapeZk τ).h i r.val k.val) = (τ ⟨i, hc⟩).1.2.2.1 r k := by
  show FlockSoundness.Refine.fOf (fAt 0 (fun r => fAt 0 ((cellAt τ i).1.2.2.1 r) k.val) r.val) = _
  rw [cellAt_lt τ hc, fAt_lt _ _ r.isLt]
  exact fAt_lt _ _ k.isLt

/-- **The cut's triple openings are the session's** at the honest pads, `ε` the coin at round `L` where the inner proof
begins (`innerPre_slot`): the cut's `inner r` is `ζ·h_fa + h_tx`, `h_fb + h_ty` at the tape's free pads (`innerMsg`),
`ζ` read at that round, and the tape's free pads are the executable's (`exeTape`). -/
theorem innerTripleE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) {L : ℕ} (hL : L = repSlot x (2 * j.val + r.val) (innerOff 0)) :
    ((srcE x c w τ j.val).rep r.val).rhoIn =
        FlockComplete.Replay.InZ.rho (innerLayE x j) (innerPadsE x τ j r)
          (FlockComplete.SimInner.cutCoins (fun k i => slotAt c k i) L) ∧
      ((srcE x c w τ j.val).rep r.val).sigmaIn =
        FlockComplete.Replay.InZ.sigma (innerLayE x j) (innerPadsE x τ j r) := by
  subst hL
  have hj := lt_of_jOf j
  have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
  have htab : (FlockDeploy.tabsOf (stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (stAt x j) := by
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) j.val (by simpa [FlockDeploy.tabsOf] using hj)]
    simp [FlockDeploy.tabsOf]
    rfl
  set Ly := _root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0 with hLy
  have hlay : innerLayE x j = Ly := by
    show _root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0 = Ly
    rw [htab]
    rfl
  have hsrc : (srcE x c w τ j.val).rep r.val = repSrcCut PE x (czExe PE x c) (cutE x c w τ) j r := by
    simp only [srcE, tabSrcCut, j.isLt, r.isLt, ↓reduceDIte]
  have hz := fits_zkTab PE (zkTab x j)
  have hE : (tabC PE x j).z.nE = Ly.hab := by
    show Ly.len - 4 = Ly.hab
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
    omega
  have hE64 : (tabC PE x j).z.nE ≤ 2 ^ 64 := hz.nE
  have hfa : Ly.fa < Ly.fb := Composed.ZerocheckMasked.fa_lt_fb Ly
  have hfb : Ly.fb < Ly.hab := Composed.ZerocheckMasked.fb_lt_hab Ly
  have htx : Ly.tx = Ly.hab + 1 := rfl
  have hty : Ly.ty = Ly.hab + 2 := rfl
  have htw : Ly.tw = Ly.hab + 3 := rfl
  have hlen : Ly.len = Ly.hab + 4 := rfl
  have hfaE : Ly.fa < (tabC PE x j).z.nE := by omega
  have hfbE : Ly.fb < (tabC PE x j).z.nE := by omega
  have hget : ∀ k (hk : k < 2 ^ 64 + 4), k < Ly.len → k ≠ Ly.hab → k ≠ Ly.tw →
      FlockSoundness.Refine.fOf (innerPadsE x τ j r).h[k]! = (τ ⟨j.val, hc⟩).1.2.2.1 r ⟨k, hk⟩ := by
    intro k hk h1 h2 h3
    show FlockSoundness.Refine.fOf (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (innerLayE x j)).h[k]! = _
    rw [hlay, FlockComplete.padsOf_get _ _ _ Ly k h1 h2 h3]
    exact tapeH_eq τ hc r ⟨k, hk⟩
  have hin : (cutE x c w τ j).inner r =
      (FlockSoundness.Refine.fOf (slotAt c (repSlot x (2 * j.val + r.val) (innerOff 0)) 0) *
          (τ ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inl ⟨Ly.fa, hfaE⟩)) + (τ ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inr 0)),
        (τ ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inl ⟨Ly.fb, hfbE⟩)) + (τ ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inr 1))) :=
    rfl
  have e1 : hEmb hz (.inl ⟨Ly.fa, hfaE⟩) = ⟨Ly.fa, by omega⟩ := rfl
  have e2 : hEmb hz (.inl ⟨Ly.fb, hfbE⟩) = ⟨Ly.fb, by omega⟩ := rfl
  have e3 : hEmb hz (.inr 0) = ⟨Ly.tx, by omega⟩ := Fin.ext (by show (tabC PE x j).z.nE + 1 + 0 = Ly.tx; omega)
  have e4 : hEmb hz (.inr 1) = ⟨Ly.ty, by omega⟩ := Fin.ext (by show (tabC PE x j).z.nE + 1 + 1 = Ly.ty; omega)
  rw [hsrc]
  constructor
  · show FlockLevel3.GF128.val ((cutE x c w τ j).inner r).1 =
      slotAt c (repSlot x (2 * j.val + r.val) (innerOff 0)) 0 * (innerPadsE x τ j r).h[(innerLayE x j).fa]! +
        (innerPadsE x τ j r).h[(innerLayE x j).tx]!
    rw [hin, e1, e3, hlay, ← hget Ly.fa (by omega) (by omega) (by omega) (by omega),
      ← hget Ly.tx (by omega) (by omega) (by omega) (by omega), val_mul_add]
  · show FlockLevel3.GF128.val ((cutE x c w τ j).inner r).2 =
      (innerPadsE x τ j r).h[(innerLayE x j).fb]! + (innerPadsE x τ j r).h[(innerLayE x j).ty]!
    rw [hin, e2, e4, hlay, ← hget Ly.fb (by omega) (by omega) (by omega) (by omega),
      ← hget Ly.ty (by omega) (by omega) (by omega) (by omega), val_add_of]

theorem val_add_mul (a b d : _root_.Flock.F128) :
    FlockLevel3.GF128.val (FlockSoundness.Refine.fOf a + FlockSoundness.Refine.fOf b * FlockSoundness.Refine.fOf d) =
      a + b * d := by
  rw [FlockLevel3.GF128.mul_eq, FlockLevel3.GF128.add_eq]
  rfl

theorem val_mul_add_mul (a b d e : _root_.Flock.F128) :
    FlockLevel3.GF128.val (FlockSoundness.Refine.fOf a * FlockSoundness.Refine.fOf b +
      FlockSoundness.Refine.fOf d * FlockSoundness.Refine.fOf e) = a * b + d * e := by
  rw [FlockLevel3.GF128.mul_eq, FlockLevel3.GF128.mul_eq, FlockLevel3.GF128.add_eq]
  rfl

theorem tapeMu_eq (τ : UE.Tape) {i : ℕ} (hc : i < Claims.Flock.zkTableCap) (r : Fin 2) (k : Fin (2 ^ 64 + 4)) :
    FlockSoundness.Refine.fOf ((UE.tapeZk τ).mu i r.val k.val) = (τ ⟨i, hc⟩).1.2.2.2.2.2.1 r k := by
  show FlockSoundness.Refine.fOf (fAt 0 (fun r => fAt 0 ((cellAt τ i).1.2.2.2.2.2.1 r) k.val) r.val) = _
  rw [cellAt_lt τ hc, fAt_lt _ _ r.isLt]
  exact fAt_lt _ _ k.isLt

theorem tapePadH_eq (τ : UE.Tape) {i : ℕ} (hc : i < Claims.Flock.zkTableCap) (r : Fin 2)
    (k : Fin _root_.Flock.Zk.K_PAD) :
    FlockSoundness.Refine.fOf ((UE.tapeZk τ).padH i r.val k.val) = (τ ⟨i, hc⟩).1.2.2.2.1 r k := by
  show FlockSoundness.Refine.fOf (fAt 0 (fun r => fAt 0 ((cellAt τ i).1.2.2.2.1 r) k.val) r.val) = _
  rw [cellAt_lt τ hc, fAt_lt _ _ r.isLt]
  exact fAt_lt _ _ k.isLt

theorem tapePadMu_eq (τ : UE.Tape) {i : ℕ} (hc : i < Claims.Flock.zkTableCap) (r : Fin 2)
    (k : Fin _root_.Flock.Zk.K_PAD) :
    FlockSoundness.Refine.fOf ((UE.tapeZk τ).padMu i r.val k.val) = (τ ⟨i, hc⟩).1.2.2.2.2.1 r k := by
  show FlockSoundness.Refine.fOf (fAt 0 (fun r => fAt 0 ((cellAt τ i).1.2.2.2.2.1 r) k.val) r.val) = _
  rw [cellAt_lt τ hc, fAt_lt _ _ r.isLt]
  exact fAt_lt _ _ k.isLt

private theorem get_append_right' (a b : Array _root_.Flock.F128) {k : ℕ} (hk : k < b.size) :
    (a ++ b)[a.size + k]! = b[k]! := by
  rw [getElem!_pos (a ++ b) _ (by simp; omega), getElem!_pos b k hk, Array.getElem_append_right (by omega)]
  simp

/-- A cut's `Y` that was sent is the view's. -/
theorem cut_Y1_of_ne {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    {T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL} {S : T.Sent} {v : T.PreView} {r : Fin 2}
    (h : (T.cut S v).Y1 r ≠ none) : (T.cut S v).Y1 r = some (v.Y1 r) := by
  by_cases hf : S.fin r
  · simp [FlockSoundness.ZK.Table.cut, hf]
  · exact absurd (by simp [FlockSoundness.ZK.Table.cut, hf]) h

/-- **`yAt` entry by entry**: where each exposed pad, each of the four tail pads in `yAt`'s order and each padding entry
reads as `g` at its index, so does every entry. -/
theorem yAt_of {nE nPad : ℕ} (Y1 : FlockSoundness.ZK.PadIdx nE → FlockLevel3.GF128) (Y2 : Fin nPad → FlockLevel3.GF128)
    (i : Fin (nE + 4 + nPad)) (g : ℕ → _root_.Flock.F128)
    (hlo : ∀ k (hk : k < nE), FlockLevel3.GF128.val (Y1 (.inl ⟨k, hk⟩)) = g k)
    (hmid : ∀ o : Fin 4, FlockLevel3.GF128.val (Y1 (.inr ((![2, 0, 1, 3] : Fin 4 → Fin 4) o))) = g (nE + o.val))
    (hhi : ∀ k : Fin nPad, FlockLevel3.GF128.val (Y2 k) = g (nE + 4 + k.val)) :
    yAt Y1 Y2 i = g i.val := by
  unfold yAt
  split_ifs with h1 h2
  · exact hlo _ h1
  · have key := hmid ⟨i.val - nE, by omega⟩
    rw [show nE + (i.val - nE) = i.val by omega] at key
    exact key
  · have key := hhi ⟨i.val - (nE + 4), by have := i.isLt; omega⟩
    rw [show nE + 4 + (i.val - (nE + 4)) = i.val by omega] at key
    exact key

/-- **The cut's `Y` is the session's** at the honest pads, `β` the coin two rounds after the inner proof begins
(`betaSlot`), wherever the cut sent it: the cut's `Y₁ = h + β·μ` and `Y₂ = pad_h + β·pad_μ` (`ZK.Table.view`) read in
`yAt`'s order are `h ‖ pad_h + β·(μ ‖ pad_μ)` at the tape's pads, `h_ab` and `h_tw` the products `padsOf` sets. -/
theorem innerYE {x : Msg} {s spec} (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2)
    {L : ℕ} (hL : L = repSlot x (2 * j.val + r.val) (innerOff 0)) (hY : (cutE x c w τ j).Y1 r ≠ none) :
    ((srcE x c w τ j.val).rep r.val).y =
      FlockComplete.Replay.InZ.y (innerPadsE x τ j r)
        (FlockComplete.SimInner.cutCoins (fun k i => slotAt c k i) L) := by
  have hj := lt_of_jOf j
  have hr := r.isLt
  have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
  have hst : (stsOf x)[j.val]! = stAt x j := getElem!_pos (stsOf x) j.val hj
  have htab : (FlockDeploy.tabsOf (stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (stAt x j) := by
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) j.val (by simpa [FlockDeploy.tabsOf] using hj)]
    simp [FlockDeploy.tabsOf]
    rfl
  set Ly := _root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0 with hLy
  have hpd : innerPadsE x τ j r = FlockComplete.padsOf (UE.tapeZk τ) j.val r.val Ly := by
    show FlockComplete.padsOf _ _ _ (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0) = _
    rw [htab]
    rfl
  have hsrc : (srcE x c w τ j.val).rep r.val = repSrcCut PE x (czExe PE x c) (cutE x c w τ) j r := by
    simp only [srcE, tabSrcCut, j.isLt, r.isLt, ↓reduceDIte]
  have hz := fits_zkTab PE (zkTab x j)
  have hE : (tabC PE x j).z.nE = Ly.hab := by
    show Ly.len - 4 = Ly.hab
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
    omega
  have hTE : ((tabC PE x j).T (czExe PE x c j)).tab.nE = Ly.hab := hE
  have hE64 : (tabC PE x j).z.nE ≤ 2 ^ 64 := hz.nE
  have hfa : Ly.fa < Ly.fb := Composed.ZerocheckMasked.fa_lt_fb Ly
  have hfb : Ly.fb < Ly.hab := Composed.ZerocheckMasked.fb_lt_hab Ly
  have htx : Ly.tx = Ly.hab + 1 := rfl
  have hty : Ly.ty = Ly.hab + 2 := rfl
  have htw : Ly.tw = Ly.hab + 3 := rfl
  have hlen : Ly.len = Ly.hab + 4 := rfl
  have hfaE : Ly.fa < (tabC PE x j).z.nE := by omega
  have hfbE : Ly.fb < (tabC PE x j).z.nE := by omega
  have hbs : betaSlot x j.val r.val = L + 2 := by
    subst hL
    unfold betaSlot repSlot innerOff
    rw [hs]
    simp only [show (2 * j.val + r.val) / 2 = j.val by omega, hst, (zkTab x j).hz]
    omega
  have hβ : betaAt c j.val r.val = slotAt c (L + 2) 0 := by unfold betaAt; rw [hbs]
  set P := FlockComplete.padsOf (UE.tapeZk τ) j.val r.val Ly with hP
  have hhs : P.h.size = Ly.len := FlockComplete.padsOf_size _ _ _ _
  have hpH : P.padH.size = _root_.Flock.Zk.K_PAD := by simp [hP, FlockComplete.padsOf]
  have hms : P.mu.size = Ly.len := by simp [hP, FlockComplete.padsOf]
  have hpM : P.padMu.size = _root_.Flock.Zk.K_PAD := by simp [hP, FlockComplete.padsOf]
  have hRh : ∀ k (hk : k < 2 ^ 64 + 4), k < Ly.len → k ≠ Ly.hab → k ≠ Ly.tw →
      FlockSoundness.Refine.fOf (P.h ++ P.padH)[k]! = (τ ⟨j.val, hc⟩).1.2.2.1 r ⟨k, hk⟩ := by
    intro k hk h1 h2 h3
    rw [FlockComplete.get_append_left _ _ (by rw [hhs]; exact h1), hP, FlockComplete.padsOf_get _ _ _ Ly k h1 h2 h3]
    exact tapeH_eq τ hc r ⟨k, hk⟩
  have hRm : ∀ k (hk : k < 2 ^ 64 + 4), k < Ly.len →
      FlockSoundness.Refine.fOf (P.mu ++ P.padMu)[k]! = (τ ⟨j.val, hc⟩).1.2.2.2.2.2.1 r ⟨k, hk⟩ := by
    intro k hk h1
    rw [FlockComplete.get_append_left _ _ (by rw [hms]; exact h1), ← tapeMu_eq τ hc r ⟨k, hk⟩,
      getElem!_pos P.mu k (by rw [hms]; exact h1)]
    simp [hP, FlockComplete.padsOf]
  have hRpH : ∀ k : Fin _root_.Flock.Zk.K_PAD,
      FlockSoundness.Refine.fOf (P.h ++ P.padH)[Ly.len + k.val]! = (τ ⟨j.val, hc⟩).1.2.2.2.1 r k := by
    intro k
    rw [← hhs, get_append_right' _ _ (by rw [hpH]; exact k.isLt), ← tapePadH_eq τ hc r k,
      getElem!_pos P.padH k.val (by rw [hpH]; exact k.isLt)]
    simp [hP, FlockComplete.padsOf]
  have hRpM : ∀ k : Fin _root_.Flock.Zk.K_PAD,
      FlockSoundness.Refine.fOf (P.mu ++ P.padMu)[Ly.len + k.val]! = (τ ⟨j.val, hc⟩).1.2.2.2.2.1 r k := by
    intro k
    rw [← hms, get_append_right' _ _ (by rw [hpM]; exact k.isLt), ← tapePadMu_eq τ hc r k,
      getElem!_pos P.padMu k.val (by rw [hpM]; exact k.isLt)]
    simp [hP, FlockComplete.padsOf]
  have hRab : (P.h ++ P.padH)[Ly.hab]! = (UE.tapeZk τ).h j.val r.val Ly.fa * (UE.tapeZk τ).h j.val r.val Ly.fb := by
    rw [FlockComplete.get_append_left _ _ (by rw [hhs]; omega), hP, FlockComplete.padsOf_hab,
      FlockComplete.padsOf_get _ _ _ Ly Ly.fa (by omega) (by omega) (by omega),
      FlockComplete.padsOf_get _ _ _ Ly Ly.fb (by omega) (by omega) (by omega)]
  have hRtw : (P.h ++ P.padH)[Ly.tw]! = (UE.tapeZk τ).h j.val r.val Ly.tx * (UE.tapeZk τ).h j.val r.val Ly.ty := by
    rw [FlockComplete.get_append_left _ _ (by rw [hhs]; omega), hP, FlockComplete.padsOf_tw,
      FlockComplete.padsOf_get _ _ _ Ly Ly.tx (by omega) (by omega) (by omega),
      FlockComplete.padsOf_get _ _ _ Ly Ly.ty (by omega) (by omega) (by omega)]
  have hY1 : (cutE x c w τ j).Y1 r = some _ := cut_Y1_of_ne hY
  have hyi : ∀ i (h2 : i < (FlockComplete.Replay.InZ.y P
      (FlockComplete.SimInner.cutCoins (fun k i => slotAt c k i) L)).size),
      (FlockComplete.Replay.InZ.y P (FlockComplete.SimInner.cutCoins (fun k i => slotAt c k i) L))[i] =
        (P.h ++ P.padH)[i]! + slotAt c (L + 2) 0 * (P.mu ++ P.padMu)[i]! := by
    intro i h2
    simp only [FlockComplete.Replay.InZ.y, Array.getElem_map, Array.getElem_range]
    rfl
  rw [hsrc, hpd]
  simp only [repSrcCut, hY1]
  apply Array.ext
  · rw [Array.size_ofFn]
    simp only [FlockComplete.Replay.InZ.y, Array.size_map, Array.size_range, Array.size_append, hhs, hpH, hlen, ← hTE]
    rfl
  · intro i h1 h2
    rw [Array.getElem_ofFn]
    rw [hyi]
    refine yAt_of _ _ _ (fun k => (P.h ++ P.padH)[k]! + slotAt c (L + 2) 0 * (P.mu ++ P.padMu)[k]!) ?_ ?_ ?_
    · intro k hk
      have hk' : k < (tabC PE x j).z.nE := hk
      show FlockLevel3.GF128.val ((τ ⟨j.val, hc⟩).1.2.2.1 r ⟨k, by omega⟩ +
        FlockSoundness.Refine.fOf (betaAt c j.val r.val) * (τ ⟨j.val, hc⟩).1.2.2.2.2.2.1 r ⟨k, by omega⟩) = _
      rw [← hRh k (by omega) (by omega) (by omega) (by omega), ← hRm k (by omega) (by omega), val_add_mul, hβ]
    · intro o
      fin_cases o
      · show FlockLevel3.GF128.val ((τ ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inl ⟨Ly.fa, hfaE⟩)) *
            (τ ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inl ⟨Ly.fb, hfbE⟩)) +
          FlockSoundness.Refine.fOf (betaAt c j.val r.val) * (τ ⟨j.val, hc⟩).1.2.2.2.2.2.1 r (padEmb' hz (.inr 2))) = _
        rw [show hEmb hz (.inl ⟨Ly.fa, hfaE⟩) = ⟨Ly.fa, by omega⟩ from rfl,
          show hEmb hz (.inl ⟨Ly.fb, hfbE⟩) = ⟨Ly.fb, by omega⟩ from rfl,
          show padEmb' hz (.inr 2) = ⟨Ly.hab, by omega⟩ from Fin.ext (by simp [padEmb', Composed.padIdx]; exact hE),
          ← tapeH_eq τ hc r ⟨Ly.fa, by omega⟩, ← tapeH_eq τ hc r ⟨Ly.fb, by omega⟩,
          ← hRm Ly.hab (by omega) (by omega), val_mul_add_mul, hβ]
        simp only [Nat.add_zero, hTE, hRab]
      · show FlockLevel3.GF128.val ((τ ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inr 0)) +
          FlockSoundness.Refine.fOf (betaAt c j.val r.val) * (τ ⟨j.val, hc⟩).1.2.2.2.2.2.1 r (padEmb' hz (.inr 0))) = _
        rw [show hEmb hz (.inr 0) = ⟨Ly.tx, by omega⟩ from Fin.ext (by show (tabC PE x j).z.nE + 1 + 0 = Ly.tx; omega),
          show padEmb' hz (.inr 0) = ⟨Ly.tx, by omega⟩ from Fin.ext (by simp [padEmb', Composed.padIdx, htx]; exact hE),
          ← hRh Ly.tx (by omega) (by omega) (by omega) (by omega), ← hRm Ly.tx (by omega) (by omega), val_add_mul, hβ]
        simp only [hTE, htx]
      · show FlockLevel3.GF128.val ((τ ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inr 1)) +
          FlockSoundness.Refine.fOf (betaAt c j.val r.val) * (τ ⟨j.val, hc⟩).1.2.2.2.2.2.1 r (padEmb' hz (.inr 1))) = _
        rw [show hEmb hz (.inr 1) = ⟨Ly.ty, by omega⟩ from Fin.ext (by show (tabC PE x j).z.nE + 1 + 1 = Ly.ty; omega),
          show padEmb' hz (.inr 1) = ⟨Ly.ty, by omega⟩ from Fin.ext (by simp [padEmb', Composed.padIdx, hty]; exact hE),
          ← hRh Ly.ty (by omega) (by omega) (by omega) (by omega), ← hRm Ly.ty (by omega) (by omega), val_add_mul, hβ]
        simp only [hTE, hty]
      · show FlockLevel3.GF128.val ((τ ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inr 0)) *
            (τ ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inr 1)) +
          FlockSoundness.Refine.fOf (betaAt c j.val r.val) * (τ ⟨j.val, hc⟩).1.2.2.2.2.2.1 r (padEmb' hz (.inr 3))) = _
        rw [show hEmb hz (.inr 0) = ⟨Ly.tx, by omega⟩ from Fin.ext (by show (tabC PE x j).z.nE + 1 + 0 = Ly.tx; omega),
          show hEmb hz (.inr 1) = ⟨Ly.ty, by omega⟩ from Fin.ext (by show (tabC PE x j).z.nE + 1 + 1 = Ly.ty; omega),
          show padEmb' hz (.inr 3) = ⟨Ly.tw, by omega⟩ from Fin.ext (by simp [padEmb', Composed.padIdx, htw]; exact hE),
          ← tapeH_eq τ hc r ⟨Ly.tx, by omega⟩, ← tapeH_eq τ hc r ⟨Ly.ty, by omega⟩,
          ← hRm Ly.tw (by omega) (by omega), val_mul_add_mul, hβ]
        simp only [hTE, htw]
        rw [show Ly.hab + 3 = Ly.tw from rfl, hRtw]
    · intro k
      have hk : k.val < _root_.Flock.Zk.K_PAD := k.isLt
      show FlockLevel3.GF128.val ((τ ⟨j.val, hc⟩).1.2.2.2.1 r (Fin.castLE hz.nPad k) +
        FlockSoundness.Refine.fOf (betaAt c j.val r.val) * (τ ⟨j.val, hc⟩).1.2.2.2.2.1 r (Fin.castLE hz.nPad k)) = _
      rw [← hRpH (Fin.castLE hz.nPad k), ← hRpM (Fin.castLE hz.nPad k), val_add_mul, hβ]
      simp only [hTE, hlen]
      rfl

/-- **`InnerSrcE`'s `triple`**: wherever the shared phases reach table `j`'s rep `r` inner proof, the cut's source
carries the session's `ρ_in`, `σ_in` at round `l'.size`'s coin (`innerPre_slot`, `innerTripleE`). -/
theorem innerSrcE_triple {x : Msg} {s spec} (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (w : Claims.Flock.Wit) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) (j : Fin (jOf x)) (r : Fin 2)
    (dom : ByteArray) (cm0 : FlockComplete.Commit) (ic : FlockComplete.InnerCommit)
    (extras : Array _root_.Flock.Claim)
    (he : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    {l : Array Msg} (hl : l.size = repBase x (2 * j.val + r.val)) {o t' l'}
    (ho : ((innerPreE x w τ c n j r dom cm0 extras ic).run (_root_.Flock.Transcript.init #[] dom)).go l =
      (.ok (o, t'), l')) :
    l'.size < n →
      ((srcE x c w τ j.val).rep r.val).rhoIn =
          FlockComplete.Replay.InZ.rho (innerLayE x j) (innerPadsE x τ j r)
            (FlockComplete.SimInner.cutCoins (fun k i => slotAt c k i) l'.size) ∧
        ((srcE x c w τ j.val).rep r.val).sigmaIn =
          FlockComplete.Replay.InZ.sigma (innerLayE x j) (innerPadsE x τ j r) :=
  fun _ => innerTripleE x c w τ j r (innerPre_slot hs w τ c n j r dom cm0 ic extras he hl ho)

/-- **What hybrid 2's inner proof reads from the cut** (`SimInner`'s open content): at a live statement and a witness in
the deployed relation, at any opened coins `c` and stop `n` at which no refusal is seen, wherever the phases hybrids 2
and 3 share (`innerPreE`) reach table `j`'s rep `r` inner proof after `L` bodies with the constraints `cons`, the cut's
source carries the session's inner values at the honest pads (`InnerSrc`): `ρ`, `σ` at round `L`'s coin, the batched
constraint and `τ`'s commitment at round `L + 1`'s, and `Y` at round `L + 2`'s, each where its round is answered. -/
def InnerSrcE : Prop :=
  ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ), ¬ Refused x c n →
    ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
    ∀ (j : Fin (jOf x)) (r : Fin 2) (dom : ByteArray) (cm0 : FlockComplete.Commit) (ic : FlockComplete.InnerCommit)
      (extras : Array _root_.Flock.Claim),
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras →
    ∀ l : Array Msg, l.size = repBase x (2 * j.val + r.val) →
    ∀ o t' l', ((innerPreE x w τ c n j r dom cm0 extras ic).run (_root_.Flock.Transcript.init #[] dom)).go l =
        (.ok (o, t'), l') →
      FlockComplete.SimInner.InnerSrc (fun k i => slotAt c k i) n _root_.Flock.Sha512.hash (innerLayE x j)
        (innerPadsE x τ j r) o.2.2.2 ((srcE x c w τ j.val).rep r.val) l'.size

/-- **`SimInner` from what the cut carries for the inner proof**: hybrids 2 and 3 run the same shared phases
(`repMix_eq`), and where those reach the inner proof, `innerCut` on the cut's source and `innerP` on the honest pads
send the same bodies and agree up to what `eraseZ` forgets (`innerCut_alike`). -/
theorem simInner_of (h : InnerSrcE) : SimInner := by
  intro x hx w hw τ c n hn s spec hs j r dom cm0 ic extras he l hl
  unfold mixE
  rw [FlockComplete.SimInner.repMix_eq 2 le_rfl, FlockComplete.SimInner.repMix_eq (2 + 1) (by omega)]
  refine FlockComplete.SimInner.go_bind_congr_r fun o t' l' ho => ?_
  rw [ite_eq_left le_rfl, ite_eq_right (by omega)]
  exact FlockComplete.SimInner.innerCut_alike (fun i i' e => by simp only [FlockComplete.eraseZ, e])
    (h x hx w hw τ c n hn s spec hs j r dom cm0 ic extras he l hl o t' l' ho)

end Masked

end Claims.ZKMal
