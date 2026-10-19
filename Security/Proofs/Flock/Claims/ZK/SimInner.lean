import Proofs.Flock.Claims.ZK.Split

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
def cutCoins (coin : ℕ → ℕ → F128) (L : ℕ) : InZ.Coins := ⟨coin L, coin (L + 1), coin (L + 2), coin (L + 3)⟩

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

/-- **The phases `repMix k` shares for every `k ≥ 2`**, up to its inner proof: the session's PIOP and Ligerito (level 0's
opening erased), the replay and the constraints. Returns `final_c`, the proof, level 0's messages and the constraints. -/
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

end FlockComplete.SimInner

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- Table `j`'s pads layout at the deployed instance. -/
noncomputable abbrev layE (x : Msg) (j : Fin (jOf x)) : _root_.Flock.Zk.PadLayout :=
  _root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog
    (zkTab x j).sch.k0

/-- Table `j`'s rep `r` honest pads at the tape `τ`. -/
noncomputable abbrev padsE (x : Msg) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) : FlockComplete.PadsZk :=
  FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)

/-- **The phases hybrids 2 and 3 share** at the deployed instance (`mixE`'s arguments). -/
noncomputable def preE (x : Msg) (w : Claims.Flock.Wit) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin)
    (n : ℕ) (j : Fin (jOf x)) (r : Fin 2) (dom : ByteArray) (cm0 : FlockComplete.Commit)
    (extras : Array _root_.Flock.Claim) (ic : FlockComplete.InnerCommit) :=
  FlockComplete.SimInner.innerPre (ioE c n) hsE
    (FlockDeploy.tabsOf (stsOf x))[j.val]! (zkTab x j).sch (zkTab x j).tPad r.val dom
    (trsE x w (UE.tapeZk τ))[j.val]!
    (FlockComplete.packed (trsE x w (UE.tapeZk τ))[j.val]! (FlockDeploy.tabsOf (stsOf x))[j.val]!.m) cm0 extras
    (padsE x τ j r) ic
    ((UE.tapeZk τ).extra j.val) ((UE.tapeZk τ).lanePad j.val) ((UE.tapeZk τ).salt j.val r.val)
    ((UE.tapeZk τ).nonce j.val r.val) (priorE c j.val r.val) (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!

/-- **What hybrid 2's inner proof reads from the cut** (`SimInner`'s open content): at a live statement and a witness in
the deployed relation, at any opened coins `c` and stop `n` at which no refusal is seen, wherever the phases hybrids 2
and 3 share (`preE`) reach table `j`'s rep `r` inner proof after `L` bodies with the constraints `cons`, the cut's source
carries the session's inner values at the honest pads (`InnerSrc`): `ρ`, `σ` at round `L`'s coin, the batched constraint
and `τ`'s commitment at round `L + 1`'s, and `Y` at round `L + 2`'s, each where its round is answered. -/
def InnerSrcE : Prop :=
  ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ), ¬ Refused x c n →
    ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
    ∀ (j : Fin (jOf x)) (r : Fin 2) (dom : ByteArray) (cm0 : FlockComplete.Commit) (ic : FlockComplete.InnerCommit)
      (extras : Array _root_.Flock.Claim),
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras →
    ∀ l : Array Msg, l.size = repBase x (2 * j.val + r.val) →
    ∀ o t' l', ((preE x w τ c n j r dom cm0 extras ic).run (_root_.Flock.Transcript.init #[] dom)).go l =
        (.ok (o, t'), l') →
      FlockComplete.SimInner.InnerSrc (fun k i => slotAt c k i) n _root_.Flock.Sha512.hash (layE x j) (padsE x τ j r)
        o.2.2.2 ((srcE x c w τ j.val).rep r.val) l'.size

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
