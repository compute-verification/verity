import Proofs.Flock.Claims.Complete.Zk.Exact
import Proofs.Flock.Claims.Complete.Zk.Count

/-!
# A masked rep's sends where a refusal fires

Read forward from a run that need not return: a step that throws ends the run with the sends so far. Where one of a
masked rep's refusals fires, the rep stops right after the send whose reply it refused.

* `go_bind_cases`, `tm_bind_cases`, `lp_bind_cases`, `lforIn_stop`: a bind, or a loop over `[0:n]`, that ends at
  `(R, L)` either ran its first part (each step) to a return, or stopped there with `R` its error.
  `lforIn_bind_stop`: a loop and its rest, the loop's invariant kept by its returning steps.
* `innerP_beta_stop`: at `β = 0`, the reply to send `|l| + 2`, the inner proof stops within `3` sends.
* `ligeritoPZk_rho_stop`: at a dependent pair, the replies to sends `|l| + 2 + k_0` and `|l| + 3 + k_0`, level 0's
  masked Ligerito stops within `k_0 + 4` sends.
* `repPZk_stop`: a masked rep that stops stopped within `piopZk + k_0 + 4` sends, or after passing its pair coins
  within `piopZk + (k_0 + 7 + 11 R) + 3`, or after passing its `β` too. `repPZk_rho_stop`, `repPZk_beta_stop`: at a
  dependent pair or `β = 0`, the rep stops within those sends (a rep that returns passed both, `repPZk_exact`).
-/

namespace FlockComplete

open Flock Transcript Replay Replay.Lg

variable {α β : Type}

/-- **A bind that ends at `(R, L)`**: its first part stopped there, or returned and its rest ended there. -/
theorem go_bind_cases {y : PS α} {g : α → PS β} {l L : Array Claims.Msg} {R : Except String β}
    (h : (y >>= g).go l = (R, L)) :
    (∃ e, y.go l = (.error e, L) ∧ R = .error e) ∨ ∃ a l', y.go l = (.ok a, l') ∧ (g a).go l' = (R, L) := by
  rw [go_bind] at h
  rcases hy : y.go l with ⟨r, l'⟩
  rw [hy] at h
  cases r with
  | error e =>
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj h
    exact .inl ⟨e, rfl, rfl⟩
  | ok a => exact .inr ⟨a, l', rfl, h⟩

theorem tm_bind_cases {x : TMW PS α} {f : α → TMW PS β} {t : Transcript} {l L : Array Claims.Msg}
    {R : Except String (β × Transcript)} (h : ((x >>= f).run t).go l = (R, L)) :
    (∃ e, (x.run t).go l = (.error e, L) ∧ R = .error e) ∨
      ∃ a t₁ l₁, (x.run t).go l = (.ok (a, t₁), l₁) ∧ ((f a).run t₁).go l₁ = (R, L) := by
  rw [StateT.run_bind] at h
  rcases go_bind_cases h with ⟨e, h1, rfl⟩ | ⟨⟨a, t₁⟩, l₁, h1, h2⟩
  · exact .inl ⟨e, h1, rfl⟩
  · exact .inr ⟨a, t₁, l₁, h1, h2⟩

theorem lp_bind_cases {x : LP PS α} {f : α → LP PS β} {st : LigState} {t : Transcript} {l L : Array Claims.Msg}
    {R : Except String ((β × LigState) × Transcript)} (h : (((x >>= f).run st).run t).go l = (R, L)) :
    (∃ e, ((x.run st).run t).go l = (.error e, L) ∧ R = .error e) ∨
      ∃ a st₁ t₁ l₁, ((x.run st).run t).go l = (.ok ((a, st₁), t₁), l₁) ∧
        (((f a).run st₁).run t₁).go l₁ = (R, L) := by
  rw [StateT.run_bind, StateT.run_bind] at h
  rcases go_bind_cases h with ⟨e, h1, rfl⟩ | ⟨⟨⟨a, st₁⟩, t₁⟩, l₁, h1, h2⟩
  · exact .inl ⟨e, h1, rfl⟩
  · exact .inr ⟨a, st₁, t₁, l₁, h1, h2⟩

theorem lbind_go {x : LP PS α} {f : α → LP PS β} {st st' : LigState} {t t' : Transcript} {l l' : Array Claims.Msg}
    {a : α} (h : ((x.run st).run t).go l = (.ok ((a, st'), t'), l')) :
    (((x >>= f).run st).run t).go l = (((f a).run st').run t').go l' := by
  rw [StateT.run_bind, StateT.run_bind, go_bind, h]

theorem throw_go (e : String) (t : Transcript) (l : Array Claims.Msg) :
    ((throw e : TMW PS α).run t).go l = (.error e, l) := rfl

theorem lthrow_go (e : String) (st : LigState) (t : Transcript) (l : Array Claims.Msg) :
    (((throw e : LP PS α).run st).run t).go l = (.error e, l) := rfl

/-- **A loop over `[0:n]` that ends at `(R, L)`**, its every returning step keeping `I`: it returned with `I n`, or
some step `j < n`, from a state and bodies with `I j`, stopped there. -/
theorem lforIn_stop {σ : Type} (body : ℕ → σ → LP PS (ForInStep σ)) (I : ℕ → σ → Array Claims.Msg → Prop) (n : ℕ)
    (hb : ∀ j < n, ∀ s st t l r st' t' l', I j s l → (((body j s).run st).run t).go l = (.ok ((r, st'), t'), l') →
      ∃ s', r = .yield s' ∧ I (j + 1) s' l')
    {s : σ} {st : LigState} {t : Transcript} {l L : Array Claims.Msg} {R : Except String ((σ × LigState) × Transcript)}
    (h₀ : I 0 s l) (h : (((forIn [:n] s body).run st).run t).go l = (R, L)) :
    (∃ s' st' t', R = .ok ((s', st'), t') ∧ I n s' L) ∨
      ∃ j < n, ∃ s₁ st₁ t₁ l₁ e, I j s₁ l₁ ∧ (((body j s₁).run st₁).run t₁).go l₁ = (.error e, L) ∧
        R = .error e := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one] at h
  suffices key : ∀ k a (s : σ) st t (l : Array Claims.Msg), a + k = n → I a s l →
      (((forIn (List.range' a k) s body).run st).run t).go l = (R, L) →
      (∃ s' st' t', R = .ok ((s', st'), t') ∧ I n s' L) ∨
        ∃ j < n, ∃ s₁ st₁ t₁ l₁ e, I j s₁ l₁ ∧ (((body j s₁).run st₁).run t₁).go l₁ = (.error e, L) ∧
          R = .error e from
    key n 0 s st t l (by omega) h₀ h
  intro k
  induction k with
  | zero =>
    intro a s st t l ha hI h
    simp only [List.range'_zero, List.forIn_nil] at h
    have e : (((Pure.pure s : LP PS σ).run st).run t).go l = (.ok ((s, st), t), l) := rfl
    rw [e] at h
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj h
    exact .inl ⟨s, st, t, rfl, by simpa [← ha] using hI⟩
  | succ k ih =>
    intro a s st t l ha hI h
    rw [List.range'_succ, List.forIn_cons] at h
    rcases lp_bind_cases h with ⟨e, h₁, rfl⟩ | ⟨r, st₁, t₁, l₁, h₁, h₂⟩
    · exact .inr ⟨a, by omega, s, st, t, l, e, hI, h₁, rfl⟩
    · obtain ⟨s₁, rfl, hI₁⟩ := hb a (by omega) s st t l r st₁ t₁ l₁ hI h₁
      exact ih (a + 1) s₁ st₁ t₁ l₁ (by omega) hI₁ h₂

/-- **A loop over `[0:n]`, then the rest, that stops**: the loop returned with `I n` and the rest stopped, or some step
`j < n`, from a state and bodies with `I j`, stopped. -/
theorem lforIn_bind_stop {σ : Type} (body : ℕ → σ → LP PS (ForInStep σ)) (I : ℕ → σ → Array Claims.Msg → Prop)
    (n : ℕ) (hb : ∀ j < n, ∀ s st t l r st' t' l', I j s l →
      (((body j s).run st).run t).go l = (.ok ((r, st'), t'), l') → ∃ s', r = .yield s' ∧ I (j + 1) s' l')
    {f : σ → LP PS β} {s : σ} {st : LigState} {t : Transcript} {l L : Array Claims.Msg} {e : String}
    (h₀ : I 0 s l) (h : ((((forIn [:n] s body) >>= f).run st).run t).go l = (.error e, L)) :
    (∃ s' st' t' l', I n s' l' ∧ (((f s').run st').run t').go l' = (.error e, L)) ∨
      ∃ j < n, ∃ s₁ st₁ t₁ l₁ e', I j s₁ l₁ ∧ (((body j s₁).run st₁).run t₁).go l₁ = (.error e', L) := by
  rcases lp_bind_cases h with ⟨e', h₁, -⟩ | ⟨s', st', t', l', h₁, h₂⟩
  · rcases lforIn_stop body I n hb h₀ h₁ with ⟨_, _, _, h₃, -⟩ | ⟨j, hj, s₁, st₁, t₁, l₁, e'', hI, h₃, -⟩
    · cases h₃
    · exact .inr ⟨j, hj, s₁, st₁, t₁, l₁, e'', hI, h₃⟩
  · rcases lforIn_stop body I n hb h₀ h₁ with ⟨_, _, _, h₃, hI⟩ | ⟨_, _, _, _, _, _, _, _, _, h₃⟩
    · simp only [Except.ok.injEq, Prod.mk.injEq] at h₃
      obtain ⟨⟨rfl, -⟩, -⟩ := h₃
      exact .inl ⟨s', st', t', l', hI, h₂⟩
    · cases h₃

/-! ## The inner proof -/

/-- **At `β = 0` the inner proof stops within `3` sends**: `β` is the reply to its send `|l| + 2`, and it throws there
or at the batched constraint before. -/
theorem innerP_beta_stop (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (lay : Zk.PadLayout) (p : PadsZk)
    (ic : InnerCommit) (finalC : F128) (cons : Array (String × Zk.Lin)) {t : Transcript}
    {l L : Array Claims.Msg} {R : Except String (Zk.InnerProof × Transcript)}
    (hrun : ((innerP (asksS resp) (Hashes.ofSha512 h) lay p ic finalC cons).run t).go l = (R, L))
    (hb : (coinOf (resp (l.size + 2)) 0).isZero = true) :
    (∃ e, R = .error e) ∧ L.size ≤ l.size + 3 := by
  unfold innerP at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _), bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) _),
    bind_go (sampleP_go resp _ _)] at hrun
  dsimp only at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbSlice_eq _ _) _),
    bind_go (sampleVecP_go resp (by simp [Zk.triple]) _ _)] at hrun
  split at hrun
  · rw [StateT.run_bind, go_bind, throw_go] at hrun
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj hrun
    exact ⟨⟨_, rfl⟩, by simp⟩
  rw [tauCommitmentWith_ofSha512, bind_go (x := StateT.lift (pure _)) rfl,
    bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) _), bind_go (sampleP_go resp _ _)] at hrun
  split at hrun
  · rw [StateT.run_bind, go_bind, throw_go] at hrun
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj hrun
    exact ⟨⟨_, rfl⟩, by simp⟩
  rename_i hb'
  exact absurd (by simpa using hb) hb'

/-! ## Ligerito -/

/-- **At a dependent pair Ligerito stops within `k_0 + 4` sends**: level 0's pair coins are the replies to its sends
`|l| + 2 + k_0` and `|l| + 3 + k_0`, and it throws right after the second. -/
theorem ligeritoPZk_rho_stop (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (hn : sch.n ≠ 0)
    (hk0 : 1 ≤ sch.k0) (hL : 2 ≤ sch.levels.size)
    (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0) {t : Transcript} {l L : Array Claims.Msg}
    {R : Except String ((LigeritoProof × Zk.Level0Msgs × Zk.L0Rec) × Transcript)}
    (hrun : ((ligeritoPZk (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 tb b lay hv rep tPad extra lanePad salt
      nonce).run t).go l = (R, L))
    (hbad : Zk.rhoIndependent (coinKOf (resp (l.size + 2 + sch.k0)), coinKOf (resp (l.size + 3 + sch.k0))) = false) :
    (∃ e, R = .error e) ∧ L.size ≤ l.size + sch.k0 + 4 := by
  rcases R with e | ⟨r, t'⟩
  swap
  · have := (ligeritoPZk_exact resp h ms sch T cm0 tb b lay hv rep tPad extra lanePad salt nonce hn hk0 hL hq hrun).2
    rw [hbad] at this
    cases this
  refine ⟨⟨e, rfl⟩, ?_⟩
  unfold ligeritoPZk at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l)]
    at hrun
  dsimp only at hrun
  rcases tm_bind_cases hrun with ⟨e', h1, -⟩ | ⟨a, t₁, l₁, -, h2⟩
  swap
  · have := (Prod.mk.inj h2).1
    cases this
  rw [lbind_go (liftT_go (sampleVecP_go resp hn _ _) _), lbind_go (x := modify _) rfl,
    lbind_go (x := liftT (liftTM (absorbF _))) rfl, lbind_go (x := powP nonce 2) rfl,
    lbind_go (liftT_go (sampleP_go resp _ _) _), lbind_go (x := modify _) rfl, lbind_go (x := MonadState.get) rfl,
    lbind_go (x := sendMsg _) rfl] at h1
  have h2 := lforIn_bind_stop _ (fun j (_ : Array F256 × (F256 × F256) × Zk.Level0Msgs) (l' : Array Claims.Msg) =>
      l'.size = l.size + 2 + j + (if j = sch.k0 then 2 else 0) ∧
      (j = sch.k0 → Zk.rhoIndependent (coinKOf (resp (l.size + 2 + sch.k0)),
        coinKOf (resp (l.size + 3 + sch.k0))) = true)) sch.k0
    ?step ⟨by rw [ite_eq_right (by omega)]; simp, fun h => absurd h (by omega)⟩ h1
  case step =>
    rintro j hj s st t l₀ r st' t' l₂ ⟨hl₀, -⟩ hb
    rw [ite_eq_right (by omega), Nat.add_zero] at hl₀
    obtain ⟨t₃, l₃, hl₃, hb⟩ := lsampleKP_bind resp hb
    obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_modify _) hb
    by_cases hj1 : j + 1 = sch.k0
    · rw [ite_eq_left (by simp [hj1])] at hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind lsilent_get hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨t₄, l₄, hl₄, hb⟩ := lsampleKP_bind resp hb
      obtain ⟨t₅, l₅, hl₅, hb⟩ := lsampleKP_bind resp hb
      split at hb
      · obtain ⟨_, _, _, _, h0, _⟩ := lp_bind_ok hb
        cases h0
      rename_i hind
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_modify _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind lsilent_get hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_sendMsg _) hb
      obtain ⟨rfl, rfl⟩ := lpure_go_ok hb
      refine ⟨_, rfl, by rw [ite_eq_left hj1]; omega, fun _ => ?_⟩
      rw [show l₃.size = l.size + 2 + sch.k0 by omega, show l₄.size = l.size + 3 + sch.k0 by omega] at hind
      simpa using hind
    · rw [ite_eq_right (by simp [hj1])] at hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind lsilent_get hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_sendMsg _) hb
      obtain ⟨rfl, rfl⟩ := lpure_go_ok hb
      exact ⟨_, rfl, by rw [ite_eq_right hj1]; omega, fun h => absurd h hj1⟩
  rcases h2 with ⟨_, _, _, _, ⟨-, hρ⟩, -⟩ | ⟨j, hj, s₁, st₁, t₁, l₁, e₁, ⟨hl₁, -⟩, hb⟩
  · rw [hρ rfl] at hbad
    cases hbad
  rw [ite_eq_right (by omega), Nat.add_zero] at hl₁
  rw [lbind_go (liftT_go (sampleKP_go resp _ _) _), lbind_go (x := modify _) rfl] at hb
  by_cases hj1 : j + 1 = sch.k0
  · rw [ite_eq_left (by simp [hj1])] at hb
    rw [lbind_go (x := MonadState.get) rfl, lbind_go (x := liftT (liftTM (absorbK _))) rfl,
      lbind_go (x := liftT (liftTM (absorbK _))) rfl, lbind_go (liftT_go (sampleKP_go resp _ _) _),
      lbind_go (liftT_go (sampleKP_go resp _ _) _)] at hb
    split at hb
    · rcases lp_bind_cases hb with ⟨e₂, h₃, -⟩ | ⟨_, _, _, _, h₃, -⟩
      · rw [lthrow_go] at h₃
        obtain ⟨-, rfl⟩ := Prod.mk.inj h₃
        simp only [Array.size_push]
        omega
      · rw [lthrow_go] at h₃
        cases (Prod.mk.inj h₃).1
    · rename_i hind
      exfalso
      apply hind
      simp only [Array.size_push, hl₁]
      rw [show l.size + 2 + j + 1 = l.size + 2 + sch.k0 by omega,
        show l.size + 2 + sch.k0 + 1 = l.size + 3 + sch.k0 by omega]
      unfold coinKOf at hbad
      rw [hbad]
      rfl
  · rw [ite_eq_right (by simp [hj1])] at hb
    rw [lbind_go (x := MonadState.get) rfl, lbind_go (x := sendMsg _) rfl] at hb
    cases (Prod.mk.inj hb).1

/-! ## A rep -/

/-- A run within a sends bound sends at most that many, whether it returns or stops. -/
theorem sends_le {y : TMW PS α} {n : ℕ} (hy : (sendsC.st Transcript).P y n) {t : Transcript}
    {l L : Array Claims.Msg} {R : Except String (α × Transcript)} (h : (y.run t).go l = (R, L)) :
    L.size ≤ l.size + n := by
  have := hy t l
  change ((y.run t).go l).2.size ≤ _ at this
  rw [h] at this
  exact this

/-- A lifted check of the verifier's monad sends nothing: it stops where it fails, or the rest runs at its value. -/
theorem liftV_bind_cases {v : V α} {f : α → TMW PS β} {t : Transcript} {l L : Array Claims.Msg}
    {R : Except String (β × Transcript)}
    (h : (((StateT.lift (monadLift v : PS α) : TMW PS α) >>= f).run t).go l = (R, L)) :
    (L = l ∧ ∃ e, R = .error e) ∨ ∃ a, v = .ok a ∧ ((f a).run t).go l = (R, L) := by
  rcases tm_bind_cases h with ⟨e, h1, rfl⟩ | ⟨a, t₁, l₁, h1, h2⟩
  · change ((monadLift v : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at h1
    rcases go_bind_cases h1 with ⟨e', h3, -⟩ | ⟨b, l₂, -, h4⟩
    · rw [JsonRT.monadLift_go] at h3
      exact .inl ⟨(Prod.mk.inj h3).2.symm, e, rfl⟩
    · cases (Prod.mk.inj h4).1
  · change ((monadLift v : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at h1
    obtain ⟨b, l₂, e3, e4⟩ := JsonRT.go_bind_ok h1
    rw [JsonRT.monadLift_go] at e3
    simp only [Prod.mk.injEq] at e3
    obtain ⟨hv, rfl⟩ := e3
    cases e4
    exact .inr ⟨_, hv, h2⟩

/-- **A masked rep that stops** stopped within `piopZk + k_0 + 4` sends; or its pair coins, the replies to sends
`|l| + piopZk + 2 + k_0` and the next, are independent and it stopped within `piopZk + (k_0 + 7 + 11 R) + 3`, or its
`β`, the reply to send `|l| + piopZk + (k_0 + 7 + 11 R) + 2`, is not `0`. -/
theorem repPZk_stop (resp : ℕ → Claims.Msg) (hsh : ByteArray → ByteArray) (st : Setup) (sch : Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (tr : Trace) (tb : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m) (hn : sch.n ≠ 0)
    (hk0 : 1 ≤ sch.k0) (hL : 2 ≤ sch.levels.size)
    (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0) {t : Transcript} {l L : Array Claims.Msg}
    {e : String}
    (h : ((repPZk (asksS resp) (Hashes.ofSha512 hsh) st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt
      nonce prior words).run t).go l = (.error e, L)) :
    L.size ≤ l.size + piopZk st extras + (sch.k0 + 4) ∨
      (Zk.rhoIndependent (coinKOf (resp (l.size + piopZk st extras + 2 + sch.k0)),
        coinKOf (resp (l.size + piopZk st extras + 3 + sch.k0))) = true ∧
      (L.size ≤ l.size + piopZk st extras + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) + 3 ∨
        (coinOf (resp (l.size + piopZk st extras + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) + 2)) 0).isZero =
          false)) := by
  obtain ⟨l₁, hl₁, hz⟩ := zerocheckPZk_go resp hm st.digest (flat cm0.cap) ic.root
    (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h tr t l
  unfold repPZk at h
  rw [bind_go hz] at h
  dsimp only at h
  have h' := h
  rw [StateT.run_bind] at h'
  rcases go_bind_cases h' with ⟨e₁, hlc, -⟩ | ⟨_, _, h₁, -⟩
  · left
    have := sends_le (lincheckPZk_sends resp _ _ _ _ _ _) hlc
    unfold piopZk
    omega
  obtain ⟨full, hfold⟩ := lincheckPZk_fold resp _ _ _ _ _ _ _ _ h₁
  obtain ⟨l₂, hl₂, hl⟩ := lincheckPZk_go resp st.fold st.m (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h tr
    (Zc.out st.m (Zc.coinsAt resp l.size) tr) full _ l₁ hfold
  rw [bind_go hl] at h
  dsimp only at h
  rcases tm_bind_cases h with ⟨e₂, hk, -⟩ | ⟨u, t₃, l₃', hk, h⟩
  · left
    have := sends_le (rankCheckP_sends _ _ _ _) hk
    unfold piopZk
    omega
  obtain ⟨ht₃, hl₃⟩ := rankCheckP_go hk
  subst t₃ l₃'
  obtain ⟨l₃, hl₃, hrs⟩ := ringSwitchP_go resp tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) (Lc.coinsAt resp l₁.size (st.fold.kLog - 6))) ++
      extras) (by simp [abcClaims]) _ l₂
  rw [bind_go hrs] at h
  dsimp only at h
  have hsz : (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) (Lc.coinsAt resp l₁.size (st.fold.kLog - 6))) ++
      extras).size = 2 + extras.size := by simp [abcClaims, Nat.add_comm]
  have e3 : l₃.size = l.size + piopZk st extras := by unfold piopZk; omega
  rcases tm_bind_cases h with ⟨e₃, hlg, -⟩ | ⟨⟨lg, zl, rec⟩, t₄, l₄, hlg, h⟩
  · by_cases hρ : Zk.rhoIndependent (coinKOf (resp (l.size + piopZk st extras + 2 + sch.k0)),
        coinKOf (resp (l.size + piopZk st extras + 3 + sch.k0))) = true
    · refine .inr ⟨hρ, .inl ?_⟩
      have := sends_le (ligeritoPZk_sends resp hsh _ _ _ _ _ _ _ _ _ _ _ _ _ _) hlg
      omega
    · left
      have := (ligeritoPZk_rho_stop resp hsh _ _ _ _ _ _ _ _ _ _ _ _ _ _ hn hk0 hL hq hlg (by
        rw [e3]; simpa using hρ)).2
      omega
  obtain ⟨hl₄, hrho⟩ := ligeritoPZk_exact resp hsh st.merkle sch _ cm0 _ _ _ _ rep tPad extra lanePad salt nonce hn
    hk0 hL hq hlg
  have e4 : l₄.size = l.size + piopZk st extras + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) := by omega
  refine .inr ⟨by rw [← e3]; exact hrho, ?_⟩
  rw [bind_go (show ((get : TMW PS Transcript).run t₄).go l₄ = (.ok (t₄, t₄), l₄) from rfl)] at h
  dsimp only at h
  rcases liftV_bind_cases h with ⟨rfl, -⟩ | ⟨⟨rp, x⟩, -, h⟩
  · left
    omega
  dsimp only at h
  rcases liftV_bind_cases h with ⟨rfl, -⟩ | ⟨lgc, -, h⟩
  · left
    omega
  rcases tm_bind_cases h with ⟨e₄, hin, -⟩ | ⟨inner, t₅, l₅, hin, h⟩
  · by_cases hb : (coinOf (resp (l₄.size + 2)) 0).isZero = true
    · left
      have := (innerP_beta_stop resp hsh _ _ _ _ _ hin hb).2
      omega
    · right
      rw [← e4]
      simpa using hb
  · cases (Prod.mk.inj h).1

/-- **At a dependent pair a masked rep stops within `piopZk + k_0 + 4` sends**: its pair coins are the replies to
sends `|l| + piopZk + 2 + k_0` and the next. -/
theorem repPZk_rho_stop (resp : ℕ → Claims.Msg) (hsh : ByteArray → ByteArray) (st : Setup) (sch : Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (tr : Trace) (tb : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m) (hn : sch.n ≠ 0)
    (hk0 : 1 ≤ sch.k0) (hL : 2 ≤ sch.levels.size)
    (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0) {t : Transcript} {l L : Array Claims.Msg}
    {R : Except String (Zk.ZkProof × Transcript)}
    (h : ((repPZk (asksS resp) (Hashes.ofSha512 hsh) st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt
      nonce prior words).run t).go l = (R, L))
    (hbad : Zk.rhoIndependent (coinKOf (resp (l.size + piopZk st extras + 2 + sch.k0)),
        coinKOf (resp (l.size + piopZk st extras + 3 + sch.k0))) = false) :
    (∃ e, R = .error e) ∧ L.size ≤ l.size + piopZk st extras + (sch.k0 + 4) := by
  rcases R with e | ⟨P, t'⟩
  · refine ⟨⟨e, rfl⟩, ?_⟩
    rcases repPZk_stop resp hsh st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt nonce prior words hm hn
      hk0 hL hq h with h1 | ⟨h1, -⟩
    · exact h1
    · rw [hbad] at h1
      cases h1
  · have := (repPZk_exact resp hsh st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt nonce prior words hm
      hn hk0 hL hq h).2.1
    rw [hbad] at this
    cases this

/-- **At `β = 0` a masked rep stops within `piopZk + (k_0 + 7 + 11 R) + 3` sends**: its `β` is the reply to send
`|l| + piopZk + (k_0 + 7 + 11 R) + 2`. -/
theorem repPZk_beta_stop (resp : ℕ → Claims.Msg) (hsh : ByteArray → ByteArray) (st : Setup) (sch : Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (tr : Trace) (tb : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m) (hn : sch.n ≠ 0)
    (hk0 : 1 ≤ sch.k0) (hL : 2 ≤ sch.levels.size)
    (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0) {t : Transcript} {l L : Array Claims.Msg}
    {R : Except String (Zk.ZkProof × Transcript)}
    (h : ((repPZk (asksS resp) (Hashes.ofSha512 hsh) st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt
      nonce prior words).run t).go l = (R, L))
    (hbad : (coinOf (resp (l.size + piopZk st extras + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) + 2)) 0).isZero =
      true) :
    (∃ e, R = .error e) ∧ L.size ≤ l.size + piopZk st extras + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) + 3 := by
  rcases R with e | ⟨P, t'⟩
  · refine ⟨⟨e, rfl⟩, ?_⟩
    rcases repPZk_stop resp hsh st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt nonce prior words hm hn
      hk0 hL hq h with h1 | ⟨-, h1 | h1⟩
    · omega
    · exact h1
    · rw [hbad] at h1
      cases h1
  · have := (repPZk_exact resp hsh st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt nonce prior words hm
      hn hk0 hL hq h).2.2.1
    rw [hbad] at this
    cases this

end FlockComplete
