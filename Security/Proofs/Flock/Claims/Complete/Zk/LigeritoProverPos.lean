import Proofs.Flock.Claims.Complete.Zk.Ligerito
import Proofs.Flock.Claims.Complete.Replay.LigeritoProver

/-!
# The masked prover's Ligerito record, by send position

`ligeritoPZk_pos`: a masked Ligerito prover (`ligeritoPZk`) that returns, run in the pure session from the bodies `l`,
records the target it was given, level 0's out-of-domain coin as the reply to send `|l| + 1`, its `k_0` lane coins as
the replies to sends `|l| + 2` on, and its pair's coins as the replies to the next two sends: the prover's half of
the record agreement whose verifier half is `zkLigeritoWith_pos`.

* **Steps** (`LP PS`, read forward from a run that returns): a squeeze is answered by the reply to the next send and
  sends one body (`lsampleKP_bind`, `lsampleP_bind`, `lsampleVecP_bind`); a step that sends nothing (`LSilent`: the
  transcript's actions, the state's, a message, a nonce) leaves the bodies as they were (`lsilent_bind`); a loop over
  `[0:n]` keeps an invariant of its state and the bodies (`lforIn_inv`).
-/

namespace FlockComplete

open Flock Transcript Replay Replay.Lg

variable {α β : Type}

/-- The coin pair a reply carries, as `sampleKP` draws it. -/
def coinKOf (m : Claims.Msg) : F256 := ⟨coinOf m 0, coinOf m 1⟩

/-- A prover step that sends nothing. -/
def LSilent (x : LP PS α) : Prop :=
  ∀ st t l a st' t' l', ((x.run st).run t).go l = (.ok ((a, st'), t'), l') → l' = l

theorem lsilent_bind {x : LP PS α} {f : α → LP PS β} (hx : LSilent x) {st : LigState} {t : Transcript}
    {l l'' : Array Claims.Msg} {r : (β × LigState) × Transcript} (h : (((x >>= f).run st).run t).go l = (.ok r, l'')) :
    ∃ a st₁ t₁, (((f a).run st₁).run t₁).go l = (.ok r, l'') := by
  obtain ⟨a, st₁, t₁, l₁, h₁, h₂⟩ := lp_bind_ok h
  obtain rfl := hx st t l a st₁ t₁ l₁ h₁
  exact ⟨a, st₁, t₁, h₂⟩

theorem lsilent_liftTM (x : TM α) : LSilent (liftT (liftTM x) : LP PS α) := by
  intro st t l a st' t' l' h
  rcases hx : x t with e | ⟨a₁, t₁⟩
  · have : (((liftT (liftTM x) : LP PS α).run st).run t).go l = (.error e, l) := by
      show ((liftTM x >>= fun a => Pure.pure (a, st) : TMW PS (α × LigState)).run t).go l = _
      rw [StateT.run_bind, go_bind, liftTM_go, hx]
    rw [this] at h
    cases h
  · rw [liftT_go (liftTM_ok hx l) st] at h
    exact (ok_inj h).2.2.2.symm

theorem lsilent_modify (g : LigState → LigState) : LSilent (modify g : LP PS PUnit) := by
  intro st t l a st' t' l' h
  exact (ok_inj (show (Except.ok (((), g st), t), l) = _ from h)).2.2.2.symm

theorem lsilent_get : LSilent (MonadState.get : LP PS LigState) := by
  intro st t l a st' t' l' h
  exact (ok_inj (show (Except.ok ((st, st), t), l) = _ from h)).2.2.2.symm

theorem lsilent_sendMsg (m : F256 × F256) : LSilent (sendMsg m : LP PS Unit) := by
  intro st t l a st' t' l' h
  exact (ok_inj (show (Except.ok (((), { st with msgs := st.msgs.push m }), ap (pK m) t), l) = _ from h)).2.2.2.symm

theorem lsilent_powB (nonce : ℕ → ℕ → UInt64) : LSilent (powP nonce 2 : LP PS Unit) := by
  intro st t l a st' t' l' h
  exact (ok_inj (show (Except.ok (((), { st with cbn := st.cbn.push 0 }),
    ap (pN 0) t), l) = _ from h)).2.2.2.symm

theorem lpure_go_ok {b b' : β} {st st' : LigState} {t t' : Transcript} {l l' : Array Claims.Msg}
    (h : (((Pure.pure b : LP PS β).run st).run t).go l = (.ok ((b', st'), t'), l')) : b = b' ∧ l = l' :=
  have e := ok_inj (show (Except.ok ((b, st), t), l) = _ from h)
  ⟨e.1, e.2.2.2⟩

theorem lsampleKP_bind (resp : ℕ → Claims.Msg) {f : F256 → LP PS β} {st : LigState} {t : Transcript}
    {l l'' : Array Claims.Msg} {r : (β × LigState) × Transcript}
    (h : (((liftT (sampleKP (asksS resp)) >>= f).run st).run t).go l = (.ok r, l'')) :
    ∃ t₁ l₁, l₁.size = l.size + 1 ∧ (((f (coinKOf (resp l.size))).run st).run t₁).go l₁ = (.ok r, l'') := by
  obtain ⟨a, st₁, t₁, l₁, h₁, h₂⟩ := lp_bind_ok h
  rw [liftT_go (sampleKP_go resp t l) st] at h₁
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj h₁
  exact ⟨_, _, by simp, h₂⟩

theorem lsampleP_bind (resp : ℕ → Claims.Msg) {f : F128 → LP PS β} {st : LigState} {t : Transcript}
    {l l'' : Array Claims.Msg} {r : (β × LigState) × Transcript}
    (h : (((liftT (sampleP (asksS resp)) >>= f).run st).run t).go l = (.ok r, l'')) :
    ∃ t₁ l₁, l₁.size = l.size + 1 ∧ (((f (coinOf (resp l.size) 0)).run st).run t₁).go l₁ = (.ok r, l'') := by
  obtain ⟨a, st₁, t₁, l₁, h₁, h₂⟩ := lp_bind_ok h
  rw [liftT_go (sampleP_go resp t l) st] at h₁
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj h₁
  exact ⟨_, _, by simp, h₂⟩

theorem lsampleVecP_bind (resp : ℕ → Claims.Msg) {n : ℕ} (hn : n ≠ 0) {f : Array F128 → LP PS β} {st : LigState}
    {t : Transcript} {l l'' : Array Claims.Msg} {r : (β × LigState) × Transcript}
    (h : (((liftT (sampleVecP (asksS resp) n) >>= f).run st).run t).go l = (.ok r, l'')) :
    ∃ t₁ l₁, l₁.size = l.size + 1 ∧
      (((f ((Array.range n).map (coinOf (resp l.size)))).run st).run t₁).go l₁ = (.ok r, l'') := by
  obtain ⟨a, st₁, t₁, l₁, h₁, h₂⟩ := lp_bind_ok h
  rw [liftT_go (sampleVecP_go resp hn t l) st] at h₁
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj h₁
  exact ⟨_, _, by simp, h₂⟩

/-- **A loop over `[0:n]`** whose every step, from a state and bodies with `I j`, yields a state and bodies with
`I (j + 1)`, ends with `I n`. -/
theorem lforIn_inv {σ : Type} (body : ℕ → σ → LP PS (ForInStep σ)) (I : ℕ → σ → Array Claims.Msg → Prop) (n : ℕ)
    (hb : ∀ j < n, ∀ s st t l r st' t' l', I j s l → (((body j s).run st).run t).go l = (.ok ((r, st'), t'), l') →
      ∃ s', r = .yield s' ∧ I (j + 1) s' l')
    {s : σ} {st : LigState} {t : Transcript} {l : Array Claims.Msg} {s' : σ} {st' : LigState} {t' : Transcript}
    {l' : Array Claims.Msg} (h₀ : I 0 s l) (h : (((forIn [:n] s body).run st).run t).go l = (.ok ((s', st'), t'), l')) :
    I n s' l' := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one] at h
  suffices key : ∀ k a (s : σ) st t (l : Array Claims.Msg), a + k = n → I a s l →
      (((forIn (List.range' a k) s body).run st).run t).go l = (.ok ((s', st'), t'), l') → I n s' l' from
    key n 0 s st t l (by omega) h₀ h
  intro k
  induction k with
  | zero =>
    intro a s st t l ha hI h
    simp only [List.range'_zero, List.forIn_nil] at h
    obtain ⟨rfl, rfl⟩ := lpure_go_ok h
    simpa [← ha] using hI
  | succ k ih =>
    intro a s st t l ha hI h
    rw [List.range'_succ, List.forIn_cons] at h
    obtain ⟨r, st₁, t₁, l₁, h₁, h₂⟩ := lp_bind_ok h
    obtain ⟨s₁, rfl, hI₁⟩ := hb a (by omega) s st t l r st₁ t₁ l₁ hI h₁
    exact ih (a + 1) s₁ st₁ t₁ l₁ (by omega) hI₁ h₂

theorem push_get!_of {γ : Type} [Inhabited γ] {a : Array γ} {x : γ} {f : ℕ → γ} (ha : ∀ i < a.size, a[i]! = f i)
    (hx : x = f a.size) : ∀ i < a.size + 1, (a.push x)[i]! = f i := by
  intro i hi
  rw [getElem!_pos (a.push x) i (by simp; omega), Array.getElem_push]
  split
  · rename_i h
    rw [← getElem!_pos]
    exact ha i h
  · rename_i h
    rw [hx]
    congr 1
    omega

/-- **What a masked Ligerito prover that returns records, by send position**: run in the pure session from the bodies
`l`, its record's target is `T`, its out-of-domain coin is the reply to send `|l| + 1`, its lane coins are the replies
to sends `|l| + 2` to `|l| + 1 + k_0`, and its pair's coins the replies to the next two. -/
theorem ligeritoPZk_pos (resp : ℕ → Claims.Msg) (hs : Hashes PS) (ms : MerkleScheme) (sch : Schedule) (T : F128)
    (cm0 : Commit) (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (hn : sch.n ≠ 0)
    (hk0 : 1 ≤ sch.k0) {t t' : Transcript} {l l' : Array Claims.Msg} {P : LigeritoProof} {zl : Zk.Level0Msgs}
    {rc : Zk.L0Rec}
    (hrun : ((ligeritoPZk (asksS resp) hs ms sch T cm0 tb b lay hv rep tPad extra lanePad salt nonce).run t).go l =
      (.ok ((P, zl, rc), t'), l')) :
    rc.target = T ∧ (∃ y, rc.ood = #[(coinOf (resp (l.size + 1)) 0, y)]) ∧ rc.challenges.size = sch.k0 ∧
      (∀ j < sch.k0, rc.challenges[j]! = coinKOf (resp (l.size + 2 + j))) ∧
      rc.rho = (coinKOf (resp (l.size + 2 + sch.k0)), coinKOf (resp (l.size + 3 + sch.k0))) := by
  unfold ligeritoPZk at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l)]
    at hrun
  dsimp only at hrun
  rw [StateT.run_bind] at hrun
  obtain ⟨⟨⟨R, stF⟩, t₁⟩, l₁, h1, h2⟩ := JsonRT.go_bind_ok hrun
  obtain ⟨h21, rfl⟩ := Prod.mk.inj h2
  obtain ⟨hP, rfl⟩ := Prod.mk.inj (Except.ok.inj h21)
  obtain ⟨-, hP⟩ := Prod.mk.inj hP
  obtain ⟨-, rfl⟩ := Prod.mk.inj hP
  obtain ⟨tA, lA, hlA, h1⟩ := lsampleVecP_bind resp hn h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_modify _) h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_liftTM _) h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_powB nonce) h1
  obtain ⟨tB, lB, hlB, h1⟩ := lsampleP_bind resp h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_modify _) h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind lsilent_get h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_sendMsg _) h1
  obtain ⟨s, stC, tC, lC, hL, h1⟩ := lp_bind_ok h1
  have hI := lforIn_inv _ (fun j (s : Array F256 × (F256 × F256) × Zk.Level0Msgs) (l' : Array Claims.Msg) =>
      s.1.size = j ∧ (∀ i < j, s.1[i]! = coinKOf (resp (l.size + 2 + i))) ∧
      l'.size = l.size + 2 + j + (if j = sch.k0 then 2 else 0) ∧
      (j = sch.k0 → s.2.1 = (coinKOf (resp (l.size + 2 + sch.k0)), coinKOf (resp (l.size + 3 + sch.k0))))) sch.k0
    ?step ⟨rfl, fun i hi => absurd hi (Nat.not_lt_zero _), by rw [ite_eq_right (by omega)]; omega,
      fun h => absurd h (by omega)⟩ hL
  case step =>
    rintro j hj s st t l₀ r st' t' l₂ ⟨hsz, hch, hl₀, -⟩ hb
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
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_modify _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind lsilent_get hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_sendMsg _) hb
      obtain ⟨rfl, rfl⟩ := lpure_go_ok hb
      refine ⟨_, rfl, by simp [hsz], fun i hi => ?_, by rw [ite_eq_left hj1]; omega, fun _ => ?_⟩
      · exact push_get!_of (fun i hi => hch i (hsz ▸ hi)) (by rw [hsz, hl₀]) i (by omega)
      · show (coinKOf (resp l₃.size), coinKOf (resp l₄.size)) = _
        rw [show l₃.size = l.size + 2 + sch.k0 by omega, show l₄.size = l.size + 3 + sch.k0 by omega]
    · rw [ite_eq_right (by simp [hj1])] at hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind lsilent_get hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_sendMsg _) hb
      obtain ⟨rfl, rfl⟩ := lpure_go_ok hb
      refine ⟨_, rfl, by simp [hsz], fun i hi => ?_, by rw [ite_eq_right hj1]; omega, fun h => absurd h hj1⟩
      exact push_get!_of (fun i hi => hch i (hsz ▸ hi)) (by rw [hsz, hl₀]) i (by omega)
  iterate 11 (obtain ⟨_, _, _, _, -, h1⟩ := lp_bind_ok h1)
  obtain ⟨rfl, -⟩ := lpure_go_ok h1
  obtain ⟨hsz, hch, -, hrho⟩ := hI
  refine ⟨rfl, ⟨mleF tb ((Array.range sch.n).map (coinOf (resp l.size))) + hv[lay.lgOod 0]!, ?_⟩, hsz, hch,
    hrho rfl⟩
  show #[(coinOf (resp lA.size) 0, _)] = _
  rw [hlA]

end FlockComplete
