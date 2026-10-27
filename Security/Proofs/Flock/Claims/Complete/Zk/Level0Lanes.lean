import Proofs.Flock.Claims.Complete.ZkTables.Level0
import Proofs.Flock.Claims.Complete.Zk.Level0Run

/-!
# The masked Ligerito prover's lane phase in closed form

`ligeritoPZk_lanes`: a masked Ligerito prover (`ligeritoPZk`) that returns, run in the pure session from the bodies `l`,
has `ZkTables.Level0.LaneForm`'s record, masked out-of-domain value, lane messages, pair's sums and blinded claim.

* **Runs that only push** (`grL`, a `Total` on `LP PS`): every run that returns only pushes onto the state's lists
  (`Lg.Grow`), with a property of its value. After the lane loop the prover only pushes, so its proof's first
  out-of-domain value and first `k_0` messages are the lane phase's (`grL_lp`).
* **The lane phase, state by state** (`lmodify_bind`, `lget_bind`, `lliftTM_bind`, `lsendMsg_bind`, `lpowB_bind`, and a
  loop over `[0:n]` keeping an invariant of its state, the prover's and the sends, `lforIn_invS`): after `j` lane
  coins the table and the weights are folded at the first `j`, message `i ≤ j` is `roundPoly` of their `i`-th folds
  masked, and at `j = k_0` the pair's sums and the blinded claim are those of the `k_0`-th (`LaneInv`).
-/

namespace FlockComplete.L0

open Flock Transcript FlockComplete Replay Replay.Lg JsonRT

variable {α β σ : Type}

/-! ## Runs that only push -/

theorem liftT_ok' {y : TMW PS α} {st st' : LigState} {t t' : Transcript} {l l' : Array Claims.Msg} {a : α}
    (h : (((liftT y : LP PS α).run st).run t).go l = (.ok ((a, st'), t'), l')) :
    st' = st ∧ (y.run t).go l = (.ok (a, t'), l') := by
  change ((y >>= fun a => Pure.pure (a, st)).run t).go l = _ at h
  rw [StateT.run_bind] at h
  obtain ⟨⟨a₁, t₁⟩, l₁, h1, h2⟩ := go_bind_ok h
  simp only [StateT.run_pure, go_pure, Prod.mk.injEq, Except.ok.injEq] at h2
  obtain ⟨⟨⟨rfl, rfl⟩, rfl⟩, rfl⟩ := h2
  exact ⟨rfl, h1⟩

/-- **A Ligerito step's runs only push**: every run that returns only pushes onto the state's lists, with `P` of its
value. -/
def grL : Total (LP PS) where
  T y P := ∀ st t l a st' t' l', ((y.run st).run t).go l = (.ok ((a, st'), t'), l') → Grow st st' ∧ P a
  mono h hq st t l a st' t' l' e := ⟨(h st t l a st' t' l' e).1, hq a (h st t l a st' t' l' e).2⟩
  pure a h st t l a' st' t' l' e := by
    obtain ⟨rfl, rfl, -, -⟩ := ok_inj (show (Except.ok ((a, st), t), l) = _ from e)
    exact ⟨Grow.refl _, h⟩
  bind hx hf st t l b st'' t'' l'' e := by
    obtain ⟨a, st₁, t₁, l₁, h1, h2⟩ := lp_bind_ok e
    obtain ⟨g1, q⟩ := hx st t l a st₁ t₁ l₁ h1
    obtain ⟨g2, p⟩ := hf a q st₁ t₁ l₁ b st'' t'' l'' h2
    exact ⟨g1.trans g2, p⟩
  lift v a hv h st t l a' st' t' l' e := by
    obtain ⟨hst, e₂⟩ := liftT_ok' (y := (monadLift v : TMW PS _)) e
    change ((monadLift v : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at e₂
    rw [go_bind, monadLift_go, hv] at e₂
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at e₂
    obtain ⟨⟨rfl, -⟩, -⟩ := e₂
    exact ⟨hst ▸ Grow.refl _, h⟩

theorem grL_liftT (y : TMW PS α) : grL.T (liftT y) fun _ => True := fun _ _ _ _ _ _ _ e => by
  obtain ⟨rfl, -⟩ := liftT_ok' e
  exact ⟨Grow.refl _, trivial⟩

theorem grL_get : grL.T (MonadState.get : LP PS LigState) fun _ => True := fun st t l _ _ _ _ e => by
  obtain ⟨-, rfl, -, -⟩ := ok_inj (show (Except.ok ((st, st), t), l) = _ from e)
  exact ⟨Grow.refl _, trivial⟩

theorem grL_modify (g : LigState → LigState) (hg : ∀ s, Grow s (g s)) :
    grL.T (modify g : LP PS PUnit) fun _ => True := fun st t l _ _ _ _ e => by
  obtain ⟨-, rfl, -, -⟩ := ok_inj (show (Except.ok (((), g st), t), l) = _ from e)
  exact ⟨hg st, trivial⟩

theorem set_ok {s' st st₁ : LigState} {t t₁ : Transcript} {l l₁ : Array Claims.Msg} {a : PUnit}
    (h : (((set s' : LP PS PUnit).run st).run t).go l = (.ok ((a, st₁), t₁), l₁)) : st₁ = s' :=
  (ok_inj (show (Except.ok (((), s'), t), l) = _ from h)).2.1.symm

theorem grL_powP (nonce : ℕ → ℕ → UInt64) (kind : ℕ) : grL.T (powP nonce kind : LP PS Unit) fun _ => True := by
  intro st t l _ st' _ _ e
  obtain ⟨_, st₁, t₁, l₁, h1, h2⟩ := lp_bind_ok e
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok ((st, st), t), l) = _ from h1)
  obtain ⟨_, st₂, t₂, l₂, h3, h4⟩ := lp_bind_ok h2
  obtain rfl := set_ok h3
  obtain ⟨rfl, -⟩ := liftT_ok' h4
  refine ⟨?_, trivial⟩
  rcases kind with _ | _ | kind
  · exact ⟨Pre.refl _, Pre.refl _, Pre.refl _, Pre.push _ _, Pre.refl _⟩
  · exact ⟨Pre.refl _, Pre.refl _, Pre.refl _, Pre.refl _, Pre.push _ _⟩
  · exact ⟨Pre.refl _, Pre.refl _, Pre.push _ _, Pre.refl _, Pre.refl _⟩

theorem grL_sendMsg (m : F256 × F256) : grL.T (sendMsg (M := PS) m) fun _ => True := by
  unfold sendMsg
  repeat (first
    | extract_lets
    | (with_reducible apply Total.bind')
    | (with_reducible intro _)
    | (with_reducible apply Total.pure')
    | (with_reducible apply grL_get)
    | (with_reducible apply grL_modify; intro s; unfold Grow; refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
        (first | exact Pre.refl _ | exact Pre.push _ _))
    | (with_reducible apply grL_liftT)
    | (with_reducible apply grL_powP)
    | (with_reducible apply Total.forIn_range)
    | (with_reducible apply Total.forIn_array)
    | (with_reducible apply Total.ite)
    | split)

theorem grL_challengeP (resp : ℕ → Claims.Msg) (split : Bool) :
    grL.T (challengeP (asksS resp) split) fun _ => True := by
  unfold challengeP
  repeat (first
    | extract_lets
    | (with_reducible apply Total.bind')
    | (with_reducible intro _)
    | (with_reducible apply Total.pure')
    | (with_reducible apply grL_get)
    | (with_reducible apply grL_modify; intro s; unfold Grow; refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
        (exact Pre.refl _))
    | (with_reducible apply grL_liftT)
    | (with_reducible apply grL_sendMsg)
    | (with_reducible apply Total.ite)
    | split)

theorem grL_oodIntroP (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ) :
    grL.T (oodIntroP (asksS resp) nonce g dim) fun _ => True := by
  unfold oodIntroP
  repeat (first
    | extract_lets
    | (with_reducible apply Total.bind')
    | (with_reducible intro _)
    | (with_reducible apply grL_modify; intro s; unfold Grow; refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
        (first | exact Pre.refl _ | exact Pre.push _ _))
    | (with_reducible apply grL_liftT)
    | (with_reducible apply grL_sendMsg)
    | (with_reducible apply grL_powP))

theorem grL_queryP (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit) :
    grL.T (queryP (asksS resp) nonce ms lv cm) fun _ => True := by
  unfold queryP
  repeat (first
    | extract_lets
    | (with_reducible apply Total.bind')
    | (with_reducible intro _)
    | (with_reducible apply Total.pure')
    | (with_reducible apply grL_powP)
    | (with_reducible apply grL_liftT))

theorem Pre.get_lt {γ : Type} [Inhabited γ] {A B : Array γ} (h : Pre A B) {i : ℕ} (hi : i < A.size) :
    B[i]! = A[i]! := by
  obtain ⟨C, rfl⟩ := h
  rw [getElem!_pos (A ++ C) i (by simp; omega), getElem!_pos A i hi, Array.getElem_append_left hi]

theorem Pre.size_le {γ : Type} {A B : Array γ} (h : Pre A B) : A.size ≤ B.size := by
  obtain ⟨C, rfl⟩ := h
  simp

/-! ## The lane phase, state by state -/

theorem lmodify_bind {g : LigState → LigState} {f : PUnit → LP PS β} {st : LigState} {t : Transcript}
    {l l'' : Array Claims.Msg} {r : (β × LigState) × Transcript}
    (h : (((modify g >>= f).run st).run t).go l = (.ok r, l'')) : (((f ⟨⟩).run (g st)).run t).go l = (.ok r, l'') := by
  obtain ⟨a, st₁, t₁, l₁, h₁, h₂⟩ := lp_bind_ok h
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), g st), t), l) = _ from h₁)
  exact h₂

theorem lget_bind {f : LigState → LP PS β} {st : LigState} {t : Transcript} {l l'' : Array Claims.Msg}
    {r : (β × LigState) × Transcript} (h : (((MonadState.get >>= f).run st).run t).go l = (.ok r, l'')) :
    (((f st).run st).run t).go l = (.ok r, l'') := by
  obtain ⟨a, st₁, t₁, l₁, h₁, h₂⟩ := lp_bind_ok h
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok ((st, st), t), l) = _ from h₁)
  exact h₂

theorem lliftTM_bind {x : TM α} {f : α → LP PS β} {st : LigState} {t : Transcript} {l l'' : Array Claims.Msg}
    {r : (β × LigState) × Transcript} (h : (((liftT (liftTM x) >>= f).run st).run t).go l = (.ok r, l'')) :
    ∃ a t₁, (((f a).run st).run t₁).go l = (.ok r, l'') := by
  obtain ⟨a, st₁, t₁, l₁, h₁, h₂⟩ := lp_bind_ok h
  obtain ⟨hst, h₃⟩ := liftT_ok' h₁
  rw [liftTM_go] at h₃
  obtain ⟨-, rfl⟩ := Prod.mk.inj h₃
  exact ⟨a, t₁, hst ▸ h₂⟩

theorem lsendMsg_bind (m : F256 × F256) {f : Unit → LP PS β} {st : LigState} {t : Transcript}
    {l l'' : Array Claims.Msg} {r : (β × LigState) × Transcript}
    (h : (((sendMsg m >>= f).run st).run t).go l = (.ok r, l'')) :
    ∃ t₁, (((f ()).run { st with msgs := st.msgs.push m }).run t₁).go l = (.ok r, l'') := by
  obtain ⟨a, st₁, t₁, l₁, h₁, h₂⟩ := lp_bind_ok h
  obtain ⟨rfl, rfl, rfl, rfl⟩ :=
    ok_inj (show (Except.ok (((), { st with msgs := st.msgs.push m }), ap (pK m) t), l) = _ from h₁)
  exact ⟨_, h₂⟩

theorem lpowB_bind (nonce : ℕ → ℕ → UInt64) {f : Unit → LP PS β} {st : LigState} {t : Transcript}
    {l l'' : Array Claims.Msg} {r : (β × LigState) × Transcript}
    (h : (((powP nonce 2 >>= f).run st).run t).go l = (.ok r, l'')) :
    ∃ t₁, (((f ()).run { st with cbn := st.cbn.push 0 }).run t₁).go l = (.ok r, l'') := by
  obtain ⟨a, st₁, t₁, l₁, h₁, h₂⟩ := lp_bind_ok h
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), { st with cbn := st.cbn.push 0 }),
    ap (pN 0) t), l) = _ from h₁)
  exact ⟨_, h₂⟩

theorem lpure_ok {b b' : β} {st st' : LigState} {t t' : Transcript} {l l' : Array Claims.Msg}
    (h : (((Pure.pure b : LP PS β).run st).run t).go l = (.ok ((b', st'), t'), l')) : b = b' ∧ st = st' ∧ l = l' :=
  have e := ok_inj (show (Except.ok ((b, st), t), l) = _ from h)
  ⟨e.1, e.2.1, e.2.2.2⟩

/-- **A loop over `[0:n]`** whose every step, from a loop state, prover state and bodies with `I j`, yields ones with
`I (j + 1)`, ends with `I n`. -/
theorem lforIn_invS (body : ℕ → σ → LP PS (ForInStep σ)) (I : ℕ → σ → LigState → Array Claims.Msg → Prop) (n : ℕ)
    (hb : ∀ j < n, ∀ s st t l r st' t' l', I j s st l → (((body j s).run st).run t).go l = (.ok ((r, st'), t'), l') →
      ∃ s', r = .yield s' ∧ I (j + 1) s' st' l')
    {s : σ} {st : LigState} {t : Transcript} {l : Array Claims.Msg} {s' : σ} {st' : LigState} {t' : Transcript}
    {l' : Array Claims.Msg} (h₀ : I 0 s st l)
    (h : (((forIn [:n] s body).run st).run t).go l = (.ok ((s', st'), t'), l')) : I n s' st' l' := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one] at h
  suffices key : ∀ k a (s : σ) st t (l : Array Claims.Msg), a + k = n → I a s st l →
      (((forIn (List.range' a k) s body).run st).run t).go l = (.ok ((s', st'), t'), l') → I n s' st' l' from
    key n 0 s st t l (by omega) h₀ h
  intro k
  induction k with
  | zero =>
    intro a s st t l ha hI h
    simp only [List.range'_zero, List.forIn_nil] at h
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h
    simpa [← ha] using hI
  | succ k ih =>
    intro a s st t l ha hI h
    rw [List.range'_succ, List.forIn_cons] at h
    obtain ⟨r, st₁, t₁, l₁, h₁, h₂⟩ := lp_bind_ok h
    obtain ⟨s₁, rfl, hI₁⟩ := hb a (by omega) s st t l r st₁ t₁ l₁ hI h₁
    exact ih (a + 1) s₁ st₁ t₁ l₁ (by omega) hI₁ h₂

theorem foldKs_take_succ (ch : List F256) {j : ℕ} (hj : j < ch.length) (x : Array F256) :
    foldKs (ch.take (j + 1)) x = foldK ch[j] (foldKs (ch.take j) x) := by
  rw [List.take_add_one, List.getElem?_eq_getElem hj, Option.toList_some, foldKs, foldKs, List.foldl_append]
  rfl

theorem laneCh_length (resp : ℕ → Claims.Msg) (k0 L : ℕ) : (ZkTables.Level0.laneCh resp k0 L).length = k0 := by
  simp [ZkTables.Level0.laneCh]

theorem laneCh_get (resp : ℕ → Claims.Msg) (k0 L : ℕ) {j : ℕ} (hj : j < (ZkTables.Level0.laneCh resp k0 L).length) :
    (ZkTables.Level0.laneCh resp k0 L)[j] = coinKOf (resp (L + 2 + j)) := by
  simp [ZkTables.Level0.laneCh]

theorem toArray_of_get! {γ : Type} [Inhabited γ] {a : Array γ} {k : ℕ} {f : ℕ → γ} (hs : a.size = k)
    (h : ∀ i < k, a[i]! = f i) : a = ((List.range k).map f).toArray := by
  apply Array.ext (by simp [hs])
  intro i h1 h2
  rw [← getElem!_pos a i h1, h i (hs ▸ h1)]
  simp

open ZkTables.Level0 in
/-- **The lane loop's invariant** after `j` lane coins, from `L` sends: the coins drawn are the replies; `j + 2` sends
since `L` (and the pair's two once `j = k_0`); the out-of-domain value `y`; messages `0` to `j`, the first `k_0` of
them `roundPoly` of the folds masked; before `k_0` the table and weights folded at the first `j` coins; at `k_0` the
pair's coins, sums and blinded claim. -/
def LaneInv (resp : ℕ → Claims.Msg) (sch : Schedule) (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128)
    (rep : ℕ) (extra : ℕ → ℕ → F128) (L : ℕ) (y : F128) (j : ℕ) (s : Array F256 × (F256 × F256) × Zk.Level0Msgs)
    (st : LigState) (l : Array Claims.Msg) : Prop :=
  s.1.size = j ∧ (∀ i < j, s.1[i]! = coinKOf (resp (L + 2 + i))) ∧
  l.size = L + 2 + j + (if j = sch.k0 then 2 else 0) ∧ st.oods = #[y] ∧ st.msgs.size = j + 1 ∧
  (∀ i < j + 1, i < sch.k0 → st.msgs[i]! = maskMsg lay hv i
    (roundPoly (foldKs ((laneCh resp sch.k0 L).take i) (embedF tb))
      (foldKs ((laneCh resp sch.k0 L).take i) (laneW0 resp sch.n L b)))) ∧
  (j < sch.k0 → st.tab = foldKs ((laneCh resp sch.k0 L).take j) (embedF tb) ∧
    st.w = foldKs ((laneCh resp sch.k0 L).take j) (laneW0 resp sch.n L b)) ∧
  (j = sch.k0 → s.2.1 = laneRho resp sch.k0 L ∧
    s.2.2.e0 = dotK (foldKs (laneCh resp sch.k0 L) (laneW0 resp sch.n L b))
      (embedF (pairLane extra (2 * rep) (foldKs (laneCh resp sch.k0 L) (embedF tb)).size)) +
      ⟨hv[lay.lgE 0 0]!, hv[lay.lgE 0 1]!⟩ ∧
    s.2.2.e1 = dotK (foldKs (laneCh resp sch.k0 L) (laneW0 resp sch.n L b))
      (embedF (pairLane extra (2 * rep + 1) (foldKs (laneCh resp sch.k0 L) (embedF tb)).size)) +
      ⟨hv[lay.lgE 1 0]!, hv[lay.lgE 1 1]!⟩ ∧
    s.2.2.tPrime = dotK (foldKs (laneCh resp sch.k0 L) (laneW0 resp sch.n L b))
      (blinded (foldKs (laneCh resp sch.k0 L) (embedF tb)) (laneRho resp sch.k0 L)
        (pairLane extra (2 * rep) (foldKs (laneCh resp sch.k0 L) (embedF tb)).size)
        (pairLane extra (2 * rep + 1) (foldKs (laneCh resp sch.k0 L) (embedF tb)).size)))

open ZkTables.Level0 in
/-- **The masked Ligerito prover's lane phase in closed form**: a masked Ligerito prover that returns, run in the pure
session from the bodies `l`, has `LaneForm`'s record, out-of-domain value, lane messages, pair's sums and blinded claim
at `L = |l|`. -/
theorem ligeritoPZk_lanes (resp : ℕ → Claims.Msg) (hs : Hashes PS) (ms : MerkleScheme) (sch : Schedule) (T : F128)
    (cm0 : Commit) (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (hn : sch.n ≠ 0)
    (hk0 : 1 ≤ sch.k0) {t t' : Transcript} {l l' : Array Claims.Msg} {P : LigeritoProof} {zl : Zk.Level0Msgs}
    {rc : Zk.L0Rec}
    (hrun : ((ligeritoPZk (asksS resp) hs ms sch T cm0 tb b lay hv rep tPad extra lanePad salt nonce).run t).go l =
      (.ok ((P, zl, rc), t'), l')) :
    LaneForm resp sch T tb b lay hv rep extra l.size P zl rc := by
  unfold ligeritoPZk at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l)]
    at hrun
  dsimp only at hrun
  rw [StateT.run_bind] at hrun
  obtain ⟨⟨⟨⟨caps, rproofs, fin, op0, zl', initCh, rho, b0', y'⟩, stF⟩, t₁⟩, l₁, h1, h2⟩ := JsonRT.go_bind_ok hrun
  obtain ⟨h21, rfl⟩ := Prod.mk.inj h2
  obtain ⟨hP, rfl⟩ := Prod.mk.inj (Except.ok.inj h21)
  obtain ⟨rfl, hP₂⟩ := Prod.mk.inj hP
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj hP₂
  obtain ⟨tA, lA, hlA, h1⟩ := lsampleVecP_bind resp hn h1
  have h1 := lmodify_bind h1
  obtain ⟨_, _, h1⟩ := lliftTM_bind h1
  obtain ⟨_, h1⟩ := lpowB_bind nonce h1
  obtain ⟨tB, lB, hlB, h1⟩ := lsampleP_bind resp h1
  have h1 := lmodify_bind h1
  have h1 := lget_bind h1
  obtain ⟨_, h1⟩ := lsendMsg_bind _ h1
  obtain ⟨s, stC, tC, lC, hL, h1⟩ := lp_bind_ok h1
  have hI := lforIn_invS _ (LaneInv resp sch tb b lay hv rep extra l.size
      (mleF tb ((Array.range sch.n).map (coinOf (resp l.size))) + hv[lay.lgOod 0]!)) sch.k0 ?step ?init hL
  case init =>
    refine ⟨rfl, fun i hi => absurd hi (Nat.not_lt_zero _), ?_, rfl, rfl, fun i hi _ => ?_, fun _ => ⟨rfl, ?_⟩,
      fun h => absurd h (by omega)⟩
    · rw [ite_eq_right (by omega)]
      omega
    · obtain rfl : i = 0 := by omega
      show maskMsg lay hv 0 (roundPoly (embedF tb) (addScaled (embedF b) _ (coinOf (resp lA.size) 0))) = _
      rw [hlA]
      rfl
    · show addScaled (embedF b) _ (coinOf (resp lA.size) 0) = _
      rw [hlA]
      rfl
  case step =>
    rintro j hj s st t₀ l₀ r st' t₂ l₂ ⟨hsz, hch, hl₀, hoods, hmsz, hmsg, htw, -⟩ hb
    rw [ite_eq_right (by omega), Nat.add_zero] at hl₀
    obtain ⟨htab, hw⟩ := htw hj
    have hlen := laneCh_length resp sch.k0 l.size
    have hfold : ∀ x, foldK (coinKOf (resp l₀.size)) (foldKs ((laneCh resp sch.k0 l.size).take j) x) =
        foldKs ((laneCh resp sch.k0 l.size).take (j + 1)) x := fun x => by
      rw [foldKs_take_succ _ (by omega), laneCh_get, hl₀]
    obtain ⟨t₃, l₃, hl₃, hb⟩ := lsampleKP_bind resp hb
    have hb := lmodify_bind hb
    by_cases hj1 : j + 1 = sch.k0
    · rw [ite_eq_left (by simp [hj1])] at hb
      have hb := lget_bind hb
      obtain ⟨_, _, hb⟩ := lliftTM_bind hb
      obtain ⟨_, _, hb⟩ := lliftTM_bind hb
      obtain ⟨t₄, l₄, hl₄, hb⟩ := lsampleKP_bind resp hb
      obtain ⟨t₅, l₅, hl₅, hb⟩ := lsampleKP_bind resp hb
      split at hb
      · obtain ⟨_, _, _, _, h0, _⟩ := lp_bind_ok hb
        cases h0
      obtain ⟨_, _, hb⟩ := lliftTM_bind hb
      obtain ⟨_, _, hb⟩ := lliftTM_bind hb
      obtain ⟨_, _, hb⟩ := lliftTM_bind hb
      have hb := lmodify_bind hb
      have hb := lget_bind hb
      obtain ⟨_, hb⟩ := lsendMsg_bind _ hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      have htake : (laneCh resp sch.k0 l.size).take (j + 1) = laneCh resp sch.k0 l.size := by
        rw [hj1]
        exact List.take_of_length_le (by omega)
      have hr3 : l₃.size = l.size + 2 + sch.k0 := by omega
      have hr4 : l₄.size = l.size + 3 + sch.k0 := by omega
      refine ⟨_, rfl, by simp [hsz], fun i hi => ?_, by rw [ite_eq_left hj1]; omega, hoods, by simp [hmsz],
        fun i hi hik => ?_, fun h => absurd h (by omega), fun _ => ⟨?_, ?_, ?_, ?_⟩⟩
      · exact push_get!_of (fun i hi => hch i (hsz ▸ hi)) (by rw [hsz, hl₀]) i (by omega)
      · dsimp only
        rw [Pre.get_lt (Pre.push _ _) (by omega)]
        exact hmsg i (by omega) hik
      · show (coinKOf (resp l₃.size), coinKOf (resp l₄.size)) = _
        rw [hr3, hr4]
        rfl
      · dsimp only
        rw [htab, hw, hfold, hfold, htake]
        rfl
      · dsimp only
        rw [htab, hw, hfold, hfold, htake]
        rfl
      · dsimp only
        rw [htab, hw, hfold, hfold, htake, hr3, hr4]
        rfl
    · rw [ite_eq_right (by simp [hj1])] at hb
      have hb := lget_bind hb
      obtain ⟨_, hb⟩ := lsendMsg_bind _ hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      refine ⟨_, rfl, by simp [hsz], fun i hi => ?_, by rw [ite_eq_right hj1]; omega, hoods, by simp [hmsz],
        fun i hi hik => ?_, fun _ => ⟨?_, ?_⟩, fun h => absurd h hj1⟩
      · exact push_get!_of (fun i hi => hch i (hsz ▸ hi)) (by rw [hsz, hl₀]) i (by omega)
      · dsimp only
        by_cases hij : i < j + 1
        · rw [Pre.get_lt (Pre.push _ _) (by omega)]
          exact hmsg i hij hik
        · obtain rfl : i = j + 1 := by omega
          rw [← hmsz, Pre.get (Pre.refl _), htab, hw, hfold, hfold, hmsz]
      · dsimp only
        rw [htab, hfold]
      · dsimp only
        rw [hw, hfold]
  obtain ⟨hsz, hch, -, hoods, hmsz, hmsg, -, hfin⟩ := hI
  obtain ⟨hrho, he0, he1, htP⟩ := hfin rfl
  obtain ⟨hgr, hR⟩ := (show grL.T _ (fun R' => R'.2.2.2.2 = (s.2.2, s.1, s.2.1, coinOf (resp lA.size) 0,
      mleF tb ((Array.range sch.n).map (coinOf (resp l.size))) + hv[lay.lgOod 0]!)) from by
    repeat (first
      | extract_lets
      | (with_reducible apply Total.bind')
      | (with_reducible intro _)
      | (exact grL.pure _ rfl)
      | (with_reducible apply Total.pure')
      | (with_reducible apply grL_get)
      | (with_reducible apply grL_modify; intro s; unfold Grow; refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
        (exact Pre.refl _))
      | (with_reducible apply grL_liftT)
      | (with_reducible apply grL_powP)
      | (with_reducible apply grL_sendMsg)
      | (with_reducible apply grL_challengeP)
      | (with_reducible apply grL_oodIntroP)
      | (with_reducible apply grL_queryP)
      | (with_reducible apply Total.forIn_range)
      | (with_reducible apply Total.forIn_array)
      | (with_reducible apply Total.ite)
      | split)) _ _ _ _ _ _ _ h1
  simp only [Prod.mk.injEq] at hR
  obtain ⟨rfl, rfl, rfl, rfl, rfl⟩ := hR
  have hpre : Pre ((#[] : Array F128).push
      (mleF tb ((Array.range sch.n).map (coinOf (resp l.size))) + hv[lay.lgOod 0]!)) stF.oods := by
    rw [show ∀ x : F128, (#[] : Array F128).push x = #[x] from fun _ => rfl, ← hoods]
    exact hgr.2.1
  have hoodF : stF.oods[0]! = _ := Pre.get hpre
  refine ⟨?_, hoodF, ?_, fun i hi => ?_, he0, he1, htP⟩
  · show ({ target := T, ood := #[(coinOf (resp lA.size) 0, _)], challenges := s.1, rho := s.2.1 } : Zk.L0Rec) =
      { target := T, ood := #[(coinOf (resp (l.size + 1)) 0, stF.oods[0]!)], challenges := _, rho := _ }
    rw [hlA, hoodF, hrho, toArray_of_get! hsz hch]
    rfl
  · show sch.k0 ≤ stF.msgs.size
    have := Pre.size_le hgr.1
    omega
  · show stF.msgs[i]! = _
    rw [Pre.get_lt hgr.1 (by omega)]
    exact hmsg i (by omega) hi

end FlockComplete.L0
