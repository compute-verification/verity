import Proofs.Flock.Claims.Complete.Reps

/-!
# What the honest rep's steps keep: a Hoare layer on `Total`

`Total.H C x pre post`: from every state where `pre` holds, the stateful computation `x` returns (`C.T`), with `post` of
its value and the state after. `H.pure`, `H.bind`, `H.get`, `H.modify`, `H.lift` and `H.forIn_upto` (a loop over
`[0:n]` that keeps an invariant of its value and state) walk a computation by its steps.

At Ligerito's state, `meas` is what the opening's lengths read: its lists' sizes and the folded table's size, and whether
every grinding nonce is `0` (`Ms`, `zeros_of_meas`).
Each of Ligerito's steps moves it by a fixed amount: `sendMsg_H` (one message), `powP_H0`, `powP_H1` and `powP_H2` (one
query, consistency or claim nonce), `challengeP_H` (one message, the table halved, then doubled where it splits),
`oodIntroP_H` (a message, a claim nonce and an out-of-domain value) and `queryP_H` (a query and a consistency nonce, and
the opening at the level's positions of some coins).
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy FlockSoundness.Refine

namespace Total

/-- **`x` returns from every state `pre` holds at**, with `post` of its value and the state after. -/
def H {M : Type → Type} [Monad M] [MonadLiftT V M] (C : Total M) {σ α : Type} (x : StateT σ M α)
    (pre : σ → Prop) (post : α → σ → Prop) : Prop :=
  ∀ s, pre s → C.T (x s) fun p => post p.1 p.2

theorem H.pure {M : Type → Type} [Monad M] [MonadLiftT V M] {C : Total M} {σ α : Type} {pre : σ → Prop}
    {post : α → σ → Prop} (a : α) (h : ∀ s, pre s → post a s) :
    C.H (Pure.pure a : StateT σ M α) pre post :=
  fun s hs => C.pure (a, s) (h s hs)

theorem H.bind {M : Type → Type} [Monad M] [MonadLiftT V M] {C : Total M} {σ α β : Type} {x : StateT σ M α}
    {f : α → StateT σ M β} {pre : σ → Prop} {mid : α → σ → Prop} {post : β → σ → Prop}
    (hx : C.H x pre mid) (hf : ∀ a, C.H (f a) (mid a) post) : C.H (x >>= f) pre post :=
  fun s hs => C.bind (hx s hs) fun p hp => hf p.1 p.2 hp

theorem H.pre {M : Type → Type} [Monad M] [MonadLiftT V M] {C : Total M} {σ α : Type} {x : StateT σ M α}
    {pre pre' : σ → Prop} {post : α → σ → Prop} (h : C.H x pre post) (hp : ∀ s, pre' s → pre s) :
    C.H x pre' post :=
  fun s hs => h s (hp s hs)

theorem H.mono {M : Type → Type} [Monad M] [MonadLiftT V M] {C : Total M} {σ α : Type} {x : StateT σ M α}
    {pre : σ → Prop} {post post' : α → σ → Prop} (h : C.H x pre post) (hp : ∀ a s, post a s → post' a s) :
    C.H x pre post' :=
  fun s hs => C.mono (h s hs) fun p h => hp p.1 p.2 h

theorem H.get {M : Type → Type} [Monad M] [MonadLiftT V M] {C : Total M} {σ : Type} {pre : σ → Prop} :
    C.H (MonadState.get : StateT σ M σ) pre fun a s => a = s ∧ pre s :=
  fun s hs => C.pure (s, s) ⟨rfl, hs⟩

theorem H.get_bind {M : Type → Type} [Monad M] [MonadLiftT V M] {C : Total M} {σ β : Type}
    {f : σ → StateT σ M β} {pre : σ → Prop} {post : β → σ → Prop} (hf : ∀ a, C.H (f a) pre post) :
    C.H (MonadState.get >>= f) pre post :=
  H.bind H.get fun a => H.pre (hf a) fun _ h => h.2

theorem H.modify {M : Type → Type} [Monad M] [MonadLiftT V M] {C : Total M} {σ : Type} {pre : σ → Prop}
    (f : σ → σ) : C.H (_root_.modify f : StateT σ M PUnit) pre fun _ s => ∃ s₀, pre s₀ ∧ s = f s₀ :=
  fun s hs => C.pure (PUnit.unit, f s) ⟨s, hs, rfl⟩

theorem H.lift {M : Type → Type} [Monad M] [MonadLiftT V M] {C : Total M} {σ α : Type} {pre : σ → Prop}
    {x : M α} {Q : α → Prop} (hx : C.T x Q) : C.H (StateT.lift x : StateT σ M α) pre fun a s => pre s ∧ Q a :=
  fun s hs => C.bind hx fun a ha => C.pure (a, s) ⟨hs, ha⟩

theorem H.lift' {M : Type → Type} [Monad M] [MonadLiftT V M] {C : Total M} {σ α : Type} {pre : σ → Prop}
    {x : M α} (hx : C.T x fun _ => True) : C.H (StateT.lift x : StateT σ M α) pre fun _ s => pre s :=
  fun s hs => C.bind hx fun a _ => C.pure (a, s) hs

/-- A loop over `[0:n]` that keeps `I` of its value and state, at the steps taken. -/
theorem H.forIn_upto {M : Type → Type} [Monad M] [MonadLiftT V M] {C : Total M} {σ β : Type} (n : Nat) (b : β)
    {f : Nat → β → StateT σ M (ForInStep β)} (I : Nat → β → σ → Prop)
    (h : ∀ i < n, ∀ b, C.H (f i b) (I i b) fun r s => ∃ b', r = .yield b' ∧ I (i + 1) b' s) :
    C.H (forIn [0:n] b f) (I 0 b) (I n) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices ∀ j s (b : β), s + j = n → C.H (forIn (List.range' s j 1) b f) (I s b) (I n) from this n 0 b (by omega)
  intro j
  induction j with
  | zero =>
    intro s b hs
    rw [List.range'_zero, List.forIn_nil]
    exact H.pure b fun _ h => by rw [← hs]; exact h
  | succ j ih =>
    intro s b hs
    rw [List.range'_succ, List.forIn_cons]
    refine H.bind (h s (by omega) b) fun r => ?_
    intro t ht
    obtain ⟨b', rfl, hb'⟩ := ht
    exact ih (s + 1) b' (by omega) t hb'

end Total

/-- What the Ligerito walk tracks of its state: its lists' lengths, the folded table's size, and whether every grinding
nonce is `0` (R4). -/
structure Ms where
  msgs : ℕ
  qn : ℕ
  cn : ℕ
  cbn : ℕ
  oods : ℕ
  tab : ℕ
  z : Bool

def meas (s : LigState) : Ms :=
  ⟨s.msgs.size, s.qn.size, s.cn.size, s.cbn.size, s.oods.size, s.tab.size,
    s.qn.all (· == 0) && s.cn.all (· == 0) && s.cbn.all (· == 0)⟩

theorem zeros_of_meas {s : LigState} (h : (meas s).z = true) :
    (∀ x ∈ s.qn, x = 0) ∧ (∀ x ∈ s.cbn, x = 0) ∧ (∀ x ∈ s.cn, x = 0) := by
  simp only [meas, Bool.and_eq_true, Array.all_eq_true', beq_iff_eq] at h
  exact ⟨h.1.1, h.2, h.1.2⟩

theorem H_modify (f : LigState → LigState) (g : Ms → Ms) (hf : ∀ s, meas (f s) = g (meas s)) (v : Ms) :
    totT.H (modify f : LP PS PUnit) (fun s => meas s = v) fun _ s => meas s = g v :=
  Total.H.mono (Total.H.modify f) fun _ _ ⟨s₀, h₀, e⟩ => by rw [e, hf, h₀]

theorem H_liftT {α : Type} {x : TMW PS α} {pre : LigState → Prop} (hx : totT.T x fun _ => True) :
    totT.H (liftT x) pre fun _ s => pre s :=
  Total.H.lift' hx

theorem H_liftTM {pre : LigState → Prop} (x : TM Unit) (hx : ∀ t, ∃ p, x t = .ok p) :
    totT.H (liftT (Flock.liftTM x : TMW PS Unit)) pre fun _ s => pre s :=
  H_liftT (Total.liftTM _ x hx)

theorem sendMsg_H (m : F256 × F256) (v : Ms) :
    totT.H (sendMsg (M := PS) m) (fun s => meas s = v) fun _ s => meas s = { v with msgs := v.msgs + 1 } := by
  unfold sendMsg
  refine Total.H.bind (H_modify _ (fun v => { v with msgs := v.msgs + 1 }) (fun s => by simp [meas]) v) fun _ => ?_
  exact Total.H.bind (H_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => H_liftTM _ fun _ => ⟨_, rfl⟩

theorem powP_H (nonce : ℕ → ℕ → UInt64) (kind : ℕ) (v : Ms) :
    totT.H (powP (M := PS) nonce kind) (fun s => meas s = v) fun _ s => meas s =
      match kind with
      | 0 => { v with qn := v.qn + 1 }
      | 1 => { v with cn := v.cn + 1 }
      | _ => { v with cbn := v.cbn + 1 } := by
  unfold powP
  refine Total.H.bind Total.H.get fun st => ?_
  refine Total.H.bind (Total.H.modify _) fun _ => ?_
  refine Total.H.mono (H_liftTM _ fun _ => ⟨_, rfl⟩) ?_
  rintro _ s ⟨s₀, ⟨rfl, rfl⟩, rfl⟩
  rcases kind with _ | _ | k <;> simp [meas, Array.all_push]

theorem powP_H0 (nonce : ℕ → ℕ → UInt64) (v : Ms) :
    totT.H (powP (M := PS) nonce 0) (fun s => meas s = v) fun _ s => meas s = { v with qn := v.qn + 1 } :=
  powP_H nonce 0 v

theorem powP_H1 (nonce : ℕ → ℕ → UInt64) (v : Ms) :
    totT.H (powP (M := PS) nonce 1) (fun s => meas s = v) fun _ s => meas s = { v with cn := v.cn + 1 } :=
  powP_H nonce 1 v

theorem powP_H2 (nonce : ℕ → ℕ → UInt64) (v : Ms) :
    totT.H (powP (M := PS) nonce 2) (fun s => meas s = v) fun _ s => meas s = { v with cbn := v.cbn + 1 } :=
  powP_H nonce 2 v

theorem foldK_size (r : F256) (a : Array F256) : (foldK r a).size = a.size / 2 := by simp [foldK]

theorem embedF_size (g : Array F128) : (embedF g).size = g.size := by simp [embedF]

theorem splitK_size (f : Array F256) : (splitK f).size = 2 * f.size := by
  suffices ∀ (l : List F256) (acc : Array F128), (l.foldl (fun acc v => (acc.push v.c0).push v.c1) acc).size =
      acc.size + 2 * l.length by
    simpa [splitK, ← Array.foldl_toList] using this f.toList #[]
  intro l
  induction l with
  | nil => simp
  | cons a l ih => intro acc; rw [List.foldl_cons, ih]; simp; omega

theorem challengeP_H (resp : ℕ → Claims.Msg) (split : Bool) (v : Ms) :
    totT.H (challengeP (asksS resp) split) (fun s => meas s = v) fun _ s =>
      meas s = { v with msgs := v.msgs + 1, tab := if split then 2 * (v.tab / 2) else v.tab / 2 } := by
  unfold challengeP
  refine Total.H.bind (H_liftT (sampleKP_tot resp)) fun r => ?_
  refine Total.H.bind (H_modify _ (fun v => { v with tab := v.tab / 2 }) (fun s => by simp [meas, foldK_size]) v)
    fun _ => ?_
  have hk : ∀ w : Ms, totT.H (do let st ← get; sendMsg (roundPoly st.tab st.w); pure r : LP PS F256)
      (fun s => meas s = w) fun _ s => meas s = { w with msgs := w.msgs + 1 } := fun w =>
    Total.H.get_bind fun st => Total.H.bind (sendMsg_H _ _) fun _ => Total.H.pure _ fun s h => h
  cases split
  · simpa using hk _
  · dsimp only
    simp only [↓reduceIte]
    refine Total.H.bind (H_modify _ (fun w => { w with tab := 2 * w.tab }) (fun s => by
      simp [meas, embedF_size, splitK_size]) _) fun _ => ?_
    exact hk _

theorem oodIntroP_H (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ) (v : Ms) :
    totT.H (oodIntroP (asksS resp) nonce g dim) (fun s => meas s = v) fun _ s =>
      meas s = { v with msgs := v.msgs + 1, cbn := v.cbn + 1, oods := v.oods + 1 } := by
  unfold oodIntroP
  refine Total.H.bind (H_liftT (sampleVecP_tot resp dim)) fun z => ?_
  refine Total.H.bind (H_modify _ (fun v => { v with oods := v.oods + 1 }) (fun s => by simp [meas]) v) fun _ => ?_
  refine Total.H.bind (H_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => ?_
  refine Total.H.bind (sendMsg_H _ _) fun _ => ?_
  refine Total.H.bind (powP_H2 nonce _) fun _ => ?_
  refine Total.H.bind (H_liftT (sampleP_tot resp)) fun b => ?_
  exact H_modify _ id (by intro s; rfl) _

theorem queryP_H (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level)
    (cm : Commit) (v : Ms) :
    totT.H (queryP (asksS resp) nonce ms lv cm) (fun s => meas s = v) fun a s =>
      meas s = { v with qn := v.qn + 1, cn := v.cn + 1 } ∧ ∃ coins, a.1 = openP ms cm lv (positions lv coins) := by
  unfold queryP
  refine Total.H.bind (powP_H0 nonce v) fun _ => ?_
  refine Total.H.bind (Total.H.lift (Q := fun _ => True) (sampleVecP_tot resp lv.q)) fun coins => ?_
  refine Total.H.bind (Total.H.pre (powP_H1 nonce _) fun _ h => h.1) fun _ => ?_
  refine Total.H.bind (H_liftT (sampleVecP_tot resp _)) fun alpha => ?_
  exact Total.H.pure _ fun s h => ⟨h, coins, rfl⟩

end FlockComplete.JsonRT
