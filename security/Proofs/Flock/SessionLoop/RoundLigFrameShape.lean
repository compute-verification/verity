module

public import Proofs.Flock.Claims.ZK.FrameOpenCut
public import Proofs.Flock.Claims.ZK.FrameRounds
public import Proofs.Flock.Claims.Complete.Link
public import Proofs.Flock.Verifier.Hash

/-!
# Ligerito's bodies from level 0's close on (`ligeritoPZk_shape`)

`ligeritoPZk` answered by `asksCut coin N` from a log of `s` bodies sends level 0's `k_0 + 4` bodies, then at log index
`P = s + k_0 + 4` level 0's closing round, and after it only rounds outside every masked phase. Where the cut reaches
the close (`P ≤ N`), each body the log holds from `P` on (`LogQ`) is good frames (`Good`: ones a round holds before its
squeeze, inside a phase, no domain frame), after the closing label at `P`, then a squeeze of `n ≠ 0` coins below
`2 ^ 64` as the frame takes it (`Body`).

* `HT`, `HL`: from a precondition on the transcript and the log, the log is `LogQ`'s wherever the run stops, and where it
  returns it ends in the postcondition; through the transcript and through the prover's state. Binds, pure, throws, the
  state's steps, branches and loops with an invariant carry them.
* Level 0 (`Sz`): the log's size, which each squeeze of coins moves by one; past `ρ_1`'s squeeze at `P` the pending bytes
  are empty, and the closing label starts `Inv`.
* Past the close (`Inv`): each absorb appends a good frame, a squeeze of `0` coins its frame, and a squeeze of coins sends
  a `Body`; a commitment's cap is below `2 ^ 64` bytes at a level of `64 · 2 ^ d < 2 ^ 64`.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockComplete.LigShape

open Flock Transcript Replay
open FlockSoundness.Refine (Frame encs label_enc absorbBytes_enc absorbSlice_enc absorbF_enc framed_enc)
open Claims.ZKServer (midFrame plainFrame noDom)

/-! ## The bodies -/

/-- **A frame a round past level 0's close holds before its squeeze**: one the frame takes before a squeeze
(`midFrame`), inside a phase (`plainFrame`), and no domain frame. -/
def Good (f : Frame) : Prop := midFrame f ∧ plainFrame f ∧ noDom f

/-- **A body past level 0's close**: the frames `pre`, good frames, then a squeeze of `n ≠ 0` coins below `2 ^ 64` as the
frame takes it. -/
def Body (pre : List Frame) (b : Claims.Msg) : Prop :=
  ∃ fs k n, b = encs (pre ++ fs) ++ (Frame.sq k n).enc ∧ (∀ f ∈ fs, Good f) ∧ (k = 2 ∨ (k = 1 ∧ n = 1)) ∧ n ≠ 0 ∧
    n < 2 ^ 64

/-- The frames before a round's good ones: level 0's closing label at the close round `P`, none after it. -/
def preAt (P p : ℕ) : List Frame := if p = P then [.label Zk.LABEL_CLOSE.toUTF8] else []

/-- **A log's bodies from the close round `P` on.** -/
def LogQ (P : ℕ) (L : Array Claims.Msg) : Prop :=
  ∀ p (hp : p < L.size), P ≤ p → Body (preAt P p) L[p]

/-- **Past level 0's close at round `P`**: the log reached it, its bodies from it on are `LogQ`'s, and the pending bytes
are the closing label at round `P`, then good frames. -/
def Inv (P : ℕ) (t : Transcript) (l : Array Claims.Msg) : Prop :=
  P ≤ l.size ∧ LogQ P l ∧ ∃ fs, t.pending.data.toList = encs (preAt P l.size ++ fs) ∧ ∀ f ∈ fs, Good f

/-- Level 0's log of `s` bodies. -/
def Sz (s : ℕ) (_ : Transcript) (l : Array Claims.Msg) : Prop := l.size = s

variable {P : ℕ}

theorem logQ_le {l : Array Claims.Msg} (h : l.size ≤ P) : LogQ P l :=
  fun p hp hP => absurd hp (by omega)

theorem logQ_push {l : Array Claims.Msg} {b : Claims.Msg} (h : LogQ P l) (hb : P ≤ l.size → Body (preAt P l.size) b) :
    LogQ P (l.push b) := by
  intro p hp hP
  rw [Array.size_push] at hp
  rcases Nat.lt_or_ge p l.size with h1 | h1
  · rw [Array.getElem_push_lt h1]
    exact h p h1 hP
  · obtain rfl : p = l.size := by omega
    rw [Array.getElem_push_eq]
    exact hb hP

theorem good_sq0 : Good (.sq 2 0) := ⟨⟨rfl, rfl⟩, ⟨rfl, rfl⟩, trivial⟩

/-- A squeeze's body from pending bytes `pre`, then good frames. -/
theorem body_framed {t : Transcript} {pre fs : List Frame} (ht : t.pending.data.toList = encs (pre ++ fs))
    (hfs : ∀ f ∈ fs, Good f) {kind : UInt8} {k : ℕ} (hk : kind = 2 ∨ (kind = 1 ∧ k = 1)) (h0 : k ≠ 0)
    (h64 : k < 2 ^ 64) : Body pre (Claims.toMsg (framed t kind k)) :=
  ⟨fs, kind, k, by rw [show Claims.toMsg (framed t kind k) = (framed t kind k).data.toList from rfl, framed_enc, ht],
    hfs, hk, h0, h64⟩

/-! ## Pre- and postconditions through the transcript -/

variable {α β σ : Type}

/-- **`x` from `Pre`**: wherever it stops its log is `LogQ`'s, and where it returns `a` it ends in `Post a`. -/
def HT (P : ℕ) (Pre : Transcript → Array Claims.Msg → Prop) (Post : α → Transcript → Array Claims.Msg → Prop)
    (x : TMW PS α) : Prop :=
  ∀ t l, Pre t l → (∀ e L, (x.run t).go l = (.error e, L) → LogQ P L) ∧
    ∀ a t' L, (x.run t).go l = (.ok (a, t'), L) → Post a t' L

theorem ht_bind {Pre : Transcript → Array Claims.Msg → Prop} {Mid : α → Transcript → Array Claims.Msg → Prop}
    {Post : β → Transcript → Array Claims.Msg → Prop} {x : TMW PS α} {f : α → TMW PS β} (hx : HT P Pre Mid x)
    (hf : ∀ a, HT P (Mid a) Post (f a)) : HT P Pre Post (x >>= f) := by
  intro t l hp
  obtain ⟨he, hok⟩ := hx t l hp
  rw [StateT.run_bind, go_bind]
  rcases e : (x.run t).go l with ⟨r, L⟩
  cases r with
  | error err =>
    refine ⟨fun e' L' h => ?_, fun a t' L' h => ?_⟩
    · simp only [Prod.mk.injEq] at h
      rw [← h.2]
      exact he err L e
    · simp at h
  | ok p =>
    obtain ⟨a, t'⟩ := p
    exact hf a t' L (hok a t' L e)

theorem ht_pure {Pre : Transcript → Array Claims.Msg → Prop} {Post : α → Transcript → Array Claims.Msg → Prop} {a : α}
    (hP : ∀ t l, Pre t l → Post a t l) : HT P Pre Post (pure a) := by
  intro t l hp
  refine ⟨fun e L h => ?_, fun a' t' L h => ?_⟩
  · have e' : ((Except.ok (a, t) : Except String (α × Transcript)), l) = (.error e, L) := h
    cases e'
  · have e : ((Except.ok (a, t) : Except String (α × Transcript)), l) = (.ok (a', t'), L) := h
    simp only [Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨⟨rfl, rfl⟩, rfl⟩ := e
    exact hP t l hp

theorem ht_mono {Pre Pre' : Transcript → Array Claims.Msg → Prop} {Post Post' : α → Transcript → Array Claims.Msg → Prop}
    {x : TMW PS α} (h : HT P Pre Post x) (hpre : ∀ t l, Pre' t l → Pre t l)
    (hpost : ∀ a t l, Post a t l → Post' a t l) : HT P Pre' Post' x := fun t l hp =>
  ⟨(h t l (hpre t l hp)).1, fun a t' L e => hpost a t' L ((h t l (hpre t l hp)).2 a t' L e)⟩

/-- A transcript action, where `Pre` stops only at a `LogQ` log and holds after the action. -/
theorem ht_liftTM {Pre : Transcript → Array Claims.Msg → Prop} (x : TM α) (he : ∀ t l, Pre t l → LogQ P l)
    (hx : ∀ t a t' l, x t = .ok (a, t') → Pre t l → Pre t' l) : HT P Pre (fun _ => Pre) (liftTM x) := by
  intro t l hp
  rw [liftTM_go]
  refine ⟨fun e L h => ?_, fun a t' L h => ?_⟩
  · simp only [Prod.mk.injEq] at h
    rw [← h.2]
    exact he t l hp
  · simp only [Prod.mk.injEq] at h
    rw [← h.2]
    exact hx t a t' l h.1 hp

/-- A computation of the pure session that sends nothing, lifted to the transcript. -/
theorem ht_liftPS {Pre : Transcript → Array Claims.Msg → Prop} {Q : α → Prop} (y : PS α)
    (hy : ∀ l, (y.go l).2 = l) (he : ∀ t l, Pre t l → LogQ P l) (hq : ∀ l a L, y.go l = (.ok a, L) → Q a) :
    HT P Pre (fun a t l => Pre t l ∧ Q a) (StateT.lift y) := by
  intro t l hp
  have e0 : ((StateT.lift y : TMW PS α).run t).go l = (y >>= fun a => Pure.pure (a, t)).go l := rfl
  rw [e0, go_bind]
  have hyl := hy l
  rcases e : y.go l with ⟨r, L⟩
  rw [e] at hyl
  dsimp only at hyl
  subst hyl
  cases r with
  | error err =>
    refine ⟨fun e' L' h => ?_, fun a t' L' h => ?_⟩
    · have e'' : ((Except.error err : Except String (α × Transcript)), L) = (.error e', L') := h
      simp only [Prod.mk.injEq] at e''
      rw [← e''.2]
      exact he t _ hp
    · have e'' : ((Except.error err : Except String (α × Transcript)), L) = (.ok (a, t'), L') := h
      cases e''
  | ok a =>
    refine ⟨fun e' L' h => ?_, fun a' t' L' h => ?_⟩
    · have e'' : ((Except.ok (a, t) : Except String (α × Transcript)), L) = (.error e', L') := h
      cases e''
    · have e' : ((Except.ok (a, t) : Except String (α × Transcript)), L) = (.ok (a', t'), L') := h
      simp only [Prod.mk.injEq, Except.ok.injEq] at e'
      obtain ⟨⟨rfl, rfl⟩, rfl⟩ := e'
      exact ⟨hp, hq _ a _ e⟩

/-! ## Through the prover's state -/

/-- `HT` from every prover state, the postcondition on the value. -/
def HL (P : ℕ) (Pre : Transcript → Array Claims.Msg → Prop) (Post : α → Transcript → Array Claims.Msg → Prop)
    (x : LP PS α) : Prop :=
  ∀ s, HT P Pre (fun p => Post p.1) (x.run s)

theorem hl_bind {Pre : Transcript → Array Claims.Msg → Prop} {Mid : α → Transcript → Array Claims.Msg → Prop}
    {Post : β → Transcript → Array Claims.Msg → Prop} {x : LP PS α} {f : α → LP PS β} (hx : HL P Pre Mid x)
    (hf : ∀ a, HL P (Mid a) Post (f a)) : HL P Pre Post (x >>= f) := fun s => by
  rw [StateT.run_bind]
  exact ht_bind (hx s) fun p => hf p.1 p.2

theorem hl_pure {Pre : Transcript → Array Claims.Msg → Prop} {Post : α → Transcript → Array Claims.Msg → Prop} {a : α}
    (h : ∀ t l, Pre t l → Post a t l) : HL P Pre Post (pure a) := fun s => by
  rw [StateT.run_pure]
  exact ht_pure h

theorem hl_mono {Pre Pre' : Transcript → Array Claims.Msg → Prop} {Post Post' : α → Transcript → Array Claims.Msg → Prop}
    {x : LP PS α} (h : HL P Pre Post x) (hpre : ∀ t l, Pre' t l → Pre t l)
    (hpost : ∀ a t l, Post a t l → Post' a t l) : HL P Pre' Post' x :=
  fun s => ht_mono (h s) hpre fun p t l hp => hpost p.1 t l hp

theorem hl_liftT {Pre : Transcript → Array Claims.Msg → Prop} {Post : α → Transcript → Array Claims.Msg → Prop}
    {x : TMW PS α} (h : HT P Pre Post x) : HL P Pre Post (liftT x) := fun s => by
  unfold liftT
  rw [StateT.run_lift]
  exact ht_bind h fun a => ht_pure fun _ _ hp => hp

theorem hl_get {Pre : Transcript → Array Claims.Msg → Prop} : HL P Pre (fun _ => Pre) (get : LP PS LigState) :=
  fun s => by
    rw [StateT.run_get]
    exact ht_pure fun _ _ hp => hp

theorem hl_set {Pre : Transcript → Array Claims.Msg → Prop} (s' : LigState) :
    HL P Pre (fun _ => Pre) (set s' : LP PS PUnit) := fun s => by
  rw [StateT.run_set]
  exact ht_pure fun _ _ hp => hp

theorem hl_modify {Pre : Transcript → Array Claims.Msg → Prop} (F : LigState → LigState) :
    HL P Pre (fun _ => Pre) (modify F : LP PS PUnit) := fun s => by
  rw [StateT.run_modify]
  exact ht_pure fun _ _ hp => hp

theorem hl_ite {Pre : Transcript → Array Claims.Msg → Prop} {Post : α → Transcript → Array Claims.Msg → Prop}
    (c : Prop) [Decidable c] {x x' : LP PS α} (h1 : c → HL P Pre Post x) (h2 : ¬ c → HL P Pre Post x') :
    HL P Pre Post (if c then x else x') := by
  by_cases hc : c
  · rw [ite_eq_left hc]
    exact h1 hc
  · rw [ite_eq_right hc]
    exact h2 hc

theorem hl_throw_bind {Pre : Transcript → Array Claims.Msg → Prop} {Post : β → Transcript → Array Claims.Msg → Prop}
    (e : String) {f : α → LP PS β} (he : ∀ t l, Pre t l → LogQ P l) : HL P Pre Post (throw e >>= f) := by
  intro s t l hp
  have h' : ((((throw e : LP PS α) >>= f).run s).run t).go l = (.error e, l) := rfl
  rw [h']
  refine ⟨fun e' L h => ?_, fun a t' L h => ?_⟩
  · simp only [Prod.mk.injEq] at h
    rw [← h.2]
    exact he t l hp
  · simp at h

/-- **A loop over `[0:n]`**, step `i` from `J i u` yielding a value `v` with `J (i + 1) v`. -/
theorem hl_forIn (n : ℕ) (J : ℕ → σ → Transcript → Array Claims.Msg → Prop) (body : ℕ → σ → LP PS (ForInStep σ))
    (hb : ∀ i < n, ∀ u, HL P (J i u) (fun r t l => ∃ v, r = .yield v ∧ J (i + 1) v t l) (body i u)) (u : σ) :
    HL P (J 0 u) (fun r => J n r) (forIn [:n] u body) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (u : σ), a + k = n → HL P (J a u) (fun r => J n r) (forIn (List.range' a k) u body) from
    key n 0 u (by omega)
  intro k
  induction k with
  | zero =>
    intro a u ha
    obtain rfl : a = n := by omega
    exact hl_pure fun _ _ hp => hp
  | succ k ih =>
    intro a u ha
    rw [List.range'_succ, List.forIn_cons]
    refine hl_bind (hb a (by omega) u) fun r => ?_
    intro s t l hr
    obtain ⟨v, rfl, hv⟩ := hr
    exact ih (a + 1) v (by omega) s t l hv

theorem hl_forIn_bind {n : ℕ} (J : ℕ → σ → Transcript → Array Claims.Msg → Prop)
    {Pre : Transcript → Array Claims.Msg → Prop} {Post : β → Transcript → Array Claims.Msg → Prop}
    {body : ℕ → σ → LP PS (ForInStep σ)} {u : σ} {f : σ → LP PS β} (h0 : ∀ t l, Pre t l → J 0 u t l)
    (hb : ∀ i < n, ∀ u, HL P (J i u) (fun r t l => ∃ v, r = .yield v ∧ J (i + 1) v t l) (body i u))
    (hf : ∀ r, HL P (J n r) Post (f r)) : HL P Pre Post (forIn [:n] u body >>= f) :=
  hl_mono (hl_bind (hl_forIn n J body hb u) hf) h0 fun _ _ _ h => h

/-- A fact the precondition carries beside `Pre`, taken out. -/
theorem hl_pre_and {Q : Prop} {Pre : Transcript → Array Claims.Msg → Prop}
    {Post : α → Transcript → Array Claims.Msg → Prop} {x : LP PS α} (h : Q → HL P Pre Post x) :
    HL P (fun t l => Q ∧ Pre t l) Post x := fun s t l hp => h hp.1 s t l hp.2

theorem hl_pre_and' {Q : Prop} {Pre : Transcript → Array Claims.Msg → Prop}
    {Post : α → Transcript → Array Claims.Msg → Prop} {x : LP PS α} (h : Q → HL P Pre Post x) :
    HL P (fun t l => Pre t l ∧ Q) Post x := fun s t l hp => h hp.2 s t l hp.1

theorem hl_forIn_list {Pre : Transcript → Array Claims.Msg → Prop} {τ : Type} (l : List τ)
    (body : τ → σ → LP PS (ForInStep σ))
    (hb : ∀ x u, HL P Pre (fun r t l => (∃ v, r = .yield v) ∧ Pre t l) (body x u)) (u : σ) :
    HL P Pre (fun _ => Pre) (forIn l u body) := by
  induction l generalizing u with
  | nil => exact hl_pure fun _ _ hp => hp
  | cons x l ih =>
    rw [List.forIn_cons]
    refine hl_bind (hb x u) fun r => ?_
    intro s t L hr
    obtain ⟨⟨v, rfl⟩, hp⟩ := hr
    exact ih v s t L hp

theorem hl_forIn_array {Pre : Transcript → Array Claims.Msg → Prop} {τ : Type} (l : Array τ)
    (body : τ → σ → LP PS (ForInStep σ))
    (hb : ∀ x u, HL P Pre (fun r t l => (∃ v, r = .yield v) ∧ Pre t l) (body x u)) (u : σ) :
    HL P Pre (fun _ => Pre) (forIn l u body) := by
  rw [← Array.forIn_toList]
  exact hl_forIn_list l.toList body hb u

theorem ht_run_bind {Pre : Transcript → Array Claims.Msg → Prop} {Mid : α → Transcript → Array Claims.Msg → Prop}
    {Post : β → Transcript → Array Claims.Msg → Prop} {p : LP PS α} {st : LigState} {f : α × LigState → TMW PS β}
    (hp : HL P Pre Mid p) (hf : ∀ a, HT P (Mid a.1) Post (f a)) : HT P Pre Post (p.run st >>= f) :=
  ht_bind (hp st) hf

/-! ## Level 0: the log's size -/

variable {coin : ℕ → ℕ → F128} {N : ℕ}

theorem sz_logQ {s : ℕ} (hs : s ≤ P) : ∀ t l, Sz s t l → LogQ P l := fun _ _ h => logQ_le (h ▸ hs)

/-- **A squeeze of coins at level 0** moves the log's size by one, and leaves no pending bytes. -/
theorem ht_sqA (kind : UInt8) {k : ℕ} (hk : k ≠ 0) {s : ℕ} (hsN : s < N) (hsP : s < P) :
    HT P (Sz s) (fun _ t l => l.size = s + 1 ∧ t.pending = ByteArray.empty)
      (squeezeP (asksCut coin N) kind k) := by
  intro t l hp
  have hp' : l.size = s := hp
  rw [LigLock.squeezeP_cut_ok coin kind hk t (by omega)]
  refine ⟨fun e L h => (by cases h), fun a t' L h => ?_⟩
  simp only [Prod.mk.injEq, Except.ok.injEq] at h
  obtain ⟨⟨-, rfl⟩, rfl⟩ := h
  exact ⟨by rw [Array.size_push, hp'], rfl⟩

theorem ht_sqA' (kind : UInt8) {k : ℕ} (hk : k ≠ 0) {s : ℕ} (hsN : s < N) (hsP : s < P) :
    HT P (Sz s) (fun _ => Sz (s + 1)) (squeezeP (asksCut coin N) kind k) :=
  ht_mono (ht_sqA kind hk hsN hsP) (fun _ _ h => h) fun _ _ _ h => h.1

theorem hl_sampleKPA {s : ℕ} (hsN : s < N) (hsP : s < P) :
    HL P (Sz s) (fun _ => Sz (s + 1)) (liftT (sampleKP (asksCut coin N))) := by
  unfold sampleKP
  exact hl_liftT (ht_bind (ht_sqA' 2 two_ne_zero hsN hsP) fun _ => ht_pure fun _ _ h => h)

theorem hl_samplePA {s : ℕ} (hsN : s < N) (hsP : s < P) :
    HL P (Sz s) (fun _ => Sz (s + 1)) (liftT (sampleP (asksCut coin N))) := by
  unfold sampleP
  exact hl_liftT (ht_bind (ht_sqA' 1 one_ne_zero hsN hsP) fun _ => ht_pure fun _ _ h => h)

theorem hl_sampleKPA' {s : ℕ} (hsN : s < N) (hsP : s < P) :
    HL P (Sz s) (fun _ t l => l.size = s + 1 ∧ t.pending = ByteArray.empty) (liftT (sampleKP (asksCut coin N))) := by
  unfold sampleKP
  exact hl_liftT (ht_bind (ht_sqA 2 two_ne_zero hsN hsP) fun _ => ht_pure fun _ _ h => h)

theorem hl_lTMA {s : ℕ} (hs : s ≤ P) (x : TM α) : HL P (Sz s) (fun _ => Sz s) (liftT (liftTM x) : LP PS α) :=
  hl_liftT (ht_liftTM x (sz_logQ hs) fun _ _ _ _ _ h => h)

theorem hl_sendMsgA {s : ℕ} (hs : s ≤ P) (m : F256 × F256) : HL P (Sz s) (fun _ => Sz s) (sendMsg m : LP PS Unit) := by
  unfold sendMsg
  exact hl_bind (hl_modify _) fun _ => hl_bind (hl_lTMA hs _) fun _ => hl_lTMA hs _

theorem hl_powPA {s : ℕ} (hs : s ≤ P) (nonce : ℕ → ℕ → UInt64) (kind : ℕ) :
    HL P (Sz s) (fun _ => Sz s) (powP nonce kind : LP PS Unit) := by
  unfold powP
  exact hl_bind hl_get fun _ => hl_bind (hl_set _) fun _ => hl_lTMA hs _

/-- **Level 0's closing label** at round `P`, after `ρ_1`'s squeeze: past the close. -/
theorem hl_closeLabel :
    HL P (fun t l => l.size = P ∧ t.pending = ByteArray.empty) (fun _ => Inv P)
      (liftT (liftTM (label Zk.LABEL_CLOSE)) : LP PS Unit) := by
  refine hl_liftT fun t l hp => ?_
  have hlab : label Zk.LABEL_CLOSE t = .ok ((), { t with pending := pLabel Zk.LABEL_CLOSE t.pending }) := rfl
  rw [liftTM_ok hlab]
  refine ⟨fun e L h => (by cases h), fun a t' L h => ?_⟩
  simp only [Prod.mk.injEq, Except.ok.injEq] at h
  obtain ⟨⟨-, rfl⟩, rfl⟩ := h
  obtain ⟨hl, ht⟩ := hp
  refine ⟨hl.ge, logQ_le hl.le, [], ?_, fun _ h => nomatch h⟩
  show (t.pending ++ header 2 0 Zk.LABEL_CLOSE.toUTF8.size ++ padded Zk.LABEL_CLOSE.toUTF8).data.toList = _
  rw [label_enc, ht, preAt, if_pos hl]
  simp [encs, Claims.ZKServer.empty_toList]

/-! ## Past the close: good frames and bodies -/

theorem inv_logQ : ∀ t l, Inv P t l → LogQ P l := fun _ _ h => h.2.1

/-- An action appending a good frame keeps `Inv`. -/
theorem ht_frameB (x : TM α)
    (hx : ∀ t a t', x t = .ok (a, t') → ∃ f, Good f ∧ t'.pending.data.toList = t.pending.data.toList ++ f.enc) :
    HT P (Inv P) (fun _ => Inv P) (liftTM x) :=
  ht_liftTM x inv_logQ fun t a t' l e hI => by
    obtain ⟨h1, h2, fs, hfs, hg⟩ := hI
    obtain ⟨f, hf, he⟩ := hx t a t' e
    refine ⟨h1, h2, fs ++ [f], ?_, fun g hg' => ?_⟩
    · rw [he, hfs]
      simp [encs]
    · rcases List.mem_append.1 hg' with hg' | hg'
      · exact hg g hg'
      · rw [List.mem_singleton.1 hg']; exact hf

theorem hl_lTMB (x : TM α)
    (hx : ∀ t a t', x t = .ok (a, t') → ∃ f, Good f ∧ t'.pending.data.toList = t.pending.data.toList ++ f.enc) :
    HL P (Inv P) (fun _ => Inv P) (liftT (liftTM x) : LP PS α) :=
  hl_liftT (ht_frameB x hx)

theorem hl_absorbF (v : F128) : HL P (Inv P) (fun _ => Inv P) (liftT (liftTM (absorbF v)) : LP PS Unit) :=
  hl_lTMB _ fun t _ _ h => by
    cases h
    exact ⟨.one v, ⟨trivial, trivial, trivial⟩, absorbF_enc _ _⟩

theorem hl_absorbK (v : F256) : HL P (Inv P) (fun _ => Inv P) (liftT (liftTM (absorbK v)) : LP PS Unit) :=
  hl_lTMB _ fun t _ _ h => by
    cases h
    exact ⟨.vec (#[v.c0, v.c1] : Array F128).toList, ⟨show [v.c0, v.c1].length < 2 ^ 64 by simp, trivial, trivial⟩,
      absorbSlice_enc _ _⟩

theorem hl_absorbSlice (vs : Array F128) (hn : vs.size < 2 ^ 64) :
    HL P (Inv P) (fun _ => Inv P) (liftT (liftTM (absorbSlice vs)) : LP PS Unit) :=
  hl_lTMB _ fun t _ _ h => by
    cases h
    exact ⟨.vec vs.toList, ⟨show vs.toList.length < 2 ^ 64 by simpa using hn, trivial, trivial⟩, absorbSlice_enc _ _⟩

theorem hl_absorbBytes (b : ByteArray) (hn : b.size < 2 ^ 64) :
    HL P (Inv P) (fun _ => Inv P) (liftT (liftTM (absorbBytes b)) : LP PS Unit) :=
  hl_lTMB _ fun t _ _ h => by
    cases h
    exact ⟨.bytes b, ⟨hn, trivial, trivial⟩, absorbBytes_enc _ _⟩

theorem hl_pow (v : UInt64) : HL P (Inv P) (fun _ => Inv P) (liftT (liftTM (pow v)) : LP PS Unit) :=
  hl_lTMB _ fun t _ _ h => by
    cases h
    have h8 : (le64 v).size = 8 := by
      have := congrArg List.length (FlockSoundness.Refine.le64_data v)
      simpa using this
    exact ⟨.bytes (le64 v), ⟨show (le64 v).size < 2 ^ 64 by rw [h8]; decide, trivial, trivial⟩,
      absorbBytes_enc _ _⟩

/-- **A squeeze past the close**: of `0` coins its frame stays pending; else it sends a `Body` and leaves no pending
bytes. -/
theorem ht_sqB (kind : UInt8) {k : ℕ} (hk : kind = 2 ∨ (kind = 1 ∧ k = 1)) (h64 : k < 2 ^ 64) :
    HT P (Inv P) (fun _ => Inv P) (squeezeP (asksCut coin N) kind k) := by
  intro t l hI
  obtain ⟨h1, h2, fs, hfs, hg⟩ := hI
  by_cases h0 : k = 0
  · subst h0
    have hk2 : kind = 2 := hk.resolve_right fun h => absurd h.2 (by decide)
    subst hk2
    have e : ((squeezeP (asksCut coin N) 2 0).run t).go l = (.ok (#[], { t with pending := framed t 2 0 }), l) := rfl
    rw [e]
    refine ⟨fun e L h => (by cases h), fun a t' L h => ?_⟩
    simp only [Prod.mk.injEq, Except.ok.injEq] at h
    obtain ⟨⟨-, rfl⟩, rfl⟩ := h
    refine ⟨h1, h2, fs ++ [.sq 2 0], ?_, fun g hg' => ?_⟩
    · show (framed t 2 0).data.toList = _
      rw [framed_enc, hfs]
      simp [encs]
    · rcases List.mem_append.1 hg' with hg' | hg'
      · exact hg g hg'
      · rw [List.mem_singleton.1 hg']; exact good_sq0
  · have hb := body_framed hfs hg hk h0 h64
    have hq : LogQ P (l.push (Claims.toMsg (framed t kind k))) := logQ_push h2 fun _ => hb
    by_cases hN : N ≤ l.size
    · rw [LigLock.squeezeP_cut_err coin kind h0 t hN]
      refine ⟨fun e L h => ?_, fun a t' L h => by cases h⟩
      simp only [Prod.mk.injEq] at h
      rw [← h.2]
      exact hq
    · rw [LigLock.squeezeP_cut_ok coin kind h0 t (by omega)]
      refine ⟨fun e L h => (by cases h), fun a t' L h => ?_⟩
      simp only [Prod.mk.injEq, Except.ok.injEq] at h
      obtain ⟨⟨-, rfl⟩, rfl⟩ := h
      refine ⟨by rw [Array.size_push]; omega, hq, [], ?_, fun _ h => nomatch h⟩
      rw [Array.size_push, preAt, _root_.if_neg (show l.size + 1 ≠ P by omega)]
      rfl

theorem hl_sampleP : HL P (Inv P) (fun _ => Inv P) (liftT (sampleP (asksCut coin N))) := by
  unfold sampleP
  exact hl_liftT (ht_bind (ht_sqB 1 (.inr ⟨rfl, rfl⟩) (by decide)) fun _ => ht_pure fun _ _ h => h)

theorem hl_sampleKP : HL P (Inv P) (fun _ => Inv P) (liftT (sampleKP (asksCut coin N))) := by
  unfold sampleKP
  exact hl_liftT (ht_bind (ht_sqB 2 (.inl rfl) (by decide)) fun _ => ht_pure fun _ _ h => h)

theorem hl_sampleVecP {k : ℕ} (h64 : k < 2 ^ 64) :
    HL P (Inv P) (fun _ => Inv P) (liftT (sampleVecP (asksCut coin N) k)) := by
  unfold sampleVecP
  exact hl_liftT (ht_sqB 2 (.inl rfl) h64)

theorem hl_sendMsg (m : F256 × F256) : HL P (Inv P) (fun _ => Inv P) (sendMsg m : LP PS Unit) := by
  unfold sendMsg
  exact hl_bind (hl_modify _) fun _ => hl_bind (hl_absorbK _) fun _ => hl_absorbK _

theorem hl_powP (nonce : ℕ → ℕ → UInt64) (kind : ℕ) : HL P (Inv P) (fun _ => Inv P) (powP nonce kind : LP PS Unit) := by
  unfold powP
  exact hl_bind hl_get fun _ => hl_bind (hl_set _) fun _ => hl_pow _

/-- **A level's commitment past the close**: it sends nothing, and its cap is below `2 ^ 64` bytes. -/
theorem hl_commit (ms : MerkleScheme) (hms : ms = MerkleScheme.hm96Sha512) (lv : Level) (g : Array F128)
    (salt : ℕ → ByteArray) (hd : 64 * 2 ^ lv.d < 2 ^ 64) :
    HL P (Inv P) (fun cm t l => Inv P t l ∧ (flat cm.cap).size < 2 ^ 64)
      (liftT (StateT.lift (commitP (Hashes.ofSha512 Sha512.hash : Hashes PS) ms lv g salt)) : LP PS Commit) := by
  have hq := commitP_quiet Sha512.hash ms lv g salt
  refine hl_liftT (ht_liftPS _ (fun l => by rw [hq l]) inv_logQ fun l a L e => ?_)
  rw [hq l] at e
  simp only [Prod.mk.injEq] at e
  obtain ⟨hc1, hc2⟩ := commitP_cap Sha512.hash_size lv g salt (.inl hms) e.1
  rw [flat_size hc2]
  omega

theorem hl_challengeP (split : Bool) :
    HL P (Inv P) (fun _ => Inv P) (challengeP (asksCut coin N) split) := by
  unfold challengeP
  apply hl_bind hl_sampleKP; intro _
  apply hl_bind (hl_modify _); intro _
  dsimp only
  apply hl_ite
  · intro _
    apply hl_bind (hl_modify _); intro _
    apply hl_bind hl_get; intro _
    apply hl_bind (hl_sendMsg _); intro _
    exact hl_pure fun _ _ h => h
  · intro _
    apply hl_bind hl_get; intro _
    apply hl_bind (hl_sendMsg _); intro _
    exact hl_pure fun _ _ h => h

theorem hl_oodIntroP (nonce : ℕ → ℕ → UInt64) (g : Array F128) {dim : ℕ} (hd : dim < 2 ^ 64) :
    HL P (Inv P) (fun _ => Inv P) (oodIntroP (asksCut coin N) nonce g dim) := by
  unfold oodIntroP
  exact hl_bind (hl_sampleVecP hd) fun _ => hl_bind (hl_modify _) fun _ => hl_bind (hl_absorbF _) fun _ =>
    hl_bind (hl_sendMsg _) fun _ => hl_bind (hl_powP nonce 2) fun _ => hl_bind hl_sampleP fun _ => hl_modify _

theorem hl_queryP (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit) (hq : lv.q < 2 ^ 64) :
    HL P (Inv P) (fun _ => Inv P) (queryP (asksCut coin N) nonce ms lv cm) := by
  unfold queryP
  have hl : Nat.log2 (lv.q - 1) + 1 < 2 ^ 64 := by
    have := Nat.log2_le_self (lv.q - 1)
    omega
  exact hl_bind (hl_powP nonce 0) fun _ => hl_bind (hl_sampleVecP hq) fun _ => hl_bind (hl_powP nonce 1) fun _ =>
    hl_bind (hl_sampleVecP hl) fun _ => hl_pure fun _ _ h => h

/-! ## The opening -/

theorem pairs_size (ys : Array F256) :
    (ys.foldl (fun acc v => (acc.push v.c0).push v.c1) (#[] : Array F128)).size = 2 * ys.size := by
  rw [← Array.foldl_toList, ← Array.length_toList]
  suffices ∀ (zs : List F256) (acc : Array F128),
      (zs.foldl (fun acc v => (acc.push v.c0).push v.c1) acc).size = acc.size + 2 * zs.length by
    simpa using this ys.toList #[]
  intro zs
  induction zs with
  | nil => intro acc; simp
  | cons z zs ih =>
    intro acc
    rw [List.foldl_cons, ih]
    simp only [Array.size_push, List.length_cons]
    omega

set_option maxHeartbeats 4000000 in
/-- **Ligerito's opening from a log of `s` bodies, cut past level 0's close**: wherever it stops or returns, its log's
bodies from round `s + k_0 + 4` on are `Body`s, the first after level 0's closing label. -/
theorem ligeritoPZk_shape (coin : ℕ → ℕ → F128) (N : ℕ) (sch : Schedule) (T : F128) (cm0 : Commit)
    (t b : Array F128) (lay : Zk.PadLayout) (h : Array F128) (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (s : ℕ) (hk : 1 ≤ sch.k0) (hn0 : sch.n ≠ 0)
    (hn : sch.n + 1 < 2 ^ 64) (hq : ∀ i, (sch.levels.getD i default).q < 2 ^ 64) (htp : 2 * tPad < 2 ^ 64)
    (hd : ∀ i, 64 * 2 ^ (sch.levels.getD i default).d < 2 ^ 64) (hN : s + sch.k0 + 4 ≤ N) :
    HT (s + sch.k0 + 4) (Sz s) (fun _ _ L => LogQ (s + sch.k0 + 4) L)
      (ligeritoPZk (asksCut coin N) (Hashes.ofSha512 Sha512.hash) MerkleScheme.hm96Sha512 sch T cm0 t b lay h rep
        tPad extra lanePad salt nonce) := by
  have hsP : s ≤ s + sch.k0 + 4 := by omega
  unfold ligeritoPZk
  apply ht_bind (ht_liftTM _ (sz_logQ hsP) fun _ _ _ _ _ h => h); intro _
  apply ht_bind (ht_liftTM _ (sz_logQ hsP) fun _ _ _ _ _ h => h); intro _
  apply ht_bind (ht_liftTM _ (sz_logQ hsP) fun _ _ _ _ _ h => h); intro _
  apply ht_bind (ht_liftTM _ (sz_logQ hsP) fun _ _ _ _ _ h => h); intro _
  dsimp only
  apply ht_run_bind (Mid := fun _ => Inv (s + sch.k0 + 4))
  · -- `z`'s squeeze, `β_0`'s and the first message
    apply hl_bind (hl_liftT (ht_sqA' (coin := coin) (N := N) 2 hn0 (s := s) (by omega) (by omega))); intro _
    apply hl_bind (hl_modify _); intro _
    apply hl_bind (hl_lTMA (by omega) _); intro _
    apply hl_bind (hl_powPA (by omega) nonce 2); intro _
    apply hl_bind (hl_samplePA (coin := coin) (N := N) (s := s + 1) (by omega) (by omega)); intro _
    apply hl_bind (hl_modify _); intro _
    apply hl_bind hl_get; intro _
    apply hl_bind (hl_sendMsgA (by omega) _); intro _
    -- the lane rounds: round `s + 2 + i` lane round `i`'s challenge; the last one's pair, `ρ`'s and the close
    apply hl_forIn_bind (fun i _ t l => (i < sch.k0 ∧ Sz (s + 2 + i) t l) ∨ (i = sch.k0 ∧ Inv (s + sch.k0 + 4) t l))
      (fun _ l h => .inl ⟨by omega, by
        have : l.size = s + 1 + 1 := h
        show l.size = s + 2 + 0
        omega⟩)
    · intro i hi u
      refine hl_mono (Pre := Sz (s + 2 + i)) ?_ (fun _ _ h => by
        rcases h with ⟨-, h⟩ | ⟨h, -⟩
        · exact h
        · omega) fun _ _ _ h => h
      apply hl_bind (hl_sampleKPA (s := s + 2 + i) (by omega) (by omega)); intro _
      apply hl_bind (hl_modify _); intro _
      apply hl_ite
      · intro hlast
        have hl : i + 1 = sch.k0 := by simpa using hlast
        apply hl_bind hl_get; intro _
        apply hl_bind (hl_lTMA (by omega) _); intro _
        apply hl_bind (hl_lTMA (by omega) _); intro _
        apply hl_bind (hl_sampleKPA (s := s + 2 + i + 1) (by omega) (by omega)); intro _
        apply hl_bind (hl_sampleKPA' (s := s + 2 + i + 1 + 1) (by omega) (by omega)); intro _
        apply hl_ite
        · intro _
          exact hl_throw_bind _ fun _ l h => logQ_le (by have := h.1; omega)
        · intro _
          refine hl_bind (hl_mono hl_closeLabel (fun _ _ h => ⟨by have := h.1; omega, h.2⟩) fun _ _ _ h => h) ?_
          intro _
          apply hl_bind (hl_absorbK _); intro _
          apply hl_bind (hl_absorbSlice _ (by rw [pairs_size, Array.size_map, Array.size_range]; exact htp)); intro _
          apply hl_bind (hl_modify _); intro _
          apply hl_bind hl_get; intro _
          apply hl_bind (hl_sendMsg _); intro _
          exact hl_pure fun _ _ h => ⟨_, rfl, .inr ⟨hl, h⟩⟩
      · intro hlast
        have hl : i + 1 ≠ sch.k0 := by simpa using hlast
        apply hl_bind hl_get; intro _
        apply hl_bind (hl_sendMsgA (by omega) _); intro _
        exact hl_pure fun _ l h => ⟨_, rfl, .inl ⟨by omega, show l.size = s + 2 + (i + 1) by
          have : l.size = s + 2 + i + 1 := h
          omega⟩⟩
    · intro _
      refine hl_mono (Pre := Inv (s + sch.k0 + 4)) ?_ (fun _ _ h => by
        rcases h with ⟨h, -⟩ | ⟨-, h⟩
        · omega
        · exact h) fun _ _ _ h => h
      -- level 1's commitment, its out-of-domain samples, level 0's queries
      apply hl_bind hl_get; intro _
      apply hl_bind (hl_commit _ rfl _ _ _ (hd 1)); intro _
      refine hl_pre_and' fun hc => ?_
      apply hl_bind (hl_absorbBytes _ hc); intro _
      apply hl_bind (hl_oodIntroP nonce _ (dim := sch.n - sch.k0 + 1) (by omega)); intro _
      apply hl_bind (hl_oodIntroP nonce _ (dim := sch.n - sch.k0 + 1) (by omega)); intro _
      apply hl_bind (hl_queryP nonce _ _ cm0 (hq 0)); intro _
      apply hl_bind (hl_sendMsg _); intro _
      apply hl_bind (hl_powP nonce 2); intro _
      apply hl_bind hl_sampleP; intro _
      apply hl_bind (hl_modify _); intro _
      -- the recursive levels and the final one
      apply hl_forIn_bind (fun _ u t l => u.1 ≤ sch.n + 1 ∧ Inv (s + sch.k0 + 4) t l)
        (fun _ _ h => ⟨by dsimp only; omega, h⟩)
      · intro i _ u
        obtain ⟨sd, pc, cs, rps, fn⟩ := u
        refine hl_pre_and fun hsd => ?_
        replace hsd : sd ≤ sch.n + 1 := hsd
        apply hl_forIn_bind (fun _ _ => Inv (s + sch.k0 + 4)) (fun _ _ h => h)
        · intro j _ v
          apply hl_ite
          · intro _
            apply hl_bind hl_sampleKP; intro _
            apply hl_bind (hl_modify _); intro _
            apply hl_bind hl_get; intro _
            apply hl_bind (hl_modify _); intro _
            apply hl_bind hl_get; intro _
            apply hl_bind (hl_sendMsg _); intro _
            exact hl_pure fun _ _ h => ⟨_, rfl, h⟩
          · intro _
            apply hl_bind (hl_challengeP false); intro _
            exact hl_pure fun _ _ h => ⟨_, rfl, h⟩
        · intro _
          apply hl_ite
          · intro _
            apply hl_bind (hl_commit _ rfl _ _ _ (hd (i + 2))); intro _
            refine hl_pre_and' fun hc => ?_
            apply hl_bind (hl_absorbBytes _ hc); intro _
            apply hl_bind (hl_oodIntroP nonce _ (dim := sd - 4 + 1) (by omega)); intro _
            apply hl_bind (hl_oodIntroP nonce _ (dim := sd - 4 + 1) (by omega)); intro _
            apply hl_bind (hl_queryP nonce _ _ _ (hq (i + 1))); intro _
            apply hl_bind (hl_sendMsg _); intro _
            apply hl_bind (hl_powP nonce 2); intro _
            apply hl_bind hl_sampleP; intro _
            apply hl_bind (hl_modify _); intro _
            exact hl_pure fun _ _ h => ⟨_, rfl, by dsimp only; omega, h⟩
          · intro _
            refine hl_bind (hl_forIn_array _ _ (fun _ _ => hl_bind (hl_absorbF _) fun _ =>
              hl_pure fun _ _ h => ⟨⟨_, rfl⟩, h⟩) _) fun _ => ?_
            apply hl_bind (hl_queryP nonce _ _ _ (hq _)); intro _
            apply hl_bind (hl_powP nonce 2); intro _
            apply hl_bind hl_sampleP; intro _
            exact hl_pure fun _ _ h => ⟨_, rfl, hsd, h⟩
      · intro _
        exact hl_pure fun _ _ h => h.2
  · intro _
    exact ht_pure fun _ _ h => h.2.1

/-- **Ligerito's log from round `s + k_0 + 4` on**, cut past it: bodies, the first after level 0's closing label. -/
theorem ligeritoPZk_logQ (coin : ℕ → ℕ → F128) (N : ℕ) (sch : Schedule) (T : F128) (cm0 : Commit)
    (t b : Array F128) (lay : Zk.PadLayout) (h : Array F128) (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (tr : Transcript) (l : Array Claims.Msg)
    (hk : 1 ≤ sch.k0) (hn0 : sch.n ≠ 0) (hn : sch.n + 1 < 2 ^ 64) (hq : ∀ i, (sch.levels.getD i default).q < 2 ^ 64)
    (htp : 2 * tPad < 2 ^ 64) (hd : ∀ i, 64 * 2 ^ (sch.levels.getD i default).d < 2 ^ 64)
    (hN : l.size + sch.k0 + 4 ≤ N) :
    LogQ (l.size + sch.k0 + 4)
      (((ligeritoPZk (asksCut coin N) (Hashes.ofSha512 Sha512.hash) MerkleScheme.hm96Sha512 sch T cm0 t b lay h rep
        tPad extra lanePad salt nonce).run tr).go l).2 := by
  have H := ligeritoPZk_shape coin N sch T cm0 t b lay h rep tPad extra lanePad salt nonce l.size hk hn0 hn hq htp hd
    hN tr l rfl
  generalize ((ligeritoPZk (asksCut coin N) (Hashes.ofSha512 Sha512.hash) MerkleScheme.hm96Sha512 sch T cm0 t b lay h
    rep tPad extra lanePad salt nonce).run tr).go l = r at H
  obtain ⟨v, L⟩ := r
  cases v with
  | error e => exact H.1 e L rfl
  | ok p => exact H.2 p.1 p.2 L rfl

end FlockComplete.LigShape
