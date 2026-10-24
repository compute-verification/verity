import Proofs.Flock.Claims.Complete.Zk.RepRounds
import Proofs.Flock.Claims.ZK.CausalExe
import Proofs.Flock.Claims.ZK.FrameRounds

/-!
# A cut rep's rounds are its log's bodies (`repCut_rounds`)

`repCut` answered by `asksCut coin N` records a round at each squeeze of `n > 0` coins and logs the squeeze's framed bytes
as its body. Here, by a walk of the rep under a simulation that keeps an invariant of the transcript and the log
(`logS`, `Inv`), each recorded round `j` from a log of `L` bodies is (`RoundAt`):

* at most `B` coins, all of them `coin (L + j)`'s, with no digest;
* its bytes the log's body `L + j`, which end in the squeeze's frame, after pending bytes that start with the rep's domain
  frame at round `0` and with no domain frame after (`Lead`).

The squeezes' sizes are those of `RepRounds.lean`'s walk (`repPZk_grows`): the zerocheck's `6` and `m - 13`, the ring
switch's `7` and claims, Ligerito's `n` and its levels' (the recursive levels' out-of-domain sizes carried by the loop,
`loopR_logs`), and the inner proof's constraints and `Zk.Q`.
-/

namespace FlockComplete.RepLog

open Flock Transcript FlockComplete With
open FlockComplete.JsonRT (go_pure go_bind_ok monadLift_go loopR forIn_loopR)
open FlockSoundness.Refine (Frame label_enc absorbF_enc absorbSlice_enc absorbBytes_enc framed_enc init_enc header_data)

/-! ## The invariant -/

/-- **A round's pending bytes as they start**: the stream's domain frame first at round `0`, and no domain frame (op `1`)
first after it. -/
def Lead (dom : ByteArray) (j : ℕ) (b : List UInt8) : Prop :=
  if j = 0 then ∃ r, b = (Frame.dom dom).enc ++ r else b.head? ≠ some 1

theorem enc_head? (f : Frame) : f.enc.head? = some f.op := by
  simp [Frame.enc, header_data]

theorem lead_append {dom : ByteArray} {j : ℕ} {b : List UInt8} (h : Lead dom j b) (f : Frame) (hf : f.op ≠ 1) :
    Lead dom j (b ++ f.enc) := by
  unfold Lead at h ⊢
  by_cases hj : j = 0
  · rw [ite_eq_left hj] at h ⊢
    obtain ⟨r, rfl⟩ := h
    exact ⟨r ++ f.enc, by rw [List.append_assoc]⟩
  · rw [ite_eq_right hj] at h ⊢
    cases b with
    | nil =>
      rw [List.nil_append, enc_head?]
      exact fun e => hf (Option.some.inj e)
    | cons a b => simpa using h

theorem op_label (b : ByteArray) : (Frame.label b).op ≠ 1 := by show (2 : UInt8) ≠ 1; decide
theorem op_one (v : F128) : (Frame.one v).op ≠ 1 := by show (3 : UInt8) ≠ 1; decide
theorem op_vec (vs : List F128) : (Frame.vec vs).op ≠ 1 := by show (3 : UInt8) ≠ 1; decide
theorem op_bytes (b : ByteArray) : (Frame.bytes b).op ≠ 1 := by show (5 : UInt8) ≠ 1; decide
theorem op_sq (k : UInt8) (n : ℕ) : (Frame.sq k n).op ≠ 1 := by show (4 : UInt8) ≠ 1; decide

/-- **Round `j` of a rep whose log started at `L` bodies**: at most `B` coins, `coin (L + j)`'s, no digest, and bytes the
log's body `L + j`, its pending bytes (`Lead`) then the squeeze's frame. -/
structure RoundAt (coin : ℕ → ℕ → F128) (B : ℕ) (dom : ByteArray) (L : ℕ) (l : Array Claims.Msg) (j : ℕ) (r : Round) :
    Prop where
  pos : 0 < r.n
  le : r.n ≤ B
  sha : r.sha = ByteArray.empty
  coins : r.coins = (Array.range r.n).map (coin (L + j))
  sent : ∃ c, r.bytes = some c ∧ l[L + j]? = some (Claims.toMsg c) ∧
    ∃ p kind, c.data.toList = p ++ (Frame.sq kind r.n).enc ∧ Lead dom j p

/-- **The invariant**: one body logged per round from `L`, the pending bytes as a round's start, and each round as above. -/
structure Inv (coin : ℕ → ℕ → F128) (B : ℕ) (dom : ByteArray) (L : ℕ) (t : Transcript) (l : Array Claims.Msg) :
    Prop where
  size : l.size = L + t.rounds.size
  lead : Lead dom t.rounds.size t.pending.data.toList
  rounds : ∀ j r, t.rounds[j]? = some r → RoundAt coin B dom L l j r

variable {coin : ℕ → ℕ → F128} {B N : ℕ} {dom : ByteArray} {L : ℕ}

theorem RoundAt.push {l : Array Claims.Msg} {j : ℕ} {r : Round} (h : RoundAt coin B dom L l j r)
    (hj : L + j < l.size) (m : Claims.Msg) : RoundAt coin B dom L (l.push m) j r := by
  obtain ⟨h1, h2, h3, h4, c, hc, hl, hp⟩ := h
  refine ⟨h1, h2, h3, h4, c, hc, ?_, hp⟩
  rw [Array.getElem?_push, ite_eq_right (by omega)]
  exact hl

theorem Inv.absorb {t t' : Transcript} {l : Array Claims.Msg} (h : Inv coin B dom L t l) (hr : t'.rounds = t.rounds)
    (f : Frame) (hf : f.op ≠ 1) (hp : t'.pending.data.toList = t.pending.data.toList ++ f.enc) :
    Inv coin B dom L t' l :=
  ⟨by rw [hr]; exact h.size, by rw [hr, hp]; exact lead_append h.lead f hf,
    fun j r hj => h.rounds j r (by rw [← hr]; exact hj)⟩

theorem Inv.squeeze {t : Transcript} {l : Array Claims.Msg} (h : Inv coin B dom L t l) {n : ℕ} (h0 : 0 < n)
    (hn : n ≤ B) (kind : UInt8) :
    Inv coin B dom L (Replay.afterSq t kind n (coin l.size)) (l.push (Claims.toMsg (framed t kind n))) := by
  obtain ⟨hs, hl, hr⟩ := h
  refine ⟨?_, ?_, ?_⟩
  · simp only [Replay.afterSq, Array.size_push, hs]
    omega
  · show Lead dom (t.rounds.push _).size ByteArray.empty.data.toList
    unfold Lead
    rw [ite_eq_right (by simp)]
    show ([] : List UInt8).head? ≠ some 1
    simp
  · intro j r hj
    simp only [Replay.afterSq] at hj
    rw [Array.getElem?_push] at hj
    split at hj
    · rename_i hje
      cases hj
      subst hje
      refine ⟨h0, hn, rfl, ?_, framed t kind n, rfl, ?_, t.pending.data.toList, kind, framed_enc t kind n, hl⟩
      · simp only [Replay.sent]
        rw [hs]
      · rw [Array.getElem?_push, ite_eq_left hs.symm]
    · have hlt := (Array.getElem?_eq_some_iff.1 hj).1
      exact (hr j r hj).push (by omega) _

/-! ## The simulation -/

/-- A transcript computation of the pure session that keeps `I` of its transcript and log. -/
def Logs (I : Transcript → Array Claims.Msg → Prop) {α : Type} (y : TMW PS α) : Prop :=
  ∀ (t : Transcript) (l : Array Claims.Msg) (a : α) (t' : Transcript) (l' : Array Claims.Msg),
    (y.run t).go l = (.ok (a, t'), l') → I t l → I t' l'

/-- **The simulation of `Logs`**: a transcript computation of the pure session is related to itself when it keeps `I`. -/
def logS (I : Transcript → Array Claims.Msg → Prop) : With.Sim (TMW PS) (TMW PS) where
  R _ y := Logs I y
  pure a t l a' t' l' h hI := by
    simp only [StateT.run_pure, go_pure, Prod.mk.injEq, Except.ok.injEq] at h
    obtain ⟨⟨-, rfl⟩, rfl⟩ := h
    exact hI
  bind {α β x y f g} hxy hfg t l b t'' l'' h hI := by
    rw [StateT.run_bind] at h
    obtain ⟨⟨a, t'⟩, l', h1, h2⟩ := go_bind_ok h
    exact hfg a t' l' b t'' l'' h2 (hxy t l a t' l' h1 hI)
  throw e t l a t' l' h := by
    change ((throw e : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at h
    obtain ⟨_, _, h1, -⟩ := go_bind_ok h
    cases h1
  lift v t l a t' l' h hI := by
    change ((monadLift v : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at h
    obtain ⟨b, l₁, h1, h2⟩ := go_bind_ok h
    cases v with
    | error _ => cases h1
    | ok c =>
      cases h1
      cases h2
      exact hI

/-! ## The steps -/

/-- A squeeze of at most `B` coins keeps the invariant: of `0`, its frame stays pending; else it records the round. -/
theorem squeezeP_logs (kind : UInt8) {n : ℕ} (hn : n ≤ B) :
    (logS (Inv coin B dom L)).R (squeezeP (asksCut coin N) kind n) (squeezeP (asksCut coin N) kind n) := by
  intro t l a t' l' h hI
  by_cases h0 : n = 0
  · subst h0
    have e : ((squeezeP (asksCut coin N) kind 0).run t).go l =
        (.ok (#[], { t with pending := framed t kind 0 }), l) := rfl
    rw [e] at h
    cases h
    exact hI.absorb rfl (.sq kind 0) (op_sq kind 0) (framed_enc t kind 0)
  · obtain ⟨-, -, rfl, rfl⟩ := squeezeP_cut_inv coin kind h0 h
    exact hI.squeeze (Nat.pos_of_ne_zero h0) hn kind

/-- A verifier action that appends a frame other than a domain frame and keeps the rounds keeps the invariant. -/
theorem liftTM_logs {α : Type} (x : TM α)
    (hx : ∀ t a t', x t = .ok (a, t') → t'.rounds = t.rounds ∧
      ∃ f : Frame, f.op ≠ 1 ∧ t'.pending.data.toList = t.pending.data.toList ++ f.enc) :
    (logS (Inv coin B dom L)).R (liftTM x : TMW PS α) (liftTM x) := by
  intro t l a t' l' h hI
  change (monadLift (x t) : PS _).go l = _ at h
  rcases e : x t with _ | ⟨a', t''⟩
  · rw [e] at h; cases h
  · rw [e] at h
    cases h
    obtain ⟨hr, f, hf, hp⟩ := hx t _ _ e
    exact hI.absorb hr f hf hp

theorem label_logs (s : String) :
    (logS (Inv coin B dom L)).R (liftTM (label s) : TMW PS Unit) (liftTM (label s)) :=
  liftTM_logs _ fun t _ _ h => by cases h; exact ⟨rfl, .label s.toUTF8, op_label _, label_enc _ _⟩

theorem absorbF_logs (v : F128) :
    (logS (Inv coin B dom L)).R (liftTM (absorbF v) : TMW PS Unit) (liftTM (absorbF v)) :=
  liftTM_logs _ fun t _ _ h => by cases h; exact ⟨rfl, .one v, op_one _, absorbF_enc _ _⟩

theorem absorbSlice_logs (vs : Array F128) :
    (logS (Inv coin B dom L)).R (liftTM (absorbSlice vs) : TMW PS Unit) (liftTM (absorbSlice vs)) :=
  liftTM_logs _ fun t _ _ h => by cases h; exact ⟨rfl, .vec vs.toList, op_vec _, absorbSlice_enc _ _⟩

theorem absorbK_logs (v : F256) :
    (logS (Inv coin B dom L)).R (liftTM (absorbK v) : TMW PS Unit) (liftTM (absorbK v)) :=
  liftTM_logs _ fun t _ _ h => by
    cases h; exact ⟨rfl, .vec (#[v.c0, v.c1] : Array F128).toList, op_vec _, absorbSlice_enc _ _⟩

theorem absorbBytes_logs (b : ByteArray) :
    (logS (Inv coin B dom L)).R (liftTM (absorbBytes b) : TMW PS Unit) (liftTM (absorbBytes b)) :=
  liftTM_logs _ fun t _ _ h => by cases h; exact ⟨rfl, .bytes b, op_bytes _, absorbBytes_enc _ _⟩

theorem pow_logs (v : UInt64) : (logS (Inv coin B dom L)).R (liftTM (pow v) : TMW PS Unit) (liftTM (pow v)) :=
  liftTM_logs _ fun t _ _ h => by cases h; exact ⟨rfl, .bytes (le64 v), op_bytes _, absorbBytes_enc _ _⟩

theorem get_logs : (logS (Inv coin B dom L)).R (get : TMW PS Transcript) get := by
  intro t l a t' l' h hI
  simp only [StateT.run_get, go_pure, Prod.mk.injEq, Except.ok.injEq] at h
  obtain ⟨⟨-, rfl⟩, rfl⟩ := h
  exact hI

/-- A computation of the pure session that sends nothing, lifted to the transcript, keeps the invariant. -/
theorem liftPS_logs {α : Type} (y : PS α) (hq : ∀ l, (y.go l).2 = l) :
    (logS (Inv coin B dom L)).R (StateT.lift y : TMW PS α) (StateT.lift y) := by
  intro t l a t' l' e hI
  change (y >>= fun a => Pure.pure (a, t)).go l = _ at e
  obtain ⟨b, l₁, e1, e2⟩ := go_bind_ok e
  have hl := hq l
  rw [e1] at hl
  dsimp only at hl
  subst hl
  cases e2
  exact hI

theorem liftT_logs {α : Type} {x y : TMW PS α} (h : (logS (Inv coin B dom L)).R x y) :
    ((logS (Inv coin B dom L)).st LigState).R (liftT x) (liftT y) :=
  With.Sim.lift_st _ h

theorem commitP_logs (ms : MerkleScheme) (lv : Level) (g : Array F128) (salt : ℕ → ByteArray) :
    (logS (Inv coin B dom L)).R
      (StateT.lift (commitP (Hashes.ofSha512 Sha512.hash : Hashes PS) ms lv g salt) : TMW PS Commit)
      (StateT.lift (commitP (Hashes.ofSha512 Sha512.hash) ms lv g salt)) :=
  liftPS_logs _ fun l => by rw [commitP_quiet Sha512.hash ms lv g salt l]

theorem sampleP_logs (h1 : 1 ≤ B) :
    (logS (Inv coin B dom L)).R (sampleP (asksCut coin N)) (sampleP (asksCut coin N)) := by
  unfold sampleP
  sim_walk [squeezeP_logs _ h1]

theorem sampleKP_logs (h2 : 2 ≤ B) :
    (logS (Inv coin B dom L)).R (sampleKP (asksCut coin N)) (sampleKP (asksCut coin N)) := by
  unfold sampleKP
  sim_walk [squeezeP_logs _ h2]

theorem sendMsg_logs (m : F256 × F256) :
    ((logS (Inv coin B dom L)).st LigState).R (sendMsg m : LP PS Unit) (sendMsg m) := by
  unfold sendMsg
  sim_walk [liftT_logs, absorbK_logs]

theorem powP_logs (nonce : ℕ → ℕ → UInt64) (kind : ℕ) :
    ((logS (Inv coin B dom L)).st LigState).R (powP nonce kind : LP PS Unit) (powP nonce kind) := by
  unfold powP
  sim_walk [liftT_logs, pow_logs]

theorem challengeP_logs (split : Bool) (h2 : 2 ≤ B) :
    ((logS (Inv coin B dom L)).st LigState).R (challengeP (asksCut coin N) split)
      (challengeP (asksCut coin N) split) := by
  unfold challengeP
  sim_walk [liftT_logs, sampleKP_logs h2, sendMsg_logs]

theorem oodIntroP_logs (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ) (hd : dim ≤ B) (h1 : 1 ≤ B) :
    ((logS (Inv coin B dom L)).st LigState).R (oodIntroP (asksCut coin N) nonce g dim)
      (oodIntroP (asksCut coin N) nonce g dim) := by
  unfold oodIntroP sampleVecP
  sim_walk [liftT_logs, squeezeP_logs _ hd, sampleP_logs h1, sendMsg_logs, powP_logs, absorbF_logs]

theorem queryP_logs (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit) (hq : lv.q ≤ B)
    (hl : Nat.log2 (lv.q - 1) + 1 ≤ B) :
    ((logS (Inv coin B dom L)).st LigState).R (queryP (asksCut coin N) nonce ms lv cm)
      (queryP (asksCut coin N) nonce ms lv cm) := by
  unfold queryP sampleVecP
  sim_walk [liftT_logs, squeezeP_logs _ hq, squeezeP_logs _ hl, powP_logs]

theorem queryP_logs_lv (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (sch : Schedule)
    (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ B) (j : ℕ) (cm : Commit) :
    ((logS (Inv coin B dom L)).st LigState).R (queryP (asksCut coin N) nonce ms (sch.levels.getD j default) cm)
      (queryP (asksCut coin N) nonce ms (sch.levels.getD j default) cm) :=
  queryP_logs nonce ms _ cm (by have := hq j; omega)
    (by have := hq j; have := Nat.log2_le_self ((sch.levels.getD j default).q - 1); omega)

/-! ## The loops, with an invariant of their state -/

/-- `Logs` of a transcript computation that returns a value with `Q`. -/
def Post (I : Transcript → Array Claims.Msg → Prop) {α : Type} (y : TMW PS α) (Q : α → Prop) : Prop :=
  ∀ (t : Transcript) (l : Array Claims.Msg) (a : α) (t' : Transcript) (l' : Array Claims.Msg),
    (y.run t).go l = (.ok (a, t'), l') → I t l → I t' l' ∧ Q a

section Post

variable {I : Transcript → Array Claims.Msg → Prop} {α β : Type}

theorem Post.of_logs {y : TMW PS α} (h : (logS I).R y y) : Post I y fun _ => True :=
  fun t l a t' l' e hI => ⟨h t l a t' l' e hI, trivial⟩

theorem Post.to_logs {y : TMW PS α} {Q : α → Prop} (h : Post I y Q) : (logS I).R y y :=
  fun t l a t' l' e hI => (h t l a t' l' e hI).1

theorem Post.bind {x : TMW PS α} {f : α → TMW PS β} {Q : α → Prop} {P : β → Prop} (hx : Post I x Q)
    (hf : ∀ a, Q a → Post I (f a) P) : Post I (x >>= f) P := by
  intro t l b t'' l'' h hI
  rw [StateT.run_bind] at h
  obtain ⟨⟨a, t'⟩, l', h1, h2⟩ := go_bind_ok h
  obtain ⟨i1, q1⟩ := hx t l a t' l' h1 hI
  exact hf a q1 t' l' b t'' l'' h2 i1

theorem Post.bind_logs {x : TMW PS α} {f : α → TMW PS β} {P : β → Prop} (hx : (logS I).R x x)
    (hf : ∀ a, Post I (f a) P) : Post I (x >>= f) P :=
  Post.bind (Post.of_logs hx) fun a _ => hf a

theorem Post.pure {a : α} {P : α → Prop} (h : P a) : Post I (Pure.pure a : TMW PS α) P := by
  intro t l b t' l' e hI
  simp only [StateT.run_pure, go_pure, Prod.mk.injEq, Except.ok.injEq] at e
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := e
  exact ⟨hI, h⟩

theorem Post.liftV (v : V α) : Post I (StateT.lift (monadLift v : PS α) : TMW PS α) fun a => v = .ok a := by
  intro t l a t' l' e hI
  change ((monadLift v : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at e
  obtain ⟨b, l₁, e1, e2⟩ := go_bind_ok e
  rw [monadLift_go] at e1
  simp only [Prod.mk.injEq] at e1
  obtain ⟨hv, rfl⟩ := e1
  cases e2
  exact ⟨hI, by cases v <;> simp_all⟩

/-- `Post` of a Ligerito step, from every Ligerito state. -/
def PostL (I : Transcript → Array Claims.Msg → Prop) {α : Type} (y : LP PS α) (Q : α → Prop) : Prop :=
  ∀ s, Post I (y.run s) fun p => Q p.1

theorem PostL.bind {x : LP PS α} {f : α → LP PS β} {Q : α → Prop} {P : β → Prop} (hx : PostL I x Q)
    (hf : ∀ a, Q a → PostL I (f a) P) : PostL I (x >>= f) P := by
  intro s
  rw [StateT.run_bind]
  exact Post.bind (hx s) fun p hp => hf p.1 hp p.2

theorem PostL.of_logs {y : LP PS α} (h : ((logS I).st LigState).R y y) : PostL I y fun _ => True :=
  fun s => Post.of_logs (h s)

theorem PostL.bind_logs {x : LP PS α} {f : α → LP PS β} {P : β → Prop} (hx : ((logS I).st LigState).R x x)
    (hf : ∀ a, PostL I (f a) P) : PostL I (x >>= f) P :=
  PostL.bind (PostL.of_logs hx) fun a _ => hf a

theorem PostL.pure {a : α} {P : α → Prop} (h : P a) : PostL I (Pure.pure a : LP PS α) P := by
  intro s t l b t' l' e hI
  simp only [StateT.run_pure, go_pure, Prod.mk.injEq, Except.ok.injEq] at e
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := e
  exact ⟨hI, h⟩

theorem PostL.forIn_list (f : ℕ → β → LP PS (ForInStep β)) (J : β → Prop)
    (hf : ∀ i b, J b → PostL I (f i b) fun s => J s.value) :
    ∀ (xs : List ℕ) (b : β), J b → PostL I (forIn xs b f) J := by
  intro xs
  induction xs with
  | nil => intro b hb; exact PostL.pure hb
  | cons x xs ih =>
    intro b hb
    rw [List.forIn_cons]
    refine PostL.bind (hf x b hb) fun r hr => ?_
    cases r with
    | done b' => exact PostL.pure hr
    | yield b' => exact ih b' hr

/-- **A loop that keeps an invariant of its state keeps `I`.** -/
theorem loopR_logs (J : β → Prop) (r : Std.Legacy.Range) (b : β) (f : ℕ → β → LP PS (ForInStep β)) (hb : J b)
    (hf : ∀ i b, J b → PostL I (f i b) fun s => J s.value) :
    ((logS I).st LigState).R (loopR r b f) (loopR r b f) := by
  intro s t l a t' l' e hI
  unfold loopR at e
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at e
  exact (PostL.forIn_list f J hf _ b hb s t l a t' l' e hI).1

end Post

theorem oodIntroP_logs_loop (nonce : ℕ → ℕ → UInt64) (D : ℕ) (hD : D ≤ B) (h1 : 1 ≤ B) (g : Array F128)
    (b : ℕ × Commit × Array (Array ByteArray) × Array LevelProof × FinalProof) (hb : b.1 ≤ D) :
    ((logS (Inv coin B dom L)).st LigState).R (oodIntroP (asksCut coin N) nonce g (b.1 - 4 + 1))
      (oodIntroP (asksCut coin N) nonce g (b.1 - 4 + 1)) :=
  oodIntroP_logs nonce g _ (by omega) h1

/-! ## The rep's phases -/

theorem zerocheckCut_logs (m : ℕ) (digest cap root : ByteArray) (lay : Zk.PadLayout) (mk : ℕ → F128) (h6 : 6 ≤ B)
    (hm : m - 13 ≤ B) :
    (logS (Inv coin B dom L)).R (zerocheckCut (asksCut coin N) m digest cap root lay mk)
      (zerocheckCut (asksCut coin N) m digest cap root lay mk) := by
  unfold zerocheckCut sampleVecP
  sim_walk [squeezeP_logs _ h6, squeezeP_logs _ hm, sampleP_logs (by omega : 1 ≤ B), label_logs, absorbBytes_logs,
    absorbSlice_logs, absorbF_logs]

theorem lincheckCut_logs (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (mk : ℕ → F128) (zc : ZcOut) (h1 : 1 ≤ B) :
    (logS (Inv coin B dom L)).R (lincheckCut (asksCut coin N) c m lay mk zc)
      (lincheckCut (asksCut coin N) c m lay mk zc) := by
  unfold lincheckCut
  sim_walk [sampleP_logs h1, label_logs, absorbSlice_logs, absorbF_logs]

theorem ringSwitchCut_logs (sv : ℕ → Array F128) (claims : Array Claim) (h7 : 7 ≤ B) (hc : claims.size ≤ B) :
    (logS (Inv coin B dom L)).R (ringSwitchCut (asksCut coin N) sv claims)
      (ringSwitchCut (asksCut coin N) sv claims) := by
  unfold ringSwitchCut sampleVecP
  sim_walk [squeezeP_logs _ h7, squeezeP_logs _ hc, label_logs, absorbSlice_logs]

theorem rankCheckP_logs (m kLog : ℕ) (prior : Array (List F128)) (words : List ℕ) :
    (logS (Inv coin B dom L)).R (rankCheckP m kLog prior words : TMW PS Unit) (rankCheckP m kLog prior words) := by
  unfold rankCheckP
  sim_walk [get_logs]

theorem piopCut_logs (st : Setup) (lay : Zk.PadLayout) (src : RepSrc) (cap root : ByteArray) (extras : Array Claim)
    (prior : Array (List F128)) (words : List ℕ) (h7 : 7 ≤ B) (hm : st.m - 13 ≤ B) (hx : extras.size + 2 ≤ B) :
    (logS (Inv coin B dom L)).R (piopCut (asksCut coin N) st lay src cap root extras prior words)
      (piopCut (asksCut coin N) st lay src cap root extras prior words) := by
  unfold piopCut
  refine (logS _).bind (zerocheckCut_logs _ _ _ _ _ _ (by omega) hm) fun r => ?_
  obtain ⟨zp, zc⟩ := r
  refine (logS _).bind (lincheckCut_logs _ _ _ _ _ (by omega)) fun r => ?_
  obtain ⟨lp, lc⟩ := r
  refine (logS _).bind (rankCheckP_logs _ _ _ _) fun _ => ?_
  refine (logS _).bind (ringSwitchCut_logs _ _ h7 ?_) fun r => ?_
  · simp only [Array.size_append, abcClaims, List.size_toArray, List.length_cons, List.length_nil]
    omega
  obtain ⟨ss, T, sws⟩ := r
  exact (logS _).pure _

set_option maxHeartbeats 1000000 in
/-- **The cut's Ligerito opening keeps the invariant**, given `2`, `n + 1` and every level's `q + 1` at most `B`. -/
theorem ligeritoCut_logs (ms : MerkleScheme) (sch : Schedule) (T : F128) (cm0 : Commit) (b : Array F128)
    (lay : Zk.PadLayout) (src : RepSrc) (h2 : 2 ≤ B) (hn : sch.n + 1 ≤ B)
    (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ B) :
    (logS (Inv coin B dom L)).R
      (ligeritoCut (asksCut coin N) (Hashes.ofSha512 Sha512.hash) ms sch T cm0 b lay src)
      (ligeritoCut (asksCut coin N) (Hashes.ofSha512 Sha512.hash) ms sch T cm0 b lay src) := by
  have h1 : 1 ≤ B := by omega
  unfold ligeritoCut sampleVecP
  simp (config := { zeta := false }) only [forIn_loopR]
  sim_walk [liftT_logs, squeezeP_logs _ (show sch.n ≤ B by omega), sampleP_logs h1, sampleKP_logs h2, sendMsg_logs,
    powP_logs, challengeP_logs _ h2, oodIntroP_logs _ _ (sch.n - sch.k0 + 1) (by omega) h1,
    queryP_logs_lv _ ms sch hq, commitP_logs, label_logs, absorbF_logs, absorbBytes_logs, absorbK_logs,
    absorbSlice_logs,
    loopR_logs (fun s : ℕ × Commit × Array (Array ByteArray) × Array LevelProof × FinalProof => s.1 ≤ sch.n + 1)]
  · simp (config := { zetaDelta := true }) only []
    omega
  · intro i b hb
    repeat (first
      | extract_lets
      | (refine PostL.bind_logs ?_ fun _ => ?_
         sim_walk [liftT_logs, sampleP_logs h1, sampleKP_logs h2, sendMsg_logs, powP_logs, challengeP_logs _ h2,
           oodIntroP_logs_loop _ (sch.n + 1) hn h1, queryP_logs_lv _ ms sch hq, commitP_logs, absorbF_logs,
           absorbBytes_logs])
      | (refine PostL.pure ?_
         simp (config := { zetaDelta := true }) only [ForInStep.value]
         omega)
      | split)
  all_goals sim_walk

theorem squeezeP_logs_cons (lay : Zk.PadLayout) (cons : Array (String × Zk.Lin)) (hc : cons.size + 3 ≤ B)
    (e r s : F128) :
    (logS (Inv coin B dom L)).R (squeezeP (asksCut coin N) 2 (cons ++ Zk.triple lay e r s).size)
      (squeezeP (asksCut coin N) 2 (cons ++ Zk.triple lay e r s).size) :=
  squeezeP_logs 2 (by simp [Zk.triple]; omega)

theorem innerCut_logs (lay : Zk.PadLayout) (ic : InnerCommit) (finalC : F128) (src : RepSrc)
    (cons : Array (String × Zk.Lin)) (hc : cons.size + 3 ≤ B) (hQ : Zk.Q ≤ B) :
    (logS (Inv coin B dom L)).R (innerCut (asksCut coin N) lay ic finalC src cons)
      (innerCut (asksCut coin N) lay ic finalC src cons) := by
  have h1 : 1 ≤ B := le_trans (by decide) hQ
  unfold innerCut sampleVecP
  sim_walk [sampleP_logs h1, squeezeP_logs _ hQ, squeezeP_logs_cons lay cons hc, label_logs, absorbF_logs,
    absorbSlice_logs, absorbBytes_logs]

/-- **A cut rep keeps the invariant**, given `Zk.Q`, `m - 13`, its claims, `n + 1` and every level's `q + 1` at most
`B`. -/
theorem repCut_logs (st : Setup) (sch : Schedule) (rep : ℕ) (dm : ByteArray) (cm0 : Commit) (extras : Array Claim)
    (ic : InnerCommit) (src : RepSrc) (prior : Array (List F128)) (words : List ℕ) (hQ : Zk.Q ≤ B)
    (hm : st.m - 13 ≤ B) (hx : extras.size + 2 ≤ B) (hn : sch.n + 1 ≤ B)
    (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ B) :
    (logS (Inv coin B dom L)).R
      (repCut (asksCut coin N) (Hashes.ofSha512 Sha512.hash) st sch rep dm cm0 extras ic src prior words)
      (repCut (asksCut coin N) (Hashes.ofSha512 Sha512.hash) st sch rep dm cm0 extras ic src prior words) := by
  have hB : 168 ≤ B := hQ
  unfold repCut
  refine Post.to_logs (Q := fun _ => True) ?_
  refine Post.bind_logs (piopCut_logs _ _ _ _ _ _ _ _ (by omega) hm hx) fun r => ?_
  obtain ⟨zp, lp, ss, T, sws⟩ := r
  refine Post.bind_logs (ligeritoCut_logs _ sch _ _ _ _ _ (by omega) hn hq) fun r => ?_
  obtain ⟨lg, zl, rec⟩ := r
  refine Post.bind_logs get_logs fun tr' => ?_
  refine Post.bind (Post.liftV _) fun r hrp => ?_
  obtain ⟨rp, t''⟩ := r
  refine Post.bind (Post.liftV _) fun lgc hlgc => ?_
  have h3 := FlockSoundness.Discharge.ZkExec.replay_cons_size _ _ _ _ hrp
  have h2 := FlockSoundness.Discharge.ZkExec.ligeritoConstraints_size hlgc
  refine Post.bind_logs (innerCut_logs _ _ _ _ _ ?_ hQ) fun _ => Post.pure trivial
  simp only [Array.size_append, List.size_toArray, List.length_cons, List.length_nil, h3, h2]
  omega

/-- **A cut rep's rounds are its log's bodies**: where `repCut` answered by `asksCut coin N` returns from the rep's
domain's fresh transcript and a log of `l.size` bodies, it logged one body per round, and round `j` is `RoundAt` the
log's body `l.size + j`. -/
theorem repCut_rounds (st : Setup) (sch : Schedule) (rep : ℕ) (cm0 : Commit) (extras : Array Claim)
    (ic : InnerCommit) (src : RepSrc) (prior : Array (List F128)) (words : List ℕ) (hQ : Zk.Q ≤ B)
    (hm : st.m - 13 ≤ B) (hx : extras.size + 2 ≤ B) (hn : sch.n + 1 ≤ B)
    (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ B) {l l' : Array Claims.Msg} {p : Zk.ZkProof} {tr : Transcript}
    (h : ((repCut (asksCut coin N) (Hashes.ofSha512 Sha512.hash) st sch rep dom cm0 extras ic src prior words).run
      (Transcript.init #[] dom)).go l = (.ok (p, tr), l')) :
    l'.size = l.size + tr.rounds.size ∧ ∀ j r, tr.rounds[j]? = some r → RoundAt coin B dom l.size l' j r := by
  have h0 : Inv coin B dom l.size (Transcript.init #[] dom) l := by
    refine ⟨by simp [Transcript.init], ?_, fun j r hj => by simp [Transcript.init] at hj⟩
    show Lead dom 0 _
    unfold Lead
    rw [ite_eq_left rfl, init_enc]
    exact ⟨[], by rw [List.append_nil]⟩
  have hI := repCut_logs (coin := coin) (N := N) st sch rep dom cm0 extras ic src prior words hQ hm hx hn hq
    _ _ _ _ _ h h0
  exact ⟨hI.size, hI.rounds⟩

end FlockComplete.RepLog
