import Proofs.Flock.Claims.Complete.Zk.ProverOk
import Proofs.Flock.Claims.Complete.Replay.RepRounds
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysRep
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsReplay

/-!
# Each masked rep records a round, each of at most `65536` coins (`RepRoundsZk`)

`repP_repOk` at the masked prover, by a walk of the rep:

* **Keeping the rounds small** (`Keeps`, its simulation `keepS`): a transcript computation keeps every round at most `B`
  coins and drops none. Each masked step keeps them, given its squeezes' sizes: the zerocheck (`6` and `m - 13`), the
  lincheck, the ring switch (`7` and its claims), the masked Ligerito (`ligeritoPZk_keeps`; its recursive levels'
  out-of-domain sizes are carried by the loop, which is walked with an invariant, `loopR_keeps`) and the inner proof
  (`innerP_keeps`: `Zk.Q` and its constraints and three).
* **Growing** (`Grows B k`): keeping the rounds small, adding at least `k` of them, with a property of the result. The
  masked zerocheck adds its first round (`zerocheckPZk_grows`), so a rep adds one (`repPZk_grows`), its constraints
  being the replay's three, the ring switch's two and level 0's lane phase's two (`replay_cons_size`,
  `ligeritoConstraints_size`).
* The `--zk` schedules squeeze fewer than `65536` coins (`zkSchedule_small`), and a masked case's table has at most
  1,024 regions (`CaseZk.regions`), so at most 2,050 claims: `repRoundsZk_goal`.
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy With

/-- Every round of `t` is of at most `B` coins. -/
def CoinsLe (B : ℕ) (t : Transcript) : Prop := ∀ x ∈ t.rounds, x.n ≤ B

/-- A transcript computation of the pure session that keeps every round at most `B` coins and drops none. -/
def Keeps (B : ℕ) {α : Type} (y : TMW PS α) : Prop :=
  ∀ (t : Transcript) (l : Array Claims.Msg) (a : α) (t' : Transcript) (l' : Array Claims.Msg),
    (y.run t).go l = (.ok (a, t'), l') → (CoinsLe B t → CoinsLe B t') ∧ t.rounds.size ≤ t'.rounds.size

theorem Keeps.same {B : ℕ} {t t' : Transcript} (h : t'.rounds = t.rounds) :
    (CoinsLe B t → CoinsLe B t') ∧ t.rounds.size ≤ t'.rounds.size := by
  unfold CoinsLe
  rw [h]
  exact ⟨id, le_rfl⟩

/-- **The simulation of `Keeps`**: a transcript computation of the pure session is related to itself when it keeps its
rounds at most `B` coins. -/
def keepS (B : ℕ) : With.Sim (TMW PS) (TMW PS) where
  R _ y := Keeps B y
  pure a t l a' t' l' h := by
    simp only [StateT.run_pure, go_pure, Prod.mk.injEq, Except.ok.injEq] at h
    obtain ⟨⟨-, rfl⟩, rfl⟩ := h
    exact Keeps.same rfl
  bind {α β x y f g} hxy hfg t l b t'' l'' h := by
    rw [StateT.run_bind] at h
    obtain ⟨⟨a, t'⟩, l', h1, h2⟩ := go_bind_ok h
    obtain ⟨k1, s1⟩ := hxy t l a t' l' h1
    obtain ⟨k2, s2⟩ := hfg a t' l' b t'' l'' h2
    exact ⟨k2 ∘ k1, s1.trans s2⟩
  throw e t l a t' l' h := by
    change ((throw e : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at h
    obtain ⟨_, _, h1, -⟩ := go_bind_ok h
    cases h1
  lift v t l a t' l' h := by
    change ((monadLift v : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at h
    obtain ⟨b, l₁, h1, h2⟩ := go_bind_ok h
    cases v with
    | error _ => cases h1
    | ok c =>
      cases h1
      cases h2
      exact Keeps.same rfl

/-- A squeeze of at most `B` coins keeps the rounds at most `B` coins. -/
theorem squeezeP_keeps (resp : ℕ → Claims.Msg) (kind : UInt8) (n : ℕ) {B : ℕ} (hn : n ≤ B) :
    (keepS B).R (squeezeP (asksS resp) kind n) (squeezeP (asksS resp) kind n) := by
  intro t l a t' l' h
  change (squeezeP (asksS resp) kind n t).go l = _ at h
  unfold squeezeP at h
  split at h
  · simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at h
    obtain ⟨⟨-, rfl⟩, rfl⟩ := h
    exact Keeps.same rfl
  · obtain ⟨coin, l₁, h1, h2⟩ := go_bind_ok h
    rw [round_go] at h1
    simp only [Prod.mk.injEq, Except.ok.injEq] at h1
    obtain ⟨rfl, rfl⟩ := h1
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at h2
    obtain ⟨⟨-, rfl⟩, rfl⟩ := h2
    refine ⟨fun hs x hx => ?_, by simp⟩
    rcases Array.mem_push.1 hx with hx | rfl
    · exact hs x hx
    · exact hn

/-- A verifier action that keeps the rounds. -/
theorem liftTM_keeps {B : ℕ} {α : Type} (x : TM α) (hx : ∀ t a t', x t = .ok (a, t') → t'.rounds = t.rounds) :
    (keepS B).R (liftTM x : TMW PS α) (liftTM x) := by
  intro t l a t' l' h
  change (monadLift (x t) : PS _).go l = _ at h
  rcases e : x t with _ | ⟨a', t''⟩
  · rw [e] at h; cases h
  · rw [e] at h
    cases h
    exact Keeps.same (hx t _ _ e)

theorem label_keeps (B : ℕ) (s : String) : (keepS B).R (liftTM (label s) : TMW PS Unit) (liftTM (label s)) :=
  liftTM_keeps _ fun _ _ _ h => by cases h; rfl

theorem absorbF_keeps (B : ℕ) (v : F128) : (keepS B).R (liftTM (absorbF v) : TMW PS Unit) (liftTM (absorbF v)) :=
  liftTM_keeps _ fun _ _ _ h => by cases h; rfl

theorem absorbSlice_keeps (B : ℕ) (vs : Array F128) :
    (keepS B).R (liftTM (absorbSlice vs) : TMW PS Unit) (liftTM (absorbSlice vs)) :=
  liftTM_keeps _ fun _ _ _ h => by cases h; rfl

theorem absorbK_keeps (B : ℕ) (v : F256) : (keepS B).R (liftTM (absorbK v) : TMW PS Unit) (liftTM (absorbK v)) :=
  liftTM_keeps _ fun _ _ _ h => by cases h; rfl

theorem absorbBytes_keeps (B : ℕ) (b : ByteArray) :
    (keepS B).R (liftTM (absorbBytes b) : TMW PS Unit) (liftTM (absorbBytes b)) :=
  liftTM_keeps _ fun _ _ _ h => by cases h; rfl

theorem pow_keeps (B : ℕ) (v : UInt64) : (keepS B).R (liftTM (pow v) : TMW PS Unit) (liftTM (pow v)) :=
  liftTM_keeps _ fun _ _ _ h => by cases h; rfl

theorem get_keeps (B : ℕ) : (keepS B).R (get : TMW PS Transcript) get := by
  intro t l a t' l' h
  simp only [StateT.run_get, go_pure, Prod.mk.injEq, Except.ok.injEq] at h
  obtain ⟨⟨-, rfl⟩, rfl⟩ := h
  exact Keeps.same rfl

/-- A computation of the pure session, lifted to the transcript, keeps its rounds. -/
theorem liftPS_keeps (B : ℕ) {α : Type} (y : PS α) : (keepS B).R (StateT.lift y : TMW PS α) (StateT.lift y) := by
  intro t l a t' l' e
  change (y >>= fun a => Pure.pure (a, t)).go l = _ at e
  obtain ⟨b, l₁, e1, e2⟩ := go_bind_ok e
  cases e2
  exact Keeps.same rfl

theorem liftT_keeps {B : ℕ} {α : Type} {x y : TMW PS α} (h : (keepS B).R x y) :
    ((keepS B).st LigState).R (liftT x) (liftT y) :=
  With.Sim.lift_st _ h

theorem sampleP_keeps (resp : ℕ → Claims.Msg) {B : ℕ} (h1 : 1 ≤ B) :
    (keepS B).R (sampleP (asksS resp)) (sampleP (asksS resp)) := by
  unfold sampleP
  sim_walk [squeezeP_keeps resp _ _ h1]

theorem sampleKP_keeps (resp : ℕ → Claims.Msg) {B : ℕ} (h2 : 2 ≤ B) :
    (keepS B).R (sampleKP (asksS resp)) (sampleKP (asksS resp)) := by
  unfold sampleKP
  sim_walk [squeezeP_keeps resp _ _ h2]

theorem lincheckPZk_keeps (resp : ℕ → Claims.Msg) (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (h : Array F128)
    (tr : Trace) (zc : ZcOut) {B : ℕ} (h1 : 1 ≤ B) :
    (keepS B).R (lincheckPZk (asksS resp) c m lay h tr zc) (lincheckPZk (asksS resp) c m lay h tr zc) := by
  unfold lincheckPZk
  sim_walk [sampleP_keeps resp h1, label_keeps, absorbSlice_keeps, absorbF_keeps]

theorem ringSwitchP_keeps (resp : ℕ → Claims.Msg) (tr : Trace) (claims : Array Claim) {B : ℕ} (h7 : 7 ≤ B)
    (hc : claims.size ≤ B) :
    (keepS B).R (ringSwitchP (asksS resp) tr claims) (ringSwitchP (asksS resp) tr claims) := by
  unfold ringSwitchP sampleVecP
  sim_walk [squeezeP_keeps resp _ _ h7, squeezeP_keeps resp _ _ hc, label_keeps, absorbSlice_keeps]

theorem sendMsg_keeps (B : ℕ) (m : F256 × F256) : ((keepS B).st LigState).R (sendMsg m : LP PS Unit) (sendMsg m) := by
  unfold sendMsg
  sim_walk [liftT_keeps, absorbK_keeps]

theorem powP_keeps (B : ℕ) (nonce : ℕ → ℕ → UInt64) (kind : ℕ) :
    ((keepS B).st LigState).R (powP nonce kind : LP PS Unit) (powP nonce kind) := by
  unfold powP
  sim_walk [liftT_keeps, pow_keeps]

theorem challengeP_keeps (resp : ℕ → Claims.Msg) (split : Bool) {B : ℕ} (h2 : 2 ≤ B) :
    ((keepS B).st LigState).R (challengeP (asksS resp) split) (challengeP (asksS resp) split) := by
  unfold challengeP
  sim_walk [liftT_keeps, sampleKP_keeps resp h2, sendMsg_keeps]

theorem oodIntroP_keeps (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ) {B : ℕ}
    (hd : dim ≤ B) (h1 : 1 ≤ B) :
    ((keepS B).st LigState).R (oodIntroP (asksS resp) nonce g dim) (oodIntroP (asksS resp) nonce g dim) := by
  unfold oodIntroP sampleVecP
  sim_walk [liftT_keeps, squeezeP_keeps resp _ _ hd, sampleP_keeps resp h1, sendMsg_keeps, powP_keeps,
    absorbF_keeps]

theorem queryP_keeps (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit)
    {B : ℕ} (hq : lv.q ≤ B) (hl : Nat.log2 (lv.q - 1) + 1 ≤ B) :
    ((keepS B).st LigState).R (queryP (asksS resp) nonce ms lv cm) (queryP (asksS resp) nonce ms lv cm) := by
  unfold queryP sampleVecP
  sim_walk [liftT_keeps, squeezeP_keeps resp _ _ hq, squeezeP_keeps resp _ _ hl, powP_keeps]

theorem tau_keeps (B : ℕ) (h : ByteArray → ByteArray) (tau : F128) (salt : ByteArray) :
    (keepS B).R (StateT.lift (Zk.tauCommitmentWith (Hashes.ofSha512 h : Hashes PS) tau salt) : TMW PS ByteArray)
      (StateT.lift (Zk.tauCommitmentWith (Hashes.ofSha512 h : Hashes PS) tau salt)) :=
  liftPS_keeps B _

theorem squeezeP_keeps_cons (resp : ℕ → Claims.Msg) (lay : Zk.PadLayout) (cons : Array (String × Zk.Lin)) {B : ℕ}
    (hc : cons.size + 3 ≤ B) (e r s : F128) :
    (keepS B).R (squeezeP (asksS resp) 2 (cons ++ Zk.triple lay e r s).size)
      (squeezeP (asksS resp) 2 (cons ++ Zk.triple lay e r s).size) :=
  squeezeP_keeps resp 2 _ (by simp [Zk.triple]; omega)

/-- **The inner proof keeps the rounds small**, given its constraints and three, and `Zk.Q`, at most `B`. -/
theorem innerP_keeps (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (lay : Zk.PadLayout) (p : PadsZk)
    (ic : InnerCommit) (finalC : F128) (cons : Array (String × Zk.Lin)) {B : ℕ} (hc : cons.size + 3 ≤ B)
    (hQ : Zk.Q ≤ B) :
    (keepS B).R (innerP (asksS resp) (Hashes.ofSha512 h) lay p ic finalC cons)
      (innerP (asksS resp) (Hashes.ofSha512 h) lay p ic finalC cons) := by
  have h1 : 1 ≤ B := le_trans (by decide) hQ
  unfold innerP sampleVecP
  sim_walk [sampleP_keeps resp h1, squeezeP_keeps resp _ _ hQ, squeezeP_keeps_cons resp lay cons hc, label_keeps,
    absorbF_keeps, absorbSlice_keeps, absorbBytes_keeps, tau_keeps B h]


theorem queryP_keeps_lv (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (sch : Schedule)
    {B : ℕ} (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ B) (j : ℕ) (cm : Commit) :
    ((keepS B).st LigState).R (queryP (asksS resp) nonce ms (sch.levels.getD j default) cm)
      (queryP (asksS resp) nonce ms (sch.levels.getD j default) cm) :=
  queryP_keeps resp nonce ms _ cm (by have := hq j; omega)
    (by have := hq j; have := Nat.log2_le_self ((sch.levels.getD j default).q - 1); omega)

theorem commitP_keeps (B : ℕ) (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (g : Array F128)
    (salt : ℕ → ByteArray) :
    (keepS B).R (StateT.lift (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt) : TMW PS Commit)
      (StateT.lift (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt)) :=
  liftPS_keeps B _

/-- A transcript computation of the pure session that keeps every round at most `B` coins, adds at least `k` rounds,
and returns a value with `Q`. -/
def Grows (B k : ℕ) {α : Type} (y : TMW PS α) (Q : α → Prop) : Prop :=
  ∀ (t : Transcript) (l : Array Claims.Msg) (a : α) (t' : Transcript) (l' : Array Claims.Msg),
    (y.run t).go l = (.ok (a, t'), l') → ((CoinsLe B t → CoinsLe B t') ∧ t.rounds.size + k ≤ t'.rounds.size) ∧ Q a

theorem Grows.of_keeps {B : ℕ} {α : Type} {y : TMW PS α} (h : (keepS B).R y y) : Grows B 0 y fun _ => True :=
  fun t l a t' l' e => ⟨h t l a t' l' e, trivial⟩

theorem Grows.bind {B k₁ k₂ : ℕ} {α β : Type} {x : TMW PS α} {f : α → TMW PS β} {Q : α → Prop} {P : β → Prop}
    (hx : Grows B k₁ x Q) (hf : ∀ a, Q a → Grows B k₂ (f a) P) : Grows B (k₁ + k₂) (x >>= f) P := by
  intro t l b t'' l'' h
  rw [StateT.run_bind] at h
  obtain ⟨⟨a, t'⟩, l', h1, h2⟩ := go_bind_ok h
  obtain ⟨k1, q1⟩ := hx t l a t' l' h1
  obtain ⟨k2, q2⟩ := hf a q1 t' l' b t'' l'' h2
  exact ⟨⟨k2.1 ∘ k1.1, by omega⟩, q2⟩

theorem Grows.bind_keeps {B k : ℕ} {α β : Type} {x : TMW PS α} {f : α → TMW PS β} {P : β → Prop}
    (hx : (keepS B).R x x) (hf : ∀ a, Grows B k (f a) P) : Grows B k (x >>= f) P := by
  simpa using Grows.bind (Grows.of_keeps hx) fun a _ => hf a

theorem Grows.bind_left {B k : ℕ} {α β : Type} {x : TMW PS α} {f : α → TMW PS β} {Q : α → Prop} {P : β → Prop}
    (hx : Grows B k x Q) (hf : ∀ a, Q a → Grows B 0 (f a) P) : Grows B k (x >>= f) P :=
  Grows.bind hx hf

theorem Grows.liftV {B : ℕ} {α : Type} (v : V α) :
    Grows B 0 (StateT.lift (monadLift v : PS α) : TMW PS α) fun a => v = .ok a := by
  intro t l a t' l' e
  change ((monadLift v : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at e
  obtain ⟨b, l₁, e1, e2⟩ := go_bind_ok e
  rw [monadLift_go] at e1
  simp only [Prod.mk.injEq] at e1
  obtain ⟨hv, rfl⟩ := e1
  cases e2
  exact ⟨Keeps.same rfl, by cases v <;> simp_all⟩

/-- A squeeze of a positive number of coins adds a round. -/
theorem squeezeP_grows (resp : ℕ → Claims.Msg) (kind : UInt8) (n : ℕ) {B : ℕ} (h0 : 0 < n) (hn : n ≤ B) :
    Grows B 1 (squeezeP (asksS resp) kind n) fun _ => True := by
  intro t l a t' l' h
  change (squeezeP (asksS resp) kind n t).go l = _ at h
  unfold squeezeP at h
  rw [ite_eq_right_of_eq_false _ _ (eq_false (by omega))] at h
  obtain ⟨coin, l₁, h1, h2⟩ := go_bind_ok h
  rw [round_go] at h1
  simp only [Prod.mk.injEq, Except.ok.injEq] at h1
  obtain ⟨rfl, rfl⟩ := h1
  simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at h2
  obtain ⟨⟨-, rfl⟩, rfl⟩ := h2
  refine ⟨⟨fun hs x hx => ?_, by simp⟩, trivial⟩
  rcases Array.mem_push.1 hx with hx | rfl
  · exact hs x hx
  · exact hn

/-- `Grows` of a Ligerito step, from every Ligerito state. -/
def GrowsL (B : ℕ) {α : Type} (y : LP PS α) (Q : α → Prop) : Prop := ∀ s, Grows B 0 (y.run s) fun p => Q p.1

theorem GrowsL.bind {B : ℕ} {α β : Type} {x : LP PS α} {f : α → LP PS β} {Q : α → Prop} {P : β → Prop}
    (hx : GrowsL B x Q) (hf : ∀ a, Q a → GrowsL B (f a) P) : GrowsL B (x >>= f) P := by
  intro s
  rw [StateT.run_bind]
  exact Grows.bind (hx s) fun p hp => hf p.1 hp p.2

theorem GrowsL.of_keeps {B : ℕ} {α : Type} {y : LP PS α} (h : ((keepS B).st LigState).R y y) :
    GrowsL B y fun _ => True :=
  fun s => Grows.of_keeps (h s)

theorem GrowsL.bind_keeps {B : ℕ} {α β : Type} {x : LP PS α} {f : α → LP PS β} {P : β → Prop}
    (hx : ((keepS B).st LigState).R x x) (hf : ∀ a, GrowsL B (f a) P) : GrowsL B (x >>= f) P :=
  GrowsL.bind (GrowsL.of_keeps hx) fun a _ => hf a

theorem GrowsL.pure {B : ℕ} {α : Type} {a : α} {P : α → Prop} (h : P a) : GrowsL B (Pure.pure a : LP PS α) P := by
  intro s t l b t' l' e
  simp only [StateT.run_pure, go_pure, Prod.mk.injEq, Except.ok.injEq] at e
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := e
  exact ⟨Keeps.same rfl, h⟩

theorem GrowsL.forIn_list {B : ℕ} {β : Type} (f : ℕ → β → LP PS (ForInStep β)) (I : β → Prop)
    (hf : ∀ i b, I b → GrowsL B (f i b) fun s => I s.value) :
    ∀ (xs : List ℕ) (b : β), I b → GrowsL B (forIn xs b f) I := by
  intro xs
  induction xs with
  | nil => intro b hb; exact GrowsL.pure hb
  | cons x xs ih =>
    intro b hb
    rw [List.forIn_cons]
    refine GrowsL.bind (hf x b hb) fun r hr => ?_
    cases r with
    | done b' => exact GrowsL.pure hr
    | yield b' => exact ih b' hr

/-- A loop over a range, named so that the walk stops at it. -/
def loopR {β : Type} (r : Std.Legacy.Range) (b : β) (f : ℕ → β → LP PS (ForInStep β)) : LP PS β := forIn r b f

theorem forIn_loopR (r : Std.Legacy.Range) (b : ℕ × Commit × Array (Array ByteArray) × Array LevelProof × FinalProof)
    (f : ℕ → ℕ × Commit × Array (Array ByteArray) × Array LevelProof × FinalProof →
      LP PS (ForInStep (ℕ × Commit × Array (Array ByteArray) × Array LevelProof × FinalProof))) :
    forIn r b f = loopR r b f := rfl

/-- **A loop that keeps an invariant of its state keeps its rounds.** -/
theorem loopR_keeps {B : ℕ} {β : Type} (I : β → Prop) (r : Std.Legacy.Range) (b : β)
    (f : ℕ → β → LP PS (ForInStep β)) (hb : I b) (hf : ∀ i b, I b → GrowsL B (f i b) fun s => I s.value) :
    ((keepS B).st LigState).R (loopR r b f) (loopR r b f) := by
  intro s t l a t' l' e
  unfold loopR at e
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at e
  simpa using (GrowsL.forIn_list f I hf _ b hb s t l a t' l' e).1

theorem oodIntroP_keeps_loop (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (N : ℕ) {B : ℕ} (hN : N ≤ B)
    (h1 : 1 ≤ B) (g : Array F128) (b : ℕ × Commit × Array (Array ByteArray) × Array LevelProof × FinalProof)
    (hb : b.1 ≤ N) :
    ((keepS B).st LigState).R (oodIntroP (asksS resp) nonce g (b.1 - 4 + 1))
      (oodIntroP (asksS resp) nonce g (b.1 - 4 + 1)) :=
  oodIntroP_keeps resp nonce g _ (by omega) h1

/-- **The masked Ligerito keeps the rounds small**, given `2`, `n + 1` and every level's `q + 1` at most `B`. -/
theorem ligeritoPZk_keeps (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (t b : Array F128) (lay : Zk.PadLayout) (hp : Array F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) {B : ℕ} (h2 : 2 ≤ B)
    (hn : sch.n + 1 ≤ B) (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ B) :
    (keepS B).R
      (ligeritoPZk (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 t b lay hp rep tPad extra lanePad salt nonce)
      (ligeritoPZk (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 t b lay hp rep tPad extra lanePad salt nonce) := by
  have h1 : 1 ≤ B := by omega
  unfold ligeritoPZk sampleVecP
  simp (config := { zeta := false }) only [forIn_loopR]
  sim_walk [liftT_keeps, squeezeP_keeps resp _ sch.n (by omega), sampleP_keeps resp h1, sampleKP_keeps resp h2,
    sendMsg_keeps, powP_keeps, challengeP_keeps resp _ h2, oodIntroP_keeps resp nonce _ (sch.n - sch.k0 + 1)
    (by omega) h1, queryP_keeps_lv resp nonce ms sch hq, commitP_keeps B h, label_keeps, absorbF_keeps,
    absorbBytes_keeps, absorbK_keeps, absorbSlice_keeps,
    loopR_keeps (fun s : ℕ × Commit × Array (Array ByteArray) × Array LevelProof × FinalProof => s.1 ≤ sch.n + 1)]
  · simp (config := { zetaDelta := true }) only []
    omega
  · intro i b hb
    repeat (first
      | extract_lets
      | (refine GrowsL.bind_keeps ?_ fun _ => ?_
         sim_walk [liftT_keeps, sampleP_keeps resp h1, sampleKP_keeps resp h2, sendMsg_keeps, powP_keeps,
           challengeP_keeps resp _ h2, oodIntroP_keeps_loop resp nonce (sch.n + 1) hn h1,
           queryP_keeps_lv resp nonce ms sch hq, commitP_keeps B h, absorbF_keeps, absorbBytes_keeps])
      | (refine GrowsL.pure ?_
         simp (config := { zetaDelta := true }) only [ForInStep.value]
         omega)
      | split)
  all_goals sim_walk

theorem Grows.pure {B : ℕ} {α : Type} {a : α} {P : α → Prop} (h : P a) : Grows B 0 (Pure.pure a : TMW PS α) P := by
  intro t l b t' l' e
  simp only [StateT.run_pure, go_pure, Prod.mk.injEq, Except.ok.injEq] at e
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := e
  exact ⟨Keeps.same rfl, h⟩

/-- **The masked zerocheck adds a round**: its first squeeze, of `6` coins. -/
theorem zerocheckPZk_grows (resp : ℕ → Claims.Msg) (m : ℕ) (digest cap root : ByteArray) (lay : Zk.PadLayout)
    (h : Array F128) (tr : Trace) {B : ℕ} (h6 : 6 ≤ B) (hm : m - 13 ≤ B) :
    Grows B 1 (zerocheckPZk (asksS resp) m digest cap root lay h tr) fun _ => True := by
  unfold zerocheckPZk sampleVecP
  refine Grows.bind_keeps (label_keeps B _) fun _ => ?_
  refine Grows.bind_keeps (absorbBytes_keeps B _) fun _ => ?_
  refine Grows.bind_keeps (absorbBytes_keeps B _) fun _ => ?_
  refine Grows.bind_keeps (label_keeps B _) fun _ => ?_
  refine Grows.bind_keeps (label_keeps B _) fun _ => ?_
  refine Grows.bind_keeps (absorbBytes_keeps B _) fun _ => ?_
  refine Grows.bind_left (squeezeP_grows resp _ 6 (by decide) h6) fun _ _ => Grows.of_keeps ?_
  sim_walk [squeezeP_keeps resp _ _ hm, sampleP_keeps resp (by omega : 1 ≤ B), absorbSlice_keeps, absorbF_keeps]

theorem rankCheckP_keeps (B m kLog : ℕ) (prior : Array (List F128)) (words : List ℕ) :
    (keepS B).R (rankCheckP m kLog prior words : TMW PS Unit) (rankCheckP m kLog prior words) := by
  unfold rankCheckP
  sim_walk [get_keeps B]

/-- **A masked rep adds a round and keeps the rounds small**, given `Zk.Q`, `m - 13`, its claims, `n + 1` and every
level's `q + 1` at most `B`. -/
theorem repPZk_grows (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Setup) (sch : Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (tr : Trace) (t : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List F128)) (words : List ℕ) {B : ℕ} (hQ : Zk.Q ≤ B)
    (hm : st.m - 13 ≤ B) (hx : extras.size + 2 ≤ B) (hn : sch.n + 1 ≤ B)
    (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ B) :
    Grows B 1 (repPZk (asksS resp) (Hashes.ofSha512 h) st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt
      nonce prior words) fun _ => True := by
  have hB : 168 ≤ B := hQ
  unfold repPZk
  refine Grows.bind_left (zerocheckPZk_grows resp _ _ _ _ _ _ _ (by omega) hm) fun ⟨zp, zc⟩ _ => ?_
  refine Grows.bind_keeps (lincheckPZk_keeps resp _ _ _ _ _ _ (by omega)) fun ⟨lp, lc⟩ => ?_
  refine Grows.bind_keeps (rankCheckP_keeps B _ _ _ _) fun _ => ?_
  refine Grows.bind_keeps (ringSwitchP_keeps resp _ _ (by omega) ?_) fun ⟨ss, T, sws⟩ => ?_
  · simp only [Array.size_append, abcClaims, List.size_toArray, List.length_cons, List.length_nil]
    omega
  refine Grows.bind_keeps (ligeritoPZk_keeps resp h _ sch _ _ _ _ _ _ _ _ _ _ _ _ (by omega) hn hq)
    fun ⟨lg, zl, rec⟩ => ?_
  refine Grows.bind_keeps (get_keeps B) fun tr' => ?_
  refine Grows.bind_left (Grows.liftV _) fun ⟨rp, t''⟩ hrp => ?_
  refine Grows.bind_left (Grows.liftV _) fun lgc hlgc => ?_
  have h3 := FlockSoundness.Discharge.ZkExec.replay_cons_size _ _ _ _ hrp
  have h2 := FlockSoundness.Discharge.ZkExec.ligeritoConstraints_size hlgc
  refine Grows.bind_keeps (innerP_keeps resp h _ _ _ _ _ ?_ hQ) fun _ => Grows.pure trivial
  simp only [Array.size_append, List.size_toArray, List.length_cons, List.length_nil, h3, h2]
  omega

/-- The `--zk` schedules squeeze few coins: `n + 1` and every level's `q + 1` are at most `65536`. -/
theorem zkSchedule_small {m : ℕ} (h1 : 25 ≤ m) (h2 : m ≤ 35) {v : Schedule × ℕ} (h : Zk.schedule m = .ok v) :
    v.1.n + 1 ≤ 65536 ∧ ∀ j, (v.1.levels.getD j default).q + 1 ≤ 65536 := by
  have key : ∀ k : Fin 11, (match Zk.schedule (k.val + 25) with
      | .ok v => decide (v.1.n + 1 ≤ 65536 ∧ ∀ j < v.1.levels.size, (v.1.levels.getD j default).q + 1 ≤ 65536)
      | .error _ => false) = true := by
    decide +kernel
  have := key ⟨m - 25, by omega⟩
  simp only [show m - 25 + 25 = m by omega, h, decide_eq_true_eq] at this
  obtain ⟨hn, hq⟩ := this
  refine ⟨hn, fun j => ?_⟩
  by_cases hj : j < v.1.levels.size
  · exact hq j hj
  · rw [Array.getD, dite_eq_right (fun h => absurd h hj)]
    decide

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The masked case's table has at most 1,024 regions**: the scope check's `checkInRange`. -/
theorem _root_.FlockComplete.CaseZk.regions (c : CaseZk H rel) :
    ∃ st, c.sts = #[st] ∧ st.regions.size ≤ 1024 := by
  obtain ⟨p, hp, hsc⟩ := (Readable.scope_iff (deployZk rel) rel (readOf true) rfl
    (fun _ _ h => readOf_zk h) c.x).1 c.scope
  have hi := (setupOf_some c.setup).1
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := c.s.read) hp
  obtain ⟨hc, -⟩ := (Readable.scope_ok_iff p).1 hsc
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  have hset := Ins.setup_read (setupOf_some c.setup).2
  rw [← ht] at hset
  unfold Readable.Read.setup at hset
  rw [hpi] at hset
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okAny hset with ⟨-, st, h1, -, hsts⟩ | ⟨hc1, hn, -⟩
  · refine ⟨st, hsts, ?_⟩
    rcases Bool.eq_false_or_eq_true p.tags.hiddenOutputs with hh | hh
    · rw [FlockSoundness.Discharge.HiddenExec.setupOne_hidden hh] at h1
      exact FlockVBridge.Family.spec_regions_le (FlockSoundness.Discharge.Hidden.setupHidden_spec' h1)
    · rw [FlockSoundness.Discharge.HiddenExec.setupOne_public hh] at h1
      exact FlockVBridge.Family.setupH_regions_le h1
  · have := c.ready_parts.1
    omega

/-- **Each masked rep records a round, each of at most `65536` coins** (`RepRoundsZk`). -/
theorem repRoundsZk_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    RepRoundsZk H rel := by
  intro c st hst v hv rep dom tr tb cm₀ extras pads ic extra lanePad salt nonce prior ws hex
  obtain ⟨st', hst', hspec, hm1, hm2⟩ := c.table
  obtain ⟨st'', hst'', hreg⟩ := c.regions
  rw [hst'] at hst hst''
  obtain rfl : st = st' := by simpa using hst
  obtain rfl : st = st'' := by simpa using hst''
  have hx : extras.size = st.regions.size * 2 := by
    rw [extraClaims_size hex]
    simp [JsonRT.ptsOf, hspec, Stmt.spec]
  have hm : (Setup.ofCircuit st).m = st.m := rfl
  rw [hm] at hv
  obtain ⟨hn, hq⟩ := zkSchedule_small hm1 hm2 hv
  rintro l ⟨p, t'⟩ l' hrun
  obtain ⟨⟨hs, hsz⟩, -⟩ := repPZk_grows c.resp _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ (B := 65536) (by decide)
    (by simp [Setup.ofCircuit]; omega) (by omega) hn hq _ l _ _ _ hrun
  exact ⟨show 0 < t'.rounds.size by simp [Transcript.init] at hsz; omega, hs (by simp [CoinsLe, Transcript.init])⟩

end FlockComplete.JsonRT
