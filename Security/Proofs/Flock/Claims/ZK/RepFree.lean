import Proofs.Flock.Claims.ZK.FinishSrc
import Proofs.Flock.Claims.ZK.PiopFree

/-!
# The cut rep returns on its coins alone

Answered by `asksCut`, a rep's sends are counted by its coins and its public sizes: every squeeze sends one body and is
answered by the coin of the log's size, every absorb, commitment and state step sends nothing, and the only refusals
past the PIOP read coins (the pair `ρ₀, ρ₁`, `β`). So where the firewall's rep returns, the rep from any source returns
from a log of the same size, with a log of the same size, whatever values either sends.

* **Forward runs** (`Fw R x y`): where `x` returns, `y` returns with a log of the same size and a result related by
  `R`. Binds (`fw_bind`), the shared squeezes (`Coined`), and on Ligerito's state, `FwL`: from states with as many
  messages, the same.
* **`ligeritoPZk_cut_fw`**: the firewall's masked Ligerito opening and the cut's, from logs of one size: where the
  firewall's returns, the cut's returns, with as many messages, lane challenges and out-of-domain values.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockComplete.CutFree

open Flock Transcript Replay LigLock

variable {α β γ δ : Type}

/-! ## Forward runs -/

/-- **Where `x` returns, `y` returns** with a log of the same size and a result related by `R`. -/
def Fw (R : α → β → Prop) (x : Except String (α × Transcript) × Array Claims.Msg)
    (y : Except String (β × Transcript) × Array Claims.Msg) : Prop :=
  ∀ a t L, x = (.ok (a, t), L) → ∃ b t' L', y = (.ok (b, t'), L') ∧ L'.size = L.size ∧ R a b

theorem run_ok {x : TMW PS α} {f : α → TMW PS β} {t t₂ : Transcript} {l L₂ : Array Claims.Msg} {b : β}
    (h : ((x >>= f).run t).go l = (.ok (b, t₂), L₂)) :
    ∃ a t₁ L, (x.run t).go l = (.ok (a, t₁), L) ∧ ((f a).run t₁).go L = (.ok (b, t₂), L₂) := by
  rcases e : (x.run t).go l with ⟨r, L⟩
  rcases r with err | ⟨a, t₁⟩
  · rw [bind_go_err e] at h
    simp at h
  · exact ⟨a, t₁, L, rfl, by rw [bind_go e] at h; exact h⟩

theorem fw_bind {R : α → β → Prop} {S : γ → δ → Prop} {x : TMW PS α} {y : TMW PS β} {f : α → TMW PS γ}
    {g : β → TMW PS δ} {t t' : Transcript} {l l' : Array Claims.Msg}
    (hxy : Fw R ((x.run t).go l) ((y.run t').go l'))
    (hfg : ∀ a b t₁ t₁' L L', L'.size = L.size → R a b → Fw S (((f a).run t₁).go L) (((g b).run t₁').go L')) :
    Fw S (((x >>= f).run t).go l) (((y >>= g).run t').go l') := by
  intro c t₂ L₂ h
  obtain ⟨a, t₁, L, h₁, h₂⟩ := run_ok h
  obtain ⟨b, t₁', L', e₁, hL, hR⟩ := hxy a t₁ L h₁
  rw [bind_go e₁]
  exact hfg a b t₁ t₁' L L' hL hR c t₂ L₂ h₂

theorem fw_pure {R : α → β → Prop} {a : α} {b : β} {t t' : Transcript} {l l' : Array Claims.Msg}
    (hl : l'.size = l.size) (h : R a b) :
    Fw R (((pure a : TMW PS α).run t).go l) (((pure b : TMW PS β).run t').go l') := by
  intro c t₂ L₂ hc
  rw [pure_go] at hc
  simp only [Prod.mk.injEq, Except.ok.injEq] at hc
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := hc
  exact ⟨b, t', l', pure_go _ _ _, hl, h⟩

/-- **Two transcript actions that send nothing**, the second one that never throws. -/
theorem fw_liftTM_bind {S : γ → δ → Prop} {x x' : TM Unit} {f : Unit → TMW PS γ} {g : Unit → TMW PS δ}
    {t t' : Transcript} {l l' : Array Claims.Msg} (hx' : ∃ t₁', x' t' = .ok ((), t₁'))
    (h : ∀ t₁ t₁', Fw S (((f ()).run t₁).go l) (((g ()).run t₁').go l')) :
    Fw S (((liftTM x >>= f).run t).go l) (((liftTM x' >>= g).run t').go l') := by
  intro c t₂ L₂ hc
  obtain ⟨a, t₁, L, h₁, h₂⟩ := run_ok hc
  obtain rfl := liftTM_inv h₁
  obtain ⟨t₁', e⟩ := hx'
  rw [bind_go (liftTM_ok e l')]
  exact h t₁ t₁' c t₂ L₂ h₂

/-! ## The shared squeezes -/

/-- **A step whose return reads only the log's size**: from a log of the same size it returns the same, with a log of
the same size. -/
def Coined (m : TMW PS α) : Prop :=
  ∀ (t t' : Transcript) (l l' : Array Claims.Msg) (a : α) (t₁ : Transcript) (L : Array Claims.Msg),
    (m.run t).go l = (.ok (a, t₁), L) → l'.size = l.size →
    ∃ t₁' L', (m.run t').go l' = (.ok (a, t₁'), L') ∧ L'.size = L.size

theorem coined_pure (a : α) : Coined (pure a : TMW PS α) := by
  intro t t' l l' b t₁ L h hl
  rw [pure_go] at h
  simp only [Prod.mk.injEq, Except.ok.injEq] at h
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
  exact ⟨t', l', pure_go _ _ _, hl⟩

theorem coined_bind {x : TMW PS α} {f : α → TMW PS β} (hx : Coined x) (hf : ∀ a, Coined (f a)) :
    Coined (x >>= f) := by
  intro t t' l l' b t₂ L₂ h hl
  obtain ⟨a, t₁, L, h₁, h₂⟩ := run_ok h
  obtain ⟨t₁', L', e₁, hL⟩ := hx t t' l l' a t₁ L h₁ hl
  obtain ⟨t₂', L₂', e₂, hL₂⟩ := hf a t₁ t₁' L L' b t₂ L₂ h₂ hL
  exact ⟨t₂', L₂', by rw [bind_go e₁]; exact e₂, hL₂⟩

theorem squeezeP_zero_go (coin : ℕ → ℕ → F128) (n : ℕ) (kind : UInt8) (t : Transcript) (l : Array Claims.Msg) :
    ((squeezeP (asksCut coin n) kind 0).run t).go l = (.ok (#[], { t with pending := framed t kind 0 }), l) := by
  simp only [squeezeP, StateT.run, ↓reduceIte]
  rfl

theorem coined_squeezeP (coin : ℕ → ℕ → F128) (n : ℕ) (kind : UInt8) (k : ℕ) :
    Coined (squeezeP (asksCut coin n) kind k) := by
  intro t t' l l' a t₁ L h hl
  by_cases hk : k = 0
  · subst hk
    rw [squeezeP_zero_go] at h
    simp only [Prod.mk.injEq, Except.ok.injEq] at h
    obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
    exact ⟨_, _, squeezeP_zero_go coin n kind t' l', hl⟩
  · obtain ⟨hn, rfl, hL⟩ := squeezeP_cut_inv coin kind hk h
    refine ⟨afterSq t' kind k (coin l'.size), l'.push (Claims.toMsg (framed t' kind k)), ?_, ?_⟩
    · rw [squeezeP_cut_ok coin kind hk t' (by omega), hl]
    · simp only [Array.size_push, hL, hl]

theorem coined_sampleP (coin : ℕ → ℕ → F128) (n : ℕ) : Coined (sampleP (asksCut coin n)) := by
  unfold sampleP
  exact coined_bind (coined_squeezeP coin n 1 1) fun _ => coined_pure _

theorem coined_sampleKP (coin : ℕ → ℕ → F128) (n : ℕ) : Coined (sampleKP (asksCut coin n)) := by
  unfold sampleKP
  exact coined_bind (coined_squeezeP coin n 2 2) fun _ => coined_pure _

theorem coined_sampleVecP (coin : ℕ → ℕ → F128) (n k : ℕ) : Coined (sampleVecP (asksCut coin n) k) :=
  coined_squeezeP coin n 2 k

/-! ## Ligerito's steps -/

/-- Two Ligerito states with as many messages. -/
abbrev SR (s s' : LigState) : Prop := s.msgs.size = s'.msgs.size

/-- **Forward Ligerito runs**: from states with as many messages, transcripts and logs of one size, where `x` returns,
`y` returns, with as many messages, a log of the same size and a result related by `R`. -/
def FwL (R : α → β → Prop) (x : LP PS α) (y : LP PS β) : Prop :=
  ∀ (s s' : LigState) (t t' : Transcript) (l l' : Array Claims.Msg), SR s s' → l'.size = l.size →
    Fw (fun a b => R a.1 b.1 ∧ SR a.2 b.2) (((x.run s).run t).go l) (((y.run s').run t').go l')

theorem liftT_run (m : TMW PS α) (s : LigState) : (liftT m : LP PS α).run s = m >>= fun a => pure (a, s) := by
  simp only [liftT, StateT.run_lift]

theorem fwL_bind {R : α → β → Prop} {S : γ → δ → Prop} {x : LP PS α} {y : LP PS β} {f : α → LP PS γ}
    {g : β → LP PS δ} (hxy : FwL R x y) (hfg : ∀ a b, R a b → FwL S (f a) (g b)) : FwL S (x >>= f) (y >>= g) := by
  intro s s' t t' l l' hS hl
  rw [StateT.run_bind, StateT.run_bind]
  exact fw_bind (f := fun p => (f p.1).run p.2) (g := fun p => (g p.1).run p.2) (hxy s s' t t' l l' hS hl)
    fun p q t₁ t₁' L L' hL hpq => hfg p.1 q.1 hpq.1 p.2 q.2 t₁ t₁' L L' hpq.2 hL

theorem fwL_pure {R : α → β → Prop} {a : α} {b : β} (h : R a b) : FwL R (pure a) (pure b) :=
  fun _ _ _ _ _ _ hS hl => fw_pure hl ⟨h, hS⟩

theorem fwL_coined {m : TMW PS α} (hm : Coined m) : FwL (· = ·) (liftT m) (liftT m) := by
  intro s s' t t' l l' hS hl c t₂ L₂ hc
  rw [liftT_run] at hc
  obtain ⟨a, t₁, L, h₁, h₂⟩ := run_ok hc
  simp only [pure_go, Prod.mk.injEq, Except.ok.injEq] at h₂
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h₂
  obtain ⟨t₁', L', e₁, hL⟩ := hm t t' l l' a t₁ L h₁ hl
  exact ⟨(a, s'), t₁', L', by rw [liftT_run, bind_go e₁]; rfl, hL, rfl, hS⟩

/-- **Two transcript actions that send nothing**, the second one that never throws. -/
theorem fwL_liftTM {x x' : TM Unit} (hx' : ∀ t', ∃ t₁', x' t' = .ok ((), t₁')) :
    FwL (fun _ _ => True) (liftT (liftTM x)) (liftT (liftTM x')) := by
  intro s s' t t' l l' hS hl c t₂ L₂ hc
  rw [liftT_run] at hc
  obtain ⟨a, t₁, L, h₁, h₂⟩ := run_ok hc
  obtain rfl := liftTM_inv h₁
  simp only [pure_go, Prod.mk.injEq, Except.ok.injEq] at h₂
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h₂
  obtain ⟨t₁', e⟩ := hx' t'
  exact ⟨((), s'), t₁', l', by rw [liftT_run, bind_go (liftTM_ok e l')]; rfl, hl, trivial, hS⟩

theorem fwL_modify (F G : LigState → LigState) (hFG : ∀ s s', SR s s' → SR (F s) (G s')) :
    FwL (fun _ _ => True) (modify F) (modify G) := by
  intro s s' t t' l l' hS hl c t₂ L₂ hc
  rw [StateT.run_modify] at hc
  rw [pure_go] at hc
  simp only [Prod.mk.injEq, Except.ok.injEq] at hc
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := hc
  exact ⟨(PUnit.unit, G s'), t', l', by rw [StateT.run_modify]; rfl, hl, trivial, hFG s s' hS⟩

theorem fwL_set {a a' : LigState} (h : SR a a') : FwL (fun _ _ => True) (set a) (set a') := by
  intro s s' t t' l l' hS hl c t₂ L₂ hc
  rw [StateT.run_set, pure_go] at hc
  simp only [Prod.mk.injEq, Except.ok.injEq] at hc
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := hc
  exact ⟨(PUnit.unit, a'), t', l', by rw [StateT.run_set]; rfl, hl, trivial, h⟩

theorem fwL_get {S : γ → δ → Prop} {f : LigState → LP PS γ} {g : LigState → LP PS δ}
    (h : ∀ s s', SR s s' → FwL S (f s) (g s')) : FwL S (get >>= f) (get >>= g) := by
  intro s s' t t' l l' hS hl
  rw [get_bind_run, get_bind_run]
  exact h s s' hS s s' t t' l l' hS hl

theorem fwL_get_left {S : γ → δ → Prop} {f : LigState → LP PS γ} {y : LP PS δ} (h : ∀ s, FwL S (f s) y) :
    FwL S (get >>= f) y := by
  intro s s' t t' l l' hS hl
  rw [get_bind_run]
  exact h s s s' t t' l l' hS hl

theorem fwL_throw_bind {S : γ → δ → Prop} {e : String} {f : α → LP PS γ} {y : LP PS δ} :
    FwL S (throw e >>= f) y := by
  intro s s' t t' l l' _ _ c t₂ L₂ hc
  rw [StateT.run_bind, bind_go_err (LigLock.throw_go e s t l)] at hc
  simp at hc

theorem fwL_ite {R : α → β → Prop} {c : Prop} [Decidable c] {x x' : LP PS α} {y y' : LP PS β}
    (h₁ : c → FwL R x y) (h₂ : ¬ c → FwL R x' y') : FwL R (if c then x else x') (if c then y else y') := by
  by_cases hc : c
  · rw [if_pos hc, if_pos hc]
    exact h₁ hc
  · rw [if_neg hc, if_neg hc]
    exact h₂ hc

theorem fwL_sendMsg (m m' : F256 × F256) : FwL (fun _ _ => True) (sendMsg m) (sendMsg m') := by
  unfold sendMsg
  refine fwL_bind (fwL_modify _ _ fun s s' h => ?_) fun _ _ _ => ?_
  · show (s.msgs.push m).size = (s'.msgs.push m').size
    simp only [Array.size_push, h]
  refine fwL_bind (fwL_liftTM fun _ => ⟨_, rfl⟩) fun _ _ _ => ?_
  exact fwL_liftTM fun _ => ⟨_, rfl⟩

theorem fwL_powP (nonce : ℕ → ℕ → UInt64) (kind : ℕ) : FwL (fun _ _ => True) (powP nonce kind) (powP nonce kind) := by
  unfold powP
  refine fwL_get fun s s' hS => ?_
  refine fwL_bind (fwL_set ?_) fun _ _ _ => fwL_liftTM fun _ => ⟨_, rfl⟩
  rcases kind with _ | _ | kind <;> exact hS

/-- A level's commitment sends nothing and returns. -/
theorem lift_commit_go (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (g : Array F128)
    (salt : ℕ → ByteArray) (t : Transcript) (l : Array Claims.Msg) :
    ∃ cm, ((StateT.lift (commitP (Hashes.ofSha512 h) ms lv g salt) : TMW PS Commit).run t).go l = (.ok (cm, t), l) := by
  obtain ⟨cm, hcm⟩ := JsonRT.commitP_ok h ms lv g salt
  refine ⟨cm, ?_⟩
  rw [StateT.run_lift, go_bind, commitP_quiet h ms lv g salt l, hcm]
  rfl

theorem fwL_commit {h h' : ByteArray → ByteArray} {ms ms' : MerkleScheme} {lv lv' : Level} {g g' : Array F128}
    {salt salt' : ℕ → ByteArray} :
    FwL (fun _ _ => True) (liftT (StateT.lift (commitP (Hashes.ofSha512 h) ms lv g salt)))
      (liftT (StateT.lift (commitP (Hashes.ofSha512 h') ms' lv' g' salt'))) := by
  intro s s' t t' l l' hS hl c t₂ L₂ hc
  obtain ⟨cm, e⟩ := lift_commit_go h ms lv g salt t l
  obtain ⟨cm', e'⟩ := lift_commit_go h' ms' lv' g' salt' t' l'
  rw [liftT_run, bind_go e] at hc
  simp only [pure_go, Prod.mk.injEq, Except.ok.injEq] at hc
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := hc
  exact ⟨(cm', s'), t', l', by rw [liftT_run, bind_go e']; rfl, hl, trivial, hS⟩

/-- **A loop of forward steps**: if step `i`, from values related by `I i`, yields values related by `I (i + 1)`, the
loops over `[0:n]` end related by `I n`. -/
theorem fwL_forIn {σ τ : Type} (n : ℕ) (body : ℕ → σ → LP PS (ForInStep σ)) (body' : ℕ → τ → LP PS (ForInStep τ))
    (I : ℕ → σ → τ → Prop)
    (hb : ∀ i < n, ∀ u u', I i u u' →
      FwL (fun r r' => ∃ v v', r = .yield v ∧ r' = .yield v' ∧ I (i + 1) v v') (body i u) (body' i u'))
    {u : σ} {u' : τ} (h₀ : I 0 u u') : FwL (I n) (forIn [:n] u body) (forIn [:n] u' body') := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (u : σ) (u' : τ), a + k = n → I a u u' →
      FwL (I n) (forIn (List.range' a k) u body) (forIn (List.range' a k) u' body') from
    key n 0 u u' (by omega) h₀
  intro k
  induction k with
  | zero =>
    intro a u u' ha hI
    obtain rfl : a = n := by omega
    exact fwL_pure hI
  | succ k ih =>
    intro a u u' ha hI
    rw [List.range'_succ, List.forIn_cons, List.forIn_cons]
    refine fwL_bind (hb a (by omega) u u' hI) fun r r' hr => ?_
    obtain ⟨v, v', rfl, rfl, hI₁⟩ := hr
    exact ih (a + 1) v v' (by omega) hI₁

/-- **A run that sends nothing and keeps the state.** -/
def Quiet (x : LP PS α) : Prop :=
  ∀ s t l, ∃ a t', (((x.run s).run t).go l) = (.ok ((a, s), t'), l)

theorem quiet_pure (a : α) : Quiet (pure a : LP PS α) := fun s t l => ⟨a, t, rfl⟩

theorem quiet_bind {x : LP PS α} {f : α → LP PS β} (hx : Quiet x) (hf : ∀ a, Quiet (f a)) : Quiet (x >>= f) := by
  intro s t l
  obtain ⟨a, t₁, e₁⟩ := hx s t l
  obtain ⟨b, t₂, e₂⟩ := hf a s t₁ l
  refine ⟨b, t₂, ?_⟩
  rw [StateT.run_bind, bind_go e₁]
  exact e₂

theorem quiet_liftTM {x : TM Unit} (hx : ∀ t, ∃ t', x t = .ok ((), t')) : Quiet (liftT (liftTM x) : LP PS Unit) := by
  intro s t l
  obtain ⟨t', e⟩ := hx t
  exact ⟨(), t', by rw [liftT_run, bind_go (liftTM_ok e l)]; rfl⟩

theorem quiet_forIn_list {σ τ : Type} {body : τ → σ → LP PS (ForInStep σ)} (hb : ∀ x u, Quiet (body x u)) :
    ∀ (xs : List τ) (u : σ), Quiet (forIn xs u body)
  | [], u => by
    rw [List.forIn_nil]
    exact quiet_pure u
  | x :: xs, u => by
    rw [List.forIn_cons]
    refine quiet_bind (hb x u) fun r => ?_
    cases r with
    | done v => exact quiet_pure v
    | yield v => exact quiet_forIn_list hb xs v

theorem quiet_forIn_array {σ τ : Type} {body : τ → σ → LP PS (ForInStep σ)} (hb : ∀ x u, Quiet (body x u))
    (xs : Array τ) (u : σ) : Quiet (forIn xs u body) := by
  rw [← Array.forIn_toList]
  exact quiet_forIn_list hb xs.toList u

theorem fwL_quiet {x : LP PS α} {y : LP PS β} (hx : Quiet x) (hy : Quiet y) : FwL (fun _ _ => True) x y := by
  intro s s' t t' l l' hS hl c t₂ L₂ hc
  obtain ⟨a, t₁, e⟩ := hx s t l
  rw [e] at hc
  simp only [Prod.mk.injEq, Except.ok.injEq] at hc
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := hc
  obtain ⟨b, t₁', e'⟩ := hy s' t' l'
  exact ⟨(b, s'), t₁', l', e', hl, trivial, hS⟩

theorem fwL_queryP (coin : ℕ → ℕ → F128) (n : ℕ) {nonce : ℕ → ℕ → UInt64} {ms ms' : MerkleScheme} {lv : Level}
    {cm cm' : Commit} :
    FwL (fun _ _ => True) (queryP (asksCut coin n) nonce ms lv cm) (queryP (asksCut coin n) nonce ms' lv cm') := by
  unfold queryP
  refine fwL_bind (fwL_powP _ _) fun _ _ _ => ?_
  refine fwL_bind (fwL_coined (coined_sampleVecP coin n _)) fun a b hab => ?_
  subst hab
  try dsimp only
  refine fwL_bind (fwL_powP _ _) fun _ _ _ => ?_
  refine fwL_bind (fwL_coined (coined_sampleVecP coin n _)) fun a' b' hab' => ?_
  subst hab'
  exact fwL_pure trivial

theorem fwL_oodIntroP (coin : ℕ → ℕ → F128) (n : ℕ) {nonce : ℕ → ℕ → UInt64} {g g' : Array F128} {dim : ℕ} :
    FwL (fun _ _ => True) (oodIntroP (asksCut coin n) nonce g dim) (oodIntroP (asksCut coin n) nonce g' dim) := by
  unfold oodIntroP
  refine fwL_bind (fwL_coined (coined_sampleVecP coin n dim)) fun z z' hz => ?_
  subst hz
  try dsimp only
  refine fwL_bind (fwL_modify _ _ fun _ _ h => h) fun _ _ _ => ?_
  refine fwL_bind (fwL_liftTM fun _ => ⟨_, rfl⟩) fun _ _ _ => ?_
  try dsimp only
  refine fwL_bind (fwL_sendMsg _ _) fun _ _ _ => ?_
  refine fwL_bind (fwL_powP _ _) fun _ _ _ => ?_
  refine fwL_bind (fwL_coined (coined_sampleP coin n)) fun b b' hb => ?_
  subst hb
  exact fwL_modify _ _ fun _ _ h => h

theorem fwL_challengeP (coin : ℕ → ℕ → F128) (n : ℕ) {split : Bool} :
    FwL (fun _ _ => True) (challengeP (asksCut coin n) split) (challengeP (asksCut coin n) split) := by
  unfold challengeP
  refine fwL_bind (fwL_coined (coined_sampleKP coin n)) fun r r' hr => ?_
  subst hr
  refine fwL_bind (fwL_modify _ _ fun _ _ h => h) fun _ _ _ => ?_
  try dsimp only
  cases split
  · try simp only [Bool.false_eq_true, ↓reduceIte]
    refine fwL_get fun _ _ _ => ?_
    exact fwL_bind (fwL_sendMsg _ _) fun _ _ _ => fwL_pure trivial
  · try simp only [↓reduceIte]
    refine fwL_bind (fwL_modify _ _ fun _ _ h => h) fun _ _ _ => ?_
    refine fwL_get fun _ _ _ => ?_
    exact fwL_bind (fwL_sendMsg _ _) fun _ _ _ => fwL_pure trivial

/-- **A run of Ligerito's program, then a transcript step**, forward. -/
theorem fw_runL_bind {R : α → β → Prop} {S : γ → δ → Prop} {x : LP PS α} {y : LP PS β}
    {f : α × LigState → TMW PS γ} {g : β × LigState → TMW PS δ} {s s' : LigState} {t t' : Transcript}
    {l l' : Array Claims.Msg} (hxy : FwL R x y) (hS : SR s s') (hl : l'.size = l.size)
    (hfg : ∀ a b s₁ s₁' t₁ t₁' L L', L'.size = L.size → R a b → SR s₁ s₁' →
      Fw S (((f (a, s₁)).run t₁).go L) (((g (b, s₁')).run t₁').go L')) :
    Fw S (((x.run s >>= f).run t).go l) (((y.run s' >>= g).run t').go l') :=
  fw_bind (hxy s s' t t' l l' hS hl) fun p q t₁ t₁' L L' hL hpq =>
    hfg p.1 q.1 p.2 q.2 t₁ t₁' L L' hL hpq.1 hpq.2

/-! ## The Ligerito opening -/

set_option maxHeartbeats 4000000 in
/-- **The cut's Ligerito opening returns where the firewall's did**: answered by `asksCut` from logs of one size, where
the masked opening (`ligeritoPZk`, every nonce `0`) returns, the opening from any source returns, with a log of the
same size and as many messages, lane challenges and out-of-domain values; whatever the tables, bases, layouts, pads,
commitments, schemes and hashes on either side. -/
theorem ligeritoPZk_cut_fw (coin : ℕ → ℕ → F128) (n : ℕ) (h h' : ByteArray → ByteArray) (ms ms' : MerkleScheme)
    (sch : Schedule) (T T' : F128) (cm0 cm0' : Commit) (t b b' : Array F128) (lay lay' : Zk.PadLayout)
    (hp : Array F128) (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (src : RepSrc)
    (tr tr' : Transcript) (l l' : Array Claims.Msg) (hl : l'.size = l.size) :
    Fw (fun a c => a.1.messages.size = c.1.messages.size ∧ a.2.2.challenges.size = c.2.2.challenges.size ∧
        a.2.2.ood.size = c.2.2.ood.size)
      (((ligeritoPZk (asksCut coin n) (Hashes.ofSha512 h) ms sch T cm0 t b lay hp rep tPad extra lanePad salt
        fun _ _ => 0).run tr).go l)
      (((ligeritoCut (asksCut coin n) (Hashes.ofSha512 h') ms' sch T' cm0' b' lay' src).run tr').go l') := by
  unfold ligeritoPZk ligeritoCut
  try dsimp only
  refine fw_liftTM_bind ⟨_, rfl⟩ fun _ _ => ?_
  refine fw_liftTM_bind ⟨_, rfl⟩ fun _ _ => ?_
  refine fw_liftTM_bind ⟨_, rfl⟩ fun _ _ => ?_
  refine fw_liftTM_bind ⟨_, rfl⟩ fun _ _ => ?_
  refine fw_runL_bind (R := fun (a a' : Array (Array ByteArray) × Array LevelProof × FinalProof × LevelProof ×
      Zk.Level0Msgs × Array F256 × (F256 × F256) × F128 × F128) => a.2.2.2.2.2.1.size = a'.2.2.2.2.2.1.size)
    ?prog rfl hl ?tail
  case tail =>
    intro a a' s₁ s₁' t₁ t₁' L L' hL hR hS
    rcases a with ⟨caps, rproofs, fin, op0, zl, initCh, rho, b0, y⟩
    rcases a' with ⟨caps', rproofs', fin', op0', zl', initCh', rho', b0', y'⟩
    exact fw_pure hL ⟨hS, hR, rfl⟩
  case prog =>
    refine fwL_bind (fwL_coined (coined_sampleVecP coin n _)) fun z z' hz => ?_
    subst hz
    try dsimp only
    refine fwL_bind (fwL_modify _ _ fun _ _ h => h) fun _ _ _ => ?_
    refine fwL_bind (fwL_liftTM fun _ => ⟨_, rfl⟩) fun _ _ _ => ?_
    refine fwL_bind (fwL_powP _ _) fun _ _ _ => ?_
    refine fwL_bind (fwL_coined (coined_sampleP coin n)) fun b0 b0' hb0 => ?_
    subst hb0
    refine fwL_bind (fwL_modify _ _ fun _ _ h => h) fun _ _ _ => ?_
    refine fwL_get_left fun _ => ?_
    refine fwL_bind (fwL_sendMsg _ _) fun _ _ _ => ?_
    try dsimp only
    refine fwL_bind (fwL_forIn sch.k0 _ _
      (fun _ (u u' : Array F256 × (F256 × F256) × Zk.Level0Msgs) => u.1.size = u'.1.size) ?step rfl)
      fun r r' hr => ?rest
    case step =>
      intro j hj ⟨initCh, rho, zl⟩ ⟨initCh', rho', zl'⟩ hI
      try dsimp only at hI ⊢
      refine fwL_bind (fwL_coined (coined_sampleKP coin n)) fun r r' hr => ?_
      subst hr
      try dsimp only
      refine fwL_bind (fwL_modify _ _ fun _ _ h => h) fun _ _ _ => ?_
      refine fwL_ite (fun _ => ?_) (fun _ => ?_)
      · refine fwL_get fun _ _ _ => ?_
        try dsimp only
        refine fwL_bind (fwL_liftTM fun _ => ⟨_, rfl⟩) fun _ _ _ => ?_
        refine fwL_bind (fwL_liftTM fun _ => ⟨_, rfl⟩) fun _ _ _ => ?_
        refine fwL_bind (fwL_coined (coined_sampleKP coin n)) fun rh0 rh0' h0 => ?_
        subst h0
        refine fwL_bind (fwL_coined (coined_sampleKP coin n)) fun rh1 rh1' h1 => ?_
        subst h1
        try dsimp only
        refine fwL_ite (fun _ => fwL_throw_bind) (fun _ => ?_)
        refine fwL_bind (fwL_liftTM fun _ => ⟨_, rfl⟩) fun _ _ _ => ?_
        try dsimp only
        refine fwL_bind (fwL_liftTM fun _ => ⟨_, rfl⟩) fun _ _ _ => ?_
        try dsimp only
        refine fwL_bind (fwL_liftTM fun _ => ⟨_, rfl⟩) fun _ _ _ => ?_
        try dsimp only
        refine fwL_bind (fwL_modify _ _ fun _ _ h => h) fun _ _ _ => ?_
        refine fwL_get fun _ _ _ => ?_
        refine fwL_bind (fwL_sendMsg _ _) fun _ _ _ => ?_
        exact fwL_pure ⟨_, _, rfl, rfl, by simp only [Array.size_push, hI]⟩
      · refine fwL_get_left fun _ => ?_
        refine fwL_bind (fwL_sendMsg _ _) fun _ _ _ => ?_
        exact fwL_pure ⟨_, _, rfl, rfl, by simp only [Array.size_push, hI]⟩
    case rest =>
      rcases r with ⟨initCh, rho, zl⟩
      rcases r' with ⟨initCh', rho', zl'⟩
      try dsimp only at hr ⊢
      refine fwL_get fun _ _ _ => ?_
      try dsimp only
      refine fwL_bind fwL_commit fun _ _ _ => ?_
      refine fwL_bind (fwL_liftTM fun _ => ⟨_, rfl⟩) fun _ _ _ => ?_
      refine fwL_bind (fwL_oodIntroP coin n) fun _ _ _ => ?_
      refine fwL_bind (fwL_oodIntroP coin n) fun _ _ _ => ?_
      refine fwL_bind (fwL_queryP coin n) fun ⟨_, _⟩ ⟨_, _⟩ _ => ?_
      try dsimp only
      refine fwL_bind (fwL_sendMsg _ _) fun _ _ _ => ?_
      refine fwL_bind (fwL_powP _ _) fun _ _ _ => ?_
      refine fwL_bind (fwL_coined (coined_sampleP coin n)) fun bq bq' hbq => ?_
      subst hbq
      refine fwL_bind (fwL_modify _ _ fun _ _ h => h) fun _ _ _ => ?_
      try dsimp only
      refine fwL_bind (fwL_forIn _ _ _ (fun _ u u' => u.1 = u'.1) ?lv rfl) fun _ _ _ => ?done
      case done =>
        try dsimp only
        exact fwL_pure hr
      case lv =>
        intro i hi ⟨sd, pc, cp, rp, fi⟩ ⟨sd', pc', cp', rp', fi'⟩ hu
        try dsimp only at hu ⊢
        subst hu
        refine fwL_bind (fwL_forIn 4 _ _ (fun _ _ _ => True) ?inner trivial) fun _ _ _ => ?_
        case inner =>
          intro j hj f f' _
          try dsimp only
          refine fwL_ite (fun _ => ?_) (fun _ => ?_)
          · refine fwL_bind (fwL_coined (coined_sampleKP coin n)) fun r r' hr => ?_
            subst hr
            refine fwL_bind (fwL_modify _ _ fun _ _ h => h) fun _ _ _ => ?_
            refine fwL_get fun _ _ _ => ?_
            try dsimp only
            refine fwL_bind (fwL_modify _ _ fun _ _ h => h) fun _ _ _ => ?_
            refine fwL_get fun _ _ _ => ?_
            refine fwL_bind (fwL_sendMsg _ _) fun _ _ _ => ?_
            exact fwL_pure ⟨_, _, rfl, rfl, trivial⟩
          · refine fwL_bind (fwL_challengeP coin n) fun _ _ _ => ?_
            exact fwL_pure ⟨_, _, rfl, rfl, trivial⟩
        try dsimp only
        refine fwL_ite (fun _ => ?_) (fun _ => ?_)
        · refine fwL_bind fwL_commit fun _ _ _ => ?_
          refine fwL_bind (fwL_liftTM fun _ => ⟨_, rfl⟩) fun _ _ _ => ?_
          refine fwL_bind (fwL_oodIntroP coin n) fun _ _ _ => ?_
          refine fwL_bind (fwL_oodIntroP coin n) fun _ _ _ => ?_
          refine fwL_bind (fwL_queryP coin n) fun ⟨_, _⟩ ⟨_, _⟩ _ => ?_
          try dsimp only
          refine fwL_bind (fwL_sendMsg _ _) fun _ _ _ => ?_
          refine fwL_bind (fwL_powP _ _) fun _ _ _ => ?_
          refine fwL_bind (fwL_coined (coined_sampleP coin n)) fun bb bb' hbb => ?_
          subst hbb
          refine fwL_bind (fwL_modify _ _ fun _ _ h => h) fun _ _ _ => ?_
          exact fwL_pure ⟨_, _, rfl, rfl, rfl⟩
        · refine fwL_bind (fwL_quiet (quiet_forIn_array (fun _ _ => ?_) _ _)
            (quiet_forIn_array (fun _ _ => ?_) _ _)) fun _ _ _ => ?_
          · try dsimp only
            exact quiet_bind (quiet_liftTM fun _ => ⟨_, rfl⟩) fun _ => quiet_pure _
          · try dsimp only
            exact quiet_bind (quiet_liftTM fun _ => ⟨_, rfl⟩) fun _ => quiet_pure _
          refine fwL_bind (fwL_queryP coin n) fun ⟨_, _⟩ ⟨_, _⟩ _ => ?_
          try dsimp only
          refine fwL_bind (fwL_powP _ _) fun _ _ _ => ?_
          refine fwL_bind (fwL_coined (coined_sampleP coin n)) fun _ _ _ => ?_
          exact fwL_pure ⟨_, _, rfl, rfl, rfl⟩

end FlockComplete.CutFree
