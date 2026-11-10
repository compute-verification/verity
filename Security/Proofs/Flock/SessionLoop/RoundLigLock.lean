module

public import Proofs.Flock.Claims.ZK.SimLig
public import Proofs.Flock.Claims.Complete.Link

/-!
# Ligerito past the restart, in lockstep (`ligeritoCut_lock`)

`ligeritoCut` (the cut's opening, from the source) and `ligeritoPZk` (the session's, at any pads `h`) answered by the same
`asksCut coin N` squeeze at the same rounds, so their logs have the same length throughout, and they refuse the same
pair `ρ₀, ρ₁`. Before the restart they absorb different values (the source's masked values against the session's, and
the header's `T` and `cm0`), so their bodies differ; from the pair's coins on both pendings are empty, and where the
source holds the session's `y₁'`, `ȳ` and levels ≥ 1's salts at the coins (`LigHyp`'s three fields), they absorb the same
from the same tables and weights, so every body from the restart on is the same.

* **Two runs whose logs agree from body `B` on** (`LogEq`, `Lk`): the same length and the same bodies from `B`, both
  refused or both returned, the pendings equal where `Q` holds of the length.
* **Steps** in lockstep: binds (`lk_bind`, `lk_lbind`), absorbs (`lk_tabs`, `lk_lF`, `lk_lK`, …), squeezes (`lk_lsq`,
  `lk_lsqAny`), commitments (`lk_commit`), loops (`lk_forIn`, `lk_absorbs`), and the opening's blocks (`lk_challenge`,
  `lk_ood`, `lk_query`).
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockComplete.LigPast

open Flock Transcript Replay LigLock

/-! ## Logs that agree from a body on -/

/-- **Two logs agreeing from body `B` on**: the same length, and the same body at each index `≥ B`. -/
structure LogEq (B : ℕ) (l l' : Array Claims.Msg) : Prop where
  size : l.size = l'.size
  get : ∀ k, B ≤ k → l[k]? = l'[k]?

theorem LogEq.of_size {B : ℕ} {l l' : Array Claims.Msg} (hl : l.size = l'.size) (hB : l.size ≤ B) : LogEq B l l' :=
  ⟨hl, fun k hk => by rw [Array.getElem?_eq_none (by omega), Array.getElem?_eq_none (by omega)]⟩

theorem LogEq.push {B : ℕ} {l l' : Array Claims.Msg} (h : LogEq B l l') {m m' : Claims.Msg}
    (hm : B ≤ l.size → m = m') : LogEq B (l.push m) (l'.push m') := by
  refine ⟨by simp [h.size], fun k hk => ?_⟩
  rw [Array.getElem?_push, Array.getElem?_push, ← h.size]
  by_cases he : k = l.size
  · rw [if_pos he, if_pos he, hm (by omega)]
  · rw [if_neg he, if_neg he]
    exact h.get k hk

/-- **Two runs in lockstep from body `B` on**: logs agreeing from `B` on; both refused, or both returned, with pendings
equal where `Q` holds of the log's length and values related by `R`. -/
structure Lk {α β : Type} (B : ℕ) (Q : ℕ → Prop) (R : α → β → Array Claims.Msg → Prop)
    (x : Except String (α × Transcript) × Array Claims.Msg) (y : Except String (β × Transcript) × Array Claims.Msg) :
    Prop where
  log : LogEq B x.2 y.2
  res : (∃ e e', x.1 = .error e ∧ y.1 = .error e') ∨
    ∃ a b t t', x.1 = .ok (a, t) ∧ y.1 = .ok (b, t') ∧ (Q x.2.size → t.pending = t'.pending) ∧ R a b x.2

variable {α β γ δ : Type} {B : ℕ} {Q Q' : ℕ → Prop}

/-- **A bind in lockstep**, the continuations from every pair of returns. -/
theorem lk_bind {R : α → β → Array Claims.Msg → Prop} {S : γ → δ → Array Claims.Msg → Prop} {x : TMW PS α}
    {y : TMW PS β} {f : α → TMW PS γ} {g : β → TMW PS δ} {t t' : Transcript} {l l' : Array Claims.Msg}
    (hxy : Lk B Q R ((x.run t).go l) ((y.run t').go l'))
    (hfg : ∀ a b t₁ t₁' l₁ l₁', LogEq B l₁ l₁' → (Q l₁.size → t₁.pending = t₁'.pending) → R a b l₁ →
      Lk B Q' S (((f a).run t₁).go l₁) (((g b).run t₁').go l₁')) :
    Lk B Q' S (((x >>= f).run t).go l) (((y >>= g).run t').go l') := by
  obtain ⟨hL, h⟩ := hxy
  rcases hx : (x.run t).go l with ⟨rx, Lx⟩
  rcases hy : (y.run t').go l' with ⟨ry, Ly⟩
  rw [hx, hy] at hL h
  dsimp only at hL h
  rcases h with ⟨e, e', rfl, rfl⟩ | ⟨a, b, t₁, t₁', rfl, rfl, hp, hR⟩
  · rw [bind_go_err hx, bind_go_err hy]
    exact ⟨hL, .inl ⟨e, e', rfl, rfl⟩⟩
  · rw [bind_go hx, bind_go hy]
    exact hfg a b t₁ t₁' Lx Ly hL hp hR

theorem lk_pure {R : α → β → Array Claims.Msg → Prop} {a : α} {b : β} {t t' : Transcript} {l l' : Array Claims.Msg}
    (hL : LogEq B l l') (hp : Q l.size → t.pending = t'.pending) (hR : R a b l) :
    Lk B Q R (((pure a : TMW PS α).run t).go l) (((pure b : TMW PS β).run t').go l') :=
  ⟨hL, .inr ⟨a, b, t, t', rfl, rfl, hp, hR⟩⟩

/-- **An `LP` bind in lockstep.** -/
theorem lk_lbind {R : α × LigState → β × LigState → Array Claims.Msg → Prop}
    {S : γ × LigState → δ × LigState → Array Claims.Msg → Prop} {x : LP PS α} {y : LP PS β} {f : α → LP PS γ}
    {g : β → LP PS δ} {s s' : LigState} {t t' : Transcript} {l l' : Array Claims.Msg}
    (hxy : Lk B Q R (((x.run s).run t).go l) (((y.run s').run t').go l'))
    (hfg : ∀ a s₁ b s₁' t₁ t₁' l₁ l₁', LogEq B l₁ l₁' → (Q l₁.size → t₁.pending = t₁'.pending) →
      R (a, s₁) (b, s₁') l₁ → Lk B Q' S ((((f a).run s₁).run t₁).go l₁) ((((g b).run s₁').run t₁').go l₁')) :
    Lk B Q' S ((((x >>= f).run s).run t).go l) ((((y >>= g).run s').run t').go l') := by
  rw [StateT.run_bind, StateT.run_bind]
  exact lk_bind (f := fun p => (f p.1).run p.2) (g := fun p => (g p.1).run p.2) hxy
    fun ⟨a, s₁⟩ ⟨b, s₁'⟩ t₁ t₁' l₁ l₁' hL hp hR => hfg a s₁ b s₁' t₁ t₁' l₁ l₁' hL hp hR

theorem lk_lpure {S : α × LigState → β × LigState → Array Claims.Msg → Prop} {a : α} {b : β} {s s' : LigState}
    {t t' : Transcript} {l l' : Array Claims.Msg} (hL : LogEq B l l') (hp : Q l.size → t.pending = t'.pending)
    (hS : S (a, s) (b, s') l) :
    Lk B Q S ((((pure a : LP PS α).run s).run t).go l) ((((pure b : LP PS β).run s').run t').go l') :=
  ⟨hL, .inr ⟨(a, s), (b, s'), t, t', rfl, rfl, hp, hS⟩⟩

/-! ## Absorbs -/

/-- A transcript action that only appends to the pending bytes, by `P`. -/
def PendOnly (x : TM Unit) (P : ByteArray → ByteArray) : Prop :=
  ∀ t : Transcript, x t = .ok ((), { t with pending := P t.pending })

theorem pend_label (s : String) : PendOnly (label s) (pLabel s) := fun _ => rfl
theorem pend_absorbF (v : F128) : PendOnly (absorbF v) (pF v) := fun _ => rfl
theorem pend_absorbSlice (vs : Array F128) : PendOnly (absorbSlice vs) (pSlice vs) := fun _ => rfl
theorem pend_absorbK (v : F256) : PendOnly (absorbK v) (pSlice #[v.c0, v.c1]) := fun _ => rfl
theorem pend_absorbBytes (b : ByteArray) : PendOnly (absorbBytes b) (pBytes b) := fun _ => rfl
theorem pend_pow (n : UInt64) : PendOnly (pow n) (pBytes (le64 n)) := fun _ => rfl

variable {S : γ → δ → Array Claims.Msg → Prop}

/-- **Two absorbs** of the transcript's monad. -/
theorem lk_tabs {x x' : TM Unit} {P P' : ByteArray → ByteArray} (hx : PendOnly x P) (hx' : PendOnly x' P')
    {f : Unit → TMW PS γ} {g : Unit → TMW PS δ} {t t' : Transcript} {l l' : Array Claims.Msg}
    (k : Lk B Q S (((f ()).run { t with pending := P t.pending }).go l)
      (((g ()).run { t' with pending := P' t'.pending }).go l')) :
    Lk B Q S (((liftTM x >>= f).run t).go l) (((liftTM x' >>= g).run t').go l') := by
  rw [bind_go (liftTM_ok (hx t) l), bind_go (liftTM_ok (hx' t') l')]
  exact k

variable {S : γ × LigState → δ × LigState → Array Claims.Msg → Prop}

/-- **Two absorbs** of the opening's monad. -/
theorem lk_labsorb {x x' : TM Unit} {P P' : ByteArray → ByteArray} (hx : PendOnly x P) (hx' : PendOnly x' P')
    {f : Unit → LP PS γ} {g : Unit → LP PS δ} {s s' : LigState} {t t' : Transcript} {l l' : Array Claims.Msg}
    (k : Lk B Q S ((((f ()).run s).run { t with pending := P t.pending }).go l)
      ((((g ()).run s').run { t' with pending := P' t'.pending }).go l')) :
    Lk B Q S ((((liftT (liftTM x) >>= f).run s).run t).go l) ((((liftT (liftTM x') >>= g).run s').run t').go l') := by
  rw [liftT_bind_run, liftT_bind_run, bind_go (liftTM_ok (hx t) l), bind_go (liftTM_ok (hx' t') l')]
  exact k

theorem lk_lF {v v' : F128} {f : Unit → LP PS γ} {g : Unit → LP PS δ} {s s' : LigState} {t t' : Transcript}
    {l l' : Array Claims.Msg}
    (k : ∀ t₁ t₁' : Transcript, (v = v' → t.pending = t'.pending → t₁.pending = t₁'.pending) →
      Lk B Q S ((((f ()).run s).run t₁).go l) ((((g ()).run s').run t₁').go l')) :
    Lk B Q S ((((liftT (liftTM (absorbF v)) >>= f).run s).run t).go l)
      ((((liftT (liftTM (absorbF v')) >>= g).run s').run t').go l') :=
  lk_labsorb (pend_absorbF v) (pend_absorbF v') (k _ _ fun hv hp => by
    subst hv
    exact congrArg (pF v) hp)

theorem lk_lK {v v' : F256} {f : Unit → LP PS γ} {g : Unit → LP PS δ} {s s' : LigState} {t t' : Transcript}
    {l l' : Array Claims.Msg}
    (k : ∀ t₁ t₁' : Transcript, (v = v' → t.pending = t'.pending → t₁.pending = t₁'.pending) →
      Lk B Q S ((((f ()).run s).run t₁).go l) ((((g ()).run s').run t₁').go l')) :
    Lk B Q S ((((liftT (liftTM (absorbK v)) >>= f).run s).run t).go l)
      ((((liftT (liftTM (absorbK v')) >>= g).run s').run t').go l') :=
  lk_labsorb (pend_absorbK v) (pend_absorbK v') (k _ _ fun hv hp => by
    subst hv
    exact congrArg (pSlice #[v.c0, v.c1]) hp)

theorem lk_lSl {vs vs' : Array F128} {f : Unit → LP PS γ} {g : Unit → LP PS δ} {s s' : LigState}
    {t t' : Transcript} {l l' : Array Claims.Msg}
    (k : ∀ t₁ t₁' : Transcript, (vs = vs' → t.pending = t'.pending → t₁.pending = t₁'.pending) →
      Lk B Q S ((((f ()).run s).run t₁).go l) ((((g ()).run s').run t₁').go l')) :
    Lk B Q S ((((liftT (liftTM (absorbSlice vs)) >>= f).run s).run t).go l)
      ((((liftT (liftTM (absorbSlice vs')) >>= g).run s').run t').go l') :=
  lk_labsorb (pend_absorbSlice vs) (pend_absorbSlice vs') (k _ _ fun hv hp => by
    subst hv
    exact congrArg (pSlice vs) hp)

theorem lk_lBy {b b' : ByteArray} {f : Unit → LP PS γ} {g : Unit → LP PS δ} {s s' : LigState} {t t' : Transcript}
    {l l' : Array Claims.Msg}
    (k : ∀ t₁ t₁' : Transcript, (b = b' → t.pending = t'.pending → t₁.pending = t₁'.pending) →
      Lk B Q S ((((f ()).run s).run t₁).go l) ((((g ()).run s').run t₁').go l')) :
    Lk B Q S ((((liftT (liftTM (absorbBytes b)) >>= f).run s).run t).go l)
      ((((liftT (liftTM (absorbBytes b')) >>= g).run s').run t').go l') :=
  lk_labsorb (pend_absorbBytes b) (pend_absorbBytes b') (k _ _ fun hv hp => by
    subst hv
    exact congrArg (pBytes b) hp)

theorem lk_lLab (str : String) {f : Unit → LP PS γ} {g : Unit → LP PS δ} {s s' : LigState} {t t' : Transcript}
    {l l' : Array Claims.Msg}
    (k : ∀ t₁ t₁' : Transcript, (t.pending = t'.pending → t₁.pending = t₁'.pending) →
      Lk B Q S ((((f ()).run s).run t₁).go l) ((((g ()).run s').run t₁').go l')) :
    Lk B Q S ((((liftT (liftTM (label str)) >>= f).run s).run t).go l)
      ((((liftT (liftTM (label str)) >>= g).run s').run t').go l') :=
  lk_labsorb (pend_label str) (pend_label str) (k _ _ fun hp => congrArg (pLabel str) hp)

/-- **Two messages sent**: each state records its own; the pendings agree where the messages and the pendings did. -/
theorem lk_send {m m' : F256 × F256} {f : Unit → LP PS γ} {g : Unit → LP PS δ} {s s' : LigState}
    {t t' : Transcript} {l l' : Array Claims.Msg}
    (k : ∀ t₁ t₁' : Transcript, (m = m' → t.pending = t'.pending → t₁.pending = t₁'.pending) →
      Lk B Q S ((((f ()).run { s with msgs := s.msgs.push m }).run t₁).go l)
        ((((g ()).run { s' with msgs := s'.msgs.push m' }).run t₁').go l')) :
    Lk B Q S ((((sendMsg m >>= f).run s).run t).go l) ((((sendMsg m' >>= g).run s').run t').go l') := by
  rw [sendMsg_bind_run, sendMsg_bind_run]
  refine lk_lK fun t₁ t₁' h₁ => ?_
  refine lk_lK fun t₂ t₂' h₂ => ?_
  exact k t₂ t₂' fun hm hp => by
    subst hm
    exact h₂ rfl (h₁ rfl hp)

/-- **Two grinding nonces**: the tables and weights untouched. -/
theorem lk_powP {nonce nonce' : ℕ → ℕ → UInt64} {kind : ℕ} {f : Unit → LP PS γ} {g : Unit → LP PS δ}
    {s s' : LigState} {t t' : Transcript} {l l' : Array Claims.Msg}
    (k : ∀ s₁ s₁' : LigState, s₁.tab = s.tab → s₁.w = s.w → s₁'.tab = s'.tab → s₁'.w = s'.w →
      ∀ t₁ t₁' : Transcript, (t.pending = t'.pending → t₁.pending = t₁'.pending) →
      Lk B Q S ((((f ()).run s₁).run t₁).go l) ((((g ()).run s₁').run t₁').go l')) :
    Lk B Q S ((((powP nonce kind >>= f).run s).run t).go l) ((((powP nonce' kind >>= g).run s').run t').go l') := by
  simp only [powP, bind_assoc, get_bind_run, set_bind_run]
  refine lk_labsorb (pend_pow 0) (pend_pow 0) (k _ _ ?_ ?_ ?_ ?_ _ _ fun hp => congrArg (pBytes (le64 0)) hp) <;>
    rcases kind with _ | _ | _ <;> rfl

/-! ## Squeezes -/

/-- **Two squeezes of `k ≠ 0` coins** at the same round: both refused past round `N`, else the same coins and both
pendings empty; the bodies agree from `B` on where the pendings did. -/
theorem lk_tsq {S : γ → δ → Array Claims.Msg → Prop} (coin : ℕ → ℕ → F128) (N : ℕ) (kind : UInt8) {k : ℕ}
    (hk : k ≠ 0) {f : Array F128 → TMW PS γ} {g : Array F128 → TMW PS δ} {t t' : Transcript}
    {l l' : Array Claims.Msg} (hL : LogEq B l l') (hp : B ≤ l.size → t.pending = t'.pending)
    (kont : l.size < N → ∀ t₁ t₁' : Transcript, t₁.pending = t₁'.pending → ∀ l₁ l₁' : Array Claims.Msg,
      LogEq B l₁ l₁' → l₁.size = l.size + 1 →
      Lk B Q S (((f ((Array.range k).map (coin l.size))).run t₁).go l₁)
        (((g ((Array.range k).map (coin l.size))).run t₁').go l₁')) :
    Lk B Q S (((squeezeP (asksCut coin N) kind k >>= f).run t).go l)
      (((squeezeP (asksCut coin N) kind k >>= g).run t').go l') := by
  have hL₁ := hL.push (m := Claims.toMsg (framed t kind k)) (m' := Claims.toMsg (framed t' kind k)) fun h => by
    unfold framed
    rw [hp h]
  by_cases hn : N ≤ l.size
  · rw [bind_go_err (squeezeP_cut_err coin kind hk t hn),
      bind_go_err (squeezeP_cut_err coin kind hk t' (by rw [← hL.size]; exact hn))]
    exact ⟨hL₁, .inl ⟨_, _, rfl, rfl⟩⟩
  · rw [bind_go (squeezeP_cut_ok coin kind hk t (by omega)),
      bind_go (squeezeP_cut_ok coin kind hk t' (by rw [← hL.size]; omega)), ← hL.size]
    exact kont (by omega) (afterSq t kind k (coin l.size)) (afterSq t' kind k (coin l.size)) rfl _ _ hL₁ (by simp)

theorem lk_lsq (coin : ℕ → ℕ → F128) (N : ℕ) (kind : UInt8) {k : ℕ} (hk : k ≠ 0)
    {f : Array F128 → LP PS γ} {g : Array F128 → LP PS δ} {s s' : LigState} {t t' : Transcript}
    {l l' : Array Claims.Msg} (hL : LogEq B l l') (hp : B ≤ l.size → t.pending = t'.pending)
    (kont : l.size < N → ∀ t₁ t₁' : Transcript, t₁.pending = t₁'.pending → ∀ l₁ l₁' : Array Claims.Msg,
      LogEq B l₁ l₁' → l₁.size = l.size + 1 →
      Lk B Q S ((((f ((Array.range k).map (coin l.size))).run s).run t₁).go l₁)
        ((((g ((Array.range k).map (coin l.size))).run s').run t₁').go l₁')) :
    Lk B Q S ((((liftT (squeezeP (asksCut coin N) kind k) >>= f).run s).run t).go l)
      ((((liftT (squeezeP (asksCut coin N) kind k) >>= g).run s').run t').go l') := by
  rw [liftT_bind_run, liftT_bind_run]
  exact lk_tsq coin N kind hk hL hp kont

theorem lk_lsampleVec (coin : ℕ → ℕ → F128) (N : ℕ) {k : ℕ} (hk : k ≠ 0)
    {f : Array F128 → LP PS γ} {g : Array F128 → LP PS δ} {s s' : LigState} {t t' : Transcript}
    {l l' : Array Claims.Msg} (hL : LogEq B l l') (hp : B ≤ l.size → t.pending = t'.pending)
    (kont : l.size < N → ∀ t₁ t₁' : Transcript, t₁.pending = t₁'.pending → ∀ l₁ l₁' : Array Claims.Msg,
      LogEq B l₁ l₁' → l₁.size = l.size + 1 →
      Lk B Q S ((((f ((Array.range k).map (coin l.size))).run s).run t₁).go l₁)
        ((((g ((Array.range k).map (coin l.size))).run s').run t₁').go l₁')) :
    Lk B Q S ((((liftT (sampleVecP (asksCut coin N) k) >>= f).run s).run t).go l)
      ((((liftT (sampleVecP (asksCut coin N) k) >>= g).run s').run t').go l') := by
  unfold sampleVecP
  exact lk_lsq coin N 2 hk hL hp kont

theorem lk_lsampleP (coin : ℕ → ℕ → F128) (N : ℕ) {f : F128 → LP PS γ} {g : F128 → LP PS δ} {s s' : LigState}
    {t t' : Transcript} {l l' : Array Claims.Msg} (hL : LogEq B l l') (hp : B ≤ l.size → t.pending = t'.pending)
    (kont : l.size < N → ∀ t₁ t₁' : Transcript, t₁.pending = t₁'.pending → ∀ l₁ l₁' : Array Claims.Msg,
      LogEq B l₁ l₁' → l₁.size = l.size + 1 →
      Lk B Q S ((((f (coin l.size 0)).run s).run t₁).go l₁) ((((g (coin l.size 0)).run s').run t₁').go l₁')) :
    Lk B Q S ((((liftT (sampleP (asksCut coin N)) >>= f).run s).run t).go l)
      ((((liftT (sampleP (asksCut coin N)) >>= g).run s').run t').go l') := by
  rw [liftT_bind_run, liftT_bind_run]
  unfold sampleP
  simp only [bind_assoc, pure_bind]
  refine lk_tsq coin N 1 one_ne_zero hL hp fun hn t₁ t₁' ht l₁ l₁' hL₁ hs => ?_
  have h0 : ((Array.range 1).map (coin l.size)).getD 0 0 = coin l.size 0 := by simp
  rw [h0]
  exact kont hn t₁ t₁' ht l₁ l₁' hL₁ hs

theorem lk_lsampleKP (coin : ℕ → ℕ → F128) (N : ℕ) {f : F256 → LP PS γ} {g : F256 → LP PS δ} {s s' : LigState}
    {t t' : Transcript} {l l' : Array Claims.Msg} (hL : LogEq B l l') (hp : B ≤ l.size → t.pending = t'.pending)
    (kont : l.size < N → ∀ t₁ t₁' : Transcript, t₁.pending = t₁'.pending → ∀ l₁ l₁' : Array Claims.Msg,
      LogEq B l₁ l₁' → l₁.size = l.size + 1 →
      Lk B Q S ((((f ⟨coin l.size 0, coin l.size 1⟩).run s).run t₁).go l₁)
        ((((g ⟨coin l.size 0, coin l.size 1⟩).run s').run t₁').go l₁')) :
    Lk B Q S ((((liftT (sampleKP (asksCut coin N)) >>= f).run s).run t).go l)
      ((((liftT (sampleKP (asksCut coin N)) >>= g).run s').run t').go l') := by
  rw [liftT_bind_run, liftT_bind_run]
  unfold sampleKP
  simp only [bind_assoc, pure_bind]
  refine lk_tsq coin N 2 two_ne_zero hL hp fun hn t₁ t₁' ht l₁ l₁' hL₁ hs => ?_
  have h0 : ((Array.range 2).map (coin l.size)).getD 0 0 = coin l.size 0 := by simp
  have h1 : ((Array.range 2).map (coin l.size)).getD 1 0 = coin l.size 1 := by simp
  rw [h0, h1]
  exact kont hn t₁ t₁' ht l₁ l₁' hL₁ hs

/-- **Two squeezes of any count from equal pendings**: the same coins, equal pendings after, logs agreeing. -/
theorem lk_lsqAny (coin : ℕ → ℕ → F128) (N : ℕ) (kind : UInt8) (k : ℕ)
    {f : Array F128 → LP PS γ} {g : Array F128 → LP PS δ} {s s' : LigState} {t t' : Transcript}
    {l l' : Array Claims.Msg} (hL : LogEq B l l') (hp : t.pending = t'.pending)
    (kont : ∀ t₁ t₁' : Transcript, t₁.pending = t₁'.pending → ∀ l₁ l₁' : Array Claims.Msg, LogEq B l₁ l₁' →
      Lk B Q S ((((f ((Array.range k).map (coin l.size))).run s).run t₁).go l₁)
        ((((g ((Array.range k).map (coin l.size))).run s').run t₁').go l₁')) :
    Lk B Q S ((((liftT (squeezeP (asksCut coin N) kind k) >>= f).run s).run t).go l)
      ((((liftT (squeezeP (asksCut coin N) kind k) >>= g).run s').run t').go l') := by
  by_cases hk : k = 0
  · subst hk
    rw [liftT_bind_run, liftT_bind_run]
    have e : ((squeezeP (asksCut coin N) kind 0).run t).go l =
        (.ok (#[], { t with pending := framed t kind 0 }), l) := rfl
    have e' : ((squeezeP (asksCut coin N) kind 0).run t').go l' =
        (.ok (#[], { t' with pending := framed t' kind 0 }), l') := rfl
    rw [bind_go e, bind_go e']
    have h0 : (Array.range 0).map (coin l.size) = #[] := by simp
    have := kont { t with pending := framed t kind 0 } { t' with pending := framed t' kind 0 }
      (by show framed t kind 0 = framed t' kind 0; unfold framed; rw [hp]) l l' hL
    rwa [h0] at this
  · exact lk_lsq coin N kind hk hL (fun _ => hp) fun _ t₁ t₁' h₁ l₁ l₁' hL₁ _ => kont t₁ t₁' h₁ l₁ l₁' hL₁

theorem lk_lsampleVecAny (coin : ℕ → ℕ → F128) (N : ℕ) (k : ℕ)
    {f : Array F128 → LP PS γ} {g : Array F128 → LP PS δ} {s s' : LigState} {t t' : Transcript}
    {l l' : Array Claims.Msg} (hL : LogEq B l l') (hp : t.pending = t'.pending)
    (kont : ∀ t₁ t₁' : Transcript, t₁.pending = t₁'.pending → ∀ l₁ l₁' : Array Claims.Msg, LogEq B l₁ l₁' →
      Lk B Q S ((((f ((Array.range k).map (coin l.size))).run s).run t₁).go l₁)
        ((((g ((Array.range k).map (coin l.size))).run s').run t₁').go l₁')) :
    Lk B Q S ((((liftT (sampleVecP (asksCut coin N) k) >>= f).run s).run t).go l)
      ((((liftT (sampleVecP (asksCut coin N) k) >>= g).run s').run t').go l') := by
  unfold sampleVecP
  exact lk_lsqAny coin N 2 k hL hp kont

theorem lk_lsamplePAny (coin : ℕ → ℕ → F128) (N : ℕ) {f : F128 → LP PS γ} {g : F128 → LP PS δ}
    {s s' : LigState} {t t' : Transcript} {l l' : Array Claims.Msg} (hL : LogEq B l l') (hp : t.pending = t'.pending)
    (kont : ∀ t₁ t₁' : Transcript, t₁.pending = t₁'.pending → ∀ l₁ l₁' : Array Claims.Msg, LogEq B l₁ l₁' →
      Lk B Q S ((((f (coin l.size 0)).run s).run t₁).go l₁) ((((g (coin l.size 0)).run s').run t₁').go l₁')) :
    Lk B Q S ((((liftT (sampleP (asksCut coin N)) >>= f).run s).run t).go l)
      ((((liftT (sampleP (asksCut coin N)) >>= g).run s').run t').go l') :=
  lk_lsampleP coin N hL (fun _ => hp) fun _ t₁ t₁' h₁ l₁ l₁' hL₁ _ => kont t₁ t₁' h₁ l₁ l₁' hL₁

theorem lk_lsampleKPAny (coin : ℕ → ℕ → F128) (N : ℕ) {f : F256 → LP PS γ} {g : F256 → LP PS δ}
    {s s' : LigState} {t t' : Transcript} {l l' : Array Claims.Msg} (hL : LogEq B l l') (hp : t.pending = t'.pending)
    (kont : ∀ t₁ t₁' : Transcript, t₁.pending = t₁'.pending → ∀ l₁ l₁' : Array Claims.Msg, LogEq B l₁ l₁' →
      Lk B Q S ((((f ⟨coin l.size 0, coin l.size 1⟩).run s).run t₁).go l₁)
        ((((g ⟨coin l.size 0, coin l.size 1⟩).run s').run t₁').go l₁')) :
    Lk B Q S ((((liftT (sampleKP (asksCut coin N)) >>= f).run s).run t).go l)
      ((((liftT (sampleKP (asksCut coin N)) >>= g).run s').run t').go l') :=
  lk_lsampleKP coin N hL (fun _ => hp) fun _ t₁ t₁' h₁ l₁ l₁' hL₁ _ => kont t₁ t₁' h₁ l₁ l₁' hL₁

/-! ## Commitments and loops -/

theorem lift_quiet_go {m : PS α} {r : Except String α} (h : ∀ l, m.go l = (r, l)) (t : Transcript)
    (l : Array Claims.Msg) : ((StateT.lift m : TMW PS α).run t).go l = ((fun a => (a, t)) <$> r, l) := by
  rw [StateT.run_lift, go_bind, h l]
  cases r <;> rfl

/-- A commitment reads its salt function only on its positions, cut to the scheme's width. -/
theorem commitV_congr (hs : Hashes V) (ms : MerkleScheme) (lv : Level) (g : Array F128) {s s' : ℕ → ByteArray}
    (h : ∀ q < 2 ^ lv.d, fitBytes ms.saltBytes (s q) = fitBytes ms.saltBytes (s' q)) :
    (commitP hs ms lv g s : V Commit) = commitP hs ms lv g s' := by
  have hsalts : (Array.range (2 ^ lv.d)).map (fun q => fitBytes ms.saltBytes (s q)) =
      (Array.range (2 ^ lv.d)).map (fun q => fitBytes ms.saltBytes (s' q)) := by
    apply Array.ext (by simp)
    intro i h1 _
    simp only [Array.getElem_map, Array.getElem_range]
    exact h i (by simpa using h1)
  simp only [commitP, hsalts]

/-- **Two equal commitments**: they send nothing and return the same. -/
theorem lk_commit (H : ByteArray → ByteArray) {ms ms' : MerkleScheme} {lv lv' : Level} {g g' : Array F128}
    {salt salt' : ℕ → ByteArray}
    (hc : (commitP (Hashes.ofSha512 H) ms lv g salt : V Commit) = commitP (Hashes.ofSha512 H) ms' lv' g' salt')
    {f : Commit → LP PS γ} {f' : Commit → LP PS δ} {s s' : LigState} {t t' : Transcript} {l l' : Array Claims.Msg}
    (hL : LogEq B l l') (k : ∀ cm, Lk B Q S ((((f cm).run s).run t).go l) ((((f' cm).run s').run t').go l')) :
    Lk B Q S ((((liftT (StateT.lift (commitP (Hashes.ofSha512 H) ms lv g salt)) >>= f).run s).run t).go l)
      ((((liftT (StateT.lift (commitP (Hashes.ofSha512 H) ms' lv' g' salt')) >>= f').run s').run t').go l') := by
  rw [liftT_bind_run, liftT_bind_run, StateT.run_bind, StateT.run_bind, go_bind, go_bind,
    lift_quiet_go (commitP_quiet H ms lv g salt) t l, lift_quiet_go (commitP_quiet H ms' lv' g' salt') t' l', ← hc]
  rcases (commitP (Hashes.ofSha512 H) ms lv g salt : V Commit) with e | cm
  · exact ⟨hL, .inl ⟨e, e, rfl, rfl⟩⟩
  · exact k cm

/-- **A loop in lockstep**: if step `i`, from values, states and logs related by `I i`, yields ones related by
`I (i + 1)`, the loops over `[0:n]` end related by `I n`. -/
theorem lk_forIn {σ τ : Type} (body : ℕ → σ → LP PS (ForInStep σ)) (body' : ℕ → τ → LP PS (ForInStep τ))
    (I : ℕ → σ × LigState → τ × LigState → Array Claims.Msg → Prop) (n : ℕ)
    (hb : ∀ i < n, ∀ (u : σ) (s : LigState) (u' : τ) (s' : LigState) (t t' : Transcript) (l l' : Array Claims.Msg),
      LogEq B l l' → (Q l.size → t.pending = t'.pending) → I i (u, s) (u', s') l →
      Lk B Q (fun r r' l₁ => ∃ v v', r.1 = .yield v ∧ r'.1 = .yield v' ∧ I (i + 1) (v, r.2) (v', r'.2) l₁)
        ((((body i u).run s).run t).go l) ((((body' i u').run s').run t').go l'))
    {u : σ} {u' : τ} {s s' : LigState} {t t' : Transcript} {l l' : Array Claims.Msg} (hL : LogEq B l l')
    (hp : Q l.size → t.pending = t'.pending) (h₀ : I 0 (u, s) (u', s') l) :
    Lk B Q (fun r r' l₁ => I n r r' l₁)
      ((((forIn [:n] u body).run s).run t).go l) ((((forIn [:n] u' body').run s').run t').go l') := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (u : σ) (s : LigState) (u' : τ) (s' : LigState) (t t' : Transcript)
      (l l' : Array Claims.Msg), a + k = n → LogEq B l l' → (Q l.size → t.pending = t'.pending) →
      I a (u, s) (u', s') l →
      Lk B Q (fun r r' l₁ => I n r r' l₁)
        ((((forIn (List.range' a k) u body).run s).run t).go l)
        ((((forIn (List.range' a k) u' body').run s').run t').go l') from
    key n 0 u s u' s' t t' l l' (by omega) hL hp h₀
  intro k
  induction k with
  | zero =>
    intro a u s u' s' t t' l l' ha hL hp hI
    have e : a = n := by omega
    exact ⟨hL, .inr ⟨(u, s), (u', s'), t, t', rfl, rfl, hp, e ▸ hI⟩⟩
  | succ k ih =>
    intro a u s u' s' t t' l l' ha hL hp hI
    rw [List.range'_succ, List.forIn_cons, List.forIn_cons]
    refine lk_lbind (hb a (by omega) u s u' s' t t' l l' hL hp hI) fun r s₁ r' s₁' t₁ t₁' l₁ l₁' hL₁ hp₁ hr => ?_
    obtain ⟨v, v', hv, hv', hI₁⟩ := hr
    dsimp only at hv hv'
    subst hv hv'
    exact ih (a + 1) v s₁ v' s₁' t₁ t₁' l₁ l₁' (by omega) hL₁ hp₁ hI₁

/-- Absorbing a list of values one by one. -/
theorem absorbs_go (body : F128 → PUnit → LP PS (ForInStep PUnit))
    (hb : ∀ x u, body x u = liftT (liftTM (absorbF x)) >>= fun _ => pure (ForInStep.yield PUnit.unit))
    (xs : List F128) (s : LigState) :
    ∀ (t : Transcript) (l : Array Claims.Msg), (((forIn xs PUnit.unit body).run s).run t).go l =
      (.ok ((PUnit.unit, s), { t with pending := xs.foldl (fun P x => pF x P) t.pending }), l) := by
  induction xs with
  | nil => intro t l; rfl
  | cons x xs ih =>
    intro t l
    rw [List.forIn_cons, hb]
    simp only [bind_assoc, pure_bind]
    rw [liftT_bind_run, bind_go (liftTM_ok (pend_absorbF x t) l), ih]
    rfl

/-- **Two loops absorbing values one by one.** -/
theorem lk_absorbs (body body' : F128 → PUnit → LP PS (ForInStep PUnit))
    (hb : ∀ x u, body x u = liftT (liftTM (absorbF x)) >>= fun _ => pure (ForInStep.yield PUnit.unit))
    (hb' : ∀ x u, body' x u = liftT (liftTM (absorbF x)) >>= fun _ => pure (ForInStep.yield PUnit.unit))
    {xs xs' : Array F128} {f : PUnit → LP PS γ} {g : PUnit → LP PS δ} {s s' : LigState} {t t' : Transcript}
    {l l' : Array Claims.Msg}
    (k : ∀ t₁ t₁' : Transcript, (xs = xs' → t.pending = t'.pending → t₁.pending = t₁'.pending) →
      Lk B Q S ((((f PUnit.unit).run s).run t₁).go l) ((((g PUnit.unit).run s').run t₁').go l')) :
    Lk B Q S ((((forIn xs PUnit.unit body >>= f).run s).run t).go l)
      ((((forIn xs' PUnit.unit body' >>= g).run s').run t').go l') := by
  rw [← Array.forIn_toList, ← Array.forIn_toList, StateT.run_bind (forIn xs.toList PUnit.unit body),
    StateT.run_bind (forIn xs'.toList PUnit.unit body'),
    bind_go (absorbs_go body hb xs.toList s t l), bind_go (absorbs_go body' hb' xs'.toList s' t' l')]
  exact k _ _ fun hx hp => by
    subst hx
    show xs.toList.foldl (fun P x => pF x P) t.pending = xs.toList.foldl (fun P x => pF x P) t'.pending
    rw [hp]

/-! ## The opening's blocks from equal tables, weights and pendings -/

/-- **Two challenges** (no split) from equal tables and weights: the same coin, equal tables and weights after. -/
theorem lk_challenge (coin : ℕ → ℕ → F128) (N : ℕ) {f : F256 → LP PS γ} {g : F256 → LP PS δ}
    {s s' : LigState} {t t' : Transcript} {l l' : Array Claims.Msg} (hL : LogEq B l l')
    (hp : t.pending = t'.pending) (htab : s.tab = s'.tab) (hw : s.w = s'.w)
    (k : ∀ (r : F256) (s₁ s₁' : LigState) (t₁ t₁' : Transcript) (l₁ l₁' : Array Claims.Msg), LogEq B l₁ l₁' →
      t₁.pending = t₁'.pending → s₁.tab = s₁'.tab → s₁.w = s₁'.w →
      Lk B Q S ((((f r).run s₁).run t₁).go l₁) ((((g r).run s₁').run t₁').go l₁')) :
    Lk B Q S ((((challengeP (asksCut coin N) false >>= f).run s).run t).go l)
      ((((challengeP (asksCut coin N) false >>= g).run s').run t').go l') := by
  simp only [challengeP, bind_assoc, Bool.false_eq_true, ↓reduceIte, pure_bind]
  refine lk_lsampleKPAny coin N hL hp fun t₁ t₁' hp₁ l₁ l₁' hL₁ => ?_
  rw [modify_bind_run, modify_bind_run, get_bind_run, get_bind_run]
  refine lk_send fun t₂ t₂' hp₂ => ?_
  exact k _ _ _ _ _ _ _ hL₁ (hp₂ (by simp only [htab, hw]) hp₁) (by simp only [htab]) (by simp only [hw])

/-- **Two out-of-domain samples** of equal tables from equal tables, weights and pendings. -/
theorem lk_ood (coin : ℕ → ℕ → F128) (N : ℕ) (nonce nonce' : ℕ → ℕ → UInt64) {g g' : Array F128}
    (hg : g = g') {dim : ℕ} (hdim : dim ≠ 0) {f : Unit → LP PS γ} {f' : Unit → LP PS δ} {s s' : LigState}
    {t t' : Transcript} {l l' : Array Claims.Msg} (hL : LogEq B l l') (hp : t.pending = t'.pending)
    (htab : s.tab = s'.tab) (hw : s.w = s'.w)
    (k : ∀ (s₁ s₁' : LigState) (t₁ t₁' : Transcript) (l₁ l₁' : Array Claims.Msg), LogEq B l₁ l₁' →
      t₁.pending = t₁'.pending → s₁.tab = s₁'.tab → s₁.w = s₁'.w →
      Lk B Q S ((((f ()).run s₁).run t₁).go l₁) ((((f' ()).run s₁').run t₁').go l₁')) :
    Lk B Q S ((((oodIntroP (asksCut coin N) nonce g dim >>= f).run s).run t).go l)
      ((((oodIntroP (asksCut coin N) nonce' g' dim >>= f').run s').run t').go l') := by
  subst hg
  simp only [oodIntroP, bind_assoc]
  refine lk_lsampleVec coin N hdim hL (fun _ => hp) fun _ t₁ t₁' hp₁ l₁ l₁' hL₁ _ => ?_
  rw [modify_bind_run, modify_bind_run]
  refine lk_lF fun t₂ t₂' hp₂ => ?_
  refine lk_send fun t₃ t₃' hp₃ => ?_
  refine lk_powP fun s₄ s₄' h4t h4w h4t' h4w' t₄ t₄' hp₄ => ?_
  refine lk_lsamplePAny coin N hL₁ (hp₄ (hp₃ rfl (hp₂ rfl hp₁))) fun t₅ t₅' hp₅ l₅ l₅' hL₅ => ?_
  rw [modify_bind_run, modify_bind_run]
  exact k _ _ _ _ _ _ hL₅ hp₅ (by simp only [h4t, h4t', htab]) (by simp only [h4w, h4w', hw])

/-- **Two levels' queries** from equal tables, weights and pendings: the same consistency weight, the openings the same
where the commitments are. -/
theorem lk_query (coin : ℕ → ℕ → F128) (N : ℕ) (nonce nonce' : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level)
    (cm cm' : Commit) {f : LevelProof × Array F128 → LP PS γ} {f' : LevelProof × Array F128 → LP PS δ}
    {s s' : LigState} {t t' : Transcript} {l l' : Array Claims.Msg} (hL : LogEq B l l')
    (hp : t.pending = t'.pending) (htab : s.tab = s'.tab) (hw : s.w = s'.w)
    (k : ∀ (op op' : LevelProof) (v : Array F128) (s₁ s₁' : LigState) (t₁ t₁' : Transcript)
      (l₁ l₁' : Array Claims.Msg), (cm = cm' → op = op') → LogEq B l₁ l₁' → t₁.pending = t₁'.pending →
      s₁.tab = s₁'.tab → s₁.w = s₁'.w →
      Lk B Q S ((((f (op, v)).run s₁).run t₁).go l₁) ((((f' (op', v)).run s₁').run t₁').go l₁')) :
    Lk B Q S ((((queryP (asksCut coin N) nonce ms lv cm >>= f).run s).run t).go l)
      ((((queryP (asksCut coin N) nonce' ms lv cm' >>= f').run s').run t').go l') := by
  simp only [queryP, bind_assoc, pure_bind]
  refine lk_powP fun s₁ s₁' h1t h1w h1t' h1w' t₁ t₁' hp₁ => ?_
  refine lk_lsampleVecAny coin N lv.q hL (hp₁ hp) fun t₂ t₂' hp₂ l₂ l₂' hL₂ => ?_
  refine lk_powP fun s₃ s₃' h3t h3w h3t' h3w' t₃ t₃' hp₃ => ?_
  refine lk_lsampleVecAny coin N _ hL₂ (hp₃ hp₂) fun t₄ t₄' hp₄ l₄ l₄' hL₄ => ?_
  exact k _ _ _ _ _ _ _ _ _ (fun hc => by rw [hc]) hL₄ hp₄ (by rw [h3t, h3t', h1t, h1t', htab])
    (by rw [h3w, h3w', h1w, h1w', hw])

/-! ## The two openings -/

/-- **The cut's opening and the session's in lockstep from the restart on**: answered by `asksCut coin N` from logs of
one length `p₀`, where the source holds the session's `y₁'`, `ȳ` and levels ≥ 1's salts at the coins from round `p₀`
(`LigHyp`'s `y1`, `ybar`, `salt`), the two logs agree from body `p₀ + k₀ + 4` on, and both refuse or both return. -/
theorem ligeritoCut_lock (H : ByteArray → ByteArray) (coin : ℕ → ℕ → F128) (N : ℕ) (ms : MerkleScheme)
    (sch : Schedule) (T T' : F128) (cm0 cm0' : Commit) (t b : Array F128) (lay : Zk.PadLayout) (h : Array F128)
    (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (src : RepSrc) (tr tr' : Transcript) (l l' : Array Claims.Msg) (hl : l.size = l'.size) (hk0 : 1 ≤ sch.k0)
    (hn0 : sch.n ≠ 0) (hlev : 2 ≤ sch.levels.size)
    (hy1 : l.size + 3 + sch.k0 < N → Zk.rhoIndependent (rhoS coin l.size sch.k0) = true →
      src.y1 = (Array.range (tabS coin l.size t sch.k0).size).map fun c =>
        (tabS coin l.size t sch.k0)[c]! +
          (rhoS coin l.size sch.k0).1.smul
            ((Array.range (tabS coin l.size t sch.k0).size).map (extra (2 * rep)))[c]! +
          (rhoS coin l.size sch.k0).2.smul
            ((Array.range (tabS coin l.size t sch.k0).size).map (extra (2 * rep + 1)))[c]!)
    (hyb : l.size + 3 + sch.k0 < N → Zk.rhoIndependent (rhoS coin l.size sch.k0) = true →
      src.ybar = (Array.range tPad).map fun i =>
        (List.range (2 ^ (sch.levels.getD 0 default).lanes)).foldl
          (fun acc e => acc + ((eqTableK ((Array.range sch.k0).map (chS coin l.size)))[e]!).smul (lanePad e i))
          ((rhoS coin l.size sch.k0).1.smul (lanePad (2 ^ (sch.levels.getD 0 default).lanes + 2 * rep) i) +
            (rhoS coin l.size sch.k0).2.smul (lanePad (2 ^ (sch.levels.getD 0 default).lanes + 2 * rep + 1) i)))
    (hsalt : l.size + 3 + sch.k0 < N → Zk.rhoIndependent (rhoS coin l.size sch.k0) = true →
      ∀ i, 1 ≤ i → i < sch.levels.size → ∀ q < 2 ^ (sch.levels.getD i default).d,
        fitBytes ms.saltBytes (src.hiSalt i q) = fitBytes ms.saltBytes (salt i q)) :
    Lk (l.size + sch.k0 + 4) (fun _ => True) (fun _ _ _ => True)
      (((ligeritoCut (asksCut coin N) (Hashes.ofSha512 H) ms sch T cm0 b lay src).run tr).go l)
      (((ligeritoPZk (asksCut coin N) (Hashes.ofSha512 H) ms sch T' cm0' t b lay h rep tPad extra lanePad salt
        nonce).run tr').go l') := by
  have hL0 : LogEq (l.size + sch.k0 + 4) l l' := LogEq.of_size hl (by omega)
  unfold ligeritoCut ligeritoPZk
  dsimp only
  refine lk_tabs (pend_label _) (pend_label _) ?_
  refine lk_tabs (pend_absorbF _) (pend_absorbF _) ?_
  refine lk_tabs (pend_absorbBytes _) (pend_absorbBytes _) ?_
  refine lk_tabs (pend_label _) (pend_label _) ?_
  refine lk_bind (Q := fun _ => True) (R := fun _ _ _ => True) ?prog
    fun a a' t₁ t₁' l₁ l₁' hL₁ hp₁ _ => ?tail
  case tail =>
    obtain ⟨⟨_, _, _, _, _, _, _, _, _⟩, _⟩ := a
    obtain ⟨⟨_, _, _, _, _, _, _, _, _⟩, _⟩ := a'
    exact ⟨hL₁, .inr ⟨_, _, _, _, rfl, rfl, fun _ => hp₁ trivial, trivial⟩⟩
  case prog =>
    -- level 0's out-of-domain point, its value, the claim nonce and `b₀`
    refine lk_lsampleVec coin N hn0 hL0 (fun h => absurd h (by omega)) fun _ t₁ t₁' _ l₁ l₁' hL₁ hs₁ => ?_
    rw [modify_bind_run, modify_bind_run]
    refine lk_lF fun t₂ t₂' _ => ?_
    refine lk_powP fun s₃ s₃' h3t h3w h3t' h3w' t₃ t₃' _ => ?_
    refine lk_lsampleP coin N hL₁ (fun h => absurd h (by omega)) fun _ t₄ t₄' _ l₂ l₂' hL₂ hs₂ => ?_
    rw [modify_bind_run, modify_bind_run, get_bind_run]
    refine lk_send fun t₅ t₅' _ => ?_
    -- the lane rounds and the restart
    refine lk_lbind (Q := fun k => l.size + sch.k0 + 4 ≤ k)
      (lk_forIn _ _ (fun j (p p' : (Array F256 × (F256 × F256) × Zk.Level0Msgs) × LigState) (L : Array Claims.Msg) =>
        (j < sch.k0 → L.size = l.size + 2 + j ∧ p.2.w = wS coin l.size b sch.n j ∧
          p'.2.w = wS coin l.size b sch.n j ∧ p'.2.tab = tabS coin l.size t j ∧
          p'.1.1 = (Array.range j).map (chS coin l.size)) ∧
        (sch.k0 ≤ j → L.size = l.size + 4 + sch.k0 ∧ p.2.tab = p'.2.tab ∧ p.2.w = p'.2.w ∧
          l.size + 3 + sch.k0 < N ∧ Zk.rhoIndependent (rhoS coin l.size sch.k0) = true))
        sch.k0 ?step hL₂ (fun h => absurd h (by omega)) ?init)
      fun r s₁ r' s₁' t₆ t₆' l₆ l₆' hL₆ hp₆ hI => ?rest
    case init =>
      refine ⟨fun _ => ⟨by omega, ?_, ?_, ?_, by simp⟩, fun h => absurd h (by omega)⟩
      · show addScaled s₃.w _ _ = _
        rw [h3w, hs₁]
        rfl
      · show addScaled s₃'.w _ _ = _
        rw [h3w', hs₁]
        rfl
      · rw [h3t']
        rfl
    case step =>
      intro j hj u s u' s' tj tj' lj lj' hLj _ ⟨hlt, _⟩
      obtain ⟨hlj, hw, hw', htab', hch⟩ := hlt hj
      rcases u with ⟨initCh, rho, zl⟩
      rcases u' with ⟨initCh', rho', zl'⟩
      dsimp only at hch ⊢
      subst hch
      refine lk_lsampleKP coin N hLj (fun h => absurd h (by omega)) fun _ tr₁ tr₁' _ lr lr' hLr hsr => ?_
      rw [modify_bind_run, modify_bind_run, hw, hw', htab', hlj]
      by_cases hk : j + 1 = sch.k0
      · have hb : (j + 1 == sch.k0) = true := by simpa using hk
        rw [ite_eq_left hb, ite_eq_left hb, get_bind_run, get_bind_run]
        refine lk_lK fun t₁ t₁' _ => ?_
        refine lk_lK fun t₂ t₂' _ => ?_
        refine lk_lsampleKP coin N hLr (fun h => absurd h (by omega)) fun _ t₃ t₃' _ l₃ l₃' hL₃ hs₃ => ?_
        refine lk_lsampleKP coin N hL₃ (fun h => absurd h (by omega)) fun h4 t₄ t₄' hp₄ l₄ l₄' hL₄ hs₄ => ?_
        have e3 : lr.size = l.size + 2 + sch.k0 := by omega
        have e4 : l₃.size = l.size + 3 + sch.k0 := by omega
        rw [e3, e4]
        by_cases hρ : Zk.rhoIndependent (rhoS coin l.size sch.k0) = true
        · have hρu : Zk.rhoIndependent (⟨coin (l.size + 2 + sch.k0) 0, coin (l.size + 2 + sch.k0) 1⟩,
              ⟨coin (l.size + 3 + sch.k0) 0, coin (l.size + 3 + sch.k0) 1⟩) = true := hρ
          have hc : ¬ (!Zk.rhoIndependent (⟨coin (l.size + 2 + sch.k0) 0, coin (l.size + 2 + sch.k0) 1⟩,
              ⟨coin (l.size + 3 + sch.k0) 0, coin (l.size + 3 + sch.k0) 1⟩)) = true := by simp [hρu]
          rw [ite_eq_right hc, ite_eq_right hc]
          have hnl : l.size + 3 + sch.k0 < N := by omega
          have htabK : foldK ⟨coin (l.size + 2 + j) 0, coin (l.size + 2 + j) 1⟩ (tabS coin l.size t j) =
              tabS coin l.size t sch.k0 := by
            rw [← hk]
            rfl
          have hpush : ((Array.range j).map (chS coin l.size)).push ⟨coin (l.size + 2 + j) 0, coin (l.size + 2 + j) 1⟩ =
              (Array.range sch.k0).map (chS coin l.size) := by
            rw [← hk]
            simp [Array.range_succ, chS]
          rw [hy1 hnl hρ, hyb hnl hρ, htabK, hpush]
          dsimp only [rhoS]
          refine lk_lLab _ fun t₅ t₅' hp₅ => ?_
          refine lk_lK fun t₆ t₆' hp₆ => ?_
          refine lk_lSl fun t₇ t₇' hp₇ => ?_
          rw [modify_bind_run, modify_bind_run, get_bind_run, get_bind_run]
          refine lk_send fun t₈ t₈' hp₈ => ?_
          exact lk_lpure hL₄ (fun _ => hp₈ rfl (hp₇ rfl (hp₆ rfl (hp₅ hp₄))))
            ⟨_, _, rfl, rfl, fun h => absurd h (by omega), fun _ => ⟨by omega, rfl, rfl, hnl, hρ⟩⟩
        · have hρu : ¬ Zk.rhoIndependent (⟨coin (l.size + 2 + sch.k0) 0, coin (l.size + 2 + sch.k0) 1⟩,
              ⟨coin (l.size + 3 + sch.k0) 0, coin (l.size + 3 + sch.k0) 1⟩) = true := hρ
          have hc : (!Zk.rhoIndependent (⟨coin (l.size + 2 + sch.k0) 0, coin (l.size + 2 + sch.k0) 1⟩,
              ⟨coin (l.size + 3 + sch.k0) 0, coin (l.size + 3 + sch.k0) 1⟩)) = true := by simpa using hρu
          rw [ite_eq_left hc, ite_eq_left hc]
          exact ⟨hL₄, .inl ⟨_, _, rfl, rfl⟩⟩
      · have hb : ¬ (j + 1 == sch.k0) = true := by simpa using hk
        rw [ite_eq_right hb, ite_eq_right hb, get_bind_run]
        refine lk_send fun t₁ t₁' _ => ?_
        refine lk_lpure hLr (fun h => absurd h (by omega))
          ⟨_, _, rfl, rfl, fun _ => ⟨by omega, rfl, rfl, rfl, ?_⟩, fun h => absurd h (by omega)⟩
        simp [Array.range_succ, chS]
    case rest =>
      obtain ⟨-, hge⟩ := hI
      obtain ⟨-, htab, hw, hnl, hρ⟩ := hge le_rfl
      have hp : t₆.pending = t₆'.pending := hp₆ (by omega)
      rcases r with ⟨initCh, rho, zl⟩
      rcases r' with ⟨initCh', rho', zl'⟩
      dsimp only at htab hw ⊢
      have hc : ∀ i, 1 ≤ i → i < sch.levels.size → ∀ g,
          (commitP (Hashes.ofSha512 H) ms (sch.levels.getD i default) g (src.hiSalt i) : V Commit) =
            commitP (Hashes.ofSha512 H) ms (sch.levels.getD i default) g (salt i) := fun i h1 h2 g =>
        commitV_congr _ ms _ g (hsalt hnl hρ i h1 h2)
      rw [get_bind_run, get_bind_run, ← htab]
      refine lk_commit H (hc 1 le_rfl (by omega) _) hL₆ fun cm1 => ?_
      refine lk_lBy fun t₂ t₂' hp₂ => ?_
      refine lk_ood coin N _ _ rfl (by omega) hL₆ (hp₂ rfl hp) htab hw
        fun s₃ s₃' t₃ t₃' l₃ l₃' hL₃ hp₃ htab₃ hw₃ => ?_
      refine lk_ood coin N _ _ rfl (by omega) hL₃ hp₃ htab₃ hw₃ fun s₄ s₄' t₄ t₄' l₄ l₄' hL₄ hp₄ htab₄ hw₄ => ?_
      refine lk_query coin N _ _ ms _ cm0 cm0' hL₄ hp₄ htab₄ hw₄
        fun op0 op0' v0 s₅ s₅' t₅ t₅' l₅ l₅' _ hL₅ hp₅ htab₅ hw₅ => ?_
      dsimp only
      refine lk_send fun t₆ t₆' hp₆ => ?_
      refine lk_powP fun s₇ s₇' h7t h7w h7t' h7w' t₇ t₇' hp₇ => ?_
      refine lk_lsamplePAny coin N hL₅ (hp₇ (hp₆ rfl hp₅)) fun t₈ t₈' hp₈ l₈ l₈' hL₈ => ?_
      rw [modify_bind_run, modify_bind_run]
      have htab₈ : s₇.tab = s₇'.tab := by rw [h7t, h7t']; exact htab₅
      have hw₈ : s₇.w = s₇'.w := by rw [h7w, h7w']; exact hw₅
      -- the levels ≥ 1
      refine lk_lbind (Q := fun _ => True)
        (lk_forIn _ _ (fun _ p p' _ => p.1 = p'.1 ∧ p.2.tab = p'.2.tab ∧ p.2.w = p'.2.w) _ ?lvl hL₈
          (fun _ => hp₈) ⟨rfl, htab₈, by dsimp only; rw [hw₈]⟩)
        fun _ s₉ _ s₉' t₉ t₉' l₉ l₉' hL₉ hp₉ _ => lk_lpure hL₉ (fun _ => hp₉ trivial) trivial
      case lvl =>
        intro i hi u s u' s' ti ti' li li' hLi hpi ⟨hu, htabi, hwi⟩
        dsimp only at hu
        subst hu
        have hpi := hpi trivial
        refine lk_lbind (Q := fun _ => True)
          (lk_forIn _ _ (fun _ p p' _ => p.1 = p'.1 ∧ p.2.tab = p'.2.tab ∧ p.2.w = p'.2.w) 4 ?inner hLi
            (fun _ => hpi) ⟨rfl, htabi, hwi⟩)
          fun f s₁ f' s₁' t₁ t₁' l₁ l₁' hL₁ hp₁ hI₁ => ?after
        case inner =>
          intro j hj f s f' s' tj tj' lj lj' hLj hpj ⟨hf, htabj, hwj⟩
          dsimp only at hf htabj hwj
          subst hf
          have hpj := hpj trivial
          by_cases h4 : (j + 1 == 4) = true
          · rw [ite_eq_left h4]
            refine lk_lsampleKPAny coin N hLj hpj fun t₁ t₁' hp₁ l₁ l₁' hL₁ => ?_
            rw [modify_bind_run, modify_bind_run, get_bind_run, get_bind_run, modify_bind_run, modify_bind_run,
              get_bind_run, get_bind_run]
            refine lk_send fun t₂ t₂' hp₂ => ?_
            exact lk_lpure hL₁ (fun _ => hp₂ (by simp only [htabj, hwj]) hp₁)
              ⟨_, _, rfl, rfl, by simp only [htabj], by simp only [htabj], by simp only [hwj]⟩
          · rw [ite_eq_right h4]
            exact lk_challenge coin N hLj hpj htabj hwj fun _ s₁ s₁' t₁ t₁' l₁ l₁' hL₁ hp₁ htab₁ hw₁ =>
              lk_lpure hL₁ (fun _ => hp₁) ⟨_, _, rfl, rfl, rfl, htab₁, hw₁⟩
        case after =>
          obtain ⟨hf, htab₁, hw₁⟩ := hI₁
          dsimp only at hf
          subst hf
          have hp₁ := hp₁ trivial
          by_cases hlast : i + 1 = sch.levels.size - 1
          · rw [ite_eq_right (by simp [hlast]), ite_eq_right (by simp [hlast])]
            refine lk_absorbs _ _ (fun _ _ => rfl) (fun _ _ => rfl) fun t₂ t₂' hp₂ => ?_
            refine lk_query coin N _ _ ms _ _ _ hL₁ (hp₂ rfl hp₁) htab₁ hw₁
              fun op op' _ s₃ s₃' t₃ t₃' l₃ l₃' hop hL₃ hp₃ htab₃ hw₃ => ?_
            obtain rfl := hop rfl
            dsimp only
            refine lk_powP fun s₄ s₄' h4t h4w h4t' h4w' t₄ t₄' hp₄ => ?_
            refine lk_lsamplePAny coin N hL₃ (hp₄ hp₃) fun t₅ t₅' hp₅ l₅ l₅' hL₅ => ?_
            exact lk_lpure hL₅ (fun _ => hp₅) ⟨_, _, rfl, rfl, rfl, by rw [h4t, h4t']; exact htab₃,
              by rw [h4w, h4w']; exact hw₃⟩
          · rw [ite_eq_left (by simp [hlast]), ite_eq_left (by simp [hlast])]
            refine lk_commit H (hc (i + 2) (by omega) (by omega) _) hL₁ fun cmN => ?_
            refine lk_lBy fun t₂ t₂' hp₂ => ?_
            refine lk_ood coin N _ _ rfl (by omega) hL₁ (hp₂ rfl hp₁) htab₁ hw₁
              fun s₃ s₃' t₃ t₃' l₃ l₃' hL₃ hp₃ htab₃ hw₃ => ?_
            refine lk_ood coin N _ _ rfl (by omega) hL₃ hp₃ htab₃ hw₃
              fun s₄ s₄' t₄ t₄' l₄ l₄' hL₄ hp₄ htab₄ hw₄ => ?_
            refine lk_query coin N _ _ ms _ _ _ hL₄ hp₄ htab₄ hw₄
              fun op op' v s₅ s₅' t₅ t₅' l₅ l₅' hop hL₅ hp₅ htab₅ hw₅ => ?_
            obtain rfl := hop rfl
            dsimp only
            refine lk_send fun t₆ t₆' hp₆ => ?_
            refine lk_powP fun s₇ s₇' h7t h7w h7t' h7w' t₇ t₇' hp₇ => ?_
            refine lk_lsamplePAny coin N hL₅ (hp₇ (hp₆ rfl hp₅)) fun t₈ t₈' hp₈ l₈ l₈' hL₈ => ?_
            rw [modify_bind_run, modify_bind_run]
            exact lk_lpure hL₈ (fun _ => hp₈) ⟨_, _, rfl, rfl, rfl, by rw [h7t, h7t']; exact htab₅,
              by dsimp only; rw [h7w, h7w', hw₅]⟩

end FlockComplete.LigPast
