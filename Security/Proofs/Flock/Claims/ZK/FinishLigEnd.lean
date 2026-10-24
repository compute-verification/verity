import Proofs.Flock.Claims.ZK.FinishFc

/-!
# Ligerito from a source ends on a squeeze

Every run of `ligeritoCut` at `asksCut` that returns ends with no pending bytes: its last step that touches the
transcript is a squeeze (the last level's query coin, or level 0's `bq` where there are no further levels), after which
it only updates its own state. So the inner proof that follows begins its first round with its own label
(`ligeritoCut_end`).

* `EndT`, `KeepT`, `EndL`, `KeepL`: a step that leaves no pending bytes, and one that keeps none, on the transcript and
  on Ligerito's state over it.
-/

namespace Flock.Firewall.Finish

open Flock Transcript
open FlockComplete (PS)

/-- A step that, where it returns, leaves no pending bytes. -/
def EndT {α : Type} (p : TMW PS α) : Prop :=
  ∀ t l a t' l', (p.run t).go l = (.ok (a, t'), l') → t'.pending = ByteArray.empty

/-- A step that, begun with no pending bytes, leaves none where it returns. -/
def KeepT {α : Type} (p : TMW PS α) : Prop :=
  ∀ t l a t' l', t.pending = ByteArray.empty → (p.run t).go l = (.ok (a, t'), l') → t'.pending = ByteArray.empty

/-- `EndT` for Ligerito's steps. -/
def EndL {α : Type} (p : FlockComplete.LP PS α) : Prop :=
  ∀ st t l a st' t' l', ((p.run st).run t).go l = (.ok ((a, st'), t'), l') → t'.pending = ByteArray.empty

/-- `KeepT` for Ligerito's steps. -/
def KeepL {α : Type} (p : FlockComplete.LP PS α) : Prop :=
  ∀ st t l a st' t' l', t.pending = ByteArray.empty → ((p.run st).run t).go l = (.ok ((a, st'), t'), l') →
    t'.pending = ByteArray.empty

section
variable {α β : Type}

theorem endT_bind {p : TMW PS α} {f : α → TMW PS β} (hf : ∀ a, EndT (f a)) : EndT (p >>= f) := by
  intro t l b t'' l'' h
  obtain ⟨a, t', l', -, h⟩ := run_bind_ok h
  exact hf a t' l' b t'' l'' h

theorem endT_seq {p : TMW PS α} {f : α → TMW PS β} (hp : EndT p) (hf : ∀ a, KeepT (f a)) : EndT (p >>= f) := by
  intro t l b t'' l'' h
  obtain ⟨a, t', l', h₁, h⟩ := run_bind_ok h
  exact hf a t' l' b t'' l'' (hp t l a t' l' h₁) h

theorem keepT_pure (a : α) : KeepT (pure a : TMW PS α) := by
  intro t l b t' l' h0 h
  rw [FlockComplete.Replay.pure_go] at h
  simp only [Prod.mk.injEq, Except.ok.injEq] at h
  obtain ⟨⟨-, rfl⟩, -⟩ := h
  exact h0

theorem endT_run {p : FlockComplete.LP PS α} (hp : EndL p) (s : FlockComplete.LigState) : EndT (p.run s) :=
  fun t l a t' l' h => hp s t l a.1 a.2 t' l' h

theorem endT_sampleCut (coin : ℕ → ℕ → F128) (n : ℕ) : EndT (FlockComplete.sampleP (FlockComplete.asksCut coin n)) := by
  intro t l a t' l' h
  rw [FlockComplete.SimInner.sampleCut_go] at h
  obtain ⟨-, -, rfl, -⟩ := FlockComplete.SimInner.ite_ok h
  rfl

theorem endL_bind {p : FlockComplete.LP PS α} {f : α → FlockComplete.LP PS β} (hf : ∀ a, EndL (f a)) :
    EndL (p >>= f) := by
  intro st t l b st'' t'' l'' h
  obtain ⟨a, st', t', l', -, h⟩ := FlockComplete.Replay.Lg.lp_bind_ok h
  exact hf a st' t' l' b st'' t'' l'' h

theorem endL_seq {p : FlockComplete.LP PS α} {f : α → FlockComplete.LP PS β} (hp : EndL p)
    (hf : ∀ a, KeepL (f a)) : EndL (p >>= f) := by
  intro st t l b st'' t'' l'' h
  obtain ⟨a, st', t', l', h₁, h⟩ := FlockComplete.Replay.Lg.lp_bind_ok h
  exact hf a st' t' l' b st'' t'' l'' (hp st t l a st' t' l' h₁) h

theorem keepL_bind {p : FlockComplete.LP PS α} {f : α → FlockComplete.LP PS β} (hp : KeepL p)
    (hf : ∀ a, KeepL (f a)) : KeepL (p >>= f) := by
  intro st t l b st'' t'' l'' h0 h
  obtain ⟨a, st', t', l', h₁, h⟩ := FlockComplete.Replay.Lg.lp_bind_ok h
  exact hf a st' t' l' b st'' t'' l'' (hp st t l a st' t' l' h0 h₁) h

theorem keepL_of_endL {p : FlockComplete.LP PS α} (hp : EndL p) : KeepL p :=
  fun st t l a st' t' l' _ h => hp st t l a st' t' l' h

theorem keepL_pure (a : α) : KeepL (pure a : FlockComplete.LP PS α) := by
  intro st t l b st' t' l' h0 h
  obtain ⟨-, -, rfl, -⟩ := FlockComplete.Replay.Lg.ok_inj (show (Except.ok ((a, st), t), l) = _ from h)
  exact h0

theorem keepL_modify (f : FlockComplete.LigState → FlockComplete.LigState) :
    KeepL (modify f : FlockComplete.LP PS PUnit) := by
  intro st t l b st' t' l' h0 h
  obtain ⟨-, -, rfl, -⟩ := FlockComplete.Replay.Lg.ok_inj (show (Except.ok (((), f st), t), l) = _ from h)
  exact h0

theorem endL_ite {c : Prop} [Decidable c] {p q : FlockComplete.LP PS α} (hp : EndL p) (hq : EndL q) :
    EndL (if c then p else q) := by
  split <;> assumption

theorem endL_liftT {x : TMW PS α} (hx : EndT x) : EndL (FlockComplete.liftT x) := by
  intro st t l a st' t' l' h
  change ((x >>= fun a => Pure.pure (a, st)).run t).go l = _ at h
  obtain ⟨a', t₁, l₁, h₁, h⟩ := run_bind_ok h
  rw [FlockComplete.Replay.pure_go] at h
  simp only [Prod.mk.injEq, Except.ok.injEq] at h
  obtain ⟨⟨-, rfl⟩, -⟩ := h
  exact hx t l a' t₁ l₁ h₁

theorem keepL_forIn_list {σ : Type} (xs : List ℕ) (b : σ) {f : ℕ → σ → FlockComplete.LP PS (ForInStep σ)}
    (hf : ∀ i b, KeepL (f i b)) : KeepL (forIn xs b f) := by
  induction xs generalizing b with
  | nil =>
    rw [List.forIn_nil]
    exact keepL_pure b
  | cons x xs ih =>
    rw [List.forIn_cons]
    refine keepL_bind (hf x b) fun r => ?_
    cases r with
    | done b' => exact keepL_pure b'
    | yield b' => exact ih b'

theorem keepL_forIn_range {σ : Type} (n : ℕ) (b : σ) {f : ℕ → σ → FlockComplete.LP PS (ForInStep σ)}
    (hf : ∀ i b, KeepL (f i b)) : KeepL (forIn [:n] b f) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  exact keepL_forIn_list _ b hf

end

/-- Walks a Ligerito step to its last squeeze (`EndL`), backtracking where a squeeze is followed by more transcript
work. -/
syntax "end_l" : tactic
/-- Walks a Ligerito step that only updates its own state (`KeepL`). -/
syntax "keep_l" : tactic
macro_rules
  | `(tactic| end_l) => `(tactic| first
      | exact endL_liftT (endT_sampleCut _ _)
      | (refine endL_seq (endL_liftT (endT_sampleCut _ _)) fun _ => ?_; keep_l)
      | (refine endL_bind fun _ => ?_; end_l)
      | (refine endL_ite ?_ ?_ <;> end_l)
      | (split <;> end_l))
macro_rules
  | `(tactic| keep_l) => `(tactic| first
      | exact keepL_pure _
      | exact keepL_modify _
      | (refine keepL_bind (keepL_modify _) fun _ => ?_; keep_l)
      | (refine keepL_bind (keepL_forIn_range _ _ fun _ _ => keepL_of_endL ?_) fun _ => ?_; end_l; keep_l)
      | (split <;> keep_l))

set_option maxHeartbeats 4000000 in
/-- **Ligerito from a source ends on a squeeze**: at `asksCut`, where `ligeritoCut` returns, it leaves no pending
bytes. -/
theorem ligeritoCut_end (coin : ℕ → ℕ → F128) (n : ℕ) (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128)
    (cm0 : FlockComplete.Commit) (b : Array F128) (lay : Zk.PadLayout) (src : FlockComplete.RepSrc) :
    EndT (FlockComplete.ligeritoCut (FlockComplete.asksCut coin n) (Hashes.ofSha512 Sha512.hash) ms sch T cm0 b lay
      src) := by
  unfold FlockComplete.ligeritoCut
  try dsimp only
  refine endT_bind fun _ => endT_bind fun _ => endT_bind fun _ => endT_bind fun _ => ?_
  refine endT_seq (endT_run ?_ _) fun _ => ?_
  · end_l
  · first
      | exact keepT_pure _
      | (split; exact keepT_pure _)

end Flock.Firewall.Finish
