import Proofs.Flock.Claims.Complete.Discharge

/-!
# What the pure session returns

A small program logic over the pure session's monad `PS`, for the goals that read the session's outputs:

* `Ret y P`: every value `y` returns satisfies `P`, from every state. `Ret.pure`, `Ret.bind`, `Ret.true`, and the loops
  (`Ret.forIn_list`, `Ret.forIn_array`, by an invariant on the iteration count).
* `sessionP_proofs`: the session returns one proof file per rep stream.
* `proofCount_goal` (`ProofCount`): one proof file per rep stream at every honest run.
-/

open Flock FlockDeploy

namespace FlockComplete

/-- **`y` returns only values satisfying `P`**, from every state. -/
def Ret {α : Type} (y : PS α) (P : α → Prop) : Prop :=
  ∀ l a l', y.go l = (.ok a, l') → P a

theorem Ret.true {α : Type} (y : PS α) : Ret y fun _ => True := fun _ _ _ _ => trivial

theorem Ret.pure {α : Type} {P : α → Prop} {a : α} (h : P a) : Ret (Pure.pure a : PS α) P := by
  intro l b l' e
  simp only [StV.go, Id.run] at e
  cases e
  exact h

theorem Ret.bind {α β : Type} {x : PS α} {f : α → PS β} {Q : α → Prop} {P : β → Prop} (hx : Ret x Q)
    (hf : ∀ a, Q a → Ret (f a) P) : Ret (x >>= f) P := by
  intro l b l' e
  rw [go_bind] at e
  rcases hy : x.go l with ⟨r, l''⟩
  rw [hy] at e
  cases r with
  | error _ => cases e
  | ok a => exact hf a (hx l a l'' hy) l'' b l' e

theorem Ret.forIn_list {α β : Type} (f : α → β → PS (ForInStep β)) (inv : ℕ → β → Prop)
    (hs : ∀ k a b, inv k b → Ret (f a b) fun r => ∃ b', r = .yield b' ∧ inv (k + 1) b') :
    ∀ (l : List α) (k : ℕ) (b : β), inv k b → Ret (forIn l b f) (inv (k + l.length)) := by
  intro l
  induction l with
  | nil => intro k b h; exact Ret.pure (by simpa using h)
  | cons a as ih =>
    intro k b h
    rw [List.forIn_cons]
    refine Ret.bind (hs k a b h) ?_
    rintro r ⟨b', rfl, hb'⟩
    have := ih (k + 1) b' hb'
    rw [List.length_cons, show k + (as.length + 1) = k + 1 + as.length by omega]
    exact this

theorem Ret.forIn_array {α β : Type} (xs : Array α) (b : β) (f : α → β → PS (ForInStep β)) (inv : ℕ → β → Prop)
    (h0 : inv 0 b) (hs : ∀ k a b, inv k b → Ret (f a b) fun r => ∃ b', r = .yield b' ∧ inv (k + 1) b') :
    Ret (forIn xs b f) (inv xs.size) := by
  rw [← Array.forIn_toList]
  have := Ret.forIn_list f inv hs xs.toList 0 b h0
  simpa using this

/-- **The session returns one proof file per rep stream.** -/
theorem sessionP_proofs (h : ByteArray → ByteArray) (resp : ℕ → Claims.Msg) (spec : SessionSpec)
    (tabs : Array Setup) (trs : Array Trace) (tape : Tape) :
    Ret (sessionP (asksS resp) (Hashes.ofSha512 h) spec tabs trs tape) fun r => r.2.size = spec.repStreams.size := by
  unfold sessionP
  refine Ret.bind (Ret.true _) fun _ _ => ?_
  refine Ret.bind (Ret.true _) fun _ _ => ?_
  refine Ret.bind (Ret.true _) fun _ _ => ?_
  refine Ret.bind (Ret.forIn_array _ _ _ (fun k b => b.snd.size = k) rfl ?_) ?_
  · rintro k ⟨i, r, name⟩ ⟨ss, ps⟩ hk
    refine Ret.bind (Ret.true _) fun _ _ => ?_
    refine Ret.bind (Ret.true _) fun _ _ => ?_
    exact Ret.pure ⟨_, rfl, by simp [hk]⟩
  · rintro ⟨ss, ps⟩ hk
    refine Ret.bind (Ret.true _) fun _ _ => ?_
    exact Ret.pure hk

/-- **One proof file per rep stream**: the goal `ProofCount`. -/
theorem proofCount_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    ProofCount H rel := by
  intro o
  have := sessionP_proofs (Claims.hOf H) (fun j => (Claims.coinSeq o.r j).bytes) o.spec (tabsOf o.sts)
    (tracesOf o.sts o.w) o.ω #[] _ _ o.ran
  simp [this]

end FlockComplete
