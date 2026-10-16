import Proofs.Flock.Recursive.Stage
import Proofs.Flock.Recursive.ForkRuns

/-!
# Recursion: every outcome of V*'s sessions is written

`Game.Leaves P g`: every leaf of `g` satisfies `P`, whatever the prover sends and the coins, so every outcome a strategy
reaches does (`Leaves.outcome`). It passes through `bind`, `map`, `play` and `seqC` (`leaves_seqC`), so:

* `ZkOuter.leaves_game`: every outcome of a session of V* is a transcript it writes at the registration it records
  (`ZkOuter.Writes`);
* `leaves_sessions`: every outcome of V*'s sessions in a row is, at each session, a transcript that session writes.

`prob_recT_le` reads `leaves_sessions` at the outcomes the transported prover reaches, so the checked sessions' gate
(`Flock.Guarantees.keepOwn`, a written, accepted transcript) holds where the event's does (`acceptsV_checked`).
-/

namespace FlockSoundness.Game

variable {α β : Type}

/-- **Every leaf of `g` satisfies `P`**, whatever the prover sends and the coins. -/
def Leaves (P : α → Prop) : Game α → Prop
  | .ret a => P a
  | .send _ k => ∀ m, Leaves P (k m)
  | @Game.coin _ _ _ _ k => ∀ c, Leaves P (k c)

theorem Leaves.outcome {P : α → Prop} : ∀ (g : Game α) (s : Strategy g) {a : α}, Leaves P g → Outcome g s a → P a
  | .ret _, _, _, h, ha => by cases ha; exact h
  | .send _ k, s, _, h, ha => Leaves.outcome (k s.1) s.2 (h s.1) ha
  | @Game.coin _ _ _ _ k, s, _, h, ha => by
      obtain ⟨c, ha⟩ := ha
      exact Leaves.outcome (k c) (s c) (h c) ha

theorem Leaves.bind {P : α → Prop} {Q : β → Prop} {f : α → Game β} (hf : ∀ a, P a → Leaves Q (f a)) :
    ∀ {g : Game α}, Leaves P g → Leaves Q (g.bind f)
  | .ret a, h => hf a h
  | .send _ _, h => fun m => Leaves.bind hf (h m)
  | @Game.coin _ _ _ _ _, h => fun c => Leaves.bind hf (h c)

theorem Leaves.map {P : α → Prop} {Q : β → Prop} {f : α → β} (hf : ∀ a, P a → Q (f a)) {g : Game α}
    (h : Leaves P g) : Leaves Q (g.map f) :=
  Leaves.bind (Q := Q) (f := fun a => .ret (f a)) hf h

/-- Every leaf of `g.map f` is in `f`'s image. -/
theorem Leaves.map_true {Q : β → Prop} {f : α → β} (hf : ∀ a, Q (f a)) : ∀ g : Game α, Leaves Q (g.map f)
  | .ret a => hf a
  | .send _ k => fun m => Leaves.map_true hf (k m)
  | @Game.coin _ _ _ _ k => fun c => Leaves.map_true hf (k c)

end FlockSoundness.Game

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec

variable {α : Type}

theorem leaves_play {P : α → Prop} : ∀ {g : Game α} (τ : Strategy g), Leaves P g → Leaves P (play g τ)
  | .ret _, _, h => h
  | .send _ k, τ, h => leaves_play (g := k τ.1) τ.2 (h τ.1)
  | @Game.coin _ _ _ _ k, τ, h => fun c => leaves_play (g := k c) (τ c) (h c)

/-- **Every leaf of games in a row satisfies each game's predicate at its outcome**, where every leaf of each game
does. -/
theorem leaves_seqC : ∀ (S : ℕ) {T α : Fin S → Type} {gs : (j : Fin S) → Game (α j)}
    {of : (j : Fin S) → T j → Strategy (gs j)} {Q : (j : Fin S) → α j → Prop},
    (∀ j, Leaves (Q j) (gs j)) → Leaves (fun b => ∀ j, Q j (b j).2) (seqC S gs of)
  | 0, _, _, _, _, _, _ => fun j => j.elim0
  | S + 1, T, α, _, of, Q, h =>
      Leaves.bind (P := fun r => Q 0 r.2)
        (fun r hr => Leaves.map (P := fun b => ∀ j : Fin S, Q j.succ (b j).2)
          (fun rs hrs j => Fin.cases (motive := fun j => Q j ((Fin.cons r rs : (j : Fin (S + 1)) → T j × α j) j).2)
            hr hrs j)
          (leaves_seqC S (T := fun j => T j.succ) (fun j => h j.succ)))
        (fun τ => Leaves.map (P := Q 0) (Q := fun r => Q 0 r.2) (f := fun a => (τ, a)) (fun _ ha => ha)
          (leaves_play (of 0 τ) (h 0)))

namespace ZkOuter

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv}

/-- **Every outcome of a session of V* is a transcript it writes** at the registration the outcome records. -/
theorem leaves_game (zo : ZkOuter mPts nv J Reg' c hU hscope Lv) (dg : ByteArray → ByteArray) (N : ℕ) :
    Leaves (fun o => zo.Writes o.1 o.2.2) (zo.game dg N) := fun R ω =>
  Leaves.map (P := fun t => ∃ x, t = zo.wr (Lv.draw ω) R x) (Q := fun o => zo.Writes o.1 o.2.2)
    (f := fun t => (R, Lv.draw ω, t)) (fun _ ⟨x, hx⟩ => ⟨ω, x, hx⟩)
    (Leaves.map_true (Q := fun t => ∃ x, t = zo.wr (Lv.draw ω) R x) (f := zo.wr (Lv.draw ω) R) (fun x => ⟨x, rfl⟩) _)

end ZkOuter

/-- **Every outcome of V*'s sessions in a row is, at each session, a transcript it writes.** -/
theorem leaves_sessions {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type} (zs : (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ) :
    Leaves (fun b => ∀ j, (zs j).Writes (b j).2.1 (b j).2.2.2) (sessions zs dg N) :=
  leaves_seqC S (Q := fun j (o : Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript) =>
    (zs j).Writes o.1 o.2.2) fun j => (zs j).leaves_game dg N

end FlockSoundness.Discharge.Recursion
