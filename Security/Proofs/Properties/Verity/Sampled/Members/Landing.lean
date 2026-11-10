module

public import Proofs.Core.Game.TwoRun
public import Definitions.Flock.Game.Tape

/-!
# A strategy that leaves a state where its game ends

A strategy of games played one after another (`FlockSoundness.Game.seqG`) that simulates one play of another game
carries the simulation's state from each game's end into the next. A strategy of a game says nothing at its ends, so
`Strategy.bind` continues from the outcome alone. A landing is a strategy with a state at every end:

* `Landing P g`: a message at every `send` node of `g` and a state of `P` at every end; `Landing.strat` is the strategy
  it plays, and `Landing.Lands` the outcomes and states its play ends on, over the coins.
* `Landing.law`: the game's coins with the landing's messages played, ending on a reading of the outcome and the state
  left. An event of the outcome has, under the landing's strategy, the probability of the same event of the reading
  (`Landing.prob_strat`), and the law's support is the readings of where the landing ends (`Landing.lands_of_law`).
* `Landing.bind`, `Landing.map`, `Landing.seq`: one landing after another from the state the first leaves, a relabelled
  game's, and landings one after another over a list, with where each ends (`lands_bind`, `lands_map`,
  `lands_seq_cons`, `lands_seq_nil`) and a relation every landing keeps kept by all of them (`lands_seq_rel`).
-/

@[expose] public section

namespace Verity.Game

open FlockSoundness.Game (seqG)
open scoped ENNReal

universe u v

variable {α β : Type} {P : Type u} {Q : Type v}

/-- **A strategy that leaves a state where the game ends**: a message at every `send` node, as a strategy, and a state
of `P` at every end. -/
def Landing (P : Type u) : Game α → Type u
  | .ret _ => P
  | .send _ k => (m : _) × Landing P (k m)
  | @Game.coin _ C _ _ k => (c : C) → Landing P (k c)

namespace Landing

/-- The strategy a landing plays. -/
def strat : (g : Game α) → Landing P g → Strategy g
  | .ret _, _ => PUnit.unit
  | .send _ k, s => ⟨s.1, strat (k s.1) s.2⟩
  | @Game.coin _ _ _ _ k, s => fun c => strat (k c) (s c)

/-- **A landing's play ends on the outcome `a` and leaves the state `p`**, on some coins. -/
def Lands : (g : Game α) → Landing P g → α → P → Prop
  | .ret a, p, a', p' => a' = a ∧ p' = p
  | .send _ k, s, a, p => Lands (k s.1) s.2 a p
  | @Game.coin _ C _ _ k, s, a, p => ∃ c : C, Lands (k c) (s c) a p

variable {γ : Type}

/-- **A landing's law** at the reading `r` of its outcomes and states: over the game's coins, the reading of the outcome
of its play and the state it leaves. -/
def law (r : α → P → γ) : (g : Game α) → Landing P g → Game γ
  | .ret a, p => .ret (r a p)
  | .send _ k, s => law r (k s.1) s.2
  | @Game.coin _ C iF iN k, s => @Game.coin _ C iF iN fun c => law r (k c) (s c)

/-- A landing's law's strategy: it has no message to send. -/
def lawS (r : α → P → γ) : (g : Game α) → (s : Landing P g) → Strategy (law r g s)
  | .ret _, _ => PUnit.unit
  | .send _ k, s => lawS r (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => fun c => lawS r (k c) (s c)

/-- **A landing's play has its events' probabilities under its law**: an event of the outcome has the probability of
any event of the reading that holds exactly when it does. -/
theorem prob_strat (r : α → P → γ) (p : α → Prop) (q : γ → Prop) (h : ∀ a x, q (r a x) ↔ p a) :
    ∀ (g : Game α) (s : Landing P g) (w : Strategy (law r g s)), prob p g (strat g s) = prob q (law r g s) w
  | .ret a, x, _ => by
    classical
    show (if p a then (1 : ℝ≥0∞) else 0) = if q (r a x) then 1 else 0
    exact if_congr (h a x).symm rfl rfl
  | .send _ k, s, w => prob_strat r p q h (k s.1) s.2 w
  | @Game.coin _ _ _ _ k, s, w =>
    Finset.sum_congr rfl fun c _ => congrArg (HMul.hMul _) (prob_strat r p q h (k c) (s c) (w c))

/-- **A landing's law's support**: the reading of an outcome of its play and the state it leaves there. -/
theorem lands_of_law (r : α → P → γ) : ∀ (g : Game α) (s : Landing P g) (w : Strategy (law r g s)) (o : γ),
    PlayReaches (law r g s) w o → ∃ a x, Lands g s a x ∧ o = r a x
  | .ret a, x, _, _, h => ⟨a, x, ⟨rfl, rfl⟩, h⟩
  | .send _ k, s, w, o, h => lands_of_law r (k s.1) s.2 w o h
  | @Game.coin _ _ _ _ k, s, w, o, ⟨c, h⟩ =>
    let ⟨a, x, hx, ho⟩ := lands_of_law r (k c) (s c) (w c) o h
    ⟨a, x, ⟨c, hx⟩, ho⟩

/-- **Play `s` in `g`, then from the state it leaves `t` in `f`'s game.** -/
def bind : (g : Game α) → (f : α → Game β) → Landing P g → ((a : α) → P → Landing Q (f a)) → Landing Q (g.bind f)
  | .ret a, _, p, t => t a p
  | .send _ k, f, s, t => ⟨s.1, bind (k s.1) f s.2 t⟩
  | @Game.coin _ _ _ _ k, f, s, t => fun c => bind (k c) f (s c) t

/-- The landing of `g.map f`: `s` in `g`, with the state it leaves. -/
def map (f : α → β) (g : Game α) (s : Landing P g) : Landing P (g.map f) :=
  bind g _ s fun _ p => p

/-- **Where one landing after another ends**: where the second does, from an outcome and a state the first leaves. -/
theorem lands_bind : ∀ (g : Game α) (f : α → Game β) (s : Landing P g) (t : (a : α) → P → Landing Q (f a)) (b : β)
    (q : Q), Lands (g.bind f) (bind g f s t) b q ↔ ∃ a p, Lands g s a p ∧ Lands (f a) (t a p) b q
  | .ret a, _, p, _, _, _ => ⟨fun h => ⟨a, p, ⟨rfl, rfl⟩, h⟩, fun ⟨_, _, ⟨ha, hp⟩, h⟩ => ha ▸ hp ▸ h⟩
  | .send _ k, f, s, t, b, q => lands_bind (k s.1) f s.2 t b q
  | @Game.coin _ _ _ _ k, f, s, t, b, q =>
    ⟨fun ⟨c, h⟩ => let ⟨a, p, h₁, h₂⟩ := (lands_bind (k c) f (s c) t b q).1 h; ⟨a, p, ⟨c, h₁⟩, h₂⟩,
      fun ⟨a, p, ⟨c, h₁⟩, h₂⟩ => ⟨c, (lands_bind (k c) f (s c) t b q).2 ⟨a, p, h₁, h₂⟩⟩⟩

/-- Where a relabelled game's landing ends: where the landing does, relabelled. -/
theorem lands_map (f : α → β) (g : Game α) (s : Landing P g) (b : β) (p : P) :
    Lands (g.map f) (map f g s) b p ↔ ∃ a, Lands g s a p ∧ b = f a :=
  (lands_bind g _ s _ b p).trans
    ⟨fun ⟨a, _, h, hb, hp⟩ => ⟨a, hp ▸ h, hb⟩, fun ⟨a, h, hb⟩ => ⟨a, p, h, hb, rfl⟩⟩

/-- The strategy a relabelled game's landing plays is the landing's. -/
theorem strat_map (f : α → β) : ∀ (g : Game α) (s : Landing P g),
    strat (g.map f) (map f g s) = Strategy.map f g (strat g s)
  | .ret _, _ => rfl
  | .send _ k, s => congrArg (Sigma.mk s.1) (strat_map f (k s.1) s.2)
  | @Game.coin _ _ _ _ k, s => funext fun c => strat_map f (k c) (s c)

/-- **Landings one after another**: `S k b` in `g b` for the `k`-th `b` of `l` on, each from the state the one before
leaves. -/
def seq (g : β → Game α) (S : ℕ → (b : β) → P → Landing P (g b)) : (l : List β) → ℕ → P → Landing P (seqG g l)
  | [], _, p => p
  | b :: l, k, p => bind (g b) _ (S k b p) fun a p' => map (a :: ·) (seqG g l) (seq g S l (k + 1) p')

/-- **Where landings one after another end**, from the `k`-th on: the first's outcome and state, then the rest's from
that state. -/
theorem lands_seq_cons (g : β → Game α) (S : ℕ → (b : β) → P → Landing P (g b)) (b : β) (l : List β) (k : ℕ) (p : P)
    (as : List α) (q : P) :
    Lands (seqG g (b :: l)) (seq g S (b :: l) k p) as q ↔
      ∃ a p' as', Lands (g b) (S k b p) a p' ∧ Lands (seqG g l) (seq g S l (k + 1) p') as' q ∧ as = a :: as' :=
  (lands_bind (g b) _ (S k b p) _ as q).trans
    ⟨fun ⟨a, p', h₁, h₂⟩ => let ⟨as', h₃, h₄⟩ := (lands_map (a :: ·) _ _ as q).1 h₂; ⟨a, p', as', h₁, h₃, h₄⟩,
      fun ⟨a, p', as', h₁, h₃, h₄⟩ => ⟨a, p', h₁, (lands_map (a :: ·) _ _ as q).2 ⟨as', h₃, h₄⟩⟩⟩

/-- Where no landing ends: on no outcome, in the state it starts from. -/
theorem lands_seq_nil (g : β → Game α) (S : ℕ → (b : β) → P → Landing P (g b)) (k : ℕ) (p : P) (as : List α) (q : P) :
    Lands (seqG g []) (seq g S [] k p) as q ↔ as = [] ∧ q = p :=
  Iff.rfl

/-- **A relation every landing keeps from the state it starts from to the state it leaves, kept by landings one after
another**, where it is reflexive and transitive. -/
theorem lands_seq_rel (g : β → Game α) (S : ℕ → (b : β) → P → Landing P (g b)) (I : P → P → Prop)
    (hr : ∀ p, I p p) (ht : ∀ p q r, I p q → I q r → I p r)
    (hS : ∀ k b p a q, Lands (g b) (S k b p) a q → I p q) :
    ∀ (l : List β) (k : ℕ) (p : P) (as : List α) (q : P), Lands (seqG g l) (seq g S l k p) as q → I p q
  | [], _, p, _, _, h => h.2 ▸ hr p
  | b :: l, k, p, as, q, h =>
    let ⟨a, p', as', h₁, h₂, _⟩ := (lands_seq_cons g S b l k p as q).1 h
    ht p p' q (hS k b p a p' h₁) (lands_seq_rel g S I hr ht hS l (k + 1) p' as' q h₂)

end Landing

end Verity.Game
