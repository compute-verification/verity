module

public import Definitions.Flock.Game

/-!
# A strategy's coins up to a point of the game, sampled ahead

A deterministic strategy `s` of `g.bind f` plays `g` and then `f a` at `g`'s outcome `a`. `Tape f T g s` holds the coins
`g` samples at `s`, each coin with, for every value of it, the coins after it, and then a tape `T a s'` of the
continuation at the strategy `s'` that `s` leaves for `f a`. `Tape.land` is where a tape lands: `g`'s outcome, the
strategy `s` leaves there, and the continuation's tape.

`seqG g l` plays the games `g b` for `b` in `l`, one after another, and outputs their outputs in order. At a strategy `s`
of it, `seqTape g l k h s` is the coins of the first `k` games, and `seqStrat g l k h s t` the strategy that plays game
`k` after them on the tape `t`. That an event of game `k`'s output has the same probability over `seqG`'s coins as over
the tape and game `k`'s own coins (`prob_seqG`), and the averages it needs, are in `Proofs.Flock.Soundness.Game.Tape`.
-/

@[expose] public section

namespace FlockSoundness.Game

variable {α β : Type}

section Tape

variable (f : α → Game β) (T : (a : α) → Strategy (f a) → Type)

/-- **The coins `g` samples at a strategy of `g.bind f`, then the continuation's tape `T`.** At a coin `C`, the coin and,
for every value of it, the rest. -/
def Tape : (g : Game α) → Strategy (g.bind f) → Type
  | .ret a, s => T a s
  | .send _ k, s => Tape (k s.1) s.2
  | @Game.coin _ C _ _ k, s => C × ((c : C) → Tape (k c) (s c))

/-- **Where a tape lands**: `g`'s outcome, the strategy left for `f` there, and the continuation's tape. -/
def Tape.land : (g : Game α) → (s : Strategy (g.bind f)) → Tape f T g s → (a : α) × (s' : Strategy (f a)) × T a s'
  | .ret a, s, t => ⟨a, s, t⟩
  | .send _ k, s, t => Tape.land (k s.1) s.2 t
  | @Game.coin _ _ _ _ k, s, t => Tape.land (k t.1) (s t.1) (t.2 t.1)

open Classical in
/-- A tape over finite continuation tapes is finite. -/
@[instance_reducible]
noncomputable def Tape.fintype (hT : ∀ a s, Fintype (T a s)) : (g : Game α) → (s : Strategy (g.bind f)) →
    Fintype (Tape f T g s)
  | .ret a, s => hT a s
  | .send _ k, s => Tape.fintype hT (k s.1) s.2
  | @Game.coin _ C _ _ k, s =>
      letI := fun c => Tape.fintype hT (k c) (s c)
      inferInstanceAs (Fintype (C × ((c : C) → Tape f T (k c) (s c))))

/-- A tape over nonempty continuation tapes is nonempty. -/
theorem Tape.nonempty (hT : ∀ a s, Nonempty (T a s)) : ∀ (g : Game α) (s : Strategy (g.bind f)),
    Nonempty (Tape f T g s)
  | .ret a, s => hT a s
  | .send _ k, s => Tape.nonempty hT (k s.1) s.2
  | @Game.coin _ C _ _ k, s =>
      haveI := fun c => Tape.nonempty hT (k c) (s c)
      inferInstanceAs (Nonempty (C × ((c : C) → Tape f T (k c) (s c))))

end Tape

/-- **Games one after another**: `g b` for each `b` of `l` in order, with their outputs in order. -/
def seqG (g : β → Game α) : List β → Game (List α)
  | [] => .ret []
  | b :: l => (g b).bind fun a => (seqG g l).map (a :: ·)

/-- **The coins of the games before game `k` of `seqG g l`**, at the strategy `s`. -/
def seqTape (g : β → Game α) : (l : List β) → (k : ℕ) → k < l.length → Strategy (seqG g l) → Type
  | [], _, _, _ => PUnit
  | _ :: _, 0, _, _ => PUnit
  | _ :: l, k + 1, h, s => Tape (fun a => (seqG g l).map (a :: ·))
      (fun a s' => seqTape g l k (Nat.lt_of_succ_lt_succ h) (Strategy.ofMap (a :: ·) (seqG g l) s')) _ s

/-- **The strategy game `k` of `seqG g l` is played with**, at the strategy `s` and on the tape `t` of the coins before
it. -/
def seqStrat (g : β → Game α) : (l : List β) → (k : ℕ) → (h : k < l.length) → (s : Strategy (seqG g l)) →
    seqTape g l k h s → Strategy (g (l.get ⟨k, h⟩))
  | [], _, h, _, _ => absurd h (Nat.not_lt_zero _)
  | b :: _, 0, _, s, _ => Strategy.ofBind _ (g b) s
  | b :: l, k + 1, h, s, t =>
      let x := Tape.land (fun a => (seqG g l).map (a :: ·))
        (fun a s' => seqTape g l k (Nat.lt_of_succ_lt_succ h) (Strategy.ofMap (a :: ·) (seqG g l) s')) (g b) s t
      seqStrat g l k (Nat.lt_of_succ_lt_succ h) (Strategy.ofMap (x.1 :: ·) (seqG g l) x.2.1) x.2.2

/-- The tape of the games before game `k` is finite. -/
@[instance_reducible]
noncomputable def seqTape.fintype (g : β → Game α) : (l : List β) → (k : ℕ) → (h : k < l.length) →
    (s : Strategy (seqG g l)) → Fintype (seqTape g l k h s)
  | [], _, _, _ => inferInstanceAs (Fintype PUnit)
  | _ :: _, 0, _, _ => inferInstanceAs (Fintype PUnit)
  | b :: l, k + 1, h, s => Tape.fintype (fun a => (seqG g l).map (a :: ·))
      (fun a s' => seqTape g l k (Nat.lt_of_succ_lt_succ h) (Strategy.ofMap (a :: ·) (seqG g l) s'))
      (fun _ _ => seqTape.fintype g l k _ _) (g b) s

/-- The tape of the games before game `k` is nonempty. -/
theorem seqTape.nonempty (g : β → Game α) : ∀ (l : List β) (k : ℕ) (h : k < l.length)
    (s : Strategy (seqG g l)), Nonempty (seqTape g l k h s)
  | [], _, _, _ => ⟨PUnit.unit⟩
  | _ :: _, 0, _, _ => ⟨PUnit.unit⟩
  | b :: l, k + 1, h, s => Tape.nonempty (fun a => (seqG g l).map (a :: ·))
      (fun a s' => seqTape g l k (Nat.lt_of_succ_lt_succ h) (Strategy.ofMap (a :: ·) (seqG g l) s'))
      (fun _ _ => seqTape.nonempty g l k _ _) (g b) s

end FlockSoundness.Game
