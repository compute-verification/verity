import Verity.Protocol.Game

/-!
# Playing a batch one game after another (provisional; trusted)

**Provisional.** Core-only definitions about `Verity.Game`, held here until they move to core's package beside
`Verity.Game.batch`. A strategy for each game of a list (`Strategies`) gives a strategy for their batch that plays
them one after another in list order (`Strategy.batchSeq`): it always advances the first game that has not ended. The
honest prover of the hidden audit's proof phase is one (`HiddenSystem.honestProver`).
-/

namespace Pouw.Draft.Game

open Verity Verity.Game

variable {α β : Type}

/-- A strategy for each game of a list. -/
def Strategies : List (Game α) → Type
  | [] => PUnit
  | g :: gs => Strategy g × Strategies gs

/-- The strategy of `Strategies (pre ++ g :: post)` for `g`, the game at position `pre.length`. -/
def Strategies.get {g : Game α} {post : List (Game α)} :
    (pre : List (Game α)) → Strategies (pre ++ g :: post) → Strategy g
  | [], ss => ss.1
  | _ :: pre, ss => Strategies.get pre ss.2

/-- A strategy for each game of `ts.map G`, from one for each `G t`. -/
def Strategies.ofMap {ι : Type} (G : ι → Game α) (hon : (t : ι) → Strategy (G t)) :
    (ts : List ι) → Strategies (ts.map G)
  | [] => PUnit.unit
  | t :: ts => (hon t, Strategies.ofMap G hon ts)

/-- A strategy for `interleaveWith adv₁ fin₁ h` that advances the first interaction at once: `sadv` from there, and
`sfin b` if the second has already ended with `b`. -/
def Strategy.seqWith (adv₁ : Game β → Game (α × β)) (fin₁ : β → Game (α × β)) :
    (h : Game β) → Strategy (adv₁ h) → ((b : β) → Strategy (fin₁ b)) → Strategy (interleaveWith adv₁ fin₁ h)
  | .ret b, _, sfin => sfin b
  | .send _ _, sadv, _ => ⟨true, sadv⟩
  | @Game.coin _ _ _ _ _, sadv, _ => ⟨true, sadv⟩

/-- **One game, then the other**: the strategy for `interleave g h` that plays `g` by `sg` until it ends, then `h` by
`sh`. -/
def Strategy.seq : (g : Game α) → (h : Game β) → Strategy g → Strategy h → Strategy (interleave g h)
  | .ret a, h, _, sh => Strategy.toMap (fun b => (a, b)) h sh
  | .send M k, h, sg, sh => Strategy.seqWith (fun g₂ => .send M fun m => interleave (k m) g₂)
      (fun b => (Game.send M k).map fun a => (a, b)) h ⟨sg.1, Strategy.seq (k sg.1) h sg.2 sh⟩
      (fun b => Strategy.toMap (fun a => (a, b)) (.send M k) sg)
  | @Game.coin _ C iF iN k, h, sg, sh => Strategy.seqWith (fun g₂ => @Game.coin _ C iF iN fun c => interleave (k c) g₂)
      (fun b => (@Game.coin _ C iF iN k).map fun a => (a, b)) h (fun c => Strategy.seq (k c) h (sg c) sh)
      (fun b => Strategy.toMap (fun a => (a, b)) (@Game.coin _ C iF iN k) sg)

/-- **A batch played one game after another**, in list order, each game by its own strategy. -/
def Strategy.batchSeq : (L : List (Game α)) → Strategies L → Strategy (batch L)
  | [], _ => PUnit.unit
  | g :: gs, ss => Strategy.toMap (fun x : α × List α => x.1 :: x.2) _
      (Strategy.seq g (batch gs) ss.1 (Strategy.batchSeq gs ss.2))

/-- The batch of `L` is the batch of no other list with a game at some position: every way of reading it as
`batch (pre ++ g :: post)` is `L` itself. `batch` is not injective: `[ret false, send Bool fun _ => ret true]` and
`[send Bool fun _ => ret false, ret true]` have the same batch. -/
def Unambiguous (L : List (Game α)) : Prop :=
  ∀ (pre : List (Game α)) (g : Game α) (post : List (Game α)), batch (pre ++ g :: post) = batch L →
    pre ++ g :: post = L

end Pouw.Draft.Game
