import Definitions.Core.Game

/-!
# Decisive games: decisiveness, not completeness

A game is decisive (`Decisive`) when every message type it asks for is inhabited (`Productive`), so that every position
has a run to the end, and two of its runs end differently (`Reaches`): for a verdict, some run accepts and some run
rejects. A batch of decisive games reads as no other list of games (PoUW's `Pouw.Draft.Game.Unambiguous`, by
`Pouw.Draft.Game.unambiguous_of_decisive`). Non-constancy is asked of the whole game only: a position from which every
run rejects is allowed. A game with a constant verdict is not enough: its moves and the schedule's choices can be folded
into another game of the batch.

**Existential, so no completeness claim.** `Reaches` picks the messages and the coins. The accepting run of a decisive
verdict game may need messages the honest prover never sends and coins that fall its way: C-Flock's all-zero run
accepts every table, satisfiable or not. So decisiveness gives `Unambiguous` and nothing more. It says nothing about how
often the honest prover, or any prover, is accepted; that is the reference prover's completeness, which stays open.
-/

namespace Verity.Game

variable {α : Type}

/-- Every message type `g` asks for is inhabited (coin types are, by `Game.coin`'s `Nonempty`). -/
def Productive : Game α → Prop
  | .ret _ => True
  | .send M k => Nonempty M ∧ ∀ m, Productive (k m)
  | @Game.coin _ _ _ _ k => ∀ c, Productive (k c)

/-- Some run of `g`, for some messages and some coins, ends with `a`. -/
def Reaches : Game α → α → Prop
  | .ret b, a => b = a
  | .send _ k, a => ∃ m, Reaches (k m) a
  | @Game.coin _ _ _ _ k, a => ∃ c, Reaches (k c) a

/-- **Decisive**: every message type of `g` is inhabited, and two runs of `g` end with different outputs. For a game
relabelled to its verdict: some run accepts and some run rejects. -/
def Decisive (g : Game α) : Prop :=
  Productive g ∧ ∃ a b, a ≠ b ∧ Reaches g a ∧ Reaches g b

end Verity.Game
