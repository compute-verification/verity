import Definitions.Core.Game

/-!
# Decisive games, and batches that read only one way

A game is decisive (`Decisive`) when every message type it asks for is inhabited (`Productive`), so that every position
has a run to the end, and two of its runs end differently (`Reaches`): for a verdict, some run accepts and some run
rejects. C-Flock's soundness states the same three over its own games (`FlockSoundness.Discharge.Exec.SessionLDecisive`,
whose `audit_tile_decisive_verdict` is a one-stage audit's verdict being decisive).

`batch` is not injective: `[ret false, send Bool fun _ => ret true]` and `[send Bool fun _ => ret false, ret true]` have
the same batch, the second game's move folded into the first's position. A list is `Unambiguous` when its batch reads
as no other list with a game at some position, so that a statement over every `batch (pre ++ g :: post) = batch L`
(as `Pouw.PearlC.BoundedByInduced`'s) speaks only of `L`'s own games at their own positions. A list of decisive games is
unambiguous (`Verity.Game.unambiguous_of_decisive`); a constant game can break it
(`Verity.Game.not_unambiguous_of_constant`).

**Existential, so no completeness claim.** `Reaches` picks the messages and the coins. The accepting run of a decisive
verdict game may need messages the honest prover never sends and coins that fall its way: C-Flock's all-zero run
accepts every table, satisfiable or not. Decisiveness says nothing about how often the honest prover, or any prover, is
accepted.
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

/-- The batch of `L` is the batch of no other list with a game at some position: every way of reading it as
`batch (pre ++ g :: post)` is `L` itself. -/
def Unambiguous (L : List (Game α)) : Prop :=
  ∀ (pre : List (Game α)) (g : Game α) (post : List (Game α)), batch (pre ++ g :: post) = batch L →
    pre ++ g :: post = L

end Verity.Game
