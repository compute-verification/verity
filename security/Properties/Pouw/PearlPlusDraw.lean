module

public import Definitions.Pouw.Game.Defs
public import Properties.Spot.Draw
public import PouwPearlPlus
public import Mathlib.Algebra.BigOperators.Fin

/-!
# Pearl+'s run sample is `t` uniform work-weighted tickets (`PouwCheck.PearlPlus.draw`)

The sample a verdict opens (`PlanMeans`'s tiles): `t` work-weighted samples with replacement over every retained pass's
tickets together (`tickets`, `pearlDraw`'s `weightedDraw`). Its units are the run's tiles in `tickets`' order, tile `u`
holding `weightsOf tk u` tickets, so a sample is a `Pouw.Ticket (weightsOf tk) tk.length`: the sampling design `Pouw.MissBound` is
stated over. Two statements tie the code's sample to that sampling design:

* `DrawIsOnItsStreams`: sample `j` reads its own stream of the run's sample key, index `("u", j)`, `drawBytes` bytes long,
  takes the uniform integer `uniformOn` gives below the tickets' total `W`, and names the tile of the ticket with that
  number in `Ticket`'s order (`ticketAt`, unit by unit and each unit's tickets in order, which is the running sum
  `bisectRight` inverts). It throws exactly where a stream runs out.
* `CoinBytesGiveUniformTickets`: over `t` independent streams of `drawBytes` uniform bytes, the tickets the integers
  number fall in any event no more often than `t` uniform tickets do. A word `uniformOn` rejects takes probability from
  every ticket alike, and a stream that runs out refuses the run (`plan`'s `ValueError`), so it lands in no event.

The streams are modelled as independent and uniform, as `Properties.Pouw.Credit.WorkDrawMisses` models the window
sample's: the run derives sample `j`'s as `stream key ["u", j] (drawBytes tk)`, its index distinct for each sample, and this
is that derivation in the random-oracle model. They add no named assumption: the probability is over the streams'
bytes. The key is `derive(source, "verity/pouw/pearl-plus-run-draw/v1", {run})` on the run's commitment, fixed before
the coin (`after-coin`).
-/

@[expose] public section

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus

/-- The run's ticket total `W`: each sample's uniform integer is below it. -/
def ticketTotal (tk : List ((ℕ × ℕ) × ℕ)) : ℕ := (tk.map (·.2)).sum

/-- The bytes of each sample's stream the sample reads (`DRAW_WORDS` words at `uniform`'s width for `W`). -/
def drawBytes (tk : List ((ℕ × ℕ) × ℕ)) : ℕ := DRAW_WORDS * Flock.Draw.width (ticketTotal tk)

/-- The tickets of the run's `u`-th tile, none past the last. -/
def weightsOf (tk : List ((ℕ × ℕ) × ℕ)) (u : ℕ) : ℕ := (tk.getD u ((0, 0), 0)).2

/-- The ticket numbered `r` in `Ticket`'s order: unit by unit, each unit's tickets in order. -/
def ticketAt (tk : List ((ℕ × ℕ) × ℕ)) (r : Fin (∑ u : Fin tk.length, weightsOf tk u)) :
    _root_.Pouw.Ticket (weightsOf tk) tk.length :=
  finSigmaFinEquiv.symm r

/-- The tile a ticket names. -/
def tileOf (tk : List ((ℕ × ℕ) × ℕ)) (c : _root_.Pouw.Ticket (weightsOf tk) tk.length) : ℕ × ℕ := (tk.get c.1).1

/-- **The run's sample is `t` tickets on its key's streams**: wherever the run's tickets hold some weight, `draw`
(`pearlDraw`'s `weightedDraw`) names, for sample `j`, the tile of the ticket numbered by the uniform integer below `W` its
stream gives (index `("u", j)`, `drawBytes` long), and throws exactly where a stream gives none. -/
def DrawIsOnItsStreams : Prop :=
  ∀ (passes : List Pass) (key : ByteArray) (t : ℕ), 0 < ticketTotal (tickets passes) →
    (draw passes key t).toOption =
      (List.range t).mapM fun j =>
        (FlockSoundness.Randomness.uniformOn (ticketTotal (tickets passes))
            (FlockSoundness.Randomness.stream key [.str "u", .int j] (drawBytes (tickets passes)))).bind fun r =>
          if h : r < ∑ u : Fin (tickets passes).length, weightsOf (tickets passes) u then
            some (tileOf (tickets passes) (ticketAt (tickets passes) ⟨r, h⟩))
          else none

/-- **The coin bytes give uniform tickets**: for every list of tiles and tickets `tk`, number of samples `t` and event `E`
on `t` tickets, `t` independent streams of `drawBytes tk` uniform bytes each give integers that number tickets in `E`
with probability at most `E`'s under `t` uniform tickets. -/
def CoinBytesGiveUniformTickets : Prop :=
  ∀ (tk : List ((ℕ × ℕ) × ℕ)) (t : ℕ) (E : (Fin t → _root_.Pouw.Ticket (weightsOf tk) tk.length) → Prop),
    _root_.Pouw.pr (fun g : Fin t → Fin (drawBytes tk) → Fin 256 =>
        ∃ c : Fin t → _root_.Pouw.Ticket (weightsOf tk) tk.length,
          (∀ j, FlockSoundness.Randomness.uniformOn (ticketTotal tk) (Properties.Spot.bytesOf (g j)) =
            some (finSigmaFinEquiv (c j)).val) ∧ E c)
      ≤ _root_.Pouw.pr E

end Properties.Pouw.PearlPlus
