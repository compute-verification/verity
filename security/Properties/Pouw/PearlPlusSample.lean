module

public import Definitions.Pouw.Game.Defs
public import Properties.Spot.Draw
public import PouwPearlPlus
public import Mathlib.Algebra.BigOperators.Fin

/-! Pearl+'s run sample is `t` uniform work-weighted tickets, its streams modelled as independent and uniform. -/

@[expose] public section

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus

/-- The run's ticket total `W`: each sample's uniform integer is below it. -/
def ticketTotal (tk : List ((ℕ × ℕ) × ℕ)) : ℕ := (tk.map (·.2)).sum

/-- The bytes of each sample's stream the sample reads (`DRAW_WORDS` words at `uniform`'s width for `W`). -/
def sampleBytes (tk : List ((ℕ × ℕ) × ℕ)) : ℕ := DRAW_WORDS * Flock.Draw.width (ticketTotal tk)

/-- The tickets of the run's `u`-th tile, none past the last. -/
def weightsOf (tk : List ((ℕ × ℕ) × ℕ)) (u : ℕ) : ℕ := (tk.getD u ((0, 0), 0)).2

/-- The ticket numbered `r` in `Ticket`'s order: unit by unit, each unit's tickets in order. -/
def ticketAt (tk : List ((ℕ × ℕ) × ℕ)) (r : Fin (∑ u : Fin tk.length, weightsOf tk u)) :
    _root_.Pouw.Ticket (weightsOf tk) tk.length :=
  finSigmaFinEquiv.symm r

/-- The tile a ticket names. -/
def tileOf (tk : List ((ℕ × ℕ) × ℕ)) (c : _root_.Pouw.Ticket (weightsOf tk) tk.length) : ℕ × ℕ := (tk.get c.1).1

/-- **The run's sample is `t` tickets on its key's streams**: where the run's tickets hold no weight (`W = 0`), `draw`
returns the empty sample (`pearlDraw`'s empty filter); otherwise it (`pearlDraw`'s `weightedDraw`) names, for sample
`j`, the tile of the ticket numbered by the uniform integer below `W` its stream gives (index `("u", j)`, `sampleBytes`
long), and throws exactly where a stream gives none. -/
def DrawIsOnItsStreams : Prop :=
  ∀ (passes : List Pass) (key : ByteArray) (t : ℕ),
    (draw passes key t).toOption =
      if ticketTotal (tickets passes) = 0 then some [] else
      (List.range t).mapM fun j =>
        (FlockSoundness.Randomness.uniformOn (ticketTotal (tickets passes))
            (FlockSoundness.Randomness.stream key [.str "u", .int j] (sampleBytes (tickets passes)))).bind fun r =>
          if h : r < ∑ u : Fin (tickets passes).length, weightsOf (tickets passes) u then
            some (tileOf (tickets passes) (ticketAt (tickets passes) ⟨r, h⟩))
          else none

/-- **The coin bytes give uniform tickets**: for every list of tiles and tickets `tk`, number of samples `t` and event `E`
on `t` tickets, `t` independent streams of `sampleBytes tk` uniform bytes each give integers that number tickets in `E`
with probability at most `E`'s under `t` uniform tickets. -/
def CoinBytesGiveUniformTickets : Prop :=
  ∀ (tk : List ((ℕ × ℕ) × ℕ)) (t : ℕ) (E : (Fin t → _root_.Pouw.Ticket (weightsOf tk) tk.length) → Prop),
    _root_.Pouw.pr (fun g : Fin t → Fin (sampleBytes tk) → Fin 256 =>
        ∃ c : Fin t → _root_.Pouw.Ticket (weightsOf tk) tk.length,
          (∀ j, FlockSoundness.Randomness.uniformOn (ticketTotal tk) (Properties.Spot.bytesOf (g j)) =
            some (finSigmaFinEquiv (c j)).val) ∧ E c)
      ≤ _root_.Pouw.pr E

end Properties.Pouw.PearlPlus
