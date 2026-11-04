import Definitions.Core.Draw
import Properties.SampledProofs.Draw
import PouwPearlPlus

/-!
# Pearl+'s run draw misses a set of tiles no more often than `t` uniform tickets do (`PouwCheck.PearlPlus.draw`)

The draw a verdict opens (`PlanMeans`'s tiles): `t` work-weighted draws with replacement over every retained pass's
tickets together (`tickets`, `pearlDraw`'s `weightedDraw`). Draw `j` reads its own stream of the run's draw key, index
`("u", j)`, `drawBytes` bytes long, takes the uniform integer `uniformOn` gives below the tickets' total `W`, and names
the tile whose running sum first passes it (`tileAt`). So it is a function of `t` streams (`drawOn`), and on the key's
streams it is the code's draw (`DrawIsOnItsStreams`).

The miss bound is stated in the shared draw vocabulary (`Definitions.Core.Draw`): its bound is the escape of an index
draw, `t` indices uniform and independent over `W` (`Verity.Draw.law t W`), from the tickets of `B` (`slotsOf`), which is
`(1 − w(B)/W)^t`, `w(B)` the tickets `B`'s tiles hold (`weightOf`). The claims' one-stage designs
(`Claims.Sampling.OneStage`) all draw without replacement or by Bernoulli, and none is this draw; the index draw is the
law sampled proofs' audit laws instantiate.

* `RunDrawMisses`: over `t` independent uniform streams, the draw is made and names no tile of `B` with probability at
  most `(Verity.Draw.law t W).escape (slotsOf tk B)`. A stream that runs out refuses the run (`plan`'s `ValueError`), so
  it misses nothing.
* `RunDrawMissesAtDelta`: the bound at the credit's ε (`epsilon t DELTA`, `creditAtAssumption`'s): a set of tiles that
  holds some tickets and at least that share of them (`Heavy`) is missed with probability at most δ = 1/100.

The streams are modelled as independent and uniform, as `Properties.Pouw.Credit.WorkDrawMisses` models the window
draw's: the run derives draw `j`'s as `stream key ["u", j] (drawBytes tk)`, its index distinct for each draw, and this
is that derivation in the random-oracle model. They add no named assumption: the probability is over the streams'
bytes. The key is `derive(source, "verity/pouw/pearl-plus-run-draw/v1", {run})` on the run's commitment, fixed before
the coin (`after-coin`).

**The bad tiles under tolerant acceptance.** The verifier counts debit on the drawn tiles only, and a drawn tile whose
credited row is over its cap is credited nothing rather than rejecting the run. So `B` is the tiles whose check fails
otherwise (an opening, a word, a per-row rule, the served y): one of them drawn rejects the run, and the bound is the
probability that none is. An over-cap tile is outside `B`: when drawn it lowers the credit, and the weight of the over-cap
tiles the draw leaves out enters the credit through the sampled over-cap rate, a deviation of the draw's hit count from
its mean rather than a miss, which this bound does not state.
-/

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus
open scoped ENNReal

/-- The tickets `pearlDraw` draws among, those above 0, each with its position in the run's list `tk`. -/
def positive (tk : List ((ℕ × ℕ) × ℕ)) : List (ℕ × ℕ) := (tk.map (·.2)).zipIdx.filter (·.1 > 0)

/-- Their running sums, as `weightedDraw` forms them. -/
def cumulative (tk : List ((ℕ × ℕ) × ℕ)) : List ℕ :=
  (((positive tk).map (·.1)).foldl (fun (acc : Array ℕ × ℕ) w => (acc.1.push (acc.2 + w), acc.2 + w)) (#[], 0)).1.toList

/-- The run's ticket total `W`: each draw's uniform integer is below it. -/
def ticketTotal (tk : List ((ℕ × ℕ) × ℕ)) : ℕ := (tk.map (·.2)).sum

/-- The bytes of each draw's stream the draw reads (`DRAW_WORDS` words at `uniform`'s width for `W`). -/
def drawBytes (tk : List ((ℕ × ℕ) × ℕ)) : ℕ := DRAW_WORDS * Flock.Draw.width (ticketTotal tk)

/-- The tile an integer `r < W` names: the ticket whose running sum first passes `r` (`bisectRight`), as `draw` reads
its position back into `tk`. -/
def tileAt (tk : List ((ℕ × ℕ) × ℕ)) (r : ℕ) : ℕ × ℕ :=
  (tk[((positive tk).getD (bisectRight (cumulative tk) r) (0, 0)).2]!).1

/-- **The run's draw on the streams `s`**: draw `j` the tile at its stream's uniform integer below `W`, in order; none
when a stream runs out before an integer is accepted. -/
def drawOn (tk : List ((ℕ × ℕ) × ℕ)) (t : ℕ) (s : ℕ → ByteArray) : Option (List (ℕ × ℕ)) :=
  (List.range t).mapM fun j => (FlockSoundness.Randomness.uniformOn (ticketTotal tk) (s j)).map (tileAt tk)

/-- **The run's draw is `drawOn` on its key's streams**: wherever the run's tickets hold some weight, `draw` (`pearlDraw`'s
`weightedDraw`) names the tiles `drawOn` names on the key's streams, draw `j`'s at index `("u", j)` and `drawBytes` long,
and throws exactly where `drawOn` gives none. -/
def DrawIsOnItsStreams : Prop :=
  ∀ (passes : List Pass) (key : ByteArray) (t : ℕ), 0 < ticketTotal (tickets passes) →
    (draw passes key t).toOption =
      drawOn (tickets passes) t fun j =>
        FlockSoundness.Randomness.stream key [.str "u", .int j] (drawBytes (tickets passes))

/-- `t` streams of `L` uniform bytes each, as `drawOn` reads them (`Properties.SampledProofs.bytesOf`); none past `t`. -/
def streamsOf {t L : ℕ} (g : Fin t → Fin L → Fin 256) (j : ℕ) : ByteArray :=
  if h : j < t then Properties.SampledProofs.bytesOf (g ⟨j, h⟩) else ByteArray.empty

/-- The draw on the streams `g` is made, and names no tile of `B`. -/
def MissesOn (tk : List ((ℕ × ℕ) × ℕ)) (t : ℕ) (B : Finset (ℕ × ℕ)) (g : Fin t → Fin (drawBytes tk) → Fin 256) :
    Prop :=
  ∃ d, drawOn tk t (streamsOf g) = some d ∧ ∀ x ∈ B, x ∉ d

/-- `B`'s tickets: the integers below `W` whose tile is in `B`. -/
def slotsOf (tk : List ((ℕ × ℕ) × ℕ)) (B : Finset (ℕ × ℕ)) : Finset (Fin (ticketTotal tk)) :=
  Finset.univ.filter fun r => tileAt tk r.val ∈ B

/-- **Pearl+'s run draw misses `B` no more often than `t` uniform tickets do**: for every list of tiles and tickets `tk`
(the run's is `tickets passes`) whose total `W` is positive, every number of draws `t` and every set `B` of tiles, the
draw over `t` independent uniform streams of `drawBytes tk` bytes is made and names no tile of `B` with probability at
most the escape of `t` independent uniform indices below `W` from `B`'s tickets. -/
def RunDrawMisses : Prop :=
  ∀ (tk : List ((ℕ × ℕ) × ℕ)) (t : ℕ) (B : Finset (ℕ × ℕ)) [NeZero (ticketTotal tk)],
    Verity.Game.prCoin (MissesOn tk t B) ≤ (Verity.Draw.law t (ticketTotal tk)).escape (slotsOf tk B)

/-- The tickets `B`'s tiles hold. -/
def weightOf (tk : List ((ℕ × ℕ) × ℕ)) (B : Finset (ℕ × ℕ)) : ℕ := ((tk.filter fun e => e.1 ∈ B).map (·.2)).sum

/-- **`B` is heavy at `t` draws**: its tiles hold some tickets, and at least the share `epsilon t DELTA` of them, the ε
the credit takes off (`creditAtAssumption`). -/
def Heavy (tk : List ((ℕ × ℕ) × ℕ)) (t : ℕ) (B : Finset (ℕ × ℕ)) : Prop :=
  0 < weightOf tk B ∧ epsilon t DELTA * (ticketTotal tk : ℚ) ≤ (weightOf tk B : ℚ)

/-- **The run's draw misses a heavy set with probability at most δ**: for every list of tiles and tickets `tk`, number of
draws `t` and set `B` of tiles heavy at `t` (`Heavy`), the draw over `t` independent uniform streams is made and names no
tile of `B` with probability at most δ = 1/100 (`DELTA`), the δ the run's credit is taken at. -/
def RunDrawMissesAtDelta : Prop :=
  ∀ (tk : List ((ℕ × ℕ) × ℕ)) (t : ℕ) (B : Finset (ℕ × ℕ)), Heavy tk t B →
    Verity.Game.prCoin (MissesOn tk t B) ≤ ENNReal.ofReal (DELTA : ℝ)

end Properties.Pouw.PearlPlus
