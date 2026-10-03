import Pouw.Protocol.PearlC.Tile

/-!
# The honest tiles' measured cap rate, completeness's one assumption (trusted; a `Prop` taken as a hypothesis)

The sampled audit's completeness (`CompleteSampled`, `Pouw.Protocol.PearlC.Completeness`) needs two facts about the
honest prover's drawn tile: its words are the reference's, and its replayed debit is within the cap. The first is
proved (`HonestWords`: every reference value is an FP32 word's). The second depends on the activations and the noise,
and nothing proves it; `HonestTileCapRate` assumes it at the rate a measurement gives.

It is **empirical**, not a hardness assumption: `count` is how many honest output tiles failed the per-tile cap among
`sample` honest tiles measured on served traffic (Llama-3.1-8B and Qwen3-8B, `art:<pending: FP8 served table>`). Both
are parameters, so the theorem that reads it (`completeSampledDevRev1K`) holds unchanged whatever the table's numbers.
-/

namespace Pouw.PearlC.Assumptions

open Pouw.PearlC

variable {Q R S : Type}

/-- **An honest drawn tile exceeds the cap at most `count/sample` of the time** (empirical: a measured count of honest
tiles that fail the check, not a hardness assumption). For every in-domain workload `U`, every activation input `act` in
the population `Served`, and every ticket count `w` proportional to the tiles' `W_ref`: over the oracle, the salt and
one work-weighted draw, the probability that the drawn tile is of a unit of the layout and its replayed debit on the
honest activations exceeds the cap (`TR.capOK` fails) is at most `count/sample`. `count` honest tiles failed the cap in
a sample of `sample` tiles of served traffic (`art:<pending: FP8 served table>`). -/
def HonestTileCapRate [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (TR : TileRules Q R S)
    (D : Layout → Prop) (Served : Workload → (ℕ → Codes) → Prop) (count sample : ℕ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ act : ℕ → Codes, Served U act →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ g < (TR.tiling U.layout).NT, (w g : ℝ) = κ * TR.Wref U.layout g) →
      pr (fun x : (Q → R) × S × Ticket w (TR.tiling U.layout).NT =>
          (TR.tiling U.layout).unit x.2.2.1 < U.layout.N ∧
            ¬ TR.capOK x.1 x.2.1 U x.2.2.1 (act ((TR.tiling U.layout).unit x.2.2.1)))
        ≤ (count : ℝ) / sample

end Pouw.PearlC.Assumptions
