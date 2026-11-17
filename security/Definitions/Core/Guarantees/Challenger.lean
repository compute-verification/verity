module

public import Definitions.Core.Draw
public import Definitions.Core.Assumptions.Challenger
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-! `SampleAfterRecorded`: a sample taken with a coin from the challenger's log misses a set fixed before that coin as
often as a uniform sample does. -/

@[expose] public section

namespace Verity.Guarantees

open Verity Verity.Assumptions Verity.Challenger

/-- **The sample from a challenger's coin misses a set fixed before it at most as often as the sampling design allows.** For every
sampling design `L` over blocks `U`, run of a challenger flipping `n` of the sampling design's coins with log `log`, verifier's coins
`vc`, coin index `i`, set of blocks `bad` fixed by what the log holds before coin `i` (`Log.before`: the commitments
that coin samples, and the earlier coins), and `K`: if the challenger logs its coins (`ChallengerLogsCoins`), flips
each only after what its log holds before it is recorded (`FlipsAfterRecorded`), and the verifier reads its coins from
the log (`VerifierReadsLog`), then the probability over the challenger's flips that the verifier's coin `i` exists and
its sample (`L.draw`) misses every block of a set of at least `K` blocks fixed before it is at most `L.miss K`. -/
def SampleAfterRecorded : Prop :=
  ∀ {U R : Type} (L : SamplingDesign U) (n : Nat) (log : (Fin n → L.Ω) → Log R L.Ω) (vc : (Fin n → L.Ω) → List L.Ω)
    (i : Nat) (bad : Log R L.Ω → Finset U) (K : Nat),
    ChallengerLogsCoins log → FlipsAfterRecorded log → VerifierReadsLog vc log →
      Game.prCoin (fun ω => ∃ c, (vc ω)[i]? = some c ∧ K ≤ (bad ((log ω).before i)).card ∧
        Disjoint (L.draw c) (bad ((log ω).before i))) ≤ L.miss K

end Verity.Guarantees
