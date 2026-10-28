import Definitions.Core.Draw
import Definitions.Core.Assumptions.Challenger

/-!
# The sample from a challenger's coin

`SampleAfterRecorded`: under the challenger's premises (`Definitions.Core.Assumptions.Challenger`), the sample a
verifier takes with a coin it read from the challenger's log misses a set of units fixed before that coin as often as a
uniform draw of the law misses it, at most the law's `miss K` for a set of `K` units or more. It is the live-coin step of
a sampled audit: the faulty proof blocks are fixed by the commitments the coin samples, which the challenger recorded
before it flipped the coin, so the prover can't steer them away from the sample.
-/

namespace Verity.Guarantees

open Verity Verity.Assumptions Verity.Challenger

/-- **The sample from a challenger's coin misses a set fixed before it at most as often as the law allows.** For every
sample law `L` over units `U`, run of a challenger flipping `n` of the law's coins with log `log`, verifier's coins
`vc`, coin index `i`, set of units `bad` fixed by what the log holds before coin `i` (`Log.before`: the commitments
that coin samples, and the earlier coins), and `K`: if the challenger logs its coins (`ChallengerLogsCoins`), flips
each only after what its log holds before it is recorded (`FlipsAfterRecorded`), and the verifier reads its coins from
the log (`VerifierReadsLog`), then the probability over the challenger's flips that the verifier's coin `i` exists and
its sample (`L.draw`) misses every unit of a set of at least `K` units fixed before it is at most `L.miss K`. -/
def SampleAfterRecorded : Prop :=
  ∀ {U R : Type} (L : Law U) (n : Nat) (log : (Fin n → L.Ω) → Log R L.Ω) (vc : (Fin n → L.Ω) → List L.Ω)
    (i : Nat) (bad : Log R L.Ω → Finset U) (K : Nat),
    ChallengerLogsCoins log → FlipsAfterRecorded log → VerifierReadsLog vc log →
      Game.prCoin (fun ω => ∃ c, (vc ω)[i]? = some c ∧ K ≤ (bad ((log ω).before i)).card ∧
        Disjoint (L.draw c) (bad ((log ω).before i))) ≤ L.miss K

end Verity.Guarantees
