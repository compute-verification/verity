module

public import Definitions.Pouw.Confidence

/-!
# Transcripts, acceptance and sets of actual circuits (draft)

The smallest definitions the draft statements of `Definitions.Draft.ActualCircuit` need that the specification lacks;
each would belong in `Core`. Every statement of the draft has one shape, read through `ResourceAccounting.Sound`: if
the verifier at the leaves `V` accepts in an epoch then, except with probability `δ` over its challenges, the actual
circuit lies in a set `𝒞 τ` whose members have a stated property, where `τ` is the verifier's transcript.

An *epoch* is a span of ticks, `[ts, td)` of PoUW's verdict (`ResourceAccounting.PouwWork.Verdict`), and in a
monitoring period one execution block's span (`Verity.MonitoringPeriod.executionSpan`): the nearest notions the
specification has. Nothing here needs an epoch type of its own.

* `Transcript`: the messages a verifier holds in a run, each as tick, leaf, cell, width and value, the type
  `Actual.Run.view` returns and `Sound` and `Composition` read unnamed.
* `Accepted`: the event that the verifier at `V`, deciding `acc` on its transcript, accepts in a run.
* `Lies`: the event that a run's actual circuit lies in `𝒞 τ`, for `τ` the transcript of the verifier at `V`; the set
  may depend on the messages (a credited count, a number the verifier computes).
* `ForStrategies`: a property of every strategy that `Sound` quantifies over, so that a premise stated of one strategy
  (PoUW's A2, A2b, A3) is taken of each of them.
-/

@[expose] public section

namespace Draft.ActualCircuit

open ResourceAccounting

/-- **A transcript**: the messages a verifier holds in a run, each as its tick, leaf, cell, width and value. -/
abbrev Transcript (L : Type*) : Type _ := Set (ℕ × L × ℕ × ℕ × ℕ)

/-- **Acceptance**: the verifier at the leaves `V`, deciding `acc` on its transcript, accepts in the run `R`. -/
def Accepted {L : Type*} (V : Finset L) (acc : Transcript L → Prop) (R : Actual.Run L) : Prop :=
  acc (R.view V)

/-- **The actual circuit lies in `𝒞 τ`**: at challenge `r`, the actual circuit of the run `S r` is a member of the set
`𝒞` gives at the transcript of the verifier at `V`. -/
def Lies {Ω L : Type*} (V : Finset L) (𝒞 : Transcript L → Set (Placed L)) (S : Ω → Actual.Run L) (r : Ω) : Prop :=
  (S r).A ∈ 𝒞 ((S r).view V)

/-- **For every strategy `Sound` reads**: `Q` holds of every strategy on the challenges `Ω`, causal from every tick,
with runs in `𝓕`, whose inputs off `V` don't depend on the challenges. -/
def ForStrategies {L : Type*} [LinearOrder L] (Ω : Type) (𝓕 : Set (Actual.Run L)) (V : Finset L)
    (Q : (Ω → Actual.Run L) → Prop) : Prop :=
  ∀ S : Ω → Actual.Run L, (∀ r, S r ∈ 𝓕) → CausalFrom S → (∀ r r', (S r).inputs (↑V)ᶜ = (S r').inputs (↑V)ᶜ) → Q S

end Draft.ActualCircuit
