module

public import Definitions.Pouw.Confidence
public import Definitions.Core.Interaction

/-!
# Views and sets of actual circuits (draft)

The smallest definitions the draft statements of `Definitions.Draft.ActualCircuit` need that the specification lacks;
each would belong in `Core`. Every statement of the draft has one shape, read through `ResourceAccounting.Sound`: if
the verifier at the leaves `V` accepts in an epoch then, except with probability `δ` over its challenges, the actual
circuit lies in a set whose members have a stated property; the set may depend on what the verifier holds.

An *epoch* is a span of ticks, `[ts, td)` of PoUW's verdict (`ResourceAccounting.PouwWork.Verdict`), and in a
monitoring period one execution block's span (`Verity.MonitoringPeriod.executionSpan`): the nearest notions the
specification has. Its ends are ticks of the auditor-trusted device that sends the challenges, and compute accounting
multiplies a rate by its length, so both are signed messages of the transcript; no protocol's verifier reads a tick
today (Pearl+'s reads none).

* `VerifierView`: the type of what a verifier holds in a run, its gates as tick, leaf, cell, width and value, the type
  `Actual.Run.view` returns and `Sound` and `Composition` read unnamed.
* `Lies`: the event that a run's actual circuit lies in `𝒞 v`, for `v` the view of the verifier at `V`.
* `CompiledVerifier`, `CompiledAccepts`: acceptance as the compiled verifier's answer on the transcript (draft; belongs
  in Core; the Lean lead's to define). Every statement here reads acceptance as a predicate on the view; each protocol
  needs a lemma about its verifier's own code that `CompiledAccepts` implies that predicate
  (`Draft.ActualCircuit.CompiledImpliesAcceptsOn` for forced matrix multiplication).

**Every premise here is stated a challenge at a time.** `Sound` (`security/Definitions/Pouw/Confidence.lean`) ranges
over every strategy whose runs lie in a set of runs, so a premise about one strategy has nowhere to go: taken of every
strategy it is false for any real verifier (the strategy honest at every challenge but one has no block right at every
challenge, and the strategy constant at one run has every verdict read at the wrong challenge). Either `Sound` ranges
over a set of strategies, or every useful-work premise is restated a challenge at a time, reading the challenge from the
verifier's view and the work from the one run. This draft takes the second way; which way `main` takes is the Lean
lead's open question.
-/

@[expose] public section

namespace Draft.ActualCircuit

open ResourceAccounting

/-- **What a verifier holds** in a run: its gates, each as its tick, leaf, cell, width and value. -/
abbrev VerifierView (L : Type*) : Type _ := Set (ℕ × L × ℕ × ℕ × ℕ)

/-- **The actual circuit lies in `𝒞 v`**: at challenge `r`, the actual circuit of the run `S r` is a member of the set
`𝒞` gives at the view of the verifier at `V`. -/
def Lies {Ω L : Type*} (V : Finset L) (𝒞 : VerifierView L → Set (Placed L)) (S : Ω → Actual.Run L) (r : Ω) : Prop :=
  (S r).A ∈ 𝒞 ((S r).view V)

/-- **A compiled verifier** (draft; belongs in Core; the Lean lead's to define): its executable semantics as a function
`answer` of the transcript, the messages it reads in order (`Claims.Msg`), and the epoch's ends `ts` and `td`, both
signed messages, returning its verdict; and `transcript`, the messages a verifier's view holds, in the order the
verifier reads them. -/
structure CompiledVerifier (L : Type*) where
  answer : List Claims.Msg → ℕ → ℕ → Bool
  transcript : VerifierView L → List Claims.Msg

/-- **The compiled verifier accepts in the epoch `[ts, td)`** on the view `v` (draft; belongs in Core; the Lean lead's
to define): its answer on the transcript the view holds is `true`. -/
def CompiledAccepts {L : Type*} (cv : CompiledVerifier L) (ts td : ℕ) (v : VerifierView L) : Prop :=
  cv.answer (cv.transcript v) ts td = true

end Draft.ActualCircuit
