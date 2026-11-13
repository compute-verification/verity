module

public import Proofs.Properties.Verity.SampledTwo.Within
public import Proofs.Properties.Verity.SampledTwo.Fits
public import Proofs.Properties.Verity.SampledTwo.Reached
public import Proofs.Properties.Verity.SampledTwo.RuleTwo
public import Proofs.Flock.SpotFlock.RunsAs.RoundTwoOf

/-!
# The form's accept is the run's accept: the pieces joined

`acceptsRun_form`: `AcceptsRun` at the two-stage form's reading and acceptance, from the run's accept theorem
(`OfRun.playAccepts_of_run`, with a stage's sample reading no byte past its source: `Within.playAccepts_of_run'`) and
an accepted run's bytes within its coins (`Fits.sourcesFit`). `reachedAccepts_form`: the same with the messages asked
only where the run has one, and every message reached (`Reached.reachedAccepts`).

The premises are `OfRun.lean`'s but three, each a named `Prop` about the plays the form's check accepts, and
`SessionMsgs` (`Accepts.lean`): the sessions' messages the accepted prover sends are those the check read. The three
that are proved and so not asked: `ReadsWithin` (`Within.playAccepts_of_run'`), `RuleIsTwo` (`RuleTwo.ruleIsTwo`: the
form's parser returns only a two-stage rule) and `RunsAs.RoundTwo` (`RunsAs.stmtOk_of_check`: the second stage's check
gives the model's statement check at every held copy). `playAccepts_form` is `PlayAccepts` from what is left. Nothing
here turns the form on, and none of the premises says it is off.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open SpotFlock.RunsAs.Round2 (NamesDistinct TwoHolds)

/-- **The form's accept is a play SPOT[Flock] accepts** (`PlayAccepts` at the form): `Within.playAccepts_of_run'` with
the rule two-stage by the form's parser (`ruleIsTwo`) and the second stage's hand-off proved at the held copies
(`RunsAs.stmtOk_of_check`). -/
theorem playAccepts_form (A : Live₂) (C : CaseOf) (S : SessionReading)
    (P' : Verity.Exchange → ℕ → Claims.Prover Unit) (hheld : HeldAt A C) (htwo : TwoHolds A.h)
    (hnd : NamesDistinct A.h) (hrec : RecordsAt A C) (hread : RecordsRead C) (hst : StmtsRead A C S)
    (hrounds : RoundsHeldAt A C S) (hscope : ScopeHeldAt A C S) (hpin : PinHeldAt A C S)
    (hsess : SessionsRun A C S P') : PlayAccepts A C S P' :=
  playAccepts_of_run' A C S P' (ruleIsTwo C) hheld htwo hnd hrec hread (SpotFlock.RunsAs.stmtOk_of_check A.h) hst
    hrounds hscope hpin hsess

/-- **The form's accept is the run's accept** (`AcceptsRun` at the form): where the form's check accepts a play, every
run of `SpotFlock.V` on coins agreeing with what the check read, whose messages are what the check read, accepts. -/
theorem acceptsRun_form (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hsm : SessionMsgs A C S P') (hheld : HeldAt A C) (htwo : TwoHolds A.h) (hnd : NamesDistinct A.h)
    (hrec : RecordsAt A C) (hread : RecordsRead C) (hst : StmtsRead A C S) (hrounds : RoundsHeldAt A C S)
    (hscope : ScopeHeldAt A C S) (hpin : PinHeldAt A C S) (hsess : SessionsRun A C S P') :
    AcceptsRun A x (formReading A C S dev) (FormAccepts C) :=
  acceptsRun_of' A x C S dev P' (sourcesFit A C)
    (playAccepts_form A C S P' hheld htwo hnd hrec hread hst hrounds hscope hpin hsess) hsm

/-- **The form's accept is the run's accept, with every message reached** (`ReachedAccepts` at the form): where the
form's check accepts a play, every run of `SpotFlock.V` on coins agreeing with what the check read, whose messages
are what the check read wherever the run has one, accepts and has all the verifier's `k` messages. -/
theorem reachedAccepts_form (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hsm : SessionMsgs A C S P') (hheld : HeldAt A C) (htwo : TwoHolds A.h) (hnd : NamesDistinct A.h)
    (hrec : RecordsAt A C) (hread : RecordsRead C) (hst : StmtsRead A C S) (hrounds : RoundsHeldAt A C S)
    (hscope : ScopeHeldAt A C S) (hpin : PinHeldAt A C S) (hsess : SessionsRun A C S P') :
    ReachedAccepts A x (formReading A C S dev) (FormAccepts C) :=
  reachedAccepts A x C S dev P' (sourcesFit A C)
    (playAccepts_form A C S P' hheld htwo hnd hrec hread hst hrounds hscope hpin hsess) hsm

end Verity.SampledTwo
