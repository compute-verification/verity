module

public import Proofs.Properties.Verity.SampledTwo.Within
public import Proofs.Properties.Verity.SampledTwo.Fits
public import Proofs.Properties.Verity.SampledTwo.Reached
public import Proofs.Properties.Verity.SampledTwo.RuleTwo
public import Proofs.Flock.SpotFlock.RunsAs.RoundTwoOf
public import Proofs.Flock.SpotFlock.RunsAs.SecondStageOf

/-!
# The form's accept is the run's accept: the pieces joined

`acceptsRun_form`: `AcceptsRun` at the two-stage form's reading and acceptance, from the run's accept theorem
(`OfRun.playAccepts_of_run`, with a stage's sample reading no byte past its source: `Within.playAccepts_of_run'`) and
an accepted run's bytes within its coins (`Fits.sourcesFit`). `reachedAccepts_form`: the same with the messages asked
only where the run has one, and every message reached (`Reached.reachedAccepts`).

The premises are `OfRun.lean`'s but five, each a named `Prop` about the plays the form's check accepts, and
`SessionMsgs` (`Accepts.lean`): the sessions' messages the accepted prover sends are those the check read. The five
that are proved and so not asked:

* `ReadsWithin` (`Within.playAccepts_of_run'`);
* `RuleIsTwo` (`RuleTwo.ruleIsTwo`: the form's parser returns only a two-stage rule);
* `RunsAs.RoundTwo` (`RunsAs.stmtOk_of_check`: the second stage's check gives the model's statement check at every
  held copy);
* `NamesDistinct` (`Round2.namesDistinct_of_two`: where the design evaluates, the templates' names are distinct);
* the half of `RecordsAt` that says the interior record registers no value (`Round2.registersNone_of_two`: where the
  design evaluates, the interior registration registers none), so the records are asked only to name their
  registrations' names (`RecordsHeldAt`).

`playAccepts_form` is `PlayAccepts` from what is left. Nothing here turns the form on, and none of the premises says
it is off.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open SpotFlock.RunsAs.Round2 (TwoHolds RecordsHeld)

/-- **An accepted play's records name their registrations' names** (`RecordsHeld`): `RecordsAt` less the half the
design decides. -/
def RecordsHeldAt (A : Live₂) (C : CaseOf) : Prop :=
  ∀ o i it x m₁, FormAccepts C o → inputOn C o = some i → i.interior = some it → recordMsg i.record = some x →
    recordMsg it.record = some m₁ → RecordsHeld A.h x m₁

/-- **Where the design evaluates, records that name their registrations' names are `RecordsAt`**: the interior record,
read back as the record the run held (`RecordsRead`), registers no value (`Round2.registersNone_of_two`). -/
theorem recordsAt_of_held {A : Live₂} {C : CaseOf} (htwo : TwoHolds A.h) (hrec : RecordsHeldAt A C)
    (hread : RecordsRead C) : RecordsAt A C := by
  intro o i it x m₁ hA hi hit hx hm₁
  have hrh : RecordsHeld A.h x m₁ := hrec o i it x m₁ hA hi hit hx hm₁
  exact ⟨hrh, SpotFlock.RunsAs.Round2.registersNone_of_two htwo (hread o i it x m₁ hA hi hit hx hm₁).2 hrh⟩

/-- **The form's accept is a play SPOT[Flock] accepts** (`PlayAccepts` at the form): `Within.playAccepts_of_run'` with
the rule two-stage by the form's parser (`ruleIsTwo`), the second stage's hand-off proved at the held copies
(`RunsAs.stmtOk_of_check`), and the names distinct and the interior record registering no value where the design
evaluates (`Round2.namesDistinct_of_two`, `recordsAt_of_held`). -/
theorem playAccepts_form (A : Live₂) (C : CaseOf) (S : SessionReading)
    (P' : Verity.Exchange → ℕ → Claims.Prover Unit) (hheld : HeldAt A C) (htwo : TwoHolds A.h)
    (hrec : RecordsHeldAt A C) (hread : RecordsRead C) (hst : StmtsRead A C S) (hrounds : RoundsHeldAt A C S)
    (hscope : ScopeHeldAt A C S) (hpin : PinHeldAt A C S) (hsess : SessionsRun A C S P') :
    PlayAccepts A C S P' :=
  playAccepts_of_run' A C S P' (ruleIsTwo C) hheld htwo (SpotFlock.RunsAs.Round2.namesDistinct_of_two htwo)
    (recordsAt_of_held htwo hrec hread) hread (SpotFlock.RunsAs.stmtOk_of_check A.h) hst hrounds hscope hpin hsess

/-- **The form's accept is the run's accept** (`AcceptsRun` at the form): where the form's check accepts a play, every
run of `SpotFlock.V` on coins agreeing with what the check read, whose messages are what the check read, accepts. -/
theorem acceptsRun_form (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hsm : SessionMsgs A C S P') (hheld : HeldAt A C) (htwo : TwoHolds A.h) (hrec : RecordsHeldAt A C)
    (hread : RecordsRead C) (hst : StmtsRead A C S) (hrounds : RoundsHeldAt A C S) (hscope : ScopeHeldAt A C S)
    (hpin : PinHeldAt A C S) (hsess : SessionsRun A C S P') :
    AcceptsRun A x (formReading A C S dev) (FormAccepts C) :=
  acceptsRun_of' A x C S dev P' (sourcesFit A C)
    (playAccepts_form A C S P' hheld htwo hrec hread hst hrounds hscope hpin hsess) hsm

/-- **The form's accept is the run's accept, with every message reached** (`ReachedAccepts` at the form): where the
form's check accepts a play, every run of `SpotFlock.V` on coins agreeing with what the check read, whose messages
are what the check read wherever the run has one, accepts and has all the verifier's `k` messages. -/
theorem reachedAccepts_form (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hsm : SessionMsgs A C S P') (hheld : HeldAt A C) (htwo : TwoHolds A.h) (hrec : RecordsHeldAt A C)
    (hread : RecordsRead C) (hst : StmtsRead A C S) (hrounds : RoundsHeldAt A C S) (hscope : ScopeHeldAt A C S)
    (hpin : PinHeldAt A C S) (hsess : SessionsRun A C S P') :
    ReachedAccepts A x (formReading A C S dev) (FormAccepts C) :=
  reachedAccepts A x C S dev P' (sourcesFit A C)
    (playAccepts_form A C S P' hheld htwo hrec hread hst hrounds hscope hpin hsess) hsm

end Verity.SampledTwo
