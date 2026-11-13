module

public import Proofs.Properties.Verity.SampledTwo.Within
public import Proofs.Properties.Verity.SampledTwo.Fits
public import Proofs.Properties.Verity.SampledTwo.Reached
public import Proofs.Properties.Verity.SampledTwo.RuleTwo
public import Proofs.Flock.SpotFlock.RunsAs.RoundTwoOf
public import Proofs.Flock.SpotFlock.RunsAs.SecondStageOf
public import Proofs.Properties.Verity.SampledTwo.RecordNames
public import Proofs.Properties.Verity.SampledTwo.RunChecks
public import Proofs.Properties.Verity.SampledTwo.AtStmt

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
* `RecordsAt`: the interior record registers no value (`Round2.registersNone_of_two`: where the design evaluates, the
  interior registration registers none), and both records name their registrations' names (`recordsHeldAt_of`: an
  accepted run's registration checks have no code, `RunChecks.checks_of_accepted`, and a check with no code names its
  registration's names, `RecordNames.recordOk_of_check`). For that, the held copies are asked to be the ones `heldOf`
  builds (`HeldOfAt`) and each registration to list each port once (`PortsDistinctAt`), which R3-ports doesn't
  check.

`playAccepts_form` is `PlayAccepts` from what is left. Nothing here turns the form on, and none of the premises says
it is off.

Most of the premises name the one session of record `A`, so each is false at a case map whose accepted plays carry two
boundary records. `acceptsRun_form_at` and `reachedAccepts_form_at` ask them only of the accepted plays at the
statement `x`, at the case map `atStmt C x` (`AtStmt.lean`), and conclude at `C`.
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
/-- **Each accepted play's held copies are the ones `heldOf` builds** from its input and interior. -/
def HeldOfAt (A : Live₂) (C : CaseOf) : Prop :=
  ∀ o i it, FormAccepts C o → inputOn C o = some i → i.interior = some it →
    ∃ u, SpotFlock.HeldOf.heldOf i it u = some A.h

/-- **Each accepted play's registrations list each port once**, the boundary's and the interior's. -/
def PortsDistinctAt (C : CaseOf) : Prop :=
  ∀ o i it, FormAccepts C o → inputOn C o = some i → i.interior = some it →
    (i.params.verifier.ports.map (·.1)).toList.Nodup ∧ (it.params.verifier.ports.map (·.1)).toList.Nodup

/-- **An accepted play's records name their registrations' names**: its run's registration checks have no code
(`RunChecks.checks_of_accepted`), and each names its registration's names (`RecordNames.recordOk_of_check`), which
are the held copies' (`HeldOf.namesHeld_of`). -/
theorem recordsHeldAt_of {A : Live₂} {C : CaseOf} (hof : HeldOfAt A C) (hports : PortsDistinctAt C)
    (hread : RecordsRead C) : RecordsHeldAt A C := by
  intro o i it x m₁ hA hi hit hx hm₁
  obtain ⟨u, hu⟩ := hof o i it hA hi hit
  obtain ⟨hb, hn⟩ := SpotFlock.HeldOf.namesHeld_of hu
  obtain ⟨j, hj⟩ := acceptedTwo_of_formAccepts (ruleIsTwo C) hA hi hit
  obtain ⟨⟨v, hv, hvn⟩, w, hw, hwn⟩ := SpotFlock.RunChecks.checks_of_accepted hj
  obtain ⟨hpb, hpi⟩ := hports o i it hA hi hit
  obtain ⟨hjx, hjm⟩ := hread o i it x m₁ hA hi hit hx hm₁
  have hnv : (v.ports.map (·.1)).toList.Nodup := by
    have := congrArg SpotFlock.Names.ports hvn
    simp only [SpotFlock.HeldOf.namesOf] at this
    rw [this]
    exact hpb
  have hnw : (w.ports.map (·.1)).toList.Nodup := by
    have := congrArg SpotFlock.Names.ports hwn
    simp only [SpotFlock.HeldOf.namesOf] at this
    rw [this]
    exact hpi
  have rb := SpotFlock.RecordNames.recordOk_of_check hv hnv
  have ri := SpotFlock.RecordNames.recordOk_of_check hw hnw
  rw [hvn, ← hb] at rb
  rw [hwn, ← hn] at ri
  show SpotFlock.recordsOk A.h x m₁ = true
  unfold SpotFlock.recordsOk SpotFlock.decode
  rw [hjx, hjm]
  simp only [Option.map_some, Bool.and_eq_true]
  exact ⟨rb, ri⟩

theorem recordsAt_of_held {A : Live₂} {C : CaseOf} (htwo : TwoHolds A.h) (hrec : RecordsHeldAt A C)
    (hread : RecordsRead C) : RecordsAt A C := by
  intro o i it x m₁ hA hi hit hx hm₁
  have hrh : RecordsHeld A.h x m₁ := hrec o i it x m₁ hA hi hit hx hm₁
  exact ⟨hrh, SpotFlock.RunsAs.Round2.registersNone_of_two htwo (hread o i it x m₁ hA hi hit hx hm₁).2 hrh⟩

/-- **The form's accept is a play SPOT[Flock] accepts** (`PlayAccepts` at the form): `Within.playAccepts_of_run'` with
the rule two-stage by the form's parser (`ruleIsTwo`), the second stage's hand-off proved at the held copies
(`RunsAs.stmtOk_of_check`), the names distinct and the interior record registering no value where the design evaluates
(`Round2.namesDistinct_of_two`, `recordsAt_of_held`), and the records naming their registrations' names
(`recordsHeldAt_of`). -/
theorem playAccepts_form (A : Live₂) (C : CaseOf) (S : SessionReading)
    (P' : Verity.Exchange → ℕ → Claims.Prover Unit) (hheld : HeldAt A C) (htwo : TwoHolds A.h)
    (hof : HeldOfAt A C) (hports : PortsDistinctAt C) (hread : RecordsRead C) (hst : StmtsRead A C S)
    (hrounds : RoundsHeldAt A C S) (hscope : ScopeHeldAt A C S) (hpin : PinHeldAt A C S)
    (hsess : SessionsRun A C S P') :
    PlayAccepts A C S P' :=
  playAccepts_of_run' A C S P' (ruleIsTwo C) hheld htwo (SpotFlock.RunsAs.Round2.namesDistinct_of_two htwo)
    (recordsAt_of_held htwo (recordsHeldAt_of hof hports hread) hread) hread (SpotFlock.RunsAs.stmtOk_of_check A.h) hst
    hrounds hscope hpin hsess

/-- **The form's accept is the run's accept** (`AcceptsRun` at the form): where the form's check accepts a play, every
run of `SpotFlock.V` on coins agreeing with what the check read, whose messages are what the check read, accepts. -/
theorem acceptsRun_form (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hsm : SessionMsgs A C S P') (hheld : HeldAt A C) (htwo : TwoHolds A.h) (hof : HeldOfAt A C)
    (hports : PortsDistinctAt C) (hread : RecordsRead C) (hst : StmtsRead A C S) (hrounds : RoundsHeldAt A C S)
    (hscope : ScopeHeldAt A C S) (hpin : PinHeldAt A C S) (hsess : SessionsRun A C S P') :
    AcceptsRun A x (formReading A C S dev) (FormAccepts C) :=
  acceptsRun_of' A x C S dev P' (sourcesFit A C)
    (playAccepts_form A C S P' hheld htwo hof hports hread hst hrounds hscope hpin hsess) hsm

/-- **The form's accept is the run's accept, with every message reached** (`ReachedAccepts` at the form): where the
form's check accepts a play, every run of `SpotFlock.V` on coins agreeing with what the check read, whose messages
are what the check read wherever the run has one, accepts and has all the verifier's `k` messages. -/
theorem reachedAccepts_form (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hsm : SessionMsgs A C S P') (hheld : HeldAt A C) (htwo : TwoHolds A.h) (hof : HeldOfAt A C)
    (hports : PortsDistinctAt C) (hread : RecordsRead C) (hst : StmtsRead A C S) (hrounds : RoundsHeldAt A C S)
    (hscope : ScopeHeldAt A C S) (hpin : PinHeldAt A C S) (hsess : SessionsRun A C S P') :
    ReachedAccepts A x (formReading A C S dev) (FormAccepts C) :=
  reachedAccepts A x C S dev P' (sourcesFit A C)
    (playAccepts_form A C S P' hheld htwo hof hports hread hst hrounds hscope hpin hsess) hsm

/-- **The cases at `x` carry the play's commitment where `C`'s do.** -/
theorem caseCommits_at {C : CaseOf} (x : Msg) (h : CaseCommits C) : CaseCommits (atStmt C x) :=
  fun o k hk => h o k (atStmt_some hk).1

/-- **`ReachedAccepts` at `C` from `ReachedAccepts` at the cases at `x`** (as `acceptsRun_at`). -/
theorem reachedAccepts_at {A : Live₂} {x : Msg} {C : CaseOf} {S : SessionReading}
    {dev : List Verity.Release → ℕ → Option FlockDeploy.Coin}
    (h : ReachedAccepts A x (formReading A (atStmt C x) S dev) (FormAccepts (atStmt C x))) :
    ReachedAccepts A x (formReading A C S dev) (FormAccepts C) := by
  intro o hA hx Ω P ω r hc hm
  obtain ⟨hA', hin⟩ := at_of_stmt hA hx
  obtain ⟨e₁, e₂, e₃, e₄⟩ := formReading_at (A := A) (S := S) (dev := dev) hin
  refine h o hA' (e₁.trans hx) Ω P ω r ?_ ?_
  · rw [e₂, e₄]; exact hc
  · rw [e₃]; exact hm

/-- **The form's accept is the run's accept, its premises asked at the plays at `x`**: `acceptsRun_form` at the cases
at `x` (`atStmt C x`), whose accepted plays are `C`'s at the statement `x`, the only ones `AcceptsRun` at `x` looks at
(`acceptsRun_at`). -/
theorem acceptsRun_form_at (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hsm : SessionMsgs A (atStmt C x) S P') (hheld : HeldAt A (atStmt C x)) (htwo : TwoHolds A.h)
    (hof : HeldOfAt A (atStmt C x)) (hports : PortsDistinctAt (atStmt C x)) (hread : RecordsRead (atStmt C x))
    (hst : StmtsRead A (atStmt C x) S) (hrounds : RoundsHeldAt A (atStmt C x) S)
    (hscope : ScopeHeldAt A (atStmt C x) S) (hpin : PinHeldAt A (atStmt C x) S)
    (hsess : SessionsRun A (atStmt C x) S P') :
    AcceptsRun A x (formReading A C S dev) (FormAccepts C) :=
  acceptsRun_at
    (acceptsRun_form A x (atStmt C x) S dev P' hsm hheld htwo hof hports hread hst hrounds hscope hpin hsess)

/-- **The form's accept is the run's accept, with every message reached, its premises asked at the plays at `x`**:
`reachedAccepts_form` at the cases at `x` (`reachedAccepts_at`). -/
theorem reachedAccepts_form_at (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hsm : SessionMsgs A (atStmt C x) S P') (hheld : HeldAt A (atStmt C x)) (htwo : TwoHolds A.h)
    (hof : HeldOfAt A (atStmt C x)) (hports : PortsDistinctAt (atStmt C x)) (hread : RecordsRead (atStmt C x))
    (hst : StmtsRead A (atStmt C x) S) (hrounds : RoundsHeldAt A (atStmt C x) S)
    (hscope : ScopeHeldAt A (atStmt C x) S) (hpin : PinHeldAt A (atStmt C x) S)
    (hsess : SessionsRun A (atStmt C x) S P') :
    ReachedAccepts A x (formReading A C S dev) (FormAccepts C) :=
  reachedAccepts_at
    (reachedAccepts_form A x (atStmt C x) S dev P' hsm hheld htwo hof hports hread hst hrounds hscope hpin hsess)

end Verity.SampledTwo
