module

public import Proofs.Properties.Verity.SampledTwo.ReadsOnly
public import Proofs.Flock.SpotFlock.RunsAs.OfCheck
public import Proofs.Flock.SpotFlock.RunsAs.Top
public import Proofs.Flock.Claims.Sampling.Select.Prefix

/-!
# The form's accept is a play SPOT[Flock] accepts, from the run's accept

`PlayAccepts` (`Accepts.lean`), the premise of `acceptsRun_of'` that says the proved verifier accepts the play the
form's check read, from the two theorems about the running verifier:

* `RunsAs.acceptedTwo_of_check`: where the form's check accepts at a two-stage rule, the run it made is an accepted
  two-stage run of `Spot.Run.run` with the form allowed (`acceptedTwo_of_formAccepts`, at a play);
* `RunsAs.spotRunAcceptsPlay`: such a run is a play `SpotFlock.V` accepts, at held copies that agree with what the
  run reads, on coins that start with the bytes the run read.

`playAccepts_of_run` joins them. What neither theorem decides is a named premise here, each stated at the plays the
form's check accepts and over the reading's own fields:

* `RuleIsTwo`: the parameters' rule is two-stage, which the parameters' parser requires;
* `HeldAt`: the held copies agree with what the run reads (`RunsAs.HeldMatches`); `TwoHolds` and `NamesDistinct` at
  the held copies;
* `RecordsAt`: the records name their registrations' names and the interior record registers no value;
* `RecordsRead`: each record's canonical bytes read back, by the proved verifier's reader, as the record the run held
  (proved in `AcceptsTop.lean`, `recordsRead`);
* `RoundTwo`: the second stage's statement check, passing, gives the model's;
* `StmtsRead`: the reading's session statements' message states the statements the run's second stage checked;
* `RoundsHeldAt`, `ScopeHeldAt`, `PinHeldAt`: each session statement has its template's stage count, is in scope, and
  is its template's reference statement but for the run's fields;
* `SessionsRun`: Flock's masked verifier accepts each session's prover at its statement, on coins agreeing with what
  the check read of them;
* `ReadsWithin`: a stage's sample reads no byte past its source.

The coins: the run's samples read only the bytes they report reading, and a coin that starts with a stage's bytes
agrees with them that far (`playCoins_of_starts`), so `SeqCoinsRead`'s coins are `RunsAs.PlayCoins`'s.

Nothing here turns the form on, and no premise says it is off. An earlier period's genuine challenger signature is
outside these statements: they speak of the run the check made, not of when its coins were signed.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open Lean (Json)
open SpotFlock.RunsAs (AcceptedTwo HeldMatches RoundTwo SessionsAccept PlayCoins AgreesTo)
open SpotFlock.RunsAs.Round2 (NamesDistinct StmtsAre TwoHolds RecordsHeld RegistersNone RoundsHeld ScopeHeld PinHeld)

/-! ## The premises -/

/-- **The parameters' rule is two-stage**: every case of a play decodes to parameters whose rule is two-stage. The
parameters' parser refuses any other rule. -/
def RuleIsTwo (C : CaseOf) : Prop :=
  ∀ o k p c e, C o = some k → Decoded k p c e → ∃ grain replay n, p.rule = .twoStage grain replay n

/-- **The held copies agree with what an accepted play's run reads** (`RunsAs.HeldMatches`). -/
def HeldAt (A : Live₂) (C : CaseOf) : Prop :=
  ∀ o i it, FormAccepts C o → inputOn C o = some i → i.interior = some it → HeldMatches i it A.h

/-- **An accepted play's records name their registrations' names, and the interior record registers no value**
(`RecordsHeld`, `RegistersNone`). -/
def RecordsAt (A : Live₂) (C : CaseOf) : Prop :=
  ∀ o i it x m₁, FormAccepts C o → inputOn C o = some i → i.interior = some it → recordMsg i.record = some x →
    recordMsg it.record = some m₁ → RecordsHeld A.h x m₁ ∧ RegistersNone it.record

/-- **An accepted play's records read back**: the proved verifier's reader (`SpotFlock.jsonMsg`) reads the boundary
record's canonical bytes as the record the run held, and the interior record's as the interior's. The form hands the
run each record as the proved verifier's parser reads its canonical text (`readBack`), so it holds at every case map
(`AcceptsTop.recordsRead`). -/
def RecordsRead (C : CaseOf) : Prop :=
  ∀ o i it x m₁, FormAccepts C o → inputOn C o = some i → i.interior = some it → recordMsg i.record = some x →
    recordMsg it.record = some m₁ → SpotFlock.jsonMsg x = some i.record ∧ SpotFlock.jsonMsg m₁ = some it.record

/-- **The reading's session statements are the ones the run's second stage checked**: at an accepted play the reading
has a session statements' message, and it states the statements the run of record's interior carries (`StmtsAre`). -/
def StmtsRead (A : Live₂) (C : CaseOf) (S : SessionReading) : Prop :=
  ∀ o i it, FormAccepts C o → inputOn C o = some i → i.interior = some it →
    ∃ m₂, S.stmts o = some m₂ ∧ StmtsAre A.h m₂ (it.statements.getD #[])

/-- **Each session statement has its template's stage count** (`RoundsHeld`), at the reading's statements. -/
def RoundsHeldAt (A : Live₂) (C : CaseOf) (S : SessionReading) : Prop :=
  ∀ o m₂, FormAccepts C o → S.stmts o = some m₂ → RoundsHeld A.h m₂

/-- **Each session statement is in scope** (`ScopeHeld`), at the reading's statements. -/
def ScopeHeldAt (A : Live₂) (C : CaseOf) (S : SessionReading) : Prop :=
  ∀ o m₂, FormAccepts C o → S.stmts o = some m₂ → ScopeHeld A.h m₂

/-- **Each session statement is its template's reference statement but for the run's fields** (`PinHeld`), at the
reading's statements. -/
def PinHeldAt (A : Live₂) (C : CaseOf) (S : SessionReading) : Prop :=
  ∀ o m₂, FormAccepts C o → S.stmts o = some m₂ → PinHeld A.h m₂

/-- **Flock accepts each session**: at an accepted play, on every sessions' coins agreeing with what the check read of
them, Flock's masked verifier accepts template `t`'s prover `P' o t` at the statement the reading states for it. -/
def SessionsRun (A : Live₂) (C : CaseOf) (S : SessionReading) (P' : Verity.Exchange → ℕ → Claims.Prover Unit) :
    Prop :=
  ∀ o m₂, FormAccepts C o → S.stmts o = some m₂ →
    ∀ r' : ∀ t, Fin ((FlockDeploy.V true).k (SpotFlock.stmtAt A.h m₂ t)) → FlockDeploy.Coin,
      (∀ t < A.h.ms.length, ∀ j (hj : j < (FlockDeploy.V true).k (SpotFlock.stmtAt A.h m₂ t)) c,
        S.coin o t j = some c → S.agree o t j c (r' t ⟨j, hj⟩)) →
      ∀ t < A.h.ms.length,
        (Claims.run FlockDeploy.sha512 (P' o t) (FlockDeploy.V true) (SpotFlock.stmtAt A.h m₂ t) () (r' t)).out =
          .acc

/-- **A stage's sample reads no byte past its source**: where the first stage's sample, or the second's, succeeds on
a source and reports reading `read` bytes, the source has that many. A sample takes its bytes in order and throws where
the source runs out. -/
def ReadsWithin (h : SpotFlock.Held) : Prop :=
  (∀ ξ draw proved read, Spot.Select.select h.reg.first.rule h.reg.first.program h.reg.first.query
      h.reg.first.table h.reg.first.closure ξ = .ok (draw, proved, read) → read ≤ ξ.size) ∧
    ∀ (sample : Array ℕ) ξ draw read,
      Spot.Select.selectSecond sample h.reg.ns h.reg.ks ξ = .ok (draw, read) → read ≤ ξ.size

/-! ## The coins -/

/-- **Bytes that start with `ξ` agree with `ξ` on its first `R` bytes**, for `R` within `ξ`. -/
theorem agreesTo_of_append {ξ b u : ByteArray} {R : ℕ} (hb : b = ξ ++ u) (hR : R ≤ ξ.size) : AgreesTo ξ b R := by
  subst hb
  refine ⟨?_, Properties.Spot.SecurityProofs.extract_append_of_le hR⟩
  rw [ByteArray.size_append]
  omega

/-- **Coins that start with the run's bytes carry the bytes its samples read** (`RunsAs.PlayCoins`), where a sample
reads no byte past its source. -/
theorem playCoins_of_starts {h : SpotFlock.Held} {i : Spot.Run.Input} {it : Spot.Run.Interior}
    {c₀ : FlockDeploy.Coin} {c₁ : SpotFlock.Stage h} (hw : ReadsWithin h)
    (h0 : ∃ u, SpotFlock.src c₀ = i.source ++ u) (h1 : ∃ u, SpotFlock.stageSrc c₁ = it.source ++ u) :
    PlayCoins h i it c₀ c₁ := by
  obtain ⟨u₀, hu₀⟩ := h0
  obtain ⟨u₁, hu₁⟩ := h1
  unfold SpotFlock.RunsAs.PlayCoins
  exact And.intro (fun draw proved read hs => agreesTo_of_append hu₀ (hw.1 _ draw proved read hs))
    (fun draw read hs => agreesTo_of_append hu₁ (hw.2 _ _ draw read hs))

/-! ## The run -/

/-- **An accepted play's run of record is an accepted two-stage run** (`RunsAs.acceptedTwo_of_check` at a play): where
the form's check accepts the play at a two-stage rule, `Spot.Run.run` with the form allowed accepted the run of
record's input `i`, with its interior `it`. -/
theorem acceptedTwo_of_formAccepts {C : CaseOf} {o : Verity.Exchange} {i : Spot.Run.Input}
    {it : Spot.Run.Interior} (hrule : RuleIsTwo C) (hA : FormAccepts C o) (hi : inputOn C o = some i)
    (hit : i.interior = some it) : ∃ run, AcceptedTwo i it run := by
  obtain ⟨k, hk, p, cm, e, pub, prof, hdec, hacc⟩ := hA
  obtain ⟨-, c₂, ξ₂, r, i', it', run, hin, hat, -⟩ :=
    SpotFlock.RunsAs.acceptedTwo_of_check (hrule o k p cm e hk hdec) hacc
  have hik : inputOfCase k = some i' := by
    obtain ⟨h1, h2, h3⟩ := hdec
    unfold inputOfCase
    rw [h1, h2, h3]
    simp only [bind, Except.bind, hin]
    rfl
  have hi' : inputOn C o = some i' := by
    unfold inputOn
    rw [hk]
    exact hik
  obtain rfl : i = i' := Option.some.inj (hi.symm.trans hi')
  obtain rfl : it = it' := Option.some.inj (hit.symm.trans hat.interior)
  exact ⟨run, hat⟩

/-! ## The accept -/

/-- **The form's accept is a play SPOT[Flock] accepts** (`PlayAccepts`), from the run's accept
(`RunsAs.spotRunAcceptsPlay`) at the run the form's check made (`acceptedTwo_of_formAccepts`), under the premises this
file names. -/
theorem playAccepts_of_run (A : Live₂) (C : CaseOf) (S : SessionReading)
    (P' : Verity.Exchange → ℕ → Claims.Prover Unit) (hrule : RuleIsTwo C) (hheld : HeldAt A C)
    (htwo : TwoHolds A.h) (hnd : NamesDistinct A.h) (hrec : RecordsAt A C) (hread : RecordsRead C)
    (hr2 : RoundTwo A.h) (hst : StmtsRead A C S) (hrounds : RoundsHeldAt A C S) (hscope : ScopeHeldAt A C S)
    (hpin : PinHeldAt A C S) (hsess : SessionsRun A C S P') (hw : ReadsWithin A.h) : PlayAccepts A C S P' := by
  intro o i it x m₁ hA hi hit hx hm₁
  obtain ⟨m₂, hm₂, hsa⟩ := hst o i it hA hi hit
  have hrd : RoundsHeld A.h m₂ := hrounds o m₂ hA hm₂
  refine ⟨m₂, hm₂, hrd, fun c₀ c₁ r' hcoins => ?_⟩
  obtain ⟨hc0, hc1, hcs⟩ := hcoins
  obtain ⟨run, hat⟩ := acceptedTwo_of_formAccepts hrule hA hi hit
  obtain ⟨hrh, hrn⟩ := hrec o i it x m₁ hA hi hit hx hm₁
  obtain ⟨hjx, hjm⟩ := hread o i it x m₁ hA hi hit hx hm₁
  have hS : SessionsAccept A.h FlockDeploy.sha512 (it.statements.getD #[]) m₂ (P' o) r' := by
    unfold SpotFlock.RunsAs.SessionsAccept
    exact ⟨hsa, hrd, hscope o m₂ hA hm₂, hpin o m₂ hA hm₂, hsess o m₂ hA hm₂ r' hcs⟩
  exact SpotFlock.RunsAs.spotRunAcceptsPlay i it run A.h FlockDeploy.sha512 x m₁ m₂ (P' o) r' c₀ c₁ hat
    (hheld o i it hA hi hit) htwo hnd hrh hrn hjx hjm hr2 hS (playCoins_of_starts hw hc0 hc1)

/-- **The form's accept is the run's accept, from the run's accept theorem**: `acceptsRun_of'` with `PlayAccepts`
from `playAccepts_of_run`. What remains of `acceptsRun_of'`'s own premises: an accepted run read at most the coins it
has (`SourcesFit`) and the sessions' messages are those the check read (`SessionMsgs`). -/
theorem acceptsRun_of_run (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hfit : SourcesFit A C) (hsm : SessionMsgs A C S P') (hrule : RuleIsTwo C) (hheld : HeldAt A C)
    (htwo : TwoHolds A.h) (hnd : NamesDistinct A.h) (hrec : RecordsAt A C) (hread : RecordsRead C)
    (hr2 : RoundTwo A.h) (hst : StmtsRead A C S) (hrounds : RoundsHeldAt A C S) (hscope : ScopeHeldAt A C S)
    (hpin : PinHeldAt A C S) (hsess : SessionsRun A C S P') (hw : ReadsWithin A.h) :
    AcceptsRun A x (formReading A C S dev) (FormAccepts C) :=
  acceptsRun_of' A x C S dev P' hfit
    (playAccepts_of_run A C S P' hrule hheld htwo hnd hrec hread hr2 hst hrounds hscope hpin hsess hw) hsm

end Verity.SampledTwo
