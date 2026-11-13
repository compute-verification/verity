module

public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.SpotFlock.Honest
public import Proofs.Flock.Claims.Sampling.Select.Code
public import Properties.Challenger
public import Proofs.Flock.SpotFlock.RunsAs.Statements

/-!
# A two-stage run the running verifier accepts is a play SPOT[Flock]'s two-stage verifier accepts

The two-stage guarantees are stated of `SpotFlock.V h`, a verifier over held copies `h` on coins that are its own; the
verifier that runs is `Spot.Run.run`. This module states, as `Prop`s, what ties the two: an accepted two-stage run of
`Run.run` (`AcceptedTwo`) is, at every held copy `h` that agrees with what the run reads (`HeldMatches`), a play of
`SpotFlock.V h` that it accepts, sending the run's records and then each session's messages (`proverSeq`,
`SpotRunAcceptsPlay`), and the play's coins are the
challenger's bytes the run read, in the order the guarantee assumes (`PlayCoins`).

`AcceptedTwo` is stated of `Run.run` at `form := true`, the two-stage form allowed, as a caller that decides last by
the switch (`SecondStage.FORM`) runs it; at the default the switch is off and the second stage's check refuses every
two-stage run.

What the run does not decide, each a named premise of `SpotRunAcceptsPlay`:

* `RoundTwo`: round 2's check, passing, gives the model's statement check (`stmtOk`) where the clauses round 2 does not
  make hold; `RunsAs.Round2`'s `stmtOk_of_check`;
* `SessionsAccept`: the session statements' message states the statements round 2 checked (`StmtsAre`), each with
  the clauses on its own bytes (`RoundsHeld`, `ScopeHeld`, `PinHeld`), and each template's session is a run Flock's
  masked verifier accepts. `Run.run` reads the sessions' verdicts as JSON (`Service.checkVerdicts`) and holds no
  statement or transcript of a session, so this is what the verdict of record (`flock-verify`'s) says;
* `TwoHolds`: the held copies are in the two-stage scope the model decides (`two`: `scopeOk`, `filesOk`,
  `templateOk`), which `Run.run` does not check;
* `NamesDistinct`, `RecordsHeld` and `RegistersNone`: the templates' names are distinct, the run's records name their
  registrations' names, and round 1's record registers no value.

These and the session clauses are `RunsAs.Round2`'s definitions (`Statements`), opened here.
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Claims
open Lean (Json)
open Round2 (core OwnAgrees NamesDistinct StmtsAre TwoHolds RecordsHeld RegistersNone RoundsHeld ScopeHeld PinHeld)

/-! ## The run -/

/-- **A two-stage run `Run.run` accepted**: with the two-stage form allowed (`form := true`), it decided `j` with the
symbol `accept`, under a two-stage rule, with the interior registration `it`. -/
structure AcceptedTwo (i : Spot.Run.Input) (it : Spot.Run.Interior) (j : Json) : Prop where
  run : Spot.Run.run i true = .ok j
  accept : Spot.getD j "symbol" = .str "accept"
  two : ∃ grain replay k, i.rule = .twoStage grain replay k
  interior : i.interior = some it

/-! ## The held copies the run reads -/

/-- **The held copies agree with what the run reads**: the registration the first stage samples is the one `Run.run`
passes `Select.select` (the verifier's copies with the sampling design and work table it expanded, the resolved rule, the closure
map); the second stage's cells and counts are those `Run.run` computes; the design's links, wires and rows are the
ones round 2 derives from the replay block's Definition; the templates' parts, at the replay sample the first stage
selects on the run's bytes, are those the second stage's check reads; the reference
headers and reads are the auditor's params'; and the second stage reads at most `stage` coins' bytes. Every other field
of `h` (the descriptor, the members' units, the reference statements, the frame) is the auditor's registered copy,
which the run does not read. -/
structure HeldMatches (i : Spot.Run.Input) (it : Spot.Run.Interior) (h : Held) : Prop where
  first : ∃ n rule law workTable resolved,
    Spot.nat? (Spot.getD i.record "population") = some n ∧
    i.rule.over n (Spot.Select.strataSizes i.params.verifier.program i.params.verifier.query) = some rule ∧
    rule.lawOver n = .ok law ∧
    i.rule.tableOver i.params.verifier.program i.params.verifier.query i.params.verifier.workTable = .ok workTable ∧
    Spot.Select.resolve i.rule i.params.verifier.program i.params.verifier.query = .ok resolved ∧
    h.reg.first = ⟨{ i.params.verifier with law, workTable }, resolved, i.closure⟩
  ns : Spot.Select.replayCells it.replay it.params.verifier.query = .ok h.reg.ns
  ks : ∃ grain replay k, i.rule = .twoStage grain replay k ∧ ∃ ids,
    Spot.Select.templateIds it.params.verifier.query = .ok ids ∧ k.over ids h.reg.ns = .ok h.reg.ks
  links : h.links =
    Spot.SecondStage.replayLinks i.params.verifier.program.prog it.replay.prog.programFn i.record
  wires : h.wires = Spot.SecondStage.replayWires i.params.verifier.program.prog it.replay.prog.programFn
  rows : h.rows = Spot.SecondStage.replayRows i.params.verifier.program.prog it.replay.prog.programFn
  parts : ∀ draw rest read strata, Spot.Select.select h.reg.first.rule h.reg.first.program h.reg.first.query
      h.reg.first.table h.reg.first.closure i.source = .ok (draw, rest, read) →
    Spot.Partition.strata it.params.verifier.program it.params.verifier.query = .ok strata →
    Spot.SecondStage.parts rest.size strata = some (partsAt h rest.size)
  refs : refsOf h = it.circuits.map fun (t, _, hd, _) => (t, hd)
  refReads : refRegisteredOf h = it.circuits.map fun (t, _, _, r) => (t, r)


/-! ## The second stage's check: the hand-off

`RoundTwo` is `RunsAs.Round2.StmtOkOfCheck` at `h` with `RegistersNone` for its inputs premise, so `RunsAs.RoundTwoOf`'s
`stmtOk_of_check` proves it at every `h`. -/

/-- **Round 2's check gives the model's statement check** (part 3, `Round2.StmtOkOfCheck` at `h`): where what the
verifier holds of the run is the model's and the statements round 2 checked are the session statements', round 2's
check with the form allowed reporting no code makes the model's statement check pass (`stmtOk`), given the clauses
round 2 does not make. -/
def RoundTwo (h : Held) : Prop :=
  ∀ (own : Spot.SecondStage.Own) (ss : Array Json) (x m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h)
    (st : Roots) (rec draw : Json),
    decode x = some st → jsonMsg m₁ = some rec → secondDraw h c₀ c₁ = some draw → OwnAgrees h own st rec draw c₀ →
      StmtsAre h m₂ ss → RegistersNone rec → TwoHolds h → NamesDistinct h → RecordsHeld h x m₁ →
        RoundsHeld h m₂ → ScopeHeld h m₂ → PinHeld h m₂ → (Spot.SecondStage.check true own (some ss)).codes = #[] →
          stmtOk h x m₁ m₂ c₀ c₁ = true

/-! ## The sessions -/

/-- **The sessions accept** (part 4's premise): the session statements' message `m₂` states the statements `ss`, each
with its template's rounds, in scope and its reference statement but for the run's fields (`RoundsHeld`, `ScopeHeld`,
`PinHeld`), and Flock's masked verifier accepts template `t`'s session, the prover `P' t` at the statement `m₂` states
for it on the coins `r' t`. `Run.run` holds the sessions' verdicts of record only as JSON (`Service.checkVerdicts`);
this is what a verdict of record that accepts says. -/
def SessionsAccept (h : Held) (H : Msg → Msg) (ss : Array Json) (m₂ : Msg) (P' : ℕ → Prover Unit)
    (r' : ∀ t, Fin ((FlockDeploy.V true).k (stmtAt h m₂ t)) → FlockDeploy.Coin) : Prop :=
  StmtsAre h m₂ ss ∧ RoundsHeld h m₂ ∧ ScopeHeld h m₂ ∧ PinHeld h m₂ ∧
    ∀ t < h.ms.length, (Claims.run H (P' t) (FlockDeploy.V true) (stmtAt h m₂ t) () (r' t)).out = .acc

/-! ## The coins -/

/-- **`ξ'` starts with `ξ`'s first `R` bytes.** -/
structure AgreesTo (ξ ξ' : ByteArray) (R : ℕ) : Prop where
  size_le : R ≤ ξ'.size
  extract_eq : ξ'.extract 0 R = ξ.extract 0 R

/-- **The play's coins carry the bytes the run's samples read**: where the first stage, on the bytes the run holds
(`i.source`), reads `read` bytes, round 0's coin's bytes start with those `read` bytes; and where the second stage, on
the interior's bytes (`it.source`), reads `read` bytes, the second stage's coins' bytes, coin by coin, start with
those. These are the bytes `Run.coinsCode` holds to the challenger's released coins; a sample reads no other byte
(`select_read`, `selectSecond_read`), and nothing about the rest of a coin is assumed. `RunCoinsFromDevice` says the
bytes read are the device's coins released after the boundary record's digest and after the interior record's, or a
break. -/
def PlayCoins (h : Held) (i : Spot.Run.Input) (it : Spot.Run.Interior) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) :
    Prop :=
  (∀ draw proved read, Spot.Select.select h.reg.first.rule h.reg.first.program h.reg.first.query h.reg.first.table
      h.reg.first.closure i.source = .ok (draw, proved, read) → AgreesTo i.source (src c₀) read) ∧
    ∀ draw read, Spot.Select.selectSecond (h.reg.sample (src c₀)) h.reg.ns h.reg.ks it.source = .ok (draw, read) →
      AgreesTo it.source (stageSrc c₁) read

/-- **`b` is the start of `ξ`.** -/
def Starts (b ξ : ByteArray) : Prop := b.size ≤ ξ.size ∧ ξ.extract 0 b.size = b

/-- **A stage's bytes are the device's challenges, or a break**: `read` starts the challenges of the family `label` that
a run of the device `dev` under the verifier's key `key`, one its own ledger opens on the window `w`, released after
`after`; or the model's finder outputs a break from the evidence `coins` (`Properties.Challenger.Break`: a forgery
under the key, or a SHA-512 collision among what the reader and the device hash). -/
def FromDevice (dev : Properties.Challenger.History) (key : Spot.Challenger.ChallengerKey) (w label : String)
    (after : ByteArray) (coins : Json) (read : ByteArray) : Prop :=
  (∃ s ∈ dev.runs key.pub, Properties.Challenger.OpensOn dev key.pub w s.id ∧
      Starts read (s.released label after)) ∨
    Properties.Challenger.Break dev key label coins

/-- **The run's challenges come from the device, in the order the two-stage guarantee assumes**: at an accepted
two-stage run, the bytes the first stage read are the device's challenges of the run's family (`Input.family`) released
after the boundary record's digest, and those the second stage read the device's challenges of the interior's family
(`Interior.family`) released after the interior record's digest, each on the window its verifier holds and under the
run's one key (`Input.key`); or a break. The key, the window and the family are the verifier's own: the run's check
of its challenges (`Run.coinsCode`) refuses a stage that read a byte where it holds none of them, and no evidence names
them. A stage that read no byte carries no evidence. The sessions' challenges are not read by `Run.run`. -/
def RunCoinsFromDevice : Prop :=
  ∀ (i : Spot.Run.Input) (it : Spot.Run.Interior) (j : Json) (dev : Properties.Challenger.History),
    AcceptedTwo i it j →
    ∃ (read₁ read₂ : ℕ),
      (∃ resolved workTable draw rest, Spot.Select.select resolved i.params.verifier.program
        i.params.verifier.query workTable i.closure i.source = .ok (draw, rest, read₁)) ∧
      (∃ sample ns ks draw, Spot.Select.selectSecond sample ns ks it.source = .ok (draw, read₂)) ∧
      (0 < (i.source.extract 0 read₁).size → ∃ key w label d, i.key = some key ∧
        i.params.verifier.windowRoot = some w ∧ i.family = some label ∧
        Spot.Registration.digest i.record = .ok d ∧
        FromDevice dev key w label d i.coins (i.source.extract 0 read₁)) ∧
      (0 < (it.source.extract 0 read₂).size → ∃ key w label d, i.key = some key ∧
        it.params.verifier.windowRoot = some w ∧ it.family = some label ∧
        Spot.Registration.digest it.record = .ok d ∧
        FromDevice dev key w label d it.coins (it.source.extract 0 read₂))

/-! ## The top statement -/

/-- **A two-stage run `Run.run` accepts is a play `SpotFlock.V h` accepts** (the inner statement: the form's check,
`VStar.Recursive.SampledTwo.checkBlobsAtWith`, computes the sessions' verdicts and then runs `Run.run`). At every held
copy `h` that agrees with what the run reads: if `Run.run` accepted the two-stage run `i` with interior `it`
(`AcceptedTwo`), `x` and `m₁` are the boundary and interior records' bytes, round 2's check gives the model's checks
(`RoundTwo`, part 3) and Flock accepts each template's session at the statement `m₂` states for it (`SessionsAccept`,
part 4), then on every round-0 coin and second-stage coins that start with the bytes the run read (`PlayCoins`),
`SpotFlock.V h` accepts the play that sends `m₁`, then `m₂`, then each session's messages (`proverSeq`), each session
on its own coins (`seqCoins`).

The premises the run does not decide, each said here: the design evaluates (`TwoHolds`), the templates' names are
distinct (`NamesDistinct`), the run's records name their registrations' names (`RecordsHeld`, which `Run.run`'s
registration step checks and this statement does not yet derive from it), and round 1's record registers no value
(`RegistersNone`, a premise on the run).

`AcceptedTwo` is at `form := true`: the run with the two-stage form allowed. -/
def SpotRunAcceptsPlay : Prop :=
  ∀ (i : Spot.Run.Input) (it : Spot.Run.Interior) (j : Json) (h : Held) (H : Msg → Msg) (x m₁ m₂ : Msg)
    (P' : ℕ → Prover Unit) (r' : ∀ t, Fin ((FlockDeploy.V true).k (stmtAt h m₂ t)) → FlockDeploy.Coin)
    (c₀ : FlockDeploy.Coin) (c₁ : Stage h),
    AcceptedTwo i it j → HeldMatches i it h → TwoHolds h → NamesDistinct h → RecordsHeld h x m₁ →
    RegistersNone it.record → jsonMsg x = some i.record → jsonMsg m₁ = some it.record →
    RoundTwo h → SessionsAccept h H ((it.statements).getD #[]) m₂ P' r' → PlayCoins h i it c₀ c₁ →
      (Claims.run H (proverSeq h [] m₁ m₂ P') (V h) x (fun _ => ()) (seqCoins h (x := x) c₀ c₁ r')).out = .acc

end SpotFlock.RunsAs
