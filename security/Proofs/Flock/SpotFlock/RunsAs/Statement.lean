module

public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.SpotFlock.Honest
public import Proofs.Flock.Claims.Sampling.Select.Code
public import Properties.Challenger

/-!
# A two-stage run the running verifier accepts is a play SPOT[Flock]'s two-stage verifier accepts

The two-stage guarantees are stated of `SpotFlock.V h`, a verifier over held copies `h` on coins that are its own; the
verifier that runs is `Spot.Run.run`. This module states, as `Prop`s, what ties the two: an accepted two-stage run of
`Run.run` (`AcceptedTwo`) is, at every held copy `h` that agrees with what the run reads (`HeldMatches`), a play of
`SpotFlock.V h` that it accepts, sending the run's records and then each session's messages (`proverSeq`,
`RunAcceptsPlay`), and the play's coins are the
challenger's bytes the run read, in the order the guarantee assumes (`PlayCoins`).

`Run.run` calls round 2's check with the form's switch (`SecondStage.FORM`), which is off, so no two-stage run is
accepted today. The statement is of `Run.run` as it is, and holds unchanged the day the switch is on: its proof reads
round 2's check only through `SecondStage.check b own ss` having no codes, which at every `b` gives the check with the
form allowed having none.

What the run does not decide, each a named premise of `RunAcceptsPlay`:

* `RoundTwo`: round 2's check, passing, gives the model's statement check (`stmtOk`) where the registration's clauses
  and those on the session statements' own bytes hold; proved of `SecondStage.check` on its own;
* `SessionsAccept`: the session statements' message states the statements round 2 checked (`StmtsAre`), each with
  the clauses on its own bytes (`SessionsHeld`), and each template's session is a run Flock's masked verifier accepts. `Run.run` reads the sessions'
  verdicts as JSON (`Service.checkVerdicts`) and holds no statement or transcript of a session, so this is what the
  verdict of record (`flock-verify`'s) says, taken as given;
* `ScopeHolds`: the held copies are in the two-stage scope the model decides (`two`: `scopeOk`, `filesOk`,
  `templateOk`), a property of the registration that `Run.run` does not check;
* `Fits`: the challenger's bytes the run holds for each stage fit the model's coins for it (round 0's one coin, and
  the second stage's `stage` coins).
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Claims
open Lean (Json)

/-! ## The run -/

/-- **A two-stage run `Run.run` accepted**: it decided `j` with the symbol `accept`, under a two-stage rule, with the
interior registration `it`. -/
structure AcceptedTwo (i : Spot.Run.Input) (it : Spot.Run.Interior) (j : Json) : Prop where
  run : Spot.Run.run i = .ok j
  accept : Spot.getD j "symbol" = .str "accept"
  two : ∃ grain replay k, i.rule = .twoStage grain replay k
  interior : i.interior = some it

/-! ## The held copies the run reads -/

/-- **The held copies agree with what the run reads**: the registration the first stage samples is the one `Run.run`
passes `Select.select` (the verifier's copies with the law and work table it expanded, the resolved rule, the closure
map); the second stage's cells and counts are those `Run.run` computes; the design's links, wires and rows are the
ones round 2 derives from the replay block's Definition; the templates' parts are those round 2 reads; the reference
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
  parts : ∀ strata q, Spot.Partition.strata it.params.verifier.program it.params.verifier.query = .ok strata →
    Spot.SecondStage.parts q strata = some (partsAt h q)
  refs : refsOf h = it.circuits.map fun (t, _, hd, _) => (t, hd)
  refReads : refRegisteredOf h = it.circuits.map fun (t, _, _, r) => (t, r)

/-- **The held copies are in the two-stage scope the model decides** (`two`: `scopeOk`, `filesOk`, `templateOk`).
`Run.run` makes none of these checks; they read the members' circuits and the reference statements, which it does not
hold. -/
def ScopeHolds (h : Held) : Prop := ∀ s, (two h s).isSome = true

/-- **Each stage's bytes fit the model's coins**: the challenger's bytes the run holds for the first stage
(`i.source`) are at most `L`, the model's round-0 coin's, and those for the second stage (`it.source`) at most
`stage · L`, its `stage` rounds' coins'. `Run.run` bounds neither. -/
def Fits (i : Spot.Run.Input) (it : Spot.Run.Interior) (h : Held) : Prop :=
  i.source.size ≤ L ∧ it.source.size ≤ h.stage * L

/-! ## Round 2: the hand-off -/

/-- **What the verifier holds of the run is the model's** (`OwnAgrees`): `own` is `ownOf h st rec draw c₀` in every
field `ownOf` sets; the fields it leaves at their defaults (the reported reads, the verified statements, the block's
returns, the supplied and fixed values) may be anything. -/
def OwnAgrees (h : Held) (own : Spot.SecondStage.Own) (st : Roots) (rec draw : Json) (c₀ : FlockDeploy.Coin) :
    Prop :=
  { own with reported := none, verified := #[], returns := #[], supplied := #[], fixed := #[] } =
    ownOf h st rec draw c₀

/-- **The session statements' message states `ss`**: `m₂` states one session statement a template, and template
`t`'s decodes and reads as the statement round 2 holds at position `t` (`stmtJsonOf`). -/
def StmtsAre (h : Held) (m₂ : Msg) (ss : Array Json) : Prop :=
  ∃ xs, stmtsOf m₂ = some xs ∧ xs.length = h.ms.length ∧ ∀ t < h.ms.length, ∃ w,
    FlockDeploy.decode (xs.getD t (h.xS₀ t)) = some w ∧
      stmtJsonOf h t w = Spot.SecondStage.Stmt.ofJson? (ss.getD t .null)

/-- **The registration's clauses hold at the run's statement**: `x` parses, and its partition and two-stage design
evaluate (`parse`, `partition`, `two`); round 2 makes none of these checks. -/
def RegHolds (h : Held) (x : Msg) : Prop :=
  ∃ s, parse h x = some s ∧ (partition h s).isSome = true ∧ (two h s).isSome = true

/-- **The clauses on each session statement's own bytes hold**: template `t`'s, as `m₂` states it, has `k₀ t` rounds, is
in scope (`sessionOk`) and is its reference statement but for the run's fields (`pinOk`); round 2 never reads these
bytes. -/
def SessionsHeld (h : Held) (m₂ : Msg) : Prop :=
  ∃ xs, stmtsOf m₂ = some xs ∧ ∀ t < h.ms.length,
    (FlockDeploy.V true).k (xs.getD t (h.xS₀ t)) = k₀ h t ∧ sessionOk h t (xs.getD t (h.xS₀ t)) = true ∧
      ∃ w, FlockDeploy.decode (xs.getD t (h.xS₀ t)) = some w ∧ pinOk h t w = true

/-- **Round 2's check gives the model's statement check** (part 3, `stmtOk_of_check`'s statement): where what the
verifier holds of the run is the model's (`OwnAgrees`), the session statements' message states the statements round 2
checked (`StmtsAre`), the registration's clauses hold (`RegHolds`) and so do those on the session statements' own bytes
(`SessionsHeld`), round 2's check with the form allowed reporting no code makes the model's statement check pass
(`stmtOk`). -/
def RoundTwo (h : Held) : Prop :=
  ∀ {own : Spot.SecondStage.Own} {ss : Array Json} {x m₁ m₂ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    {st : Roots} {rec draw : Json},
    decode x = some st → jsonMsg m₁ = some rec → secondDraw h c₀ c₁ = some draw →
    OwnAgrees h own st rec draw c₀ → StmtsAre h m₂ ss → RegHolds h x → SessionsHeld h m₂ →
    (Spot.SecondStage.check true own (some ss)).codes = #[] →
    stmtOk h x m₁ m₂ c₀ c₁ = true

/-! ## The sessions -/

/-- **The sessions accept** (part 4's premise): the session statements' message `m₂` states the statements `ss` holds,
one a template, each with the clauses on its own bytes (`SessionsHeld`), and Flock's masked verifier accepts template
`t`'s session, the prover `P' t` at the statement `m₂` states for it on the coins `r' t`. `Run.run` holds the sessions'
verdicts of record only as JSON (`Service.checkVerdicts`); this is what a verdict of record that accepts says. -/
def SessionsAccept (h : Held) (H : Msg → Msg) (ss : Array Json) (m₂ : Msg) (P' : ℕ → Prover Unit)
    (r' : ∀ t, Fin ((FlockDeploy.V true).k (stmtAt h m₂ t)) → FlockDeploy.Coin) : Prop :=
  StmtsAre h m₂ ss ∧ SessionsHeld h m₂ ∧
    ∀ t < h.ms.length, (Claims.run H (P' t) (FlockDeploy.V true) (stmtAt h m₂ t) () (r' t)).out = .acc

/-! ## The coins -/

/-- **The play's coins are the bytes the run read**: round 0's coin is the challenger's bytes the first stage read
(`i.source`) and then zeros, and the second stage's coins, coin by coin, those of the interior's (`it.source`) and then
zeros. `RunCoinsFromDevice` says those bytes are the device's coins released after the boundary record's
digest and after the interior record's, or a break. -/
def PlayCoins (h : Held) (i : Spot.Run.Input) (it : Spot.Run.Interior) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) :
    Prop :=
  src c₀ = i.source ++ ⟨Array.replicate (L - i.source.size) 0⟩ ∧
    stageSrc c₁ = it.source ++ ⟨Array.replicate (h.stage * L - it.source.size) 0⟩

/-- **`b` is the start of `ξ`.** -/
def Starts (b ξ : ByteArray) : Prop := b.size ≤ ξ.size ∧ ξ.extract 0 b.size = b

/-- **The evidence `drawCoins` reads of a stage's coin evidence**: the evidence without the key it carries. -/
def drawEvidence : Json → Json
  | .obj kvs => .obj (kvs.erase "key")
  | j => j

/-- **The run's coin check reads the challenger's draw under the verifier's key `key`** (the hypothesis the verifier's
key in `Challenger.coinsReason` discharges): where `coinsReason` finds no reason against the evidence `coins` for the
window `w` and the bytes `read` released after `after`, `drawCoins` under `key` reads, from the evidence without its
key, coins of some family that `read` starts. Today `coinsReason` takes the key from the evidence, so this holds only
at the key the evidence carries. -/
def CheckReadsDraw (key : Spot.Challenger.ChallengerKey) : Prop :=
  ∀ (coins : Json) (w : String) (after read : ByteArray), Spot.Challenger.coinsReason coins w after read = none →
    ∃ label ξ, Spot.Challenger.drawCoins key w label after (drawEvidence coins) = .ok ξ ∧ Starts read ξ

/-- **A stage's bytes are the device's coins, or a break**: `read` starts the coins of some family that a run of the
device `dev` under `key`, one its own ledger opens on the window `w`, released after `after`; or the model's finder
outputs a break from the evidence (`Properties.Challenger.Break`: a forgery under the key, or a SHA-512 collision
among what the reader and the device hash). -/
def FromDevice (dev : Properties.Challenger.History) (key : Spot.Challenger.ChallengerKey) (w : String)
    (after : ByteArray) (coins : Json) (read : ByteArray) : Prop :=
  ∃ label, (∃ s ∈ dev.runs key.pub, Properties.Challenger.OpensOn dev key.pub w s.id ∧
      Starts read (s.released label after)) ∨
    Properties.Challenger.Break dev key label (drawEvidence coins)

/-- **The run's coins come from the device, in the order the two-stage guarantee assumes**: at an accepted two-stage
run, where the coin check reads the challenger's draw under the verifier's key (`CheckReadsDraw`), the bytes the first
stage read are the device's coins released after the boundary record's digest, and those the second stage read the
device's coins released after the interior record's digest, each on the window its verifier holds; or a break. A stage
that read no byte carries no evidence. The sessions' coins are not read by `Run.run`. -/
def RunCoinsFromDevice (key : Spot.Challenger.ChallengerKey) : Prop :=
  ∀ (i : Spot.Run.Input) (it : Spot.Run.Interior) (j : Json) (dev : Properties.Challenger.History),
    AcceptedTwo i it j → CheckReadsDraw key →
    ∃ (read₁ read₂ : ℕ),
      (∃ resolved workTable draw rest, Spot.Select.select resolved i.params.verifier.program
        i.params.verifier.query workTable i.closure i.source = .ok (draw, rest, read₁)) ∧
      (∃ sample ns ks draw, Spot.Select.selectSecond sample ns ks it.source = .ok (draw, read₂)) ∧
      (0 < (i.source.extract 0 read₁).size → ∃ w d, i.params.verifier.windowRoot = some w ∧
        Spot.Registration.digest i.record = .ok d ∧ FromDevice dev key w d i.coins (i.source.extract 0 read₁)) ∧
      (0 < (it.source.extract 0 read₂).size → ∃ w d, it.params.verifier.windowRoot = some w ∧
        Spot.Registration.digest it.record = .ok d ∧ FromDevice dev key w d it.coins (it.source.extract 0 read₂))

/-! ## The top statement -/

/-- **A two-stage run `Run.run` accepts is a play `SpotFlock.V h` accepts.** At every held copy `h` that agrees with
what the run reads, is in the model's scope and whose stages fit the model's coins (`Fits`): if `Run.run` accepted the
two-stage run `i` with interior `it` (`AcceptedTwo`), `x` and `m₁` are the boundary and interior records' bytes, round
2's check gives the model's checks (`RoundTwo`, part 3) and Flock accepts each template's session at the statement `m₂`
states for it (`SessionsAccept`, part 4), then there are coins, round 0's and the second stage's the bytes the run read
(`PlayCoins`), on which `SpotFlock.V h` accepts the play that sends `m₁`, then `m₂`, then each session's messages
(`proverSeq`), each session on its own coins (`seqCoins`). -/
def RunAcceptsPlay : Prop :=
  ∀ (i : Spot.Run.Input) (it : Spot.Run.Interior) (j : Json) (h : Held) (H : Msg → Msg) (x m₁ m₂ : Msg)
    (P' : ℕ → Prover Unit) (r' : ∀ t, Fin ((FlockDeploy.V true).k (stmtAt h m₂ t)) → FlockDeploy.Coin),
    AcceptedTwo i it j → HeldMatches i it h → ScopeHolds h → Fits i it h →
    jsonMsg x = some i.record → jsonMsg m₁ = some it.record →
    RoundTwo h → SessionsAccept h H ((it.statements).getD #[]) m₂ P' r' →
    ∃ (c₀ : FlockDeploy.Coin) (c₁ : Stage h), PlayCoins h i it c₀ c₁ ∧
      (Claims.run H (proverSeq h [] m₁ m₂ P') (V h) x (fun _ => ()) (seqCoins h (x := x) c₀ c₁ r')).out = .acc

end SpotFlock.RunsAs
