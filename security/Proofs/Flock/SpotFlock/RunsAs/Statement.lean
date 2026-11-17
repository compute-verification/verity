module

public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.SpotFlock.Honest
public import Proofs.Flock.Claims.Sampling.Select.Code
public import Properties.Challenger
public import Proofs.Flock.SpotFlock.RunsAs.Statements
public import Spot.Run
public import Spot.Record

/-! A two-stage run `Run.run` accepts is a play `SpotFlock.V h` accepts (`SpotRunAcceptsPlay`), under named premises
the run does not decide. -/

@[expose] public section

namespace SpotFlock.RunsAs

open Claims
open Lean (Json)
open Round2 (core OwnAgrees NamesDistinct StmtsAre TwoHolds RecordsHeld RegistersNone RoundsHeld ScopeHeld PinHeld)

def runOwn (i : Spot.Run.Input) (it : Spot.Run.Interior) (sample : Array ℕ)
    (strata : Array (String × Array (ℕ × ℕ))) (draw : Json) : Spot.SecondStage.Own :=
  { sample, parts := Spot.SecondStage.parts sample.size strata, interior := it.record, boundary := i.record,
    drawn := (Spot.Draw.proved draw).map Int.toNat,
    links := Spot.SecondStage.replayLinks i.params.verifier.program.prog it.replay.prog.programFn i.record,
    reported := Spot.Record.sessionRegistered i.verdicts, verified := Spot.SecondStage.verifiedOf i.verdicts,
    refs := it.circuits.map fun (t, _, h, _) => (t, h), refReads := it.circuits.map fun (t, _, _, r) => (t, r),
    wires := Spot.SecondStage.replayWires i.params.verifier.program.prog it.replay.prog.programFn,
    rows := Spot.SecondStage.replayRows i.params.verifier.program.prog it.replay.prog.programFn,
    widths := Spot.SecondStage.replayWidths i.params.verifier.program.prog it.replay.prog.programFn,
    returns := Spot.SecondStage.replayReturns i.params.verifier.program.prog it.replay.prog.programFn i.record,
    supplied := it.params.verifier.supplied,
    fixed := it.params.verifier.pinned.map (·.1) ++ it.params.verifier.outputs.map (·.1), reads := it.reads,
    tables := it.tables }

def NoSampledOnly (i : Spot.Run.Input) (it : Spot.Run.Interior) : Prop :=
  ∀ (sample : Array ℕ) (strata : Array (String × Array (ℕ × ℕ))) (draw : Json),
    ∀ ps ss, (runOwn i it sample strata draw).parts = some ps → it.statements = some ss →
      Spot.SecondStage.anySampled (runOwn i it sample strata draw) ps ss.toList = false

structure AcceptedTwo (i : Spot.Run.Input) (it : Spot.Run.Interior) (j : Json) : Prop where
  run : Spot.Run.run i true = .ok j
  accept : Spot.getD j "symbol" = .str "accept"
  two : ∃ grain replay k, i.rule = .twoStage grain replay k
  interior : i.interior = some it
  sampledNone : NoSampledOnly i it

/-- Every field of `h` but these is the auditor's registered copy, which the run does not read. -/
structure HeldMatches (i : Spot.Run.Input) (it : Spot.Run.Interior) (h : Held) : Prop where
  first : ∃ n rule samplingDesign workTable resolved,
    Spot.nat? (Spot.getD i.record "population") = some n ∧
    i.rule.over n (Spot.Select.strataSizes i.params.verifier.program i.params.verifier.query) = some rule ∧
    rule.samplingDesignOver n = .ok samplingDesign ∧
    i.rule.tableOver i.params.verifier.program i.params.verifier.query i.params.verifier.workTable = .ok workTable ∧
    Spot.Select.resolve i.rule i.params.verifier.program i.params.verifier.query = .ok resolved ∧
    h.reg.first = ⟨{ i.params.verifier with samplingDesign, workTable }, resolved, i.closure⟩
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


def RoundTwo (h : Held) : Prop :=
  ∀ (own : Spot.SecondStage.Own) (ss : Array Json) (x m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h)
    (st : Roots) (rec draw : Json),
    decode x = some st → jsonMsg m₁ = some rec → secondDraw h c₀ c₁ = some draw → OwnAgrees h own st rec draw c₀ →
      StmtsAre h m₂ ss → RegistersNone rec → TwoHolds h → NamesDistinct h → RecordsHeld h x m₁ →
        RoundsHeld h m₂ → ScopeHeld h m₂ → PinHeld h m₂ → (Spot.SecondStage.check true own (some ss)).codes = #[] →
          stmtOk h x m₁ m₂ c₀ c₁ = true

/-- `Run.run` holds the sessions' verdicts of record only as JSON (`Service.checkVerdicts`), so this is what a
verdict of record that accepts says. -/
def SessionsAccept (h : Held) (H : Msg → Msg) (ss : Array Json) (m₂ : Msg) (P' : ℕ → Prover Unit)
    (r' : ∀ t, Fin ((FlockDeploy.V true).k (stmtAt h m₂ t)) → FlockDeploy.Coin) : Prop :=
  StmtsAre h m₂ ss ∧ RoundsHeld h m₂ ∧ ScopeHeld h m₂ ∧ PinHeld h m₂ ∧
    ∀ t < h.ms.length, (Claims.run H (P' t) (FlockDeploy.V true) (stmtAt h m₂ t) () (r' t)).out = .acc

structure AgreesTo (ξ ξ' : ByteArray) (R : ℕ) : Prop where
  size_le : R ≤ ξ'.size
  extract_eq : ξ'.extract 0 R = ξ.extract 0 R

/-- A sample reads no other byte (`select_read`, `selectSecond_read`), and nothing about the rest of a coin is
assumed. -/
def PlayCoins (h : Held) (i : Spot.Run.Input) (it : Spot.Run.Interior) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) :
    Prop :=
  (∀ draw proved read, Spot.Select.select h.reg.first.rule h.reg.first.program h.reg.first.query h.reg.first.table
      h.reg.first.closure i.source = .ok (draw, proved, read) → AgreesTo i.source (src c₀) read) ∧
    ∀ draw read, Spot.Select.selectSecond (h.reg.sample (src c₀)) h.reg.ns h.reg.ks it.source = .ok (draw, read) →
      AgreesTo it.source (stageSrc c₁) read

def Starts (b ξ : ByteArray) : Prop := b.size ≤ ξ.size ∧ ξ.extract 0 b.size = b

def FromDevice (dev : Properties.Challenger.History) (key : Spot.Challenger.ChallengerKey) (w label : String)
    (after : ByteArray) (coins : Json) (read : ByteArray) : Prop :=
  (∃ s ∈ dev.runs key.pub, Properties.Challenger.OpensOn dev key.pub w s.id ∧
      Starts read (s.released label after)) ∨
    Properties.Challenger.Break dev key label coins

/-- A stage that read no byte carries no evidence, and `Run.run` reads none of the sessions' challenges. -/
def RunCoinsFromDevice : Prop :=
  ∀ (i : Spot.Run.Input) (it : Spot.Run.Interior) (j : Json) (dev : Properties.Challenger.History),
    AcceptedTwo i it j →
    ∃ (read₁ read₂ : ℕ),
      (∃ resolved workTable draw rest, Spot.Select.select resolved i.params.verifier.program
        i.params.verifier.query workTable i.closure i.source = .ok (draw, rest, read₁)) ∧
      (∃ sample ns ks draw, Spot.Select.selectSecond sample ns ks it.source = .ok (draw, read₂)) ∧
      (0 < (i.source.extract 0 read₁).size → ∃ key w label d, i.key = some key ∧
        i.params.verifier.windowRoot = some w ∧ i.family = some label ∧
        Spot.Docket.digest i.record = .ok d ∧
        FromDevice dev key w label d i.coins (i.source.extract 0 read₁)) ∧
      (0 < (it.source.extract 0 read₂).size → ∃ key w label d, i.key = some key ∧
        it.params.verifier.windowRoot = some w ∧ it.family = some label ∧
        Spot.Docket.digest it.record = .ok d ∧
        FromDevice dev key w label d it.coins (it.source.extract 0 read₂))

/-- `RecordsHeld`, which `Run.run`'s registration step checks, is a premise here, not derived from that check; so is
`RegistersNone`, a premise on the run. -/
def SpotRunAcceptsPlay : Prop :=
  ∀ (i : Spot.Run.Input) (it : Spot.Run.Interior) (j : Json) (h : Held) (H : Msg → Msg) (x m₁ m₂ : Msg)
    (P' : ℕ → Prover Unit) (r' : ∀ t, Fin ((FlockDeploy.V true).k (stmtAt h m₂ t)) → FlockDeploy.Coin)
    (c₀ : FlockDeploy.Coin) (c₁ : Stage h),
    AcceptedTwo i it j → HeldMatches i it h → TwoHolds h → NamesDistinct h → RecordsHeld h x m₁ →
    RegistersNone it.record → jsonMsg x = some i.record → jsonMsg m₁ = some it.record →
    RoundTwo h → SessionsAccept h H ((it.statements).getD #[]) m₂ P' r' → PlayCoins h i it c₀ c₁ →
      (Claims.run H (proverSeq h [] m₁ m₂ P') (V h) x (fun _ => ()) (seqCoins h (x := x) c₀ c₁ r')).out = .acc

end SpotFlock.RunsAs
