module

public import Definitions.Claims.SamplingTwo
public import Proofs.Flock.Claims.KS.Finds

/-!
# The proof system with one wait, and a compiled verifier over it

A proof system's extractor makes `waits` waits (`Arg.waits`), which only the extractor and its cost read: the
verifier, the relation, what a witness reveals and the errors don't. So the proof system with one wait
(`Arg.oneWait`) has the same verifier, relation and errors, and a compiled verifier over `A` is one over
`A.oneWait` as it stands (`Compiled.oneWait`, `CompiledTwo.oneWait`): the same verifier, sample, sessions,
commitment and reveals, so the same accepted runs, and each contract's clauses as they are
(`Compiled.Sessions.oneWait`, `CompiledTwo.Sessions.oneWait`). What differs is the extraction: a session's is one wait
where `L`'s is `A.waits`, so a run's extractions (`Compiled.runExt`) and every trace read off them (`Compiled.regValX`,
`CompiledTwo.iregVal`) are the one-wait extraction's, and its cost per session reads `1 + k` reruns for
`1 + waits·k`.

`Arg.OneWaitFinds` is the property of `A` the one-wait trace's bound reads beside `Arg.KS`: what `A`'s extractor
returns with a property of the run's coins, the one-wait extractor returns at half the rate, up to `2·κ + √(γ·Pr[B])`
(`Claims.Finds`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The proof system with one wait**: `A` with its extractor's waits one at every shape. -/
@[reducible] def Arg.oneWait (A : Arg Coin) : Arg Coin := { A with waits := fun _ => 1 }

/-- **What the extractor finds, its one wait finds at half the rate**: on an accepted run, `A`'s extractor returns a
witness with a property `G` of the run's coins with probability at most twice the one-wait extractor's, plus
`2·κ + √(γ·Pr[B])` (`Claims.Finds` at `A.E` and `A.oneWait.E`). -/
def Arg.OneWaitFinds (H : Msg → Msg) (A : Arg Coin) : Prop := Claims.Finds H A.V A.E A.oneWait.E A.κ A.γ

variable {A : Arg Coin}

/-- **The compiled verifier over the one-wait proof system**: `L` field for field. -/
@[reducible] def Compiled.oneWait (L : Compiled Coin A) : Compiled Coin A.oneWait where
  V := L.V
  parse := L.parse
  scopeCheck := L.scopeCheck
  partition := L.partition
  commit := L.commit
  stage2 := L.stage2
  checked := L.checked
  sh := L.sh
  κb := L.κb
  γb := L.γb
  rounds := L.rounds
  session := L.session
  skey := L.skey
  nSess := L.nSess
  iroots := L.iroots
  reveal := L.reveal
  openCalls := L.openCalls
  nOut := L.nOut
  nIn := L.nIn
  Tape := L.Tape
  tapeNonempty := L.tapeNonempty
  P := L.P
  δc := L.δc

/-- **The compiled two-stage verifier over the one-wait proof system**: `L` field for field. -/
@[reducible] def CompiledTwo.oneWait (L : CompiledTwo Coin A) : CompiledTwo Coin A.oneWait where
  toCompiled := L.toCompiled.oneWait
  two := L.two
  replaySample := L.replaySample

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem Compiled.oneWait_nPU (L : Compiled Coin A) (x : Msg) : L.oneWait.nPU x = L.nPU x := rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem Compiled.oneWait_nG (L : Compiled Coin A) (x : Msg) : L.oneWait.nG x = L.nG x := rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem Compiled.oneWait_κs (L : Compiled Coin A) (x : Msg) : L.oneWait.κs x = L.κs x := rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem Compiled.oneWait_γs (L : Compiled Coin A) (x : Msg) : L.oneWait.γs x = L.γs x := rfl

omit [Nonempty Coin] in
/-- **The contract at the one-wait proof system is the contract**: no clause reads the extractor's waits. -/
theorem Compiled.Sessions.oneWait {H : Msg → Msg} {L : Compiled Coin A} (hS : L.Sessions H) :
    L.oneWait.Sessions H :=
  ⟨hS.draws, hS.slots, hS.shared, hS.keys, hS.unions, hS.refuses, hS.interior, hS.fixed, hS.runs, hS.opens⟩

omit [Nonempty Coin] in
/-- **The two-stage contract at the one-wait proof system is the contract.** -/
theorem CompiledTwo.Sessions.oneWait {H : Msg → Msg} {L : CompiledTwo Coin A} (hS : L.Sessions H) :
    L.oneWait.Sessions H :=
  ⟨hS.replayDraws, hS.secondDraws, hS.slots, hS.shared, hS.unions, hS.refuses, hS.interior, hS.fixed, hS.runs,
    hS.opens, hS.keys⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- The opening checks' count is the same: it reads the commitment and the reveals. -/
theorem Compiled.OpenCallsLe.oneWait {H : Msg → Msg} {L : Compiled Coin A} (h : L.OpenCallsLe H) :
    L.oneWait.OpenCallsLe H := h

end Claims.SamplingGeneric
