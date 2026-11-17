module

public import Definitions.Claims.SamplingTwo
public import Proofs.Flock.Claims.KS.Finds

/-! The proof system with one wait, and a compiled verifier over it, which accepts the same runs. -/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The proof system with one wait**: `A` with its extractor's waits one at every shape. -/
@[reducible] def Arg.oneWait (A : Arg Coin) : Arg Coin := { A with waits := fun _ => 1 }

def Arg.OneWaitFinds (H : Msg → Msg) (A : Arg Coin) : Prop := Claims.Finds H A.V A.E A.oneWait.E A.κ A.γ

variable {A : Arg Coin}

/-- **The compiled verifier over the one-wait proof system**: `L` field for field. -/
@[reducible] def Compiled.oneWait (L : Compiled Coin A) : Compiled Coin A.oneWait where
  V := L.V
  parse := L.parse
  scopeCheck := L.scopeCheck
  partition := L.partition
  commit := L.commit
  secondStage := L.secondStage
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
theorem Compiled.Sessions.oneWait {H : Msg → Msg} {L : Compiled Coin A} (hS : L.Sessions H) :
    L.oneWait.Sessions H :=
  ⟨hS.draws, hS.slots, hS.shared, hS.keys, hS.unions, hS.refuses, hS.interior, hS.fixed, hS.runs, hS.opens⟩

omit [Nonempty Coin] in
theorem CompiledTwo.Sessions.oneWait {H : Msg → Msg} {L : CompiledTwo Coin A} (hS : L.Sessions H) :
    L.oneWait.Sessions H :=
  ⟨hS.replaySamples, hS.secondSamples, hS.slots, hS.shared, hS.unions, hS.refuses, hS.interior, hS.fixed, hS.runs,
    hS.opens, hS.keys⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem Compiled.OpenCallsLe.oneWait {H : Msg → Msg} {L : Compiled Coin A} (h : L.OpenCallsLe H) :
    L.oneWait.OpenCallsLe H := h

end Claims.SamplingGeneric
