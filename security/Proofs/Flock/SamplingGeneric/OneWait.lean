module

public import Definitions.Claims.SamplingTwo
public import Proofs.Flock.SamplingGeneric.Efficient
public import Proofs.Flock.SamplingGeneric.TwoKSAt

/-!
# A compiled verifier read at an extractor that makes one wait

A proof system for proof blocks fixes how many waits its extractor `E^rep` makes (`Arg.waits`). The verifier, the
relation, the sessions' contract, the commitment and what a witness reveals read none of it: only the extractor's
coins (`Arg.EC`), the extractor (`Arg.ext`) and what is defined from them do (a session's extraction, a run's
extractions, the traces read off them, the sessions' expected and strict reruns).

* `Arg.oneWait`: the proof system with one wait in place of `waits`. Every other part is the proof system's, its
  errors `κ` and `γ` included, which are numbers here: nothing says the one-wait extractor meets them.
* `Compiled.oneWait`, `CompiledTwo.oneWait`: a compiled verifier's layout over that proof system, field by field the
  layout itself.
* `oneWait_sessions`, `oneWait_runsFixed`, `oneWait_costContract`, `oneWait_openCallsLe`, `oneWait_binds`: the
  contract at one wait is the contract, by unfolding, as are the proof blocks, the gates and the sessions' errors
  (`oneWait_nPU`, `oneWait_nG`, `oneWait_κs`, `oneWait_γs`).
* `Arg.oneWait_rerunsExp_le`, `Compiled.oneWait_es`: the one-wait extractor reruns at most `1 + k` times in
  expectation, and the sessions' expected reruns at one wait are the largest `1 + k` over the proof blocks' shapes.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims

/-! ## The layout at one wait -/

/-- **The proof system at one wait**: `A` with its extractor's waits set to one at every shape. The verifier, the
relation, the openings, the assemble step, the check, the accepted reruns a wait stops at, the truncation and the
errors are `A`'s. -/
def Arg.oneWait {Coin : Type} (A : Arg Coin) : Arg Coin := { A with waits := fun _ => 1 }

/-- **A compiled verifier's layout at one wait**: every field is `L`'s, over `A.oneWait`. -/
def Compiled.oneWait {Coin : Type} {A : Arg Coin} (L : Compiled Coin A) : Compiled Coin A.oneWait where
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

/-- **A compiled two-stage verifier's layout at one wait**: `Compiled.oneWait` with `L`'s two-stage design and replay
sample. -/
def CompiledTwo.oneWait {Coin : Type} {A : Arg Coin} (L : CompiledTwo Coin A) : CompiledTwo Coin A.oneWait where
  toCompiled := L.toCompiled.oneWait
  two := L.two
  replaySample := L.replaySample

/-! ## The contract at one wait is the contract -/

/-- The proof blocks at one wait are the layout's. -/
theorem Compiled.oneWait_nPU {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    {A : Arg Coin} (L : Compiled Coin A) (x : Msg) :
    L.oneWait.nPU x = L.nPU x := rfl

/-- The program's gates at one wait are the layout's. -/
theorem Compiled.oneWait_nG {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    {A : Arg Coin} (L : Compiled Coin A) (x : Msg) :
    L.oneWait.nG x = L.nG x := rfl

/-- The sessions' knowledge error at one wait is the layout's number. -/
theorem Compiled.oneWait_κs {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    {A : Arg Coin} (L : Compiled Coin A) (x : Msg) :
    L.oneWait.κs x = L.κs x := rfl

/-- The sessions' `γ` per round at one wait is the layout's number. -/
theorem Compiled.oneWait_γs {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    {A : Arg Coin} (L : Compiled Coin A) (x : Msg) :
    L.oneWait.γs x = L.γs x := rfl

/-- **The commitment binds at one wait exactly when it binds**: the commitment is the layout's. -/
theorem Compiled.oneWait_binds {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    {A : Arg Coin} {H : Msg → Msg} {L : Compiled Coin A} :
    L.oneWait.commit.Binds H ↔ L.commit.Binds H := Iff.rfl

/-- **The opening checks' count at one wait is the layout's**: it reads what a witness reveals and the commitment. -/
theorem Compiled.oneWait_openCallsLe {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    {A : Arg Coin} {H : Msg → Msg} {L : Compiled Coin A} :
    L.oneWait.OpenCallsLe H ↔ L.OpenCallsLe H := Iff.rfl

/-- **The one-stage contract at one wait is the contract**, clause by clause. -/
theorem Compiled.oneWait_sessions {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    {H : Msg → Msg} {L : Compiled Coin A} : L.oneWait.Sessions H ↔ L.Sessions H :=
  ⟨fun h => ⟨h.draws, h.slots, h.shared, h.keys, h.unions, h.refuses, h.interior, h.fixed, h.runs, h.opens⟩,
    fun h => ⟨h.draws, h.slots, h.shared, h.keys, h.unions, h.refuses, h.interior, h.fixed, h.runs, h.opens⟩⟩

/-- **The sessions' runs and fixed rounds at one wait are the layout's.** -/
theorem Compiled.oneWait_runsFixed {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    {H : Msg → Msg} {L : Compiled Coin A} : L.oneWait.RunsFixed H ↔ L.RunsFixed H :=
  ⟨fun h => ⟨h.fixed, h.runs⟩, fun h => ⟨h.fixed, h.runs⟩⟩

/-- **The facts the extractor's cost reads at one wait are the layout's.** -/
theorem Compiled.oneWait_costContract {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    {H : Msg → Msg} {L : Compiled Coin A} : L.oneWait.CostContract H ↔ L.CostContract H :=
  ⟨fun h => { toRunsFixed := ⟨h.fixed, h.runs⟩, slots := h.slots, keys := h.keys },
    fun h => { toRunsFixed := ⟨h.fixed, h.runs⟩, slots := h.slots, keys := h.keys }⟩

/-- **The two-stage contract at one wait is the contract**, clause by clause. -/
theorem CompiledTwo.oneWait_sessions {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    {H : Msg → Msg} {L : CompiledTwo Coin A} : L.oneWait.Sessions H ↔ L.Sessions H :=
  ⟨fun h => ⟨h.replayDraws, h.secondDraws, h.slots, h.shared, h.unions, h.refuses, h.interior, h.fixed, h.runs,
      h.opens, h.keys⟩,
    fun h => ⟨h.replayDraws, h.secondDraws, h.slots, h.shared, h.unions, h.refuses, h.interior, h.fixed, h.runs,
      h.opens, h.keys⟩⟩

/-! ## The one-wait extractor's reruns -/

/-- **The one-wait extractor reruns at most `1 + k` times in expectation**, at every prover and tape: it is `E^rep`
at one wait (`Arg.rerunsExp_le`). -/
theorem Arg.oneWait_rerunsExp_le {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin)
    (H : Msg → Msg) : ExtractRunsExpLe H A.oneWait.E fun x => 1 + (A.k (A.shape x) : ℝ) := by
  have h : ExtractRunsExpLe H A.oneWait.E fun x => 1 + ((1 : ℕ) : ℝ) * (A.k (A.shape x) : ℝ) :=
    A.oneWait.rerunsExp_le H
  simpa only [Nat.cast_one, one_mul] using h

/-- **The sessions' expected reruns at one wait**: the largest `1 + k` over the proof blocks' shapes. -/
theorem Compiled.oneWait_es {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    {A : Arg Coin} (L : Compiled Coin A) (x : Msg) :
    L.oneWait.es x = maxBelow (L.nPU x) fun u => 1 + (A.k (L.sh x u) : ℝ) := by
  show maxBelow (L.nPU x) (fun u => 1 + ((1 : ℕ) : ℝ) * (A.k (L.sh x u) : ℝ)) = _
  simp only [Nat.cast_one, one_mul]

end Claims.SamplingGeneric
