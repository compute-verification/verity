module

public import Proofs.Flock.Claims.Witnesses.SamplingGeneric
public import Proofs.Flock.Claims.CompleteFails
public import Proofs.Flock.Claims.Sampling.Code

/-!
# A compiled verifier whose statement has a partition and whose roots hold a bit

The network attester's lemmas at a monitoring period with a placed window (`NetTiming.Witness`'s) are stated over
this toy code:

* `posCm`: a commitment whose roots hold one bit (`rootBit`): an opening of position 0 passes at that bit and one of
  position 1 at its complement (`posOpens`), with no call, so it binds;
* `posStmt b`: `notStmt` under roots that hold `b`, and `posX`, its bytes;
* `posTops`: the openings of a statement's input and of its negation, at the bit its roots hold;
* `posL`: a compiled verifier over `toyA` that rejects every run, parses every statement as a `posStmt`, gives it the
  partition `notPart` and commits under `posCm`;
* `copyStmt`: a program whose output copies its input (`copyProg`), its one proof block sampled at rate `0`;
* `placedCode`: a sampled verifier's code for the deployed compiled form that parses every statement as `copyStmt`,
  gives it the partition `notPart`, accepts the statement `[]` after one round (`stmtV`), samples no proof block and
  opens no session.
-/

@[expose] public section

namespace Claims.Witness.SamplingGeneric

open Claims Claims.Sampling Claims.Witness Claims.Witness.Sampling
open Claims.SamplingGeneric (Arg Compiled)

/-- The bit the roots hold: set when there is a root. -/
def rootBit (roots : List Msg) : Bool := !roots.isEmpty

/-- An opening of position 0 passes at the roots' bit, one of position 1 at its complement, and no other. -/
def posOpens (roots : List Msg) (o : Opening Bool) : Bool :=
  (o.pos == 0 && o.val == rootBit roots) || (o.pos == 1 && o.val == !rootBit roots)

/-- The commitment whose roots hold the input's bit, with no call. -/
def posCm : CommitmentScheme Bool := ⟨fun roots o => .ret (posOpens roots o), fun _ _ _ => []⟩

/-- `notStmt` under roots that hold the bit `b`. -/
def posStmt (b : Bool) : AuditStatement := ⟨[], notProg, ⟨"", 0, []⟩, ⟨[.all]⟩, if b then [[]] else []⟩

/-- The statement's bytes at the bit `z`. -/
def posX (z : Bool) : Msg := if z then [0] else []

/-- The network attester's openings at a statement: the input at the roots' bit and its negation. -/
def posTops (s : AuditStatement) : List (Opening Bool) := [⟨0, rootBit s.roots, []⟩, ⟨1, !rootBit s.roots, []⟩]

/-- The compiled verifier that rejects every run, parses a statement's bytes as `posStmt` at whether there are any,
and commits under `posCm`. -/
noncomputable def posL (Coin : Type) : Compiled Coin (toyA Coin) where
  V := Claims.CompleteFails.rejV Coin
  parse x := some (posStmt (!x.isEmpty))
  scopeCheck _ := true
  partition _ := some notPart
  commit := posCm
  secondStage _ := ∅
  checked _ _ := []
  sh _ _ := ()
  κb _ _ := 0
  γb _ _ := 0
  rounds _ _ := ∅
  session _ _ _ := none
  iroots _ _ _ := []
  reveal _ _ _ _ := []
  openCalls _ := 0
  nOut _ := 1
  nIn _ := 1
  Tape := Unit
  P _ := silentP
  δc _ _ := 0

/-- One input gate and its exclusive or alone, returned: gate `1` copies the input. -/
def copyProg : Prog := ⟨[.input, .xor [0]], [1]⟩

/-- `copyProg`, its proof blocks each sampled independently at rate `0`, under no roots. -/
def copyStmt : AuditStatement := ⟨[], copyProg, ⟨"", 0, []⟩, ⟨[.bernoulli 0 1]⟩, []⟩

/-- The verifier that accepts the statement `[]` after one round, and rejects every other. -/
def stmtV : Verifier FlockDeploy.Coin := ⟨fun _ => 1, fun x _ _ => .ret (if x = [] then .acc else .rej)⟩

/-- **A code with a placed statement**: it parses every statement as `copyStmt` and gives it the partition `notPart`,
accepts the statement `[]` (`stmtV`), samples no proof block, opens no session and reveals nothing. -/
def placedCode : SamplingDeploy.Code where
  V := stmtV
  parse _ := some copyStmt
  scopeCheck _ := true
  partition _ := some notPart
  secondStage _ := ∅
  checked _ _ := []
  rounds _ _ := ∅
  session _ _ _ := none
  reveal _ _ _ _ := []
  own := #[]
  at_ _ _ := none
  layer _ _ _ := []
  place _ := ⟨1, 0⟩
  bits _ _ := false
  salt _ := []
  π _ := []
  innerN₀ _ := 0
  tables _ := 0
  reveals _ := 0
  sessions _ := 0
  step _ _ := .say []
  reads _ _ := []
  wit _ _ _ := ⟨[]⟩

end Claims.Witness.SamplingGeneric
