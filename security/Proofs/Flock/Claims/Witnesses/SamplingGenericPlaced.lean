module

public import Proofs.Flock.Claims.Witnesses.SamplingGeneric
public import Proofs.Flock.Claims.CompleteFails

/-! A compiled verifier whose statement has a partition and whose roots hold a bit: the toy code the network attester's placed-window lemmas are stated over. -/

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

end Claims.Witness.SamplingGeneric
