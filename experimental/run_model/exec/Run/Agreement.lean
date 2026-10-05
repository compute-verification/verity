/-
What both sides fix before the run, each holding its own copy: the registered program and its proof units, the
verifier's law, the inner phase's shape, and the outer statement V*, pinned by digest. Nothing in it comes from a
message.
-/
import Flock.Zk
import Run.Wire

namespace Run

structure Agreement where
  /-- the registered program's digest: the claim's circuit and partition -/
  program : Digest
  /-- the registration roots the developer commits to, one hm96 leaf each -/
  roots : Nat
  /-- the partition level the law draws, its population of proof units, and the law -/
  level : String
  population : Nat
  law : Flock.Draw.Law
  /-- the claim: at most `threshold` wrong units at `level`, except with probability `δ` -/
  threshold : Nat
  deltaNum : Nat
  deltaDen : Nat
  /-- the inner phase: its batches, the rounds of each, the coin bytes each round's answer carries -/
  batches : Nat
  rounds : Nat
  coinBytes : Nat
  /-- the 64-word rows of round `r`'s message, the schedule's (the public `Hello`'s), the same in every batch -/
  innerRows : Nat → Nat
  /-- the public inner circuit (its setup's `fold` is the comb's) -/
  inner : Flock.Setup
  /-- V*, the outer statement: its digest, the session the verifier configures (its coin tree), its rounds -/
  vstar : Digest
  /-- V*'s statement as the verifier sets it up from V*'s circuit file; its `digest` must be `vstar` -/
  vstarSetup : Flock.Setup
  outer : Flock.SessionSpec
  outerRounds : Nat
  /-- the bytes of masks and the salts the outer session takes -/
  maskBytes : Nat
  maskSalts : Nat
  deriving Inhabited

/-- The hm96 leaves a commitment to the slot carries: one per root at registration, one per row of an inner message. -/
def Agreement.commitCount (A : Agreement) : Slot → Nat
  | .registration => A.roots
  | .inner _ r => A.innerRows r
  | .outer _ => 0

/-- The inner slots in the order the coin server answers them within a batch: round `r` after round `r - 1`. -/
def Agreement.innerNext (A : Agreement) (issued : List Slot) (b r : Nat) : Bool :=
  b < A.batches && r < A.rounds && !issued.contains (.inner b r) && (r == 0 || issued.contains (.inner b (r - 1)))

/-- Every inner slot answered: the inner phase is over. -/
def Agreement.innerDone (A : Agreement) (issued : List Slot) : Bool :=
  (List.range A.batches).all fun b => (List.range A.rounds).all fun r => issued.contains (.inner b r)

end Run
