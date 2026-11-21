module

public import Proofs.Flock.Soundness.Discharge.ZkLink.CompiledLink
import Mathlib.Combinatorics.Matroid.Init

/-!
# The rows link at the compiled `--zk` value layer

`incorrectRegZ` says a unit is right when it is correct in the committed transcript and some opening per commit string
carries the program's evaluation; under a hiding commitment that opening exists whatever the row, so it binds no row.
This file names what binds the rows instead, at the plurality value layer (`XplurZC`), given a registrant's opening of
some of the commit strings (`isRead`, `wo`):

* `RowsRightZC`: at each of a unit's committed wires, the value layer carries the registrant's row where the wire's
  commit string is a registered read, and the circuit's evaluation on the value layer's inputs everywhere;
* `rowsOutZC`: the rows finder's output on a trial (`trialGZC`) at a registered read `c`: when the trial reads a value
  at `c` other than the registrant's, the binding's extractor on that opening and the registrant's (an explicit SHA-512
  collision when the two commitments agree, `ValueBindingZ.collide_spec`);
* `RowsCRZC`: `ecr/sha-512` (`Assumptions.SHA512CRExpected`) for the rows finder at each registered read, at the trial's
  cost (`cTZC`), as `LinkCRZC` asks it of the link finders.
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Extract LinkFinder Spot.Sampling Spot.Sampling.Partition ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

variable {L : SamplingDesign n} {decode : DecoderZK (C := C) planZ tab} {Dh : Type}
  {Hc : List UInt8 → Dh} (vb : ValueBindingZ A P planZ tab decode Hc)
  (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R)

/-- **Unit `u`'s rows are the registrant's and the program's** at the value layer `XplurZC`: on each of `u`'s committed
wires `g`, the value layer's bit is the registrant's row's (`vb.val (wo c)`) where `g`'s commit string `c` is a
registered read (`isRead c`), and is the circuit's evaluation on the value layer's inputs (`Circuit.eval`). -/
def RowsRightZC (isRead : vb.Pos → Prop) (wo : vb.Pos → vb.Opening) (k Rw : ℕ) (u : Fin n) : Prop :=
  ∀ g ∈ P.io u,
    (isRead (vb.pos g) → XplurZC A Hs P planZ tab vb R τ k Rw g = vb.bit (vb.val (wo (vb.pos g))) g) ∧
      XplurZC A Hs P planZ tab vb R τ k Rw g = C.eval (XplurZC A Hs P planZ tab vb R τ k Rw) g

end FlockSoundness.Discharge.ZkLink
