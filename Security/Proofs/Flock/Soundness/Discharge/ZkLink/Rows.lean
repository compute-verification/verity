import Proofs.Flock.Soundness.Discharge.ZkLink.CompiledLink

/-!
# The rows link at the compiled `--zk` value layer

`wrongRegZ` says a unit is right when it is correct in the committed transcript and some opening per commit string
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

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Extract LinkFinder Audit Audit.Partition ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

variable {L : Law n} {decode : DecoderZK (C := C) planZ tab} {Dh : Type}
  {Hc : List UInt8 → Dh} (vb : ValueBindingZ A P planZ tab decode Hc)
  (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R)

/-- **Unit `u`'s rows are the registrant's and the program's** at the value layer `XplurZC`: on each of `u`'s committed
wires `g`, the value layer's bit is the registrant's row's (`vb.val (wo c)`) where `g`'s commit string `c` is a
registered read (`isRead c`), and is the circuit's evaluation on the value layer's inputs (`Circuit.eval`). -/
def RowsRightZC (isRead : vb.Pos → Prop) (wo : vb.Pos → vb.Opening) (k Rw : ℕ) (u : Fin n) : Prop :=
  ∀ g ∈ P.io u,
    (isRead (vb.pos g) → XplurZC A Hs P planZ tab vb R τ k Rw g = vb.bit (vb.val (wo (vb.pos g))) g) ∧
      XplurZC A Hs P planZ tab vb R τ k Rw g = C.eval (XplurZC A Hs P planZ tab vb R τ k Rw) g

open Classical in
/-- **The rows finder's output** on a trial at commit string `c`: when the trial reads a value at `c` other than the
registrant's (`vb.val (wo c)`), the binding's extractor on the opening the trial found for it (`opOfVZC`) and the
registrant's opening `wo c`. -/
noncomputable def rowsOutZC (wo : vb.Pos → vb.Opening) (c : vb.Pos) (e : TrialOutZC F L D Sl) :
    Option (List UInt8 × List UInt8) :=
  if h : ∃ v ∈ rdTZC A Hs P planZ tab vb R τ c e, v ≠ vb.val (wo c) then
    some (vb.collide (opOfVZC A Hs P planZ tab vb R τ c e (Classical.choose h)) (wo c))
  else none

/-- **`ecr/sha-512` (A2) for the rows finders**: at each registered read `c` that some draw reads, the finder that plays
the trial at `c` (`trialGZC`) once and outputs `rowsOutZC`, at the trial's cost `cTZC` (`t'` per run it plays), satisfies
`Assumptions.SHA512CRExpected`.

Assumption record. Claim: `ecr/sha-512` (`Claims.ExpectedCR`). It assumes that a trial that reads a value other than
the registrant's at a registered read finds a SHA-512 collision with probability at most its expected evaluations over
`2^256`: the trial's opening and the registrant's open one commitment to two values, which is a collision
(`ValueBindingZ.collide_spec`). That is the generic bound for a hash with 512-bit outputs. A hash premise, as
`LinkCRZC`: `Assumptions.SHA512CRExpected` at each registered read's finder, so it reduces to `ExpectedCR` and asks
nothing of the verifier. -/
def RowsCRZC (isRead : vb.Pos → Prop) (wo : vb.Pos → vb.Opening) (k Rw : ℕ) (t' : ℝ) : Prop :=
  ∀ (c : vb.Pos) (_ : Nonempty (ΩcZ A P planZ tab vb L c)), isRead c →
    Assumptions.SHA512CRExpected Hc (trialGZC A Hs P planZ tab vb R τ c k Rw)
      (trialStratZC A Hs P planZ tab vb R τ c k Rw) (rowsOutZC A Hs P planZ tab vb R τ wo c) (cTZC planZ R t')

end FlockSoundness.Discharge.ZkLink
