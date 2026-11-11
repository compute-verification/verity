module

public import Proofs.Flock.Soundness.Discharge.Composed.Coined
public import Proofs.Flock.Soundness.Discharge.Composed.DefsJ
public import Proofs.Flock.Soundness.Discharge.ZkLink.DrawOk
public import Proofs.Flock.Soundness.Discharge.ZkView.Defs
public import Proofs.Flock.Soundness.ZK.Complete
public import Proofs.Flock.Soundness.Premises
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt

/-!
# The `--zk` session's assumptions

The instances of `HashDerivedKey` and `PadNonvanishing` that `--zk` headlines take, stated as Prop constants so the
audit names them, and the record's custody (`live-verifier`) that `--zk`'s soundness takes: `RecordCustodyZK` at one
table, `RecordCustodyZKJS` at `J` under a stratified sample. `RecordCustodyZKJ`, at `J` under the subset sample, is
`RecordCustodyZKJPLaw`'s instance at no public inputs (`Proofs.Flock.Soundness.Premises`), which this module imports so
its readers keep the name.
-/

@[expose] public section

namespace FlockSoundness.Assumptions.Zk

open FlockSoundness

/-- The salt hash keys Hm96 hiding at `2^-193` (`HashDerivedKey` at `ZkView.My`, `ZkView.saltHash`). -/
def HashDerivedKeyHm96 : Prop := HashDerivedKey Discharge.ZkView.My Discharge.ZkView.saltHash

/-- Every pad of at most 40 columns is nonvanishing under `execArith` at `fOf PAD_C`. -/
def PadNonvanishing40 : Prop :=
  ∀ cols ≤ 40, PadNonvanishing execArith cols (Refine.fOf Flock.Zk.PAD_C)

/-- **The record is the live session's** (`live-verifier`; the custody of `Security/Proofs/Flock/Soundness/ASSUMPTIONS.md`, with A6,
`uniform/flock-coin-server`, for the coins in it). Nothing in a session record authenticates its writer, and
`flock-verify verify --zk` reads the sample and the proofs only from the record, so no check of the record establishes
either fact:

* `RecordsLiveZK I dj wr`: the record `verify --zk` reads is the one the challenger wrote for the live
  session. At every transcript it accepts, its sample is `dj S`, its proofs are the prover's, and its root, link points
  and rounds are the live outcome's.
* `DrawFileZK dj law pop`: the sample file is the one `flock-verify draw` printed for the verifier's own sample. Every sample
  file `dj S` that `UnitDraw.ofJson` decodes decodes to `drawOf law pop S`, the sample of `S` under `law` over
  `[0, pop)`.

Both are stated at what flock-verify parses, so the record's serialization is outside them. -/
def RecordCustodyZK {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (wr : Finset (Fin n) → Reg → Refine.LiveOut mPts → Discharge.Exec.Transcript) (law : Flock.Draw.Law) (pop : ℕ) :
    Prop :=
  Discharge.ZkLink.RecordsLiveZK I dj wr ∧ Discharge.ZkLink.DrawFileZK dj law pop

/-- **The record is the live session's, at `J` tables, under a stratified sample** (`live-verifier`, as
`RecordCustodyZKJ`): the record `verify --zk --session-tables J` reads is the one the challenger wrote for the live
session (`RecordsLiveZKJ`: its sample, proofs and link points, and each table's root and rounds, are the live outcome's),
and its sample file the one `flock-verify draw` printed for the verifier's own sample under the stratified sampling design
`stratified Kd strata` over `[0, pop)` (`DrawFileZK`). That one sample covers the sampling block's units, every stratum
of every table, and `verify` gives table `j` part `j` of its units (`drawJ`, `partS`), so the tables' samples are its
parts and form one stratified sampling design together. Only the record's custody is assumed: the parts and their link points are
flock-verify's own arithmetic (`ZkHidden.drawOkZKHJ_of_size`). -/
def RecordCustodyZKJS {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (Kd : ℕ) (strata : Array Flock.Draw.Stratum) (pop : ℕ) : Prop :=
  Discharge.Composed.RecordsLiveZKJ I dj Jf wr ∧ Discharge.ZkLink.DrawFileZK dj (.stratified Kd strata) pop

end FlockSoundness.Assumptions.Zk
