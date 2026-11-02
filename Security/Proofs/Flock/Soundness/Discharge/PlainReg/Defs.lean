import Proofs.Flock.Soundness.Discharge.ZkReg.DefsH

/-!
# `verify --registered F` without `--zk`, as an event of the audit

`flock-verify verify --registered F` without `--zk` (`Main.verifyCmd`) checks a session as `Exec.Accepts` does, and holds
every table's statement to the verifier's own registered reads `own` (F's ports, `Registered.ofJson`): after
`Stmt.setupTables`, `Main.buildSessionOf` runs `Flock.checkRegistered` on each statement, the library's definition, which
`AcceptsR` reads as it is. `AcceptsR` is that verdict; it implies `Exec.Accepts` (`accepts_of_R`).

What its registered check reads at a hidden-output drawn statement is what it reads under `--zk`: `ZkReg.OwnReadsH` and
`ZkReg.regPathsH`, which read the circuit, the population's public file and the record, and no `--zk` input.
-/

namespace FlockSoundness.Discharge.PlainReg

open Flock
open FlockSoundness.Discharge.Exec (Inputs setupOf Accepts LiveAccepts)

/-- **flock-verify `--registered F` accepts the transcript without `--zk`**, `own` F's ports: the session's statements set
up from `I` and the record's draw, each passes the registered check against `own`, and `Flock.verify` accepts the
record and the proof files. -/
def AcceptsR (I : Inputs) (own : Array Registered.Port) (t : Exec.Transcript) : Prop :=
  ∃ spec sts, setupOf I t.record = .ok (spec, sts) ∧ sts.forM (checkRegistered · (some own)) = .ok () ∧
    Flock.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok ()

variable {n : ℕ} {Reg : Type}

/-- flock-verify `--registered F` accepts the live audit's transcript without `--zk`. -/
def LiveAcceptsR (I : Inputs) (own : Array Registered.Port) (o : Reg × Finset (Fin n) × Exec.Transcript) : Prop :=
  AcceptsR I own o.2.2

/-- The registered check only refuses more: what `verify --registered F` accepts, `verify` accepts. -/
theorem accepts_of_R {I : Inputs} {own : Array Registered.Port} {t : Exec.Transcript} (h : AcceptsR I own t) :
    Accepts I t :=
  let ⟨spec, sts, hs, _, hv⟩ := h
  ⟨spec, sts, hs, hv⟩

theorem liveAccepts_of_R {I : Inputs} {own : Array Registered.Port} {o : Reg × Finset (Fin n) × Exec.Transcript}
    (h : LiveAcceptsR I own o) : LiveAccepts I o :=
  accepts_of_R h

end FlockSoundness.Discharge.PlainReg
