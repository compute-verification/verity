module

public import Proofs.Flock.Soundness.Discharge.ZkReg.DefsH

/-! `verify --registered F` without `--zk`, as an event of the audit (`AcceptsR`). -/

@[expose] public section

namespace FlockSoundness.Discharge.PlainReg

open Flock
open FlockSoundness.Discharge.Exec (Inputs setupOf Accepts LiveAccepts)

/-- **verity-verify-proof `--registered F` accepts the transcript without `--zk`**, `own` F's ports: the session's statements set
up from `I` and the record's sample, each passes the registered check against `own`, and `Flock.verify` accepts the
record and the proof files. -/
def AcceptsR (I : Inputs) (own : Array Registered.Port) (t : Exec.Transcript) : Prop :=
  ∃ spec sts, setupOf I t.record = .ok (spec, sts) ∧ sts.forM (checkRegistered · (some own)) = .ok () ∧
    Flock.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok ()

variable {n : ℕ} {Reg : Type}

def LiveAcceptsR (I : Inputs) (own : Array Registered.Port) (o : Reg × Finset (Fin n) × Exec.Transcript) : Prop :=
  AcceptsR I own o.2.2

theorem accepts_of_R {I : Inputs} {own : Array Registered.Port} {t : Exec.Transcript} (h : AcceptsR I own t) :
    Accepts I t :=
  let ⟨spec, sts, hs, _, hv⟩ := h
  ⟨spec, sts, hs, hv⟩

end FlockSoundness.Discharge.PlainReg
