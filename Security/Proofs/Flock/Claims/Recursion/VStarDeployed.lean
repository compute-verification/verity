import Proofs.Flock.Claims.Recursion.HpLast

/-!
# The deployed reference prover's pieces with the last round as the deployed inner prover sends it (`refPartsFin`)

`layR`'s last round reads the openings off the outer witnesses, then `encLast` and the trailer (`finMsg`), as the
deployed inner prover's last message lays them, so the inner prover sends `finMsg` there, not an hm96 commit string:

* **The inner prover** (`PFin`): `InnerRec.P` at every round before the last (`kI x - 1`), and at the last `finMsg`
  at its history.
* **The outer witnesses** (`witFin`): `vstarWit` at the inner rounds before the last, the history `finMsg` reads, so
  the last message is the one the witnesses assemble (`hpLast`'s `hfin`) by construction (`witFin_take`).
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)
open RecDeploy.HpLast

/-- **The outer witness at the deployed builder**: `vstarWit` at the inner rounds before the last. -/
def witFin (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg)
    (w : Claims.Flock.Wit) (ρ : I.Tape) (hist : List (Msg × Msg)) (s : ℕ) : Claims.Flock.Wit :=
  vstarWit A I x w ρ (hist.take (RestRead.kI x - 1)) s

/-- **The deployed inner prover at `x`**: `InnerRec.P` before the last round, `finMsg` at it. -/
def PFin (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg)
    (w : Claims.Flock.Wit) : Claims.Prover I.Tape :=
  ⟨fun xI h ρ => if h.length + 1 = RestRead.kI x then .ret (finMsg A I x w ρ h) else (I.P w).next xI h ρ⟩

/-- **The deployed reference prover's pieces at the compiled statement `x`**: the inner prover `PFin`, the staged
values' digests `refPartsAt`'s, and the outer witnesses `witFin`. -/
def refPartsFin (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg) : RefParts where
  TI := I.Tape
  PI := PFin A I x
  digests := (refPartsAt A I x).digests
  wit := witFin A I x

/-- **The last message is `finMsg` at the history before it.** -/
theorem PFin_last (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg)
    (w : Claims.Flock.Wit) (xI : Msg) (h : List (Msg × Msg)) (ρ : I.Tape) (hl : h.length + 1 = RestRead.kI x) :
    (PFin A I x w).next xI h ρ = .ret (finMsg A I x w ρ h) := by
  simp [PFin, hl]

/-- **Before the last round, the inner prover is `InnerRec.P`.** -/
theorem PFin_before (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg)
    (w : Claims.Flock.Wit) (xI : Msg) (h : List (Msg × Msg)) (ρ : I.Tape) (hl : h.length + 1 ≠ RestRead.kI x) :
    (PFin A I x w).next xI h ρ = (I.P w).next xI h ρ := by
  simp [PFin, hl]

/-- **The outer witnesses read the history the last message reads**: at inner rounds `hist` whose first `kI x - 1`
are `h`, `witFin` is `vstarWit` at `h`, the witnesses `finMsg A I x w ρ h` assembles (`wOs`). -/
theorem witFin_take (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg)
    (w : Claims.Flock.Wit) (ρ : I.Tape) (hist h : List (Msg × Msg)) (hh : hist.take (RestRead.kI x - 1) = h)
    (s : ℕ) : witFin A I x w ρ hist s = vstarWit A I x w ρ h s := by
  rw [witFin, hh]

/-- **`hpLast` at the deployed witness**: where round `kI x - 1`'s message is `finMsg` at the history before it, every
piece part `s` holds of it at `witFin` is that message's bytes at its offset. -/
theorem hpLast_fin {A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit} {I : InnerRec} {x : Msg}
    {w : Claims.Flock.Wit} {ρ : I.Tape} {hist : List (Msg × Msg)} {s : ℕ} (hs : s < nS x) {ms : List Msg}
    (hfin : ms.getD (RestRead.kI x - 1) [] = finMsg A I x w ρ (hist.take (RestRead.kI x - 1))) :
    ∀ o m, (o, m) ∈ (DeployedR.layR OpenRows.elemR).piecesOf x s (witFin A I x w ρ hist s) (RestRead.kI x - 1) →
      ((ms.getD (RestRead.kI x - 1) []).drop o).take m.length = m :=
  hpLast hs hfin

end RecDeploy.VStarBuild
