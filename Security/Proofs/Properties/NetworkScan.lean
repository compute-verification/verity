import Properties.NetworkScan
import Proofs.Properties.NetworkScan.Accept
import Proofs.Properties.NetworkScan.Lowered
import Proofs.Flock.SamplingScope

/-!
# Toward an accepted scan's fit, at Bits' positions

`ScanFits` and `ScanScopeAcc` are stated at Bits' positions (`Properties.NetworkScan`): the statement's circuit is the
scan's audit's members' joint program (`jointOf`), its layout read through `SampledProofs.Bits.inBit`, `outBit` and
`wiresOf`. What `scanOf`'s acceptance gives of the Program is `Fits.scanOf_accepted` (`Accept`), and what the joint
program holds at each position, its inputs, outputs and reads, is `Lowered`'s. Here are the facts both statements'
witnesses read:

* `not_fits_of_no_outs`: a circuit that returns nothing fits no layout over a horizon of at least one step.
* `jointOf_nil`, `not_fits_of_no_members`: an audit of no members has the empty joint program, so no layout fits it.
* `oneSlot`, `canon_null`, `bytes_of_data_toList`: a schedule of one slot and one step, C-Flock's canonical `null`, and
  a byte array through its list of bytes.
* `nullStmt`, `acceptAllDeploy`: the statement whose descriptor is `null`'s bytes and whose circuit is empty, and a
  deployment whose scope check takes everything and which parses every message to it.
-/

namespace Properties.NetworkScan

/-- One slot of one lane and a window of one step. -/
def oneSlot : NetworkCheck.Period.Schedule := ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1⟩

/-- A circuit that returns nothing fits no layout over a horizon of at least one step: step 0's egress presence gate
isn't returned. -/
theorem not_fits_of_no_outs {L S : ℕ} (lay : NetTiming.BoundaryGates L S) (gs : List Claims.Sampling.BGate) {k : ℕ}
    (hk : 0 < k) (ℓ : Fin L) (σ : Fin S) : ¬ lay.Fits ⟨gs, []⟩ k := by
  rintro ⟨h, -, -⟩
  have := h 0 hk ℓ σ _ (List.mem_cons.2 (Or.inl rfl))
  simp at this

/-- An audit of no members has the empty joint program. -/
theorem jointOf_nil : jointOf [] = ⟨[], []⟩ := rfl

/-- An audit of no members fits no layout over a horizon of at least one step. -/
theorem not_fits_of_no_members {L S : ℕ} (lay : NetTiming.BoundaryGates L S) {k : ℕ} (hk : 0 < k) (ℓ : Fin L)
    (σ : Fin S) : ¬ lay.Fits (jointOf []) k := by
  rw [jointOf_nil]
  exact not_fits_of_no_outs lay [] hk ℓ σ

theorem canon_null : Flock.canon .null = .ok "null" := by
  simp only [Flock.canon]
  rfl

theorem bytes_of_data_toList (b : ByteArray) : (⟨b.data.toList.toArray⟩ : ByteArray) = b := rfl

/-- The statement whose descriptor is the bytes of `null` and whose circuit is empty. -/
def nullStmt : Claims.Sampling.AuditStatement :=
  { Claims.Sampling.ScopeWitness.toyStmt with desc := "null".toUTF8.data.toList, prog := ⟨[], []⟩ }

/-- A deployment whose scope check takes everything and which parses every message to `nullStmt`. -/
def acceptAllDeploy : Claims.Sampling.Deploy Bool ℕ Unit :=
  { Claims.Sampling.ScopeWitness.toyD with parse := fun _ => some nullStmt }

end Properties.NetworkScan
