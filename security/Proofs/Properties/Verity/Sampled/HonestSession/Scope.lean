module

public import Proofs.Properties.Verity.Sampled.HonestSession.CircuitFile
public import Proofs.Properties.Verity.Sampled.HonestSession.PublicFile

/-!
# The circuit is in the proved scope and its ports' published domains are distinct

`Layout.scopeOkH` holds of `cH` (one 64-word `sha512/row/v2` output row, no input row), `ProvedScope.circuitOf` reads the
circuit file to `cH`, and `PortRoots.checkOf` passes at the public file: the one port's domain is 64 bytes and has no
twin. The walk of `scopeOkH` is the one the local probe of `filesWhy` evaluates; here it reads `CircuitFile`'s fields.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.Scope

open Flock

private theorem hidden_tagsZ : tagsZ.hiddenOutputs = true := rfl

private theorem v2Row_yPort : ProvedScope.v2Row yPort = true := by
  simp [ProvedScope.v2Row, yPort]

private theorem unpinned_cH : ProvedScope.unpinned cH = true := by
  unfold ProvedScope.unpinned
  rw [CircuitFile.cH_metaJ]
  decide +kernel

private theorem bitRowsU_cH : ProvedScope.bitRowsU cH = true := by
  unfold ProvedScope.bitRowsU HmOut.nIn
  rw [CircuitFile.cH_ports, CircuitFile.cH_outRowPorts]
  simp [v2Row_yPort, yPort]

private theorem bitRows_cH : ProvedScope.bitRows cH = true := by
  unfold ProvedScope.bitRows
  simp [bitRowsU_cH, unpinned_cH]

private theorem rowsOk_cH : PortRoots.rowsOk cH = true := by
  unfold PortRoots.rowsOk
  simp [bitRows_cH]

private theorem dupName_cH : Registered.dupName cH = none := by
  unfold Registered.dupName
  rw [CircuitFile.cH_ports]
  simp

private theorem collidesBy_one (dom : Nat → ByteArray) (roots : Array ByteArray) :
    PortRoots.collidesBy 1 dom roots = false := by
  simp [PortRoots.collidesBy]

private theorem collidesWith_cH (H : ByteArray → ByteArray) (pub : Public) (nl : Nat) :
    PortRoots.collidesWith H cH pub nl = false := by
  unfold PortRoots.collidesWith PortRoots.collidesIn
  rw [CircuitFile.cH_ports]
  exact collidesBy_one _ _

private theorem collidesPubWith_cH (H : ByteArray → ByteArray) (pub : Public) :
    PortRoots.collidesPubWith H cH pub = false := by
  unfold PortRoots.collidesPubWith
  rw [CircuitFile.cH_ports]
  exact collidesBy_one _ _

private theorem wordsOk_cH : PortRoots.wordsOk cH = true := by
  unfold PortRoots.wordsOk
  rw [CircuitFile.cH_ports]
  simp [yPort]

private theorem dupId_cH (pub : Public) : PortRoots.dupId cH pub = false := by
  unfold PortRoots.dupId
  rw [CircuitFile.cH_ports]
  simp

/-- The public file's header holds the row's published domain, the 64 bytes `domY`. -/
private theorem pubO_domain : pubO.fv3Bytes "domain_ids" "y" = .ok domY := by
  sorry -- needs the header `hmOut_loadPublic` reads (`pubO.header = norm hdrJ`): a field `PublicFile` does not export

private theorem domY_size : domY.size = 64 := by decide

private theorem idsSized_cH : PortRoots.idsSized cH pubO = true := by
  unfold PortRoots.idsSized PortRoots.idOf
  rw [CircuitFile.cH_ports]
  simp [yPort, pubO_domain, Except.toOption, domY_size]

private theorem loadedWhy_cH : PortRoots.loadedWhy Sha512.hash cH pubO = none := by
  unfold PortRoots.loadedWhy
  simp [collidesWith_cH, wordsOk_cH, idsSized_cH, dupId_cH, collidesPubWith_cH]

/-- The unit the circuit file registers reads only its leaf group, returns the outputs, has at most `2^32` rows and reads
only port bits among its inputs. -/
private theorem unit_cH : cH.unit.inGroups.size = 1 ∧ cH.outNet = cH.unitNet ∧ cH.unit.unitLog ≤ 32 ∧
    FlockSoundness.Discharge.Layout.portReadsB cH.unit = true := by
  sorry -- needs the unit net `HmOut.parse` reads (`CircuitFile`'s walk): a field `CircuitFile` does not export

theorem scopeOkH : FlockSoundness.Discharge.Layout.scopeOkH cH = true := by
  obtain ⟨h1, h2, h3, h4⟩ := unit_cH
  have hb : FlockSoundness.Discharge.Layout.bitRows cH = true := bitRows_cH
  unfold FlockSoundness.Discharge.Layout.scopeOkH
  rw [h4, hb]
  simp [h1, h2, h3]

theorem circuitOf_eq : ProvedScope.circuitOf tagsZ circuitFile [] = .ok cH := by
  unfold ProvedScope.circuitOf
  rw [hidden_tagsZ]
  exact CircuitFile.parseCommit_ok

theorem ports_ok : PortRoots.checkOf (.ok cH) tagsZ publicFile = .ok () := by
  unfold PortRoots.checkOf PortRoots.checkWith
  simp [rowsOk_cH, dupName_cH, PublicFile.hmOut_loadPublic, loadedWhy_cH]

theorem ready : PortRoots.checkOf (ProvedScope.circuitOf tagsZ circuitFile []) tagsZ publicFile = .ok () := by
  rw [circuitOf_eq]
  exact ports_ok

end Verity.Sampled.Instance.HonestSession.Scope
