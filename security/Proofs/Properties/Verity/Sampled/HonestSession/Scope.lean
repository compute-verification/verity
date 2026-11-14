module

public import Proofs.Properties.Verity.Sampled.HonestSession.CircuitFile
public import Proofs.Properties.Verity.Sampled.HonestSession.PublicFile
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String

/-!
# The circuit is in the proved scope and its ports' published domains are distinct

`Layout.scopeOkH` holds of `cH` (one 64-word `sha512/row/v2` output row, no input row), `ProvedScope.circuitOf` reads the
circuit file to `cH`, and `PortRoots.checkOf` passes at the public file: the one port's domain is 64 bytes and has no
twin. The walk of `scopeOkH` is the one the local probe of `filesWhy` evaluates; here it reads `CircuitFileFacts`.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.Scope

open Flock

private theorem hidden_tagsZ : tagsZ.hiddenOutputs = true := rfl

private theorem v2Row_yPort : ProvedScope.v2Row yPort = true := by
  simp [ProvedScope.v2Row, yPort]

private theorem unpinned_cH (hC : CircuitFileFacts) : ProvedScope.unpinned cH = true := by
  unfold ProvedScope.unpinned
  rw [hC.cH_metaJ]
  decide +kernel

private theorem bitRowsU_cH (hC : CircuitFileFacts) : ProvedScope.bitRowsU cH = true := by
  unfold ProvedScope.bitRowsU HmOut.nIn
  rw [hC.cH_ports, hC.cH_outRowPorts]
  simp [v2Row_yPort]

private theorem bitRows_cH (hC : CircuitFileFacts) : ProvedScope.bitRows cH = true := by
  unfold ProvedScope.bitRows
  simp [bitRowsU_cH hC, unpinned_cH hC]

private theorem rowsOk_cH (hC : CircuitFileFacts) : PortRoots.rowsOk cH = true := by
  unfold PortRoots.rowsOk
  simp [bitRows_cH hC]

private theorem dupName_cH (hC : CircuitFileFacts) : Registered.dupName cH = none := by
  unfold Registered.dupName
  rw [hC.cH_ports]
  simp

private theorem collidesBy_one (dom : Nat → ByteArray) (roots : Array ByteArray) :
    PortRoots.collidesBy 1 dom roots = false := by
  simp [PortRoots.collidesBy]

private theorem collidesWith_cH (hC : CircuitFileFacts) (H : ByteArray → ByteArray) (pub : Public) (nl : Nat) :
    PortRoots.collidesWith H cH pub nl = false := by
  unfold PortRoots.collidesWith PortRoots.collidesIn
  rw [hC.cH_ports]
  exact collidesBy_one _ _

private theorem collidesPubWith_cH (hC : CircuitFileFacts) (H : ByteArray → ByteArray) (pub : Public) :
    PortRoots.collidesPubWith H cH pub = false := by
  unfold PortRoots.collidesPubWith
  rw [hC.cH_ports]
  exact collidesBy_one _ _

private theorem wordsOk_cH (hC : CircuitFileFacts) : PortRoots.wordsOk cH = true := by
  unfold PortRoots.wordsOk
  rw [hC.cH_ports]
  simp [yPort]

private theorem dupId_cH (hC : CircuitFileFacts) (pub : Public) : PortRoots.dupId cH pub = false := by
  unfold PortRoots.dupId
  rw [hC.cH_ports]
  simp

private theorem domY_size : domY.size = 64 := by decide

private theorem idsSized_cH (hC : CircuitFileFacts) (hP : PublicFileFacts) : PortRoots.idsSized cH pubO = true := by
  unfold PortRoots.idsSized PortRoots.idOf
  rw [hC.cH_ports]
  simp [yPort, hP.pubO_domain, Except.toOption, domY_size]

private theorem loadedWhy_cH (hC : CircuitFileFacts) (hP : PublicFileFacts) :
    PortRoots.loadedWhy Sha512.hash cH pubO = none := by
  unfold PortRoots.loadedWhy
  simp [collidesWith_cH hC, wordsOk_cH hC, idsSized_cH hC hP, dupId_cH hC, collidesPubWith_cH hC]

theorem scopeOkH (hC : CircuitFileFacts) : FlockSoundness.Discharge.Layout.scopeOkH cH = true := by
  have hb : FlockSoundness.Discharge.Layout.bitRows cH = true := bitRows_cH hC
  unfold FlockSoundness.Discharge.Layout.scopeOkH
  rw [hC.unit_portReads, hb]
  simp [hC.unit_inGroups, hC.unit_outNet, hC.unit_unitLog]

theorem circuitOf_eq (hC : CircuitFileFacts) : ProvedScope.circuitOf tagsZ circuitFile [] = .ok cH := by
  unfold ProvedScope.circuitOf
  rw [hidden_tagsZ]
  exact hC.parseCommit_ok

theorem ports_ok (hC : CircuitFileFacts) (hP : PublicFileFacts) :
    PortRoots.checkOf (.ok cH) tagsZ publicFile = .ok () := by
  unfold PortRoots.checkOf PortRoots.checkWith
  simp [rowsOk_cH hC, dupName_cH hC, hP.hmOut_loadPublic, loadedWhy_cH hC hP]

theorem ready (hC : CircuitFileFacts) (hP : PublicFileFacts) :
    PortRoots.checkOf (ProvedScope.circuitOf tagsZ circuitFile []) tagsZ publicFile = .ok () := by
  rw [circuitOf_eq hC]
  exact ports_ok hC hP

end Verity.Sampled.Instance.HonestSession.Scope
