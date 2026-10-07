import Models.Pouw.Guarantees.NCP
import Proofs.Pouw.NCP.Proofs
import Proofs.Pouw.NCP.GameProofs
import Proofs.Pouw.NCP.RelationWitness
import Proofs.Pouw.NCP.RouteUProofs
import Proofs.Pouw.NCP.RouteUWordProofs

/-!
# Proofs of NCP's guarantees

One theorem `Pouw.SecurityProofs.NCP.G : Pouw.Guarantees.NCP.G` per guarantee, which is the theorem in
`Pouw.NCP.Proofs`, `Pouw.NCP.GameProofs`, `Pouw.NCP.RelationWitness`, `Pouw.NCP.RouteUProofs` and
`Pouw.NCP.RouteUWordProofs` that proves the same statement under its old name.
-/

namespace Pouw.SecurityProofs.NCP

theorem TTNCPUIff : Pouw.Guarantees.NCP.TTNCPUIff := Pouw.NCP.RouteUProofs.ttncpUIff
theorem GammaFromTTNCP_U_v1 : Pouw.Guarantees.NCP.GammaFromTTNCP_U_v1 := Pouw.NCP.RouteUProofs.gammaFromTTNCP_U_v1
theorem GammaFromTTNCP_U_C2 : Pouw.Guarantees.NCP.GammaFromTTNCP_U_C2 := Pouw.NCP.RouteUProofs.gammaFromTTNCP_U_C2
theorem TTNCPUWitness : Pouw.Guarantees.NCP.TTNCPUWitness := Pouw.NCP.RouteUProofs.ttncpUWitness
theorem NCPWordFloorIndU : Pouw.Guarantees.NCP.NCPWordFloorIndU := Pouw.NCP.RouteUWordProofs.ncpWordFloorIndU
theorem Int7Int32UPos : Pouw.Guarantees.NCP.Int7Int32UPos := Pouw.NCP.RouteUWordProofs.int7Int32UPos
theorem WrapFinalOnlyIndU : Pouw.Guarantees.NCP.WrapFinalOnlyIndU := Pouw.NCP.RouteUWordProofs.wrapFinalOnlyIndU
theorem WrapWordFloorIndU : Pouw.Guarantees.NCP.WrapWordFloorIndU := Pouw.NCP.RouteUWordProofs.wrapWordFloorIndU
theorem Cancel : Pouw.Guarantees.NCP.Cancel := Pouw.NCP.Proofs.cancel
theorem Stacked : Pouw.Guarantees.NCP.Stacked := Pouw.NCP.Proofs.stacked
theorem Range : Pouw.Guarantees.NCP.Range := Pouw.NCP.Proofs.range
theorem ShiftBy24Admissible : Pouw.Guarantees.NCP.ShiftBy24Admissible := @Pouw.NCP.GameProofs.shiftBy24_admissible
theorem IndependentWitness : Pouw.Guarantees.NCP.IndependentWitness := Pouw.NCP.RelationWitness.independentWitness
theorem BiasU8Iff : Pouw.Guarantees.NCP.BiasU8Iff := Pouw.NCP.RouteUProofs.biasU8Iff
theorem CheckedUEq : Pouw.Guarantees.NCP.CheckedUEq := Pouw.NCP.RouteUProofs.checkedU_eq
theorem CheckedUFinal : Pouw.Guarantees.NCP.CheckedUFinal := Pouw.NCP.RouteUProofs.checkedU_final
theorem RangeU : Pouw.Guarantees.NCP.RangeU := Pouw.NCP.RouteUProofs.rangeU
theorem WrapU : Pouw.Guarantees.NCP.WrapU := Pouw.NCP.RouteUProofs.wrapU
theorem MmWUExact : Pouw.Guarantees.NCP.MmWUExact := Pouw.NCP.RouteUProofs.mmWUExact
theorem YUChecked : Pouw.Guarantees.NCP.YUChecked := @Pouw.NCP.RouteUProofs.yuChecked

end Pouw.SecurityProofs.NCP
