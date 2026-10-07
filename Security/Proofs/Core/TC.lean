import Models.Core.Guarantees.TC
import Proofs.Core.TC.Gadgets
import Proofs.Core.TC.Proofs
import Proofs.Core.TC.Ampere
import Proofs.Core.TC.Sound
import Proofs.Core.TC.AmpereSound
import Proofs.Core.TC.Boundary
import Proofs.Core.TC.Vectors

/-!
# Proofs of the tensor-core guarantees

Each guarantee of `Verity.Guarantees.TC` is the theorem of the same statement in `Verity.TC` (`Sound`, `AmpereSound`,
`Boundary`) or `Verity.TC.Gadgets`.
-/

namespace Verity.SecurityProofs

theorem IsZeroGadget : Verity.Guarantees.IsZeroGadget := Verity.TC.Gadgets.iszero_gadget
theorem RangeGadget : Verity.Guarantees.RangeGadget := Verity.TC.Gadgets.range_gadget
theorem AdaEqAmpere : Verity.Guarantees.AdaEqAmpere := Verity.TC.ada_eq_ampere

theorem AmpereStepComplete : Verity.Guarantees.AmpereStepComplete := Verity.TC.ampere_step_complete
theorem AmpereStepHw : Verity.Guarantees.AmpereStepHw := Verity.TC.ampere_step_hw
theorem AmpereStepIff : Verity.Guarantees.AmpereStepIff := Verity.TC.ampere_step_iff
theorem AmpereStepSound : Verity.Guarantees.AmpereStepSound := Verity.TC.ampere_step_sound
theorem AmpereUnitBf16 : Verity.Guarantees.AmpereUnitBf16 := Verity.TC.ampere_unit_bf16
theorem AmpereUnitSound : Verity.Guarantees.AmpereUnitSound := Verity.TC.ampere_unit_sound
theorem AmpereUnitSoundZero : Verity.Guarantees.AmpereUnitSoundZero := Verity.TC.ampere_unit_sound_zero

theorem HopperStepComplete : Verity.Guarantees.HopperStepComplete := Verity.TC.hopper_step_complete
theorem HopperStepHw : Verity.Guarantees.HopperStepHw := Verity.TC.hopper_step_hw
theorem HopperStepIff : Verity.Guarantees.HopperStepIff := Verity.TC.hopper_step_iff
theorem HopperStepSound : Verity.Guarantees.HopperStepSound := Verity.TC.hopper_step_sound
theorem HopperUnitBf16 : Verity.Guarantees.HopperUnitBf16 := Verity.TC.hopper_unit_bf16
theorem HopperUnitSound : Verity.Guarantees.HopperUnitSound := Verity.TC.hopper_unit_sound
theorem HopperUnitSoundZero : Verity.Guarantees.HopperUnitSoundZero := Verity.TC.hopper_unit_sound_zero

theorem PackSoundComplete : Verity.Guarantees.PackSoundComplete := Verity.TC.pack_sound_complete

end Verity.SecurityProofs
