module

public import Definitions.Core.Levels
public import Definitions.Flock.Audit.SamplingDesign

/-! # An audit's units sampled by levels

`ofLevels Lv idx` samples the unit `u` when the levels' sample `Lv.design` holds its block `idx u`. -/

@[expose] public section

namespace FlockSoundness.Audit.SamplingDesign

/-- The units whose blocks the levels `Lv` sample. -/
def ofLevels {n : ℕ} {G : Type} [Fintype G] [DecidableEq G] (Lv : Verity.Levels G) (idx : Fin n → Finset G) :
    SamplingDesign n where
  Ω := Lv.design.Ω
  draw ω := Finset.univ.filter fun u => idx u ∈ Lv.design.draw ω

end FlockSoundness.Audit.SamplingDesign
