import Definitions.Pouw.Guarantees.Adaptive
import Proofs.Pouw.Dimension.AdaptiveProofs

/-!
# Proofs of adaptive-dw's guarantees (draft model, pending review)

One theorem `Pouw.SecurityProofs.Adaptive.G : Pouw.Guarantees.Adaptive.G` per guarantee, each the theorem of
`AdaptiveProofs` that proves it.
-/

namespace Pouw.SecurityProofs.Adaptive

open Pouw.Dimension.Pointwise

theorem RunChunks : Pouw.Guarantees.Adaptive.RunChunks := fun _ _ _ _ _ _ m₀ P Y hY =>
  run_chunks_le m₀ P Y hY

theorem AdaptiveChunks : Pouw.Guarantees.Adaptive.AdaptiveChunks := fun _ _ _ _ _ _ Ω _ m₀ Y =>
  adaptiveDW_chunks Ω m₀ Y

theorem LevelsObstruction : Pouw.Guarantees.Adaptive.LevelsObstruction :=
  fun _ _ _ _ _ _ _ _ Ω _ _ m₀ rows steps base c B hB =>
    not_adaptiveDW_levels (Classical.arbitrary _) Ω m₀ rows steps base c B hB

end Pouw.SecurityProofs.Adaptive
