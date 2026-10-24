import Definitions.Core.TC.Pipeline

/-! The proof-only part of `Definitions.Core.TC.Pipeline`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Verity.TC

/-- `models.HOPPER_BF16_M16N8K16`: `mma.sync.m16n8k16.bf16` on Hopper (H100) and on sm_120 (RTX 5090). -/
def HOPPER_BF16_M16N8K16 : Pipeline := { groups := [16], width := 26, zeroExponent := -133 }

end Verity.TC
