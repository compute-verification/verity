/-!
# Grouped-sum pipelines (`verity.ml.tc.models.GroupSum`)

The parameters both the semantics (`Verity.TC.Spec`) and the relation (`Verity.TC.Relation`) read: the group sizes in
k order, the adder's width and the exponent floor. BF16 operands only: the relation and the semantics are
defined for BF16 pipelines, whose width is at least 24.
-/

namespace Verity.TC

/-- Python's `int.bit_length` on a natural number. -/
def bitLength (n : Nat) : Nat := if n = 0 then 0 else n.log2 + 1

/-- `models.GroupSum` for a BF16 pipeline: `groups` in k order, the adder's `width`, the exponent floor `zeroExponent`. -/
structure Pipeline where
  groups : List Nat
  width : Nat
  zeroExponent : Int
deriving DecidableEq

/-- `GroupSum.k`: products per step. -/
def Pipeline.k (p : Pipeline) : Nat := p.groups.sum

/-- `models.HOPPER_BF16_WGMMA_K16`: one accumulator coordinate of `wgmma.mma_async...m64nNk16.f32.bf16.bf16` (H100). -/
def HOPPER_BF16_WGMMA_K16 : Pipeline := { groups := [16], width := 26, zeroExponent := -133 }

/-- `models.HOPPER_BF16_M16N8K16`: `mma.sync.m16n8k16.bf16` on Hopper (H100) and on sm_120 (RTX 5090). -/
def HOPPER_BF16_M16N8K16 : Pipeline := { groups := [16], width := 26, zeroExponent := -133 }

/-- `models.AMPERE_BF16_M16N8K16`: `mma.sync.m16n8k16.bf16` on Ampere (A100). -/
def AMPERE_BF16_M16N8K16 : Pipeline := { groups := [8, 8], width := 25, zeroExponent := -132 }

/-- `models.ADA_BF16_M16N8K16`: `mma.sync.m16n8k16.bf16` on Ada (RTX 4090), Ampere's parameters. -/
def ADA_BF16_M16N8K16 : Pipeline := { groups := [8, 8], width := 25, zeroExponent := -132 }

end Verity.TC
