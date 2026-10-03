import Pous.Protocol.Model.Bandwidth

/-!
# The host link's ceiling (named assumption)

A `Prop` taken as a hypothesis by the statements that use it, never an `axiom`: `TwoTierBandwidth`, which bounds what
the host link carries in a given time. It is the only physical premise of the bulk read's placement guarantees
(`Pous.Guarantees.HbmPlacement`, `Pous.Guarantees.HbmPlacementFStar`).
-/

namespace Pous.Assumptions

open Pous.Bandwidth
open scoped NNReal

/-- **Two-tier bandwidth: the host link's ceiling.** The store sits in two tiers: device memory (HBM), and host memory
for the blocks in `host`, which the device does not hold. A host-resident block reaches the device only across the host
link, which carries at most `BWpcie` bytes per unit time, and `b` bytes cross it for each distinct host-resident block
a response reads. So a response that reads `n` distinct host-resident blocks takes at least `b · n / BWpcie`.

It is a premise about the platform, for every program the server may run, and these are the readings it rests on:
- `BWpcie` is the link's ceiling, not a measured rate: about 63 GB/s for PCIe Gen5 x16 (128b/130b), where #935
  measured 56 GB/s for contiguous host-to-device copies. Every path from off-device memory (host DRAM, a peer GPU's
  memory over P2P, storage through GPUDirect) shares that link. On a platform with another host link, NVLink-C2C on
  GH200 at 900 GB/s, `BWpcie` is that link's rate.
- `b` is the bytes the link must carry for one block, at least the block's information content: for 16,384 words mod
  `2^31 − 1`, `16,384 · 31 / 8 = 63,488` bytes, not the 65,536 a `uint32` block occupies, since a responder may keep a
  host block in any encoding.
- No ceiling is assumed for device memory: the device tier may be read at any speed (L2 hits included).
-/
def TwoTierBandwidth {N R W : ℕ} {F : Type} [Fintype F] [Nonempty F] (host : Finset (Fin N)) (b BWpcie : ℝ≥0)
    (A : BulkResponder N R W F) : Prop :=
  ∀ (C : Store N W F) (k : Key F) (S : Fin R → Fin N),
    b * ((reads C (A.prog k S) ∩ host).card : ℝ≥0) ≤ BWpcie * A.time C k S

end Pous.Assumptions
