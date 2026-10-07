import Definitions.Pouw.PearlC.Device

/-!
# The RTX PRO 6000's W1 prices (trusted; definitions only)

Measured on node 2 at locked 2,100 MHz (`r20260930-063213-c94a`, `r20260930-061717-3e84`), in units of one dense E4M3
`mma.sync` MAC with FP32 accumulation. **A tensor-core instruction is priced by its whole fragment**: an E4M3 `m16n8k32`
costs 4,096 units and a BF16 `m16n8k16` 2,048 MACs at 2, however few of its outputs a program uses (`w1-complete/sm120`).
A per-MAC price here is a share of a whole instruction, not a price per output word. Each credited step is at its
cheapest bit-exact implementation (C3's charging rule), which is the issue-bound price:
* an FP32 add is 8.00 per word (8.376 in the measured loop, which carries 3 loop instructions per 64);
* a BF16 MAC is 2.000;
* the credited forming is the k32 noise atom (32 MACs at full rate) plus the E4M3 cast at 8.0 per code alone (12.31 in
  the measured packing loop): `fs = 40`;
* the A-only forming is ρ's squared accumulate on every 8th element, one FFMA at 8.00: `qa = 1`.

sm_120a has no packed FP32 add (no FADD2): `add.rn.f32x2` compiles to two scalar FADDs (`r20260930-081301-747b`), so it
does not halve the add, which is 8.376 per FADD in the measured loop and 8.00 issue-bound; `add` stays 8.
-/

namespace Pouw.PearlC

/-- sm_120's W1 prices, issue-bound: an FP32 add 8, a BF16 MAC 2, credited forming 40 and A-only forming 1 per element. -/
def Prices.sm120 : Prices := ⟨8, 2, ⟨40, 1⟩⟩

end Pouw.PearlC
