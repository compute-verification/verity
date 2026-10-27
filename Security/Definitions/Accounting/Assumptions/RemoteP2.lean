import Definitions.Accounting.Assumptions.RateAudit
import Definitions.Accounting.Assumptions.RemoteHeld

/-!
# The remote audit over PoUS's P2 store: its codec's work and the squaring ceiling at P2 (trusted; drafted)

The remote bandwidth audit's P2 mode (`verity/core/protocols/remote_bandwidth/p2.py`, its `PROTOCOL.md` §11) audits
PoUS's `C*`: the weights `W` encoded under P2 (`p2-16448/v4`, `p = 2^16448 − 21065`), a block 2,056 bytes, and a
receipt of a unit of four blocks (8,224 bytes, 257 sectors), the trace's block (`Accounting.Remote.Held.Trace`). Its
memory line debits what the node's compute can re-encode, as the Sloth store's does, and these are the codec's terms at
P2's form, each a named `Prop`:

* `WorkHardnessP2 c_root cost`: `WorkHardness` (`rate-codec-work`'s form) at P2's block, `b = 16,448` bits and two
  roots, each priced `c_root`.
* `SquaringCeilingP2 T s B`: the held line's `SquaringCeiling` at `E = 4 · 32,890` squarings a unit.

**Drafted, not decided**: these claims await a ruling. No guarantee takes either yet, and
`audit.ledger` refuses a P2 record (`audit.Rebuild` refuses a P2 codec) until one does. The squaring rate `s` is not
measured: the H200's at 16,448 bits is the measurement the stated instance waits on.
-/

namespace Accounting.Assumptions

/-- **`WorkHardness` at P2's form** (drafted, not decided; proposed claim `p2-codec-work`). For every `x ≤ 16,448`,
completing `x` of a P2 block's 16,448 bits costs at least `x/16,448` of its encode, two modular square roots at
16,448 bits, each priced `c_root`:

  `x · (2 · c_root) ≤ 16,448 · cost x`,

`cost x` the least compute in which the stated device obtains `x` bits of a block of `C*` it neither holds nor reads,
from setup (the keys `K_j`, `t_j`), the payload the block encodes (`W`'s 2,054 bytes, which the verifier and the prover
both know) and anything else it holds, the block's other `16,448 − x` bits among them, one block at a time.

**The form.** P2's encode is `c = ρ(σ_K(ρ(m + t)))` (`pous.schemes.p2`): two roots `ρ`, each an exponentiation by
`(p + 1)/4` at 16,448 bits, with a public XOR `σ_K` between them, the shape of the masked word codec at `k = 2`
(`WorkHardness b 2 c_root cost`) at one word a block. A root is at least `bitlen((p + 1)/4) − 1 = 16,445` squarings, so
at `c_root = 16,445` a block's encode is 32,890 (the P2 store's `codec.P2Codec.block_encode_squarings`); PoUS's ring
model's recomputing responder spends `2 · 16,447 = 32,894` multiplication rounds a block (`PousP2Ring.tight_p16448`),
above it.

**Where it stands.** No attack has been tried at P2's width. The nearest evidence is the masked word codec's: no 4-dimensional
completion of a masked word at any width tried (none of 200), where the unmasked `k = 2`
codec completes up to `x ≈ b/10`. P2 differs in its width (one 16,448-bit word a block, where a
lattice's dimension and reduction cost grow) and in its offset `t_j`. PoUS's guarantee prices its responder in
sequential depth (`P2RingAuditSecure`), not in total work, so it does not give this form. -/
def WorkHardnessP2 (cRoot : ℕ) (cost : ℕ → ℕ) : Prop := WorkHardness 16448 2 cRoot cost

end Accounting.Assumptions

namespace Accounting.Remote.Held.Assumptions

variable {Ω : Type}

/-- **The squaring ceiling at P2's encode** (drafted, not decided). On every outcome, over every interval `[u, v)`, the
node's compute re-encodes at most `s / (4 · 32,890) · (v − u) + B` units of a P2 store:

* `s` squarings a tick at 16,448 bits over every GPU and the host, the adversary's best measured rate (as
  `Accounting.Assumptions.SquaringCeiling` prices the word codec's, at about 1.485× its reference kernel on the GPU);
* a unit, four 2,056-byte blocks, at least `4 · 32,890 = 131,560` of them: under `WorkHardnessP2` at
  `c_root = 16,445`, completing any `x` of a unit's `4 · 16,448` bits, spread over its blocks however, costs at least
  `x/(4 · 16,448)` of four blocks' encodes, so the trace's `rebuilt` counts completed bits in units;
* `B` one maximal rebuild batch.

**Unmeasured**: `s` at P2's width on the H200. The honest encode is 664.8 blocks a second on one H200
(98.58 s a 65,536-block segment), about `2.19·10^7` squarings a second, a floor and not the
ceiling; the ceiling wants the best kernel's rate at 16,448 bits, every GPU's and the host's, which is also the rate
PoUS's `ring-op-ceiling` reads. -/
def SquaringCeilingP2 (T : Trace Ω) (s B : ℝ) : Prop := SquaringCeiling T s (4 * 32890) B

end Accounting.Remote.Held.Assumptions
