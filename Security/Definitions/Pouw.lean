import Definitions.Pouw.Barrier.Defs
import Definitions.Pouw.Barrier.Pinned
import Definitions.Pouw.Basic.Mat
import Definitions.Pouw.Basic.Prob
import Definitions.Pouw.Basic.QTree
import Definitions.Pouw.D1
import Definitions.Pouw.Dither
import Definitions.Pouw.DitherInstance
import Definitions.Pouw.Eps1.M3
import Definitions.Pouw.Dimension.Causal
import Definitions.Pouw.Dimension.Defs
import Definitions.Pouw.Dimension.Lift
import Definitions.Pouw.Fp8Atom.Atom
import Definitions.Pouw.Fp8Atom.E4M3
import Definitions.Pouw.Fp8Atom.Fp32
import Definitions.Pouw.Fp8Atom.H1T
import Definitions.Pouw.Fp8Atom.H1TCrossLate
import Definitions.Pouw.Fp8Atom.H1TLemmas
import Definitions.Pouw.Fp8Atom.H1TPinned
import Definitions.Pouw.Fp8Atom.H1TVec
import Definitions.Pouw.Fp8Atom.H1TVectors
import Definitions.Pouw.Fp8Atom.Pinned
import Definitions.Pouw.Fp8Atom.Vectors
import Definitions.Pouw.Fp8Atom.Vectors.Ada
import Definitions.Pouw.Fp8Atom.Vectors.Add
import Definitions.Pouw.Fp8Atom.Vectors.C00
import Definitions.Pouw.Fp8Atom.Vectors.C01
import Definitions.Pouw.FpG1
import Definitions.Pouw.Game.Deadline
import Definitions.Pouw.Game.Defs
import Definitions.Pouw.Game.Params
import Definitions.Pouw.H1Zy
import Definitions.Pouw.M1
import Definitions.Pouw.MerkleExtract
import Definitions.Pouw.NCP.Defs
import Definitions.Pouw.NCP.ExtraIdentity
import Definitions.Pouw.NCP.Pinned
import Definitions.Pouw.NCP.Protocol
import Definitions.Pouw.NCP.Relation
import Definitions.Pouw.NCP.RouteU
import Definitions.Pouw.NCP.Witness
import Definitions.Pouw.NCP.Word
import Definitions.Pouw.OnsiteFreed
import Definitions.Pouw.PearlC.Accum
import Definitions.Pouw.PearlC.Completeness
import Definitions.Pouw.PearlC.Deadline
import Definitions.Pouw.PearlC.Defs
import Definitions.Pouw.PearlC.Device
import Definitions.Pouw.PearlC.DeviceChainOnly
import Definitions.Pouw.PearlC.DeviceFp4
import Definitions.Pouw.PearlC.DeviceFp4ChainOnly
import Definitions.Pouw.PearlC.DeviceFp4Hot
import Definitions.Pouw.PearlC.DeviceFp4Issue
import Definitions.Pouw.PearlC.DeviceKernelWref
import Definitions.Pouw.PearlC.DevicePrices
import Definitions.Pouw.PearlC.DevicePricesLoop
import Definitions.Pouw.PearlC.DeviceRev1
import Definitions.Pouw.PearlC.EncVectors
import Definitions.Pouw.PearlC.FormingPDefs
import Definitions.Pouw.PearlC.Fp4
import Definitions.Pouw.PearlC.Fp4Chain
import Definitions.Pouw.PearlC.Fp4Skip
import Definitions.Pouw.PearlC.Fp4Vectors
import Definitions.Pouw.PearlC.Game
import Definitions.Pouw.PearlC.Hidden
import Definitions.Pouw.PearlC.HiddenTileGame
import Definitions.Pouw.PearlC.Instance
import Definitions.Pouw.PearlC.NoiseSha512
import Definitions.Pouw.PearlC.PeelExact
import Definitions.Pouw.PearlC.PeelExactP
import Definitions.Pouw.PearlC.PeelExactVectors
import Definitions.Pouw.PearlC.PeelFp4Vectors
import Definitions.Pouw.PearlC.Row4Circuit
import Definitions.Pouw.PearlC.RowCircuit
import Definitions.Pouw.PearlC.RowPrimVectors
import Definitions.Pouw.PearlC.RowPrims
import Definitions.Pouw.PearlC.RowSem
import Definitions.Pouw.PearlC.SaltDead
import Definitions.Pouw.PearlC.Sha512
import Definitions.Pouw.PearlC.Skip
import Definitions.Pouw.PearlC.SkipP
import Definitions.Pouw.PearlC.Sm120
import Definitions.Pouw.PearlC.Sm120Vectors
import Definitions.Pouw.PearlC.Tile
import Definitions.Pouw.PearlC.Tile4Circuit
import Definitions.Pouw.PearlC.TileCheck
import Definitions.Pouw.PearlC.TileCheck8
import Definitions.Pouw.PearlC.TileCheck4
import Definitions.Pouw.PearlC.TileCircuit
import Definitions.Pouw.PearlC.W1
import Definitions.Pouw.PearlC.W1Stop
import Definitions.Pouw.PearlC.WholeCircuit
import Definitions.Pouw.PearlC.WholeCircuit4
import Definitions.Pouw.Receipt.Duo
import Definitions.Pouw.Receipt.Li
import Definitions.Pouw.Receipt.Fx1
import Definitions.Pouw.RelativeSecurity
import Definitions.Pouw.RelativeSecurityJ10R
import Definitions.Pouw.TileBound.AlignedExactDefs
import Definitions.Pouw.TileBound.Defs
import Definitions.Pouw.TileBound.Fp8Defs
import Definitions.Pouw.Window
import Definitions.Pouw.Witness
import Definitions.Pouw.Witness.Copy
import Definitions.Pouw.Witness.Model

/-!
# PoUW: the definitions its guarantees read

PoUW credits the work of a monitoring period's accepted runs: except with the probability its guarantee states, the
words an accepted check credits are words the hardware computed.

Read first:
* `Properties.Pouw.Period.H1CheckedCreditPeriod` (module `Properties.Pouw.Period`), the guarantee Security/README.md's
  table lists for PoUW: the credit over the windows the deployed check accepts, at H1's tile unit, an instance of the
  general `Properties.Pouw.Period.CheckedCreditPeriod`.
* `Pouw.Guarantees.Dither.WorkSound` and `Pouw.Guarantees.Dither.FleetSound` (`Definitions.Pouw.Guarantees.Dither`):
  the work soundness of Dither, the no-fold design the served PoUW moves to, with its card instances in
  `Definitions.Pouw.Guarantees.DitherInstance`.
* `Pouw.Guarantees.EndToEnd` (`Definitions.Pouw.Guarantees.Game`): TT gives resource exhaustion.

The named assumptions are `Definitions.Pouw.Assumptions` and the guarantees `Definitions.Pouw.Guarantees`, each by
topic. This module, `Assumptions` and `Guarantees` are the spec: they import only one another, Mathlib and
`Definitions.Core` (`Definitions.Pouw.PearlC.Defs` also Batteries). The proofs are `Proofs.Pouw`, where a topic's
modules keep their basenames (`Definitions.Pouw.Fp8Atom.Atom`, `Proofs.Pouw.Fp8Atom.Proofs`).

The topics:
* `Fp8Atom`: a bit-exact `HOPPER_E4M3_K32` (its codec, step and FP32 add), its conformance vectors from
  `verity/core/protocols/pouw/lean/scripts/fp8atom_vectors.py` and their kernel checks. H-1T's forming map, checked
  chain and credited words (`H1T`, vectors from `h1t_vectors.py` beside it), its named assumptions, and its per-tile
  count (`H1TTile`).
* `FpG1`: FP8-g1 and W4A8-g1's operand encodings, the int8 GEMM and its epilogue, and g1 on sm_120's record H1.
* `H1Zy`: H1's sampled-leaf checks on words (`Combine`, `Lane`, `Tile`), as
  `verity/core/protocols/pouw/circuit/h1zy.py` builds them.
* `PearlC`: Pearl-C's Π: its shared definitions (`Defs`), the clean-up chain (`Peel`), freezing and the forming credit,
  the game and TT_OUT (`TTOut`), the hidden tile audit (`Hidden`, `HiddenTile`, `HiddenGamma`) and its completeness.
* `Receipt`: the receipt folds duo, sm_120's `Fx1` and li (`benchmarks/pouw/receipts/receipt_folds.py`), whose binding
  statements are `Definitions.Pouw.Guarantees.Receipt`.
* `TileBound`: the FP4-tile model over `BLACKWELL_SM120_NVF4`'s tile gate.
* `Window`: an execution phase's calls, one sample that depends on the registration and then the calls' sessions in
  order, in the form of Flock's one-stage audit; its guarantees (`Definitions.Pouw.Guarantees.Window`) are open.
* `Dither`, `DitherInstance`, `Eps1`: the Dither protocol, its model on the H200 and the RTX PRO 6000, and the model of
  its words.
* `D1`, `NCP`, `Barrier`, `Dimension`, `Basic`: protocol D1's low-rank noise and its named instance, noise that cancels
  inside one checked product, the degenerate-input barrier, Theorem D with causal inputs, and integer matrices.
* `RelativeSecurity`, `RelativeSecurityJ10R`: relative security for matmul PoUW, and at call scope under J10-R.
* `OnsiteFreed`, `Game`, `MerkleExtract`, `Witness`: the onsite verdict on a monitoring period, the deadline, Merkle
  commitments over an oracle with their extractor, and the witnesses `WorkSound`'s are stated at.
-/
