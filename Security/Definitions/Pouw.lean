module

public import Definitions.Pouw.Barrier.Defs
public import Definitions.Pouw.Barrier.Pinned
public import Definitions.Pouw.Basic.Mat
public import Definitions.Pouw.Basic.Prob
public import Definitions.Pouw.Basic.QTree
public import Definitions.Pouw.D1
public import Definitions.Pouw.Dither
public import Definitions.Pouw.DitherInstance
public import Definitions.Pouw.Eps1.M3
public import Definitions.Pouw.Dimension.Causal
public import Definitions.Pouw.Dimension.Defs
public import Definitions.Pouw.Dimension.Lift
public import Definitions.Pouw.Fp8Atom.Atom
public import Definitions.Pouw.Fp8Atom.E4M3
public import Definitions.Pouw.Fp8Atom.Fp32
public import Definitions.Pouw.Fp8Atom.H1T
public import Definitions.Pouw.Fp8Atom.H1TCrossLate
public import Definitions.Pouw.Fp8Atom.H1TLemmas
public import Definitions.Pouw.Fp8Atom.H1TPinned
public import Definitions.Pouw.Fp8Atom.H1TVec
public import Definitions.Pouw.Fp8Atom.H1TVectors
public import Definitions.Pouw.Fp8Atom.Pinned
public import Definitions.Pouw.Fp8Atom.Vectors
public import Definitions.Pouw.Fp8Atom.Vectors.Ada
public import Definitions.Pouw.Fp8Atom.Vectors.Add
public import Definitions.Pouw.Fp8Atom.Vectors.C00
public import Definitions.Pouw.Fp8Atom.Vectors.C01
public import Definitions.Pouw.FpG1
public import Definitions.Pouw.Game.Deadline
public import Definitions.Pouw.Game.Defs
public import Definitions.Pouw.Game.Params
public import Definitions.Pouw.H1Zy
public import Definitions.Pouw.KeyedLeaf
public import Definitions.Pouw.M1
public import Definitions.Pouw.MerkleExtract
public import Definitions.Pouw.NCP.Defs
public import Definitions.Pouw.NCP.ExtraIdentity
public import Definitions.Pouw.NCP.Pinned
public import Definitions.Pouw.NCP.Protocol
public import Definitions.Pouw.NCP.Relation
public import Definitions.Pouw.NCP.RouteU
public import Definitions.Pouw.NCP.Witness
public import Definitions.Pouw.NCP.Word
public import Definitions.Pouw.OnsiteFreed
public import Definitions.Pouw.PearlC.Accum
public import Definitions.Pouw.PearlC.Completeness
public import Definitions.Pouw.PearlC.Deadline
public import Definitions.Pouw.PearlC.Defs
public import Definitions.Pouw.PearlC.Device
public import Definitions.Pouw.PearlC.DeviceChainOnly
public import Definitions.Pouw.PearlC.DeviceFp4
public import Definitions.Pouw.PearlC.DeviceFp4ChainOnly
public import Definitions.Pouw.PearlC.DeviceFp4Hot
public import Definitions.Pouw.PearlC.DeviceFp4Issue
public import Definitions.Pouw.PearlC.DeviceKernelWref
public import Definitions.Pouw.PearlC.DevicePrices
public import Definitions.Pouw.PearlC.DevicePricesLoop
public import Definitions.Pouw.PearlC.DeviceRev1
public import Definitions.Pouw.PearlC.EncVectors
public import Definitions.Pouw.PearlC.FormingPDefs
public import Definitions.Pouw.PearlC.Fp4
public import Definitions.Pouw.PearlC.Fp4Chain
public import Definitions.Pouw.PearlC.Fp4Skip
public import Definitions.Pouw.PearlC.Fp4Vectors
public import Definitions.Pouw.PearlC.Game
public import Definitions.Pouw.PearlC.Hidden
public import Definitions.Pouw.PearlC.HiddenTileGame
public import Definitions.Pouw.PearlC.Instance
public import Definitions.Pouw.PearlC.NoiseSha512
public import Definitions.Pouw.PearlC.PeelExact
public import Definitions.Pouw.PearlC.PeelExactP
public import Definitions.Pouw.PearlC.PeelExactVectors
public import Definitions.Pouw.PearlC.PeelFp4Vectors
public import Definitions.Pouw.PearlC.Row4Circuit
public import Definitions.Pouw.PearlC.RowCircuit
public import Definitions.Pouw.PearlC.RowPrimVectors
public import Definitions.Pouw.PearlC.RowPrims
public import Definitions.Pouw.PearlC.RowSem
public import Definitions.Pouw.PearlC.SaltDead
public import Definitions.Pouw.PearlC.Sha512
public import Definitions.Pouw.PearlC.Skip
public import Definitions.Pouw.PearlC.SkipP
public import Definitions.Pouw.PearlC.Sm120
public import Definitions.Pouw.PearlC.Sm120Vectors
public import Definitions.Pouw.PearlC.Tile
public import Definitions.Pouw.PearlC.Tile4Circuit
public import Definitions.Pouw.PearlC.TileCheck
public import Definitions.Pouw.PearlC.TileCheck8
public import Definitions.Pouw.PearlC.TileCheck4
public import Definitions.Pouw.PearlC.TileCircuit
public import Definitions.Pouw.PearlC.W1
public import Definitions.Pouw.PearlC.W1Stop
public import Definitions.Pouw.PearlC.WholeCircuit
public import Definitions.Pouw.PearlC.WholeCircuit4
public import Definitions.Pouw.Receipt.Duo
public import Definitions.Pouw.Receipt.Li
public import Definitions.Pouw.Receipt.Fx1
public import Definitions.Pouw.RelativeSecurity
public import Definitions.Pouw.RelativeSecurityJ10R
public import Definitions.Pouw.TileBound.AlignedExactDefs
public import Definitions.Pouw.TileBound.Defs
public import Definitions.Pouw.TileBound.Fp8Defs
public import Definitions.Pouw.Window
public import Definitions.Pouw.Witness
public import Definitions.Pouw.Witness.Copy
public import Definitions.Pouw.Witness.Model

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

## `Fp8Atom`: a bit-exact `HOPPER_E4M3_K32`

Its codec, step and FP32 add, its conformance vectors from `verity/core/protocols/pouw/lean/scripts/fp8atom_vectors.py`
and their kernel checks. H-1T's forming map, checked chain and credited words (`H1T`, vectors from `h1t_vectors.py`
beside it), its named assumptions, and its per-tile count (`H1TTile`).

## `FpG1`: FP8-g1 and W4A8-g1's operands

The operand encodings, the int8 GEMM and its epilogue, and g1 on sm_120's record H1.

## `H1Zy`: H1's drawn-leaf statements on words

H1's sampled-leaf checks on words (`Combine`, `Lane`, `Tile`), as `verity/core/protocols/pouw/circuit/h1zy.py` builds
them.

## `KeyedLeaf`: the lead's keyed commitments

A window's view, the key leaf and each instance's keyed digest (`Hash.Keyed`), and the two games its hiding reads, each
with `A`'s distinguisher in it: hm96's hiding game and the keyed digest's game against a random function. Their
statements are `Definitions.Pouw.Guarantees.KeyedLeaf`, proved in `Proofs.Pouw.Salt`.

## `PearlC`: Pearl-C's Π and the statements about it

Its shared definitions (`Defs`), the clean-up chain (`Peel`), freezing and the forming credit, the game and TT_OUT
(`TTOut`), the hidden tile audit (`Hidden`, `HiddenTile`, `HiddenGamma`) and its completeness.

## `Receipt`: the receipt folds

Duo, sm_120's `Fx1` and li (`benchmarks/pouw/receipts/receipt_folds.py`), whose binding statements are
`Definitions.Pouw.Guarantees.Receipt`.

## `TileBound`: the FP4-tile model

The model over `BLACKWELL_SM120_NVF4`'s tile gate.

## `Window`: a window of calls

An execution phase's calls, one sample that depends on the registration and then the calls' sessions in order, in the
form of Flock's one-stage audit; its guarantees (`Definitions.Pouw.Guarantees.Window`) are open.

## The other topics

* `Dither`, `DitherInstance`, `Eps1`: the Dither protocol, its model on the H200 and the RTX PRO 6000, and the model of
  its words.
* `D1`, `NCP`, `Barrier`, `Dimension`, `Basic`: protocol D1's low-rank noise and its named instance, noise that cancels
  inside one checked product, the degenerate-input barrier, Theorem D with causal inputs, and integer matrices.
* `RelativeSecurity`, `RelativeSecurityJ10R`: relative security for matmul PoUW, and at call scope under J10-R.
* `OnsiteFreed`, `Game`, `MerkleExtract`, `Witness`: the onsite verdict on a monitoring period, the deadline, Merkle
  commitments over an oracle with their extractor, and the witnesses `WorkSound`'s are stated at.
-/

@[expose] public section
