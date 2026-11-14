module

public import Definitions.Pouw.Advantage
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
public import Definitions.Pouw.PearlC.FlockTile.Audit
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
public import Definitions.Pouw.PearlC.TileDraw
public import Definitions.Pouw.PearlC.Unscale
public import Definitions.Pouw.PearlC.W1
public import Definitions.Pouw.PearlC.W1Stop
public import Definitions.Pouw.PearlC.WholeCircuit
public import Definitions.Pouw.PearlC.WholeCircuit4
public import Definitions.Pouw.Receipt.Duo
public import Definitions.Pouw.Receipt.Li
public import Definitions.Pouw.Receipt.Fx1
public import Definitions.Pouw.RelativeSecurity
public import Definitions.Pouw.RelativeSecurityJ10R
public import Mathlib.Data.Fintype.BigOperators
public import Mathlib.Data.Rat.Cast.Order
public import Definitions.Pouw.TileBound.Defs
public import Definitions.Pouw.TileBound.Fp8Defs
public import Definitions.Pouw.Window
public import Definitions.Pouw.Witness
public import Definitions.Pouw.Witness.Copy
public import Definitions.Pouw.Witness.Honest
public import Definitions.Pouw.Witness.Model

/-!
# PoUW's definitions (trusted)

PoUW, proof of useful work, bounds the work behind a prover's credit: what an adversary spends to pass the audit is at
least the honest work of its correct units, over `1 + ε`.

## Read first

* `Properties.Pouw.Period.H1CheckedCreditPeriod` (`Properties.Pouw.Period`): the credit over the windows the deployed
  check accepts, at H1's tile unit on the torch loop. It is PoUW's row in `security/README.md` and the lock's deployed
  PoUW guarantee.
* `Properties.Pouw.Period.CheckedCreditPeriod` (the same module): the general form it instantiates, the credit in live
  words over the windows the deployed check accepts.
* Resource accounting reads PoUW through two guarantees. `Accounting.Guarantees.ComputeOfPouw`
  (`Definitions.Accounting.Guarantees.Pouw`) is PoUW's guarantee as the compute guarantee.
  `Accounting.Guarantees.Compose.CircuitCredit` (`Definitions.Accounting.Guarantees.Compose`) is PoUW's credit on a
  registered circuit it compiles.

The area's own statements are `Definitions.Pouw.Guarantees` and the modules under it, proved under `Proofs.Pouw`.

## Assumptions

PoUW's premises are stated in `Definitions.Pouw.Assumptions` and the modules under it. The lock's named assumption
`Pouw.CostModel.Hashless` is declared in `Definitions.Assumptions.Pouw`.

## Map

This module imports the area's definitions; the lock's `layers` lists what they may import. A topic's
modules keep their basenames in `Definitions.Pouw`, `Definitions.Pouw.Assumptions`, `Definitions.Pouw.Guarantees` and
`Proofs.Pouw`, and the map names them by basename.

* The game. `Game` is the gap game, its chain's parameters and its deadline. `Basic` is integer matrices, uniform
  probability and adaptive oracle programs, and `MerkleExtract` Merkle commitments over an oracle and their extractor.
* `Advantage`: the advantage form `G_ε` at the adversary's advantage `ε`, in the game (`Pouw.Gε`) and in Pearl-C's
  (`Pouw.PearlC.Gε`). `GGap`, at the gap `ε/(1 + ε)`, is the same property, and the gap is how the proofs carry it.
  The guarantees in ε form are `Definitions.Pouw.Guarantees.Advantage`.
* The noise protocols. `D1` is the transcript with low-rank integer noise, `Dither` the no-fold Dither protocol's
  work-soundness, `DitherInstance` its model on the H200 and the RTX PRO 6000, and `Eps1` its words' model. `NCP` is
  noise that cancels inside one checked product, and `Dimension` the dimension bound.
* `Fp8Atom`: a bit-exact `HOPPER_E4M3_K32`. It holds the OCP FP8 codec, the step (`Atom`), the FP32 add of the running
  words (`Fp32`), the statements it pins (`Pinned`) and the conformance vectors (`Vectors`), which
  `verity/protocols/pouw/verifier/lean/scripts/fp8atom_vectors.py` generates.
* H1's tile chain, also in `Fp8Atom`. `H1T` is the forming map, the checked chain and its credited words, `H1TLemmas`
  and `H1TPinned` its statements, and `H1TCrossLate` the cross-column late pairs. Its conformance vectors (`H1TVec`,
  `H1TVectors`) are generated by `verity/protocols/pouw/verifier/lean/scripts/h1t_vectors.py`.
* `FpG1`: FP8-g1 and W4A8-g1's operands, in `Encoding`, `Gemm` and `H1` (g1 on sm_120's record H1).
  `Definitions.Accounting.Guarantees.FpG1`'s headlines read H1's leaves.
* `H1Zy`: H1's sampled-leaf statements on words, the bodies of the combine, lane and tile checks (`Combine`, `Lane`,
  `Tile`) on their primitives' word views.
* `PearlC`: Pearl-C's Π (`Defs`), its game (`Game`) and its named protocol and error (`Instance`). The device records
  and their prices are `Device` and `DevicePrices`, the hidden audit `Hidden` and `HiddenTileGame`, and the sampled
  audit `Tile`, `TileDraw` and `Completeness`. The units as core circuits are `RowCircuit`, `TileCircuit` and
  `WholeCircuit`, and Pearl-C4's FP4 step and chain `Fp4` and `Fp4Chain`.
* `Receipt`: the receipt folds `Duo`, `Fx1` (sm_120's) and `Li`, after
  `verity/protocols/pouw/receipts/receipt_folds.py`. Their binding statements are `Definitions.Pouw.Guarantees.Receipt`.
* `TileBound`: the FP4-tile model (`Defs`) and the FP8-tile model at sm_120's FP8 atom (`Fp8Defs`).
* `Window`: a window of calls, one sample that depends on the registration and then the calls' sessions in order, in
  the form of Flock's one-stage audit. Its statements are `Definitions.Pouw.Guarantees.Window`.
* The rest. `RelativeSecurity` and `RelativeSecurityJ10R` are relative security for matmul PoUW, the second at call
  scope. `OnsiteFreed` is the onsite protocol's verdict on a monitoring period, `Barrier` the degenerate-input barrier,
  and `Witness` the machine, protocol and period that `WorkSound`'s witnesses are stated at.

## Where to look

* Guarantees: `security/Definitions/Pouw/Guarantees.lean` and `Guarantees/`. Premises:
  `security/Definitions/Pouw/Assumptions.lean` and `Assumptions/`.
* Properties of the code: `security/Properties/Pouw/`. Proofs: `security/Proofs/Pouw.lean` and `security/Proofs/Pouw/`.
* Locks: `security/lean-audit.json` (the guarantees) and `security/Properties/lean-audit.json` (the properties).
* PoUW's Lean code and its vector scripts: `verity/protocols/pouw/verifier/lean/`.
-/

@[expose] public section
