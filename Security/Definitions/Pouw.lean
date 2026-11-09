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
# PoUW's protocol: the definitions the statements read

This layer, `Definitions.Pouw.Assumptions` and `Definitions.Pouw.Guarantees` are the trusted text: they import only each other,
Mathlib and core's `Definitions.Core` (`Definitions.Pouw.PearlC.Defs` also Batteries), so their diff, with core's, is everything a
statement reviewer reads.
`Proofs.Pouw` holds the proofs and every lemma, which nobody reviews for meaning. A topic's modules keep their
basenames in each layer (`Definitions.Pouw.Fp8Atom.E4M3`, `Proofs.Pouw.Fp8Atom.Proofs`), and the topics' maps
below name them by basename.

## `Advantage`: the advantage form

`G_ε` at the adversary's advantage `ε` with failure probability `δ(q, N)`, in the game (`Pouw.Gε`) and in Pearl-C's
(`Pouw.PearlC.Gε`): what an adversary spends is at least the honest work of its correct units over `1 + ε`. `GGap` at
the gap `g = ε/(1 + ε)` is the same property, and the gap is how the proofs carry it. The audit's statement in ε form
(`AuditGε`), the bound `Ω_max` the reduction from TT reads (`Protocol.WrefWithin`) and route U's instance conditions
(`NCP.Params.InstanceU`); the guarantees in ε form are `Definitions.Pouw.Guarantees.Advantage`, proved from their gap
forms in `Proofs.Pouw.Advantage`.

## `Fp8Atom`: a bit-exact `HOPPER_E4M3_K32`

The codec (`E4M3`), the step (`Atom`), the FP32 add of the running words (`Fp32`), the conformance vectors (`Vectors`,
generated by `verity/protocols/accounting/work/pouw/lean/scripts/fp8atom_vectors.py` from `verity.ml.tc` and numpy), their kernel checks (`Checks`), the pinned
statements (`Pinned`) and their proofs (`Proofs`, `Fp32Proofs`).

H-1T (M2): the forming map, the checked chain and its credited words (`H1T`), the conformance vectors (`H1TVec`,
`H1TVectors`, generated by `verity/protocols/accounting/work/pouw/lean/scripts/h1t_vectors.py`) and their kernel checks (`H1TChecks`), the named assumptions
`DistinctLiveH1T` and `HalfApartH1T` (`H1TAssumptions`), the pinned statements (`H1TPinned`, `H1TGap`) and their
proofs (`H1TProofs`, `H1TGapProofs`).

H-1T (M3, M4a): the lemma statements (`H1TLemmas`): the atom's value, the truncation's monotonicity and cell, one
slice's bounds, the chain's monotonicity, the range `|R| < 2^33`, the form table's coverage, and the one-flip separation
`H1TOneFlip`; their proofs (`H1TAtomProofs`, `H1TSliceProofs`, `H1TFlipProofs`). The pairs one flip cannot separate
(`Hard`, which since M4b reads the two words' columns, and since M4c the start of block 2) and the `Stuck` words'
non-constancy are the named assumption `H1TRunningWords` (`H1TRunningAssumptions`); `H1TRunningProofs` proves it
equivalent to `DistinctLiveH1T` and derives `H1TGap` from it. The statements its proof still needs, unpinned, are
staged in `H1TGapStaging`: the monotone chain gap, the last-slice split, and the cross-column late pairs.
`H1TCrossLateStaging` proves the cross-column late pairs in the exact-512 regime, from the statements in
`H1TCrossLate`; the five theorems are granted, not pinned (pin when used), and the regime is empty for
`k ≤ 19,430`. `H1TBoundedStaging` states the named assumptions and `H1TGap` for units of `k ≤ K`, proves the bounded
gap corollary, and bounds the running words below `2^31` for `k ≤ 9,048` and below `2^32` for `k ≤ 18,096`, unpinned.
`H1TShiftStaging` reduces the cross-column late pairs, where the chains after slice `T − 1`'s block 1 stay in one
binade, to one slice's two block-1 words against step-`W` staircases (`TagStaircaseSplit`, unproved); where a chain
crosses one power of two at one step, to the same with cuts (`TagCutSplit`, unproved); and below `2^e` it separates
them by the rounding margin outright, unpinned.

## `FpG1`: FP8-g1 and W4A8-g1's operands

The encodings (`Encoding`): E4M3's integer grid to a bound, the doubled E2M1 grid, round to nearest onto a grid, the
coin, an operand entry and a device's g1 grids. The int8 GEMM, the coin's rank-one terms and the epilogue (`Gemm`).
g1 on sm_120's record H1: its grids and a round's operand leaves (`H1`). `Proofs.Pouw.FpG1` proves the arithmetic.
`Definitions.Accounting.Guarantees.FpG1`'s headlines read H1's leaves; they are the record's bound at g1's windows, which
the record's headlines already cover by their own quantifier over committed values.

## `H1Zy`: H1's drawn-leaf statements on words

`Combine`: `PouwH1Combine_v1{TILE}`'s body (`h1zy._combine_body`) on its primitives' word views, the XOR and AND trees
as `h1tile._xor` and `h1zy._and_all` build them. `Proofs.Pouw.H1Zy` proves it the receipt check `LeafChecks` reads.

`Lane`: `PouwH1LaneZy_v1{TILE}`'s body (`h1zy._lane_body`) at a 32 × 32 leaf over `PouwH1Tile_v1` on its primitives'
word views: the index's mux trees, the EXACT z (`GemmAccumulatorS8_v2`), the cell check and the masked partials, with the
dequant's and the chain's words as parameters. `Proofs.Pouw.H1Zy` proves it the lane check `LeafChecks` reads.

`Tile`: `PouwH1TileZy_v1{TILE}`'s `ok` (`h1zy._body`) over `PouwH1Tile_v1` or `PouwH1Tile_v3` on its primitives' word
views: TILE's z (EXACT, or the one-stream's recombination, wrap correction and dither peel at its noise) and the
record's cell check. `Proofs.Pouw.H1Zy` proves it the check `TileChecks` reads.

## `KeyedLeaf`: the lead's keyed commitments

A window's view, the key leaf and each instance's keyed digest (`Hash.Keyed`), and the two games its hiding reads, each
with `A`'s distinguisher in it: hm96's hiding game and the keyed digest's game against a random function. Their
statements are `Definitions.Pouw.Guarantees.KeyedLeaf`, proved in `Proofs.Pouw.Salt`.

## `PearlC`: Pearl-C's Π and the statements about it

Π's shared definitions (`Defs`): the parameters, the E4M3 encoder, the BF16 conversion and the FP32 encoder, the row
statistic, the noise and the forming chains, the promoted product chain, and U's two forms. The conformance vectors of
the encoders, the conversion and the atom-mix chain (`EncVectors`, generated from `verity.ml.tc`) and their kernel
checks (`EncChecks`). P2, the clean-up chain (`Peel`): U's cancellation identities, where the rounding is, and the
clean-up terms' liveness at degenerate inputs. P1, freezing under promotion (`Freeze`), and P3, the forming credit
(`Forming`). The game (`Game`), named protocol (`Instance`), named assumption TT_OUT (`TTOut`) and the
gap from it (`Gap`). The hidden audit, the tile check as a proof over core's `Verity.Game` (`Hidden`), its open obligation
that a sampled tile's proof is sound (`HiddenTile`), what a proof system gives per tile to meet it (`HiddenTileGame`, met
in `SecurityProofs` through core's batch lemma) and the sampled audit's gap under it (`HiddenGap`). The sampled audit's
completeness on a population of activations (`Completeness`), its one assumption, the honest tiles' measured cap rate
(`HonestCap`, empirical), the proof that every honest word is an FP32 word's (`HonestWords`) and completeness from the
two (`Completeness`).

## `Receipt`: the receipt folds

`Duo`: duo's step, runs of one to four words and of any length (`benchmarks/pouw/receipts/receipt_folds.py`,
`duo_word`). `Fx1`: sm_120's fold, one keyed IMAD by an odd multiplier in front of duo's step (decision 83). `Li`: li's
half step, a chain of them and a checkpoint, the XOR of several chains' states (`li_step`). Their binding statements
are `Definitions.Pouw.Guarantees.Receipt`, proved in `Proofs.Pouw.Receipt`.

## `TileBound`: the FP4-tile model

`Defs`: the model over `BLACKWELL_SM120_NVF4`'s tile gate, with per-slot pricing and the rewrite class (trusted
definitions). `Proofs`: the block-support count and the cost bounds it gives.

## `Window`: a window of calls

One sample that depends on the registration (C1), then the calls' sessions in order, in the form of Flock's one-stage
audit: the game, the laws and their product (M1), the units' statements (self-contained tiles whose seed's position and
roots are read at their place in the committed transcript, decision 106) and PoUW's harm family (`harmSets`), each
call's analysis, the premises of the window's composition in the compiled tables' shape (a constant plus a term of the
prover's state) and the window's collision term (`coll`, bounded at the sessions' counted queries by `CollCounted`),
what the served pass committed (`ServedPass`) and the transcripts that are its (`Anchored`, `Served`), the tie to what's
served (`ServedTie`, open on main), and the deployed sample, sampled proofs' service's uniform subset in rank-ordered
sessions (`serviceLaw`, `sessionLaw`, `CopyClosed`) with its units' statements, the tiles' read in every session
(`UnitStmts.copies`, `WrongCopyClosed`).
The window's ε from its calls' (M2) and the service's sample at `η = 0` are `Definitions.Pouw.Guarantees.Window`, stated and
not yet proved. `Proofs.Pouw.Window`: the deployed wrong sets are copy-closed, PoUW's reading with the served tie, the
collision term on average over the oracle, and Flock's analysis of a session as the window's call's, with Flock's
knowledge bound at main's `--zk` sessions.
-/

@[expose] public section
