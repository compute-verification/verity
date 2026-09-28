import Models.Pouw.Assumptions.Deadline
import Models.Pouw.Assumptions.Dimension.A2Words
import Models.Pouw.Assumptions.Dimension.Assumptions
import Models.Pouw.Assumptions.Dimension.Embed
import Models.Pouw.Assumptions.Dimension.EmbedAssumptions
import Models.Pouw.Assumptions.Dimension.Forest
import Models.Pouw.Assumptions.Dimension.Fp8Mma
import Models.Pouw.Assumptions.Dimension.H100
import Models.Pouw.Assumptions.Dimension.H100Assumptions
import Models.Pouw.Assumptions.Dimension.H100Cite
import Models.Pouw.Assumptions.Dimension.H100Dense
import Models.Pouw.Assumptions.Dimension.H100Floor
import Models.Pouw.Assumptions.Dimension.H100Frag
import Models.Pouw.Assumptions.Dimension.H100LoopFree
import Models.Pouw.Assumptions.Dimension.H100Measured
import Models.Pouw.Assumptions.Dimension.H100PeakTc
import Models.Pouw.Assumptions.Dimension.H100W
import Models.Pouw.Assumptions.Dimension.Mixing
import Models.Pouw.Assumptions.Dimension.OpenPipe
import Models.Pouw.Assumptions.Dimension.Pinned
import Models.Pouw.Assumptions.Dimension.Pipe
import Models.Pouw.Assumptions.Dimension.PipeAssumptions
import Models.Pouw.Assumptions.Dimension.PipeModel
import Models.Pouw.Assumptions.Dimension.RouteU
import Models.Pouw.Assumptions.Dimension.RouteUPipe
import Models.Pouw.Assumptions.Dimension.RouteUPipeAssumptions
import Models.Pouw.Assumptions.Fp8Atom.H1TAssumptions
import Models.Pouw.Assumptions.Fp8Atom.H1TGamma
import Models.Pouw.Assumptions.Fp8Atom.H1TRunningAssumptions
import Models.Pouw.Assumptions.Fp8Atom.H1TTile
import Models.Pouw.Assumptions.Fp8Atom.H1TTileAssumptions
import Models.Pouw.Assumptions.Game
import Models.Pouw.Assumptions.NCP.Assumptions
import Models.Pouw.Assumptions.NCP.Chain
import Models.Pouw.Assumptions.NCP.GamePinned
import Models.Pouw.Assumptions.NCP.RouteUAssumptions
import Models.Pouw.Assumptions.NCP.RouteUGame
import Models.Pouw.Assumptions.NCP.RouteUWord
import Models.Pouw.Assumptions.PearlC.Deadline
import Models.Pouw.Assumptions.PearlC.DeviceChainCap
import Models.Pouw.Assumptions.PearlC.DeviceChainCapKernel
import Models.Pouw.Assumptions.PearlC.HiddenTile
import Models.Pouw.Assumptions.PearlC.HonestCap
import Models.Pouw.Assumptions.PearlC.RowSeedAssumptions
import Models.Pouw.Assumptions.PearlC.TTOut
import Models.Pouw.Assumptions.PearlC.TTOutChainOnly
import Models.Pouw.Assumptions.PearlC.TTOutFp4
import Models.Pouw.Assumptions.PearlC.TTOutFp4ChainOnly
import Models.Pouw.Assumptions.PearlC.TTOutRev1
import Models.Pouw.Assumptions.PearlC.TTOutRowSeed
import Models.Pouw.Assumptions.PearlC.TTOutRowSeedK
import Models.Pouw.Assumptions.PearlC.TTOutTileUOnly
import Models.Pouw.Assumptions.PearlC.TTOutUOnly
import Models.Pouw.Assumptions.Pinned

/-!
# PoUW's named assumptions, and the definitions and statements over them (trusted)

The modules `lean-audit.json` lists under `assumptions` state the named assumptions, each a `Prop` taken as a
hypothesis by the statements that use it, never an `axiom`; the game's are in `Game`. The layer's other modules hold
the definitions and statements that read them, which an assumption module may import. They import `Definitions.Pouw` and
each other, never a proof.
-/
