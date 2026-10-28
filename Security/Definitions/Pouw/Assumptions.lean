import Definitions.Pouw.Assumptions.Deadline
import Definitions.Pouw.Assumptions.Dimension.A2Words
import Definitions.Pouw.Assumptions.Dimension.Assumptions
import Definitions.Pouw.Assumptions.Dimension.Embed
import Definitions.Pouw.Assumptions.Dimension.EmbedAssumptions
import Definitions.Pouw.Assumptions.Dimension.Forest
import Definitions.Pouw.Assumptions.Dimension.Fp8Mma
import Definitions.Pouw.Assumptions.Dimension.H100
import Definitions.Pouw.Assumptions.Dimension.H100Assumptions
import Definitions.Pouw.Assumptions.Dimension.H100Cite
import Definitions.Pouw.Assumptions.Dimension.H100Dense
import Definitions.Pouw.Assumptions.Dimension.H100Floor
import Definitions.Pouw.Assumptions.Dimension.H100Frag
import Definitions.Pouw.Assumptions.Dimension.H100LoopFree
import Definitions.Pouw.Assumptions.Dimension.H100Measured
import Definitions.Pouw.Assumptions.Dimension.H100PeakTc
import Definitions.Pouw.Assumptions.Dimension.H100W
import Definitions.Pouw.Assumptions.Dimension.Mixing
import Definitions.Pouw.Assumptions.Dimension.OpenPipe
import Definitions.Pouw.Assumptions.Dimension.Pinned
import Definitions.Pouw.Assumptions.Dimension.Pipe
import Definitions.Pouw.Assumptions.Dimension.PipeAssumptions
import Definitions.Pouw.Assumptions.Dimension.PipeModel
import Definitions.Pouw.Assumptions.Dimension.RouteU
import Definitions.Pouw.Assumptions.Dimension.RouteUPipe
import Definitions.Pouw.Assumptions.Dimension.RouteUPipeAssumptions
import Definitions.Pouw.Assumptions.Dither
import Definitions.Pouw.Assumptions.DitherInstance
import Definitions.Pouw.Assumptions.Fp8Atom.H1TAssumptions
import Definitions.Pouw.Assumptions.Fp8Atom.H1TGamma
import Definitions.Pouw.Assumptions.Fp8Atom.H1TRunningAssumptions
import Definitions.Pouw.Assumptions.Fp8Atom.H1TTile
import Definitions.Pouw.Assumptions.Fp8Atom.H1TTileAssumptions
import Definitions.Pouw.Assumptions.Game
import Definitions.Pouw.Assumptions.NCP.Assumptions
import Definitions.Pouw.Assumptions.NCP.Chain
import Definitions.Pouw.Assumptions.NCP.GamePinned
import Definitions.Pouw.Assumptions.NCP.RouteUAssumptions
import Definitions.Pouw.Assumptions.NCP.RouteUGame
import Definitions.Pouw.Assumptions.NCP.RouteUWord
import Definitions.Pouw.Assumptions.OnsiteFreed
import Definitions.Pouw.Assumptions.PearlC.DeviceChainCap
import Definitions.Pouw.Assumptions.PearlC.DeviceChainCapKernel
import Definitions.Pouw.Assumptions.PearlC.HiddenTile
import Definitions.Pouw.Assumptions.PearlC.HonestCap
import Definitions.Pouw.Assumptions.PearlC.RowSeedAssumptions
import Definitions.Pouw.Assumptions.PearlC.TTOut
import Definitions.Pouw.Assumptions.PearlC.TTOutChainOnly
import Definitions.Pouw.Assumptions.PearlC.TTOutChainOnlyW1
import Definitions.Pouw.Assumptions.PearlC.TTOutFp4
import Definitions.Pouw.Assumptions.PearlC.TTOutFp4ChainOnly
import Definitions.Pouw.Assumptions.PearlC.TTOutFp4W1
import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
import Definitions.Pouw.Assumptions.PearlC.TTOutRowSeed
import Definitions.Pouw.Assumptions.PearlC.TTOutRowSeedK
import Definitions.Pouw.Assumptions.PearlC.TTOutTileUOnly
import Definitions.Pouw.Assumptions.PearlC.TTOutUOnly
import Definitions.Pouw.Assumptions.PearlC.TTOutW1
import Definitions.Pouw.Assumptions.Pinned
import Definitions.Pouw.Assumptions.RelativeSecurity
import Definitions.Pouw.Assumptions.ServedWork
import Definitions.Pouw.Assumptions.ShaRealize
/-!
# PoUW's named assumptions, and the definitions and statements over them (trusted)

The modules `lean-audit.json` lists under `assumptions` state the named assumptions, each a `Prop` taken as a
hypothesis by the statements that use it, never an `axiom`; the game's are in `Game`. The layer's other modules hold
the definitions and statements that read them, which an assumption module may import. They import `Definitions.Pouw` and
each other, never a proof.
-/
