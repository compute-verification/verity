import Pouw.Assumptions.Deadline
import Pouw.Assumptions.Dimension.A2Words
import Pouw.Assumptions.Dimension.Assumptions
import Pouw.Assumptions.Dimension.Embed
import Pouw.Assumptions.Dimension.EmbedAssumptions
import Pouw.Assumptions.Dimension.Forest
import Pouw.Assumptions.Dimension.Fp8Mma
import Pouw.Assumptions.Dimension.H100
import Pouw.Assumptions.Dimension.H100Assumptions
import Pouw.Assumptions.Dimension.H100Cite
import Pouw.Assumptions.Dimension.H100Dense
import Pouw.Assumptions.Dimension.H100Floor
import Pouw.Assumptions.Dimension.H100Frag
import Pouw.Assumptions.Dimension.H100LoopFree
import Pouw.Assumptions.Dimension.H100Measured
import Pouw.Assumptions.Dimension.H100PeakTc
import Pouw.Assumptions.Dimension.H100W
import Pouw.Assumptions.Dimension.Mixing
import Pouw.Assumptions.Dimension.OpenPipe
import Pouw.Assumptions.Dimension.Pinned
import Pouw.Assumptions.Dimension.Pipe
import Pouw.Assumptions.Dimension.PipeAssumptions
import Pouw.Assumptions.Dimension.PipeModel
import Pouw.Assumptions.Dimension.RouteU
import Pouw.Assumptions.Dimension.RouteUPipe
import Pouw.Assumptions.Dimension.RouteUPipeAssumptions
import Pouw.Assumptions.Fp8Atom.H1TAssumptions
import Pouw.Assumptions.Fp8Atom.H1TGamma
import Pouw.Assumptions.Fp8Atom.H1TRunningAssumptions
import Pouw.Assumptions.Fp8Atom.H1TTile
import Pouw.Assumptions.Fp8Atom.H1TTileAssumptions
import Pouw.Assumptions.Game
import Pouw.Assumptions.NCP.Assumptions
import Pouw.Assumptions.NCP.Chain
import Pouw.Assumptions.NCP.GamePinned
import Pouw.Assumptions.NCP.RouteUAssumptions
import Pouw.Assumptions.NCP.RouteUGame
import Pouw.Assumptions.NCP.RouteUWord
import Pouw.Assumptions.PearlC.Deadline
import Pouw.Assumptions.PearlC.DeviceChainCap
import Pouw.Assumptions.PearlC.DeviceChainCapKernel
import Pouw.Assumptions.PearlC.HiddenTile
import Pouw.Assumptions.PearlC.HonestCap
import Pouw.Assumptions.PearlC.RowSeedAssumptions
import Pouw.Assumptions.PearlC.TTOut
import Pouw.Assumptions.PearlC.TTOutChainOnly
import Pouw.Assumptions.PearlC.TTOutFp4
import Pouw.Assumptions.PearlC.TTOutFp4ChainOnly
import Pouw.Assumptions.PearlC.TTOutRev1
import Pouw.Assumptions.PearlC.TTOutRowSeed
import Pouw.Assumptions.PearlC.TTOutTileUOnly
import Pouw.Assumptions.PearlC.TTOutUOnly
import Pouw.Assumptions.Pinned

/-!
# PoUW's named assumptions, and the definitions and statements over them (trusted)

The modules `lean-audit.json` lists under `assumptions` state the named assumptions, each a `Prop` taken as a
hypothesis by the statements that use it, never an `axiom`; the game's are in `Game`. The layer's other modules hold
the definitions and statements that read them, which an assumption module may import. They import `Pouw.Protocol` and
each other, never a proof.
-/
