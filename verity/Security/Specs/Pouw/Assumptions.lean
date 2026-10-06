import Specs.Pouw.Assumptions.Deadline
import Specs.Pouw.Assumptions.Dimension.A2Words
import Specs.Pouw.Assumptions.Dimension.Assumptions
import Specs.Pouw.Assumptions.Dimension.Embed
import Specs.Pouw.Assumptions.Dimension.EmbedAssumptions
import Specs.Pouw.Assumptions.Dimension.Forest
import Specs.Pouw.Assumptions.Dimension.Fp8Mma
import Specs.Pouw.Assumptions.Dimension.H100
import Specs.Pouw.Assumptions.Dimension.H100Assumptions
import Specs.Pouw.Assumptions.Dimension.H100Cite
import Specs.Pouw.Assumptions.Dimension.H100Dense
import Specs.Pouw.Assumptions.Dimension.H100Floor
import Specs.Pouw.Assumptions.Dimension.H100Frag
import Specs.Pouw.Assumptions.Dimension.H100LoopFree
import Specs.Pouw.Assumptions.Dimension.H100Measured
import Specs.Pouw.Assumptions.Dimension.H100PeakTc
import Specs.Pouw.Assumptions.Dimension.H100W
import Specs.Pouw.Assumptions.Dimension.Mixing
import Specs.Pouw.Assumptions.Dimension.OpenPipe
import Specs.Pouw.Assumptions.Dimension.Pinned
import Specs.Pouw.Assumptions.Dimension.Pipe
import Specs.Pouw.Assumptions.Dimension.PipeAssumptions
import Specs.Pouw.Assumptions.Dimension.PipeModel
import Specs.Pouw.Assumptions.Dimension.RouteU
import Specs.Pouw.Assumptions.Dimension.RouteUPipe
import Specs.Pouw.Assumptions.Dimension.RouteUPipeAssumptions
import Specs.Pouw.Assumptions.Fp8Atom.H1TAssumptions
import Specs.Pouw.Assumptions.Fp8Atom.H1TGamma
import Specs.Pouw.Assumptions.Fp8Atom.H1TRunningAssumptions
import Specs.Pouw.Assumptions.Fp8Atom.H1TTile
import Specs.Pouw.Assumptions.Fp8Atom.H1TTileAssumptions
import Specs.Pouw.Assumptions.Game
import Specs.Pouw.Assumptions.NCP.Assumptions
import Specs.Pouw.Assumptions.NCP.Chain
import Specs.Pouw.Assumptions.NCP.GamePinned
import Specs.Pouw.Assumptions.NCP.RouteUAssumptions
import Specs.Pouw.Assumptions.NCP.RouteUGame
import Specs.Pouw.Assumptions.NCP.RouteUWord
import Specs.Pouw.Assumptions.PearlC.Deadline
import Specs.Pouw.Assumptions.PearlC.DeviceChainCap
import Specs.Pouw.Assumptions.PearlC.DeviceChainCapKernel
import Specs.Pouw.Assumptions.PearlC.HiddenTile
import Specs.Pouw.Assumptions.PearlC.HonestCap
import Specs.Pouw.Assumptions.PearlC.RowSeedAssumptions
import Specs.Pouw.Assumptions.PearlC.TTOut
import Specs.Pouw.Assumptions.PearlC.TTOutChainOnly
import Specs.Pouw.Assumptions.PearlC.TTOutFp4
import Specs.Pouw.Assumptions.PearlC.TTOutFp4ChainOnly
import Specs.Pouw.Assumptions.PearlC.TTOutRev1
import Specs.Pouw.Assumptions.PearlC.TTOutRowSeed
import Specs.Pouw.Assumptions.PearlC.TTOutRowSeedK
import Specs.Pouw.Assumptions.PearlC.TTOutTileUOnly
import Specs.Pouw.Assumptions.PearlC.TTOutUOnly
import Specs.Pouw.Assumptions.Pinned

/-!
# PoUW's named assumptions, and the definitions and statements over them (trusted)

The modules `lean-audit.json` lists under `assumptions` state the named assumptions, each a `Prop` taken as a
hypothesis by the statements that use it, never an `axiom`; the game's are in `Game`. The layer's other modules hold
the definitions and statements that read them, which an assumption module may import. They import `Pouw.Protocol` and
each other, never a proof.
-/
