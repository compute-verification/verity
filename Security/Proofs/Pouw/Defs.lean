import Proofs.Pouw.Defs.Assumptions.Fp8Atom.H1TRunningAssumptions
import Proofs.Pouw.Defs.Fp8Atom.H1TCrossLate
import Proofs.Pouw.Defs.Fp8Atom.H1TLemmas
import Proofs.Pouw.Defs.Fp8Atom.H1TPinned
import Proofs.Pouw.Defs.Fp8Atom.H1TVec
import Proofs.Pouw.Defs.Fp8Atom.H1TVectors
import Proofs.Pouw.Defs.Fp8Atom.Pinned
import Proofs.Pouw.Defs.Fp8Atom.Vectors
import Proofs.Pouw.Defs.Fp8Atom.Vectors.Ada
import Proofs.Pouw.Defs.Fp8Atom.Vectors.Add
import Proofs.Pouw.Defs.Fp8Atom.Vectors.C00
import Proofs.Pouw.Defs.Fp8Atom.Vectors.C01
import Proofs.Pouw.Defs.PearlC.EncVectors
import Proofs.Pouw.Defs.PearlC.Fp4Vectors
import Proofs.Pouw.Defs.PearlC.PeelExactVectors
import Proofs.Pouw.Defs.PearlC.PeelFp4Vectors
import Proofs.Pouw.Defs.PearlC.Sm120Vectors
import Proofs.Pouw.Defs.Assumptions.Dimension.A2Words
import Proofs.Pouw.Defs.Assumptions.Dimension.Fp8Mma
import Proofs.Pouw.Defs.Assumptions.Dimension.H100Cite
import Proofs.Pouw.Defs.Assumptions.Dimension.H100Dense
import Proofs.Pouw.Defs.Assumptions.Dimension.H100Floor
import Proofs.Pouw.Defs.Assumptions.Dimension.H100Frag
import Proofs.Pouw.Defs.Assumptions.Dimension.H100LoopFree
import Proofs.Pouw.Defs.Assumptions.Dimension.H100Measured
import Proofs.Pouw.Defs.Assumptions.Dimension.H100PeakTc
import Proofs.Pouw.Defs.Assumptions.Dimension.Pinned
import Proofs.Pouw.Defs.Assumptions.Dimension.RouteU
import Proofs.Pouw.Defs.Assumptions.Dimension.RouteUPipe
import Proofs.Pouw.Defs.Assumptions.Dimension.RouteUPipeAssumptions
import Proofs.Pouw.Defs.Assumptions.Fp8Atom.H1TTile
import Proofs.Pouw.Defs.Assumptions.Fp8Atom.H1TTileAssumptions
import Proofs.Pouw.Defs.Assumptions.NCP.Chain
import Proofs.Pouw.Defs.Assumptions.NCP.RouteUGame
import Proofs.Pouw.Defs.Assumptions.PearlC.DeviceChainCapKernel
import Proofs.Pouw.Defs.Assumptions.PearlC.HonestCap
import Proofs.Pouw.Defs.Assumptions.PearlC.RowSeedAssumptions
import Proofs.Pouw.Defs.Assumptions.PearlC.TTOutChainOnly
import Proofs.Pouw.Defs.Assumptions.PearlC.TTOutFp4
import Proofs.Pouw.Defs.Assumptions.PearlC.TTOutFp4ChainOnly
import Proofs.Pouw.Defs.Assumptions.PearlC.TTOutTileUOnly
import Proofs.Pouw.Defs.Barrier.Pinned
import Proofs.Pouw.Defs.Dimension.Causal
import Proofs.Pouw.Defs.FpG1.Gemm
import Proofs.Pouw.Defs.H1Zy.Stream
import Proofs.Pouw.Defs.NCP.ExtraIdentity
import Proofs.Pouw.Defs.NCP.Pinned
import Proofs.Pouw.Defs.PearlC.Completeness
import Proofs.Pouw.Defs.PearlC.DeviceFp4Hot
import Proofs.Pouw.Defs.PearlC.NoiseSha512
import Proofs.Pouw.Defs.TileBound.AlignedExactDefs
import Proofs.Pouw.Defs.TileBound.Defs
import Proofs.Pouw.Defs.TileBound.Fp8Defs

/-!
# Pouw's vocabulary that no guarantee reads

Definitions the proofs use and no listed guarantee's statement reads, kept out of the trusted text (`Definitions/`).
-/
