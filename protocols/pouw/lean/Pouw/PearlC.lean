import Pouw.PearlC.Accum
import Pouw.PearlC.AccumProofs
import Pouw.PearlC.AtomErr
import Pouw.PearlC.CapLoopGamma
import Pouw.PearlC.ChainCapKernelGamma
import Pouw.PearlC.ChainCapLoopGamma
import Pouw.PearlC.ChainOnlyGamma
import Pouw.PearlC.Defs
import Pouw.PearlC.Device
import Pouw.PearlC.DeviceCapGamma
import Pouw.PearlC.DeviceChainCap
import Pouw.PearlC.DeviceChainCapGamma
import Pouw.PearlC.DeviceChainCapKernel
import Pouw.PearlC.DeviceChainOnly
import Pouw.PearlC.DeviceGamma
import Pouw.PearlC.DeviceKernelWref
import Pouw.PearlC.DevicePrices
import Pouw.PearlC.DevicePricesLoop
import Pouw.PearlC.DeviceRev1
import Pouw.PearlC.DeviceRev1Gamma
import Pouw.PearlC.DeviceSm120Gamma
import Pouw.PearlC.DeviceSm120KernelGamma
import Pouw.PearlC.DeviceSm120LoopGamma
import Pouw.PearlC.DeviceSm120v2LoopEquiv
import Pouw.PearlC.EncChecks
import Pouw.PearlC.Forming
import Pouw.PearlC.FormingP
import Pouw.PearlC.FormingPDefs
import Pouw.PearlC.Fp4
import Pouw.PearlC.Fp4Chain
import Pouw.PearlC.Fp4Checks
import Pouw.PearlC.Fp4Freeze
import Pouw.PearlC.Fp4Skip
import Pouw.PearlC.Fp4Vectors
import Pouw.PearlC.Freeze
import Pouw.PearlC.FreezeP
import Pouw.PearlC.FreezePPure
import Pouw.PearlC.Game
import Pouw.PearlC.Gamma
import Pouw.PearlC.KernelWrefGamma
import Pouw.PearlC.Peel
import Pouw.PearlC.PeelExact
import Pouw.PearlC.PeelExactChecks
import Pouw.PearlC.PeelExactP
import Pouw.PearlC.PeelExactProofs
import Pouw.PearlC.PeelExactProofsP
import Pouw.PearlC.Rev1Witness
import Pouw.PearlC.RowSeedAssumptions
import Pouw.PearlC.RowSeedCode
import Pouw.PearlC.RowSeedFragment
import Pouw.PearlC.RowSeedGamma
import Pouw.PearlC.RowSeedSkip
import Pouw.PearlC.SaltDead
import Pouw.PearlC.Skip
import Pouw.PearlC.SkipP
import Pouw.PearlC.Sm120
import Pouw.PearlC.Sm120Checks
import Pouw.PearlC.TTOut
import Pouw.PearlC.TTOutChainOnly
import Pouw.PearlC.TTOutRev1
import Pouw.PearlC.TTOutRowSeed
import Pouw.PearlC.TTOutTileUOnly
import Pouw.PearlC.TTOutUOnly
import Pouw.PearlC.Tile
import Pouw.PearlC.TileGamma
import Pouw.PearlC.TileUOnlyGamma
import Pouw.PearlC.UOnlyGamma
import Pouw.PearlC.UOnlyLoopGamma

/-!
# `Pouw.PearlC`: Pearl-C's Π and the statements about it (the aggregator; staging, nothing pinned)

Π's shared definitions (`Defs`): the parameters, the E4M3 encoder, the BF16 conversion and the FP32 encoder, the row
statistic, the noise and the forming chains, the promoted product chain, and U's two forms. The conformance vectors of
the encoders, the conversion and the atom-mix chain (`EncVectors`, generated from `verity.ml.tc` by
`internal/pouw-fp8/pearl-c-scripts/pearlc_enc_vectors.py`) and their kernel checks (`EncChecks`). P2, the clean-up
chain (`Peel`): U's cancellation identities, where the rounding is, and the clean-up terms' liveness at degenerate
inputs. P1, freezing under promotion (`Freeze`), and P3, the forming credit (`Forming`), from their lanes. The
cheap-binding lane's game (`Game`), named protocol (`Instance`), named assumption TT_OUT (`TTOut`) and γ from it
(`Gamma`).
-/
