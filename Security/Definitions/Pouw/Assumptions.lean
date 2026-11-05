module

public import Definitions.Pouw.Assumptions.Deadline
public import Definitions.Pouw.Assumptions.Dimension.Assumptions
public import Definitions.Pouw.Assumptions.Dimension.Embed
public import Definitions.Pouw.Assumptions.Dimension.EmbedAssumptions
public import Definitions.Pouw.Assumptions.Dimension.Forest
public import Definitions.Pouw.Assumptions.Dimension.Fp8Mma
public import Definitions.Pouw.Assumptions.Dimension.H100
public import Definitions.Pouw.Assumptions.Dimension.H100Assumptions
public import Definitions.Pouw.Assumptions.Dimension.H100Cite
public import Definitions.Pouw.Assumptions.Dimension.H100Dense
public import Definitions.Pouw.Assumptions.Dimension.H100Floor
public import Definitions.Pouw.Assumptions.Dimension.H100Frag
public import Definitions.Pouw.Assumptions.Dimension.H100LoopFree
public import Definitions.Pouw.Assumptions.Dimension.H100Measured
public import Definitions.Pouw.Assumptions.Dimension.H100PeakTc
public import Definitions.Pouw.Assumptions.Dimension.H100W
public import Definitions.Pouw.Assumptions.Dimension.Mixing
public import Definitions.Pouw.Assumptions.Dimension.OpenPipe
public import Definitions.Pouw.Assumptions.Dimension.Pinned
public import Definitions.Pouw.Assumptions.Dimension.Pipe
public import Definitions.Pouw.Assumptions.Dimension.PipeAssumptions
public import Definitions.Pouw.Assumptions.Dimension.PipeModel
public import Definitions.Pouw.Assumptions.Dimension.RouteU
public import Definitions.Pouw.Assumptions.Dimension.RouteUPipe
public import Definitions.Pouw.Assumptions.Dimension.RouteUPipeAssumptions
public import Definitions.Pouw.Assumptions.Dither
public import Definitions.Pouw.Assumptions.DitherInstance
public import Definitions.Pouw.Assumptions.Fp8Atom.H1TAssumptions
public import Definitions.Pouw.Assumptions.Fp8Atom.H1TGamma
public import Definitions.Pouw.Assumptions.Fp8Atom.H1TRunningAssumptions
public import Definitions.Pouw.Assumptions.Fp8Atom.H1TTile
public import Definitions.Pouw.Assumptions.Fp8Atom.H1TTileAssumptions
public import Definitions.Pouw.Assumptions.Game
public import Definitions.Pouw.Assumptions.LeadCommit
public import Definitions.Pouw.Assumptions.NCP.Assumptions
public import Definitions.Pouw.Assumptions.NCP.Chain
public import Definitions.Pouw.Assumptions.NCP.GamePinned
public import Definitions.Pouw.Assumptions.NCP.RouteUAssumptions
public import Definitions.Pouw.Assumptions.NCP.RouteUGame
public import Definitions.Pouw.Assumptions.NCP.RouteUWord
public import Definitions.Pouw.Assumptions.OnsiteFreed
public import Definitions.Pouw.Assumptions.PearlC.Deadline
public import Definitions.Pouw.Assumptions.PearlC.DeviceChainCap
public import Definitions.Pouw.Assumptions.PearlC.DeviceChainCapKernel
public import Definitions.Pouw.Assumptions.PearlC.HiddenTile
public import Definitions.Pouw.Assumptions.PearlC.HonestCap
public import Definitions.Pouw.Assumptions.PearlC.RowSeedAssumptions
public import Definitions.Pouw.Assumptions.PearlC.TTOut
public import Definitions.Pouw.Assumptions.PearlC.TTOutChainOnly
public import Definitions.Pouw.Assumptions.PearlC.TTOutChainOnlyW1
public import Definitions.Pouw.Assumptions.PearlC.TTOutFp4
public import Definitions.Pouw.Assumptions.PearlC.TTOutFp4ChainOnly
public import Definitions.Pouw.Assumptions.PearlC.TTOutFp4W1
public import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
public import Definitions.Pouw.Assumptions.PearlC.TTOutRowSeed
public import Definitions.Pouw.Assumptions.PearlC.TTOutRowSeedK
public import Definitions.Pouw.Assumptions.PearlC.TTOutTileUOnly
public import Definitions.Pouw.Assumptions.PearlC.TTOutUOnly
public import Definitions.Pouw.Assumptions.PearlC.TTOutW1
public import Definitions.Pouw.Assumptions.Pinned
public import Definitions.Pouw.Assumptions.RelativeSecurity
public import Definitions.Pouw.Assumptions.ServedWork
public import Definitions.Pouw.Assumptions.ShaRealize
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
/-!
# PoUW's named assumptions, and the definitions and statements over them (trusted)

The modules `lean-audit.json` lists under `assumptions` state the named assumptions, each a `Prop` taken as a
hypothesis by the statements that use it, never an `axiom`; the game's are in `Game`. The layer's other modules hold
the definitions and statements that read them, which an assumption module may import. They import `Definitions.Pouw` and
each other, never a proof.
-/

@[expose] public section
