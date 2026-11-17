module

public import Proofs.Flock.Claims.KS.PlainAccPrR
public import Proofs.Flock.Claims.KS.PlainRCapPrefix
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
import Proofs.Flock.Claims.KS.PlainRConflict
public import Proofs.Flock.Claims.Recursion.TopsOpens
public import Proofs.Flock.Claims.Recursion.LastColsE
public import Proofs.Flock.Claims.Recursion.ConflictTiedE
public import Proofs.Flock.Claims.Recursion.LastBindsEC
public import Proofs.Flock.Claims.Recursion.CallsAgree
public import Proofs.Flock.Claims.Recursion.VStarCross
public import Proofs.Flock.Claims.Recursion.SetupCalls
public import Proofs.Flock.Claims.Recursion.PartAccepts
public import Proofs.Flock.Claims.Recursion.CallsAgreeE

/-! `KSFlockLin` at the deployed reader's bound record (U3), from the reader's bound wherever some run accepts
(`PlainRdInR`) alone (`ksFlockLin_R`). -/

@[expose] public section

namespace RecDeploy

open Claims (Msg)
open FlockDeploy (Coin FinalP nextR)

theorem ksFlockLin_R_of_table (rd : Msg → Option FinalP) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (htab : Claims.Recursion.TableBoundAbortLin FlockDeploy.sha512
      (FlockDeploy.deployPlainR _root_.Flock.Readable.relation rd) (FlockDeploy.KS.wordLen false)
      (FlockDeploy.KS.wide false) FlockDeploy.KS.CpDep) :
    Claims.Recursion.KSFlockLin FlockDeploy.sha512
      (bdDeployed _root_.Flock.Readable.relation (nextR rd)
        (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE join rowInDeployed final w).DE
      FlockDeploy.KS.CpDep :=
  fun decode hdec => FlockDeploy.KS.plainR_KSFlockLin _root_.Flock.Readable.relation rd
    (FlockDeploy.KS.plainR_conflictTied _ rd) htab decode hdec

theorem ksFlockLin_R (rd : Msg → Option FinalP) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x) :
    Claims.Recursion.KSFlockLin FlockDeploy.sha512
      (bdDeployed _root_.Flock.Readable.relation (nextR rd)
        (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE join rowInDeployed final w).DE
      FlockDeploy.KS.CpDep :=
  ksFlockLin_R_of_table rd wE join final w (FlockDeploy.KS.plainR_tableBoundAbortLin_R rd hIn)

end RecDeploy
