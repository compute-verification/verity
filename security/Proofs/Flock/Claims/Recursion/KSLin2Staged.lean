module

public import Proofs.Flock.Claims.KS.PlainCapsDep
public import Proofs.Flock.Claims.Recursion.BoundReaders
public import Proofs.Flock.Claims.Recursion.BoundRouteKSLin2
public import Proofs.Flock.Claims.Recursion.KSLin2Plain
public import Proofs.Flock.Claims.Recursion.LastColsDeployed
public import Proofs.Flock.Claims.Recursion.VStarBDeploy
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
import Proofs.Flock.Claims.Bridge.Hm96Tree
import Proofs.Flock.Claims.KS.ZkOuter
import Proofs.Flock.Claims.Recursion.BoundRoute
import Proofs.Flock.Claims.Recursion.DeployFacts
import Proofs.Flock.Claims.Recursion.DeployMargin
import Proofs.Flock.Claims.Recursion.DeploySessions
import Proofs.Flock.Claims.Recursion.SetupCalls

/-! `KSLin2` with V\*'s staging read off the run (`StagedB`), as `VstarShowsB` is false at `deployRec`
(`vstarShowsB_deployed_rejects`). -/

@[expose] public section

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

theorem ksLin2_route_staged (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB)
    (hRows : L.ReadsRows FlockDeploy.sha512) (hHW : L.HeldWhole) (Cp : Claims.Recursion.Caps Coin) :
    ∀ (nextE : Msg → List Msg → List Coin → Claims.OC Out)
      (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
      (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ),
      let Bd := bdDeployed rel nextE colsE wE join rowIn final w
      let D := recBound rel b L Bd.IVStar
      let bL : Brings := { b with rd := L.readers }
      Claims.Recursion.KSFlockLin FlockDeploy.sha512 Bd.DE Cp → Bd.Binds → Bd.LastBindsEC FlockDeploy.sha512 →
      Bd.LastCoinFree FlockDeploy.sha512 →
      Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO) →
      (∀ shape ∈ [Claims.Recursion.Form.Dense],
        ∃ (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
          (cmt : Msg → ℕ → Msg → Msg),
          (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
          (∀ decI, Claims.Recursion.VStarB.StagedB FlockDeploy.sha512 D
            (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape shows cmt) ∧
          (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt chk s) ∧
          (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt s) ∧
          Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
      ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
      Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt rel b L) Bd (deployZkRec relO) Cp DenseAt := by
  intro nextE colsE wE join rowIn final w Bd D bL hKS hBinds hEC hCF hF hshapes hT decI hdI decO hdO x hx Ω _ _ P M R
  let I := Claims.Recursion.Inner.ofFlock Bd.deploy decI
  let ivScope := (FlockDeploy.deployPlain rel).scopeCheck
  have hacc : ∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarAcceptsV FlockDeploy.sha512 (deployRec I.V ivScope bL) I (deployZkRec relO) shape := by
    intro shape hs
    obtain ⟨chk, shows, cmt, hHe, hSt, hCk, hOp, hDc⟩ := hshapes shape hs
    exact vstarAcceptsV_boundW FlockDeploy.sha512 I ivScope b L (deployZkRec relO) shape chk shows cmt hHW (hHe decI)
      (hSt decI) hCk hOp hDc hRows
  have hTc := callsTrue FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hT
  have hA := Claims.Recursion.innerAcceptsCollAtV_of_dense (refuses I.V ivScope bL FlockDeploy.sha512) hTc
    (hacc _ (by simp)) hx
  have hLc := callsLogCollAt_deployRec FlockDeploy.sha512 I ivScope bL (deployZkRec relO)
  obtain ⟨-, shows, cmt, hHe, hSt, -, hOp, -⟩ := hshapes _ (List.mem_singleton_self _)
  have hFk := forksBindAt_C_pick_setup FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) L.pick hEC hCF hTc hLc x
    (fun decO Ω P ω r c ws hg => by
      obtain ⟨st, hst, -⟩ := refuses I.V ivScope bL FlockDeploy.sha512 x Ω P ω r hg.1
      exact readers_run FlockDeploy.sha512 I ivScope b L (deployZkRec relO) _ shows cmt hHW (hHe decI) hOp hst
        (hx st hst) (hSt decI x st hst (hx st hst) Ω P ω r hg.1) (Claims.Recursion.eoGives_outerHolds hg)
        (hTc decO x Ω P ω r c ws hg)) hA
  have hdecC : ∀ x c, FlockDeploy.decCoin x (Bd.ans c) = c := fun _ c =>
    (congrArg (FlockDeploy.decCoin _) (bdDeployed_ans rel nextE colsE wE join rowIn final w c)).trans
      (Function.leftInverse_invFun Claims.Bridge.coin_bytes_injective c)
  have hM := roundsMargin_deployRec FlockDeploy.sha512 I ivScope bL fun x => V_k_pos_flock false x
  have hLn := callsLogCollAt_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hCF hLc
  have hAn := innerAcceptsCollAtV_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hCF x hA
  have hbody := Claims.Recursion.Bound.ksLin2_of_flockLin_binds (Bd := Bd) (decI := decI)
    (H := FlockDeploy.sha512) (D := deployRecC Bd ivScope bL) (DO := deployZkRec relO) (decO := decO) (P := P)
    (x := x) FlockDeploy.decCoin Cp hF hKS hdI hdO hdecC hBinds (refuses I.V ivScope bL FlockDeploy.sha512)
    (innerRounds I ivScope bL) (sessionNums I.V ivScope bL relO) (tableErrNonneg I.V ivScope bL)
    (sessionRounds I ivScope bL relO) (sessionAccepts I.V ivScope bL) (sessionRuns_sha512 I ivScope bL relO)
    (innerViewRun_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO)) hTc hLn hFk hAn hM
    (callsNil I.V ivScope bL FlockDeploy.sha512) M R
  exact ksLin2_of_C_setup FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) decO hCF Cp x P M R hbody

/-- What V\*'s parts still owe at the dense form is `HeldB`, `PartChecksBV`, `PartOpensB` and `DecompB`, the two parts'
facts only at staged statements. -/
theorem ksLin2_deployed_parts (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB)
    (hRows : L.ReadsRows FlockDeploy.sha512) (hHW : L.HeldWhole) (Cp : Claims.Recursion.Caps Coin) (wE : Msg → ℝ)
    (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (hC : CanonicalProofFiles)
    (hK : PlainCallsBindOrSlack FlockDeploy.sha512) (untr : TrReader) :
    let Bd := bdDeployed rel (FlockDeploy.V false).next (FlockDeploy.deployPlainTrim rel).cols wE join rowInDeployed
      final w
    let D := recBound rel b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) untr →
    Claims.Recursion.KSFlockLin FlockDeploy.sha512 Bd.DE Cp →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt rel b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) Cp DenseAt := by
  intro Bd D bL hrb hKS hparts hT
  refine ksLin2_route_staged rel _root_.Flock.Readable.relation b L hRows hHW Cp (FlockDeploy.V false).next
    (FlockDeploy.deployPlainTrim rel).cols wE join rowInDeployed final w hKS
    (binds_deployed rel _ _ wE join final w) (lastBindsEC_deployed_sha512 rel wE join final w hC hK)
    (lastCoinFree_deployed rel _ wE join rowInDeployed final w FlockDeploy.sha512) FlockDeploy.KS.zk_KS_outer
    ?_ hT
  intro shape hs
  obtain ⟨chk, hHe, hCk, hOp, hDc⟩ := hparts shape hs
  exact ⟨chk, showsPart untr b, cmtTr untr, hHe,
    fun decI => stagedB_deployRec_part FlockDeploy.sha512 (Claims.Recursion.Inner.ofFlock Bd.deploy decI)
      (FlockDeploy.deployPlain rel).scopeCheck bL untr L shape (fun x st hst _ => hrb x st hst trivial) (hHe decI),
    hCk, hOp, hDc⟩

def StepsBase (nextE : Msg → List Msg → List Coin → Claims.OC Out) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) : Prop :=
  Claims.Recursion.KSFlockLin FlockDeploy.sha512
      (bdDeployed _root_.Flock.Readable.relation nextE (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols
        wE join rowInDeployed final w).DE FlockDeploy.KS.CpDep ∧
    (bdDeployed _root_.Flock.Readable.relation nextE (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols
      wE join rowInDeployed final w).LastBindsEC FlockDeploy.sha512 ∧
    (bdDeployed _root_.Flock.Readable.relation nextE (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols
      wE join rowInDeployed final w).LastCoinFree FlockDeploy.sha512

theorem stepsBase_V (wE : Msg → ℝ) (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (hC : CanonicalProofFiles) (hK : PlainCallsBindOrSlack FlockDeploy.sha512) :
    StepsBase (FlockDeploy.V false).next wE join final w :=
  ⟨plain_KSFlockLin_DE wE, lastBindsEC_deployed_sha512 _ wE join final w hC hK,
    lastCoinFree_deployed _ _ wE join rowInDeployed final w FlockDeploy.sha512⟩

theorem ksLin2_steps_parts (nextE : Msg → List Msg → List Coin → Claims.OC Out) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB)
    (hRows : L.ReadsRows FlockDeploy.sha512) (hHW : L.HeldWhole) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (hB : StepsBase nextE wE join final w)
    (untr : TrReader) :
    let Bd := bdDeployed _root_.Flock.Readable.relation nextE
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) untr →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro Bd D bL hrb hparts hT
  obtain ⟨hKS, hEC, hCF⟩ := hB
  refine ksLin2_route_staged _root_.Flock.Readable.relation _root_.Flock.Readable.relation b L hRows hHW
    FlockDeploy.KS.CpDep nextE (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join
    rowInDeployed final w hKS (binds_deployed _ _ _ wE join final w) hEC hCF FlockDeploy.KS.zk_KS_outer ?_ hT
  intro shape hs
  obtain ⟨chk, hHe, hCk, hOp, hDc⟩ := hparts shape hs
  exact ⟨chk, showsPart untr b, cmtTr untr, hHe,
    fun decI => stagedB_deployRec_part FlockDeploy.sha512 (Claims.Recursion.Inner.ofFlock Bd.deploy decI)
      (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL untr L shape
      (fun x st hst _ => hrb x st hst trivial) (hHe decI),
    hCk, hOp, hDc⟩

theorem ksLin2_deployed_plain_parts (b : Brings) (L : Claims.Recursion.VStarB.LayoutB)
    (hRows : L.ReadsRows FlockDeploy.sha512) (hHW : L.HeldWhole)
    (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hC : CanonicalProofFiles) (hK : PlainCallsBindOrSlack FlockDeploy.sha512) (untr : TrReader) :
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) untr →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt :=
  ksLin2_steps_parts _ b L hRows hHW wE join final w (stepsBase_V wE join final w hC hK) untr

end RecDeploy
