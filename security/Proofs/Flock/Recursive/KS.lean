module

public import Proofs.Flock.Claims.KS.PlainSimR
public import Proofs.Flock.Claims.Recursion.InnerAtCoins
public import Proofs.Flock.Claims.Recursion.InnerExtras
public import Proofs.Flock.Claims.Recursion.MoreSlotsAll
public import Proofs.Flock.Claims.Recursion.OpenDirsDeployed
public import Proofs.Flock.Claims.Recursion.OutTopRows
public import Proofs.Flock.Claims.Recursion.ResLowerDD
public import Proofs.Flock.Claims.Recursion.SideCapPos
public import Proofs.Flock.Claims.Recursion.SlotsPorted
public import Proofs.Flock.Claims.Recursion.VerdictRowEFree
public import Proofs.Flock.Recursive.KSStatement
public import Proofs.Flock.Recursive.KSV
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
import Proofs.Flock.Claims.Recursion.KSDec
import Proofs.Flock.Claims.Recursion.RestVerdictBasesHold

/-! `RecursiveKSAt` from its open premises, at the deployed join and the layout of record (`recursiveKSAt_of`). -/

@[expose] public section


namespace Flock.SecurityProofs

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)

theorem recursiveKSAt_of (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (hD : ∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompBP FlockDeploy.sha512
        (RecDeploy.recBound _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
          (RecDeploy.DeployedR.layR rest)
          (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
            (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
            RecDeploy.rowInDeployed final w).IVStar)
        (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
          (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
          RecDeploy.rowInDeployed final w).IVStar
        (RecDeploy.DeployedR.layR rest) shape
        (Claims.Recursion.VStarB.PartAcceptsP FlockDeploy.sha512
          (RecDeploy.recBound _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
            (RecDeploy.DeployedR.layR rest)
            (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
              (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
              RecDeploy.rowInDeployed final w).IVStar)
          (RecDeploy.deployZkRec _root_.Flock.Readable.relation) (RecDeploy.DeployedR.layR rest) shape
          (RecDeploy.showsPart RecDeploy.recRead { b with tr := RecDeploy.recTr aux nOf hd recRest })
          (RecDeploy.cmtTr RecDeploy.recRead))) :
    Flock.Guarantees.RecursiveKSAt rd final rest wE w b aux nOf hd recRest :=
  RecDeploy.ksLin2_deployedR_layRP rd b aux nOf hd recRest rest wE FlockDeploy.joinR final w hIn
    (RecDeploy.MoreSlots.moreSlotsTrue_deployed _) hD

theorem recursiveKSAt_of_decomp (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (hD : RecDeploy.DecompBR.RecKSDecompL rd b aux nOf hd recRest wE final w (RecDeploy.DeployedR.layR rest)) :
    Flock.Guarantees.RecursiveKSAt rd final rest wE w b aux nOf hd recRest :=
  recursiveKSAt_of rd final rest wE w b aux nOf hd recRest hIn fun shape hs => by
    obtain rfl := List.mem_singleton.1 hs
    exact Claims.Recursion.VStarB.decompBP_of_decompB hD

section Leaves

open RecDeploy RecDeploy.DecompBR RecDeploy.RestLow

theorem copyOpensR_bdDeployed {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {shape : Claims.Recursion.Form → Prop}
    {untr : TrReader} {b : Brings} {cmt : Msg → ℕ → Msg → Msg} {L : Claims.Recursion.VStarB.LayoutB}
    {side : Msg → List Msg → OpenPiece → OpenSide} {pieces : Msg → List OpenPiece}
    {nextE : Msg → List Msg → List Coin → Claims.OC Out}
    {colsE colsE' : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)} {wE wE' : Msg → ℝ}
    {join : List Msg → Msg} {rowIn : Msg → Msg} {final : Msg → Option (List (List (Msg × Salt)) × Msg)}
    {w w' : Msg → ℝ}
    (h : OpenLower.CopyOpensR H D shape untr b cmt L
      (bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowIn final w) side pieces) :
    OpenLower.CopyOpensR H D shape untr b cmt L
      (bdDeployed _root_.Flock.Readable.relation nextE colsE' wE' join rowIn final w') side pieces :=
  fun x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO wt p hst q hq ho ht os last hfin hl =>
    h x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO wt p hst q hq ho ht os last hfin hl

/-- Open, each a named leaf: each algebra part's share of the pieces (`hres`), the last message's reading (`hF`, `hRd`)
and the setup's listing (`hset`). -/
theorem recursiveKSAt_of_leaves (wE w : Msg → ℝ)
    (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (hF : RestRead.FinalReads DeployedR.finalR (DeployedR.restR OpenRows.elemR))
    (hRd : RestRead.ReadsAt DeployedR.rdR DeployedR.finalR (DeployedR.restR OpenRows.elemR))
    (hres : ResLower.ResLoweredPD DeployedR.rdR wE DeployedR.finalR w (bB b aux nOf hd recRest) OpenRows.elemR
      Claims.Recursion.Form.Dense recRead (cmtTr recRead) resPD ownRD (Rows.elemReadC (Top.accRC DeployedR.rdR key))
      Rows.resVRead)
    (hset : ∀ xO, ∀ u ∈ ResLower.resSetup xO, u ∈ b.setup xO) :
    Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR wE w b aux nOf hd recRest := by
  have hO : OpenLower.CopyOpensR FlockDeploy.sha512
      (recL DeployedR.rdR b aux nOf hd recRest wE DeployedR.finalR w (DeployedR.layR OpenRows.elemR))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd recRest) (cmtTr recRead) (DeployedR.layR OpenRows.elemR)
      (bdR DeployedR.rdR wE DeployedR.finalR w) (OpenLower.sideAtR DeployedR.rdR) (Rows.rowDPieces key) :=
    copyOpensR_bdDeployed (OpenRows.copyOpensR_layR FlockDeploy.sha512
      (recL DeployedR.rdR b aux nOf hd recRest wE DeployedR.finalR w (DeployedR.layR OpenRows.elemR))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd recRest) (cmtTr recRead) (Rows.rowDPieces key)
      (fun _ => rfl) (fun _ h => h) (fun _ => rfl) (OpenLower.piecesInRangeR_rowDPieces key))
  have hI : ∀ s x c (st0 : Claims.Recursion.CompiledStatement), RecDeploy.parse x = some st0 → Claims.Recursion.Form.Dense st0.form →
      OpenDirs.InnerMR recRead (bB b aux nOf hd recRest) s x c :=
    fun s x c _ hst hdn xO hsh i hins f unit inn hf hload => by
      obtain ⟨v, hv, hm, -⟩ := OpenLower.staged_innerAt (b := bB b aux nOf hd recRest)
        (Bk := (FlockDeploy.V false).k) (s := s) (c := c) hst hdn le_rfl xO hsh i hins f unit inn hf hload
      exact ⟨v, hv, hm⟩
  have hpos : ∀ s x c (st0 : Claims.Recursion.CompiledStatement), RecDeploy.parse x = some st0 → Claims.Recursion.Form.Dense st0.form →
      OpenDirs.DirsCoinR recRead (bB b aux nOf hd recRest) (Rows.rowDPieces key) Rows.posAt
        (bdR DeployedR.rdR wE DeployedR.finalR w).DE.V.k s x c :=
    fun s x c _ hst hdn => OpenLower.posTieR_dense (b := bB b aux nOf hd recRest)
      (Bk := (bdR DeployedR.rdR wE DeployedR.finalR w).DE.V.k) (s := s) (c := c) hst hdn
      (OpenLower.piecesInRangeR_rowDPieces key) le_rfl
  have hDirs : OpenLower.CopyDirsTieR FlockDeploy.sha512
      (recL DeployedR.rdR b aux nOf hd recRest wE DeployedR.finalR w (DeployedR.layR OpenRows.elemR))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd recRest) (cmtTr recRead) (DeployedR.layR OpenRows.elemR)
      (bdR DeployedR.rdR wE DeployedR.finalR w) (Rows.rowDPieces key) Rows.posAt :=
    OpenDirs.copyDirsTieR_layR (Rows.rowDPieces key) Rows.posAt
      (OpenDirs.dirsSideR_of (Rows.rowDPieces key) Rows.posAt (fun _ => rfl) (fun _ h => h)
        (OpenLower.piecesInRangeR_rowDPieces key) hset hI hpos)
  have hOut : CapLower.OutBindRP FlockDeploy.sha512
      (recL DeployedR.rdR b aux nOf hd recRest wE DeployedR.finalR w (DeployedR.layR OpenRows.elemR))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd recRest) (cmtTr recRead) (DeployedR.layR OpenRows.elemR)
      (bdR DeployedR.rdR wE DeployedR.finalR w) (Rows.rowDPieces key) Rows.posAt :=
    CapLower.outBindRP_deployed (b := bB b aux nOf hd recRest) (elem := OpenRows.elemR)
      (IV := (bdR DeployedR.rdR wE DeployedR.finalR w).IVStar) (shape := Claims.Recursion.Form.Dense)
      (Bd := bdR DeployedR.rdR wE DeployedR.finalR w) (Rows.rowDPieces key) (fun _ => rfl) (fun _ h => h)
      (OpenLower.piecesInRangeR_rowDPieces key) (fun _ => le_rfl)
  refine recursiveKSAt_ofVD DeployedR.rdR DeployedR.finalR OpenRows.elemR wE w b aux nOf hd recRest
    (fun x _ => FlockDeploy.KS.plainRdInR DeployedR.rdR x) fun shape hs => ?_
  obtain rfl := List.mem_singleton.1 hs
  exact recKSDecompDV_of_leavesJD DeployedR.rdR b aux nOf hd recRest OpenRows.elemR wE DeployedR.finalR w resPD key
    (Rows.elemReadC (Top.accRC DeployedR.rdR key)) Rows.resVRead ownRD
    (targetRDep_readers DeployedR.rdR (resPiecesAt_D DeployedR.rdR key _ _))
    (Top.rowEJoint_holds DeployedR.rdR key _ _ _ _ _ _)
    (basesHold_spec DeployedR.rdR) hF hRd (saltRowsJ_layR DeployedR.rdR wE w OpenRows.elemR) (piecesOwned_D key)
    (resLowJPD_of_resLoweredPD DeployedR.rdR b aux nOf hd recRest wE DeployedR.finalR w OpenRows.elemR _ _ _ _ hres)
    hO hDirs
    hOut (CapLower.sideCapR_leaf wE w b aux nOf hd recRest key hRd)

end Leaves

end Flock.SecurityProofs
