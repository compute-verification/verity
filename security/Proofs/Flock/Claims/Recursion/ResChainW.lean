module

public import Proofs.Flock.Claims.Recursion.ResAccPorts
public import Proofs.Flock.Claims.Recursion.ResLowerD
public import Proofs.Flock.Claims.Recursion.ResLowerDA
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

/-! P3 from the outer witnesses' own `rec-acc` chain, split into `StepsW` and `EndsW`. -/

@[expose] public section

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf)
open Claims.Recursion.VStarB

def rowW (x : Msg) (ws : List Claims.Flock.Wit) (p : ℕ) : Option Msg :=
  (readX x).bind fun r => (accCanon r p).bind fun e => ws[e.1]?.map fun wt => readSegs wt (accSegs e.2.1 e.2.2)

/-- V\* commits one chain, so the table index is unread. -/
def chainW (x : Msg) (ws : List Claims.Flock.Wit) (_t p : ℕ) : Array _root_.Flock.F128 :=
  match rowW x ws p with
  | some v => #[elemsOf v 2 0, elemsOf v 2 1]
  | none => #[0, 0]

def StepsW (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (DRL rd wE final w b elem).parse x = some st → shape st.form →
    ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
    RecDeploy.OpenLower.JointP FlockDeploy.sha512 (DRL rd wE final w b elem) (DeployedR.layR elem)
      (AccP rd wE final w b elem shape untr cmt) x c ws →
    ∀ os last, (_root_.RecDeploy.bdR rd wE final w).final
        (((DeployedR.layR elem).msgs x ws).getD
          ((_root_.RecDeploy.bdR rd wE final w).DE.V.k (Claims.Recursion.xI (DRL rd wE final w b elem) x) - 1)
          []) = some (os, last) →
      os.length + 1 = (_root_.RecDeploy.bdR rd wE final w).DE.V.k (Claims.Recursion.xI (DRL rd wE final w b elem) x) →
      Nonempty (Rows.DecAt rd (Claims.Recursion.xI (DRL rd wE final w b elem) x)
        ((_root_.RecDeploy.bdR rd wE final w).opened os last)
        ((List.range ((_root_.RecDeploy.bdR rd wE final w).DE.V.k
          (Claims.Recursion.xI (DRL rd wE final w b elem) x))).map c)) →
      Top.ChainHolds rd key (Claims.Recursion.xI (DRL rd wE final w b elem) x)
        ((_root_.RecDeploy.bdR rd wE final w).opened os last)
        ((List.range ((_root_.RecDeploy.bdR rd wE final w).DE.V.k
          (Claims.Recursion.xI (DRL rd wE final w b elem) x))).map c) (chainW x ws)

def StepsWX (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (DRL rd wE final w b elem).parse x = some st → shape st.form →
    ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
    RecDeploy.OpenLower.JointP FlockDeploy.sha512 (DRL rd wE final w b elem) (DeployedR.layR elem)
      (AccP rd wE final w b elem shape untr cmt) x c ws →
    ∀ (xs : List Msg), (∀ s' (hs' : s' < ws.length),
        AccV rd wE final w b elem shape untr cmt s' x c (xs.getD s' []) ws[s'] ((DeployedR.layR elem).msgs x ws)) →
      (Claims.findColl ((DeployedR.layR elem).allCalls x ws ++
        Claims.Recursion.setupCalls FlockDeploy.sha512 (DRL rd wE final w b elem) xs)).isSome = false →
      XsStaged b.st x xs ws.length →
    ∀ os last, (_root_.RecDeploy.bdR rd wE final w).final
        (((DeployedR.layR elem).msgs x ws).getD
          ((_root_.RecDeploy.bdR rd wE final w).DE.V.k (Claims.Recursion.xI (DRL rd wE final w b elem) x) - 1)
          []) = some (os, last) →
      os.length + 1 = (_root_.RecDeploy.bdR rd wE final w).DE.V.k (Claims.Recursion.xI (DRL rd wE final w b elem) x) →
      Nonempty (Rows.DecAt rd (Claims.Recursion.xI (DRL rd wE final w b elem) x)
        ((_root_.RecDeploy.bdR rd wE final w).opened os last)
        ((List.range ((_root_.RecDeploy.bdR rd wE final w).DE.V.k
          (Claims.Recursion.xI (DRL rd wE final w b elem) x))).map c)) →
      Top.ChainHolds rd key (Claims.Recursion.xI (DRL rd wE final w b elem) x)
        ((_root_.RecDeploy.bdR rd wE final w).opened os last)
        ((List.range ((_root_.RecDeploy.bdR rd wE final w).DE.V.k
          (Claims.Recursion.xI (DRL rd wE final w b elem) x))).map c) (chainW x ws)

theorem stepsWX_of_stepsW {rd : Msg → Option FinalP} {key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ} {wE : Msg → ℝ}
    {final : Msg → Option (List (List (Msg × Salt)) × Msg)} {w : Msg → ℝ} {b : Brings}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))}
    {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {cmt : Msg → ℕ → Msg → Msg}
    (h : StepsW rd key wE final w b elem shape untr cmt) : StepsWX rd key wE final w b elem shape untr cmt :=
  fun x st hx hf c ws hJ _ _ _ _ => h x st hx hf c ws hJ

open Flock.RecAlgebra (el) in
def EndsW (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (DRL rd wE final w b elem).parse x = some st → shape st.form →
    ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
    RecDeploy.OpenLower.JointP FlockDeploy.sha512 (DRL rd wE final w b elem) (DeployedR.layR elem)
      (AccP rd wE final w b elem shape untr cmt) x c ws →
    ∀ (xs : List Msg), (∀ s' (hs' : s' < ws.length),
        AccV rd wE final w b elem shape untr cmt s' x c (xs.getD s' []) ws[s'] ((DeployedR.layR elem).msgs x ws)) →
      (Claims.findColl ((DeployedR.layR elem).allCalls x ws ++
        Claims.Recursion.setupCalls FlockDeploy.sha512 (DRL rd wE final w b elem) xs)).isSome = false →
      XsStaged b.st x xs ws.length →
    ∀ s (wt : Claims.Flock.Wit), ws[s]? = some wt →
    ∀ (r : XR), readX x = some r → ∀ (ss : Array VStar.Stmt), RecVStar.statements r.shape = .ok ss →
    ∀ (j : ℕ) (nm : String) (pt : VStar.Part), ss[s]? = some (.inner j nm pt) →
    ∀ (xO : Msg) (p : _root_.Flock.Readable.Parsed), xs.getD s [] = xO →
      _root_.Flock.Readable.parseBytes readOfO xO = some p →
      Claims.Flock.check (_root_.Flock.Readable.parseOf _root_.Flock.Readable.relation readOfO) xO wt = true →
      showsPart untr b s x xO c → (DeployedR.layR elem).Agrees cmt x s wt xO ((DeployedR.layR elem).msgs x ws) →
      (∀ e ∈ (DeployedR.layR elem).calls x s wt, FlockDeploy.sha512 e.1 = e.2) →
      (Claims.findColl ((DeployedR.layR elem).calls x s wt ++
        vcallsAt FlockDeploy.sha512 (DRL rd wE final w b elem) xO)).isSome = false →
    ∀ q ∈ DecompBR.resPD (Claims.Recursion.xI (DRL rd wE final w b elem) x), DecompBR.ownRD x q = some s →
    ∀ (ht : 0 < p.sts.size) (m k cl jj : ℕ) (sh : _root_.Flock.Shape) (holo : Bool) (S : _root_.Flock.RecOpen.GfS)
      (ns : List ℕ) (nv : ℕ), RecResiduals.params p.sts[0].c.unit.name = some (m, k, cl, jj) →
      RecResiduals.shape? m k cl = some sh → RecResiduals.holoOf p.sts[0].c = .ok holo →
      RecResiduals.part { sh with holo } jj = .ok (S, ns, nv) →
    ∀ os last, (_root_.RecDeploy.bdR rd wE final w).final
        (((DeployedR.layR elem).msgs x ws).getD
          ((_root_.RecDeploy.bdR rd wE final w).DE.V.k (Claims.Recursion.xI (DRL rd wE final w b elem) x) - 1)
          []) = some (os, last) →
      os.length + 1 = (_root_.RecDeploy.bdR rd wE final w).DE.V.k (Claims.Recursion.xI (DRL rd wE final w b elem) x) →
    ∀ D : Rows.DecAt rd (Claims.Recursion.xI (DRL rd wE final w b elem) x)
        ((_root_.RecDeploy.bdR rd wE final w).opened os last)
        ((List.range ((_root_.RecDeploy.bdR rd wE final w).DE.V.k
          (Claims.Recursion.xI (DRL rd wE final w b elem) x))).map c),
    ∀ e j', elemPort (Claims.Recursion.xI (DRL rd wE final w b elem) x) q.key.1 e = some j' →
      accBase (Claims.Recursion.xI (DRL rd wE final w b elem) x) ≤ j' →
      j' < accBase (Claims.Recursion.xI (DRL rd wE final w b elem) x) + 4 ∧
      _root_.Flock.Readable.rowsRead ns nv (keyCopy p ht wt q.key) e =
        el (D.accE (fun _ _ => chainW x ws) (q.key.2.1 + q.key.2.2.1))
          (j' - accBase (Claims.Recursion.xI (DRL rd wE final w b elem) x))

theorem accPorts_of_WX (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (hS : StepsWX rd key wE final w b elem Claims.Recursion.Form.Dense untr cmt)
    (hE : EndsW rd wE final w b elem Claims.Recursion.Form.Dense untr cmt) :
    AccPorts rd key wE final w b elem Claims.Recursion.Form.Dense untr cmt := by
  intro x st hx hf c ws hJ xs hxs hcol hms s wt hw r hr ss hss j nm pt hsv xO p hxs0 hxO hck hsh hAg hT hc q hq ho ht
    m k cl jj sh holo S ns nv hpar hshp hholo hpart os last hfin hos hD
  obtain ⟨D⟩ := hD
  have hW := hS x st hx hf c ws hJ xs hxs hcol hms os last hfin hos ⟨D⟩
  have hEn := hE x st hx hf c ws hJ xs hxs hcol hms s wt hw r hr ss hss j nm pt hsv xO p hxs0 hxO hck hsh hAg hT hc q
    hq ho ht m k cl jj sh holo S ns nv hpar hshp hholo hpart os last hfin hos D
  obtain ⟨v, -, p0, r0, S0, ns0, nv0, hr0, -, hk, -⟩ := DecompBR.mem_resPD' hq
  have hn : q.key.2.1 + q.key.2.2.1 < 2 := by simp only [hk]; omega
  exact accPorts_of_chain (xo := x) (sto := st) D Top.cls₀ hx hf rfl hW q.key hn
    (fun e => _root_.Flock.Readable.rowsRead ns nv (keyCopy p ht wt q.key) e)
    (fun e j' he hj => (hEn e j' he hj).1) (fun e j' he hj => (hEn e j' he hj).2)

theorem accPorts_of_W (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (hS : StepsW rd key wE final w b elem Claims.Recursion.Form.Dense untr cmt)
    (hE : EndsW rd wE final w b elem Claims.Recursion.Form.Dense untr cmt) :
    AccPorts rd key wE final w b elem Claims.Recursion.Form.Dense untr cmt :=
  accPorts_of_WX rd key wE final w b elem untr cmt (stepsWX_of_stepsW hS) hE

end RecDeploy.ResLower
