module

public import Proofs.Flock.Claims.Recursion.DeployedR
public import Proofs.Flock.Claims.Recursion.HonestPartsR
public import Proofs.Flock.Claims.Recursion.HonestWitR
public import Proofs.Flock.Claims.Recursion.SlotsCommit
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
import Proofs.Flock.Claims.Recursion.CapPorts
import Proofs.Flock.Claims.Recursion.RowsCover
import Proofs.Flock.Claims.Recursion.SchedDense
import Proofs.Flock.Claims.Recursion.SlotsCheck

/-! Each V\* part's circuit accepts on the honest outer run at the layout of record (`honest_partAccepts_layR`), at
premises about the honest run and its witness alone. -/

@[expose] public section

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB (readForms)

theorem layoutBTops_calls_true {Col T : Type} (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin)))
    {DO : Claims.Flock.Deploy Coin Col T} {st : Stage} (h : HmSlotsCommit DO st) {x : Msg} {ms : List (Msg × Out)}
    {s : ℕ} {w : Claims.Flock.Wit} (hsg : StagedPart st s x (staged st x ms s))
    (hck : Claims.Flock.check DO.parse (staged st x ms s) w = true) :
    ∀ e ∈ (layoutBTops fin).calls x s w, FlockDeploy.sha512 e.1 = e.2 := by
  intro e he
  obtain ⟨p, hp, rfl⟩ := List.mem_map.1 he
  cases hr : readX x with
  | none => simp [layoutBTops, layoutBDeployed, hr] at hp
  | some r =>
    have hp' : p ∈ layoutBDeployed.shaAt x s ∨ p ∈ capsCalls r s := by
      simpa [layoutBTops, hr] using hp
    rcases hp' with hp | hp
    · cases hpt : r.w.parts[s]? with
      | none => simp [layoutBDeployed, hr, hpt] at hp
      | some pt =>
        cases hc : partCircuit pt with
        | none => simp [layoutBDeployed, hr, hpt, hc] at hp
        | some c =>
          simp only [layoutBDeployed, hr, hpt, hc, Option.bind_some, Option.map_some, Option.getD_some,
            List.mem_flatMap] at hp
          obtain ⟨hh, hmem, hp⟩ := hp
          have hc' : r.w.parts[s]?.bind partCircuit = some c := by rw [hpt]; exact hc
          exact hmCalls_true ((h x _ s w r c hsg hck hr hc').1 hh hmem) p hp
    · obtain ⟨l, N, c, q, n, hst, hc, hq, hn, hp⟩ := mem_capsCalls hp
      exact hmCalls_true ((h x _ s w r c hsg hck hr hc).2 l N hst q hq n hn) p hp

theorem heldB_layRH {W : Type} (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings) (I : Claims.Recursion.Inner Coin W)
    (hk : ∀ x, I.V.k x = (FlockDeploy.V false).k x) :
    Claims.Recursion.VStarB.HeldB (deployRec IV ivScope b) I (DeployedR.layR elem) Claims.Recursion.Form.Dense :=
  fun x st hst hf =>
    heldB_layoutBTops IV ivScope b I hk (finOpen (DeployedR.restR elem)) _ schedMatches_dense (capPorts _)
      (rowsCover _) (finPieces_finOpen (DeployedR.restR elem) _) x st hst hf

theorem runVstar_getD_staged {W : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (b : Brings) (hN : NumsFixed b.st) {x : Msg} {Ω : Type} (P : Claims.Prover Ω) (ω : Ω)
    (r : Fin ((V I.V b.st).k x) → Coin) (hacc : (Claims.run H P (V I.V b.st) x ω r).out = .acc) {s : ℕ}
    (hs : s < nS x) :
    (Claims.Recursion.runVstar H (deployRec I.V ivScope b) I P x ω r).getD s [] =
      staged b.st x ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)) s := by
  obtain ⟨_, i, spec, sts, -, hsu, -⟩ := stagedPart_run H I.V b.st x P ω r hacc hs
  show (vstar b.st x ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x))).getD s [] = _
  rw [vstar_getD _ _ _ hs]
  exact vstarAt_staged hN (by rw [hsu]; rfl)

theorem honest_partAccepts_layR (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
    (recRest : Msg → List (Msg × Out) → List UInt8)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR elem
    let bL : Brings := { bR with rd := L.readers }
    let I := Claims.Recursion.Inner.ofFlock DeployedR.bdR.deploy decI
    let D := recBound _root_.Flock.Readable.relation bR L DeployedR.bdR.IVStar
    NumsFixed bL.st →
    ∀ {x : Msg} {w : Claims.Flock.Wit} {st : Claims.Recursion.CompiledStatement}, Claims.Recursion.R D I x w →
      D.parse x = some st → Claims.Recursion.Form.Dense st.form →
    ∀ (ω : Tape bL) (r : Fin (D.V.k x) → Coin),
      (Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).out = .acc →
      HonestWitSatisfies FlockDeploy.sha512 I.V bL x w ω r →
    ∀ s < nS x,
      let wO := bL.ref.wit w ω.1
        (((Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.take (I.V.k (xIOf x))).map histOf) s
      let ms := (Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.map Prod.fst
      (∀ xr, readX x = some xr → ∀ p ∈ DeployedR.moreCalls xr s,
        FlockDeploy.sha512 (p.1 ++ readForms wO p.2.1) = readForms wO p.2.2) →
      (Claims.findColl (L.calls x s wO ++ Claims.Recursion.VStarB.vcallsAt FlockDeploy.sha512 D
        ((Claims.Recursion.runVstar FlockDeploy.sha512 D I (refP I.V bL w) x ω r).getD s []))).isSome = false →
      (∀ i < L.rounds x, ∀ m, L.held x s wO i = some m → ms.getD i [] = m) →
      (∀ i < L.rounds x, ∀ o m, (o, m) ∈ L.piecesOf x s wO i → ((ms.getD i []).drop o).take m.length = m) →
      L.RowsRead x s wO ms →
      Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
        Claims.Recursion.Form.Dense (showsPart recRead bR) (cmtTr recRead) s x (Claims.coinSeq r) ms := by
  intro bR L bL I D hN x w st hR hx hf ω r hacc hW s hs wO ms hTm hnc hh hp hr
  have hrb : ReadsBackB FlockDeploy.sha512 I.V bL Claims.Recursion.Form.Dense recRead :=
    readsBack_recTr _ _ bL _ aux nOf hd recRest rfl
  have hHeld : Claims.Recursion.VStarB.HeldB (deployRec I.V (FlockDeploy.deployPlain
      _root_.Flock.Readable.relation).scopeCheck bL) I L Claims.Recursion.Form.Dense :=
    heldB_layRH elem I.V _ bL I fun _ => rfl
  obtain ⟨-, -, -, -, hrel, -⟩ := vstar_complete_run_setup_at FlockDeploy.sha512 I
    (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL hN hR ω r hacc hW hs
  have hck := check_of_R hrel
  have hsg := stagedPart_run FlockDeploy.sha512 I.V bL.st x (refP I.V bL w) ω r hacc hs
  have heq := runVstar_getD_staged FlockDeploy.sha512 I
    (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL hN (refP I.V bL w) ω r hacc hs
  have hT : ∀ e ∈ L.calls x s wO, FlockDeploy.sha512 e.1 = e.2 := by
    intro e he
    have he' : e ∈ (DeployedR.layR elem).calls x s wO := he
    rw [DeployedR.layR_calls] at he'
    rcases List.mem_append.1 he' with he | he
    · exact layoutBTops_calls_true _ hmSlotsCommit_of_full hsg (by rw [← heq]; exact hck) e he
    · obtain ⟨p, hp, rfl⟩ := List.mem_map.1 he
      cases hxr : readX x with
      | none => simp [hxr] at hp
      | some xr =>
        rw [hxr, Option.map_some, Option.getD_some] at hp
        exact hTm xr hxr p hp
  exact partAccepts_run_deployRec FlockDeploy.sha512 I
    (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL recRead
    (deployZkRec _root_.Flock.Readable.relation) L Claims.Recursion.Form.Dense hrb hHeld hx hf hacc hs
    (fun _ _ => rfl) hck hT hnc hh hp hr

end RecDeploy
