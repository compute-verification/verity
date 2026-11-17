module

public import Proofs.Flock.Claims.Recursion.BitsLin2
public import Proofs.Flock.Claims.Recursion.LayoutBTops
public import Proofs.Flock.Claims.Recursion.TopsStaged
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

/-! The last round's pieces at the parts that commit the earlier rounds (`finOpen`) and `FinPieces` there; that they
read the last message's bytes stays with `PartChecksBV` and `DecompB`. -/

@[expose] public section

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)

/-- **A commit string's opening** at port `q` (named `nm`) at instance `n`: the row the port reads there
(`rowHidden`), then its hm96 slot's salt (`hmSalt`). -/
def openSegs (c : Circuit) (nm : String) (q n : ℕ) : List (Msg × List HmNets.Lin) :=
  [(([] : Msg), ((rowHidden c nm n).getD []).map HmNets.Lin.var), (([] : Msg), hmSalt c q n)]

/-- **A cap's openings** at part `s`: at port `top`, the openings of its instances `ns`, in order. At a part with no
circuit or no port `top` they are read at the default circuit and port 0 instead, as `topPiece` reads its strings, so
every cap opens one row and salt an instance wherever it sits. -/
def capOpen (r : XR) (s : ℕ) (ns : List ℕ) : List (Msg × List HmNets.Lin) :=
  let c := (r.w.parts[s]?.bind partCircuit).getD default
  ns.flatMap (openSegs c "top" ((capPort c).getD 0))

/-- **The opening of the row stream round `j` absorbs**, at the part `rowPiece` holds it at: the first part with a held
row of round `j` (`heldRows`). None where no part has one. -/
def rowOpen (r : XR) (j : ℕ) : List (ℕ × List (Msg × List HmNets.Lin)) :=
  ((List.range r.w.parts.length).findSome? fun s => ((heldRows r s).find? (·.1 == j)).bind fun h =>
    (r.w.parts[s]?.bind partCircuit).map fun c => (s, openSegs c h.2.1 h.2.2.1 h.2.2.2)).toList

/-- The openings of the `e`-th top of stream round `j`, at its cap's part (`capPieceAt`'s): rep `j / R`'s `n`
instances, `rep · n + t`. -/
def capOpenAt (r : XR) (rs : Array RecVStar.SchedRound) (j e : ℕ) : ℕ × List (Msg × List HmNets.Lin) :=
  (capsFrom r rs + topsBefore rs (j % rs.size) + e,
    capOpen r (capsFrom r rs + topsBefore rs (j % rs.size) + e)
      ((List.range (((rs.getD (j % rs.size) default).2.1).getD e 0)).map fun t =>
        j / rs.size * ((rs.getD (j % rs.size) default).2.1).getD e 0 + t))

/-- **Stream round `j`'s openings**, in commit order: each top's at its cap, then the row's. -/
def roundOpen (r : XR) (rs : Array RecVStar.SchedRound) (j : ℕ) : List (ℕ × List (Msg × List HmNets.Lin)) :=
  (List.range ((rs.getD (j % rs.size) default).2.1).size).map (capOpenAt r rs j) ++ rowOpen r j

/-- **Every earlier round's openings**, round by round: round 0's at RecCap's part, over its `N` instances, then each
stream round's. -/
def openPieces (r : XR) : List (ℕ × List (Msg × List HmNets.Lin)) :=
  (capAt r).toList.map (fun p => (p.1, capOpen r p.1 (List.range p.2))) ++
    ((repSched r.shape).map fun rs => (List.range (2 * rs.size)).flatMap (roundOpen r rs)).getD []

/-- **The last round's pieces**: every earlier round's openings (`openPieces`), then the element-level last message's
pieces at the parts `rest` names. None where `readX` reads no statement. -/
def finOpen (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (x : Msg) :
    List (ℕ × List (Msg × List HmNets.Lin)) :=
  match readX x with
  | some r => openPieces r ++ (rest r).map fun p => (p.1.val, p.2)
  | none => []

theorem rowOpen_lt {r : XR} {j : ℕ} {p : ℕ × List (Msg × List HmNets.Lin)} (hp : p ∈ rowOpen r j) :
    p.1 < r.w.parts.length := by
  unfold rowOpen at hp
  simp only [Option.mem_toList] at hp
  obtain ⟨a, ha, hfa⟩ := List.exists_of_findSome?_eq_some hp
  rw [Option.bind_eq_some_iff] at hfa
  obtain ⟨h, -, hfa⟩ := hfa
  rw [Option.map_eq_some_iff] at hfa
  obtain ⟨c, -, rfl⟩ := hfa
  exact List.mem_range.1 ha

theorem roundOpen_lt {r : XR} {rs : Array RecVStar.SchedRound} {j : ℕ} (hpos : 0 < rs.size)
    (hlen : (RecVStar.topSizes rs).length ≤ r.w.parts.length - 1) {p : ℕ × List (Msg × List HmNets.Lin)}
    (hp : p ∈ roundOpen r rs j) : p.1 < r.w.parts.length := by
  unfold roundOpen at hp
  rcases List.mem_append.1 hp with hp | hp
  · obtain ⟨e, he', rfl⟩ := List.mem_map.1 hp
    have hlt := topsBefore_lt (Nat.mod_lt j hpos) (List.mem_range.1 he')
    show capsFrom r rs + topsBefore rs (j % rs.size) + e < r.w.parts.length
    unfold capsFrom
    omega
  · exact rowOpen_lt hp

theorem finPieces_finOpen (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) : FinPieces (finOpen rest) shape := by
  intro x st hst _
  obtain ⟨r, hr, -⟩ := readX_of_parse hst
  obtain ⟨N, hcap, hn0⟩ := capAt_readX hr
  have hnS : nS x = r.w.parts.length := by simp [nS, hr]
  have hf : finOpen rest x = openPieces r ++ (rest r).map fun p => (p.1.val, p.2) := by simp [finOpen, hr]
  rw [hf]
  refine ⟨by simp [openPieces, hcap], fun p hp => ?_⟩
  rw [hnS]
  rcases List.mem_append.1 hp with hp | hp
  · unfold openPieces at hp
    rcases List.mem_append.1 hp with hp | hp
    · simp only [hcap, Option.toList_some, List.map_cons, List.map_nil, List.mem_singleton] at hp
      subst hp
      show nS x - 1 < r.w.parts.length
      omega
    · cases hrs : repSched r.shape with
      | none => simp [hrs] at hp
      | some rs =>
        simp only [hrs, Option.map_some, Option.getD_some, List.mem_flatMap, List.mem_range] at hp
        obtain ⟨j, hj, hp⟩ := hp
        exact roundOpen_lt (by omega) (topsLen_readX hr hrs) hp
  · obtain ⟨q, -, rfl⟩ := List.mem_map.1 hp
    exact q.1.isLt

/-- Bounds no number by itself: `PartChecksBV`, `PartOpensB`, `DecompB`, the readers' two facts and `ReadsBackB` stay
open. -/
theorem ksLin2_deployed_open (b : Brings)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (untr : TrReader) :
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) untr →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hrb hP hRT
  exact ksLin2_deployed_staged b (finOpen rest) wE join final w untr hrb
    (fun shape hsh => ⟨finPieces_finOpen rest shape, hP shape hsh⟩) hRT

/-- A bound on a number under `StrictBirthday`, `ExpectedCR` and `ksLin2_deployed_open`'s open premises, not a proved
security level. -/
theorem bitsLin2_deployed_open (b : Brings)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (untr : TrReader) :
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) untr →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hrb hP hRT
  exact bitsLin2_deployed_staged b (finOpen rest) wE join final w untr hrb
    (fun shape hsh => ⟨finPieces_finOpen rest shape, hP shape hsh⟩) hRT

end RecDeploy
