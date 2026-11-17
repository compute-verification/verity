module

public import Proofs.Flock.Claims.Recursion.VerdictPieces
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
import Proofs.Flock.Claims.Recursion.ResAccepts

/-! The `rec-cap` pieces of the verdict as the parts compute them (`capPieces_computes`); S16 and S18 need no piece, as
`decR` reads both caps and `root_B` off round 0's nodes. -/

@[expose] public section

namespace RecDeploy

open Claims (Msg)
open FlockDeploy (Coin nodeRow nodeOf isNodeRow nodeRow_nodeOf)
open Claims.Recursion.VStarB

theorem nodeRows_of_all {rs : List ByteArray} (h : ∀ r ∈ rs, isNodeRow r = true) :
    (rs.map nodeOf).map nodeRow = rs := by
  rw [List.map_map]
  exact (List.map_congr_left fun r hr => nodeRow_nodeOf (h r hr)).trans (List.map_id rs)

structure CapPiece where
  level : ℕ
  node : ℕ
  key : ℕ × ℕ × ℕ × ℕ

def capHolds (rowAt : Msg → List Msg → CapPiece → ByteArray) (q : CapPiece) (x : Msg) (ms : List Msg)
    (_cs : List Coin) : Prop :=
  isNodeRow (rowAt x ms q) = true

def capTop (v : ℕ → Bool) : ByteArray := FlockVBridge.Family.bytesAt v 0 128

theorem capPieces_computes {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {L : LayoutB}
    {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {b : Brings} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ}
    {x : Msg} {c : ℕ → Coin} {ms : List Msg}
    (h : PartAccepts H D (deployZkRec _root_.Flock.Readable.relation) L shape (showsPart untr b) cmt s x c ms)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss)
    {l N : ℕ} (hsv : ss[s]? = some (.cap l N))
    (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
    (pieces : Msg → List CapPiece) (ownQ : Msg → CapPiece → Option ℕ)
    (rowAt : Msg → List Msg → CapPiece → ByteArray)
    (hpad : ∀ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p → L.Agrees cmt x s w xO ms →
      (∀ e ∈ L.calls x s w, H e.1 = e.2) → (Claims.findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false →
      showsTr untr s x xO c →
      ∀ t (ht : t < p.sts.size) (o g u : ℕ), p.sts[t].c.unit.name = VStar.capName l N →
        o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) → g < p.sts[t].c.g → u < p.sts[t].c.upv →
        isNodeRow (capTop (_root_.Flock.Readable.copyOf p t ht w o g u)) = true)
    (hcopy : ∀ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p → L.RowsRead x s w ms →
      ∀ q ∈ pieces (Claims.Recursion.xI D x), ownQ x q = some s →
        ∃ ht : q.key.1 < p.sts.size, q.key.2.1 < 2 ^ (p.sts[q.key.1].m - p.sts[q.key.1].c.kLog) ∧
          q.key.2.2.1 < p.sts[q.key.1].c.g ∧ q.key.2.2.2 < p.sts[q.key.1].c.upv ∧
          ∀ os last, Bd.final (ms.getD (Bd.DE.V.k (Claims.Recursion.xI D x) - 1) []) = some (os, last) →
            os.length + 1 = Bd.DE.V.k (Claims.Recursion.xI D x) →
            rowAt (Claims.Recursion.xI D x) (Bd.opened os last) q =
              capTop (_root_.Flock.Readable.copyOf p q.key.1 ht w q.key.2.1 q.key.2.2.1 q.key.2.2.2)) :
    PiecesComputes Bd D pieces (capHolds rowAt) ownQ s x c ms := by
  obtain ⟨xO, w, p, r', sv, hx, -, -, hr', ⟨ss', hss', hsv'⟩, -, hcp, hA, hT, hc, hs⟩ := partAccepts_staged h
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  have hRR : L.RowsRead x s w ms := by
    obtain ⟨-, -, -, hRR⟩ := hA
    exact hRR
  intro q hq ho os last hf hl
  obtain ⟨ht, hob, hgb, hub, hrow⟩ := hcopy xO w p hx hRR q hq ho
  unfold capHolds
  rw [hrow os last hf hl]
  exact hpad xO w p hx hA hT hc hs q.key.1 ht q.key.2.1 q.key.2.2.1 q.key.2.2.2 (hcp q.key.1 ht).1 hob hgb hub

end RecDeploy
