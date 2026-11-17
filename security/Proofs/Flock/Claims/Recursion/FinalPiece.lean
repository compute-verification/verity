module

public import Proofs.Flock.Claims.Recursion.PartAccepts
public import Proofs.Flock.Claims.Recursion.ResAccepts
public import Proofs.Flock.Claims.Recursion.RowSlots
public import Proofs.Flock.Claims.Recursion.VStarPartsB
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

/-! A `rec-residuals` piece of the final check from `PartAccepts` (`partAccepts_resZero`), at one open hypothesis, check
5's tie `ResTie`. -/

@[expose] public section

namespace RecDeploy

open Claims (Msg)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB
open FlockSoundness.Discharge.Hm (bitL)

theorem rowCol_take : ∀ (ps : List (String × ℕ)) (k j e t : ℕ) (hj : j < ps.length), e < ps[j].2 → t < 128 →
    FlockVBridge.OfVStar.rowCol ps k (128 * ((ps.take j).map (·.2)).sum + (128 * e + t)) =
      FlockVBridge.OfVStar.rowCol ps k (128 * ((ps.take j).map (·.2)).sum) + (128 * e + t)
  | [], _, _, _, _, hj, _, _ => absurd hj (Nat.not_lt_zero _)
  | p :: ps, k, 0, e, t, _, he, ht => by
    have he' : e < p.2 := he
    show FlockVBridge.OfVStar.rowCol (p :: ps) k (128 * 0 + (128 * e + t)) =
      FlockVBridge.OfVStar.rowCol (p :: ps) k (128 * 0) + (128 * e + t)
    simp only [FlockVBridge.OfVStar.rowCol]
    rw [ite_eq_left (show 128 * 0 + (128 * e + t) < 128 * p.2 by omega),
      ite_eq_left (show 128 * 0 < 128 * p.2 by omega)]
    omega
  | p :: ps, k, j + 1, e, t, hj, he, ht => by
    have hj' : j < ps.length := by simp at hj; omega
    have he' : e < ps[j].2 := he
    have ih := fun k' => rowCol_take ps k' j e t hj' he' ht
    generalize hS : ((ps.take j).map (·.2)).sum = S at ih
    show FlockVBridge.OfVStar.rowCol (p :: ps) k (128 * (p.2 + ((ps.take j).map (·.2)).sum) + (128 * e + t)) =
      FlockVBridge.OfVStar.rowCol (p :: ps) k (128 * (p.2 + ((ps.take j).map (·.2)).sum)) + (128 * e + t)
    rw [hS]
    simp only [FlockVBridge.OfVStar.rowCol]
    rw [ite_eq_right (show ¬ 128 * (p.2 + S) + (128 * e + t) < 128 * p.2 by omega),
      ite_eq_right (show ¬ 128 * (p.2 + S) < 128 * p.2 by omega),
      show 128 * (p.2 + S) + (128 * e + t) - 128 * p.2 = 128 * S + (128 * e + t) by omega,
      show 128 * (p.2 + S) - 128 * p.2 = 128 * S by omega]
    exact ih _

theorem take_ports : ∀ (ns : List ℕ) (j : ℕ), (((ns.map ("", ·)).take j).map (·.2)) = ns.take j
  | [], j => by simp
  | _ :: _, 0 => rfl
  | n :: ns, j + 1 => congrArg (List.cons n) (take_ports ns j)

theorem take_sum_lt : ∀ (ns : List ℕ) (j : ℕ) (hj : j < ns.length) (e : ℕ), e < ns[j] →
    (ns.take j).sum + e < ns.sum
  | [], _, hj, _, _ => absurd hj (Nat.not_lt_zero _)
  | n :: ns, 0, _, e, he => by
    have he' : e < n := he
    show 0 + e < n + ns.sum
    omega
  | n :: ns, j + 1, hj, e, he => by
    have hj' : j < ns.length := by simp at hj; omega
    have ih := take_sum_lt ns j hj' e he
    show n + (ns.take j).sum + e < n + ns.sum
    omega

def rowPortCol (ns : List ℕ) (j : ℕ) : ℕ := FlockVBridge.OfVStar.rowCol (ns.map ("", ·)) 0 (128 * (ns.take j).sum)

theorem rowsRead_port {ns : List ℕ} {nv : ℕ} {v : ℕ → Bool} {j : ℕ} (hj : j < ns.length) {e : ℕ}
    (he : e < ns[j]) :
    _root_.Flock.Readable.rowsRead ns nv v ((ns.take j).sum + e) =
      _root_.Flock.Readable.readElems (rowPortCol ns j + ·) ns[j] v e := by
  unfold _root_.Flock.Readable.rowsRead _root_.Flock.Readable.readElems
  rw [ite_eq_left (take_sum_lt ns j hj e he), ite_eq_left he]
  refine FlockVBridge.OfVStar.ofBits_congr fun q hq => ?_
  show v (FlockVBridge.OfVStar.rowCol (ns.map ("", ·)) 0 (128 * ((ns.take j).sum + e) + q)) =
    v (FlockVBridge.OfVStar.rowCol (ns.map ("", ·)) 0 (128 * (ns.take j).sum) + (128 * e + q))
  have h := rowCol_take (ns.map ("", ·)) 0 j e q (by simpa using hj) (by simpa using he) hq
  rw [take_ports] at h
  rw [Nat.mul_add, Nat.add_assoc, h]

def elemsOf (bs : List UInt8) (n e : ℕ) : _root_.Flock.F128 :=
  if e < n then FlockVBridge.ofBits fun t => bitL bs (128 * e + t) else 0

theorem readElems_of_colsHold {v : ℕ → Bool} {base n : ℕ} {b : ByteArray} (hb : FlockVBridge.ColsHold v base b)
    (hn : 128 * n ≤ 8 * b.size) (e : ℕ) :
    _root_.Flock.Readable.readElems (base + ·) n v e = elemsOf b.data.toList n e := by
  unfold _root_.Flock.Readable.readElems elemsOf
  split_ifs with he
  · exact FlockVBridge.OfVStar.ofBits_congr fun q hq => hb _ (by omega)
  · rfl

theorem getD_drop_of_take {l B pre : List UInt8} (h : l.take (pre ++ B).length = pre ++ B) {i : ℕ}
    (hi : i < B.length) : (l.drop pre.length).getD i 0 = B.getD i 0 := by
  have h1 : (l.take (pre ++ B).length)[pre.length + i]? = (pre ++ B)[pre.length + i]? := by rw [h]
  rw [List.getElem?_take_of_lt (by simp; omega), List.getElem?_append_right (by omega),
    Nat.add_sub_cancel_left] at h1
  simp only [List.getD_eq_getElem?_getD, List.getElem?_drop, h1]

theorem elemsOf_drop_of_take {l B pre : List UInt8} (h : l.take (pre ++ B).length = pre ++ B) {n : ℕ}
    (hn : 16 * n ≤ B.length) (e : ℕ) : elemsOf (l.drop pre.length) n e = elemsOf B n e := by
  unfold elemsOf
  split_ifs with he
  · refine FlockVBridge.OfVStar.ofBits_congr fun q hq => ?_
    show (bitL (l.drop pre.length) (128 * e + q)) = bitL B (128 * e + q)
    unfold bitL
    rw [getD_drop_of_take h (by omega)]
  · rfl

theorem rowsRead_msgs {L : LayoutB} {x : Msg} {s : ℕ} {w : Claims.Flock.Wit} {ms : List Msg}
    (hR : L.RowsRead x s w ms) {p : _root_.Flock.Readable.Parsed} {t : ℕ} {ht : t < p.sts.size} {o g u : ℕ}
    {ns : List ℕ} {nv j : ℕ} (hj : j < ns.length) {r N i off : ℕ} {pre : Msg}
    (hm : L.rowAt x s r = some (segsAt [(pre, Claims.Recursion.VStar.copyPs p t ht o g u (rowPortCol ns j) (8 * N))]))
    (htop : rowPortCol ns j + 8 * N ≤ p.sts[t].c.unit.constPos + 1) (hN : 16 * ns[j] ≤ N)
    (hi : L.rowIn x r = some (i, off)) {e : ℕ} (he : e < ns[j]) :
    _root_.Flock.Readable.rowsRead ns nv (_root_.Flock.Readable.copyOf p t ht w o g u) ((ns.take j).sum + e) =
      elemsOf (((ms.getD i []).drop off).drop pre.length) ns[j] e := by
  have hb := FlockVBridge.Family.colsHold_bytesAt (_root_.Flock.Readable.copyOf p t ht w o g u) (rowPortCol ns j) N
  have hs := FlockVBridge.Family.size_bytesAt (_root_.Flock.Readable.copyOf p t ht w o g u) (rowPortCol ns j) N
  have hB : (FlockVBridge.Family.bytesAt (_root_.Flock.Readable.copyOf p t ht w o g u) (rowPortCol ns j)
      N).data.toList.length = N := by
    rw [Array.length_toList]
    exact hs
  rw [← hs] at hm htop
  have hmsg := hR r _ i off (rowHeld_of_colsHold hm htop hb) hi
  rw [rowsRead_port hj he, readElems_of_colsHold hb (by rw [hs]; omega) e]
  exact (elemsOf_drop_of_take hmsg (by rw [hB]; omega) e).symm

def ResTie (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ) (vv : ℕ → _root_.Flock.F128) (v : ℕ → Bool) :
    Prop :=
  (∀ q < 128 * S.res.length, v (FlockVBridge.OfVStar.partBase S ns nv + q) = false) ∧
    _root_.Flock.Readable.vRead ns nv v = vv

theorem resVal_zero_of_tie {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} {vv : ℕ → _root_.Flock.F128}
    {v : ℕ → Bool} (h : _root_.Flock.Readable.ResOuts S ns nv v) (htie : ResTie S ns nv vv v) :
    ∀ q (hq : q < S.res.length), FlockVBridge.resVal (FlockVBridge.gfsOf S)
      (FlockVBridge.termVal (_root_.Flock.Readable.rowsRead ns nv v) vv) (FlockVBridge.resOf S.res[q]) = 0 := by
  intro q hq
  obtain ⟨hz, hv⟩ := htie
  unfold _root_.Flock.Readable.ResOuts FlockVBridge.OfVStar.PartCols at h
  rw [← hv]
  refine FlockVBridge.AccChain.eq_of_bitOf fun k hk => ?_
  rw [← h.1 q hq k hk, FlockVBridge.OfVStar.bitOf_zero]
  exact hz _ (by omega)

/-- `htie`, check 5's tie at every copy of an accepted witness, is its one open hypothesis. -/
theorem partAccepts_resZero {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {L : LayoutB}
    {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {b : Brings} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ}
    {x : Msg} {c : ℕ → Coin} {ms : List Msg}
    (h : PartAccepts H D (deployZkRec _root_.Flock.Readable.relation) L shape (showsPart untr b) cmt s x c ms)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm pt))
    (vv : ℕ → ℕ → ℕ → ℕ → ℕ → _root_.Flock.F128)
    (htie : ∀ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p → L.Agrees cmt x s w xO ms →
      (∀ e ∈ L.calls x s w, H e.1 = e.2) → (Claims.findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false →
      showsTr untr s x xO c →
      ∀ t (ht : t < p.sts.size) (o g u : ℕ), o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) → g < p.sts[t].c.g →
        u < p.sts[t].c.upv →
        ∀ (m k cl jj : ℕ) (sh : _root_.Flock.Shape) (holo : Bool) (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ)
          (nv : ℕ), _root_.Flock.RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, jj) →
          _root_.Flock.RecResiduals.shape? m k cl = some sh → _root_.Flock.RecResiduals.holoOf p.sts[t].c = .ok holo →
          _root_.Flock.RecResiduals.part { sh with holo } jj = .ok (S, ns, nv) →
          ResTie S ns nv (vv t o g u) (_root_.Flock.Readable.copyOf p t ht w o g u)) :
    ∃ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p ∧ L.RowsRead x s w ms ∧
      ∀ t (ht : t < p.sts.size) (o g u : ℕ), o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) → g < p.sts[t].c.g →
        u < p.sts[t].c.upv →
        ∃ (m k cl jj : ℕ) (sh : _root_.Flock.Shape) (holo : Bool) (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ)
          (nv : ℕ),
          _root_.Flock.RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, jj) ∧
          _root_.Flock.RecResiduals.shape? m k cl = some sh ∧
          _root_.Flock.RecResiduals.holoOf p.sts[t].c = .ok holo ∧
          _root_.Flock.RecResiduals.part { sh with holo } jj = .ok (S, ns, nv) ∧
          (∀ q (hq : q < S.res.length), FlockVBridge.resVal (FlockVBridge.gfsOf S)
            (FlockVBridge.termVal (_root_.Flock.Readable.rowsRead ns nv (_root_.Flock.Readable.copyOf p t ht w o g u))
              (vv t o g u)) (FlockVBridge.resOf S.res[q]) = 0) ∧
          ∀ (jp : ℕ) (hj : jp < ns.length) (rr N i off : ℕ) (pre : Msg),
            L.rowAt x s rr =
              some (segsAt [(pre, Claims.Recursion.VStar.copyPs p t ht o g u (rowPortCol ns jp) (8 * N))]) →
            rowPortCol ns jp + 8 * N ≤ p.sts[t].c.unit.constPos + 1 → 16 * ns[jp] ≤ N →
            L.rowIn x rr = some (i, off) → ∀ e, e < ns[jp] →
              _root_.Flock.Readable.rowsRead ns nv (_root_.Flock.Readable.copyOf p t ht w o g u)
                ((ns.take jp).sum + e) = elemsOf (((ms.getD i []).drop off).drop pre.length) ns[jp] e := by
  obtain ⟨xO, w, p, hx, -, hA, hT, hc, hs, hcp⟩ := partAccepts_inner h hr hss hsv
  have hRR : L.RowsRead x s w ms := by
    obtain ⟨-, -, -, hRR⟩ := hA
    exact hRR
  refine ⟨xO, w, p, hx, hRR, fun t ht o g u ho hg hu => ?_⟩
  obtain ⟨m, k, cl, jj, sh, holo, S, ns, nv, h1, h2, h3, h4, hro⟩ := hcp t ht o g u ho hg hu
  exact ⟨m, k, cl, jj, sh, holo, S, ns, nv, h1, h2, h3, h4,
    resVal_zero_of_tie hro (htie xO w p hx hA hT hc hs t ht o g u ho hg hu m k cl jj sh holo S ns nv h1 h2 h3 h4),
    fun jp hj rr N i off pre hm htop hN hi e he => rowsRead_msgs hRR hj hm htop hN hi he⟩

end RecDeploy
