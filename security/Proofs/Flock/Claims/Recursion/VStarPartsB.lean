module

public import Proofs.Flock.Claims.Recursion.VStarParts
public import Proofs.Flock.Claims.Recursion.VStarBound
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

/-! V\*'s algebra and rec-open parts at the bound layout, where only the input tie changes; `PartOpensB` takes no port of `partOpens_of_climbs`, whose setup-collision escape it doesn't admit. -/

@[expose] public section

namespace Claims.Recursion.VStarB

open Claims
open FlockVBridge.OfVStar

def LayoutB.Agrees (L : LayoutB) (cmt : Msg → ℕ → Msg → Msg) (x : Msg) (s : ℕ) (w : Flock.Wit) (xO : Msg)
    (ms : List Msg) : Prop :=
  (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) ∧
    (∀ i < L.rounds x, ∀ o m, (o, m) ∈ L.piecesOf x s w i → ((ms.getD i []).drop o).take m.length = m) ∧
    L.CmtAgrees cmt x xO ms ∧ L.RowsRead x s w ms

section Facts

variable {Coin Col T : Type}

/-- A code fact about V\*'s staging, the layout and the inner check at part `s`, none of it the circuit's semantics. -/
def ResLinksB (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : LayoutB) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin), shows s x xO c →
    ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
      ∃ t, ∃ ht : t < p.sts.size, ∃ (m k cl j : ℕ) (sh : _root_.Flock.Shape) (S : _root_.Flock.RecOpen.GfS)
        (ns : List ℕ) (nv : ℕ),
        _root_.Flock.RecResiduals.check p.sts[t].c = .ok () ∧
        _root_.Flock.RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, j) ∧
        _root_.Flock.RecResiduals.shape? m k cl = some sh ∧ _root_.Flock.RecResiduals.holoOf p.sts[t].c = .ok false ∧
        _root_.Flock.RecResiduals.part sh j = .ok (S, ns, nv) ∧
        ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
          ∀ (w : Flock.Wit) (ms : List Msg), L.Agrees cmt x s w xO ms →
            ∃ W vv : ℕ → _root_.Flock.F128, (∀ e, ns.sum ≤ e → W e = 0) ∧ (∀ e, nv ≤ e → vv e = 0) ∧
              InCols (partRep ns nv) W vv (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos) ∧
              (PartCols S W vv (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos)
                (partBase S ns nv) → chk s x c ms)

theorem partChecksB_of_res {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : LayoutB}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : ResLinksB D decode L shape shows cmt chk s) : PartChecksB H D DO L shape shows cmt chk s := by
  intro x st hst hf xO w hck _
  refine Or.inl fun c hc ms hhd hpc hcm hrr => ?_
  obtain ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, S, ns, nv, hrc, hp, hsh, hh, hpart, o, g, u, ho, hg, hu, hlink⟩ :=
    hR x st hst hf xO c hc
  obtain ⟨W, vv, hw0, hv0, hin, hout⟩ := hlink w ms ⟨hhd, hpc, hcm, hrr⟩
  rw [hDO] at hck
  exact hout (Flock.Readable.partCols_of_check hck hx ht hpi htl hrc hp hsh hh hpart ho hg hu W vv hw0 hv0 hin)

/-- S's residuals at the elements the copy reads give the inner check once those are the agreeing messages' and the coins' coefficients. -/
def ResReadsB (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : LayoutB) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin), shows s x xO c →
    ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
      ∃ t, ∃ ht : t < p.sts.size, ∃ (m k cl j : ℕ) (sh : _root_.Flock.Shape) (S : _root_.Flock.RecOpen.GfS)
        (ns : List ℕ) (nv : ℕ),
        _root_.Flock.RecResiduals.check p.sts[t].c = .ok () ∧
        _root_.Flock.RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, j) ∧
        _root_.Flock.RecResiduals.shape? m k cl = some sh ∧ _root_.Flock.RecResiduals.holoOf p.sts[t].c = .ok false ∧
        _root_.Flock.RecResiduals.part sh j = .ok (S, ns, nv) ∧
        ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
          ∀ (w : Flock.Wit) (ms : List Msg), L.Agrees cmt x s w xO ms →
            PartCols S (Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t ht w o g u))
              (Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t ht w o g u))
              (Flock.Readable.copyOf p t ht w o g u) (partBase S ns nv) → chk s x c ms

theorem resLinksB_of_reads {D : Deploy Coin} {decode : Msg → Option Flock.Readable.Read} {L : LayoutB}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} (hR : ResReadsB D decode L shape shows cmt chk s) :
    ResLinksB D decode L shape shows cmt chk s := by
  intro x st hst hf xO c hc
  obtain ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, S, ns, nv, hrc, hp, hsh, hh, hpart, o, g, u, ho, hg, hu, hlink⟩ :=
    hR x st hst hf xO c hc
  refine ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, S, ns, nv, hrc, hp, hsh, hh, hpart, o, g, u, ho, hg, hu,
    fun w ms hms => ?_⟩
  exact ⟨Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t htl w o g u),
    Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t htl w o g u),
    fun _ he => Flock.Readable.readElems_ge he, fun _ he => Flock.Readable.readElems_ge he,
    Flock.Readable.inCols_readElems ns nv _, hlink w ms hms⟩

theorem partChecksB_of_reads {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : LayoutB}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : ResReadsB D decode L shape shows cmt chk s) : PartChecksB H D DO L shape shows cmt chk s :=
  partChecksB_of_res hDO (resLinksB_of_reads hR)

/-- A code fact about V\*'s staging, the layout and the inner check at part `s`, none of it the circuit's semantics. -/
def OpenLinksB (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : LayoutB) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin), shows s x xO c →
    ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
      ∃ t, ∃ ht : t < p.sts.size, ∃ LANES Ht : ℕ,
        _root_.Flock.RecOpen.check p.sts[t].c = .ok () ∧
        _root_.Flock.RecOpen.params p.sts[t].c.unit.name = some (LANES, Ht) ∧
        ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
          ∀ (w : Flock.Wit) (ms : List Msg), L.Agrees cmt x s w xO ms →
            ∃ (msch : _root_.Flock.MerkleScheme) (r a cf : Array _root_.Flock.F128) (rest : ByteArray) (idx : ℕ)
              (ss : List ByteArray), (∀ l r, msch.node l r = _root_.Flock.Sha512.hash (l ++ r)) ∧
              Flock.Readable.OpenIns (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos) LANES Ht r a cf rest
                idx ss ∧
              (Flock.Readable.OpenOuts (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos) msch LANES Ht r a cf
                rest idx ss → chk s x c ms)

theorem partChecksB_of_open {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : LayoutB}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : OpenLinksB D decode L shape shows cmt chk s) : PartChecksB H D DO L shape shows cmt chk s := by
  intro x st hst hf xO w hck _
  refine Or.inl fun c hc ms hhd hpc hcm hrr => ?_
  obtain ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, hp, o, g, u, ho, hg, hu, hlink⟩ := hR x st hst hf xO c hc
  obtain ⟨msch, r, a, cf, rest, idx, ss, hmsch, ⟨hr, ha, hcf, hrest, hss, hrow, haux, hacc, hcoef, hdir⟩, hout⟩ :=
    hlink w ms ⟨hhd, hpc, hcm, hrr⟩
  rw [hDO] at hck
  exact hout (Flock.Readable.climb_of_check hck hx ht hpi htl ho hg hu rfl msch hmsch hrec hp r a cf rest idx ss hr
    ha hcf hrest hss hrow haux hacc hcoef hdir)

end Facts

end Claims.Recursion.VStarB
