module

public import Proofs.Flock.Claims.Recursion.VStarParts
public import Proofs.Flock.Claims.Recursion.DeployReg
public import Proofs.Flock.VBridge.Sparse.Laid
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.VBridge.OpeningUnit
import Proofs.Flock.VBridge.Sparse.Unit

/-! V\*'s parts at the holo shape: each kind's template at `Flock.check`, the links that are not the circuit's, and `PartChecks`, `PartLogs` and `PartOpens` from them. -/

@[expose] public section

namespace Flock.Readable

open FlockSoundness FlockSoundness.Discharge.Hm FlockVBridge.OfVStar

theorem partCols_of_satisfies_at {st : Flock.Stmt} (L : FlockSoundness.Layout st) (hfl : st.c.tmpl = none)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {J c : ℕ} {w : Claims.Flock.Wit}
    (hz : (L.model regs).Satisfies (zOf J w c st.m)) {o g u : ℕ} (ho : o < 2 ^ (st.m - st.c.kLog))
    (hg : g < st.c.g) (hu : u < st.c.upv) (hc : RecResiduals.check st.c = .ok ()) {m k cl j : ℕ}
    (hp : RecResiduals.params st.c.unit.name = some (m, k, cl, j)) {sh : Flock.Shape}
    (hsh : RecResiduals.shape? m k cl = some sh) {holo : Bool} (hh : RecResiduals.holoOf st.c = .ok holo)
    {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} (hpart : RecResiduals.part { sh with holo } j = .ok (S, ns, nv))
    (W vv : ℕ → Flock.F128) (hw0 : ∀ e, ns.sum ≤ e → W e = 0) (hv0 : ∀ e, nv ≤ e → vv e = 0)
    (hin : InCols (partRep ns nv) W vv
      (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos)) :
    PartCols S W vv (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos)
      (partBase S ns nv) := by
  obtain ⟨S', ns', nv', hpart', hn⟩ := check_part_at hc hp hsh hh
  rw [hpart] at hpart'
  cases hpart'
  have hL := laidRows_of_satisfies L hfl regs hz ho hg hu hn
  rw [inWords_unitLaid hn] at hL
  exact laid_part S ns nv (part_small hpart) _ hL W vv hw0 hv0 hin

theorem partCols_of_check_at {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    (hc : RecResiduals.check p.sts[t].c = .ok ()) {m k cl j : ℕ}
    (hp : RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, j)) {sh : Flock.Shape}
    (hsh : RecResiduals.shape? m k cl = some sh) {holo : Bool} (hh : RecResiduals.holoOf p.sts[t].c = .ok holo)
    {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} (hpart : RecResiduals.part { sh with holo } j = .ok (S, ns, nv))
    {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    (W vv : ℕ → Flock.F128) (hw0 : ∀ e, ns.sum ≤ e → W e = 0) (hv0 : ∀ e, nv ≤ e → vv e = 0)
    (hin : InCols (partRep ns nv) W vv (copyOf p t htl w o g u)) :
    PartCols S W vv (copyOf p t htl w o g u) (partBase S ns nv) := by
  obtain ⟨L, regs, hz⟩ := layout_of_check h hx ht hpi htl
  exact partCols_of_satisfies_at L (flat_of_check h hx ht hpi htl (recUnit_residuals hp)) regs hz ho hg hu hc hp hsh hh hpart W vv hw0 hv0 hin

theorem openOuts_of_net {st : Flock.Stmt} (L : FlockSoundness.Layout st) (hfl : st.c.tmpl = none)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {J c : ℕ} {w : Claims.Flock.Wit}
    (hz : (L.model regs).Satisfies (zOf J w c st.m)) {o g u : ℕ} (ho : o < 2 ^ (st.m - st.c.kLog))
    (hg : g < st.c.g) (hu : u < st.c.upv) {LANES H : ℕ} {name : String}
    (hvalid : Flock.RecOpen.valid LANES H = true)
    (hl : Flock.HmNets.netOf name (Flock.RecOpen.unitLaid LANES H) = .ok st.c.unit)
    (ms : Flock.MerkleScheme) (hms : ∀ l r, ms.node l r = Flock.Sha512.hash (l ++ r))
    {r a cf : Array Flock.F128} {rest : ByteArray} {idx : ℕ} {ss : List ByteArray}
    (hin : OpenIns (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos) LANES H r
      a cf rest idx ss) :
    OpenOuts (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos) ms LANES H r a
      cf rest idx ss := by
  obtain ⟨hr, ha, hcf, hrest, hss, hrow, haux, hacc, hcoef, hdir⟩ := hin
  have hH : H ≤ 32 := by
    simp only [Flock.RecOpen.valid, Bool.and_eq_true, decide_eq_true_eq] at hvalid
    exact hvalid.2
  have hL := laidRows_of_satisfies L hfl regs hz ho hg hu hl
  rw [FlockVBridge.inWords_unit hl] at hL
  exact FlockVBridge.laid_unit ms hms LANES H hH _ hL r a cf rest idx ss hr ha hcf hrest hss hrow haux hacc hcoef hdir

theorem recOpen_check_ok_any {c : Flock.Circuit} (h : Flock.RecOpen.check c = .ok ()) {LANES H : ℕ}
    (hp : Flock.RecOpen.params c.unit.name = some (LANES, H) ∨
      Flock.RecOpen.stackedParams c.unit.name = some (LANES, H)) :
    ∃ name, Flock.RecOpen.valid LANES H = true ∧
      Flock.HmNets.netOf name (Flock.RecOpen.unitLaid LANES H) = .ok c.unit := by
  rcases hp with hp | hp
  · exact ⟨_, Flock.RecOpen.check_ok h hp⟩
  · exact ⟨_, Flock.RecOpen.check_ok_stacked h hp⟩

theorem openOuts_of_check_any {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    (hrec : Flock.RecOpen.check p.sts[t].c = .ok ()) {LANES H : ℕ}
    (hp : Flock.RecOpen.params p.sts[t].c.unit.name = some (LANES, H) ∨
      Flock.RecOpen.stackedParams p.sts[t].c.unit.name = some (LANES, H))
    (ms : Flock.MerkleScheme) (hms : ∀ l r, ms.node l r = Flock.Sha512.hash (l ++ r))
    {r a cf : Array Flock.F128} {rest : ByteArray} {idx : ℕ} {ss : List ByteArray}
    (hin : OpenIns (copyOf p t htl w o g u) LANES H r a cf rest idx ss) :
    OpenOuts (copyOf p t htl w o g u) ms LANES H r a cf rest idx ss := by
  obtain ⟨L, regs, hz⟩ := layout_of_check h hx ht hpi htl
  obtain ⟨_, hvalid, hl⟩ := recOpen_check_ok_any hrec hp
  exact openOuts_of_net L (flat_of_check h hx ht hpi htl (recUnit_open hp)) regs hz ho hg hu hvalid hl ms hms hin

theorem partCols_of_check_opening {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    (hc : RecOpening.check p.sts[t].c = .ok ()) {m cl j : ℕ}
    (hp : RecOpening.params p.sts[t].c.unit.name = some (m, cl, j)) {sh : Flock.Shape}
    (hsh : RecOpening.shape? m cl = some sh) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hpart : RecOpening.part sh j = .ok (S, ns, nv))
    {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    (W vv : ℕ → Flock.F128) (hw0 : ∀ e, ns.sum ≤ e → W e = 0) (hv0 : ∀ e, nv ≤ e → vv e = 0)
    (hin : InCols (partRep ns nv) W vv (copyOf p t htl w o g u)) :
    PartCols S W vv (copyOf p t htl w o g u) (partBase S ns nv) := by
  obtain ⟨L, regs, hz⟩ := layout_of_check h hx ht hpi htl
  obtain ⟨S', ns', nv', hpart', hn⟩ := check_opening hc hp hsh
  rw [hpart] at hpart'
  cases hpart'
  have hL := laidRows_of_satisfies L (flat_of_check h hx ht hpi htl (recUnit_opening hp)) regs hz ho hg hu hn
  rw [inWords_unitLaid hn] at hL
  exact laid_part S ns nv (opening_small hpart) _ hL W vv hw0 hv0 hin

theorem inWords_sparse {name : String} {sh : Flock.RecSparse.Shape} {net : Flock.Net}
    (hn : Flock.HmNets.netOf name (Flock.RecSparse.unitLaid sh) = .ok net) :
    net.inWords * Flock.WORD = inRowsOf (FlockVBridge.Sparse.sparseInBits sh) := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, hw, -⟩ := Flock.Net.ofRows_ok hn
  rw [hw]
  simp only [Flock.RecSparse.unitLaid, layout_inGroups]
  simp [inRowsOf, FlockVBridge.Sparse.sparseInBits]

def SparseCopyOuts (sh : Flock.RecSparse.Shape) (v : ℕ → Bool) (wA vA : Array Flock.F128) (τA : Array ℕ) : Prop :=
  (∀ r (hr : r < (FlockVBridge.Sparse.outOf (FlockVBridge.Sparse.shOf sh)
      (FlockVBridge.Sparse.resArr (FlockVBridge.Sparse.shOf sh) wA vA) wA vA τA).1.size), ∀ k < 128,
    v (FlockVBridge.Sparse.sparseBase sh + (128 * r + k)) =
      FlockVBridge.bitOf (FlockVBridge.Sparse.outOf (FlockVBridge.Sparse.shOf sh)
        (FlockVBridge.Sparse.resArr (FlockVBridge.Sparse.shOf sh) wA vA) wA vA τA).1[r] k) ∧
  (sh.stacked = true → ∀ s < sh.S, ∀ b < 16,
    v (FlockVBridge.Sparse.sparseBase sh + (512 + (16 * s + b))) = (τA[s]! - τA[s]! % 2 ^ sh.tLog).testBit b)

theorem sparseCopyOuts_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    (hc : Flock.RecSparse.check p.sts[t].c = .ok ()) {sh : Flock.RecSparse.Shape}
    (hp : Flock.RecSparse.params p.sts[t].c.unit.name = some sh)
    {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    (wA vA : Array Flock.F128) (τA : Array ℕ)
    (hin : FlockVBridge.Sparse.SparseIn sh (wA[·]!) (vA[·]!) (τA[·]!) (copyOf p t htl w o g u)) :
    SparseCopyOuts sh (copyOf p t htl w o g u) wA vA τA := by
  obtain ⟨L, regs, hz⟩ := layout_of_check h hx ht hpi htl
  obtain ⟨hv, -, -, hn⟩ := FlockVBridge.Sparse.check_sparse hc hp
  have hL := laidRows_of_satisfies L (flat_of_check h hx ht hpi htl (recUnit_sparse hp)) regs hz ho hg hu (l := Flock.RecSparse.unitLaid sh) hn
  rw [inWords_sparse hn] at hL
  have ht16 : sh.tLog ≤ 16 := by
    simp only [Flock.RecSparse.Shape.valid, Bool.and_eq_true, decide_eq_true_eq] at hv
    exact hv.2
  exact FlockVBridge.Sparse.laid_sparse sh ht16 _ hL wA vA τA hin

end Flock.Readable

namespace Claims.Recursion.VStar

open FlockVBridge.OfVStar

section Facts

variable {Coin W Col T : Type}

def Agrees (L : Layout) (x : Msg) (s : ℕ) (w : Flock.Wit) (ms : List Msg) : Prop :=
  ∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m

def LevelAt (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (x : Msg) (s : ℕ) (xO : Msg) (c : ℕ → Coin) (w : Flock.Wit) : Prop :=
  ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
    ∃ t, ∃ ht : t < p.sts.size, ∃ LANES Ht : ℕ,
      _root_.Flock.RecOpen.check p.sts[t].c = .ok () ∧
      (_root_.Flock.RecOpen.params p.sts[t].c.unit.name = some (LANES, Ht) ∨
        _root_.Flock.RecOpen.stackedParams p.sts[t].c.unit.name = some (LANES, Ht)) ∧
      ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
        ∀ ms : List Msg, Agrees L x s w ms →
          ∃ (msch : _root_.Flock.MerkleScheme) (r a cf : Array _root_.Flock.F128) (rest : ByteArray) (idx : ℕ)
            (ss : List ByteArray), (∀ l r, msch.node l r = _root_.Flock.Sha512.hash (l ++ r)) ∧
            Flock.Readable.OpenIns (Flock.Readable.copyOf p t ht w o g u) LANES Ht r a cf rest idx ss ∧
            (Flock.Readable.OpenOuts (Flock.Readable.copyOf p t ht w o g u) msch LANES Ht r a cf rest idx ss →
              chk s x c ms)

def ResReadsAt (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (x : Msg) (s : ℕ) (xO : Msg) (c : ℕ → Coin) (w : Flock.Wit) : Prop :=
  ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
    ∃ t, ∃ ht : t < p.sts.size, ∃ (m k cl j : ℕ) (sh : _root_.Flock.Shape) (holo : Bool)
      (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ),
      _root_.Flock.RecResiduals.check p.sts[t].c = .ok () ∧
      _root_.Flock.RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, j) ∧
      _root_.Flock.RecResiduals.shape? m k cl = some sh ∧ _root_.Flock.RecResiduals.holoOf p.sts[t].c = .ok holo ∧
      _root_.Flock.RecResiduals.part { sh with holo } j = .ok (S, ns, nv) ∧
      ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
        ∀ ms : List Msg, Agrees L x s w ms →
          PartCols S (Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t ht w o g u))
            (Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t ht w o g u))
            (Flock.Readable.copyOf p t ht w o g u) (partBase S ns nv) → chk s x c ms

def OpeningReadsAt (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (x : Msg) (s : ℕ) (xO : Msg) (c : ℕ → Coin) (w : Flock.Wit) : Prop :=
  ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
    ∃ t, ∃ ht : t < p.sts.size, ∃ (m cl j : ℕ) (sh : _root_.Flock.Shape) (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ)
      (nv : ℕ),
      _root_.Flock.RecOpening.check p.sts[t].c = .ok () ∧
      _root_.Flock.RecOpening.params p.sts[t].c.unit.name = some (m, cl, j) ∧
      _root_.Flock.RecOpening.shape? m cl = some sh ∧ _root_.Flock.RecOpening.part sh j = .ok (S, ns, nv) ∧
      ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
        ∀ ms : List Msg, Agrees L x s w ms →
          PartCols S (Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t ht w o g u))
            (Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t ht w o g u))
            (Flock.Readable.copyOf p t ht w o g u) (partBase S ns nv) → chk s x c ms

def SparseCopyAt (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (x : Msg) (s : ℕ) (xO : Msg) (c : ℕ → Coin) (w : Flock.Wit) : Prop :=
  ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
    ∃ t, ∃ ht : t < p.sts.size, ∃ sh : _root_.Flock.RecSparse.Shape,
      _root_.Flock.RecSparse.check p.sts[t].c = .ok () ∧ _root_.Flock.RecSparse.params p.sts[t].c.unit.name = some sh ∧
      ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
        ∀ ms : List Msg, Agrees L x s w ms →
          ∃ (wA vA : Array _root_.Flock.F128) (τA : Array ℕ),
            FlockVBridge.Sparse.SparseIn sh (wA[·]!) (vA[·]!) (τA[·]!) (Flock.Readable.copyOf p t ht w o g u) ∧
            (Flock.Readable.SparseCopyOuts sh (Flock.Readable.copyOf p t ht w o g u) wA vA τA → chk s x c ms)

def PartKindAt (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (x : Msg) (s : ℕ) (xO : Msg) (c : ℕ → Coin) (w : Flock.Wit) : Prop :=
  LevelAt decode L chk x s xO c w ∨ ResReadsAt decode L chk x s xO c w ∨ OpeningReadsAt decode L chk x s xO c w ∨
    SparseCopyAt decode L chk x s xO c w

/-- A code fact about V\*'s staging, the layout and the inner check at part `s`, none of it the circuit's semantics. -/
def PartsLinks (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T)
    (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → (∀ e ∈ L.calls x s w, H e.1 = e.2) → ∀ c : ℕ → Coin, shows s x xO c →
      PartKindAt decode L chk x s xO c w

/-- The registration ties (`v`, the zero output rows, the levels' `coef`, `dirs` and top) are where a link can fail off a recorded collision. -/
def PartsLinksOr (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T)
    (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → (∀ e ∈ L.calls x s w, H e.1 = e.2) →
      (∀ c : ℕ → Coin, shows s x xO c → PartKindAt decode L chk x s xO c w) ∨ (findColl (L.calls x s w)).isSome ∨
        SetupColl H D xO (L.calls x s w)

theorem chk_of_kindAt {decode : Msg → Option Flock.Readable.Read} {L : Layout}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {x : Msg} {s : ℕ} {xO : Msg} {c : ℕ → Coin} {w : Flock.Wit}
    (hck : Flock.check (Flock.Readable.parseOf Flock.Readable.relation decode) xO w = true)
    (hK : PartKindAt decode L chk x s xO c w) {ms : List Msg} (hms : Agrees L x s w ms) : chk s x c ms := by
  rcases hK with hK | hK | hK | hK
  · obtain ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, hp, o, g, u, ho, hg, hu, hlink⟩ := hK
    obtain ⟨msch, r, a, cf, rest, idx, ss, hmsch, hin, hout⟩ := hlink ms hms
    exact hout (Flock.Readable.openOuts_of_check_any hck hx ht hpi htl ho hg hu hrec hp msch hmsch hin)
  · obtain ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, holo, S, ns, nv, hrc, hp, hsh, hh, hpart, o, g, u, ho, hg, hu,
      hlink⟩ := hK
    exact hlink ms hms (Flock.Readable.partCols_of_check_at hck hx ht hpi htl hrc hp hsh hh hpart ho hg hu
      (Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t htl w o g u))
      (Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t htl w o g u))
      (fun _ he => Flock.Readable.readElems_ge he) (fun _ he => Flock.Readable.readElems_ge he)
      (Flock.Readable.inCols_readElems ns nv _))
  · obtain ⟨p, hx, ht, hpi, t, htl, m, cl, j, sh, S, ns, nv, hrc, hp, hsh, hpart, o, g, u, ho, hg, hu, hlink⟩ := hK
    exact hlink ms hms (Flock.Readable.partCols_of_check_opening hck hx ht hpi htl hrc hp hsh hpart ho hg hu
      (Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t htl w o g u))
      (Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t htl w o g u))
      (fun _ he => Flock.Readable.readElems_ge he) (fun _ he => Flock.Readable.readElems_ge he)
      (Flock.Readable.inCols_readElems ns nv _))
  · obtain ⟨p, hx, ht, hpi, t, htl, sh, hrc, hp, o, g, u, ho, hg, hu, hlink⟩ := hK
    obtain ⟨wA, vA, τA, hin, hout⟩ := hlink ms hms
    exact hout (Flock.Readable.sparseCopyOuts_of_check hck hx ht hpi htl hrc hp ho hg hu wA vA τA hin)

theorem partChecks_of_partsLinks {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : PartsLinks H D DO decode L shape shows chk s) : PartChecks H D DO L shape shows chk s := by
  intro x st hst hf xO w hck htrue
  have hK := hR x st hst hf xO w hck htrue
  rw [hDO] at hck
  exact Or.inl fun c hc ms hms => chk_of_kindAt hck (hK c hc) hms

theorem partChecks_of_partsLinksOr {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : PartsLinksOr H D DO decode L shape shows chk s) : PartChecks H D DO L shape shows chk s := by
  intro x st hst hf xO w hck htrue
  rcases hR x st hst hf xO w hck htrue with hK | hK
  · rw [hDO] at hck
    exact Or.inl fun c hc ms hms => chk_of_kindAt hck (hK c hc) hms
  · exact Or.inr hK

theorem partsLinksOr_of_links {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {decode : Msg → Option Flock.Readable.Read} {L : Layout} {shape : Form → Prop}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ}
    (h : PartsLinks H D DO decode L shape shows chk s) : PartsLinksOr H D DO decode L shape shows chk s :=
  fun x st hst hf xO w hck htrue => Or.inl (h x st hst hf xO w hck htrue)

theorem partsLinks_of_reads {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {decode : Msg → Option Flock.Readable.Read} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} (hR : ResReads D decode L shape shows chk s) :
    PartsLinks H D DO decode L shape shows chk s := by
  intro x st hst hf xO w _ _ c hc
  obtain ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, S, ns, nv, hrc, hp, hsh, hh, hpart, o, g, u, ho, hg, hu, hlink⟩ :=
    hR x st hst hf xO c hc
  have e : { sh with holo := false } = sh := by
    have h0 := shape?_holo hsh
    cases sh
    simp only at h0
    subst h0
    rfl
  have hpart' : _root_.Flock.RecResiduals.part { sh with holo := false } j = .ok (S, ns, nv) := by
    rw [e]
    exact hpart
  exact Or.inr (Or.inl ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, false, S, ns, nv, hrc, hp, hsh, hh, hpart', o, g, u,
    ho, hg, hu, hlink w⟩)

theorem partsLinks_of_open {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {decode : Msg → Option Flock.Readable.Read} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} (hR : OpenLinks D decode L shape shows chk s) :
    PartsLinks H D DO decode L shape shows chk s := by
  intro x st hst hf xO w _ _ c hc
  obtain ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, hp, o, g, u, ho, hg, hu, hlink⟩ := hR x st hst hf xO c hc
  exact Or.inl ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, Or.inl hp, o, g, u, ho, hg, hu, hlink w⟩

theorem partChecks_holo {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : ∀ s, PartsLinksOr H D DO decode L Form.Holo shows chk s) : ∀ s, PartChecks H D DO L Form.Holo shows chk s :=
  fun s => partChecks_of_partsLinksOr hDO (hR s)

def LogsNil (D : Deploy Coin) (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
    (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin),
    shows s x xO c → ∀ ms, logs s x c ms = []

theorem partLogs_of_shown_nil {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {s : ℕ} (h : LogsNil D shape shows logs s) :
    PartLogs H D DO L shape shows logs s := fun x st hst hf xO _ _ _ => Or.inl fun c hc ms _ e he => by
  rw [h x st hst hf xO c hc ms] at he
  cases he

def LevelLogsAt (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)) (x : Msg) (s : ℕ) (xO : Msg) (c : ℕ → Coin)
    (w : Flock.Wit) : Prop :=
  LevelAt decode L (fun s x c ms => ∀ e ∈ logs s x c ms, e ∈ L.calls x s w) x s xO c w

def PartsLogLinks (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T)
    (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg))
    (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → (∀ e ∈ L.calls x s w, H e.1 = e.2) →
      (∀ c : ℕ → Coin, shows s x xO c → LevelLogsAt decode L logs x s xO c w) ∨ (findColl (L.calls x s w)).isSome ∨
        SetupColl H D xO (L.calls x s w)

theorem partLogs_of_logLinks {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : PartsLogLinks H D DO decode L shape shows logs s) : PartLogs H D DO L shape shows logs s := by
  intro x st hst hf xO w hck htrue
  rcases hR x st hst hf xO w hck htrue with hK | hK
  · rw [hDO] at hck
    exact Or.inl fun c hc ms hms =>
      chk_of_kindAt (chk := fun s x c ms => ∀ e ∈ logs s x c ms, e ∈ L.calls x s w) hck (Or.inl (hK c hc)) hms
  · exact Or.inr hK

theorem partsLogLinks_of_open {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {decode : Msg → Option Flock.Readable.Read} {L : Layout} {shape : Form → Prop}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {s : ℕ}
    (hR : OpenLogLinks D decode L shape shows logs s) : PartsLogLinks H D DO decode L shape shows logs s := by
  intro x st hst hf xO w _ _
  refine Or.inl fun c hc => ?_
  obtain ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, hp, o, g, u, ho, hg, hu, hlink⟩ := hR x st hst hf xO c hc
  exact ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, Or.inl hp, o, g, u, ho, hg, hu, hlink w⟩

theorem partLogs_holo {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (h : ∀ s, LogsNil D Form.Holo shows logs s ∨ PartsLogLinks H D DO decode L Form.Holo shows logs s) :
    ∀ s, PartLogs H D DO L Form.Holo shows logs s := fun s =>
  (h s).elim partLogs_of_shown_nil (partLogs_of_logLinks hDO)

theorem partOpens_of_none {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {opening : Opening}
    {cmt : Msg → ℕ → Msg → Msg} {s : ℕ}
    (h : ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ i, L.msgAt x s i = none) :
    PartOpens H D DO L shape shows opening cmt s := by
  intro x st hst hf xO w _ _
  exact Or.inl fun _ _ i _ m hm => by simp [Layout.held, h x st hst hf i] at hm

theorem partOpens_holo {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {opening : Opening} {cmt dig : Msg → ℕ → Msg → Msg}
    (hO : ∀ s, HeldOpens D DO L Form.Holo shows opening dig s)
    (hC : ∀ s, RegClimbs H D DO L Form.Holo shows cmt dig s) : ∀ s, PartOpens H D DO L Form.Holo shows opening cmt s :=
  fun s => partOpens_of_climbs (hO s) (hC s)

def HoloPart (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T)
    (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg))
    (s : ℕ) : Prop :=
  PartsLinksOr H D DO decode L Form.Holo shows chk s ∧
    (LogsNil D Form.Holo shows logs s ∨ PartsLogLinks H D DO decode L Form.Holo shows logs s)

theorem holoPart_of_open {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {decode : Msg → Option Flock.Readable.Read} {L : Layout} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {s : ℕ} (hC : OpenLinks D decode L Form.Holo shows chk s) (hL : OpenLogLinks D decode L Form.Holo shows logs s) :
    HoloPart H D DO decode L shows chk logs s :=
  ⟨partsLinksOr_of_links (partsLinks_of_open hC), Or.inr (partsLogLinks_of_open hL)⟩

theorem holoPart_of_nil {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {decode : Msg → Option Flock.Readable.Read} {L : Layout} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {s : ℕ} (hC : PartsLinksOr H D DO decode L Form.Holo shows chk s) (h0 : ∀ x c ms, logs s x c ms = []) :
    HoloPart H D DO decode L shows chk logs s :=
  ⟨hC, Or.inl fun x _ _ _ _ c _ ms => h0 x c ms⟩

theorem vstarAccepts_holo_of_links [Inhabited Coin] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {IV : Verifier Coin} {L : Layout} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {opening : Opening} {cmt dig : Msg → ℕ → Msg → Msg}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : ReadsThrough H D IV L.msgs L.calls) (hVL : VstarLen D)
    (hC : ∀ s, PartsLinksOr H D DO decode L Form.Holo shows chk s)
    (hO : ∀ s, HeldOpens D DO L Form.Holo shows opening dig s)
    (hRC : ∀ s, RegClimbs H D DO L Form.Holo shows cmt dig s)
    (hS : Staged H D I L Form.Holo shows cmt) (hD : Decomp H D IV Form.Holo chk) :
    VStarAcceptsR H D I DO Form.Holo :=
  vstarAccepts_holo hR hVL (partChecks_holo hDO hC) (partOpens_holo hO hRC) hS hD

theorem vstarParts_holo_of_links [Inhabited Coin] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {L : Layout} {opening : Opening} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {cmt dig : Msg → ℕ → Msg → Msg}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {pub : Msg → List (Msg × Msg)}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hP : ∀ s, HoloPart H D DO decode L shows chk logs s)
    (hO : ∀ s, HeldOpens D DO L Form.Holo shows opening dig s)
    (hRC : ∀ s, RegClimbs H D DO L Form.Holo shows cmt dig s)
    (hS : Staged H D I L Form.Holo shows cmt) (hD : Decomp H D I.V Form.Holo chk)
    (hLD : LogDecompPub H D I.V Form.Holo logs pub) (hPR : PubReg H D Form.Holo pub) (hH : HeldPub D I L Form.Holo) :
    VStarParts H D I DO L opening Form.Holo :=
  ⟨shows, chk, cmt, logs, pub, partChecks_holo hDO fun s => (hP s).1, partOpens_holo hO hRC,
    partLogs_holo hDO fun s => (hP s).2, hS, hD, hLD, hPR, hH⟩

theorem vstarPartsStaging_holo_of_links [Inhabited Coin] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {L : Layout} {opening : Opening} {decC : Msg → Msg → Coin}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    {cmt dig : Msg → ℕ → Msg → Msg} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {pub : Msg → List (Msg × Msg)} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hP : ∀ s, HoloPart H D DO decode L shows chk logs s)
    (hO : ∀ s, HeldOpens D DO L Form.Holo shows opening dig s)
    (hRC : ∀ s, RegClimbs H D DO L Form.Holo shows cmt dig s)
    (hV : VstarShows D I L Form.Holo decC shows cmt) (hD : Decomp H D I.V Form.Holo chk)
    (hLD : LogDecompPub H D I.V Form.Holo logs pub) (hPR : PubReg H D Form.Holo pub) (hH : HeldPub D I L Form.Holo) :
    VStarPartsStaging D I DO H L opening decC Form.Holo :=
  ⟨shows, chk, cmt, logs, pub, partChecks_holo hDO fun s => (hP s).1, partOpens_holo hO hRC,
    partLogs_holo hDO fun s => (hP s).2, hV, hD, hLD, hPR, hH⟩

end Facts

section Deployed

variable {W : Type} (H : Msg → Msg) (I : Inner FlockDeploy.Coin W) (ivScope : Msg → Bool) (b : RecDeploy.Brings)

theorem vstarAccepts_holo_deployRec {L : Layout} (hb : b.rd = RecDeploy.readersOf L)
    {shows : ℕ → Msg → Msg → (ℕ → FlockDeploy.Coin) → Prop} {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop}
    {opening : Opening} {cmt dig : Msg → ℕ → Msg → Msg}
    (hC : ∀ s, PartsLinksOr H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation)
      RecDeploy.readOfO L Form.Holo shows chk s)
    (hO : ∀ s, HeldOpens (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      Form.Holo shows opening dig s)
    (hRC : ∀ s, RegClimbs H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      Form.Holo shows cmt dig s)
    (hS : Staged H (RecDeploy.deployRec I.V ivScope b) I L Form.Holo shows cmt)
    (hD : Decomp H (RecDeploy.deployRec I.V ivScope b) I.V Form.Holo chk) :
    VStarAcceptsR H (RecDeploy.deployRec I.V ivScope b) I (RecDeploy.deployZkRec Flock.Readable.relation) Form.Holo :=
  vstarAccepts_holo_of_links rfl (RecDeploy.readsThrough_deployRec H I.V ivScope b L hb)
    (RecDeploy.vstarLen_deployRec I.V ivScope b) hC hO hRC hS hD

theorem vstarParts_holo_deployRec {L : Layout} {opening : Opening}
    {shows : ℕ → Msg → Msg → (ℕ → FlockDeploy.Coin) → Prop} {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop}
    {cmt dig : Msg → ℕ → Msg → Msg} {logs : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → List (Msg × Msg)}
    (hP : ∀ s, HoloPart H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation)
      RecDeploy.readOfO L shows chk logs s)
    (hO : ∀ s, HeldOpens (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      Form.Holo shows opening dig s)
    (hRC : ∀ s, RegClimbs H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      Form.Holo shows cmt dig s)
    (hS : Staged H (RecDeploy.deployRec I.V ivScope b) I L Form.Holo shows cmt)
    (hD : Decomp H (RecDeploy.deployRec I.V ivScope b) I.V Form.Holo chk) {pub : Msg → List (Msg × Msg)}
    (hLD : LogDecompPub H (RecDeploy.deployRec I.V ivScope b) I.V Form.Holo logs pub)
    (hPR : PubReg H (RecDeploy.deployRec I.V ivScope b) Form.Holo pub)
    (hH : HeldPub (RecDeploy.deployRec I.V ivScope b) I L Form.Holo) :
    VStarParts H (RecDeploy.deployRec I.V ivScope b) I (RecDeploy.deployZkRec Flock.Readable.relation) L opening
      Form.Holo :=
  vstarParts_holo_of_links rfl hP hO hRC hS hD hLD hPR hH

theorem vstarPartsStaging_holo_deployRec {L : Layout} {opening : Opening} {decC : Msg → Msg → FlockDeploy.Coin}
    {shows : ℕ → Msg → Msg → (ℕ → FlockDeploy.Coin) → Prop} {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop}
    {cmt dig : Msg → ℕ → Msg → Msg} {logs : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → List (Msg × Msg)}
    (hP : ∀ s, HoloPart H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation)
      RecDeploy.readOfO L shows chk logs s)
    (hO : ∀ s, HeldOpens (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      Form.Holo shows opening dig s)
    (hRC : ∀ s, RegClimbs H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      Form.Holo shows cmt dig s)
    (hV : VstarShows (RecDeploy.deployRec I.V ivScope b) I L Form.Holo decC shows cmt)
    (hD : Decomp H (RecDeploy.deployRec I.V ivScope b) I.V Form.Holo chk) {pub : Msg → List (Msg × Msg)}
    (hLD : LogDecompPub H (RecDeploy.deployRec I.V ivScope b) I.V Form.Holo logs pub)
    (hPR : PubReg H (RecDeploy.deployRec I.V ivScope b) Form.Holo pub)
    (hH : HeldPub (RecDeploy.deployRec I.V ivScope b) I L Form.Holo) :
    VStarPartsStaging (RecDeploy.deployRec I.V ivScope b) I (RecDeploy.deployZkRec Flock.Readable.relation) H L
      opening decC Form.Holo :=
  vstarPartsStaging_holo_of_links rfl hP hO hRC hV hD hLD hPR hH

end Deployed

end Claims.Recursion.VStar
