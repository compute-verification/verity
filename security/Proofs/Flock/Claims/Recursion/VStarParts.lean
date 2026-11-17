module

public import Proofs.Flock.Claims.Recursion.VStar
public import Proofs.Flock.Claims.Scope.Matches
public import Proofs.Flock.VBridge.ResUnit
public import Proofs.Flock.Claims.Scope.Tables
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
import Proofs.Flock.Claims.Recursion.Transport
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Claims.Scope.InScope
import Proofs.Flock.Soundness.Discharge.Hidden.Facts
import Proofs.Flock.Soundness.Lowering
import Proofs.Flock.Soundness.TmplUnitNet
import Std.Tactic.BVDecide.Normalize.BitVec
import Std.Tactic.BVDecide.Normalize.Prop

/-! V\*'s parts at the Claims relation: a unit copy's semantics at every template, and the algebra and rec-open parts' links. -/

@[expose] public section

namespace Flock.Readable

open FlockSoundness FlockSoundness.Discharge.Hm FlockVBridge.OfVStar

def copyCol (J c : ℕ) (w : Claims.Flock.Wit) (kLog o base top j : ℕ) : Bool :=
  if j ≤ top then w.hid.getD ((o * 2 ^ kLog + (base + j)) * J + c) false else false

theorem bit_toZ (b : Bool) : FlockSoundness.bit (toZ b) = b := by cases b <;> rfl

theorem laidRows_of_satisfies {st : Flock.Stmt} (L : FlockSoundness.Layout st) (hfl : st.c.tmpl = none)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {J c : ℕ} {w : Claims.Flock.Wit}
    (hz : (L.model regs).Satisfies (zOf J w c st.m)) {o g u : ℕ} (ho : o < 2 ^ (st.m - st.c.kLog))
    (hg : g < st.c.g) (hu : u < st.c.upv) {name : String} {l : Flock.HmNets.Laid}
    (hl : Flock.HmNets.netOf name l = .ok st.c.unit) :
    LaidRows l (st.c.unit.inWords * Flock.WORD)
      (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos) := by
  have hpl := L.placement regs hg
  have hs := rowsSat_of_satisfies hpl hz ⟨o, ho⟩
  have hs1 := FlockVBridge.sat_stack _ _ hu hs
  have hone := FlockVBridge.stack_one _ _ hu hs
  beta_reduce at hone
  rw [hpl.one_eq hz ⟨o, ho⟩] at hone
  have hl' : Flock.HmNets.netOf name l = .ok (FlockSoundness.lunit st.c) := by
    rw [FlockSoundness.lunit_flat hfl]; exact hl
  have hL := FlockVBridge.laidRows_ofNet L.unit_built hl' (L.lunitLog_le (L.slot_lt hg hu)) hs1
    (hone.trans (by simp [FlockSoundness.bit]))
  rw [L.unit_at.in_eq] at hL
  have hw : (unitRows st L.unit_built).w = st.c.unit.constPos + 1 := by
    show _ + _ + 1 = _
    rw [L.unitRows_nIn_nComp, FlockSoundness.lunit_flat hfl]
  have hw' : (Rows.ofNet L.unit_built).w = st.c.unit.constPos + 1 := hw
  convert hL using 1
  funext j
  unfold FlockVBridge.colVal copyCol
  split_ifs
  · have hj : st.c.unitRange.slot (g * st.c.upv + u) + j < 2 ^ (L.model regs).kLog := by
      have := L.hbM regs hg u hu ⟨j, by omega⟩
      rwa [FlockSoundness.upos_flat hfl] at this
    have e : L.col regs hg ((unitRows st L.unit_built).stackCol st.c.upv hu ⟨j, by omega⟩) = ⟨_, hj⟩ := by
      apply Fin.ext
      rw [L.col_stackCol regs hg hu, FlockSoundness.upos_flat hfl]
    exact (bit_toZ _).symm.trans
      (congrArg (fun i => FlockSoundness.bit ((L.model regs).block (zOf J w c st.m) ⟨o, ho⟩ i)) e.symm)
  · omega
  · omega
  · rfl

theorem partCols_of_satisfies {st : Flock.Stmt} (L : FlockSoundness.Layout st) (hfl : st.c.tmpl = none)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {J c : ℕ} {w : Claims.Flock.Wit}
    (hz : (L.model regs).Satisfies (zOf J w c st.m)) {o g u : ℕ} (ho : o < 2 ^ (st.m - st.c.kLog))
    (hg : g < st.c.g) (hu : u < st.c.upv) (hc : RecResiduals.check st.c = .ok ()) {m k cl j : ℕ}
    (hp : RecResiduals.params st.c.unit.name = some (m, k, cl, j)) {sh : Flock.Shape}
    (hsh : RecResiduals.shape? m k cl = some sh) (hh : RecResiduals.holoOf st.c = .ok false)
    {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} (hpart : RecResiduals.part sh j = .ok (S, ns, nv))
    (W vv : ℕ → Flock.F128) (hw0 : ∀ e, ns.sum ≤ e → W e = 0) (hv0 : ∀ e, nv ≤ e → vv e = 0)
    (hin : InCols (partRep ns nv) W vv
      (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos)) :
    PartCols S W vv (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos)
      (partBase S ns nv) := by
  obtain ⟨S', ns', nv', hpart', hn⟩ := check_part hc hp hsh hh
  rw [hpart] at hpart'
  cases hpart'
  have hL := laidRows_of_satisfies L hfl regs hz ho hg hu hn
  rw [inWords_unitLaid hn] at hL
  exact laid_part S ns nv (part_small hpart) _ hL W vv hw0 hv0 hin

theorem climb_of_satisfies {st : Flock.Stmt} (L : FlockSoundness.Layout st) (hfl : st.c.tmpl = none)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {J c : ℕ} {w : Claims.Flock.Wit}
    (hz : (L.model regs).Satisfies (zOf J w c st.m)) {o g u : ℕ} (ho : o < 2 ^ (st.m - st.c.kLog))
    (hg : g < st.c.g) (hu : u < st.c.upv) {v : ℕ → Bool}
    (hv : v = copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos)
    (ms : Flock.MerkleScheme) (hms : ∀ l r, ms.node l r = Flock.Sha512.hash (l ++ r))
    (hrec : Flock.RecOpen.check st.c = .ok ()) {LANES H : ℕ}
    (hp : Flock.RecOpen.params st.c.unit.name = some (LANES, H))
    (r a cf : Array Flock.F128) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray)
    (hr : r.size = LANES) (ha : a.size = 2) (hcf : cf.size = 2 * LANES) (hrest : rest.size = 96)
    (hss : ss.length = H) (hrow : FlockVBridge.ColsHold v 0 (Flock.MerkleScheme.rowBytes r))
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧ FlockVBridge.ColsHold v (128 * LANES + 512 * i) ss[i])
    (hacc : FlockVBridge.ColsHold v (128 * LANES + 1024 * max 1 ((H + 1) / 2))
      (Flock.MerkleScheme.rowBytes a ++ rest))
    (hcoef : FlockVBridge.ColsHold v (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024)
      (Flock.MerkleScheme.rowBytes cf))
    (hdir : ∀ h < H, v (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024 + 256 * LANES + h) = idx.testBit h) :
    FlockVBridge.ColsHold v
      (outStart ((Flock.RecOpen.recOpenNet LANES H (Flock.RecOpen.ports LANES H)).run Flock.HmNets.St.empty).2
        (Flock.RecOpen.wordPorts (FlockVBridge.unitIn LANES H)) (Flock.RecOpen.wordPorts 2048))
      ((Flock.climb ms (Flock.Sha512.hash (Flock.MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ++
        (Flock.MerkleScheme.rowBytes #[FlockVBridge.consTerm LANES r a cf 0, FlockVBridge.consTerm LANES r a cf 1] ++
          rest)) := by
  subst hv
  obtain ⟨hvalid, hl⟩ := Flock.RecOpen.check_ok hrec hp
  have hH : H ≤ 32 := by
    simp only [Flock.RecOpen.valid, Bool.and_eq_true, decide_eq_true_eq] at hvalid
    exact hvalid.2
  have hL := laidRows_of_satisfies L hfl regs hz ho hg hu hl
  rw [FlockVBridge.inWords_unit hl] at hL
  exact FlockVBridge.laid_unit ms hms LANES H hH _ hL r a cf rest idx ss hr ha hcf hrest hss hrow haux hacc hcoef hdir

theorem relS_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) :
    ∃ p, parseBytes decode x = some p ∧ Parses p ∧ Claims.Flock.RelS (toReadable relation p) w := by
  unfold Claims.Flock.check parseOf at h
  rcases hp : parseBytes decode x with _ | p
  · rw [hp] at h
    exact absurd h Bool.false_ne_true
  · rw [hp] at h
    have h' : decide (Claims.Flock.RelS (toReadable relation p) w) = true := h
    refine ⟨p, rfl, ?_, of_decide_eq_true h'⟩
    unfold parseBytes at hp
    obtain ⟨r, -, hr⟩ := Option.bind_eq_some_iff.1 hp
    exact parses_of_parse hr

theorem layout_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size) :
    ∃ (L : FlockSoundness.Layout p.sts[t]) (regs : List (Model.Region p.sts[t].c.kLog (p.sts[t].m - p.sts[t].c.kLog))),
      (L.model regs).Satisfies (zOf p.sts.size w t p.sts[t].m) := by
  obtain ⟨p', hx', hp', hR⟩ := relS_of_check h
  rw [hx] at hx'
  cases hx'
  obtain ⟨L, hz⟩ := (relS_layout hp' ht hpi w).1 hR t htl
  exact ⟨L, _, hz⟩

theorem flat_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    (hrec : Flock.HmOut.recUnit p.sts[t].c.unit.name = true) : p.sts[t].c.tmpl = none := by
  obtain ⟨p', hx', hp', -⟩ := relS_of_check h
  rw [hx] at hx'
  cases hx'
  obtain ⟨hc, -⟩ := tableAt (tablesOf_of_parses hp') ht hpi htl
  unfold ProvedScope.circuitOf at hc
  split at hc
  · rcases FlockSoundness.Discharge.Hidden.committed_of_commit hc with hc | hc
    · exact (FlockSoundness.parse_facts (FlockSoundness.Discharge.Hidden.parse_untyped ht hc).1).tmpl
    · rw [FlockSoundness.TmplValue.parseTmpl_notRec hc] at hrec
      cases hrec
  · exact (FlockSoundness.parse_facts hc).tmpl

theorem recUnit_open {n : String} {L H : ℕ}
    (h : Flock.RecOpen.params n = some (L, H) ∨ Flock.RecOpen.stackedParams n = some (L, H)) :
    Flock.HmOut.recUnit n = true := by
  rcases h with h | h <;> simp [Flock.HmOut.recUnit, h]

theorem recUnit_residuals {n : String} {q : ℕ × ℕ × ℕ × ℕ} (h : Flock.RecResiduals.params n = some q) :
    Flock.HmOut.recUnit n = true := by
  simp [Flock.HmOut.recUnit, h]

theorem recUnit_sparse {n : String} {sh : Flock.RecSparse.Shape} (h : Flock.RecSparse.params n = some sh) :
    Flock.HmOut.recUnit n = true := by
  simp [Flock.HmOut.recUnit, h]

theorem recUnit_opening {n : String} {q : ℕ × ℕ × ℕ} (h : Flock.RecOpening.params n = some q) :
    Flock.HmOut.recUnit n = true := by
  simp [Flock.HmOut.recUnit, h]

theorem partCols_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    (hc : RecResiduals.check p.sts[t].c = .ok ()) {m k cl j : ℕ}
    (hp : RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, j)) {sh : Flock.Shape}
    (hsh : RecResiduals.shape? m k cl = some sh) (hh : RecResiduals.holoOf p.sts[t].c = .ok false)
    {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} (hpart : RecResiduals.part sh j = .ok (S, ns, nv))
    {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    (W vv : ℕ → Flock.F128) (hw0 : ∀ e, ns.sum ≤ e → W e = 0) (hv0 : ∀ e, nv ≤ e → vv e = 0)
    (hin : InCols (partRep ns nv) W vv (copyCol p.sts.size t w p.sts[t].c.kLog o
      (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos)) :
    PartCols S W vv (copyCol p.sts.size t w p.sts[t].c.kLog o
      (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos) (partBase S ns nv) := by
  obtain ⟨L, regs, hz⟩ := layout_of_check h hx ht hpi htl
  exact partCols_of_satisfies L (flat_of_check h hx ht hpi htl (recUnit_residuals hp)) regs hz ho hg hu hc hp hsh hh hpart W vv hw0 hv0 hin

theorem climb_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    {v : ℕ → Bool} (hv : v = copyCol p.sts.size t w p.sts[t].c.kLog o
      (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos)
    (ms : Flock.MerkleScheme) (hms : ∀ l r, ms.node l r = Flock.Sha512.hash (l ++ r))
    (hrec : Flock.RecOpen.check p.sts[t].c = .ok ()) {LANES H : ℕ}
    (hp : Flock.RecOpen.params p.sts[t].c.unit.name = some (LANES, H))
    (r a cf : Array Flock.F128) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray)
    (hr : r.size = LANES) (ha : a.size = 2) (hcf : cf.size = 2 * LANES) (hrest : rest.size = 96)
    (hss : ss.length = H) (hrow : FlockVBridge.ColsHold v 0 (Flock.MerkleScheme.rowBytes r))
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧ FlockVBridge.ColsHold v (128 * LANES + 512 * i) ss[i])
    (hacc : FlockVBridge.ColsHold v (128 * LANES + 1024 * max 1 ((H + 1) / 2))
      (Flock.MerkleScheme.rowBytes a ++ rest))
    (hcoef : FlockVBridge.ColsHold v (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024)
      (Flock.MerkleScheme.rowBytes cf))
    (hdir : ∀ h < H, v (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024 + 256 * LANES + h) = idx.testBit h) :
    FlockVBridge.ColsHold v
      (outStart ((Flock.RecOpen.recOpenNet LANES H (Flock.RecOpen.ports LANES H)).run Flock.HmNets.St.empty).2
        (Flock.RecOpen.wordPorts (FlockVBridge.unitIn LANES H)) (Flock.RecOpen.wordPorts 2048))
      ((Flock.climb ms (Flock.Sha512.hash (Flock.MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ++
        (Flock.MerkleScheme.rowBytes #[FlockVBridge.consTerm LANES r a cf 0, FlockVBridge.consTerm LANES r a cf 1] ++
          rest)) := by
  obtain ⟨L, regs, hz⟩ := layout_of_check h hx ht hpi htl
  exact climb_of_satisfies L (flat_of_check h hx ht hpi htl (recUnit_open (.inl hp))) regs hz ho hg hu hv ms hms hrec hp r a cf rest idx ss hr ha hcf hrest hss hrow haux hacc
    hcoef hdir

def copyOf (p : Parsed) (t : ℕ) (ht : t < p.sts.size) (w : Claims.Flock.Wit) (o g u : ℕ) : ℕ → Bool :=
  copyCol p.sts.size t w p.sts[t].c.kLog o (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u))
    p.sts[t].c.unit.constPos

def readElems (col : ℕ → ℕ) (n : ℕ) (v : ℕ → Bool) (e : ℕ) : Flock.F128 :=
  if e < n then FlockVBridge.ofBits fun t => v (col (128 * e + t)) else 0

theorem wbit_readElems {col : ℕ → ℕ} {n : ℕ} {v : ℕ → Bool} {q : ℕ} (hq : q < 128 * n) :
    FlockVBridge.wbit (readElems col n v) q = v (col q) := by
  have hn : q / 128 < n := by omega
  unfold FlockVBridge.wbit readElems
  simp only [hn, ↓reduceIte]
  rw [FlockVBridge.bitOf_ofBits _ (Nat.mod_lt _ (by norm_num)), Nat.div_add_mod]

theorem readElems_ge {col : ℕ → ℕ} {n : ℕ} {v : ℕ → Bool} {e : ℕ} (he : n ≤ e) : readElems col n v e = 0 := by
  simp [readElems, show ¬e < n by omega]

def rowsRead (ns : List ℕ) (nv : ℕ) (v : ℕ → Bool) : ℕ → Flock.F128 :=
  readElems (FlockVBridge.OfVStar.rowCol (partRep ns nv).ports 0) ns.sum v

def vRead (ns : List ℕ) (nv : ℕ) (v : ℕ → Bool) : ℕ → Flock.F128 :=
  readElems (FlockVBridge.OfVStar.vBase (partRep ns nv) + ·) nv v

theorem inCols_readElems (ns : List ℕ) (nv : ℕ) (v : ℕ → Bool) :
    InCols (partRep ns nv) (rowsRead ns nv v) (vRead ns nv v) v :=
  ⟨fun _ hq => (wbit_readElems hq).symm, fun _ hq => (wbit_readElems hq).symm⟩

def OpenIns (v : ℕ → Bool) (LANES H : ℕ) (r a cf : Array Flock.F128) (rest : ByteArray) (idx : ℕ)
    (ss : List ByteArray) : Prop :=
  r.size = LANES ∧ a.size = 2 ∧ cf.size = 2 * LANES ∧ rest.size = 96 ∧ ss.length = H ∧
    FlockVBridge.ColsHold v 0 (Flock.MerkleScheme.rowBytes r) ∧
    (∀ i (hi : i < ss.length), ss[i].size = 64 ∧ FlockVBridge.ColsHold v (128 * LANES + 512 * i) ss[i]) ∧
    FlockVBridge.ColsHold v (128 * LANES + 1024 * max 1 ((H + 1) / 2)) (Flock.MerkleScheme.rowBytes a ++ rest) ∧
    FlockVBridge.ColsHold v (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024) (Flock.MerkleScheme.rowBytes cf) ∧
    ∀ h < H, v (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024 + 256 * LANES + h) = idx.testBit h

def OpenOuts (v : ℕ → Bool) (ms : Flock.MerkleScheme) (LANES H : ℕ) (r a cf : Array Flock.F128) (rest : ByteArray)
    (idx : ℕ) (ss : List ByteArray) : Prop :=
  FlockVBridge.ColsHold v
    (outStart ((Flock.RecOpen.recOpenNet LANES H (Flock.RecOpen.ports LANES H)).run Flock.HmNets.St.empty).2
      (Flock.RecOpen.wordPorts (FlockVBridge.unitIn LANES H)) (Flock.RecOpen.wordPorts 2048))
    ((Flock.climb ms (Flock.Sha512.hash (Flock.MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ++
      (Flock.MerkleScheme.rowBytes #[FlockVBridge.consTerm LANES r a cf 0, FlockVBridge.consTerm LANES r a cf 1] ++
        rest))

end Flock.Readable

namespace Claims.Recursion.VStar

open FlockVBridge.OfVStar

section Facts

variable {Coin Col T : Type}

/-- A code fact about V\*'s staging, the layout and the inner check at part `s`, none of it the circuit's semantics. -/
def ResLinks (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin), shows s x xO c →
    ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
      ∃ t, ∃ ht : t < p.sts.size, ∃ (m k cl j : ℕ) (sh : _root_.Flock.Shape) (S : _root_.Flock.RecOpen.GfS)
        (ns : List ℕ) (nv : ℕ),
        _root_.Flock.RecResiduals.check p.sts[t].c = .ok () ∧
        _root_.Flock.RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, j) ∧
        _root_.Flock.RecResiduals.shape? m k cl = some sh ∧ _root_.Flock.RecResiduals.holoOf p.sts[t].c = .ok false ∧
        _root_.Flock.RecResiduals.part sh j = .ok (S, ns, nv) ∧
        ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
          ∀ (w : Flock.Wit) (ms : List Msg), (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) →
            ∃ W vv : ℕ → _root_.Flock.F128, (∀ e, ns.sum ≤ e → W e = 0) ∧ (∀ e, nv ≤ e → vv e = 0) ∧
              InCols (partRep ns nv) W vv (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos) ∧
              (PartCols S W vv (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos)
                (partBase S ns nv) → chk s x c ms)

theorem partChecks_of_res {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : ResLinks D decode L shape shows chk s) : PartChecks H D DO L shape shows chk s := by
  intro x st hst hf xO w hck _
  refine Or.inl fun c hc ms hms => ?_
  obtain ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, S, ns, nv, hrc, hp, hsh, hh, hpart, o, g, u, ho, hg, hu, hlink⟩ :=
    hR x st hst hf xO c hc
  obtain ⟨W, vv, hw0, hv0, hin, hout⟩ := hlink w ms hms
  rw [hDO] at hck
  exact hout (Flock.Readable.partCols_of_check hck hx ht hpi htl hrc hp hsh hh hpart ho hg hu W vv hw0 hv0 hin)

/-- What is left of the input tie sits in the output tie's premise: S's residuals at the elements the copy reads give the inner check once those are the held messages' and the coins' coefficients. -/
def ResReads (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin), shows s x xO c →
    ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
      ∃ t, ∃ ht : t < p.sts.size, ∃ (m k cl j : ℕ) (sh : _root_.Flock.Shape) (S : _root_.Flock.RecOpen.GfS)
        (ns : List ℕ) (nv : ℕ),
        _root_.Flock.RecResiduals.check p.sts[t].c = .ok () ∧
        _root_.Flock.RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, j) ∧
        _root_.Flock.RecResiduals.shape? m k cl = some sh ∧ _root_.Flock.RecResiduals.holoOf p.sts[t].c = .ok false ∧
        _root_.Flock.RecResiduals.part sh j = .ok (S, ns, nv) ∧
        ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
          ∀ (w : Flock.Wit) (ms : List Msg), (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) →
            PartCols S (Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t ht w o g u))
              (Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t ht w o g u))
              (Flock.Readable.copyOf p t ht w o g u) (partBase S ns nv) → chk s x c ms

theorem resLinks_of_reads {D : Deploy Coin} {decode : Msg → Option Flock.Readable.Read} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} (hR : ResReads D decode L shape shows chk s) :
    ResLinks D decode L shape shows chk s := by
  intro x st hst hf xO c hc
  obtain ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, S, ns, nv, hrc, hp, hsh, hh, hpart, o, g, u, ho, hg, hu, hlink⟩ :=
    hR x st hst hf xO c hc
  refine ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, S, ns, nv, hrc, hp, hsh, hh, hpart, o, g, u, ho, hg, hu,
    fun w ms hms => ?_⟩
  exact ⟨Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t htl w o g u),
    Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t htl w o g u),
    fun _ he => Flock.Readable.readElems_ge he, fun _ he => Flock.Readable.readElems_ge he,
    Flock.Readable.inCols_readElems ns nv _, hlink w ms hms⟩

theorem partChecks_of_reads {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : ResReads D decode L shape shows chk s) : PartChecks H D DO L shape shows chk s :=
  partChecks_of_res hDO (resLinks_of_reads hR)

/-- A code fact about V\*'s staging, the layout and the inner check at part `s`, none of it the circuit's semantics. -/
def OpenLinks (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin), shows s x xO c →
    ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
      ∃ t, ∃ ht : t < p.sts.size, ∃ LANES Ht : ℕ,
        _root_.Flock.RecOpen.check p.sts[t].c = .ok () ∧
        _root_.Flock.RecOpen.params p.sts[t].c.unit.name = some (LANES, Ht) ∧
        ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
          ∀ (w : Flock.Wit) (ms : List Msg), (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) →
            ∃ (msch : _root_.Flock.MerkleScheme) (r a cf : Array _root_.Flock.F128) (rest : ByteArray) (idx : ℕ)
              (ss : List ByteArray), (∀ l r, msch.node l r = _root_.Flock.Sha512.hash (l ++ r)) ∧
              Flock.Readable.OpenIns (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos) LANES Ht r a cf rest
                idx ss ∧
              (Flock.Readable.OpenOuts (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos) msch LANES Ht r a cf
                rest idx ss → chk s x c ms)

theorem partChecks_of_open {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : OpenLinks D decode L shape shows chk s) : PartChecks H D DO L shape shows chk s := by
  intro x st hst hf xO w hck _
  refine Or.inl fun c hc ms hms => ?_
  obtain ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, hp, o, g, u, ho, hg, hu, hlink⟩ := hR x st hst hf xO c hc
  obtain ⟨msch, r, a, cf, rest, idx, ss, hmsch, ⟨hr, ha, hcf, hrest, hss, hrow, haux, hacc, hcoef, hdir⟩, hout⟩ :=
    hlink w ms hms
  rw [hDO] at hck
  exact hout (Flock.Readable.climb_of_check hck hx ht hpi htl ho hg hu rfl msch hmsch hrec hp r a cf rest idx ss hr
    ha hcf hrest hss hrow haux hacc hcoef hdir)

theorem partLogs_of_nil {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {s : ℕ} (h : ∀ x c ms, logs s x c ms = []) :
    PartLogs H D DO L shape shows logs s := fun _ _ _ _ _ _ _ _ => Or.inl fun _ _ _ _ e he => by
  rw [h] at he
  cases he

def OpenLogLinks (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg))
    (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin), shows s x xO c →
    ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
      ∃ t, ∃ ht : t < p.sts.size, ∃ LANES Ht : ℕ,
        _root_.Flock.RecOpen.check p.sts[t].c = .ok () ∧
        _root_.Flock.RecOpen.params p.sts[t].c.unit.name = some (LANES, Ht) ∧
        ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
          ∀ (w : Flock.Wit) (ms : List Msg), (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) →
            ∃ (msch : _root_.Flock.MerkleScheme) (r a cf : Array _root_.Flock.F128) (rest : ByteArray) (idx : ℕ)
              (ss : List ByteArray), (∀ l r, msch.node l r = _root_.Flock.Sha512.hash (l ++ r)) ∧
              Flock.Readable.OpenIns (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos) LANES Ht r a cf rest
                idx ss ∧
              (Flock.Readable.OpenOuts (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos) msch LANES Ht r a cf
                rest idx ss → ∀ e ∈ logs s x c ms, e ∈ L.calls x s w)

theorem partLogs_of_open {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : OpenLogLinks D decode L shape shows logs s) : PartLogs H D DO L shape shows logs s := by
  intro x st hst hf xO w hck _
  refine Or.inl fun c hc ms hms => ?_
  obtain ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, hp, o, g, u, ho, hg, hu, hlink⟩ := hR x st hst hf xO c hc
  obtain ⟨msch, r, a, cf, rest, idx, ss, hmsch, ⟨hr, ha, hcf, hrest, hss, hrow, haux, hacc, hcoef, hdir⟩, hout⟩ :=
    hlink w ms hms
  rw [hDO] at hck
  exact hout (Flock.Readable.climb_of_check hck hx ht hpi htl ho hg hu rfl msch hmsch hrec hp r a cf rest idx ss hr
    ha hcf hrest hss hrow haux hacc hcoef hdir)

end Facts

def chainH (H : Msg → Msg) : List (Msg → Msg) → Msg → Msg
  | [], a => a
  | f :: fs, a => chainH H fs (H (f a))

def chainIns (H : Msg → Msg) : List (Msg → Msg) → Msg → List Msg
  | [], _ => []
  | f :: fs, a => f a :: chainIns H fs (H (f a))

def StepsApart (fs gs : List (Msg → Msg)) : Prop :=
  List.Forall₂ (fun f g => ∀ u v, f u = g v → u = v) fs gs

theorem chain_coll (H : Msg → Msg) {fs gs : List (Msg → Msg)} (h : StepsApart fs gs) :
    ∀ {a b : Msg}, a ≠ b → chainH H fs a = chainH H gs b →
      ∃ u ∈ chainIns H fs a, ∃ v ∈ chainIns H gs b, u ≠ v ∧ H u = H v := by
  induction h with
  | nil => intro a b hab heq; exact absurd heq hab
  | @cons f g fs gs hfg _ ih =>
    intro a b hab heq
    simp only [chainH] at heq
    by_cases hH : H (f a) = H (g b)
    · exact ⟨f a, by simp [chainIns], g b, by simp [chainIns], fun e => hab (hfg a b e), hH⟩
    · obtain ⟨u, hu, v, hv, huv⟩ := ih hH heq
      exact ⟨u, by simp [chainIns, hu], v, by simp [chainIns, hv], huv⟩

theorem coll_of_chains {Coin : Type} {H : Msg → Msg} {D : Deploy Coin} {xO : Msg} {cs : List (Msg × Msg)}
    (htrue : ∀ e ∈ cs, H e.1 = e.2) {fs gs : List (Msg → Msg)} {a b : Msg}
    (hin : ∀ u ∈ chainIns H fs a ++ chainIns H gs b, u ∈ D.setup xO ∨ (u, H u) ∈ cs)
    (h : ∃ u ∈ chainIns H fs a, ∃ v ∈ chainIns H gs b, u ≠ v ∧ H u = H v) :
    (findColl cs).isSome ∨ SetupColl H D xO cs := by
  obtain ⟨u, hu, v, hv, hne, hH⟩ := h
  have hsu : ∀ {y}, y ∈ D.setup xO → (y, H y) ∈ setupCalls H D [xO] := fun hy => by
    simp only [setupCalls, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.mem_map]
    exact ⟨_, hy, rfl⟩
  rcases hin u (List.mem_append_left _ hu) with hu' | hu'
  · refine Or.inr (findCollIn_isSome (e := (u, H u)) (e' := (v, H v)) (hsu hu') ?_ (fun e => hne e.symm) hH.symm)
    rcases hin v (List.mem_append_right _ hv) with hv' | hv'
    · exact List.mem_append_left _ (hsu hv')
    · exact List.mem_append_right _ hv'
  · rcases hin v (List.mem_append_right _ hv) with hv' | hv'
    · exact Or.inr (findCollIn_isSome (e := (v, H v)) (e' := (u, H u)) (hsu hv') (List.mem_append_right _ hu') hne hH)
    · exact Or.inl (findColl_isSome_mem H cs htrue u v (List.mem_map.2 ⟨_, hu', rfl⟩) (List.mem_map.2 ⟨_, hv', rfl⟩)
        ⟨hne, hH⟩)

def HeldOpens {Coin Col T : Type} (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (opening : Opening) (dig : Msg → ℕ → Msg → Msg) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → ∀ c, shows s x xO c → ∀ i < L.rounds x, ∀ m, L.held x s w i = some m →
      ∃ cs, cs ⊆ L.calls x s w ∧ opening.opens (dig x i xO) m cs

def RegClimbs {Coin Col T : Type} (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : Layout)
    (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt dig : Msg → ℕ → Msg → Msg) (s : ℕ) :
    Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → ∀ c, shows s x xO c → ∀ i < L.rounds x, ∀ m, L.held x s w i = some m →
      ∃ fs gs : List (Msg → Msg), StepsApart fs gs ∧ chainH H fs (dig x i xO) = chainH H gs (cmt x i xO) ∧
        ∀ u ∈ chainIns H fs (dig x i xO) ++ chainIns H gs (cmt x i xO), u ∈ D.setup xO ∨ (u, H u) ∈ L.calls x s w

theorem partOpens_of_climbs {Coin Col T : Type} {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {opening : Opening}
    {cmt dig : Msg → ℕ → Msg → Msg} {s : ℕ} (hO : HeldOpens D DO L shape shows opening dig s)
    (hC : RegClimbs H D DO L shape shows cmt dig s) : PartOpens H D DO L shape shows opening cmt s := by
  intro x st hst hf xO w hck htrue
  by_cases hcoll : (findColl (L.calls x s w)).isSome ∨ SetupColl H D xO (L.calls x s w)
  · exact Or.inr hcoll
  refine Or.inl fun c hc i hi m hm => ?_
  obtain ⟨cs, hcs, ho⟩ := hO x st hst hf xO w hck c hc i hi m hm
  by_cases hd : dig x i xO = cmt x i xO
  · exact ⟨cs, hcs, hd ▸ ho⟩
  · obtain ⟨fs, gs, hap, heq, hin⟩ := hC x st hst hf xO w hck c hc i hi m hm
    exact absurd (coll_of_chains htrue hin (chain_coll H hap hd heq)) hcoll

end Claims.Recursion.VStar
