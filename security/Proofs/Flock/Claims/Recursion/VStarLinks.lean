module

public import Proofs.Flock.Claims.Recursion.VStarPartsHolo
public import Proofs.Flock.Claims.Recursion.VStarPartsSparse
public import Proofs.Flock.Claims.Recursion.VStarCross
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
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Recursion.DeployMargin
import Proofs.Flock.Claims.Recursion.DeployScope
import Proofs.Flock.Claims.Recursion.DeploySessions
import Proofs.Flock.Claims.Scope.InScope
import Proofs.Flock.Claims.Scope.Setup
import Proofs.Flock.Recursive.VStarCheck
import Proofs.Flock.Soundness.Discharge.Aliased.Rows
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.VBridge.OpeningUnit
import Proofs.Flock.VBridge.Pinned
import Proofs.Flock.VBridge.Sparse.Unit
import Std.Tactic.BVDecide.Normalize.BitVec
import Std.Tactic.BVDecide.Normalize.Prop

/-! V\*'s part links (the staging, the input tie and the output tie), stated once for every kind of part and every shape, with what is the circuit's own proved. -/

@[expose] public section

namespace FlockVBridge

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm

def climbCalls (acc : ByteArray) (idx : ℕ) : List ByteArray → List (ByteArray × ByteArray)
  | [] => []
  | sb :: ss =>
    let i := if idx % 2 = 0 then acc ++ sb else sb ++ acc
    (i, Sha512.hash i) :: climbCalls (Sha512.hash i) (idx / 2) ss

def rowCalls (rb : ByteArray) (idx : ℕ) (ss : List ByteArray) : List (ByteArray × ByteArray) :=
  (rb, Sha512.hash rb) :: climbCalls (Sha512.hash rb) idx ss

def CallsAt (z : Var → Bool) (fs : List (List Lin × List Lin)) (es : List (ByteArray × ByteArray)) : Prop :=
  List.Forall₂ (fun f e => Carries z f.1 e.1 ∧ Carries z f.2 e.2) fs es

def climbFromC (d : List Lin) : ℕ → List Lin → List (List Lin) → M (List Lin × List (List Lin × List Lin))
  | _, cur, [] => pure (cur, [])
  | h, cur, s :: rest => do
    let lr ← swap (d.getD h Lin.zero) cur s
    let cur' ← shaFrom (preCv [] 0) (lr.1 ++ lr.2) 0
    let r ← climbFromC d (h + 1) cur' rest
    pure (r.1, (lr.1 ++ lr.2, cur') :: r.2)

def climbRowC (H : ℕ) (row aux d : List Lin) : M (List Lin × List (List Lin × List Lin)) := do
  let leaf ← shaFrom (preCv [] 0) row 0
  let r ← climbFromC d 0 leaf (sibForms aux H)
  pure (r.1 ++ List.replicate 512 Lin.zero, (row, leaf) :: r.2)

def recOpenC (LANES H : ℕ) (row aux acc coef dirs : List Lin) :
    M ((List Lin × List Lin) × List (List Lin × List Lin)) := do
  let t ← residualForms (consStructure LANES) (row ++ acc.take 256) coef
  let out ← climbRowC H row aux (dirs.take 32)
  pure ((out.1, t ++ acc.drop 256), out.2)

def recOpenNetC (LANES H : ℕ) (p : Ports) : M ((List Lin × List Lin) × List (List Lin × List Lin)) := do
  let row ← input p.row
  let aux ← input p.aux
  let acc ← input p.acc
  let coef ← input p.coef
  let dirs ← input p.dirs
  recOpenC LANES H row aux acc coef dirs

def recOpenCalls (LANES H : ℕ) (p : Ports) : List (List Lin × List Lin) :=
  ((recOpenNetC LANES H p).run St.empty).1.2

theorem climbFromC_run (d : List Lin) : ∀ (h : ℕ) (cur : List Lin) (sibs : List (List Lin)) (s : St),
    ((climbFromC d h cur sibs).run s).1.1 = ((climbFrom d h cur sibs).run s).1 ∧
      ((climbFromC d h cur sibs).run s).2 = ((climbFrom d h cur sibs).run s).2
  | _, _, [], _ => ⟨rfl, rfl⟩
  | h, cur, sb :: rest, s => by
    rw [climbFromC, climbFrom, run_bind, run_bind, run_bind, run_bind, run_bind, run_pure]
    generalize (swap (d.getD h Lin.zero) cur sb).run s = p1
    generalize (shaFrom (preCv [] 0) (p1.1.1 ++ p1.1.2) 0).run p1.2 = p2
    exact climbFromC_run d (h + 1) p2.1 rest p2.2

theorem climbRowC_run (H : ℕ) (row aux d : List Lin) (s : St) :
    ((climbRowC H row aux d).run s).1.1 = ((climbRow H row aux d).run s).1 ∧
      ((climbRowC H row aux d).run s).2 = ((climbRow H row aux d).run s).2 := by
  rw [climbRowC, climbRow, run_bind, run_bind, run_bind, run_bind, run_pure, run_pure]
  generalize (shaFrom (preCv [] 0) row 0).run s = p1
  obtain ⟨h1, h2⟩ := climbFromC_run d 0 p1.1 (sibForms aux H) p1.2
  exact ⟨by rw [h1], h2⟩

theorem recOpenC_run (LANES H : ℕ) (row aux acc coef dirs : List Lin) (s : St) :
    ((recOpenC LANES H row aux acc coef dirs).run s).1.1 = ((recOpen LANES H row aux acc coef dirs).run s).1 ∧
      ((recOpenC LANES H row aux acc coef dirs).run s).2 = ((recOpen LANES H row aux acc coef dirs).run s).2 := by
  rw [recOpenC, recOpen, run_bind, run_bind, run_bind, run_bind, run_pure, run_pure]
  generalize (residualForms (consStructure LANES) (row ++ acc.take 256) coef).run s = p1
  obtain ⟨h1, h2⟩ := climbRowC_run H row aux (dirs.take 32) p1.2
  exact ⟨by rw [h1], h2⟩

theorem recOpenNetC_run (LANES H : ℕ) (p : Ports) : (recOpenNetC LANES H p).run St.empty =
    (recOpenC LANES H (vars 0 p.row) (vars p.row p.aux) (vars (p.row + p.aux) p.acc)
      (vars (p.row + p.aux + p.acc) p.coef) (vars (p.row + p.aux + p.acc + p.coef) p.dirs)).run
      ⟨p.row + p.aux + p.acc + p.coef + p.dirs, []⟩ := by
  have h := run_inputs5 p.row p.aux p.acc p.coef p.dirs (recOpenC LANES H) ⟨0, []⟩
  simp only [Nat.zero_add] at h
  exact h

theorem recOpenNetC_state (LANES H : ℕ) (p : Ports) :
    ((recOpenNetC LANES H p).run St.empty).2 = ((recOpenNet LANES H p).run St.empty).2 := by
  rw [recOpenNetC_run, recOpenNet_run]
  exact (recOpenC_run LANES H _ _ _ _ _ _).2

theorem sound_climbFromC (d : List Lin) : ∀ (h : ℕ) (cur : List Lin) (sibs : List (List Lin)),
    Sound (climbFromC d h cur sibs) fun z oc => ∀ (acc : ByteArray) (idx : ℕ) (ss : List ByteArray),
      acc.size = 64 → Carries z cur acc → List.Forall₂ (fun w b => b.size = 64 ∧ Carries z w b) sibs ss →
      (∀ j < sibs.length, ev z (d.getD (h + j) Lin.zero) = idx.testBit j) → CallsAt z oc.2 (climbCalls acc idx ss)
  | _, cur, [] => Sound.pure' fun z acc idx ss _ _ hss _ => by
      cases hss
      exact List.Forall₂.nil
  | h, cur, s :: rest => by
      unfold climbFromC
      refine Sound.bind' (sound_swap _ cur s) fun lr => Sound.bind' (sound_shaIv _) fun cur' =>
        Sound.bind' (sound_climbFromC d (h + 1) cur' rest) fun r =>
          Sound.pure' fun z hih hsha hsw acc idx ss hacc hcur hss hdir => ?_
      obtain _ | ⟨⟨hsb, hs⟩, hss⟩ := hss
      rename_i sb ss
      obtain ⟨h1, h2⟩ := hsw acc sb hcur hs (by rw [hacc, hsb])
      have hd : ev z (d.getD h Lin.zero) = idx.testBit 0 := hdir 0 (by simp)
      have hin : (if ev z (d.getD h Lin.zero) then sb else acc) ++ (if ev z (d.getD h Lin.zero) then acc else sb) =
          (if idx % 2 = 0 then acc ++ sb else sb ++ acc) := by
        rw [hd, Nat.testBit_zero]
        rcases Nat.mod_two_eq_zero_or_one idx with hp | hp <;> simp [hp]
      have hpair := carries_append h1 h2
      rw [hin] at hpair
      have hcur' := hsha _ hpair
      exact List.Forall₂.cons ⟨hpair, hcur'⟩ (hih _ (idx / 2) ss (Sha512.hash_size _) hcur' hss fun j hj => by
        rw [Nat.testBit_div_two, ← hdir (j + 1) (by simp; omega), show h + 1 + j = h + (j + 1) by omega])

theorem sound_climbRowC (H : ℕ) (row aux d : List Lin) : Sound (climbRowC H row aux d) fun z o =>
    ∀ (rb : ByteArray) (idx : ℕ) (ss : List ByteArray), Carries z row rb →
      List.Forall₂ (fun w b => b.size = 64 ∧ Carries z w b) (sibForms aux H) ss →
      (∀ h < H, ev z (d.getD h Lin.zero) = idx.testBit h) → CallsAt z o.2 (rowCalls rb idx ss) := by
  unfold climbRowC
  refine Sound.bind' (sound_shaIv row) fun leaf => Sound.bind' (sound_climbFromC d 0 leaf _) fun r =>
    Sound.pure' fun z hr hleaf rb idx ss hrow hss hdir => ?_
  refine List.Forall₂.cons ⟨hrow, hleaf rb hrow⟩ (hr _ idx ss (Sha512.hash_size _) (hleaf rb hrow) hss fun j hj => ?_)
  rw [Nat.zero_add]
  exact hdir j (by simpa [sibForms] using hj)

theorem sound_recOpenC (LANES H : ℕ) (hH : H ≤ 32) (row aux acc coef dirs : List Lin) :
    Sound (recOpenC LANES H row aux acc coef dirs) fun z o =>
      ∀ (rb : ByteArray) (idx : ℕ) (ss : List ByteArray), Carries z row rb →
        List.Forall₂ (fun w b => b.size = 64 ∧ Carries z w b) (sibForms aux H) ss →
        (∀ h < H, ev z (dirs.getD h Lin.zero) = idx.testBit h) → CallsAt z o.2 (rowCalls rb idx ss) := by
  unfold recOpenC
  refine Sound.bind' (sound_residualForms _ _ _) fun t => Sound.bind' (sound_climbRowC H row aux _) fun out =>
    Sound.pure' fun z hout _ rb idx ss hrow hss hdir => ?_
  exact hout rb idx ss hrow hss fun h hh => by rw [getD_take_of_lt (by omega)]; exact hdir h hh

theorem laid_recOpen_calls (LANES H : ℕ) (hH : H ≤ 32) (p : Ports) (outBits : Array ℕ) (v : ℕ → Bool)
    (h : LaidRows (recOpenLaid LANES H p outBits) (inRowsOf p.bits) v) (rb : ByteArray) (idx : ℕ)
    (ss : List ByteArray) (hss : ss.length = H) (hrowN : p.row = 8 * rb.size) (hauxN : 512 * H ≤ p.aux)
    (hdirsN : H ≤ p.dirs) (hrow : ColsHold v 0 rb)
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧ ColsHold v (p.row + 512 * i) ss[i])
    (hdir : ∀ h < H, v (p.row + p.aux + p.acc + p.coef + h) = idx.testBit h) :
    CallsAt (fun x => v (col (p.bits.foldl (· + ·) 0) (inRowsOf p.bits) x)) (recOpenCalls LANES H p)
      (rowCalls rb idx ss) := by
  have hf := foldl_bits p
  have hg : Good (p.bits.foldl (· + ·) 0) ((recOpenNet LANES H p).run St.empty).2 := by
    rw [recOpenNet_run, hf]
    exact fresh_recOpen LANES H _ _ _ _ _ _ ⟨p.row + p.aux + p.acc + p.coef + p.dirs, []⟩
      ⟨by simp, fun k r h => by simp at h⟩
  have hst := laid_rows ((recOpenNet LANES H p).run St.empty).2 p.bits outBits (recOpenOuts LANES H p).toArray v hg h
  rw [← recOpenNetC_state, recOpenNetC_run] at hst
  have hP := (sound_recOpenC LANES H hH (vars 0 p.row) (vars p.row p.aux) (vars (p.row + p.aux) p.acc)
    (vars (p.row + p.aux + p.acc) p.coef) (vars (p.row + p.aux + p.acc + p.coef) p.dirs)
    ⟨p.row + p.aux + p.acc + p.coef + p.dirs, []⟩).2 _ hst
  rw [recOpenCalls, recOpenNetC_run]
  exact hP rb idx ss (carries_vars hrowN (by rw [hf]; omega) hrow) (forall₂_sibs ss hss hauxN (by rw [hf]; omega) haux)
    fun j hj => (ev_vars_getD (by omega) (by rw [hf])).trans (hdir j hj)

def unitCol (LANES H : ℕ) (x : Var) : ℕ := col (unitIn LANES H) (inRowsOf (Flock.RecOpen.wordPorts (unitIn LANES H))) x

def unitWires (LANES H : ℕ) (v : ℕ → Bool) : Var → Bool := fun x => v (unitCol LANES H x)

def unitCalls (LANES H : ℕ) : List (List Lin × List Lin) := recOpenCalls LANES H (portsOf (Flock.RecOpen.ports LANES H))

theorem laid_unit_calls (LANES H : ℕ) (hH : H ≤ 32) (v : ℕ → Bool)
    (h : LaidRows (Flock.RecOpen.unitLaid LANES H) (inRowsOf (Flock.RecOpen.wordPorts (unitIn LANES H))) v)
    (rb : ByteArray) (idx : ℕ) (ss : List ByteArray) (hrb : rb.size = 16 * LANES) (hss : ss.length = H)
    (hrow : ColsHold v 0 rb)
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧ ColsHold v (128 * LANES + 512 * i) ss[i])
    (hdir : ∀ h < H, v (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024 + 256 * LANES + h) = idx.testBit h) :
    CallsAt (unitWires LANES H v) (unitCalls LANES H) (rowCalls rb idx ss) := by
  have hi := foldl_unitIn LANES H
  have h' : LaidRows (recOpenLaid LANES H (portsOf (Flock.RecOpen.ports LANES H)) (Flock.RecOpen.wordPorts 2048))
      (inRowsOf (portsOf (Flock.RecOpen.ports LANES H)).bits) v := by
    simp only [Flock.RecOpen.unitLaid, recOpenNet_eq, recOpenOuts_eq] at h
    rw [inRowsOf, hi] at h
    exact (laidRows_layout_congr _ _ hi rfl _ _).1 h
  have hc := laid_recOpen_calls LANES H hH (portsOf (Flock.RecOpen.ports LANES H)) (Flock.RecOpen.wordPorts 2048) v
    h' rb idx ss hss (by simp only [portsOf, Flock.RecOpen.ports]; omega)
    (by simp only [portsOf, Flock.RecOpen.ports]; omega) (by simp only [portsOf, Flock.RecOpen.ports]; omega) hrow
    haux hdir
  have hn : (portsOf (Flock.RecOpen.ports LANES H)).bits.foldl (· + ·) 0 = unitIn LANES H := by
    rw [foldl_bits, unitIn_eq]
    simp only [portsOf, Flock.RecOpen.ports]
  have hr : inRowsOf (portsOf (Flock.RecOpen.ports LANES H)).bits =
      inRowsOf (Flock.RecOpen.wordPorts (unitIn LANES H)) := by
    simp only [inRowsOf, hi]
  rw [hn, hr] at hc
  exact hc

end FlockVBridge

namespace Flock.Readable

open FlockSoundness FlockSoundness.Discharge.Hm FlockVBridge.OfVStar

theorem parseFacts_of_parse {decode : Claims.Msg → Option Read} {x : Claims.Msg} {p : Parsed}
    (hx : parseBytes decode x = some p) (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ}
    (htl : t < p.sts.size) (hf : p.sts[t].c.tmpl = none) : ∃ tags, FlockSoundness.ParseFacts tags p.sts[t].c := by
  unfold parseBytes at hx
  obtain ⟨r, -, hr⟩ := Option.bind_eq_some_iff.1 hx
  exact parseFacts_of_tablesOf (tablesOf_of_parses (parses_of_parse hr)) ht hpi htl hf

theorem parseFactsU_of_parse {decode : Claims.Msg → Option Read} {x : Claims.Msg} {p : Parsed}
    (hx : parseBytes decode x = some p) (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ}
    (htl : t < p.sts.size) : ∃ tags pre, FlockSoundness.ParseFactsU tags pre p.sts[t].c := by
  unfold parseBytes at hx
  obtain ⟨r, -, hr⟩ := Option.bind_eq_some_iff.1 hx
  exact parseFactsU_of_tablesOf (tablesOf_of_parses (parses_of_parse hr)) ht hpi htl

def sparseT (sh : Flock.RecSparse.Shape) (v : ℕ → Bool) : Array ℕ :=
  Array.ofFn fun s : Fin sh.S =>
    FlockVBridge.Family.natOfBits (fun b => v (FlockVBridge.Sparse.tauBase sh + (16 * s + b))) 16

theorem sparseIn_reads_any (sh : Flock.RecSparse.Shape) (v : ℕ → Bool) :
    FlockVBridge.Sparse.SparseIn sh ((sparseW sh v)[·]!) ((sparseV sh v)[·]!) ((sparseT sh v)[·]!) v := by
  have hW : (fun e => (sparseW sh v)[e]!) =
      readElems (rowCol (Flock.RecSparse.messagePorts sh) 0) ((Flock.RecSparse.messagePorts sh).map (·.2)).sum v :=
    funext ofFn_readElems
  have hV : (fun e => (sparseV sh v)[e]!) =
      readElems (FlockVBridge.Sparse.vBaseS sh + ·) (Flock.RecSparse.nv sh) v :=
    funext ofFn_readElems
  refine ⟨fun q hq => ?_, fun q hq => ?_, fun _ s hs b hb => ?_, fun e he => ?_, fun e he => ?_⟩
  · rw [hW]
    exact (wbit_readElems hq).symm
  · rw [hV]
    exact (wbit_readElems hq).symm
  · rw [getElem!_pos (sparseT sh v) s (by simpa [sparseT] using hs)]
    simp [sparseT, FlockVBridge.Family.testBit_natOfBits, hb]
  · exact (ofFn_readElems e).trans (readElems_ge he)
  · exact (ofFn_readElems e).trans (readElems_ge he)

/-- A gadget's input forms are XORs of columns above the leaf, not columns. -/
def LevelCalls (v : ℕ → Bool) (LANES H : ℕ) : Prop :=
  FlockVBridge.CallsAt (FlockVBridge.unitWires LANES H v) (FlockVBridge.unitCalls LANES H)
    (FlockVBridge.rowCalls (Flock.MerkleScheme.rowBytes (openRow v LANES)) (openIdx v LANES H) (openSibs v LANES H))

theorem levelCalls_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size) {LANES H : ℕ}
    (hp : Flock.RecOpen.params p.sts[t].c.unit.name = some (LANES, H) ∨
      Flock.RecOpen.stackedParams p.sts[t].c.unit.name = some (LANES, H))
    {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv) :
    LevelCalls (copyOf p t htl w o g u) LANES H := by
  obtain ⟨L, regs, hz⟩ := layout_of_check h hx ht hpi htl
  obtain ⟨_, _, P⟩ := parseFactsU_of_parse hx ht hpi htl
  have hnet : ∃ name, H ≤ 32 ∧
      Flock.HmNets.netOf name (Flock.RecOpen.unitLaid LANES H) = .ok p.sts[t].c.unit := by
    rcases hp with hp | hp
    · obtain ⟨hv, hl⟩ := Flock.RecOpen.check_ok P.recNet hp
      simp only [Flock.RecOpen.valid, Bool.and_eq_true, decide_eq_true_eq] at hv
      exact ⟨_, hv.2, hl⟩
    · obtain ⟨hv, hl⟩ := Flock.RecOpen.check_ok_stacked P.recNet hp
      simp only [Flock.RecOpen.valid, Bool.and_eq_true, decide_eq_true_eq] at hv
      exact ⟨_, hv.2, hl⟩
  obtain ⟨name, hH, hl⟩ := hnet
  obtain ⟨hr, -, -, -, hss, hrow, haux, -, -, hdir⟩ := openIns_reads (copyOf p t htl w o g u) LANES H
  have hL := laidRows_of_satisfies L (flat_of_check h hx ht hpi htl (recUnit_open hp)) regs hz ho hg hu hl
  rw [FlockVBridge.inWords_unit hl] at hL
  exact FlockVBridge.laid_unit_calls LANES H hH _ hL _ _ _ (by rw [FlockVBridge.size_rowBytes, hr]) hss hrow haux hdir

def LevelOuts (v : ℕ → Bool) (LANES H : ℕ) : Prop :=
  OpenOuts v Flock.MerkleScheme.hm96Sha512 LANES H (openRow v LANES) (openAcc v LANES H) (openCoefs v LANES H)
    (openRest v LANES H) (openIdx v LANES H) (openSibs v LANES H) ∧ LevelCalls v LANES H

def ResOuts (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ) (v : ℕ → Bool) : Prop :=
  PartCols S (rowsRead ns nv v) (vRead ns nv v) v (partBase S ns nv)

def SparseReadOuts (sh : Flock.RecSparse.Shape) (v : ℕ → Bool) : Prop :=
  SparseOuts sh v (sparseW sh v) (sparseV sh v) (sparseT sh v)

def ResPartAny (c : Flock.Circuit) (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ) : Prop :=
  (∃ (m k cl j : ℕ) (sh : Flock.Shape) (holo : Bool), RecResiduals.params c.unit.name = some (m, k, cl, j) ∧
    RecResiduals.shape? m k cl = some sh ∧ RecResiduals.holoOf c = .ok holo ∧
    RecResiduals.part { sh with holo } j = .ok (S, ns, nv)) ∨
  ∃ (m cl j : ℕ) (sh : Flock.Shape), RecOpening.params c.unit.name = some (m, cl, j) ∧
    RecOpening.shape? m cl = some sh ∧ RecOpening.part sh j = .ok (S, ns, nv)

theorem resPartAny_of_resPart {c : Flock.Circuit} {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (h : ResPart c S ns nv) : ResPartAny c S ns nv := by
  rcases h with ⟨m, k, cl, j, sh, hp, hsh, hh, hpart⟩ | h
  · have e : { sh with holo := false } = sh := by
      have h0 := shape?_holo hsh
      cases sh
      simp only at h0
      subst h0
      rfl
    exact Or.inl ⟨m, k, cl, j, sh, false, hp, hsh, hh, e ▸ hpart⟩
  · exact Or.inr h

/-- Of RecCap's and RecLink's outputs nothing is stated (`K` true). -/
def UnitOuts (c : Flock.Circuit) (K : (ℕ → Bool) → Prop) : Prop :=
  (∃ LANES H : ℕ, (Flock.RecOpen.params c.unit.name = some (LANES, H) ∨
    Flock.RecOpen.stackedParams c.unit.name = some (LANES, H)) ∧ K = fun v => LevelOuts v LANES H) ∨
  (∃ (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ), ResPartAny c S ns nv ∧ K = ResOuts S ns nv) ∨
  (∃ sh : Flock.RecSparse.Shape, Flock.RecSparse.params c.unit.name = some sh ∧ K = SparseReadOuts sh) ∨
  ∃ l N : ℕ, (c.unit.name = VStar.capName l N ∨ c.unit.name = VStar.linkName l N) ∧ K = fun _ => True

theorem unitOuts_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    {K : (ℕ → Bool) → Prop} (hK : UnitOuts p.sts[t].c K) {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog))
    (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv) : K (copyOf p t htl w o g u) := by
  obtain ⟨_, _, P⟩ := parseFactsU_of_parse hx ht hpi htl
  rcases hK with ⟨LANES, H, hp, rfl⟩ | ⟨S, ns, nv, hpart, rfl⟩ | ⟨sh, hp, rfl⟩ | ⟨l, N, -, rfl⟩
  · exact ⟨climbAt_of_check h hx ht hpi htl hp ho hg hu Flock.MerkleScheme.hm96Sha512 (fun _ _ => rfl)
      (openIns_reads _ LANES H), levelCalls_of_check h hx ht hpi htl hp ho hg hu⟩
  · rcases hpart with ⟨m, k, cl, j, sh, holo, hp, hsh, hh, hpart⟩ | ⟨m, cl, j, sh, hp, hsh, hpart⟩
    · exact partCols_of_check_at h hx ht hpi htl P.recResiduals hp hsh hh hpart ho hg hu _ _
        (fun _ he => readElems_ge he) (fun _ he => readElems_ge he) (inCols_readElems ns nv _)
    · exact partCols_of_check_opening h hx ht hpi htl P.recOpening hp hsh hpart ho hg hu _ _
        (fun _ he => readElems_ge he) (fun _ he => readElems_ge he) (inCols_readElems ns nv _)
  · exact sparseOuts_of_check h hx ht hpi htl hp ho hg hu _ _ _ (sparseIn_reads_any sh _)
  · trivial

def AllOuts (p : Parsed) (w : Claims.Flock.Wit) : Prop :=
  ∀ (t : ℕ) (ht : t < p.sts.size) (K : (ℕ → Bool) → Prop), UnitOuts p.sts[t].c K → ∀ o g u,
    o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) → g < p.sts[t].c.g → u < p.sts[t].c.upv → K (copyOf p t ht w o g u)

theorem allOuts_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) : AllOuts p w :=
  fun _ htl _ hK _ _ _ ho hg hu => unitOuts_of_check h hx ht hpi htl hK ho hg hu

theorem stackedName_inj {L H L' H' : ℕ} (h : Flock.RecOpen.stackedName L H = Flock.RecOpen.stackedName L' H') :
    L = L' ∧ H = H' := by
  have h := congrArg String.toList h
  simp only [Flock.RecOpen.stackedName, String.toList_append, toString, Nat.toList_repr, List.append_assoc] at h
  have h := List.append_cancel_left h
  have dig : ∀ n, ∀ x ∈ Nat.toDigits 10 n, x.isDigit := fun n x hx =>
    Nat.isDigit_of_mem_toDigits (by decide) (by decide) hx
  obtain ⟨h1, h2⟩ := FlockVBridge.digits_dash (dig L) (dig L') (by simpa using h)
  simp only [List.cons.injEq, true_and] at h2
  exact ⟨FlockVBridge.eq_of_toDigits h1, FlockVBridge.eq_of_toDigits h2⟩

theorem stackedName_startsWith (L H : ℕ) :
    (Flock.RecOpen.stackedName L H).startsWith "rec-open-stacked/" = true := by
  have : Flock.RecOpen.stackedName L H = "rec-open-stacked/" ++ s!"l{L}-h{H}" := by
    simp [Flock.RecOpen.stackedName, ← String.toList_inj, List.append_assoc, FlockVBridge.toString_str]
  rw [this]
  simp [String.startsWith_string_iff]

theorem stackedParams_of_check {c : Flock.Circuit} (h : Flock.RecOpen.check c = .ok ()) {L H : ℕ}
    (hn : c.unit.name = VStar.recOpenStackedName L H) : Flock.RecOpen.stackedParams c.unit.name = some (L, H) := by
  have hn' : c.unit.name = Flock.RecOpen.stackedName L H := hn
  cases hp : Flock.RecOpen.stackedParams c.unit.name with
  | some p =>
    obtain ⟨L', H'⟩ := p
    obtain ⟨rfl, rfl⟩ := stackedName_inj ((Flock.RecOpen.stackedName_of_params hp).trans hn')
    rfl
  | none =>
    have hu : Flock.RecOpen.params c.unit.name = none := by
      cases hq : Flock.RecOpen.params c.unit.name with
      | none => rfl
      | some q =>
        exact absurd ((Flock.RecOpen.unitName_of_params hq).trans hn') (Flock.RecOpen.unitName_ne_stackedName _ _ _ _)
    unfold Flock.RecOpen.check at h
    rw [hu, hp] at h
    dsimp only at h
    rw [hn', Flock.RecOpen.startsWith_stackedName, stackedName_startsWith] at h
    cases h

theorem sparseName_inj {sh sh' : Flock.RecSparse.Shape}
    (h : Flock.RecSparse.unitName sh = Flock.RecSparse.unitName sh') : sh = sh' := by
  have h := congrArg String.toList h
  simp only [Flock.RecSparse.unitName, String.toList_append, toString, Nat.toList_repr, List.append_assoc] at h
  have h := List.append_cancel_left h
  have dig : ∀ n, ∀ x ∈ Nat.toDigits 10 n, x.isDigit := fun n x hx =>
    Nat.isDigit_of_mem_toDigits (by decide) (by decide) hx
  obtain ⟨h1, h⟩ := FlockVBridge.digits_dash (dig _) (dig _) (by simpa using h)
  simp only [List.cons.injEq, true_and] at h
  obtain ⟨h2, h⟩ := FlockVBridge.digits_dash (dig _) (dig _) (by simpa using h)
  simp only [List.cons.injEq, true_and] at h
  obtain ⟨h3, h⟩ := FlockVBridge.digits_dash (dig _) (dig _) (by simpa using h)
  simp only [List.cons.injEq, true_and] at h
  obtain ⟨_, _, _, _⟩ := sh
  obtain ⟨_, _, _, _⟩ := sh'
  simp only [Flock.RecSparse.Shape.mk.injEq]
  exact ⟨FlockVBridge.eq_of_toDigits h1, FlockVBridge.eq_of_toDigits h2, FlockVBridge.eq_of_toDigits h3,
    FlockVBridge.eq_of_toDigits h⟩

theorem sparseName_startsWith (sh : Flock.RecSparse.Shape) :
    (Flock.RecSparse.unitName sh).startsWith "rec-sparse/" = true := by
  have : Flock.RecSparse.unitName sh = "rec-sparse/" ++ s!"n{sh.nu}-e{sh.ell}-s{sh.sLog}-t{sh.tLog}" := by
    simp [Flock.RecSparse.unitName, ← String.toList_inj, List.append_assoc, FlockVBridge.toString_str]
  rw [this]
  simp [String.startsWith_string_iff]

theorem sparseParams_of_check {c : Flock.Circuit} (h : Flock.RecSparse.check c = .ok ())
    {sh : Flock.RecSparse.Shape} (hn : c.unit.name = Flock.RecSparse.unitName sh) :
    Flock.RecSparse.params c.unit.name = some sh := by
  cases hp : Flock.RecSparse.params c.unit.name with
  | some sh' => rw [sparseName_inj ((FlockVBridge.Sparse.unitName_params hp).trans hn)]
  | none =>
    unfold Flock.RecSparse.check at h
    rw [hp] at h
    dsimp only at h
    rw [hn, sparseName_startsWith] at h
    cases h

theorem openingName_inj {m c j m' c' j' : ℕ}
    (h : Flock.RecOpening.unitName m c j = Flock.RecOpening.unitName m' c' j') : m = m' ∧ c = c' ∧ j = j' := by
  have h := congrArg String.toList h
  simp only [Flock.RecOpening.unitName, String.toList_append, toString, Nat.toList_repr, List.append_assoc] at h
  have h := List.append_cancel_left h
  have dig : ∀ n, ∀ x ∈ Nat.toDigits 10 n, x.isDigit := fun n x hx =>
    Nat.isDigit_of_mem_toDigits (by decide) (by decide) hx
  obtain ⟨h1, h⟩ := FlockVBridge.digits_dash (dig _) (dig _) (by simpa using h)
  simp only [List.cons.injEq, true_and] at h
  obtain ⟨h2, h⟩ := FlockVBridge.digits_dash (dig _) (dig _) (by simpa using h)
  simp only [List.cons.injEq, true_and] at h
  exact ⟨FlockVBridge.eq_of_toDigits h1, FlockVBridge.eq_of_toDigits h2, FlockVBridge.eq_of_toDigits h⟩

theorem openingName_startsWith (m c j : ℕ) :
    (Flock.RecOpening.unitName m c j).startsWith "rec-opening/" = true := by
  have : Flock.RecOpening.unitName m c j = "rec-opening/" ++ s!"m{m}-c{c}-p{j}" := by
    simp [Flock.RecOpening.unitName, ← String.toList_inj, List.append_assoc, FlockVBridge.toString_str]
  rw [this]
  simp [String.startsWith_string_iff]

theorem openingName_of_params {name : String} {m c j : ℕ} (h : Flock.RecOpening.params name = some (m, c, j)) :
    Flock.RecOpening.unitName m c j = name := by
  unfold Flock.RecOpening.params at h
  repeat' (first | split at h | dsimp only at h)
  all_goals first
    | (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨rfl, rfl, rfl⟩ := h; assumption)
    | cases h

theorem openingParams_of_check {c : Flock.Circuit} (h : Flock.RecOpening.check c = .ok ()) {m cl j : ℕ}
    (hn : c.unit.name = Flock.RecOpening.unitName m cl j) :
    Flock.RecOpening.params c.unit.name = some (m, cl, j) := by
  cases hp : Flock.RecOpening.params c.unit.name with
  | some q =>
    obtain ⟨m', cl', j'⟩ := q
    obtain ⟨rfl, rfl, rfl⟩ := openingName_inj ((openingName_of_params hp).trans hn)
    rfl
  | none =>
    unfold Flock.RecOpening.check at h
    rw [hp] at h
    dsimp only at h
    rw [hn, openingName_startsWith] at h
    cases h

theorem openingShape_of_check {c : Flock.Circuit} (h : Flock.RecOpening.check c = .ok ()) {m cl j : ℕ}
    (hp : Flock.RecOpening.params c.unit.name = some (m, cl, j)) : ∃ sh, Flock.RecOpening.shape? m cl = some sh := by
  cases hsh : Flock.RecOpening.shape? m cl with
  | some sh => exact ⟨sh, rfl⟩
  | none =>
    unfold Flock.RecOpening.check at h
    rw [hp] at h
    dsimp only at h
    rw [hsh] at h
    cases h

theorem checkPart_inner {j : ℕ} {nm : String} {pt : VStar.Part} {c : Flock.Circuit}
    (h : Flock.RecVStar.checkPart (.inner j nm pt) c = .ok ()) :
    ∃ (m k cl jj : ℕ) (sh : Flock.Shape) (holo : Bool) (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ),
      RecResiduals.params c.unit.name = some (m, k, cl, jj) ∧ RecResiduals.shape? m k cl = some sh ∧
        RecResiduals.holoOf c = .ok holo ∧ RecResiduals.part { sh with holo } jj = .ok (S, ns, nv) := by
  unfold Flock.RecVStar.checkPart at h
  dsimp only at h
  cases hp : Flock.RecResiduals.params c.unit.name with
  | none => rw [hp] at h; cases h
  | some q =>
    obtain ⟨m, k, cl, jj⟩ := q
    rw [hp] at h
    dsimp only at h
    cases hsh : Flock.RecResiduals.shape? m k cl with
    | none => rw [hsh] at h; cases h
    | some sh =>
      rw [hsh] at h
      dsimp only at h
      cases hh : Flock.RecResiduals.holoOf c with
      | error e => rw [hh] at h; cases h
      | ok holo =>
        rw [hh] at h
        dsimp only at h
        cases hpt : Flock.RecResiduals.part { sh with holo } jj with
        | error e => rw [hpt] at h; cases h
        | ok q =>
          obtain ⟨S, ns, nv⟩ := q
          exact ⟨m, k, cl, jj, sh, holo, S, ns, nv, rfl, hsh, rfl, hpt⟩

/-- Of RecCap's and RecLink's outputs nothing is stated (`.cap`, `.link`). -/
def StmtUnit (sv : VStar.Stmt) (c : Flock.Circuit) (K : (ℕ → Bool) → Prop) : Prop :=
  UnitOuts c K ∧
    match sv with
    | .recOpen _ L H _ => K = fun v => LevelOuts v L H
    | .recOpenStacked _ L H _ => K = fun v => LevelOuts v L H
    | .inner .. => ∃ (m k cl j : ℕ) (sh : Flock.Shape) (holo : Bool) (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ),
        RecResiduals.params c.unit.name = some (m, k, cl, j) ∧ RecResiduals.shape? m k cl = some sh ∧
        RecResiduals.holoOf c = .ok holo ∧ RecResiduals.part { sh with holo } j = .ok (S, ns, nv) ∧
        K = ResOuts S ns nv
    | .opening m cl j .. => ∃ (sh : Flock.Shape) (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ),
        RecOpening.shape? m cl = some sh ∧ RecOpening.part sh j = .ok (S, ns, nv) ∧ K = ResOuts S ns nv
    | .sparse sh _ => K = SparseReadOuts sh
    | .cap .. => K = fun _ => True
    | .link .. => K = fun _ => True

theorem stmtUnit_of_check {tags : Flock.Tags} {pre : Array (String × Flock.Net)} {c : Flock.Circuit}
    (P : FlockSoundness.ParseFactsU tags pre c)
    {sv : VStar.Stmt} (hn : c.unit.name = sv.name) (hp : Flock.RecVStar.checkPart sv c = .ok ()) :
    ∃ K, StmtUnit sv c K := by
  cases sv with
  | recOpen level L H n =>
    exact ⟨_, Or.inl ⟨L, H, Or.inl (FlockVBridge.params_of_check P.recNet hn), rfl⟩, rfl⟩
  | recOpenStacked level L H n =>
    exact ⟨_, Or.inl ⟨L, H, Or.inr (stackedParams_of_check P.recNet hn), rfl⟩, rfl⟩
  | inner j nm pt =>
    obtain ⟨m, k, cl, jj, sh, holo, S, ns, nv, h1, h2, h3, h4⟩ := checkPart_inner hp
    exact ⟨_, Or.inr (Or.inl ⟨S, ns, nv, Or.inl ⟨m, k, cl, jj, sh, holo, h1, h2, h3, h4⟩, rfl⟩),
      m, k, cl, jj, sh, holo, S, ns, nv, h1, h2, h3, h4, rfl⟩
  | sparse sh hRound =>
    exact ⟨_, Or.inr (Or.inr (Or.inl ⟨sh, sparseParams_of_check P.recSparse hn, rfl⟩)), rfl⟩
  | cap l N =>
    exact ⟨_, Or.inr (Or.inr (Or.inr ⟨l, N, Or.inl hn, rfl⟩)), rfl⟩
  | link UB PB N =>
    exact ⟨_, Or.inr (Or.inr (Or.inr ⟨UB, PB, Or.inr hn, rfl⟩)), rfl⟩
  | opening m cl j S ports nv =>
    have hq := openingParams_of_check P.recOpening hn
    obtain ⟨sh, hsh⟩ := openingShape_of_check P.recOpening hq
    obtain ⟨S', ns, nv', hpart, -⟩ := check_opening P.recOpening hq hsh
    exact ⟨_, Or.inr (Or.inl ⟨S', ns, nv', Or.inr ⟨m, cl, j, sh, hq, hsh, hpart⟩, rfl⟩), sh, S', ns, nv', hsh,
      hpart, rfl⟩

def StmtCopies (sv : VStar.Stmt) (p : Parsed) (w : Claims.Flock.Wit) : Prop :=
  ∀ t (ht : t < p.sts.size), p.sts[t].c.unit.name = sv.name ∧ ∃ K, StmtUnit sv p.sts[t].c K ∧
    ∀ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) → g < p.sts[t].c.g → u < p.sts[t].c.upv →
      K (copyOf p t ht w o g u)

theorem stmtCopies_of_allOuts {sv : VStar.Stmt} {p : Parsed} {w : Claims.Flock.Wit}
    (htab : ∀ t (ht : t < p.sts.size), p.sts[t].c.unit.name = sv.name ∧ ∃ K, StmtUnit sv p.sts[t].c K)
    (hO : AllOuts p w) : StmtCopies sv p w := fun t ht => by
  obtain ⟨hn, K, hK⟩ := htab t ht
  exact ⟨hn, K, hK, fun o g u ho hg hu => hO t ht K hK.1 o g u ho hg hu⟩

section Kinds

variable {p : Parsed} {w : Claims.Flock.Wit} {t : ℕ} (ht : t < p.sts.size) {o g u : ℕ}
  (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
include ht ho hg hu

theorem levelCopies {lv LANES H n : ℕ} (h : StmtCopies (.recOpen lv LANES H n) p w) :
    LevelOuts (copyOf p t ht w o g u) LANES H := by
  obtain ⟨-, K, ⟨-, hK⟩, hc⟩ := h t ht
  subst hK
  exact hc o g u ho hg hu

theorem stackedCopies {lv LANES H n : ℕ} (h : StmtCopies (.recOpenStacked lv LANES H n) p w) :
    LevelOuts (copyOf p t ht w o g u) LANES H := by
  obtain ⟨-, K, ⟨-, hK⟩, hc⟩ := h t ht
  subst hK
  exact hc o g u ho hg hu

theorem sparseCopies {sh : Flock.RecSparse.Shape} {hR : ℕ} (h : StmtCopies (.sparse sh hR) p w) :
    SparseReadOuts sh (copyOf p t ht w o g u) := by
  obtain ⟨-, K, ⟨-, hK⟩, hc⟩ := h t ht
  subst hK
  exact hc o g u ho hg hu

theorem innerCopies {j : ℕ} {nm : String} {pt : VStar.Part} (h : StmtCopies (.inner j nm pt) p w) :
    ∃ (m k cl jj : ℕ) (sh : Flock.Shape) (holo : Bool) (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ),
      RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, jj) ∧ RecResiduals.shape? m k cl = some sh ∧
      RecResiduals.holoOf p.sts[t].c = .ok holo ∧ RecResiduals.part { sh with holo } jj = .ok (S, ns, nv) ∧
      ResOuts S ns nv (copyOf p t ht w o g u) := by
  obtain ⟨-, K, ⟨-, m, k, cl, jj, sh, holo, S, ns, nv, h1, h2, h3, h4, hK⟩, hc⟩ := h t ht
  subst hK
  exact ⟨m, k, cl, jj, sh, holo, S, ns, nv, h1, h2, h3, h4, hc o g u ho hg hu⟩

theorem openingCopies {m cl j : ℕ} {S0 : Flock.RecOpen.GfS} {ports : List (String × ℕ)} {nv0 : ℕ}
    (h : StmtCopies (.opening m cl j S0 ports nv0) p w) :
    ∃ (sh : Flock.Shape) (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ),
      RecOpening.shape? m cl = some sh ∧ RecOpening.part sh j = .ok (S, ns, nv) ∧
      ResOuts S ns nv (copyOf p t ht w o g u) := by
  obtain ⟨-, K, ⟨-, sh, S, ns, nv, h1, h2, hK⟩, hc⟩ := h t ht
  subst hK
  exact ⟨sh, S, ns, nv, h1, h2, hc o g u ho hg hu⟩

end Kinds

end Flock.Readable

namespace Claims.Recursion.VStar

open FlockVBridge.OfVStar
open FlockSoundness.Discharge.Hm (bitL)

section Held

def bitsNat (bs : List Bool) : ℕ := bs.foldr (fun b a => (if b then 1 else 0) + 2 * a) 0

theorem toNat_byte (bs : List Bool) : (byte bs).toNat = bitsNat bs % 256 := by
  induction bs with
  | nil => rfl
  | cons b bs ih =>
    simp only [byte, bitsNat, List.foldr_cons] at ih ⊢
    rw [UInt8.toNat_add, UInt8.toNat_mul, ih]
    cases b <;> simp

theorem bitsNat_testBits (n : ℕ) : ∀ k, bitsNat ((List.range k).map n.testBit) = n % 2 ^ k := by
  intro k
  induction k generalizing n with
  | zero => simp [bitsNat, Nat.mod_one]
  | succ k ih =>
    rw [List.range_succ_eq_map, List.map_cons, List.map_map]
    have hf : n.testBit ∘ Nat.succ = (n / 2).testBit := funext fun i => Nat.testBit_succ n i
    simp only [bitsNat, List.foldr_cons] at ih ⊢
    rw [hf, ih, Nat.testBit_zero, Nat.pow_succ', Nat.mod_mul]
    by_cases h : n % 2 = 1 <;> simp [h]
    omega

theorem byte_testBits (b : UInt8) : byte ((List.range 8).map b.toNat.testBit) = b := by
  apply UInt8.toNat_inj.1
  rw [toNat_byte, bitsNat_testBits]
  have := UInt8.toNat_lt b
  omega

theorem bytesOf_bitL (l : List UInt8) : bytesOf ((List.range (8 * l.length)).map (bitL l)) = l := by
  apply List.ext_getElem
  · simp [bytesOf]
    omega
  · intro j h1 h2
    simp only [bytesOf, List.getElem_map, List.getElem_range]
    have hj : j < l.length := h2
    have e : ((List.map (bitL l) (List.range (8 * l.length))).drop (8 * j)).take 8 =
        (List.range 8).map (l[j]).toNat.testBit := by
      apply List.ext_getElem
      · simp
        omega
      · intro k hk1 hk2
        simp only [List.getElem_take, List.getElem_drop, List.getElem_map, List.getElem_range, bitL]
        have hk : k < 8 := by simpa using hk2
        have e1 : (8 * j + k) / 8 = j := by omega
        have e2 : (8 * j + k) % 8 = k := by omega
        rw [e1, e2, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj, Option.getD_some]
    rw [e, byte_testBits]

theorem readAt_of_bits {w : Flock.Wit} {ps : List ℕ} {l : List UInt8} (hlen : ps.length = 8 * l.length)
    (h : ∀ q (hq : q < ps.length), w.hid.getD ps[q] false = bitL l q) : readAt w ps = l := by
  unfold readAt
  have e : ps.map (fun p => w.hid.getD p false) = (List.range (8 * l.length)).map (bitL l) :=
    List.ext_getElem (by simp [hlen]) fun q h1 _ => by
      simp only [List.getElem_map, List.getElem_range]
      exact h q (by simpa using h1)
  rw [e, bytesOf_bitL]

def copyIdx (p : Flock.Readable.Parsed) (t : ℕ) (ht : t < p.sts.size) (o g u j : ℕ) : ℕ :=
  (o * 2 ^ p.sts[t].c.kLog + (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u) + j)) * p.sts.size + t

theorem copyOf_eq {p : Flock.Readable.Parsed} {t : ℕ} {ht : t < p.sts.size} {w : Flock.Wit} {o g u j : ℕ}
    (hj : j ≤ p.sts[t].c.unit.constPos) :
    Flock.Readable.copyOf p t ht w o g u j = w.hid.getD (copyIdx p t ht o g u j) false := by
  simp [Flock.Readable.copyOf, Flock.Readable.copyCol, copyIdx, hj]

def copyPs (p : Flock.Readable.Parsed) (t : ℕ) (ht : t < p.sts.size) (o g u base n : ℕ) : List ℕ :=
  (List.range n).map fun q => copyIdx p t ht o g u (base + q)

theorem held_of_colsHold {L : Layout} {x : Msg} {s i : ℕ} {p : Flock.Readable.Parsed} {t : ℕ}
    {ht : t < p.sts.size} {w : Flock.Wit} {o g u base : ℕ} {pre : Msg} {b : ByteArray} (hfx : L.fixedAt x i = none)
    (hm : L.msgAt x s i = some [(pre, copyPs p t ht o g u base (8 * b.size))])
    (htop : base + 8 * b.size ≤ p.sts[t].c.unit.constPos + 1)
    (hb : FlockVBridge.ColsHold (Flock.Readable.copyOf p t ht w o g u) base b) :
    L.held x s w i = some (pre ++ b.data.toList) := by
  simp only [Layout.held, hfx, Option.isSome_none, Bool.false_eq_true, ↓reduceIte, hm, Option.map_some, readSegs,
    List.flatMap_cons, List.flatMap_nil, List.append_nil]
  congr 2
  apply readAt_of_bits
  · simp only [copyPs, List.length_map, List.length_range, Array.length_toList]
    rfl
  · intro q hq
    have hq' : q < 8 * b.size := by simpa [copyPs] using hq
    simp only [copyPs, List.getElem_map, List.getElem_range]
    rw [← copyOf_eq (by omega)]
    exact hb q hq'

theorem agrees_colsHold {L : Layout} {x : Msg} {s i : ℕ} {p : Flock.Readable.Parsed} {t : ℕ}
    {ht : t < p.sts.size} {w : Flock.Wit} {ms : List Msg} (hA : Agrees L x s w ms) (hi : i < L.rounds x)
    (hfx : L.fixedAt x i = none) {o g u base : ℕ} {pre : Msg} {b : ByteArray}
    (hm : L.msgAt x s i = some [(pre, copyPs p t ht o g u base (8 * b.size))])
    (htop : base + 8 * b.size ≤ p.sts[t].c.unit.constPos + 1)
    (hb : FlockVBridge.ColsHold (Flock.Readable.copyOf p t ht w o g u) base b) :
    ms.getD i [] = pre ++ b.data.toList :=
  hA i hi _ (held_of_colsHold hfx hm htop hb)

theorem agrees_openRow {L : Layout} {x : Msg} {s i : ℕ} {p : Flock.Readable.Parsed} {t : ℕ} {ht : t < p.sts.size}
    {w : Flock.Wit} {ms : List Msg} (hA : Agrees L x s w ms) (hi : i < L.rounds x) (hfx : L.fixedAt x i = none)
    {o g u LANES H : ℕ} {pre : Msg} (hm : L.msgAt x s i = some [(pre, copyPs p t ht o g u 0 (128 * LANES))])
    (htop : 128 * LANES ≤ p.sts[t].c.unit.constPos + 1) :
    ms.getD i [] = pre ++ (_root_.Flock.MerkleScheme.rowBytes
      (Flock.Readable.openRow (Flock.Readable.copyOf p t ht w o g u) LANES)).data.toList := by
  have hs : 8 * (_root_.Flock.MerkleScheme.rowBytes
      (Flock.Readable.openRow (Flock.Readable.copyOf p t ht w o g u) LANES)).size = 128 * LANES := by
    rw [FlockVBridge.size_rowBytes, show (Flock.Readable.openRow (Flock.Readable.copyOf p t ht w o g u) LANES).size =
      LANES from FlockVBridge.Family.size_elems _ _ _]
    omega
  exact agrees_colsHold (b := _root_.Flock.MerkleScheme.rowBytes
      (Flock.Readable.openRow (Flock.Readable.copyOf p t ht w o g u) LANES)) hA hi hfx (by rw [hs]; exact hm)
    (by omega) (Flock.Readable.openIns_reads (Flock.Readable.copyOf p t ht w o g u) LANES H).2.2.2.2.2.1

end Held

section Calls

open FlockSoundness.Discharge.Hm (ev)
open Flock.Readable (copyOf openRow openIdx openSibs LevelCalls)

def readLin (w : Flock.Wit) (fs : List Flock.HmNets.Lin) : Msg := bytesOf (fs.map (ev fun q => w.hid.getD q false))

theorem readAt_eq_readLin (w : Flock.Wit) (ps : List ℕ) :
    readAt w ps = readLin w (ps.map fun q => ⟨[q], false⟩) := by
  simp [readAt, readLin, List.map_map, Function.comp_def, ev]

def copyForm (p : Flock.Readable.Parsed) (t : ℕ) (ht : t < p.sts.size) (o g u : ℕ) (cl : ℕ → ℕ)
    (f : Flock.HmNets.Lin) : Flock.HmNets.Lin :=
  ⟨(f.s.filter fun x => cl x ≤ p.sts[t].c.unit.constPos).map fun x => copyIdx p t ht o g u (cl x), f.c⟩

theorem ev_copyForm {p : Flock.Readable.Parsed} {t : ℕ} {ht : t < p.sts.size} {w : Flock.Wit} {o g u : ℕ}
    (cl : ℕ → ℕ) (f : Flock.HmNets.Lin) :
    ev (fun q => w.hid.getD q false) (copyForm p t ht o g u cl f) = ev (fun x => copyOf p t ht w o g u (cl x)) f := by
  obtain ⟨s, c⟩ := f
  simp only [copyForm, ev]
  induction s with
  | nil => rfl
  | cons x s ih =>
    by_cases hx : cl x ≤ p.sts[t].c.unit.constPos
    · simp only [List.filter_cons, hx, decide_true, ↓reduceIte, List.map_cons, List.foldr_cons, ih, copyOf_eq hx]
    · have h0 : copyOf p t ht w o g u (cl x) = false := by
        simp [copyOf, Flock.Readable.copyCol, hx]
      simp only [List.filter_cons, hx, decide_false, Bool.false_eq_true, ↓reduceIte, List.foldr_cons, ih, h0,
        Bool.false_xor]

theorem bytesOf_carries {z : Flock.HmNets.Var → Bool} {f : List Flock.HmNets.Lin} {b : ByteArray}
    (h : FlockVBridge.Carries z f b) : bytesOf (f.map (ev z)) = b.data.toList := by
  obtain ⟨hl, hb⟩ := h
  have hbl : b.data.toList.length = b.size := Array.length_toList
  have e : f.map (ev z) = (List.range (8 * b.data.toList.length)).map (bitL b.data.toList) :=
    List.ext_getElem (by simp [hl, hbl]) fun q h1 _ => by
      have hq : q < f.length := by simpa using h1
      rw [List.getElem_map, List.getElem_map, List.getElem_range, ← hb q hq]
      simp [FlockSoundness.Discharge.Hm.bit, hq]
  rw [e, bytesOf_bitL]

theorem levelCalls_read {p : Flock.Readable.Parsed} {t : ℕ} {ht : t < p.sts.size} {w : Flock.Wit}
    {o g u LANES H : ℕ} (h : LevelCalls (copyOf p t ht w o g u) LANES H) :
    (FlockVBridge.unitCalls LANES H).map (fun f =>
        (readLin w (f.1.map (copyForm p t ht o g u (FlockVBridge.unitCol LANES H))),
          readLin w (f.2.map (copyForm p t ht o g u (FlockVBridge.unitCol LANES H))))) =
      (FlockVBridge.rowCalls (_root_.Flock.MerkleScheme.rowBytes (openRow (copyOf p t ht w o g u) LANES))
        (openIdx (copyOf p t ht w o g u) LANES H) (openSibs (copyOf p t ht w o g u) LANES H)).map
        fun e => (e.1.data.toList, e.2.data.toList) := by
  have hr : ∀ (fs : List Flock.HmNets.Lin) (b : ByteArray),
      FlockVBridge.Carries (FlockVBridge.unitWires LANES H (copyOf p t ht w o g u)) fs b →
        readLin w (fs.map (copyForm p t ht o g u (FlockVBridge.unitCol LANES H))) = b.data.toList :=
    fun fs b hc => by
      rw [readLin, List.map_map, ← bytesOf_carries hc]
      congr 1
      exact List.map_congr_left fun l _ => ev_copyForm _ l
  unfold LevelCalls FlockVBridge.CallsAt at h
  generalize FlockVBridge.unitCalls LANES H = fs at h ⊢
  generalize FlockVBridge.rowCalls _ _ _ = es at h ⊢
  induction h with
  | nil => rfl
  | cons hfe _ ih => rw [List.map_cons, List.map_cons, ih, hr _ _ hfe.1, hr _ _ hfe.2]

/-- What is left of a level's log link is the layout's, that it records the gadgets, and the cut's, that each call the part's inner check logs is a climb call of some copy. -/
theorem levelCalls_recorded {L : Layout} {x : Msg} {s : ℕ} {p : Flock.Readable.Parsed} {t : ℕ}
    {ht : t < p.sts.size} {w : Flock.Wit} {o g u LANES H : ℕ} (h : LevelCalls (copyOf p t ht w o g u) LANES H)
    (hrec : ∀ f ∈ FlockVBridge.unitCalls LANES H,
      (readLin w (f.1.map (copyForm p t ht o g u (FlockVBridge.unitCol LANES H))),
        readLin w (f.2.map (copyForm p t ht o g u (FlockVBridge.unitCol LANES H)))) ∈ L.calls x s w) :
    ∀ e ∈ FlockVBridge.rowCalls (_root_.Flock.MerkleScheme.rowBytes (openRow (copyOf p t ht w o g u) LANES))
        (openIdx (copyOf p t ht w o g u) LANES H) (openSibs (copyOf p t ht w o g u) LANES H),
      (e.1.data.toList, e.2.data.toList) ∈ L.calls x s w := fun e he => by
  have hm := List.mem_map_of_mem (f := fun e : ByteArray × ByteArray => (e.1.data.toList, e.2.data.toList)) he
  rw [← levelCalls_read h] at hm
  obtain ⟨f, hf, hfe⟩ := List.mem_map.1 hm
  rw [← hfe]
  exact hrec f hf

end Calls

section Links

variable {Coin Col T : Type}

/-- A code fact about V\*'s staging, the layout and the goal at part `s`, none of it the circuit's semantics, which `allOuts_of_check` gives. -/
def PartReads (decode : Msg → Option Flock.Readable.Read) (L : Layout) (s : ℕ) (x xO : Msg)
    (G : Flock.Wit → List Msg → Prop) : Prop :=
  ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
    ∀ (w : Flock.Wit) (ms : List Msg), Agrees L x s w ms → Flock.Readable.AllOuts p w → G w ms

theorem partReads_of_copy {decode : Msg → Option Flock.Readable.Read} {L : Layout} {s : ℕ} {x xO : Msg}
    {G : Flock.Wit → List Msg → Prop} {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes decode xO = some p) (ht : p.tags.typed = false) (hpi : p.publicInputs = none)
    {t : ℕ} (htl : t < p.sts.size) {K : (ℕ → Bool) → Prop} (hK : Flock.Readable.UnitOuts p.sts[t].c K) {o g u : ℕ}
    (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    (h : ∀ (w : Flock.Wit) (ms : List Msg), Agrees L x s w ms → K (Flock.Readable.copyOf p t htl w o g u) → G w ms) :
    PartReads decode L s x xO G :=
  ⟨p, hx, ht, hpi, fun w ms hA hO => h w ms hA (hO t htl K hK o g u ho hg hu)⟩

theorem goal_of_partReads {DO : Flock.Deploy Coin Col T} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode) {L : Layout} {s : ℕ} {x xO : Msg}
    {w : Flock.Wit} {G : Flock.Wit → List Msg → Prop} (hck : Flock.check DO.parse xO w = true)
    (h : PartReads decode L s x xO G) {ms : List Msg} (hms : Agrees L x s w ms) : G w ms := by
  rw [hDO] at hck
  obtain ⟨p, hx, ht, hpi, hlink⟩ := h
  exact hlink w ms hms (Flock.Readable.allOuts_of_check hck hx ht hpi)

def PartReadsTo (H : Msg → Msg) (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
    (G : Msg → (ℕ → Coin) → Flock.Wit → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin), shows s x xO c →
    PartReads decode L s x xO fun w ms => G x c w ms ∨ PartColl H D L s x xO w

theorem goal_of_partReadsTo {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {decode : Msg → Option Flock.Readable.Read} {L : Layout} {shape : Form → Prop}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {G : Msg → (ℕ → Coin) → Flock.Wit → List Msg → Prop} {s : ℕ}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (h : PartReadsTo H D decode L shape shows G s) (x : Msg) (st : CompiledStatement) (hst : D.parse x = some st)
    (hf : shape st.form) (xO : Msg) (w : Flock.Wit) (hck : Flock.check DO.parse xO w = true) :
    (∀ (c : ℕ → Coin), shows s x xO c → ∀ ms : List Msg, Agrees L x s w ms → G x c w ms) ∨
      PartColl H D L s x xO w := by
  by_cases hfb : PartColl H D L s x xO w
  · exact Or.inr hfb
  · exact Or.inl fun c hc ms hms => (goal_of_partReads hDO hck (h x st hst hf xO c hc) hms).resolve_right hfb

theorem partChecks_of_partReadsTo {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (h : PartReadsTo H D decode L shape shows (fun x c _ ms => chk s x c ms) s) :
    PartChecks H D DO L shape shows chk s :=
  fun x st hst hf xO w hck _ => goal_of_partReadsTo hDO h x st hst hf xO w hck

theorem partLogs_of_partReadsTo {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {s : ℕ}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (h : PartReadsTo H D decode L shape shows (fun x c w ms => ∀ e ∈ logs s x c ms, e ∈ L.calls x s w) s) :
    PartLogs H D DO L shape shows logs s :=
  fun x st hst hf xO w hck _ => goal_of_partReadsTo hDO h x st hst hf xO w hck

theorem partReadsTo_of_resReads {H : Msg → Msg} {D : Deploy Coin} {decode : Msg → Option Flock.Readable.Read}
    {L : Layout} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} (h : ResReads D decode L shape shows chk s) :
    PartReadsTo H D decode L shape shows (fun x c _ ms => chk s x c ms) s := by
  intro x st hst hf xO c hc
  obtain ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, S, ns, nv, -, hp, hsh, hh, hpart, o, g, u, ho, hg, hu, hlink⟩ :=
    h x st hst hf xO c hc
  exact partReads_of_copy hx ht hpi htl (Or.inr (Or.inl ⟨S, ns, nv,
    Flock.Readable.resPartAny_of_resPart (Or.inl ⟨m, k, cl, j, sh, hp, hsh, hh, hpart⟩), rfl⟩)) ho hg hu
    fun w ms hms hout => Or.inl (hlink w ms hms hout)

def LinkedPart (H : Msg → Msg) (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg))
    (s : ℕ) : Prop :=
  PartReadsTo H D decode L shape shows (fun x c _ ms => chk s x c ms) s ∧
    (LogsNil D shape shows logs s ∨
      PartReadsTo H D decode L shape shows (fun x c w ms => ∀ e ∈ logs s x c ms, e ∈ L.calls x s w) s)

theorem partChecks_linked {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hP : ∀ s, LinkedPart H D decode L shape shows chk logs s) : ∀ s, PartChecks H D DO L shape shows chk s :=
  fun s => partChecks_of_partReadsTo hDO (hP s).1

theorem partLogs_linked {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hP : ∀ s, LinkedPart H D decode L shape shows chk logs s) : ∀ s, PartLogs H D DO L shape shows logs s :=
  fun s => (hP s).2.elim partLogs_of_shown_nil (partLogs_of_partReadsTo hDO)

theorem linkedPart_of_resReads {H : Msg → Msg} {D : Deploy Coin} {decode : Msg → Option Flock.Readable.Read}
    {L : Layout} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {s : ℕ} (h : ResReads D decode L shape shows chk s) (h0 : ∀ x c ms, logs s x c ms = []) :
    LinkedPart H D decode L shape shows chk logs s :=
  ⟨partReadsTo_of_resReads h, Or.inl fun x _ _ _ _ c _ ms => h0 x c ms⟩

end Links

section Bundles

variable {Coin W Col T : Type} [Inhabited Coin]

theorem vstarAccepts_of_linked {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {IV : Verifier Coin} {L : Layout} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {opening : Opening} {cmt dig : Msg → ℕ → Msg → Msg}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : ReadsThrough H D IV L.msgs L.calls) (hVL : VstarLen D)
    (hC : ∀ s, PartReadsTo H D decode L shape shows (fun x c _ ms => chk s x c ms) s)
    (hO : ∀ s, HeldOpens D DO L shape shows opening dig s) (hRC : ∀ s, RegClimbs H D DO L shape shows cmt dig s)
    (hS : Staged H D I L shape shows cmt) (hD : Decomp H D IV shape chk) : VStarAcceptsR H D I DO shape :=
  vstarAccepts_of_parts hR hVL (fun s => partChecks_of_partReadsTo hDO (hC s))
    (fun s => partOpens_of_climbs (hO s) (hRC s)) hS hD

theorem vstarParts_of_linked {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {opening : Opening} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {cmt dig : Msg → ℕ → Msg → Msg}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {pub : Msg → List (Msg × Msg)}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hP : ∀ s, LinkedPart H D decode L shape shows chk logs s)
    (hO : ∀ s, HeldOpens D DO L shape shows opening dig s) (hRC : ∀ s, RegClimbs H D DO L shape shows cmt dig s)
    (hS : Staged H D I L shape shows cmt) (hD : Decomp H D I.V shape chk)
    (hLD : LogDecompPub H D I.V shape logs pub) (hPR : PubReg H D shape pub)
    (hH : HeldPub D I L shape) : VStarParts H D I DO L opening shape :=
  ⟨shows, chk, cmt, logs, pub, partChecks_linked hDO hP, fun s => partOpens_of_climbs (hO s) (hRC s),
    partLogs_linked hDO hP, hS, hD, hLD, hPR, hH⟩

theorem vstarPartsStaging_of_linked {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {L : Layout} {opening : Opening} {decC : Msg → Msg → Coin} {shape : Form → Prop}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    {cmt dig : Msg → ℕ → Msg → Msg} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {pub : Msg → List (Msg × Msg)} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hP : ∀ s, LinkedPart H D decode L shape shows chk logs s)
    (hO : ∀ s, HeldOpens D DO L shape shows opening dig s) (hRC : ∀ s, RegClimbs H D DO L shape shows cmt dig s)
    (hV : VstarShows D I L shape decC shows cmt) (hD : Decomp H D I.V shape chk)
    (hLD : LogDecompPub H D I.V shape logs pub) (hPR : PubReg H D shape pub) (hH : HeldPub D I L shape) :
    VStarPartsStaging D I DO H L opening decC shape :=
  ⟨shows, chk, cmt, logs, pub, partChecks_linked hDO hP, fun s => partOpens_of_climbs (hO s) (hRC s),
    partLogs_linked hDO hP, hV, hD, hLD, hPR, hH⟩

end Bundles

section Deployed

variable {W : Type} (H : Msg → Msg) (I : Inner FlockDeploy.Coin W) (ivScope : Msg → Bool) (b : RecDeploy.Brings)

theorem vstarAccepts_deployRec_of_linked {L : Layout} (hb : b.rd = RecDeploy.readersOf L) {shape : Form → Prop}
    {shows : ℕ → Msg → Msg → (ℕ → FlockDeploy.Coin) → Prop} {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop}
    {opening : Opening} {cmt dig : Msg → ℕ → Msg → Msg}
    (hC : ∀ s, PartReadsTo H (RecDeploy.deployRec I.V ivScope b) RecDeploy.readOfO L shape shows
      (fun x c _ ms => chk s x c ms) s)
    (hO : ∀ s, HeldOpens (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      shape shows opening dig s)
    (hRC : ∀ s, RegClimbs H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      shape shows cmt dig s)
    (hS : Staged H (RecDeploy.deployRec I.V ivScope b) I L shape shows cmt)
    (hD : Decomp H (RecDeploy.deployRec I.V ivScope b) I.V shape chk) :
    VStarAcceptsR H (RecDeploy.deployRec I.V ivScope b) I (RecDeploy.deployZkRec Flock.Readable.relation) shape :=
  vstarAccepts_of_linked rfl (RecDeploy.readsThrough_deployRec H I.V ivScope b L hb)
    (RecDeploy.vstarLen_deployRec I.V ivScope b) hC hO hRC hS hD

theorem vstarParts_deployRec_of_linked {L : Layout} {opening : Opening} {shape : Form → Prop}
    {shows : ℕ → Msg → Msg → (ℕ → FlockDeploy.Coin) → Prop} {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop}
    {cmt dig : Msg → ℕ → Msg → Msg} {logs : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → List (Msg × Msg)}
    (hP : ∀ s, LinkedPart H (RecDeploy.deployRec I.V ivScope b) RecDeploy.readOfO L shape shows chk logs s)
    (hO : ∀ s, HeldOpens (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      shape shows opening dig s)
    (hRC : ∀ s, RegClimbs H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      shape shows cmt dig s)
    (hS : Staged H (RecDeploy.deployRec I.V ivScope b) I L shape shows cmt)
    (hD : Decomp H (RecDeploy.deployRec I.V ivScope b) I.V shape chk) {pub : Msg → List (Msg × Msg)}
    (hLD : LogDecompPub H (RecDeploy.deployRec I.V ivScope b) I.V shape logs pub)
    (hPR : PubReg H (RecDeploy.deployRec I.V ivScope b) shape pub)
    (hH : HeldPub (RecDeploy.deployRec I.V ivScope b) I L shape) :
    VStarParts H (RecDeploy.deployRec I.V ivScope b) I (RecDeploy.deployZkRec Flock.Readable.relation) L opening
      shape :=
  vstarParts_of_linked rfl hP hO hRC hS hD hLD hPR hH

theorem vstarPartsStaging_deployRec_of_linked {L : Layout} {opening : Opening} {decC : Msg → Msg → FlockDeploy.Coin}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → FlockDeploy.Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop} {cmt dig : Msg → ℕ → Msg → Msg}
    {logs : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → List (Msg × Msg)}
    (hP : ∀ s, LinkedPart H (RecDeploy.deployRec I.V ivScope b) RecDeploy.readOfO L shape shows chk logs s)
    (hO : ∀ s, HeldOpens (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      shape shows opening dig s)
    (hRC : ∀ s, RegClimbs H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      shape shows cmt dig s)
    (hV : VstarShows (RecDeploy.deployRec I.V ivScope b) I L shape decC shows cmt)
    (hD : Decomp H (RecDeploy.deployRec I.V ivScope b) I.V shape chk) {pub : Msg → List (Msg × Msg)}
    (hLD : LogDecompPub H (RecDeploy.deployRec I.V ivScope b) I.V shape logs pub)
    (hPR : PubReg H (RecDeploy.deployRec I.V ivScope b) shape pub)
    (hH : HeldPub (RecDeploy.deployRec I.V ivScope b) I L shape) :
    VStarPartsStaging (RecDeploy.deployRec I.V ivScope b) I (RecDeploy.deployZkRec Flock.Readable.relation) H L
      opening decC shape :=
  vstarPartsStaging_of_linked rfl hP hO hRC hV hD hLD hPR hH

end Deployed

end Claims.Recursion.VStar

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay)
open FlockDeploy (Coin)

/-- **Part `s`'s coins shown on `xO`, with `xO` staged**: `showsTr` at `untr` (which reads the transcript only) and
`StagedPart`. -/
def showsPart (untr : TrReader) (b : Brings) (s : ℕ) (x xO : Msg) (c : ℕ → Coin) : Prop :=
  showsTr untr s x xO c ∧ StagedPart b.st s x xO

theorem staged_deployRec_part {W : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (b : Brings) (untr : TrReader) (L : Claims.Recursion.VStar.Layout) (shape : Claims.Recursion.Form → Prop)
    (hrb : ReadsBack H I.V b shape untr)
    (hL : ∀ x st, parse x = some st → shape st.form → L.rounds x ≤ I.V.k (xIOf x)) :
    Claims.Recursion.VStar.Staged H (deployRec I.V ivScope b) I L shape (showsPart untr b) (cmtTr untr) := by
  intro x stx hst hf Ω P ω r hacc s hs
  obtain ⟨h1, h2⟩ := staged_deployRec H I ivScope b untr L shape hrb hL x stx hst hf Ω P ω r hacc s hs
  refine ⟨⟨h1, ?_⟩, h2⟩
  change (run H P (V I.V b.st) x ω r).out = .acc at hacc
  change s < nS x at hs
  obtain ⟨hnum, -⟩ := run_round H I.V b.st x P ω r hacc hs rfl rfl
  have hv : (Claims.Recursion.runVstar H (deployRec I.V ivScope b) I P x ω r).getD s [] =
      staged b.st x ((run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)) s := by
    show (vstar b.st x ((run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x))).getD s [] = _
    rw [vstar_getD b.st x _ hs]
    unfold vstarAt
    rw [ite_eq_left hnum]
  rw [hv]
  exact stagedPart_run H I.V b.st x P ω r hacc hs

theorem parse_of_setupO {xO : Msg} {i : RecIns} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (h : setupO xO = some (i, spec, sts)) :
    Flock.Readable.parseBytes readOfO xO = some { i.base.read with sts } := by
  obtain ⟨hins, hsu⟩ := setupO_some h
  have hr : readOfO xO = some i.base.read := by rw [readOfO, hins, Option.map_some]
  have hz := readOfO_zk hr
  unfold Flock.Readable.parseBytes
  rw [hr, Option.bind_some]
  unfold Flock.Readable.Read.parse Flock.Readable.Read.tablesOf
  rw [RecIns.setup_read hsu]
  simp only [hz, ↓reduceIte, RecIns.setup_shapeOk hsu hz, Option.map_some]

theorem RecIns.setup_checks {s : RecIns} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (h : s.setup = .ok (spec, sts)) :
    Flock.RecSetup.checks s.inner s.commits s.base.stmtTags s.base.read.I.tables sts s.base.read.I.circuitFile
      s.base.read.I.partition s.base.read.I.program s.base.registered s.part s.bindings s.rows = .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    obtain ⟨-, h2⟩ := Prod.mk.inj (Except.ok.inj h)
    subst h2
    assumption

theorem checks_stmt {inner : Option Flock.RecVStar.Inner} {commits : Option Flock.RecVStar.InnerCommits}
    {tags : Flock.Tags} {tables : List (String × ByteArray)} {sts : Array Flock.Stmt} {circuitB : ByteArray}
    {partition : Option Lean.Json} {program : Option ByteArray} {registered : Option (Array Flock.Registered.Port)}
    {part : Option ℕ} {bindings : Option VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (h : Flock.RecSetup.checks inner commits tags tables sts circuitB partition program registered part bindings rows =
      .ok ()) (t : ℕ) (ht : t < sts.size) :
    Flock.RecVStar.checkStmt inner tags tables sts[t] circuitB partition program registered part bindings rows =
      .ok () := by
  unfold Flock.RecSetup.checks at h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
    cases h1
  · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
    exact Flock.Readable.array_forM_ok h1 t ht

theorem RecRaw.ins_part {w : RecRaw} {s : RecIns} (h : w.ins = .ok s) : s.part = w.recPart := by
  unfold RecRaw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; rfl)

theorem insO_staged_part {st : Stage} {x : Msg} {ms : List (Msg × Out)} {s : ℕ} {i : RecIns}
    (h : insO (staged st x ms s) = some i) : i.part = some s := by
  have hne : staged st x ms s ≠ [] := fun he => by rw [he, insO_nil] at h; cases h
  obtain ⟨r, p, -, -, hd⟩ := staged_decode hne
  unfold insO at h
  rw [hd] at h
  simp only [Option.bind_eq_bind, Option.bind_some] at h
  split_ifs at h
  · simp at h
  have h' : (partRaw r st x ms s p).ins.toOption = some i := by simpa using h
  rw [RecRaw.ins_part (toOption_eq_some h')]
  rfl

theorem checkStmt_circuit {i : Flock.RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {registered : Option (Array Flock.Registered.Port)} {part : Option ℕ} {bindings : Option VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)}
    (h : Flock.RecVStar.checkStmt (some i) tags tables st circuitB partition program registered part bindings rows =
      .ok ()) :
    ∃ j f sv, part = some j ∧ bindings = some f ∧
      Flock.RecVStar.checkCircuit i j st.c (Flock.RecVStar.population st) circuitB f = .ok sv := by
  unfold Flock.RecVStar.checkStmt at h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
    cases h1
  · split at h
    · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
      cases h1
    · split at h
      · split at h
        · split at h
          · split at h
            · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
              cases h1
            obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
            obtain ⟨sv, hs, -⟩ := FlockSoundness.Refine.ebind_ok h
            exact ⟨_, _, sv, rfl, rfl, hs⟩
          · cases h
        · cases h
      · cases h

theorem stagedTable {i0 : Flock.RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {registered : Option (Array Flock.Registered.Port)} {s : ℕ} {bindings : Option VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)}
    (h : Flock.RecVStar.checkStmt (some i0) tags tables st circuitB partition program registered (some s) bindings
      rows = .ok ()) :
    ∃ ss sv, Flock.RecVStar.statements i0 = .ok ss ∧ ss[s]? = some sv ∧ sv.name = st.c.unit.name ∧
      Flock.RecVStar.checkPart sv st.c = .ok () ∧ Flock.RecVStar.checkParsed sv st.c = .ok () := by
  obtain ⟨j, f, sv, hj, -, hc⟩ := checkStmt_circuit h
  cases hj
  obtain ⟨hck, -, hq, hp⟩ := FlockSoundness.Discharge.Recursion.checkCircuit_ok hc
  obtain ⟨ss, hss, hs, hn, -, -⟩ := FlockSoundness.Discharge.Recursion.check_ok hck
  exact ⟨ss, sv, hss, hs, hn, hp, hq⟩

theorem stagedTables {st : Stage} {s : ℕ} {x xO : Msg} (h : StagedPart st s x xO) :
    ∃ (p : Flock.Readable.Parsed) (r : XR) (sv : VStar.Stmt),
      Flock.Readable.parseBytes readOfO xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
      readX x = some r ∧ (∃ ss, Flock.RecVStar.statements r.shape = .ok ss ∧ ss[s]? = some sv) ∧ 0 < p.sts.size ∧
      ∀ t (ht : t < p.sts.size), p.sts[t].c.unit.name = sv.name ∧ Flock.RecVStar.checkPart sv p.sts[t].c = .ok () ∧
        Flock.RecVStar.checkParsed sv p.sts[t].c = .ok () := by
  obtain ⟨ms, i, spec, sts, rfl, hsu, hck⟩ := h
  obtain ⟨hins, hset⟩ := setupO_some hsu
  obtain ⟨r, hr, hinner, -⟩ := insO_staged hins
  have hC := RecIns.setup_checks hset
  rw [hinner, insO_staged_part hins] at hC
  have hpi := publicInputs_none_of_scope hck
  have hsz : 0 < sts.size := by
    have hrd := RecIns.setup_read hset
    unfold Flock.Readable.Read.setup at hrd
    rw [hpi] at hrd
    exact (FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hrd).1
  obtain ⟨ss, sv, hss, hs, -, -, -⟩ := stagedTable (checks_stmt hC 0 hsz)
  refine ⟨{ i.base.read with sts }, r, sv, parse_of_setupO hsu,
    ((Flock.Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hck).1.2.1, hpi, hr, ⟨ss, hss, hs⟩, hsz, fun t ht => ?_⟩
  obtain ⟨ss', sv', hss', hs', hn, hp, hq⟩ := stagedTable (checks_stmt hC t ht)
  rw [hss] at hss'
  cases hss'
  rw [hs] at hs'
  cases hs'
  exact ⟨hn.symm, hp, hq⟩

theorem stagedReads {untr : TrReader} {b : Brings} {s : ℕ} {x xO : Msg} {c : ℕ → Coin}
    (h : showsPart untr b s x xO c) :
    ∃ (p : Flock.Readable.Parsed) (r : XR) (sv : VStar.Stmt),
      Flock.Readable.parseBytes readOfO xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
      readX x = some r ∧ (∃ ss, Flock.RecVStar.statements r.shape = .ok ss ∧ ss[s]? = some sv) ∧ 0 < p.sts.size ∧
      ∀ t (ht : t < p.sts.size), p.sts[t].c.unit.name = sv.name ∧ ∃ K, Flock.Readable.StmtUnit sv p.sts[t].c K := by
  obtain ⟨p, r, sv, hx, ht, hpi, hr, hss, hsz, htab⟩ := stagedTables h.2
  refine ⟨p, r, sv, hx, ht, hpi, hr, hss, hsz, fun t htl => ⟨(htab t htl).1, ?_⟩⟩
  obtain ⟨_, _, P⟩ := Flock.Readable.parseFactsU_of_parse hx ht hpi htl
  exact Flock.Readable.stmtUnit_of_check P (htab t htl).1 (htab t htl).2.1

theorem mapM_some_eq_map {α β : Type} {f : α → Option β} {g : α → β} :
    ∀ {l : List α} {l' : List β}, (∀ a ∈ l, ∀ b, f a = some b → b = g a) → l.mapM f = some l' → l' = l.map g
  | [], l', _, h => by simpa using h.symm
  | a :: l, l', hf, h => by
    rw [List.mapM_cons] at h
    cases ha : f a with
    | none => simp [ha] at h
    | some b =>
      cases hl : l.mapM f with
      | none => simp [ha, hl] at h
      | some bs =>
        simp only [ha, hl, Option.bind_some, Option.some.injEq, bind, pure] at h
        subst h
        rw [List.map_cons, hf a (by simp) b ha,
          mapM_some_eq_map (fun a' ha' => hf a' (List.mem_cons_of_mem _ ha')) hl]

theorem mapM_some_mem {α β : Type} {f : α → Option β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = some l' → ∀ a ∈ l, ∃ b, f a = some b
  | [], _, _, a, ha => by cases ha
  | a :: l, l', h, a', ha' => by
    rw [List.mapM_cons] at h
    cases ha : f a with
    | none => simp [ha] at h
    | some b =>
      cases hl : l.mapM f with
      | none => simp [ha, hl] at h
      | some bs =>
        rcases List.mem_cons.1 ha' with rfl | hm
        · exact ⟨b, ha⟩
        · exact mapM_some_mem hl a' hm

theorem findIdx_range_ofNat {N w k : ℕ}
    (h : ((Array.range N).map Int.ofNat).findIdx? (· == (w : Int)) = some k) : k = w := by
  obtain ⟨hk, hp, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 h
  simp at hp
  omega

theorem portCol_lt {c : Flock.Circuit} {N q i k : ℕ}
    (hl : c.leavesIn.getD 0 #[] = (Array.range N).map Int.ofNat) (h : portCol c 0 q i = some k) :
    k < (c.unit.inGroups.getD 0 default).col * Flock.WORD + 16 * N := by
  unfold portCol at h
  rw [hl] at h
  obtain ⟨k', hf, hk⟩ := Option.bind_eq_some_iff.1 h
  cases hk
  obtain ⟨hk', -, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hf
  simp at hk'
  omega

theorem leaves_le_constPos {tags : Flock.Tags} {c : Flock.Circuit} (P : FlockSoundness.ParseFacts tags c) {N : ℕ}
    (hl : c.leavesIn.getD 0 #[] = (Array.range N).map Int.ofNat) (hN0 : 0 < N) :
    (c.unit.inGroups.getD 0 default).col * Flock.WORD + 16 * N ≤ c.unit.constPos := by
  obtain ⟨ls, hcheck⟩ := P.check
  have C := FlockSoundness.check_facts P.typed hcheck
  obtain ⟨_, _, _, _, _, _, _, hb⟩ := FlockSoundness.unit_built P C
  obtain ⟨-, -, -, -, hcp, -, -, hin, -⟩ := Flock.Net.ofRows_ok hb
  have hN : N = (c.unit.inGroups.getD 0 default).bits.size := by
    have h0 : 0 < c.leavesIn.size := C.leaves.1 ▸ C.upv_pos
    have hs := C.leaves.2 0 C.upv_pos
    rw [getElem!_pos c.leavesIn 0 h0] at hs
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem h0, Option.getD_some] at hl
    rw [← hs, hl]
    simp
  obtain ⟨rest, hg0⟩ : ∃ rest, c.unit.inGroups.toList = c.unit.inGroups.getD 0 default :: rest := by
    have hs := C.unit_groups.1
    generalize c.unit.inGroups = gs at hs ⊢
    rcases gs with ⟨_ | ⟨g, l⟩⟩
    · simp at hs
    · exact ⟨l, by simp⟩
  have e := (FlockSoundness.Discharge.Aliased.portCols_one _).trans
    (FlockSoundness.Discharge.HiddenExec.portCols_16 _ _ C.unit_groups.2)
  have hmem : ((c.unit.inGroups.getD 0 default).col * Flock.WORD + 16 * (N - 1), 16) ∈
      (Flock.portCols c.unit.inGroups).toList := by
    rw [FlockSoundness.portCols_toList, hg0, List.flatMap_cons, e]
    refine List.mem_append_left _ (List.mem_map.2 ⟨N - 1, List.mem_range.2 ?_, rfl⟩)
    rw [Array.length_toList, ← hN]
    omega
  have hp : (c.unit.inGroups.getD 0 default).col * Flock.WORD + 16 * (N - 1) + 16 ≤ c.unit.inWords * Flock.WORD :=
    (FlockSoundness.nets_ok P c.unitNet).1.2.2 _ hmem
  omega

theorem portCol_of_leaves {c : Flock.Circuit} {N q i k : ℕ}
    (hl : c.leavesIn.getD 0 #[] = (Array.range N).map Int.ofNat) (h : portCol c 0 q i = some k) :
    k = (c.unit.inGroups.getD 0 default).col * Flock.WORD + 16 * (c.ports.extract 0 q).foldl (· + ·.words) 0 + i := by
  unfold portCol at h
  rw [hl] at h
  obtain ⟨k', hf, hk⟩ := Option.bind_eq_some_iff.1 h
  cases hk
  rw [findIdx_range_ofNat hf]
  omega

theorem rowHidden_of_leaves {c : Flock.Circuit} {N : ℕ} {nm : String} {n : ℕ} {ps : List ℕ}
    (hl : c.leavesIn.getD 0 #[] = (Array.range N).map Int.ofNat) (h : rowHidden c nm n = some ps) :
    ∃ (q : ℕ) (pt : Flock.Port), c.ports.findIdx? (·.name == nm) = some q ∧ c.ports[q]? = some pt ∧
      (0 < pt.bits → 16 * (c.ports.extract 0 q).foldl (· + ·.words) 0 + pt.bits ≤ 16 * N) ∧
      ps = (List.range pt.bits).map fun i => n / c.g * 2 ^ c.kLog + (c.unitRange.slot (n % c.g * c.upv) +
        ((c.unit.inGroups.getD 0 default).col * Flock.WORD + 16 * (c.ports.extract 0 q).foldl (· + ·.words) 0 +
          i)) := by
  unfold rowHidden at h
  obtain ⟨q, hq, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨pt, hpt, h⟩ := Option.bind_eq_some_iff.1 h
  refine ⟨q, pt, hq, hpt, fun h0 => ?_, mapM_some_eq_map (fun i _ b hb => ?_) h⟩
  · obtain ⟨b, hb⟩ := mapM_some_mem h (pt.bits - 1) (List.mem_range.2 (by omega))
    obtain ⟨k, hk, -⟩ := Option.map_eq_some_iff.1 hb
    have := portCol_lt hl hk
    rw [portCol_of_leaves hl hk] at this
    omega
  obtain ⟨k, hk, rfl⟩ := Option.map_eq_some_iff.1 hb
  rw [portCol_of_leaves hl hk]

theorem copyPs_one {p : Flock.Readable.Parsed} (h1 : p.sts.size = 1) (ht : 0 < p.sts.size) (o g base m : ℕ) :
    Claims.Recursion.VStar.copyPs p 0 ht o g 0 base m = (List.range m).map fun i =>
      o * 2 ^ p.sts[0].c.kLog + (p.sts[0].c.unitRange.slot (g * p.sts[0].c.upv) + (base + i)) := by
  simp [Claims.Recursion.VStar.copyPs, Claims.Recursion.VStar.copyIdx, h1]

theorem mem_partHolds {r : XR} {s j : ℕ} {ps : List ℕ} (h : (j, ps) ∈ partHolds r s) :
    ∃ sv c rs q k rep, ((Flock.RecVStar.statements r.shape).toOption.bind fun vs => vs[s]?) = some sv ∧
      r.w.parts[s]?.bind partCircuit = some c ∧ repSched r.shape = some rs ∧ q ∈ Flock.RecVStar.ownPorts sv ∧
      Flock.RecVStar.cIndex q.2.1 = some k ∧ msgRound rs rep k = some j ∧ rowHidden c q.1 rep = some ps := by
  unfold partHolds at h
  split at h
  · rename_i sv c rs hsv hc hrs
    obtain ⟨q, hq, h⟩ := List.mem_flatMap.1 h
    split at h
    · rename_i k hk
      obtain ⟨row, -, h⟩ := List.mem_filterMap.1 h
      obtain ⟨rep, -, h⟩ := Option.bind_eq_some_iff.1 h
      obtain ⟨j', hj, h⟩ := Option.bind_eq_some_iff.1 h
      obtain ⟨ps', hps, h⟩ := Option.bind_eq_some_iff.1 h
      cases h
      exact ⟨sv, c, rs, q, k, rep, hsv, hc, hrs, hq, hk, hj, hps⟩
    · cases h
  · cases h

theorem layoutOf_msgAt_mem {x : Msg} {s j : ℕ} {segs : List (Msg × List ℕ)}
    (h : layoutOf.msgAt x s (j + 1) = some segs) :
    ∃ r ps, readX x = some r ∧ (j, ps) ∈ partHolds r s ∧ segs = [([0], ps)] := by
  rw [layoutOf_msgAt_succ] at h
  obtain ⟨r, hr, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨e, he, rfl⟩ := Option.map_eq_some_iff.1 h
  have hj : e.1 = j := by simpa using List.find?_some he
  refine ⟨r, e.2, hr, ?_, rfl⟩
  rw [← hj]
  exact List.mem_of_find?_eq_some he

theorem layoutOf_fixedAt_of_msgAt {x : Msg} {s j : ℕ} {segs : List (Msg × List ℕ)}
    (h : layoutOf.msgAt x s (j + 1) = some segs) : layoutOf.fixedAt x (j + 1) = none := by
  obtain ⟨r, ps, hr, hmem, -⟩ := layoutOf_msgAt_mem h
  obtain ⟨-, -, rs, -, k, rep, -, -, hrs, -, -, hj, -⟩ := mem_partHolds hmem
  unfold msgRound at hj
  obtain ⟨t, ht, rfl⟩ := Option.map_eq_some_iff.1 hj
  obtain ⟨hlt, hflag⟩ : t < rs.size ∧ (rs.getD t default).2.2.2 = true := by
    simpa [Flock.RecVStar.commitRounds] using List.mem_of_getElem? ht
  have hmod : (rep * rs.size + t) % rs.size = t := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hlt]
  rw [layoutOf_fixedAt_succ, streamFixed, hr, Option.bind_some, hrs, Option.bind_some]
  simp only [hmod, hflag, Bool.true_eq_false, and_false, ↓reduceIte]

theorem partHolds_of_ownPorts_nil {r : XR} {s : ℕ} {sv : _root_.VStar.Stmt}
    (hsv : ((Flock.RecVStar.statements r.shape).toOption.bind fun vs => vs[s]?) = some sv)
    (h : Flock.RecVStar.ownPorts sv = []) : partHolds r s = [] := by
  unfold partHolds
  rw [hsv]
  split <;> simp_all

theorem agrees_layoutOf_of_ownPorts_nil {x : Msg} {r : XR} (hr : readX x = some r) {s : ℕ} {sv : _root_.VStar.Stmt}
    (hsv : ((Flock.RecVStar.statements r.shape).toOption.bind fun vs => vs[s]?) = some sv)
    (h : Flock.RecVStar.ownPorts sv = []) (w : Claims.Flock.Wit) (ms : List Msg) :
    Claims.Recursion.VStar.Agrees layoutOf x s w ms := by
  have hn : ∀ i, layoutOf.msgAt x s i = none := fun i => by
    cases i with
    | zero => rfl
    | succ j => rw [layoutOf_msgAt_succ, hr, Option.bind_some, partHolds_of_ownPorts_nil hsv h]; rfl
  intro i _ m hm
  simp [Claims.Recursion.VStar.Layout.held, hn i] at hm

theorem rawIns_fields {w : FlockDeploy.Raw} {i : FlockDeploy.Ins} (h : w.ins = .ok i) :
    i.read.zk = w.zk ∧ i.read.I.circuitFile = w.circuit ∧ i.read.I.tables = w.tables ∧ i.read.I.count = w.count ∧
      ∃ tags, Flock.Tags.byName? w.statement = some tags ∧ i.read.I.tags = tags := by
  unfold FlockDeploy.Raw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    cases h
    exact ⟨rfl, rfl, rfl, rfl, _, by assumption, rfl⟩

theorem rawIns_publicFile {w : FlockDeploy.Raw} {pf : ByteArray} {i i' : FlockDeploy.Ins} (h : w.ins = .ok i)
    (h' : ({ w with publicFile := pf } : FlockDeploy.Raw).ins = .ok i') :
    i'.read.tags = i.read.tags ∧ i'.read.I.circuitFile = i.read.I.circuitFile ∧ i'.read.I.tables = i.read.I.tables := by
  obtain ⟨hz, hc, htb, hn, tags, ht, htg⟩ := rawIns_fields h
  obtain ⟨hz', hc', htb', hn', tags', ht', htg'⟩ := rawIns_fields h'
  have e : tags' = tags := Option.some.inj (ht'.symm.trans ht)
  subst e
  refine ⟨?_, hc'.trans hc.symm, htb'.trans htb.symm⟩
  simp only [Flock.Readable.Read.tags, FlockSoundness.Discharge.ZkExec.zkTags, FlockSoundness.Discharge.ZkExec.zkCoins,
    hz, hz', htg, htg', hn, hn', hc, hc']

theorem tablesOf_of_parse {x : Msg} {p : Flock.Readable.Parsed}
    (h : Flock.Readable.parseBytes readOfO x = some p) :
    readOfO x = some p.toRead ∧ p.toRead.tablesOf = some p.sts := by
  unfold Flock.Readable.parseBytes at h
  obtain ⟨r, hr, h⟩ := Option.bind_eq_some_iff.1 h
  unfold Flock.Readable.Read.parse at h
  obtain ⟨sts, hs, rfl⟩ := Option.map_eq_some_iff.1 h
  exact ⟨hr, hs⟩

theorem partCircuit_staged {st : Stage} {s : ℕ} {x xO : Msg} (h : StagedPart st s x xO)
    {p : Flock.Readable.Parsed} (hp : Flock.Readable.parseBytes readOfO xO = some p) (ht : p.tags.typed = false)
    (hpi : p.publicInputs = none) {r : XR} (hr : readX x = some r) {c : Flock.Circuit}
    (hc : r.w.parts[s]?.bind partCircuit = some c) (h0 : 0 < p.sts.size) : p.sts[0].c = c := by
  obtain ⟨ms, i, spec, sts, rfl, hsu, -⟩ := h
  obtain ⟨hins, -⟩ := setupO_some hsu
  have hread : readOfO (staged st x ms s) = some i.base.read := by rw [readOfO, hins, Option.map_some]
  have hne : staged st x ms s ≠ [] := fun he => by rw [he, insO_nil] at hins; cases hins
  obtain ⟨r', pt, hr', hpt, hd⟩ := staged_decode hne
  rw [hr] at hr'
  cases hr'
  rw [hpt, Option.bind_some] at hc
  unfold partCircuit at hc
  obtain ⟨ins, hins', hc⟩ := Option.bind_eq_some_iff.1 hc
  unfold insO at hins
  rw [hd] at hins
  simp only [Option.bind_eq_bind, Option.bind_some] at hins
  split_ifs at hins
  · simp at hins
  have h' : (partRaw r st x ms s pt).ins.toOption = some i := by simpa using hins
  obtain ⟨htg, hcf, htb⟩ := rawIns_publicFile (toOption_eq_some hins') (RecRaw.ins_base (toOption_eq_some h'))
  obtain ⟨-, hcf', htb', -⟩ := rawIns_fields (toOption_eq_some hins')
  obtain ⟨hr0, hts⟩ := tablesOf_of_parse hp
  have he : p.toRead = i.base.read := Option.some.inj (hr0.symm.trans hread)
  obtain ⟨st0, hst0, hc0, -⟩ := Flock.Readable.table0 hts ht hpi
  rw [he, htg, hcf, htb, hcf', htb'] at hc0
  rw [toOption_eq_some hc] at hc0
  rw [Array.getElem?_eq_getElem h0] at hst0
  rw [Option.some.inj hst0]
  exact (Except.ok.inj hc0).symm

theorem layoutOf_msgAt_staged {st : Stage} {s j : ℕ} {x xO : Msg} (h : StagedPart st s x xO)
    {p : Flock.Readable.Parsed} (hp : Flock.Readable.parseBytes readOfO xO = some p) (ht : p.tags.typed = false)
    (hpi : p.publicInputs = none) (h0 : 0 < p.sts.size) (h1 : p.sts.size = 1) {N : ℕ}
    (hl : p.sts[0].c.leavesIn.getD 0 #[] = (Array.range N).map Int.ofNat) {segs : List (Msg × List ℕ)}
    (hm : layoutOf.msgAt x s (j + 1) = some segs) :
    ∃ (q : ℕ) (pt : Flock.Port) (rep : ℕ), p.sts[0].c.ports[q]? = some pt ∧
      (0 < pt.bits → 16 * (p.sts[0].c.ports.extract 0 q).foldl (· + ·.words) 0 + pt.bits ≤ 16 * N) ∧
      segs = [(([0] : Msg), Claims.Recursion.VStar.copyPs p 0 h0 (rep / p.sts[0].c.g) (rep % p.sts[0].c.g) 0
        ((p.sts[0].c.unit.inGroups.getD 0 default).col * Flock.WORD +
          16 * (p.sts[0].c.ports.extract 0 q).foldl (· + ·.words) 0) pt.bits)] := by
  obtain ⟨r, ps, hr, hmem, rfl⟩ := layoutOf_msgAt_mem hm
  obtain ⟨sv, c, rs, q, k, rep, -, hc, -, -, -, -, hrh⟩ := mem_partHolds hmem
  have hce := partCircuit_staged h hp ht hpi hr hc h0
  subst hce
  obtain ⟨qi, pt, -, hpt, htop, rfl⟩ := rowHidden_of_leaves hl hrh
  refine ⟨qi, pt, rep, hpt, htop, ?_⟩
  rw [copyPs_one h1 h0]

theorem agrees_layoutOf {untr : TrReader} {b : Brings} {s j : ℕ} {x xO : Msg} {c : ℕ → Coin}
    (hs : showsPart untr b s x xO c)
    {p : Flock.Readable.Parsed} (hp : Flock.Readable.parseBytes readOfO xO = some p) (h0 : 0 < p.sts.size)
    (h1 : p.sts.size = 1) {w : Claims.Flock.Wit} {ms : List Msg}
    (hA : Claims.Recursion.VStar.Agrees layoutOf x s w ms) (hj : j + 1 < layoutOf.rounds x)
    {segs : List (Msg × List ℕ)} (hm : layoutOf.msgAt x s (j + 1) = some segs) :
    ∃ (q : ℕ) (pt : Flock.Port) (rep : ℕ), p.sts[0].c.ports[q]? = some pt ∧ ∀ bs : ByteArray, 8 * bs.size = pt.bits →
      FlockVBridge.ColsHold (Flock.Readable.copyOf p 0 h0 w (rep / p.sts[0].c.g) (rep % p.sts[0].c.g) 0)
        ((p.sts[0].c.unit.inGroups.getD 0 default).col * Flock.WORD +
          16 * (p.sts[0].c.ports.extract 0 q).foldl (· + ·.words) 0) bs →
      ms.getD (j + 1) [] = 0 :: bs.data.toList := by
  obtain ⟨p', r, sv, hp', ht, hpi, -, -, -, htab⟩ := stagedTables hs.2
  rw [hp] at hp'
  cases hp'
  obtain ⟨u, -, -, -, hl, -⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0).2.2
  obtain ⟨_, P⟩ := Flock.Readable.parseFacts_of_parse hp ht hpi h0
    (FlockSoundness.Discharge.Recursion.checkParsed_flat (htab 0 h0).2.2)
  have hfx := layoutOf_fixedAt_of_msgAt hm
  obtain ⟨q, pt, rep, hpt, htop, hps⟩ := layoutOf_msgAt_staged hs.2 hp ht hpi h0 h1 hl hm
  refine ⟨q, pt, rep, hpt, fun bs hb hcol => ?_⟩
  rw [hps, ← hb] at hm
  rcases Nat.eq_zero_or_pos bs.size with hz | hz
  · have hm0 : layoutOf.msgAt x s (j + 1) = some [(([0] : Msg), Claims.Recursion.VStar.copyPs p 0 h0
        (rep / p.sts[0].c.g) (rep % p.sts[0].c.g) 0 0 (8 * bs.size))] := by
      rw [hm]
      simp [Claims.Recursion.VStar.copyPs, hz]
    exact Claims.Recursion.VStar.agrees_colsHold hA hj hfx hm0 (by omega) fun t ht => absurd ht (by omega)
  · have htop' := htop (by omega)
    have hN := leaves_le_constPos P hl (by omega)
    exact Claims.Recursion.VStar.agrees_colsHold hA hj hfx hm (by omega) hcol

end RecDeploy

namespace Claims.Recursion.VStar

section Staged

variable {W Col T : Type} (H : Msg → Msg) (I : Inner FlockDeploy.Coin W) (ivScope : Msg → Bool)
  (b : RecDeploy.Brings) (untr : RecDeploy.TrReader)

/-- What is left is the held tie (`agrees_colsHold`) and the goal from the outputs. -/
def StmtReadsTo (L : Layout) (shape : Form → Prop)
    (G : Msg → (ℕ → FlockDeploy.Coin) → Flock.Wit → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), (RecDeploy.deployRec I.V ivScope b).parse x = some st → shape st.form →
    ∀ (xO : Msg) (c : ℕ → FlockDeploy.Coin), RecDeploy.showsPart untr b s x xO c →
    ∀ (p : Flock.Readable.Parsed) (r : RecDeploy.XR) (sv : VStar.Stmt),
      Flock.Readable.parseBytes RecDeploy.readOfO xO = some p → RecDeploy.readX x = some r →
      (∃ ss, _root_.Flock.RecVStar.statements r.shape = .ok ss ∧ ss[s]? = some sv) → 0 < p.sts.size →
      ∀ (w : Flock.Wit) (ms : List Msg), Agrees L x s w ms → Flock.Readable.StmtCopies sv p w →
        G x c w ms ∨ PartColl H (RecDeploy.deployRec I.V ivScope b) L s x xO w

theorem partReadsTo_deployRec {L : Layout} {shape : Form → Prop}
    {G : Msg → (ℕ → FlockDeploy.Coin) → Flock.Wit → List Msg → Prop} {s : ℕ}
    (h : StmtReadsTo H I ivScope b untr L shape G s) :
    PartReadsTo H (RecDeploy.deployRec I.V ivScope b) RecDeploy.readOfO L shape (RecDeploy.showsPart untr b) G s := by
  intro x st hst hf xO c hc
  obtain ⟨p, r, sv, hx, ht, hpi, hr, hss, hsz, htab⟩ := RecDeploy.stagedReads hc
  exact ⟨p, hx, ht, hpi, fun w ms hA hO =>
    h x st hst hf xO c hc p r sv hx hr hss hsz w ms hA (Flock.Readable.stmtCopies_of_allOuts htab hO)⟩

def LevelLogReads (L : Layout) (shape : Form → Prop)
    (logs : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → List (Msg × Msg)) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), (RecDeploy.deployRec I.V ivScope b).parse x = some st → shape st.form →
    ∀ (xO : Msg) (c : ℕ → FlockDeploy.Coin), RecDeploy.showsPart untr b s x xO c →
    ∀ (p : Flock.Readable.Parsed) (r : RecDeploy.XR) (sv : VStar.Stmt),
      Flock.Readable.parseBytes RecDeploy.readOfO xO = some p → RecDeploy.readX x = some r →
      (∃ ss, _root_.Flock.RecVStar.statements r.shape = .ok ss ∧ ss[s]? = some sv) → 0 < p.sts.size →
      ∀ (w : Flock.Wit) (ms : List Msg), Agrees L x s w ms →
        (∃ lv LANES Ht n, (sv = .recOpen lv LANES Ht n ∨ sv = .recOpenStacked lv LANES Ht n) ∧
          (∀ e ∈ logs s x c ms, ∃ t, ∃ ht : t < p.sts.size, ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧
            g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
            ∃ e' ∈ FlockVBridge.rowCalls (_root_.Flock.MerkleScheme.rowBytes
                  (Flock.Readable.openRow (Flock.Readable.copyOf p t ht w o g u) LANES))
                (Flock.Readable.openIdx (Flock.Readable.copyOf p t ht w o g u) LANES Ht)
                (Flock.Readable.openSibs (Flock.Readable.copyOf p t ht w o g u) LANES Ht),
              e = (e'.1.data.toList, e'.2.data.toList)) ∧
          ∀ t (ht : t < p.sts.size) o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) → g < p.sts[t].c.g →
            u < p.sts[t].c.upv → ∀ f ∈ FlockVBridge.unitCalls LANES Ht,
              (readLin w (f.1.map (copyForm p t ht o g u (FlockVBridge.unitCol LANES Ht))),
                readLin w (f.2.map (copyForm p t ht o g u (FlockVBridge.unitCol LANES Ht)))) ∈ L.calls x s w) ∨
        PartColl H (RecDeploy.deployRec I.V ivScope b) L s x xO w

theorem stmtReadsTo_levelLogs {L : Layout} {shape : Form → Prop}
    {logs : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → List (Msg × Msg)} {s : ℕ}
    (h : LevelLogReads H I ivScope b untr L shape logs s) :
    StmtReadsTo H I ivScope b untr L shape (fun x c w ms => ∀ e ∈ logs s x c ms, e ∈ L.calls x s w) s := by
  intro x st hst hf xO c hc p r sv hx hr hss hsz w ms hA hC
  rcases h x st hst hf xO c hc p r sv hx hr hss hsz w ms hA with ⟨lv, LANES, Ht, n, hsv, hlog, hrec⟩ | hcoll
  · refine Or.inl fun e he => ?_
    obtain ⟨t, ht, o, g, u, ho, hg, hu, e', he', rfl⟩ := hlog e he
    have hL : Flock.Readable.LevelOuts (Flock.Readable.copyOf p t ht w o g u) LANES Ht := by
      rcases hsv with rfl | rfl
      · exact Flock.Readable.levelCopies ht ho hg hu hC
      · exact Flock.Readable.stackedCopies ht ho hg hu hC
    exact levelCalls_recorded hL.2 (hrec t ht o g u ho hg hu) e' he'
  · exact Or.inr hcoll

def StagedLinked (L : Layout) (shape : Form → Prop)
    (chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop)
    (logs : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → List (Msg × Msg)) (s : ℕ) : Prop :=
  StmtReadsTo H I ivScope b untr L shape (fun x c _ ms => chk s x c ms) s ∧
    (LogsNil (RecDeploy.deployRec I.V ivScope b) shape (RecDeploy.showsPart untr b) logs s ∨
      StmtReadsTo H I ivScope b untr L shape (fun x c w ms => ∀ e ∈ logs s x c ms, e ∈ L.calls x s w) s)

theorem linkedPart_deployRec {L : Layout} {shape : Form → Prop}
    {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop}
    {logs : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → List (Msg × Msg)} {s : ℕ}
    (h : StagedLinked H I ivScope b untr L shape chk logs s) :
    LinkedPart H (RecDeploy.deployRec I.V ivScope b) RecDeploy.readOfO L shape (RecDeploy.showsPart untr b) chk
      logs s :=
  ⟨partReadsTo_deployRec H I ivScope b untr h.1, h.2.imp id (partReadsTo_deployRec H I ivScope b untr)⟩

theorem vstarParts_deployRec_staged {L : Layout} {opening : Opening} {shape : Form → Prop}
    {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop} {dig : Msg → ℕ → Msg → Msg}
    {logs : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → List (Msg × Msg)}
    (hrb : RecDeploy.ReadsBack H I.V b shape untr)
    (hL : ∀ x st, RecDeploy.parse x = some st → shape st.form → L.rounds x ≤ I.V.k (RecDeploy.xIOf x))
    (hP : ∀ s, StagedLinked H I ivScope b untr L shape chk logs s)
    (hO : ∀ s, HeldOpens (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      shape (RecDeploy.showsPart untr b) opening dig s)
    (hRC : ∀ s, RegClimbs H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      shape (RecDeploy.showsPart untr b) (RecDeploy.cmtTr untr) dig s)
    (hD : Decomp H (RecDeploy.deployRec I.V ivScope b) I.V shape chk) {pub : Msg → List (Msg × Msg)}
    (hLD : LogDecompPub H (RecDeploy.deployRec I.V ivScope b) I.V shape logs pub)
    (hPR : PubReg H (RecDeploy.deployRec I.V ivScope b) shape pub)
    (hH : HeldPub (RecDeploy.deployRec I.V ivScope b) I L shape) :
    VStarParts H (RecDeploy.deployRec I.V ivScope b) I (RecDeploy.deployZkRec Flock.Readable.relation) L opening
      shape :=
  vstarParts_deployRec_of_linked H I ivScope b (fun s => linkedPart_deployRec H I ivScope b untr (hP s)) hO hRC
    (RecDeploy.staged_deployRec_part H I ivScope b untr L shape hrb hL) hD hLD hPR hH

theorem vstarAccepts_deployRec_staged {L : Layout} (hb : b.rd = RecDeploy.readersOf L) {opening : Opening}
    {shape : Form → Prop} {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop} {dig : Msg → ℕ → Msg → Msg}
    (hrb : RecDeploy.ReadsBack H I.V b shape untr)
    (hL : ∀ x st, RecDeploy.parse x = some st → shape st.form → L.rounds x ≤ I.V.k (RecDeploy.xIOf x))
    (hC : ∀ s, StmtReadsTo H I ivScope b untr L shape (fun x c _ ms => chk s x c ms) s)
    (hO : ∀ s, HeldOpens (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      shape (RecDeploy.showsPart untr b) opening dig s)
    (hRC : ∀ s, RegClimbs H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      shape (RecDeploy.showsPart untr b) (RecDeploy.cmtTr untr) dig s)
    (hD : Decomp H (RecDeploy.deployRec I.V ivScope b) I.V shape chk) :
    VStarAcceptsR H (RecDeploy.deployRec I.V ivScope b) I (RecDeploy.deployZkRec Flock.Readable.relation) shape :=
  vstarAccepts_deployRec_of_linked H I ivScope b hb (fun s => partReadsTo_deployRec H I ivScope b untr (hC s)) hO hRC
    (RecDeploy.staged_deployRec_part H I ivScope b untr L shape hrb hL) hD

end Staged

end Claims.Recursion.VStar
