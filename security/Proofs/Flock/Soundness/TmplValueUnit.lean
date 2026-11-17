module

public import Proofs.Flock.Soundness.TmplValue
public import Proofs.Flock.VBridge.Unit
public import Flock.TmplComputes
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.MeasureTheory.Integral.Bochner.Basic
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
import Proofs.Flock.Soundness.ExecRows
import VCVio.Prelude.Core

/-! A templated unit of a partition computes its Definitions at its local evaluation (`unit_eval`, `unit_computes`); no
verifier path reads this file. -/

@[expose] public section

namespace FlockSoundness.TmplValue

open Flock Flock.Tmpl FlockSoundness.Audit

variable {C : Verity.Boolean.Circuit} {m : ℕ} {P : Partition C m} {u : Fin m} {t : Tmpl} {net : Net} {k : ℕ}

theorem copyVal_const (h : Flock.Tmpl.unroll t = .ok net) (I : P.IsRowsUnit u ((Rows.ofNet (unroll_built h)).stack k))
    (X : Fin C.N → Bool) {i : ℕ} (hi : i < k) :
    FlockVBridge.copyVal I X hi t.constPos = X (I.wire ((Rows.ofNet (unroll_built h)).stack k).one) := by
  have h' := h
  unfold Flock.Tmpl.unroll at h'
  obtain ⟨_, -, hn⟩ := Flock.TmplOrder.bind_ok h'
  obtain ⟨-, -, -, -, hc, -⟩ := Net.ofRows_ok hn
  have h1 : (Rows.ofNet (unroll_built h)).one.val = t.constPos := (Rows.ofNet_one _).trans hc
  rw [← h1, FlockVBridge.copyVal, FlockVBridge.colVal_fin]
  exact (FlockVBridge.sat_copy I X hi).2

theorem unit_eval (hwf : t.wellFormed = true) (ha : t.apart = true) (ho : t.ordered = true)
    (h : Flock.Tmpl.unroll t = .ok net) (hu : net.unitLog ≤ 32)
    (I : P.IsRowsUnit u ((Rows.ofNet (unroll_built h)).stack k)) (X : Fin C.N → Bool) {i : ℕ} (hi : i < k)
    {c : Call} {j f e o : ℕ} {inp : List ℕ} (hc : c ∈ (t.defs.getD t.root default).calls.toList)
    (hb : t.inCols ≤ c.base) (hj : j < c.count)
    (hw : Within t.defs t.constPos t.defs.size c.def_ (c.base + j * c.stride)
      (c.bound ((t.defs.getD t.root default).ref 0 [] t.constPos) j) f e o inp) :
    ∀ y < (t.defs.getD e default).cols, FlockVBridge.copyVal I X hi (o + y) =
      t.evalDef e (inp.map (FlockVBridge.copyVal I X hi))
        (X (I.wire ((Rows.ofNet (unroll_built h)).stack k).one)) y := by
  have hs := (FlockVBridge.sat_copy I X hi).1
  have he : (fun c : Fin (Rows.ofNet (unroll_built h)).w => FlockVBridge.copyVal I X hi c.val) =
      fun c => P.localEval X u (I.wire ((Rows.ofNet (unroll_built h)).stackCol k hi c)) :=
    funext fun c => by rw [FlockVBridge.copyVal, FlockVBridge.colVal_fin]
  rw [← he] at hs
  intro y hy
  rw [← copyVal_const h I X hi]
  exact unroll_eval hwf ha ho h hu hs hc hb hj hw y hy

theorem computes_of_eval {e o : ℕ} {inp : List ℕ} {f : ℕ → Bool} (hlen : inp.length = (t.defs.getD e default).ins)
    (hev : ∀ y < (t.defs.getD e default).cols, f (o + y) = t.evalDef e (inp.map f) true y)
    {outs : List ℕ} {spec : (ℕ → Bool) → List Bool} (hD : t.DefComputes e outs spec)
    (hout : ∀ q < outs.length, outs.getD q 0 < (t.defs.getD e default).cols) :
    ∀ q < outs.length, f (o + outs.getD q 0) = (spec fun x => f (inp.getD x 0)).getD q false := by
  intro q hq
  rw [hev _ (hout q hq)]
  have hinp : inp.map f = (List.range (t.defs.getD e default).ins).map fun x => f (inp.getD x 0) := by
    apply List.ext_getElem
    · simp [hlen]
    · intro n h1 _
      have hn : n < inp.length := by simpa using h1
      simp only [List.getElem_map, List.getElem_range]
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hn]
      rfl
  rw [hinp]
  exact hD _ q hq

theorem unit_computes (hwf : t.wellFormed = true) (ha : t.apart = true) (ho : t.ordered = true)
    (h : Flock.Tmpl.unroll t = .ok net) (hu : net.unitLog ≤ 32)
    (I : P.IsRowsUnit u ((Rows.ofNet (unroll_built h)).stack k)) (X : Fin C.N → Bool) {i : ℕ} (hi : i < k)
    (hone : X (I.wire ((Rows.ofNet (unroll_built h)).stack k).one) = true)
    {c : Call} {j f e o : ℕ} {inp : List ℕ} (hc : c ∈ (t.defs.getD t.root default).calls.toList)
    (hb : t.inCols ≤ c.base) (hj : j < c.count)
    (hw : Within t.defs t.constPos t.defs.size c.def_ (c.base + j * c.stride)
      (c.bound ((t.defs.getD t.root default).ref 0 [] t.constPos) j) f e o inp)
    {outs : List ℕ} {spec : (ℕ → Bool) → List Bool} (hD : t.DefComputes e outs spec)
    (hout : ∀ q < outs.length, outs.getD q 0 < (t.defs.getD e default).cols) :
    ∀ q < outs.length, FlockVBridge.copyVal I X hi (o + outs.getD q 0) =
      (spec fun x => FlockVBridge.copyVal I X hi (inp.getD x 0)).getD q false :=
  computes_of_eval (f := FlockVBridge.copyVal I X hi)
    (Flock.TmplValue.within_ins (Flock.TmplValue.wf_getD hwf) hw
      (Flock.TmplValue.call_bound_length (Flock.TmplValue.wf_getD hwf t.root) hc _ j))
    (fun y hy => by
      have hv := unit_eval hwf ha ho h hu I X hi hc hb hj hw y hy
      rwa [hone] at hv) hD hout

end FlockSoundness.TmplValue
