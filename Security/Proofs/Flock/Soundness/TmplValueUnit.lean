import Proofs.Flock.Soundness.TmplValue
import Proofs.Flock.VBridge.Unit

/-!
# A templated unit of a partition computes its Definitions at its local evaluation (PRF-133)

Let unit `u` of a partition be an instance of `k` stacked copies of the audit's `Rows` of the net `Tmpl.unroll` builds
(`Rows.ofNet`, `Rows.stack`), as a session's instance unit is of its circuit's unit net. At any transcript `X`, copy `i`'s
column values (`FlockVBridge.copyVal`, the unit's local evaluation at the copy's wires) satisfy the template's rows
(`FlockVBridge.sat_copy`), and its constant column carries the transcript's constant (`copyVal_const`). So at every
column of every copy of a Definition placed under one of the root's calls past the input rows, at any depth, the unit's
local evaluation is that Definition's value (`Tmpl.evalDef`) on the local evaluation at its bound inputs (`unit_eval`).
No verifier path reads this file.
-/

namespace FlockSoundness.TmplValue

open Flock Flock.Tmpl FlockSoundness.Audit

variable {C : Audit.Circuit} {m : ℕ} {P : Partition C m} {u : Fin m} {t : Tmpl} {net : Net} {k : ℕ}

/-- A copy's constant column carries the transcript's value at the stack's constant. -/
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

/-- **A templated unit computes its Definitions**: in a unit that is an instance of `k` stacked copies of the audit's
`Rows` of `Tmpl.unroll`'s net, copy `i`'s local evaluation, at every column of every copy of a Definition placed under one
of the root's calls past the input rows, at any depth, is that Definition's value on the local evaluation at its bound
inputs, the constant the transcript's at the stack's constant. -/
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

end FlockSoundness.TmplValue
