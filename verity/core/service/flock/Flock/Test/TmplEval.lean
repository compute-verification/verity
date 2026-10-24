/-
`Flock.TmplEval.SatAt` used as a piece fact and composed: a copy of `Flock.Test.Tmpl`'s AND Definition that holds computes
the AND of its bound inputs, from its rows alone; and a Definition placing two copies of it on its own inputs and XORing
them gets `0`, from its row and its calls' copies (`rows_sat` gives `SatAt` at every copy of a unit's root).
-/
import Flock.TmplEval

namespace Flock.Test.TmplEval

open Flock.Tmpl Flock.TmplEval

/-- `Flock.Test.Tmpl`'s AND (`DEF 1 2 1 0`, `0 1 2 1 1`): one own column, two inputs, its row `[in 0]·[in 1]`. -/
def andDef : Def := { cols := 1, ins := 2, rows := #[(0, #[1], #[2])], calls := #[] }

/-- `Flock.Test.Tmpl`'s pair (`DEF 3 2 1 1`, `2 2 1 0 1 5`, `CALL 0 2 0 1 1 2 3 0 1 0`): two AND copies at own columns 0
and 1, each bound to the pair's two inputs, and own column 2 their XOR times the constant. -/
def pairCall : Call :=
  { def_ := 0, count := 2, base := 0, stride := 1, segs := #[{ len := 2, at_ := 3, step := 0, reps := 1, rstride := 0 }] }

def pairDef : Def := { cols := 3, ins := 2, rows := #[(2, #[1, 0], #[5])], calls := #[pairCall] }

def defs : Array Def := #[andDef, pairDef]

/-- **A piece fact from its rows**: a copy of the AND Definition at `o`, bound to `a` and `b`, that holds computes
`a ∧ b`. -/
theorem and_sat {k : Nat} {v : Nat → Bool} {fuel o a b : Nat} (h : SatAt defs k v (fuel + 1) 0 o [a, b]) :
    v o = (v a && v b) := by
  obtain ⟨df, hdf, hrows, -⟩ := h
  obtain rfl : df = andDef := by simpa [defs] using hdf.symm
  simpa [andDef, xorL, xorList, Def.ref] using hrows (0, #[1], #[2]) (by simp [andDef])

/-- **Composed**: a copy of the pair that holds, at a constant `1`, has own column 2 equal to `0`: its two AND copies
agree (`and_sat` at each), and its row XORs them. -/
example {k : Nat} {v : Nat → Bool} {fuel o a b : Nat} (h : SatAt defs k v (fuel + 2) 1 o [a, b]) (hk : v k = true) :
    v (o + 2) = false := by
  obtain ⟨df, hdf, hrows, hcalls⟩ := h
  obtain rfl : df = pairDef := by simpa [defs] using hdf.symm
  have hc := hcalls pairCall (by simp [pairDef])
  have hb : ∀ j, pairCall.bound (pairDef.ref o [a, b] k) j = [a, b] := fun _ => rfl
  have h0 : v o = (v a && v b) := and_sat (o := o) (by have := hc 0 (by decide); rw [hb] at this; exact this)
  have h1 : v (o + 1) = (v a && v b) := and_sat (o := o + 1) (by have := hc 1 (by decide); rw [hb] at this; exact this)
  have hr := hrows (2, #[1, 0], #[5]) (by simp [pairDef])
  simp only [pairDef, xorL, xorList, Def.ref] at hr
  simp at hr
  rw [hr, h0, h1, hk]
  cases v a <;> cases v b <;> rfl

end Flock.Test.TmplEval
