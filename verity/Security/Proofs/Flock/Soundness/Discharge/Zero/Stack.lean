import Proofs.Flock.Soundness.Discharge.Zero.Delta

/-!
# A VU's stacked inputs that read the zero (`hZero`, e2e, untyped statements)

An untyped statement's audit unit is a VU: `upv` copies of the unit net, stacked (`Rows.stack`), so stacked input `x` is
copy `x / W`'s input at slot offset `x % W`, with `W = inWords · WORD` the unit's input columns (`unitRows_nIn`). At VU
`g` that copy sits at slot `g · upv + x / W`.

- `ZeroIn c x`: stacked input `x` is a port bit that reads the zero (`ZeroAt`). Decidable, and the same at every VU, so a
  program can wire exactly these inputs to its zero gate.
- `zeroAt_vu`: `ZeroAt` reads a slot only through its copy index, so `ZeroIn c x` is `ZeroAt` at slot `g · upv + x / W`
  for every VU `g`.
- `flatSrcAt_zeroIn`: at such an input, Δ's source (`flatSrcAt`, which `delta_flat` gives an untyped statement too) is a
  forced-zero row: the slot's own or the first slot's.
- `withZero`: a wiring of the units' stacked inputs that puts exactly the `ZeroIn` inputs on the zero gate
  (`withZero_eq_zer`), the zero clause of the program's input wiring.
-/

open Flock

namespace FlockSoundness.Discharge.Zero

/-- **Slot offset `off` is a bit of one of the unit's input ports**, not an input padding row. -/
def InPort (c : Circuit) (off : ℕ) : Prop := off ∈ portBits c.unit.inGroups

/-- **Stacked input `x` of a VU reads the zero**: copy `x / W` of the unit, at slot offset `x % W`
(`W = inWords · WORD`), is a port bit that reads the zero. -/
def ZeroIn (c : Circuit) (x : ℕ) : Prop :=
  InPort c (x % (c.unit.inWords * WORD)) ∧ ZeroAt c (x / (c.unit.inWords * WORD)) (x % (c.unit.inWords * WORD))

instance instDecidablePredNatZeroIn (c : Circuit) : DecidablePred (ZeroIn c) := fun _ => by
  unfold ZeroIn InPort ZeroAt ZeroPort
  infer_instance

/-- `ZeroAt` reads a slot only through its copy index. -/
theorem zeroAt_mod (c : Circuit) (s₀ off : ℕ) : ZeroAt c (s₀ % c.upv) off ↔ ZeroAt c s₀ off := by
  simp only [ZeroAt, ZeroPort, Nat.mod_mod]

/-- **`ZeroAt` is the same at every VU**: at slot `g · upv + i` as at `i`. -/
theorem zeroAt_vu (c : Circuit) (g i off : ℕ) : ZeroAt c (g * c.upv + i) off ↔ ZeroAt c i off := by
  rw [← zeroAt_mod c (g * c.upv + i), ← zeroAt_mod c i, Nat.add_comm, Nat.add_mul_mod_self_right]

/-- **At VU `g`, a stacked input that reads the zero is a port bit of its copy's slot that reads the zero.** -/
theorem zeroIn_vu {c : Circuit} {x : ℕ} (h : ZeroIn c x) (g : ℕ) :
    InPort c (x % (c.unit.inWords * WORD)) ∧
      ZeroAt c (g * c.upv + x / (c.unit.inWords * WORD)) (x % (c.unit.inWords * WORD)) :=
  ⟨h.1, (zeroAt_vu c g _ _).2 h.2⟩

/-- **Δ copies a forced-zero row into a stacked input that reads the zero**, at every VU `g`: the input's own slot's
(a zero leaf) or the first slot's (a cut bit past 16). -/
theorem flatSrcAt_zeroIn {c : Circuit} {x : ℕ} (h : ZeroIn c x) (g : ℕ) :
    flatSrcAt c (g * c.upv + x / (c.unit.inWords * WORD)) (x % (c.unit.inWords * WORD)) =
        c.unitRange.slot (g * c.upv + x / (c.unit.inWords * WORD)) + c.unit.useful ∨
      flatSrcAt c (g * c.upv + x / (c.unit.inWords * WORD)) (x % (c.unit.inWords * WORD)) =
        c.unitRange.slot 0 + c.unit.useful :=
  flatSrcAt_zero (zeroIn_vu h g).2

/-- **A wiring of the units' stacked inputs with the zero clause**: an input that reads the zero goes to the gate `zer`,
any other as `rest` wires it. -/
def withZero (c : Circuit) {n m k : ℕ} (zer : Fin k) (rest : Fin n → Fin m → Fin k) (u : Fin n) (x : Fin m) : Fin k :=
  if ZeroIn c x.val then zer else rest u x

/-- **Exactly the inputs that read the zero are on `zer`**, when `rest` keeps off it. -/
theorem withZero_eq_zer (c : Circuit) {n m k : ℕ} {zer : Fin k} {rest : Fin n → Fin m → Fin k}
    (hrest : ∀ u x, rest u x ≠ zer) (u : Fin n) (x : Fin m) : withZero c zer rest u x = zer ↔ ZeroIn c x.val := by
  unfold withZero
  split
  · exact iff_of_true rfl ‹_›
  · exact iff_of_false (hrest u x) ‹_›

/-- No input is on a gate `one` that is not `zer` and `rest` keeps off. -/
theorem withZero_ne (c : Circuit) {n m k : ℕ} {zer one : Fin k} {rest : Fin n → Fin m → Fin k} (hz1 : zer ≠ one)
    (hrest : ∀ u x, rest u x ≠ one) (u : Fin n) (x : Fin m) : withZero c zer rest u x ≠ one := by
  unfold withZero
  split
  · exact hz1
  · exact hrest u x

end FlockSoundness.Discharge.Zero
