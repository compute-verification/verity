import Properties.Flock.Circuit.Rows
import Properties.Flock.Circuit.Keys

/-!
# The circuit's program on Δ's source keys

`keyProg c hU n` is the program of `n` units of the circuit `c`'s template instance rows (`circuitRows`, the unit
net's rows stacked `units_per_vu` times), each unit's input `x` on the input gate of its key (`Aliased.keyIdx`):
`uniform` on the source-indexed input gates (`inGates`, `insOf`, `circuitProg`). Moved unchanged from
`Proofs.Flock.Soundness.Discharge.Placed.{Setup,Wiring,Keyed}`; the lemmas about them stay there.
-/

namespace FlockSoundness.Discharge.Placed

/-- **A template instance's rows for a circuit**: `units_per_vu` copies of the unit net's rows, one shared constant
(`Rows.stack`). -/
noncomputable def circuitRows (c : Flock.Circuit) (hU : Built c.unit) : Rows := (Rows.ofNet hU).stack c.upv

/-- The gates of `n` units with rows `R` after `k` input gates. -/
def gates (k : ℕ) (R : Rows) : ℕ → ℕ
  | 0 => k
  | n + 1 => gates k R n + R.nComp

theorem le_gates (k : ℕ) (R : Rows) : ∀ n, k ≤ gates k R n
  | 0 => le_rfl
  | n + 1 => (le_gates k R n).trans (Nat.le_add_right _ _)

/-- **`n` units of rows `R` on `k` input gates**: unit `u`'s input columns read the input gates `ins u`, its constant the
input gate `one`, which none of them reads. No unit reads another's gates. -/
def uniform (k : ℕ) (R : Rows) (ins : ℕ → Fin R.nIn → Fin k) (one : Fin k) (hone : ∀ u c, ins u c ≠ one) :
    (n : ℕ) → Prog (gates k R n) n
  | 0 => .inputs k
  | n + 1 => .snoc (uniform k R ins one hone n) R (fun c => Fin.castLE (le_gates k R n) (ins n c))
      (Fin.castLE (le_gates k R n) one) (fun c h => hone n c (Fin.castLE_injective _ h))

variable (nU B : ℕ) {nIn : ℕ} (src : Fin nIn → Option (Fin B))

/-- The input gates: the constant, the zero, and `B` sources per unit for `nU` units. -/
abbrev inGates : ℕ := 2 + nU * B

/-- The constant's gate. -/
def oneGate : Fin (inGates nU B) := ⟨0, by show 0 < 2 + nU * B; omega⟩

/-- The zero's gate. -/
def zeroGate : Fin (inGates nU B) := ⟨1, by show 1 < 2 + nU * B; omega⟩

theorem source_lt {u : ℕ} (hu : u < nU) (b : Fin B) : 2 + u * B + b.val < inGates nU B := by
  have : u * B + b.val < nU * B := by
    calc u * B + b.val < u * B + B := by omega
      _ = (u + 1) * B := by ring
      _ ≤ nU * B := Nat.mul_le_mul_right _ hu
  show 2 + u * B + b.val < 2 + nU * B
  omega

/-- Unit `u`'s gate of a source: `2 + u·B + b`, or the zero's for none. -/
def srcGate {u : ℕ} (hu : u < nU) : Option (Fin B) → Fin (inGates nU B)
  | none => zeroGate nU B
  | some b => ⟨2 + u * B + b.val, source_lt nU B hu b⟩

/-- **Unit `u`'s input column `x`'s gate**: its source's (`srcGate`). -/
def insOf (u : ℕ) (x : Fin nIn) : Fin (inGates nU B) :=
  if hu : u < nU then srcGate nU B hu (src x) else zeroGate nU B

theorem srcGate_val_ge {u : ℕ} (hu : u < nU) (s : Option (Fin B)) : 1 ≤ (srcGate nU B hu s).val := by
  cases s
  · simp [srcGate, zeroGate]
  · simp only [srcGate]; omega

/-- No input column reads the constant's gate. -/
theorem insOf_ne_one (u : ℕ) (x : Fin nIn) : insOf nU B src u x ≠ oneGate nU B := fun h => by
  have : 1 ≤ (insOf nU B src u x).val := by
    unfold insOf
    split
    · exact srcGate_val_ge nU B _ _
    · simp [zeroGate]
  rw [h] at this
  simp [oneGate] at this

/-- **The program of a circuit's units on source-indexed input gates**: `nU` units of the circuit's template instance rows, input column
`x` of each on its source's gate (`insOf`), the constant on `oneGate`. -/
noncomputable def circuitProg (c : Flock.Circuit) (hU : Built c.unit) (nU B : ℕ)
    (src : Fin (circuitRows c hU).nIn → Option (Fin B)) : Prog (gates (inGates nU B) (circuitRows c hU) nU) nU :=
  uniform (inGates nU B) (circuitRows c hU) (insOf nU B src) (oneGate nU B) (insOf_ne_one nU B src) nU

/-- **The circuit's program on Δ's source keys**: `n` units of the circuit's template instance rows, input `x` of each
on the gate of its key (`Aliased.keyIdx`), the zero's gate when it reads the zero. -/
noncomputable abbrev keyProg (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ) :
    Prog (gates (inGates n (circuitRows c hU).nIn) (circuitRows c hU) n) n :=
  circuitProg c hU n (circuitRows c hU).nIn (Aliased.keyIdx c (circuitRows c hU).nIn)

end FlockSoundness.Discharge.Placed
