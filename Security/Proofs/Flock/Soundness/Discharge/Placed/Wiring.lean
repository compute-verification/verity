import Proofs.Flock.Soundness.Discharge.Placed.Setup

/-!
# The input gates of a circuit's program

`uniform` wires each unit's input columns to input gates. Here the gates are indexed by source: gate 0 is the constant,
gate 1 the zero, and unit `u`'s `B` sources (its row's message bits) are gates `2 + u·B + b`. An input column reads
source `src x` (`none`: a forced zero). Two input columns of one unit share a gate exactly when they read one source, or
both a zero (`insOf_eq_iff`); no two units share a source gate.
-/

namespace FlockSoundness.Discharge.Placed

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

variable {nU B src}

theorem srcGate_inj {u : ℕ} (hu : u < nU) (s s' : Option (Fin B)) :
    srcGate nU B hu s = srcGate nU B hu s' ↔ s = s' := by
  cases s <;> cases s' <;> simp [srcGate, zeroGate, Fin.ext_iff] <;> omega

theorem insOf_of_lt {u : ℕ} (hu : u < nU) (x : Fin nIn) : insOf nU B src u x = srcGate nU B hu (src x) := by
  simp [insOf, hu]

/-- **An input reads the zero exactly when it has no source.** -/
theorem insOf_eq_zero_iff {u : ℕ} (hu : u < nU) (x : Fin nIn) :
    insOf nU B src u x = zeroGate nU B ↔ src x = none := by
  rw [insOf_of_lt hu, show zeroGate nU B = srcGate nU B hu none from rfl, srcGate_inj]

/-- **Two inputs of one unit share a gate exactly when they read one source**, or both none. -/
theorem insOf_eq_iff {u : ℕ} (hu : u < nU) (x x' : Fin nIn) :
    insOf nU B src u x = insOf nU B src u x' ↔ src x = src x' := by
  rw [insOf_of_lt hu, insOf_of_lt hu, srcGate_inj]

/-- **Inputs of two units share a gate only at the zero**: a source gate is one unit's. -/
theorem insOf_eq_unit {u u' : ℕ} (hu : u < nU) (hu' : u' < nU) {x x' : Fin nIn}
    (h : insOf nU B src u x = insOf nU B src u' x') (hx : src x ≠ none) : u = u' := by
  rw [insOf_of_lt hu, insOf_of_lt hu'] at h
  obtain ⟨b, hb⟩ := Option.ne_none_iff_exists'.1 hx
  rw [hb] at h
  cases hs' : src x' with
  | none => rw [hs'] at h; simp [srcGate, zeroGate, Fin.ext_iff] at h; omega
  | some b' =>
    rw [hs'] at h
    simp only [srcGate, Fin.ext_iff] at h
    by_contra hne
    rcases Nat.lt_or_gt_of_ne hne with hl | hl
    · have : u * B + b.val < u' * B := by
        calc u * B + b.val < u * B + B := by omega
          _ = (u + 1) * B := by ring
          _ ≤ u' * B := Nat.mul_le_mul_right _ hl
      omega
    · have : u' * B + b'.val < u * B := by
        calc u' * B + b'.val < u' * B + B := by omega
          _ = (u' + 1) * B := by ring
          _ ≤ u * B := Nat.mul_le_mul_right _ hl
      omega

/-- **The program of a circuit's units on source-indexed input gates**: `nU` units of the circuit's template instance rows, input column
`x` of each on its source's gate (`insOf`), the constant on `oneGate`. -/
noncomputable def circuitProg (c : Flock.Circuit) (hU : Built c.unit) (nU B : ℕ)
    (src : Fin (circuitRows c hU).nIn → Option (Fin B)) : Prog (gates (inGates nU B) (circuitRows c hU) nU) nU :=
  uniform (inGates nU B) (circuitRows c hU) (insOf nU B src) (oneGate nU B) (insOf_ne_one nU B src) nU

/-- Every unit of the circuit's program has the circuit's template instance rows. -/
theorem rows_circuitProg (c : Flock.Circuit) (hU : Built c.unit) (nU B : ℕ)
    (src : Fin (circuitRows c hU).nIn → Option (Fin B)) (u : Fin nU) :
    (circuitProg c hU nU B src).rows u = circuitRows c hU :=
  rows_uniform _ _ _ _ _ nU u

end FlockSoundness.Discharge.Placed
