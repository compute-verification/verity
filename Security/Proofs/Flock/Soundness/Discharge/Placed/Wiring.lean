import Proofs.Flock.Soundness.Discharge.Placed.Setup
import Properties.Flock.Circuit.Keyed

/-!
# The input gates of a circuit's program

`uniform` wires each unit's input columns to input gates. Here the gates are indexed by source: gate 0 is the constant,
gate 1 the zero, and unit `u`'s `B` sources (its row's message bits) are gates `2 + u·B + b`. An input column reads
source `src x` (`none`: a forced zero). Two input columns of one unit share a gate exactly when they read one source, or
both a zero (`insOf_eq_iff`); no two units share a source gate.
-/

namespace FlockSoundness.Discharge.Placed

variable (nU B : ℕ) {nIn : ℕ} (src : Fin nIn → Option (Fin B))

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

/-- Every unit of the circuit's program has the circuit's template instance rows. -/
theorem rows_circuitProg (c : Flock.Circuit) (hU : Built c.unit) (nU B : ℕ)
    (src : Fin (circuitRows c hU).nIn → Option (Fin B)) (u : Fin nU) :
    (circuitProg c hU nU B src).rows u = circuitRows c hU :=
  rows_uniform _ _ _ _ _ nU u

end FlockSoundness.Discharge.Placed
