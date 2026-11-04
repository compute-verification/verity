import Properties.Flock.Circuit.Keys
import SampledProofs.Bits

/-!
# Bits' source keys are the bridge's (`SampledProofs.Bits.srcKey`, `keyIdx`)

`SampledProofs.Bits.srcAt` and `Aliased.srcAt` name the same column Δ copies into a unit input, but for one: a unit
input bit with no leaf, cut or wire is its own free source in Bits (`Src.free k p t`) and the one fixed column 0 in the
bridge (`Src.lit 0`). `srcOf` maps Bits' sources to the bridge's, a free one to `.lit 0`:

* `srcAt_eq`, `srcKey_eq`: the bridge's source is `srcOf` of Bits'.
* `keyIdx_eq`: at an input whose Bits source isn't free, Bits' key is the bridge's. Other inputs' sources don't enter:
  `srcOf` identifies only free sources.
-/

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Discharge

/-- **Bits' source as the bridge's**: constructor for constructor, a free source the fixed column 0. -/
def srcOf : SampledProofs.Bits.Src → Aliased.Src
  | .zero => .zero
  | .msg p byte bit => .msg p byte bit
  | .wire net slot col => .wire net slot col
  | .free _ _ _ => .lit 0
  | .pad k off => .pad k off

private theorem starts_eq (c : Flock.Circuit) : SampledProofs.Bits.starts c = Typed.starts c := rfl

private theorem portBits_eq (gs : Array Flock.Group) : SampledProofs.Bits.portBits gs = portBits gs := rfl

private theorem portOf_eq (pc : Array (ℕ × ℕ)) (off : ℕ) : SampledProofs.Bits.portOf pc off = portOf pc off := rfl

/-- **The bridge's source of copy `k`'s port `p`, bit `t`, is Bits'**, under `srcOf`. -/
theorem srcAt_eq (c : Flock.Circuit) (k p t : ℕ) :
    Aliased.srcAt c k p t = srcOf (SampledProofs.Bits.srcAt c k p t) := by
  unfold Aliased.srcAt SampledProofs.Bits.srcAt
  rw [starts_eq]
  by_cases h1 : p < (c.unit.inGroups.getD 0 default).bits.size
  · by_cases h2 : (c.leavesIn.getD k #[])[p]! < 0
    · simp only [h1, h2, ite_true]
      rfl
    · simp only [h1, h2, ite_true, ite_false]
      split_ifs <;> rfl
  · simp only [h1, ite_false]
    split
    · split_ifs <;> rfl
    · split <;> rfl

/-- **The bridge's key of stacked input `x` is Bits'**, under `srcOf`. -/
theorem srcKey_eq (c : Flock.Circuit) (x : ℕ) : Aliased.srcKey c x = srcOf (SampledProofs.Bits.srcKey c x) := by
  unfold Aliased.srcKey SampledProofs.Bits.srcKey
  simp only [portBits_eq, portOf_eq]
  by_cases h : x % (c.unit.inWords * WORD) ∈ portBits c.unit.inGroups
  · simp only [h, ite_true]
    exact srcAt_eq _ _ _ _
  · simp only [h, ite_false]
    rfl

/-- `srcOf` identifies only free sources. -/
theorem srcOf_eq_iff {s : SampledProofs.Bits.Src} (hs : ∀ k p t, s ≠ .free k p t) (s' : SampledProofs.Bits.Src) :
    srcOf s' = srcOf s ↔ s' = s := by
  refine ⟨fun h => ?_, fun h => h ▸ rfl⟩
  cases s <;> cases s' <;> simp_all [srcOf]

theorem srcOf_eq_zero {s : SampledProofs.Bits.Src} : srcOf s = .zero ↔ s = .zero := by
  cases s <;> simp [srcOf]

private theorem finRange_map_val (n : ℕ) : (List.finRange n).map Fin.val = List.range n :=
  List.ext_getElem (by simp) fun i _ _ => by simp

/-- **Bits' key of input `x` is the bridge's**, where its Bits source isn't free. -/
theorem keyIdx_eq (c : Flock.Circuit) {B x : ℕ} (hx : x < B)
    (hs : ∀ k p t, SampledProofs.Bits.srcKey c x ≠ .free k p t) :
    SampledProofs.Bits.keyIdx c B x = (Aliased.keyIdx c B ⟨x, hx⟩).map Fin.val := by
  unfold SampledProofs.Bits.keyIdx Aliased.keyIdx
  have hz : Aliased.srcKey c x = .zero ↔ SampledProofs.Bits.srcKey c x = .zero := by
    rw [srcKey_eq, srcOf_eq_zero]
  by_cases h0 : SampledProofs.Bits.srcKey c x = .zero
  · rw [ite_eq_left_of_eq_true _ _ (eq_true h0), ite_eq_left_of_eq_true _ _ (eq_true (hz.2 h0))]
    rfl
  · rw [ite_eq_right_of_eq_false _ _ (eq_false h0), ite_eq_right_of_eq_false _ _ (eq_false (mt hz.1 h0)),
      ← finRange_map_val, List.find?_map]
    congr 1
    refine List.find?_congr fun i _ => ?_
    simp only [Function.comp_apply, srcKey_eq c, srcOf_eq_iff hs]

end Flock.Bridge
