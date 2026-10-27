import Proofs.Flock.Soundness.Discharge.Aliased.Keys

/-!
# Δ's source keys with public inputs

With public input ports (§16.10, `HmIn`), a public leaf is a zero leaf of `leaves_in`, so its `srcKey` is the zero; but
`HmIn.delta` drops its Δ entries (`HmIn.publicCol`) and an in region pins it to its instance's public word
(`Layout.public_pinned`). `srcKeyP` keys it by that word instead: bit `t` of public word `q` (`KeyP.pub`), so that
`keyIdxP`, `keyIdx` at these keys, gives each unit one input gate per public word bit, whose value the instance's
public file fixes. Off a public leaf it is `srcKey`'s, and with no public leaf map (`HmIn.parse` at a circuit without
public inputs) `keyIdxP` is `keyIdx` (`keyIdxP_empty`).
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Aliased

/-- **A stacked input's key with public inputs**: `srcKey`'s, or bit `t` of public word `q` at a public leaf. -/
inductive KeyP
  | src (s : Src)
  | pub (q t : ℕ)
  deriving DecidableEq

/-- Stacked input `x`'s public word, `-1` off a public leaf: at a port bit of the leaf group, leaf `p` (its port) of
copy `x / W` (`W = inWords · WORD`) in the public leaf map. -/
def pubLeaf (c : Circuit) (pi : HmIn.Ports) (x : ℕ) : ℤ :=
  if x % (c.unit.inWords * WORD) ∈ portBits c.unit.inGroups ∧
      portOf (portCols c.unit.inGroups) (x % (c.unit.inWords * WORD)) < (c.unit.inGroups.getD 0 default).bits.size then
    (pi.leaves.getD (x / (c.unit.inWords * WORD)) #[]).getD
      (portOf (portCols c.unit.inGroups) (x % (c.unit.inWords * WORD))) (-1)
  else -1

/-- **Stacked input `x`'s key with public inputs**: at a public leaf, bit `t` of its public word (`t` its port bit),
else `srcKey`'s. -/
def srcKeyP (c : Circuit) (pi : HmIn.Ports) (x : ℕ) : KeyP :=
  if 0 ≤ pubLeaf c pi x then
    .pub (pubLeaf c pi x).toNat (x % (c.unit.inWords * WORD) -
      (portCols c.unit.inGroups)[portOf (portCols c.unit.inGroups) (x % (c.unit.inWords * WORD))]!.1)
  else .src (srcKey c x)

/-- **Equal codes are equal keys, with public inputs**: `keyIdx` at `srcKeyP`. -/
def keyIdxP (c : Circuit) (pi : HmIn.Ports) (n : ℕ) (x : Fin n) : Option (Fin n) :=
  if srcKeyP c pi x.val = .src .zero then none
  else (List.finRange n).find? fun i => srcKeyP c pi i.val = srcKeyP c pi x.val

theorem srcKeyP_of_neg {c : Circuit} {pi : HmIn.Ports} {x : ℕ} (h : pubLeaf c pi x < 0) :
    srcKeyP c pi x = .src (srcKey c x) := by
  unfold srcKeyP
  rw [ite_eq_right (show ¬ (0 ≤ pubLeaf c pi x) by omega)]

theorem srcKeyP_of_nonneg {c : Circuit} {pi : HmIn.Ports} {x : ℕ} (h : 0 ≤ pubLeaf c pi x) :
    ∃ t, srcKeyP c pi x = .pub (pubLeaf c pi x).toNat t := by
  unfold srcKeyP
  rw [ite_eq_left h]
  exact ⟨_, rfl⟩

theorem pubLeaf_empty {c : Circuit} {pi : HmIn.Ports} (h : pi.leaves = #[]) (x : ℕ) : pubLeaf c pi x = -1 := by
  unfold pubLeaf
  split <;> simp [h]

/-- **A public key is no zero**: the zero key is `srcKey`'s, off a public leaf. -/
theorem srcKeyP_eq_zero {c : Circuit} {pi : HmIn.Ports} {x : ℕ} :
    srcKeyP c pi x = .src .zero ↔ pubLeaf c pi x < 0 ∧ srcKey c x = .zero := by
  by_cases h : 0 ≤ pubLeaf c pi x
  · obtain ⟨t, ht⟩ := srcKeyP_of_nonneg h
    rw [ht]
    exact iff_of_false KeyP.noConfusion (fun h' => absurd h'.1 (by omega))
  · rw [srcKeyP_of_neg (by omega), KeyP.src.injEq]
    exact ⟨fun h' => ⟨by omega, h'⟩, fun h' => h'.2⟩

theorem keyIdxP_eq_none {c : Circuit} {pi : HmIn.Ports} {n : ℕ} (x : Fin n) :
    keyIdxP c pi n x = none ↔ srcKeyP c pi x.val = .src .zero := by
  unfold keyIdxP
  split
  · exact iff_of_true rfl ‹_›
  · refine iff_of_false (fun h => ?_) ‹_›
    rw [List.find?_eq_none] at h
    exact h x (List.mem_finRange x) (by simp)

theorem keyIdxP_eq {c : Circuit} {pi : HmIn.Ports} {n : ℕ} {x x' : Fin n} (h : keyIdxP c pi n x = keyIdxP c pi n x') :
    srcKeyP c pi x.val = srcKeyP c pi x'.val := by
  by_cases hz : srcKeyP c pi x.val = .src .zero
  · have := (keyIdxP_eq_none x).2 hz
    rw [h] at this
    rw [hz, (keyIdxP_eq_none x').1 this]
  · have hz' : srcKeyP c pi x'.val ≠ .src .zero := fun hz' =>
      hz ((keyIdxP_eq_none x).1 (h.trans ((keyIdxP_eq_none x').2 hz')))
    unfold keyIdxP at h
    rw [ite_eq_right hz, ite_eq_right hz'] at h
    rcases hf : (List.finRange n).find? (fun i => srcKeyP c pi i.val = srcKeyP c pi x.val) with _ | i
    · rw [List.find?_eq_none] at hf
      exact absurd (by simp) (hf x (List.mem_finRange x))
    · rw [hf] at h
      have h1 := List.find?_some hf
      have h2 := List.find?_some h.symm
      simp only [decide_eq_true_eq] at h1 h2
      exact h1.symm.trans h2

/-- The first input with an input's key has that key. -/
theorem keyIdxP_some {c : Circuit} {pi : HmIn.Ports} {m : ℕ} {x b : Fin m} (h : keyIdxP c pi m x = some b) :
    srcKeyP c pi b.val = srcKeyP c pi x.val := by
  unfold keyIdxP at h
  split at h
  · cases h
  · simpa using List.find?_some h

/-- **Without a public leaf map, the keys are `srcKey`'s**: `keyIdxP` is `keyIdx`. -/
theorem keyIdxP_empty {c : Circuit} {pi : HmIn.Ports} (h : pi.leaves = #[]) (n : ℕ) : keyIdxP c pi n = keyIdx c n := by
  have hk : ∀ x, srcKeyP c pi x = .src (srcKey c x) := fun x =>
    srcKeyP_of_neg (by rw [pubLeaf_empty h]; omega)
  funext x
  simp only [keyIdxP, keyIdx, hk, KeyP.src.injEq]

end FlockSoundness.Discharge.Aliased
