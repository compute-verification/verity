import Proofs.Flock.Bridge.Targets

/-!
# The bridge's `P_a`

`designOf kd` is one rule, `subset kd` with every proof unit of a drawn replay unit checked, and `partOf P` is one class
of one stratum of `n` replay units. So `Design.Pa` is that class's `OneStage.Pa [n] (.subset kd)` on `B`'s slots in
`[0, n)`, which are all of `B` when `B ⊆ range n`: `P_a(B) = C(n − |B|, kd)/C(n, kd)` (`pa_eq_choose`, `paTarget`).
-/

namespace Flock.Bridge

/-- **`P_a` at the bridge**: on `partOf P`, `designOf kd` misses a set `B` of slots with probability
`C(n − |B|, kd)/C(n, kd)`. -/
theorem pa_eq_choose {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (kd : ℕ)
    (B : Finset ℕ) (hB : B ⊆ Finset.range n) :
    (designOf kd).Pa (partOf P) B = ((n - B.card).choose kd : ℝ) / (n.choose kd : ℝ) := by
  have hl : ((List.finRange n).map fun u => [unitGates P u]).length = n := by simp
  have hw : Claims.Sampling.window 0 n B = B := by
    ext v
    simp only [Claims.Sampling.window, Finset.mem_image, Finset.mem_filter, Nat.sub_zero, zero_add]
    exact ⟨fun ⟨w, ⟨hw, _⟩, he⟩ => he ▸ hw, fun hv => ⟨v, ⟨hv, Nat.zero_le _, Finset.mem_range.1 (hB hv)⟩, rfl⟩⟩
  change ((([((List.finRange n).map fun u => [unitGates P u]).length].sum -
      (Claims.Sampling.window 0 ([(List.finRange n).map fun u => [unitGates P u]].flatten.length) B).card).choose kd :
        ℕ) : ℝ) / (([((List.finRange n).map fun u => [unitGates P u]).length].sum.choose kd : ℕ) : ℝ) * 1 = _
  simp only [List.flatten_cons, List.flatten_nil, List.append_nil, List.sum_cons, List.sum_nil, add_zero, hl, hw,
    mul_one]

theorem paTarget : PaTarget := fun P kd B hB => pa_eq_choose P kd B hB

end Flock.Bridge
