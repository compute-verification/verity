import Proofs.Flock.Bridge.Targets
import Proofs.Flock.Bridge.Lowers
import Proofs.Flock.Claims.Sampling.EndToEndDraw

/-!
# The draw at `stmtOf` and `partOf`: one class, `subset kd`, `n` replay units

`EndToEndDraw.choose_le_missAt` reads three facts of the statement and partition it is used at; at the bridge's
`stmtOf P kd desc q roots` and `partOf P` each holds by their definitions:

* (i) `stmtOf_subsetDesign`: the design is `EndToEndDraw.SubsetDesign` with the same `kd`;
* (ii) `partOf_classes`: the partition has one class;
* (iii) `partRus : PartRusTarget`: that class has `n` replay units, one per unit of `P`.

From them, `pa : PaTarget` (the design's `P_a` is the subset miss over `n`) and `count : CountTarget` (outside `RelS` at
`δ = C(n − K₀, kd)/C(n, kd)`, at least `min K₀ (n − kd + 1)` units are faulty).
-/

namespace Flock.Bridge

open FlockSoundness.Audit

/-- (i) `stmtOf`'s design is a `SubsetDesign` with its `kd`. -/
theorem stmtOf_subsetDesign {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    (kd : ℕ) (desc : Claims.Msg) (q : Claims.Sampling.Query) (roots : List Claims.Msg) :
    Claims.Sampling.EndToEndDraw.SubsetDesign (stmtOf P kd desc q roots).design kd := rfl

/-- (ii) `partOf P` has one class. -/
theorem partOf_classes {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) :
    (partOf P).classes = [[(List.finRange n).map fun u => [unitGates P u]]] := rfl

/-- (iii) `partOf P`'s one class has `n` replay units. -/
theorem partRus : PartRusTarget := fun P => by
  rw [rus_partOf, List.length_map, List.length_finRange]

theorem pa : PaTarget := fun P kd B hB => by
  rw [Claims.Sampling.EndToEndDraw.Pa_subset (d := designOf kd) (kd := kd) rfl (partOf_classes P) B, partRus P,
    Claims.Sampling.EndToEndDraw.window_zero fun v hv => Finset.mem_range.1 (hB hv)]

/-- `C(a, k) < C(b, k)` for `a < b` and `1 ≤ k ≤ b`. -/
theorem choose_lt_choose {a b k : ℕ} (hab : a < b) (hk : 1 ≤ k) (hkb : k ≤ b) : a.choose k < b.choose k := by
  obtain ⟨b, rfl⟩ : ∃ b', b = b' + 1 := ⟨b - 1, by omega⟩
  obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
  rw [Nat.choose_succ_succ']
  have h1 := Nat.choose_pos (n := b) (k := k) (by omega)
  have h2 := Nat.choose_le_choose (k + 1) (show a ≤ b by omega)
  omega

theorem count : CountTarget := by
  intro C n P kd K₀ hkd hkn H Cm desc q roots w hrel
  set F := Claims.Sampling.faulty H Cm (stmtOf P kd desc q roots) (partOf P) w
  have hF : F ⊆ Finset.range n := by
    intro v hv
    have := (Finset.mem_filter.1 hv).1
    rwa [partRus P] at this
  have hrel' : ¬ (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) < (designOf kd).Pa (partOf P) F) := hrel
  have hle := not_lt.1 hrel'
  rw [pa P kd F hF] at hle
  have hpos : (0 : ℝ) < (n.choose kd : ℝ) := by exact_mod_cast Nat.choose_pos hkn
  have hle' : (n - F.card).choose kd ≤ (n - K₀).choose kd := by
    exact_mod_cast (div_le_div_iff_of_pos_right hpos).1 hle
  by_contra hlt
  rw [not_le, lt_min_iff] at hlt
  exact absurd hle' (not_le.2 (choose_lt_choose (by omega) hkd (by omega)))

end Flock.Bridge
