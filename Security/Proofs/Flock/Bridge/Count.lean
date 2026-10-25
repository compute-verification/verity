import Proofs.Flock.Bridge.Targets
import Proofs.Flock.Claims.Sampling.EndToEndDraw

/-!
# The bridge's slot count, `P_a` and count lemma

`partOf P` is one class of `n` replay units (`partOf_classes`, `rus_length_partOf`), and `designOf kd` is a
`SubsetDesign` (`designOf_subset`), so `EndToEndDraw`'s bounds apply at `stmtOf`. From them: `PartRusTarget`,
`PaTarget` (`P_a(B) = C(n − |B|, kd)/C(n, kd)`), and `CountTarget`: a witness outside `RelS` at
`δ = C(n − K₀, kd)/C(n, kd)` has at least `min K₀ (n − kd + 1)` faulty units, because `C(·, kd)` is strictly increasing
from `kd` on (`choose_lt_of_lt`).
-/

namespace Flock.Bridge

open FlockSoundness.Audit

theorem designOf_subset (kd : ℕ) : Claims.Sampling.EndToEndDraw.SubsetDesign (designOf kd) kd := rfl

theorem partOf_classes {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) :
    (partOf P).classes = [[(List.finRange n).map fun u => [unitGates P u]]] := rfl

theorem rus_length_partOf {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) : (partOf P).rus.length = n := by
  simp [partOf, Claims.Sampling.Partition.rus]

theorem partRus : PartRusTarget := fun P => rus_length_partOf P

theorem pa_partOf {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (kd : ℕ) (B : Finset ℕ) (hB : B ⊆ Finset.range n) :
    (designOf kd).Pa (partOf P) B = ((n - B.card).choose kd : ℝ) / (n.choose kd : ℝ) := by
  rw [Claims.Sampling.EndToEndDraw.Pa_subset (designOf_subset kd) (partOf_classes P), rus_length_partOf P,
    Claims.Sampling.EndToEndDraw.window_zero fun v hv => Finset.mem_range.1 (hB hv)]

theorem pa : PaTarget := fun P kd B hB => pa_partOf P kd B hB

/-- `C(·, k)` is strictly increasing from `k` on. -/
theorem choose_lt_of_lt {a b k : ℕ} (hk : 1 ≤ k) (hab : a < b) (hkb : k ≤ b) : a.choose k < b.choose k := by
  obtain ⟨m, rfl⟩ : ∃ m, b = m + 1 := ⟨b - 1, by omega⟩
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  rw [Nat.choose_succ_succ']
  have h1 : a.choose (j + 1) ≤ m.choose (j + 1) := Nat.choose_le_choose _ (by omega)
  have h2 : 0 < m.choose j := Nat.choose_pos (by omega)
  omega

theorem count : CountTarget := by
  intro C n P kd K₀ hkd hkdn H Cm desc q roots w h
  have hFsub : Claims.Sampling.faulty H Cm (stmtOf P kd desc q roots) (partOf P) w ⊆ Finset.range n := by
    intro v hv
    unfold Claims.Sampling.faulty at hv
    rw [Finset.mem_filter, rus_length_partOf] at hv
    exact hv.1
  have hle : (designOf kd).Pa (partOf P) (Claims.Sampling.faulty H Cm (stmtOf P kd desc q roots) (partOf P) w) ≤
      ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) := not_lt.1 h
  rw [pa_partOf P kd _ hFsub] at hle
  generalize Claims.Sampling.faulty H Cm (stmtOf P kd desc q roots) (partOf P) w = F at hFsub hle ⊢
  have hFn : F.card ≤ n := by simpa using Finset.card_le_card hFsub
  have hpos : (0 : ℝ) < (n.choose kd : ℝ) := by exact_mod_cast Nat.choose_pos hkdn
  have hc : (n - F.card).choose kd ≤ (n - K₀).choose kd := by
    exact_mod_cast (div_le_div_iff_of_pos_right hpos).1 hle
  by_contra hlt
  rw [not_le, lt_min_iff] at hlt
  obtain ⟨h1, h2⟩ := hlt
  have := choose_lt_of_lt hkd (show n - K₀ < n - F.card by omega) (show kd ≤ n - F.card by omega)
  omega

end Flock.Bridge
