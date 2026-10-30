import Proofs.Flock.Bridge.Targets
import Proofs.Flock.Bridge.Lowers
import Proofs.Flock.Claims.Sampling.EndToEndDraw

/-!
# The bridge's slot count, `P_a`, count lemma and scope

`partOf P κ` is one class, the population's `n` replay units (`rus_length_partOf`), each unit's slot holding the input
rows it owns. `designOf kd` draws it `subset kd`, so its `P_a` is the subset miss at the population slots
(`pa_partOf_window`). From them: `PartRusTarget`, `PaTarget` (`P_a(B) = C(n − |B|, kd)/C(n, kd)` for `B` of population
slots), and `CountTarget`: a witness outside `RelS` at `δ = C(n − K₀, kd)/C(n, kd)` has at least `min K₀ (n − kd + 1)`
faulty population slots, because `C(·, kd)` is strictly increasing from `kd` on (`choose_lt_of_lt`). A unit whose owned
input row doesn't open is one of them.

Scope: `designOf kd` is well formed on `partOf P κ` at `1 ≤ kd ≤ n` (`designOf_wellFormed`).
-/

namespace Flock.Bridge

open FlockSoundness.Audit

variable {C : FlockSoundness.Audit.Circuit} {n : ℕ} {κ : Fin C.N → Option Bool}

theorem rus_length_partOf (P : FlockSoundness.Audit.Partition C n) : (partOf P κ).rus.length = n := by
  simp [rus_partOf]

theorem partRus : PartRusTarget := fun P _ => rus_length_partOf P

theorem window_zero_filter (n : ℕ) (B : Finset ℕ) : Claims.Sampling.window 0 n B = B.filter (· < n) := by
  ext v
  simp [Claims.Sampling.window]

/-- `designOf kd`'s `P_a` on `partOf P κ`: the population's subset miss at `B`'s population slots. -/
theorem pa_partOf_window (P : FlockSoundness.Audit.Partition C n) (kd : ℕ) (B : Finset ℕ) :
    (designOf kd).Pa (partOf P κ) B =
      ((n - (Claims.Sampling.window 0 n B).card).choose kd : ℝ) / (n.choose kd : ℝ) := by
  simp [Claims.Sampling.Design.Pa, designOf, partOf, Claims.Sampling.classesPa, Claims.Sampling.Rule.Pa,
    Claims.Sampling.OneStage.Pa, Claims.Sampling.strataSizes]

theorem pa_partOf (P : FlockSoundness.Audit.Partition C n) (kd : ℕ) (B : Finset ℕ) (hB : B ⊆ Finset.range n) :
    (designOf kd).Pa (partOf P κ) B = ((n - B.card).choose kd : ℝ) / (n.choose kd : ℝ) := by
  rw [pa_partOf_window, Claims.Sampling.EndToEndDraw.window_zero fun v hv => Finset.mem_range.1 (hB hv)]

theorem pa : PaTarget := fun P κ kd B hB => pa_partOf (κ := κ) P kd B hB

/-- `C(·, k)` is strictly increasing from `k` on. -/
theorem choose_lt_of_lt {a b k : ℕ} (hk : 1 ≤ k) (hab : a < b) (hkb : k ≤ b) : a.choose k < b.choose k := by
  obtain ⟨m, rfl⟩ : ∃ m, b = m + 1 := ⟨b - 1, by omega⟩
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  rw [Nat.choose_succ_succ']
  have h1 : a.choose (j + 1) ≤ m.choose (j + 1) := Nat.choose_le_choose _ (by omega)
  have h2 : 0 < m.choose j := Nat.choose_pos (by omega)
  omega

theorem count : CountTarget := by
  classical
  intro C n P kd K₀ hkd hkdn H Cm κ desc q roots w h
  have hle : (designOf kd).Pa (partOf P κ) (Claims.Sampling.faulty H Cm (stmtOf P κ kd desc q roots) (partOf P κ) w) ≤
      ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) := not_lt.1 h
  rw [pa_partOf_window, window_zero_filter] at hle
  have hFn : ((Claims.Sampling.faulty H Cm (stmtOf P κ kd desc q roots) (partOf P κ) w).filter (· < n)).card ≤ n := by
    calc _ ≤ (Finset.range n).card :=
          Finset.card_le_card fun v hv => Finset.mem_range.2 (Finset.mem_filter.1 hv).2
      _ = n := Finset.card_range n
  generalize (Claims.Sampling.faulty H Cm (stmtOf P κ kd desc q roots) (partOf P κ) w).filter (· < n) = F at hle hFn ⊢
  have hpos : (0 : ℝ) < (n.choose kd : ℝ) := by exact_mod_cast Nat.choose_pos hkdn
  have hc : (n - F.card).choose kd ≤ (n - K₀).choose kd := by
    exact_mod_cast (div_le_div_iff_of_pos_right hpos).1 hle
  by_contra hlt
  rw [not_le, lt_min_iff] at hlt
  obtain ⟨h1, h2⟩ := hlt
  have := choose_lt_of_lt hkd (show n - K₀ < n - F.card by omega) (show kd ≤ n - F.card by omega)
  omega

/-! ## Scope -/

/-- **`designOf kd` is well formed on `partOf P κ`**: one rule for one class, `subset kd` with `1 ≤ kd ≤ n`. -/
theorem designOf_wellFormed (P : FlockSoundness.Audit.Partition C n) {kd : ℕ} (hkd : 1 ≤ kd) (hkdn : kd ≤ n) :
    (designOf kd).WellFormed (partOf P κ) := by
  refine ⟨rfl, fun p hp => ?_⟩
  simp only [designOf, partOf, List.zip_cons_cons, List.zip_nil_right, List.mem_cons,
    List.not_mem_nil, or_false] at hp
  subst hp
  exact ⟨⟨hkd, by simpa [Claims.Sampling.strataSizes] using hkdn⟩, fun k hk => by simp at hk⟩

end Flock.Bridge
