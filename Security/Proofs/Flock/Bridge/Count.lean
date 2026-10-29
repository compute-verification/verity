import Proofs.Flock.Bridge.Targets
import Proofs.Flock.Bridge.Lowers
import Proofs.Flock.Claims.Sampling.EndToEndDraw

/-!
# The bridge's slot count, `P_a`, count lemma and scope

`partOf P` is two classes: the population's `n` replay units, then one per input gate (`partOf_classes`,
`rus_length_partOf`). `designOf kd` draws the population `subset kd` and the input class at rate `0`, so its `P_a` is
the population's subset miss at the population slots (`pa_partOf_window`). From them: `PartRusTarget`, `PaTarget`
(`P_a(B) = C(n − |B|, kd)/C(n, kd)` for `B` of population slots), and `CountTarget`: a witness outside `RelS` at
`δ = C(n − K₀, kd)/C(n, kd)` has at least `min K₀ (n − kd + 1)` faulty population slots, because `C(·, kd)` is strictly
increasing from `kd` on (`choose_lt_of_lt`). A faulty input slot is outside this bound: the deployed verifier checks
every input slot, and the design draws none.

Scope: `designOf kd` is well formed on `partOf P` at `1 ≤ kd ≤ n` (`designOf_wellFormed`), and every gate of
`progOf C κ`, its input gates and constants included, is in a proof unit of `partOf P` (`mem_gates_partOf`,
`input_mem_partOf`), which is `Partitions`' covering clause (`partOf_covers`).
-/

namespace Flock.Bridge

open FlockSoundness.Audit

variable {C : FlockSoundness.Audit.Circuit} {n : ℕ} {κ : Fin C.N → Option Bool}

theorem designOf_rules (kd : ℕ) :
    (designOf kd).rules = [⟨.subset kd, none⟩, ⟨.bernoulli 0 1, none⟩] := rfl

theorem partOf_classes (P : FlockSoundness.Audit.Partition C n) :
    (partOf P).classes = [[(List.finRange n).map fun u => [unitGates P u]], inputClassOf C] := rfl

theorem rus_length_partOf (P : FlockSoundness.Audit.Partition C n) :
    (partOf P).rus.length = n + (inputsOf C).length := by
  simp [rus_partOf]

theorem partRus : PartRusTarget := fun P => rus_length_partOf P

theorem window_zero_filter (n : ℕ) (B : Finset ℕ) : Claims.Sampling.window 0 n B = B.filter (· < n) := by
  ext v
  simp [Claims.Sampling.window]

/-- `designOf kd`'s `P_a` on `partOf P`: the population's subset miss at `B`'s population slots; the input class,
drawn at rate `0`, contributes `1`. -/
theorem pa_partOf_window (P : FlockSoundness.Audit.Partition C n) (kd : ℕ) (B : Finset ℕ) :
    (designOf kd).Pa (partOf P) B =
      ((n - (Claims.Sampling.window 0 n B).card).choose kd : ℝ) / (n.choose kd : ℝ) := by
  simp [Claims.Sampling.Design.Pa, designOf, partOf, Claims.Sampling.classesPa, Claims.Sampling.Rule.Pa,
    Claims.Sampling.OneStage.Pa, Claims.Sampling.strataSizes]

theorem pa_partOf (P : FlockSoundness.Audit.Partition C n) (kd : ℕ) (B : Finset ℕ) (hB : B ⊆ Finset.range n) :
    (designOf kd).Pa (partOf P) B = ((n - B.card).choose kd : ℝ) / (n.choose kd : ℝ) := by
  rw [pa_partOf_window, Claims.Sampling.EndToEndDraw.window_zero fun v hv => Finset.mem_range.1 (hB hv)]

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
  classical
  intro C n P kd K₀ hkd hkdn H Cm κ desc q roots w h
  have hle : (designOf kd).Pa (partOf P) (Claims.Sampling.faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) w) ≤
      ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) := not_lt.1 h
  rw [pa_partOf_window, window_zero_filter] at hle
  have hFn : ((Claims.Sampling.faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) w).filter (· < n)).card ≤ n := by
    calc _ ≤ (Finset.range n).card :=
          Finset.card_le_card fun v hv => Finset.mem_range.2 (Finset.mem_filter.1 hv).2
      _ = n := Finset.card_range n
  generalize (Claims.Sampling.faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) w).filter (· < n) = F at hle hFn ⊢
  have hpos : (0 : ℝ) < (n.choose kd : ℝ) := by exact_mod_cast Nat.choose_pos hkdn
  have hc : (n - F.card).choose kd ≤ (n - K₀).choose kd := by
    exact_mod_cast (div_le_div_iff_of_pos_right hpos).1 hle
  by_contra hlt
  rw [not_le, lt_min_iff] at hlt
  obtain ⟨h1, h2⟩ := hlt
  have := choose_lt_of_lt hkd (show n - K₀ < n - F.card by omega) (show kd ≤ n - F.card by omega)
  omega

/-! ## Scope -/

/-- **`designOf kd` is well formed on `partOf P`**: two rules for two classes, the population's `subset kd` with
`1 ≤ kd ≤ n`, and the input class's rate `0/1`. -/
theorem designOf_wellFormed (P : FlockSoundness.Audit.Partition C n) {kd : ℕ} (hkd : 1 ≤ kd) (hkdn : kd ≤ n) :
    (designOf kd).WellFormed (partOf P) := by
  refine ⟨rfl, fun p hp => ?_⟩
  simp only [designOf, partOf, List.zip_cons_cons, List.zip_nil_right, List.mem_cons, List.not_mem_nil,
    or_false] at hp
  rcases hp with rfl | rfl
  · exact ⟨⟨hkd, by simpa [Claims.Sampling.strataSizes] using hkdn⟩, fun k hk => by simp at hk⟩
  · exact ⟨⟨Nat.one_pos, Nat.zero_le _⟩, fun k hk => by simp at hk⟩

/-- **Every gate of `progOf C κ` is in a proof unit of `partOf P`**: a computed wire's lowered gates are in its unit's
(`mem_unitGates`), an input wire's one gate is its input slot's. -/
theorem mem_gates_partOf (P : FlockSoundness.Audit.Partition C n) {i : ℕ} (hi : i < (progOf C κ).gates.length) :
    i ∈ (partOf P).gates := by
  obtain ⟨g, j, hj, rfl, -⟩ := exists_block (List.getElem?_eq_getElem hi)
  unfold Claims.Sampling.Partition.gates
  rw [rus_partOf]
  cases hu : P.unit g with
  | none =>
    have hin : C.op g = .input := by
      have h := (P.input_iff g).1 hu
      cases hop : C.op g <;> simp_all [FlockSoundness.Audit.Op.isInput]
    have hj0 : j = 0 := by rw [hin] at hj; simpa [gateSize] using hj
    have hg : g ∈ inputsOf C := by simp [inputsOf, hin, FlockSoundness.Audit.Op.isInput]
    have hpos : off C g.val + j = posOf C g := by rw [posOf_eq, hin, hj0]; simp [gateSize]
    rw [hpos]
    exact List.mem_flatten.2 ⟨[posOf C g], List.mem_flatten.2 ⟨[[posOf C g]],
      List.mem_append_right _ (List.mem_map.2 ⟨g, hg, rfl⟩), List.mem_singleton_self _⟩, List.mem_singleton_self _⟩
  | some u =>
    exact List.mem_flatten.2 ⟨unitGates P u, List.mem_flatten.2 ⟨[unitGates P u],
      List.mem_append_left _ (List.mem_map.2 ⟨u, List.mem_finRange u, rfl⟩), List.mem_singleton_self _⟩,
      mem_unitGates.2 ⟨g, hu, j, hj, rfl⟩⟩

/-- **The input clause of `InScope`**: every input gate of `progOf C κ` is in a proof unit of `partOf P`. -/
theorem input_mem_partOf (P : FlockSoundness.Audit.Partition C n) {i : ℕ}
    (hi : (progOf C κ).gates[i]? = some .input) : i ∈ (partOf P).gates :=
  mem_gates_partOf P (List.getElem?_eq_some_iff.1 hi).1

/-- `Partitions`' covering clause at `stmtOf`'s program and `partOf P`. -/
theorem partOf_covers (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool) (kd : ℕ)
    (desc : Claims.Msg) (q : Claims.Sampling.Query) (roots : List Claims.Msg) :
    ∀ i < (stmtOf P κ kd desc q roots).prog.gates.length,
      (∀ b, (stmtOf P κ kd desc q roots).prog.gates[i]? ≠ some (.const b)) → i ∈ (partOf P).gates :=
  fun _ hi _ => mem_gates_partOf P hi

end Flock.Bridge
