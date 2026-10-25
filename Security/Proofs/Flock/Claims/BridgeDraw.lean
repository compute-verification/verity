import Proofs.Flock.Claims.Sampling.Escape
import Proofs.Flock.Soundness.ExecDrawOS

/-!
# The bridge's draw law (A4): EndToEnd's draw is the claims' subset design

`Flock.Guarantees.EndToEnd` draws with `Law.execOS src (.subset kd) n`, the live `flock-verify draw` on the verifier's
operating-system bytes `src`, uniform under A3 (`Assumptions.UniformRandomBytes`). `Claims.Sampling.KS` reads its draw
only through the statement's design: `Design.Pa` (its `M_δ`) and `missAt`. At a statement whose design is one class
drawn as a uniform `kd`-subset of its `n` replay units, checking every proof unit (`SubsetDesign`), the two agree:

* `designPa_subset`: the design's `P_a(B)` is `C(n − |B|, kd)/C(n, kd)`;
* `execOS_escape_le_Pa`: the game form's draw misses `B` with probability at most the claims' `P_a(B)`, under A3;
* `missAt_subset`, `ofReal_missAt_subset`: the claims' `missAt t` is EndToEnd's miss term `C(n − t, kd)/C(n, kd)`,
  and `execOS_miss_le_missAt` bounds the game form's count curve by it;
* `notMem_M_iff`: the count threshold. At every `δ` there is a `K₀ ≤ n + 1` such that a set of units is outside
  `M_δ` exactly when it has at least `K₀` members, and at `K₀ ≤ n` EndToEnd's miss term at `K₀` is at most `δ`. So the
  claims' event `faulty ∉ M_δ` is EndToEnd's `K₀ ≤ |wrong|` at that `K₀` (empty at `K₀ = n + 1`), once A1 maps the faulty
  set into the wrong set.
-/

namespace Claims.Sampling.BridgeDraw

open FlockSoundness.Audit Finset
open scoped ENNReal

/-- **The subset design at `kd` over `n` units**: the statement's design is one rule, a uniform `kd`-subset first
stage checking every proof unit, on a partition of one class of `n` replay units. -/
structure SubsetDesign (s : Stmt) (P : Partition) (n kd : ℕ) : Prop where
  rules : s.design.rules = [⟨.subset kd, none⟩]
  one : ∃ c, P.classes = [c]
  card : P.rus.length = n

/-- The subset design's acceptance probability at `t` units. -/
noncomputable def ratio (n kd t : ℕ) : ℝ := ((n - t).choose kd : ℝ) / (n.choose kd : ℝ)

theorem ratio_anti (n kd : ℕ) {t t' : ℕ} (h : t ≤ t') : ratio n kd t' ≤ ratio n kd t := by
  unfold ratio
  gcongr
  exact Nat.choose_le_choose kd (by omega)

theorem ratio_nonneg (n kd t : ℕ) : 0 ≤ ratio n kd t := by
  unfold ratio; positivity

theorem ofReal_ratio (n kd t : ℕ) (hkd : kd ≤ n) :
    ENNReal.ofReal (ratio n kd t) = ((n - t).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) := by
  unfold ratio
  rw [ENNReal.ofReal_div_of_pos (by exact_mod_cast Nat.choose_pos hkd), ENNReal.ofReal_natCast,
    ENNReal.ofReal_natCast]

/-- **The subset design's `P_a(B)`** is `C(n − |B ∩ [0, n)|, kd)/C(n, kd)`. -/
theorem designPa_subset {s : Stmt} {P : Partition} {n kd : ℕ} (h : SubsetDesign s P n kd) (B : Finset ℕ) :
    s.design.Pa P B = ratio n kd (window 0 n B).card := by
  obtain ⟨hr, ⟨c, hc⟩, hn⟩ := h
  have hrus : P.rus = c.flatten := by simp [Partition.rus, hc]
  have hsum : (strataSizes c).sum = n := by
    rw [← hn, hrus, List.length_flatten, strataSizes]
  have hlen : c.flatten.length = n := by rw [← hn, hrus]
  simp only [Design.Pa, hr, hc, classesPa, mul_one, Rule.Pa, OneStage.Pa, hsum, hlen, ratio]

/-- **The draw law, at a set**: under A3, the game form's draw `Law.execOS src (.subset kd) n` misses `B` with
probability at most the claims' design's `P_a(B)`. -/
theorem execOS_escape_le_Pa {s : Stmt} {P : Partition} {n kd : ℕ} (h : SubsetDesign s P n kd) (hkd : kd ≤ n)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hA3 : FlockSoundness.Assumptions.UniformRandomBytes Lb src) (B : Finset (Fin n)) :
    (Law.execOS (n := n) src (.subset kd) n).escape B ≤ ENNReal.ofReal (s.design.Pa P (B.map Fin.valEmbedding)) := by
  have hw : window 0 n (B.map Fin.valEmbedding) = B.map Fin.valEmbedding :=
    window_zero fun u hu => by
      obtain ⟨v, -, rfl⟩ := Finset.mem_map.1 hu
      exact Finset.mem_range.2 v.isLt
  rw [designPa_subset h, hw, Finset.card_map, ofReal_ratio n kd _ hkd, ← Law.subset_escape hkd]
  exact Law.execOS_subset_escape_le n kd hkd Lb src hA3 B

/-- **The subset design's miss at `t ≤ n`** is `C(n − t, kd)/C(n, kd)`. -/
theorem designMiss_subset {s : Stmt} {P : Partition} {n kd : ℕ} (h : SubsetDesign s P n kd) {t : ℕ} (ht : t ≤ n) :
    s.design.miss P t = ratio n kd t := by
  have hle : ∀ B : {B : Finset ℕ // B ⊆ Finset.range P.rus.length ∧ t ≤ B.card},
      s.design.Pa P B.1 ≤ ratio n kd t := by
    rintro ⟨B, hB, htB⟩
    rw [designPa_subset h, ← h.card, window_zero hB, h.card]
    exact ratio_anti n kd htB
  obtain ⟨B₀, hB₀, hc⟩ := Finset.exists_subset_card_eq (s := Finset.range P.rus.length)
    (by rw [Finset.card_range, h.card]; exact ht)
  have : Nonempty {B : Finset ℕ // B ⊆ Finset.range P.rus.length ∧ t ≤ B.card} := ⟨⟨B₀, hB₀, hc.ge⟩⟩
  refine le_antisymm (ciSup_le hle) ?_
  refine le_ciSup_of_le ⟨ratio n kd t, Set.forall_mem_range.2 hle⟩ ⟨B₀, hB₀, hc.ge⟩ ?_
  show ratio n kd t ≤ s.design.Pa P B₀
  rw [designPa_subset h, ← h.card, window_zero hB₀, h.card, hc]

section Deploy

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The claims' `missAt t`** at a statement whose parsed design is the subset design is EndToEnd's miss term
`C(n − t, kd)/C(n, kd)`, at `t ≤ n`. -/
theorem missAt_subset (D : Deploy Coin Col T) {x : Msg} {s : Stmt} {P : Partition} {n kd : ℕ}
    (hx : D.parse x = some s) (hP : D.partition s = some P) (h : SubsetDesign s P n kd) {t : ℕ} (ht : t ≤ n) :
    missAt D t x = ratio n kd t := by
  have hpart : D.part s = P := by simp [Deploy.part, hP]
  simp only [missAt, hx, hpart, designMiss_subset h ht]
  exact max_eq_right (ratio_nonneg n kd t)

/-- `missAt_subset` in `ℝ≥0∞`, as EndToEnd states its miss term. -/
theorem ofReal_missAt_subset (D : Deploy Coin Col T) {x : Msg} {s : Stmt} {P : Partition} {n kd : ℕ}
    (hx : D.parse x = some s) (hP : D.partition s = some P) (h : SubsetDesign s P n kd) (hkd : kd ≤ n) {t : ℕ}
    (ht : t ≤ n) :
    ENNReal.ofReal (missAt D t x) = ((n - t).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) := by
  rw [missAt_subset D hx hP h ht, ofReal_ratio n kd t hkd]

/-- **The draw law, as count curves**: under A3, the game form's draw misses `t ≤ n` or more units with probability at
most the claims' `missAt t`. -/
theorem execOS_miss_le_missAt (D : Deploy Coin Col T) {x : Msg} {s : Stmt} {P : Partition} {n kd : ℕ}
    (hx : D.parse x = some s) (hP : D.partition s = some P) (h : SubsetDesign s P n kd) (hkd : kd ≤ n)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hA3 : FlockSoundness.Assumptions.UniformRandomBytes Lb src) {t : ℕ} (ht : t ≤ n) :
    (Law.execOS (n := n) src (.subset kd) n).miss t ≤ ENNReal.ofReal (missAt D t x) := by
  rw [ofReal_missAt_subset D hx hP h hkd ht, ← Law.subset_miss hkd ht]
  exact iSup₂_mono fun B _ => Law.execOS_subset_escape_le n kd hkd Lb src hA3 B

end Deploy

/-- **The count threshold**: at every `δ`, some `K₀ ≤ n + 1` makes a set of units outside the claims' `M_δ` exactly
when it has at least `K₀` members, with EndToEnd's miss term at `K₀` at most `δ` when `K₀ ≤ n`. -/
theorem notMem_M_iff {s : Stmt} {P : Partition} {n kd : ℕ} (h : SubsetDesign s P n kd) (δ : ℝ) :
    ∃ K₀ ≤ n + 1, (K₀ ≤ n → ratio n kd K₀ ≤ δ) ∧
      ∀ B ⊆ Finset.range P.rus.length, (B ∉ s.design.M P δ ↔ K₀ ≤ B.card) := by
  classical
  let p : ℕ → Prop := fun K => K = n + 1 ∨ ratio n kd K ≤ δ
  have hex : ∃ K, p K := ⟨n + 1, Or.inl rfl⟩
  refine ⟨Nat.find hex, Nat.find_min' hex (Or.inl rfl), fun hK => ?_, fun B hB => ?_⟩
  · rcases Nat.find_spec hex with h1 | h1
    · omega
    · exact h1
  · have hcard : B.card ≤ n := by
      have := Finset.card_le_card hB
      rwa [Finset.card_range, h.card] at this
    have hPa : s.design.Pa P B = ratio n kd B.card := by
      rw [designPa_subset h, ← h.card, window_zero hB, h.card]
    simp only [Design.M, Set.mem_setOf_eq, not_lt, hPa]
    constructor
    · intro hle
      exact Nat.find_min' hex (Or.inr hle)
    · intro hK
      rcases Nat.find_spec hex with h1 | h1
      · omega
      · exact (ratio_anti n kd hK).trans h1

end Claims.Sampling.BridgeDraw
