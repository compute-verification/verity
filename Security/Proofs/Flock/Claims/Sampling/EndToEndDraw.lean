import Proofs.Flock.Soundness.ExecDrawOS
import Definitions.Claims.Sampling

/-!
# `Flock.Guarantees.EndToEnd`'s draw against the claims' design

`EndToEnd` draws a uniform `kd`-subset of its `n` units from the verifier's operating-system bytes
(`Law.execOS src (.subset kd) n`, A3 on `src`). The claims read a draw only through the statement's design (`Design.Pa`,
`Design.miss`, `missAt`). At a design of one class under `OneStage.subset kd`, every proof unit of a drawn replay unit
checked (`SubsetDesign`), over a partition of `n` replay units, the live draw escapes a set `B` no more often than the
design's `P_a(B)` (`execOS_escape_le_Pa`), and its count curve, and `EndToEnd`'s sampling term
`C(n − K₀, kd)/C(n, kd)`, are at most the design's miss at `K₀` (`execOS_miss_le_miss`, `choose_le_miss`) and the
deployment's `missAt` (`choose_le_missAt`).
-/

namespace Claims.Sampling.EndToEndDraw

open scoped ENNReal

/-- A design of one class under a uniform `kd`-subset of its replay units, every proof unit of a drawn one checked. -/
def SubsetDesign (d : Design) (kd : ℕ) : Prop := d.rules = [⟨.subset kd, none⟩]

theorem window_zero {n : ℕ} {B : Finset ℕ} (h : ∀ v ∈ B, v < n) : window 0 n B = B := by
  ext v
  simp only [window, Finset.mem_image, Finset.mem_filter, Nat.sub_zero, zero_add]
  exact ⟨fun ⟨w, ⟨hw, _⟩, he⟩ => he ▸ hw, fun hv => ⟨v, ⟨hv, Nat.zero_le _, h v hv⟩, rfl⟩⟩

theorem rus_length {P : Partition} {c : List (List RU)} (hP : P.classes = [c]) :
    P.rus.length = c.flatten.length := by
  simp [Partition.rus, hP]

theorem strataSizes_sum (c : List (List RU)) : (strataSizes c).sum = c.flatten.length := by
  unfold strataSizes
  rw [List.length_flatten]

/-- At a `SubsetDesign`, `P_a(B)` is the hypergeometric escape over the partition's replay units. -/
theorem Pa_subset {d : Design} {kd : ℕ} (hd : SubsetDesign d kd) {P : Partition} {c : List (List RU)}
    (hP : P.classes = [c]) (B : Finset ℕ) :
    d.Pa P B = ((P.rus.length - (window 0 P.rus.length B).card).choose kd : ℝ) /
      (P.rus.length.choose kd : ℝ) := by
  have h1 : d.Pa P B = (Rule.mk (.subset kd) none).Pa (strataSizes c) (ruSizes c) (window 0 c.flatten.length B) * 1 := by
    unfold Design.Pa
    rw [show d.rules = [⟨.subset kd, none⟩] from hd, hP]
    rfl
  rw [h1, mul_one, rus_length hP]
  change ((((strataSizes c).sum - _).choose kd : ℕ) : ℝ) / (((strataSizes c).sum.choose kd : ℕ) : ℝ) = _
  rw [strataSizes_sum]

theorem Pa_le_one {d : Design} {kd : ℕ} (hd : SubsetDesign d kd) {P : Partition} {c : List (List RU)}
    (hP : P.classes = [c]) (hkd : kd ≤ P.rus.length) (B : Finset ℕ) : d.Pa P B ≤ 1 := by
  rw [Pa_subset hd hP]
  exact (div_le_one (by exact_mod_cast Nat.choose_pos hkd)).2
    (by exact_mod_cast Nat.choose_le_choose kd (Nat.sub_le _ _))

theorem choose_div_ofReal (m kd : ℕ) (hkd : kd ≤ m) (a : ℕ) :
    ENNReal.ofReal ((a.choose kd : ℝ) / (m.choose kd : ℝ)) = (a.choose kd : ℝ≥0∞) / (m.choose kd : ℝ≥0∞) := by
  rw [ENNReal.ofReal_div_of_pos (by exact_mod_cast Nat.choose_pos hkd), ENNReal.ofReal_natCast,
    ENNReal.ofReal_natCast]

/-- **`EndToEnd`'s draw escapes no more often than the design's `P_a`**: at every set `B` of units, the live subset
draw on a uniform byte source (A3) misses `B` with probability at most `P_a(B)` of a `SubsetDesign` over `n` replay
units. -/
theorem execOS_escape_le_Pa (n kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : FlockSoundness.Assumptions.UniformRandomBytes Lb src)
    {d : Design} (hd : SubsetDesign d kd) {P : Partition} {c : List (List RU)} (hP : P.classes = [c])
    (hn : P.rus.length = n) (B : Finset (Fin n)) :
    (FlockSoundness.Audit.Law.execOS (n := n) src (.subset kd) n).escape B ≤
      ENNReal.ofReal (d.Pa P (B.map Fin.valEmbedding)) := by
  refine (FlockSoundness.Audit.Law.execOS_subset_escape_le n kd hkd Lb src hA3 B).trans (le_of_eq ?_)
  rw [FlockSoundness.Audit.Law.subset_escape, Pa_subset hd hP, hn,
    window_zero (n := n) (B := B.map Fin.valEmbedding) fun v hv => by
      obtain ⟨a, -, rfl⟩ := Finset.mem_map.1 hv
      exact a.2,
    Finset.card_map, choose_div_ofReal n kd hkd]

/-- **`EndToEnd`'s sampling term is at most the design's miss**: `C(n − K₀, kd)/C(n, kd) ≤ miss(K₀)` at a
`SubsetDesign` over `n` replay units. -/
theorem choose_le_miss (n kd : ℕ) (hkd : kd ≤ n) {d : Design} (hd : SubsetDesign d kd) {P : Partition}
    {c : List (List RU)} (hP : P.classes = [c]) (hn : P.rus.length = n) (K₀ : ℕ) (hK₀ : K₀ ≤ n) :
    ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) ≤ ENNReal.ofReal (d.miss P K₀) := by
  have hB : Finset.range K₀ ⊆ Finset.range P.rus.length ∧ K₀ ≤ (Finset.range K₀).card :=
    ⟨fun v hv => by simp only [Finset.mem_range] at hv ⊢; omega, by simp⟩
  have hbdd : BddAbove (Set.range fun B : {B : Finset ℕ // B ⊆ Finset.range P.rus.length ∧ K₀ ≤ B.card} =>
      d.Pa P B.1) := ⟨1, by rintro _ ⟨B, rfl⟩; exact Pa_le_one hd hP (by omega) B.1⟩
  refine le_trans (le_of_eq ?_) (ENNReal.ofReal_le_ofReal (le_ciSup hbdd ⟨_, hB⟩))
  show _ = ENNReal.ofReal (d.Pa P (Finset.range K₀))
  rw [Pa_subset hd hP, hn, window_zero (n := n) (B := Finset.range K₀) fun v hv => by
      simp only [Finset.mem_range] at hv; omega,
    Finset.card_range, choose_div_ofReal n kd hkd]

/-- **`EndToEnd`'s draw's count curve is at most the design's miss**, on a uniform byte source (A3). -/
theorem execOS_miss_le_miss (n kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : FlockSoundness.Assumptions.UniformRandomBytes Lb src)
    {d : Design} (hd : SubsetDesign d kd) {P : Partition} {c : List (List RU)} (hP : P.classes = [c])
    (hn : P.rus.length = n) (K₀ : ℕ) (hK₀ : K₀ ≤ n) :
    (FlockSoundness.Audit.Law.execOS (n := n) src (.subset kd) n).miss K₀ ≤ ENNReal.ofReal (d.miss P K₀) := by
  refine le_trans ?_ ((FlockSoundness.Audit.Law.subset_miss hkd hK₀).le.trans
    (choose_le_miss n kd hkd hd hP hn K₀ hK₀))
  unfold Verity.Law.miss
  refine iSup₂_le fun B hB => ?_
  exact (FlockSoundness.Audit.Law.execOS_subset_escape_le n kd hkd Lb src hA3 B).trans
    ((FlockSoundness.Audit.Law.subset n kd hkd).escape_le_miss hB)

/-- **At the deployment**: at a statement `x` that parses to `s` with a `SubsetDesign` over `n` replay units of one
class, `EndToEnd`'s sampling term is at most `missAt D K₀ x`, the δ of the claims' `KSCount`. -/
theorem choose_le_missAt {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Zero Col] (D : Deploy Coin Col T) (x : Msg) (s : AuditStatement) (hs : D.parse x = some s)
    (n kd : ℕ) (hkd : kd ≤ n) (hd : SubsetDesign s.design kd) {c : List (List RU)}
    (hP : (D.part s).classes = [c]) (hn : (D.part s).rus.length = n) (K₀ : ℕ) (hK₀ : K₀ ≤ n) :
    ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) ≤ ENNReal.ofReal (missAt D K₀ x) := by
  refine (choose_le_miss n kd hkd hd hP hn K₀ hK₀).trans (ENNReal.ofReal_le_ofReal ?_)
  unfold missAt
  rw [hs]
  exact le_max_right _ _

end Claims.Sampling.EndToEndDraw
