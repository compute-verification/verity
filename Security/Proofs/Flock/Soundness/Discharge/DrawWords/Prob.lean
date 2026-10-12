import Proofs.Flock.Soundness.Discharge.DrawWords.Words

/-!
# The live run reads past its source's bytes with probability at most `(∑ s, k s) · 2^-64`

* `word_uniform`, `word_reject`: a word at a fixed offset of a uniform tape is uniform below `2^(8w)` (`avg_split`
  twice and `avg_beL`), so `Flock.Draw.uniform m` rejects it with probability `(2^(8w) mod m) / 2^(8w) < m / 2^(8w)`,
  and `8 · width m ≥ log₂ m + 65` makes that at most `2^-64`.
* `fallback_rare`: a failed run has a rejected word among the first `k_s` of some stratum (`words_run`). Each such word
  reads one stratum's stream, uniform under A3 (`prCoin_coord`), so the union bound over the `∑ s, k s` words gives
  `(∑ s, k s) · 2^-64`. No independence between words is used.
* `draw_card_rare`: the live law's draw has `∑ s, k s` units on every run that succeeds (`execOS_draw_card`).
-/

namespace FlockSoundness.Discharge.DrawWords

open FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Audit.Law FlockSoundness.Audit.DrawExec
open FlockSoundness.Discharge.Exec Finset
open scoped ENNReal

theorem avg_word (F : List UInt8 → ℝ≥0∞) (pos w r : ℕ) :
    avg (fun g : Fin (pos + (w + r)) → Fin 256 => F (((bsOf g).drop pos).take w)) =
      avg fun c : Fin w → Fin 256 => F (bsOf c) := by
  rw [avg_split (H := fun bs => F ((bs.drop pos).take w))]
  have h1 : ∀ (a : Fin pos → Fin 256) (b : Fin (w + r) → Fin 256),
      ((bsOf a ++ bsOf b).drop pos).take w = (bsOf b).take w := fun a b => by
    rw [List.drop_left' (length_bsOf a)]
  simp only [h1]
  rw [avg_const, avg_split (H := fun bs => F (bs.take w))]
  have h2 : ∀ (c : Fin w → Fin 256) (d : Fin r → Fin 256), (bsOf c ++ bsOf d).take w = bsOf c := fun c d =>
    List.take_left' (length_bsOf c)
  simp only [h2, avg_const]

/-- **A word at a fixed offset of a uniform tape**: the probability that it meets `P` is the share of `[0, 2^(8w))`
that meets `P`. -/
theorem prCoin_beNat (P : ℕ → Prop) [DecidablePred P] {ℓ : ℕ} (pos w : ℕ) (h : pos + w ≤ ℓ) :
    prCoin (fun g : Fin ℓ → Fin 256 => P (Flock.Draw.beNat (streamOf g) pos w)) =
      (((range (2 ^ (8 * w))).filter P).card : ℝ≥0∞) / ((2 ^ (8 * w) : ℕ) : ℝ≥0∞) := by
  obtain ⟨r, rfl⟩ : ∃ r, ℓ = pos + (w + r) := ⟨ℓ - (pos + w), by omega⟩
  have hS : 2 ^ (8 * w) = 256 ^ w := by rw [pow_mul]; norm_num
  rw [show (fun g : Fin (pos + (w + r)) → Fin 256 => P (Flock.Draw.beNat (streamOf g) pos w)) =
      fun g => P (beL (((bsOf g).drop pos).take w)) from funext fun g => by rw [beNat_streamOf g pos w h]]
  rw [← avg_ite_eq_prCoin, avg_word (fun bs => if P (beL bs) then (1 : ℝ≥0∞) else 0) pos w r,
    avg_beL (fun z => if P z then (1 : ℝ≥0∞) else 0) w, Finset.sum_boole, hS]

theorem word_uniform : WordUniform := by
  intro ℓ pos w h B
  rw [prCoin_beNat (· ∈ B) pos w h]
  congr 3
  ext z
  simp only [mem_filter, mem_range]
  exact and_comm

theorem reject_card (S T : ℕ) : ((range S).filter fun z => ¬ z < T).card = S - T := by
  rw [show (range S).filter (fun z => ¬ z < T) = Ico T S from by ext z; simp only [mem_filter, mem_range, mem_Ico]; omega]
  exact Nat.card_Ico T S

theorem reject_le (m : ℕ) (hm : 0 < m) :
    2 ^ (8 * Flock.Draw.width m) % m * 2 ^ 64 ≤ 2 ^ (8 * Flock.Draw.width m) := by
  have h1 : 2 ^ (8 * Flock.Draw.width m) % m < m := Nat.mod_lt _ hm
  have h2 : m < 2 ^ (m.log2 + 1) := Nat.lt_log2_self
  have h3 : m.log2 + 1 + 64 ≤ 8 * Flock.Draw.width m := by unfold Flock.Draw.width; omega
  calc 2 ^ (8 * Flock.Draw.width m) % m * 2 ^ 64 ≤ 2 ^ (m.log2 + 1) * 2 ^ 64 :=
        Nat.mul_le_mul_right _ (by omega)
    _ = 2 ^ (m.log2 + 1 + 64) := (pow_add 2 (m.log2 + 1) 64).symm
    _ ≤ 2 ^ (8 * Flock.Draw.width m) := Nat.pow_le_pow_right (by norm_num) h3

theorem word_reject : WordReject := by
  intro ℓ pos m hm h
  set S := 2 ^ (8 * Flock.Draw.width m) with hS
  have hSpos : 0 < S := by positivity
  refine ⟨?_, ?_⟩
  · rw [prCoin_beNat (fun z => ¬ z < S - S % m) pos _ h, ← hS, reject_card S (S - S % m),
      Nat.sub_sub_self (Nat.mod_le _ _)]
  · have hS0 : ((S : ℕ) : ℝ≥0∞) ≠ 0 := by exact_mod_cast hSpos.ne'
    rw [ENNReal.div_le_iff hS0 (ENNReal.natCast_ne_top _), ← ENNReal.div_eq_inv_mul,
      ENNReal.le_div_iff_mul_le (Or.inl (by norm_num)) (Or.inl (by norm_num))]
    exact_mod_cast reject_le m hm

/-- **One stream of a uniform source**: an event of stratum `s`'s bytes alone has its probability on those bytes. -/
theorem prCoin_coord {m : ℕ} {L : Fin m → ℕ} (s : Fin m) (P : (Fin (L s) → Fin 256) → Prop) :
    prCoin (fun f : (i : Fin m) → Fin (L i) → Fin 256 => P (f s)) = prCoin P := by
  have key := prCoin_pi_forall (Ω := fun i => Fin (L i) → Fin 256) {s} fun i g => ∀ h : i = s, P (h ▸ g)
  rw [Finset.prod_singleton] at key
  have e1 : (fun f : (i : Fin m) → Fin (L i) → Fin 256 => P (f s)) =
      fun f => ∀ i ∈ ({s} : Finset (Fin m)), ∀ h : i = s, P (h ▸ f i) := by
    funext f
    refine propext ⟨fun H i _ h => ?_, fun H => H s (mem_singleton_self s) rfl⟩
    subst h
    exact H
  have e2 : (fun g : Fin (L s) → Fin 256 => ∀ h : s = s, P (h ▸ g)) = P :=
    funext fun g => propext ⟨fun H => H rfl, fun H _ => H⟩
  rw [e1, key, e2]

theorem fallback_rare : FallbackRare := by
  intro n m σ k hk S hS K pop Ω _ L t ht hL
  rw [ht fun f => runOS f (.stratified K (Array.ofFn S)) pop = none]
  calc prCoin (fun f : (i : Fin m) → Fin (L i) → Fin 256 => runOS f (.stratified K (Array.ofFn S)) pop = none)
      ≤ prCoin (fun f : (i : Fin m) → Fin (L i) → Fin 256 =>
          ∃ s ∈ (univ : Finset (Fin m)), ∃ i ∈ range (k s), ¬ WordOK (S s).size (streamOf (f s)) i) :=
        prCoin_mono fun f hf => by
          by_contra hno
          push Not at hno
          obtain ⟨d, hd⟩ := words_run S K pop f hL fun s i hi =>
            hno s (mem_univ _) i (mem_range.2 (by rw [← (hS s).1]; exact hi))
          rw [hd] at hf
          exact absurd hf (by simp)
    _ ≤ ∑ s, prCoin (fun f : (i : Fin m) → Fin (L i) → Fin 256 =>
          ∃ i ∈ range (k s), ¬ WordOK (S s).size (streamOf (f s)) i) := prCoin_exists_le _ _
    _ ≤ ∑ s, ∑ i ∈ range (k s), prCoin (fun f : (i : Fin m) → Fin (L i) → Fin 256 =>
          ¬ WordOK (S s).size (streamOf (f s)) i) := sum_le_sum fun s _ => prCoin_exists_le _ _
    _ ≤ ∑ s, ∑ _i ∈ range (k s), (2 ^ 64 : ℝ≥0∞)⁻¹ := sum_le_sum fun s _ => sum_le_sum fun i hi => by
        have hi : i < (S s).k := by rw [(hS s).1]; exact mem_range.1 hi
        have hsz : 0 < (S s).size - i := by
          have := hk s; have := (hS s).2.1; have := (hS s).1; omega
        rw [prCoin_coord s fun g => ¬ WordOK (S s).size (streamOf g) i]
        exact (word_reject (wordOff (S s).size i) ((S s).size - i) hsz
          ((word_end_le_need (S s) hi).trans (hL s))).1.le.trans (word_reject _ _ hsz
          ((word_end_le_need (S s) hi).trans (hL s))).2
    _ = ((∑ s, k s : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
        simp only [sum_const, card_range, nsmul_eq_mul, Nat.cast_sum, Finset.sum_mul]

theorem draw_card_rare : DrawCardRare := by
  intro n m σ k hk S hS K pop Ω _ _ L t ht hL
  refine le_trans (prCoin_mono fun ω hω => ?_) (fallback_rare σ k hk S hS K pop L t ht hL)
  cases h : runOS (t ω) (.stratified K (Array.ofFn S)) pop with
  | none => rfl
  | some d => exact absurd (execOS_draw_card σ k S hS K pop t ω h) hω

end FlockSoundness.Discharge.DrawWords
