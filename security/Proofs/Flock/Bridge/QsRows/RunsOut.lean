module

public import Proofs.Flock.Soundness.ExecDrawOS
public import Proofs.Flock.Claims.Sampling.Select.SecondWitness
public import Proofs.Flock.Claims.Sampling.Select.Size

/-!
# A sample runs out of bytes with probability at most `k·2⁻⁶⁴`

`Flock.Draw.subset k n` returns `none` when its stream runs out before its `k` values are sampled. Each value is one
`width`-byte word unless the rejection sampler rejects it, and a uniform word is rejected with probability
`(2^(8w) mod m)/2^(8w) ≤ 2⁻⁶⁴`, a word being 64 bits wider than `m`. On a stream of `k·width n` bytes the sample
therefore runs out only if one of its `k` steps rejects its first word.

* `reject_nat`, `avg_reject_le`: a uniform word is rejected with probability at most `2⁻⁶⁴`.
* `failL`, `failL_le`: the loop's stream runs out with probability at most `c·2⁻⁶⁴` over `c` steps, on a uniform stream
  of at least `c·width n` bytes.
* `subset_none_le`: `Flock.Draw.subset k n` on a uniform stream of at least `k·width n` bytes returns `none` with
  probability at most `k·2⁻⁶⁴`.
* `onWith_first`, `runOS_subset_none`: the live sample reads `2·k·width n + 64` bytes first and extends the stream only
  when the sampler returns `none` on them. So on a source holding that many bytes it runs out only if the sampler
  does on its first read.
* `runOS_subset_none_le`: the live subset sample runs out with probability at most `k·2⁻⁶⁴`, on a uniform source (A3)
  of at least `2·k·width n + 64` bytes. A shorter source runs out always, its first read failing, so `k·width n`
  bytes do not suffice for the live sample.
* `execOS_subset_draw_card_le`: a live subset sample that does not run out holds at most `k` units.
-/

@[expose] public section

namespace FlockSoundness.Audit.DrawExec

open Game Finset
open scoped ENNReal

/-! ## A uniform word is rejected with probability at most `2⁻⁶⁴` -/

theorem log2_mono {a b : ℕ} (h : a ≤ b) : a.log2 ≤ b.log2 := by
  rcases Nat.eq_zero_or_pos a with rfl | ha
  · rw [Nat.log2_zero]
    exact Nat.zero_le _
  · exact (Nat.le_log2 (by omega)).2 ((Nat.log2_self_le (by omega)).trans h)

/-- A larger population's words are at least as wide. -/
theorem drawWidth_mono {a b : ℕ} (h : a ≤ b) : Flock.Draw.width a ≤ Flock.Draw.width b := by
  have := log2_mono h
  unfold Flock.Draw.width
  omega

/-- **The words `uniform m` rejects are at most a `2⁻⁶⁴` share**: there are `2^(8w) mod m < m` of them, and a word
is 64 bits wider than `m`. -/
theorem reject_nat (m : ℕ) (hm : 0 < m) :
    2 ^ (8 * Flock.Draw.width m) % m * 2 ^ 64 ≤ 2 ^ (8 * Flock.Draw.width m) := by
  have h1 : 2 ^ (8 * Flock.Draw.width m) % m ≤ 2 ^ (m.log2 + 1) :=
    (Nat.le_of_lt (Nat.mod_lt _ hm)).trans (Nat.le_of_lt Nat.lt_log2_self)
  have h2 : m.log2 + 1 + 64 ≤ 8 * Flock.Draw.width m := by
    unfold Flock.Draw.width
    omega
  calc 2 ^ (8 * Flock.Draw.width m) % m * 2 ^ 64 ≤ 2 ^ (m.log2 + 1) * 2 ^ 64 := Nat.mul_le_mul_right _ h1
    _ = 2 ^ (m.log2 + 1 + 64) := (Nat.pow_add 2 _ _).symm
    _ ≤ 2 ^ (8 * Flock.Draw.width m) := Nat.pow_le_pow_right (by omega) h2

theorem sum_range_ge (S A : ℕ) (hA : A ≤ S) :
    ∑ z ∈ range S, (if z < A then (0 : ℝ≥0∞) else 1) = ((S - A : ℕ) : ℝ≥0∞) := by
  obtain ⟨R, rfl⟩ : ∃ R, S = A + R := ⟨S - A, by omega⟩
  have h1 : (∑ z ∈ range A, if z < A then (0 : ℝ≥0∞) else 1) = ∑ _z ∈ range A, (0 : ℝ≥0∞) :=
    Finset.sum_congr rfl fun z hz => by rw [ite_eq_left (Finset.mem_range.1 hz)]
  have h2 : (∑ x ∈ range R, if A + x < A then (0 : ℝ≥0∞) else 1) = ∑ _x ∈ range R, (1 : ℝ≥0∞) :=
    Finset.sum_congr rfl fun x _ => by rw [ite_eq_right (by omega)]
  rw [Finset.sum_range_add, h1, h2]
  simp

theorem div_le_inv_of_mul_le {R S N : ℕ} (hN : 0 < N) (h : R * N ≤ S) :
    (R : ℝ≥0∞) / (S : ℝ≥0∞) ≤ (N : ℝ≥0∞)⁻¹ := by
  refine ENNReal.div_le_of_le_mul ?_
  have hN' : (N : ℝ≥0∞) ≠ 0 := by exact_mod_cast hN.ne'
  have hNR : (N : ℝ≥0∞) * R ≤ S := by exact_mod_cast (show N * R ≤ S by rw [Nat.mul_comm]; exact h)
  calc (R : ℝ≥0∞) = (N : ℝ≥0∞)⁻¹ * ((N : ℝ≥0∞) * R) := by
        rw [← mul_assoc, ENNReal.inv_mul_cancel hN' (ENNReal.natCast_ne_top _), one_mul]
    _ ≤ (N : ℝ≥0∞)⁻¹ * S := mul_le_mul' le_rfl hNR

/-- **A uniform word is rejected with probability at most `2⁻⁶⁴`.** -/
theorem avg_reject_le (m : ℕ) (hm : 0 < m) :
    avg (fun a : Fin (Flock.Draw.width m) → Fin 256 =>
      if beL (bsOf a) < 2 ^ (8 * Flock.Draw.width m) - 2 ^ (8 * Flock.Draw.width m) % m
        then (0 : ℝ≥0∞) else 1) ≤ (2 ^ 64 : ℝ≥0∞)⁻¹ := by
  have hSw : 2 ^ (8 * Flock.Draw.width m) = 256 ^ Flock.Draw.width m := by
    rw [pow_mul]
    norm_num
  have hsum := sum_range_ge (2 ^ (8 * Flock.Draw.width m))
    (2 ^ (8 * Flock.Draw.width m) - 2 ^ (8 * Flock.Draw.width m) % m) (Nat.sub_le _ _)
  rw [Nat.sub_sub_self (Nat.mod_le _ _)] at hsum
  have havg := avg_beL (fun z => if z < 2 ^ (8 * Flock.Draw.width m) - 2 ^ (8 * Flock.Draw.width m) % m
    then (0 : ℝ≥0∞) else 1) (Flock.Draw.width m)
  rw [← hSw, hsum] at havg
  have h := div_le_inv_of_mul_le (R := 2 ^ (8 * Flock.Draw.width m) % m) (S := 2 ^ (8 * Flock.Draw.width m))
    (N := 2 ^ 64) (by positivity) (reject_nat m hm)
  have hc : ((2 ^ 64 : ℕ) : ℝ≥0∞) = 2 ^ 64 := by norm_num
  rw [hc] at h
  exact (le_of_eq havg).trans h

/-- One step of the rejection sampler on a stream whose first word is `a`. -/
theorem uniformL_step (m fuel : ℕ) {ℓ' : ℕ} (a : Fin (Flock.Draw.width m) → Fin 256) (b : Fin ℓ' → Fin 256) :
    uniformL m (fuel + 1) (bsOf a ++ bsOf b) =
      if beL (bsOf a) < 2 ^ (8 * Flock.Draw.width m) - 2 ^ (8 * Flock.Draw.width m) % m
        then some (beL (bsOf a) % m, bsOf b) else uniformL m fuel (bsOf b) := by
  have hl : (bsOf a).length = Flock.Draw.width m := length_bsOf a
  simp only [uniformL, List.length_append, hl, length_bsOf, List.take_left' hl, List.drop_left' hl]
  simp

/-! ## The loop runs out with probability at most `c·2⁻⁶⁴` -/

/-- **The loop's stream runs out**: one of its steps' samples returns nothing. -/
def failL (n F : ℕ) : List ℕ → List UInt8 → Prop
  | [], _ => False
  | i :: is, bs => match uniformL (n - i) F bs with
    | none => True
    | some (_, rest) => failL n F is rest

theorem failL_of_loopL_none (n F : ℕ) : ∀ (l : List ℕ) (π : Equiv.Perm ℕ) (bs : List UInt8),
    loopL n F l π bs = none → failL n F l bs := by
  intro l
  induction l with
  | nil => intro π bs h; simp [loopL] at h
  | cons i is ih =>
    intro π bs h
    simp only [loopL] at h
    simp only [failL]
    cases hu : uniformL (n - i) F bs with
    | none => trivial
    | some rr =>
      obtain ⟨r, rest⟩ := rr
      rw [hu] at h
      exact ih _ _ h

theorem loopP_none_of_loopSw (n : ℕ) (src : ByteArray) : ∀ (l : List ℕ) (sw : Std.HashMap ℕ ℕ) (π : Equiv.Perm ℕ)
    (pos : ℕ), loopSw n src l sw pos = none → loopP n src l π pos = none := by
  intro l
  induction l with
  | nil => intro sw π pos h; simp [loopSw] at h
  | cons i is ih =>
    intro sw π pos h
    simp only [loopSw] at h
    simp only [loopP]
    cases hu : Flock.Draw.uniform (n - i) src (src.size + 1) pos with
    | none => rfl
    | some rp =>
      obtain ⟨r, p⟩ := rp
      rw [hu] at h
      exact ih _ _ _ h

/-- A `subset` sample that returns `none` on a stream has its loop run out on that stream's bytes. -/
theorem subset_none_failL {L : ℕ} (f : Fin L → Fin 256) (k n : ℕ)
    (h : Flock.Draw.subset k n (Law.streamOf f) = none) : failL n (L + 1) (List.range' 0 k) (bsOf f) := by
  rw [DrawExec.subset_eq] at h
  cases hl : loopSw n (Law.streamOf f) (List.range' 0 k) ∅ 0 with
  | some sw => rw [hl] at h; exact absurd h (by simp)
  | none =>
    have hp := loopP_none_of_loopSw n (Law.streamOf f) _ _ (Equiv.refl ℕ) _ hl
    rw [loopP_streamOf f n _ _ 0 (Nat.zero_le _), List.drop_zero] at hp
    exact failL_of_loopL_none n (L + 1) _ _ _ hp

open Classical in
/-- **The loop runs out with probability at most `c·2⁻⁶⁴`** over its `c` steps from `i`, on a uniform stream of at
least `c·width n` bytes: each step's first word is there, and the step reads past it only if it rejects it
(`avg_reject_le`). -/
theorem failL_le (n F : ℕ) : ∀ (c i ℓ : ℕ), i + c ≤ n → c * Flock.Draw.width n ≤ ℓ →
    prCoin (fun h : Fin ℓ → Fin 256 => failL n (F + 1) (List.range' i c) (bsOf h)) ≤
      (c : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
  intro c
  induction c with
  | zero =>
    intro i ℓ _ _
    have h0 : ∀ h : Fin ℓ → Fin 256, ¬ failL n (F + 1) (List.range' i 0) (bsOf h) := fun h hf => by
      simp only [List.range'_zero, failL] at hf
    refine le_trans (prCoin_mono (q := fun _ => False) fun h hf => h0 h hf) ?_
    rw [prCoin_false]
    exact zero_le
  | succ c ih =>
    intro i ℓ hi hℓ
    have hm : 0 < n - i := by omega
    have hwn : Flock.Draw.width (n - i) ≤ Flock.Draw.width n := drawWidth_mono (Nat.sub_le n i)
    have hℓ2 : c * Flock.Draw.width n + Flock.Draw.width n ≤ ℓ :=
      le_trans (le_of_eq (Nat.add_one_mul c (Flock.Draw.width n)).symm) hℓ
    obtain ⟨ℓ', rfl⟩ : ∃ ℓ', ℓ = Flock.Draw.width (n - i) + ℓ' := ⟨ℓ - Flock.Draw.width (n - i), by omega⟩
    have hℓ' : c * Flock.Draw.width n ≤ ℓ' := by omega
    have hev : ∀ (a : Fin (Flock.Draw.width (n - i)) → Fin 256) (b : Fin ℓ' → Fin 256),
        beL (bsOf a) < 2 ^ (8 * Flock.Draw.width (n - i)) - 2 ^ (8 * Flock.Draw.width (n - i)) % (n - i) →
        (failL n (F + 1) (List.range' i (c + 1)) (bsOf a ++ bsOf b) ↔
          failL n (F + 1) (List.range' (i + 1) c) (bsOf b)) := by
      intro a b hlt
      rw [List.range'_succ]
      simp only [failL, uniformL_step, ite_eq_left hlt]
    unfold prCoin
    rw [avg_split (H := fun bs => if failL n (F + 1) (List.range' i (c + 1)) bs then 1 else 0)]
    calc (avg fun a : Fin (Flock.Draw.width (n - i)) → Fin 256 => avg fun b : Fin ℓ' → Fin 256 =>
          if failL n (F + 1) (List.range' i (c + 1)) (bsOf a ++ bsOf b) then (1 : ℝ≥0∞) else 0)
        ≤ avg fun a : Fin (Flock.Draw.width (n - i)) → Fin 256 =>
            (if beL (bsOf a) < 2 ^ (8 * Flock.Draw.width (n - i)) - 2 ^ (8 * Flock.Draw.width (n - i)) % (n - i)
              then (0 : ℝ≥0∞) else 1) + (c : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
          refine avg_mono fun a => ?_
          by_cases hlt : beL (bsOf a) <
              2 ^ (8 * Flock.Draw.width (n - i)) - 2 ^ (8 * Flock.Draw.width (n - i)) % (n - i)
          · rw [ite_eq_left hlt, zero_add]
            refine le_trans (le_of_eq ?_) (ih (i + 1) ℓ' (by omega) hℓ')
            unfold prCoin
            congr 1
            funext b
            congr 1
            exact propext (hev a b hlt)
          · rw [ite_eq_right hlt]
            exact le_trans (prCoin_le_one fun b : Fin ℓ' → Fin 256 =>
              failL n (F + 1) (List.range' i (c + 1)) (bsOf a ++ bsOf b)) le_self_add
      _ = avg (fun a : Fin (Flock.Draw.width (n - i)) → Fin 256 =>
            if beL (bsOf a) < 2 ^ (8 * Flock.Draw.width (n - i)) - 2 ^ (8 * Flock.Draw.width (n - i)) % (n - i)
              then (0 : ℝ≥0∞) else 1) + (c : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
          rw [avg_add, avg_const]
      _ ≤ (2 ^ 64 : ℝ≥0∞)⁻¹ + (c : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
          exact add_le_add (avg_reject_le (n - i) hm) le_rfl
      _ = ((c + 1 : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
          rw [Nat.cast_add, Nat.cast_one]
          ring

/-- **`Flock.Draw.subset k n` runs out with probability at most `k·2⁻⁶⁴`**, on a uniform stream of at least
`k·width n` bytes. -/
theorem subset_none_le (k n L : ℕ) (hk : k ≤ n) (hL : k * Flock.Draw.width n ≤ L) :
    prCoin (fun f : Fin L → Fin 256 => Flock.Draw.subset k n (Law.streamOf f) = none) ≤
      (k : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ :=
  (prCoin_mono fun f h => subset_none_failL f k n h).trans (failL_le n L k 0 L (by omega) hL)

end FlockSoundness.Audit.DrawExec

namespace FlockSoundness.Audit.Law

open Game Finset
open scoped ENNReal

variable {n : ℕ}

/-! ## The live sample's first read -/

/-- **The sampler on a stream's first `N` bytes runs out with probability at most `k·2⁻⁶⁴`**, on a uniform stream of
at least `N ≥ k·width n` bytes. -/
theorem subset_prefix_none_le (n k : ℕ) (hk : k ≤ n) (N L : ℕ) (hN : k * Flock.Draw.width n ≤ N) (hL : N ≤ L) :
    prCoin (fun g : Fin L → Fin 256 => Flock.Draw.subset k n ((streamOf g).extract 0 N) = none) ≤
      (k : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
  obtain ⟨ℓ, rfl⟩ : ∃ ℓ, L = N + ℓ := ⟨L - N, by omega⟩
  have happ := _root_.Properties.Spot.SecurityProofs.prCoin_append (w := N) (ℓ := ℓ)
    (fun src => Flock.Draw.subset k n (src.extract 0 N) = none) (fun a => Flock.Draw.subset k n a = none)
    (fun _ => True) (fun a b ha => by
      show Flock.Draw.subset k n ((a ++ b).extract 0 N) = none ↔ Flock.Draw.subset k n a = none ∧ True
      rw [ByteArray.extract_append_eq_left ha.symm, and_true])
  refine le_trans (le_of_eq happ) ?_
  refine le_trans (mul_le_mul' le_rfl (DrawExec.prCoin_le_one _)) ?_
  rw [mul_one]
  exact DrawExec.subset_none_le k n N hk hN

theorem onWithFuel_succ_some {m : Type → Type} [Monad m] (get : ℕ → m ByteArray) (need : ℕ)
    (run : ByteArray → Option (Array ℕ)) (f : ℕ) (src : ByteArray) (u : Array ℕ) (h : run src = some u) :
    _root_.Properties.Spot.SecurityProofs.onWithFuel get need run (f + 1) src = pure (some u) := by
  simp only [_root_.Properties.Spot.SecurityProofs.onWithFuel, h]

/-- **`onWith` in the model returns on its first read** when the sampler samples on it: from a stream not yet read
that holds `need` bytes. -/
theorem onWith_first (T : ℕ → ByteArray) (i need : ℕ) (run : ByteArray → Option (Array ℕ)) (D : Array ℕ)
    (pos : ℕ → ℕ) (hpos : pos i = 0) (hT : need ≤ (T i).size) (hrun : run ((T i).extract 0 need) = some D) :
    (Flock.Draw.onWith (tapeGet T i) need run).run pos = some (some D, Function.update pos i need) := by
  have hg : (tapeGet T i need).run pos = some ((T i).extract 0 need, Function.update pos i need) := by
    show (if pos i + need ≤ (T i).size then
      some ((T i).extract (pos i) (pos i + need), Function.update pos i (pos i + need)) else none) = _
    rw [hpos, Nat.zero_add, ite_eq_left hT]
  have hfuel : (_root_.Properties.Spot.SecurityProofs.onWithFuel (tapeGet T i) need run (0 + 1)
      ((T i).extract 0 need)).run (Function.update pos i need) = some (some D, Function.update pos i need) :=
    (congrArg (fun x : StateT (ℕ → ℕ) Option (Option (Array ℕ)) => x.run (Function.update pos i need))
      (onWithFuel_succ_some (tapeGet T i) need run 0 _ D hrun)).trans rfl
  rw [_root_.Properties.Spot.SecurityProofs.onWith_fuel (tapeGet T i) need run (0 + 1), StateT.run_bind, hg,
    Option.bind_eq_bind, Option.bind_some]
  exact hfuel

/-- **A live subset sample that runs out has its sampler run out on the first read**, on a source that holds the
first read's `2·k·width n + 64` bytes. -/
theorem runOS_subset_none {L : Fin 1 → ℕ} (f : (i : Fin 1) → Fin (L i) → Fin 256) (k n : ℕ)
    (hL : 2 * k * Flock.Draw.width n + 64 ≤ L 0) (h : runOS f (.subset k) n = none) :
    Flock.Draw.subset k n ((streamOf (f 0)).extract 0 (2 * k * Flock.Draw.width n + 64)) = none := by
  have h2 : tapesOf f 0 = streamOf (f 0) := by simp [tapesOf, Array.getD_eq_getD_getElem?]
  cases hs : Flock.Draw.subset k n ((streamOf (f 0)).extract 0 (2 * k * Flock.Draw.width n + 64)) with
  | none => rfl
  | some D =>
    exfalso
    have hrun := onWith_first (tapesOf f) 0 (2 * k * Flock.Draw.width n + 64) (Flock.Draw.subset k n) D
      (fun _ => 0) rfl (by rw [h2, DrawExec.size_streamOf]; exact hL) (by rw [h2]; exact hs)
    unfold runOS at h
    rw [drawWith_subset_run, hrun] at h
    simp at h

/-- **A live subset sample that returns a sample returns the sampler's on the whole stream** (`onWith_tape`). -/
theorem runOS_subset_some {L : Fin 1 → ℕ} (f : (i : Fin 1) → Fin (L i) → Fin 256) (k n : ℕ)
    (d : Flock.Draw.UnitDraw) (hd : runOS f (.subset k) n = some d) :
    Flock.Draw.subset k n (streamOf (f 0)) = some d.units := by
  unfold runOS at hd
  rw [drawWith_subset_run] at hd
  cases ho : (Flock.Draw.onWith (tapeGet (tapesOf f) 0) (2 * k * Flock.Draw.width n + 64)
      (Flock.Draw.subset k n)).run (fun _ => 0) with
  | none => rw [ho] at hd; exact absurd hd (by simp)
  | some p =>
    obtain ⟨u, pos'⟩ := p
    rw [ho] at hd
    cases u with
    | none => exact absurd hd (by simp)
    | some D =>
      have hD : D = d.units := by rw [← Option.some.inj hd]
      have h1 := (onWith_tape (tapesOf f) 0 _ (by omega) _ (fun a b u h => subset_append _ _ a b u h) _ pos' _
        rfl ho).2 D rfl
      have h2 : tapesOf f 0 = streamOf (f 0) := by simp [tapesOf, Array.getD_eq_getD_getElem?]
      rw [h2] at h1
      rw [← hD]; exact h1

/-- **The live subset sample runs out with probability at most `k·2⁻⁶⁴`**, on a uniform byte source (A3) that holds
the first read's `2·k·width n + 64` bytes. -/
theorem runOS_subset_none_le (n k : ℕ) (hk : k ≤ n) {Ω : Type} [Fintype Ω] (L : Fin 1 → ℕ)
    (t : Ω → (i : Fin 1) → Fin (L i) → Fin 256) (ht : Assumptions.UniformRandomBytes L t)
    (hL : 2 * k * Flock.Draw.width n + 64 ≤ L 0) :
    prCoin (fun ω => runOS (t ω) (.subset k) n = none) ≤ (k : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
  classical
  have hkW : k * Flock.Draw.width n ≤ 2 * k * Flock.Draw.width n + 64 := by
    rw [Nat.mul_assoc]
    omega
  rw [ht fun f => runOS f (.subset k) n = none]
  calc prCoin (fun f : (i : Fin 1) → Fin (L i) → Fin 256 => runOS f (.subset k) n = none)
      ≤ prCoin (fun f : (i : Fin 1) → Fin (L i) → Fin 256 => ∀ s ∈ (univ : Finset (Fin 1)),
          (fun s (g : Fin (L s) → Fin 256) => Flock.Draw.subset k n
            ((streamOf g).extract 0 (2 * k * Flock.Draw.width n + 64)) = none) s (f s)) := by
        refine prCoin_mono fun f h s _ => ?_
        obtain rfl : s = 0 := Subsingleton.elim _ _
        exact runOS_subset_none f k n hL h
    _ = ∏ s : Fin 1, prCoin (fun g : Fin (L s) → Fin 256 => Flock.Draw.subset k n
          ((streamOf g).extract 0 (2 * k * Flock.Draw.width n + 64)) = none) :=
        prCoin_pi_forall (Ω := fun s => Fin (L s) → Fin 256) univ (fun s (g : Fin (L s) → Fin 256) =>
          Flock.Draw.subset k n ((streamOf g).extract 0 (2 * k * Flock.Draw.width n + 64)) = none)
    _ ≤ (k : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
        rw [Fin.prod_univ_one]
        exact subset_prefix_none_le n k hk _ (L 0) hkW hL

/-! ## A sample that does not run out holds at most `k` units -/

/-- The units of `[0, n)` an array holds are at most its size. -/
theorem card_filter_mem_le (n : ℕ) (D : Array ℕ) {dec : DecidablePred fun u : Fin n => u.val ∈ D} :
    (@Finset.filter (Fin n) (fun u => u.val ∈ D) dec univ).card ≤ D.size := by
  classical
  have hsub : (@Finset.filter (Fin n) (fun u => u.val ∈ D) dec univ).image Fin.val ⊆ D.toList.toFinset := by
    intro x hx
    obtain ⟨u, hu, rfl⟩ := Finset.mem_image.1 hx
    exact List.mem_toFinset.2 (Array.mem_toList_iff.2 (Finset.mem_filter.1 hu).2)
  calc (@Finset.filter (Fin n) (fun u => u.val ∈ D) dec univ).card
      = ((@Finset.filter (Fin n) (fun u => u.val ∈ D) dec univ).image Fin.val).card :=
        (Finset.card_image_of_injective _ Fin.val_injective).symm
    _ ≤ D.toList.toFinset.card := Finset.card_le_card hsub
    _ ≤ D.toList.length := List.toFinset_card_le _
    _ = D.size := Array.length_toList

/-- **A live subset sample that does not run out holds at most `k` units**: it holds the units of the sampler's `k`
values (`subset_size`). -/
theorem execOS_subset_draw_card_le {Ω : Type} [Fintype Ω] [Nonempty Ω] {L : Fin 1 → ℕ}
    (t : Ω → (i : Fin 1) → Fin (L i) → Fin 256) (k : ℕ) (ω : Ω) (h : runOS (t ω) (.subset k) n ≠ none) :
    ((execOS (n := n) t (.subset k) n).draw ω).card ≤ k := by
  change ((match runOS (t ω) (.subset k) n with
    | some d => univ.filter fun u : Fin n => u.val ∈ d.units
    | none => univ : Finset (Fin n))).card ≤ k
  cases hd : runOS (t ω) (.subset k) n with
  | none => exact absurd hd h
  | some d =>
    exact (card_filter_mem_le n d.units).trans
      (le_of_eq (_root_.Properties.Spot.SecurityProofs.subset_size (runOS_subset_some (t ω) k n d hd)))

end FlockSoundness.Audit.Law
