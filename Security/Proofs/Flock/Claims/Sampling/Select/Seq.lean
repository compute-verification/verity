import Proofs.Flock.Soundness.ExecDraw

/-!
# A sequential draw from one byte source misses no more often than its stages' bounds multiplied

`Slots.select` reads its coins from one byte source in order (`SampledProofs.Draw.take`): a stratified or work draw runs
its strata in turn, each stratum's sampler on a stream it extends and reruns until it draws (`Flock.Draw.onWith`), the
next stratum starting where it stopped. Each stratum stops at a time its own bytes decide, and the bytes after it are
uniform and unread, so the strata's draws are independent.

* `StopsAt`, `SeqMiss`: a sampler stopping at its `j`-th read of `N` bytes from `p`, and stages drawing in turn, each
  draw meeting its own condition.
* `prCoin_append`: an event of the bytes before a position and the bytes after it has the product of their
  probabilities.
* `seqMiss_le`: the stages' draws all meet their conditions with probability at most the product of each one's bound on
  a stream of any length, for samplers that keep a draw on a longer stream.
-/

namespace Properties.SampledProofs.SecurityProofs

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Audit.DrawExec Finset
open scoped ENNReal

/-- A sampler drawing `D` at its `j`-th read of `N` bytes of `src` from `p`, its first draw: it draws nothing on the reads
before. -/
def StopsAt (N : ℕ) (run : ByteArray → Option (Array ℕ)) (src : ByteArray) (p j : ℕ) (D : Array ℕ) : Prop :=
  1 ≤ j ∧ p + j * N ≤ src.size ∧ run (src.extract p (p + j * N)) = some D ∧
    ∀ i, 1 ≤ i → i < j → run (src.extract p (p + i * N)) = none

/-- A stage of a sequential draw: its reads' size, its sampler, and the condition on its draw. -/
structure Stage where
  need : ℕ
  run : ByteArray → Option (Array ℕ)
  miss : Array ℕ → Prop

/-- The stages draw in turn from `src`, the first from `p`, each from where the one before stopped, and each draw meets
its condition. -/
def SeqMiss : List Stage → ByteArray → ℕ → Prop
  | [], _, _ => True
  | s :: ss, src, p => ∃ j D, StopsAt s.need s.run src p j D ∧ s.miss D ∧ SeqMiss ss src (p + j * s.need)

theorem stopsAt_shift {N : ℕ} {run : ByteArray → Option (Array ℕ)} {src : ByteArray} {p : ℕ} (hp : p ≤ src.size)
    (q j : ℕ) (D : Array ℕ) : StopsAt N run src (p + q) j D ↔ StopsAt N run (src.extract p src.size) q j D := by
  have hsz : (src.extract p src.size).size = src.size - p := by
    rw [ByteArray.size_extract, Nat.min_self]
  have hex : ∀ i, p + q + i * N ≤ src.size →
      (src.extract p src.size).extract q (q + i * N) = src.extract (p + q) (p + q + i * N) := by
    intro i hi
    rw [ByteArray.extract_extract, Nat.min_eq_left (by omega), Nat.add_assoc]
  constructor
  · rintro ⟨hj, hle, hrun, hnone⟩
    refine ⟨hj, by omega, ?_, fun i hi hij => ?_⟩
    · rw [hex j hle]; exact hrun
    · have : i * N ≤ j * N := Nat.mul_le_mul_right _ hij.le
      rw [hex i (by omega)]; exact hnone i hi hij
  · rintro ⟨hj, hle, hrun, hnone⟩
    have hle' : p + q + j * N ≤ src.size := by omega
    refine ⟨hj, hle', ?_, fun i hi hij => ?_⟩
    · rw [← hex j hle']; exact hrun
    · have : i * N ≤ j * N := Nat.mul_le_mul_right _ hij.le
      rw [← hex i (by omega)]; exact hnone i hi hij

theorem seqMiss_shift {src : ByteArray} {p : ℕ} (hp : p ≤ src.size) :
    ∀ (ss : List Stage) (q : ℕ), SeqMiss ss src (p + q) ↔ SeqMiss ss (src.extract p src.size) q
  | [], _ => Iff.rfl
  | s :: ss, q => by
    simp only [SeqMiss]
    refine exists_congr fun j => exists_congr fun D => ?_
    rw [stopsAt_shift hp q j D, Nat.add_assoc, seqMiss_shift hp ss (q + j * s.need)]

/-- A sampler's reads from the start of `a ++ b` up to `a`'s end are `a`'s. -/
theorem stopsAt_append {N : ℕ} {run : ByteArray → Option (Array ℕ)} {a b : ByteArray} {j : ℕ} (ha : a.size = j * N)
    (D : Array ℕ) : StopsAt N run (a ++ b) 0 j D ↔ StopsAt N run a 0 j D := by
  have hex : ∀ i, i * N ≤ a.size → (a ++ b).extract 0 (0 + i * N) = a.extract 0 (0 + i * N) := by
    intro i hi
    rw [ByteArray.extract_append, Nat.zero_sub, Nat.zero_add, Nat.sub_eq_zero_of_le hi, ByteArray.extract_same]
    exact ByteArray.ext (by simp)
  have hsz : (a ++ b).size = a.size + b.size := ByteArray.size_append
  constructor
  · rintro ⟨hj, -, hrun, hnone⟩
    refine ⟨hj, by omega, ?_, fun i hi hij => ?_⟩
    · rw [← hex j ha.ge]; exact hrun
    · rw [← hex i (ha ▸ Nat.mul_le_mul_right _ hij.le)]; exact hnone i hi hij
  · rintro ⟨hj, -, hrun, hnone⟩
    refine ⟨hj, by omega, ?_, fun i hi hij => ?_⟩
    · rw [hex j ha.ge]; exact hrun
    · rw [hex i (ha ▸ Nat.mul_le_mul_right _ hij.le)]; exact hnone i hi hij

theorem streamOf_data {L : ℕ} (f : Fin L → Fin 256) : (Law.streamOf f).data.toList = bsOf f := by
  simp [Law.streamOf, bsOf, Array.toList_ofFn]

theorem streamOf_append {w ℓ : ℕ} (a : Fin w → Fin 256) (b : Fin ℓ → Fin 256) :
    Law.streamOf (Fin.append a b) = Law.streamOf a ++ Law.streamOf b := by
  refine ByteArray.ext (Array.ext' ?_)
  rw [ByteArray.data_append, Array.toList_append, streamOf_data, streamOf_data, streamOf_data, bsOf_append]

/-- **The bytes before `w` and after it are independent**: an event that is an event of a stream's first `w` bytes and
one of the rest has the product of their probabilities. -/
theorem prCoin_append {w ℓ : ℕ} (E P Q : ByteArray → Prop)
    (h : ∀ a b : ByteArray, a.size = w → (E (a ++ b) ↔ P a ∧ Q b)) :
    prCoin (fun f : Fin (w + ℓ) → Fin 256 => E (Law.streamOf f)) =
      prCoin (fun a : Fin w → Fin 256 => P (Law.streamOf a)) * prCoin (fun b : Fin ℓ → Fin 256 => Q (Law.streamOf b)) := by
  classical
  unfold prCoin
  rw [← avg_equiv (appendEquiv w ℓ), avg_prod]
  have hE : ∀ (a : Fin w → Fin 256) (b : Fin ℓ → Fin 256),
      (if E (Law.streamOf (appendEquiv w ℓ (a, b))) then (1 : ℝ≥0∞) else 0) =
        (if P (Law.streamOf a) then (1 : ℝ≥0∞) else 0) * (if Q (Law.streamOf b) then (1 : ℝ≥0∞) else 0) := by
    intro a b
    have : appendEquiv w ℓ (a, b) = Fin.append a b := rfl
    rw [this, streamOf_append, h _ _ (size_streamOf a)]
    by_cases hP : P (Law.streamOf a) <;> by_cases hQ : Q (Law.streamOf b) <;> simp [hP, hQ]
  simp only [hE]
  have hb : ∀ a : Fin w → Fin 256, (avg fun b : Fin ℓ → Fin 256 =>
      (if P (Law.streamOf a) then (1 : ℝ≥0∞) else 0) * (if Q (Law.streamOf b) then (1 : ℝ≥0∞) else 0)) =
      (if P (Law.streamOf a) then (1 : ℝ≥0∞) else 0) *
        avg fun b : Fin ℓ → Fin 256 => if Q (Law.streamOf b) then (1 : ℝ≥0∞) else 0 := by
    intro a
    simp only [mul_comm (if P (Law.streamOf a) then (1 : ℝ≥0∞) else 0)]
    rw [avg_mul_const, mul_comm]
  simp only [hb]
  rw [avg_mul_const]

/-- **Events of which at most one holds at a time** sum to at most the probability of an event each implies. -/
theorem sum_prCoin_le {C ι : Type} [Fintype C] (s : Finset ι) (p : ι → C → Prop) (q : C → Prop)
    (hone : ∀ c, ∀ i ∈ s, ∀ i' ∈ s, p i c → p i' c → i = i') (hq : ∀ i ∈ s, ∀ c, p i c → q c) :
    ∑ i ∈ s, prCoin (p i) ≤ prCoin q := by
  classical
  unfold prCoin avg
  rw [Finset.sum_comm]
  refine Finset.sum_le_sum fun c _ => ?_
  rw [← Finset.mul_sum]
  refine mul_le_mul_right ?_ _
  by_cases hc : ∃ i ∈ s, p i c
  · obtain ⟨i, hi, hpi⟩ := hc
    rw [Finset.sum_eq_single_of_mem i hi fun i' hi' hne =>
      ite_eq_right_iff.2 fun hp' => absurd (hone c i' hi' i hi hp' hpi) hne]
    simp [hpi, hq i hi c hpi]
  · rw [Finset.sum_eq_zero fun i hi => ite_eq_right_iff.2 fun hp => absurd ⟨i, hi, hp⟩ hc]
    exact zero_le

/-- A stage's bound: its reads are not empty, its sampler keeps a draw on a longer stream, and on a stream of any length
its draw meets its condition with probability at most `q`. -/
def Stage.Bounded (s : Stage) (q : ℝ≥0∞) : Prop :=
  0 < s.need ∧ (∀ a b D, s.run a = some D → s.run (a ++ b) = some D) ∧
    ∀ L, prCoin (fun f : Fin L → Fin 256 => ∃ D, s.run (Law.streamOf f) = some D ∧ s.miss D) ≤ q

/-- **A sequential draw misses no more often than its stages' bounds multiplied**: on `L` uniform bytes, the stages draw
in turn from the start and each draw meets its condition with probability at most the product of their bounds. -/
theorem seqMiss_le : ∀ (ss : List (Stage × ℝ≥0∞)), (∀ p ∈ ss, p.1.Bounded p.2) →
    ∀ L, prCoin (fun f : Fin L → Fin 256 => SeqMiss (ss.map (·.1)) (Law.streamOf f) 0) ≤ (ss.map (·.2)).prod
  | [], _, L => by simpa [SeqMiss] using prCoin_le_one (fun _ : Fin L → Fin 256 => True)
  | (s, q) :: ss, hb, L => by
    obtain ⟨hN, hrun, hq⟩ : s.Bounded q := hb (s, q) List.mem_cons_self
    have ih := seqMiss_le ss fun p hp => hb p (List.mem_cons_of_mem _ hp)
    simp only [List.map_cons, List.prod_cons]
    set rest := (ss.map (·.2)).prod
    let X : ℕ → ByteArray → Prop := fun j a => ∃ D, StopsAt s.need s.run a 0 j D ∧ s.miss D
    let A : ℕ → ByteArray → Prop := fun j a => X j a ∧ SeqMiss (ss.map (·.1)) a (j * s.need)
    have hE : prCoin (fun f : Fin L → Fin 256 => SeqMiss (s :: ss.map (·.1)) (Law.streamOf f) 0) ≤
        ∑ j ∈ range (L + 1), prCoin (fun f : Fin L → Fin 256 => A j (Law.streamOf f)) := by
      refine le_trans (prCoin_mono fun f hf => ?_) (prCoin_exists_le _ _)
      obtain ⟨j, D, hst, hmiss, hrest⟩ := hf
      have hjL : j ≤ L := by
        have := hst.2.1
        rw [size_streamOf] at this
        exact le_trans (Nat.le_mul_of_pos_right j hN) (by omega)
      exact ⟨j, mem_range.2 (by omega), ⟨D, hst, hmiss⟩, by simpa using hrest⟩
    have hterm : ∀ L j, prCoin (fun f : Fin L → Fin 256 => A j (Law.streamOf f)) ≤
        prCoin (fun f : Fin L → Fin 256 => X j (Law.streamOf f)) * rest := by
      intro L j
      by_cases hjL : j * s.need ≤ L
      · obtain ⟨ℓ, rfl⟩ := Nat.exists_eq_add_of_le hjL
        have hsplit : ∀ a b : ByteArray, a.size = j * s.need → (X j (a ++ b) ↔ X j a) := by
          intro a b ha
          exact exists_congr fun D => and_congr_left' (stopsAt_append ha D)
        rw [prCoin_append (A j) (X j) (fun b => SeqMiss (ss.map (·.1)) b 0) fun a b ha => by
            refine and_congr (hsplit a b ha) ?_
            have hp : j * s.need ≤ (a ++ b).size := by rw [ByteArray.size_append]; omega
            rw [← Nat.add_zero (j * s.need), seqMiss_shift hp, ← ha,
              ByteArray.extract_append_eq_right rfl ByteArray.size_append],
          prCoin_append (X j) (X j) (fun _ => True) fun a b ha => by rw [hsplit a b ha, and_true]]
        have h1 : prCoin (fun _ : Fin ℓ → Fin 256 => True) = 1 := by
          simp [prCoin, avg_const]
        rw [h1, mul_one]
        exact mul_le_mul_right (ih ℓ) _
      · have h0 : ∀ f : Fin L → Fin 256, ¬ A j (Law.streamOf f) := by
          rintro f ⟨⟨D, hst, -⟩, -⟩
          have := hst.2.1
          rw [size_streamOf] at this
          omega
        rw [show (fun f : Fin L → Fin 256 => A j (Law.streamOf f)) = fun _ => False from
          funext fun f => propext ⟨h0 f, False.elim⟩, prCoin_false]
        exact zero_le
    have hsum : ∑ j ∈ range (L + 1), prCoin (fun f : Fin L → Fin 256 => X j (Law.streamOf f)) ≤ q := by
      refine le_trans (sum_prCoin_le _ (fun j f => X j (Law.streamOf f))
        (fun f => ∃ D, s.run (Law.streamOf f) = some D ∧ s.miss D) ?_ ?_) (hq L)
      · rintro f j - j' - ⟨D, ⟨hj, -, hD, hnone⟩, -⟩ ⟨D', ⟨hj', -, hD', hnone'⟩, -⟩
        by_contra hne
        rcases Nat.lt_or_gt_of_ne hne with hlt | hlt
        · rw [hnone' j hj hlt] at hD
          exact absurd hD (by simp)
        · rw [hnone j' hj' hlt] at hD'
          exact absurd hD' (by simp)
      · rintro j - f ⟨D, ⟨-, hle, hD, -⟩, hmiss⟩
        refine ⟨D, ?_, hmiss⟩
        have hsplit := ByteArray.extract_eq_extract_append_extract (a := Law.streamOf f) (i := 0)
          (k := (Law.streamOf f).size) (j * s.need) (Nat.zero_le _) (by omega)
        rw [ByteArray.extract_zero_size] at hsplit
        rw [hsplit]
        exact hrun _ _ _ (by simpa using hD)
    calc prCoin (fun f : Fin L → Fin 256 => SeqMiss (s :: ss.map (·.1)) (Law.streamOf f) 0)
        ≤ ∑ j ∈ range (L + 1), prCoin (fun f : Fin L → Fin 256 => A j (Law.streamOf f)) := hE
      _ ≤ ∑ j ∈ range (L + 1), prCoin (fun f : Fin L → Fin 256 => X j (Law.streamOf f)) * rest :=
          Finset.sum_le_sum fun j _ => hterm L j
      _ = (∑ j ∈ range (L + 1), prCoin (fun f : Fin L → Fin 256 => X j (Law.streamOf f))) * rest :=
          (Finset.sum_mul _ _ _).symm
      _ ≤ q * rest := mul_le_mul_left hsum _

end Properties.SampledProofs.SecurityProofs
