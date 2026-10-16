import Proofs.Flock.Claims.GS.Flock

/-!
# List decoding within a radius, at Flock's level-0 code

The list-decoding spec in its radius form (`Flock.ListWithin`): the opened columns are written into the zero-filled
word of the trace's codeword length `n` (`Flock.fill`: at each position the first opened column there, zero where none
was opened, positions from `n` on ignored), and the decoder lists every trace `M` whose encoding agrees with that word
on at least `(1 − θ M)·n` positions. The radius `θ`, like the length `n`, is read per trace. Agreement is counted over
all `n` positions, so a prover gains nothing by leaving positions unopened. `fill` and `ListWithin` are stated here in
the shape `Definitions/Claims.lean` is to take.

At Flock's level-0 code (`n = 2^level₀.logLen`, rate `ρ = 2^logCols / 2^logLen`, `θ = 1 − √ρ − 1/200`):

* `flockDecodeR h`: fill the word, then `flockDecode` (the counted Guruswami–Sudan decoder) on its `n` points
  `(i, fill n os i)` at agreement `√ρ + 1/200`.
* `listWithin_flockDecodeR`: it meets `ListWithin` at `θ`. The `n` points are distinct, and the Johnson count fits the
  codeword (`johnsonCount_flock_le`): at agreement `√ρ + 1/200` the count of `k` columns is at most `n` once
  `n·k > 10^4` (`johnsonCount_le_len`), and every `fast100` level 0 has at least `2^9` message columns, so
  `n·k ≥ 2^18`.
* `flockDecodeR_complexity`: its count (`n` lookups of at most `os.length` comparisons each and one step per point, then
  `flockCounted` on the `n` points) is at most `flockPolyR = flockPoly + X^2 + X` at the size `(os.length + n)·2^k0·128`.
-/

namespace Claims.Flock

/-- The zero-filled word of length `n`: at each position the first opened column there, zero where none was opened. -/
def fill {Col : Type} [Zero Col] (n : ℕ) (os : List (ℕ × Col)) : Fin n → Col :=
  fun i => ((os.find? fun p => p.1 == i.val).map Prod.snd).getD 0

/-- **List decoding within radius `θ`**: `decode` lists every trace whose encoding agrees with the zero-filled word of
its codeword length `len M` on at least `(1 - θ M)·len M` positions. The radius is read per trace, like the length. -/
def ListWithin {Col T : Type} [DecidableEq Col] [Zero Col] (len : T → ℕ) (enc : T → ℕ → Col) (θ : T → ℝ)
    (decode : List (ℕ × Col) → List T) : Prop :=
  ∀ (os : List (ℕ × Col)) (M : T),
    (1 - θ M) * len M ≤ ((Finset.univ : Finset (Fin (len M))).filter fun i => enc M i.val = fill (len M) os i).card →
    M ∈ decode os

end Claims.Flock

namespace Claims.GS

open Polynomial FlockSoundness FlockSoundness.Model FlockLevel3

/-! ## The zero-filled word, counted -/

/-- The zero-filled word's `n` points `(i, fill n os i)`. -/
def fillPts {Col : Type} [Zero Col] (n : ℕ) (os : List (ℕ × Col)) : List (ℕ × Col) :=
  (List.finRange n).map fun i => (i.val, Flock.fill n os i)

/-- The first opened column at position `i`, zero where none is, with the comparisons made to find it: one per column
read. -/
def lookupC {Col : Type} [Zero Col] (i : ℕ) : List (ℕ × Col) → Col × ℕ
  | [] => (0, 0)
  | p :: os => if p.1 == i then (p.2, 1) else ((lookupC i os).1, (lookupC i os).2 + 1)

theorem lookupC_fst {Col : Type} [Zero Col] (i : ℕ) :
    ∀ os : List (ℕ × Col), (lookupC i os).1 = ((os.find? fun p => p.1 == i).map Prod.snd).getD 0
  | [] => rfl
  | p :: os => by
    by_cases hp : (p.1 == i) = true
    · simp [lookupC, hp]
    · simp only [Bool.not_eq_true] at hp
      simp [lookupC, hp, lookupC_fst i os]

theorem lookupC_snd_le {Col : Type} [Zero Col] (i : ℕ) : ∀ os : List (ℕ × Col), (lookupC i os).2 ≤ os.length
  | [] => le_rfl
  | p :: os => by
    by_cases hp : (p.1 == i) = true
    · simp [lookupC, hp]
    · simp only [Bool.not_eq_true] at hp
      simp [lookupC, hp, lookupC_snd_le i os]

/-- The zero-filled word's `n` points, each looked up by `lookupC`, with the count: each lookup's comparisons and one
step per point. -/
def fillPtsC {Col : Type} [Zero Col] (n : ℕ) (os : List (ℕ × Col)) : List (ℕ × Col) × ℕ :=
  ((List.finRange n).map fun i => (i.val, (lookupC i.val os).1),
    ((List.finRange n).map fun i => (lookupC i.val os).2 + 1).sum)

theorem fillPtsC_fst {Col : Type} [Zero Col] (n : ℕ) (os : List (ℕ × Col)) :
    (fillPtsC n os).1 = fillPts n os := by
  simp only [fillPtsC, fillPts, Flock.fill, lookupC_fst]

/-- **The fill step's count**: at most `n·(os.length + 1)`. -/
theorem fillPtsC_snd_le {Col : Type} [Zero Col] (n : ℕ) (os : List (ℕ × Col)) :
    (fillPtsC n os).2 ≤ n * (os.length + 1) := by
  have := List.sum_le_length_nsmul ((List.finRange n).map fun i => (lookupC i.val os).2 + 1) (os.length + 1)
    fun x hx => by
      obtain ⟨i, -, rfl⟩ := List.mem_map.1 hx
      exact Nat.add_le_add_right (lookupC_snd_le i.val os) 1
  simpa [fillPtsC] using this

theorem length_fillPts {Col : Type} [Zero Col] (n : ℕ) (os : List (ℕ × Col)) : (fillPts n os).length = n := by
  simp [fillPts]

theorem nodup_fillPts {Col : Type} [Zero Col] (n : ℕ) (os : List (ℕ × Col)) :
    ((fillPts n os).map Prod.fst).Nodup := by
  rw [fillPts, List.map_map]
  exact (List.nodup_finRange n).map Fin.val_injective

theorem lt_of_mem_fillPts {Col : Type} [Zero Col] {n : ℕ} {os : List (ℕ × Col)} {p : ℕ × Col}
    (hp : p ∈ fillPts n os) : p.1 < n := by
  obtain ⟨i, -, rfl⟩ := List.mem_map.1 hp
  exact i.isLt

theorem length_filter_finRange {n : ℕ} (p : Fin n → Prop) [DecidablePred p] :
    ((List.finRange n).filter fun i => decide (p i)).length = (Finset.univ.filter p).card := by
  rw [← List.toFinset_card_of_nodup ((List.nodup_finRange n).filter _), List.toFinset_filter, List.toFinset_finRange]
  simp

/-- The points of the zero-filled word a trace's encoding agrees with are as many as the word's positions it agrees
with. -/
theorem length_filter_fillPts {Col : Type} [DecidableEq Col] [Zero Col] (enc : ℕ → Col) (n : ℕ)
    (os : List (ℕ × Col)) :
    ((fillPts n os).filter fun p => decide (enc p.1 = p.2)).length =
      ((Finset.univ : Finset (Fin n)).filter fun i => enc i.val = Flock.fill n os i).card := by
  rw [fillPts, List.filter_map, List.length_map]
  exact length_filter_finRange fun i : Fin n => enc i.val = Flock.fill n os i

/-! ## The Johnson count fits the codeword -/

/-- **The Johnson count fits the codeword**: at rate `ρ = k/n` and agreement `√ρ + 1/200`, the Johnson count of `k` is
at most `n` once `n·k > 10^4` (that is, `n·√ρ > 100`). -/
theorem johnsonCount_le_len {k n : ℕ} (hn : 0 < n) (hnk : 10000 < n * k) :
    johnsonCount k (Real.sqrt ((k : ℝ) / n) + 1 / 200) ≤ n := by
  set s := Real.sqrt ((k : ℝ) / n)
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hs0 : 0 ≤ s := Real.sqrt_nonneg _
  have hss : s ^ 2 = (k : ℝ) / n := Real.sq_sqrt (by positivity)
  have hnss : (n : ℝ) * s ^ 2 = k := by rw [hss]; field_simp
  have hnk' : (10000 : ℝ) < n * k := by exact_mod_cast hnk
  have hns : 100 < (n : ℝ) * s := by
    have h2 : (10000 : ℝ) < ((n : ℝ) * s) ^ 2 := by
      calc (10000 : ℝ) < n * k := hnk'
        _ = ((n : ℝ) * s) ^ 2 := by rw [← hnss]; ring
    nlinarith [mul_nonneg hn'.le hs0]
  have hlt : ((k : ℝ) + 1) / (s + 1 / 200) ^ 2 < n := by
    rw [div_lt_iff₀ (by positivity)]
    have e : (n : ℝ) * (s + 1 / 200) ^ 2 = n * s ^ 2 + n * s / 100 + n / 40000 := by ring
    rw [e, hnss]
    linarith
  have := (Nat.floor_lt (by positivity)).2 hlt
  unfold johnsonCount
  omega

/-- Every `fast100` schedule's level 0 has `n·k > 10^4`: at least `2^9` message columns, and `n ≥ k`. -/
theorem fast100_level₀_nk {m : ℕ} {sch : Accounting.Schedule} (h : Accounting.fast100 m = some sch) :
    10000 < 2 ^ sch.level₀.logLen * 2 ^ sch.level₀.logCols := by
  have hc := (Flock.fast100_level₀_queries h).2
  have hl : sch.level₀.logCols ≤ sch.level₀.logLen := by unfold Accounting.Level.logLen; omega
  calc 10000 < 2 ^ 9 * 2 ^ 9 := by norm_num
    _ ≤ _ := Nat.mul_le_mul (Nat.pow_le_pow_right (by norm_num) (hc.trans hl)) (Nat.pow_le_pow_right (by norm_num) hc)

/-- **The radius decoder's agreement** `√ρ + 1/200`, with `ρ = 2^logCols / 2^logLen` level 0's rate. -/
noncomputable def agreeR (sch : Accounting.Schedule) : ℝ :=
  Real.sqrt ((2 : ℝ) ^ sch.level₀.logCols / 2 ^ sch.level₀.logLen) + 1 / 200

theorem agreeR_pos (sch : Accounting.Schedule) : 0 < agreeR sch := by
  unfold agreeR
  positivity

/-- **The Johnson count fits Flock's level-0 codeword**: at agreement `√ρ + 1/200` it is at most `n = 2^logLen`, for
every `fast100` schedule. -/
theorem johnsonCount_flock_le {m : ℕ} {sch : Accounting.Schedule} (h : Accounting.fast100 m = some sch) :
    johnsonCount (2 ^ sch.level₀.logCols) (agreeR sch) ≤ 2 ^ sch.level₀.logLen := by
  have := johnsonCount_le_len (k := 2 ^ sch.level₀.logCols) (n := 2 ^ sch.level₀.logLen) (by positivity)
    (fast100_level₀_nk h)
  push_cast at this
  exact this

/-! ## The radius decoder at Flock's level-0 code -/

/-- **The radius decoder at Flock's level-0 code**: the zero-filled word of length `n = 2^logLen`, then `flockDecode`
on its `n` points at agreement `√ρ + 1/200`. -/
noncomputable def flockDecodeR {m : ℕ} {sch : Accounting.Schedule} (h : Accounting.fast100 m = some sch)
    (os : List (ℕ × (Fin (2 ^ sch.k0) → GF128))) : List (Message GF128 sch.k0 sch.level₀.logCols) :=
  flockDecode h (agreeR sch) (fillPts (2 ^ sch.level₀.logLen) os)

/-- `flockDecodeR` with its count: the fill step's (`fillPtsC`), then `flockCounted`'s on the `n` points. -/
noncomputable def flockCountedR {m : ℕ} {sch : Accounting.Schedule} (h : Accounting.fast100 m = some sch)
    (os : List (ℕ × (Fin (2 ^ sch.k0) → GF128))) : List (Message GF128 sch.k0 sch.level₀.logCols) × ℕ :=
  ((flockCounted h (agreeR sch) (fillPtsC (2 ^ sch.level₀.logLen) os).1).1,
    (fillPtsC (2 ^ sch.level₀.logLen) os).2 + (flockCounted h (agreeR sch) (fillPtsC (2 ^ sch.level₀.logLen) os).1).2)

/-- **The radius decoder's complexity polynomial**: `flockPoly + X^2 + X`, that is
`20·X^9 + 171·X^7 + 9·X^6 + 12·X^4 + 2·X^2 + 2·X + 1 + 3·64·129²·X^2·(3X + 3)^4 + 2·(X + 1)·X^2`. -/
noncomputable def flockPolyR : Polynomial ℕ := flockPoly + X ^ 2 + X

/-- **The radius decoder meets `ListWithin`** at Flock's level-0 code, at `θ = 1 − √ρ − 1/200`, for every `fast100`
schedule. -/
theorem listWithin_flockDecodeR {m : ℕ} {sch : Accounting.Schedule} (h : Accounting.fast100 m = some sch) :
    Flock.ListWithin (fun _ => 2 ^ sch.level₀.logLen) (Flock.levelZeroEnc execArith sch.k0 sch.level₀.logCols)
      (fun _ => 1 - Real.sqrt ((2 : ℝ) ^ sch.level₀.logCols / 2 ^ sch.level₀.logLen) - 1 / 200) (flockDecodeR h) := by
  intro os M hM
  refine mem_flockDecode h (agreeR_pos sch) _ (nodup_fillPts _ os) (fun p hp => lt_of_mem_fillPts hp)
    (by rw [length_fillPts]; exact johnsonCount_flock_le h) M ?_
  rw [length_fillPts, length_filter_fillPts]
  have e : agreeR sch = 1 - (1 - Real.sqrt ((2 : ℝ) ^ sch.level₀.logCols / 2 ^ sch.level₀.logLen) - 1 / 200) := by
    unfold agreeR
    ring
  rw [e]
  exact hM

/-- **The radius decoder's complexity**: at most `flockPolyR` at the size `(os.length + n)·2^k0·128`. -/
theorem flockDecodeR_complexity {m : ℕ} {sch : Accounting.Schedule} (h : Accounting.fast100 m = some sch) :
    DecoderComplexity (flockDecodeR h) (flockCountedR h)
      (fun os => (os.length + 2 ^ sch.level₀.logLen) * 2 ^ sch.k0 * 128) flockPolyR := by
  intro os
  have hc := flockDecode_complexity h (agreeR sch) (fillPts (2 ^ sch.level₀.logLen) os)
  simp only [flockCountedR, flockDecodeR, fillPtsC_fst]
  refine ⟨hc.1, ?_⟩
  have hf := fillPtsC_snd_le (2 ^ sch.level₀.logLen) os
  have hg := hc.2
  simp only [length_fillPts] at hg
  simp only [flockPolyR, eval_add, eval_pow, eval_X]
  generalize hQ : (os.length + 2 ^ sch.level₀.logLen) * 2 ^ sch.k0 * 128 = Q
  have hw : 1 ≤ 2 ^ sch.k0 * 128 := Nat.one_le_iff_ne_zero.2 (by positivity)
  have hnQ : 2 ^ sch.level₀.logLen * 2 ^ sch.k0 * 128 ≤ Q := by
    rw [← hQ, mul_assoc, mul_assoc]
    exact Nat.mul_le_mul_right _ (Nat.le_add_left _ _)
  have hn : 2 ^ sch.level₀.logLen ≤ Q := le_trans (by rw [mul_assoc]; exact Nat.le_mul_of_pos_right _ hw) hnQ
  have hL : os.length ≤ Q := by
    rw [← hQ, mul_assoc]
    exact le_trans (Nat.le_add_right _ _) (Nat.le_mul_of_pos_right _ hw)
  have hg' := hg.trans (eval_mono_nat flockPoly hnQ)
  have hf' : 2 ^ sch.level₀.logLen * (os.length + 1) ≤ Q ^ 2 + Q := by
    calc 2 ^ sch.level₀.logLen * (os.length + 1) ≤ Q * (Q + 1) := Nat.mul_le_mul hn (by omega)
      _ = Q ^ 2 + Q := by ring
  omega

end Claims.GS
