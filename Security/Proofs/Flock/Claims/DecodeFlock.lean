import Proofs.Flock.Claims.Decode
import Proofs.Flock.Soundness.ListSize
import Proofs.Flock.Soundness.Instance

/-!
# Flock's level-0 code as an `IRS`, and its decoder

Flock's first committed codeword is level 0 of its `fast100` schedule: `2^lanes` lanes (`k₀`, 4 to 6), each the
Reed–Solomon encoding over `F` (GF(2^128) in the executable's arithmetic) of `2^cols` message columns on the novel
basis, at the points `ω_q` (`Arith.encodeAt`), at rate `1/2`. `levelZero` is that code as an `IRS`, for any arithmetic
meeting `Arith.Correct`: a message is a `Message F lanes cols`, a column is `Fin (2^lanes) → F`, and the column at
position `q` is every lane's symbol there (`levelZeroEnc`, `levelZero_enc`). So its decoder meets `ListComplete` for
those columns at the Johnson count `johnsonCount (2^cols) agree = ⌊(2^cols + 1)/agree²⌋ + 1`, at every `agree > 0`
(`listComplete_levelZero`; such a decoder exists, `listComplete_exists_levelZero`), and on opened columns at distinct
positions below `2^64` it is the Guruswami–Sudan path once `agree ≤ 1` is strictly above `√((2^cols + 1)/N)`
(`levelZero_decode_eq_gsPath`), so from the Johnson count on, and `[]` below it unless two positions share a point
(`levelZero_decode_eq_nil`).

`flockLevelZero` is that code for the executable's arithmetic (`execArith`, `F = GF(2^128)`) at level 0 of a `fast100`
schedule: `2^k0` lanes of `2^level₀.logCols` columns, positions below `2^level₀.logLen`
(`listComplete_flockLevelZero`, `listComplete_exists_flockLevelZero`, `flockLevelZero_decode_eq_gsPath`,
`flockLevelZero_decode_eq_nil`).

One run opens fewer level-0 columns than the Johnson count: every `fast100` schedule makes `218` level-0 queries on at
least `2^9` message columns, and at `agree ≤ 1` the count is above `2^9 + 1` (`fast100_level₀_queries_lt`). The count
is reached only by `E^rep`'s columns, gathered over its accepted reruns.

`Deploy`'s `T`, `Col` and `enc` are not fixed yet: these are the ones this decoder fits.
-/

namespace Claims.Flock

open FlockSoundness FlockSoundness.Model Polynomial

/-- A lane's row polynomial determines the lane: the novel basis is linearly independent. -/
theorem rowPoly_injective {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) {L : ℕ} (hL : L ≤ 40) :
    Function.Injective (rowPoly A hA hL) := by
  intro row row' h
  have hsum : ∑ c, (row c - row' c) • xP A hA L hL c = 0 := by
    rw [← sub_eq_zero.2 h]
    simp only [rowPoly, ← Finset.sum_sub_distrib, sub_smul, Polynomial.smul_eq_C_mul, ← sub_mul]
  funext c
  exact sub_eq_zero.1 (Fintype.linearIndependent_iff.1 (xP_linearIndependent A hA L hL) _ hsum c)

/-- **Flock's level-0 code**, `2^lanes` lanes of `2^cols` message columns, as an interleaved Reed–Solomon code: each
lane's row polynomial on the novel basis (degree below `2^cols`), at the points `ω_q`. -/
noncomputable def levelZero {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (lanes cols : ℕ) (hcols : cols ≤ 40) :
    IRS F (Message F lanes cols) (2 ^ lanes) where
  k := 2 ^ cols
  dom := A.omega
  rows M ℓ := rowPoly A hA hcols (M ℓ)
  natDegree_rows M ℓ := rowPoly_natDegree A hA hcols (M ℓ)
  ofRows := Function.invFun fun (M : Message F lanes cols) ℓ => rowPoly A hA hcols (M ℓ)
  ofRows_rows := Function.leftInverse_invFun fun _ _ h =>
    funext fun ℓ => rowPoly_injective A hA hcols (congrFun h ℓ)

/-- Flock's level-0 column at position `q`: every lane's codeword symbol there. -/
def levelZeroEnc {F K : Type} [Field F] [Field K] (A : Arith F K) (lanes cols : ℕ) (M : Message F lanes cols)
    (q : ℕ) : Fin (2 ^ lanes) → F :=
  fun ℓ => A.encodeAt cols (M ℓ) q

theorem levelZero_enc {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (lanes cols : ℕ) (hcols : cols ≤ 40) :
    (levelZero A hA lanes cols hcols).enc = levelZeroEnc A lanes cols := by
  funext M q ℓ
  exact rowPoly_eval A hA hcols (M ℓ) q

/-- **Flock's level-0 decoder meets `ListComplete`** for the level's columns, at the Johnson count
`⌊(2^cols + 1)/agree²⌋ + 1`, at every `agree > 0`. -/
theorem listComplete_levelZero {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (lanes cols : ℕ) (hcols : cols ≤ 40) (agree : ℝ)
    (ha : 0 < agree) :
    ListComplete (levelZeroEnc A lanes cols) agree (fun _ => johnsonCount (2 ^ cols) agree)
      ((levelZero A hA lanes cols hcols).decode agree) := by
  rw [← levelZero_enc A hA lanes cols hcols]
  exact (levelZero A hA lanes cols hcols).listComplete_decode agree ha

/-- **Non-vacuity at Flock's level 0**: a decoder meets `ListComplete` for the level's columns at the Johnson count, at
every `agree > 0`. -/
theorem listComplete_exists_levelZero {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (lanes cols : ℕ) (hcols : cols ≤ 40) (agree : ℝ)
    (ha : 0 < agree) :
    ∃ decode, ListComplete (levelZeroEnc A lanes cols) agree (fun _ => johnsonCount (2 ^ cols) agree) decode :=
  ⟨_, listComplete_levelZero A hA lanes cols hcols agree ha⟩

/-- Opened level-0 columns at distinct positions below `2^64` sit at distinct points. -/
theorem levelZero_points_nodup {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (lanes cols : ℕ) (hcols : cols ≤ 40)
    (os : List (ℕ × (Fin (2 ^ lanes) → F))) (hnd : (os.map Prod.fst).Nodup) (hlt : ∀ p ∈ os, p.1 < 2 ^ 64) :
    (os.map fun p => (levelZero A hA lanes cols hcols).dom p.1).Nodup := by
  have hpts : (os.map fun p => A.omega p.1) = (os.map Prod.fst).map A.omega := by rw [List.map_map]; rfl
  change (os.map fun p => A.omega p.1).Nodup
  rw [hpts]
  refine hnd.map_on fun x hx y hy hxy => hA.omega_injective 64 le_rfl ?_ ?_ hxy
  · obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hx
    exact hlt p hp
  · obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hy
    exact hlt p hp

/-- **Flock's Johnson regime**: opened columns at distinct positions below `2^64`, and `agree ≤ 1` strictly above
`√((2^cols + 1)/N)`. -/
theorem levelZero_regime {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (lanes cols : ℕ) (hcols : cols ≤ 40) (agree : ℝ)
    (os : List (ℕ × (Fin (2 ^ lanes) → F))) (hnd : (os.map Prod.fst).Nodup) (hlt : ∀ p ∈ os, p.1 < 2 ^ 64)
    (ha : agree ≤ 1) (hJ : Real.sqrt ((2 ^ cols + 1) * os.length) < agree * os.length) :
    (levelZero A hA lanes cols hcols).Regime agree os :=
  (levelZero A hA lanes cols hcols).regime_of_johnson agree os
    (levelZero_points_nodup A hA lanes cols hcols os hnd hlt) ha (by simpa [levelZero] using hJ)

/-- In Flock's Johnson regime the level-0 decoder is the Guruswami–Sudan path. -/
theorem levelZero_decode_eq_gsPath {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (lanes cols : ℕ) (hcols : cols ≤ 40) (agree : ℝ)
    (os : List (ℕ × (Fin (2 ^ lanes) → F))) (hnd : (os.map Prod.fst).Nodup) (hlt : ∀ p ∈ os, p.1 < 2 ^ 64)
    (ha : agree ≤ 1) (hJ : Real.sqrt ((2 ^ cols + 1) * os.length) < agree * os.length) :
    (levelZero A hA lanes cols hcols).decode agree os =
      (levelZero A hA lanes cols hcols).gsPath agree os
        (levelZero_regime A hA lanes cols hcols agree os hnd hlt ha hJ).1 :=
  (levelZero A hA lanes cols hcols).decode_eq_gsPath agree os
    (levelZero_regime A hA lanes cols hcols agree os hnd hlt ha hJ)

/-- Outside Flock's Johnson regime, on opened columns at distinct positions below `2^64`, the level-0 decoder returns
`[]`. -/
theorem levelZero_decode_eq_nil {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (lanes cols : ℕ) (hcols : cols ≤ 40) (agree : ℝ)
    (os : List (ℕ × (Fin (2 ^ lanes) → F))) (hnd : (os.map Prod.fst).Nodup) (hlt : ∀ p ∈ os, p.1 < 2 ^ 64)
    (h : ¬ (levelZero A hA lanes cols hcols).Regime agree os) :
    (levelZero A hA lanes cols hcols).decode agree os = [] :=
  (levelZero A hA lanes cols hcols).decode_eq_nil agree os (levelZero_points_nodup A hA lanes cols hcols os hnd hlt) h

open Accounting FlockLevel3

theorem fast100_level₀_fits {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) : sch.level₀.Fits :=
  (levelGood_spec (levelsChained_head_good (fast100_shape _ _ h).2.2.2)).2.1

/-- **Flock's level-0 code in the executable's arithmetic**: GF(2^128), `2^k0` lanes of `2^level₀.logCols` columns,
for a `fast100` schedule. -/
noncomputable def flockLevelZero {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) :
    IRS GF128 (Message GF128 sch.k0 sch.level₀.logCols) (2 ^ sch.k0) :=
  levelZero execArith execArith_correct sch.k0 sch.level₀.logCols (fast100_level₀_fits h).1

/-- **The executable's level-0 decoder meets `ListComplete`** for every `fast100` schedule, at the Johnson count
`⌊(2^level₀.logCols + 1)/agree²⌋ + 1`, at every `agree > 0`. -/
theorem listComplete_flockLevelZero {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) (agree : ℝ)
    (ha : 0 < agree) :
    ListComplete (levelZeroEnc execArith sch.k0 sch.level₀.logCols) agree
      (fun _ => johnsonCount (2 ^ sch.level₀.logCols) agree) ((flockLevelZero h).decode agree) :=
  listComplete_levelZero execArith execArith_correct sch.k0 sch.level₀.logCols (fast100_level₀_fits h).1 agree ha

/-- **Non-vacuity at the executable's level 0**: for every `fast100` schedule and every `agree > 0`, a decoder meets
`ListComplete` for the level's columns at the Johnson count `⌊(2^level₀.logCols + 1)/agree²⌋ + 1`. -/
theorem listComplete_exists_flockLevelZero {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) (agree : ℝ)
    (ha : 0 < agree) :
    ∃ decode, ListComplete (levelZeroEnc execArith sch.k0 sch.level₀.logCols) agree
      (fun _ => johnsonCount (2 ^ sch.level₀.logCols) agree) decode :=
  ⟨_, listComplete_flockLevelZero h agree ha⟩

/-- On opened level-0 columns at distinct positions of the codeword, with `agree ≤ 1` strictly above
`√((2^level₀.logCols + 1)/N)`, the executable's level-0 decoder is the Guruswami–Sudan path. -/
theorem flockLevelZero_decode_eq_gsPath {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) (agree : ℝ)
    (os : List (ℕ × (Fin (2 ^ sch.k0) → GF128))) (hnd : (os.map Prod.fst).Nodup)
    (hlt : ∀ p ∈ os, p.1 < 2 ^ sch.level₀.logLen) (ha : agree ≤ 1)
    (hJ : Real.sqrt ((2 ^ sch.level₀.logCols + 1) * os.length) < agree * os.length) :
    ∃ hnd' : (os.map fun p => (flockLevelZero h).dom p.1).Nodup,
      (flockLevelZero h).decode agree os = (flockLevelZero h).gsPath agree os hnd' := by
  have hlt' : ∀ p ∈ os, p.1 < 2 ^ 64 := fun p hp =>
    lt_of_lt_of_le (hlt p hp) (Nat.pow_le_pow_right (by norm_num) (fast100_level₀_fits h).2)
  exact ⟨_, levelZero_decode_eq_gsPath execArith execArith_correct sch.k0 sch.level₀.logCols
    (fast100_level₀_fits h).1 agree os hnd hlt' ha hJ⟩

/-- Outside the regime, on opened level-0 columns at distinct positions of the codeword, the executable's level-0
decoder returns `[]`. -/
theorem flockLevelZero_decode_eq_nil {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) (agree : ℝ)
    (os : List (ℕ × (Fin (2 ^ sch.k0) → GF128))) (hnd : (os.map Prod.fst).Nodup)
    (hlt : ∀ p ∈ os, p.1 < 2 ^ sch.level₀.logLen) (hr : ¬ (flockLevelZero h).Regime agree os) :
    (flockLevelZero h).decode agree os = [] :=
  levelZero_decode_eq_nil execArith execArith_correct sch.k0 sch.level₀.logCols (fast100_level₀_fits h).1 agree os
    hnd (fun p hp => lt_of_lt_of_le (hlt p hp) (Nat.pow_le_pow_right (by norm_num) (fast100_level₀_fits h).2)) hr

/-- Every `fast100` schedule makes `218` level-0 queries, on at least `2^9` message columns. -/
theorem fast100_level₀_queries {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) :
    sch.level₀.queries = 218 ∧ 9 ≤ sch.level₀.logCols := by
  have hm : 22 ≤ m ∧ m ≤ 35 := by
    by_contra hc
    have : fast100 m = none := by unfold fast100; split <;> simp_all
    simp [this] at h
  obtain ⟨h1, h2⟩ := hm
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at h; subst h; decide)

/-- **One run's level-0 openings are fewer than the Johnson count**: at every `0 < agree ≤ 1` and every `fast100`
schedule, the verifier's `218` level-0 queries are below `johnsonCount (2^level₀.logCols) agree`. -/
theorem fast100_level₀_queries_lt {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) {agree : ℝ} (ha0 : 0 < agree)
    (ha1 : agree ≤ 1) : sch.level₀.queries < johnsonCount (2 ^ sch.level₀.logCols) agree := by
  obtain ⟨hq, hc⟩ := fast100_level₀_queries h
  have := lt_johnsonCount (k := 2 ^ sch.level₀.logCols) ha0 ha1
  have : 2 ^ 9 ≤ 2 ^ sch.level₀.logCols := Nat.pow_le_pow_right (by norm_num) hc
  omega

end Claims.Flock
