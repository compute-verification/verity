/-
The pooled figures over a monitoring period of `B` server-time blocks, each evaluated from the statement that states it,
beside the joint bound (`CoreOnsite.Bound`). Every figure is exact or rounded only toward the safe side: a logarithm the
statement reads is replaced by a certified bound (`ln`), the upper one where the statement needs a count at least
`ln(1/δ)/q` and the lower one where it divides by `ln z`.

* `faultyBlocks`: `Flock.Guarantees.IntegrityHorizon`. If every block is accepted, the period's faulty proof blocks
  total less than `ln(1/δ)/q`, except with probability `δ + Σ_t (ε_ks,t + ε_bind,t)`; the figure is the most they can
  total, `m − 1` for the least `m` with `q·m ≥ ln(1/δ)`.
* `threshold`: `Flock.Guarantees.IntegrityPeriod` at `δ/B` a block. Every accepted block has fewer than `K₀` faulty proof
  blocks except with probability `δ + Σ_t (ε_ks,t + ε_bind,t)`, `K₀` the least threshold whose sampling term
  `C(n − K₀, kd)/C(n, kd)` is at most `δ/B`.
* `shortCount`: `Pous.Period.SecurityProofs.pooledShort_log` (a proved theorem, which no lock lists as a guarantee). At
  least `M` of `U` sampling periods accepted short has probability at most `δ'` once `((z − 1)·U·p + ln(1/δ'))/ln z ≤ M`
  for some `z > 1`, so at least `U − M + 1` periods certify their storage; `M` is tried at `z = M/(U·p)`, where the left
  side is least.
* `leakBits`: `NetTiming.Guarantees.HorizonLeakAccepted`, conditional on its per-block inputs `hesc` and `hleak`. At
  `Fmax`, the least count with `ln(1/δ) ≤ q·Fmax`, the faulty proof blocks cost `b·Fmax + ⌈log2 (N + 1)^Fmax⌉` bits once
  over the monitoring period's `N` proof blocks (`b = 16`, a proof block's outputs), bounded here by
  `Fmax·(b + ⌈log2 (N + 1)⌉)`, and each server-time block pays its own schedule, cell, free-input and jitter terms.
-/
import CoreProtocol.Ratio

namespace Core.Onsite.Pooled

/-- Bits after the point of `ln`'s bounds. -/
def PREC : Nat := 256

/-- A proof block's outputs are at most 16 bits, so a faulty one hides at most that many (`b` in `HorizonLeakAccepted`). -/
def BLOCK_BITS : Nat := 16

def cdiv (a b : Nat) : Nat := (a + b - 1) / b

/-- `⌈log2 x⌉`, 0 for `x ≤ 1`. -/
def clog2 (x : Nat) : Nat := if x ≤ 1 then 0 else Nat.log2 (x - 1) + 1

/-- `Σ_{k < n} f k`. -/
def sumBelow (f : Nat → Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => sumBelow f n + f n

/-- The terms `ln` sums. -/
def TERMS : Nat := 90

/-- `2^PREC` times the `k`-th term of `2·atanh(a/b) = Σ_k 2 (a/b)^(2k+1)/(2k+1)`, as a numerator over a denominator. -/
def termNum (a k : Nat) : Nat := 2 * 2 ^ PREC * a ^ (2 * k + 1)

def termDen (b k : Nat) : Nat := (2 * k + 1) * b ^ (2 * k + 1)

/-- `2·atanh(a/b)` for `3a ≤ b`, as `(lo, hi)` with `lo ≤ 2^PREC · 2·atanh(a/b) ≤ hi`. The series has positive terms,
so the sum of the first `TERMS` terms' floors is below it; the rest is at most `9/8` of the next term (`a/b ≤ 1/3`),
which is below `2^-PREC`. -/
def atanh2 (a b : Nat) : Nat × Nat :=
  (sumBelow (fun k => termNum a k / termDen b k) TERMS,
    sumBelow (fun k => cdiv (termNum a k) (termDen b k)) TERMS + cdiv (9 * termNum a TERMS) (8 * termDen b TERMS))

/-- `ln x` for `x ≥ 1`, as `(lo, hi)` with `lo ≤ 2^PREC · ln x ≤ hi`: `x = 2^k·r` with `1 ≤ r < 2`, and
`ln x = k·ln 2 + 2·atanh((r − 1)/(r + 1))`, `ln 2 = 2·atanh(1/3)`. -/
def ln (x : Ratio) : Nat × Nat :=
  let k := Nat.log2 (x.num / x.den)
  let m := x.den * 2 ^ k
  let l2 := atanh2 1 3
  let r := atanh2 (x.num - m) (x.num + m)
  (k * l2.1 + r.1, k * l2.2 + r.2)

/-- The least `m` with `q·m ≥ ln(1/δ)`, from `ln(1/δ)`'s upper bound, for `0 < δ < 1` and `q > 0`. -/
def pooledCount (q δ : Ratio) : Nat :=
  cdiv ((ln ⟨δ.den, δ.num⟩).2 * q.den) (q.num * 2 ^ PREC)

/-- `IntegrityHorizon`'s figure: the most faulty proof blocks the monitoring period's accepted blocks total at `δ`, each
block's draw covering at least a fraction `q` of its proof blocks. -/
def faultyBlocks (q δ : Ratio) : Nat := pooledCount q δ - 1

/-- The least `k` in `[lo, hi)` with `ok k`, else `hi`, for `ok` monotone; whatever `ok` is, the result is `hi` or passes
`ok`. -/
def least (ok : Nat → Bool) : (fuel lo hi : Nat) → Nat
  | 0, _, hi => hi
  | fuel + 1, lo, hi =>
    if hi ≤ lo then hi
    else
      let mid := (lo + hi) / 2
      if ok mid then least ok fuel lo mid else least ok fuel (mid + 1) hi

/-- Enough halvings to search `[0, n]`. -/
def fuel (n : Nat) : Nat := Nat.log2 n + 2

/-- `C(n, k)`, each step `C(n, i)·(n − i)/(i + 1) = C(n, i + 1)` exact. -/
def choose (n k : Nat) : Nat :=
  if n < k then 0 else (List.range k).foldl (fun c i => c * (n - i) / (i + 1)) 1

/-- `IntegrityPeriod`'s threshold: the least `K₀ ≤ n` whose sampling term `C(n − K₀, kd)/C(n, kd)` is at most `e`,
`n + 1` when none is. -/
def threshold (n kd : Nat) (e : Ratio) : Nat :=
  let total := choose n kd
  least (fun k => choose (n - k) kd * e.den ≤ e.num * total) (fuel (n + 1)) 0 (n + 1)

/-- `pooledShort_log`'s condition, `((z − 1)·U·p + ln(1/δ'))/ln z ≤ M`, from `ln(1/δ')`'s upper bound and `ln z`'s lower
one; false unless `z > 1`. -/
def shortOk (U : Nat) (p δ z : Ratio) (M : Nat) : Bool :=
  0 < z.den && z.den < z.num &&
    (z.num - z.den) * U * p.num * 2 ^ PREC + (ln ⟨δ.den, δ.num⟩).2 * z.den * p.den ≤ M * (ln z).1 * z.den * p.den

/-- The least `M` that `pooledShort_log` bounds at `δ'` over `U` sampling periods at `p`, `U + 1` when none is: each `M > U·p`
tried at `z = M/(U·p)`, and at `p = 0` `M = 1` at `z = 1/δ' + 1`. -/
def shortCount (U : Nat) (p δ : Ratio) : Nat :=
  if p.num = 0 then
    if shortOk U p δ ⟨δ.den + δ.num, δ.num⟩ 1 then 1 else U + 1
  else
    least (fun M => shortOk U p δ ⟨M * p.den, U * p.num⟩ M) (fuel (U + 1)) (U * p.num / p.den + 1) (U + 1)

/-- The sampling periods `pooledShort_log` certifies at `δ'`: at least `U − M + 1` of `U`. -/
def certified (U : Nat) (p δ : Ratio) : Nat := U + 1 - shortCount U p δ

/-- `HorizonLeakAccepted` over `B` server-time blocks, each paying `every` bits of its own terms, over `B·units` proof
blocks with each block's draw covering at least a fraction `q` of them: `(K, Fmax, the faulty proof blocks' bits)`. -/
def leakBits (B every units : Nat) (q δ : Ratio) : Nat × Nat × Nat :=
  let f := pooledCount q δ
  let faulty := f * (BLOCK_BITS + clog2 (B * units + 1))
  (B * every + faulty, f, faulty)

-- ln 2 and ln 100 to within 2^-PREC of their digits; the bounds bracket them
#guard let (lo, hi) := ln ⟨2, 1⟩; lo ≤ hi ∧ hi - lo < 128 ∧
  lo * 10 ^ 12 / 2 ^ PREC == 693147180559 ∧ hi * 10 ^ 12 / 2 ^ PREC == 693147180559
#guard (ln ⟨100, 1⟩).1 * 10 ^ 12 / 2 ^ PREC == 4605170185988
#guard ln ⟨1, 1⟩ == (0, 0)
-- 16·ln 100 = 73.68 and 16·ln 10^6 = 221.05
#guard (faultyBlocks ⟨1, 16⟩ ⟨1, 100⟩, faultyBlocks ⟨1, 16⟩ ⟨1, 1000000⟩) == (73, 221)
#guard (choose 10 3, choose 3 10, choose 1024 0) == (120, 0, 1)
-- C(1024 − K₀, 64)/C(1024, 64) ≤ 1/100 first at K₀ = 69
#guard threshold 1024 64 ⟨1, 100⟩ == 69 ∧ choose (1024 - 69) 64 * 100 ≤ choose 1024 64 ∧
  choose (1024 - 68) 64 * 100 > choose 1024 64
-- a machine-week of 60 s sampling periods at p = 1% + 2^-128 certifies 9,948 of its 10,080 at δ' = 1%
#guard certified 10080 ⟨2 ^ 128 + 100, 100 * 2 ^ 128⟩ ⟨1, 100⟩ == 9948
#guard (certified 10 ⟨0, 1⟩ ⟨1, 100⟩, certified 10 ⟨1, 1⟩ ⟨1, 100⟩) == (10, 0)
#guard leakBits 168 20 1024 ⟨1, 16⟩ ⟨1, 100⟩ == (168 * 20 + 74 * (16 + 18), 74, 74 * (16 + 18))

end Core.Onsite.Pooled
