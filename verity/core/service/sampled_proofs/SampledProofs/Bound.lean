/-
The profile's bounds on wrong units at δ, exact (`profile.Stratified.count_bound` and `stratum_bound`): independent uniform
draws of `k_s` of the `n_s` units of each stratum miss a set of `m_s` wrong units in each with probability
`Π_s C(n_s − m_s, k_s) / C(n_s, k_s)`, the escape `Security/Proofs/Flock/Soundness/Audit/OneStage.lean`'s `audit_count`
bounds by. `miss K` is the largest escape of a set of `K` wrong units: the product of the `K` largest factors
`(n_s − j − k_s) / (n_s − j)` over every stratum `s` and `j < finite_s`, taken greedily because each stratum's factors fall
as `j` grows. `countBound δ` is the largest `K` with `miss K ≥ δ`: except with probability δ (plus the soundness terms), an
accepted transcript has at most that many wrong units. `densityBound` is the harm such a set holds at δ in closed form
(`optimize.density_bound`, which `certify` reports), in rationals.
-/
import CoreProtocol.Profile

namespace SampledProofs.Bound

open Core (Ratio Stratum)

/-- The most wrong units the stratum holds at a positive escape. -/
def Stratum.finite (s : Stratum) : Nat := if s.k == 0 then s.size else s.size - s.k

/-- A non-negative rational `num / den`, reduced. -/
private def reduce (num den : Nat) : Nat × Nat :=
  let g := Nat.gcd num den
  if g ≤ 1 then (num, den) else (num / g, den / g)

/-- `num / den ≥ δ`. -/
private def atLeast (num den : Nat) (δ : Ratio) : Bool := δ.num * den ≤ num * δ.den

/-- The next wrong unit the likeliest escape takes, given each stratum's wrong units so far: the stratum, with room left,
whose next factor `(n − m − k) / (n − m)` is largest, that is whose `k / (n − m)` is smallest (the first such). -/
private def next (st : Array (Stratum × Nat)) : Option (Nat × Stratum × Nat) :=
  st.zipIdx.foldl (init := none) fun best ((s, m), i) =>
    if Stratum.finite s ≤ m then best else
    match best with
    | none => some (i, s, m)
    | some (_, t, mt) => if s.k * (t.size - mt) < t.k * (s.size - m) then some (i, s, m) else best

/-- `Stratified.count_bound`: the largest `K ≤ population` with `miss K ≥ δ`. -/
def countBound (strata : Array Stratum) (δ : Ratio) : Nat := Id.run do
  let population := strata.foldl (fun n s => n + s.size) 0
  let mut st := strata.map fun s => (s, 0)
  let mut num := 1
  let mut den := 1
  for K in [0:population] do
    let some (i, s, m) := next st | return K
    let (a, b) := reduce (num * (s.size - m - s.k)) (den * (s.size - m))
    if !atLeast a b δ then return K
    num := a
    den := b
    st := st.modify i fun (s, m) => (s, m + 1)
  return population

/-- `Stratified.stratum_bound`: the largest `m ≤ size` with `C(size − m, k) / C(size, k) ≥ δ`, every wrong unit in the
stratum. -/
def stratumBound (s : Stratum) (δ : Ratio) : Nat := Id.run do
  let mut num := 1
  let mut den := 1
  for m in [0:s.size] do
    if Stratum.finite s ≤ m then return m
    let (a, b) := reduce (num * (s.size - m - s.k)) (den * (s.size - m))
    if !atLeast a b δ then return m
    num := a
    den := b
  return s.size

/-- `(1 − p)^m ≥ δ`, exactly. -/
private def escapes (p δ : Ratio) (m : Nat) : Bool := δ.num * p.den ^ m ≤ (p.den - p.num) ^ m * δ.den

/-- `Bernoulli.count_bound`: the largest `M ≤ population` with `(1 − p)^M ≥ δ`. Every unit is drawn independently with
probability `p`, so a set of `M` wrong units escapes with probability exactly `(1 − p)^M`, which falls as `M` grows: all
`population` when `1 − population · p ≥ δ` (Bernoulli's inequality), else the search doubles `M` while it escapes, then
halves. -/
def bernoulliBound (population : Nat) (p δ : Ratio) : Nat := Id.run do
  if population * p.num ≤ p.den && δ.num * p.den ≤ (p.den - population * p.num) * δ.den then return population
  let mut lo := 0
  let mut hi := 1
  for _ in [0:population.log2 + 2] do
    if population < hi || !escapes p δ hi then break
    lo := hi
    hi := 2 * hi
  hi := min hi (population + 1)
  for _ in [0:population.log2 + 2] do
    if hi ≤ lo + 1 then break
    let mid := (lo + hi) / 2
    if escapes p δ mid then lo := mid else hi := mid
  return lo

/-- The strata a profile's draw reads as (`IntegrityProfile.strata`): its own, or one stratum for a uniform k-subset;
`none` for a Bernoulli draw. -/
def strataOf (population : Nat) : Core.Draw → Option (Array Stratum)
  | .subset k => some #[{ name := "unit", size := population, k }]
  | .stratified strata => some strata
  | .bernoulli _ => none

/-- `IntegrityProfile.bound`: the most wrong units an accepted transcript has at δ, over the strata the draw reads as
(`countBound`), or for a Bernoulli draw `bernoulliBound`. -/
def boundOf (population : Nat) (d : Core.Draw) (δ : Ratio) : Nat :=
  match d with
  | .bernoulli p => bernoulliBound population p δ
  | d => match strataOf population d with
    | some ss => countBound ss δ
    | none => 0

/-! ## The harm bound in closed form

`optimize.density_bound`, exactly. A set with `c_s` wrong units in stratum `s` escapes with probability at most
`exp(−∑ s k_s c_s / n_s)` and holds none past `n_s − k_s`, so at δ its harm, `h_s` per unit, is at most the fractional
knapsack's value over `∑ s k_s c_s / n_s ≤ ln(1/δ)`: `Security/Proofs/Flock/Soundness/Audit/HarmBound.lean`'s
`stratified_isHarmBound_fill` at the greedy fill, and `stratified_isHarmBound_dual` at any budget above `ln(1/δ)`, which
`lnUpper` is. -/

/-- `optimize.LN_TERMS`: the atanh series' terms `lnUpper` sums. -/
def lnTerms : Nat := 32

/-- `optimize.LN_BITS`: `lnUpper` rounds up to a multiple of `2^-lnBits`. -/
def lnBits : Nat := 64

/-- At least `2 atanh t = ln((1 + t)/(1 − t))` for `0 ≤ t ≤ 1/3`: the series `2 ∑ i t^(2i+1)/(2i+1)` to `lnTerms` terms,
plus `2 t^(2N+1) / ((2N + 1)(1 − t²))`, which bounds the rest. -/
def twoAtanhUpper (t : Rat) : Rat := Id.run do
  let t2 := t * t
  let mut s : Rat := 0
  let mut p := t
  for i in [0:lnTerms] do
    s := s + p / ((2 * i + 1 : Nat) : Rat)
    p := p * t2
  return 2 * (s + p / (((2 * lnTerms + 1 : Nat) : Rat) * (1 - t2)))

/-- `optimize.ln_upper`: a rational at least `ln(1/δ)` for `0 < δ < 1`, within about `2^-lnBits` of it. With
`1/δ = 2^j z`, `1 ≤ z < 2`, it is `j ln 2 + ln z`, `ln 2 = 2 atanh(1/3)` and `ln z = 2 atanh((z − 1)/(z + 1))`, rounded
up. -/
def lnUpper (δ : Rat) : Rat :=
  let y := 1 / δ
  let j0 := y.num.toNat.log2 - y.den.log2
  let j := if y < ((2 ^ j0 : Nat) : Rat) then j0 - 1 else j0
  let z := y / ((2 ^ j : Nat) : Rat)
  let v := (j : Rat) * twoAtanhUpper (1 / 3) + twoAtanhUpper ((z - 1) / (z + 1))
  ((v * ((2 ^ lnBits : Nat) : Rat)).ceil : Rat) / ((2 ^ lnBits : Nat) : Rat)

/-- `optimize.fill`: the greedy fill of rows `(n, k, h)` at a risk budget. A stratum with none drawn counts whole and one
drawn whole not at all; the rest go in decreasing harm per unit of risk `h n / k`, each whole (`n − k` units at risk
`k / n` each) until the budget runs out, the last one fractionally. Strata of equal `h n / k` add the same value per unit
of risk, so their order doesn't change it. -/
def fill (rows : Array (Nat × Nat × Rat)) (budget : Rat) : Rat := Id.run do
  let mut out : Rat := 0
  let mut items : Array (Rat × Nat × Nat × Nat × Rat) := #[]
  for (n, k, h) in rows do
    if h ≤ 0 ∨ n ≤ k then continue
    if k == 0 then
      out := out + (n : Rat) * h
    else
      items := items.push (h * n / k, n - k, n, k, h)
  let mut b := budget
  for (_, top, n, k, h) in items.qsort (fun a c => c.1 < a.1) do
    let x := min (top : Rat) (b * n / k)
    out := out + x * h
    b := b - x * k / n
    if b ≤ 0 then break
  return out

/-- The closed form's harm bound at δ: `fill` at `lnUpper δ`. Its floor is the bits form's (`stratified_isHarmBound_bits`)
when `h` is a unit's output bits. -/
def densityBound (rows : Array (Nat × Nat × Rat)) (δ : Rat) : Rat := fill rows (lnUpper δ)

#guard lnUpper (1 / 2) == 3196577161300663915 / 4611686018427387904
#guard lnUpper (1 / 100) == 10618799479599967255 / 2305843009213693952
#guard fill #[(10, 1, 1), (100, 0, 2), (10, 10, 5)] 1 == 209
#guard fill #[(10, 1, 1), (20, 2, 3)] (1 / 2) == 15
#guard fill #[(10, 1, 1), (20, 2, 0)] 100 == 9

#guard countBound #[{ name := "unit", size := 10, k := 3 }] ⟨1, 10⟩ == 4
#guard countBound #[{ name := "unit", size := 6, k := 0 }] ⟨1, 2⟩ == 6
#guard countBound #[{ name := "a", size := 4, k := 4 }, { name := "b", size := 5, k := 1 }] ⟨1, 2⟩ == 2
#guard stratumBound { name := "unit", size := 10, k := 3 } ⟨1, 10⟩ == 4
#guard bernoulliBound 1000 ⟨1, 2⟩ ⟨1, 1024⟩ == 10
#guard bernoulliBound 1000 ⟨1, 2⟩ ⟨1, 1025⟩ == 10
#guard bernoulliBound 1000 ⟨1, 2⟩ ⟨1, 1023⟩ == 9
#guard bernoulliBound 7 ⟨1, 2⟩ ⟨1, 1024⟩ == 7
#guard bernoulliBound 1000 ⟨0, 1⟩ ⟨1, 10⟩ == 1000
#guard bernoulliBound 1000 ⟨1, 1⟩ ⟨1, 10⟩ == 0
#guard bernoulliBound 0 ⟨1, 3⟩ ⟨1, 10⟩ == 0
#guard bernoulliBound 100 ⟨1, 100⟩ ⟨1, 2⟩ == 68
#guard boundOf 10 (.subset 3) ⟨1, 10⟩ == 4

end SampledProofs.Bound
