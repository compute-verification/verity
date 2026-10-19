/-
The profile's bounds on wrong units at δ, exact (`profile.Stratified.count_bound` and `stratum_bound`): independent uniform
draws of `k_s` of the `n_s` units of each stratum miss a set of `m_s` wrong units in each with probability
`Π_s C(n_s − m_s, k_s) / C(n_s, k_s)`, the escape `Security/Proofs/Flock/Soundness/Audit/OneStage.lean`'s `audit_count`
bounds by. `miss K` is the largest escape of a set of `K` wrong units: the product of the `K` largest factors
`(n_s − j − k_s) / (n_s − j)` over every stratum `s` and `j < finite_s`, taken greedily because each stratum's factors fall
as `j` grows. `countBound δ` is the largest `K` with `miss K ≥ δ`: except with probability δ (plus the soundness terms), an
accepted transcript has at most that many wrong units.
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
