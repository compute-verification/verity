/-
PROTOCOL.md §15: the `fast100` Ligerito schedule, as data.
-/
namespace Flock

structure Level where
  /-- log2 of the inverse rate -/
  rate : Nat
  /-- queries -/
  q : Nat
  /-- message columns (log2) -/
  cols : Nat
  /-- codeword block (log2): `cols + rate` -/
  d : Nat
  /-- stratified decomposition of `q`: `Σ 2^{c_i} = q`, non-increasing -/
  strata : Array Nat
  /-- Merkle cap depth: the first stratum -/
  cap : Nat
  /-- lanes (log2): `k0` at level 0, 4 above -/
  lanes : Nat
  deriving Repr, Inhabited

structure Schedule where
  m : Nat
  /-- `n = m − 7`, the packed table's variables -/
  n : Nat
  k0 : Nat
  levels : Array Level
  /-- the final level's columns (`yr` has `2^{ext+1}` words) -/
  ext : Nat
  deriving Repr, Inhabited

/-- Greedy power-of-two decomposition: `c_i = min(⌊log2 rem⌋, d)`. -/
def decompose (q d : Nat) : Array Nat := Id.run do
  let mut rem := q
  let mut out := #[]
  for _ in [0:q] do
    if rem == 0 then break
    let c := min (Nat.log2 rem) d
    out := out.push c
    rem := rem - 2 ^ c
  return out

def queriesAt : Array Nat := #[218, 106, 71, 53, 43, 36, 32]

/-- The schedule for `m ∈ [22, 35]`; none otherwise (§15: no schedule, reject). -/
def fast100 (m : Nat) : Option Schedule := do
  if m < 22 || m > 35 then none
  let levels := if m ≤ 24 then 3 else if m ≤ 27 then 4 else if m ≤ 30 then 5 else if m ≤ 33 then 6 else 7
  let k0 := if m == 28 then 4 else if m == 29 then 5 else 6
  let n := m - 7
  let mut ls : Array Level := #[]
  let mut cols := n - k0
  for l in [0:levels] do
    let (rate, q) := if m == 22 && l == 2 then (4, 53) else (l + 1, queriesAt.getD l 32)
    let d := cols + rate
    let strata := decompose q d
    ls := ls.push { rate, q, cols, d, strata, cap := strata.getD 0 0, lanes := if l == 0 then k0 else 4 }
    cols := cols - 3
  let ext := (ls.back?.map (·.cols)).getD 0
  return { m, n, k0, levels := ls, ext }

end Flock
