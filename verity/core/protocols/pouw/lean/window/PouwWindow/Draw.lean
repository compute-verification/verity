import PouwWindow.Grid

/-!
# The draw (`window.py`, "The draw"; L3)

`kPrime` is K′: the smallest `K ≥ 1` with `(1 − p)^K ≤ δ` in rationals, `p = ε·credited/(n·w_max)`, capped at `n`; `1` when
`p = 1`. It is exact. `missAtMost` decides `q^k ≤ δ` in integers while they are small, and otherwise from fixed-point
bounds that round down and up, at a precision that doubles until they decide; `kPrime` searches the least `K` by
bisection, since `q^K` falls with `K` for `0 ≤ q < 1`. Python's `k_prime` starts from a binary64 estimate and steps with
the same exact comparison; the difftest holds the two to one answer.
-/

namespace PouwWindow

/-- `⌈x / y⌉`. -/
def ceilDiv (x y : Nat) : Nat := (x + y - 1) / y

/-- `(lo, hi)` with `lo ≤ (a/b)^k·2^prec ≤ hi` (`one = 2^prec`): square-and-multiply over the bits of `k`, low first,
rounding the lower bound down and the upper bound up at every product. `fuel` is `k`'s bit length. -/
def fixedPowGo (one : Nat) : Nat → Nat → Nat → Nat → Nat → Nat → Nat × Nat
  | 0, _, lo, hi, _, _ => (lo, hi)
  | fuel + 1, k, lo, hi, bl, bh =>
    let lohi := if k % 2 = 1 then (lo * bl / one, ceilDiv (hi * bh) one) else (lo, hi)
    fixedPowGo one fuel (k / 2) lohi.1 lohi.2 (bl * bl / one) (ceilDiv (bh * bh) one)

def fixedPow (a b k prec : Nat) : Nat × Nat :=
  fixedPowGo (2 ^ prec) (bitLength k) k (2 ^ prec) (2 ^ prec) (a * 2 ^ prec / b) (ceilDiv (a * 2 ^ prec) b)

/-- `(a/b)^k` is compared in integers while `k·bitLength b` is at most this many bits (`EXACT_BITS`). -/
def exactBits : Nat := 2 ^ 14

/-- The exact comparison `(a/b)^k ≤ dn/dd`, in integers. -/
def missAtMostExact (a b k dn dd : Nat) : Bool := a ^ k * dd ≤ dn * b ^ k

/-- The bounds at `prec`, doubled up to `fuel` times, then the exact comparison. Each answer the bounds give is the exact
one; for `0 ≤ a/b < 1` in lowest terms and `k ≥ bitLength dd` they decide at some precision (`q^k = δ` would need
`b^k = dd`), and `fuel` only keeps the function total. -/
def missAtMostGo (a b k dn dd : Nat) : Nat → Nat → Bool
  | 0, _ => missAtMostExact a b k dn dd
  | fuel + 1, prec =>
    let (lo, hi) := fixedPow a b k prec
    if hi * dd ≤ dn * 2 ^ prec then true
    else if dn * 2 ^ prec < lo * dd then false
    else missAtMostGo a b k dn dd fuel (2 * prec)

/-- `miss_at_most`: `q^k ≤ δ`, exactly (for `0 ≤ q < 1` and `0 < δ`). -/
def missAtMost (q : Rat) (k : Nat) (δ : Rat) : Bool :=
  let a := q.num.toNat
  let b := q.den
  let dn := δ.num.toNat
  let dd := δ.den
  if k < bitLength dd ∨ k * bitLength b ≤ exactBits then missAtMostExact a b k dn dd
  else missAtMostGo a b k dn dd 24 (64 + 2 * bitLength k + bitLength dd)

/-- The least `k` in `(lo, hi]` with `pred k`, given `pred hi` and `pred` monotone: bisection. -/
def leastAbove (pred : Nat → Bool) (lo hi : Nat) : Nat :=
  if h : lo + 1 < hi then
    let mid := (lo + hi) / 2
    if pred mid then leastAbove pred lo mid else leastAbove pred mid hi
  else hi
termination_by hi - lo
decreasing_by all_goals omega

/-- The least `k` in `[1, n]` with `pred k` (monotone), and `n` if there is none. -/
def leastUpTo (pred : Nat → Bool) (n : Nat) : Nat :=
  if pred n then leastAbove pred 0 n else n

/-- `k_prime`'s domain: `0 < ε ≤ 1`, `0 < δ < 1`, `n, w_max ≥ 1` and `0 < credited ≤ n·w_max`. -/
def KPrimeDomain (eps δ : Rat) (credited n w : Int) : Prop :=
  0 < eps ∧ eps ≤ 1 ∧ 0 < δ ∧ δ < 1 ∧ 1 ≤ n ∧ 1 ≤ w ∧ 0 < credited ∧ credited ≤ n * w

instance instDecidableKPrimeDomain (eps δ : Rat) (credited n w : Int) : Decidable (KPrimeDomain eps δ credited n w) := by
  unfold KPrimeDomain; infer_instance

/-- `p = ε·credited/(n·w_max)`, the least share of the tiles a wrong set holding a share ε of the credited work has. -/
def missShare (eps : Rat) (credited n w : Int) : Rat := eps * credited / (n * w)

/-- `k_prime`: L3's K′, or Python's refusal. -/
def kPrime (eps δ : Rat) (credited n w : Int) : Except String Nat :=
  if KPrimeDomain eps δ credited n w then
    let p := missShare eps credited n w
    if p = 1 then .ok 1 else .ok (leastUpTo (fun k => missAtMost (1 - p) k δ) n.toNat)
  else
    .error s!"k_prime needs 0 < ε <= 1, 0 < δ < 1 and 0 < credited <= n·w_max, not {eps}, {δ}, {credited}, {n}, {w}"

/-- The `sampling` item: ε, δ, the grid and the law `subset:K′` they fix with the level (every tile does `tileWork`). -/
structure Sampling where
  eps : Rat
  delta : Rat
  gridBits : Nat
  k : Nat
  deriving DecidableEq, Repr

/-- `sampling`. -/
def sampling (eps δ : Rat) (work : Work) (tileWork : Int) (bits : Nat) : Except String Sampling := do
  let k ← kPrime eps δ (work.credited * tileWork) work.n tileWork
  pure ⟨eps, δ, bits, k⟩

end PouwWindow
