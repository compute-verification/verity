/-!
# The openings' pooled bound

`audit.pooled`'s horizon reading of a session's openings, in exact rationals: the most wrong receipts
W = Σ_t f_t·U_t its windows' draws leave except with δ, each window's share at most min(1 − s_t/U_t, B/s_t) (`floors`),
and the largest of those (`floor`). Window t has U_t receipts and drew s_t of them.

The budget B stands for d̄·S = S·(1 − δ^(1/S)), S = Σ_t s_t, from above: B = S·(1 − q) for q = m/2^`ROOT_BITS`
whose S-th power is at most δ, certified in integers (`below`: the power by squaring, each product rounded up at
`POW_BITS`). Since x ↦ x^S is monotone, q ≤ δ^(1/S), so B ≥ S·(1 − δ^(1/S)). m is the largest such numerator the
bisection finds (`root`), so B exceeds S·(1 − δ^(1/S)) by at most S·2^-`ROOT_BITS` and the power's rounding.

W is the greedy over the windows by U_t/s_t falling (`order`, `fill`): each takes the largest share its cap and the
budget left allow, the fractional knapsack's optimum. `Properties.PoubCheck.PooledCovers` states that W is at least
`Accounting.Poub.Draw.pooled`, the bound `PooledDraw` reads. `pooled.py` is held to this by the difftest
(`tests/test_pooled.py`). Code: Lean core only.
-/

namespace PoubCheck.Pooled

/-- q's numerator's bits: q = m/2^64. -/
def ROOT_BITS : Nat := 64

/-- The bits each product of the certified power is rounded up to. -/
def POW_BITS : Nat := 256

/-- ⌈a·b/2^p⌉. -/
def mulUp (p a b : Nat) : Nat := (a * b + 2 ^ p - 1) / 2 ^ p

/-- At least (x/2^p)^n, times 2^p: the power by squaring, each product rounded up (`mulUp`). -/
def powUp (p x n : Nat) : Nat :=
  if h : n = 0 then 2 ^ p
  else
    let r := powUp p x (n / 2)
    let sq := mulUp p r r
    if n % 2 = 1 then mulUp p sq x else sq
termination_by n
decreasing_by omega

/-- Whether (m/2^`ROOT_BITS`)^S ≤ δ, certified: the rounded-up power is at most δ·2^`POW_BITS`. -/
def below (S : Nat) (δ : Rat) (m : Nat) : Bool :=
  decide (((powUp POW_BITS (m * 2 ^ (POW_BITS - ROOT_BITS)) S : Nat) : Int) * δ.den ≤ δ.num * 2 ^ POW_BITS)

/-- The bisection over [lo, hi), keeping lo where `below` holds and hi where it doesn't. -/
def bisect (S : Nat) (δ : Rat) : Nat → Nat → Nat → Nat
  | 0, lo, _ => lo
  | fuel + 1, lo, hi =>
    if hi ≤ lo + 1 then lo
    else if below S δ ((lo + hi) / 2) then bisect S δ fuel ((lo + hi) / 2) hi
    else bisect S δ fuel lo ((lo + hi) / 2)

/-- q's numerator m: the bisection's over [0, 2^`ROOT_BITS`), where `below` holds there, else 0. -/
def root (S : Nat) (δ : Rat) : Nat :=
  let m := bisect S δ (ROOT_BITS + 1) 0 (2 ^ ROOT_BITS)
  if below S δ m then m else 0

/-- B = S·(1 − m/2^`ROOT_BITS`), at least S·(1 − δ^(1/S)). -/
def budget (S : Nat) (δ : Rat) : Rat := (S : Rat) * (1 - (root S δ : Rat) / ((2 ^ ROOT_BITS : Nat) : Rat))

/-- A window's receipts per draw, U/s. -/
def ratio (w : Nat × Nat) : Rat := (w.1 : Rat) / (w.2 : Rat)

/-- The windows' indices by U_t/s_t falling, ties in window order. -/
def order (ws : Array (Nat × Nat)) : List Nat :=
  (List.range ws.size).mergeSort fun i j => decide (ratio ws[j]! ≤ ratio ws[i]!)

/-- The most of window `w`'s receipts that can be wrong, as a share, with `b` of the budget left:
min(1 − s/U, b/s). -/
def share (w : Nat × Nat) (b : Rat) : Rat := min (1 - (w.2 : Rat) / (w.1 : Rat)) (b / (w.2 : Rat))

/-- The greedy over the windows `is`, in order, with `b` of the budget left: each takes its largest share f, f·U wrong
receipts, at f·s of the budget. -/
def fill (ws : Array (Nat × Nat)) : List Nat → Rat → Rat
  | [], _ => 0
  | i :: rest, b =>
    let f := share ws[i]! b
    f * (ws[i]!.1 : Rat) + fill ws rest (max 0 (b - f * (ws[i]!.2 : Rat)))

/-- Why windows have no pooled bound: the first that draws no receipt, or more than it has. -/
def refusal (ws : Array (Nat × Nat)) : Option String :=
  (List.range ws.size).findSome? fun t =>
    if ws[t]!.2 = 0 then some s!"window {t} draws no receipt"
    else if ws[t]!.1 < ws[t]!.2 then some s!"window {t} draws {ws[t]!.2} receipts of its {ws[t]!.1}"
    else none

/-- The pooled bound: the budget B, each window's share at most (`floors`), the wrong receipts W and the largest
share. -/
structure Bound where
  budget : Rat
  floors : Array Rat
  wrong : Rat
  floor : Rat

/-- The pooled bound of windows `ws`, window t's receipts U_t and draws s_t, at δ; or why there is none. -/
def pooled (ws : Array (Nat × Nat)) (δ : Rat) : Except String Bound := do
  if ws.size = 0 then throw "a session has one or more windows"
  if δ ≤ 0 ∨ 1 ≤ δ then throw "δ is above 0 and below 1"
  if let some why := refusal ws then throw why
  let B := budget (ws.toList.map (·.2)).sum δ
  let floors := ws.map (share · B)
  pure ⟨B, floors, fill ws (order ws) B, floors.foldl max 0⟩

end PoubCheck.Pooled
