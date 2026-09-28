/-!
# The grid a window's tile count is padded to (`window.py`, "The grid")

The grid of `s` bits is the positive integers with at most `s + 1` significant bits. A window of `real` real tiles
registers one level ℓ, the largest point `g_ℓ ≤ real`, is credited `g_ℓ` tiles and pads to `g_(ℓ+1)` (`bucket`).
Python's `n >> e << e` and `1 << b` are `n / 2 ^ e * 2 ^ e` and `2 ^ b` here.
-/

namespace PouwWindow

/-- `GRID_BITS`: s = 3. -/
def gridBits : Nat := 3

/-- Python's `int.bit_length`. -/
def bitLength (n : Nat) : Nat := if n = 0 then 0 else n.log2 + 1

/-- `_shift`: how many low bits a grid point at n's magnitude has zero, `max(0, bit_length − bits − 1)`. -/
def shift (bits n : Nat) : Nat := bitLength n - bits - 1

/-- `on_grid` on a natural number. -/
def onGrid (bits n : Nat) : Bool := 0 < n && n % 2 ^ shift bits n == 0

/-- `floor_point`: the largest grid point at most n (for n ≥ 1). -/
def floorPoint (bits n : Nat) : Nat := n / 2 ^ shift bits n * 2 ^ shift bits n

/-- `level`: grid point g's index, from 1 at level 0 (for g on the grid). -/
def level (bits g : Nat) : Nat :=
  if g < 2 ^ (bits + 1) then g - 1
  else 2 ^ (bits + 1) - 1 + (shift bits g - 1) * 2 ^ bits + g / 2 ^ shift bits g - 2 ^ bits

/-- `point`: the grid point at level ℓ, `level`'s inverse. -/
def point (bits ell : Nat) : Nat :=
  if ell < 2 ^ (bits + 1) - 1 then ell + 1
  else
    let r := ell - (2 ^ (bits + 1) - 1)
    (2 ^ bits + r % 2 ^ bits) * 2 ^ (r / 2 ^ bits + 1)

/-- `pad`: the least grid point at least n (for n ≥ 1), a unit count's public bucket. -/
def pad (bits n : Nat) : Nat :=
  let g := floorPoint bits n
  if g = n then g else point bits (level bits g + 1)

/-- `levels`: q, the grid levels a tile count in [lo, hi] can register (for lo, hi ≥ 1; not positive when hi < lo, as in
Python). -/
def levels (bits lo hi : Nat) : Int :=
  (level bits (floorPoint bits hi) : Int) - level bits (floorPoint bits lo) + 1

/-- `Work`: a window's work level; it is credited `credited` tiles, `point level`, and pads to `n`, `point (level + 1)`. -/
structure Work where
  level : Nat
  credited : Nat
  n : Nat
  deriving DecidableEq, Repr

/-- `Work.at`. -/
def Work.at (bits ell : Nat) : Work := ⟨ell, point bits ell, point bits (ell + 1)⟩

/-- `bucket`: the level of a window of `real` real tiles, credited `floorPoint real` and padded to the next point. -/
def bucket (bits real : Nat) : Work := Work.at bits (level bits (floorPoint bits real))

/-! Python's functions on any integer, with its refusals (`ValueError`'s message). -/

/-- `on_grid` on an integer: false off the positive integers. -/
def onGridInt (bits : Nat) (n : Int) : Bool := 1 ≤ n && onGrid bits n.toNat

def floorPoint? (bits : Nat) (n : Int) : Except String Nat :=
  if 1 ≤ n then .ok (floorPoint bits n.toNat) else .error s!"the grid is over positive integers, not {n}"

def level? (bits : Nat) (g : Int) : Except String Nat :=
  if onGridInt bits g then .ok (level bits g.toNat) else .error s!"{g} is not a point of the {bits}-bit grid"

def point? (bits : Nat) (ell : Int) : Except String Nat :=
  if 0 ≤ ell then .ok (point bits ell.toNat) else .error s!"a level is a natural number, not {ell}"

def pad? (bits : Nat) (n : Int) : Except String Nat := do
  let _ ← floorPoint? bits n
  pure (pad bits n.toNat)

/-- Python evaluates `floor_point(hi)` first, so a refusal of both names hi. -/
def levels? (bits : Nat) (lo hi : Int) : Except String Int := do
  let _ ← floorPoint? bits hi
  let _ ← floorPoint? bits lo
  pure (levels bits lo.toNat hi.toNat)

def Work.at? (bits : Nat) (ell : Int) : Except String Work := do
  let _ ← point? bits ell
  pure (Work.at bits ell.toNat)

def bucket? (bits : Nat) (real : Int) : Except String Work := do
  let _ ← floorPoint? bits real
  pure (bucket bits real.toNat)

end PouwWindow
