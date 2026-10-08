import Difftest
import PouwWindow.Json

/-!
# The difftest's functions: the window's Lean behind `Difftest.serve`

One function per ported function of `window.py`, each on a JSON object of its arguments, for the shared harness
(`verity/core/difftest`): an output, or `.error` where Python raises `ValueError` (a reject). The port's
suite (`tests/test_difftest.py`) draws the inputs and holds each to Python.

* `on_grid`, `floor_point`, `pad`, `level`, `bucket` (`bits`, `n`); `point`, `work_at` (`bits`, `ell`); `levels` (`bits`,
  `lo`, `hi`);
* `k_prime` (`eps`, `delta` as `[num, den]`, `credited`, `n`, `w_max`); `sampling` (`eps`, `delta`, `work`, `tile_work`,
  `bits`); `work_to_json` (`work`, `tile_work`);
* `constants`: `PUBLIC`, `REGISTERED`, `KINDS` and `GRID_BITS`.
-/

namespace PouwWindow.Difftest

open Lean (Json ToJson toJson)

def need {α : Type} (what : String) : Option α → Except String α
  | some v => .ok v
  | none => .error s!"the input's {what} is malformed"

def get (j : Json) (k : String) : Except String Json := need k (field? j k)
def int (j : Json) (k : String) : Except String Int := do need k (intOf? (← get j k))
def nat (j : Json) (k : String) : Except String Nat := do need k (natOf? (← get j k))

/-- A rational `[num, den]`, `den > 0`. -/
def rat (j : Json) (k : String) : Except String Rat := do
  let .arr #[n, d] ← get j k | throw s!"the input's {k} is not [num, den]"
  let n ← need k (intOf? n)
  let d ← need k (natOf? d)
  if d = 0 then throw s!"the input's {k} has den 0"
  pure (mkRat n d)

def work (j : Json) : Except String Work := do
  let w ← get j "work"
  pure ⟨← nat w "level", ← nat w "credited", ← nat w "n"⟩

def arr (j : Json) (k : String) : Except String (Array Json) := do
  let .arr a ← get j k | throw s!"the input's {k} is not a list"
  pure a

/-- A function of `(bits, n)`. -/
def ofN {α : Type} [ToJson α] (f : Nat → Int → Except String α) : _root_.Difftest.Fn := fun j => do
  pure (toJson (← f (← nat j "bits") (← int j "n")))

def table : String → Option _root_.Difftest.Fn := _root_.Difftest.table [
  ("on_grid", fun j => do pure (toJson (onGridInt (← nat j "bits") (← int j "n")))),
  ("floor_point", ofN floorPoint?),
  ("pad", ofN pad?),
  ("level", ofN level?),
  ("bucket", fun j => do pure ((← bucket? (← nat j "bits") (← int j "n")).toJson)),
  ("point", fun j => do pure (toJson (← point? (← nat j "bits") (← int j "ell")))),
  ("work_at", fun j => do pure ((← Work.at? (← nat j "bits") (← int j "ell")).toJson)),
  ("levels", fun j => do pure (toJson (← levels? (← nat j "bits") (← int j "lo") (← int j "hi")))),
  ("k_prime", fun j => do
    pure (toJson (← kPrime (← rat j "eps") (← rat j "delta") (← int j "credited") (← int j "n") (← int j "w_max")))),
  ("sampling", fun j => do
    pure (← sampling (← rat j "eps") (← rat j "delta") (← work j) (← int j "tile_work") (← nat j "bits")).toJson),
  ("work_to_json", fun j => do pure ((← work j).itemJson (← int j "tile_work"))),
  ("constants", fun _ => pure <| Json.mkObj [("PUBLIC", Json.arr (publicItems.map PublicItem.toJson).toArray),
    ("REGISTERED", toJson registered), ("KINDS", toJson kinds), ("GRID_BITS", toJson gridBits)])]

end PouwWindow.Difftest
