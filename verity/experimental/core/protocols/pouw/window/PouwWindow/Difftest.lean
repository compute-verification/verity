import PouwWindow.Json

/-!
# The difftest: the window's Lean against `window.py`, on generated inputs

`scripts/difftest_vectors.py` writes instances, one JSON object per line with its `op`, and `window.py`'s answers on each.
`PouwWindow/DifftestMain.lean` reads both and fails unless `evaluate` gives every answer. A refusal is
`{"refused": reason}`, Python's `ValueError` message or the window statement's reason, so the test holds which check
refused as well as whether one did; a value is `{"ok": value}`.

* `grid` (`bits`, `n`, `ell`): `on_grid`, `floor_point`, `pad`, `level` and `bucket` of `n`, and `point` and `Work.at` of
  the level `ell`;
* `levels` (`bits`, `lo`, `hi`);
* `k_prime` (`eps`, `delta` as `[num, den]`, `credited`, `n`, `w_max`);
* `sampling` (`eps`, `delta`, `work`, `tile_work`, `bits`), and `Work.to_json`;
* `window` (`entries`, `work`, `bound`, `bits`, `draws`, `within`): `window_problem`, and on a window it accepts `tile_at`
  of each draw and `tile_in` of each `[entry, index]`;
* `constants`: `PUBLIC`, `REGISTERED`, `KINDS` and `GRID_BITS`.

`evaluate` is one instance to one answer, so a harness that drives the Python reference and this executable over other
inputs needs only the same reader and writer.
-/

namespace PouwWindow.Difftest

open Lean (Json ToJson toJson)

def result {α : Type} (f : α → Json) : Except String α → Json
  | .ok v => Json.mkObj [("ok", f v)]
  | .error e => Json.mkObj [("refused", e)]

def need {α : Type} (what : String) : Option α → Except String α
  | some v => .ok v
  | none => .error s!"the instance's {what} is malformed"

def get (j : Json) (k : String) : Except String Json := need k (field? j k)
def int (j : Json) (k : String) : Except String Int := do need k (intOf? (← get j k))
def nat (j : Json) (k : String) : Except String Nat := do need k (natOf? (← get j k))

/-- A rational `[num, den]`, `den > 0`. -/
def rat (j : Json) (k : String) : Except String Rat := do
  let .arr #[n, d] ← get j k | throw s!"the instance's {k} is not [num, den]"
  let n ← need k (intOf? n)
  let d ← need k (natOf? d)
  if d = 0 then throw s!"the instance's {k} has den 0"
  pure (mkRat n d)

def work (j : Json) : Except String Work := do
  let w ← get j "work"
  pure ⟨← nat w "level", ← nat w "credited", ← nat w "n"⟩

def arr (j : Json) (k : String) : Except String (Array Json) := do
  let .arr a ← get j k | throw s!"the instance's {k} is not a list"
  pure a

def workJson : Work → Json := Work.toJson

/-- What the window's Lean gives on one instance, in the shape of `window.py`'s answers. -/
def evaluate (j : Json) : Except String Json := do
  let .str op ← get j "op" | throw "an instance's op is a string"
  match op with
  | "grid" =>
    let bits ← nat j "bits"
    let n ← int j "n"
    let ell ← int j "ell"
    pure <| Json.mkObj [("on_grid", toJson (onGridInt bits n)), ("floor_point", result toJson (floorPoint? bits n)),
      ("pad", result toJson (pad? bits n)), ("level", result toJson (level? bits n)),
      ("point", result toJson (point? bits ell)), ("work_at", result workJson (Work.at? bits ell)),
      ("bucket", result workJson (bucket? bits n))]
  | "levels" =>
    pure <| Json.mkObj [("levels", result toJson (levels? (← nat j "bits") (← int j "lo") (← int j "hi")))]
  | "k_prime" =>
    let k := kPrime (← rat j "eps") (← rat j "delta") (← int j "credited") (← int j "n") (← int j "w_max")
    pure <| Json.mkObj [("k_prime", result toJson k)]
  | "sampling" =>
    let w ← work j
    let tw ← int j "tile_work"
    let s := sampling (← rat j "eps") (← rat j "delta") w tw (← nat j "bits")
    pure <| Json.mkObj [("sampling", result Sampling.toJson s), ("to_json", w.itemJson tw)]
  | "window" =>
    let es := (← arr j "entries").toList.map rawOf
    let p := windowProblem es (← work j) (← int j "bound") (← nat j "bits")
    let draws ← (← arr j "draws").toList.mapM fun t => need "draw" (natOf? t)
    let within ← (← arr j "within").toList.mapM fun x => do
      let .arr #[e, t] := x | throw "a within is [entry, index]"
      pure (← need "within" (natOf? e), ← need "within" (natOf? t))
    let at3 := fun (e, i, k) => Json.arr #[toJson e, toJson i, toJson k]
    let at2 := fun (i, k) => Json.arr #[toJson i, toJson k]
    pure <| Json.mkObj [("problem", match p with | some q => toJson q.text | none => Json.null),
      ("tile_at", Json.arr (draws.map fun t => result at3 (tileAt? es t)).toArray),
      ("tile_in", Json.arr (within.map fun (e, t) => result at2 (tileIn? (es.getD e ⟨none, none, none, none, none⟩) t)).toArray)]
  | "constants" =>
    pure <| Json.mkObj [("PUBLIC", Json.arr (publicItems.map PublicItem.toJson).toArray),
      ("REGISTERED", toJson registered), ("KINDS", toJson (kinds.map Kind.name)), ("GRID_BITS", toJson gridBits)]
  | op => throw s!"no op {op}"

def lines (path : System.FilePath) : IO (Array String) := do
  pure ((← IO.FS.lines path).filter (· ≠ ""))

/-- `pouw-window-difftest INSTANCES PYTHON N`: every one of the N instances gives Python's answer. -/
def main (args : List String) : IO UInt32 := do
  let [ip, pp, n] := args | IO.eprintln "usage: pouw-window-difftest INSTANCES PYTHON N"; return 2
  let insts ← lines ip
  let want ← lines pp
  let some n := n.toNat? | IO.eprintln s!"not a count: {n}"; return 2
  unless insts.size = n ∧ want.size = n do
    IO.eprintln s!"expected {n} instances and answers, read {insts.size} and {want.size}"; return 1
  let mut bad := 0
  let mut ops : Std.HashMap String Nat := {}
  for i in [0:n] do
    let got := do evaluate (← Json.parse insts[i]!)
    let exp := Json.parse want[i]!
    let op := ((Json.parse insts[i]!).toOption.bind (field? · "op")).bind (·.getStr?.toOption) |>.getD "?"
    ops := ops.insert op (ops.getD op 0 + 1)
    match got, exp with
    | .ok g, .ok e =>
      if g.compress ≠ e.compress then
        bad := bad + 1
        if bad ≤ 10 then IO.eprintln s!"instance {i} ({op}): Lean {g.compress}\n  Python {e.compress}"
    | .error g, _ => bad := bad + 1; if bad ≤ 10 then IO.eprintln s!"instance {i}: Lean could not evaluate it: {g}"
    | _, .error e => bad := bad + 1; if bad ≤ 10 then IO.eprintln s!"answer {i}: not JSON: {e}"
  let counts := ", ".intercalate (ops.toList.map fun (k, v) => s!"{k} {v}")
  if bad = 0 then
    IO.println s!"pouw-window difftest: all {n} instances agree ({counts})"; return 0
  IO.eprintln s!"pouw-window difftest: {bad} of {n} instances differ"; return 1

end PouwWindow.Difftest
