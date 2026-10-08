import PouwWindow.Draw
import PouwWindow.Public
import Lean.Data.Json

/-!
# The window's values as JSON

A number is an integer only when written without a fraction or exponent, as Python's `json` writes an `int`. The items
are written as Python writes them: `Work` as its dataclass fields, `Work.to_json`, the `sampling` item with `str(Fraction)`.
-/

namespace PouwWindow

open Lean (Json)

/-- An integer, as Python's `json` reads one (`1.0` is a float there and here). -/
def intOf? : Json → Option Int
  | .num ⟨m, 0⟩ => some m
  | _ => none

def natOf? (j : Json) : Option Nat := do
  let i ← intOf? j
  if 0 ≤ i then some i.toNat else none

def field? (j : Json) (k : String) : Option Json :=
  match j with
  | .obj _ => (j.getObjVal? k).toOption
  | _ => none

def Work.toJson (w : Work) : Json :=
  Json.mkObj [("level", w.level), ("credited", w.credited), ("n", w.n)]

/-- `Work.to_json`. -/
def Work.itemJson (w : Work) (tileWork : Int) : Json :=
  Json.mkObj [("level", w.level), ("n", w.n), ("credited_tiles", w.credited), ("credited", (w.credited : Int) * tileWork)]

/-- The `sampling` item. -/
def Sampling.toJson (s : Sampling) : Json :=
  Json.mkObj [("eps", toString s.eps), ("delta", toString s.delta), ("grid_bits", s.gridBits),
    ("law", Json.mkObj [("law", "subset"), ("k", s.k)])]

def PublicItem.toJson (p : PublicItem) : Json := Json.arr #[p.item, p.kind, p.source, p.reason]

end PouwWindow
