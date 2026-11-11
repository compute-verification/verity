/-
A JSON value compared with a string by a match. A check that compares this way has properties a proof can state and
close; one that compares with `==` on JSON values doesn't (`strIs`).
-/
module

public import Lean.Data.Json

@[expose] public section

namespace Flock.J

open Lean (Json)

/-- `j == .str s`, by a match the kernel reduces: `BEq Json` is a `partial def`, opaque to the kernel, so a property of
a check that compares with it can't be proved. -/
def strIs (j : Json) (s : String) : Bool := match j with | .str t => t == s | _ => false

example : strIs (.str "composite") "composite" = true := by decide
example : strIs (.str "batch") "composite" = false := by decide
example : strIs (.arr #[.str "composite"]) "composite" = false := by decide
example : strIs .null "" = false := by decide
example : strIs (.obj (.ofList [("form", Json.str "composite")])) "composite" = false := by decide

end Flock.J
