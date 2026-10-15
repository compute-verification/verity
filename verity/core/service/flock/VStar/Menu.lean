import VStar.Algebra

/-!
# F6's menu: a session's shape to the inner session V* is staged over

A menu, pinned per model (`VStar.Menu.Pinned`), is a list of entries: a session shape (its slots and its nonzero bound),
the inner session that proves a session of that shape (its table's `m` and `k_log`, and its claims), the unit positions
each slot has, and its unit types' count. A shape may carry several entries (two buckets at the same slots and bound); a
session of that shape is staged over the largest of their inner sessions, by `m` and then `k_log`, at the most claims
among them, each slot has the most positions among them, and its type table holds the most of their types.

`sessions` lists each shape's inner session; `vstar menu` runs V* over them, which is how `Menu.Staged` is checked at
the pinned menus.
-/

namespace VStar.Menu

/-- One entry of a pinned menu (`menu.json`'s `slots`, `bound`, `inner_m`, `k_log`, `claims`, `positions` and its
`unit_types`' count). -/
structure Entry where
  slots : Nat
  bound : Nat
  m : Nat
  kLog : Nat
  claims : Nat
  positions : Nat
  types : Nat
  deriving Repr, DecidableEq

/-- The larger of two entries' inner sessions, by `m` and then `k_log`; `a` on a tie. -/
def Entry.max (a b : Entry) : Entry := if a.m < b.m ∨ (a.m = b.m ∧ a.kLog < b.kLog) then b else a

/-- The entries at a session shape. -/
def at_ (es : List Entry) (slots bound : Nat) : List Entry := es.filter fun e => e.slots == slots && e.bound == bound

/-- The most claims among some entries. -/
def claims (es : List Entry) : Nat := es.foldl (fun n e => Nat.max n e.claims) 0

/-- The inner session of a session with `slots` slots and nonzero bound `bound`: its entries' largest, at their most
claims; `none` off the menu. -/
def inner (es : List Entry) (slots bound : Nat) : Option Algebra.Shape :=
  match at_ es slots bound with
  | [] => none
  | e :: rest => let w := rest.foldl Entry.max e; some ⟨w.m, w.kLog, claims (e :: rest)⟩

/-- The unit positions each slot of a session shape has: the most among its entries. -/
def positions (es : List Entry) (slots bound : Nat) : Nat :=
  (at_ es slots bound).foldl (fun n e => Nat.max n e.positions) 0

/-- The type bound of a session shape: the most unit types among its entries. -/
def types (es : List Entry) (slots bound : Nat) : Nat := (at_ es slots bound).foldl (fun n e => Nat.max n e.types) 0

/-- The inner sessions of a menu's shapes, one per entry (a shape with several entries repeats its own). -/
def sessions (es : List Entry) : List Algebra.Shape := es.filterMap fun e => inner es e.slots e.bound

end VStar.Menu
