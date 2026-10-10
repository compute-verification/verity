import VStar.Algebra

/-!
# F6's menu: a session's shape to the inner session V* is staged over

A menu, pinned per model (`VStar.Menu.Pinned`), is a list of entries: a session shape (its slots and its nonzero bound),
the inner session that proves a session of that shape (its table's `m` and `k_log`, with two claims) and its unit types'
count. A shape may carry several entries (two buckets at the same slots and bound); a session of that shape is staged
over the largest of their inner sessions, by `m` and then `k_log`, and its type table holds the most of their types.
-/

namespace VStar.Menu

/-- One entry of a pinned menu (`menu.json`'s `slots`, `bound`, `inner_m`, `k_log` and its `unit_types`' count). -/
structure Entry where
  slots : Nat
  bound : Nat
  m : Nat
  kLog : Nat
  types : Nat
  deriving Repr, DecidableEq

/-- The entry's inner session: its `m` and `k_log`, two claims. -/
def Entry.inner (e : Entry) : Algebra.Shape := ⟨e.m, e.kLog, 2⟩

/-- The larger of two entries' inner sessions, by `m` and then `k_log`; `a` on a tie. -/
def Entry.max (a b : Entry) : Entry := if a.m < b.m ∨ (a.m = b.m ∧ a.kLog < b.kLog) then b else a

/-- The entries at a session shape. -/
def at_ (es : List Entry) (slots bound : Nat) : List Entry := es.filter fun e => e.slots == slots && e.bound == bound

/-- The inner session of a session with `slots` slots and nonzero bound `bound`: its entries' largest; `none` off the
menu. -/
def inner (es : List Entry) (slots bound : Nat) : Option Algebra.Shape :=
  match at_ es slots bound with
  | [] => none
  | e :: rest => some (rest.foldl Entry.max e).inner

/-- The type bound of a session shape: the most unit types among its entries. -/
def types (es : List Entry) (slots bound : Nat) : Nat := (at_ es slots bound).foldl (fun n e => Nat.max n e.types) 0

end VStar.Menu
