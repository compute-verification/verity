import PouwWindow.Grid

/-!
# The advice and the window statement (`window.py`, "The advice"; L2, L4)

One entry per committed call, then one filler entry, then empty entries up to the public bound. A call entry holds its
sides' row counts and the rows its tiles keep, and its tile range `[lo, hi)`; its tiles are the kept rows' products,
row-major. `windowProblem` is the window statement on the entries as read (`Raw`), with Python's checks in Python's order
and its reasons (`Problem.text`); `tileIn` and `tileAt` map an index of a window it accepts to its tile.

An entry is read from JSON (`PouwWindow.Json`) into `Raw`, which keeps exactly what the checks read: its kind when it is an
object whose `kind` is one of the three, `lo` and `hi` when they are natural numbers, and each side when it is an object
whose `rows` is a natural number and whose `keep` is a list of them. Anything else is `none`, which fails the check that
reads it as Python's does: a negative `lo` or `hi` never equals or exceeds the running end, which is at least 0, and a
negative row count or kept row is no side's.
-/

namespace PouwWindow

inductive Kind where
  | call
  | filler
  | none
  deriving DecidableEq, Repr

/-- `KINDS`. -/
def kinds : List Kind := [.call, .filler, .none]

def Kind.name : Kind → String
  | .call => "call"
  | .filler => "filler"
  | .none => "none"

def Kind.ofName? (s : String) : Option Kind := kinds.find? (·.name == s)

/-- One side of an entry: its row count and the rows its tiles keep. -/
structure Side where
  rows : Nat
  keep : List Nat
  deriving DecidableEq, Repr

/-- `_keep_ok`: at least one kept row, each a row of the side, strictly ascending, and at least one row. -/
def Side.ok (s : Side) : Bool :=
  1 ≤ s.rows && !s.keep.isEmpty && s.keep.all (· < s.rows) && (s.keep.zip s.keep.tail).all fun (x, y) => x < y

/-- An entry as the window statement reads it. -/
structure Raw where
  kind : Option Kind
  lo : Option Nat
  hi : Option Nat
  a : Option Side
  b : Option Side
  deriving DecidableEq, Repr

/-- `tiles`: the kept rows' products (none for an empty entry). -/
def Raw.tiles (e : Raw) : Nat :=
  if e.kind = some .none then 0 else ((e.a.map (·.keep.length)).getD 0) * ((e.b.map (·.keep.length)).getD 0)

/-- Why the window statement refuses, with what Python's reason names. -/
inductive Problem where
  | bound (entries : Nat) (bound : Int)
  | kind
  | order
  | keep (e : Nat)
  | range (e : Nat)
  | callTiles (e size tiles : Nat)
  | fillerTiles (e size tiles : Nat)
  | ends (at_ n : Nat)
  | level (real realLevel level : Nat)
  deriving DecidableEq, Repr

/-- Python's reason, word for word. -/
def Problem.text : Problem → String
  | .bound m b => s!"{m} entries, not the bound {b}"
  | .kind => "an entry's kind is none of ('call', 'filler', 'none')"
  | .order => "the entries are the calls, then one filler, then empty entries"
  | .keep e => s!"entry {e}: its sides' kept rows are not ascending rows of the side"
  | .range e => s!"entry {e}: its range does not start where the last ended, or is empty"
  | .callTiles e size t => s!"entry {e}: its range is {size} tiles, not its {t} kept tiles"
  | .fillerTiles e size t => s!"entry {e}: the filler's range is {size} tiles, beyond its {t}"
  | .ends at_ n => s!"the ranges end at {at_}, not the padded count {n}"
  | .level real l wl => s!"{real} real tiles are level {l}, not the registered level {wl}"

/-- The entries' shape: some calls (at least one), then one filler, then empty entries. -/
def ordered (ks : List (Option Kind)) : Option Nat :=
  match ks.findIdx? (· == some .filler) with
  | none => none
  | some calls =>
    if 1 ≤ calls ∧ (ks.take calls).all (· == some .call) ∧ (ks.drop (calls + 1)).all (· == some .none) then some calls
    else none

/-- The calls' and the filler's checks, in order from entry `e` with the ranges so far ending at `at_`: where the ranges
end, or the first problem. -/
def rangesFrom : Nat → Nat → List Raw → Except Problem Nat
  | _, at_, [] => .ok at_
  | e, at_, x :: xs =>
    match x.a, x.b with
    | some a, some b =>
      if !(a.ok && b.ok) then .error (.keep e) else
      match x.lo, x.hi with
      | some lo, some hi =>
        if lo ≠ at_ ∨ hi ≤ at_ then .error (.range e) else
        if x.kind = some .call ∧ hi - lo ≠ x.tiles then .error (.callTiles e (hi - lo) x.tiles) else
        if x.kind = some .filler ∧ x.tiles < hi - lo then .error (.fillerTiles e (hi - lo) x.tiles) else
        rangesFrom (e + 1) hi xs
      | _, _ => .error (.range e)
    | _, _ => .error (.keep e)

/-- `window_problem`: the window statement (L4) on the advice, or `none`: `bound` entries, the calls, then one filler, then
empty ones; the ranges tile `[0, work.n)` in order; a call's range is its kept tiles; the filler's holds the rest, within
its rows; and the level is `bucket` of the calls' tiles. -/
def windowProblem (es : List Raw) (work : Work) (bound : Int) (bits : Nat) : Option Problem :=
  if (es.length : Int) ≠ bound then some (.bound es.length bound)
  else if es.any (·.kind.isNone) then some .kind
  else match ordered (es.map (·.kind)) with
    | none => some .order
    | some calls =>
      match rangesFrom 0 0 (es.take (calls + 1)) with
      | .error p => some p
      | .ok at_ =>
        let real := ((es.getD calls ⟨none, none, none, none, none⟩).lo).getD 0
        if at_ ≠ work.n then some (.ends at_ work.n)
        else if bucket bits real ≠ work then some (.level real (bucket bits real).level work.level)
        else none

/-- `tile_in`: (A row, B row) of index t in its entry's range, the kept rows' products row-major; `none` outside it. -/
def tileIn (e : Raw) (t : Nat) : Option (Nat × Nat) :=
  match e.kind, e.lo, e.hi with
  | some k, some lo, some hi =>
    if k = .none ∨ ¬(lo ≤ t ∧ t < hi) then none
    else
      let ka := (e.a.map (·.keep)).getD []
      let kb := (e.b.map (·.keep)).getD []
      let u := t - lo
      some (ka.getD (u / kb.length) 0, kb.getD (u % kb.length) 0)
  | _, _, _ => none

/-- `tile_at`: (entry, A row, B row) of drawn index t, from the first non-empty entry whose range holds it. -/
def tileAt (es : List Raw) (t : Nat) : Option (Nat × Nat × Nat) :=
  go 0 es
where
  go : Nat → List Raw → Option (Nat × Nat × Nat)
    | _, [] => none
    | e, x :: xs => match tileIn x t with
      | some (i, j) => some (e, i, j)
      | none => go (e + 1) xs

/-- `tile_in`'s refusal. -/
def tileIn? (e : Raw) (t : Nat) : Except String (Nat × Nat) :=
  match tileIn e t with
  | some ij => .ok ij
  | none => .error s!"index {t} is not in the entry's range"

/-- `tile_at`'s refusal. -/
def tileAt? (es : List Raw) (t : Nat) : Except String (Nat × Nat × Nat) :=
  match tileAt es t with
  | some r => .ok r
  | none => .error s!"index {t} is in no entry's range"

end PouwWindow
