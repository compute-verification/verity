import PouwWindow.Lanes
import Flock.Registered

/-!
# A drawn H1 leaf's registered data ports

The ports of the verifier's own `--registered` file (`Flock.Registered.Port`) that a drawn leaf's statement reads at a
registered value's row (`Source.read`): `leaf_lanes.registered_file` at `leaf_lanes.reads`, whose rows
`tests/test_lanes.py` holds `lanePorts`' and `combinePorts`' sources to. Each value has one tree, the same in all 33
statements (`roots v`: its registered root, leaf count, row width, domain and schema), read at the layout's row:

* `ownOf roots T k`: statement `k`'s data ports (lane `l`'s at `some l`, the combine's at `none`), port `name` reading
  row `row` of `roots v` (`reads := #[some row]`, the statement's one unit);
* `readFull T k`: the ports statement `k` marks `read_in_full` (`Flock.KnownRead.READ_KEY`), its reads of an A row, a B
  row or a partials row (`fullValue`: lane `l`'s `a<s>`, `b<t>` and `partials`, the combine's `p<x>`), each row read by
  several statements, which flock-verify holds to one width read in every bit (`Flock.KnownRead.fullRows`);
* `checkOwn want own`: the leaf verifier's check that a statement's parsed `--registered` file (`own`) holds the ports
  `want` (`ownOf` at the roots it takes from the accepted registration and produced record, `PouwWindow.Lanes.rootsOf?`),
  each the file's first port of its name (`checkOwn_ok`), which `pouw-leaf-own` runs on every statement's file.
-/

namespace PouwWindow.Lanes

/-- Statement `k`'s ports: lane `l`'s at `some l` (`lanePorts`), the combine's at `none` (`combinePorts`). -/
def portsOf (T : Tile) : Option Nat → List Port
  | some l => lanePorts T l
  | none => combinePorts T

/-- **Statement `k`'s registered reads** (`leaf_lanes.reads`): each port opened at a registered value's row, its name,
the value and the row. -/
def regReads (T : Tile) (k : Option Nat) : List (String × String × Nat) :=
  (portsOf T k).filterMap fun p => match p.source with
    | .read v row => some (p.name, v, row)
    | .known _ => none

/-- **Statement `k`'s registered data ports** (`leaf_lanes.registered_file`'s ports of `reads`): port `name` reads row
`row` of the value `v`, registered once at `roots v` for all 33 statements. -/
def ownOf (roots : String → Flock.Registered.Port) (T : Tile) (k : Option Nat) : List Flock.Registered.Port :=
  (regReads T k).map fun (nm, v, row) => { roots v with name := nm, value := v, reads := #[some row] }

/-- **The values a statement reads in full** (`leaf_lanes.FULL`): the A rows, the B rows and the partials, each row
read by several statements. -/
def fullValue (v : String) : Bool := v == "a" || v == "b" || v == "partials"

/-- **The ports statement `k` reads in full** (`leaf_lanes.registered_file`'s `read_in_full`): its reads of a
`fullValue`, lane `l`'s `a<s>`, `b<t>` and `partials` and the combine's `p<x>`. -/
def readFull (T : Tile) (k : Option Nat) : Array String :=
  ((regReads T k).filterMap fun (nm, v, _) => if fullValue v then some nm else none).toArray

/-- Two registered ports are one: every field equal, a byte string's by its bytes. -/
def portEq (p q : Flock.Registered.Port) : Bool :=
  p.name == q.name && p.value == q.value && p.root.data == q.root.data && p.leaves == q.leaves && p.words == q.words &&
    p.wordBits == q.wordBits && p.domain.data == q.domain.data && p.reads == q.reads && p.schema == q.schema

theorem eq_of_portEq {p q : Flock.Registered.Port} (h : portEq p q = true) : p = q := by
  obtain ⟨n, v, ⟨r⟩, l, w, b, ⟨d⟩, rd, s⟩ := p
  obtain ⟨n', v', ⟨r'⟩, l', w', b', ⟨d'⟩, rd', s'⟩ := q
  simp only [portEq, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨rfl, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩ := h
  rfl

/-- `own` holds the port `p`: its first port of `p`'s name is `p`. -/
def holds (own : Array Flock.Registered.Port) (p : Flock.Registered.Port) : Bool :=
  match own.toList.find? (·.name == p.name) with
  | some q => portEq q p
  | none => false

/-- **The registered file holds the ports `want`** (`ownOf` at the verifier's roots, `rootsOf?`): the file's first port
of each such port's name is that port, field for field; else the first port that it doesn't hold, named. -/
def checkOwn (want : List Flock.Registered.Port) (own : Array Flock.Registered.Port) : Except String Unit :=
  match want.find? fun p => !holds own p with
  | none => .ok ()
  | some p => .error s!"registered reads: port {p.name} is not the layout's read of {p.value} at the registered root"

theorem checkOwn_ok {want : List Flock.Registered.Port} {own : Array Flock.Registered.Port}
    (h : checkOwn want own = .ok ()) : ∀ p ∈ want, own.toList.find? (·.name == p.name) = some p := by
  intro p hp
  have hn : want.find? (fun p => !holds own p) = none := by
    unfold checkOwn at h
    split at h
    · assumption
    · cases h
  have hh : holds own p = true := by simpa using List.find?_eq_none.1 hn p hp
  unfold holds at hh
  split at hh
  · next q hq => rw [hq, eq_of_portEq hh]
  · cases hh

end PouwWindow.Lanes
