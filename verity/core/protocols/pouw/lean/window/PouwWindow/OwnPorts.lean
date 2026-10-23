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
* `readFull T k`: the ports statement `k` marks `read_in_full` (`Flock.KnownRead.READ_KEY`), lane `l`'s `partials` and
  the combine's `p<x>`: two statements' reads of one partials row, which flock-verify holds to one width read in every
  bit (`Flock.KnownRead.fullRows`).
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

/-- **The ports statement `k` reads in full** (`leaf_lanes.registered_file`'s `read_in_full`): its reads of the
partials, lane `l`'s `partials` and the combine's `p<x>`. -/
def readFull (T : Tile) (k : Option Nat) : Array String :=
  ((regReads T k).filterMap fun (nm, v, _) => if v == "partials" then some nm else none).toArray

end PouwWindow.Lanes
