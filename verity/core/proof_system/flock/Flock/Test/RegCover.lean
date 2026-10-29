/-
`Registered.checkCover`: with the verifier's own program and registered reads, every parameter of the circuit's template
that reads a `RegisteredInput<w>_v1` root node of the program is read by a port of the reads (by its name, or its `param`);
an uncovered one is refused by name, and so is a registered-reading program under a partition that is not a template
query or a circuit whose META names no template. `ProvedScope.programHeld`: a session verified without the verifier's
program is refused. The programs are `Flock.Test.RegProgram`'s: in `reg`, `TWiden16_v1`'s `a` reads the
`RegisteredInput16_v1` node and `TStep_v1`'s `k` the `RegisteredInput32_v1` node; `TPair_v1` reads neither.
-/
import Flock.Test.RegProgram
import Flock.ProvedScope

namespace Flock.Test.RegCover

open Lean (Json)
open Flock.Test.RegProgram (reg plain)

def P : Program.Prog := (Program.decode reg).toOption.get!

def names (t : String) : Array String := (Registered.regParams P t).map (·.1)

-- the descriptor decodes, and its registered reads are found per template from its root's nodes
#guard (Program.decode reg).toBool
#guard Registered.regRoots P
#guard !Registered.regRoots ((Program.decode plain).toOption.get!)
#guard names "TWiden16_v1" == #["a"]
#guard names "TStep_v1" == #["k"]
#guard names "TPair_v1" == #[]

-- a parameter is covered by a port of its name, or of each row of an array
#guard (Registered.coverMiss P "TStep_v1" #["k"]).isNone
#guard (Registered.coverMiss P "TStep_v1" #["x", "c"]).map (·.1) == some "k"
#guard Registered.coversParam #["c[0]", "c[1]"] ("c", .array 2 (.value 32))
#guard !Registered.coversParam #["c[0]"] ("c", .array 2 (.value 32))
#guard !Registered.coversParam #[] ("c", .array 0 (.value 32))

/-- A circuit of template `t` (META `program_digests` names it alone). -/
def circ (t : String) : Circuit :=
  { (default : Circuit) with metaJ := Json.mkObj [("program_digests", Json.mkObj [(t, Json.str "00")])] }

def tq : Json := Json.mkObj [("query", Json.mkObj [("name", Json.str "Q_template_instance"),
  ("version", Lean.toJson (0 : Nat)), ("params", Json.mkObj [("template", Json.str "TStep_v1")])])]

def own : Option (Array Registered.Port) := some #[]

def why (c : Circuit) (program : ByteArray) (partition : Option Json) (params : Array String) : String :=
  match Registered.checkCover c (some program) partition own params with
  | .error e => e
  | .ok () => ""

-- covered: passes; uncovered: refused by the parameter's name
#guard why (circ "TStep_v1") reg (some tq) #["k"] == ""
#guard why (circ "TStep_v1") reg (some tq) #["x"] ==
  "registered reads (cover): parameter k of TStep_v1 reads a registered input of the verifier's program, and no port of the verifier's registered reads reads it: its rows would be committed afresh"
#guard why (circ "TWiden16_v1") reg (some tq) #["k"] ==
  "registered reads (cover): parameter a of TWiden16_v1 reads a registered input of the verifier's program, and no port of the verifier's registered reads reads it: its rows would be committed afresh"
-- a template that reads no registered input needs no port
#guard why (circ "TPair_v1") reg (some tq) #[] == ""

-- a program that reads a registered value: only under a template query, and a circuit whose META names its template
#guard (why (circ "TStep_v1") reg none #["k"]).startsWith "registered reads (cover): the verifier's program reads a registered value, and its partition is not a template query"
#guard (why default reg (some tq) #["k"]).startsWith "registered reads (cover): META program_digests does not name"
-- a program without a registered input holds nothing
#guard why default plain none #[] == ""
-- a descriptor that does not decode is refused
#guard !(Registered.checkCover default (some "{}".toUTF8) none own #[]).toBool
-- without the program or the registered reads `checkCover` holds nothing: `programHeld` and `checkProgram` refuse those
#guard (Registered.checkCover default none none own #[]).toBool
#guard (Registered.checkCover default (some reg) (some tq) none #[]).toBool

-- a port's parameter: its `param`, else its name
def file : Json := Json.mkObj [("format", Json.str Registered.FORMAT), ("ports", Json.mkObj [
  ("k_0", Json.mkObj [(Registered.PARAM_KEY, Json.str "k")]), ("a", Json.mkObj [])])]
#guard (Registered.paramsOf file).toOption == some #["a", "k"]

-- a session without the verifier's program is refused, whatever else it passes
#guard (match ProvedScope.programHeld none with | .error e => e | .ok () => "") == ProvedScope.NO_PROGRAM
#guard (ProvedScope.programHeld (some reg)).toBool

end Flock.Test.RegCover
