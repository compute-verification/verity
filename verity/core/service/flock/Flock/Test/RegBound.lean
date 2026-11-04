/-
`RegBound.check`, which `flock-verify` runs on every statement whenever `--registered` is given: without bound ports it is
`Registered.checkComplete`; with them, `Flock.Test.RegComplete`'s lane statement, whose index row `pub` is committed afresh,
passes when the verifier's file binds `pub` at the public header's frame-v3-sha512 root and domain, and is refused when it
binds it at another root or domain, binds a port the circuit lacks or a registered port, or leaves `pub` out. A port named
after a parameter the verifier's own program reads a registered input through (`Flock.Test.RegCover`'s `TStep_v1`'s `k`)
is never bound.
-/
import Flock.Test.RegComplete
import Flock.Test.RegCover
import Flock.RegBound

namespace Flock.Test.RegBound

open Lean (Json)
open Flock.Test.RegComplete (lane pub pubPort wEntry)
open Flock.Test.RegProgram (reg plain)

def root : ByteArray := ByteArray.mk (Array.replicate 64 7)
def dom : ByteArray := ByteArray.mk (Array.replicate 64 9)
def other : ByteArray := ByteArray.mk (Array.replicate 64 8)

/-- A header's `frame_v3` naming each of `ports` at `root` and `dom`. -/
def frame (ports : List String) : Json :=
  Json.mkObj [("roots", Json.mkObj (ports.map (·, Json.str (hexOf root)))),
    ("domain_ids", Json.mkObj (ports.map (·, Json.str (hexOf dom))))]

/-- `RegComplete.pub` with its header's frame-v3-sha512 root and domain of the index row `pub`. -/
def pubF : Public := { pub with header := pub.header.setObjVal! "frame_v3" (frame ["pub"]) }

def boundOf (ports : List (String × ByteArray × ByteArray)) : Json :=
  Json.mkObj (ports.map fun (p, r, d) => (p, Json.mkObj [("root", Json.str (hexOf r)), ("domain", Json.str (hexOf d))]))

/-- The verifier's file: `w` registered, and `bound` beside it. -/
def file (bound : Json) (format : String := Registered.FORMAT) : Json :=
  Json.mkObj [("format", Json.str format), ("ports", Json.mkObj [("w", wEntry)]), (RegBound.KEY, bound)]

/-- `flock-verify`'s registered reads of `f`: the file's bound ports and registered reads, `Registered.check`, then
`RegBound.check` under the verifier's program. -/
def run (f : Json) (program : Option ByteArray := some plain) : Except String Unit := do
  let bound ← RegBound.ofJson f
  let own ← Registered.ofJson (RegBound.ownOf f)
  Registered.check lane pubF (some own) true
  RegBound.check lane pubF program own bound

def msg (r : Except String Unit) : String :=
  match r with
  | .error e => e
  | .ok () => ""

def why (f : Json) (program : Option ByteArray := some plain) : String := msg (run f program)

-- `pub` bound at the header's root and domain: accepted
#guard why (file (boundOf [("pub", root, dom)])) == ""
#guard (RegBound.ownOf (file (boundOf [("pub", root, dom)]))).compress == Flock.Test.RegComplete.file.compress

-- at another root, or another domain: refused
#guard why (file (boundOf [("pub", other, dom)])) ==
  "registered reads (bound): port pub's frame-v3-sha512 root or domain in the public file is not the window's committed one the verifier binds it to"
#guard why (file (boundOf [("pub", root, other)])) ==
  "registered reads (bound): port pub's frame-v3-sha512 root or domain in the public file is not the window's committed one the verifier binds it to"

-- without bound ports the check is `checkComplete`, which refuses the index row by name
#guard msg (RegBound.check lane pubF (some plain) #[{ (default : Registered.Port) with name := "w" }] #[]) ==
  msg (Registered.checkComplete lane #[{ (default : Registered.Port) with name := "w" }])
#guard (Flock.Test.RegComplete.why lane Flock.Test.RegComplete.file).startsWith
  "registered reads (complete): the statement's port pub is no port of the verifier's registered reads"

-- a bound port the circuit lacks, or a registered port bound too: refused
#guard why (file (boundOf [("pub", root, dom), ("q", root, dom)])) ==
  "registered reads (bound): q is no port of the statement's circuit"
#guard why (file (boundOf [("pub", root, dom), ("w", root, dom)])) ==
  "registered reads (bound): port w is a registered port and bound: each port is one or the other"

-- a circuit port neither registered nor bound: refused
def lane3 : Circuit := { lane with ports := lane.ports.push { pubPort with name := "idx" } }
#guard msg (RegBound.check lane3 pubF (some plain) #[{ (default : Registered.Port) with name := "w" }]
    #[{ name := "pub", root, domain := dom }]) ==
  "registered reads (complete): the statement's port idx is neither a port of the verifier's registered reads nor bound to the window's committed values (bound): its rows would be committed afresh, stated by the public file"

-- the file's form: a v1 key, an object of ports each of a 64-byte root and domain alone
#guard (why (file (boundOf [("pub", root, dom)]) Registered.FORMAT_V0)).startsWith
  "registered reads: bound is a verity/registered-reads/v1 key"
#guard why (file (Json.mkObj [])) == "registered reads: bound names no port"
#guard why (file (Json.arr #[])) == "registered reads: bound is not an object"
#guard (why (file (Json.mkObj [("pub", Json.mkObj [("root", Json.str (hexOf root)), ("domain", Json.str (hexOf dom)),
  ("leaves", Lean.toJson (1 : Nat))])]))).startsWith "registered reads: bound port pub: it holds the keys"
#guard why (file (boundOf [("pub", root.extract 0 32, dom)])) == "registered reads: bound port pub: a 64-byte root and domain"

-- bound ports need the verifier's program
#guard (why (file (boundOf [("pub", root, dom)])) none).startsWith
  "registered reads (bound): the file binds ports (bound), and the verifier holds no program"

-- a port named after a parameter the program reads a registered input through is never bound, the others may be
def step : Circuit := { Flock.Test.RegCover.circ "TStep_v1" with
  ports := #[{ pubPort with name := "c" }, { pubPort with name := "x" }, { pubPort with name := "k" }] }
def pubS : Public := { pub with header := pub.header.setObjVal! "frame_v3" (frame ["c", "x"]) }
def kPort : Registered.Port := { (default : Registered.Port) with name := "k" }
#guard msg (RegBound.check step pubS (some reg) #[kPort]
  #[{ name := "c", root, domain := dom }, { name := "x", root, domain := dom }]) == ""
#guard msg (RegBound.check step pubS (some reg) #[{ kPort with name := "c" }]
    #[{ name := "x", root, domain := dom }, { name := "k", root, domain := dom }]) ==
  "registered reads (bound): port k reads parameter k, a registered input of the verifier's program: its rows open the registered root, never a window's (name it among the ports)"
#guard RegBound.namesParam "c[1]" ("c", .array 2 (.value 32))
#guard !RegBound.namesParam "c[2]" ("c", .array 2 (.value 32))
#guard !RegBound.namesParam "cc" ("c", .value 32)

end Flock.Test.RegBound
