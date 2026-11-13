/-
`RegBound.check`, which `flock-verify` runs on every statement whenever `--registered` is given: without bound ports it is
`Registered.checkComplete`; with them, `Flock.Test.RegComplete`'s lane statement, whose index row `pub` is committed afresh,
passes when the verifier's file binds `pub` at the public header's frame-v3-sha512 root and domain, and is refused when it
binds it at another root or domain, binds a port the circuit lacks or a registered port, or leaves `pub` out. A port named
after a parameter the verifier's own program reads a registered input through (`Flock.Test.RegCover`'s `TStep_v1`'s `k`)
is never bound, and nor is a port the window's registered interface declares registered (`RegBound.declaredOf`), an
output `o1` as an opening's verdict: bound or left out, it is refused, and a file binding ports without the interface is
refused.
-/
module

public import Flock.Test.RegComplete
public import Flock.Test.RegCover
public import Flock.RegBound
meta import Flock.Test.RegComplete
meta import Flock.Test.RegCover
meta import Flock.RegBound

@[expose] public section

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

/-- The lane's circuit as template `TLane_v1`'s (META `program_digests`), whose ports the interface declares. -/
def laneT : Circuit := { lane with metaJ := (Flock.Test.RegCover.circ "TLane_v1").metaJ }

/-- The window's interface: `TLane_v1`'s `w` declared registered, and another template's `pub`. -/
def decl : Array (String × String) := #[("TLane_v1", "w"), ("TOther_v1", "pub")]

/-- `RegComplete.pub` with its header's frame-v3-sha512 root and domain of the index row `pub`. -/
def pubF : Public := { pub with header := pub.header.setObjVal! "frame_v3" (frame ["pub"]) }

def boundOf (ports : List (String × ByteArray × ByteArray)) : Json :=
  Json.mkObj (ports.map fun (p, r, d) => (p, Json.mkObj [("root", Json.str (hexOf r)), ("domain", Json.str (hexOf d))]))

/-- The verifier's file: `w` registered, and `bound` beside it. -/
def file (bound : Json) (format : String := Registered.FORMAT) : Json :=
  Json.mkObj [("format", Json.str format), ("ports", Json.mkObj [("w", wEntry)]), (RegBound.KEY, bound)]

/-- `flock-verify`'s registered reads of `f`: the file's bound ports and registered reads, `Registered.check`, then
`RegBound.check` under the verifier's program and the window's interface `declared`. -/
def run (f : Json) (program : Option ByteArray := some plain) (declared : Option (Array (String × String)) := some decl) :
    Except String Unit := do
  let bound ← RegBound.ofJson f
  let own ← Registered.ofJson (RegBound.ownOf f)
  Registered.check laneT pubF (some own) true
  RegBound.check laneT pubF program own bound declared

def msg (r : Except String Unit) : String :=
  match r with
  | .error e => e
  | .ok () => ""

def why (f : Json) (program : Option ByteArray := some plain) (declared : Option (Array (String × String)) := some decl) :
    String :=
  msg (run f program declared)

-- `pub` bound at the header's root and domain: accepted
#guard why (file (boundOf [("pub", root, dom)])) == ""
#guard (RegBound.ownOf (file (boundOf [("pub", root, dom)]))).compress == Flock.Test.RegComplete.file.compress

-- at another root, or another domain: refused
#guard why (file (boundOf [("pub", other, dom)])) ==
  "registered reads (bound): port pub's frame-v3-sha512 root or domain in the public file is not the window's committed one the verifier binds it to"
#guard why (file (boundOf [("pub", root, other)])) ==
  "registered reads (bound): port pub's frame-v3-sha512 root or domain in the public file is not the window's committed one the verifier binds it to"

-- without bound ports the check is `checkComplete`, which refuses the index row by name
#guard msg (RegBound.check lane pubF (some plain) #[{ (default : Registered.Port) with name := "w" }] #[] none) ==
  msg (Registered.checkComplete lane #[{ (default : Registered.Port) with name := "w" }])
#guard (Flock.Test.RegComplete.why lane Flock.Test.RegComplete.file).startsWith
  "registered reads (complete): the statement's port pub is no port of the verifier's registered reads"

-- a bound port the circuit lacks, or a registered port bound too: refused
#guard why (file (boundOf [("pub", root, dom), ("q", root, dom)])) ==
  "registered reads (bound): q is no port of the statement's circuit"
#guard why (file (boundOf [("pub", root, dom), ("w", root, dom)])) ==
  "registered reads (bound): port w is a registered port and bound: each port is one or the other"

-- a circuit port neither registered nor bound: refused
def lane3 : Circuit := { laneT with ports := lane.ports.push { pubPort with name := "idx" } }
#guard msg (RegBound.check lane3 pubF (some plain) #[{ (default : Registered.Port) with name := "w" }]
    #[{ name := "pub", root, domain := dom }] (some decl)) ==
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
  #[{ name := "c", root, domain := dom }, { name := "x", root, domain := dom }] (some #[])) == ""
#guard msg (RegBound.check step pubS (some reg) #[{ kPort with name := "c" }]
    #[{ name := "x", root, domain := dom }, { name := "k", root, domain := dom }] (some #[])) ==
  "registered reads (bound): port k reads parameter k, a registered input of the verifier's program: its rows open the registered root, never a window's (name it among the ports)"
#guard RegBound.namesParam "c[1]" ("c", .array 2 (.value 32))
#guard !RegBound.namesParam "c[2]" ("c", .array 2 (.value 32))
#guard !RegBound.namesParam "cc" ("c", .value 32)

-- the window's interface declares `o1`, the lane's output (an opening's verdict), registered: registered, it passes;
-- moved to `bound` or left out, it is refused, whatever the file's builder omitted
def laneO : Circuit := { laneT with ports := laneT.ports.push { pubPort with name := "o1" } }
def declO : Array (String × String) := decl.push ("TLane_v1", "o1")
def wPort : Registered.Port := { (default : Registered.Port) with name := "w" }
def pubB : RegBound.Bound := { name := "pub", root, domain := dom }
#guard msg (RegBound.check laneO pubF (some plain) #[wPort, { wPort with name := "o1" }] #[pubB] (some declO)) == ""
#guard msg (RegBound.check laneO pubF (some plain) #[wPort] #[pubB, { name := "o1", root, domain := dom }] (some declO)) ==
  "registered reads (declared): port o1 is one the window's interface declares registered: it reads at a registered root, never a window's (name it among the ports)"
#guard msg (RegBound.check laneO pubF (some plain) #[wPort] #[pubB] (some declO)) ==
  "registered reads (declared): the statement's port o1 is one the window's interface declares registered, and no port of the verifier's registered reads: it reads at a registered root, never bound or committed afresh"
-- declared for another template, `o1` is the lane's own to bind or leave: left out, the complete check refuses it
#guard msg (RegBound.check laneO pubF (some plain) #[wPort] #[pubB] (some (decl.push ("TOther_v1", "o1")))) ==
  "registered reads (complete): the statement's port o1 is neither a port of the verifier's registered reads nor bound to the window's committed values (bound): its rows would be committed afresh, stated by the public file"
-- a declared port names its rows of an array too
#guard RegBound.declaredIn #["o"] "o[1]" && RegBound.declaredIn #["o"] "o"
#guard !RegBound.declaredIn #["o"] "oo" && !RegBound.declaredIn #["o"] "o[1"

-- a file binding ports needs the interface, and a circuit naming its template
#guard (why (file (boundOf [("pub", root, dom)])) (some plain) none).startsWith
  "registered reads (declared): the file binds ports (bound), and the verifier holds no registered interface (--interface)"
#guard (msg (RegBound.check lane pubF (some plain) #[wPort] #[pubB] (some decl))).startsWith
  "registered reads (declared): META program_digests does not name the circuit's one template"

-- the interface: a window circuit record of the partition object's program; its reads', table reads' and openings'
-- ports, the openings' verdicts
def iface (program : String := "ab") (fmt : String := RegBound.INTERFACE) : Json :=
  Json.mkObj [("format", Json.str fmt), ("program", Json.str program), ("registered", Json.mkObj [
    ("reads", Json.arr #[Json.mkObj [("reader", Json.str "TLane_v1"), ("port", Json.str "w"), ("param", Json.str "weights")]]),
    ("roots", Json.arr #[Json.mkObj [("reader", Json.str "TEmb_v1"), ("port", Json.str "root")]]),
    ("openings", Json.arr #[Json.mkObj [("reader", Json.str "TEmb_v1"), ("verdict", Json.str "o1"),
      ("ports", Json.arr #[Json.str "row"])]])])]
#guard (RegBound.declaredOf (iface) (some "ab")).toOption ==
  some #[("TLane_v1", "w"), ("TEmb_v1", "root"), ("TEmb_v1", "o1")]
#guard (RegBound.declaredOf (Json.mkObj [("format", Json.str RegBound.INTERFACE), ("program", Json.str "ab")]) (some "ab")).toOption ==
  some #[]
#guard msg ((RegBound.declaredOf (iface "cd") (some "ab")).map fun _ => ()) ==
  "registered interface: the record's program is not the partition object's: it is another window's interface"
#guard msg ((RegBound.declaredOf (iface) none).map fun _ => ()) ==
  "registered interface: its program is held to the partition object's, and the verifier holds none (--partition)"
#guard msg ((RegBound.declaredOf (iface (fmt := "verity/window-circuit/v0")) (some "ab")).map fun _ => ()) ==
  "registered interface: not a verity/window-circuit/v1 record"
-- the record's bytes: JSON naming no key twice in one object
#guard (RegBound.declaredOfBytes (iface).compress.toUTF8 (some "ab")).toOption == (RegBound.declaredOf (iface) (some "ab")).toOption
#guard msg ((RegBound.declaredOfBytes "{\"format\":\"verity/window-circuit/v1\",\"program\":\"ab\",\"program\":\"cd\"}".toUTF8
    (some "ab")).map fun _ => ()) ==
  "registered interface: an object of the record names the key \"program\" twice, which a JSON parser would read as once"
#guard (msg ((RegBound.declaredOfBytes "{".toUTF8 (some "ab")).map fun _ => ())).startsWith "registered interface: not JSON"

end Flock.Test.RegBound
