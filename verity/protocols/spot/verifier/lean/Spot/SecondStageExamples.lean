/-
**Round 2's examples**: `SecondStage`'s checks run, at elaboration, on small records, statements and replay blocks
(`#guard`): the statement check's codes, the reads a link, a wire or a port of its own may make, and the links, wires
and rows the verifier derives from a replay block's Definition.
-/
module

public import Spot.SecondStage
meta import Spot.SecondStage

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.SecondStage

open Lean (Json)

private def exRoot (i : Nat) : String := String.ofList (List.replicate 128 (Char.ofNat (48 + i)))

private def exRec (population : Nat) (roots : List (String × Nat × Nat)) (window : Json := Json.mkObj []) : Json :=
  Json.mkObj [("scheme", "frame-v3-sha512/hm96-sha512"), ("population", Lean.toJson population),
    ("program", "program-0"), ("partition", "partition-0"), ("window", window),
    ("roots", Json.arr (roots.toArray.map fun (p, r, d) =>
      Json.mkObj [("port", p), ("root", exRoot r), ("domain", exRoot d), ("leaves", Lean.toJson population)]))]

private def exRun : String := "run-0"

/-- The interior registration's window: the run, and the replay sample `sample` it lays out. -/
private def exWindow (sample : Json) (run : Json := .str exRun) : Json :=
  Json.mkObj [("kind", "served"), ("run", run), ("interior", Json.bool true), ("replay_sample", sample)]

private def exInterior (window : Json := exWindow (natsJson #[1, 3])) : Json :=
  exRec 6 [("x", 1, 2), ("out", 3, 4)] window

private def exSchemas : Json := Json.mkObj [("x", "hm96-sha512/row/v2"), ("out", "hm96-sha512/row/v2")]

/-- Port `p`'s binding in run `exRun`, which the served header names. -/
private def exBinding (p : String) : Json := .str ((portBinding exInterior p "hm96-sha512/row/v2" exRun).getD "")

/-- `exStmt`'s reference header, fixed before the run: its scheme, the set it was staged under, its ports' schemas and
their bindings there, and none of the run's fields. -/
private def exRef : Json :=
  Json.mkObj [("scheme", "frame-v3-sha512/hm96-sha512"), ("set", "unit-Pair_v0"), ("units", Json.mkObj []),
    ("frame_v3", Json.mkObj [("schemas", exSchemas), ("bindings", Json.mkObj [("x", "00"), ("out", "00")])])]

/-- A replay sample of blocks 1 and 3 of 4, three proof blocks of one template in each, units 2 and 4 sampled. -/
private def exOwn : Own :=
  { sample := #[1, 3], parts := parts 2 #[("Pair_v0", #[(0, 6)])], interior := exInterior,
    boundary := exRec 4 [("bx", 5, 6)], drawn := #[2, 4], refs := #[("Pair_v0", exRef)],
    refReads := #[("Pair_v0", .null)] }

private def exStmt (indices units : List Nat) : Json :=
  Json.mkObj [("template", "Pair_v0"), ("registered", .null),
    ("header", Json.mkObj [("scheme", "frame-v3-sha512/hm96-sha512"), ("set", .str (setName exRun 0)),
      ("instances", Lean.toJson indices.length), ("units", Json.mkObj [("indices", natsJson indices.toArray)]),
      ("frame_v3", Json.mkObj [("roots", Json.mkObj [("x", exRoot 1), ("out", exRoot 3)]),
        ("domain_ids", Json.mkObj [("x", exRoot 2), ("out", exRoot 4)]), ("schemas", exSchemas),
        ("bindings", Json.mkObj [("x", exBinding "x"), ("out", exBinding "out")])])]),
    ("draw", Json.mkObj [("law", "subset"), ("population", (6 : Nat)), ("k", Lean.toJson units.length),
      ("units", natsJson units.toArray)])]

/-- `own` with a verifier that checked statements `ss`' sessions, naming each one's header, registered reads
and sample (`Stmt.checked`). -/
private def exVerifying (ss : Array Json) (own : Own := exOwn) : Own :=
  { own with verified := ss.filterMap fun j => (Stmt.ofJson? j).bind Stmt.checked }

/-- `own` with the params giving each of statements `ss`' templates the registered reads its statement reads as its
reference statement's, as an honest run's are. -/
private def exReferring (ss : Array Json) (own : Own) : Own :=
  { own with refReads := ss.filterMap fun j => (Stmt.ofJson? j).map fun s => (s.template, s.registered) }

/-- Round 2's check of statements `ss`, each one's session checked by the verifier and its template's reference
statement reading registered what it reads. -/
private def exCheck (allow : Bool) (own : Own) (ss : Array Json) : Check :=
  check allow (exVerifying ss (exReferring ss own)) (some ss)

private def exHonest : Json := exStmt [0, 1, 2, 3, 4, 5] [2, 4]

-- Python's `window_binding("program-0", "partition-0", "x", "hm96-sha512/row/v2", 6, "run-0")`
#guard portBinding exInterior "x" "hm96-sha512/row/v2" exRun == some
  "009d0ab56f7953138bd8853c525c88a1ede372e270f1243e8c7b1980de7390130205daa6b324fe330e87a215a663b19a011fd6dbe3391eb47c5e7116ca32192f"
#guard portBinding exInterior "x" "hm96-sha512/row/v2" exRun != portBinding exInterior "out" "hm96-sha512/row/v2" exRun
#guard portBinding exInterior "y" "hm96-sha512/row/v2" exRun == none
#guard setName exRun 2 == "served/run-0/m2"

#guard exOwn.parts == some #[{ template := "Pair_v0", lo := 0, n := 3 }]
#guard (exCheck false exOwn #[exHonest]).codes == #["two-stage-form"]
#guard (exCheck true exOwn #[exHonest]).codes.isEmpty
#guard (exCheck true exOwn #[exStmt [0, 1, 2, 3, 4, 6] [2, 4]]).codes == #["second-population"]
#guard (exCheck true exOwn #[exStmt [0, 1, 2, 3, 4, 5] [1, 4]]).codes == #["second-sample"]
#guard (exCheck true exOwn #[exHonest, exHonest]).codes == #["second-template"]
#guard (check true exOwn (some #[])).codes == #["second-cover"]
#guard (check true exOwn none).codes == #["second-statements"]
#guard (check true { exOwn with parts := parts 4 #[("Pair_v0", #[(0, 6)])] } none).codes == #["second-cells"]

/-- The honest statement, its header naming circuit file `c` by digest. -/
private def exNaming (c : Nat) : Json :=
  exHonest.setObjVal! "header" ((getD exHonest "header").setObjVal! "circuit_sha512" (exRoot c))

/-- `exOwn`, the params giving its template the reference header that names circuit file `c` (`exNaming`). -/
private def exNamed (c : Nat) : Own := { exOwn with refs := #[("Pair_v0", getD (exNaming c) "header")] }

/-- The verdict of record a verifier that checked statement `j`'s session as `tables` tables writes, accepted or not;
naming no table count with none. -/
private def exVerdict (j : Json) (accepted : Bool := true) (tables : Option Nat := some 1) : Json :=
  Json.mkObj ([("of_record", Json.bool true), ("accepted", Json.bool accepted),
    ("public_header_sha512", Json.str ((canonSha512 (getD j "header")).toOption.getD "")),
    ("unit_draw", getD j "draw")] ++ (tables.map fun n => [("session_tables", Lean.toJson n)]).getD [])

#guard (check true exOwn (some #[exHonest])).codes == #["second-verified"]
#guard (check true { exOwn with verified := verifiedOf #[("lean", exVerdict exHonest)] } (some #[exHonest])).codes.isEmpty
#guard (check true { exOwn with verified := verifiedOf #[("lean", exVerdict exHonest false)] } (some #[exHonest])).codes ==
  #["second-verified"]
#guard (verifiedOf #[("lean", (exVerdict exHonest).setObjVal! "of_record" (.bool false))]).isEmpty
#guard (check true (exVerifying #[exNaming 7] (exNamed 7)) (some #[exNaming 7])).codes.isEmpty
#guard (check true (exVerifying #[exNaming 7] (exNamed 8)) (some #[exNaming 8])).codes == #["second-verified"]
#guard (check true (exVerifying #[exStmt [0, 1, 2, 3, 4, 5] [1, 4]]) (some #[exHonest])).codes == #["second-verified"]
#guard (check true { exOwn with verified := verifiedOf #[("lean", exVerdict exHonest (tables := some 2))] }
  (some #[exHonest])).codes == #["second-tables"]
#guard (check true { exOwn with verified := verifiedOf #[("lean", exVerdict exHonest (tables := none))] }
  (some #[exHonest])).codes == #["second-tables"]
#guard (check true { exOwn with verified := verifiedOf #[("a", exVerdict exHonest),
  ("b", exVerdict exHonest (tables := some 2))] } (some #[exHonest])).codes == #["second-tables"]
#guard (check true { exOwn with verified := verifiedOf #[("a", exVerdict exHonest),
  ("b", exVerdict (exNaming 9) (tables := some 2))] } (some #[exHonest])).codes.isEmpty

/-- `exStmt`'s honest statement, its header `f` of the honest one. -/
private def exHeader (f : Json → Json) : Json :=
  let s := exStmt [0, 1, 2, 3, 4, 5] [2, 4]
  s.setObjVal! "header" (f (getD s "header"))

/-- Header `h` with its object `k`'s field `f` set to `v`. -/
private def exIn (k f : String) (v : Json) (h : Json) : Json := h.setObjVal! k ((getD h k).setObjVal! f v)

/-- Round 2's codes, the form allowed, for `exHeader f` alone, its session checked by the verifier. -/
private def exCodes (own : Own) (f : Json → Json) : Array String := (exCheck true own #[exHeader f]).codes

#guard (exCodes exOwn id).isEmpty
#guard (exCodes { exOwn with refs := #[("Pair_v0", getD (exHeader id) "header")] } id).isEmpty
#guard (exCodes exOwn (·.setObjVal! "registered" (Json.mkObj [("ports", Json.mkObj [])]))).isEmpty
#guard exCodes exOwn (·.setObjVal! "set" (.str "served/other/m0")) == #["second-header"]
#guard exCodes exOwn (·.setObjVal! "set" (.str (setName exRun 1))) == #["second-header"]
#guard exCodes exOwn (exIn "frame_v3" "bindings" (Json.mkObj [("x", exBinding "x"), ("out", exBinding "x")])) ==
  #["second-header"]
#guard exCodes exOwn (exIn "frame_v3" "bindings" (Json.mkObj [("x", exBinding "x")])) == #["second-header"]
#guard exCodes exOwn (exIn "frame_v3" "schemas" (Json.mkObj [("x", "hm96-sha512/row/v2")])) == #["second-header"]
#guard exCodes { exOwn with interior := exInterior (exWindow (natsJson #[1, 3]) (.str "run-1")) } id ==
  #["second-header"]
#guard exCodes { exOwn with interior := exInterior (exWindow (natsJson #[1, 3]) .null) } id == #["second-header"]
#guard exCodes { exOwn with interior := exInterior (exWindow (natsJson #[3, 1])) } id == #["second-interior-order"]
#guard exCodes { exOwn with interior := exInterior (exWindow (natsJson #[1])) } id == #["second-interior-order"]
#guard exCodes { exOwn with interior := exInterior (exWindow (.arr #[.bool true, Lean.toJson (3 : Nat)])) } id ==
  #["second-interior-order"]
#guard exCodes { exOwn with interior := exInterior (exWindow .null) } id == #["second-interior-order"]
#guard (exCheck false { exOwn with interior := exInterior (exWindow (natsJson #[3, 1])) } #[exHeader id]).codes ==
  #["second-interior-order"]
#guard (check true { exOwn with interior := exInterior (exWindow (natsJson #[3, 1])) } none).codes ==
  #["second-interior-order", "second-statements"]
#guard (check true { exOwn with interior := exInterior (exWindow .null), links := none } none).codes ==
  #["second-interior-order", "second-links"]
#guard (exCodes exOwn (exIn "units" "blocks" (natsJson #[0]))).isEmpty
#guard (exCodes { exOwn with refs := #[("Pair_v0", exIn "units" "blocks" (natsJson #[9]) exRef)] } id).isEmpty
#guard exCodes exOwn (exIn "units" "units_per_vu" (Lean.toJson (2 : Nat))) == #["second-header"]
#guard exCodes exOwn (exIn "frame_v3" "owner" (.str "other")) == #["second-header"]
#guard exCodes exOwn (·.setObjVal! "scheme" (.str "frame-v3-sha512/other")) == #["second-header", "second-roots"]
#guard exCodes exOwn (·.setObjVal! "instances" (Lean.toJson (7 : Nat))) == #["second-population"]
#guard exCodes exOwn (exIn "units" "indices" (natsJson #[1, 2, 3, 4, 5, 6])) == #["second-population"]
#guard exCodes exOwn (exIn "frame_v3" "roots" (Json.mkObj [("x", exRoot 1), ("out", exRoot 5)])) == #["second-roots"]
#guard exCodes exOwn (exIn "frame_v3" "domain_ids" (Json.mkObj [("x", exRoot 2)])) == #["second-roots"]
#guard exCodes { exOwn with refs := #[] } id == #["second-header"]
#guard exCodes { exOwn with refs := #[("Other_v0", exRef)] } id == #["second-header"]
#guard (exCheck false { exOwn with refs := #[] } #[exHeader id]).codes == #["second-header"]

/-- Port `x`'s statement reading `value`'s tree of `leaves` leaves at root `root`, domain `dom`, slot `k` its `reads[k]`. -/
private def exReading (value : String) (root dom leaves : Nat) (reads : List Nat) : Json :=
  (exStmt [0, 1, 2, 3, 4, 5] [2, 4]).setObjVal! "registered" (Json.mkObj [("ports", Json.mkObj [("x", Json.mkObj [
    ("value", value), ("root", exRoot root), ("domain", exRoot dom), ("leaves", Lean.toJson leaves),
    ("reads_from", (0 : Nat)), ("reads", natsJson reads.toArray)])])])

private def exReported (value : String) (root : Nat) (own : Own := exOwn) : Own :=
  { own with reported := some #[(value, .str (exRoot root))] }

/-- `exOwn` with `Pair_v0`'s `x` linked to the boundary's `bx`, one row a block, which its three units a block each
read. -/
private def exLinked : Own :=
  { exOwn with links := some #[⟨"Pair_v0", "x", "bx"⟩], rows := #[⟨"Pair_v0", "x", 1, #[⟨0, 3, 0, 0⟩]⟩] }

/-- The boundary registration's record of population 4, its port `bx` of `leaves` leaves. -/
private def exBoundary (leaves : Nat) : Json :=
  (exRec 4 [("bx", 5, 6)]).setObjVal! "roots" (Json.arr #[Json.mkObj [("port", "bx"), ("root", exRoot 5),
    ("domain", exRoot 6), ("leaves", Lean.toJson leaves)]])

/-- `exLinked` with `bx` of `per` rows a block, its units reading rows `runs`. -/
private def exLinkedAt (per : Nat) (runs : Array Run) : Own :=
  { exLinked with boundary := exBoundary (4 * per), rows := #[⟨"Pair_v0", "x", per, runs⟩] }

/-- Round 2's codes for the statement reading `bx`, of `leaves` leaves, at rows `reads` over `exLinkedAt per runs`, its
`bx` of `held` leaves. -/
private def exAt (per : Nat) (runs : Array Run) (reads : List Nat) (held : Nat := 4 * per) (leaves : Nat := held) :
    Array String :=
  (exCheck true (exReported "bx" 5 { exLinkedAt per runs with boundary := exBoundary held })
    #[exReading "bx" 5 6 leaves reads]).codes

#guard (exCheck true (exReported "x" 1) #[exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]]).codes.isEmpty
#guard (exCheck true exOwn #[exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]]).codes == #["second-reported"]
#guard (exCheck true (exReported "x" 1) #[exReading "x" 1 2 6 [1, 1, 2, 3, 4, 5]]).codes == #["second-read"]
#guard (exCheck true (exReported "out" 3) #[exReading "out" 3 4 6 [0, 1, 2, 3, 4, 5]]).codes == #["second-read"]
#guard (exCheck true (exReported "bx" 5 exLinked) #[exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 3]]).codes.isEmpty
#guard (exCheck true (exReported "bx" 5 exLinked) #[exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 2]]).codes == #["second-read"]
#guard (exCheck true (exReported "bx" 5) #[exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 3]]).codes == #["second-read"]
#guard (exCheck false (exReported "bx" 5 exLinked) #[exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 3]]).codes ==
  #["two-stage-form"]
#guard (exCheck true (exReported "x" 1 exLinked) #[exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]]).codes == #["second-read"]
#guard (exCheck true exLinked #[exHonest]).codes == #["second-read"]
#guard (exCheck true { exOwn with links := none } #[exHonest]).codes == #["second-links"]
#guard (check true { exOwn with links := none } none).codes == #["second-links"]
#guard (check true { exOwn with links := some #[⟨"Other_v0", "x", "bx"⟩] } none).codes == #["second-links"]
-- a link's units each read the row their caller's slice names, of the block's rows: here units 0, 1 and 2 rows 0, 1
-- and 2 of three a block, in blocks 1 and 3; another unit's row, or one past the block's, refused
#guard (exAt 3 #[⟨0, 3, 0, 1⟩] [3, 4, 5, 9, 10, 11]).isEmpty
#guard exAt 3 #[⟨0, 3, 0, 1⟩] [4, 3, 5, 9, 10, 11] == #["second-read"]
#guard exAt 3 #[⟨0, 3, 0, 1⟩] [3, 3, 3, 9, 9, 9] == #["second-read"]
#guard (exAt 3 #[⟨0, 1, 2, 0⟩, ⟨1, 2, 0, 1⟩] [5, 3, 4, 11, 9, 10]).isEmpty
#guard exAt 3 #[⟨0, 3, 1, 1⟩] [4, 5, 6, 10, 11, 12] == #["second-read"]
-- the boundary port holds its population's rows a block, and the read is at its leaves
#guard exAt 3 #[⟨0, 3, 0, 1⟩] [3, 4, 5, 9, 10, 11] (held := 4) == #["second-read"]
#guard exAt 3 #[⟨0, 3, 0, 1⟩] [3, 4, 5, 9, 10, 11] (leaves := 16) == #["second-read"]
-- rows that don't place the part's units, or no rows, make no links
#guard (check true (exLinkedAt 3 #[⟨0, 2, 0, 1⟩]) none).codes == #["second-links"]
#guard (check true (exLinkedAt 3 #[⟨0, 3, 0, 1⟩, ⟨3, 1, 0, 0⟩]) none).codes == #["second-links"]
#guard (check true (exLinkedAt 3 #[⟨1, 3, 0, 1⟩]) none).codes == #["second-links"]
#guard (check true { exLinked with rows := #[] } none).codes == #["second-links"]
#guard (check true { exLinked with rows := #[⟨"Pair_v0", "out", 1, #[⟨0, 3, 0, 0⟩]⟩] } none).codes == #["second-links"]
#guard (check true (exReported "x" 1 (exVerifying #[exReading "x" 1 2 6 [0, 1, 2, 3, 4, 4]]
  (exReferring #[exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]] exOwn)))
  (some #[exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]])).codes == #["second-verified"]

/-- A read of `leaves` leaves, slot `k` reading row `reads[k]`. -/
private def exRows (leaves : Nat) (reads : List Nat) : Json :=
  Json.mkObj [("leaves", Lean.toJson leaves), ("reads_from", (0 : Nat)), ("reads", natsJson reads.toArray)]

-- a slot reads its unit's first row, so in a tree of two rows a unit it can't pick its unit's other
#guard rowsOk 2 ⟨"Pair_v0", 0, 3⟩ (exRows 6 [0, 1, 2, 3, 4, 5]) 6 some
#guard rowsOk 2 ⟨"Pair_v0", 0, 3⟩ (exRows 12 [0, 2, 4, 6, 8, 10]) 6 some
#guard !rowsOk 2 ⟨"Pair_v0", 0, 3⟩ (exRows 12 [1, 3, 5, 7, 9, 11]) 6 some
#guard !rowsOk 2 ⟨"Pair_v0", 0, 3⟩ (exRows 12 [0, 2, 4, 6, 8, 11]) 6 some

/-- `exOwn` with the boundary registering value `w`, a tree of `leaves` leaves. -/
private def exRegisteringW (leaves : Nat) : Own :=
  { exOwn with boundary := (exRec 4 [("bx", 5, 6)]).setObjVal! "registered" (Json.arr #[Json.mkObj [("value", "w"),
    ("root", exRoot 7), ("domain", exRoot 8), ("leaves", Lean.toJson leaves)]]) }

/-- A read of registered value `w` at its root and domain, its tree of `leaves` leaves. -/
private def exReadW (leaves : Nat) : Json :=
  Json.mkObj [("value", "w"), ("root", exRoot 7), ("domain", exRoot 8), ("leaves", Lean.toJson leaves)]

-- a read's positions are 64-bit, so its tree has at most `2^64` leaves
#guard readOk (exRegisteringW (2 ^ 64)) false ⟨"Pair_v0", 0, 3⟩ "y" (exReadW (2 ^ 64))
#guard !readOk (exRegisteringW (2 ^ 64 + 1)) false ⟨"Pair_v0", 0, 3⟩ "y" (exReadW (2 ^ 64 + 1))
#guard !readOk (exRegisteringW 6) false ⟨"Pair_v0", 0, 3⟩ "y" (exReadW 7)

/-- `exOwn`, the interior registration registering values `w1` and `w2`, six leaves each at its own root and domain. -/
private def exRegistering : Own :=
  { exOwn with interior := exInterior.setObjVal! "registered" (Json.arr #[
      Json.mkObj [("value", "w1"), ("root", exRoot 7), ("domain", exRoot 8), ("leaves", Lean.toJson (6 : Nat))],
      Json.mkObj [("value", "w2"), ("root", exRoot 9), ("domain", exRoot 5), ("leaves", Lean.toJson (6 : Nat))]]) }

/-- Round 2's codes for statement `j` alone, its session checked by the verifier, the params giving its
template's reference statement the registered reads of statement `r`. -/
private def exValues (own : Own) (r j : Json) : Array String :=
  (check true (exVerifying #[j] (exReferring #[r] own)) (some #[j])).codes

/-- Statement `j`, its port `x`'s registered read's field `k` set to `v`. -/
private def exReadField (k : String) (v : Json) (j : Json) : Json :=
  let r := getD j "registered"
  j.setObjVal! "registered" (r.setObjVal! "ports" (Json.mkObj [("x", (getD (getD r "ports") "x").setObjVal! k v)]))

-- a registered value's rows are the reference's; a linked or own port's the run places
#guard jeq (blankReads exRegistering "Pair_v0" (getD (exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]) "registered"))
  (Json.mkObj [("ports", Json.mkObj [("x", Json.mkObj [("value", "w2"), ("root", .null), ("domain", .null),
    ("leaves", .null), ("reads_from", (0 : Nat)), ("reads", natsJson #[0, 1, 2, 3, 4, 5])])])])
#guard jeq (blankReads exOwn "Pair_v0" (getD (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]) "registered")) (Json.mkObj
  [("ports", Json.mkObj [("x", Json.mkObj [("value", "x"), ("root", .null), ("domain", .null), ("leaves", .null),
    ("reads_from", .null), ("reads", .null)])])])
#guard jeq (blankReads exLinked "Pair_v0" (getD (exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 3]) "registered")) (Json.mkObj
  [("ports", Json.mkObj [("x", Json.mkObj [("value", "bx"), ("root", .null), ("domain", .null), ("leaves", .null),
    ("reads_from", .null), ("reads", .null)])])])
#guard jeq (blankReads exOwn "Pair_v0" .null) .null
-- a port no link names reads a value either registration registers (`second-read`): the one its reference reads, at
-- the run's root, domain and leaves and the reference's rows
#guard (exCheck true (exReported "w2" 9 exRegistering) #[exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]]).codes.isEmpty
#guard (exValues (exReported "w2" 9 exRegistering) (exReading "w2" 1 2 3 [0, 1, 2, 3, 4, 5])
  (exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5])).isEmpty
-- the rows at which it reads a registered value, which `second-read` leaves to the run: refused at any but the
-- reference's, while a linked or own port's rows are the run's to place
#guard exValues (exReported "w2" 9 exRegistering) (exReading "w2" 9 5 6 [5])
  (exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]) == #["second-values"]
#guard exValues (exReported "w2" 9 exRegistering) (exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 4])
  (exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]) == #["second-values"]
#guard exValues (exReported "w2" 9 exRegistering) (exReadField "reads_from" (3 : Nat)
  (exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5])) (exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]) == #["second-values"]
#guard (exValues (exReported "x" 1) (exReading "x" 1 2 6 [5, 4, 3, 2, 1, 0])
  (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5])).isEmpty
#guard (exValues (exReported "bx" 5 exLinked) (exReading "bx" 5 6 4 [0, 0, 0, 0, 0, 0])
  (exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 3])).isEmpty
-- another value a registration registers, which `second-read` alone takes, so the run's choice: refused
#guard exValues (exReported "w2" 9 exRegistering) (exReading "w1" 7 8 6 [0, 1, 2, 3, 4, 5])
  (exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]) == #["second-values"]
#guard exValues (exReported "w1" 7 exRegistering) (exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5])
  (exReading "w1" 7 8 6 [0, 1, 2, 3, 4, 5]) == #["second-values"]
-- a value only the interior registers (a history table, say), read only at its root, domain and leaves
-- (`second-read`); and a value neither registers, refused
#guard (exCheck true (exReported "w2" 1 exRegistering) #[exReading "w2" 1 5 6 [0, 1, 2, 3, 4, 5]]).codes ==
  #["second-read"]
#guard (exCheck true (exReported "w2" 9 exRegistering) #[exReading "w2" 9 1 6 [0, 1, 2, 3, 4, 5]]).codes ==
  #["second-read"]
#guard (exCheck true (exReported "w2" 9 exRegistering) #[exReading "w2" 9 5 3 [0, 1, 2, 3, 4, 5]]).codes ==
  #["second-read"]
#guard (exCheck true (exReported "w3" 9 exRegistering) #[exReading "w3" 9 5 6 [0, 1, 2, 3, 4, 5]]).codes ==
  #["second-read"]

/-- `exRegistering` with the boundary registering the interior's values too. -/
private def exBothRegistering : Own :=
  { exRegistering with boundary := exOwn.boundary.setObjVal! "registered" (getD exRegistering.interior "registered") }

/-- `exOwn`, the interior registration registering a value of the boundary's port `bx`'s name beside `w2`. -/
private def exRegisteringBx : Own :=
  { exOwn with interior := exInterior.setObjVal! "registered" (Json.arr #[
      Json.mkObj [("value", "bx"), ("root", exRoot 5), ("domain", exRoot 6), ("leaves", Lean.toJson (4 : Nat))],
      Json.mkObj [("value", "w2"), ("root", exRoot 9), ("domain", exRoot 5), ("leaves", Lean.toJson (6 : Nat))]]) }

-- the reads that rest on the interior's own values, which the attached record names (`interiorReads`, `attach`): a
-- value only the interior registers; not an own port's, a linked port's, or a value the boundary registers too
#guard jeq (.arr (interiorReads exRegistering [exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]]))
  (.arr #[Json.mkObj [("template", "Pair_v0"), ("port", "x"), ("value", "w2")]])
#guard (interiorReads exRegistering [exHonest]).isEmpty
#guard (interiorReads exOwn [exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]]).isEmpty
#guard (interiorReads exLinked [exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 3]]).isEmpty
#guard (interiorReads exBothRegistering [exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]]).isEmpty
-- a name is one registration's (`namesOk`): the interior registering a value the boundary registers too, or a value of
-- one of the boundary's ports' names, is refused, whatever the statements read (`second-interior-names`)
#guard (exCheck true (exReported "w2" 9 exBothRegistering) #[exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]]).codes ==
  #["second-interior-names"]
#guard namesOk exRegistering && namesOk (exRegisteringW 6) && !namesOk exBothRegistering && !namesOk exRegisteringBx
#guard (exCheck true (exReported "w2" 9 exRegisteringBx) #[exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]]).codes ==
  #["second-interior-names"]
-- what a replay block returns is read only at round 1's port (`returnsOk`): the interior registering a value of a name
-- the block's return has in the boundary record is refused, whatever the statements read and with the form off too
-- (`second-return`); a value of another name is not, and neither is one where the return has no such name
#guard returnsOk exRegistering && returnsOk { exRegistering with returns := #["o0"] } &&
  !returnsOk { exRegistering with returns := #["o0", "w2"] } && returnsOk { exOwn with returns := #["w2"] }
#guard (exCheck true (exReported "w2" 9 { exRegistering with returns := #["o0"] })
  #[exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]]).codes.isEmpty
#guard (exCheck true (exReported "w2" 9 { exRegistering with returns := #["w2"] })
  #[exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]]).codes == #["second-return"]
#guard (exCheck false (exReported "w2" 9 { exRegistering with returns := #["w2"] })
  #[exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]]).codes == #["second-return"]
#guard (exCheck true (exReported "w2" 9 { exRegisteringBx with returns := #["bx"] })
  #[exReading "w2" 9 5 6 [0, 1, 2, 3, 4, 5]]).codes == #["second-interior-names", "second-return"]
#guard (check true { exRegistering with returns := #["w1"] } none).codes == #["second-return", "second-statements"]
#guard jeq (getD (attach (Json.mkObj [("checks", Json.mkObj [])]) {} #[Json.mkObj [("value", "w2")]]) "interior_reads")
  (.arr #[Json.mkObj [("value", "w2")]])
#guard jeq (getD (attach (Json.mkObj [("checks", Json.mkObj [])]) {}) "interior_reads") .null
#guard (exValues (exReported "x" 1) (exReadField "reads_from" (3 : Nat) (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]))
  (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5])).isEmpty
-- reads where the reference reads none, none where it reads one, a field of a port's other than the run's, or no
-- reference reads for the template
#guard exValues (exReported "x" 1) exHonest (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]) == #["second-values"]
#guard exValues exOwn (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]) exHonest == #["second-values"]
#guard exValues (exReported "x" 1) (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5])
  (exReadField "row_schema" "hm96-sha512/row/v1" (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5])) == #["second-values"]
#guard (check true (exVerifying #[exHonest] { exOwn with refReads := #[] }) (some #[exHonest])).codes ==
  #["second-values"]
#guard (check true (exVerifying #[exHonest] { exOwn with refReads := #[("Other_v0", .null)] })
  (some #[exHonest])).codes == #["second-values"]

/-- Port `p`'s read of the interior record's own port `p` at root `root`, domain `dom`, slot `k` reading row `reads[k]`. -/
private def exOwnRead (p : String) (root dom : Nat) (reads : List Nat := [0, 1, 2, 3, 4, 5]) : String × Json :=
  (p, Json.mkObj [("value", .str p), ("root", exRoot root), ("domain", exRoot dom), ("leaves", Lean.toJson (6 : Nat)),
    ("reads_from", (0 : Nat)), ("reads", natsJson reads.toArray)])

/-- Tables of six rows at `ports`, as a header shares them (`shared_rows`). -/
private def exTables (ports : List String) : Json := Json.mkObj (ports.map fun p => (p, Lean.toJson (6 : Nat)))

/-- The honest statement, its header sharing rows at `shared`, reading `reads` registered (`none`: no registered
reads). -/
private def exSharing (shared : List String) (reads : Option (List (String × Json))) : Json :=
  (exHeader (·.setObjVal! "shared_rows" (exTables shared))).setObjVal! "registered"
    ((reads.map fun rs => Json.mkObj [("ports", Json.mkObj rs)]).getD .null)

/-- `exOwn` with its template's reference sharing rows at `shared`, and the verifiers of record reporting reads of
`read`, each value at its root. -/
private def exShares (shared : List String) (read : List (String × Nat)) : Own :=
  { exOwn with
    refs := #[("Pair_v0", exRef.setObjVal! "shared_rows" (exTables shared))],
    reported := some (read.toArray.map fun (v, r) => (v, .str (exRoot r))) }

-- a header sharing rows at `out` and `x` reads both registered, each slot its own row
#guard (exCheck true (exShares ["out", "x"] [("out", 3), ("x", 1)])
  #[exSharing ["out", "x"] (some [exOwnRead "out" 3 4, exOwnRead "x" 1 2])]).codes.isEmpty
#guard (exCheck true (exShares ["x"] [("x", 1)]) #[exSharing ["x"] (some [exOwnRead "x" 1 2])]).codes.isEmpty
-- `out` shared and not read registered: its refs are bounded only, so one could name another unit's row
#guard (exCheck true (exShares ["out", "x"] [("x", 1)]) #[exSharing ["out", "x"] (some [exOwnRead "x" 1 2])]).codes ==
  #["second-refs"]
#guard (exCheck true (exShares ["out", "x"] []) #[exSharing ["out", "x"] none]).codes == #["second-refs"]
-- a shared port read at another slot's row, which the session's verifier would hold slot 0's ref to
#guard (exCheck true (exShares ["out", "x"] [("out", 3), ("x", 1)])
  #[exSharing ["out", "x"] (some [exOwnRead "out" 3 4, exOwnRead "x" 1 2 [1, 0, 2, 3, 4, 5]])]).codes ==
  #["second-read"]
-- rows shared where the template's reference shares none
#guard (exCheck true (exReported "x" 1) #[exSharing ["out", "x"] (some [exOwnRead "x" 1 2])]).codes ==
  #["second-header", "second-refs"]
-- a session reading registered a port its template's reference statement doesn't, or not one it does
#guard exValues (exReported "x" 1) (exSharing [] (some [exOwnRead "out" 3 4]))
  (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]) == #["second-values"]
#guard exValues (exReported "x" 1) (exSharing [] (some [exOwnRead "out" 3 4, exOwnRead "x" 1 2]))
  (exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]) == #["second-values"]

/-- A statement for template `t`'s part. -/
private def exFor (t : String) : Json := (exStmt [] []).setObjVal! "template" (.str t)

private def exTwoParts : Array Part := #[⟨"Pair_v0", 0, 3⟩, ⟨"Triple_v0", 6, 1⟩]

#guard [posOf exTwoParts "Pair_v0", posOf exTwoParts "Triple_v0", posOf exTwoParts "Other_v0"] == [0, 1, 2]
#guard (orderFailures exTwoParts [exFor "Pair_v0", exFor "Triple_v0"]).isEmpty
#guard (orderFailures exTwoParts [exFor "Triple_v0", exFor "Pair_v0"]).map (·.1) == ["second-order", "second-order"]
#guard (orderFailures exTwoParts [exFor "Triple_v0"]).map (·.1) == ["second-order"]
#guard (orderFailures exTwoParts [exFor "Pair_v0", exFor "Pair_v0", exFor "Other_v0", .null]).isEmpty
#guard (check true { exOwn with parts := some exTwoParts } (some #[exFor "Triple_v0", exFor "Pair_v0"])).codes.contains
  "second-order"
#guard (exCheck true { exOwn with parts := some exTwoParts, drawn := #[2] } #[exStmt [0, 1, 2, 3, 4, 5] [2]]).codes ==
  #["second-cover"]

private def exArg (p : Bool) (i : Nat) : Flock.Program.Arg :=
  { ty := .value 8, seq := Flock.Program.Seq.ofParts #[.affine p i 0 1] }

private def exNode (form : Flock.Program.Form) (fn : Nat) (args : Array Flock.Program.Arg) : Flock.Program.Node :=
  { form, fn, args, out := .value 8 }

private def exDefn (key : String) (prim : Bool) (params : Array String) (nodes : Array Flock.Program.Node)
    (ret : Array Flock.Program.Part) (rty : Flock.Program.Ty := .value 8) : Flock.Program.Defn :=
  { key, id := key, prim, params := params.map (·, .value 8), ret := rty, gates := 0, paramLeaves := 0, outLeaves := 0,
    body := { params := params.map (·, .value 8), retType := rty, nodes, ret := .ofParts ret, offsets := #[] } }

/-- A template of one parameter `x` (0), calling a primitive (2); a replay block of parameters `x` and `w` (1), of body
`block` returning `ret`. -/
private def exDefs (block : Array Flock.Program.Node) (ret : Array Flock.Program.Part)
    (rty : Flock.Program.Ty := .value 8) : Array Flock.Program.Defn :=
  #[exDefn "Pair_v0" false #["x"] #[exNode .prim 2 #[exArg true 0]] #[.affine false 0 0 1],
    exDefn "Block_v0" false #["x", "w"] block ret rty, exDefn "Xor_v1" true #["a"] #[] #[]]

#guard linksOf (exDefs #[exNode .batch 0 #[exArg true 0]] #[.affine false 0 0 1]) 1 ==
  some #[⟨"Pair_v0", "x", "x"⟩, ⟨"Pair_v0", "o0", "o0"⟩]
/-- An argument reading `parts`. -/
private def exParts (parts : Array Flock.Program.Part) : Flock.Program.Arg :=
  { ty := .value 8, seq := .ofParts parts }

-- a caller reads one aligned row of one parameter alone, never explicit leaves; a block returning none of its leaves
#guard linksOf (exDefs #[exNode .call 0 #[exArg true 1]] #[] (.tuple #[])) 1 == some #[⟨"Pair_v0", "x", "w"⟩]
#guard linksOf (exDefs #[exNode .call 0 #[exParts #[.explicit #[⟨true, 1, 0⟩]]]] #[] (.tuple #[])) 1 == none
#guard linksOf (exDefs #[exNode .call 0 #[exArg true 1]] #[]) 1 == none
#guard linksOf (exDefs #[exNode .call 0 #[exArg false 0]] #[.affine false 0 0 1]) 1 ==
  some #[⟨"Pair_v0", "o0", "o0"⟩]
-- several callers of a template read a linked port alike, of one value
#guard linksOf (exDefs #[exNode .call 0 #[exArg true 0], exNode .call 0 #[exArg true 1]] #[] (.tuple #[])) 1 == none
#guard linkRuns (exDefs #[exNode .call 0 #[exArg true 0], exNode .call 0 #[exArg true 0]] #[] (.tuple #[])) 1 ==
  some #[(⟨"Pair_v0", "x", "x"⟩, 1, #[⟨0, 2, 0, 0⟩])]
#guard linksOf (exDefs #[exNode .call 0 #[exArg true 0], exNode .call 0 #[exArg false 0]] #[] (.tuple #[])) 1 == none
#guard linksOf (exDefs #[exNode .call 0 #[exParts #[.affine true 0 0 1, .affine false 1 0 1]]] #[]) 1 == none
#guard linksOf (exDefs #[exNode .prim 2 #[exArg true 0]] #[]) 1 == none
#guard linksOf (exDefs #[exNode .batch 0 #[exArg true 0]] #[.affine true 1 0 1]) 1 == none
#guard linksOf (exDefs #[exNode .batch 0 #[exArg true 0]] #[.affine false 0 0 1] (.tuple #[.value 8, .value 8])) 1 ==
  none
#guard linksOf (exDefs #[exNode .batch 0 #[exArg true 0]] #[.affine false 0 0 1]) 2 == none

/-- `exDefs` with a second template (3), the replay block calling `Pair_v0` on `x` and it on `w`, returning `ret` of
type `rty`. -/
private def exTwo (ret : Array Flock.Program.Part) (rty : Flock.Program.Ty) : Array Flock.Program.Defn :=
  (exDefs #[exNode .call 0 #[exArg true 0], exNode .call 3 #[exArg true 1]] ret rty).push
    (exDefn "Triple_v0" false #["x"] #[exNode .prim 2 #[exArg true 0]] #[.affine false 0 0 1])

#guard linksOf (exTwo #[.affine false 0 0 1, .affine false 1 0 1] (.tuple #[.value 8, .value 8])) 1 ==
  some #[⟨"Pair_v0", "x", "x"⟩, ⟨"Triple_v0", "x", "w"⟩, ⟨"Pair_v0", "o0", "o0"⟩, ⟨"Triple_v0", "o0", "o1"⟩]
#guard linksOf (exTwo #[.affine false 1 0 1, .affine false 0 0 1] (.record #[("a", .value 8), ("b", .value 8)])) 1 ==
  some #[⟨"Pair_v0", "x", "x"⟩, ⟨"Triple_v0", "x", "w"⟩, ⟨"Triple_v0", "o0", "o0"⟩, ⟨"Pair_v0", "o0", "o1"⟩]
#guard linksOf (exTwo #[.affine false 0 0 1, .affine false 0 0 1] (.tuple #[.value 8, .value 8])) 1 == none
#guard linksOf (exTwo #[.affine false 0 0 1, .affine true 1 0 1] (.tuple #[.value 8, .value 8])) 1 == none
#guard linksOf (exTwo #[.affine false 0 0 1, .affine false 1 0 1] (.tuple #[.value 8, .array 2 (.value 8)])) 1 == none
#guard linksOf (exTwo #[.affine false 0 0 1] (.tuple #[.value 8, .value 8])) 1 == none

/-- `exDefs` with the replay block's parameter `x` an array of `n` leaves, its body `block` returning `ret` of type
`rty`. -/
private def exDefsN (n : Nat) (block : Array Flock.Program.Node) (ret : Array Flock.Program.Part)
    (rty : Flock.Program.Ty) : Array Flock.Program.Defn :=
  (exDefs block ret rty).modify 1 fun d =>
    let ps : Array (String × Flock.Program.Ty) := #[("x", .array n (.value 8)), ("w", .value 8)]
    { d with params := ps, body := { d.body with params := ps } }

/-- `Pair_v0` called on leaf `i` of the replay block's `x`. -/
private def exPairAt (i : Nat) : Flock.Program.Node := exNode .call 0 #[exParts #[.affine true 0 i 1]]

/-- `exDefsN 2` calling `Pair_v0` on `x`'s two leaves in order and returning nodes `ks`' outputs as its two leaves. -/
private def exReturning (ks : List Nat) : Array Flock.Program.Defn :=
  exDefsN 2 #[exPairAt 0, exPairAt 1] (ks.toArray.map fun k => .affine false k 0 1) (.array 2 (.value 8))

-- each caller's units read the row of the parameter its slice is, and a template returned whole links its `o0` to
-- its field's rows, in its units' order; a caller returned twice, or out of order, makes none
#guard linkRuns (exReturning [0, 1]) 1 ==
  some #[(⟨"Pair_v0", "x", "x"⟩, 2, #[⟨0, 2, 0, 1⟩]), (⟨"Pair_v0", "o0", "o0"⟩, 2, #[⟨0, 2, 0, 1⟩])]
#guard linkRuns (exDefsN 3 #[exPairAt 2, exPairAt 0, exPairAt 1] #[] (.tuple #[])) 1 ==
  some #[(⟨"Pair_v0", "x", "x"⟩, 3, #[⟨0, 1, 2, 0⟩, ⟨1, 2, 0, 1⟩])]
#guard linksOf (exReturning [1, 0]) 1 == none
#guard linksOf (exReturning [0, 0]) 1 == none

/-- A template `Two_v0` of one parameter `x`, returning two fields. -/
private def exTwoDefn : Flock.Program.Defn :=
  exDefn "Two_v0" false #["x"] #[exNode .prim 2 #[exArg true 0], exNode .prim 2 #[exArg true 0]]
    #[.affine false 0 0 1, .affine false 1 0 1] (.tuple #[.value 8, .value 8])

/-- `Two_v0` (3) called on leaf `i` of the replay block's `x`. -/
private def exTwoAt (i : Nat) : Flock.Program.Node :=
  { exNode .call 3 #[exParts #[.affine true 0 i 1]] with out := .tuple #[.value 8, .value 8] }

/-- `exDefsN 2` with `Two_v0` (3), the replay block's body `block` returning `ret` of type `rty`. -/
private def exHeads (ret : Array Flock.Program.Part) (rty : Flock.Program.Ty := .array 2 (.value 8))
    (block : Array Flock.Program.Node := #[exTwoAt 0, exTwoAt 1]) : Array Flock.Program.Defn :=
  (exDefsN 2 block ret rty).push exTwoDefn

-- a caller of one instance returns one whole field `f` of its output, linking the template's `o<f>`, each field its
-- own link; its whole output of two fields, the same field unlike its callers, a batch's field, or a caller's output
-- not its template's, none
#guard linkRuns (exHeads #[.affine false 0 0 1, .affine false 1 0 1]) 1 ==
  some #[(⟨"Two_v0", "x", "x"⟩, 2, #[⟨0, 2, 0, 1⟩]), (⟨"Two_v0", "o0", "o0"⟩, 2, #[⟨0, 2, 0, 1⟩])]
#guard linksOf (exHeads #[.affine false 0 1 1, .affine false 1 1 1]) 1 ==
  some #[⟨"Two_v0", "x", "x"⟩, ⟨"Two_v0", "o1", "o0"⟩]
#guard linkRuns (exHeads #[.affine false 0 0 1, .affine false 1 0 1, .affine false 0 1 1, .affine false 1 1 1]
    (.tuple #[.array 2 (.value 8), .array 2 (.value 8)])) 1 ==
  some #[(⟨"Two_v0", "x", "x"⟩, 2, #[⟨0, 2, 0, 1⟩]), (⟨"Two_v0", "o0", "o0"⟩, 2, #[⟨0, 2, 0, 1⟩]),
    (⟨"Two_v0", "o1", "o1"⟩, 2, #[⟨0, 2, 0, 1⟩])]
#guard linksOf (exHeads #[.affine false 0 0 2, .affine false 1 0 2] (.array 4 (.value 8))) 1 == none
#guard linksOf (exHeads #[.affine false 0 0 1, .affine false 1 1 1]) 1 == none
#guard linksOf (exHeads #[.affine false 0 0 1] (.value 8) #[{ exNode .batch 3
    #[{ exParts #[.affine true 0 0 2] with dims := some (2, 1) }] with n := 2, out := .array 2 exTwoDefn.ret }]) 1 ==
  none
#guard linksOf (exHeads #[.affine false 0 0 1] (.value 8) #[exNode .call 3 #[exParts #[.affine true 0 0 1]]]) 1 == none
-- a batch's members read one row each, an outer axis mapped, or the one row a shared argument is
#guard linkRuns (exDefsN 2 #[{ exNode .batch 0 #[{ exParts #[.affine true 0 0 2] with dims := some (2, 1) }] with
    n := 2 }] #[] (.tuple #[])) 1 == some #[(⟨"Pair_v0", "x", "x"⟩, 2, #[⟨0, 2, 0, 1⟩])]
#guard linkRuns (exDefsN 2 #[{ exNode .batch 0 #[exParts #[.affine true 0 1 1]] with n := 2 }] #[] (.tuple #[])) 1 ==
  some #[(⟨"Pair_v0", "x", "x"⟩, 2, #[⟨0, 2, 1, 0⟩])]
#guard linksOf (exDefsN 2 #[{ exNode .batch 0 #[{ exParts #[.affine true 0 0 1] with dims := some (2, 1) }] with
    n := 2 }] #[] (.tuple #[])) 1 == none
#guard rowAt #[⟨0, 1, 2, 0⟩, ⟨1, 2, 0, 1⟩] 0 == some 2 && rowAt #[⟨0, 1, 2, 0⟩, ⟨1, 2, 0, 1⟩] 2 == some 1 &&
  rowAt #[⟨0, 1, 2, 0⟩, ⟨1, 2, 0, 1⟩] 3 == none
#guard fits #[⟨0, 1, 2, 0⟩, ⟨1, 2, 0, 1⟩] 3 && !fits #[⟨0, 1, 2, 0⟩, ⟨1, 2, 0, 1⟩] 2 && !fits #[⟨1, 2, 0, 1⟩] 2 &&
  !fits #[⟨0, 0, 0, 0⟩, ⟨0, 2, 0, 1⟩] 2

private def exProg : Flock.Program.Prog :=
  { defs := exDefs #[exNode .batch 0 #[exArg true 0]] #[.affine false 0 0 1], root := 1, programFn := "Block_v0",
    index := Std.HashMap.ofList [("Pair_v0", 0), ("Block_v0", 1), ("Xor_v1", 2)], sha512 := ByteArray.empty }

#guard replayLinks exProg "Block_v0" (exRec 4 [("Block_v0/x", 5, 6)]) ==
  some #[⟨"Pair_v0", "x", "Block_v0/x"⟩, ⟨"Pair_v0", "o0", "Block_v0/o0"⟩]
#guard replayLinks exProg "Block_v0" (exRec 4 [("x", 5, 6)]) ==
  some #[⟨"Pair_v0", "x", "x"⟩, ⟨"Pair_v0", "o0", "o0"⟩]
#guard replayLinks exProg "Other_v0" (exRec 4 [("x", 5, 6)]) == none

/-- `exDefs` with a second template (3), the replay block calling `Pair_v0` on `x` and it on `arg`, which reads
`Pair_v0`'s node, returning its output. -/
private def exChain (arg : Flock.Program.Arg) : Array Flock.Program.Defn :=
  (exDefs #[exNode .call 0 #[exArg true 0], exNode .call 3 #[arg]] #[.affine false 1 0 1]).push
    (exDefn "Triple_v0" false #["x"] #[exNode .prim 2 #[exArg true 0]] #[.affine false 0 0 1])

-- a template's parameter that is another's whole output is a wire from it; a link reads only a parameter
#guard wiresOf (exChain (exArg false 0)) 1 == some #[⟨"Triple_v0", "x", "Pair_v0", "o0"⟩]
#guard linksOf (exChain (exArg false 0)) 1 == some #[⟨"Pair_v0", "x", "x"⟩, ⟨"Triple_v0", "o0", "o0"⟩]
#guard wiresOf (exChain (exArg true 1)) 1 == some #[]
#guard wiresOf (exDefs #[exNode .batch 0 #[exArg true 0]] #[.affine false 0 0 1]) 1 == some #[]
-- part of a row, a row twice, explicit rows, a node with a parameter, a node no template's, or no replay block's
#guard wiresOf (exChain (exParts #[.affine false 0 1 1])) 1 == none
#guard wiresOf (exChain (exParts #[.affine false 0 0 1, .affine false 0 0 1])) 1 == none
#guard wiresOf (exChain (exParts #[.explicit #[⟨false, 0, 0⟩]])) 1 == none
#guard wiresOf (exChain (exParts #[.affine false 0 0 1, .affine true 1 0 1])) 1 == none
#guard wiresOf (exDefs #[exNode .prim 2 #[exArg true 0], exNode .call 0 #[exArg false 0]] #[]) 1 == none
#guard wiresOf (exChain (exArg false 0)) 2 == none
private def exChainProg : Flock.Program.Prog :=
  { exProg with
    defs := exChain (exArg false 0)
    index := Std.HashMap.ofList [("Pair_v0", 0), ("Block_v0", 1), ("Xor_v1", 2), ("Triple_v0", 3)] }

#guard replayWires exChainProg "Block_v0" == some #[⟨"Triple_v0", "x", "Pair_v0", "o0"⟩]
#guard replayWires exProg "Block_v0" == some #[]
#guard replayWires exProg "Other_v0" == none
#guard replayRows exChainProg "Block_v0" == #[⟨"Pair_v0", "x", 1, #[⟨0, 1, 0, 0⟩]⟩,
  ⟨"Triple_v0", "o0", 1, #[⟨0, 1, 0, 0⟩]⟩, ⟨"Triple_v0", "x", 1, #[⟨0, 1, 0, 0⟩]⟩]
#guard replayRows exProg "Other_v0" == #[]

/-- `exDefs` with `Triple_v0` (3) and a template `Two_v0` (4) returning two fields, the replay block calling `Two_v0` on
`x` and `Triple_v0` on `arg`, returning `Triple_v0`'s output. -/
private def exFields (arg : Flock.Program.Arg) : Array Flock.Program.Defn :=
  ((exDefs #[{ exNode .call 4 #[exArg true 0] with out := .tuple #[.value 8, .value 8] }, exNode .call 3 #[arg]]
      #[.affine false 1 0 1]).push
    (exDefn "Triple_v0" false #["x"] #[exNode .prim 2 #[exArg true 0]] #[.affine false 0 0 1])).push
    (exDefn "Two_v0" false #["x"] #[exNode .prim 2 #[exArg true 0], exNode .prim 2 #[exArg true 0]]
      #[.affine false 0 0 1, .affine false 1 0 1] (.tuple #[.value 8, .value 8]))

/-- `exDefs` with `Triple_v0` (3), the replay block calling `Pair_v0` on `x` twice and `Triple_v0` on `arg`, returning
`Triple_v0`'s output. -/
private def exTwice (arg : Flock.Program.Node) : Array Flock.Program.Defn :=
  (exDefs #[exNode .call 0 #[exArg true 0], exNode .call 0 #[exArg true 0], arg] #[]
      (.tuple #[])).push
    (exDefn "Triple_v0" false #["x"] #[exNode .prim 2 #[exArg true 0]] #[.affine false 0 0 1])

-- a parameter that is one field of a writer's output wires that field, `o<f>`; one of several callers' outputs wires
-- its unit's row, at every member of a batch reading it alike
#guard wiresOf (exFields (exParts #[.affine false 0 1 1])) 1 == some #[⟨"Triple_v0", "x", "Two_v0", "o1"⟩]
#guard wiresOf (exFields (exParts #[.affine false 0 0 1])) 1 == some #[⟨"Triple_v0", "x", "Two_v0", "o0"⟩]
#guard wireRuns (exTwice (exNode .call 3 #[exArg false 1])) 1 ==
  some #[(⟨"Triple_v0", "x", "Pair_v0", "o0"⟩, 2, #[⟨0, 1, 1, 0⟩])]
#guard wireRuns (exTwice { exNode .batch 3 #[exArg false 0] with n := 2 }) 1 ==
  some #[(⟨"Triple_v0", "x", "Pair_v0", "o0"⟩, 2, #[⟨0, 2, 0, 0⟩])]
/-- `exTwice` with `Pair_v0`'s first caller a batch of two members. -/
private def exBatchWriter (arg : Flock.Program.Node) : Array Flock.Program.Defn :=
  (exTwice arg).modify 1 fun d => { d with body := { d.body with nodes := d.body.nodes.modify 0 fun x =>
    { x with form := .batch, n := 2, out := .array 2 (.value 8) } } }

-- a writer's units count its batches' members
#guard wireRuns (exBatchWriter (exNode .call 3 #[exArg false 1])) 1 ==
  some #[(⟨"Triple_v0", "x", "Pair_v0", "o0"⟩, 3, #[⟨0, 1, 2, 0⟩])]
-- both fields, a batch's member, a batched argument, or a gather of two writers' outputs, none
#guard wiresOf (exFields (exParts #[.affine false 0 0 2])) 1 == none
#guard wiresOf (exBatchWriter (exNode .call 3 #[exArg false 0])) 1 == none
#guard wiresOf (exTwice { exNode .batch 3 #[{ exArg false 0 with dims := some (1, 1) }] with n := 1 }) 1 == none
#guard wiresOf (exTwice (exNode .call 3 #[exParts #[.affine false 0 0 1, .affine false 1 0 1]])) 1 == none

private def exWire : Wire := ⟨"Triple_v0", "x", "Pair_v0", "out"⟩

#guard wiresOf? (Json.arr #[exWire.toJson]) == some #[exWire]
#guard wiresOf? (Json.arr #[exWire.toJson.setObjVal! "value" (.str "x")]) == none
#guard wiresOf? (Json.arr #[exWire.toJson.setObjVal! "wport" (Lean.toJson (0 : Nat))]) == none
#guard wiresOf? .null == none

/-- An interior record naming its ports by template, six leaves each: `Pair_v0`'s `x` and `out` and `Triple_v0`'s
`out`, and no tree of `Triple_v0`'s wired `x`. -/
private def exWiredInterior : Json :=
  exRec 6 [("Pair_v0/x", 1, 2), ("Pair_v0/out", 3, 4), ("Triple_v0/out", 5, 6)] (exWindow (natsJson #[1, 3]))

/-- `exTwoParts` over `exWiredInterior`, `Triple_v0`'s `x` wired from `Pair_v0`'s `out`, its unit reading its writer's
first unit of three a block. -/
private def exWired : Own :=
  { exOwn with
    parts := some exTwoParts, interior := exWiredInterior, wires := some #[exWire]
    rows := #[⟨"Triple_v0", "x", 3, #[⟨0, 1, 0, 0⟩]⟩] }

/-- `exWired`, `Triple_v0`'s unit reading its writer's unit `j`, of `per` a block. -/
private def exWiredAt (per j : Nat) : Own := { exWired with rows := #[⟨"Triple_v0", "x", per, #[⟨0, 1, j, 0⟩]⟩] }

private def exTriple : Part := ⟨"Triple_v0", 6, 1⟩

/-- A read of value `v` at root `root` and domain `dom`, slot `6 + k` reading row `reads[k]`. -/
private def exWiredRead (v : String) (root dom : Nat) (reads : List Nat) : Json :=
  Json.mkObj [("value", .str v), ("root", exRoot root), ("domain", exRoot dom), ("leaves", Lean.toJson (6 : Nat)),
    ("reads_from", (6 : Nat)), ("reads", natsJson reads.toArray)]

-- a wired port reads its writer's output value, each slot the row of its writer's unit its rows name, in its replay
-- block
#guard readOk exWired true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [0, 3])
#guard readOk exWired true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [0])
#guard readOk (exWiredAt 3 2) true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [2, 5])
#guard !readOk (exWiredAt 3 2) true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [0, 3])
#guard !readOk (exWiredAt 3 3) true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [3, 6])
#guard !readOk (exWiredAt 2 1) true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [1, 3])
#guard !readOk { exWired with rows := #[] } true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [0, 3])
-- its own port, which has no tree, another of its writer's rows, another port, or at another root
#guard !readOk exWired true exTriple "x" (exWiredRead "Triple_v0/x" 3 4 [0, 3])
#guard !readOk exWired true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [1, 3])
#guard !readOk exWired true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [0, 4])
#guard !readOk exWired true exTriple "x" (exWiredRead "Pair_v0/x" 1 2 [0, 3])
#guard !readOk exWired true exTriple "x" (exWiredRead "Pair_v0/out" 5 4 [0, 3])
-- no wire: the port reads only its own
#guard !readOk { exWired with wires := some #[] } true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [0, 3])
#guard !readOk { exWired with parts := some #[exTriple] } true exTriple "x" (exWiredRead "Pair_v0/out" 3 4 [0, 3])
-- a wired port not read
#guard wiredRead exWired "Triple_v0" #[("x", .null)]
#guard !wiredRead exWired "Triple_v0" #[("out", .null)]
#guard wiredRead exWired "Pair_v0" #[]

/-- The binding round 2 gives `Triple_v0`'s `x`, of row schema v2, placed in the run (`atRun`). -/
private def exWiredBinding (own : Own) : Option String := do
  let r ← atRun own true 1 "Triple_v0"
    (Json.mkObj [("frame_v3", Json.mkObj [("schemas", Json.mkObj [("x", "hm96-sha512/row/v2")])])])
  let .str b := getD (getD (getD r "frame_v3") "bindings") "x" | none
  pure b

-- a wired port is bound by its writer's port, which holds it
#guard heldPort exWired true "Triple_v0" "x" == "Pair_v0/out"
#guard heldPort exWired true "Triple_v0" "out" == "Triple_v0/out"
#guard (exWiredBinding exWired).isSome
#guard exWiredBinding exWired == portBinding exWiredInterior "Pair_v0/out" "hm96-sha512/row/v2" exRun
#guard exWiredBinding { exWired with wires := some #[] } == none

#guard (check true exWired none).codes == #["second-statements"]
#guard (check true { exWired with wires := none } none).codes == #["second-wires"]
#guard (check true { exWired with wires := some #[{ exWire with template := "Other_v0" }] } none).codes ==
  #["second-wires"]
#guard (check true { exWired with wires := some #[{ exWire with writer := "Other_v0" }] } none).codes ==
  #["second-wires"]
#guard (check true { exWired with wires := none, links := none } none).codes == #["second-links", "second-wires"]
#guard (check true { exWired with wires := none } (some #[])).codes == #["second-wires"]
-- rows that don't place the reader's units, or its writer's, make no wires
#guard (check true { exWired with rows := #[] } none).codes == #["second-wires"]
#guard (check true (exWiredAt 2 1) none).codes == #["second-wires"]
#guard (check true { exWired with rows := #[⟨"Triple_v0", "x", 3, #[⟨0, 2, 0, 0⟩]⟩] } none).codes == #["second-wires"]
#guard (check true (exWiredAt 3 2) none).codes == #["second-statements"]
-- a wire whose writer's port a link names reads an output the replay block returns, and one whose writer's part is not
-- before its reader's, or is its reader's own, is out of order: neither makes wires
#guard wiresOk exTwoParts exWired
#guard !wiresOk exTwoParts { exWired with links := some #[⟨"Pair_v0", "out", "o0"⟩] }
#guard (check true { exWired with
    links := some #[⟨"Pair_v0", "out", "o0"⟩]
    rows := exWired.rows.push ⟨"Pair_v0", "out", 3, #[⟨0, 3, 0, 1⟩]⟩ } none).codes == #["second-wires"]
#guard !wiresOk exTwoParts.reverse exWired
#guard (check true { exWired with parts := some exTwoParts.reverse } none).codes == #["second-wires"]
#guard !wiresOk exTwoParts { exWiredAt 1 0 with wires := some #[{ exWire with writer := "Triple_v0" }] }

end Spot.SecondStage
