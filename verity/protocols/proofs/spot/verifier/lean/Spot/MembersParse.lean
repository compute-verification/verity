/-
`members PARAMS EVIDENCE`, a command of the one-stage v2 executable (`ZkMain`): what the per-member form's parsers
read of a case's parameters and evidence, and nothing of the audit.

It runs the form's two parsers (`membersParamsOfJson`, `membersEvidenceOf`) and the statements' public structure
(`membersHeld`). It reads no registration, no blob and no coin, and it never runs the form's check (`checkMembersAt`),
whose acceptance stays refused (`MULTI_REFUSED`): `protocol` reads `paramsOfJson` and `evidenceOf`, which refuse the
form's keys, and this command is outside `protocol`. Its line says `parsed, not checked` (`PARSED`) and holds no
outcome of the audit, so nothing reads it as one: it holds a writer of the form's keys to the parsers.
-/
module

public import Spot.OneStageZk
meta import Spot.OneStageZk

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.OneStageZk

open Lean (Json)
open Flock (V)

/-- What `members` says of what it prints: the two files were parsed, and no check of the audit ran. -/
def PARSED : String := "parsed, not checked"

/-- A member as `members` prints it: its name, its circuit and public files' digests, whether the parameters name a
registered file for it, its units' ranges and its circuit's units a proof block. -/
def Member.parsed (m : Member) : Json :=
  Json.mkObj [("circuit", .str m.own.circuit), ("name", .str m.name), ("public", .str m.own.pub),
    ("registered", .bool m.own.registered.isSome),
    ("units", Json.arr (m.units.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi])),
    ("vus_per_block", Lean.toJson m.g)]

/-- A statement as `members` prints it: its member's name and its sessions' names, in order. -/
def StatementFiles.parsed (st : StatementFiles) : Json :=
  Json.mkObj [("member", match st.member with | some name => .str name | none => .null),
    ("sessions", Json.arr (st.sessions.map fun s => .str s.name))]

/-- **What the per-member form's parsers read** of the parameters `params` and the evidence `evidence`, each canonical
JSON: the members (`membersParamsOfJson`), the statements (`membersEvidenceOf`), and the statements as public
structure (`membersHeld`: each names a member no earlier one named, in the members' order, with exactly one session).
One object, exactly `{"members", "parsed", "statements"}`, its `parsed` the words `PARSED`; a refusal says which file
refused, as the audit's does. -/
def membersParsed (params evidence : ByteArray) : V Json := do
  let p ← (Core.canonicalJson params >>= membersParamsOfJson).mapError ("params: " ++ ·)
  let e ← (membersEvidenceOf evidence).mapError ("evidence: " ++ ·)
  let (some ms, some sts) := (p.members, e.statements)
    | throw "proofs: the parameters and the evidence are not the per-member form's"
  let _ ← (membersHeld ms sts).mapError ("proofs: " ++ ·)
  pure (Json.mkObj [("members", Json.arr (ms.map Member.parsed)), ("parsed", .str PARSED),
    ("statements", Json.arr (sts.map StatementFiles.parsed))])

/-- `main`'s commands, and `members PARAMS EVIDENCE`: one line, `membersParsed`'s object or `{"refused": why}`, exit 0
or 1. The line names no verdict. -/
def membersMain (args : List String) : IO UInt32 := do
  match args with
  | ["members", params, evidence] =>
    let r := membersParsed (← IO.FS.readBinFile params) (← IO.FS.readBinFile evidence)
    let line := match r with
      | .ok j => j
      | .error why => Json.mkObj [("refused", Json.str why)]
    IO.println ((Flock.canon line).toOption.getD line.compress)
    return if r.toBool then 0 else 1
  | _ => main args

private def own0 : Own :=
  { programJ := .null, program := .empty, partition := .null, circuit := "c", pub := "p", registered := none }

private def own1 : Own := { own0 with registered := some .null }

#guard (Member.parsed { name := "t/c0", units := #[(2, 5)], g := 4, own := own0 }).compress ==
  "{\"circuit\":\"c\",\"name\":\"t/c0\",\"public\":\"p\",\"registered\":false,\"units\":[[2,5]],\"vus_per_block\":4}"
#guard (Member.parsed { name := "t", units := #[(0, 1), (3, 4)], g := 1, own := own1 }).compress ==
  "{\"circuit\":\"c\",\"name\":\"t\",\"public\":\"p\",\"registered\":true,\"units\":[[0,1],[3,4]],\"vus_per_block\":1}"
#guard (StatementFiles.parsed
    { member := some "t/c0", sessions := #[{ name := "s", record := "r", reps := #[] }] }).compress ==
  "{\"member\":\"t/c0\",\"sessions\":[\"s\"]}"

private def parsedWhy (r : V Json) : String :=
  match r with
  | .error why => why
  | .ok _ => ""

#guard parsedWhy (membersParsed "{}".toUTF8 "{}".toUTF8) ==
  "params: the parameters name no proofs (the members and their statements' files)"
#guard parsedWhy (membersParsed "{\"proofs\":{\"circuit\":\"\",\"public\":\"\",\"registered\":null}}".toUTF8
    "{}".toUTF8) ==
  "params: the keys of proofs are not members"

end Spot.OneStageZk
