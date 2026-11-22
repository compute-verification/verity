module

public import PouwVerifier.Credit
public import PouwVerifier.CreditOnce
public import PouwVerifier.Pass
public import PouwVerifier.WindowSalt
public import PouwVerifier.Lanes
public import PouwVerifier.Verified
public import PouwVerifier.CommittedFiles
meta import PouwVerifier.Credit
meta import PouwVerifier.CreditOnce
meta import PouwVerifier.Pass
meta import PouwVerifier.WindowSalt
meta import PouwVerifier.Lanes
meta import PouwVerifier.Verified
meta import PouwVerifier.CommittedFiles

/-!
# One guard per refusal code PoUW's check makes

Each `#guard` builds the smallest input that reaches one refusal code and asserts the code, through the function that
makes it. Pearl+'s own codes are guarded in `PouwPearlPlus.Refusals`. `order` is guarded through
`Credit.Once.periodWith`, which is `Credit.period` with the ledger's replay a parameter (`Credit.Once.period_eq`, by
`rfl`), at a replay that trusts the ledger. No input reaches `sampler` (`Credit.drawnOf`, `drawCheck`): it is the coin
stream running out, and `Flock.Draw` gives each draw 64 spare bits and never more draws than the stratum has members.
-/

namespace PouwVerifier.Refusals

open Lean (Json)

/-- The refusal code of `r`, none when it accepts. -/
def code? {α : Type} (r : R α) : Option String :=
  match r with
  | .error (c, _) => some c
  | .ok _ => none

/-- A well-formed challenger key (`Spot.Challenger.ChallengerKey.ofJson`): LMS W = 8, H = 1, 80 public bytes. -/
def KEY : Json :=
  Json.mkObj [("H", Lean.toJson (1 : Nat)), ("W", Lean.toJson (8 : Nat)), ("public", .str ("".pushn '0' 160))]

/-- A SHA-512 digest in lowercase hex. -/
def DIGEST : Json := .str ("".pushn 'a' 128)

/-- An execution phase's Commit record with no registered values, as `Credit.commitOf?` reads it. -/
def COMMIT : Json := .str (Flock.hexOf s!"\{\"format\":\"{Credit.COMMIT_FORMAT}\",\"registered\":[]}".toUTF8)

def creditedRoot (key salt calls commit rounds draw : Json) : Option String :=
  code? (Credit.Internal.creditedRoot key salt calls .null commit rounds draw)

#guard creditedRoot .null .null .null .null .null .null == some "coin"
#guard creditedRoot KEY .null .null .null .null .null == some "manifest"
#guard creditedRoot KEY DIGEST (Json.arr #[Json.mkObj [("index", Lean.toJson (0 : Nat))]]) .null .null .null ==
  some "credit"
#guard creditedRoot KEY DIGEST (Json.arr #[]) .null .null .null == some "commit"
#guard creditedRoot KEY DIGEST (Json.arr #[]) COMMIT .null .null == some "rounds"
#guard creditedRoot KEY DIGEST (Json.arr #[]) COMMIT (Json.arr #[]) .null == some "draw"

/-- A Program whose function is a window's (`Credit.WINDOW_FN`) with one tile of K = `k`. -/
def windowProg (k : Nat) : Flock.Program.Prog :=
  { defs := #[], index := {}, root := 0, sha512 := .empty,
    programFn := s!"{Credit.WINDOW_FN}\{TILES=[[{k},\{\"fn\":\"t\"}]],LINEARS=[],CALLS=[],M=1,PREFIX=\"h1/\"}" }

#guard code? (Credit.layoutOf (windowProg 1) "{}".toUTF8) == some "tile-k"
#guard code? (Credit.layoutOf (windowProg 32) "{}".toUTF8) == some "layout"

/-- A ledger line holding `body` as an unsigned entry of kind `kind`: `Spot.Challenger.ledgerEntry?` reads it, and
only the ledger's replay checks its signature. -/
def line (kind : String) (body : Json) : Json :=
  .str <| (Spot.PyJson.canonical (Json.mkObj [("body", body), ("kind", .str kind), ("prev", .str ("".pushn '0' 128)),
    ("schema", .str Spot.Challenger.LEDGER_SCHEMA), ("seq", Lean.toJson (0 : Nat)), ("session", .null),
    ("sig", .str "00000000")])).toOption.getD ""

def ROOT_B : Json := .str ("".pushn 'b' 128)

/-- A period `p` opened, and closed complete with two windows: roots `DIGEST` then `ROOT_B`. -/
def PERIOD_LEDGER : Json :=
  let named := [("by", Json.str "pouw"), ("lane", .str "pouw"), ("period", .str "p")]
  let window (root : Json) (s : String) := Json.mkObj [("complete", .bool true), ("root", root), ("session", .str s)]
  Json.arr #[line "period-open" (Json.mkObj (named ++ [("cap", Lean.toJson (4 : Nat)), ("circuit", .str ("".pushn 'a' 128)),
      ("key_sha512", .str ("".pushn 'c' 128)), ("pinned", .str ("".pushn 'a' 40))])),
    line "period-close" (Json.mkObj (named ++ [("complete", .bool true),
      ("windows", Json.arr #[window DIGEST "o-0", window ROOT_B "o-1"])]))]

def periodCode (roots : Array Json) : Option String :=
  code? (Credit.Once.periodWith (fun _ => pure (#[], #[])) KEY PERIOD_LEDGER (.str "p") (Json.arr roots))

#guard periodCode #[DIGEST, ROOT_B] == none
#guard periodCode #[ROOT_B, DIGEST] == some "order"

#guard code? (Credit.strata .null) == some "layout"
#guard code? (Credit.samplingDesign #[] .null) == some "budget"
#guard code? (Credit.Internal.aLink .null .null 0) == some "a-link"
#guard code? (Credit.Internal.arow #[.null, .null] .null #[] (Json.mkObj [("calls", Json.arr #[.null])]) "") ==
  some "arow"

#guard code? (Credit.proved #[] .null) == some "outcome"
#guard code? (Credit.proved #[0] (Json.arr #[])) == some "missing"
#guard code? (Credit.proved #[0] (Json.arr #[Json.mkObj [("program", DIGEST), ("registered", Json.mkObj []),
  ("unit", Lean.toJson (0 : Nat)), ("verdict", .str "reject")]])) == some "failed"

#guard (Credit.pinHygiene "p" 0 (Json.arr #[.str "a"]) "a" (some "b") none).map (·.1) == some "pin"
#guard code? (Credit.committedWeights .null "p") == some "weights"
#guard code? (Credit.committedLength .null "p") == some "length"

#guard code? (CommittedFiles.committedFiles .null "p") == some "period"
#guard code? (CommittedFiles.committedFiles (Json.arr #[]) "p") == some "circuit"
#guard code? (CommittedFiles.filesOf (Json.mkObj [])) == some "error"

#guard code? (Verified.asks .null (Json.mkObj []) "" ⟨"", []⟩ .null .null "" [] 0 "" .null) == some "proof"
#guard code? (Lanes.lanesRecord .null .null ⟨"", []⟩ "" "" "" []) == some "registration"
#guard (match Lanes.leafAsks none #[] (Json.mkObj [(s!"{Lanes.LANE}0", .null)]) "" .null ⟨"", []⟩ .null "" [0] 0 ""
    .null with
  | .ok (some (c, _)) => some c
  | _ => none) == some "lanes-shape"

#guard code? (drawCheck .null .null .null .null .null .null .null .null (.str "k")) == some "coin-key"
#guard code? (executionPhaseSaltCheck .null
  ⟨.empty, "", "", "", "", "", 0, .null, #[], ⟨"", "", .null, none, none⟩⟩ .null) == some "salt"

end PouwVerifier.Refusals
