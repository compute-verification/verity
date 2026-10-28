import PouwCommit.Coins
import PouwCommit.Draws
import PouwCommit.Rows
import Lean.Data.Json

/-!
# The difftest: the Python held to PoUW's commitments in Lean, its verifier of record, on generated cases

`tests/difftest_vectors.py` writes `difftest/cases.jsonl`, one case a line: `{"kind", "in", "python"}`. An outcome is
`{"ok": v}`, `{"reason": text}` (a coin refused, in `coin_reason`'s words) or `{"error": …}` (Python raised; Lean threw),
and errors agree by kind alone. `main` computes each case's outcome from the Lean and fails on every case where Python's
is not it, naming the Lean's. The kinds:

* `tagged`: `protocol.tagged` of a tag and parts (`{"b": hex}`, `{"s": text}`, `{"i": n}`);
* `bindings`: `weight_binding`, `activation_binding`, `weights_root` and `transcript_binding`;
* `domain`: `CommitmentDomain(binding, owner, RangeIndexedDomain(count), hash).domain_id`, or its refusal;
* `commit_rows`: `commit_rows(…).commitment.root`, plain and hm96-sha512, rows and tile digests;
* `transcript`: the transcript tree's root over tile digests (`Prover.prove`);
* `verify`: `verify_opening` against `rows_domain`, of openings Python's trees made and of tampered ones;
* `coins`: `coin_reason` of `CoinCommitment.from_json` and `CoinOpening.from_json`, honest and tampered;
* `salt_of`, `unverified_salt`: the epoch's salt from a coin, and `Epoch.unverified`'s from a seed;
* `epoch_salt`, `draw_coin`: `Epoch.start`'s salt and `draw_coin`'s coin, or their refusals in their words (`{"reason"}`);
* `key`: the audit, draw and replay keys and `draw_after`;
* `sampled`, `pearl_draw`: the clear audit's work-weighted draws over given tickets, `Sampled.select`'s and
  `clear_work.draw`'s (`benchmarks/pouw/clear`).
-/

namespace PouwCommit.Difftest

open Lean Flock

abbrev E := Except String

def str (j : Json) (k : String) : E String := do
  match ← field j k with
  | .str s => pure s
  | _ => throw s!"{k} is not a string"

def nat (j : Json) (k : String) : E Nat := do
  match jsonInt? (← field j k) with
  | some n => if n < 0 then throw s!"{k} is negative" else pure n.toNat
  | none => throw s!"{k} is not an integer"

def int (j : Json) (k : String) : E Int := do
  match jsonInt? (← field j k) with
  | some n => pure n
  | none => throw s!"{k} is not an integer"

def bool (j : Json) (k : String) : E Bool := do
  match ← field j k with
  | .bool b => pure b
  | _ => throw s!"{k} is not a boolean"

def hexes (j : Json) (k : String) : E (Array ByteArray) := do
  match ← field j k with
  | .arr xs => xs.mapM fun
    | .str s => unhex s
    | _ => throw s!"{k} holds a non-string"
  | _ => throw s!"{k} is not an array"

def part : Json → E Part
  | .obj kv => match kv.get? "b", kv.get? "s", kv.get? "i" with
    | some (.str h), none, none => return .bytes (← unhex h)
    | none, some (.str s), none => return .str s
    | none, none, some n => match jsonInt? n with
      | some i => if i < 0 then throw "a part's integer is negative" else return .int i.toNat
      | none => throw "a part's integer is not one"
    | _, _, _ => throw "a part is {\"b\"}, {\"s\"} or {\"i\"}"
  | _ => throw "a part is an object"

def hexJ (b : ByteArray) : Json := .str (hexOf b)

/-- The case's outcome from the port. -/
def outcome (kind : String) (i : Json) : E Json := do
  match kind with
  | "tagged" =>
    let parts ← match ← field i "parts" with
      | .arr ps => ps.toList.mapM part
      | _ => throw "parts is not an array"
    return hexJ (tagged (← str i "tag") parts)
  | "bindings" =>
    let roots ← weightRoots i
    let salt ← unhexField i "salt"
    let wr ← weightsRoot roots
    return Json.mkObj [("weight", hexJ (weightBinding (← nat i "weight"))),
      ("activation", hexJ (activationBinding salt (← nat i "index"))),
      ("weights_root", hexJ wr), ("transcript", hexJ (transcriptBinding salt wr))]
  | "domain" =>
    let h ← match ← str i "hash" with
      | "sha256" => pure Hash.sha256
      | "sha512" => pure Hash.sha512
      | h => throw s!"no frame hash {h}"
    return hexJ (← Domain.mk? h (← unhexField i "binding") (← int i "owner") (← nat i "count")).id
  | "commit_rows" =>
    let key ← match ← field i "salt_key" with
      | .null => pure none
      | .str s => pure (some (← unhex s))
      | _ => throw "salt_key is hex or null"
    return hexJ (← commitRows (← unhexField i "binding") (← int i "owner") (← hexes i "rows") (← str i "hash") key
      (← nat i "nonce") (← bool i "digests") (← bool i "public"))
  | "transcript" =>
    return hexJ (← transcriptRoot (← unhexField i "salt") (← unhexField i "weights_root") (← hexes i "digests")
      (← str i "hash") (← unhexField i "salt_key"))
  | "verify" =>
    let d ← rowsDomain (← unhexField i "binding") (← int i "owner") (← nat i "count") (← str i "hash") (← bool i "digests")
    let root ← unhexField i "root"
    checkRoot root
    let path := (← hexes i "path").toList
    checkPath path
    return .bool (← d.verify root (← nat i "position") (← str i "schema") (← unhexField i "value") path)
  | "coins" =>
    let c ← CoinCommitment.ofJson (← field i "commitment")
    let o ← CoinOpening.ofJson (← field i "opening")
    return match ← coinReason c o (← str i "label") (← unhexField i "after") with
      | none => Json.mkObj [("ok", .null)]
      | some r => Json.mkObj [("reason", .str r)]
  | "salt_of" =>
    let c ← CoinCommitment.ofJson (← field i "commitment")
    return hexJ (← saltOf c (← unhexField i "coin") (← unhexField i "weights_root"))
  | "unverified_salt" =>
    return hexJ (← unverifiedSalt (← unhexField i "seed") (← weightsRoot (← weightRoots i)))
  | "epoch_salt" =>
    let c ← CoinCommitment.ofJson (← field i "commitment")
    let o ← CoinOpening.ofJson (← field i "opening")
    return refusable (← epochSalt c o (← weightsRoot (← weightRoots i)))
  | "draw_coin" =>
    let c ← match ← field i "commitment" with
      | .null => pure none
      | j => pure (some (← CoinCommitment.ofJson j))
    let o ← CoinOpening.ofJson (← field i "opening")
    return refusable (← drawCoin c o (← unhexField i "after"))
  | "key" =>
    let (tr, rd, salt) := (← unhexField i "transcript", ← unhexField i "records", ← unhexField i "salt")
    match ← str i "which" with
    | "audit" => return hexJ (← auditKey (← unhexField i "source") tr salt)
    | "draw" => return hexJ (← drawKey (← unhexField i "source") tr rd salt)
    | "replay" => return hexJ (← replayKey (← nat i "round") (← unhexField i "source") tr rd salt)
    | "after" => return hexJ (drawAfter tr rd)
    | w => throw s!"no key {w}"
  | "sampled" =>
    let key ← auditKey (← unhexField i "source") (← unhexField i "transcript") (← unhexField i "salt")
    return draws (← weightedDraw key (← tickets i) (← nat i "t"))
  | "pearl_draw" =>
    let key ← drawKey (← unhexField i "source") (← unhexField i "transcript") (← unhexField i "records") (← unhexField i "salt")
    return draws (← pearlDraw key (← tickets i) (← nat i "t"))
  | k => throw s!"no kind {k}"
where
  unhexField (j : Json) (k : String) : E ByteArray := do unhex (← str j k)
  weightRoots (j : Json) : E (List (Nat × ByteArray)) := do
    match ← field j "weight_roots" with
    | .arr ps => ps.toList.mapM fun
      | .arr #[w, .str r] => do
        let some n := jsonInt? w | throw "a weight is an integer"
        if n < 0 then throw "a weight is nonnegative"
        return (n.toNat, ← unhex r)
      | _ => throw "a weight root is [w, root]"
    | _ => throw "weight_roots is not an array"
  tickets (j : Json) : E (List Nat) := do
    match ← field j "tickets" with
    | .arr ts => ts.toList.mapM fun t => match jsonInt? t with
      | some n => if n < 0 then throw "a ticket is negative" else pure n.toNat
      | none => throw "a ticket is not an integer"
    | _ => throw "tickets is not an array"
  refusable : Except String ByteArray → Json
    | .ok b => Json.mkObj [("ok", hexJ b)]
    | .error r => Json.mkObj [("reason", .str r)]
  draws (ids : List Nat) : Json := .arr (ids.toArray.map fun (n : Nat) => toJson n)

/-- The kinds whose outcomes are already `{"ok"}` or `{"reason"}`. -/
def REFUSABLE : List String := ["coins", "epoch_salt", "draw_coin"]

/-- An outcome as the cases write it. -/
def wrap (kind : String) : E Json → Json
  | .ok v => if REFUSABLE.contains kind then v else Json.mkObj [("ok", v)]
  | .error e => Json.mkObj [("error", .str e)]

def isError (j : Json) : Bool := (j.getObjVal? "error").toOption.isSome

/-- The case's differences: none when Python's outcome is the Lean's. -/
def differences (n : Nat) (line : String) : List String :=
  match Json.parse line with
  | .error e => [s!"case {n}: not JSON: {e}"]
  | .ok c =>
    let kind := (c.getObjValAs? String "kind").toOption.getD ""
    let python := (c.getObjVal? "python").toOption.getD .null
    let lean := wrap kind (outcome kind ((c.getObjVal? "in").toOption.getD .null))
    if (isError python && isError lean) || python == lean then []
    else [s!"case {n} ({kind}): Python gives {python.compress} where the Lean, of record, gives {lean.compress}"]

def main (args : List String) : IO UInt32 := do
  let [cases, n] := args | IO.eprintln "usage: PouwCommit/DifftestMain.lean CASES N"; return 2
  let lines := ((← IO.FS.readFile cases).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  if n.toNat? != some lines.size then
    IO.eprintln s!"difftest: {lines.size} cases, not {n}"
    return 1
  -- Interpreted, each task builds hm96's mask tables (`Hm96.Default512.tables`) once, so the cases go in four chunks.
  let cases := lines.toList.zipIdx
  let chunk := (cases.length + 3) / 4
  let tasks := (List.range 4).map fun k => Task.spawn fun _ =>
    ((cases.drop (k * chunk)).take chunk).flatMap fun (l, i) => differences i l
  let found := tasks.flatMap Task.get
  for d in found.take 50 do IO.eprintln d
  IO.println s!"difftest: {lines.size} cases compared, {found.length} differences"
  return if found.isEmpty then 0 else 1

end PouwCommit.Difftest
