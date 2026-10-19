import Difftest
import RemoteAudit.Session

/-!
# The difftest's Lean half

The functions `tests/` holds `audit.py`, `receipts.py` and `codec.py` to through `verity_difftest.Port`, over JSON (byte
strings lowercase hex, indices integers): a whole session's events (`session`: every event's result, the outcome and
the record), and the pieces a mismatch there would point to (the codec's decode, a plan, a receipt, a frame-v3-sha512
opening, a leaf's receipt, a window's draws and a registration's digest), and the registration `audit.register` makes
with its codec seed. `remote-audit-difftest` serves them.
-/

namespace RemoteAudit.Difftest

open Lean (Json ToJson toJson)
open Flock

private def bytesOf (j : Json) (ctx : String) : V ByteArray := do
  match unhex (← J.asStr j ctx) with
  | .ok b => pure b
  | .error e => throw s!"{ctx}: {e}"

private def hex (j : Json) (k : String) : V ByteArray := do bytesOf (← J.get j k "input") k

private def hexOpt (j : Json) (k : String) : V (Option ByteArray) := do
  match ← J.get j k "input" with
  | .null => pure none
  | v => some <$> bytesOf v k

private def hexes (j : Json) (k : String) : V (Array ByteArray) := do (← J.arr j k "input").mapM (bytesOf · k)

private def int (j : Json) (k : String) : V Int := do J.asInt (← J.get j k "input") k

private def nat (j : Json) (k : String) : V Nat := J.nat j k "input"

private def runsOf (j : Json) (ctx : String) : V (Array (Int × Int)) := do
  (← J.asArr j ctx).mapM fun r => do
    let xs ← J.asArr r ctx
    if xs.size ≠ 2 then throw s!"{ctx}: a run is [first, count]"
    pure (← J.asInt xs[0]! ctx, ← J.asInt xs[1]! ctx)

private def hexJ (b : ByteArray) : Json := .str (hexOf b)

private def pairJ (x : Nat × Nat) : Json := .arr #[toJson x.1, toJson x.2]

private def runsJ (pl : Plan) : Json := .arr (pl.runs.map pairJ)

def codecOf (c : Json) : V Codec.Codec := do
  let some codec := Codec.Codec.make? (← J.str c "field" "codec") (← J.nat c "rounds" "codec") (← J.bool c "masked" "codec")
    | throw "a codec is p1024 or p512 at 2 ≤ rounds < 2^32 − 4"
  pure codec

def registrationOf (j : Json) : V Registration := do
  let some kind := Kind.ofName? (← J.str j "receipt" "registration") | throw "a receipt kind is sketch-sha512 or block-sha512"
  Registration.make? (← hex j "node") (← nat j "blocks") (← hex j "payload_root") (← hex j "codec_seed")
    (← codecOf (← J.get j "codec" "registration")) kind (← J.bool j "bind" "registration") (← J.str j "layout" "registration")
    (← nat j "rows")

def openingOf (j : Json) : V Opening := do
  pure
    { pass := ← int j "pass"
      position := ← int j "position"
      block := ← hex j "block"
      leaf := ← hex j "leaf"
      leafSalt := ← hex j "leaf_salt"
      receiptPath := ← hexes j "receipt_path"
      payloadSalt := ← hex j "payload_salt"
      payloadPath := ← hexes j "payload_path"
      tile := ← hex j "tile"
      digest := ← hex j "digest" }

def eventOf (j : Json) : V Session.Event := do
  match ← J.str j "kind" "event" with
  | "start" => pure (.start (← nat j "now"))
  | "boundary" => pure (.boundary (← nat j "now"))
  | "append" => pure (.append (← hex j "root") (← int j "count"))
  | "release" => pure (.release (← runsOf (← J.get j "runs" "event") "runs") (← runsOf (← J.get j "withdrawn" "event") "withdrawn"))
  | "receive" =>
    let runs ← match ← J.get j "runs" "event" with
      | .null => pure none
      | v => some <$> runsOf v "runs"
    let change ← match ← J.get j "change" "event" with
      | .null => pure none
      | v => do
        let xs ← J.asArr v "change"
        if xs.size ≠ 2 then throw "a change is [added, removed]"
        pure (some (← runsOf xs[0]! "added", ← runsOf xs[1]! "removed"))
    let step ← match ← J.get j "step" "event" with
      | .null => pure none
      | v => some <$> J.asInt v "step"
    pure (.receive (← int j "t") (← int j "i") (← hex j "root") runs change step (← nat j "now"))
  | "close" => pure (.close (← int j "t") (← nat j "now"))
  | "verify" => pure (.verify (← int j "t") (← (← J.arr j "openings" "event").mapM openingOf))
  | k => throw s!"no event kind {k}"

def resultJ : Session.Result → Json
  | .refused _ => Json.mkObj [("refused", .bool true)]
  | .rejected t f at? =>
    Json.mkObj [("failure", .str f.name), ("window", toJson t), ("at", match at? with | some x => pairJ x | none => .null)]
  | .released k => Json.mkObj [("released", match k with | some k => toJson k | none => .null)]
  | .appended first coin => Json.mkObj [("first", toJson first), ("coin", hexJ coin)]
  | .ok => Json.mkObj [("ok", .bool true)]
  | .drew ds => Json.mkObj [("draws", .arr (ds.map pairJ))]
  | .accepted t => Json.mkObj [("accepted", toJson t)]

def summaryJ (registration : ByteArray) (x : Session.Summary) : Json :=
  Json.mkObj [("registration", hexJ registration), ("appended", toJson x.appended), ("released", toJson x.released),
    ("appends", match x.appends with | some h => hexJ h | none => .null),
    ("windows", .arr (x.windows.map fun w => Json.mkObj [("lengths", toJson w.lengths), ("steps", toJson w.steps),
      ("post_seed", toJson w.post), ("released_ns", toJson w.releasedNs), ("deadline_ns", toJson w.deadlineNs),
      ("opened_ns", toJson w.openedNs), ("drawn", toJson w.drawn)]))]

private def positive (p : Json) (k : String) : V Nat := do
  let n ← J.nat p k "params"
  if n = 0 then throw s!"{k} is a positive integer"
  pure n

/-- A session's events through the verifier: `{"registration", "params", "not_before_ns", "coins", "append_coins",
"events"}` to every event's result, the outcome and the record. The parameters name whether the prover defers an
appended block's residency to its encode (`deferred`), which `audit.Params` refuses at a lifetime of one window: a
block appended in window w is resident from w or w + 1, so its re-key falls at age L − 1. -/
def sessionFn (j : Json) : V Json := do
  let reg ← registrationOf (← J.get j "registration" "input")
  let p ← J.get j "params" "input"
  let params : Session.Params :=
    { deadlineNs := ← positive p "deadline_ns"
      windows := ← positive p "windows"
      samples := ← positive p "samples"
      minPasses := ← positive p "min_passes"
      windowNs := ← positive p "window_ns"
      lifetime := ← positive p "lifetime" }
  if (← J.bool p "deferred" "params") && params.lifetime < 2 then
    throw "a session that defers residency has a lifetime of at least 2 windows"
  let coins ← hexes j "coins"
  if coins.size ≠ 2 * params.windows || coins.any (·.size ≠ Codec.COIN_BYTES) then throw "2W coins of 32 bytes"
  let appendCoins ← hexes j "append_coins"
  if appendCoins.any (·.size ≠ Codec.COIN_BYTES) then throw "append coins are 32 bytes"
  let events ← (← J.arr j "events" "input").mapM eventOf
  let (results, s) ← Session.run (Session.State.init params reg coins appendCoins (← nat j "not_before_ns")) events
  pure (Json.mkObj [("results", .arr (results.map resultJ)),
    ("outcome", match s.outcome with | some o => .str o | none => .null),
    ("record", match s.summary with | some x => summaryJ s.digest x | none => .null)])

def registrationDigestFn (j : Json) : V Json := do pure (hexJ (← registrationOf j).digest)

/-- `audit.register` on a given source and nonce: the registration's codec seed and digest. -/
def registerFn (j : Json) : V Json := do
  let some kind := Kind.ofName? (← J.str j "receipt" "input") | throw "a receipt kind is sketch-sha512 or block-sha512"
  let r ← Registration.register (← hex j "source") (← hex j "node") (← nat j "blocks") (← hex j "payload_root")
    (← codecOf (← J.get j "codec" "input")) kind (← J.bool j "bind" "input") (← J.str j "layout" "input") (← nat j "rows")
    (← hex j "nonce")
  pure (Json.mkObj [("codec_seed", hexJ r.codecSeed), ("digest", hexJ r.digest)])

def codecDecodeFn (j : Json) : V Json := do
  let codec ← codecOf (← J.get j "codec" "input")
  let seed ← hex j "seed"
  if seed.size ≠ Codec.SEED_BYTES then throw "a codec seed is 32 bytes"
  let bound ← match ← hexOpt j "digest", ← hexOpt j "coin" with
    | none, some _ => throw "an append's coin keys a bound block"
    | none, none => pure none
    | some d, c => pure (some (d, c))
  match Codec.decode (← hex j "block") (← nat j "j") (Codec.key seed codec) bound with
  | .ok (some m) => pure (hexJ m)
  | .ok none => pure .null
  | .error e => throw e

def planFn (j : Json) : V Json := do
  let some pl := Plan.make? (← runsOf (← J.get j "runs" "input") "runs") (← nat j "blocks") | throw "not a plan"
  let ps ← (← J.arr j "positions" "input").mapM (J.asNat · "positions")
  if ps.any (· ≥ pl.length) then throw "a position past the plan"
  pure (Json.mkObj [("length", toJson pl.length), ("digest", hexJ pl.digest), ("blocks", toJson (ps.map pl.block))])

def planChangedFn (j : Json) : V Json := do
  let some pl := Plan.make? (← runsOf (← J.get j "runs" "input") "runs") (← nat j "blocks") | throw "not a plan"
  let some out := pl.changed (← runsOf (← J.get j "added" "input") "added") (← runsOf (← J.get j "removed" "input") "removed")
    (← nat j "extent") | throw "not a change of the plan"
  pure (runsJ out)

def receiptFn (j : Json) : V Json := do
  let some kind := Kind.ofName? (← J.str j "kind" "input") | throw "a receipt kind is sketch-sha512 or block-sha512"
  let rows ← nat j "rows"
  if !kind.rowsOk rows then throw "the rows are not the kind's"
  let s ← hex j "seed"
  if s.size ≠ 64 then throw "a pass seed is 64 bytes"
  let jj ← nat j "j"
  if jj ≥ 2 ^ 64 then throw "a block index is a u64"
  let block ← hex j "block"
  if block.size ≠ Codec.BLOCK_BYTES then throw "a block is 4096 bytes"
  pure (hexJ ((receipter kind s rows).receipt jj block))

def verifyOpeningFn (j : Json) : V Json := do
  let binding ← hex j "binding"
  let owner ← int j "owner"
  let root ← hex j "root"
  let path ← hexes j "path"
  if !(binding.size == 32 || binding.size == 64) || owner < -2 || !(root.size == 32 || root.size == 64)
      || path.any fun x => !(x.size == 32 || x.size == 64) then
    throw "not a domain, commitment and opening"
  let d := FrameV3.Domain.make binding owner (← nat j "start") (← nat j "count")
  pure (.bool (FrameV3.verifyOpening d root (← nat j "position") (← hex j "value") (← J.str j "schema" "input") path))

def leafReceiptFn (j : Json) : V Json := do
  match leafReceipt (← hex j "leaf") (← nat j "p") (← nat j "positions") with
  | some y => pure (hexJ y)
  | none => pure .null

def drawsFn (j : Json) : V Json := do
  let lengths ← (← J.arr j "lengths" "input").mapM (J.asNat · "lengths")
  if lengths.isEmpty || lengths.any (· = 0) then throw "a window's passes are each one or more receipts"
  let some ds := draws (← hex j "registration") (← nat j "window") (← hex j "coin") (← nat j "samples") lengths
    | throw "the draw ran past its reads"
  pure (.arr (ds.map pairJ))

def table : String → Option _root_.Difftest.Fn := _root_.Difftest.table [
  ("session", sessionFn), ("registration_digest", registrationDigestFn), ("register", registerFn),
  ("codec_decode", codecDecodeFn),
  ("plan", planFn), ("plan_changed", planChangedFn), ("receipt", receiptFn), ("verify_opening", verifyOpeningFn),
  ("leaf_receipt", leafReceiptFn), ("draws", drawsFn)]

end RemoteAudit.Difftest
