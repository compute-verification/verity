import Difftest
import RemoteAudit.Line
import RemoteAudit.P2
import RemoteAudit.P2Session
import RemoteAudit.Registry
import RemoteAudit.Schedule
import RemoteAudit.Store
import RemoteAudit.Wire

/-!
# The difftest's Lean half

The functions `tests/` holds `audit.py`, `receipts.py` and `codec.py` to through `verity_difftest.Port`, over JSON (byte
strings lowercase hex, indices integers): a whole session's events (`session`: every event's result, the outcome and
the record), and the pieces a mismatch there would point to (the codec's decode, a plan, a receipt, a frame-v3-sha512
opening, a leaf's receipt, a window's draws and a registration's digest), the registration `audit.register` makes
with its codec seed, the verifier's registry (`registry`: registrations, sessions over them and their events,
closes, and `registry/v0` both ways), and `store.py`'s binding (`RemoteAudit.Store`: a pin from its file's bytes,
canonical JSON, `checkpoint_sha512`, a payload salt, the expected root, `Binding.hold`, and `binding`, the binding as
the verifier's process starts it and holds each hello to it, its `bind` and `hold` requests), `line.py`'s memory
line (`line` of groups and windows given, `line_record` read off an accepted session's record, `RemoteAudit.Line`:
rationals as strings `n` or `n/d`) and `pooled.py`'s pooled bound, which the line
reads (`pooled`, `RemoteAudit.Pooled`), whose refusals are outputs (`{"refused": why}`),
so their reasons are compared, as are those of `schedule.py`'s schedule check (`schedule`, `RemoteAudit.Schedule`).
`p2.py`'s P2 store (`RemoteAudit.P2`): a registration's digest (`p2_registration_digest`), the node whose store a
registration log registers (`p2_stored`), a unit's partials, sketch
and receipt (`p2_sketch`) and an opening's check on PoUS's vk (`p2_opening`); and a whole P2 session as
`p2_party.py` runs it (`p2_session`, `RemoteAudit.P2Session`), on the registration file's bytes and the prover's lines.
`remote-audit-difftest` serves them.
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

/-- An event as JSON gives it. A `verify`'s openings are either decoded (`openings`) or the window's frames as the wire
carried them (`frames`, lowercase hex, read as the verifier's process reads them: `Wire.openingsOf`). -/
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
  | "verify" =>
    if J.has j "frames" then pure (.verify (← int j "t") (Wire.openingsOf (← hexes j "frames")))
    else pure (.verify (← int j "t") (← (← J.arr j "openings" "event").mapM openingOf))
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

/-- The record `Session.record` makes; its pace names `in_flight` only at one window in flight (`Flight`). -/
def summaryJ (registration : ByteArray) (x : Session.Summary) (inFlight : Nat := 2) : Json :=
  Json.mkObj [("registration", hexJ registration), ("blocks", toJson x.blocks), ("appended", toJson x.appended),
    ("released", toJson x.released), ("lifetime", toJson x.lifetime),
    ("pace", match x.pace with
      | some p => Json.mkObj ([("k", toJson p.k), ("window_ns", toJson p.windowNs)] ++
          if inFlight == 1 then [("in_flight", toJson inFlight)] else [])
      | none => .null),
    ("appends", match x.appends with | some h => hexJ h | none => .null),
    ("windows", .arr (x.windows.map fun w => Json.mkObj [("lengths", toJson w.lengths), ("steps", toJson w.steps),
      ("post_seed", toJson w.post), ("heads", .arr (w.heads.map fun h => match h with | some h => hexJ h | none => .null)),
      ("stale", toJson w.stale), ("arrived_ns", toJson w.arrivedNs), ("released_ns", toJson w.releasedNs),
      ("deadline_ns", toJson w.deadlineNs), ("opened_ns", toJson w.openedNs), ("drawn", toJson w.drawn)]))]

private def positive (p : Json) (k : String) : V Nat := do
  let n ← J.nat p k "params"
  if n = 0 then throw s!"{k} is a positive integer"
  pure n

/-- A session's `groups_per_coin` (`Shared`): 1 when its parameters don't name it. -/
def perCoinOf (j : Json) : V Nat := do
  let p ← J.get j "params" "input"
  if J.has p "groups_per_coin" then positive p "groups_per_coin" else pure 1

/-- A paced session's `in_flight` (`Flight`), 1 or 2: 2 when its pace doesn't name it or it is unpaced. -/
def flightOf (j : Json) : V Nat := do
  match ← J.get (← J.get j "params" "input") "pace" "params" with
  | .null => pure 2
  | q =>
    if !J.has q "in_flight" then pure 2
    else do
      let f ← J.nat q "in_flight" "pace"
      if f == 1 || f == 2 then pure f else throw "a pace's in_flight is 1 or 2"

/-- A session's `{"params", "coins", "append_coins", "not_before_ns"}`. The parameters name whether the prover defers
an appended block's residency to its encode (`deferred`), which `audit.Params` refuses at a lifetime of one window: a
block appended in window w is resident from w or w + 1, so its re-key falls at age L − 1; its pace (`null`, or
`{"k", "window_ns"}`, with `"in_flight"` if it names it, `flightOf`), which `audit.Params` takes only with at least
two windows; and, if they name it, the windows a draw's openings coin covers (`groups_per_coin`, `perCoinOf`,
`Shared.run`). The coins are W challenges and ⌈W/g⌉ openings coins. -/
def sessionOf (j : Json) : V (Session.Params × Array ByteArray × Array ByteArray × Nat) := do
  let p ← J.get j "params" "input"
  let pace ← match ← J.get p "pace" "params" with
    | .null => pure none
    | q => do pure (some ({ k := ← positive q "k", windowNs := ← positive q "window_ns" } : Session.Pace))
  let _ ← flightOf j
  let params : Session.Params :=
    { deadlineNs := ← positive p "deadline_ns"
      windows := ← positive p "windows"
      samples := ← positive p "samples"
      minPasses := ← positive p "min_passes"
      windowNs := ← positive p "window_ns"
      lifetime := ← positive p "lifetime"
      pace }
  if (← J.bool p "deferred" "params") && params.lifetime < 2 then
    throw "a session that defers residency has a lifetime of at least 2 windows"
  if pace.isSome && params.windows < 2 then
    throw "a paced session has at least one window and the closing window"
  let g ← perCoinOf j
  let coins ← hexes j "coins"
  if coins.size ≠ params.windows + (params.windows + g - 1) / g || coins.any (·.size ≠ Codec.COIN_BYTES) then
    throw "W + ⌈W/g⌉ coins of 32 bytes"
  let appendCoins ← hexes j "append_coins"
  if appendCoins.any (·.size ≠ Codec.COIN_BYTES) then throw "append coins are 32 bytes"
  pure (params, coins, appendCoins, ← nat j "not_before_ns")

/-- The record of a session at `inFlight` windows in flight and `perCoin` windows a coin: `summaryJ`'s, naming
`groups_per_coin` when it isn't 1. -/
def recordJ (registration : ByteArray) (x : Session.Summary) (inFlight perCoin : Nat) : Json :=
  if perCoin == 1 then summaryJ registration x inFlight
  else (summaryJ registration x inFlight).setObjVal! "groups_per_coin" (toJson perCoin)

/-- A session's events through the verifier: `{"registration", "params", "not_before_ns", "coins", "append_coins",
"events"}` to every event's result, the outcome and the record (`recordJ`), which names `groups_per_coin` when it isn't
1 and its pace `in_flight` at 1 (`Flight.run f g`: `Shared.run g` at f = 2, which is `Session.run` at g = 1). -/
def sessionFn (j : Json) : V Json := do
  let reg ← registrationOf (← J.get j "registration" "input")
  let (params, coins, appendCoins, notBeforeNs) ← sessionOf j
  let g ← perCoinOf j
  let f ← flightOf j
  let events ← (← J.arr j "events" "input").mapM eventOf
  let (results, s) ← Flight.run f g (Session.State.init params reg coins appendCoins notBeforeNs) events
  let record := match s.summary with
    | some x => recordJ s.digest x f g
    | none => .null
  pure (Json.mkObj [("results", .arr (results.map resultJ)),
    ("outcome", match s.outcome with | some o => .str o | none => .null), ("record", record)])

def registrationDigestFn (j : Json) : V Json := do pure (hexJ (← registrationOf j).digest)

/-- `audit.register` on a given source and nonce: the registration's codec seed and digest. -/
def registerFn (j : Json) : V Json := do
  let some kind := Kind.ofName? (← J.str j "receipt" "input") | throw "a receipt kind is sketch-sha512 or block-sha512"
  let r ← Registration.register (← hex j "source") (← hex j "node") (← nat j "blocks") (← hex j "payload_root")
    (← codecOf (← J.get j "codec" "input")) kind (← J.bool j "bind" "input") (← J.str j "layout" "input") (← nat j "rows")
    (← hex j "nonce")
  pure (Json.mkObj [("codec_seed", hexJ r.codecSeed), ("digest", hexJ r.digest)])

/-- A call on the registry. A session names its registration as one drawn earlier in the case, by its place among
those drawn, or as one given in full. -/
inductive Op
  | register (node : ByteArray) (blocks : Nat) (payloadRoot : ByteArray) (codec : Codec.Codec) (receipt : Kind)
      (bind : Bool) (layout : String) (rows : Nat) (nonce : ByteArray)
  | registerStored (view : Core.Registration.View) (store : String) (codec : Codec.Codec) (receipt : Kind)
      (bind : Bool) (layout : String) (rows : Nat) (nonce : ByteArray)
  | session (registration : Nat ⊕ Registration) (params : Session.Params) (coins appendCoins : Array ByteArray)
      (notBeforeNs : Nat) (index : Option Int) (inFlight perCoin : Nat)
  | event (node : ByteArray) (k : Nat) (e : Session.Event)
  | close (node : ByteArray)
  | live (node : ByteArray)
  | drawn (node : ByteArray)

/-- The view of an accepted registration log, `{"params", "log"}` as `Core.Registration.protocol` reads them; a log of
another composite, or one the check refuses, is an error. -/
def loggedView (j : Json) : V Core.Registration.View := do
  let p ← Core.Registration.Params.ofJson (← J.get j "params" "op")
  let l ← J.get j "log" "op"
  Core.Registration.ascii l "the log"
  let c ← Core.Registration.Committed.ofJson l
  if c.composite != p.composite then throw "the log is another composite's"
  match Core.Registration.check p.auditor p.composite c.log with
  | .ok s => pure s.view
  | .error (i, r) => throw s!"entry {i}: {r.name}"

def opOf (j : Json) : V Op := do
  match ← J.str j "op" "op" with
  | "register" =>
    let bind ← J.bool j "bind" "op"
    let some kind := Kind.ofName? (← J.str j "receipt" "op") | throw "a receipt kind is sketch-sha512 or block-sha512"
    pure (.register (← hex j "node") (← nat j "blocks") (← hex j "payload_root") (← codecOf (← J.get j "codec" "op")) kind
      bind (← J.str j "layout" "op") (← nat j "rows") (← hex j "nonce"))
  | "register_stored" =>
    let some kind := Kind.ofName? (← J.str j "receipt" "op") | throw "a receipt kind is sketch-sha512 or block-sha512"
    pure (.registerStored (← loggedView j) (← J.str j "store" "op") (← codecOf (← J.get j "codec" "op")) kind
      (← J.bool j "bind" "op") (← J.str j "layout" "op") (← nat j "rows") (← hex j "nonce"))
  | "session" =>
    let which ← match ← J.get j "registration" "op" with
      | .num n => .inl <$> J.asNat (.num n) "registration"
      | v => .inr <$> registrationOf v
    let (params, coins, appendCoins, notBeforeNs) ← sessionOf j
    let index ← match ← J.get j "index" "op" with
      | .null => pure none
      | v => some <$> J.asInt v "index"
    pure (.session which params coins appendCoins notBeforeNs index (← flightOf j) (← perCoinOf j))
  | "event" => pure (.event (← hex j "node") (← nat j "session") (← eventOf (← J.get j "event" "op")))
  | "close" => pure (.close (← hex j "node"))
  | "live" => pure (.live (← hex j "node"))
  | "drawn" => pure (.drawn (← hex j "node"))
  | k => throw s!"no registry op {k}"

private def refusedJ : Json := Json.mkObj [("refused", .bool true)]

/-- A registration call's result: the registration drawn, or `refused`. -/
private def afterDraw (drawn : Array Registration) (out : Array Json) :
    Option Registration → Array Registration × Array Json
  | some reg => (drawn.push reg, out.push (Json.mkObj [("codec_seed", hexJ reg.codecSeed), ("digest", hexJ reg.digest)]))
  | none => (drawn, out.push refusedJ)

/-- `audit.Registry` under a given source: `{"source", "registry", "ops"}`, from the registry `registry/v0` reads back
(`null` for an empty one), to each call's result (`refused` where `Registry` raises `RegistryError`) and the registry
after them. A session's events go to it by its node and its place among its registration's sessions. -/
def registryFn (j : Json) : V Json := do
  let source ← hex j "source"
  let mut r ← match ← J.get j "registry" "input" with
    | .null => pure ({} : Registry)
    | v => Registry.ofJson v
  let ops ← (← J.arr j "ops" "input").mapM opOf
  let mut drawn : Array Registration := #[]
  let mut out : Array Json := #[]
  for op in ops do
    match op with
    | .register node blocks root codec kind bind layout rows nonce =>
      let (reg?, r') ← r.register source node blocks root codec kind bind layout rows nonce
      r := r'
      (drawn, out) := afterDraw drawn out reg?
    | .registerStored v store codec kind bind layout rows nonce =>
      let (reg?, r') ← r.registerStored source v store codec kind bind layout rows nonce
      r := r'
      (drawn, out) := afterDraw drawn out reg?
    | .session which params coins appendCoins notBeforeNs index f g =>
      let reg ← match which with
        | .inl i => match drawn[i]? with
          | some reg => pure reg
          | none => throw s!"no registration drawn at {i}"
        | .inr reg => pure reg
      match r.session reg params coins appendCoins notBeforeNs index f g with
      | some (k, r') =>
        r := r'
        out := out.push (Json.mkObj [("session", toJson k)])
      | none => out := out.push refusedJ
    | .event node k e =>
      let (res, r') ← r.step node k e
      r := r'
      out := out.push (match res with | some x => resultJ x | none => refusedJ)
    | .close node =>
      let (c, r') ← r.close node
      r := r'
      out := out.push (match c with | some c => c.toJson | none => refusedJ)
    | .live node => out := out.push (match ← r.liveDigest node with | some d => hexJ d | none => .null)
    | .drawn node => out := out.push (toJson (← r.drawn node))
  pure (Json.mkObj [("results", .arr out), ("registry", r.toJson)])

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
  let d := FV3.Sha512.Domain.make binding owner (← nat j "start") (← nat j "count")
  pure (.bool (d.verifyOpening root d.count (← nat j "position") (← hex j "value") (← J.str j "schema" "input") path))

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

private def refusedOr (x : Except String Json) : Json :=
  match x with
  | .ok v => v
  | .error why => Json.mkObj [("refused", .str why)]

/-- `store.pin_of` (`model_pin` with `model`) on a pin file's bytes `data`, read from `path`. -/
def pinFn (j : Json) : V Json := do
  let data ← hex j "data"
  let path ← J.str j "path" "input"
  let role ← match ← J.get j "role" "input" with
    | .null => pure none
    | v => some <$> J.asStr v "role"
  let model ← J.bool j "model" "input"
  let pin ← match model, role with
    | true, some r => pure (Store.modelPin data path r)
    | true, none => throw "a binding's pin is a role's"
    | false, r => pure (Store.pinOf data path r)
  pure (refusedOr ((fun p => Json.mkObj [("pin", p.toJson)]) <$> pin))

/-- `store.canonical_json` of a value with no number. -/
def canonicalJsonFn (j : Json) : V Json := do
  let some s := Store.pyJson (← J.get j "value" "input") | throw "a value with a number"
  pure (.str s)

/-- `store.checkpoint_sha512` of `{file: string}`, with the canonical JSON it hashes. -/
def checkpointFn (j : Json) : V Json := do
  let .obj kvs ← J.get j "files" "input" | throw "files is an object"
  let files ← kvs.toArray.mapM fun (f, h) => do pure (f, ← J.asStr h "files")
  pure (Json.mkObj [("canonical", .str (Store.filesJson files)), ("sha512", .str (Store.checkpointSha512 files))])

def storeSaltFn (j : Json) : V Json := do
  let c ← nat j "c"
  if c ≥ 2 ^ 64 then throw "a block index is a u64"
  pure (hexJ (Store.storeSalt (← hex j "key") (← hex j "label") c))

/-- An arena's weights blocks' digests, `digests` as a list of 64-byte strings, as one string of 64 bytes a block. -/
def digestsOf (j : Json) : V ByteArray := do
  let ds ← hexes j "digests"
  if ds.isEmpty || ds.any (·.size ≠ 64) then throw "an arena's weights are one or more blocks' digests of 64 bytes"
  pure (ds.foldl (· ++ ·) ByteArray.empty)

/-- `store.expected_root` of a store of `engines` arenas of `arena_blocks`, then `host_blocks`, its subtrees of
`2^chunk_log2` leaves on `workers` threads. -/
def expectedRootFn (j : Json) : V Json := do
  let digests ← digestsOf j
  let (engines, arenaBlocks, host) := (← nat j "engines", ← nat j "arena_blocks", ← nat j "host_blocks")
  let node ← hex j "node"
  if engines = 0 || arenaBlocks < Store.weightsOf digests then throw "a store is one or more arenas of at least the weights"
  if !(1 ≤ node.size && node.size ≤ 64) then throw "a node's id is 1 to 64 bytes"
  pure (hexJ (Store.expectedRoot engines arenaBlocks host digests node (← hex j "key") (← nat j "workers")
    (← nat j "chunk_log2")))

def regionOf (j : Json) : V Store.Region := do
  let xs ← J.asArr j "layout"
  if xs.size ≠ 4 then throw "a region is [kind, tier, first, blocks]"
  let kind ← J.asStr xs[0]! "kind"
  let tier ← J.asStr xs[1]! "tier"
  if !["weights", "kv", "filler", "activations"].contains kind || !["hbm", "host"].contains tier then
    throw "a region's kind and tier are regions_of's"
  pure (kind, tier, ← J.asInt xs[2]! "first", ← J.asInt xs[3]! "blocks")

def regionJ (r : Store.Region) : Json := .arr #[.str r.1, .str r.2.1, toJson r.2.2.1, toJson r.2.2.2]

/-- `Binding.hold`'s outcome as the verifier answers it: the regions bound, or why not, and the expected root
`record()` keeps. -/
def heldJ (h : Store.Held) : Json :=
  Json.mkObj [match h.regions with
      | .ok rs => ("regions", .arr (rs.map regionJ))
      | .error why => ("refused", .str why),
    ("expected_root", match h.expectedRoot with | some r => hexJ r | none => .null)]

/-- A hello's store as `Binding.hold` reads it: `{"node", "blocks", "payload_root", "key", "layout"}`, the key the
verifier's payload key and the layout `null` or `regions_of`'s regions as `[kind, tier, first, blocks]`. -/
def helloOf (j : Json) : V (ByteArray × Int × ByteArray × ByteArray × Option (Array Store.Region)) := do
  let layout ← match ← J.get j "layout" "input" with
    | .null => pure none
    | v => some <$> (← J.asArr v "layout").mapM regionOf
  pure (← hex j "node", ← int j "blocks", ← hex j "payload_root", ← hex j "key", layout)

/-- `Binding.hold` of the binding `b` on a hello (`helloOf`), the root on `workers` threads. -/
def holdOf (b : Store.Binding) (hello : Json) (workers : Nat) : V Json := do
  let (node, blocks, root, key, layout) ← helloOf hello
  pure (heldJ (b.hold node blocks root key layout workers))

/-- What the verifier answers of a binding it starts (`Store.bind`): the pin and the weights blocks, or why not. -/
def boundJ : Except String Store.Binding → Json
  | .ok b => Json.mkObj [("pin", b.pin.toJson), ("weights_blocks", toJson (Store.weightsOf b.digests))]
  | .error why => Json.mkObj [("refused", .str why)]

/-- The binding of `role` over an arena of the weights blocks' digests `digests` and `engines` arenas of
`arena_blocks` (`Store`'s check as `Binding` starts), then `Binding.hold` of a hello: the regions bound, or why not,
and the expected root `record()` keeps. -/
def holdFn (j : Json) : V Json := do
  let role ← J.str j "role" "input"
  let digests ← digestsOf j
  let (engines, arenaBlocks) := (← int j "engines", ← int j "arena_blocks")
  let hello ← helloOf j
  match Store.shape engines arenaBlocks (Store.weightsOf digests) with
  | .error why => pure (Json.mkObj [("refused", .str why), ("expected_root", .null)])
  | .ok () =>
    let (node, blocks, root, key, layout) := hello
    pure (heldJ (Store.hold role digests engines.toNat arenaBlocks.toNat node blocks root key layout))

/-- The verifier's binding as its process runs it (`RemoteAudit.Verifier`'s `bind` and `hold`): `{"data", "path",
"role", "digests", "engines", "arena_blocks", "workers", "hellos"}` to what `bind` answers (`bind`) and, once bound,
what `hold` answers of each hello (`holds`, `helloOf`'s). -/
def bindingFn (j : Json) : V Json := do
  let data ← hex j "data"
  let path ← J.str j "path" "input"
  let role ← J.str j "role" "input"
  let digests ← digestsOf j
  let (engines, arenaBlocks, workers) := (← int j "engines", ← int j "arena_blocks", ← nat j "workers")
  let hellos ← J.arr j "hellos" "input"
  let b := Store.bind data path role digests engines arenaBlocks
  let holds ← match b with
    | .ok b => hellos.mapM (holdOf b · workers)
    | .error _ => pure #[]
  pure (Json.mkObj [("bind", boundJ b), ("holds", .arr holds)])

private def ratIn (j : Json) (k ctx : String) : V Rat := do
  match ← J.get j k ctx with
  | .num n => pure ((← J.asInt (.num n) s!"{ctx}.{k}" : Int) : Rat)
  | .str s => match Line.ratOf? s with
    | some q => pure q
    | none => throw s!"{ctx}.{k}: a rational is a string n or n/d, d positive"
  | _ => throw s!"{ctx}.{k}: not an integer or a string"

private def groupOf (j : Json) : V Line.Group := do
  pure
    { releasedNs := ← J.nat j "released_ns" "group"
      lastNs := ← J.nat j "last_ns" "group"
      passes := ← J.nat j "passes" "group"
      least := ← J.nat j "least" "group"
      weights := ← J.nat j "weights" "group"
      hbm := ← J.nat j "hbm" "group"
      host := ← J.nat j "host" "group" }

private def ratJ (q : Rat) : Json := .str (toString q)

private def readingJ (r : Line.Reading) : Json :=
  Json.mkObj [("filler", ratJ r.filler), ("hbm", ratJ r.hbm), ("lost_blocks", ratJ r.lost), ("weights", ratJ r.weights)]

private def stretchJ (x : Line.Stretch) : Json :=
  Json.mkObj [("fill", ratJ x.fill),
    ("held", Json.mkObj [("hbm", ratJ x.hbm), ("host", ratJ x.host), ("weights", ratJ x.weights)]),
    ("hi", toJson x.hi), ("l_max_ns", toJson x.lMaxNs), ("l_sum_ns", toJson x.lSumNs), ("lo", toJson x.lo),
    ("passes", toJson x.passes), ("rebuild_blocks", ratJ x.rebuild), ("snapshot", readingJ x.snapshot),
    ("span_ns", toJson x.spanNs), ("stretch", toJson x.stretch), ("time", readingJ x.time)]

private def formJ (nodeBytes : Nat) (f : Line.Form) : Json :=
  let reading (time : Bool) : Json :=
    let x := Line.worst f.stretches time
    let r := x.reading time
    let share (q : Rat) : Json := ratJ (q / (nodeBytes : Rat))
    Json.mkObj [("filler_share", share r.filler), ("hbm_share", share r.hbm), ("stretch", toJson x.stretch),
      ("weights_share", share r.weights)]
  Json.mkObj [("burst", ratJ f.charge.burst), ("rate", ratJ f.charge.rate),
    ("stretches", .arr (f.stretches.map stretchJ)), ("time", reading true), ("snapshot", reading false)]

private def windowsOf (j : Json) : V (Array (Nat × Nat)) := do
  (← J.arr j "windows" "input").mapM fun w => do
    let xs ← J.asArr w "windows"
    if xs.size ≠ 2 then throw "windows: a window is [receipts, draws]"
    pure (← J.asNat xs[0]! "windows", ← J.asNat xs[1]! "windows")

/-- `pooled.run_json`: the pooled bound of `{"windows": [[U, s], ...], "delta"}`, or why there is none. -/
def pooledFn (j : Json) : V Json := do
  let got := Pooled.pooled (← windowsOf j) (← ratIn j "delta" "input")
  pure (refusedOr ((fun (b : Pooled.Bound) => Json.mkObj [("budget", ratJ b.budget), ("floor", ratJ b.floor),
    ("floors", .arr (b.floors.map ratJ)), ("wrong", ratJ b.wrong)]) <$> got))

private def ratesOf (j : Json) : V Line.Rates := do
  let r ← J.get j "rates" "input"
  pure
    { squaringsPerS := ← ratIn r "squarings_per_s" "rates"
      encodeSquarings := ← J.nat r "encode_squarings" "rates"
      diskBytesPerS := ← ratIn r "disk_bytes_per_s" "rates"
      linkBytesPerS := ← ratIn r "link_bytes_per_s" "rates"
      rebuildBurstBytes := ← ratIn r "rebuild_burst_bytes" "rates"
      diskBurstBytes := ← ratIn r "disk_burst_bytes" "rates"
      linkBurstBytes := ← ratIn r "link_burst_bytes" "rates" }

private def lineJ (got : Except String Line.Line) : Json :=
  refusedOr ((fun (l : Line.Line) => Json.mkObj ((l.forms.toList.map fun (f : Line.Form) =>
    (f.name, formJ l.nodeBytes f)) ++ [("wrong", ratJ l.wrong)])) <$> got)

/-- `line.run_json`: the line of `{"groups", "per", "blocks", "block_bytes", "chip_bytes", "node_bytes", "windows",
"delta", "rates"}` in both forms with the wrong receipts it read, or why it has none. -/
def lineFn (j : Json) : V Json := do
  let groups ← (← J.arr j "groups" "input").mapM groupOf
  pure (lineJ (Line.line groups (← nat j "per") (← nat j "blocks") (← nat j "block_bytes") (← nat j "chip_bytes")
    (← nat j "node_bytes") (← windowsOf j) (← ratIn j "delta" "input") (← ratesOf j)))

private def natsOf (j : Json) (k : String) : V (Array Nat) := do (← J.arr j k "window").mapM (J.asNat · k)

private def hexOrNull (j : Json) (ctx : String) : V (Option ByteArray) := do
  match j with
  | .null => pure none
  | v => some <$> bytesOf v ctx

/-- An accepted session's record as `summaryJ` writes it (the verifier's `outcome` request's `record`), read back. -/
def summaryOf (j : Json) : V Session.Summary := do
  let pace ← match ← J.get j "pace" "record" with
    | .null => pure none
    | q => do pure (some ({ k := ← J.nat q "k" "pace", windowNs := ← J.nat q "window_ns" "pace" } : Session.Pace))
  let windows ← (← J.arr j "windows" "record").mapM fun w => do
    pure ({ lengths := ← natsOf w "lengths", steps := ← natsOf w "steps", post := ← natsOf w "post_seed",
            heads := ← (← J.arr w "heads" "window").mapM (hexOrNull · "heads"), stale := ← natsOf w "stale",
            arrivedNs := ← natsOf w "arrived_ns", releasedNs := ← J.nat w "released_ns" "window",
            deadlineNs := ← J.nat w "deadline_ns" "window", openedNs := ← J.nat w "opened_ns" "window",
            drawn := ← J.nat w "drawn" "window" } : Session.WindowSummary)
  pure { windows, blocks := ← J.nat j "blocks" "record", appended := ← J.nat j "appended" "record",
         released := ← J.nat j "released" "record", lifetime := ← J.nat j "lifetime" "record", pace,
         appends := ← hexOrNull (← J.get j "appends" "record") "appends" }

private def seenOf (j : Json) : V Line.Seen := do
  pure
    { releasedNs := ← J.nat j "released_ns" "seen"
      lastNs := ← J.nat j "last_ns" "seen"
      least := ← J.nat j "least" "seen"
      weights := ← J.nat j "weights" "seen"
      hbm := ← J.nat j "hbm" "seen"
      host := ← J.nat j "host" "seen" }

/-- `line.record_run_json`: the line of an accepted session's record, `{"record", "registration", "samples", "seen",
"per", "block_bytes", "chip_bytes", "node_bytes", "delta", "rates"}` (`record` as the verifier's `outcome` request gives
it, `recordJ`'s; `registration` as `registrationOf` reads one), read off the record (`Line.lineOfRecordOn` at the
record's `groups_per_coin` g, 1 when it doesn't name it: above 1 its pooled bound reads its draws, `Line.ofSummaryAt`),
or why it has none: none for a record drawn more than one window a coin at one window in flight, which no property
covers (`Line.SHARED`), and none for a record not of `registration`, its digest or its registered blocks another
registration's (`Line.lineOfRegistered`). `clock`, if given, says
whose clock the times seen are on: `record` (the default), the record's own, and held to it (`Line.timesOff`), or
`challenger` (the challenger's release log's) or `replay` (a replay's at another round trip), which the record doesn't
hold.

With `session`, the session's transcript (`sessionFn`'s input: its registration, parameters, `not_before_ns`, events
and coins), the line is of the record `Shared.run g` makes of it at its parameters' g (`Line.lineOfRun`, `LineHeld`'s
`lineOf` at a node of g windows a draw, `Session.run` at g = 1, its run shared with the record's check), at its
parameters' `samples`, refused for a transcript run on another registration than `registration`. Refused unless the
transcript has two windows in flight, `samples` is its parameters', and `record`, if given, is the run's (`recordJ`'s
form). -/
def lineRecordFn (j : Json) : V Json := do
  let onRecord ← if !J.has j "clock" then pure true else match ← J.str j "clock" "input" with
    | "record" => pure true
    | "challenger" | "replay" => pure false
    | c => throw s!"clock: record, challenger or replay, not {c}"
  let seen ← (← J.arr j "seen" "input").mapM seenOf
  let samples ← nat j "samples"
  let per ← nat j "per"
  let b ← nat j "block_bytes"
  let chip ← nat j "chip_bytes"
  let node ← nat j "node_bytes"
  let δ ← ratIn j "delta" "input"
  let rates ← ratesOf j
  let registration ← registrationOf (← J.get j "registration" "input")
  if !J.has j "session" then
    let rj ← J.get j "record" "input"
    let digest ← hex rj "registration"
    let sm ← summaryOf rj
    let g ← if J.has rj "groups_per_coin" then J.nat rj "groups_per_coin" "record" else pure 1
    let f ← match ← J.get rj "pace" "record" with
      | .null => pure 2
      | q => if J.has q "in_flight" then J.nat q "in_flight" "pace" else pure 2
    if g != 1 && f == 1 then return lineJ (.error Line.SHARED)
    return lineJ (Line.lineOfRegistered onRecord g registration digest sm samples seen per b chip node δ rates)
  let x ← J.get j "session" "input"
  let g ← perCoinOf x
  let reg ← registrationOf (← J.get x "registration" "session")
  let (params, coins, appendCoins, notBeforeNs) ← sessionOf x
  match ← J.get (← J.get x "params" "session") "pace" "params" with
  | .null => pure ()
  | q =>
    if J.has q "in_flight" then
      if (← J.nat q "in_flight" "pace") != 2 then throw "a line's session has two windows in flight (in_flight 2)"
  if samples ≠ params.samples then throw s!"samples: the session's {params.samples}, not {samples}"
  let events ← (← J.arr x "events" "session").mapM eventOf
  let ran := Shared.run g (Session.State.init params reg coins appendCoins notBeforeNs) events
  if J.has j "record" then
    let given ← J.get j "record" "input"
    let made := match ran with
      | .ok (_, s') => s'.summary.map fun x => recordJ s'.digest x 2 g
      | .error _ => none
    if made.any (· != given) then
      return refusedOr (.error "the record is not the one Session.run makes of the session's transcript")
  return lineJ (match ran with
    | .ok (_, s') => match s'.summary with
      | some sm => Line.lineOfRegistered onRecord g registration s'.digest sm samples seen per b chip node δ rates
      | none => Line.lineOfSummary? onRecord g none samples seen per b chip node δ rates
    | .error _ => Line.lineOfSummary? onRecord g none samples seen per b chip node δ rates)

private def natsOfN (j : Json) (n : Nat) (ctx : String) : V (Array Nat) := do
  let xs ← J.asArr j ctx
  if xs.size ≠ n then throw s!"{ctx}: {n} integers"
  xs.mapM (J.asNat · ctx)

private def blockOf (j : Json) : V Schedule.Block := do
  match j with
  | .null => pure none
  | v =>
    let xs ← J.asArr v "block"
    if xs.size ≠ 2 then throw "an appended block is null or [write, free]"
    let free ← match xs[1]! with
      | .null => pure none
      | f => some <$> J.asNat f "free"
    pure (some (← J.asNat xs[0]! "write", free))

/-- `schedule.run_json`: the schedule check of `{"blocks", "cap", "schedule": {"registered", "appends", "steps"},
"passes": [[step, runs], ...], "appends": [[first, count], ...]}`, to `{"bits"}` or why not. -/
def scheduleFn (j : Json) : V Json := do
  let s ← J.get j "schedule" "input"
  let registered ← (← J.arr s "registered" "schedule").mapM fun r => do
    let xs ← natsOfN r 3 "registered"
    pure (xs[0]!, xs[1]!, xs[2]!)
  let appends ← (← J.arr s "appends" "schedule").mapM fun a => do (← J.asArr a "append").mapM blockOf
  let steps ← (← J.arr s "steps" "schedule").mapM (J.asNat · "steps")
  let passes ← (← J.arr j "passes" "input").mapM fun p => do
    let xs ← J.asArr p "pass"
    if xs.size ≠ 2 then throw "a pass is [step, runs]"
    let runs ← (← J.asArr xs[1]! "runs").mapM fun r => do
      let ys ← natsOfN r 2 "run"
      pure (ys[0]!, ys[1]!)
    pure (← J.asNat xs[0]! "step", runs)
  let made ← (← J.arr j "appends" "input").mapM fun a => do
    let ys ← natsOfN a 2 "append"
    pure (ys[0]!, ys[1]!)
  let got := Schedule.check ⟨registered, appends, steps⟩ (← nat j "blocks") (← nat j "cap") passes made
  pure (refusedOr ((fun b => Json.mkObj [("bits", toJson b)]) <$> got))

def p2RegistrationOf (j : Json) : V P2.Registration := do
  let c ← J.get j "codec" "registration"
  P2.Registration.make? (← hex j "node") (← J.nat c "w" "codec") (← J.nat c "c0" "codec") (← J.str c "expander" "codec")
    (← J.nat c "blocks_per_segment" "codec") (← J.nat c "segments" "codec") (← hex j "commitment") (← hex j "salt")
    (← hex j "root") (← nat j "rows")

/-- `p2.Registration.digest`. -/
def p2RegistrationDigestFn (j : Json) : V Json := do pure (hexJ (← p2RegistrationOf j).digest)

/-- `p2.stored` on an accepted registration log: `{"params", "log", "store", "blocks"}` to the node. -/
def p2StoredFn (j : Json) : V Json := do
  pure (hexJ (← P2.stored? (← loggedView j) (← J.str j "store" "input") (← nat j "blocks")))

private def p2Seed (j : Json) : V ByteArray := do
  let s ← hex j "seed"
  if s.size ≠ DIGEST_BYTES then throw "a pass seed is 64 bytes"
  pure s

private def p2Blocks (j : Json) : V (Array ByteArray) := do
  let bs ← hexes j "blocks"
  if bs.size ≠ P2.UNIT_BLOCKS || bs.any (·.size ≠ P2.BLOCK_BYTES) then throw "a unit is four 2,056-byte blocks"
  pure bs

/-- `p2.Sketcher` on unit g's four blocks: each block's partial π (the decode's, `Sketcher.block`), their sum σ
(`Sketcher.total`) and the receipt from σ (`Sketcher.receipt`), which the verifier's, of the unit's bytes, must be. -/
def p2SketchFn (j : Json) : V Json := do
  let s ← p2Seed j
  let rows ← nat j "rows"
  if !Kind.sketch.rowsOk rows then throw "sketch-sha512's rows are 16 or 64"
  let g ← nat j "g"
  if g ≥ 2 ^ 64 then throw "a unit index is a u64"
  let bs ← p2Blocks j
  let signs := P2.unitSigns s rows
  let ps := (Array.range P2.UNIT_BLOCKS).map fun b => P2.blockPartial signs (P2.UNIT_BLOCKS * g + b) bs[b]!
  let sigma := P2.int32s (P2.total rows ps)
  pure (Json.mkObj [("partials", .arr (ps.map (hexJ ∘ P2.int32s))), ("sigma", hexJ sigma),
    ("receipt", hexJ (P2.unitReceipt s g sigma))])

private def p2OpeningOf (j : Json) : V P2.Opening := do
  pure
    { pass := ← int j "pass"
      position := ← int j "position"
      blocks := ← hexes j "blocks"
      paths := ← (← J.arr j "paths" "input").mapM fun x => do (← J.asArr x "paths").mapM (bytesOf · "paths")
      leaf := ← hex j "leaf"
      leafSalt := ← hex j "leaf_salt"
      receiptPath := ← hexes j "receipt_path" }

/-- `p2.check` of an opening of position `p` of pass `i` under `seed`, the plan of `runs` over the registration's
units and the pass's receipt root `root` (64 bytes, as a session receives it): the failure's name, or null. -/
def p2OpeningFn (j : Json) : V Json := do
  let reg ← p2RegistrationOf (← J.get j "registration" "input")
  let s ← p2Seed j
  let some plan := Plan.make? (← runsOf (← J.get j "runs" "input") "runs") reg.units | throw "not a plan"
  let p ← nat j "p"
  if p ≥ plan.length then throw "a position past the plan"
  let o ← p2OpeningOf (← J.get j "opening" "input")
  let root ← hex j "root"
  if root.size ≠ DIGEST_BYTES then throw "a pass's root is 64 bytes"
  pure (match P2.checkOpening reg s plan root (← nat j "i") p o with
    | some f => .str f.name
    | none => .null)

/-- A P2 session of record (`P2Session`) on the bytes the verifier reads: `{"registration", "request", "not_before_ns",
"coins", "chunk", "events"}`, the registration file's bytes and the prover's request line (lowercase hex), the session's
2W coins, the draws a thread checks, and its events: `start` and `boundary` `{"now"}`, `close` `{"t", "now"}`, `root`
`{"t", "i", "now", "line"}` and `openings` `{"t", "line"}` (each line lowercase hex, as the prover sent it), and `fail`.
The held store (`Held.json`), every event's result (`resultJ`) and the answer (`P2Session.answer`); a registration
or a request refused is `{"refused": "registration"}` or `{"refused": "request"}`. -/
def p2SessionFn (j : Json) : V Json := do
  let .ok held := P2Session.heldOf (← hex j "registration") | pure (Json.mkObj [("refused", .str "registration")])
  let .ok params := P2Session.paramsOf held (← hex j "request") | pure (Json.mkObj [("refused", .str "request")])
  let coins ← hexes j "coins"
  if coins.size ≠ 2 * params.windows || coins.any (·.size ≠ Codec.COIN_BYTES) then throw "2W coins of 32 bytes"
  let chunk ← nat j "chunk"
  let mut s := P2Session.init held params coins (← nat j "not_before_ns")
  let mut results : Array Json := #[]
  for e in ← J.arr j "events" "input" do
    let (r, s') ← match ← J.str e "kind" "event" with
      | "start" => pure (Session.start s (← nat e "now"))
      | "boundary" => pure (Session.boundary s (← nat e "now"))
      | "close" => Session.close s (← int e "t") (← nat e "now")
      | "root" => pure (P2Session.receive held s (← int e "t") (← int e "i") (← hex e "line") (← nat e "now"))
      | "openings" => pure (P2Session.verify chunk held s (← int e "t") (← hex e "line"))
      | "fail" => pure (P2Session.fail s)
      | k => throw s!"no P2 event kind {k}"
    results := results.push (resultJ r)
    s := s'
  pure (Json.mkObj [("held", held.json), ("results", .arr results), ("answer", P2Session.answer held s)])

def table : String → Option _root_.Difftest.Fn := _root_.Difftest.table [
  ("session", sessionFn), ("registration_digest", registrationDigestFn), ("register", registerFn),
  ("registry", registryFn), ("codec_decode", codecDecodeFn),
  ("plan", planFn), ("plan_changed", planChangedFn), ("receipt", receiptFn), ("verify_opening", verifyOpeningFn),
  ("leaf_receipt", leafReceiptFn), ("draws", drawsFn), ("schedule", scheduleFn),
  ("pin", pinFn), ("canonical_json", canonicalJsonFn), ("checkpoint_sha512", checkpointFn),
  ("store_salt", storeSaltFn), ("expected_root", expectedRootFn), ("hold", holdFn), ("binding", bindingFn),
  ("line", lineFn), ("line_record", lineRecordFn), ("pooled", pooledFn),
  ("p2_registration_digest", p2RegistrationDigestFn), ("p2_stored", p2StoredFn), ("p2_sketch", p2SketchFn), ("p2_opening", p2OpeningFn),
  ("p2_session", p2SessionFn)]

end RemoteAudit.Difftest
