import Std.Sync.Mutex
import RemoteAudit.Difftest
import RemoteAudit.P2Session
import RemoteAudit.Wire

/-!
# The verifier's process: `remote-audit-verifier`

The remote audit's Lean verifier as a process the driver (`benchmarks/remote_session/verifier.py`) runs beside its
Python: it holds the verifier's registry (`Registry`) and every session's state (`Session.State`) and takes the events
as they happen, one request a line on stdin, each answered by one line on stdout, `{"answer": JSON}` or
`{"error": MESSAGE}`, in canonical JSON (`Flock.canon`). Requests (byte strings lowercase hex, an event as the
difftest's `session` reads one, `Difftest.eventOf`):

* `{"op": "init", "source", "registry", "chunk"}`: the verifier's secret, the registry `registry/v0` reads back (`null`
  for none), and the draws a thread checks at once; the answer is the registry;
* `{"op": "bind", "bytes": n, "pin", "path", "role", "engines", "arena_blocks", "workers"}` and then n bytes, the
  pinned arena's weights blocks' digests (64 bytes a block, `store.weight_digests`'s file): the store binding of
  `verifier.py --model role` (`Store.bind`: the pin from its file's bytes `pin`, read from `path`, and the store's
  shape), `{"pin", "weights_blocks"}` or `{"refused": why}` (`Difftest.boundJ`), the verifier's binding from then on;
* `{"op": "hold", "node", "blocks", "payload_root", "key", "layout"}`: a hello's store held to that binding
  (`Store.Binding.hold`, its payload root's subtrees on `workers` threads), `{"regions", "expected_root"}` or
  `{"refused": why, "expected_root"}` (`Difftest.heldJ`), the decision of record on the hello's `bound`;
* `{"op": "register", "node", "blocks", "payload_root", "codec", "receipt", "bind", "rows", "layout", "nonce"}`:
  `Registry.register`'s draw, `{"codec_seed", "digest"}`, or `{"refused": true}` (an error for a node with none live
  whose registration would bind PoUW's tile outputs: receipts here are of a block alone);
* `{"op": "session", "node", "params", "coins", "append_coins", "not_before_ns", "index"}`: the next session over the
  node's live registration drawn here (`Registry.session`), `{"session": k}`, or `{"refused": true}`, its windows in
  flight and the windows a draw's openings coin covers (`groups_per_coin` g, `Shared`) its parameters', and an error
  at one window in flight with g above 1, which no property covers (`Flight`); `coins` `null` takes its W + ⌈W/g⌉
  live, each with the event that releases it (`event`'s `coin`);
* `{"op": "event", "node", "session", "event", "coin"}`: one event through session k at its windows in flight and its
  g (`eventOn`, refused as `Registry.step` refuses a session it isn't running), its result as the difftest writes it
  (`Difftest.resultJ`); `coin` (or `null`) is the coin the challenger released for the event,
  set first (`withCoin`): a start's or a boundary's the challenge of the window it releases, a close's window t's
  draw's openings coin, `W + t / g`, which only the draw's last window's close releases, an append's pushed onto the
  session's append coins. A coin set is never replaced by another, and an event that releases a challenge or draws a
  window's openings from a coin never set is an error (`unset`);
* `{"op": "openings", "node", "session", "t", "bytes": n}` and then n bytes: one frame of window t's openings as the
  wire carries them (`Wire.decodeOpenings`), kept until the window is verified;
* `{"op": "verify", "node", "session", "t", "id"}`: window t's openings, every frame in the order it came, checked on
  threads of their own (`windowCheck`), answered `{"spawned": true}` at once, or refused as `Session.verify`
  refuses; when the check is done the process writes `{"done": id, "accepted": bool}` on a line of its own;
* `{"op": "settle", "id"}`: that verdict applied to the session as it stands (`Session.verifyBy`), its result with
  the openings it read (`openings`) and the nanoseconds the check took (`ns`);
* `{"op": "abort", "node", "session", "outcome"}`: the session ended with the driver's outcome (`Registry.abort`);
* `{"op": "outcome", "node", "session"}`: `{"outcome", "record"}`, as the difftest's `session` writes them
  (`Difftest.recordJ`);
* `{"op": "close", "node"}`: `Registry.close`'s registration, or `{"refused": true}`;
* `{"op": "registry"}`: the registry, `registry/v0`;
* `{"op": "line_record", "record", "registration", "samples", "seen", "per", "block_bytes", "chip_bytes",
  "node_bytes", "delta", "rates"}`, with `"clock"` and `"session"` if given: an accepted paced session's memory line
  read off its record (`record`, as `outcome` gives it: `Line.lineOfRecordOn`, its groups, windows and registered
  blocks the record's, the times and reads beside each group `seen`, held to the record's times on its `clock`, its W
  the pooled bound of its windows, or at `groups_per_coin` g > 1 of its draws, `Pooled.pooled`), refused unless the
  record is of `registration` (`Line.lineOfRegistered`), or off the record `Shared.run g` makes of the session's
  transcript `session` (`Line.lineOfRun`), as the difftest's `line_record` writes it (`Difftest.lineRecordFn`), or
  `{"refused": why}`, a record drawn g > 1 a coin at one window in flight among them (`Line.SHARED`): the line of
  record (`benchmarks/remote_session/auditor_line.py`) is this answer, held equal to `line.py`'s;
* `{"op": "line", "groups", "per", "blocks", "block_bytes", "chip_bytes", "node_bytes", "windows", "delta", "rates"}`:
  the same line of groups and windows given (`Line.line`, `Difftest.lineFn`).

A session over PoUS's P2 store is of record here too (`P2Session`), as `p2_party.py` runs it beside `p2.Session`, which
can only veto: the process holds one P2 store and its sessions, each read from the bytes the verifier holds and the
prover sends.

* `{"op": "p2_hold", "bytes": n, "chunk"}` and then n bytes, the auditor's registration file
  (`p2_party.registration_json`'s): the store held (`P2Session.heldOf`), `{"digest", "plan_units", "units"}`, and the
  draws a thread checks;
* `{"op": "p2_session", "bytes": n, "not_before_ns"}` and then the prover's request line: `{"session": k}`, its coins
  released one an event as `event`'s are (`withCoin`), or `{"refused": why}` (`P2Session.paramsOf`);
* `{"op": "p2_event", "session", "event", "coin"}`: a start, a boundary or a close through P2 session k at two windows
  in flight and one window a coin (`eventOn 2 1`), `Difftest.resultJ`'s result;
* `{"op": "p2_root", "session", "t", "i", "now", "bytes": n}` and then root i of window t's line as the prover sent it
  (`P2Session.receive`), and `{"op": "p2_openings", "session", "t", "bytes": n}` and then window t's openings line
  (`P2Session.verify`, checked on threads of their own), each `Difftest.resultJ`'s result;
* `{"op": "p2_fail", "session"}`: the session rejected `malformed`, the prover gone (`P2Session.fail`);
* `{"op": "p2_answer", "session"}`: the session's answer (`P2Session.answer`).

No property states yet what this loop accepts of a P2 session; `remote-audit-difftest`'s `p2_session` holds it to
`p2.Session` on recorded sessions.

A window's verdict is `checkWindowChunked`'s, which is `checkWindow`'s, on the window's openings as `Wire.openingsOf`
reads its frames: any frame that doesn't decode leaves it none, and the transcript records the frames so. A window
with no draws is rejected `malformed` at its settle (`Session.verifyBy`), whatever its frames, so an undecodable frame
is refused on every window, as the driver's Python refuses it. `Properties.RemoteAudit.LiveIsRun` states that what
this loop accepts at two windows in flight and g windows a draw's openings coin, its coins arriving one an event and
each verdict applied at its settle, is what `Shared.run g` accepts on the events it records (`Session.run` at g = 1);
none yet covers one in flight (`Flight`), and the `session` request refuses one in flight at g above 1. Openings are
checked in the clear.
-/

namespace RemoteAudit.Verifier

open Lean (Json ToJson toJson)
open Flock

private def bytesOf (j : Json) (k : String) : V ByteArray := do
  match unhex (← J.str j k "request") with
  | .ok b => pure b
  | .error e => throwThe String s!"request.{k}: {e}"

private def nat (j : Json) (k : String) : V Nat := J.nat j k "request"

private def refusedJ : Json := Json.mkObj [("refused", .bool true)]

private def ofV {α : Type} (x : V α) : ExceptT String IO α := ExceptT.mk (pure x)

/-- A window's check on threads of its own: its verdict, the openings it read and the nanoseconds it took. -/
structure Check where
  id : Nat
  node : ByteArray
  session : Nat
  t : Int
  task : Task (Except IO.Error (Verdict × Nat × Nat))

structure Server where
  source : ByteArray := ByteArray.empty
  registry : Registry := {}
  chunk : Nat := 1024
  bodies : Array ((ByteArray × Nat × Int) × Array ByteArray) := #[]
  checks : Array Check := #[]
  binding : Option Store.Binding := none
  workers : Nat := 1
  p2 : Option P2Session.Held := none
  p2Chunk : Nat := 64
  p2Sessions : Array Session.State := #[]

private def doneLine (id : Nat) (v : Verdict) : String :=
  "{\"accepted\":" ++ (if v == .accept then "true" else "false") ++ ",\"done\":" ++ toString id ++ "}"

/-- Window `w`'s verdict on its frames `bodies` in session `s`, with the number of openings it read. -/
def windowCheck (chunk : Nat) (s : Session.State) (w : WindowRecord) (drawn : Array (Nat × Nat))
    (bodies : Array ByteArray) : Verdict × Nat :=
  let openings := Wire.openingsOf bodies
  (checkWindowChunked chunk s.reg s.digest s.appends w drawn openings, openings.size)

/-- Window t's openings checked on a thread of its own, `emit`ting the done line once its verdict is known. -/
def spawnCheck (chunk : Nat) (s : Session.State) (w : WindowRecord) (drawn : Array (Nat × Nat))
    (bodies : Array ByteArray) (id : Nat) (emit : String → IO Unit) :
    BaseIO (Task (Except IO.Error (Verdict × Nat × Nat))) :=
  IO.asTask (prio := .dedicated) do
    let t0 ← IO.monoNanosNow
    let (v, n) := windowCheck chunk s w drawn bodies
    emit (doneLine id v)
    let t1 ← IO.monoNanosNow
    return (v, n, t1 - t0)

private def runOf (r : Registry) (node : ByteArray) (k : Nat) : Option Registry.Run :=
  (r.live.find? (·.node == node)).bind (·.sessions[k]?)

/-- Session `s`, drawn `g` windows a coin, with the coin `c` the challenger released for event `e` (module docstring),
or why not: a close's is its draw's openings coin, `W + t / g`, and only the draw's last window's close releases it. -/
def withCoin (g : Nat) (s : Session.State) (e : Session.Event) (c : ByteArray) : Except String Session.State := do
  if c.size ≠ Codec.COIN_BYTES then throw "a coin is 32 bytes"
  let set (i : Nat) : Except String Session.State :=
    match s.coins[i]? with
    | none => throw s!"the session has no coin {i}"
    | some x =>
      if x.isEmpty || x == c then pure { s with coins := s.coins.set! i c }
      else throw s!"coin {i} is released once: the challenger sent another"
  match e with
  | .start _ | .boundary _ =>
    if s.windows.size < s.params.windows then set s.windows.size
    else throw "every window's challenge is released"
  | .close t _ =>
    if !(0 ≤ t && t < s.params.windows) then throw s!"no window {t}"
    else if t.toNat + 1 ≠ min (t.toNat / g * g + g) s.params.windows then
      throw s!"window {t} is not its draw's last: only that window's close releases the draw's openings coin"
    else set (s.params.windows + t.toNat / g)
  | .append .. => pure { s with appendCoins := s.appendCoins.push c }
  | _ => throw "only a start, a boundary, a close or an append releases a coin"

/-- Why event `e`'s result `res` read a coin of `s`, the session after it, drawn `g` windows a coin, that the
challenger never sent; none when it read none. -/
def unset (g : Nat) (s : Session.State) (e : Session.Event) (res : Option Session.Result) : Option String :=
  let missing (i : Nat) : Bool := (s.coins[i]?.map (·.size != Codec.COIN_BYTES)).getD true
  match e, res with
  | .start _, some (.released (some t)) | .boundary _, some (.released (some t)) =>
    if missing t then some s!"window {t}'s challenge left without the challenger's coin" else none
  | .close t _, some (.drew _) =>
    if missing (s.params.windows + t.toNat / g) then
      some s!"window {t}'s openings were drawn without the challenger's coin"
    else none
  | _, _ => none

/-- Event `e` through running session `s` at its `f` windows in flight and `g` windows a draw's openings coin
(`Flight.step f g`, as `Registry.step` steps it), the challenger's coin for it set first (`withCoin`), refused as
`unset` refuses an event that read a coin never sent. -/
def eventOn (f g : Nat) (s : Session.State) (e : Session.Event) (coin : Option ByteArray) :
    Except String (Session.Result × Session.State) := do
  let s ← match coin with
    | some c => withCoin g s e c
    | none => pure s
  let (res, s) ← Flight.step f g s e
  if let some why := unset g s e (some res) then throw why
  pure (res, s)

private def merge (a : Json) (extra : List (String × Json)) : Json :=
  match a with
  | .obj kvs => Json.mkObj (kvs.toList ++ extra)
  | _ => a

/-- Read exactly `n` bytes of the input. -/
def readBytes (stdin : IO.FS.Stream) (n : Nat) : IO ByteArray := do
  let mut buf := ByteArray.emptyWithCapacity n
  for _ in [0:n + 1] do
    if buf.size ≥ n then break
    let got ← stdin.read (n - buf.size).toUSize
    if got.isEmpty then throw (IO.userError s!"the input ended {n - buf.size} bytes into an openings frame")
    buf := buf ++ got
  return buf

/-- The P2 store the server holds and its session `k`. -/
private def p2At (srv : Server) (k : Nat) : V (P2Session.Held × Session.State) := do
  let some held := srv.p2 | throw "no P2 store: p2_hold holds one first"
  let some s := srv.p2Sessions[k]? | throw s!"no P2 session {k}"
  pure (held, s)

/-- One request's answer and the server after it. -/
def handle (srv : Server) (req : Json) (stdin : IO.FS.Stream) (emit : String → IO Unit) :
    IO (Except String (Json × Server)) := ExceptT.run do
  match ← ofV (J.str req "op" "request") with
  | "init" =>
    let registry ← match ← ofV (J.get req "registry" "request") with
      | .null => pure ({} : Registry)
      | v => ofV (Registry.ofJson v)
    let source ← ofV (bytesOf req "source")
    let chunk ← ofV (nat req "chunk")
    pure (Json.mkObj [("registry", registry.toJson)], { srv with source, registry, chunk })
  | "bind" =>
    let digests ← readBytes stdin (← ofV (nat req "bytes"))
    if srv.binding.isSome then throwThe String "the verifier's store is bound once"
    if digests.isEmpty || digests.size % 64 ≠ 0 then
      throwThe String "an arena's weights are one or more blocks' digests of 64 bytes"
    let b := Store.bind (← ofV (bytesOf req "pin")) (← ofV (J.str req "path" "request"))
      (← ofV (J.str req "role" "request")) digests (← ofV (J.asInt (← ofV (J.get req "engines" "request")) "engines"))
      (← ofV (J.asInt (← ofV (J.get req "arena_blocks" "request")) "arena_blocks"))
    let workers ← ofV (nat req "workers")
    pure (Difftest.boundJ b, { srv with binding := b.toOption, workers := max 1 workers })
  | "hold" =>
    let some b := srv.binding | throwThe String "no binding: a hello is held to the store `bind` bound"
    pure (← ofV (Difftest.holdOf b req srv.workers), srv)
  | "register" =>
    let some kind := Kind.ofName? (← ofV (J.str req "receipt" "request"))
      | throwThe String "a receipt kind is sketch-sha512 or block-sha512"
    let (reg?, r) ← ofV (srv.registry.register srv.source (← ofV (bytesOf req "node")) (← ofV (nat req "blocks"))
      (← ofV (bytesOf req "payload_root")) (← ofV (Difftest.codecOf (← ofV (J.get req "codec" "request")))) kind
      (← ofV (J.bool req "bind" "request")) (← ofV (J.str req "layout" "request")) (← ofV (nat req "rows"))
      (← ofV (bytesOf req "nonce")))
    let out := match reg? with
      | some reg => Json.mkObj [("codec_seed", .str (hexOf reg.codecSeed)), ("digest", .str (hexOf reg.digest))]
      | none => refusedJ
    pure (out, { srv with registry := r })
  | "session" =>
    let node ← ofV (bytesOf req "node")
    let live := match (← ofV (J.get req "coins" "request")) with
      | .null => true
      | _ => false
    let zero := Json.str (hexOf (ByteArray.mk (Array.replicate Codec.COIN_BYTES 0)))
    let g ← ofV (Difftest.perCoinOf req)
    let req ← if live then do
        let w ← ofV (J.nat (← ofV (J.get req "params" "request")) "windows" "params")
        pure (req.setObjVal! "coins" (.arr (Array.replicate (w + (w + g - 1) / g) zero)))
      else pure req
    let (params, coins, appendCoins, notBeforeNs) ← ofV (Difftest.sessionOf req)
    let f ← ofV (Difftest.flightOf req)
    if f == 1 && g != 1 then
      throwThe String ("at one window in flight the verifier draws a window an openings coin: groups_per_coin is 1 "
        ++ "(no property covers one in flight at more)")
    let coins := if live then Array.replicate coins.size ByteArray.empty else coins
    let index ← match ← ofV (J.get req "index" "request") with
      | .null => pure none
      | v => some <$> ofV (J.asInt v "index")
    let some reg := (srv.registry.live.find? (·.node == node)).bind (·.drawn) | pure (refusedJ, srv)
    match srv.registry.session reg params coins appendCoins notBeforeNs index f g with
    | some (k, r) => pure (Json.mkObj [("session", toJson k)], { srv with registry := r })
    | none => pure (refusedJ, srv)
  | "event" =>
    let node ← ofV (bytesOf req "node")
    let k ← ofV (nat req "session")
    let e ← ofV (Difftest.eventOf (← ofV (J.get req "event" "request")))
    let coin ← ofV (J.get req "coin" "request")
    match runOf srv.registry node k with
    | some (.running s f g) =>
      let coin ← match coin with
        | .null => pure none
        | c => some <$> ofV (unhex (← ofV (J.asStr c "coin")))
      let (res, s) ← ofV (eventOn f g s e coin)
      pure (Difftest.resultJ res, { srv with registry := srv.registry.update node k s })
    | _ =>
      let (res, r) ← ofV (srv.registry.step node k e)
      pure ((res.map Difftest.resultJ).getD refusedJ, { srv with registry := r })
  | "openings" =>
    let node ← ofV (bytesOf req "node")
    let k ← ofV (nat req "session")
    let t ← ofV (J.asInt (← ofV (J.get req "t" "request")) "t")
    let body ← readBytes stdin (← ofV (nat req "bytes"))
    let key := (node, k, t)
    let bodies := match srv.bodies.findIdx? (·.1 == key) with
      | some i => srv.bodies.modify i fun (x, bs) => (x, bs.push body)
      | none => srv.bodies.push (key, #[body])
    pure (Json.mkObj [("ok", .bool true)], { srv with bodies })
  | "verify" =>
    let node ← ofV (bytesOf req "node")
    let k ← ofV (nat req "session")
    let t ← ofV (J.asInt (← ofV (J.get req "t" "request")) "t")
    let id ← ofV (nat req "id")
    if srv.checks.any (·.id == id) then throwThe String s!"check {id} is already running"
    let key := (node, k, t)
    let bodies := ((srv.bodies.find? (·.1 == key)).map (·.2)).getD #[]
    let srv := { srv with bodies := srv.bodies.filter (·.1 != key) }
    let some s := srv.registry.running node k | pure (refusedJ, srv)
    let some (w, drawn) := s.awaiting t | pure (refusedJ, srv)
    let task ← spawnCheck srv.chunk s w drawn bodies id emit
    pure (Json.mkObj [("spawned", .bool true)], { srv with checks := srv.checks.push { id, node, session := k, t, task } })
  | "settle" =>
    let id ← ofV (nat req "id")
    let some c := srv.checks.find? (·.id == id) | throwThe String s!"no check {id}"
    let srv := { srv with checks := srv.checks.filter (·.id != id) }
    let (v, n, ns) ← match ← IO.wait c.task with
      | .ok x => pure x
      | .error e => throwThe String s!"check {id}: {e}"
    let extra := [("openings", toJson n), ("ns", toJson ns)]
    let some s := srv.registry.running c.node c.session | pure (merge refusedJ extra, srv)
    let (res, s) := Session.verifyBy s c.t fun _ => v
    pure (merge (Difftest.resultJ res) extra, { srv with registry := srv.registry.update c.node c.session s })
  | "abort" =>
    let node ← ofV (bytesOf req "node")
    let k ← ofV (nat req "session")
    let o ← ofV (J.str req "outcome" "request")
    if o.isEmpty then throwThe String "an outcome is a nonempty string"
    pure (Json.mkObj [("ok", .bool true)], { srv with registry := srv.registry.abort node k o })
  | "outcome" =>
    let node ← ofV (bytesOf req "node")
    let k ← ofV (nat req "session")
    let out := match runOf srv.registry node k with
      | some (.running s f g) => Json.mkObj [("outcome", match s.outcome with | some o => .str o | none => .null),
          ("record", match s.summary with | some x => Difftest.recordJ s.digest x f g | none => .null)]
      | some (.read o) => Json.mkObj [("outcome", .str o), ("record", .null)]
      | none => refusedJ
    pure (out, srv)
  | "close" =>
    let (c, r) ← ofV (srv.registry.close (← ofV (bytesOf req "node")))
    pure ((c.map Registry.Closed.toJson).getD refusedJ, { srv with registry := r })
  | "registry" => pure (srv.registry.toJson, srv)
  | "line" => pure (← ofV (Difftest.lineFn req), srv)
  | "line_record" => pure (← ofV (Difftest.lineRecordFn req), srv)
  | "p2_hold" =>
    let bytes ← readBytes stdin (← ofV (nat req "bytes"))
    if srv.p2.isSome then throwThe String "the verifier holds one P2 store"
    let held ← ofV (P2Session.heldOf bytes)
    let chunk ← ofV (nat req "chunk")
    pure (held.json, { srv with p2 := some held, p2Chunk := max 1 chunk })
  | "p2_session" =>
    let line ← readBytes stdin (← ofV (nat req "bytes"))
    let notBeforeNs ← ofV (nat req "not_before_ns")
    let some held := srv.p2 | throwThe String "no P2 store: p2_hold holds one first"
    match P2Session.paramsOf held line with
    | .error why => pure (Json.mkObj [("refused", .str why)], srv)
    | .ok params =>
      let s := P2Session.init held params (Array.replicate (2 * params.windows) ByteArray.empty) notBeforeNs
      pure (Json.mkObj [("session", toJson srv.p2Sessions.size)], { srv with p2Sessions := srv.p2Sessions.push s })
  | "p2_event" =>
    let k ← ofV (nat req "session")
    let e ← ofV (Difftest.eventOf (← ofV (J.get req "event" "request")))
    let coin ← match ← ofV (J.get req "coin" "request") with
      | .null => pure none
      | c => some <$> ofV (unhex (← ofV (J.asStr c "coin")))
    let (_, s) ← ofV (p2At srv k)
    match e with
    | .start _ | .boundary _ | .close .. =>
      let (res, s) ← ofV (eventOn 2 1 s e coin)
      pure (Difftest.resultJ res, { srv with p2Sessions := srv.p2Sessions.set! k s })
    | _ => throwThe String "a P2 session's events are a start, a boundary or a close: a root or openings is a line"
  | "p2_root" =>
    let line ← readBytes stdin (← ofV (nat req "bytes"))
    let k ← ofV (nat req "session")
    let t ← ofV (J.asInt (← ofV (J.get req "t" "request")) "t")
    let i ← ofV (J.asInt (← ofV (J.get req "i" "request")) "i")
    let now ← ofV (nat req "now")
    let (held, s) ← ofV (p2At srv k)
    let (res, s) := P2Session.receive held s t i line now
    pure (Difftest.resultJ res, { srv with p2Sessions := srv.p2Sessions.set! k s })
  | "p2_openings" =>
    let line ← readBytes stdin (← ofV (nat req "bytes"))
    let k ← ofV (nat req "session")
    let t ← ofV (J.asInt (← ofV (J.get req "t" "request")) "t")
    let (held, s) ← ofV (p2At srv k)
    let (res, s) := P2Session.verify srv.p2Chunk held s t line
    pure (Difftest.resultJ res, { srv with p2Sessions := srv.p2Sessions.set! k s })
  | "p2_fail" =>
    let k ← ofV (nat req "session")
    let (_, s) ← ofV (p2At srv k)
    let (res, s) := P2Session.fail s
    pure (Difftest.resultJ res, { srv with p2Sessions := srv.p2Sessions.set! k s })
  | "p2_answer" =>
    let (held, s) ← ofV (p2At srv (← ofV (nat req "session")))
    pure (P2Session.answer held s, srv)
  | op => throwThe String s!"no op {op}"

private def line (key : String) (value : String) : String := "{" ++ Flock.canonStr key ++ ":" ++ value ++ "}"

/-- Serve requests until the input ends. -/
def serve : IO UInt32 := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let lock ← Std.BaseMutex.new
  let emit : String → IO Unit := fun s => do
    lock.lock
    try
      stdout.putStrLn s
      stdout.flush
    finally
      lock.unlock
  let mut srv : Server := {}
  repeat
    let request ← stdin.getLine
    if request.isEmpty then break
    let (out, srv') ← match Json.parse request with
      | .error e => pure (line "error" (Flock.canonStr s!"request: not JSON: {e}"), srv)
      | .ok req => do
        match ← handle srv req stdin emit with
        | .error e => pure (line "error" (Flock.canonStr e), srv)
        | .ok (a, srv') =>
          match Flock.canon a with
          | .ok text => pure (line "answer" text, srv')
          | .error e => pure (line "error" (Flock.canonStr s!"answer: {e}"), srv')
    srv := srv'
    emit out
  return 0

end RemoteAudit.Verifier
