import Std.Sync.Mutex
import RemoteAudit.Difftest
import RemoteAudit.Wire

/-!
# The verifier's process: `remote-audit-verifier`

The remote audit's verifier of record as a process the driver (`benchmarks/remote_session/verifier.py`) runs beside its
Python: it holds the verifier's registry (`Registry`) and every session's state (`Session.State`) and takes the events
as they happen, one request a line on stdin, each answered by one line on stdout, `{"answer": JSON}` or
`{"error": MESSAGE}`, in canonical JSON (`Flock.canon`). Requests (byte strings lowercase hex, an event as the
difftest's `session` reads one, `Difftest.eventOf`):

* `{"op": "init", "source", "registry", "chunk"}`: the verifier's secret, the registry `registry/v0` reads back (`null`
  for none), and the draws a thread checks at once; the answer is the registry;
* `{"op": "register", "node", "blocks", "payload_root", "codec", "receipt", "rows", "layout", "nonce"}`:
  `Registry.register`'s draw, `{"codec_seed", "digest"}`, or `{"refused": true}`;
* `{"op": "session", "node", "params", "coins", "append_coins", "not_before_ns", "index"}`: the next session over the
  node's live registration drawn here (`Registry.session`), `{"session": k}`, or `{"refused": true}`;
* `{"op": "event", "node", "session", "event", "coin"}`: one event through session k (`Registry.step`), its result as
  the difftest writes it (`Difftest.resultJ`); `coin` (or `null`) is the append coin the driver drew for an append,
  pushed onto the session's append coins first;
* `{"op": "openings", "node", "session", "t", "bytes": n}` and then n bytes: one frame of window t's openings as the
  wire carries them (`Wire.decodeOpenings`), kept until the window is verified;
* `{"op": "verify", "node", "session", "t", "id"}`: window t's openings, every frame in the order it came, checked on
  threads of their own (`checkWindowChunked`), answered `{"spawned": true}` at once, or refused as `Session.verify`
  refuses; when the check is done the process writes `{"done": id, "accepted": bool}` on a line of its own;
* `{"op": "settle", "id"}`: that verdict applied to the session as it stands (`Session.verifyBy`), its result with
  the openings it read (`openings`) and the nanoseconds the check took (`ns`);
* `{"op": "abort", "node", "session", "outcome"}`: the session ended with the driver's outcome (`Registry.abort`);
* `{"op": "outcome", "node", "session"}`: `{"outcome", "record"}`, as the difftest's `session` writes them;
* `{"op": "close", "node"}`: `Registry.close`'s registration, or `{"refused": true}`;
* `{"op": "registry"}`: the registry, `registry/v0`.

A window's verdict is `checkWindowChunked`'s, which is `checkWindow`'s on the window's openings: any frame that doesn't
decode leaves it none, as the driver's Python does. Openings are checked in the clear.
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

private def doneLine (id : Nat) (v : Verdict) : String :=
  "{\"accepted\":" ++ (if v == .accept then "true" else "false") ++ ",\"done\":" ++ toString id ++ "}"

/-- Window t's openings checked on a thread of its own, `emit`ting the done line once its verdict is known. -/
def spawnCheck (chunk : Nat) (s : Session.State) (w : WindowRecord) (drawn : Array (Nat × Nat))
    (bodies : Array ByteArray) (id : Nat) (emit : String → IO Unit) :
    BaseIO (Task (Except IO.Error (Verdict × Nat × Nat))) :=
  IO.asTask (prio := .dedicated) do
    let t0 ← IO.monoNanosNow
    let openings := ((bodies.mapM Wire.decodeOpenings).map (·.flatten)).getD #[]
    let v := checkWindowChunked chunk s.reg s.digest s.appends w drawn openings
    emit (doneLine id v)
    let t1 ← IO.monoNanosNow
    return (v, openings.size, t1 - t0)

private def runOf (r : Registry) (node : ByteArray) (k : Nat) : Option Registry.Run :=
  (r.live.find? (·.node == node)).bind (·.sessions[k]?)

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
  | "register" =>
    let some kind := Kind.ofName? (← ofV (J.str req "receipt" "request"))
      | throwThe String "a receipt kind is sketch-sha512 or block-sha512"
    let (reg?, r) ← ofV (srv.registry.register srv.source (← ofV (bytesOf req "node")) (← ofV (nat req "blocks"))
      (← ofV (bytesOf req "payload_root")) (← ofV (Difftest.codecOf (← ofV (J.get req "codec" "request")))) kind false
      (← ofV (J.str req "layout" "request")) (← ofV (nat req "rows")) (← ofV (bytesOf req "nonce")))
    let out := match reg? with
      | some reg => Json.mkObj [("codec_seed", .str (hexOf reg.codecSeed)), ("digest", .str (hexOf reg.digest))]
      | none => refusedJ
    pure (out, { srv with registry := r })
  | "session" =>
    let node ← ofV (bytesOf req "node")
    let (params, coins, appendCoins, notBeforeNs) ← ofV (Difftest.sessionOf req)
    let index ← match ← ofV (J.get req "index" "request") with
      | .null => pure none
      | v => some <$> ofV (J.asInt v "index")
    let some reg := (srv.registry.live.find? (·.node == node)).bind (·.drawn) | pure (refusedJ, srv)
    match srv.registry.session reg params coins appendCoins notBeforeNs index with
    | some (k, r) => pure (Json.mkObj [("session", toJson k)], { srv with registry := r })
    | none => pure (refusedJ, srv)
  | "event" =>
    let node ← ofV (bytesOf req "node")
    let k ← ofV (nat req "session")
    let e ← ofV (Difftest.eventOf (← ofV (J.get req "event" "request")))
    let r ← match ← ofV (J.get req "coin" "request"), srv.registry.running node k with
      | .null, _ => pure srv.registry
      | c, some s => do
        let c ← ofV (unhex (← ofV (J.asStr c "coin")))
        if c.size ≠ Codec.COIN_BYTES then throwThe String "an append coin is 32 bytes"
        pure (srv.registry.update node k { s with appendCoins := s.appendCoins.push c })
      | _, none => pure srv.registry
    let (res, r) ← ofV (r.step node k e)
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
      | some (.running s) => Json.mkObj [("outcome", match s.outcome with | some o => .str o | none => .null),
          ("record", match s.summary with | some x => Difftest.summaryJ s.digest x | none => .null)]
      | some (.read o) => Json.mkObj [("outcome", .str o), ("record", .null)]
      | none => refusedJ
    pure (out, srv)
  | "close" =>
    let (c, r) ← ofV (srv.registry.close (← ofV (bytesOf req "node")))
    pure ((c.map Registry.Closed.toJson).getD refusedJ, { srv with registry := r })
  | "registry" => pure (srv.registry.toJson, srv)
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
