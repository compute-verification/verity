/-
# The ZK server's value service (live PROTOCOL.md §10.2): `core-firewall values`, `flock-firewall values`

The serving stack commits every value whose `b ‖ c` or root goes public through this service, which holds the ZK server's
salt key (`Salts.loadKey`), and the prover asks it for the salts its witness takes. Each connection is one request, its
line and then its body up to the end of what the client sends (it shuts its side down), answered whole:

* `commit J D₀ N₀ S₀ … D_{J−1} N_{J−1} S_{J−1}`, the body every table's digests, table-major, 64 bytes a leaf
  (`Salts.parseCommit`, `Salts.commitBody`). Only once it holds the whole request does the service draw the session `s`
  from the operating system, and the session joins the epoch (`Salts.Epoch`). It answers `ok S`, then each table's
  `ok ROOT TAG N` followed by its `N` leaves' `b ‖ c` (`Salts.commitAnswer`). A body of any other length, a salt beside
  each digest among them, is refused, and nothing is drawn for it. Where the service numbers registrations
  (`core-firewall values`), an epoch opens one session: a commit while the current epoch's session is not yet registered
  is refused, nothing of it answered, so every record counts one session and the stack chooses none to register. There
  the session is kept (`Record.epoch`) before its commit is answered, and the same request again is answered at it.
* `salts`, the body a refs file (`Salts.parseRefs`): `ok LEN` and the `LEN` bytes of its rows' salts, every table ref's
  tag checked (`Salts.saltsAnswer`).
* `register S`, no body: the registration of the epoch's session `S`, whole. It closes the epoch, so no session of it
  registers after, takes the next number on the ZK server's record (`Numbering`: `core-firewall values` appends the
  registration there, its record whole), and answers `ok M LEN` and the `LEN` bytes of its record (`Salts.recordOf`),
  `be64 n` first, `M` the number of sessions the epoch opened, 1. A session the current epoch did not open is refused,
  and so is every registration where the service has no record to number it on (`flock-firewall values`), or where its
  record can't take it (its epoch is closed all the same).

Anything else is answered `err WHY`. `answer` is the whole of it but the socket. `core-firewall values` keeps its epoch's
session in `DIR/epoch`, its owner's alone, and starts again in that epoch, so a restart opens none: one cut short before
its commit's answer answers the same request again, and one cut short after numbering the registration finds the record
holds it and drops it. `flock-firewall values`' epoch lives only in its process. Only the ZK server, under its own OS
user, starts the service (`zk-server-own-user`).
-/
import Flock.Firewall.Salts
import Flock.Witness
import Std.Async.TCP

namespace Flock.Firewall.Values

open Std.Net
open Std.Async
open Std.Async.TCP

/-- **How a registration is numbered** on the ZK server's record: given its record at each number, the record at the
number it took, appended to the ZK server's record in the same step, or why the record took none. -/
abbrev Numbering := (Nat → ByteArray) → IO (Except String ByteArray)

/-- **The ZK server's record**, where the service has one: how it numbers a registration (`number`), the file in which it
keeps its epoch's session (`epoch`, `Salts.Opened.bytes`), and whether the record holds a registration of a session. -/
structure Record where
  number : Numbering
  epoch : System.FilePath
  holds : ByteArray → IO Bool

/-- Writes `b` to `path` in one step, in a file its owner alone may read and write. -/
def keep (path : System.FilePath) (b : ByteArray) : IO Unit := do
  let tmp := path.addExtension "new"
  let h ← IO.FS.Handle.mk tmp .write
  IO.setAccessRights tmp { user := { read := true, write := true } }
  h.write b
  h.flush
  IO.FS.rename tmp path

/-- **The answer to one request** `req` (its line, then its body) under the key `k` and the epoch `epoch`, `draw` the
operating system's bytes, read for a commit's session only once the request is whole and well formed, and `record` the
ZK server's record, if the service has one: then an epoch opens one session, kept in `record.epoch` before its commit is
answered, and the same request again is answered at it. Each request changes the epoch in one step
(`IO.Ref.modifyGet`), whatever other requests run beside it, so of commits that run together into an epoch with no
session, one opens it. -/
def answer (k : ByteArray) (draw : Nat → IO ByteArray) (epoch : IO.Ref Salts.Epoch) (record : Option Record)
    (req : ByteArray) : IO ByteArray := do
  let line : String → ByteArray := fun s => (s ++ "\n").toUTF8
  let (head, body) := match findByte req 10 with
    | some nl => ((String.fromUTF8? (req.extract 0 nl)).getD "?", req.extract (nl + 1) req.size)
    | none => ((String.fromUTF8? req).getD "?", .empty)
  match head.splitOn " " with
  | "commit" :: ws =>
    match Salts.parseCommit ws with
    | .error e => return line s!"err {e}"
    | .ok ts =>
      if body.size != Salts.commitBody ts then
        return line s!"err a commit body of {body.size} bytes, where its tables' digests are {Salts.commitBody ts} (64 a leaf, and nothing beside them)"
      let answered := fun (s : ByteArray) (ans : Array (ByteArray × ByteArray × ByteArray)) => ans.foldl
        (fun acc (root, tag, bcs) => acc ++ line s!"ok {hexOf root} {hexOf tag} {bcs.size / Salts.BC_BYTES}" ++ bcs)
        (line s!"ok {hexOf s}")
      let s ← draw Salts.SESSION_BYTES
      let ans := Salts.commitAnswer k s ts body
      let opened : Salts.Opened := { s, tables := (Array.range ts.size).map (fun j => (ts[j]!, ans[j]!.1)), req := Sha512.hash req }
      let got ← epoch.modifyGet fun e => match record, e.opened[0]? with
        | none, _ => (some none, { e with opened := e.opened.push opened })
        | some _, none => if e.closing then (none, e) else (some none, { e with opened := #[opened] })
        | some _, some o => (if o.req == opened.req then some (some o.s) else none, e)
      match got with
      | none => return line "err the current epoch's session is not yet registered: an epoch opens one commit session (`register S` registers it, and the next commit opens the next epoch's)"
      | some (some s') => return answered s' (Salts.commitAnswer k s' ts body)
      | some none =>
        if let some r := record then
          try keep r.epoch (opened.bytes k)
          catch e =>
            epoch.set {}
            return line s!"err the ZK server kept none of the epoch's session ({e}), so it answers none of it"
        return answered s ans
  | ["salts"] =>
    match Salts.saltsAnswer k body with
    | .ok b => return line s!"ok {b.size}" ++ b
    | .error e => return line s!"err {e}"
  | ["register", sh] =>
    if !body.isEmpty then return line s!"err a register request with a body of {body.size} bytes, where it takes none"
    if sh.length != 2 * Salts.SESSION_BYTES then
      return line s!"err a session `{String.ofList (sh.toList.take 200)}` that is not {2 * Salts.SESSION_BYTES} hex digits"
    let some record := record
      | return line "err a registration is numbered on the ZK server's record: `core-firewall values --state DIR`"
    match unhex sh with
    | .error e => return line s!"err {e}"
    | .ok s =>
      let got ← epoch.modifyGet fun e => match e.opened.find? (·.s == s) with
        | some o => (some (e.opened.size, o), { closing := true })
        | none => (none, e)
      match got with
      | some (m, o) =>
        let r ← record.number fun n => Salts.recordOf k n m o
        try IO.FS.removeFile record.epoch catch _ => pure ()
        epoch.set {}
        match r with
        | .ok r => return line s!"ok {m} {r.size}" ++ r
        | .error e => return line s!"err the ZK server's record took no registration: {e}"
      | none => return line s!"err the session {sh} is none the current epoch opened: a session registers once, in the epoch that opened it"
  | _ => return line s!"err a request `{String.ofList (head.toList.take 200)}`"

/-- Everything `c` sends until it shuts its side down. -/
def recvAll (c : Socket.Client) : IO ByteArray := do
  let mut acc := ByteArray.empty
  repeat
    match ← (c.recv? 65536).block with
    | none => break
    | some b =>
      if b.isEmpty then break
      acc := acc ++ b
  return acc

/-- The value service: it reads its salt key from `--salt-key FILE`, creating it where it does not exist
(`Salts.loadKey`), listens on `127.0.0.1:PORT` (`--listen`, by default a port the system picks), prints
`VALUES 127.0.0.1:PORT` once it does, and serves each connection as one request (`answer`, its registrations numbered on
`record`) on its own thread until its standard input closes. It starts in the epoch `record` kept, unless the record
holds its session. -/
def serve (usage : String) (args : List String) (record : Option Record) : IO UInt32 := do
  let some kf := Witness.argOf args "--salt-key" | IO.eprintln usage; return 2
  let some listen := ((Witness.argOf args "--listen").getD "0").toNat? | IO.eprintln usage; return 2
  let k ← match ← Salts.loadKey kf true with
    | .ok k => pure k
    | .error e => IO.eprintln e; return 2
  let epoch ← IO.mkRef ({} : Salts.Epoch)
  if let some r := record then
    if ← r.epoch.pathExists then
      match Salts.Opened.parse k (← IO.FS.readBinFile r.epoch) with
      | .error e => IO.eprintln s!"{r.epoch}: {e}"; return 2
      | .ok o => if ← r.holds o.s then IO.FS.removeFile r.epoch else epoch.set { opened := #[o] }
  let srv ← Socket.Server.mk
  srv.bind (.v4 { addr := IPv4Addr.ofParts 127 0 0 1, port := listen.toUInt16 })
  srv.listen 128
  IO.println s!"VALUES 127.0.0.1:{(← srv.getSockName).port}"
  (← IO.getStdout).flush
  let _ ← IO.asTask (prio := .dedicated) do
    repeat
      let c ← srv.accept.block
      let _ ← IO.asTask (prio := .dedicated) do
        try
          let a ← answer k (fun n => IO.getRandomBytes n.toUSize) epoch record (← recvAll c)
          (c.sendAll #[a]).block
        catch e => IO.eprintln s!"values: {e}"
        try (c.shutdown).block catch _ => pure ()
  let stdin ← IO.getStdin
  repeat
    if (← stdin.getLine).isEmpty then break
  IO.Process.exit 0

/-- `flock-firewall values --salt-key FILE [--listen PORT]`: the value service with no record of the ZK server's, so it
commits and answers salts but registers nothing (`core-firewall values --state DIR` registers). -/
def cmd (args : List String) : IO UInt32 :=
  serve "usage: flock-firewall values --salt-key FILE [--listen PORT]" args none

end Flock.Firewall.Values
