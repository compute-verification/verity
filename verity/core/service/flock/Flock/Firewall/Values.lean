/-
# The ZK server's value service (live PROTOCOL.md §10.2): `flock-firewall values`

The serving stack commits every value whose `b ‖ c` or root goes public through this service, which holds the ZK server's
salt key (`Salts.loadKey`), and the prover asks it for the salts its witness takes. Each connection is one request, its
line and then its body up to the end of what the client sends (it shuts its side down), answered whole:

* `commit J D₀ N₀ S₀ … D_{J−1} N_{J−1} S_{J−1}`, the body every table's digests, table-major, 64 bytes a leaf
  (`Salts.parseCommit`, `Salts.commitBody`). Only once it holds the whole request does the service draw the session `s`
  from the operating system. It answers `ok S`, then each table's `ok ROOT TAG N` followed by its `N` leaves' `b ‖ c`
  (`Salts.commitAnswer`). A body of any other length, a salt beside each digest among them, is refused, and nothing is
  drawn for it.
* `salts`, the body a refs file (`Salts.parseRefs`): `ok LEN` and the `LEN` bytes of its rows' salts, every table ref's
  tag checked (`Salts.saltsAnswer`).

Anything else is answered `err WHY`. `answer` is the whole of it but the socket.
-/
import Flock.Firewall.Salts
import Flock.Witness
import Std.Async.TCP

namespace Flock.Firewall.Values

open Std.Net
open Std.Async
open Std.Async.TCP

/-- **The answer to one request** `req` (its line, then its body) under the key `k`, `draw` the operating system's bytes,
read for a commit's session only once the request is whole and well formed. -/
def answer (k : ByteArray) (draw : Nat → IO ByteArray) (req : ByteArray) : IO ByteArray := do
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
      let s ← draw Salts.SESSION_BYTES
      return (Salts.commitAnswer k s ts body).foldl
        (fun acc (root, tag, bcs) => acc ++ line s!"ok {hexOf root} {hexOf tag} {bcs.size / Salts.BC_BYTES}" ++ bcs)
        (line s!"ok {hexOf s}")
  | ["salts"] =>
    match Salts.saltsAnswer k body with
    | .ok b => return line s!"ok {b.size}" ++ b
    | .error e => return line s!"err {e}"
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

/-- `flock-firewall values --salt-key FILE [--listen PORT]`: the ZK server's value service. It reads its salt key from
`FILE`, creating it where it does not exist (`Salts.loadKey`), listens on `127.0.0.1:PORT` (by default a port the system
picks), prints `VALUES 127.0.0.1:PORT` once it does, and serves each connection as one request (`answer`) on its own
thread until its standard input closes. -/
def cmd (args : List String) : IO UInt32 := do
  let usage := "usage: flock-firewall values --salt-key FILE [--listen PORT]"
  let some kf := Witness.argOf args "--salt-key" | IO.eprintln usage; return 2
  let some listen := ((Witness.argOf args "--listen").getD "0").toNat? | IO.eprintln usage; return 2
  let k ← match ← Salts.loadKey kf true with
    | .ok k => pure k
    | .error e => IO.eprintln e; return 2
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
          let a ← answer k (fun n => IO.getRandomBytes n.toUSize) (← recvAll c)
          (c.sendAll #[a]).block
        catch e => IO.eprintln s!"values: {e}"
        try (c.shutdown).block catch _ => pure ()
  let stdin ← IO.getStdin
  repeat
    if (← stdin.getLine).isEmpty then break
  IO.Process.exit 0

end Flock.Firewall.Values
