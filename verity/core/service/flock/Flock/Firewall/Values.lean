/-
# The ZK server's value service (live PROTOCOL.md §10.2): `flock-firewall values`

The serving stack commits every value whose `b ‖ c` or root goes public through this service, which holds the ZK server's
salt key (`Salts.loadKey`), and the prover asks it for the salts its witness takes. Each connection is one request, its
line and then its body up to the end of what the client sends (it shuts its side down), answered whole:

* `commit J D₀ N₀ S₀ … D_{J−1} N_{J−1} S_{J−1}`, the body every table's digests, table-major, 64 bytes a leaf
  (`Salts.parseCommit`, `Salts.commitBody`). Only once it holds the whole request does the service draw the session `s`
  from the operating system, and the session joins the epoch (`Salts.Epoch`). It answers `ok S`, then each table's
  `ok ROOT TAG N` followed by its `N` leaves' `b ‖ c` (`Salts.commitAnswer`). A body of any other length, a salt beside
  each digest among them, is refused, and nothing is drawn for it.
* `salts`, the body a refs file (`Salts.parseRefs`): `ok LEN` and the `LEN` bytes of its rows' salts, every table ref's
  tag checked (`Salts.saltsAnswer`).
* `register S`, no body: the registration of the epoch's session `S`, whole. It answers `ok M LEN` and the `LEN` bytes of
  its record (`Salts.recordOf`), `M` the number of sessions the epoch opened, and opens a new epoch, so no session of the
  closed one registers. A session the current epoch did not open is refused.

Anything else is answered `err WHY`. `answer` is the whole of it but the socket. The epoch lives only in the process:
one started again opens a new epoch, in which no earlier session registers.
-/
module

public import Flock.Firewall.Salts
public import Flock.Witness
public import Std.Async.TCP

@[expose] public section

namespace Flock.Firewall.Values

open Std.Net
open Std.Async
open Std.Async.TCP

/-- **The answer to one request** `req` (its line, then its body) under the key `k` and the epoch `epoch`, `draw` the
operating system's bytes, read for a commit's session only once the request is whole and well formed. Each request
changes the epoch in one step (`IO.Ref.modify`, `IO.Ref.modifyGet`), whatever other requests run beside it. -/
def answer (k : ByteArray) (draw : Nat → IO ByteArray) (epoch : IO.Ref Salts.Epoch) (req : ByteArray) : IO ByteArray := do
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
      let ans := Salts.commitAnswer k s ts body
      let opened : Salts.Opened := { s, tables := (Array.range ts.size).map fun j => (ts[j]!, ans[j]!.1) }
      epoch.modify fun e => { e with opened := e.opened.push opened }
      return ans.foldl
        (fun acc (root, tag, bcs) => acc ++ line s!"ok {hexOf root} {hexOf tag} {bcs.size / Salts.BC_BYTES}" ++ bcs)
        (line s!"ok {hexOf s}")
  | ["salts"] =>
    match Salts.saltsAnswer k body with
    | .ok b => return line s!"ok {b.size}" ++ b
    | .error e => return line s!"err {e}"
  | ["register", sh] =>
    if !body.isEmpty then return line s!"err a register request with a body of {body.size} bytes, where it takes none"
    if sh.length != 2 * Salts.SESSION_BYTES then
      return line s!"err a session `{String.ofList (sh.toList.take 200)}` that is not {2 * Salts.SESSION_BYTES} hex digits"
    match unhex sh with
    | .error e => return line s!"err {e}"
    | .ok s =>
      let got ← epoch.modifyGet fun e => match e.opened.find? (·.s == s) with
        | some o => (some (e.opened.size, o), {})
        | none => (none, e)
      match got with
      | some (m, o) =>
        let r := Salts.recordOf k m o
        return line s!"ok {m} {r.size}" ++ r
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
  let epoch ← IO.mkRef ({} : Salts.Epoch)
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
          let a ← answer k (fun n => IO.getRandomBytes n.toUSize) epoch (← recvAll c)
          (c.sendAll #[a]).block
        catch e => IO.eprintln s!"values: {e}"
        try (c.shutdown).block catch _ => pure ()
  let stdin ← IO.getStdin
  repeat
    if (← stdin.getLine).isEmpty then break
  IO.Process.exit 0

end Flock.Firewall.Values
