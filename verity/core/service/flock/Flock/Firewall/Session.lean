/-
# The firewall's session loop, in Lean (live PROTOCOL.md §9, §10)

A firewalled prover's sessions go through one `flock-firewall session` process, which owns the auditor's connection. The
prover's workers connect to it as they would to the auditor (`TcpTransport`, the wire's frames `u32le(len) ‖ bytes`), and
each connection is one session, relayed on a connection of its own to the auditor. What crosses is decided here:

* `Register` is relayed, and its answer (`Resp::Draw`) is the session's draw (`State.answer`).
* `Hello`: the prover's text is dropped, and what leaves is the Lean `Hello` at that draw (`Hello.msg`), at a nonce this
  process draws from the operating system (`State.sendHello`). The coin check reads the session's tree, link points and
  nonce from the setup's values (`helloOf`), never from the text.
* `Hello`'s answer must be the coin tree's root and session key (`Coins.Session.start`), under a tree whose streams are
  the session's (`streamsOk`), and it is relayed as it is.
* `Commit` and `Round` leave only as the schedule's next opening (`State.request`, `reqAt`: the session stream's at
  `Commit`, its count `Hello`'s link points', then each rep stream's rounds in order at the counts `Zk.verifyRep` draws,
  `planOf`), never at a place the prover names. The coins answered must open the committed tree there
  (`Coins.Session.open`), and only the coins, stripped of their opening, reach the prover (`Coins.frame`).
* `Finish` leaves only once the schedule's last opening has (`State.request`). What it carries is `Firewall::finish`'s
  (ZKFIN's), relayed as it is.
* Every other frame is relayed as it is. For now those are the rounds and roots the firewall's own Lean processes
  released (`frame`, `link`), and `Proof`, which `Firewall::finish` decides.

A refusal answers the prover `Resp::Refused(WHY)`, and every coin after it is refused too, without asking the auditor: no
coin after a refused one reaches the prover. `kind` and `State` are pure; `cmd` only
moves frames.
-/
import Flock.Firewall.SessionPlan
import Std.Async.TCP
import Std.Async.DNS

namespace Flock.Firewall.Session

open Lean (Json)
open Std.Net
open Std.Async
open Std.Async.TCP

/-- `u32le(v)`. -/
def u32le (v : Nat) : ByteArray := ⟨(Coins.u32be v).reverse.toArray⟩

/-- Byte `i` of `b`. -/
def byte? (b : ByteArray) (i : Nat) : Option UInt8 := if i < b.size then some (b.get! i) else none

/-- The `u32le` at byte `o` of `b`. -/
def getU32 (b : ByteArray) (o : Nat) : Option Nat :=
  if o + 4 ≤ b.size then
    some ((b.get! o).toNat + 2 ^ 8 * (b.get! (o + 1)).toNat + 2 ^ 16 * (b.get! (o + 2)).toNat +
      2 ^ 24 * (b.get! (o + 3)).toNat)
  else none

/-- The length-prefixed bytes at byte `o` of `b`, and the byte after them. -/
def getBytes (b : ByteArray) (o : Nat) : Option (ByteArray × Nat) := do
  let n ← getU32 b o
  if o + 4 + n ≤ b.size then some (b.extract (o + 4) (o + 4 + n), o + 4 + n) else none

/-- `Resp::Refused(why)`'s frame: tag `3`, `why`'s UTF-8 length-prefixed. -/
def refused (why : String) : ByteArray := ByteArray.mk #[3] ++ u32le why.utf8ByteSize ++ why.toUTF8

/-- What the loop reads of a request frame (`Req::encode`): `Register` (tag `10`), `Hello` (`1`), `Commit` (`7`), a
`Round` (`4`: its stream, its content, the `u32le` count of coins it squeezes), `Finish` (`6`), and anything else. -/
inductive Kind where
  | register
  | hello
  | commit
  | round (stream : String) (n : Nat)
  | finish
  | other
  deriving BEq, Repr

/-- The kind of the request frame `f`. -/
def kind (f : ByteArray) : Except String Kind :=
  match byte? f 0 with
  | none => .error "an empty request frame"
  | some 10 => .ok .register
  | some 1 => .ok .hello
  | some 7 => .ok .commit
  | some 6 => .ok .finish
  | some 4 =>
    match getBytes f 1 with
    | none => .error "a Round frame without its stream"
    | some (s, o) => match String.fromUTF8? s, getBytes f o with
      | some stream, some (_, o') => match getU32 f o' with
        | some n => .ok (.round stream n)
        | none => .error "a Round frame without its count"
      | none, _ => .error "a Round frame whose stream is not UTF-8"
      | _, none => .error "a Round frame without its content"
  | some _ => .ok .other

/-- A session as the loop has seen it: the auditor's draw, whether the `Hello` left, what the coin check reads of it
(`helloOf`) and its statement's openings (`planOf`), the coin tree once `Hello`'s answer checked, and the refusal that
stopped it. -/
structure State where
  draw : Option Json := none
  sent : Bool := false
  hello : Option Coins.Hello := none
  plan : Array (String × Nat) := #[]
  session : Option Coins.Session := none
  stop : Option String := none

/-- The session stopped by `why`, and what the prover receives: `Resp::Refused(why)`. -/
def State.refuse (S : State) (why : String) : State × ByteArray :=
  ({ S with stop := some why }, refused why)

/-- **Whether `Finish` leaves**: only once the schedule's last opening has (the session stream's and `planOf`'s), so that
it is the session's round `planOf`'s size `+ 2`; `none` where it does, else the refusal the prover receives in its place
and the session after. -/
def State.finish (S : State) : Option (State × ByteArray) :=
  match S.stop, S.session with
  | some why, _ => some (S, refused why)
  | none, none => some (S.refuse s!"{Coins.STOP}a Finish before the step-0 coin commitment")
  | none, some C =>
    if C.opened.length = S.plan.size + 1 then none
    else some (S.refuse
      s!"{Coins.STOP}a Finish after {C.opened.length} openings, where the schedule's are {S.plan.size + 1}")

/-- **Whether a request of kind `k` leaves**: `none` where it does, else the refusal the prover receives in its place and
the session after. A `Commit` or `Round` leaves only as the schedule's next opening (`reqAt`): no coin is asked for after
a refusal, before `Hello`'s answer, or at a place the prover picks. `Finish` leaves only after the last (`State.finish`). -/
def State.request (S : State) (k : Kind) : Option (State × ByteArray) :=
  if k == .finish then S.finish else
  let opening : Option (String × Nat) := match k with
    | .commit => some (Coins.SESSION, 0)
    | .round stream n => some (stream, n)
    | _ => none
  match opening, S.stop, S.session with
  | none, _, _ => none
  | some _, some why, _ => some (S, refused why)
  | some _, none, none => some (S.refuse s!"{Coins.STOP}coins before the step-0 coin commitment")
  | some (stream, n), none, some C =>
    let j := C.opened.length
    let (st, want) := reqAt S.plan C.hello.links j
    if j = 0 then
      if k == .commit then none
      else some (S.refuse s!"{Coins.STOP}a round of {stream} before the step-0 coin commitment")
    else if k == .commit then some (S.refuse s!"{Coins.STOP}a second step-0 coin commitment")
    else if j > S.plan.size then some (S.refuse s!"{Coins.STOP}a round of {stream} past the schedule's {S.plan.size}")
    else if stream == st && n == want then none
    else some (S.refuse s!"{Coins.STOP}opening {j}: {n} coins of {stream}, where the schedule's are {want} of {st}")

/-- The coins the auditor answered `a` to the schedule's next opening (`reqAt`), once they open the committed tree; the
session with the round counted. -/
def State.coins (S : State) (a : ByteArray) : State × ByteArray :=
  match S.stop, S.session with
  | some why, _ => (S, refused why)
  | none, none => S.refuse s!"{Coins.STOP}coins before the step-0 coin commitment"
  | none, some C =>
    let (stream, n) := reqAt S.plan C.hello.links C.opened.length
    match C.open stream n a.data.toList with
    | .ok (C', c) => ({ S with session := some C' }, ⟨(Coins.frame (c.length / 16) c).toArray⟩)
    | .error e => S.refuse e

/-- **What the prover receives** for the auditor's answer `a` to a request of kind `k`, and the session after: the draw
taken from `Register`'s answer, `Hello`'s answer checked, a round's coins opened (`State.coins`); anything else as it is. -/
def State.answer (S : State) (k : Kind) (a : ByteArray) : State × ByteArray :=
  match k, byte? a 0 with
  | .register, some 6 => match getBytes a 1 with
    | some (d, _) => match String.fromUTF8? d with
      | some text => match Hello.drawOf text with
        | .ok draw => ({ S with draw }, a)
        | .error e => S.refuse s!"{Coins.STOP}{e}"
      | none => S.refuse s!"{Coins.STOP}the auditor's draw is not UTF-8"
    | none => S.refuse s!"{Coins.STOP}the auditor's draw is not a string"
  | .hello, _ => match S.stop, S.hello with
    | some why, _ => (S, refused why)
    | none, none =>
      if S.sent then
        S.refuse s!"{Coins.STOP}the session has no coin tree of its own streams (session, then each table's rep0 and rep1, distinct)"
      else S.refuse s!"{Coins.STOP}Hello's answer before the Hello"
    | none, some h => match Coins.Session.start h a.data.toList with
      | .ok C => ({ S with session := some C }, a)
      | .error e => S.refuse e
  | .commit, some 2 => S.coins a
  | .round _ _, some 2 => S.coins a
  | _, _ => (S, a)

/-- **The `Hello` that leaves** at the set-up statement `s` and the nonce `nonce` (`Hello.msg`), and the session after:
what the coin check reads of it (`helloOf`, the setup's values) and the schedule's openings (`planOf`); or the refusal
the prover receives in its place, where the statement does not set up or the verifier's `Hello` is not JSON. -/
def State.sendHello (S : State) (s : V (SessionSpec × Array Stmt)) (nonce : ByteArray) :
    Except (State × ByteArray) (State × ByteArray) :=
  match S.stop, s with
  | some why, _ => .error (S, refused why)
  | none, .error e => .error (S.refuse s!"{Coins.STOP}the Lean Hello: setup: {e}")
  | none, .ok (spec, sts) =>
    match Hello.msg spec nonce with
    | [] => .error (S.refuse s!"{Coins.STOP}the Lean Hello: the verifier's Hello is not JSON")
    | m => .ok ({ S with sent := true, hello := helloOf spec sts nonce, plan := planOf spec sts }, ⟨m.toArray⟩)

/-- `n` bytes from the socket `c`, or `none` where it closes first. -/
def recvExact (c : Socket.Client) (n : Nat) : IO (Option ByteArray) := do
  let mut acc := ByteArray.empty
  while acc.size < n do
    match ← (c.recv? (n - acc.size).toUInt64).block with
    | none => return none
    | some b =>
      if b.isEmpty then return none
      acc := acc ++ b
  return some acc

/-- One frame from `c` (`read_frame`), or `none` where the connection closes. -/
def recvFrame (c : Socket.Client) : IO (Option ByteArray) := do
  let some h ← recvExact c 4 | return none
  let some n := getU32 h 0 | return none
  recvExact c n

/-- The frame `f` to `c` (`write_frame`). -/
def sendFrame (c : Socket.Client) (f : ByteArray) : IO Unit :=
  (c.sendAll #[u32le f.size, f]).block

/-- The address `HOST:PORT` names, resolved. -/
def resolve (addr : String) : IO SocketAddress := do
  let (host, port) := match addr.splitOn ":" with
    | [h, p] => (h, p)
    | _ => ("", "")
  let some p := port.toNat? | throw (IO.userError s!"an address `{addr}` that is not HOST:PORT")
  let ips ← (DNS.getAddrInfo host port (some .ipv4)).block
  let some ip := ips[0]? | throw (IO.userError s!"{host} resolves to no IPv4 address")
  match ip with
  | .v4 a => pure (.v4 { addr := a, port := p.toUInt16 })
  | .v6 a => pure (.v6 { addr := a, port := p.toUInt16 })

/-- **One session**: the prover's connection `p`, relayed on a connection of its own to the auditor at `auditor` until
either side closes. Each request is read whole, decided (`kind`, `State.sendHello`, `State.answer`) and only then sent. -/
def serve (S : Hello.Setup) (specs : Hello.Specs) (auditor : SocketAddress) (p : Socket.Client) : IO Unit := do
  let a ← Socket.Client.mk
  (a.connect auditor).block
  a.noDelay
  let mut st : State := {}
  repeat
    let some f ← recvFrame p | break
    match kind f with
    | .error e =>
      sendFrame p (refused s!"{Coins.STOP}{e}")
      break
    | .ok .hello =>
      let nonce ← IO.getRandomBytes Hello.NONCE_BYTES.toUSize
      match st.sendHello (← S.setupAt specs st.draw) nonce with
      | .error (st', out) =>
        st := st'
        sendFrame p out
      | .ok (st', h) =>
        st := st'
        sendFrame a h
        let some r ← recvFrame a | break
        let (st', out) := st.answer .hello r
        st := st'
        sendFrame p out
    | .ok k =>
      if let some (st', out) := st.request k then
        st := st'
        sendFrame p out
        continue
      sendFrame a f
      let some r ← recvFrame a | break
      let (st', out) := st.answer k r
      st := st'
      sendFrame p out
  (a.shutdown).block
  (p.shutdown).block

/-- `flock-firewall session --verifier HOST:PORT [--listen PORT] [--statement ID] --circuit F --public F …` (the statement
flags `Hello.FLAGS`): the firewall's session loop. It listens on `127.0.0.1:PORT` (by default a port the system picks),
prints `LISTENING 127.0.0.1:PORT` once it does, and serves each connection as one session (`serve`), each on its own
thread, every session's `Hello` at one statement set up once a draw (`Hello.Setup.specAt`). It runs until its standard
input closes. `flock-firewall session --schedule …` instead reads a draw's JSON (`null` for none) on its standard input
and prints the openings after the session stream's at it (`planOf`), `ok N` and a line `STREAM COINS` each, or
`refused WHY`. -/
def cmd (args : List String) : IO UInt32 := do
  let usage := s!"usage: flock-firewall session (--verifier HOST:PORT [--listen PORT] | --schedule) {Hello.FLAGS}"
  let S ← match ← Hello.Setup.ofArgs usage args with
    | .ok S => pure S
    | .error e => IO.eprintln e; return 2
  if args.contains "--schedule" then
    let stdin ← IO.getStdin
    let mut text := ""
    repeat
      let line ← stdin.getLine
      if line.isEmpty then break
      text := text ++ line
    let p := (Hello.drawOf text).bind fun draw => (S.setup draw).map fun (spec, sts) => planOf spec sts
    match p with
    | .error e => IO.println s!"refused {e}"
    | .ok p =>
      IO.println s!"ok {p.size}"
      for (stream, n) in p do IO.println s!"{stream} {n}"
    return 0
  let some verifier := Witness.argOf args "--verifier" | IO.eprintln usage; return 2
  let some listen := ((Witness.argOf args "--listen").getD "0").toNat? | IO.eprintln usage; return 2
  let auditor ← resolve verifier
  let specs : Hello.Specs ← IO.mkRef #[]
  let srv ← Socket.Server.mk
  srv.bind (.v4 { addr := IPv4Addr.ofParts 127 0 0 1, port := listen.toUInt16 })
  srv.listen 128
  IO.println s!"LISTENING 127.0.0.1:{(← srv.getSockName).port}"
  (← IO.getStdout).flush
  let _ ← IO.asTask (prio := .dedicated) do
    repeat
      let p ← srv.accept.block
      let _ ← IO.asTask (prio := .dedicated) do
        try serve S specs auditor p
        catch e => IO.eprintln s!"session: {e}"
  let stdin ← IO.getStdin
  repeat
    if (← stdin.getLine).isEmpty then break
  IO.Process.exit 0

end Flock.Firewall.Session
