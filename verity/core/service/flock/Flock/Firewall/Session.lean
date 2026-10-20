/-
# The firewall's session loop, in Lean (live PROTOCOL.md §9, §10)

A firewalled prover's sessions go through one `flock-firewall session` process, which owns the auditor's connection. The
prover's workers connect to it as they would to the auditor (`TcpTransport`, the wire's frames `u32le(len) ‖ bytes`), and
each connection is one session, relayed on a connection of its own to the auditor. What crosses is decided here:

* `Register` is relayed, and its answer (`Resp::Draw`) is the session's draw (`State.answer`).
* `Hello`: the prover's text is dropped, and what leaves is the Lean `Hello` at that draw (`Hello.textAt`), at a nonce
  this process draws from the operating system (`State.sendHello`).
* `Hello`'s answer must be the coin tree's root and session key (`Coins.Session.start`), and it is relayed as it is.
* `Commit` and `Round`: the coins answered must open the committed tree (`Coins.Session.open`; the link points' count is
  `Hello`'s), and only the coins, stripped of their opening, reach the prover (`Coins.frame`).
* Every other frame is relayed as it is. For now those are the rounds and roots the firewall's own Lean processes
  released (`frame`, `link`), and `Proof` and `Finish`, which `Firewall::finish` decides (ZKFIN's).

A refusal answers the prover `Resp::Refused(WHY)`, and every coin after it is refused too, as `coin_tree::Checked` does
with the Lean check's verdict: no coin after a refused one reaches the prover. `kind` and `State` are pure; `cmd` only
moves frames.
-/
import Flock.Firewall.Hello
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
`Round` (`4`: its stream, its content, the `u32le` count of coins it squeezes), and anything else. -/
inductive Kind where
  | register
  | hello
  | commit
  | round (stream : String) (n : Nat)
  | other
  deriving BEq, Repr

/-- The kind of the request frame `f`. -/
def kind (f : ByteArray) : Except String Kind :=
  match byte? f 0 with
  | none => .error "an empty request frame"
  | some 10 => .ok .register
  | some 1 => .ok .hello
  | some 7 => .ok .commit
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

/-- A session as the loop has seen it: the auditor's draw, the `Hello` that left, the coin tree once `Hello`'s answer
checked, and the refusal that stopped it. -/
structure State where
  draw : Option Json := none
  hello : Option Coins.Hello := none
  session : Option Coins.Session := none
  stop : Option String := none

/-- The session stopped by `why`, and what the prover receives: `Resp::Refused(why)`. -/
def State.refuse (S : State) (why : String) : State × ByteArray :=
  ({ S with stop := some why }, refused why)

/-- The coins the auditor answered `a`, of `n` coins of `stream`'s next round, once they open the committed tree; the
session with the round counted. -/
def State.coins (S : State) (stream : String) (n : Nat) (a : ByteArray) : State × ByteArray :=
  match S.stop, S.session with
  | some why, _ => (S, refused why)
  | none, none => S.refuse s!"{Coins.STOP}coins before the step-0 coin commitment"
  | none, some C => match C.open stream n a.data.toList with
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
        | .error e => S.refuse s!"{Coins.STOP}the auditor's {e}"
      | none => S.refuse s!"{Coins.STOP}the auditor's draw is not UTF-8"
    | none => S.refuse s!"{Coins.STOP}the auditor's draw is not a string"
  | .hello, _ => match S.stop, S.hello with
    | some why, _ => (S, refused why)
    | none, none => S.refuse s!"{Coins.STOP}Hello's answer before the Hello"
    | none, some h => match Coins.Session.start h a.data.toList with
      | .ok C => ({ S with session := some C }, a)
      | .error e => S.refuse e
  | .commit, some 2 => S.coins Coins.SESSION 0 a
  | .round stream n, some 2 => S.coins stream n a
  | _, _ => (S, a)

/-- **The `Hello` that leaves** at the set-up statement `spec` and the nonce `nonce` (`Hello.textAt`), and the session
after, which the coin tree's checks read; or the refusal the prover receives in its place. -/
def State.sendHello (S : State) (spec : V SessionSpec) (nonce : ByteArray) : Except (State × ByteArray) (State × ByteArray) :=
  match S.stop with
  | some why => .error (S, refused why)
  | none => match Hello.textAt spec nonce with
    | .error e => .error (S.refuse s!"{Coins.STOP}the Lean Hello: {e}")
    | .ok text => match Coins.Hello.parse text with
      | .ok h => .ok ({ S with hello := some h }, ⟨(Hello.frame text).toArray⟩)
      | .error e => .error (S.refuse e)

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
      match st.sendHello (← S.specAt specs st.draw) nonce with
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
input closes. -/
def cmd (args : List String) : IO UInt32 := do
  let usage := s!"usage: flock-firewall session --verifier HOST:PORT [--listen PORT] {Hello.FLAGS}"
  let some verifier := Witness.argOf args "--verifier" | IO.eprintln usage; return 2
  let some listen := ((Witness.argOf args "--listen").getD "0").toNat? | IO.eprintln usage; return 2
  let S ← match ← Hello.Setup.ofArgs usage args with
    | .ok S => pure S
    | .error e => IO.eprintln e; return 2
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
  return 0

end Flock.Firewall.Session
