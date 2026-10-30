/-
# The firewall's session loop, in Lean (live PROTOCOL.md §9, §10)

A firewalled prover's sessions go through one `flock-firewall session` process, which owns the auditor's connection. The
prover's workers connect to it as they would to the auditor (`TcpTransport`, the wire's frames `u32le(len) ‖ bytes`), and
each connection is one session, relayed on a connection of its own to the auditor. What crosses is decided here, and
every frame that leaves is the loop's own encoding of what it decided (`roundFrame`, `commitFrame`, `State.wire`), never
the prover's bytes:

* `Register` leaves as `REGISTER`, and its answer (`Resp::Draw`) is the session's draw (`State.answer`).
* `Hello`: the prover's text is dropped, and what leaves is the Lean `Hello` at that draw (`Hello.msg`), at a nonce this
  process draws from the operating system (`State.sendHello`). The coin check reads the session's tree, link points and
  nonce from the setup's values (`helloOf`), never from the text. A setup whose sizes are past what the loop's frames
  read (`sizesOk`) is refused there: no `Hello` leaves, and the session is stopped.
* `Hello`'s answer must be the coin tree's root and session key (`Coins.Session.start`), under a tree whose streams are
  the session's (`streamsOk`), and it is relayed as it is.
* `Commit` and `Round` leave only as the schedule's next opening (`State.request`, `reqAt`: the session stream's at
  `Commit`, its count `Hello`'s link points', then each rep stream's rounds in order at the counts `Zk.verifyRep` draws,
  `planOf`), never at a place the prover names. The coins answered must open the committed tree there
  (`Coins.Session.open`), and only the coins, stripped of their opening, reach the prover (`Coins.frame`).
* Each rep's frame is the loop's own, started on its frames port for the session and the table's rep
  (`Frames.startRep`), and the loop issues every request to it (`Frames`, on the schedule `Drive.reqs` fixes). A rep
  stream's `Round` carries the device's round, and what leaves in its place is the frame's release against the
  statement's round there (`Frames.round`), in a `Round` of the schedule's stream and count (`roundFrame`; any byte the
  prover put after its round is dropped): the algebraic phase's scheduled round, the opening's (the loop's own
  `Drive.openStep`, on the session's `Commit` coins and the rep's salts for levels ≥ 1), then the inner proof's own. Its
  coins, once opened, are the frame's (`Frames.feedOf`, `Frames.coins`), and the prover receives them only once the frame
  took them.
* Each table's level 0 is the loop's: on its frames port the loop draws a table's level-0 hiding and salts from the
  operating system for the packed witness the prover hands it, and commits level 0 on them (`level0`, `Level0.draw`),
  for the live session and table the request names (by its coin tree's root, `Level0.nameOf`), once a table: a second
  request for a table is refused (`Level0.Registry.draw`). A `Commit`'s roots leave only as each table's one draw for
  its own session, at its table's `m` (`Level0.commit`), and the `Commit` that leaves is the
  loop's (`commitFrame`): `root_F` the session's Σ, the roots its tables' level 0 now bound, and no public words. Every
  level 0 drawn for a session, bound or not, is gone when it ends (`Level0.Registry.close`).
* `Finish` leaves only once the schedule's last opening has (`State.request`), as `FINISH`. The proof files are the
  loop's own finish's, a connection to the frames port (`Finish.serveLoop`): it reads each table's level 0 from the
  loop's `Level0` of the table's root, draws each rep's salts for levels ≥ 1 itself, which the prover's trees then take,
  refuses a rep's request whose draws, passed values, extra claims or coins are not its frame's (`Frames.ownOf`,
  `Finish.RepIn.differs`), refuses a table's level-0 data from the host (`table`) and any rep's proof file but the one
  its live inner request computed (`Finish.Kept`), and keeps each rep's proof file, once (`Frames.Registry.file`).
* A `Proof` the prover sends is answered `Resp::Ok` and dropped. The proofs that leave, just before `Finish`, are the
  loop's kept files (`ownProofs`), every rep's of the session's bound level 0 or none, and only once every rep's frame
  ended not stopped (`Frames.finish`). The finish computes none where a constraint fails on the pads. Where a frame
  stopped its rep, `Finish` leaves without them (the auditor sees the stop there, as where a file is missing) and the
  prover's `Finish` is refused.
* `Ping` leaves as the prover's probe of 64 zero bytes (`PING`), `Link` as zeros at the session's link points
  (`linkFrame`), and `Open` as its rep stream's at the stream's domain (`openFrame`). Any other request (`Prime`, an
  unknown tag) is refused, as are a `Link` before the `Hello` and an `Open` of a stream the session does not have
  (`State.wire`).

A refusal answers the prover `Resp::Refused(WHY)`, and every coin after it is refused too, without asking the auditor: no
coin after a refused one reaches the prover. `kind` and `State` are pure; `cmd` only
moves frames.

The release side (live PROTOCOL.md §10.5) is `core-firewall session`'s, around this loop (`run` with a `Hold`): a held
session's `Ping`s are answered here until its first other request, which the hold admits or refuses before anything
reaches the auditor, and its `Finish`, with the proofs that leave before it, is held until the hold lets them go.
`flock-firewall session` has no hold.
-/
import Flock.Firewall.SessionPlan
import Flock.Firewall.Frames
import Flock.Firewall.Level0
import Flock.Firewall.Finish
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
`Round` (`4`: its stream, its content, the `u32le` count of coins it squeezes), `Finish` (`6`), `Link` (`2`), `Open`
(`3`: its stream), `Proof` (`5`), `Ping` (`9`), and anything else. -/
inductive Kind where
  | register
  | hello
  | commit
  | round (stream : String) (n : Nat)
  | finish
  | link
  | openStream (stream : String)
  | proof
  | ping
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
  | some 2 => .ok .link
  | some 5 => .ok .proof
  | some 9 => .ok .ping
  | some 3 =>
    match (getBytes f 1).bind fun (s, _) => String.fromUTF8? s with
    | some stream => .ok (.openStream stream)
    | none => .error "an Open frame without its stream"
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

/-! ### The frames that leave

What reaches the auditor is never the prover's bytes: each request leaves as the loop's own encoding (`Req::encode`) of
what the loop decided, so nothing the prover appends or chooses beyond what the loop checks crosses. -/

/-- `b`, length-prefixed (`put_bytes`). -/
def field (b : ByteArray) : ByteArray := u32le b.size ++ b

/-- `u64le(v)`. -/
def u64le (v : Nat) : ByteArray := ⟨(Array.range 8).map fun i => (v / 2 ^ (8 * i) % 256).toUInt8⟩

/-- `Req::Register`. -/
def REGISTER : ByteArray := ByteArray.mk #[10]

/-- `Req::Finish`. -/
def FINISH : ByteArray := ByteArray.mk #[6]

/-- `Req::Ping` of the prover's probe, 64 zero bytes (`ping_rtt`). -/
def PING : ByteArray := ByteArray.mk #[9] ++ field ⟨Array.replicate 64 0⟩

/-- `Req::Round` on `stream` with the content `c`, squeezing `n` coins. -/
def roundFrame (stream : String) (c : ByteArray) (n : Nat) : ByteArray :=
  ByteArray.mk #[4] ++ field stream.toUTF8 ++ field c ++ u32le n

/-- `Req::Commit` with `root_F` the session's Σ, each table's level-0 root in table order, and no public words. -/
def commitFrame (sigma : ByteArray) (roots : List (String × ByteArray)) : ByteArray :=
  roots.foldl (fun b (t, r) => b ++ field t.toUTF8 ++ field r)
    (ByteArray.mk #[7] ++ field sigma ++ u32le roots.length) ++ u32le 0

/-- `Req::Link` of `n` zero words: the link values `y`, which the zero-knowledge session sends as zeros. -/
def linkFrame (n : Nat) : ByteArray := ByteArray.mk #[2] ++ field ⟨Array.replicate (16 * n) 0⟩

/-- `Req::Open` of the root stream `stream`, at its transcript domain `label`. -/
def openFrame (stream label : String) : ByteArray :=
  ByteArray.mk #[3] ++ field stream.toUTF8 ++ field .empty ++ field label.toUTF8 ++ u64le 0

/-- `Req::Proof` of table `table`'s rep `rep`, its proof file `b`. -/
def proofFrame (table : String) (rep : Nat) (b : ByteArray) : ByteArray :=
  ByteArray.mk #[5] ++ field table.toUTF8 ++ u32le rep ++ field b

/-- A session as the loop has seen it: the auditor's draw, whether the `Hello` left, what the coin check reads of it
(`helloOf`), its statement's openings (`planOf`) and its rep streams' domain frames, the coin tree once `Hello`'s answer
checked, and the refusal that stopped it. -/
structure State where
  draw : Option Json := none
  sent : Bool := false
  hello : Option Coins.Hello := none
  plan : Array (String × Nat) := #[]
  domains : Array (String × ByteArray) := #[]
  /-- each rep stream's table and rep -/
  reps : Array (String × Nat × Nat) := #[]
  /-- each rep stream's transcript domain -/
  labels : Array (String × String) := #[]
  tables : Array (String × Nat) := #[]
  /-- the session's Σ and its link points, a table's -/
  sigma : ByteArray := .empty
  points : Nat := 0
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

/-- **The sizes the loop's frames read**, at the setup `spec`, `sts`: at each rep stream `k`, its table's link points the
session's, nonzero and within `Frames.COIN_MAX` words at the session's points; its domain, and its table's regions at
`COIN_MAX` link points, within a `u64`'s count; and `Drive.start`'s layout plus `Zk.K_PAD`, its pad row, within a
`u64`'s count. The loop serves no session whose setup fails it (`State.sendHello`). -/
def sizesOk (spec : SessionSpec) (sts : Array Stmt) : Bool :=
  (List.range spec.repStreams.size).all fun k =>
    decide (sts[k / 2]!.mPts = spec.mPts ∧ 0 < spec.mPts ∧ spec.points * spec.mPts ≤ Frames.COIN_MAX) &&
      decide ((spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8.size < 2 ^ 64) &&
      decide (sts[k / 2]!.regions.size * Frames.COIN_MAX + 2 < 2 ^ 64) &&
      match Drive.start sts[k / 2]! with
      | .ok (lay, _) => decide (lay.len + Zk.K_PAD < 2 ^ 64)
      | .error _ => true

/-- **The `Hello` that leaves** at the set-up statement `s` and the nonce `nonce` (`Hello.msg`), and the session after:
what the coin check reads of it (`helloOf`, the setup's values) and the schedule's openings (`planOf`); or the refusal
the prover receives in its place, where the statement does not set up, its sizes are past what the loop's frames read
(`sizesOk`), or the verifier's `Hello` is not JSON. -/
def State.sendHello (S : State) (s : V (SessionSpec × Array Stmt)) (nonce : ByteArray) :
    Except (State × ByteArray) (State × ByteArray) :=
  match S.stop, s with
  | some why, _ => .error (S, refused why)
  | none, .error e => .error (S.refuse s!"{Coins.STOP}the Lean Hello: setup: {e}")
  | none, .ok (spec, sts) =>
    if !sizesOk spec sts then
      .error (S.refuse s!"{Coins.STOP}the Lean Hello: the session's sizes are past the frames' (`sizesOk`)")
    else match Hello.msg spec nonce with
      | [] => .error (S.refuse s!"{Coins.STOP}the Lean Hello: the verifier's Hello is not JSON")
      | m =>
        let domains := spec.repStreams.map fun (i, r, s) => (s, Frames.domainFrame (spec.domain spec.tables[i]! r))
        let tables := spec.tables.zip (sts.map (·.m))
        let reps := spec.repStreams.map fun (i, r, s) => (s, i, r)
        let labels := spec.repStreams.map fun (i, r, s) => (s, spec.domain spec.tables[i]! r)
        .ok ({ S with sent := true, hello := helloOf spec sts nonce, plan := planOf spec sts, domains, reps,
                      labels, tables, sigma := spec.sigma, points := spec.points },
             ⟨m.toArray⟩)

/-- **What leaves for a request of kind `k`** that carries no coin, the loop's own frame (`Register`, `Ping`, `Link` and
`Open`); else why it is refused: a `Link` before the session's `Hello`, an `Open` of a stream that is none of its rep
streams, and any request the loop does not serve (`Prime`, an unknown tag). A `Link` leaves as zeros at the session's
link points, each table's, and an `Open` at its stream's domain. -/
def State.wire (S : State) (k : Kind) : Except String ByteArray :=
  match k with
  | .register => .ok REGISTER
  | .ping => .ok PING
  | .link => if S.sent then .ok (linkFrame (S.points * S.tables.size)) else .error "a Link before the session's Hello"
  | .openStream s => match S.labels.find? (·.1 == s) with
    | some (_, d) => .ok (openFrame s d)
    | none => .error s!"an Open of {s}, which is none of the session's rep streams"
  | _ => .error "a request the session loop does not serve"

/-- `k` named byte strings of `f` from byte `o`, after `acc` (reversed). -/
def namedList (f : ByteArray) : Nat → Nat → List (String × ByteArray) → Option (List (String × ByteArray))
  | 0, _, acc => some acc.reverse
  | k + 1, o, acc => do
    let (t, o) ← getBytes f o
    let (r, o) ← getBytes f o
    let t ← String.fromUTF8? t
    namedList f k o ((t, r) :: acc)

/-- A `Commit` frame's roots (`Req::Commit`'s, after its `root_f`): each table's name and level-0 root, in order. -/
def commitRoots (f : ByteArray) : Option (List (String × ByteArray)) := do
  let (_, o) ← getBytes f 1
  let n ← getU32 f o
  namedList f n (o + 4) []

/-- A `Round` frame's content. -/
def roundContent (f : ByteArray) : ByteArray :=
  match getBytes f 1 with
  | some (_, o) => ((getBytes f o).map (·.1)).getD .empty
  | none => .empty

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

/-- `c` shut down, where it still can be. -/
def close (c : Socket.Client) : IO Unit :=
  try (c.shutdown).block catch _ => pure ()

/-- `Resp::Ok`'s frame. -/
def ok : ByteArray := ByteArray.mk #[1]

/-- The prefix of what a held session's prover is told when the release side refuses or stops it. -/
def REFUSED : String := "FIREWALL-REFUSED: "

/-- `Resp::Verdict`'s frame for the loop's own verdict on a session the release side refused or stopped:
`{"accepted": false, "why": WHY}`, tag `4`, the JSON's UTF-8 length-prefixed. -/
def notAccepted (why : String) : ByteArray :=
  let v := (Json.mkObj [("accepted", false), ("why", why)]).compress
  ByteArray.mk #[4] ++ u32le v.utf8ByteSize ++ v.toUTF8

/-- Whether `r` is a `Resp::Verdict` that accepts (`"accepted": true`). -/
def accepts (r : ByteArray) : Bool :=
  match byte? r 0, getBytes r 1 with
  | some 4, some (v, _) => match (String.fromUTF8? v).map Json.parse with
    | some (.ok j) => j.getObjValD "accepted" == Json.bool true
    | _ => false
  | _, _ => false

/-- **An outer session's release side** around the loop (`core-firewall session`), each hook run on the session's own
thread. -/
structure Hold where
  /-- At the session's first request other than `Ping`, before anything of it reaches the auditor: `none` serves it, else
  why it is refused. -/
  admit : IO (Option String)
  /-- At the session's `Finish`: given why the loop stopped the session, if it did (`State.finish`), and how many of its
  `need` proofs the loop holds, `none` once they and `Finish` may leave (at the session's release time), else what the
  prover is told of its stop. -/
  barrier : (stopped : Option String) → (proofs need : Nat) → IO (Option String)
  /-- The auditor's verdict on what left (whether it accepts), or the fault that cut the session while it left. -/
  sent : Except String Bool → IO Unit
  /-- An admitted session ended before its `Finish` was decided: why. -/
  cut : String → IO Unit
  /-- The session is over, whatever its outcome. -/
  leave : IO Unit

/-- A refused session's requests from `f` on, each answered with the refusal `why` (`Finish` with a verdict that names
it) until the prover stops; nothing reaches the auditor. -/
def refuseAll (p : Socket.Client) (f : ByteArray) (why : String) : IO Unit := do
  let mut f := f
  repeat
    if byte? f 0 == some 6 then
      sendFrame p (notAccepted why)
      return
    sendFrame p (refused why)
    let some g ← recvFrame p | return
    f := g

/-- A held session up to its first request other than `Ping` (each `Ping` answered here), then `Hold.admit`: that
request, where it is admitted; `none` where the prover ended first, or where it is refused, its requests then answered
with the refusal once it has left the slot. -/
def admitted (p : Socket.Client) (h : Hold) : IO (Option ByteArray) := do
  repeat
    let some f ← recvFrame p | return none
    if byte? f 0 == some 9 then
      sendFrame p ok
      continue
    match ← h.admit with
    | none => return some f
    | some why =>
      h.leave
      refuseAll p f (REFUSED ++ why)
      return none
  return none

/-- Where a held session stands: relaying, its proofs and `Finish` leaving, or its outcome given to the hold. -/
inductive Phase where
  | relay
  | sending
  | done
  deriving BEq

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

/-- A refusal's reason as the prover receives it: a stop's as it is, anything else after `Coins.STOP`. -/
def stopOf (e : String) : String := if e.startsWith Coins.STOP then e else s!"{Coins.STOP}{e}"

/-- **The proofs that leave for session `sid`**, the loop's own: each of its tables' rep 0 and rep 1 proof file as the
loop's finish computed it on the table's level 0, the one the session's `Commit` bound (`Frames.Registry.files`,
`Level0.ofSession`), in table order; none unless every rep's is there. -/
def ownProofs (reg : IO.Ref Frames.Registry) (l0s : IO.Ref Level0.Registry) (sid : Nat)
    (tables : Array (String × Nat)) : IO (Array ByteArray) := do
  let R ← reg.get
  let L ← l0s.get
  let fs := tables.toList.flatMap fun (t, _) => [0, 1].map fun r => do
    let l ← Level0.ofSession L sid t
    let (_, _, b) ← R.files.find? fun (x, q, _) => x == l.root && q == r
    pure (proofFrame t r b)
  return if fs.all (·.isSome) then (fs.filterMap id).toArray else #[]

/-- **A session's relay** once it reaches the auditor `a` (`serve`), from its first request `first` if the hold read it:
each request decided and its own frame sent (the prover's `Proof`s dropped), and with a hold the loop's proofs and
`Finish` held for `Hold.barrier`. `phase` is where a held
session stands, and `ended` why it ended if the hold has no outcome of it. -/
def relay (S : Hello.Setup) (specs : Hello.Specs) (reg : IO.Ref Frames.Registry) (l0s : IO.Ref Level0.Registry) (sid : Nat)
    (p : Socket.Client) (hold : Option Hold) (a : Socket.Client) (first : Option ByteArray) (phase : IO.Ref Phase)
    (ended : IO.Ref String) : IO Unit := do
  let mut st : State := {}
  let mut next := first
  repeat
    let mut f := ByteArray.empty
    if let some g := next then
      f := g
      next := none
    else
      let some g ← recvFrame p | break
      f := g
    if byte? f 0 == some 5 then
      sendFrame p ok
      continue
    if byte? f 0 == some 6 then
      let complete := st.finish.isNone
      let stopped : Option String ← match st.finish with
        | some (S', _) => pure S'.stop
        | none => do
          match ← Frames.finish reg sid st.tables.size with
          | .ok () => pure none
          | .error e => pure (some (stopOf e))
      let held ← if stopped.isNone then ownProofs reg l0s sid st.tables else pure #[]
      match hold with
      | some h =>
        phase.set .done
        match ← h.barrier stopped held.size (2 * st.tables.size) with
        | some why => sendFrame p (notAccepted (REFUSED ++ why))
        | none =>
          phase.set .sending
          for pf in held do
            sendFrame a pf
            let some r ← recvFrame a | throw (IO.userError "the auditor ended the session before its verdict")
            unless byte? r 0 == some 1 do throw (IO.userError "the auditor did not take a proof")
          sendFrame a FINISH
          let some r ← recvFrame a | throw (IO.userError "the auditor ended the session before its verdict")
          phase.set .done
          h.sent (.ok (accepts r))
          sendFrame p r
        break
      | none =>
        match stopped with
        | some why =>
          let (st', out) := st.refuse why
          st := st'
          if complete then
            sendFrame a FINISH
            let some _ ← recvFrame a | ended.set "the auditor ended the session"; break
          sendFrame p out
        | none =>
          for pf in held do
            sendFrame a pf
            let some r ← recvFrame a | throw (IO.userError "the auditor ended the session before its verdict")
            unless byte? r 0 == some 1 do throw (IO.userError "the auditor did not take a proof")
          sendFrame a FINISH
          let some r ← recvFrame a | ended.set "the auditor ended the session"; break
          sendFrame p r
        continue
    match kind f with
    | .error e =>
      sendFrame p (refused s!"{Coins.STOP}{e}")
      break
    | .ok .hello =>
      let nonce ← IO.getRandomBytes Hello.NONCE_BYTES.toUSize
      let set ← S.setupAt specs st.draw
      match st.sendHello set nonce with
      | .error (st', out) =>
        st := st'
        sendFrame p out
      | .ok (st', h) =>
        st := st'
        if let .ok (spec, sts) := set then reg.modify (·.begin sid spec sts)
        sendFrame a h
        let some r ← recvFrame a | break
        let (st', out) := st.answer .hello r
        if st.session.isNone && st'.session.isSome then l0s.modify (·.name (Level0.nameOf r) sid st.tables.size)
        st := st'
        sendFrame p out
    | .ok k =>
      if let some (st', out) := st.request k then
        st := st'
        sendFrame p out
        continue
      let mut g := ByteArray.empty
      let mut rep : Option (Nat × Nat) := none
      if let .round stream n := k then
        let dom := ((st.domains.find? (·.1 == stream)).map (·.2)).getD .empty
        let (i, r) := ((st.reps.find? (·.1 == stream)).map (·.2)).getD (0, 0)
        match ← Frames.round reg l0s sid i r dom (roundContent f) n with
        | .ok out =>
          g := roundFrame stream out n
          rep := some (i, r)
        | .error e =>
          let (st', out) := st.refuse (stopOf e)
          st := st'
          sendFrame p out
          continue
      else if k == .commit then
        let tables := st.tables
        let bound : Except String (List (String × ByteArray)) ← l0s.modifyGet fun R => match commitRoots f with
          | none => (.error "a Commit frame whose roots the loop cannot read", R)
          | some roots => match Level0.commit R sid tables roots with
            | .ok R' =>
              (.ok (tables.toList.map fun (t, _) => (t, ((Level0.ofSession R' sid t).map (·.root)).getD .empty)), R')
            | .error e => (.error e, R)
        match bound with
        | .ok roots => g := commitFrame st.sigma roots
        | .error e =>
          let (st', out) := st.refuse s!"{Coins.STOP}{e}"
          st := st'
          sendFrame p out
          continue
      else
        match st.wire k with
        | .ok w => g := w
        | .error e =>
          let (st', out) := st.refuse s!"{Coins.STOP}{e}"
          st := st'
          sendFrame p out
          continue
      sendFrame a g
      let some r ← recvFrame a | ended.set "the auditor ended the session"; break
      let (st', out) := st.answer k r
      st := st'
      if let (some (i, j), .round _ n, some 2, none) := (rep, k, byte? r 0, st.stop) then
        match ← Frames.coins reg sid i j (Frames.feedOf n r.data.toList) with
        | .ok () => pure ()
        | .error e =>
          let (st', out') := st.refuse (stopOf e)
          st := st'
          sendFrame p out'
          continue
      if k == .commit && byte? r 0 == some 2 && st.stop.isNone then
        reg.modify (·.link sid (Frame.words out 5 ((out.size - 5) / 16)))
      sendFrame p out

/-- **One session**, the `sid`-th: the prover's connection `p`, relayed on a connection of its own to the auditor at
`auditor` until either side closes. Each request is read whole, decided (`kind`, `State.sendHello`, `State.request`, a
`Round` stepped through its rep's frame in `reg` at the schedule's place, `Frames.round`, a `Commit` against the tables'
level 0 `l0s`, `Level0.commit`) and only then sent, as the loop's own frame; a `Round`'s coins, once opened
(`State.answer`), are its frame's (`Frames.coins`) before the prover receives them, and the proofs, the loop's
finish's (`ownProofs`), leave only once every rep's frame finished (`Frames.finish`): where one stopped, `Finish` leaves
alone once the schedule's last opening has, and the prover is refused.

With a hold `h`, the session is admitted first (`admitted`), and only then is the auditor reached. At its `Finish` the
hold decides (`Hold.barrier`, given the loop's own refusal and how many of its own proofs the loop has, two a table): a
stop is answered with a verdict that does not accept and cuts the auditor's connection without `Finish`; otherwise
those proofs and `Finish` leave, and the auditor's verdict goes to the hold
(`Hold.sent`) before the prover receives it. A session that ends before its `Finish` is decided, or a fault of the
loop's, is the hold's to record (`Hold.cut`). -/
def serve (S : Hello.Setup) (specs : Hello.Specs) (auditor : SocketAddress) (reg : IO.Ref Frames.Registry)
    (l0s : IO.Ref Level0.Registry) (sid : Nat) (p : Socket.Client) (hold : Option Hold := none) : IO Unit := do
  let mut first : Option ByteArray := none
  if let some h := hold then
    match ← admitted p h with
    | some f => first := some f
    | none =>
      close p
      return
  let phase ← IO.mkRef Phase.relay
  let ended ← IO.mkRef "the prover ended the session before its Finish"
  try
    let a ← Socket.Client.mk
    (a.connect auditor).block
    a.noDelay
    relay S specs reg l0s sid p hold a first phase ended
    if let some h := hold then
      if (← phase.get) == .relay then
        phase.set .done
        h.cut (← ended.get)
    close a
  catch e =>
    match hold with
    | none => throw e
    | some h =>
      match ← phase.get with
      | .relay =>
        phase.set .done
        h.cut s!"the firewall's fault ({e})"
      | .sending =>
        phase.set .done
        h.sent (.error s!"{e}")
      | .done => pure ()
  finally
    if let some h := hold then h.leave
  close p

/-- The first byte `v` of `b`, if any. -/
def findByte (b : ByteArray) (v : UInt8) : Option Nat := Id.run do
  for i in [0:b.size] do
    if b.get! i == v then return some i
  return none

/-- The connection `c` as a frame's channel (`Frame.Chan`): lines and bytes read through one buffer. -/
def sockChan (c : Socket.Client) : IO Frame.Chan := do
  let buf ← IO.mkRef ByteArray.empty
  let fill : IO Bool := do
    match ← (c.recv? 65536).block with
    | none => return false
    | some b =>
      if b.isEmpty then return false
      buf.modify (· ++ b)
      return true
  let getLine : IO String := do
    while (findByte (← buf.get) 10).isNone do
      unless ← fill do
        let rest ← buf.modifyGet fun b => (b, .empty)
        return (String.fromUTF8? rest).getD "?"
    let b ← buf.modifyGet fun b => (b, .empty)
    let j := (findByte b 10).getD 0
    buf.set (b.extract (j + 1) b.size)
    return (String.fromUTF8? (b.extract 0 (j + 1))).getD "?\n"
  let readExact (n : Nat) : IO ByteArray := do
    while (← buf.get).size < n do
      unless ← fill do throw (IO.userError s!"frame: the connection closed before a {n}-byte message")
    let b ← buf.modifyGet fun b => (b, .empty)
    buf.set (b.extract n b.size)
    return b.extract 0 n
  return { getLine, readExact, write := fun b => (c.sendAll #[b]).block, flush := pure () }

/-- **A table's level 0, drawn by the loop**, for the request line `line` on `ch`: `level0 SESSION TABLE M LEN` followed by
the `LEN` bytes of the session's table `TABLE`'s packed witness is answered `ok N R` followed by the `N` bytes of the
hiding the loop drew for it from the operating system and the root it committed, the last `R` of them (`Level0.draw`,
`Level0.drawsOf`). The table's `Level0` is kept in `l0s` for table `TABLE` of the live session named `SESSION`
(`Level0.nameOf`: the hex of its coin tree's root) until that session ends, and only that session's `Commit` may carry its
root, as that table's. A table's level 0 is drawn once a session: a second request for it, or one for a table the session
does not have, is refused, and nothing drawn for it is kept or answered (`Level0.Registry.draw`, in the one update that
keeps the first). Anything else is answered `err WHY`. -/
def level0 (ch : Frame.Chan) (l0s : IO.Ref Level0.Registry) (line : String) : IO Unit := do
  let say := fun (s : String) => ch.write (s ++ "\n").toUTF8
  match (line.replace "\n" "").splitOn " " with
  | ["level0", name, j, m, len] =>
    let (some j, some m, some len) := (j.toNat?, m.toNat?, len.toNat?) | say s!"err a request `{line.trim}`"
    let b ← ch.readExact len
    if len % 16 != 0 then return (← say s!"err a {len}-byte witness, not whole words")
    match ← Level0.draw IO.getRandomBytes m (Frame.words b 0 (len / 16)) with
    | .ok l =>
      let kept : Except String Unit ← l0s.modifyGet fun R => match R.draw name j l with
        | .ok R' => (.ok (), R')
        | .error e => (.error e, R)
      if let .error e := kept then return (← say s!"err {e}")
      let d := Level0.drawsOf l
      say s!"ok {d.size} {l.root.size}"
      ch.write d
    | .error e => say s!"err {e.replace "\n" " "}"
  | _ => say s!"err a request `{line.trim}`"

/-- **The loop's frames port**: each connection to `srv` is, by its first line, one rep's frame starting (`rep …`,
`Frames.startRep`, the frame then the loop's own in `reg`); a rep's next place or the values its frame took (`place …`,
`vals …`, `Frames.ask`); one table's level 0 (`level0 …`, `level0`); or a session's
proof files on the tables' level 0 in `l0s` (`finish …`, `Finish.serveLoop`, at the statement `S`), each rep's salts for
levels ≥ 1 kept in `reg` for its opening, its proof file for the session's `Proof` (`Frames.Registry.file`), and its
requests refused unless they carry its frame's draws, the values it took, the session's extra claims and the coins the
stream answered (`Frames.ownOf`).
Anything else is answered `err WHY`. -/
def frames (S : Hello.Setup) (specs : Hello.Specs) (srv : Socket.Server) (reg : IO.Ref Frames.Registry)
    (l0s : IO.Ref Level0.Registry) : IO Unit := do
  let keep := fun (root : ByteArray) (r : Nat) (s : Array ByteArray) =>
    reg.modify fun R => { R with salts := R.salts.push (root, r, s) }
  let file := fun (root : ByteArray) (r : Nat) (b : ByteArray) => reg.modifyGet fun R => match R.file root r b with
    | .ok R' => (none, R')
    | .error e => (some e, R)
  let own := fun (i r : Nat) (root : ByteArray) => do
    return some (Frames.ownOf (← reg.get) (← l0s.get) i r root)
  repeat
    let c ← srv.accept.block
    let _ ← IO.asTask (prio := .dedicated) do
      let ch ← sockChan c
      let first ← ch.getLine
      if first.startsWith "level0 " then
        try level0 ch l0s first catch e => IO.eprintln s!"level 0: {e}"
      else if first.startsWith "finish " then
        try Finish.serveLoop ch S specs l0s keep file own first catch e => IO.eprintln s!"finish: {e}"
      else if first.startsWith "rep " then
        try Frames.startRep ch reg l0s first catch e => IO.eprintln s!"rep: {e}"
      else if first.startsWith "place " || first.startsWith "vals " then
        try Frames.ask ch reg l0s first catch e => IO.eprintln s!"{first.trim}: {e}"
      else
        try ch.write s!"err a request `{first.trim}`\n".toUTF8 catch _ => pure ()
      try (c.shutdown).block catch _ => pure ()

/-- `flock-firewall session --verifier HOST:PORT [--listen PORT] [--statement ID] --circuit F --public F …` (the statement
flags `Hello.FLAGS`): the firewall's session loop. It listens on `127.0.0.1:PORT` (by default a port the system picks)
for sessions and on a port the system picks for the reps' frames, the tables' level 0 and the sessions' finish, prints `LISTENING 127.0.0.1:PORT` and then
`FRAMES 127.0.0.1:PORT` once it does, and serves each connection as one session (`serve`) or one frame (`frames`), each
on its own thread, every session's `Hello` at one statement set up once a draw (`Hello.Setup.specAt`). When a session
ends, every level 0 drawn for it is freed (`Level0.Registry.close`), and where there were any, its standard error says
`level 0: session SID ended; K drawn for it freed, N kept`. It runs until its standard
input closes. `flock-firewall session --schedule …` instead reads a draw's JSON (`null` for none) on its standard input
and prints the openings after the session stream's at it (`planOf`), `ok N` and a line `STREAM COINS` each, or
`refused WHY`. `run` is the same loop with each session held by a hold `mkHold` makes for it (`core-firewall session`). -/
def run (usage : String) (args : List String) (mkHold : Option (IO Hold)) : IO UInt32 := do
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
  let reg : IO.Ref Frames.Registry ← IO.mkRef {}
  let l0s : IO.Ref Level0.Registry ← IO.mkRef {}
  let sids ← IO.mkRef 0
  let srv ← Socket.Server.mk
  srv.bind (.v4 { addr := IPv4Addr.ofParts 127 0 0 1, port := listen.toUInt16 })
  srv.listen 128
  let fsrv ← Socket.Server.mk
  fsrv.bind (.v4 { addr := IPv4Addr.ofParts 127 0 0 1, port := 0 })
  fsrv.listen 128
  IO.println s!"LISTENING 127.0.0.1:{(← srv.getSockName).port}"
  IO.println s!"FRAMES 127.0.0.1:{(← fsrv.getSockName).port}"
  (← IO.getStdout).flush
  let _ ← IO.asTask (prio := .dedicated) (frames S specs fsrv reg l0s)
  let _ ← IO.asTask (prio := .dedicated) do
    repeat
      let p ← srv.accept.block
      let sid ← sids.modifyGet fun n => (n, n + 1)
      let _ ← IO.asTask (prio := .dedicated) do
        try
          let hold ← match mkHold with
            | some m => some <$> m
            | none => pure none
          serve S specs auditor reg l0s sid p hold
        catch e => IO.eprintln s!"session: {e}"
        Frames.close reg sid (((← l0s.get).slots.filter (·.owner == sid)).map (·.l0.root))
        let (k, n) ← l0s.modifyGet fun R =>
          let R' := R.close sid
          ((R.slots.size - R'.slots.size, R'.slots.size), R')
        if k > 0 then
          IO.eprintln s!"level 0: session {sid} ended; {k} drawn for it freed, {n} kept"
          (← IO.getStderr).flush
  let stdin ← IO.getStdin
  repeat
    if (← stdin.getLine).isEmpty then break
  IO.Process.exit 0

/-- `flock-firewall session`: `run`, with no hold. -/
def cmd (args : List String) : IO UInt32 :=
  run s!"usage: flock-firewall session (--verifier HOST:PORT [--listen PORT] | --schedule) {Hello.FLAGS}" args none

end Flock.Firewall.Session
