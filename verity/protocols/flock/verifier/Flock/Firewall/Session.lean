/-
# The ZK server's session loop, in Lean (live PROTOCOL.md §9, §10)

A firewalled prover's sessions go through one `flock-firewall session` process, which owns the auditor's connection. The
prover's workers connect to it as they would to the auditor (`TcpTransport`, the wire's frames `u32le(len) ‖ bytes`), and
each connection is one session, relayed on a connection of its own to the auditor. What crosses is decided here, and
every frame that leaves is the loop's own encoding of what it decided (`roundFrame`, `commitFrame`, `State.wire`), never
the prover's bytes:

* `Register` leaves as `REGISTER`, and its answer (`Resp::Draw`) is the session's sample (`State.answer`).
* `Hello`: the prover's text is dropped, and what leaves is the Lean `Hello` at that sample (`Hello.msg`), at a nonce this
  process samples from the operating system (`State.sendHello`). The coin check reads the session's tree, link points and
  nonce from the setup's values (`helloOf`), never from the text. A setup whose sizes are past what the loop's frames
  read (`sizesOk`) is refused there: no `Hello` leaves, and the session is stopped.
* `Hello`'s answer must be the coin tree's root and session key (`Coins.Release.start`), under a tree whose streams are
  the session's (`streamsOk`), and it never reaches the prover: what the prover receives in its place is the loop's id
  for the session, sampled from the operating system with the `Hello`'s nonce (`idFrame`), by which the session is named
  on the loop's frames port once its answer checked (`Level0.nameOf`).
* `Commit` and `Round` leave only as the schedule's next opening (`State.request`, `reqAt`: the session stream's at
  `Commit`, its count `Hello`'s link points', then each rep stream's rounds in order at the counts `Zk.verifyRep` samples,
  `planOf`), never at a place the prover names. The coins answered must open the committed tree there
  (`Coins.Release.open`), and only the coins, stripped of their opening, reach the prover (`Coins.frame`). An answer
  that is not coins stops the session, refused (`State.noCoins`), as a refused opening does.
* Each rep's frame is the loop's own, started on its frames port for the session and the table's rep
  (`Frames.startRep`), and the loop issues every request to it (`Frames`, on the schedule `Drive.reqs` fixes). A rep
  stream's `Round` carries the device's round, and what leaves in its place is the frame's release against the
  statement's round there (`Frames.round`), in a `Round` of the schedule's stream and count (`roundFrame`; any byte the
  prover put after its round is dropped): the algebraic phase's scheduled round, the opening's (the loop's own
  `Drive.openStep`, on the session's `Commit` coins and the rep's salts for levels ≥ 1), then the inner proof's, the
  frame's own on the rounds the loop's finish computed and handed the loop's frames before the prover relays them
  (`Frame.innerSched`, `Frames.Sessions.innerRound`). An inner round the finish has not computed is refused, whatever
  the prover relays (`Frames.roundPlace`). Nor does any rep's frame take a device round once a rep of the session
  stopped (`Frames.Sessions.stopped`): the session stops at its first failure, every rep with it. Its coins, once
  opened, are the frame's (`Frames.feedOf`, `Frames.coins`), and the prover receives them only once the frame took
  them.
* Each table's level 0 is the loop's: on its frames port the prover hands the loop every table's packed witness in one
  request, and only then does the loop sample each table's mask-slot words, level-0 hiding and salts from the operating
  system, write the words into the witness's mask slot, and commit level 0 on them (`level0`, `Level0.draw`), for the
  live session the request names (by the loop's id for it, `Level0.nameOf`), at each statement's `m`, once a session: a
  second request, or one without a witness for each table, is refused (`Level0.Slots.draw`), and no rep's frame starts
  before it (`Frames.startRep`). So every witness is the prover's before any of the session's samples reaches it, as the
  model fixes the witness before the tape. A `Commit`'s roots leave only as each table's one sample for
  its own session, at its table's `m` (`Level0.commit`), and the `Commit` that leaves is the
  loop's (`commitFrame`): `root_F` the session's Σ, the roots its tables' level 0 now bound, and no public words. Every
  level 0 sampled for a session, bound or not, is gone when it ends (`Level0.Slots.close`).
* `Finish` leaves only once the schedule's last opening has (`State.request`), as `FINISH`. The proof files are the
  loop's own finish's, a connection to the frames port (`Finish.serveLoop`): it reads each table's level 0 from the
  loop's `Level0` of the table's root, samples each rep's salts for levels ≥ 1 itself, only once the rep's frame took its
  algebraic phase's last coins (`Finish.Own.algebraEnded`) and once a rep on any of the session's finishes, which the
  prover's trees then take, proves each rep with its frame's samples, which never reach the prover (`Finish.drawsOf`), refuses a rep's request whose
  passed values, extra claims or coins are not its frame's (`Frames.ownOf`, `Finish.RepIn.differs`) or whose `final_c`
  its passed values do not fix (`Finish.consWhy`), and coins answering its
  inner proof's rounds that are not the stream's (`Finish.coinsWhy`), refuses a table's level-0 data from the host
  (`table`) and any rep's proof file but the one its live inner request computed (`Finish.Kept`), and keeps each rep's
  proof file, once (`Frames.Sessions.file`).
* A `Proof` the prover sends before a stop is answered `Resp::Ok` and dropped. The proofs that leave, just before `Finish`, are the
  loop's kept files (`ownProofs`), every rep's of the session's bound level 0 or none, and only once every rep's frame
  ended not stopped (`Frames.finish`). The finish computes none where a constraint fails on the pads. Where a frame
  stopped its rep, `Finish` leaves without them (the auditor sees the stop there, as where a file is missing) and the
  prover's `Finish` is refused.
* `Link` leaves as zeros at the session's link points (`linkFrame`), and `Open` as its rep stream's at the stream's domain
  (`openFrame`). `Ping` never leaves: the loop answers it `Resp::Ok` itself, before a stop and after. Any other request
  (`Prime`, an unknown tag) is refused (`State.wire`).
* Each request leaves at most once, at its place in the session's schedule, the honest prover's order (`State.place`, on
  what of the session's frames that carry no coin left before it, `Placed`; a `Commit` or `Round` at the opening its
  coins count, `State.request`, where an answer that is not coins ends the session, `State.answer`, so no opening is
  asked for twice): `Register`; `Hello`; the first rep stream's `Open`; `Commit`; `Link`; the first rep stream's rounds;
  then each later rep stream's `Open` and its rounds, in `SessionSpec.repStreams` order, a stream's `Open` only once the
  rounds before it are past; and `Finish` once every rep stream opened. A second `Register`, `Hello`, `Link` or `Open`
  of a stream, and any of them out of its place, is a refusal. So what leaves is fixed by the statement and the
  schedule, as the model has it: each frame that carries no coin once, at its place, and one message a round
  (`Claims.ZKServer.LeanRelease.toSrv`), and nothing the prover places.

A refusal answers the prover `Resp::Refused(WHY)` and stops the session: every request after it is refused for the same
reason without asking the auditor (`State.request`), `Ping` alone answered `Resp::Ok` by the loop, so nothing of the
session reaches the auditor after its stop, and the auditor's connection closes at the session's end as at any other
stop. No coin after a refused one reaches the prover. The session ends at its `Finish`, whatever the loop decided of it:
once the prover has its verdict or refusal, the loop reads nothing more of the session and closes both connections, so
no `Finish` and no proof leaves twice. `kind` and `State` are pure; `cmd` only moves frames.

The release side (live PROTOCOL.md §10.5) is `core-firewall session`'s, around this loop (`run` with a `Hold`): a held
session's `Ping`s are answered here until its first other request, which the hold admits or refuses before anything
reaches the auditor, and its `Finish`, with the proofs that leave before it, is held until the hold lets them go.
`flock-firewall session` has no hold.
-/
module

public import Flock.Firewall.SessionPlan
public import Flock.Firewall.Frames
public import Flock.Firewall.Level0
public import Flock.Firewall.Finish
public import Std.Async.TCP
public import Std.Async.DNS
public import Flock.JsonParse

@[expose] public section

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

/-- A session as the loop has seen it: the auditor's sample, whether the `Hello` left, what the coin check reads of it
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
  session : Option Coins.Release := none
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

/-- **What of a session's frames that carry no coin has left**, beside `State`: its `Register`, its `Link`, and how many of
its rep streams' `Open`s, which leave in `SessionSpec.repStreams` order. -/
structure Placed where
  register : Bool := false
  link : Bool := false
  opens : Nat := 0

/-- `P` once a request of kind `k` left. -/
def Placed.after (P : Placed) : Kind → Placed
  | .register => { P with register := true }
  | .link => { P with link := true }
  | .openStream _ => { P with opens := P.opens + 1 }
  | _ => P

/-- **Whether a request of kind `k` is at its place** in the session's schedule, once `P` left: `none` where it is, else
why not. The schedule is the honest prover's order (`flock-circuit`'s): `Register`; `Hello`; the first rep stream's `Open`;
`Commit`; `Link`, just after `Commit`'s coins; the first rep stream's rounds; then each later rep stream's `Open`, once
the rounds before it are past, and its rounds, in `SessionSpec.repStreams` order; `Finish` once every rep stream opened.
So `Register`, `Hello`, `Link` and each stream's `Open` have one place each, and a second one has none. Which opening a
`Commit` or `Round` is, and whether `Finish` comes after the last, are `State.request`'s (`reqAt`, `State.finish`).
`Ping` and `Proof` never leave, and any other request (`Prime`, an unknown tag) has no place. -/
def State.place (S : State) (P : Placed) (k : Kind) : Option String :=
  let j := (S.session.map (·.opened.length)).getD 0
  let streamAt := fun (i : Nat) => (S.labels[i]?.map (·.1)).getD ""
  match k with
  | .register => if P.register then some "a second Register" else none
  | .hello =>
    if !P.register then some "a Hello before the Register"
    else if S.sent then some "a second Hello"
    else none
  | .link =>
    if !S.sent then some "a Link before the session's Hello"
    else if P.link then some "a second Link"
    else if j != 1 then some s!"a Link at opening {j}, where its place is just after the step-0 coin commitment"
    else none
  | .openStream s =>
    match S.labels.findIdx? (·.1 == s) with
    | none => some s!"an Open of {s}, which is none of the session's rep streams"
    | some i =>
      if i < P.opens then some s!"a second Open of {s}"
      else if P.opens < i then some s!"an Open of {s} before the Open of {streamAt P.opens}"
      else if i = 0 then none
      else if !P.link then some s!"an Open of {s} before the Link"
      else if (S.plan[j - 1]?.map (·.1)) == some (streamAt (i - 1)) then
        some s!"an Open of {s} before the last round of {streamAt (i - 1)}"
      else none
  | .commit => if P.opens = 0 then some s!"a Commit before the Open of {streamAt 0}" else none
  | .round stream _ =>
    if !P.link then some s!"a round of {stream} before the Link"
    else if (S.labels.extract 0 P.opens).any (·.1 == stream) then none
    else some s!"a round of {stream} before its Open"
  | .finish => if P.opens < S.labels.size then some s!"a Finish before the Open of {streamAt P.opens}" else none
  | .ping | .proof => none
  | .other => some "a request the session loop does not serve"

/-- **Whether a request of kind `k` leaves**, once `P` of the session's frames that carry no coin left: `none` where it
does, else the refusal the prover receives in its place and the session after. After a stop every request is refused for
its reason, so nothing more of the session reaches the auditor. A `Commit` or `Round` leaves only as the schedule's next
opening (`reqAt`): no coin is asked for before `Hello`'s answer or at a place the prover picks. `Finish` leaves only after
the last (`State.finish`). Every request leaves only at its place (`State.place`). -/
def State.request (S : State) (P : Placed) (k : Kind) : Option (State × ByteArray) :=
  match S.stop with
  | some why => some (S, refused why)
  | none =>
    let coins : Option (State × ByteArray) := if k == .finish then S.finish else
      let opening : Option (String × Nat) := match k with
        | .commit => some (Coins.SESSION, 0)
        | .round stream n => some (stream, n)
        | _ => none
      match opening, S.session with
      | none, _ => none
      | some _, none => some (S.refuse s!"{Coins.STOP}coins before the step-0 coin commitment")
      | some (stream, n), some C =>
        let j := C.opened.length
        let (st, want) := reqAt S.plan C.hello.links j
        if j = 0 then
          if k == .commit then none
          else some (S.refuse s!"{Coins.STOP}a round of {stream} before the step-0 coin commitment")
        else if k == .commit then some (S.refuse s!"{Coins.STOP}a second step-0 coin commitment")
        else if j > S.plan.size then some (S.refuse s!"{Coins.STOP}a round of {stream} past the schedule's {S.plan.size}")
        else if stream == st && n == want then none
        else some (S.refuse s!"{Coins.STOP}opening {j}: {n} coins of {stream}, where the schedule's are {want} of {st}")
    match coins with
    | some r => some r
    | none => (S.place P k).map fun why => S.refuse s!"{Coins.STOP}{why}"

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

/-- **The auditor answered the coin request `what` with `a`, which is not coins**: the session stopped, refused, as the
model stops at coins it refuses (`Claims.ZKServer.LeanRelease.refusesCoins`), so the request never leaves again. The
prover receives the refusal, with the auditor's reason where `a` is a `Resp::Refused`, and nothing else of `a`. -/
def State.noCoins (S : State) (what : String) (a : ByteArray) : State × ByteArray :=
  match S.stop with
  | some why => (S, refused why)
  | none =>
    let said := match byte? a 0, (getBytes a 1).bind fun (b, _) => String.fromUTF8? b with
      | some 3, some why => s!": {why}"
      | _, _ => ""
    S.refuse s!"{Coins.STOP}the auditor answered {what} without coins{said}"

/-- `Resp::Coins`' frame of the loop's id `id` for a session: what the prover receives for `Hello`'s answer. -/
def idFrame (id : ByteArray) : ByteArray := ⟨(Coins.frame (id.size / 16) id.data.toList).toArray⟩

/-- **What the prover receives** for the auditor's answer `a` to a request of kind `k`, and the session after: the sample
taken from `Register`'s answer, the loop's id `id` for the session once `Hello`'s answer checked (`idFrame`), a round's
coins opened (`State.coins`), a `Commit`'s or a `Round`'s answer that is not coins the session's stop
(`State.noCoins`); anything else as it is. -/
def State.answer (S : State) (k : Kind) (a : ByteArray) (id : ByteArray := .empty) : State × ByteArray :=
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
    | none, some h => match Coins.Release.start h a.data.toList with
      | .ok C => ({ S with session := some C }, idFrame id)
      | .error e => S.refuse e
  | .commit, some 2 => S.coins a
  | .round _ _, some 2 => S.coins a
  | .commit, _ => S.noCoins "the step-0 coin commitment" a
  | .round stream _, _ => S.noCoins s!"a round of {stream}" a
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

/-- **What leaves for a request of kind `k`** that carries no coin, at its place (`State.place`), the loop's own frame
(`Register`, `Link` and `Open`); else why it is refused: an `Open` of a stream that is none of its rep streams, and any
request the loop does not serve (`Ping`, which the loop answers itself, `Prime`, an unknown tag). A `Link` leaves as zeros
at the session's link points, each table's, and an `Open` at its stream's domain. -/
def State.wire (S : State) (k : Kind) : Except String ByteArray :=
  match k with
  | .register => .ok REGISTER
  | .link => .ok (linkFrame (S.points * S.tables.size))
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
  | some 4, some (v, _) => match (String.fromUTF8? v).map JsonParse.parse with
    | some (.ok j) => j.getObjValD "accepted" == Json.bool true
    | _ => false
  | _, _ => false

/-- **An outer session's release side** around the loop (`core-firewall session`), each hook run on the session's own
thread. -/
structure Hold where
  /-- At the session's first request other than `Ping`, before anything of it reaches the auditor: `none` serves it, else
  why it is refused. -/
  admit : IO (Option String)
  /-- At the auditor's answer to the session's `Register`, before the prover receives it and before `Hello` leaves:
  given what follows the draw in it (`drawAnswer`: the verifier's list of the statement's known trees), `none` goes on,
  else why the session is refused, which the hold has recorded. -/
  known : ByteArray → IO (Option String)
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

/-- **The auditor's answer `a` to `Register`** as its draw's frame and what follows it: `Resp::Draw`'s tag and
length-prefixed text, which is all the prover receives of it and all `State.answer` reads, then the verifier's list of
the statement's known trees (`Salts.parseKnown`), which is the release side's (`Hold.known`). An answer that is no draw
is whole, with nothing after it. -/
def drawAnswer (a : ByteArray) : ByteArray × ByteArray :=
  match byte? a 0, getBytes a 1 with
  | some 6, some (_, o) => (a.extract 0 o, a.extract o a.size)
  | _, _ => (a, .empty)

/-- A refused session's requests from `f` on, each answered with the refusal `why` (`Finish` with a verdict that names
it) until the prover stops; nothing more reaches the auditor. -/
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
loop's finish computed it on the table's level 0, the one the session's `Commit` bound (`Frames.Sessions.files`,
`Level0.ofSession`), in table order; none unless every rep's is there. -/
def ownProofs (reg : IO.Ref Frames.Sessions) (l0s : IO.Ref Level0.Slots) (sid : Nat)
    (tables : Array (String × Nat)) : IO (Array ByteArray) := do
  let R ← reg.get
  let L ← l0s.get
  let fs := tables.toList.flatMap fun (t, _) => [0, 1].map fun r => do
    let l ← Level0.ofSession L sid t
    let (_, _, b) ← R.files.find? fun (x, q, _) => x == l.root && q == r
    pure (proofFrame t r b)
  return if fs.all (·.isSome) then (fs.filterMap id).toArray else #[]

/-- **A session's relay** once it reaches the auditor `a` (`serve`), from its first request `first` if the hold read it:
each request decided and its own frame sent at its place (`State.request`, the prover's `Proof`s dropped and its `Ping`s
answered by the loop), nothing after a stop, the session ended at its `Finish` whatever its outcome, and with a hold
the loop's proofs and `Finish` held for `Hold.barrier`. Of the auditor's answer to `Register` the prover receives the
draw's frame and nothing after it (`drawAnswer`); with a hold, what follows is `Hold.known`'s, and a session it refuses
has the auditor's connection cut and is answered with the refusal from then on.
`phase` is where a held session stands, and `ended` why it ended if the hold has no outcome of it. -/
def relay (S : Hello.Setup) (specs : Hello.Specs) (reg : IO.Ref Frames.Sessions) (l0s : IO.Ref Level0.Slots) (sid : Nat)
    (p : Socket.Client) (hold : Option Hold) (a : Socket.Client) (first : Option ByteArray) (phase : IO.Ref Phase)
    (ended : IO.Ref String) : IO Unit := do
  let mut st : State := {}
  let mut pl : Placed := {}
  let mut next := first
  repeat
    let mut f := ByteArray.empty
    if let some g := next then
      f := g
      next := none
    else
      let some g ← recvFrame p | break
      f := g
    if byte? f 0 == some 9 then
      sendFrame p ok
      continue
    if byte? f 0 == some 5 && st.stop.isNone then
      sendFrame p ok
      continue
    if byte? f 0 == some 6 then
      let fin := st.request pl .finish
      let complete := fin.isNone
      let stopped : Option String ← match fin with
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
      | none =>
        match stopped with
        | some why =>
          if complete then
            sendFrame a FINISH
            let some _ ← recvFrame a | ended.set "the auditor ended the session"; break
          sendFrame p (refused why)
        | none =>
          for pf in held do
            sendFrame a pf
            let some r ← recvFrame a | throw (IO.userError "the auditor ended the session before its verdict")
            unless byte? r 0 == some 1 do throw (IO.userError "the auditor did not take a proof")
          sendFrame a FINISH
          let some r ← recvFrame a | ended.set "the auditor ended the session"; break
          sendFrame p r
      break
    match kind f with
    | .error e =>
      sendFrame p (refused (st.stop.getD s!"{Coins.STOP}{e}"))
      break
    | .ok .hello =>
      if let some (st', out) := st.request pl .hello then
        st := st'
        sendFrame p out
        continue
      let nonce ← IO.getRandomBytes Hello.NONCE_BYTES.toUSize
      let id ← IO.getRandomBytes Level0.ID_BYTES.toUSize
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
        let (st', out) := st.answer .hello r id
        if st.session.isNone && st'.session.isSome then l0s.modify (·.name (Level0.nameOf id) sid st.tables.size)
        st := st'
        sendFrame p out
    | .ok k =>
      if let some (st', out) := st.request pl k then
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
      pl := pl.after k
      let some whole ← recvFrame a | ended.set "the auditor ended the session"; break
      let mut r := whole
      if k == .register then
        let (draw, list) := drawAnswer whole
        r := draw
        if let some h := hold then
          if let some why ← h.known list then
            phase.set .done
            h.leave
            close a
            refuseAll p f (REFUSED ++ why)
            break
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
`auditor` until either side closes or the prover has the answer to its `Finish`. Each request is read whole, decided
(`kind`, `State.sendHello`, `State.request`, a `Round` stepped through its rep's frame in `reg` at the schedule's place,
`Frames.round`, a `Commit` against the tables' level 0 `l0s`, `Level0.commit`) and only then sent, as the loop's own
frame; a `Round`'s coins, once opened (`State.answer`), are its frame's (`Frames.coins`) before the prover receives
them, and the proofs, the loop's finish's (`ownProofs`), leave only once every rep's frame finished (`Frames.finish`):
where one stopped, `Finish` leaves alone once the schedule's last opening has, and the prover is refused.

With a hold `h`, the session is admitted first (`admitted`), and only then is the auditor reached. At its `Finish` the
hold decides (`Hold.barrier`, given the loop's own refusal and how many of its own proofs the loop has, two a table): a
stop is answered with a verdict that does not accept and cuts the auditor's connection without `Finish`; otherwise
those proofs and `Finish` leave, and the auditor's verdict goes to the hold
(`Hold.sent`) before the prover receives it. A session that ends before its `Finish` is decided, or a fault of the
loop's, is the hold's to record (`Hold.cut`). -/
def serve (S : Hello.Setup) (specs : Hello.Specs) (auditor : SocketAddress) (reg : IO.Ref Frames.Sessions)
    (l0s : IO.Ref Level0.Slots) (sid : Nat) (p : Socket.Client) (hold : Option Hold := none) : IO Unit := do
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

/-- A level-0 request's sizes, `M LEN` for each table in table order; none unless there is a pair and every word is a
number. -/
def level0Sizes : List String → Option (Array (Nat × Nat))
  | [] => none
  | ws => go ws #[]
where
  go : List String → Array (Nat × Nat) → Option (Array (Nat × Nat))
    | [], acc => some acc
    | m :: len :: rest, acc => do go rest (acc.push (← m.toNat?, ← len.toNat?))
    | _, _ => none

/-- **Every table's level 0, sampled by the loop once it holds every table's witness**, for the request line `line` on
`ch`: `level0 SESSION M₀ LEN₀ … M_{J−1} LEN_{J−1}`, a pair for each of the session's `J` tables in table order, followed
by each table's packed witness in that order, `LEN_j` bytes at its statement's `M_j`. Only once it has read every
table's witness does the loop sample anything: for each table the mask slot's words, which it writes into the witness's
mask slot (`Frames.Sessions.maskOf`, `Frame.MaskSlot.fill`), and the hiding it commits level 0 on (`Level0.draw`). It
answers each table in order `ok N R` followed by the `N` bytes of the mask slot's words, the hiding and the root, the
last `R` of them (`Level0.drawsOf`). So no table's witness can read any of the session's samples: the prover has handed
every witness before it receives any of them, and before any rep's frame starts (`Frames.startRep`). The tables'
`Level0`s are kept in `l0s` for the live session named `SESSION` (`Level0.nameOf`: the hex of the loop's id for it)
until that session ends, and only that session's `Commit` may carry their roots, each as its table's. A session's level 0
is sampled once: a second request, one whose witnesses are not one for each of the session's tables, or one at another
`M`, is refused, and nothing sampled for it is kept or answered (`Level0.Slots.draw`, in the one update that keeps the
first). Anything else is answered `err WHY`. -/
def level0 (ch : Frame.Chan) (reg : IO.Ref Frames.Sessions) (l0s : IO.Ref Level0.Slots) (line : String) : IO Unit := do
  let say := fun (s : String) => ch.write (s ++ "\n").toUTF8
  match (line.replace "\n" "").splitOn " " with
  | "level0" :: name :: sizes =>
    let some asks := level0Sizes sizes | say s!"err a request `{line.trim}`"
    let mut bs : Array ByteArray := #[]
    for (_, len) in asks do
      bs := bs.push (← ch.readExact len)
    if let some (_, len) := asks.find? (·.2 % 16 != 0) then
      return (← say s!"err a {len}-byte witness, not whole words")
    if let .error e := (← l0s.get).admit name asks.size then return (← say s!"err {e}")
    let mut ls : Array Level0.Level0 := #[]
    let mut ds : Array ByteArray := #[]
    for ((m, len), b) in asks.zip bs do
      let (slot, kLog) ← match (← reg.get).maskOf (← l0s.get) name ls.size m with
        | .ok x => pure x
        | .error e => return (← say s!"err {e}")
      let w := Frame.words (← IO.getRandomBytes (16 * slot.size).toUSize) 0 slot.size
      let g ← match slot.fill kLog (Frame.words b 0 (len / 16)) w with
        | .ok g => pure g
        | .error e => return (← say s!"err {e}")
      match ← Level0.draw IO.getRandomBytes m g with
      | .ok l =>
        ls := ls.push l
        ds := ds.push (Frame.wordBytes w ++ Level0.drawsOf l)
      | .error e => return (← say s!"err {e.replace "\n" " "}")
    let kept : Except String Unit ← l0s.modifyGet fun R => match R.draw name ls with
      | .ok R' => (.ok (), R')
      | .error e => (.error e, R)
    if let .error e := kept then return (← say s!"err {e}")
    for (d, l) in ds.zip ls do
      say s!"ok {d.size} {l.root.size}"
      ch.write d
  | _ => say s!"err a request `{line.trim}`"

/-- **The loop's frames port**: each connection to `srv` is, by its first line, one rep's frame starting (`rep …`,
`Frames.startRep`, the frame then the loop's own in `reg`); a rep's next place or the values its frame took (`place …`,
`vals …`, `Frames.ask`, which only reads `reg`); a session's tables' level 0 (`level0 …`, `level0`); or a session's
proof files on the tables' level 0 in `l0s` (`finish …`, `Finish.serveLoop`, at the statement `S`), each rep's salts for
levels ≥ 1 kept in `reg` for its opening, once (another finish's sample of them is refused), its inner proof's rounds for
the frame to release
(`Frames.Sessions.innerRound`), its proof file for the session's `Proof` (`Frames.Sessions.file`), its proofs on its
frame's samples, and its requests refused unless they carry the values its frame took, the session's extra claims and the
coins the stream answered (`Frames.ownOf`).
Anything else is answered `err WHY`. -/
def frames (S : Hello.Setup) (specs : Hello.Specs) (srv : Socket.Server) (reg : IO.Ref Frames.Sessions)
    (l0s : IO.Ref Level0.Slots) : IO Unit := do
  let keep := fun (root : ByteArray) (r : Nat) (s : Array ByteArray) => reg.modifyGet fun R =>
    if R.salts.any fun (x, q, _) => x == root && q == r then
      (some s!"rep {r}'s salts for levels ≥ 1 were drawn already, on another finish", R)
    else (none, { R with salts := R.salts.push (root, r, s) })
  let file := fun (root : ByteArray) (r : Nat) (b : ByteArray) => reg.modifyGet fun R => match R.file root r b with
    | .ok R' => (none, R')
    | .error e => (some e, R)
  let own := fun (i r : Nat) (root : ByteArray) => do
    return some (Frames.ownOf (← reg.get) (← l0s.get) i r root)
  let inner := fun (root : ByteArray) (r j : Nat) (b : ByteArray) => reg.modifyGet fun R =>
    match R.innerRound root r j b with
    | .ok R' => (none, R')
    | .error e => (some e, R)
  repeat
    let c ← srv.accept.block
    let _ ← IO.asTask (prio := .dedicated) do
      let ch ← sockChan c
      let first ← ch.getLine
      if first.startsWith "level0 " then
        try level0 ch reg l0s first catch e => IO.eprintln s!"level 0: {e}"
      else if first.startsWith "finish " then
        try Finish.serveLoop ch S specs l0s keep file own inner first catch e => IO.eprintln s!"finish: {e}"
      else if first.startsWith "rep " then
        try Frames.startRep ch reg l0s first catch e => IO.eprintln s!"rep: {e}"
      else if first.startsWith "place " || first.startsWith "vals " then
        try Frames.ask ch reg l0s first catch e => IO.eprintln s!"{first.trim}: {e}"
      else
        try ch.write s!"err a request `{first.trim}`\n".toUTF8 catch _ => pure ()
      try (c.shutdown).block catch _ => pure ()

/-- `flock-firewall session --verifier HOST:PORT [--listen PORT] [--statement ID] --circuit F --public F …` (the
statement flags `Hello.FLAGS`): the ZK server's session loop. It listens on `127.0.0.1:PORT` (by default a port the system picks)
for sessions and on a port the system picks for the reps' frames, the tables' level 0 and the sessions' finish, prints `LISTENING 127.0.0.1:PORT` and then
`FRAMES 127.0.0.1:PORT` once it does, and serves each connection as one session (`serve`) or one frame (`frames`), each
on its own thread, every session's `Hello` at one statement set up once a sample (`Hello.Setup.specAt`). When a session
ends, every level 0 sampled for it is freed (`Level0.Slots.close`), and where there were any, its standard error says
`level 0: session SID ended; K drawn for it freed, N kept`. It runs until its standard
input closes. `flock-firewall session --schedule …` instead reads a sample's JSON (`null` for none) on its standard input
and prints the openings after the session stream's at it (`planOf`), `ok N` and a line `STREAM COINS` each, or
`refused WHY`. `run` is the same loop with each session held by a hold (`core-firewall session`): `mkHold`, given the
statement once it is set up and before the loop listens, makes each session's, or says why the loop serves none of the
statement, which it prints as `refused WHY` before it exits. Without a hold the loop admits no registration, so it
refuses `--salt-key`: a statement's value commitments are admitted on the ZK server's record (`core-firewall session
--state DIR --salt-key FILE`). -/
def run (usage : String) (args : List String) (mkHold : Option (Hello.Setup → IO (Except String (IO Hold)))) :
    IO UInt32 := do
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
  let made : Except String (Option (IO Hold)) ← match mkHold with
    | some f => (·.map some) <$> f S
    | none =>
      pure <| if (Witness.argOf args "--salt-key").isSome then
        .error "--salt-key: a statement's value commitments are admitted on the ZK server's record (core-firewall session --state DIR --salt-key FILE)"
      else .ok none
  let mkHold ← match made with
    | .ok m => pure m
    | .error why =>
      IO.println s!"refused {why}"
      (← IO.getStdout).flush
      return 2
  let auditor ← resolve verifier
  let specs : Hello.Specs ← IO.mkRef #[]
  let reg : IO.Ref Frames.Sessions ← IO.mkRef {}
  let l0s : IO.Ref Level0.Slots ← IO.mkRef {}
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
