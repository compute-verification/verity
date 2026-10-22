/-
# The session loop's frames: each rep's frame, driven by the loop on the one schedule

The session loop (`Session`) holds each firewalled rep's frame itself (`Frame.step` at the frame's `Cfg` and `State`) and
issues every request to it, on the schedule `Drive.reqs` fixes. Nothing the prover sends names a frame's request:

* `startRep`: a rep's frame starts at the loop's frames port (`rep NAME TABLE REP`), for a live session and one of its
  tables' reps, at the statement's start (`Drive.start`: its pads layout and mask slot), its draws from the operating
  system, which the prover proves with.
* `round`: a session's `Round` of a rep stream carries the device's round. Its place is the statement's: the algebraic
  phase's scheduled round (`Skeleton.rounds`, on the table's level 0, the loop's own), then the opening's
  (`Drive.openStep`, the rep's opening a round at a time, on the extra claims at the session's `Commit` coins,
  `Drive.extras`, and the rep's salts for levels ≥ 1, the loop's finish's), then none, at the inner proof's own rounds. Rep 1's frame first takes
  rep 0's algebraic-phase coins (`prior`). What leaves is the frame's release, at the stream's first round after its
  domain frame (`domainFrame`, `Transcript.init`'s).
* `coins`: once the coins answered that round open the committed tree (`State.answer`), the frame takes them
  (`feedOf` the answer), and the opening goes on with them where the round was its; the prover receives them only once the frame did.
* `finish`: the session's `Finish` leaves only once every rep's frame answers `finish` at its end, not stopped.

So every coin a frame takes is the loop's own reading of the auditor's answer, and every round in a device round's place
is the loop's own, a function of the statement, the table's level 0, the rep's salts and the coins: the opening's are
`Drive.opening`'s, which feeds `Drive.openStep` the same coins (`Drive.Step.feed`).
-/
import Flock.Firewall.Drive

namespace Flock.Firewall.Frames

open Frame (words wordBytes)
open Flock Transcript Finish

/-- The most coins of one round a frame is fed. -/
def COIN_MAX : Nat := 65536

/-- **The coins the loop feeds a frame** for an opening `a` of `n` coins (`Resp::Coins`' frame): its first `n` words,
16 bytes each after the tag and count, as `Coins.check` cuts them, up to `COIN_MAX`. -/
def feedOf (n : Nat) (a : List UInt8) : Array F128 := words ⟨(a.drop 5).toArray⟩ 0 (min n COIN_MAX)

/-- A stream's domain frame, which its first round begins with (`Transcript.init`, `frame_domain`). -/
def domainFrame (d : String) : ByteArray :=
  let b := d.toUTF8
  Transcript.header 1 0 b.size ++ Transcript.padded b

/-- A rep's frame in the loop: whose it is, its configuration and state, the algebraic phase's scheduled rounds once its
first round came, the coins of each round it released, and its opening once it began. -/
structure Rep where
  sid : Nat
  table : Nat
  rep : Nat
  cfg : Frame.Cfg
  st : Frame.State := {}
  alg : Array (ByteArray × Nat) := #[]
  coins : Array (Array F128) := #[]
  /-- the rep's opening, at its next round (`Drive.openStep`) -/
  opening : Option (Drive.Step Unit) := none
  /-- the opening ended: every round after it is the inner proof's -/
  opened : Bool := false
  /-- the round last released was the opening's, whose coins go to it too -/
  toOpening : Bool := false

/-- A session the loop serves frames for: its statement, and the `Commit`'s coins once they were answered. -/
structure Session where
  sid : Nat
  spec : SessionSpec
  sts : Array Stmt
  link : Option (Array F128) := none

/-- **The loop's frames**: the sessions, each rep's frame, and each rep's salts for levels ≥ 1 by its table's level-0
root, the loop's finish's (`Finish.serve`'s `keep`). -/
structure Registry where
  sessions : Array Session := #[]
  reps : Array Rep := #[]
  salts : Array (ByteArray × Nat × Array ByteArray) := #[]

/-- Session `sid`, set up on `spec` and `sts` at its `Hello`. -/
def Registry.begin (R : Registry) (sid : Nat) (spec : SessionSpec) (sts : Array Stmt) : Registry :=
  { R with sessions := R.sessions.push { sid, spec, sts } }

/-- Session `sid`'s `Commit` answered the coins `c`. -/
def Registry.link (R : Registry) (sid : Nat) (c : Array F128) : Registry :=
  { R with sessions := R.sessions.map fun s => if s.sid == sid then { s with link := some c } else s }

/-- The frame of table `i`'s rep `r` in session `sid`. -/
def Registry.find (R : Registry) (sid i r : Nat) : Option (Nat × Rep) := do
  let j ← R.reps.findIdx? fun p => p.sid == sid && p.table == i && p.rep == r
  return (j, ← R.reps[j]?)

/-- That frame, its state set to `p`. -/
def Registry.put (R : Registry) (p : Rep) : Registry :=
  { R with reps := R.reps.map fun q => if q.sid == p.sid && q.table == p.table && q.rep == p.rep then p else q }

/-- **Table `i`'s rep `r` of the session named `name` starts**, for the request line `line` on `ch`: `rep NAME I R` is
answered `ok N` followed by the `N` bytes of the rep's draws and the pads' root (`Frame.Draws.toBytes`), drawn from the
operating system at the statement's start (`Drive.start`); a rep's frame starts once. Anything else is answered
`err WHY`. -/
def startRep (ch : Frame.Chan) (reg : IO.Ref Registry) (l0s : IO.Ref Level0.Registry) (line : String) : IO Unit := do
  let say := fun (s : String) => ch.write (s ++ "\n").toUTF8
  let err := fun (why : String) => say s!"err {why.replace "\n" " "}"
  match (line.replace "\n" "").splitOn " " with
  | ["rep", name, i, r] =>
    let (some i, some r) := (i.toNat?, r.toNat?) | err s!"a request `{line.trim}`"
    if r > 1 then return (← err s!"rep {r}: a table has reps 0 and 1")
    let some (_, sid) := (← l0s.get).sessions.find? (·.1 == name)
      | err s!"a rep for a session the loop is not serving (coin tree root {name})"
    let some s := (← reg.get).sessions.find? (·.sid == sid) | err s!"a rep of session {sid} before its Hello"
    let some st := s.sts[i]? | err s!"table {i} of the session's {s.sts.size}"
    let (lay, mask) ← match Drive.start st with
      | .ok x => pure x
      | .error e => return (← err s!"the rep's start: {e}")
    let d ← match ← Frame.drawsWith IO.getRandomBytes lay with
      | .ok d => pure d
      | .error e => return (← err s!"the rep's start: {e}")
    let c := Frame.Cfg.ofDraws lay d (some mask)
    if let .error e := c.check then return (← err s!"the rep's start: {e}")
    let added ← reg.modifyGet fun R =>
      if (R.find sid i r).isSome then (false, R) else (true, { R with reps := R.reps.push { sid, table := i, rep := r, cfg := c } })
    unless added do return (← err s!"table {i} rep {r}'s frame started already")
    let b := d.toBytes c.root
    say s!"ok {b.size}"
    ch.write b
  | _ => err s!"a request `{line.trim}`"

/-- What stands in rep `p`'s next round's place, and `p` after: the algebraic phase's scheduled round, then the
opening's (`Drive.openStep` begun at its first round, then at the round its last coins asked for), then none. -/
def placeOf (reg : IO.Ref Registry) (l0 : Level0.Level0) (s : Session) (p : Rep) (dom : ByteArray) :
    IO (Except String (Frame.Place × Rep)) := do
  let k := p.coins.size
  if h : k < p.alg.size then return .ok (.sched p.alg[k].1, p)
  if p.opened then return .ok (.own, p)
  let mut p := p
  if p.opening.isNone then
    let some st := s.sts[p.table]? | return .error s!"table {p.table} of the session's {s.sts.size}"
    let some c := s.link | return .error "the opening before the session's Commit coins"
    let some (_, _, hi) := (← reg.get).salts.find? fun (r, q, _) => r == l0.root && q == p.rep
      | return .error s!"table {p.table} rep {p.rep}'s opening before its salts for levels ≥ 1"
    let t ← match tabOfLevel0 (Setup.ofCircuit st) l0 with
      | .ok t => pure t
      | .error e => return .error e
    let xs := Drive.extras st (Drive.linkPoints st c)
    p := { p with opening := some (Drive.openStep t p.rep dom xs hi (p.coins.extract 0 p.alg.size)) }
  match p.opening with
  | some (.ask c _ _) => return .ok (.sched c, { p with toOpening := true })
  | some (.done _) => return .ok (.own, { p with opened := true, opening := none })
  | some (.fail e) => return .error e
  | none => return .error "no opening"

/-- The algebraic phase's scheduled rounds of a rep of the statement `st` whose stream's domain is `dom`, on its level-0
root `root` (`Skeleton.rounds`). -/
def algOf (st : Stmt) (root dom : ByteArray) : V (Array (ByteArray × Nat)) := do
  let (sch, _) ← Zk.schedule st.m
  Skeleton.rounds (Setup.ofCircuit st) sch.k0 root dom

/-- **Session `sid`'s device round `content` of table `i`'s rep `r`**, squeezing `n` coins, on a stream whose domain
frame is `dom`, the stream's first round opening with it: the round that leaves in its place, the frame's release
(`Frame.step`, which takes the first round whole and releases it without the domain frame), after the stream's domain
frame at its first round; or why none does. -/
def round (reg : IO.Ref Registry) (l0s : IO.Ref Level0.Registry) (sid i r : Nat) (dom content : ByteArray) (n : Nat) :
    IO (Except String ByteArray) := do
  let R ← reg.get
  let some (_, p) := R.find sid i r | return .error s!"a round of table {i} rep {r}, whose frame the loop has not started"
  let some s := R.sessions.find? (·.sid == sid) | return .error s!"a round of session {sid}, which the loop is not serving"
  let some st := s.sts[i]? | return .error s!"table {i} of the session's {s.sts.size}"
  let some name := s.spec.tables[i]? | return .error s!"table {i} of the session's {s.spec.tables.size}"
  let some l0 := Level0.ofSession (← l0s.get) sid name | return .error s!"a round of {name} before the session's Commit"
  let first := p.coins.isEmpty && !p.st.begun
  let mut p := p
  let c := content
  if first then
    if (c.extract 0 dom.size).data != dom.data then return .error "a stream's first round without its domain frame"
    let alg ← match algOf st l0.root (s.spec.domain name r).toUTF8 with
      | .ok a => pure a
      | .error e => return .error e
    p := { p with alg }
    if r == 1 then
      let some (_, p0) := R.find sid i 0 | return .error "rep 1's first round before rep 0's frame"
      if p0.coins.size < alg.size then return .error "rep 1's first round before rep 0's algebraic phase ended"
      let prior := (p0.coins.extract 0 alg.size).foldl (· ++ ·) #[]
      match Frame.step p.cfg p.st (.prior prior.size (wordBytes prior)) with
      | (.ok, st') => p := { p with st := st' }
      | (a, _) => return .error s!"rep 0's algebraic-phase coins: {repr' a}"
  let (place, p') ← match ← placeOf reg l0 s p (s.spec.domain name r).toUTF8 with
    | .ok x => pure x
    | .error e => return .error e
  p := p'
  match Frame.step p.cfg p.st (.round n c place) with
  | (.round out v, st') =>
    if let (none, some (q, w, k)) := (p.st.m.rank, st'.m.rank) then
      IO.eprintln s!"FRAMERANK\t\{\"points\": {q}, \"mask_words\": {w}, \"rank\": {k}}"
    if v.startsWith " fail" then
      IO.eprintln s!"frames: session {sid} table {i} rep {r} round {p.coins.size} stops the rep:{v.drop 5}"
    reg.modify (·.put { p with st := st' })
    return .ok (if first then dom ++ out else out)
  | (a, st') =>
    reg.modify (·.put { p with st := st' })
    return .error (repr' a)
where
  repr' : Frame.Reply → String
    | .err e | .refuse e | .stop e => e
    | .ok => "ok"
    | .round _ v => v
    | .done s => if s then "stop" else "ok"

/-- **The coins `cs` session `sid`'s stream answered table `i`'s rep `r`'s round**: the frame takes them (`Frame.step`),
and the opening too where the round was its; or why the frame does not, the rep's stop. -/
def coins (reg : IO.Ref Registry) (sid i r : Nat) (cs : Array F128) : IO (Except String Unit) := do
  let some (_, p) := (← reg.get).find sid i r | return .error s!"coins of table {i} rep {r}, whose frame the loop has not started"
  match Frame.step p.cfg p.st (.coins (.ok cs)) with
  | (.ok, st') =>
    let opening := match p.toOpening, p.opening with
      | true, some (.ask _ _ k) => some (k cs)
      | _, o => o
    reg.modify (·.put { p with st := st', coins := p.coins.push cs, toOpening := false, opening })
    return .ok ()
  | (.stop e, _) | (.refuse e, _) | (.err e, _) => return .error e
  | _ => return .error "coins the frame answered as no coins"

/-- **Whether session `sid`'s `Finish` may leave**: every rep of its `tables` tables has a frame, which answers `finish`
at its end, not stopped; else why not. -/
def finish (reg : IO.Ref Registry) (sid tables : Nat) : IO (Except String Unit) := do
  let R ← reg.get
  for i in [0:tables] do
    for r in [0:2] do
      let some (_, p) := R.find sid i r | return .error s!"a Finish where table {i} rep {r}'s frame never started"
      match Frame.step p.cfg p.st .finish with
      | (.done false, _) => pure ()
      | (.done true, _) => return .error s!"{Frame.STOP}the Lean frame stopped table {i} rep {r}"
      | (.refuse why, _) | (.err why, _) | (.stop why, _) => return .error s!"table {i} rep {r}: {why}"
      | _ => return .error s!"table {i} rep {r}: no answer to finish"
  return .ok ()

/-- Session `sid` ended: its frames, their openings and the salts of its tables' level 0 (`roots`) are gone. -/
def close (reg : IO.Ref Registry) (sid : Nat) (roots : Array ByteArray) : IO Unit :=
  reg.modify fun R =>
    { sessions := R.sessions.filter (·.sid != sid), reps := R.reps.filter (·.sid != sid),
      salts := R.salts.filter fun (root, _, _) => !roots.contains root }

end Flock.Firewall.Frames
