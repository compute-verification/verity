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
* `ask`: the prover learns each rep's next place (`place`), so that a rep whose device stopped goes on with the
  schedule's rounds, which the frame stands in for, and the values the frame took (`vals`), which its finish request
  carries: the prover keeps no schedule and no check of its own.

A round and a round's coins are pure steps of the loop's frames (`roundStep`, `coinsStep`, on the `Sessions` as a
value); `round` and `coins` only read the sessions, run the step, put the rep's frame back and log. So what a rep's frame
takes is a function of the loop's frames and the request alone.

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
  /-- the draws the frame took at the rep's start, which the finish holds the rep's requests to (`ownOf`) -/
  draws : Frame.Draws := default
  st : Frame.State := {}
  alg : Array (ByteArray × Nat) := #[]
  coins : Array (Array F128) := #[]
  /-- the rep's opening, at its next round (`Drive.openStep`) -/
  opening : Option (Drive.Step Unit) := none
  /-- the opening ended: every round after it is the inner proof's -/
  opened : Bool := false
  /-- the round last released was the opening's, whose coins go to it too -/
  toOpening : Bool := false
  /-- the algebraic phase's values the frame took, in its order: a passed device round's, `0` in a round it failed -/
  vals : Array F128 := #[]

/-- A session the loop serves frames for: its statement, and the `Commit`'s coins once they were answered. -/
structure Release where
  sid : Nat
  spec : SessionSpec
  sts : Array Stmt
  link : Option (Array F128) := none

/-- **The loop's frames**: the sessions, each rep's frame, and each rep's salts for levels ≥ 1 and its proof file by its
table's level-0 root, the loop's finish's (`Finish.serve`'s `keep` and `file`). -/
structure Sessions where
  sessions : Array Release := #[]
  reps : Array Rep := #[]
  salts : Array (ByteArray × Nat × Array ByteArray) := #[]
  files : Array (ByteArray × Nat × ByteArray) := #[]

/-- Rep `r`'s proof file `b` on the table of level-0 root `root`, as the loop's finish computed it; a rep's file is kept
once, and asking again is refused. -/
def Sessions.file (R : Sessions) (root : ByteArray) (r : Nat) (b : ByteArray) : Except String Sessions :=
  if R.files.any fun (x, q, _) => x == root && q == r then
    .error s!"rep {r}'s proof file of its table was computed already"
  else .ok { R with files := R.files.push (root, r, b) }

/-- Session `sid`, set up on `spec` and `sts` at its `Hello`. -/
def Sessions.begin (R : Sessions) (sid : Nat) (spec : SessionSpec) (sts : Array Stmt) : Sessions :=
  { R with sessions := R.sessions.push { sid, spec, sts } }

/-- Session `sid`'s `Commit` answered the coins `c`. -/
def Sessions.link (R : Sessions) (sid : Nat) (c : Array F128) : Sessions :=
  { R with sessions := R.sessions.map fun s => if s.sid == sid then { s with link := some c } else s }

/-- The frame of table `i`'s rep `r` in session `sid`. -/
def Sessions.find (R : Sessions) (sid i r : Nat) : Option (Nat × Rep) := do
  let j ← R.reps.findIdx? fun p => p.sid == sid && p.table == i && p.rep == r
  return (j, ← R.reps[j]?)

/-- That frame, its state set to `p`. -/
def Sessions.put (R : Sessions) (p : Rep) : Sessions :=
  { R with reps := R.reps.map fun q => if q.sid == p.sid && q.table == p.table && q.rep == p.rep then p else q }

/-- **Table `i`'s rep `r` of the session named `name` starts**, for the request line `line` on `ch`: `rep NAME I R` is
answered `ok N` followed by the `N` bytes of the rep's draws and the pads' root (`Frame.Draws.toBytes`), drawn from the
operating system at the statement's start (`Drive.start`); a rep's frame starts once. Anything else is answered
`err WHY`. -/
def startRep (ch : Frame.Chan) (reg : IO.Ref Sessions) (l0s : IO.Ref Level0.Slots) (line : String) : IO Unit := do
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
      if (R.find sid i r).isSome then (false, R) else (true, { R with reps := R.reps.push { sid, table := i, rep := r, cfg := c, draws := d } })
    unless added do return (← err s!"table {i} rep {r}'s frame started already")
    let b := d.toBytes c.root
    say s!"ok {b.size}"
    ch.write b
  | _ => err s!"a request `{line.trim}`"

/-- What stands in rep `p`'s next round's place, the coins it squeezes, and `p` after, at the loop's salts `salts`: the
algebraic phase's scheduled round, then the opening's (`Drive.openStep` begun at its first round, then at the round its
last coins asked for), then none (and no coins). -/
def placeOf (salts : Array (ByteArray × Nat × Array ByteArray)) (l0 : Level0.Level0) (s : Release) (p : Rep)
    (dom : ByteArray) : Except String (Frame.Place × Nat × Rep) := do
  let k := p.coins.size
  if h : k < p.alg.size then return (.sched p.alg[k].1, p.alg[k].2, p)
  if p.opened then return (.own, 0, p)
  let mut p := p
  if p.opening.isNone then
    let some st := s.sts[p.table]? | throw s!"table {p.table} of the session's {s.sts.size}"
    let some c := s.link | throw "the opening before the session's Commit coins"
    let some (_, _, hi) := salts.find? fun (r, q, _) => r == l0.root && q == p.rep
      | throw s!"table {p.table} rep {p.rep}'s opening before its salts for levels ≥ 1"
    let t ← tabOfLevel0 (Setup.ofCircuit st) l0
    let xs := Drive.extras st (Drive.linkPoints st c)
    p := { p with opening := some (Drive.openStep t p.rep dom xs hi (p.coins.extract 0 p.alg.size)) }
  match p.opening with
  | some (.ask c n _) => return (.sched c, n, { p with toOpening := true })
  | some (.done _) => return (.own, 0, { p with opened := true, opening := none })
  | some (.fail e) => throw e
  | none => throw "no opening"

/-- The algebraic phase's scheduled rounds of a rep of the statement `st` whose stream's domain is `dom`, on its level-0
root `root` (`Skeleton.rounds`). -/
def algOf (st : Stmt) (root dom : ByteArray) : V (Array (ByteArray × Nat)) := do
  let (sch, _) ← Zk.schedule st.m
  Skeleton.rounds (Setup.ofCircuit st) sch.k0 root dom

/-- `p` with its algebraic phase's scheduled rounds (`algOf`) on the session's level 0 `l0`, the table being `name`,
computed once. -/
def withAlg (s : Release) (l0 : Level0.Level0) (name : String) (p : Rep) : V Rep := do
  if !p.alg.isEmpty then return p
  let some st := s.sts[p.table]? | throw s!"table {p.table} of the session's {s.sts.size}"
  return { p with alg := ← algOf st l0.root (s.spec.domain name p.rep).toUTF8 }

/-- The values the frame takes from the device round `content` at the free offsets `free` of the scheduled round in
its place (`Frame.roundFree`), by its verdict `v`: the device's where it passed, else `0` each. -/
def valsOf (content : ByteArray) (free : Array Nat) (v : String) : Array F128 :=
  if v == " pass" then free.map fun o => (Frame.words content o 1)[0]! else free.map fun _ => 0

/-- Why the frame's answer `a` released nothing. -/
def replyWhy : Frame.Reply → String
  | .err e | .refuse e | .stop e => e
  | .ok => "ok"
  | .round _ v => v
  | .done s => if s then "stop" else "ok"

/-- **The frame and place of session `sid`'s round `content` of table `i`'s rep `r`**, in the loop's frames `R` and
level 0s `L0`, on a stream whose domain frame is `dom`: the rep's frame ready to take it (at the stream's first round, its
algebraic phase's scheduled rounds on the session's level 0, `withAlg`, and rep 1's after rep 0's algebraic-phase coins),
the round's place (`placeOf`) and whether it is the stream's first round; or why the round has none. -/
def roundPlace (R : Sessions) (L0 : Level0.Slots) (sid i r : Nat) (dom content : ByteArray) :
    Except String (Rep × Frame.Place × Bool) := do
  let some (_, p) := R.find sid i r | throw s!"a round of table {i} rep {r}, whose frame the loop has not started"
  let some s := R.sessions.find? (·.sid == sid) | throw s!"a round of session {sid}, which the loop is not serving"
  let some _ := s.sts[i]? | throw s!"table {i} of the session's {s.sts.size}"
  let some name := s.spec.tables[i]? | throw s!"table {i} of the session's {s.spec.tables.size}"
  let some l0 := Level0.ofSession L0 sid name | throw s!"a round of {name} before the session's Commit"
  let first := p.coins.isEmpty && !p.st.begun
  let mut p := p
  if first then
    if (content.extract 0 dom.size).data != dom.data then throw "a stream's first round without its domain frame"
    p ← withAlg s l0 name p
    if r == 1 then
      let some (_, p0) := R.find sid i 0 | throw "rep 1's first round before rep 0's frame"
      if p0.coins.size < p.alg.size then throw "rep 1's first round before rep 0's algebraic phase ended"
      let prior := (p0.coins.extract 0 p.alg.size).foldl (· ++ ·) #[]
      match Frame.step p.cfg p.st (.prior prior.size (wordBytes prior)) with
      | (.ok, st') => p := { p with st := st' }
      | (a, _) => throw s!"rep 0's algebraic-phase coins: {replyWhy a}"
  let (place, _, p') ← placeOf R.salts l0 s p (s.spec.domain name r).toUTF8
  return (p', place, first)

/-- A round's effect on the loop: what leaves (or why nothing does), the rep's frame after it, which the loop puts back,
the rank check's result where the round made it, and the round and why the rep stopped where the round stopped it. -/
structure RoundStep where
  out : Except String ByteArray
  put : Option Rep := none
  rank : Option (Nat × Nat × Nat) := none
  stops : Option (Nat × String) := none

/-- **Session `sid`'s device round `content` of table `i`'s rep `r`**, squeezing `n` coins, in the loop's frames `R` and
level 0s `L0`, on a stream whose domain frame is `dom`, the stream's first round opening with it: the frame's release
(`Frame.step`, which takes the first round whole and releases it without the domain frame), after the stream's domain
frame at its first round, at the round's place (`roundPlace`). -/
def roundStep (R : Sessions) (L0 : Level0.Slots) (sid i r : Nat) (dom content : ByteArray) (n : Nat) : RoundStep :=
  match roundPlace R L0 sid i r dom content with
  | .error e => { out := .error e }
  | .ok (p, place, first) =>
    let free := match place with
      | .sched sr => match Frame.roundFree p.cfg p.st.m sr n with
        | .ok (_, _, f) => f
        | .error _ => #[]
      | _ => #[]
    match Frame.step p.cfg p.st (.round n content place) with
    | (.round out v, st') =>
      { out := .ok (if first then dom ++ out else out)
        put := some { p with st := st', vals := p.vals ++ valsOf content free v }
        rank := (match p.st.m.rank, st'.m.rank with
          | none, some k => some k
          | _, _ => none)
        stops := if v.startsWith " fail" then some (p.coins.size, s!"{v.drop 5}") else none }
    | (a, st') => { out := .error (replyWhy a), put := some { p with st := st' } }

/-- **Session `sid`'s device round `content` of table `i`'s rep `r`** in the loop: `roundStep` on the loop's frames and
level 0s, its rep's frame put back, and its rank check and stop logged; the round that leaves in its place, or why none
does. -/
def round (reg : IO.Ref Sessions) (l0s : IO.Ref Level0.Slots) (sid i r : Nat) (dom content : ByteArray) (n : Nat) :
    IO (Except String ByteArray) := do
  let o := roundStep (← reg.get) (← l0s.get) sid i r dom content n
  if let some (q, w, k) := o.rank then
    IO.eprintln s!"FRAMERANK\t\{\"points\": {q}, \"mask_words\": {w}, \"rank\": {k}}"
  if let some (k, why) := o.stops then
    IO.eprintln s!"frames: session {sid} table {i} rep {r} round {k} stops the rep:{why}"
  if let some p := o.put then reg.modify (·.put p)
  return o.out

/-- **The coins `cs` session `sid`'s stream answered table `i`'s rep `r`'s round**, in the loop's frames `R`: the rep's
frame after the frame takes them (`Frame.step`), and the opening too where the round was its; or why the frame does not,
the rep's stop. -/
def coinsStep (R : Sessions) (sid i r : Nat) (cs : Array F128) : Except String Rep := do
  let some (_, p) := R.find sid i r | throw s!"coins of table {i} rep {r}, whose frame the loop has not started"
  match Frame.step p.cfg p.st (.coins (.ok cs)) with
  | (.ok, st') =>
    let opening := match p.toOpening, p.opening with
      | true, some (.ask _ _ k) => some (k cs)
      | _, o => o
    return { p with st := st', coins := p.coins.push cs, toOpening := false, opening }
  | (.stop e, _) | (.refuse e, _) | (.err e, _) => throw e
  | _ => throw "coins the frame answered as no coins"

/-- The coins `cs` in the loop: `coinsStep` on the loop's frames, the rep's frame after them put back. -/
def coins (reg : IO.Ref Sessions) (sid i r : Nat) (cs : Array F128) : IO (Except String Unit) := do
  match coinsStep (← reg.get) sid i r cs with
  | .ok p =>
    reg.modify (·.put p)
    return .ok ()
  | .error e => return .error e

/-- **Whether session `sid`'s `Finish` may leave**: every rep of its `tables` tables has a frame, which answers `finish`
at its end, not stopped; else why not. -/
def finish (reg : IO.Ref Sessions) (sid tables : Nat) : IO (Except String Unit) := do
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

/-- **What the loop holds of table `i`'s rep `r`** on the level 0 of root `root`, in its frames `R` and level 0s `L0`, which
the loop's finish holds the rep's requests to (`Finish.Own`): the draws its frame took at its start, the algebraic phase's
values the frame took at their pads (`Frame.absorbed`'s order, the rest `0`), the extra claims at the session's link points
(`Drive.extras`, as `placeOf` opens with them), and the coins the stream answered each of the rep's rounds so far. -/
def ownOf (R : Sessions) (L0 : Level0.Slots) (i r : Nat) (root : ByteArray) : Except String Finish.Own := do
  let some sl := L0.slots.find? (·.l0.root == root) | throw s!"table {i}: a level 0 the session loop did not draw"
  let some (sid, name) := sl.bound | throw s!"table {i}: a level 0 no session's Commit carries"
  let some s := R.sessions.find? (·.sid == sid) | throw s!"table {i}: session {sid}, which the loop is not serving"
  unless s.spec.tables[i]? == some name do throw s!"table {i}: the level 0 of the session's table {name}"
  let some st := s.sts[i]? | throw s!"table {i} of the session's {s.sts.size}"
  let some c := s.link | throw "a rep's request before the session's Commit coins"
  let some (_, p) := R.find sid i r | throw s!"a request of table {i} rep {r}, whose frame the loop has not started"
  let vals := (p.cfg.order.zip p.vals).foldl (fun v (k, x) => v.set! k x) (Array.replicate (p.cfg.lay.zp 64) 0)
  return { draws := p.draws, vals, xs := Drive.extras st (Drive.linkPoints st c), coins := p.coins }

/-- **What the loop tells the prover of table `i`'s rep `r`** in the live session named `name`, for the request line
`line` on `ch`, past the session's `Commit`: `place NAME I R` is answered by its next round's place (`placeOf`), `alg N`
or `open N` for the algebraic phase's or the opening's scheduled round, squeezing `N` coins, else `own`; `vals NAME I R`
by `ok N` followed by the `N` bytes of the values the frame took so far (`Rep.vals`). A prover whose device stops sends
each place's round empty (its first, the domain frame alone): the frame fails it, and the scheduled round leaves in its
place. Anything else is answered `err WHY`. -/
def ask (ch : Frame.Chan) (reg : IO.Ref Sessions) (l0s : IO.Ref Level0.Slots) (line : String) : IO Unit := do
  let say := fun (s : String) => ch.write (s ++ "\n").toUTF8
  let err := fun (why : String) => say s!"err {why.replace "\n" " "}"
  match (line.replace "\n" "").splitOn " " with
  | [what, name, i, r] =>
    let (some i, some r) := (i.toNat?, r.toNat?) | err s!"a request `{line.trim}`"
    unless what == "place" || what == "vals" do return (← err s!"a request `{line.trim}`")
    let some (_, sid) := (← l0s.get).sessions.find? (·.1 == name)
      | err s!"a {what} for a session the loop is not serving (coin tree root {name})"
    let R ← reg.get
    let some s := R.sessions.find? (·.sid == sid) | err s!"a {what} of session {sid} before its Hello"
    let some (_, p) := R.find sid i r | err s!"a {what} of table {i} rep {r}, whose frame the loop has not started"
    if what == "vals" then
      let b := wordBytes p.vals
      say s!"ok {b.size}"
      return (← ch.write b)
    let some tname := s.spec.tables[i]? | err s!"table {i} of the session's {s.spec.tables.size}"
    let some l0 := Level0.ofSession (← l0s.get) sid tname | err s!"a place of {tname} before the session's Commit"
    let p ← match withAlg s l0 tname p with
      | .ok p => pure p
      | .error e => return (← err e)
    let alg := p.coins.size < p.alg.size
    match placeOf R.salts l0 s p (s.spec.domain tname r).toUTF8 with
    | .error e => err e
    | .ok (place, n, p') =>
      reg.modify (·.put p')
      match place with
      | .sched _ => say s!"{if alg then "alg" else "open"} {n}"
      | _ => say "own"
  | _ => err s!"a request `{line.trim}`"

/-- Session `sid` ended: its frames, their openings and the salts and proof files of its tables' level 0 (`roots`) are
gone. -/
def close (reg : IO.Ref Sessions) (sid : Nat) (roots : Array ByteArray) : IO Unit :=
  reg.modify fun R =>
    { sessions := R.sessions.filter (·.sid != sid), reps := R.reps.filter (·.sid != sid),
      salts := R.salts.filter fun (root, _, _) => !roots.contains root,
      files := R.files.filter fun (root, _, _) => !roots.contains root }

end Flock.Firewall.Frames
