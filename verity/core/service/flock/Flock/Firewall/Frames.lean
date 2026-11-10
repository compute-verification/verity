/-
# The session loop's frames: each rep's frame, driven by the loop on the one schedule

The session loop (`Session`) holds each firewalled rep's frame itself (`Frame.step` at the frame's `Cfg` and `State`) and
issues every request to it, on the schedule `Drive.reqs` fixes. Nothing the prover sends names a frame's request:

* `startRep`: a rep's frame starts at the loop's frames port (`rep NAME TABLE REP`), for a live session whose level 0 the
  loop drew and one of its tables' reps, at the statement's start (`Drive.start`: its pads layout and mask slot), its
  samples from the operating system, which never leave the loop: the prover gets the pads' root alone.
* `round`: a session's `Round` of a rep stream carries the device's round. Its place is the statement's: the algebraic
  phase's scheduled round (`Skeleton.rounds`, on the table's level 0, the loop's own), then the opening's
  (`Drive.openStepM`, the rep's opening a round at a time, on the extra claims at the session's `Commit` coins,
  `Drive.extras`, and the rep's salts for levels ≥ 1, the loop's finish's), then the inner proof's: the frame's own round
  (`Frame.innerSched`) on the rounds the loop's finish computed for the rep before they went to the host
  (`Sessions.inner`, `Finish.serve`'s `inner`), so the host only relays them. A round the finish has not computed has
  no place: whatever the host relays there is refused, and the session stops. A stop is the session's: once any rep's
  frame of it stopped (a failed round, a refused one, a failed worker's rounds; `Sessions.stopped`), the frame takes no
  device round at any of its reps' rounds, and every rep of the session stops, its rounds leaving as the scheduled ones
  of a stopped rep. Rep 1's frame first takes rep 0's algebraic-phase coins (`prior`). What leaves is the frame's
  release, at the stream's first round after its domain frame (`domainFrame`, `Transcript.init`'s).
* `coins`: once the coins answered that round open the committed tree (`State.answer`), the frame takes them
  (`feedOf` the answer), and the opening goes on with them where the round was its; the prover receives them only once the frame did.
* `finish`: the session's `Finish` leaves only once every rep's frame answers `finish` at its end, not stopped
  (`finishOf`).
* `ask`: the prover learns each rep's next place (`place`), so that a rep whose device stopped goes on with the
  schedule's rounds, which the frame stands in for, and the values the frame took (`vals`), which its finish request
  carries: the prover keeps no schedule and no check of its own. Its answer only reads the loop's frames (`askOf`).

A round, a round's coins and a `Finish` are pure steps of the loop's frames (`roundStep`, `coinsStep`, `finishOf`, on
the `Sessions` as a value); `round`, `coins` and `finish` only read the sessions, run the step, put the rep's frame back
and log. Between its start
(`startRep`) and its session's end (`close`) a rep's frame is written by them alone, at its own session's requests, one
at a time, so no other task puts back a frame from before one of them. So what a rep's frame takes is a function of the
loop's frames and the request alone.

So every coin a frame takes is the loop's own reading of the auditor's answer, and every round in a device round's place
is the loop's own, a function of the statement, the table's level 0, the rep's salts and the coins: the opening's are
`Drive.opening`'s, which feeds `Drive.openStep`, the loop's `Drive.openStepM` but for what it ends with, the same coins
(`Drive.Step.feed`). What it ends with, the opening's switched vectors and batched basis (`Finish.Memo`), the rep's
proof reads back (`ownOf`) rather than computing them again.
-/
module

public import Flock.Firewall.Drive

@[expose] public section

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
  /-- the samples the frame took at the rep's start, which the finish holds the rep's requests to (`ownOf`) -/
  draws : Frame.Draws := default
  st : Frame.State := {}
  alg : Array (ByteArray × Nat) := #[]
  coins : Array (Array F128) := #[]
  /-- the rep's opening, at its next round (`Drive.openStepM`); once it ended, until the inner proof's first round, what
  it computed that the rep's proof reads back (`ownOf`) -/
  opening : Option (Drive.Step Memo) := none
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

/-- **The loop's frames**: the sessions, each rep's frame, and each rep's salts for levels ≥ 1, its inner proof's rounds
and its proof file by its table's level-0 root, the loop's finish's (`Finish.serve`'s `keep`, `inner` and `file`). -/
structure Sessions where
  sessions : Array Release := #[]
  reps : Array Rep := #[]
  salts : Array (ByteArray × Nat × Array ByteArray) := #[]
  files : Array (ByteArray × Nat × ByteArray) := #[]
  inner : Array (ByteArray × Nat × Array ByteArray) := #[]

/-- Rep `r`'s proof file `b` on the table of level-0 root `root`, as the loop's finish computed it; a rep's file is kept
once, and asking again is refused. -/
def Sessions.file (R : Sessions) (root : ByteArray) (r : Nat) (b : ByteArray) : Except String Sessions :=
  if R.files.any fun (x, q, _) => x == root && q == r then
    .error s!"rep {r}'s proof file of its table was computed already"
  else .ok { R with files := R.files.push (root, r, b) }

/-- The inner proof's rounds the loop's finish computed for rep `r` on the table of level-0 root `root`, so far. -/
def Sessions.innerOf (R : Sessions) (root : ByteArray) (r : Nat) : Array ByteArray :=
  ((R.inner.find? fun (x, q, _) => x == root && q == r).map (·.2.2)).getD #[]

/-- Rep `r`'s inner proof's round `j`, `b`, on the table of level-0 root `root`, as the loop's finish computed it before
it went to the host: the rounds come in order, from one inner proof, each once; any other is refused. -/
def Sessions.innerRound (R : Sessions) (root : ByteArray) (r j : Nat) (b : ByteArray) : Except String Sessions :=
  let got := R.innerOf root r
  if got.size != j then
    .error s!"rep {r}'s inner round {j}, where the loop's finish computed {got.size} of its table's rounds already"
  else if j == 0 then .ok { R with inner := R.inner.push (root, r, #[b]) }
  else .ok { R with inner := R.inner.map fun (x, q, rs) => if x == root && q == r then (x, q, rs.push b) else (x, q, rs) }

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

/-- **Whether session `sid` stopped** in the loop's frames `R`: a frame of one of its reps stopped (`Masker.stopped`: a
round of the rep failed, or the frame refused one), from which on every rep of the session stops (`roundPlace`). -/
def Sessions.stopped (R : Sessions) (sid : Nat) : Bool := R.reps.any fun p => p.sid == sid && p.st.m.stopped

/-- **The live session named `name`** (`Level0.nameOf`) in the loop's frames `R` and level 0s `L0`, for a request `what`;
else why not: a name no live session has, or a session before its `Hello`. -/
def Sessions.named (R : Sessions) (L0 : Level0.Slots) (what name : String) : Except String Release := do
  let some (_, sid) := L0.sessions.find? (·.1 == name)
    | throw s!"a {what} for a session the loop is not serving (id {name})"
  let some s := R.sessions.find? (·.sid == sid) | throw s!"a {what} of session {sid} before its Hello"
  return s

/-- **Where the loop writes words of its own into table `j`'s packed witness** for a level 0 at `m` of the session named
`name`: the mask slot of the table's statement (`Drive.maskSlot`) and its `kLog`; else why not: a session the loop is not
serving, a table it does not have, or an `m` other than the statement's. -/
def Sessions.maskOf (R : Sessions) (L0 : Level0.Slots) (name : String) (j m : Nat) :
    Except String (Frame.MaskSlot × Nat) := do
  let s ← R.named L0 "level-0 request" name
  let some st := s.sts[j]? | throw s!"a level-0 request for table {j} of a session with {s.sts.size} tables"
  if m != st.m then throw s!"a level-0 request at m = {m}, where table {j}'s statement is at m = {st.m}"
  return (← Drive.maskSlot st, st.c.kLog)

/-- **Table `i`'s rep `r` of the session named `name` starts**, for the request line `line` on `ch`: `rep NAME I R` is
answered `ok 64` followed by the pads' root (`Frame.padsRoot`) over the rep's samples, sampled from the operating system at
the statement's start (`Drive.start`) and kept in the rep's frame, never sent; a rep's frame starts once, and only once
the session's level 0 is sampled (`Level0.Slots.drawn`), so after every table's witness is in (`Session.level0`). Anything
else is answered `err WHY`. -/
def startRep (ch : Frame.Chan) (reg : IO.Ref Sessions) (l0s : IO.Ref Level0.Slots) (line : String) : IO Unit := do
  let say := fun (s : String) => ch.write (s ++ "\n").toUTF8
  let err := fun (why : String) => say s!"err {why.replace "\n" " "}"
  match (line.replace "\n" "").splitOn " " with
  | ["rep", name, i, r] =>
    let (some i, some r) := (i.toNat?, r.toNat?) | err s!"a request `{line.trim}`"
    if r > 1 then return (← err s!"rep {r}: a table has reps 0 and 1")
    let s ← match (← reg.get).named (← l0s.get) "rep" name with
      | .ok s => pure s
      | .error e => return (← err e)
    let sid := s.sid
    let some st := s.sts[i]? | err s!"table {i} of the session's {s.sts.size}"
    unless (← l0s.get).drawn sid do
      return (← err s!"table {i} rep {r} before the session's level 0: a rep's frame starts once every table's witness is in")
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
    say s!"ok {c.root.size}"
    ch.write c.root
  | _ => err s!"a request `{line.trim}`"

/-- What stands in rep `p`'s next round's place, the coins it squeezes, and `p` after, at the loop's salts `salts`: the
algebraic phase's scheduled round, then the opening's (`Drive.openStepM` begun at its first round, then at the round its
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
    p := { p with opening := some (Drive.openStepM t p.rep dom xs hi (p.coins.extract 0 p.alg.size)) }
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

/-! ## The lincheck fold's refusal alone

`Finish.lincheckFw` takes of its circuit fold only whether it refuses, and why: it discards the fold's value. The skeleton
(`algOf`) and the opening (`placeOf`) run it on the statement's setup with the fold `foldChk`, which computes that refusal
alone (`foldWhy`), in place of `Stmt.fold`'s `2^k_log` products; each is proved equal to its definition, so every round
and coin is the same. -/

/-- The refusal `why`, or none. -/
def orOk : Option String → V Unit
  | some e => .error e
  | none => .ok ()

theorem orOk_error {α : Type} {e : String} {w : Option String}
    (h : (fun _ => ()) <$> (Except.error e : V α) = orOk w) : w = some e := by
  cases w with
  | none =>
    have h' : (Except.error e : V Unit) = .ok () := h
    cases h'
  | some e' =>
    have h' : (Except.error e : V Unit) = .error e' := h
    cases h'
    rfl

theorem orOk_ok {α : Type} {b : α} {w : Option String} (h : (fun _ => ()) <$> (Except.ok b : V α) = orOk w) :
    w = none := by
  cases w with
  | none => rfl
  | some e' =>
    have h' : (Except.ok () : V Unit) = .error e' := h
    cases h'

/-- Why `Stmt.slotBase` refuses the slot type `key`, or none. -/
def slotWhy (st : Stmt) (key : Kind × Nat) : Option String :=
  match key.1 with
  | .comp => if key.2 != st.leaf.comp.spec.kLog then some "lincheck fold: a compression slot of the wrong size" else none
  | .net i =>
    if (st.c.nets.getD i default).2.unitLog != key.2 then some "lincheck fold: a unit slot of the wrong size" else none
  | .mask => none

/-- Why `Stmt.foldRanges` refuses the ranges `rs`: the first range's slot type `slotWhy` refuses. -/
def rangesWhy (st : Stmt) : List Range → Option String
  | [] => none
  | r :: rs => match slotWhy st (r.kind, r.slotLog) with
    | some e => some e
    | none => rangesWhy st rs

/-- Why `Stmt.fold` refuses the point `lw`, `rho`, or none. -/
def foldWhy (st : Stmt) (lw rho : Array F128) : Option String :=
  if rho.size + 6 != st.c.kLog || lw.size != 64 then some "lincheck fold: the point is not the block's"
  else rangesWhy st st.c.ranges.toList

theorem slotBase_map (st : Stmt) (a : F128) (lw rho : Array F128) (key : Kind × Nat) :
    (fun _ => ()) <$> st.slotBase a lw rho key = orOk (slotWhy st key) := by
  obtain ⟨k, sl⟩ := key
  cases k with
  | comp =>
    simp only [Stmt.slotBase, slotWhy]
    cases (sl != st.leaf.comp.spec.kLog) <;> rfl
  | net i =>
    simp only [Stmt.slotBase, slotWhy]
    cases ((st.c.nets.getD i default).2.unitLog != sl) <;> rfl
  | mask => rfl

theorem foldRanges_map (st : Stmt) (a : F128) (lw rho : Array F128) :
    ∀ (rs : List Range) (bases : Array ((Kind × Nat) × Array F128)) (out : Array F128),
      (∀ kb ∈ bases, slotWhy st kb.1 = none) →
        (fun _ => ()) <$> st.foldRanges a lw rho rs bases out = orOk (rangesWhy st rs)
  | [], _, _, _ => rfl
  | r :: rs, bases, out, hb => by
    simp only [Stmt.foldRanges]
    cases hc : bases.find? (fun kb => decide (kb.1 = (r.kind, r.slotLog))) with
    | some kb =>
      obtain ⟨key, b⟩ := kb
      have h1 := Array.find?_some hc
      have hk : key = (r.kind, r.slotLog) := of_decide_eq_true h1
      have hw : slotWhy st (r.kind, r.slotLog) = none := by
        rw [← hk]
        exact hb _ (Array.mem_of_find?_eq_some hc)
      rw [rangesWhy, hw]
      exact foldRanges_map st a lw rho rs bases _ hb
    | none =>
      have hs := slotBase_map st a lw rho (r.kind, r.slotLog)
      cases hb' : st.slotBase a lw rho (r.kind, r.slotLog) with
      | error e =>
        rw [hb'] at hs
        rw [rangesWhy, orOk_error hs]
        rfl
      | ok b =>
        rw [hb'] at hs
        rw [rangesWhy, orOk_ok hs]
        refine foldRanges_map st a lw rho rs (bases.push ((r.kind, r.slotLog), b)) _ fun kb hkb => ?_
        rcases Array.mem_push.mp hkb with h | h
        · exact hb kb h
        · subst h
          exact orOk_ok hs

theorem fold_map (st : Stmt) (a : F128) (lw rho : Array F128) :
    (fun _ => ()) <$> st.fold a lw rho = orOk (foldWhy st lw rho) := by
  have hr := foldRanges_map st a lw rho st.c.ranges.toList #[] (Array.replicate (2 ^ st.c.kLog) 0)
    fun kb h => absurd h (Array.not_mem_empty kb)
  simp only [Stmt.fold, foldWhy]
  generalize st.foldRanges a lw rho st.c.ranges.toList #[] (Array.replicate (2 ^ st.c.kLog) 0) = x at hr ⊢
  cases (rho.size + 6 != st.c.kLog || lw.size != 64)
  · show _ = orOk (rangesWhy st st.c.ranges.toList)
    rw [← hr]
    cases x <;> rfl
  · rfl

/-- **`Stmt.fold`'s refusal alone** (`foldWhy`), its value `#[]`. -/
def foldChk (st : Stmt) (_alpha : F128) (lw rho : Array F128) : V (Array F128) :=
  (fun _ => #[]) <$> orOk (foldWhy st lw rho)

theorem foldChk_map (st : Stmt) (a : F128) (lw rho : Array F128) :
    (fun _ => ()) <$> foldChk st a lw rho = (fun _ => ()) <$> st.fold a lw rho := by
  rw [fold_map]
  unfold foldChk
  cases foldWhy st lw rho <;> rfl

/-- The statement's setup (`Setup.ofCircuit`) with the fold `foldChk`. -/
def ofCircuitChk (st : Stmt) : Setup := { Setup.ofCircuit st with fold := { st.circuitFold with fold := foldChk st } }

/-- `lincheckFw` reads of its fold only the refusal. -/
theorem lincheckFw_fold {M : Type → Type} [Monad M] [LawfulMonad M] [MonadExceptOf String M] [MonadLiftT V M]
    (io : Asks M) (c c' : CircuitFold) (hk : c.kLog = c'.kLog)
    (hf : ∀ a w r, (fun _ => ()) <$> c.fold a w r = (fun _ => ()) <$> c'.fold a w r)
    (lay : Zk.PadLayout) (mk : Nat → F128) (zc : ZcOut) :
    lincheckFw io c lay mk zc = lincheckFw io c' lay mk zc := by
  obtain ⟨k, p, f, fd⟩ := c
  obtain ⟨k', p', f', fd'⟩ := c'
  change k = k' at hk
  subst hk
  unfold lincheckFw
  dsimp only
  refine bind_congr fun _ => bind_congr fun alpha => bind_congr fun _ => ?_
  have h := hf alpha (skipWeights zc.z) (zc.rho.extract 0 (k - 6))
  dsimp only at h
  revert h
  cases f alpha (skipWeights zc.z) (zc.rho.extract 0 (k - 6)) <;>
    cases f' alpha (skipWeights zc.z) (zc.rho.extract 0 (k - 6)) <;> intro h
  · have h' : (Except.error _ : V Unit) = .error _ := h
    cases h'
    rfl
  · have h' : (Except.error _ : V Unit) = .ok () := h
    cases h'
  · have h' : (Except.ok () : V Unit) = .error _ := h
    cases h'
  · simp only [pure_bind]

theorem repPiop_fold {M : Type → Type} [Monad M] [LawfulMonad M] [MonadExceptOf String M] [MonadLiftT V M]
    (io : Asks M) (S : Setup) (c : CircuitFold) (hk : c.kLog = S.fold.kLog)
    (hf : ∀ a w r, (fun _ => ()) <$> c.fold a w r = (fun _ => ()) <$> S.fold.fold a w r)
    (lay : Zk.PadLayout) (cap root : ByteArray) (extras : Array Claim) (mk : Nat → F128) :
    repPiop io { S with fold := c } lay cap root extras mk = repPiop io S lay cap root extras mk := by
  simp only [repPiop, lincheckFw_fold io c S.fold hk hf]

theorem repOpeningMemo_fold {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]
    (memo : Option Memo) (io : Asks M) (commit : Level → Array F128 → (Nat → ByteArray) → M Commit) (S : Setup)
    (c : CircuitFold) (sch : Flock.Schedule) (lay : Zk.PadLayout) (tPad rep : Nat) (g : Array F128) (cm0 : Commit)
    (claims : Array Claim) (h : Array F128) (extra lanePad : Nat → Nat → F128) (salt : Nat → Nat → ByteArray) :
    repOpeningMemo memo io commit { S with fold := c } sch lay tPad rep g cm0 claims h extra lanePad salt =
      repOpeningMemo memo io commit S sch lay tPad rep g cm0 claims h extra lanePad salt := rfl

theorem rounds_fold (S : Setup) (c : CircuitFold) (hk : c.kLog = S.fold.kLog)
    (hf : ∀ a w r, (fun _ => ()) <$> c.fold a w r = (fun _ => ()) <$> S.fold.fold a w r) (k0 : Nat) (cap dom : ByteArray) :
    Skeleton.rounds { S with fold := c } k0 cap dom = Skeleton.rounds S k0 cap dom := by
  simp only [Skeleton.rounds, Skeleton.head, repPiop_fold Skeleton.recAsks S c hk hf]
  all_goals rfl

instance instLawfulStep : LawfulMonad Drive.Step := LawfulMonad.mk' Drive.Step
  (id_map := fun x => by
    induction x with
    | done _ => rfl
    | fail _ => rfl
    | ask c n k ih => exact congrArg (Drive.Step.ask c n) (funext ih))
  (pure_bind := fun _ _ => rfl)
  (bind_assoc := fun x f g => by
    induction x with
    | done _ => rfl
    | fail _ => rfl
    | ask c n k ih => exact congrArg (Drive.Step.ask c n) (funext ih))

theorem openStepM_fold (t : Tab) (S : Setup) (c : CircuitFold) (hk : c.kLog = S.fold.kLog)
    (hf : ∀ a w r, (fun _ => ()) <$> c.fold a w r = (fun _ => ()) <$> S.fold.fold a w r) (rep : Nat) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (coins : Array (Array F128)) :
    Drive.openStepM { t with st := { S with fold := c } } rep dom xs hi coins =
      Drive.openStepM { t with st := S } rep dom xs hi coins := by
  simp only [Drive.openStepM, repPiop_fold (Drive.stepAsks coins false) S c hk hf, repOpeningMemo_fold]

theorem ok_bind {α β : Type} (a : α) (f : α → V β) : ((Except.ok a : V α) >>= f) = f a := rfl

/-- `tabOfLevel0` reads of its setup only the Merkle scheme and `m`. -/
theorem tabOfLevel0_with (S S' : Setup) (hm : S'.merkle = S.merkle) (hn : S'.m = S.m) (l : Level0.Level0) :
    tabOfLevel0 S' l = (fun t => { t with st := S' }) <$> tabOfLevel0 S l := by
  unfold tabOfLevel0
  rw [hm, hn]
  cases (S.merkle.id != MerkleScheme.hm96Sha512.id)
  · cases (l.m != S.m)
    · cases Zk.schedule S.m with
      | error e => rfl
      | ok p =>
        obtain ⟨sch, tPad⟩ := p
        cases h : (l.tPad != tPad) <;> simp only [ok_bind, h] <;> rfl
    · rfl
  · rfl

/-- A rep's opening (`Drive.openStepM`) on the table's level 0 `l0` at the statement's setup `setup st`. -/
def openingOf (setup : Stmt → Setup) (st : Stmt) (l0 : Level0.Level0) (rep : Nat) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (coins : Array (Array F128)) : V (Drive.Step Memo) :=
  (fun t => Drive.openStepM t rep dom xs hi coins) <$> tabOfLevel0 (setup st) l0

theorem openingOf_chk : openingOf Setup.ofCircuit = openingOf ofCircuitChk := by
  funext st l0 rep dom xs hi coins
  unfold openingOf
  rw [tabOfLevel0_with (Setup.ofCircuit st) (ofCircuitChk st) rfl rfl l0,
    tabOfLevel0_with (Setup.ofCircuit st) (Setup.ofCircuit st) rfl rfl l0]
  simp only [Functor.map_map]
  refine congrArg (fun f => Functor.map f (tabOfLevel0 (Setup.ofCircuit st) l0)) (funext fun t => ?_)
  exact (openStepM_fold t (Setup.ofCircuit st) { st.circuitFold with fold := foldChk st } rfl (foldChk_map st) rep dom xs
    hi coins).symm

/-- `placeOf` with the rep's opening `op st l0 rep dom xs hi coins` (`openingOf`). -/
def placeOfWith (op : Stmt → Level0.Level0 → Nat → ByteArray → Array (Array F128) → Array ByteArray →
    Array (Array F128) → V (Drive.Step Memo)) (salts : Array (ByteArray × Nat × Array ByteArray))
    (l0 : Level0.Level0) (s : Release) (p : Rep) (dom : ByteArray) : Except String (Frame.Place × Nat × Rep) := do
  let k := p.coins.size
  if h : k < p.alg.size then return (.sched p.alg[k].1, p.alg[k].2, p)
  if p.opened then return (.own, 0, p)
  let mut p := p
  if p.opening.isNone then
    let some st := s.sts[p.table]? | throw s!"table {p.table} of the session's {s.sts.size}"
    let some c := s.link | throw "the opening before the session's Commit coins"
    let some (_, _, hi) := salts.find? fun (r, q, _) => r == l0.root && q == p.rep
      | throw s!"table {p.table} rep {p.rep}'s opening before its salts for levels ≥ 1"
    let o ← op st l0 p.rep dom (Drive.extras st (Drive.linkPoints st c)) hi (p.coins.extract 0 p.alg.size)
    p := { p with opening := some o }
  match p.opening with
  | some (.ask c n _) => return (.sched c, n, { p with toOpening := true })
  | some (.done _) => return (.own, 0, { p with opened := true, opening := none })
  | some (.fail e) => throw e
  | none => throw "no opening"

theorem placeOf_eq_with : placeOf = placeOfWith (openingOf Setup.ofCircuit) := by
  funext salts l0 s p dom
  simp only [placeOf, placeOfWith, openingOf, bind_map_left] <;> rfl

def placeOfChk : Array (ByteArray × Nat × Array ByteArray) → Level0.Level0 → Release → Rep → ByteArray →
    Except String (Frame.Place × Nat × Rep) := placeOfWith (openingOf ofCircuitChk)

/-- `algOf` on the statement's setup `setup st`. -/
def algOfWith (setup : Stmt → Setup) (st : Stmt) (root dom : ByteArray) : V (Array (ByteArray × Nat)) := do
  let (sch, _) ← Zk.schedule st.m
  Skeleton.rounds (setup st) sch.k0 root dom

def algOfChk : Stmt → ByteArray → ByteArray → V (Array (ByteArray × Nat)) := algOfWith ofCircuitChk

theorem rounds_chk (st : Stmt) (k0 : Nat) (cap dom : ByteArray) :
    Skeleton.rounds (ofCircuitChk st) k0 cap dom = Skeleton.rounds (Setup.ofCircuit st) k0 cap dom :=
  rounds_fold (Setup.ofCircuit st) { st.circuitFold with fold := foldChk st } rfl (foldChk_map st) k0 cap dom

/-- Compiled code runs `placeOfChk` and `algOfChk`, the lincheck fold's refusal alone (`foldChk`); the kernel and every
proof see `placeOf` and `algOf`. `@[csimp]` rewrites only code compiled after it, so it stays above `withAlg` and
`roundPlace`. -/
@[csimp] theorem placeOf_eq_chk : @placeOf = @placeOfChk :=
  placeOf_eq_with.trans (congrArg placeOfWith openingOf_chk)

@[csimp] theorem algOf_eq_chk : @algOf = @algOfChk := by
  show algOfWith Setup.ofCircuit = algOfWith ofCircuitChk
  funext st root dom
  simp only [algOfWith, rounds_chk]

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

/-- **The frame and place of session `sid`'s round `content` of table `i`'s rep `r`**, squeezing `n` coins, in the loop's
frames `R` and level 0s `L0`, on a stream whose domain frame is `dom`: the rep's frame ready to take it (at the stream's
first round, its algebraic phase's scheduled rounds on the session's level 0, `withAlg`, and rep 1's after rep 0's
algebraic-phase coins), the round's place, whether it is the stream's first round, and why the frame takes no device
round in that place, if it takes none; or why the round has none. The place is `placeOf`'s, but past the opening: there
the inner proof's round `j` is the frame's own (`Frame.innerSched`) on the rounds the loop's finish computed for the rep
(`Sessions.innerOf`). A round `j` the finish has not computed has none, whatever the host relays: the loop refuses it,
and the session stops there (`Session`). Nor does the frame take a device round at any round of a session that stopped
(`Sessions.stopped`): the session stops at its first failure, and every rep's round from there on leaves as the
scheduled round of a stopped rep, whatever the device sent. -/
def roundPlace (R : Sessions) (L0 : Level0.Slots) (sid i r : Nat) (dom content : ByteArray) (n : Nat) :
    Except String (Rep × Frame.Place × Bool × Option String) := do
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
  let stopped := if R.stopped sid then some s!"session {sid} stopped at a rep's failure" else none
  match place with
  | .own =>
    let j := p'.st.m.inner.getD 0
    let fin := R.innerOf l0.root r
    if Frame.INNER_ROUNDS ≤ j then throw s!"a round after the inner proof's {Frame.INNER_ROUNDS}"
    else if fin.size ≤ j then throw s!"table {i} rep {r}'s inner round {j}, which the loop's finish has not computed"
    else return (p', .sched (Frame.innerSched p'.cfg p'.st.m fin n j), first, stopped)
  | _ => return (p', place, first, stopped)

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
frame at its first round, at the round's place (`roundPlace`). Where `roundPlace` takes no device round, the frame steps
on none (`ByteArray.empty`, which no scheduled round passes): the scheduled round stands in, and the rep stops. -/
def roundStep (R : Sessions) (L0 : Level0.Slots) (sid i r : Nat) (dom content : ByteArray) (n : Nat) : RoundStep :=
  match roundPlace R L0 sid i r dom content n with
  | .error e => { out := .error e }
  | .ok (p, place, first, noDevice) =>
    let content := if noDevice.isSome then ByteArray.empty else content
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
        stops := if v.startsWith " fail" then some (p.coins.size, (noDevice.map (" " ++ ·)).getD s!"{v.drop 5}") else none }
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

/-- **Whether session `sid`'s `Finish` may leave**, in the loop's frames `R`: every rep of its `tables` tables has a
frame, which answers `finish` at its end, not stopped; else why not. -/
def finishOf (R : Sessions) (sid tables : Nat) : Except String Unit := do
  for i in [0:tables] do
    for r in [0:2] do
      let some (_, p) := R.find sid i r | throw s!"a Finish where table {i} rep {r}'s frame never started"
      match Frame.step p.cfg p.st .finish with
      | (.done false, _) => pure ()
      | (.done true, _) => throw s!"{Frame.STOP}the Lean frame stopped table {i} rep {r}"
      | (.refuse why, _) | (.err why, _) | (.stop why, _) => throw s!"table {i} rep {r}: {why}"
      | _ => throw s!"table {i} rep {r}: no answer to finish"

/-- `finishOf` on the loop's frames as they are. -/
def finish (reg : IO.Ref Sessions) (sid tables : Nat) : IO (Except String Unit) :=
  return finishOf (← reg.get) sid tables

/-- **What the loop holds of table `i`'s rep `r`** on the level 0 of root `root`, in its frames `R` and level 0s `L0`
(`Finish.Own`): the samples its frame took at its start, which the loop's finish proves the rep with (`Finish.drawsOf`), and
what it holds the rep's requests to: the algebraic phase's values the frame took at their pads (`Frame.absorbed`'s order,
the rest `0`), the extra claims at the session's link points (`Drive.extras`, as `placeOf` opens with them), the coins
the stream answered each of the rep's rounds so far, and whether those cover every round of its algebraic phase (`alg`,
known once its first round came); and once the rep's opening ended, until its inner proof's first round, what the opening
computed that the rep's proof reads back (`Finish.Own.memo`). -/
def ownOf (R : Sessions) (L0 : Level0.Slots) (i r : Nat) (root : ByteArray) : Except String Finish.Own := do
  let some sl := L0.slots.find? (·.l0.root == root) | throw s!"table {i}: a level 0 the session loop did not draw"
  let some (sid, name) := sl.bound | throw s!"table {i}: a level 0 no session's Commit carries"
  let some s := R.sessions.find? (·.sid == sid) | throw s!"table {i}: session {sid}, which the loop is not serving"
  unless s.spec.tables[i]? == some name do throw s!"table {i}: the level 0 of the session's table {name}"
  let some st := s.sts[i]? | throw s!"table {i} of the session's {s.sts.size}"
  let some c := s.link | throw "a rep's request before the session's Commit coins"
  let some (_, p) := R.find sid i r | throw s!"a request of table {i} rep {r}, whose frame the loop has not started"
  let vals := (p.cfg.order.zip p.vals).foldl (fun v (k, x) => v.set! k x) (Array.replicate (p.cfg.lay.zp 64) 0)
  return { draws := p.draws, vals, xs := Drive.extras st (Drive.linkPoints st c), coins := p.coins,
           algebraEnded := !p.alg.isEmpty && p.alg.size ≤ p.coins.size,
           memo := match p.opening with | some (.done m) => some m | _ => none }

/-- **What the loop tells the prover of table `i`'s rep `r`** in the live session named `name`, for the request line
`line` on `ch`, past the session's `Commit`: `place NAME I R` is answered by its next round's place (`placeOf`), `alg N`
or `open N` for the algebraic phase's or the opening's scheduled round, squeezing `N` coins, `salts` where the algebraic
phase ended and the opening waits for the rep's salts for levels ≥ 1, which the loop's finish samples only then
(`Finish.serve`'s `salts`, `Finish.Own.algebraEnded`), else `own`: the inner proof's,
whose rounds the host relays from the loop's finish (`roundPlace`); `vals NAME I R`
by `ok N` followed by the `N` bytes of the values the frame took so far (`Rep.vals`). A prover whose device stops sends
each place's round empty (its first, the domain frame alone): the frame fails it, and the scheduled round leaves in its
place. Anything else is refused, with why. The answer is read off the loop's frames `R` and level 0s `L0` alone, as the
line it sends and the bytes after it: no rep's frame is written, so what the prover asks while one of the rep's own
requests runs (`round`, `coins`, which compute its place themselves, `roundPlace`) never puts back a frame from before
it. -/
def askOf (R : Sessions) (L0 : Level0.Slots) (line : String) : Except String (String × ByteArray) := do
  let [what, name, i, r] := (line.replace "\n" "").splitOn " " | throw s!"a request `{line.trim}`"
  let (some i, some r) := (i.toNat?, r.toNat?) | throw s!"a request `{line.trim}`"
  unless what == "place" || what == "vals" do throw s!"a request `{line.trim}`"
  let s ← R.named L0 what name
  let some (_, p) := R.find s.sid i r | throw s!"a {what} of table {i} rep {r}, whose frame the loop has not started"
  if what == "vals" then
    let b := wordBytes p.vals
    return (s!"ok {b.size}", b)
  let some tname := s.spec.tables[i]? | throw s!"table {i} of the session's {s.spec.tables.size}"
  let some l0 := Level0.ofSession L0 s.sid tname | throw s!"a place of {tname} before the session's Commit"
  let p ← withAlg s l0 tname p
  let alg := p.coins.size < p.alg.size
  if !alg && !p.opened && p.opening.isNone && !R.salts.any (fun (x, q, _) => x == l0.root && q == r) then
    return ("salts", .empty)
  let (place, n, _) ← placeOf R.salts l0 s p (s.spec.domain tname r).toUTF8
  match place with
  | .sched _ => return (s!"{if alg then "alg" else "open"} {n}", .empty)
  | _ => return ("own", .empty)

/-- `askOf` on the loop's frames and level 0s as they are, its answer sent on `ch`, `err WHY` where it refuses. -/
def ask (ch : Frame.Chan) (reg : IO.Ref Sessions) (l0s : IO.Ref Level0.Slots) (line : String) : IO Unit := do
  match askOf (← reg.get) (← l0s.get) line with
  | .ok (l, b) =>
    ch.write (l ++ "\n").toUTF8
    unless b.isEmpty do ch.write b
  | .error why => ch.write s!"err {why.replace "\n" " "}\n".toUTF8

/-- Session `sid` ended: its frames, their openings and the salts, inner proofs' rounds and proof files of its tables'
level 0 (`roots`) are gone. -/
def close (reg : IO.Ref Sessions) (sid : Nat) (roots : Array ByteArray) : IO Unit :=
  reg.modify fun R =>
    { sessions := R.sessions.filter (·.sid != sid), reps := R.reps.filter (·.sid != sid),
      salts := R.salts.filter fun (root, _, _) => !roots.contains root,
      files := R.files.filter fun (root, _, _) => !roots.contains root,
      inner := R.inner.filter fun (root, _, _) => !roots.contains root }

end Flock.Firewall.Frames
