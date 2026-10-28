/-
# The firewall's contract (`verity/firewall-transcript/v0`)

The firewall is the developer's only path from the workers to the auditor; the inner phase's is the program
`Flock.Firewall.run` (`Flock/Firewall.lean`). This file is the property every firewall's sessions must have, stated over
what it sends. Its contract: before release, every outbound value is one
of three things.

* (a) A commitment under a salt the workers never see, drawn from the tape and used for no other commitment:
  `Item.commit`, the hm96-sha512 commit string `b ‖ c` of SHA-512 of the message (`commitBC`), whose tree leaf is
  `commitLeaf`. That leaf is meant to be the firewall's commitment the recursive guarantee states
  (`Flock.Guarantees.firewallLeaf`, on the rec-thm branch, not in this package); that is not proved here. The
  commitments come exactly where and as many as the schedule says: each node of the Commit's level-0 tops, and each node
  of a round's fresh tops (`tops`) and its one message row (`commits`), right after the schedule item that carries them
  (`State.owed`). A round's copies of a level-0 top (`level0`) carry none: Commit committed them.
* (b) A value masked with a pad the workers never see (`Item.masked`: `v + h` in GF(2^128)), then checked against every
  relation the verifier will check (`Item.check`, each of the public input's `relations`, of which there must be one),
  before anything that depends on its pad leaves (an `Item.fixed` naming the pad) and before the session ends.
* (c) A value uniquely fixed by what is already committed or public (`Item.fixed`): by the public schedule (the contract
  derives the value itself, and holds the items to the schedule's order), by the public statement (the value the public
  input names), or by the firewall's own values (its level-0 recheck, its opening, the values it computes from its pads,
  under names the public input lists). Under D8 (a) the last three are today's Rust, which the transcript reports and
  this contract does not recompute.

What the contract holds a transcript to is the verifier's public input (`PublicInput`), never the transcript's own
copy of it. The tape's salts are pairwise distinct as bytes, and so are its pads, so a tape number names one value and a
salt or pad the workers saw under one number is not used under another.

Coins go to the workers exactly as the challenger sent them (`Item.coins`), and only after every commitment the schedule
owes (`State.owed`): in the inner phase each round's coins are live, drawn after its commitments, as `RecursiveKS`'s game
and `RecursiveZK` draw them. A public input may also declare the challenger's commitments to its coins
(`PublicInput.coins`): for each stream, one per round, the hm96-sha512 commit string `b ‖ c` (`commitBC`, the commit items'
commitment) of the round's coin message (`coinMsg`: the stream, the round and the coins) under a salt of the challenger's.
A coins item then carries that salt as its opening, and it leaves only if the opening opens the round's commitment to its
coins, once a round: a session whose public input declares commitments is refused coins without an opening, coins that
don't open, and coins of a round it holds no commitment for. A session that declares none (the inner phase, and the outer
phase, whose coins are C-Flock's coin tree, checked by its prover before the transcript is written) relays coins without
an opening, and it refuses coins that carry one.

A transcript is the firewall's candidate outbound items in order: the values it would release, each tagged with its case,
with its tape (the salts and pads it drew, by id) and the tape entries it handed the workers. The firewall's send check
decides each candidate as it comes; so does `run`. The first item that breaks the contract is the cut: a refused item
stops the session, nothing of the session's own after it leaves, and the auditor sees a stop (in the outer phase at the
session's `Finish`: the firewall releases the statement's schedule in the refused item's place, live PROTOCOL.md §10.1, and
the transcript ends at the cut). A session is accepted when every item may leave
and the last one is `finish`; one that ends without `finish` is a stop at its end.

`holds_commit`, `holds_masked`, `holds_fixed`, `holds_coins` and `holds_finish` say what an accepted session lets out, in
those terms; `holds_coins_binding` is the coins' binding in the fork form: two accepted sessions under one public input
that relay different coin messages for one committed round give a SHA-512 collision on two inputs computed from them. The transcript's JSON form is `Transcript.ofJson`'s and the public input's `PublicInput.ofJson`'s;
`flock-firewall contract [--public FILE] TRANSCRIPT…` prints `run`'s verdict for each transcript.
-/
import Flock.Canon
import Flock.Hm96

namespace Flock.Firewall.Contract

open Lean (Json)

def FORMAT : String := "verity/firewall-transcript/v0"

def PUBLIC_FORMAT : String := "verity/firewall-public/v0"

/-- The inner phase's schedule, rec-step3's (`rec_live.check_schedule`). -/
def SCHEDULE_FORMAT : String := "rec/inner-transcript/v1/schedule"

/-- hm96-sha512's salt (192 bytes). -/
def SALT_BYTES : Nat := 192

/-- What fixes a case-(c) value before it leaves. -/
inductive By where
  /-- the session's public schedule: the contract derives the value from it (`scheduled`) -/
  | schedule
  /-- the public statement: the value the public input names (the outer firewall's skeleton, `firewall::skeleton`) -/
  | statement
  /-- the firewall's own commitment of its own witness (the outer firewall's level-0 recheck) -/
  | recheck
  /-- the firewall's own opening of that commitment (the session's Lean finish's, `Finish.openLive`) -/
  | shadow
  /-- a value the firewall computes from its own tape and the values it passed (`final_c'`, the inner proof) -/
  | own
  deriving BEq, DecidableEq, Repr, Inhabited

def By.name : By → String
  | .schedule => "the schedule"
  | .statement => "the statement"
  | .recheck => "the firewall's recheck"
  | .shadow => "the firewall's opening"
  | .own => "the firewall's own values"

/-- One step of a firewall transcript, in the order the firewall takes it. Every step but `check` is an outbound item. -/
inductive Item where
  /-- (a): a commitment to `msg` under the firewall's salt number `salt`; `sent` leaves -/
  | commit (msg : ByteArray) (salt : Nat) (sent : ByteArray)
  /-- (b): `value` masked with the firewall's pad number `pad`; `sent` leaves -/
  | masked (value : F128) (pad : Nat) (sent : F128)
  /-- (c): the value `what`, fixed by `src` to `want` (for `By.schedule` and `By.statement` the contract takes the value
  from the public input and ignores `want`); `sent` leaves, and depends on the values masked under `pads` -/
  | fixed (what : String) (src : By) (want sent : ByteArray) (pads : Array Nat)
  /-- the firewall's check of `relation` on the values masked under `pads`, and its outcome (nothing leaves) -/
  | check (relation : String) (pads : Array Nat) (ok : Bool)
  /-- round `round` of `stream`'s coins from the challenger, `relayed` to the workers, with the challenger's `opening` of its
  commitment to them (its salt), empty when it made none -/
  | coins (stream : String) (round : Nat) (coins relayed : Array F128) (opening : ByteArray)
  /-- the session's end -/
  | finish
  deriving Inhabited

structure Transcript where
  phase : String
  /-- the firewall's tape: salt number `i` is `salts[i]`, pad number `i` is `pads[i]` -/
  salts : Array ByteArray
  pads : Array F128
  /-- the tape entries the firewall handed the workers -/
  exposedSalts : Array Nat
  exposedPads : Array Nat
  items : Array Item
  deriving Inhabited

/-- The verifier's public input, which the contract holds a transcript to and never reads from it: the session's public
schedule (the inner phase's `SCHEDULE_FORMAT`), every relation the verifier checks on the masked values,
the values the public statement fixes by name, the names of the values the firewall fixes by its own computation, and the
challenger's commitments to its coins, if it made any before the session: per stream, round `k`'s at `k`. -/
structure PublicInput where
  schedule : Option Json := none
  relations : Array String := #[]
  statement : Array (String × ByteArray) := #[]
  computes : Array String := #[]
  coins : Array (String × Array ByteArray) := #[]
  deriving Inhabited

/-- The value the public statement fixes for `what`. -/
def PublicInput.lookup (pub : PublicInput) (what : String) : Option ByteArray :=
  (pub.statement.find? (·.1 == what)).map (·.2)

/-- The challenger's commitment to round `k` of `stream`'s coins, if it made one. -/
def PublicInput.coinCommit (pub : PublicInput) (stream : String) (k : Nat) : Option ByteArray :=
  (pub.coins.find? (·.1 == stream)).bind (·.2[k]?)

theorem PublicInput.coinCommit_empty {pub : PublicInput} (h : pub.coins = #[]) (stream : String) (k : Nat) :
    pub.coinCommit stream k = none := by
  simp [PublicInput.coinCommit, h]

def bytesEq (a b : ByteArray) : Bool := a.data == b.data

def f128Eq (a b : F128) : Bool := decide (a = b)

/-- Whether `xs`'s entries are pairwise distinct under `eq`. -/
def pairwiseDistinct {α : Type} [Inhabited α] (eq : α → α → Bool) (xs : Array α) : Bool :=
  (List.range xs.size).all fun i => (List.range xs.size).all fun j => i == j || !eq xs[i]! xs[j]!

/-- (a)'s commit string: hm96-sha512's `b ‖ c` for the inner digest SHA-512(`msg`) under `salt`. -/
def commitBC (msg salt : ByteArray) : ByteArray := Hm96.Default512.commitString (Sha512.hash msg) salt

/-- (a)'s commitment as a tree leaf: hm96-sha512's leaf of SHA-512(`msg`) under `salt`. It is meant to be
`Flock.Guarantees.firewallLeaf`'s value (rec-thm), which this package does not state. -/
def commitLeaf (msg salt : ByteArray) : ByteArray := Hm96.Default512.leaf (Sha512.hash msg) salt

theorem commitLeaf_eq (msg salt : ByteArray) : commitLeaf msg salt = Hm96.Default512.treeLeaf (commitBC msg salt) := rfl

/-- The coin message's tag, which no other SHA-512 input of a session begins with. -/
def COINS_TAG : ByteArray := "verity/firewall-coins/v0".toUTF8.push 0

/-- The message the challenger commits to for round `k` of `stream`'s coins `cs`: the tag, the stream's UTF-8 bytes after
their length, the round, the coin count (each number 4 bytes big-endian), then each coin's 16 bytes, `le64(lo) ‖ le64(hi)`
(`F128.toBytes`). -/
def coinMsg (stream : String) (k : Nat) (cs : Array F128) : ByteArray :=
  cs.foldl F128.pushBytes
    (COINS_TAG ++ be32 stream.utf8ByteSize.toUInt32 ++ stream.toUTF8 ++ be32 k.toUInt32 ++ be32 cs.size.toUInt32)

/-- The schedule's items in their one order: `hello`, `roots`, `link.y`, then each `round:<i>` in the schedule's order,
every stream's `open:<s>` just before its first round, and `finish` after the last round. Each item has one place, so the
order carries nothing either: where a worker sent its Open does not reach the challenger. -/
structure Order where
  stage : Nat := 0
  opened : Array String := #[]
  rounds : Nat := 0
  deriving Inhabited

structure State where
  usedSalt : Array Bool
  usedPad : Array Bool
  /-- per pad, the relations that have passed on the value masked under it -/
  passed : Array (Array String)
  order : Order := {}
  /-- the commitments the last schedule item carries that have not yet come; nothing else may leave until they have -/
  owed : Nat := 0
  /-- the committed rounds whose coins have gone to the workers -/
  coined : Array (String × Nat) := #[]
  finished : Bool := false
  deriving Inhabited

def State.init (t : Transcript) : State :=
  { usedSalt := Array.replicate t.salts.size false, usedPad := Array.replicate t.pads.size false,
    passed := Array.replicate t.pads.size #[] }

private def canonBytes (j : Json) : V ByteArray := do return (← canon j).toUTF8

private def field (j : Json) (k : String) : V Json :=
  match j.getObjVal? k with
  | .ok v => .ok v
  | .error _ => .error s!"the schedule has no {k}"

/-- The schedule's streams, in its order (the record's: sorted by name). -/
def streams (sched : Json) : V (Array String) := do
  match ← field sched "streams" with
  | Json.obj kvs => return (kvs.toArray.map (·.1)).qsort (· < ·)
  | _ => throw "the schedule's streams are not an object"

/-- The schedule's round `o.rounds` (for the item `what`), in `SCHEDULE_FORMAT`'s form and no other: exactly its `stream`,
`k`, `n`, the node counts of the fresh tops it commits (`tops`), its copies of a level-0 top (`level0`) and `commits`, its
message row (one, or none for a round without a message). -/
def roundAt (sched : Json) (o : Order) (what : String) : V Json := do
  let rounds ← J.asArr (← field sched "rounds") "the schedule's rounds"
  let r ← match rounds[o.rounds]? with
    | some r => pure r
    | none => throw s!"{what}: past the schedule's {rounds.size} rounds"
  let ctx := s!"the schedule's {what}"
  let keys ← match r with
    | Json.obj kvs => pure ((kvs.toArray.map (·.1)).qsort (· < ·))
    | _ => throw s!"{ctx}: not an object"
  if keys != #["commits", "k", "level0", "n", "stream", "tops"] then
    throw s!"{ctx}: not a round of {SCHEDULE_FORMAT} (exactly stream, k, n, tops, level0 and commits)"
  let _ ← J.str r "stream" ctx
  let _ ← J.nat r "k" ctx
  let _ ← J.nat r "n" ctx
  let _ ← J.nat r "level0" ctx
  for c in ← J.arr r "tops" ctx do
    let _ ← J.asNat c s!"{ctx}.tops"
  let rows ← J.nat r "commits" ctx
  if rows > 1 then throw s!"{ctx}: {rows} message rows, where a round commits one or none"
  return r

/-- The value the schedule fixes for `what`, and the order it holds `what` to. -/
def scheduled (sched : Json) (o : Order) (what : String) : V (ByteArray × Order) := do
  let need := fun (b : Bool) (why : String) => if b then pure () else throw (α := Unit) why
  if what == "hello" then
    need (o.stage == 0) "Hello: not the session's first item"
    let kind := match sched.getObjVal? "kind" with
      | .ok (.str k) => k
      | _ => ""
    need (kind == SCHEDULE_FORMAT) s!"Hello: the public schedule is not a {SCHEDULE_FORMAT}"
    return (← canonBytes (← field sched "hello"), { o with stage := 1 })
  if what == "roots" then
    need (o.stage == 1) "Commit's roots: not just after Hello"
    return (← canonBytes (← field sched "roots"), { o with stage := 2 })
  if what == "link.y" then
    need (o.stage == 2) "Link's y: not after Commit"
    let n ← J.nat (← field sched "link") "y" "the schedule's link"
    return (ByteArray.mk (Array.replicate (16 * n) 0), { o with stage := 3 })
  if what == "finish" then
    let rounds ← J.asArr (← field sched "rounds") "the schedule's rounds"
    need (o.stage == 3 && o.rounds == rounds.size && o.opened.size == (← streams sched).size)
      "Finish: not after the schedule's last round"
    let ss ← match ← field sched "streams" with
      | Json.obj kvs => kvs.toArray.mapM fun (s, v) => do return (s, Json.num (← J.nat v "rounds" s!"stream {s}"))
      | _ => throw "the schedule's streams are not an object"
    let v := Json.mkObj [("rounds", Json.num rounds.size), ("streams", Json.mkObj ss.toList)]
    return (← canonBytes v, { o with stage := 4 })
  match what.splitOn ":" with
  | ["open", s] =>
    need (o.stage == 3 && !o.opened.contains s) s!"Open {s}: not after Link, or a second time"
    let rounds ← J.asArr (← field sched "rounds") "the schedule's rounds"
    let next ← match rounds[o.rounds]? with
      | some r => J.str r "stream" s!"the schedule's round {o.rounds}"
      | none => pure ""
    need (next == s) s!"Open {s}: not just before the stream's first round"
    let v ← J.get (← field sched "streams") s "the schedule's streams"
    let at_ ← J.get v "at" s!"stream {s}"
    let parent ← J.get v "parent" s!"stream {s}"
    return (← canonBytes (Json.mkObj [("at", at_), ("parent", parent)]), { o with opened := o.opened.push s })
  | ["round", i] =>
    need (o.stage == 3) s!"round {i}: not after Link"
    need (i.toNat? == some o.rounds) s!"round {i}: not the schedule's next round, {o.rounds}"
    let r ← roundAt sched o what
    let s ← J.str r "stream" s!"the schedule's round {i}"
    need (o.opened.contains s) s!"round {i}: on stream {s}, not yet opened"
    return (← canonBytes r, { o with rounds := o.rounds + 1 })
  | _ => throw s!"{what}: not an item of the schedule"

/-- The commitments that follow the schedule's value for `what` before anything else leaves: one a node of the Commit's
level-0 tops, and one a node of a round's fresh tops or its message row (its copies of a level-0 top owe none). -/
def owes (sched : Json) (o : Order) (what : String) : V Nat := do
  if what == "roots" then
    match ← field sched "roots" with
    | Json.obj kvs => kvs.toArray.foldlM (fun n (t, v) => do return n + (← J.asNat v s!"the schedule's roots.{t}")) 0
    | _ => throw "the schedule's roots are not an object"
  else match what.splitOn ":" with
    | ["round", _] =>
      let r ← roundAt sched o what
      let tops ← (← J.arr r "tops" s!"the schedule's {what}").foldlM
        (fun n c => do return n + (← J.asNat c s!"the schedule's {what}.tops")) 0
      return tops + (← J.nat r "commits" s!"the schedule's {what}")
    | _ => return 0

/-- What `src` fixes `what` to (`want` for the firewall's own values), the schedule's order after it, and the commitments
that must follow it. -/
def fixes (pub : PublicInput) (o : Order) (what : String) (src : By) (want : ByteArray) : V (ByteArray × Order × Nat) :=
  match src with
  | .schedule =>
    match pub.schedule with
    | none => .error s!"{what}: fixed by a schedule the public input does not carry"
    | some sched =>
      match scheduled sched o what with
      | .error e => .error e
      | .ok wo =>
        match owes sched o what with
        | .error e => .error e
        | .ok n => .ok (wo.1, wo.2, n)
  | .statement =>
    match pub.lookup what with
    | none => .error s!"{what}: a value the public statement does not fix"
    | some w => .ok (w, o, 0)
  | .recheck | .shadow | .own =>
    if pub.computes.contains what then .ok (want, o, 0)
    else .error s!"{what}: not a value the public input lets the firewall fix by its own computation"

def need (b : Bool) (why : String) : V Unit := if b then pure () else throw why

/-- Whether every relation has passed on the value masked under pad `p`. -/
def State.checked (s : State) (pub : PublicInput) (p : Nat) : Bool :=
  pub.relations.all fun r => (s.passed.getD p #[]).contains r

/-- One item: the state after it, or why it may not leave. -/
def step (pub : PublicInput) (t : Transcript) (s : State) : Item → V State
  | .commit msg i sent => do
    need (s.owed > 0) s!"a commitment under salt {i} that the schedule does not owe here"
    need (i < t.salts.size) s!"a commitment under salt {i}, which the firewall never drew"
    need (!s.usedSalt[i]!) s!"a commitment under salt {i}, used before"
    need (!t.exposedSalts.contains i) s!"a commitment under salt {i}, which the workers saw"
    let y := t.salts[i]!
    need (y.size == SALT_BYTES) s!"salt {i} is {y.size} bytes, not {SALT_BYTES}"
    need (bytesEq sent (commitBC msg y)) s!"a commitment under salt {i} that is not the message's"
    return { s with usedSalt := s.usedSalt.set! i true, owed := s.owed - 1 }
  | .masked v p sent => do
    need (s.owed == 0) s!"a value masked with pad {p}, before the commitments the schedule owes"
    need (!pub.relations.isEmpty) s!"a value masked with pad {p}, but the verifier checks no relation on it"
    need (p < t.pads.size) s!"a value masked with pad {p}, which the firewall never drew"
    need (!s.usedPad[p]!) s!"a value masked with pad {p}, used before"
    need (!t.exposedPads.contains p) s!"a value masked with pad {p}, which the workers saw"
    need (decide (sent = v + t.pads[p]!)) s!"a value sent under pad {p} that is not the value masked with it"
    return { s with usedPad := s.usedPad.set! p true }
  | .check r ps ok => do
    need (s.owed == 0) s!"a check of {r}, before the commitments the schedule owes"
    need (pub.relations.contains r) s!"a check of {r}, which is not a relation the verifier checks"
    need (ps.all fun p => s.usedPad.getD p false) s!"a check of {r} on a pad that masks nothing yet"
    need ok s!"the relation {r} fails"
    return { s with passed := ps.foldl (fun a p => a.modify p (·.push r)) s.passed }
  | .fixed what src want sent ps => do
    need (s.owed == 0) s!"{what}: before the commitments the schedule owes"
    let r ← fixes pub s.order what src want
    need (bytesEq sent r.1) s!"{what}: not the value {src.name} fixes"
    need (ps.all fun p => s.usedPad.getD p false && s.checked pub p)
      s!"{what}: depends on a masked value not yet checked against every relation"
    return { s with order := r.2.1, owed := r.2.2 }
  | .coins stream k cs relayed opening => do
    need (s.owed == 0) s!"{stream} round {k}: coins before the commitments the schedule owes"
    need (decide (relayed = cs)) s!"{stream} round {k}: the workers' coins are not the challenger's"
    match pub.coinCommit stream k with
    | some c =>
      need (opening.size != 0) s!"{stream} round {k}: coins without the opening of the challenger's commitment"
      need (opening.size == SALT_BYTES) s!"{stream} round {k}: an opening of {opening.size} bytes, not a {SALT_BYTES}-byte salt"
      need (bytesEq c (commitBC (coinMsg stream k cs) opening))
        s!"{stream} round {k}: coins that do not open the challenger's commitment"
      need (!s.coined.contains (stream, k)) s!"{stream} round {k}: coins a second time"
      return { s with coined := s.coined.push (stream, k) }
    | none =>
      need pub.coins.isEmpty s!"{stream} round {k}: coins the challenger never committed to"
      need (opening.size == 0) s!"{stream} round {k}: an opening of coins the challenger never committed to"
      return s
  | .finish => do
    need (s.owed == 0) "Finish: before the commitments the schedule owes"
    need ((List.range t.pads.size).all fun p => !s.usedPad[p]! || s.checked pub p)
      "Finish: a masked value not checked against every relation"
    return { s with finished := true }

/-- `run`'s verdict: the first `released` items may leave; `cut`, when some item may not, says why (the item at
`released`), or that the session ended without `finish`. -/
structure Verdict where
  released : Nat
  cut : Option String
  deriving Repr, Inhabited

def Verdict.accepted (v : Verdict) : Bool := v.cut.isNone

def go (pub : PublicInput) (t : Transcript) : State → List Item → Nat → Verdict
  | s, [], n => if s.finished then ⟨n, none⟩ else ⟨n, some "the session ended without Finish"⟩
  | s, it :: rest, n =>
    if s.finished then ⟨n, some "an item after Finish"⟩
    else match step pub t s it with
      | .ok s' => go pub t s' rest (n + 1)
      | .error why => ⟨n, some why⟩

/-- Why the tape is not one a number can name: two salt numbers, or two pad numbers, with one value. -/
def tapeError (t : Transcript) : Option String :=
  if !pairwiseDistinct bytesEq t.salts then some "the tape's salts are not pairwise distinct: two salt numbers are one salt"
  else if !pairwiseDistinct f128Eq t.pads then some "the tape's pads are not pairwise distinct: two pad numbers are one pad"
  else none

/-- The contract on one transcript, held to the verifier's public input `pub`; a tape whose salts or pads repeat a value
releases nothing. -/
def run (pub : PublicInput) (t : Transcript) : Verdict :=
  match tapeError t with
  | some why => ⟨0, some why⟩
  | none => go pub t (State.init t) t.items.toList 0

/-- The contract holds of the transcript under the public input: every item may leave, and the session ends with
`finish`. -/
def Holds (pub : PublicInput) (t : Transcript) : Prop := (run pub t).accepted = true

/-! ## What an accepted session lets out -/

/-- Every item of a list `go` lets through whole is one `step` takes from some state before Finish. -/
theorem go_steps {pub : PublicInput} {t : Transcript} : ∀ {s : State} {l : List Item} {n : Nat},
    (go pub t s l n).cut = none → ∀ it ∈ l, ∃ s₀ s₁, s₀.finished = false ∧ step pub t s₀ it = .ok s₁
  | s, [], n, _ => by intro it h; cases h
  | s, it :: rest, n, h => by
    unfold go at h
    split at h
    · simp at h
    · rename_i hf
      split at h
      · rename_i s' hs
        intro x hx
        cases hx with
        | head => exact ⟨s, s', by simpa using hf, hs⟩
        | tail _ hx => exact go_steps h x hx
      · simp at h

private theorem need_ok {b : Bool} {why : String} {α : Type} {k : Unit → V α} {x : α}
    (h : (need b why >>= k) = .ok x) : b = true ∧ k () = .ok x := by
  cases b
  · simp [need, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  · exact ⟨rfl, by simpa [need, bind, Except.bind, pure, Except.pure] using h⟩

private theorem bind_ok {α β : Type} {m : V α} {k : α → V β} {x : β}
    (h : (m >>= k) = .ok x) : ∃ a, m = .ok a ∧ k a = .ok x := by
  cases m <;> simp_all [bind, Except.bind]

/-- (a) item by item: a commitment leaves only as `b ‖ c` of its message under a drawn salt of 192 bytes, unused and
unseen by the workers. -/
theorem step_commit {pub : PublicInput} {t : Transcript} {s s' : State} {msg : ByteArray} {i : Nat} {sent : ByteArray}
    (h : step pub t s (.commit msg i sent) = .ok s') :
    i < t.salts.size ∧ s.usedSalt[i]! = false ∧ i ∉ t.exposedSalts ∧ (t.salts[i]!).size = SALT_BYTES ∧
      sent.data = (commitBC msg t.salts[i]!).data := by
  simp only [step, need, bind, Except.bind] at h
  repeat' (split at h <;> try simp_all [pure, Except.pure, throw, throwThe, MonadExceptOf.throw])
  all_goals simp_all [bytesEq]

/-- (b) item by item: a masked value leaves only under a drawn pad, unused and unseen by the workers. -/
theorem step_masked {pub : PublicInput} {t : Transcript} {s s' : State} {v : F128} {p : Nat} {sent : F128}
    (h : step pub t s (.masked v p sent) = .ok s') :
    p < t.pads.size ∧ s.usedPad[p]! = false ∧ p ∉ t.exposedPads ∧ sent = v + t.pads[p]! := by
  simp only [step, need, bind, Except.bind] at h
  repeat' (split at h <;> try simp_all [pure, Except.pure, throw, throwThe, MonadExceptOf.throw])

/-- The workers' coins are the challenger's, and either they open the challenger's commitment to their round, a round
whose coins have not gone before, or the session declares no commitment, and they carry no opening. -/
theorem step_coins {pub : PublicInput} {t : Transcript} {s s' : State} {stream : String} {k : Nat}
    {cs relayed : Array F128} {opening : ByteArray} (h : step pub t s (.coins stream k cs relayed opening) = .ok s') :
    relayed = cs ∧
      ((∃ c, pub.coinCommit stream k = some c ∧ opening.size = SALT_BYTES ∧
          c.data = (commitBC (coinMsg stream k cs) opening).data ∧ s.coined.contains (stream, k) = false) ∨
        (pub.coinCommit stream k = none ∧ pub.coins = #[] ∧ opening.size = 0)) := by
  simp only [step] at h
  obtain ⟨-, h⟩ := need_ok h
  obtain ⟨h1, h⟩ := need_ok h
  refine ⟨by simpa using h1, ?_⟩
  split at h
  · rename_i c hc
    obtain ⟨-, h⟩ := need_ok h
    obtain ⟨h3, h⟩ := need_ok h
    obtain ⟨h4, h⟩ := need_ok h
    obtain ⟨h5, -⟩ := need_ok h
    exact .inl ⟨c, hc, by simpa using h3, by simpa [bytesEq] using h4, by simpa using h5⟩
  · rename_i hc
    obtain ⟨h3, h⟩ := need_ok h
    obtain ⟨h4, -⟩ := need_ok h
    simp only [Array.isEmpty_iff] at h3
    exact .inr ⟨hc, h3, by simpa using h4⟩

/-- What `fixes` fixes a value to: the schedule's own value, the public statement's, or, for a name the public input lists
as the firewall's own computation, the value the transcript reports. -/
theorem fixes_spec {pub : PublicInput} {o : Order} {what : String} {src : By} {want : ByteArray}
    {r : ByteArray × Order × Nat} (h : fixes pub o what src want = .ok r) :
    (src = .schedule → ∃ sched o', pub.schedule = some sched ∧ scheduled sched o what = .ok (r.1, o')) ∧
    (src = .statement → pub.lookup what = some r.1) ∧
    (src ≠ .schedule → src ≠ .statement → r.1 = want ∧ pub.computes.contains what = true) := by
  cases src <;> simp only [fixes] at h
  · refine ⟨fun _ => ?_, fun e => (by cases e), fun e _ => absurd rfl e⟩
    split at h
    · cases h
    · rename_i sched hs
      split at h
      · cases h
      · rename_i wo hsc
        split at h
        · cases h
        · cases h
          exact ⟨sched, wo.2, hs, hsc⟩
  · refine ⟨fun e => (by cases e), fun _ => ?_, fun _ e => absurd rfl e⟩
    split at h
    · cases h
    · rename_i w hw
      cases h
      exact hw
  all_goals
    refine ⟨fun e => (by cases e), fun e => (by cases e), fun _ _ => ?_⟩
    split at h
    · rename_i hc
      cases h
      exact ⟨rfl, hc⟩
    · cases h

/-- (c) item by item: a fixed value leaves only as `fixes` fixes it, and after every pad it depends on has passed every
relation. -/
theorem step_fixed {pub : PublicInput} {t : Transcript} {s s' : State} {what : String} {src : By} {want sent : ByteArray}
    {ps : Array Nat} (h : step pub t s (.fixed what src want sent ps) = .ok s') :
    (∃ r, fixes pub s.order what src want = .ok r ∧ sent.data = r.1.data) ∧
    ∀ p ∈ ps, s.usedPad.getD p false = true ∧ s.checked pub p = true := by
  simp only [step] at h
  obtain ⟨-, h⟩ := need_ok h
  obtain ⟨r, hr, h⟩ := bind_ok h
  obtain ⟨h1, h⟩ := need_ok h
  obtain ⟨h2, -⟩ := need_ok h
  refine ⟨⟨r, hr, by simpa [bytesEq] using h1⟩, fun p hp => ?_⟩
  have := Array.all_eq_true_iff_forall_mem.mp h2 p hp
  simpa using this

theorem step_finished {pub : PublicInput} {t : Transcript} {s s' : State} {it : Item} (h : step pub t s it = .ok s')
    (hit : it ≠ .finish) : s'.finished = s.finished := by
  cases it <;> simp only [step, need, bind, Except.bind] at h
  all_goals repeat' (split at h <;> try simp_all [pure, Except.pure, throw, throwThe, MonadExceptOf.throw])
  all_goals first | (subst h; rfl) | simp_all

/-- A list `go` lets through whole, from a state before Finish, ends with Finish. -/
theorem go_last {pub : PublicInput} {t : Transcript} : ∀ {s : State} {l : List Item} {n : Nat},
    (go pub t s l n).cut = none → s.finished = false → l.getLast? = some .finish
  | s, [], n, h, hf => by simp [go, hf] at h
  | s, it :: rest, n, h, hf => by
    unfold go at h
    simp only [hf, Bool.false_eq_true, ↓reduceIte] at h
    split at h
    · rename_i s' hs
      cases rest with
      | nil =>
        have h1 : s'.finished = true := by
          unfold go at h; split at h <;> simp_all
        by_cases hit : it = .finish
        · simp [hit]
        · have := step_finished hs hit; simp_all
      | cons x xs =>
        have hf' : s'.finished = false := by
          unfold go at h; split at h
          · simp at h
          · simp_all
        have := go_last h hf'
        simpa [List.getLast?_cons_cons] using this
    · simp at h

/-- An accepted session's tape names each salt and each pad once, and `go` lets every item through. -/
theorem holds_go {pub : PublicInput} {t : Transcript} (h : Holds pub t) :
    tapeError t = none ∧ (go pub t (State.init t) t.items.toList 0).cut = none := by
  have h' : (run pub t).cut = none := by simpa [Holds, Verdict.accepted, Option.isNone_iff_eq_none] using h
  unfold run at h'
  cases he : tapeError t with
  | some why => simp [he] at h'
  | none =>
    simp only [he] at h'
    exact ⟨rfl, h'⟩

theorem pairwiseDistinct_ne {α : Type} [Inhabited α] {eq : α → α → Bool} {xs : Array α}
    (h : pairwiseDistinct eq xs = true) {i j : Nat} (hi : i < xs.size) (hj : j < xs.size) (hne : i ≠ j) :
    eq xs[i]! xs[j]! = false := by
  simp only [pairwiseDistinct, List.all_eq_true, List.mem_range] at h
  have := h i hi j hj
  simpa [hne] using this

theorem tapeError_none {t : Transcript} (h : tapeError t = none) :
    pairwiseDistinct bytesEq t.salts = true ∧ pairwiseDistinct f128Eq t.pads = true := by
  unfold tapeError at h
  by_cases a : pairwiseDistinct bytesEq t.salts = true <;> by_cases b : pairwiseDistinct f128Eq t.pads = true <;>
    simp_all

/-- An accepted session ends with Finish: a session that stops is never accepted. -/
theorem holds_finish {pub : PublicInput} {t : Transcript} (h : Holds pub t) : t.items.toList.getLast? = some .finish :=
  go_last (holds_go h).2 rfl

/-- (a) on an accepted session: every commitment that left is hm96-sha512's `b ‖ c` of its message under a salt the
firewall drew, of 192 bytes, that the workers never saw and that no other salt number of the tape names (`commitLeaf`
is its tree leaf). That no two commitments share a salt number is `step`'s check, not stated here. -/
theorem holds_commit {pub : PublicInput} {t : Transcript} (h : Holds pub t) {msg : ByteArray} {i : Nat}
    {sent : ByteArray} (hm : Item.commit msg i sent ∈ t.items.toList) :
    i < t.salts.size ∧ i ∉ t.exposedSalts ∧ (t.salts[i]!).size = SALT_BYTES ∧
      sent.data = (commitBC msg t.salts[i]!).data ∧
      ∀ j < t.salts.size, j ≠ i → (t.salts[j]!).data ≠ (t.salts[i]!).data := by
  obtain ⟨ht, hg⟩ := holds_go h
  obtain ⟨_, _, -, hs⟩ := go_steps hg _ hm
  obtain ⟨a, -, b, c, d⟩ := step_commit hs
  refine ⟨a, b, c, d, fun j hj hne => ?_⟩
  have := pairwiseDistinct_ne (tapeError_none ht).1 hj a hne
  simpa [bytesEq] using this

/-- (b) on an accepted session: every masked value that left is its value plus a pad the firewall drew that the workers
never saw and that no other pad number of the tape names. -/
theorem holds_masked {pub : PublicInput} {t : Transcript} (h : Holds pub t) {v : F128} {p : Nat} {sent : F128}
    (hm : Item.masked v p sent ∈ t.items.toList) :
    p < t.pads.size ∧ p ∉ t.exposedPads ∧ sent = v + t.pads[p]! ∧
      ∀ q < t.pads.size, q ≠ p → t.pads[q]! ≠ t.pads[p]! := by
  obtain ⟨ht, hg⟩ := holds_go h
  obtain ⟨_, _, -, hs⟩ := go_steps hg _ hm
  obtain ⟨a, -, b, c⟩ := step_masked hs
  refine ⟨a, b, c, fun q hq hne => ?_⟩
  have := pairwiseDistinct_ne (tapeError_none ht).2 hq a hne
  simpa [f128Eq] using this

/-- On an accepted session the workers received exactly the challenger's coins. In a session whose public input declares
commitments they are the coins the challenger committed to before the session: the opening, a 192-byte salt, opens the
public input's commitment to their round to their coin message, hm96-sha512's `b ‖ c` of it (`commitBC`). A session that
declares none (the inner phase, whose coins are live, and the outer phase) relays them with no opening. -/
theorem holds_coins {pub : PublicInput} {t : Transcript} (h : Holds pub t) {stream : String} {k : Nat}
    {cs relayed : Array F128} {opening : ByteArray} (hm : Item.coins stream k cs relayed opening ∈ t.items.toList) :
    relayed = cs ∧
      ((∃ c, pub.coinCommit stream k = some c ∧ opening.size = SALT_BYTES ∧
          c.data = (commitBC (coinMsg stream k cs) opening).data) ∨
        (pub.coins = #[] ∧ opening.size = 0)) := by
  obtain ⟨_, _, -, hs⟩ := go_steps (holds_go h).2 _ hm
  obtain ⟨h1, h2 | h2⟩ := step_coins hs
  · obtain ⟨c, a, b, d, -⟩ := h2
    exact ⟨h1, .inl ⟨c, a, b, d⟩⟩
  · exact ⟨h1, .inr h2.2⟩

/-- A session that declares commitments relays, in an accepted run, only committed coins. -/
theorem holds_coins_committed {pub : PublicInput} {t : Transcript} (h : Holds pub t) {stream : String} {k : Nat}
    {cs relayed : Array F128} {opening : ByteArray} (hm : Item.coins stream k cs relayed opening ∈ t.items.toList)
    (hd : pub.coins ≠ #[]) :
    relayed = cs ∧ ∃ c, pub.coinCommit stream k = some c ∧ opening.size = SALT_BYTES ∧
      c.data = (commitBC (coinMsg stream k cs) opening).data := by
  obtain ⟨h1, h2 | h2⟩ := holds_coins h hm
  · exact ⟨h1, h2⟩
  · exact absurd h2.1 hd

/-! ## The coins' binding -/

theorem sha512_size (m : ByteArray) : (Sha512.hash m).size = 64 := by
  unfold Sha512.hash
  exact Array.size_ofFn

theorem eq_of_get!_64 {x x' : ByteArray} (hx : x.size = 64) (hx' : x'.size = 64)
    (h : ∀ i, i < 64 → x.get! i = x'.get! i) : x = x' := by
  apply ByteArray.ext
  apply Array.ext
  · rw [ByteArray.size_data, ByteArray.size_data, hx, hx']
  · intro i h1 h2
    have hi : i < 64 := by rw [ByteArray.size_data, hx] at h1; exact h1
    have := h i hi
    change x.data[i]! = x'.data[i]! at this
    rwa [getElem!_pos x.data i h1, getElem!_pos x'.data i h2] at this

/-- `b`, the masked digest of a commit string: `x ⊕ m` over 64 bytes. -/
def xor64 (x m : ByteArray) : ByteArray := ByteArray.mk ((Array.range 64).map fun i => x.get! i ^^^ m.get! i)

theorem xor64_size (x m : ByteArray) : (xor64 x m).size = 64 := by
  rw [← ByteArray.size_data]
  simp [xor64]

theorem xor64_cancel {x x' m : ByteArray} (hx : x.size = 64) (hx' : x'.size = 64) (h : xor64 x m = xor64 x' m) :
    x = x' := by
  apply eq_of_get!_64 hx hx'
  intro i hi
  have := congrArg (fun b : ByteArray => b.data[i]!) h
  simp only [xor64] at this
  rw [getElem!_pos _ i (by simpa using hi), getElem!_pos _ i (by simpa using hi)] at this
  simpa [Array.getElem_map, Array.getElem_range] using this

theorem commitBC_eq (msg y : ByteArray) :
    commitBC msg y = xor64 (Sha512.hash msg) (Hm96.Default512.mask y) ++ Sha512.hash (Hm96.Default512.saltPrefix ++ y) :=
  rfl

/-- The two SHA-512 inputs that collide when two openings `y`, `y'` of one commitment open the messages `m`, `m'`: the
messages when the salts are one (the masks then cancel), else the salts' with the salt prefix. -/
def coinPair (m m' y y' : ByteArray) : ByteArray × ByteArray :=
  if Hm96.Default512.saltPrefix ++ y = Hm96.Default512.saltPrefix ++ y' then (m, m')
  else (Hm96.Default512.saltPrefix ++ y, Hm96.Default512.saltPrefix ++ y')

/-- **hm96-sha512's commit string binds its message**: two different messages with one `b ‖ c` give a SHA-512 collision
on `coinPair`'s two inputs, computed from the two openings. -/
theorem commitBC_binding {m m' y y' : ByteArray} (hne : m ≠ m') (h : (commitBC m y).data = (commitBC m' y').data) :
    (coinPair m m' y y').1 ≠ (coinPair m m' y y').2 ∧
      Sha512.hash (coinPair m m' y y').1 = Sha512.hash (coinPair m m' y y').2 := by
  have h' : commitBC m y = commitBC m' y' := ByteArray.ext h
  rw [commitBC_eq, commitBC_eq] at h'
  have hb := ByteArray.append_inj_left h' (by rw [xor64_size, xor64_size])
  rw [hb] at h'
  have hc := (ByteArray.append_right_inj _).mp h'
  by_cases hs : Hm96.Default512.saltPrefix ++ y = Hm96.Default512.saltPrefix ++ y'
  · simp only [coinPair, hs, ↓reduceIte]
    have hy : y = y' := (ByteArray.append_right_inj _).mp hs
    subst hy
    exact ⟨hne, xor64_cancel (sha512_size _) (sha512_size _) hb⟩
  · simp only [coinPair, hs, ↓reduceIte]
    exact ⟨hs, hc⟩

/-- **The coins' binding, in the fork form**: two accepted sessions under one public input that relay coins of one
committed round whose coin messages differ give a SHA-512 collision on two inputs computed from the two coins items. The
inner phase's coins are live, and its public input declares no committed round. -/
theorem holds_coins_binding {pub : PublicInput} {t t' : Transcript} (h : Holds pub t) (h' : Holds pub t')
    {stream : String} {k : Nat} {cs cs' relayed relayed' : Array F128} {y y' : ByteArray}
    (hm : Item.coins stream k cs relayed y ∈ t.items.toList) (hm' : Item.coins stream k cs' relayed' y' ∈ t'.items.toList)
    (hc : pub.coinCommit stream k ≠ none) (hne : coinMsg stream k cs ≠ coinMsg stream k cs') :
    (coinPair (coinMsg stream k cs) (coinMsg stream k cs') y y').1 ≠
        (coinPair (coinMsg stream k cs) (coinMsg stream k cs') y y').2 ∧
      Sha512.hash (coinPair (coinMsg stream k cs) (coinMsg stream k cs') y y').1 =
        Sha512.hash (coinPair (coinMsg stream k cs) (coinMsg stream k cs') y y').2 := by
  have one : ∀ {t : Transcript} {cs r : Array F128} {y : ByteArray}, Holds pub t →
      Item.coins stream k cs r y ∈ t.items.toList →
      ∃ c, pub.coinCommit stream k = some c ∧ c.data = (commitBC (coinMsg stream k cs) y).data := by
    intro t cs r y h hm
    obtain ⟨-, ⟨c, a, -, b⟩ | ⟨e, -⟩⟩ := holds_coins h hm
    · exact ⟨c, a, b⟩
    · exact absurd (PublicInput.coinCommit_empty e stream k) hc
  obtain ⟨c, a, b⟩ := one h hm
  obtain ⟨c', a', b'⟩ := one h' hm'
  rw [a] at a'
  cases a'
  exact commitBC_binding hne (b.symm.trans b')

/-- (c) on an accepted session: every fixed value that left is the public schedule's own (`scheduled`), the public
statement's, or, under a name the public input lists as the firewall's own computation, the value the transcript reports
the firewall computing, which the contract does not recompute (`FirewallComputes` names that assumption). -/
theorem holds_fixed {pub : PublicInput} {t : Transcript} (h : Holds pub t) {what : String} {src : By}
    {want sent : ByteArray} {ps : Array Nat} (hm : Item.fixed what src want sent ps ∈ t.items.toList) :
    (src = .schedule → ∃ sched o w o', pub.schedule = some sched ∧ scheduled sched o what = .ok (w, o') ∧
      sent.data = w.data) ∧
    (src = .statement → ∃ w, pub.lookup what = some w ∧ sent.data = w.data) ∧
    (src ≠ .schedule → src ≠ .statement → sent.data = want.data ∧ pub.computes.contains what = true) := by
  obtain ⟨s, _, -, hs⟩ := go_steps (holds_go h).2 _ hm
  obtain ⟨⟨r, hr, he⟩, -⟩ := step_fixed hs
  obtain ⟨a, b, c⟩ := fixes_spec hr
  refine ⟨fun e => ?_, fun e => ⟨r.1, b e, he⟩, fun e1 e2 => ?_⟩
  · obtain ⟨sched, o', h1, h2⟩ := a e
    exact ⟨sched, s.order, r.1, o', h1, h2, he⟩
  · obtain ⟨c1, c2⟩ := c e1 e2
    exact ⟨by rw [← c1]; exact he, c2⟩

/-! ## The transcript's JSON form -/

private def hexField (j : Json) (k ctx : String) : V ByteArray := do
  match unhex (← J.str j k ctx) with
  | .ok b => pure b
  | .error e => throw s!"{ctx}.{k}: {e}"

private def f128Field (j : Json) (k ctx : String) : V F128 := do
  match f128OfHex (← J.str j k ctx) with
  | .ok v => pure v
  | .error e => throw s!"{ctx}.{k}: {e}"

private def f128s (j : Json) (k ctx : String) : V (Array F128) := do
  (← J.arr j k ctx).mapM fun v => do
    match f128OfHex (← J.asStr v s!"{ctx}.{k}") with
    | .ok x => pure x
    | .error e => throw s!"{ctx}.{k}: {e}"

private def nats (j : Json) (k ctx : String) : V (Array Nat) := do
  if !J.has j k then return #[]
  (← J.arr j k ctx).mapM (J.asNat · s!"{ctx}.{k}")

def By.ofString : String → V By
  | "schedule" => pure .schedule
  | "statement" => pure .statement
  | "recheck" => pure .recheck
  | "shadow" => pure .shadow
  | "own" => pure .own
  | s => throw s!"fixed by {s}: not a source the contract knows"

def Item.ofJson (j : Json) (ctx : String) : V Item := do
  match j with
  | Json.obj kvs =>
    match kvs.toArray with
    | #[("commit", v)] => return .commit (← hexField v "msg" ctx) (← J.nat v "salt" ctx) (← hexField v "sent" ctx)
    | #[("masked", v)] => return .masked (← f128Field v "value" ctx) (← J.nat v "pad" ctx) (← f128Field v "sent" ctx)
    | #[("fixed", v)] =>
      let want ← if J.has v "want" then hexField v "want" ctx else pure ByteArray.empty
      return .fixed (← J.str v "what" ctx) (← By.ofString (← J.str v "by" ctx)) want (← hexField v "sent" ctx)
        (← nats v "pads" ctx)
    | #[("check", v)] => return .check (← J.str v "relation" ctx) (← nats v "pads" ctx) (← J.bool v "ok" ctx)
    | #[("coins", v)] =>
      let opening ← if J.has v "opening" then hexField v "opening" ctx else pure ByteArray.empty
      return .coins (← J.str v "stream" ctx) (← J.nat v "round" ctx) (← f128s v "coins" ctx) (← f128s v "relayed" ctx) opening
    | #[("finish", _)] => return .finish
    | _ => throw s!"{ctx}: not one item of commit, masked, fixed, check, coins or finish"
  | _ => throw s!"{ctx}: not an object"

def Transcript.ofJson (j : Json) : V Transcript := do
  let kind ← J.str j "kind" "transcript"
  need (kind == FORMAT) s!"transcript: kind {kind}, not {FORMAT}"
  let tape ← J.get j "tape" "transcript"
  let salts ← (← J.arr tape "salts" "tape").mapM fun v => do
    match unhex (← J.asStr v "tape.salts") with
    | .ok b => pure b
    | .error e => throw s!"tape.salts: {e}"
  let exposed ← if J.has j "exposed" then J.get j "exposed" "transcript" else pure (Json.mkObj [])
  let items ← (← J.arr j "items" "transcript").mapIdxM fun i v => Item.ofJson v s!"item {i}"
  return { phase := ← J.str j "phase" "transcript", salts, pads := ← f128s tape "pads" "tape",
           exposedSalts := ← nats exposed "salts" "exposed", exposedPads := ← nats exposed "pads" "exposed", items }

/-- The public input's JSON form, `verity/firewall-public/v0`: `schedule` (or `null`), `relations` and `computes` (lists
of names), `statement`, an object from each name the statement fixes to its value in hex, and `coins`, an object from each
stream to the challenger's commitments to its rounds' coins in order, each `b ‖ c` in hex. A transcript's own
`schedule` or `relations`, if it carries them, are never read. -/
def PublicInput.ofJson (j : Json) : V PublicInput := do
  let kind ← J.str j "kind" "public input"
  need (kind == PUBLIC_FORMAT) s!"public input: kind {kind}, not {PUBLIC_FORMAT}"
  let schedule := match j.getObjVal? "schedule" with
    | .ok Json.null => none
    | .ok s => some s
    | .error _ => none
  let names : String → V (Array String) := fun k => do
    if !J.has j k then return #[]
    (← J.arr j k "public input").mapM (J.asStr · s!"public input.{k}")
  let statement ← if !J.has j "statement" then pure #[] else
    match ← J.get j "statement" "public input" with
    | Json.obj kvs => kvs.toArray.mapM fun (w, v) => do
        match unhex (← J.asStr v s!"public input.statement.{w}") with
        | .ok b => pure (w, b)
        | .error e => throw s!"public input.statement.{w}: {e}"
    | _ => throw "public input.statement: not an object"
  let coins ← if !J.has j "coins" then pure #[] else
    match ← J.get j "coins" "public input" with
    | Json.obj kvs => kvs.toArray.mapM fun (s, v) => do
        let ctx := s!"public input.coins.{s}"
        let bcs ← (← J.asArr v ctx).mapM fun c => do
          match unhex (← J.asStr c ctx) with
          | .ok b => pure b
          | .error e => throw s!"{ctx}: {e}"
        pure (s, bcs)
    | _ => throw "public input.coins: not an object"
  return { schedule, relations := ← names "relations", statement, computes := ← names "computes", coins }

def Verdict.toJson (v : Verdict) : Json :=
  Json.mkObj [("accepted", Json.bool v.accepted), ("released", Json.num v.released),
    ("why", match v.cut with | some w => Json.str w | none => Json.null)]

/-- `flock-firewall contract [--public FILE] FILE…` (`args` after `contract`): each transcript's verdict under the public
input in `--public`'s file, one JSON line each; without `--public` the public input is empty, so every transcript that
leans on a schedule, a statement, a relation or the firewall's own values is refused. Exit 2 when a file is not a
transcript or the public input is not one. -/
def cmd (args : List String) : IO UInt32 := do
  let (pubFile, files) := match args with
    | "--public" :: f :: rest => (some f, rest)
    | rest => (none, rest)
  if files.isEmpty || files.head? == some "--public" then
    IO.eprintln "usage: flock-firewall contract [--public PUBLIC.json] TRANSCRIPT.json..."
    return 2
  let pub ← match pubFile with
    | none => pure (Except.ok ({} : PublicInput))
    | some f => do
      let text ← IO.FS.readFile f
      pure (Json.parse text >>= PublicInput.ofJson)
  match pub with
  | .error e =>
    IO.println (Json.mkObj [("public", Json.str (pubFile.getD "")), ("error", Json.str e)]).compress
    return 2
  | .ok pub =>
    let mut rc : UInt32 := 0
    for f in files do
      let text ← IO.FS.readFile f
      match Json.parse text >>= Transcript.ofJson with
      | .ok t =>
        IO.println ((run pub t).toJson.setObjVal! "file" (Json.str f)).compress
      | .error e =>
        IO.println (Json.mkObj [("file", Json.str f), ("error", Json.str e)]).compress
        rc := 2
    return rc

end Flock.Firewall.Contract
