/-
# The firewall's contract (`verity/firewall-transcript/v0`)

The firewall is the developer's only path from the workers to the auditor; the inner phase's is the program
`Flock.Firewall.run` (`Flock/Firewall.lean`). This file is the property every firewall's sessions must have, stated over
what it sends. Its contract (Daniel's draft, "Why the auditor learns nothing"): before release, every outbound value is one
of three things.

* (a) A commitment under a fresh salt the workers never see: `Item.commit`, the hm96-sha512 commit string `b ‖ c` of
  SHA-512 of the message (`commitBC`), whose tree leaf is `firewallLeaf`. That leaf is the gateway's commitment the
  recursive guarantee states (`Flock.Guarantees.gatewayLeaf`, on the rec-thm branch, not in this package), which
  `ZkView.leafSplit` equates with `Hm96.Default512.leaf` of the digest.
* (b) A value masked with a fresh pad the workers never see (`Item.masked`: `v + h` in GF(2^128)), then checked against
  every relation the verifier will check (`Item.check`, each of the transcript's `relations`), before anything that
  depends on its pad leaves (an `Item.fixed` naming the pad) and before the session ends.
* (c) A value uniquely fixed by what is already committed or public (`Item.fixed`): by the public schedule (the contract
  derives the value itself, and holds the items to the schedule's order), by the statement, or by the firewall's own
  values (its level-0 recheck, its opening, the values it computes from its pads). Under D8 (a) the last three are
  today's Rust, which the transcript reports and this contract does not recompute.

Coins go to the workers exactly as the challenger sent them (`Item.coins`). They are live coins the challenger never
committed to, so a coins item carries no opening, and one that does is refused.

A transcript is the firewall's candidate outbound items in order: the values it would release, each tagged with its case,
with its tape (the salts and pads it drew, by id) and the tape entries it handed the workers. The firewall's send check
decides each candidate as it comes; so does `run`. The first item that breaks the contract is the cut: a refused item
stops the session, nothing after it leaves, and the auditor sees a stop. A session is accepted when every item may leave
and the last one is `finish`; one that ends without `finish` is a stop at its end.

`holds_commit`, `holds_masked`, `holds_fixed`, `holds_coins` and `holds_finish` say what an accepted session lets out, in
those terms. The transcript's JSON form is `Transcript.ofJson`'s; `flock-firewall contract FILE…` prints `run`'s verdict
for each file.
-/
import Flock.Canon
import Flock.Hm96

namespace Flock.Firewall.Contract

open Lean (Json)

def FORMAT : String := "verity/firewall-transcript/v0"

/-- hm96-sha512's salt (192 bytes). -/
def SALT_BYTES : Nat := 192

/-- What fixes a case-(c) value before it leaves. -/
inductive By where
  /-- the session's public schedule: the contract derives the value from it (`scheduled`) -/
  | schedule
  /-- the statement's public shape (the outer gate's skeleton, `gate::skeleton`) -/
  | statement
  /-- the firewall's own commitment of its own witness (the outer gate's level-0 recheck) -/
  | recheck
  /-- the firewall's own opening of that commitment (the outer gate's shadow) -/
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
  /-- (c): the value `what`, fixed by `src` to `want` (for `By.schedule` the contract derives it and ignores `want`);
  `sent` leaves, and depends on the values masked under `pads` -/
  | fixed (what : String) (src : By) (want sent : ByteArray) (pads : Array Nat)
  /-- the firewall's check of `relation` on the values masked under `pads`, and its outcome (nothing leaves) -/
  | check (relation : String) (pads : Array Nat) (ok : Bool)
  /-- round `round` of `stream`'s coins from the challenger, `relayed` to the workers, with the challenger's `opening` -/
  | coins (stream : String) (round : Nat) (coins relayed opening : Array F128)
  /-- the session's end -/
  | finish
  deriving Inhabited

structure Transcript where
  phase : String
  /-- the public schedule the session is held to (the inner phase's `rec/inner-transcript/v0/schedule`) -/
  schedule : Option Json
  /-- every relation the verifier will check on the masked values -/
  relations : Array String
  /-- the firewall's tape: salt number `i` is `salts[i]`, pad number `i` is `pads[i]` -/
  salts : Array ByteArray
  pads : Array F128
  /-- the tape entries the firewall handed the workers -/
  exposedSalts : Array Nat
  exposedPads : Array Nat
  items : Array Item
  deriving Inhabited

def bytesEq (a b : ByteArray) : Bool := a.data == b.data

/-- (a)'s commit string: hm96-sha512's `b ‖ c` for the inner digest SHA-512(`msg`) under `salt`. -/
def commitBC (msg salt : ByteArray) : ByteArray := Hm96.Default512.commitString (Sha512.hash msg) salt

/-- (a)'s commitment as a tree leaf: hm96-sha512's leaf of SHA-512(`msg`) under `salt`, `gatewayLeaf`'s value. -/
def firewallLeaf (msg salt : ByteArray) : ByteArray := Hm96.Default512.leaf (Sha512.hash msg) salt

theorem firewallLeaf_eq (msg salt : ByteArray) : firewallLeaf msg salt = Hm96.Default512.treeLeaf (commitBC msg salt) := rfl

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

/-- The value the schedule fixes for `what`, and the order it holds `what` to. -/
def scheduled (sched : Json) (o : Order) (what : String) : V (ByteArray × Order) := do
  let need := fun (b : Bool) (why : String) => if b then pure () else throw (α := Unit) why
  if what == "hello" then
    need (o.stage == 0) "Hello: not the session's first item"
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
    let rounds ← J.asArr (← field sched "rounds") "the schedule's rounds"
    let r ← match rounds[o.rounds]? with
      | some r => pure r
      | none => throw s!"round {i}: past the schedule's {rounds.size} rounds"
    let s ← J.str r "stream" s!"the schedule's round {i}"
    need (o.opened.contains s) s!"round {i}: on stream {s}, not yet opened"
    return (← canonBytes r, { o with rounds := o.rounds + 1 })
  | _ => throw s!"{what}: not an item of the schedule"

def need (b : Bool) (why : String) : V Unit := if b then pure () else throw why

/-- Whether every relation has passed on the value masked under pad `p`. -/
def State.checked (s : State) (t : Transcript) (p : Nat) : Bool :=
  t.relations.all fun r => (s.passed.getD p #[]).contains r

/-- One item: the state after it, or why it may not leave. -/
def step (t : Transcript) (s : State) : Item → V State
  | .commit msg i sent => do
    need (i < t.salts.size) s!"a commitment under salt {i}, which the firewall never drew"
    need (!s.usedSalt[i]!) s!"a commitment under salt {i}, used before: not fresh"
    need (!t.exposedSalts.contains i) s!"a commitment under salt {i}, which the workers saw"
    let y := t.salts[i]!
    need (y.size == SALT_BYTES) s!"salt {i} is {y.size} bytes, not {SALT_BYTES}"
    need (bytesEq sent (commitBC msg y)) s!"a commitment under salt {i} that is not the message's"
    return { s with usedSalt := s.usedSalt.set! i true }
  | .masked v p sent => do
    need (p < t.pads.size) s!"a value masked with pad {p}, which the firewall never drew"
    need (!s.usedPad[p]!) s!"a value masked with pad {p}, used before: not fresh"
    need (!t.exposedPads.contains p) s!"a value masked with pad {p}, which the workers saw"
    need (decide (sent = v + t.pads[p]!)) s!"a value sent under pad {p} that is not the value masked with it"
    return { s with usedPad := s.usedPad.set! p true }
  | .check r ps ok => do
    need (t.relations.contains r) s!"a check of {r}, which is not a relation the verifier checks"
    need (ps.all fun p => s.usedPad.getD p false) s!"a check of {r} on a pad that masks nothing yet"
    need ok s!"the relation {r} fails"
    return { s with passed := ps.foldl (fun a p => a.modify p (·.push r)) s.passed }
  | .fixed what src want sent ps => do
    let (want, o) ← if src = .schedule then
        match t.schedule with
        | some sched => scheduled sched s.order what
        | none => throw s!"{what}: fixed by a schedule the transcript does not carry"
      else pure (want, s.order)
    need (bytesEq sent want) s!"{what}: not the value {src.name} fixes"
    need (ps.all fun p => s.usedPad.getD p false && s.checked t p)
      s!"{what}: depends on a masked value not yet checked against every relation"
    return { s with order := o }
  | .coins stream k cs relayed opening => do
    need (decide (relayed = cs)) s!"{stream} round {k}: the workers' coins are not the challenger's"
    need opening.isEmpty s!"{stream} round {k}: an opening of coins the challenger never committed to"
    return s
  | .finish => do
    need ((List.range t.pads.size).all fun p => !s.usedPad[p]! || s.checked t p)
      "Finish: a masked value not checked against every relation"
    return { s with finished := true }

/-- `run`'s verdict: the first `released` items may leave; `cut`, when some item may not, says why (the item at
`released`), or that the session ended without `finish`. -/
structure Verdict where
  released : Nat
  cut : Option String
  deriving Repr, Inhabited

def Verdict.accepted (v : Verdict) : Bool := v.cut.isNone

def go (t : Transcript) : State → List Item → Nat → Verdict
  | s, [], n => if s.finished then ⟨n, none⟩ else ⟨n, some "the session ended without Finish"⟩
  | s, it :: rest, n =>
    if s.finished then ⟨n, some "an item after Finish"⟩
    else match step t s it with
      | .ok s' => go t s' rest (n + 1)
      | .error why => ⟨n, some why⟩

/-- The contract on one transcript. -/
def run (t : Transcript) : Verdict := go t (State.init t) t.items.toList 0

/-- The contract holds of the transcript: every item may leave, and the session ends with `finish`. -/
def Holds (t : Transcript) : Prop := (run t).accepted = true

/-! ## What an accepted session lets out -/

/-- Every item of a list `go` lets through whole is one `step` takes from some state before Finish. -/
theorem go_steps {t : Transcript} : ∀ {s : State} {l : List Item} {n : Nat},
    (go t s l n).cut = none → ∀ it ∈ l, ∃ s₀ s₁, s₀.finished = false ∧ step t s₀ it = .ok s₁
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

/-- (a) item by item: a commitment leaves only as `b ‖ c` of its message under a drawn salt of 192 bytes, unused and
unseen by the workers. -/
theorem step_commit {t : Transcript} {s s' : State} {msg : ByteArray} {i : Nat} {sent : ByteArray}
    (h : step t s (.commit msg i sent) = .ok s') :
    i < t.salts.size ∧ s.usedSalt[i]! = false ∧ i ∉ t.exposedSalts ∧ (t.salts[i]!).size = SALT_BYTES ∧
      sent.data = (commitBC msg t.salts[i]!).data := by
  simp only [step, need, bind, Except.bind] at h
  repeat' (split at h <;> try simp_all [pure, Except.pure, throw, throwThe, MonadExceptOf.throw])
  all_goals simp_all [bytesEq]


/-- (b) item by item: a masked value leaves only under a drawn pad, unused and unseen by the workers. -/
theorem step_masked {t : Transcript} {s s' : State} {v : F128} {p : Nat} {sent : F128}
    (h : step t s (.masked v p sent) = .ok s') :
    p < t.pads.size ∧ s.usedPad[p]! = false ∧ p ∉ t.exposedPads ∧ sent = v + t.pads[p]! := by
  simp only [step, need, bind, Except.bind] at h
  repeat' (split at h <;> try simp_all [pure, Except.pure, throw, throwThe, MonadExceptOf.throw])

/-- The workers' coins are the challenger's. -/
theorem step_coins {t : Transcript} {s s' : State} {stream : String} {k : Nat} {cs relayed opening : Array F128}
    (h : step t s (.coins stream k cs relayed opening) = .ok s') : relayed = cs := by
  simp only [step, need, bind, Except.bind] at h
  repeat' (split at h <;> try simp_all [pure, Except.pure, throw, throwThe, MonadExceptOf.throw])

/-- (c) item by item: a fixed value leaves only as the schedule derives it (`By.schedule`) or as the value the transcript
reports fixing it, and after every pad it depends on has passed every relation. -/
theorem step_fixed {t : Transcript} {s s' : State} {what : String} {src : By} {want sent : ByteArray} {ps : Array Nat}
    (h : step t s (.fixed what src want sent ps) = .ok s') :
    (src = .schedule → ∃ sched w o, t.schedule = some sched ∧ scheduled sched s.order what = .ok (w, o) ∧
      sent.data = w.data) ∧
    (src ≠ .schedule → sent.data = want.data) ∧ ∀ p ∈ ps, s.usedPad.getD p false = true ∧ s.checked t p = true := by
  simp only [step, need, bind, Except.bind] at h
  repeat' (split at h <;> try simp_all [pure, Except.pure, throw, throwThe, MonadExceptOf.throw])
  all_goals simp_all [bytesEq, Array.mem_iff_getElem]
  all_goals rename_i _ hp
  · exact ⟨⟨_, ⟨_, rfl⟩, rfl⟩, fun p i hi e => e ▸ hp i hi⟩
  · exact fun p i hi e => e ▸ hp i hi

theorem step_finished {t : Transcript} {s s' : State} {it : Item} (h : step t s it = .ok s') (hit : it ≠ .finish) :
    s'.finished = s.finished := by
  cases it <;> simp only [step, need, bind, Except.bind] at h
  all_goals repeat' (split at h <;> try simp_all [pure, Except.pure, throw, throwThe, MonadExceptOf.throw])
  all_goals (subst h; rfl)

/-- A list `go` lets through whole, from a state before Finish, ends with Finish. -/
theorem go_last {t : Transcript} : ∀ {s : State} {l : List Item} {n : Nat},
    (go t s l n).cut = none → s.finished = false → l.getLast? = some .finish
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

theorem holds_go {t : Transcript} (h : Holds t) : (go t (State.init t) t.items.toList 0).cut = none := by
  simpa [Holds, run, Verdict.accepted, Option.isNone_iff_eq_none] using h

/-- An accepted session ends with Finish: a session that stops is never accepted. -/
theorem holds_finish {t : Transcript} (h : Holds t) : t.items.toList.getLast? = some .finish :=
  go_last (holds_go h) rfl

/-- (a) on an accepted session: every commitment that left is hm96-sha512's `b ‖ c` of its message under a salt the
firewall drew, of 192 bytes, that the workers never saw (`firewallLeaf` is its tree leaf). -/
theorem holds_commit {t : Transcript} (h : Holds t) {msg : ByteArray} {i : Nat} {sent : ByteArray}
    (hm : Item.commit msg i sent ∈ t.items.toList) :
    i < t.salts.size ∧ i ∉ t.exposedSalts ∧ (t.salts[i]!).size = SALT_BYTES ∧ sent.data = (commitBC msg t.salts[i]!).data := by
  obtain ⟨_, _, -, hs⟩ := go_steps (holds_go h) _ hm
  obtain ⟨a, -, b, c, d⟩ := step_commit hs
  exact ⟨a, b, c, d⟩

/-- (b) on an accepted session: every masked value that left is its value plus a pad the firewall drew that the workers
never saw. -/
theorem holds_masked {t : Transcript} (h : Holds t) {v : F128} {p : Nat} {sent : F128}
    (hm : Item.masked v p sent ∈ t.items.toList) : p < t.pads.size ∧ p ∉ t.exposedPads ∧ sent = v + t.pads[p]! := by
  obtain ⟨_, _, -, hs⟩ := go_steps (holds_go h) _ hm
  obtain ⟨a, -, b, c⟩ := step_masked hs
  exact ⟨a, b, c⟩

/-- On an accepted session the workers received exactly the challenger's coins. -/
theorem holds_coins {t : Transcript} (h : Holds t) {stream : String} {k : Nat} {cs relayed opening : Array F128}
    (hm : Item.coins stream k cs relayed opening ∈ t.items.toList) : relayed = cs := by
  obtain ⟨_, _, -, hs⟩ := go_steps (holds_go h) _ hm
  exact step_coins hs

/-- (c) on an accepted session: every fixed value that left is the schedule's own (`scheduled`), or for the other
sources the value the transcript reports fixing it. -/
theorem holds_fixed {t : Transcript} (h : Holds t) {what : String} {src : By} {want sent : ByteArray} {ps : Array Nat}
    (hm : Item.fixed what src want sent ps ∈ t.items.toList) :
    (src = .schedule → ∃ sched o w o', t.schedule = some sched ∧ scheduled sched o what = .ok (w, o') ∧ sent.data = w.data) ∧
    (src ≠ .schedule → sent.data = want.data) := by
  obtain ⟨s, _, -, hs⟩ := go_steps (holds_go h) _ hm
  obtain ⟨a, b, -⟩ := step_fixed hs
  refine ⟨fun e => ?_, b⟩
  obtain ⟨sched, w, o', h1, h2, h3⟩ := a e
  exact ⟨sched, s.order, w, o', h1, h2, h3⟩

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
      let opening ← if J.has v "opening" then f128s v "opening" ctx else pure #[]
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
  let schedule := match j.getObjVal? "schedule" with
    | .ok Json.null => none
    | .ok s => some s
    | .error _ => none
  let items ← (← J.arr j "items" "transcript").mapIdxM fun i v => Item.ofJson v s!"item {i}"
  let relations ← if J.has j "relations" then (← J.arr j "relations" "transcript").mapM (J.asStr · "relations")
    else pure #[]
  return { phase := ← J.str j "phase" "transcript", schedule, relations, salts, pads := ← f128s tape "pads" "tape",
           exposedSalts := ← nats exposed "salts" "exposed", exposedPads := ← nats exposed "pads" "exposed", items }

def Verdict.toJson (v : Verdict) : Json :=
  Json.mkObj [("accepted", Json.bool v.accepted), ("released", Json.num v.released),
    ("why", match v.cut with | some w => Json.str w | none => Json.null)]

/-- `flock-firewall contract FILE…` (`args` the files): each file's verdict, one JSON line each; exit 2 when a file is not
a transcript. -/
def cmd (args : List String) : IO UInt32 := do
  match args with
  | files@(_ :: _) =>
    let mut rc : UInt32 := 0
    for f in files do
      let text ← IO.FS.readFile f
      match Json.parse text >>= fun j => (Transcript.ofJson j).mapError id with
      | .ok t =>
        IO.println ((run t).toJson.setObjVal! "file" (Json.str f)).compress
      | .error e =>
        IO.println (Json.mkObj [("file", Json.str f), ("error", Json.str e)]).compress
        rc := 2
    return rc
  | _ =>
    IO.eprintln "usage: flock-firewall contract TRANSCRIPT.json..."
    return 2

end Flock.Firewall.Contract
