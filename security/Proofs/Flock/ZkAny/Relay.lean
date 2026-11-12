module

public import Flock.Firewall.Session

/-!
# The deployed relay's release is the schedule's

`Flock.Firewall.Session.relay` is the ZK server's session loop: it reads the worker's requests one at a time and decides
what leaves for the auditor from the session's state, with `State.request`, `State.place`, `State.answer`,
`State.sendHello` and `State.wire`. Here its loop body is `stepK`, which makes the same decisions with the same pure
functions. What the loop reads from its sockets, its rep frames and the operating system at a request is gathered in an
`Env`: the auditor's answer, a `Round`'s release by its rep's frame (`Frames.round`), a `Commit`'s roots
(`Level0.commit`), the rep's frame's take of the coins (`Frames.coins`), `Frames.finish`, the loop's own proofs
(`ownProofs`) and the hold's barrier. `out` runs `stepK` over the requests. `relay` is held to `stepK` by reading: each
of `relay`'s branches is one of `stepK`'s, in the same order. The worker is any: its requests are any list, each one
possibly chosen after reading the answers before it, and so are the answers.

* `slot S P`: the one kind of request the session lets out next. In order: `Register`, `Hello`, the first rep stream's
  `Open`, `Commit`, `Link`, each opening's `Round`, each later rep stream's `Open` once the rounds before it are past,
  and `Finish` once every stream is open. `schedOut` is the loop where each request is the slot's.
* `Reach`: what holds of the session at each request before a stop. The plan goes stream by stream in the rep
  streams' order (`Grouped`, `grouped_planOf`), and the opens, the `Link` and the openings agree with it.
  `reach_step` keeps it at every request that leaves.
* `request_slot`: before a stop, a request `State.request` lets through is the slot's.
* `out_stopped`: nothing leaves after a stop, where the hold's barrier refuses a stopped session (the deployed hold's
  does: `Core.Firewall.Decision.after` stops at the loop's refusal).
* `out_sched`: at every list of requests, what leaves is what leaves at the schedule's slots, at the requests that
  reach one (`pick`), a sublist of what the loop read. A request the session refuses sends nothing and stops it, and a
  `Ping` or a `Proof` sends nothing.
* `tmplOf s`: the schedule at the setup `s`, a function of the setup alone: `Register`, `Hello`, then `slotAt`'s slots
  from the start (`tmplFrom`, at the rep streams `labelsOf` and the plan `planOf` the `Hello` sets, `sendHello_eq`),
  each position (`Pos`: the opens, the openings answered, the `Link`) advanced by its slot's kind, up to `Finish`.
* `pos_step`: each request that leaves and keeps the session going advances the position by its kind. An answer to a
  coin request that is not coins stops the session (`State.noCoins`, `noCoins_stop`), so no slot comes twice.
* `sched_prefix`, `relay_prefix`: the slots of a run are the schedule cut after one of them, and each control slot sent
  its frame (`Framed`).
* `relay_release_pre`, `relay_release_take`: where what left at each model slot (the `Hello`, each coin request and the
  `Finish`) is the next of some frames `ms`, the last possibly cut where the release ended (`RelayPre`), what leaves for
  the auditor is the control frames at their fixed places (`ctlOf s`: the `Register`'s, the `Link`'s and each `Open`'s,
  functions of the setup) and `ms` at the model's, woven over the whole schedule (`weave`) and cut after the frames that
  left (`relayLen`), then nothing. So the release is a function of how many frames left and `ms` alone, whatever the
  worker sends and the auditor answers.

Scope: the released bytes and their order, not their times. When each frame leaves is not modeled.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open Flock.Firewall.Session
open Flock.Firewall hiding State
open Flock (V SessionSpec Stmt)

/-- **What the relay reads at one request beyond the request's frame and its own state**: the setup at the sample
(`Hello.Setup.setupAt`), the nonce and the id it samples from the operating system, the auditor's answer to what left
(`none` where the auditor closed first, or where the hold refused the session at its answer to `Register`), a
`Round`'s release by its rep's frame (`Frames.round`), a `Commit`'s roots bound by the tables' level 0
(`Level0.commit`), whether the rep's frame took a round's coins (`Frames.coins`), whether every rep's frame finished
(`Frames.finish`), the loop's own proofs (`ownProofs`), how many of them the auditor took before one it didn't, and
the hold's barrier at the stop it is given. -/
structure Env where
  setup : V (SessionSpec × Array Stmt)
  nonce : ByteArray
  id : ByteArray
  ans : Option ByteArray
  round : Except String ByteArray
  commit : Except String (List (String × ByteArray))
  coins : Except String Unit
  finish : Except String Unit
  proofs : List ByteArray
  taken : Nat
  barrier : Option String → Option String

/-- One request's effect: what left for the auditor, the session after, and whether the relay reads another request. -/
structure Res where
  sent : List ByteArray
  st : State
  pl : Placed
  go : Bool

/-- The proofs `held` and `Finish` leaving, cut after the first proof the auditor did not take, the `taken`-th. -/
def proofsOut (held : List ByteArray) (taken : Nat) : List ByteArray :=
  if taken < held.length then held.take (taken + 1) else held ++ [FINISH]

/-- **What leaves at a `Finish`**, with a hold or without. -/
def finishOut (hold : Bool) (st : State) (pl : Placed) (e : Env) : List ByteArray :=
  let fin := st.request pl .finish
  let stopped : Option String := match fin with
    | some (S', _) => S'.stop
    | none => match e.finish with
      | .ok () => none
      | .error why => some (stopOf why)
  let held := if stopped.isNone then e.proofs else []
  if hold then
    if (e.barrier stopped).isSome then [] else proofsOut held e.taken
  else match stopped with
    | some _ => if fin.isNone then [FINISH] else []
    | none => proofsOut held e.taken

/-- **The frame that leaves for a request of kind `k`** at its place: a `Round`'s release, a `Commit`'s roots, or
`State.wire`'s; else why the relay refuses it. -/
def frameOf (st : State) (k : Kind) (e : Env) : Except String ByteArray :=
  match k with
  | .round stream n => match e.round with
    | .ok out => .ok (roundFrame stream out n)
    | .error why => .error (stopOf why)
  | .commit => match e.commit with
    | .ok roots => .ok (commitFrame st.sigma roots)
    | .error why => .error s!"{Coins.STOP}{why}"
  | k => match st.wire k with
    | .ok w => .ok w
    | .error why => .error s!"{Coins.STOP}{why}"

/-- The session once the auditor answered `r` to what left for a request of kind `k`: `State.answer`, then for a
`Round`'s coins its rep's frame's take of them. -/
def afterAnswer (st : State) (k : Kind) (r : ByteArray) (e : Env) : State :=
  let st' := (st.answer k r).1
  match k with
  | .round _ _ =>
    if byte? r 0 == some 2 && st'.stop.isNone then
      match e.coins with
      | .ok () => st'
      | .error why => (st'.refuse (stopOf why)).1
    else st'
  | _ => st'

/-- **One request of the relay** (`relay`'s loop body), its first byte `b` and kind `k`. -/
def stepK (hold : Bool) (st : State) (pl : Placed) (b : Option UInt8) (k : Except String Kind) (e : Env) : Res :=
  if b == some 9 then ⟨[], st, pl, true⟩
  else if b == some 5 && st.stop.isNone then ⟨[], st, pl, true⟩
  else if b == some 6 then ⟨finishOut hold st pl e, st, pl, false⟩
  else match k with
  | .error _ => ⟨[], st, pl, false⟩
  | .ok .hello => match st.request pl .hello with
    | some (st', _) => ⟨[], st', pl, true⟩
    | none => match st.sendHello e.setup e.nonce with
      | .error (st', _) => ⟨[], st', pl, true⟩
      | .ok (st', h) => match e.ans with
        | none => ⟨[h], st', pl, false⟩
        | some r => ⟨[h], (st'.answer .hello r e.id).1, pl, true⟩
  | .ok k => match st.request pl k with
    | some (st', _) => ⟨[], st', pl, true⟩
    | none => match frameOf st k e with
      | .error why => ⟨[], (st.refuse why).1, pl, true⟩
      | .ok g => match e.ans with
        | none => ⟨[g], st, pl.after k, false⟩
        | some r => ⟨[g], afterAnswer st k r e, pl.after k, true⟩

/-- **What leaves for the auditor** over the requests `ins`, each with what the relay read at it. -/
def out (hold : Bool) : State → Placed → List (ByteArray × Env) → List ByteArray
  | _, _, [] => []
  | st, pl, (f, e) :: rest =>
    let r := stepK hold st pl (byte? f 0) (kind f) e
    r.sent ++ if r.go then out hold r.st r.pl rest else []

/-- A request kind's first byte (`Req::encode`'s tag). -/
def tagOf : Kind → UInt8
  | .register => 10
  | .hello => 1
  | .commit => 7
  | .finish => 6
  | .link => 2
  | .openStream _ => 3
  | .round _ _ => 4
  | .proof => 5
  | .ping => 9
  | .other => 0

/-- How many openings the auditor answered. -/
def opened (S : State) : Nat := (S.session.map (·.opened.length)).getD 0

/-- **The session's next place**: the one kind of request that can leave next, or none. -/
def slot (S : State) (P : Placed) : Option Kind :=
  if !P.register then some .register
  else if !S.sent then some .hello
  else if P.opens = 0 then S.labels[0]?.map fun l => .openStream l.1
  else if opened S = 0 then some .commit
  else if !P.link then some .link
  else match S.plan[opened S - 1]? with
    | some (s, n) =>
      if (S.labels.extract 0 P.opens).any (·.1 == s) then some (.round s n)
      else S.labels[P.opens]?.map fun l => .openStream l.1
    | none => match S.labels[P.opens]? with
      | some l => some (.openStream l.1)
      | none => if opened S = S.plan.size + 1 then some .finish else none

/-- **What leaves at the schedule's places**: at each of `es` in turn, a request of the session's next place's kind. -/
def schedOut (hold : Bool) : State → Placed → List Env → List ByteArray
  | _, _, [] => []
  | st, pl, e :: rest => match slot st pl with
    | none => []
    | some k =>
      let r := stepK hold st pl (some (tagOf k)) (.ok k) e
      r.sent ++ if r.go then schedOut hold r.st r.pl rest else []

/-- The rep streams' names. -/
def names (S : State) : List String := S.labels.toList.map (·.1)

/-- **The schedule's openings go stream by stream**, in the rep streams' order. -/
def Grouped (plan : Array (String × Nat)) (names : List String) : Prop :=
  ∀ (i j : Nat) (x y : String × Nat), i ≤ j → plan[i]? = some x → plan[j]? = some y → names.idxOf x.1 ≤ names.idxOf y.1

/-- **What holds of the session at each request** the relay reads before a stop. -/
structure Reach (S : State) (P : Placed) : Prop where
  reg : S.sent = true → P.register = true
  unsent : S.sent = false → S.labels = #[] ∧ S.session = none
  sess : S.sent = true → S.session.isSome
  grouped : Grouped S.plan (names S)
  size : S.labels.size ≠ 1
  opens_le : P.opens ≤ S.labels.size
  opens_pos : 1 ≤ opened S → 1 ≤ P.opens
  link_pos : P.link = true → 1 ≤ opened S
  link_two : 2 ≤ opened S → P.link = true
  link_opens : 2 ≤ P.opens → P.link = true
  le_plan : opened S ≤ S.plan.size + 1
  idx : ∀ (i : Nat) (x : String × Nat), opened S = i + 1 → S.plan[i]? = some x → P.opens - 1 ≤ (names S).idxOf x.1

theorem refuse_stop (S : State) (why : String) : (S.refuse why).1.stop = some why := rfl

theorem finish_stop {S S' : State} {o : ByteArray} (h : S.stop = none) (hf : S.finish = some (S', o)) :
    S'.stop.isSome := by
  revert hf
  unfold State.finish State.refuse
  rw [h]
  split
  · rename_i heq; cases heq
  · intro hf; cases hf; rfl
  · split
    · simp
    · intro hf; cases hf; rfl

/-- **A request refused before a stop stops the session.** -/
theorem request_stop {S S' : State} {P : Placed} {k : Kind} {o : ByteArray} (h : S.stop = none)
    (hr : S.request P k = some (S', o)) : S'.stop.isSome := by
  revert hr
  unfold State.request
  rw [h]
  dsimp only
  split
  · rename_i r hc
    intro hr
    simp only [Option.some.injEq] at hr
    subst hr
    revert hc
    split
    · exact finish_stop h
    · unfold State.refuse
      repeat' split
      all_goals intro hc; first
        | (simp only [Option.some.injEq, Prod.mk.injEq] at hc; rw [← hc.1]; rfl)
        | simp at hc
  · rcases hp : S.place P k with _ | why
    · simp
    · simp only [Option.map_some, Option.some.injEq]
      intro hr; cases hr; rfl

theorem request_stopped {S : State} {why : String} (P : Placed) (k : Kind) (h : S.stop = some why) :
    S.request P k = some (S, refused why) := by
  unfold State.request; rw [h]

/-- **A refused `Finish` sends nothing**, where the hold refuses a stopped session's release. -/
theorem finishOut_of_some {hold : Bool} {st S' : State} {pl : Placed} {e : Env} {o : ByteArray}
    (hb : ∀ why, (e.barrier (some why)).isSome) (hf : st.request pl .finish = some (S', o)) (hs : S'.stop.isSome) :
    finishOut hold st pl e = [] := by
  obtain ⟨why, hw⟩ := Option.isSome_iff_exists.mp hs
  unfold finishOut
  simp only [hf, hw, Option.isNone_some]
  cases hold
  · rfl
  · simp only [hb why, ite_true]

theorem stepK_stopped {hold : Bool} {st : State} {pl : Placed} {b : Option UInt8} {k : Except String Kind}
    {e : Env} (hs : st.stop.isSome) (hb : ∀ why, (e.barrier (some why)).isSome) :
    (stepK hold st pl b k e).sent = [] ∧ (stepK hold st pl b k e).st.stop.isSome := by
  obtain ⟨why, hw⟩ := Option.isSome_iff_exists.mp hs
  unfold stepK
  split
  · exact ⟨rfl, hs⟩
  split
  · exact ⟨rfl, hs⟩
  split
  · exact ⟨finishOut_of_some hb (request_stopped pl .finish hw) hs, hs⟩
  rcases k with _ | k
  · exact ⟨rfl, hs⟩
  rcases k <;> simp only [request_stopped _ _ hw] <;> exact ⟨trivial, hs⟩

/-- **Nothing leaves after a stop.** -/
theorem out_stopped (hold : Bool) : ∀ (st : State) (pl : Placed) (ins : List (ByteArray × Env)),
    st.stop.isSome → (∀ p ∈ ins, ∀ why, (p.2.barrier (some why)).isSome) → out hold st pl ins = []
  | _, _, [], _, _ => rfl
  | st, pl, (f, e) :: rest, hs, hb => by
    obtain ⟨h1, h2⟩ := stepK_stopped (hold := hold) (pl := pl) (b := byte? f 0) (k := kind f) (e := e) hs
      (hb (f, e) (List.mem_cons_self ..))
    simp only [out, h1, List.nil_append]
    split
    · exact out_stopped hold _ _ rest h2 fun p hp => hb p (List.mem_cons_of_mem _ hp)
    · rfl

theorem beq_finish (k : Kind) : (k == .finish) = true ↔ k = .finish := by
  cases k <;> first | exact ⟨fun _ => rfl, fun _ => rfl⟩ | exact ⟨fun h => (nomatch h), fun h => (nomatch h)⟩

theorem beq_commit (k : Kind) : (k == .commit) = true ↔ k = .commit := by
  cases k <;> first | exact ⟨fun _ => rfl, fun _ => rfl⟩ | exact ⟨fun h => (nomatch h), fun h => (nomatch h)⟩

/-- `State.request`'s coin check: whether a request of kind `k` is the schedule's next opening. -/
def coinsOf (S : State) (k : Kind) : Option (State × ByteArray) :=
  if k == .finish then S.finish else
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

theorem request_none {S : State} {P : Placed} {k : Kind} (h : S.stop = none) (hr : S.request P k = none) :
    coinsOf S k = none ∧ S.place P k = none := by
  have e : S.request P k = match coinsOf S k with
      | some r => some r
      | none => (S.place P k).map fun why => S.refuse s!"{Coins.STOP}{why}" := by
    unfold State.request; rw [h]; rfl
  rw [e] at hr
  rcases hc : coinsOf S k with _ | r
  · rw [hc] at hr
    exact ⟨rfl, Option.map_eq_none_iff.mp hr⟩
  · rw [hc] at hr; simp at hr

theorem finish_none {S : State} (h : S.stop = none) (hf : S.finish = none) :
    ∃ C, S.session = some C ∧ C.opened.length = S.plan.size + 1 := by
  unfold State.finish at hf
  rw [h] at hf
  rcases hs : S.session with _ | C
  · rw [hs] at hf; simp at hf
  · rw [hs] at hf
    by_cases hC : C.opened.length = S.plan.size + 1
    · exact ⟨C, rfl, hC⟩
    · simp [hC] at hf

theorem coins_finish {S : State} (h : S.stop = none) (hc : coinsOf S .finish = none) :
    ∃ C, S.session = some C ∧ C.opened.length = S.plan.size + 1 :=
  finish_none h hc

theorem coins_commit {S : State} (hc : coinsOf S .commit = none) :
    ∃ C, S.session = some C ∧ C.opened.length = 0 := by
  revert hc
  unfold coinsOf
  simp only [show (Kind.commit == Kind.finish) = false from rfl, show (Kind.commit == Kind.commit) = true from rfl]
  rcases hs : S.session with _ | C
  · simp [State.refuse]
  · intro hc
    refine ⟨C, rfl, ?_⟩
    by_cases hne : C.opened.length = 0
    · exact hne
    · simp [hne] at hc

theorem coins_round {S : State} {s : String} {n : Nat} (hc : coinsOf S (.round s n) = none) :
    ∃ C, S.session = some C ∧ 1 ≤ C.opened.length ∧ S.plan[C.opened.length - 1]? = some (s, n) := by
  revert hc
  unfold coinsOf
  simp only [show (Kind.round s n == Kind.finish) = false from rfl,
    show (Kind.round s n == Kind.commit) = false from rfl]
  rcases hs : S.session with _ | C
  · simp [State.refuse]
  · intro hc
    refine ⟨C, rfl, ?_⟩
    by_cases h0 : C.opened.length = 0
    · simp [h0] at hc
    by_cases hp : C.opened.length > S.plan.size
    · simp [h0, hp] at hc
    simp only [h0, hp, ite_false, Bool.false_eq_true] at hc
    refine ⟨Nat.one_le_iff_ne_zero.mpr h0, ?_⟩
    unfold reqAt at hc
    simp only [h0, ite_false] at hc
    split at hc
    · rename_i hq
      simp only [Bool.and_eq_true, beq_iff_eq] at hq
      rw [Array.getElem?_eq_getElem (by omega)]
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega)] at hq
      simp only [Option.getD_some] at hq
      rw [hq.1, hq.2]
    · simp at hc

theorem idxOf_le : ∀ (l : List String) (i : Nat) (h : i < l.length), l.idxOf l[i] ≤ i
  | [], _, h => absurd h (Nat.not_lt_zero _)
  | x :: xs, 0, _ => by simp
  | x :: xs, i + 1, h => by
    rw [List.idxOf_cons]
    split
    · omega
    · have := idxOf_le xs i (by simp at h; omega)
      simp only [List.getElem_cons_succ]
      omega

/-- A stream among the first `n` rep streams is at an index below `n`. -/
theorem idxOf_lt_of_any {S : State} {n : Nat} {s : String}
    (h : (S.labels.extract 0 n).any (·.1 == s) = true) : (names S).idxOf s < n := by
  obtain ⟨i, hi, hp⟩ := Array.any_eq_true.mp h
  rw [Array.size_extract] at hi
  simp only [Array.getElem_extract, Nat.zero_add] at hp
  have hs : (names S)[i]'(by simp [names]; omega) = s := by
    simp only [names, List.getElem_map, Array.getElem_toList]
    exact beq_iff_eq.mp hp
  have := idxOf_le (names S) i (by simp [names]; omega)
  rw [hs] at this
  omega

theorem sent_of_session {S : State} {P : Placed} (hR : Reach S P) (hs : S.session.isSome) : S.sent = true := by
  cases h : S.sent
  · rw [(hR.unsent h).2] at hs; simp at hs
  · rfl

theorem sent_of_labels {S : State} {P : Placed} (hR : Reach S P) (hs : 0 < S.labels.size) : S.sent = true := by
  cases h : S.sent
  · rw [(hR.unsent h).1] at hs; simp at hs
  · rfl

theorem opened_of {S : State} {C : Coins.Release} (hs : S.session = some C) : opened S = C.opened.length := by
  simp [opened, hs]

/-- **A request that leaves is at the session's next place.** -/
theorem request_slot {S : State} {P : Placed} {k : Kind} (hR : Reach S P) (h : S.stop = none)
    (hk : k ≠ .ping) (hk' : k ≠ .proof) (hr : S.request P k = none) : slot S P = some k := by
  obtain ⟨hc, hp⟩ := request_none h hr
  cases k with
  | register =>
    have : P.register = false := by revert hp; simp only [State.place]; cases P.register <;> simp
    simp [slot, this]
  | hello =>
    have h1 : P.register = true ∧ S.sent = false := by
      revert hp; simp only [State.place]; cases P.register <;> cases S.sent <;> simp
    simp [slot, h1.1, h1.2]
  | commit =>
    obtain ⟨C, hs, h0⟩ := coins_commit hc
    have hsent := sent_of_session hR (by simp [hs])
    have hop : P.opens ≠ 0 := by revert hp; simp only [State.place]; intro hp hz; simp [hz] at hp
    simp [slot, hR.reg hsent, hsent, hop, opened_of hs, h0]
  | link =>
    have h1 : S.sent = true ∧ P.link = false ∧ opened S = 1 := by
      revert hp; simp only [State.place, opened]
      cases S.sent <;> cases P.link <;> simp
    have hop : P.opens ≠ 0 := by have := hR.opens_pos (by omega); omega
    simp [slot, hR.reg h1.1, h1.1, h1.2.1, h1.2.2, hop]
  | round s n =>
    obtain ⟨C, hs, h1, hq⟩ := coins_round hc
    have hsent := sent_of_session hR (by simp [hs])
    have h2 : P.link = true ∧ (S.labels.extract 0 P.opens).any (·.1 == s) = true := by
      revert hp; simp only [State.place]
      cases P.link <;> simp
    have hop : P.opens ≠ 0 := by have := hR.opens_pos (by rw [opened_of hs]; omega); omega
    have hj : opened S ≠ 0 := by rw [opened_of hs]; omega
    simp only [slot, hR.reg hsent, hsent, hop, hj, h2.1, Bool.not_true, Bool.false_eq_true, ite_false]
    rw [opened_of hs, hq]
    simp only [h2.2, ↓reduceIte]
  | finish =>
    obtain ⟨C, hs, hlen⟩ := coins_finish h hc
    have hsent := sent_of_session hR (by simp [hs])
    have hsz : S.labels.size ≤ P.opens := by
      revert hp; simp only [State.place]; intro hp; apply Nat.le_of_not_lt; intro hz; simp [hz] at hp
    have hj : opened S = S.plan.size + 1 := by rw [opened_of hs, hlen]
    have hop : 1 ≤ P.opens := hR.opens_pos (by omega)
    have hlink : P.link = true := by
      by_cases h2 : 2 ≤ opened S
      · exact hR.link_two h2
      · apply hR.link_opens
        have := hR.opens_le; have := hR.size; omega
    have hpl : S.plan[S.plan.size + 1 - 1]? = none := by simp
    have hlb : S.labels[P.opens]? = none := by simp; omega
    have hop' : P.opens ≠ 0 := by omega
    simp only [slot, hR.reg hsent, hsent, hlink, hop', hpl, hlb, hj, Bool.not_true, Bool.false_eq_true, ite_false]
    simp
  | openStream s =>
    have hj : (S.session.map (·.opened.length)).getD 0 = opened S := rfl
    have hp' := hp
    simp only [State.place, hj] at hp'
    rcases hf : S.labels.findIdx? (·.1 == s) with _ | i
    · rw [hf] at hp'; simp at hp'
    rw [hf] at hp'
    obtain ⟨hi, hpi, -⟩ := Array.findIdx?_eq_some_iff_getElem.mp hf
    have hsent := sent_of_labels hR (by omega)
    have hs' : S.labels[i].1 = s := beq_iff_eq.mp hpi
    have hio : i = P.opens := by
      by_cases a : i < P.opens
      · simp [a] at hp'
      by_cases b : P.opens < i
      · simp [a, b] at hp'
      omega
    have hlb : S.labels[P.opens]? = some S.labels[i] := by rw [← hio]; simp
    by_cases h0 : i = 0
    · have hz : P.opens = 0 := by omega
      simp only [slot, hR.reg hsent, hsent, hz, Bool.not_true, Bool.false_eq_true, ite_false, ite_true]
      rw [← hz, hlb]
      simp [hs']
    have hlt : ¬ i < P.opens := by omega
    have hgt : ¬ P.opens < i := by omega
    simp only [hlt, hgt, h0, ite_false] at hp'
    have hlink : P.link = true := by revert hp'; cases P.link <;> simp
    simp only [hlink, Bool.not_true, Bool.false_eq_true, ite_false] at hp'
    have hne : ¬ ((S.plan[opened S - 1]?.map (·.1)) == some ((S.labels[i - 1]?.map (·.1)).getD "")) = true := by
      intro hc; simp [hc] at hp'
    have hop : P.opens ≠ 0 := by omega
    have hj0 : opened S ≠ 0 := by have := hR.link_pos hlink; omega
    simp only [slot, hR.reg hsent, hsent, hop, hj0, hlink, Bool.not_true, Bool.false_eq_true, ite_false]
    rcases hq : S.plan[opened S - 1]? with _ | ⟨s', n'⟩
    · simp [hlb, hs']
    · have hany : ¬ (S.labels.extract 0 P.opens).any (·.1 == s') = true := by
        intro ha
        have h1 := idxOf_lt_of_any ha
        have h2 := hR.idx (opened S - 1) (s', n') (by omega) hq
        have hnl : (names S).length = S.labels.size := by simp [names]
        have hlen : (names S).idxOf s' < (names S).length := by have := hR.opens_le; omega
        have he : (names S).idxOf s' = i - 1 := by simp only at h2; omega
        have key : ∀ m (hm : m < (names S).length), m = (names S).idxOf s' → (names S)[m] = s' := by
          intro m hm e; subst e; exact List.getElem_idxOf hlen
        have h3 := key (i - 1) (by omega) he.symm
        simp only [names, List.getElem_map, Array.getElem_toList] at h3
        apply hne
        rw [hq, Array.getElem?_eq_getElem (show i - 1 < S.labels.size by omega)]
        simp [h3]
      simp only [hany, hlb, Option.map_some]
      rw [hs']
      rfl
  | proof => exact absurd rfl hk'
  | ping => exact absurd rfl hk
  | other => simp [State.place] at hp

/-- **A request of a kind that carries its own frame**, at its place: its frame leaves, and the session goes on as the
auditor's answer leaves it. -/
theorem stepK_frame {hold : Bool} {st : State} {pl : Placed} {k : Kind} {e : Env} (hk1 : k ≠ .hello)
    (hk2 : k ≠ .finish) (hk3 : k ≠ .ping) (hk4 : k ≠ .proof) (hr : st.request pl k = none) :
    stepK hold st pl (some (tagOf k)) (.ok k) e = match frameOf st k e with
      | .error why => ⟨[], (st.refuse why).1, pl, true⟩
      | .ok g => match e.ans with
        | none => ⟨[g], st, pl.after k, false⟩
        | some r => ⟨[g], afterAnswer st k r e, pl.after k, true⟩ := by
  cases k <;> first
    | exact absurd rfl hk1 | exact absurd rfl hk2 | exact absurd rfl hk3 | exact absurd rfl hk4
    | simp only [stepK, tagOf, hr] <;> rfl

theorem stepK_frame_go {hold : Bool} {st : State} {pl : Placed} {k : Kind} {e : Env} (hk1 : k ≠ .hello)
    (hk2 : k ≠ .finish) (hk3 : k ≠ .ping) (hk4 : k ≠ .proof) (hr : st.request pl k = none)
    (hgo : (stepK hold st pl (some (tagOf k)) (.ok k) e).go = true) :
    (stepK hold st pl (some (tagOf k)) (.ok k) e).st.stop.isSome ∨
      ∃ r, (stepK hold st pl (some (tagOf k)) (.ok k) e).st = afterAnswer st k r e ∧
        (stepK hold st pl (some (tagOf k)) (.ok k) e).pl = pl.after k := by
  rw [stepK_frame hk1 hk2 hk3 hk4 hr] at hgo ⊢
  revert hgo
  rcases frameOf st k e with why | g
  · intro _; exact .inl rfl
  · rcases e.ans with _ | r
    · intro hgo; simp at hgo
    · intro _; exact .inr ⟨r, rfl, rfl⟩

theorem stepK_frame_ans {hold : Bool} {st : State} {pl : Placed} {k : Kind} {e : Env} (hk1 : k ≠ .hello)
    (hk2 : k ≠ .finish) (hk3 : k ≠ .ping) (hk4 : k ≠ .proof) (hr : st.request pl k = none)
    (hgo : (stepK hold st pl (some (tagOf k)) (.ok k) e).go = true) :
    (stepK hold st pl (some (tagOf k)) (.ok k) e).st.stop.isSome ∨
      ∃ r, e.ans = some r ∧ (stepK hold st pl (some (tagOf k)) (.ok k) e).st = afterAnswer st k r e ∧
        (stepK hold st pl (some (tagOf k)) (.ok k) e).pl = pl.after k := by
  rw [stepK_frame hk1 hk2 hk3 hk4 hr] at hgo ⊢
  revert hgo
  rcases frameOf st k e with why | g
  · intro _; exact .inl rfl
  · rcases ha : e.ans with _ | r
    · intro hgo; simp at hgo
    · intro _; exact .inr ⟨r, rfl, rfl, rfl⟩

theorem open_ok {C C' : Coins.Release} {s : String} {n : Nat} {m c : List UInt8}
    (h : C.open s n m = .ok (C', c)) : C'.opened.length = C.opened.length + 1 ∧ C'.hello = C.hello := by
  revert h
  unfold Coins.Release.open
  dsimp only
  split
  · split
    · intro h; cases h; simp
    · intro h; cases h
  · intro h; cases h

/-- A coin's answer: the session stopped, or the next opening counted. -/
theorem coins_cases {st : State} {r : ByteArray} {C : Coins.Release} (h : st.stop = none) (hs : st.session = some C) :
    ((st.coins r).1.stop.isSome) ∨ ∃ C', (st.coins r).1 = { st with session := some C' } ∧
      C'.opened.length = C.opened.length + 1 ∧ C'.hello = C.hello := by
  unfold State.coins
  rw [h, hs]
  dsimp only
  rcases ho : C.open (reqAt st.plan C.hello.links C.opened.length).1 (reqAt st.plan C.hello.links C.opened.length).2
      r.data.toList with e | ⟨C', c⟩
  · exact .inl rfl
  · exact .inr ⟨C', rfl, open_ok ho⟩

theorem grouped_of_pairwise {plan : Array (String × Nat)} {names : List String}
    (h : plan.toList.Pairwise fun x y => names.idxOf x.1 ≤ names.idxOf y.1) : Grouped plan names := by
  intro i j x y hij hx hy
  rw [← Array.getElem?_toList] at hx hy
  obtain ⟨hi, rfl⟩ := List.getElem?_eq_some_iff.mp hx
  obtain ⟨hj, rfl⟩ := List.getElem?_eq_some_iff.mp hy
  rcases Nat.lt_or_eq_of_le hij with hlt | rfl
  · exact List.pairwise_iff_getElem.mp h i j hi hj hlt
  · exact Nat.le_refl _

/-- **The plan goes stream by stream**, in the rep streams' order, where their names are distinct. -/
theorem grouped_planOf {spec : SessionSpec} {sts : Array Stmt}
    (hn : (spec.repStreams.toList.map (·.2.2)).Nodup) :
    Grouped (planOf spec sts) (spec.repStreams.toList.map (·.2.2)) := by
  apply grouped_of_pairwise
  simp only [planOf, Array.toList_flatMap, Array.toList_range, Array.toList_map]
  rw [List.pairwise_flatMap]
  have hname : ∀ k, k < spec.repStreams.size →
      (spec.repStreams.toList.map (·.2.2)).idxOf (repName spec k) = k := by
    intro k hk
    have hl : k < (spec.repStreams.toList.map (·.2.2)).length := by simp; exact hk
    have : repName spec k = (spec.repStreams.toList.map (·.2.2))[k] := by
      simp [repName, Array.getElem?_eq_getElem hk]
    rw [this, List.Nodup.idxOf_getElem hn k hl]
  refine ⟨fun k _ => ?_, ?_⟩
  · apply List.Pairwise.imp_of_mem (R := fun _ _ => True) _
      (List.pairwise_iff_getElem.mpr fun _ _ _ _ _ => trivial)
    intro a b ha hb _
    obtain ⟨_, _, rfl⟩ := List.mem_map.mp ha
    obtain ⟨_, _, rfl⟩ := List.mem_map.mp hb
    exact Nat.le_refl _
  · apply List.Pairwise.imp_of_mem _ List.pairwise_lt_range
    intro a b ha hb hab x hx y hy
    obtain ⟨_, _, rfl⟩ := List.mem_map.mp hx
    obtain ⟨_, _, rfl⟩ := List.mem_map.mp hy
    rw [List.mem_range] at ha hb
    rw [hname a ha, hname b hb]
    exact Nat.le_of_lt hab

theorem slot_register {S : State} {P : Placed} (h : slot S P = some .register) : P.register = false := by
  unfold slot at h
  revert h
  cases P.register
  · intro _; rfl
  · simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte]
    repeat' split
    all_goals simp

theorem slot_hello {S : State} {P : Placed} (h : slot S P = some .hello) : P.register = true ∧ S.sent = false := by
  unfold slot at h
  revert h
  cases P.register <;> cases S.sent
  all_goals simp only [Bool.not_true, Bool.not_false, Bool.false_eq_true, ↓reduceIte]
  all_goals first | (intro _; exact ⟨rfl, rfl⟩) | (repeat' split) <;> simp

theorem slot_commit {S : State} {P : Placed} (h : slot S P = some .commit) :
    S.sent = true ∧ P.opens ≠ 0 ∧ opened S = 0 := by
  unfold slot at h
  split at h; · simp at h
  split at h; · simp at h
  rename_i h2
  split at h; · simp at h
  rename_i h3
  split at h
  · exact ⟨by simpa using h2, h3, ‹_›⟩
  · split at h; · simp at h
    repeat' split at h
    all_goals simp at h

theorem slot_link {S : State} {P : Placed} (h : slot S P = some .link) :
    opened S ≠ 0 ∧ P.link = false := by
  unfold slot at h
  split at h; · simp at h
  split at h; · simp at h
  split at h; · simp at h
  split at h; · simp at h
  rename_i h4
  split at h
  · rename_i h5
    exact ⟨h4, by simpa using h5⟩
  · repeat' split at h
    all_goals simp at h

theorem slot_round {S : State} {P : Placed} {s : String} {n : Nat} (h : slot S P = some (.round s n)) :
    S.sent = true ∧ opened S ≠ 0 ∧ P.link = true ∧ S.plan[opened S - 1]? = some (s, n) := by
  unfold slot at h
  split at h; · simp at h
  split at h; · simp at h
  rename_i h2
  split at h; · simp at h
  split at h; · simp at h
  rename_i h4
  split at h; · simp at h
  rename_i h5
  split at h
  · rename_i s' n' hq
    split at h
    · simp only [Option.some.injEq, Kind.round.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨by simpa using h2, h4, by simpa using h5, hq⟩
    · simp at h
  · repeat' split at h
    all_goals simp at h

theorem slot_open {S : State} {P : Placed} {s : String} (h : slot S P = some (.openStream s)) :
    ∃ hl : P.opens < S.labels.size, S.labels[P.opens].1 = s ∧
      (P.opens = 0 ∨ (P.link = true ∧ ∀ i x, opened S = i + 1 → S.plan[i]? = some x →
        (S.labels.extract 0 P.opens).any (·.1 == x.1) = false)) := by
  unfold slot at h
  split at h; · simp at h
  split at h; · simp at h
  split at h
  · rename_i h3
    obtain ⟨l, hl, hk⟩ := Option.map_eq_some_iff.mp h
    obtain ⟨hlt, rfl⟩ := Array.getElem?_eq_some_iff.mp hl
    cases hk
    exact ⟨by rw [h3]; exact hlt, by simp only [h3], .inl h3⟩
  rename_i h3
  split at h; · simp at h
  split at h; · simp at h
  rename_i h5
  have hlink : P.link = true := by simpa using h5
  split at h
  · rename_i s' n' hq
    split at h; · simp at h
    rename_i hany
    obtain ⟨l, hl, hk⟩ := Option.map_eq_some_iff.mp h
    obtain ⟨hlt, rfl⟩ := Array.getElem?_eq_some_iff.mp hl
    cases hk
    refine ⟨hlt, rfl, .inr ⟨hlink, fun i x hi hx => ?_⟩⟩
    rw [hi, Nat.add_sub_cancel] at hq
    rw [hq] at hx
    cases hx
    simpa using hany
  · rename_i hq
    split at h
    · rename_i l hl
      cases h
      obtain ⟨hlt, rfl⟩ := Array.getElem?_eq_some_iff.mp hl
      refine ⟨hlt, rfl, .inr ⟨hlink, fun i x hi hx => ?_⟩⟩
      rw [hi, Nat.add_sub_cancel] at hq
      rw [hq] at hx
      cases hx
    · split at h <;> simp at h

theorem answer_register (S : State) (r : ByteArray) :
    (S.answer .register r).1.stop.isSome ∨ ∃ d, (S.answer .register r).1 = { S with draw := d } := by
  unfold State.answer
  split
  · repeat' split
    all_goals first | exact .inl rfl | exact .inr ⟨_, rfl⟩
  all_goals first | exact .inr ⟨S.draw, rfl⟩ | simp_all

theorem answer_link (S : State) (r : ByteArray) : (S.answer .link r).1 = S := by
  unfold State.answer; split <;> simp_all

theorem answer_open (S : State) (s : String) (r : ByteArray) : (S.answer (.openStream s) r).1 = S := by
  unfold State.answer; split <;> simp_all

theorem noCoins_stop (S : State) (what : String) (a : ByteArray) : (S.noCoins what a).1.stop.isSome := by
  unfold State.noCoins
  split
  · rename_i h; simp [h]
  · first | rfl | simp [State.refuse]

theorem answer_commit (S : State) (r : ByteArray) :
    (S.answer .commit r).1 =
      if byte? r 0 = some 2 then (S.coins r).1 else (S.noCoins "the step-0 coin commitment" r).1 := by
  unfold State.answer; split <;> simp_all

theorem answer_round (S : State) (s : String) (n : Nat) (r : ByteArray) :
    (S.answer (.round s n) r).1 =
      if byte? r 0 = some 2 then (S.coins r).1 else (S.noCoins s!"a round of {s}" r).1 := by
  unfold State.answer; split <;> simp_all

theorem start_opened {h : Coins.Hello} {a : List UInt8} {C : Coins.Release} (hs : Coins.Release.start h a = .ok C) :
    C.opened = [] := by
  revert hs
  unfold Coins.Release.start
  split
  · intro hs; cases hs; rfl
  · intro hs; cases hs

theorem answer_hello {S : State} {r id : ByteArray} (h : S.stop = none) :
    (S.answer .hello r id).1.stop.isSome ∨
      ∃ hh C, S.hello = some hh ∧ (S.answer .hello r id).1 = { S with session := some C } ∧ C.opened = [] := by
  unfold State.answer
  simp only [h]
  rcases hhel : S.hello with _ | hh
  · dsimp only
    split <;> exact .inl rfl
  · dsimp only
    rcases hs : Coins.Release.start hh r.data.toList with e | C
    · exact .inl rfl
    · exact .inr ⟨hh, C, rfl, rfl, start_opened hs⟩

theorem sendHello_error {S S' : State} {s : V (SessionSpec × Array Stmt)} {n o : ByteArray} (h : S.stop = none)
    (he : S.sendHello s n = .error (S', o)) : S'.stop.isSome := by
  revert he
  unfold State.sendHello
  simp only [h]
  rcases s with e | ⟨spec, sts⟩
  · intro he; cases he; rfl
  · dsimp only
    split
    · intro he; cases he; rfl
    · split
      · intro he; cases he; rfl
      · intro he; cases he

theorem sendHello_ok {S S' : State} {s : V (SessionSpec × Array Stmt)} {n o : ByteArray}
    (he : S.sendHello s n = .ok (S', o)) :
    ∃ spec sts, S'.stop = S.stop ∧ S'.sent = true ∧ S'.hello = helloOf spec sts n ∧ S'.plan = planOf spec sts ∧
      names S' = spec.repStreams.toList.map (·.2.2) ∧ S'.labels.size = spec.repStreams.size := by
  revert he
  unfold State.sendHello
  split
  · intro he; cases he
  · intro he; cases he
  · rename_i spec sts _
    split
    · intro he; cases he
    · split
      · intro he; cases he
      · intro he
        cases he
        refine ⟨spec, sts, rfl, rfl, rfl, rfl, ?_, by simp⟩
        simp only [names, Array.toList_map, List.map_map]
        apply List.map_congr_left
        intro x _
        rfl

theorem helloOf_some {spec : SessionSpec} {sts : Array Stmt} {n : ByteArray} {hh : Coins.Hello}
    (h : helloOf spec sts n = some hh) :
    (spec.repStreams.toList.map (·.2.2)).Nodup ∧ spec.repStreams.size = 2 * sts.size := by
  revert h
  unfold helloOf
  split
  · rename_i ct _
    split
    · rename_i hok
      intro _
      simp only [streamsOk, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hok
      obtain ⟨⟨hl, hn⟩, hs⟩ := hok
      rw [hl] at hn
      exact ⟨(List.nodup_cons.mp hn).2, hs⟩
    · intro h; cases h
  · intro h; cases h

theorem slot_kind {S : State} {P : Placed} {k : Kind} (h : slot S P = some k) :
    k ≠ .ping ∧ k ≠ .proof ∧ k ≠ .other ∧ k ≠ .hello ∨ k = .hello := by
  unfold slot at h
  repeat' split at h
  all_goals first
    | (cases h; simp)
    | (obtain ⟨_, _, rfl⟩ := Option.map_eq_some_iff.mp h; simp)
    | simp at h

theorem any_of_idxOf_lt {S : State} {n : Nat} {s : String} (h : (names S).idxOf s < n) (hn : n ≤ S.labels.size) :
    (S.labels.extract 0 n).any (·.1 == s) = true := by
  have hlen : (names S).length = S.labels.size := by simp [names]
  have hmem : (names S).idxOf s < (names S).length := by omega
  have hget := List.getElem_idxOf hmem
  generalize (names S).idxOf s = i at h hmem hget
  refine Array.any_eq_true.mpr ⟨i, by rw [Array.size_extract]; omega, ?_⟩
  simp only [names, List.getElem_map, Array.getElem_toList] at hget
  simpa using hget

theorem reach_draw {S : State} {P : Placed} (hR : Reach S P) (d : Option Lean.Json) : Reach { S with draw := d } P :=
  { reg := hR.reg, unsent := hR.unsent, sess := hR.sess, grouped := hR.grouped, size := hR.size,
    opens_le := hR.opens_le, opens_pos := hR.opens_pos, link_pos := hR.link_pos, link_two := hR.link_two,
    link_opens := hR.link_opens, le_plan := hR.le_plan, idx := hR.idx }

theorem reach_register {S : State} {P : Placed} (hR : Reach S P) : Reach S { P with register := true } :=
  { reg := fun _ => rfl, unsent := hR.unsent, sess := hR.sess, grouped := hR.grouped, size := hR.size,
    opens_le := hR.opens_le, opens_pos := hR.opens_pos, link_pos := hR.link_pos, link_two := hR.link_two,
    link_opens := hR.link_opens, le_plan := hR.le_plan, idx := hR.idx }

theorem reach_link {S : State} {P : Placed} (hR : Reach S P) (hj : opened S ≠ 0) : Reach S { P with link := true } :=
  { reg := hR.reg, unsent := hR.unsent, sess := hR.sess, grouped := hR.grouped, size := hR.size,
    opens_le := hR.opens_le, opens_pos := hR.opens_pos, link_pos := fun _ => by omega, link_two := fun _ => rfl,
    link_opens := fun _ => rfl, le_plan := hR.le_plan, idx := hR.idx }

theorem reach_open {S : State} {P : Placed} {s : String} (hR : Reach S P) (hs : slot S P = some (.openStream s)) :
    Reach S { P with opens := P.opens + 1 } := by
  obtain ⟨hl, _, hor⟩ := slot_open hs
  refine
    { reg := hR.reg
      unsent := hR.unsent
      sess := hR.sess
      grouped := hR.grouped
      size := hR.size
      opens_le := hl
      opens_pos := fun _ => Nat.le_add_left _ _
      link_pos := hR.link_pos
      link_two := hR.link_two
      link_opens := fun h2 => ?_
      le_plan := hR.le_plan
      idx := fun i x hi hx => ?_ }
  · rcases hor with h0 | ⟨hlink, _⟩
    · simp only at h2; omega
    · exact hlink
  · simp only [Nat.add_sub_cancel]
    rcases hor with h0 | ⟨_, hany⟩
    · rw [h0]; exact Nat.zero_le _
    · have := hany i x hi hx
      apply Nat.le_of_not_lt
      intro hlt
      rw [any_of_idxOf_lt hlt (Nat.le_of_lt hl)] at this
      cases this

/-- **A coin's answer that the session counts keeps `Reach`.** -/
theorem reach_coin {S : State} {P : Placed} {C' : Coins.Release} (hR : Reach S P) (hsent : S.sent = true)
    (hC' : C'.opened.length = opened S + 1) (hle : opened S ≤ S.plan.size) (hlink : 1 ≤ opened S → P.link = true)
    (hopens : 1 ≤ P.opens) (hidx : ∀ x, S.plan[opened S]? = some x → P.opens - 1 ≤ (names S).idxOf x.1) :
    Reach { S with session := some C' } P := by
  have ho : opened { S with session := some C' } = opened S + 1 := by simp [opened, hC']
  refine
    { reg := hR.reg
      unsent := fun h => ?_
      sess := fun _ => rfl
      grouped := hR.grouped
      size := hR.size
      opens_le := hR.opens_le
      opens_pos := fun _ => hopens
      link_pos := fun _ => ?_
      link_two := fun h2 => ?_
      link_opens := hR.link_opens
      le_plan := ?_
      idx := fun i x hi hx => ?_ }
  · simp only at h; rw [hsent] at h; cases h
  · rw [ho]; omega
  · rw [ho] at h2; exact hlink (by omega)
  · rw [ho]; simp only; omega
  · rw [ho] at hi
    have : i = opened S := by omega
    subst this
    exact hidx x hx

theorem reach_hello {S S' : State} {P : Placed} {C : Coins.Release} {spec : SessionSpec} {sts : Array Stmt}
    (hR : Reach S P) (hs : slot S P = some .hello) (hsent : S'.sent = true) (hplan : S'.plan = planOf spec sts)
    (hnames : names S' = spec.repStreams.toList.map (·.2.2)) (hsize : S'.labels.size = spec.repStreams.size)
    (hn : (spec.repStreams.toList.map (·.2.2)).Nodup) (hsz : spec.repStreams.size = 2 * sts.size)
    (hC : C.opened = []) : Reach { S' with session := some C } P := by
  obtain ⟨hreg, hunsent⟩ := slot_hello hs
  obtain ⟨hl0, hs0⟩ := hR.unsent hunsent
  have hop : P.opens = 0 := by have := hR.opens_le; rw [hl0] at this; simpa using this
  have hlink : P.link = false := by
    cases hP : P.link
    · rfl
    · have := hR.link_pos hP; simp [opened, hs0] at this
  have ho : opened { S' with session := some C } = 0 := by simp [opened, hC]
  have hnm : names { S' with session := some C } = names S' := rfl
  refine
    { reg := fun _ => hreg
      unsent := fun h => ?_
      sess := fun _ => rfl
      grouped := ?_
      size := ?_
      opens_le := ?_
      opens_pos := fun h1 => ?_
      link_pos := fun h1 => ?_
      link_two := fun h1 => ?_
      link_opens := fun h1 => ?_
      le_plan := ?_
      idx := fun i x hi _ => ?_ }
  · simp only at h; rw [hsent] at h; cases h
  · rw [hnm, hnames]; simp only; rw [hplan]; exact grouped_planOf hn
  · simp only; rw [hsize]; omega
  · simp only; rw [hop]; exact Nat.zero_le _
  · rw [ho] at h1; cases h1
  · rw [hlink] at h1; cases h1
  · rw [ho] at h1; cases h1
  · rw [hop] at h1; cases h1
  · rw [ho]; exact Nat.zero_le _
  · rw [ho] at hi; cases hi

theorem stepK_hello {hold : Bool} {st : State} {pl : Placed} {e : Env} (hr : st.request pl .hello = none) :
    stepK hold st pl (some (tagOf .hello)) (.ok .hello) e = match st.sendHello e.setup e.nonce with
      | .error (st', _) => ⟨[], st', pl, true⟩
      | .ok (st', h) => match e.ans with
        | none => ⟨[h], st', pl, false⟩
        | some r => ⟨[h], (st'.answer .hello r e.id).1, pl, true⟩ := by
  simp only [stepK, tagOf, hr]; rfl

theorem stepK_finish {hold : Bool} {st : State} {pl : Placed} {e : Env} :
    stepK hold st pl (some (tagOf .finish)) (.ok .finish) e = ⟨finishOut hold st pl e, st, pl, false⟩ := rfl

/-- **Each frame the schedule lets out keeps `Reach`**, or stops the session. -/
theorem reach_step {hold : Bool} {st : State} {pl : Placed} {k : Kind} {e : Env} (hR : Reach st pl)
    (h : st.stop = none) (hs : slot st pl = some k) (hr : st.request pl k = none)
    (hgo : (stepK hold st pl (some (tagOf k)) (.ok k) e).go = true) :
    (stepK hold st pl (some (tagOf k)) (.ok k) e).st.stop.isSome ∨
      Reach (stepK hold st pl (some (tagOf k)) (.ok k) e).st (stepK hold st pl (some (tagOf k)) (.ok k) e).pl := by
  cases k with
  | hello =>
    rw [stepK_hello hr] at hgo ⊢
    revert hgo
    rcases hh : st.sendHello e.setup e.nonce with ⟨st', o⟩ | ⟨st', o⟩
    · intro _; exact .inl (sendHello_error h hh)
    · rcases e.ans with _ | r
      · intro hgo; cases hgo
      · intro _
        obtain ⟨spec, sts, hstop, hsent, hhello, hplan, hnames, hsize⟩ := sendHello_ok hh
        rcases answer_hello (r := r) (id := e.id) (show st'.stop = none by rw [hstop, h]) with
          hst | ⟨hh', C, hhel, heq, hC⟩
        · exact .inl hst
        · right
          dsimp only
          rw [heq]
          rw [hhello] at hhel
          obtain ⟨hn, hsz⟩ := helloOf_some hhel
          exact reach_hello hR hs hsent hplan hnames hsize hn hsz hC
  | finish => rw [stepK_finish] at hgo; cases hgo
  | ping => rcases slot_kind hs with ⟨h1, -⟩ | h1 <;> first | exact absurd rfl h1 | cases h1
  | proof => rcases slot_kind hs with ⟨-, h1, -⟩ | h1 <;> first | exact absurd rfl h1 | cases h1
  | other => rcases slot_kind hs with ⟨-, -, h1, -⟩ | h1 <;> first | exact absurd rfl h1 | cases h1
  | register =>
    rcases stepK_frame_go (by simp) (by simp) (by simp) (by simp) hr hgo with hst | ⟨r, hst, hpl⟩
    · exact .inl hst
    · rw [hst, hpl]
      simp only [afterAnswer, Placed.after]
      rcases answer_register st r with h1 | ⟨d, h1⟩
      · exact .inl h1
      · rw [h1]; exact .inr (reach_draw (reach_register hR) d)
  | link =>
    rcases stepK_frame_go (by simp) (by simp) (by simp) (by simp) hr hgo with hst | ⟨r, hst, hpl⟩
    · exact .inl hst
    · rw [hst, hpl]
      simp only [afterAnswer, Placed.after, answer_link]
      exact .inr (reach_link hR (slot_link hs).1)
  | openStream s =>
    rcases stepK_frame_go (by simp) (by simp) (by simp) (by simp) hr hgo with hst | ⟨r, hst, hpl⟩
    · exact .inl hst
    · rw [hst, hpl]
      simp only [afterAnswer, Placed.after, answer_open]
      exact .inr (reach_open hR hs)
  | commit =>
    rcases stepK_frame_go (by simp) (by simp) (by simp) (by simp) hr hgo with hst | ⟨r, hst, hpl⟩
    · exact .inl hst
    · rw [hst, hpl]
      simp only [afterAnswer, Placed.after, answer_commit]
      obtain ⟨hsent, hop, hj⟩ := slot_commit hs
      split
      · obtain ⟨C, hC⟩ := Option.isSome_iff_exists.mp (hR.sess hsent)
        rcases coins_cases (r := r) h hC with h1 | ⟨C', h1, hlen, -⟩
        · exact .inl h1
        · rw [h1]
          right
          have hj' : C.opened.length = 0 := by rw [← opened_of hC]; exact hj
          refine reach_coin hR hsent (by rw [hlen, opened_of hC]) (by rw [hj]; exact Nat.zero_le _)
            (fun h1 => by omega) (by omega) (fun x _ => ?_)
          have : pl.opens ≤ 1 := by
            apply Nat.le_of_not_lt
            intro h2
            have := hR.link_pos (hR.link_opens h2)
            omega
          omega
      · exact .inl (noCoins_stop _ _ _)
  | round s n =>
    rcases stepK_frame_go (by simp) (by simp) (by simp) (by simp) hr hgo with hst | ⟨r, hst, hpl⟩
    · exact .inl hst
    · rw [hst, hpl]
      simp only [afterAnswer, Placed.after, answer_round]
      obtain ⟨hsent, hj, hlink, hq⟩ := slot_round hs
      split
      · rename_i h2
        obtain ⟨C, hC⟩ := Option.isSome_iff_exists.mp (hR.sess hsent)
        rcases coins_cases (r := r) h hC with h1 | ⟨C', h1, hlen, -⟩
        · split
          · split
            · exact .inl h1
            · exact .inl rfl
          · exact .inl h1
        · rw [h1]
          have hreach : Reach { st with session := some C' } pl := by
            refine reach_coin hR hsent (by rw [hlen, opened_of hC]) ?_ (fun _ => hlink)
              (hR.opens_pos (by omega)) (fun x hx => ?_)
            · have := (Array.getElem?_eq_some_iff.mp hq).1
              omega
            · have h3 := hR.idx (opened st - 1) (s, n) (by omega) hq
              have h4 := hR.grouped (opened st - 1) (opened st) (s, n) x (by omega) hq hx
              exact Nat.le_trans h3 h4
          simp only [h2, beq_self_eq_true, Bool.true_and]
          split
          · split
            · exact .inr hreach
            · exact .inl rfl
          · exact .inr hreach
      · rename_i h2
        simp only [show (byte? r 0 == some 2) = false by simp [h2], Bool.false_and, Bool.false_eq_true, ↓reduceIte]
        exact .inl (noCoins_stop _ _ _)

/-- A request's first byte is its kind's tag. -/
theorem byte_kind {f : ByteArray} {k : Kind} (h : kind f = .ok k) (hk : k ≠ .other) : byte? f 0 = some (tagOf k) := by
  revert h
  unfold kind
  split <;> rename_i hb
  all_goals first
    | (intro h; cases h; exact hb)
    | (intro h; cases h; exact absurd rfl hk)
    | (intro h; cases h; done)
    | skip
  · split
    · intro h; cases h; exact hb
    · intro h; cases h
  · split
    · intro h; cases h
    · split
      · split
        · intro h; cases h; exact hb
        · intro h; cases h
      · intro h; cases h
      · intro h; cases h

theorem stepK_error {hold : Bool} {st : State} {pl : Placed} {b : Option UInt8} {why : String} {e : Env}
    (h9 : b ≠ some 9) (h5 : b ≠ some 5) (h6 : b ≠ some 6) :
    stepK hold st pl b (.error why) e = ⟨[], st, pl, false⟩ := by
  simp [stepK, h9, h5, h6]

theorem stepK_refused {hold : Bool} {st st' : State} {pl : Placed} {b : Option UInt8} {k : Kind} {e : Env}
    {o : ByteArray} (h9 : b ≠ some 9) (h5 : b ≠ some 5) (h6 : b ≠ some 6) (hr : st.request pl k = some (st', o)) :
    stepK hold st pl b (.ok k) e = ⟨[], st', pl, true⟩ := by
  cases k <;> simp [stepK, h9, h5, h6, hr]

/-- **The requests that reach a slot**: of `ins`, at each what the relay read, those whose request leaves at the
schedule's place it reaches, in order, up to the session's stop or end (`out_sched`). A `Ping` or a `Proof` reaches
none. -/
def pick (hold : Bool) : State → Placed → List (ByteArray × Env) → List Env
  | _, _, [] => []
  | st, pl, (f, e) :: rest =>
    if st.stop.isSome then []
    else if byte? f 0 = some 9 ∨ byte? f 0 = some 5 then pick hold st pl rest
    else if byte? f 0 = some 6 then (if (st.request pl .finish).isNone then [e] else [])
    else match kind f with
      | .error _ => []
      | .ok k => match st.request pl k with
        | some _ => []
        | none =>
          if (stepK hold st pl (some (tagOf k)) (.ok k) e).go then
            e :: pick hold (stepK hold st pl (some (tagOf k)) (.ok k) e).st
              (stepK hold st pl (some (tagOf k)) (.ok k) e).pl rest
          else [e]

/-- **The relay's release is the schedule's**: at every requests the worker sends, what leaves for the auditor is what
leaves at the schedule's places, each request that reaches one (`pick`) at the one the session allows next, a sublist of
what the relay read. -/
theorem out_sched (hold : Bool) : ∀ (st : State) (pl : Placed) (ins : List (ByteArray × Env)),
    (st.stop.isSome ∨ (st.stop = none ∧ Reach st pl)) → (∀ p ∈ ins, ∀ why, (p.2.barrier (some why)).isSome) →
    (pick hold st pl ins).Sublist (ins.map (·.2)) ∧ out hold st pl ins = schedOut hold st pl (pick hold st pl ins)
  | st, pl, [], _, _ => ⟨List.Sublist.slnil, rfl⟩
  | st, pl, (f, e) :: rest, hst, hb => by
    have hbr : ∀ p ∈ rest, ∀ why, (p.2.barrier (some why)).isSome := fun p hp => hb p (List.mem_cons_of_mem _ hp)
    have he : ∀ why, (e.barrier (some why)).isSome := hb (f, e) (List.mem_cons_self ..)
    rcases hst with hst | ⟨hst, hR⟩
    · have hp : pick hold st pl ((f, e) :: rest) = [] := by simp only [pick, hst, ite_true]
      rw [hp]
      exact ⟨List.nil_sublist _, by rw [out_stopped hold st pl _ hst hb]; rfl⟩
    have hst' : st.stop.isSome = false := by rw [hst]; rfl
    have skip : stepK hold st pl (byte? f 0) (kind f) e = ⟨[], st, pl, true⟩ →
        pick hold st pl ((f, e) :: rest) = pick hold st pl rest →
        (pick hold st pl ((f, e) :: rest)).Sublist (((f, e) :: rest).map (·.2)) ∧
          out hold st pl ((f, e) :: rest) = schedOut hold st pl (pick hold st pl ((f, e) :: rest)) := by
      intro hstep hp
      obtain ⟨hsub, heq⟩ := out_sched hold st pl rest (.inr ⟨hst, hR⟩) hbr
      rw [hp]
      refine ⟨hsub.cons _, ?_⟩
      simp only [out, hstep, List.nil_append, ite_true]
      exact heq
    have stopped : ∀ st', st'.stop.isSome → stepK hold st pl (byte? f 0) (kind f) e = ⟨[], st', pl, true⟩ →
        pick hold st pl ((f, e) :: rest) = [] →
        (pick hold st pl ((f, e) :: rest)).Sublist (((f, e) :: rest).map (·.2)) ∧
          out hold st pl ((f, e) :: rest) = schedOut hold st pl (pick hold st pl ((f, e) :: rest)) := by
      intro st' hs' hstep hp
      rw [hp]
      refine ⟨List.nil_sublist _, ?_⟩
      simp only [out, hstep, List.nil_append, ite_true]
      exact out_stopped hold st' pl rest hs' hbr
    by_cases h9 : byte? f 0 = some 9
    · exact skip (by unfold stepK; rw [h9]; rfl) (by simp only [pick, hst', h9, true_or, ite_true]; rfl)
    by_cases h5 : byte? f 0 = some 5
    · exact skip (by unfold stepK; rw [h5, hst]; rfl) (by simp only [pick, hst', h5, or_true, ite_true]; rfl)
    have h95 : ¬(byte? f 0 = some 9 ∨ byte? f 0 = some 5) := fun h => h.elim h9 h5
    by_cases h6 : byte? f 0 = some 6
    · have hstep : stepK hold st pl (byte? f 0) (kind f) e = ⟨finishOut hold st pl e, st, pl, false⟩ := by
        unfold stepK; rw [h6]; rfl
      rcases hfin : st.request pl .finish with _ | ⟨S', o⟩
      · have hp : pick hold st pl ((f, e) :: rest) = [e] := by
          simp only [pick, hst', h6, hfin, Bool.false_eq_true, ite_false, ite_true, Option.isNone_none]
          exact ite_eq_right_of_eq_false _ _ (eq_false (by decide))
        rw [hp]
        have hslot := request_slot hR hst (by simp) (by simp) hfin
        refine ⟨(List.nil_sublist _).cons_cons _, ?_⟩
        simp only [out, hstep, schedOut, hslot, stepK_finish, Bool.false_eq_true, ite_false]
      · have hp : pick hold st pl ((f, e) :: rest) = [] := by
          simp only [pick, hst', h6, hfin, Bool.false_eq_true, ite_false, ite_true, Option.isNone_some]
          exact ite_eq_right_of_eq_false _ _ (eq_false (by decide))
        rw [hp]
        refine ⟨List.nil_sublist _, ?_⟩
        simp only [out, hstep, finishOut_of_some he hfin (request_stop hst hfin), Bool.false_eq_true, ite_false]
        rfl
    rcases hk : kind f with why | k
    · have hp : pick hold st pl ((f, e) :: rest) = [] := by
        simp only [pick, hst', h95, h6, hk, Bool.false_eq_true, ite_false]
      rw [hp]
      refine ⟨List.nil_sublist _, ?_⟩
      simp only [out, hk, stepK_error h9 h5 h6, Bool.false_eq_true, ite_false]
      rfl
    rcases hreq : st.request pl k with _ | ⟨st', o⟩
    · have hkp : k ≠ .ping := by rintro rfl; exact h9 (byte_kind hk (by simp))
      have hkq : k ≠ .proof := by rintro rfl; exact h5 (byte_kind hk (by simp))
      have hslot := request_slot hR hst hkp hkq hreq
      have hko : k ≠ .other := by
        rcases slot_kind hslot with ⟨-, -, h1, -⟩ | rfl
        · exact h1
        · simp
      have hbk := byte_kind hk hko
      have hp0 : pick hold st pl ((f, e) :: rest) =
          if (stepK hold st pl (some (tagOf k)) (.ok k) e).go then
            e :: pick hold (stepK hold st pl (some (tagOf k)) (.ok k) e).st
              (stepK hold st pl (some (tagOf k)) (.ok k) e).pl rest
          else [e] := by
        simp only [pick, hst', h95, h6, hk, hreq, Bool.false_eq_true, ite_false]
      rcases hr : stepK hold st pl (some (tagOf k)) (.ok k) e with ⟨sent, st₁, pl₁, go⟩
      rw [hr] at hp0
      cases go
      · simp only [Bool.false_eq_true, ite_false] at hp0
        rw [hp0]
        refine ⟨(List.nil_sublist _).cons_cons _, ?_⟩
        simp only [out, hk, hbk, hr, schedOut, hslot, Bool.false_eq_true, ite_false]
      · simp only [ite_true] at hp0
        rw [hp0]
        have hgo : (stepK hold st pl (some (tagOf k)) (.ok k) e).go = true := by rw [hr]
        have hnext := reach_step hR hst hslot hreq hgo
        rw [hr] at hnext
        have hnext' : st₁.stop.isSome ∨ (st₁.stop = none ∧ Reach st₁ pl₁) := by
          rcases hnext with h1 | h1
          · exact .inl h1
          · cases h2 : st₁.stop
            · exact .inr ⟨rfl, h1⟩
            · exact .inl rfl
        obtain ⟨hsub, heq⟩ := out_sched hold st₁ pl₁ rest hnext' hbr
        refine ⟨hsub.cons_cons _, ?_⟩
        simp only [out, hk, hbk, hr, schedOut, hslot, ite_true]
        rw [heq]
    · have hp : pick hold st pl ((f, e) :: rest) = [] := by
        simp only [pick, hst', h95, h6, hk, hreq, Bool.false_eq_true, ite_false]
      exact stopped st' (request_stop hst hreq) (by rw [hk]; exact stepK_refused h9 h5 h6 hreq) hp

/-- **Where the session is in its schedule after the `Hello`**: how many rep streams opened, how many openings the
auditor answered, and whether the `Link` left. -/
structure Pos where
  opens : Nat
  opened : Nat
  link : Bool

/-- The position of a session and its placement. -/
def posOf (S : State) (P : Placed) : Pos := ⟨P.opens, opened S, P.link⟩

/-- **The slot at a position**, given the rep streams and the plan: `slot` after the `Hello`. -/
def slotAt (labels : Array (String × String)) (plan : Array (String × Nat)) (p : Pos) : Option Kind :=
  if p.opens = 0 then labels[0]?.map fun l => .openStream l.1
  else if p.opened = 0 then some .commit
  else if !p.link then some .link
  else match plan[p.opened - 1]? with
    | some (s, n) =>
      if (labels.extract 0 p.opens).any (·.1 == s) then some (.round s n)
      else labels[p.opens]?.map fun l => .openStream l.1
    | none => match labels[p.opens]? with
      | some l => some (.openStream l.1)
      | none => if p.opened = plan.size + 1 then some .finish else none

/-- The position once a request of kind `k` left at it and, for a coin request, its coins were answered. -/
def Pos.adv (p : Pos) : Kind → Pos
  | .openStream _ => { p with opens := p.opens + 1 }
  | .commit | .round _ _ => { p with opened := p.opened + 1 }
  | .link => { p with link := true }
  | _ => p

/-- **The schedule after the `Hello`**, from the position `p`, for at most `fuel` slots: each slot's kind in turn, up to
the `Finish`. -/
def tmplFrom (labels : Array (String × String)) (plan : Array (String × Nat)) : Nat → Pos → List Kind
  | 0, _ => []
  | f + 1, p => match slotAt labels plan p with
    | none => []
    | some .finish => [.finish]
    | some k => k :: tmplFrom labels plan f (p.adv k)

/-- **The session's schedule**: `Register`, `Hello`, then the slots after it, at the rep streams and plan the `Hello`
set. -/
def tmpl (labels : Array (String × String)) (plan : Array (String × Nat)) : List Kind :=
  .register :: .hello :: tmplFrom labels plan (labels.size + plan.size + 4) ⟨0, 0, false⟩

theorem slot_sent {S : State} {P : Placed} (hreg : P.register = true) (hsent : S.sent = true) :
    slot S P = slotAt S.labels S.plan (posOf S P) := by
  simp only [slot, hreg, hsent, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
  rfl

/-- **What leaves at each slot**, in order: the slot's kind and what left for it, up to the session's stop or end. -/
def schedTrace (hold : Bool) : State → Placed → List Env → List (Kind × List ByteArray)
  | _, _, [] => []
  | st, pl, e :: rest => match slot st pl with
    | none => []
    | some k =>
      let r := stepK hold st pl (some (tagOf k)) (.ok k) e
      (k, r.sent) :: if r.go && r.st.stop.isNone then schedTrace hold r.st r.pl rest else []

theorem schedOut_stopped (hold : Bool) : ∀ (st : State) (pl : Placed) (es : List Env),
    st.stop.isSome → (∀ e ∈ es, ∀ why, (e.barrier (some why)).isSome) → schedOut hold st pl es = []
  | _, _, [], _, _ => rfl
  | st, pl, e :: rest, hs, hb => by
    unfold schedOut
    split
    · rfl
    · rename_i k _
      obtain ⟨h1, h2⟩ := stepK_stopped (hold := hold) (pl := pl) (b := some (tagOf k)) (k := .ok k) (e := e) hs
        (hb e (List.mem_cons_self ..))
      simp only [h1, List.nil_append]
      split
      · exact schedOut_stopped hold _ _ rest h2 fun e he => hb e (List.mem_cons_of_mem _ he)
      · rfl

/-- **What leaves at the schedule's places is what leaves at each slot, in order.** -/
theorem schedOut_trace (hold : Bool) : ∀ (st : State) (pl : Placed) (es : List Env),
    (∀ e ∈ es, ∀ why, (e.barrier (some why)).isSome) →
    schedOut hold st pl es = (schedTrace hold st pl es).flatMap (·.2)
  | _, _, [], _ => rfl
  | st, pl, e :: rest, hb => by
    have hbr : ∀ e ∈ rest, ∀ why, (e.barrier (some why)).isSome := fun e he => hb e (List.mem_cons_of_mem _ he)
    rcases hsl : slot st pl with _ | k
    · simp only [schedOut, schedTrace, hsl, List.flatMap_nil]
    · simp only [schedOut, schedTrace, hsl, List.flatMap_cons]
      congr 1
      cases hgo : (stepK hold st pl (some (tagOf k)) (.ok k) e).go
      · simp
      · cases hs : (stepK hold st pl (some (tagOf k)) (.ok k) e).st.stop
        · simp only [ite_true, Option.isNone_none, Bool.and_self]
          exact schedOut_trace hold _ _ rest hbr
        · simp only [ite_true, Option.isNone_some, Bool.and_false, Bool.false_eq_true, ite_false, List.flatMap_nil]
          exact schedOut_stopped hold _ _ rest (by rw [hs]; rfl) hbr

/-- **The slots left** at a position: the rep streams not yet open, the openings not yet answered, the `Link` if it
hasn't left, and the `Finish`. -/
def Pos.left (labels : Array (String × String)) (plan : Array (String × Nat)) (p : Pos) : Nat :=
  (labels.size - p.opens) + (plan.size + 1 - p.opened) + (if p.link then 0 else 1) + 1

/-- **What a request after the `Hello` leaves of the session**: its rep streams, plan, sizes and Σ. -/
structure Fixed (S S' : State) : Prop where
  labels : S'.labels = S.labels
  plan : S'.plan = S.plan
  sent : S'.sent = S.sent
  points : S'.points = S.points
  tables : S'.tables = S.tables
  sigma : S'.sigma = S.sigma

theorem Fixed.refl (S : State) : Fixed S S := ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem Fixed.session (S : State) (C : Coins.Release) : Fixed S { S with session := some C } :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem Fixed.trans {S S' S'' : State} (h : Fixed S S') (h' : Fixed S' S'') : Fixed S S'' :=
  ⟨h'.labels.trans h.labels, h'.plan.trans h.plan, h'.sent.trans h.sent, h'.points.trans h.points,
    h'.tables.trans h.tables, h'.sigma.trans h.sigma⟩

theorem posOf_session {S : State} {P : Placed} {C C' : Coins.Release} (hs : S.session = some C)
    (hlen : C'.opened.length = C.opened.length + 1) :
    posOf { S with session := some C' } P = { posOf S P with opened := opened S + 1 } := by
  simp only [posOf, opened, hs, Option.map_some, Option.getD_some, hlen]

/-- **A request after the `Hello` that leaves and keeps the session going** advances its position by its kind: a coin
request the auditor answered without coins stopped the session (`State.noCoins`). -/
theorem pos_step {hold : Bool} {st : State} {pl : Placed} {k : Kind} {e : Env} (hR : Reach st pl)
    (h : st.stop = none) (hs : slot st pl = some k) (hk1 : k ≠ .hello) (hk2 : k ≠ .register)
    (hr : st.request pl k = none) (hgo : (stepK hold st pl (some (tagOf k)) (.ok k) e).go = true)
    (hst : (stepK hold st pl (some (tagOf k)) (.ok k) e).st.stop = none) :
    Fixed st (stepK hold st pl (some (tagOf k)) (.ok k) e).st ∧
      (stepK hold st pl (some (tagOf k)) (.ok k) e).pl.register = pl.register ∧
      posOf (stepK hold st pl (some (tagOf k)) (.ok k) e).st (stepK hold st pl (some (tagOf k)) (.ok k) e).pl =
        (posOf st pl).adv k := by
  have hfin : k ≠ .finish := by rintro rfl; rw [stepK_finish] at hgo; cases hgo
  rcases slot_kind hs with ⟨hp, hq, -, -⟩ | rfl
  · rcases stepK_frame_ans hk1 hfin hp hq hr hgo with hs' | ⟨r, -, hst', hpl⟩
    · rw [hst] at hs'; cases hs'
    rw [hst'] at hst ⊢
    rw [hpl]
    cases k with
    | hello => exact absurd rfl hk1
    | register => exact absurd rfl hk2
    | finish => exact absurd rfl hfin
    | ping => exact absurd rfl hp
    | proof => exact absurd rfl hq
    | other => rcases slot_kind hs with ⟨-, -, h3, -⟩ | h3 <;> first | exact absurd rfl h3 | cases h3
    | link =>
      simp only [afterAnswer, Placed.after, answer_link]
      exact ⟨Fixed.refl _, by first | rfl | trivial, rfl⟩
    | openStream s =>
      simp only [afterAnswer, Placed.after, answer_open]
      exact ⟨Fixed.refl _, by first | rfl | trivial, rfl⟩
    | commit =>
      simp only [afterAnswer, Placed.after, answer_commit] at hst ⊢
      obtain ⟨hsent, -, -⟩ := slot_commit hs
      by_cases h2 : byte? r 0 = some 2
      · obtain ⟨C, hC⟩ := Option.isSome_iff_exists.mp (hR.sess hsent)
        simp only [h2, ite_true] at hst ⊢
        rcases coins_cases (r := r) h hC with h1 | ⟨C', h1, hlen, -⟩
        · rw [hst] at h1; cases h1
        · rw [h1]
          refine ⟨Fixed.session _ _, by first | rfl | trivial, ?_⟩
          rw [posOf_session hC hlen]
          rfl
      · simp only [h2, ite_false] at hst
        exact absurd hst (Option.isSome_iff_ne_none.mp (noCoins_stop _ _ _))
    | round s n =>
      simp only [afterAnswer, Placed.after, answer_round] at hst ⊢
      obtain ⟨hsent, -, -, -⟩ := slot_round hs
      by_cases h2 : byte? r 0 = some 2
      · obtain ⟨C, hC⟩ := Option.isSome_iff_exists.mp (hR.sess hsent)
        simp only [h2, ite_true] at hst ⊢
        rcases coins_cases (r := r) h hC with h1 | ⟨C', h1, hlen, -⟩
        · obtain ⟨w, hw⟩ := Option.isSome_iff_exists.mp h1
          simp only [hw, Option.isNone_some, Bool.and_false, Bool.false_eq_true, ite_false] at hst
          cases hst
        · have hn : ({ st with session := some C' } : State).stop.isNone = true := by simp [h]
          rw [h1] at hst ⊢
          simp only [hn, beq_self_eq_true, Bool.and_self, ite_true] at hst ⊢
          rcases hc : e.coins with why | ⟨⟩
          · simp only [hc, State.refuse] at hst; cases hst
          · simp only
            refine ⟨Fixed.session _ _, by first | rfl | trivial, ?_⟩
            rw [posOf_session hC hlen]
            rfl
      · simp only [h2, ite_false, show (byte? r 0 == some 2) = false by simp [h2], Bool.false_and,
          Bool.false_eq_true] at hst
        exact absurd hst (Option.isSome_iff_ne_none.mp (noCoins_stop _ _ _))
  · exact absurd rfl hk1

/-- The kinds the schedule has after the `Hello`. -/
def schedKind : Kind → Bool
  | .openStream _ | .commit | .link | .round _ _ | .finish => true
  | _ => false

theorem slotAt_sched {L : Array (String × String)} {Pl : Array (String × Nat)} {p : Pos} {k : Kind}
    (h : slotAt L Pl p = some k) : schedKind k = true := by
  unfold slotAt at h
  split at h
  · rcases hl : L[0]? with _ | l <;> simp [hl] at h
    subst h; rfl
  · split at h
    · cases h; rfl
    · split at h
      · cases h; rfl
      · split at h
        · split at h
          · cases h; rfl
          · rcases hl : L[p.opens]? with _ | l <;> simp [hl] at h
            subst h; rfl
        · split at h
          · cases h; rfl
          · split at h
            · cases h; rfl
            · cases h

theorem lt_of_getElem? {α : Type} {a : Array α} {i : Nat} {x : α} (h : a[i]? = some x) : i < a.size := by
  rcases Nat.lt_or_ge i a.size with hi | hi
  · exact hi
  · simp [Array.getElem?_eq_none hi] at h

/-- **Each slot but the `Finish` takes one of the slots left.** -/
theorem left_adv {L : Array (String × String)} {Pl : Array (String × Nat)} {p : Pos} {k : Kind}
    (h : slotAt L Pl p = some k) (hk : k ≠ .finish) : Pos.left L Pl (p.adv k) + 1 = Pos.left L Pl p := by
  obtain ⟨o, d, l⟩ := p
  unfold slotAt at h
  simp only at h
  split at h
  · rcases hl : L[0]? with _ | x <;> simp [hl] at h
    subst h
    have := lt_of_getElem? hl
    cases l <;> simp only [Pos.adv, Pos.left, ite_true, Bool.false_eq_true, ite_false] <;> omega
  · split at h
    · cases h
      cases l <;> simp only [Pos.adv, Pos.left, ite_true, Bool.false_eq_true, ite_false] <;> omega
    · split at h
      · rename_i h2
        cases h
        cases l
        · simp only [Pos.adv, Pos.left, ite_true, Bool.false_eq_true, ite_false]
        · simp at h2
      · split at h
        · rename_i s n hp
          have := lt_of_getElem? hp
          split at h
          · cases h
            cases l <;> simp only [Pos.adv, Pos.left, ite_true, Bool.false_eq_true, ite_false] <;> omega
          · rcases hl : L[o]? with _ | x <;> simp [hl] at h
            subst h
            have := lt_of_getElem? hl
            cases l <;> simp only [Pos.adv, Pos.left, ite_true, Bool.false_eq_true, ite_false] <;> omega
        · split at h
          · rename_i x hl
            cases h
            have := lt_of_getElem? hl
            cases l <;> simp only [Pos.adv, Pos.left, ite_true, Bool.false_eq_true, ite_false] <;> omega
          · split at h
            · cases h; exact absurd rfl hk
            · cases h

theorem tmplFrom_succ {L : Array (String × String)} {Pl : Array (String × Nat)} {p : Pos} {k : Kind} (f : Nat)
    (h : slotAt L Pl p = some k) (hk : k ≠ .finish) : tmplFrom L Pl (f + 1) p = k :: tmplFrom L Pl f (p.adv k) := by
  simp only [tmplFrom, h]

theorem tmplFrom_finish {L : Array (String × String)} {Pl : Array (String × Nat)} {p : Pos} (f : Nat)
    (h : slotAt L Pl p = some .finish) : tmplFrom L Pl (f + 1) p = [.finish] := by
  simp only [tmplFrom, h]

theorem tagOf_sched {k : Kind} (h : schedKind k = true) (hk : k ≠ .finish) :
    some (tagOf k) ≠ some 9 ∧ some (tagOf k) ≠ some 5 ∧ some (tagOf k) ≠ some 6 := by
  cases k <;> simp_all [schedKind, tagOf]

/-- A model slot: the `Hello`, a coin request or the `Finish`, whose frames are the session's messages. The others, the
`Register`, the `Link` and each `Open`, are control slots. -/
def isModel : Kind → Bool
  | .hello | .commit | .round _ _ | .finish => true
  | _ => false

/-- **A control slot's frame** at the session `S`: `State.wire`'s. -/
def ctlFrame (S : State) (k : Kind) : ByteArray :=
  match S.wire k with
  | .ok w => w
  | .error _ => .empty

theorem ctlFrame_fixed {S S' : State} (h : Fixed S S') (k : Kind) : ctlFrame S' k = ctlFrame S k := by
  simp only [ctlFrame, State.wire, h.labels, h.points, h.tables]

/-- **What left at each control slot is its frame**, or nothing at the last slot. -/
def Framed (ctl : Kind → ByteArray) : List (Kind × List ByteArray) → Prop
  | [] => True
  | (k, s) :: tl => (isModel k = false → s = [ctl k] ∨ (s = [] ∧ tl = [])) ∧ Framed ctl tl

theorem framed_one {ctl : Kind → ByteArray} {k : Kind} {s : List ByteArray} {P : Prop}
    (h : isModel k = false → s = [ctl k] ∨ (s = [] ∧ P)) : Framed ctl [(k, s)] :=
  ⟨fun hm => (h hm).imp id fun h => ⟨h.1, rfl⟩, trivial⟩

theorem tagOf_ne {k : Kind} (hp : k ≠ .ping) (hq : k ≠ .proof) (hf : k ≠ .finish) :
    some (tagOf k) ≠ some 9 ∧ some (tagOf k) ≠ some 5 ∧ some (tagOf k) ≠ some 6 := by
  cases k <;> first | exact absurd rfl hp | exact absurd rfl hq | exact absurd rfl hf | simp [tagOf]

/-- **At a control slot its frame leaves**, or nothing, and the session stops. -/
theorem sent_ctl {hold : Bool} {st : State} {pl : Placed} {k : Kind} (e : Env) (h : st.stop = none)
    (hs : slot st pl = some k) (hk : isModel k = false) :
    (stepK hold st pl (some (tagOf k)) (.ok k) e).sent = [ctlFrame st k] ∨
      ((stepK hold st pl (some (tagOf k)) (.ok k) e).sent = [] ∧
        (stepK hold st pl (some (tagOf k)) (.ok k) e).st.stop.isSome) := by
  have hk' : k ≠ .ping ∧ k ≠ .proof ∧ k ≠ .other := by
    rcases slot_kind hs with ⟨h1, h2, h3, -⟩ | rfl
    · exact ⟨h1, h2, h3⟩
    · cases hk
  obtain ⟨hp, hq, ho⟩ := hk'
  have hh : k ≠ .hello := by rintro rfl; cases hk
  have hf : k ≠ .finish := by rintro rfl; cases hk
  rcases hr : st.request pl k with _ | ⟨st', o⟩
  · rw [stepK_frame hh hf hp hq hr]
    have hw : frameOf st k e = match st.wire k with
        | .ok w => .ok w
        | .error why => .error s!"{Coins.STOP}{why}" := by
      cases k <;> first | rfl | cases hk | exact absurd rfl hp | exact absurd rfl hq | exact absurd rfl ho
    rw [hw]
    rcases hwk : st.wire k with why | w
    · exact .inr ⟨rfl, rfl⟩
    · have : ctlFrame st k = w := by simp only [ctlFrame, hwk]
      rw [this]
      rcases e.ans with _ | r <;> exact .inl rfl
  · obtain ⟨h9, h5, h6⟩ := tagOf_ne hp hq hf
    rw [stepK_refused h9 h5 h6 hr]
    exact .inr ⟨rfl, request_stop h hr⟩

/-- **What leaves after the `Hello` is at the schedule's slots in order**, cut after one of them, and each control
slot's frame is its own. -/
theorem sched_prefix (hold : Bool) (L : Array (String × String)) (Pl : Array (String × Nat))
    (ctl : Kind → ByteArray) :
    ∀ (es : List Env) (f : Nat) (st : State) (pl : Placed), Reach st pl → st.stop = none → pl.register = true →
      st.sent = true → st.labels = L → st.plan = Pl → Pos.left L Pl (posOf st pl) < f →
      (∀ k, ctlFrame st k = ctl k) →
      (schedTrace hold st pl es).map (·.1) <+: tmplFrom L Pl f (posOf st pl) ∧ Framed ctl (schedTrace hold st pl es)
  | [], _, _, _, _, _, _, _, _, _, _, _ => ⟨List.nil_prefix, trivial⟩
  | e :: rest, f, st, pl, hR, h, hreg, hsent, hL, hPl, hf, hc => by
    have hsl := slot_sent (S := st) (P := pl) hreg hsent
    rw [hL, hPl] at hsl
    obtain ⟨f', rfl⟩ : ∃ f', f = f' + 1 := ⟨f - 1, by omega⟩
    rcases hsa : slotAt L Pl (posOf st pl) with _ | k
    · simp only [schedTrace, hsl, hsa, List.map_nil]
      exact ⟨List.nil_prefix, trivial⟩
    · have hslot : slot st pl = some k := hsl.trans hsa
      have hsc : isModel k = false → (stepK hold st pl (some (tagOf k)) (.ok k) e).sent = [ctl k] ∨
          ((stepK hold st pl (some (tagOf k)) (.ok k) e).sent = [] ∧
            (stepK hold st pl (some (tagOf k)) (.ok k) e).st.stop.isSome) := by
        intro hm
        rw [← hc k]
        exact sent_ctl e h hslot hm
      by_cases hk : k = .finish
      · subst hk
        rw [tmplFrom_finish f' hsa]
        simp only [schedTrace, hslot, stepK_finish, Bool.false_and, Bool.false_eq_true, ite_false,
          List.map_cons, List.map_nil]
        exact ⟨List.prefix_refl _, framed_one (P := True) fun h => by simp [isModel] at h⟩
      · rw [tmplFrom_succ f' hsa hk]
        have hks := slotAt_sched hsa
        simp only [schedTrace, hslot, List.map_cons]
        have last : ∀ T, [k] <+: k :: T := fun _ => List.cons_prefix_cons.mpr ⟨rfl, List.nil_prefix⟩
        have hfr1 : Framed ctl [(k, (stepK hold st pl (some (tagOf k)) (.ok k) e).sent)] := framed_one hsc
        rcases hr : st.request pl k with _ | ⟨st', o⟩
        · cases hgo : (stepK hold st pl (some (tagOf k)) (.ok k) e).go
          · simp only [Bool.false_and, Bool.false_eq_true, ite_false, List.map_nil]
            exact ⟨last _, hfr1⟩
          · cases hst : (stepK hold st pl (some (tagOf k)) (.ok k) e).st.stop
            · simp only [Option.isNone_none, Bool.and_self, ite_true]
              have hk1 : k ≠ .hello := by rintro rfl; cases hks
              have hk2 : k ≠ .register := by rintro rfl; cases hks
              rcases reach_step hR h hslot hr hgo with hs | hR'
              · rw [hst] at hs; cases hs
              obtain ⟨hfx, hreg', hpos⟩ := pos_step hR h hslot hk1 hk2 hr hgo hst
              have hadv := left_adv hsa hk
              have hc' : ∀ k', ctlFrame (stepK hold st pl (some (tagOf k)) (.ok k) e).st k' = ctl k' := by
                intro k'
                rw [ctlFrame_fixed hfx k', hc k']
              have hhead : isModel k = false → (stepK hold st pl (some (tagOf k)) (.ok k) e).sent = [ctl k] ∨
                  ((stepK hold st pl (some (tagOf k)) (.ok k) e).sent = [] ∧
                    schedTrace hold (stepK hold st pl (some (tagOf k)) (.ok k) e).st
                      (stepK hold st pl (some (tagOf k)) (.ok k) e).pl rest = []) := by
                intro hm
                rcases hsc hm with h1 | ⟨-, h2⟩
                · exact .inl h1
                · rw [hst] at h2; cases h2
              have ih := sched_prefix hold L Pl ctl rest f' _ _ hR' hst (hreg'.trans hreg)
                (hfx.sent.trans hsent) (hfx.labels.trans hL) (hfx.plan.trans hPl) (by rw [hpos]; omega) hc'
              rw [hpos] at ih
              exact ⟨List.cons_prefix_cons.mpr ⟨rfl, ih.1⟩, hhead, ih.2⟩
            · simp only [Option.isNone_some, Bool.and_false, Bool.false_eq_true, ite_false, List.map_nil]
              exact ⟨last _, hfr1⟩
        · obtain ⟨h9, h5, h6⟩ := tagOf_sched hks hk
          have hs' := request_stop h hr
          rw [stepK_refused h9 h5 h6 hr]
          obtain ⟨w, hw⟩ := Option.isSome_iff_exists.mp hs'
          simp only [hw, Option.isNone_some, Bool.and_false, Bool.false_eq_true, ite_false, List.map_nil]
          exact ⟨last _, framed_one (P := True) fun _ => .inr ⟨rfl, trivial⟩⟩

/-- The rep streams' names and domains the `Hello` sets at the setup's spec. -/
def labelsOf (spec : SessionSpec) : Array (String × String) :=
  spec.repStreams.map fun (i, r, s) => (s, spec.domain spec.tables[i]! r)

/-- **The session after its `Hello`** at the setup `spec`, `sts` and the nonce `n`. -/
def helloAt (S : State) (spec : SessionSpec) (sts : Array Stmt) (n : ByteArray) : State :=
  { S with sent := true, hello := helloOf spec sts n, plan := planOf spec sts,
           domains := spec.repStreams.map fun (i, r, s) => (s, Frames.domainFrame (spec.domain spec.tables[i]! r)),
           reps := spec.repStreams.map fun (i, r, s) => (s, i, r),
           labels := labelsOf spec, tables := spec.tables.zip (sts.map (·.m)), sigma := spec.sigma,
           points := spec.points }

/-- **The `Hello` that leaves is the setup's**, at the nonce, and so is the session after it. -/
theorem sendHello_eq {S S' : State} {s : V (SessionSpec × Array Stmt)} {n o : ByteArray}
    (he : S.sendHello s n = .ok (S', o)) :
    ∃ spec sts, s = .ok (spec, sts) ∧ S' = helloAt S spec sts n ∧ o = ⟨(Hello.msg spec n).toArray⟩ := by
  revert he
  unfold State.sendHello
  split
  · intro he; cases he
  · intro he; cases he
  · rename_i spec sts _
    split
    · intro he; cases he
    · split
      · intro he; cases he
      · intro he
        cases he
        exact ⟨spec, sts, rfl, rfl, rfl⟩

theorem reach_init : Reach {} {} where
  reg := fun h => by cases h
  unsent := fun _ => ⟨rfl, rfl⟩
  sess := fun h => by cases h
  grouped := fun _ _ _ _ _ h => by simp at h
  size := by decide
  opens_le := Nat.le_refl _
  opens_pos := fun h => by simp [opened] at h
  link_pos := fun h => by cases h
  link_two := fun h => by simp [opened] at h
  link_opens := fun h => by simp at h
  le_plan := by simp [opened]
  idx := fun _ _ h => by simp [opened] at h

/-- **The session's schedule at its setup**: at a setup that fails, the `Register` and the `Hello` it refuses. -/
def tmplOf : V (SessionSpec × Array Stmt) → List Kind
  | .ok (spec, sts) => tmpl (labelsOf spec) (planOf spec sts)
  | .error _ => [.register, .hello]

theorem tmplOf_cons (s : V (SessionSpec × Array Stmt)) : ∃ T, tmplOf s = .register :: .hello :: T := by
  rcases s with _ | ⟨spec, sts⟩
  · exact ⟨[], rfl⟩
  · exact ⟨_, rfl⟩

/-- **The control slots' frames at the setup**: the `Register`'s, and the `Link`'s and each `Open`'s at the session
after its `Hello`. -/
def ctlOf : V (SessionSpec × Array Stmt) → Kind → ByteArray
  | .ok (spec, sts) => ctlFrame (helloAt {} spec sts .empty)
  | .error _ => ctlFrame {}

/-- **What leaves over the relay's session is at the schedule's slots in order**, from its start: the `Register`, the
`Hello` and the slots after it at the setup the relay reads, cut after one of them, and each control slot's frame the
setup's. -/
theorem relay_prefix (hold : Bool) (s : V (SessionSpec × Array Stmt)) (es : List Env)
    (hs : ∀ e ∈ es, e.setup = s) :
    (schedTrace hold {} {} es).map (·.1) <+: tmplOf s ∧ Framed (ctlOf s) (schedTrace hold {} {} es) := by
  obtain ⟨T, hT⟩ := tmplOf_cons s
  rw [hT]
  rcases es with _ | ⟨e₁, rest⟩
  · exact ⟨List.nil_prefix, trivial⟩
  have hs1 : slot ({} : State) ({} : Placed) = some .register := rfl
  have hreg0 : ctlFrame ({} : State) .register = ctlOf s .register := by rcases s with _ | ⟨_, _⟩ <;> rfl
  have hsc1 := sent_ctl (hold := hold) e₁ (st := {}) rfl hs1 rfl
  rw [hreg0] at hsc1
  simp only [schedTrace, hs1, List.map_cons]
  have two : ∀ ks : List Kind, ks <+: [.hello] → .register :: ks <+: .register :: .hello :: T :=
    fun _ h => (List.cons_prefix_cons.mpr ⟨rfl, h⟩).trans (List.prefix_append _ T)
  generalize hr1 : stepK hold {} {} (some (tagOf .register)) (.ok .register) e₁ = r₁ at hsc1
  have fr1 : Framed (ctlOf s) [(.register, r₁.sent)] := framed_one fun _ => hsc1
  cases hgo1 : r₁.go
  · simp only [Bool.false_and, Bool.false_eq_true, ite_false]
    exact ⟨two [] List.nil_prefix, fr1⟩
  cases hst1 : r₁.st.stop
  case some =>
    simp only [Option.isNone_some, Bool.and_false, Bool.false_eq_true, ite_false]
    exact ⟨two [] List.nil_prefix, fr1⟩
  simp only [Option.isNone_none, Bool.and_self, ite_true]
  have hsent1' : r₁.sent = [ctlOf s .register] := by
    rcases hsc1 with h | ⟨-, h⟩
    · exact h
    · rw [hst1] at h; cases h
  rcases hreq1 : ({} : State).request {} .register with _ | ⟨st', o⟩
  case some =>
    have := request_stop (S := {}) rfl hreq1
    rw [stepK_refused (by decide) (by decide) (by decide) hreq1] at hr1
    subst hr1
    obtain ⟨w, hw⟩ := Option.isSome_iff_exists.mp this
    simp [hw] at hst1
  have hgo1' : (stepK hold {} {} (some (tagOf .register)) (.ok .register) e₁).go = true := by rw [hr1]; exact hgo1
  have hR1 : Reach r₁.st r₁.pl := by
    rcases reach_step reach_init rfl hs1 hreq1 hgo1' with h | h
    · rw [hr1, hst1] at h; cases h
    · rw [hr1] at h; exact h
  rcases stepK_frame_ans (k := .register) (by intro h; cases h) (by intro h; cases h)
    (by intro h; cases h) (by intro h; cases h) hreq1 hgo1' with h | ⟨a, -, hst1', hpl1⟩
  · rw [hr1, hst1] at h; cases h
  rw [hr1] at hst1' hpl1
  have hsent1 : r₁.st.sent = false := by
    rw [hst1']
    simp only [afterAnswer]
    rcases answer_register ({} : State) a with h | ⟨d, hd⟩
    · rw [hst1'] at hst1; simp only [afterAnswer] at hst1; rw [hst1] at h; cases h
    · rw [hd]
  have hreg1 : r₁.pl.register = true := by rw [hpl1]; rfl
  rcases rest with _ | ⟨e₂, rest⟩
  · exact ⟨two [] List.nil_prefix, fr1⟩
  have hs2 : slot r₁.st r₁.pl = some .hello := by simp [slot, hreg1, hsent1]
  simp only [schedTrace, hs2, List.map_cons]
  have frh : ∀ (s₂ : List ByteArray) (tl : List (Kind × List ByteArray)), Framed (ctlOf s) tl →
      Framed (ctlOf s) ((.register, r₁.sent) :: (.hello, s₂) :: tl) :=
    fun _ _ h => ⟨fun _ => .inl hsent1', fun hm => by simp [isModel] at hm, h⟩
  have three : [.register, .hello] <+: .register :: .hello :: T := two [.hello] (List.prefix_refl _)
  generalize hr2 : stepK hold r₁.st r₁.pl (some (tagOf .hello)) (.ok .hello) e₂ = r₂
  cases hgo2 : r₂.go
  · simp only [Bool.false_and, Bool.false_eq_true, ite_false]
    exact ⟨three, frh _ _ trivial⟩
  cases hst2 : r₂.st.stop
  case some =>
    simp only [Option.isNone_some, Bool.and_false, Bool.false_eq_true, ite_false]
    exact ⟨three, frh _ _ trivial⟩
  simp only [Option.isNone_none, Bool.and_self, ite_true]
  rcases hreq2 : r₁.st.request r₁.pl .hello with _ | ⟨st', o⟩
  case some =>
    have := request_stop hst1 hreq2
    rw [stepK_refused (by decide) (by decide) (by decide) hreq2] at hr2
    subst hr2
    obtain ⟨w, hw⟩ := Option.isSome_iff_exists.mp this
    simp [hw] at hst2
  have hgo2' : (stepK hold r₁.st r₁.pl (some (tagOf .hello)) (.ok .hello) e₂).go = true := by rw [hr2]; exact hgo2
  have hR2 : Reach r₂.st r₂.pl := by
    rcases reach_step hR1 hst1 hs2 hreq2 hgo2' with h | h
    · rw [hr2, hst2] at h; cases h
    · rw [hr2] at h; exact h
  rw [stepK_hello hreq2] at hr2
  rcases hsh : r₁.st.sendHello e₂.setup e₂.nonce with ⟨S', o⟩ | ⟨S', h⟩
  · rw [hsh] at hr2
    subst hr2
    have := sendHello_error hst1 hsh
    obtain ⟨w, hw⟩ := Option.isSome_iff_exists.mp this
    simp [hw] at hst2
  rw [hsh] at hr2
  obtain ⟨spec, sts, hsp, hS', -⟩ := sendHello_eq hsh
  have hss : s = .ok (spec, sts) := (hs e₂ (by simp)).symm.trans hsp
  subst hss
  rcases hans : e₂.ans with _ | a₂
  · rw [hans] at hr2; subst hr2; cases hgo2
  rw [hans] at hr2
  have hS'stop : S'.stop = none := by rw [hS']; exact hst1
  rcases answer_hello (r := a₂) (id := e₂.id) hS'stop with h | ⟨hh, C, -, hC, hCo⟩
  · subst hr2; rw [hst2] at h; cases h
  subst hr2
  simp only at hC hR2 hst2 ⊢
  rw [hC] at hR2 hst2 ⊢
  have hpos : posOf { S' with session := some C } r₁.pl = ⟨0, 0, false⟩ := by
    simp only [posOf, opened, Option.map_some, Option.getD_some, hCo, List.length_nil]
    rw [hpl1]
    rfl
  have hT' : T = tmplFrom (labelsOf spec) (planOf spec sts) ((labelsOf spec).size + (planOf spec sts).size + 4)
      ⟨0, 0, false⟩ := by
    simp only [tmplOf, tmpl, List.cons.injEq, true_and] at hT
    exact hT.symm
  have hctl : ∀ k, ctlFrame { S' with session := some C } k = ctlOf (.ok (spec, sts)) k := by
    intro k
    rw [hS']
    rfl
  have key := sched_prefix hold (labelsOf spec) (planOf spec sts) (ctlOf (.ok (spec, sts))) rest
    ((labelsOf spec).size + (planOf spec sts).size + 4) _ _ hR2 hst2 hreg1 (by rw [hS']; rfl)
    (by rw [hS']; rfl) (by rw [hS']; rfl) (by rw [hpos]; simp only [Pos.left, Bool.false_eq_true, ite_false]; omega)
    hctl
  rw [hpos, ← hT'] at key
  exact ⟨List.cons_prefix_cons.mpr ⟨rfl, List.cons_prefix_cons.mpr ⟨rfl, key.1⟩⟩, frh _ _ key.2⟩

/-- **The release at the slots `ks`**: each control slot's frame `ctl k` in its place, and at each model slot the next
of `ms`. -/
def weave (ctl : Kind → ByteArray) : List Kind → List (List ByteArray) → List ByteArray
  | [], _ => []
  | k :: ks, ms => if isModel k then ms.headD [] ++ weave ctl ks ms.tail else ctl k :: weave ctl ks ms

/-- **What left at each model slot of the trace `tr` is the next of `ms`**, but at the trace's last slot, where it may
be any prefix of it: where the session stopped or ended there, or at its `Finish`, where the auditor stopped taking its
proofs or the hold kept them. -/
def RelayPre : List (Kind × List ByteArray) → List (List ByteArray) → Prop
  | [], _ => True
  | (k, s) :: tl, ms =>
    if isModel k then (s = ms.headD [] ∧ RelayPre tl ms.tail) ∨ (s <+: ms.headD [] ∧ tl = []) else RelayPre tl ms

/-- **A release over a cut of the slots is a prefix of the release over them all.** -/
theorem weave_prefix (ctl : Kind → ByteArray) :
    ∀ (ks ks' : List Kind) (ms : List (List ByteArray)), ks <+: ks' → weave ctl ks ms <+: weave ctl ks' ms
  | [], _, _, _ => List.nil_prefix
  | k :: ks, _, ms, ⟨t, rfl⟩ => by
    simp only [List.cons_append, weave]
    split
    · exact (List.prefix_append_right_inj _).mpr (weave_prefix ctl ks (ks ++ t) _ ⟨t, rfl⟩)
    · exact List.cons_prefix_cons.mpr ⟨rfl, weave_prefix ctl ks (ks ++ t) ms ⟨t, rfl⟩⟩

/-- **A trace whose control slots sent their frames and whose model slots sent `ms` is a prefix of its slots woven with
`ms`.** -/
theorem flatMap_pre (ctl : Kind → ByteArray) :
    ∀ (tr : List (Kind × List ByteArray)) (ms : List (List ByteArray)), Framed ctl tr → RelayPre tr ms →
      tr.flatMap (·.2) <+: weave ctl (tr.map (·.1)) ms
  | [], _, _, _ => List.nil_prefix
  | (k, s) :: tl, ms, ⟨hk, htl⟩, hp => by
    simp only [List.flatMap_cons, List.map_cons, weave]
    cases hm : isModel k
    · simp only [RelayPre, hm, Bool.false_eq_true, ite_false] at hp
      simp only [Bool.false_eq_true, ite_false]
      rcases hk hm with rfl | ⟨rfl, rfl⟩
      · exact List.cons_prefix_cons.mpr ⟨rfl, flatMap_pre ctl tl ms htl hp⟩
      · exact List.nil_prefix
    · simp only [RelayPre, hm, ite_true] at hp
      simp only [ite_true]
      rcases hp with ⟨rfl, hp⟩ | ⟨hs, rfl⟩
      · exact (List.prefix_append_right_inj _).mpr (flatMap_pre ctl tl _ htl hp)
      · simpa using hs.trans (List.prefix_append _ _)

/-- **What leaves at each slot of the relay's session**, from its start: the requests that reach a slot (`pick`), at
the schedule's places. -/
def relayTrace (hold : Bool) (ins : List (ByteArray × Env)) : List (Kind × List ByteArray) :=
  schedTrace hold {} {} (pick hold {} {} ins)

/-- **The deployed relay's release is a prefix of the schedule's**: at every requests the worker sends and every answer
of the auditor, at a setup `s` the relay reads, where what left at each model slot (the `Hello`, each coin request and
the `Finish`) is the next of `ms`, the last possibly cut (`RelayPre`), what leaves for the auditor is a prefix of the
session's control frames at their fixed places (`ctlOf s`: the `Register`, the `Link`, each `Open`) and `ms` at the
model's, over the whole schedule `tmplOf s`; nothing leaves after it. -/
theorem relay_release_pre (hold : Bool) (s : V (SessionSpec × Array Stmt)) (ins : List (ByteArray × Env))
    (ms : List (List ByteArray)) (hs : ∀ p ∈ ins, p.2.setup = s)
    (hb : ∀ p ∈ ins, ∀ why, (p.2.barrier (some why)).isSome) (h : RelayPre (relayTrace hold ins) ms) :
    out hold {} {} ins <+: weave (ctlOf s) (tmplOf s) ms := by
  obtain ⟨hsub, heq⟩ := out_sched hold {} {} ins (.inr ⟨rfl, reach_init⟩) hb
  have hmem : ∀ e ∈ pick hold {} {} ins, ∃ p ∈ ins, p.2 = e := fun e he => List.mem_map.mp (hsub.subset he)
  have hbe : ∀ e ∈ pick hold {} {} ins, ∀ why, (e.barrier (some why)).isSome := by
    intro e he
    obtain ⟨p, hp, rfl⟩ := hmem e he
    exact hb p hp
  have hse : ∀ e ∈ pick hold {} {} ins, e.setup = s := by
    intro e he
    obtain ⟨p, hp, rfl⟩ := hmem e he
    exact hs p hp
  obtain ⟨hpre, hfr⟩ := relay_prefix hold s _ hse
  rw [heq, schedOut_trace hold {} {} _ hbe]
  exact (flatMap_pre (ctlOf s) _ ms hfr h).trans (weave_prefix _ _ _ ms hpre)

/-- **How many frames left over the relay's session**, at most `B`. -/
def relayLen (hold : Bool) (B : Nat) (ins : List (ByteArray × Env)) : Fin (B + 1) :=
  ⟨min (out hold {} {} ins).length B, by omega⟩

/-- **The deployed relay's release is the schedule's, cut where it ended**: at `RelayPre`, the control frames and `ms`
woven over the whole schedule, cut after the frames that left (`relayLen`, where the woven release is at most `B`
long). -/
theorem relay_release_take {hold : Bool} {s : V (SessionSpec × Array Stmt)} {ins : List (ByteArray × Env)}
    {ms : List (List ByteArray)} {B : Nat} (hB : (weave (ctlOf s) (tmplOf s) ms).length ≤ B)
    (hs : ∀ p ∈ ins, p.2.setup = s) (hb : ∀ p ∈ ins, ∀ why, (p.2.barrier (some why)).isSome)
    (h : RelayPre (relayTrace hold ins) ms) :
    out hold {} {} ins = (weave (ctlOf s) (tmplOf s) ms).take (relayLen hold B ins) := by
  have hp := relay_release_pre hold s ins ms hs hb h
  have hl : (relayLen hold B ins : Nat) = (out hold {} {} ins).length :=
    Nat.min_eq_left (Nat.le_trans hp.length_le hB)
  rw [hl]
  exact List.prefix_iff_eq_take.mp hp

end Claims.ZKServer.Relay
