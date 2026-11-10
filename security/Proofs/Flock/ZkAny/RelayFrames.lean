module

public import Proofs.Flock.ZkAny.Relay
public import Proofs.Flock.Claims.Complete.Encode

/-!
# The relay's model frames are the model's messages

`Relay.relay_release_pre` reads `RelayPre`: what left at each model slot of the relay's session (the `Hello`, the
`Commit`, each `Round` and the `Finish`) is the next of some frames, the last possibly cut. Here those frames are the
model's messages under a public encoding at the setup (`encOf`), and `RelayPre` holds of the held relay where what it
read at each model slot is the model's message there (`EnvPre`).

* `encK`: the frames the relay sends for a model message at a slot of kind `k`: the `Hello`'s bytes; the link's level-0
  roots (`FlockDeploy.linkOf`), each named by its table, in a `Commit` frame at the setup's Σ; a round message's
  release, its bytes after the tag `0`, in its `Round` frame, after its stream's domain frame (`domOf`) at the stream's
  first round (`leaveOf`), as the loop sends it; the last message's proof files (`FlockDeploy.finalOf`) in `Proof`
  frames at their rep streams, then `Finish`. `encOf`: the messages at the schedule's model slots (`tmplOf`), in order,
  a stream's first round the first of its rounds there (`seenAfter`).
* `EnvOk`: what the relay read at a model slot is the message's: its nonce makes the message's `Hello`; the roots
  `Level0.commit` bound, where it bound any, are the link's; the release its rep's frame returned for a `Round`, where
  it returned one, is the round message's as it leaves (`leaveOf`); where the hold's barrier passes, the loop's own
  proofs are the last message's. `EnvPre`: `EnvOk` at each model slot of the relay's session (`relaySteps`), in order.
* `stepK_setup`: the session's Σ and its rep streams' domain frames, reps and labels are the setup's once its `Hello`
  left. `stepK_plan`: and so is its plan (`PlanOk`).
* `step_model`: at a model slot, what leaves is a prefix of the message's frames, and all of them where the session goes
  on.
* `relayPre_of_env`: `EnvPre` gives `RelayPre` at `encOf`.
* `relayB`, `weave_encOf_length`: the woven encoding is at most the schedule's slots times one more than the rep
  streams long, a public bound at the setup.
* `envOk_commit`, `envOk_round`, `envOk_finish`: `EnvOk` at the model's link, round and last messages, where what the
  relay read is those messages' contents.
* `dom_setupOk`, `leaveOf_first`: the domain frame the loop reads for a round's stream is `domOf`'s, and its release
  is `leaveOf`'s where its first round of the stream is the schedule's.

Only the held relay (`hold = true`) is covered. Without a hold, the `Finish` leaves with no barrier: alone where a rep's
frame did not finish, and without the proofs where the loop holds fewer than every rep's (`ownProofs`), where the worker
skipped its `rep` requests. Neither is a prefix of the model's last message.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session
open _root_.Flock.Firewall hiding State
open _root_.Flock (V SessionSpec Stmt)

/-- **A link message's roots** at the setup `spec`: its tables' level-0 roots (`FlockDeploy.linkOf`), each named by its
table, in table order. -/
def commitOf (spec : SessionSpec) (m : Claims.Msg) : Option (List (String × ByteArray)) :=
  (FlockDeploy.linkOf m).map fun (rootB, _) => spec.tables.toList.zip rootB

/-- **A round message's release**: its bytes after the tag `0`. -/
def roundOf : Claims.Msg → Option ByteArray
  | 0 :: c => some ⟨c.toArray⟩
  | _ => none

/-- **Proof files as `Proof` frames** at the setup `spec`: each at its rep stream's table and rep, in `spec.repStreams`
order. -/
def proofsOf (spec : SessionSpec) (ps : Array ByteArray) : List ByteArray :=
  (spec.repStreams.toList.zip ps.toList).map fun ((i, r, _), b) => proofFrame spec.tables[i]! r b

/-- **A last message's `Proof` frames** at the setup `spec`: its proof files (`FlockDeploy.finalOf`). -/
def finishOf (spec : SessionSpec) (m : Claims.Msg) : Option (List ByteArray) :=
  (FlockDeploy.finalOf m).map fun f => proofsOf spec f.proofs

/-- **The rep streams' domain frames** at the setup `spec`, each its transcript domain's (`Frames.domainFrame`), as the
`Hello` sets them (`State.domains`). -/
def domsOf (spec : SessionSpec) : Array (String × ByteArray) :=
  spec.repStreams.map fun (i, r, s) => (s, Frames.domainFrame (spec.domain spec.tables[i]! r))

/-- **The rep streams' tables and reps** at the setup `spec`, each its stream's, as the `Hello` sets them
(`State.reps`). -/
def streamRepsOf (spec : SessionSpec) : Array (String × Nat × Nat) :=
  spec.repStreams.map fun (i, r, s) => (s, i, r)

/-- **A rep stream's domain frame** at the setup `spec`, as the loop reads it from the session (`Session.relay`'s
`Round`). -/
def domOf (spec : SessionSpec) (stream : String) : ByteArray :=
  (((domsOf spec).find? (·.1 == stream)).map (·.2)).getD .empty

/-- **A round's release as it leaves** on the rep stream `stream`, where the streams whose rounds came before are
`seen`: after the stream's domain frame at its first round (`Frames.roundStep`). -/
def leaveOf (spec : SessionSpec) (seen : List String) (stream : String) (c : ByteArray) : ByteArray :=
  if seen.contains stream then c else domOf spec stream ++ c

/-- The rep streams whose rounds came, after a slot of kind `k`. -/
def seenAfter (seen : List String) : Kind → List String
  | .round stream _ => stream :: seen
  | _ => seen

/-- **The frames the relay sends for the model message `m` at a slot of kind `k`**, at the setup `spec`, where the
streams whose rounds came before are `seen`. -/
def encK (spec : SessionSpec) (seen : List String) : Kind → Claims.Msg → List ByteArray
  | .hello, m => [⟨m.toArray⟩]
  | .commit, m => match commitOf spec m with
    | some roots => [commitFrame spec.sigma roots]
    | none => []
  | .round stream n, m => match roundOf m with
    | some c => [roundFrame stream (leaveOf spec seen stream c) n]
    | none => []
  | .finish, m => match finishOf spec m with
    | some ps => ps ++ [FINISH]
    | none => []
  | _, _ => []

/-- **The model's frames over the slots `ks`**, after the rounds of the streams `seen`: at each model slot the next of
the messages `ms`, encoded at its kind. -/
def encT (spec : SessionSpec) : List String → List Kind → List Claims.Msg → List (List ByteArray)
  | _, [], _ => []
  | seen, k :: ks, ms =>
    if isModel k then encK spec seen k (ms.headD []) :: encT spec (seenAfter seen k) ks ms.tail
    else encT spec (seenAfter seen k) ks ms

/-- **The model's frames at the setup**: its messages `ms` at the schedule's model slots, in order. -/
def encOf (spec : SessionSpec) (sts : Array Stmt) (ms : List Claims.Msg) : List (List ByteArray) :=
  encT spec [] (tmplOf (.ok (spec, sts))) ms

/-- **What the relay read at a model slot of kind `k` is the model message `m`'s**, where the streams whose rounds came
before are `seen`: at the `Hello`, its nonce makes `m`; at the `Commit`, the roots `Level0.commit` bound, where it bound
any, are `m`'s; at a `Round`, the release its rep's frame returned, where it returned one, is `m`'s as it leaves
(`leaveOf`); at the `Finish`, where the hold's barrier passes, the loop's own proofs are `m`'s. -/
def EnvOk (spec : SessionSpec) (seen : List String) : Kind → Env → Claims.Msg → Prop
  | .hello, e, m => Hello.msg spec e.nonce = m
  | .commit, e, m => ∀ roots, e.commit = .ok roots → commitOf spec m = some roots
  | .round stream _, e, m => ∀ c, e.round = .ok c → ∃ c', roundOf m = some c' ∧ c = leaveOf spec seen stream c'
  | .finish, e, m => e.barrier none = none → finishOf spec m = some e.proofs
  | _, _, _ => True

/-- **Each slot's request as the relay read it**, in order: the slot's kind, the session and placement before it, and
what the relay read at it, up to the session's stop or end (`schedTrace`'s slots). -/
def schedSteps (hold : Bool) : State → Placed → List Env → List (Kind × State × Placed × Env)
  | _, _, [] => []
  | st, pl, e :: rest => match slot st pl with
    | none => []
    | some k =>
      let r := stepK hold st pl (some (tagOf k)) (.ok k) e
      (k, st, pl, e) :: if r.go && r.st.stop.isNone then schedSteps hold r.st r.pl rest else []

/-- The requests at the relay's session's slots, from its start (`relayTrace`'s). -/
def relaySteps (hold : Bool) (ins : List (ByteArray × Env)) : List (Kind × State × Placed × Env) :=
  schedSteps hold {} {} (pick hold {} {} ins)

/-- **What the relay read at each model slot is the model's message there**, the messages `ms` in order, after the
rounds of the streams `seen`. -/
def EnvPre (spec : SessionSpec) : List String → List (Kind × State × Placed × Env) → List Claims.Msg → Prop
  | _, [], _ => True
  | seen, (k, _, _, e) :: tl, ms =>
    if isModel k then EnvOk spec seen k e (ms.headD []) ∧ EnvPre spec (seenAfter seen k) tl ms.tail
    else EnvPre spec (seenAfter seen k) tl ms

/-! ## The session's Σ and domains -/

/-- What a request keeps of the session: its Σ, its rep streams' domain frames, whether its `Hello` left, its rep
streams' reps and labels, and its plan. -/
abbrev Keeps (S S' : State) : Prop :=
  S'.sigma = S.sigma ∧ S'.domains = S.domains ∧ S'.sent = S.sent ∧ S'.reps = S.reps ∧ S'.labels = S.labels ∧
    S'.plan = S.plan

theorem finish_keeps {S S' : State} {o : ByteArray} (hf : S.finish = some (S', o)) : Keeps S S' := by
  revert hf
  unfold State.finish State.refuse
  repeat' split
  all_goals intro hf; first
    | (simp only [Option.some.injEq, Prod.mk.injEq] at hf; rw [← hf.1]; exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩)
    | simp at hf

theorem request_keeps {S S' : State} {P : Placed} {k : Kind} {o : ByteArray} (hr : S.request P k = some (S', o)) :
    Keeps S S' := by
  revert hr
  unfold State.request
  split
  · intro hr
    simp only [Option.some.injEq, Prod.mk.injEq] at hr
    rw [← hr.1]
    exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
  · dsimp only
    split
    · rename_i r hc
      intro hr
      simp only [Option.some.injEq] at hr
      subst hr
      revert hc
      split
      · exact finish_keeps
      · unfold State.refuse
        repeat' split
        all_goals intro hc; first
          | (simp only [Option.some.injEq, Prod.mk.injEq] at hc; rw [← hc.1]; exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩)
          | simp at hc
    · rcases hp : S.place P k with _ | why
      · simp
      · simp only [Option.map_some, Option.some.injEq]
        intro hr
        cases hr
        exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem coins_keeps (S : State) (a : ByteArray) : Keeps S (S.coins a).1 := by
  unfold State.coins State.refuse
  repeat' split
  all_goals exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem noCoins_keeps (S : State) (what : String) (a : ByteArray) : Keeps S (S.noCoins what a).1 := by
  unfold State.noCoins State.refuse
  split
  all_goals exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem answer_keeps (S : State) (k : Kind) (a id : ByteArray) : Keeps S (S.answer k a id).1 := by
  unfold State.answer
  split
  all_goals first
    | exact coins_keeps S a
    | exact noCoins_keeps S _ a
    | exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
    | ((unfold State.refuse; repeat' split) <;> exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩)

theorem sendHello_keeps {S S' : State} {s : V (SessionSpec × Array Stmt)} {n o : ByteArray}
    (he : S.sendHello s n = .error (S', o)) : Keeps S S' := by
  revert he
  unfold State.sendHello State.refuse
  repeat' split
  all_goals intro he; first | (cases he; exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩) | cases he

theorem afterAnswer_keeps (st : State) (k : Kind) (r : ByteArray) (e : Env) : Keeps st (afterAnswer st k r e) := by
  cases k
  case round s n =>
    have h := answer_keeps st (.round s n) r .empty
    unfold afterAnswer
    dsimp only
    repeat' split
    all_goals exact h
  all_goals exact answer_keeps _ _ _ _

/-- **The session's Σ and its rep streams' domain frames, reps and labels are the setup's** once its `Hello` left. -/
def SetupOk (spec : SessionSpec) (S : State) : Prop :=
  S.sent = true →
    S.sigma = spec.sigma ∧ S.domains = domsOf spec ∧ S.reps = streamRepsOf spec ∧ S.labels = labelsOf spec

theorem setupOk_keeps {spec : SessionSpec} {S S' : State} (h : SetupOk spec S) (hk : Keeps S S') : SetupOk spec S' := by
  intro hs
  obtain ⟨h1, h2, h3, h4⟩ := h (hk.2.2.1.symm.trans hs)
  exact ⟨hk.1.trans h1, hk.2.1.trans h2, hk.2.2.2.1.trans h3, hk.2.2.2.2.1.trans h4⟩

/-- **Each request keeps the session's Σ, domain frames, reps and labels the setup's**, where the relay reads the
setup. -/
theorem stepK_setup {hold : Bool} {spec : SessionSpec} {sts : Array Stmt} {st : State} {pl : Placed}
    {b : Option UInt8} {k : Except String Kind} {e : Env} (he : e.setup = .ok (spec, sts)) (h : SetupOk spec st) :
    SetupOk spec (stepK hold st pl b k e).st := by
  have hello : ∀ {S' : State} {o : ByteArray}, st.sendHello e.setup e.nonce = .ok (S', o) → SetupOk spec S' := by
    intro S' o hh _
    obtain ⟨spec', sts', hsp, rfl, -⟩ := sendHello_eq hh
    rw [he] at hsp
    cases hsp
    exact ⟨rfl, rfl, rfl, rfl⟩
  unfold stepK
  split
  · exact h
  split
  · exact h
  split
  · exact h
  split
  · exact h
  · split
    · exact setupOk_keeps h (request_keeps ‹_›)
    · split
      · exact setupOk_keeps h (sendHello_keeps ‹_›)
      · rename_i hh
        split
        · exact hello hh
        · exact setupOk_keeps (hello hh) (answer_keeps _ _ _ _)
  · split
    · exact setupOk_keeps h (request_keeps ‹_›)
    · split
      · exact setupOk_keeps h ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
      · split
        · exact h
        · exact setupOk_keeps h (afterAnswer_keeps _ _ _ _)

/-- **The session's plan is the setup's** once its `Hello` left. -/
def PlanOk (spec : SessionSpec) (sts : Array Stmt) (S : State) : Prop :=
  S.sent = true → S.plan = _root_.Flock.Firewall.Session.planOf spec sts

theorem planOk_keeps {spec : SessionSpec} {sts : Array Stmt} {S S' : State} (h : PlanOk spec sts S) (hk : Keeps S S') :
    PlanOk spec sts S' := fun hs => hk.2.2.2.2.2.trans (h (hk.2.2.1.symm.trans hs))

/-- **Each request keeps the session's plan the setup's**, where the relay reads the setup. -/
theorem stepK_plan {hold : Bool} {spec : SessionSpec} {sts : Array Stmt} {st : State} {pl : Placed}
    {b : Option UInt8} {k : Except String Kind} {e : Env} (he : e.setup = .ok (spec, sts)) (h : PlanOk spec sts st) :
    PlanOk spec sts (stepK hold st pl b k e).st := by
  have hello : ∀ {S' : State} {o : ByteArray}, st.sendHello e.setup e.nonce = .ok (S', o) → PlanOk spec sts S' := by
    intro S' o hh _
    obtain ⟨spec', sts', hsp, rfl, -⟩ := sendHello_eq hh
    rw [he] at hsp
    cases hsp
    rfl
  unfold stepK
  split
  · exact h
  split
  · exact h
  split
  · exact h
  split
  · exact h
  · split
    · exact planOk_keeps h (request_keeps ‹_›)
    · split
      · exact planOk_keeps h (sendHello_keeps ‹_›)
      · rename_i hh
        split
        · exact hello hh
        · exact planOk_keeps (hello hh) (answer_keeps _ _ _ _)
  · split
    · exact planOk_keeps h (request_keeps ‹_›)
    · split
      · exact planOk_keeps h ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
      · split
        · exact h
        · exact planOk_keeps h (afterAnswer_keeps _ _ _ _)

/-! ## At a model slot -/

theorem proofsOut_prefix (held : List ByteArray) (taken : Nat) : proofsOut held taken <+: held ++ [FINISH] := by
  unfold proofsOut
  split
  · exact (List.take_prefix _ _).trans (List.prefix_append _ _)
  · exact List.prefix_refl _

theorem finishOut_true (st : State) (pl : Placed) (e : Env) : ∃ stopped : Option String,
    finishOut true st pl e =
      if (e.barrier stopped).isSome then [] else proofsOut (if stopped.isNone then e.proofs else []) e.taken := by
  unfold finishOut
  simp only [↓reduceIte]
  exact ⟨_, rfl⟩

/-- **A held `Finish` sends nothing, or where the hold's barrier passes a prefix of the loop's proofs and `Finish`.** -/
theorem finishOut_hold {st : State} {pl : Placed} {e : Env} (hb : ∀ why, (e.barrier (some why)).isSome) :
    finishOut true st pl e = [] ∨ (e.barrier none = none ∧ finishOut true st pl e <+: e.proofs ++ [FINISH]) := by
  obtain ⟨stopped, h⟩ := finishOut_true st pl e
  rw [h]
  rcases stopped with _ | why
  · rcases hn : e.barrier none with _ | w
    · refine .inr ⟨rfl, ?_⟩
      simpa using proofsOut_prefix e.proofs e.taken
    · exact .inl (by simp)
  · exact .inl (by simp [hb why])

theorem step_frame {hold : Bool} {st : State} {pl : Placed} {k : Kind} {e : Env} {F : List ByteArray}
    (h : st.stop = none) (hk1 : k ≠ .hello) (hk2 : k ≠ .finish) (hk3 : k ≠ .ping) (hk4 : k ≠ .proof)
    (hF : ∀ g, frameOf st k e = .ok g → [g] = F) :
    (stepK hold st pl (some (tagOf k)) (.ok k) e).sent <+: F ∧
      ((stepK hold st pl (some (tagOf k)) (.ok k) e).go = true →
        (stepK hold st pl (some (tagOf k)) (.ok k) e).st.stop = none →
        (stepK hold st pl (some (tagOf k)) (.ok k) e).sent = F) := by
  obtain ⟨h9, h5, h6⟩ := tagOf_ne hk3 hk4 hk2
  rcases hr : st.request pl k with _ | ⟨st', o⟩
  · rw [stepK_frame hk1 hk2 hk3 hk4 hr]
    rcases hf : frameOf st k e with why | g
    · dsimp only
      refine ⟨List.nil_prefix, fun _ hs => ?_⟩
      simp [State.refuse] at hs
    · have hg := hF g hf
      dsimp only
      rcases ha : e.ans with _ | r
      · exact ⟨⟨[], by rw [List.append_nil]; exact hg⟩, fun _ _ => hg⟩
      · exact ⟨⟨[], by rw [List.append_nil]; exact hg⟩, fun _ _ => hg⟩
  · rw [stepK_refused h9 h5 h6 hr]
    dsimp only
    refine ⟨List.nil_prefix, fun _ hs => ?_⟩
    have hx := request_stop h hr
    rw [hs] at hx
    cases hx

theorem step_hello {hold : Bool} {spec : SessionSpec} {sts : Array Stmt} {seen : List String} {st : State}
    {pl : Placed} {e : Env} {m : Claims.Msg} (h : st.stop = none) (hs : e.setup = .ok (spec, sts))
    (he : Hello.msg spec e.nonce = m) :
    (stepK hold st pl (some (tagOf .hello)) (.ok .hello) e).sent <+: encK spec seen .hello m ∧
      ((stepK hold st pl (some (tagOf .hello)) (.ok .hello) e).go = true →
        (stepK hold st pl (some (tagOf .hello)) (.ok .hello) e).st.stop = none →
        (stepK hold st pl (some (tagOf .hello)) (.ok .hello) e).sent = encK spec seen .hello m) := by
  rcases hr : st.request pl .hello with _ | ⟨st', o⟩
  · rw [stepK_hello hr]
    rcases hh : st.sendHello e.setup e.nonce with ⟨st', o⟩ | ⟨st', o⟩
    · dsimp only
      refine ⟨List.nil_prefix, fun _ hs' => ?_⟩
      have hx := sendHello_error h hh
      rw [hs'] at hx
      cases hx
    · obtain ⟨spec', sts', hsp, -, ho⟩ := sendHello_eq hh
      rw [hs] at hsp
      cases hsp
      have hg : [o] = encK spec seen .hello m := by
        show [o] = [⟨m.toArray⟩]
        rw [ho, he]
      dsimp only
      rcases ha : e.ans with _ | r
      · exact ⟨⟨[], by rw [List.append_nil]; exact hg⟩, fun _ _ => hg⟩
      · exact ⟨⟨[], by rw [List.append_nil]; exact hg⟩, fun _ _ => hg⟩
  · rw [stepK_refused (by decide) (by decide) (by decide) hr]
    dsimp only
    refine ⟨List.nil_prefix, fun _ hs' => ?_⟩
    have hx := request_stop h hr
    rw [hs'] at hx
    cases hx

theorem step_finish {spec : SessionSpec} {seen : List String} {st : State} {pl : Placed} {e : Env} {m : Claims.Msg}
    (hb : ∀ why, (e.barrier (some why)).isSome) (he : e.barrier none = none → finishOf spec m = some e.proofs) :
    (stepK true st pl (some (tagOf .finish)) (.ok .finish) e).sent <+: encK spec seen .finish m ∧
      ((stepK true st pl (some (tagOf .finish)) (.ok .finish) e).go = true →
        (stepK true st pl (some (tagOf .finish)) (.ok .finish) e).st.stop = none →
        (stepK true st pl (some (tagOf .finish)) (.ok .finish) e).sent = encK spec seen .finish m) := by
  rw [stepK_finish]
  refine ⟨?_, fun hgo => by cases hgo⟩
  rcases finishOut_hold (st := st) (pl := pl) hb with h0 | ⟨hn, hp⟩
  · rw [h0]
    exact List.nil_prefix
  · have he' : encK spec seen .finish m = e.proofs ++ [FINISH] := by simp only [encK, he hn]
    rw [he']
    exact hp

/-- **At a model slot of the held relay what leaves is a prefix of the message's frames**, and all of them where the
session goes on, where what the relay read there is the message's (`EnvOk`). -/
theorem step_model {spec : SessionSpec} {sts : Array Stmt} {seen : List String} {st : State} {pl : Placed}
    {k : Kind} {e : Env} {m : Claims.Msg} (h : st.stop = none) (hsl : slot st pl = some k) (hm : isModel k = true)
    (hσ : SetupOk spec st) (hs : e.setup = .ok (spec, sts)) (hb : ∀ why, (e.barrier (some why)).isSome)
    (he : EnvOk spec seen k e m) :
    (stepK true st pl (some (tagOf k)) (.ok k) e).sent <+: encK spec seen k m ∧
      ((stepK true st pl (some (tagOf k)) (.ok k) e).go = true →
        (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop = none →
        (stepK true st pl (some (tagOf k)) (.ok k) e).sent = encK spec seen k m) := by
  cases k
  case hello => exact step_hello h hs he
  case finish => exact step_finish hb he
  case commit =>
    refine step_frame h (by intro h; cases h) (by intro h; cases h) (by intro h; cases h) (by intro h; cases h) ?_
    intro g hg
    rcases hc : e.commit with why | roots
    · simp [frameOf, hc] at hg
    · have hg' : g = commitFrame st.sigma roots := by
        simp [frameOf, hc] at hg
        exact hg.symm
      have hsig : st.sigma = spec.sigma := (hσ (slot_commit hsl).1).1
      have hco : commitOf spec m = some roots := he roots hc
      simp only [encK, hco, hg', hsig]
  case round stream n =>
    refine step_frame h (by intro h; cases h) (by intro h; cases h) (by intro h; cases h) (by intro h; cases h) ?_
    intro g hg
    rcases hc : e.round with why | c
    · simp [frameOf, hc] at hg
    · have hg' : g = roundFrame stream c n := by
        simp [frameOf, hc] at hg
        exact hg.symm
      obtain ⟨c', hro, rfl⟩ := he c hc
      simp only [encK, hro, hg']
  all_goals cases hm

/-! ## Over the session -/

theorem prefix_cons_inv {α : Type} {a : α} {l ks : List α} (h : a :: l <+: ks) :
    ∃ ks', ks = a :: ks' ∧ l <+: ks' := by
  obtain ⟨t, rfl⟩ := h
  exact ⟨l ++ t, rfl, ⟨t, rfl⟩⟩

/-- **The held relay's model frames are the messages' at the schedule's places**, from a session going on with its Σ
the setup's, after the rounds of the streams `seen`, where the slots that come are a prefix of `ks` and what the relay
read at each model slot is the message's there. -/
theorem relayPre_sched {spec : SessionSpec} {sts : Array Stmt} :
    ∀ (es : List Env) (st : State) (pl : Placed) (seen : List String) (ks : List Kind) (ms : List Claims.Msg),
      st.stop = none → SetupOk spec st → (∀ e ∈ es, e.setup = .ok (spec, sts)) →
      (∀ e ∈ es, ∀ why, (e.barrier (some why)).isSome) → (schedTrace true st pl es).map (·.1) <+: ks →
      EnvPre spec seen (schedSteps true st pl es) ms → RelayPre (schedTrace true st pl es) (encT spec seen ks ms)
  | [], _, _, _, _, _, _, _, _, _, _, _ => trivial
  | e :: rest, st, pl, seen, ks, ms, h, hσ, hs, hb, hpre, henv => by
    rcases hsl : slot st pl with _ | k
    · simp only [schedTrace, hsl, RelayPre]
    have hse : e.setup = .ok (spec, sts) := hs e (List.mem_cons_self ..)
    have hbe := hb e (List.mem_cons_self ..)
    have hsr : ∀ e ∈ rest, e.setup = .ok (spec, sts) := fun e he => hs e (List.mem_cons_of_mem _ he)
    have hbr : ∀ e ∈ rest, ∀ why, (e.barrier (some why)).isSome := fun e he => hb e (List.mem_cons_of_mem _ he)
    have hσ' := stepK_setup (hold := true) (pl := pl) (b := some (tagOf k)) (k := .ok k) hse hσ
    simp only [schedTrace, hsl, List.map_cons] at hpre ⊢
    simp only [schedSteps, hsl] at henv
    obtain ⟨ks', rfl, hpre'⟩ := prefix_cons_inv hpre
    by_cases hm : isModel k = true
    · simp only [EnvPre, hm, ↓reduceIte] at henv
      have hstep := step_model h hsl hm hσ hse hbe henv.1
      generalize hr : stepK true st pl (some (tagOf k)) (.ok k) e = r at hσ' hpre' henv hstep ⊢
      simp only [RelayPre, encT, hm, ↓reduceIte, List.headD_cons, List.tail_cons]
      by_cases hc : (r.go && r.st.stop.isNone) = true
      · rw [ite_eq_left hc] at hpre' henv ⊢
        obtain ⟨hgo, hst⟩ := (Bool.and_eq_true _ _).mp hc
        have hst' : r.st.stop = none := Option.isNone_iff_eq_none.mp hst
        exact .inl ⟨hstep.2 hgo hst',
          relayPre_sched rest r.st r.pl (seenAfter seen k) ks' ms.tail hst' hσ' hsr hbr hpre' henv.2⟩
      · rw [ite_eq_right hc]
        exact .inr ⟨hstep.1, rfl⟩
    · have hm' : isModel k = false := by simpa using hm
      simp only [EnvPre, hm', Bool.false_eq_true, ite_false] at henv
      generalize hr : stepK true st pl (some (tagOf k)) (.ok k) e = r at hσ' hpre' henv ⊢
      simp only [RelayPre, encT, hm', Bool.false_eq_true, ite_false]
      by_cases hc : (r.go && r.st.stop.isNone) = true
      · rw [ite_eq_left hc] at hpre' henv ⊢
        obtain ⟨-, hst⟩ := (Bool.and_eq_true _ _).mp hc
        exact relayPre_sched rest r.st r.pl (seenAfter seen k) ks' ms (Option.isNone_iff_eq_none.mp hst) hσ' hsr hbr
          hpre' henv
      · rw [ite_eq_right hc]
        trivial

/-- **The held relay's model frames are the model's messages, encoded**: at every requests the worker sends and every
answer of the auditor, at the setup `spec`, `sts` the relay reads, where what the relay read at each model slot is the
model's message there, in order (`EnvPre`), what left at each model slot is that message's frames (`encOf`), the last
possibly cut (`RelayPre`). -/
theorem relayPre_of_env {spec : SessionSpec} {sts : Array Stmt} (ins : List (ByteArray × Env)) (ms : List Claims.Msg)
    (hs : ∀ p ∈ ins, p.2.setup = .ok (spec, sts)) (hb : ∀ p ∈ ins, ∀ why, (p.2.barrier (some why)).isSome)
    (he : EnvPre spec [] (relaySteps true ins) ms) : RelayPre (relayTrace true ins) (encOf spec sts ms) := by
  obtain ⟨hsub, -⟩ := out_sched true {} {} ins (.inr ⟨rfl, reach_init⟩) hb
  have hmem : ∀ e ∈ pick true {} {} ins, ∃ p ∈ ins, p.2 = e := fun e he => List.mem_map.mp (hsub.subset he)
  have hse : ∀ e ∈ pick true {} {} ins, e.setup = .ok (spec, sts) := by
    intro e he
    obtain ⟨p, hp, rfl⟩ := hmem e he
    exact hs p hp
  have hbe : ∀ e ∈ pick true {} {} ins, ∀ why, (e.barrier (some why)).isSome := by
    intro e he
    obtain ⟨p, hp, rfl⟩ := hmem e he
    exact hb p hp
  exact relayPre_sched _ {} {} [] _ ms rfl (fun h => by cases h) hse hbe (relay_prefix true _ _ hse).1 he

/-! ## The release's length -/

theorem proofsOf_length (spec : SessionSpec) (ps : Array ByteArray) :
    (proofsOf spec ps).length ≤ spec.repStreams.size := by
  simp only [proofsOf, List.length_map, List.length_zip, Array.length_toList]
  omega

theorem finishOf_length {spec : SessionSpec} {m : Claims.Msg} {ps : List ByteArray} (h : finishOf spec m = some ps) :
    ps.length ≤ spec.repStreams.size := by
  unfold finishOf at h
  rcases hf : FlockDeploy.finalOf m with _ | f
  · simp [hf] at h
  · obtain rfl : proofsOf spec f.proofs = ps := by simpa [hf] using h
    exact proofsOf_length spec f.proofs

/-- **A model slot sends at most one frame per rep stream, and one more.** -/
theorem encK_length (spec : SessionSpec) (seen : List String) (k : Kind) (m : Claims.Msg) :
    (encK spec seen k m).length ≤ spec.repStreams.size + 1 := by
  unfold encK
  split
  · simp only [List.length_cons, List.length_nil]
    omega
  · split <;> simp only [List.length_cons, List.length_nil] <;> omega
  · split <;> simp only [List.length_cons, List.length_nil] <;> omega
  · split
    · have := finishOf_length ‹finishOf _ _ = some _›
      simp only [List.length_append, List.length_cons, List.length_nil]
      omega
    · simp only [List.length_nil]
      omega
  · simp only [List.length_nil]
    omega

theorem weave_encT_length (spec : SessionSpec) (ctl : Kind → ByteArray) :
    ∀ (seen : List String) (ks : List Kind) (ms : List Claims.Msg),
      (weave ctl ks (encT spec seen ks ms)).length ≤ ks.length * (spec.repStreams.size + 1)
  | _, [], _ => by simp [weave]
  | seen, k :: ks, ms => by
    rw [List.length_cons, Nat.add_mul, Nat.one_mul]
    by_cases hm : isModel k = true
    · have := encK_length spec seen k (ms.headD [])
      have := weave_encT_length spec ctl (seenAfter seen k) ks ms.tail
      simp only [encT, weave, hm, ↓reduceIte, List.headD_cons, List.tail_cons, List.length_append]
      omega
    · have hm' : isModel k = false := by simpa using hm
      have := weave_encT_length spec ctl (seenAfter seen k) ks ms
      simp only [encT, weave, hm', Bool.false_eq_true, ite_false, List.length_cons]
      omega

/-- **A bound on the held relay's release at the setup**: the schedule's slots, each sending at most one frame per rep
stream and one more. -/
def relayB (spec : SessionSpec) (sts : Array Stmt) : Nat :=
  (tmplOf (.ok (spec, sts))).length * (spec.repStreams.size + 1)

/-- **The model's frames at the setup, woven with its control frames, are at most `relayB` long.** -/
theorem weave_encOf_length (spec : SessionSpec) (sts : Array Stmt) (ms : List Claims.Msg) :
    (weave (ctlOf (.ok (spec, sts))) (tmplOf (.ok (spec, sts))) (encOf spec sts ms)).length ≤ relayB spec sts :=
  weave_encT_length spec _ [] _ ms

/-! ## At the model's messages -/

/-- **At the model's link**, `EnvOk` where the roots `Level0.commit` bound are the link's, named by the setup's
tables. -/
theorem envOk_commit {spec : SessionSpec} {seen : List String} {e : Env} {rootB publics : List ByteArray}
    (hr : rootB.length < 2 ^ 64) (hp : publics.length < 2 ^ 64) (hrb : ∀ b ∈ rootB, b.size < 2 ^ 64)
    (hpb : ∀ b ∈ publics, b.size < 2 ^ 64)
    (h : ∀ roots, e.commit = .ok roots → roots = spec.tables.toList.zip rootB) :
    EnvOk spec seen .commit e (0 :: FlockDeploy.linkBody rootB publics) := by
  intro roots hc
  simp only [commitOf, FlockComplete.linkOf_linkBody hr hp hrb hpb, Option.map_some, h roots hc]

/-- **At a round message**, `EnvOk` where the release the rep's frame returned is the message's as it leaves. -/
theorem envOk_round {spec : SessionSpec} {seen : List String} {e : Env} {stream : String} {n : Nat} {m : Claims.Msg}
    (h : ∀ c, e.round = .ok c → ∃ c', 0 :: c'.data.toList = m ∧ c = leaveOf spec seen stream c') :
    EnvOk spec seen (.round stream n) e m := by
  intro c hc
  obtain ⟨c', hm, rfl⟩ := h c hc
  refine ⟨c', ?_, rfl⟩
  rw [← hm]
  cases c'
  simp [roundOf]

/-- **The loop's domain frame for a stream is the setup's** once the session's `Hello` left (`Session.relay`'s `Round`
reads it from `State.domains`). -/
theorem dom_setupOk {spec : SessionSpec} {st : State} (h : SetupOk spec st) (hs : st.sent = true) (stream : String) :
    ((st.domains.find? (·.1 == stream)).map (·.2)).getD .empty = domOf spec stream := by
  rw [(h hs).2.1]
  rfl

/-- **A round's release as the loop sends it is `leaveOf`'s**, where the loop's first round of the stream is the
schedule's (no round of it is in `seen`) and its domain frame is the setup's. -/
theorem leaveOf_first {spec : SessionSpec} {seen : List String} {stream : String} {f : Bool} {dom c : ByteArray}
    (hf : f = !seen.contains stream) (hd : dom = domOf spec stream) :
    (if f then dom ++ c else c) = leaveOf spec seen stream c := by
  subst hf hd
  unfold leaveOf
  cases seen.contains stream <;> rfl

/-- **At the model's last message**, `EnvOk` where, once the hold's barrier passes, the loop's own proofs are its proof
files as `Proof` frames. -/
theorem envOk_finish {spec : SessionSpec} {seen : List String} {e : Env} {f : FlockDeploy.Final}
    (hs : f.stmt.length < 2 ^ 64) (hr : f.record.toUTF8.size < 2 ^ 64) (hp : f.proofs.size < 2 ^ 64)
    (hb : ∀ b ∈ f.proofs, b.size < 2 ^ 64)
    (h : e.barrier none = none → e.proofs = proofsOf spec f.proofs) : EnvOk spec seen .finish e f.encode := by
  intro hn
  simp only [finishOf, FlockComplete.finalOf_encode hs hr hp hb, Option.map_some, h hn]

end Claims.ZKServer.Relay
