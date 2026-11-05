module

public import Proofs.Flock.ZkAny.RelayInv
public import Proofs.Flock.ZkAny.RelayEnv
public import Proofs.Flock.ZkAny.RelayFresh

/-!
# The relay's requests as the deployed loop answers them

`envPre_inv` gives `EnvPre` from any invariant of the relay's requests that each slot's step keeps. Here is the
invariant at the deployed loop, with the cases of the slots that read no model message proved:

* `LoopAt`: the loop between two of the session's requests: its frames, its level-0 slots and how many of the model's
  messages the session's slots took.
* `LoopRead`: what the relay reads at a slot as the loop answers it, and the loop at the session's next request: the
  `Hello`'s nonce the relay drew, the `Commit`'s roots `commitRead` reads off the slots, a frame round as the loop ran
  it (`RoundRun`), the `Finish`'s proofs the loop's proof files as `Proof` frames; at every other slot only the updates
  between the session's requests (`Between`).
* `LoopReads`: the requests to come, each read so, up to the session's stop or end.
* `LoopFacts`: the session's Σ and domain frames the setup's, each rep whose stream had no round yet with its frame
  absent or fresh (`FreshAt`), and no model message taken before the `Hello` left.
* `LoopP`: the model's messages to come those past the ones the slots took, `LoopFacts`, `LoopReads` from there, and
  every request reading the session's setup.
* `stepK_sent`: each request but a `Hello` keeps whether the `Hello` left. `stepK_hello_sent`: the session goes on
  past its `Hello` only with it sent.
* `SlotCase`: `envPre_inv`'s step at the slots of one kind. `slotCase_carried`: at a slot that reads no model message,
  `LoopP` carries over. `slotCase_hello`: the `Hello`'s, where the model's first message is the `Hello` at the drawn
  nonce.
* **`envPre_loop`**: `EnvPre` from `LoopP`, given `SlotCase` at the `Hello`, the `Commit`, each frame round and the
  `Finish`.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session
open _root_.Flock.Firewall hiding State
open _root_.Flock (SessionSpec Stmt)
open _root_.Flock.Firewall.Frames (Sessions)

/-- **The loop between two of the session's requests**: its frames, its level-0 slots, and how many of the model's
messages the session's slots took so far. -/
structure LoopAt where
  R : Sessions
  L0 : _root_.Flock.Firewall.Level0.Slots
  c : Nat

/-- **What the relay reads at a slot of kind `k` of session `sid`**, from the session `st` and placement `pl`, as the
loop at `σ` answers it, the loop being at `σ'` at the session's next request: at the `Hello` the nonce `nonce` the
relay drew; at the `Commit` the roots `commitRead` reads off the loop's level-0 slots at the session's tables (the
setup's, each with its statement's `m`); at a frame round a `RoundRun` from `σ`'s frames and level 0s to `σ'`'s frames;
at the `Finish` the loop's proof files as `Proof` frames, once the hold's barrier passes. Between the session's
requests only the updates of the loop's other requests come (`Between`), and each slot that reads a model message
takes one. -/
def LoopRead (sid : Nat) (spec : SessionSpec) (sts : Array Stmt) (nonce : ByteArray) :
    Session.Kind → State → Placed → Env → LoopAt → LoopAt → Prop
  | .hello, _, _, e, σ, σ' => e.nonce = nonce ∧ Between sid σ.R σ'.R ∧ σ'.c = σ.c + 1
  | .commit, _, _, e, σ, σ' =>
    (∃ f, e.commit = (commitRead σ.L0 sid (spec.tables.zip (sts.map (·.m))) f).1) ∧ Between sid σ.R σ'.R ∧
      σ'.c = σ.c + 1
  | .round stream n, st, pl, e, σ, σ' =>
    (∃ i r dom content, RoundRun sid st pl stream n i r dom content e σ.R σ.L0 σ'.R) ∧ σ'.c = σ.c + 1
  | .finish, _, _, e, σ, σ' =>
    (∃ ps : Array ByteArray, e.barrier none = none → e.proofs = proofsOf spec ps) ∧ σ'.c = σ.c + 1
  | _, _, _, _, σ, σ' => Between sid σ.R σ'.R ∧ σ'.c = σ.c

/-- **The relay's requests `es` to come as the loop answers them**, from the session `st`, placement `pl` and the loop
at `σ`: at each request at a slot, what `LoopRead` reads, and the rest so from the loop at the session's next request
where the session goes on; nothing past the session's end. -/
inductive LoopReads (sid : Nat) (spec : SessionSpec) (sts : Array Stmt) (nonce : ByteArray) :
    LoopAt → State → Placed → List Env → Prop
  | nil (σ : LoopAt) (st : State) (pl : Placed) : LoopReads sid spec sts nonce σ st pl []
  | stop {σ : LoopAt} {st : State} {pl : Placed} {e : Env} {rest : List Env} :
      slot st pl = none → LoopReads sid spec sts nonce σ st pl (e :: rest)
  | step {σ σ' : LoopAt} {st : State} {pl : Placed} {e : Env} {rest : List Env} (k : Session.Kind) :
      slot st pl = some k → LoopRead sid spec sts nonce k st pl e σ σ' →
      (((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
          (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true →
        LoopReads sid spec sts nonce σ' (stepK true st pl (some (tagOf k)) (.ok k) e).st
          (stepK true st pl (some (tagOf k)) (.ok k) e).pl rest) →
      LoopReads sid spec sts nonce σ st pl (e :: rest)

/-- **What holds of the session and the loop between its requests**, the streams whose rounds came being `seen`: the
session's Σ and domain frames are the setup's once its `Hello` left, each rep whose stream had no round yet has its
frame absent or fresh, and until the `Hello` left the slots took none of the model's messages. -/
structure LoopFacts (sid : Nat) (spec : SessionSpec) (seen : List String) (st : State) (σ : LoopAt) : Prop where
  setup : SetupOk spec st
  fresh : ∀ i r s, (i, r, s) ∈ spec.repStreams.toList → seen.contains s = false → FreshAt σ.R sid i r
  start : st.sent = false → σ.c = 0

/-- **The relay's invariant at the deployed loop**, against the model's messages `ms₀`: at some loop `σ`, the model's
messages to come are those past the `σ.c` the session's slots took, `LoopFacts` holds, the requests to come are read
as the loop answers them, and each reads the session's setup. -/
def LoopP (sid : Nat) (spec : SessionSpec) (sts : Array Stmt) (nonce : ByteArray) (ms₀ : List Claims.Msg)
    (seen : List String) (st : State) (pl : Placed) (es : List Env) (ms : List Claims.Msg) : Prop :=
  ∃ σ : LoopAt, ms = ms₀.drop σ.c ∧ LoopFacts sid spec seen st σ ∧ LoopReads sid spec sts nonce σ st pl es ∧
    ∀ e ∈ es, e.setup = .ok (spec, sts)

/-- **`envPre_inv`'s step at the slots of kind `k`**, at the invariant `P`: at each request at such a slot where `P`
holds, what the relay read is the model's next message if the slot reads one, and `P` holds at the next request where
the session goes on. -/
def SlotCase (spec : SessionSpec) (P : List String → State → Placed → List Env → List Claims.Msg → Prop)
    (k : Session.Kind) : Prop :=
  ∀ seen st pl e rest ms, slot st pl = some k → P seen st pl (e :: rest) ms →
    (isModel k = true → EnvOk spec seen k e (ms.headD [])) ∧
    (((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
        (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true →
      P (seenAfter seen k) (stepK true st pl (some (tagOf k)) (.ok k) e).st
        (stepK true st pl (some (tagOf k)) (.ok k) e).pl rest (if isModel k then ms.tail else ms))

/-- **Each request but a `Hello` keeps whether the session's `Hello` left.** -/
theorem stepK_sent {hold : Bool} {st : State} {pl : Placed} {b : Option UInt8} {k : Except String Session.Kind}
    {e : Env} (hk : k ≠ .ok .hello) : (stepK hold st pl b k e).st.sent = st.sent := by
  unfold stepK
  split
  · rfl
  split
  · rfl
  split
  · rfl
  split
  · rfl
  · exact absurd rfl hk
  · split
    · exact (request_keeps ‹_›).2.2
    · split
      · rfl
      · split
        · rfl
        · exact (afterAnswer_keeps _ _ _ _).2.2

/-- **The session goes on past its `Hello` only with the `Hello` sent**: a refused `Hello` stops it, and one with no
answer ends the relay. -/
theorem stepK_hello_sent {st : State} {pl : Placed} {e : Env}
    (hgo : ((stepK true st pl (some (tagOf .hello)) (.ok .hello) e).go &&
      (stepK true st pl (some (tagOf .hello)) (.ok .hello) e).st.stop.isNone) = true) :
    (stepK true st pl (some (tagOf .hello)) (.ok .hello) e).st.sent = true := by
  rcases hr : st.request pl .hello with _ | ⟨st', o⟩
  · have hs : st.stop = none := by
      revert hr
      unfold State.request
      split
      · intro h
        cases h
      · intro _
        assumption
    rw [stepK_hello hr] at hgo ⊢
    rcases hh : st.sendHello e.setup e.nonce with ⟨st', o⟩ | ⟨st', o⟩
    · rw [hh] at hgo
      have hst := sendHello_error hs hh
      simp only [Bool.true_and] at hgo
      rw [Option.isSome_iff_ne_none] at hst
      exact absurd (Option.isNone_iff_eq_none.mp hgo) hst
    · obtain ⟨_, _, _, hsent, -⟩ := sendHello_ok hh
      rcases ha : e.ans with _ | r
      · rw [hh, ha] at hgo
        simp only [Bool.false_and] at hgo
        cases hgo
      · exact (answer_keeps st' .hello r e.id).2.2.trans hsent
  · rw [stepK_refused (by decide) (by decide) (by decide) hr] at hgo
    have hst : st'.stop.isSome := by
      rcases hs : st.stop with _ | why
      · exact request_stop hs hr
      · revert hr
        unfold State.request
        rw [hs]
        intro h
        cases h
        rw [hs]
        rfl
    simp only [Bool.true_and] at hgo
    rw [Option.isSome_iff_ne_none] at hst
    exact absurd (Option.isNone_iff_eq_none.mp hgo) hst

/-- **At a slot that reads no model message `LoopP` carries over**: where the loop's reading of it brings only the
updates between the session's requests and takes no model message, the request keeps the session's setup, the streams
seen, whether its `Hello` left and the model's messages to come, and the updates keep each unseen rep's frame absent or
fresh. -/
theorem slotCase_carried {sid : Nat} {spec : SessionSpec} {sts : Array Stmt} {nonce : ByteArray}
    {ms₀ : List Claims.Msg} {k : Session.Kind} (hm : isModel k = false) (hs : ∀ seen, seenAfter seen k = seen)
    (hr : ∀ st pl e σ σ', LoopRead sid spec sts nonce k st pl e σ σ' → Between sid σ.R σ'.R ∧ σ'.c = σ.c) :
    SlotCase spec (LoopP sid spec sts nonce ms₀) k := by
  intro seen st pl e rest ms hsl hP
  obtain ⟨σ, hms, hF, hRd, hset⟩ := hP
  refine ⟨fun h => absurd (hm.symm.trans h) Bool.false_ne_true, fun hgo => ?_⟩
  cases hRd with
  | stop h =>
    rw [hsl] at h
    cases h
  | step k' hk' hra hrest =>
    obtain rfl : k = k' := Option.some.inj (hsl.symm.trans hk')
    obtain ⟨hbt, hcc⟩ := hr _ _ _ _ _ hra
    have hk : (Except.ok k : Except String Session.Kind) ≠ .ok .hello := by
      intro h
      cases h
      cases hm
    refine ⟨_, ?_, ⟨?_, ?_, ?_⟩, hrest hgo, fun e' he' => hset e' (List.mem_cons_of_mem _ he')⟩
    · simp only [hm, Bool.false_eq_true, ↓reduceIte]
      rw [hcc]
      exact hms
    · exact stepK_setup (hset e List.mem_cons_self) hF.setup
    · intro i r s hmem hseen
      rw [hs] at hseen
      exact freshAt_kept (hbt.1 i r) (hF.fresh i r s hmem hseen)
    · intro h
      rw [stepK_sent hk] at h
      rw [hcc]
      exact hF.start h

/-- **The `Hello`'s slot case**: where the model's first message is the `Hello` at the nonce the relay drew, what the
relay read at the `Hello` is the model's message, and `LoopP` holds at the next request with that message taken. -/
theorem slotCase_hello {sid : Nat} {spec : SessionSpec} {sts : Array Stmt} {nonce : ByteArray}
    {ms₀ : List Claims.Msg} (h0 : ms₀.headD [] = _root_.Flock.Firewall.Hello.msg spec nonce) :
    SlotCase spec (LoopP sid spec sts nonce ms₀) .hello := by
  intro seen st pl e rest ms hsl hP
  obtain ⟨σ, hms, hF, hRd, hset⟩ := hP
  have hc0 : σ.c = 0 := hF.start (slot_hello hsl).2
  cases hRd with
  | stop h =>
    rw [hsl] at h
    cases h
  | step k' hk' hra hrest =>
    obtain rfl : Session.Kind.hello = k' := Option.some.inj (hsl.symm.trans hk')
    obtain ⟨hn, hbt, hcc⟩ := hra
    refine ⟨fun _ => ?_, fun hgo => ?_⟩
    · show _root_.Flock.Firewall.Hello.msg spec e.nonce = ms.headD []
      rw [hn, hms, hc0, List.drop_zero, h0]
    · refine ⟨_, ?_, ⟨?_, ?_, ?_⟩, hrest hgo, fun e' he' => hset e' (List.mem_cons_of_mem _ he')⟩
      · simp only [isModel, ↓reduceIte]
        rw [hcc, hms, List.tail_drop]
      · exact stepK_setup (hset e List.mem_cons_self) hF.setup
      · intro i r s hmem hseen
        exact freshAt_kept (hbt.1 i r) (hF.fresh i r s hmem hseen)
      · intro h
        rw [stepK_hello_sent hgo] at h
        cases h

/-- **`EnvPre` from the relay's invariant at the deployed loop**: given `envPre_inv`'s step at the `Hello`, the
`Commit`, each frame round and the `Finish`, `EnvPre` holds wherever `LoopP` does; the step at every other slot is
`slotCase_carried`'s. -/
theorem envPre_loop {sid : Nat} {spec : SessionSpec} {sts : Array Stmt} {nonce : ByteArray} {ms₀ : List Claims.Msg}
    (hHello : SlotCase spec (LoopP sid spec sts nonce ms₀) .hello)
    (hCommit : SlotCase spec (LoopP sid spec sts nonce ms₀) .commit)
    (hRound : ∀ stream n, SlotCase spec (LoopP sid spec sts nonce ms₀) (.round stream n))
    (hFinish : SlotCase spec (LoopP sid spec sts nonce ms₀) .finish) :
    ∀ (es : List Env) (seen : List String) (st : State) (pl : Placed) (ms : List Claims.Msg),
      LoopP sid spec sts nonce ms₀ seen st pl es ms → EnvPre spec seen (schedSteps true st pl es) ms :=
  envPre_inv (LoopP sid spec sts nonce ms₀) fun seen st pl e rest ms k => by
    have hc : ∀ k : Session.Kind, isModel k = false → (∀ seen, seenAfter seen k = seen) →
        (∀ st pl e σ σ', LoopRead sid spec sts nonce k st pl e σ σ' → Between sid σ.R σ'.R ∧ σ'.c = σ.c) →
        SlotCase spec (LoopP sid spec sts nonce ms₀) k := fun _ hm hs hr => slotCase_carried hm hs hr
    cases k with
    | hello => exact hHello seen st pl e rest ms
    | commit => exact hCommit seen st pl e rest ms
    | round stream n => exact hRound stream n seen st pl e rest ms
    | finish => exact hFinish seen st pl e rest ms
    | register => exact hc .register rfl (fun _ => rfl) (fun _ _ _ _ _ h => h) seen st pl e rest ms
    | link => exact hc .link rfl (fun _ => rfl) (fun _ _ _ _ _ h => h) seen st pl e rest ms
    | openStream s => exact hc (.openStream s) rfl (fun _ => rfl) (fun _ _ _ _ _ h => h) seen st pl e rest ms
    | proof => exact hc .proof rfl (fun _ => rfl) (fun _ _ _ _ _ h => h) seen st pl e rest ms
    | ping => exact hc .ping rfl (fun _ => rfl) (fun _ _ _ _ _ h => h) seen st pl e rest ms
    | other => exact hc .other rfl (fun _ => rfl) (fun _ _ _ _ _ h => h) seen st pl e rest ms

end Claims.ZKServer.Relay
