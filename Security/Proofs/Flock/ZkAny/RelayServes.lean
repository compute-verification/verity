module

public import Proofs.Flock.ZkAny.RelayLoop

/-!
# The relay's run as its IO code serves it

`LoopReads` reads the relay's requests as the loop answers them. Most of its clauses say what the relay's IO code did at
a slot; two couple the run to the model's: the auditor answering as `V*` did (`LoopModel.Ans`, and at the `Finish` the
model's run having a message there), and the model's worker giving the worker's round values. Here they stand apart:
the IO code's readings are the named trust `flock-relay-serves`, the worker's the named trust
`flock-worker-deterministic`, and the auditor's a statement about `V*`:

* `modelSlots`: the requests of the relay's run at its model slots (`isModel`), each with its slot's kind, in order, up
  to the session's stop or end, as `LoopReads` walks them.
* `LoopServe`: `LoopRead` without the couplings: a frame round runs on the worker's round value at the model's history.
  `LoopServe.count_model`, `LoopServe.count_ctl`: each model slot takes one of the model's messages, and no other slot
  does.
* `ReadsBy`: `LoopReads`' walk over the requests at any reading of a slot. `AudReads`: the relay's answers are the
  auditor's, to everything that left so far.
* **`RelayServes`**: the relay's requests from its start, each served so (`ReadsBy` at `LoopServe`), with the
  auditor's answers (`AudReads`).
* **`WorkerDeterministic`**: the model's worker gives the worker's round values, a function of the history.
* `AnsPart`: what `LoopRead` asks of a slot beyond `LoopServe`'s clauses and the worker's values; `AuditorAnswers`: at
  the relay's `j`-th model slot, `AnsPart` at the model's `j`-th message.
* `loopRead_of_serve`, **`loopReads_of_serves`**, `RelayServes.loopReads`: so the requests are read as the loop answers
  them (`LoopReads`), where the model's worker answers the `Finish` with the firewall's own message.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session
open _root_.Flock.Firewall hiding State
open _root_.Flock (SessionSpec Stmt)

/-- **The requests of the relay's run at its model slots**, from the session `st` and placement `pl`: of the requests
`es`, each at a `Hello`, a coin request or the `Finish` (`isModel`), with its slot's kind, in order, up to the session's
stop or end, as `LoopReads` walks them. -/
def modelSlots : State → Placed → List Env → List (Session.Kind × Env)
  | _, _, [] => []
  | st, pl, e :: rest =>
    match slot st pl with
    | none => []
    | some k =>
      (if isModel k then [(k, e)] else []) ++
        if ((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
            (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true then
          modelSlots (stepK true st pl (some (tagOf k)) (.ok k) e).st
            (stepK true st pl (some (tagOf k)) (.ok k) e).pl rest
        else []

/-- **What the relay's IO code does at a slot of kind `k` of session `sid`**: `LoopRead`'s clauses but the run's
couplings to the model's (the auditor's answers, at the `Finish` the run having a message, and the worker's values), a
frame round running on the worker's round value `rv` at the model's history. -/
def LoopServe {W Rr : Type} (M : LoopModel W Rr) (rv : List (Claims.Msg × Claims.Msg) → Option Claims.Msg)
    (sid : Nat) (spec : SessionSpec) (sts : Array Stmt) (nonce : ByteArray) :
    Session.Kind → State → Placed → Env → LoopAt → LoopAt → Prop
  | .hello, _, _, e, σ, σ' =>
    e.nonce = nonce ∧ Between sid σ.R σ'.R ∧ σ'.c = σ.c + 1 ∧
      M.L.toSrv.liveBefore M.wk M.x M.w M.ρ M.κ (M.hist σ'.c) = true ∧
      (e.setup = .ok (spec, sts) → SessOn spec sts σ'.R sid)
  | .commit, st, _, e, σ, σ' =>
    (∃ f, e.commit = (commitRead σ.L0 sid (spec.tables.zip (sts.map (·.m))) f).1 ∧
        ∀ name, _root_.Flock.Firewall.Level0.ofSession σ'.L0 sid name =
          _root_.Flock.Firewall.Level0.ofSession (commitRead σ.L0 sid (spec.tables.zip (sts.map (·.m))) f).2 sid name) ∧
      (∃ ct, _root_.Flock.Firewall.Session.streamsOk spec sts ct = true) ∧
      (∃ (ls : Array _root_.Flock.Firewall.Level0.Level0) (publics : List ByteArray), AfterDraw sid ls σ.L0 ∧
        (∀ j (hj : j < ls.size), ∃ g d hi, M.src (2 * j) = .ok (g, d, hi) ∧
          _root_.Flock.Firewall.Level0.ofDraws sts[j]!.m g d = .ok ls[j]) ∧
        M.L.ownMsg M.x M.w M.ρ (M.hist σ.c) = 0 :: FlockDeploy.linkBody (ls.toList.map (·.root)) publics ∧
        ls.size < 2 ^ 64 ∧ publics.length < 2 ^ 64 ∧ (∀ l ∈ ls, l.root.size < 2 ^ 64) ∧
        ∀ b ∈ publics, b.size < 2 ^ 64) ∧
      (∀ a, e.ans = some a → byte? a 0 = some 2 → (st.answer .commit a).1.stop = none →
        ∀ rel, sessAt σ'.R sid = some rel →
          rel.link = some (_root_.Flock.Firewall.Frame.words (st.answer .commit a).2 5
            (((st.answer .commit a).2.size - 5) / 16))) ∧
      Between sid σ.R σ'.R ∧ σ'.c = σ.c + 1
  | .round stream n, st, pl, e, σ, σ' =>
    (∃ i r dom, ∃ v : Claims.Msg, rv (M.hist σ.c) = some v ∧
      RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R ∧
      (∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) ∧ InnerSrc M sid i r σ ∧
      StartOk sts sid i r σ) ∧
      σ'.c = σ.c + 1 ∧ Stays sid σ σ' ∧ SaltsSrc M sid σ
  | .finish, _, _, e, σ, σ' =>
    (e.barrier none = none → σ.R.stopped sid = false ∧ σ.c = M.L.last M.x ∧
        M.L.finStops M.x M.w M.ρ (M.hist σ.c) = false ∧
        ∃ f : FlockDeploy.Final, M.L.ownMsg M.x M.w M.ρ (M.hist σ.c) = f.encode ∧ e.proofs = proofsOf spec f.proofs ∧
          f.stmt.length < 2 ^ 64 ∧ f.record.toUTF8.size < 2 ^ 64 ∧ f.proofs.size < 2 ^ 64 ∧
          ∀ b ∈ f.proofs, b.size < 2 ^ 64) ∧
      σ'.c = σ.c + 1
  | _, _, _, _, σ, σ' => Between sid σ.R σ'.R ∧ σ'.c = σ.c ∧ Stays sid σ σ'

/-- **The relay's requests `es` to come, each read by `Rd`**, from the session `st`, placement `pl` and the loop at
`σ`: `LoopReads`' walk at the reading `Rd` of a slot, the rest so from the loop at the session's next request where the
session goes on; nothing past the session's end. -/
inductive ReadsBy (Rd : Session.Kind → State → Placed → Env → LoopAt → LoopAt → Prop) :
    LoopAt → State → Placed → List Env → Prop
  | nil (σ : LoopAt) (st : State) (pl : Placed) : ReadsBy Rd σ st pl []
  | stop {σ : LoopAt} {st : State} {pl : Placed} {e : Env} {rest : List Env} :
      slot st pl = none → ReadsBy Rd σ st pl (e :: rest)
  | step {σ σ' : LoopAt} {st : State} {pl : Placed} {e : Env} {rest : List Env} (k : Session.Kind) :
      slot st pl = some k → Rd k st pl e σ σ' →
      (((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
          (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true →
        ReadsBy Rd σ' (stepK true st pl (some (tagOf k)) (.ok k) e).st
          (stepK true st pl (some (tagOf k)) (.ok k) e).pl rest) →
      ReadsBy Rd σ st pl (e :: rest)

/-- **The relay's answers are the auditor's** `aud`, the frames that left so far being `v`: at the `Hello`, the
`Commit` and each frame round, its answer to everything that left through the slot, and none where nothing left
there; and so at each slot to come, up to the session's stop or end. -/
def AudReads (aud : List ByteArray → Option ByteArray) : List ByteArray → State → Placed → List Env → Prop
  | _, _, _, [] => True
  | v, st, pl, e :: rest => ∀ k, slot st pl = some k →
    (isModel k = true → k ≠ .finish →
      e.ans = if (stepK true st pl (some (tagOf k)) (.ok k) e).sent.isEmpty then none
        else aud (v ++ (stepK true st pl (some (tagOf k)) (.ok k) e).sent)) ∧
    (((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
        (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true →
      AudReads aud (v ++ (stepK true st pl (some (tagOf k)) (.ok k) e).sent)
        (stepK true st pl (some (tagOf k)) (.ok k) e).st (stepK true st pl (some (tagOf k)) (.ok k) e).pl rest)

/-- `flock-relay-serves` (a trust in the relay's IO code, which Lean reads but does not run). **The relay serves
session `sid`'s requests `ins` as its IO code reads**, from its start with the loop at `σ`: at each request that
reaches a slot, `LoopServe`'s clauses (`ReadsBy`), and the answers it reads are the auditor `aud`'s to everything that
left (`AudReads`). They read `Flock.Firewall.Session.relay`'s loop body (`stepK`, held to it by reading), the frames
it puts on the auditor's connection (`sendFrame a`) and the answers it reads there (`recvFrame a`), the rep frames'
tasks it calls on the shared frames (`Frames.round` on the request's `roundContent`, `Frames.coins` on the auditor's
coins, `Frames.finish`; their pure updates `RegStep`, `RoundRun`), the `Commit`'s `Level0.commit` against the level
0s `Session.level0` drew (`Level0.draw`), the `Link`'s put, the loop's own proofs (`ownProofs`) and the updates of the
loop's other requests between the session's (`Between`, `Stays`), with the rep frames' starts (`Frames.startRep`,
`StartOk`), the rounds of the inner proof the loop's finish computes (`Sessions.innerRound`, `InnerSrc`) and the salts
`Finish.serve` keeps (`SaltsSrc`), each against the model's run `M`; a frame round's content being the worker's round
value `rv` at the model's history (`WorkerDeterministic`). Why it is reasonable: each clause is one IO function's
effect on the loop's references or the auditor's connection, stated with the pure functions the function calls. -/
def RelayServes {W Rr : Type} (M : LoopModel W Rr) (rv : List (Claims.Msg × Claims.Msg) → Option Claims.Msg)
    (aud : List ByteArray → Option ByteArray) (sid : Nat) (spec : SessionSpec) (sts : Array Stmt) (nonce : ByteArray)
    (σ : LoopAt) (ins : List (ByteArray × Env)) : Prop :=
  ReadsBy (LoopServe M rv sid spec sts nonce) σ {} {} (pick true {} {} ins) ∧
    AudReads aud [] {} {} (pick true {} {} ins)

/-- `flock-worker-deterministic` (a trust in the worker's code). **The model's worker gives the worker's round
values**: at each history `h` where the worker sends a frame round's content `v` (`rv h = some v`: the content
`LiveChallenger::round` in `verity/ml/flock/live/src/lib.rs` sends in its `Req::Round`, everything the prover absorbed
since its previous squeeze), the model's worker gives `v` there and goes on. Why it is reasonable: the worker is the
honest prover behind the ZK server, whose round contents its code computes from the statement, the witness, its tapes
and the coins it received, the history. -/
def WorkerDeterministic {W Rr : Type} (M : LoopModel W Rr)
    (rv : List (Claims.Msg × Claims.Msg) → Option Claims.Msg) : Prop :=
  ∀ h v, rv h = some v → M.wk M.x M.w M.ρ h 0 = some (v, false)

/-- **What `LoopRead` asks of a slot of kind `k` beyond what the IO code serves and the worker's values**, with the
loop at the model's `c`-th message: at a `Hello`, the `Commit` or a coin request, that the auditor answered `e` as `V*`
did (`LoopModel.Ans`); at the `Finish`, that the model's run has a `c`-th message; nothing at a control slot. -/
def AnsPart {W Rr : Type} (M : LoopModel W Rr) (c : Nat) (e : Env) : Session.Kind → Prop
  | .hello | .commit | .round _ _ => M.Ans c e
  | .finish => c < M.τ.length
  | _ => True

/-- **The auditor answered the relay's model slots as `V*` did**: at the `j`-th of the model slots `sl`, `AnsPart` at
the model's `j`-th message. -/
def AuditorAnswers {W Rr : Type} (M : LoopModel W Rr) (sl : List (Session.Kind × Env)) : Prop :=
  ∀ j k e, sl[j]? = some (k, e) → AnsPart M j e k

/-- **A control slot asks nothing beyond what the IO code serves.** -/
theorem AnsPart.of_ctl {W Rr : Type} {M : LoopModel W Rr} {c : Nat} {e : Env} {k : Session.Kind}
    (hm : isModel k = false) : AnsPart M c e k := by
  cases k
  case hello | commit | round | finish => exact absurd hm (by decide : ¬(true = false))
  all_goals exact True.intro

/-- **A model slot takes one of the model's messages.** -/
theorem LoopServe.count_model {W Rr : Type} {M : LoopModel W Rr}
    {rv : List (Claims.Msg × Claims.Msg) → Option Claims.Msg} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {nonce : ByteArray} {k : Session.Kind} {st : State} {pl : Placed} {e : Env} {σ σ' : LoopAt}
    (h : LoopServe M rv sid spec sts nonce k st pl e σ σ') (hm : isModel k = true) : σ'.c = σ.c + 1 := by
  cases k
  case hello =>
    obtain ⟨-, -, hc, -, -⟩ := h
    exact hc
  case commit =>
    obtain ⟨-, -, -, -, -, hc⟩ := h
    exact hc
  case round =>
    obtain ⟨-, hc, -, -⟩ := h
    exact hc
  case finish =>
    obtain ⟨-, hc⟩ := h
    exact hc
  all_goals exact absurd hm (by decide : ¬(false = true))

/-- **A control slot takes none of the model's messages.** -/
theorem LoopServe.count_ctl {W Rr : Type} {M : LoopModel W Rr}
    {rv : List (Claims.Msg × Claims.Msg) → Option Claims.Msg} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {nonce : ByteArray} {k : Session.Kind} {st : State} {pl : Placed} {e : Env} {σ σ' : LoopAt}
    (h : LoopServe M rv sid spec sts nonce k st pl e σ σ') (hm : isModel k = false) : σ'.c = σ.c := by
  cases k
  case hello | commit | round | finish => exact absurd hm (by decide : ¬(true = false))
  all_goals
    obtain ⟨-, hc, -⟩ := h
    exact hc

/-- **What the IO code serves at a slot is what the relay reads there**, where the slot's `AnsPart` holds (`hA`), the
model's worker gives the worker's round values (`hw`), and it answers the `Finish` with the firewall's own message
(`hfw`). -/
theorem loopRead_of_serve {W Rr : Type} {M : LoopModel W Rr}
    {rv : List (Claims.Msg × Claims.Msg) → Option Claims.Msg} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {nonce : ByteArray} {k : Session.Kind} {st : State} {pl : Placed} {e : Env} {σ σ' : LoopAt}
    (hs : LoopServe M rv sid spec sts nonce k st pl e σ σ') (hw : WorkerDeterministic M rv)
    (hfw : ∀ h : List (Claims.Msg × Claims.Msg), h.length = M.L.last M.x →
      M.wk M.x M.w M.ρ h 0 = some (M.L.ownMsg M.x M.w M.ρ h, false))
    (hA : AnsPart M σ.c e k) :
    LoopRead M sid spec sts nonce k st pl e σ σ' := by
  cases k
  case hello => exact ⟨hA, hs⟩
  case commit => exact ⟨hA, hs⟩
  case round stream n =>
    obtain ⟨⟨i, r, dom, v, hv, hrr, hcf, hin, hsk⟩, hc, hst, hsa⟩ := hs
    exact ⟨hA, ⟨i, r, dom, v, hw _ _ hv, hrr, hcf, hin, hsk⟩, hc, hst, hsa⟩
  case finish =>
    obtain ⟨hb, hc⟩ := hs
    have hlt : σ.c < M.τ.length := hA
    refine ⟨hlt, fun hbn => ?_, hc⟩
    obtain ⟨hst, hl, hfs, hf⟩ := hb hbn
    refine ⟨hst, hl, hfs, hfw _ ?_, hf⟩
    rw [LoopModel.hist, length_histOf_take, Nat.min_eq_left hlt.le]
    exact hl
  all_goals exact hs

/-- **The requests the IO code serves are read as the loop answers them**: at `ReadsBy` at `LoopServe` from the loop
at `σ`, where at the `j`-th of its model slots (`modelSlots`) `AnsPart` holds at the model's `σ.c + j`-th message, the
model's worker gives the worker's round values (`hw`) and answers the `Finish` with the firewall's own message
(`hfw`): `LoopReads`, each model slot taking the model's next message (`LoopServe.count_model`,
`LoopServe.count_ctl`). -/
theorem loopReads_of_serves {W Rr : Type} {M : LoopModel W Rr}
    {rv : List (Claims.Msg × Claims.Msg) → Option Claims.Msg} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {nonce : ByteArray} (hw : WorkerDeterministic M rv)
    (hfw : ∀ h : List (Claims.Msg × Claims.Msg), h.length = M.L.last M.x →
      M.wk M.x M.w M.ρ h 0 = some (M.L.ownMsg M.x M.w M.ρ h, false))
    {σ : LoopAt} {st : State} {pl : Placed} {es : List Env}
    (h : ReadsBy (LoopServe M rv sid spec sts nonce) σ st pl es) :
    (∀ j k e, (modelSlots st pl es)[j]? = some (k, e) → AnsPart M (σ.c + j) e k) →
      LoopReads M sid spec sts nonce σ st pl es := by
  induction h with
  | nil σ st pl => exact fun _ => .nil σ st pl
  | stop hs => exact fun _ => .stop hs
  | @step σ σ' st pl e rest k hk hs _ ih =>
    intro hA
    have hms : modelSlots st pl (e :: rest) = (if isModel k then [(k, e)] else []) ++
        if ((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
            (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true then
          modelSlots (stepK true st pl (some (tagOf k)) (.ok k) e).st
            (stepK true st pl (some (tagOf k)) (.ok k) e).pl rest
        else [] := by
      simp only [modelSlots, hk]
    refine .step k hk (loopRead_of_serve hs hw hfw ?_) fun hgo => ih hgo fun j k' e' hj => ?_
    · rcases hm : isModel k with _ | _
      · exact AnsPart.of_ctl hm
      · exact hA 0 k e (by rw [hms, ite_eq_left hm]; simp)
    · rw [ite_eq_left hgo] at hms
      rcases hm : isModel k with _ | _
      · rw [ite_eq_right (show ¬(isModel k = true) by rw [hm]; decide), List.nil_append] at hms
        rw [hs.count_ctl hm]
        exact hA j k' e' (by rw [hms]; exact hj)
      · rw [ite_eq_left hm] at hms
        rw [hs.count_model hm, show σ.c + 1 + j = σ.c + (j + 1) by omega]
        exact hA (j + 1) k' e' (by rw [hms]; simpa using hj)

/-- **The relay's requests, served from its start, are read as the loop answers them**: at `RelayServes` from the loop
at `σ`, whose count of the model's messages is `0`, where `AnsPart` holds at the relay's model slots
(`AuditorAnswers`), the model's worker gives the worker's round values (`WorkerDeterministic`) and answers the
`Finish` with the firewall's own message (`hfw`). -/
theorem RelayServes.loopReads {W Rr : Type} {M : LoopModel W Rr}
    {rv : List (Claims.Msg × Claims.Msg) → Option Claims.Msg} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {nonce : ByteArray} {σ : LoopAt} {ins : List (ByteArray × Env)} {aud : List ByteArray → Option ByteArray}
    (hR : RelayServes M rv aud sid spec sts nonce σ ins) (hw : WorkerDeterministic M rv)
    (hfw : ∀ h : List (Claims.Msg × Claims.Msg), h.length = M.L.last M.x →
      M.wk M.x M.w M.ρ h 0 = some (M.L.ownMsg M.x M.w M.ρ h, false))
    (hc : σ.c = 0) (hA : AuditorAnswers M (modelSlots {} {} (pick true {} {} ins))) :
    LoopReads M sid spec sts nonce σ {} {} (pick true {} {} ins) :=
  loopReads_of_serves hw hfw hR.1 fun j k e h => by
    rw [hc, Nat.zero_add]
    exact hA j k e h

end Claims.ZKServer.Relay
