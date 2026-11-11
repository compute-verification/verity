module

public import Proofs.Flock.ZkAny.RelayInv
public import Proofs.Flock.ZkAny.RelayEnv
public import Proofs.Flock.ZkAny.RelayFresh

/-!
# The relay's requests as the deployed loop answers them

`envPre_inv` gives `EnvPre` from any invariant of the relay's requests that each slot's step keeps. Here is the
invariant at the deployed loop, against the model's run, with the cases of the slots that read no model message and
the `Hello`'s proved:

* `LoopModel`: the model's run the loop serves: the deferred release, its worker, the statement, witness and tapes,
  the run's messages, each with `V*`'s answer, what the host and the operating system hand the loop, and the rounds of
  the inner proof its finish computes.
  `LoopModel.msgs`, `LoopModel.hist`: the run's messages, and its history before the `j`-th. `LoopModel.Own`: each
  message the deferred release's at the history before it.
  `LoopModel.Ans`: the run has a `j`-th message, and where the auditor answered coins to it, `V*` answered their bytes.
* `LoopAt`: the loop between two of the session's requests: its frames, its level-0 slots and how many of the model's
  messages the session's slots took.
* `Stays`: from one of the session's requests to the next, its tables' level 0 as they were, and at each bound level
  0's root its entry and salts (`Stay`).
* `SaltsSrc`: the salts for levels ≥ 1 the loop holds at the session's level-0 roots the ones the host and the operating
  system handed the loop for the rep stream.
* `InnerSrc`: the rounds of the inner proof the loop's finish computed so far for a rep the model's finish's.
* `StartOk`: a rep has a frame only where its table's statement starts (`Drive.start`).
* `SessOn`: the session's entry set up on the setup. `sessOn_sessKept`: the updates that keep the session's stop keep
  it.
* `LoopRead`: what the relay reads at a slot as the loop answers it, and the loop at the session's next request: the
  `Hello`'s nonce the relay drew, the model's `Hello` round passing and the session's entry on the setup (`SessOn`);
  the `Commit`'s roots `commitRead` reads off
  the slots, the level 0s it binds kept, the coin tree's streams the session's, the ZK server's own link at the
  level 0s the loop drew for the session, each `Level0.ofDraws` on the host's and the operating system's bytes, and
  the session's entry linked to the words of the coins the prover receives (`Frame.words`); a
  frame round as the loop ran it (`RoundRun`) on the worker's value at the model's history, the rep's frame
  at the model's configuration there, its rounds of the inner proof `InnerSrc`, its start `StartOk` and the loop's
  salts `SaltsSrc`; at the
  `Finish`, once the hold's barrier passes, the session not stopped, the run at its last round, the ZK server's own
  `Finish` not stopping, the worker's value the ZK server's own, and that message the `Final` of the loop's proof files
  as `Proof` frames; at the `Hello`, the `Commit` and each frame round the auditor answering as `V*` did, and at the
  `Finish` the run having a message there; at every other slot only the updates between the session's requests
  (`Between`); and at a frame round and every other slot `Stays`.
* `LoopReads`: the requests to come, each read so, up to the session's stop or end.
* `LoopFacts`: the session going on (`Reach`, no stop), its Σ, domain frames, reps and labels the setup's, each rep
  whose stream had no round yet with its frame absent or fresh (`FreshAt`), and the model's messages the slots took:
  none before the `Hello` left, and from there one more than the openings the auditor answered.
* `LoopKept`: an invariant of the loop's frames and the session kept at the slots of a kind, at each request that
  reads the session's setup.
* `LoopP`: the model's messages to come those past the ones the slots took, `LoopFacts`, an invariant `J`, `LoopReads`
  from there, and every request reading the session's setup.
* `stepK_sent`: each request but a `Hello` keeps whether the `Hello` left. `stepK_hello_sent`: the session goes on past
  its `Hello` only with it sent. `stepK_reach`: `Reach` at the session's next request. `stepK_ctl_session`: a request
  that reads no model message keeps the session's coins. `stepK_hello_opened`: past its `Hello` the session opened
  nothing. `stepK_coin_opened`: past a `Commit` or a frame round one more opening.
* `SlotCase`: `envPre_inv`'s step at the slots of one kind. `slotCase_carried`: at a slot that reads no model message,
  `LoopP` carries over. `slotCase_hello`: the `Hello`'s, where the model's first message is the `Hello` at the sampled
  nonce.
* **`envPre_loop`**: `EnvPre` from `LoopP`, given `SlotCase` at the `Hello`, the `Commit`, each frame round and the
  `Finish`, and `J` kept at every other slot.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session
open _root_.Flock.Firewall hiding State
open _root_.Flock (SessionSpec Stmt)
open _root_.Flock.Firewall.Frames (Sessions)

/-- **The model's run the loop serves**: the deferred release `L` at `nN` messages, with the worker `wk`, on the
statement `x`, witness `w` and tapes `ρ`, `κ`, the run's messages, each with `V*`'s answer (`τ`), and what the loop's
host and operating system hand it for each rep stream (`src`, in `Loop.Src`'s form): the packed witness and the bytes
`Level0.draw` reads, and the salts for levels ≥ 1 `Finish.serve` reads; and at each history the rounds of the inner
proof the loop's finish computes for the rep stream the history locates (`fin`, as `LeanRelease.withInner` reads
them). -/
structure LoopModel (W Rr : Type) where
  L : LeanRelease W Rr
  nN : Claims.Msg → Nat
  wk : L.toSrv.Worker
  x : Claims.Msg
  w : W
  ρ : Rr
  κ : L.toSrv.Chk
  τ : List (Claims.Msg × Claims.Out)
  src : Nat → _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)
  fin : List (Claims.Msg × Claims.Msg) → Array ByteArray

namespace LoopModel

variable {W Rr : Type} (M : LoopModel W Rr)

/-- The run's messages. -/
def msgs : List Claims.Msg := M.τ.map (·.1)

open Claims.ZKMal in
/-- The run's history before its `j`-th message. -/
def hist (j : Nat) : List (Claims.Msg × Claims.Msg) := histOf (M.τ.take j)

/-- **Each of the run's messages is the deferred release's** at the history before it. -/
def Own : Prop :=
  ∀ j (hj : j < M.τ.length), (M.τ[j]'hj).1 = M.L.toSrv.nextD M.L.fillOf M.nN M.wk M.x M.w M.ρ M.κ (M.hist j)

/-- **The auditor answered the run's `j`-th message as `V*` did**: the run has a `j`-th message, and where the
auditor's answer `a` to it, read from the request `e`, is coins, `V*`'s answer is `a`'s bytes. -/
def Ans (j : Nat) (e : Env) : Prop :=
  ∃ hj : j < M.τ.length, ∀ a, e.ans = some a → byte? a 0 = some 2 → (M.τ[j]'hj).2 = .msg a.data.toList

end LoopModel

/-- **The loop between two of the session's requests**: its frames, its level-0 slots, and how many of the model's
messages the session's slots took so far. -/
structure LoopAt where
  R : Sessions
  L0 : _root_.Flock.Firewall.Level0.Slots
  c : Nat

/-- **From one of session `sid`'s requests, the loop at `σ`, to its next, at `σ'`, what its rounds read of the session
stays**: each table's level 0 as it was (none before the session's `Commit` binds them), and at each bound level 0's
root the session's entry and the salts for levels ≥ 1 (`Stay`: no `link` of the session but its `Commit`'s, and no end
of a session sharing the root). -/
def Stays (sid : Nat) (σ σ' : LoopAt) : Prop :=
  (∀ name, _root_.Flock.Firewall.Level0.ofSession σ'.L0 sid name =
      _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name) ∧
    ∀ name l0, _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 → Stay sid l0.root σ.R σ'.R

/-- **The salts for levels ≥ 1 the loop holds at session `sid`'s level 0s are the ones the host and the operating system
handed it for the rep stream**, at the loop at `σ`: where the loop's salts hold some at table `i`'s level-0 root and rep
`r` (the first a finish kept there, `Finish.serve`'s `salts` handing them to `keep`), they are rep stream `2i + r`'s
(`LoopModel.src`'s, the bytes `Finish.serve`'s `hiSaltsOf` reads). -/
def SaltsSrc {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (σ : LoopAt) : Prop :=
  ∀ (i r : Nat) (rel : _root_.Flock.Firewall.Frames.Release) (name : String)
    (l0 : _root_.Flock.Firewall.Level0.Level0) (r0 : ByteArray) (q0 : Nat) (hi : Array ByteArray), r < 2 →
    sessAt σ.R sid = some rel → rel.spec.tables[i]? = some name →
    _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 →
    σ.R.salts.find? (fun x => x.1 == l0.root && x.2.1 == r) = some (r0, q0, hi) →
    ∃ g d, M.src (2 * i + r) = .ok (g, d, hi)

/-- **The rounds of the inner proof the loop holds for table `i`'s rep `r` of session `sid` are the model's finish's**,
at the loop at `σ`: each round the loop's finish computed so far at the table's level-0 root and the rep
(`Sessions.innerOf`, which `Sessions.innerRound` appends as the finish hands them to the frame) is the round at its
index of the model's finish's rounds at the history so far (`LoopModel.fin`). -/
def InnerSrc {W Rr : Type} (M : LoopModel W Rr) (sid i r : Nat) (σ : LoopAt) : Prop :=
  ∀ (rel : _root_.Flock.Firewall.Frames.Release) (name : String) (l0 : _root_.Flock.Firewall.Level0.Level0),
    sessAt σ.R sid = some rel → rel.spec.tables[i]? = some name →
    _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 →
    ∀ (k : Nat) (b : ByteArray), (σ.R.innerOf l0.root r)[k]? = some b → (M.fin (M.hist σ.c))[k]? = some b

/-- **Table `i`'s rep `r` of session `sid` has a frame only where the table's statement starts**, at the loop at `σ`:
the loop pushes a rep's frame only once `Drive.start` passed at the session's statement for the table
(`Frames.startRep`). -/
def StartOk (sts : Array Stmt) (sid i r : Nat) (σ : LoopAt) : Prop :=
  ∀ p, frameAt σ.R sid i r = some p → ∃ lm, _root_.Flock.Firewall.Drive.start sts[i]! = .ok lm

/-- **Session `sid`'s entry in the loop's frames `R` is set up on `spec` and `sts`**, as its `Hello` began it
(`Sessions.begin`). -/
def SessOn (spec : SessionSpec) (sts : Array Stmt) (R : Sessions) (sid : Nat) : Prop :=
  ∃ rel : _root_.Flock.Firewall.Frames.Release, sessAt R sid = some rel ∧ rel.spec = spec ∧ rel.sts = sts

theorem SessOn.eq {spec : SessionSpec} {sts : Array Stmt} {R : Sessions} {sid : Nat} (h : SessOn spec sts R sid)
    {rel : _root_.Flock.Firewall.Frames.Release} (hr : sessAt R sid = some rel) : rel.spec = spec ∧ rel.sts = sts := by
  obtain ⟨rel', hr', hs, ht⟩ := h
  rw [hr] at hr'
  cases hr'
  exact ⟨hs, ht⟩

/-- **An update that keeps the session's stop keeps its entry's setup**: a `begin` pushes behind it, a `link` sets
only the `Commit`'s coins, and another session's end leaves it. -/
theorem sessOn_sessKeptStep {spec : SessionSpec} {sts : Array Stmt} {sid : Nat} {R R' : Sessions}
    (h : SessKeptStep sid R R') (hs : SessOn spec sts R sid) : SessOn spec sts R' sid := by
  obtain ⟨rel, hr, hsp, hst⟩ := hs
  have hS : (sessAt R sid).isSome := Option.isSome_iff_exists.mpr ⟨rel, hr⟩
  cases h with
  | begin sid' spec' sts' =>
    exact ⟨rel, (sessAt_stayStep (root := ByteArray.empty) (.begin R sid' spec' sts') hS).trans hr, hsp, hst⟩
  | link sid' c =>
    refine ⟨if rel.sid == sid' then { rel with link := some c } else rel, ?_, by split <;> exact hsp,
      by split <;> exact hst⟩
    unfold sessAt at hr ⊢
    show (R.sessions.map fun (s : _root_.Flock.Firewall.Frames.Release) =>
      if s.sid == sid' then { s with link := some c } else s).find? (·.sid == sid) = _
    rw [Array.find?_map]
    have hc : ((·.sid == sid) ∘ fun (s : _root_.Flock.Firewall.Frames.Release) =>
        if s.sid == sid' then { s with link := some c } else s) = (·.sid == sid) := by
      funext s
      simp only [Function.comp]
      split <;> rfl
    rw [hc, hr] <;> rfl
  | start q _ => exact ⟨rel, (sessAt_stayStep (root := ByteArray.empty) (.start R q) hS).trans hr, hsp, hst⟩
  | put p _ => exact ⟨rel, hr, hsp, hst⟩
  | keep root' r' s =>
    exact ⟨rel, (sessAt_stayStep (root := ByteArray.empty) (.keep R root' r' s) hS).trans hr, hsp, hst⟩
  | file root' r' b hf =>
    exact ⟨rel, (sessAt_stayStep (root := ByteArray.empty) (.file root' r' b hf) hS).trans hr, hsp, hst⟩
  | inner root' r' j b hf =>
    exact ⟨rel, (sessAt_stayStep (root := ByteArray.empty) (.inner root' r' j b hf) hS).trans hr, hsp, hst⟩
  | close sid' roots hne =>
    refine ⟨rel, ?_, hsp, hst⟩
    rw [← hr]
    unfold sessAt closeOf
    simp only
    rw [Array.find?_filter]
    congr 1
    funext s
    show decide ((s.sid != sid') = true ∧ (s.sid == sid) = true) = (s.sid == sid)
    rcases hk : s.sid == sid with _ | _
    · simp
    · have hs' : s.sid = sid := by simpa using hk
      have hb : (s.sid != sid') = true := by simp [hs', Ne.symm hne]
      simp [hb]

/-- **Updates that keep the session's stop keep its entry's setup.** -/
theorem sessOn_sessKept {spec : SessionSpec} {sts : Array Stmt} {sid : Nat} {R R' : Sessions}
    (h : SessKept sid R R') (hs : SessOn spec sts R sid) : SessOn spec sts R' sid := by
  induction h with
  | refl => exact hs
  | step _ hst ih => exact sessOn_sessKeptStep hst ih

/-- **What the relay reads at a slot of kind `k` of session `sid`**, from the session `st` and placement `pl`, as the
loop at `σ` answers it serving the model's run `M`, the loop being at `σ'` at the session's next request: at the
`Hello` the nonce `nonce` the relay drew, the model's `Hello` round passing (every round before the history the
slot leaves passed, `liveBefore`), and where the relay's setup is `spec` and `sts` the session's entry set up on them
(`SessOn`: the loop began the session on its setup, at a session id no other session has); at the `Commit` the roots
`commitRead` reads off the loop's level-0 slots at the session's tables (the setup's, each with its statement's `m`),
the session's tables' level 0 at its next request those `commitRead` left, the coin tree's streams the session's
(`streamsOk`), and the ZK server's own message the link at the roots of the level 0s `ls` the loop drew for the session
(`AfterDraw`), table `j`'s being `Level0.ofDraws` at its statement's `m` on what the host and the operating system
handed the loop for rep stream `2j` (`LoopModel.src`, as `Session.level0` samples it), and where the auditor answered
coins the session goes on past, the session's entry at its next request linked to the words of what the prover
receives (`Frame.words` after its tag and count, as the loop links them); at a frame round a `RoundRun` from
`σ`'s frames and level 0s to `σ'`'s frames, its content the worker's value at the model's history so far, the rep's
frame at the model's configuration there (as the loop started it, on the samples the model's tapes hold), the rounds of
the inner proof the loop's finish computed so far for the rep the model's finish's (`InnerSrc`), the rep's frame only
where its table's statement starts (`StartOk`), and the salts the loop holds at the session's level 0s those the host
handed it for each rep stream (`SaltsSrc`); at the `Finish`,
once the hold's barrier passes, the session not stopped, the run at its last round, the ZK server's own `Finish` not
stopping, the worker's value the ZK server's own, and that message the `Final` of the loop's proof files as `Proof`
frames. At the `Hello`, the `Commit` and each frame round the auditor answers as `V*` did (`LoopModel.Ans`), and at
the `Finish` the run has a message. Between the session's requests only the updates of the loop's other requests come
(`Between`), at a frame round and every other slot keeping what the session's rounds read of it (`Stays`), and each
slot that reads a model message takes one. -/
def LoopRead {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (spec : SessionSpec) (sts : Array Stmt)
    (nonce : ByteArray) : Session.Kind → State → Placed → Env → LoopAt → LoopAt → Prop
  | .hello, _, _, e, σ, σ' =>
    M.Ans σ.c e ∧ e.nonce = nonce ∧ Between sid σ.R σ'.R ∧ σ'.c = σ.c + 1 ∧
      M.L.toSrv.liveBefore M.wk M.x M.w M.ρ M.κ (M.hist σ'.c) = true ∧
      (e.setup = .ok (spec, sts) → SessOn spec sts σ'.R sid)
  | .commit, st, _, e, σ, σ' =>
    M.Ans σ.c e ∧ (∃ f, e.commit = (commitRead σ.L0 sid (spec.tables.zip (sts.map (·.m))) f).1 ∧
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
    M.Ans σ.c e ∧ (∃ i r dom, ∃ v : Claims.Msg, M.wk M.x M.w M.ρ (M.hist σ.c) 0 = some (v, false) ∧
      RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R ∧
      (∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) ∧ InnerSrc M sid i r σ ∧
      StartOk sts sid i r σ) ∧
      σ'.c = σ.c + 1 ∧ Stays sid σ σ' ∧ SaltsSrc M sid σ
  | .finish, _, _, e, σ, σ' =>
    σ.c < M.τ.length ∧
      (e.barrier none = none → σ.R.stopped sid = false ∧ σ.c = M.L.last M.x ∧
        M.L.finStops M.x M.w M.ρ (M.hist σ.c) = false ∧
        M.wk M.x M.w M.ρ (M.hist σ.c) 0 = some (M.L.ownMsg M.x M.w M.ρ (M.hist σ.c), false) ∧
        ∃ f : FlockDeploy.Final, M.L.ownMsg M.x M.w M.ρ (M.hist σ.c) = f.encode ∧ e.proofs = proofsOf spec f.proofs ∧
          f.stmt.length < 2 ^ 64 ∧ f.record.toUTF8.size < 2 ^ 64 ∧ f.proofs.size < 2 ^ 64 ∧
          ∀ b ∈ f.proofs, b.size < 2 ^ 64) ∧
      σ'.c = σ.c + 1
  | _, _, _, _, σ, σ' => Between sid σ.R σ'.R ∧ σ'.c = σ.c ∧ Stays sid σ σ'

/-- **The relay's requests `es` to come as the loop answers them**, from the session `st`, placement `pl` and the loop
at `σ`: at each request at a slot, what `LoopRead` reads, and the rest so from the loop at the session's next request
where the session goes on; nothing past the session's end. -/
inductive LoopReads {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (spec : SessionSpec) (sts : Array Stmt)
    (nonce : ByteArray) : LoopAt → State → Placed → List Env → Prop
  | nil (σ : LoopAt) (st : State) (pl : Placed) : LoopReads M sid spec sts nonce σ st pl []
  | stop {σ : LoopAt} {st : State} {pl : Placed} {e : Env} {rest : List Env} :
      slot st pl = none → LoopReads M sid spec sts nonce σ st pl (e :: rest)
  | step {σ σ' : LoopAt} {st : State} {pl : Placed} {e : Env} {rest : List Env} (k : Session.Kind) :
      slot st pl = some k → LoopRead M sid spec sts nonce k st pl e σ σ' →
      (((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
          (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true →
        LoopReads M sid spec sts nonce σ' (stepK true st pl (some (tagOf k)) (.ok k) e).st
          (stepK true st pl (some (tagOf k)) (.ok k) e).pl rest) →
      LoopReads M sid spec sts nonce σ st pl (e :: rest)

/-- **What holds of the session and the loop between its requests**, the streams whose rounds came being `seen`: the
session goes on (`Reach`, no stop), its Σ, domain frames, reps and labels are the setup's once its `Hello` left, each
rep whose stream had no round yet has its frame absent or fresh, and the slots took none of the model's messages before
the `Hello` left, and from there one more than the openings the auditor answered. -/
structure LoopFacts (sid : Nat) (spec : SessionSpec) (seen : List String) (st : State) (pl : Placed) (σ : LoopAt) :
    Prop where
  setup : SetupOk spec st
  fresh : ∀ i r s, (i, r, s) ∈ spec.repStreams.toList → seen.contains s = false → FreshAt σ.R sid i r
  live : st.stop = none
  reach : Reach st pl
  start : st.sent = false → σ.c = 0
  count : st.sent = true → σ.c = opened st + 1

/-- **An invariant `J` of the loop and the session kept at the slots of kind `k`**: at each request at such a slot
that reads the session's setup, where `LoopFacts` and `J` hold and the loop answers it as `LoopRead` reads it, `J`
holds at the session's next request where the session goes on, after the slot's stream. -/
def LoopKept {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (spec : SessionSpec) (sts : Array Stmt)
    (nonce : ByteArray) (J : List String → State → LoopAt → Prop) (k : Session.Kind) : Prop :=
  ∀ seen st pl e σ σ', slot st pl = some k → e.setup = .ok (spec, sts) → LoopFacts sid spec seen st pl σ →
    J seen st σ → LoopRead M sid spec sts nonce k st pl e σ σ' →
    ((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
        (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true →
      J (seenAfter seen k) (stepK true st pl (some (tagOf k)) (.ok k) e).st σ'

/-- **The relay's invariant at the deployed loop**, against the model's run `M`: at some loop `σ`, the model's
messages to come are those past the `σ.c` the session's slots took, `LoopFacts` and the invariant `J` hold, the
requests to come are read as the loop answers them, and each reads the session's setup. -/
def LoopP {W Rr : Type} (M : LoopModel W Rr) (J : List String → State → LoopAt → Prop) (sid : Nat)
    (spec : SessionSpec) (sts : Array Stmt) (nonce : ByteArray) (seen : List String) (st : State) (pl : Placed)
    (es : List Env) (ms : List Claims.Msg) : Prop :=
  ∃ σ : LoopAt, ms = M.msgs.drop σ.c ∧ LoopFacts sid spec seen st pl σ ∧ J seen st σ ∧
    LoopReads M sid spec sts nonce σ st pl es ∧ ∀ e ∈ es, e.setup = .ok (spec, sts)

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

/-- A request the session goes on past let it go on, with no stop. -/
theorem go_live {r : Res} (h : (r.go && r.st.stop.isNone) = true) : r.go = true ∧ r.st.stop = none := by
  simp only [Bool.and_eq_true, Option.isNone_iff_eq_none] at h
  exact h

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
    · exact (request_keeps ‹_›).2.2.1
    · split
      · rfl
      · split
        · rfl
        · exact (afterAnswer_keeps _ _ _ _).2.2.1

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
      · exact (answer_keeps st' .hello r e.id).2.2.1.trans hsent
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

/-- **`Reach` at the session's next request**: a request at the session's slot keeps it where the session goes on. -/
theorem stepK_reach {st : State} {pl : Placed} {k : Session.Kind} {e : Env} (hR : Reach st pl)
    (h : st.stop = none) (hs : slot st pl = some k)
    (hgo : ((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
      (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true) :
    Reach (stepK true st pl (some (tagOf k)) (.ok k) e).st (stepK true st pl (some (tagOf k)) (.ok k) e).pl := by
  obtain ⟨hg, hn⟩ := go_live hgo
  rcases hr : st.request pl k with _ | ⟨st', o⟩
  · rcases reach_step hR h hs hr hg with hst | hR'
    · exact absurd hn (Option.isSome_iff_ne_none.mp hst)
    · exact hR'
  · have hf : k ≠ .finish := by
      rintro rfl
      rw [stepK_finish] at hg
      cases hg
    have hk : k ≠ .ping ∧ k ≠ .proof := by
      rcases slot_kind hs with ⟨h1, h2, -⟩ | rfl
      · exact ⟨h1, h2⟩
      · exact ⟨by simp, by simp⟩
    obtain ⟨h9, h5, h6⟩ := tagOf_ne hk.1 hk.2 hf
    rw [stepK_refused h9 h5 h6 hr] at hn
    exact absurd hn (Option.isSome_iff_ne_none.mp (request_stop h hr))

/-- **A request that reads no model message keeps the session's coins**, where the session goes on: a `Register`'s
answer only its sample, a `Link`'s or an `Open`'s nothing. -/
theorem stepK_ctl_session {st : State} {pl : Placed} {k : Session.Kind} {e : Env} (h : st.stop = none)
    (hs : slot st pl = some k) (hm : isModel k = false)
    (hgo : ((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
      (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true) :
    (stepK true st pl (some (tagOf k)) (.ok k) e).st.session = st.session := by
  obtain ⟨hg, hn⟩ := go_live hgo
  have hk := slot_kind hs
  rcases hr : st.request pl k with _ | ⟨st', o⟩
  · cases k with
    | register =>
      rcases stepK_frame_go (by simp) (by simp) (by simp) (by simp) hr hg with hst | ⟨r, hst, -⟩
      · exact absurd hn (Option.isSome_iff_ne_none.mp hst)
      · rw [hst] at hn ⊢
        rcases answer_register st r with h1 | ⟨d, h1⟩
        · exact absurd hn (Option.isSome_iff_ne_none.mp h1)
        · exact (congrArg State.session h1).trans rfl
    | link =>
      rcases stepK_frame_go (by simp) (by simp) (by simp) (by simp) hr hg with hst | ⟨r, hst, -⟩
      · exact absurd hn (Option.isSome_iff_ne_none.mp hst)
      · rw [hst]
        exact congrArg State.session (answer_link st r)
    | openStream s =>
      rcases stepK_frame_go (by simp) (by simp) (by simp) (by simp) hr hg with hst | ⟨r, hst, -⟩
      · exact absurd hn (Option.isSome_iff_ne_none.mp hst)
      · rw [hst]
        exact congrArg State.session (answer_open st s r)
    | hello => cases hm
    | commit => cases hm
    | round _ _ => cases hm
    | finish => cases hm
    | ping => rcases hk with ⟨h1, -⟩ | h1 <;> first | exact absurd rfl h1 | cases h1
    | proof => rcases hk with ⟨-, h1, -⟩ | h1 <;> first | exact absurd rfl h1 | cases h1
    | other => rcases hk with ⟨-, -, h1, -⟩ | h1 <;> first | exact absurd rfl h1 | cases h1
  · have hk' : k ≠ .ping ∧ k ≠ .proof ∧ k ≠ .finish := by
      rcases hk with ⟨h1, h2, -⟩ | rfl
      · exact ⟨h1, h2, fun hf => by subst hf; cases hm⟩
      · cases hm
    obtain ⟨h9, h5, h6⟩ := tagOf_ne hk'.1 hk'.2.1 hk'.2.2
    rw [stepK_refused h9 h5 h6 hr] at hn
    exact absurd hn (Option.isSome_iff_ne_none.mp (request_stop h hr))

/-- **Past its `Hello` the session has opened nothing**, where it goes on: the `Hello`'s answer starts its coins. -/
theorem stepK_hello_opened {st : State} {pl : Placed} {e : Env} (h : st.stop = none)
    (hgo : ((stepK true st pl (some (tagOf .hello)) (.ok .hello) e).go &&
      (stepK true st pl (some (tagOf .hello)) (.ok .hello) e).st.stop.isNone) = true) :
    opened (stepK true st pl (some (tagOf .hello)) (.ok .hello) e).st = 0 := by
  obtain ⟨hg, hn⟩ := go_live hgo
  rcases hr : st.request pl .hello with _ | ⟨st', o⟩
  · rw [stepK_hello hr] at hg hn ⊢
    rcases hh : st.sendHello e.setup e.nonce with ⟨st', o⟩ | ⟨st', o⟩
    · rw [hh] at hn
      exact absurd hn (Option.isSome_iff_ne_none.mp (sendHello_error h hh))
    · obtain ⟨_, _, hstop, -⟩ := sendHello_ok hh
      rcases ha : e.ans with _ | r
      · rw [hh, ha] at hg
        simp at hg
      · rw [hh, ha] at hn
        rcases answer_hello (r := r) (id := e.id) (show st'.stop = none by rw [hstop, h]) with
          hst | ⟨_, C, -, heq, hC⟩
        · exact absurd hn (Option.isSome_iff_ne_none.mp hst)
        · show opened (st'.answer .hello r e.id).1 = 0
          rw [heq]
          simp [opened, hC]
  · rw [stepK_refused (by decide) (by decide) (by decide) hr] at hn
    exact absurd hn (Option.isSome_iff_ne_none.mp (request_stop h hr))

/-- **A coin's answer the session goes on past counts one more opening.** -/
theorem coins_opened {st : State} {r : ByteArray} {C : Coins.Release} (h : st.stop = none)
    (hs : st.session = some C) (hn : (st.coins r).1.stop = none) : opened (st.coins r).1 = opened st + 1 := by
  rcases coins_cases (r := r) h hs with h1 | ⟨C', h1, hlen, -⟩
  · exact absurd hn (Option.isSome_iff_ne_none.mp h1)
  · rw [h1, opened_of hs]
    simp [opened, hlen]

/-- **The auditor's answer to a `Commit` or a frame round the session goes on past counts one more opening.** -/
theorem answer_coin_opened {st : State} {k : Session.Kind} {r : ByteArray} {C : Coins.Release} (h : st.stop = none)
    (hs : st.session = some C) (hk : k = .commit ∨ ∃ s n, k = .round s n)
    (hn : (st.answer k r).1.stop = none) : opened (st.answer k r).1 = opened st + 1 := by
  rcases hk with rfl | ⟨s, n, rfl⟩
  · rw [answer_commit] at hn ⊢
    by_cases hb : byte? r 0 = some 2
    · rw [ite_eq_left hb] at hn ⊢
      exact coins_opened h hs hn
    · rw [ite_eq_right hb] at hn
      exact absurd hn (Option.isSome_iff_ne_none.mp (noCoins_stop _ _ _))
  · rw [answer_round] at hn ⊢
    by_cases hb : byte? r 0 = some 2
    · rw [ite_eq_left hb] at hn ⊢
      exact coins_opened h hs hn
    · rw [ite_eq_right hb] at hn
      exact absurd hn (Option.isSome_iff_ne_none.mp (noCoins_stop _ _ _))

/-- **Where the session goes on past the auditor's answer, the session after it is `State.answer`'s**: a frame
round's coins its rep's frame refused stop the session. -/
theorem afterAnswer_coin {st : State} {k : Session.Kind} {r : ByteArray} {e : Env}
    (hn : (afterAnswer st k r e).stop = none) : afterAnswer st k r e = (st.answer k r).1 := by
  cases k with
  | round s n =>
    revert hn
    unfold afterAnswer
    dsimp only
    split
    · split
      · intro _
        rfl
      · intro hn
        cases hn
    · intro _
      rfl
  | _ => rfl

/-- **Past a `Commit` or a frame round the session has opened one more**, where it goes on. -/
theorem stepK_coin_opened {st : State} {pl : Placed} {k : Session.Kind} {e : Env} (hR : Reach st pl)
    (h : st.stop = none) (hs : slot st pl = some k) (hk : k = .commit ∨ ∃ s n, k = .round s n)
    (hgo : ((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
      (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true) :
    opened (stepK true st pl (some (tagOf k)) (.ok k) e).st = opened st + 1 := by
  obtain ⟨hg, hn⟩ := go_live hgo
  have hsent : st.sent = true := by
    rcases hk with rfl | ⟨s, n, rfl⟩
    · exact (slot_commit hs).1
    · exact (slot_round hs).1
  obtain ⟨C, hC⟩ := Option.isSome_iff_exists.mp (hR.sess hsent)
  have hk' : k ≠ .hello ∧ k ≠ .finish ∧ k ≠ .ping ∧ k ≠ .proof := by
    rcases hk with rfl | ⟨s, n, rfl⟩ <;> exact ⟨by simp, by simp, by simp, by simp⟩
  rcases hr : st.request pl k with _ | ⟨st', o⟩
  · rcases stepK_frame_go hk'.1 hk'.2.1 hk'.2.2.1 hk'.2.2.2 hr hg with hst | ⟨r, hst, -⟩
    · exact absurd hn (Option.isSome_iff_ne_none.mp hst)
    · rw [hst] at hn ⊢
      have ha := afterAnswer_coin hn
      rw [ha] at hn ⊢
      exact answer_coin_opened h hC hk hn
  · obtain ⟨h9, h5, h6⟩ := tagOf_ne hk'.2.2.1 hk'.2.2.2 hk'.2.1
    rw [stepK_refused h9 h5 h6 hr] at hn
    exact absurd hn (Option.isSome_iff_ne_none.mp (request_stop h hr))

/-- **At a slot that reads no model message `LoopP` carries over**: where the loop's reading of it brings only the
updates between the session's requests and takes no model message, and `J` is kept there, the request keeps the
session's setup, the streams seen, whether its `Hello` left, its coins and the model's messages to come, and the
updates keep each unseen rep's frame absent or fresh. -/
theorem slotCase_carried {W Rr : Type} {M : LoopModel W Rr} {J : List String → State → LoopAt → Prop} {sid : Nat}
    {spec : SessionSpec} {sts : Array Stmt} {nonce : ByteArray} {k : Session.Kind} (hm : isModel k = false)
    (hs : ∀ seen, seenAfter seen k = seen)
    (hr : ∀ st pl e σ σ', LoopRead M sid spec sts nonce k st pl e σ σ' → Between sid σ.R σ'.R ∧ σ'.c = σ.c)
    (hJ : LoopKept M sid spec sts nonce J k) :
    SlotCase spec (LoopP M J sid spec sts nonce) k := by
  intro seen st pl e rest ms hsl hP
  obtain ⟨σ, hms, hF, hj, hRd, hset⟩ := hP
  refine ⟨fun h => absurd (hm.symm.trans h) Bool.false_ne_true, fun hgo => ?_⟩
  cases hRd with
  | stop h =>
    rw [hsl] at h
    cases h
  | step k' hk' hra hrest =>
    obtain rfl : k = k' := Option.some.inj (hsl.symm.trans hk')
    have hJ' := hJ seen st pl e σ _ hsl (hset e List.mem_cons_self) hF hj hra hgo
    obtain ⟨hbt, hcc⟩ := hr _ _ _ _ _ hra
    have hk : (Except.ok k : Except String Session.Kind) ≠ .ok .hello := by
      intro h
      cases h
      cases hm
    have hop : opened (stepK true st pl (some (tagOf k)) (.ok k) e).st = opened st := by
      unfold opened
      rw [stepK_ctl_session hF.live hsl hm hgo]
    refine ⟨_, ?_, ⟨?_, ?_, (go_live hgo).2, stepK_reach hF.reach hF.live hsl hgo, ?_, ?_⟩, hJ', hrest hgo,
      fun e' he' => hset e' (List.mem_cons_of_mem _ he')⟩
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
    · intro h
      rw [stepK_sent hk] at h
      rw [hcc, hop]
      exact hF.count h

/-- **The `Hello`'s slot case**: where the model's first message is the `Hello` at the nonce the relay drew and `J` is
kept at the `Hello`, what the relay read at the `Hello` is the model's message, and `LoopP` holds at the next request
with that message taken, the session's coins started with nothing opened. -/
theorem slotCase_hello {W Rr : Type} {M : LoopModel W Rr} {J : List String → State → LoopAt → Prop} {sid : Nat}
    {spec : SessionSpec} {sts : Array Stmt} {nonce : ByteArray}
    (h0 : M.msgs.headD [] = _root_.Flock.Firewall.Hello.msg spec nonce)
    (hJ : LoopKept M sid spec sts nonce J .hello) :
    SlotCase spec (LoopP M J sid spec sts nonce) .hello := by
  intro seen st pl e rest ms hsl hP
  obtain ⟨σ, hms, hF, hj, hRd, hset⟩ := hP
  have hc0 : σ.c = 0 := hF.start (slot_hello hsl).2
  cases hRd with
  | stop h =>
    rw [hsl] at h
    cases h
  | step k' hk' hra hrest =>
    obtain rfl : Session.Kind.hello = k' := Option.some.inj (hsl.symm.trans hk')
    have hJ' := hJ seen st pl e σ _ hsl (hset e List.mem_cons_self) hF hj hra
    obtain ⟨-, hn, hbt, hcc, -⟩ := hra
    refine ⟨fun _ => ?_, fun hgo => ?_⟩
    · show _root_.Flock.Firewall.Hello.msg spec e.nonce = ms.headD []
      rw [hn, hms, hc0, List.drop_zero, h0]
    · refine ⟨_, ?_, ⟨?_, ?_, (go_live hgo).2, stepK_reach hF.reach hF.live hsl hgo, ?_, ?_⟩, hJ' hgo,
        hrest hgo, fun e' he' => hset e' (List.mem_cons_of_mem _ he')⟩
      · simp only [isModel, ↓reduceIte]
        rw [hcc, hms, List.tail_drop]
      · exact stepK_setup (hset e List.mem_cons_self) hF.setup
      · intro i r s hmem hseen
        exact freshAt_kept (hbt.1 i r) (hF.fresh i r s hmem hseen)
      · intro h
        rw [stepK_hello_sent hgo] at h
        cases h
      · intro _
        rw [hcc, hc0, stepK_hello_opened hF.live hgo]

/-- **`EnvPre` from the relay's invariant at the deployed loop**: given `envPre_inv`'s step at the `Hello`, the
`Commit`, each frame round and the `Finish`, and `J` kept at every other slot, `EnvPre` holds wherever `LoopP` does;
the step at every other slot is `slotCase_carried`'s. -/
theorem envPre_loop {W Rr : Type} {M : LoopModel W Rr} {J : List String → State → LoopAt → Prop} {sid : Nat}
    {spec : SessionSpec} {sts : Array Stmt} {nonce : ByteArray}
    (hJ : ∀ k, isModel k = false → LoopKept M sid spec sts nonce J k)
    (hHello : SlotCase spec (LoopP M J sid spec sts nonce) .hello)
    (hCommit : SlotCase spec (LoopP M J sid spec sts nonce) .commit)
    (hRound : ∀ stream n, SlotCase spec (LoopP M J sid spec sts nonce) (.round stream n))
    (hFinish : SlotCase spec (LoopP M J sid spec sts nonce) .finish) :
    ∀ (es : List Env) (seen : List String) (st : State) (pl : Placed) (ms : List Claims.Msg),
      LoopP M J sid spec sts nonce seen st pl es ms → EnvPre spec seen (schedSteps true st pl es) ms :=
  envPre_inv (LoopP M J sid spec sts nonce) fun seen st pl e rest ms k => by
    have hc : ∀ k : Session.Kind, isModel k = false → (∀ seen, seenAfter seen k = seen) →
        (∀ st pl e σ σ', LoopRead M sid spec sts nonce k st pl e σ σ' → Between sid σ.R σ'.R ∧ σ'.c = σ.c) →
        SlotCase spec (LoopP M J sid spec sts nonce) k := fun k hm hs hr => slotCase_carried hm hs hr (hJ k hm)
    cases k with
    | hello => exact hHello seen st pl e rest ms
    | commit => exact hCommit seen st pl e rest ms
    | round stream n => exact hRound stream n seen st pl e rest ms
    | finish => exact hFinish seen st pl e rest ms
    | register => exact hc .register rfl (fun _ => rfl) (fun _ _ _ _ _ h => ⟨h.1, h.2.1⟩) seen st pl e rest ms
    | link => exact hc .link rfl (fun _ => rfl) (fun _ _ _ _ _ h => ⟨h.1, h.2.1⟩) seen st pl e rest ms
    | openStream s => exact hc (.openStream s) rfl (fun _ => rfl) (fun _ _ _ _ _ h => ⟨h.1, h.2.1⟩) seen st pl e rest ms
    | proof => exact hc .proof rfl (fun _ => rfl) (fun _ _ _ _ _ h => ⟨h.1, h.2.1⟩) seen st pl e rest ms
    | ping => exact hc .ping rfl (fun _ => rfl) (fun _ _ _ _ _ h => ⟨h.1, h.2.1⟩) seen st pl e rest ms
    | other => exact hc .other rfl (fun _ => rfl) (fun _ _ _ _ _ h => ⟨h.1, h.2.1⟩) seen st pl e rest ms

end Claims.ZKServer.Relay
