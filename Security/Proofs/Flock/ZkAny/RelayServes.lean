module

public import Proofs.Flock.ZkAny.RelayLoop

/-!
# The relay's run as its IO code serves it

`LoopReads` reads the relay's requests as the loop answers them. Most of its clauses say what the relay's IO code did at
a slot; two couple the run to the model's: the auditor answering as `V*` did (`LoopModel.Ans`, and at the `Finish` the
model's run having a message there), and the model's worker giving the worker's round values. Here they stand apart:
the IO code's readings are the named trust `flock-relay-serves`, the worker's the named trust
`flock-worker-deterministic`, and the auditor's a statement about `V*`:

* `LoopServe`: `LoopRead` without the couplings: a frame round runs on the worker's round value at the model's history.
* `ReadsBy`: `LoopReads`' walk over the requests at any reading of a slot. `AudReads`: the relay's answers are the
  auditor's, to everything that left so far.
* **`RelayServes`**: the relay's requests from its start, each served so (`ReadsBy` at `LoopServe`), with the
  auditor's answers (`AudReads`).
* **`WorkerDeterministic`**: the model's worker gives the worker's round values, a function of the history.
* `AnsPart`: what `LoopRead` asks of a slot beyond `LoopServe`'s clauses and the worker's values.
* `loopRead_of_serve`: so a slot is read as the loop answers it (`LoopRead`) where its `AnsPart` holds and the model's
  worker answers the `Finish` with the ZK server's own message. `RelayAuditor`'s `loopReads_auditorV` gives `AnsPart` at
  each slot where the auditor is `V*`, and so `LoopReads`.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session
open _root_.Flock.Firewall hiding State
open _root_.Flock (SessionSpec Stmt)

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

/-- **A control slot asks nothing beyond what the IO code serves.** -/
theorem AnsPart.of_ctl {W Rr : Type} {M : LoopModel W Rr} {c : Nat} {e : Env} {k : Session.Kind}
    (hm : isModel k = false) : AnsPart M c e k := by
  cases k
  case hello | commit | round | finish => exact absurd hm (by decide : ¬(true = false))
  all_goals exact True.intro

/-- **What the IO code serves at a slot is what the relay reads there**, where the slot's `AnsPart` holds (`hA`), the
model's worker gives the worker's round values (`hw`), and it answers the `Finish` with the ZK server's own message
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

end Claims.ZKServer.Relay
