module

public import Proofs.Flock.ZkAny.RelayLoop

/-! The relay's run as its IO code serves it, under the named trusts `flock-relay-serves` and `flock-worker-deterministic` (`loopRead_of_serve`). -/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.ZkServer.Session
open _root_.Flock.ZkServer hiding State
open _root_.Flock (SessionSpec Stmt)

def LoopServe {W Rr : Type} (M : LoopModel W Rr) (rv : List (Claims.Msg × Claims.Msg) → Option Claims.Msg)
    (sid : Nat) (spec : SessionSpec) (sts : Array Stmt) (nonce : ByteArray) :
    Session.Kind → State → Placed → Env → LoopAt → LoopAt → Prop
  | .hello, _, _, e, σ, σ' =>
    e.nonce = nonce ∧ Between sid σ.R σ'.R ∧ σ'.c = σ.c + 1 ∧
      M.L.toSrv.liveBefore M.wk M.x M.w M.ρ M.κ (M.hist σ'.c) = true ∧
      (e.setup = .ok (spec, sts) → SessOn spec sts σ'.R sid)
  | .commit, st, _, e, σ, σ' =>
    (∃ f, e.commit = (commitRead σ.L0 sid (spec.tables.zip (sts.map (·.m))) f).1 ∧
        ∀ name, _root_.Flock.ZkServer.Level0.ofSession σ'.L0 sid name =
          _root_.Flock.ZkServer.Level0.ofSession (commitRead σ.L0 sid (spec.tables.zip (sts.map (·.m))) f).2 sid name) ∧
      (∃ ct, _root_.Flock.ZkServer.Session.streamsOk spec sts ct = true) ∧
      (∃ (ls : Array _root_.Flock.ZkServer.Level0.Level0) (publics : List ByteArray), AfterSample sid ls σ.L0 ∧
        (∀ j (hj : j < ls.size), ∃ g d hi, M.src (2 * j) = .ok (g, d, hi) ∧
          _root_.Flock.ZkServer.Level0.ofSamples sts[j]!.wideLeaf sts[j]!.m g d = .ok ls[j]) ∧
        M.L.ownMsg M.x M.w M.ρ (M.hist σ.c) = 0 :: FlockDeploy.linkBody (ls.toList.map (·.root)) publics ∧
        ls.size < 2 ^ 64 ∧ publics.length < 2 ^ 64 ∧ (∀ l ∈ ls, l.root.size < 2 ^ 64) ∧
        ∀ b ∈ publics, b.size < 2 ^ 64) ∧
      (∀ a, e.ans = some a → byte? a 0 = some 2 → (st.answer .commit a).1.stop = none →
        ∀ rel, sessAt σ'.R sid = some rel →
          rel.link = some (_root_.Flock.ZkServer.Frame.words (st.answer .commit a).2 5
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

/-- `flock-relay-serves`, a trust in the relay's IO code, which Lean reads but does not run: each clause is one IO
function's effect on the loop's references or the auditor's connection, held to `Session.relay`'s loop body by reading. -/
def RelayServes {W Rr : Type} (M : LoopModel W Rr) (rv : List (Claims.Msg × Claims.Msg) → Option Claims.Msg)
    (aud : List ByteArray → Option ByteArray) (sid : Nat) (spec : SessionSpec) (sts : Array Stmt) (nonce : ByteArray)
    (σ : LoopAt) (ins : List (ByteArray × Env)) : Prop :=
  ReadsBy (LoopServe M rv sid spec sts nonce) σ {} {} (pick true {} {} ins) ∧
    AudReads aud [] {} {} (pick true {} {} ins)

/-- `flock-worker-deterministic`, a trust in the worker's code: where the worker sends a frame round's content `v` at a
history (`LiveChallenger::round`'s `Req::Round`), the model's worker gives `v` there. -/
def WorkerDeterministic {W Rr : Type} (M : LoopModel W Rr)
    (rv : List (Claims.Msg × Claims.Msg) → Option Claims.Msg) : Prop :=
  ∀ h v, rv h = some v → M.wk M.x M.w M.ρ h 0 = some (v, false)

def AnsPart {W Rr : Type} (M : LoopModel W Rr) (c : Nat) (e : Env) : Session.Kind → Prop
  | .hello | .commit | .round _ _ => M.Ans c e
  | .finish => c < M.τ.length
  | _ => True

theorem AnsPart.of_ctl {W Rr : Type} {M : LoopModel W Rr} {c : Nat} {e : Env} {k : Session.Kind}
    (hm : isModel k = false) : AnsPart M c e k := by
  cases k
  case hello | commit | round | finish => exact absurd hm (by decide : ¬(true = false))
  all_goals exact True.intro

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
