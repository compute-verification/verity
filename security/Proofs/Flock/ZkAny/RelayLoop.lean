module

public import Proofs.Flock.ZkAny.RelayInv
public import Proofs.Flock.ZkAny.RelayEnv
public import Proofs.Flock.ZkAny.RelayFresh

/-! The relay's requests as the deployed loop answers them: the invariant `LoopP` against the model's run, and `EnvPre` from it (`envPre_loop`). -/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.ZkServer.Session
open _root_.Flock.ZkServer hiding State
open _root_.Flock (SessionSpec Stmt)
open _root_.Flock.ZkServer.Frames (Sessions)

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

def msgs : List Claims.Msg := M.τ.map (·.1)

open Claims.ZKMal in
def hist (j : Nat) : List (Claims.Msg × Claims.Msg) := histOf (M.τ.take j)

def Own : Prop :=
  ∀ j (hj : j < M.τ.length), (M.τ[j]'hj).1 = M.L.toSrv.nextD M.L.fillOf M.nN M.wk M.x M.w M.ρ M.κ (M.hist j)

def Ans (j : Nat) (e : Env) : Prop :=
  ∃ hj : j < M.τ.length, ∀ a, e.ans = some a → byte? a 0 = some 2 → (M.τ[j]'hj).2 = .msg a.data.toList

end LoopModel

structure LoopAt where
  R : Sessions
  L0 : _root_.Flock.ZkServer.Level0.Slots
  c : Nat

def Stays (sid : Nat) (σ σ' : LoopAt) : Prop :=
  (∀ name, _root_.Flock.ZkServer.Level0.ofSession σ'.L0 sid name =
      _root_.Flock.ZkServer.Level0.ofSession σ.L0 sid name) ∧
    ∀ name l0, _root_.Flock.ZkServer.Level0.ofSession σ.L0 sid name = some l0 → Stay sid l0.root σ.R σ'.R

def SaltsSrc {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (σ : LoopAt) : Prop :=
  ∀ (i r : Nat) (rel : _root_.Flock.ZkServer.Frames.Release) (name : String)
    (l0 : _root_.Flock.ZkServer.Level0.Level0) (r0 : ByteArray) (q0 : Nat) (hi : Array ByteArray), r < 2 →
    sessAt σ.R sid = some rel → rel.spec.tables[i]? = some name →
    _root_.Flock.ZkServer.Level0.ofSession σ.L0 sid name = some l0 →
    σ.R.salts.find? (fun x => x.1 == l0.root && x.2.1 == r) = some (r0, q0, hi) →
    ∃ g d, M.src (2 * i + r) = .ok (g, d, hi)

def InnerSrc {W Rr : Type} (M : LoopModel W Rr) (sid i r : Nat) (σ : LoopAt) : Prop :=
  ∀ (rel : _root_.Flock.ZkServer.Frames.Release) (name : String) (l0 : _root_.Flock.ZkServer.Level0.Level0),
    sessAt σ.R sid = some rel → rel.spec.tables[i]? = some name →
    _root_.Flock.ZkServer.Level0.ofSession σ.L0 sid name = some l0 →
    ∀ (k : Nat) (b : ByteArray), (σ.R.innerOf l0.root r)[k]? = some b → (M.fin (M.hist σ.c))[k]? = some b

def StartOk (sts : Array Stmt) (sid i r : Nat) (σ : LoopAt) : Prop :=
  ∀ p, frameAt σ.R sid i r = some p → ∃ lm, _root_.Flock.ZkServer.Drive.start sts[i]! = .ok lm

def SessOn (spec : SessionSpec) (sts : Array Stmt) (R : Sessions) (sid : Nat) : Prop :=
  ∃ rel : _root_.Flock.ZkServer.Frames.Release, sessAt R sid = some rel ∧ rel.spec = spec ∧ rel.sts = sts

theorem SessOn.eq {spec : SessionSpec} {sts : Array Stmt} {R : Sessions} {sid : Nat} (h : SessOn spec sts R sid)
    {rel : _root_.Flock.ZkServer.Frames.Release} (hr : sessAt R sid = some rel) : rel.spec = spec ∧ rel.sts = sts := by
  obtain ⟨rel', hr', hs, ht⟩ := h
  rw [hr] at hr'
  cases hr'
  exact ⟨hs, ht⟩

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
    show (R.sessions.map fun (s : _root_.Flock.ZkServer.Frames.Release) =>
      if s.sid == sid' then { s with link := some c } else s).find? (·.sid == sid) = _
    rw [Array.find?_map]
    have hc : ((·.sid == sid) ∘ fun (s : _root_.Flock.ZkServer.Frames.Release) =>
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

theorem sessOn_sessKept {spec : SessionSpec} {sts : Array Stmt} {sid : Nat} {R R' : Sessions}
    (h : SessKept sid R R') (hs : SessOn spec sts R sid) : SessOn spec sts R' sid := by
  induction h with
  | refl => exact hs
  | step _ hst ih => exact sessOn_sessKeptStep hst ih

def LoopRead {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (spec : SessionSpec) (sts : Array Stmt)
    (nonce : ByteArray) : Session.Kind → State → Placed → Env → LoopAt → LoopAt → Prop
  | .hello, _, _, e, σ, σ' =>
    M.Ans σ.c e ∧ e.nonce = nonce ∧ Between sid σ.R σ'.R ∧ σ'.c = σ.c + 1 ∧
      M.L.toSrv.liveBefore M.wk M.x M.w M.ρ M.κ (M.hist σ'.c) = true ∧
      (e.setup = .ok (spec, sts) → SessOn spec sts σ'.R sid)
  | .commit, st, _, e, σ, σ' =>
    M.Ans σ.c e ∧ (∃ f, e.commit = (commitRead σ.L0 sid (spec.tables.zip (sts.map (·.m))) f).1 ∧
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

structure LoopFacts (sid : Nat) (spec : SessionSpec) (seen : List String) (st : State) (pl : Placed) (σ : LoopAt) :
    Prop where
  setup : SetupOk spec st
  fresh : ∀ i r s, (i, r, s) ∈ spec.repStreams.toList → seen.contains s = false → FreshAt σ.R sid i r
  live : st.stop = none
  reach : Reach st pl
  start : st.sent = false → σ.c = 0
  count : st.sent = true → σ.c = opened st + 1

def LoopKept {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (spec : SessionSpec) (sts : Array Stmt)
    (nonce : ByteArray) (J : List String → State → LoopAt → Prop) (k : Session.Kind) : Prop :=
  ∀ seen st pl e σ σ', slot st pl = some k → e.setup = .ok (spec, sts) → LoopFacts sid spec seen st pl σ →
    J seen st σ → LoopRead M sid spec sts nonce k st pl e σ σ' →
    ((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
        (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true →
      J (seenAfter seen k) (stepK true st pl (some (tagOf k)) (.ok k) e).st σ'

def LoopP {W Rr : Type} (M : LoopModel W Rr) (J : List String → State → LoopAt → Prop) (sid : Nat)
    (spec : SessionSpec) (sts : Array Stmt) (nonce : ByteArray) (seen : List String) (st : State) (pl : Placed)
    (es : List Env) (ms : List Claims.Msg) : Prop :=
  ∃ σ : LoopAt, ms = M.msgs.drop σ.c ∧ LoopFacts sid spec seen st pl σ ∧ J seen st σ ∧
    LoopReads M sid spec sts nonce σ st pl es ∧ ∀ e ∈ es, e.setup = .ok (spec, sts)

def SlotCase (spec : SessionSpec) (P : List String → State → Placed → List Env → List Claims.Msg → Prop)
    (k : Session.Kind) : Prop :=
  ∀ seen st pl e rest ms, slot st pl = some k → P seen st pl (e :: rest) ms →
    (isModel k = true → EnvOk spec seen k e (ms.headD [])) ∧
    (((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
        (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true →
      P (seenAfter seen k) (stepK true st pl (some (tagOf k)) (.ok k) e).st
        (stepK true st pl (some (tagOf k)) (.ok k) e).pl rest (if isModel k then ms.tail else ms))

theorem go_live {r : Res} (h : (r.go && r.st.stop.isNone) = true) : r.go = true ∧ r.st.stop = none := by
  simp only [Bool.and_eq_true, Option.isNone_iff_eq_none] at h
  exact h

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

theorem coins_opened {st : State} {r : ByteArray} {C : Coins.Release} (h : st.stop = none)
    (hs : st.session = some C) (hn : (st.coins r).1.stop = none) : opened (st.coins r).1 = opened st + 1 := by
  rcases coins_cases (r := r) h hs with h1 | ⟨C', h1, hlen, -⟩
  · exact absurd hn (Option.isSome_iff_ne_none.mp h1)
  · rw [h1, opened_of hs]
    simp [opened, hlen]

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

theorem slotCase_hello {W Rr : Type} {M : LoopModel W Rr} {J : List String → State → LoopAt → Prop} {sid : Nat}
    {spec : SessionSpec} {sts : Array Stmt} {nonce : ByteArray}
    (h0 : M.msgs.headD [] = _root_.Flock.ZkServer.Hello.msg spec nonce)
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
    · show _root_.Flock.ZkServer.Hello.msg spec e.nonce = ms.headD []
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
