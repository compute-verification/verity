module

public import Proofs.Flock.ZkAny.RelayLoop
public import Proofs.Flock.Claims.ZK.SaltProver

/-!
# The session's link at the deployed loop

At the session's `Commit` the loop links the session's entry in its frames to the words of the coins the prover
receives for the auditor's answer (`Sessions.link`, `LoopRead`'s `Commit` clause), and nothing but that `Commit` links
it. Here that link is the model's: the coins of the history's link round at the loop's count, from the `Commit` on.

* `LinksOk`: the session's `Hello` and its coins count the setup's link points (`helloOf`). `stepK_links`: each request
  keeps it where the relay reads the setup; `linksOk_empty`: it holds before the session's first request.
* `commit_words`: at the `Commit`, the words the loop links are the coins the frame is fed for the auditor's answer at
  the session stream's count (`Frames.feedOf`), within `Frames.COIN_MAX` words. `coins_session`, `open_parts`: the
  coins the session's first opening returns, the opening's parts after its tag and count.
* `wordsAt_hist_two`, `wordsAt_hist_zero`: the history's link coins at the loop's count are those of the auditor's
  answer to the run's second message, at every history past it.
* `LinkOn`: the session's entry linked to the history's link coins. `linkOn_stays`: kept from one of the session's
  requests to the next where what its rounds read of the loop stays (`Stays`), past the link. `linkOn_commit`: it holds
  past the `Commit`, where the session goes on.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session
open _root_.Flock.Firewall hiding State
open _root_.Flock (SessionSpec Stmt)
open _root_.Flock.Firewall.Frames (Sessions)

/-! ## The session's link points -/

/-- **The session's `Hello` and its coins count the setup's link points**: the session stream's coin count
`points · m_pts`, as `helloOf` sets it. -/
def LinksOk (spec : SessionSpec) (S : State) : Prop :=
  (∀ hh, S.hello = some hh → hh.links = spec.points * spec.mPts) ∧
    ∀ C, S.session = some C → C.hello.links = spec.points * spec.mPts

/-- What a request keeps of the session's coins: its `Hello` and its session. -/
abbrev KeepsC (S S' : State) : Prop := S'.hello = S.hello ∧ S'.session = S.session

theorem linksOk_keepsC {spec : SessionSpec} {S S' : State} (h : LinksOk spec S) (hk : KeepsC S S') :
    LinksOk spec S' :=
  ⟨fun hh e => h.1 hh (hk.1.symm.trans e), fun C e => h.2 C (hk.2.symm.trans e)⟩

theorem linksOk_empty (spec : SessionSpec) : LinksOk spec {} :=
  ⟨fun _ h => (by cases h), fun _ h => (by cases h)⟩

theorem finish_keepsC {S S' : State} {o : ByteArray} (hf : S.finish = some (S', o)) : KeepsC S S' := by
  revert hf
  unfold State.finish State.refuse
  repeat' split
  all_goals intro hf; first
    | (simp only [Option.some.injEq, Prod.mk.injEq] at hf; rw [← hf.1]; exact ⟨rfl, rfl⟩)
    | simp at hf

theorem request_keepsC {S S' : State} {P : Placed} {k : Session.Kind} {o : ByteArray}
    (hr : S.request P k = some (S', o)) : KeepsC S S' := by
  revert hr
  unfold State.request
  split
  · intro hr
    simp only [Option.some.injEq, Prod.mk.injEq] at hr
    rw [← hr.1]
    exact ⟨rfl, rfl⟩
  · dsimp only
    split
    · rename_i r hc
      intro hr
      simp only [Option.some.injEq] at hr
      subst hr
      revert hc
      split
      · exact finish_keepsC
      · unfold State.refuse
        repeat' split
        all_goals intro hc; first
          | (simp only [Option.some.injEq, Prod.mk.injEq] at hc; rw [← hc.1]; exact ⟨rfl, rfl⟩)
          | simp at hc
    · rcases hp : S.place P k with _ | why
      · simp
      · simp only [Option.map_some, Option.some.injEq]
        intro hr
        cases hr
        exact ⟨rfl, rfl⟩

theorem sendHello_keepsC {S S' : State} {s : _root_.Flock.V (SessionSpec × Array Stmt)} {n o : ByteArray}
    (he : S.sendHello s n = .error (S', o)) : KeepsC S S' := by
  revert he
  unfold State.sendHello State.refuse
  repeat' split
  all_goals intro he; first | (cases he; exact ⟨rfl, rfl⟩) | cases he

theorem noCoins_keepsC (S : State) (what : String) (a : ByteArray) : KeepsC S (S.noCoins what a).1 := by
  unfold State.noCoins State.refuse
  split
  all_goals exact ⟨rfl, rfl⟩

theorem start_hello_eq {h : Coins.Hello} {a : List UInt8} {C : Coins.Release}
    (hs : Coins.Release.start h a = .ok C) : C.hello = h := by
  revert hs
  unfold Coins.Release.start
  split
  · intro hs; cases hs; rfl
  · intro hs; cases hs

/-- **The `Hello` the setup gives counts its link points.** -/
theorem helloOf_links {spec : SessionSpec} {sts : Array Stmt} {n : ByteArray} {hh : Coins.Hello}
    (h : helloOf spec sts n = some hh) : hh.links = spec.points * spec.mPts := by
  revert h
  unfold helloOf
  split
  · split
    · intro h
      cases h
      rfl
    · intro h
      cases h
  · intro h
    cases h

/-- **A coin's answer keeps `LinksOk`**: the opening keeps the session's `Hello`. -/
theorem coins_links {spec : SessionSpec} {S : State} (a : ByteArray) (h : LinksOk spec S) :
    LinksOk spec (S.coins a).1 := by
  rcases hs : S.stop with _ | why
  · rcases hC : S.session with _ | C
    · unfold State.coins
      rw [hs, hC]
      exact linksOk_keepsC h ⟨rfl, rfl⟩
    · unfold State.coins
      rw [hs, hC]
      dsimp only
      rcases ho : C.open (reqAt S.plan C.hello.links C.opened.length).1
          (reqAt S.plan C.hello.links C.opened.length).2 a.data.toList with e | ⟨C', c⟩
      · exact linksOk_keepsC h ⟨rfl, rfl⟩
      · refine ⟨h.1, fun C'' e => ?_⟩
        have e' : some C' = some C'' := e
        cases e'
        rw [(open_ok ho).2]
        exact h.2 C hC
  · unfold State.coins
    rw [hs]
    exact linksOk_keepsC h ⟨rfl, rfl⟩

/-- **The `Hello`'s answer keeps `LinksOk`**: the session it starts has the session's `Hello`. -/
theorem answer_hello_links {spec : SessionSpec} {S : State} {r id : ByteArray} (h : LinksOk spec S) :
    LinksOk spec (S.answer .hello r id).1 := by
  rcases hs : S.stop with _ | why
  · unfold State.answer
    simp only [hs]
    rcases hhel : S.hello with _ | hh
    · dsimp only
      split <;> exact linksOk_keepsC h ⟨rfl, rfl⟩
    · dsimp only
      rcases hst : Coins.Release.start hh r.data.toList with e | C
      · exact linksOk_keepsC h ⟨rfl, rfl⟩
      · refine ⟨fun hh' e => h.1 hh' (hhel.trans e), fun C' e => ?_⟩
        have e' : some C = some C' := e
        cases e'
        rw [start_hello_eq hst]
        exact h.1 hh hhel
  · unfold State.answer
    simp only [hs]
    first | exact h | exact linksOk_keepsC h ⟨rfl, rfl⟩

theorem answer_links {spec : SessionSpec} {S : State} (k : Session.Kind) (a id : ByteArray) (h : LinksOk spec S) :
    LinksOk spec (S.answer k a id).1 := by
  have hh := answer_hello_links (spec := spec) (S := S) (r := a) (id := id) h
  unfold State.answer
  split
  all_goals first
    | exact coins_links a h
    | exact linksOk_keepsC h (noCoins_keepsC S _ a)
    | exact linksOk_keepsC h ⟨rfl, rfl⟩
    | exact hh
    | ((unfold State.refuse; repeat' split) <;> exact linksOk_keepsC h ⟨rfl, rfl⟩)

theorem afterAnswer_links {spec : SessionSpec} {st : State} (k : Session.Kind) (r : ByteArray) (e : Env)
    (h : LinksOk spec st) : LinksOk spec (afterAnswer st k r e) := by
  have ha := answer_links (S := st) k r .empty h
  cases k
  case round s n =>
    unfold afterAnswer
    dsimp only
    repeat' split
    all_goals first | exact ha | exact linksOk_keepsC ha ⟨rfl, rfl⟩
  all_goals exact ha

/-- **Each request keeps the session's `Hello` and coins at the setup's link points**, where the relay reads the
setup: the `Hello` sets the setup's (`helloOf`), and only it and the coins it starts set them. -/
theorem stepK_links {hold : Bool} {spec : SessionSpec} {sts : Array Stmt} {st : State} {pl : Placed}
    {b : Option UInt8} {k : Except String Session.Kind} {e : Env} (he : e.setup = .ok (spec, sts))
    (h : LinksOk spec st) : LinksOk spec (stepK hold st pl b k e).st := by
  have hello : ∀ {S' : State} {o : ByteArray}, st.sendHello e.setup e.nonce = .ok (S', o) → LinksOk spec S' := by
    intro S' o hh
    obtain ⟨spec', sts', hsp, rfl, -⟩ := sendHello_eq hh
    rw [he] at hsp
    cases hsp
    exact ⟨fun hh' hx => helloOf_links (spec := spec) (sts := sts) (n := e.nonce) hx, h.2⟩
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
    · exact linksOk_keepsC h (request_keepsC ‹_›)
    · split
      · exact linksOk_keepsC h (sendHello_keepsC ‹_›)
      · rename_i hh
        split
        · exact hello hh
        · exact answer_links _ _ _ (hello hh)
  · split
    · exact linksOk_keepsC h (request_keepsC ‹_›)
    · split
      · exact linksOk_keepsC h ⟨rfl, rfl⟩
      · split
        · exact h
        · exact afterAnswer_links _ _ _ h

/-! ## The words the `Commit` links -/

theorem check_parts {spec : _root_.Flock.CoinTree.Spec} {ν : List UInt8} {key : ByteArray} {root : List UInt8}
    {s r n : Nat} {m c : List UInt8} (h : Coins.check spec ν key root s r n m = .ok c) :
    m = Coins.frame (Coins.openingWords spec n) (Coins.parts n (Coins.depth spec) (m.drop 5)).join ∧
      c = (Coins.parts n (Coins.depth spec) (m.drop 5)).cs.flatten := by
  unfold Coins.check at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hp
    split at h
    · cases h
      have hp' : Coins.parses spec s r n m = true := by simpa using hp
      simp only [Coins.parses, Bool.and_eq_true, decide_eq_true_eq] at hp'
      exact ⟨hp'.2, rfl⟩
    · cases h

/-- **An opening's coins are its parts' after its tag and count**, and the opening is exactly `Resp::Coins` of them. -/
theorem open_parts {C C' : Coins.Release} {s : String} {n : Nat} {m c : List UInt8}
    (h : C.open s n m = .ok (C', c)) :
    m = Coins.frame (Coins.openingWords C.hello.spec (if s == Coins.SESSION then C.hello.links else n))
        (Coins.parts (if s == Coins.SESSION then C.hello.links else n) (Coins.depth C.hello.spec) (m.drop 5)).join ∧
      c = (Coins.parts (if s == Coins.SESSION then C.hello.links else n) (Coins.depth C.hello.spec)
        (m.drop 5)).cs.flatten := by
  unfold Coins.Release.open at h
  dsimp only at h
  by_cases hs : C.hello.spec.streams.toList.idxOf s < C.hello.spec.streams.size
  · simp only [hs, ↓reduceIte] at h
    cases hc : Coins.check C.hello.spec C.hello.nonce C.key C.root (C.hello.spec.streams.toList.idxOf s)
        (C.opened.count (C.hello.spec.streams.toList.idxOf s)) (if s == Coins.SESSION then C.hello.links else n)
        m with
    | ok c' =>
      rw [hc] at h
      cases h
      exact check_parts hc
    | error e =>
      rw [hc] at h
      cases h
  · simp only [hs, ↓reduceIte] at h
    cases h

theorem fix_length (n : Nat) (b : List UInt8) : (Coins.fix n b).length = n := by
  unfold Coins.fix
  rw [List.length_take, List.length_append, List.length_replicate]
  omega

/-- **An opening of `n` coins holds `16 n` bytes of them.** -/
theorem parts_cs_length (n d : Nat) (b : List UInt8) : (Coins.parts n d b).cs.flatten.length = 16 * n := by
  show (((List.range n).map fun i => Coins.fix 16 (b.drop (16 * i))).flatten).length = 16 * n
  induction n with
  | zero => first | rfl | simp
  | succ n ih =>
    rw [List.range_succ, List.map_append, List.flatten_append, List.length_append, ih]
    simp only [List.map_cons, List.map_nil, List.flatten_cons, List.flatten_nil, List.append_nil, fix_length]
    omega

theorem u32be_length (v : Nat) : (Coins.u32be v).length = 4 := by
  first | rfl | simp [Coins.u32be]

theorem frame_drop5 (W : Nat) (b : List UInt8) : (Coins.frame W b).drop 5 = b := by
  rw [Coins.frame, List.drop_succ_cons]
  exact List.drop_left' (by rw [List.length_reverse, u32be_length])

theorem frame_length (W : Nat) (c : List UInt8) : (Coins.frame W c).length = c.length + 5 := by
  rw [Coins.frame, List.length_cons, List.length_append, List.length_reverse, u32be_length]
  omega

theorem size_toBytes (l : List UInt8) : (ByteArray.mk l.toArray).size = l.length := by
  first | rfl | simp

theorem get!_toBytes (l : List UInt8) (k : Nat) : (ByteArray.mk l.toArray).get! k = (l[k]?).getD 0 := by
  rw [FlockSoundness.Discharge.Hm.get!_eq_getD, List.getD_eq_getElem?_getD]
  all_goals rfl

/-- **A little-endian `u64` reads only its eight bytes.** -/
theorem u64OfLE_congr {b b' : ByteArray} {o o' : Nat} (h : ∀ j, j < 8 → b.get! (o + j) = b'.get! (o' + j)) :
    _root_.Flock.u64OfLE b o = _root_.Flock.u64OfLE b' o' := by
  rw [FlockSoundness.Discharge.ZkView.u64OfLE_eq, FlockSoundness.Discharge.ZkView.u64OfLE_eq]
  exact ZKMal.Masked.foldl_congr_mem _ _ fun j hj _ => by
    simp only [h j (by have := (List.mem_range'_1.1 hj).2; omega)]

theorem answer_commit_coins (S : State) {a : ByteArray} (hb : byte? a 0 = some 2) :
    S.answer .commit a = S.coins a := by
  unfold State.answer
  split <;> simp_all

/-- **The session's first opening**, where the session goes on past it: the coins its stream's opening returns at the
session's link points, and what the prover receives, `Resp::Coins` of them. -/
theorem coins_session {st : State} {a : ByteArray} {C : Coins.Release} (hst : st.stop = none)
    (hC : st.session = some C) (h0 : C.opened.length = 0) (hn : (st.coins a).1.stop = none) :
    ∃ C' c, C.open Coins.SESSION C.hello.links a.data.toList = .ok (C', c) ∧
      (st.coins a).2 = ⟨(Coins.frame (c.length / 16) c).toArray⟩ := by
  have hr : reqAt st.plan C.hello.links C.opened.length = (Coins.SESSION, C.hello.links) := by
    unfold reqAt
    exact if_pos h0
  revert hn
  unfold State.coins
  rw [hst, hC]
  dsimp only
  rw [hr]
  dsimp only
  rcases ho : C.open Coins.SESSION C.hello.links a.data.toList with e | ⟨C', c⟩
  · intro hn
    first | cases hn | simp [State.refuse] at hn
  · intro _
    exact ⟨C', c, rfl, rfl⟩

/-- **The words the `Commit` links are the coins the frame is fed for the auditor's answer**: where the auditor
answered the session's first opening with coins the session goes on past, the words of what the prover receives
(`Frame.words` after its tag and count, as the loop links them) are the answer's first words at the session stream's
count (`Frames.feedOf`), within `Frames.COIN_MAX`. -/
theorem commit_words {st : State} {a : ByteArray} {C : Coins.Release} (hst : st.stop = none)
    (hC : st.session = some C) (h0 : C.opened.length = 0) (hb : byte? a 0 = some 2)
    (hn : (st.answer .commit a).1.stop = none) (hle : C.hello.links ≤ _root_.Flock.Firewall.Frames.COIN_MAX) :
    _root_.Flock.Firewall.Frame.words (st.answer .commit a).2 5 (((st.answer .commit a).2.size - 5) / 16) =
      _root_.Flock.Firewall.Frames.feedOf C.hello.links a.data.toList := by
  rw [answer_commit_coins st hb] at hn ⊢
  obtain ⟨C', cs, ho, hout⟩ := coins_session hst hC h0 hn
  obtain ⟨hm, hc⟩ := open_parts ho
  rw [ite_self] at hm hc
  have hcl : cs.length = 16 * C.hello.links := by
    rw [hc]
    exact parts_cs_length _ _ _
  revert hout
  revert hc
  revert hm
  generalize Coins.parts C.hello.links (Coins.depth C.hello.spec) (a.data.toList.drop 5) = P
  generalize (st.coins a).2 = out
  intro hm hc hout
  have hdrop : a.data.toList.drop 5 = P.join := by
    rw [hm, frame_drop5]
  have hJ : ∀ k, k < cs.length → P.join[k]? = cs[k]? := by
    intro k hk
    have e : P.join = cs ++ (P.tn ++ (P.y ++ P.sibs.flatten)) := by
      rw [hc]
      simp only [Coins.Parts.join, List.append_assoc]
    rw [e]
    exact List.getElem?_append_left hk
  have hsz : out.size = cs.length + 5 := by
    rw [hout, size_toBytes, frame_length]
  have hbyte : ∀ k, k < cs.length →
      out.get! (5 + k) = (ByteArray.mk (a.data.toList.drop 5).toArray).get! k := by
    intro k hk
    rw [hout, get!_toBytes, get!_toBytes, ← List.getElem?_drop, frame_drop5, hdrop, hJ k hk]
  rw [hsz, hcl, show (16 * C.hello.links + 5 - 5) / 16 = C.hello.links by omega]
  unfold _root_.Flock.Firewall.Frames.feedOf _root_.Flock.Firewall.Frame.words
  rw [Nat.min_eq_left hle]
  apply Array.map_congr_left
  intro i hi
  rw [Array.mem_range] at hi
  have e1 : _root_.Flock.u64OfLE out (5 + 16 * i) =
      _root_.Flock.u64OfLE (ByteArray.mk (a.data.toList.drop 5).toArray) (0 + 16 * i) :=
    u64OfLE_congr fun j hj => by
      rw [show 5 + 16 * i + j = 5 + (16 * i + j) by omega, show 0 + 16 * i + j = 16 * i + j by omega]
      exact hbyte _ (by omega)
  have e2 : _root_.Flock.u64OfLE out (5 + 16 * i + 8) =
      _root_.Flock.u64OfLE (ByteArray.mk (a.data.toList.drop 5).toArray) (0 + 16 * i + 8) :=
    u64OfLE_congr fun j hj => by
      rw [show 5 + 16 * i + 8 + j = 5 + (16 * i + 8 + j) by omega,
        show 0 + 16 * i + 8 + j = 16 * i + 8 + j by omega]
      exact hbyte _ (by omega)
  simp only [e1, e2]

/-! ## The history's link coins -/

open Claims.ZKMal in
/-- **The history's link coins are the auditor's answer to the run's second message**, at the loop's count: where
`V*` answered that message with `m`, the first coins past the history's `Hello` at the loop's count for opening `0`
are `m`'s first words there (`Frames.feedOf`). -/
theorem wordsAt_hist_two {W Rr : Type} (M : LoopModel W Rr) {c : Nat} {m : Claims.Msg} (hj : c < M.τ.length)
    (hc : c = 1) (h1 : (M.τ[c]'hj).2 = .msg m) :
    Loop.wordsAt M.x ((M.hist (c + 1)).tail.map fun e => ZKMal.CoinLayer.rd e.2) 0 =
      _root_.Flock.Firewall.Frames.feedOf (Loop.loopNr M.x 0) m := by
  subst hc
  rw [Loop.feedOf_rd]
  unfold LoopModel.hist
  revert hj h1
  generalize M.τ = τ
  intro hj h1
  rcases τ with _ | ⟨p0, _ | ⟨⟨v1, o1⟩, t⟩⟩
  · simp at hj
  · simp at hj
  · simp at h1
    subst h1
    simp [histOf, Loop.wordsAt, msgOf]

open Claims.ZKMal in
theorem wordsAt_hist_two_le {W Rr : Type} (M : LoopModel W Rr) {k : Nat} (hk : 2 ≤ k) :
    Loop.wordsAt M.x ((M.hist k).tail.map fun e => ZKMal.CoinLayer.rd e.2) 0 =
      Loop.wordsAt M.x ((M.hist 2).tail.map fun e => ZKMal.CoinLayer.rd e.2) 0 := by
  obtain ⟨k0, rfl⟩ : ∃ k0, k = k0 + 2 := ⟨k - 2, by omega⟩
  unfold LoopModel.hist
  generalize M.τ = τ
  rcases τ with _ | ⟨p0, _ | ⟨p1, t⟩⟩ <;> simp [histOf, Loop.wordsAt]

/-- **The history's link coins are the same at every history past the link.** -/
theorem wordsAt_hist_zero {W Rr : Type} (M : LoopModel W Rr) {c c' : Nat} (hc : 2 ≤ c) (hc' : 2 ≤ c') :
    Loop.wordsAt M.x ((M.hist c).tail.map fun e => ZKMal.CoinLayer.rd e.2) 0 =
      Loop.wordsAt M.x ((M.hist c').tail.map fun e => ZKMal.CoinLayer.rd e.2) 0 :=
  (wordsAt_hist_two_le M hc).trans (wordsAt_hist_two_le M hc').symm

/-! ## The session's entry linked -/

/-- **Session `sid`'s entry in the loop's frames at `σ` is linked to the history's link coins**, against the model's
run `M`: the coins of the model's history so far that answer its link, at the loop's count (`Loop.wordsAt` at opening
`0`). -/
def LinkOn {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (σ : LoopAt) : Prop :=
  ∀ rel : _root_.Flock.Firewall.Frames.Release, sessAt σ.R sid = some rel →
    rel.link = some (Loop.wordsAt M.x ((M.hist σ.c).tail.map fun e => ZKMal.CoinLayer.rd e.2) 0)

theorem linkOn_stay {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {root : ByteArray} {σ σ' : LoopAt}
    (hs : Stay sid root σ.R σ'.R) (hS : (sessAt σ.R sid).isSome) (hc : 2 ≤ σ.c) (hc' : 2 ≤ σ'.c)
    (h : LinkOn M sid σ) : LinkOn M sid σ' := by
  intro rel hrel
  rw [sessAt_stay hs hS] at hrel
  rw [h rel hrel, wordsAt_hist_zero M hc hc']

/-- **`LinkOn` is kept from one of the session's requests to the next**, past the link, at each level 0 bound to the
session, where what its rounds read of the loop stays (`Stays`: the session's entry at the bound level 0's root) and
the session's entry is set up. -/
theorem linkOn_stays {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {σ σ' : LoopAt} (hS : SessOn spec sts σ.R sid) (hs : Stays sid σ σ') (h2 : 2 ≤ σ.c) (hc : σ.c ≤ σ'.c)
    (h : ∀ name l0, _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 → LinkOn M sid σ) :
    ∀ name l0, _root_.Flock.Firewall.Level0.ofSession σ'.L0 sid name = some l0 → LinkOn M sid σ' := by
  intro name l0 hl
  rw [hs.1 name] at hl
  obtain ⟨rel, hr, -⟩ := hS
  exact linkOn_stay (hs.2 name l0 hl) (by rw [hr]; rfl) h2 (by omega) (h name l0 hl)

/-- Past the link the session's `Hello` left. -/
theorem sent_of_two {sid : Nat} {spec : SessionSpec} {seen : List String} {st : State} {pl : Placed} {σ : LoopAt}
    (hF : LoopFacts sid spec seen st pl σ) (h : 2 ≤ σ.c) : st.sent = true := by
  cases hs : st.sent
  · exact absurd (hF.start hs) (by omega)
  · rfl

/-- **Past the `Commit`, where the session goes on, its entry is linked to the history's link coins**: the auditor
answered the `Commit` with coins, as `V*` answered the run's link (`LoopModel.Ans`), and the loop linked the session to
the words of what the prover receives (`LoopRead`), which are those coins at the session's link points
(`commit_words`), the setup's (`LinksOk`), the loop's count for opening `0`, within `Frames.COIN_MAX` words. -/
theorem linkOn_commit {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {nonce : ByteArray} {seen : List String} {st : State} {pl : Placed} {e : Env} {σ σ' : LoopAt} {s0 : FlockDeploy.Ins}
    (hx : FlockDeploy.setupOf true M.x = some (s0, spec, sts))
    (hpm : spec.points * spec.mPts ≤ _root_.Flock.Firewall.Frames.COIN_MAX) (hsl : slot st pl = some .commit)
    (hF : LoopFacts sid spec seen st pl σ) (hL : LinksOk spec st)
    (hra : LoopRead M sid spec sts nonce .commit st pl e σ σ')
    (hgo : ((stepK true st pl (some (tagOf .commit)) (.ok .commit) e).go &&
      (stepK true st pl (some (tagOf .commit)) (.ok .commit) e).st.stop.isNone) = true) :
    LinkOn M sid σ' := by
  obtain ⟨⟨hlt, hans⟩, -, -, -, hcl, -, hcc⟩ := hra
  obtain ⟨hs1, -, hop⟩ := slot_commit hsl
  have hc1 : σ.c = 1 := by
    have := hF.count hs1
    omega
  obtain ⟨hg, hn⟩ := go_live hgo
  obtain ⟨C, hC⟩ := Option.isSome_iff_exists.mp (hF.reach.sess hs1)
  have h0 : C.opened.length = 0 := (opened_of hC).symm.trans hop
  have hlinks : C.hello.links = spec.points * spec.mPts := hL.2 C hC
  have hloop : Loop.loopNr M.x 0 = spec.points * spec.mPts := by
    show (Loop.reqOf M.x 0).2 = _
    rw [Loop.reqOf_eq hx 0]
    all_goals rfl
  rcases hr : st.request pl .commit with _ | ⟨st', o⟩
  · rcases stepK_frame_ans (by simp) (by simp) (by simp) (by simp) hr hg with hst | ⟨a, ha, hst, -⟩
    · exact absurd hn (Option.isSome_iff_ne_none.mp hst)
    · rw [hst] at hn
      have hn' : (st.answer .commit a).1.stop = none := by
        rw [← afterAnswer_coin hn]
        exact hn
      have hb : byte? a 0 = some 2 := by
        by_cases hb : byte? a 0 = some 2
        · exact hb
        · rw [answer_commit, ite_eq_right hb] at hn'
          exact absurd hn' (Option.isSome_iff_ne_none.mp (noCoins_stop _ _ _))
      intro rel hrel
      rw [hcl a ha hb hn' rel hrel, commit_words hF.live hC h0 hb hn' (by rw [hlinks]; exact hpm), hcc,
        wordsAt_hist_two M hlt hc1 (hans a ha hb), hloop, hlinks]
  · rw [stepK_refused (by decide) (by decide) (by decide) hr] at hn
    exact absurd hn (Option.isSome_iff_ne_none.mp (request_stop hF.live hr))

end Claims.ZKServer.Relay
