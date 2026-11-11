module

public import Proofs.Flock.ZkAny.RelayLoop

/-!
# A frame round's slot at the deployed loop

At a frame round's slot `envPre_inv`'s step reads the worker's value at the model's history (`LoopRead`'s round clause).
Here its case, given what the loop's frames and the model hold there:

* `slot_round_open`: a round's slot is for a stream whose `Open` left. `request_round_slot`: before a stop the session
  lets a round at its slot through, the coin check (`coinsOf`) and the place (`State.place`) both passing.
* `afterAnswer_coins`, `stepK_round_go`: a frame round the session goes on past released, the auditor answered it
  coins the session took, and the rep's frame took them.
* `RoundFacts`: what the loop's frames and the model hold at a frame round's slot, `envOk_round_run`'s premises and
  every other rep stream's rep another. **`slotCase_round`**: the frame round's slot case, given `RoundFacts` there and
  `J` kept.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session
open _root_.Flock.Firewall hiding State
open _root_.Flock (SessionSpec Stmt)
open _root_.Flock.Firewall.Frame (Masker)
open _root_.Flock.Firewall.Frames (Sessions Rep)

/-- **A round's slot is for a stream whose `Open` left.** -/
theorem slot_round_open {S : State} {P : Placed} {s : String} {n : Nat} (h : slot S P = some (.round s n)) :
    (S.labels.extract 0 P.opens).any (·.1 == s) = true := by
  unfold slot at h
  split at h; · simp at h
  split at h; · simp at h
  split at h; · simp at h
  split at h; · simp at h
  split at h; · simp at h
  split at h
  · split at h
    · rename_i hany
      simp only [Option.some.injEq, Session.Kind.round.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact hany
    · simp at h
  · repeat' split at h
    all_goals simp at h

/-- **Before a stop the session lets a round at its slot through**: the round is the schedule's next opening, so the
coin check passes, and its stream's `Open` left, so its place does. -/
theorem request_round_slot {S : State} {P : Placed} {s : String} {n : Nat} (hR : Reach S P) (h : S.stop = none)
    (hsl : slot S P = some (.round s n)) : S.request P (.round s n) = none := by
  obtain ⟨hsent, hj, hlink, hq⟩ := slot_round hsl
  obtain ⟨C, hC⟩ := Option.isSome_iff_exists.mp (hR.sess hsent)
  rw [opened_of hC] at hj hq
  have hlt : C.opened.length - 1 < S.plan.size := (Array.getElem?_eq_some_iff.mp hq).1
  have hg : S.plan.getD (C.opened.length - 1) ("", 0) = (s, n) := by
    rw [Array.getD_eq_getD_getElem?, hq, Option.getD_some]
  have hc : coinsOf S (.round s n) = none := by
    have hgt : ¬ C.opened.length > S.plan.size := by omega
    simp [coinsOf, hC, reqAt, hj, hg, hgt, show (Session.Kind.round s n == .finish) = false from rfl,
      show (Session.Kind.round s n == .commit) = false from rfl]
  have hp : S.place P (.round s n) = none := by
    simp only [State.place, hlink, slot_round_open hsl, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
  have e : S.request P (.round s n) = match coinsOf S (.round s n) with
      | some r => some r
      | none => (S.place P (.round s n)).map fun why => S.refuse s!"{Coins.STOP}{why}" := by
    unfold State.request; rw [h]; rfl
  rw [e]
  simp only [hc, hp, Option.map_none]

/-- **Coins the session took that the session goes on past, the round's rep's frame took.** -/
theorem afterAnswer_coins {st : State} {s : String} {n : Nat} {a : ByteArray} {e : Env} (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round s n) a).1.stop = none) (hn : (afterAnswer st (.round s n) a e).stop = none) :
    e.coins = .ok () := by
  rcases hc : e.coins with why | ⟨⟩
  · revert hn
    unfold afterAnswer
    simp [hb, hs, hc, refuse_stop]
  · rfl

/-- **A frame round at its slot the session goes on past released**, and the auditor answered it coins that the session
took and the rep's frame took. -/
theorem stepK_round_go {st : State} {pl : Placed} {s : String} {n : Nat} {e : Env} (hR : Reach st pl)
    (h : st.stop = none) (hs : slot st pl = some (.round s n))
    (hgo : ((stepK true st pl (some (tagOf (.round s n))) (.ok (.round s n)) e).go &&
      (stepK true st pl (some (tagOf (.round s n))) (.ok (.round s n)) e).st.stop.isNone) = true) :
    (∃ o, e.round = .ok o) ∧ ∃ a, e.ans = some a ∧ byte? a 0 = some 2 ∧
      (st.answer (.round s n) a).1.stop = none ∧ e.coins = .ok () := by
  obtain ⟨hg, hn⟩ := go_live hgo
  rw [stepK_frame (by simp) (by simp) (by simp) (by simp) (request_round_slot hR h hs)] at hg hn
  rcases hro : e.round with why | o
  · have hfo : frameOf st (.round s n) e = .error (stopOf why) := by simp only [frameOf, hro]
    rw [hfo] at hn
    change (st.refuse (stopOf why)).1.stop = none at hn
    rw [refuse_stop] at hn
    cases hn
  · have hfo : frameOf st (.round s n) e = .ok (roundFrame s o n) := by simp only [frameOf, hro]
    rw [hfo] at hg hn
    rcases hae : e.ans with _ | a
    · rw [hae] at hg
      cases hg
    · rw [hae] at hn
      change (afterAnswer st (.round s n) a e).stop = none at hn
      have hn' : (st.answer (.round s n) a).1.stop = none := afterAnswer_coin hn ▸ hn
      have hb : byte? a 0 = some 2 := by
        by_cases hb : byte? a 0 = some 2
        · exact hb
        · rw [answer_round, ite_eq_right hb] at hn'
          exact absurd hn' (Option.isSome_iff_ne_none.mp (noCoins_stop _ _ _))
      exact ⟨⟨o, rfl⟩, a, rfl, hb, hn', afterAnswer_coins hb hn' hn⟩

/-- A frame at one rep's key is at no other rep's. -/
theorem key_ne {sid i r i' r' : Nat} {p : Rep} (hk : keyOf p = (sid, i, r)) (hne : (i', r') ≠ (i, r)) :
    keyOf p ≠ (sid, i', r') := by
  rw [hk]
  intro heq
  apply hne
  cases heq
  rfl

/-- **What the loop's frames and the model hold at a frame round's slot** of session `sid`, the loop at `σ`, for the
round of `n` coins of `stream` the loop ran on table `i`'s rep `r` at the domain frame `dom`, its content the worker's
value `v` at the model's history so far: the model's coin count there is `n`, its round there masked, not its last,
with a next message, and `v` crosses where it fails to pass; the rep has a frame, which `roundPlace` readies at the
model's schedule, configuration and masking state but for `stopped`, as the stream's first round where `seen` has no
round of it, with a stop exactly where the frames have one; the frames have stopped the session exactly where the
model's run has; and every other rep stream of the setup is on another rep. -/
structure RoundFacts {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (spec : SessionSpec) (seen : List String)
    (stream : String) (n i r : Nat) (dom : ByteArray) (v : Claims.Msg) (σ : LoopAt) : Prop where
  coins : M.L.coins M.x (M.hist σ.c).length = n
  kind : M.L.kindAt M.x (M.hist σ.c).length = .masked
  next : (M.hist σ.c).length + 1 < M.nN M.x
  last : (M.hist σ.c).length < M.L.last M.x
  cross : M.L.cross M.x (M.hist σ.c).length M.w M.ρ (M.hist σ.c) v = false →
    M.L.pass M.x M.w M.ρ (M.hist σ.c) v = false
  frame : (frameAt σ.R sid i r).isSome
  place : ∃ (m : Masker) (q : Rep) (f : Bool) (nd : Option String), M.L.answered M.x M.ρ (M.hist σ.c) = .ok m ∧
    _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i r dom ⟨v.toArray⟩ n =
      .ok (q, .sched ⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c)
        (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c)))).toArray⟩, f, nd) ∧
    nd.isSome = σ.R.stopped sid ∧ q.cfg = M.L.cfg M.x M.ρ (M.hist σ.c) ∧ unstop q.st.m = unstop m ∧
    f = !seen.contains stream
  stopped : σ.R.stopped sid = !M.L.toSrv.liveBefore M.wk M.x M.w M.ρ M.κ (M.hist σ.c)
  other : ∀ i' r' s', (i', r', s') ∈ spec.repStreams.toList → s' ≠ stream → (i', r') ≠ (i, r)

/-- **The frame round's slot case**: where each of the run's messages is the deferred release's (`LoopModel.Own`),
`RoundFacts` holds at each request at the slot as the loop answers it, the run having a message there, the rep's frame
at the model's configuration and the round out, and `J` is kept there, what the relay read for the round is the
model's next message, and `LoopP`
holds at the session's next request, one model message on: the round's rep frame's puts touch no other rep's frame,
so each rep of a stream with no round yet stays fresh. -/
theorem slotCase_round {W Rr : Type} {M : LoopModel W Rr} {J : List String → State → LoopAt → Prop} {sid : Nat}
    {spec : SessionSpec} {sts : Array Stmt} {nonce : ByteArray} {stream : String} {n : Nat} (hOwn : M.Own)
    (hA : ∀ (seen : List String) (st : State) (pl : Placed) (e : Env) (σ σ' : LoopAt) (i r : Nat) (dom : ByteArray)
      (v : Claims.Msg), slot st pl = some (.round stream n) → LoopFacts sid spec seen st pl σ → J seen st σ →
      σ.c < M.τ.length → M.wk M.x M.w M.ρ (M.hist σ.c) 0 = some (v, false) →
      RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R →
      (∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) → SaltsSrc M sid σ →
      InnerSrc M sid i r σ → StartOk sts sid i r σ → (∃ o, e.round = .ok o) →
      RoundFacts M sid spec seen stream n i r dom v σ)
    (hJ : LoopKept M sid spec sts nonce J (.round stream n)) :
    SlotCase spec (LoopP M J sid spec sts nonce) (.round stream n) := by
  intro seen st pl e rest ms hsl hP
  obtain ⟨σ, hms, hF, hj, hRd, hset⟩ := hP
  obtain ⟨hsent, -, -, -⟩ := slot_round hsl
  cases hRd with
  | stop h =>
    rw [hsl] at h
    cases h
  | step k' hk' hra hrest =>
    obtain rfl : Session.Kind.round stream n = k' := Option.some.inj (hsl.symm.trans hk')
    have hJ' := hJ seen st pl e σ _ hsl (hset e List.mem_cons_self) hF hj hra
    obtain ⟨⟨hlt, -⟩, ⟨i, r, dom, v, hv, hr, hpc, hin, hsk⟩, hcc, -, hsal⟩ := hra
    have hRA := fun o (ho : e.round = .ok o) =>
      hA seen st pl e σ _ i r dom v hsl hF hj hlt hv hr hpc hsal hin hsk ⟨o, ho⟩
    have hreq := request_round_slot hF.reach hF.live hsl
    refine ⟨fun _ => ?_, fun hgo => ?_⟩
    · intro c hc
      have hRA := hRA c hc
      obtain ⟨m, q, f, nd, hm, hrp, hnd, hcfg, hu, hf⟩ := hRA.place
      have hlt' : σ.c < M.msgs.length := by simpa [LoopModel.msgs] using hlt
      have hhead : ms.headD [] = M.L.toSrv.nextD M.L.fillOf M.nN M.wk M.x M.w M.ρ M.κ (M.hist σ.c) := by
        rw [hms, List.drop_eq_getElem_cons hlt', List.headD_cons]
        simp only [LoopModel.msgs, List.getElem_map]
        exact hOwn σ.c hlt
      rw [hhead]
      rw [← hRA.coins] at hr hreq hrp
      exact envOk_round_run M.L M.nN M.wk M.x M.w M.ρ M.κ (M.hist σ.c) hRA.kind hRA.next hRA.last hv hRA.cross hm
        hr hreq hRA.frame hrp hnd hcfg hu hRA.stopped hF.setup hsent hf (k := n) c hc
    · obtain ⟨⟨o, hro⟩, a, hae, hb, hsa, hco⟩ := stepK_round_go hF.reach hF.live hsl hgo
      have hRA := hRA o hro
      obtain ⟨_, _, _, _, -, hrp, -⟩ := hRA.place
      obtain ⟨p₁, hput⟩ := roundStep_put_some hrp
      obtain ⟨R₁, R₂, R₂', p', h01, h12, hcs, h22, h23⟩ := hr.coins hreq hro hput hae hb hsa hco
      have hk : (Except.ok (Session.Kind.round stream n) : Except String Session.Kind) ≠ .ok .hello := by
        intro h
        cases h
      refine ⟨_, ?_, ⟨?_, ?_, (go_live hgo).2, stepK_reach hF.reach hF.live hsl hgo, ?_, ?_⟩, hJ' hgo, hrest hgo,
        fun e' he' => hset e' (List.mem_cons_of_mem _ he')⟩
      · simp only [isModel, ↓reduceIte]
        rw [hcc, hms, List.tail_drop]
      · exact stepK_setup (hset e List.mem_cons_self) hF.setup
      · intro i' r' s' hmem hseen
        have hs' : s' ≠ stream ∧ seen.contains s' = false := by
          simpa only [seenAfter, List.contains_cons, Bool.or_eq_false_iff, beq_eq_false_iff_ne] using hseen
        have hne := hRA.other i' r' s' hmem hs'.1
        have hK : Kept sid i' r' σ.R _ :=
          (((((h01.1 i' r').step (.put _ _ (key_ne (roundStep_key hput) hne))).trans (h12.1 i' r')).trans
            (h22.1 i' r')).step (.put _ _ (key_ne (coinsStep_key hcs) hne))).trans (h23.1 i' r')
        exact freshAt_kept hK (hF.fresh i' r' s' hmem hs'.2)
      · intro h
        rw [stepK_sent hk, hsent] at h
        cases h
      · intro _
        rw [hcc, hF.count hsent, stepK_coin_opened hF.reach hF.live hsl (.inr ⟨stream, n, rfl⟩) hgo]

end Claims.ZKServer.Relay
