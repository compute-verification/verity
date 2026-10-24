import Proofs.Flock.SessionLoop.FinishParts
import Proofs.Flock.Claims.ZK.FrameTaken

/-!
# (d) `FinRepRounds`: the reps' rounds

* **`LeanRelease.msg_of_liveRoundEq`**: on a live history of `refK x` openings whose openings pass and whose messages are
  the deferred release's own (`ZkServer.OwnHist`), each frame round's message (rounds `2` to `refK x`) is the session's
  switched message at the coins the openings before it open, without its leading domain frame: `LiveRoundEq` at that
  prefix, which is live (`live_take`), takes its coins (`refusesCoins_of_live`) and passes its openings (`allOk_eval`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal

section Out

open _root_.Flock (V)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker INNER_ROUNDS roundFree)

set_option maxHeartbeats 1000000 in
/-- **A round the frame takes releases its squeeze last**: `roundFree`'s loop returns only at the squeeze of the `n`
coins it was asked for, which it appends to the round's release. -/
theorem roundFree_out {c : Cfg} {s : Masker} {xb : ByteArray} {n : ℕ} {q : ByteArray × Masker × Array ℕ}
    (h : roundFree c s xb n = .ok q) : ∃ o k, q.1 = o ++ header 4 k n := by
  by_cases hs : s.inner = some INNER_ROUNDS
  · unfold roundFree at h
    rw [ite_eq_left (by simp [hs])] at h
    cases h
  unfold roundFree at h
  simp only [show (s.inner == some INNER_ROUNDS) = false from by simpa using hs, Bool.false_eq_true, ↓reduceIte,
    Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  obtain ⟨y, hy, hk⟩ := bindV_ok h
  have hP := forIn_ok_inv (fun x => ∀ r, x.1 = some r → ∃ o k, r.1 = o ++ header 4 k n) _ ?hg _ _
    (fun r hr => by cases hr) y hy
  · clear h hy
    revert hk hP
    obtain ⟨o, rest⟩ := y
    cases o with
    | none => intro hk _; cases hk
    | some r => intro hk hP; cases hk; exact hP _ rfl
  · clear h hy hk
    intro i x z hx hz r hr
    beta_reduce at hz
    by_cases h1 : x.2.2.2.2 + 16 > xb.size
    · rw [ite_eq_left h1] at hz
      rw [ite_eq_iff] at hz
      rcases hz with ⟨_, hz⟩ | ⟨_, hz⟩ <;> obtain ⟨_, h, _⟩ := bindV_ok hz <;> cases h
    · rw [ite_eq_right h1] at hz
      split at hz
      all_goals
        repeat' (first
          | (obtain ⟨_, _, hz⟩ := bindV_ok hz; try beta_reduce at hz)
          | (rw [ite_eq_iff] at hz; rcases hz with ⟨_, hz⟩ | ⟨_, hz⟩)
          | (split at hz))
      all_goals first | (cases hz; cases hr; done) | (cases hz; done) | contradiction | skip
      all_goals (cases hz; cases hr; exact ⟨_, _, rfl⟩)

end Out

namespace LeanRelease

/-- **A release the frame takes ends in the squeeze of its round's coins**: the frame's round returns its release with
the squeeze of the `coins x h.length` it was asked for last. -/
theorem release_out_of_takes {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (v : Msg)
    (ht : L.takes x ρ h v = true) : ∃ o k, L.release x ρ h false v =
      0 :: (o ++ _root_.Flock.Transcript.header 4 k (L.coins x h.length)).data.toList := by
  unfold takes at ht
  unfold release
  cases hs : L.step x ρ h false v with
  | error e => rw [hs] at ht; cases ht
  | ok p =>
    obtain ⟨out, m⟩ := p
    obtain ⟨m₀, -, hr⟩ := bindV_ok hs
    unfold _root_.Flock.Firewall.Frame.round at hr
    obtain ⟨q, hq, hr⟩ := bindV_ok hr
    obtain ⟨o, k, ho⟩ := roundFree_out hq
    obtain ⟨o', s', f'⟩ := q
    cases hr
    exact ⟨o, k, by simp only at ho; rw [ho]⟩

/-- **Each frame round of a `Finish` `FinishIs` reads is the session's switched message** at the coins the openings
before it open, without its leading domain frame. -/
theorem msg_of_liveRoundEq (S : RefSession) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg)
    (w : Flock.Wit) (hE : LiveRoundEq S vs L H wk x w) (hlast : L.last x = refK x + 1) (ν : S.Nonce) (t : S.Tape)
    (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hlen : ops.length = refK x)
    (hl : L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true)
    (hown : L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops)) (j : ℕ) (h2 : 2 ≤ j) (hj : j ≤ refK x) :
    (((m₀, ρ₀) :: ops)[j]'(by simp only [List.length_cons]; omega)).1 =
      dropDom (S.swMsgAt x (S.wt x w) (vs x w ν) t ((ops.take (j - 1)).map fun p => S.decOf p.2)) := by
  have hj' : j < ((m₀, ρ₀) :: ops).length := by simp only [List.length_cons]; omega
  have htake : ((m₀, ρ₀) :: ops).take j = (m₀, ρ₀) :: ops.take (j - 1) := by
    obtain ⟨i, rfl⟩ : ∃ i, j = i + 1 := ⟨j - 1, by omega⟩
    simp
  have hlj : (((m₀, ρ₀) :: ops).take j).length = j := by
    simp only [List.length_take, List.length_cons]
    omega
  have hk : L.kindAt x (((m₀, ρ₀) :: ops).take j).length = .masked := by
    rw [hlj]
    unfold kindAt
    split_ifs with h0 h1
    · omega
    · rw [hlast] at h1
      omega
    · rfl
  have hl' := L.live_take wk x w (ν, t) κ _ hl j
  obtain ⟨v, hv, -⟩ := L.live_some wk x w (ν, t) κ _ hk hl'
  have hu : L.toSrv.wkD L.fillOf wk κ x w (ν, t) (((m₀, ρ₀) :: ops).take j) 0 = some (v, false) := by
    simp only [ZkServer.wkD, hl', ↓reduceIte, hv]
    rfl
  have hr := L.refusesCoins_of_live wk x w (ν, t) κ _ hk hl'
  refine (hown j hj').trans ?_
  rw [L.frameD_toSrv L.fillOf wk x w (ν, t) κ _ hk hu]
  rw [htake] at hl' hr hv
  rw [htake]
  have hlen' : (ops.take (j - 1)).length = j - 1 := by
    simp only [List.length_take]
    omega
  refine hE ν t κ m₀ ρ₀ (ops.take (j - 1)) v (by rw [hlen']; omega) (by rw [hlen']; omega) hl' ?_ hr hv
  rw [allOk_eval] at hok ⊢
  intro i hi
  simp only [List.length_map] at hi
  have := hok i (by simp only [List.length_map]; omega)
  simpa [List.getElem_map, List.getElem_take] using this

/-- **Each frame round of a `Finish` `FinishIs` reads is the switched round encoding at the opened coins**, without its
leading domain frame: `msg_of_liveRoundEq`, whose prefix coins agree with the opened ones below the round
(`swMsgAt_take`, from `SwCausal`). -/
theorem msg_of_swCausal (S : RefSession) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg)
    (w : Flock.Wit) (hE : LiveRoundEq S vs L H wk x w) (hc : S.SwCausal vs) (hlast : L.last x = refK x + 1)
    (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hlen : ops.length = refK x)
    (hl : L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true)
    (hown : L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops)) (j : ℕ) (h2 : 2 ≤ j) (hj : j ≤ refK x) :
    (((m₀, ρ₀) :: ops)[j]'(by simp only [List.length_cons]; omega)).1 =
      dropDom (S.enc x (S.pad (K := refK x) (ops.map fun p => S.decOf p.2)) (j - 1)
        (S.swCutOf x (S.pad (ops.map fun p => S.decOf p.2)) (S.wt x w)
          (vs x w ν (S.pad (ops.map fun p => S.decOf p.2))) t)) := by
  refine (msg_of_liveRoundEq S vs L H wk x w hE hlast ν t κ m₀ ρ₀ ops hlen hl hok hown j h2 hj).trans ?_
  congr 1
  have hcs : (ops.map fun p => S.decOf p.2).length ≤ refK x := by simp [hlen]
  have e1 := S.ofFn_pad_take (K := refK x) (ops.map fun p => S.decOf p.2) hcs
  simp only [List.length_map, hlen, Nat.min_self] at e1
  have e : (ops.take (j - 1)).map (fun p => S.decOf p.2) =
      (List.ofFn (S.pad (K := refK x) (ops.map fun p => S.decOf p.2))).take (j - 1) := by
    calc (ops.take (j - 1)).map (fun p => S.decOf p.2) = (ops.map fun p => S.decOf p.2).take (j - 1) := by
          simp [List.map_take]
      _ = ((List.ofFn (S.pad (K := refK x) (ops.map fun p => S.decOf p.2))).take (refK x)).take (j - 1) := by
          rw [e1]
      _ = (List.ofFn (S.pad (K := refK x) (ops.map fun p => S.decOf p.2))).take (j - 1) := by
          rw [List.take_take]
          congr 1
          omega
  have hlt : j - 1 < refK x := by omega
  rw [e, S.swMsgAt_take vs x w ν hc t (S.pad (ops.map fun p => S.decOf p.2)) (j - 1) (by omega), ite_eq_left hlt]

/-- Rounds `2` to `refK x` are frame rounds, with the `Finish` one past the last coin. -/
theorem kindAt_mid {W R : Type} {L : LeanRelease W R} {x : Msg} (hlast : L.last x = refK x + 1) {j : ℕ}
    (h2 : 2 ≤ j) (hj : j ≤ refK x) : L.kindAt x j = .masked := by
  unfold kindAt
  split_ifs with h0 h1
  · omega
  · rw [hlast] at h1
    omega
  · rfl

/-- **A release the frame takes is a round message**, tagged `0`. -/
theorem release_cons_of_takes {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (v : Msg)
    (ht : L.takes x ρ h v = true) : ∃ b, L.release x ρ h false v = 0 :: b := by
  unfold takes at ht
  unfold release
  cases hs : L.step x ρ h false v with
  | error e => rw [hs] at ht; cases ht
  | ok p =>
    obtain ⟨out, m⟩ := p
    exact ⟨out.data.toList, rfl⟩

/-- **A live run's frame rounds are round messages**: on a live history whose messages are the deferred release's own,
each frame round's message is the Lean frame's release of the worker's round, which the frame took. -/
theorem msg_cons {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hl : L.toSrv.live wk x w ρ κ h = true)
    (hown : L.toSrv.OwnHist L.fillOf wk x w ρ κ h) (j : ℕ) (hj : j < h.length) (hk : L.kindAt x j = .masked) :
    ∃ b, (h[j]'hj).1 = 0 :: b := by
  have hlj : (h.take j).length = j := by
    simp only [List.length_take]
    omega
  have hk' : L.kindAt x (h.take j).length = .masked := by rw [hlj]; exact hk
  have hl' := L.live_take wk x w ρ κ h hl j
  obtain ⟨v, hv, -, -, ht⟩ := L.live_some wk x w ρ κ _ hk' hl'
  have hu : L.toSrv.wkD L.fillOf wk κ x w ρ (h.take j) 0 = some (v, false) := by
    simp only [ZkServer.wkD, hl', ↓reduceIte, hv]
    rfl
  obtain ⟨b, hb⟩ := L.release_cons_of_takes x ρ _ v ht
  refine ⟨b, (hown j hj).trans ?_⟩
  rw [L.frameD_toSrv L.fillOf wk x w ρ κ _ hk' hu]
  exact hb

end LeanRelease

namespace Loop

theorem enc_finS_eq {x : Msg} (c : Fin (refK x) → finS.Coin) (n : ℕ)
    (cut : ∀ j, ((finS.C x j).T (finS.cz x c j)).tab.Cut) :
    finS.enc x c n cut = Masked.encStop (Masked.encExe Masked.PE) x c n cut := rfl

/-- **The session's round `n ≥ 1` at the deployed tables**: `[]`, or `sessionCut`'s body `n` on the cut, answered by
the coins `c` and stopped right after it, tagged `0` (`encExe`, `encStop`). -/
theorem enc_finS {x : Msg} (c : Fin (refK x) → finS.Coin) {n : ℕ} (hn : n ≠ 0)
    (cut : ∀ j, ((finS.C x j).T (finS.cz x c j)).tab.Cut) :
    finS.enc x c n cut = [] ∨ ∃ s spec, FlockDeploy.setupOf true x = some (s, spec, Masked.stsOf x) ∧
      finS.enc x c n cut =
        0 :: (Masked.runCut x c n spec (Masked.tabSrcCut Masked.PE x (Masked.czExe Masked.PE x c) cut)).2[n]! := by
  rw [enc_finS_eq]
  unfold Masked.encStop
  split_ifs with hR
  · exact .inl rfl
  · unfold Masked.encExe
    rw [ite_eq_right hn]
    cases hsx : FlockDeploy.setupOf true x with
    | none => exact .inl rfl
    | some v =>
      obtain ⟨s, spec, sts⟩ := v
      have hst : Masked.stsOf x = sts := by
        unfold Masked.stsOf
        rw [hsx]
      subst hst
      dsimp only
      split_ifs with hlt
      · exact .inr ⟨s, spec, rfl, rfl⟩
      · exact .inl rfl

/-- **Each frame round of a `Finish` `FinishIs` reads is the cut session's body at the opened coins**: at the deployed
tables, message `j` (`2 ≤ j ≤ refK x`) is `sessionCut`'s body `j - 1` on the switched cut at the opened coins, stopped
right after it, tagged `0`, without its leading domain frame. It is a round message (`msg_cons`), so the session did not
stop before it (`enc_finS`). -/
theorem msg_runCut (vs : finS.SwVals) (L : LeanRelease Flock.Wit Tp) (H : Msg → Msg) (wk : L.toSrv.Worker)
    (x : Msg) (w : Flock.Wit) (hE : LeanRelease.LiveRoundEq finS vs L H wk x w) (hc : finS.SwCausal vs)
    (hlast : L.last x = refK x + 1) (ν : finS.Nonce) (t : finS.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (hlen : ops.length = refK x)
    (hl : L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk (finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true)
    (hown : L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops)) (j : ℕ) (h2 : 2 ≤ j) (hj : j ≤ refK x) :
    ∃ s spec, FlockDeploy.setupOf true x = some (s, spec, Masked.stsOf x) ∧
      (((m₀, ρ₀) :: ops)[j]'(by simp only [List.length_cons]; omega)).1 =
        dropDom (0 :: (Masked.runCut x (finCoins x ops) (j - 1) spec
          (Masked.tabSrcCut Masked.PE x (Masked.czExe Masked.PE x (finCoins x ops))
            (finS.swCutOf x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t))).2[j - 1]!) := by
  have e := LeanRelease.msg_of_swCausal finS vs L H wk x w hE hc hlast ν t κ m₀ ρ₀ ops hlen hl hok hown j h2 hj
  obtain ⟨b, hb⟩ := LeanRelease.msg_cons L wk x w (ν, t) κ _ hl hown j
    (by simp only [List.length_cons]; omega) (LeanRelease.kindAt_mid hlast h2 hj)
  rcases enc_finS (finCoins x ops) (n := j - 1) (by omega)
      (finS.swCutOf x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t) with h0 | ⟨s, spec, hsx, he⟩
  · have h' : dropDom (finS.enc x (finCoins x ops) (j - 1)
        (finS.swCutOf x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t)) = 0 :: b := e.symm.trans hb
    rw [h0] at h'
    simp [dropDom] at h'
  · exact ⟨s, spec, hsx, e.trans (congrArg dropDom he)⟩

end Loop

end Claims.ZKServer
