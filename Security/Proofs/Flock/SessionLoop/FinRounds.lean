import Proofs.Flock.SessionLoop.FinishParts
import Proofs.Flock.Claims.ZK.FrameTaken
import Proofs.Flock.Claims.ZK.RepLog
import Proofs.Flock.Claims.ZK.CutPrefix
import Proofs.Flock.Claims.ZK.FrameUnrefused
import Proofs.Flock.Claims.ZK.FrameInner

/-!
# (d) `FinRepRounds`: the reps' rounds

* **`LeanRelease.msg_of_liveRoundEq`**: on a live history of `refK x` openings whose openings pass and whose messages are
  the deferred release's own (`ZkServer.OwnHist`), each frame round's message (rounds `2` to `refK x`) is the session's
  switched message at the coins the openings before it open, without its leading domain frame: `LiveRoundEq` at that
  prefix, which is live (`live_take`), takes its coins (`refusesCoins_of_live`) and passes its openings (`allOk_eval`).
* **`LeanRelease.msg_of_swCausal`**: the same at the opened coins, from the session's causality (`SwCausal`).
* **`Loop.msg_runCut`**, **`Loop.runCut_body`**: that message is the cut session's body at the opened coins, which is
  the body the view rep's run sends (`CutPrefix.repCut_cut_agree`, `RepLog.repCut_rounds`).
* **`LeanRelease.msg_release`**: a frame round's message is a release ending in the squeeze of its coins, fewer than
  `2 ^ 64` (`release_out_of_takes`).
* **`Loop.viewRep_rounds`**, **`Loop.finRepRounds_of`**: the view rep's rounds are the auditor's (`streamOf`), so (d)
  holds wherever (a) to (c) do; **`Loop.finishIs_of_abc`**: (a) to (c) give `FinishIs`.
-/

namespace FlockComplete.CutPrefix

open Flock

theorem forall₂_length {α β : Type} {R : α → β → Prop} :
    ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ → l₁.length = l₂.length
  | _, _, .nil => rfl
  | _, _, .cons _ hf => by simp [forall₂_length hf]

theorem forall₂_getElem {α β : Type} {R : α → β → Prop} :
    ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ → ∀ i (h₁ : i < l₁.length) (h₂ : i < l₂.length),
      R l₁[i] l₂[i]
  | _, _, .nil, _, h₁, _ => absurd h₁ (by simp)
  | _, _, .cons hr _, 0, _, _ => hr
  | _, _, .cons _ hf, i + 1, h₁, h₂ => forall₂_getElem hf i (by simpa using h₁) (by simpa using h₂)

theorem forall₂_map {α β δ : Type} {R : α → β → Prop} (φ : α → δ) (ψ : β → δ) (hR : ∀ a b, R a b → φ a = ψ b) :
    ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ → l₁.map φ = l₂.map ψ
  | _, _, .nil => rfl
  | _, _, .cons hr hf => by simp [hR _ _ hr, forall₂_map φ ψ hR hf]

/-- **Two verdict loops in step**: where each element `g` returns at, `f` returns at a related value, the loops return
related lists. -/
theorem list_mapM_rel {α β γ : Type} {g : α → _root_.Flock.V β} {f : α → _root_.Flock.V γ} (R : β → γ → Prop)
    (h : ∀ a b, g a = .ok b → ∃ c, f a = .ok c ∧ R b c) :
    ∀ (xs : List α) (bs : List β), xs.mapM g = .ok bs → ∃ cs, xs.mapM f = .ok cs ∧ List.Forall₂ R bs cs
  | [], bs, hb => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at hb
    subst hb
    exact ⟨[], rfl, .nil⟩
  | a :: xs, bs, hb => by
    rw [List.mapM_cons] at hb ⊢
    cases hg : g a with
    | error e =>
      rw [hg] at hb
      simp [bind, Except.bind] at hb
    | ok b =>
      cases hm : xs.mapM g with
      | error e =>
        rw [hg, hm] at hb
        simp [bind, Except.bind] at hb
      | ok bs' =>
        rw [hg, hm] at hb
        simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq] at hb
        subst hb
        obtain ⟨c, hc, hR⟩ := h a b hg
        obtain ⟨cs, hcs, hF⟩ := list_mapM_rel R h xs bs' hm
        refine ⟨c :: cs, ?_, .cons hR hF⟩
        rw [hc, hcs]
        rfl

end FlockComplete.CutPrefix

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
the squeeze of the `coins x h.length` it was asked for last, a count its 64-bit length field holds (`roundFree_sq`). -/
theorem release_out_of_takes {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (v : Msg)
    (ht : L.takes x ρ h v = true) : L.coins x h.length < 2 ^ 64 ∧ ∃ o k, L.release x ρ h false v =
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
    have hlt := (roundFree_sq (Loop.roundFree_ok_inner hq) hq).2.2.1
    obtain ⟨o', s', f'⟩ := q
    cases hr
    refine ⟨?_, o, k, by simp only at ho; rw [ho]⟩
    rw [← hlt]
    exact UInt64.toNat_lt _

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

/-- **A live run's frame rounds end in their coins' squeeze**: `msg_cons`, whose release the frame took ends in the
squeeze of round `j`'s `coins x j`, a count below `2 ^ 64` (`release_out_of_takes`). -/
theorem msg_release {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hl : L.toSrv.live wk x w ρ κ h = true)
    (hown : L.toSrv.OwnHist L.fillOf wk x w ρ κ h) (j : ℕ) (hj : j < h.length) (hk : L.kindAt x j = .masked) :
    L.coins x j < 2 ^ 64 ∧ ∃ o k,
      (h[j]'hj).1 = 0 :: (o ++ _root_.Flock.Transcript.header 4 k (L.coins x j)).data.toList := by
  have hlj : (h.take j).length = j := by
    simp only [List.length_take]
    omega
  have hk' : L.kindAt x (h.take j).length = .masked := by rw [hlj]; exact hk
  have hl' := L.live_take wk x w ρ κ h hl j
  obtain ⟨v, hv, -, -, ht⟩ := L.live_some wk x w ρ κ _ hk' hl'
  have hu : L.toSrv.wkD L.fillOf wk κ x w ρ (h.take j) 0 = some (v, false) := by
    simp only [ZkServer.wkD, hl', ↓reduceIte, hv]
    rfl
  obtain ⟨hlt, o, k, hb⟩ := L.release_out_of_takes x ρ _ v ht
  rw [hlj] at hlt hb
  refine ⟨hlt, o, k, (hown j hj).trans ?_⟩
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

/-- **A recorded round is the auditor's**: round `j` of a rep whose log started at `L` bodies (`RepLog.RoundAt`), whose
history message `m` is body `L + j` tagged `0` without its leading domain frame and is a release ending in the squeeze of
`n'` coins, has `n'` coins, `coin (L + j)`'s, no digest, and as bytes the message past its tag, after the stream's domain
frame at round `0`. -/
theorem round_of_logs {coin : ℕ → ℕ → _root_.Flock.F128} {B : ℕ} {d : String} {L : ℕ} {l : Array Msg} {j : ℕ}
    {r : _root_.Flock.Round} (hr : FlockComplete.RepLog.RoundAt coin B d.toUTF8 L l j r) (hB : B < 2 ^ 64)
    (hd : d.toUTF8.size < 2 ^ 64) {m : Msg} (hm : m = dropDom (0 :: l[L + j]!)) {o : ByteArray} {k' : UInt8}
    {n' : ℕ} (hn' : n' < 2 ^ 64) (hrel : m = 0 :: (o ++ _root_.Flock.Transcript.header 4 k' n').data.toList) :
    r = { n := n', sha := .empty, coins := (Array.range n').map (coin (L + j)),
          bytes := some (if j = 0 then _root_.Flock.Firewall.Frames.domainFrame d ++ ⟨m.tail.toArray⟩
            else ⟨m.tail.toArray⟩) } := by
  obtain ⟨-, hle, hsha, hcoins, c, hc, hl, p, kind, hcp, hlead⟩ := hr
  obtain ⟨hlt, hle'⟩ := Array.getElem?_eq_some_iff.1 hl
  rw [getElem!_pos l (L + j) hlt, hle', Claims.toMsg, hcp] at hm
  have hsq : (FlockSoundness.Refine.Frame.sq kind r.n).enc = (_root_.Flock.Transcript.header 4 kind r.n).data.toList := by
    simp [FlockSoundness.Refine.Frame.enc, FlockSoundness.Refine.Frame.op, FlockSoundness.Refine.Frame.kind,
      FlockSoundness.Refine.Frame.len, FlockSoundness.Refine.Frame.pay]
  have htail : ∃ P, m.tail = P ++ (_root_.Flock.Transcript.header 4 kind r.n).data.toList ∧
      (if j = 0 then _root_.Flock.Firewall.Frames.domainFrame d ++ ⟨m.tail.toArray⟩ else ⟨m.tail.toArray⟩) = c := by
    unfold FlockComplete.RepLog.Lead at hlead
    by_cases hj : j = 0
    · rw [ite_eq_left hj] at hlead ⊢
      obtain ⟨rr, rfl⟩ := hlead
      have e := dropDom_dom d.toUTF8 [] (rr ++ (FlockSoundness.Refine.Frame.sq kind r.n).enc) hd
      simp only [FlockSoundness.Refine.encs, List.flatMap_cons, List.flatMap_nil, List.append_nil,
        List.nil_append] at e
      rw [List.append_assoc, e] at hm
      subst hm
      refine ⟨rr, by rw [List.tail_cons, hsq], ?_⟩
      apply ByteArray.ext
      apply Array.ext'
      rw [hcp]
      simp [ByteArray.data_append, _root_.Flock.Firewall.Frames.domainFrame, FlockSoundness.Refine.Frame.enc,
        FlockSoundness.Refine.Frame.op, FlockSoundness.Refine.Frame.kind, FlockSoundness.Refine.Frame.len,
        FlockSoundness.Refine.Frame.pay]
    · rw [ite_eq_right hj] at hlead ⊢
      have hh : (p ++ (FlockSoundness.Refine.Frame.sq kind r.n).enc)[0]! ≠ 1 := by
        cases p with
        | nil =>
          rw [List.nil_append, hsq, FlockSoundness.Refine.header_data, List.cons_append, List.getElem!_cons_zero]
          decide
        | cons a p => simpa using hlead
      have e : dropDom (0 :: (p ++ (FlockSoundness.Refine.Frame.sq kind r.n).enc)) =
          0 :: (p ++ (FlockSoundness.Refine.Frame.sq kind r.n).enc) := by
        unfold dropDom
        exact ite_eq_right fun h => hh h.2
      rw [e] at hm
      subst hm
      refine ⟨p, by rw [List.tail_cons, hsq], ?_⟩
      apply ByteArray.ext
      apply Array.ext'
      rw [hcp]
      simp
  obtain ⟨P, hP, hbytes⟩ := htail
  have hm' : m.tail = o.data.toList ++ (_root_.Flock.Transcript.header 4 k' n').data.toList := by
    rw [hrel, List.tail_cons, ByteArray.data_append, Array.toList_append]
  rw [hP] at hm'
  obtain ⟨-, hH⟩ := List.append_inj' hm' (by rw [FlockSoundness.Refine.Frame.header_length,
    FlockSoundness.Refine.Frame.header_length])
  obtain ⟨-, -, hn⟩ := FlockSoundness.Refine.Frame.header_inj (by omega) hn' hH
  obtain ⟨rn, rsha, rcoins, rbytes⟩ := r
  dsimp only at hn hsha hcoins hc
  subst hn hsha hcoins hc
  rw [hbytes]

/-- **(d)'s count**: in the masked scope, rep stream `k`'s run from the views, from a history at its first round, sends
its rep's bodies (`repCut_eqAt` against itself), as many as its recorded rounds (`repRoundsT_eq`), so it ends at rep
`k + 1`'s first round. -/
theorem viewRep_size (vs : finS.SwVals) {x : Msg} {w : Flock.Wit}
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    (ν : finS.Nonce) (t : finS.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) {s0}
    {spec : _root_.Flock.SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : _root_.Flock.V _) = .ok schs)
    (cms : Array FlockComplete.Commit) {k : ℕ} (hk : k < 2 * spec.tables.size) {extras : Array _root_.Flock.Claim}
    (hex : (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras)
    (ic : FlockComplete.InnerCommit) {l : Array Msg} (hl : l.size = Masked.repBase x k)
    {p : _root_.Flock.Zk.ZkProof} {tr : _root_.Flock.Transcript} {l' : Array Msg}
    (h : (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k extras ic).go l = (.ok (p, tr), l')) :
    l'.size = Masked.repBase x (k + 1) := by
  have hx : Masked.Live x := Masked.live_of hsc hR
  obtain ⟨-, -, hts, hall⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hJ : Masked.jOf x = (Masked.stsOf x).size := by
    unfold Masked.jOf
    exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  have hi : k / 2 < Masked.jOf x := by omega
  have hi' : k / 2 < (Masked.stsOf x).size := by omega
  obtain ⟨schs', hs', hschs⟩ := Masked.schedsPZk_live hx
  rw [hs] at hs'
  cases hs'
  have hsch := hschs ⟨k / 2, hi⟩
  have hst : (Masked.stsOf x)[k / 2]! = Masked.stAt x ⟨k / 2, hi⟩ := getElem!_pos _ _ _
  have htab : (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! =
      _root_.Flock.Setup.ofCircuit (Masked.stsOf x)[k / 2]! := by
    rw [getElem!_pos (FlockDeploy.tabsOf (Masked.stsOf x)) (k / 2) (by simpa [FlockDeploy.tabsOf] using hi'),
      getElem!_pos (Masked.stsOf x) (k / 2) hi']
    simp [FlockDeploy.tabsOf]
  have hS : FlockComplete.CutEq.SchOk (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! schs[k / 2]!.1 := by
    rw [htab, hsch, hst]
    exact Masked.schOk_of (Masked.zkTab x ⟨k / 2, hi⟩)
  have hB : l.size ≤ refK x := by
    rw [hl, Masked.repBase_eq hsx, Masked.refK_eq hsx, ← FlockComplete.ZkTables.repStart_last]
    exact FlockComplete.ZkTables.repStart_mono _ _ (by omega)
  have hA : FlockComplete.CutEq.RepAgree (refK x) l.size (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!
      schs[k / 2]!.1 extras ((finSrc vs x w ν t ops (k / 2)).rep (k % 2))
      ((finSrc vs x w ν t ops (k / 2)).rep (k % 2)) := by
    constructor <;> intros <;> simp
  unfold viewRep at h
  obtain ⟨he, -, -⟩ := (FlockComplete.CutEq.repCut_eqAt (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x)
    _ _ hS (k % 2) _ cms[k / 2]! extras ic _ _ _ _ _ rfl hB hA).2 _ _ h
  have hv : _root_.Flock.Zk.schedule (Masked.stsOf x)[k / 2]!.m = .ok schs[k / 2]! := by
    rw [hsch, hst]
    exact (Masked.zkTab x ⟨k / 2, hi⟩).hz
  have hes : extras.size = spec.points * (Masked.stsOf x)[k / 2]!.regions.size := by
    rw [htab] at hex
    rw [FlockComplete.JsonRT.extraClaims_size hex]
    simp [Nat.mul_comm, viewPoints]
  have hm : 6 ≤ (Masked.stsOf x)[k / 2]!.m := by
    rw [getElem!_pos (Masked.stsOf x) (k / 2) hi']
    have := (hall (k / 2) hi').1
    omega
  rw [he, hl, Masked.repBase_eq hsx, Masked.repBase_eq hsx, FlockComplete.ZkTables.repStart_succ,
    ← FlockComplete.ZkTables.repRoundsT_eq hv hes, htab]
  have e6 : (_root_.Flock.Setup.ofCircuit (Masked.stsOf x)[k / 2]!).fold.kLog = (Masked.stsOf x)[k / 2]!.c.kLog := rfl
  have e7 : (_root_.Flock.Setup.ofCircuit (Masked.stsOf x)[k / 2]!).m = (Masked.stsOf x)[k / 2]!.m := rfl
  simp only [FlockComplete.CutEq.repSends, FlockComplete.CutEq.innerAt, FlockComplete.CutEq.ligAt,
    FlockComplete.ZkTables.repRoundsT, e6, e7]
  omega

/-! ## The view rep against the cut rep -/

/-- **A `Finish` `FinishIs` reads opens coins every table admits**, at a statement the loop serves: no coin fault shows
(`frameUnrefused_ofLoop`), and the deployed tables admit unrefused coins (`tablesAdmitUnrefused_dataE`). -/
theorem admits_of_finCtx (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit} (hServe : Serves x)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (LeanRelease.ofLoopLink L₀ src real).toSrv.Chk} {m₀ ρ₀ : Msg}
    {ops : List (Msg × Msg)} (hF : FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops) :
    finS.Admits x (finCoins x ops) := by
  exact LeanRelease.tablesAdmitUnrefused_dataE x _ fun hx =>
    LeanRelease.frameUnrefused_ofLoop Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real
      CoinOpen.shaH wk x w hServe hx ν t κ m₀ ρ₀ ops hF.1 hF.2.1 hF.2.2.1 hF.2.2.2.1

/-- **Where every table admits the coins, the switched cut is the switched views'** with every guarded message sent
(`cutV_swViewsAt`, `famGuard_admits`). -/
theorem swCutOf_viewCut {x : Msg} {c : Fin (refK x) → finS.Coin} (hA : finS.Admits x c)
    (wt : ∀ j, FlockSoundness.Model.Witness (finS.C x j).S.m) (v : ∀ j, ((finS.C x j).T (finS.cz x c j)).tab.Vals)
    (t : finS.Tape) (j : Fin (finS.J x)) :
    finS.swCutOf x c wt v t j = Masked.viewCut (finS.swViewsAt x c wt v t j) := by
  rw [← RefSession.cutV_swViewsAt]
  have hP : ∀ r p, famPasses (fun j => ((finS.C x j).T (finS.cz x c j)).tab)
      (fun j => ((finS.C x j).T (finS.cz x c j)).rep) j r p :=
    fun _ _ j' r' p' _ => Masked.famGuard_admits hA j' r' p'
  show FlockSoundness.ZK.Table.cut _ (famSent (fun j => ((finS.C x j).T (finS.cz x c j)).tab)
    (fun j => ((finS.C x j).T (finS.cz x c j)).rep) j) (viewPre _ (finS.swViewsAt x c wt v t j)) = _
  simp only [FlockSoundness.ZK.Table.cut, famSent, hP, ite_true, Masked.viewCut, viewPre]

/-- **Where every table admits the coins, a rep's source from the views is the switched cut's** but for what only the
final message reads (`swCutOf_viewCut`). -/
theorem eraseS_finSrc (vs : finS.SwVals) {x : Msg} (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape)
    {ops : List (Msg × Msg)} (hA : finS.Admits x (finCoins x ops)) {i r : ℕ} (hi : i < Masked.jOf x) (hr : r < 2) :
    Masked.eraseS ((finSrc vs x w ν t ops i).rep r) =
      (Masked.tabSrcCut Masked.PE x (Masked.czExe Masked.PE x (finCoins x ops))
        (finS.swCutOf x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t) i).rep r := by
  have e : (fun j => Masked.viewCut
      (finS.swViewsAt x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t j)) =
      finS.swCutOf x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t :=
    funext fun j => (swCutOf_viewCut hA _ _ t j).symm
  unfold finSrc
  generalize finS.swViewsAt x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t = V at e ⊢
  generalize finS.swCutOf x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t = cu at e ⊢
  subst e
  unfold Masked.tabSrcView Masked.tabSrcCut
  simp only [hi, hr, ↓reduceDIte]
  refine Masked.eraseS_upd _ _ _ _ _ _ ?_ ?_ ?_ ?_ ?_ <;> rfl

/-- **A rep sends the same at sources equal but for what only the final message reads**, at level 0's commitments
with one cap and tree and pads commitments with one size, root and tree (`repCut_eraseIS`, `repCut_eraseC`). -/
theorem repCut_eraseZ (io : FlockComplete.Asks FlockComplete.PS) (hs : _root_.Flock.Hashes FlockComplete.PS)
    (st : _root_.Flock.Setup) (sch : _root_.Flock.Schedule) (rep : ℕ) (dom : ByteArray)
    {cm cm' : FlockComplete.Commit} (hcap : cm.cap = cm'.cap) (hlay : cm.layers = cm'.layers)
    (extras : Array _root_.Flock.Claim) {ic ic' : FlockComplete.InnerCommit} (hn : ic.ch.size = ic'.ch.size)
    (hroot : ic.root = ic'.root) (hilay : ic.layers = ic'.layers) {src src' : FlockComplete.RepSrc}
    (hsrc : Masked.eraseS src = Masked.eraseS src') (prior : Array (List _root_.Flock.F128)) (words : List ℕ) :
    FlockComplete.eraseZ <$> FlockComplete.repCut io hs st sch rep dom cm extras ic src prior words =
      FlockComplete.eraseZ <$> FlockComplete.repCut io hs st sch rep dom cm' extras ic' src' prior words := by
  rw [Masked.FinRep.repCut_eraseIS io hs st sch rep dom cm extras hn hroot hilay hsrc prior words,
    Masked.FinRep.repCut_eraseC io hs st sch rep dom hcap hlay extras ic' src' prior words]

/-- **Runs equal up to `eraseZ` send the same**: where one returns, the other returns the same transcript at the same
history, its proof equal up to `eraseZ`. -/
theorem go_of_eraseZ {m m' : _root_.Flock.TMW FlockComplete.PS _root_.Flock.Zk.ZkProof}
    (he : FlockComplete.eraseZ <$> m = FlockComplete.eraseZ <$> m') {t : _root_.Flock.Transcript} {l : Array Msg}
    {p : _root_.Flock.Zk.ZkProof} {tr : _root_.Flock.Transcript} {l' : Array Msg}
    (h : (m.run t).go l = (.ok (p, tr), l')) :
    ∃ p', (m'.run t).go l = (.ok (p', tr), l') ∧ FlockComplete.eraseZ p' = FlockComplete.eraseZ p := by
  have e := congrArg (fun z => (z.run t).go l) he
  simp only [StateT.run_map, FlockComplete.go_map', h] at e
  rcases h' : (m'.run t).go l with ⟨r, l''⟩
  rw [h'] at e
  cases r with
  | error err => simp [Except.map] at e
  | ok q =>
    obtain ⟨p', tr'⟩ := q
    simp only [Except.map, Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨⟨h1, h2⟩, h3⟩ := e
    subst h2 h3
    exact ⟨p', rfl, h1.symm⟩

/-! ## The cut run against the view run -/

/-- **The switched cut's source** at the opened coins: what the deployed session's round encodings read
(`Masked.encExe`, `msg_runCut`). -/
noncomputable def cutSrc (vs : finS.SwVals) (x : Msg) (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape)
    (ops : List (Msg × Msg)) : ℕ → FlockComplete.TabSrc :=
  Masked.tabSrcCut Masked.PE x (Masked.czExe Masked.PE x (finCoins x ops))
    (finS.swCutOf x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t)

theorem leaf0_cutSrc (vs : finS.SwVals) {x : Msg} (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape)
    {ops : List (Msg × Msg)} (hA : finS.Admits x (finCoins x ops)) (i : ℕ) :
    (cutSrc vs x w ν t ops i).leaf0 = (finSrc vs x w ν t ops i).leaf0 := by
  have e : (fun j => Masked.viewCut
      (finS.swViewsAt x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t j)) =
      finS.swCutOf x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t :=
    funext fun j => (swCutOf_viewCut hA _ _ t j).symm
  unfold cutSrc finSrc
  generalize finS.swViewsAt x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t = V at e ⊢
  generalize finS.swCutOf x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t = cu at e ⊢
  subst e
  unfold Masked.tabSrcView
  by_cases hi : i < Masked.jOf x
  · rw [dite_eq_left hi]
    try rfl
  · rw [dite_eq_right hi]
    unfold Masked.tabSrcCut
    rw [dite_eq_right hi]
    try rfl

theorem padLeaf_cutSrc (vs : finS.SwVals) {x : Msg} (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape)
    {ops : List (Msg × Msg)} (hA : finS.Admits x (finCoins x ops)) {i r : ℕ} (hi : i < Masked.jOf x) (hr : r < 2) :
    ((cutSrc vs x w ν t ops i).rep r).padLeaf = ((finSrc vs x w ν t ops i).rep r).padLeaf :=
  (congrArg FlockComplete.RepSrc.padLeaf (eraseS_finSrc vs w ν t hA hi hr)).symm

theorem eraseS_cutSrc (vs : finS.SwVals) {x : Msg} (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape)
    {ops : List (Msg × Msg)} (hA : finS.Admits x (finCoins x ops)) {i r : ℕ} (hi : i < Masked.jOf x) (hr : r < 2) :
    Masked.eraseS ((finSrc vs x w ν t ops i).rep r) = Masked.eraseS ((cutSrc vs x w ν t ops i).rep r) := by
  rw [show (cutSrc vs x w ν t ops i).rep r = Masked.eraseS ((finSrc vs x w ν t ops i).rep r) from
    (eraseS_finSrc vs w ν t hA hi hr).symm]
  generalize (finSrc vs x w ν t ops i).rep r = s
  cases s
  rfl

theorem array_mapM_rel {α β γ : Type} {g : α → _root_.Flock.V β} {f : α → _root_.Flock.V γ} (R : β → γ → Prop)
    (h : ∀ a b, g a = .ok b → ∃ c, f a = .ok c ∧ R b c) (xs : Array α) {bs : Array β} (hb : xs.mapM g = .ok bs) :
    ∃ cs : Array γ, xs.mapM f = .ok cs ∧ List.Forall₂ R bs.toList cs.toList := by
  rw [Array.mapM_eq_mapM_toList] at hb ⊢
  cases hL : xs.toList.mapM g with
  | error e =>
    rw [hL] at hb
    cases hb
  | ok bs' =>
    rw [hL] at hb
    obtain ⟨cs, hcs, hF⟩ := FlockComplete.CutPrefix.list_mapM_rel R h xs.toList bs' hL
    have e : bs'.toArray = bs := Except.ok.inj hb
    subst e
    refine ⟨cs.toArray, ?_, by simpa using hF⟩
    rw [hcs]
    rfl

/-- **Level 0's commitments from the switched cut** return where the views' do (`commitsView`), at the same caps and
trees, so at the same roots (`commit0Cut_leaf0`: the cut's leaves are the views'). -/
theorem cmsCut_ok (vs : finS.SwVals) {x : Msg} (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape)
    {ops : List (Msg × Msg)} (hA : finS.Admits x (finCoins x ops)) (schs : Array (_root_.Flock.Schedule × ℕ))
    {cms : Array FlockComplete.Commit} (hcms : ∀ l, (commitsView vs x w ν t ops schs).go l = (.ok cms, l)) :
    ∃ cmsC : Array FlockComplete.Commit,
      (∀ l, ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
        FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes FlockComplete.PS)
          (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
          (cutSrc vs x w ν t ops i)).go l = (.ok cmsC, l)) ∧
      cmsC.map (fun c => FlockComplete.flat c.cap) = cms.map (fun c => FlockComplete.flat c.cap) ∧
      cms.size = (FlockDeploy.tabsOf (Masked.stsOf x)).size ∧ cmsC.size = cms.size ∧
      ∀ i : ℕ, cmsC[i]!.cap = cms[i]!.cap ∧ cmsC[i]!.layers = cms[i]!.layers := by
  have hq : ∀ src : ℕ → FlockComplete.TabSrc, FlockComplete.quietS.R
      ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
        FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes _root_.Flock.V)
          (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default) (src i))
      ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
        FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes FlockComplete.PS)
          (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default) (src i)) := by
    intro src
    sim_walk [FlockComplete.CutEq.commit0Cut_quiet]
  have hv := hcms #[]
  unfold commitsView at hv
  rw [hq (finSrc vs x w ν t ops)] at hv
  have hV := (Prod.mk.inj hv).1
  have hsz := (FlockSoundness.Discharge.Exec.range_mapM_ok hV).1
  obtain ⟨cs, hcs, hF⟩ := array_mapM_rel (fun b c : FlockComplete.Commit => c.cap = b.cap ∧ c.layers = b.layers)
    (f := fun i => FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes _root_.Flock.V)
      (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default) (cutSrc vs x w ν t ops i))
    (fun a b hb => by
      rw [FlockComplete.commit0Cut_leaf0 _ _ _ (leaf0_cutSrc vs w ν t hA a), hb]
      exact ⟨_, rfl, rfl, rfl⟩) _ hV
  have hlen := FlockComplete.CutPrefix.forall₂_length hF
  simp only [Array.length_toList] at hlen
  refine ⟨cs, fun l => by rw [hq (cutSrc vs x w ν t ops) l, hcs], ?_, hsz, hlen.symm, fun i => ?_⟩
  · have e := FlockComplete.CutPrefix.forall₂_map (fun c : FlockComplete.Commit => FlockComplete.flat c.cap)
      (fun c : FlockComplete.Commit => FlockComplete.flat c.cap) (fun b c h => by rw [h.1]) hF
    apply Array.ext' 
    simpa using e.symm
  · by_cases hi : i < cms.size
    · have hi' : i < cs.size := by omega
      have hR := FlockComplete.CutPrefix.forall₂_getElem hF i (by simpa using hi) (by simpa using hi')
      simp only [Array.getElem_toList] at hR
      rw [getElem!_pos cs i hi', getElem!_pos cms i hi]
      exact hR
    · rw [getElem!_neg cs i (by omega), getElem!_neg cms i hi]
      exact ⟨rfl, rfl⟩

/-- **Rep `r`'s pads commitment from the switched cut** returns where the views' does, at the same size, root and tree
(`padsTreeCut_leaf`: the cut's pads leaves are the views'). -/
theorem icCut_ok (vs : finS.SwVals) {x : Msg} (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape)
    {ops : List (Msg × Msg)} (hA : finS.Admits x (finCoins x ops)) {i r : ℕ} (hi : i < Masked.jOf x) (hr : r < 2)
    (N : ℕ) {ic : FlockComplete.InnerCommit}
    (hic : ∀ l, (FlockComplete.padsTreeCut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes FlockComplete.PS) N
      ((finSrc vs x w ν t ops i).rep r)).go l = (.ok ic, l)) :
    ∃ icC : FlockComplete.InnerCommit,
      (∀ l, (FlockComplete.padsTreeCut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes FlockComplete.PS) N
        ((cutSrc vs x w ν t ops i).rep r)).go l = (.ok icC, l)) ∧
      ic.ch.size = icC.ch.size ∧ ic.root = icC.root ∧ ic.layers = icC.layers := by
  have hp := padLeaf_cutSrc vs w ν t hA hi hr
  have e1 := FlockComplete.padsTreeCut_leaf (M := FlockComplete.PS)
    (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) N hp
  have e2 := FlockComplete.padsTreeCut_leaf (M := FlockComplete.PS)
    (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) N hp.symm
  have h2 := hic #[]
  rw [e2, FlockComplete.go_map', e1, FlockComplete.go_map', hic #[]] at h2
  have h2' := congrArg Prod.fst h2
  simp only [Except.map, Except.ok.injEq] at h2'
  have hN : ic.ch.size = N := by
    rw [← h2']
    simp
  refine ⟨{ ic with
      ch := (Array.range N).map ((cutSrc vs x w ν t ops i).rep r).ch
      cm := (Array.range N).map ((cutSrc vs x w ν t ops i).rep r).cm
      salts := (Array.range N).map ((cutSrc vs x w ν t ops i).rep r).padSalt },
    fun l => by rw [e1, FlockComplete.go_map', hic l]; rfl, by simp [hN], rfl, rfl⟩

/-- **A pads commitment from the views sends nothing from any history** where it sends nothing from one
(`padsTreeCut_quiet`). -/
theorem padsView_all (vs : finS.SwVals) {x : Msg} (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape)
    {ops : List (Msg × Msg)} {schs : Array (_root_.Flock.Schedule × ℕ)} {k : ℕ} {ic : FlockComplete.InnerCommit}
    {l : Array Msg} (h : (padsView vs x w ν t ops schs k).go l = (.ok ic, l)) (l' : Array Msg) :
    (padsView vs x w ν t ops schs k).go l' = (.ok ic, l') := by
  unfold padsView at h ⊢
  rw [FlockComplete.CutEq.padsTreeCut_quiet] at h ⊢
  rw [(Prod.mk.inj h).1]

/-- **A loop that steps, then throws**: where each of the first `k` bodies yields from every state and history the
invariant `I` holds at, to a state and history it holds at the next step, the loop runs to the `k`-th body at a state and
history `I k` holds at, and throws where that body throws. -/
theorem forIn_go_err {α β : Type} (f : α → β → FlockComplete.PS (ForInStep β)) :
    ∀ (xs : List α) (I : ℕ → β → Array Msg → Prop) (k : ℕ) (hk : k < xs.length)
      (_ : ∀ k' (hk' : k' < k) b l, I k' b l →
        ∃ b' l', (f (xs[k']'(lt_trans hk' hk)) b).go l = (.ok (.yield b'), l') ∧ I (k' + 1) b' l')
      (b : β) (l : Array Msg), I 0 b l →
        ∃ b₀ l₀, I k b₀ l₀ ∧ ∀ e L, (f xs[k] b₀).go l₀ = (.error e, L) → (forIn xs b f).go l = (.error e, L)
  | [], _, _, hk, _, _, _, _ => absurd hk (by simp)
  | a :: as, I, 0, _, _, b, l, h0 => ⟨b, l, h0, fun e L he => by
      rw [List.forIn_cons, FlockComplete.go_bind]
      simp only [List.getElem_cons_zero] at he
      rw [he]⟩
  | a :: as, I, k + 1, hk, h, b, l, h0 => by
    obtain ⟨b', l', e1, h1⟩ := h 0 (by omega) b l h0
    obtain ⟨b₀, l₀, hI, hE⟩ := forIn_go_err f as (fun k => I (k + 1)) k (by simpa using hk)
      (fun k' hk' b l hI => by simpa using h (k' + 1) (by omega) b l hI) b' l' h1
    refine ⟨b₀, l₀, hI, fun e L he => ?_⟩
    rw [List.forIn_cons, FlockComplete.go_bind]
    simp only [List.getElem_cons_zero] at e1
    rw [e1]
    exact hE e L (by simpa using he)

/-- `forIn_go_err` under a bind: what holds of the throwing body's history holds of the run's. -/
theorem forIn_bind_err {α β γ : Type} {xs : List α} {b : β} {f : α → β → FlockComplete.PS (ForInStep β)}
    {g : β → FlockComplete.PS γ} {l : Array Msg} (P : Array Msg → Prop) (I : ℕ → β → Array Msg → Prop) (k : ℕ)
    (hk : k < xs.length)
    (h : ∀ k' (hk' : k' < k) b l, I k' b l →
      ∃ b' l', (f (xs[k']'(lt_trans hk' hk)) b).go l = (.ok (.yield b'), l') ∧ I (k' + 1) b' l')
    (h0 : I 0 b l) (hlast : ∀ b l, I k b l → ∃ e L, (f xs[k] b).go l = (.error e, L) ∧ P L) :
    P ((forIn xs b f >>= g).go l).2 := by
  obtain ⟨b₀, l₀, hI, hE⟩ := forIn_go_err f xs I k hk h b l h0
  obtain ⟨e, L, he, hP⟩ := hlast b₀ l₀ hI
  rw [FlockComplete.go_bind, hE e L he]
  exact hP

theorem getElem!_append_shift {a a' bs : Array Msg} (hs : a.size = a'.size) {n : ℕ} (hn : a.size ≤ n) :
    (a ++ bs)[n]! = (a' ++ bs)[n]! := by
  by_cases h : n < (a ++ bs).size
  · have h' : n < (a' ++ bs).size := by simp only [Array.size_append] at h ⊢; omega
    rw [getElem!_pos _ n h, getElem!_pos _ n h', Array.getElem_append_right (by omega),
      Array.getElem_append_right (by omega)]
    simp only [hs]
  · have h' : ¬ n < (a' ++ bs).size := by simp only [Array.size_append] at h ⊢; omega
    rw [getElem!_neg _ n h, getElem!_neg _ n h']

/-- **The cut session stopped at body `n` sent rep stream `k`'s body `n` as the view rep did**: where the reps before
`k` run from the views to the streams the auditor saw (`streamOf`), rep stream `k`'s run from the views from a history at
its first round sends, at each body `n` it sends, what the deployed session on the switched cut, stopped right after body
`n`, sends there (`msg_runCut`'s right side). The session runs its schedules, level 0's commitments from the cut (the
views' up to rows and salts, `cmsCut_ok`) and the link, then rep streams before `k` as from the views (`repCut_eraseZ`,
then `eq_of_agree`: they end by body `n`), then rep stream `k` from a history of the same size as `l`
(`repCut_shift`), which, stopped at `n`, throws right after the view rep's body `n` (`prefix_of_agree`). -/
theorem runCut_body (vs : finS.SwVals) {x : Msg} {w : Flock.Wit}
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    (ν : finS.Nonce) (t : finS.Tape) (m₀ ρ₀ : Msg) {ops : List (Msg × Msg)} (hA : finS.Admits x (finCoins x ops))
    {s0} {spec : _root_.Flock.SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : _root_.Flock.V _) = .ok schs)
    {cms : Array FlockComplete.Commit} (hcms : ∀ l, (commitsView vs x w ν t ops schs).go l = (.ok cms, l))
    (hext : ∀ k < 2 * spec.tables.size, ∃ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras)
    (hpads : ∀ k < 2 * spec.tables.size, ∀ l, ∃ ic, (padsView vs x w ν t ops schs k).go l = (.ok ic, l))
    {k : ℕ} (hk : k < 2 * spec.tables.size)
    (hprev : ∀ k' < k, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k' / 2]!.extra (viewPoints x ops spec) = .ok extras →
      ∀ ic l, (padsView vs x w ν t ops schs k').go l = (.ok ic, l) → l.size = Masked.repBase x k' →
        ∃ p tr l', (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k' extras ic).go l = (.ok (p, tr), l') ∧
          tr.rounds = streamOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) k' ∧ l'.size = Masked.repBase x (k' + 1))
    {extras : Array _root_.Flock.Claim}
    (hex : (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras)
    {ic : FlockComplete.InnerCommit} {l : Array Msg} (hic : (padsView vs x w ν t ops schs k).go l = (.ok ic, l))
    (hl : l.size = Masked.repBase x k) {p : _root_.Flock.Zk.ZkProof} {tr : _root_.Flock.Transcript}
    {l' : Array Msg} (h : (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k extras ic).go l = (.ok (p, tr), l'))
    {n : ℕ} (hn0 : l.size ≤ n) (hn : n < l'.size) :
    (Masked.runCut x (finCoins x ops) n spec (cutSrc vs x w ν t ops)).2[n]! = l'[n]! := by
  have hx : Masked.Live x := Masked.live_of hsc hR
  obtain ⟨-, -, hts, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hJ : Masked.jOf x = (Masked.stsOf x).size := by
    unfold Masked.jOf
    exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  have hl' := viewRep_size vs hsc hR ν t m₀ ρ₀ ops hsx hs cms hk hex ic hl h
  have hK : Masked.repBase x (k + 1) ≤ refK x := by
    rw [Masked.repBase_eq hsx, Masked.refK_eq hsx, ← FlockComplete.ZkTables.repStart_last]
    exact FlockComplete.ZkTables.repStart_mono _ _ (by omega)
  have hnN : n ≤ refK x := by omega
  have h1 : 1 ≤ n := by
    have := hl
    rw [Masked.repBase_eq hsx] at this
    unfold FlockComplete.repStartZk at this
    omega
  obtain ⟨cmsC, hcmsC, -, hcsz, hcsz', hcm⟩ := cmsCut_ok vs w ν t hA schs hcms
  -- the cut rep stream `k'` against the view one, at the answerer stopped at `N`
  have hcmp : ∀ k' < 2 * spec.tables.size, ∀ N extras' ic' icC,
      (∀ l, (padsView vs x w ν t ops schs k').go l = (.ok ic', l)) →
      (∀ l, (FlockComplete.padsTreeCut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes FlockComplete.PS)
        (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[k' / 2]!.m
          (FlockDeploy.tabsOf (Masked.stsOf x))[k' / 2]!.kLog schs[k' / 2]!.1.k0).len +
          _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
        ((cutSrc vs x w ν t ops (k' / 2)).rep (k' % 2))).go l = (.ok icC, l)) →
      ic'.ch.size = icC.ch.size ∧ ic'.root = icC.root ∧ ic'.layers = icC.layers →
      ∀ prior,
      FlockComplete.eraseZ <$> FlockComplete.repCut
          (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) N)
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) (FlockDeploy.tabsOf (Masked.stsOf x))[k' / 2]!
          schs[k' / 2]!.1 (k' % 2) (spec.domain spec.tables[k' / 2]! (k' % 2)).toUTF8 cms[k' / 2]! extras' ic'
          ((finSrc vs x w ν t ops (k' / 2)).rep (k' % 2)) prior (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[k' / 2]! =
        FlockComplete.eraseZ <$> FlockComplete.repCut
          (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) N)
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) (FlockDeploy.tabsOf (Masked.stsOf x))[k' / 2]!
          schs[k' / 2]!.1 (k' % 2) (spec.domain spec.tables[k' / 2]! (k' % 2)).toUTF8 cmsC[k' / 2]! extras' icC
          ((cutSrc vs x w ν t ops (k' / 2)).rep (k' % 2)) prior
          (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[k' / 2]! := by
    intro k' hk' N extras' ic' icC _ _ ⟨hn', hroot, hilay⟩ prior
    exact repCut_eraseZ _ _ _ _ _ _ (hcm (k' / 2)).1.symm (hcm (k' / 2)).2.symm extras' hn' hroot hilay
      (eraseS_cutSrc vs w ν t hA (by omega) (Nat.mod_lt _ (by norm_num))) prior _
  unfold Masked.runCut
  rw [FlockComplete.finSessionCut_eq, FlockComplete.go_bind, FlockComplete.schedsPZk_quiet, hs]
  dsimp only
  rw [FlockComplete.go_bind, hcmsC]
  dsimp only
  rw [FlockComplete.go_bind, FlockComplete.finAsksCut_link_go,
    ite_eq_right_of_eq_false _ _ (eq_false (by simp only [Array.size_empty]; omega))]
  try dsimp only
  unfold FlockComplete.afterLinkCut
  try simp only [Array.size_empty]
  try dsimp only
  rw [← Array.forIn_toList, FlockDeploy.repStreams_toList]
  refine forIn_bind_err (fun L => L[n]! = l'[n]!)
    (fun k b l => b.1 = (Array.range k).map (streamOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x)) ∧
      l.size = Masked.repBase x k) k (by simpa using hk) ?_ ⟨by simp, ?_⟩ ?_
  · rintro k' hk' ⟨streams, proofs⟩ ls ⟨hb, hls⟩
    dsimp only at hb
    subst hb
    have hk'' : k' < 2 * spec.tables.size := by omega
    obtain ⟨extras', hxe'⟩ := hext k' hk''
    obtain ⟨ic', hic'⟩ := hpads k' hk'' ls
    have hic'' := padsView_all vs w ν t hic'
    obtain ⟨icC, hicC, hI⟩ := icCut_ok vs w ν t hA (i := k' / 2) (r := k' % 2) (by omega) (Nat.mod_lt _ (by norm_num))
      _ (fun l => by have := hic'' l; unfold padsView at this; exact this)
    obtain ⟨p', tr', l'', hrep, hrounds, hsz⟩ := hprev k' hk' extras' hxe' ic' ls hic' hls
    unfold viewRep at hrep
    obtain ⟨p'', hcut, -⟩ := go_of_eraseZ (hcmp k' hk'' (refK x) extras' ic' icC hic'' hicC hI _) hrep
    have hmono : Masked.repBase x (k' + 1) ≤ Masked.repBase x k := by
      rw [Masked.repBase_eq hsx, Masked.repBase_eq hsx]
      exact FlockComplete.ZkTables.repStart_mono _ _ (by omega)
    have hmono' : Masked.repBase x k' ≤ Masked.repBase x (k' + 1) := by
      rw [Masked.repBase_eq hsx, Masked.repBase_eq hsx]
      exact FlockComplete.ZkTables.repStart_mono _ _ (by omega)
    have hcutn := FlockComplete.CutPrefix.eq_of_agree
      (FlockComplete.CutPrefix.repCut_cut_agree _ hnN _ _ _ _ _ _ _ _ _ _ _) (by omega) hcut (by omega)
    unfold viewPoints at hxe'
    refine ⟨?_, l'', ?_, ?_, hsz⟩
    rotate_left
    · simp only [List.getElem_map, List.getElem_range]
      try dsimp only
      rw [FlockComplete.go_bind, FlockComplete.go_liftV, hxe']
      try dsimp only
      rw [FlockComplete.go_bind, hicC]
      try dsimp only
      rw [FlockComplete.go_bind]
      try simp only [StateT.run] at hcutn ⊢
      rw [hcutn]
      try dsimp only
      rfl
    · try dsimp only
      rw [map_range_succ, hrounds]
  · rw [Masked.repBase_eq hsx]
    simp [FlockComplete.repStartZk]
  · rintro ⟨streams, proofs⟩ l₀ ⟨hb, hl₀⟩
    dsimp only at hb
    subst hb
    have hk2 : k / 2 < Masked.jOf x := by omega
    have hic'' := padsView_all vs w ν t hic
    obtain ⟨icC, hicC, hI⟩ := icCut_ok vs w ν t hA (i := k / 2) (r := k % 2) hk2 (Nat.mod_lt _ (by norm_num))
      _ (fun l => by have := hic'' l; unfold padsView at this; exact this)
    unfold viewRep at h
    obtain ⟨bs, rfl, hsh⟩ := FlockComplete.CutPrefix.go_of_shift
      (FlockComplete.CutPrefix.repCut_shift _ _ _ _ _ _ _ _ _ _ _ _ _) (hl.trans hl₀.symm) h
    obtain ⟨p'', hcut, -⟩ := go_of_eraseZ (hcmp k hk (refK x) extras ic icC hic'' hicC hI _) hsh
    have hsz : (l₀ ++ bs).size = (l ++ bs).size := by simp only [Array.size_append, hl, hl₀]
    obtain ⟨e, L, hcn, -, hpre⟩ := FlockComplete.CutPrefix.prefix_of_agree'
      (FlockComplete.CutPrefix.repCut_cut_agree _ hnN _ _ _ _ _ _ _ _ _ _ _) (by omega) hcut (by omega)
    refine ⟨e, L, ?_, ?_⟩
    · unfold viewPoints at hex
      simp only [List.getElem_map, List.getElem_range]
      try dsimp only
      rw [FlockComplete.go_bind, FlockComplete.go_liftV, hex]
      try dsimp only
      rw [FlockComplete.go_bind, hicC]
      try dsimp only
      rw [FlockComplete.go_bind]
      try simp only [StateT.run] at hcn ⊢
      rw [hcn]
    · rw [hpre n (by omega)]
      exact getElem!_append_shift (hl₀.trans hl.symm) (by omega)

/-! ## The view rep's rounds as the history's -/

theorem streamOf_size (x : Msg) (h : List (Msg × Msg)) (spec : _root_.Flock.SessionSpec)
    (sts : Array _root_.Flock.Stmt) (k : ℕ) :
    (streamOf x h spec sts k).size = (roundsOf spec sts).getD k 0 := by
  simp [streamOf, seenRounds]

/-- Round `j` of rep stream `k` as the auditor records it: rep round `streamAt + j`, the history's entry past `Hello`
and the link that carries it. -/
theorem streamOf_getElem (x : Msg) (h : List (Msg × Msg)) (spec : _root_.Flock.SessionSpec)
    (sts : Array _root_.Flock.Stmt) (k j : ℕ) (hj : j < (streamOf x h spec sts k).size) :
    (streamOf x h spec sts k)[j] =
      { n := loopNr x (streamAt (roundsOf spec sts) k + j + 1), sha := .empty,
        coins := wordsAt x (h.tail.map fun p => CoinLayer.rd p.2) (streamAt (roundsOf spec sts) k + j + 1),
        bytes := some (if j = 0 then
            _root_.Flock.Firewall.Frames.domainFrame (spec.domain spec.tables[k / 2]! (k % 2)) ++
              ⟨((h.tail.getD (streamAt (roundsOf spec sts) k + j + 1) ([], [])).1).tail.toArray⟩
          else ⟨((h.tail.getD (streamAt (roundsOf spec sts) k + j + 1) ([], [])).1).tail.toArray⟩) } := by
  simp only [streamOf, seenRounds, List.getElem_toArray, List.getElem_mapIdx, List.getElem_map, List.getElem_range]

/-- **(d)'s rounds at rep stream `k`**: where rep stream `k`'s run from the views returns from the history at its first
round, and the streams before it return with their rounds the auditor's, its round `j` is the auditor's. The run sent
body `L + j` for it (`repCut_rounds`, at `65536` coins a squeeze: `zkSchedule_small`, `live_bounds`), which is the cut
session's body there (`runCut_body`), so history message `L + j + 1` (`msg_runCut`), a release ending in the squeeze of
`loopNr x (L + j)` coins (`msg_release`): the round has those coins, the opened coin's words (`wordsAt_slot`), and the
message's bytes (`round_of_logs`). -/
theorem viewRep_rounds (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src real) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs) {ν : finS.Nonce} {t : finS.Tape} {κ : (LeanRelease.ofLoopLink L₀ src real).toSrv.Chk}
    {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)} (hF : FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops)
    {s0} {spec : _root_.Flock.SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : _root_.Flock.V _) = .ok schs)
    {cms : Array FlockComplete.Commit} (hcms : ∀ l, (commitsView vs x w ν t ops schs).go l = (.ok cms, l))
    (hext : ∀ k < 2 * spec.tables.size, ∃ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras)
    (hpads : ∀ k < 2 * spec.tables.size, ∀ l, ∃ ic, (padsView vs x w ν t ops schs k).go l = (.ok ic, l))
    {k : ℕ} (hk : k < 2 * spec.tables.size)
    (hprev : ∀ k' < k, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k' / 2]!.extra (viewPoints x ops spec) = .ok extras →
      ∀ ic l, (padsView vs x w ν t ops schs k').go l = (.ok ic, l) → l.size = Masked.repBase x k' →
        ∃ p tr l', (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k' extras ic).go l = (.ok (p, tr), l') ∧
          tr.rounds = streamOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) k' ∧ l'.size = Masked.repBase x (k' + 1))
    {extras : Array _root_.Flock.Claim}
    (hex : (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras)
    {ic : FlockComplete.InnerCommit} {l : Array Msg} (hic : (padsView vs x w ν t ops schs k).go l = (.ok ic, l))
    (hl : l.size = Masked.repBase x k) {p : _root_.Flock.Zk.ZkProof} {tr : _root_.Flock.Transcript}
    {l' : Array Msg} (h : (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k extras ic).go l = (.ok (p, tr), l')) :
    tr.rounds = streamOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) k := by
  have hx : Masked.Live x := Masked.live_of hsc hR
  have hA := admits_of_finCtx L₀ src real wk hServe hF
  have hlast : (LeanRelease.ofLoopLink L₀ src real).last x = refK x + 1 :=
    LeanRelease.loop_last loopNr Masked.dataE _ x
  obtain ⟨-, -, hts, hall⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  obtain ⟨hpts2, -, hreg⟩ := Masked.live_bounds hx hsx
  have hJ : Masked.jOf x = (Masked.stsOf x).size := by
    unfold Masked.jOf
    exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  have hi : k / 2 < Masked.jOf x := by omega
  have hi' : k / 2 < (Masked.stsOf x).size := by omega
  obtain ⟨schs', hs', hschs⟩ := Masked.schedsPZk_live hx
  rw [hs] at hs'
  cases hs'
  have hsch := hschs ⟨k / 2, hi⟩
  have hst : (Masked.stsOf x)[k / 2]! = Masked.stAt x ⟨k / 2, hi⟩ := getElem!_pos _ _ _
  have htab : (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! =
      _root_.Flock.Setup.ofCircuit (Masked.stsOf x)[k / 2]! := by
    rw [getElem!_pos (FlockDeploy.tabsOf (Masked.stsOf x)) (k / 2) (by simpa [FlockDeploy.tabsOf] using hi'),
      getElem!_pos (Masked.stsOf x) (k / 2) hi']
    simp [FlockDeploy.tabsOf]
  have hv : _root_.Flock.Zk.schedule (Masked.stsOf x)[k / 2]!.m = .ok schs[k / 2]! := by
    rw [hsch, hst]
    exact (Masked.zkTab x ⟨k / 2, hi⟩).hz
  have hes : extras.size = spec.points * (Masked.stsOf x)[k / 2]!.regions.size := by
    have hex' := hex
    rw [htab] at hex'
    rw [FlockComplete.JsonRT.extraClaims_size hex']
    simp [Nat.mul_comm, viewPoints]
  have hm : 25 ≤ (Masked.stsOf x)[k / 2]!.m ∧ (Masked.stsOf x)[k / 2]!.m ≤ 35 := by
    rw [getElem!_pos (Masked.stsOf x) (k / 2) hi']
    exact ⟨(hall (k / 2) hi').1, (hall (k / 2) hi').2.1⟩
  have hrg : (Masked.stsOf x)[k / 2]!.regions.size ≤ 1024 := by
    rw [getElem!_pos (Masked.stsOf x) (k / 2) hi']
    exact hreg (k / 2) hi'
  obtain ⟨hn6, hq6⟩ := FlockComplete.JsonRT.zkSchedule_small hm.1 hm.2 hv
  have hl' := viewRep_size vs hsc hR ν t m₀ ρ₀ ops hsx hs cms hk hex ic hl h
  have h' := h
  unfold viewRep at h'
  obtain ⟨hsz, hrd⟩ := FlockComplete.RepLog.repCut_rounds (B := 65536) _ _ _ _ _ _ _ _ _
    (by show (168 : ℕ) ≤ 65536; omega)
    (by rw [htab]; show (Masked.stsOf x)[k / 2]!.m - 13 ≤ 65536; omega)
    (by rw [hes, hpts2]; omega) hn6 hq6 h'
  have hK : Masked.repBase x (k + 1) ≤ refK x := by
    rw [Masked.repBase_eq hsx, Masked.refK_eq hsx, ← FlockComplete.ZkTables.repStart_last]
    exact FlockComplete.ZkTables.repStart_mono _ _ (by omega)
  have hrk : k < spec.repStreams.size := by
    have hrs : spec.repStreams.size = 2 * spec.tables.size := by
      have e := congrArg List.length (FlockDeploy.repStreams_toList spec)
      simpa using e
    omega
  have hL : l.size = 1 + streamAt (roundsOf spec (Masked.stsOf x)) k := by
    rw [hl, Masked.repBase_eq hsx, repStartZk_eq spec _ hrk.le, streamAt]
  have hcount : tr.rounds.size = (roundsOf spec (Masked.stsOf x)).getD k 0 := by
    have e1 : (roundsOf spec (Masked.stsOf x)).getD k 0 =
        FlockDeploy.repRounds true spec (Masked.stsOf x)[k / 2]! := by
      simp [roundsOf, hrk]
    have e2 := FlockComplete.ZkTables.repStart_succ spec (Masked.stsOf x) k
    have hl0 := hl
    have hl1 := hl'
    rw [Masked.repBase_eq hsx] at hl0 hl1
    rw [e1]
    omega
  have hlen := hF.1
  refine Array.ext (by rw [hcount, streamOf_size]) fun j h1 h2 => ?_
  have hr := hrd j (tr.rounds[j]'h1) (Array.getElem?_eq_getElem h1)
  have hlt : l.size + j < ops.length := by omega
  have hbody := runCut_body vs hsc hR ν t m₀ ρ₀ hA hsx hs hcms hext hpads hk hprev hex hic hl h
    (n := l.size + j) (by omega) (by omega)
  obtain ⟨s1, spec1, hsx1, hmsg⟩ := msg_runCut vs (LeanRelease.ofLoopLink L₀ src real) CoinOpen.shaH wk x w hE hc
    hlast ν t κ m₀ ρ₀ ops hF.1 hF.2.1 hF.2.2.1 hF.2.2.2.2.2 (l.size + j + 1) (by omega) (by omega)
  rw [hsx] at hsx1
  cases hsx1
  rw [Nat.add_sub_cancel] at hmsg
  have hmsg' : (((m₀, ρ₀) :: ops)[l.size + j + 1]'(by simp only [List.length_cons]; omega)).1 =
      dropDom (0 :: l'[l.size + j]!) := by
    rw [hmsg]
    exact congrArg (fun z => dropDom (0 :: z)) hbody
  obtain ⟨hlt64, o, k', hrel⟩ := LeanRelease.msg_release (LeanRelease.ofLoopLink L₀ src real) wk x w (ν, t) κ _
    hF.2.1 hF.2.2.2.2.2 (l.size + j + 1) (by simp only [List.length_cons]; omega)
    (LeanRelease.kindAt_mid hlast (by omega) (by omega))
  have hco : (LeanRelease.ofLoopLink L₀ src real).coins x (l.size + j + 1) = loopNr x (l.size + j) := rfl
  rw [hco] at hlt64 hrel
  have hd := (sizesOk_at (sizesOk_of_serves hServe s0 spec _ hsx) hrk).2.1
  have hround := round_of_logs hr (by norm_num) hd hmsg' hlt64 hrel
  have hcap : loopNr x (l.size + j) ≤ 65536 := by
    have := hr.le
    rw [hround] at this
    exact this
  have hws := wordsAt_size x ops (s := ops.length) (p := l.size + j) hlt le_rfl
  rw [List.take_length] at hws
  have hcoins : (Array.range (loopNr x (l.size + j))).map
      (fun i => Masked.slotAt (finCoins x ops) (l.size + j) i) =
      wordsAt x (ops.map fun p => CoinLayer.rd p.2) (l.size + j) := by
    refine Array.ext ?_ fun e he1 he2 => ?_
    · rw [Array.size_map, Array.size_range, hws,
        Nat.min_eq_left (show loopNr x (l.size + j) ≤ _root_.Flock.Firewall.Frames.COIN_MAX from hcap)]
    · have es := wordsAt_slot Masked.dataE x ops hF.1 (s := ops.length) (p := l.size + j) (e := e) hlt le_rfl
        (by rw [List.take_length]; exact he2)
      rw [List.take_length] at es
      rw [Array.getElem_map, Array.getElem_range, ← getElem!_pos _ e he2, es]
      all_goals rfl
  have hm2 : (((m₀, ρ₀) :: ops).tail.getD (l.size + j) ([], [])).1 =
      (((m₀, ρ₀) :: ops)[l.size + j + 1]'(by simp only [List.length_cons]; omega)).1 := by
    show (ops.getD (l.size + j) ([], [])).1 = (ops[l.size + j]'hlt).1
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hlt, Option.getD_some]
  rw [hround, streamOf_getElem x _ spec _ k j h2,
    show streamAt (roundsOf spec (Masked.stsOf x)) k + j + 1 = l.size + j by omega, hm2]
  congr 1

/-- **(d) from (a) to (c)**: at a statement in the masked verifier's scope that the loop serves, where the frame rounds
are the session's (`LiveRoundEq`, `SwCausal`), each rep stream's run from the views has the auditor's rounds
(`viewRep_rounds`) and ends at the next one's first round (`viewRep_size`), by induction on the stream: the streams
before it return (c), with their rounds the auditor's. -/
theorem finRepRounds_of (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src real) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs) (hA : FinCommits vs L₀ src real wk x w) (hB : FinPoints L₀ src real wk x w)
    (hC : FinRepProofs vs L₀ src real wk x w) : FinRepRounds vs L₀ src real wk x w := by
  intro ν t κ m₀ ρ₀ ops hF s0 spec hsx schs hs cms hcms
  obtain ⟨schs', hs', cms', -, -, hpads⟩ := hA ν t κ m₀ ρ₀ ops hF s0 spec hsx
  rw [hs] at hs'
  cases hs'
  have hext := (hB ν t κ m₀ ρ₀ ops hF s0 spec hsx).2
  have hfs : (filesE src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops)).isOk = true := by
    have h5 := hF.2.2.2.2.1
    change (!(filesE src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops)).isOk) = false at h5
    simpa using h5
  unfold filesE at hfs
  rw [hsx] at hfs
  dsimp only at hfs
  cases hp : proofsOf src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) with
  | error e => rw [hp] at hfs; cases hfs
  | ok files =>
  intro k
  induction k using Nat.strong_induction_on with
  | _ k ih =>
  intro hk extras hex ic l hic hl p tr l' h
  have hprev : ∀ k' < k, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k' / 2]!.extra (viewPoints x ops spec) = .ok extras →
      ∀ ic l, (padsView vs x w ν t ops schs k').go l = (.ok ic, l) → l.size = Masked.repBase x k' →
        ∃ p tr l', (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k' extras ic).go l = (.ok (p, tr), l') ∧
          tr.rounds = streamOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) k' ∧
          l'.size = Masked.repBase x (k' + 1) := by
    intro k' hk' extras' hex' ic' l₁ hic' hl₁
    obtain ⟨p', tr', l₂, hrun, -⟩ := hC ν t κ m₀ ρ₀ ops hF s0 spec hsx schs hs cms hcms files hp k' (by omega)
      extras' hex' ic' l₁ hic' hl₁
    exact ⟨p', tr', l₂, hrun, ih k' hk' (by omega) extras' hex' ic' l₁ hic' hl₁ p' tr' l₂ hrun⟩
  exact ⟨viewRep_rounds vs L₀ src real wk hsc hR hServe hE hc hF hsx hs hcms hext hpads hk hprev hex hic hl h,
    viewRep_size vs hsc hR ν t m₀ ρ₀ ops hsx hs cms hk hex ic hl h⟩

/-- **(a) to (c) give `FinishIs`**: (d) follows from them (`finRepRounds_of`), then `finishIs_of_parts`. -/
theorem finishIs_of_abc (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src real) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs) (hA : FinCommits vs L₀ src real wk x w) (hB : FinPoints L₀ src real wk x w)
    (hC : FinRepProofs vs L₀ src real wk x w) :
    LeanRelease.FinishIs finS vs (LeanRelease.ofLoopLink L₀ src real) CoinOpen.shaH wk x w :=
  finishIs_of_parts vs L₀ src real wk x w
    ⟨hA, hB, hC, finRepRounds_of vs L₀ src real wk hsc hR hServe hE hc hA hB hC⟩

end Loop

end Claims.ZKServer
