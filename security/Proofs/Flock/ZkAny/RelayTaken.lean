module

public import Proofs.Flock.Claims.ZK.FrameProg

/-! The model's taking of the coins without the rank stop (`answeredR`), which ties the loop's frame to the model between a round's coins and the next round's opening label (`answered_of_answeredR`). -/

@[expose] public section

namespace Claims.ZKServer

open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.ZkServer.Frame (Cfg Masker)

namespace LeanRelease

def takeAtR (n : ℕ) (same : Bool) (cN cP : Cfg) (prog : Masker) (fresh : Except String Masker)
    (cs : Array F128) : Except String Masker :=
  if n ≤ 2 then fresh else if same then prog.answer cN cs else prog.answer cP cs >>= fun _ => fresh

def frameAnswerR {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) :
    Except String Masker :=
  match h.getLast? with
  | some _ =>
    takeAtR h.length (decide (L.rep x h.length = L.rep x (h.length - 1))) (L.cfg x ρ h) (L.cfgAt x ρ (h.length - 1))
      (L.prog x h) (L.fresh x h) (L.coinsOf x h)
  | none => pure (L.prog x h)

def answeredR {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) : Except String Masker :=
  if L.opens x ρ h && L.tookEarlier x ρ h then L.frameAnswerR x ρ h else .error coinRefused

theorem takeAtR_of_takeAt {n : ℕ} {b : Bool} {cN cP : Cfg} {p : Masker} {f : Except String Masker}
    {cs : Array F128} {m : Masker} (h : takeAt n b cN cP p f cs = .ok m) : takeAtR n b cN cP p f cs = .ok m := by
  unfold takeAt at h
  unfold takeAtR
  by_cases hn : n ≤ 2
  · rw [if_pos hn] at h ⊢
    exact h
  · rw [if_neg hn] at h ⊢
    cases b with
    | true =>
      simp only [↓reduceIte] at h ⊢
      exact (bind_rankStop_ok h).1
    | false =>
      simp only [Bool.false_eq_true, ↓reduceIte] at h ⊢
      exact h

theorem takeAt_of_takeAtR {n : ℕ} {b : Bool} {cN cP : Cfg} {p : Masker} {f : Except String Masker}
    {cs : Array F128} {m : Masker} (h : takeAtR n b cN cP p f cs = .ok m)
    (hr : ¬ n ≤ 2 → b = true → rankFails cN m = false) : takeAt n b cN cP p f cs = .ok m := by
  unfold takeAtR at h
  unfold takeAt
  by_cases hn : n ≤ 2
  · rw [if_pos hn] at h ⊢
    exact h
  · rw [if_neg hn] at h ⊢
    cases b with
    | true =>
      simp only [↓reduceIte] at h ⊢
      rw [h]
      exact rankStop_of_not (hr hn rfl)
    | false =>
      simp only [Bool.false_eq_true, ↓reduceIte] at h ⊢
      exact h

theorem frameAnswer_cons {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (e : Msg × Msg)
    (t : List (Msg × Msg)) :
    L.frameAnswer x ρ (e :: t) =
      takeAt (e :: t).length (decide (L.rep x (e :: t).length = L.rep x ((e :: t).length - 1)))
        (L.cfg x ρ (e :: t)) (L.cfgAt x ρ ((e :: t).length - 1)) (L.prog x (e :: t)) (L.fresh x (e :: t))
        (L.coinsOf x (e :: t)) := by
  unfold frameAnswer
  cases hg : (e :: t).getLast? with
  | none => simp at hg
  | some _ => rfl

theorem frameAnswerR_cons {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (e : Msg × Msg)
    (t : List (Msg × Msg)) :
    L.frameAnswerR x ρ (e :: t) =
      takeAtR (e :: t).length (decide (L.rep x (e :: t).length = L.rep x ((e :: t).length - 1)))
        (L.cfg x ρ (e :: t)) (L.cfgAt x ρ ((e :: t).length - 1)) (L.prog x (e :: t)) (L.fresh x (e :: t))
        (L.coinsOf x (e :: t)) := by
  unfold frameAnswerR
  cases hg : (e :: t).getLast? with
  | none => simp at hg
  | some _ => rfl

theorem frameAnswerR_nil {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) :
    L.frameAnswerR x ρ [] = pure (L.prog x []) :=
  rfl

theorem frameAnswerR_of_frameAnswer {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)}
    {m : Masker} (hf : L.frameAnswer x ρ h = .ok m) : L.frameAnswerR x ρ h = .ok m := by
  rcases h with _ | ⟨e, t⟩
  · rw [frameAnswer_nil] at hf
    rw [frameAnswerR_nil]
    exact hf
  · rw [frameAnswer_cons] at hf
    rw [frameAnswerR_cons]
    exact takeAtR_of_takeAt hf

theorem frameAnswer_of_frameAnswerR {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)}
    {m : Masker} (hf : L.frameAnswerR x ρ h = .ok m)
    (hr : 3 ≤ h.length → L.rep x h.length = L.rep x (h.length - 1) → rankFails (L.cfg x ρ h) m = false) :
    L.frameAnswer x ρ h = .ok m := by
  rcases h with _ | ⟨e, t⟩
  · rw [frameAnswerR_nil] at hf
    rw [frameAnswer_nil]
    exact hf
  · rw [frameAnswerR_cons] at hf
    rw [frameAnswer_cons]
    exact takeAt_of_takeAtR hf (fun hn hb => hr (by omega) (of_decide_eq_true hb))

theorem answeredR_ok {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)} {m : Masker}
    (hA : L.answeredR x ρ h = .ok m) :
    L.opens x ρ h = true ∧ L.tookEarlier x ρ h = true ∧ L.frameAnswerR x ρ h = .ok m := by
  unfold answeredR at hA
  split at hA
  · rename_i hc
    exact ⟨((Bool.and_eq_true _ _).mp hc).1, ((Bool.and_eq_true _ _).mp hc).2, hA⟩
  · cases hA

theorem answeredR_of_opens {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg))
    (ho : L.opens x ρ h = true) (ht : L.tookEarlier x ρ h = true) : L.answeredR x ρ h = L.frameAnswerR x ρ h := by
  unfold answeredR
  rw [ho, ht]
  rfl

theorem answeredR_of_answered {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)}
    {m : Masker} (hA : L.answered x ρ h = .ok m) : L.answeredR x ρ h = .ok m := by
  obtain ⟨ho, ht, hf⟩ := answered_ok hA
  rw [L.answeredR_of_opens x ρ h ho ht]
  exact frameAnswerR_of_frameAnswer hf

theorem answered_of_answeredR {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)}
    {m : Masker} (hA : L.answeredR x ρ h = .ok m)
    (hr : 3 ≤ h.length → L.rep x h.length = L.rep x (h.length - 1) → rankFails (L.cfg x ρ h) m = false) :
    L.answered x ρ h = .ok m := by
  obtain ⟨ho, ht, hf⟩ := answeredR_ok hA
  rw [L.answered_of_opens x ρ h ho ht]
  exact frameAnswer_of_frameAnswerR hf hr

theorem cfg_eq_cfgAt_pred {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) {h : List (Msg × Msg)}
    (hs : L.rep x h.length = L.rep x (h.length - 1)) : L.cfg x ρ h = L.cfgAt x ρ (h.length - 1) := by
  unfold cfg cfgAt
  rw [hs]

theorem frameAnswerR_same {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) {h : List (Msg × Msg)}
    (hn : 3 ≤ h.length) (hs : L.rep x h.length = L.rep x (h.length - 1)) :
    L.frameAnswerR x ρ h = (L.prog x h).answer (L.cfg x ρ h) (L.coinsOf x h) := by
  rcases h with _ | ⟨e, t⟩
  · simp at hn
  · rw [frameAnswerR_cons]
    unfold takeAtR
    rw [if_neg (by omega), if_pos (decide_eq_true hs)]

open Loop in
theorem ofLoop_frameAnswerR_same (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q j : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1)
    (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1) :
    (ofLoop T L₀ real).frameAnswerR x ρ (e0 :: h1) =
      (progOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2)).answer ((ofLoop T L₀ real).cfg x ρ (e0 :: h1))
        (coinsAt x (e0 :: h1)) := by
  obtain ⟨q', rfl⟩ : ∃ q', q = q' + 1 := by
    rcases q with _ | q'
    · rw [locate_zero] at hj
      omega
    · exact ⟨q', rfl⟩
  have hp := locate_pred (roundsOf spec sts) 0 q' j hj
  have e1 : (e0 :: h1).length = q' + 1 + 2 := by simp only [List.length_cons, hq]
  have hs : (ofLoop T L₀ real).rep x (e0 :: h1).length = (ofLoop T L₀ real).rep x ((e0 :: h1).length - 1) := by
    show repOf x (e0 :: h1).length = repOf x ((e0 :: h1).length - 1)
    rw [e1, show q' + 1 + 2 - 1 = q' + 2 by omega, repOf_add_two hx, repOf_add_two hx, hp]
  rw [frameAnswerR_same _ x ρ (by rw [e1]; omega) hs]
  rfl

end LeanRelease

end Claims.ZKServer
