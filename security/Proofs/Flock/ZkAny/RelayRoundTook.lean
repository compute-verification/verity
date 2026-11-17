module

public import Proofs.Flock.ZkAny.RelayTook
public import Proofs.Flock.ZkAny.RelayTaken
public import Proofs.Flock.ZkAny.RelayRank
public import Proofs.Flock.ZkAny.RelayCarry
public import Proofs.Flock.Claims.Complete.Replay.RingSwitch
public import Proofs.Flock.Claims.Complete.Zk.InnerGo

/-! The model takes the coins after a frame round's history, from what the loop took (`answered_of_took`, `rank_of_release`). -/

@[expose] public section

namespace Claims.ZKServer

open _root_.Flock.ZkServer.Frame (Cfg Masker Place)

namespace LeanRelease

theorem frameAnswerR_of_fed {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)}
    (hf : L.fed x ρ h = true) (hn : 3 ≤ h.length) (hs : L.rep x h.length = L.rep x (h.length - 1)) :
    ∃ m, L.frameAnswerR x ρ h = .ok m := by
  unfold fed at hf
  rw [decide_eq_false (show ¬ h.length ≤ 2 by omega), Bool.false_or] at hf
  rw [frameAnswerR_same L x ρ hn hs, cfg_eq_cfgAt_pred L x ρ hs]
  cases ha : (L.prog x h).answer (L.cfgAt x ρ (h.length - 1)) (L.coinsOf x h) with
  | error e =>
    rw [ha] at hf
    first
      | cases hf
      | simp [Except.isOk, Except.toBool] at hf
  | ok m => exact ⟨m, rfl⟩

theorem answeredR_same {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)}
    (ho : L.opens x ρ h = true) (ht : L.tookEarlier x ρ h = true) (hf : L.fed x ρ h = true) (hn : 3 ≤ h.length)
    (hs : L.rep x h.length = L.rep x (h.length - 1)) : ∃ m, L.answeredR x ρ h = .ok m := by
  rw [L.answeredR_of_opens x ρ h ho ht]
  exact frameAnswerR_of_fed hf hn hs

theorem answeredR_switch {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)}
    (ho : L.opens x ρ h = true) (ht : L.tookEarlier x ρ h = true) (hf : L.fed x ρ h = true) (h2 : 2 ≤ h.length)
    (hs : 3 ≤ h.length → L.rep x h.length ≠ L.rep x (h.length - 1)) {m : Masker} (hm : L.fresh x h = .ok m) :
    L.answeredR x ρ h = .ok m := by
  rw [L.answeredR_of_opens x ρ h ho ht]
  rcases h with _ | ⟨e, t⟩
  · simp at h2
  rw [frameAnswerR_cons]
  unfold takeAtR
  by_cases hn : (e :: t).length ≤ 2
  · rw [if_pos hn]
    exact hm
  · rw [if_neg hn, if_neg (fun hb => hs (by omega) (of_decide_eq_true hb))]
    unfold fed at hf
    rw [decide_eq_false hn, Bool.false_or] at hf
    cases ha : (L.prog x (e :: t)).answer (L.cfgAt x ρ ((e :: t).length - 1)) (L.coinsOf x (e :: t)) with
    | error e' =>
      rw [ha] at hf
      first
        | cases hf
        | simp [Except.isOk, Except.toBool] at hf
    | ok a => exact hm

theorem answered_of_took {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)}
    (ho : L.opens x ρ h = true) (ht : L.tookEarlier x ρ h = true) (hf : L.fed x ρ h = true) (h2 : 2 ≤ h.length)
    (hfr : (3 ≤ h.length → L.rep x h.length ≠ L.rep x (h.length - 1)) → ∃ m, L.fresh x h = .ok m)
    (hrk : ∀ m, L.answeredR x ρ h = .ok m → 3 ≤ h.length → L.rep x h.length = L.rep x (h.length - 1) →
      rankFails (L.cfg x ρ h) m = false) :
    ∃ m, L.answered x ρ h = .ok m := by
  by_cases hs : 3 ≤ h.length ∧ L.rep x h.length = L.rep x (h.length - 1)
  · obtain ⟨m, hm⟩ := answeredR_same ho ht hf hs.1 hs.2
    exact ⟨m, answered_of_answeredR hm (hrk m hm)⟩
  · have hs' : 3 ≤ h.length → L.rep x h.length ≠ L.rep x (h.length - 1) := fun h3 he => hs ⟨h3, he⟩
    obtain ⟨m, hm⟩ := hfr hs'
    exact ⟨m, answered_of_answeredR (answeredR_switch ho ht hf h2 hs' hm) fun h3 he => absurd he (hs' h3)⟩

end LeanRelease

namespace Flags

open _root_.Flock.ZkServer.Frame (roundFree check)
open FlockSoundness.Refine (Frame encs)

theorem roundFree_count {c : Cfg} {s : Masker} {b : ByteArray} {n n' : ℕ} {k : UInt8} {pre : List UInt8}
    {q : ByteArray × Masker × Array ℕ} (hb : b.data.toList = pre ++ (Frame.sq k n').enc) (hn : n' < 2 ^ 64)
    (h : roundFree c s b n = .ok q) : n = n' := by
  obtain ⟨-, -, hl, -⟩ := roundFree_sq (Loop.roundFree_ok_inner h) h
  have hd : (pre ++ (Frame.sq k n').enc).drop pre.length = (Frame.sq k n').enc ++ [] := by
    rw [List.drop_left, List.append_nil]
  obtain ⟨-, -, hlen⟩ := header_at _ _ _ _ hd hn
  rw [eq_mk_encs hb] at hl
  have hsz : (ByteArray.mk (pre ++ (Frame.sq k n').enc).toArray).size - 16 = pre.length := by
    show (pre ++ (Frame.sq k n').enc).toArray.size - 16 = pre.length
    first
      | (rw [List.size_toArray, List.length_append, Frame.enc_length]; simp [Frame.payLen])
      | simp [Frame.enc_length, Frame.payLen]
  rw [hsz] at hl
  exact hl.symm.trans hlen

theorem check_count {c : Cfg} {s : Masker} {v b : ByteArray} {n n' : ℕ} {k : UInt8} {pre : List UInt8}
    {q : ByteArray × Masker × Option String} (hb : b.data.toList = pre ++ (Frame.sq k n').enc) (hn : n' < 2 ^ 64)
    (h : check c s v b n = .ok q) : n = n' := by
  rcases q with ⟨out, s', _ | w⟩
  · obtain ⟨q', hr, -⟩ := passes_none (check_none_passes h)
    exact roundFree_count hb hn hr
  · have hr := check_some h
    unfold _root_.Flock.ZkServer.Frame.round at hr
    obtain ⟨q', hq', -⟩ := bindV_ok hr
    exact roundFree_count hb hn hq'

theorem pcsRound_of_step {c : Cfg} {st st' : _root_.Flock.ZkServer.Frame.State} {n n' : ℕ} {v b out : ByteArray}
    {vd : String} (hq : PcsRound (b, n'))
    (h : _root_.Flock.ZkServer.Frame.step c st (.round n v (.sched b)) = (.round out vd, st')) :
    PcsRound (b, n) := by
  have e : n = n' := by
    obtain ⟨fs, k, hd, -, -, -, hn⟩ := hq
    obtain ⟨why, hck⟩ := _root_.Flock.ZkServer.Frame.check_of_step h
    exact check_count hd hn hck
  rw [e]
  exact hq

end Flags

namespace Relay

open _root_.Flock.ZkServer.Session (State)

theorem LoopModel.TookAt.opens {W Rr : Type} {M : LoopModel W Rr} {st : State} {c : Nat} (h : M.TookAt st c)
    {C : _root_.Flock.ZkServer.Coins.Release} (hC : st.session = some C) :
    M.L.opens M.x M.ρ (M.hist c) = true :=
  LeanRelease.opens_of_verdictAt (h.1 C hC)

theorem rankFails_of_release {c : Cfg} {st st' : _root_.Flock.ZkServer.Frame.State} {m : Masker} {n : Nat}
    {v b out : ByteArray} {vd : String} (hu : unstop st.m = unstop m) (hq : Flags.PcsRound (b, n))
    (h : _root_.Flock.ZkServer.Frame.step c st (.round n v (.sched b)) = (.round out vd, st')) :
    LeanRelease.rankFails c m = false :=
  (Flags.rankFails_unstop c hu).symm.trans (Flags.rankFails_of_step (r := (b, n)) hq h)

theorem rank_of_release {W Rr : Type} {L : LeanRelease W Rr} {x : Msg} {ρ : Rr} {h : List (Msg × Msg)}
    {q : _root_.Flock.ZkServer.Frames.Rep} (hu : ∀ m, L.answeredR x ρ h = .ok m → unstop q.st.m = unstop m)
    (hcf : q.cfg = L.cfg x ρ h) {n : Nat} {v out : ByteArray} {pc : Place} {vd : String}
    {st' : _root_.Flock.ZkServer.Frame.State}
    (hs : _root_.Flock.ZkServer.Frame.step q.cfg q.st (.round n v pc) = (.round out vd, st'))
    (hal : ∀ m, L.answeredR x ρ h = .ok m → LeanRelease.rankFails (L.cfg x ρ h) m = true →
      ∃ b, pc = .sched b ∧ (b = ByteArray.empty ∨ ∃ n', Flags.PcsRound (b, n'))) :
    ∀ m, L.answeredR x ρ h = .ok m → LeanRelease.rankFails (L.cfg x ρ h) m = false := by
  intro m hm
  cases hr : LeanRelease.rankFails (L.cfg x ρ h) m with
  | false => rfl
  | true =>
    obtain ⟨b, rfl, hb | ⟨n', hb⟩⟩ := hal m hm hr
    · subst hb
      obtain ⟨why, hck⟩ := _root_.Flock.ZkServer.Frame.check_of_step hs
      exact absurd hck (Loop.check_empty _ _ _ _ _)
    · have hb' := Flags.pcsRound_of_step hb hs
      rw [hcf] at hs
      have h' := rankFails_of_release (hu m hm) hb' hs
      rw [hr] at h'
      cases h'

open Loop LeanRelease in
theorem ofLoop_fed (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → _root_.Flock.V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {c : Cfg} (hc : CfgAlike c ((ofLoop T L₀ real).cfgAt x ρ h1.length))
    {st st' : _root_.Flock.ZkServer.Frame.State}
    (hu : unstop st.m = unstop (progOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2)))
    (hco : _root_.Flock.ZkServer.Frame.step c st (.coins (.ok (coinsAt x (e0 :: h1)))) = (.ok, st')) :
    (ofLoop T L₀ real).fed x ρ (e0 :: h1) = true := by
  obtain ⟨a', ha', -⟩ := answer_unstop hc hu (_root_.Flock.ZkServer.Frame.answer_of_step hco)
  show (decide ((e0 :: h1).length ≤ 2) || ((progOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2)).answer
    ((ofLoop T L₀ real).cfgAt x ρ ((e0 :: h1).length - 1)) (coinsAt x (e0 :: h1))).isOk) = true
  rw [List.length_cons, Nat.add_sub_cancel, ha']
  simp [Except.isOk, Except.toBool]

open Loop LeanRelease in
theorem coins_run_fed (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → _root_.Flock.V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (v : Msg) (a : ByteArray) {p₁ : _root_.Flock.ZkServer.Frames.Rep}
    (hc : CfgAlike p₁.cfg ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)))
    (hu : unstop p₁.st.m = unstop (progOf x ((h1 ++ [(v, a.data.toList)]).map fun p => ZKMal.CoinLayer.rd p.2)))
    {st₂ : _root_.Flock.ZkServer.Frame.State}
    (hco : _root_.Flock.ZkServer.Frame.step p₁.cfg p₁.st
      (.coins (.ok (_root_.Flock.ZkServer.Frames.feedOf (loopNr x h1.length) a.data.toList))) = (.ok, st₂)) :
    (ofLoop T L₀ real).fed x ρ (e0 :: (h1 ++ [(v, a.data.toList)])) = true := by
  rw [← coinsAt_cons_snoc x e0 h1 v] at hco
  have e : (ofLoop T L₀ real).cfg x ρ (e0 :: h1) =
      (ofLoop T L₀ real).cfgAt x ρ (h1 ++ [(v, a.data.toList)]).length := by
    rw [List.length_append, List.length_singleton]
    rfl
  rw [e] at hc
  exact ofLoop_fed T L₀ real x ρ e0 (h1 ++ [(v, a.data.toList)]) hc hu hco

open Loop LeanRelease in
theorem roundRun_fed (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → _root_.Flock.V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {sid : Nat} {st : State} {pl : _root_.Flock.ZkServer.Session.Placed}
    {stream : String} {i r : Nat} {dm : ByteArray} {v : Msg} {e : Env} {R : _root_.Flock.ZkServer.Frames.Sessions}
    {L0 : _root_.Flock.ZkServer.Level0.Slots} {R₃ : _root_.Flock.ZkServer.Frames.Sessions}
    (hr : RoundRun sid st pl stream (loopNr x h1.length) i r dm ⟨v.toArray⟩ e R L0 R₃)
    (hreq : st.request pl (.round stream (loopNr x h1.length)) = none) (hfr : (frameAt R sid i r).isSome)
    {q p₁ : _root_.Flock.ZkServer.Frames.Rep} {pl' : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length) =
      .ok (q, pl', f, nd))
    (ho : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).out = .ok o)
    (hput : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).put =
      some p₁) {a : ByteArray} (ha : e.ans = some a) (hb : _root_.Flock.ZkServer.Session.byte? a 0 = some 2)
    (hs : (st.answer (.round stream (loopNr x h1.length)) a).1.stop = none) (hco : e.coins = .ok ())
    {pre : List _root_.Flock.ZkServer.Frame.Req} {ev : List (Nat × ByteArray × Place × Array _root_.Flock.F128)}
    (hR : RepRun pre ev q) (hcq : q.cfg = (ofLoop T L₀ real).cfg x ρ (e0 :: h1))
    (hu : unstop (_root_.Flock.ZkServer.Frame.step q.cfg q.st
        (.round (loopNr x h1.length) (if nd.isSome then ByteArray.empty else (⟨v.toArray⟩ : ByteArray)) pl')).2.m =
      unstop (progOf x ((h1 ++ [(v, a.data.toList)]).map fun p => ZKMal.CoinLayer.rd p.2))) :
    (ofLoop T L₀ real).fed x ρ (e0 :: (h1 ++ [(v, a.data.toList)])) = true := by
  obtain ⟨st₂, hc, -, -⟩ := roundRun_next hr hreq hfr hrp ho hput ha hb hs hco hR
  obtain ⟨hpc, hps⟩ := roundStep_put_of hrp hput
  rw [← hps] at hu
  exact coins_run_fed T L₀ real x ρ e0 h1 v a (by rw [hpc, hcq]; exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩) hu hc

end Relay

end Claims.ZKServer
