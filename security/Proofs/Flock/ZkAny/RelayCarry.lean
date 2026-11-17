module

public import Proofs.Flock.ZkAny.RelayNext

/-! After a session round's coins, a rep's masking state is the model's for its next round, but for `stopped` (`roundRun_carry`). -/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (V F128 SessionSpec Stmt)
open _root_.Flock.ZkServer.Session (State Placed byte?)
open _root_.Flock.ZkServer.Frames (Sessions Rep)
open _root_.Flock.ZkServer.Frame (Cfg Masker Req Place)

theorem coinsAt_cons_snoc (x : Msg) (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (v a : Msg) :
    Loop.coinsAt x (e0 :: (h1 ++ [(v, a)])) =
      _root_.Flock.ZkServer.Frames.feedOf (Loop.loopNr x h1.length) a := by
  simp [Loop.coinsAt]

open Loop LeanRelease in
theorem coins_run_answered (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (v : Msg) (a : ByteArray) {s0 : _} {spec : SessionSpec}
    {sts : Array Stmt} {j : Nat} (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 h1.length).2 = j + 1) {p₁ : Rep}
    (hc : CfgAlike p₁.cfg ((ofLoop T L₀ real).cfg x ρ (e0 :: (h1 ++ [(v, a.data.toList)]))))
    (hu : unstop p₁.st.m = unstop (progOf x ((h1 ++ [(v, a.data.toList)]).map fun p => ZKMal.CoinLayer.rd p.2)))
    {st₂ : _root_.Flock.ZkServer.Frame.State}
    (hco : _root_.Flock.ZkServer.Frame.step p₁.cfg p₁.st
      (.coins (.ok (_root_.Flock.ZkServer.Frames.feedOf (loopNr x h1.length) a.data.toList))) = (.ok, st₂))
    {m : Masker} (hA : (ofLoop T L₀ real).answeredR x ρ (e0 :: (h1 ++ [(v, a.data.toList)])) = .ok m) :
    unstop st₂.m = unstop m := by
  rw [← coinsAt_cons_snoc x e0 h1 v] at hco
  exact coins_answered T L₀ real x ρ e0 (h1 ++ [(v, a.data.toList)]) hx (by simp) hj hc hu hco hA

theorem roundStep_put_of {R : Sessions} {L0 : _root_.Flock.ZkServer.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q p₁ : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    (hput : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dom content n).put = some p₁) :
    p₁.cfg = q.cfg ∧
      p₁.st = (_root_.Flock.ZkServer.Frame.step q.cfg q.st
        (.round n (if nd.isSome then ByteArray.empty else content) pl)).2 := by
  rcases hres : _root_.Flock.ZkServer.Frame.step q.cfg q.st
    (.round n (if nd.isSome then ByteArray.empty else content) pl) with ⟨a, st'⟩
  obtain ⟨v, hv⟩ := roundStep_put hrp hres
  rw [hv, Option.some.injEq] at hput
  subst hput
  exact ⟨rfl, rfl⟩

open Loop LeanRelease in
theorem roundRun_carry (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {j : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 h1.length).2 = j + 1) {sid : Nat} {st : State}
    {pl : Placed} {stream : String} {i r : Nat} {dom : ByteArray} {v : Msg} {e : Env} {R : Sessions}
    {L0 : _root_.Flock.ZkServer.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream (loopNr x h1.length) i r dom ⟨v.toArray⟩ e R L0 R₃)
    (hreq : st.request pl (.round stream (loopNr x h1.length)) = none) (hfr : (frameAt R sid i r).isSome)
    {q p₁ : Rep} {pl' : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace R L0 sid i r dom ⟨v.toArray⟩ (loopNr x h1.length) =
      .ok (q, pl', f, nd))
    (ho : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dom ⟨v.toArray⟩ (loopNr x h1.length)).out = .ok o)
    (hput : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dom ⟨v.toArray⟩ (loopNr x h1.length)).put =
      some p₁) {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (loopNr x h1.length)) a).1.stop = none) (hc : e.coins = .ok ())
    {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)} (hR : RepRun pre ev q)
    (hcfg : CfgAlike q.cfg ((ofLoop T L₀ real).cfg x ρ (e0 :: (h1 ++ [(v, a.data.toList)]))))
    (hu : unstop (_root_.Flock.ZkServer.Frame.step q.cfg q.st
        (.round (loopNr x h1.length) (if nd.isSome then ByteArray.empty else (⟨v.toArray⟩ : ByteArray)) pl')).2.m =
      unstop (progOf x ((h1 ++ [(v, a.data.toList)]).map fun p => ZKMal.CoinLayer.rd p.2)))
    {m : Masker} (hA : (ofLoop T L₀ real).answeredR x ρ (e0 :: (h1 ++ [(v, a.data.toList)])) = .ok m) :
    ∃ p, frameAt R₃ sid i r = some p ∧
      RepRun pre (ev ++ [(loopNr x h1.length, if nd.isSome then ByteArray.empty else (⟨v.toArray⟩ : ByteArray), pl',
        _root_.Flock.ZkServer.Frames.feedOf (loopNr x h1.length) a.data.toList)]) p ∧
      unstop p.st.m = unstop m := by
  obtain ⟨st₂, hco, hat, hrun⟩ := roundRun_next hr hreq hfr hrp ho hput ha hb hs hc hR
  obtain ⟨hpc, hps⟩ := roundStep_put_of hrp hput
  rw [← hpc] at hcfg
  rw [← hps] at hu
  exact ⟨_, hat, hrun, coins_run_answered T L₀ real x ρ e0 h1 v a hx hj hcfg hu hco hA⟩

end Claims.ZKServer.Relay
