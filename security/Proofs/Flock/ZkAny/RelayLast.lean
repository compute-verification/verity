module

public import Proofs.Flock.ZkAny.RelaySame

/-! After a round past its first, a rep's frame keeps its run, words and schedule so far, whichever rep's round comes next (`roundRun_last`). -/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (V F128 SessionSpec Stmt)
open _root_.Flock.ZkServer.Session (State Placed byte?)
open _root_.Flock.ZkServer.Frames (Sessions Rep)
open _root_.Flock.ZkServer.Frame (Cfg Masker Req Place)

open Loop in
theorem roundRun_last {W Rr : Type} {L : LeanRelease W Rr} {ρ : Rr} (x : Msg) (e0 : Msg × Msg)
    (h1 : List (Msg × Msg)) (v : Msg) {rs : List Nat} {q₀ j₀ : Nat} (hq : h1.length = q₀ + 1)
    (hj₀ : (ZKMal.CoinLayer.locate rs 0 q₀).2 = j₀ + 1)
    {sid : Nat} {st : State} {pl : Placed} {stream : String} {i r : Nat} {dm : ByteArray} {e : Env}
    {R : Sessions} {L0 : _root_.Flock.ZkServer.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream (loopNr x h1.length) i r dm ⟨v.toArray⟩ e R L0 R₃)
    (hreq : st.request pl (.round stream (loopNr x h1.length)) = none)
    {q p₁ : Rep} {pl' : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length) =
      .ok (q, pl', f, nd))
    (ho : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).out = .ok o)
    (hput : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).put =
      some p₁) {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (loopNr x h1.length)) a).1.stop = none) (hco : e.coins = .ok ())
    {p : Rep} (hfa : frameAt R sid i r = some p) {s : _root_.Flock.ZkServer.Frames.Release} {name : String}
    {l0 : _root_.Flock.ZkServer.Level0.Level0} (hss : R.sessions.find? (·.sid == sid) = some s)
    (hn : s.spec.tables[i]? = some name) (hl0 : _root_.Flock.ZkServer.Level0.ofSession L0 sid name = some l0)
    {t : _root_.Flock.ZkServer.Finish.Tab} {xs : Array (Array F128)} {hi : Array ByteArray} {stm : Stmt}
    {c : Array F128} {r0 : ByteArray} {q0 : Nat} (hst : s.sts[p.table]? = some stm) (hlink : s.link = some c)
    (hsalt : p.alg.size ≤ p.coins.size → p.opened = false → p.opening = none →
      R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    (htab : _root_.Flock.ZkServer.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t)
    (hxs : xs = _root_.Flock.ZkServer.Drive.extras stm (_root_.Flock.ZkServer.Drive.linkPoints stm c))
    {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)}
    (hI : SameInv L x ρ (e0 :: h1) (h1.map fun p => ZKMal.CoinLayer.rd p.2) q₀ j₀ t
      (s.spec.domain name r).toUTF8 xs hi pre ev p)
    {sched : Array (ByteArray × Nat)}
    (hsc : schedSoFar t p.rep (s.spec.domain name r).toUTF8 xs hi p.coins = .ok sched) :
    ∃ p₃, frameAt R₃ sid i r = some p₃ ∧
      RepRun pre (ev ++ [(loopNr x h1.length, if nd.isSome then ByteArray.empty else (⟨v.toArray⟩ : ByteArray), pl',
        _root_.Flock.ZkServer.Frames.feedOf (loopNr x h1.length) a.data.toList)]) p₃ ∧
      p₃.coins = (Array.range (j₀ + 2)).map (fun k =>
        wordsAt x ((h1 ++ [(v, a.data.toList)]).map fun p => ZKMal.CoinLayer.rd p.2) (q₀ + 1 - (j₀ + 2) + k + 1)) ∧
      RepInv t (s.spec.domain name r).toUTF8 xs hi p₃ := by
  have hfr : (frameAt R sid i r).isSome := by
    rw [hfa]
    rfl
  obtain ⟨jf, hf⟩ := find_of_frameAt hfa
  obtain ⟨hl, hRq, -, -, hcw⟩ := sameInv_place hI hfa hrp
  obtain ⟨-, P1, hat, hinv⟩ :=
    roundRun_repInv hr hreq hfr hrp ho hput ha hb hs hco hRq hf hl hss hn hl0 hst hlink hsalt htab hxs hI.inv hsc
  have hjq : j₀ + 1 ≤ q₀ := by
    have := Loop.locate_snd_le rs 0 q₀
    rw [hj₀] at this
    exact this
  obtain ⟨P2, hat2, hcw2⟩ := roundRun_coins x e0 h1 v hq hjq hr hreq hfr hrp ho hput ha hb hs hco hRq hcw
  obtain rfl := Option.some.inj (hat2.symm.trans hat)
  obtain ⟨st₂, -, hat4, hrun⟩ := roundRun_next hr hreq hfr hrp ho hput ha hb hs hco hRq
  obtain rfl := Option.some.inj (hat4.symm.trans hat)
  exact ⟨_, hat, hrun, hcw2, hinv⟩

end Claims.ZKServer.Relay
