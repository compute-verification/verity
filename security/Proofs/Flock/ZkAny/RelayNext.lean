module

public import Proofs.Flock.ZkAny.RelayRun

/-!
# A rep's frame at the session's next round

At a frame round as the loop ran it (`RoundRun`) that releases, where the auditor answered coins the rep's frame took,
the session's next round reads the rep's frame the coins put back:

* **`roundRun_next`**: there the rep's frame is `Loop.coinsOf` of the frame the round put, on the coins
  `Frames.feedOf` reads off the auditor's answer, and where the frame `roundPlace` readied was its run on the rep's
  requests so far (`RepRun`), it is its run on them and the round in its place with its coins (`repRun_round_coins`).
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (F128)
open _root_.Flock.Firewall.Session (State Placed byte?)
open _root_.Flock.Firewall.Frames (Sessions Rep)
open _root_.Flock.Firewall.Frame (Req Place)

/-- **The rep's frame at the session's next round, and its run**: at a `RoundRun` whose round releases and whose
coins the rep's frame took, the frame the session's next round finds is `Loop.coinsOf` of the round's put on the coins
`Frames.feedOf` reads off the auditor's answer, and it is the run of the frame `roundPlace` readied one round on. -/
theorem roundRun_next {sid : Nat} {st : State} {pl : Placed} {stream : String} {n i r : Nat}
    {dom content : ByteArray} {e : Env} {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream n i r dom content e R L0 R₃) (hreq : st.request pl (.round stream n) = none)
    (hfr : (frameAt R sid i r).isSome) {q p₁ : Rep} {pl' : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl', f, nd))
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁) {a : ByteArray}
    (ha : e.ans = some a) (hb : byte? a 0 = some 2) (hs : (st.answer (.round stream n) a).1.stop = none)
    (hc : e.coins = .ok ()) {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)}
    (hR : RepRun pre ev q) :
    ∃ st₂, _root_.Flock.Firewall.Frame.step p₁.cfg p₁.st
        (.coins (.ok (_root_.Flock.Firewall.Frames.feedOf n a.data.toList))) = (.ok, st₂) ∧
      frameAt R₃ sid i r = some (Loop.coinsOf p₁ st₂ (_root_.Flock.Firewall.Frames.feedOf n a.data.toList)) ∧
      RepRun pre (ev ++ [(n, if nd.isSome then ByteArray.empty else content, pl',
          _root_.Flock.Firewall.Frames.feedOf n a.data.toList)])
        (Loop.coinsOf p₁ st₂ (_root_.Flock.Firewall.Frames.feedOf n a.data.toList)) := by
  obtain ⟨R₁, R₂, R₂', p', h01, h12, hcs, h22, h23⟩ :=
    hr.coins hreq ((hr.round hreq).trans ho) hput ha hb hs hc
  have hfa : frameAt R₂ sid i r = some p₁ := frameAt_own hfr (roundStep_key hput) (h01.1 i r) (h12.1 i r)
  obtain ⟨j, hj⟩ := find_of_frameAt hfa
  obtain ⟨st₂, hst, rfl⟩ := coinsStep_inv hj hcs
  obtain ⟨-, hR'⟩ := repRun_round_coins hR hfr hrp ho hput (h01.1 i r) (h12.1 i r) hst
  have hs₂ : (frameAt R₂ sid i r).isSome := by
    rw [hfa]
    rfl
  exact ⟨st₂, hst, frameAt_own hs₂ (coinsStep_key hcs) (h22.1 i r) (h23.1 i r), hR'⟩

end Claims.ZKServer.Relay
