module

public import Proofs.Flock.ZkAny.RelayPlace

/-!
# A rep's frame holds the history's words for its rounds so far

The model reads a rep's schedule so far on the history's words for the rep's rounds (`Loop.schedSoFar` at
`Loop.wordsAt`, the last `j + 1` coins before the round), and the loop's frame on the coins it took. Here the frame's
coins stay those words from one round of the rep to the next:

* `coinsAt_read`, `feedOf_words`: the coins the relay feeds a round's frame off the auditor's answer `a` are the
  history's words at that round, once the history's last answer is `a`.
* `window_snoc`, `coins_window_snoc`: the rep's words one round on are its words so far and the new round's.
* **`roundRun_coins`**: so at a `RoundRun` whose round releases and whose coins the rep's frame took, where the frame
  `roundPlace` readied holds the history's words for the rep's rounds so far, the frame the session's next round finds
  holds them for the history with the auditor's answer.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (F128)
open _root_.Flock.Firewall.Session (State Placed byte?)
open _root_.Flock.Firewall.Frames (Sessions Rep)
open _root_.Flock.Firewall.Frame (Req Place)

/-- **The coins the loop's frame reads after a history are its last answer's words**, as many as its request's count. -/
theorem coinsAt_read (x : Msg) (e0 : Msg × Msg) (ops : List (Msg × Msg)) :
    Loop.coinsAt x (e0 :: ops) = Loop.readCoins x (ops.map fun p => ZKMal.CoinLayer.rd p.2) := by
  unfold Loop.coinsAt Loop.readCoins
  simp only [List.tail_cons, List.getLast?_map, List.length_map]
  rcases ops.getLast? with _ | ⟨a, b⟩
  · rfl
  · exact Loop.feedOf_rd _ b

/-- **The coins the relay feeds a round's frame are the history's words there**: off the auditor's answer `a` to the
round after `e0 :: h1`, at the loop's count, they are the words of the history with `(v, a)` at that round. -/
theorem feedOf_words (x : Msg) (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (v a : Msg) {q : Nat}
    (hq : h1.length = q + 1) :
    _root_.Flock.Firewall.Frames.feedOf (Loop.loopNr x h1.length) a =
      Loop.wordsAt x ((h1 ++ [(v, a)]).map fun p => ZKMal.CoinLayer.rd p.2) (q + 1) := by
  have hlen : ((h1 ++ [(v, a)]).map fun p => ZKMal.CoinLayer.rd p.2).length = q + 1 + 1 := by
    simp [hq]
  rw [← coinsAt_cons_snoc x e0 h1 v a, coinsAt_read x e0, Loop.readCoins_eq x _ hlen]

/-- **A window one step on**: the values at the `j + 1` places before `q + 1`, then the one at `q + 1`, are the values
at the `j + 2` places up to `q + 1`, where the two functions agree up to `q`. -/
theorem window_snoc {α : Type} (f g : Nat → α) {q j : Nat} (hj : j + 1 ≤ q) (hfg : ∀ i, i ≤ q → f i = g i) :
    ((Array.range (j + 1)).map fun i => f (q - (j + 1) + i + 1)).push (g (q + 1)) =
      (Array.range (j + 2)).map fun i => g (q + 1 - (j + 2) + i + 1) := by
  have hr : Array.range (j + 2) = (Array.range (j + 1)).push (j + 1) := by
    rw [Array.push_eq_append]
    exact Array.range_succ
  rw [hr, Array.map_push]
  have e1 : q + 1 - (j + 2) + (j + 1) + 1 = q + 1 := by omega
  rw [e1]
  congr 1
  apply Array.map_congr_left
  intro i hi
  rw [Array.mem_range] at hi
  rw [show q + 1 - (j + 2) = q - (j + 1) by omega]
  exact hfg _ (by omega)

/-- **A rep's words one round on**: its words for its `j + 1` rounds before coin `q + 1`, then that coin's, are its
words for its `j + 2` rounds in the coins with one more. -/
theorem coins_window_snoc (x : Msg) (cs : List FlockDeploy.Coin) (d : FlockDeploy.Coin) {q j : Nat}
    (hq : cs.length = q + 1) (hj : j + 1 ≤ q) :
    ((Array.range (j + 1)).map fun k => Loop.wordsAt x cs (q - (j + 1) + k + 1)).push
        (Loop.wordsAt x (cs ++ [d]) (q + 1)) =
      (Array.range (j + 2)).map fun k => Loop.wordsAt x (cs ++ [d]) (q + 1 - (j + 2) + k + 1) :=
  window_snoc (fun k => Loop.wordsAt x cs k) (fun k => Loop.wordsAt x (cs ++ [d]) k) hj
    fun k hk => (Loop.wordsAt_append x cs [d] (by omega)).symm

/-- **A rep's frame holds the history's words one round on**: at a `RoundRun` at the loop's count after `e0 :: h1`
whose round releases and whose coins the rep's frame took off the auditor's answer `a`, where the frame `roundPlace`
readied holds the words of the rep's `j + 1` rounds so far, the frame the session's next round finds holds the words of
its `j + 2` rounds in the history with `(v, a)`. -/
theorem roundRun_coins (x : Msg) (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (v : Msg) {q j : Nat}
    (hq : h1.length = q + 1) (hj : j + 1 ≤ q) {sid : Nat} {st : State} {pl : Placed} {stream : String} {i r : Nat}
    {dom content : ByteArray} {e : Env} {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream (Loop.loopNr x h1.length) i r dom content e R L0 R₃)
    (hreq : st.request pl (.round stream (Loop.loopNr x h1.length)) = none) (hfr : (frameAt R sid i r).isSome)
    {q' p₁ : Rep} {pl' : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content (Loop.loopNr x h1.length) =
      .ok (q', pl', f, nd))
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content (Loop.loopNr x h1.length)).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content (Loop.loopNr x h1.length)).put =
      some p₁) {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (Loop.loopNr x h1.length)) a).1.stop = none) (hc : e.coins = .ok ())
    {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)} (hR : RepRun pre ev q')
    (hcw : q'.coins = (Array.range (j + 1)).map fun k =>
      Loop.wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) (q - (j + 1) + k + 1)) :
    ∃ p₃, frameAt R₃ sid i r = some p₃ ∧ p₃.coins = (Array.range (j + 2)).map fun k =>
      Loop.wordsAt x ((h1 ++ [(v, a.data.toList)]).map fun p => ZKMal.CoinLayer.rd p.2) (q + 1 - (j + 2) + k + 1) := by
  obtain ⟨st₂, -, hat, -⟩ := roundRun_next hr hreq hfr hrp ho hput ha hb hs hc hR
  rcases hres : _root_.Flock.Firewall.Frame.step q'.cfg q'.st
    (.round (Loop.loopNr x h1.length) (if nd.isSome then ByteArray.empty else content) pl') with ⟨a', st'⟩
  obtain ⟨w, hw⟩ := roundStep_put hrp hres
  rw [hw, Option.some.injEq] at hput
  subst hput
  refine ⟨_, hat, ?_⟩
  have hsplit : ((h1 ++ [(v, a.data.toList)]).map fun p => ZKMal.CoinLayer.rd p.2) =
      (h1.map fun p => ZKMal.CoinLayer.rd p.2) ++ [ZKMal.CoinLayer.rd a.data.toList] := by
    simp
  show q'.coins.push (_root_.Flock.Firewall.Frames.feedOf (Loop.loopNr x h1.length) a.data.toList) = _
  rw [feedOf_words x e0 h1 v a.data.toList hq, hsplit, hcw]
  exact coins_window_snoc x _ _ (by rw [List.length_map, hq]) hj

end Claims.ZKServer.Relay
