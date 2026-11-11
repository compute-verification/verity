module

public import Proofs.Flock.ZkAny.RelayLast

/-!
# A rep's frame between its stream's last round and the next stream's first

After a rep's round the session's next round finds the rep's frame as its run one round on (`roundRun_last`). Where the
round was its stream's last, the next stream's first round comes next. Until then only updates that keep the rep's
frame come (`Kept`), so that round finds it unchanged, holding the history's words for all the stream's rounds:

* `locate_last`, `count_last`: where the round after a stream's round is the next stream's, the round is its stream's
  last, and its stream's count is one more than the round's index in it.
* **`roundRun_handoff`**: at a `RoundRun` of a stream's last round past its first, after updates that keep the rep's
  frame, the frames hold the rep's frame with the history's words for all the stream's rounds, as many as its count.
  At rep 0's last round this is the frame and the count rep 1's first round reads (`roundPlace_switch1`'s `h0`, `hn0`
  and `hco`).
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (F128 Stmt)
open _root_.Flock.Firewall.Session (State Placed byte?)
open _root_.Flock.Firewall.Frames (Sessions Rep)
open _root_.Flock.Firewall.Frame (Req Place)

/-- **A stream's last round**: where the round after the `i`th is another stream's, the `i`th round's index in its
stream is one short of the stream's count. -/
theorem locate_last : ∀ (rs : List Nat) (k i : Nat),
    (ZKMal.CoinLayer.locate rs k (i + 1)).1 ≠ (ZKMal.CoinLayer.locate rs k i).1 →
    (ZKMal.CoinLayer.locate rs k i).2 + 1 = rs.getD ((ZKMal.CoinLayer.locate rs k i).1 - k) 0
  | [], _, _, h => absurd rfl h
  | c :: cs, k, i, h => by
    by_cases h1 : i + 1 < c
    · rw [Loop.locate_cons_lt h1, Loop.locate_cons_lt (Nat.lt_of_succ_lt h1)] at h
      exact absurd rfl h
    · by_cases h0 : i < c
      · rw [Loop.locate_cons_lt h0]
        simp only [Nat.sub_self, List.getD_cons_zero]
        omega
      · rw [Loop.locate_cons_ge h1, Loop.locate_cons_ge h0] at h
        have e : i + 1 - c = i - c + 1 := by omega
        rw [e] at h
        have hge := Loop.locate_fst_ge cs (k + 1) (i - c)
        rw [Loop.locate_cons_ge h0, show (ZKMal.CoinLayer.locate cs (k + 1) (i - c)).1 - k =
          (ZKMal.CoinLayer.locate cs (k + 1) (i - c)).1 - (k + 1) + 1 by omega, List.getD_cons_succ]
        exact locate_last cs (k + 1) (i - c) h

/-- **The count of a stream whose last round is past its first**: where the round after the `q`th is the next
stream's and the `q`th is its stream's round `j + 1`, the stream before the next one has `j + 2` rounds. -/
theorem count_last {rs : List Nat} {q j : Nat} (hj : (ZKMal.CoinLayer.locate rs 0 q).2 = j + 1)
    (hnext : (ZKMal.CoinLayer.locate rs 0 (q + 1)).1 = (ZKMal.CoinLayer.locate rs 0 q).1 + 1) :
    rs.getD ((ZKMal.CoinLayer.locate rs 0 (q + 1)).1 - 1) 0 = j + 2 := by
  have h := locate_last rs 0 q (by rw [hnext]; omega)
  rw [hnext, Nat.add_sub_cancel]
  rw [Nat.sub_zero, hj] at h
  omega

open Loop in
/-- **A rep's frame from its stream's last round to the next stream's first**: at `roundRun_last`'s `RoundRun`, where
the round after it is the next stream's (`hnext`), after updates from the frames the session's next round reads that
keep the rep's frame, the frames find the rep's frame holding the history's words with `(v, a)` for all the stream's
rounds, as many as the stream's count. -/
theorem roundRun_handoff {W Rr : Type} {L : LeanRelease W Rr} {ρ : Rr} (x : Msg) (e0 : Msg × Msg)
    (h1 : List (Msg × Msg)) (v : Msg) {rs : List Nat} {q₀ j₀ : Nat} (hq : h1.length = q₀ + 1)
    (hj₀ : (ZKMal.CoinLayer.locate rs 0 q₀).2 = j₀ + 1)
    (hnext : (ZKMal.CoinLayer.locate rs 0 (q₀ + 1)).1 = (ZKMal.CoinLayer.locate rs 0 q₀).1 + 1)
    {sid : Nat} {st : State} {pl : Placed} {stream : String} {i r : Nat} {dm : ByteArray} {e : Env}
    {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream (loopNr x h1.length) i r dm ⟨v.toArray⟩ e R L0 R₃)
    (hreq : st.request pl (.round stream (loopNr x h1.length)) = none)
    {q p₁ : Rep} {pl' : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length) =
      .ok (q, pl', f, nd))
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dm ⟨v.toArray⟩ (loopNr x h1.length)).put =
      some p₁) {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (loopNr x h1.length)) a).1.stop = none) (hco : e.coins = .ok ())
    {p : Rep} (hfa : frameAt R sid i r = some p) {s : _root_.Flock.Firewall.Frames.Release} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} (hss : R.sessions.find? (·.sid == sid) = some s)
    (hn : s.spec.tables[i]? = some name) (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    {t : _root_.Flock.Firewall.Finish.Tab} {xs : Array (Array F128)} {hi : Array ByteArray} {stm : Stmt}
    {c : Array F128} {r0 : ByteArray} {q0 : Nat} (hst : s.sts[p.table]? = some stm) (hlink : s.link = some c)
    (hsalt : p.alg.size ≤ p.coins.size → p.opened = false → p.opening = none →
      R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    (htab : _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t)
    (hxs : xs = _root_.Flock.Firewall.Drive.extras stm (_root_.Flock.Firewall.Drive.linkPoints stm c))
    {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)}
    (hI : SameInv L x ρ (e0 :: h1) (h1.map fun p => ZKMal.CoinLayer.rd p.2) q₀ j₀ t
      (s.spec.domain name r).toUTF8 xs hi pre ev p)
    {sched : Array (ByteArray × Nat)}
    (hsc : schedSoFar t p.rep (s.spec.domain name r).toUTF8 xs hi p.coins = .ok sched)
    {R' : Sessions} (hk : Kept sid i r R₃ R') :
    ∃ j0 p0, R'.find sid i r = some (j0, p0) ∧ rs.getD ((ZKMal.CoinLayer.locate rs 0 (q₀ + 1)).1 - 1) 0 = j₀ + 2 ∧
      p0.coins = (Array.range (j₀ + 2)).map (fun k =>
        wordsAt x ((h1 ++ [(v, a.data.toList)]).map fun p => ZKMal.CoinLayer.rd p.2) (q₀ + 1 - (j₀ + 2) + k + 1)) := by
  obtain ⟨p₃, hat, -, hcw, -⟩ := roundRun_last x e0 h1 v hq hj₀ hr hreq hrp ho hput ha hb hs hco hfa hss hn hl0 hst
    hlink hsalt htab hxs hI hsc
  have hat' : frameAt R' sid i r = some p₃ := by
    rw [frameAt_kept hk (by rw [hat]; rfl), hat]
  obtain ⟨j0, hj0⟩ := find_of_frameAt hat'
  exact ⟨j0, p₃, hj0, count_last hj₀ hnext, hcw⟩

end Claims.ZKServer.Relay
