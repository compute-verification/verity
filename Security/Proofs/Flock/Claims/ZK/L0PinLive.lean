import Proofs.Flock.Claims.ZK.L0PinBind
import Proofs.Flock.Claims.ZK.FrameOpenLive
import Proofs.Flock.Claims.ZK.FrameTaken

/-!
# A live history's level-0 rounds

On a live history whose messages are the deferred release's own, each frame round's message is the frame's release of
the worker's round, which the frame's check passed against the firewall's scheduled round. Where the scheduled round
carries no free value, as past the algebraic phase, passing is equality byte for byte (`passes_eq`), so:

* **`LeanRelease.msg_sched`**: the message is the frame's release of the scheduled round itself.
* **`LeanRelease.msg_l0`**: at a sent level-0 round (`L0Sent`) scheduled as its zero-pad body, the message is the body at
  the pads (`L0Sent.release`).
* **`LeanRelease.msg_openBind`**: at the opening's binding, the message is the binding with `z`'s squeeze
  (`release_openBind`).
-/

namespace Claims.ZKServer

open _root_.Flock (F128)
open _root_.Flock.Firewall.Frame (Cfg Masker roundFree check)
open FlockSoundness.Refine (Frame encs)

namespace LeanRelease

/-- **A live round scheduled with no free value is the release of the scheduled round**: where the frame takes the coins
before round `j` at `m` and the scheduled round carries no free value there, round `j`'s message is the frame's release
of the scheduled round, as the device's round that passed is it byte for byte. -/
theorem msg_sched {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hl : L.toSrv.live wk x w ρ κ h = true)
    (hown : L.toSrv.OwnHist L.fillOf wk x w ρ κ h) (j : ℕ) (hj : j < h.length) (hk : L.kindAt x j = .masked)
    {m : Masker} (hA : L.answered x ρ (h.take j) = .ok m) {o : ByteArray} {t : Masker}
    (hr : roundFree (L.cfg x ρ (h.take j)) { m with stopped := false }
      ⟨(L.scheduled x w ρ (h.take j) (computedOf (L.computed x w ρ (h.take j)))).toArray⟩
      (L.coins x (h.take j).length) = .ok (o, t, #[])) :
    (h[j]'hj).1 = L.release x ρ (h.take j) false
      (L.scheduled x w ρ (h.take j) (computedOf (L.computed x w ρ (h.take j)))) := by
  have hlj : (h.take j).length = j := by
    simp only [List.length_take]
    omega
  have hk' : L.kindAt x (h.take j).length = .masked := by rw [hlj]; exact hk
  have hl' := L.live_take wk x w ρ κ h hl j
  obtain ⟨v, hv, -, hp, -⟩ := L.live_some wk x w ρ κ _ hk' hl'
  have hu : L.toSrv.wkD L.fillOf wk κ x w ρ (h.take j) 0 = some (v, false) := by
    simp only [ZkServer.wkD, hl', ↓reduceIte, hv]
    rfl
  have hck : ∃ out s', check (L.cfg x ρ (h.take j)) { m with stopped := false } ⟨v.toArray⟩
      ⟨(L.scheduled x w ρ (h.take j) (computedOf (L.computed x w ρ (h.take j)))).toArray⟩
      (L.coins x (h.take j).length) = .ok (out, s', none) := by
    unfold pass at hp
    rw [hA] at hp
    dsimp only at hp
    split at hp
    · exact ⟨_, _, by assumption⟩
    · cases hp
  obtain ⟨out, s', hc⟩ := hck
  have hv' := passes_eq hr (check_none_passes hc)
  have e : v = L.scheduled x w ρ (h.take j) (computedOf (L.computed x w ρ (h.take j))) := by
    have := congrArg Array.toList (ByteArray.mk.inj hv')
    simpa using this
  refine (hown j hj).trans ?_
  rw [L.frameD_toSrv L.fillOf wk x w ρ κ _ hk' hu, e]

/-- **A live sent level-0 round is its body at the pads**: where the frame takes the coins before round `j` at level 0's
masked phase and the round's position, and the firewall schedules round `j` as a sent level-0 round's zero-pad body,
round `j`'s message is its body at the pads, tagged `0`. -/
theorem msg_l0 {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hl : L.toSrv.live wk x w ρ κ h = true)
    (hown : L.toSrv.OwnHist L.fillOf wk x w ρ κ h) (j : ℕ) (hj : j < h.length) (hk : L.kindAt x j = .masked)
    {m : Masker} (hA : L.answered x ρ (h.take j) = .ok m) {p n : ℕ} {k : UInt8} {Bz Bh : Msg}
    (hS : FlockComplete.LigLock.L0Sent (L.cfg x ρ (h.take j)) p n k (L.coins x j) Bz Bh)
    (hsch : L.scheduled x w ρ (h.take j) (computedOf (L.computed x w ρ (h.take j))) = Bz)
    (hc : ∀ q, q < (L.cfg x ρ (h.take j)).order.size →
      (L.cfg x ρ (h.take j)).order[q]! < (L.cfg x ρ (h.take j)).h.size)
    (hon : m.on = true) (hi : m.inner = none) (hpos : m.pos = p)
    (hk2 : k = 2 ∨ (k = 1 ∧ L.coins x j = 1)) (h0 : ¬ (L.coins x j = 0 ∧ k = 2))
    (hn : L.coins x j < 2 ^ 64) (hb : p + n ≤ (L.cfg x ρ (h.take j)).order.size) :
    (h[j]'hj).1 = 0 :: Bh := by
  have hlj : (h.take j).length = j := by
    simp only [List.length_take]
    omega
  obtain ⟨G, H, hR, hBz, -⟩ := id hS
  obtain ⟨o, hro, -⟩ := roundFree_on_fr (L.cfg x ρ (h.take j)) { m with stopped := false } G k (L.coins x j) hc hon
    hi hR.mid_G hR.plain_G hk2 h0 hn (show m.pos + nAbs G ≤ _ by rw [hR.abs_G, hpos]; exact hb)
    (show _ ≤ m.pos by rw [hpos]; exact hR.piop)
  rw [L.msg_sched wk x w ρ κ h hl hown j hj hk hA (by rw [hsch, hBz, hlj]; exact hro), hsch]
  exact FlockComplete.LigLock.L0Sent.release (L := L) false (by rw [hlj]; exact hS) hA hc hon hi hpos
    (by rw [hlj]; exact hk2) (by rw [hlj]; exact h0) (by rw [hlj]; exact hn) hb

/-- **A live binding round is the binding**: where the frame takes the coins before round `j` outside a masked phase,
after the algebraic phase's opening and before level 0 closes, at a Ligerito rep, and the firewall schedules round `j` as
the opening's binding with `z`'s squeeze, round `j`'s message is that round, tagged `0`. -/
theorem msg_openBind {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hl : L.toSrv.live wk x w ρ κ h = true)
    (hown : L.toSrv.OwnHist L.fillOf wk x w ρ κ h) (j : ℕ) (hj : j < h.length) (hk : L.kindAt x j = .masked)
    {m : Masker} (hA : L.answered x ρ (h.take j) = .ok m) (T : F128) (cap : ByteArray)
    (hsch : L.scheduled x w ρ (h.take j) (computedOf (L.computed x w ρ (h.take j))) =
      encs (openBind T cap) ++ (Frame.sq 2 (L.coins x j)).enc)
    (hon : m.on = false) (hd : m.done = true) (hlig : (L.cfg x ρ (h.take j)).lig = true) (hl0 : m.doneL0 = false)
    (hi : m.inner = none) (hcap : cap.size < 2 ^ 64) (hn0 : L.coins x j ≠ 0) (hn : L.coins x j < 2 ^ 64) :
    (h[j]'hj).1 = 0 :: (encs (openBind T cap) ++ (Frame.sq 2 (L.coins x j)).enc) := by
  have hlj : (h.take j).length = j := by
    simp only [List.length_take]
    omega
  obtain ⟨o, hro, -⟩ := roundFree_openBind (L.cfg x ρ (h.take j)) { m with stopped := false } T cap (L.coins x j)
    hon hd hlig hl0 hi hcap hn0 hn
  rw [L.msg_sched wk x w ρ κ h hl hown j hj hk hA (by rw [hsch, hlj]; exact hro), hsch]
  have e := L.release_openBind x ρ (h.take j) false T cap hA hon hd hlig hl0 hi hcap (by rw [hlj]; exact hn0)
    (by rw [hlj]; exact hn)
  rw [hlj] at e
  exact e

open Loop in
/-- **A live round past the link has its rep's table and schedule**: where the loop's release is live on `e0 :: h1`
and round `|h1| + 1` is a frame round, the located rep's table is answered and its schedule so far is not refused, as
the frame's check refuses every device round in place of the empty round (`ofLoop_checkAt_empty`). -/
theorem ofLoop_sched_live (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (κ : (ofLoop T L₀ real).toSrv.Chk) (e0 : Msg × Msg) (h1 : List (Msg × Msg))
    (hl : (ofLoop T L₀ real).toSrv.live wk x w ρ κ (e0 :: h1) = true)
    (hk : (ofLoop T L₀ real).kindAt x (h1.length + 1) = .masked)
    {s0 : _} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt} {q : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1) :
    ∃ t hi sched, real x w ρ (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 = .ok (t, hi) ∧
      schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2)
        (spec.domain (spec.tables[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!)
          ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2)).toUTF8
        (_root_.Flock.Firewall.Drive.extras sts[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!
          (_root_.Flock.Firewall.Drive.linkPoints sts[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!
            (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0))) hi
        ((Array.range (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2).map
          fun j => wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2)
            (q - (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 + j + 1)) = .ok sched := by
  obtain ⟨v, -, -, hp, -⟩ := (ofLoop T L₀ real).live_some wk x w ρ κ (e0 :: h1)
    (by rw [List.length_cons]; exact hk) hl
  unfold pass at hp
  cases hA : (ofLoop T L₀ real).answered x ρ (e0 :: h1) with
  | error e => rw [hA] at hp; cases hp
  | ok m =>
    rw [hA] at hp
    dsimp only at hp
    have hck : ∃ qq, (ofLoop T L₀ real).checkAt x w ρ (e0 :: h1) m false v = .ok qq := by
      split at hp
      · exact ⟨_, by assumption⟩
      · cases hp
    obtain ⟨qq, hqq⟩ := hck
    have hcs : (h1.map fun p => ZKMal.CoinLayer.rd p.2).length = q + 1 := by rw [List.length_map, hq]
    rcases hr : real x w ρ (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 with e | ⟨t, hi⟩
    · exact absurd hqq
        (ofLoop_checkAt_empty T L₀ real x w ρ e0 h1 (schedOf_real_error (real x w ρ) x _ hx hcs hr) m false v qq)
    · rcases hs : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2)
          (spec.domain (spec.tables[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!)
            ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2)).toUTF8
          (_root_.Flock.Firewall.Drive.extras sts[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!
            (_root_.Flock.Firewall.Drive.linkPoints sts[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!
              (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0))) hi
          ((Array.range (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2).map
            fun j => wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2)
              (q - (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 + j + 1)) with e | sched
      · exact absurd hqq
          (ofLoop_checkAt_empty T L₀ real x w ρ e0 h1 (schedOf_sched_error (real x w ρ) x _ hx hcs hr hs) m false v qq)
      · exact ⟨t, hi, sched, by first | rfl | exact hr, by first | rfl | exact hs⟩

open Loop in
/-- **The loop's scheduled round is its rep's schedule's**, at the located rep round, where the rep's table is answered
and its schedule so far reaches the round. -/
theorem ofLoop_scheduled (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (vals : _)
    {s0 : _} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt} {q : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1)
    {t : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray}
    (hreal : real x w ρ (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 = .ok (t, hi))
    {sched : Array (ByteArray × ℕ)}
    (hs : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2)
      (spec.domain (spec.tables[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!)
        ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints sts[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!
          (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0))) hi
      ((Array.range (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2).map
        fun j => wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2)
          (q - (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 + j + 1)) = .ok sched)
    (hlt : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 < sched.size) :
    (ofLoop T L₀ real).scheduled x w ρ (e0 :: h1) vals =
      (sched.getD (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 (.empty, 0)).1.data.toList := by
  show (match schedOf (real x w ρ) x (h1.map fun p => ZKMal.CoinLayer.rd p.2) with
    | some r => r.data.toList
    | none => L₀.scheduled x w ρ (e0 :: h1) vals) = _
  rw [schedOf_eq (real x w ρ) x _ hx (by rw [List.length_map, hq]) hreal hs hlt]

end LeanRelease

end Claims.ZKServer
