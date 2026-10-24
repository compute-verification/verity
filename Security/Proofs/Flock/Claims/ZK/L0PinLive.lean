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

end LeanRelease

end Claims.ZKServer
