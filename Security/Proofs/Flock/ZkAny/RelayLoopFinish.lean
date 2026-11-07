module

public import Proofs.Flock.ZkAny.RelayLoopCommit

/-!
# The `Finish`'s slot case at the deployed loop

At the `Finish` the relay reads the loop's own proofs once the hold's barrier passes, and the model's message there is
the deferred release's last: the ZK server's own `Finish` where every round passed, else the stop. The session goes on
past no `Finish`, so only what the relay read there is the model's. Here is `envPre_loop`'s step at the `Finish`:

* `slot_finish`: the `Finish`'s slot comes after the `Hello` left.
* `nextD_finish`: at the last round, where every earlier round passed, the server refuses none of it and the worker's
  value is the ZK server's own, the deferred release is the ZK server's own message.
* **`slotCase_finish`**: where the run's messages are the deferred release's and, at the `Finish`, the session's stop
  is whether an earlier round failed, what the relay read at the `Finish` is the model's last message.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session
open _root_.Flock.Firewall hiding State
open _root_.Flock (SessionSpec Stmt)
open _root_.Flock.Firewall.Frames (Sessions)

/-- **The `Finish`'s slot comes after the `Hello` left.** -/
theorem slot_finish {S : State} {P : Placed} (h : slot S P = some .finish) : S.sent = true := by
  unfold slot at h
  split at h; · simp at h
  split at h; · simp at h
  rename_i h2
  simpa using h2

/-- **At the last round the deferred release is the ZK server's own message**, where every earlier round passed, the
ZK server's own `Finish` does not stop and the worker's value is the ZK server's own: the run is live there. -/
theorem nextD_finish {W Rr : Type} (L : LeanRelease W Rr) (n : Claims.Msg → Nat) (wk : L.toSrv.Worker)
    (x : Claims.Msg) (w : W) (ρ : Rr) (κ : L.toSrv.Chk) (h : List (Claims.Msg × Claims.Msg))
    (hl : h.length = L.last x) (h0 : h.length ≠ 0) (hlb : L.toSrv.liveBefore wk x w ρ κ h = true)
    (hv : wk x w ρ h 0 = some (L.ownMsg x w ρ h, false)) (hf : L.finStops x w ρ h = false) :
    L.toSrv.nextD L.fillOf n wk x w ρ κ h = L.ownMsg x w ρ h := by
  have hk : L.kindAt x h.length = .recomputed := by
    unfold LeanRelease.kindAt
    rw [ite_eq_right h0, ite_eq_left (Or.inr hl : h.length = 1 ∨ h.length = L.last x)]
  have hr : L.toSrv.refuses x w ρ h = false := by
    change (decide (h.length = L.last x) && L.finStops x w ρ h) = false
    rw [hf, Bool.and_false]
  have hp : L.toSrv.pass wk x w ρ κ h = true := by
    show (!L.toSrv.refuses x w ρ h && (List.range 1).all _) = _
    simp only [List.range_one, List.all_cons, List.all_nil, Bool.and_true]
    rw [hr, Bool.not_false, Bool.true_and]
    unfold ZkServer.check
    rw [show L.toSrv.kind x h.length 0 = .recomputed from hk, hv]
    simp only [decide_eq_true_eq]
    rfl
  have hc : h.length + 1 < n x ∨ L.toSrv.live wk x w ρ κ h = true := by
    refine Or.inr ?_
    rw [ZkServer.live_eq_liveBefore, hlb, hp, Bool.and_true]
  unfold ZkServer.nextD
  rw [ite_eq_left hc]
  exact L.frameD_recomp L.fillOf wk x w ρ κ h hk

/-- **The `Finish`'s slot case**: where each of the run's messages is the deferred release's and, at the `Finish`,
the session's stop is whether an earlier round of the model's run failed, what the relay read at the `Finish` is the
model's last message: once the hold's barrier passes, the session has not stopped, so every earlier round passed, and
the run is live at its last round, its message the ZK server's own `Finish` of the loop's proof files. The session goes
on past no `Finish`. -/
theorem slotCase_finish {W Rr : Type} {M : LoopModel W Rr} {J : List String → State → LoopAt → Prop} {sid : Nat}
    {spec : SessionSpec} {sts : Array Stmt} {nonce : ByteArray} (hOwn : M.Own)
    (hJs : ∀ seen st pl σ, slot st pl = some .finish → LoopFacts sid spec seen st pl σ → J seen st σ →
      σ.R.stopped sid = !M.L.toSrv.liveBefore M.wk M.x M.w M.ρ M.κ (M.hist σ.c)) :
    SlotCase spec (LoopP M J sid spec sts nonce) .finish := by
  intro seen st pl e rest ms hsl hP
  obtain ⟨σ, hms, hF, hj, hRd, -⟩ := hP
  cases hRd with
  | stop h =>
    rw [hsl] at h
    cases h
  | step k' hk' hra _ =>
    obtain rfl : Session.Kind.finish = k' := Option.some.inj (hsl.symm.trans hk')
    obtain ⟨hlt, hfin, -⟩ := hra
    refine ⟨fun _ => ?_, fun hgo => ?_⟩
    · show e.barrier none = none → finishOf spec (ms.headD []) = some e.proofs
      intro hb
      obtain ⟨hst, hlast, hfs, hv, f, hf, hpr, hb1, hb2, hb3, hb4⟩ := hfin hb
      have hlen : (M.hist σ.c).length = M.L.last M.x := by rw [M.length_hist (Nat.le_of_lt hlt), hlast]
      have h0 : (M.hist σ.c).length ≠ 0 := by
        rw [M.length_hist (Nat.le_of_lt hlt), hF.count (slot_finish hsl)]
        exact Nat.succ_ne_zero _
      have hlb : M.L.toSrv.liveBefore M.wk M.x M.w M.ρ M.κ (M.hist σ.c) = true := by
        have hs := hJs seen st pl σ hsl hF hj
        rw [hst] at hs
        simpa using hs.symm
      rw [hms, M.msgs_drop_headD hlt, hOwn σ.c hlt, nextD_finish M.L M.nN M.wk M.x M.w M.ρ M.κ _ hlen h0 hlb hv hfs,
        hf]
      exact envOk_finish (seen := seen) hb1 hb2 hb3 hb4 (fun _ => hpr) hb
    · rw [stepK_finish] at hgo
      exact absurd hgo (by simp)

end Claims.ZKServer.Relay
