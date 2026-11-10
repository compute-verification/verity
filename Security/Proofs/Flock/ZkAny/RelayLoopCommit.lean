module

public import Proofs.Flock.ZkAny.RelayLoop
public import Proofs.Flock.ZkAny.RelayEnv

/-!
# The `Commit`'s slot case at the deployed loop

At the `Commit` the relay reads the roots `commitRead` reads off the loop's level-0 slots, and the model's message
there is the deferred release's link. Here is `envPre_loop`'s step at the `Commit`:

* `LoopModel.length_hist`: the history before the run's `j`-th message has `j` rounds; `LoopModel.msgs_drop_headD`:
  the run's next message past `j` is its `j`-th.
* `nextD_link`: at the link (round `1`), below the last round, the deferred release is the ZK server's own message,
  whatever the worker sends.
* **`slotCase_commit`**: where the run's messages are the deferred release's and `J` is kept at the `Commit`, what the
  relay read at the `Commit` is the model's link, whose roots are the level 0 the loop drew for the session, and
  `LoopP` holds at the next request with that message taken and one more opening.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session
open _root_.Flock.Firewall hiding State
open _root_.Flock (SessionSpec Stmt)
open _root_.Flock.Firewall.Frames (Sessions)

/-- **The history before the run's `j`-th message has `j` rounds.** -/
theorem LoopModel.length_hist {W Rr : Type} (M : LoopModel W Rr) {j : Nat} (hj : j ≤ M.τ.length) :
    (M.hist j).length = j := by
  rw [LoopModel.hist, length_histOf_take, Nat.min_eq_left hj]

/-- **The run's next message past its first `j` is its `j`-th.** -/
theorem LoopModel.msgs_drop_headD {W Rr : Type} (M : LoopModel W Rr) {j : Nat} (hj : j < M.τ.length) :
    (M.msgs.drop j).headD [] = (M.τ[j]'hj).1 := by
  unfold LoopModel.msgs
  rw [← List.map_drop, List.drop_eq_getElem_cons hj, List.map_cons, List.headD_cons]

/-- **At the link the deferred release is the ZK server's own message**, below the last round, whatever the worker
sends. -/
theorem nextD_link {W Rr : Type} (L : LeanRelease W Rr) (n : Claims.Msg → Nat) (wk : L.toSrv.Worker)
    (x : Claims.Msg) (w : W) (ρ : Rr) (κ : L.toSrv.Chk) (h : List (Claims.Msg × Claims.Msg)) (h1 : h.length = 1)
    (hn : 2 < n x) : L.toSrv.nextD L.fillOf n wk x w ρ κ h = L.ownMsg x w ρ h := by
  have hk : L.kindAt x h.length = .recomputed := by
    rw [h1]
    simp [LeanRelease.kindAt]
  have hc : h.length + 1 < n x ∨ L.toSrv.live wk x w ρ κ h = true := Or.inl (by omega)
  unfold ZkServer.nextD
  rw [ite_eq_left hc]
  exact L.frameD_recomp L.fillOf wk x w ρ κ h hk

/-- **The `Commit`'s slot case**: where each of the run's messages is the deferred release's, the run has more than
two, and `J` is kept at the `Commit`, what the relay read at the `Commit` is the model's link, the ZK server's own
message there, whose roots are the level 0 the loop drew for the session; and `LoopP` holds at the next request with
that message taken, the session having opened one more. -/
theorem slotCase_commit {W Rr : Type} {M : LoopModel W Rr} {J : List String → State → LoopAt → Prop} {sid : Nat}
    {spec : SessionSpec} {sts : Array Stmt} {nonce : ByteArray} (hOwn : M.Own) (hn : 2 < M.nN M.x)
    (hJ : LoopKept M sid spec sts nonce J .commit) :
    SlotCase spec (LoopP M J sid spec sts nonce) .commit := by
  intro seen st pl e rest ms hsl hP
  obtain ⟨σ, hms, hF, hj, hRd, hset⟩ := hP
  have hc1 : σ.c = 1 := by
    have h1 := hF.count (slot_commit hsl).1
    have h2 := (slot_commit hsl).2.2
    omega
  cases hRd with
  | stop h =>
    rw [hsl] at h
    cases h
  | step k' hk' hra hrest =>
    obtain rfl : Session.Kind.commit = k' := Option.some.inj (hsl.symm.trans hk')
    have hJ' := hJ seen st pl e σ _ hsl (hset e List.mem_cons_self) hF hj hra
    obtain ⟨⟨hlt, -⟩, ⟨f, hf, -⟩, ⟨ct, hok⟩, ⟨ls, publics, hda, -, hlk, hb1, hb2, hb3, hb4⟩, -, hbt, hcc⟩ := hra
    have hk : (Except.ok Session.Kind.commit : Except String Session.Kind) ≠ .ok .hello := fun h => by cases h
    refine ⟨fun _ => ?_, fun hgo => ?_⟩
    · have hlen : (M.hist σ.c).length = 1 := by rw [M.length_hist (Nat.le_of_lt hlt), hc1]
      rw [hms, M.msgs_drop_headD hlt, hOwn σ.c hlt, nextD_link M.L M.nN M.wk M.x M.w M.ρ M.κ _ hlen hn, hlk]
      exact envOk_commit_after hf hok hda hb1 hb2 hb3 hb4
    · refine ⟨_, ?_, ⟨?_, ?_, (go_live hgo).2, stepK_reach hF.reach hF.live hsl hgo, ?_, ?_⟩, hJ' hgo,
        hrest hgo, fun e' he' => hset e' (List.mem_cons_of_mem _ he')⟩
      · simp only [isModel, ↓reduceIte]
        rw [hcc, hms, List.tail_drop]
      · exact stepK_setup (hset e List.mem_cons_self) hF.setup
      · intro i r s hmem hseen
        exact freshAt_kept (hbt.1 i r) (hF.fresh i r s hmem hseen)
      · intro h
        rw [stepK_sent hk, (slot_commit hsl).1] at h
        cases h
      · intro _
        rw [hcc, stepK_coin_opened hF.reach hF.live hsl (.inl rfl) hgo, hF.count (slot_commit hsl).1]

end Claims.ZKServer.Relay
