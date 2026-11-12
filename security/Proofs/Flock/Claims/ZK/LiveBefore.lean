module

public import Proofs.Flock.Claims.ZK.Release

/-!
# The model's stop, round by round

The deferred release stands in for the worker once a round of the session failed (`ZkServer.wkD`, at `ZkServer.live`,
which reads every round so far, the current one's included). Here is that bit one round at a time, in the terms the
session loop's stop (`Frames.Sessions.stopped`) is stated in:

* `ZkServer.liveBefore`: every round before the one after `h` passed; `live_eq_liveBefore`: a history is live where
  every earlier round passed and the current one does; `liveBefore_snoc`: one round on, every earlier round passed
  exactly where the history was live; `liveBefore_snoc_pass`: so a round that passes leaves it as it was.
* `LeanRelease.takes_of_pass`: a device round the frame passes it takes; `cross_and_pass`: where Rust's checks fail
  only device rounds the frame fails, they decide nothing beside it.
* `LeanRelease.toSrv_pass_frame`, `toSrv_pass_own`, `toSrv_pass_recomp`: below the `Finish`, the server's pass at a
  frame round is Rust's checks' and the frame's pass decision on the worker's device round; at the `Hello` it passes;
  at the link it passes where the worker's value is the ZK server's own.
* **`LeanRelease.nextD_frame_stop`**: so at a frame round below the `Finish` the deferred release is the Lean frame's
  release of the device round where no earlier round failed (`b`, the stop bit) and it passes, else of the scheduled
  round as a stopped rep's; **`stop_frame_snoc`**: and one round on, the stop bit is again whether a round failed.
* `nextD_frame_cross`, `liveBefore_snoc_cross`: a frame round Rust's checks fail is released as the scheduled round
  of a stopped rep, and one round on a round has failed.
-/

@[expose] public section

namespace Claims.ZKServer

namespace ZkServer

/-- **Every round before the one after `h` passed.** -/
def liveBefore {W R : Type} (S : ZkServer W R) (wk : S.Worker) (x : Msg) (w : W) (ρ : R) (κ : S.Chk)
    (h : List (Msg × Msg)) : Bool :=
  (List.range h.length).all fun i => S.pass wk x w ρ κ (h.take i)

/-- **A history is live where every earlier round passed and the current one does.** -/
theorem live_eq_liveBefore {W R : Type} (S : ZkServer W R) (wk : S.Worker) (x : Msg) (w : W) (ρ : R) (κ : S.Chk)
    (h : List (Msg × Msg)) : S.live wk x w ρ κ h = (S.liveBefore wk x w ρ κ h && S.pass wk x w ρ κ h) := by
  unfold live liveBefore
  rw [List.range_succ, List.all_append]
  simp only [List.all_cons, List.all_nil, Bool.and_true, List.take_length]

/-- **One round on, every earlier round passed exactly where the history was live.** -/
theorem liveBefore_snoc {W R : Type} (S : ZkServer W R) (wk : S.Worker) (x : Msg) (w : W) (ρ : R) (κ : S.Chk)
    (h : List (Msg × Msg)) (e : Msg × Msg) : S.liveBefore wk x w ρ κ (h ++ [e]) = S.live wk x w ρ κ h := by
  unfold liveBefore live
  rw [List.length_append, List.length_singleton]
  apply Bool.eq_iff_iff.mpr
  simp only [List.all_eq_true, List.mem_range]
  refine forall_congr' fun i => imp_congr_right fun hi => ?_
  rw [List.take_append_of_le_length (by omega)]

/-- **A round that passes leaves whether every earlier round passed as it was.** -/
theorem liveBefore_snoc_pass {W R : Type} (S : ZkServer W R) (wk : S.Worker) (x : Msg) (w : W) (ρ : R) (κ : S.Chk)
    (h : List (Msg × Msg)) (e : Msg × Msg) (hp : S.pass wk x w ρ κ h = true) :
    S.liveBefore wk x w ρ κ (h ++ [e]) = S.liveBefore wk x w ρ κ h := by
  rw [liveBefore_snoc, live_eq_liveBefore, hp, Bool.and_true]

end ZkServer

namespace LeanRelease

/-- **A device round the frame passes it takes.** -/
theorem takes_of_pass {W R : Type} (L : LeanRelease W R) {x : Msg} {w : W} {ρ : R} {h : List (Msg × Msg)} {v : Msg}
    (hp : L.pass x w ρ h v = true) : L.takes x ρ h v = true := by
  unfold pass at hp
  unfold takes step
  cases hm : L.answered x ρ h with
  | error e =>
    rw [hm] at hp
    cases hp
  | ok m =>
    rw [hm] at hp
    simp only at hp
    show (match _root_.Flock.Firewall.Frame.round (L.cfg x ρ h) { m with stopped := false } ⟨v.toArray⟩
        (L.coins x h.length) with
      | .ok _ => true
      | .error _ => false) = true
    revert hp
    cases hc : _root_.Flock.Firewall.Frame.check (L.cfg x ρ h) { m with stopped := false } ⟨v.toArray⟩
        ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩ (L.coins x h.length) with
    | error e =>
      intro hp
      cases hp
    | ok t =>
      obtain ⟨out, s', why⟩ := t
      cases why with
      | none =>
        intro _
        rw [check_none hc]
      | some e =>
        intro hp
        cases hp

/-- **Where Rust's checks fail only device rounds the frame fails, they decide nothing beside it.** -/
theorem cross_and_pass {W R : Type} {L : LeanRelease W R} {x : Msg} {j : ℕ} {w : W} {ρ : R} {h : List (Msg × Msg)}
    {v : Msg} (hcr : L.cross x j w ρ h v = false → L.pass x w ρ h v = false) :
    (L.cross x j w ρ h v && L.pass x w ρ h v) = L.pass x w ρ h v := by
  cases hc : L.cross x j w ρ h v
  · rw [hcr hc]
    rfl
  · rfl

/-- The server refuses no round below the `Finish`. -/
theorem toSrv_refuses_lt {W R : Type} (L : LeanRelease W R) {x : Msg} {w : W} {ρ : R} {h : List (Msg × Msg)}
    (hlt : h.length < L.last x) : L.toSrv.refuses x w ρ h = false := by
  change (decide (h.length = L.last x) && L.finStops x w ρ h) = false
  rw [decide_eq_false (Nat.ne_of_lt hlt), Bool.false_and]

/-- **Below the `Finish` the server's pass at a frame round is Rust's checks' and the frame's pass decision** on the
worker's device round `v`. -/
theorem toSrv_pass_frame {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked) (hlt : h.length < L.last x)
    {v : Msg} (hv : wk x w ρ h 0 = some (v, false)) :
    L.toSrv.pass wk x w ρ κ h = (L.cross x h.length w ρ h v && L.pass x w ρ h v) := by
  show (!L.toSrv.refuses x w ρ h && (List.range 1).all _) = _
  simp only [List.range_one, List.all_cons, List.all_nil, Bool.and_true]
  rw [L.toSrv_refuses_lt hlt, Bool.not_false, Bool.true_and]
  unfold ZkServer.check
  rw [show L.toSrv.kind x h.length 0 = .masked from hk, hv]
  change (!false && decide (h.length < L.last x) && L.cross x h.length w ρ h v && L.pass x w ρ h v &&
    L.takes x ρ h v) = _
  rw [decide_eq_true hlt]
  cases hp : L.pass x w ρ h v
  · cases L.cross x h.length w ρ h v <;> rfl
  · rw [L.takes_of_pass hp]
    cases L.cross x h.length w ρ h v <;> rfl

/-- **The `Hello` passes**, below the `Finish`. -/
theorem toSrv_pass_own {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .own) (hlt : h.length < L.last x) :
    L.toSrv.pass wk x w ρ κ h = true := by
  show (!L.toSrv.refuses x w ρ h && (List.range 1).all _) = _
  simp only [List.range_one, List.all_cons, List.all_nil, Bool.and_true]
  rw [L.toSrv_refuses_lt hlt, Bool.not_false, Bool.true_and]
  unfold ZkServer.check
  rw [show L.toSrv.kind x h.length 0 = .own from hk]

/-- **The link passes where the worker's value is the ZK server's own**, below the `Finish`. -/
theorem toSrv_pass_recomp {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .recomputed) (hlt : h.length < L.last x)
    (hv : wk x w ρ h 0 = some (L.ownMsg x w ρ h, false)) : L.toSrv.pass wk x w ρ κ h = true := by
  show (!L.toSrv.refuses x w ρ h && (List.range 1).all _) = _
  simp only [List.range_one, List.all_cons, List.all_nil, Bool.and_true]
  rw [L.toSrv_refuses_lt hlt, Bool.not_false, Bool.true_and]
  unfold ZkServer.check
  rw [show L.toSrv.kind x h.length 0 = .recomputed from hk, hv]
  simp only [decide_eq_true_eq]
  rfl

/-- **The deferred release at a frame round below the `Finish`** is the Lean frame's release of the worker's device
round `v` where every earlier round passed and `v` passes Rust's checks and the frame's, else of the scheduled round
as a stopped rep's. -/
theorem nextD_frame {W R : Type} (L : LeanRelease W R) (n : Msg → ℕ) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked) (hn : h.length + 1 < n x)
    (hlt : h.length < L.last x) {v : Msg} (hv : wk x w ρ h 0 = some (v, false)) :
    L.toSrv.nextD L.fillOf n wk x w ρ κ h =
      if (L.toSrv.liveBefore wk x w ρ κ h && L.cross x h.length w ρ h v && L.pass x w ρ h v) = true then
        L.release x ρ h false v
      else L.release x ρ h true (L.scheduled x w ρ h (computedOf (L.computed x w ρ h))) := by
  have hl : L.toSrv.live wk x w ρ κ h =
      (L.toSrv.liveBefore wk x w ρ κ h && L.cross x h.length w ρ h v && L.pass x w ρ h v) := by
    rw [ZkServer.live_eq_liveBefore, L.toSrv_pass_frame wk x w ρ κ h hk hlt hv, Bool.and_assoc]
  unfold ZkServer.nextD
  rw [ite_eq_left (Or.inl hn), ← hl]
  cases hlv : L.toSrv.live wk x w ρ κ h
  · rw [L.frameD_toSrv L.fillOf wk x w ρ κ h hk (u := L.fillOf x h.length 0 w ρ h)
      (by unfold ZkServer.wkD; simp only [hlv, Bool.false_eq_true, ↓reduceIte])]
    rfl
  · rw [L.frameD_toSrv L.fillOf wk x w ρ κ h hk (u := (v, false))
      (by unfold ZkServer.wkD; simp only [hlv, ↓reduceIte]; exact hv)]
    rfl

/-- **The deferred release at a frame round below the `Finish`, at the stop bit**: where `b` is whether an earlier
round failed and Rust's checks fail only device rounds the frame fails, it is the Lean frame's release of the device
round `v` where `b` is off and `v` passes the frame's pass decision, else of the scheduled round as a stopped rep's. -/
theorem nextD_frame_stop {W R : Type} (L : LeanRelease W R) (n : Msg → ℕ) (wk : L.toSrv.Worker) (x : Msg) (w : W)
    (ρ : R) (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked) (hn : h.length + 1 < n x)
    (hlt : h.length < L.last x) {v : Msg} (hv : wk x w ρ h 0 = some (v, false))
    (hcr : L.cross x h.length w ρ h v = false → L.pass x w ρ h v = false) {b : Bool}
    (hb : b = !L.toSrv.liveBefore wk x w ρ κ h) :
    L.toSrv.nextD L.fillOf n wk x w ρ κ h =
      if (!b && L.pass x w ρ h v) = true then L.release x ρ h false v
      else L.release x ρ h true (L.scheduled x w ρ h (computedOf (L.computed x w ρ h))) := by
  rw [L.nextD_frame n wk x w ρ κ h hk hn hlt hv, Bool.and_assoc, cross_and_pass hcr, hb, Bool.not_not]

/-- **One frame round on, the stop bit is again whether a round failed**: where `b` was whether an earlier round
failed, it or the round's failing the frame's pass decision is whether a round up to it failed. -/
theorem stop_frame_snoc {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked) (hlt : h.length < L.last x)
    {v : Msg} (hv : wk x w ρ h 0 = some (v, false))
    (hcr : L.cross x h.length w ρ h v = false → L.pass x w ρ h v = false) {b : Bool}
    (hb : b = !L.toSrv.liveBefore wk x w ρ κ h) (e : Msg × Msg) :
    (b || !L.pass x w ρ h v) = !L.toSrv.liveBefore wk x w ρ κ (h ++ [e]) := by
  rw [ZkServer.liveBefore_snoc, ZkServer.live_eq_liveBefore, L.toSrv_pass_frame wk x w ρ κ h hk hlt hv,
    cross_and_pass hcr, hb, Bool.not_and]

/-- **A frame round Rust's checks fail is released as the scheduled round of a stopped rep**, below the `Finish`. -/
theorem nextD_frame_cross {W R : Type} (L : LeanRelease W R) (n : Msg → ℕ) (wk : L.toSrv.Worker) (x : Msg) (w : W)
    (ρ : R) (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked) (hn : h.length + 1 < n x)
    (hlt : h.length < L.last x) {v : Msg} (hv : wk x w ρ h 0 = some (v, false))
    (hc : L.cross x h.length w ρ h v = false) :
    L.toSrv.nextD L.fillOf n wk x w ρ κ h =
      L.release x ρ h true (L.scheduled x w ρ h (computedOf (L.computed x w ρ h))) := by
  rw [L.nextD_frame n wk x w ρ κ h hk hn hlt hv, hc, Bool.and_false, Bool.false_and,
    ite_eq_right Bool.false_ne_true]

/-- **One frame round on from a round Rust's checks fail, a round has failed.** -/
theorem liveBefore_snoc_cross {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked) (hlt : h.length < L.last x)
    {v : Msg} (hv : wk x w ρ h 0 = some (v, false)) (hc : L.cross x h.length w ρ h v = false) (e : Msg × Msg) :
    L.toSrv.liveBefore wk x w ρ κ (h ++ [e]) = false := by
  rw [ZkServer.liveBefore_snoc, ZkServer.live_eq_liveBefore, L.toSrv_pass_frame wk x w ρ κ h hk hlt hv, hc,
    Bool.false_and, Bool.and_false]

end LeanRelease

end Claims.ZKServer
