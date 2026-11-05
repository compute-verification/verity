module

public import Proofs.Flock.ZkAny.RelayLoop
public import Proofs.Flock.ZkAny.RelaySame

/-!
# The relay's invariant at the deployed loop, kept at the control slots and the `Hello`

`LoopP`'s invariant `J` of the loop's frames and the session, against the model's run, is `LoopJ`:

* `StreamAt`: what the loop holds of a rep stream's frame at the model's history, on the session's entry and the
  table's level 0. Where the next round is the stream's, past its first, the frame keeps `SameInv` there, on the
  session's link, the table's statement and the table its level 0 gives. Where the next round is the next stream's
  first, the frame holds the history's words for each of the stream's rounds (`roundPlace_switch1`'s `hco`).
* `RepJ`: a rep's frame, the session's entry and the rep's table's level 0 bound for the session, with `StreamAt` on
  them at the model's history so far.
* `nextAt`: the model's next round's rep stream and its index there. `SeenAt`: the rep streams seen are those with a
  round before the model's next. `seenAt_nil`: none before the model's first round.
* `LoopJ`: the session stopped in the loop's frames exactly where its `Hello` left and an earlier round of the model's
  run failed; no stream seen before the `Hello` left; `RepJ` at each rep stream seen; the session's plan the setup's
  (`PlanOk`); and `SeenAt` at the model's history so far.
* `repJ_stays`: updates that keep the rep's frame and what the session's rounds read of it (`Stays`) keep `RepJ`, the
  model's history as it was.
* **`loopKept_ctl`**, `loopKept_ctl_all`: `LoopJ` is kept at every slot that reads no model message.
  **`loopKept_hello`**: and at the `Hello`, whose round the model's run passes.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session (State Placed)
open _root_.Flock (SessionSpec Stmt F128)

/-- **What the loop holds of rep stream `k`'s frame** `p` (table `k / 2`'s rep `k % 2`) at the model `M`'s history
`h`, on the session's entry `rel`, the table's name `name` and its level 0 `l0`, the next round being at
`h.length - 2` of the setup's rounds (`roundsOf`). Where that round is stream `k`'s, past its first, the frame keeps
`SameInv` there, on the session's link `c`, the table's statement and the table its level 0 gives, whose commitment's
cap is the level-0 root. Where it is stream `k + 1`'s first, the frame holds the history's words for each of stream
`k`'s rounds. -/
def StreamAt {W Rr : Type} (M : LoopModel W Rr) (spec : SessionSpec) (sts : Array Stmt) (k : Nat)
    (h : List (Claims.Msg × Claims.Msg)) (rel : _root_.Flock.Firewall.Frames.Release) (name : String)
    (l0 : _root_.Flock.Firewall.Level0.Level0) (p : _root_.Flock.Firewall.Frames.Rep) : Prop :=
  ((ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2)).1 = k →
    ∃ (j : Nat) (c : Array F128) (stm : Stmt) (t : _root_.Flock.Firewall.Finish.Tab) (xs : Array (Array F128))
      (hi : Array ByteArray) (pre : List _root_.Flock.Firewall.Frame.Req)
      (ev : List (Nat × ByteArray × _root_.Flock.Firewall.Frame.Place × Array F128)),
      (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2)).2 = j + 1 ∧ rel.link = some c ∧
        rel.sts[k / 2]? = some stm ∧
        _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t ∧
        _root_.Flock.Firewall.Finish.flat t.cm0.cap = l0.root ∧
        xs = _root_.Flock.Firewall.Drive.extras stm (_root_.Flock.Firewall.Drive.linkPoints stm c) ∧
        SameInv M.L M.x M.ρ h (h.tail.map fun e => ZKMal.CoinLayer.rd e.2) (h.length - 2) j t
          (rel.spec.domain name (k % 2)).toUTF8 xs hi pre ev p) ∧
  ((ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2)).1 = k + 1 →
    (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2)).2 = 0 →
    p.coins = (Array.range ((Loop.roundsOf spec sts).getD k 0)).map fun j =>
      Loop.wordsAt M.x (h.tail.map fun e => ZKMal.CoinLayer.rd e.2)
        (h.length - 2 - (Loop.roundsOf spec sts).getD k 0 + j + 1))

/-- **Table `i`'s rep `r` in the loop at `σ`**, against the model's run `M`: its frame, the session's entry, the table's
name and its level 0 bound for the session, with `StreamAt` at rep stream `2 * i + r` on them at the model's history
so far. -/
def RepJ {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (spec : SessionSpec) (sts : Array Stmt) (i r : Nat)
    (σ : LoopAt) : Prop :=
  ∃ (p : _root_.Flock.Firewall.Frames.Rep) (rel : _root_.Flock.Firewall.Frames.Release) (name : String)
    (l0 : _root_.Flock.Firewall.Level0.Level0),
    frameAt σ.R sid i r = some p ∧ sessAt σ.R sid = some rel ∧ rel.spec.tables[i]? = some name ∧
      _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 ∧
      StreamAt M spec sts (2 * i + r) (M.hist σ.c) rel name l0 p

/-- **The model's next round** at its history `h`: its rep stream and its index in the stream, among the setup's
rounds (`roundsOf`). -/
def nextAt (spec : SessionSpec) (sts : Array Stmt) (h : List (Claims.Msg × Claims.Msg)) : Nat × Nat :=
  ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2)

/-- **The rep streams `seen` are those with a round before the model's next**, at its history `h`: a rep stream seen
is before the next round's stream or is that stream past its first round, and that stream past its first round is
seen. -/
def SeenAt (spec : SessionSpec) (sts : Array Stmt) (h : List (Claims.Msg × Claims.Msg)) (seen : List String) : Prop :=
  ∀ i r s, (i, r, s) ∈ spec.repStreams.toList →
    (seen.contains s = true → 2 * i + r < (nextAt spec sts h).1 ∨
      (2 * i + r = (nextAt spec sts h).1 ∧ 0 < (nextAt spec sts h).2)) ∧
    (2 * i + r = (nextAt spec sts h).1 → 0 < (nextAt spec sts h).2 → seen.contains s = true)

/-- Before the model's first round no rep stream is seen. -/
theorem seenAt_nil {spec : SessionSpec} {sts : Array Stmt} {h : List (Claims.Msg × Claims.Msg)}
    (hl : h.length ≤ 2) : SeenAt spec sts h [] := by
  intro i r s _
  refine ⟨fun hc => by simp at hc, fun _ h0 => ?_⟩
  unfold nextAt at h0
  rw [show h.length - 2 = 0 by omega, Loop.locate_zero] at h0
  exact absurd h0 (Nat.lt_irrefl 0)

/-- **The relay's invariant at the deployed loop**, against the model's run `M`, at the session `st`, the streams whose
rounds came being `seen`, the loop at `σ`: the session stopped in the loop's frames exactly where its `Hello` left and
an earlier round of the model's run failed (`liveBefore`); no stream seen before the `Hello` left; `RepJ` at each rep
stream seen; the session's plan the setup's once its `Hello` left; and the streams seen those with a round before the
model's next (`SeenAt`). -/
structure LoopJ {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (spec : SessionSpec) (sts : Array Stmt)
    (seen : List String) (st : State) (σ : LoopAt) : Prop where
  stop : σ.R.stopped sid = (st.sent && !M.L.toSrv.liveBefore M.wk M.x M.w M.ρ M.κ (M.hist σ.c))
  unsent : st.sent = false → seen = []
  reps : ∀ i r s, (i, r, s) ∈ spec.repStreams.toList → seen.contains s = true → RepJ M sid spec sts i r σ
  plan : PlanOk spec sts st
  seenAt : SeenAt spec sts (M.hist σ.c) seen

/-- **`RepJ` is kept from one of the session's requests to the next** where the updates between keep the rep's frame
(`Kept`) and what the session's rounds read of it (`Stays`), and the slot took no model message. -/
theorem repJ_stays {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {i r : Nat} {σ σ' : LoopAt} (hk : Kept sid i r σ.R σ'.R) (hs : Stays sid σ σ') (hc : σ'.c = σ.c)
    (h : RepJ M sid spec sts i r σ) : RepJ M sid spec sts i r σ' := by
  obtain ⟨p, rel, name, l0, hf, hss, hn, hl0, hA⟩ := h
  refine ⟨p, rel, name, l0, ?_, ?_, hn, ?_, ?_⟩
  · rw [frameAt_kept hk (by rw [hf]; rfl)]
    exact hf
  · rw [sessAt_stay (hs.2 name l0 hl0) (by rw [hss]; rfl)]
    exact hss
  · rw [hs.1 name]
    exact hl0
  · rw [hc]
    exact hA

/-- **`LoopJ` is kept at a slot that reads no model message**: where the loop's reading of it brings only updates that
keep every rep's frame, the session's stop and what its rounds read of it, and takes no model message. -/
theorem loopKept_ctl {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {nonce : ByteArray} {k : _root_.Flock.Firewall.Session.Kind} (hm : isModel k = false)
    (hs : ∀ seen, seenAfter seen k = seen)
    (hr : ∀ st pl e σ σ', LoopRead M sid spec sts nonce k st pl e σ σ' →
      Between sid σ.R σ'.R ∧ σ'.c = σ.c ∧ Stays sid σ σ') :
    LoopKept M sid spec sts nonce (LoopJ M sid spec sts) k := by
  intro seen st pl e σ σ' _ he _ hJ hra _
  obtain ⟨hbt, hcc, hst⟩ := hr _ _ _ _ _ hra
  have hk : (Except.ok k : Except String _root_.Flock.Firewall.Session.Kind) ≠ .ok .hello := by
    intro h
    cases h
    cases hm
  have hsent := stepK_sent (hold := true) (st := st) (pl := pl) (b := some (tagOf k)) (e := e) hk
  rw [hs seen]
  refine ⟨?_, fun h => hJ.unsent (by rw [hsent] at h; exact h),
    fun i r s hmem hseen => repJ_stays (hbt.1 i r) hst hcc (hJ.reps i r s hmem hseen), stepK_plan he hJ.plan, ?_⟩
  · rw [hsent, hcc, stopped_sessKept hbt.2]
    exact hJ.stop
  · rw [hcc]
    exact hJ.seenAt

/-- **`LoopJ` is kept at every slot that reads no model message**: each such slot's reading is only the updates between
the session's requests, keeping what its rounds read of it. -/
theorem loopKept_ctl_all {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {nonce : ByteArray} : ∀ k, isModel k = false → LoopKept M sid spec sts nonce (LoopJ M sid spec sts) k := by
  intro k hm
  cases k with
  | register => exact loopKept_ctl rfl (fun _ => rfl) (fun _ _ _ _ _ h => h)
  | link => exact loopKept_ctl rfl (fun _ => rfl) (fun _ _ _ _ _ h => h)
  | openStream s => exact loopKept_ctl rfl (fun _ => rfl) (fun _ _ _ _ _ h => h)
  | proof => exact loopKept_ctl rfl (fun _ => rfl) (fun _ _ _ _ _ h => h)
  | ping => exact loopKept_ctl rfl (fun _ => rfl) (fun _ _ _ _ _ h => h)
  | other => exact loopKept_ctl rfl (fun _ => rfl) (fun _ _ _ _ _ h => h)
  | hello => cases hm
  | commit => cases hm
  | round _ _ => cases hm
  | finish => cases hm

/-- **`LoopJ` is kept at the `Hello`**: before it no stream was seen and the session had not stopped; past it the
session has sent its `Hello`, the model's `Hello` round passed, and the updates between keep the session's stop. -/
theorem loopKept_hello {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {nonce : ByteArray} : LoopKept M sid spec sts nonce (LoopJ M sid spec sts) .hello := by
  intro seen st pl e σ σ' hsl he hF hJ hra hgo
  obtain ⟨-, hbt, hcc, hlb⟩ := hra
  have hs0 : st.sent = false := (slot_hello hsl).2
  have hseen : seen = [] := hJ.unsent hs0
  have hsent := stepK_hello_sent hgo
  refine ⟨?_, fun h => ?_, fun i r s _ hc => ?_, stepK_plan he hJ.plan, ?_⟩
  · rw [hsent, hlb, stopped_sessKept hbt.2, hJ.stop, hs0] <;> rfl
  · rw [hsent] at h
    cases h
  · simp [seenAfter, hseen] at hc
  · show SeenAt spec sts (M.hist σ'.c) seen
    rw [hseen]
    exact seenAt_nil (by rw [hcc, hF.start hs0, LoopModel.hist, length_histOf_take]; omega)

end Claims.ZKServer.Relay
