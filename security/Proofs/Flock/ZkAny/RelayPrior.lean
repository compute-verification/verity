module

public import Proofs.Flock.ZkAny.RelaySame

/-!
# A rep 1 stream's first round, at the model's state

At rep 1's first round the loop readies its frame after rep 0's algebraic-phase coins, which the model's start of rep 1
(`Loop.freshOf`) takes too, there as the history's words for rep 0's rounds:

* `roundPlace_prior`: at rep 1's first round, on its frame's run on no requests (`RepRun [] []`), the frame
  `roundPlace` readies has the found frame's configuration and is its run on the coins of rep 0's frame for the rep's
  algebraic phase (`repRun_first1`).
* **`roundPlace_switch1`**: so where rep 0's frame holds the history's words for rep 0's rounds and the rep's
  algebraic phase has as many scheduled rounds as the placeholder's, the frame `roundPlace` readies is its run on
  rep 0's algebraic-phase words, at the model's state for a rep 1 stream's first round (`repRun_prior_fresh`).
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (F128 SessionSpec Stmt)
open _root_.Flock.Firewall.Frames (Sessions Rep Release)
open _root_.Flock.Firewall.Frame (Cfg Masker Place)

set_option maxHeartbeats 1000000 in
/-- **At rep 1's first round the readied frame is its run on rep 0's algebraic-phase coins**: on its frame's run on
no requests, the found frame with its algebraic phase's scheduled rounds (`withAlg`) takes the coins of rep 0's frame
for them, at the found frame's configuration. -/
theorem roundPlace_prior {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i j j0 : Nat}
    {dom content : ByteArray} {n : Nat} {p p0 p' q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hf : R.find sid i 1 = some (j, p))
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i 1 dom content n = .ok (q, pl, f, nd))
    (hR : RepRun [] [] p) {s : Release} {name : String} {l0 : _root_.Flock.Firewall.Level0.Level0}
    (hs : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    (hw : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok p') (h0 : R.find sid i 0 = some (j0, p0)) :
    q.cfg = p.cfg ∧ RepRun [.prior ((p0.coins.extract 0 p'.alg.size).foldl (· ++ ·) #[]).size
      (_root_.Flock.Firewall.Frame.wordBytes ((p0.coins.extract 0 p'.alg.size).foldl (· ++ ·) #[]))] [] q := by
  have hfst : (p.coins.isEmpty && !p.st.begun) = true := (first_repRun hR fun _ hm => by cases hm).trans rfl
  have h11 : ((1 : Nat) == 1) = true := rfl
  unfold _root_.Flock.Firewall.Frames.roundPlace at h
  simp only [hf, bind, Except.bind, pure, Except.pure] at h
  repeat' (first
    | simp only [hs, hn, hl0, hfst, hw, h11, h0, bind, Except.bind, pure, Except.pure, Bool.false_eq_true,
        ↓reduceIte] at h
    | split at h)
  all_goals first
    | (cases h; done)
    | exact absurd (by assumption) throw_ne_ok
    | (cases h
       exact ⟨by
         have e := (placeOf_run' (by assumption)).1
         exact e.trans (withAlg_run hw).1, repRun_first1 hR hw (by assumption) (by assumption)⟩)

/-- **At rep 1's first round the readied frame is at the model's state**: on its frame's run on no requests, where
rep 0's frame holds the history's words for rep 0's `n0` rounds, the found frame's configuration is of one layout with
the placeholder's and the rep's algebraic phase has as many scheduled rounds as the placeholder's, the frame
`roundPlace` readies is its run on rep 0's algebraic-phase words, at the model's state for a rep 1 stream's first
round (`Loop.freshOf`). -/
theorem roundPlace_switch1 {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i j j0 : Nat}
    {dom content : ByteArray} {n : Nat} {p p0 p' q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hf : R.find sid i 1 = some (j, p))
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i 1 dom content n = .ok (q, pl, f, nd))
    (hR : RepRun [] [] p) {s : Release} {name : String} {l0 : _root_.Flock.Firewall.Level0.Level0}
    (hs : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    (hw : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok p') (h0 : R.find sid i 0 = some (j0, p0))
    (x : Msg) (cs : List FlockDeploy.Coin) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q₀ n0 : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : cs.length = q₀ + 1)
    (hk : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 % 2 ≠ 0)
    (hn0 : (Loop.roundsOf spec sts).getD ((ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 - 1) 0 = n0)
    {c' : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom' : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray}
    (hph : Loop.placeholder spec sts (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 (Loop.wordsAt x cs 0) =
      .ok (c', t, dom', xs, hi)) {alg : Array (ByteArray × Nat)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom' =
      .ok alg)
    (hc : CfgAlike p.cfg c') (hsz : p'.alg.size = alg.size)
    (hco : p0.coins = (Array.range n0).map fun k => Loop.wordsAt x cs (q₀ - n0 + k + 1))
    {m : Masker} (hm : Loop.freshOf x cs = .ok m) :
    RepRun [.prior (((Array.range n0).map fun k => Loop.wordsAt x cs (q₀ - n0 + k + 1)).extract 0 alg.size
        |>.foldl (· ++ ·) #[]).size
      (_root_.Flock.Firewall.Frame.wordBytes
        (((Array.range n0).map fun k => Loop.wordsAt x cs (q₀ - n0 + k + 1)).extract 0 alg.size
          |>.foldl (· ++ ·) #[]))] [] q ∧ q.st.m = m := by
  obtain ⟨hcq, hrun⟩ := roundPlace_prior hf h hR hs hn hl0 hw h0
  rw [hco, hsz] at hrun
  obtain ⟨c'', t'', dom'', xs'', hi'', alg'', hph', hal', hcase⟩ := Loop.freshOf_ok x cs hx hq hm
  rw [hph] at hph'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hph'
  obtain ⟨rfl, rfl, rfl, rfl, rfl⟩ := hph'
  simp only [hal, Except.ok.injEq] at hal'
  subst hal'
  rcases hcase with ⟨hk0, -⟩ | ⟨-, st', hst', rfl⟩
  · exact absurd hk0 hk
  · rw [hn0] at hst'
    exact ⟨hrun, by rw [repRun_prior_fresh hrun (by rw [hcq]; exact hc) hst']⟩

end Claims.ZKServer.Relay
