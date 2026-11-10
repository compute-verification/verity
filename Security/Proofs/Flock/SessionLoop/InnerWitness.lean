module

public import Proofs.Flock.SessionLoop.Inner
public import Proofs.Flock.Claims.Recursion.Assumptions.LoopFixture

/-!
# `LoopInnerRound0Own` says something

Both lemmas are at a deployed session loop's run, `Loop.ReachesInner` (the named assumption of
`Proofs.Flock.Claims.Recursion.Assumptions.LoopFixture`), whose history stands at a rep stream's inner proof's round 0:

* `Flock.SecurityProofs.LoopInnerRound0Own_honest`: there `LoopInnerRound0`'s premises hold at the deployed tables
  (`Masked.dataE`) and the loop's own tables (`Loop.realOf`), and the loop's own release (`withInner`, at every `L₀` and
  finish) releases the inner proof's label frame and the finish's `final_c'` between its headers.
* `Flock.SecurityProofs.LoopInnerRound0Own_nontrivial`: there `LoopInnerRound0` fails at `LeanRelease.silent`, the
  release that releases nothing, whose round is empty where the inner proof's round 0 is at least its two headers.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (labelFrame wordBytes words)
open _root_.Flock.Firewall.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- the session's tape types, as `LoopInnerRound0Own`'s in `Inner`

/-- **The release that releases nothing**: every round, link, file and stop empty, every check refusing. -/
def LeanRelease.silent {W R : Type} : LeanRelease W R where
  rep _ _ := 0
  lay _ _ := default
  slot _ _ := none
  draws _ _ := .empty
  prog _ _ := {}
  coinsOf _ _ := #[]
  fresh _ _ := .error ""
  device _ _ _ _ _ := []
  computed _ _ _ _ := default
  cross _ _ _ _ _ _ := false
  scheduled _ _ _ _ _ := []
  coins _ _ := 0
  sched _ _ := 0
  stop := []
  hello _ _ := []
  helloChk _ _ := .error ""
  req _ _ := ("", 0)
  link _ _ _ _ := []
  files _ _ _ _ := ("", #[])
  finStops _ _ _ _ := false
  last _ := 0

/-- **`LoopInnerRound0`'s premises hold at a deployed session loop's run, and there the loop's own release releases the
inner proof's round 0**: at the fixture's run, `x` sets up a session, the history stands at rep stream `k`'s round
`j + 1`, the rep's table and salts are the loop's own (`Loop.realOf src`) and its schedule so far is `j + 1` rounds; at
every `L₀`, finish `fin` and values, `withInner`'s round there is the inner proof's label frame, then the first word of
the finish's round 0 past its label frame and header (`final_c'`) between the round's headers. -/
theorem _root_.Flock.SecurityProofs.LoopInnerRound0Own_honest (hfix : Loop.ReachesInner) :
    ∃ (src : Msg → Flock.Wit →
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
          (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand → ℕ → V (Array F128 × ByteArray × Array ByteArray))
      (x : Msg) (w : Flock.Wit)
      (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)
      (h : List (Msg × Msg)) (s0 : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt) (q k j : ℕ) (tr : Tab)
      (hr : Array ByteArray),
      FlockDeploy.setupOf true x = some (s0, spec, sts) ∧ h.length = q + 2 ∧
      CoinLayer.locate (Loop.roundsOf spec sts) 0 q = (k, j + 1) ∧ Loop.realOf src x w ρ k = .ok (tr, hr) ∧
      Except.map Array.size (Loop.schedSoFar tr (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
        (_root_.Flock.Firewall.Drive.extras sts[k / 2]! (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]!
          (Loop.wordsAt x (h.tail.map fun p => CoinLayer.rd p.2) 0))) hr
        ((Array.range (j + 1)).map
          fun i => Loop.wordsAt x (h.tail.map fun p => CoinLayer.rd p.2) (q - (j + 1) + i + 1))) = .ok (j + 1) ∧
      ∀ (L₀ : LeanRelease Flock.Wit
          ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
            (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
        (fin : Msg → Flock.Wit →
          (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
            (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand → List (Msg × Msg) → Array ByteArray)
        (vals : String → Option ByteArray),
        (LeanRelease.withInner Masked.dataE L₀ (Loop.realOf src) fin).scheduled x w ρ h vals =
          (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++
            wordBytes (words ((fin x w ρ h).getD 0 .empty)
              ((labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).size + 16) 1) ++ header 4 1 1).data.toList := by
  obtain ⟨src, x, w, ρ, h, s0, spec, sts, q, k, j, tr, hr, rsched, hx, hlen, hloc, hreal, hs, hsize⟩ := hfix
  refine ⟨src, x, w, ρ, h, s0, spec, sts, q, k, j, tr, hr, hx, hlen, hloc, hreal, by rw [hs, ← hsize]; rfl, ?_⟩
  intro L₀ fin vals
  have hq : (h.tail.map fun p => CoinLayer.rd p.2).length = q + 1 := by
    simp only [List.length_map, List.length_tail]; omega
  have h0 : j + 1 - rsched.size = 0 := by omega
  have e := Loop.innerOf_eq (Loop.realOf src x w ρ)
    ((LeanRelease.withFrames Masked.dataE L₀ (Loop.realOf src)).cfg x ρ h) (fin x w ρ h) x _ hx hq hloc hreal hs
    (by omega) (by rw [h0]; decide)
  rw [h0] at e
  show (match Loop.innerOf (Loop.realOf src x w ρ)
      ((LeanRelease.withFrames Masked.dataE L₀ (Loop.realOf src)).cfg x ρ h) (fin x w ρ h) x
      (h.tail.map fun p => CoinLayer.rd p.2) with
    | some r => r.data.toList
    | none => L₀.scheduled x w ρ h vals) = _
  rw [e]
  rfl

/-- **`LoopInnerRound0` rules out a release that releases nothing**: at the fixture's run, at the deployed tables and
the loop's own tables, `LoopInnerRound0` fails at `LeanRelease.silent`, whose round at the inner proof's round 0 is
empty, where the ZK server's is the inner proof's label frame and `final_c'` between two 16-byte headers. -/
theorem _root_.Flock.SecurityProofs.LoopInnerRound0Own_nontrivial (hfix : Loop.ReachesInner) :
    ∃ (src : Msg → Flock.Wit →
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
          (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand → ℕ → V (Array F128 × ByteArray × Array ByteArray))
      (x : Msg) (w : Flock.Wit),
      ¬ _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE LeanRelease.silent (Loop.realOf src)
        x w := by
  obtain ⟨src, x, w, ρ, h, s0, spec, sts, q, k, j, tr, hr, rsched, hx, hlen, hloc, hreal, hs, hsize⟩ := hfix
  refine ⟨src, x, w, fun hL => ?_⟩
  obtain ⟨w', -, he⟩ := hL ρ h (fun _ => none) s0 spec sts q k j tr hr rsched hx hlen hloc hreal hs hsize
  change [] = _ at he
  have e := congrArg List.length he
  simp only [List.length_nil, ByteArray.data_append, Array.toList_append, List.length_append] at e
  have h3 : 8 ≤ (header 3 1 1).data.toList.length := by
    rw [_root_.Flock.Transcript.header, ByteArray.data_append, Array.toList_append, List.length_append]
    exact Nat.le_add_right _ _
  omega

end Claims.ZKServer
