import Proofs.Flock.SessionLoop.Statement

/-!
# The link at the session loop's own level 0

The loop's `Commit` carries each table's level-0 root that `Level0.ofDraws` drew for the session (`Session.relay`'s
`commitFrame` of `Level0.ofSession`'s roots). Here that link is held to `LeanRelease.LinkIs` at the deployed tables:

* `Loop.level0Of`, `Loop.linkOf`: the level 0 the loop draws for table `j` from a level-0 source `src` (at rep stream
  `2j`'s source, as `Loop.realOf` reads it), and the link message of their roots.
* `Loop.LinkSrc`: the level 0 drawn for each table is the tape's, `Masked.linkIn` (the host's packed witness at the
  tape's mask words, and the tape's extra lanes, lane paddings and level-0 salts in the operating system's bytes).
* **`LeanRelease.linkIs_ofLoop`**: at a `Live` statement, where the release's link is the loop's (`Loop.linkOf src`) and
  `src` draws the tape's level 0, the firewall's link is `linkMsg` (`LinkIs`), so `LinkRoundEq` holds
  (`linkRoundEq_ofLoop`, by `linkRoundEq_of_linkIs`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked
open _root_.Flock (V F128 Stmt)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

/-- **The level 0 the loop draws for table `j`** at the source `src`: `Level0.ofDraws` at table `j`'s `m` on the packed
witness and drawn bytes `src` gives rep stream `2j`. -/
def Loop.level0Of {R : Type} (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (ρ : R) (sts : Array Stmt) (j : ℕ) : V _root_.Flock.Firewall.Level0.Level0 := do
  let (g, d, _) ← src x w ρ (2 * j)
  _root_.Flock.Firewall.Level0.ofDraws sts[j]!.m g d

/-- **The loop's own link** at the source `src`: each table's drawn level-0 root (the empty root where the draw is
refused, which the loop's `Commit` check refuses), with no public words. -/
def Loop.linkOf {R : Type} (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (ρ : R) : Msg :=
  match FlockDeploy.setupOf true x with
  | some (_, _, sts) =>
    0 :: FlockDeploy.linkBody
      (List.ofFn fun j : Fin sts.size => ((Loop.level0Of src x w ρ sts j.val).toOption.map (·.root)).getD .empty)
      (List.replicate sts.size ByteArray.empty)
  | none => []

/-- **The loop draws the tape's level 0** at `x`: at every nonce and tape, table `j`'s drawn level 0 is `Masked.linkIn`'s
(a trust in the host's packed witness and the operating system's bytes, not in how a root is committed). -/
def Loop.LinkSrc (src : Msg → Flock.Wit →
      (session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (session Loop.loopNr dataE).Nonce) (t : (session Loop.loopNr dataE).Tape) (j : Fin (jOf x)),
    ∃ l, Loop.level0Of src x w (ν, t) (stsOf x) j.val = .ok l ∧ l.toIn = linkIn x w t j

/-- A drawn level 0's root is `Link.root` of its own inputs (`Level0.ofDraws_ok`). -/
theorem Loop.level0Of_root {R : Type} {src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray)}
    {x : Msg} {w : Flock.Wit} {ρ : R} {sts : Array Stmt} {j : ℕ} {l : _root_.Flock.Firewall.Level0.Level0}
    (h : Loop.level0Of src x w ρ sts j = .ok l) : l.root = _root_.Flock.Firewall.Link.root l.toIn := by
  unfold Loop.level0Of at h
  rcases hs : src x w ρ (2 * j) with e | ⟨g, d, hi⟩
  · rw [hs] at h; cases h
  · rw [hs] at h
    exact (_root_.Flock.Firewall.Level0.ofDraws_ok h).2.2.2.2.2

namespace LeanRelease

/-- **The loop's link is the recheck's link message** at a `Live` statement, where the release's link is the loop's
(`Loop.linkOf src`) and `src` draws the tape's level 0 (`Loop.LinkSrc`). -/
theorem linkIs_ofLoop
    (L₀ : LeanRelease Flock.Wit ((session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (hL : L₀.link = fun x w ρ _ => Loop.linkOf src x w ρ) {x : Msg} (hx : Live x) {w : Flock.Wit}
    (hs : Loop.LinkSrc src x w) : LinkIs Loop.loopNr (ofLoop dataE L₀ real) x w := by
  intro ν t m₀ ρ₀ _ _
  show L₀.link x w (ν, t) [(m₀, ρ₀)] = linkMsg x w t
  rw [hL]
  show Loop.linkOf src x w (ν, t) = linkMsg x w t
  unfold Loop.linkOf linkMsg
  rcases hsx : FlockDeploy.setupOf true x with _ | ⟨s0, spec, sts⟩
  · rfl
  · have hst : stsOf x = sts := by
      unfold stsOf
      rw [hsx]
    have hn : jOf x = sts.size := by
      simp only [jOf, hx, ↓reduceIte, hst]
    simp only [Option.isNone_some, Bool.false_eq_true, ↓reduceIte]
    congr 2
    · rw [List.ofFn_congr hn.symm]
      congr 1
      funext j
      obtain ⟨l, hl, hi⟩ := hs ν t j
      rw [hst] at hl
      simp only [Fin.val_cast]
      rw [hl, ← hi, ← Loop.level0Of_root hl]
      rfl
    · rw [hn]

/-- **`LinkRoundEq` at the loop's own link**: `linkIs_ofLoop`, then `linkRoundEq_of_linkIs`. -/
theorem linkRoundEq_ofLoop (vs : (session Loop.loopNr dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit ((session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (hL : L₀.link = fun x w ρ _ => Loop.linkOf src x w ρ) {x : Msg} (hx : Live x) {w : Flock.Wit}
    (hs : Loop.LinkSrc src x w) : LinkRoundEq (session Loop.loopNr dataE) vs (ofLoop dataE L₀ real) x w :=
  linkRoundEq_of_linkIs Loop.loopNr vs _ x w (linkIs_ofLoop L₀ src real hL hx hs)

end LeanRelease

end Claims.ZKServer
