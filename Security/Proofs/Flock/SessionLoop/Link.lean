import Proofs.Flock.SessionLoop.Statement
import Proofs.Flock.Claims.ZK.MaskedTables

/-!
# The link at the session loop's own level 0

The loop's `Commit` carries each table's level-0 root that `Level0.ofDraws` drew for the session (`Session.relay`'s
`commitFrame` of `Level0.ofSession`'s roots). Here that link is held to `LeanRelease.LinkIs` at the deployed tables:

* `Loop.level0Of_root`: a drawn level 0's root is `Link.root` of its own inputs (`Level0.ofDraws_ok`).
* **`LeanRelease.linkIs_ofLoop`**: at a `Live` statement, where the release's link is the loop's (`Loop.linkOf src`,
  `Statement`) and `src` draws the tape's level 0 (`Loop.LinkLevel0`), the firewall's link is `linkMsg` (`LinkIs`), so
  `LinkRoundEq` holds (`linkRoundEq_ofLoop`, by `linkRoundEq_of_linkIs`).
* **`LeanRelease.linkRoundEq_ofLoopLink`**: that at `ofLoopLink`, whose link is the loop's by definition, at a statement
  in the masked scope with a witness (`Masked.live_of`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked
open _root_.Flock (V F128 Stmt)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

/-- A drawn level 0's root is `Link.root` of its own inputs (`Level0.ofDraws_ok`). -/
theorem Loop.level0Of_root {R : Type} {src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray)}
    {x : Msg} {w : Flock.Wit} {ρ : R} {sts : Array Stmt} {j : ℕ} {l : _root_.Flock.Firewall.Level0.Level0}
    (h : Loop.level0Of src x w ρ sts j = .ok l) : l.root = _root_.Flock.Firewall.Link.root l.toIn := by
  unfold Loop.level0Of at h
  rcases hs : src x w ρ (2 * j) with e | ⟨g, d, hi⟩
  · rw [hs] at h; cases h
  · rw [hs] at h
    exact (_root_.Flock.Firewall.Level0.ofDraws_ok h).2.2.2.2.2

/-- The link's level 0 is rep stream `2j`'s (`Loop.level0At`). -/
theorem Loop.level0Of_eq_level0At {R : Type}
    (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray)) (x : Msg) (w : Flock.Wit) (ρ : R)
    (sts : Array Stmt) (j : ℕ) : Loop.level0Of src x w ρ sts j = Loop.level0At src x w ρ sts (2 * j) := by
  unfold Loop.level0Of Loop.level0At
  rw [Nat.mul_div_cancel_left j (by omega : 0 < 2)]

/-- **`Loop.Src`'s level-0 half gives the link's level 0**: rep `0`'s. -/
theorem Loop.Level0Src.linkLevel0 {src : Msg → Flock.Wit →
      (session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray)} {x : Msg} {w : Flock.Wit}
    (h : Loop.Level0Src src x w) : Loop.LinkLevel0 src x w := by
  intro ν t j
  obtain ⟨l, hl, hin⟩ := h ν t j 0
  exact ⟨l, by rw [Loop.level0Of_eq_level0At]; exact hl, hin⟩

namespace LeanRelease

/-- **The loop's link is the recheck's link message** at a `Live` statement, where the release's link is the loop's
(`Loop.linkOf src`) and `src` draws the tape's level 0 (`Loop.LinkLevel0`). -/
theorem linkIs_ofLoop
    (L₀ : LeanRelease Flock.Wit ((session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (hL : L₀.link = fun x w ρ _ => Loop.linkOf src x w ρ) {x : Msg} (hx : Live x) {w : Flock.Wit}
    (hs : Loop.LinkLevel0 src x w) : LinkIs Loop.loopNr (ofLoop dataE L₀ real) x w := by
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
    (hs : Loop.LinkLevel0 src x w) : LinkRoundEq (session Loop.loopNr dataE) vs (ofLoop dataE L₀ real) x w :=
  linkRoundEq_of_linkIs Loop.loopNr vs _ x w (linkIs_ofLoop L₀ src real hL hx hs)

/-- **`LinkRoundEq` at the loop's own frames and link**: at a statement in the masked scope with a witness, which is
`Live` (`Masked.live_of`), where `src` draws the tape's level 0. -/
theorem linkRoundEq_ofLoopLink (vs : (session Loop.loopNr dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit ((session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (session Loop.loopNr dataE).refP.Nonce × (session Loop.loopNr dataE).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    {x : Msg} {w : Flock.Wit} (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hs : Loop.LinkLevel0 src x w) :
    LinkRoundEq (session Loop.loopNr dataE) vs (ofLoopLink L₀ src real) x w :=
  linkRoundEq_ofLoop vs (withFiles (withLink L₀ src) src) src real rfl (live_of hsc hR) hs

end LeanRelease

end Claims.ZKServer
