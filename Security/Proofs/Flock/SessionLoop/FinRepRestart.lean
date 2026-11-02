import Proofs.Flock.SessionLoop.FinRepRows
import Proofs.Flock.Claims.ZK.FinishFwRestart

/-!
# `FinRepRestart`: level 0's restart is `Σ y₁'·bFold`

At a `Finish` `FinishIs` reads, the firewall's rep stream `2j + r` (`Loop.RepRun`) runs its Ligerito on the session's
table and packed table, with the tape's extra lanes (`Loop.finRepTab`, `Loop.Level0Src.realOf`), and the request's
extra claims at the region points of the view's link points (`Loop.linkPoints_wordsAt`). There `Masked.fwRestart_at`
reads the proof's level-0 restart `t'` as `Σ_col y₁'(col)·bFold(col)`.

**`Loop.finRepRestart`**: in the masked scope, with a witness, at a statement the loop serves whose level 0 the source
draws from the tape, `FinRepRestart` holds. It reads the firewall's run, its coins, the tape and the witness, never the
workers' clear values or the rows: `y₁'` reads the witness lanes, the mask slot and the extra lanes.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge
open FlockSoundness.Refine (fOf gOf)
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (real : Msg → Flock.Wit → Tp → ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **`FinRepRestart` at a served statement**: at table `j`'s rep `r`'s run, the run is on the session's table and
packed table with the tape's extra lanes (`finRepTab`, `schedsPZk_live`, `Level0Src.realOf`), and the request's extra
coordinates are the statement's region points at the view's link points (`linkPoints_wordsAt`), so
`Masked.fwRestart_at` gives `t' = Σ y₁'·bFold`. -/
theorem finRepRestart (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    (hS : Loop.Level0Src src x w) :
    FinRepRestart vs L₀ src real wk x w := by
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  have hx : Masked.Live x := Masked.live_of hsc hR
  intro ν t κ m₀ ρ₀ ops s0 spec j r tb hi q f ic' l p tr l' hrr
  obtain ⟨hctx, hsx, hRep, -, hl, hgo⟩ := hrr
  have hj : j.val < (Masked.stsOf x).size := Masked.lt_of_jOf j
  have hk : 2 * j.val + r.val < 2 * spec.tables.size := by
    obtain ⟨-, -, hTs, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
    have := r.isLt
    omega
  have hk2 : (2 * j.val + r.val) / 2 = j.val := by
    have := r.isLt
    omega
  have hkr : (2 * j.val + r.val) % 2 = r.val := by
    have := r.isLt
    omega
  obtain ⟨schs, hschs, hschZ⟩ := Masked.schedsPZk_live hx
  -- the session's table, schedule, packed table and extra lanes
  obtain ⟨htb, hsb⟩ := finRepTab (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hschs hk hRep
  rw [hk2] at htb hsb
  have hst : tb.st = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) := htb.symm.trans (Masked.tabsOf_stAt x j)
  have hsch : tb.sch = (Masked.zkTab x j).sch := (Prod.mk.inj (hsb.symm.trans (hschZ j))).1
  obtain ⟨-, hext, -, hg⟩ := hS.realOf ν t j r hRep.1
  -- the extra claims sit at the region points of the view's link points
  have hqx : q.xs = _root_.Flock.Firewall.Drive.extras (Masked.stAt x j) (viewPoints x ops spec) := by
    have hq := hRep.2.1
    unfold repInOf at hq
    obtain ⟨⟨lay, ms⟩, -, hq⟩ := bok hq
    have hqx : q.xs = _root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
          (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0)) := by
      cases hq
      rfl
    rw [linkPoints_wordsAt x w hsc hR hctx.1 hsx hk m₀ ρ₀, hk2,
      show (Masked.stsOf x)[j.val]! = Masked.stAt x j from getElem!_pos (Masked.stsOf x) j.val hj] at hqx
    exact hqx
  have hxs : (_root_.Flock.Firewall.Finish.claimsOf q.xs).map (·.x) =
      _root_.Flock.Firewall.Drive.extras (Masked.stAt x j) (Masked.ptsE (finCoins x ops) spec) := by
    have hmx : (_root_.Flock.Firewall.Finish.claimsOf q.xs).map (·.x) = q.xs := by
      apply Array.ext'
      simp [_root_.Flock.Firewall.Finish.claimsOf, Function.comp_def]
    rw [hmx, hqx]
    rfl
  have hxsz : (_root_.Flock.Firewall.Finish.claimsOf q.xs).size = spec.points * (Masked.stAt x j).regions.size := by
    simp only [_root_.Flock.Firewall.Finish.claimsOf, Array.size_map]
    rw [hqx]
    unfold _root_.Flock.Firewall.Drive.extras
    rw [flatMap_size_const _ _ _ fun _ => Array.size_map]
    simp only [viewPoints, Array.size_map, Array.size_range]
    exact Nat.mul_comm _ _
  rw [hkr, hk2, hsch] at hgo
  exact Masked.fwRestart_at x hx w t (finCoins x ops) (refK x) s0 spec hsx j r (vs x w ν (finCoins x ops))
    (_root_.Flock.Firewall.Finish.claimsOf q.xs) hxsz hxs tb.st hst tb.tPad _ tb.g hg tb.cm0.toC
    (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic'.toC q.vals tb.extra tb.lanePad hext _ hl hgo

end Loop

end Claims.ZKServer
