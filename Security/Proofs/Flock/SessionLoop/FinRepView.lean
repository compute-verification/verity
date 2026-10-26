import Proofs.Flock.SessionLoop.FinRep

/-!
# (c.3) `FinRepFwView`, in parts

`FinRepFwView` asks that where the session's rep on the loop's table and request (`Loop.RepAt`), answered by the opened
coins, returns a proof `p`, rep stream `k`'s run from the views (`Loop.viewRep`) return a proof whose file at the
session's shape is `p`'s at the table's. The file is `encZk` of the whole proof, the final-only data (level 0's opened
rows and salts, `τ` and its salt, the pads' opened columns and salts) included, so the two proofs must be one. Its parts,
each named here, none a premise of a guarantee:

* **`Loop.FinRepErased`**: the run from the views returns, with the same transcript, a proof equal to `p` up to
  `eraseZ`: `Masked.swSrcE_repCut` at the views' source, the views' values tied to the history's by `LiveRoundEq`.
* **`Loop.FinRepFinal`**: where it returns so, the final-only data agree (`Masked.FinalEq`).
* **`Loop.FinRepXs`**: the extra claims from the views have the request's coordinates (`Drive.extras`), so as many
  claims as `claimsOf`.

**`Loop.finRepFwView_of_parts`**: in the masked scope they give `FinRepFwView`: the proofs are one
(`Masked.eq_of_eraseZ`), the table and schedule are the loop's (`Loop.finRepTab`), and the two encoders are one
(`Finish.encZk_view`, `Finish.shapeZk_view`). **`Loop.finRepProofs_of_view`** then gives (c) with `Loop.finRepCut`.
-/

namespace Flock.Firewall.Finish

open Flock

/-- The firewall's proof file is the session's encoder's. -/
theorem encZk_view (sh : Shape) (p : Zk.ZkProof) : encZk sh p = FlockSoundness.Discharge.ZkView.encZk sh p := rfl

/-- The firewall's file shape is the session's. -/
theorem shapeZk_view (st : Setup) (sch : Flock.Schedule) (n : ℕ) : shapeZk st sch n = FlockComplete.shapeZk st sch n := rfl

end Flock.Firewall.Finish

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (real : Msg → Flock.Wit → Tp → ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **The rep from the views up to `eraseZ`**: at (c.3)'s context, where the session's rep on the loop's table and
request, answered by the opened coins, returns `(p, tr)` from a history at the rep's first round, the run from the views
returns `(p', tr)` with `eraseZ p' = eraseZ p`. -/
def FinRepErased : Prop :=
  ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops →
    ∀ s0 spec, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
    ∀ schs, (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : V _) = .ok schs →
    ∀ cms, (∀ l, (commitsView vs x w ν t ops schs).go l = (.ok cms, l)) →
    ∀ k < 2 * spec.tables.size, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras →
    ∀ ic l, (padsView vs x w ν t ops schs k).go l = (.ok ic, l) → l.size = Masked.repBase x k →
    ∀ tb hi q f, RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec k tb hi q f →
    ∀ ic' p tr, (_root_.Flock.Firewall.Finish.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
        (_root_.Flock.Firewall.Finish.padsOf q.draws) (_root_.Flock.Firewall.Finish.saltAt q.draws.salts) :
        V _root_.Flock.Firewall.Finish.InnerCommit) = .ok ic' →
      (((_root_.Flock.Firewall.Finish.repFwC
          (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) tb.st tb.sch tb.tPad (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 tb.g tb.cm0.toC
          (_root_.Flock.Firewall.Finish.claimsOf q.xs) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic'.toC q.vals
          tb.extra tb.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l).1 =
        .ok (p, tr) →
      ∃ p' l', (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k extras ic).go l = (.ok (p', tr), l') ∧
        FlockComplete.eraseZ p' = FlockComplete.eraseZ p

/-- **The final-only data from the views**: at (c.3)'s context, where the session's rep returns `(p, tr)` and the run
from the views returns a proof equal to `p` up to `eraseZ`, their final-only data agree. -/
def FinRepFinal : Prop :=
  ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops →
    ∀ s0 spec, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
    ∀ schs, (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : V _) = .ok schs →
    ∀ cms, (∀ l, (commitsView vs x w ν t ops schs).go l = (.ok cms, l)) →
    ∀ k < 2 * spec.tables.size, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras →
    ∀ ic l, (padsView vs x w ν t ops schs k).go l = (.ok ic, l) → l.size = Masked.repBase x k →
    ∀ tb hi q f, RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec k tb hi q f →
    ∀ ic' p tr, (_root_.Flock.Firewall.Finish.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
        (_root_.Flock.Firewall.Finish.padsOf q.draws) (_root_.Flock.Firewall.Finish.saltAt q.draws.salts) :
        V _root_.Flock.Firewall.Finish.InnerCommit) = .ok ic' →
      (((_root_.Flock.Firewall.Finish.repFwC
          (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) tb.st tb.sch tb.tPad (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 tb.g tb.cm0.toC
          (_root_.Flock.Firewall.Finish.claimsOf q.xs) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic'.toC q.vals
          tb.extra tb.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l).1 =
        .ok (p, tr) →
      ∀ p' l', (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k extras ic).go l = (.ok (p', tr), l') →
        FlockComplete.eraseZ p' = FlockComplete.eraseZ p → Masked.FinalEq p' p

/-- **The extra claims' coordinates**: at a `Finish` `FinishIs` reads, rep stream `k`'s extra claims from the views are
at its request's coordinates. -/
def FinRepXs : Prop :=
  ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops →
    ∀ s0 spec, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
    ∀ k < 2 * spec.tables.size, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras →
    ∀ tb hi q f, RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec k tb hi q f →
      extras.map (·.x) = q.xs

/-- **The session's table and schedule are the loop's**: in the masked scope, rep stream `k`'s table is table `k / 2`
of the session's (`tabsOf`), and its `--zk` schedule and padding are the session's (`schedsPZk`). -/
theorem finRepTab (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) {s0 spec}
    (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) {schs}
    (hschs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : V _) = .ok schs) {k : ℕ}
    (hk : k < 2 * spec.tables.size) {ρ h tb hi q f} (hR : RepAt L₀.draws src x w ρ h spec k tb hi q f) :
    (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! = tb.st ∧ schs[k / 2]! = (tb.sch, tb.tPad) := by
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  obtain ⟨-, -, hT, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hj : k / 2 < (Masked.stsOf x).size := by omega
  obtain ⟨hr, -, -⟩ := hR
  unfold realOf at hr
  rw [hsx] at hr
  try dsimp only at hr
  obtain ⟨⟨g, d, hi'⟩, -, hr⟩ := bok hr
  obtain ⟨l, -, hr⟩ := bok hr
  obtain ⟨tb', htb, hr⟩ := bok hr
  have htb' : tb' = tb := by cases hr; rfl
  rw [htb'] at htb
  obtain ⟨hts, hsch⟩ := Count.tabOfLevel0_sched htb
  have hj' : k / 2 < (FlockDeploy.tabsOf (Masked.stsOf x)).size := by simpa [FlockDeploy.tabsOf] using hj
  have htab : (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! = tb.st := by
    rw [hts, getElem!_pos (FlockDeploy.tabsOf (Masked.stsOf x)) (k / 2) hj', getElem!_pos (Masked.stsOf x) (k / 2) hj]
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
  refine ⟨htab, ?_⟩
  unfold FlockComplete.schedsPZk at hschs
  obtain ⟨hsz, hall⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem hschs
  have hk' : k / 2 < schs.size := by rw [hsz]; exact hj'
  have e : _root_.Flock.Zk.schedule (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2].m = .ok schs[k / 2] :=
    hall (k / 2) hj' hk'
  rw [getElem!_pos schs (k / 2) hk']
  rw [← getElem!_pos (FlockDeploy.tabsOf (Masked.stsOf x)) (k / 2) hj', htab, hts, hsch] at e
  exact (Except.ok.inj e).symm

/-- **The parts give (c.3)**: the run from the views returns a proof equal to the session's up to `eraseZ`
(`FinRepErased`) whose final-only data are the session's (`FinRepFinal`), so it is the session's proof
(`Masked.eq_of_eraseZ`); its table, schedule (`finRepTab`) and claim count (`FinRepXs`) are the loop's, and the two
encoders are one. -/
theorem finRepFwView_of_parts (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (h : FinRepErased vs L₀ src real wk x w ∧ FinRepFinal vs L₀ src real wk x w ∧ FinRepXs L₀ src real wk x w) :
    FinRepFwView vs L₀ src real wk x w := by
  obtain ⟨hE, hF, hX⟩ := h
  intro ν t κ m₀ ρ₀ ops hctx s0 spec hsx schs hschs cms hcms k hk extras hxe ic l hic hl tb hi q f hR ic' p tr hic' hrun
  obtain ⟨p', l', hv, hz⟩ := hE ν t κ m₀ ρ₀ ops hctx s0 spec hsx schs hschs cms hcms k hk extras hxe ic l hic hl
    tb hi q f hR ic' p tr hic' hrun
  have hfe := hF ν t κ m₀ ρ₀ ops hctx s0 spec hsx schs hschs cms hcms k hk extras hxe ic l hic hl
    tb hi q f hR ic' p tr hic' hrun p' l' hv hz
  have hpp : p' = p := Masked.eq_of_eraseZ hz hfe
  obtain ⟨htab, hsch⟩ := finRepTab (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hschs hk hR
  have hxs := hX ν t κ m₀ ρ₀ ops hctx s0 spec hsx k hk extras hxe tb hi q f hR
  have hsz : extras.size = (_root_.Flock.Firewall.Finish.claimsOf q.xs).size := by
    rw [← hxs]
    simp only [_root_.Flock.Firewall.Finish.claimsOf, Array.size_map]
  refine ⟨p', tr, l', hv, ?_⟩
  rw [hpp, htab, hsch, hsz, _root_.Flock.Firewall.Finish.encZk_view, _root_.Flock.Firewall.Finish.shapeZk_view]

/-- **(c) from (c.3)'s parts**, with `FinRepCut` in the masked scope (`finRepCut`). -/
theorem finRepProofs_of_view (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (h : FinRepErased vs L₀ src real wk x w ∧ FinRepFinal vs L₀ src real wk x w ∧ FinRepXs L₀ src real wk x w) :
    FinRepProofs vs L₀ src real wk x w :=
  finRepProofs_of_parts vs L₀ src real wk x w
    ⟨finRepCut L₀ src real wk x w hsc, finRepFwView_of_parts vs L₀ src real wk x w hsc h⟩

end Loop

end Claims.ZKServer
