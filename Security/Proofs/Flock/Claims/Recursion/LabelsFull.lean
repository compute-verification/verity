import Proofs.Flock.Claims.Recursion.VerdictRowEOpens
import Proofs.Flock.Claims.Recursion.CapOpens
import Proofs.Flock.C3.RegLevels.AccReads
import Proofs.Flock.Claims.Recursion.DeployPremises
import Proofs.Flock.Claims.Recursion.SlotsCheck
import Proofs.Flock.Claims.Recursion.DeployScope
import Proofs.Flock.Claims.Recursion.DeployOuter
import Proofs.Flock.Claims.JsonNorm
import Proofs.Flock.Claims.Complete.Json.Stmt
import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Proofs.Flock.C3.PubRows.Check
import Proofs.Flock.Recursive.Witness
import Proofs.Flock.Claims.KS.PlainWrP
import Proofs.Flock.Claims.Bridge.Custody
import Proofs.Flock.Claims.DeployTables
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts

/-!
# A staged part's table labels its instances `0..n-1`

`accRow_table` takes a staged part's table 0 to label its instances in order (`hidx`). At every accepted decision that
holds, from the decision's own checks (`labels_full`):

* the record's draw is the statement's: `checks` compares them up to key order (`drawEq`), and both are rebuilt
  (`Flock.norm`: `recordDraw_norm`, `insO_draw_norm`), so they are equal (`eq_of_rel_norm`);
* the verdict's last refusals (`fullDrawO`) decode that draw to every unit of its population, with no closure
  (`fullDraw_some`, `Witness.fullDraw_ok`), so the units it proves are `range pop`;
* the setup's `oneFull` leaves one table (`count ≤ 1`), set up at the statement's draw, whose instances are the units the
  draw proves (`setupOne_stmt`, `drawn_spec`);
* under the inner shape the setup ran `checkLabels` (`RecIns.setup_labels`), which at one table checks table 0's labels
  against the draw itself (`tableDraw d 1 0 = d`).

`labels_run` and `labels_stagedAll` state it at each part an accepted run stages (`stagedPart_run`, `stagedAll_run`).
-/

namespace RecDeploy

open Flock
open Claims (Msg Out Prover run coinSeq replay)
open FlockDeploy (Coin)
open Lean (Json)

/-- The checks after the verdict compare the record's draw with the statement's (`drawEq`). -/
theorem checks_drawEq {zk : Bool} {s : FlockDeploy.Ins} {spec : SessionSpec} {x : Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Msg × Coin)} {f : FlockDeploy.Final}
    (h : FlockDeploy.checks zk s spec x link c₁ mids f = .ok ()) :
    FlockDeploy.drawEq (FlockSoundness.Discharge.Exec.recordDraw f.record) s.read.draw = true := by
  unfold FlockDeploy.checks at h
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨_, h, -⟩ := FlockSoundness.Refine.ebind_ok h
  unfold Flock.Record.check at h
  split at h
  · assumption
  · simp [throw, throwThe, MonadExceptOf.throw] at h

/-- A record's draw is rebuilt (`Flock.norm`). -/
theorem recordDraw_norm {r : String} {d : Json} (h : FlockSoundness.Discharge.Exec.recordDraw r = some d) :
    Flock.norm d = d := by
  unfold FlockSoundness.Discharge.Exec.recordDraw at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (obtain rfl := Option.some.inj h; exact FlockDeploy.norm_norm _)

/-- A recursive statement's draw is rebuilt (`Flock.norm`, `Raw.ins_draw`). -/
theorem insO_draw_norm {x : Msg} {i : RecIns} (h : insO x = some i) {d : Json} (hd : i.base.read.draw = some d) :
    Flock.norm d = d := by
  unfold insO at h
  obtain ⟨w, -, hw⟩ := Option.bind_eq_some_iff.1 h
  split_ifs at hw with hz
  · cases hw
  change w.ins.toOption = some i at hw
  cases hi : w.ins with
  | error e => rw [hi] at hw; cases hw
  | ok s' =>
    rw [hi] at hw
    cases hw
    obtain ⟨draw, -, hdr⟩ := FlockDeploy.Raw.ins_draw (RecRaw.ins_base hi)
    rw [hdr] at hd
    obtain ⟨d₀, -, rfl⟩ := Option.map_eq_some_iff.1 hd
    exact FlockDeploy.norm_norm d₀

/-- Under an inner shape the recursive setup ran `RecVStar.checkLabels` at the statement's draw. -/
theorem RecIns.setup_labels {s : RecIns} {spec : SessionSpec} {sts : Array Stmt} (h : s.setup = .ok (spec, sts))
    {n : RecVStar.Inner} (hn : s.inner = some n) :
    RecVStar.checkLabels s.base.read.draw s.base.read.I.count sts = .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure, hn, Option.isSome_some, ↓reduceIte] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals first
    | (obtain ⟨-, h2⟩ := Prod.mk.inj (Except.ok.inj h)
       subst h2
       assumption)
    | simp_all

/-- Under the verifier's inner shape, `RecSetup.fullDraw` at a draw passes only where the draw decodes and the
witness's draw check passes at its population. -/
theorem fullDraw_some {n : RecVStar.Inner} {rec : Bool} {d : Json} {record : String}
    (h : (RecSetup.fullDraw (some n) rec (some d) record).isOk = true) :
    ∃ U, Draw.UnitDraw.ofJson d = .ok U ∧ Flock.Witness.fullDraw U.population record = .ok () := by
  unfold RecSetup.fullDraw at h
  cases hU : Draw.UnitDraw.ofJson d with
  | error e =>
    simp [hU, bind, Except.bind, pure, Except.pure, Except.isOk, Except.toBool, throw, throwThe,
      MonadExceptOf.throw] at h
  | ok U =>
    refine ⟨U, rfl, ?_⟩
    cases hw : Flock.Witness.fullDraw U.population record with
    | error e =>
      simp [hU, hw, bind, Except.bind, pure, Except.pure, Except.isOk, Except.toBool, Except.mapError, throw,
        throwThe, MonadExceptOf.throw] at h
    | ok u => cases u; rfl

/-- **At an accepted decision, table 0 labels its instances `0..n-1`**: where the recursive setup decodes under the
verifier's inner shape, and the decision's checks and its last refusals pass, table 0's `units.indices` are
`range pub.n`. -/
theorem labels_full {x : Msg} {i : RecIns} {spec spec' : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Msg × Coin)} {f : FlockDeploy.Final}
    (hsu : setupO x = some (i, spec, sts)) (hin : i.inner.isSome = true) (h0 : 0 < sts.size)
    (hck : FlockDeploy.checks true i.base spec' x link c₁ mids f = .ok ()) (hfd : fullDrawO x f = true) :
    Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n) := by
  obtain ⟨hio, hset⟩ := setupO_some hsu
  obtain ⟨n, hn⟩ := Option.isSome_iff_exists.1 hin
  obtain ⟨ds, hds⟩ := FlockDeploy.KS.checks_draw hck
  have hrel := (FlockDeploy.drawEq_iff _ _).1 (checks_drawEq hck)
  have heq : FlockSoundness.Discharge.Exec.recordDraw f.record = i.base.read.draw :=
    FlockDeploy.eq_of_rel_norm hrel (fun _ h => recordDraw_norm h) (fun _ h => insO_draw_norm hio h)
  rw [hds] at heq
  have hfull := fullDrawO_fullDraw hfd
  have hinn : innerOfO x = some n := by simp [innerOfO, hio, hn]
  have hwd : Flock.Witness.recordDraw f.record = some ds := heq
  rw [hinn, hwd] at hfull
  obtain ⟨U, hU, hw⟩ := fullDraw_some hfull
  obtain ⟨d', D, hd', hD, -, -, hunits, hcl⟩ := FlockSoundness.Discharge.Recursion.Witness.fullDraw_ok hw
  rw [heq] at hd'
  cases hd'
  rw [hU] at hD
  cases hD
  have hprov : U.proved = Array.range U.population := by
    simp [Draw.UnitDraw.proved, hcl, hunits]
  have hread := RecIns.setup_read hset
  have hpi := Claims.Bridge.Custody.publicInputs_none_of_checks hck
  have hof := RecIns.setup_oneFull hset
  rw [hn] at hof
  obtain ⟨h1, -⟩ := oneFull_ok hof
  have hsr := hread
  unfold Flock.Readable.Read.setup at hsr
  rw [hpi] at hsr
  have hsz := FlockDeploy.setupTables_size hsr
  have hc1 : i.base.read.I.count ≤ 1 := by
    rw [h1] at hsz
    exact (le_max_right 1 _).trans hsz.symm.le
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hsr
  obtain ⟨d, hd, hdd⟩ := hall 0 h0
  rw [hdd hc1, hds] at hd
  obtain ⟨⟨c, pop, -, hdr⟩, -⟩ := FlockComplete.JsonRT.setupOne_stmt hd
  obtain ⟨D', hD', -, hnp, -⟩ := FlockSoundness.Discharge.Layout.drawn_spec (hdr ds rfl)
  rw [hU] at hD'
  cases hD'
  have hlab := RecIns.setup_labels hset hn
  rw [hds] at hlab
  have hl0 := FlockC3.PubRows.checkLabels_at hlab 0 h0
  rw [show RecVStar.tableDraw ds i.base.read.I.count 0 = ds from if_pos hc1] at hl0
  cases hu : Registered.unitIndices sts[0].pub with
  | error e => simp [RecVStar.checkLabel, hU, hu, bind, Except.bind] at hl0
  | ok idx => rw [FlockC3.PubRows.checkLabel_ok hl0 hU hu, hnp, hprov, Array.size_range]

section StagedRun

variable (H : Msg → Msg) (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
  (r : Fin ((RecDeploy.V IV st).k x) → Coin)

/-- **An accepted run's staged parts label their instances `0..n-1`**: part `s`'s session decided and accepted
(`run_decides`, `nextO_decide`), at the part's statement as `stagedPart_run` stages it, whose inner shape is the
verifier's (`insO_staged`), so `labels_full` holds at its setup. -/
theorem labels_run (hacc : (run H P (RecDeploy.V IV st) x ω r).out = .acc) {s : ℕ} (hs : s < nS x) {i : RecIns}
    {spec : SessionSpec} {sts : Array Stmt}
    (hsu : setupO (staged st x ((run H P (RecDeploy.V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) =
      some (i, spec, sts))
    (h0 : 0 < sts.size) : Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n) := by
  obtain ⟨j, f, hj, ⟨j', xO, as', cs', ins, spec₀, sts₀, link, c₁, mids, hr, hO⟩, hn⟩ :=
    run_decides H IV st x P ω r hacc hs
  rw [next_sess H hr] at hn
  have hacc' : ((nextO xO as' cs').eval H).1 = .acc := by
    rcases nextO_decide_out H hO with ho | ho
    · exact ho
    · rw [ho] at hn
      exact absurd rfl hn
  obtain ⟨⟨spec', sts', -, -, hc⟩, hf⟩ := (nextO_decide H hO).1 hacc'
  obtain ⟨i', hi', hib⟩ := actO_decide hO
  obtain ⟨hk, -, -⟩ := ract_sess_locate hr
  obtain ⟨-, -, hxO⟩ := ract_sess hr
  have hK := run_acc_length H IV st x P ω r hacc
  have hlen : (((run H P (RecDeploy.V IV st) x ω r).msgs.take (j + 1)).map Prod.fst).length = j + 1 := by
    simp only [List.length_map, List.length_take]
    omega
  rw [hlen] at hk
  have hip := innerPairs_run H IV st x P ω r hacc (off := j + 1) hk.le (by omega) []
    (range_map_take (coinSeq r) hk.le)
  rw [List.append_nil] at hip
  rw [hip] at hxO
  subst hxO
  have e := hi'.symm.trans hsu
  simp only [Option.some.injEq, Prod.mk.injEq] at e
  obtain ⟨rfl, rfl, rfl⟩ := e
  obtain ⟨hins, -⟩ := setupO_some hi'
  obtain ⟨_, -, hinner, -⟩ := insO_staged hins
  rw [← hib] at hc
  exact labels_full hi' (by simp [hinner]) h0 hc hf

end StagedRun

/-- **An accepted run's staged parts label their instances `0..n-1`, at its inner and staging rounds**: `labels_run`
at `stagedAll` (`take_roundsOf`), the parts `stagedAll_run` stages. -/
theorem labels_stagedAll (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (x : Msg) {Ω : Type}
    (P : Prover Ω) (ω : Ω) (r : Fin ((RecDeploy.V IV b.st).k x) → Coin)
    (hacc : (run H P (RecDeploy.V IV b.st) x ω r).out = .acc) :
    ∀ s < nS x, ∀ (i : RecIns) (spec : SessionSpec) (sts : Array Stmt),
      setupO ((stagedAll b.st x (runAs (IV.k (xIOf x) + kS x) (run H P (RecDeploy.V IV b.st) x ω r))
        (runCs (IV.k (xIOf x) + kS x) r)).getD s []) = some (i, spec, sts) →
      ∀ h0 : 0 < sts.size, Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n) := by
  intro s hs
  rw [stagedAll, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hs, Option.map_some,
    Option.getD_some, ← take_roundsOf H IV b x P ω r hacc]
  intro i spec sts hsu h0
  exact labels_run H IV b.st x P ω r hacc hs hsu h0

end RecDeploy
