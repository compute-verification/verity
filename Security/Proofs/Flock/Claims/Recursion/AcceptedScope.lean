import Proofs.Flock.Claims.Recursion.DeployedCounts
import Proofs.Flock.Claims.Recursion.DeployScopeRuns
import Proofs.Flock.Claims.Recursion.InnerForm
import Proofs.Flock.Claims.Recursion.DeployMargin
import Proofs.Flock.Claims.KS.PlainStep

/-!
# The deployed brings' scope at an accepted statement

`lamLin2_deployedB_ge_joint` takes `hin` (the inner statement's plain setup in `PlainIn`) and `hO` (each staged V\*
part's setup in `PartIn`). Both are what the compiled verifier's own setup establishes wherever it accepts, so at a
statement some run of it accepts they need no hypothesis (`lamLin2_deployedB_ge_acc`):

* **A part** (`partIn_of_run_acc`): every session of an accepted run ends on a decision the verifier doesn't refuse
  (`run_decides`), where the outer verifier accepts (`nextO_decide_out`), so its custody checks pass and its scope check
  refuses `--public-inputs` (`publicInputs_none_of_scope`). A staged part's `publicInputs` is its part's in `x` at any
  inner rounds (`staged_publicInputs`), so the same holds at `pick`'s. Where a session with no public inputs sets up, its
  tables number from 1 to 64 at two link points of at most 64 coordinates, each in range (`read_setup_in`:
  `setupTables_setupOne`, `setupOne_range`), with `m ≤ 35` (`setupO_counts`) and, masked (`readOfO_zk`), `25 ≤ m`
  (`RecIns.setup_shapeOk`, `m25_of_shapeOk`) (`partIn_of_setupO`).
* **The inner statement** (`plainIn_of_run_acc`): an accepted run parses (`acc_parse`), so `xIOf x` is the inner
  statement `readX` read, which the plain verifier sets up as one table with no public inputs (`innerSetup_files`), at
  the `m` of the verifier's inner shape (`checkShape_files`), which `RecVStar.statements` puts in `[22, 35]`
  (`statements_ok`).

What is left are `hS` and `hU`, both counts of the verifier's own hash evaluations and neither a statement about the
prover: `hS` the inner session's setup inputs (`innerSetupIns`, `setupLen`), and `hU` the bound `n` on the compiled
record's inner-verifier calls (`innerCalls`, `BitsLin2`'s `n`). The prover's queries are `BitsLin2`'s `t`, which multiplies
`2^{-λ₂}` and doesn't enter `λ₂`.
-/

set_option Elab.async false

namespace FlockDeploy

theorem Raw.ins_publicInputs {w : Raw} {s : Ins} (h : w.ins = .ok s) : s.read.publicInputs = w.publicInputs := by
  unfold Raw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; rfl)

end FlockDeploy

namespace RecDeploy

open Flock
open Claims (Msg Out Prover run)
open FlockDeploy (Coin Ins)

/-- **A session with no public inputs that sets up is in range**: one to `64` tables, two link points, and each
table's `k_log`, regions and link points within `checkInRange`'s range. -/
theorem read_setup_in {r : Flock.Readable.Read} {spec : SessionSpec} {sts : Array Stmt}
    (h : r.setup = .ok (spec, sts)) (hpi : r.publicInputs = none) :
    0 < sts.size ∧ sts.size ≤ 64 ∧ spec.points = 2 ∧ ∀ st ∈ sts.toList, (FlockDeploy.acct spec st).InRange := by
  have hle := (FlockDeploy.setupTables_le h).1
  have hmax : Flock.Tags.MAX_TABLES = 64 := rfl
  unfold Flock.Readable.Read.setup at h
  rw [hpi] at h
  obtain ⟨h0, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne h
  obtain ⟨hp2, hpts⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_pointsAny h
  refine ⟨h0, by omega, hp2, fun st hst => ?_⟩
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hst)
  obtain ⟨d, hd, -⟩ := hall j hj
  obtain ⟨hr, hwf⟩ := FlockDeploy.KS.setupOne_range hd
  obtain ⟨hk, hreg, h64⟩ := (Flock.checkInRange_ok _ _ _).1 hr
  have hm := hwf.mPts
  have hpj := hpts j hj
  show sts[j].c.kLog ≤ 27 ∧ sts[j].regions.size ≤ 1024 ∧ spec.mPts ≤ 64
  exact ⟨hk, hreg, by omega⟩

/-- **A V\* part that sets up with no public inputs is in scope** (`PartIn`). -/
theorem partIn_of_setupO {xO : Msg} {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupO xO = some (i, spec, sts)) (hpi : i.base.read.publicInputs = none) : PartIn xO i spec sts := by
  obtain ⟨hins, hsu⟩ := setupO_some hs
  obtain ⟨h0, h64, hp2, hrange⟩ := read_setup_in (RecIns.setup_read hsu) hpi
  have hz : i.base.read.zk = true := readOfO_zk (r := i.base.read) (by rw [readOfO, hins, Option.map_some])
  have hsh := RecIns.setup_shapeOk hsu hz
  have hc := (setupO_counts hs).2
  refine ⟨hs, h0, by omega, by omega, fun st hst => ⟨?_, (hc st hst).m35, hrange st hst⟩⟩
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hst)
  exact Flock.Readable.m25_of_shapeOk (Flock.Readable.array_forM_ok hsh j hj)

/-- **A staged part's public inputs are its part's in `x`**, at any inner rounds. -/
theorem staged_publicInputs {st : Stage} {x : Msg} {ms : List (Msg × Out)} {s : ℕ} {i : RecIns}
    (h : insO (staged st x ms s) = some i) :
    ∃ r p, readX x = some r ∧ r.w.parts[s]? = some p ∧ i.base.read.publicInputs = p.base.publicInputs := by
  have hne : staged st x ms s ≠ [] := fun he => by rw [he, insO_nil] at h; cases h
  obtain ⟨r, p, hr, hp, hd⟩ := staged_decode hne
  unfold insO at h
  rw [hd] at h
  simp only [Option.bind_eq_bind, Option.bind_some] at h
  split_ifs at h
  · simp at h
  have h' : (partRaw r st x ms s p).ins.toOption = some i := by simpa using h
  have hpi := FlockDeploy.Raw.ins_publicInputs (RecRaw.ins_base (toOption_eq_some h'))
  exact ⟨r, p, hr, hp, hpi⟩

section Runs

variable {H : Msg → Msg} {IV : Claims.Verifier Coin} {st : Stage} {x : Msg} {Ω : Type} {P : Prover Ω} {ω : Ω}
  {r : Fin ((V IV st).k x) → Coin}

/-- **`hO` at an accepted statement**: on a run the compiled verifier accepts, every staged part at `pick`'s inner
rounds is in scope. -/
theorem partIn_of_run_acc (hacc : (run H P (V IV st) x ω r).out = .acc) :
    ∀ j < nS x, ∃ i spec' sts', PartIn (staged st x (pick st x j) j) i spec' sts' := by
  intro s hs
  obtain ⟨k, f, -, ⟨j, xO, as', cs', ins, spec, sts, link, c₁, mids, hr, ha⟩, hn⟩ :=
    run_decides H IV st x P ω r hacc hs
  rw [next_sess H hr] at hn
  have hacO : ((nextO xO as' cs').eval H).1 = .acc := by
    rcases nextO_decide_out H ha with ho | ho
    · exact ho
    · rw [ho] at hn; exact absurd rfl hn
  obtain ⟨⟨_, _, -, -, hc⟩, -⟩ := (nextO_decide H ha).1 hacO
  obtain ⟨-, -, hxO⟩ := ract_sess hr
  obtain ⟨i₀, hi₀, rfl⟩ := actO_decide ha
  have hpi₀ := publicInputs_none_of_scope (scope_of_checks hc)
  rw [hxO] at hi₀
  obtain ⟨hins₀, -⟩ := setupO_some hi₀
  obtain ⟨r₀, p₀, hr₀, hp₀, he₀⟩ := staged_publicInputs hins₀
  have hex : ∃ ms, (setupO (staged st x ms s)).isSome := ⟨_, Option.isSome_iff_exists.2 ⟨_, hi₀⟩⟩
  have hpk : (setupO (staged st x (pick st x s) s)).isSome := by
    unfold pick
    simp only [hex, ↓reduceDIte]
    exact Classical.choose_spec hex
  obtain ⟨⟨i, spec', sts'⟩, hsp⟩ := Option.isSome_iff_exists.1 hpk
  obtain ⟨hins, -⟩ := setupO_some hsp
  obtain ⟨r₁, p₁, hr₁, hp₁, he₁⟩ := staged_publicInputs hins
  rw [hr₀] at hr₁
  cases hr₁
  rw [hp₀] at hp₁
  cases hp₁
  exact ⟨i, spec', sts', partIn_of_setupO hsp (he₁.trans (he₀.symm.trans hpi₀))⟩

/-- **`hin` at an accepted statement**: on a run the compiled verifier accepts, the inner statement's plain setup is in
`PlainIn`, one table at the `m` of the verifier's inner shape. -/
theorem plainIn_of_run_acc (hacc : (run H P (V IV st) x ω r).out = .acc) :
    ∃ s spec sts, FlockDeploy.PlainIn (xIOf x) s spec sts := by
  obtain ⟨xr, hx, hacc', -⟩ := run_acc_scope H IV st x P ω r hacc
  obtain ⟨as', ha⟩ := stepWith_acc H P x ω _ _ _ _ _ hacc
  obtain ⟨xr', hx', hp, -⟩ := acc_parse (IV := IV) (st := st) H ha
  have hxx : xr' = xr := Option.some.inj (hx'.symm.trans hx)
  rw [hxx] at hp
  have hxI : xIOf x = xr.w.inner := by
    unfold xIOf
    rw [hp]
    rfl
  obtain ⟨f, spec', st', s, spec, st₀, hf, hfs, hset, hm, -, -, -, -, hpi⟩ := innerSetup_files hx hacc'
  have ha2 := hacc'
  unfold RecScope.accepts at ha2
  simp only [Bool.and_eq_true, RecScope.isOk_iff] at ha2
  obtain ⟨⟨⟨⟨ss, hss⟩, -⟩, ⟨u, hsh⟩⟩, -⟩ := ha2
  have hss : RecVStar.statements xr.shape = .ok ss := hss
  have hsh : RecSetup.checkShape (some xr.shape) (filesOf xr) = .ok u := hsh
  rw [hf] at hsh
  obtain ⟨spec'', st'', hfs', hm', -⟩ := checkShape_files hsh
  rw [hfs] at hfs'
  cases hfs'
  obtain ⟨h22, h35, -⟩ := Flock.RecScope.statements_ok hss
  obtain ⟨-, hsu⟩ := FlockDeploy.setupOf_some hset
  obtain ⟨h0, h64, hp2, hrange⟩ := read_setup_in (FlockDeploy.Ins.setup_read hsu) hpi
  refine ⟨s, spec, #[st₀], by rw [hxI]; exact hset, h0, by omega, by omega, fun t ht => ?_⟩
  have ht' : t = st₀ := by simpa using ht
  rw [ht']
  have hri : st₀.c.kLog ≤ 27 ∧ st₀.regions.size ≤ 1024 ∧ spec.mPts ≤ 64 := hrange st₀ (by simp)
  refine ⟨by omega, by omega, ?_⟩
  show st₀.c.kLog ≤ 27 ∧ st₀.regions.size ≤ 1026 ∧ spec.mPts ≤ 64
  exact ⟨hri.1, by have := hri.2.1; omega, hri.2.2⟩

end Runs

/-- **Route 2C's `λ₂ ≥ 100` at the deployed brings, at an accepted statement**: `lamLin2_deployedB_ge_joint` with `hin`
and `hO` from the run the compiled verifier accepts (`plainIn_of_run_acc`, `partIn_of_run_acc`), and the compiled
verifier's own evaluations at its V\* statements at most `2^55`. What is left, `hS` (the inner session's setup inputs) and
`hU` (the record's inner-verifier calls), counts the verifier's own hash evaluations; the prover's queries don't enter
`λ₂`. -/
theorem lamLin2_deployedB_ge_acc (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (H : Msg → Msg)
    {x : Msg} {IV : Claims.Verifier Coin} {H' : Msg → Msg} {Ω : Type} {P : Prover Ω} {ω : Ω}
    {r : Fin ((V IV (deployedB b).st).k x) → Coin} (hacc : (run H' P (V IV (deployedB b).st) x ω r).out = .acc)
    (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 56) (ps : List (Msg × Out)) {n : ℕ} (hU : n ≤ 2 ^ 55) :
    100 ≤ Claims.Recursion.lamLin2
      ((recBoundAt rel (deployedB b) L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
      FlockDeploy.KS.CpDep x n (2 ^ 55) ∧
    (Claims.Recursion.setupCalls H
      (recBound rel (deployedB b) L (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      ((recBound rel (deployedB b) L (bdDeployed rel nextE colsE wE join rowIn final w).IVStar).vstar x
        ps)).length ≤ 2 ^ 55 := by
  obtain ⟨s, spec, sts, hin⟩ := plainIn_of_run_acc hacc
  exact lamLin2_deployedB_ge_joint rel b L nextE colsE wE join rowIn final w decI H hin (partIn_of_run_acc hacc) hS ps
    hU

end RecDeploy
