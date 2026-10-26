import Proofs.Flock.Claims.Recursion.CopySiteDeployed
import Proofs.Flock.Claims.Recursion.SlotsCheck
import Proofs.Flock.Claims.Recursion.DeployMargin
import Proofs.Flock.Claims.Recursion.CoinPos
import Proofs.Flock.C3.PubRows.Check
import Proofs.Flock.Soundness.Discharge.Layout.QSort
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.Hidden.Public
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Properties.Nci.ReadsExact

set_option Elab.async false

/-!
# A staged table's labels ascend

`CopySite.LabelsAscend` at every V\* table staged as V\* stages it (`labels_staged`), from the recursive setup's own
checks (`RecIns.setup`), with no premise:

* A V\* session is one table (`RecSetup.oneFull`, `oneFull_staged`), so `setupTables` sets table 0 up at the session's
  draw (`setupTables_okAny`), and the label check reads that draw (`RecVStar.tableDraw` at one table).
* Without a draw the table's public file is the registered population's (`setupOne_pub`), whose labels `loadPublic`
  refuses unless they ascend, one per instance (`loadPublic_asc`).
* With a draw the table is the drawn units' (`HmRow.drawn`, `Layout.drawn_spec`: `n` is the proved units' count), and
  the setup's label check (`RecVStar.checkLabels`, which runs under the verifier's inner shape, `insO_staged`) makes its
  labels the proved units (`checkLabel_idx`). These ascend (`proved_asc`): `UnitDraw.ofJson` refuses drawn units, or a
  closure, that are not ascending and distinct, or a closure holding a drawn unit (`ofJson_asc`), and with a closure
  `proved` is their `qsort`, which sorts (`ReadsExact.qsort_sorted`) and permutes (`CoinPos.qsort_perm`) them.
-/

namespace RecDeploy.CopySite

open Flock FlockSoundness.Refine
open Claims (Msg)

/-! ## The proved units ascend -/

theorem asc_of_all {a : Array ℕ} {N : ℕ}
    (h : (List.range a.size).all (fun i => a[i]! < N && (i == 0 || a[i - 1]! < a[i]!)) = true) :
    ∀ j, 0 < j → j < a.size → a[j - 1]! < a[j]! := by
  intro j hj0 hj
  have := List.all_eq_true.1 h j (List.mem_range.2 hj)
  simp only [Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at this
  rcases this.2 with h' | h'
  · omega
  · exact h'

theorem not_mem_of_any {cs us : Array ℕ} (h : ¬ (cs.any us.contains) = true) : ∀ u ∈ cs, u ∉ us := by
  intro u hu hm
  exact h (Array.any_eq_true'.2 ⟨u, hu, Array.contains_iff_mem.2 hm⟩)

set_option maxHeartbeats 2000000 in
/-- **A draw's units ascend, and so does its closure, which holds no drawn unit**: `UnitDraw.ofJson` refuses any other. -/
theorem ofJson_asc {d : Lean.Json} {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson d = .ok D) :
    (∀ j, 0 < j → j < D.units.size → D.units[j - 1]! < D.units[j]!) ∧
      (∀ j, 0 < j → j < D.closure.size → D.closure[j - 1]! < D.closure[j]!) ∧ ∀ u ∈ D.closure, u ∉ D.units := by
  unfold Draw.UnitDraw.ofJson at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨asc_of_all (FlockSoundness.Discharge.Layout.of_not_not_true ‹¬(!(List.range _).all _) = true›), ?_⟩
    have hcl := ‹(pure _ : Flock.V (Array ℕ)) = Except.ok _›
    simp only [pure, Except.pure, Except.ok.injEq] at hcl
    subst hcl
    first
    | exact ⟨asc_of_all (FlockSoundness.Discharge.Layout.of_not_or_not_true
        ‹¬((_ : Array ℕ).isEmpty || !(List.range _).all _) = true›), not_mem_of_any ‹¬(_ : Array ℕ).any _ = true›⟩
    | simp

theorem pairwise_of_asc {a : Array ℕ} (h : ∀ j, 0 < j → j < a.size → a[j - 1]! < a[j]!) :
    a.toList.Pairwise (· < ·) := by
  have hk : ∀ k i, i + k + 1 < a.size → a[i]! < a[i + k + 1]! := by
    intro k
    induction k with
    | zero =>
      intro i hi
      have := h (i + 1) (by omega) (by omega)
      simpa using this
    | succ k ih =>
      intro i hi
      have h1 := ih i (by omega)
      have h2 := h (i + k + 2) (by omega) (by omega)
      rw [show i + k + 2 - 1 = i + k + 1 by omega] at h2
      rw [show i + (k + 1) + 1 = i + k + 2 by omega]
      omega
  rw [List.pairwise_iff_getElem]
  intro i j hi hj hij
  simp only [Array.length_toList] at hi hj
  simp only [Array.getElem_toList]
  have := hk (j - i - 1) i (by omega)
  rw [show i + (j - i - 1) + 1 = j by omega, getElem!_pos a i hi, getElem!_pos a j hj] at this
  exact this

theorem asc_of_pairwise {a : Array ℕ} (h : a.toList.Pairwise (· < ·)) :
    ∀ j, 0 < j → j < a.size → a[j - 1]! < a[j]! := by
  intro j hj0 hj
  have := List.pairwise_iff_getElem.1 h (j - 1) j (by simp; omega) (by simpa using hj) (by omega)
  simp only [Array.getElem_toList] at this
  rw [getElem!_pos a (j - 1) (by omega), getElem!_pos a j hj]
  exact this

/-- **A draw's proved units ascend**: its drawn units without a closure (`ofJson_asc`), else their `qsort` with the
closure, sorted (`qsort_sorted`) and distinct, since it permutes distinct units (`qsort_perm`). -/
theorem proved_asc {d : Lean.Json} {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson d = .ok D) :
    ∀ j, 0 < j → j < D.proved.size → D.proved[j - 1]! < D.proved[j]! := by
  obtain ⟨hu, hc, hdis⟩ := ofJson_asc h
  unfold Draw.UnitDraw.proved
  split
  · exact hu
  · refine asc_of_pairwise ?_
    have hs : ((D.units ++ D.closure).qsort (· < ·)).toList.Pairwise (· ≤ ·) :=
      NciCheck.Properties.SecurityProofs.ReadsExact.qsort_sorted id (D.units ++ D.closure)
    have hnd : (D.units ++ D.closure).toList.Nodup := by
      rw [Array.toList_append]
      show List.Pairwise (fun a b => a ≠ b) (D.units.toList ++ D.closure.toList)
      refine List.pairwise_append.2 ⟨(pairwise_of_asc hu).imp Nat.ne_of_lt, (pairwise_of_asc hc).imp Nat.ne_of_lt,
        fun a ha b hb hab => ?_⟩
      exact hdis b (by simpa using hb) (by simpa [hab] using ha)
    have hnd' : ((D.units ++ D.closure).qsort (· < ·)).toList.Nodup :=
      (Array.perm_iff_toList_perm.1 (RecDeploy.CoinPos.qsort_perm (D.units ++ D.closure) (· < ·))).nodup_iff.2 hnd
    exact (hs.and hnd').imp fun ⟨h1, h2⟩ => Nat.lt_of_le_of_ne h1 h2

/-! ## The registered population's labels ascend -/

theorem asc_of_any {a : Array ℕ} {n : ℕ} {b : Bool}
    (h : ¬ (a.size != n || (List.range a.size).any (fun i => i > 0 && a[i - 1]! ≥ a[i]!) || b) = true) :
    a.size = n ∧ ∀ j, 0 < j → j < n → a[j - 1]! < a[j]! := by
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not, List.any_eq_true, List.mem_range,
    Bool.and_eq_true, decide_eq_true_eq, not_exists, not_and] at h
  obtain ⟨⟨hs, ha⟩, -⟩ := h
  refine ⟨hs, fun j hj0 hj => ?_⟩
  have := ha j (by omega)
  by_contra hge
  exact this hj0 (Nat.le_of_not_lt hge)

set_option maxHeartbeats 4000000 in
/-- **A registered population's labels ascend, one per instance**: `loadPublic` refuses any other. -/
theorem loadPublic_asc {tags : Tags} {c : Circuit} {file : ByteArray} {pop : Public}
    (h : HmRow.loadPublic tags c file = .ok pop) : LabelsAscend pop := by
  unfold HmRow.loadPublic at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    obtain ⟨hs, ha⟩ := asc_of_any (by assumption)
    refine ⟨_, ?_, hs, ha⟩
    unfold Registered.unitIndices
    simp only [‹J.get _ "units" _ = _›, ‹J.arr _ "indices" _ = _›, bind, Except.bind]
    assumption

/-! ## The staged table's public file -/

set_option maxHeartbeats 4000000 in
theorem setupH_none {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupH tags cf pf coins tables none partition program = .ok st) :
    HmRow.loadPublic tags st.c pf = .ok st.pub := by
  unfold Stmt.setupH at h
  dsimp only at h
  simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    assumption

/-- **A table's public file** (`setupOne`, either output mode): the registered population's without a draw, the drawn
units' (`HmRow.drawn`) at one. -/
theorem setupOne_pub {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : FlockSoundness.Discharge.HiddenExec.setupOne tags cf pf coins tables draw partition program = .ok st) :
    ∃ c pop, HmRow.loadPublic tags c pf = .ok pop ∧ (draw = none → st.pub = pop) ∧
      ∀ d, draw = some d → HmRow.drawn c pop d = .ok st.pub := by
  rcases Bool.eq_false_or_eq_true tags.hiddenOutputs with hh | hh
  · rw [FlockSoundness.Discharge.HiddenExec.setupOne_hidden hh] at h
    obtain ⟨pop, hl, hn, hd⟩ := (FlockSoundness.Discharge.Hidden.setupHidden_spec' h).pub
    exact ⟨_, pop, FlockSoundness.Discharge.Hidden.loadPublic_nil hl, hn, fun d e => hd d e⟩
  · rw [FlockSoundness.Discharge.HiddenExec.setupOne_public hh] at h
    cases draw with
    | none => exact ⟨st.c, st.pub, setupH_none h, fun _ => rfl, fun d (e : (none : Option Lean.Json) = some d) => nomatch e⟩
    | some d =>
      obtain ⟨pop, hl, hd⟩ := FlockSoundness.Discharge.Layout.setupH_drawn h
      exact ⟨st.c, pop, hl, (fun (e : some d = none) => nomatch e), fun d' (e : some d = some d') => by cases e; exact hd⟩

/-! ## The setup's label check -/

theorem recSetup_labels {s : RecIns} {spec : SessionSpec} {sts : Array Stmt} (h : s.setup = .ok (spec, sts))
    (hi : s.inner.isSome = true) :
    RecVStar.checkLabels s.base.read.draw s.base.read.I.count sts = .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; assumption)

theorem checkLabel_idx {dj : Lean.Json} {pub : Public} {j : ℕ} (h : RecVStar.checkLabel dj pub j = .ok ())
    {D : Draw.UnitDraw} (hD : Draw.UnitDraw.ofJson dj = .ok D) :
    ∃ idx, Registered.unitIndices pub = .ok idx ∧ idx = D.proved := by
  cases hidx : Registered.unitIndices pub with
  | error e =>
    simp [RecVStar.checkLabel, hD, hidx, Bind.bind, Except.bind] at h
  | ok idx => exact ⟨idx, rfl, FlockC3.PubRows.checkLabel_ok h hD hidx⟩

/-! ## At a staged table -/

/-- **A staged table's labels ascend, one per instance**: its one table is set up at the session's draw; without one
its labels are the registered population's (`loadPublic_asc`), with one the proved units (`checkLabel_idx`), which
ascend (`proved_asc`) and are its instances (`drawn_spec`). -/
theorem labels_staged {st : RecDeploy.Stage} {s : ℕ} {x xO : Msg} (hS : RecDeploy.StagedPart st s x xO) {i : RecIns}
    {spec : SessionSpec} {sts : Array Stmt} (hsu : RecDeploy.setupO xO = some (i, spec, sts)) (h0 : 0 < sts.size) :
    LabelsAscend sts[0].pub := by
  have h1 := (RecDeploy.oneFull_staged hS hsu).1
  obtain ⟨ms, i', spec', sts', rfl, hsu', hck⟩ := hS
  rw [hsu] at hsu'
  simp only [Option.some.injEq, Prod.mk.injEq] at hsu'
  obtain ⟨rfl, rfl, rfl⟩ := hsu'
  obtain ⟨hins, hset⟩ := RecDeploy.setupO_some hsu
  obtain ⟨r, -, hinner, -⟩ := RecDeploy.insO_staged hins
  have hl := recSetup_labels hset (by rw [hinner]; rfl)
  have hrd := RecDeploy.RecIns.setup_read hset
  unfold Flock.Readable.Read.setup at hrd
  rw [RecDeploy.publicInputs_none_of_scope hck] at hrd
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okAny hrd with ⟨hc, st0, hst0, -, rfl⟩ | ⟨hc, hsz, -⟩
  swap
  · omega
  show LabelsAscend st0.pub
  obtain ⟨c, pop, hpop, hnone, hdr⟩ := setupOne_pub hst0
  cases hd : i.base.read.draw with
  | none => rw [hnone hd]; exact loadPublic_asc hpop
  | some d =>
    rw [hd] at hl
    have hl0 := FlockC3.PubRows.checkLabels_at hl 0 (by simp)
    rw [show RecVStar.tableDraw d i.base.read.I.count 0 = d by unfold RecVStar.tableDraw; exact ite_eq_left hc] at hl0
    change RecVStar.checkLabel d st0.pub 0 = .ok () at hl0
    obtain ⟨D, hD, -, hn, -⟩ := FlockSoundness.Discharge.Layout.drawn_spec (hdr d hd)
    obtain ⟨idx, hidx, rfl⟩ := checkLabel_idx hl0 hD
    exact ⟨_, hidx, hn.symm, fun j hj0 hj => proved_asc hD j hj0 (by omega)⟩

end RecDeploy.CopySite
