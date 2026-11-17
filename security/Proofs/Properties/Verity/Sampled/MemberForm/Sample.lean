module

public import Proofs.Properties.Verity.Sampled.MemberForm.Passes
public import Proofs.Properties.Verity.Sampled.MemberForm.Ranges
public import Proofs.Properties.Verity.Sampled.Held
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic

/-! The sample a session's record holds, at a member of the per-member form. -/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open FlockSoundness.Refine (ebind_ok)

theorem first_one {vm : _root_.Spot.OneStageZk.Member} {lo hi : ℕ} (h : vm.units = #[(lo, hi)]) :
    vm.first = lo := by
  unfold _root_.Spot.OneStageZk.Member.first
  rw [h]
  exact first_range lo hi

theorem count_one {vm : _root_.Spot.OneStageZk.Member} {lo hi : ℕ} (h : vm.units = #[(lo, hi)]) :
    vm.count = hi - lo := by
  unfold _root_.Spot.OneStageZk.Member.count
  rw [h]
  exact count_range lo hi

theorem indices_one {vm : _root_.Spot.OneStageZk.Member} {lo hi : ℕ} (h : vm.units = #[(lo, hi)]) :
    vm.indices = (Array.range (hi - lo)).map (lo + ·) := by
  unfold _root_.Spot.OneStageZk.Member.indices
  rw [h]
  exact indices_range lo hi

theorem indices_at {vm : _root_.Spot.OneStageZk.Member} {lo hi : ℕ} (h : vm.units = #[(lo, hi)]) {u v : ℕ}
    (hu : vm.indices[u]? = some v) : u < hi - lo ∧ v = lo + u := by
  rw [indices_one h] at hu
  exact indices_range_at hu

theorem sampled_one {vm : _root_.Spot.OneStageZk.Member} {lo hi : ℕ} (h : vm.units = #[(lo, hi)])
    (want : Array ℕ) : vm.sampled want = true ↔ ∃ u ∈ want, lo ≤ u ∧ u < hi := by
  unfold _root_.Spot.OneStageZk.Member.sampled
  rw [h]
  exact sampled_range lo hi want

theorem laidHeld_one {ms : Array _root_.Spot.OneStageZk.Member} {rec : Json}
    (h : _root_.Spot.OneStageZk.laidHeld ms rec = .ok ()) : ∀ vm ∈ ms, ∃ lo hi, vm.units = #[(lo, hi)] := by
  unfold _root_.Spot.OneStageZk.laidHeld at h
  split at h
  · cases h
  · split at h
    · rename_i hc
      rw [Bool.and_eq_true] at hc
      intro vm hvm
      have hone := Array.all_eq_true'.1 hc.1 vm hvm
      rw [Bool.and_eq_true, beq_iff_eq] at hone
      exact one_range hone.1
    · cases h

theorem laidHeld_sum {ms : Array _root_.Spot.OneStageZk.Member} {rec : Json}
    (h : _root_.Spot.OneStageZk.laidHeld ms rec = .ok ()) :
    ∃ n, _root_.Spot.nat? (_root_.Spot.getD rec "population") = some n ∧
      (ms.toList.map fun vm => vm.count).sum = n := by
  unfold _root_.Spot.OneStageZk.laidHeld at h
  split at h
  · cases h
  · rename_i n hn
    split at h
    · rename_i hc
      rw [Bool.and_eq_true, beq_iff_eq] at hc
      exact ⟨n, hn, (foldl_add_eq_sum (fun vm : _root_.Spot.OneStageZk.Member => vm.count) ms).symm.trans hc.2⟩
    · cases h

theorem membersStructure_laid {p : _root_.Spot.OneStageZk.Params} {ms : Array _root_.Spot.OneStageZk.Member}
    {c : _root_.Spot.OneStage.Commitment} {e : _root_.Spot.OneStageZk.Evidence}
    {sts : Array _root_.Spot.OneStageZk.StatementFiles} {bs : _root_.Core.Blobs}
    (h : _root_.Spot.OneStageZk.membersStructure p ms c e sts bs = .ok ()) :
    _root_.Spot.OneStageZk.laidHeld ms c.record = .ok () := by
  unfold _root_.Spot.OneStageZk.membersStructure at h
  obtain ⟨held, -, h⟩ := ebind_ok h
  dsimp only at h
  split at h
  · obtain ⟨_, ht, -⟩ := ebind_ok h
    cases ht
  obtain ⟨ts, -, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, -, h⟩ := ebind_ok h
  obtain ⟨lines, -, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, -, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, -, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, hlaid, -⟩ := ebind_ok h
  exact hlaid

theorem ExecutionPhaseOn.total {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α] {V : Vocab d α}
    {T : PinnedTables} {θ : ByteArray} {F : Array ByteArray} {W : ExecutionPhase} (h : ExecutionPhaseOn V T θ F W)
    (hp : ∀ m ∈ W.ms, FilesPass m) : (W.ms.map fun m => m.n).sum = W.nJ := by
  obtain ⟨m₀, rest, hW, hread⟩ := h.read
  have hu : V.units θ F = ((W.ms.map fileRead).map Verity.FileRead.units).sum := by
    rw [hW, List.map_cons]
    show (V.read θ F).units = _
    rw [hread]
    exact Verity.FileRead.unitsWith_eq_sum (fileRead m₀) (rest.map fileRead)
  refine Eq.trans ?_ h.units
  rw [hu, List.map_map]
  refine congrArg List.sum (List.map_congr_left fun m hm => ?_)
  exact ((m.toV5 0).fileRead_units ((hp m hm).checked_all 0)).symm

theorem ExecutionPhaseOn.population {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α] {V : Vocab d α}
    {T : PinnedTables} {θ : ByteArray} {F : Array ByteArray} {W : ExecutionPhase} (h : ExecutionPhaseOn V T θ F W)
    (hp : ∀ m ∈ W.ms, FilesPass m) {p : _root_.Spot.OneStageZk.Params}
    {ms : Array _root_.Spot.OneStageZk.Member} (hpa : ParamsAt θ p) (hms : p.members = some ms) {rec : Json}
    (hl : _root_.Spot.OneStageZk.laidHeld ms rec = .ok ()) {n : ℕ}
    (hn : _root_.Spot.nat? (_root_.Spot.getD rec "population") = some n) : W.nJ = n := by
  obtain ⟨p', ms', vms, hp', hms', hperm, -, hall, -⟩ := h.members
  obtain rfl : p = p' := hpa.unique hp'
  obtain rfl : ms = ms' := Option.some.inj (hms.symm.trans hms')
  obtain ⟨n', hn', hsum⟩ := laidHeld_sum hl
  obtain rfl : n = n' := Option.some.inj (hn.symm.trans hn')
  have h1 : vms.map (fun vm : _root_.Spot.OneStageZk.Member => vm.count) =
      W.ms.map (fun m : Member W.nJ => m.n) :=
    forall₂_map_eq hall fun vm m hon => hon.units.symm
  have h2 := sum_perm (fun vm : _root_.Spot.OneStageZk.Member => vm.count) hperm
  have h3 : (W.ms.map fun m : Member W.nJ => m.n).sum = W.nJ := h.total hp
  exact h3.symm.trans ((congrArg List.sum h1).symm.trans (h2.trans hsum))

theorem sessionVerdict_subset {a : StepAt} {bs : _root_.Core.Blobs} {keys : Array String}
    {s : _root_.Spot.OneStageZk.RunFiles} {record : String} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup}
    {rest : Json × Json} {proofs : Array ByteArray} {units : Array ℕ}
    (h : SessionVerdict a bs keys s record spec tabs rest proofs units) :
    ∃ dj D kd, Flock.Witness.recordDraw record = some dj ∧ Flock.Draw.UnitSample.ofJson dj = .ok D ∧
      D.samplingDesign = .subset kd ∧
      ∃ sts, Flock.Stmt.setupTables (Flock.Witness.zkTags a.I) a.I.circuitFile a.I.publicFile
        (Flock.Witness.zkCoins a.I) a.I.count a.I.tables (some dj) a.I.partition a.I.program = .ok (spec, sts) := by
  obtain ⟨dj, hdj⟩ := Option.isSome_iff_exists.1 (verdictOf_sampled h.verdict)
  have hst := h.setup
  rw [hdj] at hst
  obtain ⟨D, hD, hw⟩ := sampleDesign_ok (setupAt_sampleDesign hst)
  obtain ⟨sts, hss, -⟩ := setupAt_ok hst
  obtain ⟨hset, -⟩ := sessionAt_ok hss
  rcases samplingDesignWhy_zk hw with ⟨kd, hk⟩ | ⟨K, St, hk⟩
  · exact ⟨dj, D, kd, hdj, hD, hk, sts, hset⟩
  · exact (setupTables_stratified hset hD hk).elim

theorem row_count {vm : _root_.Spot.OneStageZk.Member} {rec : Json} {roots : Std.HashMap String (Array Json)}
    {pubB : ByteArray} (hpop : vm.population (vm.slice rec roots) pubB = .ok ()) {tags : Flock.Tags}
    {c : Flock.Circuit} {pop : Flock.Public} (hl : Flock.HmRow.loadPublic tags c pubB = .ok pop) :
    pop.n = vm.count := by
  obtain ⟨hd', hh, hheld⟩ := population_header hpop
  rw [← Flock.With.loadPublicWith_std] at hl
  obtain ⟨nl, hd, counts, -, hhd, hi, -⟩ := Flock.Bridge.loadPublic_core _ _ _ _ _ hl
  have e : hd = hd' := Except.ok.inj ((header_of_headerOf hhd).symm.trans hh)
  subst e
  have hj := headerHeld_instances hheld
  have hg : _root_.Spot.getD hd "instances" = .num ⟨Int.ofNat pop.n, 0⟩ := by
    simp only [_root_.Spot.getD, _root_.Spot.field?, FlockComplete.JsonRT.nat_ok hi]
    rfl
  rw [hg, slice_population] at hj
  exact Properties.Spot.Credited.eq_of_jeq hj

theorem population_units {vm : _root_.Spot.OneStageZk.Member} {slice : Json} {pubB : ByteArray}
    (h : vm.population slice pubB = .ok ()) :
    ∃ hd, _root_.Spot.FilePart.header pubB = .ok hd ∧
      _root_.Spot.OneStageZk.unitsHeld vm.indices hd = true := by
  unfold _root_.Spot.OneStageZk.Member.population at h
  obtain ⟨prog, -, h⟩ := ebind_ok h
  obtain ⟨strata, -, h⟩ := ebind_ok h
  obtain ⟨hd, hh, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, hat, -⟩ := ebind_ok h
  refine ⟨hd, hh, ?_⟩
  unfold _root_.Spot.OneStageZk.Member.heldAt at hat
  split at hat
  · cases hat
  · split at hat
    · assumption
    · cases hat

theorem indicesOf_of_unitsHeld {us : Array ℕ} {h : Json} (hi : _root_.Spot.OneStageZk.unitsHeld us h = true) :
    _root_.Spot.FilePart.indicesOf h = .ok us := by
  unfold _root_.Spot.OneStageZk.unitsHeld at hi
  split at hi
  · rename_i is his
    rw [Bool.and_eq_true, beq_iff_eq, List.all_eq_true] at hi
    obtain ⟨hsz, hall⟩ := hi
    unfold _root_.Spot.FilePart.indicesOf
    simp only [his]
    rw [Array.mapM_eq_mapM_toList, mapM_ok_at _ is.toList us.toList
      (by rw [Array.length_toList, Array.length_toList]; exact hsz) fun i h₁ h₂ => ?_]
    · rfl
    · have h₁' : i < is.size := by rw [Array.length_toList] at h₁; exact h₁
      have h₂' : i < us.size := by rw [Array.length_toList] at h₂; exact h₂
      have hat := hall i (List.mem_range.2 h₂')
      rw [beq_iff_eq, Array.getElem?_eq_getElem h₁', Array.getElem?_eq_getElem h₂'] at hat
      have hn : _root_.Spot.nat? is[i] = some us[i] := hat
      rw [Array.getElem_toList, Array.getElem_toList]
      simp only [hn]
      rfl
  · cases hi

theorem provedUnits_at {pubB : ByteArray} {record : String} {hd j : Json} {inds us units : Array ℕ}
    (hh : _root_.Spot.FilePart.header pubB = .ok hd)
    (hi : _root_.Spot.FilePart.indicesOf hd = .ok inds)
    (hj : Flock.JsonParse.parse record = .ok j)
    (hu : _root_.Spot.getD (_root_.Spot.getD j "unit_draw") "units" =
      .arr (us.map fun x => Lean.toJson x))
    (h : _root_.Spot.OneStageZk.provedUnits pubB record = .ok units) :
    ∃ vs : Array ℕ, units = vs.qsort (· < ·) ∧ vs.size = us.size ∧
      ∀ i (h₁ : i < vs.size) (h₂ : i < us.size), inds[us[i]]? = some vs[i] := by
  unfold _root_.Spot.OneStageZk.provedUnits at h
  obtain ⟨_, e1, h⟩ := ebind_ok h
  rw [hh] at e1
  cases e1
  obtain ⟨_, e2, h⟩ := ebind_ok h
  rw [hi] at e2
  cases e2
  obtain ⟨_, e3, h⟩ := ebind_ok h
  rw [hj] at e3
  cases e3
  simp only [hu] at h
  obtain ⟨vs, hm, h⟩ := ebind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  obtain ⟨hs, hb⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem hm
  have hsz : vs.size = us.size := by simpa using hs
  refine ⟨vs, rfl, hsz, fun i h₁ h₂ => ?_⟩
  have hat := hb i (by simpa using h₂) h₁
  simp only [Array.getElem_map] at hat
  have hn : _root_.Spot.nat? (Lean.toJson us[i]) = some us[i] := rfl
  simp only [hn, bind, Option.bind] at hat
  cases hr : inds[us[i]]? with
  | none => rw [hr] at hat; cases hat
  | some u =>
    rw [hr] at hat
    simp only [pure, Except.pure, Except.ok.injEq] at hat
    rw [hat]

theorem session_sample {vm : _root_.Spot.OneStageZk.Member} {rec : Json}
    {roots : Std.HashMap String (Array Json)} {bs : _root_.Core.Blobs} {a : _root_.Spot.OneStageZk.StepAt}
    {lo hi : ℕ} (hu : vm.units = #[(lo, hi)]) (hs : vm.statement (vm.slice rec roots) bs = .ok a)
    {keys : Array String} {s : _root_.Spot.OneStageZk.RunFiles} {record : String} {spec : Flock.SessionSpec}
    {tabs : Array Flock.Setup} {rest : Json × Json} {proofs : Array ByteArray} {units : Array ℕ}
    (hv : SessionVerdict (stepAt a) bs keys s record spec tabs rest proofs units) :
    ∃ us : Array ℕ, unitsAscending vm.count us = true ∧
      Flock.Witness.recordDraw record =
        some (Flock.Draw.UnitSample.toJson ⟨.subset us.size, vm.count, us, #[]⟩) ∧
      units = us.map (lo + ·) := by
  obtain ⟨tags0, circuitB, pubB, names, -, -, -, -, -, hI⟩ := statement_reads hs
  obtain ⟨-, epop⟩ := statement_checks hs
  obtain ⟨dj, D, kd, hdj, hD, hl, sts, hset⟩ := sessionVerdict_subset hv
  obtain ⟨hasc, hprint⟩ := recordSampled_of_subset hdj hD hl
  have hpc : a.population = .ok () := verdictOf_population hv.verdict
  have hpop : vm.population (vm.slice rec roots) a.I.publicFile = .ok () := epop.symm.trans hpc
  have hc1 : a.I.count = 1 := by
    rw [hI]
    rfl
  have hset' : Flock.Stmt.setupTables (Flock.Witness.zkTags a.I) a.I.circuitFile a.I.publicFile
      (Flock.Witness.zkCoins a.I) a.I.count a.I.tables (some dj) a.I.partition a.I.program = .ok (spec, sts) :=
    hset
  rw [hc1] at hset'
  obtain ⟨c, pop, hlp, hDp⟩ := setupTables_population hset' hD
  have hn : D.population = vm.count := hDp.trans (row_count hpop hlp)
  obtain ⟨hd, hh, hheld⟩ := population_units hpop
  have hinds := indicesOf_of_unitsHeld hheld
  obtain ⟨j, d, hj, hjd, hwd, hdn⟩ := recordSample_raw hdj
  have hdjp : dj = Flock.Draw.UnitSample.toJson ⟨.subset D.units.size, D.population, D.units, #[]⟩ :=
    Option.some.inj (hdj.symm.trans hprint)
  have hunits : _root_.Spot.getD (_root_.Spot.getD j "unit_draw") "units" =
      .arr (D.units.map fun x => Lean.toJson x) := by
    rw [hjd]
    exact units_of_norm hwd (hdn.symm.trans hdjp)
  have hpu : _root_.Spot.OneStageZk.provedUnits a.I.publicFile record = .ok units := hv.units
  obtain ⟨vs, hvs, hsz, hat⟩ := provedUnits_at hh hinds hj hunits hpu
  have hmap : vs = D.units.map (lo + ·) := by
    refine Array.ext (by rw [Array.size_map]; exact hsz) fun i h₁ h₂ => ?_
    rw [Array.getElem_map]
    exact (indices_at hu (hat i h₁ (by rw [Array.size_map] at h₂; exact h₂))).2
  have hsorted : (D.units.map (lo + ·)).toList.Pairwise (· < ·) := by
    rw [Array.toList_map, List.pairwise_map]
    exact (unitsAscending_spec hasc).2.imp fun hlt => Nat.add_lt_add_left hlt lo
  refine ⟨D.units, ?_, ?_, ?_⟩
  · rw [← hn]
    exact hasc
  · rw [← hn]
    exact hprint
  · rw [hvs, hmap]
    exact qsort_pairwise hsorted

end Verity.Sampled.MemberForm
