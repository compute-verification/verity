import Proofs.Flock.Soundness.Discharge.HiddenExec.AcceptsP
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepted
import Proofs.Flock.Soundness.Discharge.HiddenExec.TablesP

/-!
# What a hidden-output setup at public inputs establishes

`Hidden/Facts.lean`, `HiddenExec/Setup.lean`, `Accepts.lean` and `Accepted.lean` at the verifier's public inputs `own`,
read off `Hidden.SpecP` (`Hidden.setupHidden_specP`) and the walk of `setupTables_okHP`:

* `Hidden.regionsIn_wf`: every region `HmIn.regions` builds is `mkRegion`'s, so well formed;
* `Hidden.setupHidden_tagsP`, `setupHidden_hm96P`, `setupHidden_wfP`, `setupHidden_parseCP`, `setupHidden_nP`,
  `setupHidden_drawnP`: `setupHidden_tags`, `setupHidden_hm96`, `setupHidden_wf`, `setupHidden_parseC`,
  `setupHidden_n` and `setupHidden_drawn` (the draw over the public input ports, `HmIn.drawn`);
* `setupTables_retainHP`, `setupTables_pointsHP`: `setupTables_retainH` and `setupTables_pointsAny` under hidden
  outputs;
* `drawSetupHP_of_setupHiddenI`: `drawSetupH_of_setupHiddenI`.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Hidden

open Flock FlockLevel3 Refine
open Lean (Json)

/-- Every region `HmIn.regions` builds is `mkRegion`'s, so well formed (none without public input ports). -/
theorem regionsIn_wf {c : Circuit} {pi : HmIn.Ports} {pub : Public} {regs : Array Region}
    (h : HmIn.regions c pi pub = .ok regs) : ∀ x ∈ regs.toList, RegionWF c.kLog x := by
  unfold HmIn.regions at h
  split at h
  · simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp
  · obtain ⟨-, hall⟩ := Exec.range_mapM_ok h
    intro x hx
    obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
    have hj' := hall j (by simpa using hj)
    dsimp only at hj'
    first
      | exact mkRegion_wf hj'
      | (split at hj'; exact mkRegion_wf hj')

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program own : Option ByteArray} {st : Stmt}

/-- `SpecP.tagsEq`. -/
theorem setupHidden_tagsP (h : Stmt.setupHidden tags cf pf coins tables draw partition program own = .ok st) :
    st.tags = tags :=
  (setupHidden_specP h).choose_spec.tagsEq

/-- `SpecP.parseF`, at a flat statement's tags. -/
theorem setupHidden_parseCP (hi : tags.isTmpl = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program own = .ok st) :
    HmOut.parse tags cf tables = .ok st.c :=
  (setupHidden_specP h).choose_spec.parseF hi

/-- **The leaf scheme at `own`** (`setupHidden_hm96`). -/
theorem setupHidden_hm96P (hl : tags.merkleLeaf = "hm96-sha512/v1")
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program own = .ok st) :
    st.merkle = MerkleScheme.hm96Sha512 := by
  have s := (setupHidden_specP h).choose_spec
  have hf := s.merkle
  rw [parseCommit_leafScheme s.parse, hl, Exec.find_hm96] at hf
  exact (Option.some.inj hf).symm

/-- **The statement is well formed at `own`** (`setupHidden_wf`): its regions are `HmOut`'s and `HmIn`'s. -/
theorem setupHidden_wfP (h : Stmt.setupHidden tags cf pf coins tables draw partition program own = .ok st) :
    StmtWF st ∧ RegionsWF st := by
  obtain ⟨pi, s⟩ := setupHidden_specP h
  obtain ⟨ro, ri, hro, hri, hr⟩ := s.regions
  refine ⟨wf_of_facts (parseCommit_slots s.parse) (hm_pin_ok s.pin) s.m, s.m, s.mPts, ?_⟩
  intro x hx
  rw [hr, Array.toList_append, List.mem_append] at hx
  rcases hx with hx | hx
  · exact regions_wf hro x hx
  · exact regionsIn_wf hri x hx

/-- `SpecP.n` and `SpecP.nbl` (`HiddenExec.setupHidden_n`). -/
theorem setupHidden_nP (h : Stmt.setupHidden tags cf pf coins tables draw partition program own = .ok st) :
    st.n = st.pub.n ∧ st.nbl = max 3 (log2ceil (max ((st.n + st.c.g - 1) / st.c.g) 1)) := by
  have s := (setupHidden_specP h).choose_spec
  exact ⟨s.n, by rw [s.nbl, s.blocksReal]⟩

/-- `SpecP.pub` at a draw: the statement's public file is the population's drawn over the public input ports. -/
theorem setupHidden_drawnP {d : Json}
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program own = .ok st) :
    ∃ pi pop, HmIn.parse st.c = .ok pi ∧ HmIn.loadPublic tags st.c pi pf = .ok pop ∧
      HmIn.drawn st.c pi pop d = .ok st.pub := by
  obtain ⟨pi, s⟩ := setupHidden_specP h
  obtain ⟨pop, hl, -, -, hd⟩ := s.pub
  exact ⟨pi, pop, s.ports, hl, hd d rfl⟩

end FlockSoundness.Discharge.Hidden

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Refine
open FlockSoundness.Discharge.Exec (Inputs)
open FlockSoundness.Discharge.Integrate (model_sched fits_blocks)
open Lean (Json)

section Tables

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program own : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
  {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Flock.Stmt}

set_option maxHeartbeats 4000000 in
/-- The session's spec retains rounds exactly when its tags do, under hidden outputs at `own`
(`setupTables_retainH`). -/
theorem setupTables_retainHP (hh : tags.hiddenOutputs = true)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure own =
      .ok (spec, sts)) :
    spec.retainRounds = tags.retainRounds := by
  unfold Stmt.setupTables at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h : jp0 () = .ok (spec, sts) := by
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h : jp1 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp1] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  replace h : jp2 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp2] at h
  dsimp only at h
  simp only [throw_bind', pure_bind, ite_eq_left hh] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
  · exact congrArg Tags.retainRounds (Hidden.setupHidden_tagsP (FlockSoundness.Discharge.Exec.ok_of_refuse (by assumption)))
  all_goals rfl

/-- The session's two link points have every table's length, under hidden outputs at `own`
(`setupTables_pointsAny`). -/
theorem setupTables_pointsHP (hh : tags.hiddenOutputs = true)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure own =
      .ok (spec, sts)) :
    spec.points = 2 ∧ ∀ (j : ℕ) (hj : j < sts.size), sts[j].mPts = spec.mPts := by
  rcases setupTables_okHP hh h with ⟨-, st, -, rfl, rfl⟩ | ⟨-, -, -, hm, hp⟩
  · exact ⟨rfl, fun j hj => by
      have : j = 0 := by simpa using hj
      subst this
      rfl⟩
  · exact ⟨hp, hm⟩

end Tables

variable {I : Inputs} {own : Option ByteArray} {mPts n : ℕ} {S : Finset (Fin n)} {d : Lean.Json} {c : Flock.Circuit}
  (ht : I.tags.typed = false) (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
include ht hc

/-- **A `DrawSetupHP` from setupHidden's acceptance at `own` at a draw** (`drawSetupH_of_setupHiddenI`), a `fast100`
schedule of its `m`, the draw proving `S`, and its link points `mPts`. -/
theorem drawSetupHP_of_setupHiddenI {st : Flock.Stmt}
    (h : Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program own =
      .ok st)
    {sch : Flock.Schedule} (hf : Flock.fast100 st.m = some sch) {D : Draw.UnitDraw}
    (hD : Draw.UnitDraw.ofJson d = .ok D) (hS : ∀ u ∈ S, u.val ∈ D.proved) (hmPts : st.mPts = mPts) :
    ∃ x : DrawSetupHP I own mPts S d, x.st = st := by
  have hcst : st.c = c := Except.ok.inj ((Hidden.setupHidden_parseCP (HmOut.parse_isTmpl hc) h).symm.trans hc)
  have hg : 0 < st.c.g := by
    have P := parse_facts (parse_untyped ht hc).1
    obtain ⟨ls, hls⟩ := P.check
    have hp := (check_facts P.typed hls).g_pow2
    simp only [isPow2, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hp
    exact hcst ▸ hp.1
  obtain ⟨pi, pop, -, -, hdr⟩ := Hidden.setupHidden_drawnP h
  obtain ⟨D', hD', -, hn, -⟩ := Layout.drawn_spec
    (c := { st.c with outPorts := if pi.ports.isEmpty then #[] else #[("public inputs", pi.words)] }) hdr
  rw [hD] at hD'
  cases hD'
  obtain ⟨hn', hnbl⟩ := Hidden.setupHidden_nP h
  have hblk : D.proved.size ≤ st.c.g * 2 ^ st.nbl := by
    rw [hnbl, ← hn, ← hn']
    exact fits_blocks hg
  obtain ⟨k0, l₀, l₁, rest, ext, hsch⟩ := model_sched hf
  exact ⟨⟨st, h, D, hD, hS, hg, hblk, st.m, k0, l₀, l₁, rest, ext, hsch,
    by have := exec_fast100_ge hf; omega, hmPts, Hidden.setupHidden_wfP h⟩, rfl⟩

end FlockSoundness.Discharge.HiddenExec
