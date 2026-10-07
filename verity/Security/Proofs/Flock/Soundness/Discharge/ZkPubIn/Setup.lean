import Proofs.Flock.Soundness.Discharge.ZkPubIn.Spec
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Tables
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepted

/-!
# What `Stmt.setupHidden` and `Stmt.setupTables` set up with the verifier's public inputs

HiddenExec's facts about `Stmt.setupHidden` and `Stmt.setupTables` at `publicInputs := none`
(`HiddenExec/Setup.lean`, `Accepts.lean`, `Accepted.lean`, `ZkHidden/SoundDrawOk.lean`) at `some own`, each read off
`setupHidden_specP` as the originals are off `Hidden.setupHidden_spec`:

* `setupHidden_parseCP`, `setupHidden_tagsP`, `setupHidden_hm96P`, `setupHidden_nP`;
* `setupHidden_wfP`: `StmtWF` and `RegionsWF`, the in regions (`HmIn.regions`) as well formed as the out regions
  (`inRegions_wf`);
* `setupHidden_drawnP`, `setupHidden_mPts_drawnP`, `setupHidden_mPts_of_provedP`: a drawn statement's public file is
  `HmRow.drawn` at the circuit with its public input words as its one output port (`HmIn.drawn`), so its `n` is the
  draw's proved units;
* `setupTables_okHP`, `setupTables_retainHP`, `setupTables_pointsHP`: one table under hidden outputs is
  `setupHidden … (some own)`'s statement;
* `drawSetupHP_of_setupHidden`: a `DrawSetupHP` from setupHidden's acceptance with `own`.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.ZkPubIn

open Flock FlockLevel3 Refine Exec HiddenExec
open Lean (Json)

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {own : ByteArray} {st : Stmt}

theorem setupHidden_parseCP
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program (some own) = .ok st) :
    HmOut.parse tags cf tables = .ok st.c :=
  (setupHidden_specP h).choose_spec.parse

theorem setupHidden_tagsP
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program (some own) = .ok st) : st.tags = tags :=
  (setupHidden_specP h).choose_spec.tagsEq

theorem setupHidden_hm96P (hl : tags.merkleLeaf = "hm96-sha512/v1")
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program (some own) = .ok st) :
    st.merkle = MerkleScheme.hm96Sha512 := by
  obtain ⟨_, s⟩ := setupHidden_specP h
  have hf := s.merkle
  rw [Hidden.parse_leafScheme s.parse, hl, Exec.find_hm96] at hf
  exact (Option.some.inj hf).symm

theorem setupHidden_nP
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program (some own) = .ok st) :
    st.n = st.pub.n ∧ st.nbl = max 3 (log2ceil (max ((st.n + st.c.g - 1) / st.c.g) 1)) := by
  obtain ⟨_, s⟩ := setupHidden_specP h
  exact ⟨s.n, by rw [s.nbl, s.blocksReal]⟩

/-- `HmIn.regions`' in regions are well formed, each `mkRegion`'s. -/
theorem inRegions_wf {c : Circuit} {pi : HmIn.Ports} {pub : Public} {regs : Array Region}
    (h : HmIn.regions c pi pub = .ok regs) : ∀ x ∈ regs.toList, RegionWF c.kLog x := by
  unfold HmIn.regions at h
  split at h
  · simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp
  · obtain ⟨-, hall⟩ := Exec.range_mapM_ok h
    intro x hx
    obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
    exact mkRegion_wf (hall j (by simpa using hj))

/-- **The statement is well formed** (`StmtWF`, `RegionsWF`), as `Hidden.setupHidden_wf`. -/
theorem setupHidden_wfP
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program (some own) = .ok st) :
    StmtWF st ∧ RegionsWF st := by
  obtain ⟨_, s⟩ := setupHidden_specP h
  obtain ⟨ro, ri, hro, hri, he⟩ := s.regions
  refine ⟨wf_of_facts (Hidden.parse_slots s.parse) (hm_pin_ok s.pin) s.m, s.m, s.mPts, ?_⟩
  rw [he, Array.toList_append]
  intro x hx
  rcases List.mem_append.1 hx with hx | hx
  · exact Hidden.regions_wf hro x hx
  · exact inRegions_wf hri x hx

/-- `SpecP.pub` at a draw: the drawn public file is `HmRow.drawn`'s at the circuit with the public input words as its
one output port. -/
theorem setupHidden_drawnP {d : Json}
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program (some own) = .ok st) :
    ∃ (pi : HmIn.Ports) (pop : Public), pi.ports.isEmpty = false ∧
      HmRow.drawn { st.c with outPorts := #[("public inputs", pi.words)] } pop d = .ok st.pub := by
  obtain ⟨pi, s⟩ := setupHidden_specP h
  obtain ⟨pop, -, -, -, -, hd⟩ := s.pub
  have hdr := hd d rfl
  unfold HmIn.drawn at hdr
  simp only [s.ports, Bool.false_eq_true, ↓reduceIte] at hdr
  exact ⟨pi, pop, s.ports, hdr⟩

/-- **`m_pts` at a drawn session with the verifier's public inputs** is `mPtsOf` of the circuit's `g` and the number of
units the draw proves (`ZkHidden.setupHidden_mPts_drawn`). -/
theorem setupHidden_mPts_drawnP {d : Json}
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program (some own) = .ok st) :
    ∃ ud, Draw.UnitDraw.ofJson d = .ok ud ∧ st.mPts = mPtsOf st.c.g ud.proved.size := by
  obtain ⟨_, s⟩ := setupHidden_specP h
  obtain ⟨_, pop, -, hdr⟩ := setupHidden_drawnP h
  obtain ⟨ud, hud, hn⟩ := drawn_n hdr
  refine ⟨ud, hud, ?_⟩
  rw [s.mPts, s.nbl, s.blocksReal, s.n, hn]
  rfl

/-- **`m_pts` from the circuit and the draw's proved count** (`ZkHidden.setupHidden_mPts_of_proved`). -/
theorem setupHidden_mPts_of_provedP {c : Circuit} {d : Json} {N₀ : ℕ} (hc : HmOut.parse tags cf tables = .ok c)
    (hD : ∀ D, Draw.UnitDraw.ofJson d = .ok D → D.proved.size = N₀)
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program (some own) = .ok st) :
    st.mPts = mPtsOf c.g N₀ := by
  obtain ⟨ud, hud, hm⟩ := setupHidden_mPts_drawnP h
  have e := setupHidden_parseCP h
  rw [hc] at e
  rw [hm, ← Except.ok.inj e, hD ud hud]

variable {count : ℕ} {work : Option (ℕ × Array (String × ℕ × ℕ))} {stratified : Option ℕ}
  {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Flock.Stmt}

set_option maxHeartbeats 4000000 in
/-- **`setupTables` with the verifier's public inputs sets up each table with `setupHidden … (some own)`** under a
tag with hidden outputs (`HiddenExec.setupTables_okH`). -/
theorem setupTables_okHP (hh : tags.hiddenOutputs = true)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure (some own) =
      .ok (spec, sts)) :
    (count ≤ 1 ∧ ∃ st, Stmt.setupHidden tags cf pf coins tables draw partition program (some own) = .ok st ∧
        spec = st.spec ∧ sts = #[st]) ∨
      (1 < count ∧ sts.size = count ∧ (∀ (j : ℕ) (hj : j < sts.size), ∃ d,
        Stmt.setupHidden { tags with identity := tags.tableIdentity count j } cf pf coins tables (some d) partition
          program (some own) = .ok sts[j]) ∧ (∀ (j : ℕ) (hj : j < sts.size), sts[j].mPts = spec.mPts) ∧
        spec.points = 2) := by
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
  · exact Or.inl ⟨by assumption, _, by assumption, rfl, rfl⟩
  all_goals
    obtain ⟨hsz, hall⟩ := range_mapM_ok (by assumption)
    exact Or.inr ⟨by omega, hsz, fun j hj => ⟨_, hall j hj⟩, mPts_of_any (by assumption), rfl⟩

/-- One table: its spec retains rounds exactly when its tags do, and its two link points have the table's length. -/
theorem setupTables_oneHP (hh : tags.hiddenOutputs = true) (hc : count ≤ 1)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure (some own) =
      .ok (spec, sts)) :
    ∃ st, Stmt.setupHidden tags cf pf coins tables draw partition program (some own) = .ok st ∧ spec = st.spec ∧
      sts = #[st] ∧ spec.retainRounds = tags.retainRounds ∧ spec.points = 2 ∧
      ∀ (j : ℕ) (hj : j < sts.size), sts[j].mPts = spec.mPts := by
  rcases setupTables_okHP hh h with ⟨-, st, hst, rfl, rfl⟩ | ⟨hc', -⟩
  · refine ⟨st, hst, rfl, rfl, congrArg Tags.retainRounds (setupHidden_tagsP hst), rfl, fun j hj => ?_⟩
    have : j = 0 := by simpa using hj
    subst this
    rfl
  · omega

variable {I : Inputs} {mPts n : ℕ} {S : Finset (Fin n)} {d : Lean.Json} {c : Flock.Circuit}

/-- **A `DrawSetupHP` from setupHidden's acceptance with `own` at a draw** (`HiddenExec.drawSetupH_of_setupHidden`). -/
theorem drawSetupHP_of_setupHidden (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
    (h : Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program
      (some own) = .ok st)
    {sch : Flock.Schedule} (hf : Flock.fast100 st.m = some sch) {D : Draw.UnitDraw}
    (hD : Draw.UnitDraw.ofJson d = .ok D) (hS : ∀ u ∈ S, u.val ∈ D.proved) (hmPts : st.mPts = mPts) :
    ∃ x : DrawSetupHP I own mPts S d, x.st = st := by
  have hcst : st.c = c := Except.ok.inj ((setupHidden_parseCP h).symm.trans hc)
  have hg : 0 < st.c.g := hcst ▸ Integrate.g_pos (parse_untyped ht hc).1 ho
  obtain ⟨_, pop, -, hdr⟩ := setupHidden_drawnP h
  obtain ⟨D', hD', -, hn, -⟩ := Layout.drawn_spec hdr
  rw [hD] at hD'
  cases hD'
  obtain ⟨hn', hnbl⟩ := setupHidden_nP h
  have hblk : D.proved.size ≤ st.c.g * 2 ^ st.nbl := by
    rw [hnbl, ← hn, ← hn']
    exact Integrate.fits_blocks hg
  obtain ⟨k0, l₀, l₁, rest, ext, hsch⟩ := Integrate.model_sched hf
  exact ⟨⟨st, h, D, hD, hS, hg, hblk, st.m, k0, l₀, l₁, rest, ext, hsch,
    by have := exec_fast100_ge hf; omega, hmPts, setupHidden_wfP h⟩, rfl⟩

end FlockSoundness.Discharge.ZkPubIn
