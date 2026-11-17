module

public import Proofs.Properties.Verity.Sampled.SessionAt
public import Proofs.Properties.Verity.Sampled.ParamsProgram
public import Proofs.Flock.Bridge.SampledDesign
public import Proofs.Flock.Bridge.Refused
public import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
public import Proofs.Flock.Soundness.Discharge.ZkBind.Rate
public import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundSampleOk
public import Proofs.Flock.Soundness.Discharge.ZkHidden.Sched
public import Proofs.Flock.Soundness.Discharge.HiddenExec.SetupP
public import Proofs.Flock.Soundness.Discharge.Hidden.PortsP
public import Proofs.Flock.Soundness.Discharge.Layout.Drawn
public import Proofs.Flock.Claims.ZK.MaskedTables

/-! What v2's acceptance gives of the session at the registered files (`liveOf`). -/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open FlockSoundness.Refine (ebind_ok)

open FlockSoundness.Refine in
theorem oneStage_paramsOfJson_program {j : Json} {p : _root_.Spot.OneStage.Params}
    (h : _root_.Spot.OneStage.paramsOfJson j = .ok p) :
    ∃ b, _root_.Spot.Program.parse b = .ok p.verifier.program := by
  revert p
  show Ret (_root_.Spot.OneStage.paramsOfJson j) fun p => ∃ b, _root_.Spot.Program.parse b = .ok p.verifier.program
  rw [_root_.Spot.OneStage.paramsOfJson.eq_def]
  ret_walk_aux [_root_.Spot.OneStage.paramsParsed] =>
    exact paramsParsed_program (by assumption)

theorem oneStageZk_paramsParsed_program {j : Json} {p : _root_.Spot.OneStageZk.Params}
    (h : _root_.Spot.OneStageZk.paramsParsed j = .ok p) :
    ∃ b, _root_.Spot.Program.parse b = .ok p.base.verifier.program := by
  unfold _root_.Spot.OneStageZk.paramsParsed at h
  repeat' (first
    | (obtain ⟨_, _, h⟩ := ebind_ok h)
    | (split at h)
    | (cases h; done))
  all_goals first
    | (simp [throw, throwThe, MonadExceptOf.throw] at h; done)
    | (simp only [pure, Except.pure, Except.ok.injEq] at h
       subst h
       exact oneStage_paramsOfJson_program ‹_›)

theorem oneStageZk_paramsOfJson_program {j : Json} {p : _root_.Spot.OneStageZk.Params}
    (h : _root_.Spot.OneStageZk.paramsOfJson j = .ok p) :
    ∃ b, _root_.Spot.Program.parse b = .ok p.base.verifier.program :=
  oneStageZk_paramsParsed_program (_root_.Spot.OneStageZk.paramsOfJson_parsed h)

theorem program_of_paramsAt {θ : ByteArray} {p : _root_.Spot.OneStageZk.Params} (hp : ParamsAt θ p) :
    ∃ desc : Claims.Msg, _root_.Spot.Program.parse ⟨desc.toArray⟩ = .ok p.base.verifier.program := by
  obtain ⟨_, -, x, -, hj⟩ := hp
  obtain ⟨_, -, hj⟩ := ebind_ok hj
  obtain ⟨_, hr⟩ := dispatch_params hj
  obtain ⟨_, p', _, hp', hb⟩ := _root_.Spot.OneStageZk.paramsEitherOfJson_base hr
  obtain ⟨b, hb'⟩ := oneStageZk_paramsParsed_program hp'
  refine ⟨b.data.toList, ?_⟩
  rw [hb]
  exact hb'

theorem kindOf_paramsAt {θ : ByteArray} {p : _root_.Spot.OneStageZk.Params} (hp : ParamsAt θ p)
    (hm : p.members = none) :
    Properties.Spot.SecurityProofs.kindOf p.base.verifier.samplingDesign = "subset" := by
  obtain ⟨_, -, x, -, hj⟩ := hp
  obtain ⟨_, -, hj⟩ := ebind_ok hj
  obtain ⟨_, hr⟩ := dispatch_params hj
  unfold Properties.Spot.SecurityProofs.kindOf
  rw [_root_.Spot.OneStageZk.paramsOfJson_subset (_root_.Spot.OneStageZk.paramsEitherOfJson_one hr hm)]

theorem samplingDesignAt_designAt {desc : Claims.Msg} {v : _root_.Spot.Docket.Verifier} {lo : Json}
    (hp : _root_.Spot.Program.parse ⟨desc.toArray⟩ = .ok v.program)
    (hlo : _root_.Spot.Draw.derive v.program v.query v.samplingDesign v.workTable = .ok lo) {n : ℕ} {src : ByteArray}
    {r : Array ℕ × ℕ} (h : _root_.Spot.OneStage.sampledFrom v lo n src = .ok r) :
    samplingDesignAt v n = .ok (Flock.Bridge.designAt desc v.query v.samplingDesign v.workTable n) := by
  obtain ⟨samplingDesign, ⟨hl, -⟩, -⟩ := Flock.Bridge.samplesUnder_of_sampledFrom hp hlo h
  obtain ⟨p', derived, parsed, hp', hd, hpar, rfl⟩ := Flock.Bridge.sampledDesign_ok hl
  obtain rfl : p' = v.program := Except.ok.inj (hp'.symm.trans hp)
  have hdes : Flock.Bridge.designAt desc v.query v.samplingDesign v.workTable n = derived.getD parsed := by
    unfold Flock.Bridge.designAt
    rw [hl]
  rw [hdes]
  have hlo' := hlo
  unfold _root_.Spot.Draw.derive at hlo'
  obtain ⟨derived', hd', hlo'⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 hlo'
  obtain rfl : derived' = derived := Except.ok.inj (hd'.symm.trans hd)
  obtain rfl : Properties.Spot.SecurityProofs.samplingDesignObj v.samplingDesign derived' = lo := Except.ok.inj hlo'
  simp only [samplingDesignAt, hlo, hd, hpar, ok_bind]
  rfl

def TablesRegistered (T : PinnedTables) (s : Verity.Sent) : Prop :=
  ∀ t ∈ Flock.MUFU_TABLES, ∀ a, tableFile t.sha512 s.blobs = some a → ∃ b, T.file t.sha512 = some b

theorem pinnedTables_eq {T : PinnedTables} {s : Verity.Sent} {circuitB : ByteArray}
    (hl : _root_.Spot.OneStageZk.lookupsOf circuitB =
      .ok ((_root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray s.blobs)).map (·.1)))
    (ht : TablesSent T s) (hT : TablesRegistered T s) :
    pinnedTables T circuitB = _root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray s.blobs) := by
  classical
  simp only [pinnedTables, hl]
  rw [tablesOf_ofArray]
  refine List.filterMap_congr fun t htm => ?_
  cases hf : tableFile t.sha512 s.blobs with
  | none =>
    rw [Option.map_none, if_neg]
    intro hc
    have hm : t.name ∈ ((Flock.MUFU_TABLES.filterMap fun t =>
        (tableFile t.sha512 s.blobs).map fun b => (t.name, b)).map (·.1)) := by simpa using hc
    obtain ⟨y, hy, hyn⟩ := List.mem_map.1 hm
    obtain ⟨t', ht', hft'⟩ := List.mem_filterMap.1 hy
    obtain ⟨a, ha, rfl⟩ := Option.map_eq_some_iff.1 hft'
    have htt : t' = t := List.inj_on_of_nodup_map FlockSoundness.Refine.mufu_nodup ht' htm hyn
    subst htt
    rw [hf] at ha
    cases ha
  | some a =>
    have hmem : t.name ∈ ((Flock.MUFU_TABLES.filterMap fun t =>
        (tableFile t.sha512 s.blobs).map fun b => (t.name, b)).map (·.1)) :=
      List.mem_map.2 ⟨(t.name, a), List.mem_filterMap.2 ⟨t, htm, by rw [hf]; rfl⟩, rfl⟩
    rw [if_pos (by simpa using hmem)]
    obtain ⟨b, hb⟩ := hT t htm a hf
    rw [hb, ht t htm a b hf hb]

def scopeOf (I : Flock.Witness.Inputs) (pubB : ByteArray) (rd : _root_.Spot.OneStageZk.Reads) : Flock.V Unit := do
  Flock.ProvedScope.checkOf
    (Flock.ProvedScope.circuitOf (Flock.Witness.zkTags I) I.circuitFile I.tables) (Flock.Witness.zkTags I) true pubB
    I.count false rd.registered.isSome
  _root_.Spot.OneStageZk.templateOk
    (← Flock.ProvedScope.circuitOf (Flock.Witness.zkTags I) I.circuitFile I.tables)

theorem verdictOf_scope {I : Flock.Witness.Inputs} {scope ksFiles population : Flock.V Unit}
    {coinKeys : Array String} {record : String} {draw : Option Json} {st : _root_.Spot.OneStageZk.SetUp}
    {files : Flock.V (Array ByteArray)} {coinKey : Option String}
    (h : _root_.Spot.OneStageZk.verdictOf I scope ksFiles population coinKeys record draw st files coinKey
      = .ok ()) : scope = .ok () := by
  unfold _root_.Spot.OneStageZk.verdictOf at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨⟨_, _, _⟩, -, h⟩ := ebind_ok h
  dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, hs, -⟩ := ebind_ok h
  exact hs

theorem step_scope {I : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads}
    {bs : _root_.Core.Blobs} {scope ksFiles population : Flock.V Unit} {programSha : Option String} {want : Json}
    {s : _root_.Spot.OneStageZk.RunFiles} {r : _root_.Spot.OneStageZk.Run}
    (h : _root_.Spot.OneStageZk.step I rd bs scope ksFiles population programSha want ⟨#[], #[], #[], true⟩ s
      = .ok r)
    (hall : r.all = true) : scope = .ok () := by
  unfold _root_.Spot.OneStageZk.step at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  dsimp only at h
  split at h
  · obtain ⟨_, -, h⟩ := ebind_ok h
    exact verdictOf_scope (push_ok h hall)
  · cases h

theorem run_scope {p : _root_.Spot.OneStageZk.Own} {s : _root_.Spot.OneStageZk.RunFiles}
    {pu : Option (Array ℕ)} {bs : _root_.Core.Blobs} {r : _root_.Spot.OneStageZk.Run}
    (h : _root_.Spot.OneStageZk.run p #[s] pu bs = .ok r) :
    ∃ tags0 circuitB pubB rd, Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧
      bs.get p.circuit "circuit" = .ok circuitB ∧ bs.get p.pub "public file" = .ok pubB ∧
      ∃ ksFiles population programSha want,
        _root_.Spot.OneStageZk.step (_root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs) rd bs
          (scopeOf (_root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs) pubB rd) ksFiles population
          programSha want ⟨#[], #[], #[], true⟩ s = .ok r := by
  unfold _root_.Spot.OneStageZk.run at h
  dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  split at h
  · rename_i tags0 htags
    obtain ⟨_, -, h⟩ := ebind_ok h
    obtain ⟨_, -, h⟩ := ebind_ok h
    obtain ⟨circuitB, hc, h⟩ := ebind_ok h
    obtain ⟨pubB, hp, h⟩ := ebind_ok h
    obtain ⟨_, -, h⟩ := ebind_ok h
    obtain ⟨rd, -, h⟩ := ebind_ok h
    obtain ⟨r', hr', h⟩ := ebind_ok h
    have e : r' = r := by
      cases pu with
      | none => simpa only [pure, Except.pure, Except.ok.injEq] using h
      | some want =>
        repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
        all_goals simpa only [pure, Except.pure, Except.ok.injEq] using h
    subst e
    exact ⟨tags0, circuitB, pubB, rd, htags, hc, hp, _, _, _, _, foldlM_single hr'⟩
  · cases h

theorem checkBlobsAt_scope {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ξ : ByteArray} {e : _root_.Spot.OneStageZk.Evidence} {bs : _root_.Core.Blobs}
    {pub : _root_.Spot.OneStage.Public} {prof : _root_.Core.Integrity}
    (hm : p.members = none) (h : _root_.Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ tags0 circuitB pubB reg, Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧
      bs.get p.own.circuit "circuit" = .ok circuitB ∧ bs.get p.own.pub "public file" = .ok pubB ∧
      Flock.ProvedScope.check
        (Flock.Witness.zkTags (_root_.Spot.OneStageZk.inputsOf tags0 p.own circuitB pubB bs)) true circuitB pubB
        (_root_.Spot.OneStageZk.tablesOf bs) 1 false reg = .ok () := by
  obtain ⟨r, -, hsz, -, -, hrun, hall⟩ := checkBlobsAt_run hm h
  rw [eq_single hsz] at hrun
  obtain ⟨tags0, circuitB, pubB, rd, ht, hc, hp, _, _, _, _, hstep⟩ := run_scope hrun
  have hs := step_scope hstep hall
  unfold scopeOf at hs
  obtain ⟨⟨⟩, hck, -⟩ := ebind_ok hs
  exact ⟨tags0, circuitB, pubB, rd.registered.isSome, ht, hc, hp, hck⟩

theorem checkBlobsAt_setup {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ξ : ByteArray} {e : _root_.Spot.OneStageZk.Evidence} {bs : _root_.Core.Blobs}
    {pub : _root_.Spot.OneStage.Public} {prof : _root_.Core.Integrity}
    (hm : p.members = none) (h : _root_.Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ tags0 circuitB pubB rd record x, Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧
      bs.get p.own.circuit "circuit" = .ok circuitB ∧ bs.get p.own.pub "public file" = .ok pubB ∧
      sessionRecord e bs = some record ∧
      _root_.Spot.OneStageZk.setupAt (_root_.Spot.OneStageZk.inputsOf tags0 p.own circuitB pubB bs) rd
        (Flock.Witness.recordDraw record) = .ok x := by
  obtain ⟨r, -, hsz, -, -, hrun, hall⟩ := checkBlobsAt_run hm h
  have hs := eq_single hsz
  rw [hs] at hrun
  obtain ⟨tags0, circuitB, pubB, rd, ht, hc, hp, -, _, _, _, hstep⟩ := run_event hrun
  obtain ⟨recB, record, _, _, _, _, hb, hrec, hst, -⟩ := step_event hstep hall
  exact ⟨tags0, circuitB, pubB, rd, record, _, ht, hc, hp, sessionRecord_eq hs hb hrec, hst⟩

open FlockSoundness.Discharge.Recursion.Witness (toExec) in
theorem setupAt_hidden {W : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads} {record : String}
    {x : Flock.SessionSpec × Array Flock.Setup × Json × Json} (hc1 : W.count = 1)
    (hh : (Flock.Witness.zkTags W).hiddenOutputs = true)
    (h : _root_.Spot.OneStageZk.setupAt W rd (Flock.Witness.recordDraw record) = .ok x) :
    ∃ st, Flock.Stmt.setupHidden (FlockSoundness.Discharge.Composed.inputsJ (toExec W) 0).tags
        (FlockSoundness.Discharge.Composed.inputsJ (toExec W) 0).circuitFile
        (FlockSoundness.Discharge.Composed.inputsJ (toExec W) 0).publicFile
        (FlockSoundness.Discharge.Composed.inputsJ (toExec W) 0).coins
        (FlockSoundness.Discharge.Composed.inputsJ (toExec W) 0).tables
        (FlockSoundness.Discharge.Exec.recordDraw record)
        (FlockSoundness.Discharge.Composed.inputsJ (toExec W) 0).partition
        (FlockSoundness.Discharge.Composed.inputsJ (toExec W) 0).program none = .ok st ∧
      Flock.Zk.shapeOk st = .ok () := by
  obtain ⟨spec, tabs, y⟩ := x
  obtain ⟨sts, hset, hshape, -⟩ := setupAt_setupOfP h
  unfold FlockSoundness.Discharge.Exec.setupOfP at hset
  have htJ : (FlockSoundness.Discharge.Composed.inputsJ (toExec W) 0).tags =
      FlockSoundness.Discharge.ZkExec.zkTags (toExec W) := by
    show FlockSoundness.Discharge.Exec.tagsAt _ W.count 0 = _
    unfold FlockSoundness.Discharge.Exec.tagsAt
    exact ite_eq_left (by omega)
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okHP hh hset with ⟨-, st, hst, -, rfl⟩ | ⟨hc, -⟩
  · refine ⟨st, ?_, FlockSoundness.Discharge.ZkExec.array_forM_ok hshape 0 (by simp)⟩
    rw [htJ]
    exact hst
  · exact absurd hc (by show ¬ 1 < W.count; omega)

theorem desc_of_ports {I : FlockSoundness.Discharge.Exec.Inputs}
    (hflat : (FlockSoundness.Discharge.ZkExec.zkTags I).hiddenOutputs = true ∧
      (FlockSoundness.Discharge.ZkExec.zkTags I).isTmpl = false)
    (ht : (FlockSoundness.Discharge.ZkExec.zkTags I).typed = false)
    (h : Flock.PortRoots.checkOf
      (Flock.ProvedScope.circuitOf (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables)
      (FlockSoundness.Discharge.ZkExec.zkTags I) I.publicFile = .ok ()) :
    ∃ D pubO, Flock.Desc.parse (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok D ∧
      FlockSoundness.Built D.unroll.unit ∧
      Flock.HmOut.loadPublic (FlockSoundness.Discharge.ZkExec.zkTags I) D.unroll I.publicFile = .ok pubO ∧
      Flock.ProvedScope.circuitOf (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok D.unroll := by
  rw [circuitOf_parse hflat.1 hflat.2] at h ⊢
  unfold Flock.PortRoots.checkOf Flock.PortRoots.checkWith at h
  split at h
  · cases h
  rename_i c hp
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  rename_i pubO hl
  have hD : Flock.Desc.parse (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok (.single c) := by
    unfold Flock.Desc.parse
    rw [hp]
  refine ⟨.single c, pubO, hD, ?_, hl, hp⟩
  refine FlockSoundness.Discharge.Hidden.built_of_parseStmt (cf := I.circuitFile) (tables := I.tables) (D := .single c) ?_ ht
  unfold Flock.Desc.parseStmt
  rw [if_neg (by simp [hflat.2])]
  exact hD

theorem scopeH_of_form {t : Flock.Tags} {cf pf : ByteArray} {tables : List (String × ByteArray)} {count : ℕ}
    {draw : Option Json} {c : Flock.Circuit} (hc : Flock.ProvedScope.circuitOf t cf tables = .ok c)
    (hi : t.isTmpl = false) (h : Flock.KsScope.checkOf (Flock.ProvedScope.circuitOf t cf tables) t pf count draw = .ok ()) :
    FlockSoundness.Discharge.Layout.scopeOkH c = true := by
  rw [hc] at h
  unfold Flock.KsScope.checkOf at h
  cases htw : Flock.KsScope.tagsWhy t with
  | some _ => simp [htw] at h
  | none =>
    cases hl : Flock.HmOut.loadPublic t c pf with
    | error _ => simp [htw, hl] at h
    | ok pub =>
      cases hfw : Flock.KsScope.filesWhy t count c pub pf with
      | some _ => simp [htw, hl, hfw] at h
      | none =>
        unfold Flock.KsScope.filesWhy at hfw
        cases hu : Flock.KsScope.unitWhy t count c with
        | some _ => simp [hu] at hfw
        | none =>
          unfold Flock.KsScope.unitWhy at hu
          rw [← FlockSoundness.Discharge.FailClosed.scopeOkH_eq]
          cases hs : Flock.ProvedScope.scopeOkH c <;> simp [hi, hs] at hu ⊢

theorem printed_units {n : ℕ} {units : Array ℕ} (h : unitsAscending n units = true) :
    ∃ S : Finset (Fin n), ((S.image Fin.val).sort (· ≤ ·)).toArray = units := by
  classical
  obtain ⟨hlt, hp⟩ := unitsAscending_spec h
  refine ⟨Finset.univ.filter (fun i : Fin n => i.val ∈ units.toList), ?_⟩
  have himg : (Finset.univ.filter (fun i : Fin n => i.val ∈ units.toList)).image Fin.val = units.toList.toFinset := by
    ext x
    simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and, List.mem_toFinset]
    exact ⟨fun ⟨i, hi, he⟩ => he ▸ hi, fun hx => ⟨⟨x, hlt x hx⟩, hx, rfl⟩⟩
  rw [himg, (List.toFinset_sort (· ≤ ·) (hp.imp ne_of_lt)).2 (hp.imp le_of_lt)]

theorem proved_of_recordDrawn {m n : ℕ} {units : Array ℕ} {record : String}
    (h : recordSampled (.subset m) n units record) {D : Flock.Draw.UnitSample}
    (hD : Flock.Draw.UnitSample.ofJson (Flock.Draw.UnitSample.toJson ⟨.subset m, n, units, #[]⟩) = .ok D) :
    D.proved = units := by
  obtain ⟨S, hS⟩ := printed_units h.1
  have e : Flock.Draw.UnitSample.toJson ⟨.subset m, n, units, #[]⟩ = printedSample (n := n) (.subset m) S := by
    unfold printedSample FlockSoundness.Discharge.ZkLink.drawOf
    rw [hS]
  rw [e] at hD
  rw [sampleFileZK_printedSample m S D hD]
  show (if (#[] : Array ℕ).isEmpty then _ else _) = units
  simp only [Array.isEmpty_empty, ite_true]
  exact hS

theorem samplingDesignOn_designAt {desc : Claims.Msg} {qr lj : Json} {tb : Option Json} {n : ℕ}
    (hpop : Flock.Bridge.PopMatches desc qr n) :
    Flock.Bridge.SamplingDesignOn (Flock.Bridge.designAt desc qr lj tb n) (Flock.Bridge.nsOf desc qr)
      (Flock.Bridge.samplingDesignK (Flock.Bridge.designAt desc qr lj tb n)) := by
  rcases Flock.Bridge.designAt_cases desc qr lj tb n with h | h
  · exact Flock.Bridge.samplingDesignOn_samplesUnder hpop h (Flock.Bridge.notBernoulli_samplesUnder h)
  · rw [h]
    exact ⟨rfl, Nat.zero_le _⟩

open FlockSoundness.Discharge.Composed (inputsJ nTab) in
theorem y₀_of_setup (n : ℕ) {I : FlockSoundness.Discharge.Exec.Inputs} {D : Flock.Desc} {kd n' : ℕ} {recS : String}
    {units : Array ℕ} {st : Flock.Stmt} (hc1 : I.count = 1) (ht : I.tags.typed = false)
    (hD : Flock.Desc.parse (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok D)
    (hrd : recordSampled (.subset units.size) n' units recS) (hsize : units.size = kd)
    (hst : Flock.Stmt.setupHidden (inputsJ I 0).tags (inputsJ I 0).circuitFile (inputsJ I 0).publicFile
      (inputsJ I 0).coins (inputsJ I 0).tables (FlockSoundness.Discharge.Exec.recordDraw recS)
      (inputsJ I 0).partition (inputsJ I 0).program none = .ok st)
    (hshape : Flock.Zk.shapeOk st = .ok ()) :
    ∃ (S₀ : Finset (Fin n)) (d₀ : Json),
      Nonempty (FlockSoundness.Discharge.ZkHidden.SampleSetupZHJP (inputsJ I 0) none
        (FlockSoundness.Discharge.Exec.mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀) := by
  have hdj : FlockSoundness.Discharge.Exec.recordDraw recS =
      some (Flock.Draw.UnitSample.toJson ⟨.subset units.size, n', units, #[]⟩) :=
    (exec_recordSample recS).trans hrd.2
  rw [hdj] at hst
  have htJ : (inputsJ I 0).tags = FlockSoundness.Discharge.ZkExec.zkTags I := by
    show FlockSoundness.Discharge.Exec.tagsAt _ I.count 0 = _
    unfold FlockSoundness.Discharge.Exec.tagsAt
    exact ite_eq_left (by omega)
  have hn1 : nTab I = 1 := by
    unfold nTab
    simp [hc1]
  obtain ⟨⟨sch, tPad⟩, hz⟩ := FlockSoundness.Discharge.ZkHidden.schedule_ok_of_shapeOk hshape
  obtain ⟨sch₀, hf⟩ := FlockSoundness.Discharge.ZkHidden.fast100_of_schedule hz
  have hst' := hst
  rw [htJ] at hst'
  have hmPts : st.mPts = FlockSoundness.Discharge.Exec.mPtsOf D.unroll.g (kd / nTab I) := by
    rw [hn1, Nat.div_one, ← hsize]
    exact FlockSoundness.Discharge.ZkHidden.setupHidden_mPts_of_proved (Flock.Desc.parse_flat hD)
      (fun D' hD' => by rw [proved_of_recordDrawn hrd hD']) hst'
  obtain ⟨pi', pop, -, -, hdr⟩ := FlockSoundness.Discharge.Hidden.setupHidden_sampledP hst
  obtain ⟨Dd, hDd, -⟩ := FlockSoundness.Discharge.Layout.sampled_spec
    (c := { st.c with outPorts := if pi'.ports.isEmpty then #[] else #[("public inputs", pi'.words)] }) hdr
  obtain ⟨xx, hxx⟩ := FlockSoundness.Discharge.HiddenExec.sampleSetupHP_of_setupHiddenI (I := inputsJ I 0)
    (own := none) (S := (∅ : Finset (Fin n))) (FlockSoundness.Discharge.Composed.typedJ ht 0) hst hf hDd
    (fun v hv => by simp at hv) hmPts
  obtain ⟨y, -, -, -⟩ := FlockSoundness.Discharge.ZkHidden.sampleSetupZHJP_of xx (by rw [hxx]; exact hshape)
    (sch := sch) (tPad := tPad) (by rw [hxx]; exact hz)
  exact ⟨∅, _, ⟨y⟩⟩

open FlockSoundness.Discharge.Composed (inputsJ) in
theorem pub_of_setup {I : FlockSoundness.Discharge.Exec.Inputs} {D : Flock.Desc} {pubO : Flock.Public} {n : ℕ}
    {d : Json} {st : Flock.Stmt} (hc1 : I.count = 1)
    (hD : Flock.Desc.parse (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok D)
    (hst : Flock.Stmt.setupHidden (inputsJ I 0).tags (inputsJ I 0).circuitFile (inputsJ I 0).publicFile
      (inputsJ I 0).coins (inputsJ I 0).tables (some d) (inputsJ I 0).partition (inputsJ I 0).program none = .ok st)
    (hpubO : Flock.HmOut.loadPublic (FlockSoundness.Discharge.ZkExec.zkTags I) D.unroll I.publicFile = .ok pubO)
    (hnO : pubO.n = n)
    (hports : Flock.PortRoots.checkOf (.ok D.unroll) (FlockSoundness.Discharge.ZkExec.zkTags I) I.publicFile = .ok ()) :
    Flock.HmIn.parse D.unroll = .ok {} ∧
      ∃ pub, Flock.HmIn.loadPublic (FlockSoundness.Discharge.ZkExec.zkTags I) D.unroll {} I.publicFile = .ok pub ∧
        Flock.PortRoots.dupId D.unroll pub = false ∧ pub.n = n := by
  have htJ : (inputsJ I 0).tags = FlockSoundness.Discharge.ZkExec.zkTags I := by
    show FlockSoundness.Discharge.Exec.tagsAt _ I.count 0 = _
    unfold FlockSoundness.Discharge.Exec.tagsAt
    exact ite_eq_left (by omega)
  rw [htJ] at hst
  have hc := FlockSoundness.Discharge.HiddenExec.setupHidden_c_of_parse (Flock.Desc.parse_flat hD) hst
  obtain ⟨pi, s⟩ := FlockSoundness.Discharge.Hidden.setupHidden_specP hst
  obtain ⟨pop, hlp, hown, -⟩ := s.pub
  have hpi : pi = {} :=
    FlockSoundness.Discharge.Hidden.parse_empty s.ports (FlockSoundness.Discharge.Hidden.checkOwn_none hown)
  have hpp := s.ports
  rw [hc, hpi] at hpp hlp
  have hn : pop.n = n := (Flock.Bridge.loadPublic_inOut hlp hpubO).1.symm.trans hnO
  exact ⟨hpp, pop, hlp, (Flock.Bridge.v5Files_of_checks hlp hn hports).1, hn⟩

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

open FlockSoundness.Discharge.Recursion.Witness (toExec) in
theorem accepted_parts {V : Vocab d α} {T : PinnedTables} {θ : ByteArray} (hD : DigestsAt V T θ)
    {k : Verity.Registration.Composite} {F : Array ByteArray} {c : ByteArray} {o : Verity.Exchange}
    (hone : OneStatement θ)
    (hP : ∃ p, ParamsAt θ p) (hreach : Verity.Game.Reaches (V.game θ F c) o) (hv : V.Verdict k .sampled o)
    (ht : TablesSent T o.sent) (hT : TablesRegistered T o.sent) :
    ∃ p tags0 pubB record proofs n desc, ParamsAt θ p ∧
      Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧
      pubB = V.publicFile θ F ∧
      Accepted (toExec (_root_.Spot.OneStageZk.inputsOf tags0 p.own (V.circuitFile θ F) pubB
        (_root_.Core.Blobs.ofArray o.sent.blobs))) { record := record, proofs := proofs } p.base.verifier n ∧
      _root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray o.sent.blobs) = pinnedTables T (V.circuitFile θ F) ∧
      _root_.Spot.Program.parse ⟨desc.toArray⟩ = .ok p.base.verifier.program ∧
      samplingDesignAt p.base.verifier n = .ok (Flock.Bridge.designAt desc p.base.verifier.query p.base.verifier.samplingDesign
        p.base.verifier.workTable n) ∧
      ∃ reg, Flock.ProvedScope.check (Flock.Witness.zkTags (_root_.Spot.OneStageZk.inputsOf tags0 p.own
          (V.circuitFile θ F) pubB (_root_.Core.Blobs.ofArray o.sent.blobs))) true (V.circuitFile θ F) pubB
        (_root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray o.sent.blobs)) 1 false reg = .ok () ∧
      ∃ recS units rd x, recordSampled (.subset units.size) n units recS ∧
        units.size = Flock.Bridge.samplingDesignK (Flock.Bridge.designAt desc p.base.verifier.query p.base.verifier.samplingDesign
          p.base.verifier.workTable n) ∧
        _root_.Spot.OneStageZk.setupAt (_root_.Spot.OneStageZk.inputsOf tags0 p.own (V.circuitFile θ F) pubB
          (_root_.Core.Blobs.ofArray o.sent.blobs)) rd (Flock.Witness.recordDraw recS) = .ok x := by
  obtain ⟨hθ, hFo, -⟩ := Binding.play_reaches hreach
  have hreg : V.Registered θ F := by
    have h1 := hv.1
    rw [hθ, hFo] at h1
    exact h1
  obtain ⟨p₀, hp₀⟩ := hP
  have hm : p₀.members = none := hone p₀ hp₀
  have hp₀' : ParamsAt o.sent.params p₀ := by
    rw [show o.sent.params = θ from hθ]
    exact hp₀
  have hone' : OneStatement o.params := by
    rw [show o.params = θ from hθ]
    exact hone
  obtain ⟨a, b, e, p, cm, ev, ξ, dr, hcase, hacc⟩ := case_drew hv.2.1 hp₀' (verdict_case hv)
  have hpp : p₀ = p := paramsAt_case hv.2.1 hcase hacc hp₀'
  subst hpp
  have hacc0 := hacc
  obtain ⟨-, -, -, -, -, pub, prof, hck⟩ := hacc
  obtain ⟨tags0, circuitB, pubB, record, proofs, n, ht0, hc, hpb, -, -, hn, hAc⟩ := checkBlobsAt_event hm hck
  have hs : o.sent.blobs = F ++ o.blobs := by
    show o.files ++ o.blobs = _
    rw [hFo]
  have hcf : circuitB = V.circuitFile θ F := by
    rw [hs] at hc
    exact get_named (hD.circuit p₀ hp₀) hreg.1 hc
  have hpf : pubB = V.publicFile θ F := by
    rw [hs] at hpb
    exact get_named (hD.publicFile p₀ hp₀) hreg.2 hpb
  obtain ⟨circuitB', hg, hl⟩ := tables_registered hp₀' hone' hv
  have e0 : circuitB' = circuitB := Except.ok.inj (hg.symm.trans hc)
  rw [e0] at hl
  have htab := (pinnedTables_eq hl ht hT).symm
  obtain ⟨desc, hdesc⟩ := program_of_paramsAt hp₀
  obtain ⟨out, pub', prof', hat⟩ := checkBlobsAt_checkAt hck
  obtain ⟨lo, n', read, hlo, hn', hdf⟩ := checkAt_drew hat
  have hnn : n' = n := Option.some.inj (hn'.symm.trans hn)
  rw [hnn] at hdf
  obtain ⟨tags1, circuitB1, pubB1, reg, ht1, hc1, hp1, hsc⟩ := checkBlobsAt_scope hm hck
  have e1 : tags1 = tags0 := Option.some.inj (ht1.symm.trans ht0)
  have e2 : circuitB1 = circuitB := Except.ok.inj (hc1.symm.trans hc)
  have e3 : pubB1 = pubB := Except.ok.inj (hp1.symm.trans hpb)
  rw [e1, e2, e3] at hsc
  obtain ⟨tags2, circuitB2, pubB2, rd, recS, x, ht2, hc2, hp2, hsr2, hsetup⟩ := checkBlobsAt_setup hm hck
  have e4 : tags2 = tags0 := Option.some.inj (ht2.symm.trans ht0)
  have e5 : circuitB2 = circuitB := Except.ok.inj (hc2.symm.trans hc)
  have e6 : pubB2 = pubB := Except.ok.inj (hp2.symm.trans hpb)
  rw [e4, e5, e6] at hsetup
  obtain ⟨-, -, -, -, -, hcs, -, -, -⟩ := sent_verified hv.2.1 hcase hacc0 hm
  have hrecS : sentRecord o.sent = some recS := by
    unfold sentRecord
    rw [hcs]
    exact hsr2
  obtain ⟨derived, fl, hder, hfl, hdu⟩ := sampledFrom_ok hdf
  have hdrew : Drew k o.sent dr (derived.getD fl) n ξ (sampledBlocks ev.base.draw) :=
    ⟨a, b, e, p₀, cm, ev, hcase, hacc0, ⟨lo, derived, fl, hlo, hnn ▸ hn', hder, hfl, rfl⟩, rfl, read, hdu⟩
  have hrd := sampleHeld k o.sent dr _ n ξ _ recS hv.2.1 hdrew hone' hrecS
  have hlawD := samplingDesignAt_designAt hdesc hlo hdf
  have hlawL : samplingDesignAt p₀.base.verifier n = .ok (derived.getD fl) := by
    simp only [samplingDesignAt, hlo, hder, hfl, ok_bind]
    rfl
  have hLd := Except.ok.inj (hlawL.symm.trans hlawD)
  rw [hLd] at hdu
  have hsize := Flock.Bridge.sampleUnits_size (samplingDesignOn_designAt (Flock.Bridge.popMatches_of_popOk hdesc hAc.pop)) hdu
  subst hcf
  exact ⟨p₀, tags0, pubB, record, proofs, n, desc, hp₀, ht0, hpf, hAc, htab, hdesc, hlawD, reg, hsc,
    recS, sampledBlocks ev.base.draw, rd, x, hrd, hsize, hsetup⟩

open FlockSoundness.Discharge.Recursion.Witness (toExec) in
theorem session_parts {V : Vocab d α} {T : PinnedTables} {θ : ByteArray} (hD : DigestsAt V T θ)
    {k : Verity.Registration.Composite} {F : Array ByteArray} {c : ByteArray} {o : Verity.Exchange}
    (hone : OneStatement θ) (hP : ∃ p, ParamsAt θ p) (hreach : Verity.Game.Reaches (V.game θ F c) o) (hv : V.Verdict k .sampled o)
    (ht : TablesSent T o.sent) (hT : TablesRegistered T o.sent) :
    ∃ (p : _root_.Spot.OneStageZk.Params) (n : ℕ) (I : FlockSoundness.Discharge.Exec.Inputs) (D : Flock.Desc)
      (pubO : Flock.Public) (desc : Claims.Msg), ParamsAt θ p ∧
      Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some I.tags ∧ I.count = 1 ∧ I.work = none ∧
      I.stratified = none ∧ I.closure = none ∧ I.circuitFile = V.circuitFile θ F ∧
      I.publicFile = V.publicFile θ F ∧ I.tables = pinnedTables T (V.circuitFile θ F) ∧
      I.partition = some p.own.partition ∧ I.program = some p.own.program ∧
      I.tags.typed = false ∧ FlockSoundness.Discharge.ZkHidden.TagsOkZHJ I ∧
      Flock.Desc.parse (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok D ∧
      FlockSoundness.Built D.unroll.unit ∧
      Flock.HmOut.loadPublic (FlockSoundness.Discharge.ZkExec.zkTags I) D.unroll I.publicFile = .ok pubO ∧
      FlockSoundness.Discharge.Layout.scopeOkH D.unroll = true ∧
      Flock.PortRoots.checkOf (.ok D.unroll) (FlockSoundness.Discharge.ZkExec.zkTags I) I.publicFile = .ok () ∧
      pubO.n = n ∧ _root_.Spot.Program.parse ⟨desc.toArray⟩ = .ok p.base.verifier.program ∧
      Flock.Bridge.PopMatches desc p.base.verifier.query n ∧
      samplingDesignAt p.base.verifier n = .ok (Flock.Bridge.designAt desc p.base.verifier.query p.base.verifier.samplingDesign
        p.base.verifier.workTable n) ∧
      ∃ recS units st, recordSampled (.subset units.size) n units recS ∧
        units.size = Flock.Bridge.samplingDesignK (Flock.Bridge.designAt desc p.base.verifier.query p.base.verifier.samplingDesign
          p.base.verifier.workTable n) ∧
        Flock.Stmt.setupHidden (FlockSoundness.Discharge.Composed.inputsJ I 0).tags
          (FlockSoundness.Discharge.Composed.inputsJ I 0).circuitFile
          (FlockSoundness.Discharge.Composed.inputsJ I 0).publicFile
          (FlockSoundness.Discharge.Composed.inputsJ I 0).coins
          (FlockSoundness.Discharge.Composed.inputsJ I 0).tables
          (FlockSoundness.Discharge.Exec.recordDraw recS)
          (FlockSoundness.Discharge.Composed.inputsJ I 0).partition
          (FlockSoundness.Discharge.Composed.inputsJ I 0).program none = .ok st ∧
        Flock.Zk.shapeOk st = .ok () := by
  obtain ⟨p, tags0, pubB, record, proofs, n, desc, hp, ht0, hpf, hAc, htab, hdesc, hlaw, reg, hsc,
    recS, units, rd, x, hrd, hsize, hsetup⟩ := accepted_parts hD hone hP hreach hv ht hT
  subst hpf
  obtain ⟨⟨-, htyp, hret, hleaf, -⟩, -⟩ := FlockSoundness.Discharge.FailClosed.check_ok hsc
  obtain ⟨D, pubO, hDp, hU, hpubO, hcD⟩ := desc_of_ports hAc.flat htyp hAc.ports
  have hports := hAc.ports
  rw [hcD] at hports
  refine ⟨p, n, toExec (_root_.Spot.OneStageZk.inputsOf tags0 p.own (V.circuitFile θ F) (V.publicFile θ F)
      (_root_.Core.Blobs.ofArray o.sent.blobs)), D, pubO, desc, hp, ht0, rfl, rfl, rfl, rfl, rfl, rfl, htab, rfl, rfl,
    htyp, ⟨hAc.flat.1, hret, hleaf.resolve_right fun h => absurd h.1 (by decide)⟩, hDp, hU, hpubO,
    scopeH_of_form hcD hAc.flat.2 hAc.form, hports, hAc.loads hpubO,
    hdesc, Flock.Bridge.popMatches_of_popOk hdesc hAc.pop, hlaw, recS, units, ?_⟩
  obtain ⟨st, hst, hshape⟩ := setupAt_hidden rfl
    ((FlockSoundness.Discharge.ZkExec.zkTags_hiddenOutputs _).trans hAc.flat.1) hsetup
  exact ⟨st, hrd, hsize, hst, hshape⟩

open FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkBind FlockSoundness.Discharge.ZkHidden
  FlockSoundness.Discharge.Composed in
theorem hr_planZHJP {n mPts : ℕ} {I : FlockSoundness.Discharge.Exec.Inputs} {own : Option ByteArray}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} {d₀ : Json}
    (y₀ : SampleSetupZHJP (inputsJ I 0) own mPts S₀ d₀) {Reg : Type} {k : ℕ} {ρ : ℝ}
    (hkρ : (2 : ℝ) ^ 23 ≤ ρ * (Real.exp 1 * k)) (S : Finset (Fin n)) (R : Reg)
    (j : Fin (planZHJP (Reg := Reg) dj y₀ S R).length) :
    rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ := by
  unfold rateZ
  rw [show (planZHJP (Reg := Reg) dj y₀ S R).get j = tableZHJP dj y₀ S (Fin.cast (planZHJP_length dj y₀ S R) j) from
    List.get_ofFn _ _]
  refine (div_le_div_of_nonneg_right (pow_le_pow_right₀ (by norm_num) ?_) (by positivity)).trans (cap_div_le hkρ)
  unfold tableZHJP
  split_ifs with h
  · exact zkLogLenCap_23 _ _ _ _ (Classical.choice h).hz
  · exact zkLogLenCap_23 _ _ _ _ y₀.hz

def SubsetDesign (θ : ByteArray) : Prop :=
  ∀ p, ParamsAt θ p → ∀ (desc : Claims.Msg) n,
    _root_.Spot.Program.parse ⟨desc.toArray⟩ = .ok p.base.verifier.program →
    Flock.Bridge.designAt desc p.base.verifier.query p.base.verifier.samplingDesign p.base.verifier.workTable n =
      .subset (Flock.Bridge.samplingDesignK
        (Flock.Bridge.designAt desc p.base.verifier.query p.base.verifier.samplingDesign p.base.verifier.workTable n))

theorem subsetDesign (θ : ByteArray) (hone : OneStatement θ) : SubsetDesign θ :=
  fun p hp _ n _ => Flock.Bridge.designAt_subset n (kindOf_paramsAt hp (hone p hp))

structure SessionBound.Ok (b : SessionBound) : Prop where
  k : 1 ≤ b.k
  Rw : 1 ≤ b.Rw
  M : 1 ≤ b.M
  ρ : b.ρ < 1
  kρ : (2 : ℝ) ^ 23 ≤ b.ρ * (Real.exp 1 * b.k)

open FlockSoundness.Discharge.Composed (inputsJ nTab) in
theorem liveSpec_of_accepted {V : Vocab d α} {T : PinnedTables} {b : SessionBound} (hb : b.Ok) {θ : ByteArray}
    (hD : DigestsAt V T θ)
    (hc : ∀ θ, V.programDigest θ = programDigestOf θ) (hp : ∀ θ, V.publicDigest θ = publicDigestOf θ)
    {k : Verity.Registration.Composite} {F : Array ByteArray} {c : ByteArray} {o : Verity.Exchange}
    (hread : V.read θ F = Verity.ExecutionPhaseFiles.one (readOf T θ F)) (hone : OneStatement θ)
    (hP : ∃ p, ParamsAt θ p) (hreach : Verity.Game.Reaches (V.game θ F c) o) (hv : V.Verdict k .sampled o)
    (ht : TablesSent T o.sent) (hT : TablesRegistered T o.sent) : ∃ A, LiveSpec V b θ F A := by
  obtain ⟨p, n, I, D, pubO, desc, hpa, htags, hcount, hwork, hstrat, hclos, hcirc, hpubf, htab, hpart, hprog, htyp,
    hTags, hDp, hU, hpubO, hscope, hports, hnO, hdesc, hpop, hlaw, recS, units, st, hrd, hsize, hst, hshape⟩ :=
    session_parts hD hone hP hreach hv ht hT
  have hreg : V.Registered θ F := by
    obtain ⟨hθ, hFo, -⟩ := Binding.play_reaches hreach
    have h1 := hv.1
    rw [hθ, hFo] at h1
    exact h1
  have hst2 := hst
  rw [(exec_recordSample recS).trans hrd.2] at hst2
  obtain ⟨hpiD, pub, hpub, hdup, hn⟩ := pub_of_setup hcount hDp hst2 hpubO hnO hports
  obtain ⟨S₀, d₀, hy⟩ := y₀_of_setup n hcount htyp hDp hrd hsize hst hshape
  have hn1 : nTab I = 1 := by
    unfold nTab
    simp [hcount]
  let A : Live :=
    { n, I, own := none, ht := htyp, D, hD := hDp, hU, hpi := hpiD, pub, hpub, pubO, hpubO, hscope, hdup, hn, hnO,
      desc, qr := p.base.verifier.query, hpop, lj := p.base.verifier.samplingDesign, tb := p.base.verifier.workTable,
      hsub := subsetDesign θ hone p hpa desc n hdesc, hdiv := by rw [hn1]; exact Nat.mod_one _, S₀, d₀,
      y₀ := Classical.choice hy,
      Ω := DeployedCoins, fΩ := inferInstance, nΩ := inferInstance, Lb := deployedLb, src := deployedSrc,
      hA3 := uniformRandomBytes_deployed, hTags, k := b.k, Rw := b.Rw, M := b.M, hk := hb.k, hRw := hb.Rw,
      hM := hb.M, ρ := b.ρ, hρ := hb.ρ, hr := fun S R j => hr_planZHJP _ _ hb.kρ S R j, ε := (V.certOf θ).ε,
      v := b.v, w := b.w, h := b.h }
  refine ⟨A, ⟨⟨p, hpa, hpart, hprog, hlaw, hdesc, rfl⟩, rfl, htags, hcount, hwork, hstrat, hclos, rfl, hcirc, hpubf,
    hread.trans (congrArg Verity.ExecutionPhaseFiles.one (readOf_eq V T hc hp (A := A) htags hcount hcirc hpubf htab)), rfl, rfl,
    rfl, rfl, rfl, rfl, rfl, rfl, hreg, hone⟩⟩

theorem liveOf_ne_none {V : Vocab d α} {T : PinnedTables} {b : SessionBound} (hb : b.Ok) {θ : ByteArray}
    (hD : DigestsAt V T θ)
    (hc : ∀ θ, V.programDigest θ = programDigestOf θ) (hp : ∀ θ, V.publicDigest θ = publicDigestOf θ)
    {k : Verity.Registration.Composite} {F : Array ByteArray} {c : ByteArray} {o : Verity.Exchange}
    (hread : V.read θ F = Verity.ExecutionPhaseFiles.one (readOf T θ F)) (hone : OneStatement θ)
    (hP : ∃ p, ParamsAt θ p) (hreach : Verity.Game.Reaches (V.game θ F c) o) (hv : V.Verdict k .sampled o)
    (ht : TablesSent T o.sent) (hT : TablesRegistered T o.sent) : liveOf V b θ F ≠ none := by
  unfold liveOf
  rw [dif_pos (liveSpec_of_accepted hb hD hc hp hread hone hP hreach hv ht hT)]
  exact Option.some_ne_none _

end Verity.Sampled
